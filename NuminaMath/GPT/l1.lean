import Mathlib

namespace log_equivalence_l1_1075

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence_l1_1075


namespace euler_phi_divisibility_l1_1517

def euler_phi (n : ℕ) : ℕ := sorry -- Placeholder for the Euler phi-function

theorem euler_phi_divisibility (n : ℕ) (hn : n > 0) :
    2^(n * (n + 1)) ∣ 32 * euler_phi (2^(2^n) - 1) :=
sorry

end euler_phi_divisibility_l1_1517


namespace glass_volume_l1_1526

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l1_1526


namespace average_height_students_l1_1054

/-- Given the average heights of female and male students, and the ratio of men to women, the average height -/
theorem average_height_students
  (avg_female_height : ℕ)
  (avg_male_height : ℕ)
  (ratio_men_women : ℕ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 182)
  (h3 : ratio_men_women = 5) :
  (avg_female_height + 5 * avg_male_height) / (1 + 5) = 180 :=
by
  sorry

end average_height_students_l1_1054


namespace solve_for_A_plus_B_l1_1664

theorem solve_for_A_plus_B (A B : ℤ) (h : ∀ ω, ω^2 + ω + 1 = 0 → ω^103 + A * ω + B = 0) : A + B = -1 :=
sorry

end solve_for_A_plus_B_l1_1664


namespace rate_of_interest_l1_1138

variable (P SI T R : ℝ)
variable (hP : P = 400)
variable (hSI : SI = 160)
variable (hT : T = 2)

theorem rate_of_interest :
  (SI = (P * R * T) / 100) → R = 20 :=
by
  intro h
  have h1 : P = 400 := hP
  have h2 : SI = 160 := hSI
  have h3 : T = 2 := hT
  sorry

end rate_of_interest_l1_1138


namespace solve_equation_l1_1443

theorem solve_equation (m x : ℝ) (hm_pos : m > 0) (hm_ne_one : m ≠ 1) :
  7.320 * m^(1 + Real.log x / Real.log 3) + m^(1 - Real.log x / Real.log 3) = m^2 + 1 ↔ x = 3 ∨ x = 1/3 :=
by
  sorry

end solve_equation_l1_1443


namespace solution_eq1_solution_eq2_l1_1157

-- Definitions corresponding to the conditions of the problem.
def eq1 (x : ℝ) : Prop := 16 * x^2 = 49
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 64

-- Statements for the proof problem.
theorem solution_eq1 (x : ℝ) : eq1 x → (x = 7 / 4 ∨ x = - (7 / 4)) :=
by
  intro h
  sorry

theorem solution_eq2 (x : ℝ) : eq2 x → (x = 10 ∨ x = -6) :=
by
  intro h
  sorry

end solution_eq1_solution_eq2_l1_1157


namespace inequality_proof_l1_1372

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c) ^ 2 :=
sorry

end inequality_proof_l1_1372


namespace fraction_sum_l1_1752

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l1_1752


namespace complement_problem_l1_1780

open Set

variable (U A : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_problem
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3}) :
  complement U A = {2, 4, 5} :=
by
  rw [complement, hU, hA]
  sorry

end complement_problem_l1_1780


namespace mice_needed_l1_1603

-- Definitions for relative strength in terms of M (Mouse strength)
def C (M : ℕ) : ℕ := 6 * M
def J (M : ℕ) : ℕ := 5 * C M
def G (M : ℕ) : ℕ := 4 * J M
def B (M : ℕ) : ℕ := 3 * G M
def D (M : ℕ) : ℕ := 2 * B M

-- Condition: all together can pull up the Turnip with strength 1237M
def total_strength_with_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M + M

-- Condition: without the Mouse, they cannot pull up the Turnip
def total_strength_without_mouse (M : ℕ) : ℕ :=
  D M + B M + G M + J M + C M

theorem mice_needed (M : ℕ) (h : total_strength_with_mouse M = 1237 * M) (h2 : total_strength_without_mouse M < 1237 * M) :
  1237 = 1237 :=
by
  -- using sorry to indicate proof is not provided
  sorry

end mice_needed_l1_1603


namespace remainder_prod_mod_7_l1_1580

theorem remainder_prod_mod_7 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 7 = 4 :=
by
  sorry

end remainder_prod_mod_7_l1_1580


namespace amanda_days_needed_to_meet_goal_l1_1726

def total_tickets : ℕ := 80
def first_day_friends : ℕ := 5
def first_day_per_friend : ℕ := 4
def first_day_tickets : ℕ := first_day_friends * first_day_per_friend
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_days_needed_to_meet_goal : 
  first_day_tickets + second_day_tickets + third_day_tickets = total_tickets → 
  3 = 3 :=
by
  intro h
  sorry

end amanda_days_needed_to_meet_goal_l1_1726


namespace range_of_a_l1_1094

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) → (a > 3 ∨ a < -1) :=
by
  sorry

end range_of_a_l1_1094


namespace num_correct_props_geometric_sequence_l1_1586

-- Define what it means to be a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Original Proposition P
def Prop_P (a : ℕ → ℝ) :=
  a 1 < a 2 ∧ a 2 < a 3 → ∀ n : ℕ, a n < a (n + 1)

-- Converse of Proposition P
def Conv_Prop_P (a : ℕ → ℝ) :=
  ( ∀ n : ℕ, a n < a (n + 1) ) → a 1 < a 2 ∧ a 2 < a 3

-- Inverse of Proposition P
def Inv_Prop_P (a : ℕ → ℝ) :=
  ¬(a 1 < a 2 ∧ a 2 < a 3) → ¬( ∀ n : ℕ, a n < a (n + 1) )

-- Contrapositive of Proposition P
def Contra_Prop_P (a : ℕ → ℝ) :=
  ¬( ∀ n : ℕ, a n < a (n + 1) ) → ¬(a 1 < a 2 ∧ a 2 < a 3)

-- Main theorem to be proved
theorem num_correct_props_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a → 
  Prop_P a ∧ Conv_Prop_P a ∧ Inv_Prop_P a ∧ Contra_Prop_P a := by
  sorry

end num_correct_props_geometric_sequence_l1_1586


namespace lcm_gcf_ratio_280_450_l1_1736

open Nat

theorem lcm_gcf_ratio_280_450 :
  let a := 280
  let b := 450
  lcm a b / gcd a b = 1260 :=
by
  let a := 280
  let b := 450
  sorry

end lcm_gcf_ratio_280_450_l1_1736


namespace power_mod_remainder_l1_1747

theorem power_mod_remainder :
  3 ^ 3021 % 13 = 1 :=
by
  sorry

end power_mod_remainder_l1_1747


namespace value_of_expression_l1_1359

theorem value_of_expression (x : ℝ) (h : x = 5) : (x^2 + x - 12) / (x - 4) = 18 :=
by 
  sorry

end value_of_expression_l1_1359


namespace axis_of_symmetry_shift_l1_1592

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem axis_of_symmetry_shift (f : ℝ → ℝ) (hf : is_even_function f) :
  (∃ a : ℝ, ∀ x : ℝ, f (x + 1) = f (-(x + 1))) :=
sorry

end axis_of_symmetry_shift_l1_1592


namespace projections_on_hypotenuse_l1_1432

variables {a b c p q : ℝ}
variables {ρa ρb : ℝ}

-- Given conditions
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a < b)
variable (h3 : p = a * a / c)
variable (h4 : q = b * b / c)
variable (h5 : ρa = (a * (b + c - a)) / (a + b + c))
variable (h6 : ρb = (b * (a + c - b)) / (a + b + c))

-- Proof goal
theorem projections_on_hypotenuse 
  (h_right_triangle: a^2 + b^2 = c^2) : p < ρa ∧ q > ρb :=
by
  sorry

end projections_on_hypotenuse_l1_1432


namespace smallest_natural_number_exists_l1_1983

theorem smallest_natural_number_exists (n : ℕ) : (∃ n, ∃ a b c : ℕ, n = 15 ∧ 1998 = a * (5 ^ 4) + b * (3 ^ 4) + c * (1 ^ 4) ∧ a + b + c = 15) :=
sorry

end smallest_natural_number_exists_l1_1983


namespace opposite_direction_of_vectors_l1_1280

theorem opposite_direction_of_vectors
  (x : ℝ)
  (a : ℝ × ℝ := (x, 1))
  (b : ℝ × ℝ := (4, x)) :
  (∃ k : ℝ, k ≠ 0 ∧ a = -k • b) → x = -2 := 
sorry

end opposite_direction_of_vectors_l1_1280


namespace total_non_overlapping_area_of_squares_l1_1370

theorem total_non_overlapping_area_of_squares 
  (side_length : ℕ) 
  (num_squares : ℕ)
  (overlapping_areas_count : ℕ)
  (overlapping_width : ℕ)
  (overlapping_height : ℕ)
  (total_area_with_overlap: ℕ)
  (final_missed_patch_ratio: ℕ)
  (final_adjustment: ℕ) 
  (total_area: ℕ :=  total_area_with_overlap-final_missed_patch_ratio ):
  side_length = 2 ∧ 
  num_squares = 4 ∧ 
  overlapping_areas_count = 3 ∧ 
  overlapping_width = 1 ∧ 
  overlapping_height = 2 ∧
  total_area_with_overlap = 16- 3  ∧
  final_missed_patch_ratio = 3-> 
  total_area = 13 := 
 by sorry

end total_non_overlapping_area_of_squares_l1_1370


namespace b_contribution_l1_1058

/-- A starts business with Rs. 3500.
    After 9 months, B joins as a partner.
    After a year, the profit is divided in the ratio 2:3.
    Prove that B's contribution to the capital is Rs. 21000. -/
theorem b_contribution (a_capital : ℕ) (months_a : ℕ) (b_time : ℕ) (profit_ratio_num : ℕ) (profit_ratio_den : ℕ)
  (h_a_capital : a_capital = 3500)
  (h_months_a : months_a = 12)
  (h_b_time : b_time = 3)
  (h_profit_ratio : profit_ratio_num = 2 ∧ profit_ratio_den = 3) :
  (21000 * b_time * profit_ratio_num) / (3 * profit_ratio_den) = 3500 * months_a := by
  sorry

end b_contribution_l1_1058


namespace find_y_l1_1045

-- Definitions of the angles
def angle_ABC : ℝ := 80
def angle_BAC : ℝ := 70
def angle_BCA : ℝ := 180 - angle_ABC - angle_BAC -- calculation of third angle in triangle ABC

-- Right angle in triangle CDE
def angle_ECD : ℝ := 90

-- Defining the proof problem
theorem find_y (y : ℝ) : 
  angle_BCA = 30 →
  angle_CDE = angle_BCA →
  angle_CDE + y + angle_ECD = 180 → 
  y = 60 := by
  intro h1 h2 h3
  sorry

end find_y_l1_1045


namespace find_third_divisor_l1_1838

theorem find_third_divisor 
  (h1 : ∃ (n : ℕ), n = 1014 - 3 ∧ n % 12 = 0 ∧ n % 16 = 0 ∧ n % 21 = 0 ∧ n % 28 = 0) 
  (h2 : 1011 - 3 = 1008) : 
  (∃ d, d = 3 ∧ 1008 % d = 0 ∧ 1008 % 12 = 0 ∧ 1008 % 16 = 0 ∧ 1008 % 21 = 0 ∧ 1008 % 28 = 0) :=
sorry

end find_third_divisor_l1_1838


namespace lewis_earnings_during_harvest_l1_1524

-- Define the conditions
def regular_earnings_per_week : ℕ := 28
def overtime_earnings_per_week : ℕ := 939
def number_of_weeks : ℕ := 1091

-- Define the total earnings per week
def total_earnings_per_week := regular_earnings_per_week + overtime_earnings_per_week

-- Define the total earnings during the harvest season
def total_earnings_during_harvest := total_earnings_per_week * number_of_weeks

-- Theorem statement
theorem lewis_earnings_during_harvest : total_earnings_during_harvest = 1055497 := by
  sorry

end lewis_earnings_during_harvest_l1_1524


namespace inverse_of_h_l1_1767

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end inverse_of_h_l1_1767


namespace solve_cubic_equation_l1_1053

theorem solve_cubic_equation : 
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 - y^3 = 999 ∧ (x, y) = (12, 9) ∨ (x, y) = (10, 1) := 
  by
  sorry

end solve_cubic_equation_l1_1053


namespace recreation_percentage_l1_1968

variable (W : ℝ) 

def recreation_last_week (W : ℝ) : ℝ := 0.10 * W
def wages_this_week (W : ℝ) : ℝ := 0.90 * W
def recreation_this_week (W : ℝ) : ℝ := 0.40 * (wages_this_week W)

theorem recreation_percentage : 
  (recreation_this_week W) / (recreation_last_week W) * 100 = 360 :=
by sorry

end recreation_percentage_l1_1968


namespace sum_interior_numbers_eighth_row_of_pascals_triangle_l1_1761

theorem sum_interior_numbers_eighth_row_of_pascals_triangle :
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  sum_interior_numbers = 126 :=
by
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  show sum_interior_numbers = 126
  sorry

end sum_interior_numbers_eighth_row_of_pascals_triangle_l1_1761


namespace heads_at_least_once_in_three_tosses_l1_1676

theorem heads_at_least_once_in_three_tosses :
  let total_outcomes := 8
  let all_tails_outcome := 1
  (1 - (all_tails_outcome / total_outcomes) = (7 / 8)) :=
by
  let total_outcomes := 8
  let all_tails_outcome := 1
  sorry

end heads_at_least_once_in_three_tosses_l1_1676


namespace gratuity_is_four_l1_1110

-- Define the prices and tip percentage (conditions)
def a : ℕ := 10
def b : ℕ := 13
def c : ℕ := 17
def p : ℚ := 0.1

-- Define the total bill and gratuity based on the given definitions
def total_bill : ℕ := a + b + c
def gratuity : ℚ := total_bill * p

-- Theorem (proof problem): Prove that the gratuity is $4
theorem gratuity_is_four : gratuity = 4 := by
  sorry

end gratuity_is_four_l1_1110


namespace angela_height_l1_1051

def height_of_Amy : ℕ := 150
def height_of_Helen : ℕ := height_of_Amy + 3
def height_of_Angela : ℕ := height_of_Helen + 4

theorem angela_height : height_of_Angela = 157 := by
  sorry

end angela_height_l1_1051


namespace solve_for_x_and_n_l1_1352

theorem solve_for_x_and_n (x n : ℕ) : 2^n = x^2 + 1 ↔ (x = 0 ∧ n = 0) ∨ (x = 1 ∧ n = 1) := 
sorry

end solve_for_x_and_n_l1_1352


namespace truth_values_set1_truth_values_set2_l1_1449

-- Definitions for set (1)
def p1 : Prop := Prime 3
def q1 : Prop := Even 3

-- Definitions for set (2)
def p2 (x : Int) : Prop := x = -2 ∧ (x^2 + x - 2 = 0)
def q2 (x : Int) : Prop := x = 1 ∧ (x^2 + x - 2 = 0)

-- Theorem for set (1)
theorem truth_values_set1 : 
  (p1 ∨ q1) = true ∧ (p1 ∧ q1) = false ∧ (¬p1) = false := by sorry

-- Theorem for set (2)
theorem truth_values_set2 (x : Int) :
  (p2 x ∨ q2 x) = true ∧ (p2 x ∧ q2 x) = true ∧ (¬p2 x) = false := by sorry

end truth_values_set1_truth_values_set2_l1_1449


namespace messages_tuesday_l1_1889

theorem messages_tuesday (T : ℕ) (h1 : 300 + T + (T + 300) + 2 * (T + 300) = 2000) : 
  T = 200 := by
  sorry

end messages_tuesday_l1_1889


namespace first_negative_term_position_l1_1011

def a1 : ℤ := 1031
def d : ℤ := -3
def nth_term (n : ℕ) : ℤ := a1 + (n - 1 : ℤ) * d

theorem first_negative_term_position : ∃ n : ℕ, nth_term n < 0 ∧ n = 345 := 
by 
  -- Placeholder for proof
  sorry

end first_negative_term_position_l1_1011


namespace gloria_turtle_time_l1_1385

theorem gloria_turtle_time (g_time : ℕ) (george_time : ℕ) (gloria_time : ℕ) 
  (h1 : g_time = 6) 
  (h2 : george_time = g_time - 2)
  (h3 : gloria_time = 2 * george_time) : 
  gloria_time = 8 :=
sorry

end gloria_turtle_time_l1_1385


namespace common_sum_of_matrix_l1_1456

theorem common_sum_of_matrix :
  let S := (1 / 2 : ℝ) * 25 * (10 + 34)
  let adjusted_total := S + 10
  let common_sum := adjusted_total / 6
  common_sum = 93.33 :=
by
  sorry

end common_sum_of_matrix_l1_1456


namespace correct_option_C_l1_1295

theorem correct_option_C (m n : ℤ) : 
  (4 * m + 1) * 2 * m = 8 * m^2 + 2 * m :=
by
  sorry

end correct_option_C_l1_1295


namespace field_dimension_area_l1_1227

theorem field_dimension_area (m : ℝ) : (3 * m + 8) * (m - 3) = 120 → m = 7 :=
by
  sorry

end field_dimension_area_l1_1227


namespace total_games_to_determine_winner_l1_1072

-- Conditions: Initial number of teams in the preliminary round
def initial_teams : ℕ := 24

-- Condition: Preliminary round eliminates 50% of the teams
def preliminary_round_elimination (n : ℕ) : ℕ := n / 2

-- Function to compute the required games for any single elimination tournament
def single_elimination_games (teams : ℕ) : ℕ :=
  if teams = 0 then 0
  else teams - 1

-- Proof Statement: Total number of games to determine the winner
theorem total_games_to_determine_winner (n : ℕ) (h : n = 24) :
  preliminary_round_elimination n + single_elimination_games (preliminary_round_elimination n) = 23 :=
by
  sorry

end total_games_to_determine_winner_l1_1072


namespace lines_identical_pairs_count_l1_1114

theorem lines_identical_pairs_count :
  (∃ a d : ℝ, (4 * x + a * y + d = 0 ∧ d * x - 3 * y + 15 = 0)) →
  (∃! n : ℕ, n = 2) := 
sorry

end lines_identical_pairs_count_l1_1114


namespace N_is_composite_l1_1009

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end N_is_composite_l1_1009


namespace product_of_two_numbers_l1_1737

theorem product_of_two_numbers (a b : ℤ) (h1 : Int.gcd a b = 10) (h2 : Int.lcm a b = 90) : a * b = 900 := 
sorry

end product_of_two_numbers_l1_1737


namespace fraction_value_l1_1993

theorem fraction_value (x : ℝ) (h₀ : x^2 - 3 * x - 1 = 0) (h₁ : x ≠ 0) : 
  x^2 / (x^4 + x^2 + 1) = 1 / 12 := 
by
  sorry

end fraction_value_l1_1993


namespace trip_duration_17_hours_l1_1298

theorem trip_duration_17_hours :
  ∃ T : ℝ, 
    (∀ d₁ d₂ : ℝ,
      (d₁ / 30 + 1 + (150 - d₁) / 4 = T) ∧ 
      (d₁ / 30 + d₂ / 30 + (150 - (d₁ - d₂)) / 30 = T) ∧ 
      ((d₁ - d₂) / 4 + (150 - (d₁ - d₂)) / 30 = T))
  → T = 17 :=
by
  sorry

end trip_duration_17_hours_l1_1298


namespace least_5_digit_divisible_by_12_15_18_l1_1817

theorem least_5_digit_divisible_by_12_15_18 : 
  ∃ n, n >= 10000 ∧ n < 100000 ∧ (180 ∣ n) ∧ n = 10080 :=
by
  -- Proof goes here
  sorry

end least_5_digit_divisible_by_12_15_18_l1_1817


namespace four_digit_not_multiples_of_4_or_9_l1_1406

theorem four_digit_not_multiples_of_4_or_9 (h1 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 4 ∣ n ↔ (250 ≤ n / 4 ∧ n / 4 ≤ 2499))
                                         (h2 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 9 ∣ n ↔ (112 ≤ n / 9 ∧ n / 9 ≤ 1111))
                                         (h3 : ∀ n : ℕ, n ≥ 1000 ∧ n ≤ 9999 → 36 ∣ n ↔ (28 ≤ n / 36 ∧ n / 36 ≤ 277)) :
                                         (9000 - ((2250 : ℕ) + 1000 - 250)) = 6000 :=
by sorry

end four_digit_not_multiples_of_4_or_9_l1_1406


namespace find_angle_l1_1956

theorem find_angle (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = (Real.sqrt 2) / 2) :
  θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_l1_1956


namespace rita_remaining_money_l1_1924

-- Defining the conditions
def num_dresses := 5
def price_dress := 20
def num_pants := 3
def price_pant := 12
def num_jackets := 4
def price_jacket := 30
def transport_cost := 5
def initial_amount := 400

-- Calculating the total cost
def total_cost : ℕ :=
  (num_dresses * price_dress) + 
  (num_pants * price_pant) + 
  (num_jackets * price_jacket) + 
  transport_cost

-- Stating the proof problem 
theorem rita_remaining_money : initial_amount - total_cost = 139 := by
  sorry

end rita_remaining_money_l1_1924


namespace fraction_pow_zero_l1_1785

theorem fraction_pow_zero (a b : ℤ) (hb_nonzero : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 :=
by 
  sorry

end fraction_pow_zero_l1_1785


namespace jeremy_gifted_37_goats_l1_1469

def initial_horses := 100
def initial_sheep := 29
def initial_chickens := 9

def total_initial_animals := initial_horses + initial_sheep + initial_chickens
def animals_bought_by_brian := total_initial_animals / 2
def animals_left_after_brian := total_initial_animals - animals_bought_by_brian

def total_male_animals := 53
def total_female_animals := 53
def total_remaining_animals := total_male_animals + total_female_animals

def goats_gifted_by_jeremy := total_remaining_animals - animals_left_after_brian

theorem jeremy_gifted_37_goats :
  goats_gifted_by_jeremy = 37 := 
by 
  sorry

end jeremy_gifted_37_goats_l1_1469


namespace race_position_problem_l1_1989

theorem race_position_problem 
  (Cara Bruno Emily David Fiona Alan: ℕ)
  (participants : Finset ℕ)
  (participants_card : participants.card = 12)
  (hCara_Bruno : Cara = Bruno - 3)
  (hEmily_David : Emily = David + 1)
  (hAlan_Bruno : Alan = Bruno + 4)
  (hDavid_Fiona : David = Fiona + 3)
  (hFiona_Cara : Fiona = Cara - 2)
  (hBruno : Bruno = 9)
  (Cara_in_participants : Cara ∈ participants)
  (Bruno_in_participants : Bruno ∈ participants)
  (Emily_in_participants : Emily ∈ participants)
  (David_in_participants : David ∈ participants)
  (Fiona_in_participants : Fiona ∈ participants)
  (Alan_in_participants : Alan ∈ participants)
  : David = 7 := 
sorry

end race_position_problem_l1_1989


namespace function_pass_through_point_l1_1717

theorem function_pass_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), y = a^(x-2) - 1 ∧ (x, y) = (2, 0) := 
by
  use 2
  use 0
  sorry

end function_pass_through_point_l1_1717


namespace height_of_cuboid_l1_1209

theorem height_of_cuboid (A l w : ℝ) (h : ℝ) (hA : A = 442) (hl : l = 7) (hw : w = 8) : h = 11 :=
by
  sorry

end height_of_cuboid_l1_1209


namespace max_roses_l1_1704

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 7.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  ∃ n, n = 316 :=
by
  sorry

end max_roses_l1_1704


namespace find_a4b4_l1_1224

theorem find_a4b4 
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * b1 + a2 * b3 = 1)
  (h2 : a1 * b2 + a2 * b4 = 0)
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) :
  a4 * b4 = -6 :=
sorry

end find_a4b4_l1_1224


namespace determine_dress_and_notebooks_l1_1558

structure Girl :=
  (name : String)
  (dress_color : String)
  (notebook_color : String)

def colors := ["red", "yellow", "blue"]

def Sveta : Girl := ⟨"Sveta", "red", "red"⟩
def Ira : Girl := ⟨"Ira", "blue", "yellow"⟩
def Tania : Girl := ⟨"Tania", "yellow", "blue"⟩

theorem determine_dress_and_notebooks :
  (Sveta.dress_color = Sveta.notebook_color) ∧
  (¬ Tania.dress_color = "red") ∧
  (¬ Tania.notebook_color = "red") ∧
  (Ira.notebook_color = "yellow") ∧
  (Sveta ∈ [Sveta, Ira, Tania]) ∧
  (Ira ∈ [Sveta, Ira, Tania]) ∧
  (Tania ∈ [Sveta, Ira, Tania]) →
  ([Sveta, Ira, Tania] = 
   [{name := "Sveta", dress_color := "red", notebook_color := "red"},
    {name := "Ira", dress_color := "blue", notebook_color := "yellow"},
    {name := "Tania", dress_color := "yellow", notebook_color := "blue"}])
:=
by
  intro h
  sorry

end determine_dress_and_notebooks_l1_1558


namespace find_special_N_l1_1877

theorem find_special_N : ∃ N : ℕ, 
  (Nat.digits 10 N).length = 1112 ∧
  (Nat.digits 10 N).sum % 2000 = 0 ∧
  (Nat.digits 10 (N + 1)).sum % 2000 = 0 ∧
  (Nat.digits 10 N).contains 1 ∧
  (N = 9 * 10^1111 + 1 * 10^221 + 9 * (10^220 - 1) / 9 + 10^890 - 1) :=
sorry

end find_special_N_l1_1877


namespace find_integer_n_l1_1411

theorem find_integer_n (n : ℤ) :
  (⌊ (n^2 : ℤ) / 9 ⌋ - ⌊ n / 3 ⌋^2 = 3) → (n = 8 ∨ n = 10) :=
  sorry

end find_integer_n_l1_1411


namespace initial_books_l1_1497

variable (B : ℤ)

theorem initial_books (h1 : 4 / 6 * B = B - 3300) (h2 : 3300 = 2 / 6 * B) : B = 9900 :=
by
  sorry

end initial_books_l1_1497


namespace intersection_point_with_y_axis_l1_1757

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l1_1757


namespace smaller_cube_edge_length_l1_1655

-- Given conditions
variables (s : ℝ) (volume_large_cube : ℝ) (n : ℝ)
-- n = 8 (number of smaller cubes), volume_large_cube = 1000 cm³

theorem smaller_cube_edge_length (h1 : n = 8) (h2 : volume_large_cube = 1000) :
  s^3 = volume_large_cube / n → s = 5 :=
by
  sorry

end smaller_cube_edge_length_l1_1655


namespace smaller_number_is_180_l1_1567

theorem smaller_number_is_180 (a b : ℕ) (h1 : a = 3 * b) (h2 : a + 4 * b = 420) :
  a = 180 :=
sorry

end smaller_number_is_180_l1_1567


namespace factorize_expression_l1_1543

variable (x y : ℝ)

theorem factorize_expression :
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) :=
by 
  sorry

end factorize_expression_l1_1543


namespace average_of_three_quantities_l1_1653

theorem average_of_three_quantities 
  (five_avg : ℚ) (three_avg : ℚ) (two_avg : ℚ) 
  (h_five_avg : five_avg = 10) 
  (h_two_avg : two_avg = 19) : 
  three_avg = 4 := 
by 
  let sum_5 := 5 * 10
  let sum_2 := 2 * 19
  let sum_3 := sum_5 - sum_2
  let three_avg := sum_3 / 3
  sorry

end average_of_three_quantities_l1_1653


namespace incorrect_statements_l1_1970

-- Define basic properties for lines and their equations.
def point_slope_form (y y1 x x1 k : ℝ) : Prop := (y - y1) = k * (x - x1)
def intercept_form (x y a b : ℝ) : Prop := x / a + y / b = 1
def distance_to_origin_on_y_axis (k b : ℝ) : ℝ := abs b
def slope_intercept_form (y m x c : ℝ) : Prop := y = m * x + c

-- The conditions specified in the problem.
variables (A B C D : Prop)
  (hA : A ↔ ∀ (y y1 x x1 k : ℝ), ¬point_slope_form y y1 x x1 k)
  (hB : B ↔ ∀ (x y a b : ℝ), intercept_form x y a b)
  (hC : C ↔ ∀ (k b : ℝ), distance_to_origin_on_y_axis k b = abs b)
  (hD : D ↔ ∀ (y m x c : ℝ), slope_intercept_form y m x c)

theorem incorrect_statements : ¬ B ∧ ¬ C ∧ ¬ D :=
by
  -- Intermediate steps would be to show each statement B, C, and D are false.
  sorry

end incorrect_statements_l1_1970


namespace completing_the_square_l1_1398

theorem completing_the_square (x : ℝ) : (x^2 - 2 * x - 5 = 0) → ((x - 1)^2 = 6) :=
by
  sorry

end completing_the_square_l1_1398


namespace intersection_eq_union_eq_l1_1240

noncomputable def A := {x : ℝ | -2 < x ∧ x <= 3}
noncomputable def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_eq : A ∩ B = {x : ℝ | -2 < x ∧ x < -1} := by
  sorry

theorem union_eq : A ∪ B = {x : ℝ | x <= 3 ∨ x > 4} := by
  sorry

end intersection_eq_union_eq_l1_1240


namespace total_tagged_numbers_l1_1666

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end total_tagged_numbers_l1_1666


namespace sqrt_sq_eq_abs_l1_1038

theorem sqrt_sq_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| :=
sorry

end sqrt_sq_eq_abs_l1_1038


namespace correct_computation_gives_l1_1343

variable (x : ℝ)

theorem correct_computation_gives :
  ((3 * x - 12) / 6 = 60) → ((x / 3) + 12 = 160 / 3) :=
by
  sorry

end correct_computation_gives_l1_1343


namespace verify_tin_amount_l1_1936

def ratio_to_fraction (part1 part2 : ℕ) : ℚ :=
  part2 / (part1 + part2 : ℕ)

def tin_amount_in_alloy (total_weight : ℚ) (ratio : ℚ) : ℚ :=
  total_weight * ratio

def alloy_mixture_tin_weight_is_correct
    (weight_A weight_B : ℚ)
    (ratio_A_lead ratio_A_tin : ℕ)
    (ratio_B_tin ratio_B_copper : ℕ) : Prop :=
  let tin_ratio_A := ratio_to_fraction ratio_A_lead ratio_A_tin
  let tin_ratio_B := ratio_to_fraction ratio_B_tin ratio_B_copper
  let tin_weight_A := tin_amount_in_alloy weight_A tin_ratio_A
  let tin_weight_B := tin_amount_in_alloy weight_B tin_ratio_B
  tin_weight_A + tin_weight_B = 146.57

theorem verify_tin_amount :
    alloy_mixture_tin_weight_is_correct 130 160 2 3 3 4 :=
by
  sorry

end verify_tin_amount_l1_1936


namespace work_days_l1_1337

theorem work_days (Dx Dy : ℝ) (H1 : Dy = 45) (H2 : 8 / Dx + 36 / Dy = 1) : Dx = 40 :=
by
  sorry

end work_days_l1_1337


namespace xyz_final_stock_price_l1_1540

def initial_stock_price : ℝ := 120
def first_year_increase_rate : ℝ := 0.80
def second_year_decrease_rate : ℝ := 0.30

def final_stock_price_after_two_years : ℝ :=
  (initial_stock_price * (1 + first_year_increase_rate)) * (1 - second_year_decrease_rate)

theorem xyz_final_stock_price :
  final_stock_price_after_two_years = 151.2 := by
  sorry

end xyz_final_stock_price_l1_1540


namespace lucy_crayons_correct_l1_1122

-- Define the number of crayons Willy has.
def willyCrayons : ℕ := 5092

-- Define the number of extra crayons Willy has compared to Lucy.
def extraCrayons : ℕ := 1121

-- Define the number of crayons Lucy has.
def lucyCrayons : ℕ := willyCrayons - extraCrayons

-- Statement to prove
theorem lucy_crayons_correct : lucyCrayons = 3971 := 
by
  -- The proof is omitted as per instructions
  sorry

end lucy_crayons_correct_l1_1122


namespace find_value_of_2a10_minus_a12_l1_1869

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the given conditions
def condition (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (a 4 + a 6 + a 8 + a 10 + a 12 = 120)

-- State the theorem
theorem find_value_of_2a10_minus_a12 (a : ℕ → ℝ) (h : condition a) : 2 * a 10 - a 12 = 24 :=
by sorry

end find_value_of_2a10_minus_a12_l1_1869


namespace min_additional_packs_needed_l1_1204

-- Defining the problem conditions
def total_sticker_packs : ℕ := 40
def packs_per_basket : ℕ := 7

-- The statement to prove
theorem min_additional_packs_needed : 
  ∃ (additional_packs : ℕ), 
    (total_sticker_packs + additional_packs) % packs_per_basket = 0 ∧ 
    (total_sticker_packs + additional_packs) / packs_per_basket = 6 ∧ 
    additional_packs = 2 :=
by 
  sorry

end min_additional_packs_needed_l1_1204


namespace abc_divisible_by_6_l1_1357

theorem abc_divisible_by_6 (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) :=
by
  sorry

end abc_divisible_by_6_l1_1357


namespace minimum_c_value_l1_1230

theorem minimum_c_value
  (a b c k : ℕ) (h1 : b = a + k) (h2 : c = b + k) (h3 : a < b) (h4 : b < c) (h5 : k > 0) :
  c = 6005 :=
sorry

end minimum_c_value_l1_1230


namespace algebraic_expression_value_l1_1556

theorem algebraic_expression_value (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : (a + b) ^ 2005 = -1 :=
by
  sorry

end algebraic_expression_value_l1_1556


namespace simplify_expression_l1_1367

theorem simplify_expression :
  (123 / 999) * 27 = 123 / 37 :=
by sorry

end simplify_expression_l1_1367


namespace factorial_units_digit_l1_1626

theorem factorial_units_digit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hba : a < b) : 
  ¬ (∃ k : ℕ, (b! - a!) % 10 = 7) := 
sorry

end factorial_units_digit_l1_1626


namespace min_ab_12_min_rec_expression_2_l1_1980

noncomputable def condition1 (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 / a + 3 / b = 1)

theorem min_ab_12 {a b : ℝ} (h : condition1 a b) : 
  a * b = 12 :=
sorry

theorem min_rec_expression_2 {a b : ℝ} (h : condition1 a b) :
  (1 / (a - 1)) + (3 / (b - 3)) = 2 :=
sorry

end min_ab_12_min_rec_expression_2_l1_1980


namespace intervals_of_monotonicity_minimum_value_l1_1739

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem intervals_of_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x, 0 < x ∧ x ≤ 1 / a → f a x ≤ f a (1 / a)) ∧
  (∀ x, x ≥ 1 / a → f a x ≥ f a (1 / a)) :=
sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (a < Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = -a) ∧
  (a ≥ Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = Real.log 2 - 2 * a) :=
sorry

end intervals_of_monotonicity_minimum_value_l1_1739


namespace checkerboard_corner_sum_is_164_l1_1176

def checkerboard_sum_corners : ℕ :=
  let top_left := 1
  let top_right := 9
  let bottom_left := 73
  let bottom_right := 81
  top_left + top_right + bottom_left + bottom_right

theorem checkerboard_corner_sum_is_164 :
  checkerboard_sum_corners = 164 :=
by
  sorry

end checkerboard_corner_sum_is_164_l1_1176


namespace which_is_linear_l1_1271

-- Define what it means to be a linear equation in two variables
def is_linear_equation_in_two_vars (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y = (a * x + b * y = c)

-- Define each of the given equations
def equation_A (x y : ℝ) : Prop := x / 2 + 3 * y = 2
def equation_B (x y : ℝ) : Prop := x / 2 + 1 = 3 * x * y
def equation_C (x y : ℝ) : Prop := 2 * x + 1 = 3 * x
def equation_D (x y : ℝ) : Prop := 3 * x + 2 * y^2 = 1

-- Theorem stating which equation is linear in two variables
theorem which_is_linear : 
  is_linear_equation_in_two_vars equation_A ∧ 
  ¬ is_linear_equation_in_two_vars equation_B ∧ 
  ¬ is_linear_equation_in_two_vars equation_C ∧ 
  ¬ is_linear_equation_in_two_vars equation_D := 
by 
  sorry

end which_is_linear_l1_1271


namespace curve_is_circle_l1_1904

theorem curve_is_circle (r : ℝ) (θ : ℝ) (h : r = 3) : 
  ∃ (c : ℝ) (p : ℝ × ℝ), c = 3 ∧ p = (3 * Real.cos θ, 3 * Real.sin θ) := 
sorry

end curve_is_circle_l1_1904


namespace range_of_a_l1_1916

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l1_1916


namespace pool_houses_count_l1_1169

-- Definitions based on conditions
def total_houses : ℕ := 65
def num_garage : ℕ := 50
def num_both : ℕ := 35
def num_neither : ℕ := 10
def num_pool : ℕ := total_houses - num_garage - num_neither + num_both

theorem pool_houses_count :
  num_pool = 40 := by
  -- Simplified form of the problem expressed in Lean 4 theorem statement.
  sorry

end pool_houses_count_l1_1169


namespace kangaroo_can_jump_exact_200_in_30_jumps_l1_1036

/-!
  A kangaroo can jump:
  - 3 meters using its left leg
  - 5 meters using its right leg
  - 7 meters using both legs
  - -3 meters backward
  We need to prove that the kangaroo can jump exactly 200 meters in 30 jumps.
 -/

theorem kangaroo_can_jump_exact_200_in_30_jumps :
  ∃ (n3 n5 n7 nm3 : ℕ),
    (n3 + n5 + n7 + nm3 = 30) ∧
    (3 * n3 + 5 * n5 + 7 * n7 - 3 * nm3 = 200) :=
sorry

end kangaroo_can_jump_exact_200_in_30_jumps_l1_1036


namespace total_splash_width_l1_1102

theorem total_splash_width :
  let pebble_splash := 1 / 4
  let rock_splash := 1 / 2
  let boulder_splash := 2
  let pebbles := 6
  let rocks := 3
  let boulders := 2
  let total_pebble_splash := pebbles * pebble_splash
  let total_rock_splash := rocks * rock_splash
  let total_boulder_splash := boulders * boulder_splash
  let total_splash := total_pebble_splash + total_rock_splash + total_boulder_splash
  total_splash = 7 := by
  sorry

end total_splash_width_l1_1102


namespace muirheadable_decreasing_columns_iff_l1_1507

def isMuirheadable (n : ℕ) (grid : List (List ℕ)) : Prop :=
  -- Placeholder definition; the actual definition should specify the conditions
  sorry

theorem muirheadable_decreasing_columns_iff (n : ℕ) (h : n > 0) :
  (∃ grid : List (List ℕ), isMuirheadable n grid) ↔ n ≠ 3 :=
by 
  sorry

end muirheadable_decreasing_columns_iff_l1_1507


namespace number_of_soccer_balls_in_first_set_l1_1168

noncomputable def cost_of_soccer_ball : ℕ := 50
noncomputable def first_cost_condition (F c : ℕ) : Prop := 3 * F + c = 155
noncomputable def second_cost_condition (F : ℕ) : Prop := 2 * F + 3 * cost_of_soccer_ball = 220

theorem number_of_soccer_balls_in_first_set (F : ℕ) :
  (first_cost_condition F 50) ∧ (second_cost_condition F) → 1 = 1 :=
by
  sorry

end number_of_soccer_balls_in_first_set_l1_1168


namespace min_f_of_shangmei_number_l1_1291

def is_shangmei_number (a b c d : ℕ) : Prop :=
  a + c = 11 ∧ b + d = 11

def f (a b : ℕ) : ℚ :=
  (b - (11 - b) : ℚ) / (a - (11 - a))

def G (a b : ℕ) : ℤ :=
  20 * a + 2 * b - 121

def is_multiple_of_7 (x : ℤ) : Prop :=
  ∃ k : ℤ, x = 7 * k

theorem min_f_of_shangmei_number :
  ∃ (a b c d : ℕ), a < b ∧ is_shangmei_number a b c d ∧ is_multiple_of_7 (G a b) ∧ f a b = -3 :=
sorry

end min_f_of_shangmei_number_l1_1291


namespace abs_x_minus_y_l1_1677

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l1_1677


namespace polynomial_divisible_l1_1955

theorem polynomial_divisible (p q : ℤ) (h_p : p = -26) (h_q : q = 25) :
  ∀ x : ℤ, (x^4 + p*x^2 + q) % (x^2 - 6*x + 5) = 0 :=
by
  sorry

end polynomial_divisible_l1_1955


namespace eval_expr_l1_1908
open Real

theorem eval_expr : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end eval_expr_l1_1908


namespace sweets_ratio_l1_1862

theorem sweets_ratio (number_orange_sweets : ℕ) (number_grape_sweets : ℕ) (max_sweets_per_tray : ℕ)
  (h1 : number_orange_sweets = 36) (h2 : number_grape_sweets = 44) (h3 : max_sweets_per_tray = 4) :
  (number_orange_sweets / max_sweets_per_tray) / (number_grape_sweets / max_sweets_per_tray) = 9 / 11 :=
by
  sorry

end sweets_ratio_l1_1862


namespace find_special_integer_l1_1932

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end find_special_integer_l1_1932


namespace remainder_of_sum_l1_1141

theorem remainder_of_sum (a b : ℤ) (k m : ℤ)
  (h1 : a = 84 * k + 78)
  (h2 : b = 120 * m + 114) :
  (a + b) % 42 = 24 :=
  sorry

end remainder_of_sum_l1_1141


namespace raviraj_cycle_distance_l1_1200

theorem raviraj_cycle_distance :
  ∃ (d : ℝ), d = Real.sqrt ((425: ℝ)^2 + (200: ℝ)^2) ∧ d = 470 := 
by
  sorry

end raviraj_cycle_distance_l1_1200


namespace possible_sums_of_digits_l1_1582

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits_l1_1582


namespace range_of_a_l1_1365

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
by sorry

end range_of_a_l1_1365


namespace fifteenth_term_geometric_sequence_l1_1065

theorem fifteenth_term_geometric_sequence :
  let a1 := 5
  let r := (1 : ℝ) / 2
  let fifteenth_term := a1 * r^(14 : ℕ)
  fifteenth_term = (5 : ℝ) / 16384 := by
sorry

end fifteenth_term_geometric_sequence_l1_1065


namespace students_in_sixth_level_l1_1807

theorem students_in_sixth_level (S : ℕ)
  (h1 : ∃ S₄ : ℕ, S₄ = 4 * S)
  (h2 : ∃ S₇ : ℕ, S₇ = 2 * (4 * S))
  (h3 : S + 4 * S + 2 * (4 * S) = 520) :
  S = 40 :=
by
  sorry

end students_in_sixth_level_l1_1807


namespace power_function_monotonic_decreasing_l1_1329

theorem power_function_monotonic_decreasing (α : ℝ) (h : ∀ x y : ℝ, 0 < x → x < y → x^α > y^α) : α < 0 :=
sorry

end power_function_monotonic_decreasing_l1_1329


namespace transformed_roots_l1_1368

noncomputable def specific_polynomial : Polynomial ℝ :=
  Polynomial.C 1 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 4

theorem transformed_roots (a b c d : ℝ) :
  (a^4 - b*a - 5 = 0) ∧ (b^4 - b*b - 5 = 0) ∧ (c^4 - b*c - 5 = 0) ∧ (d^4 - b*d - 5 = 0) →
  specific_polynomial.eval ((a + b + c) / d)^2 = 0 ∧
  specific_polynomial.eval ((a + b + d) / c)^2 = 0 ∧
  specific_polynomial.eval ((a + c + d) / b)^2 = 0 ∧
  specific_polynomial.eval ((b + c + d) / a)^2 = 0 :=
  by
    sorry

end transformed_roots_l1_1368


namespace pizza_varieties_l1_1799

-- Definition of the problem conditions
def base_flavors : ℕ := 4
def topping_options : ℕ := 4  -- No toppings, extra cheese, mushrooms, both

-- The math proof problem statement
theorem pizza_varieties : base_flavors * topping_options = 16 := by 
  sorry

end pizza_varieties_l1_1799


namespace find_point_on_line_and_distance_l1_1531

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem find_point_on_line_and_distance :
  ∃ P : ℝ × ℝ, (2 * P.1 - 3 * P.2 + 5 = 0) ∧ (distance P (2, 3) = 13) →
  (P = (5, 5) ∨ P = (-1, 1)) :=
by
  sorry

end find_point_on_line_and_distance_l1_1531


namespace alice_has_ball_after_two_turns_l1_1111

def prob_alice_keeps_ball : ℚ := (2/3 * 1/2) + (1/3 * 1/3)

theorem alice_has_ball_after_two_turns :
  prob_alice_keeps_ball = 4 / 9 :=
by
  -- This line is just a placeholder for the actual proof
  sorry

end alice_has_ball_after_two_turns_l1_1111


namespace num_possible_values_of_M_l1_1471

theorem num_possible_values_of_M :
  ∃ n : ℕ, n = 8 ∧
  ∃ (a b : ℕ), (10 <= 10*a + b) ∧ (10*a + b < 100) ∧ (9*(a - b) ∈ {k : ℕ | ∃ m : ℕ, k = m^2}) := sorry

end num_possible_values_of_M_l1_1471


namespace sum_of_digits_of_63_l1_1315

theorem sum_of_digits_of_63 (x y : ℕ) (h : 10 * x + y = 63) (h1 : x + y = 9) (h2 : x - y = 3) : x + y = 9 :=
by
  sorry

end sum_of_digits_of_63_l1_1315


namespace problem1_problem2_l1_1523

-- problem (1): Prove that if a = 1 and (p ∨ q) is true, then the range of x is 1 < x < 3
def p (a x : ℝ) : Prop := x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : a = 1) (h₂ : p a x ∨ q x) : 
    1 < x ∧ x < 3 :=
sorry

-- problem (2): Prove that if p is a necessary but not sufficient condition for q,
-- then the range of a is 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end problem1_problem2_l1_1523


namespace interval_of_increase_l1_1016

noncomputable def f (x : ℝ) : ℝ :=
  -abs x

theorem interval_of_increase :
  ∀ x, f x ≤ f (x + 1) ↔ x ≤ 0 := by
  sorry

end interval_of_increase_l1_1016


namespace rod_length_l1_1448

theorem rod_length (num_pieces : ℝ) (length_per_piece : ℝ) (h1 : num_pieces = 118.75) (h2 : length_per_piece = 0.40) : 
  num_pieces * length_per_piece = 47.5 := by
  sorry

end rod_length_l1_1448


namespace exists_n_ge_1_le_2020_l1_1718

theorem exists_n_ge_1_le_2020
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j : ℕ, 1 ≤ i → i ≤ 2020 → 1 ≤ j → j ≤ 2020 → i ≠ j → a i ≠ a j)
  (h_periodic1 : a 2021 = a 1)
  (h_periodic2 : a 2022 = a 2) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2020 ∧ a n ^ 2 + a (n + 1) ^ 2 ≥ a (n + 2) ^ 2 + n ^ 2 + 3 := 
sorry

end exists_n_ge_1_le_2020_l1_1718


namespace average_rst_l1_1331

theorem average_rst (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 :=
sorry

end average_rst_l1_1331


namespace paintable_wall_area_l1_1745

theorem paintable_wall_area :
  let bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let doorway_window_area := 70
  let area_one_bedroom := 
    2 * (length * height) + 2 * (width * height) - doorway_window_area
  let total_paintable_area := bedrooms * area_one_bedroom
  total_paintable_area = 1520 := by
  sorry

end paintable_wall_area_l1_1745


namespace fraction_of_bones_in_foot_is_approx_one_eighth_l1_1242

def number_bones_human_body : ℕ := 206
def number_bones_one_foot : ℕ := 26
def fraction_bones_one_foot (total_bones foot_bones : ℕ) : ℚ := foot_bones / total_bones

theorem fraction_of_bones_in_foot_is_approx_one_eighth :
  fraction_bones_one_foot number_bones_human_body number_bones_one_foot = 13 / 103 ∧ 
  (abs ((13 / 103 : ℚ) - (1 / 8)) < 1 / 103) := 
sorry

end fraction_of_bones_in_foot_is_approx_one_eighth_l1_1242


namespace marikas_father_twice_her_age_l1_1458

theorem marikas_father_twice_her_age (birth_year : ℤ) (marika_age : ℤ) (father_multiple : ℕ) :
  birth_year = 2006 ∧ marika_age = 10 ∧ father_multiple = 5 →
  ∃ x : ℤ, birth_year + x = 2036 ∧ (father_multiple * marika_age + x) = 2 * (marika_age + x) :=
by {
  sorry
}

end marikas_father_twice_her_age_l1_1458


namespace triangle_sin_a_triangle_area_l1_1401

theorem triangle_sin_a (B : ℝ) (a b c : ℝ) (hB : B = π / 4)
  (h_bc : b = Real.sqrt 5 ∧ c = Real.sqrt 2 ∨ a = 3 ∧ c = Real.sqrt 2) :
  Real.sin A = (3 * Real.sqrt 10) / 10 :=
sorry

theorem triangle_area (B a b c : ℝ) (hB : B = π / 4) (hb : b = Real.sqrt 5)
  (h_ac : a + c = 3) : 1 / 2 * a * c * Real.sin B = Real.sqrt 2 - 1 :=
sorry

end triangle_sin_a_triangle_area_l1_1401


namespace mike_took_23_green_marbles_l1_1614

-- Definition of the conditions
def original_green_marbles : ℕ := 32
def remaining_green_marbles : ℕ := 9

-- Definition of the statement we want to prove
theorem mike_took_23_green_marbles : original_green_marbles - remaining_green_marbles = 23 := by
  sorry

end mike_took_23_green_marbles_l1_1614


namespace cone_heights_l1_1985

theorem cone_heights (H x r1 r2 : ℝ) (H_frustum : H - x = 18)
  (A_lower : 400 * Real.pi = Real.pi * r1^2)
  (A_upper : 100 * Real.pi = Real.pi * r2^2)
  (ratio_radii : r2 / r1 = 1 / 2)
  (ratio_heights : x / H = 1 / 2) :
  x = 18 ∧ H = 36 :=
by
  sorry

end cone_heights_l1_1985


namespace arith_seq_ratio_l1_1683

theorem arith_seq_ratio {a b : ℕ → ℕ} {S T : ℕ → ℕ}
  (h₁ : ∀ n, S n = (n * (2 * a n - a 1)) / 2)
  (h₂ : ∀ n, T n = (n * (2 * b n - b 1)) / 2)
  (h₃ : ∀ n, S n / T n = (5 * n + 3) / (2 * n + 7)) :
  (a 9 / b 9 = 88 / 41) :=
sorry

end arith_seq_ratio_l1_1683


namespace greatest_multiple_of_30_less_than_800_l1_1503

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l1_1503


namespace solve_inequality_l1_1234

theorem solve_inequality (x : ℝ) : (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 4 ↔ x ∈ Set.Ico (-1/4) 0 ∪ Set.Ioc 2 3 := 
sorry

end solve_inequality_l1_1234


namespace profit_difference_l1_1416

-- Define the initial investments
def investment_A : ℚ := 8000
def investment_B : ℚ := 10000
def investment_C : ℚ := 12000

-- Define B's profit share
def profit_B : ℚ := 1700

-- Prove that the difference between A and C's profit shares is Rs. 680
theorem profit_difference (investment_A investment_B investment_C profit_B: ℚ) (hA : investment_A = 8000) (hB : investment_B = 10000) (hC : investment_C = 12000) (pB : profit_B = 1700) :
    let ratio_A : ℚ := 4
    let ratio_B : ℚ := 5
    let ratio_C : ℚ := 6
    let part_value : ℚ := profit_B / ratio_B
    let profit_A : ℚ := ratio_A * part_value
    let profit_C : ℚ := ratio_C * part_value
    profit_C - profit_A = 680 := 
by
  sorry

end profit_difference_l1_1416


namespace frame_painting_ratio_l1_1513

theorem frame_painting_ratio :
  ∃ (x : ℝ), (20 + 2 * x) * (30 + 6 * x) = 1800 → 1 = 2 * (20 + 2 * x) / (30 + 6 * x) :=
by
  sorry

end frame_painting_ratio_l1_1513


namespace PolyCoeffInequality_l1_1912

open Real

variable (p q : ℝ[X])
variable (a : ℝ)
variable (n : ℕ)
variable (h k : ℝ)
variable (deg_p : p.degree = n)
variable (deg_q : q.degree = n - 1)
variable (hp : ∀ i, i ≤ n → |p.coeff i| ≤ h)
variable (hq : ∀ i, i < n → |q.coeff i| ≤ k)
variable (hpq : p = (X + C a) * q)

theorem PolyCoeffInequality : k ≤ h^n := by
  sorry

end PolyCoeffInequality_l1_1912


namespace cube_inverse_sum_l1_1598

theorem cube_inverse_sum (x : ℂ) (h : x + 1/x = -3) : x^3 + (1/x)^3 = -18 :=
by
  sorry

end cube_inverse_sum_l1_1598


namespace range_of_values_for_k_l1_1148

theorem range_of_values_for_k (k : ℝ) (h : k ≠ 0) :
  (1 : ℝ) ∈ { x : ℝ | k^2 * x^2 - 6 * k * x + 8 ≥ 0 } ↔ (k ≥ 4 ∨ k ≤ 2) := 
by
  -- proof 
  sorry

end range_of_values_for_k_l1_1148


namespace exactly_two_talents_l1_1629

open Nat

def total_students : Nat := 50
def cannot_sing_students : Nat := 20
def cannot_dance_students : Nat := 35
def cannot_act_students : Nat := 15

theorem exactly_two_talents : 
  (total_students - cannot_sing_students) + 
  (total_students - cannot_dance_students) + 
  (total_students - cannot_act_students) - total_students = 30 := by
  sorry

end exactly_two_talents_l1_1629


namespace solve_for_A_l1_1693

def f (A B x : ℝ) : ℝ := A * x ^ 2 - 3 * B ^ 3
def g (B x : ℝ) : ℝ := 2 * B * x + B ^ 2

theorem solve_for_A (B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) :
  A = 3 / (16 / B + 8 + B ^ 3) :=
by
  sorry

end solve_for_A_l1_1693


namespace findValuesForFibSequence_l1_1930

noncomputable def maxConsecutiveFibonacciTerms (A B C : ℝ) : ℝ :=
  if A ≠ 0 then 4 else 0

theorem findValuesForFibSequence :
  maxConsecutiveFibonacciTerms (1/2) (-1/2) 2 = 4 ∧ maxConsecutiveFibonacciTerms (1/2) (1/2) 2 = 4 :=
by
  -- This statement will follow from the given conditions and the solution provided.
  sorry

end findValuesForFibSequence_l1_1930


namespace sum_smallest_numbers_eq_six_l1_1982

theorem sum_smallest_numbers_eq_six :
  let smallest_natural := 0
  let smallest_prime := 2
  let smallest_composite := 4
  smallest_natural + smallest_prime + smallest_composite = 6 := by
  sorry

end sum_smallest_numbers_eq_six_l1_1982


namespace blue_paint_needed_l1_1442

/-- 
If the ratio of blue paint to green paint is \(4:1\), and Sarah wants to make 40 cans of the mixture,
prove that the number of cans of blue paint needed is 32.
-/
theorem blue_paint_needed (r: ℕ) (total_cans: ℕ) (h_ratio: r = 4) (h_total: total_cans = 40) : 
  ∃ b: ℕ, b = 4 / 5 * total_cans ∧ b = 32 :=
by
  sorry

end blue_paint_needed_l1_1442


namespace tan_five_pi_over_four_eq_one_l1_1851

theorem tan_five_pi_over_four_eq_one : Real.tan (5 * Real.pi / 4) = 1 :=
by sorry

end tan_five_pi_over_four_eq_one_l1_1851


namespace BoatCrafters_total_canoes_l1_1156

def canoe_production (n : ℕ) : ℕ :=
  if n = 0 then 5 else 3 * canoe_production (n-1) - 1

theorem BoatCrafters_total_canoes : 
  (canoe_production 0 - 1) + (canoe_production 1 - 1) + (canoe_production 2 - 1) + (canoe_production 3 - 1) = 196 := 
by
  sorry

end BoatCrafters_total_canoes_l1_1156


namespace fractional_part_inequality_l1_1819

noncomputable def frac (z : ℝ) : ℝ := z - ⌊z⌋

theorem fractional_part_inequality (x y : ℝ) : frac (x + y) ≤ frac x + frac y := 
sorry

end fractional_part_inequality_l1_1819


namespace total_apples_l1_1943

theorem total_apples (baskets apples_per_basket : ℕ) (h1 : baskets = 37) (h2 : apples_per_basket = 17) : baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l1_1943


namespace Lizzie_has_27_crayons_l1_1374

variable (Lizzie Bobbie Billie : ℕ)

axiom Billie_crayons : Billie = 18
axiom Bobbie_crayons : Bobbie = 3 * Billie
axiom Lizzie_crayons : Lizzie = Bobbie / 2

theorem Lizzie_has_27_crayons : Lizzie = 27 :=
by
  sorry

end Lizzie_has_27_crayons_l1_1374


namespace side_lengths_sum_eq_225_l1_1656

noncomputable def GX (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - x

noncomputable def GY (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - y

noncomputable def GZ (x y z : ℝ) : ℝ :=
  (1 / 3) * (x + y + z) - z

theorem side_lengths_sum_eq_225
  (x y z : ℝ)
  (h : GX x y z ^ 2 + GY x y z ^ 2 + GZ x y z ^ 2 = 75) :
  (x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2 = 225 := by {
  sorry
}

end side_lengths_sum_eq_225_l1_1656


namespace arithmetic_sequence_sum_l1_1906

theorem arithmetic_sequence_sum :
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  Sₙ = 1830 :=
by
  intros
  let a₁ := 3
  let d := 4
  let n := 30
  let aₙ := a₁ + (n - 1) * d
  let Sₙ := (n / 2) * (a₁ + aₙ)
  sorry

end arithmetic_sequence_sum_l1_1906


namespace xy_sum_l1_1083

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y + 20) : x + y = 12 + 2 * Real.sqrt 6 ∨ x + y = 12 - 2 * Real.sqrt 6 :=
by
  sorry

end xy_sum_l1_1083


namespace expression_value_l1_1445

theorem expression_value {a b c d m : ℝ} (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := 
by
  sorry

end expression_value_l1_1445


namespace min_value_of_quadratic_form_l1_1611

theorem min_value_of_quadratic_form (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + 2 * y^2 + 3 * z^2 ≥ 1/3 :=
sorry

end min_value_of_quadratic_form_l1_1611


namespace machine_part_masses_l1_1538

theorem machine_part_masses :
  ∃ (x y : ℝ), (y - 2 * x = 100) ∧ (875 / x - 900 / y = 3) ∧ (x = 175) ∧ (y = 450) :=
by {
  sorry
}

end machine_part_masses_l1_1538


namespace part1_part2_l1_1698

-- Definitions based on the conditions
def original_sales : ℕ := 30
def profit_per_shirt_initial : ℕ := 40

-- Additional shirts sold for each 1 yuan price reduction
def additional_shirts_per_yuan : ℕ := 2

-- Price reduction example of 3 yuan
def price_reduction_example : ℕ := 3

-- New sales quantity after 3 yuan reduction
def new_sales_quantity_example := 
  original_sales + (price_reduction_example * additional_shirts_per_yuan)

-- Prove that the sales quantity is 36 shirts for a reduction of 3 yuan
theorem part1 : new_sales_quantity_example = 36 := by
  sorry

-- General price reduction variable
def price_reduction_per_item (x : ℕ) : ℕ := x
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt_initial - x
def new_sales_quantity (x : ℕ) : ℕ := original_sales + (additional_shirts_per_yuan * x)
def daily_sales_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (new_sales_quantity x)

-- Goal for daily sales profit of 1200 yuan
def goal_profit : ℕ := 1200

-- Prove that a price reduction of 25 yuan per shirt achieves a daily sales profit of 1200 yuan
theorem part2 : daily_sales_profit 25 = goal_profit := by
  sorry

end part1_part2_l1_1698


namespace largest_expression_is_D_l1_1485

-- Define each expression
def exprA : ℤ := 3 - 1 + 4 + 6
def exprB : ℤ := 3 - 1 * 4 + 6
def exprC : ℤ := 3 - (1 + 4) * 6
def exprD : ℤ := 3 - 1 + 4 * 6
def exprE : ℤ := 3 * (1 - 4) + 6

-- The theorem stating that exprD is the largest value among the given expressions.
theorem largest_expression_is_D : 
  exprD = 26 ∧ 
  exprD > exprA ∧ 
  exprD > exprB ∧ 
  exprD > exprC ∧ 
  exprD > exprE := 
by {
  sorry
}

end largest_expression_is_D_l1_1485


namespace ellipse_foci_distance_l1_1095

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∀ (a b : ℝ), a = 6 → b = 3 → distance_between_foci a b = 3 * Real.sqrt 3 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  simp [distance_between_foci]
  sorry

end ellipse_foci_distance_l1_1095


namespace game_winning_starting_numbers_count_l1_1123

theorem game_winning_starting_numbers_count : 
  ∃ win_count : ℕ, (win_count = 6) ∧ 
                  ∀ n : ℕ, (1 ≤ n ∧ n < 10) → 
                  (n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) ↔ 
                  ((∃ m, (2 * n ≤ m ∧ m ≤ 3 * n) ∧ m < 2007)  → 
                   (∃ k, (2 * m ≤ k ∧ k ≤ 3 * m) ∧ k ≥ 2007) = false) := 
sorry

end game_winning_starting_numbers_count_l1_1123


namespace complex_fifth_roots_wrong_statement_l1_1354

noncomputable def x : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)
noncomputable def y : ℂ := Complex.exp (-2 * Real.pi * Complex.I / 5)

theorem complex_fifth_roots_wrong_statement :
  ¬(x^5 + y^5 = 1) :=
sorry

end complex_fifth_roots_wrong_statement_l1_1354


namespace segments_after_cuts_l1_1151

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end segments_after_cuts_l1_1151


namespace indoor_tables_count_l1_1262

theorem indoor_tables_count
  (I : ℕ)  -- the number of indoor tables
  (O : ℕ)  -- the number of outdoor tables
  (H1 : O = 12)  -- Condition 1: O = 12
  (H2 : 3 * I + 3 * O = 60)  -- Condition 2: Total number of chairs
  : I = 8 :=
by
  -- Insert the actual proof here
  sorry

end indoor_tables_count_l1_1262


namespace range_of_a_l1_1935

theorem range_of_a (a : ℝ) : 
  ( ∃ x y : ℝ, (x^2 + 4 * (y - a)^2 = 4) ∧ (x^2 = 4 * y)) ↔ a ∈ Set.Ico (-1 : ℝ) (5 / 4 : ℝ) := 
sorry

end range_of_a_l1_1935


namespace common_ratio_of_geometric_seq_l1_1057

variable {α : Type*} [Field α]

-- Definition of the geometric sequence
def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ n

-- Sum of the first three terms of the geometric sequence
def sum_first_three_terms (a q: α) : α :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

theorem common_ratio_of_geometric_seq (a q : α) (h : sum_first_three_terms a q = 3 * a) : q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_seq_l1_1057


namespace train_speed_on_time_l1_1688

theorem train_speed_on_time (v : ℕ) (t : ℕ) :
  (15 / v + 1 / 4 = 15 / 50) ∧ (t = 15) → v = 300 := by
  sorry

end train_speed_on_time_l1_1688


namespace mark_initial_fries_l1_1373

variable (Sally_fries_before : ℕ)
variable (Sally_fries_after : ℕ)
variable (Mark_fries_given : ℕ)
variable (Mark_fries_initial : ℕ)

theorem mark_initial_fries (h1 : Sally_fries_before = 14) (h2 : Sally_fries_after = 26) (h3 : Mark_fries_given = Sally_fries_after - Sally_fries_before) (h4 : Mark_fries_given = 1/3 * Mark_fries_initial) : Mark_fries_initial = 36 :=
by sorry

end mark_initial_fries_l1_1373


namespace exists_zero_in_interval_l1_1166

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem exists_zero_in_interval : ∃ c ∈ Set.Ioo 0 (1/2 : ℝ), f c = 0 := by
  -- proof to be filled in
  sorry

end exists_zero_in_interval_l1_1166


namespace bc_fraction_of_ad_l1_1663

theorem bc_fraction_of_ad
  {A B D C : Type}
  (length_AB length_BD length_AC length_CD length_AD length_BC : ℝ)
  (h1 : length_AB = 3 * length_BD)
  (h2 : length_AC = 4 * length_CD)
  (h3 : length_AD = length_AB + length_BD + length_CD)
  (h4 : length_BC = length_AC - length_AB) :
  length_BC / length_AD = 5 / 6 :=
by sorry

end bc_fraction_of_ad_l1_1663


namespace smallest_special_number_l1_1347

def is_special (n : ℕ) : Prop :=
  (∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
   n = a * 1000 + b * 100 + c * 10 + d)

theorem smallest_special_number (n : ℕ) (h1 : n > 3429) (h2 : is_special n) : n = 3450 :=
sorry

end smallest_special_number_l1_1347


namespace Marcy_sips_interval_l1_1340

theorem Marcy_sips_interval:
  ∀ (total_volume_ml sip_volume_ml total_time min_per_sip: ℕ),
  total_volume_ml = 2000 →
  sip_volume_ml = 40 →
  total_time = 250 →
  min_per_sip = total_time / (total_volume_ml / sip_volume_ml) →
  min_per_sip = 5 :=
by
  intros total_volume_ml sip_volume_ml total_time min_per_sip hv hs ht hm
  rw [hv, hs, ht] at hm
  simp at hm
  exact hm

end Marcy_sips_interval_l1_1340


namespace number_of_people_l1_1568

theorem number_of_people (x y z : ℕ) 
  (h1 : x + y + z = 12) 
  (h2 : 2 * x + y / 2 + z / 4 = 12) : 
  x = 5 ∧ y = 1 ∧ z = 6 := 
by
  sorry

end number_of_people_l1_1568


namespace probability_interval_l1_1539

-- Define the probability distribution and conditions
def P (xi : ℕ) (c : ℚ) : ℚ := c / (xi * (xi + 1))

-- Given conditions
variables (c : ℚ)
axiom condition : P 1 c + P 2 c + P 3 c + P 4 c = 1

-- Define the interval probability
def interval_prob (c : ℚ) : ℚ := P 1 c + P 2 c

-- Prove that the computed probability matches the expected value
theorem probability_interval : interval_prob (5 / 4) = 5 / 6 :=
by
  -- skip proof
  sorry

end probability_interval_l1_1539


namespace kayak_rental_cost_l1_1708

theorem kayak_rental_cost
    (canoe_cost_per_day : ℕ := 14)
    (total_revenue : ℕ := 288)
    (canoe_kayak_ratio : ℕ × ℕ := (3, 2))
    (canoe_kayak_difference : ℕ := 4)
    (number_of_kayaks : ℕ := 8)
    (number_of_canoes : ℕ := number_of_kayaks + canoe_kayak_difference)
    (canoe_revenue : ℕ := number_of_canoes * canoe_cost_per_day) :
    number_of_kayaks * kayak_cost_per_day = total_revenue - canoe_revenue →
    kayak_cost_per_day = 15 := 
by
  sorry

end kayak_rental_cost_l1_1708


namespace correct_operation_l1_1212

variable (a b m : ℕ)

theorem correct_operation :
  (3 * a^2 * 2 * a^2 ≠ 5 * a^2) ∧
  ((2 * a^2)^3 = 8 * a^6) ∧
  (m^6 / m^3 ≠ m^2) ∧
  ((a + b)^2 ≠ a^2 + b^2) →
  ((2 * a^2)^3 = 8 * a^6) :=
by
  intros
  sorry

end correct_operation_l1_1212


namespace necessary_but_not_sufficient_condition_for_inequality_l1_1047

theorem necessary_but_not_sufficient_condition_for_inequality 
    {a b c : ℝ} (h : a * c^2 ≥ b * c^2) : ¬(a > b → (a * c^2 < b * c^2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_inequality_l1_1047


namespace spends_at_arcade_each_weekend_l1_1195

def vanessa_savings : ℕ := 20
def parents_weekly_allowance : ℕ := 30
def dress_cost : ℕ := 80
def weeks : ℕ := 3

theorem spends_at_arcade_each_weekend (arcade_weekend_expense : ℕ) :
  (vanessa_savings + weeks * parents_weekly_allowance - dress_cost = weeks * parents_weekly_allowance - arcade_weekend_expense * weeks) →
  arcade_weekend_expense = 30 :=
by
  intro h
  sorry

end spends_at_arcade_each_weekend_l1_1195


namespace game_winning_strategy_l1_1942

theorem game_winning_strategy (n : ℕ) (h : n ≥ 3) :
  (∃ k : ℕ, n = 3 * k + 2) → (∃ k : ℕ, n = 3 * k + 2 ∨ ∀ k : ℕ, n ≠ 3 * k + 2) :=
by
  sorry

end game_winning_strategy_l1_1942


namespace square_of_105_l1_1805

theorem square_of_105 : 105^2 = 11025 := by
  sorry

end square_of_105_l1_1805


namespace sum_four_digit_integers_ending_in_zero_l1_1386

def arithmetic_series_sum (a l d : ℕ) : ℕ := 
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_four_digit_integers_ending_in_zero : 
  arithmetic_series_sum 1000 9990 10 = 4945500 :=
by
  sorry

end sum_four_digit_integers_ending_in_zero_l1_1386


namespace find_number_l1_1905

theorem find_number (N : ℝ) 
    (h : 0.20 * ((0.05)^3 * 0.35 * (0.70 * N)) = 182.7) : 
    N = 20880000 :=
by
  -- proof to be filled
  sorry

end find_number_l1_1905


namespace cooks_number_l1_1740

variable (C W : ℕ)

theorem cooks_number (h1 : 10 * C = 3 * W) (h2 : 14 * C = 3 * (W + 12)) : C = 9 :=
by
  sorry

end cooks_number_l1_1740


namespace collinear_magnitude_a_perpendicular_magnitude_b_l1_1113

noncomputable section

open Real

-- Defining the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Defining the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Given conditions and respective proofs
theorem collinear_magnitude_a (x : ℝ) (h : 1 * 3 = x ^ 2) : magnitude (a x) = 2 :=
by sorry

theorem perpendicular_magnitude_b (x : ℝ) (h : 1 * x + x * 3 = 0) : magnitude (b x) = 3 :=
by sorry

end collinear_magnitude_a_perpendicular_magnitude_b_l1_1113


namespace simplify_expression_l1_1898

variable (a b : ℕ)

theorem simplify_expression (a b : ℕ) : 5 * a * b - 7 * a * b + 3 * a * b = a * b := by
  sorry

end simplify_expression_l1_1898


namespace maximum_fraction_l1_1855

theorem maximum_fraction (A B : ℕ) (h1 : A ≠ B) (h2 : 0 < A ∧ A < 1000) (h3 : 0 < B ∧ B < 1000) :
  ∃ (A B : ℕ), (A = 500) ∧ (B = 499) ∧ (A ≠ B) ∧ (0 < A ∧ A < 1000) ∧ (0 < B ∧ B < 1000) ∧ (A - B = 1) ∧ (A + B = 999) ∧ (499 / 500 = 0.998) := sorry

end maximum_fraction_l1_1855


namespace painting_time_l1_1050

-- Definitions translated from conditions
def total_weight_tons := 5
def weight_per_ball_kg := 4
def number_of_students := 10
def balls_per_student_per_6_minutes := 5

-- Derived Definitions
def total_weight_kg := total_weight_tons * 1000
def total_balls := total_weight_kg / weight_per_ball_kg
def balls_painted_by_all_students_per_6_minutes := number_of_students * balls_per_student_per_6_minutes
def required_intervals := total_balls / balls_painted_by_all_students_per_6_minutes
def total_time_minutes := required_intervals * 6

-- The theorem statement
theorem painting_time : total_time_minutes = 150 := by
  sorry

end painting_time_l1_1050


namespace g_h_of_2_eq_869_l1_1438

-- Define the functions g and h
def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -2 * x^3 - 1

-- State the theorem we need to prove
theorem g_h_of_2_eq_869 : g (h 2) = 869 := by
  sorry

end g_h_of_2_eq_869_l1_1438


namespace largest_value_after_2001_presses_l1_1144

noncomputable def max_value_after_presses (n : ℕ) : ℝ :=
if n = 0 then 1 else sorry -- Placeholder for the actual function definition

theorem largest_value_after_2001_presses :
  max_value_after_presses 2001 = 1 :=
sorry

end largest_value_after_2001_presses_l1_1144


namespace rectangle_width_length_ratio_l1_1342

theorem rectangle_width_length_ratio (w : ℕ) (h : ℕ) (P : ℕ) (H1 : h = 10) (H2 : P = 30) (H3 : 2 * w + 2 * h = P) :
  w / h = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l1_1342


namespace sum_not_zero_l1_1060

theorem sum_not_zero (a b c d : ℝ) (h1 : a * b * c - d = 1) (h2 : b * c * d - a = 2) 
  (h3 : c * d * a - b = 3) (h4 : d * a * b - c = -6) : a + b + c + d ≠ 0 :=
sorry

end sum_not_zero_l1_1060


namespace tan_add_pi_over_4_l1_1055

variable {α : ℝ}

theorem tan_add_pi_over_4 (h : Real.tan (α - Real.pi / 4) = 1 / 4) : Real.tan (α + Real.pi / 4) = -4 :=
sorry

end tan_add_pi_over_4_l1_1055


namespace w_identity_l1_1213

theorem w_identity (w : ℝ) (h_pos : w > 0) (h_eq : w - 1 / w = 5) : (w + 1 / w) ^ 2 = 29 := by
  sorry

end w_identity_l1_1213


namespace income_calculation_l1_1951

theorem income_calculation (x : ℕ) (h1 : ∃ x : ℕ, income = 8*x ∧ expenditure = 7*x)
  (h2 : savings = 5000)
  (h3 : income = expenditure + savings) : income = 40000 :=
by {
  sorry
}

end income_calculation_l1_1951


namespace shuttle_speed_l1_1149

theorem shuttle_speed (v : ℕ) (h : v = 9) : v * 3600 = 32400 :=
by
  sorry

end shuttle_speed_l1_1149


namespace math_proof_l1_1573

def problem_statement : Prop :=
  ∃ x : ℕ, (2 * x + 3 = 19) ∧ (x + (2 * x + 3) = 27)

theorem math_proof : problem_statement :=
  sorry

end math_proof_l1_1573


namespace john_needs_more_usd_l1_1021

noncomputable def additional_usd (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ) : ℝ :=
  let eur_to_usd := 1 / 0.84
  let sgd_to_usd := 1 / 1.34
  let jpy_to_usd := 1 / 110.35
  let total_needed_usd := needed_eur * eur_to_usd + needed_sgd * sgd_to_usd
  let total_has_usd := has_usd + has_jpy * jpy_to_usd
  total_needed_usd - total_has_usd

theorem john_needs_more_usd :
  ∀ (needed_eur needed_sgd : ℝ) (has_usd has_jpy : ℝ),
    needed_eur = 7.50 → needed_sgd = 5.00 → has_usd = 2.00 → has_jpy = 500 →
    additional_usd needed_eur needed_sgd has_usd has_jpy = 6.13 :=
by
  intros needed_eur needed_sgd has_usd has_jpy
  intros hneeded_eur hneeded_sgd hhas_usd hhas_jpy
  unfold additional_usd
  rw [hneeded_eur, hneeded_sgd, hhas_usd, hhas_jpy]
  sorry

end john_needs_more_usd_l1_1021


namespace geographic_info_tech_helps_western_development_l1_1459

namespace GeographicInfoTech

def monitors_three_gorges_project : Prop :=
  -- Point ①
  true

def monitors_ecological_environment_meteorological_changes_and_provides_accurate_info : Prop :=
  -- Point ②
  true

def tracks_migration_tibetan_antelopes : Prop :=
  -- Point ③
  true

def addresses_ecological_environment_issues_in_southwest : Prop :=
  -- Point ④
  true

noncomputable def provides_services_for_development_western_regions : Prop :=
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes -- A (①②③)

-- Theorem stating that geographic information technology helps in ①, ②, ③ given its role
theorem geographic_info_tech_helps_western_development (h : provides_services_for_development_western_regions) :
  monitors_three_gorges_project ∧ 
  monitors_ecological_environment_meteorological_changes_and_provides_accurate_info ∧ 
  tracks_migration_tibetan_antelopes := 
by
  exact h

end GeographicInfoTech

end geographic_info_tech_helps_western_development_l1_1459


namespace total_cost_correct_l1_1744

def sandwich_cost : ℝ := 2.45
def soda_cost : ℝ := 0.87
def sandwich_quantity : ℕ := 2
def soda_quantity : ℕ := 4

theorem total_cost_correct :
  sandwich_quantity * sandwich_cost + soda_quantity * soda_cost = 8.38 := 
  by
    sorry

end total_cost_correct_l1_1744


namespace not_factorial_tail_numbers_lt_1992_l1_1336

noncomputable def factorial_tail_number_count (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + factorial_tail_number_count (n / 5)

theorem not_factorial_tail_numbers_lt_1992 :
  ∃ n, n < 1992 ∧ n = 1992 - (1992 / 5 + (1992 / 25 + (1992 / 125 + (1992 / 625 + 0)))) :=
sorry

end not_factorial_tail_numbers_lt_1992_l1_1336


namespace classroom_activity_solution_l1_1684

theorem classroom_activity_solution 
  (x y : ℕ) 
  (h1 : x - y = 6) 
  (h2 : x * y = 45) : 
  x = 11 ∧ y = 5 :=
by
  sorry

end classroom_activity_solution_l1_1684


namespace prob_three_students_exactly_two_absent_l1_1476

def prob_absent : ℚ := 1 / 30
def prob_present : ℚ := 29 / 30

theorem prob_three_students_exactly_two_absent :
  (prob_absent * prob_absent * prob_present) * 3 = 29 / 9000 := by
  sorry

end prob_three_students_exactly_two_absent_l1_1476


namespace quadratic_relationship_l1_1225

theorem quadratic_relationship (a b c : ℝ) (α : ℝ) (h₁ : α + α^2 = -b / a) (h₂ : α^3 = c / a) : b^2 = 3 * a * c + c^2 :=
by
  sorry

end quadratic_relationship_l1_1225


namespace find_n_l1_1211

theorem find_n (n : ℕ) (h₁ : 3 * n + 4 = 13) : n = 3 :=
by 
  sorry

end find_n_l1_1211


namespace ping_pong_balls_sold_l1_1746

theorem ping_pong_balls_sold (total_baseballs initial_baseballs initial_pingpong total_baseballs_sold total_balls_left : ℕ)
  (h1 : total_baseballs = 2754)
  (h2 : initial_pingpong = 1938)
  (h3 : total_baseballs_sold = 1095)
  (h4 : total_balls_left = 3021) :
  initial_pingpong - (total_balls_left - (total_baseballs - total_baseballs_sold)) = 576 :=
by sorry

end ping_pong_balls_sold_l1_1746


namespace sin_four_thirds_pi_l1_1518

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_four_thirds_pi_l1_1518


namespace largest_integer_less_than_100_leaving_remainder_4_l1_1823

theorem largest_integer_less_than_100_leaving_remainder_4 (n : ℕ) (h1 : n < 100) (h2 : n % 7 = 4) : n = 95 := 
sorry

end largest_integer_less_than_100_leaving_remainder_4_l1_1823


namespace integer_points_inequality_l1_1894

theorem integer_points_inequality
  (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b + a - b - 5 = 0)
  (M := max ((a : ℤ)^2 + (b : ℤ)^2)) :
  (3 * x^2 + 2 * y^2 <= M) → ∃ (n : ℕ), n = 51 :=
by sorry

end integer_points_inequality_l1_1894


namespace num_sequences_eq_15_l1_1792

noncomputable def num_possible_sequences : ℕ :=
  let angles_increasing_arith_seq := ∃ (x d : ℕ), x > 0 ∧ x + 4 * d < 140 ∧ 5 * x + 10 * d = 540 ∧ d ≠ 0
  by sorry

theorem num_sequences_eq_15 : num_possible_sequences = 15 := 
  by sorry

end num_sequences_eq_15_l1_1792


namespace function_values_at_mean_l1_1226

noncomputable def f (x : ℝ) : ℝ := x^2 - 10 * x + 16

theorem function_values_at_mean (x₁ x₂ : ℝ) (h₁ : x₁ = 8) (h₂ : x₂ = 2) :
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  f x' = -9 ∧ f x'' = -8 := by
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  have hx' : x' = 5 := sorry
  have hx'' : x'' = 4 := sorry
  have hf_x' : f x' = -9 := sorry
  have hf_x'' : f x'' = -8 := sorry
  exact ⟨hf_x', hf_x''⟩

end function_values_at_mean_l1_1226


namespace length_of_major_axis_l1_1665

theorem length_of_major_axis (x y : ℝ) (h : (x^2 / 25) + (y^2 / 16) = 1) : 10 = 10 :=
by
  sorry

end length_of_major_axis_l1_1665


namespace radius_of_circumscribed_circle_l1_1429

-- Definitions based on conditions
def sector (radius : ℝ) (central_angle : ℝ) : Prop :=
  central_angle = 120 ∧ radius = 10

-- Statement of the theorem we want to prove
theorem radius_of_circumscribed_circle (r R : ℝ) (h : sector r 120) : R = 20 := 
by
  sorry

end radius_of_circumscribed_circle_l1_1429


namespace pizza_area_difference_l1_1402

def hueys_hip_pizza (small_size : ℕ) (small_cost : ℕ) (large_size : ℕ) (large_cost : ℕ) : ℕ :=
  let small_area := small_size * small_size
  let large_area := large_size * large_size
  let individual_money := 30
  let pooled_money := 2 * individual_money

  let individual_small_total_area := (individual_money / small_cost) * small_area * 2
  let pooled_large_total_area := (pooled_money / large_cost) * large_area

  pooled_large_total_area - individual_small_total_area

theorem pizza_area_difference :
  hueys_hip_pizza 6 10 9 20 = 27 :=
by
  sorry

end pizza_area_difference_l1_1402


namespace increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l1_1798

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m * Real.log x

-- Part (1): Prove m >= 2 is the range for which f(x) is increasing
theorem increasing_f_iff_m_ge_two (m : ℝ) : (∀ x > 0, (2 * x - 4 + m / x) ≥ 0) ↔ m ≥ 2 := sorry

-- Part (2): Prove the given inequality for m = 3
theorem inequality_when_m_equals_three (x : ℝ) (h : x > 0) : (1 / 9) * x ^ 3 - (f x 3) > 2 := sorry

end increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l1_1798


namespace sum_of_fourth_powers_l1_1431

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25 / 6 := 
sorry

end sum_of_fourth_powers_l1_1431


namespace farm_owns_60_more_horses_than_cows_l1_1091

-- Let x be the number of cows initially
-- The number of horses initially is 4x
-- After selling 15 horses and buying 15 cows, the ratio of horses to cows becomes 7:3

theorem farm_owns_60_more_horses_than_cows (x : ℕ) (h_pos : 0 < x)
  (h_ratio : (4 * x - 15) / (x + 15) = 7 / 3) :
  (4 * x - 15) - (x + 15) = 60 :=
by
  sorry

end farm_owns_60_more_horses_than_cows_l1_1091


namespace base10_to_base7_conversion_l1_1643

theorem base10_to_base7_conversion :
  ∃ b1 b2 b3 b4 b5 : ℕ, 3 * 7^3 + 1 * 7^2 + 6 * 7^1 + 6 * 7^0 = 3527 ∧ 
  b1 = 1 ∧ b2 = 3 ∧ b3 = 1 ∧ b4 = 6 ∧ b5 = 6 ∧ (3527:ℕ) = (1*7^4 + b1*7^3 + b2*7^2 + b3*7^1 + b4*7^0) := by
sorry

end base10_to_base7_conversion_l1_1643


namespace prob_KH_then_Ace_l1_1622

noncomputable def probability_KH_then_Ace_drawn_in_sequence : ℚ :=
  let prob_first_card_is_KH := 1 / 52
  let prob_second_card_is_Ace := 4 / 51
  prob_first_card_is_KH * prob_second_card_is_Ace

theorem prob_KH_then_Ace : probability_KH_then_Ace_drawn_in_sequence = 1 / 663 := by
  sorry

end prob_KH_then_Ace_l1_1622


namespace shopkeeper_percentage_gain_l1_1659

theorem shopkeeper_percentage_gain 
    (original_price : ℝ) 
    (price_increase : ℝ) 
    (first_discount : ℝ) 
    (second_discount : ℝ)
    (new_price : ℝ) 
    (discounted_price1 : ℝ) 
    (final_price : ℝ) 
    (percentage_gain : ℝ) 
    (h1 : original_price = 100)
    (h2 : price_increase = original_price * 0.34)
    (h3 : new_price = original_price + price_increase)
    (h4 : first_discount = new_price * 0.10)
    (h5 : discounted_price1 = new_price - first_discount)
    (h6 : second_discount = discounted_price1 * 0.15)
    (h7 : final_price = discounted_price1 - second_discount)
    (h8 : percentage_gain = ((final_price - original_price) / original_price) * 100) :
    percentage_gain = 2.51 :=
by sorry

end shopkeeper_percentage_gain_l1_1659


namespace r_pow_four_solution_l1_1872

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l1_1872


namespace tangent_range_of_a_l1_1624

theorem tangent_range_of_a 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + a * x + 2 * y + a^2 = 0)
  (A : ℝ × ℝ) 
  (A_eq : A = (1, 2)) :
  -2 * Real.sqrt 3 / 3 < a ∧ a < 2 * Real.sqrt 3 / 3 :=
by
  sorry

end tangent_range_of_a_l1_1624


namespace coins_count_l1_1264

variable (x : ℕ)

def total_value : ℕ → ℕ := λ x => x + (x * 50) / 100 + (x * 25) / 100

theorem coins_count (h : total_value x = 140) : x = 80 :=
sorry

end coins_count_l1_1264


namespace least_whole_number_clock_equiv_l1_1002

theorem least_whole_number_clock_equiv (h : ℕ) (h_gt_10 : h > 10) : 
  ∃ k, k = 12 ∧ (h^2 - h) % 12 = 0 ∧ h = 12 :=
by 
  sorry

end least_whole_number_clock_equiv_l1_1002


namespace balloon_permutations_l1_1917

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l1_1917


namespace no_such_abc_exists_l1_1022

theorem no_such_abc_exists :
  ¬ ∃ (a b c : ℝ), 
      ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0) ∨
       (a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∧ c < 0 ∧ a > 0)) ∧
      ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∨ c < 0 ∧ a > 0) ∨
       (a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0)) :=
by {
  sorry
}

end no_such_abc_exists_l1_1022


namespace total_amount_spent_l1_1692

theorem total_amount_spent (num_pigs num_hens avg_price_hen avg_price_pig : ℕ)
                          (h_num_pigs : num_pigs = 3)
                          (h_num_hens : num_hens = 10)
                          (h_avg_price_hen : avg_price_hen = 30)
                          (h_avg_price_pig : avg_price_pig = 300) :
                          num_hens * avg_price_hen + num_pigs * avg_price_pig = 1200 :=
by
  sorry

end total_amount_spent_l1_1692


namespace dealer_gross_profit_l1_1121

theorem dealer_gross_profit (purchase_price : ℝ) (markup_rate : ℝ) (selling_price : ℝ) (gross_profit : ℝ) 
  (purchase_price_cond : purchase_price = 150)
  (markup_rate_cond : markup_rate = 0.25)
  (selling_price_eq : selling_price = purchase_price + (markup_rate * selling_price))
  (gross_profit_eq : gross_profit = selling_price - purchase_price) : 
  gross_profit = 50 :=
by
  sorry

end dealer_gross_profit_l1_1121


namespace weight_of_one_bowling_ball_l1_1309

-- Definitions from the problem conditions
def weight_canoe := 36
def num_canoes := 4
def num_bowling_balls := 9

-- Calculate the total weight of the canoes
def total_weight_canoes := num_canoes * weight_canoe

-- Prove the weight of one bowling ball
theorem weight_of_one_bowling_ball : (total_weight_canoes / num_bowling_balls) = 16 := by
  sorry

end weight_of_one_bowling_ball_l1_1309


namespace richard_older_than_david_l1_1815

theorem richard_older_than_david
  (R D S : ℕ)   -- ages of Richard, David, Scott
  (x : ℕ)       -- the number of years Richard is older than David
  (h1 : R = D + x)
  (h2 : D = S + 8)
  (h3 : R + 8 = 2 * (S + 8))
  (h4 : D = 14) : 
  x = 6 := sorry

end richard_older_than_david_l1_1815


namespace seq_value_at_2018_l1_1958

noncomputable def f (x : ℝ) : ℝ := sorry
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = f 0 ∧ ∀ (n : ℕ), n > 0 → f (a (n + 1)) = 1 / f (-2 - a n)

theorem seq_value_at_2018 (a : ℕ → ℝ) (h_seq : seq a) : a 2018 = 4035 := 
by sorry

end seq_value_at_2018_l1_1958


namespace campers_afternoon_l1_1414

theorem campers_afternoon (x : ℕ) 
  (h1 : 44 = x + 5) : 
  x = 39 := 
by
  sorry

end campers_afternoon_l1_1414


namespace units_digit_of_3_pow_7_pow_6_l1_1536

theorem units_digit_of_3_pow_7_pow_6 :
  (3 ^ (7 ^ 6) % 10) = 3 := 
sorry

end units_digit_of_3_pow_7_pow_6_l1_1536


namespace total_emails_received_l1_1380

theorem total_emails_received (E : ℝ)
    (h1 : (3/5) * (3/4) * E = 180) :
    E = 400 :=
sorry

end total_emails_received_l1_1380


namespace range_of_a_l1_1181

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - a / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem range_of_a (e : ℝ) (a : ℝ) (H : ∀ x ∈ Set.Icc 1 e, f x ≥ g x a) :
  -2 ≤ a ∧ a ≤ (2 * e) / (e - 1) :=
by
  sorry

end range_of_a_l1_1181


namespace wall_length_proof_l1_1489

-- Define the conditions from the problem
def wall_height : ℝ := 100 -- Height in cm
def wall_thickness : ℝ := 5 -- Thickness in cm
def brick_length : ℝ := 25 -- Brick length in cm
def brick_width : ℝ := 11 -- Brick width in cm
def brick_height : ℝ := 6 -- Brick height in cm
def number_of_bricks : ℝ := 242.42424242424244

-- Calculate the volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Calculate the total volume of the bricks
def total_brick_volume : ℝ := brick_volume * number_of_bricks

-- Define the proof problem
theorem wall_length_proof : total_brick_volume = wall_height * wall_thickness * 800 :=
sorry

end wall_length_proof_l1_1489


namespace solution1_solution2_l1_1027

noncomputable def problem1 (a : ℝ) : Prop :=
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + 4 < 0)

theorem solution1 (a : ℝ) : problem1 a ↔ a < -3 ∨ a ≥ -1 := 
  sorry

noncomputable def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (-2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0)

noncomputable def condition2 (a x : ℝ) : Prop :=
  (2*a < x ∧ x < a+1)

theorem solution2 (a : ℝ) : (∀ x, condition2 a x → problem2 a x) → a ≥ -1/2 :=
  sorry

end solution1_solution2_l1_1027


namespace total_cost_family_visit_l1_1107

/-
Conditions:
1. entrance_ticket_cost: $5 per person
2. attraction_ticket_cost_kid: $2 per kid
3. attraction_ticket_cost_parent: $4 per parent
4. family_discount_threshold: A family of 6 or more gets a 10% discount on entrance tickets
5. senior_discount: Senior citizens get a 50% discount on attraction tickets
6. family_composition: 4 children, 2 parents, and 1 grandmother
7. visit_attraction: The family plans to visit at least one attraction
-/

def entrance_ticket_cost : ℝ := 5
def attraction_ticket_cost_kid : ℝ := 2
def attraction_ticket_cost_parent : ℝ := 4
def family_discount_threshold : ℕ := 6
def family_discount_rate : ℝ := 0.10
def senior_discount_rate : ℝ := 0.50
def number_of_kids : ℕ := 4
def number_of_parents : ℕ := 2
def number_of_seniors : ℕ := 1

theorem total_cost_family_visit : 
  let total_entrance_fee := (number_of_kids + number_of_parents + number_of_seniors) * entrance_ticket_cost 
  let total_entrance_fee_discounted := total_entrance_fee * (1 - family_discount_rate)
  let total_attraction_fee_kids := number_of_kids * attraction_ticket_cost_kid
  let total_attraction_fee_parents := number_of_parents * attraction_ticket_cost_parent
  let total_attraction_fee_seniors := number_of_seniors * attraction_ticket_cost_parent * (1 - senior_discount_rate)
  let total_attraction_fee := total_attraction_fee_kids + total_attraction_fee_parents + total_attraction_fee_seniors
  (number_of_kids + number_of_parents + number_of_seniors ≥ family_discount_threshold) → 
  (total_entrance_fee_discounted + total_attraction_fee = 49.50) :=
by
  -- Assuming we calculate entrance fee and attraction fee correctly, state the theorem
  sorry

end total_cost_family_visit_l1_1107


namespace problem_statement_l1_1154

theorem problem_statement (a b c : ℤ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 3) 
  (h3 : |c| = 6) 
  (h4 : |a + b| = - (a + b)) 
  (h5 : |a + c| = a + c) : 
  a - b + c = -2 ∨ a - b + c = 4 :=
sorry

end problem_statement_l1_1154


namespace pyramid_volume_is_1_12_l1_1487

def base_rectangle_length_1 := 1
def base_rectangle_width_1_4 := 1 / 4
def pyramid_height_1 := 1

noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * (base_rectangle_length_1 * base_rectangle_width_1_4) * pyramid_height_1

theorem pyramid_volume_is_1_12 : pyramid_volume = 1 / 12 :=
sorry

end pyramid_volume_is_1_12_l1_1487


namespace maximum_expression_value_l1_1220

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l1_1220


namespace sean_whistles_l1_1548

def charles_whistles : ℕ := 13
def extra_whistles : ℕ := 32

theorem sean_whistles : charles_whistles + extra_whistles = 45 := by
  sorry

end sean_whistles_l1_1548


namespace find_f_1988_l1_1273

def f : ℕ+ → ℕ+ := sorry

axiom functional_equation (m n : ℕ+) : f (f m + f n) = m + n

theorem find_f_1988 : f 1988 = 1988 :=
by sorry

end find_f_1988_l1_1273


namespace savings_increase_is_100_percent_l1_1066

variable (I : ℝ) -- Initial income
variable (S : ℝ) -- Initial savings
variable (I2 : ℝ) -- Income in the second year
variable (E1 : ℝ) -- Expenditure in the first year
variable (E2 : ℝ) -- Expenditure in the second year
variable (S2 : ℝ) -- Second year savings

-- Initial conditions
def initial_savings (I : ℝ) : ℝ := 0.25 * I
def first_year_expenditure (I : ℝ) (S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.25 * I

-- Total expenditure condition
def total_expenditure_condition (E1 : ℝ) (E2 : ℝ) : Prop := E1 + E2 = 2 * E1

-- Prove that the savings increase in the second year is 100%
theorem savings_increase_is_100_percent :
   ∀ (I S E1 I2 E2 S2 : ℝ),
     S = initial_savings I →
     E1 = first_year_expenditure I S →
     I2 = second_year_income I →
     total_expenditure_condition E1 E2 →
     S2 = I2 - E2 →
     ((S2 - S) / S) * 100 = 100 := by
  sorry

end savings_increase_is_100_percent_l1_1066


namespace evaluate_expression_l1_1256

theorem evaluate_expression : 
  (Real.sqrt 3 + 3 + (1 / (Real.sqrt 3 + 3))^2 + 1 / (3 - Real.sqrt 3)) = Real.sqrt 3 + 3 + 5 / 6 := by
  sorry

end evaluate_expression_l1_1256


namespace conjugate_system_solution_l1_1899

theorem conjugate_system_solution (a b : ℝ) :
  (∀ x y : ℝ,
    (x + (2-a) * y = b + 1) ∧ ((2*a-7) * x + y = -5 - b)
    ↔ x + (2*a-7) * y = -5 - b ∧ (x + (2-a) * y = b + 1))
  ↔ a = 3 ∧ b = -3 := by
  sorry

end conjugate_system_solution_l1_1899


namespace students_in_school_l1_1953

variable (S : ℝ)
variable (W : ℝ)
variable (L : ℝ)

theorem students_in_school {S W L : ℝ} 
  (h1 : W = 0.55 * 0.25 * S)
  (h2 : L = 0.45 * 0.25 * S)
  (h3 : W = L + 50) : 
  S = 2000 := 
sorry

end students_in_school_l1_1953


namespace phoneExpences_l1_1886

structure PhonePlan where
  fixed_fee : ℝ
  free_minutes : ℕ
  excess_rate : ℝ -- rate per minute

def JanuaryUsage : ℕ := 15 * 60 + 17 -- 15 hours 17 minutes in minutes
def FebruaryUsage : ℕ := 9 * 60 + 55 -- 9 hours 55 minutes in minutes

def computeBill (plan : PhonePlan) (usage : ℕ) : ℝ :=
  let excess_minutes := (usage - plan.free_minutes).max 0
  plan.fixed_fee + (excess_minutes * plan.excess_rate)

theorem phoneExpences (plan : PhonePlan) :
  plan = { fixed_fee := 18.00, free_minutes := 600, excess_rate := 0.03 } →
  computeBill plan JanuaryUsage + computeBill plan FebruaryUsage = 45.51 := by
  sorry

end phoneExpences_l1_1886


namespace triangle_area_zero_vertex_l1_1253

theorem triangle_area_zero_vertex (x1 y1 x2 y2 : ℝ) :
  (1 / 2) * |x1 * y2 - x2 * y1| = 
    abs (1 / 2 * (x1 * y2 - x2 * y1)) := 
sorry

end triangle_area_zero_vertex_l1_1253


namespace distributive_laws_none_hold_l1_1387

def star (a b : ℝ) : ℝ := a + b + a * b

theorem distributive_laws_none_hold (x y z : ℝ) :
  ¬ (x * (y + z) = (x * y) + (x * z)) ∧
  ¬ (x + (y * z) = (x + y) * (x + z)) ∧
  ¬ (x * (y * z) = (x * y) * (x * z)) :=
by
  sorry

end distributive_laws_none_hold_l1_1387


namespace intersection_coordinates_l1_1266

theorem intersection_coordinates (x y : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : y = x + 1) : 
  x = 2 ∧ y = 3 := 
by 
  sorry

end intersection_coordinates_l1_1266


namespace least_number_of_teams_l1_1206

theorem least_number_of_teams
  (total_athletes : ℕ)
  (max_team_size : ℕ)
  (h_total : total_athletes = 30)
  (h_max : max_team_size = 12) :
  ∃ (number_of_teams : ℕ) (team_size : ℕ),
    number_of_teams * team_size = total_athletes ∧
    team_size ≤ max_team_size ∧
    number_of_teams = 3 :=
by
  sorry

end least_number_of_teams_l1_1206


namespace algebra_problem_l1_1671

-- Definition of variable y
variable (y : ℝ)

-- Given the condition
axiom h : 2 * y^2 + 3 * y + 7 = 8

-- We need to prove that 4 * y^2 + 6 * y - 9 = -7 given the condition
theorem algebra_problem : 4 * y^2 + 6 * y - 9 = -7 :=
by sorry

end algebra_problem_l1_1671


namespace ellipse_eccentricity_half_l1_1186

-- Definitions and assumptions
variable (a b c e : ℝ)
variable (h₁ : a = 2 * c)
variable (h₂ : b = sqrt 3 * c)
variable (eccentricity_def : e = c / a)

-- Theorem statement
theorem ellipse_eccentricity_half : e = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_half_l1_1186


namespace least_number_to_add_l1_1816

theorem least_number_to_add (LCM : ℕ) (a : ℕ) (x : ℕ) :
  LCM = 23 * 29 * 31 →
  a = 1076 →
  x = LCM - a →
  (a + x) % LCM = 0 :=
by
  sorry

end least_number_to_add_l1_1816


namespace painting_problem_l1_1719

theorem painting_problem (initial_painters : ℕ) (initial_days : ℚ) (initial_rate : ℚ) (new_days : ℚ) (new_rate : ℚ) : 
  initial_painters = 6 ∧ initial_days = 5/2 ∧ initial_rate = 2 ∧ new_days = 2 ∧ new_rate = 2.5 →
  ∃ additional_painters : ℕ, additional_painters = 0 :=
by
  intros h
  sorry

end painting_problem_l1_1719


namespace quadratic_range_l1_1589

noncomputable def quadratic_condition (a m : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ (- (1 + 1 / m) > 0) ∧
  (3 * m^2 - 2 * m - 1 ≤ 0)

theorem quadratic_range (a m : ℝ) :
  quadratic_condition a m → - (1 / 3) ≤ m ∧ m < 0 :=
by sorry

end quadratic_range_l1_1589


namespace tips_earned_l1_1648

theorem tips_earned
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (tip_customers := total_customers - no_tip_customers)
  (total_tips := tip_customers * tip_amount)
  (h1 : total_customers = 9)
  (h2 : no_tip_customers = 5)
  (h3 : tip_amount = 8) :
  total_tips = 32 := by
  -- Proof goes here
  sorry

end tips_earned_l1_1648


namespace relay_race_total_distance_l1_1579

theorem relay_race_total_distance
  (Sadie_speed : ℝ) (Sadie_time : ℝ) (Ariana_speed : ℝ) (Ariana_time : ℝ) (Sarah_speed : ℝ) (total_race_time : ℝ)
  (h1 : Sadie_speed = 3) (h2 : Sadie_time = 2)
  (h3 : Ariana_speed = 6) (h4 : Ariana_time = 0.5)
  (h5 : Sarah_speed = 4) (h6 : total_race_time = 4.5) :
  (Sadie_speed * Sadie_time + Ariana_speed * Ariana_time + Sarah_speed * (total_race_time - (Sadie_time + Ariana_time))) = 17 :=
by
  sorry

end relay_race_total_distance_l1_1579


namespace tangent_slope_at_zero_l1_1301

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_slope_at_zero :
  (deriv f 0) = 1 := by 
  sorry

end tangent_slope_at_zero_l1_1301


namespace net_effect_sale_value_net_effect_sale_value_percentage_increase_l1_1913

def sale_value (P Q : ℝ) : ℝ := P * Q

theorem net_effect_sale_value (P Q : ℝ) :
  sale_value (0.8 * P) (1.8 * Q) = 1.44 * sale_value P Q :=
by
  sorry

theorem net_effect_sale_value_percentage_increase (P Q : ℝ) :
  (sale_value (0.8 * P) (1.8 * Q) - sale_value P Q) / sale_value P Q = 0.44 :=
by
  sorry

end net_effect_sale_value_net_effect_sale_value_percentage_increase_l1_1913


namespace list_length_eq_12_l1_1828

-- Define a list of numbers in the sequence
def seq : List ℝ := [1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5]

-- Define the theorem that states the number of elements in the sequence
theorem list_length_eq_12 : seq.length = 12 := 
by 
  -- Proof here
  sorry

end list_length_eq_12_l1_1828


namespace least_number_conditioned_l1_1270

theorem least_number_conditioned (n : ℕ) :
  n % 56 = 3 ∧ n % 78 = 3 ∧ n % 9 = 0 ↔ n = 2187 := 
sorry

end least_number_conditioned_l1_1270


namespace fraction_broke_off_l1_1946

variable (p p_1 p_2 : ℝ)
variable (k : ℝ)

-- Conditions
def initial_mass : Prop := p_1 + p_2 = p
def value_relation : Prop := p_1^2 + p_2^2 = 0.68 * p^2

-- Goal
theorem fraction_broke_off (h1 : initial_mass p p_1 p_2)
                           (h2 : value_relation p p_1 p_2) :
  (p_2 / p) = 1 / 5 :=
sorry

end fraction_broke_off_l1_1946


namespace find_a_l1_1439

noncomputable def A := {x : ℝ | x^2 - 8 * x + 15 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 1 = 0}

theorem find_a (a : ℝ) : (A ∩ B a = B a) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end find_a_l1_1439


namespace calculate_area_correct_l1_1020

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the rotation angles in degrees
def rotation_angles : List ℝ := [0, 30, 45, 60]

-- Define the area calculation function (to be implemented)
def calculate_overlap_area (s : ℝ) (angles : List ℝ) : ℝ := sorry

-- Define the proof that the calculated area is equal to 123.475
theorem calculate_area_correct : calculate_overlap_area side_length rotation_angles = 123.475 :=
by
  sorry

end calculate_area_correct_l1_1020


namespace max_sum_of_roots_l1_1801

theorem max_sum_of_roots (a b : ℝ) (h_a : a ≠ 0) (m : ℝ) :
  (∀ x : ℝ, (2 * x ^ 2 - 5 * x + m = 0) → 25 - 8 * m ≥ 0) →
  (∃ s, s = -5 / 2) → m = 25 / 8 :=
by
  sorry

end max_sum_of_roots_l1_1801


namespace find_c_l1_1703

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_c (a b c : ℝ) 
  (h1 : perpendicular (a / 2) (-2 / b))
  (h2 : a = b)
  (h3 : a * 1 - 2 * (-5) = c) 
  (h4 : 2 * 1 + b * (-5) = -c) : 
  c = 13 := by
  sorry

end find_c_l1_1703


namespace geometric_sequence_property_l1_1163

open Classical

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property :
  ∃ (a : ℕ → ℝ) (q : ℝ), q < 0 ∧ geometric_sequence a q ∧
    a 1 = 1 - a 0 ∧ a 3 = 4 - a 2 ∧ a 3 + a 4 = -8 :=
by
  sorry

end geometric_sequence_property_l1_1163


namespace point_Q_in_third_quadrant_l1_1796

-- Define point P in the fourth quadrant with coordinates a and b.
variable (a b : ℝ)
variable (h1 : a > 0)  -- Condition for the x-coordinate of P in fourth quadrant
variable (h2 : b < 0)  -- Condition for the y-coordinate of P in fourth quadrant

-- Point Q is defined by the coordinates (-a, b-1). We need to show it lies in the third quadrant.
theorem point_Q_in_third_quadrant : (-a < 0) ∧ (b - 1 < 0) :=
  by
    sorry

end point_Q_in_third_quadrant_l1_1796


namespace locate_z_in_fourth_quadrant_l1_1890

def z_in_quadrant_fourth (z : ℂ) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem locate_z_in_fourth_quadrant (z : ℂ) (i : ℂ) (h : i * i = -1) 
(hz : z * (1 + i) = 1) : z_in_quadrant_fourth z :=
sorry

end locate_z_in_fourth_quadrant_l1_1890


namespace students_in_either_but_not_both_l1_1126

-- Definitions and conditions
def both : ℕ := 18
def geom : ℕ := 35
def only_stats : ℕ := 16

-- Correct answer to prove
def total_not_both : ℕ := geom - both + only_stats

theorem students_in_either_but_not_both : total_not_both = 33 := by
  sorry

end students_in_either_but_not_both_l1_1126


namespace power_inequality_l1_1198

theorem power_inequality (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx : abs x < 1) : 
  2^n > (1 - x)^n + (1 + x)^n := 
sorry

end power_inequality_l1_1198


namespace smallest_d_value_l1_1283

theorem smallest_d_value : 
  ∃ d : ℝ, (d ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d + 5) = 4 * d) ∧ ∀ d' : ℝ, (d' ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d' + 5) = 4 * d') → (3 ≤ d') → d = 3 := 
by
  sorry

end smallest_d_value_l1_1283


namespace ratio_transformation_l1_1724

theorem ratio_transformation (x1 y1 x2 y2 : ℚ) (h₁ : x1 / y1 = 7 / 5) (h₂ : x2 = x1 * y1) (h₃ : y2 = y1 * x1) : x2 / y2 = 1 := by
  sorry

end ratio_transformation_l1_1724


namespace union_of_sets_l1_1994

-- Definitions based on conditions
def A : Set ℕ := {2, 3}
def B (a : ℕ) : Set ℕ := {1, a}
def condition (a : ℕ) : Prop := A ∩ (B a) = {2}

-- Main theorem to be proven
theorem union_of_sets (a : ℕ) (h : condition a) : A ∪ (B a) = {1, 2, 3} :=
sorry

end union_of_sets_l1_1994


namespace parameterize_circle_l1_1502

noncomputable def parametrization (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 1) / (t^2 + 1), (-2 * t) / (t^2 + 1) )

theorem parameterize_circle (t : ℝ) : 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  (x^2 + y^2) = 1 :=
by 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  sorry

end parameterize_circle_l1_1502


namespace sum_of_altitudes_of_triangle_l1_1711

theorem sum_of_altitudes_of_triangle : 
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  altitude1 + altitude2 + altitude3 = (22 * Real.sqrt 73 + 48) / Real.sqrt 73 :=
by
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  sorry

end sum_of_altitudes_of_triangle_l1_1711


namespace total_flowers_l1_1496

def tulips : ℕ := 3
def carnations : ℕ := 4

theorem total_flowers : tulips + carnations = 7 := by
  sorry

end total_flowers_l1_1496


namespace unique_n_value_l1_1207

def is_n_table (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ i j, 
    (∀ k : Fin n, A i j ≥ A i k) ∧   -- Max in its row
    (∀ k : Fin n, A i j ≤ A k j)     -- Min in its column

theorem unique_n_value 
  {n : ℕ} (h : 2 ≤ n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (hA : ∀ i j, A i j ∈ Finset.range (n^2)) -- Each number appears exactly once
  (hn : is_n_table n A) : 
  ∃! a, ∃ i j, A i j = a ∧ 
           (∀ k : Fin n, a ≥ A i k) ∧ 
           (∀ k : Fin n, a ≤ A k j) := 
sorry

end unique_n_value_l1_1207


namespace polar_to_rectangular_conversion_l1_1615

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular 5 (5 * Real.pi / 4) = (-5 * Real.sqrt 2 / 2, -5 * Real.sqrt 2 / 2) := by
  sorry

end polar_to_rectangular_conversion_l1_1615


namespace min_value_a_p_a_q_l1_1766

theorem min_value_a_p_a_q (a : ℕ → ℕ) (p q : ℕ) (h_arith_geom : ∀ n, a (n + 2) = a (n + 1) + a n * 2)
(h_a9 : a 9 = a 8 + 2 * a 7)
(h_ap_aq : a p * a q = 8 * a 1 ^ 2) :
    (1 / p : ℝ) + (4 / q : ℝ) = 9 / 5 := by
    sorry

end min_value_a_p_a_q_l1_1766


namespace hyperbola_equation_l1_1759

open Real

-- Define the conditions in Lean
def is_hyperbola_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_positive (x : ℝ) : Prop := x > 0

def parabola_focus : (ℝ × ℝ) := (1, 0)

def hyperbola_vertex_eq_focus (a : ℝ) : Prop := a = parabola_focus.1

def hyperbola_eccentricity (e a c : ℝ) : Prop := e = c / a

-- Our proof statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), is_positive a ∧ is_positive b ∧
  hyperbola_vertex_eq_focus a ∧
  hyperbola_eccentricity (sqrt 5) a (sqrt 5) ∧
  is_hyperbola_form a b 1 0 :=
by sorry

end hyperbola_equation_l1_1759


namespace electric_fan_wattage_l1_1098

theorem electric_fan_wattage (hours_per_day : ℕ) (energy_per_month : ℝ) (days_per_month : ℕ) 
  (h1 : hours_per_day = 8) (h2 : energy_per_month = 18) (h3 : days_per_month = 30) : 
  (energy_per_month * 1000) / (days_per_month * hours_per_day) = 75 := 
by { 
  -- Placeholder for the proof
  sorry 
}

end electric_fan_wattage_l1_1098


namespace incorrect_statement_D_l1_1037

theorem incorrect_statement_D :
  (∃ x : ℝ, x ^ 3 = -64 ∧ x = -4) ∧
  (∃ y : ℝ, y ^ 2 = 49 ∧ y = 7) ∧
  (∃ z : ℝ, z ^ 3 = 1 / 27 ∧ z = 1 / 3) ∧
  (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4 ∨ w = -1 / 4)
  → ¬ (∀ w : ℝ, w ^ 2 = 1 / 16 → w = 1 / 4) :=
by
  sorry

end incorrect_statement_D_l1_1037


namespace max_value_of_f_l1_1344

def f (x : ℝ) : ℝ := 10 * x - 2 * x ^ 2

theorem max_value_of_f : ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) :=
  ⟨12.5, sorry⟩

end max_value_of_f_l1_1344


namespace inequality_solution_set_compare_mn_and_2m_plus_2n_l1_1595

def f (x : ℝ) : ℝ := |x| + |x - 3|

theorem inequality_solution_set :
  {x : ℝ | f x - 5 ≥ x} = { x : ℝ | x ≤ -2 / 3 } ∪ { x : ℝ | x ≥ 8 } :=
sorry

theorem compare_mn_and_2m_plus_2n (m n : ℝ) (hm : ∃ x, m = f x) (hn : ∃ x, n = f x) :
  2 * (m + n) < m * n + 4 :=
sorry

end inequality_solution_set_compare_mn_and_2m_plus_2n_l1_1595


namespace probability_of_diamond_or_ace_at_least_one_l1_1236

noncomputable def prob_at_least_one_diamond_or_ace : ℚ := 
  1 - (9 / 13) ^ 2

theorem probability_of_diamond_or_ace_at_least_one :
  prob_at_least_one_diamond_or_ace = 88 / 169 := 
by
  sorry

end probability_of_diamond_or_ace_at_least_one_l1_1236


namespace cos_theta_plus_5π_div_6_l1_1043

theorem cos_theta_plus_5π_div_6 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcond : Real.sin (θ / 2 + π / 6) = 3 / 5) :
  Real.cos (θ + 5 * π / 6) = -24 / 25 :=
by
  sorry -- Proof is skipped as instructed

end cos_theta_plus_5π_div_6_l1_1043


namespace sum_due_is_l1_1029

-- Definitions and conditions from the problem
def BD : ℤ := 288
def TD : ℤ := 240
def face_value (FV : ℤ) : Prop := BD = TD + (TD * TD) / FV

-- Proof statement
theorem sum_due_is (FV : ℤ) (h : face_value FV) : FV = 1200 :=
sorry

end sum_due_is_l1_1029


namespace unique_non_congruent_rectangle_with_conditions_l1_1988

theorem unique_non_congruent_rectangle_with_conditions :
  ∃! (w h : ℕ), 2 * (w + h) = 80 ∧ w * h = 400 :=
by
  sorry

end unique_non_congruent_rectangle_with_conditions_l1_1988


namespace ce_length_l1_1399

noncomputable def CE_in_parallelogram (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) : ℝ :=
  280

theorem ce_length (AB AD BD : ℝ) (AB_eq : AB = 480) (AD_eq : AD = 200) (BD_eq : BD = 625) :
  CE_in_parallelogram AB AD BD AB_eq AD_eq BD_eq = 280 :=
by
  sorry

end ce_length_l1_1399


namespace exists_polynomials_Q_R_l1_1160

theorem exists_polynomials_Q_R (P : Polynomial ℝ) (hP : ∀ x > 0, P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ a, 0 ≤ a → ∀ b, 0 ≤ b → Q.coeff a ≥ 0 ∧ R.coeff b ≥ 0) ∧ ∀ x > 0, P.eval x = (Q.eval x) / (R.eval x) := 
by
  sorry

end exists_polynomials_Q_R_l1_1160


namespace Chandler_saves_enough_l1_1850

theorem Chandler_saves_enough (total_cost gift_money weekly_earnings : ℕ)
  (h_cost : total_cost = 550)
  (h_gift : gift_money = 130)
  (h_weekly : weekly_earnings = 18) : ∃ x : ℕ, (130 + 18 * x) >= 550 ∧ x = 24 := 
by
  sorry

end Chandler_saves_enough_l1_1850


namespace reduced_price_l1_1566

theorem reduced_price (P R : ℝ) (Q : ℝ) 
  (h1 : R = 0.80 * P) 
  (h2 : 600 = Q * P) 
  (h3 : 600 = (Q + 4) * R) : 
  R = 30 :=
by
  sorry

end reduced_price_l1_1566


namespace prob_4_consecutive_baskets_prob_exactly_4_baskets_l1_1296

theorem prob_4_consecutive_baskets 
  (p : ℝ) (h : p = 1/2) : 
  (p^4 * (1 - p) + (1 - p) * p^4) = 1/16 :=
by sorry

theorem prob_exactly_4_baskets 
  (p : ℝ) (h : p = 1/2) : 
  5 * p^4 * (1 - p) = 5/32 :=
by sorry

end prob_4_consecutive_baskets_prob_exactly_4_baskets_l1_1296


namespace silvia_last_play_without_breach_l1_1837

theorem silvia_last_play_without_breach (N : ℕ) : 
  36 * N < 2000 ∧ 72 * N ≥ 2000 ↔ N = 28 :=
by
  sorry

end silvia_last_play_without_breach_l1_1837


namespace ratio_of_sums_l1_1618

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2
axiom a4_eq_2a3 : a 4 = 2 * a 3

theorem ratio_of_sums (a : ℕ → ℝ) (S : ℕ → ℝ)
                      (arithmetic_sum : ∀ n, S n = n * (a 1 + a n) / 2)
                      (a4_eq_2a3 : a 4 = 2 * a 3) :
  S 7 / S 5 = 14 / 5 :=
by sorry

end ratio_of_sums_l1_1618


namespace baker_cakes_l1_1383

theorem baker_cakes (C : ℕ) (h1 : 154 = 78 + 76) (h2 : C = 78) : C = 78 :=
sorry

end baker_cakes_l1_1383


namespace cloud_ratio_l1_1638

theorem cloud_ratio (D Carson Total : ℕ) (h1 : Carson = 6) (h2 : Total = 24) (h3 : Carson + D = Total) :
  (D / Carson) = 3 := by
  sorry

end cloud_ratio_l1_1638


namespace fraction_difference_l1_1786

theorem fraction_difference : 7 / 12 - 3 / 8 = 5 / 24 := 
by 
  sorry

end fraction_difference_l1_1786


namespace sticks_per_chair_l1_1856

-- defining the necessary parameters and conditions
def sticksPerTable := 9
def sticksPerStool := 2
def sticksPerHour := 5
def chairsChopped := 18
def tablesChopped := 6
def stoolsChopped := 4
def hoursKeptWarm := 34

-- calculation of total sticks needed
def totalSticksNeeded := sticksPerHour * hoursKeptWarm

-- the main theorem to prove the number of sticks a chair makes
theorem sticks_per_chair (C : ℕ) : (chairsChopped * C) + (tablesChopped * sticksPerTable) + (stoolsChopped * sticksPerStool) = totalSticksNeeded → C = 6 := by
  sorry

end sticks_per_chair_l1_1856


namespace eggs_broken_l1_1545

theorem eggs_broken (brown_eggs white_eggs total_pre total_post broken_eggs : ℕ) 
  (h1 : brown_eggs = 10)
  (h2 : white_eggs = 3 * brown_eggs)
  (h3 : total_pre = brown_eggs + white_eggs)
  (h4 : total_post = 20)
  (h5 : broken_eggs = total_pre - total_post) : broken_eggs = 20 :=
by
  sorry

end eggs_broken_l1_1545


namespace triangle_third_side_l1_1861

theorem triangle_third_side (a b c : ℝ) (h1 : a = 5) (h2 : b = 7) (h3 : 2 < c ∧ c < 12) : c = 6 :=
sorry

end triangle_third_side_l1_1861


namespace find_n_l1_1064

noncomputable def arctan_sum_eq_pi_over_2 (n : ℕ) : Prop :=
  Real.arctan (1 / 3) + Real.arctan (1 / 4) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2

theorem find_n (h : ∃ n, arctan_sum_eq_pi_over_2 n) : ∃ n, n = 54 := by
  obtain ⟨n, hn⟩ := h
  have H : 1 / 3 + 1 / 4 + 1 / 7 < 1 := by sorry
  sorry

end find_n_l1_1064


namespace no_nat_numbers_satisfy_eqn_l1_1765

theorem no_nat_numbers_satisfy_eqn (a b : ℕ) : a^2 - 3 * b^2 ≠ 8 := by
  sorry

end no_nat_numbers_satisfy_eqn_l1_1765


namespace b_catches_a_distance_l1_1501

-- Define the initial conditions
def a_speed : ℝ := 10  -- A's speed in km/h
def b_speed : ℝ := 20  -- B's speed in km/h
def start_delay : ℝ := 3  -- B starts cycling 3 hours after A in hours

-- Define the target distance to prove
theorem b_catches_a_distance : ∃ (d : ℝ), d = 60 := 
by 
  sorry

end b_catches_a_distance_l1_1501


namespace triangle_is_isosceles_l1_1612

variable {α β γ : ℝ} (quadrilateral_angles : List ℝ)

-- Conditions from the problem
axiom triangle_angle_sum : α + β + γ = 180
axiom quadrilateral_angle_sum : quadrilateral_angles.sum = 360
axiom quadrilateral_angle_conditions : ∀ (a b : ℝ), a ∈ [α, β, γ] → b ∈ [α, β, γ] → a ≠ b → (a + b ∈ quadrilateral_angles)

-- Proof statement
theorem triangle_is_isosceles : (α = β) ∨ (β = γ) ∨ (γ = α) := 
  sorry

end triangle_is_isosceles_l1_1612


namespace problem_1_problem_2_problem_3_l1_1233

def M := {n : ℕ | 0 < n ∧ n < 1000}

def circ (a b : ℕ) : ℕ :=
  if a * b < 1000 then a * b
  else 
    let k := (a * b) / 1000
    let r := (a * b) % 1000
    if k + r < 1000 then k + r
    else (k + r) % 1000 + 1

theorem problem_1 : circ 559 758 = 146 := 
by
  sorry

theorem problem_2 : ∃ (x : ℕ) (h : x ∈ M), circ 559 x = 1 ∧ x = 361 :=
by
  sorry

theorem problem_3 : ∀ (a b c : ℕ) (h₁ : a ∈ M) (h₂ : b ∈ M) (h₃ : c ∈ M), circ a (circ b c) = circ (circ a b) c :=
by
  sorry

end problem_1_problem_2_problem_3_l1_1233


namespace intercept_sum_modulo_l1_1999

theorem intercept_sum_modulo (x_0 y_0 : ℤ) (h1 : 0 ≤ x_0) (h2 : x_0 < 17) (h3 : 0 ≤ y_0) (h4 : y_0 < 17)
                       (hx : 5 * x_0 ≡ 2 [ZMOD 17])
                       (hy : 3 * y_0 ≡ 15 [ZMOD 17]) :
    x_0 + y_0 = 19 := 
by
  sorry

end intercept_sum_modulo_l1_1999


namespace library_charge_l1_1641

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l1_1641


namespace marbles_problem_l1_1777

theorem marbles_problem (h_total: ℕ) (h_each: ℕ) (h_total_eq: h_total = 35) (h_each_eq: h_each = 7) :
    h_total / h_each = 5 := by
  sorry

end marbles_problem_l1_1777


namespace least_sub_to_make_div_by_10_l1_1285

theorem least_sub_to_make_div_by_10 : 
  ∃ n, n = 8 ∧ ∀ k, 427398 - k = 10 * m → k ≥ n ∧ k = 8 :=
sorry

end least_sub_to_make_div_by_10_l1_1285


namespace fine_per_day_of_absence_l1_1031

theorem fine_per_day_of_absence :
  ∃ x: ℝ, ∀ (total_days work_wage total_received_days absent_days: ℝ),
  total_days = 30 →
  work_wage = 10 →
  total_received_days = 216 →
  absent_days = 7 →
  (total_days - absent_days) * work_wage - (absent_days * x) = total_received_days :=
sorry

end fine_per_day_of_absence_l1_1031


namespace min_n_Sn_greater_1020_l1_1418

theorem min_n_Sn_greater_1020 : ∃ n : ℕ, (n ≥ 0) ∧ (2^(n+1) - 2 - n > 1020) ∧ ∀ m : ℕ, (m ≥ 0) ∧ (m < n) → (2^(m+1) - 2 - m ≤ 1020) :=
by
  sorry

end min_n_Sn_greater_1020_l1_1418


namespace football_team_progress_l1_1428

theorem football_team_progress (lost_yards gained_yards : Int) : lost_yards = -5 → gained_yards = 13 → lost_yards + gained_yards = 8 := 
by
  intros h_lost h_gained
  rw [h_lost, h_gained]
  sorry

end football_team_progress_l1_1428


namespace find_alpha_angle_l1_1836

theorem find_alpha_angle :
  ∃ α : ℝ, (7 * α + 8 * α + 45) = 180 ∧ α = 9 :=
by 
  sorry

end find_alpha_angle_l1_1836


namespace exponent_multiplication_l1_1023

theorem exponent_multiplication (a : ℝ) : (a^3) * (a^2) = a^5 := 
by
  -- Using the property of exponents: a^m * a^n = a^(m + n)
  sorry

end exponent_multiplication_l1_1023


namespace calculate_product_l1_1585

theorem calculate_product : (3 * 5 * 7 = 38) → (13 * 15 * 17 = 268) → 1 * 3 * 5 = 15 :=
by
  intros h1 h2
  sorry

end calculate_product_l1_1585


namespace area_is_prime_number_l1_1478

open Real Int

noncomputable def area_of_triangle (a : Int) : Real :=
  (a * a : Real) / 20

theorem area_is_prime_number 
  (a : Int) 
  (h1 : ∃ p : ℕ, Nat.Prime p ∧ p = ((a * a) / 20 : Real)) :
  ((a * a) / 20 : Real) = 5 :=
by 
  sorry

end area_is_prime_number_l1_1478


namespace waiter_customers_before_lunch_l1_1300

theorem waiter_customers_before_lunch (X : ℕ) (A : X + 20 = 49) : X = 29 := by
  -- The proof is omitted based on the instructions
  sorry

end waiter_customers_before_lunch_l1_1300


namespace product_pricing_and_savings_l1_1754

theorem product_pricing_and_savings :
  ∃ (x y : ℝ),
    (6 * x + 3 * y = 600) ∧
    (40 * x + 30 * y = 5200) ∧
    x = 40 ∧
    y = 120 ∧
    (80 * x + 100 * y - (80 * 0.8 * x + 100 * 0.75 * y) = 3640) := 
by
  sorry

end product_pricing_and_savings_l1_1754


namespace trajectory_curve_point_F_exists_l1_1468

noncomputable def curve_C := { p : ℝ × ℝ | (p.1 - 1/2)^2 + (p.2 - 1/2)^2 = 4 }

theorem trajectory_curve (M : ℝ × ℝ) (p : ℝ × ℝ) (q : ℝ × ℝ) :
    M = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) → 
    p.1^2 + p.2^2 = 9 → 
    q.1^2 + q.2^2 = 9 →
    (p.1 - 1)^2 + (p.2 - 1)^2 > 0 → 
    (q.1 - 1)^2 + (q.2 - 1)^2 > 0 → 
    ((p.1 - 1) * (q.1 - 1) + (p.2 - 1) * (q.2 - 1) = 0) →
    (M.1 - 1/2)^2 + (M.2 - 1/2)^2 = 4 :=
sorry

theorem point_F_exists (E D : ℝ × ℝ) (F : ℝ × ℝ) (H : ℝ × ℝ) :
    E = (9/2, 1/2) → D = (1/2, 1/2) → F.2 = 1/2 → 
    (∃ t : ℝ, t ≠ 9/2 ∧ F.1 = t) →
    (H ∈ curve_C) →
    ((H.1 - 9/2)^2 + (H.2 - 1/2)^2) / ((H.1 - F.1)^2 + (H.2 - 1/2)^2) = 24 * (15 - 8 * H.1) / ((t^2 + 15/4) * (24)) :=
sorry

end trajectory_curve_point_F_exists_l1_1468


namespace sum_of_ages_l1_1959

theorem sum_of_ages (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4) 
  (h2 : rachel_age = 19) : rachel_age + leah_age = 34 :=
by
  -- Proof steps are omitted since we only need the statement
  sorry

end sum_of_ages_l1_1959


namespace race_time_l1_1314

theorem race_time (v_A v_B : ℝ) (t tB : ℝ) (h1 : 200 / v_A = t) (h2 : 144 / v_B = t) (h3 : 200 / v_B = t + 7) : t = 18 :=
by
  sorry

end race_time_l1_1314


namespace product_of_two_numbers_l1_1153

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end product_of_two_numbers_l1_1153


namespace correlated_relationships_l1_1769

-- Definitions for the conditions are arbitrary
-- In actual use cases, these would be replaced with real mathematical conditions
def great_teachers_produce_outstanding_students : Prop := sorry
def volume_of_sphere_with_radius : Prop := sorry
def apple_production_climate : Prop := sorry
def height_and_weight : Prop := sorry
def taxi_fare_distance_traveled : Prop := sorry
def crows_cawing_bad_omen : Prop := sorry

-- The final theorem statement
theorem correlated_relationships : 
  great_teachers_produce_outstanding_students ∧
  apple_production_climate ∧
  height_and_weight ∧
  ¬ volume_of_sphere_with_radius ∧ 
  ¬ taxi_fare_distance_traveled ∧ 
  ¬ crows_cawing_bad_omen :=
sorry

end correlated_relationships_l1_1769


namespace maciek_total_purchase_cost_l1_1645

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l1_1645


namespace find_m_range_l1_1404

def proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0) ∧ (1 > 0)

def proposition_q (m : ℝ) : Prop :=
  16 * (m - 2)^2 - 16 < 0

theorem find_m_range : {m : ℝ // proposition_p m ∧ proposition_q m} = {m : ℝ // 2 < m ∧ m < 3} :=
by
  sorry

end find_m_range_l1_1404


namespace find_function_range_of_a_l1_1804

variables (a b : ℝ) (f : ℝ → ℝ) 

-- Given: f(x) = ax + b where a ≠ 0 
--        f(2x + 1) = 4x + 1
-- Prove: f(x) = 2x - 1
theorem find_function (h1 : ∀ x, f (2 * x + 1) = 4 * x + 1) : 
  ∃ a b, a = 2 ∧ b = -1 ∧ ∀ x, f x = a * x + b :=
by sorry

-- Given: A = {x | a - 1 < x < 2a +1 }
--        B = {x | 1 < f(x) < 3 }
--        B ⊆ A
-- Prove: 1/2 ≤ a ≤ 2
theorem range_of_a (Hf : ∀ x, f x = 2 * x - 1) (Hsubset: ∀ x, 1 < f x ∧ f x < 3 → a - 1 < x ∧ x < 2 * a + 1) :
  1 / 2 ≤ a ∧ a ≤ 2 :=
by sorry

end find_function_range_of_a_l1_1804


namespace parabola_equation_through_origin_point_l1_1734

-- Define the conditions
def vertex_origin := (0, 0)
def point_on_parabola := (-2, 4)

-- Define what it means to be a standard equation of a parabola passing through a point
def standard_equation_passing_through (p : ℝ) (x y : ℝ) : Prop :=
  (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)

-- The theorem stating the conclusion
theorem parabola_equation_through_origin_point :
  ∃ p > 0, standard_equation_passing_through p (-2) 4 ∧
  (4^2 = -8 * (-2) ∨ (-2)^2 = 4) := 
sorry

end parabola_equation_through_origin_point_l1_1734


namespace range_of_m_l1_1257

-- Definitions for the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Prove that m ≥ 2 given the condition A ∪ B = A 
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 :=
by
  sorry

end range_of_m_l1_1257


namespace mass_percentage_Al_in_Al2O3_l1_1743

-- Define the atomic masses and formula unit
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

-- Define the statement for the mass percentage of Al in Al2O3
theorem mass_percentage_Al_in_Al2O3 : (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100 = 52.91 :=
by
  sorry -- Proof to be filled in

end mass_percentage_Al_in_Al2O3_l1_1743


namespace yolkino_to_palkino_distance_l1_1059

theorem yolkino_to_palkino_distance 
  (n : ℕ) 
  (digit_sum : ℕ → ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → digit_sum k + digit_sum (n - k) = 13) : 
  n = 49 := 
by 
  sorry

end yolkino_to_palkino_distance_l1_1059


namespace find_divisor_value_l1_1551

theorem find_divisor_value
  (D : ℕ) 
  (h1 : ∃ k : ℕ, 242 = k * D + 6)
  (h2 : ∃ l : ℕ, 698 = l * D + 13)
  (h3 : ∃ m : ℕ, 940 = m * D + 5) : 
  D = 14 :=
by
  sorry

end find_divisor_value_l1_1551


namespace temple_run_red_coins_l1_1082

variables (x y z : ℕ)

theorem temple_run_red_coins :
  x + y + z = 2800 →
  x + 3 * y + 5 * z = 7800 →
  z = y + 200 →
  y = 700 := 
by 
  intro h1 h2 h3
  sorry

end temple_run_red_coins_l1_1082


namespace problem1_problem2_problem3_problem4_l1_1680

-- Problem 1
theorem problem1 :
  -11 - (-8) + (-13) + 12 = -4 :=
  sorry

-- Problem 2
theorem problem2 :
  3 + 1 / 4 + (- (2 + 3 / 5)) + (5 + 3 / 4) - (8 + 2 / 5) = -2 :=
  sorry

-- Problem 3
theorem problem3 :
  -36 * (5 / 6 - 4 / 9 + 11 / 12) = -47 :=
  sorry

-- Problem 4
theorem problem4 :
  12 * (-1 / 6) + 27 / abs (3 ^ 2) + (-2) ^ 3 = -7 :=
  sorry

end problem1_problem2_problem3_problem4_l1_1680


namespace largest_piece_length_l1_1394

theorem largest_piece_length (v : ℝ) (hv : v + (3/2) * v + (9/4) * v = 95) : 
  (9/4) * v = 45 :=
by sorry

end largest_piece_length_l1_1394


namespace rectangle_side_divisible_by_4_l1_1436

theorem rectangle_side_divisible_by_4 (a b : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ a → i % 4 = 0)
  (h2 : ∀ j, 1 ≤ j ∧ j ≤ b → j % 4 = 0): 
  (a % 4 = 0) ∨ (b % 4 = 0) :=
sorry

end rectangle_side_divisible_by_4_l1_1436


namespace find_C_and_D_l1_1892

variables (C D : ℝ)

theorem find_C_and_D (h : 4 * C + 2 * D + 5 = 30) : C = 5.25 ∧ D = 2 :=
by
  sorry

end find_C_and_D_l1_1892


namespace max_value_of_z_l1_1223

theorem max_value_of_z (x y : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) :
  x^2 + y^2 ≤ 2 :=
by {
  sorry
}

end max_value_of_z_l1_1223


namespace unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l1_1897

-- Definitions of points and lines
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) -- Represented as ax + by = 0

-- Given conditions
variables {α : Type*} [Field α]
variables (P Q : Point α)
variables (L1 L2 : Line α) -- L1 and L2 are perpendicular

-- Proof problem statement
theorem unique_ellipse_through_points_with_perpendicular_axes (P Q : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
(P ≠ Q) → 
∃! (E : Set (Point α)), -- E represents the ellipse as a set of points
(∀ (p : Point α), p ∈ E → (p = P ∨ p = Q)) ∧ -- E passes through P and Q
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

theorem infinite_ellipses_when_points_coincide (P : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
∃ (E : Set (Point α)), -- E represents an ellipse
(∀ (p : Point α), p ∈ E → p = P) ∧ -- E passes through P
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

end unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l1_1897


namespace xiao_hua_correct_questions_l1_1052

-- Definitions of the problem conditions
def n : Nat := 20
def p_correct : Int := 5
def p_wrong : Int := -2
def score : Int := 65

-- Theorem statement to prove the number of correct questions
theorem xiao_hua_correct_questions : 
  ∃ k : Nat, k = ((n : Int) - ((n * p_correct - score) / (p_correct - p_wrong))) ∧ 
               k = 15 :=
by
  sorry

end xiao_hua_correct_questions_l1_1052


namespace lily_typing_speed_l1_1397

-- Define the conditions
def wordsTyped : ℕ := 255
def totalMinutes : ℕ := 19
def breakTime : ℕ := 2
def typingInterval : ℕ := 10
def effectiveMinutes : ℕ := totalMinutes - breakTime

-- Define the number of words typed in effective minutes
def wordsPerMinute (words : ℕ) (minutes : ℕ) : ℕ := words / minutes

-- Statement to be proven
theorem lily_typing_speed : wordsPerMinute wordsTyped effectiveMinutes = 15 :=
by
  -- proof goes here
  sorry

end lily_typing_speed_l1_1397


namespace fraction_addition_l1_1555

theorem fraction_addition :
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  a + b = 7 / 12 :=
by
  let a := (1 : ℚ) / 6
  let b := (5 : ℚ) / 12
  have : a + b = 7 / 12 := sorry
  exact this

end fraction_addition_l1_1555


namespace peter_spent_on_repairs_l1_1847

variable (C : ℝ)

def repairs_cost (C : ℝ) := 0.10 * C

def profit (C : ℝ) := 1.20 * C - C

theorem peter_spent_on_repairs :
  ∀ C, profit C = 1100 → repairs_cost C = 550 :=
by
  intro C
  sorry

end peter_spent_on_repairs_l1_1847


namespace find_b_l1_1294

variables (a b : ℕ)

theorem find_b
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 25 * 315 * b) :
  b = 7 :=
sorry

end find_b_l1_1294


namespace basketball_game_first_half_points_l1_1841

noncomputable def total_points_first_half
  (eagles_points : ℕ → ℕ) (lions_points : ℕ → ℕ) (common_ratio : ℕ) (common_difference : ℕ) : ℕ :=
  eagles_points 0 + eagles_points 1 + lions_points 0 + lions_points 1

theorem basketball_game_first_half_points 
  (eagles_points lions_points : ℕ → ℕ)
  (common_ratio : ℕ) (common_difference : ℕ)
  (h1 : eagles_points 0 = lions_points 0)
  (h2 : ∀ n, eagles_points (n + 1) = common_ratio * eagles_points n)
  (h3 : ∀ n, lions_points (n + 1) = lions_points n + common_difference)
  (h4 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 =
        lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 + 3)
  (h5 : eagles_points 0 + eagles_points 1 + eagles_points 2 + eagles_points 3 ≤ 120)
  (h6 : lions_points 0 + lions_points 1 + lions_points 2 + lions_points 3 ≤ 120) :
  total_points_first_half eagles_points lions_points common_ratio common_difference = 15 :=
sorry

end basketball_game_first_half_points_l1_1841


namespace range_of_m_l1_1647

open Set

variable (f : ℝ → ℝ) (m : ℝ)

theorem range_of_m (h1 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h2 : f (2 * m) > f (1 + m)) : m < 1 :=
by {
  -- The proof would go here.
  sorry
}

end range_of_m_l1_1647


namespace complement_union_l1_1834

open Set

variable (U M N : Set ℕ)

def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem complement_union (hU : U = {0, 1, 2, 3, 4, 5, 6})
                          (hM : M = {1, 3, 5})
                          (hN : N = {2, 4, 6}) :
  (complement_U U M) ∪ (complement_U U N) = {0, 1, 2, 3, 4, 5, 6} :=
by 
  sorry

end complement_union_l1_1834


namespace kenya_peanut_count_l1_1818

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the number of additional peanuts Kenya has more than Jose
def additional_peanuts : ℕ := 48

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := jose_peanuts + additional_peanuts

-- Theorem to prove the number of peanuts Kenya has
theorem kenya_peanut_count : kenya_peanuts = 133 := by
  sorry

end kenya_peanut_count_l1_1818


namespace largest_rhombus_diagonal_in_circle_l1_1324

theorem largest_rhombus_diagonal_in_circle (r : ℝ) (h : r = 10) : (2 * r = 20) :=
by
  sorry

end largest_rhombus_diagonal_in_circle_l1_1324


namespace y_mul_k_is_perfect_square_l1_1232

-- Defining y as given in the problem with its prime factorization
def y : Nat := 3^4 * (2^2)^5 * 5^6 * (2 * 3)^7 * 7^8 * (2^3)^9 * (3^2)^10

-- Since the question asks for an integer k (in this case 75) such that y * k is a perfect square
def k : Nat := 75

-- The statement that needs to be proved
theorem y_mul_k_is_perfect_square : ∃ n : Nat, (y * k) = n^2 := 
by
  sorry

end y_mul_k_is_perfect_square_l1_1232


namespace binomial_expansion_calculation_l1_1814

theorem binomial_expansion_calculation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end binomial_expansion_calculation_l1_1814


namespace translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l1_1673

def f_translation : ℝ → ℝ :=
  fun x => (x - 1)^2 - 2

def f_quad (a x : ℝ) : ℝ :=
  x^2 - 2*a*x - 1

theorem translated_quadratic :
  ∀ x, f_translation x = (x - 1)^2 - 2 :=
by
  intro x
  simp [f_translation]

theorem range_of_translated_quadratic :
  ∀ x, 0 ≤ x ∧ x ≤ 4 → -2 ≤ f_translation x ∧ f_translation x ≤ 7 :=
by
  sorry

theorem min_value_on_interval :
  ∀ a, 
    (a ≤ 0 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a x ≥ -1)) ∧
    (0 < a ∧ a < 2 → f_quad a a = -a^2 - 1) ∧
    (a ≥ 2 → ∀ x, 0 ≤ x ∧ x ≤ 2 → (f_quad a 2 = -4*a + 3)) :=
by
  sorry

end translated_quadratic_range_of_translated_quadratic_min_value_on_interval_l1_1673


namespace Dorottya_should_go_first_l1_1728

def probability_roll_1_or_2 : ℚ := 2 / 10

def probability_no_roll_1_or_2 : ℚ := 1 - probability_roll_1_or_2

variables {P_1 P_2 P_3 P_4 P_5 P_6 : ℚ}
  (hP1 : P_1 = probability_roll_1_or_2 * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP2 : P_2 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 1) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP3 : P_3 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 2) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP4 : P_4 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 3) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP5 : P_5 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 4) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))
  (hP6 : P_6 = probability_roll_1_or_2 * (probability_no_roll_1_or_2 ^ 5) * ∑' n, (probability_no_roll_1_or_2 ^ (6 * n)))

theorem Dorottya_should_go_first : P_1 > P_2 ∧ P_2 > P_3 ∧ P_3 > P_4 ∧ P_4 > P_5 ∧ P_5 > P_6 :=
by {
  -- Skipping actual proof steps
  sorry
}

end Dorottya_should_go_first_l1_1728


namespace max_value_of_a_l1_1032

theorem max_value_of_a (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a < 3 * b) (h2 : b < 2 * c) (h3 : c < 5 * d) (h4 : d < 150) : a ≤ 4460 :=
by
  sorry

end max_value_of_a_l1_1032


namespace sum_of_arithmetic_sequence_l1_1444

noncomputable def arithmetic_sequence_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
n * a_1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a_1 d : ℝ) (p q : ℕ) (h₁ : p ≠ q) (h₂ : arithmetic_sequence_sum a_1 d p = q) (h₃ : arithmetic_sequence_sum a_1 d q = p) : 
arithmetic_sequence_sum a_1 d (p + q) = - (p + q) := sorry

end sum_of_arithmetic_sequence_l1_1444


namespace equation_root_a_plus_b_l1_1139

theorem equation_root_a_plus_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b ≥ 0) 
(h_root : (∃ x : ℝ, x > 0 ∧ x^3 - x^2 + 18 * x - 320 = 0 ∧ x = Real.sqrt a - ↑b)) : 
a + b = 25 := by
  sorry

end equation_root_a_plus_b_l1_1139


namespace digital_root_8_pow_n_l1_1319

-- Define the conditions
def n : ℕ := 1989

-- Define the simplified problem
def digital_root (x : ℕ) : ℕ := if x % 9 = 0 then 9 else x % 9

-- Statement of the problem
theorem digital_root_8_pow_n : digital_root (8 ^ n) = 8 := by
  have mod_nine_eq : 8^n % 9 = 8 := by
    sorry
  simp [digital_root, mod_nine_eq]

end digital_root_8_pow_n_l1_1319


namespace loss_equates_to_balls_l1_1658

theorem loss_equates_to_balls
    (SP_20 : ℕ) (CP_1: ℕ) (Loss: ℕ) (x: ℕ)
    (h1 : SP_20 = 720)
    (h2 : CP_1 = 48)
    (h3 : Loss = (20 * CP_1 - SP_20))
    (h4 : Loss = x * CP_1) :
    x = 5 :=
by
  sorry

end loss_equates_to_balls_l1_1658


namespace find_y_l1_1537

theorem find_y (x y : ℕ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : y = 1 :=
sorry

end find_y_l1_1537


namespace cherries_in_mix_l1_1490

theorem cherries_in_mix (total_fruit : ℕ) (blueberries : ℕ) (raspberries : ℕ) (cherries : ℕ) 
  (H1 : total_fruit = 300)
  (H2: raspberries = 3 * blueberries)
  (H3: cherries = 5 * blueberries)
  (H4: total_fruit = blueberries + raspberries + cherries) : cherries = 167 :=
by
  sorry

end cherries_in_mix_l1_1490


namespace batsman_new_average_l1_1118

variable (A : ℝ) -- Assume that A is the average before the 17th inning
variable (score : ℝ) -- The score in the 17th inning
variable (new_average : ℝ) -- The new average after the 17th inning

-- The conditions
axiom H1 : score = 85
axiom H2 : new_average = A + 3

-- The statement to prove
theorem batsman_new_average : 
    new_average = 37 :=
by 
  sorry

end batsman_new_average_l1_1118


namespace fantasy_gala_handshakes_l1_1137

theorem fantasy_gala_handshakes
    (gremlins imps : ℕ)
    (gremlin_handshakes : ℕ)
    (imp_handshakes : ℕ)
    (imp_gremlin_handshakes : ℕ)
    (total_handshakes : ℕ)
    (h1 : gremlins = 30)
    (h2 : imps = 20)
    (h3 : gremlin_handshakes = (30 * 29) / 2)
    (h4 : imp_handshakes = (20 * 5) / 2)
    (h5 : imp_gremlin_handshakes = 20 * 30)
    (h6 : total_handshakes = gremlin_handshakes + imp_handshakes + imp_gremlin_handshakes) :
    total_handshakes = 1085 := by
    sorry

end fantasy_gala_handshakes_l1_1137


namespace mult_mod_7_zero_l1_1128

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l1_1128


namespace find_a_b_find_c_range_l1_1040

noncomputable def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem find_a_b (a b c : ℝ) (extreme_x1 extreme_x2 : ℝ) (h1 : extreme_x1 = 1) (h2 : extreme_x2 = 2) 
  (h3 : (deriv (f a b c) 1) = 0) (h4 : (deriv (f a b c) 2) = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem find_c_range (c : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f (-3) 4 c x < c^2) : 
  c ∈ Set.Iio (-1) ∪ Set.Ioi 9 :=
by sorry

end find_a_b_find_c_range_l1_1040


namespace austin_pairs_of_shoes_l1_1974

theorem austin_pairs_of_shoes (S : ℕ) :
  0.45 * (S : ℝ) + 11 = S → S / 2 = 10 :=
by
  sorry

end austin_pairs_of_shoes_l1_1974


namespace x_cubed_plus_y_cubed_l1_1541

theorem x_cubed_plus_y_cubed:
  ∀ (x y : ℝ), (x * (x ^ 4 + y ^ 4) = y ^ 5) → (x ^ 2 * (x + y) ≠ y ^ 3) → (x ^ 3 + y ^ 3 = 1) :=
by
  intros x y h1 h2
  sorry

end x_cubed_plus_y_cubed_l1_1541


namespace sufficient_but_not_necessary_l1_1842

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x < 1 / 2) : x > 2 ∨ x < 0 :=
by
  sorry

end sufficient_but_not_necessary_l1_1842


namespace find_x_when_y_is_10_l1_1945

-- Definitions of inverse proportionality and initial conditions
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Given constants
def k : ℝ := 160
def x_initial : ℝ := 40
def y_initial : ℝ := 4

-- Theorem statement to prove the value of x when y = 10
theorem find_x_when_y_is_10 (h : inversely_proportional x_initial y_initial k) : 
  ∃ (x : ℝ), inversely_proportional x 10 k :=
sorry

end find_x_when_y_is_10_l1_1945


namespace smallest_cut_length_l1_1125

theorem smallest_cut_length (x : ℕ) (h₁ : 9 ≥ x) (h₂ : 12 ≥ x) (h₃ : 15 ≥ x)
  (h₄ : x ≥ 6) (h₅ : x ≥ 12) (h₆ : x ≥ 18) : x = 6 :=
by
  sorry

end smallest_cut_length_l1_1125


namespace multiply_72517_9999_l1_1074

theorem multiply_72517_9999 : 72517 * 9999 = 725097483 :=
by
  sorry

end multiply_72517_9999_l1_1074


namespace factorize_problem_1_factorize_problem_2_l1_1388

theorem factorize_problem_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := 
by sorry

theorem factorize_problem_2 (x y : ℝ) : 2 * x^3 - 12 * x^2 * y + 18 * x * y^2 = 2 * x * (x - 3 * y)^2 :=
by sorry

end factorize_problem_1_factorize_problem_2_l1_1388


namespace simultaneous_in_Quadrant_I_l1_1602

def in_Quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem simultaneous_in_Quadrant_I (c x y : ℝ) : 
  (2 * x - y = 5) ∧ (c * x + y = 4) ↔ in_Quadrant_I x y ∧ (-2 < c ∧ c < 8 / 5) :=
sorry

end simultaneous_in_Quadrant_I_l1_1602


namespace samantha_probability_l1_1424

noncomputable def probability_of_selecting_yellow_apples 
  (total_apples : ℕ) (yellow_apples : ℕ) (selection_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_apples selection_size
  let yellow_ways := Nat.choose yellow_apples selection_size
  yellow_ways / total_ways

theorem samantha_probability : 
  probability_of_selecting_yellow_apples 10 5 3 = 1 / 12 := 
by 
  sorry

end samantha_probability_l1_1424


namespace min_value_of_2x_plus_y_l1_1322

theorem min_value_of_2x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 8 / y = 2) : 2 * x + y ≥ 7 :=
sorry

end min_value_of_2x_plus_y_l1_1322


namespace golden_ratio_problem_l1_1362

noncomputable def m := 2 * Real.sin (Real.pi * 18 / 180)
noncomputable def n := 4 - m^2
noncomputable def target_expression := m * Real.sqrt n / (2 * (Real.cos (Real.pi * 27 / 180))^2 - 1)

theorem golden_ratio_problem :
  target_expression = 2 :=
by
  -- Proof will be placed here
  sorry

end golden_ratio_problem_l1_1362


namespace problem_statement_l1_1571

variables {a b c p q r : ℝ}

-- Given conditions
axiom h1 : 19 * p + b * q + c * r = 0
axiom h2 : a * p + 29 * q + c * r = 0
axiom h3 : a * p + b * q + 56 * r = 0
axiom h4 : a ≠ 19
axiom h5 : p ≠ 0

-- Statement to prove
theorem problem_statement : 
  (a / (a - 19)) + (b / (b - 29)) + (c / (c - 56)) = 1 :=
sorry

end problem_statement_l1_1571


namespace Amy_balloons_l1_1024

-- Defining the conditions
def James_balloons : ℕ := 1222
def more_balloons : ℕ := 208

-- Defining Amy's balloons as a proof goal
theorem Amy_balloons : ∀ (Amy_balloons : ℕ), James_balloons - more_balloons = Amy_balloons → Amy_balloons = 1014 :=
by
  intros Amy_balloons h
  sorry

end Amy_balloons_l1_1024


namespace jo_reading_time_l1_1472

structure Book :=
  (totalPages : Nat)
  (currentPage : Nat)
  (pageOneHourAgo : Nat)

def readingTime (b : Book) : Nat :=
  let pagesRead := b.currentPage - b.pageOneHourAgo
  let pagesLeft := b.totalPages - b.currentPage
  pagesLeft / pagesRead

theorem jo_reading_time :
  ∀ (b : Book), b.totalPages = 210 → b.currentPage = 90 → b.pageOneHourAgo = 60 → readingTime b = 4 :=
by
  intro b h1 h2 h3
  sorry

end jo_reading_time_l1_1472


namespace area_under_curve_l1_1451

theorem area_under_curve : 
  ∫ x in (1/2 : ℝ)..(2 : ℝ), (1 / x) = 2 * Real.log 2 := by
  sorry

end area_under_curve_l1_1451


namespace emily_coloring_books_l1_1981

variable (initial_books : ℕ) (given_away : ℕ) (total_books : ℕ) (bought_books : ℕ)

theorem emily_coloring_books :
  initial_books = 7 →
  given_away = 2 →
  total_books = 19 →
  initial_books - given_away + bought_books = total_books →
  bought_books = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end emily_coloring_books_l1_1981


namespace assignment_methods_l1_1771

theorem assignment_methods : 
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  (doctors * (nurses.choose nurses_per_school)) = 12 := by
  sorry

end assignment_methods_l1_1771


namespace color_triplet_exists_l1_1544

theorem color_triplet_exists (color : ℕ → Prop) :
  (∀ n, color n ∨ ¬ color n) → ∃ x y z : ℕ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ color x = color y ∧ color y = color z ∧ x * y = z ^ 2 :=
by
  sorry

end color_triplet_exists_l1_1544


namespace find_f_10_l1_1221

def f : ℕ → ℚ := sorry
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = f x / (1 + f x)
axiom f_initial : f 1 = 1

theorem find_f_10 : f 10 = 1 / 10 :=
by
  sorry

end find_f_10_l1_1221


namespace volume_of_region_l1_1616

theorem volume_of_region (r1 r2 : ℝ) (h : r1 = 5) (h2 : r2 = 8) : 
  let V_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
  let V_cylinder (r : ℝ) := Real.pi * r^2 * r
  (V_sphere r2) - (V_sphere r1) - (V_cylinder r1) = 391 * Real.pi :=
by
  -- Placeholder proof
  sorry

end volume_of_region_l1_1616


namespace composite_p_squared_plus_36_l1_1323

theorem composite_p_squared_plus_36 (p : ℕ) (h_prime : Prime p) : 
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ (k * m = p^2 + 36) :=
by {
  sorry
}

end composite_p_squared_plus_36_l1_1323


namespace midpoint_coord_sum_l1_1927

theorem midpoint_coord_sum (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = -2) (hx2 : x2 = -4) (hy2 : y2 = 8)
: (x1 + x2) / 2 + (y1 + y2) / 2 = 6 :=
by
  rw [hx1, hx2, hy1, hy2]
  /-
  Have (10 + (-4)) / 2 + (-2 + 8) / 2 = (6 / 2) + (6 / 2)
  Prove that (6 / 2) + (6 / 2) = 6
  -/
  sorry

end midpoint_coord_sum_l1_1927


namespace find_a_given_difference_l1_1477

theorem find_a_given_difference (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : |a - a^2| = 6) : a = 3 :=
sorry

end find_a_given_difference_l1_1477


namespace gumballs_ensure_four_same_color_l1_1581

-- Define the total number of gumballs in each color
def red_gumballs : ℕ := 10
def white_gumballs : ℕ := 9
def blue_gumballs : ℕ := 8
def green_gumballs : ℕ := 7

-- Define the minimum number of gumballs to ensure four of the same color
def min_gumballs_to_ensure_four_same_color : ℕ := 13

-- Prove that the minimum number of gumballs to ensure four of the same color is 13
theorem gumballs_ensure_four_same_color (n : ℕ) 
  (h₁ : red_gumballs ≥ 3)
  (h₂ : white_gumballs ≥ 3)
  (h₃ : blue_gumballs ≥ 3)
  (h₄ : green_gumballs ≥ 3)
  : n ≥ min_gumballs_to_ensure_four_same_color := 
sorry

end gumballs_ensure_four_same_color_l1_1581


namespace solve_for_x_l1_1035

theorem solve_for_x (x : ℝ) (h :  9 / x^2 = x / 25) : x = 5 :=
by 
  sorry

end solve_for_x_l1_1035


namespace smallest_angle_of_trapezoid_l1_1189

theorem smallest_angle_of_trapezoid (a d : ℝ) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) → 
  (a + 3 * d = 150) → 
  a = 15 :=
by
  sorry

end smallest_angle_of_trapezoid_l1_1189


namespace day_of_week_dec_26_l1_1089

theorem day_of_week_dec_26 (nov_26_is_thu : true) : true :=
sorry

end day_of_week_dec_26_l1_1089


namespace difference_in_nickels_is_correct_l1_1952

variable (q : ℤ)

def charles_quarters : ℤ := 7 * q + 2
def richard_quarters : ℤ := 3 * q + 8

theorem difference_in_nickels_is_correct :
  5 * (charles_quarters - richard_quarters) = 20 * q - 30 :=
by
  sorry

end difference_in_nickels_is_correct_l1_1952


namespace factor_polynomial_sum_l1_1687

theorem factor_polynomial_sum (P Q : ℤ) :
  (∀ x : ℂ, (x^2 + 4*x + 5) ∣ (x^4 + P*x^2 + Q)) → P + Q = 19 :=
by
  intro h
  sorry

end factor_polynomial_sum_l1_1687


namespace masha_can_pay_exactly_with_11_ruble_bills_l1_1597

theorem masha_can_pay_exactly_with_11_ruble_bills (m n k p : ℕ) 
  (h1 : 3 * m + 4 * n + 5 * k = 11 * p) : 
  ∃ q : ℕ, 9 * m + n + 4 * k = 11 * q := 
by {
  sorry
}

end masha_can_pay_exactly_with_11_ruble_bills_l1_1597


namespace arun_weight_average_l1_1375

theorem arun_weight_average :
  ∀ (w : ℝ), (w > 61 ∧ w < 72) ∧ (w > 60 ∧ w < 70) ∧ (w ≤ 64) →
  (w = 62 ∨ w = 63) →
  (62 + 63) / 2 = 62.5 :=
by
  intros w h1 h2
  sorry

end arun_weight_average_l1_1375


namespace cookie_cost_l1_1853

theorem cookie_cost
  (classes3 : ℕ) (students_per_class3 : ℕ)
  (classes4 : ℕ) (students_per_class4 : ℕ)
  (classes5 : ℕ) (students_per_class5 : ℕ)
  (hamburger_cost : ℝ) (carrot_cost : ℝ) (total_lunch_cost : ℝ) (cookie_cost : ℝ)
  (h1 : classes3 = 5) (h2 : students_per_class3 = 30)
  (h3 : classes4 = 4) (h4 : students_per_class4 = 28)
  (h5 : classes5 = 4) (h6 : students_per_class5 = 27)
  (h7 : hamburger_cost = 2.10) (h8 : carrot_cost = 0.50)
  (h9 : total_lunch_cost = 1036):
  ((classes3 * students_per_class3) + (classes4 * students_per_class4) + (classes5 * students_per_class5)) * (cookie_cost + hamburger_cost + carrot_cost) = total_lunch_cost → 
  cookie_cost = 0.20 := 
by 
  sorry

end cookie_cost_l1_1853


namespace new_area_after_increasing_length_and_width_l1_1175

theorem new_area_after_increasing_length_and_width
  (L W : ℝ)
  (hA : L * W = 450)
  (hL' : 1.2 * L = L')
  (hW' : 1.3 * W = W') :
  (1.2 * L) * (1.3 * W) = 702 :=
by sorry

end new_area_after_increasing_length_and_width_l1_1175


namespace smallest_points_to_guarantee_victory_l1_1519

noncomputable def pointsForWinning : ℕ := 5
noncomputable def pointsForSecond : ℕ := 3
noncomputable def pointsForThird : ℕ := 1

theorem smallest_points_to_guarantee_victory :
  ∀ (student_points : ℕ),
  (exists (x y z : ℕ), (x = pointsForWinning ∨ x = pointsForSecond ∨ x = pointsForThird) ∧
                         (y = pointsForWinning ∨ y = pointsForSecond ∨ y = pointsForThird) ∧
                         (z = pointsForWinning ∨ z = pointsForSecond ∨ z = pointsForThird) ∧
                         student_points = x + y + z) →
  (∃ (victory_points : ℕ), victory_points = 13) →
  (∀ other_points : ℕ, other_points < victory_points) :=
sorry

end smallest_points_to_guarantee_victory_l1_1519


namespace find_a_b_sum_l1_1447

-- Definitions for the conditions
def equation1 (a : ℝ) : Prop := 3 = (1 / 3) * 6 + a
def equation2 (b : ℝ) : Prop := 6 = (1 / 3) * 3 + b

theorem find_a_b_sum : 
  ∃ (a b : ℝ), equation1 a ∧ equation2 b ∧ (a + b = 6) :=
sorry

end find_a_b_sum_l1_1447


namespace smallest_class_number_l1_1950

theorem smallest_class_number (sum_classes : ℕ) (n_classes interval number_of_classes : ℕ) 
                              (h_sum : sum_classes = 87) (h_n_classes : n_classes = 30) 
                              (h_interval : interval = 5) (h_number_of_classes : number_of_classes = 6) : 
                              ∃ x, x + (interval + x) + (2 * interval + x) + (3 * interval + x) 
                              + (4 * interval + x) + (5 * interval + x) = sum_classes ∧ x = 2 :=
by {
  use 2,
  sorry
}

end smallest_class_number_l1_1950


namespace f_lg_equality_l1_1992

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_lg_equality : f (Real.log 2) + f (Real.log (1 / 2)) = 2 := sorry

end f_lg_equality_l1_1992


namespace find_c_for_given_radius_l1_1210

theorem find_c_for_given_radius (c : ℝ) : (∃ x y : ℝ, (x^2 - 2 * x + y^2 + 6 * y + c = 0) ∧ ((x - 1)^2 + (y + 3)^2 = 25)) → c = -15 :=
by
  sorry

end find_c_for_given_radius_l1_1210


namespace inequality_a_b_c_l1_1515

noncomputable def a := Real.log (Real.pi / 3)
noncomputable def b := Real.log (Real.exp 1 / 3)
noncomputable def c := Real.exp (0.5)

theorem inequality_a_b_c : c > a ∧ a > b := by
  sorry

end inequality_a_b_c_l1_1515


namespace baby_frogs_on_rock_l1_1061

theorem baby_frogs_on_rock (f_l f_L f_T : ℕ) (h1 : f_l = 5) (h2 : f_L = 3) (h3 : f_T = 32) : 
  f_T - (f_l + f_L) = 24 :=
by sorry

end baby_frogs_on_rock_l1_1061


namespace maria_threw_out_carrots_l1_1109

theorem maria_threw_out_carrots (initially_picked: ℕ) (picked_next_day: ℕ) (total_now: ℕ) (carrots_thrown_out: ℕ) :
  initially_picked = 48 → 
  picked_next_day = 15 → 
  total_now = 52 → 
  (initially_picked + picked_next_day - total_now = carrots_thrown_out) → 
  carrots_thrown_out = 11 :=
by
  intros
  sorry

end maria_threw_out_carrots_l1_1109


namespace B_share_is_2400_l1_1972

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l1_1972


namespace problem_statement_l1_1499

def scientific_notation_correct (x : ℝ) : Prop :=
  x = 5.642 * 10 ^ 5

theorem problem_statement : scientific_notation_correct 564200 :=
by
  sorry

end problem_statement_l1_1499


namespace gcd_solution_l1_1843

theorem gcd_solution {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := 
sorry

end gcd_solution_l1_1843


namespace selling_price_of_radio_l1_1316

theorem selling_price_of_radio
  (cost_price : ℝ)
  (loss_percentage : ℝ) :
  loss_percentage = 13 → cost_price = 1500 → 
  (cost_price - (loss_percentage / 100) * cost_price) = 1305 :=
by
  intros h1 h2
  sorry

end selling_price_of_radio_l1_1316


namespace fence_calculation_l1_1008

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def fence_needed : ℕ := 2 * length + 2 * width

theorem fence_calculation : fence_needed = 22 := by
  sorry

end fence_calculation_l1_1008


namespace infinite_solutions_distinct_natural_numbers_l1_1321

theorem infinite_solutions_distinct_natural_numbers :
  ∃ (x y z : ℕ), (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (x ^ 2015 + y ^ 2015 = z ^ 2016) :=
by
  sorry

end infinite_solutions_distinct_natural_numbers_l1_1321


namespace complex_magnitude_of_3_minus_4i_l1_1563

open Complex

theorem complex_magnitude_of_3_minus_4i : Complex.abs ⟨3, -4⟩ = 5 := sorry

end complex_magnitude_of_3_minus_4i_l1_1563


namespace find_equation_of_ellipse_C_l1_1822

def equation_of_ellipse_C (a b : ℝ) : Prop :=
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_equation_of_ellipse_C :
  ∀ (a b : ℝ), (a = 2) → (b = 1) →
  (equation_of_ellipse_C a b) →
  equation_of_ellipse_C 2 1 :=
by
  intros a b ha hb h
  sorry

end find_equation_of_ellipse_C_l1_1822


namespace sum_of_prime_factors_143_l1_1840

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l1_1840


namespace math_students_count_l1_1330

noncomputable def students_in_math (total_students history_students english_students all_three_classes two_classes : ℕ) : ℕ :=
total_students - history_students - english_students + (two_classes - all_three_classes)

theorem math_students_count :
  students_in_math 68 21 34 3 7 = 14 :=
by
  sorry

end math_students_count_l1_1330


namespace volume_PQRS_is_48_39_cm3_l1_1479

noncomputable def area_of_triangle (a h : ℝ) : ℝ := 0.5 * a * h

noncomputable def volume_of_tetrahedron (base_area height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def height_from_area (area base : ℝ) : ℝ := (2 * area) / base

noncomputable def volume_of_tetrahedron_PQRS : ℝ :=
  let PQ := 5
  let area_PQR := 18
  let area_PQS := 16
  let angle_PQ := 45
  let h_PQR := height_from_area area_PQR PQ
  let h_PQS := height_from_area area_PQS PQ
  let h := h_PQS * (Real.sin (angle_PQ * Real.pi / 180))
  volume_of_tetrahedron area_PQR h

theorem volume_PQRS_is_48_39_cm3 : volume_of_tetrahedron_PQRS = 48.39 := by
  sorry

end volume_PQRS_is_48_39_cm3_l1_1479


namespace sin_960_eq_sqrt3_over_2_neg_l1_1145

-- Conditions
axiom sine_periodic : ∀ θ, Real.sin (θ + 360 * Real.pi / 180) = Real.sin θ

-- Theorem to prove
theorem sin_960_eq_sqrt3_over_2_neg : Real.sin (960 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  -- skipping the proof
  sorry

end sin_960_eq_sqrt3_over_2_neg_l1_1145


namespace range_of_b_for_local_minimum_l1_1613

variable {x : ℝ}
variable (b : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ :=
  x^3 - 6 * b * x + 3 * b

def f' (x : ℝ) (b : ℝ) : ℝ :=
  3 * x^2 - 6 * b

theorem range_of_b_for_local_minimum
  (h1 : f' 0 b < 0)
  (h2 : f' 1 b > 0) :
  0 < b ∧ b < 1 / 2 :=
by
  sorry

end range_of_b_for_local_minimum_l1_1613


namespace gcd_three_numbers_l1_1286

def a : ℕ := 13680
def b : ℕ := 20400
def c : ℕ := 47600

theorem gcd_three_numbers (a b c : ℕ) : Nat.gcd (Nat.gcd a b) c = 80 :=
by
  sorry

end gcd_three_numbers_l1_1286


namespace complex_root_product_value_l1_1069

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l1_1069


namespace average_marbles_of_other_colors_l1_1550

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l1_1550


namespace Liza_rent_l1_1179

theorem Liza_rent :
  (800 - R + 1500 - 117 - 100 - 70 = 1563) -> R = 450 :=
by
  intros h
  sorry

end Liza_rent_l1_1179


namespace inequality_5positives_l1_1289

variable {x1 x2 x3 x4 x5 : ℝ}

theorem inequality_5positives (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) :
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end inequality_5positives_l1_1289


namespace rods_in_one_mile_l1_1915

-- Definitions of the conditions
def mile_to_furlong := 10
def furlong_to_rod := 50

-- Theorem statement corresponding to the proof problem
theorem rods_in_one_mile : mile_to_furlong * furlong_to_rod = 500 := 
by sorry

end rods_in_one_mile_l1_1915


namespace marbles_in_jar_is_144_l1_1292

noncomputable def marbleCount (M : ℕ) : Prop :=
  M / 16 - M / 18 = 1

theorem marbles_in_jar_is_144 : ∃ M : ℕ, marbleCount M ∧ M = 144 :=
by
  use 144
  unfold marbleCount
  sorry

end marbles_in_jar_is_144_l1_1292


namespace mass_ratio_speed_ratio_l1_1164

variable {m1 m2 : ℝ} -- masses of the two balls
variable {V0 V : ℝ} -- velocities before and after collision
variable (h1 : V = 4 * V0) -- speed of m2 is four times that of m1 after collision

theorem mass_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                   (h3 : m1 * V0 = m1 * V + 4 * m2 * V) :
  m2 / m1 = 1 / 2 := sorry

theorem speed_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                    (h3 : m1 * V0 = m1 * V + 4 * m2 * V)
                    (h4 : m2 / m1 = 1 / 2) :
  V0 / V = 3 := sorry

end mass_ratio_speed_ratio_l1_1164


namespace total_distance_traveled_l1_1809

variable (vm vr t d_up d_down : ℝ)
variable (H_river_speed : vr = 3)
variable (H_row_speed : vm = 6)
variable (H_time : t = 1)

theorem total_distance_traveled (H_upstream : d_up = vm - vr) 
                                (H_downstream : d_down = vm + vr) 
                                (total_time : d_up / (vm - vr) + d_down / (vm + vr) = t) : 
                                2 * (d_up + d_down) = 4.5 := 
                                by
  sorry

end total_distance_traveled_l1_1809


namespace jerome_contacts_total_l1_1778

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l1_1778


namespace average_income_correct_l1_1929

def incomes : List ℕ := [250, 400, 750, 400, 500]

noncomputable def average : ℕ := (incomes.sum) / incomes.length

theorem average_income_correct : average = 460 :=
by 
  sorry

end average_income_correct_l1_1929


namespace digit_problem_l1_1900

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end digit_problem_l1_1900


namespace min_value_of_box_l1_1403

theorem min_value_of_box (a b : ℤ) (h_ab : a * b = 30) : 
  ∃ (m : ℤ), m = 61 ∧ (∀ (c : ℤ), a * b = 30 → a^2 + b^2 = c → c ≥ m) := 
sorry

end min_value_of_box_l1_1403


namespace root_of_quadratic_eq_l1_1010

theorem root_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, (x₁ = 0 ∧ x₂ = 2) ∧ ∀ x : ℝ, x^2 - 2 * x = 0 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end root_of_quadratic_eq_l1_1010


namespace area_of_region_l1_1691

theorem area_of_region : 
  (∃ x y : ℝ, |5 * x - 10| + |4 * y + 20| ≤ 10) →
  ∃ area : ℝ, 
  area = 10 :=
sorry

end area_of_region_l1_1691


namespace range_of_a_l1_1604

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l1_1604


namespace birdseed_weekly_consumption_l1_1415

def parakeets := 3
def parakeet_consumption := 2
def parrots := 2
def parrot_consumption := 14
def finches := 4
def finch_consumption := parakeet_consumption / 2
def canaries := 5
def canary_consumption := 3
def african_grey_parrots := 2
def african_grey_parrot_consumption := 18
def toucans := 3
def toucan_consumption := 25

noncomputable def daily_consumption := 
  parakeets * parakeet_consumption +
  parrots * parrot_consumption +
  finches * finch_consumption +
  canaries * canary_consumption +
  african_grey_parrots * african_grey_parrot_consumption +
  toucans * toucan_consumption

noncomputable def weekly_consumption := 7 * daily_consumption

theorem birdseed_weekly_consumption : weekly_consumption = 1148 := by
  sorry

end birdseed_weekly_consumption_l1_1415


namespace determine_m_to_satisfy_conditions_l1_1887

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 5) * x^(m - 1)

theorem determine_m_to_satisfy_conditions : 
  ∃ (m : ℝ), (m = 3) ∧ ∀ (x : ℝ), (0 < x → (m^2 - m - 5 > 0) ∧ (m - 1 > 0)) :=
by
  sorry

end determine_m_to_satisfy_conditions_l1_1887


namespace remainder_mod_5_l1_1621

theorem remainder_mod_5 :
  let a := 1492
  let b := 1776
  let c := 1812
  let d := 1996
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end remainder_mod_5_l1_1621


namespace line_equation_exists_l1_1810

theorem line_equation_exists 
  (a b : ℝ) 
  (ha_pos: a > 0)
  (hb_pos: b > 0)
  (h_area: 1 / 2 * a * b = 2) 
  (h_diff: a - b = 3 ∨ b - a = 3) : 
  (∀ x y : ℝ, (x + 4 * y = 4 ∧ (x / a + y / b = 1)) ∨ (4 * x + y = 4 ∧ (x / a + y / b = 1))) :=
sorry

end line_equation_exists_l1_1810


namespace max_number_of_9_letter_palindromes_l1_1042

theorem max_number_of_9_letter_palindromes : 26^5 = 11881376 :=
by sorry

end max_number_of_9_letter_palindromes_l1_1042


namespace slope_of_asymptotes_is_one_l1_1873

-- Given definitions and axioms
variables (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (A1 : ℝ × ℝ := (-a, 0))
  (A2 : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (c, b^2 / a))
  (C : ℝ × ℝ := (c, -b^2 / a))
  (Perp : (b^2 / a) / (c + a) * -(b^2 / a) / (c - a) = -1)

-- Proof goal
theorem slope_of_asymptotes_is_one : a = b → (∀ m : ℝ, m = (b / a) ∨ m = -(b / a)) ↔ ∀ m : ℝ, m = 1 ∨ m = -1 :=
by
  sorry

end slope_of_asymptotes_is_one_l1_1873


namespace negation_is_correct_l1_1284

-- Define the original proposition as a predicate on real numbers.
def original_prop : Prop := ∀ x : ℝ, 4*x^2 - 3*x + 2 < 0

-- State the negation of the original proposition
def negation_of_original_prop : Prop := ∃ x : ℝ, 4*x^2 - 3*x + 2 ≥ 0

-- The theorem to prove the correctness of the negation of the original proposition
theorem negation_is_correct : ¬original_prop ↔ negation_of_original_prop := by
  sorry

end negation_is_correct_l1_1284


namespace derivative_of_f_l1_1452

noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * (Real.cos x + Real.sin x)

theorem derivative_of_f (x : ℝ) : deriv f x = -2 * Real.exp (-x) * Real.sin x :=
by sorry

end derivative_of_f_l1_1452


namespace expand_expression_l1_1788

theorem expand_expression (x y : ℝ) : 
  (16 * x + 18 - 7 * y) * (3 * x) = 48 * x^2 + 54 * x - 21 * x * y :=
by
  sorry

end expand_expression_l1_1788


namespace hounds_score_points_l1_1070

theorem hounds_score_points (x y : ℕ) (h_total : x + y = 82) (h_margin : x - y = 18) : y = 32 :=
sorry

end hounds_score_points_l1_1070


namespace number_of_best_friends_l1_1565

-- Constants and conditions
def initial_tickets : ℕ := 37
def tickets_per_friend : ℕ := 5
def tickets_left : ℕ := 2

-- Problem statement
theorem number_of_best_friends : (initial_tickets - tickets_left) / tickets_per_friend = 7 :=
by
  sorry

end number_of_best_friends_l1_1565


namespace sequence_periodic_l1_1312

theorem sequence_periodic (a : ℕ → ℕ) (h : ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10) :
  ∃ n₀, ∀ k, a (n₀ + k) = a (n₀ + k + 4) :=
by {
  sorry
}

end sequence_periodic_l1_1312


namespace poly_coeff_sum_l1_1504

theorem poly_coeff_sum :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ,
  (∀ x : ℤ, ((x^2 + 1) * (x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 + a_11 * x^11))
  ∧ a_0 = -512) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510) :=
by
  sorry

end poly_coeff_sum_l1_1504


namespace equivalent_expression_l1_1933

theorem equivalent_expression :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) =
  5^128 - 2^128 := by
  sorry

end equivalent_expression_l1_1933


namespace chess_tournament_participants_l1_1228

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 378) : n = 28 :=
sorry

end chess_tournament_participants_l1_1228


namespace calculate_uphill_distance_l1_1731

noncomputable def uphill_speed : ℝ := 30
noncomputable def downhill_speed : ℝ := 40
noncomputable def downhill_distance : ℝ := 50
noncomputable def average_speed : ℝ := 32.73

theorem calculate_uphill_distance : ∃ d : ℝ, d = 99.86 ∧ 
  32.73 = (d + downhill_distance) / (d / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end calculate_uphill_distance_l1_1731


namespace bob_has_17_pennies_l1_1229

-- Definitions based on the problem conditions
variable (a b : ℕ)
def condition1 : Prop := b + 1 = 4 * (a - 1)
def condition2 : Prop := b - 2 = 2 * (a + 2)

-- The main statement to be proven
theorem bob_has_17_pennies (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 17 :=
by
  sorry

end bob_has_17_pennies_l1_1229


namespace stratified_sampling_11th_grade_representatives_l1_1596

theorem stratified_sampling_11th_grade_representatives 
  (students_10th : ℕ)
  (students_11th : ℕ)
  (students_12th : ℕ)
  (total_rep : ℕ)
  (total_students : students_10th + students_11th + students_12th = 5000)
  (Students_10th : students_10th = 2500)
  (Students_11th : students_11th = 1500)
  (Students_12th : students_12th = 1000)
  (Total_rep : total_rep = 30) : 
  (9 : ℕ) = (3 : ℚ) / (10 : ℚ) * (30 : ℕ) :=
sorry

end stratified_sampling_11th_grade_representatives_l1_1596


namespace smallest_positive_integer_between_101_and_200_l1_1218

theorem smallest_positive_integer_between_101_and_200 :
  ∃ n : ℕ, n > 1 ∧ n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1 ∧ 101 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_positive_integer_between_101_and_200_l1_1218


namespace seeds_sum_l1_1488

def Bom_seeds : ℕ := 300

def Gwi_seeds : ℕ := Bom_seeds + 40

def Yeon_seeds : ℕ := 3 * Gwi_seeds

def total_seeds : ℕ := Bom_seeds + Gwi_seeds + Yeon_seeds

theorem seeds_sum : total_seeds = 1660 := by
  sorry

end seeds_sum_l1_1488


namespace ball_hits_ground_at_l1_1348

variable (t : ℚ) 

def height_eqn (t : ℚ) : ℚ :=
  -16 * t^2 + 30 * t + 50

theorem ball_hits_ground_at :
  (height_eqn t = 0) -> t = 47 / 16 :=
by
  sorry

end ball_hits_ground_at_l1_1348


namespace full_size_mustang_length_l1_1793

theorem full_size_mustang_length 
  (smallest_model_length : ℕ)
  (mid_size_factor : ℕ)
  (full_size_factor : ℕ)
  (h1 : smallest_model_length = 12)
  (h2 : mid_size_factor = 2)
  (h3 : full_size_factor = 10) :
  (smallest_model_length * mid_size_factor) * full_size_factor = 240 := 
sorry

end full_size_mustang_length_l1_1793


namespace triangle_area_x_value_l1_1263

theorem triangle_area_x_value (x : ℝ) (h1 : x > 0) (h2 : 1 / 2 * x * (2 * x) = 64) : x = 8 :=
by
  sorry

end triangle_area_x_value_l1_1263


namespace intersection_of_M_and_N_l1_1625

def M : Set ℝ := { x | (x - 3) / (x - 1) ≤ 0 }
def N : Set ℝ := { x | -6 * x^2 + 11 * x - 4 > 0 }

theorem intersection_of_M_and_N : M ∩ N = { x | 1 < x ∧ x < 4 / 3 } :=
by 
  sorry

end intersection_of_M_and_N_l1_1625


namespace smallest_whole_number_above_perimeter_triangle_l1_1729

theorem smallest_whole_number_above_perimeter_triangle (s : ℕ) (h1 : 12 < s) (h2 : s < 26) :
  53 = Nat.ceil ((7 + 19 + s : ℕ) / 1) := by
  sorry

end smallest_whole_number_above_perimeter_triangle_l1_1729


namespace alice_instructors_l1_1600

noncomputable def num_students : ℕ := 40
noncomputable def num_life_vests_Alice_has : ℕ := 20
noncomputable def percent_students_with_their_vests : ℕ := 20
noncomputable def num_additional_life_vests_needed : ℕ := 22

-- Constants based on calculated conditions
noncomputable def num_students_with_their_vests : ℕ := (percent_students_with_their_vests * num_students) / 100
noncomputable def num_students_without_their_vests : ℕ := num_students - num_students_with_their_vests
noncomputable def num_life_vests_needed_for_students : ℕ := num_students_without_their_vests - num_life_vests_Alice_has
noncomputable def num_life_vests_needed_for_instructors : ℕ := num_additional_life_vests_needed - num_life_vests_needed_for_students

theorem alice_instructors : num_life_vests_needed_for_instructors = 10 := 
by
  sorry

end alice_instructors_l1_1600


namespace find_digit_to_make_divisible_by_seven_l1_1583

/-- 
  Given a number formed by concatenating 2023 digits of 6 with 2023 digits of 5.
  In a three-digit number 6*5, find the digit * to make this number divisible by 7.
  i.e., We must find the digit x such that the number 600 + 10x + 5 is divisible by 7.
-/
theorem find_digit_to_make_divisible_by_seven :
  ∃ x : ℕ, x < 10 ∧ (600 + 10 * x + 5) % 7 = 0 :=
sorry

end find_digit_to_make_divisible_by_seven_l1_1583


namespace renata_final_money_l1_1258

-- Defining the initial condition and the sequence of financial transactions.
def initial_money := 10
def donation := 4
def prize := 90
def slot_loss1 := 50
def slot_loss2 := 10
def slot_loss3 := 5
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65

-- Prove that given all these transactions, the final amount of money is $94.
theorem renata_final_money :
  initial_money 
  - donation 
  + prize 
  - slot_loss1 
  - slot_loss2 
  - slot_loss3 
  - water_cost 
  - lottery_ticket_cost 
  + lottery_prize 
  = 94 := 
by
  sorry

end renata_final_money_l1_1258


namespace odd_function_property_l1_1811

noncomputable def odd_function := {f : ℝ → ℝ // ∀ x : ℝ, f (-x) = -f x}

theorem odd_function_property (f : odd_function) (h1 : f.1 1 = -2) : f.1 (-1) + f.1 0 = 2 := by
  sorry

end odd_function_property_l1_1811


namespace slope_angle_of_tangent_line_expx_at_0_l1_1081

theorem slope_angle_of_tangent_line_expx_at_0 :
  let f := fun x : ℝ => Real.exp x 
  let f' := fun x : ℝ => Real.exp x
  ∀ x : ℝ, f' x = Real.exp x → 
  (∃ α : ℝ, Real.tan α = 1) →
  α = Real.pi / 4 :=
by
  intros f f' h_deriv h_slope
  sorry

end slope_angle_of_tangent_line_expx_at_0_l1_1081


namespace sin_cos_sum_l1_1773

theorem sin_cos_sum (x y r : ℝ) (h : r = Real.sqrt (x^2 + y^2)) (ha : (x = 5) ∧ (y = -12)) :
  (y / r) + (x / r) = -7 / 13 :=
by
  sorry

end sin_cos_sum_l1_1773


namespace ant_travel_distance_l1_1775

theorem ant_travel_distance (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 10) (h3 : r3 = 15) :
  let A_large := (1/3) * 2 * Real.pi * r3
  let D_radial := (r3 - r2) + (r2 - r1)
  let A_middle := (1/3) * 2 * Real.pi * r2
  let D_small := 2 * r1
  let A_small := (1/2) * 2 * Real.pi * r1
  A_large + D_radial + A_middle + D_small + A_small = (65 * Real.pi / 3) + 20 :=
by
  sorry

end ant_travel_distance_l1_1775


namespace arithmetic_sequence_common_difference_l1_1962

-- Define the conditions
variables {S_3 a_1 a_3 : ℕ}
variables (d : ℕ)
axiom h1 : S_3 = 6
axiom h2 : a_3 = 4
axiom h3 : S_3 = 3 * (a_1 + a_3) / 2

-- Prove that the common difference d is 2
theorem arithmetic_sequence_common_difference :
  d = (a_3 - a_1) / 2 → d = 2 :=
by
  sorry -- Proof to be completed

end arithmetic_sequence_common_difference_l1_1962


namespace find_S30_l1_1716

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l1_1716


namespace jim_juice_amount_l1_1617

def susan_juice : ℚ := 3 / 8
def jim_fraction : ℚ := 5 / 6

theorem jim_juice_amount : jim_fraction * susan_juice = 5 / 16 := by
  sorry

end jim_juice_amount_l1_1617


namespace ladybugs_with_spots_l1_1783

theorem ladybugs_with_spots (total_ladybugs without_spots with_spots : ℕ) 
  (h1 : total_ladybugs = 67082) 
  (h2 : without_spots = 54912) 
  (h3 : with_spots = total_ladybugs - without_spots) : 
  with_spots = 12170 := 
by 
  -- hole for the proof 
  sorry

end ladybugs_with_spots_l1_1783


namespace income_increase_l1_1147

variable (a : ℝ)

theorem income_increase (h : ∃ a : ℝ, a > 0):
  a * 1.142 = a * 1 + a * 0.142 :=
by
  sorry

end income_increase_l1_1147


namespace arithmetic_sequence_sum_l1_1358

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S_9 : ℝ)
  (h1 : a 1 + a 4 + a 7 = 15)
  (h2 : a 3 + a 6 + a 9 = 3)
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n) :
  S_9 = 27 :=
by
  sorry

end arithmetic_sequence_sum_l1_1358


namespace intersection_P_Q_l1_1610

def P (k : ℤ) (α : ℝ) : Prop := 2 * k * Real.pi ≤ α ∧ α ≤ (2 * k + 1) * Real.pi
def Q (α : ℝ) : Prop := -4 ≤ α ∧ α ≤ 4

theorem intersection_P_Q :
  (∃ k : ℤ, P k α) ∧ Q α ↔ (-4 ≤ α ∧ α ≤ -Real.pi) ∨ (0 ≤ α ∧ α ≤ Real.pi) :=
by
  sorry

end intersection_P_Q_l1_1610


namespace line_equation_passing_through_point_and_opposite_intercepts_l1_1670

theorem line_equation_passing_through_point_and_opposite_intercepts 
  : ∃ (a b : ℝ), (y = a * x) ∨ (x - y = b) :=
by
  use (3/2), (-1)
  sorry

end line_equation_passing_through_point_and_opposite_intercepts_l1_1670


namespace distance_from_dorm_to_city_l1_1569

theorem distance_from_dorm_to_city (D : ℝ) (h1 : D = (1/4)*D + (1/2)*D + 10 ) : D = 40 :=
sorry

end distance_from_dorm_to_city_l1_1569


namespace find_digits_l1_1806

theorem find_digits (x y z : ℕ) (hx : x ≤ 9) (hy : y ≤ 9) (hz : z ≤ 9)
    (h_eq : (10*x+5) * (300 + 10*y + z) = 7850) : x = 2 ∧ y = 1 ∧ z = 4 :=
by {
  sorry
}

end find_digits_l1_1806


namespace product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l1_1895

theorem product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers (n : ℤ) :
  let T := (n - 1) * n * (n + 1) * (n + 2)
  let M := n * (n + 1)
  T = (M - 2) * M :=
by
  -- proof here
  sorry

end product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l1_1895


namespace middle_tree_less_half_tallest_tree_l1_1963

theorem middle_tree_less_half_tallest_tree (T M S : ℝ)
  (hT : T = 108)
  (hS : S = 1/4 * M)
  (hS_12 : S = 12) :
  (1/2 * T) - M = 6 := 
sorry

end middle_tree_less_half_tallest_tree_l1_1963


namespace floor_ineq_l1_1527

theorem floor_ineq (x y : ℝ) : 
  Int.floor (2 * x) + Int.floor (2 * y) ≥ Int.floor x + Int.floor y + Int.floor (x + y) := 
sorry

end floor_ineq_l1_1527


namespace divisible_by_6_l1_1491

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end divisible_by_6_l1_1491


namespace sticks_left_is_correct_l1_1920

-- Define the initial conditions
def initial_popsicle_sticks : ℕ := 170
def popsicle_sticks_per_group : ℕ := 15
def number_of_groups : ℕ := 10

-- Define the total number of popsicle sticks given out to the groups
def total_sticks_given : ℕ := popsicle_sticks_per_group * number_of_groups

-- Define the number of popsicle sticks left
def sticks_left : ℕ := initial_popsicle_sticks - total_sticks_given

-- Prove that the number of sticks left is 20
theorem sticks_left_is_correct : sticks_left = 20 :=
by
  sorry

end sticks_left_is_correct_l1_1920


namespace find_second_divisor_l1_1390

theorem find_second_divisor :
  ∃ y : ℝ, (320 / 2) / y = 53.33 ∧ y = 160 / 53.33 :=
by
  sorry

end find_second_divisor_l1_1390


namespace tailor_cut_difference_l1_1859

def dress_silk_cut : ℝ := 0.75
def dress_satin_cut : ℝ := 0.60
def dress_chiffon_cut : ℝ := 0.55
def pants_cotton_cut : ℝ := 0.50
def pants_polyester_cut : ℝ := 0.45

theorem tailor_cut_difference :
  (dress_silk_cut + dress_satin_cut + dress_chiffon_cut) - (pants_cotton_cut + pants_polyester_cut) = 0.95 :=
by
  sorry

end tailor_cut_difference_l1_1859


namespace division_theorem_l1_1705

theorem division_theorem (k : ℕ) (h : k = 6) : 24 / k = 4 := by
  sorry

end division_theorem_l1_1705


namespace chromium_percentage_in_new_alloy_l1_1825

noncomputable def percentage_chromium_new_alloy (w1 w2 p1 p2 : ℝ) : ℝ :=
  ((p1 * w1 + p2 * w2) / (w1 + w2)) * 100

theorem chromium_percentage_in_new_alloy :
  percentage_chromium_new_alloy 15 35 0.12 0.10 = 10.6 :=
by
  sorry

end chromium_percentage_in_new_alloy_l1_1825


namespace coordinates_of_OC_l1_1918

-- Define the given vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the dot product for ℝ × ℝ
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define OC as a point on line OP, parameterized by t
def OC (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Define CA and CB
def CA (t : ℝ) : ℝ × ℝ := (OA.1 - (OC t).1, OA.2 - (OC t).2)
def CB (t : ℝ) : ℝ × ℝ := (OB.1 - (OC t).1, OB.2 - (OC t).2)

-- Prove that minimization of dot_product (CA t) (CB t) occurs at OC = (4, 2)
noncomputable def find_coordinates_at_min_dot_product : Prop :=
  ∃ (t : ℝ), t = 2 ∧ OC t = (4, 2)

-- The theorem statement
theorem coordinates_of_OC : find_coordinates_at_min_dot_product :=
sorry

end coordinates_of_OC_l1_1918


namespace f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l1_1706

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Part (1)
theorem f_positive_for_all_x (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k > -2 := sorry

-- Part (2)
theorem f_min_value_negative_two (k : ℝ) : (∀ x : ℝ, f x k ≥ -2) → k = -8 := sorry

-- Part (3)
theorem f_triangle_sides (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, (f x1 k + f x2 k > f x3 k) ∧ (f x2 k + f x3 k > f x1 k) ∧ (f x3 k + f x1 k > f x2 k)) ↔ (-1/2 ≤ k ∧ k ≤ 4) := sorry

end f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l1_1706


namespace g_at_3_value_l1_1591

theorem g_at_3_value (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : g 1 = 7)
  (h2 : g 2 = 11)
  (h3 : ∀ x : ℝ, g x = c * x + d * x + 3) : 
  g 3 = 15 :=
by
  sorry

end g_at_3_value_l1_1591


namespace cardinal_transitivity_l1_1033

variable {α β γ : Cardinal}

theorem cardinal_transitivity (h1 : α < β) (h2 : β < γ) : α < γ :=
  sorry

end cardinal_transitivity_l1_1033


namespace pipe_a_filling_time_l1_1702

theorem pipe_a_filling_time
  (pipeA_fill_time : ℝ)
  (pipeB_fill_time : ℝ)
  (both_pipes_open : Bool)
  (pipeB_shutoff_time : ℝ)
  (overflow_time : ℝ)
  (pipeB_rate : ℝ)
  (combined_rate : ℝ)
  (a_filling_time : ℝ) :
  pipeA_fill_time = 1 / 2 :=
by
  -- Definitions directly from conditions in a)
  let pipeA_fill_time := a_filling_time
  let pipeB_fill_time := 1  -- Pipe B fills in 1 hour
  let both_pipes_open := True
  let pipeB_shutoff_time := 0.5 -- Pipe B shuts 30 minutes before overflow
  let overflow_time := 0.5  -- Tank overflows in 30 minutes
  let pipeB_rate := 1 / pipeB_fill_time
  
  -- Goal to prove
  sorry

end pipe_a_filling_time_l1_1702


namespace two_pow_1000_mod_17_l1_1681

theorem two_pow_1000_mod_17 : 2^1000 % 17 = 0 :=
by {
  sorry
}

end two_pow_1000_mod_17_l1_1681


namespace range_of_k_l1_1987

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k - 1) * x^2 + (k - 1) * x + 2 > 0) ↔ 1 ≤ k ∧ k < 9 :=
by
  sorry

end range_of_k_l1_1987


namespace lower_limit_of_range_l1_1857

theorem lower_limit_of_range (x y : ℝ) (hx1 : 3 < x) (hx2 : x < 8) (hx3 : y < x) (hx4 : x < 10) (hx5 : x = 7) : 3 < y ∧ y ≤ 7 :=
by
  sorry

end lower_limit_of_range_l1_1857


namespace direct_proportion_k_l1_1249

theorem direct_proportion_k (k x : ℝ) : ((k-1) * x + k^2 - 1 = 0) ∧ (k ≠ 1) ↔ k = -1 := 
sorry

end direct_proportion_k_l1_1249


namespace flea_never_lands_on_all_points_l1_1934

noncomputable def a_n (n : ℕ) : ℕ := (n * (n + 1) / 2) % 300

theorem flea_never_lands_on_all_points :
  ∃ k : ℕ, k < 300 ∧ ∀ n : ℕ, a_n n ≠ k :=
sorry

end flea_never_lands_on_all_points_l1_1934


namespace find_birth_rate_l1_1926

noncomputable def average_birth_rate (B : ℕ) : Prop :=
  let death_rate := 3
  let net_increase_per_2_seconds := B - death_rate
  let seconds_per_hour := 3600
  let hours_per_day := 24
  let seconds_per_day := seconds_per_hour * hours_per_day
  let net_increase_times := seconds_per_day / 2
  let total_net_increase := net_increase_times * net_increase_per_2_seconds
  total_net_increase = 172800

theorem find_birth_rate (B : ℕ) (h : average_birth_rate B) : B = 7 :=
  sorry

end find_birth_rate_l1_1926


namespace height_of_sarah_building_l1_1269

-- Define the conditions
def shadow_length_building : ℝ := 75
def height_pole : ℝ := 15
def shadow_length_pole : ℝ := 30

-- Define the height of the building
def height_building : ℝ := 38

-- Height of Sarah's building given the conditions
theorem height_of_sarah_building (h : ℝ) (H1 : shadow_length_building = 75)
    (H2 : height_pole = 15) (H3 : shadow_length_pole = 30) :
    h = height_building :=
by
  -- State the ratio of the height of the pole to its shadow
  have ratio_pole : ℝ := height_pole / shadow_length_pole

  -- Set up the ratio for Sarah's building and solve for h
  have h_eq : ℝ := ratio_pole * shadow_length_building

  -- Provide the proof (skipped here)
  sorry

end height_of_sarah_building_l1_1269


namespace book_distribution_l1_1393

theorem book_distribution (a b : ℕ) (h1 : a + b = 282) (h2 : (3 / 4) * a = (5 / 9) * b) : a = 120 ∧ b = 162 := by
  sorry

end book_distribution_l1_1393


namespace solve_for_x_l1_1159

open Real

-- Define the condition and the target result
def target (x : ℝ) : Prop :=
  sqrt (9 + sqrt (16 + 3 * x)) + sqrt (3 + sqrt (4 + x)) = 3 + 3 * sqrt 2

theorem solve_for_x (x : ℝ) (h : target x) : x = 8 * sqrt 2 / 3 :=
  sorry

end solve_for_x_l1_1159


namespace find_speeds_and_circumference_l1_1832

variable (Va Vb : ℝ)
variable (l : ℝ)

axiom smaller_arc_condition : 10 * (Va + Vb) = 150
axiom larger_arc_condition : 14 * (Va + Vb) = l - 150
axiom travel_condition : l / Va = 90 / Vb 

theorem find_speeds_and_circumference :
  Va = 12 ∧ Vb = 3 ∧ l = 360 := by
  sorry

end find_speeds_and_circumference_l1_1832


namespace four_genuine_coin_probability_l1_1669

noncomputable def probability_all_genuine_given_equal_weight : ℚ :=
  let total_coins := 20
  let genuine_coins := 12
  let counterfeit_coins := 8

  -- Calculate the probability of selecting two genuine coins from total coins
  let prob_first_pair_genuine := (genuine_coins / total_coins) * 
                                    ((genuine_coins - 1) / (total_coins - 1))

  -- Updating remaining counts after selecting the first pair
  let remaining_genuine_coins := genuine_coins - 2
  let remaining_total_coins := total_coins - 2

  -- Calculate the probability of selecting another two genuine coins
  let prob_second_pair_genuine := (remaining_genuine_coins / remaining_total_coins) * 
                                    ((remaining_genuine_coins - 1) / (remaining_total_coins - 1))

  -- Probability of A ∩ B
  let prob_A_inter_B := prob_first_pair_genuine * prob_second_pair_genuine

  -- Assuming prob_B represents the weighted probabilities including complexities
  let prob_B := (110 / 1077) -- This is an estimated combined probability for the purpose of this definition

  -- Conditional probability P(A | B)
  prob_A_inter_B / prob_B

theorem four_genuine_coin_probability :
  probability_all_genuine_given_equal_weight = 110 / 1077 := sorry

end four_genuine_coin_probability_l1_1669


namespace max_snacks_l1_1682

-- Define the conditions and the main statement we want to prove

def single_snack_cost : ℕ := 2
def four_snack_pack_cost : ℕ := 6
def six_snack_pack_cost : ℕ := 8
def budget : ℕ := 20

def max_snacks_purchased : ℕ := 14

theorem max_snacks (h1 : single_snack_cost = 2) 
                   (h2 : four_snack_pack_cost = 6) 
                   (h3 : six_snack_pack_cost = 8) 
                   (h4 : budget = 20) : 
                   max_snacks_purchased = 14 := 
by {
  sorry
}

end max_snacks_l1_1682


namespace find_triplet_x_y_z_l1_1712

theorem find_triplet_x_y_z :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + 1 / (y + 1 / z : ℝ) = (10 : ℝ) / 7) ∧ (x = 1 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_triplet_x_y_z_l1_1712


namespace fraction_equality_l1_1464

theorem fraction_equality (a b : ℝ) (h : (1 / a) - (1 / b) = 4) :
  (a - 2 * a * b - b) / (2 * a - 2 * b + 7 * a * b) = 6 :=
by
  sorry

end fraction_equality_l1_1464


namespace find_value_expression_l1_1511

theorem find_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 4)
  (h3 : z^2 + x * z + x^2 = 79) :
  x * y + y * z + x * z = 20 := 
sorry

end find_value_expression_l1_1511


namespace minimum_travel_time_l1_1205

structure TravelSetup where
  distance_ab : ℝ
  number_of_people : ℕ
  number_of_bicycles : ℕ
  speed_cyclist : ℝ
  speed_pedestrian : ℝ
  unattended_rule : Prop

theorem minimum_travel_time (setup : TravelSetup) : setup.distance_ab = 45 → 
                                                    setup.number_of_people = 3 → 
                                                    setup.number_of_bicycles = 2 → 
                                                    setup.speed_cyclist = 15 → 
                                                    setup.speed_pedestrian = 5 → 
                                                    setup.unattended_rule → 
                                                    ∃ t : ℝ, t = 3 := 
by
  intros
  sorry

end minimum_travel_time_l1_1205


namespace master_craftsman_total_parts_l1_1475

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l1_1475


namespace largest_invertible_interval_l1_1848

def g (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 4

theorem largest_invertible_interval (x : ℝ) (hx : x = 2) : 
  ∃ I : Set ℝ, (I = Set.univ ∩ {y | y ≥ 3 / 2}) ∧ ∀ y ∈ I, g y = 3 * (y - 3 / 2) ^ 2 - 11 / 4 ∧ g y ∈ I ∧ Function.Injective (g ∘ (fun z => z : I → ℝ)) :=
sorry

end largest_invertible_interval_l1_1848


namespace induction_base_case_not_necessarily_one_l1_1440

theorem induction_base_case_not_necessarily_one :
  (∀ (P : ℕ → Prop) (n₀ : ℕ), (P n₀) → (∀ n, n ≥ n₀ → P n → P (n + 1)) → ∀ n, n ≥ n₀ → P n) ↔
  (∃ n₀ : ℕ, n₀ ≠ 1) :=
sorry

end induction_base_case_not_necessarily_one_l1_1440


namespace kitty_vacuum_time_l1_1034

theorem kitty_vacuum_time
  (weekly_toys : ℕ := 5)
  (weekly_windows : ℕ := 15)
  (weekly_furniture : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  (weeks : ℕ := 4)
  : (weekly_toys + weekly_windows + weekly_furniture) * weeks < total_cleaning_time ∧ ((total_cleaning_time - ((weekly_toys + weekly_windows + weekly_furniture) * weeks)) / weeks = 20)
  := by
  sorry

end kitty_vacuum_time_l1_1034


namespace tom_spending_is_correct_l1_1039

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l1_1039


namespace initially_calculated_average_l1_1317

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end initially_calculated_average_l1_1317


namespace DE_minimal_length_in_triangle_l1_1254

noncomputable def min_length_DE (BC AC : ℝ) (angle_B : ℝ) : ℝ :=
  if BC = 5 ∧ AC = 12 ∧ angle_B = 13 then 2 * Real.sqrt 3 else sorry

theorem DE_minimal_length_in_triangle :
  min_length_DE 5 12 13 = 2 * Real.sqrt 3 :=
sorry

end DE_minimal_length_in_triangle_l1_1254


namespace problem_conditions_l1_1351

theorem problem_conditions (x y : ℝ) (hx : x * (Real.exp x + Real.log x + x) = 1) (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  (0 < x ∧ x < 1) ∧ (y - x > 1) ∧ (y - x < 3 / 2) :=
by
  sorry

end problem_conditions_l1_1351


namespace max_contribution_l1_1642

theorem max_contribution (n : ℕ) (total : ℝ) (min_contribution : ℝ)
  (h1 : n = 12) (h2 : total = 20) (h3 : min_contribution = 1)
  (h4 : ∀ i : ℕ, i < n → min_contribution ≤ min_contribution) :
  ∃ max_contrib : ℝ, max_contrib = 9 :=
by
  sorry

end max_contribution_l1_1642


namespace grid_cut_990_l1_1852

theorem grid_cut_990 (grid : Matrix (Fin 1000) (Fin 1000) (Fin 2)) :
  (∃ (rows_to_remove : Finset (Fin 1000)), rows_to_remove.card = 990 ∧ 
   ∀ col : Fin 1000, ∃ row ∈ (Finset.univ \ rows_to_remove), grid row col = 1) ∨
  (∃ (cols_to_remove : Finset (Fin 1000)), cols_to_remove.card = 990 ∧ 
   ∀ row : Fin 1000, ∃ col ∈ (Finset.univ \ cols_to_remove), grid row col = 0) :=
sorry

end grid_cut_990_l1_1852


namespace Gunther_typing_correct_l1_1868

def GuntherTypingProblem : Prop :=
  let first_phase := (160 * (120 / 3))
  let second_phase := (200 * (180 / 3))
  let third_phase := (50 * 60)
  let fourth_phase := (140 * (90 / 3))
  let total_words := first_phase + second_phase + third_phase + fourth_phase
  total_words = 26200

theorem Gunther_typing_correct : GuntherTypingProblem := by
  sorry

end Gunther_typing_correct_l1_1868


namespace percent_calculation_l1_1557

theorem percent_calculation (y : ℝ) : (0.3 * (0.6 * y) = 0.18 * y) → (0.18 * y / y) * 100 = 18 := by
  sorry

end percent_calculation_l1_1557


namespace gary_current_weekly_eggs_l1_1910

noncomputable def egg_laying_rates : List ℕ := [6, 5, 7, 4]

def total_eggs_per_day (rates : List ℕ) : ℕ :=
  rates.foldl (· + ·) 0

def total_eggs_per_week (eggs_per_day : ℕ) : ℕ :=
  eggs_per_day * 7

theorem gary_current_weekly_eggs : 
  total_eggs_per_week (total_eggs_per_day egg_laying_rates) = 154 :=
by
  sorry

end gary_current_weekly_eggs_l1_1910


namespace square_area_from_circles_l1_1048

theorem square_area_from_circles :
  (∀ (r : ℝ), r = 7 → ∀ (n : ℕ), n = 4 → (∃ (side_length : ℝ), side_length = 2 * (2 * r))) →
  ∀ (side_length : ℝ), side_length = 28 →
  (∃ (area : ℝ), area = side_length * side_length ∧ area = 784) :=
sorry

end square_area_from_circles_l1_1048


namespace swimmer_speeds_l1_1789

variable (a s r : ℝ)
variable (x z y : ℝ)

theorem swimmer_speeds (h : s < r) (h' : r < 100 * s / (50 + s)) :
    (100 * s - 50 * r - r * s) / ((3 * s - r) * a) = x ∧ 
    (100 * s - 50 * r - r * s) / ((r - s) * a) = z := by
    sorry

end swimmer_speeds_l1_1789


namespace least_positive_four_digit_multiple_of_6_l1_1460

theorem least_positive_four_digit_multiple_of_6 : 
  ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 6 = 0 ∧ (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 6 = 0 → n ≤ m) := 
sorry

end least_positive_four_digit_multiple_of_6_l1_1460


namespace taxi_ride_cost_l1_1115

-- Definitions based on conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def minimum_charge : ℝ := 5.00
def fare (miles : ℝ) : ℝ := base_fare + miles * cost_per_mile

-- Theorem statement reflecting the problem
theorem taxi_ride_cost (miles : ℝ) (h : miles < 4) : fare miles < minimum_charge → fare miles = minimum_charge :=
by
  sorry

end taxi_ride_cost_l1_1115


namespace greatest_possible_length_l1_1549

theorem greatest_possible_length :
  ∃ (g : ℕ), g = Nat.gcd 700 (Nat.gcd 385 1295) ∧ g = 35 :=
by
  sorry

end greatest_possible_length_l1_1549


namespace possible_distance_between_houses_l1_1827

variable (d : ℝ)

theorem possible_distance_between_houses (h_d1 : 1 ≤ d) (h_d2 : d ≤ 5) : 1 ≤ d ∧ d ≤ 5 :=
by
  exact ⟨h_d1, h_d2⟩

end possible_distance_between_houses_l1_1827


namespace common_chord_circle_eq_l1_1494

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end common_chord_circle_eq_l1_1494


namespace CarlYardAreaIsCorrect_l1_1619

noncomputable def CarlRectangularYardArea (post_count : ℕ) (distance_between_posts : ℕ) (long_side_factor : ℕ) :=
  let x := post_count / (2 * (1 + long_side_factor))
  let short_side := (x - 1) * distance_between_posts
  let long_side := (long_side_factor * x - 1) * distance_between_posts
  short_side * long_side

theorem CarlYardAreaIsCorrect :
  CarlRectangularYardArea 24 5 3 = 825 := 
by
  -- calculation steps if needed or
  sorry

end CarlYardAreaIsCorrect_l1_1619


namespace number_of_3digit_even_numbers_divisible_by_9_l1_1506

theorem number_of_3digit_even_numbers_divisible_by_9 : 
    ∃ n : ℕ, (n = 50) ∧
    (∀ k, (108 + (k - 1) * 18 = 990) ↔ (108 ≤ 108 + (k - 1) * 18 ∧ 108 + (k - 1) * 18 ≤ 999)) :=
by {
  sorry
}

end number_of_3digit_even_numbers_divisible_by_9_l1_1506


namespace cube_probability_l1_1944

theorem cube_probability :
  let m := 1
  let n := 504
  ∀ (faces : Finset (Fin 6)) (nums : Finset (Fin 9)), 
    faces.card = 6 → nums.card = 9 →
    (∀ f ∈ faces, ∃ n ∈ nums, true) →
    m + n = 505 :=
by
  sorry

end cube_probability_l1_1944


namespace unique_four_digit_numbers_l1_1919

theorem unique_four_digit_numbers (digits : Finset ℕ) (odd_digits : Finset ℕ) :
  digits = {2, 3, 4, 5, 6} → 
  odd_digits = {3, 5} → 
  ∃ (n : ℕ), n = 14 :=
by
  sorry

end unique_four_digit_numbers_l1_1919


namespace benny_added_march_l1_1231

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end benny_added_march_l1_1231


namespace evaluate_expression_l1_1099

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

axiom g_1_2 : g 1 = 2
axiom g_4_7 : g 4 = 7
axiom g_3_8 : g 3 = 8

theorem evaluate_expression :
  g_inv (g_inv 8 * g_inv 2) = 3 :=
by
  sorry

end evaluate_expression_l1_1099


namespace part_one_part_two_l1_1606

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- (1) Prove that if a = 2, then ∀ x, f(x, 2) ≤ 6 implies -1 ≤ x ≤ 3
theorem part_one (x : ℝ) : f x 2 ≤ 6 → -1 ≤ x ∧ x ≤ 3 :=
by sorry

-- (2) Prove that ∀ a ∈ ℝ, ∀ x ∈ ℝ, (f(x, a) + g(x) ≥ 3 → a ∈ [2, +∞))
theorem part_two (a x : ℝ) : f x a + g x ≥ 3 → 2 ≤ a :=
by sorry

end part_one_part_two_l1_1606


namespace boy_speed_l1_1252

theorem boy_speed (d : ℝ) (v₁ v₂ : ℝ) (t₁ t₂ l e : ℝ) :
  d = 2 ∧ v₂ = 8 ∧ l = 7 / 60 ∧ e = 8 / 60 ∧ t₁ = d / v₁ ∧ t₂ = d / v₂ ∧ t₁ - t₂ = l + e → v₁ = 4 :=
by
  sorry

end boy_speed_l1_1252


namespace find_number_l1_1188

theorem find_number (x : ℝ) (h : (x / 4) + 9 = 15) : x = 24 :=
by
  sorry

end find_number_l1_1188


namespace time_to_cover_length_correct_l1_1117

-- Given conditions
def speed_escalator := 20 -- ft/sec
def length_escalator := 210 -- feet
def speed_person := 4 -- ft/sec

-- Time is distance divided by speed
def time_to_cover_length : ℚ :=
  length_escalator / (speed_escalator + speed_person)

theorem time_to_cover_length_correct :
  time_to_cover_length = 8.75 := by
  sorry

end time_to_cover_length_correct_l1_1117


namespace calculate_sin_product_l1_1813

theorem calculate_sin_product (α β : ℝ) (h1 : Real.sin (α + β) = 0.2) (h2 : Real.cos (α - β) = 0.3) :
  Real.sin (α + π/4) * Real.sin (β + π/4) = 0.25 :=
by
  sorry

end calculate_sin_product_l1_1813


namespace sum_of_numbers_l1_1389

theorem sum_of_numbers (avg : ℝ) (count : ℕ) (h_avg : avg = 5.7) (h_count : count = 8) : (avg * count = 45.6) :=
by
  sorry

end sum_of_numbers_l1_1389


namespace max_area_of_triangle_l1_1302

noncomputable def maxAreaTriangle (m_a m_b m_c : ℝ) : ℝ :=
  1/3 * Real.sqrt (2 * (m_a^2 * m_b^2 + m_b^2 * m_c^2 + m_c^2 * m_a^2) - (m_a^4 + m_b^4 + m_c^4))

theorem max_area_of_triangle (m_a m_b m_c : ℝ) (h1 : m_a ≤ 2) (h2 : m_b ≤ 3) (h3 : m_c ≤ 4) :
  maxAreaTriangle m_a m_b m_c ≤ 4 :=
sorry

end max_area_of_triangle_l1_1302


namespace ratio_B_C_l1_1068

variable (A B C : ℕ)
variable (h1 : A = B + 2)
variable (h2 : A + B + C = 37)
variable (h3 : B = 14)

theorem ratio_B_C : B / C = 2 := by
  sorry

end ratio_B_C_l1_1068


namespace well_depth_l1_1101

theorem well_depth (e x a b c d : ℝ)
  (h1 : x = 2 * a + b)
  (h2 : x = 3 * b + c)
  (h3 : x = 4 * c + d)
  (h4 : x = 5 * d + e)
  (h5 : x = 6 * e + a) :
  x = 721 / 76 * e ∧
  a = 265 / 76 * e ∧
  b = 191 / 76 * e ∧
  c = 37 / 19 * e ∧
  d = 129 / 76 * e :=
sorry

end well_depth_l1_1101


namespace equal_clubs_and_students_l1_1434

theorem equal_clubs_and_students (S C : ℕ) 
  (h1 : ∀ c : ℕ, c < C → ∃ (m : ℕ → Prop), (∃ p, m p ∧ p = 3))
  (h2 : ∀ s : ℕ, s < S → ∃ (n : ℕ → Prop), (∃ p, n p ∧ p = 3)) :
  S = C := 
by
  sorry

end equal_clubs_and_students_l1_1434


namespace perimeter_ratio_of_divided_square_l1_1605

theorem perimeter_ratio_of_divided_square
  (S_ΔADE : ℝ) (S_EDCB : ℝ)
  (S_ratio : S_ΔADE / S_EDCB = 5 / 19)
  : ∃ (perim_ΔADE perim_EDCB : ℝ),
  perim_ΔADE / perim_EDCB = 15 / 22 :=
by
  -- Let S_ΔADE = 5x and S_EDCB = 19x
  -- x can be calculated based on the given S_ratio = 5/19
  -- Apply geometric properties and simplifications analogous to the described solution.
  sorry

end perimeter_ratio_of_divided_square_l1_1605


namespace correct_average_is_40_point_3_l1_1794

noncomputable def incorrect_average : ℝ := 40.2
noncomputable def incorrect_total_sum : ℝ := incorrect_average * 10
noncomputable def incorrect_first_number_adjustment : ℝ := 17
noncomputable def incorrect_second_number_actual : ℝ := 31
noncomputable def incorrect_second_number_provided : ℝ := 13
noncomputable def correct_total_sum : ℝ := incorrect_total_sum - incorrect_first_number_adjustment + (incorrect_second_number_actual - incorrect_second_number_provided)
noncomputable def number_of_values : ℝ := 10

theorem correct_average_is_40_point_3 :
  correct_total_sum / number_of_values = 40.3 :=
by
  sorry

end correct_average_is_40_point_3_l1_1794


namespace parabola_fixed_point_l1_1529

theorem parabola_fixed_point (t : ℝ) : ∃ y, y = 4 * 3^2 + 2 * t * 3 - 3 * t ∧ y = 36 :=
by
  exists 36
  sorry

end parabola_fixed_point_l1_1529


namespace equality_of_a_and_b_l1_1844

theorem equality_of_a_and_b
  (a b : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (h : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := 
sorry

end equality_of_a_and_b_l1_1844


namespace geometric_series_sum_eq_l1_1360

theorem geometric_series_sum_eq :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5
  (∀ S_n, S_n = a * (1 - r^n) / (1 - r) → S_n = 1 / 3) :=
by
  intro a r n S_n
  sorry

end geometric_series_sum_eq_l1_1360


namespace parallelogram_sides_l1_1182

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 3 * x + 6 = 15) 
  (h2 : 10 * y - 2 = 12) :
  x + y = 4.4 := 
sorry

end parallelogram_sides_l1_1182


namespace eleven_hash_five_l1_1238

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five_l1_1238


namespace average_contribution_increase_l1_1369

theorem average_contribution_increase
  (average_old : ℝ)
  (num_people_old : ℕ)
  (john_donation : ℝ)
  (increase_percentage : ℝ) :
  average_old = 75 →
  num_people_old = 3 →
  john_donation = 150 →
  increase_percentage = 25 :=
by {
  sorry
}

end average_contribution_increase_l1_1369


namespace paper_clips_in_two_cases_l1_1923

-- Defining the problem statement in Lean 4
theorem paper_clips_in_two_cases (c b : ℕ) :
  2 * (c * b * 300) = 2 * c * b * 300 :=
by
  sorry

end paper_clips_in_two_cases_l1_1923


namespace rainbow_nerds_total_l1_1977

theorem rainbow_nerds_total
  (purple yellow green red blue : ℕ)
  (h1 : purple = 10)
  (h2 : yellow = purple + 4)
  (h3 : green = yellow - 2)
  (h4 : red = 3 * green)
  (h5 : blue = red / 2) :
  (purple + yellow + green + red + blue = 90) :=
by
  sorry

end rainbow_nerds_total_l1_1977


namespace new_ratio_books_to_clothes_l1_1079

-- Given initial conditions
def initial_ratio := (7, 4, 3)
def electronics_weight : ℕ := 12
def clothes_removed : ℕ := 8

-- Definitions based on the problem
def part_weight : ℕ := electronics_weight / initial_ratio.2.2
def initial_books_weight : ℕ := initial_ratio.1 * part_weight
def initial_clothes_weight : ℕ := initial_ratio.2.1 * part_weight
def new_clothes_weight : ℕ := initial_clothes_weight - clothes_removed

-- Proof of the new ratio
theorem new_ratio_books_to_clothes : (initial_books_weight, new_clothes_weight) = (7 * part_weight, 2 * part_weight) :=
sorry

end new_ratio_books_to_clothes_l1_1079


namespace value_of_a_plus_b_l1_1158

-- Define the main problem conditions
variables (a b : ℝ)

-- State the problem in Lean
theorem value_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = 3) (h3 : |a - b| = - (a - b)) :
  a + b = 5 ∨ a + b = 1 :=
sorry

end value_of_a_plus_b_l1_1158


namespace find_cookies_per_tray_l1_1239

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray_l1_1239


namespace lateral_surface_area_of_square_pyramid_l1_1938

-- Definitions based on the conditions in a)
def baseEdgeLength : ℝ := 4
def slantHeight : ℝ := 3

-- Lean 4 statement for the proof problem
theorem lateral_surface_area_of_square_pyramid :
  let height := Real.sqrt (slantHeight^2 - (baseEdgeLength / 2)^2)
  let lateralArea := (1 / 2) * 4 * (baseEdgeLength * height)
  lateralArea = 8 * Real.sqrt 5 :=
by
  sorry

end lateral_surface_area_of_square_pyramid_l1_1938


namespace geometry_problem_l1_1824

noncomputable def vertices_on_hyperbola (A B C : ℝ × ℝ) : Prop :=
  (∃ x1 y1, A = (x1, y1) ∧ 2 * x1^2 - y1^2 = 4) ∧
  (∃ x2 y2, B = (x2, y2) ∧ 2 * x2^2 - y2^2 = 4) ∧
  (∃ x3 y3, C = (x3, y3) ∧ 2 * x3^2 - y3^2 = 4)

noncomputable def midpoints (A B C M N P : ℝ × ℝ) : Prop :=
  (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
  (N = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
  (P = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))

noncomputable def slopes (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 = M.2 / M.1 ∧ k2 = N.2 / N.1 ∧ k3 = P.2 / P.1

noncomputable def sum_of_slopes (A B C : ℝ × ℝ) (k1 k2 k3 : ℝ) : Prop :=
  ((A.2 - B.2) / (A.1 - B.1) +
   (B.2 - C.2) / (B.1 - C.1) +
   (C.2 - A.2) / (C.1 - A.1)) = -1

theorem geometry_problem 
  (A B C M N P : ℝ × ℝ) (k1 k2 k3 : ℝ) 
  (h1 : vertices_on_hyperbola A B C)
  (h2 : midpoints A B C M N P) 
  (h3 : slopes A B C M N P k1 k2 k3) 
  (h4 : sum_of_slopes A B C k1 k2 k3) :
  1/k1 + 1/k2 + 1/k3 = -1 / 2 :=
sorry

end geometry_problem_l1_1824


namespace square_of_1023_l1_1486

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l1_1486


namespace find_m_l1_1546

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 3, 2*m - 1}) (hB: B = {3, m^2}) (h_subset: B ⊆ A) : m = 1 :=
by
  sorry

end find_m_l1_1546


namespace not_satisfiable_conditions_l1_1878

theorem not_satisfiable_conditions (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) 
    (h3 : 10 * x + y % 80 = 0) (h4 : x + y = 2) : false := 
by 
  -- The proof is omitted because we are only asked for the statement.
  sorry

end not_satisfiable_conditions_l1_1878


namespace prob_high_quality_correct_l1_1335

noncomputable def prob_high_quality_seeds :=
  let p_first := 0.955
  let p_second := 0.02
  let p_third := 0.015
  let p_fourth := 0.01
  let p_hq_first := 0.5
  let p_hq_second := 0.15
  let p_hq_third := 0.1
  let p_hq_fourth := 0.05
  let p_hq := p_first * p_hq_first + p_second * p_hq_second + p_third * p_hq_third + p_fourth * p_hq_fourth
  p_hq

theorem prob_high_quality_correct : prob_high_quality_seeds = 0.4825 :=
  by sorry

end prob_high_quality_correct_l1_1335


namespace average_of_three_numbers_is_78_l1_1772

theorem average_of_three_numbers_is_78 (x y z : ℕ) (h1 : z = 2 * y) (h2 : y = 4 * x) (h3 : x = 18) :
  (x + y + z) / 3 = 78 :=
by sorry

end average_of_three_numbers_is_78_l1_1772


namespace burger_cost_l1_1457

theorem burger_cost (days_in_june : ℕ) (burgers_per_day : ℕ) (total_spent : ℕ) (h1 : days_in_june = 30) (h2 : burgers_per_day = 2) (h3 : total_spent = 720) : 
  total_spent / (burgers_per_day * days_in_june) = 12 :=
by
  -- We will prove this in Lean, but skipping the proof here
  sorry

end burger_cost_l1_1457


namespace find_MN_l1_1947

theorem find_MN (d D : ℝ) (h_d_lt_D : d < D) :
  ∃ MN : ℝ, MN = (d * D) / (D - d) :=
by
  sorry

end find_MN_l1_1947


namespace find_principal_amount_l1_1405

noncomputable def principal_amount (SI R T : ℝ) : ℝ :=
  SI / (R * T / 100)

theorem find_principal_amount :
  principal_amount 4052.25 9 5 = 9005 := by
sorry

end find_principal_amount_l1_1405


namespace simplify_expression_to_polynomial_l1_1768

theorem simplify_expression_to_polynomial :
    (3 * x^2 + 4 * x + 8) * (2 * x + 1) - 
    (2 * x + 1) * (x^2 + 5 * x - 72) + 
    (4 * x - 15) * (2 * x + 1) * (x + 6) = 
    12 * x^3 + 22 * x^2 - 12 * x - 10 :=
by
    sorry

end simplify_expression_to_polynomial_l1_1768


namespace proof_problem_l1_1080

theorem proof_problem (x : ℕ) (h : (x - 4) / 10 = 5) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l1_1080


namespace goals_scored_by_each_l1_1866

theorem goals_scored_by_each (total_goals : ℕ) (percentage : ℕ) (two_players_goals : ℕ) (each_player_goals : ℕ)
  (H1 : total_goals = 300)
  (H2 : percentage = 20)
  (H3 : two_players_goals = (percentage * total_goals) / 100)
  (H4 : two_players_goals / 2 = each_player_goals) :
  each_player_goals = 30 := by
  sorry

end goals_scored_by_each_l1_1866


namespace tan_A_in_triangle_ABC_l1_1984

theorem tan_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (ha : 0 < A) (ha_90 : A < π / 2) 
(hb : b = 3 * a * Real.sin B) : Real.tan A = Real.sqrt 2 / 4 :=
sorry

end tan_A_in_triangle_ABC_l1_1984


namespace t_plus_reciprocal_l1_1876

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l1_1876


namespace find_k_unique_solution_l1_1791

theorem find_k_unique_solution :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (k - x)/8) → (3*x^2 + (8 - 3*k)*x = 0)) →
    k = 8 / 3 :=
by
  intros k h
  -- Using sorry here to skip the proof
  sorry

end find_k_unique_solution_l1_1791


namespace probability_of_D_given_T_l1_1482

-- Definitions based on the conditions given in the problem.
def pr_D : ℚ := 1 / 400
def pr_Dc : ℚ := 399 / 400
def pr_T_given_D : ℚ := 1
def pr_T_given_Dc : ℚ := 0.05
def pr_T : ℚ := pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Statement to prove 
theorem probability_of_D_given_T : pr_T ≠ 0 → (pr_T_given_D * pr_D) / pr_T = 20 / 419 :=
by
  intros h1
  unfold pr_T pr_D pr_Dc pr_T_given_D pr_T_given_Dc
  -- Mathematical steps are skipped in Lean by inserting sorry
  sorry

-- Check that the statement can be built successfully
example : pr_D = 1 / 400 := by rfl
example : pr_Dc = 399 / 400 := by rfl
example : pr_T_given_D = 1 := by rfl
example : pr_T_given_Dc = 0.05 := by rfl
example : pr_T = (1 * (1 / 400) + 0.05 * (399 / 400)) := by rfl

end probability_of_D_given_T_l1_1482


namespace total_jellybeans_l1_1601

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l1_1601


namespace represent_259BC_as_neg259_l1_1675

def year_AD (n: ℤ) : ℤ := n

def year_BC (n: ℕ) : ℤ := -(n : ℤ)

theorem represent_259BC_as_neg259 : year_BC 259 = -259 := 
by 
  rw [year_BC]
  norm_num

end represent_259BC_as_neg259_l1_1675


namespace mike_earnings_l1_1261

def prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]

theorem mike_earnings :
  List.sum prices = 75 :=
by
  sorry

end mike_earnings_l1_1261


namespace cube_inequality_of_greater_l1_1135

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l1_1135


namespace smallest_possible_value_l1_1867

open Nat

theorem smallest_possible_value (c d : ℕ) (hc : c > d) (hc_pos : 0 < c) (hd_pos : 0 < d) (odd_cd : ¬Even (c + d)) :
  (∃ (y : ℚ), y > 0 ∧ y = (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ∧ y = 10 / 3) :=
by
  sorry

end smallest_possible_value_l1_1867


namespace soccer_league_games_l1_1450

theorem soccer_league_games (n_teams games_played : ℕ) (h1 : n_teams = 10) (h2 : games_played = 45) :
  ∃ k : ℕ, (n_teams * (n_teams - 1)) / 2 = games_played ∧ k = 1 :=
by
  sorry

end soccer_league_games_l1_1450


namespace find_quadratic_function_find_vertex_find_range_l1_1019

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def satisfies_points (a b c : ℝ) : Prop :=
  quadratic_function a b c (-1) = 0 ∧
  quadratic_function a b c 0 = -3 ∧
  quadratic_function a b c 2 = -3

theorem find_quadratic_function : ∃ a b c, satisfies_points a b c ∧ (a = 1 ∧ b = -2 ∧ c = -3) :=
sorry

theorem find_vertex (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∃ x y, x = 1 ∧ y = -4 ∧ ∀ x', x' > 1 → quadratic_function a b c x' > quadratic_function a b c x :=
sorry

theorem find_range (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = -3) :
  ∀ x, -1 < x ∧ x < 2 → -4 < quadratic_function a b c x ∧ quadratic_function a b c x < 0 :=
sorry

end find_quadratic_function_find_vertex_find_range_l1_1019


namespace find_b_squared_l1_1928

theorem find_b_squared
    (b : ℝ)
    (c_ellipse c_hyperbola a_ellipse a2_hyperbola b2_hyperbola : ℝ)
    (h1: a_ellipse^2 = 25)
    (h2 : b2_hyperbola = 9 / 4)
    (h3 : a2_hyperbola = 4)
    (h4 : c_hyperbola = Real.sqrt (a2_hyperbola + b2_hyperbola))
    (h5 : c_ellipse = c_hyperbola)
    (h6 : b^2 = a_ellipse^2 - c_ellipse^2)
: b^2 = 75 / 4 :=
sorry

end find_b_squared_l1_1928


namespace range_of_k_l1_1004

noncomputable def f (k : ℝ) (x : ℝ) := (Real.exp x) / (x^2) + 2 * k * Real.log x - k * x

theorem range_of_k (k : ℝ) (h₁ : ∀ x > 0, (deriv (f k) x = 0) → x = 2) : k < Real.exp 2 / 4 :=
by
  sorry

end range_of_k_l1_1004


namespace consecutive_grouping_probability_l1_1132

theorem consecutive_grouping_probability :
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements = 1 / 4620 :=
by
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  have h : (block_arrangements * green_factorial * orange_factorial * blue_factorial) = 103680 := sorry
  have h1 : (total_arrangements) = 479001600 := sorry
  calc
    (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements
    _ = 103680 / 479001600 := by rw [h, h1]
    _ = 1 / 4620 := sorry

end consecutive_grouping_probability_l1_1132


namespace quoted_price_of_shares_l1_1931

theorem quoted_price_of_shares (investment : ℝ) (face_value : ℝ) (rate_dividend : ℝ) (annual_income : ℝ) (num_shares : ℝ) (quoted_price : ℝ) :
  investment = 4455 ∧ face_value = 10 ∧ rate_dividend = 0.12 ∧ annual_income = 648 ∧ num_shares = annual_income / (rate_dividend * face_value) →
  quoted_price = investment / num_shares :=
by sorry

end quoted_price_of_shares_l1_1931


namespace Fedya_age_statement_l1_1320

theorem Fedya_age_statement (d a : ℕ) (today : ℕ) (birthday : ℕ) 
    (H1 : d + 2 = a) 
    (H2 : a + 2 = birthday + 3) 
    (H3 : birthday = today + 1) :
    ∃ sameYear y, (birthday < today + 2 ∨ today < birthday) ∧ ((sameYear ∧ y - today = 1) ∨ (¬ sameYear ∧ y - today = 0)) :=
by
  sorry

end Fedya_age_statement_l1_1320


namespace num_convex_numbers_without_repeats_l1_1846

def is_convex_number (a b c : ℕ) : Prop :=
  a < b ∧ b > c

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n < 10

def distinct_digits (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem num_convex_numbers_without_repeats : 
  (∃ (numbers : Finset (ℕ × ℕ × ℕ)), 
    (∀ a b c, (a, b, c) ∈ numbers -> is_convex_number a b c ∧ is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ distinct_digits a b c) ∧
    numbers.card = 204) :=
sorry

end num_convex_numbers_without_repeats_l1_1846


namespace right_triangle_area_is_integer_l1_1420

theorem right_triangle_area_is_integer (a b : ℕ) (h1 : ∃ (A : ℕ), A = (1 / 2 : ℚ) * ↑a * ↑b) : (a % 2 = 0) ∨ (b % 2 = 0) :=
sorry

end right_triangle_area_is_integer_l1_1420


namespace union_of_complements_eq_l1_1725

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_of_complements_eq :
  U = {1, 2, 3, 4, 5, 6, 7} →
  A = {2, 4, 5, 7} →
  B = {3, 4, 5} →
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 6, 7}) :=
by
  intros hU hA hB
  sorry

end union_of_complements_eq_l1_1725


namespace muffin_price_proof_l1_1071

noncomputable def price_per_muffin (s m t : ℕ) (contribution : ℕ) : ℕ :=
  contribution / (s + m + t)

theorem muffin_price_proof :
  ∀ (sasha_muffins melissa_muffins : ℕ) (h1 : sasha_muffins = 30) (h2 : melissa_muffins = 4 * sasha_muffins)
  (tiffany_muffins total_muffins : ℕ) (h3 : total_muffins = sasha_muffins + melissa_muffins)
  (h4 : tiffany_muffins = total_muffins / 2)
  (h5 : total_muffins = sasha_muffins + melissa_muffins + tiffany_muffins)
  (contribution : ℕ) (h6 : contribution = 900),
  price_per_muffin sasha_muffins melissa_muffins tiffany_muffins contribution = 4 :=
by
  intros sasha_muffins melissa_muffins h1 h2 tiffany_muffins total_muffins h3 h4 h5 contribution h6
  simp [price_per_muffin]
  sorry

end muffin_price_proof_l1_1071


namespace zongzi_profit_l1_1893

def initial_cost : ℕ := 10
def initial_price : ℕ := 16
def initial_bags_sold : ℕ := 200
def additional_sales_per_yuan (x : ℕ) : ℕ := 80 * x
def profit_per_bag (x : ℕ) : ℕ := initial_price - x - initial_cost
def number_of_bags_sold (x : ℕ) : ℕ := initial_bags_sold + additional_sales_per_yuan x
def total_profit (profit_per_bag : ℕ) (number_of_bags_sold : ℕ) : ℕ := profit_per_bag * number_of_bags_sold

theorem zongzi_profit (x : ℕ) : 
  total_profit (profit_per_bag x) (number_of_bags_sold x) = 1440 := 
sorry

end zongzi_profit_l1_1893


namespace quadratic_root_conditions_l1_1346

theorem quadratic_root_conditions : ∃ p q : ℝ, (p - 1)^2 - 4 * q > 0 ∧ (p + 1)^2 - 4 * q > 0 ∧ p^2 - 4 * q < 0 := 
sorry

end quadratic_root_conditions_l1_1346


namespace problem_statement_l1_1000

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 2) : a^2 + b^2 = 29 := 
by
  sorry

end problem_statement_l1_1000


namespace triangle_proof_problem_l1_1790

-- The conditions and question programmed as a Lean theorem statement
theorem triangle_proof_problem
    (A B C : ℝ)
    (h1 : A > B)
    (S T : ℝ)
    (h2 : A = C)
    (K : ℝ)
    (arc_mid_A : K = A): -- K is midpoint of the arc A
    
    RS = K := sorry

end triangle_proof_problem_l1_1790


namespace mixture_problem_l1_1427

theorem mixture_problem :
  ∀ (x P : ℝ), 
    let initial_solution := 70
    let initial_percentage := 0.20
    let final_percentage := 0.40
    let final_amount := 70
    (x = 70) →
    (initial_percentage * initial_solution + P * x = final_percentage * (initial_solution + x)) →
    (P = 0.60) :=
by
  intros x P initial_solution initial_percentage final_percentage final_amount hx h_eq
  sorry

end mixture_problem_l1_1427


namespace parabola_x0_range_l1_1755

variables {x₀ y₀ : ℝ}
def parabola (x₀ y₀ : ℝ) : Prop := y₀^2 = 8 * x₀

def focus (x : ℝ) : ℝ := 2

def directrix (x : ℝ) : Prop := x = -2

/-- Prove that for any point (x₀, y₀) on the parabola y² = 8x and 
if a circle centered at the focus intersects the directrix, then x₀ > 2. -/
theorem parabola_x0_range (x₀ y₀ : ℝ) (h1 : parabola x₀ y₀)
  (h2 : ((x₀ - 2)^2 + y₀^2)^(1/2) > (2 : ℝ)) : x₀ > 2 := 
sorry

end parabola_x0_range_l1_1755


namespace distance_from_neg2_eq4_l1_1922

theorem distance_from_neg2_eq4 (x : ℤ) : |x + 2| = 4 ↔ x = 2 ∨ x = -6 :=
by
  sorry

end distance_from_neg2_eq4_l1_1922


namespace unique_exponential_solution_l1_1964

theorem unique_exponential_solution (a x : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hx_pos : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by
  sorry

end unique_exponential_solution_l1_1964


namespace count_even_digits_in_base_5_of_567_l1_1937

def is_even (n : ℕ) : Bool := n % 2 = 0

def base_5_representation (n : ℕ) : List ℕ :=
  if h : n > 0 then
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else loop (n / 5) ((n % 5) :: acc)
    loop n []
  else [0]

def count_even_digits_in_base_5 (n : ℕ) : ℕ :=
  (base_5_representation n).filter is_even |>.length

theorem count_even_digits_in_base_5_of_567 :
  count_even_digits_in_base_5 567 = 2 := by
  sorry

end count_even_digits_in_base_5_of_567_l1_1937


namespace Rajesh_days_to_complete_l1_1454

theorem Rajesh_days_to_complete (Mahesh_days : ℕ) (Rajesh_days : ℕ) (Total_days : ℕ)
  (h1 : Mahesh_days = 45) (h2 : Total_days - 20 = Rajesh_days) (h3 : Total_days = 54) :
  Rajesh_days = 34 :=
by
  sorry

end Rajesh_days_to_complete_l1_1454


namespace intersect_trihedral_angle_l1_1574

-- Definitions of variables
variables {a b c : ℝ} (S : Type) 

-- Definition of a valid intersection condition
def valid_intersection (a b c : ℝ) : Prop :=
  a^2 + b^2 - c^2 > 0 ∧ b^2 + c^2 - a^2 > 0 ∧ a^2 + c^2 - b^2 > 0

-- Theorem statement
theorem intersect_trihedral_angle (h : valid_intersection a b c) : 
  ∃ (SA SB SC : ℝ), (SA^2 + SB^2 = a^2 ∧ SA^2 + SC^2 = b^2 ∧ SB^2 + SC^2 = c^2) :=
sorry

end intersect_trihedral_angle_l1_1574


namespace ratio_a_to_c_l1_1088

-- Define the variables and ratios
variables (x y z a b c d : ℝ)

-- Define the conditions as given ratios
variables (h1 : a / b = 2 * x / (3 * y))
variables (h2 : b / c = z / (5 * z))
variables (h3 : a / d = 4 * x / (7 * y))
variables (h4 : d / c = 7 * y / (3 * z))

-- Statement to prove the ratio of a to c
theorem ratio_a_to_c (x y z a b c d : ℝ) 
  (h1 : a / b = 2 * x / (3 * y)) 
  (h2 : b / c = z / (5 * z)) 
  (h3 : a / d = 4 * x / (7 * y)) 
  (h4 : d / c = 7 * y / (3 * z)) : a / c = 2 * x / (15 * y) :=
sorry

end ratio_a_to_c_l1_1088


namespace find_f_neg_6_l1_1640

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 2) / Real.log 2 + (a - 1) * x + b else -(Real.log (-x + 2) / Real.log 2 + (a - 1) * -x + b)

theorem find_f_neg_6 (a b : ℝ) (h1 : ∀ x : ℝ, f x a b = -f (-x) a b) 
                     (h2 : ∀ x : ℝ, x ≥ 0 → f x a b = Real.log (x + 2) / Real.log 2 + (a - 1) * x + b)
                     (h3 : f 2 a b = -1) : f (-6) 0 (-1) = 4 :=
by
  sorry

end find_f_neg_6_l1_1640


namespace inverse_proportion_points_l1_1217

theorem inverse_proportion_points (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : x2 > 0)
  (h3 : y1 = -8 / x1)
  (h4 : y2 = -8 / x2) :
  y2 < 0 ∧ 0 < y1 :=
by
  sorry

end inverse_proportion_points_l1_1217


namespace triangle_area_ratio_l1_1639

noncomputable def area_ratio (AD DC : ℝ) (h : ℝ) : ℝ :=
  (1 / 2) * AD * h / ((1 / 2) * DC * h)

theorem triangle_area_ratio (AD DC : ℝ) (h : ℝ) (condition1 : AD = 5) (condition2 : DC = 7) :
  area_ratio AD DC h = 5 / 7 :=
by
  sorry

end triangle_area_ratio_l1_1639


namespace english_class_students_l1_1694

variables (e f s u v w : ℕ)

theorem english_class_students
  (h1 : e + u + v + w + f + s + 2 = 40)
  (h2 : e + u + v = 3 * (f + w))
  (h3 : e + u + w = 2 * (s + v)) : 
  e = 30 := 
sorry

end english_class_students_l1_1694


namespace complaints_over_3_days_l1_1353

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l1_1353


namespace value_of_x_l1_1973

noncomputable def sum_integers_30_to_50 : ℕ :=
  (50 - 30 + 1) * (30 + 50) / 2

def even_count_30_to_50 : ℕ :=
  11

theorem value_of_x 
  (x := sum_integers_30_to_50)
  (y := even_count_30_to_50)
  (h : x + y = 851) : x = 840 :=
sorry

end value_of_x_l1_1973


namespace no_valid_n_l1_1199

theorem no_valid_n : ¬ ∃ (n : ℕ), (n > 0) ∧ (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by {
  sorry
}

end no_valid_n_l1_1199


namespace remainder_of_largest_divided_by_second_smallest_l1_1100

theorem remainder_of_largest_divided_by_second_smallest 
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  c % b = 1 :=
by
  -- We assume and/or prove the necessary statements here.
  -- The core of the proof uses existing facts or assumptions.
  -- We insert the proof strategy or intermediate steps here.
  
  sorry

end remainder_of_largest_divided_by_second_smallest_l1_1100


namespace distance_from_dormitory_to_city_l1_1662

theorem distance_from_dormitory_to_city (D : ℝ) 
  (h : (1/5) * D + (2/3) * D + 4 = D) : 
  D = 30 :=
sorry

end distance_from_dormitory_to_city_l1_1662


namespace base8_operations_l1_1192

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations_l1_1192


namespace ellipse_foci_x_axis_l1_1084

theorem ellipse_foci_x_axis (m n : ℝ) (h_eq : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)
  (h_foci : ∃ (c : ℝ), c = 0 ∧ (c^2 = 1 - n/m)) : n > m ∧ m > 0 ∧ n > 0 :=
sorry

end ellipse_foci_x_axis_l1_1084


namespace annual_interest_rate_l1_1690

-- Definitions based on conditions
def initial_amount : ℝ := 1000
def spent_amount : ℝ := 440
def final_amount : ℝ := 624

-- The main theorem
theorem annual_interest_rate (x : ℝ) : 
  (initial_amount * (1 + x) - spent_amount) * (1 + x) = final_amount →
  x = 0.04 :=
by
  intro h
  sorry

end annual_interest_rate_l1_1690


namespace odd_factors_360_l1_1086

theorem odd_factors_360 : 
  (∃ n m : ℕ, 360 = 2^3 * 3^n * 5^m ∧ n ≤ 2 ∧ m ≤ 1) ↔ (∃ k : ℕ, k = 6) :=
sorry

end odd_factors_360_l1_1086


namespace trigonometric_identity_l1_1512

variable {θ u : ℝ} {n : ℤ}

-- Given condition
def cos_condition (θ u : ℝ) : Prop := 2 * Real.cos θ = u + (1 / u)

-- Theorem to prove
theorem trigonometric_identity (h : cos_condition θ u) : 2 * Real.cos (n * θ) = u^n + (1 / u^n) :=
sorry

end trigonometric_identity_l1_1512


namespace max_knights_between_knights_l1_1634

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l1_1634


namespace symmetric_point_of_A_is_correct_l1_1679

def symmetric_point_with_respect_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_of_A_is_correct :
  symmetric_point_with_respect_to_x_axis (3, 4) = (3, -4) :=
by
  sorry

end symmetric_point_of_A_is_correct_l1_1679


namespace certain_number_div_5000_l1_1311

theorem certain_number_div_5000 (num : ℝ) (h : num / 5000 = 0.0114) : num = 57 :=
sorry

end certain_number_div_5000_l1_1311


namespace seventieth_even_integer_l1_1721

theorem seventieth_even_integer : 2 * 70 = 140 :=
by
  sorry

end seventieth_even_integer_l1_1721


namespace calculate_value_l1_1722

theorem calculate_value (x y : ℝ) (h : 2 * x + y = 6) : 
    ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 :=
by 
  sorry

end calculate_value_l1_1722


namespace samir_climbed_318_stairs_l1_1732

theorem samir_climbed_318_stairs 
  (S : ℕ)
  (h1 : ∀ {V : ℕ}, V = (S / 2) + 18 → S + V = 495) 
  (half_S : ∃ k : ℕ, S = k * 2) -- assumes S is even 
  : S = 318 := 
by
  sorry

end samir_climbed_318_stairs_l1_1732


namespace total_volume_of_water_in_container_l1_1276

def volume_each_hemisphere : ℝ := 4
def number_of_hemispheres : ℝ := 2735

theorem total_volume_of_water_in_container :
  (volume_each_hemisphere * number_of_hemispheres) = 10940 :=
by
  sorry

end total_volume_of_water_in_container_l1_1276


namespace distance_from_center_to_line_l1_1196

-- Define the conditions 
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

-- Define the assertion that we want to prove
theorem distance_from_center_to_line (ρ θ : ℝ) 
  (h_circle: circle_polar_eq ρ θ) 
  (h_line: line_polar_eq ρ θ) : 
  ∃ d : ℝ, d = (Real.sqrt 5) / 5 := 
sorry

end distance_from_center_to_line_l1_1196


namespace find_multiplier_l1_1161

-- Define the numbers and the equation based on the conditions
def n : ℝ := 3.0
def m : ℝ := 7

-- State the problem in Lean 4
theorem find_multiplier : m * n = 3 * n + 12 := by
  -- Specific steps skipped; only structure is needed
  sorry

end find_multiplier_l1_1161


namespace sum_3x_4y_l1_1577

theorem sum_3x_4y (x y N : ℝ) (H1 : 3 * x + 4 * y = N) (H2 : 6 * x - 4 * y = 12) (H3 : x * y = 72) : 3 * x + 4 * y = 60 := 
sorry

end sum_3x_4y_l1_1577


namespace each_niece_gets_fifty_ice_cream_sandwiches_l1_1707

theorem each_niece_gets_fifty_ice_cream_sandwiches
  (total_sandwiches : ℕ)
  (total_nieces : ℕ)
  (h1 : total_sandwiches = 1857)
  (h2 : total_nieces = 37) :
  (total_sandwiches / total_nieces) = 50 :=
by
  sorry

end each_niece_gets_fifty_ice_cream_sandwiches_l1_1707


namespace probability_neither_red_nor_purple_l1_1787

theorem probability_neither_red_nor_purple
  (total_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)
  (yellow_balls : ℕ)
  (red_balls : ℕ)
  (purple_balls : ℕ)
  (h_total : total_balls = 60)
  (h_white : white_balls = 22)
  (h_green : green_balls = 18)
  (h_yellow : yellow_balls = 17)
  (h_red : red_balls = 3)
  (h_purple : purple_balls = 1) :
  ((total_balls - red_balls - purple_balls) / total_balls : ℚ) = 14 / 15 :=
by
  sorry

end probability_neither_red_nor_purple_l1_1787


namespace solve_inequality_l1_1413

variable {c : ℝ}
variable (h_c_ne_2 : c ≠ 2)

theorem solve_inequality :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - (1 + 2) * x + 2 ≤ 0) ∧
  (c > 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x > c ∨ x < 2)) ∧
  (c < 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x < c ∨ x > 2)) :=
by
  sorry

end solve_inequality_l1_1413


namespace a100_gt_2pow99_l1_1087

theorem a100_gt_2pow99 (a : Fin 101 → ℕ) 
  (h_pos : ∀ i, a i > 0) 
  (h_initial : a 1 > a 0) 
  (h_rec : ∀ k, 2 ≤ k → a k = 3 * a (k - 1) - 2 * a (k - 2)) 
  : a 100 > 2 ^ 99 :=
by
  sorry

end a100_gt_2pow99_l1_1087


namespace calculate_selling_price_l1_1849

theorem calculate_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1500 → 
  loss_percentage = 0.17 →
  selling_price = cost_price - (loss_percentage * cost_price) →
  selling_price = 1245 :=
by 
  intros hc hl hs
  rw [hc, hl] at hs
  norm_num at hs
  exact hs

end calculate_selling_price_l1_1849


namespace gcd_lcm_240_l1_1941

theorem gcd_lcm_240 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 240) : 
  ∃ n, ∃ gcds : Finset ℕ, (gcds.card = n) ∧ (Nat.gcd a b ∈ gcds) :=
by
  sorry

end gcd_lcm_240_l1_1941


namespace sufficient_but_not_necessary_condition_l1_1749

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → a > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1_1749


namespace simultaneous_inequalities_l1_1776

theorem simultaneous_inequalities (x : ℝ) 
    (h1 : x^3 - 11 * x^2 + 10 * x < 0) 
    (h2 : x^3 - 12 * x^2 + 32 * x > 0) : 
    (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
sorry

end simultaneous_inequalities_l1_1776


namespace total_cost_is_2160_l1_1152

variables (x y z : ℝ)

-- Conditions
def cond1 : Prop := x = 0.45 * y
def cond2 : Prop := y = 0.8 * z
def cond3 : Prop := z = x + 640

-- Goal
def total_cost := x + y + z

theorem total_cost_is_2160 (x y z : ℝ) (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 x z) :
  total_cost x y z = 2160 :=
by
  sorry

end total_cost_is_2160_l1_1152


namespace productivity_increase_l1_1863

theorem productivity_increase (a b : ℝ) : (7 / 8) * (1 + 20 / 100) = 1.05 :=
by
  sorry

end productivity_increase_l1_1863


namespace x_squared_plus_y_squared_l1_1883

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := 
by
  sorry

end x_squared_plus_y_squared_l1_1883


namespace gcd_of_36_and_54_l1_1701

theorem gcd_of_36_and_54 : Nat.gcd 36 54 = 18 := by
  -- Proof details are omitted; replaced with sorry.
  sorry

end gcd_of_36_and_54_l1_1701


namespace angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1_1097

open Real

variable {A B C a b c : ℝ}
variable (AM BM MC : ℝ)

-- Conditions
axiom triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)
axiom BM_MC_relation : BM = (1 / 2) * MC

-- Part 1: Measure of angle A
theorem angle_A_is_pi_over_3 (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) : 
  A = π / 3 :=
by sorry

-- Part 2: Minimum value of |AM|^2 / S
noncomputable def area_triangle (a b c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * sin A

axiom condition_b_eq_2c : b = 2 * c

theorem minimum_value_AM_sq_div_S (AM BM MC : ℝ) (S : ℝ) (H : BM = (1 / 2) * MC) 
  (triangle_sides : b / (sin A + sin C) = (a - c) / (sin B - sin C)) 
  (area : S = area_triangle a b c A)
  (condition_b_eq_2c : b = 2 * c) : 
  (AM ^ 2) / S ≥ (8 * sqrt 3) / 9 :=
by sorry

end angle_A_is_pi_over_3_minimum_value_AM_sq_div_S_l1_1097


namespace smallest_x_abs_eq_15_l1_1854

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, (|x - 8| = 15) ∧ ∀ y : ℝ, (|y - 8| = 15) → y ≥ x :=
sorry

end smallest_x_abs_eq_15_l1_1854


namespace shoe_cost_on_monday_l1_1093

theorem shoe_cost_on_monday 
  (price_thursday : ℝ) 
  (increase_rate : ℝ) 
  (decrease_rate : ℝ) 
  (price_thursday_eq : price_thursday = 40)
  (increase_rate_eq : increase_rate = 0.10)
  (decrease_rate_eq : decrease_rate = 0.10)
  :
  let price_friday := price_thursday * (1 + increase_rate)
  let discount := price_friday * decrease_rate
  let price_monday := price_friday - discount
  price_monday = 39.60 :=
by
  sorry

end shoe_cost_on_monday_l1_1093


namespace difference_of_roots_l1_1235

theorem difference_of_roots 
  (a b c : ℝ)
  (h : ∀ x, x^2 - 2 * (a^2 + b^2 + c^2 - 2 * a * c) * x + (b^2 - a^2 - c^2 + 2 * a * c)^2 = 0) :
  ∃ (x1 x2 : ℝ), (x1 - x2 = 4 * b * (a - c)) ∨ (x1 - x2 = -4 * b * (a - c)) :=
sorry

end difference_of_roots_l1_1235


namespace perpendicular_line_through_center_l1_1667

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * x + y^2 - 3 = 0

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the line we want to prove passes through the center of the circle and is perpendicular to the given line
def wanted_line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Main statement: Prove that the line that passes through the center of the circle and is perpendicular to the given line has the equation x - y + 1 = 0
theorem perpendicular_line_through_center (x y : ℝ) :
  (circle_eq (-1) 0) ∧ (line_eq x y) → wanted_line_eq x y :=
by
  sorry

end perpendicular_line_through_center_l1_1667


namespace largest_value_among_given_numbers_l1_1056

theorem largest_value_among_given_numbers :
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  d >= a ∧ d >= b ∧ d >= c ∧ d >= e :=
by
  let a := 2 * 0 * 2006
  let b := 2 * 0 + 6
  let c := 2 + 0 * 2006
  let d := 2 * (0 + 6)
  let e := 2006 * 0 + 0 * 6
  sorry

end largest_value_among_given_numbers_l1_1056


namespace x_plus_y_value_l1_1608

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem x_plus_y_value :
  let x := sum_of_integers 50 70
  let y := count_even_integers 50 70
  x + y = 1271 := by
    let x := sum_of_integers 50 70
    let y := count_even_integers 50 70
    sorry

end x_plus_y_value_l1_1608


namespace georgie_enter_and_exit_ways_l1_1978

-- Define the number of windows
def num_windows := 8

-- Define the magical barrier window
def barrier_window := 8

-- Define a function to count the number of ways Georgie can enter and exit the house
def count_ways_to_enter_and_exit : Nat :=
  let entry_choices := num_windows
  let exit_choices_from_normal := 6
  let exit_choices_from_barrier := 7
  let ways_from_normal := (entry_choices - 1) * exit_choices_from_normal  -- entering through windows 1 to 7
  let ways_from_barrier := 1 * exit_choices_from_barrier  -- entering through window 8
  ways_from_normal + ways_from_barrier

-- Prove the correct number of ways is 49
theorem georgie_enter_and_exit_ways : count_ways_to_enter_and_exit = 49 :=
by
  -- The calculation details are skipped with 'sorry'
  sorry

end georgie_enter_and_exit_ways_l1_1978


namespace smallest_positive_angle_l1_1462

theorem smallest_positive_angle :
  ∀ (x : ℝ), 12 * (Real.sin x)^3 * (Real.cos x)^3 - 2 * (Real.sin x)^3 * (Real.cos x)^3 = 1 → 
  x = 15 * (Real.pi / 180) :=
by
  intros x h
  sorry

end smallest_positive_angle_l1_1462


namespace more_balloons_l1_1421

theorem more_balloons (you_balloons : ℕ) (friend_balloons : ℕ) (h_you : you_balloons = 7) (h_friend : friend_balloons = 5) : 
  you_balloons - friend_balloons = 2 :=
sorry

end more_balloons_l1_1421


namespace decryption_proof_l1_1025

-- Definitions
def Original_Message := "МОСКВА"
def Encrypted_Text_1 := "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ"
def Encrypted_Text_2 := "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП"
def Encrypted_Text_3 := "РТПАИОМВСВТИЕОБПРОЕННИИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК"

noncomputable def Encrypted_Message_1 := "ЙМЫВОТСЬЛКЪГВЦАЯЯ"
noncomputable def Encrypted_Message_2 := "УКМАПОЧСРКЩВЗАХ"
noncomputable def Encrypted_Message_3 := "ШМФЭОГЧСЙЪКФЬВЫЕАКК"

def Decrypted_Message_1_and_3 := "ПОВТОРЕНИЕМАТЬУЧЕНИЯ"
def Decrypted_Message_2 := "СМОТРИВКОРЕНЬ"

-- Theorem statement
theorem decryption_proof :
  (Encrypted_Text_1 = Encrypted_Text_3 ∧ Original_Message = "МОСКВА" ∧ Encrypted_Message_1 = Encrypted_Message_3) →
  (Decrypted_Message_1_and_3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ Decrypted_Message_2 = "СМОТРИВКОРЕНЬ") :=
by 
  sorry

end decryption_proof_l1_1025


namespace de_morgan_implication_l1_1756

variables (p q : Prop)

theorem de_morgan_implication (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
sorry

end de_morgan_implication_l1_1756


namespace ab_conditions_l1_1474

theorem ab_conditions (a b : ℝ) : ¬((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) :=
by 
  sorry

end ab_conditions_l1_1474


namespace study_days_needed_l1_1041

theorem study_days_needed :
  let math_chapters := 4
  let math_worksheets := 7
  let physics_chapters := 5
  let physics_worksheets := 9
  let chemistry_chapters := 6
  let chemistry_worksheets := 8

  let math_chapter_hours := 2.5
  let math_worksheet_hours := 1.5
  let physics_chapter_hours := 3.0
  let physics_worksheet_hours := 2.0
  let chemistry_chapter_hours := 3.5
  let chemistry_worksheet_hours := 1.75

  let daily_study_hours := 7.0
  let breaks_first_3_hours := 3 * 10 / 60.0
  let breaks_next_3_hours := 3 * 15 / 60.0
  let breaks_final_hour := 1 * 20 / 60.0
  let snack_breaks := 2 * 20 / 60.0
  let lunch_break := 45 / 60.0

  let break_time_per_day := breaks_first_3_hours + breaks_next_3_hours + breaks_final_hour + snack_breaks + lunch_break
  let effective_study_time_per_day := daily_study_hours - break_time_per_day

  let total_math_hours := (math_chapters * math_chapter_hours) + (math_worksheets * math_worksheet_hours)
  let total_physics_hours := (physics_chapters * physics_chapter_hours) + (physics_worksheets * physics_worksheet_hours)
  let total_chemistry_hours := (chemistry_chapters * chemistry_chapter_hours) + (chemistry_worksheets * chemistry_worksheet_hours)

  let total_study_hours := total_math_hours + total_physics_hours + total_chemistry_hours
  let total_study_days := total_study_hours / effective_study_time_per_day
  
  total_study_days.ceil = 23 := by sorry

end study_days_needed_l1_1041


namespace greatest_s_property_l1_1255

noncomputable def find_greatest_s (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] : ℕ :=
if h : m > 0 ∧ n > 0 then m else 0

theorem greatest_s_property (m n : ℕ) (p : ℕ) [Fact (Nat.Prime p)] (H : 0 < m) (H1 : 0 < n)  :
  ∃ s, (s = find_greatest_s m n p) ∧ s * n * p ≤ m * n * p :=
by 
  sorry

end greatest_s_property_l1_1255


namespace sum_of_series_l1_1530

def series_sum : ℕ := 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))

theorem sum_of_series : series_sum = 2730 := by
  -- Expansion: 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))) = 2 + 2 * 4 + 2 * 4^2 + 2 * 4^3 + 2 * 4^4 + 2 * 4^5 
  -- Geometric series sum formula application: S = 2 + 2*4 + 2*4^2 + 2*4^3 + 2*4^4 + 2*4^5 = 2730
  sorry

end sum_of_series_l1_1530


namespace stratified_sampling_seniors_l1_1657

theorem stratified_sampling_seniors
  (total_students : ℕ)
  (seniors : ℕ)
  (sample_size : ℕ)
  (senior_sample_size : ℕ)
  (h1 : total_students = 4500)
  (h2 : seniors = 1500)
  (h3 : sample_size = 300)
  (h4 : senior_sample_size = seniors * sample_size / total_students) :
  senior_sample_size = 100 :=
  sorry

end stratified_sampling_seniors_l1_1657


namespace equation1_solution_equation2_solution_l1_1820

theorem equation1_solution (x : ℝ) : (x - 4)^2 - 9 = 0 ↔ (x = 7 ∨ x = 1) := 
sorry

theorem equation2_solution (x : ℝ) : (x + 1)^3 = -27 ↔ (x = -4) := 
sorry

end equation1_solution_equation2_solution_l1_1820


namespace simplify_fraction_l1_1197

theorem simplify_fraction (c : ℚ) : (6 + 5 * c) / 9 + 3 = (33 + 5 * c) / 9 := 
sorry

end simplify_fraction_l1_1197


namespace remainder_when_M_divided_by_52_l1_1735

def M : Nat := 123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051

theorem remainder_when_M_divided_by_52 : M % 52 = 0 :=
by
  sorry

end remainder_when_M_divided_by_52_l1_1735


namespace complex_number_solution_l1_1333

theorem complex_number_solution (z : ℂ) (i : ℂ) (H1 : i * i = -1) (H2 : z * i = 2 - 2 * i) : z = -2 - 2 * i :=
by
  sorry

end complex_number_solution_l1_1333


namespace evaluate_integral_l1_1660

noncomputable def integral_problem : Real :=
  ∫ x in (-2 : Real)..(2 : Real), (Real.sqrt (4 - x^2) - x^2017)

theorem evaluate_integral :
  integral_problem = 2 * Real.pi :=
sorry

end evaluate_integral_l1_1660


namespace frequency_of_a_is_3_l1_1672

def sentence : String := "Happy Teachers'Day!"

def frequency_of_a_in_sentence (s : String) : Nat :=
  s.foldl (λ acc c => if c = 'a' then acc + 1 else acc) 0

theorem frequency_of_a_is_3 : frequency_of_a_in_sentence sentence = 3 :=
  by
    sorry

end frequency_of_a_is_3_l1_1672


namespace water_left_l1_1455

-- Conditions
def initial_water : ℚ := 3
def water_used : ℚ := 11 / 8

-- Proposition to be proven
theorem water_left :
  initial_water - water_used = 13 / 8 := by
  sorry

end water_left_l1_1455


namespace solve_xyz_l1_1808

def is_solution (x y z : ℕ) : Prop :=
  x * y + y * z + z * x = 2 * (x + y + z)

theorem solve_xyz (x y z : ℕ) :
  is_solution x y z ↔ (x = 1 ∧ y = 2 ∧ z = 4) ∨
                     (x = 1 ∧ y = 4 ∧ z = 2) ∨
                     (x = 2 ∧ y = 1 ∧ z = 4) ∨
                     (x = 2 ∧ y = 4 ∧ z = 1) ∨
                     (x = 2 ∧ y = 2 ∧ z = 2) ∨
                     (x = 4 ∧ y = 1 ∧ z = 2) ∨
                     (x = 4 ∧ y = 2 ∧ z = 1) := sorry

end solve_xyz_l1_1808


namespace total_questions_in_two_hours_l1_1355

theorem total_questions_in_two_hours (r : ℝ) : 
  let Fiona_questions := 36 
  let Shirley_questions := Fiona_questions * r
  let Kiana_questions := (Fiona_questions + Shirley_questions) / 2
  let one_hour_total := Fiona_questions + Shirley_questions + Kiana_questions
  let two_hour_total := 2 * one_hour_total
  two_hour_total = 108 + 108 * r :=
by
  sorry

end total_questions_in_two_hours_l1_1355


namespace min_value_reciprocal_sum_l1_1184

theorem min_value_reciprocal_sum (m n : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) :
  1 / m + 1 / n ≥ 4 :=
sorry

end min_value_reciprocal_sum_l1_1184


namespace simplify_expression_l1_1304

variable {x : ℝ}

theorem simplify_expression : 8 * x - 3 + 2 * x - 7 + 4 * x + 15 = 14 * x + 5 :=
by
  sorry

end simplify_expression_l1_1304


namespace alice_favorite_number_l1_1966

-- Define the conditions for Alice's favorite number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

-- Define the problem statement
theorem alice_favorite_number :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
           n % 13 = 0 ∧
           n % 3 ≠ 0 ∧
           sum_of_digits n % 4 = 0 ∧
           n = 130 :=
by
  sorry

end alice_favorite_number_l1_1966


namespace unique_point_graph_eq_l1_1991

theorem unique_point_graph_eq (c : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → x = -1 ∧ y = 6) ↔ c = 39 :=
sorry

end unique_point_graph_eq_l1_1991


namespace OfficerHoppsTotalTickets_l1_1800

theorem OfficerHoppsTotalTickets : 
  (15 * 8 + (31 - 15) * 5 = 200) :=
  by
    sorry

end OfficerHoppsTotalTickets_l1_1800


namespace bonus_points_amount_l1_1247

def points_per_10_dollars : ℕ := 50

def beef_price : ℕ := 11
def beef_quantity : ℕ := 3

def fruits_vegetables_price : ℕ := 4
def fruits_vegetables_quantity : ℕ := 8

def spices_price : ℕ := 6
def spices_quantity : ℕ := 3

def other_groceries_total : ℕ := 37

def total_points : ℕ := 850

def total_spent : ℕ :=
  (beef_price * beef_quantity) +
  (fruits_vegetables_price * fruits_vegetables_quantity) +
  (spices_price * spices_quantity) +
  other_groceries_total

def points_from_spending : ℕ :=
  (total_spent / 10) * points_per_10_dollars

theorem bonus_points_amount :
  total_spent > 100 → total_points - points_from_spending = 250 :=
by
  sorry

end bonus_points_amount_l1_1247


namespace glorias_ratio_l1_1516

variable (Q : ℕ) -- total number of quarters
variable (dimes : ℕ) -- total number of dimes, given as 350
variable (quarters_left : ℕ) -- number of quarters left

-- Given conditions
def conditions (Q dimes quarters_left : ℕ) : Prop :=
  dimes = 350 ∧
  quarters_left = (3 * Q) / 5 ∧
  (dimes + quarters_left = 392)

-- The ratio of dimes to quarters left
def ratio_of_dimes_to_quarters_left (dimes quarters_left : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd dimes quarters_left
  (dimes / gcd, quarters_left / gcd)

theorem glorias_ratio (Q : ℕ) (quarters_left : ℕ) : conditions Q 350 quarters_left → ratio_of_dimes_to_quarters_left 350 quarters_left = (25, 3) := by 
  sorry

end glorias_ratio_l1_1516


namespace fifth_dog_weight_l1_1453

theorem fifth_dog_weight (y : ℝ) (h : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y) / 5) : y = 31 :=
by
  sorry

end fifth_dog_weight_l1_1453


namespace maximize_profit_l1_1888

noncomputable def g (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 13.5 - (1 / 30) * x^2
else if h : (x > 10) then 168 / x - 2000 / (3 * x^2)
else 0 -- default case included for totality

noncomputable def y (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then 8.1 * x - (1 / 30) * x^3 - 20
else if h : (x > 10) then 148 - 2 * (1000 / (3 * x) + 2.7 * x)
else 0 -- default case included for totality

theorem maximize_profit (x : ℝ) : 0 < x → y 9 = 28.6 :=
by sorry

end maximize_profit_l1_1888


namespace range_of_a_increasing_function_l1_1174

noncomputable def f (x a : ℝ) := x^3 + a * x + 1 / x

noncomputable def f' (x a : ℝ) := 3 * x^2 - 1 / x^2 + a

theorem range_of_a_increasing_function (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → f' x a ≥ 0) ↔ a ≥ 13 / 4 := 
sorry

end range_of_a_increasing_function_l1_1174


namespace pirate_treasure_division_l1_1237

theorem pirate_treasure_division (initial_treasure : ℕ) (p1_share p2_share p3_share p4_share p5_share remaining : ℕ)
  (h_initial : initial_treasure = 3000)
  (h_p1_share : p1_share = initial_treasure / 10)
  (h_p1_rem : remaining = initial_treasure - p1_share)
  (h_p2_share : p2_share = 2 * remaining / 10)
  (h_p2_rem : remaining = remaining - p2_share)
  (h_p3_share : p3_share = 3 * remaining / 10)
  (h_p3_rem : remaining = remaining - p3_share)
  (h_p4_share : p4_share = 4 * remaining / 10)
  (h_p4_rem : remaining = remaining - p4_share)
  (h_p5_share : p5_share = 5 * remaining / 10)
  (h_p5_rem : remaining = remaining - p5_share)
  (p6_p9_total : ℕ)
  (h_p6_p9_total : p6_p9_total = 20 * 4)
  (final_remaining : ℕ)
  (h_final_remaining : final_remaining = remaining - p6_p9_total) :
  final_remaining = 376 :=
by sorry

end pirate_treasure_division_l1_1237


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l1_1802

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l1_1802


namespace initial_masses_l1_1723

def area_of_base : ℝ := 15
def density_water : ℝ := 1
def density_ice : ℝ := 0.92
def change_in_water_level : ℝ := 5
def final_height_of_water : ℝ := 115

theorem initial_masses (m_ice m_water : ℝ) :
  m_ice = 675 ∧ m_water = 1050 :=
by
  -- Calculate the change in volume of water
  let delta_v := area_of_base * change_in_water_level

  -- Relate this volume change to the volume difference between ice and water
  let lhs := m_ice / density_ice - m_ice / density_water
  let eq1 := delta_v

  -- Solve for the mass of ice
  have h_ice : m_ice = 675 := 
  sorry

  -- Determine the final volume of water
  let final_volume_of_water := final_height_of_water * area_of_base

  -- Determine the initial mass of water
  let mass_of_water_total := density_water * final_volume_of_water
  let initial_mass_of_water :=
    mass_of_water_total - m_ice

  have h_water : m_water = 1050 := 
  sorry

  exact ⟨h_ice, h_water⟩

end initial_masses_l1_1723


namespace final_ranking_l1_1954

-- Define data types for participants and their initial positions
inductive Participant
| X
| Y
| Z

open Participant

-- Define the initial conditions and number of position changes
def initial_positions : List Participant := [X, Y, Z]

def position_changes : Participant → Nat
| X => 5
| Y => 0  -- Not given explicitly but derived from the conditions.
| Z => 6

-- Final condition stating Y finishes before X
def Y_before_X : Prop := True

-- The theorem stating the final ranking
theorem final_ranking :
  Y_before_X →
  (initial_positions = [X, Y, Z]) →
  (position_changes X = 5) →
  (position_changes Z = 6) →
  (position_changes Y = 0) →
  [Y, X, Z] = [Y, X, Z] :=
by
  intros
  exact rfl

end final_ranking_l1_1954


namespace max_songs_played_l1_1594

theorem max_songs_played (n m t : ℕ) (h1 : n = 50) (h2 : m = 50) (h3 : t = 180) :
  3 * n + 5 * (m - ((t - 3 * n) / 5)) = 56 :=
by
  sorry

end max_songs_played_l1_1594


namespace sequence_length_l1_1884

theorem sequence_length (a : ℕ) (h : a = 10800) (h1 : ∀ n, (n ≠ 0 → ∃ m, n = 2 * m ∧ m ≠ 0) ∧ 2 ∣ n)
  : ∃ k : ℕ, k = 5 := 
sorry

end sequence_length_l1_1884


namespace square_can_be_divided_into_40_smaller_squares_l1_1885

theorem square_can_be_divided_into_40_smaller_squares 
: ∃ (n : ℕ), n * n = 40 := 
sorry

end square_can_be_divided_into_40_smaller_squares_l1_1885


namespace matrix_cubic_l1_1921

noncomputable def matrix_entries (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

theorem matrix_cubic (x y z : ℝ) (N : Matrix (Fin 3) (Fin 3) ℝ)
    (hN : N = matrix_entries x y z)
    (hn : N ^ 2 = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ))
    (hxyz : x * y * z = -2) :
  x^3 + y^3 + z^3 = -6 + 2 * Real.sqrt 2 ∨ x^3 + y^3 + z^3 = -6 - 2 * Real.sqrt 2 :=
by
  sorry

end matrix_cubic_l1_1921


namespace find_principal_l1_1116

/-- Given that the simple interest SI is Rs. 90, the rate R is 3.5 percent, and the time T is 4 years,
prove that the principal P is approximately Rs. 642.86 using the simple interest formula. -/
theorem find_principal
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 90) (h2 : R = 3.5) (h3 : T = 4) 
  : P = 90 * 100 / (3.5 * 4) :=
by
  sorry

end find_principal_l1_1116


namespace equal_water_and_alcohol_l1_1632

variable (a m : ℝ)

-- Conditions:
-- Cup B initially contains m liters of water.
-- Transfers as specified in the problem.

theorem equal_water_and_alcohol (h : m > 0) :
  (a * (m / (m + a)) = a * (m / (m + a))) :=
by
  sorry

end equal_water_and_alcohol_l1_1632


namespace find_a_l1_1797

theorem find_a (a : ℝ) : 
  (∃ (a : ℝ), a * 15 + 6 = -9) → a = -1 :=
by
  intro h
  sorry

end find_a_l1_1797


namespace total_crayons_lost_or_given_away_l1_1076

/-
Paul gave 52 crayons to his friends.
Paul lost 535 crayons.
Paul had 492 crayons left.
Prove that the total number of crayons lost or given away is 587.
-/
theorem total_crayons_lost_or_given_away
  (crayons_given : ℕ)
  (crayons_lost : ℕ)
  (crayons_left : ℕ)
  (h_crayons_given : crayons_given = 52)
  (h_crayons_lost : crayons_lost = 535)
  (h_crayons_left : crayons_left = 492) :
  crayons_given + crayons_lost = 587 := 
sorry

end total_crayons_lost_or_given_away_l1_1076


namespace length_of_platform_is_280_l1_1534

-- Add conditions for speed, times and conversions
def speed_kmph : ℕ := 72
def time_platform : ℕ := 30
def time_man : ℕ := 16

-- Conversion from km/h to m/s
def speed_mps : ℤ := speed_kmph * 1000 / 3600

-- The length of the train when it crosses the man
def length_of_train : ℤ := speed_mps * time_man

-- The length of the platform
def length_of_platform : ℤ := (speed_mps * time_platform) - length_of_train

theorem length_of_platform_is_280 :
  length_of_platform = 280 := by
  sorry

end length_of_platform_is_280_l1_1534


namespace largest_e_l1_1018

variable (a b c d e : ℤ)

theorem largest_e 
  (h1 : a - 1 = b + 2) 
  (h2 : a - 1 = c - 3)
  (h3 : a - 1 = d + 4)
  (h4 : a - 1 = e - 6) 
  : e > a ∧ e > b ∧ e > c ∧ e > d := 
sorry

end largest_e_l1_1018


namespace unbroken_seashells_left_l1_1631

-- Definitions based on given conditions
def total_seashells : ℕ := 6
def cone_shells : ℕ := 3
def conch_shells : ℕ := 3
def broken_cone_shells : ℕ := 2
def broken_conch_shells : ℕ := 2
def given_away_conch_shells : ℕ := 1

-- Mathematical statement to prove the final count of unbroken seashells
theorem unbroken_seashells_left : 
  (cone_shells - broken_cone_shells) + (conch_shells - broken_conch_shells - given_away_conch_shells) = 1 :=
by 
  -- Calculation (steps omitted per instructions)
  sorry

end unbroken_seashells_left_l1_1631


namespace find_x_squared_plus_y_squared_l1_1860

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 :=
by
  sorry

end find_x_squared_plus_y_squared_l1_1860


namespace equal_probability_of_selection_l1_1784

-- Define the number of total students
def total_students : ℕ := 86

-- Define the number of students to be eliminated through simple random sampling
def eliminated_students : ℕ := 6

-- Define the number of students selected through systematic sampling
def selected_students : ℕ := 8

-- Define the probability calculation
def probability_not_eliminated : ℚ := 80 / 86
def probability_selected : ℚ := 8 / 80
def combined_probability : ℚ := probability_not_eliminated * probability_selected

theorem equal_probability_of_selection :
  combined_probability = 4 / 43 :=
by
  -- We do not need to complete the proof as per instruction
  sorry

end equal_probability_of_selection_l1_1784


namespace employee_payment_l1_1173

theorem employee_payment
  (A B C : ℝ)
  (h_total : A + B + C = 1500)
  (h_A : A = 1.5 * B)
  (h_C : C = 0.8 * B) :
  A = 682 ∧ B = 454 ∧ C = 364 := by
  sorry

end employee_payment_l1_1173


namespace triangle_constructibility_l1_1272

variables (a b c γ : ℝ)

-- definition of the problem conditions
def valid_triangle_constructibility_conditions (a b_c_diff γ : ℝ) : Prop :=
  γ < 90 ∧ b_c_diff < a * Real.cos γ

-- constructibility condition
def is_constructible (a b c γ : ℝ) : Prop :=
  b - c < a * Real.cos γ

-- final theorem statement
theorem triangle_constructibility (a b c γ : ℝ) (h1 : γ < 90) (h2 : b > c) :
  (b - c < a * Real.cos γ) ↔ valid_triangle_constructibility_conditions a (b - c) γ :=
by sorry

end triangle_constructibility_l1_1272


namespace words_per_page_l1_1433

theorem words_per_page 
    (p : ℕ) 
    (h1 : 150 > 0) 
    (h2 : 150 * p ≡ 200 [MOD 221]) :
    p = 118 := 
by sorry

end words_per_page_l1_1433


namespace simplify_sqrt_eight_l1_1376

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry

end simplify_sqrt_eight_l1_1376


namespace ratio_of_pens_to_notebooks_is_5_to_4_l1_1720

theorem ratio_of_pens_to_notebooks_is_5_to_4 (P N : ℕ) (hP : P = 50) (hN : N = 40) :
  (P / Nat.gcd P N) = 5 ∧ (N / Nat.gcd P N) = 4 :=
by
  -- Proof goes here
  sorry

end ratio_of_pens_to_notebooks_is_5_to_4_l1_1720


namespace tommy_initial_balloons_l1_1812

theorem tommy_initial_balloons (initial_balloons balloons_added total_balloons : ℝ)
  (h1 : balloons_added = 34.5)
  (h2 : total_balloons = 60.75)
  (h3 : total_balloons = initial_balloons + balloons_added) :
  initial_balloons = 26.25 :=
by sorry

end tommy_initial_balloons_l1_1812


namespace function_pair_solution_l1_1553

-- Define the conditions for f and g
variables (f g : ℝ → ℝ)

-- Define the main hypothesis
def main_hypothesis : Prop := 
∀ (x y : ℝ), 
  x ≠ 0 → y ≠ 0 → 
  f (x + y) = g (1/x + 1/y) * (x * y) ^ 2008

-- The theorem that proves f and g are of the given form
theorem function_pair_solution (c : ℝ) (h : main_hypothesis f g) : 
  (∀ x, f x = c * x ^ 2008) ∧ 
  (∀ x, g x = c * x ^ 2008) :=
sorry

end function_pair_solution_l1_1553


namespace candidates_count_l1_1288

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 90) : n = 10 :=
by
  sorry

end candidates_count_l1_1288


namespace fraction_inequality_l1_1297

theorem fraction_inequality (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b / a) < ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l1_1297


namespace ellipse_area_l1_1410

theorem ellipse_area
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (a : { endpoints_major_axis : (ℝ × ℝ) × (ℝ × ℝ) // endpoints_major_axis = ((x1, y1), (x2, y2)) })
  (b : { point_on_ellipse : ℝ × ℝ // point_on_ellipse = (x3, y3) }) :
  (-5 : ℝ) = x1 ∧ (2 : ℝ) = y1 ∧ (15 : ℝ) = x2 ∧ (2 : ℝ) = y2 ∧
  (8 : ℝ) = x3 ∧ (6 : ℝ) = y3 → 
  100 * Real.pi * Real.sqrt (16 / 91) = 100 * Real.pi * Real.sqrt (16 / 91) :=
by
  sorry

end ellipse_area_l1_1410


namespace base_conversion_correct_l1_1925

def convert_base_9_to_10 (n : ℕ) : ℕ :=
  3 * 9^2 + 6 * 9^1 + 1 * 9^0

def convert_base_13_to_10 (n : ℕ) (C : ℕ) : ℕ :=
  4 * 13^2 + C * 13^1 + 5 * 13^0

theorem base_conversion_correct :
  convert_base_9_to_10 361 + convert_base_13_to_10 4 12 = 1135 :=
by
  sorry

end base_conversion_correct_l1_1925


namespace arithmetic_sequence_a5_l1_1572

-- Define the concept of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- The problem's conditions
def a₁ : ℕ := 2
def d : ℕ := 3

-- The proof problem
theorem arithmetic_sequence_a5 : arithmetic_sequence a₁ d 5 = 14 := by
  sorry

end arithmetic_sequence_a5_l1_1572


namespace natural_eq_rational_exists_diff_l1_1136

-- Part (a)
theorem natural_eq (x y : ℕ) (h : x^3 + y = y^3 + x) : x = y := 
by sorry

-- Part (b)
theorem rational_exists_diff (x y : ℚ) (h : x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) : ∃ (x y : ℚ), x ≠ y ∧ x^3 + y = y^3 + x := 
by sorry

end natural_eq_rational_exists_diff_l1_1136


namespace tower_height_l1_1328

theorem tower_height (h : ℝ) (hd : ¬ (h ≥ 200)) (he : ¬ (h ≤ 150)) (hf : ¬ (h ≤ 180)) : 180 < h ∧ h < 200 := 
by 
  sorry

end tower_height_l1_1328


namespace triangles_in_pentadecagon_l1_1588

theorem triangles_in_pentadecagon : (Nat.choose 15 3) = 455 :=
by
  sorry

end triangles_in_pentadecagon_l1_1588


namespace max_neg_integers_l1_1686

-- Definitions for the conditions
def areIntegers (a b c d e f : Int) : Prop := True
def sumOfProductsNeg (a b c d e f : Int) : Prop := (a * b + c * d * e * f) < 0

-- The theorem to prove
theorem max_neg_integers (a b c d e f : Int) (h1 : areIntegers a b c d e f) (h2 : sumOfProductsNeg a b c d e f) : 
  ∃ s : Nat, s = 4 := 
sorry

end max_neg_integers_l1_1686


namespace find_a_values_l1_1363

def setA (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.snd - 3) / (p.fst - 2) = a + 1}

def setB (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (a^2 - 1) * p.fst + (a - 1) * p.snd = 15}

def sets_disjoint (A B : Set (ℝ × ℝ)) : Prop := ∀ p : ℝ × ℝ, p ∉ A ∪ B

theorem find_a_values (a : ℝ) :
  sets_disjoint (setA a) (setB a) ↔ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
sorry

end find_a_values_l1_1363


namespace rain_difference_l1_1441

theorem rain_difference (r_m r_t : ℝ) (h_monday : r_m = 0.9) (h_tuesday : r_t = 0.2) : r_m - r_t = 0.7 :=
by sorry

end rain_difference_l1_1441


namespace pear_price_is_6300_l1_1609

def price_of_pear (P : ℕ) : Prop :=
  P + (P + 2400) = 15000

theorem pear_price_is_6300 : ∃ (P : ℕ), price_of_pear P ∧ P = 6300 :=
by
  sorry

end pear_price_is_6300_l1_1609


namespace div_by_240_l1_1417

theorem div_by_240 (a b c d : ℕ) : 240 ∣ (a ^ (4 * b + d) - a ^ (4 * c + d)) :=
sorry

end div_by_240_l1_1417


namespace jill_and_emily_total_peaches_l1_1821

-- Define each person and their conditions
variables (Steven Jake Jill Maria Emily : ℕ)

-- Given conditions
def steven_has_peaches : Steven = 14 := sorry
def jake_has_fewer_than_steven : Jake = Steven - 6 := sorry
def jake_has_more_than_jill : Jake = Jill + 3 := sorry
def maria_has_twice_jake : Maria = 2 * Jake := sorry
def emily_has_fewer_than_maria : Emily = Maria - 9 := sorry

-- The theorem statement combining the conditions and the required result
theorem jill_and_emily_total_peaches (Steven Jake Jill Maria Emily : ℕ)
  (h1 : Steven = 14) 
  (h2 : Jake = Steven - 6) 
  (h3 : Jake = Jill + 3) 
  (h4 : Maria = 2 * Jake) 
  (h5 : Emily = Maria - 9) : 
  Jill + Emily = 12 := 
sorry

end jill_and_emily_total_peaches_l1_1821


namespace minimum_students_for_200_candies_l1_1170

theorem minimum_students_for_200_candies (candies : ℕ) (students : ℕ) (h_candies : candies = 200) : students = 21 :=
by
  sorry

end minimum_students_for_200_candies_l1_1170


namespace crossing_time_proof_l1_1967

/-
  Problem:
  Given:
  1. length_train: 600 (length of the train in meters)
  2. time_signal_post: 40 (time taken to cross the signal post in seconds)
  3. time_bridge_minutes: 20 (time taken to cross the bridge in minutes)

  Prove:
  t_cross_bridge: the time it takes to cross the bridge and the full length of the train is 1240 seconds
-/

def length_train : ℕ := 600
def time_signal_post : ℕ := 40
def time_bridge_minutes : ℕ := 20

-- Converting time to cross the bridge from minutes to seconds
def time_bridge_seconds : ℕ := time_bridge_minutes * 60

-- Finding the speed
def speed_train : ℕ := length_train / time_signal_post

-- Finding the length of the bridge
def length_bridge : ℕ := speed_train * time_bridge_seconds

-- Finding the total distance covered
def total_distance : ℕ := length_train + length_bridge

-- Given distance and speed, find the time to cross
def time_to_cross : ℕ := total_distance / speed_train

theorem crossing_time_proof : time_to_cross = 1240 := by
  sorry

end crossing_time_proof_l1_1967


namespace Zoe_given_card_6_l1_1120

-- Define the cards and friends
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def friends : List String := ["Eliza", "Miguel", "Naomi", "Ivan", "Zoe"]

-- Define scores 
def scores (name : String) : ℕ :=
  match name with
  | "Eliza"  => 15
  | "Miguel" => 11
  | "Naomi"  => 9
  | "Ivan"   => 13
  | "Zoe"    => 10
  | _ => 0

-- Each friend is given a pair of cards
def cardAssignments (name : String) : List (ℕ × ℕ) :=
  match name with
  | "Eliza"  => [(6,9), (7,8), (5,10), (4,11), (3,12)]
  | "Miguel" => [(1,10), (2,9), (3,8), (4,7), (5,6)]
  | "Naomi"  => [(1,8), (2,7), (3,6), (4,5)]
  | "Ivan"   => [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
  | "Zoe"    => [(1,9), (2,8), (3,7), (4,6)]
  | _ => []

-- The proof statement
theorem Zoe_given_card_6 : ∃ c1 c2, (c1, c2) ∈ cardAssignments "Zoe" ∧ (c1 = 6 ∨ c2 = 6)
:= by
  sorry -- Proof omitted as per the instructions

end Zoe_given_card_6_l1_1120


namespace find_m_l1_1738

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 2, m}
def B : Set ℝ := {3, 4}

-- The intersection condition
def intersect_condition (m : ℝ) : Prop := A m ∩ B = {3}

-- The statement to prove
theorem find_m : ∃ m : ℝ, intersect_condition m → m = 3 :=
by {
  use 3,
  sorry
}

end find_m_l1_1738


namespace eccentricity_of_ellipse_l1_1251

theorem eccentricity_of_ellipse :
  ∀ (A B : ℝ × ℝ) (has_axes_intersection : A.2 = 0 ∧ B.2 = 0) 
    (product_of_slopes : ∀ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B → (P.2 / (P.1 - A.1)) * (P.2 / (P.1 + B.1)) = -1/2),
  ∃ (e : ℝ), e = 1 / Real.sqrt 2 :=
by
  sorry

end eccentricity_of_ellipse_l1_1251


namespace ratio_of_speeds_l1_1483

theorem ratio_of_speeds (P R : ℝ) (total_time : ℝ) (time_rickey : ℝ)
  (h1 : total_time = 70)
  (h2 : time_rickey = 40)
  (h3 : total_time - time_rickey = 30) :
  P / R = 3 / 4 :=
by
  sorry

end ratio_of_speeds_l1_1483


namespace arithmetic_sequence_sum_l1_1965

def sum_of_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℕ)
  (h₁ : a₁ + (a₁ + 6 * d) + (a₁ + 13 * d) + (a₁ + 17 * d) = 120) :
  sum_of_arithmetic_sequence a₁ d 19 = 570 :=
by
  sorry

end arithmetic_sequence_sum_l1_1965


namespace circle_packing_line_equation_l1_1267

theorem circle_packing_line_equation
  (d : ℝ) (n1 n2 n3 : ℕ) (slope : ℝ)
  (l_intersects_tangencies : ℝ → ℝ → Prop)
  (l_divides_R : Prop)
  (gcd_condition : ℕ → ℕ → ℕ → ℕ)
  (a b c : ℕ)
  (a_pos : 0 < a) (b_neg : b < 0) (c_pos : 0 < c)
  (gcd_abc : gcd_condition a b c = 1)
  (correct_equation_format : Prop) :
  n1 = 4 ∧ n2 = 4 ∧ n3 = 2 →
  d = 2 →
  slope = 5 →
  l_divides_R →
  l_intersects_tangencies 1 1 →
  l_intersects_tangencies 4 6 → 
  correct_equation_format → 
  a^2 + b^2 + c^2 = 42 :=
by sorry

end circle_packing_line_equation_l1_1267


namespace simple_interest_rate_l1_1281

-- Define the entities and conditions
variables (P A T : ℝ) (R : ℝ)

-- Conditions given in the problem
def principal := P = 12500
def amount := A = 16750
def time := T = 8

-- Result that needs to be proved
def correct_rate := R = 4.25

-- Main statement to be proven: Given the conditions, the rate is 4.25%
theorem simple_interest_rate :
  principal P → amount A → time T → (A - P = (P * R * T) / 100) → correct_rate R :=
by
  intros hP hA hT hSI
  sorry

end simple_interest_rate_l1_1281


namespace largest_subset_size_l1_1246

theorem largest_subset_size (T : Finset ℕ) (h : ∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x - y) % 2021 ≠ 5 ∧ (x - y) % 2021 ≠ 8) :
  T.card ≤ 918 := sorry

end largest_subset_size_l1_1246


namespace rahuls_share_l1_1090

theorem rahuls_share (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)
  (rahul_work_one_day : rahul_days > 0) (rajesh_work_one_day : rajesh_days > 0)
  (total_payment_eq : total_payment = 105) 
  (rahul_days_eq : rahul_days = 3) 
  (rajesh_days_eq : rajesh_days = 2) :
  rahul_share = 42 := 
by
  sorry

end rahuls_share_l1_1090


namespace triangle_semicircle_l1_1636

noncomputable def triangle_semicircle_ratio : ℝ :=
  let AB := 8
  let BC := 6
  let CA := 2 * Real.sqrt 7
  let radius_AB := AB / 2
  let radius_BC := BC / 2
  let radius_CA := CA / 2
  let area_semicircle_AB := (1 / 2) * Real.pi * radius_AB ^ 2
  let area_semicircle_BC := (1 / 2) * Real.pi * radius_BC ^ 2
  let area_semicircle_CA := (1 / 2) * Real.pi * radius_CA ^ 2
  let area_triangle := AB * BC / 2
  let total_shaded_area := (area_semicircle_AB + area_semicircle_BC + area_semicircle_CA) - area_triangle
  let area_circle_CA := Real.pi * (radius_CA ^ 2)
  total_shaded_area / area_circle_CA

theorem triangle_semicircle : triangle_semicircle_ratio = 2 - (12 * Real.sqrt 3) / (7 * Real.pi) := by
  sorry

end triangle_semicircle_l1_1636


namespace continuous_function_triples_l1_1882

theorem continuous_function_triples (f g h : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g) (h₃ : Continuous h)
  (h₄ : ∀ x y : ℝ, f (x + y) = g x + h y) :
  ∃ (c a b : ℝ), (∀ x : ℝ, f x = c * x + a + b) ∧ (∀ x : ℝ, g x = c * x + a) ∧ (∀ x : ℝ, h x = c * x + b) :=
sorry

end continuous_function_triples_l1_1882


namespace ratio_of_milk_and_water_l1_1208

theorem ratio_of_milk_and_water (x y : ℝ) (hx : 9 * x = 9 * y) : 
  let total_milk := (7 * x + 8 * y)
  let total_water := (2 * x + y)
  (total_milk / total_water) = 5 :=
by
  sorry

end ratio_of_milk_and_water_l1_1208


namespace arithmetic_arrangement_result_l1_1909

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l1_1909


namespace second_competitor_distance_difference_l1_1939

theorem second_competitor_distance_difference (jump1 jump2 jump3 jump4 : ℕ) : 
  jump1 = 22 → 
  jump4 = 24 → 
  jump3 = jump2 - 2 → 
  jump4 = jump3 + 3 → 
  jump2 - jump1 = 1 :=
by
  sorry

end second_competitor_distance_difference_l1_1939


namespace find_a_and_b_l1_1177

open Function

theorem find_a_and_b (a b : ℚ) (k : ℚ)  (hA : (6 : ℚ) = k * (-3))
    (hB : (a : ℚ) = k * 2)
    (hC : (-1 : ℚ) = k * b) : 
    a = -4 ∧ b = 1 / 2 :=
by
  sorry

end find_a_and_b_l1_1177


namespace pages_needed_l1_1646

def new_cards : ℕ := 2
def old_cards : ℕ := 10
def cards_per_page : ℕ := 3
def total_cards : ℕ := new_cards + old_cards

theorem pages_needed : total_cards / cards_per_page = 4 := by
  sorry

end pages_needed_l1_1646


namespace solve_for_x_l1_1044

theorem solve_for_x (x : ℝ) : (1 + 2*x + 3*x^2) / (3 + 2*x + x^2) = 3 → x = -2 :=
by
  intro h
  sorry

end solve_for_x_l1_1044


namespace eqn_distinct_real_roots_l1_1560

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then x^2 + 2 else 4 * x * Real.cos x + 1

theorem eqn_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, f x = m * x + 1) → 
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2 * Real.pi) Real.pi ∧ x₂ ∈ Set.Icc (-2 * Real.pi) Real.pi :=
  sorry

end eqn_distinct_real_roots_l1_1560


namespace new_ratio_l1_1268

def milk_to_water_initial_ratio (M W : ℕ) : Prop := 4 * W = M

def total_volume (V M W : ℕ) : Prop := V = M + W

def new_water_volume (W_new W A : ℕ) : Prop := W_new = W + A

theorem new_ratio (V M W W_new A : ℕ) 
  (h1: milk_to_water_initial_ratio M W) 
  (h2: total_volume V M W) 
  (h3: A = 23) 
  (h4: new_water_volume W_new W A) 
  (h5: V = 45) 
  : 9 * W_new = 8 * M :=
by 
  sorry

end new_ratio_l1_1268


namespace range_of_f3_l1_1106

def f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_of_f3 (a c : ℝ)
  (h1 : -4 ≤ f a c 1 ∧ f a c 1 ≤ -1)
  (h2 : -1 ≤ f a c 2 ∧ f a c 2 ≤ 5) :
  -1 ≤ f a c 3 ∧ f a c 3 ≤ 20 := 
sorry

end range_of_f3_l1_1106


namespace Jonah_paid_commensurate_l1_1628

def price_per_pineapple (P : ℝ) :=
  let number_of_pineapples := 6
  let rings_per_pineapple := 12
  let total_rings := number_of_pineapples * rings_per_pineapple
  let price_per_4_rings := 5
  let price_per_ring := price_per_4_rings / 4
  let total_revenue := total_rings * price_per_ring
  let profit := 72
  total_revenue - number_of_pineapples * P = profit

theorem Jonah_paid_commensurate {P : ℝ} (h : price_per_pineapple P) :
  P = 3 :=
  sorry

end Jonah_paid_commensurate_l1_1628


namespace power_modulo_remainder_l1_1422

theorem power_modulo_remainder :
  (17 ^ 2046) % 23 = 22 := 
sorry

end power_modulo_remainder_l1_1422


namespace find_number_l1_1871

theorem find_number (x : ℝ) (h₁ : |x| + 1/x = 0) (h₂ : x ≠ 0) : x = -1 :=
sorry

end find_number_l1_1871


namespace first_group_men_count_l1_1750

/-- Given that 10 men can complete a piece of work in 90 hours,
prove that the number of men M in the first group who can complete
the same piece of work in 25 hours is 36. -/
theorem first_group_men_count (M : ℕ) (h : (10 * 90 = 25 * M)) : M = 36 :=
by
  sorry

end first_group_men_count_l1_1750


namespace min_value_m_l1_1584

theorem min_value_m (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + a 1)
  (h_geometric : ∀ n, b (n + 1) = b 1 * (b 1 ^ n))
  (h_b1_mean : 2 * b 1 = a 1 + a 2)
  (h_a3 : a 3 = 5)
  (h_b3 : b 3 = a 4 + 1)
  (h_S_formula : ∀ n, S n = n^2)
  (h_S_le_b : ∀ n ≥ 4, S n ≤ b n) :
  ∃ m, ∀ n, (n ≥ m → S n ≤ b n) ∧ m = 4 := sorry

end min_value_m_l1_1584


namespace find_number_l1_1341

theorem find_number (x : ℤ) : 17 * (x + 99) = 3111 → x = 84 :=
by
  sorry

end find_number_l1_1341


namespace domain_of_sqrt_sin_l1_1364

open Real Set

noncomputable def domain_sqrt_sine : Set ℝ :=
  {x | ∃ (k : ℤ), 2 * π * k + π / 6 ≤ x ∧ x ≤ 2 * π * k + 5 * π / 6}

theorem domain_of_sqrt_sin (x : ℝ) :
  (∃ y, y = sqrt (2 * sin x - 1)) ↔ x ∈ domain_sqrt_sine :=
sorry

end domain_of_sqrt_sin_l1_1364


namespace max_value_of_expression_l1_1903

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y + 1) + xz / (x + z + 1) + yz / (y + z + 1)) ≤ 1 :=
sorry

end max_value_of_expression_l1_1903


namespace vertices_after_cut_off_four_corners_l1_1995

-- Definitions for the conditions
def regular_tetrahedron.num_vertices : ℕ := 4

def new_vertices_per_cut : ℕ := 3

def total_vertices_after_cut : ℕ := 
  regular_tetrahedron.num_vertices + regular_tetrahedron.num_vertices * new_vertices_per_cut

-- The theorem to prove the question
theorem vertices_after_cut_off_four_corners :
  total_vertices_after_cut = 12 :=
by
  -- sorry is used to skip the proof steps, as per instructions
  sorry

end vertices_after_cut_off_four_corners_l1_1995


namespace athlete_B_more_stable_l1_1162

variable (average_scores_A average_scores_B : ℝ)
variable (s_A_squared s_B_squared : ℝ)

theorem athlete_B_more_stable
  (h_avg : average_scores_A = average_scores_B)
  (h_var_A : s_A_squared = 1.43)
  (h_var_B : s_B_squared = 0.82) :
  s_A_squared > s_B_squared :=
by 
  rw [h_var_A, h_var_B]
  sorry

end athlete_B_more_stable_l1_1162


namespace roots_of_quadratic_l1_1180

theorem roots_of_quadratic :
  ∃ m n : ℝ, (∀ x : ℝ, x^2 - 4 * x - 1 = 0 → (x = m ∨ x = n)) ∧
            (m + n = 4) ∧
            (m * n = -1) ∧
            (m + n - m * n = 5) :=
by
  sorry

end roots_of_quadratic_l1_1180


namespace sum_of_interior_angles_of_hexagon_l1_1774

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l1_1774


namespace Chris_has_6_Teslas_l1_1338

theorem Chris_has_6_Teslas (x y z : ℕ) (h1 : z = 13) (h2 : z = x + 10) (h3 : x = y / 2):
  y = 6 :=
by
  sorry

end Chris_has_6_Teslas_l1_1338


namespace solve_linear_system_l1_1713

/-- Let x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈ be real numbers that satisfy the following system of equations:
1. x₁ + x₂ + x₃ = 6
2. x₂ + x₃ + x₄ = 9
3. x₃ + x₄ + x₅ = 3
4. x₄ + x₅ + x₆ = -3
5. x₅ + x₆ + x₇ = -9
6. x₆ + x₇ + x₈ = -6
7. x₇ + x₈ + x₁ = -2
8. x₈ + x₁ + x₂ = 2
Prove that the solution is
  x₁ = 1, x₂ = 2, x₃ = 3, x₄ = 4, x₅ = -4, x₆ = -3, x₇ = -2, x₈ = -1
-/
theorem solve_linear_system :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
  x₁ + x₂ + x₃ = 6 →
  x₂ + x₃ + x₄ = 9 →
  x₃ + x₄ + x₅ = 3 →
  x₄ + x₅ + x₆ = -3 →
  x₅ + x₆ + x₇ = -9 →
  x₆ + x₇ + x₈ = -6 →
  x₇ + x₈ + x₁ = -2 →
  x₈ + x₁ + x₂ = 2 →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  intros x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, the proof steps would go
  sorry

end solve_linear_system_l1_1713


namespace johns_tour_program_days_l1_1030

/-- John has Rs 360 for his expenses. If he exceeds his days by 4 days, he must cut down daily expenses by Rs 3. Prove that the number of days of John's tour program is 20. -/
theorem johns_tour_program_days
    (d e : ℕ)
    (h1 : 360 = e * d)
    (h2 : 360 = (e - 3) * (d + 4)) : 
    d = 20 := 
  sorry

end johns_tour_program_days_l1_1030


namespace find_a_l1_1078

theorem find_a (x y a : ℤ) (h1 : a * x + y = 40) (h2 : 2 * x - y = 20) (h3 : 3 * y^2 = 48) : a = 3 :=
sorry

end find_a_l1_1078


namespace length_of_equal_pieces_l1_1470

theorem length_of_equal_pieces (total_length : ℕ) (num_pieces : ℕ) (num_unequal_pieces : ℕ) (unequal_piece_length : ℕ)
    (equal_pieces : ℕ) (equal_piece_length : ℕ) :
    total_length = 11650 ∧ num_pieces = 154 ∧ num_unequal_pieces = 4 ∧ unequal_piece_length = 100 ∧ equal_pieces = 150 →
    equal_piece_length = 75 :=
by
  sorry

end length_of_equal_pieces_l1_1470


namespace division_addition_l1_1392

theorem division_addition : (-300) / (-75) + 10 = 14 := by
  sorry

end division_addition_l1_1392


namespace cost_price_of_cricket_bat_l1_1986

variable (CP_A CP_B SP_C : ℝ)

-- Conditions
def condition1 : CP_B = 1.20 * CP_A := sorry
def condition2 : SP_C = 1.25 * CP_B := sorry
def condition3 : SP_C = 234 := sorry

-- The statement to prove
theorem cost_price_of_cricket_bat : CP_A = 156 := sorry

end cost_price_of_cricket_bat_l1_1986


namespace intersection_A_B_l1_1400

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℝ := { x | 2^x > 2 * Real.sqrt 2 }

theorem intersection_A_B :
  A ∩ { x : ℤ | x > 3 / 2 } = {3, 5, 7} :=
by
  sorry

end intersection_A_B_l1_1400


namespace olympics_year_zodiac_l1_1839

-- Define the list of zodiac signs
def zodiac_cycle : List String :=
  ["rat", "ox", "tiger", "rabbit", "dragon", "snake", "horse", "goat", "monkey", "rooster", "dog", "pig"]

-- Function to compute the zodiac sign for a given year
def zodiac_sign (start_year : ℕ) (year : ℕ) : String :=
  let index := (year - start_year) % 12
  zodiac_cycle.getD index "unknown"

-- Proof statement: the zodiac sign of the year 2008 is "rabbit"
theorem olympics_year_zodiac :
  zodiac_sign 1 2008 = "rabbit" :=
by
  -- Proof omitted
  sorry

end olympics_year_zodiac_l1_1839


namespace dorothy_needs_more_money_l1_1384

structure Person :=
  (age : ℕ)

def Discount (age : ℕ) : ℝ :=
  if age <= 11 then 0.5 else
  if age >= 65 then 0.8 else
  if 12 <= age && age <= 18 then 0.7 else 1.0

def ticketCost (age : ℕ) : ℝ :=
  (10 : ℝ) * Discount age

def specialExhibitCost : ℝ := 5

def totalCost (family : List Person) : ℝ :=
  (family.map (λ p => ticketCost p.age + specialExhibitCost)).sum

def salesTaxRate : ℝ := 0.1

def finalCost (family : List Person) : ℝ :=
  let total := totalCost family
  total + (total * salesTaxRate)

def dorothy_money_after_trip (dorothy_money : ℝ) (family : List Person) : ℝ :=
  dorothy_money - finalCost family

theorem dorothy_needs_more_money :
  dorothy_money_after_trip 70 [⟨15⟩, ⟨10⟩, ⟨40⟩, ⟨42⟩, ⟨65⟩] = -1.5 := by
  sorry

end dorothy_needs_more_money_l1_1384


namespace train_length_is_500_l1_1689

def speed_kmph : ℕ := 360
def time_sec : ℕ := 5

def speed_mps (v_kmph : ℕ) : ℕ :=
  v_kmph * 1000 / 3600

def length_of_train (v_mps : ℕ) (t_sec : ℕ) : ℕ :=
  v_mps * t_sec

theorem train_length_is_500 :
  length_of_train (speed_mps speed_kmph) time_sec = 500 := 
sorry

end train_length_is_500_l1_1689


namespace number_of_male_animals_l1_1891

def total_original_animals : ℕ := 100 + 29 + 9
def animals_bought_by_brian : ℕ := total_original_animals / 2
def animals_after_brian : ℕ := total_original_animals - animals_bought_by_brian
def animals_after_jeremy : ℕ := animals_after_brian + 37

theorem number_of_male_animals : animals_after_jeremy / 2 = 53 :=
by
  sorry

end number_of_male_animals_l1_1891


namespace xiao_wang_fourth_place_l1_1649

section Competition
  -- Define the participants and positions
  inductive Participant
  | XiaoWang : Participant
  | XiaoZhang : Participant
  | XiaoZhao : Participant
  | XiaoLi : Participant

  inductive Position
  | First : Position
  | Second : Position
  | Third : Position
  | Fourth : Position

  open Participant Position

  -- Conditions given in the problem
  variables
    (place : Participant → Position)
    (hA1 : place XiaoWang = First → place XiaoZhang = Third)
    (hA2 : place XiaoWang = First → place XiaoZhang ≠ Third)
    (hB1 : place XiaoLi = First → place XiaoZhao = Fourth)
    (hB2 : place XiaoLi = First → place XiaoZhao ≠ Fourth)
    (hC1 : place XiaoZhao = Second → place XiaoWang = Third)
    (hC2 : place XiaoZhao = Second → place XiaoWang ≠ Third)
    (no_ties : ∀ x y, place x = place y → x = y)
    (half_correct : ∀ p, (p = A → ((place XiaoWang = First ∨ place XiaoZhang = Third) ∧ (place XiaoWang ≠ First ∨ place XiaoZhang ≠ Third)))
                          ∧ (p = B → ((place XiaoLi = First ∨ place XiaoZhao = Fourth) ∧ (place XiaoLi ≠ First ∨ place XiaoZhao ≠ Fourth)))
                          ∧ (p = C → ((place XiaoZhao = Second ∨ place XiaoWang = Third) ∧ (place XiaoZhao ≠ Second ∨ place XiaoWang ≠ Third)))) 

  -- The goal to prove
  theorem xiao_wang_fourth_place : place XiaoWang = Fourth :=
  sorry
end Competition

end xiao_wang_fourth_place_l1_1649


namespace power_function_odd_l1_1327

-- Define the conditions
def f : ℝ → ℝ := sorry
def condition1 (f : ℝ → ℝ) : Prop := f 1 = 3

-- Define the statement of the problem as a Lean theorem
theorem power_function_odd (f : ℝ → ℝ) (h : condition1 f) : ∀ x, f (-x) = -f x := sorry

end power_function_odd_l1_1327


namespace owner_overtakes_thief_l1_1480

theorem owner_overtakes_thief :
  ∀ (speed_thief speed_owner : ℕ) (time_theft_discovered : ℝ), 
    speed_thief = 45 →
    speed_owner = 50 →
    time_theft_discovered = 0.5 →
    (time_theft_discovered + (45 * 0.5) / (speed_owner - speed_thief)) = 5 := 
by
  intros speed_thief speed_owner time_theft_discovered h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end owner_overtakes_thief_l1_1480


namespace unique_ab_for_interval_condition_l1_1644

theorem unique_ab_for_interval_condition : 
  ∃! (a b : ℝ), (∀ x, (0 ≤ x ∧ x ≤ 1) → |x^2 - a * x - b| ≤ 1 / 8) ∧ a = 1 ∧ b = -1 / 8 := by
  sorry

end unique_ab_for_interval_condition_l1_1644


namespace quadratic_equation_general_form_l1_1067

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 2 * (x + 2)^2 + (x + 3) * (x - 2) = -11 ↔ 3 * x^2 + 9 * x + 13 = 0 :=
sorry

end quadratic_equation_general_form_l1_1067


namespace solve_inequality_l1_1178

theorem solve_inequality (a x : ℝ) :
  (a = 1/2 → (x ≠ 1/2 → (x - a) * (x + a - 1) > 0)) ∧
  (a < 1/2 → ((x > (1 - a) ∨ x < a) → (x - a) * (x + a - 1) > 0)) ∧
  (a > 1/2 → ((x > a ∨ x < (1 - a)) → (x - a) * (x + a - 1) > 0)) :=
by
  sorry

end solve_inequality_l1_1178


namespace decimal_to_binary_correct_l1_1305

-- Define the decimal number
def decimal_number : ℕ := 25

-- Define the binary equivalent of 25
def binary_representation : ℕ := 0b11001

-- The condition indicating how the conversion is done
def is_binary_representation (decimal : ℕ) (binary : ℕ) : Prop :=
  -- Check if the binary representation matches the manual decomposition
  decimal = (binary / 2^4) * 2^4 + 
            ((binary % 2^4) / 2^3) * 2^3 + 
            (((binary % 2^4) % 2^3) / 2^2) * 2^2 + 
            ((((binary % 2^4) % 2^3) % 2^2) / 2^1) * 2^1 + 
            (((((binary % 2^4) % 2^3) % 2^2) % 2^1) / 2^0) * 2^0

-- Proof statement
theorem decimal_to_binary_correct : is_binary_representation decimal_number binary_representation :=
  by sorry

end decimal_to_binary_correct_l1_1305


namespace problem_l1_1299

theorem problem : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end problem_l1_1299


namespace correct_operation_l1_1105

theorem correct_operation :
  (3 * a^3 - 2 * a^3 = a^3) ∧ ¬(m - 4 * m = -3) ∧ ¬(a^2 * b - a * b^2 = 0) ∧ ¬(2 * x + 3 * x = 5 * x^2) :=
by
  sorry

end correct_operation_l1_1105


namespace ratio_mets_to_redsox_l1_1103

theorem ratio_mets_to_redsox (Y M R : ℕ)
  (h1 : Y / M = 3 / 2)
  (h2 : M = 96)
  (h3 : Y + M + R = 360) :
  M / R = 4 / 5 :=
by sorry

end ratio_mets_to_redsox_l1_1103


namespace tyre_punctures_deflation_time_l1_1275

theorem tyre_punctures_deflation_time :
  (1 / (1 / 9 + 1 / 6)) = 3.6 :=
by
  sorry

end tyre_punctures_deflation_time_l1_1275


namespace find_special_four_digit_square_l1_1325

theorem find_special_four_digit_square :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    n = 8281 ∧
    a = c ∧
    b + 1 = d ∧
    n = (91 : ℕ) ^ 2 :=
by
  sorry

end find_special_four_digit_square_l1_1325


namespace find_f_of_2_l1_1096

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem find_f_of_2 (a b : ℝ)
  (h1 : 3 + 2 * a + b = 0)
  (h2 : 1 + a + b + a^2 = 10)
  (ha : a = 4)
  (hb : b = -11) :
  f 2 a b = 18 := by {
  -- We assume the values of a and b provided by the user as the correct pair.
  sorry
}

end find_f_of_2_l1_1096


namespace original_proposition_converse_negation_contrapositive_l1_1326

variable {a b : ℝ}

-- Original Proposition: If \( x^2 + ax + b \leq 0 \) has a non-empty solution set, then \( a^2 - 4b \geq 0 \)
theorem original_proposition (h : ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b ≥ 0 := sorry

-- Converse: If \( a^2 - 4b \geq 0 \), then \( x^2 + ax + b \leq 0 \) has a non-empty solution set
theorem converse (h : a^2 - 4 * b ≥ 0) : ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

-- Negation: If \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set, then \( a^2 - 4b < 0 \)
theorem negation (h : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b < 0 := sorry

-- Contrapositive: If \( a^2 - 4b < 0 \), then \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set
theorem contrapositive (h : a^2 - 4 * b < 0) : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

end original_proposition_converse_negation_contrapositive_l1_1326


namespace volume_of_pyramid_l1_1782

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end volume_of_pyramid_l1_1782


namespace common_solution_ys_l1_1484

theorem common_solution_ys : 
  {y : ℝ | ∃ x : ℝ, x^2 + y^2 = 9 ∧ x^2 + 2*y = 7} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
sorry

end common_solution_ys_l1_1484


namespace solution_set_ineq_l1_1781

theorem solution_set_ineq (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ 5 < x ∧ x ≤ 13 / 2 :=
sorry

end solution_set_ineq_l1_1781


namespace increase_in_surface_area_l1_1696

-- Define the edge length of the original cube and other conditions
variable (a : ℝ)

-- Define the increase in surface area problem
theorem increase_in_surface_area (h : 1 ≤ 27) : 
  let original_surface_area := 6 * a^2
  let smaller_cube_edge := a / 3
  let smaller_surface_area := 6 * (smaller_cube_edge)^2
  let total_smaller_surface_area := 27 * smaller_surface_area
  total_smaller_surface_area - original_surface_area = 12 * a^2 :=
by
  -- Provided the proof to satisfy Lean 4 syntax requirements to check for correctness
  sorry

end increase_in_surface_area_l1_1696


namespace sales_tax_difference_l1_1017

theorem sales_tax_difference (price : ℝ) (rate1 rate2 : ℝ) : 
  rate1 = 0.085 → rate2 = 0.07 → price = 50 → 
  (price * rate1 - price * rate2) = 0.75 := 
by 
  intros h_rate1 h_rate2 h_price
  rw [h_rate1, h_rate2, h_price] 
  simp
  sorry

end sales_tax_difference_l1_1017


namespace find_integer_pairs_l1_1203

theorem find_integer_pairs :
  ∃ (n : ℤ) (a : ℤ) (b : ℤ),
    (∀ a b : ℤ, (∃ m : ℤ, a^2 - 4*b = m^2) ∧ (∃ k : ℤ, b^2 - 4*a = k^2) ↔ 
    (a = 0 ∧ ∃ n : ℤ, b = n^2) ∨
    (b = 0 ∧ ∃ n : ℤ, a = n^2) ∨
    (b > 0 ∧ ∃ a : ℤ, a^2 > 0 ∧ b = -1 - a) ∨
    (a > 0 ∧ ∃ b : ℤ, b^2 > 0 ∧ a = -1 - b) ∨
    (a = 4 ∧ b = 4) ∨
    (a = 5 ∧ b = 6) ∨
    (a = 6 ∧ b = 5)) :=
sorry

end find_integer_pairs_l1_1203


namespace at_least_one_inequality_holds_l1_1222

theorem at_least_one_inequality_holds
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l1_1222


namespace triangle_inequality_l1_1092

variable {a b c : ℝ}
variable {x y z : ℝ}

theorem triangle_inequality (ha : a ≥ b) (hb : b ≥ c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hx_yz_sum : x + y + z = π) :
  bc + ca - ab < bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ∧
  bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 := sorry

end triangle_inequality_l1_1092


namespace sequence_x_y_sum_l1_1277

theorem sequence_x_y_sum :
  ∃ (r x y : ℝ), 
    (r * 3125 = 625) ∧ 
    (r * 625 = 125) ∧ 
    (r * 125 = x) ∧ 
    (r * x = y) ∧ 
    (r * y = 1) ∧
    (r * 1 = 1/5) ∧ 
    (r * (1/5) = 1/25) ∧ 
    x + y = 30 := 
by
  -- A placeholder for the actual proof
  sorry

end sequence_x_y_sum_l1_1277


namespace chord_length_l1_1881

-- Define radii of the circles
def r1 : ℝ := 5
def r2 : ℝ := 12
def r3 : ℝ := r1 + r2

-- Define the centers of the circles
variable (O1 O2 O3 : ℝ)

-- Define the points of tangency and foot of the perpendicular
def T1 : ℝ := O1 + r1
def T2 : ℝ := O2 + r2
def T : ℝ := O3 - r3

-- Given the conditions
theorem chord_length (m n p : ℤ) : 
  (∃ (C1 C2 C3 : ℝ) (tangent1 tangent2 : ℝ),
    C1 = r1 ∧ C2 = r2 ∧ C3 = r3 ∧
    -- Externally tangent: distance between centers of C1 and C2 is r1 + r2
    dist O1 O2 = r1 + r2 ∧
    -- Internally tangent: both C1 and C2 are tangent to C3
    dist O1 O3 = r3 - r1 ∧
    dist O2 O3 = r3 - r2 ∧
    -- The chord in C3 is a common external tangent to C1 and C2
    tangent1 = O3 + ((O1 * O2) - (O1 * O3)) / r1 ∧
    tangent2 = O3 + ((O2 * O1) - (O2 * O3)) / r2 ∧
    m = 10 ∧ n = 546 ∧ p = 7 ∧
    m + n + p = 563)
  := sorry

end chord_length_l1_1881


namespace chemical_x_added_l1_1481

theorem chemical_x_added (initial_volume : ℝ) (initial_percentage : ℝ) (final_percentage : ℝ) : 
  initial_volume = 80 → initial_percentage = 0.2 → final_percentage = 0.36 → 
  ∃ (a : ℝ), 0.20 * initial_volume + a = 0.36 * (initial_volume + a) ∧ a = 20 :=
by
  intros h1 h2 h3
  use 20
  sorry

end chemical_x_added_l1_1481


namespace correct_exp_operation_l1_1948

theorem correct_exp_operation (a b : ℝ) : (-a^3 * b) ^ 2 = a^6 * b^2 :=
  sorry

end correct_exp_operation_l1_1948


namespace log_value_l1_1028

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_value (x : ℝ) (h : log_base 3 (5 * x) = 3) : log_base x 125 = 3 / 2 :=
  by
  sorry

end log_value_l1_1028


namespace smallest_X_divisible_by_15_l1_1858

theorem smallest_X_divisible_by_15 (T : ℕ) (h_pos : T > 0) (h_digits : ∀ (d : ℕ), d ∈ (Nat.digits 10 T) → d = 0 ∨ d = 1)
  (h_div15 : T % 15 = 0) : ∃ X : ℕ, X = T / 15 ∧ X = 74 :=
sorry

end smallest_X_divisible_by_15_l1_1858


namespace find_number_of_children_l1_1214

-- Definitions based on conditions
def decorative_spoons : Nat := 2
def new_set_large_spoons : Nat := 10
def new_set_tea_spoons : Nat := 15
def total_spoons : Nat := 39
def spoons_per_child : Nat := 3
def new_set_spoons := new_set_large_spoons + new_set_tea_spoons

-- The main statement to prove the number of children
theorem find_number_of_children (C : Nat) :
  3 * C + decorative_spoons + new_set_spoons = total_spoons → C = 4 :=
by
  -- Proof would go here
  sorry

end find_number_of_children_l1_1214


namespace sum_sequence_formula_l1_1216

-- Define the sequence terms as a function.
def seq_term (x a : ℕ) (n : ℕ) : ℕ :=
x ^ (n + 1) + (n + 1) * a

-- Define the sum of the first nine terms of the sequence.
def sum_first_nine_terms (x a : ℕ) : ℕ :=
(x * (x ^ 9 - 1)) / (x - 1) + 45 * a

-- State the theorem to prove that the sum S is as expected.
theorem sum_sequence_formula (x a : ℕ) (h : x ≠ 1) : 
  sum_first_nine_terms x a = (x ^ 10 - x) / (x - 1) + 45 * a := by
  sorry

end sum_sequence_formula_l1_1216


namespace max_volume_range_of_a_x1_x2_inequality_l1_1498

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (a * x^2) - Real.exp 1 * x + a * x^2 - 1) / x

theorem max_volume (x : ℝ) (hx : 1 < x) :
  ∃ V : ℝ, V = (Real.pi / 3) * ((Real.log x)^2 / x) ∧ V = (4 * Real.pi / (3 * (Real.exp 2)^2)) :=
sorry

theorem range_of_a (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  0 < a ∧ a < (1/2) * (Real.exp 1) :=
sorry

theorem x1_x2_inequality (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  x1^2 + x2^2 > 2 / Real.exp 1 :=
sorry

end max_volume_range_of_a_x1_x2_inequality_l1_1498


namespace smallest_positive_integer_satisfying_conditions_l1_1907

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (N : ℕ), N = 242 ∧
    ( ∃ (i : Fin 4), (N + i) % 8 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 9 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 25 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 121 = 0 ) :=
sorry

end smallest_positive_integer_satisfying_conditions_l1_1907


namespace sum_of_squares_of_consecutive_even_integers_l1_1715

theorem sum_of_squares_of_consecutive_even_integers (n : ℤ) (h : (2 * n - 2) * (2 * n) * (2 * n + 2) = 12 * ((2 * n - 2) + (2 * n) + (2 * n + 2))) :
  (2 * n - 2) ^ 2 + (2 * n) ^ 2 + (2 * n + 2) ^ 2 = 440 :=
by
  sorry

end sum_of_squares_of_consecutive_even_integers_l1_1715


namespace Jenny_walked_distance_l1_1879

-- Given: Jenny ran 0.6 mile.
-- Given: Jenny ran 0.2 miles farther than she walked.
-- Prove: Jenny walked 0.4 miles.

variable (r w : ℝ)

theorem Jenny_walked_distance
  (h1 : r = 0.6) 
  (h2 : r = w + 0.2) : 
  w = 0.4 :=
sorry

end Jenny_walked_distance_l1_1879


namespace unique_solution_l1_1466

theorem unique_solution (x y z n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hn : 2 ≤ n) (h_y_bound : y ≤ 5 * 2^(2*n)) :
  x^(2*n+1) - y^(2*n+1) = x * y * z + 2^(2*n+1) → (x, y, z, n) = (3, 1, 70, 2) :=
by
  sorry

end unique_solution_l1_1466


namespace car_travel_time_l1_1674

theorem car_travel_time (speed distance : ℝ) (h₁ : speed = 65) (h₂ : distance = 455) :
  distance / speed = 7 :=
by
  -- We will invoke the conditions h₁ and h₂ to conclude the theorem
  sorry

end car_travel_time_l1_1674


namespace number_of_neutrons_l1_1875

def mass_number (element : Type) : ℕ := 61
def atomic_number (element : Type) : ℕ := 27

theorem number_of_neutrons (element : Type) : mass_number element - atomic_number element = 34 :=
by
  -- Place the proof here
  sorry

end number_of_neutrons_l1_1875


namespace major_axis_length_of_ellipse_l1_1183

theorem major_axis_length_of_ellipse :
  ∀ {y x : ℝ},
  (y^2 / 25 + x^2 / 15 = 1) → 
  2 * Real.sqrt 25 = 10 :=
by
  intro y x h
  sorry

end major_axis_length_of_ellipse_l1_1183


namespace number_of_participants_eq_14_l1_1637

theorem number_of_participants_eq_14 (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
by
  sorry

end number_of_participants_eq_14_l1_1637


namespace shorter_side_of_rectangle_l1_1278

theorem shorter_side_of_rectangle (a b : ℕ) (h_perimeter : 2 * a + 2 * b = 62) (h_area : a * b = 240) : b = 15 :=
by
  sorry

end shorter_side_of_rectangle_l1_1278


namespace area_of_triangle_is_correct_l1_1419

def point := ℚ × ℚ

def A : point := (4, -4)
def B : point := (-1, 1)
def C : point := (2, -7)

def vector_sub (p1 p2 : point) : point :=
(p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℚ :=
v.1 * w.2 - v.2 * w.1

def area_of_triangle (A B C : point) : ℚ :=
(abs (determinant (vector_sub C A) (vector_sub C B))) / 2

theorem area_of_triangle_is_correct :
  area_of_triangle A B C = 12.5 :=
by sorry

end area_of_triangle_is_correct_l1_1419


namespace shortest_distance_between_semicircles_l1_1026

theorem shortest_distance_between_semicircles
  (ABCD : Type)
  (AD : ℝ)
  (shaded_area : ℝ)
  (is_rectangle : true)
  (AD_eq_10 : AD = 10)
  (shaded_area_eq_100 : shaded_area = 100) :
  ∃ d : ℝ, d = 2.5 * Real.pi :=
by
  sorry

end shortest_distance_between_semicircles_l1_1026


namespace diagonal_of_square_l1_1349

-- Definitions based on conditions
def square_area := 8 -- Area of the square is 8 square centimeters

def diagonal_length (x : ℝ) : Prop :=
  (1/2) * x ^ 2 = square_area

-- Proof problem statement
theorem diagonal_of_square : ∃ x : ℝ, diagonal_length x ∧ x = 4 := 
sorry  -- statement only, proof skipped

end diagonal_of_square_l1_1349


namespace find_certain_number_l1_1727

theorem find_certain_number : 
  ∃ (certain_number : ℕ), 1038 * certain_number = 173 * 240 ∧ certain_number = 40 :=
by
  sorry

end find_certain_number_l1_1727


namespace find_a1_l1_1243

-- Definitions stemming from the conditions in the problem
def arithmetic_seq (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

def is_geometric (a₁ a₃ a₆ : ℕ) : Prop :=
  ∃ r : ℕ, a₃ = r * a₁ ∧ a₆ = r^2 * a₁

theorem find_a1 :
  ∀ a₁ : ℕ,
    (arithmetic_seq a₁ 3 1 = a₁) ∧
    (arithmetic_seq a₁ 3 3 = a₁ + 6) ∧
    (arithmetic_seq a₁ 3 6 = a₁ + 15) ∧
    is_geometric a₁ (a₁ + 6) (a₁ + 15) →
    a₁ = 12 :=
by
  intros
  sorry

end find_a1_l1_1243


namespace number_of_distinct_intersection_points_l1_1201

theorem number_of_distinct_intersection_points :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
  let line := {p : ℝ × ℝ | p.1 = 4}
  let intersection_points := circle ∩ line
  ∃! p : ℝ × ℝ, p ∈ intersection_points :=
by
  sorry

end number_of_distinct_intersection_points_l1_1201


namespace largest_three_digit_multiple_of_six_with_sum_fifteen_l1_1407

theorem largest_three_digit_multiple_of_six_with_sum_fifteen : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (n % 6 = 0) ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ (m % 6 = 0) ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
  sorry

end largest_three_digit_multiple_of_six_with_sum_fifteen_l1_1407


namespace find_k_l1_1630

theorem find_k (k : ℝ) : (∀ x y : ℝ, y = k * x + 3 → (x, y) = (2, 5)) → k = 1 :=
by
  sorry

end find_k_l1_1630


namespace trajectory_eq_find_m_l1_1633

-- First problem: Trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  A = (1, 0) → B = (-1, 0) → 
  (dist P A) * (dist A B) = (dist P B) * (dist A B) → 
  P.snd ^ 2 = 4 * P.fst :=
by sorry

-- Second problem: Value of m
theorem find_m (P : ℝ × ℝ) (M N : ℝ × ℝ) (m : ℝ) :
  P.snd ^ 2 = 4 * P.fst → 
  M.snd = M.fst + m → 
  N.snd = N.fst + m →
  (M.fst - N.fst) * (M.snd - N.snd) + (N.snd - M.snd) * (N.fst - M.fst) = 0 →
  m ≠ 0 →
  m < 1 →
  m = -4 :=
by sorry

end trajectory_eq_find_m_l1_1633


namespace common_number_in_sequence_l1_1901

theorem common_number_in_sequence 
  (a b c d e f g h i j : ℕ) 
  (h1 : (a + b + c + d + e) / 5 = 4) 
  (h2 : (f + g + h + i + j) / 5 = 9)
  (h3 : (a + b + c + d + e + f + g + h + i + j) / 10 = 7)
  (h4 : e = f) :
  e = 5 :=
by
  sorry

end common_number_in_sequence_l1_1901


namespace mary_sheep_purchase_l1_1627

theorem mary_sheep_purchase: 
  ∀ (mary_sheep bob_sheep add_sheep : ℕ), 
    mary_sheep = 300 → 
    bob_sheep = 2 * mary_sheep + 35 → 
    add_sheep = (bob_sheep - 69) - mary_sheep → 
    add_sheep = 266 :=
by
  intros mary_sheep bob_sheep add_sheep _ _
  sorry

end mary_sheep_purchase_l1_1627


namespace total_bill_l1_1699

def number_of_adults := 2
def number_of_children := 5
def meal_cost := 3

theorem total_bill : number_of_adults * meal_cost + number_of_children * meal_cost = 21 :=
by
  sorry

end total_bill_l1_1699


namespace max_correct_answers_l1_1155

-- Definitions based on the conditions
def total_problems : ℕ := 12
def points_per_correct : ℕ := 6
def points_per_incorrect : ℕ := 3
def max_score : ℤ := 37 -- Final score, using ℤ to handle potential negatives in deducting points

-- The statement to prove
theorem max_correct_answers :
  ∃ (c w : ℕ), c + w = total_problems ∧ points_per_correct * c - points_per_incorrect * (total_problems - c) = max_score ∧ c = 8 :=
by
  sorry

end max_correct_answers_l1_1155


namespace investment_ratio_correct_l1_1073

-- Constants representing the savings and investments
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 4
def cost_per_share : ℕ := 50
def shares_bought : ℕ := 25

-- Derived quantities from the conditions
def total_savings_wife : ℕ := weekly_savings_wife * weeks_in_month * months_saving
def total_savings_husband : ℕ := monthly_savings_husband * months_saving
def total_savings : ℕ := total_savings_wife + total_savings_husband
def total_invested_in_stocks : ℕ := shares_bought * cost_per_share
def investment_ratio_nat : ℚ := (total_invested_in_stocks : ℚ) / (total_savings : ℚ)

-- Proof statement
theorem investment_ratio_correct : investment_ratio_nat = 1 / 2 := by
  sorry

end investment_ratio_correct_l1_1073


namespace distance_from_plate_to_bottom_edge_l1_1599

theorem distance_from_plate_to_bottom_edge :
  ∀ (W T d : ℕ), W = 73 ∧ T = 20 ∧ (T + d = W) → d = 53 :=
by
  intros W T d
  rintro ⟨hW, hT, h⟩
  rw [hW, hT] at h
  linarith

end distance_from_plate_to_bottom_edge_l1_1599


namespace beta_angle_relationship_l1_1535

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end beta_angle_relationship_l1_1535


namespace max_value_expr_max_l1_1829

noncomputable def max_value_expr (x : ℝ) : ℝ :=
  (x^2 + 3 - (x^4 + 9).sqrt) / x

theorem max_value_expr_max (x : ℝ) (hx : 0 < x) :
  max_value_expr x ≤ (6 * (6:ℝ).sqrt) / (6 + 3 * (2:ℝ).sqrt) :=
sorry

end max_value_expr_max_l1_1829


namespace sum_of_two_numbers_l1_1492

theorem sum_of_two_numbers (a b : ℝ) (h1 : a + b = 25) (h2 : a * b = 144) (h3 : |a - b| = 7) : a + b = 25 := 
  by
  sorry

end sum_of_two_numbers_l1_1492


namespace meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l1_1830

-- Distance between locations A and B
def distance : ℝ := 448

-- Speed of the slow train
def slow_speed : ℝ := 60

-- Speed of the fast train
def fast_speed : ℝ := 80

-- Problem 1: Prove the two trains meet 3.2 hours after the fast train departs (both trains heading towards each other, departing at the same time)
theorem meet_time_same_departure : 
  (slow_speed + fast_speed) * 3.2 = distance :=
by
  sorry

-- Problem 2: Prove the two trains meet 3 hours after the fast train departs (slow train departs 28 minutes before the fast train)
theorem meet_time_staggered_departure : 
  (slow_speed * (28/60) + (slow_speed + fast_speed) * 3) = distance :=
by
  sorry

-- Problem 3: Prove the fast train catches up to the slow train 22.4 hours after departure (both trains heading in the same direction, departing at the same time)
theorem catch_up_time_same_departure : 
  (fast_speed - slow_speed) * 22.4 = distance :=
by
  sorry

end meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l1_1830


namespace find_a6_l1_1845

-- Defining the conditions of the problem
def a1 := 2
def S3 := 12

-- Defining the necessary arithmetic sequence properties
def Sn (a1 d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2
def an (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Proof statement in Lean
theorem find_a6 (d : ℕ) (a1_val S3_val : ℕ) (h1 : a1_val = 2) (h2 : S3_val = 12) 
    (h3 : 3 * (2 * a1_val + (3 - 1) * d) / 2 = S3_val) : an a1_val d 6 = 12 :=
by 
  -- omitted proof
  sorry

end find_a6_l1_1845


namespace range_of_a_l1_1532

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → 1 ≤ a :=
by 
  sorry

end range_of_a_l1_1532


namespace S_11_eq_22_l1_1307

variable {S : ℕ → ℕ}

-- Condition: given that S_8 - S_3 = 10
axiom h : S 8 - S 3 = 10

-- Proof goal: we want to show that S_11 = 22
theorem S_11_eq_22 : S 11 = 22 :=
by
  sorry

end S_11_eq_22_l1_1307


namespace problem_statement_l1_1313

-- Given conditions
noncomputable def S : ℕ → ℝ := sorry
axiom S_3_eq_2 : S 3 = 2
axiom S_6_eq_6 : S 6 = 6

-- Prove that a_{13} + a_{14} + a_{15} = 32
theorem problem_statement : (S 15 - S 12) = 32 :=
by sorry

end problem_statement_l1_1313


namespace simplify_expression_l1_1975

variable (x y : ℕ)

theorem simplify_expression :
  7 * x + 9 * y + 3 - x + 12 * y + 15 = 6 * x + 21 * y + 18 :=
by
  sorry

end simplify_expression_l1_1975


namespace calculate_income_l1_1260

theorem calculate_income (I : ℝ) (T : ℝ) (a b c d : ℝ) (h1 : a = 0.15) (h2 : b = 40000) (h3 : c = 0.20) (h4 : T = 8000) (h5 : T = a * b + c * (I - b)) : I = 50000 :=
by
  sorry

end calculate_income_l1_1260


namespace proof_problem_l1_1971

-- Define the proportional relationship
def proportional_relationship (y x : ℝ) (k : ℝ) : Prop :=
  y - 1 = k * (x + 2)

-- Define the function y = 2x + 5
def function_y_x (y x : ℝ) : Prop :=
  y = 2 * x + 5

-- The theorem for part (1) and (2)
theorem proof_problem (x y a : ℝ) (h1 : proportional_relationship 7 1 2) (h2 : proportional_relationship y x 2) :
  function_y_x y x ∧ function_y_x (-2) a → a = -7 / 2 :=
by
  sorry

end proof_problem_l1_1971


namespace heather_aprons_l1_1143

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l1_1143


namespace olympic_high_school_amc10_l1_1382

/-- At Olympic High School, 2/5 of the freshmen and 4/5 of the sophomores took the AMC-10.
    Given that the number of freshmen and sophomore contestants was the same, there are twice as many freshmen as sophomores. -/
theorem olympic_high_school_amc10 (f s : ℕ) (hf : f > 0) (hs : s > 0)
  (contest_equal : (2 / 5 : ℚ)*f = (4 / 5 : ℚ)*s) : f = 2 * s :=
by
  sorry

end olympic_high_school_amc10_l1_1382


namespace cost_price_is_925_l1_1833

-- Definitions for the conditions
def SP : ℝ := 1110
def profit_percentage : ℝ := 0.20

-- Theorem to prove that the cost price is 925
theorem cost_price_is_925 (CP : ℝ) (h : SP = (CP * (1 + profit_percentage))) : CP = 925 := 
by sorry

end cost_price_is_925_l1_1833


namespace maximum_value_of_f_l1_1949

noncomputable def f (x : ℝ) : ℝ := ((x - 3) * (12 - x)) / x

theorem maximum_value_of_f :
  ∀ x : ℝ, 3 < x ∧ x < 12 → f x ≤ 3 :=
by
  sorry

end maximum_value_of_f_l1_1949


namespace find_g_neg1_l1_1940

-- Define the function f and its property of being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Given conditions
variables {f : ℝ → ℝ}
variable  (h_odd : odd_function f)
variable  (h_g1 : g f 1 = 1)

-- The statement we want to prove
theorem find_g_neg1 : g f (-1) = 3 :=
sorry

end find_g_neg1_l1_1940


namespace combined_weight_difference_l1_1007

def john_weight : ℕ := 81
def roy_weight : ℕ := 79
def derek_weight : ℕ := 91
def samantha_weight : ℕ := 72

theorem combined_weight_difference :
  derek_weight - samantha_weight = 19 :=
by
  sorry

end combined_weight_difference_l1_1007


namespace triangle_side_ratio_eq_one_l1_1185

theorem triangle_side_ratio_eq_one
    (a b c C : ℝ)
    (h1 : a = 2 * b * Real.cos C)
    (cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
    (b / c = 1) := 
by 
    sorry

end triangle_side_ratio_eq_one_l1_1185


namespace only_ten_perfect_square_l1_1172

theorem only_ten_perfect_square (n : ℤ) :
  ∃ k : ℤ, n^4 + 6 * n^3 + 11 * n^2 + 3 * n + 31 = k^2 ↔ n = 10 :=
by
  sorry

end only_ten_perfect_square_l1_1172


namespace line_through_nodes_l1_1533

def Point := (ℤ × ℤ)

structure Triangle :=
  (A B C : Point)

def is_node (p : Point) : Prop := 
  ∃ (x y : ℤ), p = (x, y)

def strictly_inside (p : Point) (t : Triangle) : Prop := 
  -- Assume we have a function that defines if a point is strictly inside a triangle
  sorry

def nodes_inside (t : Triangle) (nodes : List Point) : Prop := 
  nodes.length = 2 ∧ ∀ p, p ∈ nodes → strictly_inside p t

theorem line_through_nodes (t : Triangle) (node1 node2 : Point) (h_inside : nodes_inside t [node1, node2]) :
   ∃ (v : Point), v ∈ [t.A, t.B, t.C] ∨
   (∃ (s : Triangle -> Point -> Point -> Prop), s t node1 node2) := 
sorry

end line_through_nodes_l1_1533


namespace nonnegative_solution_exists_l1_1607

theorem nonnegative_solution_exists
  (a b c d n : ℕ)
  (h_npos : 0 < n)
  (h_gcd_abc : Nat.gcd (Nat.gcd a b) c = 1)
  (h_gcd_ab : Nat.gcd a b = d)
  (h_conds : n > a * b / d + c * d - a - b - c) :
  ∃ x y z : ℕ, a * x + b * y + c * z = n := 
by
  sorry

end nonnegative_solution_exists_l1_1607


namespace ribbon_problem_l1_1241

variable (Ribbon1 Ribbon2 : ℕ)
variable (L : ℕ)

theorem ribbon_problem
    (h1 : Ribbon1 = 8)
    (h2 : ∀ L, L > 0 → Ribbon1 % L = 0 → Ribbon2 % L = 0)
    (h3 : ∀ k, (k > 0 ∧ Ribbon1 % k = 0 ∧ Ribbon2 % k = 0) → k ≤ 8) :
    Ribbon2 = 8 := by
  sorry

end ribbon_problem_l1_1241


namespace probability_both_girls_l1_1997

def club_probability (total_members girls chosen_members : ℕ) : ℚ :=
  (Nat.choose girls chosen_members : ℚ) / (Nat.choose total_members chosen_members : ℚ)

theorem probability_both_girls (H1 : total_members = 12) (H2 : girls = 7) (H3 : chosen_members = 2) :
  club_probability 12 7 2 = 7 / 22 :=
by {
  sorry
}

end probability_both_girls_l1_1997


namespace range_a_l1_1865

theorem range_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → x^2 - 2 * a * x + 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end range_a_l1_1865


namespace max_value_ineq_l1_1339

theorem max_value_ineq (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 1) :
  (a + 3 * b + 5 * c) * (a + b / 3 + c / 5) ≤ 9 / 5 :=
sorry

end max_value_ineq_l1_1339


namespace increasing_function_in_interval_l1_1119

noncomputable def y₁ (x : ℝ) : ℝ := abs (x + 1)
noncomputable def y₂ (x : ℝ) : ℝ := 3 - x
noncomputable def y₃ (x : ℝ) : ℝ := 1 / x
noncomputable def y₄ (x : ℝ) : ℝ := -x^2 + 4

theorem increasing_function_in_interval : ∀ x, (0 < x ∧ x < 1) → 
  y₁ x > y₁ (x - 0.1) ∧ y₂ x < y₂ (x - 0.1) ∧ y₃ x < y₃ (x - 0.1) ∧ y₄ x < y₄ (x - 0.1) :=
by {
  sorry
}

end increasing_function_in_interval_l1_1119


namespace translation_result_l1_1308

-- Define the original point A
structure Point where
  x : ℤ
  y : ℤ

def A : Point := { x := 3, y := -2 }

-- Define the translation function
def translate_right (p : Point) (dx : ℤ) : Point :=
  { x := p.x + dx, y := p.y }

-- Prove that translating point A 2 units to the right gives point A'
theorem translation_result :
  translate_right A 2 = { x := 5, y := -2 } :=
by sorry

end translation_result_l1_1308


namespace class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l1_1521

noncomputable def average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + pitch + innovation) / 3

noncomputable def weighted_average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + 7 * pitch + 2 * innovation) / 10

theorem class_7th_grade_1_has_higher_average_score :
  average_score 90 77 85 > average_score 74 95 80 :=
by sorry

theorem class_7th_grade_2_has_higher_weighted_score :
  weighted_average_score 74 95 80 > weighted_average_score 90 77 85 :=
by sorry

end class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l1_1521


namespace existence_of_same_remainder_mod_36_l1_1167

theorem existence_of_same_remainder_mod_36
  (a : Fin 7 → ℕ) :
  ∃ (i j k l : Fin 7), i < j ∧ k < l ∧ (a i)^2 + (a j)^2 % 36 = (a k)^2 + (a l)^2 % 36 := by
  sorry

end existence_of_same_remainder_mod_36_l1_1167


namespace sum_of_first_n_terms_l1_1377

theorem sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 + 2 * a 2 = 3)
  (h2 : ∀ n, a (n + 1) = a n + 2) :
  ∀ n, S n = n * (n - 4 / 3) := 
sorry

end sum_of_first_n_terms_l1_1377


namespace jar_initial_water_fraction_l1_1554

theorem jar_initial_water_fraction (C W : ℝ) (hC : C > 0) (hW : W + C / 4 = 0.75 * C) : W / C = 0.5 :=
by
  -- necessary parameters and sorry for the proof 
  sorry

end jar_initial_water_fraction_l1_1554


namespace geom_seq_arith_seq_l1_1678

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def isGeomSeq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

theorem geom_seq_arith_seq (h1 : ∀ n, 0 < a n) 
  (h2 : isGeomSeq a q)
  (h3 : 2 * (1 / 2 * a 5) = a 3 + a 4)
  : (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := 
sorry

end geom_seq_arith_seq_l1_1678


namespace books_in_bin_after_actions_l1_1350

theorem books_in_bin_after_actions (x y : ℕ) (z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 250) : x - y + (z / 100) * x = 11 :=
by
  rw [hx, hy, hz]
  -- x - y + (z / 100) * x = 4 - 3 + (250 / 100) * 4
  norm_num
  sorry

end books_in_bin_after_actions_l1_1350


namespace problem1_problem2_l1_1996

variables (a b : ℝ)

-- Problem 1: Prove that 3a^2 - 6a^2 - a^2 = -4a^2
theorem problem1 : (3 * a^2 - 6 * a^2 - a^2 = -4 * a^2) :=
by sorry

-- Problem 2: Prove that (5a - 3b) - 3(a^2 - 2b) = -3a^2 + 5a + 3b
theorem problem2 : ((5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b) :=
by sorry

end problem1_problem2_l1_1996


namespace circle_area_l1_1831

noncomputable def pointA : ℝ × ℝ := (2, 7)
noncomputable def pointB : ℝ × ℝ := (8, 5)

def is_tangent_with_intersection_on_x_axis (A B C : ℝ × ℝ) : Prop :=
  ∃ R : ℝ, ∃ r : ℝ, ∀ M : ℝ × ℝ, dist M C = R → dist A M = r ∧ dist B M = r

theorem circle_area (A B : ℝ × ℝ) (hA : A = (2, 7)) (hB : B = (8, 5))
    (h : ∃ C : ℝ × ℝ, is_tangent_with_intersection_on_x_axis A B C) 
    : ∃ R : ℝ, π * R^2 = 12.5 * π := 
sorry

end circle_area_l1_1831


namespace intersection_of_lines_l1_1562

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ∧ 
  x = 98 / 29 ∧ 
  y = 87 / 58 :=
by 
  sorry

end intersection_of_lines_l1_1562


namespace astroid_area_l1_1150

-- Definitions coming from the conditions
noncomputable def x (t : ℝ) := 4 * (Real.cos t)^3
noncomputable def y (t : ℝ) := 4 * (Real.sin t)^3

-- The theorem stating the area of the astroid
theorem astroid_area : (∫ t in (0 : ℝ)..(Real.pi / 2), y t * (deriv x t)) * 4 = 24 * Real.pi :=
by
  sorry

end astroid_area_l1_1150


namespace joel_age_when_dad_is_twice_l1_1661

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end joel_age_when_dad_is_twice_l1_1661


namespace problem_solution_l1_1896

noncomputable def equilateral_triangle_area_to_perimeter_square_ratio (s : ℝ) (h : s = 10) : ℝ :=
  let altitude := s * Real.sqrt 3 / 2
  let area := 1 / 2 * s * altitude
  let perimeter := 3 * s
  let perimeter_squared := perimeter^2
  area / perimeter_squared

theorem problem_solution :
  equilateral_triangle_area_to_perimeter_square_ratio 10 rfl = Real.sqrt 3 / 36 :=
sorry

end problem_solution_l1_1896


namespace least_positive_integer_solution_l1_1250

theorem least_positive_integer_solution :
  ∃ x : ℤ, x > 0 ∧ ∃ n : ℤ, (3 * x + 29)^2 = 43 * n ∧ x = 19 :=
by
  sorry

end least_positive_integer_solution_l1_1250


namespace tom_tim_typing_ratio_l1_1426

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
by
  sorry

end tom_tim_typing_ratio_l1_1426


namespace find_2a_plus_6b_l1_1753

theorem find_2a_plus_6b (a b : ℕ) (n : ℕ)
  (h1 : 3 * a + 5 * b ≡ 19 [MOD n + 1])
  (h2 : 4 * a + 2 * b ≡ 25 [MOD n + 1])
  (hn : n = 96) :
  2 * a + 6 * b = 96 :=
by
  sorry

end find_2a_plus_6b_l1_1753


namespace fraction_A_BC_l1_1014

-- Definitions for amounts A, B, C and the total T
variable (T : ℝ) (A : ℝ) (B : ℝ) (C : ℝ)

-- Given conditions
def conditions : Prop :=
  T = 300 ∧
  A = 120.00000000000001 ∧
  B = (6 / 9) * (A + C) ∧
  A + B + C = T

-- The fraction of the amount A gets compared to B and C together
def fraction (x : ℝ) : Prop :=
  A = x * (B + C)

-- The proof goal
theorem fraction_A_BC : conditions T A B C → fraction A B C (2 / 3) :=
by
  sorry

end fraction_A_BC_l1_1014


namespace determine_function_f_l1_1215

noncomputable def f (c x : ℝ) : ℝ := c ^ (1 / Real.log x)

theorem determine_function_f (f : ℝ → ℝ) (c : ℝ) (Hc : c > 1) :
  (∀ x, 1 < x → 1 < f x) →
  (∀ (x y : ℝ) (u v : ℝ), 1 < x → 1 < y → 0 < u → 0 < v →
    f (x ^ 4 * y ^ v) ≤ (f x) ^ (1 / (4 * u)) * (f y) ^ (1 / (4 * v))) →
  (∀ x : ℝ, 1 < x → f x = c ^ (1 / Real.log x)) :=
by
  sorry

end determine_function_f_l1_1215


namespace remainder_of_n_plus_2024_l1_1345

-- Define the assumptions
def n : ℤ := sorry  -- n will be some integer
def k : ℤ := sorry  -- k will be some integer

-- Main statement to be proved
theorem remainder_of_n_plus_2024 (h : n % 8 = 3) : (n + 2024) % 8 = 3 := sorry

end remainder_of_n_plus_2024_l1_1345


namespace intersection_of_M_and_N_l1_1509

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Statement of the theorem proving the intersection of M and N
theorem intersection_of_M_and_N :
  M ∩ N = {2, 3} :=
by sorry

end intersection_of_M_and_N_l1_1509


namespace compare_y1_y2_l1_1437

noncomputable def quadratic (x : ℝ) : ℝ := -x^2 + 2

theorem compare_y1_y2 :
  let y1 := quadratic 1
  let y2 := quadratic 3
  y1 > y2 :=
by
  let y1 := quadratic 1
  let y2 := quadratic 3
  sorry

end compare_y1_y2_l1_1437


namespace profit_percent_l1_1282

variable (C S : ℝ)
variable (h : (1 / 3) * S = 0.8 * C)

theorem profit_percent (h : (1 / 3) * S = 0.8 * C) : 
  ((S - C) / C) * 100 = 140 := 
by
  sorry

end profit_percent_l1_1282


namespace calibration_measurements_l1_1795

theorem calibration_measurements (holes : Fin 15 → ℝ) (diameter : ℝ)
  (h1 : ∀ i : Fin 15, holes i = 10 + i.val * 0.04)
  (h2 : 10 ≤ diameter ∧ diameter ≤ 10 + 14 * 0.04) :
  ∃ tries : ℕ, (tries ≤ 4) ∧ (∀ (i : Fin 15), if diameter ≤ holes i then True else False) :=
sorry

end calibration_measurements_l1_1795


namespace choose_four_socks_from_seven_l1_1133

theorem choose_four_socks_from_seven : (Nat.choose 7 4) = 35 :=
by
  sorry

end choose_four_socks_from_seven_l1_1133


namespace total_tips_l1_1112

def tips_per_customer := 2
def customers_friday := 28
def customers_saturday := 3 * customers_friday
def customers_sunday := 36

theorem total_tips : 
  (tips_per_customer * (customers_friday + customers_saturday + customers_sunday) = 296) :=
by
  sorry

end total_tips_l1_1112


namespace bricks_in_wall_l1_1559

theorem bricks_in_wall (x : ℕ) (r₁ r₂ combined_rate : ℕ) :
  (r₁ = x / 8) →
  (r₂ = x / 12) →
  (combined_rate = r₁ + r₂ - 15) →
  (6 * combined_rate = x) →
  x = 360 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end bricks_in_wall_l1_1559


namespace polygon_sides_l1_1635

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1980) : n = 13 := 
by sorry

end polygon_sides_l1_1635


namespace value_of_a_l1_1303

theorem value_of_a (m n a : ℚ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + a) + 5) : 
  a = 2 / 5 :=
by
  sorry

end value_of_a_l1_1303


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l1_1495

variable {n : ℕ}

-- Defining sequences and sums
def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry
def T (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Given conditions
axiom h1 : 2 * S n = 3 * a n - 3
axiom h2 : b 1 = a 1
axiom h3 : b 7 = b 1 * b 2
axiom a1_value : a 1 = 3
axiom d_value : ∃ d : ℕ, b 2 = b 1 + d ∧ b 7 = b 1 + 6 * d

theorem geometric_sequence_general_term : a n = 3 ^ n :=
by sorry

theorem arithmetic_sequence_sum : T n = n^2 + 2*n :=
by sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l1_1495


namespace prove_remaining_area_is_24_l1_1412

/-- A rectangular piece of paper with length 12 cm and width 8 cm has four identical isosceles 
right triangles with legs of 6 cm cut from it. Prove that the remaining area is 24 cm². --/
def remaining_area : ℕ := 
  let length := 12
  let width := 8
  let rect_area := length * width
  let triangle_leg := 6
  let triangle_area := (triangle_leg * triangle_leg) / 2
  let total_triangle_area := 4 * triangle_area
  rect_area - total_triangle_area

theorem prove_remaining_area_is_24 : (remaining_area = 24) :=
  by sorry

end prove_remaining_area_is_24_l1_1412


namespace least_common_multiple_l1_1520

theorem least_common_multiple (x : ℕ) (hx : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end least_common_multiple_l1_1520


namespace machine_value_after_two_years_l1_1880

theorem machine_value_after_two_years (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) (value_after_two_years : ℝ) :
  initial_value = 8000 ∧ decrease_rate = 0.30 ∧ years = 2 → value_after_two_years = 3200 := by
  intros h
  sorry

end machine_value_after_two_years_l1_1880


namespace cirrus_clouds_count_l1_1293

theorem cirrus_clouds_count 
  (cirrus_clouds cumulus_clouds cumulonimbus_clouds : ℕ)
  (h1 : cirrus_clouds = 4 * cumulus_clouds)
  (h2 : cumulus_clouds = 12 * cumulonimbus_clouds)
  (h3 : cumulonimbus_clouds = 3) : 
  cirrus_clouds = 144 :=
by sorry

end cirrus_clouds_count_l1_1293


namespace find_x_in_inches_l1_1652

noncomputable def x_value (x : ℝ) : Prop :=
  let area_larger_square := (4 * x) ^ 2
  let area_smaller_square := (3 * x) ^ 2
  let area_triangle := (1 / 2) * (3 * x) * (4 * x)
  let total_area := area_larger_square + area_smaller_square + area_triangle
  total_area = 1100 ∧ x = Real.sqrt (1100 / 31)

theorem find_x_in_inches (x : ℝ) : x_value x :=
by sorry

end find_x_in_inches_l1_1652


namespace solve_m_n_l1_1127

theorem solve_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 :=
sorry

end solve_m_n_l1_1127


namespace number_of_routes_A_to_B_l1_1547

theorem number_of_routes_A_to_B :
  (∃ f : ℕ × ℕ → ℕ,
  (∀ n m, f (n + 1, m) = f (n, m) + f (n + 1, m - 1)) ∧
  f (0, 0) = 1 ∧ 
  (∀ i, f (i, 0) = 1) ∧ 
  (∀ j, f (0, j) = 1) ∧ 
  f (3, 5) = 23) :=
sorry

end number_of_routes_A_to_B_l1_1547


namespace roots_of_quadratic_function_l1_1409

variable (a b x : ℝ)

theorem roots_of_quadratic_function (h : a + b = 0) : (b * x * x + a * x = 0) → (x = 0 ∨ x = 1) :=
by {sorry}

end roots_of_quadratic_function_l1_1409


namespace molecular_weight_of_7_moles_boric_acid_l1_1758

-- Define the given constants.
def atomic_weight_H : ℝ := 1.008
def atomic_weight_B : ℝ := 10.81
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula for boric acid.
def molecular_weight_H3BO3 : ℝ :=
  3 * atomic_weight_H + 1 * atomic_weight_B + 3 * atomic_weight_O

-- Define the number of moles.
def moles_boric_acid : ℝ := 7

-- Calculate the total weight for 7 moles of boric acid.
def total_weight_boric_acid : ℝ :=
  moles_boric_acid * molecular_weight_H3BO3

-- The target statement to prove.
theorem molecular_weight_of_7_moles_boric_acid :
  total_weight_boric_acid = 432.838 := by
  sorry

end molecular_weight_of_7_moles_boric_acid_l1_1758


namespace uma_income_l1_1287

theorem uma_income
  (x y : ℝ)
  (h1 : 8 * x - 7 * y = 2000)
  (h2 : 7 * x - 6 * y = 2000) :
  8 * x = 16000 := by
  sorry

end uma_income_l1_1287


namespace stamps_on_last_page_l1_1564

theorem stamps_on_last_page (total_books : ℕ) (pages_per_book : ℕ) (stamps_per_page_initial : ℕ) (stamps_per_page_new : ℕ)
    (full_books_new : ℕ) (pages_filled_seventh_book : ℕ) (total_stamps : ℕ) (stamps_in_seventh_book : ℕ) 
    (remaining_stamps : ℕ) :
    total_books = 10 →
    pages_per_book = 50 →
    stamps_per_page_initial = 8 →
    stamps_per_page_new = 12 →
    full_books_new = 6 →
    pages_filled_seventh_book = 37 →
    total_stamps = total_books * pages_per_book * stamps_per_page_initial →
    stamps_in_seventh_book = 4000 - (600 * full_books_new) →
    remaining_stamps = stamps_in_seventh_book - (pages_filled_seventh_book * stamps_per_page_new) →
    remaining_stamps = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end stamps_on_last_page_l1_1564


namespace expression_evaluation_l1_1505

theorem expression_evaluation (a : ℝ) (h : a = 9) : ( (a ^ (1 / 3)) / (a ^ (1 / 5)) ) = a^(2 / 15) :=
by
  sorry

end expression_evaluation_l1_1505


namespace sum_equidistant_terms_l1_1361

def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n m : ℕ, (n < m) → a (n+1) - a n = a (m+1) - a m

variable {a : ℕ → ℤ}

theorem sum_equidistant_terms (h_seq : is_arithmetic_sequence a)
  (h_4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end sum_equidistant_terms_l1_1361


namespace n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1_1012

theorem n_plus_floor_sqrt2_plus1_pow_n_is_odd (n : ℕ) (h : n > 0) : 
  Odd (n + ⌊(Real.sqrt 2 + 1) ^ n⌋) :=
by sorry

end n_plus_floor_sqrt2_plus1_pow_n_is_odd_l1_1012


namespace smallest_constant_l1_1709

theorem smallest_constant (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + b^2 + a * b) / c^2 ≥ 3 / 4 :=
sorry

end smallest_constant_l1_1709


namespace tetrahedron_edges_vertices_product_l1_1870

theorem tetrahedron_edges_vertices_product :
  let vertices := 4
  let edges := 6
  edges * vertices = 24 :=
by
  let vertices := 4
  let edges := 6
  sorry

end tetrahedron_edges_vertices_product_l1_1870


namespace pickle_to_tomato_ratio_l1_1001

theorem pickle_to_tomato_ratio 
  (mushrooms : ℕ) 
  (cherry_tomatoes : ℕ) 
  (pickles : ℕ) 
  (bacon_bits : ℕ) 
  (red_bacon_bits : ℕ) 
  (h1 : mushrooms = 3) 
  (h2 : cherry_tomatoes = 2 * mushrooms)
  (h3 : red_bacon_bits = 32)
  (h4 : bacon_bits = 3 * red_bacon_bits)
  (h5 : bacon_bits = 4 * pickles) : 
  pickles/cherry_tomatoes = 4 :=
by
  sorry

end pickle_to_tomato_ratio_l1_1001


namespace sin_330_eq_neg_half_l1_1528

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l1_1528


namespace no_three_by_three_red_prob_l1_1764

theorem no_three_by_three_red_prob : 
  ∃ (m n : ℕ), 
  Nat.gcd m n = 1 ∧ 
  m / n = 340 / 341 ∧ 
  m + n = 681 :=
by
  sorry

end no_three_by_three_red_prob_l1_1764


namespace gasoline_added_correct_l1_1430

def tank_capacity := 48
def initial_fraction := 3 / 4
def final_fraction := 9 / 10

def gasoline_at_initial_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_at_final_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_added (initial: ℝ) (final: ℝ) : ℝ := final - initial

theorem gasoline_added_correct (capacity: ℝ) (initial_fraction: ℝ) (final_fraction: ℝ)
  (h_capacity : capacity = 48) (h_initial : initial_fraction = 3 / 4) (h_final : final_fraction = 9 / 10) :
  gasoline_added (gasoline_at_initial_fraction capacity initial_fraction) (gasoline_at_final_fraction capacity final_fraction) = 7.2 :=
by
  sorry

end gasoline_added_correct_l1_1430


namespace emily_subtracts_99_from_50sq_to_get_49sq_l1_1473

-- Define the identity for squares
theorem emily_subtracts_99_from_50sq_to_get_49sq :
  ∀ (x : ℕ), (49 : ℕ) = (50 - 1) → (x = 50 → 49^2 = 50^2 - 99) := by
  intro x h1 h2
  sorry

end emily_subtracts_99_from_50sq_to_get_49sq_l1_1473


namespace distance_to_station_is_6_l1_1146

noncomputable def distance_man_walks (walking_speed1 walking_speed2 time_diff: ℝ) : ℝ :=
  let D := (time_diff * walking_speed1 * walking_speed2) / (walking_speed1 - walking_speed2)
  D

theorem distance_to_station_is_6 :
  distance_man_walks 5 6 (12 / 60) = 6 :=
by
  sorry

end distance_to_station_is_6_l1_1146


namespace probability_circle_containment_l1_1265

theorem probability_circle_containment :
  let a_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
  let circle_C_contained (a : ℕ) : Prop := a > 3
  let m : ℕ := (a_set.filter circle_C_contained).card
  let n : ℕ := a_set.card
  let p : ℚ := m / n
  p = 4 / 7 := 
by
  sorry

end probability_circle_containment_l1_1265


namespace speed_difference_valid_l1_1668

-- Definitions of the conditions
def speed (s : ℕ) : ℕ := s^2 + 2 * s

-- Theorem statement that needs to be proven
theorem speed_difference_valid : 
  (speed 5 - speed 3) = 20 :=
  sorry

end speed_difference_valid_l1_1668


namespace massive_crate_chocolate_bars_l1_1356

theorem massive_crate_chocolate_bars :
  (54 * 24 * 37 = 47952) :=
by
  sorry

end massive_crate_chocolate_bars_l1_1356


namespace proof_problem_l1_1741

def label_sum_of_domains_specified (labels: List Nat) (domains: List Nat) : Nat :=
  let relevant_labels := labels.filter (fun l => domains.contains l)
  relevant_labels.foldl (· + ·) 0

def label_product_of_continuous_and_invertible (labels: List Nat) (properties: List Bool) : Nat :=
  let relevant_labels := labels.zip properties |>.filter (fun (_, p) => p) |>.map (·.fst)
  relevant_labels.foldl (· * ·) 1

theorem proof_problem :
  label_sum_of_domains_specified [1, 2, 3, 4] [4] = 4 ∧ label_product_of_continuous_and_invertible [1, 2, 3, 4] [true, false, true, false] = 3 :=
by
  sorry

end proof_problem_l1_1741


namespace shaded_l_shaped_area_l1_1248

def square (side : ℕ) : ℕ := side * side
def rectangle (length width : ℕ) : ℕ := length * width

theorem shaded_l_shaped_area :
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  areaABCD - total_area_small_shapes = 20 :=
by
  let sideABCD := 6
  let sideEFGH := 2
  let sideIJKL := 2
  let widthMNPQ := 2
  let heightMNPQ := 4

  let areaABCD := square sideABCD
  let areaEFGH := square sideEFGH
  let areaIJKL := square sideIJKL
  let areaMNPQ := rectangle widthMNPQ heightMNPQ

  let total_area_small_shapes := 2 * areaEFGH + areaMNPQ

  have h : areaABCD - total_area_small_shapes = 20 := sorry
  exact h

end shaded_l_shaped_area_l1_1248


namespace cost_of_ox_and_sheep_l1_1290

variable (x y : ℚ)

theorem cost_of_ox_and_sheep :
  (5 * x + 2 * y = 10) ∧ (2 * x + 8 * y = 8) → (x = 16 / 9 ∧ y = 5 / 9) :=
by
  sorry

end cost_of_ox_and_sheep_l1_1290


namespace radius_comparison_l1_1334

theorem radius_comparison 
  (a b c : ℝ)
  (da db dc r ρ : ℝ)
  (h₁ : da ≤ r)
  (h₂ : db ≤ r)
  (h₃ : dc ≤ r)
  (h₄ : 1 / 2 * (a * da + b * db + c * dc) = ρ * ((a + b + c) / 2)) :
  r ≥ ρ := 
sorry

end radius_comparison_l1_1334


namespace joan_trip_time_l1_1960

-- Definitions of given conditions as parameters
def distance : ℕ := 480
def speed : ℕ := 60
def lunch_break_minutes : ℕ := 30
def bathroom_break_minutes : ℕ := 15
def number_of_bathroom_breaks : ℕ := 2

-- Conversion factors
def minutes_to_hours (m : ℕ) : ℚ := m / 60

-- Calculation of total time taken
def total_time : ℚ := 
  (distance / speed) + 
  (minutes_to_hours lunch_break_minutes) + 
  (number_of_bathroom_breaks * minutes_to_hours bathroom_break_minutes)

-- Statement of the problem
theorem joan_trip_time : total_time = 9 := 
  by 
    sorry

end joan_trip_time_l1_1960


namespace parabola_2_second_intersection_x_l1_1006

-- Definitions of the conditions in the problem
def parabola_1_intersects : Prop := 
  (∀ x : ℝ, (x = 10 ∨ x = 13) → (∃ y : ℝ, (x, y) ∈ ({p | p = (10, 0)} ∪ {p | p = (13, 0)})))

def parabola_2_intersects : Prop := 
  (∃ x : ℝ, x = 13)

def vertex_bisects_segment : Prop := 
  (∃ a : ℝ, 2 * 11.5 = a)

-- The theorem we want to prove
theorem parabola_2_second_intersection_x : 
  parabola_1_intersects ∧ parabola_2_intersects ∧ vertex_bisects_segment → 
  (∃ t : ℝ, t = 33) := 
  by
  sorry

end parabola_2_second_intersection_x_l1_1006


namespace triangle_side_difference_l1_1914

theorem triangle_side_difference (x : ℕ) : 3 < x ∧ x < 17 → (∃ a b : ℕ, 3 < a ∧ a < 17 ∧ 3 < b ∧ b < 17 ∧ a - b = 12) :=
by
  sorry

end triangle_side_difference_l1_1914


namespace Sarah_score_l1_1654

theorem Sarah_score (G S : ℕ) (h1 : S = G + 60) (h2 : (S + G) / 2 = 108) : S = 138 :=
by
  sorry

end Sarah_score_l1_1654


namespace inequality_condition_l1_1379

theorem inequality_condition {a b c : ℝ} :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (Real.sqrt (a^2 + b^2) < c) :=
by
  sorry

end inequality_condition_l1_1379


namespace sum_of_roots_of_quadratic_l1_1063

theorem sum_of_roots_of_quadratic :
  ∀ x : ℝ, x^2 + 2000*x - 2000 = 0 ->
  (∃ x1 x2 : ℝ, (x1 ≠ x2 ∧ x1^2 + 2000*x1 - 2000 = 0 ∧ x2^2 + 2000*x2 - 2000 = 0 ∧ x1 + x2 = -2000)) :=
sorry

end sum_of_roots_of_quadratic_l1_1063


namespace annual_interest_rate_of_second_investment_l1_1770

-- Definitions for the conditions
def total_income : ℝ := 575
def investment1 : ℝ := 3000
def rate1 : ℝ := 0.085
def income1 : ℝ := investment1 * rate1
def investment2 : ℝ := 5000
def target_income : ℝ := total_income - income1

-- Lean 4 statement to prove the annual simple interest rate of the second investment
theorem annual_interest_rate_of_second_investment : ∃ (r : ℝ), target_income = investment2 * (r / 100) ∧ r = 6.4 :=
by sorry

end annual_interest_rate_of_second_investment_l1_1770


namespace parallelogram_area_l1_1279

/-- The area of a parallelogram is given by the product of its base and height. 
Given a parallelogram ABCD with base BC of 4 units and height of 2 units, 
prove its area is 8 square units. --/
theorem parallelogram_area (base height : ℝ) (h_base : base = 4) (h_height : height = 2) : 
  base * height = 8 :=
by
  rw [h_base, h_height]
  norm_num
  done

end parallelogram_area_l1_1279


namespace max_value_x2y_l1_1500

theorem max_value_x2y : 
  ∃ (x y : ℕ), 
    7 * x + 4 * y = 140 ∧
    (∀ (x' y' : ℕ),
       7 * x' + 4 * y' = 140 → 
       x' ^ 2 * y' ≤ x ^ 2 * y) ∧
    x ^ 2 * y = 2016 :=
by {
  sorry
}

end max_value_x2y_l1_1500


namespace value_of_expression_l1_1542

theorem value_of_expression (a : ℚ) (h : a = 1/3) : (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by
  sorry

end value_of_expression_l1_1542


namespace valid_numbers_count_l1_1435

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d < 10

def count_valid_numbers : ℕ :=
  let first_digit_choices := 8 -- from 1 to 9 excluding 5
  let second_digit_choices := 8 -- from the digits (0-9 excluding 5 and first digit)
  let third_digit_choices := 7 -- from the digits (0-9 excluding 5 and first two digits)
  let fourth_digit_choices := 6 -- from the digits (0-9 excluding 5 and first three digits)
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem valid_numbers_count : count_valid_numbers = 2688 :=
  by
  sorry

end valid_numbers_count_l1_1435


namespace sulfuric_acid_percentage_l1_1748

theorem sulfuric_acid_percentage 
  (total_volume : ℝ)
  (first_solution_percentage : ℝ)
  (final_solution_percentage : ℝ)
  (second_solution_volume : ℝ)
  (expected_second_solution_percentage : ℝ) :
  total_volume = 60 ∧
  first_solution_percentage = 0.02 ∧
  final_solution_percentage = 0.05 ∧
  second_solution_volume = 18 →
  expected_second_solution_percentage = 12 :=
by
  sorry

end sulfuric_acid_percentage_l1_1748


namespace people_per_car_l1_1371

theorem people_per_car (total_people cars : ℕ) (h1 : total_people = 63) (h2 : cars = 9) :
  total_people / cars = 7 :=
by
  sorry

end people_per_car_l1_1371


namespace skittles_problem_l1_1015

def initial_skittles : ℕ := 76
def shared_skittles : ℕ := 72
def final_skittles (initial shared : ℕ) : ℕ := initial - shared

theorem skittles_problem : final_skittles initial_skittles shared_skittles = 4 := by
  sorry

end skittles_problem_l1_1015


namespace patrick_purchased_pencils_l1_1590

theorem patrick_purchased_pencils (c s : ℝ) : 
  (∀ n : ℝ, n * c = 1.375 * n * s ∧ (n * c - n * s = 30 * s) → n = 80) :=
by sorry

end patrick_purchased_pencils_l1_1590


namespace building_height_l1_1003

noncomputable def height_of_building (H_f L_f L_b : ℝ) : ℝ :=
  (H_f * L_b) / L_f

theorem building_height (H_f L_f L_b H_b : ℝ)
  (H_f_val : H_f = 17.5)
  (L_f_val : L_f = 40.25)
  (L_b_val : L_b = 28.75)
  (H_b_val : H_b = 12.4375) :
  height_of_building H_f L_f L_b = H_b := by
  rw [H_f_val, L_f_val, L_b_val, H_b_val]
  -- sorry to skip the proof
  sorry

end building_height_l1_1003


namespace original_number_is_842_l1_1013

theorem original_number_is_842 (x y z : ℕ) (h1 : x * z = y^2)
  (h2 : 100 * z + x = 100 * x + z - 594)
  (h3 : 10 * z + y = 10 * y + z - 18)
  (hx : x = 8) (hy : y = 4) (hz : z = 2) :
  100 * x + 10 * y + z = 842 :=
by
  sorry

end original_number_is_842_l1_1013


namespace trapezoid_inscribed_circles_radii_l1_1187

open Real

variables (a b m n : ℝ)
noncomputable def r := (a * sqrt b) / (sqrt a + sqrt b)
noncomputable def R := (b * sqrt a) / (sqrt a + sqrt b)

theorem trapezoid_inscribed_circles_radii
  (h : a < b)
  (hM : m = sqrt (a * b))
  (hN : m = sqrt (a * b)) :
  (r a b = (a * sqrt b) / (sqrt a + sqrt b)) ∧
  (R a b = (b * sqrt a) / (sqrt a + sqrt b)) :=
by
  sorry

end trapezoid_inscribed_circles_radii_l1_1187


namespace marbles_selection_l1_1396

theorem marbles_selection : 
  ∃ (n : ℕ), n = 1540 ∧ 
  ∃ marbles : Finset ℕ, marbles.card = 15 ∧
  ∃ rgb : Finset ℕ, rgb ⊆ marbles ∧ rgb.card = 3 ∧
  ∃ yellow : ℕ, yellow ∈ marbles ∧ yellow ∉ rgb ∧ 
  ∀ (selection : Finset ℕ), selection.card = 5 →
  (∃ red green blue : ℕ, red ∈ rgb ∧ green ∈ rgb ∧ blue ∈ rgb ∧ 
  (red ∈ selection ∨ green ∈ selection ∨ blue ∈ selection) ∧ yellow ∉ selection) → 
  (selection.card = 5) :=
by
  sorry

end marbles_selection_l1_1396


namespace sum_of_ages_l1_1957

-- Definitions based on given conditions
def J : ℕ := 19
def age_difference (B J : ℕ) : Prop := B - J = 32

-- Theorem stating the problem
theorem sum_of_ages (B : ℕ) (H : age_difference B J) : B + J = 70 :=
sorry

end sum_of_ages_l1_1957


namespace cube_difference_div_l1_1378

theorem cube_difference_div (a b : ℕ) (h_a : a = 64) (h_b : b = 27) : 
  (a^3 - b^3) / (a - b) = 6553 := by
  sorry

end cube_difference_div_l1_1378


namespace grade_point_average_one_third_classroom_l1_1552

theorem grade_point_average_one_third_classroom
  (gpa1 : ℝ) -- grade point average of one third of the classroom
  (gpa_rest : ℝ) -- grade point average of the rest of the classroom
  (gpa_whole : ℝ) -- grade point average of the whole classroom
  (h_rest : gpa_rest = 45)
  (h_whole : gpa_whole = 48) :
  gpa1 = 54 :=
by
  sorry

end grade_point_average_one_third_classroom_l1_1552


namespace evaluate_double_sum_l1_1463

theorem evaluate_double_sum :
  ∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m + 1) ^ 2 / (n + 1) / (m + n + 3) = 1 := by
  sorry

end evaluate_double_sum_l1_1463


namespace lindsey_savings_l1_1508

theorem lindsey_savings
  (september_savings : Nat := 50)
  (october_savings : Nat := 37)
  (november_savings : Nat := 11)
  (additional_savings : Nat := 25)
  (video_game_cost : Nat := 87)
  (total_savings := september_savings + october_savings + november_savings)
  (mom_bonus : Nat := if total_savings > 75 then additional_savings else 0)
  (final_amount := total_savings + mom_bonus - video_game_cost) :
  final_amount = 36 := by
  sorry

end lindsey_savings_l1_1508


namespace abs_x_equals_4_l1_1576

-- Define the points A and B as per the conditions
def point_A (x : ℝ) : ℝ := 3 + x
def point_B (x : ℝ) : ℝ := 3 - x

-- Define the distance between points A and B
def distance (x : ℝ) : ℝ := abs ((point_A x) - (point_B x))

theorem abs_x_equals_4 (x : ℝ) (h : distance x = 8) : abs x = 4 :=
by
  sorry

end abs_x_equals_4_l1_1576


namespace germination_rate_proof_l1_1467

def random_number_table := [[78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279],
                            [43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820],
                            [61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636],
                            [63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421],
                            [42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983]]

noncomputable def first_4_tested_seeds : List Nat :=
  let numbers_in_random_table := [390, 737, 924, 220, 372]
  numbers_in_random_table.filter (λ x => x < 850) |>.take 4

theorem germination_rate_proof :
  first_4_tested_seeds = [390, 737, 220, 372] := 
by 
  sorry

end germination_rate_proof_l1_1467


namespace arrange_in_order_l1_1130

noncomputable def x1 : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def x2 : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def x3 : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))
noncomputable def x4 : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))

theorem arrange_in_order : 
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 := 
by 
  sorry

end arrange_in_order_l1_1130


namespace election_votes_total_l1_1191

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total_l1_1191


namespace f_2202_minus_f_2022_l1_1695

-- Definitions and conditions
def f : ℕ+ → ℕ+ := sorry -- The exact function is provided through conditions and will be proven property-wise.

axiom f_increasing {a b : ℕ+} : a < b → f a < f b
axiom f_range (n : ℕ+) : ∃ m : ℕ+, f n = ⟨m, sorry⟩ -- ensuring f maps to ℕ+
axiom f_property (n : ℕ+) : f (f n) = 3 * n

-- Prove the statement
theorem f_2202_minus_f_2022 : f 2202 - f 2022 = 1638 :=
by sorry

end f_2202_minus_f_2022_l1_1695


namespace compute_four_at_seven_l1_1142

def operation (a b : ℤ) : ℤ :=
  5 * a - 2 * b

theorem compute_four_at_seven : operation 4 7 = 6 :=
by
  sorry

end compute_four_at_seven_l1_1142


namespace distance_to_destination_l1_1005

-- Conditions
def Speed : ℝ := 65 -- speed in km/hr
def Time : ℝ := 3   -- time in hours

-- Question to prove
theorem distance_to_destination : Speed * Time = 195 := by
  sorry

end distance_to_destination_l1_1005


namespace number_of_donuts_finished_l1_1587

-- Definitions from conditions
def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def total_spent : ℕ := 18

-- Theorem statement
theorem number_of_donuts_finished (H1 : ounces_per_donut = 2)
                                   (H2 : ounces_per_pot = 12)
                                   (H3 : cost_per_pot = 3)
                                   (H4 : total_spent = 18) : 
  ∃ n : ℕ, n = 36 :=
  sorry

end number_of_donuts_finished_l1_1587


namespace lisa_interest_l1_1245

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem lisa_interest (hP : ℝ := 1500) (hr : ℝ := 0.02) (hn : ℕ := 10) :
  (compound_interest hP hr hn - hP) = 328.49 :=
by
  sorry

end lisa_interest_l1_1245


namespace total_amount_proof_l1_1062

-- Define the relationships between x, y, and z in terms of the amounts received
variables (x y z : ℝ)

-- Given: For each rupee x gets, y gets 0.45 rupees and z gets 0.50 rupees
def relationship1 : Prop := ∀ (k : ℝ), y = 0.45 * k ∧ z = 0.50 * k ∧ x = k

-- Given: The share of y is Rs. 54
def condition1 : Prop := y = 54

-- The total amount x + y + z is Rs. 234
def total_amount (x y z : ℝ) : ℝ := x + y + z

-- Prove that the total amount is Rs. 234
theorem total_amount_proof (x y z : ℝ) (h1: relationship1 x y z) (h2: condition1 y) : total_amount x y z = 234 :=
sorry

end total_amount_proof_l1_1062


namespace figurine_cost_is_one_l1_1124

-- Definitions from the conditions
def cost_per_tv : ℕ := 50
def num_tvs : ℕ := 5
def num_figurines : ℕ := 10
def total_spent : ℕ := 260

-- The price of a single figurine
def cost_per_figurine (total_spent num_tvs cost_per_tv num_figurines : ℕ) : ℕ :=
  (total_spent - num_tvs * cost_per_tv) / num_figurines

-- The theorem statement
theorem figurine_cost_is_one : cost_per_figurine total_spent num_tvs cost_per_tv num_figurines = 1 :=
by
  sorry

end figurine_cost_is_one_l1_1124


namespace repeating_decimal_as_fraction_l1_1651

-- Define the repeating decimal
def repeating_decimal := 3 + (127 / 999)

-- State the goal
theorem repeating_decimal_as_fraction : repeating_decimal = (3124 / 999) := 
by 
  sorry

end repeating_decimal_as_fraction_l1_1651


namespace carA_speed_calc_l1_1465

-- Defining the conditions of the problem
def carA_time : ℕ := 8
def carB_speed : ℕ := 25
def carB_time : ℕ := 4
def distance_ratio : ℕ := 4
def carB_distance : ℕ := carB_speed * carB_time
def carA_distance : ℕ := distance_ratio * carB_distance

-- Mathematical statement to be proven
theorem carA_speed_calc : carA_distance / carA_time = 50 := by
  sorry

end carA_speed_calc_l1_1465


namespace num_handshakes_l1_1077

-- Definition of the conditions
def num_teams : Nat := 4
def women_per_team : Nat := 2
def total_women : Nat := num_teams * women_per_team
def handshakes_per_woman : Nat := total_women -1 - women_per_team

-- Statement of the problem to prove
theorem num_handshakes (h: total_women = 8) : (total_women * handshakes_per_woman) / 2 = 24 := 
by
  -- Proof goes here
  sorry

end num_handshakes_l1_1077


namespace find_height_of_triangular_prism_l1_1108

-- Define the conditions
def volume (V : ℝ) : Prop := V = 120
def base_side1 (a : ℝ) : Prop := a = 3
def base_side2 (b : ℝ) : Prop := b = 4

-- The final proof problem
theorem find_height_of_triangular_prism (V : ℝ) (a : ℝ) (b : ℝ) (h : ℝ) 
  (h1 : volume V) (h2 : base_side1 a) (h3 : base_side2 b) : h = 20 :=
by
  -- The actual proof goes here
  sorry

end find_height_of_triangular_prism_l1_1108


namespace calculation_error_l1_1760

theorem calculation_error (x y : ℕ) : (25 * x + 5 * y) = 25 * x + 5 * y :=
by
  sorry

end calculation_error_l1_1760


namespace octagon_non_intersecting_diagonals_l1_1561

-- Define what an octagon is
def octagon : Type := { vertices : Finset (Fin 8) // vertices.card = 8 }

-- Define non-intersecting diagonals in an octagon
def non_intersecting_diagonals (oct : octagon) : ℕ :=
  8  -- Given the cyclic pattern and star formation, we know the number is 8

-- The theorem we want to prove
theorem octagon_non_intersecting_diagonals (oct : octagon) : non_intersecting_diagonals oct = 8 :=
by sorry

end octagon_non_intersecting_diagonals_l1_1561


namespace part2_l1_1425

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) - x

theorem part2 (a : ℝ) (h : a > 0) (x : ℝ) : f a x < (a - 1) * Real.log a + a^2 := 
  sorry

end part2_l1_1425


namespace douglas_votes_in_county_X_l1_1902

theorem douglas_votes_in_county_X (V : ℝ) :
  (0.64 * (2 * V + V) - 0.4000000000000002 * V) / (2 * V) * 100 = 76 := by
sorry

end douglas_votes_in_county_X_l1_1902


namespace relationship_abc_l1_1623

noncomputable def a (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin x) / x
noncomputable def b (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := (Real.sin (x^3)) / (x^3)
noncomputable def c (x : ℝ) (hx : 0 < x ∧ x < 1) : ℝ := ((Real.sin x)^3) / (x^3)

theorem relationship_abc (x : ℝ) (hx : 0 < x ∧ x < 1) : b x hx > a x hx ∧ a x hx > c x hx :=
by
  sorry

end relationship_abc_l1_1623


namespace measure_of_arc_BD_l1_1730

-- Definitions for conditions
def diameter (A B M : Type) : Prop := sorry -- Placeholder definition for diameter
def chord (C D M : Type) : Prop := sorry -- Placeholder definition for chord intersecting at point M
def angle_measure (A B C : Type) (angle_deg: ℝ) : Prop := sorry -- Placeholder for angle measure
def arc_measure (C B : Type) (arc_deg: ℝ) : Prop := sorry -- Placeholder for arc measure

-- Main theorem to prove
theorem measure_of_arc_BD
  (A B C D M : Type)
  (h_diameter : diameter A B M)
  (h_chord : chord C D M)
  (h_angle_CMB : angle_measure C M B 73)
  (h_arc_BC : arc_measure B C 110) :
  ∃ (arc_BD : ℝ), arc_BD = 144 :=
by
  sorry

end measure_of_arc_BD_l1_1730


namespace new_students_joined_l1_1274

theorem new_students_joined (orig_avg_age new_avg_age : ℕ) (decrease_in_avg_age : ℕ) (orig_strength : ℕ) (new_students_avg_age : ℕ) :
  orig_avg_age = 40 ∧ new_avg_age = 36 ∧ decrease_in_avg_age = 4 ∧ orig_strength = 18 ∧ new_students_avg_age = 32 →
  ∃ x : ℕ, ((orig_strength * orig_avg_age) + (x * new_students_avg_age) = new_avg_age * (orig_strength + x)) ∧ x = 18 :=
by
  sorry

end new_students_joined_l1_1274


namespace total_jumps_l1_1620

def taehyung_jumps_per_day : ℕ := 56
def taehyung_days : ℕ := 3
def namjoon_jumps_per_day : ℕ := 35
def namjoon_days : ℕ := 4

theorem total_jumps : taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end total_jumps_l1_1620


namespace hall_width_l1_1575

theorem hall_width
  (L H E C : ℝ)
  (hL : L = 20)
  (hH : H = 5)
  (hE : E = 57000)
  (hC : C = 60) :
  ∃ w : ℝ, (w * 50 + 100) * C = E ∧ w = 17 :=
by
  use 17
  simp [hL, hH, hE, hC]
  sorry

end hall_width_l1_1575


namespace sum_of_numbers_l1_1835

theorem sum_of_numbers (a b c : ℝ) (h1 : 2 * a + b = 46) (h2 : b + 2 * c = 53) (h3 : 2 * c + a = 29) :
  a + b + c = 48.8333 :=
by
  sorry

end sum_of_numbers_l1_1835


namespace find_maximum_value_of_f_φ_has_root_l1_1570

open Set Real

noncomputable section

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := -6 * (sin x + cos x) - 3

-- Definition of the function φ(x)
def φ (x : ℝ) : ℝ := f x + 10

-- The assumptions on the interval
def interval := Icc 0 (π / 4)

-- Statement to prove that the maximum value of f(x) is -9
theorem find_maximum_value_of_f : ∀ x ∈ interval, f x ≤ -9 ∧ ∃ x_0 ∈ interval, f x_0 = -9 := sorry

-- Statement to prove that φ(x) has a root in the interval
theorem φ_has_root : ∃ x ∈ interval, φ x = 0 := sorry

end find_maximum_value_of_f_φ_has_root_l1_1570


namespace no_solution_exists_l1_1423

theorem no_solution_exists :
  ¬ ∃ a b : ℝ, a^2 + 3 * b^2 + 2 = 3 * a * b :=
by
  sorry

end no_solution_exists_l1_1423


namespace cover_square_with_rectangles_l1_1104

theorem cover_square_with_rectangles :
  ∃ n : ℕ, n = 24 ∧
  ∀ (rect_area : ℕ) (square_area : ℕ), rect_area = 2 * 3 → square_area = 12 * 12 → square_area / rect_area = n :=
by
  use 24
  sorry

end cover_square_with_rectangles_l1_1104


namespace system_of_equations_has_solution_l1_1979

theorem system_of_equations_has_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 :=
by
  sorry

end system_of_equations_has_solution_l1_1979


namespace total_cupcakes_baked_l1_1998

theorem total_cupcakes_baked
    (boxes : ℕ)
    (cupcakes_per_box : ℕ)
    (left_at_home : ℕ)
    (total_given_away : ℕ)
    (total_baked : ℕ)
    (h1 : boxes = 17)
    (h2 : cupcakes_per_box = 3)
    (h3 : left_at_home = 2)
    (h4 : total_given_away = boxes * cupcakes_per_box)
    (h5 : total_baked = total_given_away + left_at_home) :
    total_baked = 53 := by
  sorry

end total_cupcakes_baked_l1_1998


namespace motorcycles_count_l1_1134

/-- 
Prove that the number of motorcycles in the parking lot is 28 given the conditions:
1. Each car has 5 wheels (including one spare).
2. Each motorcycle has 2 wheels.
3. Each tricycle has 3 wheels.
4. There are 19 cars in the parking lot.
5. There are 11 tricycles in the parking lot.
6. Altogether all vehicles have 184 wheels.
-/
theorem motorcycles_count 
  (cars := 19) 
  (tricycles := 11) 
  (total_wheels := 184) 
  (wheels_per_car := 5) 
  (wheels_per_tricycle := 3) 
  (wheels_per_motorcycle := 2) :
  (184 - (19 * 5 + 11 * 3)) / 2 = 28 :=
by 
  sorry

end motorcycles_count_l1_1134


namespace complex_in_third_quadrant_l1_1779

theorem complex_in_third_quadrant (x : ℝ) : 
  (x^2 - 6*x + 5 < 0) ∧ (x - 2 < 0) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complex_in_third_quadrant_l1_1779


namespace marys_mother_paid_correct_total_l1_1961

def mary_and_friends_payment_per_person : ℕ := 1 -- $1 each
def number_of_people : ℕ := 3 -- Mary and two friends

def total_chicken_cost : ℕ := mary_and_friends_payment_per_person * number_of_people -- Total cost of the chicken

def beef_cost_per_pound : ℕ := 4 -- $4 per pound
def total_beef_pounds : ℕ := 3 -- 3 pounds of beef
def total_beef_cost : ℕ := beef_cost_per_pound * total_beef_pounds -- Total cost of the beef

def oil_cost : ℕ := 1 -- $1 for 1 liter of oil

def total_grocery_cost : ℕ := total_chicken_cost + total_beef_cost + oil_cost -- Total grocery cost

theorem marys_mother_paid_correct_total : total_grocery_cost = 16 := by
  -- Here you would normally provide the proof steps which we're skipping per instructions.
  sorry

end marys_mother_paid_correct_total_l1_1961


namespace remainder_when_divided_by_11_l1_1395

theorem remainder_when_divided_by_11 {k x : ℕ} (h : x = 66 * k + 14) : x % 11 = 3 :=
by
  sorry

end remainder_when_divided_by_11_l1_1395


namespace quadratic_solution_product_l1_1219

theorem quadratic_solution_product :
  let r := 9 / 2
  let s := -11
  (r + 4) * (s + 4) = -119 / 2 :=
by
  -- Define the quadratic equation and its solutions
  let r := 9 / 2
  let s := -11

  -- Prove the statement
  sorry

end quadratic_solution_product_l1_1219


namespace days_gumballs_last_l1_1381

def pairs_day_1 := 3
def gumballs_per_pair := 9
def gumballs_day_1 := pairs_day_1 * gumballs_per_pair

def pairs_day_2 := pairs_day_1 * 2
def gumballs_day_2 := pairs_day_2 * gumballs_per_pair

def pairs_day_3 := pairs_day_2 - 1
def gumballs_day_3 := pairs_day_3 * gumballs_per_pair

def total_gumballs := gumballs_day_1 + gumballs_day_2 + gumballs_day_3
def gumballs_eaten_per_day := 3

theorem days_gumballs_last : total_gumballs / gumballs_eaten_per_day = 42 :=
by
  sorry

end days_gumballs_last_l1_1381


namespace solve_for_k_l1_1446

theorem solve_for_k :
  ∀ (k : ℝ), (∃ x : ℝ, (3*x + 8)*(x - 6) = -50 + k*x) ↔
    k = -10 + 2*Real.sqrt 6 ∨ k = -10 - 2*Real.sqrt 6 := by
  sorry

end solve_for_k_l1_1446


namespace speed_of_current_11_00448_l1_1864

/-- 
  The speed at which a man can row a boat in still water is 25 kmph.
  He takes 7.999360051195905 seconds to cover 80 meters downstream.
  Prove that the speed of the current is 11.00448 km/h.
-/
theorem speed_of_current_11_00448 :
  let speed_in_still_water_kmph := 25
  let distance_m := 80
  let time_s := 7.999360051195905
  (distance_m / time_s) * 3600 / 1000 - speed_in_still_water_kmph = 11.00448 :=
by
  sorry

end speed_of_current_11_00448_l1_1864


namespace determine_m_l1_1969

noncomputable def has_equal_real_roots (m : ℝ) : Prop :=
  m ≠ 0 ∧ (m^2 - 8 * m = 0)

theorem determine_m (m : ℝ) (h : has_equal_real_roots m) : m = 8 :=
  sorry

end determine_m_l1_1969


namespace bipin_chandan_age_ratio_l1_1193

-- Define the condition statements
def AlokCurrentAge : Nat := 5
def BipinCurrentAge : Nat := 6 * AlokCurrentAge
def ChandanCurrentAge : Nat := 7 + 3

-- Define the ages after 10 years
def BipinAgeAfter10Years : Nat := BipinCurrentAge + 10
def ChandanAgeAfter10Years : Nat := ChandanCurrentAge + 10

-- Define the ratio and the statement to prove
def AgeRatio := BipinAgeAfter10Years / ChandanAgeAfter10Years

-- The theorem to prove the ratio is 2
theorem bipin_chandan_age_ratio : AgeRatio = 2 := by
  sorry

end bipin_chandan_age_ratio_l1_1193


namespace gcd_48_30_is_6_l1_1762

/-- Prove that the Greatest Common Divisor (GCD) of 48 and 30 is 6. -/
theorem gcd_48_30_is_6 : Int.gcd 48 30 = 6 := by
  sorry

end gcd_48_30_is_6_l1_1762


namespace stratified_sampling_l1_1049

theorem stratified_sampling :
  let total_employees := 150
  let middle_managers := 30
  let senior_managers := 10
  let selected_employees := 30
  let selection_probability := selected_employees / total_employees
  let selected_middle_managers := middle_managers * selection_probability
  let selected_senior_managers := senior_managers * selection_probability
  selected_middle_managers = 6 ∧ selected_senior_managers = 2 :=
by
  sorry

end stratified_sampling_l1_1049


namespace joan_socks_remaining_l1_1408

-- Definitions based on conditions
def total_socks : ℕ := 1200
def white_socks : ℕ := total_socks / 4
def blue_socks : ℕ := total_socks * 3 / 8
def red_socks : ℕ := total_socks / 6
def green_socks : ℕ := total_socks / 12
def white_socks_lost : ℕ := white_socks / 3
def blue_socks_sold : ℕ := blue_socks / 2
def remaining_white_socks : ℕ := white_socks - white_socks_lost
def remaining_blue_socks : ℕ := blue_socks - blue_socks_sold

-- Theorem to prove the total number of remaining socks
theorem joan_socks_remaining :
  remaining_white_socks + remaining_blue_socks + red_socks + green_socks = 725 := by
  sorry

end joan_socks_remaining_l1_1408


namespace distance_from_minus_one_is_four_or_minus_six_l1_1710

theorem distance_from_minus_one_is_four_or_minus_six :
  {x : ℝ | abs (x + 1) = 5} = {-6, 4} :=
sorry

end distance_from_minus_one_is_four_or_minus_six_l1_1710


namespace calculate_f_17_69_l1_1085

noncomputable def f (x y: ℕ) : ℚ := sorry

axiom f_self : ∀ x, f x x = x
axiom f_symm : ∀ x y, f x y = f y x
axiom f_add : ∀ x y, (x + y) * f x y = y * f x (x + y)

theorem calculate_f_17_69 : f 17 69 = 73.3125 := sorry

end calculate_f_17_69_l1_1085


namespace avg_age_decrease_l1_1874

/-- Define the original average age of the class -/
def original_avg_age : ℕ := 40

/-- Define the number of original students -/
def original_strength : ℕ := 17

/-- Define the average age of the new students -/
def new_students_avg_age : ℕ := 32

/-- Define the number of new students joining -/
def new_students_strength : ℕ := 17

/-- Define the total original age of the class -/
def total_original_age : ℕ := original_strength * original_avg_age

/-- Define the total age of the new students -/
def total_new_students_age : ℕ := new_students_strength * new_students_avg_age

/-- Define the new total strength of the class after joining of new students -/
def new_total_strength : ℕ := original_strength + new_students_strength

/-- Define the new total age of the class after joining of new students -/
def new_total_age : ℕ := total_original_age + total_new_students_age

/-- Define the new average age of the class -/
def new_avg_age : ℕ := new_total_age / new_total_strength

/-- Prove that the average age decreased by 4 years when the new students joined -/
theorem avg_age_decrease : original_avg_age - new_avg_age = 4 := by
  sorry

end avg_age_decrease_l1_1874


namespace fraction_of_calls_processed_by_team_B_l1_1514

theorem fraction_of_calls_processed_by_team_B
  (C_B : ℕ) -- the number of calls processed by each member of team B
  (B : ℕ)  -- the number of call center agents in team B
  (C_A : ℕ := C_B / 5) -- each member of team A processes 1/5 the number of calls as each member of team B
  (A : ℕ := 5 * B / 8) -- team A has 5/8 as many agents as team B
: 
  (B * C_B) / ((A * C_A) + (B * C_B)) = (8 / 9 : ℚ) :=
sorry

end fraction_of_calls_processed_by_team_B_l1_1514


namespace jacob_walked_8_miles_l1_1510

theorem jacob_walked_8_miles (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 := by
  -- conditions
  have hr : rate = 4 := h_rate
  have ht : time = 2 := h_time
  -- problem
  sorry

end jacob_walked_8_miles_l1_1510


namespace value_of_expression_l1_1140

theorem value_of_expression (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 53) :
  x^3 - y^3 - 2 * (x + y) + 10 = 2011 :=
sorry

end value_of_expression_l1_1140


namespace six_digit_number_condition_l1_1318

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end six_digit_number_condition_l1_1318


namespace determine_a_from_equation_l1_1976

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end determine_a_from_equation_l1_1976


namespace probability_three_consecutive_heads_four_tosses_l1_1593

theorem probability_three_consecutive_heads_four_tosses :
  let total_outcomes := 16
  let favorable_outcomes := 2
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 8 := by
    sorry

end probability_three_consecutive_heads_four_tosses_l1_1593


namespace binomial_expansion_max_coefficient_l1_1522

theorem binomial_expansion_max_coefficient (n : ℕ) (h : n > 0) 
  (h_max_coefficient: ∀ m : ℕ, m ≠ 5 → (Nat.choose n m ≤ Nat.choose n 5)) : 
  n = 10 :=
sorry

end binomial_expansion_max_coefficient_l1_1522


namespace abc_value_l1_1190

variables (a b c d e f : ℝ)
variables (h1 : b * c * d = 65)
variables (h2 : c * d * e = 750)
variables (h3 : d * e * f = 250)
variables (h4 : (a * f) / (c * d) = 0.6666666666666666)

theorem abc_value : a * b * c = 130 :=
by { sorry }

end abc_value_l1_1190


namespace limit_fraction_l1_1202

theorem limit_fraction :
  ∀ ε > 0, ∃ (N : ℕ), ∀ n ≥ N, |((4 * n - 1) / (2 * n + 1) : ℚ) - 2| < ε := 
  by sorry

end limit_fraction_l1_1202


namespace arithmetic_sequence_ratio_a10_b10_l1_1763

variable {a : ℕ → ℕ} {b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- We assume S_n and T_n are the sums of the first n terms of sequences a and b respectively.
-- We also assume the provided ratio condition between S_n and T_n.
axiom sum_of_first_n_terms_a (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
axiom sum_of_first_n_terms_b (n : ℕ) : T n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1))) / 2
axiom ratio_condition (n : ℕ) : (S n) / (T n) = (3 * n - 1) / (2 * n + 3)

theorem arithmetic_sequence_ratio_a10_b10 : (a 10) / (b 10) = 56 / 41 :=
by sorry

end arithmetic_sequence_ratio_a10_b10_l1_1763


namespace tangent_line_values_l1_1046

theorem tangent_line_values (m : ℝ) :
  (∃ s : ℝ, 3 * s^2 = 12 ∧ 12 * s + m = s^3 - 2) ↔ (m = -18 ∨ m = 14) :=
by
  sorry

end tangent_line_values_l1_1046


namespace Yoongi_has_smaller_number_l1_1742

def Jungkook_number : ℕ := 6 + 3
def Yoongi_number : ℕ := 4

theorem Yoongi_has_smaller_number : Yoongi_number < Jungkook_number :=
by
  exact sorry

end Yoongi_has_smaller_number_l1_1742


namespace martian_angle_conversion_l1_1525

-- Defines the full circle measurements
def full_circle_clerts : ℕ := 600
def full_circle_degrees : ℕ := 360
def angle_degrees : ℕ := 60

-- The main statement to prove
theorem martian_angle_conversion : 
    (full_circle_clerts * angle_degrees) / full_circle_degrees = 100 :=
by
  sorry  

end martian_angle_conversion_l1_1525


namespace f_1993_of_3_l1_1493

def f (x : ℚ) := (1 + x) / (1 - 3 * x)

def f_n (x : ℚ) : ℕ → ℚ
| 0 => x
| (n + 1) => f (f_n x n)

theorem f_1993_of_3 :
  f_n 3 1993 = 1 / 5 :=
sorry

end f_1993_of_3_l1_1493


namespace geometric_sequence_common_ratio_l1_1990

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : 3 * a 0 + 2 * a 1 = a 2 / 0.5) :
  q = 3 :=
  sorry

end geometric_sequence_common_ratio_l1_1990


namespace midpoint_chord_hyperbola_l1_1306

theorem midpoint_chord_hyperbola (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (∃ (mx my : ℝ), (mx / a^2 + my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2))) →
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) →
  ∃ (mx my : ℝ), (mx / a^2 - my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2) := 
sorry

end midpoint_chord_hyperbola_l1_1306


namespace find_value_l1_1194

variables (x1 x2 y1 y2 : ℝ)

def condition1 := x1 ^ 2 + 5 * x2 ^ 2 = 10
def condition2 := x2 * y1 - x1 * y2 = 5
def condition3 := x1 * y1 + 5 * x2 * y2 = Real.sqrt 105

theorem find_value (h1 : condition1 x1 x2) (h2 : condition2 x1 x2 y1 y2) (h3 : condition3 x1 x2 y1 y2) :
  y1 ^ 2 + 5 * y2 ^ 2 = 23 :=
sorry

end find_value_l1_1194


namespace sum_of_cube_faces_l1_1332

theorem sum_of_cube_faces (a d b e c f : ℕ) (h1: a > 0) (h2: d > 0) (h3: b > 0) (h4: e > 0) (h5: c > 0) (h6: f > 0)
(h7 : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1491) :
  a + d + b + e + c + f = 41 := 
sorry

end sum_of_cube_faces_l1_1332


namespace xiaoli_estimate_larger_l1_1700

theorem xiaoli_estimate_larger (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : 
  (1.1 * x) / (0.9 * y) > x / y :=
by
  sorry

end xiaoli_estimate_larger_l1_1700


namespace bob_cleaning_time_l1_1751

-- Define the conditions
def timeAlice : ℕ := 30
def fractionBob : ℚ := 1 / 3

-- Define the proof problem
theorem bob_cleaning_time : (fractionBob * timeAlice : ℚ) = 10 := by
  sorry

end bob_cleaning_time_l1_1751


namespace potatoes_left_l1_1578

def p_initial : ℕ := 8
def p_eaten : ℕ := 3
def p_left : ℕ := p_initial - p_eaten

theorem potatoes_left : p_left = 5 := by
  sorry

end potatoes_left_l1_1578


namespace number_of_integers_divisible_by_18_or_21_but_not_both_l1_1911

theorem number_of_integers_divisible_by_18_or_21_but_not_both :
  let num_less_2019_div_by_18 := 112
  let num_less_2019_div_by_21 := 96
  let num_less_2019_div_by_both := 16
  num_less_2019_div_by_18 + num_less_2019_div_by_21 - 2 * num_less_2019_div_by_both = 176 :=
by
  sorry

end number_of_integers_divisible_by_18_or_21_but_not_both_l1_1911


namespace remainder_17_pow_45_div_5_l1_1733

theorem remainder_17_pow_45_div_5 : (17 ^ 45) % 5 = 2 :=
by
  -- proof goes here
  sorry

end remainder_17_pow_45_div_5_l1_1733


namespace Jean_spots_l1_1803

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l1_1803


namespace find_divisor_l1_1129

def positive_integer := {e : ℕ // e > 0}

theorem find_divisor (d : ℕ) :
  (∃ e : positive_integer, (e.val % 13 = 2)) →
  (∃ n : ℕ, n < 180 ∧ n % d = 5 ∧ ∀ m < 180, m % d = 5 → m = n) →
  d = 175 :=
by
  sorry

end find_divisor_l1_1129


namespace num_races_necessary_l1_1244

/-- There are 300 sprinters registered for a 200-meter dash at a local track meet,
where the track has only 8 lanes. In each race, 3 of the competitors advance to the
next round, while the rest are eliminated immediately. Determine how many races are
needed to identify the champion sprinter. -/
def num_races_to_champion (total_sprinters : ℕ) (lanes : ℕ) (advance_per_race : ℕ) : ℕ :=
  if h : advance_per_race < lanes ∧ lanes > 0 then
    let eliminations_per_race := lanes - advance_per_race
    let total_eliminations := total_sprinters - 1
    Nat.ceil (total_eliminations / eliminations_per_race)
  else
    0

theorem num_races_necessary
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (advance_per_race : ℕ)
  (h_total_sprinters : total_sprinters = 300)
  (h_lanes : lanes = 8)
  (h_advance_per_race : advance_per_race = 3) :
  num_races_to_champion total_sprinters lanes advance_per_race = 60 := by
  sorry

end num_races_necessary_l1_1244


namespace hyperbolas_same_asymptotes_l1_1171

-- Define the given hyperbolas
def hyperbola1 (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
def hyperbola2 (x y M : ℝ) : Prop := (y^2 / 25) - (x^2 / M) = 1

-- The main theorem statement
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, hyperbola1 x y → hyperbola2 x y M) ↔ M = 225/16 :=
by
  sorry

end hyperbolas_same_asymptotes_l1_1171


namespace cost_per_person_l1_1714

-- Definitions based on conditions
def totalCost : ℕ := 13500
def numberOfFriends : ℕ := 15

-- Main statement
theorem cost_per_person : totalCost / numberOfFriends = 900 :=
by sorry

end cost_per_person_l1_1714


namespace fewest_printers_l1_1391

theorem fewest_printers (x y : ℕ) (h1 : 350 * x = 200 * y) : x + y = 11 := 
by
  sorry

end fewest_printers_l1_1391


namespace Linda_sold_7_tees_l1_1165

variables (T : ℕ)
variables (jeans_price tees_price total_money_from_jeans total_money total_money_from_tees : ℕ)
variables (jeans_sold : ℕ)

def tees_sold :=
  jeans_price = 11 ∧ tees_price = 8 ∧ jeans_sold = 4 ∧
  total_money = 100 ∧ total_money_from_jeans = jeans_sold * jeans_price ∧
  total_money_from_tees = total_money - total_money_from_jeans ∧
  T = total_money_from_tees / tees_price
  
theorem Linda_sold_7_tees (h : tees_sold T jeans_price tees_price total_money_from_jeans total_money total_money_from_tees jeans_sold) : T = 7 :=
by
  sorry

end Linda_sold_7_tees_l1_1165


namespace square_area_l1_1685

theorem square_area (x : ℝ) (A B C D E F : ℝ)
  (h1 : E = x / 3)
  (h2 : F = (2 * x) / 3)
  (h3 : abs (B - E) = 40)
  (h4 : abs (E - F) = 40)
  (h5 : abs (F - D) = 40) :
  x^2 = 2880 :=
by
  -- Main proof here
  sorry

end square_area_l1_1685


namespace sunflower_packets_correct_l1_1366

namespace ShyneGarden

-- Define the given conditions
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def eggplant_packets_bought := 4
def total_plants := 116

-- Define the function to calculate the number of sunflower packets bought
def sunflower_packets_bought (eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants : ℕ) : ℕ :=
  (total_plants - (eggplant_packets_bought * eggplants_per_packet)) / sunflowers_per_packet

-- State the theorem to prove the number of sunflower packets
theorem sunflower_packets_correct :
  sunflower_packets_bought eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants = 6 :=
by
  sorry

end ShyneGarden

end sunflower_packets_correct_l1_1366


namespace abc_perfect_ratio_l1_1650

theorem abc_perfect_ratio {a b c : ℚ} (h1 : ∃ t : ℤ, a + b + c = t ∧ a^2 + b^2 + c^2 = t) :
  ∃ (p q : ℤ), (abc = p^3 / q^2) ∧ (IsCoprime p q) := 
sorry

end abc_perfect_ratio_l1_1650


namespace fraction_to_decimal_l1_1826

theorem fraction_to_decimal : (7 / 32 : ℚ) = 0.21875 := 
by {
  sorry
}

end fraction_to_decimal_l1_1826


namespace medicine_supply_duration_l1_1131

theorem medicine_supply_duration
  (pills_per_three_days : ℚ := 1 / 3)
  (total_pills : ℕ := 60)
  (days_per_month : ℕ := 30) :
  (((total_pills : ℚ) * ( 3 / pills_per_three_days)) / days_per_month) = 18 := sorry

end medicine_supply_duration_l1_1131


namespace people_with_diploma_percentage_l1_1697

-- Definitions of the given conditions
def P_j_and_not_d := 0.12
def P_not_j_and_d := 0.15
def P_j := 0.40

-- Definitions for intermediate values
def P_not_j := 1 - P_j
def P_not_j_d := P_not_j * P_not_j_and_d

-- Definition of the result to prove
def P_d := (P_j - P_j_and_not_d) + P_not_j_d

theorem people_with_diploma_percentage : P_d = 0.43 := by
  -- Placeholder for the proof
  sorry

end people_with_diploma_percentage_l1_1697


namespace smallest_integer_n_satisfying_inequality_l1_1461

theorem smallest_integer_n_satisfying_inequality 
  (x y z : ℝ) : 
  (x^2 + y^2 + z^2)^2 ≤ 3 * (x^4 + y^4 + z^4) :=
sorry

end smallest_integer_n_satisfying_inequality_l1_1461


namespace arrange_books_l1_1259

-- Definition of the problem
def total_books : ℕ := 5 + 3

-- Definition of the combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Prove that arranging 5 copies of Introduction to Geometry and 
-- 3 copies of Introduction to Number Theory into total_books positions can be done in 56 ways.
theorem arrange_books : combination total_books 5 = 56 := by
  sorry

end arrange_books_l1_1259


namespace find_m_of_parabola_and_line_l1_1310

theorem find_m_of_parabola_and_line (k m x1 x2 : ℝ) 
  (h_parabola_line : ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} → 
                                   y = k * x + m → true)
  (h_intersection : x1 * x2 = -4) : m = 1 := 
sorry

end find_m_of_parabola_and_line_l1_1310
