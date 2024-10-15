import Mathlib

namespace NUMINAMATH_GPT_f_1986_eq_one_l807_80780

def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b) + 1
axiom f_one : f 1 = 1

theorem f_1986_eq_one : f 1986 = 1 :=
sorry

end NUMINAMATH_GPT_f_1986_eq_one_l807_80780


namespace NUMINAMATH_GPT_average_weight_of_section_A_l807_80793

theorem average_weight_of_section_A (nA nB : ℕ) (WB WC : ℝ) (WA : ℝ) :
  nA = 50 →
  nB = 40 →
  WB = 70 →
  WC = 58.89 →
  50 * WA + 40 * WB = 58.89 * 90 →
  WA = 50.002 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_average_weight_of_section_A_l807_80793


namespace NUMINAMATH_GPT_quotient_of_integers_l807_80701

variable {x y : ℤ}

theorem quotient_of_integers (h : 1996 * x + y / 96 = x + y) : 
  (x / y = 1 / 2016) ∨ (y / x = 2016) := by
  sorry

end NUMINAMATH_GPT_quotient_of_integers_l807_80701


namespace NUMINAMATH_GPT_largest_of_six_consecutive_sum_2070_is_347_l807_80785

theorem largest_of_six_consecutive_sum_2070_is_347 (n : ℕ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 2070 → n + 5 = 347 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_largest_of_six_consecutive_sum_2070_is_347_l807_80785


namespace NUMINAMATH_GPT_least_integer_value_x_l807_80773

theorem least_integer_value_x (x : ℤ) (h : |(2 : ℤ) * x + 3| ≤ 12) : x = -7 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_x_l807_80773


namespace NUMINAMATH_GPT_max_annual_profit_at_x_9_l807_80731

noncomputable def annual_profit (x : ℝ) : ℝ :=
if h : (0 < x ∧ x ≤ 10) then
  8.1 * x - x^3 / 30 - 10
else
  98 - 1000 / (3 * x) - 2.7 * x

theorem max_annual_profit_at_x_9 (x : ℝ) (h1 : 0 < x) (h2 : x ≤ 10) :
  annual_profit x ≤ annual_profit 9 :=
sorry

end NUMINAMATH_GPT_max_annual_profit_at_x_9_l807_80731


namespace NUMINAMATH_GPT_problem1_problem2_l807_80715

-- Problem 1: Simplification and Evaluation
theorem problem1 (x : ℝ) : (x = -3) → 
  ((x^2 - 6*x + 9) / (x^2 - 1)) / ((x^2 - 3*x) / (x + 1))
  = -1 / 2 := sorry

-- Problem 2: Solving the Equation
theorem problem2 (x : ℝ) : 
  (∀ y, (y = x) → 
    (y / (y + 1) = 2*y / (3*y + 3) - 1)) → x = -3 / 4 := sorry

end NUMINAMATH_GPT_problem1_problem2_l807_80715


namespace NUMINAMATH_GPT_max_m_value_l807_80747

theorem max_m_value (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^4 + 16 * m + 8 = k * (k + 1)) : m ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_m_value_l807_80747


namespace NUMINAMATH_GPT_already_installed_windows_l807_80712

-- Definitions based on given conditions
def total_windows : ℕ := 9
def hours_per_window : ℕ := 6
def remaining_hours : ℕ := 18

-- Main statement to prove
theorem already_installed_windows : (total_windows - remaining_hours / hours_per_window) = 6 :=
by
  -- To prove: total_windows - (remaining_hours / hours_per_window) = 6
  -- This step is intentionally left incomplete (proof to be filled in by the user)
  sorry

end NUMINAMATH_GPT_already_installed_windows_l807_80712


namespace NUMINAMATH_GPT_pen_and_notebook_cost_l807_80710

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 15 * p + 5 * n = 130 ∧ p > n ∧ p + n = 10 := by
  sorry

end NUMINAMATH_GPT_pen_and_notebook_cost_l807_80710


namespace NUMINAMATH_GPT_find_number_l807_80704

theorem find_number (x : ℝ) (h : ((x / 3) * 24) - 7 = 41) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l807_80704


namespace NUMINAMATH_GPT_center_of_hyperbola_l807_80718

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end NUMINAMATH_GPT_center_of_hyperbola_l807_80718


namespace NUMINAMATH_GPT_prank_people_combinations_l807_80769

theorem prank_people_combinations (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (hMonday : Monday = 2)
  (hTuesday : Tuesday = 3)
  (hWednesday : Wednesday = 6)
  (hThursday : Thursday = 4)
  (hFriday : Friday = 3) :
  Monday * Tuesday * Wednesday * Thursday * Friday = 432 :=
  by sorry

end NUMINAMATH_GPT_prank_people_combinations_l807_80769


namespace NUMINAMATH_GPT_find_integer_x_l807_80755

theorem find_integer_x : ∃ x : ℤ, x^5 - 3 * x^2 = 216 ∧ x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_integer_x_l807_80755


namespace NUMINAMATH_GPT_area_between_curves_l807_80736

-- Function definitions:
def quartic (a b c d e x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e
def line (p q x : ℝ) : ℝ := p * x + q

-- Conditions:
variables (a b c d e p q α β : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (α_lt_β : α < β)
variable (touch_at_α : quartic a b c d e α = line p q α ∧ deriv (quartic a b c d e) α = p)
variable (touch_at_β : quartic a b c d e β = line p q β ∧ deriv (quartic a b c d e) β = p)

-- Theorem:
theorem area_between_curves :
  ∫ x in α..β, |quartic a b c d e x - line p q x| = (a * (β - α)^5) / 30 :=
by sorry

end NUMINAMATH_GPT_area_between_curves_l807_80736


namespace NUMINAMATH_GPT_ratio_of_amount_divided_to_total_savings_is_half_l807_80760

theorem ratio_of_amount_divided_to_total_savings_is_half :
  let husband_weekly_contribution := 335
  let wife_weekly_contribution := 225
  let weeks_in_six_months := 6 * 4
  let total_weekly_contribution := husband_weekly_contribution + wife_weekly_contribution
  let total_savings := total_weekly_contribution * weeks_in_six_months
  let amount_per_child := 1680
  let number_of_children := 4
  let total_amount_divided := amount_per_child * number_of_children
  (total_amount_divided : ℝ) / total_savings = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_amount_divided_to_total_savings_is_half_l807_80760


namespace NUMINAMATH_GPT_smaller_number_l807_80787

theorem smaller_number (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 16) : y = 4 := by
  sorry

end NUMINAMATH_GPT_smaller_number_l807_80787


namespace NUMINAMATH_GPT_who_is_who_l807_80728

-- Defining the structure and terms
structure Brother :=
  (name : String)
  (has_purple_card : Bool)

-- Conditions
def first_brother := Brother.mk "Tralalya" true
def second_brother := Brother.mk "Trulalya" false

/-- Proof that the names and cards of the brothers are as stated. -/
theorem who_is_who :
  ((first_brother.name = "Tralalya" ∧ first_brother.has_purple_card = false) ∧
   (second_brother.name = "Trulalya" ∧ second_brother.has_purple_card = true)) :=
by sorry

end NUMINAMATH_GPT_who_is_who_l807_80728


namespace NUMINAMATH_GPT_calculation_l807_80776

theorem calculation : 
  let a := 20 / 9 
  let b := -53 / 4 
  (⌈ a * ⌈ b ⌉ ⌉ - ⌊ a * ⌊ b ⌋ ⌋) = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculation_l807_80776


namespace NUMINAMATH_GPT_value_of_x_l807_80738

theorem value_of_x (x : ℝ) :
  (4 / x) * 12 = 8 ↔ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l807_80738


namespace NUMINAMATH_GPT_count_four_digit_numbers_divisible_by_5_ending_in_45_l807_80771

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_divisible_by_5_ending_in_45_l807_80771


namespace NUMINAMATH_GPT_marbles_per_friend_l807_80745

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) (h_total : total_marbles = 30) (h_friends : num_friends = 5) :
  total_marbles / num_friends = 6 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_marbles_per_friend_l807_80745


namespace NUMINAMATH_GPT_rosie_laps_l807_80784

theorem rosie_laps (lou_distance : ℝ) (track_length : ℝ) (lou_speed_factor : ℝ) (rosie_speed_multiplier : ℝ) 
    (number_of_laps_by_lou : ℝ) (number_of_laps_by_rosie : ℕ) :
  lou_distance = 3 ∧ 
  track_length = 1 / 4 ∧ 
  lou_speed_factor = 0.75 ∧ 
  rosie_speed_multiplier = 2 ∧ 
  number_of_laps_by_lou = lou_distance / track_length ∧ 
  number_of_laps_by_rosie = rosie_speed_multiplier * number_of_laps_by_lou → 
  number_of_laps_by_rosie = 18 := 
sorry

end NUMINAMATH_GPT_rosie_laps_l807_80784


namespace NUMINAMATH_GPT_area_of_new_geometric_figure_correct_l807_80733

noncomputable def area_of_new_geometric_figure (a b : ℝ) : ℝ := 
  let d := Real.sqrt (a^2 + b^2)
  a * b + (b * d) / 4

theorem area_of_new_geometric_figure_correct (a b : ℝ) :
  area_of_new_geometric_figure a b = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_new_geometric_figure_correct_l807_80733


namespace NUMINAMATH_GPT_prove_a_eq_1_l807_80799

variables {a b c d k m : ℕ}
variables (h_odd_a : a%2 = 1) 
          (h_odd_b : b%2 = 1) 
          (h_odd_c : c%2 = 1) 
          (h_odd_d : d%2 = 1)
          (h_a_pos : 0 < a) 
          (h_ineq1 : a < b) 
          (h_ineq2 : b < c) 
          (h_ineq3 : c < d)
          (h_eqn1 : a * d = b * c)
          (h_eqn2 : a + d = 2^k) 
          (h_eqn3 : b + c = 2^m)

theorem prove_a_eq_1 
  (h_odd_a : a%2 = 1) 
  (h_odd_b : b%2 = 1) 
  (h_odd_c : c%2 = 1) 
  (h_odd_d : d%2 = 1)
  (h_a_pos : 0 < a) 
  (h_ineq1 : a < b) 
  (h_ineq2 : b < c) 
  (h_ineq3 : c < d)
  (h_eqn1 : a * d = b * c)
  (h_eqn2 : a + d = 2^k) 
  (h_eqn3 : b + c = 2^m) :
  a = 1 := by
  sorry

end NUMINAMATH_GPT_prove_a_eq_1_l807_80799


namespace NUMINAMATH_GPT_total_dots_not_visible_l807_80700

-- Define the total dot sum for each die
def sum_of_dots_per_die : Nat := 1 + 2 + 3 + 4 + 5 + 6

-- Define the total number of dice
def number_of_dice : Nat := 4

-- Calculate the total dot sum for all dice
def total_dots_all_dice : Nat := sum_of_dots_per_die * number_of_dice

-- Sum of visible dots
def sum_of_visible_dots : Nat := 1 + 1 + 2 + 2 + 3 + 3 + 4 + 5 + 6 + 6

-- Prove the total dots not visible
theorem total_dots_not_visible : total_dots_all_dice - sum_of_visible_dots = 51 := by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l807_80700


namespace NUMINAMATH_GPT_polynomial_form_l807_80726

def is_even_poly (P : ℝ → ℝ) : Prop := 
  ∀ x, P x = P (-x)

theorem polynomial_form (P : ℝ → ℝ) (hP : ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) : 
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x ^ 4 + b * x ^ 2 := 
  sorry

end NUMINAMATH_GPT_polynomial_form_l807_80726


namespace NUMINAMATH_GPT_trader_profit_l807_80797

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def purchase_price (P : ℝ) : ℝ := 0.8 * P
noncomputable def depreciation1 (P : ℝ) : ℝ := 0.04 * P
noncomputable def depreciation2 (P : ℝ) : ℝ := 0.038 * P
noncomputable def value_after_depreciation (P : ℝ) : ℝ := 0.722 * P
noncomputable def taxes (P : ℝ) : ℝ := 0.024 * P
noncomputable def insurance (P : ℝ) : ℝ := 0.032 * P
noncomputable def maintenance (P : ℝ) : ℝ := 0.01 * P
noncomputable def total_cost (P : ℝ) : ℝ := value_after_depreciation P + taxes P + insurance P + maintenance P
noncomputable def selling_price (P : ℝ) : ℝ := 1.70 * total_cost P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def profit_percent (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) : profit_percent P = 33.96 :=
  by
    sorry

end NUMINAMATH_GPT_trader_profit_l807_80797


namespace NUMINAMATH_GPT_snowman_volume_l807_80743

theorem snowman_volume (r1 r2 r3 : ℝ) (V1 V2 V3 : ℝ) (π : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 6) (h3 : r3 = 8) 
  (hV1 : V1 = (4/3) * π * (r1^3)) 
  (hV2 : V2 = (4/3) * π * (r2^3)) 
  (hV3 : V3 = (4/3) * π * (r3^3)) :
  V1 + V2 + V3 = (3168/3) * π :=
by 
  sorry

end NUMINAMATH_GPT_snowman_volume_l807_80743


namespace NUMINAMATH_GPT_line_eq_l807_80752

theorem line_eq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_eq : 1 / a + 9 / b = 1) (h_min_interp : a + b = 16) : 
  ∃ l : ℝ × ℝ → ℝ, ∀ x y : ℝ, l (x, y) = 3 * x + y - 12 :=
by
  sorry

end NUMINAMATH_GPT_line_eq_l807_80752


namespace NUMINAMATH_GPT_length_of_rectangle_l807_80789

-- Given conditions as per the problem statement
variables {s l : ℝ} -- side length of the square, length of the rectangle
def width_rectangle : ℝ := 10 -- width of the rectangle

-- Conditions
axiom sq_perimeter : 4 * s = 200
axiom area_relation : s^2 = 5 * (l * width_rectangle)

-- Goal to prove
theorem length_of_rectangle : l = 50 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l807_80789


namespace NUMINAMATH_GPT_sector_area_l807_80729

noncomputable def radius_of_sector (l α : ℝ) : ℝ := l / α

noncomputable def area_of_sector (r l : ℝ) : ℝ := (1 / 2) * r * l

theorem sector_area {α l S : ℝ} (hα : α = 2) (hl : l = 3 * Real.pi) (hS : S = 9 * Real.pi ^ 2 / 4) :
  area_of_sector (radius_of_sector l α) l = S := 
by 
  rw [hα, hl, hS]
  rw [radius_of_sector, area_of_sector]
  sorry

end NUMINAMATH_GPT_sector_area_l807_80729


namespace NUMINAMATH_GPT_size_of_first_file_l807_80748

theorem size_of_first_file (internet_speed_mbps : ℝ) (time_hours : ℝ) (file2_mbps : ℝ) (file3_mbps : ℝ) (total_downloaded_mbps : ℝ) :
  internet_speed_mbps = 2 →
  time_hours = 2 →
  file2_mbps = 90 →
  file3_mbps = 70 →
  total_downloaded_mbps = internet_speed_mbps * 60 * time_hours →
  total_downloaded_mbps - (file2_mbps + file3_mbps) = 80 :=
by
  intros
  sorry

end NUMINAMATH_GPT_size_of_first_file_l807_80748


namespace NUMINAMATH_GPT_problem1_problem2_l807_80795

-- Problem (1)
theorem problem1 (x : ℝ) : (2 * |x - 1| ≥ 1) ↔ (x ≤ 1/2 ∨ x ≥ 3/2) := sorry

-- Problem (2)
theorem problem2 (a : ℝ) (h : a > 0) : (∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) ↔ a ≥ 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l807_80795


namespace NUMINAMATH_GPT_ratio_of_chicken_to_beef_l807_80757

theorem ratio_of_chicken_to_beef
  (beef_pounds : ℕ)
  (chicken_price_per_pound : ℕ)
  (total_cost : ℕ)
  (beef_price_per_pound : ℕ)
  (beef_cost : ℕ)
  (chicken_cost : ℕ)
  (chicken_pounds : ℕ) :
  beef_pounds = 1000 →
  beef_price_per_pound = 8 →
  total_cost = 14000 →
  beef_cost = beef_pounds * beef_price_per_pound →
  chicken_cost = total_cost - beef_cost →
  chicken_price_per_pound = 3 →
  chicken_pounds = chicken_cost / chicken_price_per_pound →
  chicken_pounds / beef_pounds = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_of_chicken_to_beef_l807_80757


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l807_80762

-- Definitions of conversion rates used in the conditions
def sq_m_to_sq_dm : Nat := 100
def hectare_to_sq_m : Nat := 10000
def sq_cm_to_sq_dm_div : Nat := 100
def sq_km_to_hectare : Nat := 100

-- The problem statement with the expected values
theorem problem1 : 3 * sq_m_to_sq_dm = 300 := by
  sorry

theorem problem2 : 2 * hectare_to_sq_m = 20000 := by
  sorry

theorem problem3 : 5000 / sq_cm_to_sq_dm_div = 50 := by
  sorry

theorem problem4 : 8 * sq_km_to_hectare = 800 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l807_80762


namespace NUMINAMATH_GPT_total_money_shared_l807_80732

theorem total_money_shared (k t : ℕ) (h1 : k = 1750) (h2 : t = 2 * k) : k + t = 5250 :=
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l807_80732


namespace NUMINAMATH_GPT_value_of_a_l807_80714

theorem value_of_a (a : ℤ) (h1 : 2 * a + 6 + (3 - a) = 0) : a = -9 :=
sorry

end NUMINAMATH_GPT_value_of_a_l807_80714


namespace NUMINAMATH_GPT_log_sum_l807_80753

-- Define the common logarithm function using Lean's natural logarithm with a change of base
noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum : log_base_10 5 + log_base_10 0.2 = 0 :=
by
  -- Placeholder for the proof to be completed
  sorry

end NUMINAMATH_GPT_log_sum_l807_80753


namespace NUMINAMATH_GPT_arithmetic_sequence_100th_term_l807_80735

-- Define the first term and the common difference
def first_term : ℕ := 3
def common_difference : ℕ := 7

-- Define the formula for the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem: The 100th term of the arithmetic sequence is 696.
theorem arithmetic_sequence_100th_term :
  nth_term first_term common_difference 100 = 696 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_100th_term_l807_80735


namespace NUMINAMATH_GPT_solve_x_l807_80739

theorem solve_x (x: ℝ) (h: -4 * x - 15 = 12 * x + 5) : x = -5 / 4 :=
sorry

end NUMINAMATH_GPT_solve_x_l807_80739


namespace NUMINAMATH_GPT_intersection_S_T_l807_80756

def S := {x : ℝ | (x - 2) * (x - 3) ≥ 0}
def T := {x : ℝ | x > 0}

theorem intersection_S_T :
  (S ∩ T) = (Set.Ioc 0 2 ∪ Set.Ici 3) :=
by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l807_80756


namespace NUMINAMATH_GPT_remainder_when_divided_by_5_l807_80764

theorem remainder_when_divided_by_5 : (1234 * 1987 * 2013 * 2021) % 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_5_l807_80764


namespace NUMINAMATH_GPT_solve_fraction_zero_l807_80786

theorem solve_fraction_zero (x : ℝ) (h : (x + 5) / (x - 2) = 0) : x = -5 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_zero_l807_80786


namespace NUMINAMATH_GPT_find_t_from_x_l807_80770

theorem find_t_from_x (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_t_from_x_l807_80770


namespace NUMINAMATH_GPT_inequality_solution_l807_80763

noncomputable def solve_inequality : Set ℝ :=
  {x | (x - 5) / ((x - 3)^2) < 0}

theorem inequality_solution :
  solve_inequality = {x | x < 3} ∪ {x | 3 < x ∧ x < 5} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l807_80763


namespace NUMINAMATH_GPT_number_of_teams_l807_80775

-- Given the conditions and the required proof problem
theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_teams_l807_80775


namespace NUMINAMATH_GPT_fish_tagged_initially_l807_80782

theorem fish_tagged_initially (N T : ℕ) (hN : N = 1500) 
  (h_ratio : 2 / 50 = (T:ℕ) / N) : T = 60 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_fish_tagged_initially_l807_80782


namespace NUMINAMATH_GPT_tetrahedron_faces_congruent_iff_face_angle_sum_straight_l807_80754

-- Defining the Tetrahedron and its properties
structure Tetrahedron (V : Type*) :=
(A B C D : V)
(face_angle_sum_at_vertex : V → Prop)
(congruent_faces : Prop)

-- Translating the problem into a Lean 4 theorem statement
theorem tetrahedron_faces_congruent_iff_face_angle_sum_straight (V : Type*) 
  (T : Tetrahedron V) :
  T.face_angle_sum_at_vertex T.A = T.face_angle_sum_at_vertex T.B ∧ 
  T.face_angle_sum_at_vertex T.B = T.face_angle_sum_at_vertex T.C ∧ 
  T.face_angle_sum_at_vertex T.C = T.face_angle_sum_at_vertex T.D ↔ T.congruent_faces :=
sorry


end NUMINAMATH_GPT_tetrahedron_faces_congruent_iff_face_angle_sum_straight_l807_80754


namespace NUMINAMATH_GPT_jack_further_down_l807_80737

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end NUMINAMATH_GPT_jack_further_down_l807_80737


namespace NUMINAMATH_GPT_ratio_of_areas_l807_80709

noncomputable def area (A B C D : ℝ) : ℝ := 0  -- Placeholder, exact area definition will require geometrical formalism.

variables (A B C D P Q R S : ℝ)

-- Define the conditions
variables (h1 : AB = BP) (h2 : BC = CQ) (h3 : CD = DR) (h4 : DA = AS)

-- Lean 4 statement for the proof problem
theorem ratio_of_areas : area A B C D / area P Q R S = 1/5 :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_l807_80709


namespace NUMINAMATH_GPT_apples_b_lighter_than_a_l807_80794

-- Definitions based on conditions
def total_weight : ℕ := 72
def weight_basket_a : ℕ := 42
def weight_basket_b : ℕ := total_weight - weight_basket_a

-- Theorem to prove the question equals the answer given the conditions
theorem apples_b_lighter_than_a : (weight_basket_a - weight_basket_b) = 12 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_apples_b_lighter_than_a_l807_80794


namespace NUMINAMATH_GPT_tom_books_total_l807_80790

theorem tom_books_total :
  (2 + 6 + 10 + 14 + 18) = 50 :=
by {
  -- Proof steps would go here.
  sorry
}

end NUMINAMATH_GPT_tom_books_total_l807_80790


namespace NUMINAMATH_GPT_Maggie_earnings_l807_80772

theorem Maggie_earnings :
  let family_commission := 7
  let neighbor_commission := 6
  let bonus_fixed := 10
  let bonus_threshold := 10
  let bonus_per_subscription := 1
  let monday_family := 4 + 1 
  let tuesday_neighbors := 2 + 2 * 2
  let wednesday_family := 3 + 1
  let total_family := monday_family + wednesday_family
  let total_neighbors := tuesday_neighbors
  let total_subscriptions := total_family + total_neighbors
  let bonus := if total_subscriptions > bonus_threshold then 
                 bonus_fixed + bonus_per_subscription * (total_subscriptions - bonus_threshold)
               else 0
  let total_earnings := total_family * family_commission + total_neighbors * neighbor_commission + bonus
  total_earnings = 114 := 
by {
  -- Placeholder for the proof. We assume this step will contain a verification of derived calculations.
  sorry
}

end NUMINAMATH_GPT_Maggie_earnings_l807_80772


namespace NUMINAMATH_GPT_simplify_expression_l807_80791

-- Define the statement we want to prove
theorem simplify_expression (s : ℕ) : (105 * s - 63 * s) = 42 * s :=
  by
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_simplify_expression_l807_80791


namespace NUMINAMATH_GPT_sixteen_powers_five_equals_four_power_ten_l807_80711

theorem sixteen_powers_five_equals_four_power_ten : 
  (16 * 16 * 16 * 16 * 16 = 4 ^ 10) :=
by
  sorry

end NUMINAMATH_GPT_sixteen_powers_five_equals_four_power_ten_l807_80711


namespace NUMINAMATH_GPT_remainder_when_1_stmt_l807_80751

-- Define the polynomial g(s)
def g (s : ℚ) : ℚ := s^15 + 1

-- Define the remainder theorem statement in the context of this problem
theorem remainder_when_1_stmt (s : ℚ) : g 1 = 2 :=
  sorry

end NUMINAMATH_GPT_remainder_when_1_stmt_l807_80751


namespace NUMINAMATH_GPT_quotient_of_5_divided_by_y_is_5_point_3_l807_80750

theorem quotient_of_5_divided_by_y_is_5_point_3 (y : ℝ) (h : 5 / y = 5.3) : y = 26.5 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_5_divided_by_y_is_5_point_3_l807_80750


namespace NUMINAMATH_GPT_subset_of_intervals_l807_80777

def A (x : ℝ) := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def is_subset_of (B A : ℝ → Prop) := ∀ x, B x → A x
def possible_values_m (m : ℝ) := m ≤ 3

theorem subset_of_intervals (m : ℝ) :
  is_subset_of (B m) A ↔ possible_values_m m := by
  sorry

end NUMINAMATH_GPT_subset_of_intervals_l807_80777


namespace NUMINAMATH_GPT_remainder_determined_l807_80788

theorem remainder_determined (p a b : ℤ) (h₀: Nat.Prime (Int.natAbs p)) (h₁ : ¬ (p ∣ a)) (h₂ : ¬ (p ∣ b)) :
  ∃ (r : ℤ), (r ≡ a [ZMOD p]) ∧ (r ≡ b [ZMOD p]) ∧ (r ≡ (a * b) [ZMOD p]) →
  (a ≡ r [ZMOD p]) := sorry

end NUMINAMATH_GPT_remainder_determined_l807_80788


namespace NUMINAMATH_GPT_Darla_electricity_bill_l807_80742

theorem Darla_electricity_bill :
  let tier1_rate := 4
  let tier2_rate := 3.5
  let tier3_rate := 3
  let tier1_limit := 300
  let tier2_limit := 500
  let late_fee1 := 150
  let late_fee2 := 200
  let late_fee3 := 250
  let consumption := 1200
  let cost_tier1 := tier1_limit * tier1_rate
  let cost_ttier2 := tier2_limit * tier2_rate
  let cost_tier3 := (consumption - (tier1_limit + tier2_limit)) * tier3_rate
  let total_cost := cost_tier1 + cost_tier2 + cost_tier3
  let late_fee := late_fee3
  let final_cost := total_cost + late_fee
  final_cost = 4400 :=
by
  sorry

end NUMINAMATH_GPT_Darla_electricity_bill_l807_80742


namespace NUMINAMATH_GPT_subtracted_number_divisible_by_5_l807_80796

theorem subtracted_number_divisible_by_5 : ∃ k : ℕ, 9671 - 1 = 5 * k :=
by
  sorry

end NUMINAMATH_GPT_subtracted_number_divisible_by_5_l807_80796


namespace NUMINAMATH_GPT_final_points_l807_80727

-- Definitions of the points in each round
def first_round_points : Int := 16
def second_round_points : Int := 33
def last_round_points : Int := -48

-- The theorem to prove Emily's final points
theorem final_points :
  first_round_points + second_round_points + last_round_points = 1 :=
by
  sorry

end NUMINAMATH_GPT_final_points_l807_80727


namespace NUMINAMATH_GPT_value_of_y_at_x_eq_1_l807_80749

noncomputable def quadractic_function (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem value_of_y_at_x_eq_1 (m : ℝ) (h1 : ∀ x : ℝ, x ≤ -2 → quadractic_function x m < quadractic_function (x + 1) m)
    (h2 : ∀ x : ℝ, x ≥ -2 → quadractic_function x m < quadractic_function (x + 1) m) :
    quadractic_function 1 16 = 25 :=
sorry

end NUMINAMATH_GPT_value_of_y_at_x_eq_1_l807_80749


namespace NUMINAMATH_GPT_rate_per_kg_for_mangoes_l807_80761

theorem rate_per_kg_for_mangoes (quantity_grapes : ℕ)
    (rate_grapes : ℕ)
    (quantity_mangoes : ℕ)
    (total_payment : ℕ)
    (rate_mangoes : ℕ) :
    quantity_grapes = 8 →
    rate_grapes = 70 →
    quantity_mangoes = 9 →
    total_payment = 1055 →
    8 * 70 + 9 * rate_mangoes = 1055 →
    rate_mangoes = 55 := by
  intros h1 h2 h3 h4 h5
  have h6 : 8 * 70 = 560 := by norm_num
  have h7 : 560 + 9 * rate_mangoes = 1055 := by rw [h5]
  have h8 : 1055 - 560 = 495 := by norm_num
  have h9 : 9 * rate_mangoes = 495 := by linarith
  have h10 : rate_mangoes = 55 := by linarith
  exact h10

end NUMINAMATH_GPT_rate_per_kg_for_mangoes_l807_80761


namespace NUMINAMATH_GPT_absolute_value_inequality_l807_80798

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_l807_80798


namespace NUMINAMATH_GPT_geometric_series_sum_l807_80781

theorem geometric_series_sum :
  let a := 2
  let r := -2
  let n := 10
  let Sn := (a : ℚ) * (r^n - 1) / (r - 1)
  Sn = 2050 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l807_80781


namespace NUMINAMATH_GPT_ski_boat_rental_cost_per_hour_l807_80768

-- Let the cost per hour to rent a ski boat be x dollars
variable (x : ℝ)

-- Conditions
def cost_sailboat : ℝ := 60
def duration : ℝ := 3 * 2 -- 3 hours a day for 2 days
def cost_ken : ℝ := cost_sailboat * 2 -- Ken's total cost
def additional_cost : ℝ := 120
def cost_aldrich : ℝ := cost_ken + additional_cost -- Aldrich's total cost

-- Statement to prove
theorem ski_boat_rental_cost_per_hour (h : (duration * x = cost_aldrich)) : x = 40 := by
  sorry

end NUMINAMATH_GPT_ski_boat_rental_cost_per_hour_l807_80768


namespace NUMINAMATH_GPT_number_of_cities_experienced_protests_l807_80758

variables (days_of_protest : ℕ) (arrests_per_day : ℕ) (days_pre_trial : ℕ) 
          (days_post_trial_in_weeks : ℕ) (combined_weeks_jail : ℕ)

def total_days_in_jail_per_person := days_pre_trial + (days_post_trial_in_weeks * 7) / 2

theorem number_of_cities_experienced_protests 
  (h1 : days_of_protest = 30) 
  (h2 : arrests_per_day = 10) 
  (h3 : days_pre_trial = 4) 
  (h4 : days_post_trial_in_weeks = 2) 
  (h5 : combined_weeks_jail = 9900) : 
  (combined_weeks_jail * 7) / total_days_in_jail_per_person 
  = 21 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cities_experienced_protests_l807_80758


namespace NUMINAMATH_GPT_john_age_multiple_of_james_age_l807_80719

-- Define variables for the problem conditions
def john_current_age : ℕ := 39
def john_age_3_years_ago : ℕ := john_current_age - 3

def james_brother_age : ℕ := 16
def james_brother_older : ℕ := 4

def james_current_age : ℕ := james_brother_age - james_brother_older
def james_age_in_6_years : ℕ := james_current_age + 6

-- The goal is to prove the multiple relationship
theorem john_age_multiple_of_james_age :
  john_age_3_years_ago = 2 * james_age_in_6_years :=
by {
  -- Skip the proof
  sorry
}

end NUMINAMATH_GPT_john_age_multiple_of_james_age_l807_80719


namespace NUMINAMATH_GPT_range_of_f_l807_80702

noncomputable def f (x : ℝ) : ℝ := 1 / x - 4 / Real.sqrt x + 3

theorem range_of_f : ∀ y, (∃ x, (1/16 : ℝ) ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l807_80702


namespace NUMINAMATH_GPT_student_good_probability_l807_80725

-- Defining the conditions as given in the problem
def P_A1 := 0.25          -- Probability of selecting a student from School A
def P_A2 := 0.4           -- Probability of selecting a student from School B
def P_A3 := 0.35          -- Probability of selecting a student from School C

def P_B_given_A1 := 0.3   -- Probability that a student's level is good given they are from School A
def P_B_given_A2 := 0.6   -- Probability that a student's level is good given they are from School B
def P_B_given_A3 := 0.5   -- Probability that a student's level is good given they are from School C

-- Main theorem statement
theorem student_good_probability : 
  P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 0.49 := 
by sorry

end NUMINAMATH_GPT_student_good_probability_l807_80725


namespace NUMINAMATH_GPT_true_supporters_of_rostov_l807_80783

theorem true_supporters_of_rostov
  (knights_liars_fraction : ℕ → ℕ)
  (rostov_support_yes : ℕ)
  (zenit_support_yes : ℕ)
  (lokomotiv_support_yes : ℕ)
  (cska_support_yes : ℕ)
  (h1 : knights_liars_fraction 100 = 10)
  (h2 : rostov_support_yes = 40)
  (h3 : zenit_support_yes = 30)
  (h4 : lokomotiv_support_yes = 50)
  (h5 : cska_support_yes = 0):
  rostov_support_yes - knights_liars_fraction 100 = 30 := 
sorry

end NUMINAMATH_GPT_true_supporters_of_rostov_l807_80783


namespace NUMINAMATH_GPT_compare_logs_l807_80721

noncomputable def e := Real.exp 1
noncomputable def log_base_10 (x : Real) := Real.log x / Real.log 10

theorem compare_logs (x : Real) (hx : e < x ∧ x < 10) :
  let a := Real.log (Real.log x)
  let b := log_base_10 (log_base_10 x)
  let c := Real.log (log_base_10 x)
  let d := log_base_10 (Real.log x)
  c < b ∧ b < d ∧ d < a := 
sorry

end NUMINAMATH_GPT_compare_logs_l807_80721


namespace NUMINAMATH_GPT_relationship_between_T_and_S_l807_80717

variable (a b : ℝ)

def T : ℝ := a + 2 * b
def S : ℝ := a + b^2 + 1

theorem relationship_between_T_and_S : T a b ≤ S a b := by
  sorry

end NUMINAMATH_GPT_relationship_between_T_and_S_l807_80717


namespace NUMINAMATH_GPT_min_value_of_expression_l807_80713

theorem min_value_of_expression (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : a * b * c = 27) :
  a^2 + 2*a*b + b^2 + 3*c^2 ≥ 324 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l807_80713


namespace NUMINAMATH_GPT_arithmetic_geometric_proof_l807_80707

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_proof
  (a : ℕ → ℤ) (b : ℕ → ℤ) (d r : ℤ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence b r)
  (h_cond1 : 3 * a 1 - a 8 * a 8 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10):
  b 3 * b 17 = 36 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_proof_l807_80707


namespace NUMINAMATH_GPT_find_x2017_l807_80774

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define that f is increasing
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y
  
-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + n * d

-- Main theorem
theorem find_x2017
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (Hodd : is_odd_function f)
  (Hinc : is_increasing_function f)
  (Hseq : ∀ n, x (n + 1) = x n + 2)
  (H7_8 : f (x 7) + f (x 8) = 0) :
  x 2017 = 4019 := 
sorry

end NUMINAMATH_GPT_find_x2017_l807_80774


namespace NUMINAMATH_GPT_sum_of_three_numbers_l807_80766

theorem sum_of_three_numbers (a b c : ℕ) (h1 : b = 10)
                            (h2 : (a + b + c) / 3 = a + 15)
                            (h3 : (a + b + c) / 3 = c - 25) :
                            a + b + c = 60 :=
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l807_80766


namespace NUMINAMATH_GPT_smallest_possible_N_l807_80720

theorem smallest_possible_N (l m n : ℕ) (h_visible : (l - 1) * (m - 1) * (n - 1) = 252) : l * m * n = 392 :=
sorry

end NUMINAMATH_GPT_smallest_possible_N_l807_80720


namespace NUMINAMATH_GPT_largest_prime_value_of_quadratic_expression_l807_80792

theorem largest_prime_value_of_quadratic_expression : 
  ∃ n : ℕ, n > 0 ∧ Prime (n^2 - 12 * n + 27) ∧ ∀ m : ℕ, m > 0 → Prime (m^2 - 12 * m + 27) → (n^2 - 12 * n + 27) ≥ (m^2 - 12 * m + 27) := 
by
  sorry


end NUMINAMATH_GPT_largest_prime_value_of_quadratic_expression_l807_80792


namespace NUMINAMATH_GPT_white_roses_needed_l807_80734

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end NUMINAMATH_GPT_white_roses_needed_l807_80734


namespace NUMINAMATH_GPT_no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l807_80740

-- Part (a)
theorem no_integers_a_b_existence (a b : ℤ) :
  ¬(a^2 - 3 * (b^2) = 8) :=
sorry

-- Part (b)
theorem no_positive_integers_a_b_c_existence (a b c : ℕ) (ha: a > 0) (hb: b > 0) (hc: c > 0 ) :
  ¬(a^2 + b^2 = 3 * (c^2)) :=
sorry

end NUMINAMATH_GPT_no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l807_80740


namespace NUMINAMATH_GPT_find_natural_numbers_l807_80705

theorem find_natural_numbers (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 3^5) : 
  (x = 6 ∧ y = 3) := 
sorry

end NUMINAMATH_GPT_find_natural_numbers_l807_80705


namespace NUMINAMATH_GPT_product_of_a_and_b_is_zero_l807_80744

theorem product_of_a_and_b_is_zero
  (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)
  (h2 : b < 10)
  (h3 : a * (b + 10) = 190) :
  a * b = 0 :=
sorry

end NUMINAMATH_GPT_product_of_a_and_b_is_zero_l807_80744


namespace NUMINAMATH_GPT_people_receiving_roses_l807_80746

-- Defining the conditions.
def initial_roses : Nat := 40
def stolen_roses : Nat := 4
def roses_per_person : Nat := 4

-- Stating the theorem.
theorem people_receiving_roses : 
  (initial_roses - stolen_roses) / roses_per_person = 9 :=
by sorry

end NUMINAMATH_GPT_people_receiving_roses_l807_80746


namespace NUMINAMATH_GPT_pencils_given_away_l807_80759

-- Define the basic values and conditions
def initial_pencils : ℕ := 39
def bought_pencils : ℕ := 22
def final_pencils : ℕ := 43

-- Let x be the number of pencils Brian gave away
variable (x : ℕ)

-- State the theorem we need to prove
theorem pencils_given_away : (initial_pencils - x) + bought_pencils = final_pencils → x = 18 := by
  sorry

end NUMINAMATH_GPT_pencils_given_away_l807_80759


namespace NUMINAMATH_GPT_unique_integer_solution_l807_80716

theorem unique_integer_solution (x y z : ℤ) (h : 2 * x^2 + 3 * y^2 = z^2) : x = 0 ∧ y = 0 ∧ z = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_integer_solution_l807_80716


namespace NUMINAMATH_GPT_erased_number_l807_80723

theorem erased_number (n i : ℕ) (h : (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17) : i = 7 :=
sorry

end NUMINAMATH_GPT_erased_number_l807_80723


namespace NUMINAMATH_GPT_sin_theta_value_l807_80779

open Real

noncomputable def sin_theta_sol (theta : ℝ) : ℝ :=
  (-5 + Real.sqrt 41) / 4

theorem sin_theta_value (theta : ℝ) (h1 : 5 * tan theta = 2 * cos theta) (h2 : 0 < theta) (h3 : theta < π) :
  sin theta = sin_theta_sol theta :=
by
  sorry

end NUMINAMATH_GPT_sin_theta_value_l807_80779


namespace NUMINAMATH_GPT_bike_growth_equation_l807_80765

-- Declare the parameters
variables (b1 b3 : ℕ) (x : ℝ)
-- Define the conditions
def condition1 : b1 = 1000 := sorry
def condition2 : b3 = b1 + 440 := sorry

-- Define the proposition to be proved
theorem bike_growth_equation (cond1 : b1 = 1000) (cond2 : b3 = b1 + 440) :
  b1 * (1 + x)^2 = b3 :=
sorry

end NUMINAMATH_GPT_bike_growth_equation_l807_80765


namespace NUMINAMATH_GPT_problem_inequality_l807_80722

theorem problem_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l807_80722


namespace NUMINAMATH_GPT_total_books_proof_l807_80730

noncomputable def economics_books (T : ℝ) := (1/4) * T + 10
noncomputable def rest_books (T : ℝ) := T - economics_books T
noncomputable def social_studies_books (T : ℝ) := (3/5) * rest_books T - 5
noncomputable def other_books := 13
noncomputable def science_books := 12
noncomputable def total_books_equation (T : ℝ) :=
  T = economics_books T + social_studies_books T + science_books + other_books

theorem total_books_proof : ∃ T : ℝ, total_books_equation T ∧ T = 80 := by
  sorry

end NUMINAMATH_GPT_total_books_proof_l807_80730


namespace NUMINAMATH_GPT_probability_product_divisible_by_4_gt_half_l807_80778

theorem probability_product_divisible_by_4_gt_half :
  let n := 2023
  let even_count := n / 2
  let four_div_count := n / 4
  let select_five := 5
  (true) ∧ (even_count = 1012) ∧ (four_div_count = 505)
  → 0.5 < (1 - ((2023 - even_count) / 2023) * ((2022 - (even_count - 1)) / 2022) * ((2021 - (even_count - 2)) / 2021) * ((2020 - (even_count - 3)) / 2020) * ((2019 - (even_count - 4)) / 2019)) :=
by
  sorry

end NUMINAMATH_GPT_probability_product_divisible_by_4_gt_half_l807_80778


namespace NUMINAMATH_GPT_inscribed_circle_radius_l807_80706

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (∀ (R : ℝ), R = 12 →
      (∀ (d : ℝ), d = 12 → r = 3)) :=
by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l807_80706


namespace NUMINAMATH_GPT_percentage_of_sum_is_14_l807_80703

-- Define variables x, y as real numbers
variables (x y P : ℝ)

-- Define condition 1: y is 17.647058823529413% of x
def y_is_percentage_of_x : Prop := y = 0.17647058823529413 * x

-- Define condition 2: 20% of (x - y) is equal to P% of (x + y)
def percentage_equation : Prop := 0.20 * (x - y) = (P / 100) * (x + y)

-- Define the statement to be proved: P is 14
theorem percentage_of_sum_is_14 (h1 : y_is_percentage_of_x x y) (h2 : percentage_equation x y P) : 
  P = 14 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_sum_is_14_l807_80703


namespace NUMINAMATH_GPT_find_number_l807_80724

theorem find_number (n x : ℤ)
  (h1 : (2 * x + 1) = (x - 7)) 
  (h2 : ∃ x : ℤ, n = (2 * x + 1) ^ 2) : 
  n = 25 := 
sorry

end NUMINAMATH_GPT_find_number_l807_80724


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l807_80708

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l807_80708


namespace NUMINAMATH_GPT_john_ate_12_ounces_of_steak_l807_80741

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end NUMINAMATH_GPT_john_ate_12_ounces_of_steak_l807_80741


namespace NUMINAMATH_GPT_inv_100_mod_101_l807_80767

theorem inv_100_mod_101 : (100 : ℤ) * (100 : ℤ) % 101 = 1 := by
  sorry

end NUMINAMATH_GPT_inv_100_mod_101_l807_80767
