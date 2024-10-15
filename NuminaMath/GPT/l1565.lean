import Mathlib

namespace NUMINAMATH_GPT_age_intervals_l1565_156503

theorem age_intervals (A1 A2 A3 A4 A5 : ℝ) (x : ℝ) (h1 : A1 = 7)
  (h2 : A2 = A1 + x) (h3 : A3 = A1 + 2 * x) (h4 : A4 = A1 + 3 * x) (h5 : A5 = A1 + 4 * x)
  (sum_ages : A1 + A2 + A3 + A4 + A5 = 65) :
  x = 3.7 :=
by
  -- Sketch a proof or leave 'sorry' for completeness
  sorry

end NUMINAMATH_GPT_age_intervals_l1565_156503


namespace NUMINAMATH_GPT_divisibility_by_7_l1565_156557

theorem divisibility_by_7 (m a : ℤ) (h : 0 ≤ a ∧ a ≤ 9) (B : ℤ) (hB : B = m - 2 * a) (h7 : B % 7 = 0) : (10 * m + a) % 7 = 0 := 
sorry

end NUMINAMATH_GPT_divisibility_by_7_l1565_156557


namespace NUMINAMATH_GPT_compare_a_b_l1565_156523

def a := 1 / 3 + 1 / 4
def b := 1 / 5 + 1 / 6 + 1 / 7

theorem compare_a_b : a > b := 
  sorry

end NUMINAMATH_GPT_compare_a_b_l1565_156523


namespace NUMINAMATH_GPT_rectangle_length_difference_l1565_156546

variable (s l w : ℝ)

-- Conditions
def condition1 : Prop := 2 * (l + w) = 4 * s + 4
def condition2 : Prop := w = s - 2

-- Theorem to prove
theorem rectangle_length_difference
  (s l w : ℝ)
  (h1 : condition1 s l w)
  (h2 : condition2 s w) : l = s + 4 :=
by
sorry

end NUMINAMATH_GPT_rectangle_length_difference_l1565_156546


namespace NUMINAMATH_GPT_sector_angle_l1565_156583

theorem sector_angle (r : ℝ) (S : ℝ) (α : ℝ) (h₁ : r = 10) (h₂ : S = 50 * π / 3) (h₃ : S = 1 / 2 * r^2 * α) : 
  α = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_angle_l1565_156583


namespace NUMINAMATH_GPT_Caden_total_money_l1565_156534

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end NUMINAMATH_GPT_Caden_total_money_l1565_156534


namespace NUMINAMATH_GPT_arithmetic_progression_11th_term_l1565_156504

theorem arithmetic_progression_11th_term:
  ∀ (a d : ℝ), (15 / 2) * (2 * a + 14 * d) = 56.25 → a + 6 * d = 3.25 → a + 10 * d = 5.25 :=
by
  intros a d h_sum h_7th
  sorry

end NUMINAMATH_GPT_arithmetic_progression_11th_term_l1565_156504


namespace NUMINAMATH_GPT_unique_solution_l1565_156513

def is_valid_func (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 2001 ∨ f (f n) + f n = 2 * n + 2002

theorem unique_solution (f : ℕ → ℕ) (hf : is_valid_func f) :
  ∀ n, f n = n + 667 :=
sorry

end NUMINAMATH_GPT_unique_solution_l1565_156513


namespace NUMINAMATH_GPT_additional_wolves_in_pack_l1565_156584

-- Define the conditions
def wolves_out_hunting : ℕ := 4
def meat_per_wolf_per_day : ℕ := 8
def hunting_days : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat per wolf for hunting days
def meat_per_wolf_total : ℕ := meat_per_wolf_per_day * hunting_days

-- Calculate wolves fed per deer
def wolves_fed_per_deer : ℕ := meat_per_deer / meat_per_wolf_total

-- Calculate total deer killed by wolves out hunting
def total_deers_killed : ℕ := wolves_out_hunting

-- Calculate total meat provided by hunting wolves
def total_meat_provided : ℕ := total_deers_killed * meat_per_deer

-- Calculate number of wolves fed by total meat provided
def total_wolves_fed : ℕ := total_meat_provided / meat_per_wolf_total

-- Define the main theorem to prove the answer
theorem additional_wolves_in_pack (total_wolves_fed wolves_out_hunting : ℕ) : 
  total_wolves_fed - wolves_out_hunting = 16 :=
by
  sorry

end NUMINAMATH_GPT_additional_wolves_in_pack_l1565_156584


namespace NUMINAMATH_GPT_product_of_equal_numbers_l1565_156587

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end NUMINAMATH_GPT_product_of_equal_numbers_l1565_156587


namespace NUMINAMATH_GPT_p_div_q_is_12_l1565_156595

-- Definition of binomials and factorials required for the proof
open Nat

/-- Define the number of ways to distribute balls for configuration A -/
def config_A : ℕ :=
  @choose 5 1 * @choose 4 2 * @choose 2 1 * (factorial 20) / (factorial 2 * factorial 4 * factorial 4 * factorial 3 * factorial 7)

/-- Define the number of ways to distribute balls for configuration B -/
def config_B : ℕ :=
  @choose 5 2 * @choose 3 3 * (factorial 20) / (factorial 3 * factorial 3 * factorial 4 * factorial 4 * factorial 4)

/-- The ratio of probabilities p/q for the given distributions of balls into bins is 12 -/
theorem p_div_q_is_12 : config_A / config_B = 12 :=
by
  sorry

end NUMINAMATH_GPT_p_div_q_is_12_l1565_156595


namespace NUMINAMATH_GPT_square_area_less_than_circle_area_l1565_156510

theorem square_area_less_than_circle_area (a : ℝ) (ha : 0 < a) :
    let S1 := (a / 4) ^ 2
    let r := a / (2 * Real.pi)
    let S2 := Real.pi * r^2
    (S1 < S2) := by
sorry

end NUMINAMATH_GPT_square_area_less_than_circle_area_l1565_156510


namespace NUMINAMATH_GPT_number_of_sodas_in_pack_l1565_156558

/-- Billy has twice as many brothers as sisters -/
def twice_as_many_brothers_as_sisters (brothers sisters : ℕ) : Prop :=
  brothers = 2 * sisters

/-- Billy has 2 sisters -/
def billy_has_2_sisters : Prop :=
  ∃ sisters : ℕ, sisters = 2

/-- Billy can give 2 sodas to each of his siblings if he wants to give out the entire pack while giving each sibling the same number of sodas -/
def divide_sodas_evenly (total_sodas siblings sodas_per_sibling : ℕ) : Prop :=
  total_sodas = siblings * sodas_per_sibling

/-- Determine the total number of sodas in the pack given the conditions -/
theorem number_of_sodas_in_pack : 
  ∃ (sisters brothers total_sodas : ℕ), 
    (twice_as_many_brothers_as_sisters brothers sisters) ∧ 
    (billy_has_2_sisters) ∧ 
    (divide_sodas_evenly total_sodas (sisters + brothers + 1) 2) ∧
    (total_sodas = 12) :=
by
  sorry

end NUMINAMATH_GPT_number_of_sodas_in_pack_l1565_156558


namespace NUMINAMATH_GPT_weeks_per_month_l1565_156586

-- Define the given conditions
def num_employees_initial : Nat := 500
def additional_employees : Nat := 200
def hourly_wage : Nat := 12
def daily_work_hours : Nat := 10
def weekly_work_days : Nat := 5
def total_monthly_pay : Nat := 1680000

-- Calculate the total number of employees after hiring
def total_employees : Nat := num_employees_initial + additional_employees

-- Calculate the pay rates
def daily_pay_per_employee : Nat := hourly_wage * daily_work_hours
def weekly_pay_per_employee : Nat := daily_pay_per_employee * weekly_work_days

-- Calculate the total weekly pay for all employees
def total_weekly_pay : Nat := weekly_pay_per_employee * total_employees

-- Define the statement to be proved
theorem weeks_per_month
  (h1 : total_employees = num_employees_initial + additional_employees)
  (h2 : daily_pay_per_employee = hourly_wage * daily_work_hours)
  (h3 : weekly_pay_per_employee = daily_pay_per_employee * weekly_work_days)
  (h4 : total_weekly_pay = weekly_pay_per_employee * total_employees)
  (h5 : total_monthly_pay = 1680000) :
  total_monthly_pay / total_weekly_pay = 4 :=
by sorry

end NUMINAMATH_GPT_weeks_per_month_l1565_156586


namespace NUMINAMATH_GPT_olivia_card_value_l1565_156511

theorem olivia_card_value (x : ℝ) (hx1 : 90 < x ∧ x < 180)
  (h_sin_pos : Real.sin x > 0) (h_cos_neg : Real.cos x < 0) (h_tan_neg : Real.tan x < 0)
  (h_olivia_distinguish : ∀ (a b c : ℝ), 
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (∃! a, a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x)) :
  Real.sin 135 = Real.cos 45 := 
sorry

end NUMINAMATH_GPT_olivia_card_value_l1565_156511


namespace NUMINAMATH_GPT_max_value_M_l1565_156529

theorem max_value_M : 
  ∃ t : ℝ, (t = (3 / (4 ^ (1 / 3)))) ∧ 
    (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 
      a^3 + b^3 + c^3 - 3 * a * b * c ≥ t * (a * b^2 + b * c^2 + c * a^2 - 3 * a * b * c)) :=
sorry

end NUMINAMATH_GPT_max_value_M_l1565_156529


namespace NUMINAMATH_GPT_traffic_safety_team_eq_twice_fire_l1565_156556

-- Define initial members in the teams
def t0 : ℕ := 8
def f0 : ℕ := 7

-- Define the main theorem
theorem traffic_safety_team_eq_twice_fire (x : ℕ) : t0 + x = 2 * (f0 - x) :=
by sorry

end NUMINAMATH_GPT_traffic_safety_team_eq_twice_fire_l1565_156556


namespace NUMINAMATH_GPT_perpendicular_lines_l1565_156527

theorem perpendicular_lines (a : ℝ) : 
  (3 * y + x + 4 = 0) → 
  (4 * y + a * x + 5 = 0) → 
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → - (1 / 3 : ℝ) * - (a / 4 : ℝ) = -1) → 
  a = -12 := 
by
  intros h1 h2 h_perpendicularity
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1565_156527


namespace NUMINAMATH_GPT_sum_not_equals_any_l1565_156594

-- Define the nine special natural numbers a1 to a9
def a1 (k : ℕ) : ℕ := (10^k - 1) / 9
def a2 (m : ℕ) : ℕ := 2 * (10^m - 1) / 9
def a3 (p : ℕ) : ℕ := 3 * (10^p - 1) / 9
def a4 (q : ℕ) : ℕ := 4 * (10^q - 1) / 9
def a5 (r : ℕ) : ℕ := 5 * (10^r - 1) / 9
def a6 (s : ℕ) : ℕ := 6 * (10^s - 1) / 9
def a7 (t : ℕ) : ℕ := 7 * (10^t - 1) / 9
def a8 (u : ℕ) : ℕ := 8 * (10^u - 1) / 9
def a9 (v : ℕ) : ℕ := 9 * (10^v - 1) / 9

-- Statement of the problem
theorem sum_not_equals_any (k m p q r s t u v : ℕ) :
  ¬ (a1 k = a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a2 m = a1 k + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a3 p = a1 k + a2 m + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a4 q = a1 k + a2 m + a3 p + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a5 r = a1 k + a2 m + a3 p + a4 q + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a6 s = a1 k + a2 m + a3 p + a4 q + a5 r + a7 t + a8 u + a9 v) ∧
  ¬ (a7 t = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a8 u + a9 v) ∧
  ¬ (a8 u = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a9 v) ∧
  ¬ (a9 v = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u) :=
  sorry

end NUMINAMATH_GPT_sum_not_equals_any_l1565_156594


namespace NUMINAMATH_GPT_not_possible_to_color_plane_l1565_156549

theorem not_possible_to_color_plane :
  ¬ ∃ (color : ℕ → ℕ × ℕ → ℕ) (c : ℕ), 
    (c = 2016) ∧
    (∀ (A B C : ℕ × ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
                        (color c A = color c B) ∨ (color c B = color c C) ∨ (color c C = color c A)) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_to_color_plane_l1565_156549


namespace NUMINAMATH_GPT_large_integer_value_l1565_156517

theorem large_integer_value :
  (2 + 3) * (2^2 + 3^2) * (2^4 - 3^4) * (2^8 + 3^8) * (2^16 - 3^16) * (2^32 + 3^32) * (2^64 - 3^64)
  > 0 := 
by
  sorry

end NUMINAMATH_GPT_large_integer_value_l1565_156517


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l1565_156553

-- Define the conditions
def area (l w : ℝ) : Prop := l * w = 180
def length_three_times_width (l w : ℝ) : Prop := l = 3 * w

-- Define the problem
theorem perimeter_of_rectangle (l w : ℝ) (h₁ : area l w) (h₂ : length_three_times_width l w) : 
  2 * (l + w) = 16 * Real.sqrt 15 := 
sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l1565_156553


namespace NUMINAMATH_GPT_total_pieces_of_clothing_l1565_156533

def number_of_pieces_per_drawer : ℕ := 2
def number_of_drawers : ℕ := 4

theorem total_pieces_of_clothing : 
  (number_of_pieces_per_drawer * number_of_drawers = 8) :=
by sorry

end NUMINAMATH_GPT_total_pieces_of_clothing_l1565_156533


namespace NUMINAMATH_GPT_line_passes_through_3_1_l1565_156554

open Classical

noncomputable def line_passes_through_fixed_point (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_passes_through_3_1 (m : ℝ) :
  line_passes_through_fixed_point m 3 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_3_1_l1565_156554


namespace NUMINAMATH_GPT_total_blocks_l1565_156535

theorem total_blocks (red_blocks yellow_blocks blue_blocks : ℕ) 
  (h1 : red_blocks = 18) 
  (h2 : yellow_blocks = red_blocks + 7) 
  (h3 : blue_blocks = red_blocks + 14) : 
  red_blocks + yellow_blocks + blue_blocks = 75 := 
by
  sorry

end NUMINAMATH_GPT_total_blocks_l1565_156535


namespace NUMINAMATH_GPT_race_order_l1565_156531

inductive Position where
| First | Second | Third | Fourth | Fifth
deriving DecidableEq, Repr

structure Statements where
  amy1 : Position → Prop
  amy2 : Position → Prop
  bruce1 : Position → Prop
  bruce2 : Position → Prop
  chris1 : Position → Prop
  chris2 : Position → Prop
  donna1 : Position → Prop
  donna2 : Position → Prop
  eve1 : Position → Prop
  eve2 : Position → Prop

def trueStatements : Statements := {
  amy1 := fun p => p = Position.Second,
  amy2 := fun p => p = Position.Third,
  bruce1 := fun p => p = Position.Second,
  bruce2 := fun p => p = Position.Fourth,
  chris1 := fun p => p = Position.First,
  chris2 := fun p => p = Position.Second,
  donna1 := fun p => p = Position.Third,
  donna2 := fun p => p = Position.Fifth,
  eve1 := fun p => p = Position.Fourth,
  eve2 := fun p => p = Position.First,
}

theorem race_order (f : Statements) :
  f.amy1 Position.Second ∧ f.amy2 Position.Third ∧
  f.bruce1 Position.First ∧ f.bruce2 Position.Fourth ∧
  f.chris1 Position.Fifth ∧ f.chris2 Position.Second ∧
  f.donna1 Position.Fourth ∧ f.donna2 Position.Fifth ∧
  f.eve1 Position.Fourth ∧ f.eve2 Position.First :=
by
  sorry

end NUMINAMATH_GPT_race_order_l1565_156531


namespace NUMINAMATH_GPT_arccos_of_half_eq_pi_over_three_l1565_156577

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_arccos_of_half_eq_pi_over_three_l1565_156577


namespace NUMINAMATH_GPT_symmetric_abs_necessary_not_sufficient_l1565_156515

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def y_axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem symmetric_abs_necessary_not_sufficient (f : ℝ → ℝ) :
  is_odd_function f → y_axis_symmetric f := sorry

end NUMINAMATH_GPT_symmetric_abs_necessary_not_sufficient_l1565_156515


namespace NUMINAMATH_GPT_work_completed_in_5_days_l1565_156528

-- Define the rates of work for A, B, and C
def rateA : ℚ := 1 / 15
def rateB : ℚ := 1 / 14
def rateC : ℚ := 1 / 16

-- Summing their rates to get the combined rate
def combined_rate : ℚ := rateA + rateB + rateC

-- This is the statement we need to prove, i.e., the time required for A, B, and C to finish the work together is 5 days.
theorem work_completed_in_5_days (hA : rateA = 1 / 15) (hB : rateB = 1 / 14) (hC : rateC = 1 / 16) :
  (1 / combined_rate) = 5 :=
by
  sorry

end NUMINAMATH_GPT_work_completed_in_5_days_l1565_156528


namespace NUMINAMATH_GPT_window_width_l1565_156526

theorem window_width (h_pane_height : ℕ) (h_to_w_ratio_num : ℕ) (h_to_w_ratio_den : ℕ) (gaps : ℕ) 
(border : ℕ) (columns : ℕ) 
(panes_per_row : ℕ) (pane_height : ℕ) 
(heights_equal : h_pane_height = pane_height)
(ratio : h_to_w_ratio_num * pane_height = h_to_w_ratio_den * panes_per_row)
: columns * (h_to_w_ratio_den * pane_height / h_to_w_ratio_num) + 
  gaps + 2 * border = 57 := sorry

end NUMINAMATH_GPT_window_width_l1565_156526


namespace NUMINAMATH_GPT_intersection_empty_condition_l1565_156547

-- Define the sets M and N under the given conditions
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The theorem that we need to prove based on the problem statement
theorem intersection_empty_condition (b : ℝ) :
  (∀ m : ℝ, M ∩ N m b = ∅) ↔ (b^2 > 6 * m^2 + 2) := sorry

end NUMINAMATH_GPT_intersection_empty_condition_l1565_156547


namespace NUMINAMATH_GPT_valid_parameterizations_l1565_156585

open Real

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 7

def valid_parametrization (p d : ℝ × ℝ) : Prop :=
  lies_on_line p ∧ is_scalar_multiple d (1, 2)

theorem valid_parameterizations :
  valid_parametrization (4, 1) (-2, -4) ∧ 
  ¬ valid_parametrization (12, 17) (5, 10) ∧ 
  valid_parametrization (3.5, 0) (1, 2) ∧ 
  valid_parametrization (-2, -11) (0.5, 1) ∧ 
  valid_parametrization (0, -7) (10, 20) :=
by {
  sorry
}

end NUMINAMATH_GPT_valid_parameterizations_l1565_156585


namespace NUMINAMATH_GPT_max_b_plus_c_triangle_l1565_156550

theorem max_b_plus_c_triangle (a b c : ℝ) (A : ℝ) 
  (h₁ : a = 4) (h₂ : A = Real.pi / 3) (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b + c ≤ 8 :=
by
  -- sorry is added to skip the proof for now.
  sorry

end NUMINAMATH_GPT_max_b_plus_c_triangle_l1565_156550


namespace NUMINAMATH_GPT_sequence_term_l1565_156588

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * n - 2

def S_n (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem sequence_term (n : ℕ) : a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_term_l1565_156588


namespace NUMINAMATH_GPT_matrix_pow_sub_l1565_156593

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end NUMINAMATH_GPT_matrix_pow_sub_l1565_156593


namespace NUMINAMATH_GPT_motorcyclist_average_speed_BC_l1565_156589

theorem motorcyclist_average_speed_BC :
  ∀ (d_AB : ℝ) (theta : ℝ) (d_BC_half_d_AB : ℝ) (avg_speed_trip : ℝ)
    (time_ratio_AB_BC : ℝ) (total_speed : ℝ) (t_AB : ℝ) (t_BC : ℝ),
    d_AB = 120 →
    theta = 10 →
    d_BC_half_d_AB = 1 / 2 →
    avg_speed_trip = 30 →
    time_ratio_AB_BC = 3 →
    t_AB = 4.5 →
    t_BC = 1.5 →
    t_AB = time_ratio_AB_BC * t_BC →
    avg_speed_trip = total_speed →
    total_speed = (d_AB + (d_AB * d_BC_half_d_AB)) / (t_AB + t_BC) →
    t_AB / 3 = t_BC →
    ((d_AB * d_BC_half_d_AB) / t_BC = 40) :=
by
  intros d_AB theta d_BC_half_d_AB avg_speed_trip time_ratio_AB_BC total_speed
        t_AB t_BC h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_motorcyclist_average_speed_BC_l1565_156589


namespace NUMINAMATH_GPT_initial_green_marbles_l1565_156570

theorem initial_green_marbles (m g' : ℕ) (h_m : m = 23) (h_g' : g' = 9) : (g' + m = 32) :=
by
  subst h_m
  subst h_g'
  rfl

end NUMINAMATH_GPT_initial_green_marbles_l1565_156570


namespace NUMINAMATH_GPT_number_of_blue_butterflies_l1565_156559

theorem number_of_blue_butterflies 
  (total_butterflies : ℕ)
  (B Y : ℕ)
  (H1 : total_butterflies = 11)
  (H2 : B = 2 * Y)
  (H3 : total_butterflies = B + Y + 5) : B = 4 := 
sorry

end NUMINAMATH_GPT_number_of_blue_butterflies_l1565_156559


namespace NUMINAMATH_GPT_A_inter_complement_B_eq_l1565_156548

-- Define set A
def set_A : Set ℝ := {x | -3 < x ∧ x < 6}

-- Define set B
def set_B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of set B in the real numbers
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- Define the intersection of set A with the complement of set B
def A_inter_complement_B : Set ℝ := set_A ∩ complement_B

-- Stating the theorem to prove
theorem A_inter_complement_B_eq : A_inter_complement_B = {x | -3 < x ∧ x ≤ 2} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_A_inter_complement_B_eq_l1565_156548


namespace NUMINAMATH_GPT_algebraic_expression_value_l1565_156520

-- Define the equation and its roots.
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 = 0

def is_root (x : ℝ) : Prop := quadratic_eq x

-- The main theorem.
theorem algebraic_expression_value (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) :
  (x1 + x2) / (1 + x1 * x2) = 1 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1565_156520


namespace NUMINAMATH_GPT_even_function_a_value_l1565_156565

def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_a_value (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 := by
  sorry

end NUMINAMATH_GPT_even_function_a_value_l1565_156565


namespace NUMINAMATH_GPT_increase_in_area_is_44_percent_l1565_156562

-- Let's define the conditions first
variables {r : ℝ} -- radius of the medium pizza
noncomputable def radius_large (r : ℝ) := 1.2 * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Now we state the Lean theorem that expresses the problem
theorem increase_in_area_is_44_percent (r : ℝ) : 
  (area (radius_large r) - area r) / area r * 100 = 44 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_area_is_44_percent_l1565_156562


namespace NUMINAMATH_GPT_complex_number_z_l1565_156566

theorem complex_number_z (z : ℂ) (h : (3 + 1 * I) * z = 4 - 2 * I) : z = 1 - I :=
by
  sorry

end NUMINAMATH_GPT_complex_number_z_l1565_156566


namespace NUMINAMATH_GPT_single_digit_solution_l1565_156537

theorem single_digit_solution :
  ∃ A : ℕ, A < 10 ∧ A^3 = 210 + A ∧ A = 6 :=
by
  existsi 6
  sorry

end NUMINAMATH_GPT_single_digit_solution_l1565_156537


namespace NUMINAMATH_GPT_pipe_A_fill_time_l1565_156512

theorem pipe_A_fill_time (t : ℝ) (h1 : t > 0) (h2 : ∃ tA tB, tA = t ∧ tB = t / 6 ∧ (tA + tB) = 3) : t = 21 :=
by
  sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l1565_156512


namespace NUMINAMATH_GPT_additional_money_needed_l1565_156563

-- Define the initial conditions as assumptions
def initial_bales : ℕ := 15
def previous_cost_per_bale : ℕ := 20
def multiplier : ℕ := 3
def new_cost_per_bale : ℕ := 27

-- Define the problem statement
theorem additional_money_needed :
  let initial_cost := initial_bales * previous_cost_per_bale 
  let new_bales := initial_bales * multiplier
  let new_cost := new_bales * new_cost_per_bale
  new_cost - initial_cost = 915 :=
by
  sorry

end NUMINAMATH_GPT_additional_money_needed_l1565_156563


namespace NUMINAMATH_GPT_missing_dimension_of_soap_box_l1565_156561

theorem missing_dimension_of_soap_box 
  (volume_carton : ℕ) 
  (volume_soap_box : ℕ)
  (number_of_boxes : ℕ)
  (x : ℕ) 
  (h1 : volume_carton = 25 * 48 * 60) 
  (h2 : volume_soap_box = x * 6 * 5)
  (h3: number_of_boxes = 300)
  (h4 : number_of_boxes * volume_soap_box = volume_carton) : 
  x = 8 := by 
  sorry

end NUMINAMATH_GPT_missing_dimension_of_soap_box_l1565_156561


namespace NUMINAMATH_GPT_protein_in_steak_is_correct_l1565_156580

-- Definitions of the conditions
def collagen_protein_per_scoop : ℕ := 18 / 2 -- 9 grams
def protein_powder_per_scoop : ℕ := 21 -- 21 grams

-- Define the total protein consumed
def total_protein (collagen_scoops protein_scoops : ℕ) (protein_from_steak : ℕ) : ℕ :=
  collagen_protein_per_scoop * collagen_scoops + protein_powder_per_scoop * protein_scoops + protein_from_steak

-- Condition in the problem
def total_protein_consumed : ℕ := 86

-- Prove that the protein in the steak is 56 grams
theorem protein_in_steak_is_correct : 
  total_protein 1 1 56 = total_protein_consumed :=
sorry

end NUMINAMATH_GPT_protein_in_steak_is_correct_l1565_156580


namespace NUMINAMATH_GPT_inequality_solution_l1565_156599

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1565_156599


namespace NUMINAMATH_GPT_jan_keeps_on_hand_l1565_156507

theorem jan_keeps_on_hand (total_length : ℕ) (section_length : ℕ) (friend_fraction : ℚ) (storage_fraction : ℚ) 
  (total_sections : ℕ) (sections_to_friend : ℕ) (remaining_sections : ℕ) (sections_in_storage : ℕ) (sections_on_hand : ℕ) :
  total_length = 1000 → section_length = 25 → friend_fraction = 1 / 4 → storage_fraction = 1 / 2 →
  total_sections = total_length / section_length →
  sections_to_friend = friend_fraction * total_sections →
  remaining_sections = total_sections - sections_to_friend →
  sections_in_storage = storage_fraction * remaining_sections →
  sections_on_hand = remaining_sections - sections_in_storage →
  sections_on_hand = 15 :=
by sorry

end NUMINAMATH_GPT_jan_keeps_on_hand_l1565_156507


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l1565_156597

theorem cylindrical_to_rectangular (r θ z : ℝ) (h₁ : r = 10) (h₂ : θ = Real.pi / 6) (h₃ : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5 * Real.sqrt 3, 5, 2) := 
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l1565_156597


namespace NUMINAMATH_GPT_point_not_in_region_l1565_156516

-- Define the inequality
def inequality (x y : ℝ) : Prop := 3 * x + 2 * y < 6

-- Points definition
def point := ℝ × ℝ

-- Points to be checked
def p1 : point := (0, 0)
def p2 : point := (1, 1)
def p3 : point := (0, 2)
def p4 : point := (2, 0)

-- Conditions stating that certain points satisfy the inequality
axiom h1 : inequality p1.1 p1.2
axiom h2 : inequality p2.1 p2.2
axiom h3 : inequality p3.1 p3.2

-- Goal: Prove that point (2,0) does not satisfy the inequality
theorem point_not_in_region : ¬ inequality p4.1 p4.2 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_point_not_in_region_l1565_156516


namespace NUMINAMATH_GPT_garden_area_increase_l1565_156564

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 20
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def circle_radius : ℝ := original_garden_perimeter / (2 * Real.pi)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius ^ 2)

noncomputable def area_increase : ℝ := circle_area - original_garden_area

theorem garden_area_increase :
  area_increase = (6400 / Real.pi) - 1200 :=
by 
  sorry -- proof goes here

end NUMINAMATH_GPT_garden_area_increase_l1565_156564


namespace NUMINAMATH_GPT_effective_writing_speed_is_750_l1565_156509

-- Definitions based on given conditions in problem part a)
def total_words : ℕ := 60000
def total_hours : ℕ := 100
def break_hours : ℕ := 20
def effective_hours : ℕ := total_hours - break_hours
def effective_writing_speed : ℕ := total_words / effective_hours

-- Statement to be proved
theorem effective_writing_speed_is_750 : effective_writing_speed = 750 := by
  sorry

end NUMINAMATH_GPT_effective_writing_speed_is_750_l1565_156509


namespace NUMINAMATH_GPT_monthly_interest_rate_l1565_156543

-- Define the principal amount (initial amount).
def principal : ℝ := 200

-- Define the final amount after 2 months (A).
def amount_after_two_months : ℝ := 222

-- Define the number of months (n).
def months : ℕ := 2

-- Define the monthly interest rate (r) we need to prove.
def interest_rate : ℝ := 0.053

-- Main statement to prove
theorem monthly_interest_rate :
  amount_after_two_months = principal * (1 + interest_rate)^months :=
sorry

end NUMINAMATH_GPT_monthly_interest_rate_l1565_156543


namespace NUMINAMATH_GPT_time_interval_for_7_students_l1565_156506

-- Definitions from conditions
def students_per_ride : ℕ := 7
def total_students : ℕ := 21
def total_time : ℕ := 15

-- Statement of the problem
theorem time_interval_for_7_students : (total_time / (total_students / students_per_ride)) = 5 := 
by sorry

end NUMINAMATH_GPT_time_interval_for_7_students_l1565_156506


namespace NUMINAMATH_GPT_ratio_debt_manny_to_annika_l1565_156576

-- Define the conditions
def money_jericho_has : ℕ := 30
def debt_to_annika : ℕ := 14
def remaining_money_after_debts : ℕ := 9

-- Define the amount Jericho owes Manny
def debt_to_manny : ℕ := money_jericho_has - debt_to_annika - remaining_money_after_debts

-- Prove the ratio of amount Jericho owes Manny to the amount he owes Annika is 1:2
theorem ratio_debt_manny_to_annika :
  debt_to_manny * 2 = debt_to_annika :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_debt_manny_to_annika_l1565_156576


namespace NUMINAMATH_GPT_cistern_fill_time_l1565_156590

-- Define the problem conditions
def pipe_p_fill_time : ℕ := 10
def pipe_q_fill_time : ℕ := 15
def joint_filling_time : ℕ := 2
def remaining_fill_time : ℕ := 10 -- This is the answer we need to prove

-- Prove that the remaining fill time is equal to 10 minutes
theorem cistern_fill_time :
  (joint_filling_time * (1 / pipe_p_fill_time + 1 / pipe_q_fill_time) + (remaining_fill_time / pipe_q_fill_time)) = 1 :=
sorry

end NUMINAMATH_GPT_cistern_fill_time_l1565_156590


namespace NUMINAMATH_GPT_sqrt_extraction_count_l1565_156519

theorem sqrt_extraction_count (p : ℕ) [Fact p.Prime] : 
    ∃ k, k = (p + 1) / 2 ∧ ∀ n < p, ∃ x < p, x^2 ≡ n [MOD p] ↔ n < k := 
by
  sorry

end NUMINAMATH_GPT_sqrt_extraction_count_l1565_156519


namespace NUMINAMATH_GPT_balls_in_boxes_l1565_156539

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1565_156539


namespace NUMINAMATH_GPT_consecutive_tree_distance_l1565_156598

theorem consecutive_tree_distance (yard_length : ℕ) (num_trees : ℕ) (distance : ℚ)
  (h1 : yard_length = 520) 
  (h2 : num_trees = 40) :
  distance = yard_length / (num_trees - 1) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_consecutive_tree_distance_l1565_156598


namespace NUMINAMATH_GPT_inequality_abc_ge_1_sqrt_abcd_l1565_156500

theorem inequality_abc_ge_1_sqrt_abcd
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_ge_1_sqrt_abcd_l1565_156500


namespace NUMINAMATH_GPT_ab_equiv_l1565_156578

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ab_equiv_l1565_156578


namespace NUMINAMATH_GPT_sugar_percentage_in_new_solution_l1565_156572

open Real

noncomputable def original_volume : ℝ := 450
noncomputable def original_sugar_percentage : ℝ := 20 / 100
noncomputable def added_sugar : ℝ := 7.5
noncomputable def added_water : ℝ := 20
noncomputable def added_kola : ℝ := 8.1
noncomputable def added_flavoring : ℝ := 2.3

noncomputable def original_sugar_amount : ℝ := original_volume * original_sugar_percentage
noncomputable def total_sugar_amount : ℝ := original_sugar_amount + added_sugar
noncomputable def new_total_volume : ℝ := original_volume + added_water + added_kola + added_flavoring + added_sugar
noncomputable def new_sugar_percentage : ℝ := (total_sugar_amount / new_total_volume) * 100

theorem sugar_percentage_in_new_solution : abs (new_sugar_percentage - 19.97) < 0.01 := sorry

end NUMINAMATH_GPT_sugar_percentage_in_new_solution_l1565_156572


namespace NUMINAMATH_GPT_lanes_on_road_l1565_156591

theorem lanes_on_road (num_lanes : ℕ)
  (h1 : ∀ trucks_per_lane cars_per_lane total_vehicles, 
          cars_per_lane = 2 * (trucks_per_lane * num_lanes) ∧
          trucks_per_lane = 60 ∧
          total_vehicles = num_lanes * (trucks_per_lane + cars_per_lane) ∧
          total_vehicles = 2160) :
  num_lanes = 12 :=
by
  sorry

end NUMINAMATH_GPT_lanes_on_road_l1565_156591


namespace NUMINAMATH_GPT_candidate_failed_by_25_marks_l1565_156532

-- Define the given conditions
def maximum_marks : ℝ := 127.27
def passing_percentage : ℝ := 0.55
def marks_secured : ℝ := 45

-- Define the minimum passing marks
def minimum_passing_marks : ℝ := passing_percentage * maximum_marks

-- Define the number of failing marks the candidate missed
def failing_marks : ℝ := minimum_passing_marks - marks_secured

-- Define the main theorem to prove the candidate failed by 25 marks
theorem candidate_failed_by_25_marks :
  failing_marks = 25 := 
by
  sorry

end NUMINAMATH_GPT_candidate_failed_by_25_marks_l1565_156532


namespace NUMINAMATH_GPT_discard_sacks_l1565_156551

theorem discard_sacks (harvested_sacks_per_day : ℕ) (oranges_per_day : ℕ) (oranges_per_sack : ℕ) :
  harvested_sacks_per_day = 76 → oranges_per_day = 600 → oranges_per_sack = 50 → 
  harvested_sacks_per_day - oranges_per_day / oranges_per_sack = 64 :=
by
  intros h1 h2 h3
  -- Automatically passes the proof as a placeholder
  sorry

end NUMINAMATH_GPT_discard_sacks_l1565_156551


namespace NUMINAMATH_GPT_reeya_third_subject_score_l1565_156544

theorem reeya_third_subject_score
  (score1 score2 score4 : ℕ)
  (avg_score : ℕ)
  (num_subjects : ℕ)
  (total_score : ℕ)
  (score3 : ℕ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  avg_score = 75 →
  num_subjects = 4 →
  total_score = avg_score * num_subjects →
  score1 + score2 + score3 + score4 = total_score →
  score3 = 83 :=
by
  intros h1 h2 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_reeya_third_subject_score_l1565_156544


namespace NUMINAMATH_GPT_evaluate_expression_l1565_156575

theorem evaluate_expression (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
    (x - (y - z)) - ((x - y) - z) = 14 := by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1565_156575


namespace NUMINAMATH_GPT_percentOfNonUnionWomenIs90_l1565_156502

variable (totalEmployees : ℕ) (percentMen : ℚ) (percentUnionized : ℚ) (percentUnionizedMen : ℚ)

noncomputable def percentNonUnionWomen : ℚ :=
  let numberOfMen := percentMen * totalEmployees
  let numberOfUnionEmployees := percentUnionized * totalEmployees
  let numberOfUnionMen := percentUnionizedMen * numberOfUnionEmployees
  let numberOfNonUnionEmployees := totalEmployees - numberOfUnionEmployees
  let numberOfNonUnionMen := numberOfMen - numberOfUnionMen
  let numberOfNonUnionWomen := numberOfNonUnionEmployees - numberOfNonUnionMen
  (numberOfNonUnionWomen / numberOfNonUnionEmployees) * 100

theorem percentOfNonUnionWomenIs90
  (h1 : percentMen = 46 / 100)
  (h2 : percentUnionized = 60 / 100)
  (h3 : percentUnionizedMen = 70 / 100) : percentNonUnionWomen 100 46 60 70 = 90 :=
sorry

end NUMINAMATH_GPT_percentOfNonUnionWomenIs90_l1565_156502


namespace NUMINAMATH_GPT_extremum_condition_l1565_156508

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b * x + a^2)
  (h2 : f 1 = 10)
  (h3 : deriv f 1 = 0) :
  a + b = -7 :=
sorry

end NUMINAMATH_GPT_extremum_condition_l1565_156508


namespace NUMINAMATH_GPT_largest_possible_perimeter_l1565_156592

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 11) : 
    5 + 6 + x ≤ 21 := 
  sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l1565_156592


namespace NUMINAMATH_GPT_inverse_of_f_at_neg2_l1565_156579

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the property of the inverse function we need to prove
theorem inverse_of_f_at_neg2 : f (-(3/2)) = -2 :=
  by
    -- Placeholder for the proof
    sorry

end NUMINAMATH_GPT_inverse_of_f_at_neg2_l1565_156579


namespace NUMINAMATH_GPT_geometric_sequence_150th_term_l1565_156530

-- Given conditions
def a1 : ℤ := 5
def a2 : ℤ := -10

-- Computation of common ratio
def r : ℤ := a2 / a1

-- Definition of the n-th term in geometric sequence
def nth_term (n : ℕ) : ℤ :=
  a1 * r^(n-1)

-- Statement to prove
theorem geometric_sequence_150th_term :
  nth_term 150 = -5 * 2^149 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_150th_term_l1565_156530


namespace NUMINAMATH_GPT_difference_of_squares_l1565_156596

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 26) (h2 : x * y = 168) : x^2 - y^2 = 52 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1565_156596


namespace NUMINAMATH_GPT_cubic_sum_l1565_156581

theorem cubic_sum (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + x * z + y * z = -5) (h3 : x * y * z = -6) :
  x^3 + y^3 + z^3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l1565_156581


namespace NUMINAMATH_GPT_clea_ride_escalator_time_l1565_156522

def clea_time_not_walking (x k y : ℝ) : Prop :=
  60 * x = y ∧ 24 * (x + k) = y ∧ 1.5 * x = k ∧ 40 = y / k

theorem clea_ride_escalator_time :
  ∀ (x y k : ℝ), 60 * x = y → 24 * (x + k) = y → (1.5 * x = k) → 40 = y / k :=
by
  intros x y k H1 H2 H3
  sorry

end NUMINAMATH_GPT_clea_ride_escalator_time_l1565_156522


namespace NUMINAMATH_GPT_difference_of_sums_l1565_156568

def even_numbers_sum (n : ℕ) : ℕ := (n * (n + 1))
def odd_numbers_sum (n : ℕ) : ℕ := n^2

theorem difference_of_sums : 
  even_numbers_sum 3003 - odd_numbers_sum 3003 = 7999 := 
by {
  sorry 
}

end NUMINAMATH_GPT_difference_of_sums_l1565_156568


namespace NUMINAMATH_GPT_probability_of_success_l1565_156555

def prob_successful_attempt := 0.5

def prob_unsuccessful_attempt := 1 - prob_successful_attempt

def all_fail_prob := prob_unsuccessful_attempt ^ 4

def at_least_one_success_prob := 1 - all_fail_prob

theorem probability_of_success :
  at_least_one_success_prob = 0.9375 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_probability_of_success_l1565_156555


namespace NUMINAMATH_GPT_num_divisors_count_l1565_156560

theorem num_divisors_count (n : ℕ) (m : ℕ) (H : m = 32784) :
  (∃ S : Finset ℕ, (∀ x ∈ S, x ∈ (Finset.range 10) ∧ m % x = 0) ∧ S.card = n) ↔ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_divisors_count_l1565_156560


namespace NUMINAMATH_GPT_boy_age_proof_l1565_156552

theorem boy_age_proof (P X : ℕ) (hP : P = 16) (hcond : P - X = (P + 4) / 2) : X = 6 :=
by
  sorry

end NUMINAMATH_GPT_boy_age_proof_l1565_156552


namespace NUMINAMATH_GPT_original_length_before_sharpening_l1565_156524

/-- Define the current length of the pencil after sharpening -/
def current_length : ℕ := 14

/-- Define the length of the pencil that was sharpened off -/
def sharpened_off_length : ℕ := 17

/-- Prove that the original length of the pencil before sharpening was 31 inches -/
theorem original_length_before_sharpening : current_length + sharpened_off_length = 31 := by
  sorry

end NUMINAMATH_GPT_original_length_before_sharpening_l1565_156524


namespace NUMINAMATH_GPT_smallest_positive_integer_x_for_2520x_eq_m_cubed_l1565_156540

theorem smallest_positive_integer_x_for_2520x_eq_m_cubed :
  ∃ (M x : ℕ), x > 0 ∧ 2520 * x = M^3 ∧ (∀ y, y > 0 ∧ 2520 * y = M^3 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_for_2520x_eq_m_cubed_l1565_156540


namespace NUMINAMATH_GPT_result_of_fractions_mult_l1565_156541

theorem result_of_fractions_mult (a b c d : ℚ) (x : ℕ) :
  a = 3 / 4 →
  b = 1 / 2 →
  c = 2 / 5 →
  d = 5100 →
  a * b * c * d = 765 := by
  sorry

end NUMINAMATH_GPT_result_of_fractions_mult_l1565_156541


namespace NUMINAMATH_GPT_proof_l_squared_l1565_156525

noncomputable def longest_line_segment (diameter : ℝ) (sectors : ℕ) : ℝ :=
  let R := diameter / 2
  let theta := (2 * Real.pi) / sectors
  2 * R * (Real.sin (theta / 2))

theorem proof_l_squared :
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  l^2 = 162 := by
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  have h : l^2 = 162 := sorry
  exact h

end NUMINAMATH_GPT_proof_l_squared_l1565_156525


namespace NUMINAMATH_GPT_smallest_three_digit_number_l1565_156573

theorem smallest_three_digit_number (x : ℤ) (h1 : x - 7 % 7 = 0) (h2 : x - 8 % 8 = 0) (h3 : x - 9 % 9 = 0) : x = 504 := 
sorry

end NUMINAMATH_GPT_smallest_three_digit_number_l1565_156573


namespace NUMINAMATH_GPT_least_odd_prime_factor_of_2023_pow_8_add_1_l1565_156545

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end NUMINAMATH_GPT_least_odd_prime_factor_of_2023_pow_8_add_1_l1565_156545


namespace NUMINAMATH_GPT_points_difference_l1565_156569

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end NUMINAMATH_GPT_points_difference_l1565_156569


namespace NUMINAMATH_GPT_coeff_x4_in_expansion_l1565_156521

open Nat

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def coefficient_x4_term : ℕ := binom 9 4

noncomputable def constant_term : ℕ := 243 * 4

theorem coeff_x4_in_expansion : coefficient_x4_term * 972 * Real.sqrt 2 = 122472 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x4_in_expansion_l1565_156521


namespace NUMINAMATH_GPT_smallest_x_2_abs_eq_24_l1565_156582

theorem smallest_x_2_abs_eq_24 : ∃ x : ℝ, (2 * |x - 10| = 24) ∧ (∀ y : ℝ, (2 * |y - 10| = 24) -> x ≤ y) := 
sorry

end NUMINAMATH_GPT_smallest_x_2_abs_eq_24_l1565_156582


namespace NUMINAMATH_GPT_number_of_sheep_l1565_156501

theorem number_of_sheep (s d : ℕ) 
  (h1 : s + d = 15)
  (h2 : 4 * s + 2 * d = 22 + 2 * (s + d)) : 
  s = 11 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sheep_l1565_156501


namespace NUMINAMATH_GPT_correct_statement_about_Digital_Earth_l1565_156571

-- Definitions of the statements
def statement_A : Prop :=
  "Digital Earth is a reflection of the real Earth through digital means" = "Correct statement about Digital Earth"

def statement_B : Prop :=
  "Digital Earth is an extension of GIS technology" = "Correct statement about Digital Earth"

def statement_C : Prop :=
  "Digital Earth can only achieve global information sharing through the internet" = "Correct statement about Digital Earth"

def statement_D : Prop :=
  "The core idea of Digital Earth is to use digital means to uniformly address Earth's issues" = "Correct statement about Digital Earth"

-- Theorem that needs to be proved 
theorem correct_statement_about_Digital_Earth : statement_C :=
by 
  sorry

end NUMINAMATH_GPT_correct_statement_about_Digital_Earth_l1565_156571


namespace NUMINAMATH_GPT_diagonals_in_eight_sided_polygon_l1565_156542

-- Definitions based on the conditions
def n := 8  -- Number of sides
def right_angles := 2  -- Number of right angles

-- Calculating the number of diagonals using the formula
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Lean statement for the problem
theorem diagonals_in_eight_sided_polygon : num_diagonals n = 20 :=
by
  -- Substitute n = 8 into the formula and simplify
  sorry

end NUMINAMATH_GPT_diagonals_in_eight_sided_polygon_l1565_156542


namespace NUMINAMATH_GPT_basketball_free_throws_l1565_156574

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : b = a - 2)
  (h3 : 2 * a + 3 * b + x = 68) : x = 44 :=
by
  sorry

end NUMINAMATH_GPT_basketball_free_throws_l1565_156574


namespace NUMINAMATH_GPT_find_a_solve_inequality_intervals_of_monotonicity_l1565_156514

-- Problem 1: Prove a = 2 given conditions
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : Real.log 3 / Real.log a > Real.log 2 / Real.log a) 
    (h₃ : Real.log (2 * a) / Real.log a - Real.log a / Real.log a = 1) : a = 2 := 
  by
  sorry

-- Problem 2: Prove the solution interval for inequality
theorem solve_inequality (x a : ℝ) (h₀ : 1 < x) (h₁ : x < 3 / 2) : 
    Real.log (x - 1) / Real.log (1 / 3) > Real.log (a - x) / Real.log (1 / 3) :=
  by
  have ha : a = 2 := sorry
  sorry

-- Problem 3: Prove intervals of monotonicity for g(x)
theorem intervals_of_monotonicity (x : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = 1 - Real.log x / Real.log 2) ∧ 
  (∀ x : ℝ, x > 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = Real.log x / Real.log 2 - 1) :=
  by
  sorry

end NUMINAMATH_GPT_find_a_solve_inequality_intervals_of_monotonicity_l1565_156514


namespace NUMINAMATH_GPT_light_flash_time_l1565_156567

/--
A light flashes every few seconds. In 3/4 of an hour, it flashes 300 times.
Prove that it takes 9 seconds for the light to flash once.
-/
theorem light_flash_time : 
  (3 / 4 * 60 * 60) / 300 = 9 :=
by
  sorry

end NUMINAMATH_GPT_light_flash_time_l1565_156567


namespace NUMINAMATH_GPT_flavors_needed_this_year_l1565_156536

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end NUMINAMATH_GPT_flavors_needed_this_year_l1565_156536


namespace NUMINAMATH_GPT_isosceles_triangle_range_l1565_156538

theorem isosceles_triangle_range (x : ℝ) (h1 : 0 < x) (h2 : 2 * x + (10 - 2 * x) = 10):
  (5 / 2) < x ∧ x < 5 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_range_l1565_156538


namespace NUMINAMATH_GPT_valid_third_side_length_l1565_156518

theorem valid_third_side_length (x : ℝ) : 4 < x ∧ x < 14 ↔ (((5 : ℝ) + 9 > x) ∧ (x + 5 > 9) ∧ (x + 9 > 5)) :=
by 
  sorry

end NUMINAMATH_GPT_valid_third_side_length_l1565_156518


namespace NUMINAMATH_GPT_overall_average_score_l1565_156505

variables (average_male average_female sum_male sum_female total_sum : ℕ)
variables (count_male count_female total_count : ℕ)

def average_score (sum : ℕ) (count : ℕ) : ℕ := sum / count

theorem overall_average_score
  (average_male : ℕ := 84)
  (count_male : ℕ := 8)
  (average_female : ℕ := 92)
  (count_female : ℕ := 24)
  (sum_male : ℕ := count_male * average_male)
  (sum_female : ℕ := count_female * average_female)
  (total_sum : ℕ := sum_male + sum_female)
  (total_count : ℕ := count_male + count_female) :
  average_score total_sum total_count = 90 := 
sorry

end NUMINAMATH_GPT_overall_average_score_l1565_156505
