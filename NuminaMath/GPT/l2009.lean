import Mathlib

namespace NUMINAMATH_GPT_dot_product_MN_MO_is_8_l2009_200931

-- Define the circle O as a set of points (x, y) such that x^2 + y^2 = 9
def is_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the length of the chord MN in the circle
def chord_length (M N : ℝ × ℝ) : Prop :=
  let (x1, y1) := M
  let (x2, y2) := N
  (x1 - x2)^2 + (y1 - y2)^2 = 16

-- Define the vector MN and MO
def vector_dot_product (M N O : ℝ × ℝ) : ℝ :=
  let (x1, y1) := M
  let (x2, y2) := N
  let (x0, y0) := O
  let v1 := (x2 - x1, y2 - y1)
  let v2 := (x0 - x1, y0 - y1)
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the origin point O (center of the circle)
def O : ℝ × ℝ := (0, 0)

-- The theorem to prove
theorem dot_product_MN_MO_is_8 (M N : ℝ × ℝ) (hM : is_circle M.1 M.2) (hN : is_circle N.1 N.2) (hMN : chord_length M N) :
  vector_dot_product M N O = 8 :=
sorry

end NUMINAMATH_GPT_dot_product_MN_MO_is_8_l2009_200931


namespace NUMINAMATH_GPT_find_s_and_x_l2009_200901

theorem find_s_and_x (s x t : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3.75) :
  s = 0.5 ∧ x = s / 2 → x = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_find_s_and_x_l2009_200901


namespace NUMINAMATH_GPT_triangle_inequality_l2009_200982

-- Define the conditions as Lean hypotheses
variables {a b c : ℝ}

-- Lean statement for the problem
theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < 1/2 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l2009_200982


namespace NUMINAMATH_GPT_C_younger_than_A_l2009_200978

variables (A B C : ℕ)

-- Original Condition
axiom age_condition : A + B = B + C + 17

-- Lean Statement to Prove
theorem C_younger_than_A (A B C : ℕ) (h : A + B = B + C + 17) : C + 17 = A :=
by {
  -- Proof would go here but is omitted.
  sorry
}

end NUMINAMATH_GPT_C_younger_than_A_l2009_200978


namespace NUMINAMATH_GPT_perfect_squares_in_range_100_400_l2009_200981

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end NUMINAMATH_GPT_perfect_squares_in_range_100_400_l2009_200981


namespace NUMINAMATH_GPT_pq_plus_qr_plus_rp_cubic_1_l2009_200907

theorem pq_plus_qr_plus_rp_cubic_1 (p q r : ℝ) 
  (h1 : p + q + r = 0)
  (h2 : p * q + p * r + q * r = -2)
  (h3 : p * q * r = 2) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = -6 :=
by
  sorry

end NUMINAMATH_GPT_pq_plus_qr_plus_rp_cubic_1_l2009_200907


namespace NUMINAMATH_GPT_randy_used_36_blocks_l2009_200938

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks left
def blocks_left : ℕ := 23

-- Define the number of blocks used
def blocks_used (initial left : ℕ) : ℕ := initial - left

-- Prove that Randy used 36 blocks
theorem randy_used_36_blocks : blocks_used initial_blocks blocks_left = 36 := 
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_randy_used_36_blocks_l2009_200938


namespace NUMINAMATH_GPT_parabola_directrix_l2009_200969

-- Defining the given condition
def given_parabola_equation (x y : ℝ) : Prop := y = 2 * x^2

-- Defining the expected directrix equation for the parabola
def directrix_equation (y : ℝ) : Prop := y = -1 / 8

-- The theorem we aim to prove
theorem parabola_directrix :
  (∀ x y : ℝ, given_parabola_equation x y) → (directrix_equation (-1 / 8)) :=
by
  -- Using 'sorry' here since the proof is not required
  sorry

end NUMINAMATH_GPT_parabola_directrix_l2009_200969


namespace NUMINAMATH_GPT_first_stopover_distance_l2009_200903

theorem first_stopover_distance 
  (total_distance : ℕ) 
  (second_stopover_distance : ℕ) 
  (distance_after_second_stopover : ℕ) :
  total_distance = 436 → 
  second_stopover_distance = 236 → 
  distance_after_second_stopover = 68 →
  second_stopover_distance - (total_distance - second_stopover_distance - distance_after_second_stopover) = 104 :=
by
  intros
  sorry

end NUMINAMATH_GPT_first_stopover_distance_l2009_200903


namespace NUMINAMATH_GPT_incircle_angle_b_l2009_200909

open Real

theorem incircle_angle_b
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (angle_AOC_eq_4_MKN : ∀ (MKN : ℝ), 4 * MKN = 180 - (180 - γ) / 2 - (180 - α) / 2) :
    β = 108 :=
by
  -- Proof will be handled here.
  sorry

end NUMINAMATH_GPT_incircle_angle_b_l2009_200909


namespace NUMINAMATH_GPT_inequality_solution_1_inequality_solution_2_l2009_200977

-- Definition for part 1
theorem inequality_solution_1 (x : ℝ) : x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 :=
sorry

-- Definition for part 2
theorem inequality_solution_2 (x : ℝ) : (1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5 :=
sorry

end NUMINAMATH_GPT_inequality_solution_1_inequality_solution_2_l2009_200977


namespace NUMINAMATH_GPT_investment_duration_l2009_200998

theorem investment_duration 
  (P : ℝ) (A : ℝ) (r : ℝ) (t : ℝ)
  (h1 : P = 939.60)
  (h2 : A = 1120)
  (h3 : r = 8) :
  t = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_investment_duration_l2009_200998


namespace NUMINAMATH_GPT_dot_product_OA_OB_l2009_200918

theorem dot_product_OA_OB :
  let A := (Real.cos 110, Real.sin 110)
  let B := (Real.cos 50, Real.sin 50)
  (A.1 * B.1 + A.2 * B.2) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_OA_OB_l2009_200918


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_problem3_solution_l2009_200940

-- Problem 1
theorem problem1_solution (x : ℝ) :
  (6 * x - 1) ^ 2 = 25 ↔ (x = 1 ∨ x = -2 / 3) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) :
  4 * x^2 - 1 = 12 * x ↔ (x = 3 / 2 + (Real.sqrt 10) / 2 ∨ x = 3 / 2 - (Real.sqrt 10) / 2) :=
sorry

-- Problem 3
theorem problem3_solution (x : ℝ) :
  x * (x - 7) = 8 * (7 - x) ↔ (x = 7 ∨ x = -8) :=
sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_problem3_solution_l2009_200940


namespace NUMINAMATH_GPT_contractor_net_amount_l2009_200930

-- Definitions based on conditions
def total_days : ℕ := 30
def pay_per_day : ℝ := 25
def fine_per_absence_day : ℝ := 7.5
def days_absent : ℕ := 6

-- Calculate days worked
def days_worked : ℕ := total_days - days_absent

-- Calculate total earnings
def earnings : ℝ := days_worked * pay_per_day

-- Calculate total fine
def fine : ℝ := days_absent * fine_per_absence_day

-- Calculate net amount received by the contractor
def net_amount : ℝ := earnings - fine

-- Problem statement: Prove that the net amount is Rs. 555
theorem contractor_net_amount : net_amount = 555 := by
  sorry

end NUMINAMATH_GPT_contractor_net_amount_l2009_200930


namespace NUMINAMATH_GPT_roots_product_eq_348_l2009_200917

theorem roots_product_eq_348 (d e : ℤ) 
  (h : ∀ (s : ℂ), s^2 - 2*s - 1 = 0 → s^5 - d*s - e = 0) : 
  d * e = 348 :=
sorry

end NUMINAMATH_GPT_roots_product_eq_348_l2009_200917


namespace NUMINAMATH_GPT_remainder_when_a_plus_b_div_40_is_28_l2009_200934

theorem remainder_when_a_plus_b_div_40_is_28 :
  ∃ k j : ℤ, (a = 80 * k + 74 ∧ b = 120 * j + 114) → (a + b) % 40 = 28 := by
  sorry

end NUMINAMATH_GPT_remainder_when_a_plus_b_div_40_is_28_l2009_200934


namespace NUMINAMATH_GPT_total_families_l2009_200966

theorem total_families (F_2dogs F_1dog F_2cats total_animals total_families : ℕ) 
  (h1: F_2dogs = 15)
  (h2: F_1dog = 20)
  (h3: total_animals = 80)
  (h4: 2 * F_2dogs + F_1dog + 2 * F_2cats = total_animals) :
  total_families = F_2dogs + F_1dog + F_2cats := 
by 
  sorry

end NUMINAMATH_GPT_total_families_l2009_200966


namespace NUMINAMATH_GPT_income_ratio_l2009_200943

variable (U B: ℕ) -- Uma's and Bala's incomes
variable (x: ℕ)  -- Common multiplier for expenditures
variable (savings_amt: ℕ := 2000)  -- Savings amount for both
variable (ratio_expenditure_uma : ℕ := 7)
variable (ratio_expenditure_bala : ℕ := 6)
variable (uma_income : ℕ := 16000)
variable (bala_expenditure: ℕ)

-- Conditions of the problem
-- Uma's Expenditure Calculation
axiom ua_exp_calc : savings_amt = uma_income - ratio_expenditure_uma * x
-- Bala's Expenditure Calculation
axiom bala_income_calc : savings_amt = B - ratio_expenditure_bala * x

theorem income_ratio (h1: U = uma_income) (h2: B = bala_expenditure):
  U * ratio_expenditure_bala = B * ratio_expenditure_uma :=
sorry

end NUMINAMATH_GPT_income_ratio_l2009_200943


namespace NUMINAMATH_GPT_number_of_integers_satisfying_condition_l2009_200974

def satisfies_condition (n : ℤ) : Prop :=
  1 + Int.floor (101 * n / 102) = Int.ceil (98 * n / 99)

noncomputable def number_of_solutions : ℤ :=
  10198

theorem number_of_integers_satisfying_condition :
  (∃ n : ℤ, satisfies_condition n) ↔ number_of_solutions = 10198 :=
sorry

end NUMINAMATH_GPT_number_of_integers_satisfying_condition_l2009_200974


namespace NUMINAMATH_GPT_sum_of_four_consecutive_integers_divisible_by_two_l2009_200972

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n-1) + n + (n+1) + (n+2)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_four_consecutive_integers_divisible_by_two_l2009_200972


namespace NUMINAMATH_GPT_Pyarelal_loss_l2009_200973

variables (capital_of_pyarelal capital_of_ashok : ℝ) (total_loss : ℝ)

def is_ninth (a b : ℝ) : Prop := a = b / 9

def applied_loss (loss : ℝ) (ratio : ℝ) : ℝ := ratio * loss

theorem Pyarelal_loss (h1: is_ninth capital_of_ashok capital_of_pyarelal) 
                        (h2: total_loss = 1600) : 
                        applied_loss total_loss (9/10) = 1440 :=
by 
  unfold is_ninth at h1
  sorry

end NUMINAMATH_GPT_Pyarelal_loss_l2009_200973


namespace NUMINAMATH_GPT_equalize_expenses_l2009_200926

variable {x y : ℝ} 

theorem equalize_expenses (h : x > y) : (x + y) / 2 - y = (x - y) / 2 :=
by sorry

end NUMINAMATH_GPT_equalize_expenses_l2009_200926


namespace NUMINAMATH_GPT_sin_35pi_over_6_l2009_200912

theorem sin_35pi_over_6 : Real.sin (35 * Real.pi / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_35pi_over_6_l2009_200912


namespace NUMINAMATH_GPT_production_days_l2009_200992

theorem production_days (n : ℕ) (P : ℕ)
  (h1 : P = 40 * n)
  (h2 : (P + 90) / (n + 1) = 45) :
  n = 9 :=
by
  sorry

end NUMINAMATH_GPT_production_days_l2009_200992


namespace NUMINAMATH_GPT_rahul_deepak_age_ratio_l2009_200997

-- Define the conditions
variables (R D : ℕ)
axiom deepak_age : D = 33
axiom rahul_future_age : R + 6 = 50

-- Define the theorem to prove the ratio
theorem rahul_deepak_age_ratio : R / D = 4 / 3 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_rahul_deepak_age_ratio_l2009_200997


namespace NUMINAMATH_GPT_sum_a_b_when_pow_is_max_l2009_200971

theorem sum_a_b_when_pow_is_max (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 1) (h_pow : a^b < 500) 
(h_max : ∀ (a' b' : ℕ), (a' > 0) -> (b' > 1) -> (a'^b' < 500) -> a^b >= a'^b') : a + b = 24 := by
  sorry

end NUMINAMATH_GPT_sum_a_b_when_pow_is_max_l2009_200971


namespace NUMINAMATH_GPT_problem_inequality_problem_equality_condition_l2009_200983

theorem problem_inequality (a b c : ℕ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

theorem problem_equality_condition (a b c : ℕ) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ a + 1 = b ∧ b + 1 = c :=
sorry

end NUMINAMATH_GPT_problem_inequality_problem_equality_condition_l2009_200983


namespace NUMINAMATH_GPT_men_with_ac_at_least_12_l2009_200996

-- Define the variables and conditions
variable (total_men : ℕ) (married_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (men_with_all_four : ℕ)

-- Assume the given conditions
axiom h1 : total_men = 100
axiom h2 : married_men = 82
axiom h3 : tv_men = 75
axiom h4 : radio_men = 85
axiom h5 : men_with_all_four = 12

-- Define the number of men with AC
variable (ac_men : ℕ)

-- State the proposition that the number of men with AC is at least 12
theorem men_with_ac_at_least_12 : ac_men ≥ 12 := sorry

end NUMINAMATH_GPT_men_with_ac_at_least_12_l2009_200996


namespace NUMINAMATH_GPT_initial_volume_of_solution_is_six_l2009_200993

theorem initial_volume_of_solution_is_six
  (V : ℝ)
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) :
  V = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_of_solution_is_six_l2009_200993


namespace NUMINAMATH_GPT_smallest_base_b_l2009_200994

theorem smallest_base_b (b : ℕ) : (b ≥ 1) → (b^2 ≤ 82) → (82 < b^3) → b = 5 := by
  sorry

end NUMINAMATH_GPT_smallest_base_b_l2009_200994


namespace NUMINAMATH_GPT_max_value_is_one_eighth_l2009_200945

noncomputable def find_max_value (a b c : ℝ) : ℝ :=
  a^2 * b^2 * c^2 * (a + b + c) / ((a + b)^3 * (b + c)^3)

theorem max_value_is_one_eighth (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  find_max_value a b c ≤ 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_is_one_eighth_l2009_200945


namespace NUMINAMATH_GPT_smallest_divisor_l2009_200988

noncomputable def even_four_digit_number (m : ℕ) : Prop :=
  1000 ≤ m ∧ m < 10000 ∧ m % 2 = 0

def divisor_ordered (m : ℕ) (d : ℕ) : Prop :=
  d ∣ m

theorem smallest_divisor (m : ℕ) (h1 : even_four_digit_number m) (h2 : divisor_ordered m 437) :
  ∃ d,  d > 437 ∧ divisor_ordered m d ∧ (∀ e, e > 437 → divisor_ordered m e → d ≤ e) ∧ d = 874 :=
sorry

end NUMINAMATH_GPT_smallest_divisor_l2009_200988


namespace NUMINAMATH_GPT_AM_QM_Muirhead_Inequality_l2009_200936

open Real

theorem AM_QM_Muirhead_Inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  ((a + b + c) / 3 = sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) ∧
  (sqrt ((a^2 + b^2 + c^2) / 3) = ((ab / c) + (bc / a) + (ca / b)) / 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_GPT_AM_QM_Muirhead_Inequality_l2009_200936


namespace NUMINAMATH_GPT_correct_statement_l2009_200932

-- Define the necessary variables
variables {a b c : ℝ}

-- State the theorem including the condition and the conclusion
theorem correct_statement (h : a > b) : b - c < a - c :=
by linarith


end NUMINAMATH_GPT_correct_statement_l2009_200932


namespace NUMINAMATH_GPT_arithmetic_seq_2a9_a10_l2009_200920

theorem arithmetic_seq_2a9_a10 (a : ℕ → ℕ) (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (arith_seq : ∀ n : ℕ, ∃ d : ℕ, a n = a 1 + (n - 1) * d) : 2 * a 9 - a 10 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_2a9_a10_l2009_200920


namespace NUMINAMATH_GPT_find_a_l2009_200951

noncomputable def pure_imaginary_simplification (a : ℝ) (i : ℂ) (hi : i * i = -1) : Prop :=
  let denom := (3 : ℂ) - (4 : ℂ) * i
  let numer := (15 : ℂ)
  let complex_num := a + numer / denom
  let simplified_real := a + (9 : ℝ) / (5 : ℝ)
  simplified_real = 0

theorem find_a (i : ℂ) (hi : i * i = -1) : pure_imaginary_simplification (- 9 / 5 : ℝ) i hi :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2009_200951


namespace NUMINAMATH_GPT_watermelons_eaten_l2009_200980

theorem watermelons_eaten (original left : ℕ) (h1 : original = 4) (h2 : left = 1) :
  original - left = 3 :=
by {
  -- Providing the proof steps is not necessary as per the instructions
  sorry
}

end NUMINAMATH_GPT_watermelons_eaten_l2009_200980


namespace NUMINAMATH_GPT_beneficial_for_kati_l2009_200955

variables (n : ℕ) (x y : ℝ)

theorem beneficial_for_kati (hn : n > 0) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y) / (n + 2) > (x + y / 2) / (n + 1) :=
sorry

end NUMINAMATH_GPT_beneficial_for_kati_l2009_200955


namespace NUMINAMATH_GPT_no_real_solutions_l2009_200904

theorem no_real_solutions : ∀ (x y : ℝ), ¬ (3 * x^2 + y^2 - 9 * x - 6 * y + 23 = 0) :=
by sorry

end NUMINAMATH_GPT_no_real_solutions_l2009_200904


namespace NUMINAMATH_GPT_triangle_angle_sum_l2009_200939

theorem triangle_angle_sum (angle_Q R P : ℝ)
  (h1 : R = 3 * angle_Q)
  (h2 : angle_Q = 30)
  (h3 : P + angle_Q + R = 180) :
    P = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l2009_200939


namespace NUMINAMATH_GPT_john_hiking_probability_l2009_200910

theorem john_hiking_probability :
  let P_rain := 0.3
  let P_sunny := 0.7
  let P_hiking_if_rain := 0.1
  let P_hiking_if_sunny := 0.9

  let P_hiking := P_rain * P_hiking_if_rain + P_sunny * P_hiking_if_sunny

  P_hiking = 0.66 := by
    sorry

end NUMINAMATH_GPT_john_hiking_probability_l2009_200910


namespace NUMINAMATH_GPT_max_servings_l2009_200976

-- Definitions based on the conditions
def servings_recipe := 3
def bananas_per_serving := 2 / servings_recipe
def strawberries_per_serving := 1 / servings_recipe
def yogurt_per_serving := 2 / servings_recipe

def emily_bananas := 4
def emily_strawberries := 3
def emily_yogurt := 6

-- Prove that Emily can make at most 6 servings while keeping the proportions the same
theorem max_servings :
  min (emily_bananas / bananas_per_serving) 
      (min (emily_strawberries / strawberries_per_serving) 
           (emily_yogurt / yogurt_per_serving)) = 6 := sorry

end NUMINAMATH_GPT_max_servings_l2009_200976


namespace NUMINAMATH_GPT_leak_drain_time_l2009_200963

theorem leak_drain_time (P L : ℝ) (hP : P = 1/2) (h_combined : P - L = 3/7) : 1 / L = 14 :=
by
  -- Definitions of the conditions
  -- The rate of the pump filling the tank
  have hP : P = 1 / 2 := hP
  -- The combined rate of the pump (filling) and leak (draining)
  have h_combined : P - L = 3 / 7 := h_combined
  -- From these definitions, continue the proof
  sorry

end NUMINAMATH_GPT_leak_drain_time_l2009_200963


namespace NUMINAMATH_GPT_find_y_l2009_200959

variable (h : ℕ) -- integral number of hours

-- Distance between A and B
def distance_AB : ℕ := 60

-- Speed and distance walked by woman starting at A
def speed_A : ℕ := 3
def distance_A (h : ℕ) : ℕ := speed_A * h

-- Speed and distance walked by woman starting at B
def speed_B_1st_hour : ℕ := 2
def distance_B (h : ℕ) : ℕ := (h * (h + 3)) / 2

-- Meeting point equation
def meeting_point_eqn (h : ℕ) : Prop := (distance_A h) + (distance_B h) = distance_AB

-- Requirement: y miles nearer to A whereas y = distance_AB - 2 * distance_B (since B meets closer to A by y miles)
def y_nearer_A (h : ℕ) : ℕ := distance_AB - 2 * (distance_A h)

-- Prove y = 6 for the specific value of h
theorem find_y : ∃ (h : ℕ), meeting_point_eqn h ∧ y_nearer_A h = 6 := by
  sorry

end NUMINAMATH_GPT_find_y_l2009_200959


namespace NUMINAMATH_GPT_min_value_of_f_l2009_200989

noncomputable def f (a b x : ℝ) : ℝ :=
  (a / (Real.sin x) ^ 2) + b * (Real.sin x) ^ 2

theorem min_value_of_f (a b : ℝ) (h1 : a = 2) (h2 : b = 1) (h3 : a > b) (h4 : b > 0) :
  ∃ x, f a b x = 3 := 
sorry

end NUMINAMATH_GPT_min_value_of_f_l2009_200989


namespace NUMINAMATH_GPT_f_eq_n_for_all_n_l2009_200928

noncomputable def f : ℕ → ℕ := sorry

axiom f_pos_int_valued (n : ℕ) (h : 0 < n) : f n = f n

axiom f_2_eq_2 : f 2 = 2

axiom f_mul_prop (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m * f n

axiom f_monotonic (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : m > n) : f m > f n

theorem f_eq_n_for_all_n (n : ℕ) (hn : 0 < n) : f n = n := sorry

end NUMINAMATH_GPT_f_eq_n_for_all_n_l2009_200928


namespace NUMINAMATH_GPT_fraction_to_decimal_l2009_200961

theorem fraction_to_decimal : (3 : ℝ) / 50 = 0.06 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2009_200961


namespace NUMINAMATH_GPT_average_monthly_income_P_and_R_l2009_200984

theorem average_monthly_income_P_and_R 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : P = 4000) :
  (P + R) / 2 = 5200 :=
sorry

end NUMINAMATH_GPT_average_monthly_income_P_and_R_l2009_200984


namespace NUMINAMATH_GPT_inequality_solution_l2009_200975

theorem inequality_solution (x : ℝ) : (x ≠ -2) ↔ (0 ≤ x^2 / (x + 2)^2) := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2009_200975


namespace NUMINAMATH_GPT_points_per_game_without_bonus_l2009_200933

-- Definition of the conditions
def b : ℕ := 82
def n : ℕ := 79
def P : ℕ := 15089

-- Theorem statement
theorem points_per_game_without_bonus :
  (P - b * n) / n = 109 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_points_per_game_without_bonus_l2009_200933


namespace NUMINAMATH_GPT_y_worked_days_l2009_200916

-- Definitions based on conditions
def work_rate_x := 1 / 20 -- x's work rate (W per day)
def work_rate_y := 1 / 16 -- y's work rate (W per day)

def remaining_work_by_x := 5 * work_rate_x -- Work finished by x after y left
def total_work := 1 -- Assume the total work W is 1 unit for simplicity

def days_y_worked (d : ℝ) := d * work_rate_y + remaining_work_by_x = total_work

-- The statement we need to prove
theorem y_worked_days :
  (exists d : ℕ, days_y_worked d ∧ d = 15) :=
sorry

end NUMINAMATH_GPT_y_worked_days_l2009_200916


namespace NUMINAMATH_GPT_ball_distribution_l2009_200965

theorem ball_distribution (n : ℕ) (P_white P_red P_yellow : ℚ) (num_white num_red num_yellow : ℕ) 
  (total_balls : n = 6)
  (prob_white : P_white = 1/2)
  (prob_red : P_red = 1/3)
  (prob_yellow : P_yellow = 1/6) :
  num_white = 3 ∧ num_red = 2 ∧ num_yellow = 1 := 
sorry

end NUMINAMATH_GPT_ball_distribution_l2009_200965


namespace NUMINAMATH_GPT_sneakers_sold_l2009_200950

theorem sneakers_sold (total_shoes sandals boots : ℕ) (h1 : total_shoes = 17) (h2 : sandals = 4) (h3 : boots = 11) :
  total_shoes - (sandals + boots) = 2 :=
by
  -- proof steps will be included here
  sorry

end NUMINAMATH_GPT_sneakers_sold_l2009_200950


namespace NUMINAMATH_GPT_jorge_total_spent_l2009_200927

-- Definitions based on the problem conditions
def price_adult_ticket : ℝ := 10
def price_child_ticket : ℝ := 5
def num_adult_tickets : ℕ := 12
def num_child_tickets : ℕ := 12
def discount_adult : ℝ := 0.40
def discount_child : ℝ := 0.30
def extra_discount : ℝ := 0.10

-- The desired statement to prove
theorem jorge_total_spent :
  let total_adult_cost := num_adult_tickets * price_adult_ticket
  let total_child_cost := num_child_tickets * price_child_ticket
  let discounted_adult := total_adult_cost * (1 - discount_adult)
  let discounted_child := total_child_cost * (1 - discount_child)
  let total_cost_before_extra_discount := discounted_adult + discounted_child
  let final_cost := total_cost_before_extra_discount * (1 - extra_discount)
  final_cost = 102.60 :=
by 
  sorry

end NUMINAMATH_GPT_jorge_total_spent_l2009_200927


namespace NUMINAMATH_GPT_quadratic_equation_solutions_l2009_200914

theorem quadratic_equation_solutions : ∀ x : ℝ, x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := 
by sorry

end NUMINAMATH_GPT_quadratic_equation_solutions_l2009_200914


namespace NUMINAMATH_GPT_max_fruit_to_teacher_l2009_200991

theorem max_fruit_to_teacher (A G : ℕ) : (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_max_fruit_to_teacher_l2009_200991


namespace NUMINAMATH_GPT_B_is_left_of_A_l2009_200925

-- Define the coordinates of points A and B
def A_coord : ℚ := 5 / 8
def B_coord : ℚ := 8 / 13

-- The statement we want to prove: B is to the left of A
theorem B_is_left_of_A : B_coord < A_coord :=
  by {
    sorry
  }

end NUMINAMATH_GPT_B_is_left_of_A_l2009_200925


namespace NUMINAMATH_GPT_youngest_is_dan_l2009_200958

notation "alice" => 21
notation "bob" => 18
notation "clare" => 22
notation "dan" => 16
notation "eve" => 28

theorem youngest_is_dan :
  let a := alice
  let b := bob
  let c := clare
  let d := dan
  let e := eve
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105 →
  min (min (min (min a b) c) d) e = d :=
by {
  sorry
}

end NUMINAMATH_GPT_youngest_is_dan_l2009_200958


namespace NUMINAMATH_GPT_negation_of_proposition_l2009_200941

theorem negation_of_proposition (m : ℤ) : 
  (¬ (∃ x : ℤ, x^2 + 2*x + m ≤ 0)) ↔ ∀ x : ℤ, x^2 + 2*x + m > 0 :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l2009_200941


namespace NUMINAMATH_GPT_clarence_initial_oranges_l2009_200922

variable (initial_oranges : ℕ)
variable (obtained_from_joyce : ℕ := 3)
variable (total_oranges : ℕ := 8)

theorem clarence_initial_oranges (initial_oranges : ℕ) :
  initial_oranges + obtained_from_joyce = total_oranges → initial_oranges = 5 :=
by
  sorry

end NUMINAMATH_GPT_clarence_initial_oranges_l2009_200922


namespace NUMINAMATH_GPT_James_total_water_capacity_l2009_200960

theorem James_total_water_capacity : 
  let cask_capacity := 20 -- capacity of a cask in gallons
  let barrel_capacity := 2 * cask_capacity + 3 -- capacity of a barrel in gallons
  let total_capacity := 4 * barrel_capacity + cask_capacity -- total water storage capacity
  total_capacity = 192 := by
    let cask_capacity := 20
    let barrel_capacity := 2 * cask_capacity + 3
    let total_capacity := 4 * barrel_capacity + cask_capacity
    have h : total_capacity = 192 := by sorry
    exact h

end NUMINAMATH_GPT_James_total_water_capacity_l2009_200960


namespace NUMINAMATH_GPT_empty_set_condition_l2009_200902

def isEmptySet (s : Set ℝ) : Prop := s = ∅

def A : Set ℕ := {n : ℕ | n^2 ≤ 0}
def B : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def C : Set ℝ := {x : ℝ | x^2 + x + 1 = 0}
def D : Set ℝ := {0}

theorem empty_set_condition : isEmptySet C := by
  sorry

end NUMINAMATH_GPT_empty_set_condition_l2009_200902


namespace NUMINAMATH_GPT_planks_ratio_l2009_200952

theorem planks_ratio (P S : ℕ) (H : S + 100 + 20 + 30 = 200) (T : P = 200) (R : S = 200 / 2) : 
(S : ℚ) / P = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_planks_ratio_l2009_200952


namespace NUMINAMATH_GPT_find_solutions_l2009_200935

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 9 * x ^ 2 + 6

theorem find_solutions :
  ∃ x1 x2 x3 : ℝ, f x1 = Real.sqrt 2 ∧ f x2 = Real.sqrt 2 ∧ f x3 = Real.sqrt 2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
sorry

end NUMINAMATH_GPT_find_solutions_l2009_200935


namespace NUMINAMATH_GPT_sub_decimal_proof_l2009_200949

theorem sub_decimal_proof : 2.5 - 0.32 = 2.18 :=
  by sorry

end NUMINAMATH_GPT_sub_decimal_proof_l2009_200949


namespace NUMINAMATH_GPT_greatest_k_l2009_200923

noncomputable def n : ℕ := sorry
def k : ℕ := sorry

axiom d : ℕ → ℕ

axiom h1 : d n = 72
axiom h2 : d (5 * n) = 90

theorem greatest_k : ∃ k : ℕ, (∀ m : ℕ, m > k → ¬(5^m ∣ n)) ∧ 5^k ∣ n ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_greatest_k_l2009_200923


namespace NUMINAMATH_GPT_carrots_total_l2009_200986
-- import the necessary library

-- define the conditions as given
def sandy_carrots : Nat := 6
def sam_carrots : Nat := 3

-- state the problem as a theorem to be proven
theorem carrots_total : sandy_carrots + sam_carrots = 9 := by
  sorry

end NUMINAMATH_GPT_carrots_total_l2009_200986


namespace NUMINAMATH_GPT_union_of_A_and_B_l2009_200990

def A := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 3)}
def B := {y : ℝ | ∃ (x : ℝ), y = Real.exp x}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 0} := by
sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2009_200990


namespace NUMINAMATH_GPT_binomial_np_sum_l2009_200948

-- Definitions of variance and expectation for a binomial distribution
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)
def binomial_expectation (n : ℕ) (p : ℚ) : ℚ := n * p

-- Statement of the problem
theorem binomial_np_sum (n : ℕ) (p : ℚ) (h_var : binomial_variance n p = 4) (h_exp : binomial_expectation n p = 12) :
    n + p = 56 / 3 := by
  sorry

end NUMINAMATH_GPT_binomial_np_sum_l2009_200948


namespace NUMINAMATH_GPT_shampoo_duration_l2009_200999

-- Conditions
def rose_shampoo : ℚ := 1/3
def jasmine_shampoo : ℚ := 1/4
def daily_usage : ℚ := 1/12

-- Question
theorem shampoo_duration : (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := by
  sorry

end NUMINAMATH_GPT_shampoo_duration_l2009_200999


namespace NUMINAMATH_GPT_chords_intersecting_theorem_l2009_200947

noncomputable def intersecting_chords_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) : ℝ :=
  sorry

theorem chords_intersecting_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) :
  (P - A) * (P - B) = (P - C) * (P - D) :=
by sorry

end NUMINAMATH_GPT_chords_intersecting_theorem_l2009_200947


namespace NUMINAMATH_GPT_total_bottles_in_box_l2009_200946

def dozens (n : ℕ) := 12 * n

def water_bottles : ℕ := dozens 2

def apple_bottles : ℕ := water_bottles + 6

def total_bottles : ℕ := water_bottles + apple_bottles

theorem total_bottles_in_box : total_bottles = 54 := 
by
  sorry

end NUMINAMATH_GPT_total_bottles_in_box_l2009_200946


namespace NUMINAMATH_GPT_convert_to_polar_l2009_200967

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r, θ)

theorem convert_to_polar (x y : ℝ) (hx : x = 8) (hy : y = 3 * Real.sqrt 3) :
  polar_coordinates x y = (Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  rw [hx, hy]
  simp [polar_coordinates]
  -- place to handle conversions and simplifications if necessary
  sorry

end NUMINAMATH_GPT_convert_to_polar_l2009_200967


namespace NUMINAMATH_GPT_sampling_methods_correct_l2009_200964

def company_sales_outlets (A B C D : ℕ) : Prop :=
  A = 150 ∧ B = 120 ∧ C = 180 ∧ D = 150 ∧ A + B + C + D = 600

def investigation_samples (total_samples large_outlets region_C_sample : ℕ) : Prop :=
  total_samples = 100 ∧ large_outlets = 20 ∧ region_C_sample = 7

def appropriate_sampling_methods (investigation1_method investigation2_method : String) : Prop :=
  investigation1_method = "Stratified sampling" ∧ investigation2_method = "Simple random sampling"

theorem sampling_methods_correct :
  company_sales_outlets 150 120 180 150 →
  investigation_samples 100 20 7 →
  appropriate_sampling_methods "Stratified sampling" "Simple random sampling" :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sampling_methods_correct_l2009_200964


namespace NUMINAMATH_GPT_subject_difference_l2009_200944

-- Define the problem in terms of conditions and question
theorem subject_difference (C R M : ℕ) (hC : C = 10) (hR : R = C + 4) (hM : M + R + C = 41) : M - R = 3 :=
by
  -- Lean expects a proof here, we skip it with sorry
  sorry

end NUMINAMATH_GPT_subject_difference_l2009_200944


namespace NUMINAMATH_GPT_percentage_exceeds_self_l2009_200924

theorem percentage_exceeds_self (N : ℝ) (P : ℝ) (hN : N = 75) (h_condition : N = (P / 100) * N + 63) : P = 16 := by
  sorry

end NUMINAMATH_GPT_percentage_exceeds_self_l2009_200924


namespace NUMINAMATH_GPT_proof_problem_l2009_200979

noncomputable def M : Set ℝ := { x | x ≥ 2 }
noncomputable def a : ℝ := Real.pi

theorem proof_problem : a ∈ M ∧ {a} ⊂ M :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2009_200979


namespace NUMINAMATH_GPT_rectangle_area_l2009_200962

-- Define the length and width of the rectangle based on given ratio
def length (k: ℝ) := 5 * k
def width (k: ℝ) := 2 * k

-- The perimeter condition
def perimeter (k: ℝ) := 2 * (length k) + 2 * (width k) = 280

-- The diagonal condition
def diagonal_condition (k: ℝ) := (width k) * Real.sqrt 2 = (length k) / 2

-- The area of the rectangle
def area (k: ℝ) := (length k) * (width k)

-- The main theorem to be proven
theorem rectangle_area : ∃ k: ℝ, perimeter k ∧ diagonal_condition k ∧ area k = 4000 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2009_200962


namespace NUMINAMATH_GPT_find_multiple_of_brothers_l2009_200970

theorem find_multiple_of_brothers : 
  ∃ x : ℕ, (x * 4) - 2 = 6 :=
by
  -- Provide the correct Lean statement for the problem
  sorry

end NUMINAMATH_GPT_find_multiple_of_brothers_l2009_200970


namespace NUMINAMATH_GPT_contrapositive_l2009_200915

theorem contrapositive (a : ℝ) : (a > 0 → a > 1) → (a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_GPT_contrapositive_l2009_200915


namespace NUMINAMATH_GPT_sum_of_squares_not_7_mod_8_l2009_200942

theorem sum_of_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_not_7_mod_8_l2009_200942


namespace NUMINAMATH_GPT_remaining_painting_time_l2009_200900

-- Define the given conditions as Lean definitions
def total_rooms : ℕ := 9
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 5

-- Formulate the main theorem to prove the remaining time is 32 hours
theorem remaining_painting_time : 
  (total_rooms - rooms_painted) * hours_per_room = 32 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_painting_time_l2009_200900


namespace NUMINAMATH_GPT_total_number_of_lives_l2009_200906

theorem total_number_of_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
                              (h1 : initial_players = 7) (h2 : additional_players = 2) (h3 : lives_per_player = 7) : 
                              initial_players + additional_players * lives_per_player = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_lives_l2009_200906


namespace NUMINAMATH_GPT_archipelago_max_value_l2009_200911

noncomputable def archipelago_max_islands (N : ℕ) : Prop :=
  N ≥ 7 ∧ 
  (∀ (a b : ℕ), a ≠ b → a ≤ N → b ≤ N → ∃ c : ℕ, c ≤ N ∧ (∃ d, d ≠ c ∧ d ≤ N → d ≠ a ∧ d ≠ b)) ∧ 
  (∀ (a : ℕ), a ≤ N → ∃ b, b ≠ a ∧ b ≤ N ∧ (∃ c, c ≤ N ∧ c ≠ b ∧ c ≠ a))

theorem archipelago_max_value : archipelago_max_islands 36 := sorry

end NUMINAMATH_GPT_archipelago_max_value_l2009_200911


namespace NUMINAMATH_GPT_find_five_value_l2009_200987

def f (x : ℝ) : ℝ := x^2 - x

theorem find_five_value : f 5 = 20 := by
  sorry

end NUMINAMATH_GPT_find_five_value_l2009_200987


namespace NUMINAMATH_GPT_percentage_error_in_calculated_area_l2009_200919

theorem percentage_error_in_calculated_area 
  (s : ℝ) 
  (measured_side : ℝ) 
  (h : measured_side = s * 1.04) :
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 8.16 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_in_calculated_area_l2009_200919


namespace NUMINAMATH_GPT_gift_box_spinning_tops_l2009_200957

theorem gift_box_spinning_tops
  (red_box_cost : ℕ) (red_box_tops : ℕ)
  (yellow_box_cost : ℕ) (yellow_box_tops : ℕ)
  (total_spent : ℕ) (total_boxes : ℕ)
  (h_red_box_cost : red_box_cost = 5)
  (h_red_box_tops : red_box_tops = 3)
  (h_yellow_box_cost : yellow_box_cost = 9)
  (h_yellow_box_tops : yellow_box_tops = 5)
  (h_total_spent : total_spent = 600)
  (h_total_boxes : total_boxes = 72) :
  ∃ (red_boxes : ℕ) (yellow_boxes : ℕ), (red_boxes + yellow_boxes = total_boxes) ∧
  (red_box_cost * red_boxes + yellow_box_cost * yellow_boxes = total_spent) ∧
  (red_box_tops * red_boxes + yellow_box_tops * yellow_boxes = 336) :=
by
  sorry

end NUMINAMATH_GPT_gift_box_spinning_tops_l2009_200957


namespace NUMINAMATH_GPT_sum_inequality_l2009_200905

theorem sum_inequality (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2) :
  (x + y + z) * (x⁻¹ + y⁻¹ + z⁻¹) ≥ 6 * (x / (y + z) + y / (z + x) + z / (x + y)) := sorry

end NUMINAMATH_GPT_sum_inequality_l2009_200905


namespace NUMINAMATH_GPT_exists_g_l2009_200956

variable {R : Type} [Field R]

-- Define the function f with the given condition
def f (x y : R) : R := sorry

-- The main theorem to prove the existence of g
theorem exists_g (f_condition: ∀ x y z : R, f x y + f y z + f z x = 0) : ∃ g : R → R, ∀ x y : R, f x y = g x - g y := 
by 
  sorry

end NUMINAMATH_GPT_exists_g_l2009_200956


namespace NUMINAMATH_GPT_perimeter_of_larger_triangle_is_65_l2009_200929

noncomputable def similar_triangle_perimeter : ℝ :=
  let a := 7
  let b := 7
  let c := 12
  let longest_side_similar := 30
  let perimeter_small := a + b + c
  let ratio := longest_side_similar / c
  ratio * perimeter_small

theorem perimeter_of_larger_triangle_is_65 :
  similar_triangle_perimeter = 65 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_larger_triangle_is_65_l2009_200929


namespace NUMINAMATH_GPT_Kyle_is_25_l2009_200953

-- Definitions based on the conditions
def Tyson_age : Nat := 20
def Frederick_age : Nat := 2 * Tyson_age
def Julian_age : Nat := Frederick_age - 20
def Kyle_age : Nat := Julian_age + 5

-- The theorem to prove
theorem Kyle_is_25 : Kyle_age = 25 := by
  sorry

end NUMINAMATH_GPT_Kyle_is_25_l2009_200953


namespace NUMINAMATH_GPT_part1_part2_l2009_200908

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

-- (1) Given a = -1, prove that the inequality f(x, -1) ≤ 0 implies x ≤ -1/3
theorem part1 (x : ℝ) : (f x (-1) ≤ 0) ↔ (x ≤ -1/3) :=
by
  sorry

-- (2) Given f(x) ≥ 0 for all x ≥ -1, prove that the range for a is a ≤ -3 or a ≥ 1
theorem part2 (a : ℝ) : (∀ x, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2009_200908


namespace NUMINAMATH_GPT_evaluate_expression_l2009_200913

theorem evaluate_expression : (-1:ℤ)^2022 + |(-2:ℤ)| - (1/2 : ℚ)^0 - 2 * Real.tan (Real.pi / 4) = 0 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2009_200913


namespace NUMINAMATH_GPT_cos_difference_identity_cos_phi_value_l2009_200954

variables (α β θ φ : ℝ)
variables (a b : ℝ × ℝ)

-- Part I
theorem cos_difference_identity (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) (hβ : 0 ≤ β ∧ β ≤ 2 * Real.pi) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
sorry

-- Part II
theorem cos_phi_value (hθ : 0 < θ ∧ θ < Real.pi / 2) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (ha : a = (Real.sin θ, -2)) (hb : b = (1, Real.cos θ)) (dot_ab_zero : a.1 * b.1 + a.2 * b.2 = 0)
  (h_sin_diff : Real.sin (theta - phi) = Real.sqrt 10 / 10) :
  Real.cos φ = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_cos_difference_identity_cos_phi_value_l2009_200954


namespace NUMINAMATH_GPT_first_number_is_45_l2009_200937

theorem first_number_is_45 (a b : ℕ) (h1 : a / gcd a b = 3) (h2 : b / gcd a b = 4) (h3 : lcm a b = 180) : a = 45 := by
  sorry

end NUMINAMATH_GPT_first_number_is_45_l2009_200937


namespace NUMINAMATH_GPT_hexagon_division_ratio_l2009_200921

theorem hexagon_division_ratio
  (hex_area : ℝ)
  (hexagon : ∀ (A B C D E F : ℝ), hex_area = 8)
  (line_PQ_splits : ∀ (above_area below_area : ℝ), above_area = 4 ∧ below_area = 4)
  (below_PQ : ℝ)
  (unit_square_area : ∀ (unit_square : ℝ), unit_square = 1)
  (triangle_base : ℝ)
  (triangle_height : ℝ)
  (triangle_area : ∀ (base height : ℝ), triangle_base = 4 ∧ (base * height) / 2 = 3)
  (XQ QY : ℝ)
  (bases_sum : ∀ (XQ QY : ℝ), XQ + QY = 4) :
  XQ / QY = 2 / 3 :=
sorry

end NUMINAMATH_GPT_hexagon_division_ratio_l2009_200921


namespace NUMINAMATH_GPT_ten_numbers_exists_l2009_200968

theorem ten_numbers_exists :
  ∃ (a : Fin 10 → ℕ), 
    (∀ i j : Fin 10, i ≠ j → ¬ (a i ∣ a j))
    ∧ (∀ i j : Fin 10, i ≠ j → a i ^ 2 ∣ a j * a j) :=
sorry

end NUMINAMATH_GPT_ten_numbers_exists_l2009_200968


namespace NUMINAMATH_GPT_first_month_sale_l2009_200995

theorem first_month_sale 
(sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
(avg_sale : ℕ) 
(h_avg: avg_sale = 6500)
(h_sale2: sale_2 = 6927)
(h_sale3: sale_3 = 6855)
(h_sale4: sale_4 = 7230)
(h_sale5: sale_5 = 6562)
(h_sale6: sale_6 = 4791)
: sale_1 = 6635 := by
  sorry

end NUMINAMATH_GPT_first_month_sale_l2009_200995


namespace NUMINAMATH_GPT_two_digit_number_formed_l2009_200985

theorem two_digit_number_formed (A B C D E F : ℕ) 
  (A_C_D_const : A + C + D = constant)
  (A_B_const : A + B = constant)
  (B_D_F_const : B + D + F = constant)
  (E_F_const : E + F = constant)
  (E_B_C_const : E + B + C = constant)
  (B_eq_C_D : B = C + D)
  (B_D_eq_E : B + D = E)
  (E_C_eq_A : E + C = A) 
  (hA : A = 6) 
  (hB : B = 3)
  : 10 * A + B = 63 :=
by sorry

end NUMINAMATH_GPT_two_digit_number_formed_l2009_200985
