import Mathlib

namespace trig_identity_l1654_165490

variable {α : ℝ}

theorem trig_identity (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end trig_identity_l1654_165490


namespace angles_in_interval_l1654_165483

-- Define the main statement we need to prove
theorem angles_in_interval (theta : ℝ) (h1 : 0 ≤ theta) (h2 : theta ≤ 2 * Real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos theta - x * (1 - x) + (1-x)^2 * Real.sin theta < 0) →
  (Real.pi / 2 < theta ∧ theta < 3 * Real.pi / 2) :=
by
  sorry

end angles_in_interval_l1654_165483


namespace geometric_sequence_a_10_l1654_165429

noncomputable def geometric_sequence := ℕ → ℝ

def a_3 (a r : ℝ) := a * r^2 = 3
def a_5_equals_8a_7 (a r : ℝ) := a * r^4 = 8 * a * r^6

theorem geometric_sequence_a_10 (a r : ℝ) (seq : geometric_sequence) (h₁ : a_3 a r) (h₂ : a_5_equals_8a_7 a r) :
  seq 10 = a * r^9 := by
  sorry

end geometric_sequence_a_10_l1654_165429


namespace kanul_total_amount_l1654_165492

theorem kanul_total_amount (T : ℝ) (h1 : 500 + 400 + 0.10 * T = T) : T = 1000 :=
  sorry

end kanul_total_amount_l1654_165492


namespace compare_negatives_l1654_165459

noncomputable def isNegative (x : ℝ) : Prop := x < 0
noncomputable def absValue (x : ℝ) : ℝ := if x < 0 then -x else x
noncomputable def sqrt14 : ℝ := Real.sqrt 14

theorem compare_negatives : -4 < -Real.sqrt 14 := by
  have h1: Real.sqrt 16 = 4 := by
    sorry
  
  have h2: absValue (-4) = 4 := by
    sorry

  have h3: absValue (-(sqrt14)) = sqrt14 := by
    sorry

  have h4: Real.sqrt 16 > Real.sqrt 14 := by
    sorry

  show -4 < -Real.sqrt 14
  sorry

end compare_negatives_l1654_165459


namespace ordered_notebooks_amount_l1654_165417

def initial_notebooks : ℕ := 10
def ordered_notebooks (x : ℕ) : ℕ := x
def lost_notebooks : ℕ := 2
def current_notebooks : ℕ := 14

theorem ordered_notebooks_amount (x : ℕ) (h : initial_notebooks + ordered_notebooks x - lost_notebooks = current_notebooks) : x = 6 :=
by
  sorry

end ordered_notebooks_amount_l1654_165417


namespace point_in_third_quadrant_l1654_165401

def quadrant_of_point (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "first"
  else if x < 0 ∧ y > 0 then "second"
  else if x < 0 ∧ y < 0 then "third"
  else if x > 0 ∧ y < 0 then "fourth"
  else "on_axis"

theorem point_in_third_quadrant : quadrant_of_point (-2) (-3) = "third" :=
  by sorry

end point_in_third_quadrant_l1654_165401


namespace no_nonconstant_arithmetic_progression_l1654_165467

theorem no_nonconstant_arithmetic_progression (x : ℝ) :
  2 * (2 : ℝ)^(x^2) ≠ (2 : ℝ)^x + (2 : ℝ)^(x^3) :=
sorry

end no_nonconstant_arithmetic_progression_l1654_165467


namespace altitude_inequality_not_universally_true_l1654_165440

noncomputable def altitudes (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a m_b m_c : ℝ, m_a ≤ m_b ∧ m_b ≤ m_c 

noncomputable def seg_to_orthocenter (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a_star m_b_star m_c_star : ℝ, True

theorem altitude_inequality (a b c m_a m_b m_c : ℝ) 
  (h₀ : a ≥ b) (h₁ : b ≥ c) (h₂ : m_a ≤ m_b) (h₃ : m_b ≤ m_c) :
  (a + m_a ≥ b + m_b) ∧ (b + m_b ≥ c + m_c) :=
by
  sorry

theorem not_universally_true (a b c m_a_star m_b_star m_c_star : ℝ)
  (h₀ : a ≥ b) (h₁ : b ≥ c) :
  ¬(a + m_a_star ≥ b + m_b_star ∧ b + m_b_star ≥ c + m_c_star) :=
by
  sorry

end altitude_inequality_not_universally_true_l1654_165440


namespace molecular_weight_of_oxygen_part_l1654_165448

-- Define the known variables as constants
def atomic_weight_oxygen : ℝ := 16.00
def num_oxygen_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 88.00

-- Define the problem as a theorem
theorem molecular_weight_of_oxygen_part :
  16.00 * 2 = 32.00 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_of_oxygen_part_l1654_165448


namespace gcd_poly_l1654_165449

theorem gcd_poly (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) : 
  Int.gcd (4 * b ^ 2 + 63 * b + 144) (2 * b + 7) = 1 := 
by 
  sorry

end gcd_poly_l1654_165449


namespace cub_eqn_root_sum_l1654_165487

noncomputable def cos_x := Real.cos (Real.pi / 5)

theorem cub_eqn_root_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
(h3 : a * cos_x ^ 3 - b * cos_x - 1 = 0) : a + b = 12 :=
sorry

end cub_eqn_root_sum_l1654_165487


namespace not_divisible_by_121_l1654_165485

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 2014)) :=
sorry

end not_divisible_by_121_l1654_165485


namespace ellipse_range_of_k_l1654_165491

theorem ellipse_range_of_k (k : ℝ) :
  (∃ (eq : ((x y : ℝ) → (x ^ 2 / (3 + k) + y ^ 2 / (2 - k) = 1))),
  ((3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k))) ↔
  (k ∈ Set.Ioo (-3 : ℝ) ((-1) / 2) ∪ Set.Ioo ((-1) / 2) 2) :=
by sorry

end ellipse_range_of_k_l1654_165491


namespace range_of_k_l1654_165446

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + k^2 - 1 ≤ 0) ↔ (-Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2) :=
by 
  sorry

end range_of_k_l1654_165446


namespace problem_statement_l1654_165431

theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, (1/2 < x ∧ x < 2 → ax^2 + 5 * x - 2 > 0)) →
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) → ax^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end problem_statement_l1654_165431


namespace selling_price_before_brokerage_l1654_165411

variables (CR BR SP : ℝ)
variables (hCR : CR = 120.50) (hBR : BR = 1 / 400)

theorem selling_price_before_brokerage :
  SP = (CR * 400) / (399) := 
by
  sorry

end selling_price_before_brokerage_l1654_165411


namespace range_of_a_l1654_165423

noncomputable def satisfies_condition (a : ℝ) : Prop :=
∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs ((1 / 2) * x^3 - a * x) ≤ 1

theorem range_of_a :
  {a : ℝ | satisfies_condition a} = {a : ℝ | - (1 / 2) ≤ a ∧ a ≤ (3 / 2)} :=
by
  sorry

end range_of_a_l1654_165423


namespace sum_of_values_l1654_165451

theorem sum_of_values (N : ℝ) (R : ℝ) (hN : N ≠ 0) (h_eq : N - 3 / N = R) :
  let N1 := (-R + Real.sqrt (R^2 + 12)) / 2
  let N2 := (-R - Real.sqrt (R^2 + 12)) / 2
  N1 + N2 = R :=
by
  sorry

end sum_of_values_l1654_165451


namespace age_ratio_l1654_165489
open Nat

theorem age_ratio (B_c : ℕ) (h1 : B_c = 42) (h2 : ∀ A_c, A_c = B_c + 12) : (A_c + 10) / (B_c - 10) = 2 :=
by
  sorry

end age_ratio_l1654_165489


namespace find_x_l1654_165466

theorem find_x
  (x : ℤ)
  (h : 3 * x + 3 * 15 + 3 * 18 + 11 = 152) :
  x = 14 :=
by
  sorry

end find_x_l1654_165466


namespace eggs_leftover_l1654_165474

theorem eggs_leftover :
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  total_eggs % 10 = 0 := by
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  exact Nat.mod_eq_zero_of_dvd (show 10 ∣ total_eggs from by norm_num)

end eggs_leftover_l1654_165474


namespace range_of_a_l1654_165486

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l1654_165486


namespace max_quarters_l1654_165430

-- Definitions stating the conditions
def total_money_in_dollars : ℝ := 4.80
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10

-- Theorem statement
theorem max_quarters (q : ℕ) (h1 : total_money_in_dollars = (q * value_of_quarter) + (2 * q * value_of_dime)) : q ≤ 10 :=
by {
  -- Injecting a placeholder to facilitate proof development
  sorry
}

end max_quarters_l1654_165430


namespace number_of_hens_l1654_165419

theorem number_of_hens (H C : Nat) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := 
by
  sorry

end number_of_hens_l1654_165419


namespace range_of_a_l1654_165472

theorem range_of_a (a : ℝ) (an bn : ℕ → ℝ)
  (h_an : ∀ n, an n = (-1) ^ (n + 2013) * a)
  (h_bn : ∀ n, bn n = 2 + (-1) ^ (n + 2014) / n)
  (h_condition : ∀ n : ℕ, 1 ≤ n → an n < bn n) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l1654_165472


namespace vertex_h_is_3_l1654_165478

open Real

theorem vertex_h_is_3 (a b c : ℝ) (h : ℝ)
    (h_cond : 3 * (a * 3^2 + b * 3 + c) + 6 = 3) : 
    4 * (a * x^2 + b * x + c) = 12 * (x - 3)^2 + 24 → 
    h = 3 := 
by 
sorry

end vertex_h_is_3_l1654_165478


namespace find_sum_principal_l1654_165434

theorem find_sum_principal (P R : ℝ) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 → P = 300 :=
by
  sorry

end find_sum_principal_l1654_165434


namespace max_possible_value_l1654_165499

theorem max_possible_value (a b : ℝ) (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n) :
  ∃ a b, ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n → ∃ s : ℝ, (s = 0 ∨ s = 1 ∨ s = 2) →
  max (1 / a^(2009) + 1 / b^(2009)) = 2 :=
sorry

end max_possible_value_l1654_165499


namespace polar_to_cartesian_l1654_165409

-- Definitions for the polar coordinates conversion
noncomputable def polar_to_cartesian_eq (C : ℝ → ℝ → Prop) :=
  ∀ (ρ θ : ℝ), (ρ^2 * (1 + 3 * (Real.sin θ)^2) = 4) → C (ρ * (Real.cos θ)) (ρ * (Real.sin θ))

-- Define the Cartesian equation
def cartesian_eq (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 = 1)

-- The main theorem
theorem polar_to_cartesian 
  (C : ℝ → ℝ → Prop)
  (h : polar_to_cartesian_eq C) :
  ∀ x y : ℝ, C x y ↔ cartesian_eq x y :=
by
  sorry

end polar_to_cartesian_l1654_165409


namespace b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l1654_165498

-- Define the sequences a_n, b_n, and c_n along with their properties

-- Definitions
def a_seq (n : ℕ) : ℕ := sorry            -- Define a_n

def S_seq (n : ℕ) : ℕ := sorry            -- Define S_n

def b_seq (n : ℕ) : ℕ := a_seq (n+1) - 2 * a_seq n

def c_seq (n : ℕ) : ℕ := a_seq n / 2^n

-- Conditions
axiom S_n_condition (n : ℕ) : S_seq (n+1) = 4 * a_seq n + 2
axiom a_1_condition : a_seq 1 = 1

-- Goals
theorem b_seq_formula (n : ℕ) : b_seq n = 3 * 2^(n-1) := sorry

theorem c_seq_arithmetic (n : ℕ) : c_seq (n+1) - c_seq n = 3 / 4 := sorry

theorem c_seq_formula (n : ℕ) : c_seq n = (3 * n - 1) / 4 := sorry

theorem a_seq_formula (n : ℕ) : a_seq n = (3 * n - 1) * 2^(n-2) := sorry

theorem sum_S_5 : S_seq 5 = 178 := sorry

end b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l1654_165498


namespace value_of_k_l1654_165484

theorem value_of_k (k x : ℕ) (h1 : 2^x - 2^(x - 2) = k * 2^10) (h2 : x = 12) : k = 3 := by
  sorry

end value_of_k_l1654_165484


namespace range_of_m_l1654_165450

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end range_of_m_l1654_165450


namespace factorize_expression_l1654_165402

theorem factorize_expression (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) :=
by
  sorry

end factorize_expression_l1654_165402


namespace sum_of_longest_altitudes_l1654_165444

theorem sum_of_longest_altitudes (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) :
  a + b = 14 :=
by
  sorry

end sum_of_longest_altitudes_l1654_165444


namespace simplify_expression_l1654_165438

theorem simplify_expression (x : ℝ) (h : x = 9) : 
  ((x^9 - 27 * x^6 + 729) / (x^6 - 27) = 730 + 1 / 26) :=
by {
 sorry
}

end simplify_expression_l1654_165438


namespace scientific_notation_population_l1654_165433

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l1654_165433


namespace find_multiple_l1654_165400

/-- 
Given:
1. Hank Aaron hit 755 home runs.
2. Dave Winfield hit 465 home runs.
3. Hank Aaron has 175 fewer home runs than a certain multiple of the number that Dave Winfield has.

Prove:
The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2.
-/
def multiple_of_dave_hr (ha_hr dw_hr diff : ℕ) (m : ℕ) : Prop :=
  ha_hr + diff = m * dw_hr

theorem find_multiple :
  multiple_of_dave_hr 755 465 175 2 :=
by
  sorry

end find_multiple_l1654_165400


namespace percentage_william_land_l1654_165497

-- Definitions of the given conditions
def total_tax_collected : ℝ := 3840
def william_tax : ℝ := 480

-- Proof statement
theorem percentage_william_land :
  ((william_tax / total_tax_collected) * 100) = 12.5 :=
by
  sorry

end percentage_william_land_l1654_165497


namespace coin_value_is_630_l1654_165426

theorem coin_value_is_630 :
  (∃ x : ℤ, x > 0 ∧ 406 * x = 63000) :=
by {
  sorry
}

end coin_value_is_630_l1654_165426


namespace find_coordinates_of_P_l1654_165470

-- Define the points
def P1 : ℝ × ℝ := (2, -1)
def P2 : ℝ × ℝ := (0, 5)

-- Define the point P
def P : ℝ × ℝ := (-2, 11)

-- Conditions encoded as vector relationships
def vector_P1_P (p : ℝ × ℝ) := (p.1 - P1.1, p.2 - P1.2)
def vector_PP2 (p : ℝ × ℝ) := (P2.1 - p.1, P2.2 - p.2)

-- The hypothesis that | P1P | = 2 * | PP2 |
axiom vector_relation : ∀ (p : ℝ × ℝ), 
  vector_P1_P p = (-2 * (vector_PP2 p).1, -2 * (vector_PP2 p).2) → p = P

theorem find_coordinates_of_P : P = (-2, 11) :=
by
  sorry

end find_coordinates_of_P_l1654_165470


namespace base6_to_decimal_l1654_165462

theorem base6_to_decimal (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end base6_to_decimal_l1654_165462


namespace log_sin_decrease_interval_l1654_165494

open Real

noncomputable def interval_of_decrease (x : ℝ) : Prop :=
  ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8)

theorem log_sin_decrease_interval (x : ℝ) :
  interval_of_decrease x ↔ ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8) :=
by
  sorry

end log_sin_decrease_interval_l1654_165494


namespace problem_solution_l1654_165461

theorem problem_solution :
  (12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545) :=
by
  sorry

end problem_solution_l1654_165461


namespace jerry_earnings_per_task_l1654_165481

theorem jerry_earnings_per_task :
  ∀ (task_hours : ℕ) (daily_hours : ℕ) (days_per_week : ℕ) (total_earnings : ℕ),
    task_hours = 2 →
    daily_hours = 10 →
    days_per_week = 5 →
    total_earnings = 1400 →
    total_earnings / ((daily_hours / task_hours) * days_per_week) = 56 :=
by
  intros task_hours daily_hours days_per_week total_earnings
  intros h_task_hours h_daily_hours h_days_per_week h_total_earnings
  sorry

end jerry_earnings_per_task_l1654_165481


namespace scientific_notation_of_number_l1654_165460

theorem scientific_notation_of_number : 15300000000 = 1.53 * (10 : ℝ)^10 := sorry

end scientific_notation_of_number_l1654_165460


namespace eight_disks_area_sum_final_result_l1654_165452

theorem eight_disks_area_sum (r : ℝ) (C : ℝ) :
  C = 1 ∧ r = (Real.sqrt 2 + 1) / 2 → 
  8 * (π * (r ^ 2)) = 2 * π * (3 + 2 * Real.sqrt 2) :=
by
  intros h
  sorry

theorem final_result :
  let a := 6
  let b := 4
  let c := 2
  a + b + c = 12 :=
by
  intros
  norm_num

end eight_disks_area_sum_final_result_l1654_165452


namespace optimal_addition_amount_l1654_165469

theorem optimal_addition_amount (a b g : ℝ) (h₁ : a = 628) (h₂ : b = 774) (h₃ : g = 718) : 
    b + a - g = 684 :=
by
  sorry

end optimal_addition_amount_l1654_165469


namespace average_rate_of_change_l1654_165475

noncomputable def f (x : ℝ) : ℝ :=
  -2 * x^2 + 1

theorem average_rate_of_change : 
  ((f 1 - f 0) / (1 - 0)) = -2 :=
by
  sorry

end average_rate_of_change_l1654_165475


namespace servant_cash_received_l1654_165482

theorem servant_cash_received (salary_cash : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ)
  (h_salary_cash : salary_cash = 90) (h_turban_value : turban_value = 70) (h_months_worked : months_worked = 9)
  (h_total_months : total_months = 12) : 
  salary_cash * months_worked / total_months + (turban_value * months_worked / total_months) - turban_value = 50 := by
sorry

end servant_cash_received_l1654_165482


namespace ball_redistribution_impossible_l1654_165443

noncomputable def white_boxes_initial_ball_count := 31
noncomputable def black_boxes_initial_ball_count := 26
noncomputable def white_boxes_new_ball_count := 21
noncomputable def black_boxes_new_ball_count := 16
noncomputable def white_boxes_target_ball_count := 15
noncomputable def black_boxes_target_ball_count := 10

theorem ball_redistribution_impossible
  (initial_white_boxes : ℕ)
  (initial_black_boxes : ℕ)
  (new_white_boxes : ℕ)
  (new_black_boxes : ℕ)
  (total_white_boxes : ℕ)
  (total_black_boxes : ℕ) :
  initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count =
  total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count →
  (new_white_boxes, new_black_boxes) = (total_white_boxes - initial_white_boxes, total_black_boxes - initial_black_boxes) →
  ¬(∃ total_white_boxes total_black_boxes, 
    total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count =
    initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count) :=
by sorry

end ball_redistribution_impossible_l1654_165443


namespace largest_y_coordinate_of_degenerate_ellipse_l1654_165457

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + (y + 5)^2 / 16 = 0) → y = -5 :=
by
  intros x y h
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l1654_165457


namespace geometric_series_common_ratio_l1654_165447

theorem geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h_a : a = 500) (h_S : S = 3000) :
  ∃ r : ℝ, r = 5 / 6 :=
by
  sorry

end geometric_series_common_ratio_l1654_165447


namespace product_grades_probabilities_l1654_165445

theorem product_grades_probabilities (P_Q P_S : ℝ) (h1 : P_Q = 0.98) (h2 : P_S = 0.21) :
  P_Q - P_S = 0.77 ∧ 1 - P_Q = 0.02 :=
by
  sorry

end product_grades_probabilities_l1654_165445


namespace negation_of_universal_proposition_l1654_165439

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) :=
by sorry

end negation_of_universal_proposition_l1654_165439


namespace apples_harvested_from_garden_l1654_165493

def number_of_pies : ℕ := 10
def apples_per_pie : ℕ := 8
def apples_to_buy : ℕ := 30

def total_apples_needed : ℕ := number_of_pies * apples_per_pie

theorem apples_harvested_from_garden : total_apples_needed - apples_to_buy = 50 :=
by
  sorry

end apples_harvested_from_garden_l1654_165493


namespace mary_total_nickels_l1654_165416

theorem mary_total_nickels : (7 + 12 + 9 = 28) :=
by
  sorry

end mary_total_nickels_l1654_165416


namespace abs_add_eq_abs_sub_implies_mul_eq_zero_l1654_165403

variable {a b : ℝ}

theorem abs_add_eq_abs_sub_implies_mul_eq_zero (h : |a + b| = |a - b|) : a * b = 0 :=
sorry

end abs_add_eq_abs_sub_implies_mul_eq_zero_l1654_165403


namespace sum_of_three_numbers_l1654_165476

theorem sum_of_three_numbers {a b c : ℝ} (h₁ : a ≤ b ∧ b ≤ c) (h₂ : b = 10)
  (h₃ : (a + b + c) / 3 = a + 20) (h₄ : (a + b + c) / 3 = c - 25) :
  a + b + c = 45 :=
by
  sorry

end sum_of_three_numbers_l1654_165476


namespace water_consumption_eq_l1654_165404

-- Define all conditions
variables (x : ℝ) (improvement : ℝ := 0.8) (water : ℝ := 80) (days_difference : ℝ := 5)

-- State the theorem
theorem water_consumption_eq (h : improvement = 0.8) (initial_water := 80) (difference := 5) : 
  initial_water / x - (initial_water * improvement) / x = difference :=
sorry

end water_consumption_eq_l1654_165404


namespace tickets_spent_l1654_165421

theorem tickets_spent (initial_tickets : ℕ) (tickets_left : ℕ) (tickets_spent : ℕ) 
  (h1 : initial_tickets = 11) (h2 : tickets_left = 8) : tickets_spent = 3 :=
by
  sorry

end tickets_spent_l1654_165421


namespace range_of_2a_sub_b_l1654_165441

theorem range_of_2a_sub_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) : -4 < 2 * a - b ∧ 2 * a - b < 2 :=
by
  sorry

end range_of_2a_sub_b_l1654_165441


namespace oranges_bought_l1654_165455

theorem oranges_bought (total_cost : ℝ) 
  (selling_price_per_orange : ℝ) 
  (profit_per_orange : ℝ) 
  (cost_price_per_orange : ℝ) 
  (h1 : total_cost = 12.50)
  (h2 : selling_price_per_orange = 0.60)
  (h3 : profit_per_orange = 0.10)
  (h4 : cost_price_per_orange = selling_price_per_orange - profit_per_orange) :
  (total_cost / cost_price_per_orange) = 25 := 
by
  sorry

end oranges_bought_l1654_165455


namespace part_a_l1654_165410

theorem part_a (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ (m : ℕ) (x1 x2 x3 x4 : ℤ), m < p ∧ (x1^2 + x2^2 + x3^2 + x4^2 = m * p) :=
sorry

end part_a_l1654_165410


namespace passing_grade_fraction_l1654_165418

variables (students : ℕ) -- total number of students in Mrs. Susna's class

-- Conditions
def fraction_A : ℚ := 1/4
def fraction_B : ℚ := 1/2
def fraction_C : ℚ := 1/8
def fraction_D : ℚ := 1/12
def fraction_F : ℚ := 1/24

-- Prove the fraction of students getting a passing grade (C or higher) is 7/8
theorem passing_grade_fraction : 
  fraction_A + fraction_B + fraction_C = 7/8 :=
by
  sorry

end passing_grade_fraction_l1654_165418


namespace num_ordered_pairs_of_squares_diff_by_144_l1654_165465

theorem num_ordered_pairs_of_squares_diff_by_144 :
  ∃ (p : Finset (ℕ × ℕ)), p.card = 4 ∧ ∀ (a b : ℕ), (a, b) ∈ p → a ≥ b ∧ a^2 - b^2 = 144 := by
  sorry

end num_ordered_pairs_of_squares_diff_by_144_l1654_165465


namespace determine_percentage_of_yellow_in_darker_green_paint_l1654_165412

noncomputable def percentage_of_yellow_in_darker_green_paint : Real :=
  let volume_light_green := 5
  let volume_darker_green := 1.66666666667
  let percentage_light_green := 0.20
  let final_percentage := 0.25
  let total_volume := volume_light_green + volume_darker_green
  let total_yellow_required := final_percentage * total_volume
  let yellow_in_light_green := percentage_light_green * volume_light_green
  (total_yellow_required - yellow_in_light_green) / volume_darker_green

theorem determine_percentage_of_yellow_in_darker_green_paint :
  percentage_of_yellow_in_darker_green_paint = 0.4 := by
  sorry

end determine_percentage_of_yellow_in_darker_green_paint_l1654_165412


namespace proof_a_eq_b_pow_n_l1654_165479

theorem proof_a_eq_b_pow_n
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n := 
by sorry

end proof_a_eq_b_pow_n_l1654_165479


namespace range_of_k_l1654_165454

theorem range_of_k 
  (k : ℝ) 
  (line_intersects_hyperbola : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) : 
  -Real.sqrt (15) / 3 < k ∧ k < Real.sqrt (15) / 3 := 
by
  sorry

end range_of_k_l1654_165454


namespace average_age_of_all_l1654_165436

theorem average_age_of_all (students parents : ℕ) (student_avg parent_avg : ℚ) 
  (h_students: students = 40) 
  (h_student_avg: student_avg = 12) 
  (h_parents: parents = 60) 
  (h_parent_avg: parent_avg = 36)
  : (students * student_avg + parents * parent_avg) / (students + parents) = 26.4 :=
by
  sorry

end average_age_of_all_l1654_165436


namespace heaps_never_empty_l1654_165407

-- Define initial conditions
def initial_heaps := (1993, 199, 19)

-- Allowed operations
def add_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a + b + c, b, c)
else if b = 199 then (a, b + a + c, c)
else (a, b, c + a + b)

def remove_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a - (b + c), b, c)
else if b = 199 then (a, b - (a + c), c)
else (a, b, c - (a + b))

-- The proof statement
theorem heaps_never_empty :
  ∀ a b c : ℕ, a = 1993 ∧ b = 199 ∧ c = 19 ∧ (∀ n : ℕ, (a + b + c) % 2 = 1) ∧ (a - (b + c) % 2 = 1) → ¬(a = 0 ∨ b = 0 ∨ c = 0) := 
by {
  sorry
}

end heaps_never_empty_l1654_165407


namespace roller_coaster_costs_4_l1654_165477

-- Definitions from conditions
def tickets_initial: ℕ := 5                     -- Jeanne initially has 5 tickets
def tickets_to_buy: ℕ := 8                      -- Jeanne needs to buy 8 more tickets
def total_tickets_needed: ℕ := tickets_initial + tickets_to_buy -- Total tickets needed
def tickets_ferris_wheel: ℕ := 5                -- Ferris wheel costs 5 tickets
def tickets_total_after_ferris_wheel: ℕ := total_tickets_needed - tickets_ferris_wheel -- Remaining tickets after Ferris wheel

-- Definition to be proved (question = answer)
def cost_roller_coaster_bumper_cars: ℕ := tickets_total_after_ferris_wheel / 2 -- Each of roller coaster and bumper cars cost

-- The theorem that corresponds to the solution
theorem roller_coaster_costs_4 :
  cost_roller_coaster_bumper_cars = 4 :=
by
  sorry

end roller_coaster_costs_4_l1654_165477


namespace non_divisible_l1654_165480

theorem non_divisible (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ¬ ∃ k : ℤ, x^2 + y^2 + z^2 = k * 3 * (x * y + y * z + z * x) :=
by sorry

end non_divisible_l1654_165480


namespace length_of_faster_train_l1654_165464

theorem length_of_faster_train
    (speed_faster : ℕ)
    (speed_slower : ℕ)
    (time_cross : ℕ)
    (h_fast : speed_faster = 72)
    (h_slow : speed_slower = 36)
    (h_time : time_cross = 15) :
    (speed_faster - speed_slower) * (1000 / 3600) * time_cross = 150 := 
by
  sorry

end length_of_faster_train_l1654_165464


namespace parallel_vectors_t_eq_neg1_l1654_165458

theorem parallel_vectors_t_eq_neg1 (t : ℝ) :
  let a := (1, -1)
  let b := (t, 1)
  (a.1 + b.1, a.2 + b.2) = (k * (a.1 - b.1), k * (a.2 - b.2)) -> t = -1 :=
by
  sorry

end parallel_vectors_t_eq_neg1_l1654_165458


namespace union_sets_l1654_165414

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets : A ∪ B = {x | -1 < x ∧ x < 2} := by
  sorry

end union_sets_l1654_165414


namespace shortest_chord_through_point_l1654_165432

theorem shortest_chord_through_point 
  (P : ℝ × ℝ) (hx : P = (2, 1))
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → (x, y) ∈ {p : ℝ × ℝ | (p.fst - 1)^2 + p.snd^2 = 4}) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ a * (P.1) + b * (P.2) + c = 0 := 
by
  -- proof skipped
  sorry

end shortest_chord_through_point_l1654_165432


namespace exists_n_l1654_165456

theorem exists_n (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ¬(2^n ∣ a^k + b^k + c^k) :=
by
  sorry

end exists_n_l1654_165456


namespace rectangle_area_unchanged_l1654_165473

theorem rectangle_area_unchanged (x y : ℕ) (h1 : x * y = (x + 5/2) * (y - 2/3)) (h2 : x * y = (x - 5/2) * (y + 4/3)) : x * y = 20 :=
by
  sorry

end rectangle_area_unchanged_l1654_165473


namespace real_root_of_P_l1654_165435

noncomputable def P : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| n+2, x => x * P (n + 1) x + (1 - x) * P n x

theorem real_root_of_P (n : ℕ) (hn : 1 ≤ n) : ∀ x : ℝ, P n x = 0 → x = 0 := 
by 
  sorry

end real_root_of_P_l1654_165435


namespace canoe_downstream_speed_l1654_165453

-- Definitions based on conditions
def upstream_speed : ℝ := 9  -- upspeed
def stream_speed : ℝ := 1.5  -- vspeed

-- Theorem to prove the downstream speed
theorem canoe_downstream_speed (V_c : ℝ) (V_d : ℝ) :
  (V_c - stream_speed = upstream_speed) →
  (V_d = V_c + stream_speed) →
  V_d = 12 := by 
  intro h1 h2
  sorry

end canoe_downstream_speed_l1654_165453


namespace factor_sum_l1654_165437

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l1654_165437


namespace abs_diff_less_abs_one_minus_prod_l1654_165442

theorem abs_diff_less_abs_one_minus_prod (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end abs_diff_less_abs_one_minus_prod_l1654_165442


namespace sequence_term_25_l1654_165468

theorem sequence_term_25 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → a n = (a (n - 1) + a (n + 1)) / 4)
  (h2 : a 1 = 1)
  (h3 : a 9 = 40545) : 
  a 25 = 57424611447841 := 
sorry

end sequence_term_25_l1654_165468


namespace numberOfCows_l1654_165463

-- Definitions coming from the conditions
def hasFoxes (n : Nat) := n = 15
def zebrasFromFoxes (z f : Nat) := z = 3 * f
def totalAnimalRequirement (total : Nat) := total = 100
def addedSheep (s : Nat) := s = 20

-- Theorem stating the desired proof
theorem numberOfCows (f z total s c : Nat) 
 (h1 : hasFoxes f)
 (h2 : zebrasFromFoxes z f) 
 (h3 : totalAnimalRequirement total) 
 (h4 : addedSheep s) :
 c = total - s - (f + z) := by
 sorry

end numberOfCows_l1654_165463


namespace graph_is_empty_l1654_165428

theorem graph_is_empty :
  ¬∃ x y : ℝ, 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 :=
by
  -- the proof logic will go here
  sorry

end graph_is_empty_l1654_165428


namespace total_cartons_accepted_l1654_165408

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_cartons_accepted_l1654_165408


namespace total_problems_l1654_165427

-- We define the conditions as provided.
variables (p t : ℕ) -- p and t are positive whole numbers
variables (p_gt_10 : 10 < p) -- p is more than 10

theorem total_problems (p t : ℕ) (p_gt_10 : 10 < p) (h : p * t = (2 * p - 4) * (t - 2)):
  p * t = 60 :=
by
  sorry

end total_problems_l1654_165427


namespace abs_neg_three_l1654_165413

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l1654_165413


namespace negation_of_forall_ge_implies_exists_lt_l1654_165496

theorem negation_of_forall_ge_implies_exists_lt :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x := by
  sorry

end negation_of_forall_ge_implies_exists_lt_l1654_165496


namespace max_product_of_two_integers_l1654_165471

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l1654_165471


namespace B_work_days_l1654_165495

theorem B_work_days (x : ℝ) :
  (1 / 3 + 1 / x = 1 / 2) → x = 6 := by
  sorry

end B_work_days_l1654_165495


namespace find_a5_l1654_165422

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a5 (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_a2 : a 2 = 2) (h_a8 : a 8 = 32) :
  a 5 = 8 :=
by
  sorry

end find_a5_l1654_165422


namespace sum_of_solutions_l1654_165424

theorem sum_of_solutions (x y : ℝ) (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : x + y = 2 := 
sorry

end sum_of_solutions_l1654_165424


namespace right_triangle_integers_solutions_l1654_165406

theorem right_triangle_integers_solutions :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a^2 + b^2 = c^2 ∧ (a + b + c : ℕ) = (1 / 2 * a * b : ℚ) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
sorry

end right_triangle_integers_solutions_l1654_165406


namespace average_age_of_others_when_youngest_was_born_l1654_165405

noncomputable def average_age_when_youngest_was_born (total_people : ℕ) (average_age : ℕ) (youngest_age : ℕ) : ℚ :=
  let total_age := total_people * average_age
  let age_without_youngest := total_age - youngest_age
  age_without_youngest / (total_people - 1)

theorem average_age_of_others_when_youngest_was_born :
  average_age_when_youngest_was_born 7 30 7 = 33.833 :=
by
  sorry

end average_age_of_others_when_youngest_was_born_l1654_165405


namespace mary_has_more_money_than_marco_l1654_165415

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l1654_165415


namespace find_x_prime_l1654_165425

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l1654_165425


namespace min_value_expression_l1654_165488

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ( (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ) ≥ 7 :=
sorry

end min_value_expression_l1654_165488


namespace three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l1654_165420

theorem three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3 :
  ∃ x1 x2 x3 : ℕ, ((x1 = 414 ∧ x2 = 444 ∧ x3 = 474) ∧ 
  (∀ n, (100 * 4 + 10 * n + 4 = x1 ∨ 100 * 4 + 10 * n + 4 = x2 ∨ 100 * 4 + 10 * n + 4 = x3) 
  → (100 * 4 + 10 * n + 4) % 3 = 0)) :=
by
  sorry

end three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l1654_165420
