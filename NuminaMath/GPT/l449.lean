import Mathlib

namespace pencils_cost_proportion_l449_44924

/-- 
If a set of 15 pencils costs 9 dollars and the price of the set is directly 
proportional to the number of pencils it contains, then the cost of a set of 
35 pencils is 21 dollars.
--/
theorem pencils_cost_proportion :
  ∀ (p : ℕ), (∀ n : ℕ, n * 9 = p * 15) -> (35 * 9 = 21 * 15) :=
by
  intro p h1
  sorry

end pencils_cost_proportion_l449_44924


namespace mixture_percent_chemical_a_l449_44927

-- Defining the conditions
def solution_x : ℝ := 0.4
def solution_y : ℝ := 0.5
def percent_x_in_mixture : ℝ := 0.3
def percent_y_in_mixture : ℝ := 1.0 - percent_x_in_mixture

-- The goal is to prove that the mixture is 47% chemical a
theorem mixture_percent_chemical_a : (solution_x * percent_x_in_mixture + solution_y * percent_y_in_mixture) * 100 = 47 :=
by
  -- Calculation here
  sorry

end mixture_percent_chemical_a_l449_44927


namespace problem_l449_44945

namespace MathProof

-- Definitions of A, B, and conditions
def A (x : ℤ) : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

-- Prove x = ± 1 when A ⊆ B, 
-- A ∪ B = { -1, 0, 1 }, 
-- and complement of A in B is { -1 }
theorem problem (x : ℤ) (hx : A x ⊆ B) : 
  (x = 1 ∨ x = -1) ∧ 
  (A x ∪ B = {-1, 0, 1}) ∧ 
  (B \ (A x) = {-1}) := 
sorry 

end MathProof

end problem_l449_44945


namespace firetruck_reachable_area_l449_44992

theorem firetruck_reachable_area :
  let speed_highway := 50
  let speed_prairie := 14
  let travel_time := 0.1
  let area := 16800 / 961
  ∀ (x r : ℝ),
    (x / speed_highway + r / speed_prairie = travel_time) →
    (0 ≤ x ∧ 0 ≤ r) →
    ∃ m n : ℕ, gcd m n = 1 ∧
    m = 16800 ∧ n = 961 ∧
    m + n = 16800 + 961 := by
  sorry

end firetruck_reachable_area_l449_44992


namespace find_a_and_b_l449_44966

noncomputable def f (x: ℝ) (b: ℝ): ℝ := x^2 + 5*x + b
noncomputable def g (x: ℝ) (b: ℝ): ℝ := 2*b*x + 3

theorem find_a_and_b (a b: ℝ):
  (∀ x: ℝ, f (g x b) b = a * x^2 + 30 * x + 24) →
  a = 900 / 121 ∧ b = 15 / 11 :=
by
  intro H
  -- Proof is omitted as requested
  sorry

end find_a_and_b_l449_44966


namespace least_possible_value_of_p_and_q_l449_44947

theorem least_possible_value_of_p_and_q 
  (p q : ℕ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 15 * (p + 1) = 29 * (q + 1)) : 
  p + q = 45 := 
sorry -- proof to be filled in

end least_possible_value_of_p_and_q_l449_44947


namespace count_boys_correct_l449_44978

def total_vans : ℕ := 5
def students_per_van : ℕ := 28
def number_of_girls : ℕ := 80

theorem count_boys_correct : 
  (total_vans * students_per_van) - number_of_girls = 60 := 
by
  sorry

end count_boys_correct_l449_44978


namespace paint_intensity_change_l449_44918

theorem paint_intensity_change (intensity_original : ℝ) (intensity_new : ℝ) (fraction_replaced : ℝ) 
  (h1 : intensity_original = 0.40) (h2 : intensity_new = 0.20) (h3 : fraction_replaced = 1) :
  intensity_new = 0.20 :=
by
  sorry

end paint_intensity_change_l449_44918


namespace compute_Z_value_l449_44961

def operation_Z (c d : ℕ) : ℤ := c^2 - 3 * c * d + d^2

theorem compute_Z_value : operation_Z 4 3 = -11 := by
  sorry

end compute_Z_value_l449_44961


namespace bridge_length_correct_l449_44999

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 72
noncomputable def crossing_time : ℝ := 12.399008079353651

-- converting train speed from km/hr to m/s
noncomputable def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- total length the train covers to cross the bridge
noncomputable def total_length : ℝ := train_speed_m_per_s * crossing_time

-- length of the bridge
noncomputable def bridge_length : ℝ := total_length - train_length

theorem bridge_length_correct :
  bridge_length = 137.98 :=
by 
  sorry

end bridge_length_correct_l449_44999


namespace f_2023_pi_over_3_eq_4_l449_44997

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.cos x
| (n + 1), x => 4 / (2 - f n x)

theorem f_2023_pi_over_3_eq_4 : f 2023 (Real.pi / 3) = 4 := 
  sorry

end f_2023_pi_over_3_eq_4_l449_44997


namespace exists_integer_roots_l449_44981

theorem exists_integer_roots : 
  ∃ (a b c d e f : ℤ), ∃ r1 r2 r3 r4 r5 r6 : ℤ,
  (r1 + a) * (r2 ^ 2 + b * r2 + c) * (r3 ^ 3 + d * r3 ^ 2 + e * r3 + f) = 0 ∧
  (r4 + a) * (r5 ^ 2 + b * r5 + c) * (r6 ^ 3 + d * r6 ^ 2 + e * r6 + f) = 0 :=
  sorry

end exists_integer_roots_l449_44981


namespace complex_number_equation_l449_44955

theorem complex_number_equation
  (f : ℂ → ℂ)
  (z : ℂ)
  (h : f (i - z) = 2 * z - i) :
  (1 - i) * f (2 - i) = -1 + 7 * i := by
  sorry

end complex_number_equation_l449_44955


namespace farmer_apples_after_giving_away_l449_44968

def initial_apples : ℕ := 127
def given_away_apples : ℕ := 88
def remaining_apples : ℕ := 127 - 88

theorem farmer_apples_after_giving_away : remaining_apples = 39 := by
  sorry

end farmer_apples_after_giving_away_l449_44968


namespace scientific_notation_of_concentration_l449_44995

theorem scientific_notation_of_concentration :
  0.000042 = 4.2 * 10^(-5) :=
sorry

end scientific_notation_of_concentration_l449_44995


namespace multiple_of_n_eventually_written_l449_44919

theorem multiple_of_n_eventually_written (a b n : ℕ) (h_a_pos: 0 < a) (h_b_pos: 0 < b)  (h_ab_neq: a ≠ b) (h_n_pos: 0 < n) :
  ∃ m : ℕ, m % n = 0 :=
sorry

end multiple_of_n_eventually_written_l449_44919


namespace cricket_overs_played_initially_l449_44969

variables (x y : ℝ)

theorem cricket_overs_played_initially 
  (h1 : y = 3.2 * x)
  (h2 : 262 - y = 5.75 * 40) : 
  x = 10 := 
sorry

end cricket_overs_played_initially_l449_44969


namespace steaks_from_15_pounds_of_beef_l449_44913

-- Definitions for conditions
def pounds_to_ounces (pounds : ℕ) : ℕ := pounds * 16

def steaks_count (total_ounces : ℕ) (ounces_per_steak : ℕ) : ℕ := total_ounces / ounces_per_steak

-- Translate the problem to Lean statement
theorem steaks_from_15_pounds_of_beef : 
  steaks_count (pounds_to_ounces 15) 12 = 20 :=
by
  sorry

end steaks_from_15_pounds_of_beef_l449_44913


namespace find_a_of_min_value_of_f_l449_44957

noncomputable def f (a x : ℝ) : ℝ := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) + 2 * a * Real.sin x + 4 * a * Real.cos x

theorem find_a_of_min_value_of_f :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≥ -6) ∧ (∃ x : ℝ, f a x = -6)) → (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by
  sorry

end find_a_of_min_value_of_f_l449_44957


namespace part1_part2_part3_l449_44915

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end part1_part2_part3_l449_44915


namespace housewife_spend_money_l449_44930

theorem housewife_spend_money (P M: ℝ) (h1: 0.75 * P = 30) (h2: M / (0.75 * P) - M / P = 5) : 
  M = 600 :=
by
  sorry

end housewife_spend_money_l449_44930


namespace total_earning_proof_l449_44964

noncomputable def total_earning (daily_wage_c : ℝ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) : ℝ :=
  let daily_wage_a := (ratio_a : ℝ) / (ratio_c : ℝ) * daily_wage_c
  let daily_wage_b := (ratio_b : ℝ) / (ratio_c : ℝ) * daily_wage_c
  (daily_wage_a * days_a) + (daily_wage_b * days_b) + (daily_wage_c * days_c)

theorem total_earning_proof : 
  total_earning 71.15384615384615 16 9 4 3 4 5 = 1480 := 
by 
  -- calculations here
  sorry

end total_earning_proof_l449_44964


namespace geometric_progression_condition_l449_44951

noncomputable def condition_for_geometric_progression (a q : ℝ) (n p : ℤ) : Prop :=
  ∃ m : ℤ, a = q^m

theorem geometric_progression_condition (a q : ℝ) (n p k : ℤ) :
  condition_for_geometric_progression a q n p ↔ a * q^(n + p) = a * q^k :=
by
  sorry

end geometric_progression_condition_l449_44951


namespace sum_of_roots_l449_44934

theorem sum_of_roots {a b : Real} (h1 : a * (a - 4) = 5) (h2 : b * (b - 4) = 5) (h3 : a ≠ b) : a + b = 4 :=
by
  sorry

end sum_of_roots_l449_44934


namespace find_constant_l449_44910

-- Given function f satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the given conditions
variable (h1 : ∀ x : ℝ, f x + 3 * f (c - x) = x)
variable (h2 : f 2 = 2)

-- Statement to prove the constant c
theorem find_constant (c : ℝ) : (f x + 3 * f (c - x) = x) → (f 2 = 2) → c = 8 :=
by
  intro h1 h2
  sorry

end find_constant_l449_44910


namespace problem_l449_44935

def otimes (x y : ℝ) : ℝ := x^3 + y - 2 * x

theorem problem (k : ℝ) : otimes k (otimes k k) = 2 * k^3 - 3 * k :=
by
  sorry

end problem_l449_44935


namespace xiaoMing_better_performance_l449_44948

-- Definitions based on conditions
def xiaoMing_scores : List Float := [90, 67, 90, 92, 96]
def xiaoLiang_scores : List Float := [87, 62, 90, 92, 92]

-- Definitions of average and variance calculation
def average (scores : List Float) : Float :=
  (scores.sum) / (scores.length.toFloat)

def variance (scores : List Float) : Float :=
  let avg := average scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length.toFloat)

-- Prove that Xiao Ming's performance is better than Xiao Liang's.
theorem xiaoMing_better_performance :
  average xiaoMing_scores > average xiaoLiang_scores ∧ variance xiaoMing_scores < variance xiaoLiang_scores :=
by
  sorry

end xiaoMing_better_performance_l449_44948


namespace problem_b_lt_a_lt_c_l449_44911

theorem problem_b_lt_a_lt_c (a b c : ℝ)
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : b < a ∧ a < c := by
  sorry

end problem_b_lt_a_lt_c_l449_44911


namespace probability_rolls_more_ones_than_eights_l449_44953

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end probability_rolls_more_ones_than_eights_l449_44953


namespace TrishulPercentageLessThanRaghu_l449_44952

-- Define the variables and conditions
variables (R T V : ℝ)

-- Raghu's investment is Rs. 2200
def RaghuInvestment := (R : ℝ) = 2200

-- Vishal invested 10% more than Trishul
def VishalInvestment := (V : ℝ) = 1.10 * T

-- Total sum of investments is Rs. 6358
def TotalInvestment := R + T + V = 6358

-- Define the proof statement
theorem TrishulPercentageLessThanRaghu (R_is_2200 : RaghuInvestment R) 
    (V_is_10_percent_more : VishalInvestment V T) 
    (total_sum_is_6358 : TotalInvestment R T V) : 
  ((2200 - T) / 2200) * 100 = 10 :=
sorry

end TrishulPercentageLessThanRaghu_l449_44952


namespace john_less_than_david_by_4_l449_44977

/-
The conditions are:
1. Zachary did 51 push-ups.
2. David did 22 more push-ups than Zachary.
3. John did 69 push-ups.

We need to prove that John did 4 push-ups less than David.
-/

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := zachary_pushups + 22
def john_pushups : ℕ := 69

theorem john_less_than_david_by_4 :
  david_pushups - john_pushups = 4 :=
by
  -- Proof goes here.
  sorry

end john_less_than_david_by_4_l449_44977


namespace volume_of_sphere_eq_4_sqrt3_pi_l449_44998

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r ^ 3

theorem volume_of_sphere_eq_4_sqrt3_pi
  (r : ℝ) (h : 4 * Real.pi * r ^ 2 = 2 * Real.sqrt 3 * Real.pi * (2 * r)) :
  volume_of_sphere r = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end volume_of_sphere_eq_4_sqrt3_pi_l449_44998


namespace find_y_payment_l449_44979

-- Defining the conditions
def total_payment : ℝ := 700
def x_payment (y_payment : ℝ) : ℝ := 1.2 * y_payment

-- The theorem we want to prove
theorem find_y_payment (y_payment : ℝ) (h1 : y_payment + x_payment y_payment = total_payment) :
  y_payment = 318.18 := 
sorry

end find_y_payment_l449_44979


namespace postage_unformable_l449_44901

theorem postage_unformable (n : ℕ) (h₁ : n > 0) (h₂ : 110 = 7 * n - 7 - n) :
  n = 19 := 
sorry

end postage_unformable_l449_44901


namespace proposition_not_true_3_l449_44936

theorem proposition_not_true_3 (P : ℕ → Prop) (h1 : ∀ n, P n → P (n + 1)) (h2 : ¬ P 4) : ¬ P 3 :=
by
  sorry

end proposition_not_true_3_l449_44936


namespace value_of_a_l449_44914

theorem value_of_a {a : ℝ} : 
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 2 = 0 ∧ ∀ y : ℝ, (a - 1) * y^2 + 4 * y - 2 ≠ 0 → y = x) → 
  (a = 1 ∨ a = -1) :=
by 
  sorry

end value_of_a_l449_44914


namespace find_somus_age_l449_44967

def somus_current_age (S F : ℕ) := S = F / 3
def somus_age_7_years_ago (S F : ℕ) := (S - 7) = (F - 7) / 5

theorem find_somus_age (S F : ℕ) 
  (h1 : somus_current_age S F) 
  (h2 : somus_age_7_years_ago S F) : S = 14 :=
sorry

end find_somus_age_l449_44967


namespace Nicole_cards_l449_44984

variables (N : ℕ)

-- Conditions from step A
def Cindy_collected (N : ℕ) : ℕ := 2 * N
def Nicole_and_Cindy_combined (N : ℕ) : ℕ := N + Cindy_collected N
def Rex_collected (N : ℕ) : ℕ := (Nicole_and_Cindy_combined N) / 2
def Rex_cards_each (N : ℕ) : ℕ := Rex_collected N / 4

-- Question: How many cards did Nicole collect? Answer: N = 400
theorem Nicole_cards (N : ℕ) (h : Rex_cards_each N = 150) : N = 400 :=
sorry

end Nicole_cards_l449_44984


namespace people_in_gym_l449_44983

-- Define the initial number of people in the gym
def initial_people : ℕ := 16

-- Define the number of additional people entering the gym
def additional_people : ℕ := 5

-- Define the number of people leaving the gym
def people_leaving : ℕ := 2

-- Define the final number of people in the gym as per the conditions
def final_people (initial : ℕ) (additional : ℕ) (leaving : ℕ) : ℕ :=
  initial + additional - leaving

-- The theorem to prove
theorem people_in_gym : final_people initial_people additional_people people_leaving = 19 :=
  by
    sorry

end people_in_gym_l449_44983


namespace sum_not_complete_residue_system_l449_44929

theorem sum_not_complete_residue_system
  (n : ℕ) (hn : Even n)
  (a b : Fin n → Fin n)
  (ha : ∀ i : Fin n, ∃ j : Fin n, a j = i)
  (hb : ∀ i : Fin n, ∃ j : Fin n, b j = i) :
  ¬ (∀ k : Fin n, ∃ i : Fin n, a i + b i = k) :=
sorry

end sum_not_complete_residue_system_l449_44929


namespace lead_atom_ratio_l449_44932

noncomputable def ratio_of_lead_atoms (average_weight : ℝ) 
  (weight_206 : ℕ) (weight_207 : ℕ) (weight_208 : ℕ) 
  (number_206 : ℕ) (number_207 : ℕ) (number_208 : ℕ) : Prop :=
  average_weight = 207.2 ∧ 
  weight_206 = 206 ∧ 
  weight_207 = 207 ∧ 
  weight_208 = 208 ∧ 
  number_208 = number_206 + number_207 →
  (number_206 : ℚ) / (number_207 : ℚ) = 3 / 2 ∧
  (number_208 : ℚ) / (number_207 : ℚ) = 5 / 2

theorem lead_atom_ratio : ratio_of_lead_atoms 207.2 206 207 208 3 2 5 :=
by sorry

end lead_atom_ratio_l449_44932


namespace gcd_A_C_gcd_B_C_l449_44946

def A : ℕ := 177^5 + 30621 * 173^3 - 173^5
def B : ℕ := 173^5 + 30621 * 177^3 - 177^5
def C : ℕ := 173^4 + 30621^2 + 177^4

theorem gcd_A_C : Nat.gcd A C = 30637 := sorry

theorem gcd_B_C : Nat.gcd B C = 30637 := sorry

end gcd_A_C_gcd_B_C_l449_44946


namespace find_line_eq_l449_44920

theorem find_line_eq (m b k : ℝ) (h1 : (2, 7) ∈ ⋃ x, {(x, m * x + b)}) (h2 : ∀ k, abs ((k^2 + 4 * k + 3) - (m * k + b)) = 4) (h3 : b ≠ 0) : (m = 10) ∧ (b = -13) := by
  sorry

end find_line_eq_l449_44920


namespace box_dimension_min_sum_l449_44926

theorem box_dimension_min_sum :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 := by
  sorry

end box_dimension_min_sum_l449_44926


namespace cubics_of_sum_and_product_l449_44973

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end cubics_of_sum_and_product_l449_44973


namespace oranges_in_each_box_l449_44958

theorem oranges_in_each_box (total_oranges : ℝ) (boxes : ℝ) (h_total : total_oranges = 72) (h_boxes : boxes = 3.0) : total_oranges / boxes = 24 :=
by
  -- Begin proof
  sorry

end oranges_in_each_box_l449_44958


namespace polynomial_roots_identity_l449_44970

theorem polynomial_roots_identity {p q α β γ δ : ℝ} 
  (h1 : α^2 + p*α + 1 = 0)
  (h2 : β^2 + p*β + 1 = 0)
  (h3 : γ^2 + q*γ + 1 = 0)
  (h4 : δ^2 + q*δ + 1 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end polynomial_roots_identity_l449_44970


namespace class_A_has_neater_scores_l449_44954

-- Definitions for the given problem conditions
def mean_Class_A : ℝ := 120
def mean_Class_B : ℝ := 120
def variance_Class_A : ℝ := 42
def variance_Class_B : ℝ := 56

-- The theorem statement to prove Class A has neater scores
theorem class_A_has_neater_scores : (variance_Class_A < variance_Class_B) := by
  sorry

end class_A_has_neater_scores_l449_44954


namespace typist_speeds_l449_44994

noncomputable def num_pages : ℕ := 72
noncomputable def ratio : ℚ := 6 / 5
noncomputable def time_difference : ℚ := 1.5

theorem typist_speeds :
  ∃ (x y : ℚ), (x = 9.6 ∧ y = 8) ∧ 
                (num_pages / x - num_pages / y = time_difference) ∧
                (x / y = ratio) :=
by
  -- Let's skip the proof for now
  sorry

end typist_speeds_l449_44994


namespace equation_represents_circle_m_condition_l449_44941

theorem equation_represents_circle_m_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0) → m < 1/2 := 
by
  sorry

end equation_represents_circle_m_condition_l449_44941


namespace felicity_gas_usage_l449_44923

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l449_44923


namespace problem_l449_44996

theorem problem (a b c : ℝ) (h1 : ∀ (x : ℝ), x^2 + 3 * x - 1 = 0 → x^4 + a * x^2 + b * x + c = 0) :
  a + b + 4 * c + 100 = 93 := 
sorry

end problem_l449_44996


namespace hancho_milk_l449_44921

def initial_milk : ℝ := 1
def ye_seul_milk : ℝ := 0.1
def ga_young_milk : ℝ := ye_seul_milk + 0.2
def remaining_milk : ℝ := 0.3

theorem hancho_milk : (initial_milk - (ye_seul_milk + ga_young_milk + remaining_milk)) = 0.3 :=
by
  sorry

end hancho_milk_l449_44921


namespace jessica_final_balance_l449_44993

variable {original_balance current_balance final_balance withdrawal1 withdrawal2 deposit1 deposit2 : ℝ}

theorem jessica_final_balance:
  (2 / 5) * original_balance = 200 → 
  current_balance = original_balance - 200 → 
  withdrawal1 = (1 / 3) * current_balance → 
  current_balance - withdrawal1 = current_balance - (1 / 3 * current_balance) → 
  deposit1 = (1 / 5) * (current_balance - (1 / 3 * current_balance)) → 
  final_balance = (current_balance - (1 / 3 * current_balance)) + deposit1 → 
  deposit2 / 7 * 3 = final_balance - (current_balance - (1 / 3 * current_balance) + deposit1) → 
  (final_balance + deposit2) = 420 :=
sorry

end jessica_final_balance_l449_44993


namespace find_full_haired_dogs_l449_44938

-- Definitions of the given conditions
def minutes_per_short_haired_dog : Nat := 10
def short_haired_dogs : Nat := 6
def total_time_minutes : Nat := 4 * 60
def twice_as_long (n : Nat) : Nat := 2 * n

-- Define the problem
def full_haired_dogs : Nat :=
  let short_haired_total_time := short_haired_dogs * minutes_per_short_haired_dog
  let remaining_time := total_time_minutes - short_haired_total_time
  remaining_time / (twice_as_long minutes_per_short_haired_dog)

-- Theorem statement
theorem find_full_haired_dogs : 
  full_haired_dogs = 9 :=
by
  sorry

end find_full_haired_dogs_l449_44938


namespace speeding_tickets_l449_44963

theorem speeding_tickets (p1 p2 : ℝ)
  (h1 : p1 = 16.666666666666664)
  (h2 : p2 = 40) :
  (p1 * (100 - p2) / 100 = 10) :=
by sorry

end speeding_tickets_l449_44963


namespace count_correct_conclusions_l449_44939

structure Point where
  x : ℝ
  y : ℝ

def isDoublingPoint (P Q : Point) : Prop :=
  2 * (P.x + Q.x) = P.y + Q.y

def P1 : Point := {x := 2, y := 0}

def Q1 : Point := {x := 2, y := 8}
def Q2 : Point := {x := -3, y := -2}

def onLine (P : Point) : Prop :=
  P.y = P.x + 2

def onParabola (P : Point) : Prop :=
  P.y = P.x ^ 2 - 2 * P.x - 3

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem count_correct_conclusions :
  (isDoublingPoint P1 Q1) ∧
  (isDoublingPoint P1 Q2) ∧
  (∃ A : Point, onLine A ∧ isDoublingPoint P1 A ∧ A = {x := -2, y := 0}) ∧
  (∃ B₁ B₂ : Point, onParabola B₁ ∧ onParabola B₂ ∧ isDoublingPoint P1 B₁ ∧ isDoublingPoint P1 B₂) ∧
  (∃ B : Point, isDoublingPoint P1 B ∧
   ∀ P : Point, isDoublingPoint P1 P → dist P1 P ≥ dist P1 B ∧
   dist P1 B = 8 * (5:ℝ)^(1/2) / 5) :=
by sorry

end count_correct_conclusions_l449_44939


namespace infinite_series_sum_l449_44989

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 1998)^n) = (3992004 / 3988009) :=
by sorry

end infinite_series_sum_l449_44989


namespace binom_divisible_by_4_l449_44972

theorem binom_divisible_by_4 (n : ℕ) : (n ≠ 0) ∧ (¬ (∃ k : ℕ, n = 2^k)) ↔ 4 ∣ n * (Nat.choose (2 * n) n) :=
by
  sorry

end binom_divisible_by_4_l449_44972


namespace minimum_value_of_f_l449_44982

noncomputable def f (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem minimum_value_of_f (x : ℝ) (hx : x > 1) : (∃ y : ℝ, f x = 5 ∧ ∀ y > 1, f y ≥ 5) :=
sorry

end minimum_value_of_f_l449_44982


namespace prob_both_selected_l449_44976

-- Define the probabilities of selection
def prob_selection_x : ℚ := 1 / 5
def prob_selection_y : ℚ := 2 / 3

-- Prove that the probability that both x and y are selected is 2 / 15
theorem prob_both_selected : prob_selection_x * prob_selection_y = 2 / 15 := 
by
  sorry

end prob_both_selected_l449_44976


namespace product_of_two_numbers_l449_44986

theorem product_of_two_numbers (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_sum : a + b = 210) (h_lcm : Nat.lcm a b = 1547) : a * b = 10829 :=
by
  sorry

end product_of_two_numbers_l449_44986


namespace problem_I_l449_44960

def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_I {x : ℝ} : f (x + 3 / 2) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end problem_I_l449_44960


namespace find_k_l449_44909

theorem find_k (k : ℝ) (hk : 0 < k) (slope_eq : (2 - k) / (k - 1) = k^2) : k = 1 :=
by sorry

end find_k_l449_44909


namespace difference_thursday_tuesday_l449_44949

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end difference_thursday_tuesday_l449_44949


namespace notebooks_problem_l449_44971

variable (a b c : ℕ)

theorem notebooks_problem (h1 : a + 6 = b + c) (h2 : b + 10 = a + c) : c = 8 :=
  sorry

end notebooks_problem_l449_44971


namespace distance_between_P_and_F2_l449_44950
open Real

theorem distance_between_P_and_F2 (x y c : ℝ) (h1 : c = sqrt 3)
    (h2 : x = -sqrt 3) (h3 : y = 1/2) : 
    sqrt ((sqrt 3 - x) ^ 2 + (0 - y) ^ 2) = 7 / 2 :=
by
  sorry

end distance_between_P_and_F2_l449_44950


namespace simplify_expression_l449_44940

theorem simplify_expression : (625:ℝ)^(1/4) * (256:ℝ)^(1/2) = 80 := 
by 
  sorry

end simplify_expression_l449_44940


namespace merchant_marked_price_l449_44962

variable (L C M S : ℝ)

theorem merchant_marked_price :
  (C = 0.8 * L) → (C = 0.8 * S) → (S = 0.8 * M) → (M = 1.25 * L) :=
by
  sorry

end merchant_marked_price_l449_44962


namespace no_rational_roots_of_odd_coefficient_quadratic_l449_44906

theorem no_rational_roots_of_odd_coefficient_quadratic 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, r * r * a + r * b + c = 0 :=
by
  sorry

end no_rational_roots_of_odd_coefficient_quadratic_l449_44906


namespace cylinder_curved_surface_area_l449_44988

theorem cylinder_curved_surface_area {r h : ℝ} (hr: r = 2) (hh: h = 5) :  2 * Real.pi * r * h = 20 * Real.pi :=
by
  rw [hr, hh]
  sorry

end cylinder_curved_surface_area_l449_44988


namespace problem_solution_l449_44922

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 1)

theorem problem_solution (x : ℝ) : x ≥ 1 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2) ↔ (x = 13.25) :=
sorry

end problem_solution_l449_44922


namespace smallest_possible_value_l449_44965

theorem smallest_possible_value (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) (h2 : n ≡ 2 [MOD 9]) (h3 : n ≡ 6 [MOD 7]) :
  n = 116 :=
by
  -- Proof omitted
  sorry

end smallest_possible_value_l449_44965


namespace abs_inequality_solution_l449_44912

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l449_44912


namespace sufficient_but_not_necessary_condition_l449_44900

noncomputable def are_parallel (a : ℝ) : Prop :=
  (2 + a) * a * 3 * a = 3 * a * (a - 2)

theorem sufficient_but_not_necessary_condition :
  (are_parallel 4) ∧ (∃ a ≠ 4, are_parallel a) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l449_44900


namespace find_starting_number_of_range_l449_44931

theorem find_starting_number_of_range :
  ∃ n : ℕ, ∀ k : ℕ, k < 7 → (n + k * 9) ∣ 9 ∧ (n + k * 9) ≤ 97 ∧ (∀ m < k, (n + m * 9) < n + (m + 1) * 9) := 
sorry

end find_starting_number_of_range_l449_44931


namespace ellipse_symmetry_range_l449_44902

theorem ellipse_symmetry_range :
  ∀ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / 2 = 1) →
  ∃ (x₁ y₁ : ℝ), (x₁ = (4 * y₀ - 3 * x₀) / 5) ∧ (y₁ = (3 * y₀ + 4 * x₀) / 5) →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
by intros x₀ y₀ h_linearity; sorry

end ellipse_symmetry_range_l449_44902


namespace golu_distance_after_turning_left_l449_44905

theorem golu_distance_after_turning_left :
  ∀ (a c b : ℝ), a = 8 → c = 10 → (c ^ 2 = a ^ 2 + b ^ 2) → b = 6 :=
by
  intros a c b ha hc hpyth
  rw [ha, hc] at hpyth
  sorry

end golu_distance_after_turning_left_l449_44905


namespace plates_remove_proof_l449_44942

noncomputable def total_weight_initial (plates: ℤ) (weight_per_plate: ℤ): ℤ :=
  plates * weight_per_plate

noncomputable def weight_limit (pounds: ℤ) (ounces_per_pound: ℤ): ℤ :=
  pounds * ounces_per_pound

noncomputable def plates_to_remove (initial_weight: ℤ) (limit: ℤ) (weight_per_plate: ℤ): ℤ :=
  (initial_weight - limit) / weight_per_plate

theorem plates_remove_proof :
  let pounds := 20
  let ounces_per_pound := 16
  let plates_initial := 38
  let weight_per_plate := 10
  let initial_weight := total_weight_initial plates_initial weight_per_plate
  let limit := weight_limit pounds ounces_per_pound
  plates_to_remove initial_weight limit weight_per_plate = 6 :=
by
  sorry

end plates_remove_proof_l449_44942


namespace increasing_function_range_l449_44937

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
  sorry

end increasing_function_range_l449_44937


namespace find_certain_number_l449_44908

theorem find_certain_number (x : ℕ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  x = 58 := by
  sorry

end find_certain_number_l449_44908


namespace sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l449_44904

variable {a b : ℝ}

theorem sufficient_condition_for_reciprocal_square :
  (b > a ∧ a > 0) → (1 / a^2 > 1 / b^2) :=
sorry

theorem not_necessary_condition_for_reciprocal_square :
  ¬((1 / a^2 > 1 / b^2) → (b > a ∧ a > 0)) :=
sorry

end sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l449_44904


namespace intersection_of_sets_l449_44987

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3}) (hB : B = { x | x < 3 ∧ x ∈ Set.univ }) :
  A ∩ B = {0, 1, 2} :=
by
  sorry

end intersection_of_sets_l449_44987


namespace billy_age_l449_44959

theorem billy_age (B J : ℕ) (h1 : B = 3 * J) (h2 : B + J = 60) : B = 45 :=
by
  sorry

end billy_age_l449_44959


namespace cary_needs_six_weekends_l449_44975

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end cary_needs_six_weekends_l449_44975


namespace not_perfect_squares_l449_44916

theorem not_perfect_squares :
  (∀ x : ℝ, x * x ≠ 8 ^ 2041) ∧ (∀ y : ℝ, y * y ≠ 10 ^ 2043) :=
by
  sorry

end not_perfect_squares_l449_44916


namespace integer_in_range_l449_44928

theorem integer_in_range (x : ℤ) 
  (h1 : 0 < x) 
  (h2 : x < 7)
  (h3 : 0 < x)
  (h4 : x < 15)
  (h5 : -1 < x)
  (h6 : x < 5)
  (h7 : 0 < x)
  (h8 : x < 3)
  (h9 : x + 2 < 4) : x = 1 := 
sorry

end integer_in_range_l449_44928


namespace abc_value_l449_44925

theorem abc_value (a b c : ℝ) (h1 : ab = 30 * (4^(1/3))) (h2 : ac = 40 * (4^(1/3))) (h3 : bc = 24 * (4^(1/3))) :
  a * b * c = 120 :=
sorry

end abc_value_l449_44925


namespace largest_value_of_a_l449_44944

theorem largest_value_of_a : 
  ∃ (a : ℚ), (3 * a + 4) * (a - 2) = 9 * a ∧ ∀ b : ℚ, (3 * b + 4) * (b - 2) = 9 * b → b ≤ 4 :=
by
  sorry

end largest_value_of_a_l449_44944


namespace find_d_l449_44990

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + 2 * c^2 + 4 = 2 * d + Real.sqrt (a^2 + b^2 + c - d)) :
  d = 1/2 :=
sorry

end find_d_l449_44990


namespace team_OT_matches_l449_44917

variable (T x M: Nat)

-- Condition: Team C played T matches in the first week.
def team_C_matches_T : Nat := T

-- Condition: Team C played x matches in the first week.
def team_C_matches_x : Nat := x

-- Condition: Team O played M matches in the first week.
def team_O_matches_M : Nat := M

-- Condition: Team C has not played against Team A.
axiom C_not_played_A : ¬ (team_C_matches_T = team_C_matches_x)

-- Condition: Team B has not played against a specified team (interpreted).
axiom B_not_played_specified : ∀ x, ¬ (team_C_matches_x = x)

-- The proof for the number of matches played by team \(\overrightarrow{OT}\).
theorem team_OT_matches : T = 4 := 
    sorry

end team_OT_matches_l449_44917


namespace rectangular_prism_sum_l449_44991

theorem rectangular_prism_sum : 
  let edges := 12
  let vertices := 8
  let faces := 6
  edges + vertices + faces = 26 := by
sorry

end rectangular_prism_sum_l449_44991


namespace sum_in_base4_l449_44985

def dec_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert (n : ℕ) (acc : ℕ) (power : ℕ) :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * power) (power * 10)
  convert n 0 1

theorem sum_in_base4 : dec_to_base4 (234 + 78) = 13020 :=
  sorry

end sum_in_base4_l449_44985


namespace find_whole_wheat_pastry_flour_l449_44907

variable (x : ℕ) -- where x is the pounds of whole-wheat pastry flour Sarah already had

-- Conditions
def rye_flour := 5
def whole_wheat_bread_flour := 10
def chickpea_flour := 3
def total_flour := 20

-- Total flour bought
def total_flour_bought := rye_flour + whole_wheat_bread_flour + chickpea_flour

-- Proof statement
theorem find_whole_wheat_pastry_flour (h : total_flour = total_flour_bought + x) : x = 2 :=
by
  -- The proof is omitted
  sorry

end find_whole_wheat_pastry_flour_l449_44907


namespace product_cos_angles_l449_44903

theorem product_cos_angles :
  (Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128) :=
sorry

end product_cos_angles_l449_44903


namespace sum_of_squares_of_diagonals_l449_44933

variable (OP R : ℝ)

theorem sum_of_squares_of_diagonals (AC BD : ℝ) :
  AC^2 + BD^2 = 8 * R^2 - 4 * OP^2 :=
sorry

end sum_of_squares_of_diagonals_l449_44933


namespace common_ratio_geometric_sequence_l449_44974

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence_l449_44974


namespace simplify_expression_l449_44980

theorem simplify_expression (x : ℝ) : 5 * x + 2 * x + 7 * x = 14 * x :=
by
  sorry

end simplify_expression_l449_44980


namespace april_roses_l449_44943

theorem april_roses (price_per_rose earnings number_of_roses_left : ℕ) 
  (h1 : price_per_rose = 7) 
  (h2 : earnings = 35) 
  (h3 : number_of_roses_left = 4) : 
  (earnings / price_per_rose + number_of_roses_left) = 9 :=
by
  sorry

end april_roses_l449_44943


namespace roots_of_quadratic_identity_l449_44956

namespace RootProperties

theorem roots_of_quadratic_identity (a b : ℝ) 
(h1 : a^2 - 2*a - 1 = 0) 
(h2 : b^2 - 2*b - 1 = 0) 
(h3 : a ≠ b) 
: a^2 + b^2 = 6 := 
by sorry

end RootProperties

end roots_of_quadratic_identity_l449_44956
