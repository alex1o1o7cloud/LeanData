import Mathlib

namespace probability_of_B_l2338_233890

theorem probability_of_B (P : Set ℕ → ℝ) (A B : Set ℕ) (hA : P A = 0.25) (hAB : P (A ∩ B) = 0.15) (hA_complement_B_complement : P (Aᶜ ∩ Bᶜ) = 0.5) : P B = 0.4 :=
by
  sorry

end probability_of_B_l2338_233890


namespace probability_problems_l2338_233830

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_l2338_233830


namespace trader_profit_percentage_l2338_233872

-- Define the conditions.
variables (indicated_weight actual_weight_given claimed_weight : ℝ)
variable (profit_percentage : ℝ)

-- Given conditions
def conditions :=
  indicated_weight = 1000 ∧
  actual_weight_given = claimed_weight / 1.5 ∧
  claimed_weight = indicated_weight ∧
  profit_percentage = (claimed_weight - actual_weight_given) / actual_weight_given * 100

-- Prove that the profit percentage is 50%
theorem trader_profit_percentage : conditions indicated_weight actual_weight_given claimed_weight profit_percentage → profit_percentage = 50 :=
by
  sorry

end trader_profit_percentage_l2338_233872


namespace batsman_average_after_11th_inning_l2338_233847

theorem batsman_average_after_11th_inning (A : ℝ) 
  (h1 : A + 5 = (10 * A + 85) / 11) : A + 5 = 35 :=
by
  sorry

end batsman_average_after_11th_inning_l2338_233847


namespace stacy_current_height_l2338_233893

-- Conditions
def last_year_height_stacy : ℕ := 50
def brother_growth : ℕ := 1
def stacy_growth : ℕ := brother_growth + 6

-- Statement to prove
theorem stacy_current_height : last_year_height_stacy + stacy_growth = 57 :=
by
  sorry

end stacy_current_height_l2338_233893


namespace expression_min_value_l2338_233842

theorem expression_min_value (a b c k : ℝ) (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  (1 : ℝ) / c^2 * ((k * c - a)^2 + (a + c)^2 + (c - a)^2) ≥ k^2 / 3 + 2 :=
sorry

end expression_min_value_l2338_233842


namespace even_function_periodic_odd_function_period_generalized_period_l2338_233803

-- Problem 1
theorem even_function_periodic (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 2 * a) = f x :=
by sorry

-- Problem 2
theorem odd_function_period (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * a) = f x :=
by sorry

-- Problem 3
theorem generalized_period (f : ℝ → ℝ) (a m n : ℝ) (h₁ : ∀ x : ℝ, 2 * n - f x = f (2 * m - x)) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * (m - a)) = f x :=
by sorry

end even_function_periodic_odd_function_period_generalized_period_l2338_233803


namespace reflected_parabola_equation_l2338_233811

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line of reflection
def reflection_line (x : ℝ) : ℝ := x + 2

-- The reflected equation statement to be proved
theorem reflected_parabola_equation (x y : ℝ) :
  (parabola x = y) ∧ (reflection_line x = y) →
  (∃ y' x', x = y'^2 - 4 * y' + 2 ∧ y = x' + 2 ∧ x' = y - 2) :=
sorry

end reflected_parabola_equation_l2338_233811


namespace cos_330_eq_sqrt3_div_2_l2338_233877

theorem cos_330_eq_sqrt3_div_2
    (h1 : ∀ θ : ℝ, Real.cos (2 * Real.pi - θ) = Real.cos θ)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
    Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  -- Proof goes here
  sorry

end cos_330_eq_sqrt3_div_2_l2338_233877


namespace polynomial_identity_l2338_233839

theorem polynomial_identity
  (x a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h : (x - 1)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  (a + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = 0 :=
by sorry

end polynomial_identity_l2338_233839


namespace divisible_by_17_l2338_233817

theorem divisible_by_17 (a b c d : ℕ) (h1 : a + b + c + d = 2023)
    (h2 : 2023 ∣ (a * b - c * d))
    (h3 : 2023 ∣ (a^2 + b^2 + c^2 + d^2))
    (h4 : ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 7 ∣ x) :
    (∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 17 ∣ x) := 
sorry

end divisible_by_17_l2338_233817


namespace inequality_problem_l2338_233853

variable {R : Type*} [LinearOrderedField R]

theorem inequality_problem
  (a b : R) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hab : a + b = 1) :
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := 
sorry

end inequality_problem_l2338_233853


namespace number_of_sheep_l2338_233829

theorem number_of_sheep (S H : ℕ) 
  (h1 : S / H = 5 / 7)
  (h2 : H * 230 = 12880) : 
  S = 40 :=
by
  sorry

end number_of_sheep_l2338_233829


namespace cyclist_speed_ratio_is_4_l2338_233824

noncomputable def ratio_of_speeds (v_a v_b v_c : ℝ) : ℝ :=
  if v_a ≤ v_b ∧ v_b ≤ v_c then v_c / v_a else 0

theorem cyclist_speed_ratio_is_4
  (v_a v_b v_c : ℝ)
  (h1 : v_a + v_b = d / 5)
  (h2 : v_b + v_c = 15)
  (h3 : 15 = (45 - d) / 3)
  (d : ℝ) : 
  ratio_of_speeds v_a v_b v_c = 4 :=
by
  sorry

end cyclist_speed_ratio_is_4_l2338_233824


namespace deductive_reasoning_is_option_A_l2338_233878

-- Define the types of reasoning.
inductive ReasoningType
| Deductive
| Analogical
| Inductive

-- Define the options provided in the problem.
def OptionA : ReasoningType := ReasoningType.Deductive
def OptionB : ReasoningType := ReasoningType.Analogical
def OptionC : ReasoningType := ReasoningType.Inductive
def OptionD : ReasoningType := ReasoningType.Inductive

-- Statement to prove that Option A is Deductive reasoning.
theorem deductive_reasoning_is_option_A : OptionA = ReasoningType.Deductive := by
  -- proof
  sorry

end deductive_reasoning_is_option_A_l2338_233878


namespace watch_cost_price_l2338_233837

theorem watch_cost_price (SP_loss SP_gain CP : ℝ) 
  (h1 : SP_loss = 0.9 * CP) 
  (h2 : SP_gain = 1.04 * CP) 
  (h3 : SP_gain - SP_loss = 196) 
  : CP = 1400 := 
sorry

end watch_cost_price_l2338_233837


namespace circle_equation_l2338_233856

theorem circle_equation 
    (a : ℝ)
    (x y : ℝ)
    (tangent_lines : x + y = 0 ∧ x + y = 4)
    (center_line : x - y = a)
    (center_point : ∃ (a : ℝ), x = a ∧ y = a) :
    ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

end circle_equation_l2338_233856


namespace average_30_matches_is_25_l2338_233809

noncomputable def average_runs_in_30_matches (average_20_matches average_10_matches : ℝ) (total_matches_20 total_matches_10 : ℕ) : ℝ :=
  let total_runs_20 := total_matches_20 * average_20_matches
  let total_runs_10 := total_matches_10 * average_10_matches
  (total_runs_20 + total_runs_10) / (total_matches_20 + total_matches_10)

theorem average_30_matches_is_25 (h1 : average_runs_in_30_matches 30 15 20 10 = 25) : 
  average_runs_in_30_matches 30 15 20 10 = 25 := 
  by
    exact h1

end average_30_matches_is_25_l2338_233809


namespace r_power_four_identity_l2338_233849

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l2338_233849


namespace line_through_points_on_parabola_l2338_233806

theorem line_through_points_on_parabola
  (p q : ℝ)
  (hpq : p^2 - 4 * q > 0) :
  ∃ (A B : ℝ × ℝ),
    (exists (x₁ x₂ : ℝ), x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧
                         A = (x₁, x₁^2 / 3) ∧ B = (x₂, x₂^2 / 3) ∧
                         (∀ x y, (x, y) = A ∨ (x, y) = B → px + 3 * y + q = 0)) :=
sorry

end line_through_points_on_parabola_l2338_233806


namespace terminating_decimal_expansion_of_13_over_320_l2338_233867

theorem terminating_decimal_expansion_of_13_over_320 : ∃ (b : ℕ) (a : ℚ), (13 : ℚ) / 320 = a / 10 ^ b ∧ a / 10 ^ b = 0.650 :=
by
  sorry

end terminating_decimal_expansion_of_13_over_320_l2338_233867


namespace average_percentage_popped_average_percentage_kernels_l2338_233805

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end average_percentage_popped_average_percentage_kernels_l2338_233805


namespace side_length_c_4_l2338_233854

theorem side_length_c_4 (A : ℝ) (b S c : ℝ) 
  (hA : A = 120) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) : 
  c = 4 :=
sorry

end side_length_c_4_l2338_233854


namespace ellipse_graph_equivalence_l2338_233884

theorem ellipse_graph_equivalence :
  ∀ x y : ℝ, x^2 + 4 * y^2 - 6 * x + 8 * y + 9 = 0 ↔ (x - 3)^2 / 4 + (y + 1)^2 / 1 = 1 := by
  sorry

end ellipse_graph_equivalence_l2338_233884


namespace Josh_pencils_left_l2338_233870

theorem Josh_pencils_left (initial_pencils : ℕ) (given_pencils : ℕ) (remaining_pencils : ℕ) 
  (h_initial : initial_pencils = 142) 
  (h_given : given_pencils = 31) 
  (h_remaining : remaining_pencils = 111) : 
  initial_pencils - given_pencils = remaining_pencils :=
by
  sorry

end Josh_pencils_left_l2338_233870


namespace weight_of_7th_person_l2338_233831

/--
There are 6 people in the elevator with an average weight of 152 lbs.
Another person enters the elevator, increasing the average weight to 151 lbs.
Prove that the weight of the 7th person is 145 lbs.
-/
theorem weight_of_7th_person
  (W : ℕ) (X : ℕ) (h1 : W / 6 = 152) (h2 : (W + X) / 7 = 151) :
  X = 145 :=
sorry

end weight_of_7th_person_l2338_233831


namespace pyramid_height_correct_l2338_233802

noncomputable def pyramid_height (a α : ℝ) : ℝ :=
  a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))

theorem pyramid_height_correct (a α : ℝ) (hα : α ≠ 0 ∧ α ≠ π) :
  ∃ m : ℝ, m = pyramid_height a α := 
by
  use a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))
  sorry

end pyramid_height_correct_l2338_233802


namespace infinite_geometric_series_sum_l2338_233897

theorem infinite_geometric_series_sum :
  let a := (1 : ℝ) / 4
  let r := (1 : ℝ) / 2
  ∑' n : ℕ, a * (r ^ n) = 1 / 2 := by
  sorry

end infinite_geometric_series_sum_l2338_233897


namespace case_b_conditions_l2338_233823

-- Definition of the polynomial
def polynomial (p q x : ℝ) : ℝ := x^2 + p * x + q

-- Main theorem
theorem case_b_conditions (p q: ℝ) (x1 x2: ℝ) (hx1: x1 ≤ 0) (hx2: x2 ≥ 2) :
    q ≤ 0 ∧ 2 * p + q + 4 ≤ 0 :=
sorry

end case_b_conditions_l2338_233823


namespace percentage_of_red_shirts_l2338_233888

theorem percentage_of_red_shirts
  (Total : ℕ) 
  (P_blue P_green : ℝ) 
  (N_other : ℕ)
  (H_Total : Total = 600)
  (H_P_blue : P_blue = 0.45) 
  (H_P_green : P_green = 0.15) 
  (H_N_other : N_other = 102) :
  ( (Total - (P_blue * Total + P_green * Total + N_other)) / Total ) * 100 = 23 := by
  sorry

end percentage_of_red_shirts_l2338_233888


namespace simple_interest_rate_l2338_233859

theorem simple_interest_rate (P : ℝ) (T : ℝ) (hT : T = 15)
  (doubles_in_15_years : ∃ R : ℝ, (P * 2 = P + (P * R * T) / 100)) :
  ∃ R : ℝ, R = 6.67 := 
by
  sorry

end simple_interest_rate_l2338_233859


namespace football_goals_even_more_probable_l2338_233834

-- Define the problem statement and conditions
variable (p_1 : ℝ) (h₀ : 0 ≤ p_1 ∧ p_1 ≤ 1) (h₁ : q_1 = 1 - p_1)

-- Define even and odd goal probabilities for the total match
def p : ℝ := p_1^2 + (1 - p_1)^2
def q : ℝ := 2 * p_1 * (1 - p_1)

-- The main statement to prove
theorem football_goals_even_more_probable (h₂ : q_1 = 1 - p_1) : p_1^2 + (1 - p_1)^2 ≥ 2 * p_1 * (1 - p_1) :=
  sorry

end football_goals_even_more_probable_l2338_233834


namespace bells_toll_together_l2338_233852

theorem bells_toll_together {a b c d : ℕ} (h1 : a = 9) (h2 : b = 10) (h3 : c = 14) (h4 : d = 18) :
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d) = 630 :=
by
  sorry

end bells_toll_together_l2338_233852


namespace right_triangle_unique_perimeter_18_l2338_233807

theorem right_triangle_unique_perimeter_18 :
  ∃! (a b c : ℤ), a^2 + b^2 = c^2 ∧ a + b + c = 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end right_triangle_unique_perimeter_18_l2338_233807


namespace eq_rectangular_eq_of_polar_eq_max_m_value_l2338_233841

def polar_to_rectangular (ρ θ : ℝ) : Prop := (ρ = 4 * Real.cos θ) → ∀ x y : ℝ, ρ^2 = x^2 + y^2

theorem eq_rectangular_eq_of_polar_eq (ρ θ : ℝ) :
  polar_to_rectangular ρ θ → ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
sorry

def max_m_condition (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → |4 + 2 * m| / Real.sqrt 5 ≤ 2

theorem max_m_value :
  (max_m_condition (Real.sqrt 5 - 2)) :=
sorry

end eq_rectangular_eq_of_polar_eq_max_m_value_l2338_233841


namespace cross_section_area_of_truncated_pyramid_l2338_233808

-- Given conditions
variables (a b : ℝ) (α : ℝ)
-- Constraints
variable (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2)

-- Proposed theorem
theorem cross_section_area_of_truncated_pyramid (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2) :
    ∃ area : ℝ, area = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) :=
sorry

end cross_section_area_of_truncated_pyramid_l2338_233808


namespace case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l2338_233840

noncomputable def solution_set (m x : ℝ) : Prop :=
  x^2 + (m-1) * x - m > 0

theorem case_m_eq_neg_1 (x : ℝ) :
  solution_set (-1) x ↔ x ≠ 1 :=
sorry

theorem case_m_gt_neg_1 (m x : ℝ) (hm : m > -1) :
  solution_set m x ↔ (x < -m ∨ x > 1) :=
sorry

theorem case_m_lt_neg_1 (m x : ℝ) (hm : m < -1) :
  solution_set m x ↔ (x < 1 ∨ x > -m) :=
sorry

end case_m_eq_neg_1_case_m_gt_neg_1_case_m_lt_neg_1_l2338_233840


namespace simplify_eval_expression_l2338_233863

theorem simplify_eval_expression (a b : ℤ) (h₁ : a = 2) (h₂ : b = -1) : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 2 * a * b) / (-2 * b) = -7 := 
by 
  sorry

end simplify_eval_expression_l2338_233863


namespace determine_c_l2338_233822

theorem determine_c (c y : ℝ) : (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 := by
  sorry

end determine_c_l2338_233822


namespace first_part_amount_l2338_233885

-- Given Definitions
def total_amount : ℝ := 3200
def interest_rate_part1 : ℝ := 0.03
def interest_rate_part2 : ℝ := 0.05
def total_interest : ℝ := 144

-- The problem to be proven
theorem first_part_amount : 
  ∃ (x : ℝ), 0.03 * x + 0.05 * (3200 - x) = 144 ∧ x = 800 :=
by
  sorry

end first_part_amount_l2338_233885


namespace original_price_of_suit_l2338_233876

theorem original_price_of_suit (P : ℝ) (h : 0.96 * P = 144) : P = 150 :=
sorry

end original_price_of_suit_l2338_233876


namespace find_multiplier_l2338_233813

theorem find_multiplier 
  (x : ℝ)
  (number : ℝ)
  (condition1 : 4 * number + x * number = 55)
  (condition2 : number = 5.0) :
  x = 7 :=
by
  sorry

end find_multiplier_l2338_233813


namespace geometric_sequence_problem_l2338_233804

variable {a : ℕ → ℝ} -- Considering the sequence is a real number sequence
variable {q : ℝ} -- Common ratio

-- Conditions
axiom a2a6_eq_16 : a 2 * a 6 = 16
axiom a4_plus_a8_eq_8 : a 4 + a 8 = 8

-- Geometric sequence definition
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_problem : a 20 / a 10 = 1 :=
  by
  sorry

end geometric_sequence_problem_l2338_233804


namespace probability_calculation_l2338_233836

noncomputable def probability_in_ellipsoid : ℝ :=
  let prism_volume := (2 - (-2)) * (1 - (-1)) * (1 - (-1))
  let ellipsoid_volume := (4 * Real.pi / 3) * 1 * 2 * 2
  ellipsoid_volume / prism_volume

theorem probability_calculation :
  probability_in_ellipsoid = Real.pi / 3 :=
sorry

end probability_calculation_l2338_233836


namespace complex_pow_diff_zero_l2338_233865

theorem complex_pow_diff_zero {i : ℂ} (h : i^2 = -1) : (2 + i)^(12) - (2 - i)^(12) = 0 := by
  sorry

end complex_pow_diff_zero_l2338_233865


namespace solve_for_x_l2338_233814

theorem solve_for_x (x : ℝ) (h : (1 / 4) + (5 / x) = (12 / x) + (1 / 15)) : x = 420 / 11 := 
by
  sorry

end solve_for_x_l2338_233814


namespace pauls_plumbing_hourly_charge_l2338_233891

theorem pauls_plumbing_hourly_charge :
  ∀ P : ℕ,
  (55 + 4 * P = 75 + 4 * 30) → 
  P = 35 :=
by
  intros P h
  sorry

end pauls_plumbing_hourly_charge_l2338_233891


namespace inverse_proportion_relation_l2338_233892

theorem inverse_proportion_relation :
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  y2 < y1 ∧ y1 < y3 :=
by
  -- Variable definitions according to conditions
  let y1 := 3 / (-2)
  let y2 := 3 / (-1)
  let y3 := 3 / 1
  -- Proof steps go here (not required for the statement)
  -- Since proof steps are omitted, we use sorry to indicate it
  sorry

end inverse_proportion_relation_l2338_233892


namespace age_of_hospital_l2338_233828

theorem age_of_hospital (grant_current_age : ℕ) (future_ratio : ℚ)
                        (grant_future_age : grant_current_age + 5 = 30)
                        (hospital_age_ratio : future_ratio = 2 / 3) :
                        (grant_current_age = 25) → 
                        (grant_current_age + 5 = future_ratio * (grant_current_age + 5 + 5)) →
                        (grant_current_age + 5 + 5 - 5 = 40) :=
by
  sorry

end age_of_hospital_l2338_233828


namespace no_consecutive_positive_integers_have_sum_75_l2338_233871

theorem no_consecutive_positive_integers_have_sum_75 :
  ∀ n a : ℕ, (n ≥ 2) → (a ≥ 1) → (n * (2 * a + n - 1) = 150) → False :=
by
  intros n a hn ha hsum
  sorry

end no_consecutive_positive_integers_have_sum_75_l2338_233871


namespace radius_of_tangent_circle_l2338_233889

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l2338_233889


namespace intercept_sum_l2338_233895

theorem intercept_sum (x y : ℤ) (h1 : 0 ≤ x) (h2 : x < 42) (h3 : 0 ≤ y) (h4 : y < 42)
  (h : 5 * x ≡ 3 * y - 2 [ZMOD 42]) : (x + y) = 36 :=
by
  sorry

end intercept_sum_l2338_233895


namespace find_a_l2338_233820

noncomputable def p (a : ℝ) : Prop := 3 < a ∧ a < 7/2
noncomputable def q (a : ℝ) : Prop := a > 3 ∧ a ≠ 7/2
theorem find_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) (hpq : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7/2 :=
sorry

end find_a_l2338_233820


namespace eleven_twelve_divisible_by_133_l2338_233894

theorem eleven_twelve_divisible_by_133 (n : ℕ) (h : n > 0) : 133 ∣ (11^(n+2) + 12^(2*n+1)) := 
by 
  sorry

end eleven_twelve_divisible_by_133_l2338_233894


namespace aunt_may_morning_milk_l2338_233835

-- Defining the known quantities as variables
def evening_milk : ℕ := 380
def sold_milk : ℕ := 612
def leftover_milk : ℕ := 15
def milk_left : ℕ := 148

-- Main statement to be proven
theorem aunt_may_morning_milk (M : ℕ) :
  M + evening_milk + leftover_milk - sold_milk = milk_left → M = 365 := 
by {
  -- Skipping the proof
  sorry
}

end aunt_may_morning_milk_l2338_233835


namespace invitations_per_package_l2338_233845

-- Definitions based on conditions in the problem.
def numPackages : Nat := 5
def totalInvitations : Nat := 45

-- Definition of the problem and proof statement.
theorem invitations_per_package :
  totalInvitations / numPackages = 9 :=
by
  sorry

end invitations_per_package_l2338_233845


namespace find_xyz_sum_l2338_233838

theorem find_xyz_sum (x y z : ℝ) (h1 : x^2 + x * y + y^2 = 108)
                               (h2 : y^2 + y * z + z^2 = 49)
                               (h3 : z^2 + z * x + x^2 = 157) :
  x * y + y * z + z * x = 84 :=
sorry

end find_xyz_sum_l2338_233838


namespace proof_AC_time_l2338_233825

noncomputable def A : ℝ := 1/10
noncomputable def B : ℝ := sorry
noncomputable def C : ℝ := 1/30

def rate_A_B (A B : ℝ) := A + B = 1/6
def rate_B_C (B C : ℝ) := B + C = 1/10
def rate_A_B_C (A B C : ℝ) := A + B + C = 1/5

theorem proof_AC_time {A B C : ℝ} (h1 : rate_A_B A B) (h2 : rate_B_C B C) (h3 : rate_A_B_C A B C) : 
  (1 : ℝ) / (A + C) = 7.5 :=
sorry

end proof_AC_time_l2338_233825


namespace train_length_l2338_233866

theorem train_length 
  (bridge_length train_length time_seconds v : ℝ)
  (h1 : bridge_length = 300)
  (h2 : time_seconds = 36)
  (h3 : v = 40) :
  (train_length = v * time_seconds - bridge_length) →
  (train_length = 1140) := by
  -- solve in a few lines
  -- This proof is omitted for the purpose of this task
  sorry

end train_length_l2338_233866


namespace number_mul_five_l2338_233868

theorem number_mul_five (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 :=
by
  sorry

end number_mul_five_l2338_233868


namespace problem_a_b_squared_l2338_233851

theorem problem_a_b_squared {a b : ℝ} (h1 : a + 3 = (b-1)^2) (h2 : b + 3 = (a-1)^2) (h3 : a ≠ b) : a^2 + b^2 = 10 :=
by
  sorry

end problem_a_b_squared_l2338_233851


namespace fraction_identity_proof_l2338_233850

theorem fraction_identity_proof (a b : ℝ) (h : 2 / a - 1 / b = 1 / (a + 2 * b)) :
  4 / (a ^ 2) - 1 / (b ^ 2) = 1 / (a * b) :=
by
  sorry

end fraction_identity_proof_l2338_233850


namespace maria_cookies_left_l2338_233861

def maria_cookies (initial: ℕ) (to_friend: ℕ) (to_family_divisor: ℕ) (eats: ℕ) : ℕ :=
  (initial - to_friend) / to_family_divisor - eats

theorem maria_cookies_left (h : maria_cookies 19 5 2 2 = 5): true :=
by trivial

end maria_cookies_left_l2338_233861


namespace twenty_four_points_game_l2338_233882

theorem twenty_four_points_game :
  let a := (-6 : ℚ)
  let b := (3 : ℚ)
  let c := (4 : ℚ)
  let d := (10 : ℚ)
  3 * (d - a + c) = 24 := 
by
  sorry

end twenty_four_points_game_l2338_233882


namespace average_temperature_l2338_233833

def temperatures :=
  ∃ T_tue T_wed T_thu : ℝ,
    (44 + T_tue + T_wed + T_thu) / 4 = 48 ∧
    (T_tue + T_wed + T_thu + 36) / 4 = 46

theorem average_temperature :
  temperatures :=
by
  sorry

end average_temperature_l2338_233833


namespace cannot_achieve_90_cents_l2338_233899

theorem cannot_achieve_90_cents :
  ∀ (p n d q : ℕ),        -- p: number of pennies, n: number of nickels, d: number of dimes, q: number of quarters
  (p + n + d + q = 6) →   -- exactly six coins chosen
  (p ≤ 4 ∧ n ≤ 4 ∧ d ≤ 4 ∧ q ≤ 4) →  -- no more than four of each kind of coin
  (p + 5 * n + 10 * d + 25 * q ≠ 90) -- total value should not equal 90 cents
:= by
  sorry

end cannot_achieve_90_cents_l2338_233899


namespace lines_perpendicular_to_same_line_l2338_233800

-- Definitions for lines and relationship types
structure Line := (name : String)
inductive RelType
| parallel 
| intersect
| skew

-- Definition stating two lines are perpendicular to the same line
def perpendicular_to_same_line (l1 l2 l3 : Line) : Prop :=
  -- (dot product or a similar condition could be specified, leaving abstract here)
  sorry

-- Theorem statement
theorem lines_perpendicular_to_same_line (l1 l2 l3 : Line) (h1 : perpendicular_to_same_line l1 l2 l3) : 
  RelType :=
by
  -- Proof to be filled in
  sorry

end lines_perpendicular_to_same_line_l2338_233800


namespace part_I_part_II_l2338_233886

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 1

theorem part_I {a : ℝ} (ha : a = 2) :
  { x : ℝ | f x a ≥ 4 - abs (x - 4)} = { x | x ≥ 11 / 2 ∨ x ≤ 1 / 2 } := 
by 
  sorry

theorem part_II {a : ℝ} (h : { x : ℝ | abs (f (2 * x + a) a - 2 * f x a) ≤ 1 } = 
      { x | 1 / 2 ≤ x ∧ x ≤ 1 }) : 
  a = 2 := 
by 
  sorry

end part_I_part_II_l2338_233886


namespace calculate_f_of_f_of_f_30_l2338_233883

-- Define the function f (equivalent to $\#N = 0.5N + 2$)
def f (N : ℝ) : ℝ := 0.5 * N + 2

-- The proof statement
theorem calculate_f_of_f_of_f_30 : 
  f (f (f 30)) = 7.25 :=
by
  sorry

end calculate_f_of_f_of_f_30_l2338_233883


namespace totalPeaches_l2338_233843

-- Definition of conditions in the problem
def redPeaches : Nat := 4
def greenPeaches : Nat := 6
def numberOfBaskets : Nat := 1

-- Mathematical proof problem
theorem totalPeaches : numberOfBaskets * (redPeaches + greenPeaches) = 10 := by
  sorry

end totalPeaches_l2338_233843


namespace max_quartets_in_5x5_max_quartets_in_mxn_l2338_233832

def quartet (c : Nat) : Bool := 
  c > 0

theorem max_quartets_in_5x5 : ∃ q, q = 5 ∧ 
  quartet q := by
  sorry

theorem max_quartets_in_mxn 
  (m n : Nat) (Hmn : m > 0 ∧ n > 0) :
  (∃ q, q = (m * (n - 1)) / 4 ∧ quartet q) ∨ 
  (∃ q, q = (m * (n - 1) - 2) / 4 ∧ quartet q) := by
  sorry

end max_quartets_in_5x5_max_quartets_in_mxn_l2338_233832


namespace fraction_sum_59_l2338_233874

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end fraction_sum_59_l2338_233874


namespace expression_value_l2338_233898

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) : 
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 := 
by
  sorry

end expression_value_l2338_233898


namespace time_juan_ran_l2338_233873

variable (Distance Speed : ℝ)
variable (h1 : Distance = 80)
variable (h2 : Speed = 10)

theorem time_juan_ran : (Distance / Speed) = 8 := by
  sorry

end time_juan_ran_l2338_233873


namespace common_tangent_intersects_x_axis_at_point_A_l2338_233881

-- Define the ellipses using their equations
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def ellipse_C2 (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 1

-- The theorem stating the coordinates of the point where the common tangent intersects the x-axis
theorem common_tangent_intersects_x_axis_at_point_A :
  (∃ x : ℝ, (ellipse_C1 x 0 ∧ ellipse_C2 x 0) ↔ x = 4) :=
sorry

end common_tangent_intersects_x_axis_at_point_A_l2338_233881


namespace tree_planting_l2338_233812

/-- The city plans to plant 500 thousand trees. The original plan 
was to plant x thousand trees per day. Due to volunteers, the actual number 
of trees planted per day exceeds the original plan by 30%. As a result, 
the task is completed 2 days ahead of schedule. Prove the equation. -/
theorem tree_planting
    (x : ℝ) 
    (hx : x > 0) : 
    (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
sorry

end tree_planting_l2338_233812


namespace probability_two_identical_l2338_233846

-- Define the number of ways to choose 3 out of 4 attractions
def choose_3_out_of_4 := Nat.choose 4 3

-- Define the total number of ways for both tourists to choose 3 attractions out of 4
def total_basic_events := choose_3_out_of_4 * choose_3_out_of_4

-- Define the number of ways to choose exactly 2 identical attractions
def ways_to_choose_2_identical := Nat.choose 4 2 * Nat.choose 2 1 * Nat.choose 1 1

-- The probability that they choose exactly 2 identical attractions
def probability : ℚ := ways_to_choose_2_identical / total_basic_events

-- Prove that this probability is 3/4
theorem probability_two_identical : probability = 3 / 4 := by
  have h1 : choose_3_out_of_4 = 4 := by sorry
  have h2 : total_basic_events = 16 := by sorry
  have h3 : ways_to_choose_2_identical = 12 := by sorry
  rw [probability, h2, h3]
  norm_num

end probability_two_identical_l2338_233846


namespace librarians_all_work_together_l2338_233801

/-- Peter works every 5 days -/
def Peter_days := 5

/-- Quinn works every 8 days -/
def Quinn_days := 8

/-- Rachel works every 10 days -/
def Rachel_days := 10

/-- Sam works every 14 days -/
def Sam_days := 14

/-- Least common multiple of the intervals at which Peter, Quinn, Rachel, and Sam work -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem librarians_all_work_together : LCM (LCM (LCM Peter_days Quinn_days) Rachel_days) Sam_days = 280 :=
  by
  sorry

end librarians_all_work_together_l2338_233801


namespace arithmetic_sequence_solution_l2338_233880

-- Definitions of a, b, c, and d in terms of d and sequence difference
def is_in_arithmetic_sequence (a b c d : ℝ) (diff : ℝ) : Prop :=
  a + diff = b ∧ b + diff = c ∧ c + diff = d

-- Conditions
def pos_real_sequence (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

def product_condition (a b c d : ℝ) (prod : ℝ) : Prop :=
  a * b * c * d = prod

-- The resulting value of d
def d_value_as_fraction (d : ℝ) : Prop :=
  d = (3 + Real.sqrt 95) / (Real.sqrt 2)

-- Proof statement
theorem arithmetic_sequence_solution :
  ∃ a b c d : ℝ, pos_real_sequence a b c d ∧ 
                 is_in_arithmetic_sequence a b c d (Real.sqrt 2) ∧ 
                 product_condition a b c d 2021 ∧ 
                 d_value_as_fraction d :=
sorry

end arithmetic_sequence_solution_l2338_233880


namespace sin_three_pi_over_two_l2338_233869

theorem sin_three_pi_over_two : Real.sin (3 * Real.pi / 2) = -1 :=
by
  sorry

end sin_three_pi_over_two_l2338_233869


namespace eccentricity_range_l2338_233860

variable {a b c : ℝ} (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (e : ℝ)

-- Assume a > 0, b > 0, and the eccentricity of the hyperbola is given by c = e * a.
variable (a_pos : 0 < a) (b_pos : 0 < b) (hyperbola : (P.1 / a)^2 - (P.2 / b)^2 = 1)
variable (on_right_branch : P.1 > 0)
variable (foci_condition : dist P F₁ = 4 * dist P F₂)
variable (eccentricity_def : c = e * a)

theorem eccentricity_range : 1 < e ∧ e ≤ 5 / 3 := by
  sorry

end eccentricity_range_l2338_233860


namespace min_value_of_a_l2338_233879

theorem min_value_of_a (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x + y) * (1/x + a/y) ≥ 16) : a ≥ 9 :=
sorry

end min_value_of_a_l2338_233879


namespace max_books_single_student_l2338_233810

theorem max_books_single_student (total_students : ℕ) (students_0_books : ℕ) (students_1_book : ℕ) (students_2_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 20 →
  students_0_books = 3 →
  students_1_book = 9 →
  students_2_books = 4 →
  avg_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 14 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_books_single_student_l2338_233810


namespace penny_money_left_is_5_l2338_233855

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l2338_233855


namespace quadratic_roots_are_correct_l2338_233896

theorem quadratic_roots_are_correct (x: ℝ) : 
    (x^2 + x - 1 = 0) ↔ (x = (-1 + Real.sqrt 5) / 2) ∨ (x = (-1 - Real.sqrt 5) / 2) := 
by sorry

end quadratic_roots_are_correct_l2338_233896


namespace wage_difference_l2338_233815

theorem wage_difference (P Q H: ℝ) (h1: P = 1.5 * Q) (h2: P * H = 300) (h3: Q * (H + 10) = 300) : P - Q = 5 :=
by
  sorry

end wage_difference_l2338_233815


namespace integer_product_is_192_l2338_233858

theorem integer_product_is_192 (A B C : ℤ)
  (h1 : A + B + C = 33)
  (h2 : C = 3 * B)
  (h3 : A = C - 23) :
  A * B * C = 192 :=
sorry

end integer_product_is_192_l2338_233858


namespace infinite_series_sum_l2338_233821

theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / ((4 * n - 1)^3 * (4 * n + 3)^3)) = 1 / 972 := 
by 
  sorry

end infinite_series_sum_l2338_233821


namespace min_value_x_fraction_l2338_233827

theorem min_value_x_fraction (x : ℝ) (h : x > 1) : 
  ∃ m, m = 3 ∧ ∀ y > 1, y + 1 / (y - 1) ≥ m :=
by
  sorry

end min_value_x_fraction_l2338_233827


namespace perfect_square_trinomial_m_eq_6_or_neg6_l2338_233819

theorem perfect_square_trinomial_m_eq_6_or_neg6
  (m : ℤ) :
  (∃ a : ℤ, x * x + m * x + 9 = (x + a) * (x + a)) → (m = 6 ∨ m = -6) :=
by
  sorry

end perfect_square_trinomial_m_eq_6_or_neg6_l2338_233819


namespace distribution_ways_l2338_233816

def count_distributions (n : ℕ) (k : ℕ) : ℕ :=
-- Calculation for count distributions will be implemented here
sorry

theorem distribution_ways (items bags : ℕ) (cond : items = 6 ∧ bags = 3):
  count_distributions items bags = 75 :=
by
  -- Proof would be implemented here
  sorry

end distribution_ways_l2338_233816


namespace john_annual_profit_is_1800_l2338_233875

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l2338_233875


namespace radius_of_cone_is_8_l2338_233826

noncomputable def r_cylinder := 8 -- cm
noncomputable def h_cylinder := 2 -- cm
noncomputable def h_cone := 6 -- cm

theorem radius_of_cone_is_8 :
  exists (r_cone : ℝ), r_cone = 8 ∧ π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone :=
by
  let r_cone := 8
  have eq_volumes : π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone := 
    sorry
  exact ⟨r_cone, by simp, eq_volumes⟩

end radius_of_cone_is_8_l2338_233826


namespace am_gm_inequality_l2338_233864

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (b * c)) + (b^3 / (c * a)) + (c^3 / (a * b)) ≥ a + b + c :=
  sorry

end am_gm_inequality_l2338_233864


namespace percentage_failed_in_Hindi_l2338_233857

-- Let Hindi_failed denote the percentage of students who failed in Hindi.
-- Let English_failed denote the percentage of students who failed in English.
-- Let Both_failed denote the percentage of students who failed in both Hindi and English.
-- Let Both_passed denote the percentage of students who passed in both subjects.

variables (Hindi_failed English_failed Both_failed Both_passed : ℝ)
  (H_condition1 : English_failed = 44)
  (H_condition2 : Both_failed = 22)
  (H_condition3 : Both_passed = 44)

theorem percentage_failed_in_Hindi:
  Hindi_failed = 34 :=
by 
  -- Proof goes here
  sorry

end percentage_failed_in_Hindi_l2338_233857


namespace polynomial_factor_l2338_233848

theorem polynomial_factor (x : ℝ) : (x^2 - 4*x + 4) ∣ (x^4 + 16) :=
sorry

end polynomial_factor_l2338_233848


namespace conjecture_a_n_l2338_233844

noncomputable def a_n (n : ℕ) : ℚ := (2^n - 1) / 2^(n-1)

noncomputable def S_n (n : ℕ) : ℚ := 2 * n - a_n n

theorem conjecture_a_n (n : ℕ) (h : n > 0) : a_n n = (2^n - 1) / 2^(n-1) :=
by 
  sorry

end conjecture_a_n_l2338_233844


namespace arithmetic_sequence_fifth_term_l2338_233862

variable (a d : ℕ)

-- Conditions
def condition1 := (a + d) + (a + 3 * d) = 10
def condition2 := a + (a + 2 * d) = 8

-- Fifth term calculation
def fifth_term := a + 4 * d

theorem arithmetic_sequence_fifth_term (h1 : condition1 a d) (h2 : condition2 a d) : fifth_term a d = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l2338_233862


namespace find_n_l2338_233887

theorem find_n (n : ℕ) (composite_n : n > 1 ∧ ¬Prime n) : 
  ((∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ 1 < d + 1 ∧ d + 1 < m) ↔ 
    (n = 4 ∨ n = 8)) :=
by sorry

end find_n_l2338_233887


namespace common_internal_tangent_length_l2338_233818

-- Definitions based on given conditions
def center_distance : ℝ := 50
def radius_small : ℝ := 7
def radius_large : ℝ := 10

-- Target theorem
theorem common_internal_tangent_length :
  let AB := center_distance
  let BE := radius_small + radius_large 
  let AE := Real.sqrt (AB^2 - BE^2)
  AE = Real.sqrt 2211 :=
by
  sorry

end common_internal_tangent_length_l2338_233818
