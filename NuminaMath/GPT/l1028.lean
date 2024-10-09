import Mathlib

namespace passengers_at_18_max_revenue_l1028_102865

noncomputable def P (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then 500 - 4 * (20 - t)^2 else
if 20 ≤ t ∧ t ≤ 30 then 500 else 0

noncomputable def Q (t : ℝ) : ℝ :=
if 10 ≤ t ∧ t < 20 then -8 * t - (1800 / t) + 320 else
if 20 ≤ t ∧ t ≤ 30 then 1400 / t else 0

-- 1. Prove P(18) = 484
theorem passengers_at_18 : P 18 = 484 := sorry

-- 2. Prove that Q(t) is maximized at t = 15 with a maximum value of 80
theorem max_revenue : ∃ t, Q t = 80 ∧ t = 15 := sorry

end passengers_at_18_max_revenue_l1028_102865


namespace cylinder_surface_area_proof_l1028_102870

noncomputable def sphere_volume := (500 * Real.pi) / 3
noncomputable def cylinder_base_diameter := 8
noncomputable def cylinder_surface_area := 80 * Real.pi

theorem cylinder_surface_area_proof :
  ∀ (R : ℝ) (r h : ℝ), 
    (4 * Real.pi / 3) * R^3 = (500 * Real.pi) / 3 → -- sphere volume condition
    2 * r = cylinder_base_diameter →               -- base diameter condition
    r * r + (h / 2)^2 = R^2 →                      -- Pythagorean theorem (half height)
    2 * Real.pi * r * h + 2 * Real.pi * r^2 = cylinder_surface_area := -- surface area formula
by
  intros R r h sphere_vol_cond base_diameter_cond pythagorean_cond
  sorry

end cylinder_surface_area_proof_l1028_102870


namespace number_of_impossible_d_vals_is_infinite_l1028_102806

theorem number_of_impossible_d_vals_is_infinite
  (t_1 t_2 s d : ℕ)
  (h1 : 2 * t_1 + t_2 - 4 * s = 4041)
  (h2 : t_1 = s + 2 * d)
  (h3 : t_2 = s + d)
  (h4 : 4 * s > 0) :
  ∀ n : ℕ, n ≠ 808 * 5 ↔ ∃ d, d > 0 ∧ d ≠ n :=
sorry

end number_of_impossible_d_vals_is_infinite_l1028_102806


namespace percentage_of_men_l1028_102816

variable (M : ℝ)

theorem percentage_of_men (h1 : 0.20 * M + 0.40 * (1 - M) = 0.33) : 
  M = 0.35 :=
sorry

end percentage_of_men_l1028_102816


namespace problem_integer_and_decimal_parts_eq_2_l1028_102872

theorem problem_integer_and_decimal_parts_eq_2 :
  let x := 3
  let y := 2 - Real.sqrt 3
  2 * x^3 - (y^3 + 1 / y^3) = 2 :=
by
  sorry

end problem_integer_and_decimal_parts_eq_2_l1028_102872


namespace coffee_shop_spending_l1028_102885

variable (R S : ℝ)

theorem coffee_shop_spending (h1 : S = 0.60 * R) (h2 : R = S + 12.50) : R + S = 50 :=
by
  sorry

end coffee_shop_spending_l1028_102885


namespace verify_total_amount_l1028_102819

noncomputable def total_withdrawable_amount (a r : ℝ) : ℝ :=
  a / r * ((1 + r) ^ 5 - (1 + r))

theorem verify_total_amount (a r : ℝ) (h_r_nonzero : r ≠ 0) :
  total_withdrawable_amount a r = a / r * ((1 + r)^5 - (1 + r)) :=
by
  sorry

end verify_total_amount_l1028_102819


namespace find_tangency_segments_equal_l1028_102815

-- Conditions of the problem as a theorem statement
theorem find_tangency_segments_equal (AB BC CD DA : ℝ) (x y : ℝ)
    (h1 : AB = 80)
    (h2 : BC = 140)
    (h3 : CD = 100)
    (h4 : DA = 120)
    (h5 : x + y = CD)
    (tangency_property : |x - y| = 0) :
  |x - y| = 0 :=
sorry

end find_tangency_segments_equal_l1028_102815


namespace cos_inequality_range_l1028_102837

theorem cos_inequality_range (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 2 * Real.pi) (h₃ : Real.cos x ≤ 1 / 2) :
  x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 3) := 
sorry

end cos_inequality_range_l1028_102837


namespace problem_statement_l1028_102820

noncomputable def h (y : ℂ) : ℂ := y^5 - y^3 + 1
noncomputable def p (y : ℂ) : ℂ := y^2 - 3

theorem problem_statement (y_1 y_2 y_3 y_4 y_5 : ℂ) (hroots : ∀ y, h y = 0 ↔ y = y_1 ∨ y = y_2 ∨ y = y_3 ∨ y = y_4 ∨ y = y_5) :
  (p y_1) * (p y_2) * (p y_3) * (p y_4) * (p y_5) = 22 :=
by
  sorry

end problem_statement_l1028_102820


namespace balls_into_boxes_l1028_102882

theorem balls_into_boxes :
  let n := 7 -- number of balls
  let k := 3 -- number of boxes
  let ways := Nat.choose (n + k - 1) (k - 1)
  ways = 36 :=
by
  sorry

end balls_into_boxes_l1028_102882


namespace Shawna_situps_l1028_102829

theorem Shawna_situps :
  ∀ (goal_per_day : ℕ) (total_days : ℕ) (tuesday_situps : ℕ) (wednesday_situps : ℕ),
  goal_per_day = 30 →
  total_days = 3 →
  tuesday_situps = 19 →
  wednesday_situps = 59 →
  (goal_per_day * total_days) - (tuesday_situps + wednesday_situps) = 12 :=
by
  intros goal_per_day total_days tuesday_situps wednesday_situps
  sorry

end Shawna_situps_l1028_102829


namespace age_solution_l1028_102876

theorem age_solution :
  ∃ me you : ℕ, me + you = 63 ∧ 
  ∃ x : ℕ, me = 2 * x ∧ you = x ∧ me = 36 ∧ you = 27 :=
by
  sorry

end age_solution_l1028_102876


namespace existence_of_inf_polynomials_l1028_102807

noncomputable def P_xy_defined (P : ℝ→ℝ) (x y z : ℝ) :=
  P x ^ 2 + P y ^ 2 + P z ^ 2 + 2 * P x * P y * P z = 1

theorem existence_of_inf_polynomials (x y z : ℝ) (P : ℕ → ℝ → ℝ) :
  (x^2 + y^2 + z^2 + 2 * x * y * z = 1) →
  (∀ n, P (n+1) = P n ∘ P n) →
  P_xy_defined (P 0) x y z →
  ∀ n, P_xy_defined (P n) x y z :=
by
  intros h1 h2 h3
  sorry

end existence_of_inf_polynomials_l1028_102807


namespace geometric_sequence_common_ratio_simple_sequence_general_term_l1028_102886

-- Question 1
theorem geometric_sequence_common_ratio (a_3 : ℝ) (S_3 : ℝ) (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) :
    q = -1 / 2 ∨ q = 1 :=
sorry

-- Question 2
theorem simple_sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h : ∀ n, S n = n^2) :
    ∀ n, a n = S n - S (n - 1) → ∀ n, a n = 2 * n - 1 :=
sorry

end geometric_sequence_common_ratio_simple_sequence_general_term_l1028_102886


namespace new_pyramid_volume_l1028_102827

/-- Given an original pyramid with volume 40 cubic inches, where the length is doubled, 
    the width is tripled, and the height is increased by 50%, 
    prove that the volume of the new pyramid is 360 cubic inches. -/
theorem new_pyramid_volume (V : ℝ) (l w h : ℝ) 
  (h_volume : V = 1 / 3 * l * w * h) 
  (h_original : V = 40) : 
  (2 * l) * (3 * w) * (1.5 * h) / 3 = 360 :=
by
  sorry

end new_pyramid_volume_l1028_102827


namespace distinct_prime_factors_2310_l1028_102839

theorem distinct_prime_factors_2310 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ (S.card = 5) ∧ (S.prod id = 2310) := by
  sorry

end distinct_prime_factors_2310_l1028_102839


namespace only_positive_integer_cube_less_than_triple_l1028_102884

theorem only_positive_integer_cube_less_than_triple (n : ℕ) (h : 0 < n ∧ n^3 < 3 * n) : n = 1 :=
sorry

end only_positive_integer_cube_less_than_triple_l1028_102884


namespace linear_eq_m_val_l1028_102838

theorem linear_eq_m_val (m : ℤ) (x : ℝ) : (5 * x ^ (m - 2) + 1 = 0) → (m = 3) :=
by
  sorry

end linear_eq_m_val_l1028_102838


namespace num_of_integers_abs_leq_six_l1028_102847

theorem num_of_integers_abs_leq_six (x : ℤ) : 
  (|x - 3| ≤ 6) → ∃ (n : ℕ), n = 13 := 
by 
  sorry

end num_of_integers_abs_leq_six_l1028_102847


namespace percentage_paid_l1028_102896

theorem percentage_paid (X Y : ℝ) (h_sum : X + Y = 572) (h_Y : Y = 260) : (X / Y) * 100 = 120 :=
by
  -- We'll prove this result by using the conditions and solving for X.
  sorry

end percentage_paid_l1028_102896


namespace total_hits_and_misses_l1028_102880

theorem total_hits_and_misses (h : ℕ) (m : ℕ) (hc : m = 3 * h) (hm : m = 50) : h + m = 200 :=
by
  sorry

end total_hits_and_misses_l1028_102880


namespace speed_of_stream_l1028_102817

theorem speed_of_stream (v : ℝ) :
  (∀ s : ℝ, s = 3 → (3 + v) / (3 - v) = 2) → v = 1 :=
by 
  intro h
  sorry

end speed_of_stream_l1028_102817


namespace prism_volume_l1028_102801

theorem prism_volume (x : ℝ) (L W H : ℝ) (hL : L = 2 * x) (hW : W = x) (hH : H = 1.5 * x) 
  (hedges_sum : 4 * L + 4 * W + 4 * H = 72) : 
  L * W * H = 192 := 
by
  sorry

end prism_volume_l1028_102801


namespace least_integer_value_of_x_l1028_102834

theorem least_integer_value_of_x (x : ℤ) (h : 3 * |x| + 4 < 19) : x = -4 :=
by sorry

end least_integer_value_of_x_l1028_102834


namespace trader_cloth_sale_l1028_102859

theorem trader_cloth_sale (total_SP : ℕ) (profit_per_meter : ℕ) (cost_per_meter : ℕ) (SP_per_meter : ℕ)
  (h1 : total_SP = 8400) (h2 : profit_per_meter = 12) (h3 : cost_per_meter = 128) (h4 : SP_per_meter = cost_per_meter + profit_per_meter) :
  ∃ (x : ℕ), SP_per_meter * x = total_SP ∧ x = 60 :=
by
  -- We will skip the proof using sorry
  sorry

end trader_cloth_sale_l1028_102859


namespace b_share_1500_l1028_102899

theorem b_share_1500 (total_amount : ℕ) (parts_A parts_B parts_C : ℕ)
  (h_total_amount : total_amount = 4500)
  (h_ratio : (parts_A, parts_B, parts_C) = (2, 3, 4)) :
  parts_B * (total_amount / (parts_A + parts_B + parts_C)) = 1500 :=
by
  sorry

end b_share_1500_l1028_102899


namespace isosceles_triangle_vertex_angle_l1028_102879

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_sum : A + B + C = 180) (h_one_angle : A = 50 ∨ B = 50 ∨ C = 50) :
  A = 80 ∨ B = 80 ∨ C = 80 ∨ A = 50 ∨ B = 50 ∨ C = 50 :=
by
  sorry

end isosceles_triangle_vertex_angle_l1028_102879


namespace simplify_complex_expression_l1028_102877

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) - 2 * i * (3 - 4 * i) = 20 - 20 * i := 
by
  sorry

end simplify_complex_expression_l1028_102877


namespace average_water_per_day_l1028_102808

variable (day1 : ℕ)
variable (day2 : ℕ)
variable (day3 : ℕ)

def total_water_over_three_days (d1 d2 d3 : ℕ) := d1 + d2 + d3

theorem average_water_per_day :
  day1 = 215 ->
  day2 = 215 + 76 ->
  day3 = 291 - 53 ->
  (total_water_over_three_days day1 day2 day3) / 3 = 248 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_water_per_day_l1028_102808


namespace percent_calculation_l1028_102803

theorem percent_calculation (Part Whole : ℝ) (h1 : Part = 120) (h2 : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  sorry

end percent_calculation_l1028_102803


namespace compute_b1c1_b2c2_b3c3_l1028_102881

theorem compute_b1c1_b2c2_b3c3 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -1 :=
by
  sorry

end compute_b1c1_b2c2_b3c3_l1028_102881


namespace sheets_of_paper_in_each_box_l1028_102898

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 30)
  (h2 : 2 * E = S)
  (h3 : 3 * E = S - 10) :
  S = 40 :=
by
  sorry

end sheets_of_paper_in_each_box_l1028_102898


namespace sum_of_x_and_y_l1028_102804

theorem sum_of_x_and_y (x y : ℚ) (h1 : 1/x + 1/y = 3) (h2 : 1/x - 1/y = -7) : x + y = -3/10 :=
by
  sorry

end sum_of_x_and_y_l1028_102804


namespace complete_the_square_l1028_102814

theorem complete_the_square (x : ℝ) (h : x^2 + 7 * x - 5 = 0) : (x + 7 / 2) ^ 2 = 69 / 4 :=
sorry

end complete_the_square_l1028_102814


namespace center_of_the_hyperbola_l1028_102843

def hyperbola_eq (x y : ℝ) : Prop := 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

structure Point where
  x : ℝ
  y : ℝ

def center_of_hyperbola_is (p : Point) : Prop :=
  hyperbola_eq (p.x + 3) (p.y + 4)

theorem center_of_the_hyperbola :
  ∀ x y : ℝ, hyperbola_eq x y → center_of_hyperbola_is {x := 3, y := 4} :=
by
  intros x y h
  sorry

end center_of_the_hyperbola_l1028_102843


namespace marc_watch_days_l1028_102848

theorem marc_watch_days (bought_episodes : ℕ) (watch_fraction : ℚ) (episodes_per_day : ℚ) (total_days : ℕ) : 
  bought_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  episodes_per_day = (50 : ℚ) * watch_fraction → 
  total_days = (bought_episodes : ℚ) / episodes_per_day →
  total_days = 10 := 
sorry

end marc_watch_days_l1028_102848


namespace baker_new_cakes_bought_l1028_102857

variable (total_cakes initial_sold sold_more_than_bought : ℕ)

def new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) : ℕ :=
  total_cakes - (initial_sold + sold_more_than_bought)

theorem baker_new_cakes_bought (total_cakes initial_sold sold_more_than_bought : ℕ) 
  (h1 : total_cakes = 170)
  (h2 : initial_sold = 78)
  (h3 : sold_more_than_bought = 47) :
  new_cakes_bought total_cakes initial_sold sold_more_than_bought = 78 :=
  sorry

end baker_new_cakes_bought_l1028_102857


namespace intersect_at_four_points_l1028_102890

theorem intersect_at_four_points (a : ℝ) : 
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = a^2) ∧ (p.2 = p.1^2 - a - 1) ∧ 
                 ∃ q : ℝ × ℝ, (q.1 ≠ p.1 ∧ q.2 ≠ p.2) ∧ (q.1^2 + q.2^2 = a^2) ∧ (q.2 = q.1^2 - a - 1) ∧ 
                 ∃ r : ℝ × ℝ, (r.1 ≠ p.1 ∧ r.1 ≠ q.1 ∧ r.2 ≠ p.2 ∧ r.2 ≠ q.2) ∧ (r.1^2 + r.2^2 = a^2) ∧ (r.2 = r.1^2 - a - 1) ∧
                 ∃ s : ℝ × ℝ, (s.1 ≠ p.1 ∧ s.1 ≠ q.1 ∧ s.1 ≠ r.1 ∧ s.2 ≠ p.2 ∧ s.2 ≠ q.2 ∧ s.2 ≠ r.2) ∧ (s.1^2 + s.2^2 = a^2) ∧ (s.2 = s.1^2 - a - 1))
  ↔ a > -1/2 := 
by 
  sorry

end intersect_at_four_points_l1028_102890


namespace farmer_land_l1028_102833

variable (A C G P T : ℝ)
variable (h1 : C = 0.90 * A)
variable (h2 : G = 0.10 * C)
variable (h3 : P = 0.80 * C)
variable (h4 : T = 450)
variable (h5 : C = G + P + T)

theorem farmer_land (A : ℝ) (h1 : C = 0.90 * A) (h2 : G = 0.10 * C) (h3 : P = 0.80 * C) (h4 : T = 450) (h5 : C = G + P + T) : A = 5000 := by
  sorry

end farmer_land_l1028_102833


namespace units_digit_of_m_squared_plus_two_to_m_is_3_l1028_102891

def m := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m_is_3 : (m^2 + 2^m) % 10 = 3 := 
by 
  sorry

end units_digit_of_m_squared_plus_two_to_m_is_3_l1028_102891


namespace constant_term_in_expansion_is_neg_42_l1028_102874

-- Define the general term formula for (x - 1/x)^8
def binomial_term (r : ℕ) : ℤ :=
  (Nat.choose 8 r) * (-1 : ℤ) ^ r

-- Define the constant term in the product expansion
def constant_term : ℤ := 
  binomial_term 4 - 2 * binomial_term 5 

-- Problem statement: Prove the constant term is -42
theorem constant_term_in_expansion_is_neg_42 :
  constant_term = -42 := 
sorry

end constant_term_in_expansion_is_neg_42_l1028_102874


namespace solve_eq_solve_ineq_l1028_102867

-- Proof Problem 1 statement
theorem solve_eq (x : ℝ) : (2 / (x + 3) - (x - 3) / (2 * x + 6) = 1) → (x = 1 / 3) :=
by sorry

-- Proof Problem 2 statement
theorem solve_ineq (x : ℝ) : (2 * x - 1 > 3 * (x - 1)) ∧ ((5 - x) / 2 < x + 4) → (-1 < x ∧ x < 2) :=
by sorry

end solve_eq_solve_ineq_l1028_102867


namespace find_bettys_balance_l1028_102846

-- Define the conditions as hypotheses
def balance_in_bettys_account (B : ℕ) : Prop :=
  -- Gina has two accounts with a combined balance equal to $1,728
  (2 * (B / 4)) = 1728

-- State the theorem to be proven
theorem find_bettys_balance (B : ℕ) (h : balance_in_bettys_account B) : B = 3456 :=
by
  -- The proof is provided here as a "sorry"
  sorry

end find_bettys_balance_l1028_102846


namespace parabola_focus_l1028_102813

theorem parabola_focus (x y : ℝ) (p : ℝ) (h_eq : x^2 = 8 * y) (h_form : x^2 = 4 * p * y) : 
  p = 2 ∧ y = (x^2 / 8) ∧ (0, p) = (0, 2) :=
by
  sorry

end parabola_focus_l1028_102813


namespace EricBenJackMoneySum_l1028_102854

noncomputable def EricBenJackTotal (E B J : ℕ) :=
  (E + B + J : ℕ)

theorem EricBenJackMoneySum :
  ∀ (E B J : ℕ), (E = B - 10) → (B = J - 9) → (J = 26) → (EricBenJackTotal E B J) = 50 :=
by
  intros E B J
  intro hE hB hJ
  rw [hJ] at hB
  rw [hB] at hE
  sorry

end EricBenJackMoneySum_l1028_102854


namespace days_to_finish_job_l1028_102868

def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 4 / 15
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

theorem days_to_finish_job (A B C : ℚ) (h1 : A + B = work_rate_a_b) (h2 : C = work_rate_c) :
  1 / (A + B + C) = 3 :=
by
  sorry

end days_to_finish_job_l1028_102868


namespace jane_total_score_l1028_102849

theorem jane_total_score :
  let correct_answers := 17
  let incorrect_answers := 12
  let unanswered_questions := 6
  let total_questions := 35
  let points_per_correct := 1
  let points_per_incorrect := -0.25
  let correct_points := correct_answers * points_per_correct
  let incorrect_points := incorrect_answers * points_per_incorrect
  let total_score := correct_points + incorrect_points
  total_score = 14 :=
by
  sorry

end jane_total_score_l1028_102849


namespace abc_inequality_l1028_102802

theorem abc_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

end abc_inequality_l1028_102802


namespace square_root_then_square_l1028_102832

theorem square_root_then_square (x : ℕ) (hx : x = 49) : (Nat.sqrt x) ^ 2 = 49 := by
  sorry

end square_root_then_square_l1028_102832


namespace function_properties_l1028_102864

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 3)

theorem function_properties :
  (∃ x : ℝ, f x = -1) = false ∧ 
  (∃ x_0 : ℝ, -1 < x_0 ∧ x_0 < 0 ∧ deriv f x_0 = 0) ∧ 
  (∀ x : ℝ, -3 < x → f x > -1 / 2) ∧ 
  (∃ x_0 : ℝ, -3 < x_0 ∧ ∀ x : ℝ, -3 < x → f x_0 ≤ f x) :=
by
  sorry

end function_properties_l1028_102864


namespace trapezoid_area_l1028_102853

-- Define the given conditions in the problem
variables (EF GH h EG FH : ℝ)
variables (EF_parallel_GH : true) -- EF and GH are parallel (not used in the calculation)
variables (EF_eq_70 : EF = 70)
variables (GH_eq_40 : GH = 40)
variables (h_eq_15 : h = 15)
variables (EG_eq_20 : EG = 20)
variables (FH_eq_25 : FH = 25)

-- Define the main theorem to prove
theorem trapezoid_area (EF GH h EG FH : ℝ) 
  (EF_eq_70 : EF = 70) 
  (GH_eq_40 : GH = 40) 
  (h_eq_15 : h = 15) 
  (EG_eq_20 : EG = 20) 
  (FH_eq_25 : FH = 25) : 
  0.5 * (EF + GH) * h = 825 := 
by 
  sorry

end trapezoid_area_l1028_102853


namespace cookies_leftover_l1028_102841

def amelia_cookies := 52
def benjamin_cookies := 63
def chloe_cookies := 25
def total_cookies := amelia_cookies + benjamin_cookies + chloe_cookies
def package_size := 15

theorem cookies_leftover :
  total_cookies % package_size = 5 := by
  sorry

end cookies_leftover_l1028_102841


namespace value_of_x_l1028_102842

theorem value_of_x (x : ℝ) (h : 0.5 * x - (1 / 3) * x = 110) : x = 660 :=
sorry

end value_of_x_l1028_102842


namespace units_digit_base8_l1028_102811

theorem units_digit_base8 (a b : ℕ) (h_a : a = 505) (h_b : b = 71) : 
  ((a * b) % 8) = 7 := 
by
  sorry

end units_digit_base8_l1028_102811


namespace snack_eaters_remaining_l1028_102892

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end snack_eaters_remaining_l1028_102892


namespace triangle_area_l1028_102810

theorem triangle_area (a b c : ℕ) (h : a = 12) (i : b = 16) (j : c = 20) (hc : c * c = a * a + b * b) :
  ∃ (area : ℕ), area = 96 :=
by
  sorry

end triangle_area_l1028_102810


namespace symmetric_line_eq_l1028_102825

theorem symmetric_line_eq (x y : ℝ) :
    3 * x - 4 * y + 5 = 0 ↔ 3 * x + 4 * (-y) + 5 = 0 :=
sorry

end symmetric_line_eq_l1028_102825


namespace average_postcards_collected_per_day_l1028_102869

theorem average_postcards_collected_per_day 
    (a : ℕ) (d : ℕ) (n : ℕ) 
    (h_a : a = 10)
    (h_d : d = 12)
    (h_n : n = 7) :
    (a + (a + (n - 1) * d)) / 2 = 46 := by
  sorry

end average_postcards_collected_per_day_l1028_102869


namespace intersection_complement_correct_l1028_102894

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A based on the condition given
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 3}

-- Define set B based on the condition given
def B : Set ℝ := {x | x > 3}

-- Define the complement of set B in the universal set U
def compl_B : Set ℝ := {x | x ≤ 3}

-- Define the expected result of A ∩ compl_B
def expected_result : Set ℝ := {x | x ≤ -3} ∪ {3}

-- State the theorem to be proven
theorem intersection_complement_correct :
  (A ∩ compl_B) = expected_result :=
sorry

end intersection_complement_correct_l1028_102894


namespace largest_number_is_a_l1028_102821

-- Define the numbers in their respective bases
def a := 8 * 9 + 5
def b := 3 * 5^2 + 0 * 5 + 1 * 5^0
def c := 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0

theorem largest_number_is_a : a > b ∧ a > c :=
by
  -- These are the expected results, rest is the proof steps which we skip using sorry
  have ha : a = 77 := rfl
  have hb : b = 76 := rfl
  have hc : c = 9 := rfl
  sorry

end largest_number_is_a_l1028_102821


namespace largest_int_value_of_m_l1028_102862

variable {x y m : ℤ}

theorem largest_int_value_of_m (h1 : x + 2 * y = 2 * m + 1)
                              (h2 : 2 * x + y = m + 2)
                              (h3 : x - y > 2) : m = -2 := 
sorry

end largest_int_value_of_m_l1028_102862


namespace solve_for_x_l1028_102878

theorem solve_for_x (x : ℝ) (h1 : x ≠ -3) (h2 : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5)) : x = -9 :=
by
  sorry

end solve_for_x_l1028_102878


namespace two_digit_number_l1028_102835

theorem two_digit_number (x y : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy : 0 ≤ y ∧ y ≤ 9)
  (h1 : x^2 + y^2 = 10*x + y + 11) (h2 : 2*x*y = 10*x + y - 5) :
  10*x + y = 95 ∨ 10*x + y = 15 := 
sorry

end two_digit_number_l1028_102835


namespace original_denominator_l1028_102823

theorem original_denominator (d : ℕ) (h : 3 * (d : ℚ) = 2) : d = 3 := 
by
  sorry

end original_denominator_l1028_102823


namespace secretary_work_hours_l1028_102888

theorem secretary_work_hours
  (x : ℕ)
  (h_ratio : 2 * x + 3 * x + 5 * x = 110) :
  5 * x = 55 := 
by
  sorry

end secretary_work_hours_l1028_102888


namespace new_computer_lasts_l1028_102805

theorem new_computer_lasts (x : ℕ) 
  (h1 : 600 = 400 + 200)
  (h2 : ∀ y : ℕ, (2 * 200 = 400) → (2 * 3 = 6) → y = 6)
  (h3 : 200 = 600 - 400) :
  x = 6 :=
by
  sorry

end new_computer_lasts_l1028_102805


namespace tom_age_ratio_l1028_102861

theorem tom_age_ratio (T N : ℕ) (h1 : sum_ages = T) (h2 : T - N = 3 * (sum_ages_N_years_ago))
  (h3 : sum_ages = T) (h4 : sum_ages_N_years_ago = T - 4 * N) :
  T / N = 11 / 2 := 
by
  sorry

end tom_age_ratio_l1028_102861


namespace odd_integers_count_between_fractions_l1028_102818

theorem odd_integers_count_between_fractions :
  ∃ (count : ℕ), count = 14 ∧
  ∀ (n : ℤ), (25:ℚ)/3 < (n : ℚ) ∧ (n : ℚ) < (73 : ℚ)/2 ∧ (n % 2 = 1) :=
sorry

end odd_integers_count_between_fractions_l1028_102818


namespace not_sufficient_nor_necessary_l1028_102856

theorem not_sufficient_nor_necessary (a b : ℝ) (hb : b ≠ 0) :
  ¬ ((a > b) ↔ (1 / a < 1 / b)) :=
by
  sorry

end not_sufficient_nor_necessary_l1028_102856


namespace quadrilateral_type_l1028_102863

theorem quadrilateral_type (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2 * m * n + 2 * p * q) : 
  (m = n ∧ p = q) ∨ (m ≠ n ∧ p ≠ q ∧ ∃ k : ℝ, k^2 * (m^2 + n^2) = p^2 + q^2) := 
sorry

end quadrilateral_type_l1028_102863


namespace value_of_x_if_additive_inverses_l1028_102828

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l1028_102828


namespace tablespoons_in_half_cup_l1028_102836

theorem tablespoons_in_half_cup
    (grains_per_cup : ℕ)
    (half_cup : ℕ)
    (tbsp_to_tsp : ℕ)
    (grains_per_tsp : ℕ)
    (h1 : grains_per_cup = 480)
    (h2 : half_cup = grains_per_cup / 2)
    (h3 : tbsp_to_tsp = 3)
    (h4 : grains_per_tsp = 10) :
    (half_cup / (tbsp_to_tsp * grains_per_tsp) = 8) :=
by
  sorry

end tablespoons_in_half_cup_l1028_102836


namespace triangle_not_always_obtuse_l1028_102800

def is_acute_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ A < 90 ∧ B < 90 ∧ C < 90

theorem triangle_not_always_obtuse : ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180 ∧ is_acute_triangle A B C :=
by
  -- Exact proof here.
  sorry

end triangle_not_always_obtuse_l1028_102800


namespace profit_is_35_percent_l1028_102840

def cost_price (C : ℝ) := C
def initial_selling_price (C : ℝ) := 1.20 * C
def second_selling_price (C : ℝ) := 1.50 * C
def final_selling_price (C : ℝ) := 1.35 * C

theorem profit_is_35_percent (C : ℝ) : 
    final_selling_price C - cost_price C = 0.35 * cost_price C :=
by
    sorry

end profit_is_35_percent_l1028_102840


namespace speed_of_train_approx_29_0088_kmh_l1028_102889

noncomputable def speed_of_train_in_kmh := 
  let length_train : ℝ := 288
  let length_bridge : ℝ := 101
  let time_seconds : ℝ := 48.29
  let total_distance : ℝ := length_train + length_bridge
  let speed_m_per_s : ℝ := total_distance / time_seconds
  speed_m_per_s * 3.6

theorem speed_of_train_approx_29_0088_kmh :
  abs (speed_of_train_in_kmh - 29.0088) < 0.001 := 
by
  sorry

end speed_of_train_approx_29_0088_kmh_l1028_102889


namespace two_digit_number_is_27_l1028_102852

theorem two_digit_number_is_27 :
  ∃ n : ℕ, (n / 10 < 10) ∧ (n % 10 < 10) ∧ 
  (100*(n) = 37*(10*(n) + 1)) ∧ 
  n = 27 :=
by {
  sorry
}

end two_digit_number_is_27_l1028_102852


namespace cookie_cost_l1028_102809

theorem cookie_cost 
    (initial_amount : ℝ := 100)
    (latte_cost : ℝ := 3.75)
    (croissant_cost : ℝ := 3.50)
    (days : ℕ := 7)
    (num_cookies : ℕ := 5)
    (remaining_amount : ℝ := 43) :
    (initial_amount - remaining_amount - (days * (latte_cost + croissant_cost))) / num_cookies = 1.25 := 
by
  sorry

end cookie_cost_l1028_102809


namespace solution_set_of_inequality_l1028_102830

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = 2^x - 4

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : satisfies_condition f) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
sorry

end solution_set_of_inequality_l1028_102830


namespace book_store_sold_total_copies_by_saturday_l1028_102831

def copies_sold_on_monday : ℕ := 15
def copies_sold_on_tuesday : ℕ := copies_sold_on_monday * 2
def copies_sold_on_wednesday : ℕ := copies_sold_on_tuesday + (copies_sold_on_tuesday / 2)
def copies_sold_on_thursday : ℕ := copies_sold_on_wednesday + (copies_sold_on_wednesday / 2)
def copies_sold_on_friday_pre_promotion : ℕ := copies_sold_on_thursday + (copies_sold_on_thursday / 2)
def copies_sold_on_friday_post_promotion : ℕ := copies_sold_on_friday_pre_promotion + (copies_sold_on_friday_pre_promotion / 4)
def copies_sold_on_saturday : ℕ := copies_sold_on_friday_pre_promotion * 7 / 10

def total_copies_sold_by_saturday : ℕ :=
  copies_sold_on_monday + copies_sold_on_tuesday + copies_sold_on_wednesday +
  copies_sold_on_thursday + copies_sold_on_friday_post_promotion + copies_sold_on_saturday

theorem book_store_sold_total_copies_by_saturday : total_copies_sold_by_saturday = 357 :=
by
  -- Proof here
  sorry

end book_store_sold_total_copies_by_saturday_l1028_102831


namespace incorrect_conclusion_D_l1028_102826

def parabola (x : ℝ) : ℝ := (x - 2) ^ 2 + 1

theorem incorrect_conclusion_D :
  ∀ x : ℝ, x < 2 → ∃ y1 y2 : ℝ, y1 = parabola x ∧ y2 = parabola (x + 1) ∧ y1 > y2 :=
by
  sorry

end incorrect_conclusion_D_l1028_102826


namespace gnomes_remaining_in_ravenswood_l1028_102844

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l1028_102844


namespace multiple_of_12_l1028_102875

theorem multiple_of_12 (x : ℤ) : 
  (7 * x - 3) % 12 = 0 ↔ (x % 12 = 9 ∨ x % 12 = 1029 % 12) :=
by
  sorry

end multiple_of_12_l1028_102875


namespace vector_simplification_l1028_102822

variables (V : Type) [AddCommGroup V]

variables (CE AC DE AD : V)

theorem vector_simplification :
  CE + AC - DE - AD = 0 :=
by
  sorry

end vector_simplification_l1028_102822


namespace coefficient_of_1_div_x_l1028_102851

open Nat

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ :=
  (1 / Real.sqrt x - 3)^n

theorem coefficient_of_1_div_x (x : ℝ) (n : ℕ) (h1 : n ∈ {m | m > 0}) (h2 : binomial_expansion x n = 16) :
  ∃ c : ℝ, c = 54 :=
by
  sorry

end coefficient_of_1_div_x_l1028_102851


namespace solution_to_inequalities_l1028_102824

theorem solution_to_inequalities (x : ℝ) : 
  (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x + 1) ↔ (1 < x ∧ x < 3) := by
  sorry

end solution_to_inequalities_l1028_102824


namespace pawpaws_basket_l1028_102897

variable (total_fruits mangoes pears lemons kiwis : ℕ)
variable (pawpaws : ℕ)

theorem pawpaws_basket
  (h1 : total_fruits = 58)
  (h2 : mangoes = 18)
  (h3 : pears = 10)
  (h4 : lemons = 9)
  (h5 : kiwis = 9)
  (h6 : total_fruits = mangoes + pears + lemons + kiwis + pawpaws) :
  pawpaws = 12 := by
  sorry

end pawpaws_basket_l1028_102897


namespace ambulance_ride_cost_correct_l1028_102873

noncomputable def total_bill : ℝ := 12000
noncomputable def medication_percentage : ℝ := 0.40
noncomputable def imaging_tests_percentage : ℝ := 0.15
noncomputable def surgical_procedure_percentage : ℝ := 0.20
noncomputable def overnight_stays_percentage : ℝ := 0.25
noncomputable def food_cost : ℝ := 300
noncomputable def consultation_fee : ℝ := 80

noncomputable def ambulance_ride_cost := total_bill - (food_cost + consultation_fee)

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 11620 :=
by
  sorry

end ambulance_ride_cost_correct_l1028_102873


namespace interest_earned_is_91_dollars_l1028_102883

-- Define the initial conditions
def P : ℝ := 2000
def r : ℝ := 0.015
def n : ℕ := 3

-- Define the compounded amount function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Prove the interest earned after 3 years is 91 dollars
theorem interest_earned_is_91_dollars : 
  (compound_interest P r n) - P = 91 :=
by
  sorry

end interest_earned_is_91_dollars_l1028_102883


namespace angle_comparison_l1028_102850

theorem angle_comparison :
  let A := 60.4
  let B := 60.24
  let C := 60.24
  A > B ∧ B = C :=
by
  sorry

end angle_comparison_l1028_102850


namespace infinite_series_sum_eq_one_fourth_l1028_102860

theorem infinite_series_sum_eq_one_fourth :
  (∑' n : ℕ, 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))) = 1 / 4 :=
sorry

end infinite_series_sum_eq_one_fourth_l1028_102860


namespace polygon_is_quadrilateral_l1028_102858

-- Problem statement in Lean 4
theorem polygon_is_quadrilateral 
  (n : ℕ) 
  (h₁ : (n - 2) * 180 = 360) :
  n = 4 :=
by
  sorry

end polygon_is_quadrilateral_l1028_102858


namespace spending_difference_l1028_102855

-- Define the conditions
def spent_on_chocolate : ℤ := 7
def spent_on_candy_bar : ℤ := 2

-- The theorem to be proven
theorem spending_difference : (spent_on_chocolate - spent_on_candy_bar = 5) :=
by sorry

end spending_difference_l1028_102855


namespace trick_or_treat_hours_l1028_102895

variable (num_children : ℕ)
variable (houses_per_hour : ℕ)
variable (treats_per_house_per_kid : ℕ)
variable (total_treats : ℕ)

theorem trick_or_treat_hours (h : num_children = 3)
  (h1 : houses_per_hour = 5)
  (h2 : treats_per_house_per_kid = 3)
  (h3 : total_treats = 180) :
  total_treats / (num_children * houses_per_hour * treats_per_house_per_kid) = 4 :=
by
  sorry

end trick_or_treat_hours_l1028_102895


namespace speed_ratio_is_2_l1028_102887

def distance_to_work : ℝ := 20
def total_hours_on_road : ℝ := 6
def speed_back_home : ℝ := 10

theorem speed_ratio_is_2 :
  (∃ v : ℝ, (20 / v) + (20 / 10) = 6) → (10 = 2 * v) :=
by sorry

end speed_ratio_is_2_l1028_102887


namespace ratio_of_money_spent_l1028_102812

theorem ratio_of_money_spent (h : ∀(a b c : ℕ), a + b + c = 75) : 
  (25 / 75 = 1 / 3) ∧ 
  (40 / 75 = 4 / 3) ∧ 
  (10 / 75 = 2 / 15) :=
by
  sorry

end ratio_of_money_spent_l1028_102812


namespace find_b_l1028_102871

variable (b : ℝ)

theorem find_b 
    (h₁ : 0 < b)
    (h₂ : b < 4)
    (area_ratio : ∃ k : ℝ, k = 4/16 ∧ (4 + b) / -b = 2 * k) :
  b = -4/3 :=
by
  sorry

end find_b_l1028_102871


namespace greatest_expression_l1028_102893

theorem greatest_expression 
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : x1 < x2) 
  (hx12 : x1 + x2 = 1) 
  (hy1 : 0 < y1) 
  (hy2 : y1 < y2) 
  (hy12 : y1 + y2 = 1) : 
  x1 * y1 + x2 * y2 > max (x1 * x2 + y1 * y2) (max (x1 * y2 + x2 * y1) (1/2)) := 
sorry

end greatest_expression_l1028_102893


namespace area_of_triangle_l1028_102866

theorem area_of_triangle (A : ℝ) (b : ℝ) (a : ℝ) (hA : A = 60) (hb : b = 4) (ha : a = 2 * Real.sqrt 3) : 
  1 / 2 * a * b * Real.sin (60 * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l1028_102866


namespace initial_tanks_hold_fifteen_fish_l1028_102845

theorem initial_tanks_hold_fifteen_fish (t : Nat) (additional_tanks : Nat) (fish_per_additional_tank : Nat) (total_fish : Nat) :
  t = 3 ∧ additional_tanks = 3 ∧ fish_per_additional_tank = 10 ∧ total_fish = 75 → 
  ∀ (F : Nat), (F * t) = 45 → F = 15 :=
by
  sorry

end initial_tanks_hold_fifteen_fish_l1028_102845
