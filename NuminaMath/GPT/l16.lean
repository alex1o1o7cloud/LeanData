import Mathlib

namespace points_on_same_sphere_l16_1655

-- Define the necessary structures and assumptions
variables {P : Type*} [MetricSpace P]

-- Definitions of spheres and points
structure Sphere (P : Type*) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(positive_radius : 0 < radius)

def symmetric_point (S A1 : P) : P := sorry -- definition to get the symmetric point A2

-- Given conditions
variables (S A B C A1 B1 C1 A2 B2 C2 : P)
variable (omega : Sphere P)
variable (Omega : Sphere P)
variable (M_S_A : P) -- midpoint of SA
variable (M_S_B : P) -- midpoint of SB
variable (M_S_C : P) -- midpoint of SC

-- Assertions of conditions
axiom sphere_through_vertex : omega.center = S
axiom first_intersections : omega.radius = dist S A1 ∧ omega.radius = dist S B1 ∧ omega.radius = dist S C1
axiom omega_Omega_intersection : ∃ (circle_center : P) (plane_parallel_to_ABC : P), true-- some conditions indicating intersection
axiom symmetric_points_A1_A2 : A2 = symmetric_point S A1
axiom symmetric_points_B1_B2 : B2 = symmetric_point S B1
axiom symmetric_points_C1_C2 : C2 = symmetric_point S C1

-- The theorem to prove
theorem points_on_same_sphere : ∃ (sphere : Sphere P), 
  (dist sphere.center A) = sphere.radius ∧ 
  (dist sphere.center B) = sphere.radius ∧ 
  (dist sphere.center C) = sphere.radius ∧ 
  (dist sphere.center A2) = sphere.radius ∧ 
  (dist sphere.center B2) = sphere.radius ∧ 
  (dist sphere.center C2) = sphere.radius := 
sorry

end points_on_same_sphere_l16_1655


namespace complex_z_calculation_l16_1688

theorem complex_z_calculation (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z :=
sorry

end complex_z_calculation_l16_1688


namespace cheese_cookies_price_is_correct_l16_1625

-- Define the problem conditions and constants
def total_boxes_per_carton : ℕ := 15
def total_packs_per_box : ℕ := 12
def discount_15_percent : ℝ := 0.15
def total_number_of_cartons : ℕ := 13
def total_cost_paid : ℝ := 2058

-- Calculate the expected price per pack
noncomputable def price_per_pack : ℝ :=
  let total_packs := total_boxes_per_carton * total_packs_per_box * total_number_of_cartons
  let total_cost_without_discount := total_cost_paid / (1 - discount_15_percent)
  total_cost_without_discount / total_packs

theorem cheese_cookies_price_is_correct : 
  abs (price_per_pack - 1.0347) < 0.0001 :=
by sorry

end cheese_cookies_price_is_correct_l16_1625


namespace part1_part2_l16_1690

-- Definition of the conditions given
def february_parcels : ℕ := 200000
def april_parcels : ℕ := 338000
def monthly_growth_rate : ℝ := 0.3

-- Problem 1: Proving the monthly growth rate is 0.3
theorem part1 (x : ℝ) (h : february_parcels * (1 + x)^2 = april_parcels) : x = monthly_growth_rate :=
  sorry

-- Problem 2: Proving the number of parcels in May is less than 450,000 with the given growth rate
theorem part2 (h : monthly_growth_rate = 0.3 ) : february_parcels * (1 + monthly_growth_rate)^3 < 450000 :=
  sorry

end part1_part2_l16_1690


namespace isosceles_triangle_angles_l16_1630

theorem isosceles_triangle_angles (A B C : ℝ)
    (h_iso : A = B ∨ B = C ∨ C = A)
    (h_one_angle : A = 36 ∨ B = 36 ∨ C = 36)
    (h_sum_angles : A + B + C = 180) :
  (A = 36 ∧ B = 36 ∧ C = 108) ∨
  (A = 72 ∧ B = 72 ∧ C = 36) :=
by 
  sorry

end isosceles_triangle_angles_l16_1630


namespace client_dropped_off_phones_l16_1674

def initial_phones : ℕ := 15
def repaired_phones : ℕ := 3
def coworker_phones : ℕ := 9

theorem client_dropped_off_phones (x : ℕ) : 
  initial_phones - repaired_phones + x = 2 * coworker_phones → x = 6 :=
by
  sorry

end client_dropped_off_phones_l16_1674


namespace sum_of_ages_26_l16_1638

-- Define an age predicate to manage the three ages
def is_sum_of_ages (kiana twin : ℕ) : Prop :=
  kiana < twin ∧ twin * twin * kiana = 180 ∧ (kiana + twin + twin = 26)

theorem sum_of_ages_26 : 
  ∃ (kiana twin : ℕ), is_sum_of_ages kiana twin :=
by 
  sorry

end sum_of_ages_26_l16_1638


namespace find_f_7_l16_1600

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 + b * x^3 + c * x - 5

theorem find_f_7 (a b c : ℝ) (h : f a b c (-7) = 7) : f a b c 7 = -17 :=
by
  dsimp [f] at *
  sorry

end find_f_7_l16_1600


namespace minimize_cost_l16_1611

theorem minimize_cost (x : ℝ) (h1 : 0 < x) (h2 : 400 / x * 40 ≤ 4 * x) : x = 20 :=
by
  sorry

end minimize_cost_l16_1611


namespace problem_solution_l16_1663

theorem problem_solution (x : ℝ) :
          ((3 * x - 4) * (x + 5) ≠ 0) → 
          (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
          (x ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioi (4 / 3)) :=
sorry

end problem_solution_l16_1663


namespace person_reaches_before_bus_l16_1665

theorem person_reaches_before_bus (dist : ℝ) (speed1 speed2 : ℝ) (miss_time_minutes : ℝ) :
  dist = 2.2 → speed1 = 3 → speed2 = 6 → miss_time_minutes = 12 →
  ((60 : ℝ) * (dist/speed1) - miss_time_minutes) - ((60 : ℝ) * (dist/speed2)) = 10 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end person_reaches_before_bus_l16_1665


namespace all_inequalities_hold_l16_1650

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x^2 < a^2
def condition2 : Prop := y^2 < b^2
def condition3 : Prop := z^2 < c^2

-- Inequalities to prove
def inequality1 : Prop := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a^2 * b^2 + b^2 * c^2 + c^2 * a^2
def inequality2 : Prop := x^4 + y^4 + z^4 < a^4 + b^4 + c^4
def inequality3 : Prop := x^2 * y^2 * z^2 < a^2 * b^2 * c^2

theorem all_inequalities_hold (h1 : condition1 a x) (h2 : condition2 b y) (h3 : condition3 c z) :
  inequality1 a b c x y z ∧ inequality2 a b c x y z ∧ inequality3 a b c x y z := by
  sorry

end all_inequalities_hold_l16_1650


namespace evaluate_expression_l16_1673

def a : ℕ := 3
def b : ℕ := 2

theorem evaluate_expression : (a^2 * a^5) / (b^2 / b^3) = 4374 := by
  sorry

end evaluate_expression_l16_1673


namespace simplified_expression_value_l16_1623

theorem simplified_expression_value (x : ℝ) (h : x = -2) :
  (x - 2)^2 - 4 * x * (x - 1) + (2 * x + 1) * (2 * x - 1) = 7 := 
  by
    -- We are given x = -2
    simp [h]
    -- sorry added to skip the actual solution in Lean
    sorry

end simplified_expression_value_l16_1623


namespace Lara_age_10_years_from_now_l16_1620

theorem Lara_age_10_years_from_now (current_year_age : ℕ) (age_7_years_ago : ℕ)
  (h1 : age_7_years_ago = 9) (h2 : current_year_age = age_7_years_ago + 7) :
  current_year_age + 10 = 26 :=
by
  sorry

end Lara_age_10_years_from_now_l16_1620


namespace revenue_highest_visitors_is_48_thousand_l16_1672

-- Define the frequencies for each day
def freq_Oct_1 : ℝ := 0.05
def freq_Oct_2 : ℝ := 0.08
def freq_Oct_3 : ℝ := 0.09
def freq_Oct_4 : ℝ := 0.13
def freq_Oct_5 : ℝ := 0.30
def freq_Oct_6 : ℝ := 0.15
def freq_Oct_7 : ℝ := 0.20

-- Define the revenue on October 1st
def revenue_Oct_1 : ℝ := 80000

-- Define the revenue is directly proportional to the frequency of visitors
def avg_daily_visitor_spending_is_constant := true

-- The goal is to prove that the revenue on the day with the highest frequency is 48 thousand yuan
theorem revenue_highest_visitors_is_48_thousand :
  avg_daily_visitor_spending_is_constant →
  revenue_Oct_1 / freq_Oct_1 = x / freq_Oct_5 →
  x = 48000 :=
by
  sorry

end revenue_highest_visitors_is_48_thousand_l16_1672


namespace lowest_common_denominator_l16_1613

theorem lowest_common_denominator (a b c : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : c = 18) : Nat.lcm (Nat.lcm a b) c = 36 :=
by
  -- Introducing the given conditions
  rw [h1, h2, h3]
  -- Compute the LCM of the provided values
  sorry

end lowest_common_denominator_l16_1613


namespace minimum_value_expression_l16_1691

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ m, (∀ x y, x > 0 ∧ y > 0 → (x + y) * (1/x + 4/y) ≥ m) ∧ m = 9 :=
sorry

end minimum_value_expression_l16_1691


namespace flour_needed_for_dozen_cookies_l16_1621

/--
Matt uses 4 bags of flour, each weighing 5 pounds, to make a total of 120 cookies.
Prove that 2 pounds of flour are needed to make a dozen cookies.
-/
theorem flour_needed_for_dozen_cookies :
  ∀ (bags_of_flour : ℕ) (weight_per_bag : ℕ) (total_cookies : ℕ),
  bags_of_flour = 4 →
  weight_per_bag = 5 →
  total_cookies = 120 →
  (12 * (bags_of_flour * weight_per_bag)) / total_cookies = 2 :=
by
  sorry

end flour_needed_for_dozen_cookies_l16_1621


namespace expression_calculation_l16_1637

theorem expression_calculation : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 28 * 21^1005 :=
by
  sorry

end expression_calculation_l16_1637


namespace correct_option_l16_1659

variable (a : ℤ)

theorem correct_option :
  (-2 * a^2)^3 = -8 * a^6 :=
by
  sorry

end correct_option_l16_1659


namespace ball_hits_ground_time_l16_1660

theorem ball_hits_ground_time (t : ℝ) : 
  (∃ t : ℝ, -10 * t^2 + 40 * t + 50 = 0 ∧ t ≥ 0) → t = 5 := 
by
  -- placeholder for proof
  sorry

end ball_hits_ground_time_l16_1660


namespace find_length_of_AC_in_triangle_ABC_l16_1671

noncomputable def length_AC_in_triangle_ABC
  (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3) :
  ℝ :=
  let cos_B := Real.cos (Real.pi / 3)
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_B
  Real.sqrt AC_squared

theorem find_length_of_AC_in_triangle_ABC :
  ∃ AC : ℝ, ∀ (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3),
    length_AC_in_triangle_ABC AB BC angle_B h_AB h_BC h_angle_B = Real.sqrt 3 :=
by sorry

end find_length_of_AC_in_triangle_ABC_l16_1671


namespace thirty_k_divisor_of_929260_l16_1636

theorem thirty_k_divisor_of_929260 (k : ℕ) (h1: 30^k ∣ 929260):
(3^k - k^3 = 2) :=
sorry

end thirty_k_divisor_of_929260_l16_1636


namespace trader_loss_percentage_l16_1680

theorem trader_loss_percentage :
  let SP := 325475
  let gain := 14 / 100
  let loss := 14 / 100
  let CP1 := SP / (1 + gain)
  let CP2 := SP / (1 - loss)
  let TCP := CP1 + CP2
  let TSP := SP + SP
  let profit_or_loss := TSP - TCP
  let profit_or_loss_percentage := (profit_or_loss / TCP) * 100
  profit_or_loss_percentage = -1.958 :=
by
  sorry

end trader_loss_percentage_l16_1680


namespace taehyung_math_score_l16_1695

theorem taehyung_math_score
  (avg_before : ℝ)
  (drop_in_avg : ℝ)
  (num_subjects_before : ℕ)
  (num_subjects_after : ℕ)
  (avg_after : ℝ)
  (total_before : ℝ)
  (total_after : ℝ)
  (math_score : ℝ) :
  avg_before = 95 →
  drop_in_avg = 3 →
  num_subjects_before = 3 →
  num_subjects_after = 4 →
  avg_after = avg_before - drop_in_avg →
  total_before = avg_before * num_subjects_before →
  total_after = avg_after * num_subjects_after →
  math_score = total_after - total_before →
  math_score = 83 :=
by
  intros
  sorry

end taehyung_math_score_l16_1695


namespace expected_value_linear_combination_l16_1666

variable (ξ η : ℝ)
variable (E : ℝ → ℝ)
axiom E_lin (a b : ℝ) (X Y : ℝ) : E (a * X + b * Y) = a * E X + b * E Y

axiom E_ξ : E ξ = 10
axiom E_η : E η = 3

theorem expected_value_linear_combination : E (3 * ξ + 5 * η) = 45 := by
  sorry

end expected_value_linear_combination_l16_1666


namespace line_not_in_first_quadrant_l16_1667

theorem line_not_in_first_quadrant (t : ℝ) : 
  (∀ x y : ℝ, ¬ ((0 < x ∧ 0 < y) ∧ (2 * t - 3) * x + y + 6 = 0)) ↔ t ≥ 3 / 2 :=
by
  sorry

end line_not_in_first_quadrant_l16_1667


namespace intersection_M_N_l16_1654

def M := {y : ℝ | y <= 4}
def N := {x : ℝ | x > 0}

theorem intersection_M_N : {x : ℝ | x > 0} ∩ {y : ℝ | y <= 4} = {z : ℝ | 0 < z ∧ z <= 4} :=
by
  sorry

end intersection_M_N_l16_1654


namespace isosceles_triangle_condition_l16_1639

-- Theorem statement
theorem isosceles_triangle_condition (N : ℕ) (h : N > 2) : 
  (∃ N1 : ℕ, N = N1 ∧ N1 = 10) ∨ (∃ N2 : ℕ, N = N2 ∧ N2 = 11) :=
by sorry

end isosceles_triangle_condition_l16_1639


namespace inequality_proof_l16_1668

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : 
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9 * x * y * z ≥ 9 * (x * y + y * z + z * x) :=
by 
  sorry

end inequality_proof_l16_1668


namespace man_son_ratio_in_two_years_l16_1653

noncomputable def man_and_son_age_ratio (M S : ℕ) (h1 : M = S + 25) (h2 : S = 23) : ℕ × ℕ :=
  let S_in_2_years := S + 2
  let M_in_2_years := M + 2
  (M_in_2_years / S_in_2_years, S_in_2_years / S_in_2_years)

theorem man_son_ratio_in_two_years : man_and_son_age_ratio 48 23 (by norm_num) (by norm_num) = (2, 1) :=
  sorry

end man_son_ratio_in_two_years_l16_1653


namespace team_X_finishes_with_more_points_than_Y_l16_1649

-- Define the number of teams and games played
def numberOfTeams : ℕ := 8
def gamesPerTeam : ℕ := numberOfTeams - 1

-- Define the probability of winning (since each team has a 50% chance to win any game)
def probOfWin : ℝ := 0.5

-- Define the event that team X finishes with more points than team Y
noncomputable def probXFinishesMorePointsThanY : ℝ := 1 / 2

-- Statement to be proved: 
theorem team_X_finishes_with_more_points_than_Y :
  (∃ p : ℝ, p = probXFinishesMorePointsThanY) :=
sorry

end team_X_finishes_with_more_points_than_Y_l16_1649


namespace third_twenty_third_wise_superior_number_l16_1605

def wise_superior_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ x = m^2 - n^2

theorem third_twenty_third_wise_superior_number :
  ∃ T_3 T_23 : ℕ, wise_superior_number T_3 ∧ wise_superior_number T_23 ∧ T_3 = 15 ∧ T_23 = 57 :=
by
  sorry

end third_twenty_third_wise_superior_number_l16_1605


namespace population_growth_l16_1608

theorem population_growth (P_present P_future : ℝ) (r : ℝ) (n : ℕ)
  (h1 : P_present = 7800)
  (h2 : P_future = 10860.72)
  (h3 : n = 2) :
  P_future = P_present * (1 + r / 100)^n → r = 18.03 :=
by sorry

end population_growth_l16_1608


namespace geometric_seq_a6_value_l16_1642

theorem geometric_seq_a6_value 
    (a : ℕ → ℝ) 
    (q : ℝ) 
    (h_q_pos : q > 0)
    (h_a_pos : ∀ n, a n > 0)
    (h_a2 : a 2 = 1)
    (h_a8_eq : a 8 = a 6 + 2 * a 4) : 
    a 6 = 4 := 
by 
  sorry

end geometric_seq_a6_value_l16_1642


namespace circle_intersection_value_l16_1687

theorem circle_intersection_value {x1 y1 x2 y2 : ℝ} 
  (h_circle : x1^2 + y1^2 = 4)
  (h_non_negative : x1 ≥ 0 ∧ y1 ≥ 0 ∧ x2 ≥ 0 ∧ y2 ≥ 0)
  (h_symmetric : x1 = y2 ∧ x2 = y1) :
  x1^2 + x2^2 = 4 := 
by
  sorry

end circle_intersection_value_l16_1687


namespace arithmetic_sequence_fourth_term_l16_1628

-- Define the arithmetic sequence and conditions
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
def a₂ := 606
def S₄ := 3834

-- Problem statement
theorem arithmetic_sequence_fourth_term :
  (a 1 + a 2 + a 3 = 1818) →
  (a 4 = 2016) :=
sorry

end arithmetic_sequence_fourth_term_l16_1628


namespace elena_meeting_percentage_l16_1696

noncomputable def workday_hours : ℕ := 10
noncomputable def first_meeting_duration_minutes : ℕ := 60
noncomputable def second_meeting_duration_minutes : ℕ := 3 * first_meeting_duration_minutes
noncomputable def total_workday_minutes := workday_hours * 60
noncomputable def total_meeting_minutes := first_meeting_duration_minutes + second_meeting_duration_minutes
noncomputable def percent_time_in_meetings := (total_meeting_minutes * 100) / total_workday_minutes

theorem elena_meeting_percentage : percent_time_in_meetings = 40 := by 
  sorry

end elena_meeting_percentage_l16_1696


namespace sum_of_three_numbers_l16_1616

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 20 := 
by sorry

end sum_of_three_numbers_l16_1616


namespace interest_rate_l16_1633

-- Define the given conditions
def principal : ℝ := 4000
def total_interest : ℝ := 630.50
def future_value : ℝ := principal + total_interest
def time : ℝ := 1.5  -- 1 1/2 years
def times_compounded : ℝ := 2  -- Compounded half yearly

-- Statement to prove the annual interest rate
theorem interest_rate (P A t n : ℝ) (hP : P = principal) (hA : A = future_value) 
    (ht : t = time) (hn : n = times_compounded) :
    ∃ r : ℝ, A = P * (1 + r / n) ^ (n * t) ∧ r = 0.1 := 
by 
  sorry

end interest_rate_l16_1633


namespace balance_scale_comparison_l16_1681

theorem balance_scale_comparison :
  (4 / 3) * Real.pi * (8 : ℝ)^3 > (4 / 3) * Real.pi * (3 : ℝ)^3 + (4 / 3) * Real.pi * (5 : ℝ)^3 :=
by
  sorry

end balance_scale_comparison_l16_1681


namespace inequality_proof_l16_1658

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 1 / b^3 - 1) * (b^3 + 1 / c^3 - 1) * (c^3 + 1 / a^3 - 1) ≤ (a * b * c + 1 / (a * b * c) - 1)^3 :=
by
  sorry

end inequality_proof_l16_1658


namespace sqrt_expression_l16_1632

theorem sqrt_expression : Real.sqrt ((4^2) * (5^6)) = 500 := by
  sorry

end sqrt_expression_l16_1632


namespace parallel_lines_m_condition_l16_1692

theorem parallel_lines_m_condition (m : ℝ) : 
  (∀ (x y : ℝ), (2 * x - m * y - 1 = 0) ↔ ((m - 1) * x - y + 1 = 0)) → m = 2 :=
by
  sorry

end parallel_lines_m_condition_l16_1692


namespace bound_diff_sqrt_two_l16_1640

theorem bound_diff_sqrt_two (a b k m : ℝ) (h : ∀ x ∈ Set.Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * Real.sqrt 2 := sorry

end bound_diff_sqrt_two_l16_1640


namespace complex_z_eq_neg_i_l16_1648

theorem complex_z_eq_neg_i (z : ℂ) (i : ℂ) (h1 : i * z = 1) (hi : i^2 = -1) : z = -i :=
sorry

end complex_z_eq_neg_i_l16_1648


namespace greatest_x_integer_l16_1647

theorem greatest_x_integer (x : ℤ) (h : ∃ n : ℤ, x^2 + 2 * x + 7 = (x - 4) * n) : x ≤ 35 :=
sorry

end greatest_x_integer_l16_1647


namespace part_a_part_b_l16_1683

variable (p : ℕ → ℕ)
axiom primes_sequence : ∀ n, (∀ m < p n, m ∣ p n → m = 1 ∨ m = p n) ∧ p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ p 4 = 7 ∧ p 5 = 11

theorem part_a (n : ℕ) (h : n ≥ 5) : p n > 2 * n := 
  by sorry

theorem part_b (n : ℕ) : p n > 3 * n ↔ n ≥ 12 := 
  by sorry

end part_a_part_b_l16_1683


namespace triangle_perimeter_l16_1634

-- Define the ratios
def ratio1 : ℚ := 1 / 2
def ratio2 : ℚ := 1 / 3
def ratio3 : ℚ := 1 / 4

-- Define the longest side
def longest_side : ℚ := 48

-- Compute the perimeter given the conditions
theorem triangle_perimeter (ratio1 ratio2 ratio3 : ℚ) (longest_side : ℚ) 
  (h_ratio1 : ratio1 = 1 / 2) (h_ratio2 : ratio2 = 1 / 3) (h_ratio3 : ratio3 = 1 / 4)
  (h_longest_side : longest_side = 48) : 
  (longest_side * 6/ (ratio1 * 12 + ratio2 * 12 + ratio3 * 12)) = 104 := by
  sorry

end triangle_perimeter_l16_1634


namespace mixed_fractions_product_l16_1635

theorem mixed_fractions_product :
  ∃ X Y : ℤ, (5 * X + 1) / X * (2 * Y + 1) / 2 = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  simp
  sorry

end mixed_fractions_product_l16_1635


namespace more_crayons_than_erasers_l16_1689

theorem more_crayons_than_erasers
  (E : ℕ) (C : ℕ) (C_left : ℕ) (E_left : ℕ)
  (hE : E = 457) (hC : C = 617) (hC_left : C_left = 523) (hE_left : E_left = E) :
  C_left - E_left = 66 := 
by
  sorry

end more_crayons_than_erasers_l16_1689


namespace find_cylinder_radius_l16_1622

-- Define the problem conditions
def cone_diameter := 10
def cone_altitude := 12
def cylinder_height_eq_diameter (r: ℚ) := 2 * r

-- Define the cone and cylinder inscribed properties
noncomputable def inscribed_cylinder_radius (r : ℚ) : Prop :=
  (cylinder_height_eq_diameter r) ≤ cone_altitude ∧
  2 * r ≤ cone_diameter ∧
  cone_altitude - cylinder_height_eq_diameter r = (cone_altitude * r) / (cone_diameter / 2)

-- The proof goal
theorem find_cylinder_radius : ∃ r : ℚ, inscribed_cylinder_radius r ∧ r = 30/11 :=
by
  sorry

end find_cylinder_radius_l16_1622


namespace total_race_time_l16_1664

theorem total_race_time 
  (num_runners : ℕ) 
  (first_five_time : ℕ) 
  (additional_time : ℕ) 
  (total_runners : ℕ) 
  (num_first_five : ℕ)
  (num_last_three : ℕ) 
  (total_expected_time : ℕ) 
  (h1 : num_runners = 8) 
  (h2 : first_five_time = 8) 
  (h3 : additional_time = 2) 
  (h4 : num_first_five = 5)
  (h5 : num_last_three = num_runners - num_first_five)
  (h6 : total_runners = num_first_five + num_last_three)
  (h7 : 5 * first_five_time + 3 * (first_five_time + additional_time) = total_expected_time)
  : total_expected_time = 70 := 
by
  sorry

end total_race_time_l16_1664


namespace total_visitors_three_days_l16_1697

def V_Rachel := 92
def V_prev_day := 419
def V_day_before_prev := 103

theorem total_visitors_three_days : V_Rachel + V_prev_day + V_day_before_prev = 614 := 
by sorry

end total_visitors_three_days_l16_1697


namespace measure_of_angle_Z_l16_1641

theorem measure_of_angle_Z (X Y Z : ℝ) (h_sum : X + Y + Z = 180) (h_XY : X + Y = 80) : Z = 100 := 
by
  -- The proof is not required.
  sorry

end measure_of_angle_Z_l16_1641


namespace exists_coprime_integers_divisible_l16_1657

theorem exists_coprime_integers_divisible {a b p : ℤ} : ∃ k l : ℤ, gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exists_coprime_integers_divisible_l16_1657


namespace real_part_is_neg4_l16_1698

def real_part_of_z (z : ℂ) : ℝ :=
  z.re

theorem real_part_is_neg4 (i : ℂ) (h : i^2 = -1) :
  real_part_of_z ((3 + 4 * i) * i) = -4 := by
  sorry

end real_part_is_neg4_l16_1698


namespace sum_of_dihedral_angles_leq_90_l16_1629
noncomputable section

-- Let θ1 and θ2 be angles formed by a line with two perpendicular planes
variable (θ1 θ2 : ℝ)

-- Define the condition stating the planes are perpendicular, and the line forms dihedral angles
def dihedral_angle_condition (θ1 θ2 : ℝ) : Prop := 
  θ1 ≥ 0 ∧ θ1 ≤ 90 ∧ θ2 ≥ 0 ∧ θ2 ≤ 90

-- The theorem statement capturing the problem
theorem sum_of_dihedral_angles_leq_90 
  (θ1 θ2 : ℝ) 
  (h : dihedral_angle_condition θ1 θ2) : 
  θ1 + θ2 ≤ 90 :=
sorry

end sum_of_dihedral_angles_leq_90_l16_1629


namespace machines_needed_l16_1601

theorem machines_needed (x Y : ℝ) (R : ℝ) :
  (4 * R * 6 = x) → (M * R * 6 = Y) → M = 4 * Y / x :=
by
  intros h1 h2
  sorry

end machines_needed_l16_1601


namespace max_product_three_distinct_nats_sum_48_l16_1609

open Nat

theorem max_product_three_distinct_nats_sum_48
  (a b c : ℕ) (h_distinct: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_sum: a + b + c = 48) :
  a * b * c ≤ 4080 :=
sorry

end max_product_three_distinct_nats_sum_48_l16_1609


namespace brenda_has_8_dollars_l16_1618

-- Define the amounts of money each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (25 * emma_money / 100) -- 25% more than Emma's money
def jeff_money : ℕ := 2 * daya_money / 5 -- Jeff has 2/5 of Daya's money
def brenda_money : ℕ := jeff_money + 4 -- Brenda has 4 more dollars than Jeff

-- The theorem stating the final question
theorem brenda_has_8_dollars : brenda_money = 8 :=
by
  sorry

end brenda_has_8_dollars_l16_1618


namespace find_four_digit_squares_l16_1652

theorem find_four_digit_squares (N : ℕ) (a b : ℕ) 
    (h1 : 100 ≤ N ∧ N < 10000)
    (h2 : 10 ≤ a ∧ a < 100)
    (h3 : 0 ≤ b ∧ b < 100)
    (h4 : N = 100 * a + b)
    (h5 : N = (a + b) ^ 2) : 
    N = 9801 ∨ N = 3025 ∨ N = 2025 :=
    sorry

end find_four_digit_squares_l16_1652


namespace problem1_problem2_l16_1675

-- Problem 1
theorem problem1 (α : ℝ) (h : 2 * Real.sin α - Real.cos α = 0) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10 / 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) :
  (Real.sin x ^ 3 + Real.sin x * Real.cos x ^ 2) / (1 - Real.tan x) = 7 * Real.sqrt 2 / 60 :=
sorry

end problem1_problem2_l16_1675


namespace probability_of_queen_after_first_queen_l16_1670

-- Define the standard deck
def standard_deck : Finset (Fin 54) := Finset.univ

-- Define the event of drawing the first queen
def first_queen (deck : Finset (Fin 54)) : Prop := -- placeholder defining first queen draw
  sorry

-- Define the event of drawing a queen immediately after the first queen
def queen_after_first_queen (deck : Finset (Fin 54)) : Prop :=
  sorry

-- Define the probability of an event given a condition
noncomputable def probability (event : Prop) (condition : Prop) : ℚ :=
  sorry

-- Main theorem statement
theorem probability_of_queen_after_first_queen : probability 
  (queen_after_first_queen standard_deck) (first_queen standard_deck) = 2/27 :=
sorry

end probability_of_queen_after_first_queen_l16_1670


namespace find_a_of_inequality_solution_l16_1603

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 ↔ x^2 - a * x < 0) → a = 1 := 
by 
  sorry

end find_a_of_inequality_solution_l16_1603


namespace find_num_students_l16_1631

variables (N T : ℕ)
variables (h1 : T = N * 80)
variables (h2 : 5 * 20 = 100)
variables (h3 : (T - 100) / (N - 5) = 90)

theorem find_num_students (h1 : T = N * 80) (h3 : (T - 100) / (N - 5) = 90) : N = 35 :=
sorry

end find_num_students_l16_1631


namespace molly_age_l16_1661

theorem molly_age
  (S M : ℕ)
  (h_ratio : S / M = 4 / 3)
  (h_sandy_future : S + 6 = 42)
  : M = 27 :=
sorry

end molly_age_l16_1661


namespace least_n_froods_score_l16_1679

theorem least_n_froods_score (n : ℕ) : (n * (n + 1) / 2 > 12 * n) ↔ (n > 23) := 
by 
  sorry

end least_n_froods_score_l16_1679


namespace production_days_l16_1643

theorem production_days (n : ℕ) (h₁ : (50 * n + 95) / (n + 1) = 55) : 
    n = 8 := 
    sorry

end production_days_l16_1643


namespace common_difference_range_l16_1699

theorem common_difference_range (a : ℕ → ℝ) (d : ℝ) (h : a 3 = 2) (h_pos : ∀ n, a n > 0) (h_arith : ∀ n, a (n + 1) = a n + d) : 0 ≤ d ∧ d < 1 :=
by
  sorry

end common_difference_range_l16_1699


namespace F_8_not_true_F_6_might_be_true_l16_1619

variable {n : ℕ}

-- Declare the proposition F
variable (F : ℕ → Prop)

-- Placeholder conditions
axiom condition1 : ¬ F 7
axiom condition2 : ∀ k : ℕ, k > 0 → (F k → F (k + 1))

-- Proof statements
theorem F_8_not_true : ¬ F 8 :=
by {
  sorry
}

theorem F_6_might_be_true : ¬ ¬ F 6 :=
by {
  sorry
}

end F_8_not_true_F_6_might_be_true_l16_1619


namespace problem1_problem2_l16_1682

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem1 (a : ℝ) (h1 : 2 / Real.exp 2 < a) (h2 : a < 1 / Real.exp 1) :
  ∃ (x1 x2 : ℝ), (0 < x1 ∧ x1 < 2) ∧ (0 < x2 ∧ x2 < 2) ∧ x1 ≠ x2 ∧ g x1 = a ∧ g x2 = a :=
sorry

theorem problem2 : ∀ x > 0, f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end problem1_problem2_l16_1682


namespace train_speed_l16_1685

theorem train_speed (train_length bridge_length cross_time : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : cross_time = 25) :
  (train_length + bridge_length) / cross_time = 16 :=
by
  sorry

end train_speed_l16_1685


namespace tournament_committee_count_l16_1646

-- Given conditions
def num_teams : ℕ := 5
def members_per_team : ℕ := 8
def committee_size : ℕ := 11
def nonhost_member_selection (n : ℕ) : ℕ := (n.choose 2) -- Selection of 2 members from non-host teams
def host_member_selection (n : ℕ) : ℕ := (n.choose 2)   -- Selection of 2 members from the remaining members of the host team; captain not considered in this choose as it's already selected

-- The total number of ways to form the required tournament committee
def total_committee_selections : ℕ :=
  num_teams * host_member_selection 7 * (nonhost_member_selection 8)^4

-- Proof stating the solution to the problem
theorem tournament_committee_count :
  total_committee_selections = 64534080 := by
  sorry

end tournament_committee_count_l16_1646


namespace rashmi_bus_stop_distance_l16_1694

theorem rashmi_bus_stop_distance
  (T D : ℝ)
  (h1 : 5 * (T + 10/60) = D)
  (h2 : 6 * (T - 10/60) = D) :
  D = 5 :=
by
  sorry

end rashmi_bus_stop_distance_l16_1694


namespace area_of_section_ABD_l16_1612
-- Import everything from the Mathlib library

-- Define the conditions
def is_equilateral_triangle (a b c : ℝ) (ABC_angle : ℝ) : Prop := 
  a = b ∧ b = c ∧ ABC_angle = 60

def plane_angle (angle : ℝ) : Prop := 
  angle = 35 + 18/60

def volume_of_truncated_pyramid (volume : ℝ) : Prop := 
  volume = 15

-- The main theorem based on the above conditions
theorem area_of_section_ABD
  (a b c ABC_angle : ℝ)
  (S : ℝ)
  (V : ℝ)
  (h1 : is_equilateral_triangle a b c ABC_angle)
  (h2 : plane_angle S)
  (h3 : volume_of_truncated_pyramid V) :
  ∃ (area : ℝ), area = 16.25 :=
by
  -- skipping the proof
  sorry

end area_of_section_ABD_l16_1612


namespace yellow_paint_quarts_l16_1662

theorem yellow_paint_quarts (ratio_r : ℕ) (ratio_y : ℕ) (ratio_w : ℕ) (qw : ℕ) : 
  ratio_r = 5 → ratio_y = 3 → ratio_w = 7 → qw = 21 → (qw * ratio_y) / ratio_w = 9 :=
by
  -- No proof required, inserting sorry to indicate missing proof
  sorry

end yellow_paint_quarts_l16_1662


namespace crates_sold_on_monday_l16_1606

variable (M : ℕ)
variable (h : M + 2 * M + (2 * M - 2) + M = 28)

theorem crates_sold_on_monday : M = 5 :=
by
  sorry

end crates_sold_on_monday_l16_1606


namespace cosine_60_degrees_l16_1678

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end cosine_60_degrees_l16_1678


namespace train_length_l16_1669

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end train_length_l16_1669


namespace jason_borrowed_amount_l16_1607

theorem jason_borrowed_amount (hours cycles value_per_cycle remaining_hrs remaining_value total_value: ℕ) : 
  hours = 39 → cycles = (hours / 7) → value_per_cycle = 28 → remaining_hrs = (hours % 7) →
  remaining_value = (1 + 2 + 3 + 4) →
  total_value = (cycles * value_per_cycle + remaining_value) →
  total_value = 150 := 
by {
  sorry
}

end jason_borrowed_amount_l16_1607


namespace red_ball_prob_gt_black_ball_prob_l16_1610

theorem red_ball_prob_gt_black_ball_prob (m : ℕ) (h : 8 > m) : m ≠ 10 :=
by
  sorry

end red_ball_prob_gt_black_ball_prob_l16_1610


namespace negation_proposition_l16_1626

theorem negation_proposition (x y : ℝ) :
  (¬ ∃ (x y : ℝ), 2 * x + 3 * y + 3 < 0) ↔ (∀ (x y : ℝ), 2 * x + 3 * y + 3 ≥ 0) :=
by {
  sorry
}

end negation_proposition_l16_1626


namespace fraction_of_students_with_buddy_l16_1602

variable (s n : ℕ)

theorem fraction_of_students_with_buddy (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l16_1602


namespace smallest_a_l16_1624

theorem smallest_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 96 * a^2 = b^3) : a = 12 :=
by
  sorry

end smallest_a_l16_1624


namespace workers_task_solution_l16_1615

-- Defining the variables for the number of days worked by A and B
variables (x y : ℕ)

-- Defining the total earnings for A and B
def total_earnings_A := 30
def total_earnings_B := 14

-- Condition: B worked 3 days less than A
def condition1 := y = x - 3

-- Daily wages of A and B
def daily_wage_A := total_earnings_A / x
def daily_wage_B := total_earnings_B / y

-- New scenario conditions
def new_days_A := x - 2
def new_days_B := y + 5

-- New total earnings in the scenario where they work changed days
def new_earnings_A := new_days_A * daily_wage_A
def new_earnings_B := new_days_B * daily_wage_B

-- Final proof to show the number of days worked and daily wages satisfying the conditions
theorem workers_task_solution 
  (h1 : y = x - 3)
  (h2 : new_earnings_A = new_earnings_B) 
  (hx : x = 10)
  (hy : y = 7) 
  (wageA : daily_wage_A = 3) 
  (wageB : daily_wage_B = 2) : 
  x = 10 ∧ y = 7 ∧ daily_wage_A = 3 ∧ daily_wage_B = 2 :=
by {
  sorry  -- Proof is skipped as instructed
}

end workers_task_solution_l16_1615


namespace ratio_problem_l16_1684

variable (a b c d : ℚ)

theorem ratio_problem
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end ratio_problem_l16_1684


namespace sequence_sum_n_eq_21_l16_1614

theorem sequence_sum_n_eq_21 (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k, a (k + 1) = a k + 1)
  (h3 : ∀ n, S n = (n * (n + 1)) / 2)
  (h4 : S n = 21) :
  n = 6 :=
sorry

end sequence_sum_n_eq_21_l16_1614


namespace contestant_wins_quiz_l16_1644

noncomputable def winProbability : ℚ :=
  let p_correct := (1 : ℚ) / 3
  let p_wrong := (2 : ℚ) / 3
  let binom := Nat.choose  -- binomial coefficient function
  ((binom 4 2 * (p_correct ^ 2) * (p_wrong ^ 2)) +
   (binom 4 3 * (p_correct ^ 3) * (p_wrong ^ 1)) +
   (binom 4 4 * (p_correct ^ 4) * (p_wrong ^ 0)))

theorem contestant_wins_quiz :
  winProbability = 11 / 27 :=
by
  simp [winProbability, Nat.choose]
  norm_num
  done

end contestant_wins_quiz_l16_1644


namespace minewaska_state_park_l16_1693

variable (B H : Nat)

theorem minewaska_state_park (hikers_bike_riders_sum : H + B = 676) (hikers_more_than_bike_riders : H = B + 178) : H = 427 :=
sorry

end minewaska_state_park_l16_1693


namespace problem_cos_tan_half_l16_1656

open Real

theorem problem_cos_tan_half
  (α : ℝ)
  (hcos : cos α = -4/5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 :=
  sorry

end problem_cos_tan_half_l16_1656


namespace subtract_fractions_l16_1676

theorem subtract_fractions : (18 / 42 - 3 / 8) = 3 / 56 :=
by
  sorry

end subtract_fractions_l16_1676


namespace maryann_free_time_l16_1651

theorem maryann_free_time
    (x : ℕ)
    (expensive_time : ℕ := 8)
    (friends : ℕ := 3)
    (total_time : ℕ := 42)
    (lockpicking_time : 3 * (x + expensive_time) = total_time) : 
    x = 6 :=
by
  sorry

end maryann_free_time_l16_1651


namespace top_leftmost_rectangle_is_B_l16_1677

structure Rectangle :=
  (w x y z : ℕ)

def RectangleA := Rectangle.mk 5 1 9 2
def RectangleB := Rectangle.mk 2 0 6 3
def RectangleC := Rectangle.mk 6 7 4 1
def RectangleD := Rectangle.mk 8 4 3 5
def RectangleE := Rectangle.mk 7 3 8 0

-- Problem Statement: Given these rectangles, prove that the top leftmost rectangle is B.
theorem top_leftmost_rectangle_is_B 
  (A : Rectangle := RectangleA)
  (B : Rectangle := RectangleB)
  (C : Rectangle := RectangleC)
  (D : Rectangle := RectangleD)
  (E : Rectangle := RectangleE) : 
  B = Rectangle.mk 2 0 6 3 := 
sorry

end top_leftmost_rectangle_is_B_l16_1677


namespace B_alone_finishes_in_21_days_l16_1645

theorem B_alone_finishes_in_21_days (W_A W_B : ℝ) (h1 : W_A = 0.5 * W_B) (h2 : W_A + W_B = 1 / 14) : W_B = 1 / 21 :=
by sorry

end B_alone_finishes_in_21_days_l16_1645


namespace selling_price_of_cycle_l16_1617

theorem selling_price_of_cycle
  (cost_price : ℕ)
  (gain_percent_decimal : ℚ)
  (h_cp : cost_price = 850)
  (h_gpd : gain_percent_decimal = 27.058823529411764 / 100) :
  ∃ selling_price : ℚ, selling_price = cost_price * (1 + gain_percent_decimal) ∧ selling_price = 1080 := 
by
  use (cost_price * (1 + gain_percent_decimal))
  sorry

end selling_price_of_cycle_l16_1617


namespace evaluate_expression_at_x_eq_3_l16_1604

theorem evaluate_expression_at_x_eq_3 :
  (3 ^ 3) ^ (3 ^ 3) = 7625597484987 := by
  sorry

end evaluate_expression_at_x_eq_3_l16_1604


namespace repeating_decimal_product_l16_1686

def repeating_decimal_12 := 12 / 99
def repeating_decimal_34 := 34 / 99

theorem repeating_decimal_product : (repeating_decimal_12 * repeating_decimal_34) = 136 / 3267 := by
  sorry

end repeating_decimal_product_l16_1686


namespace face_opposite_A_is_F_l16_1627

structure Cube where
  adjacency : String → String → Prop
  exists_face : ∃ a b c d e f : String, True

variable 
  (C : Cube)
  (adjA_B : C.adjacency "A" "B")
  (adjA_C : C.adjacency "A" "C")
  (adjB_D : C.adjacency "B" "D")

theorem face_opposite_A_is_F : 
  ∃ f : String, f = "F" ∧ ∀ g : String, (C.adjacency "A" g → g ≠ "F") :=
by 
  sorry

end face_opposite_A_is_F_l16_1627
