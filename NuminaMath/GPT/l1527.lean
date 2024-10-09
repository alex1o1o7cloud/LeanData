import Mathlib

namespace choose_most_suitable_l1527_152709

def Survey := ℕ → Bool
structure Surveys :=
  (A B C D : Survey)
  (census_suitable : Survey)

theorem choose_most_suitable (s : Surveys) :
  s.census_suitable = s.C :=
sorry

end choose_most_suitable_l1527_152709


namespace solution_greater_iff_l1527_152731

variables {c c' d d' : ℝ}
variables (hc : c ≠ 0) (hc' : c' ≠ 0)

theorem solution_greater_iff : (∃ x, x = -d / c) > (∃ x, x = -d' / c') ↔ (d' / c') < (d / c) :=
by sorry

end solution_greater_iff_l1527_152731


namespace find_cost_price_l1527_152790

def selling_price : ℝ := 150
def profit_percentage : ℝ := 25

theorem find_cost_price (cost_price : ℝ) (h : profit_percentage = ((selling_price - cost_price) / cost_price) * 100) : 
  cost_price = 120 := 
sorry

end find_cost_price_l1527_152790


namespace P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l1527_152727

-- Conditions
def center_C : (ℝ × ℝ) := (6, 8)
def radius : ℝ := 10
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 100
def origin_O : (ℝ × ℝ) := (0, 0)

-- (a) Point of intersection of the circle with the x-axis
def point_P : (ℝ × ℝ) := (12, 0)
theorem P_on_x_axis : circle_eq (point_P.1) (point_P.2) ∧ point_P.2 = 0 := sorry

-- (b) Point on the circle with maximum y-coordinate
def point_Q : (ℝ × ℝ) := (6, 18)
theorem Q_max_y : circle_eq (point_Q.1) (point_Q.2) ∧ ∀ y : ℝ, (circle_eq 6 y → y ≤ 18) := sorry

-- (c) Point on the circle such that ∠PQR = 90°
def point_R : (ℝ × ℝ) := (0, 16)
theorem PQR_90_deg : circle_eq (point_R.1) (point_R.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧ (point_P.1 - point_R.1) * (Q.1 - point_Q.1) + (point_P.2 - point_R.2) * (Q.2 - point_Q.2) = 0 := sorry

-- (d) Two points on the circle such that ∠PQS = ∠PQT = 45°
def point_S : (ℝ × ℝ) := (14, 14)
def point_T : (ℝ × ℝ) := (-2, 2)
theorem PQS_PQT_45_deg : circle_eq (point_S.1) (point_S.2) ∧ circle_eq (point_T.1) (point_T.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧
  ((point_P.1 - Q.1) * (point_S.1 - Q.1) + (point_P.2 - Q.2) * (point_S.2 - Q.2) =
  (point_P.1 - Q.1) * (point_T.1 - Q.1) + (point_P.2 - Q.2) * (point_T.2 - Q.2)) := sorry

end P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l1527_152727


namespace jason_optimal_reroll_probability_l1527_152757

-- Define the probability function based on the three dice roll problem
def probability_of_rerolling_two_dice : ℚ := 
  -- As per the problem, the computed and fixed probability is 7/64.
  7 / 64

-- Prove that Jason's optimal strategy leads to rerolling exactly two dice with a probability of 7/64.
theorem jason_optimal_reroll_probability : probability_of_rerolling_two_dice = 7 / 64 := 
  sorry

end jason_optimal_reroll_probability_l1527_152757


namespace largest_divisor_of_m_p1_l1527_152710

theorem largest_divisor_of_m_p1 (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 6 ∣ m :=
sorry

end largest_divisor_of_m_p1_l1527_152710


namespace sum_first_15_odd_from_5_l1527_152798

theorem sum_first_15_odd_from_5 : 
  let a₁ := 5 
  let d := 2 
  let n := 15 
  let a₁₅ := a₁ + (n - 1) * d 
  let S := n * (a₁ + a₁₅) / 2 
  S = 285 := by 
  sorry

end sum_first_15_odd_from_5_l1527_152798


namespace possible_landing_l1527_152736

-- There are 1985 airfields
def num_airfields : ℕ := 1985

-- 50 airfields where planes could potentially land
def num_land_airfields : ℕ := 50

-- Define the structure of the problem
structure AirfieldSetup :=
  (airfields : Fin num_airfields → Fin num_land_airfields)

-- There exists a configuration such that the conditions are met
theorem possible_landing : ∃ (setup : AirfieldSetup), 
  (∀ i : Fin num_airfields, -- For each airfield
    ∃ j : Fin num_land_airfields, -- There exists a landing airfield
    setup.airfields i = j) -- The plane lands at this airfield.
:=
sorry

end possible_landing_l1527_152736


namespace polygon_sum_13th_position_l1527_152771

theorem polygon_sum_13th_position :
  let sum_n : ℕ := (100 * 101) / 2;
  2 * sum_n = 10100 :=
by
  sorry

end polygon_sum_13th_position_l1527_152771


namespace triangle_inequality_l1527_152733

theorem triangle_inequality {A B C : ℝ} {n : ℕ} (h : B = n * C) (hA : A + B + C = π) :
  B ≤ n * C :=
by
  sorry

end triangle_inequality_l1527_152733


namespace find_f_5_l1527_152756

section
variables (f : ℝ → ℝ)

-- Given condition
def functional_equation (x : ℝ) : Prop := x * f x = 2 * f (1 - x) + 1

-- Prove that f(5) = 1/12 given the condition
theorem find_f_5 (h : ∀ x, functional_equation f x) : f 5 = 1 / 12 :=
sorry
end

end find_f_5_l1527_152756


namespace remainder_twice_original_l1527_152719

def findRemainder (N : ℕ) (D : ℕ) (r : ℕ) : ℕ :=
  2 * N % D

theorem remainder_twice_original
  (N : ℕ) (D : ℕ)
  (hD : D = 367)
  (hR : N % D = 241) :
  findRemainder N D 2 = 115 := by
  sorry

end remainder_twice_original_l1527_152719


namespace abs_eq_5_iff_l1527_152738

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l1527_152738


namespace polygon_diagonals_l1527_152786

theorem polygon_diagonals (n : ℕ) (h : 20 = n) : (n * (n - 3)) / 2 = 170 :=
by
  sorry

end polygon_diagonals_l1527_152786


namespace solve_m_l1527_152759

def f (x : ℝ) := 4 * x ^ 2 - 3 * x + 5
def g (x : ℝ) (m : ℝ) := x ^ 2 - m * x - 8

theorem solve_m : ∃ (m : ℝ), f 8 - g 8 m = 20 ∧ m = -25.5 := by
  sorry

end solve_m_l1527_152759


namespace brittany_money_times_brooke_l1527_152772

theorem brittany_money_times_brooke 
  (kent_money : ℕ) (brooke_money : ℕ) (brittany_money : ℕ) (alison_money : ℕ)
  (h1 : kent_money = 1000)
  (h2 : brooke_money = 2 * kent_money)
  (h3 : alison_money = 4000)
  (h4 : alison_money = brittany_money / 2) :
  brittany_money = 4 * brooke_money :=
by
  sorry

end brittany_money_times_brooke_l1527_152772


namespace find_initial_quantities_l1527_152779

/-- 
Given:
- x + y = 92
- (2/5) * x + (1/4) * y = 26

Prove:
- x = 20
- y = 72
-/
theorem find_initial_quantities (x y : ℝ) (h1 : x + y = 92) (h2 : (2/5) * x + (1/4) * y = 26) :
  x = 20 ∧ y = 72 :=
sorry

end find_initial_quantities_l1527_152779


namespace num_ordered_quadruples_l1527_152795

theorem num_ordered_quadruples (n : ℕ) :
  ∃ (count : ℕ), count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3) ∧
  (∀ (k1 k2 k3 k4 : ℕ), k1 ≤ n ∧ k2 ≤ n ∧ k3 ≤ n ∧ k4 ≤ n → 
    ((k1 + k3) / 2 = (k2 + k4) / 2) → 
    count = (1 / 3 : ℚ) * (n + 1) * (2 * n^2 + 4 * n + 3)) :=
by sorry

end num_ordered_quadruples_l1527_152795


namespace petya_maximum_margin_l1527_152797

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l1527_152797


namespace random_events_l1527_152758

/-- Definition of what constitutes a random event --/
def is_random_event (e : String) : Prop :=
  e = "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)" ∨
  e = "Forgetting the last digit of a phone number, randomly pressing and it is correct" ∨
  e = "Winning the first prize in a sports lottery"

/-- Define the specific events --/
def event_1 := "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)"
def event_2 := "Forgetting the last digit of a phone number, randomly pressing and it is correct"
def event_3 := "Opposite electric charges attract each other"
def event_4 := "Winning the first prize in a sports lottery"

/-- Lean 4 statement for the proof problem --/
theorem random_events :
  (is_random_event event_1) ∧
  (is_random_event event_2) ∧
  ¬(is_random_event event_3) ∧
  (is_random_event event_4) :=
by 
  sorry

end random_events_l1527_152758


namespace angle_bisector_median_ineq_l1527_152732

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (l_a l_b l_c m_a m_b m_c : ℝ)

theorem angle_bisector_median_ineq
  (hl_a : l_a > 0) (hl_b : l_b > 0) (hl_c : l_c > 0)
  (hm_a : m_a > 0) (hm_b : m_b > 0) (hm_c : m_c > 0) :
  l_a / m_a + l_b / m_b + l_c / m_c > 1 :=
sorry

end angle_bisector_median_ineq_l1527_152732


namespace blue_bordered_area_on_outer_sphere_l1527_152730

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l1527_152730


namespace profit_days_l1527_152740

theorem profit_days (total_days : ℕ) (mean_profit_month first_half_days second_half_days : ℕ)
  (mean_profit_first_half mean_profit_second_half : ℕ)
  (h1 : mean_profit_month * total_days = (mean_profit_first_half * first_half_days + mean_profit_second_half * second_half_days))
  (h2 : first_half_days + second_half_days = total_days)
  (h3 : mean_profit_month = 350)
  (h4 : mean_profit_first_half = 225)
  (h5 : mean_profit_second_half = 475)
  (h6 : total_days = 30) : 
  first_half_days = 15 ∧ second_half_days = 15 := 
by 
  sorry

end profit_days_l1527_152740


namespace seven_distinct_integers_exist_pair_l1527_152744

theorem seven_distinct_integers_exist_pair (a : Fin 7 → ℕ) (h_distinct : Function.Injective a)
  (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 126) :
  ∃ i j : Fin 7, i ≠ j ∧ (1 / 2 : ℚ) ≤ (a i : ℚ) / a j ∧ (a i : ℚ) / a j ≤ 2 := sorry

end seven_distinct_integers_exist_pair_l1527_152744


namespace larger_of_two_numbers_l1527_152796

theorem larger_of_two_numbers (x y : ℕ) (h1 : x * y = 24) (h2 : x + y = 11) : max x y = 8 :=
sorry

end larger_of_two_numbers_l1527_152796


namespace daps_equivalent_to_dips_l1527_152794

theorem daps_equivalent_to_dips (daps dops dips : ℕ) 
  (h1 : 4 * daps = 3 * dops) 
  (h2 : 2 * dops = 7 * dips) :
  35 * dips = 20 * daps :=
by
  sorry

end daps_equivalent_to_dips_l1527_152794


namespace ratio_problem_l1527_152782

theorem ratio_problem (x : ℕ) : (20 / 1 : ℝ) = (x / 10 : ℝ) → x = 200 := by
  sorry

end ratio_problem_l1527_152782


namespace trapezoid_area_l1527_152734

theorem trapezoid_area (x y : ℝ) (hx : y^2 + x^2 = 625) (hy : y^2 + (25 - x)^2 = 900) :
  1 / 2 * (11 + 36) * 24 = 564 :=
by
  sorry

end trapezoid_area_l1527_152734


namespace prob_B_given_A_l1527_152778

theorem prob_B_given_A (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.06) (h2 : P_B = 0.08) (h3 : P_A_and_B = 0.02) :
  (P_A_and_B / P_A) = (1 / 3) :=
by
  -- substitute values
  sorry

end prob_B_given_A_l1527_152778


namespace total_cantaloupes_l1527_152717

def cantaloupes (fred : ℕ) (tim : ℕ) := fred + tim

theorem total_cantaloupes : cantaloupes 38 44 = 82 := by
  sorry

end total_cantaloupes_l1527_152717


namespace cost_of_bananas_l1527_152754

/-- We are given that the rate of bananas is $6 per 3 kilograms. -/
def rate_per_3_kg : ℝ := 6

/-- We need to find the cost for 12 kilograms of bananas. -/
def weight_in_kg : ℝ := 12

/-- We are asked to prove that the cost of 12 kilograms of bananas is $24. -/
theorem cost_of_bananas (rate_per_3_kg weight_in_kg : ℝ) :
  (weight_in_kg / 3) * rate_per_3_kg = 24 :=
by
  sorry

end cost_of_bananas_l1527_152754


namespace average_brown_mms_l1527_152721

def brown_mms_bag_1 := 9
def brown_mms_bag_2 := 12
def brown_mms_bag_3 := 8
def brown_mms_bag_4 := 8
def brown_mms_bag_5 := 3

def total_brown_mms : ℕ := brown_mms_bag_1 + brown_mms_bag_2 + brown_mms_bag_3 + brown_mms_bag_4 + brown_mms_bag_5

theorem average_brown_mms :
  (total_brown_mms / 5) = 8 := by
  rw [total_brown_mms]
  norm_num
  sorry

end average_brown_mms_l1527_152721


namespace not_periodic_l1527_152711

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + Real.sin (a * x)

theorem not_periodic {a : ℝ} (ha : Irrational a) : ¬ ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f a (x + T) = f a x :=
  sorry

end not_periodic_l1527_152711


namespace problem_solution_l1527_152791

theorem problem_solution
  (a d : ℝ)
  (h : (∀ x : ℝ, (x - 3) * (x + a) = x^2 + d * x - 18)) :
  d = 3 := 
sorry

end problem_solution_l1527_152791


namespace s_eq_sin_c_eq_cos_l1527_152751

open Real

variables (s c : ℝ → ℝ)

-- Conditions
def s_prime := ∀ x, deriv s x = c x
def c_prime := ∀ x, deriv c x = -s x
def initial_conditions := (s 0 = 0) ∧ (c 0 = 1)

-- Theorem to prove
theorem s_eq_sin_c_eq_cos
  (h1 : s_prime s c)
  (h2 : c_prime s c)
  (h3 : initial_conditions s c) :
  (∀ x, s x = sin x) ∧ (∀ x, c x = cos x) :=
sorry

end s_eq_sin_c_eq_cos_l1527_152751


namespace percentage_increase_l1527_152718

theorem percentage_increase (total_capacity : ℝ) (additional_water : ℝ) (percentage_capacity : ℝ) (current_water : ℝ) : 
    additional_water + current_water = percentage_capacity * total_capacity →
    percentage_capacity = 0.70 →
    total_capacity = 1857.1428571428573 →
    additional_water = 300 →
    current_water = ((percentage_capacity * total_capacity) - additional_water) →
    (additional_water / current_water) * 100 = 30 :=
by
    sorry

end percentage_increase_l1527_152718


namespace factorize_a3_sub_a_l1527_152747

theorem factorize_a3_sub_a (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) :=
sorry

end factorize_a3_sub_a_l1527_152747


namespace value_of_x_is_10_l1527_152746

-- Define the conditions
def condition1 (x : ℕ) : ℕ := 3 * x
def condition2 (x : ℕ) : ℕ := (26 - x) + 14

-- Define the proof problem
theorem value_of_x_is_10 (x : ℕ) (h1 : condition1 x = condition2 x) : x = 10 :=
by {
  sorry
}

end value_of_x_is_10_l1527_152746


namespace inequality_solution_l1527_152725

/-- Define conditions and state the corresponding theorem -/
theorem inequality_solution (a x : ℝ) (h : a < 0) : ax - 1 > 0 ↔ x < 1 / a :=
by sorry

end inequality_solution_l1527_152725


namespace new_class_mean_l1527_152705

theorem new_class_mean :
  let students1 := 45
  let mean1 := 80
  let students2 := 4
  let mean2 := 85
  let students3 := 1
  let score3 := 90
  let total_students := students1 + students2 + students3
  let total_score := (students1 * mean1) + (students2 * mean2) + (students3 * score3)
  let class_mean := total_score / total_students
  class_mean = 80.6 := 
by
  sorry

end new_class_mean_l1527_152705


namespace sleep_hours_for_desired_average_l1527_152781

theorem sleep_hours_for_desired_average 
  (s_1 s_2 : ℝ) (h_1 h_2 : ℝ) (k : ℝ) 
  (h_inverse_relation : ∀ s h, s * h = k)
  (h_s1 : s_1 = 75)
  (h_h1 : h_1 = 6)
  (h_average : (s_1 + s_2) / 2 = 85) : 
  h_2 = 450 / 95 := 
by 
  sorry

end sleep_hours_for_desired_average_l1527_152781


namespace odds_burning_out_during_second_period_l1527_152716

def odds_burning_out_during_first_period := 1 / 3
def odds_not_burning_out_first_period := 1 - odds_burning_out_during_first_period
def odds_not_burning_out_next_period := odds_not_burning_out_first_period / 2

theorem odds_burning_out_during_second_period :
  (1 - odds_not_burning_out_next_period) = 2 / 3 := by
  sorry

end odds_burning_out_during_second_period_l1527_152716


namespace keychain_arrangement_count_l1527_152707

-- Definitions of the keys
inductive Key
| house
| car
| office
| other1
| other2

-- Function to count the number of distinct arrangements on a keychain
noncomputable def distinct_keychain_arrangements : ℕ :=
  sorry -- This will be the placeholder for the proof

-- The ultimate theorem stating the solution
theorem keychain_arrangement_count : distinct_keychain_arrangements = 2 :=
  sorry -- This will be the placeholder for the proof

end keychain_arrangement_count_l1527_152707


namespace M_plus_N_eq_2_l1527_152722

noncomputable def M : ℝ := 1^5 + 2^4 * 3^3 - (4^2 / 5^1)
noncomputable def N : ℝ := 1^5 - 2^4 * 3^3 + (4^2 / 5^1)

theorem M_plus_N_eq_2 : M + N = 2 := by
  sorry

end M_plus_N_eq_2_l1527_152722


namespace solution_interval_l1527_152749

theorem solution_interval (x : ℝ) (h1 : x / 2 ≤ 5 - x) (h2 : 5 - x < -3 * (2 + x)) :
  x < -11 / 2 := 
sorry

end solution_interval_l1527_152749


namespace inequality_no_solution_iff_a_le_neg3_l1527_152742

theorem inequality_no_solution_iff_a_le_neg3 (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x + 2| < a)) ↔ a ≤ -3 := 
sorry

end inequality_no_solution_iff_a_le_neg3_l1527_152742


namespace integer_for_all_n_l1527_152708

theorem integer_for_all_n
  (x y : ℝ)
  (f : ℕ → ℤ)
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → f n = ((x^n - y^n) / (x - y))) :
  ∀ n : ℕ, 0 < n → f n = ((x^n - y^n) / (x - y)) :=
by sorry

end integer_for_all_n_l1527_152708


namespace find_larger_number_l1527_152743

theorem find_larger_number :
  ∃ (L S : ℕ), L - S = 1365 ∧ L = 6 * S + 15 ∧ L = 1635 :=
sorry

end find_larger_number_l1527_152743


namespace number_of_sides_l1527_152787

def side_length : ℕ := 16
def perimeter : ℕ := 80

theorem number_of_sides (h1: side_length = 16) (h2: perimeter = 80) : (perimeter / side_length = 5) :=
by
  -- Proof should be inserted here.
  sorry

end number_of_sides_l1527_152787


namespace function_properties_l1527_152720

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem function_properties (a b : ℝ) (h : (a - 1) ^ 2 - 4 * b < 0) : 
  (∀ x : ℝ, f x a b > x) ∧ (∀ x : ℝ, f (f x a b) a b > x) ∧ (a + b > 0) :=
by
  sorry

end function_properties_l1527_152720


namespace polygon_sides_l1527_152752

theorem polygon_sides (n : ℕ) (hn : (n - 2) * 180 = 5 * 360) : n = 12 :=
by
  sorry

end polygon_sides_l1527_152752


namespace Jason_saturday_hours_l1527_152769

theorem Jason_saturday_hours (x y : ℕ) 
  (h1 : 4 * x + 6 * y = 88)
  (h2 : x + y = 18) : 
  y = 8 :=
sorry

end Jason_saturday_hours_l1527_152769


namespace perpendicular_vectors_t_values_l1527_152793

variable (t : ℝ)
def a := (t, 0, -1)
def b := (2, 5, t^2)

theorem perpendicular_vectors_t_values (h : (2 * t + 0 * 5 + -1 * t^2) = 0) : t = 0 ∨ t = 2 :=
by sorry

end perpendicular_vectors_t_values_l1527_152793


namespace best_fitting_model_l1527_152714

theorem best_fitting_model :
  ∀ R1 R2 R3 R4 : ℝ, 
  R1 = 0.21 → R2 = 0.80 → R3 = 0.50 → R4 = 0.98 → 
  abs (R4 - 1) < abs (R1 - 1) ∧ abs (R4 - 1) < abs (R2 - 1) 
    ∧ abs (R4 - 1) < abs (R3 - 1) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  exact sorry

end best_fitting_model_l1527_152714


namespace isosceles_triangle_sides_l1527_152763

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides_l1527_152763


namespace eval_x_power_x_power_x_at_3_l1527_152750

theorem eval_x_power_x_power_x_at_3 : (3^3)^(3^3) = 27^27 := by
    sorry

end eval_x_power_x_power_x_at_3_l1527_152750


namespace smallest_of_powers_l1527_152761

theorem smallest_of_powers :
  min (2^55) (min (3^44) (min (5^33) (6^22))) = 2^55 :=
by
  sorry

end smallest_of_powers_l1527_152761


namespace poly_has_integer_roots_iff_a_eq_one_l1527_152735

-- Definition: a positive real number
def pos_real (a : ℝ) : Prop := a > 0

-- The polynomial
def p (a : ℝ) (x : ℝ) : ℝ := a^3 * x^3 + a^2 * x^2 + a * x + a

-- The main theorem
theorem poly_has_integer_roots_iff_a_eq_one (a : ℝ) (x : ℤ) :
  (pos_real a ∧ ∃ x : ℤ, p a x = 0) ↔ a = 1 :=
by sorry

end poly_has_integer_roots_iff_a_eq_one_l1527_152735


namespace rabbit_jumps_before_dog_catches_l1527_152760

/-- Prove that the number of additional jumps the rabbit can make before the dog catches up is 700,
    given the initial conditions:
      1. The rabbit has a 50-jump head start.
      2. The dog makes 5 jumps in the time the rabbit makes 6 jumps.
      3. The distance covered by 7 jumps of the dog equals the distance covered by 9 jumps of the rabbit. -/
theorem rabbit_jumps_before_dog_catches (h_head_start : ℕ) (h_time_ratio : ℚ) (h_distance_ratio : ℚ) : 
    h_head_start = 50 → h_time_ratio = 5/6 → h_distance_ratio = 7/9 → 
    ∃ (rabbit_additional_jumps : ℕ), rabbit_additional_jumps = 700 :=
by
  intro h_head_start_intro h_time_ratio_intro h_distance_ratio_intro
  have rabbit_additional_jumps := 700
  use rabbit_additional_jumps
  sorry

end rabbit_jumps_before_dog_catches_l1527_152760


namespace least_negative_b_l1527_152737

theorem least_negative_b (x b : ℤ) (h1 : x^2 + b * x = 22) (h2 : b < 0) : b = -21 :=
sorry

end least_negative_b_l1527_152737


namespace sum_solution_equation_l1527_152703

theorem sum_solution_equation (n : ℚ) : (∃ x : ℚ, (n / x = 3 - n) ∧ (x = 1 / (n + (3 - n)))) → n = 3 / 4 := by
  intros h
  sorry

end sum_solution_equation_l1527_152703


namespace translate_line_upwards_l1527_152701

theorem translate_line_upwards (x y y' : ℝ) (h : y = -2 * x) (t : y' = y + 4) : y' = -2 * x + 4 :=
by
  sorry

end translate_line_upwards_l1527_152701


namespace yankees_mets_ratio_l1527_152724

-- Given conditions
def num_mets_fans : ℕ := 104
def total_fans : ℕ := 390
def ratio_mets_to_redsox : ℚ := 4 / 5

-- Definitions
def num_redsox_fans (M : ℕ) := (5 / 4) * M
def num_yankees_fans (Y M B : ℕ) := (total_fans - M - B)

-- Theorem statement
theorem yankees_mets_ratio (Y M B : ℕ)
  (h1 : M = num_mets_fans)
  (h2 : Y + M + B = total_fans)
  (h3 : (M : ℚ) / (B : ℚ) = ratio_mets_to_redsox) :
  (Y : ℚ) / (M : ℚ) = 3 / 2 :=
sorry

end yankees_mets_ratio_l1527_152724


namespace fifth_graders_buy_more_l1527_152777

-- Define the total payments made by eighth graders and fifth graders
def eighth_graders_payment : ℕ := 210
def fifth_graders_payment : ℕ := 240
def number_of_fifth_graders : ℕ := 25

-- The price per notebook in whole cents
def price_per_notebook (p : ℕ) : Prop :=
  ∃ k1 k2 : ℕ, k1 * p = eighth_graders_payment ∧ k2 * p = fifth_graders_payment

-- The difference in the number of notebooks bought by the fifth graders and the eighth graders
def notebook_difference (p : ℕ) : ℕ :=
  let eighth_graders_notebooks := eighth_graders_payment / p
  let fifth_graders_notebooks := fifth_graders_payment / p
  fifth_graders_notebooks - eighth_graders_notebooks

-- Theorem stating the difference in the number of notebooks equals 2
theorem fifth_graders_buy_more (p : ℕ) (h : price_per_notebook p) : notebook_difference p = 2 :=
  sorry

end fifth_graders_buy_more_l1527_152777


namespace quadratic_coefficients_l1527_152766

theorem quadratic_coefficients (a b c : ℝ) (h₀: 0 < a) 
  (h₁: |a + b + c| = 3) 
  (h₂: |4 * a + 2 * b + c| = 3) 
  (h₃: |9 * a + 3 * b + c| = 3) : 
  (a = 6 ∧ b = -24 ∧ c = 21) ∨ (a = 3 ∧ b = -15 ∧ c = 15) ∨ (a = 3 ∧ b = -9 ∧ c = 3) :=
sorry

end quadratic_coefficients_l1527_152766


namespace compare_exponents_l1527_152748

theorem compare_exponents :
  let a := (3 / 2) ^ 0.1
  let b := (3 / 2) ^ 0.2
  let c := (3 / 2) ^ 0.08
  c < a ∧ a < b := by
  sorry

end compare_exponents_l1527_152748


namespace range_of_a_l1527_152768

def p (a x : ℝ) : Prop := a * x^2 + a * x - 1 < 0
def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

theorem range_of_a (a : ℝ) :
  ¬ (∀ x, p a x ∨ q a) → a ≤ -4 ∨ 1 ≤ a :=
by sorry

end range_of_a_l1527_152768


namespace incorrect_conclusion_l1527_152788

theorem incorrect_conclusion :
  ∃ (a x y : ℝ), 
  (x + 3 * y = 4 - a ∧ x - y = 3 * a) ∧ 
  (∀ (xa ya : ℝ), (xa = 2) → (x = 2 * xa + 1) ∧ (y = 1 - xa) → ¬ (xa + ya = 4 - xa)) :=
sorry

end incorrect_conclusion_l1527_152788


namespace jimin_rank_l1527_152773

theorem jimin_rank (seokjin_rank : ℕ) (h1 : seokjin_rank = 4) (h2 : ∃ jimin_rank, jimin_rank = seokjin_rank + 1) : 
  ∃ jimin_rank, jimin_rank = 5 := 
by
  sorry

end jimin_rank_l1527_152773


namespace kim_total_water_drank_l1527_152726

noncomputable def total_water_kim_drank : Float :=
  let water_from_bottle := 1.5 * 32
  let water_from_can := 12
  let shared_bottle := (3 / 5) * 32
  water_from_bottle + water_from_can + shared_bottle

theorem kim_total_water_drank :
  total_water_kim_drank = 79.2 :=
by
  -- Proof skipped
  sorry

end kim_total_water_drank_l1527_152726


namespace conversion_1_conversion_2_conversion_3_l1527_152767

theorem conversion_1 : 2 * 1000 = 2000 := sorry

theorem conversion_2 : 9000 / 1000 = 9 := sorry

theorem conversion_3 : 8 * 1000 = 8000 := sorry

end conversion_1_conversion_2_conversion_3_l1527_152767


namespace division_problem_l1527_152715

theorem division_problem :
  250 / (5 + 12 * 3^2) = 250 / 113 :=
by sorry

end division_problem_l1527_152715


namespace largest_divisor_l1527_152723

theorem largest_divisor (A B : ℕ) (h : 24 = A * B + 4) : A ≤ 20 :=
sorry

end largest_divisor_l1527_152723


namespace polynomial_remainder_division_l1527_152785

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (3 * x^7 + 2 * x^5 - 5 * x^3 + x^2 - 9) % (x^2 + 2 * x + 1) = 14 * x - 16 :=
by
  sorry

end polynomial_remainder_division_l1527_152785


namespace youngest_person_age_l1527_152741

theorem youngest_person_age (n : ℕ) (average_age : ℕ) (average_age_when_youngest_born : ℕ) 
    (h1 : n = 7) (h2 : average_age = 30) (h3 : average_age_when_youngest_born = 24) :
    ∃ Y : ℚ, Y = 66 / 7 :=
by
  sorry

end youngest_person_age_l1527_152741


namespace conditional_probability_l1527_152745

variable (P : ℕ → ℚ)
variable (A B : ℕ)

def EventRain : Prop := P A = 4/15
def EventWind : Prop := P B = 2/15
def EventBoth : Prop := P (A * B) = 1/10

theorem conditional_probability 
  (h1 : EventRain P A) 
  (h2 : EventWind P B) 
  (h3 : EventBoth P A B) 
  : (P (A * B) / P A) = 3 / 8 := 
by
  sorry

end conditional_probability_l1527_152745


namespace some_number_value_correct_l1527_152762

noncomputable def value_of_some_number (a : ℕ) : ℕ :=
  (a^3) / (25 * 45 * 49)

theorem some_number_value_correct :
  value_of_some_number 105 = 21 := by
  sorry

end some_number_value_correct_l1527_152762


namespace evaluate_expression_l1527_152739

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end evaluate_expression_l1527_152739


namespace hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l1527_152700

-- Definitions for conditions of the problem
def ellipse_C1 (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def is_sister_conic_section (e1 e2 : ℝ) : Prop :=
  e1 * e2 = Real.sqrt 15 / 4

def hyperbola_C2 (x y : ℝ) : Prop := (x^2) / 4 - y^2 = 1

variable {b : ℝ} (hb : 0 < b ∧ b < 2)
variable {e1 e2 : ℝ} (heccentricities : is_sister_conic_section e1 e2)

theorem hyperbola_C2_equation :
  ∃ (x y : ℝ), ellipse_C1 x y b → hyperbola_C2 x y := sorry

theorem constant_ratio_kAM_kBN (G : ℝ × ℝ) :
  G = (4,0) → 
  ∀ (M N : ℝ × ℝ) (kAM kBN : ℝ), 
  (kAM / kBN = -1/3) := sorry

theorem range_of_w_kAM_kBN (kAM kBN : ℝ) :
  ∃ (w : ℝ),
  w = kAM^2 + (2 / 3) * kBN →
  (w ∈ Set.Icc (-3 / 4) (-11 / 36) ∪ Set.Icc (13 / 36) (5 / 4)) := sorry

end hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l1527_152700


namespace sum_of_products_l1527_152765

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a + b + c = 14) : 
  ab + bc + ac = 72 := 
by 
  sorry

end sum_of_products_l1527_152765


namespace triangle_type_l1527_152774

theorem triangle_type (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 30) 
  (h2 : c = 15) 
  (h3 : b = 5 * Real.sqrt 3) 
  (h4 : a ≠ 0) 
  (h5 : b ≠ 0)
  (h6 : c ≠ 0) 
  (h7 : 0 < A ∧ A < 180) 
  (h8 : 0 < B ∧ B < 180) 
  (h9 : 0 < C ∧ C < 180) 
  (h10 : A + B + C = 180) : 
  (A = 90 ∨ A = C) ∧ A + B + C = 180 :=
by 
  sorry

end triangle_type_l1527_152774


namespace certain_positive_integer_value_l1527_152792

-- Define factorial
def fact : ℕ → ℕ 
| 0     => 1
| (n+1) => (n+1) * fact n

-- Statement of the problem
theorem certain_positive_integer_value (i k m a : ℕ) :
  (fact 8 = 2^i * 3^k * 5^m * 7^a) ∧ (i + k + m + a = 11) → a = 1 :=
by 
  sorry

end certain_positive_integer_value_l1527_152792


namespace natasha_average_speed_l1527_152702

theorem natasha_average_speed :
  ∀ (time_up time_down : ℝ) (speed_up : ℝ),
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (2 * (time_up * speed_up) / (time_up + time_down) = 3) :=
by
  intros time_up time_down speed_up h_time_up h_time_down h_speed_up
  rw [h_time_up, h_time_down, h_speed_up]
  sorry

end natasha_average_speed_l1527_152702


namespace first_tap_fill_time_l1527_152775

theorem first_tap_fill_time (T : ℝ) (h1 : T > 0) (h2 : 12 > 0) 
  (h3 : 1/T - 1/12 = 1/12) : T = 6 :=
sorry

end first_tap_fill_time_l1527_152775


namespace quadrilateral_area_ratio_l1527_152764

noncomputable def area_of_octagon (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

noncomputable def area_of_square (s : ℝ) : ℝ := s^2

theorem quadrilateral_area_ratio (a : ℝ) (s : ℝ)
    (h1 : s = a * Real.sqrt (2 + Real.sqrt 2))
    : (area_of_square s) / (area_of_octagon a) = Real.sqrt 2 / 2 :=
by
  sorry

end quadrilateral_area_ratio_l1527_152764


namespace total_money_made_from_jerseys_l1527_152776

def price_per_jersey : ℕ := 76
def jerseys_sold : ℕ := 2

theorem total_money_made_from_jerseys : price_per_jersey * jerseys_sold = 152 := 
by
  -- The actual proof steps will go here
  sorry

end total_money_made_from_jerseys_l1527_152776


namespace smallest_n_such_that_no_n_digit_is_11_power_l1527_152706

theorem smallest_n_such_that_no_n_digit_is_11_power (log_11 : Real) (h : log_11 = 1.0413) : 
  ∃ n > 1, ∀ k : ℕ, ¬ (10 ^ (n - 1) ≤ 11 ^ k ∧ 11 ^ k < 10 ^ n) :=
sorry

end smallest_n_such_that_no_n_digit_is_11_power_l1527_152706


namespace plastic_bag_estimation_l1527_152729

theorem plastic_bag_estimation (a b c d e f : ℕ) (class_size : ℕ) (h1 : a = 33) 
  (h2 : b = 25) (h3 : c = 28) (h4 : d = 26) (h5 : e = 25) (h6 : f = 31) (h_class_size : class_size = 45) :
  let count := a + b + c + d + e + f
  let average := count / 6
  average * class_size = 1260 := by
{ 
  sorry 
}

end plastic_bag_estimation_l1527_152729


namespace sum_smallest_largest_eq_2y_l1527_152704

theorem sum_smallest_largest_eq_2y (n : ℕ) (y a : ℕ) 
  (h1 : 2 * a + 2 * (n - 1) / n = y) : 
  2 * y = (2 * a + 2 * (n - 1)) := 
sorry

end sum_smallest_largest_eq_2y_l1527_152704


namespace race_participants_minimum_l1527_152712

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end race_participants_minimum_l1527_152712


namespace exist_n_for_all_k_l1527_152789

theorem exist_n_for_all_k (k : ℕ) (h_k : k > 1) : 
  ∃ n : ℕ, 
    (n > 0 ∧ ((n.choose k) % n = 0) ∧ (∀ m : ℕ, (2 ≤ m ∧ m < k) → ((n.choose m) % n ≠ 0))) :=
sorry

end exist_n_for_all_k_l1527_152789


namespace arc_length_of_circle_l1527_152755

section circle_arc_length

def diameter (d : ℝ) : Prop := d = 4
def central_angle_deg (θ_d : ℝ) : Prop := θ_d = 36

theorem arc_length_of_circle
  (d : ℝ) (θ_d : ℝ) (r : ℝ := d / 2) (θ : ℝ := θ_d * (π / 180)) (l : ℝ := θ * r) :
  diameter d → central_angle_deg θ_d → l = 2 * π / 5 :=
by
  intros h1 h2
  sorry

end circle_arc_length

end arc_length_of_circle_l1527_152755


namespace intersect_point_one_l1527_152728

theorem intersect_point_one (k : ℝ) : 
  (∀ y : ℝ, (x = -3 * y^2 - 2 * y + 4 ↔ x = k)) ↔ k = 13 / 3 := 
by
  sorry

end intersect_point_one_l1527_152728


namespace fraction_quaduple_l1527_152784

variable (b a : ℤ)

theorem fraction_quaduple (h₁ : a ≠ 0) : (2 * b) / (a / 2) = 4 * (b / a) :=
by
  sorry

end fraction_quaduple_l1527_152784


namespace david_presents_l1527_152713

variables (C B E : ℕ)

def total_presents (C B E : ℕ) : ℕ := C + B + E

theorem david_presents :
  C = 60 →
  B = 3 * E →
  E = (C / 2) - 10 →
  total_presents C B E = 140 :=
by
  intros hC hB hE
  sorry

end david_presents_l1527_152713


namespace simplify_and_evaluate_l1527_152770

theorem simplify_and_evaluate (a : ℝ) (h₁ : a^2 - 4 * a + 3 = 0) (h₂ : a ≠ 3) : 
  ( (a^2 - 9) / (a^2 - 3 * a) / ( (a^2 + 9) / a + 6 ) = 1 / 4 ) :=
by 
  sorry

end simplify_and_evaluate_l1527_152770


namespace find_middle_number_l1527_152780

theorem find_middle_number (a b c : ℕ) (h1 : a + b = 16) (h2 : a + c = 21) (h3 : b + c = 27) : b = 11 := by
  sorry

end find_middle_number_l1527_152780


namespace total_logs_in_stack_l1527_152799

theorem total_logs_in_stack : 
  ∀ (a_1 a_n : ℕ) (n : ℕ), 
  a_1 = 5 → a_n = 15 → n = a_n - a_1 + 1 → 
  (a_1 + a_n) * n / 2 = 110 :=
by
  intros a_1 a_n n h1 h2 h3
  sorry

end total_logs_in_stack_l1527_152799


namespace minimum_single_discount_l1527_152783

theorem minimum_single_discount (n : ℕ) :
  (∀ x : ℝ, 0 < x → 
    ((1 - n / 100) * x < (1 - 0.18) * (1 - 0.18) * x) ∧
    ((1 - n / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x) ∧
    ((1 - n / 100) * x < (1 - 0.28) * (1 - 0.07) * x))
  ↔ n = 34 :=
by
  sorry

end minimum_single_discount_l1527_152783


namespace sin_75_mul_sin_15_eq_one_fourth_l1527_152753

theorem sin_75_mul_sin_15_eq_one_fourth : 
  Real.sin (75 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_75_mul_sin_15_eq_one_fourth_l1527_152753
