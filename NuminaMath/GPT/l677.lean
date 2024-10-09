import Mathlib

namespace no_solution_l677_67745

theorem no_solution (a : ℝ) :
  (a < -12 ∨ a > 0) →
  ∀ x : ℝ, ¬(6 * (|x - 4 * a|) + (|x - a ^ 2|) + 5 * x - 4 * a = 0) :=
by
  intros ha hx
  sorry

end no_solution_l677_67745


namespace largest_integer_chosen_l677_67762

-- Define the sequence of operations and establish the resulting constraints
def transformed_value (x : ℤ) : ℤ :=
  2 * (4 * x - 30) - 10

theorem largest_integer_chosen : 
  ∃ (x : ℤ), (10 : ℤ) ≤ transformed_value x ∧ transformed_value x ≤ (99 : ℤ) ∧ x = 21 :=
by
  sorry

end largest_integer_chosen_l677_67762


namespace min_value_of_a_plus_b_l677_67734

theorem min_value_of_a_plus_b (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc : 1 = 1) 
    (h1 : b^2 > 4 * a) (h2 : b < 2 * a) (h3 : b < a + 1) : a + b = 10 :=
sorry

end min_value_of_a_plus_b_l677_67734


namespace overall_average_runs_l677_67743

theorem overall_average_runs 
  (test_matches: ℕ) (test_avg: ℕ) 
  (odi_matches: ℕ) (odi_avg: ℕ) 
  (t20_matches: ℕ) (t20_avg: ℕ)
  (h_test_matches: test_matches = 25)
  (h_test_avg: test_avg = 48)
  (h_odi_matches: odi_matches = 20)
  (h_odi_avg: odi_avg = 38)
  (h_t20_matches: t20_matches = 15)
  (h_t20_avg: t20_avg = 28) :
  (25 * 48 + 20 * 38 + 15 * 28) / (25 + 20 + 15) = 39.67 :=
sorry

end overall_average_runs_l677_67743


namespace consecutive_days_probability_l677_67789

noncomputable def probability_of_consecutive_days : ℚ :=
  let total_days := 5
  let combinations := Nat.choose total_days 2
  let consecutive_pairs := 4
  consecutive_pairs / combinations

theorem consecutive_days_probability :
  probability_of_consecutive_days = 2 / 5 :=
by
  sorry

end consecutive_days_probability_l677_67789


namespace total_price_of_bananas_and_oranges_l677_67777

variable (price_orange price_pear price_banana : ℝ)

axiom total_cost_orange_pear : price_orange + price_pear = 120
axiom cost_pear : price_pear = 90
axiom diff_orange_pear_banana : price_orange - price_pear = price_banana

theorem total_price_of_bananas_and_oranges :
  let num_bananas := 200
  let num_oranges := 2 * num_bananas
  let cost_bananas := num_bananas * price_banana
  let cost_oranges := num_oranges * price_orange
  cost_bananas + cost_oranges = 24000 :=
by
  sorry

end total_price_of_bananas_and_oranges_l677_67777


namespace smaller_angle_at_8_15_l677_67791

def angle_minute_hand_at_8_15: ℝ := 90
def angle_hour_hand_at_8: ℝ := 240
def additional_angle_hour_hand_at_8_15: ℝ := 7.5
def total_angle_hour_hand_at_8_15 := angle_hour_hand_at_8 + additional_angle_hour_hand_at_8_15

theorem smaller_angle_at_8_15 :
  min (abs (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))
      (abs (360 - (total_angle_hour_hand_at_8_15 - angle_minute_hand_at_8_15))) = 157.5 :=
by
  sorry

end smaller_angle_at_8_15_l677_67791


namespace sam_has_75_dollars_l677_67763

variable (S B : ℕ)

def condition1 := B = 2 * S - 25
def condition2 := S + B = 200

theorem sam_has_75_dollars (h1 : condition1 S B) (h2 : condition2 S B) : S = 75 := by
  sorry

end sam_has_75_dollars_l677_67763


namespace find_m_of_hyperbola_l677_67758

theorem find_m_of_hyperbola (m : ℝ) (h : mx^2 + y^2 = 1) (s : ∃ x : ℝ, x = 2) : m = -4 := 
by
  sorry

end find_m_of_hyperbola_l677_67758


namespace find_percentage_l677_67759

variable (dollars_1 dollars_2 dollars_total interest_total percentage_unknown : ℝ)
variable (investment_1 investment_rest interest_2 : ℝ)
variable (P : ℝ)

-- Assuming given conditions
axiom H1 : dollars_total = 12000
axiom H2 : dollars_1 = 5500
axiom H3 : interest_total = 970
axiom H4 : investment_rest = dollars_total - dollars_1
axiom H5 : interest_2 = investment_rest * 0.09
axiom H6 : interest_total = dollars_1 * P + interest_2

-- Prove that P = 0.07
theorem find_percentage : P = 0.07 :=
by
  -- Placeholder for the proof that needs to be filled in
  sorry

end find_percentage_l677_67759


namespace minimum_n_for_factorable_polynomial_l677_67754

theorem minimum_n_for_factorable_polynomial :
  ∃ n : ℤ, (∀ A B : ℤ, 5 * A = 48 → 5 * B + A = n) ∧
  (∀ k : ℤ, (∀ A B : ℤ, 5 * A * B = 48 → 5 * B + A = k) → n ≤ k) :=
by
  sorry

end minimum_n_for_factorable_polynomial_l677_67754


namespace slope_angle_l677_67729

theorem slope_angle (A B : ℝ × ℝ) (θ : ℝ) (hA : A = (-1, 3)) (hB : B = (1, 1)) (hθ : θ ∈ Set.Ico 0 Real.pi)
  (hslope : Real.tan θ = (B.2 - A.2) / (B.1 - A.1)) :
  θ = (3 / 4) * Real.pi :=
by
  cases hA
  cases hB
  simp at hslope
  sorry

end slope_angle_l677_67729


namespace linda_age_13_l677_67755

variable (J L : ℕ)

-- Conditions: 
-- 1. Linda is 3 more than 2 times the age of Jane.
-- 2. In five years, the sum of their ages will be 28.
def conditions (J L : ℕ) : Prop :=
  L = 2 * J + 3 ∧ (J + 5) + (L + 5) = 28

-- Question/answer to prove: Linda's current age is 13.
theorem linda_age_13 (J L : ℕ) (h : conditions J L) : L = 13 :=
by
  sorry

end linda_age_13_l677_67755


namespace find_number_divided_by_3_equals_subtracted_5_l677_67710

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l677_67710


namespace first_tray_holds_260_cups_l677_67705

variable (x : ℕ)

def first_tray_holds_x_cups (tray1 : ℕ) := tray1 = x
def second_tray_holds_x_minus_20_cups (tray2 : ℕ) := tray2 = x - 20
def total_cups_in_both_trays (tray1 tray2: ℕ) := tray1 + tray2 = 500

theorem first_tray_holds_260_cups (tray1 tray2 : ℕ) :
  first_tray_holds_x_cups x tray1 →
  second_tray_holds_x_minus_20_cups x tray2 →
  total_cups_in_both_trays tray1 tray2 →
  x = 260 := by
  sorry

end first_tray_holds_260_cups_l677_67705


namespace tan_beta_value_l677_67707

theorem tan_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 4 / 3)
  (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2 / 11 := 
sorry

end tan_beta_value_l677_67707


namespace acd_over_b_eq_neg_210_l677_67703

theorem acd_over_b_eq_neg_210 
  (a b c d x : ℤ) 
  (h1 : x = (a + b*Real.sqrt c)/d) 
  (h2 : (7*x)/8 + 1 = 4/x) 
  : (a * c * d) / b = -210 := 
by 
  sorry

end acd_over_b_eq_neg_210_l677_67703


namespace find_number_l677_67753

theorem find_number (x : ℤ) (h : x - (28 - (37 - (15 - 16))) = 55) : x = 65 :=
sorry

end find_number_l677_67753


namespace average_of_multiples_of_6_l677_67782

def first_n_multiples_sum (n : ℕ) : ℕ :=
  (n * (6 + 6 * n)) / 2

def first_n_multiples_avg (n : ℕ) : ℕ :=
  (first_n_multiples_sum n) / n

theorem average_of_multiples_of_6 (n : ℕ) : first_n_multiples_avg n = 66 → n = 11 := by
  sorry

end average_of_multiples_of_6_l677_67782


namespace gcd_of_given_lcm_and_ratio_l677_67760

theorem gcd_of_given_lcm_and_ratio (C D : ℕ) (h1 : Nat.lcm C D = 200) (h2 : C * 5 = D * 2) : Nat.gcd C D = 5 :=
sorry

end gcd_of_given_lcm_and_ratio_l677_67760


namespace x_squared_minus_y_squared_l677_67716

open Real

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 4/9)
  (h2 : x - y = 2/9) :
  x^2 - y^2 = 8/81 :=
by
  sorry

end x_squared_minus_y_squared_l677_67716


namespace madeline_needs_work_hours_l677_67718

def rent : ℝ := 1200
def groceries : ℝ := 400
def medical_expenses : ℝ := 200
def utilities : ℝ := 60
def emergency_savings : ℝ := 200
def hourly_wage : ℝ := 15

def total_expenses : ℝ := rent + groceries + medical_expenses + utilities + emergency_savings

noncomputable def total_hours_needed : ℝ := total_expenses / hourly_wage

theorem madeline_needs_work_hours :
  ⌈total_hours_needed⌉ = 138 := by
  sorry

end madeline_needs_work_hours_l677_67718


namespace jackson_earned_on_monday_l677_67750

-- Definitions
def goal := 1000
def tuesday_earnings := 40
def avg_rate := 10
def houses := 88
def days_remaining := 3
def total_collected_remaining_days := days_remaining * (houses / 4) * avg_rate

-- The proof problem statement
theorem jackson_earned_on_monday (m : ℕ) :
  m + tuesday_earnings + total_collected_remaining_days = goal → m = 300 :=
by
  -- We will eventually provide the proof here
  sorry

end jackson_earned_on_monday_l677_67750


namespace taxi_distance_l677_67713

variable (initial_fee charge_per_2_5_mile total_charge : ℝ)
variable (d : ℝ)

theorem taxi_distance 
  (h_initial_fee : initial_fee = 2.35)
  (h_charge_per_2_5_mile : charge_per_2_5_mile = 0.35)
  (h_total_charge : total_charge = 5.50)
  (h_eq : total_charge = initial_fee + (charge_per_2_5_mile / (2/5)) * d) :
  d = 3.6 :=
sorry

end taxi_distance_l677_67713


namespace start_time_6am_l677_67711

def travel_same_time (t : ℝ) (x : ℝ) (y : ℝ) (constant_speed : Prop) : Prop :=
  (x = t + 4) ∧ (y = t + 9) ∧ constant_speed 

theorem start_time_6am
  (x y t: ℝ)
  (constant_speed : Prop) 
  (meet_noon : travel_same_time t x y constant_speed)
  (eqn : 1/t + 1/(t + 4) + 1/(t + 9) = 1) :
  t = 6 :=
by
  sorry

end start_time_6am_l677_67711


namespace consecutive_odd_sum_count_l677_67752

theorem consecutive_odd_sum_count (N : ℕ) :
  N = 20 ↔ (
    ∃ (ns : Finset ℕ), ∃ (js : Finset ℕ),
      (∀ n ∈ ns, n < 500) ∧
      (∀ j ∈ js, j ≥ 2) ∧
      ∀ n ∈ ns, ∃ j ∈ js, ∃ k, k = 3 ∧ N = j * (2 * k + j)
  ) :=
by
  sorry

end consecutive_odd_sum_count_l677_67752


namespace lower_limit_total_people_l677_67701

/-- 
  Given:
    1. Exactly 3/7 of the people in the room are under the age of 21.
    2. Exactly 5/10 of the people in the room are over the age of 65.
    3. There are 30 people in the room under the age of 21.
  Prove: The lower limit of the total number of people in the room is 70.
-/
theorem lower_limit_total_people (T : ℕ) (h1 : (3 / 7) * T = 30) : T = 70 := by
  sorry

end lower_limit_total_people_l677_67701


namespace Ann_age_is_39_l677_67778

def current_ages (A B : ℕ) : Prop :=
  A + B = 52 ∧ (B = 2 * B - A / 3) ∧ (A = 3 * B)

theorem Ann_age_is_39 : ∃ A B : ℕ, current_ages A B ∧ A = 39 :=
by
  sorry

end Ann_age_is_39_l677_67778


namespace vertical_distance_to_Felix_l677_67717

/--
  Dora is at point (8, -15).
  Eli is at point (2, 18).
  Felix is at point (5, 7).
  Calculate the vertical distance they need to walk to reach Felix.
-/
theorem vertical_distance_to_Felix :
  let Dora := (8, -15)
  let Eli := (2, 18)
  let Felix := (5, 7)
  let midpoint := ((Dora.1 + Eli.1) / 2, (Dora.2 + Eli.2) / 2)
  let vertical_distance := Felix.2 - midpoint.2
  vertical_distance = 5.5 :=
by
  sorry

end vertical_distance_to_Felix_l677_67717


namespace probability_is_correct_l677_67741

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l677_67741


namespace rita_bought_four_pounds_l677_67788

def initial_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def left_amount : ℝ := 35.68

theorem rita_bought_four_pounds :
  (initial_amount - left_amount) / cost_per_pound = 4 :=
by
  sorry

end rita_bought_four_pounds_l677_67788


namespace range_of_m_l677_67720

-- Definition of p: x / (x - 2) < 0 implies 0 < x < 2
def p (x : ℝ) : Prop := x / (x - 2) < 0

-- Definition of q: 0 < x < m
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Main theorem: If p is a necessary but not sufficient condition for q to hold, then the range of m is (2, +∞)
theorem range_of_m {m : ℝ} (h : ∀ x, p x → q x m) (hs : ∃ x, ¬(q x m) ∧ p x) : 
  2 < m :=
sorry

end range_of_m_l677_67720


namespace maria_total_money_l677_67780

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l677_67780


namespace find_correction_time_l677_67732

-- Define the conditions
def loses_minutes_per_day : ℚ := 2 + 1/2
def initial_time_set : ℚ := 1 * 60 -- 1 PM in minutes
def time_on_march_21 : ℚ := 9 * 60 -- 9 AM in minutes on March 21
def total_minutes_per_day : ℚ := 24 * 60
def days_between : ℚ := 6 - 4/24 -- 6 days minus 4 hours

-- Calculate effective functioning minutes per day
def effective_minutes_per_day : ℚ := total_minutes_per_day - loses_minutes_per_day

-- Calculate the ratio of actual time to the watch's time
def time_ratio : ℚ := total_minutes_per_day / effective_minutes_per_day

-- Calculate the total actual time in minutes between initial set time and the given time showing on the watch
def total_actual_time : ℚ := days_between * total_minutes_per_day + initial_time_set

-- Calculate the actual time according to the ratio
def actual_time_according_to_ratio : ℚ := total_actual_time * time_ratio

-- Calculate the correction required 'n'
def required_minutes_correction : ℚ := actual_time_according_to_ratio - total_actual_time

-- The theorem stating that the required correction is as calculated
theorem find_correction_time : required_minutes_correction = (14 + 14/23) := by
  sorry

end find_correction_time_l677_67732


namespace pipe_length_l677_67771

theorem pipe_length (L_short : ℕ) (hL_short : L_short = 59) : 
    L_short + 2 * L_short = 177 := by
  sorry

end pipe_length_l677_67771


namespace distance_between_centers_of_circles_l677_67700

theorem distance_between_centers_of_circles :
  ∀ (rect_width rect_height circle_radius distance_between_centers : ℝ),
  rect_width = 11 
  ∧ rect_height = 7 
  ∧ circle_radius = rect_height / 2 
  ∧ distance_between_centers = rect_width - 2 * circle_radius 
  → distance_between_centers = 4 := by
  intros rect_width rect_height circle_radius distance_between_centers
  sorry

end distance_between_centers_of_circles_l677_67700


namespace smallest_k_l677_67793

theorem smallest_k :
  ∃ k : ℕ, k > 1 ∧ (k % 19 = 1) ∧ (k % 7 = 1) ∧ (k % 3 = 1) ∧ k = 400 :=
by
  sorry

end smallest_k_l677_67793


namespace sufficient_not_necessary_l677_67739

variable (x : ℝ)

theorem sufficient_not_necessary (h : x^2 - 3 * x + 2 > 0) : x > 2 → (∀ x : ℝ, x^2 - 3 * x + 2 > 0 ↔ x > 2 ∨ x < -1) :=
by
  sorry

end sufficient_not_necessary_l677_67739


namespace graph_not_in_first_quadrant_l677_67798

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

-- Prove that the graph of f(x) does not pass through the first quadrant
theorem graph_not_in_first_quadrant : ∀ (x : ℝ), x > 0 → f x ≤ 0 := by
  intro x hx
  sorry

end graph_not_in_first_quadrant_l677_67798


namespace largest_number_of_four_consecutive_whole_numbers_l677_67797

theorem largest_number_of_four_consecutive_whole_numbers 
  (a : ℕ) (h1 : a + (a + 1) + (a + 2) = 184)
  (h2 : a + (a + 1) + (a + 3) = 201)
  (h3 : a + (a + 2) + (a + 3) = 212)
  (h4 : (a + 1) + (a + 2) + (a + 3) = 226) : 
  a + 3 = 70 := 
by sorry

end largest_number_of_four_consecutive_whole_numbers_l677_67797


namespace min_value_condition_solve_inequality_l677_67712

open Real

-- Define the function f(x) = |x - a| + |x + 2|
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 2)

-- Part I: Proving the values of a for f(x) having minimum value of 2
theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → (∃ x : ℝ, f x a = 2) → (a = 0 ∨ a = -4) :=
by
  sorry

-- Part II: Solving inequality f(x) ≤ 6 when a = 2
theorem solve_inequality : 
  ∀ x : ℝ, f x 2 ≤ 6 ↔ (x ≥ -3 ∧ x ≤ 3) :=
by
  sorry

end min_value_condition_solve_inequality_l677_67712


namespace biker_bob_east_distance_l677_67765

noncomputable def distance_between_towns : ℝ := 28.30194339616981
noncomputable def distance_west : ℝ := 30
noncomputable def distance_north_1 : ℝ := 6
noncomputable def distance_north_2 : ℝ := 18
noncomputable def total_distance_north : ℝ := distance_north_1 + distance_north_2
noncomputable def unknown_distance_east : ℝ := 45.0317 -- Expected distance east

theorem biker_bob_east_distance :
  ∃ (E : ℝ), (total_distance_north ^ 2 + (-distance_west + E) ^ 2 = distance_between_towns ^ 2) ∧ E = unknown_distance_east :=
by 
  sorry

end biker_bob_east_distance_l677_67765


namespace value_of_fraction_power_series_l677_67770

theorem value_of_fraction_power_series (x : ℕ) (h : x = 3) :
  (x^3 * x^5 * x^7 * x^9 * x^11 * x^13 * x^15 * x^17 * x^19 * x^21) /
  (x^4 * x^8 * x^12 * x^16 * x^20 * x^24) = 3^36 :=
by
  subst h
  sorry

end value_of_fraction_power_series_l677_67770


namespace triangle_angle_contradiction_l677_67776

theorem triangle_angle_contradiction (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A > 60) (h₃ : B > 60) (h₄ : C > 60) :
  false :=
by
  sorry

end triangle_angle_contradiction_l677_67776


namespace andy_last_problem_l677_67726

theorem andy_last_problem (start_num : ℕ) (num_solved : ℕ) (result : ℕ) : 
  start_num = 78 → 
  num_solved = 48 → 
  result = start_num + num_solved - 1 → 
  result = 125 :=
by
  sorry

end andy_last_problem_l677_67726


namespace product_of_roots_l677_67768

theorem product_of_roots :
  let a := 24
  let c := -216
  ∀ x : ℝ, (24 * x^2 + 36 * x - 216 = 0) → (c / a = -9) :=
by
  intros
  sorry

end product_of_roots_l677_67768


namespace term_free_of_x_l677_67747

namespace PolynomialExpansion

theorem term_free_of_x (m n k : ℕ) (h : (x : ℝ)^(m * k - (m + n) * r) = 1) :
  (m * k) % (m + n) = 0 :=
by
  sorry

end PolynomialExpansion

end term_free_of_x_l677_67747


namespace simple_interest_rate_l677_67769

theorem simple_interest_rate (P R T A : ℝ) (h_double: A = 2 * P) (h_si: A = P + P * R * T / 100) (h_T: T = 5) : R = 20 :=
by
  have h1: A = 2 * P := h_double
  have h2: A = P + P * R * T / 100 := h_si
  have h3: T = 5 := h_T
  sorry

end simple_interest_rate_l677_67769


namespace geometric_sequence_product_geometric_sequence_sum_not_definitely_l677_67708

theorem geometric_sequence_product (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ∃ r3, ∀ n, (a n * b n) = r3 * (a (n-1) * b (n-1)) :=
sorry

theorem geometric_sequence_sum_not_definitely (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ¬ ∀ r3, ∃ N, ∀ n ≥ N, (a n + b n) = r3 * (a (n-1) + b (n-1)) :=
sorry

end geometric_sequence_product_geometric_sequence_sum_not_definitely_l677_67708


namespace binom_12_9_plus_binom_12_3_l677_67749

theorem binom_12_9_plus_binom_12_3 : (Nat.choose 12 9) + (Nat.choose 12 3) = 440 := by
  sorry

end binom_12_9_plus_binom_12_3_l677_67749


namespace smallest_b_gt_4_perfect_square_l677_67722

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l677_67722


namespace odd_function_increasing_function_l677_67728

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function (x : ℝ) : 
  (f (1 / 2) (-x)) = -(f (1 / 2) x) := 
by
  sorry

theorem increasing_function : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f (1 / 2) x₁ < f (1 / 2) x₂ := 
by
  sorry

end odd_function_increasing_function_l677_67728


namespace three_digit_number_l677_67766

theorem three_digit_number (a b c : ℕ) (h1 : a * (b + c) = 33) (h2 : b * (a + c) = 40) : 
  100 * a + 10 * b + c = 347 :=
by
  sorry

end three_digit_number_l677_67766


namespace intersection_is_expected_result_l677_67785

def set_A : Set ℝ := { x | x * (x + 1) > 0 }
def set_B : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 1) }
def expected_result : Set ℝ := { x | x ≥ 1 }

theorem intersection_is_expected_result : set_A ∩ set_B = expected_result := by
  sorry

end intersection_is_expected_result_l677_67785


namespace divisible_check_l677_67738

theorem divisible_check (n : ℕ) (h : n = 287) : 
  ¬ (n % 3 = 0) ∧  ¬ (n % 4 = 0) ∧  ¬ (n % 5 = 0) ∧ ¬ (n % 6 = 0) ∧ (n % 7 = 0) := 
by {
  sorry
}

end divisible_check_l677_67738


namespace no_such_function_exists_l677_67723

namespace ProofProblem

open Nat

-- Declaration of the proposed function
def f : ℕ+ → ℕ+ := sorry

-- Statement to be proved
theorem no_such_function_exists : 
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f^[n] n = n + 1 :=
by
  sorry

end ProofProblem

end no_such_function_exists_l677_67723


namespace minimum_value_of_quadratic_expression_l677_67779

theorem minimum_value_of_quadratic_expression (x y z : ℝ)
  (h : x + y + z = 2) : 
  x^2 + 2 * y^2 + z^2 ≥ 4 / 3 :=
sorry

end minimum_value_of_quadratic_expression_l677_67779


namespace correct_negation_statement_l677_67775

def Person : Type := sorry

def is_adult (p : Person) : Prop := sorry
def is_teenager (p : Person) : Prop := sorry
def is_responsible (p : Person) : Prop := sorry
def is_irresponsible (p : Person) : Prop := sorry

axiom all_adults_responsible : ∀ p, is_adult p → is_responsible p
axiom some_adults_responsible : ∃ p, is_adult p ∧ is_responsible p
axiom no_teenagers_responsible : ∀ p, is_teenager p → ¬is_responsible p
axiom all_teenagers_irresponsible : ∀ p, is_teenager p → is_irresponsible p
axiom exists_irresponsible_teenager : ∃ p, is_teenager p ∧ is_irresponsible p
axiom all_teenagers_responsible : ∀ p, is_teenager p → is_responsible p

theorem correct_negation_statement
: (∃ p, is_teenager p ∧ ¬is_responsible p) ↔ 
  (∃ p, is_teenager p ∧ is_irresponsible p) :=
sorry

end correct_negation_statement_l677_67775


namespace geom_seq_prop_l677_67795

variable (b : ℕ → ℝ) (r : ℝ) (s t : ℕ)
variable (h : s ≠ t)
variable (h1 : s > 0) (h2 : t > 0)
variable (h3 : b 1 = 1)
variable (h4 : ∀ n, b (n + 1) = b n * r)

theorem geom_seq_prop : s ≠ t → s > 0 → t > 0 → b 1 = 1 → (∀ n, b (n + 1) = b n * r) → (b t)^(s - 1) / (b s)^(t - 1) = 1 :=
by
  intros h h1 h2 h3 h4
  sorry

end geom_seq_prop_l677_67795


namespace find_15th_term_l677_67733

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end find_15th_term_l677_67733


namespace length_of_second_platform_is_correct_l677_67773

-- Define the constants
def lt : ℕ := 70  -- Length of the train
def l1 : ℕ := 170  -- Length of the first platform
def t1 : ℕ := 15  -- Time to cross the first platform
def t2 : ℕ := 20  -- Time to cross the second platform

-- Calculate the speed of the train
def v : ℕ := (lt + l1) / t1

-- Define the length of the second platform
def l2 : ℕ := 250

-- The proof statement
theorem length_of_second_platform_is_correct : lt + l2 = v * t2 := sorry

end length_of_second_platform_is_correct_l677_67773


namespace slope_of_given_line_l677_67796

def slope_of_line (l : String) : Real :=
  -- Assuming that we have a function to parse the line equation
  -- and extract its slope. Normally, this would be a complex parsing function.
  1 -- Placeholder, as the slope calculation logic is trivial here.

theorem slope_of_given_line : slope_of_line "x - y - 1 = 0" = 1 := by
  sorry

end slope_of_given_line_l677_67796


namespace cara_bread_dinner_amount_240_l677_67709

def conditions (B L D : ℕ) : Prop :=
  8 * L = D ∧ 6 * B = D ∧ B + L + D = 310

theorem cara_bread_dinner_amount_240 :
  ∃ (B L D : ℕ), conditions B L D ∧ D = 240 :=
by
  sorry

end cara_bread_dinner_amount_240_l677_67709


namespace force_is_correct_l677_67702

noncomputable def force_computation : ℝ :=
  let m : ℝ := 5 -- kg
  let s : ℝ → ℝ := fun t => 2 * t + 3 * t^2 -- cm
  let a : ℝ := 6 / 100 -- acceleration in m/s^2
  m * a

theorem force_is_correct : force_computation = 0.3 := 
by
  -- Initial conditions
  sorry

end force_is_correct_l677_67702


namespace opponent_choice_is_random_l677_67774

-- Define the possible outcomes in the game
inductive Outcome
| rock
| paper
| scissors

-- Defining the opponent's choice set
def opponent_choice := {outcome : Outcome | outcome = Outcome.rock ∨ outcome = Outcome.paper ∨ outcome = Outcome.scissors}

-- The event where the opponent chooses "scissors"
def event_opponent_chooses_scissors := Outcome.scissors ∈ opponent_choice

-- Proving that the event of opponent choosing "scissors" is a random event
theorem opponent_choice_is_random : ¬(∀outcome ∈ opponent_choice, outcome = Outcome.scissors) ∧ (∃ outcome ∈ opponent_choice, outcome = Outcome.scissors) → event_opponent_chooses_scissors := 
sorry

end opponent_choice_is_random_l677_67774


namespace one_cow_one_bag_in_forty_days_l677_67731

theorem one_cow_one_bag_in_forty_days
    (total_cows : ℕ)
    (total_bags : ℕ)
    (total_days : ℕ)
    (husk_consumption : total_cows * total_bags = total_cows * total_days) :
  total_days = 40 :=
by sorry

end one_cow_one_bag_in_forty_days_l677_67731


namespace find_angleZ_l677_67751

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end find_angleZ_l677_67751


namespace area_of_circles_l677_67724

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l677_67724


namespace elizabeth_stickers_count_l677_67719

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l677_67719


namespace sum_of_variables_l677_67781

theorem sum_of_variables (x y z w : ℤ) 
(h1 : x - y + z = 7) 
(h2 : y - z + w = 8) 
(h3 : z - w + x = 4) 
(h4 : w - x + y = 3) : 
x + y + z + w = 11 := 
sorry

end sum_of_variables_l677_67781


namespace number_of_blue_parrots_l677_67786

-- Defining the known conditions
def total_parrots : ℕ := 120
def fraction_red : ℚ := 2 / 3
def fraction_green : ℚ := 1 / 6

-- Proving the number of blue parrots given the conditions
theorem number_of_blue_parrots : (1 - (fraction_red + fraction_green)) * total_parrots = 20 := by
  sorry

end number_of_blue_parrots_l677_67786


namespace fraction_proof_l677_67736

theorem fraction_proof (x y : ℕ) (h1 : y = 7) (h2 : x = 22) : 
  (y / (x - 1) = 1 / 3) ∧ ((y + 4) / x = 1 / 2) := by
  sorry

end fraction_proof_l677_67736


namespace independence_test_purpose_l677_67727

theorem independence_test_purpose:
  ∀ (test: String), test = "independence test" → 
  ∀ (purpose: String), purpose = "to provide the reliability of the relationship between two categorical variables" →
  (test = "independence test" ∧ purpose = "to provide the reliability of the relationship between two categorical variables") :=
by
  intros test h_test purpose h_purpose
  exact ⟨h_test, h_purpose⟩

end independence_test_purpose_l677_67727


namespace quadratic_has_two_distinct_real_roots_l677_67792

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  (∃ a b c : ℝ, a = k - 2 ∧ b = -2 ∧ c = 1 / 2 ∧ a ≠ 0 ∧ b ^ 2 - 4 * a * c > 0) ↔ (k < 4 ∧ k ≠ 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l677_67792


namespace MattSkipsRopesTimesPerSecond_l677_67740

theorem MattSkipsRopesTimesPerSecond:
  ∀ (minutes_jumped : ℕ) (total_skips : ℕ), 
  minutes_jumped = 10 → 
  total_skips = 1800 → 
  (total_skips / (minutes_jumped * 60)) = 3 :=
by
  intros minutes_jumped total_skips h_jumped h_skips
  sorry

end MattSkipsRopesTimesPerSecond_l677_67740


namespace total_balloons_l677_67706

def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  sorry

end total_balloons_l677_67706


namespace min_value_four_l677_67742

noncomputable def min_value_T (a b c : ℝ) : ℝ :=
  1 / (2 * (a * b - 1)) + a * (b + 2 * c) / (a * b - 1)

theorem min_value_four (a b c : ℝ) (h1 : (1 / a) > 0)
  (h2 : b^2 - (4 * c) / a ≤ 0) (h3 : a * b > 1) : 
  min_value_T a b c = 4 := 
by 
  sorry

end min_value_four_l677_67742


namespace de_morgan_neg_or_l677_67737

theorem de_morgan_neg_or (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by sorry

end de_morgan_neg_or_l677_67737


namespace rooms_count_l677_67784

theorem rooms_count (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) : (total_paintings / paintings_per_room) = 4 := by
  sorry

end rooms_count_l677_67784


namespace directrix_of_parabola_l677_67767

theorem directrix_of_parabola :
  ∀ (x : ℝ), (∃ k : ℝ, y = (x^2 - 8 * x + 16) / 8 → k = -2) :=
by
  sorry

end directrix_of_parabola_l677_67767


namespace count_negative_numbers_l677_67787

theorem count_negative_numbers : 
  (List.filter (λ x => x < (0:ℚ)) [-14, 7, 0, -2/3, -5/16]).length = 3 := 
by
  sorry

end count_negative_numbers_l677_67787


namespace proposition_1_proposition_3_l677_67790

variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Condition predicates
def parallel (p q : Plane) : Prop := sorry -- parallelism of p and q
def perpendicular (p q : Plane) : Prop := sorry -- perpendicularly of p and q
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- parallelism of line and plane
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry -- perpendicularity of line and plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- line is in the plane

-- Proposition ①
theorem proposition_1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ := sorry

-- Proposition ③
theorem proposition_3 (h1 : line_perpendicular_plane m α) (h2 : line_parallel_plane m β) : perpendicular α β := sorry

end proposition_1_proposition_3_l677_67790


namespace g_neg_one_add_g_one_l677_67761

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x - y) = f x * g y - f y * g x
axiom f_one_ne_zero : f 1 ≠ 0
axiom f_one_eq_f_two : f 1 = f 2

theorem g_neg_one_add_g_one : g (-1) + g 1 = 1 := by
  sorry

end g_neg_one_add_g_one_l677_67761


namespace problem1_problem2_problem3_l677_67748

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end problem1_problem2_problem3_l677_67748


namespace number_of_students_in_line_l677_67794

-- Definitions for the conditions
def yoojung_last (n : ℕ) : Prop :=
  n = 14

def eunjung_position : ℕ := 5

def students_between (n : ℕ) : Prop :=
  n = 8

noncomputable def total_students : ℕ := 14

-- The theorem to be proven
theorem number_of_students_in_line 
  (last : yoojung_last total_students) 
  (eunjung_pos : eunjung_position = 5) 
  (between : students_between 8) :
  total_students = 14 := by
  sorry

end number_of_students_in_line_l677_67794


namespace mary_gave_becky_green_crayons_l677_67783

-- Define the initial conditions
def initial_green_crayons : Nat := 5
def initial_blue_crayons : Nat := 8
def given_blue_crayons : Nat := 1
def remaining_crayons : Nat := 9

-- Define the total number of crayons initially
def total_initial_crayons : Nat := initial_green_crayons + initial_blue_crayons

-- Define the number of crayons given away
def given_crayons : Nat := total_initial_crayons - remaining_crayons

-- The crux of the problem
def given_green_crayons : Nat :=
  given_crayons - given_blue_crayons

-- Formal statement of the theorem
theorem mary_gave_becky_green_crayons
  (h_initial_green : initial_green_crayons = 5)
  (h_initial_blue : initial_blue_crayons = 8)
  (h_given_blue : given_blue_crayons = 1)
  (h_remaining : remaining_crayons = 9) :
  given_green_crayons = 3 :=
by {
  -- This should be the body of the proof, but we'll skip it for now
  sorry
}

end mary_gave_becky_green_crayons_l677_67783


namespace product_of_positive_c_for_rational_solutions_l677_67721

theorem product_of_positive_c_for_rational_solutions : 
  (∃ c₁ c₂ : ℕ, c₁ > 0 ∧ c₂ > 0 ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₁ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₂ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   c₁ * c₂ = 8) :=
sorry

end product_of_positive_c_for_rational_solutions_l677_67721


namespace balls_in_boxes_l677_67704

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l677_67704


namespace frac_sum_property_l677_67744

theorem frac_sum_property (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end frac_sum_property_l677_67744


namespace expression_equals_five_l677_67715

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l677_67715


namespace number_of_friends_shared_with_l677_67757

-- Conditions and given data
def doughnuts_samuel : ℕ := 2 * 12
def doughnuts_cathy : ℕ := 3 * 12
def total_doughnuts : ℕ := doughnuts_samuel + doughnuts_cathy
def each_person_doughnuts : ℕ := 6
def total_people := total_doughnuts / each_person_doughnuts
def samuel_and_cathy : ℕ := 2

-- Statement to prove - Number of friends they shared with
theorem number_of_friends_shared_with : (total_people - samuel_and_cathy) = 8 := by
  sorry

end number_of_friends_shared_with_l677_67757


namespace fair_people_ratio_l677_67772

def next_year_ratio (this_year next_year last_year : ℕ) (total : ℕ) :=
  this_year = 600 ∧
  last_year = next_year - 200 ∧
  this_year + last_year + next_year = total → 
  next_year = 2 * this_year

theorem fair_people_ratio :
  ∀ (next_year : ℕ),
  next_year_ratio 600 next_year (next_year - 200) 2800 → next_year = 2 * 600 := by
sorry

end fair_people_ratio_l677_67772


namespace player_reach_wingspan_l677_67756

theorem player_reach_wingspan :
  ∀ (rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan : ℕ),
  rim_height = 120 →
  player_height = 72 →
  jump_height = 32 →
  reach_above_rim = 6 →
  reach_with_jump = player_height + jump_height →
  reach_wingspan = (rim_height + reach_above_rim) - reach_with_jump →
  reach_wingspan = 22 :=
by
  intros rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan
  intros h_rim_height h_player_height h_jump_height h_reach_above_rim h_reach_with_jump h_reach_wingspan
  rw [h_rim_height, h_player_height, h_jump_height, h_reach_above_rim] at *
  simp at *
  sorry

end player_reach_wingspan_l677_67756


namespace symmetric_scanning_codes_count_l677_67730

noncomputable def countSymmetricScanningCodes : ℕ :=
  let totalConfigs := 32
  let invalidConfigs := 2
  totalConfigs - invalidConfigs

theorem symmetric_scanning_codes_count :
  countSymmetricScanningCodes = 30 :=
by
  -- Here, we would detail the steps, but we omit the actual proof for now.
  sorry

end symmetric_scanning_codes_count_l677_67730


namespace boat_distance_travelled_upstream_l677_67714

theorem boat_distance_travelled_upstream (v : ℝ) (d : ℝ) :
  ∀ (boat_speed_in_still_water upstream_time downstream_time : ℝ),
  boat_speed_in_still_water = 25 →
  upstream_time = 1 →
  downstream_time = 0.25 →
  d = (boat_speed_in_still_water - v) * upstream_time →
  d = (boat_speed_in_still_water + v) * downstream_time →
  d = 10 :=
by
  intros
  sorry

end boat_distance_travelled_upstream_l677_67714


namespace Sandy_original_number_l677_67746

theorem Sandy_original_number (x : ℝ) (h : (3 * x + 20)^2 = 2500) : x = 10 :=
by
  sorry

end Sandy_original_number_l677_67746


namespace find_second_number_l677_67799

theorem find_second_number (x : ℝ) : 217 + x + 0.217 + 2.0017 = 221.2357 → x = 2.017 :=
by
  sorry

end find_second_number_l677_67799


namespace suitable_for_census_l677_67735

-- Define types for each survey option.
inductive SurveyOption where
  | A : SurveyOption -- Understanding the vision of middle school students in our province
  | B : SurveyOption -- Investigating the viewership of "The Reader"
  | C : SurveyOption -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
  | D : SurveyOption -- Testing the lifespan of a batch of light bulbs

-- Theorem statement asserting that Option C is the suitable one for a census.
theorem suitable_for_census : SurveyOption.C = SurveyOption.C :=
by
  exact rfl

end suitable_for_census_l677_67735


namespace solve_for_a_l677_67764

theorem solve_for_a (a : ℚ) (h : 2 * a - 3 = 5 - a) : a = 8 / 3 :=
by
  sorry

end solve_for_a_l677_67764


namespace compare_abc_l677_67725

open Real

theorem compare_abc
  (a b c : ℝ)
  (ha : 0 < a ∧ a < π / 2)
  (hb : 0 < b ∧ b < π / 2)
  (hc : 0 < c ∧ c < π / 2)
  (h1 : cos a = a)
  (h2 : sin (cos b) = b)
  (h3 : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end compare_abc_l677_67725
