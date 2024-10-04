import Mathlib

namespace evaluate_expression_l90_90156

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 4 = 6564 :=
by
  sorry

end evaluate_expression_l90_90156


namespace remaining_statue_weight_l90_90181

theorem remaining_statue_weight (w_initial w1 w2 w_discarded w_remaining : ℕ) 
    (h_initial : w_initial = 80)
    (h_w1 : w1 = 10)
    (h_w2 : w2 = 18)
    (h_discarded : w_discarded = 22) :
    2 * w_remaining = w_initial - w_discarded - w1 - w2 :=
by
  sorry

end remaining_statue_weight_l90_90181


namespace original_number_of_professors_l90_90026

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l90_90026


namespace arithmetic_sequence_tenth_term_l90_90883

theorem arithmetic_sequence_tenth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 6 * d = 13) :
  a + 9 * d = 19 := 
sorry

end arithmetic_sequence_tenth_term_l90_90883


namespace ariel_years_fencing_l90_90622

-- Definitions based on given conditions
def fencing_start_year := 2006
def birth_year := 1992
def current_age := 30

-- To find: The number of years Ariel has been fencing
def current_year : ℕ := birth_year + current_age
def years_fencing : ℕ := current_year - fencing_start_year

-- Proof statement
theorem ariel_years_fencing : years_fencing = 16 := by
  sorry

end ariel_years_fencing_l90_90622


namespace quadratic_roots_l90_90238

theorem quadratic_roots (x : ℝ) : (x^2 + 4*x + 3 = 0) ↔ (x = -3 ∨ x = -1) := 
sorry

end quadratic_roots_l90_90238


namespace positive_difference_l90_90236

theorem positive_difference (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 3 * y - 4 * x = 9) : 
  abs (y - x) = 129 / 7 - (30 - 129 / 7) := 
by {
  sorry
}

end positive_difference_l90_90236


namespace max_handshakes_l90_90604

theorem max_handshakes (n m : ℕ) (cond1 : n = 30) (cond2 : m = 5) 
                       (cond3 : ∀ (i : ℕ), i < 30 → ∀ (j : ℕ), j < 30 → i ≠ j → true)
                       (cond4 : ∀ (k : ℕ), k < 5 → ∃ (s : ℕ), s ≤ 10) : 
  ∃ (handshakes : ℕ), handshakes = 325 :=
by
  sorry

end max_handshakes_l90_90604


namespace average_visitors_on_other_days_l90_90095

-- Definitions based on the conditions
def average_visitors_on_sundays  := 510
def average_visitors_per_day     := 285
def total_days_in_month := 30
def non_sunday_days_in_month := total_days_in_month - 5

-- Statement to be proven
theorem average_visitors_on_other_days :
  let total_visitors_for_month := average_visitors_per_day * total_days_in_month in
  let total_visitors_on_sundays := average_visitors_on_sundays * 5 in
  let total_visitors_on_other_days := total_visitors_for_month - total_visitors_on_sundays in
  let average_visitors_on_other_days := total_visitors_on_other_days / non_sunday_days_in_month in
  average_visitors_on_other_days = 240 :=
sorry

end average_visitors_on_other_days_l90_90095


namespace yellow_candles_count_l90_90142

def CalebCandles (grandfather_age : ℕ) (red_candles : ℕ) (blue_candles : ℕ) : ℕ :=
    grandfather_age - (red_candles + blue_candles)

theorem yellow_candles_count :
    CalebCandles 79 14 38 = 27 := by
    sorry

end yellow_candles_count_l90_90142


namespace product_of_place_values_l90_90742

theorem product_of_place_values : 
  let place_value_1 := 800000
  let place_value_2 := 80
  let place_value_3 := 0.08
  place_value_1 * place_value_2 * place_value_3 = 5120000 := 
by 
  -- proof will be provided here 
  sorry

end product_of_place_values_l90_90742


namespace future_value_proof_l90_90728

noncomputable def present_value : ℝ := 1093.75
noncomputable def interest_rate : ℝ := 0.04
noncomputable def years : ℕ := 2

def future_value (PV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PV * (1 + r) ^ n

theorem future_value_proof :
  future_value present_value interest_rate years = 1183.06 :=
by
  -- Calculation details skipped here, assuming the required proof steps are completed.
  sorry

end future_value_proof_l90_90728


namespace probability_of_snow_l90_90572

theorem probability_of_snow :
  let p_snow_each_day : ℚ := 3 / 4
  let p_no_snow_each_day : ℚ := 1 - p_snow_each_day
  let p_no_snow_four_days : ℚ := p_no_snow_each_day ^ 4
  let p_snow_at_least_once_four_days : ℚ := 1 - p_no_snow_four_days
  p_snow_at_least_once_four_days = 255 / 256 :=
by 
  unfold p_snow_each_day
  unfold p_no_snow_each_day
  unfold p_no_snow_four_days
  unfold p_snow_at_least_once_four_days
  unfold p_snow_at_least_once_four_days
  -- Sorry is used to skip the proof
  sorry

end probability_of_snow_l90_90572


namespace cube_volume_split_l90_90660

theorem cube_volume_split (x y z : ℝ) (h : x > 0) :
  ∃ y z : ℝ, y > 0 ∧ z > 0 ∧ y^3 + z^3 = x^3 :=
sorry

end cube_volume_split_l90_90660


namespace M_gt_N_l90_90169

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2*x + 2*y - 2

theorem M_gt_N : M x y > N x y :=
by
  sorry

end M_gt_N_l90_90169


namespace convex_polygon_triangles_impossible_l90_90375

theorem convex_polygon_triangles_impossible :
  ∀ (a b c : ℕ), 2016 + 2 * b + c - 2014 = 0 → a + b + c = 2014 → a = 1007 → false :=
sorry

end convex_polygon_triangles_impossible_l90_90375


namespace intersection_point_exists_l90_90817

theorem intersection_point_exists :
  ∃ (x y : ℝ), (2 * x + y - 5 = 0) ∧ (y = 2 * x^2 + 1) ∧ (-1 ≤ x ∧ x ≤ 1) ∧ (x = 1) ∧ (y = 3) :=
by
  use 1, 3
  split; norm_num
  split
  · exact -1 ≤ 1 ∧ 1 ≤ 1
  · norm_num
  · norm_num
  sorry

end intersection_point_exists_l90_90817


namespace min_ab_correct_l90_90499

noncomputable def min_ab (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) : ℝ :=
  (6 - 2 * Real.sqrt 3) / 3

theorem min_ab_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + bc + ac = 2) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) :
  a + b ≥ min_ab a b c h1 h2 :=
sorry

end min_ab_correct_l90_90499


namespace fraction_power_equality_l90_90326

theorem fraction_power_equality :
  (72000 ^ 4) / (24000 ^ 4) = 81 := 
by
  sorry

end fraction_power_equality_l90_90326


namespace david_boxes_l90_90938

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end david_boxes_l90_90938


namespace original_number_conditions_l90_90763

theorem original_number_conditions (a : ℕ) :
  ∃ (y1 y2 : ℕ), (7 * a = 10 * 9 + y1) ∧ (9 * 9 = 10 * 8 + y2) ∧ y2 = 1 ∧ (a = 13 ∨ a = 14) := sorry

end original_number_conditions_l90_90763


namespace probability_at_least_one_head_l90_90221

theorem probability_at_least_one_head (n : ℕ) (hn : n = 5) (p_tails : ℚ) (h_p : p_tails = 1 / 2) :
    (1 - (p_tails ^ n)) = 31 / 32 :=
by
  sorry

end probability_at_least_one_head_l90_90221


namespace second_daily_rate_l90_90554

noncomputable def daily_rate_sunshine : ℝ := 17.99
noncomputable def mileage_cost_sunshine : ℝ := 0.18
noncomputable def mileage_cost_second : ℝ := 0.16
noncomputable def distance : ℝ := 48.0

theorem second_daily_rate (daily_rate_second : ℝ) : 
  daily_rate_sunshine + (mileage_cost_sunshine * distance) = 
  daily_rate_second + (mileage_cost_second * distance) → 
  daily_rate_second = 18.95 :=
by 
  sorry

end second_daily_rate_l90_90554


namespace goal_l90_90127

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l90_90127


namespace time_after_9999_seconds_l90_90529

theorem time_after_9999_seconds:
  let initial_hours := 5
  let initial_minutes := 45
  let initial_seconds := 0
  let added_seconds := 9999
  let total_seconds := initial_seconds + added_seconds
  let total_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let final_hours := (initial_hours + total_hours + (initial_minutes + remaining_minutes) / 60) % 24
  let final_minutes := (initial_minutes + remaining_minutes) % 60
  initial_hours = 5 →
  initial_minutes = 45 →
  initial_seconds = 0 →
  added_seconds = 9999 →
  final_hours = 8 ∧ final_minutes = 31 ∧ remaining_seconds = 39 :=
by
  intros
  sorry

end time_after_9999_seconds_l90_90529


namespace sum_of_first_five_primes_with_units_digit_3_l90_90985

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90985


namespace range_of_a_l90_90191

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → 4 * a > x^2 - x^3) → a > 1 / 27 :=
by
  -- Proof to be filled
  sorry

end range_of_a_l90_90191


namespace find_k_l90_90062

theorem find_k (k : ℝ) : (∀ x y : ℝ, y = k * x + 3 → (x, y) = (2, 5)) → k = 1 :=
by
  sorry

end find_k_l90_90062


namespace percentage_reduction_in_price_of_oil_l90_90612

theorem percentage_reduction_in_price_of_oil :
  ∀ P : ℝ, ∀ R : ℝ, P = 800 / (800 / R - 5) ∧ R = 40 →
  (P - R) / P * 100 = 25 := by
  -- Assumptions
  intros P R h
  have hP : P = 800 / (800 / R - 5) := h.1
  have hR : R = 40 := h.2
  -- Result to be proved
  sorry

end percentage_reduction_in_price_of_oil_l90_90612


namespace perpendicular_lines_values_of_a_l90_90352

theorem perpendicular_lines_values_of_a (a : ℝ) :
  (∃ (a : ℝ), (∀ x y : ℝ, a * x - y + 2 * a = 0 ∧ (2 * a - 1) * x + a * y = 0) 
    ↔ (a = 0 ∨ a = 1))
  := sorry

end perpendicular_lines_values_of_a_l90_90352


namespace hyperbola_asymptotes_l90_90799

theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x ^ 2 / 4 - y ^ 2 / 16 = 1) → (y = 2 * x) ∨ (y = -2 * x) :=
sorry

end hyperbola_asymptotes_l90_90799


namespace count_three_digit_numbers_with_identical_digits_l90_90183

/-!
# Problem Statement:
Prove that the number of three-digit numbers with at least two identical digits is 252,
given that three-digit numbers range from 100 to 999.

## Definitions:
- Three-digit numbers are those in the range 100 to 999.

## Theorem:
The number of three-digit numbers with at least two identical digits is 252.
-/
theorem count_three_digit_numbers_with_identical_digits : 
    (∃ (n : ℕ), 100 ≤ n ∧ n ≤ 999 ∧ 
    ∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ (d1 = d2 ∨ d1 = d3 ∨ d2 = d3)) :=
sorry

end count_three_digit_numbers_with_identical_digits_l90_90183


namespace Mrs_Brown_points_l90_90315

-- Conditions given
variables (points_William points_Adams points_Daniel points_mean: ℝ) (num_classes: ℕ)

-- Define the conditions
def Mrs_William_points := points_William = 50
def Mr_Adams_points := points_Adams = 57
def Mrs_Daniel_points := points_Daniel = 57
def mean_condition := points_mean = 53.3
def num_classes_condition := num_classes = 4

-- Define the problem to prove
theorem Mrs_Brown_points :
  Mrs_William_points points_William ∧ Mr_Adams_points points_Adams ∧ Mrs_Daniel_points points_Daniel ∧ mean_condition points_mean ∧ num_classes_condition num_classes →
  ∃ (points_Brown: ℝ), points_Brown = 49 :=
by
  sorry

end Mrs_Brown_points_l90_90315


namespace sqrt_diff_inequality_l90_90218

open Real

theorem sqrt_diff_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 1) < sqrt (a - 2) - sqrt (a - 3) :=
sorry

end sqrt_diff_inequality_l90_90218


namespace A_inter_B_eq_C_l90_90535

noncomputable def A : Set ℝ := { x | ∃ α β : ℤ, α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def C : Set ℝ := {1, 2, 3, 4}

theorem A_inter_B_eq_C : A ∩ B = C :=
by
  sorry

end A_inter_B_eq_C_l90_90535


namespace find_S16_l90_90536

theorem find_S16 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 12 = -8)
  (h2 : S 9 = -9)
  (h_sum : ∀ n, S n = (n * (a 1 + a n) / 2)) :
  S 16 = -72 := 
by
  sorry

end find_S16_l90_90536


namespace additional_employees_hired_l90_90452

-- Conditions
def initial_employees : ℕ := 500
def hourly_wage : ℕ := 12
def daily_hours : ℕ := 10
def weekly_days : ℕ := 5
def weekly_hours := daily_hours * weekly_days
def monthly_weeks : ℕ := 4
def monthly_hours_per_employee := weekly_hours * monthly_weeks
def wage_per_employee_per_month := monthly_hours_per_employee * hourly_wage

-- Given new payroll
def new_monthly_payroll : ℕ := 1680000

-- Calculate the initial payroll
def initial_monthly_payroll := initial_employees * wage_per_employee_per_month

-- Statement of the proof problem
theorem additional_employees_hired :
  (new_monthly_payroll - initial_monthly_payroll) / wage_per_employee_per_month = 200 :=
by
  sorry

end additional_employees_hired_l90_90452


namespace calculate_percentage_l90_90829

theorem calculate_percentage :
  let total_students := 40
  let A_on_both := 4
  let B_on_both := 6
  let C_on_both := 3
  let D_on_Test1_C_on_Test2 := 2
  let valid_students := A_on_both + B_on_both + C_on_both + D_on_Test1_C_on_Test2
  (valid_students / total_students) * 100 = 37.5 :=
by
  sorry

end calculate_percentage_l90_90829


namespace robert_monthly_expenses_l90_90711

def robert_basic_salary : ℝ := 1250
def robert_sales : ℝ := 23600
def first_tier_limit : ℝ := 10000
def second_tier_limit : ℝ := 20000
def first_tier_rate : ℝ := 0.10
def second_tier_rate : ℝ := 0.12
def third_tier_rate : ℝ := 0.15
def savings_rate : ℝ := 0.20

def first_tier_commission : ℝ :=
  first_tier_limit * first_tier_rate

def second_tier_commission : ℝ :=
  (second_tier_limit - first_tier_limit) * second_tier_rate

def third_tier_commission : ℝ :=
  (robert_sales - second_tier_limit) * third_tier_rate

def total_commission : ℝ :=
  first_tier_commission + second_tier_commission + third_tier_commission

def total_earnings : ℝ :=
  robert_basic_salary + total_commission

def savings : ℝ :=
  total_earnings * savings_rate

def monthly_expenses : ℝ :=
  total_earnings - savings

theorem robert_monthly_expenses :
  monthly_expenses = 3192 := by
  sorry

end robert_monthly_expenses_l90_90711


namespace count_bitonic_integers_l90_90933

-- Definition of a bitonic integer
def is_bitonic (n : ℕ) : Prop :=
  ∃ (l : List ℕ), l.length ≥ 3 ∧ l.nodup ∧ l ∈ (Finset.range 9).lists
  ∧ (∀ (i : ℕ), i < l.length - 1 → l.nth_le i _ < l.nth_le (i + 1) _) -- Strictly increasing
  ∧ (∀ (i : ℕ), i > 0 → i < l.length → l.nth_le i _ < l.nth_le (i - 1) _) -- Strictly decreasing

-- Proof that the number of bitonic integers using only the digits from 1 to 9, each at most once, is 1458
theorem count_bitonic_integers : 
  (Finset.filter is_bitonic ((Finset.range 9).powerset)).card = 1458 :=
by
  sorry

end count_bitonic_integers_l90_90933


namespace initial_number_of_professors_l90_90024

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l90_90024


namespace train_platform_length_l90_90310

theorem train_platform_length 
  (speed_train_kmph : ℕ) 
  (time_cross_platform : ℕ) 
  (time_cross_man : ℕ) 
  (L_platform : ℕ) :
  speed_train_kmph = 72 ∧ 
  time_cross_platform = 34 ∧ 
  time_cross_man = 18 ∧ 
  L_platform = 320 :=
by
  sorry

end train_platform_length_l90_90310


namespace german_team_goals_possible_goal_values_l90_90109

def german_team_goal_conditions (x : ℕ) :=
  (10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ∧ (x % 2 = 1)

def exactly_two_journalists_correct (x : ℕ) :=
  (10 < x ∧ x < 17) ∨ (11 < x ∧ x < 18) ∨ (x % 2 = 0)

theorem german_team_goals : ∀ (x : ℕ), german_team_goal_conditions x → exactly_two_journalists_correct x :=
begin
  intros x h_goal_conditions,
  -- This part is meant for proving the statement that exactly two of the journalistic statements are true.
  sorry
end

def possible_goals : set ℕ := { x | x ∈ {11, 12, 14, 16, 17} }

theorem possible_goal_values : ∀ (x : ℕ), x ∈ possible_goals :=
begin
  -- Proof that given the conditions, the possible goals must lie within the set {11, 12, 14, 16, 17}.
  sorry
end

end german_team_goals_possible_goal_values_l90_90109


namespace sum_of_squares_l90_90545

theorem sum_of_squares (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 1) (h2 : b^2 + b * c + c^2 = 3) (h3 : c^2 + c * a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := 
sorry

end sum_of_squares_l90_90545


namespace total_cost_of_tickets_l90_90279

theorem total_cost_of_tickets (num_family_members num_adult_tickets num_children_tickets : ℕ)
    (cost_adult_ticket cost_children_ticket total_cost : ℝ) 
    (h1 : num_family_members = 7) 
    (h2 : cost_adult_ticket = 21) 
    (h3 : cost_children_ticket = 14) 
    (h4 : num_adult_tickets = 4) 
    (h5 : num_children_tickets = num_family_members - num_adult_tickets) 
    (h6 : total_cost = num_adult_tickets * cost_adult_ticket + num_children_tickets * cost_children_ticket) :
    total_cost = 126 :=
by
  sorry

end total_cost_of_tickets_l90_90279


namespace number_of_triangles_l90_90758

/-!
# Problem Statement
Given a square with 20 interior points connected such that the lines do not intersect and divide the square into triangles,
prove that the number of triangles formed is 42.
-/

theorem number_of_triangles (V E F : ℕ) (hV : V = 24) (hE : E = (3 * F + 1) / 2) (hF : V - E + F = 2) :
  (F - 1) = 42 :=
by
  sorry

end number_of_triangles_l90_90758


namespace burn_5_sticks_per_hour_l90_90039

-- Define the number of sticks each type of furniture makes
def sticks_per_chair := 6
def sticks_per_table := 9
def sticks_per_stool := 2

-- Define the number of each furniture Mary chopped up
def chairs_chopped := 18
def tables_chopped := 6
def stools_chopped := 4

-- Define the total number of hours Mary can keep warm
def hours_warm := 34

-- Calculate the total number of sticks of wood from each type of furniture
def total_sticks_chairs := chairs_chopped * sticks_per_chair
def total_sticks_tables := tables_chopped * sticks_per_table
def total_sticks_stools := stools_chopped * sticks_per_stool

-- Calculate the total number of sticks of wood
def total_sticks := total_sticks_chairs + total_sticks_tables + total_sticks_stools

-- The number of sticks of wood Mary needs to burn per hour
def sticks_per_hour := total_sticks / hours_warm

-- Prove that Mary needs to burn 5 sticks per hour to stay warm
theorem burn_5_sticks_per_hour : sticks_per_hour = 5 := sorry

end burn_5_sticks_per_hour_l90_90039


namespace area_of_circle_l90_90077

theorem area_of_circle (r : ℝ) (h : r = 3) : 
  (∀ A : ℝ, A = π * r^2) → A = 9 * π :=
by
  intro area_formula
  sorry

end area_of_circle_l90_90077


namespace min_value_expression_l90_90343

theorem min_value_expression : ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by
  intro x y
  sorry

end min_value_expression_l90_90343


namespace ratio_percent_l90_90610

theorem ratio_percent (x : ℕ) (h : (15 / x : ℚ) = 60 / 100) : x = 25 := 
sorry

end ratio_percent_l90_90610


namespace probability_genuine_given_equal_weight_l90_90042

noncomputable def total_coins : ℕ := 15
noncomputable def genuine_coins : ℕ := 12
noncomputable def counterfeit_coins : ℕ := 3

def condition_A : Prop := true
def condition_B (weights : Fin 6 → ℝ) : Prop :=
  weights 0 + weights 1 = weights 2 + weights 3 ∧
  weights 0 + weights 1 = weights 4 + weights 5

noncomputable def P_A_and_B : ℚ := (44 / 70) * (15 / 26) * (28 / 55)
noncomputable def P_B : ℚ := 44 / 70

theorem probability_genuine_given_equal_weight :
  P_A_and_B / P_B = 264 / 443 :=
by
  sorry

end probability_genuine_given_equal_weight_l90_90042


namespace total_number_of_candles_l90_90835

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l90_90835


namespace product_of_y_values_l90_90949

theorem product_of_y_values :
  (∀ (x y : ℤ), x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → (y = 1 ∨ y = 2)) →
  (∀ (x y₁ x' y₂ : ℤ), (x, y₁) ≠ (x', y₂) → x = x' ∨ y₁ ≠ y₂) →
  (∀ (x y : ℤ), (x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → y = 1 ∨ y = 2) →
    (∃ (y₁ y₂ : ℤ), y₁ = 1 ∧ y₂ = 2 ∧ y₁ * y₂ = 2)) :=
by {
  sorry
}

end product_of_y_values_l90_90949


namespace max_distance_from_circle_to_line_l90_90231

-- Definitions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 2 * y + 24 / 5 = 0
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 0 

-- Theorem Statement
theorem max_distance_from_circle_to_line : 
  ∃ p : ℝ × ℝ, circle_eq p.1 p.2 ∧ ∃ d : ℝ, d = abs ((-2 : ℝ) * 3 + (1 : ℝ) * 4) / sqrt (3^2 + 4^2) + sqrt 5 / 5 ∧ d = (2 + sqrt 5) / 5 :=
by
  sorry

end max_distance_from_circle_to_line_l90_90231


namespace percentage_not_pens_pencils_erasers_l90_90055

-- Define the given percentages
def percentPens : ℝ := 42
def percentPencils : ℝ := 25
def percentErasers : ℝ := 12
def totalPercent : ℝ := 100

-- The goal is to prove that the percentage of sales that were not pens, pencils, or erasers is 21%
theorem percentage_not_pens_pencils_erasers :
  totalPercent - (percentPens + percentPencils + percentErasers) = 21 := by
  sorry

end percentage_not_pens_pencils_erasers_l90_90055


namespace parallel_lines_m_eq_one_l90_90686

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y + (m - 2) = 0 ∧ 2 * m * x + 4 * y + 16 = 0 → m = 1) :=
by
  sorry

end parallel_lines_m_eq_one_l90_90686


namespace route_numbers_351_l90_90872

theorem route_numbers_351 (displayed : ℕ) (up_to_two_segments_broken : Prop) : 
  displayed = 351 → 
  (up_to_two_segments_broken → 
    {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991} = 
    {n : ℕ | ∃ route : ℕ, route = 351 ∧ up_to_two_segments_broken}) :=
by 
  intros h1 h2
  apply set.ext
  intro n
  split
  -- Direction 1: n in the left set implies n in the right set
  { intro hn
    cases hn
    -- Prove for each element in the set
    exact ⟨351, rfl, h2⟩
    -- Repeat this for each number in the set or use automation if known
  }
  -- Direction 2: n in the right set implies n in the left set
  { rintro ⟨route, hr1, hr2⟩
    rw hr1 at hr2
    -- Prove that each possible route number is in the set on the right
    exact hn
  }
  sorry

end route_numbers_351_l90_90872


namespace fruit_salad_cherries_l90_90093

theorem fruit_salad_cherries (b r g c : ℕ) 
(h1 : b + r + g + c = 360)
(h2 : r = 3 * b) 
(h3 : g = 4 * c)
(h4 : c = 5 * r) :
c = 68 := 
sorry

end fruit_salad_cherries_l90_90093


namespace probability_diagonals_intersect_l90_90249

-- Define some basic combinatorial functions in Lean
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the total number of diagonals in a regular nonagon
def total_diagonals (n : ℕ) : ℕ := binom n 2 - n 

-- Define the number of ways to pick 2 diagonals from the total diagonals
def total_pairs_of_diagonals (d : ℕ) : ℕ := binom d 2

-- Define the number of sets of intersecting diagonals
def intersecting_diagonals (n : ℕ) : ℕ := binom n 4

-- Define the given problem as a theorem in Lean
theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) : 
  ((intersecting_diagonals n : ℚ) / (total_pairs_of_diagonals (total_diagonals n) : ℚ)) = 6 / 13 := 
by
  sorry 

end probability_diagonals_intersect_l90_90249


namespace max_students_before_new_year_l90_90749

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l90_90749


namespace number_of_schools_l90_90160

theorem number_of_schools (total_students d : ℕ) (S : ℕ) (ellen frank : ℕ) (d_median : total_students = 2 * d - 1)
    (d_highest : ellen < d) (ellen_position : ellen = 29) (frank_position : frank = 50) (team_size : ∀ S, total_students = 3 * S) : 
    S = 19 := 
by 
  sorry

end number_of_schools_l90_90160


namespace lottery_ticket_not_necessarily_win_l90_90501

/-- Given a lottery with 1,000,000 tickets and a winning rate of 0.001, buying 1000 tickets may not necessarily win. -/
theorem lottery_ticket_not_necessarily_win (total_tickets : ℕ) (winning_rate : ℚ) (n_tickets : ℕ) :
  total_tickets = 1000000 →
  winning_rate = 1 / 1000 →
  n_tickets = 1000 →
  ∃ (p : ℚ), 0 < p ∧ p < 1 ∧ (p ^ n_tickets) < (1 / total_tickets) := 
by
  intros h_total h_rate h_n
  sorry

end lottery_ticket_not_necessarily_win_l90_90501


namespace EF_length_proof_l90_90548

noncomputable def length_BD (AB BC : ℝ) : ℝ := Real.sqrt (AB^2 + BC^2)

noncomputable def length_EF (BD AB BC : ℝ) : ℝ :=
  let BE := BD * AB / BD
  let BF := BD * BC / AB
  BE + BF

theorem EF_length_proof : 
  ∀ (AB BC : ℝ), AB = 4 ∧ BC = 3 →
  length_EF (length_BD AB BC) AB BC = 125 / 12 :=
by
  intros AB BC h
  rw [length_BD, length_EF]
  simp
  rw [Real.sqrt_eq_rpow]
  simp
  sorry

end EF_length_proof_l90_90548


namespace dissimilar_terms_expansion_count_l90_90792

noncomputable def num_dissimilar_terms_in_expansion (a b c d : ℝ) : ℕ :=
  let n := 8
  let k := 4
  Nat.choose (n + k - 1) (k - 1)

theorem dissimilar_terms_expansion_count : 
  num_dissimilar_terms_in_expansion a b c d = 165 := by
  sorry

end dissimilar_terms_expansion_count_l90_90792


namespace train_speed_l90_90891

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l90_90891


namespace find_value_of_alpha_beta_plus_alpha_plus_beta_l90_90806

variable (α β : ℝ)

theorem find_value_of_alpha_beta_plus_alpha_plus_beta
  (hα : α^2 + α - 1 = 0)
  (hβ : β^2 + β - 1 = 0)
  (hαβ : α ≠ β) :
  α * β + α + β = -2 := 
by
  sorry

end find_value_of_alpha_beta_plus_alpha_plus_beta_l90_90806


namespace gcd_lcm_identity_l90_90563

theorem gcd_lcm_identity (a b: ℕ) (h_lcm: (Nat.lcm a b) = 4620) (h_gcd: Nat.gcd a b = 33) (h_a: a = 231) : b = 660 := by
  sorry

end gcd_lcm_identity_l90_90563


namespace position_of_UKIMC_l90_90578

open Finset

def dictionary_order_position {α : Type*} [DecidableEq α] [LinearOrder α] (s : Finset α) (l : List α) : Option ℕ :=
  let permutations := s.perms.sort
  permutations.find_index (λ p, p = l)

theorem position_of_UKIMC : dictionary_order_position (finset.cons 'U' (finset.cons 'K' (finset.cons 'M' (finset.cons 'I' (finset.cons 'C' finset.empty) sorry) sorry) sorry) sorry) ['U', 'K', 'I', 'M', 'C'] = some 110 :=
sorry

end position_of_UKIMC_l90_90578


namespace Bella_bought_38_stamps_l90_90138

def stamps (n t r : ℕ) : ℕ :=
  n + t + r

theorem Bella_bought_38_stamps :
  ∃ (n t r : ℕ),
    n = 11 ∧
    t = n + 9 ∧
    r = t - 13 ∧
    stamps n t r = 38 := 
  by
  sorry

end Bella_bought_38_stamps_l90_90138


namespace rank_from_right_l90_90308

theorem rank_from_right (rank_from_left total_students : ℕ) (h1 : rank_from_left = 5) (h2 : total_students = 10) :
  total_students - rank_from_left + 1 = 6 :=
by 
  -- Placeholder for the actual proof.
  sorry

end rank_from_right_l90_90308


namespace students_pass_both_subjects_l90_90086

theorem students_pass_both_subjects
  (F_H F_E F_HE : ℝ)
  (h1 : F_H = 0.25)
  (h2 : F_E = 0.48)
  (h3 : F_HE = 0.27) :
  (100 - (F_H + F_E - F_HE) * 100) = 54 :=
by
  sorry

end students_pass_both_subjects_l90_90086


namespace youtube_more_than_tiktok_l90_90849

-- Definitions for followers in different social media platforms
def instagram_followers : ℕ := 240
def facebook_followers : ℕ := 500
def total_followers : ℕ := 3840

-- Number of followers on Twitter is half the sum of followers on Instagram and Facebook
def twitter_followers : ℕ := (instagram_followers + facebook_followers) / 2

-- Number of followers on TikTok is 3 times the followers on Twitter
def tiktok_followers : ℕ := 3 * twitter_followers

-- Calculate the number of followers on all social media except YouTube
def other_followers : ℕ := instagram_followers + facebook_followers + twitter_followers + tiktok_followers

-- Number of followers on YouTube
def youtube_followers : ℕ := total_followers - other_followers

-- Prove the number of followers on YouTube is greater than TikTok by a certain amount
theorem youtube_more_than_tiktok : youtube_followers - tiktok_followers = 510 := by
  -- Sorry is a placeholder for the proof
  sorry

end youtube_more_than_tiktok_l90_90849


namespace lukas_averages_points_l90_90705

theorem lukas_averages_points (total_points : ℕ) (num_games : ℕ) (average_points : ℕ)
  (h_total: total_points = 60) (h_games: num_games = 5) : average_points = total_points / num_games :=
sorry

end lukas_averages_points_l90_90705


namespace harmonic_mean_4_5_10_l90_90485

theorem harmonic_mean_4_5_10 : HarmonicMean 4 5 10 = 60 / 11 :=
by
  sorry

-- Define what HarmonicMean means
noncomputable def HarmonicMean (a b c : ℚ) : ℚ := 
  3 / (1/a + 1/b + 1/c)

end harmonic_mean_4_5_10_l90_90485


namespace abs_diff_ge_abs_sum_iff_non_positive_prod_l90_90805

theorem abs_diff_ge_abs_sum_iff_non_positive_prod (a b : ℝ) : 
  |a - b| ≥ |a| + |b| ↔ a * b ≤ 0 := 
by sorry

end abs_diff_ge_abs_sum_iff_non_positive_prod_l90_90805


namespace find_other_factor_l90_90914

theorem find_other_factor (n : ℕ) (hn : n = 75) :
    ( ∃ k, k = 25 ∧ ∃ m, (k * 3^3 * m = 75 * 2^5 * 6^2 * 7^3) ) :=
by
  sorry

end find_other_factor_l90_90914


namespace probability_same_university_l90_90373

theorem probability_same_university :
  let universities := 5
  let total_ways := universities * universities
  let favorable_ways := universities
  (favorable_ways : ℚ) / total_ways = (1 / 5 : ℚ) := 
by
  sorry

end probability_same_university_l90_90373


namespace Gina_college_total_cost_l90_90491

theorem Gina_college_total_cost :
  let credits := 14
  let cost_per_credit := 450
  let num_textbooks := 5
  let cost_per_textbook := 120
  let facilities_fee := 200
  let total_cost := (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee
  total_cost = 7100 :=
by
  let credits := 14
  let cost_per_credit := 450
  let num_textbooks := 5
  let cost_per_textbook := 120
  let facilities_fee := 200
  let total_cost := (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee
  show total_cost = 7100 from sorry

end Gina_college_total_cost_l90_90491


namespace percentage_difference_l90_90194

variable (x y : ℝ)
variable (p : ℝ)  -- percentage by which x is less than y

theorem percentage_difference (h1 : y = x * 1.3333333333333333) : p = 25 :=
by
  sorry

end percentage_difference_l90_90194


namespace find_ordered_pairs_l90_90513

theorem find_ordered_pairs :
  {p : ℝ × ℝ | p.1 > p.2 ∧ (p.1 - p.2 = 2 * p.1 / p.2 ∨ p.1 - p.2 = 2 * p.2 / p.1)} = 
  {(8, 4), (9, 3), (2, 1)} :=
sorry

end find_ordered_pairs_l90_90513


namespace red_light_max_probability_l90_90314

theorem red_light_max_probability {m : ℕ} (h1 : m > 0) (h2 : m < 35) :
  m = 3 ∨ m = 15 ∨ m = 30 ∨ m = 40 → m = 30 :=
by
  sorry

end red_light_max_probability_l90_90314


namespace fair_coin_three_flips_probability_l90_90592

theorem fair_coin_three_flips_probability :
  ∀ (prob : ℕ → ℚ) (independent : ∀ n, prob n = 1 / 2),
    prob 0 * prob 1 * prob 2 = 1 / 8 := 
by
  intros prob independent
  sorry

end fair_coin_three_flips_probability_l90_90592


namespace arithmetic_seq_a2_a4_a6_sum_l90_90526

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def S_n (a : ℕ → ℤ) (n : ℕ) := n * (a 1 + a n) / 2

def S_7_eq_14 (a : ℕ → ℤ) := S_n a 7 = 14

-- The statement to prove
theorem arithmetic_seq_a2_a4_a6_sum (h_arith: is_arithmetic_sequence a) (h_sum: S_7_eq_14 a) :
  a 2 + a 4 + a 6 = 6 :=
sorry

end arithmetic_seq_a2_a4_a6_sum_l90_90526


namespace committee_count_correct_l90_90137

noncomputable def num_possible_committees : Nat :=
sorry -- Numerical expression omitted; focus on statement structure.

theorem committee_count_correct :
  let num_men_math := 3
  let num_women_math := 3
  let num_men_stats := 2
  let num_women_stats := 3
  let num_men_cs := 2
  let num_women_cs := 3 in
  let total_men := num_men_math + num_men_stats + num_men_cs
  let total_women := num_women_math + num_women_stats + num_women_cs in
  
  (num_possible_committees = 1050) ↔
  let committee_size := 7
  let men_comm := 3
  let women_comm := 4 in
  let math_comm := 2 in

  (total_men >= men_comm) ∧
  (total_women >= women_comm) ∧
  (math_comm <= num_men_math + num_women_math) :=
  sorry -- Use the provided computational steps to verify.

end committee_count_correct_l90_90137


namespace pastry_trick_l90_90283

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l90_90283


namespace price_per_jin_of_tomatoes_is_3yuan_3jiao_l90_90470

/-- Definitions of the conditions --/
def cucumbers_cost_jin : ℕ := 5
def cucumbers_cost_yuan : ℕ := 11
def cucumbers_cost_jiao : ℕ := 8
def tomatoes_cost_jin : ℕ := 4
def difference_cost_yuan : ℕ := 1
def difference_cost_jiao : ℕ := 4

/-- Converting cost in yuan and jiao to decimal yuan --/
def cost_in_yuan (yuan jiao : ℕ) : ℕ := yuan + jiao / 10

/-- Given conditions in decimal --/
def cucumbers_cost := cost_in_yuan cucumbers_cost_yuan cucumbers_cost_jiao
def difference_cost := cost_in_yuan difference_cost_yuan difference_cost_jiao
def tomatoes_cost := cucumbers_cost + difference_cost

/-- Proof statement: price per jin of tomatoes in yuan and jiao --/
theorem price_per_jin_of_tomatoes_is_3yuan_3jiao :
  tomatoes_cost / tomatoes_cost_jin = 3 + 3 / 10 :=
by
  sorry

end price_per_jin_of_tomatoes_is_3yuan_3jiao_l90_90470


namespace evaluate_expression_l90_90154

theorem evaluate_expression :
  ∀ (x y : ℕ), x = 3 → y = 4 → 5 * x^y + 6 * y^x = 789 :=
by
  intros x y hx hy
  rw [hx, hy]
  simp
  exact sorry

end evaluate_expression_l90_90154


namespace correct_statements_in_triangle_l90_90381

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = π) :
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a^3 + b^3 = c^3 → a^2 + b^2 > c^2) :=
by
  sorry

end correct_statements_in_triangle_l90_90381


namespace chocolate_bars_in_box_l90_90641

theorem chocolate_bars_in_box (x : ℕ) (h1 : 2 * (x - 4) = 18) : x = 13 := 
by {
  sorry
}

end chocolate_bars_in_box_l90_90641


namespace miss_tree_class_children_count_l90_90579

noncomputable def number_of_children (n: ℕ) : ℕ := 7 * n + 2

theorem miss_tree_class_children_count (n : ℕ) :
  (20 < number_of_children n) ∧ (number_of_children n < 30) ∧ 7 * n + 2 = 23 :=
by {
  sorry
}

end miss_tree_class_children_count_l90_90579


namespace cost_price_of_article_l90_90774

theorem cost_price_of_article
  (C SP1 SP2 : ℝ)
  (h1 : SP1 = 0.8 * C)
  (h2 : SP2 = 1.05 * C)
  (h3 : SP2 = SP1 + 100) : 
  C = 400 := 
sorry

end cost_price_of_article_l90_90774


namespace cubic_roots_solution_sum_l90_90057

theorem cubic_roots_solution_sum (u v w : ℝ) (h1 : (u - 2) * (u - 3) * (u - 4) = 1 / 2)
                                     (h2 : (v - 2) * (v - 3) * (v - 4) = 1 / 2)
                                     (h3 : (w - 2) * (w - 3) * (w - 4) = 1 / 2)
                                     (distinct_roots : u ≠ v ∧ v ≠ w ∧ u ≠ w) :
  u^3 + v^3 + w^3 = -42 :=
sorry

end cubic_roots_solution_sum_l90_90057


namespace power_of_fraction_l90_90786

theorem power_of_fraction :
  (3 / 4) ^ 5 = 243 / 1024 :=
by sorry

end power_of_fraction_l90_90786


namespace distance_to_nearest_edge_of_picture_l90_90614

def wall_width : ℕ := 26
def picture_width : ℕ := 4
def distance_from_end (wall picture : ℕ) : ℕ := (wall - picture) / 2

theorem distance_to_nearest_edge_of_picture :
  distance_from_end wall_width picture_width = 11 :=
sorry

end distance_to_nearest_edge_of_picture_l90_90614


namespace cost_of_500_candies_l90_90560

theorem cost_of_500_candies (cost_per_candy_in_cents : ℕ) (cents_in_dollar : ℕ) : 
  (cost_per_candy_in_cents = 2) → (cents_in_dollar = 100) → ((500 * cost_per_candy_in_cents) / cents_in_dollar = 10) :=
begin
  intros h_cost h_cents,
  rw [h_cost, h_cents],
  norm_num,
end

end cost_of_500_candies_l90_90560


namespace max_blocks_fit_l90_90587

-- Define the dimensions of the block
def block_length : ℕ := 3
def block_width : ℕ := 1
def block_height : ℕ := 1

-- Define the dimensions of the box
def box_length : ℕ := 5
def box_width : ℕ := 3
def box_height : ℕ := 2

-- Theorem stating the maximum number of blocks that can fit in the box
theorem max_blocks_fit :
  (box_length * box_width * box_height) / (block_length * block_width * block_height) = 15 := sorry

end max_blocks_fit_l90_90587


namespace find_k_l90_90987

theorem find_k : ∃ k : ℕ, 32 / k = 4 ∧ k = 8 := 
sorry

end find_k_l90_90987


namespace eval_polynomial_l90_90796

theorem eval_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) : x^3 - 3 * x^2 - 9 * x + 27 = 27 := 
by
  sorry

end eval_polynomial_l90_90796


namespace simplify_sqrt_25000_l90_90050

theorem simplify_sqrt_25000 : Real.sqrt 25000 = 50 * Real.sqrt 10 := 
by
  sorry

end simplify_sqrt_25000_l90_90050


namespace num_20_paise_coins_l90_90748

theorem num_20_paise_coins (x y : ℕ) (h1 : x + y = 344) (h2 : 20 * x + 25 * y = 7100) : x = 300 :=
by
  sorry

end num_20_paise_coins_l90_90748


namespace hexagon_largest_angle_l90_90887

theorem hexagon_largest_angle (x : ℝ) 
  (h_angles_sum : 80 + 100 + x + x + x + (2 * x + 20) = 720) : 
  (2 * x + 20) = 228 :=
by 
  sorry

end hexagon_largest_angle_l90_90887


namespace geometric_progression_common_ratio_l90_90828

-- Definitions and theorems
variable {α : Type*} [OrderedCommRing α]

theorem geometric_progression_common_ratio
  (a : α) (r : α)
  (h_pos : a > 0)
  (h_geometric : ∀ n : ℕ, a * r^n = (a * r^(n + 1)) * (a * r^(n + 2))):
  r = 1 := by
  sorry

end geometric_progression_common_ratio_l90_90828


namespace range_of_u_l90_90350

variable (a b u : ℝ)

theorem range_of_u (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (9 / b) = 1) : u ≤ 16 :=
by
  sorry

end range_of_u_l90_90350


namespace david_boxes_l90_90939

theorem david_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) (boxes : ℕ) 
  (h1 : total_dogs = 28) (h2 : dogs_per_box = 4) : 
  boxes = total_dogs / dogs_per_box → boxes = 7 :=
by
  intros h
  rw [h1, h2] at h
  exact h
  sorry

end david_boxes_l90_90939


namespace initial_number_of_professors_l90_90023

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l90_90023


namespace simplified_sum_l90_90931

theorem simplified_sum :
  (-1 : ℤ) ^ 2002 + (-1 : ℤ) ^ 2003 + 2 ^ 2004 - 2 ^ 2003 = 2 ^ 2003 := 
by 
  sorry -- Proof skipped

end simplified_sum_l90_90931


namespace curve_C1_general_equation_curve_C2_cartesian_equation_PA_PB_abs_diff_l90_90378

noncomputable def curve_C1 := { p : ℝ × ℝ | ∃ (θ : ℝ), p.1 = 2 + sqrt 3 * Real.cos θ ∧ p.2 = sqrt 3 * Real.sin θ }
noncomputable def curve_C2 := { p : ℝ × ℝ | p.2 = (1 / sqrt 3) * p.1 }

theorem curve_C1_general_equation : ∀ (x y : ℝ), ((x = 2 + sqrt 3 * Real.cos θ) ∧ (y = sqrt 3 * Real.sin θ)) → (x - 2)^2 + y^2 = 3 := 
sorry

theorem curve_C2_cartesian_equation : ∀ (x y : ℝ), (y = (1 / sqrt 3) * x) ↔ (θ = π / 6) :=
sorry

theorem PA_PB_abs_diff : 
  let P : ℝ × ℝ := (3, sqrt 3),
      A : ℝ × ℝ := ((3 + sqrt 6) / 2, (sqrt 3 + sqrt 2) / 2),
      B : ℝ × ℝ := ((3 - sqrt 6) / 2, (sqrt 3 - sqrt 2) / 2) in
  abs ((Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) - (Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2))) = 2 * sqrt 2 :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_PA_PB_abs_diff_l90_90378


namespace twice_minus_three_algebraic_l90_90946

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l90_90946


namespace positive_integer_solution_inequality_l90_90877

theorem positive_integer_solution_inequality (x : ℕ) (h : 2 * (x + 1) ≥ 5 * x - 3) : x = 1 :=
by {
  sorry
}

end positive_integer_solution_inequality_l90_90877


namespace product_of_solutions_eq_zero_l90_90638

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) → (x = 0 ∨ x = -4 / 7)) → (0 = 0) := 
by
  intros h
  sorry

end product_of_solutions_eq_zero_l90_90638


namespace number_153_satisfies_l90_90726

noncomputable def sumOfCubes (n : ℕ) : ℕ :=
  (n % 10)^3 + ((n / 10) % 10)^3 + ((n / 100) % 10)^3

theorem number_153_satisfies :
  (sumOfCubes 153) = 153 ∧ 
  (153 % 10 ≠ 0) ∧ ((153 / 10) % 10 ≠ 0) ∧ ((153 / 100) % 10 ≠ 0) ∧ 
  153 ≠ 1 :=
by {
  sorry
}

end number_153_satisfies_l90_90726


namespace leonardo_needs_more_money_l90_90840

-- Defining the problem
def cost_of_chocolate : ℕ := 500 -- 5 dollars in cents
def leonardo_own_money : ℕ := 400 -- 4 dollars in cents
def borrowed_money : ℕ := 59 -- borrowed cents

-- Prove that Leonardo needs 41 more cents
theorem leonardo_needs_more_money : (cost_of_chocolate - (leonardo_own_money + borrowed_money) = 41) :=
by
  sorry

end leonardo_needs_more_money_l90_90840


namespace line_equation_through_point_slope_l90_90339

theorem line_equation_through_point_slope :
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ (a * 1 + b * 3 + c = 0) ∧ (y = -4 * x → k = -4 / 9) ∧ (∀ (x y : ℝ), y - 3 = k * (x - 1) → 4 * x + 3 * y - 13 = 0) :=
sorry

end line_equation_through_point_slope_l90_90339


namespace find_ding_score_l90_90201

noncomputable def jia_yi_bing_avg_score : ℕ := 89
noncomputable def four_avg_score := jia_yi_bing_avg_score + 2
noncomputable def four_total_score := 4 * four_avg_score
noncomputable def jia_yi_bing_total_score := 3 * jia_yi_bing_avg_score
noncomputable def ding_score := four_total_score - jia_yi_bing_total_score

theorem find_ding_score : ding_score = 97 := 
by
  sorry

end find_ding_score_l90_90201


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90963

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90963


namespace solve_recurrence_relation_l90_90224

noncomputable def a_n (n : ℕ) : ℝ := 2 * 4^n - 2 * n + 2
noncomputable def b_n (n : ℕ) : ℝ := 2 * 4^n + 2 * n - 2

theorem solve_recurrence_relation :
  a_n 0 = 4 ∧ b_n 0 = 0 ∧
  (∀ n : ℕ, a_n (n + 1) = 3 * a_n n + b_n n - 4) ∧
  (∀ n : ℕ, b_n (n + 1) = 2 * a_n n + 2 * b_n n + 2) :=
by
  sorry

end solve_recurrence_relation_l90_90224


namespace y_is_80_percent_less_than_x_l90_90302

theorem y_is_80_percent_less_than_x (x y : ℝ) (h : x = 5 * y) : ((x - y) / x) * 100 = 80 :=
by sorry

end y_is_80_percent_less_than_x_l90_90302


namespace sum_is_correct_l90_90954

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l90_90954


namespace terminal_side_third_quadrant_l90_90992

noncomputable def angle_alpha : ℝ := (7 * Real.pi) / 5

def is_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ k : ℤ, (3 * Real.pi) / 2 < angle + 2 * k * Real.pi ∧ angle + 2 * k * Real.pi < 2 * Real.pi

theorem terminal_side_third_quadrant : is_in_third_quadrant angle_alpha :=
sorry

end terminal_side_third_quadrant_l90_90992


namespace total_cleaning_time_l90_90158

-- Definition for the problem conditions
def time_to_clean_egg (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ) : ℕ :=
  (num_eggs * seconds_per_egg) / seconds_per_minute

def time_to_clean_toilet_paper (minutes_per_roll : ℕ) (num_rolls : ℕ) : ℕ :=
  num_rolls * minutes_per_roll

-- Main statement to prove the total cleaning time
theorem total_cleaning_time
  (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ)
  (minutes_per_roll : ℕ) (num_rolls : ℕ) :
  seconds_per_egg = 15 →
  num_eggs = 60 →
  seconds_per_minute = 60 →
  minutes_per_roll = 30 →
  num_rolls = 7 →
  time_to_clean_egg seconds_per_egg num_eggs seconds_per_minute +
  time_to_clean_toilet_paper minutes_per_roll num_rolls = 225 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cleaning_time_l90_90158


namespace squats_day_after_tomorrow_l90_90150

theorem squats_day_after_tomorrow (initial_squats : ℕ) (daily_increase : ℕ) (today : ℕ) (tomorrow : ℕ) (day_after_tomorrow : ℕ)
  (h1 : initial_squats = 30)
  (h2 : daily_increase = 5)
  (h3 : today = initial_squats + daily_increase)
  (h4 : tomorrow = today + daily_increase)
  (h5 : day_after_tomorrow = tomorrow + daily_increase) : 
  day_after_tomorrow = 45 := 
sorry

end squats_day_after_tomorrow_l90_90150


namespace germs_left_percentage_l90_90278

-- Defining the conditions
def first_spray_kill_percentage : ℝ := 0.50
def second_spray_kill_percentage : ℝ := 0.25
def overlap_percentage : ℝ := 0.05
def total_kill_percentage : ℝ := first_spray_kill_percentage + second_spray_kill_percentage - overlap_percentage

-- The statement to be proved
theorem germs_left_percentage :
  1 - total_kill_percentage = 0.30 :=
by
  -- The proof would go here.
  sorry

end germs_left_percentage_l90_90278


namespace five_letters_three_mailboxes_l90_90045

theorem five_letters_three_mailboxes : (∃ n : ℕ, n = 5) ∧ (∃ m : ℕ, m = 3) → ∃ k : ℕ, k = m^n :=
by
  sorry

end five_letters_three_mailboxes_l90_90045


namespace average_visitors_on_other_days_l90_90096

theorem average_visitors_on_other_days 
  (avg_sunday : ℕ) (avg_month : ℕ) 
  (days_in_month : ℕ) (sundays : ℕ) (other_days : ℕ) 
  (visitors_on_other_days : ℕ) :
  avg_sunday = 510 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  (sundays * avg_sunday + other_days * visitors_on_other_days = avg_month * days_in_month) →
  visitors_on_other_days = 240 :=
by
  intros hs hm hd hsunded hotherdays heq
  sorry

end average_visitors_on_other_days_l90_90096


namespace equation_holds_iff_b_eq_c_l90_90688

theorem equation_holds_iff_b_eq_c (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h4 : a ≠ b) (h5 : a ≠ c) (h6 : b ≠ c) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a * a + 100 * a + b + c ↔ b = c :=
by sorry

end equation_holds_iff_b_eq_c_l90_90688


namespace chicken_legs_baked_l90_90626

theorem chicken_legs_baked (L : ℕ) (H₁ : 144 / 16 = 9) (H₂ : 224 / 16 = 14) (H₃ : 16 * 9 = 144) :  L = 144 :=
by
  sorry

end chicken_legs_baked_l90_90626


namespace loaned_out_books_l90_90771

def initial_books : ℕ := 75
def added_books : ℕ := 10 + 15 + 6
def removed_books : ℕ := 3 + 2 + 4
def end_books : ℕ := 90
def return_percentage : ℝ := 0.80

theorem loaned_out_books (L : ℕ) :
  (end_books - initial_books = added_books - removed_books - ⌊(1 - return_percentage) * L⌋) →
  (L = 35) :=
sorry

end loaned_out_books_l90_90771


namespace solution_contains_non_zero_arrays_l90_90162

noncomputable def verify_non_zero_array (x y z w : ℝ) : Prop :=
  1 + (1 / x) + (2 * (x + 1) / (x * y)) + (3 * (x + 1) * (y + 2) / (x * y * z)) + 
  (4 * (x + 1) * (y + 2) * (z + 3) / (x * y * z * w)) = 0

theorem solution_contains_non_zero_arrays (x y z w : ℝ) (non_zero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0) :
  verify_non_zero_array x y z w ↔ 
  (x = -1 ∨ y = -2 ∨ z = -3 ∨ w = -4) ∧
  (if x = -1 then y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if y = -2 then x ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 else 
   if z = -3 then x ≠ 0 ∧ y ≠ 0 ∧ w ≠ 0 else 
   x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :=
sorry

end solution_contains_non_zero_arrays_l90_90162


namespace closest_point_l90_90165

noncomputable def point_on_line_closest_to (x y : ℝ) : ℝ × ℝ :=
( -11 / 5, 7 / 5 )

theorem closest_point (x y : ℝ) (h_line : y = 2 * x + 3) (h_point : (x, y) = (3, -4)) :
  point_on_line_closest_to x y = ( -11 / 5, 7 / 5 ) :=
sorry

end closest_point_l90_90165


namespace cistern_length_l90_90091

theorem cistern_length (w d A : ℝ) (h : d = 1.25 ∧ w = 4 ∧ A = 68.5) :
  ∃ L : ℝ, (L * w) + (2 * L * d) + (2 * w * d) = A ∧ L = 9 :=
by
  obtain ⟨h_d, h_w, h_A⟩ := h
  use 9
  simp [h_d, h_w, h_A]
  norm_num
  sorry

end cistern_length_l90_90091


namespace pastry_problem_minimum_n_l90_90293

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l90_90293


namespace liams_numbers_l90_90038

theorem liams_numbers (x y : ℤ) 
  (h1 : 3 * x + 2 * y = 75)
  (h2 : x = 15)
  (h3 : ∃ k : ℕ, x * y = 5 * k) : 
  y = 15 := 
by
  sorry

end liams_numbers_l90_90038


namespace four_integers_sum_6_7_8_9_l90_90163

theorem four_integers_sum_6_7_8_9 (a b c d : ℕ)
  (h1 : a + b + c = 6) 
  (h2 : a + b + d = 7) 
  (h3 : a + c + d = 8) 
  (h4 : b + c + d = 9) :
  (a = 1) ∧ (b = 2) ∧ (c = 3) ∧ (d = 4) := 
by 
  sorry

end four_integers_sum_6_7_8_9_l90_90163


namespace product_fraction_l90_90935

noncomputable def a (n : ℕ) : ℤ := int.floor (real.sqrt (real.sqrt (2 * n - 1)))
noncomputable def b (n : ℕ) : ℤ := int.floor (real.sqrt (real.sqrt (2 * n)))
noncomputable def prod_a (N : ℕ) : ℤ := (finset.range N).prod (λ n, a (n + 1))
noncomputable def prod_b (N : ℕ) : ℤ := (finset.range N).prod (λ n, b (n + 1))

theorem product_fraction:
  (prod_a 1024 : ℚ) / prod_b 1024 = 105 / 384 :=
by
  sorry

end product_fraction_l90_90935


namespace selena_ran_24_miles_l90_90047

theorem selena_ran_24_miles (S J : ℝ) (h1 : S + J = 36) (h2 : J = S / 2) : S = 24 := 
sorry

end selena_ran_24_miles_l90_90047


namespace utility_bill_amount_l90_90854

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l90_90854


namespace twice_x_minus_3_l90_90944

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l90_90944


namespace exists_negative_number_satisfying_inequality_l90_90067

theorem exists_negative_number_satisfying_inequality :
  ∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0 :=
sorry

end exists_negative_number_satisfying_inequality_l90_90067


namespace min_balloon_count_l90_90074

theorem min_balloon_count 
(R B : ℕ) (burst_red burst_blue : ℕ) 
(h1 : R = 7 * B) 
(h2 : burst_red = burst_blue / 3) 
(h3 : burst_red ≥ 1) :
R + B = 24 :=
by 
    sorry

end min_balloon_count_l90_90074


namespace simplify_and_evaluate_expression_l90_90551

theorem simplify_and_evaluate_expression 
  (x y : ℤ) (hx : x = -3) (hy : y = -2) :
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l90_90551


namespace first_wing_hall_rooms_l90_90393

theorem first_wing_hall_rooms
    (total_rooms : ℕ) (first_wing_floors : ℕ) (first_wing_halls_per_floor : ℕ)
    (second_wing_floors : ℕ) (second_wing_halls_per_floor : ℕ) (second_wing_rooms_per_hall : ℕ)
    (hotel_total_rooms : ℕ) (first_wing_total_rooms : ℕ) :
    hotel_total_rooms = total_rooms →
    first_wing_floors = 9 →
    first_wing_halls_per_floor = 6 →
    second_wing_floors = 7 →
    second_wing_halls_per_floor = 9 →
    second_wing_rooms_per_hall = 40 →
    hotel_total_rooms = first_wing_total_rooms + (second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall) →
    first_wing_total_rooms = first_wing_floors * first_wing_halls_per_floor * 32 :=
by
  sorry

end first_wing_hall_rooms_l90_90393


namespace kyle_caught_fish_l90_90475

def total_fish := 36
def fish_carla := 8
def fish_total := total_fish - fish_carla

-- kelle and tasha same number of fish means they equally divide the total fish left after deducting carla's
def fish_each_kt := fish_total / 2

theorem kyle_caught_fish :
  fish_each_kt = 14 :=
by
  -- Placeholder for the proof
  sorry

end kyle_caught_fish_l90_90475


namespace exponentiation_rule_l90_90319

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l90_90319


namespace floor_exponents_eq_l90_90700

theorem floor_exponents_eq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_inf_k : ∃ᶠ k in at_top, ∃ (k : ℕ), ⌊a ^ k⌋ + ⌊b ^ k⌋ = ⌊a⌋ ^ k + ⌊b⌋ ^ k) :
  ⌊a ^ 2014⌋ + ⌊b ^ 2014⌋ = ⌊a⌋ ^ 2014 + ⌊b⌋ ^ 2014 := by
  sorry

end floor_exponents_eq_l90_90700


namespace part1_part2_l90_90602

def my_mul (x y : Int) : Int :=
  if x = 0 then abs y
  else if y = 0 then abs x
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then abs x + abs y
  else - (abs x + abs y)

theorem part1 : my_mul (-15) (my_mul 3 0) = -18 := 
  by
  sorry

theorem part2 (a : Int) : 
  my_mul 3 a + a = 
  if a < 0 then 2 * a - 3 
  else if a = 0 then 3
  else 2 * a + 3 :=
  by
  sorry

end part1_part2_l90_90602


namespace range_of_a_l90_90371

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := 
sorry

end range_of_a_l90_90371


namespace album_ways_10_l90_90617

noncomputable def total_album_ways : ℕ := 
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := 4
  let friends := 4
  ((total_albums.choose photo_albums) * (total_albums - photo_albums).choose stamp_albums) / friends

theorem album_ways_10 :
  total_album_ways = 10 := 
by sorry

end album_ways_10_l90_90617


namespace conditions_for_a_and_b_l90_90600

variables (a b x y : ℝ)

theorem conditions_for_a_and_b (h1 : x^2 + x * y + y^2 - y = 0) (h2 : a * x^2 + b * x * y + x = 0) :
  (a + 1)^2 = 4 * (b + 1) ∧ b ≠ -1 :=
sorry

end conditions_for_a_and_b_l90_90600


namespace paul_money_duration_l90_90543

theorem paul_money_duration (earn1 earn2 spend : ℕ) (h1 : earn1 = 3) (h2 : earn2 = 3) (h_spend : spend = 3) : 
  (earn1 + earn2) / spend = 2 :=
by
  sorry

end paul_money_duration_l90_90543


namespace tony_solving_puzzles_time_l90_90419

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l90_90419


namespace count_even_sum_subsets_l90_90648

open Finset

-- Given conditions
def even_sum_subset_property (s : Finset ℕ) :=
  ∃ a b c d, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ a ≠ b ∧ c ≠ d ∧ a + b = 16 ∧ c + d = 24

-- Main theorem statement
theorem count_even_sum_subsets :
  (univ.filter even_sum_subset_property).card = 210 :=
sorry

end count_even_sum_subsets_l90_90648


namespace div_1959_l90_90710

theorem div_1959 (n : ℕ) : ∃ k : ℤ, 5^(8 * n) - 2^(4 * n) * 7^(2 * n) = k * 1959 := 
by 
  sorry

end div_1959_l90_90710


namespace pastry_trick_l90_90284

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l90_90284


namespace original_number_of_professors_l90_90029

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l90_90029


namespace coplanar_AD_eq_linear_combination_l90_90807

-- Define the points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, 3, 1⟩
def C : Point3D := ⟨3, 7, -5⟩
def D : Point3D := ⟨11, -1, 3⟩

-- Define the vectors
def vector (P Q : Point3D) : Point3D := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector A B
def AC := vector A C
def AD := vector A D

-- Coplanar definition: AD = λ AB + μ AC
theorem coplanar_AD_eq_linear_combination (lambda mu : ℝ) :
  AD = ⟨lambda * 2 + mu * (-1), lambda * (-2) + mu * 6, lambda * (-2) + mu * (-8)⟩ :=
sorry

end coplanar_AD_eq_linear_combination_l90_90807


namespace boat_distance_downstream_is_68_l90_90762

variable (boat_speed : ℕ) (stream_speed : ℕ) (time_hours : ℕ)

-- Given conditions
def effective_speed_downstream (boat_speed stream_speed : ℕ) : ℕ := boat_speed + stream_speed
def distance_downstream (speed time : ℕ) : ℕ := speed * time

theorem boat_distance_downstream_is_68 
  (h1 : boat_speed = 13) 
  (h2 : stream_speed = 4) 
  (h3 : time_hours = 4) : 
  distance_downstream (effective_speed_downstream boat_speed stream_speed) time_hours = 68 := 
by 
  sorry

end boat_distance_downstream_is_68_l90_90762


namespace tony_additional_degrees_l90_90886

-- Definitions for the conditions
def total_years : ℕ := 14
def science_degree_years : ℕ := 4
def physics_degree_years : ℕ := 2
def additional_degree_years : ℤ := total_years - (science_degree_years + physics_degree_years)
def each_additional_degree_years : ℕ := 4
def additional_degrees : ℤ := additional_degree_years / each_additional_degree_years

-- Theorem stating the problem and the answer
theorem tony_additional_degrees : additional_degrees = 2 :=
 by
     sorry

end tony_additional_degrees_l90_90886


namespace math_problem_l90_90948

-- Define the mixed numbers as fractions
def mixed_3_1_5 := 16 / 5 -- 3 + 1/5 = 16/5
def mixed_4_1_2 := 9 / 2  -- 4 + 1/2 = 9/2
def mixed_2_3_4 := 11 / 4 -- 2 + 3/4 = 11/4
def mixed_1_2_3 := 5 / 3  -- 1 + 2/3 = 5/3

-- Define the main expression
def main_expr := 53 * (mixed_3_1_5 - mixed_4_1_2) / (mixed_2_3_4 + mixed_1_2_3)

-- Define the expected answer in its fractional form
def expected_result := -78 / 5

-- The theorem to prove the main expression equals the expected mixed number
theorem math_problem : main_expr = expected_result :=
by sorry

end math_problem_l90_90948


namespace ptolemys_inequality_l90_90999

variable {A B C D : Type} [OrderedRing A]
variable (AB BC CD DA AC BD : A)

/-- Ptolemy's inequality for a quadrilateral -/
theorem ptolemys_inequality 
  (AB_ BC_ CD_ DA_ AC_ BD_ : A) :
  AC * BD ≤ AB * CD + BC * AD :=
  sorry

end ptolemys_inequality_l90_90999


namespace solution_set_of_inequality_l90_90577

open Set

theorem solution_set_of_inequality :
  {x : ℝ | - x ^ 2 - 4 * x + 5 > 0} = {x : ℝ | -5 < x ∧ x < 1} :=
sorry

end solution_set_of_inequality_l90_90577


namespace true_statement_l90_90746

theorem true_statement :
  -8 < -2 := 
sorry

end true_statement_l90_90746


namespace point_P_in_second_quadrant_l90_90544

-- Define what it means for a point to lie in a certain quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- The coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- Prove that the point P is in the second quadrant
theorem point_P_in_second_quadrant : in_second_quadrant (point_P.1) (point_P.2) :=
by
  sorry

end point_P_in_second_quadrant_l90_90544


namespace trigonometric_identity_proof_l90_90445

variable (α β : Real)

theorem trigonometric_identity_proof :
  4.28 * Real.sin (β / 2 - Real.pi / 2) ^ 2 - Real.cos (α - 3 * Real.pi / 2) ^ 2 = 
  Real.cos (α + β) * Real.cos (α - β) :=
by
  sorry

end trigonometric_identity_proof_l90_90445


namespace range_of_a_l90_90355

variable {R : Type} [LinearOrderedField R]

def is_even (f : R → R) : Prop := ∀ x, f x = f (-x)
def is_monotone_increasing_on_non_neg (f : R → R) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem range_of_a 
  (f : R → R) 
  (even_f : is_even f)
  (mono_f : is_monotone_increasing_on_non_neg f)
  (ineq : ∀ a, f (a + 1) ≤ f 4) : 
  ∀ a, -5 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l90_90355


namespace kamal_chemistry_marks_l90_90698

-- Definitions of the marks
def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def num_subjects : ℕ := 5

-- Statement to be proved
theorem kamal_chemistry_marks : ∃ (chemistry_marks : ℕ), 
  76 + 60 + 72 + 82 + chemistry_marks = 71 * 5 :=
by
sorry

end kamal_chemistry_marks_l90_90698


namespace value_of_x_l90_90556

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 5)

noncomputable def f_inv (y : ℝ) : ℝ := sorry -- Placeholder for the inverse of f

noncomputable def g (x : ℝ) : ℝ := 3 * f_inv x

theorem value_of_x (h : g 18 = 18) : x = 30 / 11 :=
by
  -- Proof is not required.
  sorry

end value_of_x_l90_90556


namespace rhombus_longer_diagonal_l90_90918

theorem rhombus_longer_diagonal (a b : ℝ)
  (side_length : a = 60)
  (shorter_diagonal : b = 56) :
  ∃ d : ℝ, d = 32 * Real.sqrt 11 :=
by
  let half_shorter_diagonal := b / 2
  have a_squared := a * a
  have b_squared := half_shorter_diagonal * half_shorter_diagonal

  let half_longer_diagonal := Real.sqrt (a_squared - b_squared)
  let longer_diagonal := 2 * half_longer_diagonal

  have longer_diagonal_squared : longer_diagonal * longer_diagonal = ((2 * half_longer_diagonal) * (2 * half_longer_diagonal)) := by sorry
 
  use 32 * Real.sqrt 11
  rw [← longer_diagonal_squared]
  sorry

end rhombus_longer_diagonal_l90_90918


namespace solution_set_of_inequality_l90_90880

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l90_90880


namespace exponent_multiplication_l90_90679

theorem exponent_multiplication (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 4) :
  a^(m + n) = 8 := by
  sorry

end exponent_multiplication_l90_90679


namespace smallest_n_for_good_sequence_l90_90773

def is_good_sequence (a : ℕ → ℝ) : Prop :=
   (∃ (a_0 : ℕ), a 0 = a_0) ∧
   (∀ i : ℕ, a (i+1) = 2 * a i + 1 ∨ a (i+1) = a i / (a i + 2)) ∧
   (∃ k : ℕ, a k = 2014)

theorem smallest_n_for_good_sequence : 
  ∀ (a : ℕ → ℝ), is_good_sequence a → ∃ n : ℕ, a n = 2014 ∧ ∀ m : ℕ, m < n → a m ≠ 2014 :=
sorry

end smallest_n_for_good_sequence_l90_90773


namespace camila_weeks_needed_l90_90322

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l90_90322


namespace symmetric_line_correct_l90_90058

-- Define the given line equation
def line_eq (x y : ℝ) := 3 * x + 4 * y = 2

-- Define the symmetry line equation
def symmetry_line (x y : ℝ) := y = x

-- Define the symmetric line transformation
def symmetric_line_eq (x y : ℝ) := line_eq y x

-- Define the target equation form for the symmetric line
def target_eq (x y : ℝ) := 4 * x + 3 * y - 2 = 0

-- Prove that the transformed equation is indeed the target equation
theorem symmetric_line_correct (x y : ℝ) : symmetric_line_eq x y ↔ target_eq x y :=
by
  unfold symmetric_line_eq
  unfold line_eq
  unfold target_eq
  sorry

end symmetric_line_correct_l90_90058


namespace second_meeting_time_l90_90991

-- Given conditions and constants.
def pool_length : ℕ := 120
def initial_george_distance : ℕ := 80
def initial_henry_distance : ℕ := 40
def george_speed (t : ℕ) : ℕ := initial_george_distance / t
def henry_speed (t : ℕ) : ℕ := initial_henry_distance / t

-- Main statement to prove the question and answer.
theorem second_meeting_time (t : ℕ) (h_t_pos : t > 0) : 
  5 * t = 15 / 2 :=
sorry

end second_meeting_time_l90_90991


namespace num_people_watched_last_week_l90_90691

variable (s f t : ℕ)
variable (h1 : s = 80)
variable (h2 : f = s - 20)
variable (h3 : t = s + 15)
variable (total_last_week total_this_week : ℕ)
variable (h4 : total_this_week = f + s + t)
variable (h5 : total_this_week = total_last_week + 35)

theorem num_people_watched_last_week :
  total_last_week = 200 := sorry

end num_people_watched_last_week_l90_90691


namespace number_of_students_before_new_year_l90_90754

variables (M N k ℓ : ℕ)
hypotheses (h1 : 100 * M = k * N)
             (h2 : 100 * (M + 1) = ℓ * (N + 3))
             (h3 : ℓ < 100)

theorem number_of_students_before_new_year (h1 : 100 * M = k * N)
                                             (h2 : 100 * (M + 1) = ℓ * (N + 3))
                                             (h3 : ℓ < 100) :
  N ≤ 197 :=
sorry

end number_of_students_before_new_year_l90_90754


namespace train_bus_ratio_is_two_thirds_l90_90099

def total_distance : ℕ := 1800
def distance_by_plane : ℕ := total_distance / 3
def distance_by_bus : ℕ := 720
def distance_by_train : ℕ := total_distance - (distance_by_plane + distance_by_bus)
def train_to_bus_ratio : ℚ := distance_by_train / distance_by_bus

theorem train_bus_ratio_is_two_thirds :
  train_to_bus_ratio = 2 / 3 := by
  sorry

end train_bus_ratio_is_two_thirds_l90_90099


namespace most_stable_scores_l90_90277

-- Definitions for the variances of students A, B, and C
def s_A_2 : ℝ := 6
def s_B_2 : ℝ := 24
def s_C_2 : ℝ := 50

-- The proof that student A has the most stable math scores
theorem most_stable_scores : 
  s_A_2 < s_B_2 ∧ s_B_2 < s_C_2 → 
  ("Student A has the most stable scores" = "Student A has the most stable scores") :=
by
  intros h
  sorry

end most_stable_scores_l90_90277


namespace matrix_C_power_50_l90_90203

open Matrix

theorem matrix_C_power_50 (C : Matrix (Fin 2) (Fin 2) ℤ) 
  (hC : C = !![3, 2; -8, -5]) : 
  C^50 = !![1, 0; 0, 1] :=
by {
  -- External proof omitted.
  sorry
}

end matrix_C_power_50_l90_90203


namespace quadratic_has_real_roots_range_l90_90687

noncomputable def has_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := 2
  let c := -1
  b^2 - 4 * a * c ≥ 0

theorem quadratic_has_real_roots_range (k : ℝ) :
  has_real_roots k ↔ k ≥ -1 ∧ k ≠ 0 := by
sorry

end quadratic_has_real_roots_range_l90_90687


namespace find_angle_C_l90_90199

theorem find_angle_C 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 10 * a * Real.cos B = 3 * b * Real.cos A) 
  (h2 : Real.cos A = (5 * Real.sqrt 26) / 26) 
  (h3 : A + B + C = π) : 
  C = (3 * π) / 4 :=
sorry

end find_angle_C_l90_90199


namespace remainder_mod_105_l90_90870

theorem remainder_mod_105 (x : ℤ) 
  (h1 : 3 + x ≡ 4 [ZMOD 27])
  (h2 : 5 + x ≡ 9 [ZMOD 125])
  (h3 : 7 + x ≡ 25 [ZMOD 343]) :
  x % 105 = 4 :=
  sorry

end remainder_mod_105_l90_90870


namespace laboratory_spent_on_flasks_l90_90455

theorem laboratory_spent_on_flasks:
  ∀ (F : ℝ), (∃ cost_test_tubes : ℝ, cost_test_tubes = (2 / 3) * F) →
  (∃ cost_safety_gear : ℝ, cost_safety_gear = (1 / 3) * F) →
  2 * F = 300 → F = 150 :=
by
  intros F h1 h2 h3
  sorry

end laboratory_spent_on_flasks_l90_90455


namespace equation_represents_hyperbola_l90_90791

theorem equation_represents_hyperbola (x y : ℝ) :
  x^2 - 4*y^2 - 2*x + 8*y - 8 = 0 → ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (a * (x - h)^2 - b * (y - k)^2 = 1) := 
sorry

end equation_represents_hyperbola_l90_90791


namespace cats_in_village_l90_90618

theorem cats_in_village (C : ℕ) (h1 : 1 / 3 * C = (1 / 4) * (1 / 3) * C)
  (h2 : (1 / 12) * C = 10) : C = 120 :=
sorry

end cats_in_village_l90_90618


namespace sum_of_first_five_primes_with_units_digit_3_l90_90978

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90978


namespace no_valid_placement_of_prisms_l90_90382

-- Definitions: Rectangular prism with edges parallel to OX, OY, and OZ axes.
structure RectPrism :=
  (x_interval : Set ℝ)
  (y_interval : Set ℝ)
  (z_interval : Set ℝ)

-- Function to determine if two rectangular prisms intersect.
def intersects (P Q : RectPrism) : Prop :=
  ¬ Disjoint P.x_interval Q.x_interval ∧
  ¬ Disjoint P.y_interval Q.y_interval ∧
  ¬ Disjoint P.z_interval Q.z_interval

-- Definition of the 12 rectangular prisms
def prisms := Fin 12 → RectPrism

-- Conditions for intersection:
def intersection_condition (prisms : prisms) : Prop :=
  ∀ i : Fin 12, ∀ j : Fin 12,
    (j = (i + 1) % 12) ∨ (j = (i - 1 + 12) % 12) ∨ intersects (prisms i) (prisms j)

theorem no_valid_placement_of_prisms :
  ¬ ∃ (prisms : prisms), intersection_condition prisms :=
sorry

end no_valid_placement_of_prisms_l90_90382


namespace find_three_digit_number_l90_90338

-- Definitions of digit constraints and the number representation
def is_three_digit_number (N : ℕ) (a b c : ℕ) : Prop :=
  N = 100 * a + 10 * b + c ∧ 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9

-- Definition of the problem condition
def sum_of_digits_condition (N : ℕ) (a b c : ℕ) : Prop :=
  a + b + c = N / 11

-- Lean theorem statement
theorem find_three_digit_number (N a b c : ℕ) :
  is_three_digit_number N a b c ∧ sum_of_digits_condition N a b c → N = 198 :=
by
  sorry

end find_three_digit_number_l90_90338


namespace quadratic_roots_l90_90240

theorem quadratic_roots (x : ℝ) : x^2 + 4 * x + 3 = 0 → x = -3 ∨ x = -1 :=
by
  intro h
  have h1 : (x + 3) * (x + 1) = 0 := by sorry
  have h2 : (x = -3 ∨ x = -1) := by sorry
  exact h2

end quadratic_roots_l90_90240


namespace last_digit_of_189_in_base_3_is_0_l90_90144

theorem last_digit_of_189_in_base_3_is_0 : 
  (189 % 3 = 0) :=
sorry

end last_digit_of_189_in_base_3_is_0_l90_90144


namespace incorrect_assignment_statement_l90_90082

theorem incorrect_assignment_statement :
  ∀ (a x y : ℕ), ¬(x * y = a) := by
sorry

end incorrect_assignment_statement_l90_90082


namespace nontrivial_solution_exists_l90_90145

theorem nontrivial_solution_exists
  (a b c : ℝ) :
  (∃ x y z : ℝ, (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ 
    a * x + b * y + c * z = 0 ∧ 
    b * x + c * y + a * z = 0 ∧ 
    c * x + a * y + b * z = 0) ↔ (a + b + c = 0 ∨ a = b ∧ b = c) := 
sorry

end nontrivial_solution_exists_l90_90145


namespace trick_proof_l90_90288

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l90_90288


namespace solution_set_of_inequalities_l90_90663

theorem solution_set_of_inequalities (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : ∀ x, mx + n > 0 ↔ x < (1/3)) : ∀ x, nx - m < 0 ↔ x < -3 :=
by
  sorry

end solution_set_of_inequalities_l90_90663


namespace range_of_a_l90_90988

/-- 
For the system of inequalities in terms of x 
    \begin{cases} 
    x - a < 0 
    ax < 1 
    \end{cases}
the range of values for the real number a such that the solution set is not empty is [-1, ∞).
-/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - a < 0 ∧ a * x < 1) ↔ -1 ≤ a :=
by sorry

end range_of_a_l90_90988


namespace mask_digit_identification_l90_90228

theorem mask_digit_identification :
  ∃ (elephant_mask mouse_mask pig_mask panda_mask : ℕ),
    (4 * 4 = 16) ∧
    (7 * 7 = 49) ∧
    (8 * 8 = 64) ∧
    (9 * 9 = 81) ∧
    elephant_mask = 6 ∧
    mouse_mask = 4 ∧
    pig_mask = 8 ∧
    panda_mask = 1 :=
by
  sorry

end mask_digit_identification_l90_90228


namespace storyteller_friends_house_number_l90_90562

theorem storyteller_friends_house_number
  (x y : ℕ)
  (htotal : 50 < x ∧ x < 500)
  (hsum : 2 * y = x * (x + 1)) :
  y = 204 :=
by
  sorry

end storyteller_friends_house_number_l90_90562


namespace integral_represents_half_volume_of_sphere_l90_90722

theorem integral_represents_half_volume_of_sphere :
  π * ∫ x in (0 : ℝ)..1, (1 - x^2) = (2 / 3) * π :=
by
  sorry

end integral_represents_half_volume_of_sphere_l90_90722


namespace restaurant_customers_prediction_l90_90451

def total_customers_saturday (breakfast_customers_friday lunch_customers_friday dinner_customers_friday : ℝ) : ℝ :=
  let breakfast_customers_saturday := 2 * breakfast_customers_friday
  let lunch_customers_saturday := lunch_customers_friday + 0.25 * lunch_customers_friday
  let dinner_customers_saturday := dinner_customers_friday - 0.15 * dinner_customers_friday
  breakfast_customers_saturday + lunch_customers_saturday + dinner_customers_saturday

theorem restaurant_customers_prediction :
  let breakfast_customers_friday := 73
  let lunch_customers_friday := 127
  let dinner_customers_friday := 87
  total_customers_saturday breakfast_customers_friday lunch_customers_friday dinner_customers_friday = 379 := 
by
  sorry

end restaurant_customers_prediction_l90_90451


namespace german_team_goals_possible_goal_values_l90_90108

def german_team_goal_conditions (x : ℕ) :=
  (10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ∧ (x % 2 = 1)

def exactly_two_journalists_correct (x : ℕ) :=
  (10 < x ∧ x < 17) ∨ (11 < x ∧ x < 18) ∨ (x % 2 = 0)

theorem german_team_goals : ∀ (x : ℕ), german_team_goal_conditions x → exactly_two_journalists_correct x :=
begin
  intros x h_goal_conditions,
  -- This part is meant for proving the statement that exactly two of the journalistic statements are true.
  sorry
end

def possible_goals : set ℕ := { x | x ∈ {11, 12, 14, 16, 17} }

theorem possible_goal_values : ∀ (x : ℕ), x ∈ possible_goals :=
begin
  -- Proof that given the conditions, the possible goals must lie within the set {11, 12, 14, 16, 17}.
  sorry
end

end german_team_goals_possible_goal_values_l90_90108


namespace alice_safe_paths_l90_90779

/-
Define the coordinate system and conditions.
-/

def total_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

def paths_through_dangerous_area : ℕ :=
  (total_paths 2 2) * (total_paths 2 1)

def safe_paths : ℕ :=
  total_paths 4 3 - paths_through_dangerous_area

theorem alice_safe_paths : safe_paths = 17 := by
  sorry

end alice_safe_paths_l90_90779


namespace number_of_paths_l90_90677

theorem number_of_paths (n m : ℕ) (h : m ≤ n) : 
  ∃ paths : ℕ, paths = Nat.choose n m := 
sorry

end number_of_paths_l90_90677


namespace expected_value_X_correct_prob_1_red_ball_B_correct_l90_90002

-- Boxes configuration
structure BoxConfig where
  white_A : ℕ -- Number of white balls in box A
  red_A : ℕ -- Number of red balls in box A
  white_B : ℕ -- Number of white balls in box B
  red_B : ℕ -- Number of red balls in box B

-- Given the problem configuration
def initialConfig : BoxConfig := {
  white_A := 2,
  red_A := 2,
  white_B := 1,
  red_B := 3,
}

-- Define random variable X (number of red balls drawn from box A)
def prob_X (X : ℕ) (cfg : BoxConfig) : ℚ :=
  if X = 0 then 1 / 6
  else if X = 1 then 2 / 3
  else if X = 2 then 1 / 6
  else 0

-- Expected value of X
noncomputable def expected_value_X (cfg : BoxConfig) : ℚ :=
  0 * (prob_X 0 cfg) + 1 * (prob_X 1 cfg) + 2 * (prob_X 2 cfg)

-- Probability of drawing 1 red ball from box B
noncomputable def prob_1_red_ball_B (cfg : BoxConfig) (X : ℕ) : ℚ :=
  if X = 0 then 1 / 2
  else if X = 1 then 2 / 3
  else if X = 2 then 5 / 6
  else 0

-- Total probability of drawing 1 red ball from box B
noncomputable def total_prob_1_red_ball_B (cfg : BoxConfig) : ℚ :=
  (prob_X 0 cfg * (prob_1_red_ball_B cfg 0))
  + (prob_X 1 cfg * (prob_1_red_ball_B cfg 1))
  + (prob_X 2 cfg * (prob_1_red_ball_B cfg 2))


theorem expected_value_X_correct : expected_value_X initialConfig = 1 := by
  sorry

theorem prob_1_red_ball_B_correct : total_prob_1_red_ball_B initialConfig = 2 / 3 := by
  sorry

end expected_value_X_correct_prob_1_red_ball_B_correct_l90_90002


namespace find_f_at_6_l90_90768

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2

theorem find_f_at_6 (f : ℝ → ℝ) (h : example_function f) : f 6 = 4 := 
by
  sorry

end find_f_at_6_l90_90768


namespace profit_per_meter_l90_90462

theorem profit_per_meter (number_of_meters : ℕ) (total_selling_price cost_price_per_meter : ℝ) 
  (h1 : number_of_meters = 85) 
  (h2 : total_selling_price = 8925) 
  (h3 : cost_price_per_meter = 90) :
  (total_selling_price - cost_price_per_meter * number_of_meters) / number_of_meters = 15 :=
  sorry

end profit_per_meter_l90_90462


namespace original_decimal_l90_90107

theorem original_decimal (x : ℝ) (h : 1000 * x / 100 = 12.5) : x = 1.25 :=
by
  sorry

end original_decimal_l90_90107


namespace pastry_problem_minimum_n_l90_90292

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l90_90292


namespace tony_total_puzzle_time_l90_90422

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l90_90422


namespace triangle_count_l90_90506

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangles : List (ℕ × ℕ × ℕ) :=
  List.filter (λ (t : ℕ × ℕ × ℕ), let (a, b, c) := t in is_triangle a b c)
    [(a, b, c) | a ← List.range 10, b ← List.range 10, c ← List.range 10, a + b + c = 9]

theorem triangle_count : valid_triangles.length = 12 := by
  sorry

end triangle_count_l90_90506


namespace max_a2b3c4_l90_90702

noncomputable def maximum_value (a b c : ℝ) : ℝ := a^2 * b^3 * c^4

theorem max_a2b3c4 (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  maximum_value a b c ≤ 143327232 / 386989855 := sorry

end max_a2b3c4_l90_90702


namespace well_filled_ways_1_5_l90_90227

-- Define a structure for representing the conditions of the figure filled with integers
structure WellFilledFigure where
  top_circle : ℕ
  shaded_circle_possibilities : Finset ℕ
  sub_diagram_possibilities : ℕ

-- Define an example of this structure corresponding to our problem
def figure1_5 : WellFilledFigure :=
  { top_circle := 5,
    shaded_circle_possibilities := {1, 2, 3, 4},
    sub_diagram_possibilities := 2 }

-- Define the theorem statement
theorem well_filled_ways_1_5 (f : WellFilledFigure) : (f.top_circle = 5) → 
  (f.shaded_circle_possibilities.card = 4) → 
  (f.sub_diagram_possibilities = 2) → 
  (4 * 2 = 8) := by
  sorry

end well_filled_ways_1_5_l90_90227


namespace solution_set_of_inequality_l90_90730

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 ≥ 0 } = { x : ℝ | x ≤ -2 ∨ 1 ≤ x } :=
by
  sorry

end solution_set_of_inequality_l90_90730


namespace find_angle_A_find_minimum_bc_l90_90519

open Real

variables (A B C a b c : ℝ)

-- Conditions
def side_opposite_angles_condition : Prop :=
  A > 0 ∧ A < π ∧ (A + B + C) = π

def collinear_vectors_condition (B C : ℝ) : Prop :=
  ∃ (k : ℝ), (2 * cos B * cos C + 1, 2 * sin B) = k • (sin C, 1)

-- Questions translated to proof statements
theorem find_angle_A (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C) :
  A = π / 3 :=
sorry

theorem find_minimum_bc (h1 : side_opposite_angles_condition A B C) (h2 : collinear_vectors_condition B C)
  (h3 : (1 / 2) * b * c * sin A = sqrt 3) :
  b + c = 4 :=
sorry

end find_angle_A_find_minimum_bc_l90_90519


namespace john_subtracts_79_l90_90738

theorem john_subtracts_79 :
  let a := 40
  let b := 1
  let n := (a - b) * (a - b)
  n = a * a - 79
:= by
  sorry

end john_subtracts_79_l90_90738


namespace product_4_6_7_14_l90_90589

theorem product_4_6_7_14 : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end product_4_6_7_14_l90_90589


namespace triangle_proof_l90_90720

variables (α β γ a b c : ℝ)

-- Definitions based on the given conditions
def angle_relation (α β : ℝ) : Prop := 3 * α + 2 * β = 180
def triangle_angle_sum (α β γ : ℝ) : Prop := α + β + γ = 180

-- Lean statement for the proof problem
theorem triangle_proof
  (h1 : angle_relation α β)
  (h2 : triangle_angle_sum α β γ) :
  a^2 + b * c = c^2 :=
sorry

end triangle_proof_l90_90720


namespace solve_system_eq_l90_90716

theorem solve_system_eq (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) → 
  ( ∃ t : ℝ, (x = (1 + t) * b) ∧ (y = (1 - t) * b) ∧ (z = 0) ∧ t^2 = -1/2 ) :=
by
  -- proof will be filled in here
  sorry

end solve_system_eq_l90_90716


namespace initial_amount_l90_90134

variable (X : ℝ)

/--
An individual deposited 20% of 25% of 30% of their initial amount into their bank account.
If the deposited amount is Rs. 750, prove that their initial amount was Rs. 50000.
-/
theorem initial_amount (h : (0.2 * 0.25 * 0.3 * X) = 750) : X = 50000 :=
by
  sorry

end initial_amount_l90_90134


namespace mark_sprint_distance_l90_90210

theorem mark_sprint_distance (t v : ℝ) (ht : t = 24.0) (hv : v = 6.0) : 
  t * v = 144.0 := 
by
  -- This theorem is formulated with the conditions that t = 24.0 and v = 6.0,
  -- we need to prove that the resulting distance is 144.0 miles.
  sorry

end mark_sprint_distance_l90_90210


namespace utility_bills_total_correct_l90_90859

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l90_90859


namespace complex_number_calculation_l90_90810

theorem complex_number_calculation (i : ℂ) (h : i * i = -1) : i^7 - 2/i = i := 
by 
  sorry

end complex_number_calculation_l90_90810


namespace total_water_in_containers_l90_90585

/-
We have four containers. The first three contain water, while the fourth is empty. 
The second container holds twice as much water as the first, and the third holds twice as much water as the second. 
We transfer half of the water from the first container, one-third of the water from the second container, 
and one-quarter of the water from the third container into the fourth container. 
Now, there are 26 liters of water in the fourth container. Prove that initially, 
there were 84 liters of water in total in the first three containers.
-/

theorem total_water_in_containers (x : ℕ) (h1 : x / 2 + 2 * x / 3 + x = 26) : x + 2 * x + 4 * x = 84 := 
sorry

end total_water_in_containers_l90_90585


namespace professors_initial_count_l90_90017

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l90_90017


namespace utility_bills_total_l90_90851

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l90_90851


namespace number_of_positive_integers_l90_90652

theorem number_of_positive_integers (n : ℕ) : 
  (0 < n ∧ n < 36 ∧ (∃ k : ℕ, n = k * (36 - k))) → 
  n = 18 ∨ n = 24 ∨ n = 30 ∨ n = 32 ∨ n = 34 ∨ n = 35 :=
sorry

end number_of_positive_integers_l90_90652


namespace sufficient_not_necessary_for_ellipse_l90_90235

-- Define the conditions
def positive_denominator_m (m : ℝ) : Prop := m > 0
def positive_denominator_2m_minus_1 (m : ℝ) : Prop := 2 * m - 1 > 0
def denominators_not_equal (m : ℝ) : Prop := m ≠ 1

-- Define the question
def is_ellipse_condition (m : ℝ) : Prop := m > 1

-- The main theorem
theorem sufficient_not_necessary_for_ellipse (m : ℝ) :
  positive_denominator_m m ∧ positive_denominator_2m_minus_1 m ∧ denominators_not_equal m → is_ellipse_condition m :=
by
  -- Proof omitted
  sorry

end sufficient_not_necessary_for_ellipse_l90_90235


namespace range_of_a_l90_90671

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) := x^3 - x^2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (1/2) 2 → x2 ∈ Set.Icc (1/2) 2 → f x1 a - g x2 ≥ 0) →
  1 ≤ a :=
by
  sorry

end range_of_a_l90_90671


namespace knife_value_l90_90582

def sheep_sold (n : ℕ) : ℕ := n * n

def valid_units_digits (m : ℕ) : Bool :=
  (m ^ 2 = 16) ∨ (m ^ 2 = 36)

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) (H1 : sheep_sold n = n * n) (H2 : n = 10 * k + m) (H3 : valid_units_digits m = true) :
  2 = 2 :=
by
  sorry

end knife_value_l90_90582


namespace bouncy_balls_total_l90_90706

theorem bouncy_balls_total :
  let red_packs := 6
  let red_per_pack := 12
  let yellow_packs := 10
  let yellow_per_pack := 8
  let green_packs := 4
  let green_per_pack := 15
  let blue_packs := 3
  let blue_per_pack := 20
  let red_balls := red_packs * red_per_pack
  let yellow_balls := yellow_packs * yellow_per_pack
  let green_balls := green_packs * green_per_pack
  let blue_balls := blue_packs * blue_per_pack
  red_balls + yellow_balls + green_balls + blue_balls = 272 := 
by
  sorry

end bouncy_balls_total_l90_90706


namespace square_vertex_distance_l90_90922

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32
noncomputable def greatest_distance : ℝ := 7 * Real.sqrt 2

theorem square_vertex_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  let inner_diagonal := Real.sqrt (inner_side ^ 2 + inner_side ^ 2)
  let outer_diagonal := Real.sqrt (outer_side ^ 2 + outer_side ^ 2)
  let distance := (inner_diagonal / 2) + (outer_diagonal / 2)
  distance = greatest_distance :=
by
  sorry

end square_vertex_distance_l90_90922


namespace minimum_pies_for_trick_l90_90295

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l90_90295


namespace probability_at_least_one_head_l90_90220

theorem probability_at_least_one_head (n : ℕ) (hn : n = 5) (p_tails : ℚ) (h_p : p_tails = 1 / 2) :
    (1 - (p_tails ^ n)) = 31 / 32 :=
by
  sorry

end probability_at_least_one_head_l90_90220


namespace trick_proof_l90_90286

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l90_90286


namespace initial_population_l90_90727

theorem initial_population (P : ℝ) (h1 : ∀ n : ℕ, n = 2 → P * (0.7 ^ n) = 3920) : P = 8000 := by
  sorry

end initial_population_l90_90727


namespace evaluate_expression_l90_90155

theorem evaluate_expression : (5 * 3 ^ 4 + 6 * 4 ^ 3 = 789) :=
by
  sorry

end evaluate_expression_l90_90155


namespace find_intersecting_lines_l90_90647

theorem find_intersecting_lines (x y : ℝ) : 
  (2 * x - y)^2 - (x + 3 * y)^2 = 0 ↔ x = 4 * y ∨ x = - (2 / 3) * y :=
by
  sorry

end find_intersecting_lines_l90_90647


namespace geometric_sequence_a3_value_l90_90380

theorem geometric_sequence_a3_value
  {a : ℕ → ℝ}
  (h1 : a 1 + a 5 = 82)
  (h2 : a 2 * a 4 = 81)
  (h3 : ∀ n : ℕ, a (n + 1) = a n * a 3 / a 2) :
  a 3 = 9 :=
sorry

end geometric_sequence_a3_value_l90_90380


namespace velocity_volleyball_league_members_l90_90850

theorem velocity_volleyball_league_members (total_cost : ℕ) (socks_cost t_shirt_cost cost_per_member members : ℕ)
  (h_socks_cost : socks_cost = 6)
  (h_t_shirt_cost : t_shirt_cost = socks_cost + 7)
  (h_cost_per_member : cost_per_member = 2 * (socks_cost + t_shirt_cost))
  (h_total_cost : total_cost = 3510)
  (h_total_cost_eq : total_cost = cost_per_member * members) :
  members = 92 :=
by
  sorry

end velocity_volleyball_league_members_l90_90850


namespace area_of_circle_l90_90078

theorem area_of_circle (r : ℝ) (h : r = 3) : 
  (∀ A : ℝ, A = π * r^2) → A = 9 * π :=
by
  intro area_formula
  sorry

end area_of_circle_l90_90078


namespace lending_period_C_l90_90300

theorem lending_period_C (R : ℝ) (P_B P_C T_B I : ℝ) (h1 : R = 13.75) (h2 : P_B = 4000) (h3 : P_C = 2000) (h4 : T_B = 2) (h5 : I = 2200) : 
  ∃ T_C : ℝ, T_C = 4 :=
by
  -- Definitions and known facts
  let I_B := (P_B * R * T_B) / 100
  let I_C := I - I_B
  let T_C := I_C / ((P_C * R) / 100)
  -- Prove the target
  use T_C
  sorry

end lending_period_C_l90_90300


namespace infinite_squares_in_arithmetic_sequence_l90_90399

open Nat Int

theorem infinite_squares_in_arithmetic_sequence
  (a d : ℤ) (h_d_nonneg : d ≥ 0) (x : ℤ) 
  (hx_square : ∃ n : ℕ, a + n * d = x * x) :
  ∃ (infinitely_many_n : ℕ → Prop), 
    (∀ k : ℕ, ∃ n : ℕ, infinitely_many_n n ∧ a + n * d = (x + k * d) * (x + k * d)) :=
sorry

end infinite_squares_in_arithmetic_sequence_l90_90399


namespace part1_part2_l90_90847

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (m n : ℝ) : f (m + n) = f m * f n
axiom positive_property (x : ℝ) (h : x > 0) : 0 < f x ∧ f x < 1

theorem part1 (x : ℝ) : f 0 = 1 ∧ (x < 0 → f x > 1) := by
  sorry

theorem part2 (x : ℝ) : 
  f (2 * x^2 - 4 * x - 1) < 1 ∧ f (x - 1) < 1 → x < -1/2 ∨ x > 2 := by
  sorry

end part1_part2_l90_90847


namespace overall_profit_percentage_l90_90616

theorem overall_profit_percentage :
  let SP_A := 900
  let SP_B := 1200
  let SP_C := 1500
  let P_A := 300
  let P_B := 400
  let P_C := 500
  let CP_A := SP_A - P_A
  let CP_B := SP_B - P_B
  let CP_C := SP_C - P_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let ProfitPercentage := (TP / TCP) * 100
  ProfitPercentage = 50 := by
  sorry

end overall_profit_percentage_l90_90616


namespace line_passing_D_parallel_l_tangent_incircle_l90_90006

-- Define a triangle ABC
variables {A B C D O : Point}

-- Angle bisector condition
def angle_bisector (A B C D : Point) : Prop :=
  (∠BAC / 2 = ∠BAD) ∧ (∠BAC / 2 = ∠CAD)

-- Tangent condition
def tangent_to_circumcircle (A O : Point) (Ω : Circle) (l : Line) : Prop :=
  tangent l Ω A

-- Parallel condition
def parallel (l m : Line) : Prop :=
  ∀ {P Q R}, distinct P Q R → (l through (P, Q)) → (m through (P, R))

-- Tangent to incircle condition
def tangent_to_incircle (m : Line) (ω : Circle) (D : Point) : Prop :=
  tangent m ω D

-- The main theorem statement in Lean 4
theorem line_passing_D_parallel_l_tangent_incircle
  (h1 : angle_bisector A B C D)
  (h2 : tangent_to_circumcircle A O Ω l)
  (h3 : parallel l m)
  (h4 : tangent_to_incircle incircle ABC D) :
  ∃ m : Line, parallel_to_line_through_D
    (∀ {ω : Circle}, incircle ABC ω  → tangent_to_incircle m ω D) :=
sorry

end line_passing_D_parallel_l_tangent_incircle_l90_90006


namespace sum_is_correct_l90_90952

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l90_90952


namespace coeff_x5_in_expansion_l90_90005

noncomputable def binomial_expansion_coeff (n k : ℕ) (x : ℝ) : ℝ :=
  Real.sqrt x ^ (n - k) * 2 ^ k * (Nat.choose n k)

theorem coeff_x5_in_expansion :
  (binomial_expansion_coeff 12 2 x) = 264 :=
by
  sorry

end coeff_x5_in_expansion_l90_90005


namespace smallest_n_satisfies_area_l90_90477

noncomputable def area (n : ℕ) : ℝ :=
  let z := (n : ℂ) + 2 * complex.I
  let z2 := z ^ 2
  let z3 := z ^ 3
  let x1 := complex.re z
  let y1 := complex.im z
  let x2 := complex.re z2
  let y2 := complex.im z2
  let x3 := complex.re z3
  let y3 := complex.im z3
  (1 / 2 : ℝ) * (x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1).abs

theorem smallest_n_satisfies_area (n : ℕ) (h : n = 20) : (area n) > 3000 := by
  sorry

end smallest_n_satisfies_area_l90_90477


namespace find_x_l90_90772

theorem find_x (x : ℝ) (h : 0 < x) (hx : 0.01 * x * x^2 = 16) : x = 12 :=
sorry

end find_x_l90_90772


namespace distribution_methods_l90_90089

theorem distribution_methods (A B C: ℕ) (h_nonneg: 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C) 
  (h_total: A + B + C = 7) (h_constraints: A ≤ 3 ∧ B ≤ 3 ∧ C ≤ 3):
  nat.factorial 7 / (nat.factorial A * nat.factorial B * nat.factorial C) = 24 :=
sorry

end distribution_methods_l90_90089


namespace calculation_l90_90932

theorem calculation :
  (-1:ℤ)^(2022) + (Real.sqrt 9) - 2 * (Real.sin (Real.pi / 6)) = 3 := by
  -- According to the mathematical problem and the given solution.
  -- Here we use essential definitions and facts provided in the problem.
  sorry

end calculation_l90_90932


namespace min_days_equal_duties_l90_90255

/--
Uncle Chernomor appoints 9 or 10 of the 33 warriors to duty each evening. 
Prove that the minimum number of days such that each warrior has been on duty the same number of times is 7.
-/
theorem min_days_equal_duties (k l m : ℕ) (k_nonneg : 0 ≤ k) (l_nonneg : 0 ≤ l)
  (h : 9 * k + 10 * l = 33 * m) (h_min : k + l = 7) : m = 2 :=
by 
  -- The necessary proof will go here
  sorry

end min_days_equal_duties_l90_90255


namespace slope_of_line_l90_90234

def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (4, 5)

theorem slope_of_line : 
  let (x1, y1) := point1
  let (x2, y2) := point2
  (x2 - x1) ≠ 0 → (y2 - y1) / (x2 - x1) = 1 := by
  sorry

end slope_of_line_l90_90234


namespace min_distance_ellipse_to_line_l90_90665

open Real

noncomputable def ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x^2 / 4) + y^2 = 1

noncomputable def line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x + y = 2 * sqrt 5

theorem min_distance_ellipse_to_line :
  let d_min := (1 / 2) * sqrt 10 in
  ∀ (P : ℝ × ℝ), ellipse P → ∃ d, 
  (∀ Q : ℝ × ℝ, line Q → ∂(P, Q) = d) ∧ d = d_min :=
sorry

end min_distance_ellipse_to_line_l90_90665


namespace interval_first_bell_l90_90320

theorem interval_first_bell (x : ℕ) : (Nat.lcm (Nat.lcm (Nat.lcm x 10) 14) 18 = 630) → x = 1 := by
  sorry

end interval_first_bell_l90_90320


namespace sum_of_first_five_prime_units_digit_3_l90_90960

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l90_90960


namespace plane_equation_l90_90198

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def point_on_plane (P : point) (a b c d : ℝ) : Prop :=
  match P with
  | (x, y, z) => a * x + b * y + c * z + d = 0

def normal_to_plane (n : vector) (a b c : ℝ) : Prop :=
  match n with
  | (nx, ny, nz) => (a, b, c) = (nx, ny, nz)

theorem plane_equation
  (P₀ : point) (u : vector)
  (x₀ y₀ z₀ : ℝ) (a b c d : ℝ)
  (h1 : P₀ = (1, 2, 1))
  (h2 : u = (-2, 1, 3))
  (h3 : point_on_plane (1, 2, 1) a b c d)
  (h4 : normal_to_plane (-2, 1, 3) a b c)
  : (2 : ℝ) * (x₀ : ℝ) - (y₀ : ℝ) - (3 : ℝ) * (z₀ : ℝ) + (3 : ℝ) = 0 :=
sorry

end plane_equation_l90_90198


namespace stanley_sold_4_cups_per_hour_l90_90052

theorem stanley_sold_4_cups_per_hour (S : ℕ) (Carl_Hour : ℕ) :
  (Carl_Hour = 7) →
  21 = (Carl_Hour * 3) →
  (21 - 9) = (S * 3) →
  S = 4 :=
by
  intros Carl_Hour_eq Carl_3hours Stanley_eq
  sorry

end stanley_sold_4_cups_per_hour_l90_90052


namespace value_of_expression_l90_90360

theorem value_of_expression (a b : ℝ) (h1 : a^2 + 2012 * a + 1 = 0) (h2 : b^2 + 2012 * b + 1 = 0) :
  (2 + 2013 * a + a^2) * (2 + 2013 * b + b^2) = -2010 := 
  sorry

end value_of_expression_l90_90360


namespace largest_good_number_is_576_smallest_bad_number_is_443_l90_90651

def is_good_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℤ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

def largest_good_number : ℕ := 576

def smallest_bad_number : ℕ := 443

theorem largest_good_number_is_576 : ∀ M : ℕ, is_good_number M → M ≤ 576 := 
by
  sorry

theorem smallest_bad_number_is_443 : ∀ M : ℕ, ¬ is_good_number M → 443 ≤ M :=
by
  sorry

end largest_good_number_is_576_smallest_bad_number_is_443_l90_90651


namespace combined_weight_l90_90392

-- We define the variables and the conditions
variables (x y : ℝ)

-- First condition 
def condition1 : Prop := y = (16 - 4) + (30 - 6) + (x - 3)

-- Second condition
def condition2 : Prop := y = 12 + 24 + (x - 3)

-- The statement to prove
theorem combined_weight (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : y = x + 33 :=
by
  -- Skipping the proof part
  sorry

end combined_weight_l90_90392


namespace solution_set_of_inequality_l90_90881

theorem solution_set_of_inequality : {x : ℝ | x^2 - 2 * x ≤ 0} = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l90_90881


namespace find_ck_l90_90069

theorem find_ck (d r k : ℕ) (a_n b_n c_n : ℕ → ℕ) 
  (h_an : ∀ n, a_n n = 1 + (n - 1) * d)
  (h_bn : ∀ n, b_n n = r ^ (n - 1))
  (h_cn : ∀ n, c_n n = a_n n + b_n n)
  (h_ckm1 : c_n (k - 1) = 30)
  (h_ckp1 : c_n (k + 1) = 300) :
  c_n k = 83 := 
sorry

end find_ck_l90_90069


namespace cube_side_length_in_cone_l90_90092

noncomputable def side_length_of_inscribed_cube (r h : ℝ) : ℝ :=
  if r = 1 ∧ h = 3 then (3 * Real.sqrt 2) / (3 + Real.sqrt 2) else 0

theorem cube_side_length_in_cone :
  side_length_of_inscribed_cube 1 3 = (3 * Real.sqrt 2) / (3 + Real.sqrt 2) :=
by
  sorry

end cube_side_length_in_cone_l90_90092


namespace simplify_sqrt_l90_90868

theorem simplify_sqrt (x : ℝ) (h : x = (Real.sqrt 3) + 1) : Real.sqrt (x^2) = Real.sqrt 3 + 1 :=
by
  -- This will serve as the placeholder for the proof.
  sorry

end simplify_sqrt_l90_90868


namespace tony_total_puzzle_time_l90_90421

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l90_90421


namespace turnip_count_example_l90_90534

theorem turnip_count_example : 6 + 9 = 15 := 
by
  -- Sorry is used to skip the actual proof
  sorry

end turnip_count_example_l90_90534


namespace part_one_part_two_l90_90832

theorem part_one (g : ℝ → ℝ) (h : ∀ x, g x = |x - 1| + 2) : {x : ℝ | |g x| < 5} = {x : ℝ | -2 < x ∧ x < 4} :=
sorry

theorem part_two (f g : ℝ → ℝ) (h1 : ∀ x, f x = |2 * x - a| + |2 * x + 3|) (h2 : ∀ x, g x = |x - 1| + 2) 
(h3 : ∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g x2) : {a : ℝ | a ≥ -1 ∨ a ≤ -5} :=
sorry

end part_one_part_two_l90_90832


namespace evaluate_f_g_3_l90_90364

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l90_90364


namespace matrix_pow_2020_l90_90325

-- Define the matrix type and basic multiplication rule
def M : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![3, 1]]

theorem matrix_pow_2020 :
  M ^ 2020 = ![![1, 0], ![6060, 1]] := by
  sorry

end matrix_pow_2020_l90_90325


namespace distance_interval_l90_90481

theorem distance_interval (d : ℝ) (h₁ : ¬ (d ≥ 8)) (h₂ : ¬ (d ≤ 6)) (h₃ : ¬ (d ≤ 3)) : 6 < d ∧ d < 8 := by
  sorry

end distance_interval_l90_90481


namespace profit_percent_l90_90301

theorem profit_percent (marked_price : ℚ) (pens_bought : ℚ) (discount : ℚ) :
  pens_bought = 120 →
  marked_price = 100 →
  discount = 2 →
  (100 * (1 - discount / 100) * pens_bought / (marked_price * pens_bought / 100)) * 100 = 205.8 :=
by
  intros h1 h2 h3
  have cp := (marked_price * pens_bought) / 100 -- cost price
  have sp := 100 * (1 - discount / 100) -- selling price per pen
  have profit := sp - cp -- profit per pen
  have profit_percent := (profit / cp) * 100 -- profit percent
  sorry

end profit_percent_l90_90301


namespace shorter_base_length_l90_90725

-- Let AB be the longer base of the trapezoid with length 24 cm
def AB : ℝ := 24

-- Let KT be the distance between midpoints of the diagonals with length 4 cm
def KT : ℝ := 4

-- Let CD be the shorter base of the trapezoid
variable (CD : ℝ)

-- The given condition is that KT is equal to half the difference of the lengths of the bases
axiom KT_eq : KT = (AB - CD) / 2

theorem shorter_base_length : CD = 16 := by
  sorry

end shorter_base_length_l90_90725


namespace sum_of_first_five_primes_with_units_digit_3_l90_90984

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90984


namespace max_area_square_l90_90401

theorem max_area_square (P : ℝ) : 
  ∀ x y : ℝ, 2 * x + 2 * y = P → (x * y ≤ (P / 4) ^ 2) :=
by
  sorry

end max_area_square_l90_90401


namespace sum_of_interior_angles_l90_90304

theorem sum_of_interior_angles (n : ℕ) (interior_angle : ℝ) :
  (interior_angle = 144) → (180 - 144) * n = 360 → n = 10 → (n - 2) * 180 = 1440 :=
by
  intros h1 h2 h3
  sorry

end sum_of_interior_angles_l90_90304


namespace cube_face_min_sum_l90_90860

open Set

theorem cube_face_min_sum (S : Finset ℕ)
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_faces_sum : ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
                    (a + b + c >= 10) ∨ 
                    (a + b + d >= 10) ∨ 
                    (a + c + d >= 10) ∨ 
                    (b + c + d >= 10)) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 16 :=
sorry

end cube_face_min_sum_l90_90860


namespace no_32_people_class_exists_30_people_class_l90_90000

-- Definition of the conditions: relationship between boys and girls
def friends_condition (B G : ℕ) : Prop :=
  3 * B = 2 * G

-- The first problem statement: No 32 people class
theorem no_32_people_class : ¬ ∃ (B G : ℕ), friends_condition B G ∧ B + G = 32 := 
sorry

-- The second problem statement: There is a 30 people class
theorem exists_30_people_class : ∃ (B G : ℕ), friends_condition B G ∧ B + G = 30 := 
sorry

end no_32_people_class_exists_30_people_class_l90_90000


namespace initial_professors_l90_90019

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l90_90019


namespace sufficient_and_necessary_condition_l90_90461

theorem sufficient_and_necessary_condition (x : ℝ) : 
  2 * x - 4 ≥ 0 ↔ x ≥ 2 :=
sorry

end sufficient_and_necessary_condition_l90_90461


namespace min_value_a_plus_3b_l90_90497

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end min_value_a_plus_3b_l90_90497


namespace snow_probability_at_least_once_l90_90573

theorem snow_probability_at_least_once (p : ℚ) (h : p = 3 / 4) : 
  let q := 1 - p in let prob_not_snow_4_days := q^4 in (1 - prob_not_snow_4_days) = 255 / 256 := 
by
  sorry

end snow_probability_at_least_once_l90_90573


namespace range_f_x1_x2_l90_90815

noncomputable def f (c x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1

theorem range_f_x1_x2 (c x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) 
  (h4 : 36 - 24 * c > 0) (h5 : ∀ x, f c x = 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1) :
  1 < f c x1 / x2 ∧ f c x1 / x2 < 5 / 2 :=
sorry

end range_f_x1_x2_l90_90815


namespace greatest_integer_a_for_domain_of_expression_l90_90646

theorem greatest_integer_a_for_domain_of_expression :
  ∃ a : ℤ, (a^2 < 60 ∧ (∀ b : ℤ, b^2 < 60 → b ≤ a)) :=
  sorry

end greatest_integer_a_for_domain_of_expression_l90_90646


namespace goal_l90_90125

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l90_90125


namespace num_valid_configurations_l90_90376

-- Definitions used in the problem
def grid := (Fin 8) × (Fin 8)
def knights_tell_truth := true
def knaves_lie := true
def statement (i j : Fin 8) (r c : grid → ℕ) := (c ⟨0,j⟩ > r ⟨i,0⟩)

-- The theorem statement to prove
theorem num_valid_configurations : ∃ n : ℕ, n = 255 :=
sorry

end num_valid_configurations_l90_90376


namespace simplify_trig_identity_l90_90552

open Real

theorem simplify_trig_identity (x y : ℝ) :
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = sin y ^ 2 := 
sorry

end simplify_trig_identity_l90_90552


namespace outdoor_tables_count_l90_90912

theorem outdoor_tables_count (num_indoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) : ℕ :=
  let num_outdoor_tables := (total_chairs - (num_indoor_tables * chairs_per_indoor_table)) / chairs_per_outdoor_table
  num_outdoor_tables

example (h₁ : num_indoor_tables = 9)
        (h₂ : chairs_per_indoor_table = 10)
        (h₃ : chairs_per_outdoor_table = 3)
        (h₄ : total_chairs = 123) :
        outdoor_tables_count 9 10 3 123 = 11 :=
by
  -- Only the statement has to be provided; proof steps are not needed
  sorry

end outdoor_tables_count_l90_90912


namespace calculate_expression_l90_90630
open Complex

-- Define the given values for a and b
def a := 3 + 2 * Complex.I
def b := 2 - 3 * Complex.I

-- Define the target expression
def target := 3 * a + 4 * b

-- The statement asserts that the target expression equals the expected result
theorem calculate_expression : target = 17 - 6 * Complex.I := by
  sorry

end calculate_expression_l90_90630


namespace bicycle_final_price_l90_90276

theorem bicycle_final_price : 
  let original_price := 200 
  let weekend_discount := 0.40 * original_price 
  let price_after_weekend_discount := original_price - weekend_discount 
  let wednesday_discount := 0.20 * price_after_weekend_discount 
  let final_price := price_after_weekend_discount - wednesday_discount 
  final_price = 96 := 
by 
  sorry

end bicycle_final_price_l90_90276


namespace power_rule_example_l90_90317

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l90_90317


namespace nonagon_diagonal_intersection_probability_l90_90253

-- Definitions based on conditions from part a)
def nonagon_diagonals_count (n : ℕ) : ℕ := (n * (n - 3)) / 2

def choose (n k : ℕ) : ℕ := nat.choose n k

-- The problem: Prove the probability that two chosen diagonals of a regular nonagon intersect inside the nonagon is 14/39
theorem nonagon_diagonal_intersection_probability : 
  let n := 9 in 
  let total_diagonals := nonagon_diagonals_count n in
  let total_ways_to_choose_diagonals := choose total_diagonals 2 in
  let ways_to_choose_intersecting_diagonals := choose n 4 in
  total_ways_to_choose_diagonals ≠ 0 →
  (ways_to_choose_intersecting_diagonals / total_ways_to_choose_diagonals : ℚ) = 14 / 39 :=
begin
  sorry -- Proof is omitted as per the instructions
end

end nonagon_diagonal_intersection_probability_l90_90253


namespace problem_statement_l90_90264

theorem problem_statement : 3.5 * 2.5 + 6.5 * 2.5 = 25 := by
  sorry

end problem_statement_l90_90264


namespace intersection_complement_l90_90820

def U : Set ℤ := Set.univ
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≠ x}
def C_U_B : Set ℤ := {x | x ≠ 0 ∧ x ≠ 1}

theorem intersection_complement :
  A ∩ C_U_B = {-1, 2} :=
by
  sorry

end intersection_complement_l90_90820


namespace total_blocks_l90_90010

theorem total_blocks (red_blocks yellow_blocks blue_blocks : ℕ) 
  (h1 : red_blocks = 18) 
  (h2 : yellow_blocks = red_blocks + 7) 
  (h3 : blue_blocks = red_blocks + 14) : 
  red_blocks + yellow_blocks + blue_blocks = 75 := 
by
  sorry

end total_blocks_l90_90010


namespace change_in_expression_l90_90950

theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let original_expr := x^2 - 5 * x + 2
  let new_x := x + b
  let new_expr := (new_x)^2 - 5 * (new_x) + 2
  new_expr - original_expr = 2 * b * x + b^2 - 5 * b :=
by
  sorry

end change_in_expression_l90_90950


namespace shaded_region_area_l90_90615

/-- A rectangle measuring 12cm by 8cm has four semicircles drawn with their diameters as the sides
of the rectangle. Prove that the area of the shaded region inside the rectangle but outside
the semicircles is equal to 96 - 52π (cm²). --/
theorem shaded_region_area (A : ℝ) (π : ℝ) (hA : A = 96 - 52 * π) : 
  ∀ (length width r1 r2 : ℝ) (hl : length = 12) (hw : width = 8) 
  (hr1 : r1 = length / 2) (hr2 : r2 = width / 2),
  (length * width) - (2 * (1/2 * π * r1^2 + 1/2 * π * r2^2)) = A := 
by 
  sorry

end shaded_region_area_l90_90615


namespace system_exactly_two_solutions_l90_90644

theorem system_exactly_two_solutions (a : ℝ) : 
  (∃ x y : ℝ, |y + x + 8| + |y - x + 8| = 16 ∧ (|x| - 15)^2 + (|y| - 8)^2 = a) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, |y₁ + x₁ + 8| + |y₁ - x₁ + 8| = 16 ∧ (|x₁| - 15)^2 + (|y₁| - 8)^2 = a → 
                      |y₂ + x₂ + 8| + |y₂ - x₂ + 8| = 16 ∧ (|x₂| - 15)^2 + (|y₂| - 8)^2 = a → 
                      x₁ = x₂ ∧ y₁ = y₂) → 
  (a = 49 ∨ a = 289) :=
sorry

end system_exactly_two_solutions_l90_90644


namespace utility_bill_amount_l90_90856

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l90_90856


namespace count_implications_l90_90635

def r : Prop := sorry
def s : Prop := sorry

def statement_1 := ¬r ∧ ¬s
def statement_2 := ¬r ∧ s
def statement_3 := r ∧ ¬s
def statement_4 := r ∧ s

def neg_rs : Prop := r ∨ s

theorem count_implications : (statement_2 → neg_rs) ∧ 
                             (statement_3 → neg_rs) ∧ 
                             (statement_4 → neg_rs) ∧ 
                             (¬(statement_1 → neg_rs)) -> 
                             3 = 3 := by
  sorry

end count_implications_l90_90635


namespace smallest_n_for_trick_l90_90291

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l90_90291


namespace kyle_caught_14_fish_l90_90473

theorem kyle_caught_14_fish (K T C : ℕ) (h1 : K = T) (h2 : C = 8) (h3 : C + K + T = 36) : K = 14 :=
by
  -- Proof goes here
  sorry

end kyle_caught_14_fish_l90_90473


namespace ivan_speed_ratio_l90_90916

/-- 
A group of tourists started a hike from a campsite. Fifteen minutes later, Ivan returned to the campsite for a flashlight 
and started catching up with the group at a faster constant speed. He reached them 2.5 hours after initially leaving. 
Prove Ivan's speed is 1.2 times the group's speed.
-/
theorem ivan_speed_ratio (d_g d_i : ℝ) (t_g t_i : ℝ) (v_g v_i : ℝ)
    (h1 : t_g = 2.25)       -- Group's travel time (2.25 hours after initial 15 minutes)
    (h2 : t_i = 2.5)        -- Ivan's total travel time
    (h3 : d_g = t_g * v_g)  -- Distance covered by group
    (h4 : d_i = 3 * (v_g * (15 / 60))) -- Ivan's distance covered
    (h5 : d_g = d_i)        -- Ivan eventually catches up with the group
  : v_i / v_g = 1.2 := sorry

end ivan_speed_ratio_l90_90916


namespace sum_of_first_five_primes_with_units_digit_three_l90_90971

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l90_90971


namespace length_PC_l90_90825

-- Define lengths of the sides of triangle ABC.
def AB := 10
def BC := 8
def CA := 7

-- Define the similarity condition
def similar_triangles (PA PC : ℝ) : Prop :=
  PA / PC = AB / CA

-- Define the extension of side BC to point P
def extension_condition (PA PC : ℝ) : Prop :=
  PA = PC + BC

theorem length_PC (PC : ℝ) (PA : ℝ) :
  similar_triangles PA PC → extension_condition PA PC → PC = 56 / 3 :=
by
  intro h_sim h_ext
  sorry

end length_PC_l90_90825


namespace cannonball_maximum_height_l90_90913

def height_function (t : ℝ) := -20 * t^2 + 100 * t + 36

theorem cannonball_maximum_height :
  ∃ t₀ : ℝ, ∀ t : ℝ, height_function t ≤ height_function t₀ ∧ height_function t₀ = 161 :=
by
  sorry

end cannonball_maximum_height_l90_90913


namespace find_dividend_l90_90895

noncomputable def divisor := (-14 : ℚ) / 3
noncomputable def quotient := (-286 : ℚ) / 5
noncomputable def remainder := (19 : ℚ) / 9
noncomputable def dividend := 269 + (2 / 45 : ℚ)

theorem find_dividend :
  dividend = (divisor * quotient) + remainder := by
  sorry

end find_dividend_l90_90895


namespace utility_bills_total_correct_l90_90857

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l90_90857


namespace worker_payment_l90_90611

theorem worker_payment (x : ℕ) (daily_return : ℕ) (non_working_days : ℕ) (total_days : ℕ) 
    (net_earning : ℕ) 
    (H1 : daily_return = 25) 
    (H2 : non_working_days = 24) 
    (H3 : total_days = 30) 
    (H4 : net_earning = 0) 
    (H5 : ∀ w, net_earning = w * x - non_working_days * daily_return) : 
  x = 100 :=
by
  sorry

end worker_payment_l90_90611


namespace distance_between_points_l90_90229

noncomputable def distance (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := A
  let (x2, y2, z2) := B
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

theorem distance_between_points :
  distance (-3, 4, 0) (2, -1, 6) = Real.sqrt 86 :=
by
  sorry

end distance_between_points_l90_90229


namespace express_y_in_terms_of_x_l90_90674

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 1) : y = -3 * x + 1 := 
by
  sorry

end express_y_in_terms_of_x_l90_90674


namespace students_before_new_year_le_197_l90_90752

variable (N M k ℓ : ℕ)

-- Conditions
axiom condition_1 : M = (k * N) / 100
axiom condition_2 : 100 * M = k * N
axiom condition_3 : 100 * (M + 1) = ℓ * (N + 3)
axiom condition_4 : ℓ < 100

-- The theorem to prove
theorem students_before_new_year_le_197 :
  N ≤ 197 :=
by
  sorry

end students_before_new_year_le_197_l90_90752


namespace garden_strawberry_area_l90_90696

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l90_90696


namespace least_number_to_make_divisible_by_9_l90_90270

theorem least_number_to_make_divisible_by_9 (n : ℕ) :
  ∃ m : ℕ, (228712 + m) % 9 = 0 ∧ n = 5 :=
by
  sorry

end least_number_to_make_divisible_by_9_l90_90270


namespace tony_solving_puzzles_time_l90_90420

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l90_90420


namespace max_number_of_books_laughlin_can_buy_l90_90386

-- Definitions of costs and the budget constraint
def individual_book_cost : ℕ := 3
def four_book_bundle_cost : ℕ := 10
def seven_book_bundle_cost : ℕ := 15
def budget : ℕ := 20

-- Condition that Laughlin must buy at least one 4-book bundle
def minimum_required_four_book_bundles : ℕ := 1

-- Define the function to calculate the maximum number of books Laughlin can buy
def max_books (budget : ℕ) (individual_book_cost : ℕ) 
              (four_book_bundle_cost : ℕ) (seven_book_bundle_cost : ℕ) 
              (min_four_book_bundles : ℕ) : ℕ :=
  let remaining_budget_after_four_bundle := budget - (min_four_book_bundles * four_book_bundle_cost)
  if remaining_budget_after_four_bundle >= seven_book_bundle_cost then
    min_four_book_bundles * 4 + 7
  else if remaining_budget_after_four_bundle >= individual_book_cost then
    min_four_book_bundles * 4 + remaining_budget_after_four_bundle / individual_book_cost
  else
    min_four_book_bundles * 4

-- Proof statement: Laughlin can buy a maximum of 7 books
theorem max_number_of_books_laughlin_can_buy : 
  max_books budget individual_book_cost four_book_bundle_cost seven_book_bundle_cost minimum_required_four_book_bundles = 7 :=
by
  sorry

end max_number_of_books_laughlin_can_buy_l90_90386


namespace polar_to_cartesian_l90_90414

theorem polar_to_cartesian (ρ : ℝ) (θ : ℝ) (hx : ρ = 3) (hy : θ = π / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.cos (π / 6), 3 * Real.sin (π / 6)) := by
  sorry

end polar_to_cartesian_l90_90414


namespace rahim_books_from_first_shop_l90_90219

variable (books_first_shop_cost : ℕ)
variable (second_shop_books : ℕ)
variable (second_shop_books_cost : ℕ)
variable (average_price_per_book : ℕ)
variable (number_of_books_first_shop : ℕ)

theorem rahim_books_from_first_shop
  (h₁ : books_first_shop_cost = 581)
  (h₂ : second_shop_books = 20)
  (h₃ : second_shop_books_cost = 594)
  (h₄ : average_price_per_book = 25)
  (h₅ : (books_first_shop_cost + second_shop_books_cost) = (number_of_books_first_shop + second_shop_books) * average_price_per_book) :
  number_of_books_first_shop = 27 :=
sorry

end rahim_books_from_first_shop_l90_90219


namespace identify_radioactive_balls_l90_90740

theorem identify_radioactive_balls (balls : Fin 11 → Bool) (measure : (Finset (Fin 11)) → Bool) :
  (∃ (t1 t2 : Fin 11), ¬ t1 = t2 ∧ balls t1 = true ∧ balls t2 = true) →
  (∃ (pairs : List (Finset (Fin 11))), pairs.length ≤ 7 ∧
    ∀ t1 t2, t1 ≠ t2 ∧ balls t1 = true ∧ balls t2 = true →
      ∃ pair ∈ pairs, measure pair = true ∧ (t1 ∈ pair ∨ t2 ∈ pair)) :=
by
  sorry

end identify_radioactive_balls_l90_90740


namespace ellipse_major_minor_ratio_l90_90814

theorem ellipse_major_minor_ratio (m : ℝ) (x y : ℝ) (h1 : x^2 + y^2 / m = 1) (h2 : 2 * 1 = 4 * Real.sqrt m) 
  : m = 1 / 4 :=
sorry

end ellipse_major_minor_ratio_l90_90814


namespace utility_bill_amount_l90_90855

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l90_90855


namespace total_number_of_students_l90_90468

theorem total_number_of_students (sample_size : ℕ) (first_year_selected : ℕ) (third_year_selected : ℕ) (second_year_students : ℕ) (second_year_selected : ℕ) (prob_selection : ℕ) :
  sample_size = 45 →
  first_year_selected = 20 →
  third_year_selected = 10 →
  second_year_students = 300 →
  second_year_selected = sample_size - first_year_selected - third_year_selected →
  prob_selection = second_year_selected / second_year_students →
  (sample_size / prob_selection) = 900 :=
by
  intros
  sorry

end total_number_of_students_l90_90468


namespace quadratic_function_passing_through_origin_l90_90818

-- Define the quadratic function y
def quadratic_function (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

-- State the problem as a theorem
theorem quadratic_function_passing_through_origin (m : ℝ) (h: quadratic_function m 0 = 0) : m = -4 :=
by
  -- Since we only need the statement, we put sorry here
  sorry

end quadratic_function_passing_through_origin_l90_90818


namespace divisible_by_65_l90_90546

theorem divisible_by_65 (n : ℕ) : 65 ∣ (5^n * (2^(2*n) - 3^n) + 2^n - 7^n) :=
sorry

end divisible_by_65_l90_90546


namespace mod_pow_solution_l90_90717

def m (x : ℕ) := x

theorem mod_pow_solution :
  ∃ (m : ℕ), 0 ≤ m ∧ m < 8 ∧ 13^6 % 8 = m ∧ m = 1 :=
by
  use 1
  sorry

end mod_pow_solution_l90_90717


namespace boat_distance_downstream_l90_90761

-- Definitions of the given conditions
def boat_speed_still_water : ℝ := 13
def stream_speed : ℝ := 4
def travel_time_downstream : ℝ := 4

-- Mathematical statement to be proved
theorem boat_distance_downstream : 
  let effective_speed_downstream := boat_speed_still_water + stream_speed
  in effective_speed_downstream * travel_time_downstream = 68 :=
by
  sorry

end boat_distance_downstream_l90_90761


namespace analysis_method_proves_sufficient_condition_l90_90781

-- Definitions and conditions from part (a)
def analysis_method_traces_cause_from_effect : Prop := true
def analysis_method_seeks_sufficient_conditions : Prop := true
def analysis_method_finds_conditions_for_inequality : Prop := true

-- The statement to be proven
theorem analysis_method_proves_sufficient_condition :
  analysis_method_finds_conditions_for_inequality →
  analysis_method_traces_cause_from_effect →
  analysis_method_seeks_sufficient_conditions →
  (B = "Sufficient condition") :=
by 
  sorry

end analysis_method_proves_sufficient_condition_l90_90781


namespace smallest_positive_integer_exists_l90_90743

theorem smallest_positive_integer_exists
    (x : ℕ) :
    (x % 7 = 2) ∧
    (x % 4 = 3) ∧
    (x % 6 = 1) →
    x = 135 :=
by
    sorry

end smallest_positive_integer_exists_l90_90743


namespace woman_away_time_l90_90776

noncomputable def angle_hour_hand (n : ℝ) : ℝ := 150 + n / 2
noncomputable def angle_minute_hand (n : ℝ) : ℝ := 6 * n

theorem woman_away_time : 
  (∀ n : ℝ, abs (angle_hour_hand n - angle_minute_hand n) = 120) → 
  abs ((540 / 11 : ℝ) - (60 / 11 : ℝ)) = 43.636 :=
by sorry

end woman_away_time_l90_90776


namespace perimeter_of_polygon_l90_90804

-- Define the dimensions of the strips and their arrangement
def strip_width : ℕ := 4
def strip_length : ℕ := 16
def num_vertical_strips : ℕ := 2
def num_horizontal_strips : ℕ := 2

-- State the problem condition and the expected perimeter
theorem perimeter_of_polygon : 
  let vertical_perimeter := num_vertical_strips * strip_length
  let horizontal_perimeter := num_horizontal_strips * strip_length
  let corner_segments_perimeter := (num_vertical_strips + num_horizontal_strips) * strip_width
  vertical_perimeter + horizontal_perimeter + corner_segments_perimeter = 80 :=
by
  sorry

end perimeter_of_polygon_l90_90804


namespace all_acute_angles_in_first_quadrant_l90_90745

def terminal_side_same (θ₁ θ₂ : ℝ) : Prop := 
  ∃ (k : ℤ), θ₁ = θ₂ + 360 * k

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def first_quadrant_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem all_acute_angles_in_first_quadrant :
  ∀ θ : ℝ, acute_angle θ → first_quadrant_angle θ :=
by
  intros θ h
  exact h

end all_acute_angles_in_first_quadrant_l90_90745


namespace times_faster_l90_90684

theorem times_faster (A B W : ℝ) (h1 : A = 3 * B) (h2 : (A + B) * 21 = A * 28) : A = 3 * B :=
by sorry

end times_faster_l90_90684


namespace pollution_control_l90_90524

theorem pollution_control (x y : ℕ) (h1 : x - y = 5) (h2 : 2 * x + 3 * y = 45) : x = 12 ∧ y = 7 :=
by
  sorry

end pollution_control_l90_90524


namespace hundred_days_from_friday_is_sunday_l90_90429

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l90_90429


namespace samatha_tosses_five_coins_l90_90223

noncomputable def probability_at_least_one_head 
  (p.toss : ℕ → ℙ) 
  (h_independence : ∀ n m : ℕ, n ≠ m → ProbInd (p.toss n) (p.toss m))
  (h_tail_prob : ∀ n : ℕ, Pr (flip_tail (p.toss n)) = 1 / 2) : ℚ :=
  1 - (1/2)^5

theorem samatha_tosses_five_coins :
  let p.toss : ℕ → ℙ := flip_coin
  in probability_at_least_one_head p.toss (by sorry) (by sorry) = 31/32 :=
by
  sorry

end samatha_tosses_five_coins_l90_90223


namespace find_polynomial_value_l90_90174

theorem find_polynomial_value
  (x y : ℝ)
  (h1 : 3 * x + y = 5)
  (h2 : x + 3 * y = 6) :
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := 
by {
  -- The proof part is omitted here
  sorry
}

end find_polynomial_value_l90_90174


namespace range_of_a_l90_90168

noncomputable def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x ^ 2 + 2 * x + a ≥ 0 }

theorem range_of_a (a : ℝ) : (a > -8) → (∃ x, x ∈ A ∧ x ∈ B a) :=
by
  sorry

end range_of_a_l90_90168


namespace relationship_between_y_values_l90_90812

theorem relationship_between_y_values 
  (m : ℝ) 
  (y1 y2 y3 : ℝ)
  (h1 : y1 = (-1 : ℝ) ^ 2 + 2 * (-1 : ℝ) + m) 
  (h2 : y2 = (3 : ℝ) ^ 2 + 2 * (3 : ℝ) + m) 
  (h3 : y3 = ((1 / 2) : ℝ) ^ 2 + 2 * ((1 / 2) : ℝ) + m) : 
  y2 > y3 ∧ y3 > y1 := 
by 
  sorry

end relationship_between_y_values_l90_90812


namespace largest_whole_number_l90_90724

theorem largest_whole_number (x : ℤ) : 9 * x < 200 → x ≤ 22 := by
  sorry

end largest_whole_number_l90_90724


namespace Gwen_walking_and_elevation_gain_l90_90943

theorem Gwen_walking_and_elevation_gain :
  ∀ (jogging_time walking_time total_time elevation_gain : ℕ)
    (jogging_feet total_feet : ℤ),
    jogging_time = 15 ∧ jogging_feet = 500 ∧ (jogging_time + walking_time = total_time) ∧
    (5 * walking_time = 3 * jogging_time) ∧ (total_time * jogging_feet = 15 * total_feet)
    → walking_time = 9 ∧ total_feet = 800 := by 
  sorry

end Gwen_walking_and_elevation_gain_l90_90943


namespace observe_three_cell_types_l90_90607

def biology_experiment
  (material : Type) (dissociation_fixative : material) (acetic_orcein_stain : material) (press_slide : Prop) : Prop :=
  ∃ (testes : material) (steps : material → Prop),
    steps testes ∧ press_slide ∧ (steps dissociation_fixative) ∧ (steps acetic_orcein_stain)

theorem observe_three_cell_types (material : Type)
  (dissociation_fixative acetic_orcein_stain : material)
  (press_slide : Prop)
  (steps : material → Prop) :
  biology_experiment material dissociation_fixative acetic_orcein_stain press_slide →
  ∃ (metaphase_of_mitosis metaphase_of_first_meiosis metaphase_of_second_meiosis : material), 
    steps metaphase_of_mitosis ∧ steps metaphase_of_first_meiosis ∧ steps metaphase_of_second_meiosis :=
sorry

end observe_three_cell_types_l90_90607


namespace find_radius_of_tangent_circle_l90_90831

def tangent_circle_radius : Prop :=
  ∃ (r : ℝ), 
    (r > 0) ∧ 
    (∀ (θ : ℝ),
      (∃ (x y : ℝ),
        x = 1 + r * Real.cos θ ∧ 
        y = 1 + r * Real.sin θ ∧ 
        x + y - 1 = 0))
    → r = (Real.sqrt 2) / 2

theorem find_radius_of_tangent_circle : tangent_circle_radius :=
sorry

end find_radius_of_tangent_circle_l90_90831


namespace contractor_daily_wage_l90_90453

theorem contractor_daily_wage 
  (total_days : ℕ)
  (daily_wage : ℝ)
  (fine_per_absence : ℝ)
  (total_earned : ℝ)
  (absent_days : ℕ)
  (H1 : total_days = 30)
  (H2 : fine_per_absence = 7.5)
  (H3 : total_earned = 555.0)
  (H4 : absent_days = 6)
  (H5 : total_earned = daily_wage * (total_days - absent_days) - fine_per_absence * absent_days) :
  daily_wage = 25 := by
  sorry

end contractor_daily_wage_l90_90453


namespace average_visitor_on_other_days_is_240_l90_90097

-- Definition of conditions: average visitors on Sundays,
-- average visitors per day, the month starts with a Sunday
def avg_visitors_sunday : ℕ := 510
def avg_visitors_month : ℕ := 285
def days_in_month : ℕ := 30
def sundays_in_month : ℕ := 5

-- Define the total number of days that are not Sunday
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors equation based on given conditions
def total_visitors (avg_visitors_other_days : ℕ) : Prop :=
  (avg_visitors_month * days_in_month) = (avg_visitors_sunday * sundays_in_month) + (avg_visitors_other_days * other_days_in_month)

-- Objective: Prove that the average number of visitors on other days is 240
theorem average_visitor_on_other_days_is_240 : ∃ (V : ℕ), total_visitors V ∧ V = 240 :=
by
  use 240
  simp [total_visitors, avg_visitors_sunday, avg_visitors_month, days_in_month, sundays_in_month, other_days_in_month]
  sorry

end average_visitor_on_other_days_is_240_l90_90097


namespace fraction_between_stops_l90_90330

/-- Prove that the fraction of the remaining distance traveled between Maria's first and second stops is 1/4. -/
theorem fraction_between_stops (total_distance first_stop_distance remaining_distance final_leg_distance : ℝ)
  (h_total : total_distance = 400)
  (h_first_stop : first_stop_distance = total_distance / 2)
  (h_remaining : remaining_distance = total_distance - first_stop_distance)
  (h_final_leg : final_leg_distance = 150)
  (h_second_leg : remaining_distance - final_leg_distance = 50) :
  50 / remaining_distance = 1 / 4 :=
by
  { sorry }

end fraction_between_stops_l90_90330


namespace system_solution_l90_90803

theorem system_solution (m n : ℚ) (x y : ℚ) 
  (h₁ : 2 * x + m * y = 5) 
  (h₂ : n * x - 3 * y = 2) 
  (h₃ : x = 3)
  (h₄ : y = 1) : 
  m / n = -3 / 5 :=
by sorry

end system_solution_l90_90803


namespace sum_of_remaining_six_numbers_l90_90410

theorem sum_of_remaining_six_numbers :
  ∀ (S T U : ℕ), 
    S = 20 * 500 → T = 14 * 390 → U = S - T → U = 4540 :=
by
  intros S T U hS hT hU
  sorry

end sum_of_remaining_six_numbers_l90_90410


namespace cubic_root_square_diff_l90_90528

theorem cubic_root_square_diff (r s : ℝ) (h1 : Polynomial.eval r (Polynomial.C 1 - (Polynomial.C (2 : ℝ) * Polynomial.X)) = 0) 
  (h2 : Polynomial.eval s (Polynomial.C 1 - (Polynomial.C (2 : ℝ) * Polynomial.X)) = 0)
  (h3 : r + s + 1 = 0) (h4 : r * s = -1) : 
  (r - s) ^ 2 = 5 := by sorry

end cubic_root_square_diff_l90_90528


namespace sam_final_amount_l90_90865

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def initial_amount : ℕ := (initial_dimes * dime_value) + (initial_quarters * quarter_value) + (initial_nickels * nickel_value)
def dad_amount : ℕ := (dad_dimes * dime_value) + (dad_quarters * quarter_value)
def mom_amount : ℕ := (mom_nickels * nickel_value) + (mom_dimes * dime_value)

def final_amount : ℕ := initial_amount + dad_amount - mom_amount

theorem sam_final_amount : final_amount = 325 := by
  sorry

end sam_final_amount_l90_90865


namespace pacific_ocean_area_rounded_l90_90874

def pacific_ocean_area : ℕ := 19996800

def ten_thousand : ℕ := 10000

noncomputable def pacific_ocean_area_in_ten_thousands (area : ℕ) : ℕ :=
  (area / ten_thousand + if (area % ten_thousand) >= (ten_thousand / 2) then 1 else 0)

theorem pacific_ocean_area_rounded :
  pacific_ocean_area_in_ten_thousands pacific_ocean_area = 2000 :=
by
  sorry

end pacific_ocean_area_rounded_l90_90874


namespace todd_ingredients_l90_90580

variables (B R N : ℕ) (P A : ℝ) (I : ℝ)

def todd_problem (B R N : ℕ) (P A I : ℝ) : Prop := 
  B = 100 ∧ 
  R = 110 ∧ 
  N = 200 ∧ 
  P = 0.75 ∧ 
  A = 65 ∧ 
  I = 25

theorem todd_ingredients :
  todd_problem 100 110 200 0.75 65 25 :=
by sorry

end todd_ingredients_l90_90580


namespace M_gt_N_l90_90993

theorem M_gt_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  M > N := by
  sorry

end M_gt_N_l90_90993


namespace max_y_difference_eq_l90_90066

theorem max_y_difference_eq (x y p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h : x * y = p * x + q * y) : y - x = (p - 1) * (q + 1) :=
sorry

end max_y_difference_eq_l90_90066


namespace greatest_power_of_2_factor_of_expr_l90_90741

theorem greatest_power_of_2_factor_of_expr :
  (∃ k, 2 ^ k ∣ 12 ^ 600 - 8 ^ 400 ∧ ∀ m, 2 ^ m ∣ 12 ^ 600 - 8 ^ 400 → m ≤ 1204) :=
sorry

end greatest_power_of_2_factor_of_expr_l90_90741


namespace polyhedron_calculation_l90_90303

def faces := 32
def triangular := 10
def pentagonal := 8
def hexagonal := 14
def edges := 79
def vertices := 49
def T := 1
def P := 2

theorem polyhedron_calculation : 
  100 * P + 10 * T + vertices = 249 := 
sorry

end polyhedron_calculation_l90_90303


namespace fraction_value_l90_90390

theorem fraction_value (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -4 / 7 :=
by
  sorry

end fraction_value_l90_90390


namespace income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l90_90312

-- Define the conditions
def annual_income (year : ℕ) : ℝ := 0.0124 * (1 + 0.2) ^ (year - 1)
def annual_repayment : ℝ := 0.05

-- Proof Problem 1: Show that the subway's annual operating income exceeds the annual repayment at year 9
theorem income_exceeds_repayment_after_9_years :
  ∀ n ≥ 9, annual_income n > annual_repayment :=
by
  sorry

-- Define the cumulative payment function for the municipal government
def cumulative_payment (years : ℕ) : ℝ :=
  (annual_repayment * years) - (List.sum (List.map annual_income (List.range years)))

-- Proof Problem 2: Show the cumulative payment by the municipal government up to year 8 is 19,541,135 RMB
theorem cumulative_payment_up_to_year_8 :
  cumulative_payment 8 = 0.1954113485 :=
by
  sorry

end income_exceeds_repayment_after_9_years_cumulative_payment_up_to_year_8_l90_90312


namespace pillows_from_feathers_l90_90213

def feathers_per_pound : ℕ := 300
def feathers_total : ℕ := 3600
def pounds_per_pillow : ℕ := 2

theorem pillows_from_feathers :
  (feathers_total / feathers_per_pound / pounds_per_pillow) = 6 :=
by
  sorry

end pillows_from_feathers_l90_90213


namespace hyperbola_sufficient_not_necessary_condition_l90_90555

-- Define the equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 16 = 1

-- Define the asymptotic line equations of the hyperbola
def asymptotes_eq (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

-- Prove that the equation of the hyperbola is a sufficient but not necessary condition for the asymptotic lines
theorem hyperbola_sufficient_not_necessary_condition :
  (∀ x y : ℝ, hyperbola_eq x y → asymptotes_eq x y) ∧ ¬ (∀ x y : ℝ, asymptotes_eq x y → hyperbola_eq x y) :=
by
  sorry

end hyperbola_sufficient_not_necessary_condition_l90_90555


namespace water_fraction_after_replacements_l90_90605

theorem water_fraction_after_replacements (initial_volume : ℕ) (removed_volume : ℕ) (replacements : ℕ) :
  initial_volume = 16 → removed_volume = 4 → replacements = 4 →
  (3 / 4 : ℚ) ^ replacements = 81 / 256 :=
by
  intros h_initial_volume h_removed_volume h_replacements
  sorry

end water_fraction_after_replacements_l90_90605


namespace smallest_angle_l90_90875

theorem smallest_angle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by sorry

end smallest_angle_l90_90875


namespace smallest_a_l90_90261

theorem smallest_a (a : ℕ) (h1 : a > 0) (h2 : (∀ b : ℕ, b > 0 → b < a → ∀ h3 : b > 0, ¬ (gcd b 72 > 1 ∧ gcd b 90 > 1)))
  (h3 : gcd a 72 > 1) (h4 : gcd a 90 > 1) : a = 2 :=
by
  sorry

end smallest_a_l90_90261


namespace infinite_solutions_congruence_l90_90540

theorem infinite_solutions_congruence (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ᶠ x in at_top, a ^ x + x ≡ b [MOD c] :=
sorry

end infinite_solutions_congruence_l90_90540


namespace german_team_goals_l90_90119

/-- Given the conditions from three journalists and knowing that only two of them can be correct,
we need to prove the possible values for the number of goals (x) scored by the German team during the championship. --/
theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : odd x) 
  (h4 : (h1 ∨ h2 ∨ h3) ∧ ¬(h1 ∧ h2 ∧ h3)) :
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := sorry

end german_team_goals_l90_90119


namespace circle_area_l90_90190

theorem circle_area (r : ℝ) (h1 : 5 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 5 / 4 := by
  sorry

end circle_area_l90_90190


namespace count_valid_c_l90_90649

theorem count_valid_c : ∃ (count : ℕ), count = 670 ∧ 
  ∀ (c : ℤ), (-2007 ≤ c ∧ c ≤ 2007) → 
    (∃ (x : ℤ), (x^2 + c) % (2^2007) = 0) ↔ count = 670 :=
sorry

end count_valid_c_l90_90649


namespace farmer_initial_tomatoes_l90_90766

theorem farmer_initial_tomatoes 
  (T : ℕ) -- The initial number of tomatoes
  (picked : ℕ)   -- The number of tomatoes picked
  (diff : ℕ) -- The difference between initial number of tomatoes and picked
  (h1 : picked = 9) -- The farmer picked 9 tomatoes
  (h2 : diff = 8) -- The difference is 8
  (h3 : T - picked = diff) -- T - 9 = 8
  :
  T = 17 := sorry

end farmer_initial_tomatoes_l90_90766


namespace minimum_sum_l90_90206

open Matrix

noncomputable def a := 54
noncomputable def b := 40
noncomputable def c := 5
noncomputable def d := 4

theorem minimum_sum 
  (a b c d : ℕ) 
  (ha : 4 * a = 24 * a - 27 * b) 
  (hb : 4 * b = 15 * a - 17 * b) 
  (hc : 3 * c = 24 * c - 27 * d) 
  (hd : 3 * d = 15 * c - 17 * d) 
  (Hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  a + b + c + d = 103 :=
by
  sorry

end minimum_sum_l90_90206


namespace prob_0_to_4_l90_90053

noncomputable def ξ_dist : ProbabilityDistribution := 
  ProbabilityDistribution.normal 2 σ^2

axiom h1 : ξ_dist.prob (λ x, x ≤ 0) = 0.2

theorem prob_0_to_4 : ξ_dist.prob (λ x, 0 ≤ x ∧ x ≤ 4) = 0.6 :=
by
  sorry

end prob_0_to_4_l90_90053


namespace probability_of_one_fork_one_spoon_one_knife_l90_90678

theorem probability_of_one_fork_one_spoon_one_knife 
  (num_forks : ℕ) (num_spoons : ℕ) (num_knives : ℕ) (total_pieces : ℕ)
  (h_forks : num_forks = 7) (h_spoons : num_spoons = 8) (h_knives : num_knives = 5)
  (h_total : total_pieces = num_forks + num_spoons + num_knives) :
  (∃ (prob : ℚ), prob = 14 / 57) :=
by
  sorry

end probability_of_one_fork_one_spoon_one_knife_l90_90678


namespace simplify_expression_l90_90402

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (3 * x^2 * x^3) = 29 * x^5 := 
  sorry

end simplify_expression_l90_90402


namespace find_a_l90_90351

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_eq (x y a : ℝ) : Prop := x + a * y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y → line_eq x y a → (x - 1)^2 + (y - 2)^2 = 4) →
  ∃ a, (a = -1) :=
sorry

end find_a_l90_90351


namespace initial_water_amount_l90_90911

theorem initial_water_amount (x : ℝ) (h : x + 6.8 = 9.8) : x = 3 := 
by
  sorry

end initial_water_amount_l90_90911


namespace find_certain_number_l90_90273

theorem find_certain_number (x : ℝ) (h : 0.7 * x = 28) : x = 40 := 
by
  sorry

end find_certain_number_l90_90273


namespace cos_double_angle_l90_90188

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) : Real.cos (2 * θ) = -1/3 := by
  sorry

end cos_double_angle_l90_90188


namespace german_team_goals_l90_90123

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l90_90123


namespace sum_of_first_five_primes_units_digit_3_l90_90969

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l90_90969


namespace min_value_of_frac_l90_90662

open Real

theorem min_value_of_frac (x : ℝ) (hx : x > 0) : 
  ∃ (t : ℝ), t = 2 * sqrt 5 + 2 ∧ (∀ y, y > 0 → (x^2 + 2 * x + 5) / x ≥ t) :=
by
  sorry

end min_value_of_frac_l90_90662


namespace smallest_n_for_trick_l90_90290

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l90_90290


namespace expenditure_ratio_l90_90879

def ratio_of_incomes (I1 I2 : ℕ) : Prop := I1 / I2 = 5 / 4
def savings (I E : ℕ) : ℕ := I - E
def ratio_of_expenditures (E1 E2 : ℕ) : Prop := E1 / E2 = 3 / 2

theorem expenditure_ratio (I1 I2 E1 E2 : ℕ) 
  (I1_income : I1 = 5500)
  (income_ratio : ratio_of_incomes I1 I2)
  (savings_equal : savings I1 E1 = 2200 ∧ savings I2 E2 = 2200)
  : ratio_of_expenditures E1 E2 :=
by 
  sorry

end expenditure_ratio_l90_90879


namespace probability_at_least_3_l90_90458

noncomputable def probability_hitting_at_least_3_of_4 (p : ℝ) (n : ℕ) : ℝ :=
  let p3 := (Nat.choose n 3) * (p^3) * ((1 - p)^(n - 3))
  let p4 := (Nat.choose n 4) * (p^4)
  p3 + p4

theorem probability_at_least_3 (h : probability_hitting_at_least_3_of_4 0.8 4 = 0.8192) : 
   True :=
by trivial

end probability_at_least_3_l90_90458


namespace count_shapes_in_figure_l90_90184

-- Definitions based on the conditions
def firstLayerTriangles : Nat := 3
def secondLayerSquares : Nat := 2
def thirdLayerLargeTriangle : Nat := 1
def totalSmallTriangles := firstLayerTriangles
def totalLargeTriangles := thirdLayerLargeTriangle
def totalTriangles := totalSmallTriangles + totalLargeTriangles
def totalSquares := secondLayerSquares

-- Lean 4 statement to prove the problem
theorem count_shapes_in_figure : totalTriangles = 4 ∧ totalSquares = 2 :=
by {
  -- The proof is not required, so we use sorry to skip it.
  sorry
}

end count_shapes_in_figure_l90_90184


namespace exponentiation_rule_l90_90318

variable {a b : ℝ}

theorem exponentiation_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end exponentiation_rule_l90_90318


namespace smallest_a1_l90_90033

noncomputable def a_seq (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 13 * a (n - 1) - 2 * n

noncomputable def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ i, a i > 0

theorem smallest_a1 : ∃ a : ℕ → ℝ, a_seq a ∧ positive_sequence a ∧ a 1 = 13 / 36 :=
by
  sorry

end smallest_a1_l90_90033


namespace sum_of_first_five_primes_with_units_digit_3_l90_90983

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90983


namespace total_movies_purchased_l90_90101

theorem total_movies_purchased (x : ℕ) (h1 : 17 * x > 0) (h2 : 4 * x > 0) (h3 : 4 * x - 4 > 0) :
  (17 * x) / (4 * x - 4) = 9 / 2 → 17 * x + 4 * x = 378 :=
by 
  intro hab
  sorry

end total_movies_purchased_l90_90101


namespace bacteria_growth_rate_l90_90833

-- Define the existence of the growth rate and the initial amount of bacteria
variable (B : ℕ → ℝ) (B0 : ℝ) (r : ℝ)

-- State the conditions from the problem
axiom bacteria_growth_model : ∀ t : ℕ, B t = B0 * r ^ t
axiom day_30_full : B 30 = B0 * r ^ 30
axiom day_26_sixteenth : B 26 = (1 / 16) * B 30

-- Theorem stating that the growth rate r of the bacteria each day is 2
theorem bacteria_growth_rate : r = 2 := by
  sorry

end bacteria_growth_rate_l90_90833


namespace triangle_ratio_l90_90195

-- Define the conditions and the main theorem statement
theorem triangle_ratio (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h_eq : b * Real.cos C + c * Real.cos B = 2 * b) 
  (h_law_sines_a : a = 2 * b * Real.sin B / Real.sin A) 
  (h_angles : A + B + C = Real.pi) :
  b / a = 1 / 2 :=
by 
  sorry

end triangle_ratio_l90_90195


namespace units_digit_2_pow_2130_l90_90395

theorem units_digit_2_pow_2130 : (Nat.pow 2 2130) % 10 = 4 :=
by sorry

end units_digit_2_pow_2130_l90_90395


namespace test_total_questions_l90_90902

theorem test_total_questions (total_points : ℕ) (num_5_point_questions : ℕ) (points_per_5_point_question : ℕ) (points_per_10_point_question : ℕ) : 
  total_points = 200 → 
  num_5_point_questions = 20 → 
  points_per_5_point_question = 5 → 
  points_per_10_point_question = 10 → 
  (total_points = (num_5_point_questions * points_per_5_point_question) + 
    ((total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) * points_per_10_point_question) →
  (num_5_point_questions + (total_points - (num_5_point_questions * points_per_5_point_question)) / points_per_10_point_question) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end test_total_questions_l90_90902


namespace max_students_before_new_year_l90_90750

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l90_90750


namespace number_of_distinct_d_l90_90207

noncomputable def calculateDistinctValuesOfD (u v w x : ℂ) (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x) : ℕ := 
by
  sorry

theorem number_of_distinct_d (u v w x : ℂ) (h : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
    (h_eqs : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
             (z - (d * u)) * (z - (d * v)) * (z - (d * w)) * (z - (d * x))) : 
    calculateDistinctValuesOfD u v w x h = 4 :=
by
  sorry

end number_of_distinct_d_l90_90207


namespace sin_neg_pi_over_three_l90_90161

theorem sin_neg_pi_over_three : Real.sin (-Real.pi / 3) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_pi_over_three_l90_90161


namespace min_distance_is_18_l90_90357

noncomputable def minimize_distance (a b c d : ℝ) : ℝ := (a - c) ^ 2 + (b - d) ^ 2

theorem min_distance_is_18 (a b c d : ℝ) (h1 : b = a - 2 * Real.exp a) (h2 : c + d = 4) :
  minimize_distance a b c d = 18 :=
sorry

end min_distance_is_18_l90_90357


namespace sara_has_total_quarters_l90_90046

-- Define the number of quarters Sara originally had
def original_quarters : ℕ := 21

-- Define the number of quarters Sara's dad gave her
def added_quarters : ℕ := 49

-- Define the total number of quarters Sara has now
def total_quarters : ℕ := original_quarters + added_quarters

-- Prove that the total number of quarters is 70
theorem sara_has_total_quarters : total_quarters = 70 := by
  -- This is where the proof would go
  sorry

end sara_has_total_quarters_l90_90046


namespace goal_l90_90124

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l90_90124


namespace normal_prob_l90_90409

open Real
open ProbabilityTheory

noncomputable def normal_dist := Normal(1, σ^2)

theorem normal_prob :
  (P({x | 0 < x ∧ x < 1} : Set Real) = 0.3) :=
sorry

end normal_prob_l90_90409


namespace B_work_days_l90_90608

theorem B_work_days
  (A_work_rate : ℝ) (B_work_rate : ℝ) (A_days_worked : ℝ) (B_days_worked : ℝ)
  (total_work : ℝ) (remaining_work : ℝ) :
  A_work_rate = 1 / 15 →
  B_work_rate = total_work / 18 →
  A_days_worked = 5 →
  remaining_work = total_work - A_work_rate * A_days_worked →
  B_days_worked = 12 →
  remaining_work = B_work_rate * B_days_worked →
  total_work = 1 →
  B_days_worked = 12 →
  B_work_rate = total_work / 18 →
  B_days_alone = total_work / B_work_rate →
  B_days_alone = 18 := 
by
  intro hA_work_rate hB_work_rate hA_days_worked hremaining_work hB_days_worked hremaining_work_eq htotal_work hB_days_worked_again hsry_mul_inv hB_days_we_alone_eq
  sorry

end B_work_days_l90_90608


namespace smaller_solid_volume_l90_90937

noncomputable def cube_edge_length : ℝ := 2

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def D := point 0 0 0
def M := point 1 2 0
def N := point 2 0 1

-- Define the condition for the plane that passes through D, M, and N
def plane (p r q : ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  let (px, py, pz) := p
  let (rx, ry, rz) := r
  let (qx, qy, qz) := q
  2 * x - 4 * y - 8 * z = 0

-- Predicate to test if point is on a plane
def on_plane (pt : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := pt
  plane D M N x y z

-- Volume of the smaller solid
theorem smaller_solid_volume :
  ∃ V : ℝ, V = 1 / 6 :=
by
  sorry

end smaller_solid_volume_l90_90937


namespace joel_strawberries_area_l90_90695

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end joel_strawberries_area_l90_90695


namespace day_of_week_in_100_days_l90_90425

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l90_90425


namespace total_blocks_l90_90011

theorem total_blocks (red_blocks yellow_blocks blue_blocks : ℕ) 
  (h1 : red_blocks = 18) 
  (h2 : yellow_blocks = red_blocks + 7) 
  (h3 : blue_blocks = red_blocks + 14) : 
  red_blocks + yellow_blocks + blue_blocks = 75 := 
by
  sorry

end total_blocks_l90_90011


namespace probability_diagonals_intersect_l90_90242

theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) :
  let total_diagonals := (nat.choose n 2) - n,
      total_pairs_of_diagonals := nat.choose total_diagonals 2,
      intersecting_pairs := nat.choose n 4 in
  total_pairs_of_diagonals > 0 ∧ (intersecting_pairs : ℚ) / total_pairs_of_diagonals = 14 / 39 :=
by
  let total_diagonals := (nat.choose n 2) - n
  let total_pairs_of_diagonals := nat.choose total_diagonals 2
  let intersecting_pairs := nat.choose n 4
  have total_diagonals_eq : total_diagonals = 27 := by rw [hn]; norm_num
  have total_pairs_of_diagonals_eq : total_pairs_of_diagonals = 351 := by rw [total_diagonals_eq]; norm_num
  have intersecting_pairs_eq : intersecting_pairs = 126 := by rw [hn]; norm_num
  refine ⟨_, _⟩; sorry

end probability_diagonals_intersect_l90_90242


namespace proof_problem_l90_90997

theorem proof_problem (a : ℝ)
  (h1 : ∀ x : ℝ, ⟨ x ⟩ = x - ⌊ x ⌋) 
  (h2 : a > 0)
  (h3 : ⟨ a⁻¹ ⟩ = ⟨ a^2 ⟩) 
  (h4 : 2 < a^2 ∧ a^2 < 3) :
  a^12 - 144 * a⁻¹ = 233 :=
by sorry

end proof_problem_l90_90997


namespace min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l90_90601

-- Problem (Ⅰ)
theorem min_value_f1 (x : ℝ) (h : x > 0) : (12 / x + 3 * x) ≥ 12 :=
sorry

theorem min_value_f1_achieved : (12 / 2 + 3 * 2) = 12 :=
by norm_num

-- Problem (Ⅱ)
theorem max_value_f2 (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

theorem max_value_f2_achieved : (1 / 6) * (1 - 3 * (1 / 6)) = 1 / 12 :=
by norm_num

end min_value_f1_min_value_f1_achieved_max_value_f2_max_value_f2_achieved_l90_90601


namespace train_speed_in_km_per_hr_l90_90465

-- Definitions of conditions
def time_to_cross_pole := 9 -- seconds
def length_of_train := 120 -- meters

-- Function to convert speed from m/s to km/hr
def convert_m_per_s_to_km_per_hr (speed_m_per_s : ℕ) : ℕ := speed_m_per_s * 3600 / 1000

-- Main theorem statement
theorem train_speed_in_km_per_hr :
  let speed_m_per_s := length_of_train / time_to_cross_pole in
  convert_m_per_s_to_km_per_hr speed_m_per_s = 48 :=
by
  -- Proof will be filled out here
  sorry

end train_speed_in_km_per_hr_l90_90465


namespace find_positive_real_x_l90_90164

noncomputable def positive_solution :=
  ∃ (x : ℝ), (1/3) * (4 * x^2 - 2) = (x^2 - 75 * x - 15) * (x^2 + 50 * x + 10) ∧ x > 0

theorem find_positive_real_x :
  positive_solution ↔ ∃ (x : ℝ), x = (75 + Real.sqrt 5693) / 2 :=
by sorry

end find_positive_real_x_l90_90164


namespace real_number_solution_l90_90590

theorem real_number_solution : ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 :=
by
  sorry

end real_number_solution_l90_90590


namespace bella_started_with_136_candies_l90_90929

/-
Theorem:
Bella started with 136 candies.
-/

-- define the initial number of candies
variable (x : ℝ)

-- define the conditions
def condition1 : Prop := (x / 2 - 3 / 4) - 5 = 9
def condition2 : Prop := x = 136

-- structure the proof statement 
theorem bella_started_with_136_candies : condition1 x -> condition2 x :=
by
  sorry

end bella_started_with_136_candies_l90_90929


namespace find_c_value_l90_90639

variable {x: ℝ}

theorem find_c_value (d e c : ℝ) (h₁ : 6 * d = 18) (h₂ : -15 + 6 * e = -5)
(h₃ : (10 / 3) * c = 15) :
  c = 4.5 :=
by
  sorry

end find_c_value_l90_90639


namespace camila_weeks_needed_l90_90321

/--
Camila has only gone hiking 7 times.
Amanda has gone on 8 times as many hikes as Camila.
Steven has gone on 15 more hikes than Amanda.
Camila plans to go on 4 hikes a week.

Prove that it will take Camila 16 weeks to achieve her goal of hiking as many times as Steven.
-/
noncomputable def hikes_needed_to_match_steven : ℕ :=
  let camila_hikes := 7
  let amanda_hikes := 8 * camila_hikes
  let steven_hikes := amanda_hikes + 15
  let additional_hikes_needed := steven_hikes - camila_hikes
  additional_hikes_needed / 4

theorem camila_weeks_needed : hikes_needed_to_match_steven = 16 := 
  sorry

end camila_weeks_needed_l90_90321


namespace fg_of_3_eq_79_l90_90366

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l90_90366


namespace negation_of_proposition_l90_90065

open Real

theorem negation_of_proposition (P : ∀ x : ℝ, sin x ≥ 1) :
  ∃ x : ℝ, sin x < 1 :=
sorry

end negation_of_proposition_l90_90065


namespace other_pencil_length_l90_90007

-- Definitions based on the conditions identified in a)
def pencil1_length : Nat := 12
def total_length : Nat := 24

-- Problem: Prove that the length of the other pencil (pencil2) is 12 cubes.
theorem other_pencil_length : total_length - pencil1_length = 12 := by 
  sorry

end other_pencil_length_l90_90007


namespace decreasing_interval_g_min_value_a_no_zeros_l90_90669

open Real

-- Define the functions from the given conditions
noncomputable def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * log x
noncomputable def g (a x : ℝ) : ℝ := f a x + x

-- Lean statement for the first problem
theorem decreasing_interval_g :
  ∃ (a : ℝ), (∀ x ∈ Ioo 0 2, (3 - a - 2 / x) < 0) :=
sorry

-- Lean statement for the second problem
theorem min_value_a_no_zeros :
  ∃ (a : ℝ), (a ≥ 2 - 4 * log 2) ∧ (∀ x ∈ Ioo 0 (1/2), f a x > 0) :=
sorry

end decreasing_interval_g_min_value_a_no_zeros_l90_90669


namespace utility_bills_total_l90_90852

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l90_90852


namespace problem_part1_problem_part2_l90_90446

open Complex

noncomputable def E1 := ((1 + I)^2 / (1 + 2 * I)) + ((1 - I)^2 / (2 - I))

theorem problem_part1 : E1 = (6 / 5) - (2 / 5) * I :=
by
  sorry

theorem problem_part2 (x y : ℝ) (h1 : (x / 2) + (y / 5) = 1) (h2 : (x / 2) + (2 * y / 5) = 3) : x = -2 ∧ y = 10 :=
by
  sorry

end problem_part1_problem_part2_l90_90446


namespace sum_of_squares_of_roots_l90_90232

theorem sum_of_squares_of_roots (x1 x2 : ℝ) (h1 : 2 * x1^2 + 5 * x1 - 12 = 0) (h2 : 2 * x2^2 + 5 * x2 - 12 = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 = 73 / 4 :=
sorry

end sum_of_squares_of_roots_l90_90232


namespace find_integer_l90_90801

theorem find_integer (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.cos (n * Real.pi / 180) = Real.sin (312 * Real.pi / 180)) :
  n = 42 :=
by
  sorry

end find_integer_l90_90801


namespace matrix_pow_50_l90_90030

open Matrix

-- Define the given matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 2; -8, -5]

-- Define the expected result for C^50
def C_50 : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-199, -100; 400, 199]

-- Proposition asserting that C^50 equals the given result matrix
theorem matrix_pow_50 :
  C ^ 50 = C_50 := 
  by
  sorry

end matrix_pow_50_l90_90030


namespace pastry_trick_l90_90285

theorem pastry_trick (fillings : Fin 10) (n : ℕ) :
  ∃ n, (n = 36 ∧ ∀ remaining_pastries, 
    (remaining_pastries.length = 45 - n) → 
    (∃ remaining_filling ∈ fillings, true)) := 
sorry

end pastry_trick_l90_90285


namespace a_gt_b_l90_90049

noncomputable def a (R : Type*) [OrderedRing R] := {x : R // 0 < x ∧ x ^ 3 = x + 1}
noncomputable def b (R : Type*) [OrderedRing R] (a : R) := {y : R // 0 < y ∧ y ^ 6 = y + 3 * a}

theorem a_gt_b (R : Type*) [OrderedRing R] (a_pos_real : a R) (b_pos_real : b R (a_pos_real.val)) : a_pos_real.val > b_pos_real.val :=
sorry

end a_gt_b_l90_90049


namespace reservoir_water_level_at_6_pm_l90_90861

/-
  Initial conditions:
  - initial_water_level: Water level at 8 a.m.
  - increase_rate: Rate of increase in water level from 8 a.m. to 12 p.m.
  - decrease_rate: Rate of decrease in water level from 12 p.m. to 6 p.m.
  - start_increase_time: Starting time of increase (in hours from 8 a.m.)
  - end_increase_time: Ending time of increase (in hours from 8 a.m.)
  - start_decrease_time: Starting time of decrease (in hours from 12 p.m.)
  - end_decrease_time: Ending time of decrease (in hours from 12 p.m.)
-/
def initial_water_level : ℝ := 45
def increase_rate : ℝ := 0.6
def decrease_rate : ℝ := 0.3
def start_increase_time : ℝ := 0 -- 8 a.m. in hours from 8 a.m.
def end_increase_time : ℝ := 4 -- 12 p.m. in hours from 8 a.m.
def start_decrease_time : ℝ := 0 -- 12 p.m. in hours from 12 p.m.
def end_decrease_time : ℝ := 6 -- 6 p.m. in hours from 12 p.m.

theorem reservoir_water_level_at_6_pm :
  initial_water_level
  + (end_increase_time - start_increase_time) * increase_rate
  - (end_decrease_time - start_decrease_time) * decrease_rate
  = 45.6 :=
by
  sorry

end reservoir_water_level_at_6_pm_l90_90861


namespace intersection_eq_l90_90505

open Set

-- Define the sets M and N
def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {-3, -1, 1, 3, 5}

-- The goal is to prove that M ∩ N = {-1, 1, 3}
theorem intersection_eq : M ∩ N = {-1, 1, 3} :=
  sorry

end intersection_eq_l90_90505


namespace regression_decrease_by_three_l90_90819

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 3 * x

-- Prove that when the explanatory variable increases by 1 unit, the predicted variable decreases by 3 units
theorem regression_decrease_by_three : ∀ x : ℝ, regression_equation (x + 1) = regression_equation x - 3 :=
by
  intro x
  unfold regression_equation
  sorry

end regression_decrease_by_three_l90_90819


namespace common_solution_l90_90489

-- Define the conditions of the equations as hypotheses
variables (x y : ℝ)

-- First equation
def eq1 := x^2 + y^2 = 4

-- Second equation
def eq2 := x^2 = 4*y - 8

-- Proof statement: If there exists real numbers x and y such that both equations hold,
-- then y must be equal to 2.
theorem common_solution (x y : ℝ) (h1 : eq1 x y) (h2 : eq2 x y) : y = 2 :=
sorry

end common_solution_l90_90489


namespace ratio_of_volumes_l90_90460

noncomputable def inscribedSphereVolume (s : ℝ) : ℝ := (4 / 3) * Real.pi * (s / 2) ^ 3

noncomputable def cubeVolume (s : ℝ) : ℝ := s ^ 3

theorem ratio_of_volumes (s : ℝ) (h : s > 0) :
  inscribedSphereVolume s / cubeVolume s = Real.pi / 6 :=
by
  sorry

end ratio_of_volumes_l90_90460


namespace day_100_days_from_friday_l90_90434

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l90_90434


namespace number_of_trees_in_yard_l90_90826

theorem number_of_trees_in_yard :
  ∀ (yard_length tree_distance : ℕ), yard_length = 360 ∧ tree_distance = 12 → 
  (yard_length / tree_distance + 1 = 31) :=
by
  intros yard_length tree_distance h
  have h1 : yard_length = 360 := h.1
  have h2 : tree_distance = 12 := h.2
  sorry

end number_of_trees_in_yard_l90_90826


namespace minimize_distance_l90_90541

noncomputable def f : ℝ → ℝ := λ x => x ^ 2
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def y : ℝ → ℝ := λ x => f x - g x

theorem minimize_distance (t : ℝ) (ht : t = Real.sqrt 2 / 2) :
  ∀ x > 0, y x ≥ y (Real.sqrt 2 / 2) := sorry

end minimize_distance_l90_90541


namespace rate_of_interest_is_4_l90_90415

theorem rate_of_interest_is_4 (R : ℝ) : 
  ∀ P : ℝ, ∀ T : ℝ, P = 3000 → T = 5 → (P * R * T / 100 = P - 2400) → R = 4 :=
by
  sorry

end rate_of_interest_is_4_l90_90415


namespace german_team_goals_l90_90112

/-- Proof that the German team could have scored 11, 12, 14, 16, or 17 goals given the conditions
    stated by the three journalists, with exactly two of their statements being true. -/
theorem german_team_goals (x : ℕ) :
  (10 < x ∧ x < 17 → 11 < x ∧ x < 18 → x % 2 = 1 → false) →
  (10 < x ∧ x < 17 ∧ ¬(11 < x ∧ x < 18) ∧ (x % 2 = 1) → false) →
  (¬(10 < x ∧ x < 17) ∧ 11 < x ∧ x < 18 ∧ (x % 2 = 1) → false) →
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end german_team_goals_l90_90112


namespace geometric_sequence_sum_l90_90520

-- Define the sequence and state the conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

-- The mathematical problem rewritten in Lean 4 statement
theorem geometric_sequence_sum (a : ℕ → ℝ) (s : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : s 2 = 7)
  (h3 : s 6 = 91)
  : ∃ s_4 : ℝ, s_4 = 28 :=
by
  sorry

end geometric_sequence_sum_l90_90520


namespace limit_log_div_x_alpha_l90_90797

open Real

theorem limit_log_div_x_alpha (α : ℝ) (hα : α > 0) :
  (Filter.Tendsto (fun x => (log x) / (x^α)) Filter.atTop (nhds 0)) :=
by
  sorry

end limit_log_div_x_alpha_l90_90797


namespace total_seeds_eaten_l90_90628

-- Definitions and conditions
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds
def first_four_players_seeds : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds
def average_seeds : ℕ := first_four_players_seeds / 4
def fifth_player_seeds : ℕ := average_seeds

-- Statement to prove
theorem total_seeds_eaten :
  first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds + fifth_player_seeds = 475 :=
by {
  sorry
}

end total_seeds_eaten_l90_90628


namespace sum_of_common_ratios_l90_90843

variable {k p r : ℝ}

theorem sum_of_common_ratios (h1 : k ≠ 0)
                             (h2 : p ≠ r)
                             (h3 : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
                             p + r = 5 := 
by
  sorry

end sum_of_common_ratios_l90_90843


namespace minimum_work_to_remove_cube_l90_90620

namespace CubeBuoyancy

def edge_length (ℓ : ℝ) := ℓ = 0.30 -- in meters
def wood_density (ρ : ℝ) := ρ = 750  -- in kg/m^3
def water_density (ρ₀ : ℝ) := ρ₀ = 1000 -- in kg/m^3

theorem minimum_work_to_remove_cube 
  {ℓ ρ ρ₀ : ℝ} 
  (h₁ : edge_length ℓ)
  (h₂ : wood_density ρ)
  (h₃ : water_density ρ₀) : 
  ∃ W : ℝ, W = 22.8 := 
sorry

end CubeBuoyancy

end minimum_work_to_remove_cube_l90_90620


namespace snow_at_least_once_l90_90567

theorem snow_at_least_once (p_snow : ℝ) (h : p_snow = 3/4) : 
  let p_no_snow := 1 - p_snow in
  let p_no_snow_4_days := p_no_snow ^ 4 in
  let p_snow_at_least_once := 1 - p_no_snow_4_days in
  p_snow_at_least_once = 255/256 :=
by
  sorry

end snow_at_least_once_l90_90567


namespace integer_pairs_solution_l90_90483

theorem integer_pairs_solution (x y : ℤ) (k : ℤ) :
  2 * x^2 - 6 * x * y + 3 * y^2 = -1 ↔
  ∃ n : ℤ, x = (2 + Real.sqrt 3)^k / 2 ∨ x = -(2 + Real.sqrt 3)^k / 2 ∧
           y = x + (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) ∨ 
           y = x - (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) :=
sorry

end integer_pairs_solution_l90_90483


namespace largest_real_x_is_120_over_11_l90_90340

noncomputable def largest_real_x (x : ℝ) : Prop :=
  floor x / x = 11 / 12

theorem largest_real_x_is_120_over_11 :
  ∃ x, largest_real_x x ∧ x ≤ 120 / 11 :=
sorry

end largest_real_x_is_120_over_11_l90_90340


namespace bob_should_give_l90_90780

theorem bob_should_give (alice_paid bob_paid charlie_paid : ℕ)
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_charlie : charlie_paid = 180) :
  bob_paid - (120 + 150 + 180) / 3 = 0 := 
by
  sorry

end bob_should_give_l90_90780


namespace professors_initial_count_l90_90016

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l90_90016


namespace sum_of_first_five_primes_units_digit_3_l90_90970

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l90_90970


namespace width_of_room_l90_90412

theorem width_of_room (length : ℝ) (cost_rate : ℝ) (total_cost : ℝ) (width : ℝ)
  (h1 : length = 5.5)
  (h2 : cost_rate = 800)
  (h3 : total_cost = 16500)
  (h4 : width = total_cost / cost_rate / length) : width = 3.75 :=
by
  sorry

end width_of_room_l90_90412


namespace fraction_ordering_l90_90634

theorem fraction_ordering :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  (b < c) ∧ (c < a) :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  sorry

end fraction_ordering_l90_90634


namespace problem_value_l90_90439

theorem problem_value:
  3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 :=
by sorry

end problem_value_l90_90439


namespace quadrilateral_area_l90_90747

def diagonal : ℝ := 15
def offset1 : ℝ := 6
def offset2 : ℝ := 4

theorem quadrilateral_area :
  (1/2) * diagonal * (offset1 + offset2) = 75 :=
by 
  sorry

end quadrilateral_area_l90_90747


namespace trick_proof_l90_90287

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l90_90287


namespace B_participated_Huangmei_Opera_l90_90714

-- Definitions using given conditions
def participated_A (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∨ 
  (c "Huangmei Flower Picking" ∧ ¬ c "Yue Family Boxing")

def participated_B (c : String → Prop) : Prop :=
  (c "Huangmei Opera" ∧ ¬ c "Huangmei Flower Picking") ∨
  (c "Yue Family Boxing" ∧ ¬ c "Huangmei Flower Picking")

def participated_C (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∧ c "Huangmei Flower Picking" ∧ c "Yue Family Boxing" ->
  (c "Huangmei Opera" ∨ c "Huangmei Flower Picking" ∨ c "Yue Family Boxing")

-- Proving the special class that B participated in
theorem B_participated_Huangmei_Opera :
  ∃ c : String → Prop, participated_A c ∧ participated_B c ∧ participated_C c → c "Huangmei Opera" :=
by
  -- proof steps would go here
  sorry

end B_participated_Huangmei_Opera_l90_90714


namespace green_disks_more_than_blue_l90_90159

theorem green_disks_more_than_blue (total_disks : ℕ) (b y g : ℕ) (h1 : total_disks = 108)
  (h2 : b / y = 3 / 7) (h3 : b / g = 3 / 8) : g - b = 30 :=
by
  sorry

end green_disks_more_than_blue_l90_90159


namespace maximize_c_l90_90258

theorem maximize_c (c d e : ℤ) (h1 : 5 * c + (d - 12)^2 + e^3 = 235) (h2 : c < d) : c ≤ 22 :=
sorry

end maximize_c_l90_90258


namespace function_equality_l90_90167

theorem function_equality (f : ℝ → ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (f ( (x + 1) / x ) = (x^2 + 1) / x^2 + 1 / x) ↔ (f x = x^2 - x + 1) :=
by
  sorry

end function_equality_l90_90167


namespace max_non_attacking_mammonths_is_20_l90_90059

def mamonth_attacking_diagonal_count (b: board) (m: mamonth): ℕ := 
    sorry -- define the function to count attacking diagonals of a given mammoth on the board

def max_non_attacking_mamonths_board (b: board) : ℕ :=
    sorry -- function to calculate max non-attacking mammonths given a board setup

theorem max_non_attacking_mammonths_is_20 : 
  ∀ (b : board), (max_non_attacking_mamonths_board b) ≤ 20 :=
by
  sorry

end max_non_attacking_mammonths_is_20_l90_90059


namespace log_simplification_l90_90713

open Real

theorem log_simplification (a b d e z y : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (ha : a ≠ 0) (hz : z ≠ 0) (hy : y ≠ 0) :
  log (a / b) + log (b / e) + log (e / d) - log (az / dy) = log (dy / z) :=
by
  sorry

end log_simplification_l90_90713


namespace age_difference_l90_90467

variable (A : ℕ) -- Albert's age
variable (B : ℕ) -- Albert's brother's age
variable (F : ℕ) -- Father's age
variable (M : ℕ) -- Mother's age

def age_conditions : Prop :=
  (B = A - 2) ∧ (F = A + 48) ∧ (M = B + 46)

theorem age_difference (h : age_conditions A B F M) : F - M = 4 :=
by
  sorry

end age_difference_l90_90467


namespace parabola_focus_l90_90645

-- Definitions and conditions from the original problem
def parabola_eq (x y : ℝ) : Prop := x^2 = (1/2) * y 

-- Define the problem to prove the coordinates of the focus
theorem parabola_focus (x y : ℝ) (h : parabola_eq x y) : (x = 0 ∧ y = 1/8) :=
sorry

end parabola_focus_l90_90645


namespace sum_of_first_five_primes_with_units_digit_3_l90_90955

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90955


namespace min_max_ab_bc_cd_de_l90_90704

theorem min_max_ab_bc_cd_de (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e) (h_sum : a + b + c + d + e = 2018) : 
  ∃ a b c d e, 
  a > 0 ∧ 
  b > 0 ∧ 
  c > 0 ∧ 
  d > 0 ∧ 
  e > 0 ∧ 
  a + b + c + d + e = 2018 ∧ 
  ∀ M, M = max (max (max (a + b) (b + c)) (max (c + d) (d + e))) ↔ M = 673  :=
sorry

end min_max_ab_bc_cd_de_l90_90704


namespace minimize_b_plus_c_l90_90998

theorem minimize_b_plus_c (a b c : ℝ) (h1 : 0 < a)
  (h2 : ∀ x, (y : ℝ) = a * x^2 + b * x + c)
  (h3 : ∀ x, (yr : ℝ) = a * (x + 2)^2 + (a - 1)^2) :
  a = 1 :=
by
  sorry

end minimize_b_plus_c_l90_90998


namespace jessica_quarters_l90_90531

theorem jessica_quarters (original_borrowed : ℕ) (quarters_borrowed : ℕ) 
  (H1 : original_borrowed = 8)
  (H2 : quarters_borrowed = 3) : 
  original_borrowed - quarters_borrowed = 5 := sorry

end jessica_quarters_l90_90531


namespace integer_solutions_l90_90798

theorem integer_solutions (x y k : ℤ) :
  21 * x + 48 * y = 6 ↔ ∃ k : ℤ, x = -2 + 16 * k ∧ y = 1 - 7 * k :=
by
  sorry

end integer_solutions_l90_90798


namespace german_team_goals_l90_90129

theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : x % 2 = 1) 
  (h_correct : (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)):
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := 
by {
  -- Proof to be filled in
  sorry
}

end german_team_goals_l90_90129


namespace faith_change_l90_90333

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l90_90333


namespace sum_of_integers_l90_90566

theorem sum_of_integers (n m : ℕ) (h1 : n * (n + 1) = 120) (h2 : (m - 1) * m * (m + 1) = 120) : 
  (n + (n + 1) + (m - 1) + m + (m + 1)) = 36 :=
by
  sorry

end sum_of_integers_l90_90566


namespace claire_needs_80_tiles_l90_90789

def room_length : ℕ := 14
def room_width : ℕ := 18
def border_width : ℕ := 2
def small_tile_side : ℕ := 1
def large_tile_side : ℕ := 3

def num_small_tiles : ℕ :=
  let perimeter_length := (2 * (room_width - 2 * border_width))
  let perimeter_width := (2 * (room_length - 2 * border_width))
  let corner_tiles := (2 * border_width) * 4
  perimeter_length + perimeter_width + corner_tiles

def num_large_tiles : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  Nat.ceil (inner_area / (large_tile_side * large_tile_side))

theorem claire_needs_80_tiles : num_small_tiles + num_large_tiles = 80 :=
by sorry

end claire_needs_80_tiles_l90_90789


namespace num_triangles_with_perimeter_9_l90_90511

-- Definitions for side lengths and their conditions
def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 9

-- The main theorem
theorem num_triangles_with_perimeter_9 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a ≤ b ∧ b ≤ c → is_triangle a b c ↔ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end num_triangles_with_perimeter_9_l90_90511


namespace concentrate_amount_l90_90311

def parts_concentrate : ℤ := 1
def parts_water : ℤ := 5
def part_ratio : ℤ := parts_concentrate + parts_water -- Total parts
def servings : ℤ := 375
def volume_per_serving : ℤ := 150
def total_volume : ℤ := servings * volume_per_serving -- Total volume of orange juice
def volume_per_part : ℤ := total_volume / part_ratio -- Volume per part of mixture

theorem concentrate_amount :
  volume_per_part = 9375 :=
by
  sorry

end concentrate_amount_l90_90311


namespace B_speaks_truth_60_l90_90102

variable (P_A P_B P_A_and_B : ℝ)

-- Given conditions
def A_speaks_truth_85 : Prop := P_A = 0.85
def both_speak_truth_051 : Prop := P_A_and_B = 0.51

-- Solution condition
noncomputable def B_speaks_truth_percentage : ℝ := P_A_and_B / P_A

-- Statement to prove
theorem B_speaks_truth_60 (hA : A_speaks_truth_85 P_A) (hAB : both_speak_truth_051 P_A_and_B) : B_speaks_truth_percentage P_A_and_B P_A = 0.6 :=
by
  rw [A_speaks_truth_85] at hA
  rw [both_speak_truth_051] at hAB
  unfold B_speaks_truth_percentage
  sorry

end B_speaks_truth_60_l90_90102


namespace total_interest_received_l90_90094

def principal_B := 5000
def principal_C := 3000
def rate := 9
def time_B := 2
def time_C := 4
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℕ := P * R * T / 100

theorem total_interest_received :
  let SI_B := simple_interest principal_B rate time_B
  let SI_C := simple_interest principal_C rate time_C
  SI_B + SI_C = 1980 := 
by
  sorry

end total_interest_received_l90_90094


namespace proof_stmt_l90_90926

variable (a x y : ℝ)
variable (ha : a > 0) (hneq : a ≠ 1)

noncomputable def S (x : ℝ) := a^x - a^(-x)
noncomputable def C (x : ℝ) := a^x + a^(-x)

theorem proof_stmt :
  2 * S a (x + y) = S a x * C a y + C a x * S a y ∧
  2 * S a (x - y) = S a x * C a y - C a x * S a y :=
by sorry

end proof_stmt_l90_90926


namespace find_m_value_l90_90354

theorem find_m_value (m x y : ℝ) (hx : x = 2) (hy : y = -1) (h_eq : m * x - y = 3) : m = 1 :=
by
  sorry

end find_m_value_l90_90354


namespace rhombus_longer_diagonal_l90_90917

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end rhombus_longer_diagonal_l90_90917


namespace power_rule_example_l90_90316

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l90_90316


namespace probability_two_balls_same_color_l90_90990

/--
From a box containing 6 colored balls (3 red, 2 yellow, 1 blue), two balls are randomly drawn.

- A box contains 6 colored balls.
- The balls are distributed as: 3 red, 2 yellow, and 1 blue.
- Two balls are randomly drawn.

Prove that the probability that the two balls are of the same color is 4/15.
-/
theorem probability_two_balls_same_color :
  let total_ways := (nat.choose 6 2) in
  let same_color_ways := (nat.choose 3 2) + (nat.choose 2 2) in
  (same_color_ways : ℚ) / total_ways = 4/15 :=
by
  sorry

end probability_two_balls_same_color_l90_90990


namespace power_calculation_l90_90517

theorem power_calculation (y : ℝ) (h : 3^y = 81) : 3^(y + 3) = 2187 :=
sorry

end power_calculation_l90_90517


namespace camila_weeks_to_goal_l90_90323

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l90_90323


namespace sum_of_external_angles_le_360_sum_of_external_angles_eq_360_if_convex_polygon_l90_90547

-- Assume K is a bounded convex curve with finite corner points
section
variables {K : Type*} [convex K] [bounded K] [finite_corners K]

theorem sum_of_external_angles_le_360 (h : bounded_convex_curve K) :
  sum_of_external_angles K ≤ 360 := 
sorry

theorem sum_of_external_angles_eq_360_if_convex_polygon (h1 : bounded_convex_curve K) (h2 : sum_of_external_angles K = 360) :
  is_convex_polygon K :=
sorry

end sum_of_external_angles_le_360_sum_of_external_angles_eq_360_if_convex_polygon_l90_90547


namespace percentage_increase_l90_90609

theorem percentage_increase (regular_rate : ℝ) (regular_hours total_compensation total_hours_worked : ℝ)
  (h1 : regular_rate = 20)
  (h2 : regular_hours = 40)
  (h3 : total_compensation = 1000)
  (h4 : total_hours_worked = 45.714285714285715) :
  let overtime_hours := total_hours_worked - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_compensation - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  let percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100
  percentage_increase = 75 := 
by
  sorry

end percentage_increase_l90_90609


namespace initial_number_of_professors_l90_90025

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l90_90025


namespace goal_l90_90126

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l90_90126


namespace twice_minus_three_algebraic_l90_90947

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l90_90947


namespace area_of_rectangular_field_l90_90459

theorem area_of_rectangular_field (W L : ℕ) (hL : L = 10) (hFencing : 2 * W + L = 146) : W * L = 680 := by
  sorry

end area_of_rectangular_field_l90_90459


namespace tangent_line_eq_at_P_tangent_lines_through_P_l90_90447

-- Define the function and point of interest
def f (x : ℝ) : ℝ := x^3
def P : ℝ × ℝ := (1, 1)

-- State the first part: equation of the tangent line at (1, 1)
theorem tangent_line_eq_at_P : 
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ y = f x ∧ x = 1 → y = 3 * x - 2) :=
sorry

-- State the second part: equations of tangent lines passing through (1, 1)
theorem tangent_lines_through_P :
  (∀ x : ℝ, x = P.1 → (f x) = P.2) → 
  (∀ (x₀ y₀ : ℝ), y₀ = x₀^3 → 
  (x₀ ≠ 1 → ∃ k : ℝ,  k = 3 * (x₀)^2 → 
  (∀ x y : ℝ, y = k * (x - 1) + 1 ∧ y = f x₀ → y = y₀))) → 
  (∃ m b m' b' : ℝ, 
    (¬ ∀ x : ℝ, ∀ y : ℝ, (y = m *x + b ∧ y = 3 * x - 2) → y = m' * x + b') ∧ 
    ((m = 3 ∧ b = -2) ∧ (m' = 3/4 ∧ b' = 1/4))) :=
sorry

end tangent_line_eq_at_P_tangent_lines_through_P_l90_90447


namespace coin_toss_tails_count_l90_90591

theorem coin_toss_tails_count (flips : ℕ) (frequency_heads : ℝ) (h_flips : flips = 20) (h_frequency_heads : frequency_heads = 0.45) : 
  (20 : ℝ) * (1 - 0.45) = 11 := 
by
  sorry

end coin_toss_tails_count_l90_90591


namespace ratio_of_cream_l90_90532

theorem ratio_of_cream
  (joes_initial_coffee : ℕ := 20)
  (joe_cream_added : ℕ := 3)
  (joe_amount_drank : ℕ := 4)
  (joanns_initial_coffee : ℕ := 20)
  (joann_amount_drank : ℕ := 4)
  (joann_cream_added : ℕ := 3) :
  let joe_final_cream := (joe_cream_added - joe_amount_drank * (joe_cream_added / (joe_cream_added + joes_initial_coffee)))
  let joann_final_cream := joann_cream_added
  (joe_final_cream / joanns_initial_coffee + joann_cream_added = 15 / 23) :=
sorry

end ratio_of_cream_l90_90532


namespace total_cleaning_time_l90_90157

-- Definition for the problem conditions
def time_to_clean_egg (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ) : ℕ :=
  (num_eggs * seconds_per_egg) / seconds_per_minute

def time_to_clean_toilet_paper (minutes_per_roll : ℕ) (num_rolls : ℕ) : ℕ :=
  num_rolls * minutes_per_roll

-- Main statement to prove the total cleaning time
theorem total_cleaning_time
  (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ)
  (minutes_per_roll : ℕ) (num_rolls : ℕ) :
  seconds_per_egg = 15 →
  num_eggs = 60 →
  seconds_per_minute = 60 →
  minutes_per_roll = 30 →
  num_rolls = 7 →
  time_to_clean_egg seconds_per_egg num_eggs seconds_per_minute +
  time_to_clean_toilet_paper minutes_per_roll num_rolls = 225 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cleaning_time_l90_90157


namespace sum_of_first_five_primes_with_units_digit_3_l90_90977

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90977


namespace find_c_l90_90878

theorem find_c (c : ℝ) 
  (h : (⟨9, c⟩ : ℝ × ℝ) = (11/13 : ℝ) • ⟨-3, 2⟩) : 
  c = 19 :=
sorry

end find_c_l90_90878


namespace tourist_growth_rate_l90_90793

theorem tourist_growth_rate (F : ℝ) (x : ℝ) 
    (hMarch : F * 0.6 = 0.6 * F)
    (hApril : F * 0.6 * 0.5 = 0.3 * F)
    (hMay : 2 * F = 2 * F):
    (0.6 * 0.5 * (1 + x) = 2) :=
by
  sorry

end tourist_growth_rate_l90_90793


namespace interest_rate_difference_l90_90104

def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def si1 (R1 : ℕ) : ℕ := simple_interest 800 R1 10
def si2 (R2 : ℕ) : ℕ := simple_interest 800 R2 10

theorem interest_rate_difference (R1 R2 : ℕ) (h : si2 R2 = si1 R1 + 400) : R2 - R1 = 5 := 
by sorry

end interest_rate_difference_l90_90104


namespace german_team_goals_l90_90128

theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : x % 2 = 1) 
  (h_correct : (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)):
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := 
by {
  -- Proof to be filled in
  sorry
}

end german_team_goals_l90_90128


namespace samatha_tosses_five_coins_l90_90222

noncomputable def probability_at_least_one_head 
  (p.toss : ℕ → ℙ) 
  (h_independence : ∀ n m : ℕ, n ≠ m → ProbInd (p.toss n) (p.toss m))
  (h_tail_prob : ∀ n : ℕ, Pr (flip_tail (p.toss n)) = 1 / 2) : ℚ :=
  1 - (1/2)^5

theorem samatha_tosses_five_coins :
  let p.toss : ℕ → ℙ := flip_coin
  in probability_at_least_one_head p.toss (by sorry) (by sorry) = 31/32 :=
by
  sorry

end samatha_tosses_five_coins_l90_90222


namespace nonagon_diagonal_intersection_probability_l90_90252

-- Definitions based on conditions from part a)
def nonagon_diagonals_count (n : ℕ) : ℕ := (n * (n - 3)) / 2

def choose (n k : ℕ) : ℕ := nat.choose n k

-- The problem: Prove the probability that two chosen diagonals of a regular nonagon intersect inside the nonagon is 14/39
theorem nonagon_diagonal_intersection_probability : 
  let n := 9 in 
  let total_diagonals := nonagon_diagonals_count n in
  let total_ways_to_choose_diagonals := choose total_diagonals 2 in
  let ways_to_choose_intersecting_diagonals := choose n 4 in
  total_ways_to_choose_diagonals ≠ 0 →
  (ways_to_choose_intersecting_diagonals / total_ways_to_choose_diagonals : ℚ) = 14 / 39 :=
begin
  sorry -- Proof is omitted as per the instructions
end

end nonagon_diagonal_intersection_probability_l90_90252


namespace select_two_subsets_union_six_elements_l90_90031

def f (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * f (n - 1) - 1

theorem select_two_subsets_union_six_elements :
  f 6 = 365 :=
by
  sorry

end select_two_subsets_union_six_elements_l90_90031


namespace part1_part2_l90_90036

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 4 ≤ x ∧ x < 8 }
def B : Set ℝ := { x | 3 < x ∧ x < 7 }

theorem part1 :
  (A ∩ B = { x | 4 ≤ x ∧ x < 7 }) ∧
  ((U \ A) ∪ B = { x | x < 7 ∨ x ≥ 8 }) :=
by
  sorry
  
def C (t : ℝ) : Set ℝ := { x | x < t + 1 }

theorem part2 (t : ℝ) :
  (A ∩ C t = ∅) → (t ≤ 3 ∨ t ≥ 7) :=
by
  sorry

end part1_part2_l90_90036


namespace simplify_expression_l90_90712

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end simplify_expression_l90_90712


namespace utility_bills_total_correct_l90_90858

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l90_90858


namespace millimeters_of_78_74_inches_l90_90361

noncomputable def inchesToMillimeters (inches : ℝ) : ℝ :=
  inches * 25.4

theorem millimeters_of_78_74_inches :
  round (inchesToMillimeters 78.74) = 2000 :=
by
  -- This theorem should assert that converting 78.74 inches to millimeters and rounding to the nearest millimeter equals 2000
  sorry

end millimeters_of_78_74_inches_l90_90361


namespace largest_integral_x_l90_90486

theorem largest_integral_x (x : ℤ) : (2 / 7 : ℝ) < (x / 6) ∧ (x / 6) < (7 / 9) → x = 4 :=
by
  sorry

end largest_integral_x_l90_90486


namespace find_k_l90_90842

noncomputable def series (k : ℝ) : ℝ := ∑' n, (7 * n - 2) / k^n

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : series k = 17 / 2) : k = 17 / 7 :=
by
  sorry

end find_k_l90_90842


namespace bezdikov_population_l90_90374

variable (W M : ℕ) -- original number of women and men
variable (W_current M_current : ℕ) -- current number of women and men

theorem bezdikov_population (h1 : W = M + 30)
                          (h2 : W_current = W / 4)
                          (h3 : M_current = M - 196)
                          (h4 : W_current = M_current + 10) : W_current + M_current = 134 :=
by
  sorry

end bezdikov_population_l90_90374


namespace min_value_of_quadratic_function_l90_90081

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end min_value_of_quadratic_function_l90_90081


namespace min_sum_of_squares_l90_90537

theorem min_sum_of_squares (a b c d : ℤ) (h1 : a^2 ≠ b^2 ∧ a^2 ≠ c^2 ∧ a^2 ≠ d^2 ∧ b^2 ≠ c^2 ∧ b^2 ≠ d^2 ∧ c^2 ≠ d^2)
                           (h2 : (a * b + c * d)^2 + (a * d - b * c)^2 = 2004) :
  a^2 + b^2 + c^2 + d^2 = 2 * Int.sqrt 2004 :=
sorry

end min_sum_of_squares_l90_90537


namespace original_treadmill_price_l90_90396

-- Given conditions in Lean definitions
def discount_rate : ℝ := 0.30
def plate_cost : ℝ := 50
def num_plates : ℕ := 2
def total_paid : ℝ := 1045

noncomputable def treadmill_price :=
  let plate_total := num_plates * plate_cost
  let treadmill_discount := (1 - discount_rate)
  (total_paid - plate_total) / treadmill_discount

theorem original_treadmill_price :
  treadmill_price = 1350 := by
  sorry

end original_treadmill_price_l90_90396


namespace determinant_of_sine_matrix_is_zero_l90_90476

theorem determinant_of_sine_matrix_is_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![Real.sin 2, Real.sin 3, Real.sin 4],
      ![Real.sin 5, Real.sin 6, Real.sin 7],
      ![Real.sin 8, Real.sin 9, Real.sin 10]]
  Matrix.det M = 0 := 
by sorry

end determinant_of_sine_matrix_is_zero_l90_90476


namespace remove_terms_to_make_sum_l90_90442

theorem remove_terms_to_make_sum (a b c d e f : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/5) (h₃ : c = 1/7) (h₄ : d = 1/9) (h₅ : e = 1/11) (h₆ : f = 1/13) :
  a + b + c + d + e + f - e - f = 3/2 :=
by
  sorry

end remove_terms_to_make_sum_l90_90442


namespace find_f_neg2_l90_90173

noncomputable def f (x : ℝ) : ℝ := -2 * (x + 1) + 1

theorem find_f_neg2 : f (-2) = 3 := by
  sorry

end find_f_neg2_l90_90173


namespace min_cubes_to_fill_box_l90_90259

theorem min_cubes_to_fill_box :
  ∀ (a b c : ℕ), a = 30 → b = 40 → c = 50 → 
  let gcd := Nat.gcd (Nat.gcd a b) c in
  gcd = 10 → 
  (a * b * c) / (gcd * gcd * gcd) = 60 :=
by
  intros a b c ha hb hc h_gcd
  rw [ha, hb, hc, Nat.mul_div_cancel'] at h_gcd
  sorry

end min_cubes_to_fill_box_l90_90259


namespace hiker_walks_18_miles_on_first_day_l90_90770

noncomputable def miles_walked_first_day (h : ℕ) : ℕ := 3 * h

def total_miles_walked (h : ℕ) : ℕ := (3 * h) + (4 * (h - 1)) + (4 * h)

theorem hiker_walks_18_miles_on_first_day :
  (∃ h : ℕ, total_miles_walked h = 62) → miles_walked_first_day 6 = 18 :=
by
  sorry

end hiker_walks_18_miles_on_first_day_l90_90770


namespace pillows_from_feathers_l90_90214

def feathers_per_pound : ℕ := 300
def feathers_total : ℕ := 3600
def pounds_per_pillow : ℕ := 2

theorem pillows_from_feathers :
  (feathers_total / feathers_per_pound / pounds_per_pillow) = 6 :=
by
  sorry

end pillows_from_feathers_l90_90214


namespace total_apples_purchased_l90_90777

theorem total_apples_purchased (M : ℝ) (T : ℝ) (W : ℝ) 
    (hM : M = 15.5)
    (hT : T = 3.2 * M)
    (hW : W = 1.05 * T) :
    M + T + W = 117.18 := by
  sorry

end total_apples_purchased_l90_90777


namespace largest_multiple_of_11_neg_greater_minus_210_l90_90896

theorem largest_multiple_of_11_neg_greater_minus_210 :
  ∃ (x : ℤ), x % 11 = 0 ∧ -x < -210 ∧ ∀ y, y % 11 = 0 ∧ -y < -210 → y ≤ x :=
sorry

end largest_multiple_of_11_neg_greater_minus_210_l90_90896


namespace fifteenth_term_geometric_seq_l90_90256

theorem fifteenth_term_geometric_seq :
  let a₁ := 12
  let r := 1 / 3
  let n := 15
  let aₙ := λ (a₁ r : ℚ) n, a₁ * r^(n - 1)
  aₙ a₁ r n = 12 / 4782969 :=
by
  sorry

end fifteenth_term_geometric_seq_l90_90256


namespace negation_of_positive_l90_90565

def is_positive (x : ℝ) : Prop := x > 0
def is_non_positive (x : ℝ) : Prop := x ≤ 0

theorem negation_of_positive (a b c : ℝ) :
  (¬ (is_positive a ∨ is_positive b ∨ is_positive c)) ↔ (is_non_positive a ∧ is_non_positive b ∧ is_non_positive c) :=
by
  sorry

end negation_of_positive_l90_90565


namespace sum_of_first_five_prime_units_digit_3_l90_90959

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l90_90959


namespace quadratic_complete_square_l90_90561

theorem quadratic_complete_square : ∃ k : ℤ, ∀ x : ℤ, x^2 + 8*x + 22 = (x + 4)^2 + k :=
by
  use 6
  sorry

end quadratic_complete_square_l90_90561


namespace rectangle_perimeter_of_divided_square_l90_90882

theorem rectangle_perimeter_of_divided_square
  (s : ℝ)
  (hs : 4 * s = 100) :
  let l := s
  let w := s / 2
  2 * (l + w) = 75 :=
by
  let l := s
  let w := s / 2
  sorry

end rectangle_perimeter_of_divided_square_l90_90882


namespace exists_fi_l90_90661

theorem exists_fi (f : ℝ → ℝ) (h_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧ 
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧ 
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧ 
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧ 
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
by
  sorry

end exists_fi_l90_90661


namespace polynomial_value_l90_90494

theorem polynomial_value (x : ℝ) :
  let a := 2009 * x + 2008
  let b := 2009 * x + 2009
  let c := 2009 * x + 2010
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 3 := by
  sorry

end polynomial_value_l90_90494


namespace exists_larger_integer_l90_90423

theorem exists_larger_integer (a b : Nat) (h1 : b > a) (h2 : b - a = 5) (h3 : a * b = 88) :
  b = 11 :=
sorry

end exists_larger_integer_l90_90423


namespace bonus_distribution_plans_l90_90299

theorem bonus_distribution_plans (x y : ℕ) (A B : ℕ) 
  (h1 : x + y = 15)
  (h2 : x = 2 * y)
  (h3 : 10 * A + 5 * B = 20000)
  (hA : A ≥ B)
  (hB : B ≥ 800)
  (hAB_mult_100 : ∃ (k m : ℕ), A = k * 100 ∧ B = m * 100) :
  (x = 10 ∧ y = 5) ∧
  ((A = 1600 ∧ B = 800) ∨
   (A = 1500 ∧ B = 1000) ∨
   (A = 1400 ∧ B = 1200)) :=
by
  -- The proof should be provided here
  sorry

end bonus_distribution_plans_l90_90299


namespace pastry_problem_minimum_n_l90_90294

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end pastry_problem_minimum_n_l90_90294


namespace digit_place_value_ratio_l90_90690

theorem digit_place_value_ratio (n : ℚ) (h1 : n = 85247.2048) (h2 : ∃ d1 : ℚ, d1 * 0.1 = 0.2) (h3 : ∃ d2 : ℚ, d2 * 0.001 = 0.004) : 
  100 = 0.1 / 0.001 :=
by
  sorry

end digit_place_value_ratio_l90_90690


namespace smallest_part_in_ratio_l90_90640

variable (b : ℝ)

theorem smallest_part_in_ratio (h : b = -2620) : 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  smallest_part = 100 :=
by 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  sorry

end smallest_part_in_ratio_l90_90640


namespace length_of_train_is_135_l90_90775

noncomputable def length_of_train (v : ℝ) (t : ℝ) : ℝ :=
  ((v * 1000) / 3600) * t

theorem length_of_train_is_135 :
  length_of_train 140 3.4711508793582233 = 135 :=
sorry

end length_of_train_is_135_l90_90775


namespace sqrt_31_between_5_and_6_l90_90331

theorem sqrt_31_between_5_and_6
  (h1 : Real.sqrt 25 = 5)
  (h2 : Real.sqrt 36 = 6)
  (h3 : 25 < 31)
  (h4 : 31 < 36) :
  5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 :=
sorry

end sqrt_31_between_5_and_6_l90_90331


namespace monotonically_decreasing_implies_a_geq_3_l90_90503

noncomputable def f (x a : ℝ): ℝ := x^3 - a * x - 1

theorem monotonically_decreasing_implies_a_geq_3 : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → f x a ≤ f x 3) →
  a ≥ 3 := 
sorry

end monotonically_decreasing_implies_a_geq_3_l90_90503


namespace class_gpa_l90_90905

theorem class_gpa (n : ℕ) (h1 : (n / 3) * 60 + (2 * (n / 3)) * 66 = total_gpa) :
  total_gpa / n = 64 :=
by
  sorry

end class_gpa_l90_90905


namespace longest_interval_green_l90_90307

-- Definitions for the conditions
def light_cycle_duration : ℕ := 180 -- total cycle duration in seconds
def green_duration : ℕ := 90 -- green light duration in seconds
def red_delay : ℕ := 10 -- red light delay between consecutive lights in seconds
def num_lights : ℕ := 8 -- number of lights

-- Theorem statement to be proved
theorem longest_interval_green (h1 : ∀ i : ℕ, i < num_lights → 
  ∃ t : ℕ, t < light_cycle_duration ∧ (∀ k : ℕ, i + k < num_lights → t + k * red_delay < light_cycle_duration ∧ t + k * red_delay + green_duration <= light_cycle_duration)):
  ∃ interval : ℕ, interval = 20 :=
sorry

end longest_interval_green_l90_90307


namespace probability_fly_reaches_8_10_l90_90915

theorem probability_fly_reaches_8_10 :
  let total_steps := 2^18
  let right_up_combinations := Nat.choose 18 8
  (right_up_combinations / total_steps : ℚ) = Nat.choose 18 8 / 2^18 := 
sorry

end probability_fly_reaches_8_10_l90_90915


namespace range_of_m_l90_90193

theorem range_of_m
  (h : ∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) :
  m < 6 ∧ m ≠ 2 :=
by
  sorry

end range_of_m_l90_90193


namespace amount_C_l90_90400

-- Define the variables and conditions.
variables (A B C : ℝ)
axiom h1 : A = (2 / 3) * B
axiom h2 : B = (1 / 4) * C
axiom h3 : A + B + C = 544

-- State the theorem.
theorem amount_C (A B C : ℝ) (h1 : A = (2 / 3) * B) (h2 : B = (1 / 4) * C) (h3 : A + B + C = 544) : C = 384 := 
sorry

end amount_C_l90_90400


namespace probability_diagonals_intersect_l90_90248

-- Define some basic combinatorial functions in Lean
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the total number of diagonals in a regular nonagon
def total_diagonals (n : ℕ) : ℕ := binom n 2 - n 

-- Define the number of ways to pick 2 diagonals from the total diagonals
def total_pairs_of_diagonals (d : ℕ) : ℕ := binom d 2

-- Define the number of sets of intersecting diagonals
def intersecting_diagonals (n : ℕ) : ℕ := binom n 4

-- Define the given problem as a theorem in Lean
theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) : 
  ((intersecting_diagonals n : ℚ) / (total_pairs_of_diagonals (total_diagonals n) : ℚ)) = 6 / 13 := 
by
  sorry 

end probability_diagonals_intersect_l90_90248


namespace three_digit_number_l90_90072

-- Define the variables involved.
variables (a b c n : ℕ)

-- Condition 1: c = 3a
def condition1 (a c : ℕ) : Prop := c = 3 * a

-- Condition 2: n is three-digit number constructed from a, b, and c.
def is_three_digit (a b c n : ℕ) : Prop := n = 100 * a + 10 * b + c

-- Condition 3: n leaves a remainder of 4 when divided by 5.
def condition2 (n : ℕ) : Prop := n % 5 = 4

-- Condition 4: n leaves a remainder of 3 when divided by 11.
def condition3 (n : ℕ) : Prop := n % 11 = 3

-- Define the main theorem
theorem three_digit_number (a b c n : ℕ) 
(h1: condition1 a c) 
(h2: is_three_digit a b c n) 
(h3: condition2 n) 
(h4: condition3 n) : 
n = 359 := 
sorry

end three_digit_number_l90_90072


namespace contrapositive_example_l90_90558

theorem contrapositive_example :
  (∀ x : ℝ, x^2 < 4 → -2 < x ∧ x < 2) ↔ (∀ x : ℝ, (x ≥ 2 ∨ x ≤ -2) → x^2 ≥ 4) :=
by
  sorry

end contrapositive_example_l90_90558


namespace expression_evaluation_l90_90787

theorem expression_evaluation : 3 * 257 + 4 * 257 + 2 * 257 + 258 = 2571 := by
  sorry

end expression_evaluation_l90_90787


namespace ratio_bob_to_jason_l90_90200

def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := 35

theorem ratio_bob_to_jason : bob_grade / jason_grade = 1 / 2 := by
  sorry

end ratio_bob_to_jason_l90_90200


namespace min_value_inv_sum_l90_90845

open Real

theorem min_value_inv_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  3 ≤ (1 / x) + (1 / y) + (1 / z) :=
sorry

end min_value_inv_sum_l90_90845


namespace num_divisors_of_30_l90_90362

theorem num_divisors_of_30 : 
  (∀ n : ℕ, n > 0 → (30 = 2^1 * 3^1 * 5^1) → (∀ k : ℕ, 0 < k ∧ k ∣ 30 → ∃ m : ℕ, k = 2^m ∧ k ∣ 30)) → 
  ∃ num_divisors : ℕ, num_divisors = 8 := 
by 
  sorry

end num_divisors_of_30_l90_90362


namespace hundred_days_from_friday_is_sunday_l90_90432

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l90_90432


namespace correct_answer_l90_90353

def A : Set ℝ := { x | x^2 + 2 * x - 3 > 0 }
def B : Set ℝ := { -1, 0, 1, 2 }

theorem correct_answer : A ∩ B = { 2 } :=
  sorry

end correct_answer_l90_90353


namespace polynomial_degree_add_sub_l90_90514

noncomputable def degree (p : Polynomial ℂ) : ℕ := 
p.natDegree

variable (M N : Polynomial ℂ)

def is_fifth_degree (M : Polynomial ℂ) : Prop :=
degree M = 5

def is_third_degree (N : Polynomial ℂ) : Prop :=
degree N = 3

theorem polynomial_degree_add_sub (hM : is_fifth_degree M) (hN : is_third_degree N) :
  degree (M + N) = 5 ∧ degree (M - N) = 5 :=
by sorry

end polynomial_degree_add_sub_l90_90514


namespace business_value_l90_90098

theorem business_value (h₁ : (2/3 : ℝ) * (3/4 : ℝ) * V = 30000) : V = 60000 :=
by
  -- conditions and definitions go here
  sorry

end business_value_l90_90098


namespace snow_at_least_once_in_four_days_l90_90570

variable prob_snow : ℚ := 3 / 4

theorem snow_at_least_once_in_four_days :
  let prob_no_snow_in_a_day := 1 - prob_snow in
  let prob_no_snow_in_four_days := prob_no_snow_in_a_day ^ 4 in
  1 - prob_no_snow_in_four_days = 255 / 256 :=
by
  let prob_no_snow_in_a_day := 1 - prob_snow
  let prob_no_snow_in_four_days := prob_no_snow_in_a_day ^ 4
  have h1 : prob_no_snow_in_a_day = 1 / 4 := by norm_num [prob_snow]
  have h2 : prob_no_snow_in_four_days = (1 / 4) ^ 4 := by rw [h1]
  have h3 : prob_no_snow_in_four_days = 1 / 256 := by norm_num
  rw [h3]
  norm_num

end snow_at_least_once_in_four_days_l90_90570


namespace age_ratio_in_9_years_l90_90345

-- Initial age definitions for Mike and Sam
def ages (m s : ℕ) : Prop :=
  (m - 5 = 2 * (s - 5)) ∧ (m - 12 = 3 * (s - 12))

-- Proof that in 9 years the ratio of their ages will be 3:2
theorem age_ratio_in_9_years (m s x : ℕ) (h_ages : ages m s) :
  (m + x) * 2 = 3 * (s + x) ↔ x = 9 :=
by {
  sorry
}

end age_ratio_in_9_years_l90_90345


namespace foci_distance_of_hyperbola_l90_90800

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end foci_distance_of_hyperbola_l90_90800


namespace smallest_prime_perimeter_l90_90919

-- Define a function that checks if a number is an odd prime
def is_odd_prime (n : ℕ) : Prop :=
  n > 2 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)) ∧ (n % 2 = 1)

-- Define a function that checks if three numbers are consecutive odd primes
def consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  b = a + 2 ∧ c = b + 2

-- Define a function that checks if three numbers form a scalene triangle and satisfy the triangle inequality
def scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem to prove
theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), consecutive_odd_primes a b c ∧ scalene_triangle a b c ∧ (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l90_90919


namespace max_new_cars_l90_90469

theorem max_new_cars (b₁ : ℕ) (r : ℝ) (M : ℕ) (L : ℕ) (x : ℝ) (h₀ : b₁ = 30) (h₁ : r = 0.94) (h₂ : M = 600000) (h₃ : L = 300000) :
  x ≤ (3.6 * 10^4) :=
sorry

end max_new_cars_l90_90469


namespace hundred_days_from_friday_is_sunday_l90_90430

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l90_90430


namespace speed_of_other_train_l90_90888

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l90_90888


namespace power_multiplication_l90_90785

theorem power_multiplication : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end power_multiplication_l90_90785


namespace hex_conversion_sum_l90_90328

-- Convert hexadecimal E78 to decimal
def hex_to_decimal (h : String) : Nat :=
  match h with
  | "E78" => 3704
  | _ => 0

-- Convert decimal to radix 7
def decimal_to_radix7 (d : Nat) : String :=
  match d with
  | 3704 => "13541"
  | _ => ""

-- Convert radix 7 to decimal
def radix7_to_decimal (r : String) : Nat :=
  match r with
  | "13541" => 3704
  | _ => 0

-- Convert decimal to hexadecimal
def decimal_to_hex (d : Nat) : String :=
  match d with
  | 3704 => "E78"
  | 7408 => "1CF0"
  | _ => ""

theorem hex_conversion_sum :
  let initial_hex : String := "E78"
  let final_decimal := 3704 
  let final_hex := decimal_to_hex (final_decimal)
  let final_sum := hex_to_decimal initial_hex + final_decimal
  (decimal_to_hex final_sum) = "1CF0" :=
by
  sorry

end hex_conversion_sum_l90_90328


namespace set_contains_one_implies_values_l90_90358

theorem set_contains_one_implies_values (x : ℝ) (A : Set ℝ) (hA : A = {x, x^2}) (h1 : 1 ∈ A) : x = 1 ∨ x = -1 := by
  sorry

end set_contains_one_implies_values_l90_90358


namespace diet_soda_count_l90_90298

theorem diet_soda_count (D : ℕ) (h1 : 81 = D + 21) : D = 60 := by
  sorry

end diet_soda_count_l90_90298


namespace line_equation_through_point_parallel_to_lines_l90_90064

theorem line_equation_through_point_parallel_to_lines (L L1 L2 : ℝ → ℝ → Prop) :
  (∀ x, L1 x (y: ℝ) ↔ 3 * x + y - 6 = 0) →
  (∀ x, L2 x (y: ℝ) ↔ 3 * x + y + 3 = 0) →
  (L 1 0) →
  (∀ x1 y1 x2 y2, L1 x1 y1 → L1 x2 y2 → (y2 - y1) / (x2 - x1) = -3) →
  ∃ A B C, (A = 1 ∧ B = -3 ∧ C = -3) ∧ (∀ x y, L x y ↔ A * x + B * y + C = 0) :=
by sorry

end line_equation_through_point_parallel_to_lines_l90_90064


namespace ratio_of_blue_to_red_area_l90_90143

theorem ratio_of_blue_to_red_area :
  let r₁ := 1 / 2
  let r₂ := 3 / 2
  let A_red := Real.pi * r₁^2
  let A_large := Real.pi * r₂^2
  let A_blue := A_large - A_red
  A_blue / A_red = 8 :=
by
  sorry

end ratio_of_blue_to_red_area_l90_90143


namespace pause_point_l90_90788

-- Definitions
def total_movie_length := 60 -- In minutes
def remaining_time := 30 -- In minutes

-- Theorem stating the pause point in the movie
theorem pause_point : total_movie_length - remaining_time = 30 := by
  -- This is the original solution in mathematical terms, omitted in lean statement.
  -- total_movie_length - remaining_time = 60 - 30 = 30
  sorry

end pause_point_l90_90788


namespace base_conversion_sum_l90_90332

def A := 10
def B := 11

def convert_base11_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 11^2
  let d1 := (n % 11^2) / 11
  let d0 := n % 11
  d2 * 11^2 + d1 * 11 + d0

def convert_base12_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 12^2
  let d1 := (n % 12^2) / 12
  let d0 := n % 12
  d2 * 12^2 + d1 * 12 + d0

def n1 := 2 * 11^2 + 4 * 11 + 9    -- = 249_11 in base 10
def n2 := 3 * 12^2 + A * 12 + B   -- = 3AB_12 in base 10

theorem base_conversion_sum :
  (convert_base11_to_base10 294 + convert_base12_to_base10 563 = 858) := by
  sorry

end base_conversion_sum_l90_90332


namespace sides_relation_l90_90719

structure Triangle :=
  (A B C : Type)
  (α β γ : ℝ)
  (a b c : ℝ)

axiom angle_relation (T : Triangle) : 3 * T.α + 2 * T.β = 180

theorem sides_relation (T : Triangle) (h : angle_relation T) : T.a^2 + T.a * T.b = T.c^2 :=
by
  sorry

end sides_relation_l90_90719


namespace dorchester_daily_pay_l90_90147

theorem dorchester_daily_pay (D : ℝ) (P : ℝ) (total_earnings : ℝ) (num_puppies : ℕ) (earn_per_puppy : ℝ) 
  (h1 : total_earnings = 76) (h2 : num_puppies = 16) (h3 : earn_per_puppy = 2.25) 
  (h4 : total_earnings = D + num_puppies * earn_per_puppy) : D = 40 :=
by
  sorry

end dorchester_daily_pay_l90_90147


namespace toilet_paper_production_per_day_l90_90152

theorem toilet_paper_production_per_day 
    (total_production_march : ℕ)
    (days_in_march : ℕ)
    (increase_factor : ℕ)
    (total_production : ℕ)
    (days : ℕ)
    (increase : ℕ)
    (production : ℕ) :
    total_production_march = total_production →
    days_in_march = days →
    increase_factor = increase →
    total_production = 868000 →
    days = 31 →
    increase = 3 →
    production = total_production / days →
    production / increase = 9333
:= by
  intros h1 h2 h3 h4 h5 h6 h7

  sorry

end toilet_paper_production_per_day_l90_90152


namespace students_before_new_year_le_197_l90_90751

variable (N M k ℓ : ℕ)

-- Conditions
axiom condition_1 : M = (k * N) / 100
axiom condition_2 : 100 * M = k * N
axiom condition_3 : 100 * (M + 1) = ℓ * (N + 3)
axiom condition_4 : ℓ < 100

-- The theorem to prove
theorem students_before_new_year_le_197 :
  N ≤ 197 :=
by
  sorry

end students_before_new_year_le_197_l90_90751


namespace karen_has_32_quarters_l90_90699

variable (k : ℕ)  -- the number of quarters Karen has

-- Define the number of quarters Christopher has
def christopher_quarters : ℕ := 64

-- Define the value of a single quarter in dollars
def quarter_value : ℚ := 0.25

-- Define the amount of money Christopher has
def christopher_money : ℚ := christopher_quarters * quarter_value

-- Define the monetary difference between Christopher and Karen
def money_difference : ℚ := 8

-- Define the amount of money Karen has
def karen_money : ℚ := christopher_money - money_difference

-- Define the number of quarters Karen has
def karen_quarters := karen_money / quarter_value

-- The theorem we need to prove
theorem karen_has_32_quarters : k = 32 :=
by
  sorry

end karen_has_32_quarters_l90_90699


namespace sum_of_digits_0_to_999_l90_90744

-- Sum of digits from 0 to 9
def sum_of_digits : ℕ := (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Sum of digits from 1 to 9
def sum_of_digits_without_zero : ℕ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)

-- Units place sum
def units_sum : ℕ := sum_of_digits * 100

-- Tens place sum
def tens_sum : ℕ := sum_of_digits * 100

-- Hundreds place sum
def hundreds_sum : ℕ := sum_of_digits_without_zero * 100

-- Total sum
def total_sum : ℕ := units_sum + tens_sum + hundreds_sum

theorem sum_of_digits_0_to_999 : total_sum = 13500 := by
  sorry

end sum_of_digits_0_to_999_l90_90744


namespace snow_at_least_once_in_four_days_l90_90569

variable prob_snow : ℚ := 3 / 4

theorem snow_at_least_once_in_four_days :
  let prob_no_snow_in_a_day := 1 - prob_snow in
  let prob_no_snow_in_four_days := prob_no_snow_in_a_day ^ 4 in
  1 - prob_no_snow_in_four_days = 255 / 256 :=
by
  let prob_no_snow_in_a_day := 1 - prob_snow
  let prob_no_snow_in_four_days := prob_no_snow_in_a_day ^ 4
  have h1 : prob_no_snow_in_a_day = 1 / 4 := by norm_num [prob_snow]
  have h2 : prob_no_snow_in_four_days = (1 / 4) ^ 4 := by rw [h1]
  have h3 : prob_no_snow_in_four_days = 1 / 256 := by norm_num
  rw [h3]
  norm_num

end snow_at_least_once_in_four_days_l90_90569


namespace a_2008_lt_5_l90_90576

theorem a_2008_lt_5 :
  ∃ a b : ℕ → ℝ, 
    a 1 = 1 ∧ 
    b 1 = 2 ∧ 
    (∀ n, a (n + 1) = (1 + a n + a n * b n) / (b n)) ∧ 
    (∀ n, b (n + 1) = (1 + b n + a n * b n) / (a n)) ∧ 
    a 2008 < 5 := 
sorry

end a_2008_lt_5_l90_90576


namespace sum_is_correct_l90_90951

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l90_90951


namespace annual_concert_tickets_l90_90313

theorem annual_concert_tickets (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : NS = 60 :=
by
  sorry

end annual_concert_tickets_l90_90313


namespace sin_neg_135_eq_neg_sqrt_2_over_2_l90_90632

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_135_eq_neg_sqrt_2_over_2_l90_90632


namespace gcd_max_value_l90_90623

theorem gcd_max_value : ∀ (n : ℕ), n > 0 → ∃ (d : ℕ), d = 9 ∧ d ∣ gcd (13 * n + 4) (8 * n + 3) :=
by
  sorry

end gcd_max_value_l90_90623


namespace largest_AC_value_l90_90387

theorem largest_AC_value : ∃ (a b c d : ℕ), 
  a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (AC BD : ℝ), AC * BD = a * c + b * d ∧
  AC ^ 2 + BD ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ∧
  AC = Real.sqrt 458) :=
sorry

end largest_AC_value_l90_90387


namespace sum_of_digits_S_l90_90080

-- Define S as 10^2021 - 2021
def S : ℕ := 10^2021 - 2021

-- Define function to calculate sum of digits of a given number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum 

theorem sum_of_digits_S :
  sum_of_digits S = 18185 :=
sorry

end sum_of_digits_S_l90_90080


namespace Shekar_average_marks_l90_90048

theorem Shekar_average_marks 
  (math_marks : ℕ := 76)
  (science_marks : ℕ := 65)
  (social_studies_marks : ℕ := 82)
  (english_marks : ℕ := 67)
  (biology_marks : ℕ := 95)
  (num_subjects : ℕ := 5) :
  (math_marks + science_marks + social_studies_marks + english_marks + biology_marks) / num_subjects = 77 := 
sorry

end Shekar_average_marks_l90_90048


namespace age_problem_l90_90533

theorem age_problem 
  (K S E F : ℕ)
  (h1 : K = S - 5)
  (h2 : S = 2 * E)
  (h3 : E = F + 9)
  (h4 : K = 33) : 
  F = 10 :=
by 
  sorry

end age_problem_l90_90533


namespace solve_for_y_l90_90405

theorem solve_for_y (y : ℕ) (h1 : 40 = 2^3 * 5) (h2 : 8 = 2^3) :
  40^3 = 8^y ↔ y = 3 :=
by sorry

end solve_for_y_l90_90405


namespace total_number_of_candles_l90_90836

theorem total_number_of_candles
  (candles_bedroom : ℕ)
  (candles_living_room : ℕ)
  (candles_donovan : ℕ)
  (h1 : candles_bedroom = 20)
  (h2 : candles_bedroom = 2 * candles_living_room)
  (h3 : candles_donovan = 20) :
  candles_bedroom + candles_living_room + candles_donovan = 50 :=
by
  sorry

end total_number_of_candles_l90_90836


namespace bucket_P_turns_to_fill_the_drum_l90_90140

-- Define the capacities of the buckets
def capacity_P := 3
def capacity_Q := 1

-- Define the total number of turns for both buckets together to fill the drum
def turns_together := 60

-- Define the total capacity of the drum that gets filled in the given scenario of the problem
def total_capacity := turns_together * (capacity_P + capacity_Q)

-- The question: How many turns does it take for bucket P alone to fill this total capacity?
def turns_P_alone : ℕ :=
  total_capacity / capacity_P

theorem bucket_P_turns_to_fill_the_drum :
  turns_P_alone = 80 :=
by
  sorry

end bucket_P_turns_to_fill_the_drum_l90_90140


namespace time_to_cross_signal_post_l90_90274

-- Definition of the conditions
def length_of_train : ℝ := 600  -- in meters
def time_to_cross_bridge : ℝ := 8  -- in minutes
def length_of_bridge : ℝ := 7200  -- in meters

-- Equivalent statement
theorem time_to_cross_signal_post (constant_speed : ℝ) (t : ℝ) 
  (h1 : constant_speed * t = length_of_train) 
  (h2 : constant_speed * time_to_cross_bridge = length_of_train + length_of_bridge) : 
  t * 60 = 36.9 := 
sorry

end time_to_cross_signal_post_l90_90274


namespace point_in_third_quadrant_l90_90003

def quadrant_of_point (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "first"
  else if x < 0 ∧ y > 0 then "second"
  else if x < 0 ∧ y < 0 then "third"
  else if x > 0 ∧ y < 0 then "fourth"
  else "on_axis"

theorem point_in_third_quadrant : quadrant_of_point (-2) (-3) = "third" :=
  by sorry

end point_in_third_quadrant_l90_90003


namespace train_speed_l90_90890

theorem train_speed (v : ℝ) 
  (h1 : 50 * 2.5 + v * 2.5 = 285) : v = 64 := 
by
  -- h1 unfolds conditions into the mathematical equation
  -- here we would have the proof steps, adding a "sorry" to skip proof steps.
  sorry

end train_speed_l90_90890


namespace kyle_caught_14_fish_l90_90472

theorem kyle_caught_14_fish (K T C : ℕ) (h1 : K = T) (h2 : C = 8) (h3 : C + K + T = 36) : K = 14 :=
by
  -- Proof goes here
  sorry

end kyle_caught_14_fish_l90_90472


namespace find_a_l90_90192

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 2

theorem find_a (a : ℝ) (h : f a (f a (Real.sqrt 2)) = -Real.sqrt 2) : 
  a = Real.sqrt 2 / 2 :=
by
  sorry

end find_a_l90_90192


namespace smallest_integer_inequality_l90_90438

theorem smallest_integer_inequality:
  ∃ x : ℤ, (2 * x < 3 * x - 10) ∧ ∀ y : ℤ, (2 * y < 3 * y - 10) → y ≥ 11 := by
  sorry

end smallest_integer_inequality_l90_90438


namespace dilation_translation_correct_l90_90344

def transformation_matrix (d: ℝ) (tx: ℝ) (ty: ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![d, 0, tx],
    ![0, d, ty],
    ![0, 0, 1]
  ]

theorem dilation_translation_correct :
  transformation_matrix 4 2 3 =
  ![
    ![4, 0, 2],
    ![0, 4, 3],
    ![0, 0, 1]
  ] :=
by
  sorry

end dilation_translation_correct_l90_90344


namespace total_cost_correct_l90_90693

def cost_of_cat_toy := 10.22
def cost_of_cage := 11.73
def cost_of_cat_food := 7.50
def cost_of_leash := 5.15
def cost_of_cat_treats := 3.98

theorem total_cost_correct : 
  cost_of_cat_toy + cost_of_cage + cost_of_cat_food + cost_of_leash + cost_of_cat_treats = 38.58 := 
by
  sorry

end total_cost_correct_l90_90693


namespace slope_angle_AB_l90_90808

noncomputable def A : ℝ × ℝ := (0, 1)
noncomputable def B : ℝ × ℝ := (1, 0)

theorem slope_angle_AB :
  let θ := Real.arctan (↑(B.2 - A.2) / ↑(B.1 - A.1))
  θ = 3 * Real.pi / 4 := 
by
  -- Proof goes here
  sorry

end slope_angle_AB_l90_90808


namespace simplified_expr_l90_90471

theorem simplified_expr : 
  (Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2) ^ 2) = (8 + 2 * Real.sqrt 2) := 
by 
  sorry

end simplified_expr_l90_90471


namespace min_ab_value_l90_90666

theorem min_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 / a + 20 / b = 4) : ab = 25 :=
sorry

end min_ab_value_l90_90666


namespace number_of_triangles_l90_90996

theorem number_of_triangles (k : ℕ) (n : ℕ → ℕ) :
  ∑ (p q r : ℕ) in finset.triple_powerset (finset.range k), n p * n q * n r +
  ∑ (p q : ℕ) in finset.twos_powerset (finset.range k), n p * nat.choose (n q) 2 + n q * nat.choose (n p) 2 =
    ∑ (1 ≤ p < q < r ≤ k), n p * n q * n r +
    ∑ (1 ≤ p < q ≤ k), n p * nat.choose (n q) 2 + n q * nat.choose (n p) 2 :=
sorry

end number_of_triangles_l90_90996


namespace minimum_pies_for_trick_l90_90297

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l90_90297


namespace area_of_rectangle_l90_90764

-- Define the problem conditions in Lean
def circle_radius := 7
def circle_diameter := 2 * circle_radius
def width_of_rectangle := circle_diameter
def length_to_width_ratio := 3
def length_of_rectangle := length_to_width_ratio * width_of_rectangle

-- Define the statement to be proved (area of the rectangle)
theorem area_of_rectangle : 
  (length_of_rectangle * width_of_rectangle) = 588 := by
  sorry

end area_of_rectangle_l90_90764


namespace average_of_three_l90_90675

-- Definitions of Conditions
variables (A B C : ℝ)
variables (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132)

-- The proof problem stating the goal
theorem average_of_three (A B C : ℝ) 
    (h1 : A + B = 147) (h2 : B + C = 123) (h3 : A + C = 132) : 
    (A + B + C) / 3 = 67 := 
sorry

end average_of_three_l90_90675


namespace sum_of_first_five_primes_units_digit_3_l90_90967

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l90_90967


namespace value_of_x_l90_90681

theorem value_of_x (x : ℤ) : (x + 1) * (x + 1) = 16 ↔ (x = 3 ∨ x = -5) := 
by sorry

end value_of_x_l90_90681


namespace root_value_algebraic_expression_l90_90176

theorem root_value_algebraic_expression {a : ℝ} (h : a^2 + 3 * a + 2 = 0) : a^2 + 3 * a = -2 :=
by
  sorry

end root_value_algebraic_expression_l90_90176


namespace sum_of_first_five_primes_with_units_digit_3_l90_90982

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90982


namespace exist_same_number_of_acquaintances_l90_90521

-- Define a group of 2014 people
variable (People : Type) [Fintype People] [DecidableEq People]
variable (knows : People → People → Prop)
variable [DecidableRel knows]

-- Conditions
def mutual_acquaintance : Prop := 
  ∀ (a b : People), knows a b ↔ knows b a

def num_people : Prop := 
  Fintype.card People = 2014

-- Theorem to prove
theorem exist_same_number_of_acquaintances 
  (h1 : mutual_acquaintance People knows) 
  (h2 : num_people People) : 
  ∃ (p1 p2 : People), p1 ≠ p2 ∧
    (Fintype.card { x // knows p1 x } = Fintype.card { x // knows p2 x }) :=
sorry

end exist_same_number_of_acquaintances_l90_90521


namespace janet_hourly_wage_l90_90384

theorem janet_hourly_wage : 
  ∃ x : ℝ, 
    (20 * x + (5 * 20 + 7 * 20) = 1640) ∧ 
    x = 70 :=
by
  use 70
  sorry

end janet_hourly_wage_l90_90384


namespace find_fourth_number_l90_90682

theorem find_fourth_number (x y : ℝ) (h1 : 0.25 / x = 2 / y) (h2 : x = 0.75) : y = 6 :=
by
  sorry

end find_fourth_number_l90_90682


namespace min_value_fraction_l90_90659

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ln : Real.log (a + b) = 0) :
  (2 / a + 3 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_fraction_l90_90659


namespace percentage_increase_of_x_compared_to_y_l90_90372

-- We are given that y = 0.5 * z and x = 0.6 * z
-- We need to prove that the percentage increase of x compared to y is 20%

theorem percentage_increase_of_x_compared_to_y (x y z : ℝ) 
  (h1 : y = 0.5 * z) 
  (h2 : x = 0.6 * z) : 
  (x / y - 1) * 100 = 20 :=
by 
  -- Placeholder for actual proof
  sorry

end percentage_increase_of_x_compared_to_y_l90_90372


namespace german_team_goals_l90_90113

/-- Proof that the German team could have scored 11, 12, 14, 16, or 17 goals given the conditions
    stated by the three journalists, with exactly two of their statements being true. -/
theorem german_team_goals (x : ℕ) :
  (10 < x ∧ x < 17 → 11 < x ∧ x < 18 → x % 2 = 1 → false) →
  (10 < x ∧ x < 17 ∧ ¬(11 < x ∧ x < 18) ∧ (x % 2 = 1) → false) →
  (¬(10 < x ∧ x < 17) ∧ 11 < x ∧ x < 18 ∧ (x % 2 = 1) → false) →
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end german_team_goals_l90_90113


namespace athletes_meet_second_time_at_l90_90908

-- Define the conditions given in the problem
def distance_AB : ℕ := 110

def man_uphill_speed : ℕ := 3
def man_downhill_speed : ℕ := 5

def woman_uphill_speed : ℕ := 2
def woman_downhill_speed : ℕ := 3

-- Define the times for the athletes' round trips
def man_round_trip_time : ℚ := (distance_AB / man_uphill_speed) + (distance_AB / man_downhill_speed)
def woman_round_trip_time : ℚ := (distance_AB / woman_uphill_speed) + (distance_AB / woman_downhill_speed)

-- Lean statement for the proof
theorem athletes_meet_second_time_at :
  ∀ (t : ℚ), t = lcm (man_round_trip_time) (woman_round_trip_time) →
  ∃ d : ℚ, d = 330 / 7 := 
by sorry

end athletes_meet_second_time_at_l90_90908


namespace find_smallest_N_l90_90346

-- Define the sum of digits functions as described
def sum_of_digits_base (n : ℕ) (b : ℕ) : ℕ :=
  n.digits b |>.sum

-- Define f(n) which is the sum of digits in base-five representation of n
def f (n : ℕ) : ℕ :=
  sum_of_digits_base n 5

-- Define g(n) which is the sum of digits in base-seven representation of f(n)
def g (n : ℕ) : ℕ :=
  sum_of_digits_base (f n) 7

-- The statement of the problem: find the smallest N such that 
-- g(N) in base-sixteen cannot be represented using only digits 0 to 9
theorem find_smallest_N : ∃ N : ℕ, (g N ≥ 10) ∧ (N % 1000 = 610) :=
by
  sorry

end find_smallest_N_l90_90346


namespace quadratic_roots_l90_90239

theorem quadratic_roots (x : ℝ) : x^2 + 4 * x + 3 = 0 → x = -3 ∨ x = -1 :=
by
  intro h
  have h1 : (x + 3) * (x + 1) = 0 := by sorry
  have h2 : (x = -3 ∨ x = -1) := by sorry
  exact h2

end quadratic_roots_l90_90239


namespace farmer_plough_l90_90767

theorem farmer_plough (x : ℝ) : 
  (∃ D : ℝ, D = 448 / x ∧ (D + 2) * 85 = 408) ∧ 
  448 - ( (448 / x + 2) * 85 - 40) = 448 - 408 :=
  x = 160 :=
begin
  sorry
end

end farmer_plough_l90_90767


namespace professors_initial_count_l90_90015

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l90_90015


namespace overall_percentage_gain_l90_90921

theorem overall_percentage_gain
    (original_price : ℝ)
    (first_increase : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (third_discount : ℝ)
    (final_increase : ℝ)
    (final_price : ℝ)
    (overall_gain : ℝ)
    (overall_percentage_gain : ℝ)
    (h1 : original_price = 100)
    (h2 : first_increase = original_price * 1.5)
    (h3 : first_discount = first_increase * 0.9)
    (h4 : second_discount = first_discount * 0.85)
    (h5 : third_discount = second_discount * 0.8)
    (h6 : final_increase = third_discount * 1.1)
    (h7 : final_price = final_increase)
    (h8 : overall_gain = final_price - original_price)
    (h9 : overall_percentage_gain = (overall_gain / original_price) * 100) :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_l90_90921


namespace complex_sum_problem_l90_90205

-- Define the problem conditions and the goal in Lean 4
theorem complex_sum_problem (n : ℕ) (a : ℕ → ℝ) (ω : ℂ) 
  (hω : ω^4 = 1 ∧ ω.im ≠ 0) 
  (h_sum : (finset.range n).sum (λ k, 1 / (a k + ω)) = 3 + 4 * complex.I) : 
  (finset.range n).sum (λ k, (3 * a k - 2) / (a k^2 - 2 * a k + 2)) = 6 :=
sorry

end complex_sum_problem_l90_90205


namespace interval_between_doses_l90_90383

noncomputable def dose_mg : ℕ := 2 * 375

noncomputable def total_mg_per_day : ℕ := 3000

noncomputable def hours_in_day : ℕ := 24

noncomputable def doses_per_day := total_mg_per_day / dose_mg

noncomputable def hours_between_doses := hours_in_day / doses_per_day

theorem interval_between_doses : hours_between_doses = 6 :=
by
  sorry

end interval_between_doses_l90_90383


namespace polynomial_remainder_l90_90487

theorem polynomial_remainder (a b : ℤ) :
  (∀ x : ℤ, 3 * x ^ 6 - 2 * x ^ 4 + 5 * x ^ 2 - 9 = (x + 1) * (x + 2) * (q : ℤ) + a * x + b) →
  (a = -174 ∧ b = -177) :=
by sorry

end polynomial_remainder_l90_90487


namespace grisha_lesha_communication_l90_90449

theorem grisha_lesha_communication :
  ∀ (deck : Finset ℕ) (grisha lesha : Finset ℕ) (kolya_card : ℕ),
  deck.card = 7 →
  grisha.card = 3 →
  lesha.card = 3 →
  kolya_card ∈ deck →
  (grisha ∪ lesha ∪ {kolya_card}) = deck →
  (∃ (announce : Finset ℕ) (possibility : Finset ℕ), 
    (announce = grisha ∨ announce = possibility) ∧
    (lesha ∪ announce ≠ lesha ∨ announce = grisha) ∧
    (announce ∪ lesha ∪ {kolya_card} = deck) →
    ∀ (kolya_info : ℕ), ∀ (announcement : ℕ), 
      kolya_info ∈ deck - grisha →
      kolya_info ∈ deck - lesha →
      grisha ∩ lesha = ∅ →
      announcement ∉ lesha →
      announcement ∉ grisha →
      True) :=
  sorry

end grisha_lesha_communication_l90_90449


namespace fruit_eating_problem_l90_90603

theorem fruit_eating_problem (a₀ p₀ o₀ : ℕ) (h₀ : a₀ = 5) (h₁ : p₀ = 8) (h₂ : o₀ = 11) :
  ¬ ∃ (d : ℕ), (a₀ - d) = (p₀ - d) ∧ (p₀ - d) = (o₀ - d) ∧ ∀ k, k ≤ d → ((a₀ - k) + (p₀ - k) + (o₀ - k) = 24 - 2 * k ∧ a₀ - k ≥ 0 ∧ p₀ - k ≥ 0 ∧ o₀ - k ≥ 0) :=
by
  sorry

end fruit_eating_problem_l90_90603


namespace find_tire_price_l90_90575

def regular_price_of_tire (x : ℝ) : Prop :=
  3 * x + 0.75 * x = 270

theorem find_tire_price (x : ℝ) (h1 : regular_price_of_tire x) : x = 72 :=
by
  sorry

end find_tire_price_l90_90575


namespace value_of_a_l90_90189

-- Define the variables and conditions as lean definitions/constants
variable (a b c : ℝ)
variable (h1 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1))
variable (h2 : a * 15 * 11 = 1)

-- Statement to prove
theorem value_of_a : a = 6 :=
by
  sorry

end value_of_a_l90_90189


namespace sum_of_first_five_primes_with_units_digit_3_l90_90981

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90981


namespace tangent_line_circle_l90_90994

open Real

theorem tangent_line_circle (m n : ℝ) :
  (∀ x y : ℝ, ((m + 1) * x + (n + 1) * y - 2 = 0) ↔ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n) ≤ 2 - 2 * sqrt 2) ∨ (2 + 2 * sqrt 2 ≤ (m + n)) := by
  sorry

end tangent_line_circle_l90_90994


namespace distinct_pen_distribution_l90_90989

theorem distinct_pen_distribution :
  ∃! (a b c d : ℕ), a + b + c + d = 10 ∧
                    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end distinct_pen_distribution_l90_90989


namespace minimum_value_l90_90342

noncomputable def polynomial_expr (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 5

theorem minimum_value : ∃ x y : ℝ, (polynomial_expr x y = 8) := 
sorry

end minimum_value_l90_90342


namespace max_min_diff_c_l90_90032

theorem max_min_diff_c {a b c : ℝ} 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 15) : 
  (∃ c_max c_min, 
    (∀ a b c, a + b + c = 3 ∧ a^2 + b^2 + c^2 = 15 → c_min ≤ c ∧ c ≤ c_max) ∧ 
    c_max - c_min = 16 / 3) :=
sorry

end max_min_diff_c_l90_90032


namespace circle_area_with_radius_three_is_9pi_l90_90076

theorem circle_area_with_radius_three_is_9pi (r : ℝ) (h : r = 3) : Real.pi * r^2 = 9 * Real.pi := by
  sorry

end circle_area_with_radius_three_is_9pi_l90_90076


namespace squats_day_after_tomorrow_l90_90148

theorem squats_day_after_tomorrow (initial_day_squats : ℕ) (increase_per_day : ℕ)
  (h1 : initial_day_squats = 30) (h2 : increase_per_day = 5) :
  let second_day_squats := initial_day_squats + increase_per_day in
  let third_day_squats := second_day_squats + increase_per_day in
  let fourth_day_squats := third_day_squats + increase_per_day in
  fourth_day_squats = 45 :=
by
  -- Placeholder proof
  sorry

end squats_day_after_tomorrow_l90_90148


namespace initial_percentage_of_water_is_12_l90_90088

noncomputable def initial_percentage_of_water (initial_volume : ℕ) (added_water : ℕ) (final_percentage : ℕ) : ℕ :=
  let final_volume := initial_volume + added_water
  let final_water_amount := (final_percentage * final_volume) / 100
  let initial_water_amount := final_water_amount - added_water
  (initial_water_amount * 100) / initial_volume

theorem initial_percentage_of_water_is_12 :
  initial_percentage_of_water 20 2 20 = 12 :=
by
  sorry

end initial_percentage_of_water_is_12_l90_90088


namespace tax_rate_calculation_l90_90444

theorem tax_rate_calculation (price_before_tax total_price : ℝ) 
  (h_price_before_tax : price_before_tax = 92) 
  (h_total_price : total_price = 98.90) : 
  (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := 
by 
  -- Proof will be provided here.
  sorry

end tax_rate_calculation_l90_90444


namespace sum_of_first_five_primes_with_units_digit_3_l90_90980

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90980


namespace carbon_atoms_in_compound_l90_90765

theorem carbon_atoms_in_compound 
    (molecular_weight : ℕ := 65)
    (carbon_weight : ℕ := 12)
    (hydrogen_weight : ℕ := 1)
    (oxygen_weight : ℕ := 16)
    (hydrogen_atoms : ℕ := 1)
    (oxygen_atoms : ℕ := 1) :
    ∃ (carbon_atoms : ℕ), molecular_weight = (carbon_atoms * carbon_weight) + (hydrogen_atoms * hydrogen_weight) + (oxygen_atoms * oxygen_weight) ∧ carbon_atoms = 4 :=
by
  sorry

end carbon_atoms_in_compound_l90_90765


namespace sector_central_angle_in_radians_l90_90811

/-- 
Given a sector of a circle where the perimeter is 4 cm 
and the area is 1 cm², prove that the central angle 
of the sector in radians is 2.
-/
theorem sector_central_angle_in_radians 
  (r l : ℝ) 
  (h_perimeter : 2 * r + l = 4) 
  (h_area : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by
  sorry

end sector_central_angle_in_radians_l90_90811


namespace sum_of_coordinates_of_C_and_D_l90_90044

structure Point where
  x : ℤ
  y : ℤ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def sum_coordinates (p1 p2 : Point) : ℤ :=
  p1.x + p1.y + p2.x + p2.y

def C : Point := { x := 3, y := -2 }
def D : Point := reflect_y C

theorem sum_of_coordinates_of_C_and_D : sum_coordinates C D = -4 := by
  sorry

end sum_of_coordinates_of_C_and_D_l90_90044


namespace part1_part2_l90_90816

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin x - 1/2 * Real.cos (2 * x) + a - 3/a + 1/2

theorem part1 (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 0 1 := sorry

theorem part2 (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 := sorry

end part1_part2_l90_90816


namespace greatest_three_digit_divisible_by_3_6_5_l90_90436

/-- Define a three-digit number and conditions for divisibility by 3, 6, and 5 -/
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_divisible_by (n : ℕ) (d : ℕ) : Prop := d ∣ n

/-- Greatest three-digit number divisible by 3, 6, and 5 is 990 -/
theorem greatest_three_digit_divisible_by_3_6_5 : ∃ n : ℕ, is_three_digit n ∧ is_divisible_by n 3 ∧ is_divisible_by n 6 ∧ is_divisible_by n 5 ∧ n = 990 :=
sorry

end greatest_three_digit_divisible_by_3_6_5_l90_90436


namespace power_of_exponents_l90_90516

theorem power_of_exponents (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 := by
  sorry

end power_of_exponents_l90_90516


namespace find_S_30_l90_90731

variable (S : ℕ → ℚ)
variable (a : ℕ → ℚ)
variable (d : ℚ)

-- Definitions based on conditions
def arithmetic_sum (n : ℕ) : ℚ := (n / 2) * (a 1 + a n)
def a_n (n : ℕ) : ℚ := a 1 + (n - 1) * d

-- Given conditions
axiom h1 : S 10 = 20
axiom h2 : S 20 = 15

-- Required Proof (the final statement to be proven)
theorem find_S_30 : S 30 = -15 := sorry

end find_S_30_l90_90731


namespace number_of_triangles_with_perimeter_nine_l90_90508

theorem number_of_triangles_with_perimeter_nine : 
  ∃ (a b c : ℕ), a + b + c = 9 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry  -- Proof steps are omitted.

end number_of_triangles_with_perimeter_nine_l90_90508


namespace sin_four_thirds_pi_l90_90488

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_four_thirds_pi_l90_90488


namespace sum_of_first_five_primes_with_units_digit_three_l90_90973

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l90_90973


namespace mileage_interval_l90_90215

-- Define the distances driven each day
def d1 : ℕ := 135
def d2 : ℕ := 135 + 124
def d3 : ℕ := 159
def d4 : ℕ := 189

-- Define the total distance driven
def total_distance : ℕ := d1 + d2 + d3 + d4

-- Define the number of intervals (charges)
def number_of_intervals : ℕ := 6

-- Define the expected mileage interval for charging
def expected_interval : ℕ := 124

-- The theorem to prove that the mileage interval is approximately 124 miles
theorem mileage_interval : total_distance / number_of_intervals = expected_interval := by
  sorry

end mileage_interval_l90_90215


namespace weight_feel_when_lowered_l90_90418

-- Conditions from the problem
def num_plates : ℕ := 10
def weight_per_plate : ℝ := 30
def technology_increase : ℝ := 0.20
def incline_increase : ℝ := 0.15

-- Calculate the contributions
def total_weight_without_factors : ℝ := num_plates * weight_per_plate
def weight_with_technology : ℝ := total_weight_without_factors * (1 + technology_increase)
def weight_with_incline : ℝ := weight_with_technology * (1 + incline_increase)

-- Theorem statement we want to prove
theorem weight_feel_when_lowered : weight_with_incline = 414 := by
  sorry

end weight_feel_when_lowered_l90_90418


namespace sqrt_4_of_10000000_eq_l90_90934

noncomputable def sqrt_4_of_10000000 : Real := Real.sqrt (Real.sqrt 10000000)

theorem sqrt_4_of_10000000_eq :
  sqrt_4_of_10000000 = 10 * Real.sqrt (Real.sqrt 10) := by
sorry

end sqrt_4_of_10000000_eq_l90_90934


namespace number_above_210_is_165_l90_90898

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2
def tetrahedral_number (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6
def row_start (k : ℕ) : ℕ := tetrahedral_number (k - 1) + 1

theorem number_above_210_is_165 :
  ∀ k, triangular_number k = 210 →
  ∃ n, n = 165 → 
  ∀ m, row_start (k - 1) ≤ m ∧ m < row_start k →
  m = 210 →
  n = m - triangular_number (k - 1) :=
  sorry

end number_above_210_is_165_l90_90898


namespace sum_of_first_five_primes_with_units_digit_3_l90_90986

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90986


namespace quadratic_factor_transformation_l90_90133

theorem quadratic_factor_transformation (x : ℝ) :
  x^2 - 6 * x + 5 = 0 → (x - 3)^2 = 14 := 
by
  sorry

end quadratic_factor_transformation_l90_90133


namespace hundred_days_from_friday_is_sunday_l90_90427

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l90_90427


namespace probability_Y_eq_neg2_l90_90813

noncomputable def two_point_distribution (p : ℝ) : ℕ → ℝ
| 0 => 1 - p
| 1 => p
| _ => 0

theorem probability_Y_eq_neg2 :
  let p := 0.6 in
  let X_dist := two_point_distribution p in
  let X := λ ω, if ω = 0 then 0 else 1 in
  let Y := λ ω, 3 * X ω - 2 in
  ∑ ω in {0, 1}, if Y ω = -2 then X_dist ω else 0 = 0.4 :=
by
  sorry

end probability_Y_eq_neg2_l90_90813


namespace speed_of_stream_l90_90416

theorem speed_of_stream (x : ℝ) (boat_speed : ℝ) (distance_one_way : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : distance_one_way = 7560) 
  (h3 : total_time = 960) 
  (h4 : (distance_one_way / (boat_speed + x)) + (distance_one_way / (boat_speed - x)) = total_time) 
  : x = 2 := 
  sorry

end speed_of_stream_l90_90416


namespace solve_for_q_l90_90404

theorem solve_for_q (m n q : ℕ) (h1 : 7/8 = m/96) (h2 : 7/8 = (n + m)/112) (h3 : 7/8 = (q - m)/144) :
  q = 210 :=
sorry

end solve_for_q_l90_90404


namespace cupcake_price_l90_90407

theorem cupcake_price
  (x : ℝ)
  (h1 : 5 * x + 6 * 1 + 4 * 2 + 15 * 0.6 = 33) : x = 2 :=
by
  sorry

end cupcake_price_l90_90407


namespace problem_statement_l90_90783

noncomputable def f (x : ℝ) : ℝ := 3^x + 3^(-x)

noncomputable def g (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem problem_statement : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, g (-x) = -g x) :=
by {
  sorry
}

end problem_statement_l90_90783


namespace rational_solutions_of_quadratic_l90_90490

theorem rational_solutions_of_quadratic (k : ℕ) (h_positive : k > 0) :
  (∃ p q : ℚ, p * p + 30 * p * q + k * (q * q) = 0) ↔ k = 9 ∨ k = 15 :=
sorry

end rational_solutions_of_quadratic_l90_90490


namespace jovana_total_shells_l90_90202

def initial_amount : ℕ := 5
def added_amount : ℕ := 23
def total_amount : ℕ := 28

theorem jovana_total_shells : initial_amount + added_amount = total_amount := by
  sorry

end jovana_total_shells_l90_90202


namespace problem_solution_l90_90363

theorem problem_solution (x : ℝ) (h : x * Real.log 4 / Real.log 3 = 1) : 
  2^x + 4^(-x) = 1 / 3 + Real.sqrt 3 :=
by 
  sorry

end problem_solution_l90_90363


namespace nonagon_diagonals_intersect_probability_l90_90247

theorem nonagon_diagonals_intersect_probability :
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let diagonals := total_pairs - n in
  let total_diagonals_pairs := Nat.choose diagonals 2 in
  let total_intersecting_diagonals := Nat.choose n 4 in
  (total_intersecting_diagonals.to_rat / total_diagonals_pairs.to_rat) = (6 / 13 : ℚ) :=
by
  sorry

end nonagon_diagonals_intersect_probability_l90_90247


namespace determine_irrational_option_l90_90925

def is_irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

def option_A : ℝ := 7
def option_B : ℝ := 0.5
def option_C : ℝ := abs (3 / 20 : ℚ)
def option_D : ℝ := 0.5151151115 -- Assume notation describes the stated behavior

theorem determine_irrational_option :
  is_irrational option_D ∧
  ¬ is_irrational option_A ∧
  ¬ is_irrational option_B ∧
  ¬ is_irrational option_C := 
by
  sorry

end determine_irrational_option_l90_90925


namespace find_second_sum_l90_90085

def total_sum : ℝ := 2691
def interest_rate_first : ℝ := 0.03
def interest_rate_second : ℝ := 0.05
def time_first : ℝ := 8
def time_second : ℝ := 3

theorem find_second_sum (x second_sum : ℝ) 
  (H : x + second_sum = total_sum)
  (H_interest : x * interest_rate_first * time_first = second_sum * interest_rate_second * time_second) :
  second_sum = 1656 :=
sorry

end find_second_sum_l90_90085


namespace stuart_initial_marbles_l90_90924

theorem stuart_initial_marbles
    (betty_marbles : ℕ)
    (stuart_marbles_after_given : ℕ)
    (percentage_given : ℚ)
    (betty_gave : ℕ):
    betty_marbles = 60 →
    stuart_marbles_after_given = 80 →
    percentage_given = 0.40 →
    betty_gave = percentage_given * betty_marbles →
    stuart_marbles_after_given = stuart_initial + betty_gave →
    stuart_initial = 56 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end stuart_initial_marbles_l90_90924


namespace ben_points_l90_90083

theorem ben_points (zach_points : ℝ) (total_points : ℝ) (ben_points : ℝ) 
  (h1 : zach_points = 42.0) 
  (h2 : total_points = 63) 
  (h3 : total_points = zach_points + ben_points) : 
  ben_points = 21 :=
by
  sorry

end ben_points_l90_90083


namespace parallelogram_isosceles_angles_l90_90893

def angle_sum_isosceles_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = b ∨ b = c ∨ a = c)

theorem parallelogram_isosceles_angles :
  ∀ (A B C D P : Type) (AB BC CD DA BD : ℝ)
    (angle_DAB angle_BCD angle_ABC angle_CDA angle_ABP angle_BAP angle_PBD angle_BDP angle_CBD angle_BCD : ℝ),
  AB ≠ BC →
  angle_DAB = 72 →
  angle_BCD = 72 →
  angle_ABC = 108 →
  angle_CDA = 108 →
  angle_sum_isosceles_triangle angle_ABP angle_BAP 108 →
  angle_sum_isosceles_triangle 72 72 angle_BDP →
  angle_sum_isosceles_triangle 108 36 36 →
  ∃! (ABP BPD BCD : Type),
   (angle_ABP = 36 ∧ angle_BAP = 36 ∧ angle_PBA = 108) ∧
   (angle_PBD = 72 ∧ angle_PDB = 72 ∧ angle_BPD = 36) ∧
   (angle_CBD = 108 ∧ angle_BCD = 36 ∧ angle_BDC = 36) :=
sorry

end parallelogram_isosceles_angles_l90_90893


namespace probability_diagonals_intersect_l90_90245

theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) :
  let diagonals := (n * (n - 3)) / 2,
      pairs_of_diagonals := diagonals.choose 2,
      combinations_of_four := n.choose 4 in
  pairs_of_diagonals > 0 ∧ 
  combinations_of_four / pairs_of_diagonals = 6 / 17 :=
by
  sorry

end probability_diagonals_intersect_l90_90245


namespace hydropump_output_l90_90054

theorem hydropump_output :
  ∀ (rate : ℕ) (time_hours : ℚ), 
    rate = 600 → 
    time_hours = 1.5 → 
    rate * time_hours = 900 :=
by
  intros rate time_hours rate_cond time_cond 
  sorry

end hydropump_output_l90_90054


namespace motorist_gallons_affordable_l90_90100

-- Definitions based on the conditions in the problem
def expected_gallons : ℕ := 12
def actual_price_per_gallon : ℕ := 150
def price_difference : ℕ := 30
def expected_price_per_gallon : ℕ := actual_price_per_gallon - price_difference
def total_initial_cents : ℕ := expected_gallons * expected_price_per_gallon

-- Theorem stating that given the conditions, the motorist can afford 9 gallons of gas
theorem motorist_gallons_affordable : 
  total_initial_cents / actual_price_per_gallon = 9 := 
by
  sorry

end motorist_gallons_affordable_l90_90100


namespace smallest_a_l90_90260

theorem smallest_a (a : ℕ) (h1 : a > 0) (h2 : (∀ b : ℕ, b > 0 → b < a → ∀ h3 : b > 0, ¬ (gcd b 72 > 1 ∧ gcd b 90 > 1)))
  (h3 : gcd a 72 > 1) (h4 : gcd a 90 > 1) : a = 2 :=
by
  sorry

end smallest_a_l90_90260


namespace find_a_l90_90502

-- Defining the curve y and its derivative y'
def y (x : ℝ) (a : ℝ) : ℝ := x^4 + a * x^2 + 1
def y' (x : ℝ) (a : ℝ) : ℝ := 4 * x^3 + 2 * a * x

theorem find_a (a : ℝ) : 
  y' (-1) a = 8 -> a = -6 := 
by
  -- proof here
  sorry

end find_a_l90_90502


namespace shane_gum_left_l90_90643

def elyse_initial_gum : ℕ := 100
def half (x : ℕ) := x / 2
def rick_gum : ℕ := half elyse_initial_gum
def shane_initial_gum : ℕ := half rick_gum
def chewed_gum : ℕ := 11

theorem shane_gum_left : shane_initial_gum - chewed_gum = 14 := by
  sorry

end shane_gum_left_l90_90643


namespace circle_area_with_radius_three_is_9pi_l90_90075

theorem circle_area_with_radius_three_is_9pi (r : ℝ) (h : r = 3) : Real.pi * r^2 = 9 * Real.pi := by
  sorry

end circle_area_with_radius_three_is_9pi_l90_90075


namespace parabola_through_point_l90_90178

-- Define the parabola equation property
def parabola (a x : ℝ) : ℝ := x^2 + (a+1) * x + a

-- Introduce the main problem statement
theorem parabola_through_point (a m : ℝ) (h : parabola a (-1) = m) : m = 0 :=
by
  sorry

end parabola_through_point_l90_90178


namespace expression_value_at_neg1_l90_90440

theorem expression_value_at_neg1
  (p q : ℤ)
  (h1 : p + q = 2016) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end expression_value_at_neg1_l90_90440


namespace gcd_12a_18b_l90_90185

theorem gcd_12a_18b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : Nat.gcd a b = 10) : Nat.gcd (12*a) (18*b) = 60 := 
by
  sorry

end gcd_12a_18b_l90_90185


namespace calc_num_articles_l90_90823

-- Definitions based on the conditions
def cost_price (C : ℝ) : ℝ := C
def selling_price (C : ℝ) : ℝ := 1.10000000000000004 * C
def num_articles (n : ℝ) (C : ℝ) (S : ℝ) : Prop := 55 * C = n * S

-- Proof Statement
theorem calc_num_articles (C : ℝ) : ∃ n : ℝ, num_articles n C (selling_price C) ∧ n = 50 :=
by sorry

end calc_num_articles_l90_90823


namespace total_number_of_students_l90_90596

-- Statement translating the problem conditions and conclusion
theorem total_number_of_students (rank_from_right rank_from_left total : ℕ) 
  (h_right : rank_from_right = 13) 
  (h_left : rank_from_left = 8) 
  (total_eq : total = rank_from_right + rank_from_left - 1) : 
  total = 20 := 
by 
  -- Proof is skipped
  sorry

end total_number_of_students_l90_90596


namespace sum_of_coefficients_l90_90650

noncomputable def P (x : ℤ) : ℤ := (x ^ 2 - 3 * x + 1) ^ 100

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

end sum_of_coefficients_l90_90650


namespace solve_quadratic_completing_square_l90_90715

theorem solve_quadratic_completing_square (x : ℝ) :
  (2 * x^2 - 4 * x - 1 = 0) ↔ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
by
  sorry

end solve_quadratic_completing_square_l90_90715


namespace student_movement_l90_90597

theorem student_movement :
  let students_total := 10
  let students_front := 3
  let students_back := 7
  let chosen_students := 2
  let ways_to_choose := Nat.choose students_back chosen_students
  let ways_to_place := (students_front + 1) * (students_front + chosen_students)
  in ways_to_choose * ways_to_place = 420 := by
  sorry

end student_movement_l90_90597


namespace neg_09_not_in_integers_l90_90756

def negative_numbers : Set ℝ := {x | x < 0}
def fractions : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

theorem neg_09_not_in_integers : -0.9 ∉ integers :=
by {
  sorry
}

end neg_09_not_in_integers_l90_90756


namespace find_k_range_l90_90673

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1 / 3)

def g (x k : ℝ) : ℝ :=
abs (x - k) + abs (x - 1)

theorem find_k_range (k : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g x2 k) → (k ≤ 3 / 4 ∨ k ≥ 5 / 4) :=
by
  sorry

end find_k_range_l90_90673


namespace kelly_chris_boxes_ratio_l90_90839

theorem kelly_chris_boxes_ratio (X : ℝ) (h : X > 0) :
  (0.4 * X) / (0.6 * X) = 2 / 3 :=
by sorry

end kelly_chris_boxes_ratio_l90_90839


namespace toothpaste_duration_l90_90733

theorem toothpaste_duration 
  (toothpaste_grams : ℕ)
  (dad_usage_per_brushing : ℕ) 
  (mom_usage_per_brushing : ℕ) 
  (anne_usage_per_brushing : ℕ) 
  (brother_usage_per_brushing : ℕ) 
  (brushes_per_day : ℕ) 
  (total_usage : ℕ) 
  (days : ℕ) 
  (h1 : toothpaste_grams = 105) 
  (h2 : dad_usage_per_brushing = 3) 
  (h3 : mom_usage_per_brushing = 2) 
  (h4 : anne_usage_per_brushing = 1) 
  (h5 : brother_usage_per_brushing = 1) 
  (h6 : brushes_per_day = 3)
  (h7 : total_usage = (3 * brushes_per_day) + (2 * brushes_per_day) + (1 * brushes_per_day) + (1 * brushes_per_day)) 
  (h8 : days = toothpaste_grams / total_usage) : 
  days = 5 :=
  sorry

end toothpaste_duration_l90_90733


namespace fewest_coach_handshakes_l90_90627

theorem fewest_coach_handshakes (n_A n_B k_A k_B : ℕ) (h1 : n_A = n_B + 2)
    (h2 : ((n_A * (n_A - 1)) / 2) + ((n_B * (n_B - 1)) / 2) + (n_A * n_B) + k_A + k_B = 620) :
  k_A + k_B = 189 := 
sorry

end fewest_coach_handshakes_l90_90627


namespace percent_students_own_cats_l90_90827

theorem percent_students_own_cats 
  (total_students : ℕ) (cat_owners : ℕ) (h1 : total_students = 300) (h2 : cat_owners = 45) :
  (cat_owners : ℚ) / total_students * 100 = 15 := 
by
  sorry

end percent_students_own_cats_l90_90827


namespace john_roommates_multiple_of_bob_l90_90012

theorem john_roommates_multiple_of_bob (bob_roommates john_roommates : ℕ) (multiple : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) 
  (h3 : john_roommates = multiple * bob_roommates + 5) : 
  multiple = 2 :=
by
  sorry

end john_roommates_multiple_of_bob_l90_90012


namespace general_solution_of_differential_eq_l90_90484

theorem general_solution_of_differential_eq (x y : ℝ) (C : ℝ) :
  (x^2 - y^2) * (y * (1 - C^2)) - 2 * (y * x) * (x) = 0 → (x^2 + y^2 = C * y) := by
  sorry

end general_solution_of_differential_eq_l90_90484


namespace minimal_reciprocal_sum_l90_90527

theorem minimal_reciprocal_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    (4 / m) + (1 / n) = (30 / (m * n)) → m = 10 ∧ n = 5 :=
sorry

end minimal_reciprocal_sum_l90_90527


namespace fill_time_calculation_l90_90225

-- Definitions based on conditions
def pool_volume : ℝ := 24000
def number_of_hoses : ℕ := 6
def water_per_hose_per_minute : ℝ := 3
def minutes_per_hour : ℝ := 60

-- Theorem statement translating the mathematically equivalent proof problem
theorem fill_time_calculation :
  pool_volume / (number_of_hoses * water_per_hose_per_minute * minutes_per_hour) = 22 :=
by
  sorry

end fill_time_calculation_l90_90225


namespace product_of_terms_form_l90_90584

theorem product_of_terms_form 
  (a b c d : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ p q : ℝ, 
    (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 
    ∧ 0 ≤ p 
    ∧ 0 ≤ q := 
by
  let p := a * c + 5 * b * d
  let q := a * d + b * c
  use p, q
  sorry

end product_of_terms_form_l90_90584


namespace sin_neg_135_degree_l90_90633

theorem sin_neg_135_degree : sin (-(135 * Real.pi / 180)) = - (Real.sqrt 2 / 2) :=
by 
  -- Here, we need to use the known properties and the equivalences given
  sorry

end sin_neg_135_degree_l90_90633


namespace normal_time_to_finish_bs_l90_90739

theorem normal_time_to_finish_bs (P : ℕ) (H1 : P = 5) (H2 : ∀ total_time, total_time = 6 → total_time = (3 / 4) * (P + B)) : B = (8 - P) :=
by sorry

end normal_time_to_finish_bs_l90_90739


namespace simplify_expression_l90_90349

theorem simplify_expression (a b : ℝ) (h1 : 2 * b - a < 3) (h2 : 2 * a - b < 5) : 
  -abs (2 * b - a - 7) - abs (b - 2 * a + 8) + abs (a + b - 9) = -6 :=
by
  sorry

end simplify_expression_l90_90349


namespace lcm_gcf_ratio_280_450_l90_90437

open Nat

theorem lcm_gcf_ratio_280_450 :
  let a := 280
  let b := 450
  lcm a b / gcd a b = 1260 :=
by
  let a := 280
  let b := 450
  sorry

end lcm_gcf_ratio_280_450_l90_90437


namespace train_speed_l90_90464

theorem train_speed 
  (length_train : ℕ) 
  (time_crossing : ℕ) 
  (speed_kmph : ℕ)
  (h_length : length_train = 120)
  (h_time : time_crossing = 9)
  (h_speed : speed_kmph = 48) : 
  length_train / time_crossing * 3600 / 1000 = speed_kmph := 
by 
  sorry

end train_speed_l90_90464


namespace inequality_system_integer_solutions_l90_90553

theorem inequality_system_integer_solutions :
  { x : ℤ | 5 * x + 1 > 3 * (x - 1) ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {-1, 0, 1, 2} := by
  sorry

end inequality_system_integer_solutions_l90_90553


namespace second_chick_eats_52_l90_90709

theorem second_chick_eats_52 (days : ℕ) (first_chick_eats : ℕ → ℕ) (second_chick_eats : ℕ → ℕ) :
  (∀ n, first_chick_eats n + second_chick_eats n = 12) →
  (∃ a b, first_chick_eats a = 7 ∧ second_chick_eats a = 5 ∧
          first_chick_eats b = 7 ∧ second_chick_eats b = 5 ∧
          12 * days = first_chick_eats a * 2 + first_chick_eats b * 6 + second_chick_eats a * 2 + second_chick_eats b * 6) →
  (first_chick_eats a * 2 + first_chick_eats b * 6 = 44) →
  (second_chick_eats a * 2 + second_chick_eats b * 6 = 52) :=
by
  sorry

end second_chick_eats_52_l90_90709


namespace original_number_of_professors_l90_90027

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l90_90027


namespace mike_picked_64_peaches_l90_90040

theorem mike_picked_64_peaches :
  ∀ (initial peaches_given total final_picked : ℕ),
    initial = 34 →
    peaches_given = 12 →
    total = 86 →
    final_picked = total - (initial - peaches_given) →
    final_picked = 64 :=
by
  intros
  sorry

end mike_picked_64_peaches_l90_90040


namespace min_squared_sum_l90_90186

theorem min_squared_sum {x y z : ℝ} (h : 2 * x + y + 2 * z = 6) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_squared_sum_l90_90186


namespace toms_animal_robots_l90_90707

theorem toms_animal_robots (h : ∀ (m t : ℕ), t = 2 * m) (hmichael : 8 = m) : ∃ t, t = 16 := 
by
  sorry

end toms_animal_robots_l90_90707


namespace total_highlighters_is_49_l90_90196

-- Define the number of highlighters of each color
def pink_highlighters : Nat := 15
def yellow_highlighters : Nat := 12
def blue_highlighters : Nat := 9
def green_highlighters : Nat := 7
def purple_highlighters : Nat := 6

-- Define the total number of highlighters
def total_highlighters : Nat := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters

-- Statement that the total number of highlighters should be 49
theorem total_highlighters_is_49 : total_highlighters = 49 := by
  sorry

end total_highlighters_is_49_l90_90196


namespace range_of_2x_minus_y_l90_90655

theorem range_of_2x_minus_y (x y : ℝ) (hx : 0 < x ∧ x < 4) (hy : 0 < y ∧ y < 6) : -6 < 2 * x - y ∧ 2 * x - y < 8 := 
sorry

end range_of_2x_minus_y_l90_90655


namespace ratio_of_AC_to_BD_l90_90863

theorem ratio_of_AC_to_BD (A B C D : ℝ) (AB BC AD AC BD : ℝ) 
  (h1 : AB = 2) (h2 : BC = 5) (h3 : AD = 14) (h4 : AC = AB + BC) (h5 : BD = AD - AB) :
  AC / BD = 7 / 12 := by
  sorry

end ratio_of_AC_to_BD_l90_90863


namespace maximum_k_l90_90658

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- Prove that the maximum integer value k satisfying k(x - 2) < f(x) for all x > 2 is 4.
theorem maximum_k (x : ℝ) (hx : x > 2) : ∃ k : ℤ, k = 4 ∧ (∀ x > 2, k * (x - 2) < f x) :=
sorry

end maximum_k_l90_90658


namespace inverse_proportion_relation_l90_90496

theorem inverse_proportion_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₂ < y₁ ∧ y₁ < 0 := 
sorry

end inverse_proportion_relation_l90_90496


namespace find_ratio_l90_90822

variables (a b c d : ℝ)

def condition1 : Prop := a / b = 5
def condition2 : Prop := b / c = 1 / 4
def condition3 : Prop := c^2 / d = 16

theorem find_ratio (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c d) :
  d / a = 1 / 25 :=
sorry

end find_ratio_l90_90822


namespace reduced_price_l90_90084

theorem reduced_price (P R : ℝ) (Q : ℝ) (h₁ : R = 0.80 * P) 
                      (h₂ : 800 = Q * P) 
                      (h₃ : 800 = (Q + 5) * R) 
                      : R = 32 :=
by
  -- Code that proves the theorem goes here.
  sorry

end reduced_price_l90_90084


namespace maximize_expression_l90_90379

theorem maximize_expression :
  ∀ (a b c d e : ℕ),
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → b ≠ c → b ≠ d → b ≠ e → c ≠ d → c ≠ e → d ≠ e →
    (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 6) → 
    (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 6) →
    (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6) →
    (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6) →
    (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4 ∨ e = 6) →
    ((a : ℚ) / 2 + (d : ℚ) / e * (c / b)) ≤ 9 :=
by
  sorry

end maximize_expression_l90_90379


namespace find_value_of_m_l90_90348

open Real

theorem find_value_of_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = sqrt 10 := by
  sorry

end find_value_of_m_l90_90348


namespace min_socks_for_pairs_l90_90001

-- Definitions for conditions
def pairs_of_socks : ℕ := 4
def sizes : ℕ := 2
def colors : ℕ := 2

-- Theorem statement
theorem min_socks_for_pairs : 
  ∃ n, n = 7 ∧ 
  ∀ (socks : ℕ), socks >= pairs_of_socks → socks ≥ 7 :=
sorry

end min_socks_for_pairs_l90_90001


namespace day_of_week_in_100_days_l90_90426

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l90_90426


namespace rectangles_with_equal_perimeters_can_have_different_shapes_l90_90073

theorem rectangles_with_equal_perimeters_can_have_different_shapes (l₁ w₁ l₂ w₂ : ℝ) 
  (h₁ : l₁ + w₁ = l₂ + w₂) : (l₁ ≠ l₂ ∨ w₁ ≠ w₂) :=
by
  sorry

end rectangles_with_equal_perimeters_can_have_different_shapes_l90_90073


namespace initial_number_of_professors_l90_90022

theorem initial_number_of_professors (p : ℕ) :
  (∃ p, (6480 / p : ℚ) < (11200 / (p + 3) : ℚ) ∧ 
   6480 % p = 0 ∧ 11200 % (p + 3) = 0 ∧ p > 4) → 
  p = 5 := 
sorry

end initial_number_of_professors_l90_90022


namespace german_team_goals_l90_90130

theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : x % 2 = 1) 
  (h_correct : (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)):
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := 
by {
  -- Proof to be filled in
  sorry
}

end german_team_goals_l90_90130


namespace gina_college_expenses_l90_90492

theorem gina_college_expenses
  (credits : ℕ)
  (cost_per_credit : ℕ)
  (num_textbooks : ℕ)
  (cost_per_textbook : ℕ)
  (facilities_fee : ℕ)
  (H_credits : credits = 14)
  (H_cost_per_credit : cost_per_credit = 450)
  (H_num_textbooks : num_textbooks = 5)
  (H_cost_per_textbook : cost_per_textbook = 120)
  (H_facilities_fee : facilities_fee = 200)
  : (credits * cost_per_credit) + (num_textbooks * cost_per_textbook) + facilities_fee = 7100 := by
  sorry

end gina_college_expenses_l90_90492


namespace sum_of_first_five_primes_with_units_digit_3_l90_90976

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90976


namespace distance_traveled_l90_90269

theorem distance_traveled (speed time : ℕ) (h_speed : speed = 20) (h_time : time = 8) : 
  speed * time = 160 := 
by
  -- Solution proof goes here
  sorry

end distance_traveled_l90_90269


namespace simplified_expression_l90_90867

-- Non-computable context since we are dealing with square roots and division
noncomputable def expr (x : ℝ) : ℝ := ((x / (x - 1)) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem simplified_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : expr x = Real.sqrt 2 / 2 := by
  sorry

end simplified_expression_l90_90867


namespace find_ratio_l90_90636

theorem find_ratio (x y c d : ℝ) (h₁ : 4 * x - 2 * y = c) (h₂ : 5 * y - 10 * x = d) (h₃ : d ≠ 0) : c / d = 0 :=
sorry

end find_ratio_l90_90636


namespace totalCandlesInHouse_l90_90838

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l90_90838


namespace miranda_pillows_l90_90211

-- Define the conditions in the problem
def pounds_per_pillow := 2
def feathers_per_pound := 300
def total_feathers := 3600

-- Define the goal in terms of these conditions
def num_pillows : Nat :=
  (total_feathers / feathers_per_pound) / pounds_per_pillow

-- Prove that the number of pillows Miranda can stuff is 6
theorem miranda_pillows : num_pillows = 6 :=
by
  sorry

end miranda_pillows_l90_90211


namespace german_team_goals_l90_90118

/-- Given the conditions from three journalists and knowing that only two of them can be correct,
we need to prove the possible values for the number of goals (x) scored by the German team during the championship. --/
theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : odd x) 
  (h4 : (h1 ∨ h2 ∨ h3) ∧ ¬(h1 ∧ h2 ∧ h3)) :
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := sorry

end german_team_goals_l90_90118


namespace gcd_of_X_and_Y_l90_90685

theorem gcd_of_X_and_Y (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : 5 * X = 4 * Y) :
  Nat.gcd X Y = 9 := 
sorry

end gcd_of_X_and_Y_l90_90685


namespace highest_car_color_is_blue_l90_90734

def total_cars : ℕ := 24
def red_cars : ℕ := total_cars / 4
def blue_cars : ℕ := red_cars + 6
def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem highest_car_color_is_blue :
  blue_cars > red_cars ∧ blue_cars > yellow_cars :=
by sorry

end highest_car_color_is_blue_l90_90734


namespace queue_adjustments_l90_90598

theorem queue_adjustments (students_front : ℕ) (students_back : ℕ) (students_move : ℕ) :
  students_front = 3 → students_back = 7 → students_move = 2 → 
  (∃ ways_to_adjust : ℕ, ways_to_adjust = 420) :=
by
  intros h1 h2 h3
  use 420
  sorry

end queue_adjustments_l90_90598


namespace nonagon_diagonals_intersect_probability_l90_90246

theorem nonagon_diagonals_intersect_probability :
  let n := 9 in
  let total_pairs := Nat.choose n 2 in
  let diagonals := total_pairs - n in
  let total_diagonals_pairs := Nat.choose diagonals 2 in
  let total_intersecting_diagonals := Nat.choose n 4 in
  (total_intersecting_diagonals.to_rat / total_diagonals_pairs.to_rat) = (6 / 13 : ℚ) :=
by
  sorry

end nonagon_diagonals_intersect_probability_l90_90246


namespace planar_figure_area_l90_90550

noncomputable def side_length : ℝ := 10
noncomputable def area_of_square : ℝ := side_length * side_length
noncomputable def number_of_squares : ℕ := 6
noncomputable def total_area_of_planar_figure : ℝ := number_of_squares * area_of_square

theorem planar_figure_area : total_area_of_planar_figure = 600 :=
by
  sorry

end planar_figure_area_l90_90550


namespace trig_identity_l90_90657

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 :=
by 
  sorry

end trig_identity_l90_90657


namespace gcd_lcm_of_consecutive_naturals_l90_90664

theorem gcd_lcm_of_consecutive_naturals (m : ℕ) (h : m > 0) (n : ℕ) (hn : n = m + 1) :
  gcd m n = 1 ∧ lcm m n = m * n :=
by
  sorry

end gcd_lcm_of_consecutive_naturals_l90_90664


namespace minimum_value_l90_90668

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y = 1) :
  ∀ (z : ℝ), z = (1/x + 1/y) → z ≥ 3 + 2*Real.sqrt 2 :=
by
  sorry

end minimum_value_l90_90668


namespace cs_share_l90_90450

-- Definitions for the conditions
def daily_work (days: ℕ) : ℚ := 1 / days

def total_work_contribution (a_days: ℕ) (b_days: ℕ) (c_days: ℕ): ℚ := 
  daily_work a_days + daily_work b_days + daily_work c_days

def total_payment (payment: ℕ) (work_contribution: ℚ) : ℚ := 
  payment * work_contribution

-- The mathematically equivalent proof problem
theorem cs_share (a_days: ℕ) (b_days: ℕ) (total_days : ℕ) (payment: ℕ) : 
  a_days = 6 → b_days = 8 → total_days = 3 → payment = 1200 →
  total_payment payment (daily_work total_days - (daily_work a_days + daily_work b_days)) = 50 :=
sorry

end cs_share_l90_90450


namespace day_100_days_from_friday_l90_90435

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l90_90435


namespace german_team_goals_l90_90120

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l90_90120


namespace polynomial_sum_l90_90187

theorem polynomial_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 :=
by
  sorry

end polynomial_sum_l90_90187


namespace product_of_nine_integers_16_to_30_equals_15_factorial_l90_90204

noncomputable def factorial (n : Nat) : Nat :=
match n with
| 0     => 1
| (n+1) => (n+1) * factorial n

theorem product_of_nine_integers_16_to_30_equals_15_factorial :
  (16 * 18 * 20 * 21 * 22 * 25 * 26 * 27 * 28) = factorial 15 := 
by sorry

end product_of_nine_integers_16_to_30_equals_15_factorial_l90_90204


namespace soda_cost_132_cents_l90_90583

theorem soda_cost_132_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s + 30 = 510)
  (h2 : 2 * b + 3 * s = 540) 
  : s = 132 :=
by
  sorry

end soda_cost_132_cents_l90_90583


namespace eccentricity_of_ellipse_l90_90613

noncomputable def eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem eccentricity_of_ellipse
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (l : ℝ → ℝ) (hl : l 0 = 0)
  (h_intersects : ∃ M N : ℝ × ℝ, M ≠ N ∧ (M.1 / a)^2 + (M.2 / b)^2 = 1 ∧ (N.1 / a)^2 + (N.2 / b)^2 = 1 ∧ l M.1 = M.2 ∧ l N.1 = N.2)
  (P : ℝ × ℝ) (hP : (P.1 / a)^2 + (P.2 / b)^2 = 1 ∧ P ≠ (0, 0))
  (h_product_slopes : ∀ (Mx Nx Px : ℝ) (k : ℝ),
    l Mx = k * Mx →
    l Nx = k * Nx →
    l Px ≠ k * Px →
    ((k * Mx - P.2) / (Mx - P.1)) * ((k * Nx - P.2) / (Nx - P.1)) = -1/3) :
  eccentricity a b h1 h2 = Real.sqrt (2 / 3) :=
by
  sorry

end eccentricity_of_ellipse_l90_90613


namespace marble_prob_l90_90479

theorem marble_prob
  (a b x y m n : ℕ)
  (h1 : a + b = 30)
  (h2 : (x : ℚ) / a * (y : ℚ) / b = 4 / 9)
  (h3 : x * y = 36)
  (h4 : Nat.gcd m n = 1)
  (h5 : (a - x : ℚ) / a * (b - y) / b = m / n) :
  m + n = 29 := 
sorry

end marble_prob_l90_90479


namespace exists_x_geq_zero_l90_90398

theorem exists_x_geq_zero (h : ∀ x : ℝ, x^2 + x - 1 < 0) : ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
sorry

end exists_x_geq_zero_l90_90398


namespace f_50_value_l90_90389

def f : ℝ → ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9 * x^2 - 15 * x

theorem f_50_value : f 50 = 146 :=
by
  sorry

end f_50_value_l90_90389


namespace max_rides_day1_max_rides_day2_l90_90784

open List 

def daily_budget : ℤ := 10

def ride_prices_day1 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 5), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6)]

def ride_prices_day2 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 7), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6), ("Haunted house", 4)]

def max_rides (budget : ℤ) (prices : List (String × ℤ)) : ℤ :=
  sorry -- We'll assume this calculates the max number of rides correctly based on the given budget and prices.

theorem max_rides_day1 : max_rides daily_budget ride_prices_day1 = 3 := by
  sorry 

theorem max_rides_day2 : max_rides daily_budget ride_prices_day2 = 3 := by
  sorry 

end max_rides_day1_max_rides_day2_l90_90784


namespace train_length_is_correct_l90_90463

noncomputable def length_of_train (speed_train_kmph : ℕ) (speed_man_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  let relative_speed_kmph := (speed_train_kmph + speed_man_kmph)
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (5 / 18)
  relative_speed_mps * (time_seconds : ℝ)

theorem train_length_is_correct :
  length_of_train 60 6 3 = 54.99 := 
by
  sorry

end train_length_is_correct_l90_90463


namespace cooper_remaining_pies_l90_90478

def total_pies (pies_per_day : ℕ) (days : ℕ) : ℕ := pies_per_day * days

def remaining_pies (total : ℕ) (eaten : ℕ) : ℕ := total - eaten

theorem cooper_remaining_pies :
  remaining_pies (total_pies 7 12) 50 = 34 :=
by sorry

end cooper_remaining_pies_l90_90478


namespace remainder_of_67_pow_67_plus_67_mod_68_l90_90443

theorem remainder_of_67_pow_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  -- Add the conditions and final proof step
  sorry

end remainder_of_67_pow_67_plus_67_mod_68_l90_90443


namespace find_m_plus_n_l90_90599

theorem find_m_plus_n (PQ QR RP : ℕ) (x y : ℕ) 
  (h1 : PQ = 26) 
  (h2 : QR = 29) 
  (h3 : RP = 25) 
  (h4 : PQ = x + y) 
  (h5 : QR = x + (QR - x))
  (h6 : RP = x + (RP - x)) : 
  30 = 29 + 1 :=
by
  -- assumptions already provided in problem statement
  sorry

end find_m_plus_n_l90_90599


namespace smallest_a_no_inverse_mod_72_90_l90_90262

theorem smallest_a_no_inverse_mod_72_90 :
  ∃ (a : ℕ), a > 0 ∧ ∀ b : ℕ, (b > 0 → gcd b 72 > 1 ∧ gcd b 90 > 1 → b ≥ a) ∧ gcd a 72 > 1 ∧ gcd a 90 > 1 ∧ a = 6 :=
by sorry

end smallest_a_no_inverse_mod_72_90_l90_90262


namespace calculate_number_of_models_l90_90557

-- Define the constants and conditions
def time_per_set : ℕ := 2  -- time per set in minutes
def sets_bathing_suits : ℕ := 2  -- number of bathing suit sets each model wears
def sets_evening_wear : ℕ := 3  -- number of evening wear sets each model wears
def total_show_time : ℕ := 60  -- total show time in minutes

-- Calculate the total time each model takes
def model_time : ℕ := 
  (sets_bathing_suits + sets_evening_wear) * time_per_set

-- Proof problem statement
theorem calculate_number_of_models : 
  (total_show_time / model_time) = 6 := by
  sorry

end calculate_number_of_models_l90_90557


namespace vasya_pastry_trick_l90_90281

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l90_90281


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90964

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90964


namespace candy_cost_l90_90559

theorem candy_cost (candy_cost_in_cents : ℕ) (pieces : ℕ) (dollar_in_cents : ℕ)
  (h1 : candy_cost_in_cents = 2) (h2 : pieces = 500) (h3 : dollar_in_cents = 100) :
  (pieces * candy_cost_in_cents) / dollar_in_cents = 10 :=
by
  sorry

end candy_cost_l90_90559


namespace tangent_perpendicular_intersection_x_4_l90_90356

noncomputable def f (x : ℝ) := (x^2 / 4) - (4 * Real.log x)
noncomputable def f' (x : ℝ) := (1/2 : ℝ) * x - 4 / x

theorem tangent_perpendicular_intersection_x_4 :
  ∀ x : ℝ, (0 < x) → (f' x = 1) → (x = 4) :=
by {
  sorry
}

end tangent_perpendicular_intersection_x_4_l90_90356


namespace angle_same_terminal_side_l90_90718

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 95 = -265 + k * 360 :=
by
  use 1
  norm_num

end angle_same_terminal_side_l90_90718


namespace evaluate_expression_l90_90153

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := sorry

end evaluate_expression_l90_90153


namespace current_babysitter_hourly_rate_l90_90041

-- Define variables
def new_babysitter_hourly_rate := 12
def extra_charge_per_scream := 3
def hours_hired := 6
def number_of_screams := 2
def cost_difference := 18

-- Define the total cost calculations
def new_babysitter_total_cost :=
  new_babysitter_hourly_rate * hours_hired + extra_charge_per_scream * number_of_screams

def current_babysitter_total_cost :=
  new_babysitter_total_cost + cost_difference

theorem current_babysitter_hourly_rate :
  current_babysitter_total_cost / hours_hired = 16 := by
  sorry

end current_babysitter_hourly_rate_l90_90041


namespace problem_sol_52_l90_90737

theorem problem_sol_52 
  (x y: ℝ)
  (h1: x + y = 7)
  (h2: 4 * x * y = 7)
  (a b c d : ℕ)
  (hx_form : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d)
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hc_pos : 0 < c)
  (hd_pos : 0 < d)
  : a + b + c + d = 52 := sorry

end problem_sol_52_l90_90737


namespace totalCandlesInHouse_l90_90837

-- Definitions for the problem's conditions
def bedroomCandles : ℕ := 20
def livingRoomCandles : ℕ := bedroomCandles / 2
def donovanCandles : ℕ := 20

-- Problem to prove
theorem totalCandlesInHouse : bedroomCandles + livingRoomCandles + donovanCandles = 50 := by
  sorry

end totalCandlesInHouse_l90_90837


namespace german_team_goals_l90_90117

/-- Given the conditions from three journalists and knowing that only two of them can be correct,
we need to prove the possible values for the number of goals (x) scored by the German team during the championship. --/
theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : odd x) 
  (h4 : (h1 ∨ h2 ∨ h3) ∧ ¬(h1 ∧ h2 ∧ h3)) :
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := sorry

end german_team_goals_l90_90117


namespace monotonicity_condition_l90_90824

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x

theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, f a x ≥ f a 1) ↔ a ∈ Set.Ici 2 :=
by
  sorry

end monotonicity_condition_l90_90824


namespace probability_odd_sum_of_six_selected_primes_l90_90869

open Finset

def firstTwelvePrimes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

theorem probability_odd_sum_of_six_selected_primes :
  let selections := (firstTwelvePrimes.choose 6)
  let odd_sum (s : Finset ℕ) := (s.sum id) % 2 = 1
  let total_ways := selections.card
  let odd_ways := (selections.filter odd_sum).card
  total_ways > 0 -> (odd_ways / total_ways : ℚ) = 1 / 2 :=
by 
  sorry

end probability_odd_sum_of_six_selected_primes_l90_90869


namespace probability_diagonals_intersect_nonagon_l90_90250

theorem probability_diagonals_intersect_nonagon :
  let n := 9 in
  let total_pairs_points := nat.choose n 2 in
  let num_sides := n in
  let num_diagonals := total_pairs_points - num_sides in
  let total_pairs_diagonals := nat.choose num_diagonals 2 in
  let intersecting_pairs := nat.choose n 4 in
  (intersecting_pairs : ℚ) / total_pairs_diagonals = 14 / 39 :=
by
  sorry

end probability_diagonals_intersect_nonagon_l90_90250


namespace find_other_number_l90_90848

def integers_three_and_four_sum (a b : ℤ) : Prop :=
  3 * a + 4 * b = 131

def one_of_the_numbers_is (x : ℤ) : Prop :=
  x = 17

theorem find_other_number (a b : ℤ) (h1 : integers_three_and_four_sum a b) (h2 : one_of_the_numbers_is a ∨ one_of_the_numbers_is b) :
  (a = 21 ∨ b = 21) :=
sorry

end find_other_number_l90_90848


namespace true_discount_correct_l90_90071

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  (BD * FV) / (BD + FV)

theorem true_discount_correct :
  true_discount 270 54 = 45 :=
by
  sorry

end true_discount_correct_l90_90071


namespace cost_of_watch_l90_90327

variable (saved amount_needed total_cost : ℕ)

-- Conditions
def connie_saved : Prop := saved = 39
def connie_needs : Prop := amount_needed = 16

-- Theorem to prove
theorem cost_of_watch : connie_saved saved → connie_needs amount_needed → total_cost = 55 :=
by
  sorry

end cost_of_watch_l90_90327


namespace complex_number_quadrant_l90_90004

noncomputable def complex_quadrant : ℂ → String
| z => if z.re > 0 ∧ z.im > 0 then "First quadrant"
      else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
      else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
      else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
      else "On the axis"

theorem complex_number_quadrant (z : ℂ) (h : z = (5 : ℂ) / (2 + I)) : complex_quadrant z = "Fourth quadrant" :=
by
  sorry

end complex_number_quadrant_l90_90004


namespace b_41_mod_49_l90_90034

noncomputable def b (n : ℕ) : ℕ :=
  6 ^ n + 8 ^ n

theorem b_41_mod_49 : b 41 % 49 = 35 := by
  sorry

end b_41_mod_49_l90_90034


namespace rectangular_prism_volume_l90_90397

theorem rectangular_prism_volume (h : ℝ) : 
  ∃ (V : ℝ), V = 120 * h :=
by
  sorry

end rectangular_prism_volume_l90_90397


namespace pyramid_structure_l90_90522

variables {d e f a b c h i j g : ℝ}

theorem pyramid_structure (h_val : h = 16)
                         (i_val : i = 48)
                         (j_val : j = 72)
                         (g_val : g = 8)
                         (d_def : d = b * a)
                         (e_def1 : e = b * c) 
                         (e_def2 : e = d * a)
                         (f_def : f = c * a)
                         (h_def : h = d * b)
                         (i_def : i = d * a)
                         (j_def : j = e * c)
                         (g_def : g = f * c) : 
   a = 3 ∧ b = 1 ∧ c = 1.5 :=
by sorry

end pyramid_structure_l90_90522


namespace find_k_l90_90759

-- Define the conditions
variables (x y k : ℕ)
axiom part_sum : x + y = 36
axiom first_part : x = 19
axiom value_eq : 8 * x + k * y = 203

-- Prove that k is 3
theorem find_k : k = 3 :=
by
  -- Insert your proof here
  sorry

end find_k_l90_90759


namespace sale_in_second_month_l90_90769

-- Define the constants for known sales and average requirement
def sale_first_month : Int := 8435
def sale_third_month : Int := 8855
def sale_fourth_month : Int := 9230
def sale_fifth_month : Int := 8562
def sale_sixth_month : Int := 6991
def average_sale_per_month : Int := 8500
def number_of_months : Int := 6

-- Define the total sales required for six months
def total_sales_required : Int := average_sale_per_month * number_of_months

-- Define the total known sales excluding the second month
def total_known_sales : Int := sale_first_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- The statement to prove: the sale in the second month is 8927
theorem sale_in_second_month : 
  total_sales_required - total_known_sales = 8927 := 
by
  sorry

end sale_in_second_month_l90_90769


namespace find_k_value_l90_90061

theorem find_k_value (k : ℝ) : (∀ (x y : ℝ), (x = 2 ∧ y = 5) → y = k * x + 3) → k = 1 := 
by 
  intro h
  have h1 := h 2 5 ⟨rfl, rfl⟩
  linarith

end find_k_value_l90_90061


namespace sam_total_cans_l90_90755

theorem sam_total_cans (bags_saturday bags_sunday bags_total cans_per_bag total_cans : ℕ)
    (h1 : bags_saturday = 3)
    (h2 : bags_sunday = 4)
    (h3 : bags_total = bags_saturday + bags_sunday)
    (h4 : cans_per_bag = 9)
    (h5 : total_cans = bags_total * cans_per_bag) : total_cans = 63 :=
sorry

end sam_total_cans_l90_90755


namespace german_team_goals_l90_90121

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l90_90121


namespace vasya_pastry_trick_l90_90280

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l90_90280


namespace smallest_difference_l90_90171

-- Definition for the given problem conditions.
def side_lengths (AB BC AC : ℕ) : Prop := 
  AB + BC + AC = 2023 ∧ AB < BC ∧ BC ≤ AC ∧ 
  AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB

theorem smallest_difference (AB BC AC : ℕ) 
  (h: side_lengths AB BC AC) : 
  ∃ (AB BC AC : ℕ), side_lengths AB BC AC ∧ (BC - AB = 1) :=
by
  sorry

end smallest_difference_l90_90171


namespace sum_of_first_five_primes_with_units_digit_three_l90_90974

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l90_90974


namespace functions_are_same_l90_90621

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_same_l90_90621


namespace last_digit_base_4_of_77_l90_90586

theorem last_digit_base_4_of_77 : (77 % 4) = 1 :=
by
  sorry

end last_digit_base_4_of_77_l90_90586


namespace number_of_boys_l90_90732

theorem number_of_boys (x g : ℕ) 
  (h1 : x + g = 150) 
  (h2 : g = (x * 150) / 100) 
  : x = 60 := 
by 
  sorry

end number_of_boys_l90_90732


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90966

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90966


namespace ratio_calc_l90_90930

theorem ratio_calc :
  (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484) /
  ((8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)) = -423 := 
by
  sorry

end ratio_calc_l90_90930


namespace smaller_of_two_digit_product_l90_90729

theorem smaller_of_two_digit_product (a b : ℕ) (h1 : a * b = 4896) (h2 : 10 ≤ a ∧ a < 100) (h3 : 10 ≤ b ∧ b < 100) : min a b = 32 :=
sorry

end smaller_of_two_digit_product_l90_90729


namespace union_complement_eq_l90_90391

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem union_complement_eq : M ∪ (U \ N) = {0, 1, 2} := by
  sorry

end union_complement_eq_l90_90391


namespace smallest_n_value_l90_90941

theorem smallest_n_value (n : ℕ) (h : 15 * n - 2 ≡ 0 [MOD 11]) : n = 6 :=
sorry

end smallest_n_value_l90_90941


namespace find_cost_price_l90_90904

variables (SP CP : ℝ)
variables (discount profit : ℝ)
variable (h1 : SP = 24000)
variable (h2 : discount = 0.10)
variable (h3 : profit = 0.08)

theorem find_cost_price 
  (h1 : SP = 24000)
  (h2 : discount = 0.10)
  (h3 : profit = 0.08)
  (h4 : SP * (1 - discount) = CP * (1 + profit)) :
  CP = 20000 := 
by
  sorry

end find_cost_price_l90_90904


namespace circumference_given_area_l90_90903

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circumference_given_area :
  (∃ r : ℝ, area_of_circle r = 616) →
  circumference_of_circle 14 = 2 * Real.pi * 14 :=
by
  sorry

end circumference_given_area_l90_90903


namespace Carol_weight_equals_nine_l90_90394

-- conditions in Lean definitions
def Mildred_weight : ℤ := 59
def weight_difference : ℤ := 50

-- problem statement to prove in Lean 4
theorem Carol_weight_equals_nine (Carol_weight : ℤ) :
  Mildred_weight = Carol_weight + weight_difference → Carol_weight = 9 :=
by
  sorry

end Carol_weight_equals_nine_l90_90394


namespace day_of_week_in_100_days_l90_90424

theorem day_of_week_in_100_days (start_day : ℕ) (h : start_day = 5) : 
  (start_day + 100) % 7 = 0 := 
by
  cases h with 
  | rfl => -- start_day is Friday, which is represented as 5
  sorry

end day_of_week_in_100_days_l90_90424


namespace tim_grew_cantaloupes_l90_90166

theorem tim_grew_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) :
  ∃ tim_cantaloupes : ℕ, tim_cantaloupes = total_cantaloupes - fred_cantaloupes ∧ tim_cantaloupes = 44 :=
by
  sorry

end tim_grew_cantaloupes_l90_90166


namespace sin_cos_identity_l90_90170

theorem sin_cos_identity (α : ℝ) (h1 : Real.sin (α - Real.pi / 6) = 1 / 3) :
    Real.sin (2 * α - Real.pi / 6) + Real.cos (2 * α) = 7 / 9 :=
sorry

end sin_cos_identity_l90_90170


namespace area_shaded_region_is_75_l90_90306

-- Define the side length of the larger square
def side_length_large_square : ℝ := 10

-- Define the side length of the smaller square
def side_length_small_square : ℝ := 5

-- Define the area of the larger square
def area_large_square : ℝ := side_length_large_square ^ 2

-- Define the area of the smaller square
def area_small_square : ℝ := side_length_small_square ^ 2

-- Define the area of the shaded region
def area_shaded_region : ℝ := area_large_square - area_small_square

-- The theorem that states the area of the shaded region is 75 square units
theorem area_shaded_region_is_75 : area_shaded_region = 75 := by
  -- The proof will be filled in here when required
  sorry

end area_shaded_region_is_75_l90_90306


namespace shauna_fifth_test_score_l90_90866

theorem shauna_fifth_test_score :
  ∀ (a1 a2 a3 a4: ℕ), a1 = 76 → a2 = 94 → a3 = 87 → a4 = 92 →
  (∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / 5 = 85 ∧ a5 = 76) :=
by
  sorry

end shauna_fifth_test_score_l90_90866


namespace vasya_pastry_trick_l90_90282

theorem vasya_pastry_trick :
  ∀ (pastries : Finset (Finset Nat))
    (filling_set : Finset Nat),
    (filling_set.card = 10) →
    (pastries.card = 45) →
    (∀ p ∈ pastries, p.card = 2 ∧ p ⊆ filling_set) →
    ∃ n, n = 36 ∧
    ∀ remain_p ∈ (pastries \ pastries.sort (λ x y, x < y)).take (45 - n), 
      ∃ f ∈ filling_set, f ∈ remain_p :=
begin
  sorry

end vasya_pastry_trick_l90_90282


namespace breadth_of_rectangular_plot_l90_90906

theorem breadth_of_rectangular_plot (b l A : ℝ) (h1 : l = 3 * b) (h2 : A = 588) (h3 : A = l * b) : b = 14 :=
by
  -- We start our proof here
  sorry

end breadth_of_rectangular_plot_l90_90906


namespace double_variable_for_1600_percent_cost_l90_90056

theorem double_variable_for_1600_percent_cost (t b0 b1 : ℝ) (h : t ≠ 0) :
    (t * b1^4 = 16 * t * b0^4) → b1 = 2 * b0 :=
by
sorry

end double_variable_for_1600_percent_cost_l90_90056


namespace original_number_is_two_thirds_l90_90216

theorem original_number_is_two_thirds (x : ℚ) (h : 1 + (1 / x) = 5 / 2) : x = 2 / 3 :=
by
  sorry

end original_number_is_two_thirds_l90_90216


namespace arithmetic_sequence_sum_l90_90523

open Function

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13)
    (h3 : ∀ n : ℕ, a (n + 1) = a n + d) :
    a 4 + a 5 + a 6 = 42 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_sum_l90_90523


namespace sum_is_correct_l90_90953

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l90_90953


namespace smallest_n_for_trick_l90_90289

theorem smallest_n_for_trick (fillings : Finset Fin 10)
  (pastries : Finset (Fin 45)) 
  (has_pairs : ∀ p ∈ pastries, ∃ f1 f2 ∈ fillings, f1 ≠ f2 ∧ p = pair f1 f2) : 
  ∃ n (tray : Finset (Fin 45)), 
    (tray.card = n ∧ n = 36 ∧ 
    ∀ remaining_p ∈ pastries \ tray, ∃ f ∈ fillings, f ∈ remaining_p) :=
by
  sorry

end smallest_n_for_trick_l90_90289


namespace project_completion_l90_90923

theorem project_completion (a b c d e : ℕ) 
  (h₁ : 1 / (a : ℝ) + 1 / b + 1 / c + 1 / d = 1 / 6)
  (h₂ : 1 / (b : ℝ) + 1 / c + 1 / d + 1 / e = 1 / 8)
  (h₃ : 1 / (a : ℝ) + 1 / e = 1 / 12) : 
  e = 48 :=
sorry

end project_completion_l90_90923


namespace hundred_days_from_friday_is_sunday_l90_90428

/-- Given that today is Friday, determine that 100 days from now is Sunday. -/
theorem hundred_days_from_friday_is_sunday (today : ℕ) (days_in_week : ℕ := 7) 
(friday : ℕ := 0) (sunday : ℕ := 2) : (((today + 100) % days_in_week) = sunday) :=
sorry

end hundred_days_from_friday_is_sunday_l90_90428


namespace fifteenth_term_of_geometric_sequence_l90_90257

theorem fifteenth_term_of_geometric_sequence :
  let a := 12
  let r := (1:ℚ) / 3
  let n := 15
  (a * r^(n-1)) = (4 / 1594323:ℚ)
:=
  by
    sorry

end fifteenth_term_of_geometric_sequence_l90_90257


namespace janice_initial_sentences_l90_90385

theorem janice_initial_sentences:
  ∀ (r t1 t2 t3 t4: ℕ), 
  r = 6 → 
  t1 = 20 → 
  t2 = 15 → 
  t3 = 40 → 
  t4 = 18 → 
  (t1 * r + t2 * r + t4 * r - t3 = 536 - 258) → 
  536 - (t1 * r + t2 * r + t4 * r - t3) = 258 := by
  intros
  sorry

end janice_initial_sentences_l90_90385


namespace sum_of_first_five_primes_with_units_digit_3_l90_90956

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90956


namespace find_k_l90_90063

theorem find_k (k : ℝ) : (∀ x y : ℝ, y = k * x + 3 → (x, y) = (2, 5)) → k = 1 :=
by
  sorry

end find_k_l90_90063


namespace probability_min_max_l90_90907

variable {Ω : Type*} {ι : Type*}
variables (σ : Ω → ι → ℝ) [Fintype ι] [DecidableEq ι]

noncomputable def xi_min (ξ : ι → ℝ) : ℝ := fintype.min ι ξ  
noncomputable def xi_max (ξ : ι → ℝ) : ℝ := fintype.max ι ξ  

theorem probability_min_max (ξ : ι → MeasureTheory.Measure Ω) (x : ℝ)
  (h_indep : ∀ i j, i ≠ j → MeasureTheory.Measure.Indep (ξ i) (ξ j)) :
  (MeasureTheory.Prob {ω : Ω | xi_min (λ i, σ ω i) ≥ x} = ∏ i, MeasureTheory.Prob {ω : Ω | σ ω i ≥ x}) ∧
  (MeasureTheory.Prob {ω : Ω | xi_max (λ i, σ ω i) < x} = ∏ i, MeasureTheory.Prob {ω : Ω | σ ω i < x}) :=
  sorry

end probability_min_max_l90_90907


namespace deepak_profit_share_l90_90927

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share_l90_90927


namespace divisor_is_20_l90_90441

theorem divisor_is_20 (D : ℕ) 
  (h1 : 242 % D = 11) 
  (h2 : 698 % D = 18) 
  (h3 : 940 % D = 9) :
  D = 20 :=
sorry

end divisor_is_20_l90_90441


namespace fraction_q_over_p_l90_90995

noncomputable def proof_problem (p q : ℝ) : Prop :=
  ∃ k : ℝ, p = 9^k ∧ q = 12^k ∧ p + q = 16^k

theorem fraction_q_over_p (p q : ℝ) (h : proof_problem p q) : q / p = (1 + Real.sqrt 5) / 2 :=
sorry

end fraction_q_over_p_l90_90995


namespace speed_of_other_train_l90_90889

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end speed_of_other_train_l90_90889


namespace smallest_a_no_inverse_mod_72_90_l90_90263

theorem smallest_a_no_inverse_mod_72_90 :
  ∃ (a : ℕ), a > 0 ∧ ∀ b : ℕ, (b > 0 → gcd b 72 > 1 ∧ gcd b 90 > 1 → b ≥ a) ∧ gcd a 72 > 1 ∧ gcd a 90 > 1 ∧ a = 6 :=
by sorry

end smallest_a_no_inverse_mod_72_90_l90_90263


namespace inequality_solution_l90_90894

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 2) :
  ∀ y : ℝ, y > 0 → 4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y :=
by
  intro y hy
  sorry

end inequality_solution_l90_90894


namespace parabola_at_point_has_value_zero_l90_90179

theorem parabola_at_point_has_value_zero (a m : ℝ) :
  (x ^ 2 + (a + 1) * x + a) = 0 -> m = 0 :=
by
  -- We know the parabola passes through the point (-1, m)
  sorry

end parabola_at_point_has_value_zero_l90_90179


namespace num_triangles_with_perimeter_9_l90_90510

-- Definitions for side lengths and their conditions
def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 9

-- The main theorem
theorem num_triangles_with_perimeter_9 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a ≤ b ∧ b ≤ c → is_triangle a b c ↔ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end num_triangles_with_perimeter_9_l90_90510


namespace tangent_curve_l90_90667

theorem tangent_curve (a : ℝ) : 
  (∃ x : ℝ, 3 * x - 2 = x^3 - 2 * a ∧ 3 * x^2 = 3) →
  a = 0 ∨ a = 2 := 
sorry

end tangent_curve_l90_90667


namespace find_circle_center_using_ruler_l90_90043

open EuclideanGeometry

theorem find_circle_center_using_ruler (A B C D : Point) (k : Circle) (P Q : LineSegment) (H : A ≠ B ∧ B ≠ C ∧ P ≠ Q) 
  (h_parallelogram: Parallelogram A B C D) 
  (h_chords : Chord k P ∧ Chord k Q ∧ Parallel P Q ∧ P.length ≠ Q.length) :
  ∃ O : Point, CircleCenteredAt k O ∧ CenterOfCircleUsingRuler A B C D k O :=
sorry

end find_circle_center_using_ruler_l90_90043


namespace tip_percentage_calculation_l90_90136

theorem tip_percentage_calculation :
  let a := 8
  let r := 20
  let w := 3
  let n_w := 2
  let d := 6
  let t := 38
  let discount := 0.5
  let full_cost_without_tip := a + r + (w * n_w) + d
  let discounted_meal_cost := a + (r - (r * discount)) + (w * n_w) + d
  let tip_amount := t - discounted_meal_cost
  let tip_percentage := (tip_amount / full_cost_without_tip) * 100
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_calculation_l90_90136


namespace number_of_triangles_with_perimeter_nine_l90_90509

theorem number_of_triangles_with_perimeter_nine : 
  ∃ (a b c : ℕ), a + b + c = 9 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry  -- Proof steps are omitted.

end number_of_triangles_with_perimeter_nine_l90_90509


namespace german_team_goals_l90_90114

/-- Proof that the German team could have scored 11, 12, 14, 16, or 17 goals given the conditions
    stated by the three journalists, with exactly two of their statements being true. -/
theorem german_team_goals (x : ℕ) :
  (10 < x ∧ x < 17 → 11 < x ∧ x < 18 → x % 2 = 1 → false) →
  (10 < x ∧ x < 17 ∧ ¬(11 < x ∧ x < 18) ∧ (x % 2 = 1) → false) →
  (¬(10 < x ∧ x < 17) ∧ 11 < x ∧ x < 18 ∧ (x % 2 = 1) → false) →
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end german_team_goals_l90_90114


namespace find_a_b_sum_l90_90413

-- Definitions for the conditions
def equation1 (a : ℝ) : Prop := 3 = (1 / 3) * 6 + a
def equation2 (b : ℝ) : Prop := 6 = (1 / 3) * 3 + b

theorem find_a_b_sum : 
  ∃ (a b : ℝ), equation1 a ∧ equation2 b ∧ (a + b = 6) :=
sorry

end find_a_b_sum_l90_90413


namespace initial_professors_l90_90021

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l90_90021


namespace faith_change_l90_90336

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l90_90336


namespace maximize_profit_l90_90233

def revenue (x : ℝ) : ℝ := 17 * x^2
def cost (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := revenue x - cost x

theorem maximize_profit : ∃ x > 0, profit x = 18 * x^2 - 2 * x^3 ∧ (∀ y > 0, y ≠ x → profit y < profit x) :=
by
  sorry

end maximize_profit_l90_90233


namespace find_b_l90_90672

theorem find_b (a b : ℝ) (h1 : (1 : ℝ)^3 + a*(1)^2 + b*1 + a^2 = 10)
    (h2 : 3*(1 : ℝ)^2 + 2*a*(1) + b = 0) : b = -11 :=
sorry

end find_b_l90_90672


namespace volume_of_fifth_section_l90_90525

theorem volume_of_fifth_section (a : ℕ → ℚ) (d : ℚ) :
  (a 1 + a 2 + a 3 + a 4) = 3 ∧ (a 9 + a 8 + a 7) = 4 ∧
  (∀ n, a n = a 1 + (n - 1) * d) →
  a 5 = 67 / 66 :=
by
  sorry

end volume_of_fifth_section_l90_90525


namespace odd_equivalence_classes_iff_n_eq_2_l90_90271

-- Defining the binary n-tuples and cyclic permutations
def B_n (n : ℕ) : Finset (Vector Bool n) :=
  Finset.univ

def cyclic_permutation {n : ℕ} (v : Vector Bool n) : Finset (Vector Bool n) :=
  Finset.image (λ k => v.rotate k) (Finset.range n)

-- The theorem statement proving the given problem
theorem odd_equivalence_classes_iff_n_eq_2 :
  ∀ (n : ℕ), n ≥ 2 →
  ((B_n n).card / n % 2 = 1) ↔ n = 2 :=
by
  sorry

end odd_equivalence_classes_iff_n_eq_2_l90_90271


namespace least_number_to_subtract_l90_90265

theorem least_number_to_subtract (n : ℕ) (h : n = 427398) : 
  ∃ k, (n - k) % 10 = 0 ∧ k = 8 :=
by
  sorry

end least_number_to_subtract_l90_90265


namespace sum_of_percentages_l90_90106

theorem sum_of_percentages : (20 / 100 : ℝ) * 40 + (25 / 100 : ℝ) * 60 = 23 := 
by 
  -- Sorry skips the proof
  sorry

end sum_of_percentages_l90_90106


namespace probability_not_paired_shoes_l90_90653

noncomputable def probability_not_pair (total_shoes : ℕ) (pairs : ℕ) (shoes_drawn : ℕ) : ℚ :=
  let total_ways := Nat.choose total_shoes shoes_drawn
  let pair_ways := pairs * Nat.choose 2 2
  let not_pair_ways := total_ways - pair_ways
  not_pair_ways / total_ways

theorem probability_not_paired_shoes (total_shoes pairs shoes_drawn : ℕ) (h1 : total_shoes = 6) 
(h2 : pairs = 3) (h3 : shoes_drawn = 2) :
  probability_not_pair total_shoes pairs shoes_drawn = 4 / 5 :=
by 
  rw [h1, h2, h3]
  simp [probability_not_pair, Nat.choose]
  sorry

end probability_not_paired_shoes_l90_90653


namespace miranda_pillows_l90_90212

-- Define the conditions in the problem
def pounds_per_pillow := 2
def feathers_per_pound := 300
def total_feathers := 3600

-- Define the goal in terms of these conditions
def num_pillows : Nat :=
  (total_feathers / feathers_per_pound) / pounds_per_pillow

-- Prove that the number of pillows Miranda can stuff is 6
theorem miranda_pillows : num_pillows = 6 :=
by
  sorry

end miranda_pillows_l90_90212


namespace index_card_area_l90_90928

theorem index_card_area 
  (a b : ℕ)
  (ha : a = 5)
  (hb : b = 7)
  (harea : (a - 2) * b = 21) :
  (a * (b - 2) = 25) :=
by
  sorry

end index_card_area_l90_90928


namespace sum_of_intercepts_l90_90456

theorem sum_of_intercepts (x y : ℝ) 
  (h_eq : y - 3 = -3 * (x - 5)) 
  (hx_intercept : y = 0 ∧ x = 6) 
  (hy_intercept : x = 0 ∧ y = 18) : 
  6 + 18 = 24 :=
by
  sorry

end sum_of_intercepts_l90_90456


namespace quadratic_roots_l90_90237

theorem quadratic_roots (x : ℝ) : (x^2 + 4*x + 3 = 0) ↔ (x = -3 ∨ x = -1) := 
sorry

end quadratic_roots_l90_90237


namespace positive_sum_minus_terms_gt_zero_l90_90703

theorem positive_sum_minus_terms_gt_zero 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 1) : 
  a^2 + a * b + b^2 - a - b > 0 := 
by
  sorry

end positive_sum_minus_terms_gt_zero_l90_90703


namespace validCardSelections_l90_90884

def numberOfValidSelections : ℕ :=
  let totalCards := 12
  let redCards := 4
  let otherColors := 8 -- 4 yellow + 4 blue
  let totalSelections := Nat.choose totalCards 3
  let nonRedSelections := Nat.choose otherColors 3
  let oneRedSelections := Nat.choose redCards 1 * Nat.choose otherColors 2
  let sameColorSelections := 3 * Nat.choose 4 3 -- 3 colors, 4 cards each, selecting 3
  (nonRedSelections + oneRedSelections)

theorem validCardSelections : numberOfValidSelections = 160 := by
  sorry

end validCardSelections_l90_90884


namespace CA_eq_A_intersection_CB_eq_l90_90656

-- Definitions as per conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | x > 1 }

-- Proof problems as per questions and answers
theorem CA_eq : (U \ A) = { x : ℝ | x ≥ 2 } :=
by
  sorry

theorem A_intersection_CB_eq : (A ∩ (U \ B)) = { x : ℝ | x ≤ 1 } :=
by
  sorry

end CA_eq_A_intersection_CB_eq_l90_90656


namespace mean_of_three_is_90_l90_90226

-- Given conditions as Lean definitions
def mean_twelve (s : ℕ) : Prop := s = 12 * 40
def added_sum (x y z : ℕ) (s : ℕ) : Prop := s + x + y + z = 15 * 50
def z_value (x z : ℕ) : Prop := z = x + 10

-- Theorem statement to prove the mean of x, y, and z is 90
theorem mean_of_three_is_90 (x y z s : ℕ) : 
  (mean_twelve s) → (z_value x z) → (added_sum x y z s) → 
  (x + y + z) / 3 = 90 := 
by 
  intros h1 h2 h3 
  sorry

end mean_of_three_is_90_l90_90226


namespace triangle_count_l90_90507

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangles : List (ℕ × ℕ × ℕ) :=
  List.filter (λ (t : ℕ × ℕ × ℕ), let (a, b, c) := t in is_triangle a b c)
    [(a, b, c) | a ← List.range 10, b ← List.range 10, c ← List.range 10, a + b + c = 9]

theorem triangle_count : valid_triangles.length = 12 := by
  sorry

end triangle_count_l90_90507


namespace part1_part2_l90_90790

-- Define the conditions
def P_condition (a x : ℝ) : Prop := 1 - a / x < 0
def Q_condition (x : ℝ) : Prop := abs (x + 2) < 3

-- First part: Given a = 3, prove the solution set P
theorem part1 (x : ℝ) : P_condition 3 x ↔ 0 < x ∧ x < 3 := by 
  sorry

-- Second part: Prove the range of values for the positive number a
theorem part2 (a : ℝ) (ha : 0 < a) : 
  (∀ x, (P_condition a x → Q_condition x)) → 0 < a ∧ a ≤ 1 := by 
  sorry

end part1_part2_l90_90790


namespace sum_of_first_five_primes_with_units_digit_3_l90_90958

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90958


namespace consecutive_numbers_count_l90_90736

theorem consecutive_numbers_count (n x : ℕ) (h_avg : (2 * n * 20 = n * (2 * x + n - 1))) (h_largest : x + n - 1 = 23) : n = 7 :=
by
  sorry

end consecutive_numbers_count_l90_90736


namespace multiplier_for_average_grade_l90_90708

/-- Conditions -/
def num_of_grades_2 : ℕ := 3
def num_of_grades_3 : ℕ := 4
def num_of_grades_4 : ℕ := 1
def num_of_grades_5 : ℕ := 1
def cash_reward : ℕ := 15

-- Definitions for sums and averages based on the conditions
def sum_of_grades : ℕ :=
  num_of_grades_2 * 2 + num_of_grades_3 * 3 + num_of_grades_4 * 4 + num_of_grades_5 * 5

def total_grades : ℕ :=
  num_of_grades_2 + num_of_grades_3 + num_of_grades_4 + num_of_grades_5

def average_grade : ℕ :=
  sum_of_grades / total_grades

/-- Proof statement -/
theorem multiplier_for_average_grade : cash_reward / average_grade = 5 := by
  sorry

end multiplier_for_average_grade_l90_90708


namespace faith_change_l90_90334

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l90_90334


namespace total_blocks_correct_l90_90008

-- Definitions given by the conditions in the problem
def red_blocks : ℕ := 18
def yellow_blocks : ℕ := red_blocks + 7
def blue_blocks : ℕ := red_blocks + 14

-- Theorem stating the goal to prove
theorem total_blocks_correct : red_blocks + yellow_blocks + blue_blocks = 75 := by
  -- Skipping the proof for now
  sorry

end total_blocks_correct_l90_90008


namespace least_value_a_l90_90683

theorem least_value_a (a : ℤ) :
  (∃ a : ℤ, a ≥ 0 ∧ (a ^ 6) % 1920 = 0) → a = 8 ∧ (a ^ 6) % 1920 = 0 :=
by
  sorry

end least_value_a_l90_90683


namespace student_marks_l90_90070

theorem student_marks (M P C X : ℕ) 
  (h1 : M + P = 60)
  (h2 : C = P + X)
  (h3 : M + C = 80) : X = 20 :=
by sorry

end student_marks_l90_90070


namespace solve_for_x_l90_90051

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.1 * (30 + x) = 15.5 → x = 83 := by 
  sorry

end solve_for_x_l90_90051


namespace emma_garden_area_l90_90795

-- Define the given conditions
def EmmaGarden (total_posts : ℕ) (posts_on_shorter_side : ℕ) (posts_on_longer_side : ℕ) (distance_between_posts : ℕ) : Prop :=
  total_posts = 24 ∧
  distance_between_posts = 6 ∧
  (posts_on_longer_side + 1) = 3 * (posts_on_shorter_side + 1) ∧
  2 * (posts_on_shorter_side + 1 + posts_on_longer_side + 1) = 24

-- The theorem to prove
theorem emma_garden_area : ∃ (length width : ℕ), EmmaGarden 24 2 8 6 ∧ (length = 6 * (2) ∧ width = 6 * (8 - 1)) ∧ (length * width = 576) :=
by
  -- proof goes here
  sorry

end emma_garden_area_l90_90795


namespace num_terms_arithmetic_sequence_is_15_l90_90182

theorem num_terms_arithmetic_sequence_is_15 :
  ∃ n : ℕ, (∀ (a : ℤ), a = -58 + (n - 1) * 7 → a = 44) ∧ n = 15 :=
by {
  sorry
}

end num_terms_arithmetic_sequence_is_15_l90_90182


namespace hi_mom_box_office_revenue_scientific_notation_l90_90624

def box_office_revenue_scientific_notation (billion : ℤ) (revenue : ℤ) : Prop :=
  revenue = 5.396 * 10^9

theorem hi_mom_box_office_revenue_scientific_notation :
  box_office_revenue_scientific_notation 53.96 53960000000 :=
by
  sorry

end hi_mom_box_office_revenue_scientific_notation_l90_90624


namespace area_ratio_of_squares_l90_90876

theorem area_ratio_of_squares (s L : ℝ) 
  (H : 4 * L = 4 * 4 * s) : (L^2) = 16 * (s^2) :=
by
  -- assuming the utilization of the given condition
  sorry

end area_ratio_of_squares_l90_90876


namespace jane_wins_l90_90530

/-- Define the total number of possible outcomes and the number of losing outcomes -/
def total_outcomes := 64
def losing_outcomes := 12

/-- Define the probability that Jane wins -/
def jane_wins_probability := (total_outcomes - losing_outcomes) / total_outcomes

/-- Problem: Jane wins with a probability of 13/16 given the conditions -/
theorem jane_wins :
  jane_wins_probability = 13 / 16 :=
sorry

end jane_wins_l90_90530


namespace quadratic_factor_transformation_l90_90132

theorem quadratic_factor_transformation (x : ℝ) :
  x^2 - 6 * x + 5 = 0 → (x - 3)^2 = 14 := 
by
  sorry

end quadratic_factor_transformation_l90_90132


namespace solve_printer_problem_l90_90900

noncomputable def printer_problem : Prop :=
  let rate_A := 10
  let rate_B := rate_A + 8
  let rate_C := rate_B - 4
  let combined_rate := rate_A + rate_B + rate_C
  let total_minutes := 20
  let total_pages := combined_rate * total_minutes
  total_pages = 840

theorem solve_printer_problem : printer_problem :=
by
  sorry

end solve_printer_problem_l90_90900


namespace andrew_age_l90_90782

/-- 
Andrew and his five cousins are ages 4, 6, 8, 10, 12, and 14. 
One afternoon two of his cousins whose ages sum to 18 went to the movies. 
Two cousins younger than 12 but not including the 8-year-old went to play baseball. 
Andrew and the 6-year-old stayed home. How old is Andrew?
-/
theorem andrew_age (ages : Finset ℕ) (andrew_age: ℕ)
  (h_ages : ages = {4, 6, 8, 10, 12, 14})
  (movies : Finset ℕ) (baseball : Finset ℕ)
  (h_movies1 : movies.sum id = 18)
  (h_baseball1 : ∀ x ∈ baseball, x < 12 ∧ x ≠ 8)
  (home : Finset ℕ) (h_home : home = {6, andrew_age}) :
  andrew_age = 12 :=
sorry

end andrew_age_l90_90782


namespace camila_weeks_to_goal_l90_90324

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l90_90324


namespace relationship_among_abc_l90_90500

noncomputable def a : ℝ := 36^(1/5)
noncomputable def b : ℝ := 3^(4/3)
noncomputable def c : ℝ := 9^(2/5)

theorem relationship_among_abc (a_def : a = 36^(1/5)) 
                              (b_def : b = 3^(4/3)) 
                              (c_def : c = 9^(2/5)) : a < c ∧ c < b :=
by
  rw [a_def, b_def, c_def]
  sorry

end relationship_among_abc_l90_90500


namespace probability_even_sum_is_half_l90_90454

-- Definitions for probability calculations
def prob_even_A : ℚ := 2 / 5
def prob_odd_A : ℚ := 3 / 5
def prob_even_B : ℚ := 1 / 2
def prob_odd_B : ℚ := 1 / 2

-- Sum is even if both are even or both are odd
def prob_even_sum := prob_even_A * prob_even_B + prob_odd_A * prob_odd_B

-- Theorem stating the final probability
theorem probability_even_sum_is_half : prob_even_sum = 1 / 2 := by
  sorry

end probability_even_sum_is_half_l90_90454


namespace juhye_initial_money_l90_90013

theorem juhye_initial_money
  (M : ℝ)
  (h1 : M - (1 / 4) * M - (2 / 3) * ((3 / 4) * M) = 2500) :
  M = 10000 := by
  sorry

end juhye_initial_money_l90_90013


namespace snow_at_least_once_l90_90568

theorem snow_at_least_once (p_snow : ℝ) (h : p_snow = 3/4) : 
  let p_no_snow := 1 - p_snow in
  let p_no_snow_4_days := p_no_snow ^ 4 in
  let p_snow_at_least_once := 1 - p_no_snow_4_days in
  p_snow_at_least_once = 255/256 :=
by
  sorry

end snow_at_least_once_l90_90568


namespace square_of_equal_side_of_inscribed_triangle_l90_90135

theorem square_of_equal_side_of_inscribed_triangle :
  ∀ (x y : ℝ),
  (x^2 + 9 * y^2 = 9) →
  ((x = 0) → (y = 1)) →
  ((x ≠ 0) → y = (x + 1)) →
  square_of_side = (324 / 25) :=
by
  intros x y hEllipse hVertex hSlope
  sorry

end square_of_equal_side_of_inscribed_triangle_l90_90135


namespace min_AP_squared_sum_value_l90_90757

-- Definitions based on given problem conditions
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 4
def D : ℝ := 7
def E : ℝ := 15

def distance_squared (x y : ℝ) : ℝ := (x - y)^2

noncomputable def min_AP_squared_sum (r : ℝ) : ℝ :=
  r^2 + distance_squared r B + distance_squared r C + distance_squared r D + distance_squared r E

theorem min_AP_squared_sum_value : ∃ (r : ℝ), (min_AP_squared_sum r) = 137.2 :=
by
  existsi 5.6
  sorry

end min_AP_squared_sum_value_l90_90757


namespace water_channel_area_l90_90411

-- Define the given conditions
def top_width := 14
def bottom_width := 8
def depth := 70

-- The area formula for a trapezium given the top width, bottom width, and height
def trapezium_area (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- The main theorem stating the area of the trapezium
theorem water_channel_area : 
  trapezium_area top_width bottom_width depth = 770 := by
  -- Proof can be completed here
  sorry

end water_channel_area_l90_90411


namespace prism_pyramid_sum_l90_90217

theorem prism_pyramid_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices = 34 :=
by
  sorry

end prism_pyramid_sum_l90_90217


namespace minibuses_not_enough_l90_90448

def num_students : ℕ := 300
def minibus_capacity : ℕ := 23
def num_minibuses : ℕ := 13

theorem minibuses_not_enough :
  num_minibuses * minibus_capacity < num_students :=
by
  sorry

end minibuses_not_enough_l90_90448


namespace no_fixed_points_range_l90_90802

def no_fixed_points (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 ≠ x

theorem no_fixed_points_range (a : ℝ) : no_fixed_points a ↔ -1 < a ∧ a < 3 := by
  sorry

end no_fixed_points_range_l90_90802


namespace find_k_value_l90_90060

theorem find_k_value (k : ℝ) : (∀ (x y : ℝ), (x = 2 ∧ y = 5) → y = k * x + 3) → k = 1 := 
by 
  intro h
  have h1 := h 2 5 ⟨rfl, rfl⟩
  linarith

end find_k_value_l90_90060


namespace total_blocks_correct_l90_90009

-- Definitions given by the conditions in the problem
def red_blocks : ℕ := 18
def yellow_blocks : ℕ := red_blocks + 7
def blue_blocks : ℕ := red_blocks + 14

-- Theorem stating the goal to prove
theorem total_blocks_correct : red_blocks + yellow_blocks + blue_blocks = 75 := by
  -- Skipping the proof for now
  sorry

end total_blocks_correct_l90_90009


namespace pentagon_area_l90_90141

theorem pentagon_area (a b c d e : ℝ)
  (ht_base ht_height : ℝ)
  (trap_base1 trap_base2 trap_height : ℝ)
  (side_a : a = 17)
  (side_b : b = 22)
  (side_c : c = 30)
  (side_d : d = 26)
  (side_e : e = 22)
  (rt_height : ht_height = 17)
  (rt_base : ht_base = 22)
  (trap_base1_eq : trap_base1 = 26)
  (trap_base2_eq : trap_base2 = 30)
  (trap_height_eq : trap_height = 22)
  : 1/2 * ht_base * ht_height + 1/2 * (trap_base1 + trap_base2) * trap_height = 803 :=
by sorry

end pentagon_area_l90_90141


namespace sufficient_not_necessary_condition_l90_90654

theorem sufficient_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x > 0 ∧ y > 0) → (x > 0 ∧ y > 0 ↔ (y/x + x/y ≥ 2)) :=
by sorry

end sufficient_not_necessary_condition_l90_90654


namespace german_team_goals_possible_goal_values_l90_90110

def german_team_goal_conditions (x : ℕ) :=
  (10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ∧ (x % 2 = 1)

def exactly_two_journalists_correct (x : ℕ) :=
  (10 < x ∧ x < 17) ∨ (11 < x ∧ x < 18) ∨ (x % 2 = 0)

theorem german_team_goals : ∀ (x : ℕ), german_team_goal_conditions x → exactly_two_journalists_correct x :=
begin
  intros x h_goal_conditions,
  -- This part is meant for proving the statement that exactly two of the journalistic statements are true.
  sorry
end

def possible_goals : set ℕ := { x | x ∈ {11, 12, 14, 16, 17} }

theorem possible_goal_values : ∀ (x : ℕ), x ∈ possible_goals :=
begin
  -- Proof that given the conditions, the possible goals must lie within the set {11, 12, 14, 16, 17}.
  sorry
end

end german_team_goals_possible_goal_values_l90_90110


namespace professors_initial_count_l90_90014

noncomputable def initialNumberOfProfessors (failureGradesLastYear : ℕ) (failureGradesNextYear : ℕ) (increaseProfessors : ℕ) : ℕ :=
if (failureGradesLastYear, failureGradesNextYear, increaseProfessors) = (6480, 11200, 3) then 5 else sorry

theorem professors_initial_count :
  initialNumberOfProfessors 6480 11200 3 = 5 := by {
  sorry
}

end professors_initial_count_l90_90014


namespace parabola_at_point_has_value_zero_l90_90180

theorem parabola_at_point_has_value_zero (a m : ℝ) :
  (x ^ 2 + (a + 1) * x + a) = 0 -> m = 0 :=
by
  -- We know the parabola passes through the point (-1, m)
  sorry

end parabola_at_point_has_value_zero_l90_90180


namespace container_capacity_l90_90910

theorem container_capacity (C : ℝ) (h1 : C > 0) (h2 : 0.40 * C + 14 = 0.75 * C) : C = 40 := 
by 
  -- Would contain the proof here
  sorry

end container_capacity_l90_90910


namespace violet_ticket_cost_l90_90892

theorem violet_ticket_cost :
  (2 * 35 + 5 * 20 = 170) ∧
  (((35 - 17.50) + 35 + 5 * 20) = 152.50) ∧
  ((152.50 - 150) = 2.50) :=
by
  sorry

end violet_ticket_cost_l90_90892


namespace probability_one_marble_each_color_l90_90606

theorem probability_one_marble_each_color :
  let total_marbles := 9
  let total_ways := Nat.choose total_marbles 3
  let favorable_ways := 3 * 3 * 3
  let probability := favorable_ways / total_ways
  probability = 9 / 28 :=
by
  sorry

end probability_one_marble_each_color_l90_90606


namespace jessica_balloons_l90_90834

-- Defining the number of blue balloons Joan, Sally, and the total number.
def balloons_joan : ℕ := 9
def balloons_sally : ℕ := 5
def balloons_total : ℕ := 16

-- The statement to prove that Jessica has 2 blue balloons
theorem jessica_balloons : balloons_total - (balloons_joan + balloons_sally) = 2 :=
by
  -- Using the given information and arithmetic, we can show the main statement
  sorry

end jessica_balloons_l90_90834


namespace garden_perimeter_equals_104_l90_90588

theorem garden_perimeter_equals_104 :
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width
  playground_area = 192 ∧ garden_perimeter = 104 :=
by {
  -- Declarations
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width

  -- Assertions
  have area_playground : playground_area = 192 := by sorry
  have perimeter_garden : garden_perimeter = 104 := by sorry

  -- Conclusion
  exact ⟨area_playground, perimeter_garden⟩
}

end garden_perimeter_equals_104_l90_90588


namespace xyz_expr_min_max_l90_90846

open Real

theorem xyz_expr_min_max (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 1) :
  ∃ m M : ℝ, m = 0 ∧ M = 1/4 ∧
    (∀ x y z : ℝ, x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
      xy + yz + zx - 3 * xyz ≥ m ∧ xy + yz + zx - 3 * xyz ≤ M) :=
sorry

end xyz_expr_min_max_l90_90846


namespace probability_king_then_queen_l90_90103

-- Definitions based on the conditions:
def total_cards : ℕ := 52
def ranks_per_suit : ℕ := 13
def suits : ℕ := 4
def kings : ℕ := 4
def queens : ℕ := 4

-- The problem statement rephrased as a theorem:
theorem probability_king_then_queen :
  (kings / total_cards : ℚ) * (queens / (total_cards - 1)) = 4 / 663 := 
by {
  sorry
}

end probability_king_then_queen_l90_90103


namespace find_a_l90_90079

theorem find_a (a : ℝ) (h : (1 / Real.log 2 / Real.log a) + (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) = 2) : a = Real.sqrt 30 := 
by 
  sorry

end find_a_l90_90079


namespace german_team_goals_l90_90122

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end german_team_goals_l90_90122


namespace product_eq_5832_l90_90347

theorem product_eq_5832 (P Q R S : ℕ) 
(h1 : P + Q + R + S = 48)
(h2 : P + 3 = Q - 3)
(h3 : Q - 3 = R * 3)
(h4 : R * 3 = S / 3) :
P * Q * R * S = 5832 := sorry

end product_eq_5832_l90_90347


namespace snow_probability_at_least_once_l90_90574

theorem snow_probability_at_least_once (p : ℚ) (h : p = 3 / 4) : 
  let q := 1 - p in let prob_not_snow_4_days := q^4 in (1 - prob_not_snow_4_days) = 255 / 256 := 
by
  sorry

end snow_probability_at_least_once_l90_90574


namespace sum_of_first_five_primes_units_digit_3_l90_90968

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l90_90968


namespace joel_strawberries_area_l90_90694

-- Define the conditions
def garden_area : ℕ := 64
def fruit_fraction : ℚ := 1 / 2
def strawberry_fraction : ℚ := 1 / 4

-- Define the desired conclusion
def strawberries_area : ℕ := 8

-- State the theorem
theorem joel_strawberries_area 
  (H1 : garden_area = 64) 
  (H2 : fruit_fraction = 1 / 2) 
  (H3 : strawberry_fraction = 1 / 4)
  : garden_area * fruit_fraction * strawberry_fraction = strawberries_area := 
sorry

end joel_strawberries_area_l90_90694


namespace surface_area_increase_l90_90266

noncomputable def percent_increase_surface_area (s p : ℝ) : ℝ :=
  let new_edge_length := s * (1 + p / 100)
  let new_surface_area := 6 * (new_edge_length)^2
  let original_surface_area := 6 * s^2
  let percent_increase := (new_surface_area / original_surface_area - 1) * 100
  percent_increase

theorem surface_area_increase (s p : ℝ) :
  percent_increase_surface_area s p = 2 * p + p^2 / 100 :=
by
  sorry

end surface_area_increase_l90_90266


namespace distinct_left_views_l90_90309

/-- Consider 10 small cubes each having dimension 1 cm × 1 cm × 1 cm.
    Each pair of adjacent cubes shares at least one edge (1 cm) or one face (1 cm × 1 cm).
    The cubes must not be suspended in the air and each cube's edges should be either
    perpendicular or parallel to the horizontal lines. Prove that the number of distinct
    left views of any arrangement of these 10 cubes is 16. -/
theorem distinct_left_views (cube_count : ℕ) (dimensions : ℝ) 
  (shared_edge : (ℝ × ℝ) → Prop) (no_suspension : Prop) (alignment : Prop) :
  cube_count = 10 →
  dimensions = 1 →
  (∀ x y, shared_edge (x, y) ↔ x = y ∨ x - y = 1) →
  no_suspension →
  alignment →
  distinct_left_views_count = 16 :=
by
  sorry

end distinct_left_views_l90_90309


namespace initial_professors_l90_90018

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l90_90018


namespace sum_of_first_five_primes_with_units_digit_three_l90_90972

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l90_90972


namespace sin_double_angle_shifted_l90_90493

theorem sin_double_angle_shifted (θ : ℝ) (h : Real.cos (θ + Real.pi) = - 1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = - 7 / 9 :=
by
  sorry

end sin_double_angle_shifted_l90_90493


namespace negation_of_universal_proposition_l90_90564

noncomputable def f (n : Nat) : Set ℕ := sorry

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, f n ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n, m ≤ n) ↔
  ∃ n_0 : ℕ, f n_0 ⊆ (Set.univ : Set ℕ) ∧ ∀ m ∈ f n_0, m ≤ n_0 :=
sorry

end negation_of_universal_proposition_l90_90564


namespace roger_ant_l90_90549

def expected_steps : ℚ := 11/3

theorem roger_ant (a b : ℕ) (h1 : expected_steps = a / b) (h2 : Nat.gcd a b = 1) : 100 * a + b = 1103 :=
sorry

end roger_ant_l90_90549


namespace books_on_shelf_l90_90272

theorem books_on_shelf (original_books : ℕ) (books_added : ℕ) (total_books : ℕ) (h1 : original_books = 38) 
(h2 : books_added = 10) : total_books = 48 :=
by 
  sorry

end books_on_shelf_l90_90272


namespace parallel_vectors_l90_90676

variables (x : ℝ)

theorem parallel_vectors (h : (1 + x) / 2 = (1 - 3 * x) / -1) : x = 3 / 5 :=
by {
  sorry
}

end parallel_vectors_l90_90676


namespace sum_of_first_five_prime_units_digit_3_l90_90961

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l90_90961


namespace probability_diagonals_intersect_nonagon_l90_90251

theorem probability_diagonals_intersect_nonagon :
  let n := 9 in
  let total_pairs_points := nat.choose n 2 in
  let num_sides := n in
  let num_diagonals := total_pairs_points - num_sides in
  let total_pairs_diagonals := nat.choose num_diagonals 2 in
  let intersecting_pairs := nat.choose n 4 in
  (intersecting_pairs : ℚ) / total_pairs_diagonals = 14 / 39 :=
by
  sorry

end probability_diagonals_intersect_nonagon_l90_90251


namespace c_geq_one_l90_90388

theorem c_geq_one {a b : ℕ} {c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : (a + 1) / (b + c) = b / a) : 1 ≤ c :=
  sorry

end c_geq_one_l90_90388


namespace f_7_eq_minus_1_l90_90841

-- Define the odd function f with the given properties
def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) :=
  ∀ x, f (x + 2) = -f x

def f_restricted (f : ℝ → ℝ) :=
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 -> f x = x

-- The main statement: Under the given conditions, f(7) = -1
theorem f_7_eq_minus_1 (f : ℝ → ℝ)
  (H1 : is_odd_function f)
  (H2 : period_2 f)
  (H3 : f_restricted f) :
  f 7 = -1 :=
by
  sorry

end f_7_eq_minus_1_l90_90841


namespace pairs_of_integers_solution_l90_90940

-- Define the main theorem
theorem pairs_of_integers_solution :
  ∃ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ∧ 
               ((x = 63 ∧ y = 58) ∨
               (x = -63 ∧ y = -58) ∨
               (x = 459 ∧ y = 58) ∨
               (x = -459 ∧ y = -58)) :=
by
  sorry

end pairs_of_integers_solution_l90_90940


namespace cost_per_liter_of_fuel_l90_90942

-- Definitions and conditions
def fuel_capacity : ℕ := 150
def initial_fuel : ℕ := 38
def change_received : ℕ := 14
def initial_money : ℕ := 350

-- Proof problem
theorem cost_per_liter_of_fuel :
  (initial_money - change_received) / (fuel_capacity - initial_fuel) = 3 :=
by
  sorry

end cost_per_liter_of_fuel_l90_90942


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90965

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l90_90965


namespace unique_root_of_linear_equation_l90_90175

theorem unique_root_of_linear_equation (a b : ℝ) (h : a ≠ 0) : ∃! x : ℝ, a * x = b :=
by
  sorry

end unique_root_of_linear_equation_l90_90175


namespace lateral_surface_area_of_cylinder_l90_90864

theorem lateral_surface_area_of_cylinder (V : ℝ) (hV : V = 27 * Real.pi) : 
  ∃ (S : ℝ), S = 18 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l90_90864


namespace emily_quiz_score_l90_90794

theorem emily_quiz_score :
  ∃ x : ℕ, 94 + 88 + 92 + 85 + 97 + x = 6 * 90 :=
by
  sorry

end emily_quiz_score_l90_90794


namespace common_tangent_range_l90_90369

theorem common_tangent_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ 2*a*x₁ = Real.exp x₂ 
   ∧ Real.exp x₂ = (Real.exp x₂ - a*x₁^2)/(x₂ - x₁)))
  ↔ a ∈ set.Ici (Real.exp 2 / 4) :=
by
  sorry

end common_tangent_range_l90_90369


namespace german_team_goals_l90_90131

theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : x % 2 = 1) 
  (h_correct : (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)):
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := 
by {
  -- Proof to be filled in
  sorry
}

end german_team_goals_l90_90131


namespace squats_day_after_tomorrow_l90_90151

theorem squats_day_after_tomorrow (initial_squats : ℕ) (daily_increase : ℕ) (today : ℕ) (tomorrow : ℕ) (day_after_tomorrow : ℕ)
  (h1 : initial_squats = 30)
  (h2 : daily_increase = 5)
  (h3 : today = initial_squats + daily_increase)
  (h4 : tomorrow = today + daily_increase)
  (h5 : day_after_tomorrow = tomorrow + daily_increase) : 
  day_after_tomorrow = 45 := 
sorry

end squats_day_after_tomorrow_l90_90151


namespace not_proportional_l90_90146

theorem not_proportional (x y : ℕ) :
  (∀ k : ℝ, y ≠ 3 * x - 7 ∧ y ≠ (13 - 4 * x) / 3) → 
  ((y = 3 * x - 7 ∨ y = (13 - 4 * x) / 3) → ¬(∃ k : ℝ, (y = k * x) ∨ (y = k / x))) := sorry

end not_proportional_l90_90146


namespace smallest_denominator_is_168_l90_90723

theorem smallest_denominator_is_168 (a b : ℕ) (h1: Nat.gcd a 600 = 1) (h2: Nat.gcd b 700 = 1) :
  ∃ k, Nat.gcd (7 * a + 6 * b) 4200 = k ∧ k = 25 ∧ (4200 / k) = 168 :=
sorry

end smallest_denominator_is_168_l90_90723


namespace wang_hua_withdrawal_correct_l90_90377

noncomputable def wang_hua_withdrawal : ℤ :=
  let d : ℤ := 14
  let c : ℤ := 32
  -- The amount Wang Hua was supposed to withdraw in yuan
  (d * 100 + c)

theorem wang_hua_withdrawal_correct (d c : ℤ) :
  let initial_amount := (100 * d + c)
  let incorrect_amount := (100 * c + d)
  let amount_spent := 350
  let remaining_amount := incorrect_amount - amount_spent
  let expected_remaining := 2 * initial_amount
  remaining_amount = expected_remaining ∧ 
  d = 14 ∧ 
  c = 32 :=
by
  sorry

end wang_hua_withdrawal_correct_l90_90377


namespace rows_of_pies_l90_90909

theorem rows_of_pies (baked_pecan_pies : ℕ) (baked_apple_pies : ℕ) (pies_per_row : ℕ) : 
  baked_pecan_pies = 16 ∧ baked_apple_pies = 14 ∧ pies_per_row = 5 → 
  (baked_pecan_pies + baked_apple_pies) / pies_per_row = 6 :=
by
  sorry

end rows_of_pies_l90_90909


namespace circuit_analysis_l90_90735

/-
There are 3 conducting branches connected between points A and B.
First branch: a 2 Volt EMF and a 2 Ohm resistor connected in series.
Second branch: a 2 Volt EMF and a 1 Ohm resistor.
Third branch: a conductor with a resistance of 1 Ohm.
Prove the currents and voltage drop are as follows:
- Current in first branch: i1 = 0.4 A
- Current in second branch: i2 = 0.8 A
- Current in third branch: i3 = 1.2 A
- Voltage between A and B: E_AB = 1.2 Volts
-/
theorem circuit_analysis :
  ∃ (i1 i2 i3 : ℝ) (E_AB : ℝ),
    (i1 = 0.4) ∧
    (i2 = 0.8) ∧
    (i3 = 1.2) ∧
    (E_AB = 1.2) ∧
    (2 = 2 * i1 + i3) ∧
    (2 = i2 + i3) ∧
    (i3 = i1 + i2) ∧
    (E_AB = i3 * 1) := sorry

end circuit_analysis_l90_90735


namespace fraction_of_green_marbles_half_l90_90275

-- Definitions based on given conditions
def initial_fraction (x : ℕ) : ℚ := 1 / 3

-- Number of blue, red, and green marbles initially
def blue_marbles (x : ℕ) : ℚ := initial_fraction x * x
def red_marbles (x : ℕ) : ℚ := initial_fraction x * x
def green_marbles (x : ℕ) : ℚ := initial_fraction x * x

-- Number of green marbles after doubling
def doubled_green_marbles (x : ℕ) : ℚ := 2 * green_marbles x

-- New total number of marbles
def new_total_marbles (x : ℕ) : ℚ := blue_marbles x + red_marbles x + doubled_green_marbles x

-- New fraction of green marbles after doubling
def new_fraction_of_green_marbles (x : ℕ) : ℚ := doubled_green_marbles x / new_total_marbles x

theorem fraction_of_green_marbles_half (x : ℕ) (hx : x > 0) :
  new_fraction_of_green_marbles x = 1 / 2 :=
by
  sorry

end fraction_of_green_marbles_half_l90_90275


namespace sum_of_interior_angles_l90_90305

theorem sum_of_interior_angles {n : ℕ} (h1 : ∀ i, i < n → (interior_angle i : ℝ) = 144) : 
  (sum_of_polygon_interior_angles n = 1440) :=
sorry

end sum_of_interior_angles_l90_90305


namespace original_number_of_professors_l90_90028

theorem original_number_of_professors (p : ℕ) 
  (h1 : 6480 % p = 0) 
  (h2 : 11200 % (p + 3) = 0) 
  (h3 : 6480 / p < 11200 / (p + 3))
  (h4 : 5 ≤ p) : 
  p = 5 :=
by {
  -- The body of the proof goes here.
  sorry
}

end original_number_of_professors_l90_90028


namespace mode_and_median_are_24_5_l90_90090

def shoe_sales : List (ℕ × ℕ) := [(23, 1), (23.5, 2), (24, 2), (24.5, 6), (25, 2)]

def mode (sales : List (ℕ × ℕ)) : ℕ :=
  sales.maximumBy (λ pair => pair.snd).fst

def median (sales : List (ℕ × ℕ)) : ℕ :=
  let ordered_sales := sales.sortBy (Order.backup1 . fst)
  ordered_sales.get! (8 - 1).fst

theorem mode_and_median_are_24_5 :
  mode shoe_sales = 24.5 ∧ median shoe_sales = 24.5 := by
  sorry

end mode_and_median_are_24_5_l90_90090


namespace number_of_students_before_new_year_l90_90753

variables (M N k ℓ : ℕ)
hypotheses (h1 : 100 * M = k * N)
             (h2 : 100 * (M + 1) = ℓ * (N + 3))
             (h3 : ℓ < 100)

theorem number_of_students_before_new_year (h1 : 100 * M = k * N)
                                             (h2 : 100 * (M + 1) = ℓ * (N + 3))
                                             (h3 : ℓ < 100) :
  N ≤ 197 :=
sorry

end number_of_students_before_new_year_l90_90753


namespace area_square_ratio_l90_90897

theorem area_square_ratio (r : ℝ) (h1 : r > 0)
  (s1 : ℝ) (hs1 : s1^2 = r^2)
  (s2 : ℝ) (hs2 : s2^2 = (4/5) * r^2) : 
  (s1^2 / s2^2) = (5 / 4) :=
by 
  sorry

end area_square_ratio_l90_90897


namespace same_type_monomials_l90_90518

theorem same_type_monomials (a b : ℤ) (h1 : 1 = a - 2) (h2 : b + 1 = 3) : (a - b) ^ 2023 = 1 := by
  sorry

end same_type_monomials_l90_90518


namespace rows_seating_8_people_l90_90480

theorem rows_seating_8_people (x : ℕ) (h₁ : x ≡ 4 [MOD 7]) (h₂ : x ≤ 6) :
  x = 4 := by
  sorry

end rows_seating_8_people_l90_90480


namespace german_team_goals_l90_90115

/-- Proof that the German team could have scored 11, 12, 14, 16, or 17 goals given the conditions
    stated by the three journalists, with exactly two of their statements being true. -/
theorem german_team_goals (x : ℕ) :
  (10 < x ∧ x < 17 → 11 < x ∧ x < 18 → x % 2 = 1 → false) →
  (10 < x ∧ x < 17 ∧ ¬(11 < x ∧ x < 18) ∧ (x % 2 = 1) → false) →
  (¬(10 < x ∧ x < 17) ∧ 11 < x ∧ x < 18 ∧ (x % 2 = 1) → false) →
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end german_team_goals_l90_90115


namespace range_of_x_l90_90495

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : 0 < x) (h₂ : A (2 * x * A x) = 5) : 1 < x ∧ x ≤ 5 / 4 := 
sorry

end range_of_x_l90_90495


namespace median_interval_70_to_74_l90_90936

theorem median_interval_70_to_74 :
  let counts := [
    (85, 89, 15),
    (80, 84, 20),
    (75, 79, 25),
    (70, 74, 30),
    (65, 69, 20),
    (60, 64, 10)  : (ℕ × ℕ × ℕ)
  ] in
  let total_students := 120 in
  let median_index := (total_students + 1) / 2 in
  ∃ (l u : ℕ), (l = 70 ∧ u = 74) ∧
               (∃ (lower_counts : ℕ), lower_counts + 30 ≥ median_index ∧ lower_counts < median_index) :=
by
  sorry

end median_interval_70_to_74_l90_90936


namespace sum_of_first_five_primes_with_units_digit_3_l90_90975

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90975


namespace find_f_of_neg3_l90_90408

noncomputable def f : ℚ → ℚ := sorry 

theorem find_f_of_neg3 (h : ∀ (x : ℚ) (hx : x ≠ 0), 5 * f (x⁻¹) + 3 * (f x) * x⁻¹ = 2 * x^2) :
  f (-3) = -891 / 22 :=
sorry

end find_f_of_neg3_l90_90408


namespace hundred_days_from_friday_is_sunday_l90_90431

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l90_90431


namespace f_g_of_3_l90_90680

def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := x^2 + 2 * x + 1

theorem f_g_of_3 : f (g 3) = 61 :=
by
  sorry

end f_g_of_3_l90_90680


namespace system_solution_exists_l90_90406

theorem system_solution_exists (x y: ℝ) :
    (y^2 = (x + 8) * (x^2 + 2) ∧ y^2 - (8 + 4 * x) * y + (16 + 16 * x - 5 * x^2) = 0) → 
    ((x = 0 ∧ (y = 4 ∨ y = -4)) ∨ (x = -2 ∧ (y = 6 ∨ y = -6)) ∨ (x = 19 ∧ (y = 99 ∨ y = -99))) :=
    sorry

end system_solution_exists_l90_90406


namespace twice_x_minus_3_l90_90945

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l90_90945


namespace machine_A_production_l90_90542

-- Definitions based on the conditions
def machine_production (A B: ℝ) (TA TB: ℝ) : Prop :=
  B = 1.10 * A ∧
  TA = TB + 10 ∧
  A * TA = 660 ∧
  B * TB = 660

-- The main statement to be proved: Machine A produces 6 sprockets per hour.
theorem machine_A_production (A B: ℝ) (TA TB: ℝ) 
  (h : machine_production A B TA TB) : 
  A = 6 := 
by sorry

end machine_A_production_l90_90542


namespace utility_bills_total_l90_90853

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l90_90853


namespace king_pages_and_ducats_l90_90862

theorem king_pages_and_ducats (n : ℕ) (d : ℕ) 
  (h_cond : ∃ (page : ℕ), page > 0 ∧ page ≤ n ∧ (n * 2 - page * 2) + 2 = 32) 
  : (n = 16 ∧ d = 992) ∨ (n = 8 ∧ d = 240) :=
by sorry

end king_pages_and_ducats_l90_90862


namespace height_of_Joaos_salary_in_kilometers_l90_90139

def real_to_cruzados (reais: ℕ) : ℕ := reais * 2750000000

def stacks (cruzados: ℕ) : ℕ := cruzados / 100

def height_in_cm (stacks: ℕ) : ℕ := stacks * 15

noncomputable def height_in_km (height_cm: ℕ) : ℕ := height_cm / 100000

theorem height_of_Joaos_salary_in_kilometers :
  height_in_km (height_in_cm (stacks (real_to_cruzados 640))) = 264000 :=
by
  sorry

end height_of_Joaos_salary_in_kilometers_l90_90139


namespace probability_of_snow_l90_90571

theorem probability_of_snow :
  let p_snow_each_day : ℚ := 3 / 4
  let p_no_snow_each_day : ℚ := 1 - p_snow_each_day
  let p_no_snow_four_days : ℚ := p_no_snow_each_day ^ 4
  let p_snow_at_least_once_four_days : ℚ := 1 - p_no_snow_four_days
  p_snow_at_least_once_four_days = 255 / 256 :=
by 
  unfold p_snow_each_day
  unfold p_no_snow_each_day
  unfold p_no_snow_four_days
  unfold p_snow_at_least_once_four_days
  unfold p_snow_at_least_once_four_days
  -- Sorry is used to skip the proof
  sorry

end probability_of_snow_l90_90571


namespace correct_operation_l90_90899

theorem correct_operation (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
by
  sorry

end correct_operation_l90_90899


namespace simplify_expression_l90_90403

theorem simplify_expression (x : ℝ) : 
  (x^3 * x^2 * x + (x^3)^2 + (-2 * x^2)^3) = -6 * x^6 := 
by 
  sorry

end simplify_expression_l90_90403


namespace exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l90_90037

def small_numbers (n : ℕ) : Prop := n ≤ 150

theorem exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest :
  ∃ (N : ℕ), (∃ (a b : ℕ), small_numbers a ∧ small_numbers b ∧ (a + 1 = b) ∧ ¬(N % a = 0) ∧ ¬(N % b = 0))
  ∧ (∀ (m : ℕ), small_numbers m → ¬(m = a ∨ m = b) → N % m = 0) :=
sorry

end exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l90_90037


namespace find_ratio_l90_90208

variable (x y : ℝ)

-- Hypotheses: x and y are distinct real numbers and the given equation holds
variable (h₁ : x ≠ y)
variable (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3)

-- We aim to prove that x / y = 0.8
theorem find_ratio (h₁ : x ≠ y) (h₂ : x / y + (x + 15 * y) / (y + 15 * x) = 3) : x / y = 0.8 :=
sorry

end find_ratio_l90_90208


namespace min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l90_90035

theorem min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared 
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  (ae^2 : ℝ) + (bf^2 : ℝ) + (cg^2 : ℝ) + (dh^2 : ℝ) ≥ 32 := 
sorry

end min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l90_90035


namespace evaluate_f_g_3_l90_90365

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l90_90365


namespace newspapers_ratio_l90_90692

theorem newspapers_ratio :
  (∀ (j m : ℕ), j = 234 → m = 4 * j + 936 → (m / 4) / j = 2) :=
by
  sorry

end newspapers_ratio_l90_90692


namespace german_team_goals_possible_goal_values_l90_90111

def german_team_goal_conditions (x : ℕ) :=
  (10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ∧ (x % 2 = 1)

def exactly_two_journalists_correct (x : ℕ) :=
  (10 < x ∧ x < 17) ∨ (11 < x ∧ x < 18) ∨ (x % 2 = 0)

theorem german_team_goals : ∀ (x : ℕ), german_team_goal_conditions x → exactly_two_journalists_correct x :=
begin
  intros x h_goal_conditions,
  -- This part is meant for proving the statement that exactly two of the journalistic statements are true.
  sorry
end

def possible_goals : set ℕ := { x | x ∈ {11, 12, 14, 16, 17} }

theorem possible_goal_values : ∀ (x : ℕ), x ∈ possible_goals :=
begin
  -- Proof that given the conditions, the possible goals must lie within the set {11, 12, 14, 16, 17}.
  sorry
end

end german_team_goals_possible_goal_values_l90_90111


namespace train_length_490_l90_90619

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_sec

theorem train_length_490 :
  train_length 63 28 = 490 := by
  -- Proof goes here
  sorry

end train_length_490_l90_90619


namespace probability_diagonals_intersect_l90_90243

theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) :
  let total_diagonals := (nat.choose n 2) - n,
      total_pairs_of_diagonals := nat.choose total_diagonals 2,
      intersecting_pairs := nat.choose n 4 in
  total_pairs_of_diagonals > 0 ∧ (intersecting_pairs : ℚ) / total_pairs_of_diagonals = 14 / 39 :=
by
  let total_diagonals := (nat.choose n 2) - n
  let total_pairs_of_diagonals := nat.choose total_diagonals 2
  let intersecting_pairs := nat.choose n 4
  have total_diagonals_eq : total_diagonals = 27 := by rw [hn]; norm_num
  have total_pairs_of_diagonals_eq : total_pairs_of_diagonals = 351 := by rw [total_diagonals_eq]; norm_num
  have intersecting_pairs_eq : intersecting_pairs = 126 := by rw [hn]; norm_num
  refine ⟨_, _⟩; sorry

end probability_diagonals_intersect_l90_90243


namespace german_team_goals_l90_90116

/-- Given the conditions from three journalists and knowing that only two of them can be correct,
we need to prove the possible values for the number of goals (x) scored by the German team during the championship. --/
theorem german_team_goals (x : ℕ) 
  (h1 : 10 < x ∧ x < 17) 
  (h2 : 11 < x ∧ x < 18) 
  (h3 : odd x) 
  (h4 : (h1 ∨ h2 ∨ h3) ∧ ¬(h1 ∧ h2 ∧ h3)) :
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 := sorry

end german_team_goals_l90_90116


namespace correct_exponentiation_calculation_l90_90267

theorem correct_exponentiation_calculation (a : ℝ) : a^2 * a^6 = a^8 :=
by sorry

end correct_exponentiation_calculation_l90_90267


namespace number_of_quarters_l90_90901
-- Definitions of the coin values
def penny_value := 1
def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

-- Number of each type of coin used in the proof
variable (pennies nickels dimes quarters half_dollars : ℕ)

-- Conditions from step (a)
axiom one_penny : pennies > 0
axiom one_nickel : nickels > 0
axiom one_dime : dimes > 0
axiom one_quarter : quarters > 0
axiom one_half_dollar : half_dollars > 0
axiom total_coins : pennies + nickels + dimes + quarters + half_dollars = 11
axiom total_value : pennies * penny_value + nickels * nickel_value + dimes * dime_value + quarters * quarter_value + half_dollars * half_dollar_value = 163

-- The conclusion we want to prove
theorem number_of_quarters : quarters = 1 := 
sorry

end number_of_quarters_l90_90901


namespace percent_of_number_l90_90642

theorem percent_of_number (x : ℝ) (h : 18 = 0.75 * x) : x = 24 := by
  sorry

end percent_of_number_l90_90642


namespace garden_strawberry_area_l90_90697

variable (total_garden_area : Real) (fruit_fraction : Real) (strawberry_fraction : Real)
variable (h1 : total_garden_area = 64)
variable (h2 : fruit_fraction = 1 / 2)
variable (h3 : strawberry_fraction = 1 / 4)

theorem garden_strawberry_area : 
  let fruit_area := total_garden_area * fruit_fraction
  let strawberry_area := fruit_area * strawberry_fraction
  strawberry_area = 8 :=
by
  sorry

end garden_strawberry_area_l90_90697


namespace eval_sequence_l90_90920

noncomputable def b : ℕ → ℤ
| 1 => 1
| 2 => 4
| 3 => 9
| n => if h : n > 3 then b (n - 1) * (b (n - 1) - 1) + 1 else 0

theorem eval_sequence :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 - (b 1 ^ 2 + b 2 ^ 2 + b 3 ^ 2 + b 4 ^ 2 + b 5 ^ 2 + b 6 ^ 2)
  = -3166598256 :=
by
  /- The proof steps are omitted. -/
  sorry

end eval_sequence_l90_90920


namespace pieces_cut_from_rod_l90_90512

theorem pieces_cut_from_rod (rod_length_m : ℝ) (piece_length_cm : ℝ) (rod_length_cm_eq : rod_length_m * 100 = 4250) (piece_length_eq : piece_length_cm = 85) :
  (4250 / 85) = 50 :=
by sorry

end pieces_cut_from_rod_l90_90512


namespace find_n_l90_90368

theorem find_n (n : ℤ) (h : (n + 1999) / 2 = -1) : n = -2001 := 
sorry

end find_n_l90_90368


namespace marie_age_l90_90209

theorem marie_age (L M O : ℕ) (h1 : L = 4 * M) (h2 : O = M + 8) (h3 : L = O) : M = 8 / 3 := by
  sorry

end marie_age_l90_90209


namespace pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l90_90760

noncomputable def pit_a_no_replant_prob : ℝ := 0.875
noncomputable def one_pit_no_replant_prob : ℝ := 0.713
noncomputable def at_least_one_pit_replant_prob : ℝ := 0.330

theorem pit_A_no_replant (p : ℝ) (h1 : p = 0.5) : pit_a_no_replant_prob = 1 - (1 - p)^3 := by
  sorry

theorem exactly_one_pit_no_replant (p : ℝ) (h1 : p = 0.5) : one_pit_no_replant_prob = 1 - 3 * (1 - p)^3 * (p^3)^(2) := by
  sorry

theorem at_least_one_replant (p : ℝ) (h1 : p = 0.5) : at_least_one_pit_replant_prob = 1 - (1 - (1 - p)^3)^3 := by
  sorry

end pit_A_no_replant_exactly_one_pit_no_replant_at_least_one_replant_l90_90760


namespace translated_point_B_coords_l90_90830

-- Define the initial point A
def point_A : ℝ × ℝ := (-2, 2)

-- Define the translation operations
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 + d, p.2)

-- Define the translation of point A to point B
def point_B :=
  translate_right (translate_down point_A 4) 3

-- The proof statement
theorem translated_point_B_coords : point_B = (1, -2) :=
  by sorry

end translated_point_B_coords_l90_90830


namespace find_speed_way_home_l90_90821

theorem find_speed_way_home
  (speed_to_mother : ℝ)
  (average_speed : ℝ)
  (speed_to_mother_val : speed_to_mother = 130)
  (average_speed_val : average_speed = 109) :
  ∃ v : ℝ, v = 109 * 130 / 151 := by
  sorry

end find_speed_way_home_l90_90821


namespace solve_triangle_l90_90689

theorem solve_triangle (a b : ℝ) (A B : ℝ) : ((A + B < π ∧ A > 0 ∧ B > 0 ∧ a > 0) ∨ (a > 0 ∧ b > 0 ∧ (π > A) ∧ (A > 0))) → ∃ c C, c > 0 ∧ (π > C) ∧ C > 0 :=
sorry

end solve_triangle_l90_90689


namespace line_tangent_to_circle_l90_90504

theorem line_tangent_to_circle {m : ℝ} : 
  (∀ x y : ℝ, y = m * x) → (∀ x y : ℝ, x^2 + y^2 - 4 * x + 2 = 0) → 
  (m = 1 ∨ m = -1) := 
by 
  sorry

end line_tangent_to_circle_l90_90504


namespace triangle_base_and_area_l90_90105

theorem triangle_base_and_area
  (height : ℝ)
  (h_height : height = 12)
  (height_base_ratio : ℝ)
  (h_ratio : height_base_ratio = 2 / 3) :
  ∃ (base : ℝ) (area : ℝ),
  base = height / height_base_ratio ∧
  area = base * height / 2 ∧
  base = 18 ∧
  area = 108 :=
by
  sorry

end triangle_base_and_area_l90_90105


namespace real_and_equal_roots_of_quadratic_l90_90329

theorem real_and_equal_roots_of_quadratic (k: ℝ) :
  (-(k+2))^2 - 4 * 3 * 12 = 0 ↔ k = 10 ∨ k = -14 :=
by
  sorry

end real_and_equal_roots_of_quadratic_l90_90329


namespace proof_problem_l90_90241

theorem proof_problem (x : ℕ) (h : 320 / (x + 26) = 4) : x = 54 := 
by 
  sorry

end proof_problem_l90_90241


namespace parabola_through_point_l90_90177

-- Define the parabola equation property
def parabola (a x : ℝ) : ℝ := x^2 + (a+1) * x + a

-- Introduce the main problem statement
theorem parabola_through_point (a m : ℝ) (h : parabola a (-1) = m) : m = 0 :=
by
  sorry

end parabola_through_point_l90_90177


namespace perimeter_triangle_APR_l90_90254

-- Define given lengths
def AB := 24
def AC := AB
def AP := 8
def AR := AP

-- Define lengths calculated from conditions 
def PB := AB - AP
def RC := AC - AR

-- Define properties from the tangent intersection at Q
def PQ := PB
def QR := RC
def PR := PQ + QR

-- Calculate the perimeter
def perimeter_APR := AP + PR + AR

-- Proof of the problem statement
theorem perimeter_triangle_APR : perimeter_APR = 48 :=
by
  -- Calculations already given in the statement
  sorry

end perimeter_triangle_APR_l90_90254


namespace vector_properties_l90_90359

/-- The vectors a, b, and c used in the problem. --/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-4, 2)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  ((∃ k : ℝ, b = k • a) ∧ (b.1 * c.1 + b.2 * c.2 = 0) ∧ (a.1*a.1 + a.2*a.2 = c.1*c.1 + c.2*c.2)) :=
  by sorry

end vector_properties_l90_90359


namespace part1_proof_part2_proof_l90_90539

-- Given conditions
variables (a b x : ℝ)
def y (a b x : ℝ) := a*x^2 + (b-2)*x + 3

-- The initial conditions
noncomputable def conditions := 
  (∀ x, -1 < x ∧ x < 3 → y a b x > 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ y a b 1 = 2)

-- Part (1): Prove that the solution set of y >= 4 is {1}
theorem part1_proof :
  conditions a b →
  {x | y a b x ≥ 4} = {1} :=
  by
    sorry

-- Part (2): Prove that the minimum value of (1/a + 4/b) is 9
theorem part2_proof :
  conditions a b →
  ∃ x, x = 1/a + 4/b ∧ x = 9 :=
  by
    sorry

end part1_proof_part2_proof_l90_90539


namespace number_divisible_by_5_l90_90871

theorem number_divisible_by_5 (A B C : ℕ) :
  (∃ (k1 k2 k3 k4 k5 k6 : ℕ), 3*10^6 + 10^5 + 7*10^4 + A*10^3 + B*10^2 + 4*10 + C = k1 ∧ 5 * k1 = 0 ∧
                          5 * k2 + 10 = 5 * k2 ∧ 5 * k3 + 5 = 5 * k3 ∧ 
                          5 * k4 + 3 = 5 * k4 ∧ 5 * k5 + 1 = 5 * k5 ∧ 
                          5 * k6 + 7 = 5 * k6) → C = 5 :=
by
  sorry

end number_divisible_by_5_l90_90871


namespace a_and_b_together_complete_work_in_12_days_l90_90595

-- Define the rate of work for b
def R_b : ℚ := 1 / 60

-- Define the rate of work for a based on the given condition that a is four times as fast as b
def R_a : ℚ := 4 * R_b

-- Define the combined rate of work for a and b working together
def R_a_plus_b : ℚ := R_a + R_b

-- Define the target time
def target_time : ℚ := 12

-- Proof statement
theorem a_and_b_together_complete_work_in_12_days :
  (R_a_plus_b * target_time) = 1 :=
by
  -- Proof omitted
  sorry

end a_and_b_together_complete_work_in_12_days_l90_90595


namespace volume_of_pyramid_l90_90466

variables (a b c : ℝ)

def triangle_face1 (a b : ℝ) : Prop := 1/2 * a * b = 1.5
def triangle_face2 (b c : ℝ) : Prop := 1/2 * b * c = 2
def triangle_face3 (c a : ℝ) : Prop := 1/2 * c * a = 6

theorem volume_of_pyramid (h1 : triangle_face1 a b) (h2 : triangle_face2 b c) (h3 : triangle_face3 c a) :
  1/3 * a * b * c = 2 :=
sorry

end volume_of_pyramid_l90_90466


namespace trigonometric_expression_value_l90_90809

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 5 := by
  sorry

end trigonometric_expression_value_l90_90809


namespace min_q_difference_l90_90844

theorem min_q_difference (p q : ℕ) (hpq : 0 < p ∧ 0 < q) (ineq1 : (7:ℚ)/12 < p/q) (ineq2 : p/q < (5:ℚ)/8) (hmin : ∀ r s : ℕ, 0 < r ∧ 0 < s ∧ (7:ℚ)/12 < r/s ∧ r/s < (5:ℚ)/8 → q ≤ s) : q - p = 2 :=
sorry

end min_q_difference_l90_90844


namespace probability_diagonals_intersect_l90_90244

theorem probability_diagonals_intersect (n : ℕ) (hn : n = 9) :
  let diagonals := (n * (n - 3)) / 2,
      pairs_of_diagonals := diagonals.choose 2,
      combinations_of_four := n.choose 4 in
  pairs_of_diagonals > 0 ∧ 
  combinations_of_four / pairs_of_diagonals = 6 / 17 :=
by
  sorry

end probability_diagonals_intersect_l90_90244


namespace minimum_pies_for_trick_l90_90296

-- Definitions from conditions
def num_fillings : ℕ := 10
def num_pastries := (num_fillings * (num_fillings - 1)) / 2
def min_pies_for_trick (n : ℕ) : Prop :=
  ∀ remaining_pies : ℕ, remaining_pies = num_pastries - n → remaining_pies ≤ 9

theorem minimum_pies_for_trick : ∃ n : ℕ, min_pies_for_trick n ∧ n = 36 :=
by
  -- We need to show that there exists n such that,
  -- min_pies_for_trick holds and n = 36
  existsi (36 : ℕ)
  -- remainder of the proof (step solution) skipped
  sorry

end minimum_pies_for_trick_l90_90296


namespace incorrect_neg_p_l90_90593

theorem incorrect_neg_p (p : ∀ x : ℝ, x ≥ 1) : ¬ (∀ x : ℝ, x < 1) :=
sorry

end incorrect_neg_p_l90_90593


namespace xiaoming_xiaoqiang_common_visit_l90_90268

-- Define the initial visit dates and subsequent visit intervals
def xiaoming_initial_visit : ℕ := 3 -- The first Wednesday of January
def xiaoming_interval : ℕ := 4

def xiaoqiang_initial_visit : ℕ := 4 -- The first Thursday of January
def xiaoqiang_interval : ℕ := 3

-- Prove that the only common visit date is January 7
theorem xiaoming_xiaoqiang_common_visit : 
  ∃! d, (d < 32) ∧ ∃ n m, d = xiaoming_initial_visit + n * xiaoming_interval ∧ d = xiaoqiang_initial_visit + m * xiaoqiang_interval :=
  sorry

end xiaoming_xiaoqiang_common_visit_l90_90268


namespace new_area_eq_1_12_original_area_l90_90594

variable (L W : ℝ)
def increased_length (L : ℝ) : ℝ := 1.40 * L
def decreased_width (W : ℝ) : ℝ := 0.80 * W
def original_area (L W : ℝ) : ℝ := L * W
def new_area (L W : ℝ) : ℝ := (increased_length L) * (decreased_width W)

theorem new_area_eq_1_12_original_area (L W : ℝ) :
  new_area L W = 1.12 * (original_area L W) :=
by
  sorry

end new_area_eq_1_12_original_area_l90_90594


namespace sum_of_first_five_primes_with_units_digit_3_l90_90979

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90979


namespace four_digit_number_condition_l90_90417

theorem four_digit_number_condition (x n : ℕ) (h1 : n = 2000 + x) (h2 : 10 * x + 2 = 2 * n + 66) : n = 2508 :=
sorry

end four_digit_number_condition_l90_90417


namespace largest_x_eq_120_div_11_l90_90341

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end largest_x_eq_120_div_11_l90_90341


namespace squats_day_after_tomorrow_l90_90149

theorem squats_day_after_tomorrow (initial_day_squats : ℕ) (increase_per_day : ℕ)
  (h1 : initial_day_squats = 30) (h2 : increase_per_day = 5) :
  let second_day_squats := initial_day_squats + increase_per_day in
  let third_day_squats := second_day_squats + increase_per_day in
  let fourth_day_squats := third_day_squats + increase_per_day in
  fourth_day_squats = 45 :=
by
  -- Placeholder proof
  sorry

end squats_day_after_tomorrow_l90_90149


namespace wrench_force_inversely_proportional_l90_90230

theorem wrench_force_inversely_proportional (F L : ℝ) (F1 F2 L1 L2 : ℝ) 
    (h1 : F1 = 375) 
    (h2 : L1 = 9) 
    (h3 : L2 = 15) 
    (h4 : ∀ L : ℝ, F * L = F1 * L1) : F2 = 225 :=
by
  sorry

end wrench_force_inversely_proportional_l90_90230


namespace ratio_of_ages_l90_90457

theorem ratio_of_ages (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = S + 20) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_l90_90457


namespace algebraic_expression_value_l90_90498

theorem algebraic_expression_value
  (a : ℝ) 
  (h : a^2 + 2 * a - 1 = 0) : 
  -a^2 - 2 * a + 8 = 7 :=
by 
  sorry

end algebraic_expression_value_l90_90498


namespace solve_for_x_l90_90337

theorem solve_for_x : ∃ x : ℝ, x^4 + 10 * x^3 + 9 * x^2 - 50 * x - 56 = 0 ↔ x = -2 :=
by
  sorry

end solve_for_x_l90_90337


namespace day_100_days_from_friday_l90_90433

-- Define the days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Define a function to get the day of the week after a given number of days
def dayOfWeekAfter (start : Day) (n : ℕ) : Day :=
  match start with
  | Sunday    => match n % 7 with
                  | 0 => Sunday
                  | 1 => Monday
                  | 2 => Tuesday
                  | 3 => Wednesday
                  | 4 => Thursday
                  | 5 => Friday
                  | 6 => Saturday
                  | _ => start
  | Monday    => match n % 7 with
                  | 0 => Monday
                  | 1 => Tuesday
                  | 2 => Wednesday
                  | 3 => Thursday
                  | 4 => Friday
                  | 5 => Saturday
                  | 6 => Sunday
                  | _ => start
  | Tuesday   => match n % 7 with
                  | 0 => Tuesday
                  | 1 => Wednesday
                  | 2 => Thursday
                  | 3 => Friday
                  | 4 => Saturday
                  | 5 => Sunday
                  | 6 => Monday
                  | _ => start
  | Wednesday => match n % 7 with
                  | 0 => Wednesday
                  | 1 => Thursday
                  | 2 => Friday
                  | 3 => Saturday
                  | 4 => Sunday
                  | 5 => Monday
                  | 6 => Tuesday
                  | _ => start
  | Thursday  => match n % 7 with
                  | 0 => Thursday
                  | 1 => Friday
                  | 2 => Saturday
                  | 3 => Sunday
                  | 4 => Monday
                  | 5 => Tuesday
                  | 6 => Wednesday
                  | _ => start
  | Friday    => match n % 7 with
                  | 0 => Friday
                  | 1 => Saturday
                  | 2 => Sunday
                  | 3 => Monday
                  | 4 => Tuesday
                  | 5 => Wednesday
                  | 6 => Thursday
                  | _ => start
  | Saturday  => match n % 7 with
                  | 0 => Saturday
                  | 1 => Sunday
                  | 2 => Monday
                  | 3 => Tuesday
                  | 4 => Wednesday
                  | 5 => Thursday
                  | 6 => Friday
                  | _ => start

-- The proof problem as a Lean theorem
theorem day_100_days_from_friday : dayOfWeekAfter Friday 100 = Sunday := by
  -- Proof will go here
  sorry

end day_100_days_from_friday_l90_90433


namespace scientific_notation_of_53_96_billion_l90_90625

theorem scientific_notation_of_53_96_billion :
  (53.96 * 10^9) = (5.396 * 10^10) :=
sorry

end scientific_notation_of_53_96_billion_l90_90625


namespace circle_through_focus_l90_90172

open Real

-- Define the parabola as a set of points
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 - 3) ^ 2 = 8 * (P.1 - 2)

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (4, 3)

-- Define the circle with center P and radius the distance from P to the y-axis
def is_tangent_circle (P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  (P.1 ^ 2 + (P.2 - 3) ^ 2 = (C.1) ^ 2 + (C.2) ^ 2 ∧ C = (4, 3))

-- The main theorem
theorem circle_through_focus (P : ℝ × ℝ) 
  (hP_on_parabola : is_on_parabola P) 
  (hP_tangent_circle : is_tangent_circle P (4, 3)) :
  is_tangent_circle P (4, 3) :=
by sorry

end circle_through_focus_l90_90172


namespace partial_fractions_sum_zero_l90_90629

theorem partial_fractions_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, 
     x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 →
     1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4)) →
  A + B + C + D + E = 0 :=
by
  intros h
  sorry

end partial_fractions_sum_zero_l90_90629


namespace fraction_calculation_l90_90631

theorem fraction_calculation : (8 / 24) - (5 / 72) + (3 / 8) = 23 / 36 :=
by
  sorry

end fraction_calculation_l90_90631


namespace kyle_caught_fish_l90_90474

def total_fish := 36
def fish_carla := 8
def fish_total := total_fish - fish_carla

-- kelle and tasha same number of fish means they equally divide the total fish left after deducting carla's
def fish_each_kt := fish_total / 2

theorem kyle_caught_fish :
  fish_each_kt = 14 :=
by
  -- Placeholder for the proof
  sorry

end kyle_caught_fish_l90_90474


namespace fifteenth_battery_replacement_month_l90_90778

theorem fifteenth_battery_replacement_month :
  (98 % 12) + 1 = 4 :=
by
  sorry

end fifteenth_battery_replacement_month_l90_90778


namespace constructible_triangle_l90_90637

theorem constructible_triangle (k c delta : ℝ) (h1 : 2 * c < k) :
  ∃ (a b : ℝ), a + b + c = k ∧ a + b > c ∧ ∃ (α β : ℝ), α - β = delta :=
by
  sorry

end constructible_triangle_l90_90637


namespace ratio_expression_l90_90515

theorem ratio_expression 
  (m n r t : ℚ)
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := 
by 
  sorry

end ratio_expression_l90_90515


namespace fg_of_3_eq_79_l90_90367

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l90_90367


namespace range_of_a_l90_90370

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4 * a) ↔ a ∈ Set.union (Set.Iio 1) (Set.Ioi 3) :=
by
  sorry

end range_of_a_l90_90370


namespace min_theta_l90_90482

theorem min_theta (theta : ℝ) (k : ℤ) (h : theta + 2 * k * Real.pi = -11 / 4 * Real.pi) : 
  theta = -3 / 4 * Real.pi :=
  sorry

end min_theta_l90_90482


namespace compute_a_b_difference_square_l90_90701

noncomputable def count_multiples (m n : ℕ) : ℕ :=
  (n - 1) / m

theorem compute_a_b_difference_square :
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  (a - b) ^ 2 = 0 :=
by
  let a := count_multiples 12 60
  let b := count_multiples 12 60
  show (a - b) ^ 2 = 0
  sorry

end compute_a_b_difference_square_l90_90701


namespace intersection_of_sets_l90_90087

-- Definitions of sets A and B based on given conditions
def setA : Set ℤ := {x | x + 2 = 0}
def setB : Set ℤ := {x | x^2 - 4 = 0}

-- The theorem to prove the intersection of A and B
theorem intersection_of_sets : setA ∩ setB = {-2} := by
  sorry

end intersection_of_sets_l90_90087


namespace combined_salaries_ABC_E_l90_90068

-- Definitions for the conditions
def salary_D : ℝ := 7000
def avg_salary_ABCDE : ℝ := 8200

-- Defining the combined salary proof
theorem combined_salaries_ABC_E : (A B C E : ℝ) → 
  (A + B + C + D + E = 5 * avg_salary_ABCDE ∧ D = salary_D) → 
  (A + B + C + E = 34000) := 
sorry

end combined_salaries_ABC_E_l90_90068


namespace faith_change_l90_90335

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def two_twenty_bills : ℕ := 2 * 20
def loose_coins : ℕ := 3
def total_cost : ℕ := flour_cost + cake_stand_cost
def total_given : ℕ := two_twenty_bills + loose_coins
def change : ℕ := total_given - total_cost

theorem faith_change : change = 10 := by
  sorry

end faith_change_l90_90335


namespace initial_weight_l90_90885

noncomputable def initial_average_weight (A : ℝ) : Prop :=
  let total_weight_initial := 20 * A
  let total_weight_new := total_weight_initial + 210
  let new_average_weight := 181.42857142857142
  total_weight_new / 21 = new_average_weight

theorem initial_weight:
  ∃ A : ℝ, initial_average_weight A ∧ A = 180 :=
by
  sorry

end initial_weight_l90_90885


namespace bird_height_l90_90721

theorem bird_height (cat_height dog_height avg_height : ℕ) 
  (cat_height_eq : cat_height = 92)
  (dog_height_eq : dog_height = 94)
  (avg_height_eq : avg_height = 95) :
  let total_height := avg_height * 3 
  let bird_height := total_height - (cat_height + dog_height)
  bird_height = 99 := 
by
  sorry

end bird_height_l90_90721


namespace digits_difference_l90_90873

-- Definitions based on conditions
variables (X Y : ℕ)

-- Condition: The difference between the original number and the interchanged number is 27
def difference_condition : Prop :=
  (10 * X + Y) - (10 * Y + X) = 27

-- Problem to prove: The difference between the two digits is 3
theorem digits_difference (h : difference_condition X Y) : X - Y = 3 :=
by sorry

end digits_difference_l90_90873


namespace sum_of_first_five_primes_with_units_digit_3_l90_90957

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l90_90957


namespace previous_year_profit_percentage_l90_90197

theorem previous_year_profit_percentage (R : ℝ) (P : ℝ) :
  (0.16 * 0.70 * R = 1.1200000000000001 * (P / 100 * R)) → P = 10 :=
by {
  sorry
}

end previous_year_profit_percentage_l90_90197


namespace initial_professors_l90_90020

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l90_90020


namespace sum_of_first_five_prime_units_digit_3_l90_90962

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l90_90962


namespace intervals_of_monotonicity_l90_90670

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi / 3)

theorem intervals_of_monotonicity :
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) → (f x ≤ f (7 * Real.pi / 12 + k * Real.pi)))) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi) → (f x ≥ f (Real.pi / 12 + k * Real.pi)))) ∧
  (f (Real.pi / 2) = -Real.sqrt 3) ∧
  (f (Real.pi / 12) = 1 - Real.sqrt 3 / 2) := sorry

end intervals_of_monotonicity_l90_90670


namespace math_problem_l90_90538

theorem math_problem (n d : ℕ) (h1 : 0 < n) (h2 : d < 10)
  (h3 : 3 * n^2 + 2 * n + d = 263)
  (h4 : 3 * n^2 + 2 * n + 4 = 396 + 7 * d) :
  n + d = 11 :=
by {
  sorry
}

end math_problem_l90_90538


namespace total_faces_is_198_l90_90581

-- Definitions for the number of dice and geometrical shapes brought by each person:
def TomDice : ℕ := 4
def TimDice : ℕ := 5
def TaraDice : ℕ := 3
def TinaDice : ℕ := 2
def TonyCubes : ℕ := 1
def TonyTetrahedrons : ℕ := 3
def TonyIcosahedrons : ℕ := 2

-- Definitions for the number of faces for each type of dice or shape:
def SixSidedFaces : ℕ := 6
def EightSidedFaces : ℕ := 8
def TwelveSidedFaces : ℕ := 12
def TwentySidedFaces : ℕ := 20
def CubeFaces : ℕ := 6
def TetrahedronFaces : ℕ := 4
def IcosahedronFaces : ℕ := 20

-- We want to prove that the total number of faces is 198:
theorem total_faces_is_198 : 
  (TomDice * SixSidedFaces) + 
  (TimDice * EightSidedFaces) + 
  (TaraDice * TwelveSidedFaces) + 
  (TinaDice * TwentySidedFaces) + 
  (TonyCubes * CubeFaces) + 
  (TonyTetrahedrons * TetrahedronFaces) + 
  (TonyIcosahedrons * IcosahedronFaces) 
  = 198 := 
by {
  sorry
}

end total_faces_is_198_l90_90581
