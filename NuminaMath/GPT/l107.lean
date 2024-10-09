import Mathlib

namespace other_solution_of_quadratic_l107_10735

theorem other_solution_of_quadratic (x : ℚ) (h1 : x = 3 / 8) 
  (h2 : 72 * x^2 + 37 = -95 * x + 12) : ∃ y : ℚ, y ≠ 3 / 8 ∧ 72 * y^2 + 95 * y + 25 = 0 ∧ y = 5 / 8 :=
by
  sorry

end other_solution_of_quadratic_l107_10735


namespace sufficient_but_not_necessary_l107_10708

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → 2 / a < 1) ∧ (2 / a < 1 → a > 2 ∨ a < 0) :=
by sorry

end sufficient_but_not_necessary_l107_10708


namespace cost_of_fencing_per_meter_l107_10738

theorem cost_of_fencing_per_meter
  (breadth : ℝ)
  (length : ℝ)
  (cost : ℝ)
  (length_eq : length = breadth + 40)
  (total_cost : cost = 5300)
  (length_given : length = 70) :
  cost / (2 * length + 2 * breadth) = 26.5 :=
by
  sorry

end cost_of_fencing_per_meter_l107_10738


namespace arithmetic_sequence_count_l107_10794

noncomputable def count_arithmetic_triplets : ℕ := 17

theorem arithmetic_sequence_count :
  ∃ S : Finset (Finset ℕ), 
    (∀ s ∈ S, s.card = 3 ∧ (∃ d, ∀ x ∈ s, ∀ y ∈ s, ∀ z ∈ s, (x ≠ y ∧ y ≠ z ∧ x ≠ z) → ((x = y + d ∨ x = z + d ∨ y = z + d) ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9))) ∧ 
    S.card = count_arithmetic_triplets :=
by
  -- placeholder for proof
  sorry

end arithmetic_sequence_count_l107_10794


namespace john_works_30_hours_per_week_l107_10706

/-- Conditions --/
def hours_per_week_fiona : ℕ := 40
def hours_per_week_jeremy : ℕ := 25
def hourly_wage : ℕ := 20
def monthly_total_payment : ℕ := 7600
def weeks_in_month : ℕ := 4

/-- Derived Definitions --/
def monthly_hours_fiona_jeremy : ℕ :=
  (hours_per_week_fiona + hours_per_week_jeremy) * weeks_in_month

def monthly_payment_fiona_jeremy : ℕ :=
  hourly_wage * monthly_hours_fiona_jeremy

def monthly_payment_john : ℕ :=
  monthly_total_payment - monthly_payment_fiona_jeremy

def hours_per_month_john : ℕ :=
  monthly_payment_john / hourly_wage

def hours_per_week_john : ℕ :=
  hours_per_month_john / weeks_in_month

/-- Theorem stating that John works 30 hours per week --/
theorem john_works_30_hours_per_week :
  hours_per_week_john = 30 := by
  sorry

end john_works_30_hours_per_week_l107_10706


namespace least_multiple_of_25_gt_390_l107_10720

theorem least_multiple_of_25_gt_390 : ∃ n : ℕ, n * 25 > 390 ∧ (∀ m : ℕ, m * 25 > 390 → m * 25 ≥ n * 25) ∧ n * 25 = 400 :=
by
  sorry

end least_multiple_of_25_gt_390_l107_10720


namespace find_cost_price_l107_10750

-- Given conditions
variables (CP SP1 SP2 : ℝ)
def condition1 : Prop := SP1 = 0.90 * CP
def condition2 : Prop := SP2 = 1.10 * CP
def condition3 : Prop := SP2 - SP1 = 500

-- Prove that CP is 2500 
theorem find_cost_price 
  (CP SP1 SP2 : ℝ)
  (h1 : condition1 CP SP1)
  (h2 : condition2 CP SP2)
  (h3 : condition3 SP1 SP2) : 
  CP = 2500 :=
sorry -- proof not required

end find_cost_price_l107_10750


namespace silk_original_amount_l107_10730

theorem silk_original_amount (s r : ℕ) (l d x : ℚ)
  (h1 : s = 30)
  (h2 : r = 3)
  (h3 : d = 12)
  (h4 : 30 - 3 = 27)
  (h5 : x / 12 = 30 / 27):
  x = 40 / 3 :=
by
  sorry

end silk_original_amount_l107_10730


namespace solution_set_of_inequality_l107_10789

theorem solution_set_of_inequality (x : ℝ) : (x * (2 - x) ≤ 0) ↔ (x ≤ 0 ∨ x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l107_10789


namespace loan_period_l107_10722

theorem loan_period (principal : ℝ) (rate_A rate_C gain_B : ℝ) (n : ℕ) 
  (h1 : principal = 3150)
  (h2 : rate_A = 0.08)
  (h3 : rate_C = 0.125)
  (h4 : gain_B = 283.5) :
  (gain_B = (rate_C * principal - rate_A * principal) * n) → n = 2 := by
  sorry

end loan_period_l107_10722


namespace cost_to_fill_pool_l107_10773

noncomputable def pool_cost : ℝ :=
  let base_width := 6
  let top_width := 4
  let length := 20
  let depth := 10
  let conversion_factor := 25
  let price_per_liter := 3
  let tax_rate := 0.08
  let discount_rate := 0.05
  let volume := 0.5 * depth * (base_width + top_width) * length
  let liters := volume * conversion_factor
  let initial_cost := liters * price_per_liter
  let cost_with_tax := initial_cost * (1 + tax_rate)
  let final_cost := cost_with_tax * (1 - discount_rate)
  final_cost

theorem cost_to_fill_pool : pool_cost = 76950 := by
  sorry

end cost_to_fill_pool_l107_10773


namespace difference_of_squares_example_l107_10725

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end difference_of_squares_example_l107_10725


namespace total_payment_is_correct_l107_10723

def daily_rental_cost : ℝ := 30
def per_mile_cost : ℝ := 0.25
def one_time_service_charge : ℝ := 15
def rent_duration : ℝ := 4
def distance_driven : ℝ := 500

theorem total_payment_is_correct :
  (daily_rental_cost * rent_duration + per_mile_cost * distance_driven + one_time_service_charge) = 260 := 
by
  sorry

end total_payment_is_correct_l107_10723


namespace each_dolphin_training_hours_l107_10793

theorem each_dolphin_training_hours
  (num_dolphins : ℕ)
  (num_trainers : ℕ)
  (hours_per_trainer : ℕ)
  (total_hours : ℕ := num_trainers * hours_per_trainer)
  (hours_per_dolphin_daily : ℕ := total_hours / num_dolphins)
  (h1 : num_dolphins = 4)
  (h2 : num_trainers = 2)
  (h3 : hours_per_trainer = 6) :
  hours_per_dolphin_daily = 3 :=
  by sorry

end each_dolphin_training_hours_l107_10793


namespace smaller_number_l107_10701

theorem smaller_number {a b : ℕ} (h_ratio : b = 5 * a / 2) (h_lcm : Nat.lcm a b = 160) : a = 64 := 
by
  sorry

end smaller_number_l107_10701


namespace smallest_x_absolute_value_l107_10740

theorem smallest_x_absolute_value :
  ∃ x : ℝ, (|5 * x + 15| = 40) ∧ (∀ y : ℝ, |5 * y + 15| = 40 → x ≤ y) ∧ x = -11 :=
sorry

end smallest_x_absolute_value_l107_10740


namespace sin_theta_plus_45_l107_10765

-- Statement of the problem in Lean 4

theorem sin_theta_plus_45 (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) (sin_θ_eq : Real.sin θ = 3 / 5) :
  Real.sin (θ + π / 4) = 7 * Real.sqrt 2 / 10 :=
sorry

end sin_theta_plus_45_l107_10765


namespace ellipse_eccentricity_l107_10792

theorem ellipse_eccentricity (a : ℝ) (h : a > 0) 
  (ell_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ x^2 / a^2 + y^2 / 5 = 1)
  (ecc_eq : (eccentricity : ℝ) = 2 / 3) : 
  a = 3 := 
sorry

end ellipse_eccentricity_l107_10792


namespace volume_of_adjacent_cubes_l107_10709

theorem volume_of_adjacent_cubes 
(side_length count : ℝ) 
(h_side : side_length = 5) 
(h_count : count = 5) : 
  (count * side_length ^ 3) = 625 :=
by
  -- Proof steps (skipped)
  sorry

end volume_of_adjacent_cubes_l107_10709


namespace cyclist_speed_l107_10707

variable (circumference : ℝ) (v₂ : ℝ) (t : ℝ)

theorem cyclist_speed (h₀ : circumference = 180) (h₁ : v₂ = 8) (h₂ : t = 12)
  (h₃ : (7 * t + v₂ * t) = circumference) : 7 = 7 :=
by
  -- From given conditions, we derived that v₁ should be 7
  sorry

end cyclist_speed_l107_10707


namespace chess_tournament_points_distribution_l107_10711

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l107_10711


namespace remainder_calculation_l107_10733

theorem remainder_calculation :
  (7 * 10^23 + 3^25) % 11 = 5 :=
by
  sorry

end remainder_calculation_l107_10733


namespace minimum_dimes_needed_l107_10796

theorem minimum_dimes_needed (n : ℕ) 
  (sneaker_cost : ℝ := 58) 
  (ten_bills : ℝ := 50)
  (five_quarters : ℝ := 1.25) :
  ten_bills + five_quarters + (0.10 * n) ≥ sneaker_cost ↔ n ≥ 68 := 
by 
  sorry

end minimum_dimes_needed_l107_10796


namespace minimum_additional_marbles_l107_10756

theorem minimum_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 34) : 
  ∃ additional_marbles : ℕ, additional_marbles = 44 :=
by
  -- The formal proof would go here.
  sorry

end minimum_additional_marbles_l107_10756


namespace solve_variables_l107_10771

theorem solve_variables (x y z : ℝ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x)
  (h3 : (z / 3) * 5 + y = 20) :
  x = 5 ∧ y = 2.5 ∧ z = 10.5 :=
by { sorry }

end solve_variables_l107_10771


namespace tank_height_l107_10788

theorem tank_height
  (r_A r_B h_A h_B : ℝ)
  (h₁ : 8 = 2 * Real.pi * r_A)
  (h₂ : h_B = 8)
  (h₃ : 10 = 2 * Real.pi * r_B)
  (h₄ : π * r_A ^ 2 * h_A = 0.56 * (π * r_B ^ 2 * h_B)) :
  h_A = 7 :=
sorry

end tank_height_l107_10788


namespace perpendicular_lines_l107_10754

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax + 2 * y + 6 = 0) ∧ (∃ x y : ℝ, x + (a - 1) * y + a^2 - 1 = 0) ∧ (∀ m1 m2 : ℝ, m1 * m2 = -1) →
  a = 2/3 :=
by
  sorry

end perpendicular_lines_l107_10754


namespace find_minimum_l107_10778

theorem find_minimum (a b c : ℝ) : ∃ (m : ℝ), m = min a (min b c) := 
  sorry

end find_minimum_l107_10778


namespace captain_age_l107_10760

theorem captain_age
  (C W : ℕ)
  (avg_team_age : ℤ)
  (avg_remaining_players_age : ℤ)
  (total_team_age : ℤ)
  (total_remaining_players_age : ℤ)
  (remaining_players_count : ℕ)
  (total_team_count : ℕ)
  (total_team_age_eq : total_team_age = total_team_count * avg_team_age)
  (remaining_players_age_eq : total_remaining_players_age = remaining_players_count * avg_remaining_players_age)
  (total_team_eq : total_team_count = 11)
  (remaining_players_eq : remaining_players_count = 9)
  (avg_team_age_eq : avg_team_age = 23)
  (avg_remaining_players_age_eq : avg_remaining_players_age = avg_team_age - 1)
  (age_diff : W = C + 5)
  (players_age_sum : total_team_age = total_remaining_players_age + C + W) :
  C = 25 :=
by
  sorry

end captain_age_l107_10760


namespace cone_volume_l107_10782

theorem cone_volume (V_cylinder V_frustum V_cone : ℝ)
  (h₁ : V_cylinder = 9)
  (h₂ : V_frustum = 63) :
  V_cone = 64 :=
sorry

end cone_volume_l107_10782


namespace John_total_weekly_consumption_l107_10713

/-
  Prove that John's total weekly consumption of water, milk, and juice in quarts is 49.25 quarts, 
  given the specified conditions on his daily and periodic consumption.
-/

def John_consumption_problem (gallons_per_day : ℝ) (pints_every_other_day : ℝ) (ounces_every_third_day : ℝ) 
  (quarts_per_gallon : ℝ) (quarts_per_pint : ℝ) (quarts_per_ounce : ℝ) : ℝ :=
  let water_per_day := gallons_per_day * quarts_per_gallon
  let water_per_week := water_per_day * 7
  let milk_per_other_day := pints_every_other_day * quarts_per_pint
  let milk_per_week := milk_per_other_day * 4 -- assuming he drinks milk 4 times a week
  let juice_per_third_day := ounces_every_third_day * quarts_per_ounce
  let juice_per_week := juice_per_third_day * 2 -- assuming he drinks juice 2 times a week
  water_per_week + milk_per_week + juice_per_week

theorem John_total_weekly_consumption :
  John_consumption_problem 1.5 3 20 4 (1/2) (1/32) = 49.25 :=
by
  sorry

end John_total_weekly_consumption_l107_10713


namespace coloring_ways_l107_10774

-- Define a factorial function
def factorial : Nat → Nat
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Define a derangement function
def derangement : Nat → Nat
| 0       => 1
| 1       => 0
| (n + 1) => n * (derangement n + derangement (n - 1))

-- Prove the main theorem
theorem coloring_ways : 
  let six_factorial := factorial 6
  let derangement_6 := derangement 6
  let derangement_5 := derangement 5
  720 * (derangement_6 + derangement_5) = 222480 := by
    let six_factorial := 720
    let derangement_6 := derangement 6
    let derangement_5 := derangement 5
    show six_factorial * (derangement_6 + derangement_5) = 222480
    sorry

end coloring_ways_l107_10774


namespace min_le_max_condition_l107_10786

variable (a b c : ℝ)

theorem min_le_max_condition
  (h1 : a ≠ 0)
  (h2 : ∃ t : ℝ, 2*a*t^2 + b*t + c = 0 ∧ |t| ≤ 1) :
  min c (a + c + 1) ≤ max (|b - a + 1|) (|b + a - 1|) :=
sorry

end min_le_max_condition_l107_10786


namespace weight_loss_percentage_l107_10742

theorem weight_loss_percentage {W : ℝ} (hW : 0 < W) :
  (((W - ((1 - 0.13 + 0.02 * (1 - 0.13)) * W)) / W) * 100) = 11.26 :=
by
  sorry

end weight_loss_percentage_l107_10742


namespace employees_without_any_benefit_l107_10776

def employees_total : ℕ := 480
def employees_salary_increase : ℕ := 48
def employees_travel_increase : ℕ := 96
def employees_both_increases : ℕ := 24
def employees_vacation_days : ℕ := 72

theorem employees_without_any_benefit : (employees_total - ((employees_salary_increase + employees_travel_increase + employees_vacation_days) - employees_both_increases)) = 288 :=
by
  sorry

end employees_without_any_benefit_l107_10776


namespace stratified_sampling_number_of_products_drawn_l107_10772

theorem stratified_sampling_number_of_products_drawn (T S W X : ℕ) 
  (h1 : T = 1024) (h2 : S = 64) (h3 : W = 128) :
  X = S * (W / T) → X = 8 :=
by
  sorry

end stratified_sampling_number_of_products_drawn_l107_10772


namespace gas_mixture_pressure_l107_10743

theorem gas_mixture_pressure
  (m : ℝ) -- mass of each gas
  (p : ℝ) -- initial pressure
  (T : ℝ) -- initial temperature
  (V : ℝ) -- volume of the container
  (R : ℝ) -- ideal gas constant
  (mu_He : ℝ := 4) -- molar mass of helium
  (mu_N2 : ℝ := 28) -- molar mass of nitrogen
  (is_ideal : True) -- assumption that the gases are ideal
  (temp_doubled : True) -- assumption that absolute temperature is doubled
  (N2_dissociates : True) -- assumption that nitrogen dissociates into atoms
  : (9 / 4) * p = p' :=
by
  sorry

end gas_mixture_pressure_l107_10743


namespace total_price_eq_2500_l107_10705

theorem total_price_eq_2500 (C P : ℕ)
  (hC : C = 2000)
  (hE : C + 500 + P = 6 * P)
  : C + P = 2500 := 
by
  sorry

end total_price_eq_2500_l107_10705


namespace roots_n_not_divisible_by_5_for_any_n_l107_10748

theorem roots_n_not_divisible_by_5_for_any_n (x1 x2 : ℝ) (n : ℕ)
  (hx : x1^2 - 6 * x1 + 1 = 0)
  (hy : x2^2 - 6 * x2 + 1 = 0)
  : ¬(∃ (k : ℕ), (x1^k + x2^k) % 5 = 0) :=
sorry

end roots_n_not_divisible_by_5_for_any_n_l107_10748


namespace factorize_difference_of_squares_l107_10785

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) :=
by
  sorry

end factorize_difference_of_squares_l107_10785


namespace positive_root_gt_1008_l107_10710

noncomputable def P (x : ℝ) : ℝ := sorry
-- where P is a non-constant polynomial with integer coefficients bounded by 2015 in absolute value
-- Assume it has been properly defined according to the conditions in the problem statement

theorem positive_root_gt_1008 (x : ℝ) (hx : 0 < x) (hroot : P x = 0) : x > 1008 := 
sorry

end positive_root_gt_1008_l107_10710


namespace quotient_in_first_division_l107_10737

theorem quotient_in_first_division (N Q Q' : ℕ) (h₁ : N = 68 * Q) (h₂ : N % 67 = 1) : Q = 1 :=
by
  -- rest of the proof goes here
  sorry

end quotient_in_first_division_l107_10737


namespace algebraic_expression_equals_one_l107_10755

variable (m n : ℝ)

theorem algebraic_expression_equals_one
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_eq : m - n = 1 / 2) :
  (m^2 - n^2) / (2 * m^2 + 2 * m * n) / (m - (2 * m * n - n^2) / m) = 1 :=
by
  sorry

end algebraic_expression_equals_one_l107_10755


namespace value_of_expression_l107_10751

theorem value_of_expression (x y : ℝ) (h1 : x + y = 3) (h2 : x^2 + y^2 - x * y = 4) : 
  x^4 + y^4 + x^3 * y + x * y^3 = 36 :=
by
  sorry

end value_of_expression_l107_10751


namespace residue_neg_998_mod_28_l107_10749

theorem residue_neg_998_mod_28 : ∃ r : ℤ, r = -998 % 28 ∧ 0 ≤ r ∧ r < 28 ∧ r = 10 := 
by sorry

end residue_neg_998_mod_28_l107_10749


namespace max_equilateral_triangles_l107_10757

theorem max_equilateral_triangles (length : ℕ) (n : ℕ) (segments : ℕ) : 
  (length = 2) → (segments = 6) → (∀ t, 1 ≤ t ∧ t ≤ 4 → t = 4) :=
by 
  intros length_eq segments_eq h
  sorry

end max_equilateral_triangles_l107_10757


namespace negation_exists_implies_forall_l107_10714

theorem negation_exists_implies_forall : 
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by
  sorry

end negation_exists_implies_forall_l107_10714


namespace trigonometric_identity_example_l107_10761

theorem trigonometric_identity_example :
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_example_l107_10761


namespace sally_cards_l107_10712

theorem sally_cards (initial_cards dan_cards bought_cards : ℕ) (h1 : initial_cards = 27) (h2 : dan_cards = 41) (h3 : bought_cards = 20) :
  initial_cards + dan_cards + bought_cards = 88 := by
  sorry

end sally_cards_l107_10712


namespace quadratic_eq_solutions_l107_10779

theorem quadratic_eq_solutions (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = -1 ∨ x = 3 := by
  sorry

end quadratic_eq_solutions_l107_10779


namespace fraction_sum_proof_l107_10759

theorem fraction_sum_proof :
    (19 / ((2^3 - 1) * (3^3 - 1)) + 
     37 / ((3^3 - 1) * (4^3 - 1)) + 
     61 / ((4^3 - 1) * (5^3 - 1)) + 
     91 / ((5^3 - 1) * (6^3 - 1))) = (208 / 1505) :=
by
  -- Proof goes here
  sorry

end fraction_sum_proof_l107_10759


namespace power_sum_result_l107_10702

theorem power_sum_result : (64 ^ (-1/3 : ℝ)) + (81 ^ (-1/4 : ℝ)) = (7 / 12 : ℝ) :=
by
  have h64 : (64 : ℝ) = 2 ^ 6 := by norm_num
  have h81 : (81 : ℝ) = 3 ^ 4 := by norm_num
  sorry

end power_sum_result_l107_10702


namespace total_animals_after_addition_l107_10700

def current_cows := 2
def current_pigs := 3
def current_goats := 6

def added_cows := 3
def added_pigs := 5
def added_goats := 2

def total_current_animals := current_cows + current_pigs + current_goats
def total_added_animals := added_cows + added_pigs + added_goats
def total_animals := total_current_animals + total_added_animals

theorem total_animals_after_addition : total_animals = 21 := by
  sorry

end total_animals_after_addition_l107_10700


namespace abc_value_l107_10763

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * (b + c) = 171) 
    (h2 : b * (c + a) = 180) 
    (h3 : c * (a + b) = 189) :
    a * b * c = 270 :=
by
  -- Place proofs here
  sorry

end abc_value_l107_10763


namespace probability_of_number_less_than_three_l107_10747

theorem probability_of_number_less_than_three :
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Finset ℕ := {1, 2}
  (favorable_outcomes.card : ℚ) / (faces.card : ℚ) = 1 / 3 :=
by
  -- This is the placeholder for the actual proof.
  sorry

end probability_of_number_less_than_three_l107_10747


namespace find_functions_l107_10717

variable (f : ℝ → ℝ)

def isFunctionPositiveReal := ∀ x : ℝ, x > 0 → f x > 0

axiom functional_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) : f (x ^ y) = f x ^ f y

theorem find_functions (hf : isFunctionPositiveReal f) :
  (∀ x : ℝ, x > 0 → f x = 1) ∨ (∀ x : ℝ, x > 0 → f x = x) := sorry

end find_functions_l107_10717


namespace ratio_of_new_r_to_original_r_l107_10744

theorem ratio_of_new_r_to_original_r
  (r₁ r₂ : ℝ)
  (a₁ a₂ : ℝ)
  (h₁ : a₁ = (2 * r₁)^3)
  (h₂ : a₂ = (2 * r₂)^3)
  (h : a₂ = 0.125 * a₁) :
  r₂ / r₁ = 1 / 2 :=
by
  sorry

end ratio_of_new_r_to_original_r_l107_10744


namespace moe_pie_share_l107_10795

theorem moe_pie_share
  (leftover_pie : ℚ)
  (num_people : ℕ)
  (H_leftover : leftover_pie = 5 / 8)
  (H_people : num_people = 4) :
  (leftover_pie / num_people = 5 / 32) :=
by
  sorry

end moe_pie_share_l107_10795


namespace fraction_problem_l107_10731

noncomputable def zero_point_one_five : ℚ := 5 / 33
noncomputable def two_point_four_zero_three : ℚ := 2401 / 999

theorem fraction_problem :
  (zero_point_one_five / two_point_four_zero_three) = (4995 / 79233) :=
by
  sorry

end fraction_problem_l107_10731


namespace range_a_and_inequality_l107_10768

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log (x + 2)
noncomputable def f' (x a : ℝ) : ℝ := 2 * x - a / (x + 2)

theorem range_a_and_inequality (a x1 x2 : ℝ) (h_deriv: ∀ (x : ℝ), f' x a = 0 → x = x1 ∨ x = x2) (h_lt: x1 < x2) (h_extreme: f (x1) a = f (x2) a):
  (-2 < a ∧ a < 0) → 
  (f (x1) a / x2 + 1 < 0) :=
by
  sorry

end range_a_and_inequality_l107_10768


namespace probability_of_neither_event_l107_10716

theorem probability_of_neither_event (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.25) (h2 : P_B = 0.40) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.50 :=
by
  rw [h1, h2, h3]
  sorry

end probability_of_neither_event_l107_10716


namespace min_value_inequality_l107_10741

theorem min_value_inequality (y1 y2 y3 : ℝ) (h_pos : 0 < y1 ∧ 0 < y2 ∧ 0 < y3) (h_sum : 2 * y1 + 3 * y2 + 4 * y3 = 120) :
  y1^2 + 4 * y2^2 + 9 * y3^2 ≥ 14400 / 29 :=
sorry

end min_value_inequality_l107_10741


namespace solve_fractional_eq1_l107_10753

theorem solve_fractional_eq1 : ¬ ∃ (x : ℝ), 1 / (x - 2) = (1 - x) / (2 - x) - 3 :=
by sorry

end solve_fractional_eq1_l107_10753


namespace rectangle_area_and_perimeter_l107_10721

-- Given conditions as definitions
def length : ℕ := 5
def width : ℕ := 3

-- Proof problems
theorem rectangle_area_and_perimeter :
  (length * width = 15) ∧ (2 * (length + width) = 16) :=
by
  sorry

end rectangle_area_and_perimeter_l107_10721


namespace problem_inequality_l107_10783

theorem problem_inequality (k m n : ℕ) (hk1 : 1 < k) (hkm : k ≤ m) (hmn : m < n) :
  (1 + m) ^ 2 > (1 + n) ^ m :=
  sorry

end problem_inequality_l107_10783


namespace problem_l107_10746

noncomputable def f (x φ : ℝ) : ℝ := 4 * Real.cos (3 * x + φ)

theorem problem 
  (φ : ℝ) (x1 x2 : ℝ)
  (hφ : |φ| < Real.pi / 2)
  (h_symm : ∀ x, f x φ = f (2 * (11 * Real.pi / 12) - x) φ)
  (hx1x2 : x1 ≠ x2)
  (hx1_range : -7 * Real.pi / 12 < x1 ∧ x1 < -Real.pi / 12)
  (hx2_range : -7 * Real.pi / 12 < x2 ∧ x2 < -Real.pi / 12)
  (h_eq : f x1 φ = f x2 φ) : 
  f (x1 + x2) (-Real.pi / 4) = 2 * Real.sqrt 2 := by
  sorry

end problem_l107_10746


namespace find_f_neg_2_l107_10703

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 3 * x + 4 else 7 - 3 * x

theorem find_f_neg_2 : f (-2) = 13 := by
  sorry

end find_f_neg_2_l107_10703


namespace cost_formula_correct_l107_10732

def total_cost (P : ℕ) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem cost_formula_correct (P : ℕ) : 
  total_cost P = (if P ≤ 2 then 15 else 15 + 5 * (P - 2)) :=
by 
  exact rfl

end cost_formula_correct_l107_10732


namespace min_slope_of_tangent_l107_10752

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem min_slope_of_tangent : (∀ x : ℝ, 3 * (x + 1)^2 + 3 ≥ 3) :=
by 
  sorry

end min_slope_of_tangent_l107_10752


namespace find_unknown_rate_l107_10719

theorem find_unknown_rate :
  ∃ x : ℝ, (300 + 750 + 2 * x) / 10 = 170 ↔ x = 325 :=
by
    sorry

end find_unknown_rate_l107_10719


namespace arc_length_of_circle_l107_10726

theorem arc_length_of_circle (r : ℝ) (θ_peripheral : ℝ) (h_r : r = 5) (h_θ : θ_peripheral = 2/3 * π) :
  r * (2/3 * θ_peripheral) = 20 * π / 3 := 
by sorry

end arc_length_of_circle_l107_10726


namespace line_equation_through_M_P_Q_l107_10787

-- Given that M is the midpoint between P and Q, we should have:
-- M = (1, -2)
-- P = (2, 0)
-- Q = (0, -4)
-- We need to prove that the line passing through these points has the equation 2x - y - 4 = 0

theorem line_equation_through_M_P_Q :
  ∀ (x y : ℝ), (1 - 2 = (2 * (x - 1)) ∧ 0 - 2 = (2 * (0 - (-2)))) ->
  (x - y - 4 = 0) := 
by
  sorry

end line_equation_through_M_P_Q_l107_10787


namespace ellipse_conjugate_diameters_l107_10781

variable (A B C D E : ℝ)

theorem ellipse_conjugate_diameters :
  (A * E - B * D = 0) ∧ (2 * B ^ 2 + (A - C) * A = 0) :=
sorry

end ellipse_conjugate_diameters_l107_10781


namespace solve_equation1_solve_equation2_l107_10739

-- Let x be a real number
variable {x : ℝ}

-- The first equation and its solutions
def equation1 (x : ℝ) : Prop := (x - 1) ^ 2 - 25 = 0

-- Asserting that the solutions to the first equation are x = 6 or x = -4
theorem solve_equation1 (x : ℝ) : equation1 x ↔ x = 6 ∨ x = -4 :=
by
  sorry

-- The second equation and its solution
def equation2 (x : ℝ) : Prop := (1 / 4) * (2 * x + 3) ^ 3 = 16

-- Asserting that the solution to the second equation is x = 1/2
theorem solve_equation2 (x : ℝ) : equation2 x ↔ x = 1 / 2 :=
by
  sorry

end solve_equation1_solve_equation2_l107_10739


namespace roy_missed_days_l107_10729

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end roy_missed_days_l107_10729


namespace rectangle_area_l107_10724

theorem rectangle_area
  (s : ℝ)
  (h_square_area : s^2 = 49)
  (rect_width : ℝ := s)
  (rect_length : ℝ := 3 * rect_width)
  (h_rect_width_eq_s : rect_width = s)
  (h_rect_length_eq_3w : rect_length = 3 * rect_width) :
  rect_width * rect_length = 147 :=
by 
  skip
  sorry

end rectangle_area_l107_10724


namespace smallest_solution_eq_l107_10777

theorem smallest_solution_eq (x : ℝ) (h : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) :
  x = 4 - Real.sqrt 2 := 
  sorry

end smallest_solution_eq_l107_10777


namespace find_principal_amount_l107_10727

theorem find_principal_amount 
  (total_interest : ℝ)
  (rate1 rate2 : ℝ)
  (years1 years2 : ℕ)
  (P : ℝ)
  (A1 A2 : ℝ) 
  (hA1 : A1 = P * (1 + rate1/100)^years1)
  (hA2 : A2 = A1 * (1 + rate2/100)^years2)
  (hInterest : A2 = P + total_interest) : 
  P = 25252.57 :=
by
  -- Given the conditions above, we prove the main statement.
  sorry

end find_principal_amount_l107_10727


namespace circle_equation_l107_10718

open Real

variable {x y : ℝ}

theorem circle_equation (a : ℝ) (h_a_positive : a > 0) 
    (h_tangent : abs (3 * a + 4) / sqrt (3^2 + 4^2) = 2) :
    (∀ x y : ℝ, (x - a)^2 + y^2 = 4) := sorry

end circle_equation_l107_10718


namespace arrange_abc_l107_10764

open Real

noncomputable def a := log 4 / log 5
noncomputable def b := (log 3 / log 5)^2
noncomputable def c := 1 / (log 4 / log 5)

theorem arrange_abc : b < a ∧ a < c :=
by
  -- Mathematical translations as Lean proof obligations
  have a_lt_one : a < 1 := by sorry
  have c_gt_one : c > 1 := by sorry
  have b_lt_a : b < a := by sorry
  have a_lt_c : a < c := by sorry
  exact ⟨b_lt_a, a_lt_c⟩

end arrange_abc_l107_10764


namespace geometric_sequence_product_l107_10767

theorem geometric_sequence_product (a1 a5 : ℚ) (a b c : ℚ) (q : ℚ) 
  (h1 : a1 = 8 / 3) 
  (h5 : a5 = 27 / 2)
  (h_common_ratio_pos : q = 3 / 2)
  (h_a : a = a1 * q)
  (h_b : b = a * q)
  (h_c : c = b * q)
  (h5_eq : a5 = a1 * q^4)
  (h_common_ratio_neg : q = -3 / 2 ∨ q = 3 / 2) :
  a * b * c = 216 := by
    sorry

end geometric_sequence_product_l107_10767


namespace no_solution_for_k_eq_six_l107_10775

theorem no_solution_for_k_eq_six :
  ∀ x k : ℝ, k = 6 → (x ≠ 2 ∧ x ≠ 7) → (x - 1) / (x - 2) = (x - k) / (x - 7) → false :=
by 
  intros x k hk hnx_eq h_eq
  sorry

end no_solution_for_k_eq_six_l107_10775


namespace inverse_of_A_cubed_l107_10797

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7],
    ![-2, -5]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3)⁻¹ = ![![13, -15],
                     ![-14, -29]] :=
by
  sorry

end inverse_of_A_cubed_l107_10797


namespace remainder_when_divided_by_17_l107_10715

theorem remainder_when_divided_by_17
  (N k : ℤ)
  (h : N = 357 * k + 36) :
  N % 17 = 2 :=
by
  sorry

end remainder_when_divided_by_17_l107_10715


namespace no_unique_solution_for_c_l107_10734

theorem no_unique_solution_for_c (k : ℕ) (hk : k = 9) (c : ℕ) :
  (∀ x y : ℕ, 9 * x + c * y = 30 → 3 * x + 4 * y = 12) → c = 12 :=
by
  sorry

end no_unique_solution_for_c_l107_10734


namespace find_a_b_find_k_range_l107_10745

-- Define the conditions for part 1
def quad_inequality (a x : ℝ) : Prop :=
  a * x^2 - 3 * x + 2 > 0

def solution_set (x b : ℝ) : Prop :=
  x < 1 ∨ x > b

theorem find_a_b (a b : ℝ) :
  (∀ x, quad_inequality a x ↔ solution_set x b) → (a = 1 ∧ b = 2) :=
sorry

-- Define the conditions for part 2
def valid_x_y (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def equation1 (a b x y : ℝ) : Prop :=
  a / x + b / y = 1

def inequality1 (x y k : ℝ) : Prop :=
  2 * x + y ≥ k^2 + k + 2

theorem find_k_range (a b : ℝ) (x y k : ℝ) :
  a = 1 → b = 2 → valid_x_y x y → equation1 a b x y → inequality1 x y k →
  (-3 ≤ k ∧ k ≤ 2) :=
sorry

end find_a_b_find_k_range_l107_10745


namespace average_of_sequence_l107_10784

theorem average_of_sequence (z : ℝ) : 
  (0 + 3 * z + 9 * z + 27 * z + 81 * z) / 5 = 24 * z :=
by
  sorry

end average_of_sequence_l107_10784


namespace batsman_average_l107_10780

theorem batsman_average (A : ℕ) (H : (16 * A + 82) / 17 = A + 3) : (A + 3 = 34) :=
sorry

end batsman_average_l107_10780


namespace roots_of_polynomial_l107_10790

theorem roots_of_polynomial : 
  ∀ (x : ℝ), (x^2 + 4) * (x^2 - 4) = 0 ↔ (x = -2 ∨ x = 2) :=
by 
  sorry

end roots_of_polynomial_l107_10790


namespace shrimp_cost_per_pound_l107_10766

theorem shrimp_cost_per_pound 
    (shrimp_per_guest : ℕ) 
    (num_guests : ℕ) 
    (shrimp_per_pound : ℕ) 
    (total_cost : ℝ)
    (H1 : shrimp_per_guest = 5)
    (H2 : num_guests = 40)
    (H3 : shrimp_per_pound = 20)
    (H4 : total_cost = 170) : 
    (total_cost / ((num_guests * shrimp_per_guest) / shrimp_per_pound) = 17) :=
by
    sorry

end shrimp_cost_per_pound_l107_10766


namespace record_expenditure_20_l107_10762

-- Define the concept of recording financial transactions
def record_income (amount : ℤ) : ℤ := amount

def record_expenditure (amount : ℤ) : ℤ := -amount

-- Given conditions
variable (income : ℤ) (expenditure : ℤ)

-- Condition: the income of 30 yuan is recorded as +30 yuan
axiom income_record : record_income 30 = 30

-- Prove an expenditure of 20 yuan is recorded as -20 yuan
theorem record_expenditure_20 : record_expenditure 20 = -20 := 
  by sorry

end record_expenditure_20_l107_10762


namespace glass_bottles_count_l107_10798

-- Declare the variables for the conditions
variable (G : ℕ)

-- Define the conditions
def aluminum_cans : ℕ := 8
def total_litter : ℕ := 18

-- State the theorem
theorem glass_bottles_count : G + aluminum_cans = total_litter → G = 10 :=
by
  intro h
  -- place proof here
  sorry

end glass_bottles_count_l107_10798


namespace smallest_n_for_property_l107_10770

theorem smallest_n_for_property (n x : ℕ) (d : ℕ) (c : ℕ) 
  (hx : x = 10 * c + d) 
  (hx_prop : 10^(n-1) * d + c = 2 * x) :
  n = 18 := 
sorry

end smallest_n_for_property_l107_10770


namespace petals_per_ounce_l107_10791

-- Definitions of the given conditions
def petals_per_rose : ℕ := 8
def roses_per_bush : ℕ := 12
def bushes_harvested : ℕ := 800
def bottles_produced : ℕ := 20
def ounces_per_bottle : ℕ := 12

-- Calculation of petals per bush
def petals_per_bush : ℕ := roses_per_bush * petals_per_rose

-- Calculation of total petals harvested
def total_petals_harvested : ℕ := bushes_harvested * petals_per_bush

-- Calculation of total ounces of perfume
def total_ounces_produced : ℕ := bottles_produced * ounces_per_bottle

-- Main theorem statement
theorem petals_per_ounce : total_petals_harvested / total_ounces_produced = 320 :=
by
  sorry

end petals_per_ounce_l107_10791


namespace complement_of_A_inter_B_eq_l107_10799

noncomputable def A : Set ℝ := {x | abs (x - 1) ≤ 1}
noncomputable def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}
noncomputable def A_inter_B : Set ℝ := {x | x ∈ A ∧ x ∈ B}
noncomputable def complement_A_inter_B : Set ℝ := {x | x ∉ A_inter_B}

theorem complement_of_A_inter_B_eq :
  complement_A_inter_B = {x : ℝ | x ≠ 0} :=
  sorry

end complement_of_A_inter_B_eq_l107_10799


namespace solve_for_x_l107_10704

theorem solve_for_x :
  (∀ x : ℝ, (1 / Real.log x / Real.log 3 + 1 / Real.log x / Real.log 4 + 1 / Real.log x / Real.log 5 = 2))
  → x = 2 * Real.sqrt 15 :=
by
  sorry

end solve_for_x_l107_10704


namespace add_fractions_l107_10769

theorem add_fractions :
  (11 / 12) + (7 / 8) + (3 / 4) = 61 / 24 :=
by
  sorry

end add_fractions_l107_10769


namespace gecko_cricket_eating_l107_10736

theorem gecko_cricket_eating :
  ∀ (total_crickets : ℕ) (first_day_percent : ℚ) (second_day_less : ℕ),
    total_crickets = 70 →
    first_day_percent = 0.3 →
    second_day_less = 6 →
    let first_day_crickets := total_crickets * first_day_percent
    let second_day_crickets := first_day_crickets - second_day_less
    total_crickets - first_day_crickets - second_day_crickets = 34 :=
by
  intros total_crickets first_day_percent second_day_less h_total h_percent h_less
  let first_day_crickets := total_crickets * first_day_percent
  let second_day_crickets := first_day_crickets - second_day_less
  have : total_crickets - first_day_crickets - second_day_crickets = 34 := sorry
  exact this

end gecko_cricket_eating_l107_10736


namespace sale_in_third_month_l107_10758

def average_sale (s1 s2 s3 s4 s5 s6 : ℕ) : ℕ :=
  (s1 + s2 + s3 + s4 + s5 + s6) / 6

theorem sale_in_third_month
  (S1 S2 S3 S4 S5 S6 : ℕ)
  (h1 : S1 = 6535)
  (h2 : S2 = 6927)
  (h4 : S4 = 7230)
  (h5 : S5 = 6562)
  (h6 : S6 = 4891)
  (havg : average_sale S1 S2 S3 S4 S5 S6 = 6500) :
  S3 = 6855 := 
sorry

end sale_in_third_month_l107_10758


namespace xiaohui_pe_score_l107_10728

-- Define the conditions
def morning_score : ℝ := 95
def midterm_score : ℝ := 90
def final_score : ℝ := 85

def morning_weight : ℝ := 0.2
def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.5

-- The problem is to prove that Xiaohui's physical education score for the semester is 88.5 points.
theorem xiaohui_pe_score :
  morning_score * morning_weight +
  midterm_score * midterm_weight +
  final_score * final_weight = 88.5 :=
by
  sorry

end xiaohui_pe_score_l107_10728
