import Mathlib

namespace simplify_expression_inequality_solution_l1367_136772

-- Simplification part
theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2):
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10 * x + 25) / (x^2 - 4)) = 
  (x - 2) / (x + 5) :=
sorry

-- Inequality system part
theorem inequality_solution (x : ℝ):
  (2 * x + 7 > 3) ∧ ((x + 1) / 3 > (x - 1) / 2) → -2 < x ∧ x < 5 :=
sorry

end simplify_expression_inequality_solution_l1367_136772


namespace mary_earns_per_home_l1367_136744

noncomputable def earnings_per_home (T : ℕ) (n : ℕ) : ℕ := T / n

theorem mary_earns_per_home :
  ∀ (T n : ℕ), T = 276 → n = 6 → earnings_per_home T n = 46 := 
by
  intros T n h1 h2
  -- Placeholder proof step
  sorry

end mary_earns_per_home_l1367_136744


namespace car_distance_ratio_l1367_136790

theorem car_distance_ratio (speed_A time_A speed_B time_B : ℕ) 
  (hA : speed_A = 70) (hTA : time_A = 10) 
  (hB : speed_B = 35) (hTB : time_B = 10) : 
  (speed_A * time_A) / gcd (speed_A * time_A) (speed_B * time_B) = 2 :=
by
  sorry

end car_distance_ratio_l1367_136790


namespace cricket_run_rate_l1367_136769

theorem cricket_run_rate 
  (run_rate_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_played : ℕ)
  (remaining_overs : ℕ)
  (correct_run_rate : ℝ)
  (h1 : run_rate_10_overs = 3.6)
  (h2 : target_runs = 282)
  (h3 : overs_played = 10)
  (h4 : remaining_overs = 40)
  (h5 : correct_run_rate = 6.15) :
  (target_runs - run_rate_10_overs * overs_played) / remaining_overs = correct_run_rate :=
sorry

end cricket_run_rate_l1367_136769


namespace total_annual_car_maintenance_expenses_is_330_l1367_136735

-- Define the conditions as constants
def annualMileage : ℕ := 12000
def milesPerOilChange : ℕ := 3000
def freeOilChangesPerYear : ℕ := 1
def costPerOilChange : ℕ := 50
def milesPerTireRotation : ℕ := 6000
def costPerTireRotation : ℕ := 40
def milesPerBrakePadReplacement : ℕ := 24000
def costPerBrakePadReplacement : ℕ := 200

-- Define the total annual car maintenance expenses calculation
def annualOilChangeExpenses (annualMileage : ℕ) (milesPerOilChange : ℕ) (freeOilChangesPerYear : ℕ) (costPerOilChange : ℕ) : ℕ :=
  let oilChangesNeeded := annualMileage / milesPerOilChange
  let paidOilChanges := oilChangesNeeded - freeOilChangesPerYear
  paidOilChanges * costPerOilChange

def annualTireRotationExpenses (annualMileage : ℕ) (milesPerTireRotation : ℕ) (costPerTireRotation : ℕ) : ℕ :=
  let tireRotationsNeeded := annualMileage / milesPerTireRotation
  tireRotationsNeeded * costPerTireRotation

def annualBrakePadReplacementExpenses (annualMileage : ℕ) (milesPerBrakePadReplacement : ℕ) (costPerBrakePadReplacement : ℕ) : ℕ :=
  let brakePadReplacementInterval := milesPerBrakePadReplacement / annualMileage
  costPerBrakePadReplacement / brakePadReplacementInterval

def totalAnnualCarMaintenanceExpenses : ℕ :=
  annualOilChangeExpenses annualMileage milesPerOilChange freeOilChangesPerYear costPerOilChange +
  annualTireRotationExpenses annualMileage milesPerTireRotation costPerTireRotation +
  annualBrakePadReplacementExpenses annualMileage milesPerBrakePadReplacement costPerBrakePadReplacement

-- Prove the total annual car maintenance expenses equals $330
theorem total_annual_car_maintenance_expenses_is_330 : totalAnnualCarMaintenanceExpenses = 330 := by
  sorry

end total_annual_car_maintenance_expenses_is_330_l1367_136735


namespace abc_order_l1367_136756

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := 0.5^3
noncomputable def c : Real := Real.log 3 / Real.log 0.5 -- log_0.5 3 is written as (log 3) / (log 0.5) in Lean

theorem abc_order : a > b ∧ b > c :=
by
  have h1 : a = Real.sqrt 3 := rfl
  have h2 : b = 0.5^3 := rfl
  have h3 : c = Real.log 3 / Real.log 0.5 := rfl
  sorry

end abc_order_l1367_136756


namespace gina_total_pay_l1367_136767

noncomputable def gina_painting_pay : ℕ :=
let roses_per_hour := 6
let lilies_per_hour := 7
let rose_order := 6
let lily_order := 14
let pay_per_hour := 30

-- Calculate total time (in hours) Gina spends to complete the order
let time_for_roses := rose_order / roses_per_hour
let time_for_lilies := lily_order / lilies_per_hour
let total_time := time_for_roses + time_for_lilies

-- Calculate the total pay
let total_pay := total_time * pay_per_hour

total_pay

-- The theorem that Gina gets paid $90 for the order
theorem gina_total_pay : gina_painting_pay = 90 := by
  sorry

end gina_total_pay_l1367_136767


namespace percentage_students_passed_is_35_l1367_136704

/-
The problem is to prove the percentage of students who passed the examination, given that 520 out of 800 students failed, is 35%.
-/

def total_students : ℕ := 800
def failed_students : ℕ := 520
def passed_students : ℕ := total_students - failed_students

def percentage_passed : ℕ := (passed_students * 100) / total_students

theorem percentage_students_passed_is_35 : percentage_passed = 35 :=
by
  -- Here the proof will go.
  sorry

end percentage_students_passed_is_35_l1367_136704


namespace fem_current_age_l1367_136789

theorem fem_current_age (F : ℕ) 
  (h1 : ∃ M : ℕ, M = 4 * F) 
  (h2 : (F + 2) + (4 * F + 2) = 59) : 
  F = 11 :=
sorry

end fem_current_age_l1367_136789


namespace unique_solution_exists_l1367_136716

theorem unique_solution_exists :
  ∃! (x y : ℝ), 4^(x^2 + 2 * y) + 4^(2 * x + y^2) = Real.cos (Real.pi * x) ∧ (x, y) = (2, -2) :=
by
  sorry

end unique_solution_exists_l1367_136716


namespace cyclists_meet_fourth_time_l1367_136706

theorem cyclists_meet_fourth_time 
  (speed1 speed2 speed3 speed4 : ℕ)
  (len : ℚ)
  (t_start : ℕ)
  (h_speed1 : speed1 = 6)
  (h_speed2 : speed2 = 9)
  (h_speed3 : speed3 = 12)
  (h_speed4 : speed4 = 15)
  (h_len : len = 1 / 3)
  (h_t_start : t_start = 12 * 60 * 60)
  : 
  (t_start + 4 * (20 * 60 + 40)) = 12 * 60 * 60 + 1600  :=
sorry

end cyclists_meet_fourth_time_l1367_136706


namespace spontaneous_low_temperature_l1367_136721

theorem spontaneous_low_temperature (ΔH ΔS T : ℝ) (spontaneous : ΔG = ΔH - T * ΔS) :
  (∀ T, T > 0 → ΔG < 0 → ΔH < 0 ∧ ΔS < 0) := 
by 
  sorry

end spontaneous_low_temperature_l1367_136721


namespace total_time_before_main_game_l1367_136786

-- Define the time spent on each activity according to the conditions
def download_time := 10
def install_time := download_time / 2
def update_time := 2 * download_time
def account_time := 5
def internet_issues_time := 15
def discussion_time := 20
def video_time := 8

-- Define the total preparation time
def preparation_time := download_time + install_time + update_time + account_time + internet_issues_time + discussion_time + video_time

-- Define the in-game tutorial time
def tutorial_time := 3 * preparation_time

-- Prove that the total time before playing the main game is 332 minutes
theorem total_time_before_main_game : preparation_time + tutorial_time = 332 := by
  -- Provide a detailed proof here
  sorry

end total_time_before_main_game_l1367_136786


namespace fewer_green_pens_than_pink_l1367_136733

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end fewer_green_pens_than_pink_l1367_136733


namespace field_ratio_l1367_136753

theorem field_ratio (side pond_area_ratio : ℝ) (field_length : ℝ) 
  (pond_is_square: pond_area_ratio = 1/18) 
  (side_length: side = 8) 
  (field_len: field_length = 48) : 
  (field_length / (pond_area_ratio * side ^ 2 / side)) = 2 :=
by
  sorry

end field_ratio_l1367_136753


namespace fraction_meaningful_if_and_only_if_l1367_136714

theorem fraction_meaningful_if_and_only_if {x : ℝ} : (2 * x - 1 ≠ 0) ↔ (x ≠ 1 / 2) :=
by
  sorry

end fraction_meaningful_if_and_only_if_l1367_136714


namespace probability_fail_then_succeed_l1367_136728

theorem probability_fail_then_succeed
  (P_fail_first : ℚ := 9 / 10)
  (P_succeed_second : ℚ := 1 / 9) :
  P_fail_first * P_succeed_second = 1 / 10 :=
by
  sorry

end probability_fail_then_succeed_l1367_136728


namespace angle_passing_through_point_l1367_136707

-- Definition of the problem conditions
def is_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  let r := Real.sqrt (x^2 + y^2);
  (x = Real.cos α * r) ∧ (y = Real.sin α * r)

-- Lean 4 statement of the problem
theorem angle_passing_through_point (α : ℝ) :
  is_terminal_side_of_angle 1 (-1) α → α = - (Real.pi / 4) :=
by sorry

end angle_passing_through_point_l1367_136707


namespace largest_product_of_three_l1367_136747

theorem largest_product_of_three :
  ∃ (a b c : ℤ), a ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 b ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 c ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
                 a * b * c = 90 := 
sorry

end largest_product_of_three_l1367_136747


namespace arithmetic_sequence_common_difference_l1367_136710

theorem arithmetic_sequence_common_difference 
    (a : ℤ) (last_term : ℤ) (sum_terms : ℤ) (n : ℕ)
    (h1 : a = 3) 
    (h2 : last_term = 58) 
    (h3 : sum_terms = 488)
    (h4 : sum_terms = n * (a + last_term) / 2)
    (h5 : last_term = a + (n - 1) * d) :
    d = 11 / 3 := by
  sorry

end arithmetic_sequence_common_difference_l1367_136710


namespace quadratic_intersection_with_x_axis_l1367_136798

theorem quadratic_intersection_with_x_axis :
  ∃ x : ℝ, (x^2 - 4*x + 4 = 0) ∧ (x = 2) ∧ (x, 0) = (2, 0) :=
sorry

end quadratic_intersection_with_x_axis_l1367_136798


namespace incorrect_statement_among_options_l1367_136797

/- Definitions and Conditions -/
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * d

/- Conditions given in the problem -/
axiom S_6_gt_S_7 : S 6 > S 7
axiom S_7_gt_S_5 : S 7 > S 5

/- Incorrect statement to be proved -/
theorem incorrect_statement_among_options :
  ¬ (∀ n, S n ≤ S 11) := sorry

end incorrect_statement_among_options_l1367_136797


namespace money_spent_on_jacket_l1367_136737

-- Define the initial amounts
def initial_money_sandy : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def additional_money_found : ℝ := 7.43

-- Amount of money left after buying the shirt
def remaining_after_shirt := initial_money_sandy - amount_spent_shirt

-- Total money after finding additional money
def total_after_additional := remaining_after_shirt + additional_money_found

-- Theorem statement: The amount Sandy spent on the jacket
theorem money_spent_on_jacket : total_after_additional = 9.28 :=
by
  sorry

end money_spent_on_jacket_l1367_136737


namespace largest_in_given_numbers_l1367_136775

noncomputable def A := 5.14322
noncomputable def B := 5.1432222222222222222 -- B = 5.143(bar)2
noncomputable def C := 5.1432323232323232323 -- C = 5.14(bar)32
noncomputable def D := 5.1432432432432432432 -- D = 5.1(bar)432
noncomputable def E := 5.1432143214321432143 -- E = 5.(bar)4321

theorem largest_in_given_numbers : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_in_given_numbers_l1367_136775


namespace mushrooms_picked_on_second_day_l1367_136750

theorem mushrooms_picked_on_second_day :
  ∃ (n2 : ℕ), (∃ (n1 n3 : ℕ), n3 = 2 * n2 ∧ n1 + n2 + n3 = 65) ∧ n2 = 21 :=
by
  sorry

end mushrooms_picked_on_second_day_l1367_136750


namespace tommy_needs_4_steaks_l1367_136746

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l1367_136746


namespace volume_multiplication_factor_l1367_136796

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end volume_multiplication_factor_l1367_136796


namespace sum_of_coefficients_l1367_136717

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l1367_136717


namespace complement_intersection_l1367_136766

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  -- Proof is omitted.
  sorry

end complement_intersection_l1367_136766


namespace a_older_than_b_l1367_136739

theorem a_older_than_b (A B : ℕ) (h1 : B = 36) (h2 : A + 10 = 2 * (B - 10)) : A - B = 6 :=
  sorry

end a_older_than_b_l1367_136739


namespace proof_problem_l1367_136703

theorem proof_problem
  (x y : ℚ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 16) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9 :=
sorry

end proof_problem_l1367_136703


namespace cos_pi_over_3_plus_2alpha_l1367_136702

theorem cos_pi_over_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_pi_over_3_plus_2alpha_l1367_136702


namespace div_by_1897_l1367_136745

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l1367_136745


namespace remainder_of_x7_plus_2_div_x_plus_1_l1367_136726

def f (x : ℤ) := x^7 + 2

theorem remainder_of_x7_plus_2_div_x_plus_1 : 
  (f (-1) = 1) := sorry

end remainder_of_x7_plus_2_div_x_plus_1_l1367_136726


namespace solve_quadratic_l1367_136732

theorem solve_quadratic : 
  (∀ x : ℚ, 2 * x^2 - x - 6 = 0 → x = -3 / 2 ∨ x = 2) ∧ 
  (∀ y : ℚ, (y - 2)^2 = 9 * y^2 → y = -1 ∨ y = 1 / 2) := 
by
  sorry

end solve_quadratic_l1367_136732


namespace average_age_calculated_years_ago_l1367_136765

theorem average_age_calculated_years_ago
  (n m : ℕ) (a b : ℕ) 
  (total_age_original : ℝ)
  (average_age_original : ℝ)
  (average_age_new : ℝ) :
  n = 6 → 
  a = 19 → 
  m = 7 → 
  b = 1 → 
  total_age_original = n * a → 
  average_age_original = a → 
  average_age_new = a →
  (total_age_original + b) / m = a → 
  1 = 1 := 
by
  intros _ _ _ _ _ _ _ _
  sorry

end average_age_calculated_years_ago_l1367_136765


namespace ali_seashells_final_count_l1367_136759

theorem ali_seashells_final_count :
  385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25))) 
  - (1 / 4) * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)))) = 82.485 :=
sorry

end ali_seashells_final_count_l1367_136759


namespace sushi_father_lollipops_l1367_136758

variable (x : ℕ)

theorem sushi_father_lollipops (h : x - 5 = 7) : x = 12 := by
  sorry

end sushi_father_lollipops_l1367_136758


namespace find_multiple_l1367_136736

-- Defining the conditions
def first_lock_time := 5
def second_lock_time (x : ℕ) := 5 * x - 3

-- Proving the multiple
theorem find_multiple : 
  ∃ x : ℕ, (5 * first_lock_time * x - 3) * 5 = 60 ∧ (x = 3) :=
by
  sorry

end find_multiple_l1367_136736


namespace excluded_angle_sum_1680_degrees_l1367_136748

theorem excluded_angle_sum_1680_degrees (sum_except_one : ℝ) (h : sum_except_one = 1680) : 
  (180 - (1680 % 180)) = 120 :=
by
  have mod_eq : 1680 % 180 = 60 := by sorry
  rw [mod_eq]

end excluded_angle_sum_1680_degrees_l1367_136748


namespace resistance_at_least_2000_l1367_136776

variable (U : ℝ) (I : ℝ) (R : ℝ)

-- Given conditions:
def voltage := U = 220
def max_current := I ≤ 0.11

-- Ohm's law in this context
def ohms_law := I = U / R

-- Proof problem statement:
theorem resistance_at_least_2000 (voltage : U = 220) (max_current : I ≤ 0.11) (ohms_law : I = U / R) : R ≥ 2000 :=
sorry

end resistance_at_least_2000_l1367_136776


namespace prob_product_less_than_36_is_15_over_16_l1367_136741

noncomputable def prob_product_less_than_36 : ℚ := sorry

theorem prob_product_less_than_36_is_15_over_16 :
  prob_product_less_than_36 = 15 / 16 := 
sorry

end prob_product_less_than_36_is_15_over_16_l1367_136741


namespace initial_fraction_spent_on_clothes_l1367_136711

-- Define the conditions and the theorem to be proved
theorem initial_fraction_spent_on_clothes 
  (M : ℝ) (F : ℝ)
  (h1 : M = 249.99999999999994)
  (h2 : (3 / 4) * (4 / 5) * (1 - F) * M = 100) :
  F = 11 / 15 :=
sorry

end initial_fraction_spent_on_clothes_l1367_136711


namespace positive_expressions_l1367_136787

-- Define the approximate values for A, B, C, D, and E.
def A := 2.5
def B := -2.1
def C := -0.3
def D := 1.0
def E := -0.7

-- Define the expressions that we need to prove as positive numbers.
def exprA := A + B
def exprB := B * C
def exprD := E / (A * B)

-- The theorem states that expressions (A + B), (B * C), and (E / (A * B)) are positive.
theorem positive_expressions : exprA > 0 ∧ exprB > 0 ∧ exprD > 0 := 
by sorry

end positive_expressions_l1367_136787


namespace max_rank_awarded_l1367_136705

theorem max_rank_awarded (num_participants rank_threshold total_possible_points : ℕ)
  (H1 : num_participants = 30)
  (H2 : rank_threshold = (30 * 29 / 2 : ℚ) * 0.60)
  (H3 : total_possible_points = (30 * 29 / 2)) :
  ∃ max_awarded : ℕ, max_awarded ≤ 23 :=
by {
  -- Proof omitted
  sorry
}

end max_rank_awarded_l1367_136705


namespace chess_game_probability_l1367_136778

theorem chess_game_probability (p_A_wins p_draw : ℝ) (h1 : p_A_wins = 0.3) (h2 : p_draw = 0.2) :
  p_A_wins + p_draw = 0.5 :=
by
  rw [h1, h2]
  norm_num

end chess_game_probability_l1367_136778


namespace find_x1_l1367_136770

noncomputable def parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem find_x1 
  (a h k m x1 : ℝ)
  (h1 : parabola a h k (-1) = 2)
  (h2 : parabola a h k 1 = -2)
  (h3 : parabola a h k 3 = 2)
  (h4 : parabola a h k (-2) = m)
  (h5 : parabola a h k x1 = m) :
  x1 = 4 := 
sorry

end find_x1_l1367_136770


namespace geom_seq_common_ratio_l1367_136724

-- We define a geometric sequence and the condition provided in the problem.
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Condition for geometric sequence: a_n = a * q^(n-1)
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^(n-1)

-- Given condition: 2a_4 = a_6 - a_5
def given_condition (a : ℕ → ℝ) : Prop := 
  2 * a 4 = a 6 - a 5

-- Proof statement
theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_seq a q) (h_cond : given_condition a) : 
    q = 2 ∨ q = -1 :=
sorry

end geom_seq_common_ratio_l1367_136724


namespace triangle_proof_l1367_136713

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions
axiom cos_rule_1 : a / cos A = c / (2 - cos C)
axiom b_value : b = 4
axiom c_value : c = 3
axiom area_equation : (1 / 2) * a * b * sin C = 3

-- The theorem statement
theorem triangle_proof : 3 * sin C + 4 * cos C = 5 := sorry

end triangle_proof_l1367_136713


namespace g_value_at_6_l1367_136788

noncomputable def g (v : ℝ) : ℝ :=
  let x := (v + 2) / 4
  x^2 - x + 2

theorem g_value_at_6 :
  g 6 = 4 := by
  sorry

end g_value_at_6_l1367_136788


namespace minimum_bailing_rate_l1367_136784

theorem minimum_bailing_rate
  (distance : ℝ) (to_shore_rate : ℝ) (water_in_rate : ℝ) (submerge_limit : ℝ) (r : ℝ)
  (h_distance : distance = 0.5) 
  (h_speed : to_shore_rate = 6) 
  (h_water_intake : water_in_rate = 12) 
  (h_submerge_limit : submerge_limit = 50)
  (h_time : (distance / to_shore_rate) * 60 = 5)
  (h_total_intake : water_in_rate * 5 = 60)
  (h_max_intake : submerge_limit - 60 = -10) :
  r = 2 := sorry

end minimum_bailing_rate_l1367_136784


namespace bad_oranges_l1367_136738

theorem bad_oranges (total_oranges : ℕ) (students : ℕ) (less_oranges_per_student : ℕ)
  (initial_oranges_per_student now_oranges_per_student shared_oranges now_total_oranges bad_oranges : ℕ) :
  total_oranges = 108 →
  students = 12 →
  less_oranges_per_student = 3 →
  initial_oranges_per_student = total_oranges / students →
  now_oranges_per_student = initial_oranges_per_student - less_oranges_per_student →
  shared_oranges = students * now_oranges_per_student →
  now_total_oranges = 72 →
  bad_oranges = total_oranges - now_total_oranges →
  bad_oranges = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bad_oranges_l1367_136738


namespace union_A_B_intersection_A_CI_B_l1367_136764

-- Define the sets
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 5, 6, 7}

-- Define the complement of B in the universal set I
def C_I (I : Set ℕ) (B : Set ℕ) : Set ℕ := {x ∈ I | x ∉ B}

-- The theorem for the union of A and B
theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6, 7} := sorry

-- The theorem for the intersection of A and the complement of B in I
theorem intersection_A_CI_B : A ∩ (C_I I B) = {1, 2, 4} := sorry

end union_A_B_intersection_A_CI_B_l1367_136764


namespace number_of_valid_n_l1367_136752

theorem number_of_valid_n : 
  (∃ (n : ℕ), ∀ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 → n = a + 2 * b + 3 * c) ↔
  (∃ (n : ℕ), n = 1000) := by 
  sorry

end number_of_valid_n_l1367_136752


namespace number_of_pairs_of_socks_l1367_136761

theorem number_of_pairs_of_socks (n : ℕ) (h : 2 * n^2 - n = 112) : n = 16 := sorry

end number_of_pairs_of_socks_l1367_136761


namespace find_x_val_l1367_136719

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end find_x_val_l1367_136719


namespace smallest_multiple_36_45_not_11_l1367_136785

theorem smallest_multiple_36_45_not_11 (n : ℕ) :
  (n = 180) ↔ (n > 0 ∧ (36 ∣ n) ∧ (45 ∣ n) ∧ ¬ (11 ∣ n)) :=
by
  sorry

end smallest_multiple_36_45_not_11_l1367_136785


namespace number_of_men_in_first_group_l1367_136708

-- Define the conditions and the proof problem
theorem number_of_men_in_first_group (M : ℕ) 
  (h1 : ∀ t : ℝ, 22 * t = M) 
  (h2 : ∀ t' : ℝ, 18 * 17.11111111111111 = t') :
  M = 14 := 
by
  sorry

end number_of_men_in_first_group_l1367_136708


namespace number_of_dress_designs_l1367_136701

open Nat

theorem number_of_dress_designs : (3 * 4 = 12) :=
by
  rfl

end number_of_dress_designs_l1367_136701


namespace sales_tax_percentage_l1367_136722

noncomputable def original_price : ℝ := 200
noncomputable def discount : ℝ := 0.25 * original_price
noncomputable def sale_price : ℝ := original_price - discount
noncomputable def total_paid : ℝ := 165
noncomputable def sales_tax : ℝ := total_paid - sale_price

theorem sales_tax_percentage : (sales_tax / sale_price) * 100 = 10 := by
  sorry

end sales_tax_percentage_l1367_136722


namespace smallest_positive_value_l1367_136760

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℚ), k = (a^2 + b^2) / (a^2 - b^2) + (a^2 - b^2) / (a^2 + b^2) ∧ k = 2 :=
sorry

end smallest_positive_value_l1367_136760


namespace more_money_from_mom_is_correct_l1367_136729

noncomputable def more_money_from_mom : ℝ :=
  let money_from_mom := 8.25
  let money_from_dad := 6.50
  let money_from_grandparents := 12.35
  let money_from_aunt := 5.10
  let money_spent_toy := 4.45
  let money_spent_snacks := 6.25
  let total_received := money_from_mom + money_from_dad + money_from_grandparents + money_from_aunt
  let total_spent := money_spent_toy + money_spent_snacks
  let money_remaining := total_received - total_spent
  let money_spent_books := 0.25 * money_remaining
  let money_left_after_books := money_remaining - money_spent_books
  money_from_mom - money_from_dad

theorem more_money_from_mom_is_correct : more_money_from_mom = 1.75 := by
  sorry

end more_money_from_mom_is_correct_l1367_136729


namespace slope_of_line_6x_minus_4y_eq_16_l1367_136720

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_of_line_6x_minus_4y_eq_16 :
  slope_of_line 6 (-4) (-16) = 3 / 2 :=
by
  -- skipping the proof
  sorry

end slope_of_line_6x_minus_4y_eq_16_l1367_136720


namespace find_f_neg_l1367_136799

noncomputable def f (a b x : ℝ) := a * x^3 + b * x - 2

theorem find_f_neg (a b : ℝ) (f_2017 : f a b 2017 = 7) : f a b (-2017) = -11 :=
by
  sorry

end find_f_neg_l1367_136799


namespace calculate_ratio_milk_l1367_136793

def ratio_milk_saturdays_weekdays (S : ℕ) : Prop :=
  let Weekdays := 15 -- total milk on weekdays
  let Sundays := 9 -- total milk on Sundays
  S + Weekdays + Sundays = 30 → S / Weekdays = 2 / 5

theorem calculate_ratio_milk : ratio_milk_saturdays_weekdays 6 :=
by
  unfold ratio_milk_saturdays_weekdays
  intros
  apply sorry -- Proof goes here

end calculate_ratio_milk_l1367_136793


namespace find_directrix_of_parabola_l1367_136783

open Real

theorem find_directrix_of_parabola (O : ℝ × ℝ) (p : ℝ) (F P Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hp_pos : p > 0)
  (hC : ∀ x y, (x, y) = P → y^2 = 2 * p * x)
  (hF : F = (p / 2, 0))
  (hPF_perpendicular_to_x : P.1 = p / 2 ∧ P.2 = p)
  (hQ_on_x_axis : Q.2 = 0)
  (hPQ_perpendicular_OP : (P.1, P.2) ≠ Q ∧ ((P.2 - Q.2) / (P.1 - Q.1) = -1 / ((P.2 - O.2) / (P.1 - O.1))))
  (hFQ_distance : abs (F.1 - Q.1) = 6) :
  x = -3 / 2 :=
sorry

end find_directrix_of_parabola_l1367_136783


namespace johns_age_l1367_136743

theorem johns_age :
  ∃ x : ℕ, (∃ n : ℕ, x - 5 = n^2) ∧ (∃ m : ℕ, x + 3 = m^3) ∧ x = 69 :=
by
  sorry

end johns_age_l1367_136743


namespace infinite_geometric_series_sum_l1367_136725

/-
Mathematical problem: Calculate the sum of the infinite geometric series 1 + (1/2) + (1/2)^2 + (1/2)^3 + ... . Express your answer as a common fraction.

Conditions:
- The first term \( a \) is 1.
- The common ratio \( r \) is \(\frac{1}{2}\).

Answer:
- The sum of the series is 2.
-/

theorem infinite_geometric_series_sum :
  let a := 1
  let r := 1 / 2
  (a * (1 / (1 - r))) = 2 :=
by
  let a := 1
  let r := 1 / 2
  have h : 1 * (1 / (1 - r)) = 2 := by sorry
  exact h

end infinite_geometric_series_sum_l1367_136725


namespace tan_neg_seven_pi_sixths_l1367_136727

noncomputable def tan_neg_pi_seven_sixths : Real :=
  -Real.sqrt 3 / 3

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l1367_136727


namespace intersect_eq_l1367_136712

variable (M N : Set Int)
def M_def : Set Int := { m | -3 < m ∧ m < 2 }
def N_def : Set Int := { n | -1 ≤ n ∧ n ≤ 3 }

theorem intersect_eq : M_def ∩ N_def = { -1, 0, 1 } := by
  sorry

end intersect_eq_l1367_136712


namespace jesse_rooms_l1367_136768

theorem jesse_rooms:
  ∀ (l w A n: ℕ), 
  l = 19 ∧ 
  w = 18 ∧ 
  A = 6840 ∧ 
  n = A / (l * w) → 
  n = 20 :=
by
  intros
  sorry

end jesse_rooms_l1367_136768


namespace op_exp_eq_l1367_136731

-- Define the operation * on natural numbers
def op (a b : ℕ) : ℕ := a ^ b

-- The theorem to be proven
theorem op_exp_eq (a b n : ℕ) : (op a b)^n = op a (b^n) := by
  sorry

end op_exp_eq_l1367_136731


namespace symmetric_sufficient_not_necessary_l1367_136742

theorem symmetric_sufficient_not_necessary (φ : Real) : 
    φ = - (Real.pi / 6) →
    ∃ f : Real → Real, (∀ x, f x = Real.sin (2 * x - φ)) ∧ 
    ∀ x, f (2 * (Real.pi / 6) - x) = f x :=
by
  sorry

end symmetric_sufficient_not_necessary_l1367_136742


namespace elberta_amount_l1367_136773

theorem elberta_amount (grannySmith_amount : ℝ) (Anjou_factor : ℝ) (extra_amount : ℝ) :
  grannySmith_amount = 45 →
  Anjou_factor = 1 / 4 →
  extra_amount = 4 →
  (extra_amount + Anjou_factor * grannySmith_amount) = 15.25 :=
by
  intros h_grannySmith h_AnjouFactor h_extraAmount
  sorry

end elberta_amount_l1367_136773


namespace iodine_atomic_weight_l1367_136700

noncomputable def atomic_weight_of_iodine : ℝ :=
  127.01

theorem iodine_atomic_weight
  (mw_AlI3 : ℝ := 408)
  (aw_Al : ℝ := 26.98)
  (formula_mw_AlI3 : mw_AlI3 = aw_Al + 3 * atomic_weight_of_iodine) :
  atomic_weight_of_iodine = 127.01 :=
by sorry

end iodine_atomic_weight_l1367_136700


namespace common_factor_l1367_136740

theorem common_factor (x y : ℝ) : 
  ∃ c : ℝ, c * (3 * x * y^2 - 4 * x^2 * y) = 6 * x^2 * y - 8 * x * y^2 ∧ c = 2 * x * y := 
by 
  sorry

end common_factor_l1367_136740


namespace magic_square_sum_l1367_136718

-- Definitions based on the conditions outlined in the problem
def magic_sum := 83
def a := 42
def b := 26
def c := 29
def e := 34
def d := 36

theorem magic_square_sum :
  d + e = 70 :=
by
  -- Proof is omitted as per instructions
  sorry

end magic_square_sum_l1367_136718


namespace combination_15_choose_3_l1367_136730

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l1367_136730


namespace combined_distance_20_birds_two_seasons_l1367_136781

theorem combined_distance_20_birds_two_seasons :
  let distance_jim_to_disney := 50
  let distance_disney_to_london := 60
  let number_of_birds := 20
  (number_of_birds * (distance_jim_to_disney + distance_disney_to_london)) = 2200 := by
  sorry

end combined_distance_20_birds_two_seasons_l1367_136781


namespace candidate_percentage_l1367_136795

theorem candidate_percentage (P : ℚ) (votes_cast : ℚ) (loss : ℚ)
  (h1 : votes_cast = 2000) 
  (h2 : loss = 640) 
  (h3 : (P / 100) * votes_cast + (P / 100) * votes_cast + loss = votes_cast) :
  P = 34 :=
by 
  sorry

end candidate_percentage_l1367_136795


namespace range_of_m_l1367_136791

theorem range_of_m (m : Real) :
  (∀ x y : Real, 0 < x ∧ x < y ∧ y < (π / 2) → 
    (m - 2 * Real.sin x) / Real.cos x > (m - 2 * Real.sin y) / Real.cos y) →
  m ≤ 2 := 
sorry

end range_of_m_l1367_136791


namespace total_weight_l1367_136734

def weight_of_blue_ball : ℝ := 6.0
def weight_of_brown_ball : ℝ := 3.12

theorem total_weight (_ : weight_of_blue_ball = 6.0) (_ : weight_of_brown_ball = 3.12) : 
  weight_of_blue_ball + weight_of_brown_ball = 9.12 :=
by
  sorry

end total_weight_l1367_136734


namespace books_price_arrangement_l1367_136757

theorem books_price_arrangement (c : ℝ) (prices : Fin 40 → ℝ)
  (h₁ : ∀ i : Fin 39, prices i.succ = prices i + 3)
  (h₂ : prices ⟨39, by norm_num⟩ = prices ⟨19, by norm_num⟩ + prices ⟨20, by norm_num⟩) :
  prices 20 = prices 19 + 3 := 
sorry

end books_price_arrangement_l1367_136757


namespace curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l1367_136751

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2

theorem curve_C1_general_equation (x y : ℝ) (α : ℝ) :
  (2 * Real.cos α = x) ∧ (Real.sqrt 2 * Real.sin α = y) →
  x^2 / 4 + y^2 / 2 = 1 :=
sorry

theorem curve_C2_cartesian_equation (ρ θ : ℝ) (x y : ℝ) :
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ polar_curve_C2 ρ θ →
  x + y = 6 :=
sorry

theorem minimum_distance_P1P2 (P1 P2 : ℝ × ℝ) (d : ℝ) :
  (∃ α, P1 = parametric_curve_C1 α) ∧ (∃ x y, P2 = (x, y) ∧ x + y = 6) →
  d = (3 * Real.sqrt 2 - Real.sqrt 3) :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l1367_136751


namespace face_opposite_to_A_is_D_l1367_136754

-- Definitions of faces
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Given conditions
def C_is_on_top : Face := C
def B_is_to_the_right_of_C : Face := B
def forms_cube (f1 f2 : Face) : Prop := -- Some property indicating that the faces are part of a folded cube
sorry

-- The theorem statement to prove that the face opposite to face A is D
theorem face_opposite_to_A_is_D (h1 : C_is_on_top = C) (h2 : B_is_to_the_right_of_C = B) (h3 : forms_cube A D)
    : ∃ f : Face, f = D := sorry

end face_opposite_to_A_is_D_l1367_136754


namespace x_mul_y_eq_4_l1367_136794

theorem x_mul_y_eq_4 (x y z w : ℝ) (hw_pos : w > 0) 
  (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) 
  (h4 : y = w) (h5 : z = 3) (h6 : w + w = w * w) : 
  x * y = 4 := by
  sorry

end x_mul_y_eq_4_l1367_136794


namespace cells_sequence_exists_l1367_136774

theorem cells_sequence_exists :
  ∃ (a : Fin 10 → ℚ), 
    a 0 = 9 ∧
    a 8 = 5 ∧
    (∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14) :=
sorry

end cells_sequence_exists_l1367_136774


namespace find_s_l1367_136771

variable {a b n r s : ℝ}

theorem find_s (h1 : Polynomial.aeval a (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h2 : Polynomial.aeval b (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h_ab : a * b = 6)
              (h_roots : Polynomial.aeval (a + 2/b) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0)
              (h_roots2 : Polynomial.aeval (b + 2/a) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0) :
  s = 32/3 := 
sorry

end find_s_l1367_136771


namespace total_skips_correct_l1367_136782

def bob_skip_rate := 12
def jim_skip_rate := 15
def sally_skip_rate := 18

def bob_rocks := 10
def jim_rocks := 8
def sally_rocks := 12

theorem total_skips_correct : 
  (bob_skip_rate * bob_rocks) + (jim_skip_rate * jim_rocks) + (sally_skip_rate * sally_rocks) = 456 := by
  sorry

end total_skips_correct_l1367_136782


namespace amplitude_combined_wave_l1367_136779

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t
noncomputable def amplitude : ℝ := 3 * Real.sqrt 5

theorem amplitude_combined_wave : ∀ t : ℝ, ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  intro t
  use amplitude
  exact sorry

end amplitude_combined_wave_l1367_136779


namespace compare_y_values_l1367_136723

theorem compare_y_values (y1 y2 : ℝ) 
  (hA : y1 = (-1)^2 - 4*(-1) - 3) 
  (hB : y2 = 1^2 - 4*1 - 3) : y1 > y2 :=
by
  sorry

end compare_y_values_l1367_136723


namespace range_of_t_l1367_136763

theorem range_of_t (a t : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  a * x^2 + t * y^2 ≥ (a * x + t * y)^2 ↔ 0 ≤ t ∧ t ≤ 1 - a :=
sorry

end range_of_t_l1367_136763


namespace Jake_weight_196_l1367_136792

def Jake_and_Sister : Prop :=
  ∃ (J S : ℕ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196)

theorem Jake_weight_196 : Jake_and_Sister :=
by
  sorry

end Jake_weight_196_l1367_136792


namespace factorial_product_trailing_zeros_l1367_136715

def countTrailingZerosInFactorialProduct : ℕ :=
  let countFactorsOfFive (n : ℕ) : ℕ := 
    (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) + (n / 78125) + (n / 390625) 
  List.range 100 -- Generates list [0, 1, ..., 99]
  |> List.map (fun k => countFactorsOfFive (k + 1)) -- Apply countFactorsOfFive to each k+1
  |> List.foldr (· + ·) 0 -- Sum all counts

theorem factorial_product_trailing_zeros : countTrailingZerosInFactorialProduct = 1124 := by
  sorry

end factorial_product_trailing_zeros_l1367_136715


namespace max_discount_rate_l1367_136755

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l1367_136755


namespace seating_capacity_for_ten_tables_in_two_rows_l1367_136777

-- Definitions based on the problem conditions
def seating_for_one_table : ℕ := 6

def seating_for_two_tables : ℕ := 10

def seating_for_three_tables : ℕ := 14

def additional_people_per_table : ℕ := 4

-- Calculating the seating capacity for n tables based on the pattern
def seating_capacity (n : ℕ) : ℕ :=
  if n = 1 then seating_for_one_table
  else seating_for_one_table + (n - 1) * additional_people_per_table

-- Proof statement without the proof
theorem seating_capacity_for_ten_tables_in_two_rows :
  (seating_capacity 5) * 2 = 44 :=
by sorry

end seating_capacity_for_ten_tables_in_two_rows_l1367_136777


namespace find_a_plus_b_l1367_136780

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b) (h2 : ∀ x, g x = 3 * x - 4) 
(h3 : ∀ x, g (f x) = 4 * x + 5) : a + b = 13 / 3 :=
sorry

end find_a_plus_b_l1367_136780


namespace max_good_pairs_1_to_30_l1367_136762

def is_good_pair (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def max_good_pairs_in_range (n : ℕ) : ℕ :=
  if n = 30 then 13 else 0

theorem max_good_pairs_1_to_30 : max_good_pairs_in_range 30 = 13 :=
by
  sorry

end max_good_pairs_1_to_30_l1367_136762


namespace jogger_distance_l1367_136709

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  -- Definition and proof would go here
  --
  sorry

end jogger_distance_l1367_136709


namespace find_angle_x_eq_38_l1367_136749

theorem find_angle_x_eq_38
  (angle_ACD angle_ECB angle_DCE : ℝ)
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  (h3 : angle_ACD + angle_ECB + angle_DCE = 180) :
  angle_DCE = 38 :=
by
  sorry

end find_angle_x_eq_38_l1367_136749
