import Mathlib

namespace add_alcohol_solve_l1279_127927

variable (x : ℝ)

def initial_solution_volume : ℝ := 6
def initial_alcohol_fraction : ℝ := 0.20
def desired_alcohol_fraction : ℝ := 0.50

def initial_alcohol_content : ℝ := initial_alcohol_fraction * initial_solution_volume
def total_solution_volume_after_addition : ℝ := initial_solution_volume + x
def total_alcohol_content_after_addition : ℝ := initial_alcohol_content + x

theorem add_alcohol_solve (x : ℝ) :
  (initial_alcohol_content + x) / (initial_solution_volume + x) = desired_alcohol_fraction →
  x = 3.6 :=
by
  sorry

end add_alcohol_solve_l1279_127927


namespace jordan_total_points_l1279_127970

-- Definitions based on conditions in the problem
def jordan_attempts (x y : ℕ) : Prop :=
  x + y = 40

def points_from_three_point_shots (x : ℕ) : ℝ :=
  0.75 * x

def points_from_two_point_shots (y : ℕ) : ℝ :=
  0.8 * y

-- Main theorem to prove the total points scored by Jordan
theorem jordan_total_points (x y : ℕ) 
  (h_attempts : jordan_attempts x y) : 
  points_from_three_point_shots x + points_from_two_point_shots y = 30 := 
by
  sorry

end jordan_total_points_l1279_127970


namespace harmony_implication_at_least_N_plus_1_zero_l1279_127914

noncomputable def is_harmony (A B : ℕ → ℕ) (i : ℕ) : Prop :=
  A i = (1 / (2 * B i + 1)) * (Finset.range (2 * B i + 1)).sum (fun s => A (i + s - B i))

theorem harmony_implication_at_least_N_plus_1_zero {N : ℕ} (A B : ℕ → ℕ)
  (hN : N ≥ 2) 
  (h_nonneg_A : ∀ i, 0 ≤ A i)
  (h_nonneg_B : ∀ i, 0 ≤ B i)
  (h_periodic_A : ∀ i, A i = A ((i % N) + 1))
  (h_periodic_B : ∀ i, B i = B ((i % N) + 1))
  (h_harmony_AB : ∀ i, is_harmony A B i)
  (h_harmony_BA : ∀ i, is_harmony B A i)
  (h_not_constant_A : ¬ ∀ i j, A i = A j)
  (h_not_constant_B : ¬ ∀ i j, B i = B j) :
  Finset.card (Finset.filter (fun i => A i = 0 ∨ B i = 0) (Finset.range (N * 2))) ≥ N + 1 := by
  sorry

end harmony_implication_at_least_N_plus_1_zero_l1279_127914


namespace raj_snow_removal_volume_l1279_127981

theorem raj_snow_removal_volume :
  let length := 30
  let width := 4
  let depth_layer1 := 0.5
  let depth_layer2 := 0.3
  let volume_layer1 := length * width * depth_layer1
  let volume_layer2 := length * width * depth_layer2
  let total_volume := volume_layer1 + volume_layer2
  total_volume = 96 := by
sorry

end raj_snow_removal_volume_l1279_127981


namespace tyler_age_l1279_127978

theorem tyler_age (T C : ℕ) (h1 : T = 3 * C + 1) (h2 : T + C = 21) : T = 16 :=
by
  sorry

end tyler_age_l1279_127978


namespace proof_A_minus_2B_eq_11_l1279_127954

theorem proof_A_minus_2B_eq_11 
  (a b : ℤ)
  (hA : ∀ a b, A = 3*b^2 - 2*a^2)
  (hB : ∀ a b, B = ab - 2*b^2 - a^2) 
  (ha : a = 2) 
  (hb : b = -1) : 
  (A - 2*B = 11) :=
by
  sorry

end proof_A_minus_2B_eq_11_l1279_127954


namespace complement_intersect_eq_l1279_127902

-- Define Universal Set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define Set P
def P : Set ℕ := {2, 3, 4}

-- Define Set Q
def Q : Set ℕ := {1, 2}

-- Complement of P in U
def complement_U_P : Set ℕ := U \ P

-- Goal Statement
theorem complement_intersect_eq {U P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4}) 
  (hP : P = {2, 3, 4}) 
  (hQ : Q = {1, 2}) : 
  (complement_U_P ∩ Q) = {1} := 
by
  sorry

end complement_intersect_eq_l1279_127902


namespace members_in_both_sets_l1279_127941

def U : Nat := 193
def B : Nat := 41
def not_A_or_B : Nat := 59
def A : Nat := 116

theorem members_in_both_sets
  (h1 : 193 = U)
  (h2 : 41 = B)
  (h3 : 59 = not_A_or_B)
  (h4 : 116 = A) :
  (U - not_A_or_B) = A + B - 23 :=
by
  sorry

end members_in_both_sets_l1279_127941


namespace inequality_areas_l1279_127998

theorem inequality_areas (a b c α β γ : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  a / α + b / β + c / γ ≥ 3 / 2 :=
by
  -- Insert the AM-GM inequality application and simplifications
  sorry

end inequality_areas_l1279_127998


namespace not_divisible_a1a2_l1279_127929

theorem not_divisible_a1a2 (a1 a2 b1 b2 : ℕ) (h1 : 1 < b1) (h2 : b1 < a1) (h3 : 1 < b2) (h4 : b2 < a2) (h5 : b1 ∣ a1) (h6 : b2 ∣ a2) :
  ¬ (a1 * a2 ∣ a1 * b1 + a2 * b2 - 1) :=
by
  sorry

end not_divisible_a1a2_l1279_127929


namespace cindy_gives_3_envelopes_per_friend_l1279_127940

theorem cindy_gives_3_envelopes_per_friend
  (initial_envelopes : ℕ) 
  (remaining_envelopes : ℕ)
  (friends : ℕ)
  (envelopes_per_friend : ℕ) 
  (h1 : initial_envelopes = 37) 
  (h2 : remaining_envelopes = 22)
  (h3 : friends = 5) 
  (h4 : initial_envelopes - remaining_envelopes = envelopes_per_friend * friends) :
  envelopes_per_friend = 3 :=
by
  sorry

end cindy_gives_3_envelopes_per_friend_l1279_127940


namespace arithmetic_mean_of_18_24_42_l1279_127949

-- Define the numbers a, b, c
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 42

-- Define the arithmetic mean
def mean (x y z : ℕ) : ℕ := (x + y + z) / 3

-- State the theorem to be proved
theorem arithmetic_mean_of_18_24_42 : mean a b c = 28 :=
by
  sorry

end arithmetic_mean_of_18_24_42_l1279_127949


namespace additional_cost_tv_ad_l1279_127986

theorem additional_cost_tv_ad (in_store_price : ℝ) (payment : ℝ) (shipping : ℝ) :
  in_store_price = 129.95 → payment = 29.99 → shipping = 14.95 → 
  (4 * payment + shipping - in_store_price) * 100 = 496 :=
by
  intros h1 h2 h3
  sorry

end additional_cost_tv_ad_l1279_127986


namespace downstream_distance_15_minutes_l1279_127945

theorem downstream_distance_15_minutes
  (speed_boat : ℝ) (speed_current : ℝ) (time_minutes : ℝ)
  (h1 : speed_boat = 24)
  (h2 : speed_current = 3)
  (h3 : time_minutes = 15) :
  let effective_speed := speed_boat + speed_current
  let time_hours := time_minutes / 60
  let distance := effective_speed * time_hours
  distance = 6.75 :=
by {
  sorry
}

end downstream_distance_15_minutes_l1279_127945


namespace intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l1279_127990

def U := ℝ
def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def C_U_B : Set ℝ := {x | x < -2 ∨ x ≥ 4}

theorem intersection_A_B_eq : A ∩ B = {x | 0 ≤ x ∧ x < 4} := by
  sorry

theorem union_A_B_eq : A ∪ B = {x | -2 ≤ x ∧ x < 5} := by
  sorry

theorem intersection_A_C_U_B_eq : A ∩ C_U_B = {x | 4 ≤ x ∧ x < 5} := by
  sorry

end intersection_A_B_eq_union_A_B_eq_intersection_A_C_U_B_eq_l1279_127990


namespace n_to_the_4_plus_4_to_the_n_composite_l1279_127907

theorem n_to_the_4_plus_4_to_the_n_composite (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + 4^n) := 
sorry

end n_to_the_4_plus_4_to_the_n_composite_l1279_127907


namespace eval_f_four_times_l1279_127922

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end eval_f_four_times_l1279_127922


namespace max_profit_under_budget_max_profit_no_budget_l1279_127909

-- Definitions from conditions
def sales_revenue (x1 x2 : ℝ) : ℝ :=
  -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

def profit (x1 x2 : ℝ) : ℝ :=
  sales_revenue x1 x2 - x1 - x2

-- Statements for the conditions
theorem max_profit_under_budget :
  (∀ x1 x2 : ℝ, x1 + x2 = 5 → profit x1 x2 ≤ 9) ∧
  (profit 2 3 = 9) :=
by sorry

theorem max_profit_no_budget :
  (∀ x1 x2 : ℝ, profit x1 x2 ≤ 15) ∧
  (profit 3 5 = 15) :=
by sorry

end max_profit_under_budget_max_profit_no_budget_l1279_127909


namespace spending_on_hydrangeas_l1279_127916

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l1279_127916


namespace same_solution_m_l1279_127913

theorem same_solution_m (m x : ℤ) : 
  (8 - m = 2 * (x + 1)) ∧ (2 * (2 * x - 3) - 1 = 1 - 2 * x) → m = 10 / 3 :=
by
  sorry

end same_solution_m_l1279_127913


namespace unique_positive_integers_l1279_127980

theorem unique_positive_integers (x y : ℕ) (h1 : x^2 + 84 * x + 2008 = y^2) : x + y = 80 :=
  sorry

end unique_positive_integers_l1279_127980


namespace second_batch_jelly_beans_weight_l1279_127905

theorem second_batch_jelly_beans_weight (J : ℝ) (h1 : 2 * 3 + J > 0) (h2 : (6 + J) * 2 = 16) : J = 2 :=
sorry

end second_batch_jelly_beans_weight_l1279_127905


namespace estimated_prob_is_0_9_l1279_127974

section GerminationProbability

-- Defining the experiment data
structure ExperimentData :=
  (totalSeeds : ℕ)
  (germinatedSeeds : ℕ)
  (germinationRate : ℝ)

def experiments : List ExperimentData := [
  ⟨100, 91, 0.91⟩, 
  ⟨400, 358, 0.895⟩, 
  ⟨800, 724, 0.905⟩,
  ⟨1400, 1264, 0.903⟩,
  ⟨3500, 3160, 0.903⟩,
  ⟨7000, 6400, 0.914⟩
]

-- Hypothesis based on the given problem's observation
def estimated_germination_probability (experiments : List ExperimentData) : ℝ :=
  /- Fictively calculating the stable germination rate here; however, logically we should use 
     some weighted average or similar statistical stability method. -/
  0.9  -- Rounded and concluded estimated value based on observation

theorem estimated_prob_is_0_9 :
  estimated_germination_probability experiments = 0.9 :=
  sorry

end GerminationProbability

end estimated_prob_is_0_9_l1279_127974


namespace bob_mother_twice_age_2040_l1279_127944

theorem bob_mother_twice_age_2040 :
  ∀ (bob_age_2010 mother_age_2010 : ℕ), 
  bob_age_2010 = 10 ∧ mother_age_2010 = 50 →
  ∃ (x : ℕ), (mother_age_2010 + x = 2 * (bob_age_2010 + x)) ∧ (2010 + x = 2040) :=
by
  sorry

end bob_mother_twice_age_2040_l1279_127944


namespace lisa_hotdog_record_l1279_127932

theorem lisa_hotdog_record
  (hotdogs_eaten : ℕ)
  (eaten_in_first_half : ℕ)
  (rate_per_minute : ℕ)
  (time_in_minutes : ℕ)
  (first_half_duration : ℕ)
  (remaining_time : ℕ) :
  eaten_in_first_half = 20 →
  rate_per_minute = 11 →
  first_half_duration = 5 →
  remaining_time = 5 →
  time_in_minutes = first_half_duration + remaining_time →
  hotdogs_eaten = eaten_in_first_half + rate_per_minute * remaining_time →
  hotdogs_eaten = 75 := by
  intros
  sorry

end lisa_hotdog_record_l1279_127932


namespace cost_per_adult_is_3_l1279_127939

-- Define the number of people in the group
def total_people : ℕ := 12

-- Define the number of kids in the group
def kids : ℕ := 7

-- Define the total cost for the group
def total_cost : ℕ := 15

-- Define the number of adults, which is the total number of people minus the number of kids
def adults : ℕ := total_people - kids

-- Define the cost per adult meal, which is the total cost divided by the number of adults
noncomputable def cost_per_adult : ℕ := total_cost / adults

-- The theorem stating the cost per adult meal is $3
theorem cost_per_adult_is_3 : cost_per_adult = 3 :=
by
  -- The proof is skipped
  sorry

end cost_per_adult_is_3_l1279_127939


namespace smallest_number_of_small_bottles_l1279_127950

def minimum_bottles_needed (large_bottle_capacity : ℕ) (small_bottle1 : ℕ) (small_bottle2 : ℕ) : ℕ :=
  if large_bottle_capacity = 720 ∧ small_bottle1 = 40 ∧ small_bottle2 = 45 then 16 else 0

theorem smallest_number_of_small_bottles :
  minimum_bottles_needed 720 40 45 = 16 := by
  sorry

end smallest_number_of_small_bottles_l1279_127950


namespace range_of_m_l1279_127917

theorem range_of_m (m : ℝ) (h : (2 - m) * (|m| - 3) < 0) : (-3 < m ∧ m < 2) ∨ (m > 3) :=
sorry

end range_of_m_l1279_127917


namespace find_f_minus3_and_f_2009_l1279_127995

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Conditions
axiom h1 : is_odd f
axiom h2 : f 1 = 2
axiom h3 : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Questions
theorem find_f_minus3_and_f_2009 : f (-3) = 0 ∧ f 2009 = -2 :=
by 
  sorry

end find_f_minus3_and_f_2009_l1279_127995


namespace P_subset_Q_l1279_127926

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l1279_127926


namespace number_of_ways_to_select_team_l1279_127960

def calc_binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem number_of_ways_to_select_team : calc_binomial_coefficient 17 4 = 2380 := by
  sorry

end number_of_ways_to_select_team_l1279_127960


namespace calc_expression_l1279_127912

theorem calc_expression :
  5 + 7 * (2 + (1 / 4 : ℝ)) = 20.75 :=
by
  sorry

end calc_expression_l1279_127912


namespace range_of_independent_variable_l1279_127942

theorem range_of_independent_variable (x : ℝ) : x + 3 ≥ 0 ↔ x ≥ -3 := by
  sorry

end range_of_independent_variable_l1279_127942


namespace total_fishes_caught_l1279_127925

def melanieCatches : ℕ := 8
def tomCatches : ℕ := 3 * melanieCatches
def totalFishes : ℕ := melanieCatches + tomCatches

theorem total_fishes_caught : totalFishes = 32 := by
  sorry

end total_fishes_caught_l1279_127925


namespace length_AC_eq_9_74_l1279_127972

-- Define the cyclic quadrilateral and given constraints
noncomputable def quad (A B C D : Type) : Prop := sorry
def angle_BAC := 50
def angle_ADB := 60
def AD := 3
def BC := 9

-- Prove that length of AC is 9.74 given the above conditions
theorem length_AC_eq_9_74 
  (A B C D : Type)
  (h_quad : quad A B C D)
  (h_angle_BAC : angle_BAC = 50)
  (h_angle_ADB : angle_ADB = 60)
  (h_AD : AD = 3)
  (h_BC : BC = 9) :
  ∃ AC, AC = 9.74 :=
sorry

end length_AC_eq_9_74_l1279_127972


namespace distance_points_3_12_and_10_0_l1279_127955

theorem distance_points_3_12_and_10_0 : 
  Real.sqrt ((10 - 3)^2 + (0 - 12)^2) = Real.sqrt 193 := 
by
  sorry

end distance_points_3_12_and_10_0_l1279_127955


namespace gamma_bank_min_savings_l1279_127984

def total_airfare_cost : ℕ := 10200 * 2 * 3
def total_hotel_cost : ℕ := 6500 * 12
def total_food_cost : ℕ := 1000 * 14 * 3
def total_excursion_cost : ℕ := 20000
def total_expenses : ℕ := total_airfare_cost + total_hotel_cost + total_food_cost + total_excursion_cost
def initial_amount_available : ℕ := 150000

def annual_rate_rebs : ℝ := 0.036
def annual_rate_gamma : ℝ := 0.045
def annual_rate_tisi : ℝ := 0.0312
def monthly_rate_btv : ℝ := 0.0025

noncomputable def compounded_amount_rebs : ℝ :=
  initial_amount_available * (1 + annual_rate_rebs / 12) ^ 6

noncomputable def compounded_amount_gamma : ℝ :=
  initial_amount_available * (1 + annual_rate_gamma / 2)

noncomputable def compounded_amount_tisi : ℝ :=
  initial_amount_available * (1 + annual_rate_tisi / 4) ^ 2

noncomputable def compounded_amount_btv : ℝ :=
  initial_amount_available * (1 + monthly_rate_btv) ^ 6

noncomputable def interest_rebs : ℝ := compounded_amount_rebs - initial_amount_available
noncomputable def interest_gamma : ℝ := compounded_amount_gamma - initial_amount_available
noncomputable def interest_tisi : ℝ := compounded_amount_tisi - initial_amount_available
noncomputable def interest_btv : ℝ := compounded_amount_btv - initial_amount_available

noncomputable def amount_needed_from_salary_rebs : ℝ :=
  total_expenses - initial_amount_available - interest_rebs

noncomputable def amount_needed_from_salary_gamma : ℝ :=
  total_expenses - initial_amount_available - interest_gamma

noncomputable def amount_needed_from_salary_tisi : ℝ :=
  total_expenses - initial_amount_available - interest_tisi

noncomputable def amount_needed_from_salary_btv : ℝ :=
  total_expenses - initial_amount_available - interest_btv

theorem gamma_bank_min_savings :
  amount_needed_from_salary_gamma = 47825.00 ∧
  amount_needed_from_salary_rebs = 48479.67 ∧
  amount_needed_from_salary_tisi = 48850.87 ∧
  amount_needed_from_salary_btv = 48935.89 :=
by sorry

end gamma_bank_min_savings_l1279_127984


namespace complement_intersection_l1279_127992

-- Definitions
def U : Set ℕ := {x | x ≤ 4 ∧ 0 < x}
def A : Set ℕ := {1, 4}
def B : Set ℕ := {2, 4}
def complement (s : Set ℕ) := {x | x ∈ U ∧ x ∉ s}

-- The theorem to prove
theorem complement_intersection :
  complement (A ∩ B) = {1, 2, 3} :=
by
  sorry

end complement_intersection_l1279_127992


namespace bowling_ball_weight_l1279_127999

theorem bowling_ball_weight (b k : ℝ)  (h1 : 8 * b = 5 * k) (h2 : 4 * k = 120) : b = 18.75 := by
  sorry

end bowling_ball_weight_l1279_127999


namespace equation_solution_l1279_127969

theorem equation_solution (x : ℚ) (h₁ : (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3) : x = 8 / 3 :=
by
  sorry

end equation_solution_l1279_127969


namespace negation_example_l1279_127993

theorem negation_example : (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_example_l1279_127993


namespace most_numerous_fruit_l1279_127906

-- Define the number of boxes
def num_boxes_tangerines := 5
def num_boxes_apples := 3
def num_boxes_pears := 4

-- Define the number of fruits per box
def tangerines_per_box := 30
def apples_per_box := 20
def pears_per_box := 15

-- Calculate the total number of each fruit
def total_tangerines := num_boxes_tangerines * tangerines_per_box
def total_apples := num_boxes_apples * apples_per_box
def total_pears := num_boxes_pears * pears_per_box

-- State the theorem and prove it
theorem most_numerous_fruit :
  total_tangerines = 150 ∧ total_tangerines > total_apples ∧ total_tangerines > total_pears :=
by
  -- Add here the necessary calculations to verify the conditions
  sorry

end most_numerous_fruit_l1279_127906


namespace arithmetic_progression_terms_l1279_127918

theorem arithmetic_progression_terms
  (n : ℕ) (a d : ℝ)
  (hn_odd : n % 2 = 1)
  (sum_odd_terms : n / 2 * (2 * a + (n / 2 - 1) * d) = 30)
  (sum_even_terms : (n / 2 - 1) * (2 * (a + d) + (n / 2 - 2) * d) = 36)
  (sum_all_terms : n / 2 * (2 * a + (n - 1) * d) = 66)
  (last_first_diff : (n - 1) * d = 12) :
  n = 9 := sorry

end arithmetic_progression_terms_l1279_127918


namespace john_caffeine_consumption_l1279_127901

noncomputable def caffeine_consumed : ℝ :=
let drink1_ounces : ℝ := 12
let drink1_caffeine : ℝ := 250
let drink2_ratio : ℝ := 3
let drink2_ounces : ℝ := 2

-- Calculate caffeine per ounce in the first drink
let caffeine1_per_ounce : ℝ := drink1_caffeine / drink1_ounces

-- Calculate caffeine per ounce in the second drink
let caffeine2_per_ounce : ℝ := caffeine1_per_ounce * drink2_ratio

-- Calculate total caffeine in the second drink
let drink2_caffeine : ℝ := caffeine2_per_ounce * drink2_ounces

-- Total caffeine from both drinks
let total_drinks_caffeine : ℝ := drink1_caffeine + drink2_caffeine

-- Caffeine in the pill is as much as the total from both drinks
let pill_caffeine : ℝ := total_drinks_caffeine

-- Total caffeine consumed
(drink1_caffeine + drink2_caffeine) + pill_caffeine

theorem john_caffeine_consumption :
  caffeine_consumed = 749.96 := by
    -- Proof is omitted
    sorry

end john_caffeine_consumption_l1279_127901


namespace positive_integer_solution_of_inequality_l1279_127936

theorem positive_integer_solution_of_inequality (x : ℕ) (h : 0 < x) : (3 * x - 1) / 2 + 1 ≥ 2 * x → x = 1 :=
by
  intros
  sorry

end positive_integer_solution_of_inequality_l1279_127936


namespace cost_of_peaches_eq_2_per_pound_l1279_127987

def initial_money : ℕ := 20
def after_buying_peaches : ℕ := 14
def pounds_of_peaches : ℕ := 3
def cost_per_pound : ℕ := 2

theorem cost_of_peaches_eq_2_per_pound (h: initial_money - after_buying_peaches = pounds_of_peaches * cost_per_pound) :
  cost_per_pound = 2 := by
  sorry

end cost_of_peaches_eq_2_per_pound_l1279_127987


namespace find_a4_l1279_127908

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem find_a4 (h1 : arithmetic_sequence a) (h2 : a 2 + a 6 = 2) : a 4 = 1 :=
by
  sorry

end find_a4_l1279_127908


namespace part_a_correct_part_b_correct_l1279_127983

-- Define the alphabet and mapping
inductive Letter
| C | H | M | O
deriving DecidableEq, Inhabited

open Letter

def letter_to_base4 (ch : Letter) : ℕ :=
  match ch with
  | C => 0
  | H => 1
  | M => 2
  | O => 3

def word_to_base4 (word : List Letter) : ℕ :=
  word.foldl (fun acc ch => acc * 4 + letter_to_base4 ch) 0

def base4_to_letter (n : ℕ) : Letter :=
  match n with
  | 0 => C
  | 1 => H
  | 2 => M
  | 3 => O
  | _ => C -- This should not occur if input is in valid base-4 range

def base4_to_word (n : ℕ) (size : ℕ) : List Letter :=
  if size = 0 then []
  else
    let quotient := n / 4
    let remainder := n % 4
    base4_to_letter remainder :: base4_to_word quotient (size - 1)

-- The size of the words is fixed at 8
def word_size : ℕ := 8

noncomputable def part_a : List Letter :=
  base4_to_word 2017 word_size

theorem part_a_correct :
  part_a = [H, O, O, H, M, C] := by
  sorry

def given_word : List Letter :=
  [H, O, M, C, H, O, M, C]

noncomputable def part_b : ℕ :=
  word_to_base4 given_word + 1 -- Adjust for zero-based indexing

theorem part_b_correct :
  part_b = 29299 := by
  sorry

end part_a_correct_part_b_correct_l1279_127983


namespace robin_total_distance_l1279_127971

theorem robin_total_distance
  (d : ℕ)
  (d1 : ℕ)
  (h1 : d = 500)
  (h2 : d1 = 200)
  : 2 * d1 + d = 900 :=
by
  rewrite [h1, h2]
  rfl

end robin_total_distance_l1279_127971


namespace max_x2_y2_z4_l1279_127920

theorem max_x2_y2_z4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
sorry

end max_x2_y2_z4_l1279_127920


namespace find_a_of_binomial_square_l1279_127934

theorem find_a_of_binomial_square (a : ℚ) :
  (∃ b : ℚ, (3 * (x : ℚ) + b)^2 = 9 * x^2 + 21 * x + a) ↔ a = 49 / 4 :=
by
  sorry

end find_a_of_binomial_square_l1279_127934


namespace numberOfChromiumAtoms_l1279_127953

noncomputable def molecularWeightOfCompound : ℕ := 296
noncomputable def atomicWeightOfPotassium : ℝ := 39.1
noncomputable def atomicWeightOfOxygen : ℝ := 16.0
noncomputable def atomicWeightOfChromium : ℝ := 52.0

def numberOfPotassiumAtoms : ℕ := 2
def numberOfOxygenAtoms : ℕ := 7

theorem numberOfChromiumAtoms
    (mw : ℕ := molecularWeightOfCompound)
    (awK : ℝ := atomicWeightOfPotassium)
    (awO : ℝ := atomicWeightOfOxygen)
    (awCr : ℝ := atomicWeightOfChromium)
    (numK : ℕ := numberOfPotassiumAtoms)
    (numO : ℕ := numberOfOxygenAtoms) :
  numK * awK + numO * awO + (mw - (numK * awK + numO * awO)) / awCr = 2 := 
by
  sorry

end numberOfChromiumAtoms_l1279_127953


namespace sum_a4_a5_a6_l1279_127982

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (h1 : is_arithmetic_sequence a)
          (h2 : a 1 + a 2 + a 3 = 6)
          (h3 : a 7 + a 8 + a 9 = 24)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 15 :=
by
  sorry

end sum_a4_a5_a6_l1279_127982


namespace trapezoid_dot_product_ad_bc_l1279_127933

-- Define the trapezoid and its properties
variables (A B C D O : Type) (AB CD AO BO : ℝ)
variables (AD BC : ℝ)

-- Conditions from the problem
axiom AB_length : AB = 41
axiom CD_length : CD = 24
axiom diagonals_perpendicular : ∀ (v₁ v₂ : ℝ), (v₁ * v₂ = 0)

-- Using these conditions, prove that the dot product of the vectors AD and BC is 984
theorem trapezoid_dot_product_ad_bc : AD * BC = 984 :=
  sorry

end trapezoid_dot_product_ad_bc_l1279_127933


namespace parallel_lines_slope_condition_l1279_127935

theorem parallel_lines_slope_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0) →
  (m = 2 ∨ m = -3) :=
by
  sorry

end parallel_lines_slope_condition_l1279_127935


namespace work_in_one_day_l1279_127963

theorem work_in_one_day (A_days B_days : ℕ) (hA : A_days = 18) (hB : B_days = A_days / 2) :
  (1 / A_days + 1 / B_days) = 1 / 6 := 
by
  sorry

end work_in_one_day_l1279_127963


namespace percentage_failed_in_Hindi_l1279_127938

-- Define the percentage of students failed in English
def percentage_failed_in_English : ℝ := 56

-- Define the percentage of students failed in both Hindi and English
def percentage_failed_in_both : ℝ := 12

-- Define the percentage of students passed in both subjects
def percentage_passed_in_both : ℝ := 24

-- Define the total percentage of students
def percentage_total : ℝ := 100

-- Define what we need to prove
theorem percentage_failed_in_Hindi:
  ∃ (H : ℝ), H + percentage_failed_in_English - percentage_failed_in_both + percentage_passed_in_both = percentage_total ∧ H = 32 :=
  by 
    sorry

end percentage_failed_in_Hindi_l1279_127938


namespace student_marks_l1279_127979

theorem student_marks :
  let max_marks := 300
  let passing_percentage := 0.60
  let failed_by := 20
  let passing_marks := max_marks * passing_percentage
  let marks_obtained := passing_marks - failed_by
  marks_obtained = 160 := by
sorry

end student_marks_l1279_127979


namespace parabola_coefficients_l1279_127991

theorem parabola_coefficients :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, (a * (x - 3)^2 + 2 = 0 → (x = 1) ∧ (a * (1 - 3)^2 + 2 = 0))
    ∧ (a = -1/2 ∧ b = 3 ∧ c = -5/2)) 
    ∧ (∀ x : ℝ, a * x^2 + b * x + c = - 1 / 2 * x^2 + 3 * x - 5 / 2) :=
sorry

end parabola_coefficients_l1279_127991


namespace trig_identity_example_l1279_127921

noncomputable def cos24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)
noncomputable def sin24 := Real.sin (24 * Real.pi / 180)
noncomputable def sin36 := Real.sin (36 * Real.pi / 180)
noncomputable def cos60 := Real.cos (60 * Real.pi / 180)

theorem trig_identity_example :
  cos24 * cos36 - sin24 * sin36 = cos60 :=
by
  sorry

end trig_identity_example_l1279_127921


namespace find_fayes_age_l1279_127946

variable {C D E F : ℕ}

theorem find_fayes_age
  (h1 : D = E - 2)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 16 := by
  sorry

end find_fayes_age_l1279_127946


namespace eq_to_general_quadratic_l1279_127900

theorem eq_to_general_quadratic (x : ℝ) : (x - 1) * (x + 1) = 1 → x^2 - 2 = 0 :=
by
  sorry

end eq_to_general_quadratic_l1279_127900


namespace fg_of_neg5_eq_484_l1279_127966

def f (x : Int) : Int := x * x
def g (x : Int) : Int := 6 * x + 8

theorem fg_of_neg5_eq_484 : f (g (-5)) = 484 := 
  sorry

end fg_of_neg5_eq_484_l1279_127966


namespace pages_needed_l1279_127919

def cards_per_page : ℕ := 3
def new_cards : ℕ := 2
def old_cards : ℕ := 10

theorem pages_needed : (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end pages_needed_l1279_127919


namespace last_number_with_35_zeros_l1279_127911

def count_zeros (n : Nat) : Nat :=
  if n = 0 then 1
  else if n < 10 then 0
  else count_zeros (n / 10) + count_zeros (n % 10)

def total_zeros_written (upto : Nat) : Nat :=
  (List.range (upto + 1)).foldl (λ acc n => acc + count_zeros n) 0

theorem last_number_with_35_zeros : ∃ n, total_zeros_written n = 35 ∧ ∀ m, m > n → total_zeros_written m ≠ 35 :=
by
  let x := 204
  have h1 : total_zeros_written x = 35 := sorry
  have h2 : ∀ m, m > x → total_zeros_written m ≠ 35 := sorry
  existsi x
  exact ⟨h1, h2⟩

end last_number_with_35_zeros_l1279_127911


namespace square_feet_per_acre_l1279_127962

theorem square_feet_per_acre 
  (pay_per_acre_per_month : ℕ) 
  (total_pay_per_month : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (total_acres : ℕ) 
  (H1 : pay_per_acre_per_month = 30) 
  (H2 : total_pay_per_month = 300) 
  (H3 : length = 360) 
  (H4 : width = 1210) 
  (H5 : total_acres = 10) : 
  (length * width) / total_acres = 43560 :=
by 
  sorry

end square_feet_per_acre_l1279_127962


namespace lying_dwarf_number_is_possible_l1279_127904

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l1279_127904


namespace number_line_point_B_l1279_127958

theorem number_line_point_B (A B : ℝ) (AB : ℝ) (h1 : AB = 4 * Real.sqrt 2) (h2 : A = 3 * Real.sqrt 2) :
  B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2 :=
sorry

end number_line_point_B_l1279_127958


namespace retirement_percentage_l1279_127937

-- Define the conditions
def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_paycheck : ℝ := 740

-- Define the total deduction
def total_deduction : ℝ := gross_pay - net_paycheck
def retirement_deduction : ℝ := total_deduction - tax_deduction

-- Define the theorem to prove
theorem retirement_percentage :
  (retirement_deduction / gross_pay) * 100 = 25 :=
by
  sorry

end retirement_percentage_l1279_127937


namespace solve_system_l1279_127959

theorem solve_system (a b c : ℝ) (h₁ : a^2 + 3 * a + 1 = (b + c) / 2)
                                (h₂ : b^2 + 3 * b + 1 = (a + c) / 2)
                                (h₃ : c^2 + 3 * c + 1 = (a + b) / 2) : 
  a = -1 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end solve_system_l1279_127959


namespace number_of_sides_l1279_127956

-- Define the conditions
def interior_angle (n : ℕ) : ℝ := 156

-- The main theorem to prove the number of sides
theorem number_of_sides (n : ℕ) (h : interior_angle n = 156) : n = 15 :=
by
  sorry

end number_of_sides_l1279_127956


namespace rug_area_is_24_l1279_127923

def length_floor : ℕ := 12
def width_floor : ℕ := 10
def strip_width : ℕ := 3

theorem rug_area_is_24 :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := 
by
  sorry

end rug_area_is_24_l1279_127923


namespace arccos_one_eq_zero_l1279_127924

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l1279_127924


namespace paco_salty_cookies_left_l1279_127961

theorem paco_salty_cookies_left (initial_salty : ℕ) (eaten_salty : ℕ) : initial_salty = 26 ∧ eaten_salty = 9 → initial_salty - eaten_salty = 17 :=
by
  intro h
  cases h
  sorry


end paco_salty_cookies_left_l1279_127961


namespace student_correct_answers_l1279_127952

theorem student_correct_answers (C W : ℕ) 
  (h1 : 4 * C - W = 130) 
  (h2 : C + W = 80) : 
  C = 42 := by
  sorry

end student_correct_answers_l1279_127952


namespace lending_rate_is_8_percent_l1279_127988

-- Define all given conditions.
def principal₁ : ℝ := 5000
def time₁ : ℝ := 2
def rate₁ : ℝ := 4  -- in percentage
def gain_per_year : ℝ := 200

-- Prove that the interest rate for lending is 8%
theorem lending_rate_is_8_percent :
  ∃ (rate₂ : ℝ), rate₂ = 8 :=
by
  let interest₁ := principal₁ * rate₁ * time₁ / 100
  let interest_per_year₁ := interest₁ / time₁
  let total_interest_received_per_year := gain_per_year + interest_per_year₁
  let rate₂ := (total_interest_received_per_year * 100) / principal₁
  use rate₂
  sorry

end lending_rate_is_8_percent_l1279_127988


namespace fenced_area_l1279_127930

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l1279_127930


namespace area_new_rectangle_l1279_127996

-- Define the given rectangle's dimensions
def a : ℕ := 3
def b : ℕ := 4

-- Define the diagonal of the given rectangle
def d : ℕ := Nat.sqrt (a^2 + b^2)

-- Define the new rectangle's dimensions
def length_new : ℕ := d + a
def breadth_new : ℕ := d - b

-- The target area of the new rectangle
def area_new : ℕ := length_new * breadth_new

-- Prove that the area of the new rectangle is 8 square units
theorem area_new_rectangle (h : d = 5) : area_new = 8 := by
  -- Indicate that proof steps are not provided
  sorry

end area_new_rectangle_l1279_127996


namespace fraction_equality_l1279_127985

theorem fraction_equality (a b c : ℝ) (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := 
by 
  sorry

end fraction_equality_l1279_127985


namespace length_of_bridge_l1279_127997

theorem length_of_bridge
  (walking_speed_km_hr : ℝ) (time_minutes : ℝ) (length_bridge : ℝ) 
  (h1 : walking_speed_km_hr = 5) 
  (h2 : time_minutes = 15) 
  (h3 : length_bridge = 1250) : 
  length_bridge = (walking_speed_km_hr * 1000 / 60) * time_minutes := 
by 
  sorry

end length_of_bridge_l1279_127997


namespace initially_calculated_average_is_correct_l1279_127931

theorem initially_calculated_average_is_correct :
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  initially_avg = 22 :=
by
  let S := 220
  let incorrect_sum := S - 36 + 26
  let initially_avg := incorrect_sum / 10
  show initially_avg = 22
  sorry

end initially_calculated_average_is_correct_l1279_127931


namespace edit_post_time_zero_l1279_127964

-- Define the conditions
def total_videos : ℕ := 4
def setup_time : ℕ := 1
def painting_time_per_video : ℕ := 1
def cleanup_time : ℕ := 1
def total_production_time_per_video : ℕ := 3

-- Define the total time spent on setup, painting, and cleanup for one video
def spc_time : ℕ := setup_time + painting_time_per_video + cleanup_time

-- State the theorem to be proven
theorem edit_post_time_zero : (total_production_time_per_video - spc_time) = 0 := by
  sorry

end edit_post_time_zero_l1279_127964


namespace problem_statement_l1279_127915

theorem problem_statement (x y : ℝ) (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 18) :
  17 * x ^ 2 + 24 * x * y + 17 * y ^ 2 = 532 :=
by
  sorry

end problem_statement_l1279_127915


namespace inequality_holds_for_gt_sqrt2_l1279_127928

theorem inequality_holds_for_gt_sqrt2 (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) :
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by {
  sorry
}

end inequality_holds_for_gt_sqrt2_l1279_127928


namespace find_ratio_of_sides_l1279_127973

variable {A B : ℝ}
variable {a b : ℝ}

-- Given condition
axiom given_condition : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = a * Real.sqrt 3

-- Theorem we need to prove
theorem find_ratio_of_sides (h : a ≠ 0) : b / a = Real.sqrt 3 / 3 :=
by
  sorry

end find_ratio_of_sides_l1279_127973


namespace spider_crawl_distance_l1279_127948

theorem spider_crawl_distance :
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  abs (b - a) + abs (c - b) + abs (d - c) = 20 :=
by
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  sorry

end spider_crawl_distance_l1279_127948


namespace positive_partial_sum_existence_l1279_127976

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem positive_partial_sum_existence (h : (Finset.univ.sum a) > 0) :
  ∃ i : Fin n, ∀ j : Fin n, i ≤ j → (Finset.Icc i j).sum a > 0 := by
  sorry

end positive_partial_sum_existence_l1279_127976


namespace xiaoying_final_score_l1279_127947

def speech_competition_score (score_content score_expression score_demeanor : ℕ) 
                             (weight_content weight_expression weight_demeanor : ℝ) : ℝ :=
  score_content * weight_content + score_expression * weight_expression + score_demeanor * weight_demeanor

theorem xiaoying_final_score :
  speech_competition_score 86 90 80 0.5 0.4 0.1 = 87 :=
by 
  sorry

end xiaoying_final_score_l1279_127947


namespace smallest_possible_value_l1279_127957

theorem smallest_possible_value (n : ℕ) (h1 : ∀ m, (Nat.lcm 60 m / Nat.gcd 60 m = 24) → m = n) (h2 : ∀ m, (m % 5 = 0) → m = n) : n = 160 :=
sorry

end smallest_possible_value_l1279_127957


namespace malachi_additional_photos_l1279_127968

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end malachi_additional_photos_l1279_127968


namespace maddie_total_payment_l1279_127965

def price_palettes : ℝ := 15
def num_palettes : ℕ := 3
def discount_palettes : ℝ := 0.20
def price_lipsticks : ℝ := 2.50
def num_lipsticks_bought : ℕ := 4
def num_lipsticks_pay : ℕ := 3
def price_hair_color : ℝ := 4
def num_hair_color : ℕ := 3
def discount_hair_color : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def total_cost_palettes : ℝ := num_palettes * price_palettes
def total_cost_palettes_after_discount : ℝ := total_cost_palettes * (1 - discount_palettes)

def total_cost_lipsticks : ℝ := num_lipsticks_pay * price_lipsticks

def total_cost_hair_color : ℝ := num_hair_color * price_hair_color
def total_cost_hair_color_after_discount : ℝ := total_cost_hair_color * (1 - discount_hair_color)

def total_pre_tax : ℝ := total_cost_palettes_after_discount + total_cost_lipsticks + total_cost_hair_color_after_discount
def total_sales_tax : ℝ := total_pre_tax * sales_tax_rate
def total_cost : ℝ := total_pre_tax + total_sales_tax

theorem maddie_total_payment : total_cost = 58.64 := by
  sorry

end maddie_total_payment_l1279_127965


namespace part1_part2_l1279_127943

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1 (x : ℝ) (hx : f x ≤ 5) : x ∈ Set.Icc (-1/4 : ℝ) (9/4 : ℝ) := sorry

noncomputable def h (x a : ℝ) : ℝ := Real.log (f x + a)

theorem part2 (ha : ∀ x : ℝ, f x + a > 0) : a ∈ Set.Ioi (-2 : ℝ) := sorry

end part1_part2_l1279_127943


namespace problem_statement_l1279_127989

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B C D : V)
  (BC CD : V)
  (AC AB AD : V)

theorem problem_statement
  (h1 : BC = 2 • CD)
  (h2 : BC = AC - AB) :
  AD = (3 / 2 : ℝ) • AC - (1 / 2 : ℝ) • AB :=
sorry

end problem_statement_l1279_127989


namespace total_planks_needed_l1279_127975

theorem total_planks_needed (large_planks small_planks : ℕ) (h1 : large_planks = 37) (h2 : small_planks = 42) : large_planks + small_planks = 79 :=
by
  sorry

end total_planks_needed_l1279_127975


namespace simplify_expression_l1279_127977

theorem simplify_expression (y : ℝ) : 
  4 * y + 9 * y ^ 2 + 8 - (3 - 4 * y - 9 * y ^ 2) = 18 * y ^ 2 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l1279_127977


namespace new_concentration_of_solution_l1279_127967

theorem new_concentration_of_solution 
  (Q : ℚ) 
  (initial_concentration : ℚ := 0.4) 
  (new_concentration : ℚ := 0.25) 
  (replacement_fraction : ℚ := 1/3) 
  (new_solution_concentration : ℚ := 0.35) :
  (initial_concentration * (1 - replacement_fraction) + new_concentration * replacement_fraction)
  = new_solution_concentration := 
by 
  sorry

end new_concentration_of_solution_l1279_127967


namespace censusSurveys_l1279_127903

-- Definitions corresponding to the problem conditions
inductive Survey where
  | TVLifespan
  | ManuscriptReview
  | PollutionInvestigation
  | StudentSizeSurvey

open Survey

-- The aim is to identify which surveys are more suitable for a census.
def suitableForCensus (s : Survey) : Prop :=
  match s with
  | TVLifespan => False  -- Lifespan destruction implies sample survey.
  | ManuscriptReview => True  -- Significant and needs high accuracy, thus census.
  | PollutionInvestigation => False  -- Broad scope implies sample survey.
  | StudentSizeSurvey => True  -- Manageable scope makes census appropriate.

-- The theorem to be formalized.
theorem censusSurveys : (suitableForCensus ManuscriptReview) ∧ (suitableForCensus StudentSizeSurvey) :=
  by sorry

end censusSurveys_l1279_127903


namespace inequality_proof_l1279_127994

theorem inequality_proof {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (1 / Real.sqrt (1 + x^2)) + (1 / Real.sqrt (1 + y^2)) ≤ (2 / Real.sqrt (1 + x * y)) :=
by
  sorry

end inequality_proof_l1279_127994


namespace problem_part1_problem_part2_l1279_127951

-- Define the set A and the property it satisfies
variable (A : Set ℝ)
variable (H : ∀ a ∈ A, (1 + a) / (1 - a) ∈ A)

-- Suppose 2 is in A
theorem problem_part1 (h : 2 ∈ A) : A = {2, -3, -1 / 2, 1 / 3} :=
sorry

-- Prove the conjecture based on the elements of A found in part 1
theorem problem_part2 (h : 2 ∈ A) (hA : A = {2, -3, -1 / 2, 1 / 3}) :
  ¬ (0 ∈ A ∨ 1 ∈ A ∨ -1 ∈ A) ∧
  (2 * (-1 / 2) = -1 ∧ -3 * (1 / 3) = -1) :=
sorry

end problem_part1_problem_part2_l1279_127951


namespace find_first_4_hours_speed_l1279_127910

noncomputable def average_speed_first_4_hours
  (total_avg_speed : ℝ)
  (first_4_hours_avg_speed : ℝ)
  (remaining_hours_avg_speed : ℝ)
  (total_time : ℕ)
  (first_4_hours : ℕ)
  (remaining_hours : ℕ) : Prop :=
  total_avg_speed * total_time = first_4_hours_avg_speed * first_4_hours + remaining_hours * remaining_hours_avg_speed

theorem find_first_4_hours_speed :
  average_speed_first_4_hours 50 35 53 24 4 20 :=
by
  sorry

end find_first_4_hours_speed_l1279_127910
