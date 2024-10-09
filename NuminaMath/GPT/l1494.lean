import Mathlib

namespace first_woman_hours_l1494_149419

-- Definitions and conditions
variables (W k y t η : ℝ)
variables (work_rate : k * y * 45 = W)
variables (total_work : W = k * (t * ((y-1) * y) / 2 + y * η))
variables (first_vs_last : (y-1) * t + η = 5 * η)

-- The goal to prove
theorem first_woman_hours :
  (y - 1) * t + η = 75 := 
by
  sorry

end first_woman_hours_l1494_149419


namespace length_squared_of_segment_CD_is_196_l1494_149425

theorem length_squared_of_segment_CD_is_196 :
  ∃ (C D : ℝ × ℝ), 
    (C.2 = 3 * C.1 ^ 2 + 6 * C.1 - 2) ∧
    (D.2 = 3 * (2 - C.1) ^ 2 + 6 * (2 - C.1) - 2) ∧
    (1 : ℝ) = (C.1 + D.1) / 2 ∧
    (0 : ℝ) = (C.2 + D.2) / 2 ∧
    ((C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 = 196) :=
by
  -- The proof would go here
  sorry

end length_squared_of_segment_CD_is_196_l1494_149425


namespace probability_white_given_popped_l1494_149404

theorem probability_white_given_popped :
  let P_white := 3 / 5
  let P_yellow := 2 / 5
  let P_popped_given_white := 2 / 5
  let P_popped_given_yellow := 4 / 5
  let P_white_and_popped := P_white * P_popped_given_white
  let P_yellow_and_popped := P_yellow * P_popped_given_yellow
  let P_popped := P_white_and_popped + P_yellow_and_popped
  let P_white_given_popped := P_white_and_popped / P_popped
  P_white_given_popped = 3 / 7 :=
by sorry

end probability_white_given_popped_l1494_149404


namespace solve_for_t_l1494_149460

theorem solve_for_t (s t : ℤ) (h1 : 11 * s + 7 * t = 160) (h2 : s = 2 * t + 4) : t = 4 :=
by
  sorry

end solve_for_t_l1494_149460


namespace Alyssa_spending_correct_l1494_149479

def cost_per_game : ℕ := 20

def last_year_in_person_games : ℕ := 13
def this_year_in_person_games : ℕ := 11
def this_year_streaming_subscription : ℕ := 120
def next_year_in_person_games : ℕ := 15
def next_year_streaming_subscription : ℕ := 150
def friends_count : ℕ := 2
def friends_join_games : ℕ := 5

def Alyssa_total_spending : ℕ :=
  (last_year_in_person_games * cost_per_game) +
  (this_year_in_person_games * cost_per_game) + this_year_streaming_subscription +
  (next_year_in_person_games * cost_per_game) + next_year_streaming_subscription -
  (friends_join_games * friends_count * cost_per_game)

theorem Alyssa_spending_correct : Alyssa_total_spending = 850 := by
  sorry

end Alyssa_spending_correct_l1494_149479


namespace question1_question2_l1494_149417

variables (θ : ℝ)

-- Condition: tan θ = 2
def tan_theta_eq : Prop := Real.tan θ = 2

-- Question 1: Prove (4 * sin θ - 2 * cos θ) / (3 * sin θ + 5 * cos θ) = 6 / 11
theorem question1 (h : tan_theta_eq θ) : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11 :=
by
  sorry

-- Question 2: Prove 1 - 4 * sin θ * cos θ + 2 * cos² θ = -1 / 5
theorem question2 (h : tan_theta_eq θ) : 1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1 / 5 :=
by
  sorry

end question1_question2_l1494_149417


namespace scaled_multiplication_l1494_149471

theorem scaled_multiplication (h : 213 * 16 = 3408) : 0.016 * 2.13 = 0.03408 :=
by
  sorry

end scaled_multiplication_l1494_149471


namespace solve_abs_eq_2x_plus_1_l1494_149435

theorem solve_abs_eq_2x_plus_1 (x : ℝ) (h : |x| = 2 * x + 1) : x = -1 / 3 :=
by 
  sorry

end solve_abs_eq_2x_plus_1_l1494_149435


namespace real_y_iff_x_l1494_149428

open Real

-- Definitions based on the conditions
def quadratic_eq (y x : ℝ) : ℝ := 9 * y^2 - 3 * x * y + x + 8

-- The main theorem to prove
theorem real_y_iff_x (x : ℝ) : (∃ y : ℝ, quadratic_eq y x = 0) ↔ x ≤ -4 ∨ x ≥ 8 := 
sorry

end real_y_iff_x_l1494_149428


namespace inequality_holds_for_all_x_iff_l1494_149405

theorem inequality_holds_for_all_x_iff (m : ℝ) :
  (∀ (x : ℝ), m * x^2 + m * x - 4 < 2 * x^2 + 2 * x - 1) ↔ -10 < m ∧ m ≤ 2 :=
by
  sorry

end inequality_holds_for_all_x_iff_l1494_149405


namespace john_sleep_total_hours_l1494_149408

-- Defining the conditions provided in the problem statement
def days_with_3_hours : ℕ := 2
def sleep_per_day_3_hours : ℕ := 3
def remaining_days : ℕ := 7 - days_with_3_hours
def recommended_sleep : ℕ := 8
def percentage_sleep : ℝ := 0.6

-- Expressing the proof problem statement
theorem john_sleep_total_hours :
  (days_with_3_hours * sleep_per_day_3_hours
  + remaining_days * (percentage_sleep * recommended_sleep)) = 30 := by
  sorry

end john_sleep_total_hours_l1494_149408


namespace combination_8_choose_2_l1494_149482

theorem combination_8_choose_2 : Nat.choose 8 2 = 28 := sorry

end combination_8_choose_2_l1494_149482


namespace nadine_total_cleaning_time_l1494_149477

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l1494_149477


namespace find_X_l1494_149438

theorem find_X (X : ℝ) (h : 45 * 8 = 0.40 * X) : X = 900 :=
sorry

end find_X_l1494_149438


namespace condition_inequality_l1494_149476

theorem condition_inequality (x y : ℝ) :
  (¬ (x ≤ y → |x| ≤ |y|)) ∧ (¬ (|x| ≤ |y| → x ≤ y)) :=
by
  sorry

end condition_inequality_l1494_149476


namespace mutually_exclusive_A_B_head_l1494_149498

variables (A_head B_head B_end : Prop)

def mut_exclusive (P Q : Prop) : Prop := ¬(P ∧ Q)

theorem mutually_exclusive_A_B_head (A_head B_head : Prop) :
  mut_exclusive A_head B_head :=
sorry

end mutually_exclusive_A_B_head_l1494_149498


namespace frisbee_total_distance_l1494_149401

-- Definitions for the conditions
def bess_initial_distance : ℝ := 20
def bess_throws : ℕ := 4
def bess_reduction : ℝ := 0.90
def holly_initial_distance : ℝ := 8
def holly_throws : ℕ := 5
def holly_reduction : ℝ := 0.95

-- Function to calculate the total distance for Bess
def total_distance_bess : ℝ :=
  let distances := List.range bess_throws |>.map (λ i => bess_initial_distance * bess_reduction ^ i)
  (distances.sum) * 2

-- Function to calculate the total distance for Holly
def total_distance_holly : ℝ :=
  let distances := List.range holly_throws |>.map (λ i => holly_initial_distance * holly_reduction ^ i)
  distances.sum

-- Proof statement
theorem frisbee_total_distance : 
  total_distance_bess + total_distance_holly = 173.76 :=
by
  sorry

end frisbee_total_distance_l1494_149401


namespace Winnie_the_Pooh_honey_consumption_l1494_149414

theorem Winnie_the_Pooh_honey_consumption (W0 W1 W2 W3 W4 : ℝ) (pot_empty : ℝ) 
  (h1 : W1 = W0 / 2)
  (h2 : W2 = W1 / 2)
  (h3 : W3 = W2 / 2)
  (h4 : W4 = W3 / 2)
  (h5 : W4 = 200)
  (h6 : pot_empty = 200) : 
  W0 - 200 = 3000 := by
  sorry

end Winnie_the_Pooh_honey_consumption_l1494_149414


namespace new_paint_intensity_l1494_149467

theorem new_paint_intensity (V : ℝ) (h1 : V > 0) :
    let initial_intensity := 0.5
    let replaced_fraction := 0.4
    let replaced_intensity := 0.25
    let new_intensity := (0.3 + 0.1 * replaced_fraction)  -- derived from (0.6 * 0.5 + 0.4 * 0.25)
    new_intensity = 0.4 :=
by
    sorry

end new_paint_intensity_l1494_149467


namespace cost_price_per_meter_l1494_149440

def total_length : ℝ := 9.25
def total_cost : ℝ := 397.75

theorem cost_price_per_meter : total_cost / total_length = 43 := sorry

end cost_price_per_meter_l1494_149440


namespace total_marbles_l1494_149442

def Mary_marbles : ℕ := 9
def Joan_marbles : ℕ := 3
def Peter_marbles : ℕ := 7

theorem total_marbles : Mary_marbles + Joan_marbles + Peter_marbles = 19 := by
  sorry

end total_marbles_l1494_149442


namespace notebooks_bought_l1494_149457

def dan_total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def pens_cost : ℕ := 1
def pencils_cost : ℕ := 1
def notebook_cost : ℕ := 3

theorem notebooks_bought :
  ∃ x : ℕ, dan_total_spent - (backpack_cost + pens_cost + pencils_cost) = x * notebook_cost ∧ x = 5 := 
by
  sorry

end notebooks_bought_l1494_149457


namespace inequality_holds_and_equality_occurs_l1494_149411

theorem inequality_holds_and_equality_occurs (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (x = 2 ∧ y = 2 → 1 / (x + 3) + 1 / (y + 3) = 2 / 5) :=
by
  sorry

end inequality_holds_and_equality_occurs_l1494_149411


namespace find_q_l1494_149455

noncomputable def common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 4 = 27 ∧ a 7 = -729 ∧ ∀ n m, a n = a m * q ^ (n - m)

theorem find_q {a : ℕ → ℝ} {q : ℝ} (h : common_ratio_of_geometric_sequence a q) :
  q = -3 :=
by {
  sorry
}

end find_q_l1494_149455


namespace intersection_in_fourth_quadrant_l1494_149470

theorem intersection_in_fourth_quadrant :
  (∃ x y : ℝ, y = -x ∧ y = 2 * x - 1 ∧ x = 1 ∧ y = -1) ∧ (1 > 0 ∧ -1 < 0) :=
by
  sorry

end intersection_in_fourth_quadrant_l1494_149470


namespace either_x_or_y_is_even_l1494_149424

theorem either_x_or_y_is_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : (2 ∣ x) ∨ (2 ∣ y) :=
by
  sorry

end either_x_or_y_is_even_l1494_149424


namespace balloon_count_l1494_149468

theorem balloon_count (gold_balloon silver_balloon black_balloon blue_balloon green_balloon total_balloon : ℕ) (h1 : gold_balloon = 141) 
                      (h2 : silver_balloon = (gold_balloon / 3) * 5) 
                      (h3 : black_balloon = silver_balloon / 2) 
                      (h4 : blue_balloon = black_balloon / 2) 
                      (h5 : green_balloon = (blue_balloon / 4) * 3) 
                      (h6 : total_balloon = gold_balloon + silver_balloon + black_balloon + blue_balloon + green_balloon): 
                      total_balloon = 593 :=
by 
  sorry

end balloon_count_l1494_149468


namespace bob_total_spend_in_usd_l1494_149433

theorem bob_total_spend_in_usd:
  let coffee_cost_yen := 250
  let sandwich_cost_yen := 150
  let yen_to_usd := 110
  (coffee_cost_yen + sandwich_cost_yen) / yen_to_usd = 3.64 := by
  sorry

end bob_total_spend_in_usd_l1494_149433


namespace investment_interests_l1494_149431

theorem investment_interests (x y : ℝ) (h₁ : x + y = 24000)
  (h₂ : 0.045 * x + 0.06 * y = 0.05 * 24000) : (x = 16000) ∧ (y = 8000) :=
  by
  sorry

end investment_interests_l1494_149431


namespace quadratic_minimum_value_interval_l1494_149462

theorem quadratic_minimum_value_interval (k : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x < 2 → (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (2*k^2 + 2*k - 1)) → (0 ≤ k ∧ k < 1) :=
by {
  sorry
}

end quadratic_minimum_value_interval_l1494_149462


namespace max_ratio_of_odd_integers_is_nine_l1494_149437

-- Define odd positive integers x and y whose mean is 55
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := 0 < n
def mean_is_55 (x y : ℕ) : Prop := (x + y) / 2 = 55

-- The problem statement
theorem max_ratio_of_odd_integers_is_nine (x y : ℕ) 
  (hx : is_positive x) (hy : is_positive y)
  (ox : is_odd x) (oy : is_odd y)
  (mean : mean_is_55 x y) : 
  ∀ r, r = (x / y : ℚ) → r ≤ 9 :=
by
  sorry

end max_ratio_of_odd_integers_is_nine_l1494_149437


namespace even_integers_in_form_3k_plus_4_l1494_149492

theorem even_integers_in_form_3k_plus_4 (n : ℕ) :
  (20 ≤ n ∧ n ≤ 180 ∧ ∃ k : ℕ, n = 3 * k + 4) → 
  (∃ s : ℕ, s = 27) :=
by
  sorry

end even_integers_in_form_3k_plus_4_l1494_149492


namespace prove_tan_570_eq_sqrt_3_over_3_l1494_149415

noncomputable def tan_570_eq_sqrt_3_over_3 : Prop :=
  Real.tan (570 * Real.pi / 180) = Real.sqrt 3 / 3

theorem prove_tan_570_eq_sqrt_3_over_3 : tan_570_eq_sqrt_3_over_3 :=
by
  sorry

end prove_tan_570_eq_sqrt_3_over_3_l1494_149415


namespace total_money_given_by_father_is_100_l1494_149474

-- Define the costs and quantities given in the problem statement.
def cost_per_sharpener := 5
def cost_per_notebook := 5
def cost_per_eraser := 4
def money_spent_on_highlighters := 30

def heaven_sharpeners := 2
def heaven_notebooks := 4
def brother_erasers := 10

-- Calculate the total amount of money given by their father.
def total_money_given : ℕ :=
  heaven_sharpeners * cost_per_sharpener +
  heaven_notebooks * cost_per_notebook +
  brother_erasers * cost_per_eraser +
  money_spent_on_highlighters

-- Lean statement to prove
theorem total_money_given_by_father_is_100 :
  total_money_given = 100 := by
  sorry

end total_money_given_by_father_is_100_l1494_149474


namespace problem_l1494_149420

theorem problem (a b : ℝ) (h : ∀ x : ℝ, (x + a) * (x + b) = x^2 + 4 * x + 3) : a + b = 4 :=
by
  sorry

end problem_l1494_149420


namespace bags_production_l1494_149469

def machines_bags_per_minute (n : ℕ) : ℕ :=
  if n = 15 then 45 else 0 -- this definition is constrained by given condition

def bags_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  machines * (machines_bags_per_minute 15 / 15) * minutes

theorem bags_production (machines minutes : ℕ) (h : machines = 150 ∧ minutes = 8):
  bags_produced machines minutes = 3600 :=
by
  cases h with
  | intro h_machines h_minutes =>
    sorry

end bags_production_l1494_149469


namespace children_tickets_l1494_149490

theorem children_tickets (A C : ℝ) (h1 : A + C = 200) (h2 : 3 * A + 1.5 * C = 510) : C = 60 := by
  sorry

end children_tickets_l1494_149490


namespace total_gallons_in_tanks_l1494_149458

def tank1_capacity : ℕ := 7000
def tank2_capacity : ℕ := 5000
def tank3_capacity : ℕ := 3000

def tank1_filled : ℚ := 3/4
def tank2_filled : ℚ := 4/5
def tank3_filled : ℚ := 1/2

theorem total_gallons_in_tanks :
  (tank1_capacity * tank1_filled + tank2_capacity * tank2_filled + tank3_capacity * tank3_filled : ℚ) = 10750 := 
by 
  sorry

end total_gallons_in_tanks_l1494_149458


namespace lottery_probability_prizes_l1494_149459

theorem lottery_probability_prizes :
  let total_tickets := 3
  let first_prize_tickets := 1
  let second_prize_tickets := 1
  let non_prize_tickets := 1
  let person_a_wins_first := (2 / 3 : ℝ)
  let person_b_wins_from_remaining := (1 / 2 : ℝ)
  (2 / 3 * 1 / 2) = (1 / 3 : ℝ) := sorry

end lottery_probability_prizes_l1494_149459


namespace sign_of_x_minus_y_l1494_149464

theorem sign_of_x_minus_y (x y a : ℝ) (h1 : x + y > 0) (h2 : a < 0) (h3 : a * y > 0) : x - y > 0 := 
by 
  sorry

end sign_of_x_minus_y_l1494_149464


namespace pencils_across_diameter_l1494_149493

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l1494_149493


namespace calculate_square_add_subtract_l1494_149488

theorem calculate_square_add_subtract (a b : ℤ) :
  (41 : ℤ)^2 = (40 : ℤ)^2 + 81 ∧ (39 : ℤ)^2 = (40 : ℤ)^2 - 79 :=
by
  sorry

end calculate_square_add_subtract_l1494_149488


namespace parallel_vectors_cosine_identity_l1494_149475

-- Defining the problem in Lean 4

theorem parallel_vectors_cosine_identity :
  ∀ α : ℝ, (∃ k : ℝ, (1 / 3, Real.tan α) = (k * Real.cos α, k)) →
  Real.cos (Real.pi / 2 + α) = -1 / 3 :=
by
  sorry

end parallel_vectors_cosine_identity_l1494_149475


namespace sum_of_first_nine_terms_l1494_149486

theorem sum_of_first_nine_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 = 3 * a 3 - 6) : 
  (9 * (a 0 + a 8)) / 2 = 27 := 
sorry

end sum_of_first_nine_terms_l1494_149486


namespace divisibility_polynomial_l1494_149452

variables {a m x n : ℕ}

theorem divisibility_polynomial (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) :=
by
  sorry

end divisibility_polynomial_l1494_149452


namespace find_k_l1494_149472

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end find_k_l1494_149472


namespace find_D_l1494_149423

variables (A B C D : ℤ)
axiom h1 : A + C = 15
axiom h2 : A - B = 1
axiom h3 : C + C = A
axiom h4 : B - D = 2
axiom h5 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

theorem find_D : D = 7 :=
by sorry

end find_D_l1494_149423


namespace solution_correct_l1494_149463

noncomputable def satisfies_conditions (f : ℤ → ℝ) : Prop :=
  (f 1 = 5 / 2) ∧ (f 0 ≠ 0) ∧ (∀ m n : ℤ, f m * f n = f (m + n) + f (m - n))

theorem solution_correct (f : ℤ → ℝ) :
  satisfies_conditions f → ∀ n : ℤ, f n = 2^n + (1/2)^n :=
by sorry

end solution_correct_l1494_149463


namespace find_x_l1494_149432

theorem find_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (7 * x + 42)) : x = -3 / 2 :=
sorry

end find_x_l1494_149432


namespace jasper_hot_dogs_fewer_l1494_149416

theorem jasper_hot_dogs_fewer (chips drinks hot_dogs : ℕ)
  (h1 : chips = 27)
  (h2 : drinks = 31)
  (h3 : drinks = hot_dogs + 12) : 27 - hot_dogs = 8 := by
  sorry

end jasper_hot_dogs_fewer_l1494_149416


namespace nicky_run_time_l1494_149444

-- Define the constants according to the conditions in the problem
def head_start : ℕ := 100 -- Nicky's head start (meters)
def cr_speed : ℕ := 8 -- Cristina's speed (meters per second)
def ni_speed : ℕ := 4 -- Nicky's speed (meters per second)

-- Define the event of Cristina catching up to Nicky
def meets_at_time (t : ℕ) : Prop :=
  cr_speed * t = head_start + ni_speed * t

-- The proof statement
theorem nicky_run_time : ∃ t : ℕ, meets_at_time t ∧ t = 25 :=
by
  sorry

end nicky_run_time_l1494_149444


namespace shoe_store_total_shoes_l1494_149429

theorem shoe_store_total_shoes (b k : ℕ) (h1 : b = 22) (h2 : k = 2 * b) : b + k = 66 :=
by
  sorry

end shoe_store_total_shoes_l1494_149429


namespace total_amount_l1494_149430

theorem total_amount
  (x y z : ℝ)
  (hy : y = 0.45 * x)
  (hz : z = 0.50 * x)
  (y_share : y = 27) :
  x + y + z = 117 :=
by
  sorry

end total_amount_l1494_149430


namespace find_sum_of_common_ratios_l1494_149465

-- Definition of the problem conditions
def is_geometric_sequence (a b c : ℕ) (k : ℕ) (r : ℕ) : Prop :=
  b = k * r ∧ c = k * r * r

-- Main theorem statement
theorem find_sum_of_common_ratios (k p r a_2 a_3 b_2 b_3 : ℕ) 
  (hk : k ≠ 0)
  (hp_neq_r : p ≠ r)
  (hp_seq : is_geometric_sequence k a_2 a_3 k p)
  (hr_seq : is_geometric_sequence k b_2 b_3 k r)
  (h_eq : a_3 - b_3 = 3 * (a_2 - b_2)) :
  p + r = 3 :=
sorry

end find_sum_of_common_ratios_l1494_149465


namespace gcf_252_96_l1494_149495

theorem gcf_252_96 : Int.gcd 252 96 = 12 := by
  sorry

end gcf_252_96_l1494_149495


namespace largest_frog_weight_l1494_149446

theorem largest_frog_weight (S L : ℕ) (h1 : L = 10 * S) (h2 : L = S + 108): L = 120 := by
  sorry

end largest_frog_weight_l1494_149446


namespace real_solutions_eq_l1494_149402

theorem real_solutions_eq :
  ∀ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) → (x = 10 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_l1494_149402


namespace negation_proposition_l1494_149412

-- Define the original proposition
def unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) → (x1 = x2)

-- Define the negation of the proposition
def negation_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ¬ unique_solution a b h

-- Define a proposition for "no unique solution"
def no_unique_solution (a b : ℝ) (h : a ≠ 0) : Prop :=
  ∃ x1 x2 : ℝ, (a * x1 = b ∧ a * x2 = b) ∧ (x1 ≠ x2)

-- The Lean 4 statement
theorem negation_proposition (a b : ℝ) (h : a ≠ 0) :
  negation_unique_solution a b h :=
sorry

end negation_proposition_l1494_149412


namespace remaining_amount_to_pay_l1494_149480

-- Define the constants and conditions
def total_cost : ℝ := 1300
def first_deposit : ℝ := 0.10 * total_cost
def second_deposit : ℝ := 2 * first_deposit
def promotional_discount : ℝ := 0.05 * total_cost
def interest_rate : ℝ := 0.02

-- Define the function to calculate the final payment
def final_payment (total_cost first_deposit second_deposit promotional_discount interest_rate : ℝ) : ℝ :=
  let total_paid := first_deposit + second_deposit
  let remaining_balance := total_cost - total_paid
  let remaining_after_discount := remaining_balance - promotional_discount
  remaining_after_discount * (1 + interest_rate)

-- Define the theorem to be proven
theorem remaining_amount_to_pay :
  final_payment total_cost first_deposit second_deposit promotional_discount interest_rate = 861.90 :=
by
  -- The proof goes here
  sorry

end remaining_amount_to_pay_l1494_149480


namespace smallest_third_term_GP_l1494_149489

theorem smallest_third_term_GP : 
  ∃ d : ℝ, 
    (11 + d) ^ 2 = 9 * (29 + 2 * d) ∧
    min (29 + 2 * 10) (29 + 2 * -14) = 1 :=
by
  sorry

end smallest_third_term_GP_l1494_149489


namespace unique_pairs_pos_int_satisfy_eq_l1494_149499

theorem unique_pairs_pos_int_satisfy_eq (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) := 
by
  sorry

end unique_pairs_pos_int_satisfy_eq_l1494_149499


namespace parallel_vectors_l1494_149451

variable (a b : ℝ × ℝ)
variable (m : ℝ)

theorem parallel_vectors (h₁ : a = (-6, 2)) (h₂ : b = (m, -3)) (h₃ : a.1 * b.2 = a.2 * b.1) : m = 9 :=
by
  sorry

end parallel_vectors_l1494_149451


namespace minimize_water_tank_construction_cost_l1494_149427

theorem minimize_water_tank_construction_cost 
  (volume : ℝ := 4800)
  (depth : ℝ := 3)
  (cost_bottom_per_m2 : ℝ := 150)
  (cost_walls_per_m2 : ℝ := 120)
  (x : ℝ) :
  (volume = x * x * depth) →
  (∀ y, y = cost_bottom_per_m2 * x * x + cost_walls_per_m2 * 4 * x * depth) →
  (x = 40) ∧ (y = 297600) :=
by
  sorry

end minimize_water_tank_construction_cost_l1494_149427


namespace minimum_value_of_f_l1494_149453

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 - (Real.log x) / x

theorem minimum_value_of_f : (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
by
  sorry

end minimum_value_of_f_l1494_149453


namespace extremum_at_neg3_l1494_149407

variable (a : ℝ)
def f (x : ℝ) : ℝ := x^3 + 5 * x^2 + a * x
def f_deriv (x : ℝ) : ℝ := 3 * x^2 + 10 * x + a

theorem extremum_at_neg3 (h : f_deriv a (-3) = 0) : a = 3 := 
  by
  sorry

end extremum_at_neg3_l1494_149407


namespace ann_top_cost_l1494_149450

noncomputable def cost_per_top (T : ℝ) := 75 = (5 * 7) + (2 * 10) + (4 * T)

theorem ann_top_cost : cost_per_top 5 :=
by {
  -- statement: prove cost per top given conditions
  sorry
}

end ann_top_cost_l1494_149450


namespace acute_angle_of_rhombus_l1494_149406

theorem acute_angle_of_rhombus (a α : ℝ) (V1 V2 : ℝ) (OA BD AN AB : ℝ) 
  (h_volumes : V1 / V2 = 1 / (2 * Real.sqrt 5)) 
  (h_V1 : V1 = (1 / 3) * Real.pi * (OA^2) * BD)
  (h_V2 : V2 = Real.pi * (AN^2) * AB)
  (h_OA : OA = a * Real.sin (α / 2))
  (h_BD : BD = 2 * a * Real.cos (α / 2))
  (h_AN : AN = a * Real.sin α)
  (h_AB : AB = a)
  : α = Real.arccos (1 / 9) :=
sorry

end acute_angle_of_rhombus_l1494_149406


namespace intersection_A_B_l1494_149445

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 3^x}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2^(-x)}

theorem intersection_A_B :
  A ∩ B = {p | p = (0, 1)} :=
by
  sorry

end intersection_A_B_l1494_149445


namespace time_to_finish_by_p_l1494_149483

theorem time_to_finish_by_p (P_rate Q_rate : ℝ) (worked_together_hours remaining_job_rate : ℝ) :
    P_rate = 1/3 ∧ Q_rate = 1/9 ∧ worked_together_hours = 2 ∧ remaining_job_rate = 1 - (worked_together_hours * (P_rate + Q_rate)) → 
    (remaining_job_rate / P_rate) * 60 = 20 := 
by
  sorry

end time_to_finish_by_p_l1494_149483


namespace flower_shop_february_roses_l1494_149484

theorem flower_shop_february_roses (roses_oct : ℕ) (roses_nov : ℕ) (roses_dec : ℕ) (roses_jan : ℕ) (d : ℕ) :
  roses_oct = 108 →
  roses_nov = 120 →
  roses_dec = 132 →
  roses_jan = 144 →
  roses_nov - roses_oct = d →
  roses_dec - roses_nov = d →
  roses_jan - roses_dec = d →
  (roses_jan + d = 156) :=
by
  intros h_oct h_nov h_dec h_jan h_diff1 h_diff2 h_diff3
  rw [h_jan, h_diff1] at *
  sorry

end flower_shop_february_roses_l1494_149484


namespace max_product_l1494_149439

def geometric_sequence (a1 q : ℝ) (n : ℕ) :=
  a1 * q ^ (n - 1)

def product_of_terms (a1 q : ℝ) (n : ℕ) :=
  (List.range n).foldr (λ i acc => acc * geometric_sequence a1 q (i + 1)) 1

theorem max_product (n : ℕ) (a1 q : ℝ) (h₁ : a1 = 1536) (h₂ : q = -1/2) :
  n = 11 ↔ ∀ m : ℕ, m ≤ 11 → product_of_terms a1 q m ≤ product_of_terms a1 q 11 :=
by
  sorry

end max_product_l1494_149439


namespace cone_height_l1494_149418

theorem cone_height (r_sphere : ℝ) (r_cone : ℝ) (waste_percentage : ℝ) 
  (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) : 
  r_sphere = 9 → r_cone = 9 → waste_percentage = 0.75 → 
  V_sphere = (4 / 3) * Real.pi * r_sphere^3 → 
  V_cone = (1 / 3) * Real.pi * r_cone^2 * h → 
  V_cone = waste_percentage * V_sphere → 
  h = 27 :=
by
  intros r_sphere_eq r_cone_eq waste_eq V_sphere_eq V_cone_eq V_cone_waste_eq
  sorry

end cone_height_l1494_149418


namespace kyle_caught_fish_l1494_149448

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

end kyle_caught_fish_l1494_149448


namespace asep_wins_in_at_most_n_minus_5_div_4_steps_l1494_149426

theorem asep_wins_in_at_most_n_minus_5_div_4_steps (n : ℕ) (h : n ≥ 14) : 
  ∃ f : ℕ → ℕ, (∀ X d : ℕ, 0 < d → d ∣ X → (X' = X + d ∨ X' = X - d) → (f X' ≤ f X + 1)) ∧ f n ≤ (n - 5) / 4 := 
sorry

end asep_wins_in_at_most_n_minus_5_div_4_steps_l1494_149426


namespace num_accompanying_year_2022_l1494_149481

theorem num_accompanying_year_2022 : 
  ∃ N : ℤ, (N = 2) ∧ 
    (∀ n : ℤ, (100 * n + 22) % n = 0 ∧ 10 ≤ n ∧ n < 100 → n = 11 ∨ n = 22) :=
by 
  sorry

end num_accompanying_year_2022_l1494_149481


namespace min_value_frac_sum_l1494_149449

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 2) : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 3 * b = 2 ∧ (2 / a + 4 / b) = 14) :=
by
  sorry

end min_value_frac_sum_l1494_149449


namespace regular_polygon_sides_l1494_149461

theorem regular_polygon_sides (D : ℕ) (h : D = 30) :
  ∃ n : ℕ, D = n * (n - 3) / 2 ∧ n = 9 :=
by
  use 9
  rw [h]
  norm_num
  sorry

end regular_polygon_sides_l1494_149461


namespace apples_number_l1494_149478

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end apples_number_l1494_149478


namespace chestnut_picking_l1494_149485

theorem chestnut_picking 
  (P : ℕ)
  (h1 : 12 + P + (P + 2) = 26) :
  12 / P = 2 :=
sorry

end chestnut_picking_l1494_149485


namespace smallest_constant_for_triangle_sides_l1494_149441

theorem smallest_constant_for_triangle_sides (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_condition : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ N, (∀ a b c, (a + b > c ∧ b + c > a ∧ c + a > b) → (a^2 + b^2) / (a * b) < N) ∧ N = 2 := by
  sorry

end smallest_constant_for_triangle_sides_l1494_149441


namespace runners_meetings_on_track_l1494_149497

def number_of_meetings (speed1 speed2 laps : ℕ) : ℕ := ((speed1 + speed2) * laps) / (2 * (speed2 - speed1))

theorem runners_meetings_on_track 
  (speed1 speed2 : ℕ) 
  (start_laps : ℕ)
  (speed1_spec : speed1 = 4) 
  (speed2_spec : speed2 = 10) 
  (laps_spec : start_laps = 28) : 
  number_of_meetings speed1 speed2 start_laps = 77 := 
by
  rw [speed1_spec, speed2_spec, laps_spec]
  -- Add further necessary steps or lemmas if required to reach the final proving statement
  sorry

end runners_meetings_on_track_l1494_149497


namespace exam_fail_percentage_l1494_149494

theorem exam_fail_percentage
  (total_candidates : ℕ := 2000)
  (girls : ℕ := 900)
  (pass_percent : ℝ := 0.32) :
  ((total_candidates - ((pass_percent * (total_candidates - girls)) + (pass_percent * girls))) / total_candidates) * 100 = 68 :=
by
  sorry

end exam_fail_percentage_l1494_149494


namespace sum_is_correct_l1494_149496

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

end sum_is_correct_l1494_149496


namespace circles_ordering_l1494_149421

theorem circles_ordering :
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  (rA < rB) ∧ (rB < rC) :=
by
  let rA := 3
  let AB := 12 * Real.pi
  let AC := 28 * Real.pi
  let rB := Real.sqrt 12
  let rC := Real.sqrt 28
  have rA_lt_rB: rA < rB := by sorry
  have rB_lt_rC: rB < rC := by sorry
  exact ⟨rA_lt_rB, rB_lt_rC⟩

end circles_ordering_l1494_149421


namespace complex_number_quadrant_l1494_149403

theorem complex_number_quadrant (a b : ℝ) (h1 : (2 + a * (0+1*I)) / (1 + 1*I) = b + 1*I) (h2: a = 4) (h3: b = 3) : 
  0 < a ∧ 0 < b :=
by
  sorry

end complex_number_quadrant_l1494_149403


namespace pages_read_on_saturday_l1494_149491

namespace BookReading

def total_pages : ℕ := 93
def pages_read_sunday : ℕ := 20
def pages_remaining : ℕ := 43

theorem pages_read_on_saturday :
  total_pages - (pages_read_sunday + pages_remaining) = 30 :=
by
  sorry

end BookReading

end pages_read_on_saturday_l1494_149491


namespace intersection_of_A_and_B_solve_inequality_l1494_149473

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - 16 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 ≥ 0}

-- Proof problem 1: Find A ∩ B
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} :=
sorry

-- Proof problem 2: Solve the inequality with respect to x
theorem solve_inequality (a : ℝ) :
  if a = 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = ∅
  else if a > 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | 1 < x ∧ x < a}
  else
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | a < x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_solve_inequality_l1494_149473


namespace common_difference_is_1_over_10_l1494_149422

open Real

noncomputable def a_n (a₁ d: ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1)) * d / 2

theorem common_difference_is_1_over_10 (a₁ d : ℝ) 
  (h : (S_n a₁ d 2017 / 2017) - (S_n a₁ d 17 / 17) = 100) : 
  d = 1 / 10 :=
by
  sorry

end common_difference_is_1_over_10_l1494_149422


namespace how_many_eyes_do_I_see_l1494_149410

def boys : ℕ := 23
def eyes_per_boy : ℕ := 2
def total_eyes : ℕ := boys * eyes_per_boy

theorem how_many_eyes_do_I_see : total_eyes = 46 := by
  sorry

end how_many_eyes_do_I_see_l1494_149410


namespace ellipse_fence_cost_is_correct_l1494_149456

noncomputable def ellipse_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3 * (a + b) - Real.sqrt ((3 * a + b) * (a + 3 * b)))

noncomputable def fence_cost_per_meter (rate : ℝ) (a b : ℝ) : ℝ :=
  rate * ellipse_perimeter a b

theorem ellipse_fence_cost_is_correct :
  fence_cost_per_meter 3 16 12 = 265.32 :=
by
  sorry

end ellipse_fence_cost_is_correct_l1494_149456


namespace find_a_plus_b_l1494_149443

theorem find_a_plus_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 :=
by {
  sorry
}

end find_a_plus_b_l1494_149443


namespace farmer_apples_count_l1494_149454

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l1494_149454


namespace women_exceed_men_l1494_149466

variable (M W : ℕ)

theorem women_exceed_men (h1 : M + W = 24) (h2 : (M : ℚ) / (W : ℚ) = 0.6) : W - M = 6 :=
sorry

end women_exceed_men_l1494_149466


namespace problem_1_problem_2_l1494_149413

-- Proof Problem 1: Prove A ∩ B = {x | -3 ≤ x ≤ -2} given m = -3
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 1}

theorem problem_1 : B (-3) ∩ A = {x | -3 ≤ x ∧ x ≤ -2} := sorry

-- Proof Problem 2: Prove m ≥ -1 given B ⊆ A
theorem problem_2 (m : ℝ) : (B m ⊆ A) → m ≥ -1 := sorry

end problem_1_problem_2_l1494_149413


namespace bounces_to_below_30_cm_l1494_149409

theorem bounces_to_below_30_cm :
  ∃ (b : ℕ), (256 * (3 / 4)^b < 30) ∧
            (∀ (k : ℕ), k < b -> 256 * (3 / 4)^k ≥ 30) :=
by 
  sorry

end bounces_to_below_30_cm_l1494_149409


namespace product_no_xx_x_eq_x_cube_plus_one_l1494_149434

theorem product_no_xx_x_eq_x_cube_plus_one (a c : ℝ) (h1 : a - 1 = 0) (h2 : c - a = 0) : 
  (x + a) * (x ^ 2 - x + c) = x ^ 3 + 1 :=
by {
  -- Here would be the proof steps, which we omit with "sorry"
  sorry
}

end product_no_xx_x_eq_x_cube_plus_one_l1494_149434


namespace train_length_l1494_149447

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 90) 
  (h2 : time_sec = 11) 
  (h3 : length_m = 275) :
  length_m = (speed_km_hr * 1000 / 3600) * time_sec :=
sorry

end train_length_l1494_149447


namespace find_number_l1494_149487

theorem find_number :
  ∃ x : Int, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 :=
by
  sorry

end find_number_l1494_149487


namespace consecutive_probability_is_two_fifths_l1494_149400

-- Conditions
def total_days : ℕ := 5
def select_days : ℕ := 2

-- Total number of basic events (number of ways to choose 2 days out of 5)
def total_events : ℕ := Nat.choose total_days select_days -- This is C(5, 2)

-- Number of basic events where 2 selected days are consecutive
def consecutive_events : ℕ := 4

-- Probability that the selected 2 days are consecutive
def consecutive_probability : ℚ := consecutive_events / total_events

-- Theorem to be proved
theorem consecutive_probability_is_two_fifths :
  consecutive_probability = 2 / 5 :=
by
  sorry

end consecutive_probability_is_two_fifths_l1494_149400


namespace product_mod_23_l1494_149436

theorem product_mod_23 :
  (2003 * 2004 * 2005 * 2006 * 2007 * 2008) % 23 = 3 :=
by 
  sorry

end product_mod_23_l1494_149436
