import Mathlib

namespace train_cross_signal_pole_in_18_seconds_l396_39651

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 550
noncomputable def crossing_time_platform : ℝ := 51
noncomputable def signal_pole_crossing_time : ℝ := 18

theorem train_cross_signal_pole_in_18_seconds (t l_p t_p t_s : ℝ)
    (h1 : t = train_length)
    (h2 : l_p = platform_length)
    (h3 : t_p = crossing_time_platform)
    (h4 : t_s = signal_pole_crossing_time) : 
    (t + l_p) / t_p = train_length / signal_pole_crossing_time :=
by
  unfold train_length platform_length crossing_time_platform signal_pole_crossing_time at *
  -- proof will go here
  sorry

end train_cross_signal_pole_in_18_seconds_l396_39651


namespace quadratic_roots_problem_l396_39608

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l396_39608


namespace new_cooks_waiters_ratio_l396_39637

-- Definitions based on the conditions
variables (cooks waiters new_waiters : ℕ)

-- Given conditions
def ratio := 3
def initial_waiters := (ratio * cooks) / 3 -- Derived from 3 cooks / 11 waiters = 9 cooks / x waiters
def hired_waiters := 12
def total_waiters := initial_waiters + hired_waiters

-- The restaurant has 9 cooks
def restaurant_cooks := 9

-- Conclusion to prove
theorem new_cooks_waiters_ratio :
  (ratio = 3) →
  (restaurant_cooks = 9) →
  (initial_waiters = (ratio * restaurant_cooks) / 3) →
  (cooks = restaurant_cooks) →
  (waiters = initial_waiters) →
  (new_waiters = waiters + hired_waiters) →
  (new_waiters = 45) →
  (cooks / new_waiters = 1 / 5) :=
by
  intros
  sorry

end new_cooks_waiters_ratio_l396_39637


namespace denomination_of_checks_l396_39646

-- Definitions based on the conditions.
def total_checks := 30
def total_worth := 1800
def checks_spent := 24
def average_remaining := 100

-- Statement to be proven.
theorem denomination_of_checks :
  ∃ x : ℝ, (total_checks - checks_spent) * average_remaining + checks_spent * x = total_worth ∧ x = 40 :=
by
  sorry

end denomination_of_checks_l396_39646


namespace friends_popcorn_l396_39648

theorem friends_popcorn (pieces_per_serving : ℕ) (jared_count : ℕ) (total_servings : ℕ) (jared_friends : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_count = 90)
  (h3 : total_servings = 9)
  (h4 : jared_friends = 3) :
  (total_servings * pieces_per_serving - jared_count) / jared_friends = 60 := by
  sorry

end friends_popcorn_l396_39648


namespace BoxC_in_BoxA_l396_39603

-- Define the relationship between the boxes
def BoxA_has_BoxB (A B : ℕ) : Prop := A = 4 * B
def BoxB_has_BoxC (B C : ℕ) : Prop := B = 6 * C

-- Define the proof problem
theorem BoxC_in_BoxA {A B C : ℕ} (h1 : BoxA_has_BoxB A B) (h2 : BoxB_has_BoxC B C) : A = 24 * C :=
by
  sorry

end BoxC_in_BoxA_l396_39603


namespace road_completion_days_l396_39673

variable (L : ℕ) (M_1 : ℕ) (W_1 : ℕ) (t1 : ℕ) (M_2 : ℕ)

theorem road_completion_days : L = 10 ∧ M_1 = 30 ∧ W_1 = 2 ∧ t1 = 5 ∧ M_2 = 60 → D = 15 :=
by
  sorry

end road_completion_days_l396_39673


namespace problem_statement_l396_39656

def S (a b : ℤ) : ℤ := 4 * a + 6 * b
def T (a b : ℤ) : ℤ := 2 * a - 3 * b

theorem problem_statement : T (S 8 3) 4 = 88 := by
  sorry

end problem_statement_l396_39656


namespace average_mark_first_class_l396_39697

theorem average_mark_first_class (A : ℝ)
  (class1_students class2_students : ℝ)
  (avg2 combined_avg total_students total_marks_combined : ℝ)
  (h1 : class1_students = 22)
  (h2 : class2_students = 28)
  (h3 : avg2 = 60)
  (h4 : combined_avg = 51.2)
  (h5 : total_students = class1_students + class2_students)
  (h6 : total_marks_combined = total_students * combined_avg)
  (h7 : 22 * A + 28 * avg2 = total_marks_combined) :
  A = 40 :=
by
  sorry

end average_mark_first_class_l396_39697


namespace half_abs_diff_of_squares_l396_39681

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l396_39681


namespace find_line_equation_l396_39654

open Real

noncomputable def line_equation (x y : ℝ) (k : ℝ) : ℝ := k * x - y + 4 - 3 * k

noncomputable def distance_to_line (x1 y1 k : ℝ) : ℝ :=
  abs (k * x1 - y1 + 4 - 3 * k) / sqrt (k^2 + 1)

theorem find_line_equation :
  (∃ k : ℝ, (k = 2 ∨ k = -2 / 3) ∧
    (∀ x y, (x, y) = (3, 4) → (2 * x - y - 2 = 0 ∨ 2 * x + 3 * y - 18 = 0)))
    ∧ (line_equation (-2) 2 2 = line_equation 4 (-2) 2)
    ∧ (line_equation (-2) 2 (-2 / 3) = line_equation 4 (-2) (-2 / 3)) :=
sorry

end find_line_equation_l396_39654


namespace minimum_value_of_a_l396_39642

theorem minimum_value_of_a (x y a : ℝ) (h1 : y = (1 / (x - 2)) * (x^2))
(h2 : x = a * y) : a = 3 :=
sorry

end minimum_value_of_a_l396_39642


namespace correct_option_is_B_l396_39686

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end correct_option_is_B_l396_39686


namespace negation_of_proposition_l396_39688

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x < 0 → x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x < 0 ∧ x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l396_39688


namespace geometric_sum_l396_39698

theorem geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
    (h1 : S 3 = 8)
    (h2 : S 6 = 7)
    (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 7 + a 8 + a 9 = 1 / 8 :=
by
  sorry

end geometric_sum_l396_39698


namespace converse_proposition_l396_39674

-- Define a proposition for vertical angles
def vertical_angles (α β : ℕ) : Prop := α = β

-- Define the converse of the vertical angle proposition
def converse_vertical_angles (α β : ℕ) : Prop := β = α

-- Prove that the converse of "Vertical angles are equal" is 
-- "Angles that are equal are vertical angles"
theorem converse_proposition (α β : ℕ) : vertical_angles α β ↔ converse_vertical_angles α β :=
by
  sorry

end converse_proposition_l396_39674


namespace even_of_even_square_sqrt_two_irrational_l396_39636

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l396_39636


namespace ethanol_combustion_heat_l396_39657

theorem ethanol_combustion_heat (Q : Real) :
  (∃ (m : Real), m = 0.1 ∧ (∀ (n : Real), n = 1 → Q * n / m = 10 * Q)) :=
by
  sorry

end ethanol_combustion_heat_l396_39657


namespace factorial_divisibility_l396_39693

theorem factorial_divisibility 
  (n k : ℕ) 
  (p : ℕ) 
  [hp : Fact (Nat.Prime p)] 
  (h1 : 0 < n) 
  (h2 : 0 < k) 
  (h3 : p ^ k ∣ n!) : 
  (p! ^ k ∣ n!) :=
sorry

end factorial_divisibility_l396_39693


namespace portion_spent_in_second_store_l396_39650

theorem portion_spent_in_second_store (M : ℕ) (X : ℕ) (H : M = 180)
  (H1 : M - (M / 2 + 14) = 76)
  (H2 : X + 16 = 76)
  (H3 : M = (M / 2 + 14) + (X + 16)) :
  (X : ℚ) / M = 1 / 3 :=
by 
  sorry

end portion_spent_in_second_store_l396_39650


namespace intersection_one_point_l396_39625

def quadratic_function (x : ℝ) : ℝ := -x^2 + 5 * x
def linear_function (x : ℝ) (t : ℝ) : ℝ := -3 * x + t
def quadratic_combined_function (x : ℝ) (t : ℝ) : ℝ := x^2 - 8 * x + t

theorem intersection_one_point (t : ℝ) : 
  (64 - 4 * t = 0) → t = 16 :=
by
  intro h
  sorry

end intersection_one_point_l396_39625


namespace methane_production_proof_l396_39683

noncomputable def methane_production
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : Prop :=
  methane_formed = 3

theorem methane_production_proof 
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : methane_production C H methane_formed h_formula h_initial_conditions h_reaction :=
by {
  sorry
}

end methane_production_proof_l396_39683


namespace concert_total_cost_l396_39609

noncomputable def total_cost (ticket_cost : ℕ) (processing_fee_rate : ℚ) (parking_fee : ℕ)
  (entrance_fee_per_person : ℕ) (num_persons : ℕ) (refreshments_cost : ℕ) 
  (merchandise_cost : ℕ) : ℚ :=
  let ticket_total := ticket_cost * num_persons
  let processing_fee := processing_fee_rate * (ticket_total : ℚ)
  ticket_total + processing_fee + (parking_fee + entrance_fee_per_person * num_persons 
  + refreshments_cost + merchandise_cost)

theorem concert_total_cost :
  total_cost 75 0.15 10 5 2 20 40 = 252.50 := by 
  sorry

end concert_total_cost_l396_39609


namespace complement_of_M_with_respect_to_U_l396_39615

open Set

def U : Set ℤ := {-1, -2, -3, -4}
def M : Set ℤ := {-2, -3}

theorem complement_of_M_with_respect_to_U :
  (U \ M) = {-1, -4} :=
by
  sorry

end complement_of_M_with_respect_to_U_l396_39615


namespace servings_in_box_l396_39629

-- Define amounts
def total_cereal : ℕ := 18
def per_serving : ℕ := 2

-- Define the statement to prove
theorem servings_in_box : total_cereal / per_serving = 9 :=
by
  sorry

end servings_in_box_l396_39629


namespace max_x_minus_y_l396_39628

theorem max_x_minus_y {x y : ℝ} (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  x - y ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l396_39628


namespace husband_monthly_savings_l396_39679

theorem husband_monthly_savings :
  let wife_weekly_savings := 100
  let weeks_in_month := 4
  let months := 4
  let total_weeks := weeks_in_month * months
  let wife_savings := wife_weekly_savings * total_weeks
  let stock_price := 50
  let number_of_shares := 25
  let invested_half := stock_price * number_of_shares
  let total_savings := invested_half * 2
  let husband_savings := total_savings - wife_savings
  let monthly_husband_savings := husband_savings / months
  monthly_husband_savings = 225 := 
by 
  sorry

end husband_monthly_savings_l396_39679


namespace minimize_intercepts_line_eqn_l396_39626

theorem minimize_intercepts_line_eqn (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : (1:ℝ)/a + (1:ℝ)/b = 1)
  (h2 : ∃ a b, a + b = 4 ∧ a = 2 ∧ b = 2) :
  ∀ (x y : ℝ), x + y - 2 = 0 :=
by 
  sorry

end minimize_intercepts_line_eqn_l396_39626


namespace work_completion_time_l396_39696

noncomputable def work_done (hours : ℕ) (a_rate : ℚ) (b_rate : ℚ) : ℚ :=
  if hours % 2 = 0 then (hours / 2) * (a_rate + b_rate)
  else ((hours - 1) / 2) * (a_rate + b_rate) + a_rate

theorem work_completion_time :
  let a_rate := 1/4
  let b_rate := 1/12
  (∃ t, work_done t a_rate b_rate = 1) → t = 6 := 
by
  intro h
  sorry

end work_completion_time_l396_39696


namespace radical_conjugate_sum_l396_39616

theorem radical_conjugate_sum:
  let a := 15 - Real.sqrt 500
  let b := 15 + Real.sqrt 500
  3 * (a + b) = 90 :=
by
  sorry

end radical_conjugate_sum_l396_39616


namespace intersection_A_complement_B_l396_39630

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { y | 0 ≤ y }

theorem intersection_A_complement_B : A ∩ -B = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end intersection_A_complement_B_l396_39630


namespace equation_solution_l396_39692

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end equation_solution_l396_39692


namespace problem_l396_39638

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

theorem problem : a = c ∧ a > b :=
by
  sorry

end problem_l396_39638


namespace son_and_daughter_current_ages_l396_39687

theorem son_and_daughter_current_ages
  (father_age_now : ℕ)
  (son_age_5_years_ago : ℕ)
  (daughter_age_5_years_ago : ℝ)
  (h_father_son_birth : father_age_now - (son_age_5_years_ago + 5) = (son_age_5_years_ago + 5))
  (h_father_daughter_birth : father_age_now - (daughter_age_5_years_ago + 5) = (daughter_age_5_years_ago + 5))
  (h_daughter_half_son_5_years_ago : daughter_age_5_years_ago = son_age_5_years_ago / 2) :
  son_age_5_years_ago + 5 = 12 ∧ daughter_age_5_years_ago + 5 = 8.5 :=
by
  sorry

end son_and_daughter_current_ages_l396_39687


namespace triangle_area_of_parabola_hyperbola_l396_39660

-- Definitions for parabola and hyperbola
def parabola_directrix (a : ℕ) (x y : ℝ) : Prop := x^2 = 16 * y
def hyperbola_asymptotes (a b : ℕ) (x y : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1

-- Theorem stating the area of the triangle formed by the intersections of the asymptotes with the directrix
theorem triangle_area_of_parabola_hyperbola (a b : ℕ) (h : a = 1) (h' : b = 1) : 
  ∃ (area : ℝ), area = 16 :=
sorry

end triangle_area_of_parabola_hyperbola_l396_39660


namespace boat_distance_l396_39618

theorem boat_distance (v_b : ℝ) (v_s : ℝ) (t_downstream : ℝ) (t_upstream : ℝ) (d : ℝ) :
  v_b = 7 ∧ t_downstream = 2 ∧ t_upstream = 5 ∧ d = (v_b + v_s) * t_downstream ∧ d = (v_b - v_s) * t_upstream → d = 20 :=
by {
  sorry
}

end boat_distance_l396_39618


namespace tea_mixture_price_l396_39624

theorem tea_mixture_price :
  ∃ P Q : ℝ, (62 * P + 72 * Q) / (3 * P + Q) = 64.5 :=
by
  sorry

end tea_mixture_price_l396_39624


namespace benjamin_weekly_walks_l396_39647

def walking_miles_in_week
  (work_days_per_week : ℕ)
  (work_distance_per_day : ℕ)
  (dog_walks_per_day : ℕ)
  (dog_walk_distance : ℕ)
  (best_friend_visits_per_week : ℕ)
  (best_friend_distance : ℕ)
  (store_visits_per_week : ℕ)
  (store_distance : ℕ)
  (hike_distance_per_week : ℕ) : ℕ :=
  (work_days_per_week * work_distance_per_day) +
  (dog_walks_per_day * dog_walk_distance * 7) +
  (best_friend_visits_per_week * (best_friend_distance * 2)) +
  (store_visits_per_week * (store_distance * 2)) +
  hike_distance_per_week

theorem benjamin_weekly_walks :
  walking_miles_in_week 5 (8 * 2) 2 3 1 5 2 4 10 = 158 := 
  by
    sorry

end benjamin_weekly_walks_l396_39647


namespace quotient_remainder_scaled_l396_39669

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end quotient_remainder_scaled_l396_39669


namespace average_age_of_students_l396_39667

theorem average_age_of_students :
  (8 * 14 + 6 * 16 + 17) / 15 = 15 :=
by
  sorry

end average_age_of_students_l396_39667


namespace seq_a_n_a_4_l396_39676

theorem seq_a_n_a_4 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ n : ℕ, a (n+1) = 2 * a n) ∧ (a 4 = 8) :=
sorry

end seq_a_n_a_4_l396_39676


namespace translate_one_chapter_in_three_hours_l396_39658

-- Definitions representing the conditions:
def jun_seok_time : ℝ := 4
def yoon_yeol_time : ℝ := 12

-- Question and Correct answer as a statement:
theorem translate_one_chapter_in_three_hours :
  (1 / (1 / jun_seok_time + 1 / yoon_yeol_time)) = 3 := by
sorry

end translate_one_chapter_in_three_hours_l396_39658


namespace minimum_value_is_138_l396_39675

-- Definition of problem conditions and question
def is_digit (n : ℕ) : Prop := n < 10
def digits (A : ℕ) : List ℕ := A.digits 10

def multiple_of_3_not_9 (A : ℕ) : Prop :=
  A % 3 = 0 ∧ A % 9 ≠ 0

def product_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· * ·) 1

def sum_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· + ·) 0

def given_condition (A : ℕ) : Prop :=
  A % 9 = 0 → False ∧
  (A + product_of_digits A) % 9 = 0

-- Main goal: Prove that the minimum value A == 138 satisfies the given conditions
theorem minimum_value_is_138 : ∃ A, A = 138 ∧
  multiple_of_3_not_9 A ∧
  given_condition A :=
sorry

end minimum_value_is_138_l396_39675


namespace M_gt_N_l396_39663

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2*x + 2*y - 2

theorem M_gt_N : M x y > N x y :=
by
  sorry

end M_gt_N_l396_39663


namespace p_necessary_not_sufficient_for_q_l396_39639

variables (a b c : ℝ) (p q : Prop)

def condition_p : Prop := a * b * c = 0
def condition_q : Prop := a = 0

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  let p := condition_p a b c
  let q := condition_q a
  sorry

end p_necessary_not_sufficient_for_q_l396_39639


namespace side_lengths_le_sqrt3_probability_is_1_over_3_l396_39645

open Real

noncomputable def probability_side_lengths_le_sqrt3 : ℝ :=
  let total_area : ℝ := 2 * π^2
  let satisfactory_area : ℝ := 2 * π^2 / 3
  satisfactory_area / total_area

theorem side_lengths_le_sqrt3_probability_is_1_over_3 :
  probability_side_lengths_le_sqrt3 = 1 / 3 :=
by
  sorry

end side_lengths_le_sqrt3_probability_is_1_over_3_l396_39645


namespace faction_with_more_liars_than_truth_tellers_l396_39610

theorem faction_with_more_liars_than_truth_tellers 
  (r1 r2 r3 l1 l2 l3 : ℕ) 
  (H1 : r1 + r2 + r3 + l1 + l2 + l3 = 2016)
  (H2 : r1 + l2 + l3 = 1208)
  (H3 : r2 + l1 + l3 = 908)
  (H4 : r3 + l1 + l2 = 608) :
  l3 - r3 = 100 :=
by
  sorry

end faction_with_more_liars_than_truth_tellers_l396_39610


namespace range_of_b_l396_39627

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x - a) / (2^x + 1)
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := Real.log (x^2 - b)

theorem range_of_b (a : ℝ) (b : ℝ) :
  (∀ x1 x2 : ℝ, f x1 a ≤ g x2 b) → b ≤ -Real.exp 1 :=
by
  sorry

end range_of_b_l396_39627


namespace max_non_triangulated_segments_correct_l396_39623

open Classical

/-
Problem description:
Given an equilateral triangle divided into smaller equilateral triangles with side length 1, 
we need to define the maximum number of 1-unit segments that can be marked such that no 
triangular subregion has all its sides marked.
-/

def total_segments (n : ℕ) : ℕ :=
  (3 * n * (n + 1)) / 2

def max_non_triangular_segments (n : ℕ) : ℕ :=
  n * (n + 1)

theorem max_non_triangulated_segments_correct (n : ℕ) :
  max_non_triangular_segments n = n * (n + 1) := by sorry

end max_non_triangulated_segments_correct_l396_39623


namespace charlene_sold_necklaces_l396_39640

theorem charlene_sold_necklaces 
  (initial_necklaces : ℕ) 
  (given_away : ℕ) 
  (remaining : ℕ) 
  (total_made : initial_necklaces = 60) 
  (given_to_friends : given_away = 18) 
  (left_with : remaining = 26) : 
  initial_necklaces - given_away - remaining = 16 := 
by
  sorry

end charlene_sold_necklaces_l396_39640


namespace total_value_of_coins_l396_39601

theorem total_value_of_coins (h1 : ∀ (q d : ℕ), q + d = 23)
                             (h2 : ∀ q, q = 16)
                             (h3 : ∀ d, d = 23 - 16)
                             (h4 : ∀ q, q * 0.25 = 4.00)
                             (h5 : ∀ d, d * 0.10 = 0.70)
                             : 4.00 + 0.70 = 4.70 :=
by
  sorry

end total_value_of_coins_l396_39601


namespace lateral_surface_area_of_cone_l396_39678

-- Definitions of the given conditions
def base_radius_cm : ℝ := 3
def slant_height_cm : ℝ := 5

-- The theorem to prove
theorem lateral_surface_area_of_cone :
  let r := base_radius_cm
  let l := slant_height_cm
  π * r * l = 15 * π := 
by
  sorry

end lateral_surface_area_of_cone_l396_39678


namespace find_naturals_for_divisibility_l396_39677

theorem find_naturals_for_divisibility (n : ℕ) (h1 : 3 * n ≠ 1) :
  (∃ k : ℤ, 7 * n + 5 = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := 
by
  sorry

end find_naturals_for_divisibility_l396_39677


namespace treasure_chest_coins_l396_39641

theorem treasure_chest_coins :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 9 = 5) ∧ (n ≥ 0) ∧
  (∀ m : ℕ, (m % 8 = 6) ∧ (m % 9 = 5) → m ≥ 0 → n ≤ m) ∧
  (∃ r : ℕ, n = 11 * (n / 11) + r ∧ r = 3) :=
by
  sorry

end treasure_chest_coins_l396_39641


namespace cuboid_breadth_l396_39649

theorem cuboid_breadth (l h A : ℝ) (w : ℝ) :
  l = 8 ∧ h = 12 ∧ A = 960 → 2 * (l * w + l * h + w * h) = A → w = 19.2 :=
by
  intros h1 h2
  sorry

end cuboid_breadth_l396_39649


namespace fraction_product_l396_39662

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l396_39662


namespace circle_arc_sum_bounds_l396_39633

open Nat

theorem circle_arc_sum_bounds :
  let red_points := 40
  let blue_points := 30
  let green_points := 20
  let total_arcs := 90
  let T := 0 * red_points + 1 * blue_points + 2 * green_points
  let S_min := 6
  let S_max := 140
  (∀ S, (S = 2 * T - A) → (0 ≤ A ∧ A ≤ 134) → (S_min ≤ S ∧ S ≤ S_max))
  → ∃ S_min S_max, S_min = 6 ∧ S_max = 140 :=
by
  intros
  sorry

end circle_arc_sum_bounds_l396_39633


namespace max_value_eq_two_l396_39612

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l396_39612


namespace train_speed_fraction_l396_39691

theorem train_speed_fraction (T : ℝ) (hT : T = 3) : T / (T + 0.5) = 6 / 7 := by
  sorry

end train_speed_fraction_l396_39691


namespace sugar_water_sweeter_l396_39664

variable (a b m : ℝ)
variable (a_pos : a > 0) (b_gt_a : b > a) (m_pos : m > 0)

theorem sugar_water_sweeter : (a + m) / (b + m) > a / b :=
by
  sorry

end sugar_water_sweeter_l396_39664


namespace number_of_symmetric_subsets_l396_39682

def has_integer_solutions (m : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = -36 ∧ x + y = -m

def M : Set ℤ :=
  {m | has_integer_solutions m}

def is_symmetric_subset (A : Set ℤ) : Prop :=
  A ⊆ M ∧ ∀ a ∈ A, -a ∈ A

theorem number_of_symmetric_subsets :
  (∃ A : Set ℤ, is_symmetric_subset A ∧ A ≠ ∅) →
  (∃ n : ℕ, n = 31) :=
by
  sorry

end number_of_symmetric_subsets_l396_39682


namespace division_correct_l396_39604

theorem division_correct :
  250 / (15 + 13 * 3^2) = 125 / 66 :=
by
  -- The proof steps can be filled in here.
  sorry

end division_correct_l396_39604


namespace solution_contains_non_zero_arrays_l396_39653

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

end solution_contains_non_zero_arrays_l396_39653


namespace problem1_l396_39694

theorem problem1 (a : ℝ) (x : ℝ) (h : a > 0) : |x - (1/a)| + |x + a| ≥ 2 :=
sorry

end problem1_l396_39694


namespace sum_of_digits_of_d_l396_39632

theorem sum_of_digits_of_d (d : ℕ) (h₁ : ∃ d_ca : ℕ, d_ca = (8 * d) / 5) (h₂ : d_ca - 75 = d) :
  (1 + 2 + 5 = 8) :=
by
  sorry

end sum_of_digits_of_d_l396_39632


namespace savings_is_22_77_cents_per_egg_l396_39659

-- Defining the costs and discount condition
def cost_per_large_egg_StoreA : ℚ := 0.55
def cost_per_extra_large_egg_StoreA : ℚ := 0.65
def discounted_cost_of_three_trays_large_StoreB : ℚ := 38
def total_eggs_in_three_trays : ℕ := 90

-- Savings calculation
def savings_per_egg : ℚ := (cost_per_extra_large_egg_StoreA - (discounted_cost_of_three_trays_large_StoreB / total_eggs_in_three_trays)) * 100

-- The statement to prove
theorem savings_is_22_77_cents_per_egg : savings_per_egg = 22.77 :=
by
  -- Here the proof would go, but we are omitting it with sorry
  sorry

end savings_is_22_77_cents_per_egg_l396_39659


namespace hypotenuse_length_l396_39635

theorem hypotenuse_length
  (a b : ℝ)
  (V1 : ℝ := (1/3) * Real.pi * a * b^2)
  (V2 : ℝ := (1/3) * Real.pi * b * a^2)
  (hV1 : V1 = 800 * Real.pi)
  (hV2 : V2 = 1920 * Real.pi) :
  Real.sqrt (a^2 + b^2) = 26 :=
by
  sorry

end hypotenuse_length_l396_39635


namespace michael_card_count_l396_39617

variable (Lloyd Mark Michael : ℕ)
variable (L : ℕ)

-- Conditions from the problem
axiom condition1 : Mark = 3 * Lloyd
axiom condition2 : Mark + 10 = Michael
axiom condition3 : Lloyd + Mark + (Michael + 80) = 300

-- The correct answer we want to prove
theorem michael_card_count : Michael = 100 :=
by
  -- Proof will be here.
  sorry

end michael_card_count_l396_39617


namespace quadratic_one_solution_m_value_l396_39634

theorem quadratic_one_solution_m_value (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) → (b^2 - 4 * a * m = 0) → m = 3 :=
by
  sorry

end quadratic_one_solution_m_value_l396_39634


namespace speed_of_train_is_correct_l396_39668

noncomputable def speedOfTrain := 
  let lengthOfTrain : ℝ := 800 -- length of the train in meters
  let timeToCrossMan : ℝ := 47.99616030717543 -- time in seconds to cross the man
  let speedOfMan : ℝ := 5 * (1000 / 3600) -- speed of the man in m/s (conversion from km/hr to m/s)
  let relativeSpeed : ℝ := lengthOfTrain / timeToCrossMan -- relative speed of the train
  let speedOfTrainInMS : ℝ := relativeSpeed + speedOfMan -- speed of the train in m/s
  let speedOfTrainInKMHR : ℝ := speedOfTrainInMS * (3600 / 1000) -- speed in km/hr
  64.9848 -- result is approximately 64.9848 km/hr

theorem speed_of_train_is_correct :
  speedOfTrain = 64.9848 :=
by
  sorry

end speed_of_train_is_correct_l396_39668


namespace largest_non_sum_l396_39600

theorem largest_non_sum (n : ℕ) : 
  ¬ (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ b ∣ 2 ∧ n = 36 * a + b) ↔ n = 104 :=
by
  sorry

end largest_non_sum_l396_39600


namespace rhombuses_in_grid_l396_39605

def number_of_rhombuses (n : ℕ) : ℕ :=
(n - 1) * n + (n - 1) * n

theorem rhombuses_in_grid :
  number_of_rhombuses 5 = 30 :=
by
  sorry

end rhombuses_in_grid_l396_39605


namespace tangent_parabola_line_l396_39695

theorem tangent_parabola_line (a : ℝ) :
  (∃ x0 : ℝ, ax0^2 + 3 = 2 * x0 + 1) ∧ (∀ x : ℝ, a * x^2 - 2 * x + 2 = 0 → x = x0) → a = 1/2 :=
by
  intro h
  sorry

end tangent_parabola_line_l396_39695


namespace distance_of_course_l396_39671

-- Definitions
def teamESpeed : ℕ := 20
def teamASpeed : ℕ := teamESpeed + 5

-- Time taken by Team E
variable (tE : ℕ)

-- Distance calculation
def teamEDistance : ℕ := teamESpeed * tE
def teamADistance : ℕ := teamASpeed * (tE - 3)

-- Proof statement
theorem distance_of_course (tE : ℕ) (h : teamEDistance tE = teamADistance tE) : teamEDistance tE = 300 :=
sorry

end distance_of_course_l396_39671


namespace B_work_time_l396_39672

theorem B_work_time :
  (∀ A_efficiency : ℝ, A_efficiency = 1 / 12 → ∀ B_efficiency : ℝ, B_efficiency = A_efficiency * 1.2 → (1 / B_efficiency = 10)) :=
by
  intros A_efficiency A_efficiency_eq B_efficiency B_efficiency_eq
  sorry

end B_work_time_l396_39672


namespace tangent_line_eqn_unique_local_minimum_l396_39613

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

def tangent_line_at_1 (x y : ℝ) : Prop :=
  2 * x + y - Real.exp 1 - 4 = 0

theorem tangent_line_eqn :
  tangent_line_at_1 1 (f 1) :=
sorry

noncomputable def h (x : ℝ) : ℝ := Real.exp x * (x - 1) - 2

theorem unique_local_minimum :
  ∃! c : ℝ, 1 < c ∧ c < 2 ∧ (∀ x < c, f x > f c) ∧ (∀ x > c, f c < f x) :=
sorry

end tangent_line_eqn_unique_local_minimum_l396_39613


namespace number_of_perfect_squares_criteria_l396_39602

noncomputable def number_of_multiples_of_40_squares_lt_4e6 : ℕ :=
  let upper_limit := 2000
  let multiple := 40
  let largest_multiple := upper_limit - (upper_limit % multiple)
  largest_multiple / multiple

theorem number_of_perfect_squares_criteria :
  number_of_multiples_of_40_squares_lt_4e6 = 49 :=
sorry

end number_of_perfect_squares_criteria_l396_39602


namespace max_wrappers_l396_39685

-- Definitions for the conditions
def total_wrappers : ℕ := 49
def andy_wrappers : ℕ := 34

-- The problem statement to prove
theorem max_wrappers : total_wrappers - andy_wrappers = 15 :=
by
  sorry

end max_wrappers_l396_39685


namespace fibonacci_factorial_sum_l396_39619

def factorial_last_two_digits(n: ℕ) : ℕ :=
  if n > 10 then 0 else 
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 3628800 % 100
  | _ => 0

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

noncomputable def sum_last_two_digits (l: List ℕ) : ℕ :=
  l.map factorial_last_two_digits |>.sum

theorem fibonacci_factorial_sum:
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end fibonacci_factorial_sum_l396_39619


namespace composite_dice_product_probability_l396_39665

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end composite_dice_product_probability_l396_39665


namespace quadratic_has_minimum_l396_39684

theorem quadratic_has_minimum 
  (a b : ℝ) (h : a ≠ 0) (g : ℝ → ℝ) 
  (H : ∀ x, g x = a * x^2 + b * x + (b^2 / a)) :
  ∃ x₀, ∀ x, g x ≥ g x₀ :=
by sorry

end quadratic_has_minimum_l396_39684


namespace swimming_speed_l396_39606

variable (v s : ℝ)

-- Given conditions
def stream_speed : Prop := s = 0.5
def time_relationship : Prop := ∀ d : ℝ, d > 0 → d / (v - s) = 2 * (d / (v + s))

-- The theorem to prove
theorem swimming_speed (h1 : stream_speed s) (h2 : time_relationship v s) : v = 1.5 :=
  sorry

end swimming_speed_l396_39606


namespace brendan_yards_per_week_l396_39670

def original_speed_flat : ℝ := 8  -- Brendan's speed on flat terrain in yards/day
def improvement_flat : ℝ := 0.5   -- Lawn mower improvement on flat terrain (50%)
def reduction_uneven : ℝ := 0.35  -- Speed reduction on uneven terrain (35%)
def days_flat : ℝ := 4            -- Days on flat terrain
def days_uneven : ℝ := 3          -- Days on uneven terrain

def improved_speed_flat : ℝ := original_speed_flat * (1 + improvement_flat)
def speed_uneven : ℝ := improved_speed_flat * (1 - reduction_uneven)

def total_yards_week : ℝ := (improved_speed_flat * days_flat) + (speed_uneven * days_uneven)

theorem brendan_yards_per_week : total_yards_week = 71.4 :=
sorry

end brendan_yards_per_week_l396_39670


namespace max_consecutive_sum_le_1000_l396_39666

theorem max_consecutive_sum_le_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → (m * (m + 1)) / 2 < 1000) ∧ ¬∃ n' : ℕ, n < n' ∧ (n' * (n' + 1)) / 2 < 1000 :=
sorry

end max_consecutive_sum_le_1000_l396_39666


namespace find_y_l396_39689

theorem find_y (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := 
sorry

end find_y_l396_39689


namespace sum_value_l396_39607

variable (T R S PV : ℝ)
variable (TD SI : ℝ) (h_td : TD = 80) (h_si : SI = 88)
variable (h1 : SI = TD + (TD * R * T) / 100)
variable (h2 : (PV * R * T) / 100 = TD)
variable (h3 : PV = S - TD)
variable (h4 : R * T = 10)

theorem sum_value : S = 880 := by
  sorry

end sum_value_l396_39607


namespace absent_children_count_l396_39631

-- Definition of conditions
def total_children := 700
def bananas_per_child := 2
def bananas_extra := 2
def total_bananas := total_children * bananas_per_child

-- The proof goal
theorem absent_children_count (A P : ℕ) (h_P : P = total_children - A)
    (h_bananas : total_bananas = P * (bananas_per_child + bananas_extra)) : A = 350 :=
by
  -- Since this is a statement only, we place a sorry here to skip the proof.
  sorry

end absent_children_count_l396_39631


namespace machine_present_value_l396_39643

theorem machine_present_value
  (r : ℝ)  -- the depletion rate
  (t : ℝ)  -- the time in years
  (V_t : ℝ)  -- the value of the machine after time t
  (V_0 : ℝ)  -- the present value of the machine
  (h1 : r = 0.10)  -- condition for depletion rate
  (h2 : t = 2)  -- condition for time
  (h3 : V_t = 729)  -- condition for machine's value after time t
  (h4 : V_t = V_0 * (1 - r) ^ t)  -- exponential decay formula
  : V_0 = 900 :=
sorry

end machine_present_value_l396_39643


namespace cars_sold_first_day_l396_39690

theorem cars_sold_first_day (c_2 c_3 : ℕ) (total : ℕ) (h1 : c_2 = 16) (h2 : c_3 = 27) (h3 : total = 57) :
  ∃ c_1 : ℕ, c_1 + c_2 + c_3 = total ∧ c_1 = 14 :=
by
  sorry

end cars_sold_first_day_l396_39690


namespace domain_of_inverse_function_l396_39680

noncomputable def log_inverse_domain : Set ℝ :=
  {y | y ≥ 5}

theorem domain_of_inverse_function :
  ∀ y, y ∈ log_inverse_domain ↔ ∃ x, x ≥ 3 ∧ y = 4 + Real.logb 2 (x - 1) :=
by
  sorry

end domain_of_inverse_function_l396_39680


namespace smallest_k_l396_39652

def arith_seq_sum (k n : ℕ) : ℕ :=
  (n + 1) * (2 * k + n) / 2

theorem smallest_k (k n : ℕ) (h_sum : arith_seq_sum k n = 100) :
  k = 9 :=
by
  sorry

end smallest_k_l396_39652


namespace circumcircle_radius_of_sector_l396_39611

theorem circumcircle_radius_of_sector (θ : Real) (r : Real) (cos_val : Real) (R : Real) :
  θ = 30 * Real.pi / 180 ∧ r = 8 ∧ cos_val = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ R = 8 * (Real.sqrt 6 - Real.sqrt 2) →
  R = 8 * (Real.sqrt 6 - Real.sqrt 2) :=
by
  sorry

end circumcircle_radius_of_sector_l396_39611


namespace least_positive_linear_combination_l396_39622

theorem least_positive_linear_combination :
  ∃ x y z : ℤ, 0 < 24 * x + 20 * y + 12 * z ∧ ∀ n : ℤ, (∃ x y z : ℤ, n = 24 * x + 20 * y + 12 * z) → 0 < n → 4 ≤ n :=
by
  sorry

end least_positive_linear_combination_l396_39622


namespace suraj_new_average_l396_39699

noncomputable def suraj_average (A : ℝ) : ℝ := A + 8

theorem suraj_new_average (A : ℝ) (h_conditions : 14 * A + 140 = 15 * (A + 8)) :
  suraj_average A = 28 :=
by
  sorry

end suraj_new_average_l396_39699


namespace carrie_remaining_money_l396_39661

def initial_money : ℝ := 200
def sweater_cost : ℝ := 36
def tshirt_cost : ℝ := 12
def tshirt_discount : ℝ := 0.10
def shoes_cost : ℝ := 45
def jeans_cost : ℝ := 52
def scarf_cost : ℝ := 18
def sales_tax_rate : ℝ := 0.05

-- Calculate tshirt price after discount
def tshirt_final_price : ℝ := tshirt_cost * (1 - tshirt_discount)

-- Sum all the item costs before tax
def total_cost_before_tax : ℝ := sweater_cost + tshirt_final_price + shoes_cost + jeans_cost + scarf_cost

-- Calculate the total sales tax
def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

-- Calculate total cost after tax
def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax

-- Calculate the remaining money
def remaining_money (initial : ℝ) (total : ℝ) : ℝ := initial - total

theorem carrie_remaining_money
  (initial_money : ℝ)
  (sweater_cost : ℝ)
  (tshirt_cost : ℝ)
  (tshirt_discount : ℝ)
  (shoes_cost : ℝ)
  (jeans_cost : ℝ)
  (scarf_cost : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : initial_money = 200)
  (h₂ : sweater_cost = 36)
  (h₃ : tshirt_cost = 12)
  (h₄ : tshirt_discount = 0.10)
  (h₅ : shoes_cost = 45)
  (h₆ : jeans_cost = 52)
  (h₇ : scarf_cost = 18)
  (h₈ : sales_tax_rate = 0.05) :
  remaining_money initial_money (total_cost_after_tax) = 30.11 := 
by 
  simp only [remaining_money, total_cost_after_tax, total_cost_before_tax, tshirt_final_price, sales_tax];
  sorry

end carrie_remaining_money_l396_39661


namespace total_white_papers_l396_39614

-- Define the given conditions
def papers_per_envelope : ℕ := 10
def number_of_envelopes : ℕ := 12

-- The theorem statement
theorem total_white_papers : (papers_per_envelope * number_of_envelopes) = 120 :=
by
  sorry

end total_white_papers_l396_39614


namespace spending_example_l396_39655

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end spending_example_l396_39655


namespace inequality_solution_l396_39621

noncomputable def inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : Prop :=
  (x^4 + y^4 + z^4) ≥ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ∧ (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) ≥ (x * y * z * (x + y + z))

theorem inequality_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  inequality_proof x y z hx hy hz :=
by 
  sorry

end inequality_solution_l396_39621


namespace probability_no_success_l396_39620

theorem probability_no_success (n : ℕ) (p : ℚ) (k : ℕ) (q : ℚ) 
  (h1 : n = 7)
  (h2 : p = 2/7)
  (h3 : k = 0)
  (h4 : q = 5/7) : 
  (1 - p) ^ n = q ^ n :=
by
  sorry

end probability_no_success_l396_39620


namespace chosen_numbers_rel_prime_l396_39644

theorem chosen_numbers_rel_prime :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 2003 → s.card = 1002 → ∃ (x y : ℕ), x ∈ s ∧ y ∈ s ∧ Nat.gcd x y = 1 :=
by
  sorry

end chosen_numbers_rel_prime_l396_39644
