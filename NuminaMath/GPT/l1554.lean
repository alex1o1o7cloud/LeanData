import Mathlib

namespace inequality_example_l1554_155499

theorem inequality_example (a b c : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (sum_eq_one : a + b + c = 1) :
  (a + 1 / a) * (b + 1 / b) * (c + 1 / c) ≥ 1000 / 27 := 
by 
  sorry

end inequality_example_l1554_155499


namespace intersecting_lines_product_l1554_155488

theorem intersecting_lines_product 
  (a b : ℝ)
  (T : Set (ℝ × ℝ)) (S : Set (ℝ × ℝ))
  (hT : T = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ a * x + y - 3 = 0})
  (hS : S = {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x - y - b = 0})
  (h_intersect : (2, 1) ∈ T) (h_intersect_S : (2, 1) ∈ S) :
  a * b = 1 := 
by
  sorry

end intersecting_lines_product_l1554_155488


namespace sum_of_tangents_slopes_at_vertices_l1554_155435

noncomputable def curve (x : ℝ) := (x + 3) * (x ^ 2 + 3)

theorem sum_of_tangents_slopes_at_vertices {x_A x_B x_C : ℝ}
  (h1 : curve x_A = x_A * (x_A ^ 2 + 6 * x_A + 9) + 3)
  (h2 : curve x_B = x_B * (x_B ^ 2 + 6 * x_B + 9) + 3)
  (h3 : curve x_C = x_C * (x_C ^ 2 + 6 * x_C + 9) + 3)
  : (3 * x_A ^ 2 + 6 * x_A + 3) + (3 * x_B ^ 2 + 6 * x_B + 3) + (3 * x_C ^ 2 + 6 * x_C + 3) = 237 :=
sorry

end sum_of_tangents_slopes_at_vertices_l1554_155435


namespace total_air_removed_after_5_strokes_l1554_155467

theorem total_air_removed_after_5_strokes:
  let initial_air := 1
  let remaining_air_after_first_stroke := initial_air * (2 / 3)
  let remaining_air_after_second_stroke := remaining_air_after_first_stroke * (3 / 4)
  let remaining_air_after_third_stroke := remaining_air_after_second_stroke * (4 / 5)
  let remaining_air_after_fourth_stroke := remaining_air_after_third_stroke * (5 / 6)
  let remaining_air_after_fifth_stroke := remaining_air_after_fourth_stroke * (6 / 7)
  initial_air - remaining_air_after_fifth_stroke = 5 / 7 := by
  sorry

end total_air_removed_after_5_strokes_l1554_155467


namespace initial_solution_amount_l1554_155404

theorem initial_solution_amount (x : ℝ) (h1 : x - 200 + 1000 = 2000) : x = 1200 := by
  sorry

end initial_solution_amount_l1554_155404


namespace rectangle_area_l1554_155412

theorem rectangle_area (a b k : ℕ)
  (h1 : k = 6 * (a + b) + 36)
  (h2 : k = 114)
  (h3 : a / b = 8 / 5) :
  a * b = 40 :=
by {
  sorry
}

end rectangle_area_l1554_155412


namespace range_of_a_l1554_155420

theorem range_of_a (A M : ℝ × ℝ) (a : ℝ) (C : ℝ × ℝ → ℝ) (hA : A = (-3, 0)) 
(hM : C M = 1) (hMA : dist M A = 2 * dist M (0, 0)) :
  a ∈ (Set.Icc (1/2 : ℝ) (3/2) ∪ Set.Icc (-3/2) (-1/2)) :=
sorry

end range_of_a_l1554_155420


namespace rug_inner_rectangle_length_l1554_155423

theorem rug_inner_rectangle_length
  (width : ℕ)
  (shaded1_width : ℕ)
  (shaded2_width : ℕ)
  (areas_in_ap : ℕ → ℕ → ℕ → Prop)
  (h1 : width = 2)
  (h2 : shaded1_width = 2)
  (h3 : shaded2_width = 2)
  (h4 : ∀ y a1 a2 a3, 
        a1 = 2 * y →
        a2 = 6 * (y + 4) →
        a3 = 10 * (y + 8) →
        areas_in_ap a1 (a2 - a1) (a3 - a2) →
        (a2 - a1 = a3 - a2)) :
  ∃ y, y = 4 :=
by
  sorry

end rug_inner_rectangle_length_l1554_155423


namespace travel_time_second_bus_l1554_155494

def distance_AB : ℝ := 100 -- kilometers
def passengers_first : ℕ := 20
def speed_first : ℝ := 60 -- kilometers per hour
def breakdown_time : ℝ := 0.5 -- hours
def passengers_second_initial : ℕ := 22
def speed_second_initial : ℝ := 50 -- kilometers per hour
def additional_passengers_speed_decrease : ℝ := 1 -- speed decrease for every additional 2 passengers
def passenger_factor : ℝ := 2
def additional_passengers : ℕ := 20
def total_time_second_bus : ℝ := 2.35 -- hours

theorem travel_time_second_bus :
  let distance_first_half := (breakdown_time * speed_first)
  let remaining_distance := distance_AB - distance_first_half
  let time_to_reach_breakdown := distance_first_half / speed_second_initial
  let new_speed_second_bus := speed_second_initial - (additional_passengers / passenger_factor) * additional_passengers_speed_decrease
  let time_from_breakdown_to_B := remaining_distance / new_speed_second_bus
  total_time_second_bus = time_to_reach_breakdown + time_from_breakdown_to_B := 
sorry

end travel_time_second_bus_l1554_155494


namespace brothers_complete_task_in_3_days_l1554_155487

theorem brothers_complete_task_in_3_days :
  (1 / 4 + 1 / 12) * 3 = 1 :=
by
  sorry

end brothers_complete_task_in_3_days_l1554_155487


namespace product_is_two_l1554_155469

theorem product_is_two : 
  ((10 : ℚ) * (1/5) * 4 * (1/16) * (1/2) * 8 = 2) :=
sorry

end product_is_two_l1554_155469


namespace age_ratio_in_9_years_l1554_155445

-- Initial age definitions for Mike and Sam
def ages (m s : ℕ) : Prop :=
  (m - 5 = 2 * (s - 5)) ∧ (m - 12 = 3 * (s - 12))

-- Proof that in 9 years the ratio of their ages will be 3:2
theorem age_ratio_in_9_years (m s x : ℕ) (h_ages : ages m s) :
  (m + x) * 2 = 3 * (s + x) ↔ x = 9 :=
by {
  sorry
}

end age_ratio_in_9_years_l1554_155445


namespace manager_salary_l1554_155447

theorem manager_salary (n : ℕ) (avg_salary : ℕ) (increment : ℕ) (new_avg_salary : ℕ) (new_total_salary : ℕ) (old_total_salary : ℕ) :
  n = 20 →
  avg_salary = 1500 →
  increment = 1000 →
  new_avg_salary = avg_salary + increment →
  old_total_salary = n * avg_salary →
  new_total_salary = (n + 1) * new_avg_salary →
  (new_total_salary - old_total_salary) = 22500 :=
by
  intros h_n h_avg_salary h_increment h_new_avg_salary h_old_total_salary h_new_total_salary
  sorry

end manager_salary_l1554_155447


namespace geometric_sequence_problem_l1554_155493

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 6) 
  (h2 : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 :=
sorry

end geometric_sequence_problem_l1554_155493


namespace correct_operation_l1554_155471

theorem correct_operation : ¬ (-2 * x + 5 * x = -7 * x) 
                          ∧ (y * x - 3 * x * y = -2 * x * y) 
                          ∧ ¬ (-x^2 - x^2 = 0) 
                          ∧ ¬ (x^2 - x = x) := 
by {
    sorry
}

end correct_operation_l1554_155471


namespace first_player_winning_strategy_l1554_155436

noncomputable def optimal_first_move : ℕ := 45

-- Prove that with 300 matches initially and following the game rules,
-- taking 45 matches on the first turn leaves the opponent in a losing position.

theorem first_player_winning_strategy (n : ℕ) (h₀ : n = 300) :
    ∃ m : ℕ, (m ≤ n / 2 ∧ n - m = 255) :=
by
  exists optimal_first_move
  sorry

end first_player_winning_strategy_l1554_155436


namespace num_ways_for_volunteers_l1554_155496

theorem num_ways_for_volunteers:
  let pavilions := 4
  let volunteers := 5
  let ways_to_choose_A := 4
  let ways_to_choose_B_after_A := 3
  let total_distributions := 
    let case_1 := 2
    let case_2 := (2^3) - 2
    case_1 + case_2
  ways_to_choose_A * ways_to_choose_B_after_A * total_distributions = 72 := 
by
  sorry

end num_ways_for_volunteers_l1554_155496


namespace factorize_polynomial_l1554_155432

theorem factorize_polynomial (a b c : ℚ) : 
  b^2 - c^2 + a * (a + 2 * b) = (a + b + c) * (a + b - c) :=
by
  sorry

end factorize_polynomial_l1554_155432


namespace solution_to_equation_l1554_155462

theorem solution_to_equation (x y : ℤ) (h : x^6 - y^2 = 648) : 
  (x = 3 ∧ y = 9) ∨ 
  (x = -3 ∧ y = 9) ∨ 
  (x = 3 ∧ y = -9) ∨ 
  (x = -3 ∧ y = -9) :=
sorry

end solution_to_equation_l1554_155462


namespace soccer_team_wins_l1554_155427

theorem soccer_team_wins 
  (total_matches : ℕ)
  (total_points : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ)
  (points_per_loss : ℕ)
  (losses : ℕ)
  (H1 : total_matches = 10)
  (H2 : total_points = 17)
  (H3 : points_per_win = 3)
  (H4 : points_per_draw = 1)
  (H5 : points_per_loss = 0)
  (H6 : losses = 3) : 
  ∃ (wins : ℕ), wins = 5 := 
by
  sorry

end soccer_team_wins_l1554_155427


namespace new_average_of_remaining_numbers_l1554_155439

theorem new_average_of_remaining_numbers (sum_12 avg_12 n1 n2 : ℝ) 
  (h1 : avg_12 = 90)
  (h2 : sum_12 = 1080)
  (h3 : n1 = 80)
  (h4 : n2 = 85)
  : (sum_12 - n1 - n2) / 10 = 91.5 := 
by
  sorry

end new_average_of_remaining_numbers_l1554_155439


namespace minimum_rotation_angle_of_square_l1554_155473

theorem minimum_rotation_angle_of_square : 
  ∀ (angle : ℝ), (∃ n : ℕ, angle = 360 / n) ∧ (n ≥ 1) ∧ (n ≤ 4) → angle = 90 :=
by
  sorry

end minimum_rotation_angle_of_square_l1554_155473


namespace b100_mod_50_l1554_155415

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b100_mod_50 : b 100 % 50 = 2 := by
  sorry

end b100_mod_50_l1554_155415


namespace negation_of_existential_l1554_155485

theorem negation_of_existential:
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 = 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existential_l1554_155485


namespace real_solutions_count_l1554_155489

noncomputable def number_of_real_solutions : ℕ := 2

theorem real_solutions_count (x : ℝ) :
  (x^2 - 5)^2 = 36 → number_of_real_solutions = 2 := by
  sorry

end real_solutions_count_l1554_155489


namespace find_added_number_l1554_155480

theorem find_added_number (R X : ℕ) (hR : R = 45) (h : 2 * (2 * R + X) = 188) : X = 4 :=
by 
  -- We would normally provide the proof here
  sorry  -- We skip the proof as per the instructions

end find_added_number_l1554_155480


namespace point_not_similar_inflection_point_ln_l1554_155419

noncomputable def similar_inflection_point (C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ (m : ℝ → ℝ), (∀ x, m x = (deriv C P.1) * (x - P.1) + P.2) ∧
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → (C (P.1 + h) > m (P.1 + h) ∧ C (P.1 - h) < m (P.1 - h)) ∨ 
                     (C (P.1 + h) < m (P.1 + h) ∧ C (P.1 - h) > m (P.1 - h))

theorem point_not_similar_inflection_point_ln :
  ¬ similar_inflection_point (fun x => Real.log x) (1, 0) :=
sorry

end point_not_similar_inflection_point_ln_l1554_155419


namespace no_periodic_sequence_first_non_zero_digit_l1554_155428

/-- 
Definition of the first non-zero digit from the unit's place in the decimal representation of n! 
-/
def first_non_zero_digit (n : ℕ) : ℕ :=
  -- This function should compute the first non-zero digit from the unit's place in n!
  -- Implementation details are skipped here.
  sorry

/-- 
Prove that no natural number \( N \) exists such that the sequence \( a_{N+1}, a_{N+2}, a_{N+3}, \ldots \) 
forms a periodic sequence, where \( a_n \) is the first non-zero digit from the unit's place in the decimal 
representation of \( n! \). 
-/
theorem no_periodic_sequence_first_non_zero_digit :
  ¬ ∃ (N : ℕ), ∃ (T : ℕ), ∀ (k : ℕ), first_non_zero_digit (N + k * T) = first_non_zero_digit (N + ((k + 1) * T)) :=
by
  sorry

end no_periodic_sequence_first_non_zero_digit_l1554_155428


namespace find_a_plus_b_l1554_155444

def star (a b : ℕ) : ℕ := a^b - a*b + 5

theorem find_a_plus_b (a b : ℕ) (ha : 2 ≤ a) (hb : 3 ≤ b) (h : star a b = 13) : a + b = 6 :=
  sorry

end find_a_plus_b_l1554_155444


namespace trains_at_initial_stations_l1554_155472

-- Define the durations of round trips for each line.
def red_round_trip : ℕ := 14
def blue_round_trip : ℕ := 16
def green_round_trip : ℕ := 18

-- Define the total time we are analyzing.
def total_time : ℕ := 2016

-- Define the statement that needs to be proved.
theorem trains_at_initial_stations : 
  (total_time % red_round_trip = 0) ∧ 
  (total_time % blue_round_trip = 0) ∧ 
  (total_time % green_round_trip = 0) := 
by
  -- The proof can be added here.
  sorry

end trains_at_initial_stations_l1554_155472


namespace range_of_a_l1554_155448

theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 - x ≤ 0 → 2^(1 - x) + a ≤ 0)) ↔ (a ≤ -2) := by
  sorry

end range_of_a_l1554_155448


namespace defective_probability_l1554_155440

variable (total_products defective_products qualified_products : ℕ)
variable (first_draw_defective second_draw_defective : Prop)

-- Definitions of the problem
def total_prods := 10
def def_prods := 4
def qual_prods := 6
def p_A := def_prods / total_prods
def p_AB := (def_prods / total_prods) * ((def_prods - 1) / (total_prods - 1))
def p_B_given_A := p_AB / p_A

-- Theorem: The probability of drawing a defective product on the second draw given the first was defective is 1/3.
theorem defective_probability 
  (hp1 : total_products = total_prods)
  (hp2 : defective_products = def_prods)
  (hp3 : qualified_products = qual_prods)
  (pA_eq : p_A = 2 / 5)
  (pAB_eq : p_AB = 2 / 15) : 
  p_B_given_A = 1 / 3 := sorry

end defective_probability_l1554_155440


namespace total_preparation_time_l1554_155431

theorem total_preparation_time
    (minutes_per_game : ℕ)
    (number_of_games : ℕ)
    (h1 : minutes_per_game = 10)
    (h2 : number_of_games = 15) :
    minutes_per_game * number_of_games = 150 :=
by
  -- Lean 4 proof goes here
  sorry

end total_preparation_time_l1554_155431


namespace moles_of_H2O_combined_l1554_155437

theorem moles_of_H2O_combined (mole_NH4Cl mole_NH4OH : ℕ) (reaction : mole_NH4Cl = 1 ∧ mole_NH4OH = 1) : 
  ∃ mole_H2O : ℕ, mole_H2O = 1 :=
by
  sorry

end moles_of_H2O_combined_l1554_155437


namespace height_table_l1554_155451

variable (l w h : ℝ)

theorem height_table (h_eq1 : l + h - w = 32) (h_eq2 : w + h - l = 28) : h = 30 := by
  sorry

end height_table_l1554_155451


namespace remaining_credit_l1554_155438

-- Define the conditions
def total_credit : ℕ := 100
def paid_on_tuesday : ℕ := 15
def paid_on_thursday : ℕ := 23

-- Statement of the problem: Prove that the remaining amount to be paid is $62
theorem remaining_credit : total_credit - (paid_on_tuesday + paid_on_thursday) = 62 := by
  sorry

end remaining_credit_l1554_155438


namespace min_steps_for_humpty_l1554_155413

theorem min_steps_for_humpty (x y : ℕ) (H : 47 * x - 37 * y = 1) : x + y = 59 :=
  sorry

end min_steps_for_humpty_l1554_155413


namespace ratio_of_potatoes_l1554_155411

-- Definitions as per conditions
def initial_potatoes : ℕ := 300
def given_to_gina : ℕ := 69
def remaining_potatoes : ℕ := 47
def k : ℕ := 2  -- Identify k is 2 based on the ratio

-- Calculate given_to_tom and total given away
def given_to_tom : ℕ := k * given_to_gina
def given_to_anne : ℕ := given_to_tom / 3

-- Arithmetical conditions derived from the problem
def total_given_away : ℕ := given_to_gina + given_to_tom + given_to_anne + remaining_potatoes

-- Proof statement to show the ratio between given_to_tom and given_to_gina is 2
theorem ratio_of_potatoes :
  k = 2 → total_given_away = initial_potatoes → given_to_tom / given_to_gina = 2 := by
  intros h1 h2
  sorry

end ratio_of_potatoes_l1554_155411


namespace min_value_AF_BF_l1554_155429

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 1

theorem min_value_AF_BF :
  ∀ (x1 x2 y1 y2 k : ℝ),
  parabola_eq x1 y1 →
  parabola_eq x2 y2 →
  line_eq k x1 = y1 →
  line_eq k x2 = y2 →
  (x1 ≠ x2) →
  parabola_focus = (0, 1) →
  (|y1 + 2| + 1) * (|y2 + 1|) = 2 * Real.sqrt 2 + 3 := 
by
  intros
  sorry

end min_value_AF_BF_l1554_155429


namespace xy_square_sum_l1554_155409

theorem xy_square_sum (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 :=
by
  sorry

end xy_square_sum_l1554_155409


namespace expected_value_of_X_is_5_over_3_l1554_155455

-- Define the probabilities of getting an interview with company A, B, and C
def P_A : ℚ := 2 / 3
def P_BC (p : ℚ) : ℚ := p

-- Define the random variable X representing the number of interview invitations
def X (P_A P_BC : ℚ) : ℚ := sorry

-- Define the probability of receiving no interview invitations
def P_X_0 (P_A P_BC : ℚ) : ℚ := (1 - P_A) * (1 - P_BC)^2

-- Given condition that P(X=0) is 1/12
def condition_P_X_0 (P_A P_BC : ℚ) : Prop := P_X_0 P_A P_BC = 1 / 12

-- Given p = 1/2 as per the problem solution
def p : ℚ := 1 / 2

-- Expected value of X
def E_X (P_A P_BC : ℚ) : ℚ := (1 * (2 * P_BC * (1 - P_BC) + 2 * P_BC^2 * (1 - P_BC) + (1 - P_A) * P_BC^2)) +
                               (2 * (P_A * P_BC * (1 - P_BC) + P_A * (1 - P_BC)^2 + P_BC * P_BC * (1 - P_A))) +
                               (3 * (P_A * P_BC^2))

-- Theorem proving the expected value of X given the above conditions
theorem expected_value_of_X_is_5_over_3 : E_X P_A (P_BC p) = 5 / 3 :=
by
  -- here you will write the proof later
  sorry

end expected_value_of_X_is_5_over_3_l1554_155455


namespace johns_number_l1554_155410

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number_l1554_155410


namespace remainder_2023_div_73_l1554_155446

theorem remainder_2023_div_73 : 2023 % 73 = 52 := 
by
  -- Proof goes here
  sorry

end remainder_2023_div_73_l1554_155446


namespace sum_of_28_terms_l1554_155454

variable {f : ℝ → ℝ}
variable {a : ℕ → ℝ}

noncomputable def sum_arithmetic_sequence (n : ℕ) (a1 d : ℝ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_of_28_terms
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h2 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y)
  (h3 : ∃ d ≠ 0, ∃ a₁, ∀ n, a (n + 1) = a₁ + n * d)
  (h4 : f (a 6) = f (a 23)) :
  sum_arithmetic_sequence 28 (a 1) ((a 2) - (a 1)) = 28 :=
by sorry

end sum_of_28_terms_l1554_155454


namespace subset_range_a_l1554_155433

def setA : Set ℝ := { x | (x^2 - 4 * x + 3) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (2^(1 - x) + a) ≤ 0 ∧ (x^2 - 2*(a + 7)*x + 5) ≤ 0 }

theorem subset_range_a (a : ℝ) : setA ⊆ setB a ↔ -4 ≤ a ∧ a ≤ -1 := 
  sorry

end subset_range_a_l1554_155433


namespace sand_loss_l1554_155452

variable (initial_sand : ℝ) (final_sand : ℝ)

theorem sand_loss (h1 : initial_sand = 4.1) (h2 : final_sand = 1.7) :
  initial_sand - final_sand = 2.4 := by
  -- With the given conditions we'll prove this theorem
  sorry

end sand_loss_l1554_155452


namespace trees_planted_l1554_155481

theorem trees_planted (current_short_trees planted_short_trees total_short_trees : ℕ)
  (h1 : current_short_trees = 112)
  (h2 : total_short_trees = 217) :
  planted_short_trees = 105 :=
by
  sorry

end trees_planted_l1554_155481


namespace multiply_inequalities_positive_multiply_inequalities_negative_l1554_155421

variable {a b c d : ℝ}

theorem multiply_inequalities_positive (h₁ : a > b) (h₂ : c > d) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) (h₆ : 0 < d) :
  a * c > b * d :=
sorry

theorem multiply_inequalities_negative (h₁ : a < b) (h₂ : c < d) (h₃ : a < 0) (h₄ : b < 0) (h₅ : c < 0) (h₆ : d < 0) :
  a * c > b * d :=
sorry

end multiply_inequalities_positive_multiply_inequalities_negative_l1554_155421


namespace sum_f_inv_l1554_155449

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 2 * x - 1 else x ^ 2

noncomputable def f_inv (y : ℝ) : ℝ :=
if y < 9 then (y + 1) / 2 else Real.sqrt y

theorem sum_f_inv :
  (f_inv (-3) + f_inv (-2) + 
   f_inv (-1) + f_inv 0 + 
   f_inv 1 + f_inv 2 + 
   f_inv 3 + f_inv 4 + 
   f_inv 9) = 9 :=
by
  sorry

end sum_f_inv_l1554_155449


namespace geometric_prod_eight_l1554_155465

theorem geometric_prod_eight
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith : ∀ n, a n ≠ 0)
  (h_eq : a 4 + 3 * a 8 = 2 * (a 7)^2)
  (h_geom : ∀ {m n : ℕ}, b m * b (m + n) = b (2 * m + n))
  (h_b_eq_a : b 7 = a 7) :
  b 2 * b 8 * b 11 = 8 :=
sorry

end geometric_prod_eight_l1554_155465


namespace range_of_y_l1554_155474

theorem range_of_y (x y : ℝ) (h1 : |y - 2 * x| = x^2) (h2 : -1 < x) (h3 : x < 0) : -3 < y ∧ y < 0 :=
by
  sorry

end range_of_y_l1554_155474


namespace point_M_coordinates_l1554_155453

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 4 * x

-- Define the condition given in the problem: instantaneous rate of change
def rate_of_change (a : ℝ) : Prop := f' a = -4

-- Define the point on the curve
def point_M (a b : ℝ) : Prop := f a = b

-- Proof statement
theorem point_M_coordinates : 
  ∃ (a b : ℝ), rate_of_change a ∧ point_M a b ∧ a = -1 ∧ b = 3 :=  
by
  sorry

end point_M_coordinates_l1554_155453


namespace meal_combinations_l1554_155417

theorem meal_combinations :
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  meats * vegetable_combinations * desserts = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  show meats * vegetable_combinations * desserts = 150
  sorry

end meal_combinations_l1554_155417


namespace find_prism_height_l1554_155401

variables (base_side_length : ℝ) (density : ℝ) (weight : ℝ) (height : ℝ)

-- Assume the base_side_length is 2 meters, density is 2700 kg/m³, and weight is 86400 kg
def given_conditions := (base_side_length = 2) ∧ (density = 2700) ∧ (weight = 86400)

-- Define the volume based on weight and density
noncomputable def volume (density weight : ℝ) : ℝ := weight / density

-- Define the area of the base
def base_area (side_length : ℝ) : ℝ := side_length * side_length

-- Define the height of the prism
noncomputable def prism_height (volume base_area : ℝ) : ℝ := volume / base_area

-- The proof statement
theorem find_prism_height (h : ℝ) : given_conditions base_side_length density weight → prism_height (volume density weight) (base_area base_side_length) = h :=
by
  intros h_cond
  sorry

end find_prism_height_l1554_155401


namespace gcd_n_cube_plus_25_n_plus_3_l1554_155478

theorem gcd_n_cube_plus_25_n_plus_3 (n : ℕ) (h : n > 3^2) : 
  Int.gcd (n^3 + 25) (n + 3) = if n % 2 = 1 then 2 else 1 :=
by
  sorry

end gcd_n_cube_plus_25_n_plus_3_l1554_155478


namespace unique_solution_abs_eq_l1554_155482

theorem unique_solution_abs_eq : 
  ∃! x : ℝ, |x - 1| = |x - 2| + |x + 3| + 1 :=
by
  use -5
  sorry

end unique_solution_abs_eq_l1554_155482


namespace prove_expression_l1554_155443

theorem prove_expression (a b : ℕ) 
  (h1 : 180 % 2^a = 0 ∧ 180 % 2^(a+1) ≠ 0)
  (h2 : 180 % 3^b = 0 ∧ 180 % 3^(b+1) ≠ 0) :
  (1 / 4 : ℚ)^(b - a) = 1 := 
sorry

end prove_expression_l1554_155443


namespace find_f_l1554_155486

theorem find_f (f : ℤ → ℤ) (h : ∀ n : ℤ, n^2 + 4 * (f n) = (f (f n))^2) :
  (∀ x : ℤ, f x = 1 + x) ∨
  (∃ a : ℤ, (∀ x ≤ a, f x = 1 - x) ∧ (∀ x > a, f x = 1 + x)) ∨
  (f 0 = 0 ∧ (∀ x < 0, f x = 1 - x) ∧ (∀ x > 0, f x = 1 + x)) :=
sorry

end find_f_l1554_155486


namespace problem_solution_l1554_155468

theorem problem_solution (a b d : ℤ) (ha : a = 2500) (hb : b = 2409) (hd : d = 81) :
  (a - b) ^ 2 / d = 102 := by
  sorry

end problem_solution_l1554_155468


namespace black_white_tile_ratio_l1554_155463

theorem black_white_tile_ratio :
  let original_black_tiles := 10
  let original_white_tiles := 15
  let total_tiles_in_original_square := original_black_tiles + original_white_tiles
  let side_length_of_original_square := Int.sqrt total_tiles_in_original_square -- this should be 5
  let side_length_of_extended_square := side_length_of_original_square + 2
  let total_black_tiles_in_border := 4 * (side_length_of_extended_square - 1) / 2 -- Each border side starts and ends with black
  let total_white_tiles_in_border := (side_length_of_extended_square * 4 - 4) - total_black_tiles_in_border 
  let new_total_black_tiles := original_black_tiles + total_black_tiles_in_border
  let new_total_white_tiles := original_white_tiles + total_white_tiles_in_border
  (new_total_black_tiles / gcd new_total_black_tiles new_total_white_tiles) / 
  (new_total_white_tiles / gcd new_total_black_tiles new_total_white_tiles) = 26 / 23 :=
by
  sorry

end black_white_tile_ratio_l1554_155463


namespace percentage_exceeds_self_l1554_155425

theorem percentage_exceeds_self (N : ℕ) (P : ℝ) (h1 : N = 150) (h2 : N = (P / 100) * N + 126) : P = 16 := by
  sorry

end percentage_exceeds_self_l1554_155425


namespace average_cookies_per_package_l1554_155456

def cookie_counts : List ℕ := [9, 11, 13, 15, 15, 17, 19, 21, 5]

theorem average_cookies_per_package :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 125 / 9 :=
by
  sorry

end average_cookies_per_package_l1554_155456


namespace polynomial_roots_l1554_155461

theorem polynomial_roots : ∀ x : ℝ, 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0 :=
by
  sorry

end polynomial_roots_l1554_155461


namespace simplify_and_evaluate_expr_l1554_155407

theorem simplify_and_evaluate_expr (x : Real) (h : x = Real.sqrt 3 - 1) :
  1 - (x / (x + 1)) / (x / (x ^ 2 - 1)) = 3 - Real.sqrt 3 :=
sorry

end simplify_and_evaluate_expr_l1554_155407


namespace resulting_ratio_correct_l1554_155466

-- Define initial conditions
def initial_coffee : ℕ := 20
def joe_drank : ℕ := 3
def joe_added_cream : ℕ := 4
def joAnn_added_cream : ℕ := 3
def joAnn_drank : ℕ := 4

-- Define the resulting amounts of cream
def joe_cream : ℕ := joe_added_cream
def joAnn_initial_cream_frac : ℚ := joAnn_added_cream / (initial_coffee + joAnn_added_cream)
def joAnn_cream_drank : ℚ := (joAnn_drank : ℚ) * joAnn_initial_cream_frac
def joAnn_cream_left : ℚ := joAnn_added_cream - joAnn_cream_drank

-- Define the resulting ratio of cream in Joe's coffee to JoAnn's coffee
def resulting_ratio : ℚ := joe_cream / joAnn_cream_left

-- Theorem stating the resulting ratio is 92/45
theorem resulting_ratio_correct : resulting_ratio = 92 / 45 :=
by
  unfold resulting_ratio joe_cream joAnn_cream_left joAnn_cream_drank joAnn_initial_cream_frac
  norm_num
  sorry

end resulting_ratio_correct_l1554_155466


namespace find_initial_order_l1554_155457

variables (x : ℕ)

def initial_order (x : ℕ) :=
  x + 60 = 72 * (x / 90 + 1)

theorem find_initial_order (h1 : initial_order x) : x = 60 :=
  sorry

end find_initial_order_l1554_155457


namespace range_of_x_l1554_155416

-- Define the problem conditions and the conclusion to be proved
theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y)
  (h_ineq : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, 1 ≤ x ∧ x < 3 / 2 :=
by
  sorry

end range_of_x_l1554_155416


namespace min_isosceles_triangle_area_l1554_155492

theorem min_isosceles_triangle_area 
  (x y n : ℕ)
  (h1 : 2 * x * y = 7 * n^2)
  (h2 : ∃ m k, m = n / 2 ∧ k = 2 * m) 
  (h3 : n % 3 = 0) : 
  x = 4 * n / 3 ∧ y = n / 3 ∧ 
  ∃ A, A = 21 / 4 := 
sorry

end min_isosceles_triangle_area_l1554_155492


namespace system_of_equations_solution_l1554_155495

theorem system_of_equations_solution 
  (x y z : ℤ) 
  (h1 : x^2 - y - z = 8) 
  (h2 : 4 * x + y^2 + 3 * z = -11) 
  (h3 : 2 * x - 3 * y + z^2 = -11) : 
  x = -3 ∧ y = 2 ∧ z = -1 :=
sorry

end system_of_equations_solution_l1554_155495


namespace find_formula_l1554_155405

variable (x : ℕ) (y : ℕ)

theorem find_formula (h1: (x = 2 ∧ y = 10) ∨ (x = 3 ∧ y = 21) ∨ (x = 4 ∧ y = 38) ∨ (x = 5 ∧ y = 61) ∨ (x = 6 ∧ y = 90)) :
  y = 3 * x^2 - 2 * x + 2 :=
  sorry

end find_formula_l1554_155405


namespace max_t_eq_one_l1554_155402

theorem max_t_eq_one {x y : ℝ} (hx : x > 0) (hy : y > 0) : 
  max (min x (y / (x^2 + y^2))) 1 = 1 :=
sorry

end max_t_eq_one_l1554_155402


namespace total_students_in_high_school_l1554_155426

theorem total_students_in_high_school (sample_size first_year third_year second_year : ℕ) (total_students : ℕ) 
  (h1 : sample_size = 45) 
  (h2 : first_year = 20) 
  (h3 : third_year = 10) 
  (h4 : second_year = 300)
  (h5 : sample_size = first_year + third_year + (sample_size - first_year - third_year)) :
  total_students = 900 :=
by
  sorry

end total_students_in_high_school_l1554_155426


namespace yield_percentage_of_stock_is_8_percent_l1554_155441

theorem yield_percentage_of_stock_is_8_percent :
  let face_value := 100
  let dividend_rate := 0.20
  let market_price := 250
  annual_dividend = dividend_rate * face_value →
  yield_percentage = (annual_dividend / market_price) * 100 →
  yield_percentage = 8 := 
by
  sorry

end yield_percentage_of_stock_is_8_percent_l1554_155441


namespace students_sampled_from_schoolB_l1554_155464

-- Definitions from the conditions in a)
def schoolA_students := 800
def schoolB_students := 500
def total_students := schoolA_students + schoolB_students
def schoolA_sampled_students := 48

-- Mathematically equivalent proof problem
theorem students_sampled_from_schoolB : 
  let proportionA := (schoolA_students : ℝ) / total_students
  let proportionB := (schoolB_students : ℝ) / total_students
  let total_sampled_students := schoolA_sampled_students / proportionA
  let b_sampled_students := proportionB * total_sampled_students
  b_sampled_students = 30 :=
by
  -- Placeholder for the actual proof
  sorry

end students_sampled_from_schoolB_l1554_155464


namespace scientific_notation_of_area_l1554_155459

theorem scientific_notation_of_area : 2720000 = 2.72 * 10^6 :=
by
  sorry

end scientific_notation_of_area_l1554_155459


namespace rectangle_perimeter_l1554_155434

theorem rectangle_perimeter (A W : ℝ) (hA : A = 300) (hW : W = 15) : 
  (2 * ((A / W) + W)) = 70 := 
  sorry

end rectangle_perimeter_l1554_155434


namespace problem_statement_l1554_155450

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def x : ℝ := alpha ^ 1000
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l1554_155450


namespace problem_statement_l1554_155406

noncomputable def r (C: ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def A (r: ℝ) : ℝ := Real.pi * r^2

noncomputable def combined_area_difference (C1 C2 C3: ℝ) : ℝ :=
  let r1 := r C1
  let r2 := r C2
  let r3 := r C3
  let A1 := A r1
  let A2 := A r2
  let A3 := A r3
  (A3 - A1) - A2

theorem problem_statement : combined_area_difference 528 704 880 = -9.76 :=
by
  sorry

end problem_statement_l1554_155406


namespace min_value_a_l1554_155418

theorem min_value_a (a b : ℕ) (h1: a = b - 2005) 
  (h2: ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p + q = a ∧ p * q = b) : a ≥ 95 := sorry

end min_value_a_l1554_155418


namespace sum_of_remainders_mod_8_l1554_155498

theorem sum_of_remainders_mod_8 
  (x y z w : ℕ)
  (hx : x % 8 = 3)
  (hy : y % 8 = 5)
  (hz : z % 8 = 7)
  (hw : w % 8 = 1) :
  (x + y + z + w) % 8 = 0 :=
by
  sorry

end sum_of_remainders_mod_8_l1554_155498


namespace chocolates_initial_count_l1554_155475

theorem chocolates_initial_count : 
  ∀ (chocolates_first_day chocolates_second_day chocolates_third_day chocolates_fourth_day chocolates_fifth_day initial_chocolates : ℕ),
  chocolates_first_day = 4 →
  chocolates_second_day = 2 * chocolates_first_day - 3 →
  chocolates_third_day = chocolates_first_day - 2 →
  chocolates_fourth_day = chocolates_third_day - 1 →
  chocolates_fifth_day = 12 →
  initial_chocolates = chocolates_first_day + chocolates_second_day + chocolates_third_day + chocolates_fourth_day + chocolates_fifth_day →
  initial_chocolates = 24 :=
by {
  -- the proof will go here,
  sorry
}

end chocolates_initial_count_l1554_155475


namespace bushes_for_zucchinis_l1554_155424

def bushes_yield := 10 -- containers per bush
def container_to_zucchini := 3 -- containers per zucchini
def zucchinis_required := 60 -- total zucchinis needed

theorem bushes_for_zucchinis (hyld : bushes_yield = 10) (ctz : container_to_zucchini = 3) (zreq : zucchinis_required = 60) :
  ∃ bushes : ℕ, bushes = 60 * container_to_zucchini / bushes_yield :=
sorry

end bushes_for_zucchinis_l1554_155424


namespace quadratic_expression_value_l1554_155430

variable (x y : ℝ)

theorem quadratic_expression_value (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 100 := 
by 
  sorry

end quadratic_expression_value_l1554_155430


namespace min_value_seq_div_n_l1554_155490

-- Definitions of the conditions
def a_seq (n : ℕ) : ℕ := 
  if n = 0 then 0 else if n = 1 then 98 else 102 + (n - 2) * (2 * n + 2)

-- The property we need to prove
theorem min_value_seq_div_n :
  (∀ n : ℕ, (n ≥ 1) → (a_seq n / n) ≥ 26) ∧ (∃ n : ℕ, (n ≥ 1) ∧ (a_seq n / n) = 26) :=
sorry

end min_value_seq_div_n_l1554_155490


namespace project_hours_l1554_155458

variable (K : ℕ)

theorem project_hours 
    (h_total : K + 2 * K + 3 * K + K / 2 = 180)
    (h_k_nearest : K = 28) :
    3 * K - K = 56 := 
by
  -- Proof goes here
  sorry

end project_hours_l1554_155458


namespace find_quotient_l1554_155442

-- Define the given conditions
def dividend : ℤ := 144
def divisor : ℤ := 11
def remainder : ℤ := 1

-- Define the quotient logically derived from the given conditions
def quotient : ℤ := dividend / divisor

-- The theorem we need to prove
theorem find_quotient : quotient = 13 := by
  sorry

end find_quotient_l1554_155442


namespace man_savings_percentage_l1554_155477

theorem man_savings_percentage
  (salary expenses : ℝ)
  (increase_percentage : ℝ)
  (current_savings : ℝ)
  (P : ℝ)
  (h1 : salary = 7272.727272727273)
  (h2 : increase_percentage = 0.05)
  (h3 : current_savings = 400)
  (h4 : current_savings + (increase_percentage * salary) = (P / 100) * salary) :
  P = 10.5 := 
sorry

end man_savings_percentage_l1554_155477


namespace ratio_city_XY_l1554_155476

variable (popZ popY popX : ℕ)

-- Definition of the conditions
def condition1 := popY = 2 * popZ
def condition2 := popX = 16 * popZ

-- The goal to prove
theorem ratio_city_XY 
  (h1 : condition1 popY popZ)
  (h2 : condition2 popX popZ) :
  popX / popY = 8 := 
  by sorry

end ratio_city_XY_l1554_155476


namespace range_of_a_l1554_155484

-- Definitions based on conditions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 4

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≤ 4 → f a x ≤ f a 4) → a ≤ -3 :=
by
  sorry

end range_of_a_l1554_155484


namespace find_function_l1554_155479

theorem find_function (f : ℚ → ℚ) (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_function_l1554_155479


namespace cylinder_volume_l1554_155460

noncomputable def volume_cylinder (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) : ℝ :=
  let ratio_r := r_cylinder / r_cone
  let ratio_h := h_cylinder / h_cone
  (3 : ℝ) * ratio_r^2 * ratio_h * V_cone

theorem cylinder_volume (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) :
    r_cylinder / r_cone = 2 / 3 →
    h_cylinder / h_cone = 4 / 3 →
    V_cone = 5.4 →
    volume_cylinder V_cone r_cylinder r_cone h_cylinder h_cone = 3.2 :=
by
  intros h1 h2 h3
  rw [volume_cylinder, h1, h2, h3]
  sorry

end cylinder_volume_l1554_155460


namespace sum_of_angles_l1554_155422

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (sin_α : Real.sin α = 2 * Real.sqrt 5 / 5) (sin_beta : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_angles_l1554_155422


namespace solve_percentage_chromium_first_alloy_l1554_155470

noncomputable def percentage_chromium_first_alloy (x : ℝ) : Prop :=
  let w1 := 15 -- weight of the first alloy
  let c2 := 10 -- percentage of chromium in the second alloy
  let w2 := 35 -- weight of the second alloy
  let w_total := 50 -- total weight of the new alloy formed by mixing
  let c_new := 10.6 -- percentage of chromium in the new alloy
  -- chromium percentage equation
  ((x / 100) * w1 + (c2 / 100) * w2) = (c_new / 100) * w_total

theorem solve_percentage_chromium_first_alloy : percentage_chromium_first_alloy 12 :=
  sorry -- proof goes here

end solve_percentage_chromium_first_alloy_l1554_155470


namespace locus_equation_of_points_at_distance_2_from_line_l1554_155497

theorem locus_equation_of_points_at_distance_2_from_line :
  {P : ℝ × ℝ | abs ((3 / 5) * P.1 - (4 / 5) * P.2 - (1 / 5)) = 2} =
    {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 - 11 = 0} ∪ {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 + 9 = 0} :=
by
  -- Proof goes here
  sorry

end locus_equation_of_points_at_distance_2_from_line_l1554_155497


namespace price_decrease_for_original_price_l1554_155403

theorem price_decrease_for_original_price (P : ℝ) (h : P > 0) :
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  decrease = 20 :=
by
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  sorry

end price_decrease_for_original_price_l1554_155403


namespace first_negative_term_at_14_l1554_155483

-- Define the n-th term of the arithmetic sequence
def a_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

-- Given values
def a₁ := 51
def d := -4

-- Proof statement
theorem first_negative_term_at_14 : ∃ n : ℕ, a_n a₁ d n < 0 ∧ ∀ m < n, a_n a₁ d m ≥ 0 :=
  by sorry

end first_negative_term_at_14_l1554_155483


namespace max_value_HMMT_l1554_155400

theorem max_value_HMMT :
  ∀ (H M T : ℤ), H * M ^ 2 * T = H + 2 * M + T → H * M ^ 2 * T ≤ 8 :=
by
  sorry

end max_value_HMMT_l1554_155400


namespace quadratic_unique_real_root_l1554_155408

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l1554_155408


namespace ball_hit_ground_in_time_l1554_155491

theorem ball_hit_ground_in_time :
  ∃ t : ℝ, t ≥ 0 ∧ -16 * t^2 - 30 * t + 180 = 0 ∧ t = 1.25 :=
by sorry

end ball_hit_ground_in_time_l1554_155491


namespace probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l1554_155414

open BigOperators

/-- Suppose 30 balls are tossed independently and at random into one 
of the 6 bins. Let p be the probability that one bin ends up with 3 
balls, another with 6 balls, another with 5, another with 4, another 
with 2, and the last one with 10 balls. Let q be the probability 
that each bin ends up with 5 balls. Calculate p / q. 
-/
theorem probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5 :
  (Nat.factorial 5 ^ 6 : ℚ) / ((Nat.factorial 3:ℚ) * Nat.factorial 6 * Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 10) = 0.125 := 
sorry

end probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l1554_155414
