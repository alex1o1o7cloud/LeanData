import Mathlib

namespace equilateral_triangle_sum_l2175_217542

noncomputable def equilateral_triangle (a b c : Complex) (s : ℝ) : Prop :=
  Complex.abs (a - b) = s ∧ Complex.abs (b - c) = s ∧ Complex.abs (c - a) = s

theorem equilateral_triangle_sum (a b c : Complex):
  equilateral_triangle a b c 18 →
  Complex.abs (a + b + c) = 36 →
  Complex.abs (b * c + c * a + a * b) = 432 := by
  intros h_triangle h_sum
  sorry

end equilateral_triangle_sum_l2175_217542


namespace expected_number_of_own_hats_l2175_217505

-- Define the number of people
def num_people : ℕ := 2015

-- Define the expectation based on the problem description
noncomputable def expected_hats (n : ℕ) : ℝ := 1

-- The main theorem representing the problem statement
theorem expected_number_of_own_hats : expected_hats num_people = 1 := sorry

end expected_number_of_own_hats_l2175_217505


namespace correct_answers_count_l2175_217579

theorem correct_answers_count
  (c w : ℕ)
  (h1 : 4 * c - 2 * w = 420)
  (h2 : c + w = 150) : 
  c = 120 :=
sorry

end correct_answers_count_l2175_217579


namespace graph_passes_quadrants_l2175_217553

theorem graph_passes_quadrants (a b : ℝ) (h_a : 1 < a) (h_b : -1 < b ∧ b < 0) : 
    ∀ x : ℝ, (0 < a^x + b ∧ x > 0) ∨ (a^x + b < 0 ∧ x < 0) ∨ (0 < x ∧ a^x + b = 0) → x ≠ 0 ∧ 0 < x :=
sorry

end graph_passes_quadrants_l2175_217553


namespace cooking_dishes_time_l2175_217597

def total_awake_time : ℝ := 16
def work_time : ℝ := 8
def gym_time : ℝ := 2
def bath_time : ℝ := 0.5
def homework_bedtime_time : ℝ := 1
def packing_lunches_time : ℝ := 0.5
def cleaning_time : ℝ := 0.5
def shower_leisure_time : ℝ := 2
def total_allocated_time : ℝ := work_time + gym_time + bath_time + homework_bedtime_time + packing_lunches_time + cleaning_time + shower_leisure_time

theorem cooking_dishes_time : total_awake_time - total_allocated_time = 1.5 := by
  sorry

end cooking_dishes_time_l2175_217597


namespace sequence_properties_l2175_217516

-- Define the sequence formula
def a_n (n : ℤ) : ℤ := n^2 - 5 * n + 4

-- State the theorem about the sequence
theorem sequence_properties :
  -- Part 1: The number of negative terms in the sequence
  (∃ (S : Finset ℤ), ∀ n ∈ S, a_n n < 0 ∧ S.card = 2) ∧
  -- Part 2: The minimum value of the sequence and the value of n at minimum
  (∀ n : ℤ, (a_n n ≥ -9 / 4) ∧ (a_n (5 / 2) = -9 / 4)) :=
by {
  sorry
}

end sequence_properties_l2175_217516


namespace apples_more_than_grapes_l2175_217522

theorem apples_more_than_grapes 
  (total_weight : ℕ) (weight_ratio_apples : ℕ) (weight_ratio_peaches : ℕ) (weight_ratio_grapes : ℕ) : 
  weight_ratio_apples = 12 → 
  weight_ratio_peaches = 8 → 
  weight_ratio_grapes = 7 → 
  total_weight = 54 →
  ((12 * total_weight / (12 + 8 + 7)) - (7 * total_weight / (12 + 8 + 7))) = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end apples_more_than_grapes_l2175_217522


namespace find_A_minus_B_l2175_217536

def A : ℤ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℤ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem find_A_minus_B : A - B = 128 := 
by
  -- Proof goes here
  sorry

end find_A_minus_B_l2175_217536


namespace solve_inequality_l2175_217548

open Set Real

theorem solve_inequality (x : ℝ) : { x : ℝ | x^2 - 4 * x > 12 } = {x : ℝ | x < -2} ∪ {x : ℝ | 6 < x} := 
sorry

end solve_inequality_l2175_217548


namespace exterior_angle_of_regular_pentagon_l2175_217586

theorem exterior_angle_of_regular_pentagon : 
  (360 / 5) = 72 := by
  sorry

end exterior_angle_of_regular_pentagon_l2175_217586


namespace arithmetic_sequence_common_difference_l2175_217582

theorem arithmetic_sequence_common_difference 
    (a_2 : ℕ → ℕ) (S_4 : ℕ) (a_n : ℕ → ℕ → ℕ) (S_n : ℕ → ℕ → ℕ → ℕ)
    (h1 : a_2 2 = 3) (h2 : S_4 = 16) 
    (h3 : ∀ n a_1 d, a_n a_1 n = a_1 + (n-1)*d)
    (h4 : ∀ n a_1 d, S_n n a_1 d = n / 2 * (2*a_1 + (n-1)*d)) : ∃ d, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l2175_217582


namespace area_of_circle_l2175_217519

open Real

theorem area_of_circle :
  ∃ (A : ℝ), (∀ x y : ℝ, (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) → A = 16 * π) :=
sorry

end area_of_circle_l2175_217519


namespace find_Y_l2175_217594

theorem find_Y (Y : ℝ) (h : (100 + Y / 90) * 90 = 9020) : Y = 20 := 
by 
  sorry

end find_Y_l2175_217594


namespace students_in_game_divisors_of_119_l2175_217527

theorem students_in_game_divisors_of_119 (n : ℕ) (h1 : ∃ (k : ℕ), k * n = 119) :
  n = 7 ∨ n = 17 :=
sorry

end students_in_game_divisors_of_119_l2175_217527


namespace solve_system_of_equations_l2175_217528

theorem solve_system_of_equations (a b c x y z : ℝ)
  (h1 : a^3 + a^2 * x + a * y + z = 0)
  (h2 : b^3 + b^2 * x + b * y + z = 0)
  (h3 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + ac + bc ∧ z = -abc :=
by {
  sorry
}

end solve_system_of_equations_l2175_217528


namespace H2O_formed_l2175_217507

-- Definition of the balanced chemical equation
def balanced_eqn : Prop :=
  ∀ (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ), HCH3CO2 + NaOH = NaCH3CO2 + H2O

-- Statement of the problem
theorem H2O_formed (HCH3CO2 NaOH NaCH3CO2 H2O : ℕ) 
  (h1 : HCH3CO2 = 1)
  (h2 : NaOH = 1)
  (balanced : balanced_eqn):
  H2O = 1 :=
by sorry

end H2O_formed_l2175_217507


namespace mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l2175_217576

def card_is_heart (c : ℕ) := c ≥ 1 ∧ c ≤ 13

def card_is_diamond (c : ℕ) := c ≥ 14 ∧ c ≤ 26

def card_is_red (c : ℕ) := c ≥ 1 ∧ c ≤ 26

def card_is_black (c : ℕ) := c ≥ 27 ∧ c ≤ 52

def card_is_face_234610 (c : ℕ) := c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 6 ∨ c = 10

def card_is_face_2345678910 (c : ℕ) :=
  c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6 ∨ c = 7 ∨ c = 8 ∨ c = 9 ∨ c = 10

def card_is_face_AKQJ (c : ℕ) :=
  c = 1 ∨ c = 11 ∨ c = 12 ∨ c = 13

def card_is_ace_king_queen_jack (c : ℕ) := c = 1 ∨ (c ≥ 11 ∧ c ≤ 13)

theorem mutually_exclusive_pair2 : ∀ c : ℕ, card_is_red c ≠ card_is_black c := by
  sorry

theorem complementary_pair2 : ∀ c : ℕ, card_is_red c ∨ card_is_black c := by
  sorry

theorem mutually_exclusive_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ≠ card_is_ace_king_queen_jack c := by
  sorry

theorem complementary_pair4 : ∀ c : ℕ, card_is_face_2345678910 c ∨ card_is_ace_king_queen_jack c := by
  sorry

end mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l2175_217576


namespace relationship_abcd_l2175_217554

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem relationship_abcd : b < a ∧ a < d ∧ d < c := by
  sorry

end relationship_abcd_l2175_217554


namespace problem_statement_l2175_217569

/-- For any positive integer n, given θ ∈ (0, π) and x ∈ ℂ such that 
x + 1/x = 2√2 cos θ - sin θ, it follows that x^n + 1/x^n = 2 cos (n α). -/
theorem problem_statement (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π)
  (x : ℂ) (hx : x + 1/x = 2 * (2:ℝ).sqrt * θ.cos - θ.sin)
  (n : ℕ) (hn : 0 < n) : x^n + x⁻¹^n = 2 * θ.cos * n := 
  sorry

end problem_statement_l2175_217569


namespace max_correct_answers_l2175_217547

theorem max_correct_answers (a b c : ℕ) (n : ℕ := 60) (p_correct : ℤ := 5) (p_blank : ℤ := 0) (p_incorrect : ℤ := -2) (S : ℤ := 150) :
        a + b + c = n ∧ p_correct * a + p_blank * b + p_incorrect * c = S → a ≤ 38 :=
by
  sorry

end max_correct_answers_l2175_217547


namespace Mitya_age_l2175_217575

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end Mitya_age_l2175_217575


namespace min_max_sum_eq_one_l2175_217540

theorem min_max_sum_eq_one 
  (x : ℕ → ℝ)
  (h_nonneg : ∀ i, 0 ≤ x i)
  (h_sum_eq_one : (x 1 + x 2 + x 3 + x 4 + x 5) = 1) :
  (min (max (x 1 + x 2) (max (x 2 + x 3) (max (x 3 + x 4) (x 4 + x 5)))) = (1 / 3)) :=
by
  sorry

end min_max_sum_eq_one_l2175_217540


namespace rectangle_area_l2175_217520

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area_l2175_217520


namespace inequality_proof_l2175_217559

theorem inequality_proof (a b x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_proof_l2175_217559


namespace percentage_died_by_bombardment_l2175_217539

theorem percentage_died_by_bombardment (P_initial : ℝ) (P_remaining : ℝ) (died_percentage : ℝ) (fear_percentage : ℝ) :
  P_initial = 3161 → P_remaining = 2553 → fear_percentage = 0.15 → 
  P_initial - (died_percentage/100) * P_initial - fear_percentage * (P_initial - (died_percentage/100) * P_initial) = P_remaining → 
  abs (died_percentage - 4.98) < 0.01 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_died_by_bombardment_l2175_217539


namespace moles_of_CaCl2_l2175_217598

theorem moles_of_CaCl2 (HCl moles_of_HCl : ℕ) (CaCO3 moles_of_CaCO3 : ℕ) 
  (reaction : (CaCO3 = 1) → (HCl = 2) → (moles_of_HCl = 6) → (moles_of_CaCO3 = 3)) :
  ∃ moles_of_CaCl2 : ℕ, moles_of_CaCl2 = 3 :=
by
  sorry

end moles_of_CaCl2_l2175_217598


namespace integer_modulo_solution_l2175_217535

theorem integer_modulo_solution (a : ℤ) : 
  (5 ∣ a^3 + 3 * a + 1) ↔ (a % 5 = 1 ∨ a % 5 = 2) := 
by
  exact sorry

end integer_modulo_solution_l2175_217535


namespace initial_number_of_students_l2175_217595

theorem initial_number_of_students (S : ℕ) (h : S + 6 = 37) : S = 31 :=
sorry

end initial_number_of_students_l2175_217595


namespace number_of_people_third_day_l2175_217537

variable (X : ℕ)
variable (total : ℕ := 246)
variable (first_day : ℕ := 79)
variable (second_day_third_day_diff : ℕ := 47)

theorem number_of_people_third_day :
  (first_day + (X + second_day_third_day_diff) + X = total) → 
  X = 60 := by
  sorry

end number_of_people_third_day_l2175_217537


namespace min_value_x3_l2175_217560

noncomputable def min_x3 (x1 x2 x3 : ℝ) : ℝ := -21 / 11

theorem min_value_x3 (x1 x2 x3 : ℝ) 
  (h1 : x1 + (1 / 2) * x2 + (1 / 3) * x3 = 1)
  (h2 : x1^2 + (1 / 2) * x2^2 + (1 / 3) * x3^2 = 3) 
  : x3 ≥ - (21 / 11) := 
by sorry

end min_value_x3_l2175_217560


namespace simplify_expression_l2175_217584

theorem simplify_expression (m n : ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3^(m * n / (m - n))) : 
  ((
    (x^(2 / m) - 9 * x^(2 / n)) *
    ((x^(1 - m))^(1 / m) - 3 * (x^(1 - n))^(1 / n))
  ) / (
    (x^(1 / m) + 3 * x^(1 / n))^2 - 12 * x^((m + n) / (m * n))
  ) = (x^(1 / m) + 3 * x^(1 / n)) / x) := 
sorry

end simplify_expression_l2175_217584


namespace fraction_difference_l2175_217532

theorem fraction_difference :
  (↑(1+4+7) / ↑(2+5+8)) - (↑(2+5+8) / ↑(1+4+7)) = - (9 / 20) :=
by
  sorry

end fraction_difference_l2175_217532


namespace complement_inter_section_l2175_217573

-- Define the sets M and N
def M : Set ℝ := { x | x^2 - 2*x - 3 >= 0 }
def N : Set ℝ := { x | abs (x - 2) <= 1 }

-- Define the complement of M in ℝ
def compl_M : Set ℝ := { x | -1 < x ∧ x < 3 }

-- Define the expected result set
def expected_set : Set ℝ := { x | 1 <= x ∧ x < 3 }

-- State the theorem to prove
theorem complement_inter_section : compl_M ∩ N = expected_set := by
  sorry

end complement_inter_section_l2175_217573


namespace log_cut_problem_l2175_217555

theorem log_cut_problem (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x + 4 * y = 100) :
  2 * x + 3 * y = 70 := by
  sorry

end log_cut_problem_l2175_217555


namespace find_perp_line_eq_l2175_217504

-- Line equation in the standard form
def line_eq (x y : ℝ) : Prop := 4 * x - 3 * y - 12 = 0

-- Equation of the required line that is perpendicular to the given line and has the same y-intercept
def perp_line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 16 = 0

theorem find_perp_line_eq (x y : ℝ) :
  (∃ k : ℝ, line_eq 0 k ∧ perp_line_eq 0 k) →
  (∃ a b c : ℝ, perp_line_eq a b) :=
by
  sorry

end find_perp_line_eq_l2175_217504


namespace min_pq_value_l2175_217534

theorem min_pq_value : 
  ∃ (p q : ℕ), p > 0 ∧ q > 0 ∧ 98 * p = q ^ 3 ∧ (∀ p' q' : ℕ, p' > 0 ∧ q' > 0 ∧ 98 * p' = q' ^ 3 → p' + q' ≥ p + q) ∧ p + q = 42 :=
sorry

end min_pq_value_l2175_217534


namespace total_skateboarding_distance_l2175_217508

def skateboarded_to_park : ℕ := 16
def skateboarded_back_home : ℕ := 9

theorem total_skateboarding_distance : 
  skateboarded_to_park + skateboarded_back_home = 25 := by 
  sorry

end total_skateboarding_distance_l2175_217508


namespace max_levels_passed_prob_pass_three_levels_l2175_217589

-- Define the conditions of the game
def max_roll (n : ℕ) : ℕ := 6 * n
def pass_condition (n : ℕ) : ℕ := 2^n

-- Problem 1: Prove the maximum number of levels a person can pass
theorem max_levels_passed : ∃ n : ℕ, (∀ m : ℕ, m > n → max_roll m ≤ pass_condition m) ∧ (∀ m : ℕ, m ≤ n → max_roll m > pass_condition m) :=
by sorry

-- Define the probabilities for passing each level
def prob_pass_level_1 : ℚ := 4 / 6
def prob_pass_level_2 : ℚ := 30 / 36
def prob_pass_level_3 : ℚ := 160 / 216

-- Problem 2: Prove the probability of passing the first three levels consecutively
theorem prob_pass_three_levels : prob_pass_level_1 * prob_pass_level_2 * prob_pass_level_3 = 100 / 243 :=
by sorry

end max_levels_passed_prob_pass_three_levels_l2175_217589


namespace geometric_sequence_b_l2175_217501

theorem geometric_sequence_b (b : ℝ) (h : b > 0) (s : ℝ) 
  (h1 : 30 * s = b) (h2 : b * s = 15 / 4) : 
  b = 15 * Real.sqrt 2 / 2 := 
by
  sorry

end geometric_sequence_b_l2175_217501


namespace probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l2175_217558

-- Definitions for the conditions
def total_balls : ℕ := 18
def initial_red_balls : ℕ := 12
def initial_white_balls : ℕ := 6
def probability_red_ball : ℚ := initial_red_balls / total_balls
def probability_white_ball_after_removal (x : ℕ) : ℚ := initial_white_balls / (total_balls - x)

-- Statement of the proof problem
theorem probability_red_ball_is_two_thirds : probability_red_ball = 2 / 3 := 
by sorry

theorem red_balls_taken_out_is_three : ∃ x : ℕ, probability_white_ball_after_removal x = 2 / 5 ∧ x = 3 := 
by sorry

end probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l2175_217558


namespace smallest_x_plus_y_l2175_217506

theorem smallest_x_plus_y (x y : ℕ) (h_xy_not_equal : x ≠ y) (h_condition : (1 : ℚ) / x + 1 / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_x_plus_y_l2175_217506


namespace average_speed_of_planes_l2175_217581

-- Definitions for the conditions
def num_passengers_plane1 : ℕ := 50
def num_passengers_plane2 : ℕ := 60
def num_passengers_plane3 : ℕ := 40
def base_speed : ℕ := 600
def speed_reduction_per_passenger : ℕ := 2

-- Calculate speeds of each plane according to given conditions
def speed_plane1 := base_speed - num_passengers_plane1 * speed_reduction_per_passenger
def speed_plane2 := base_speed - num_passengers_plane2 * speed_reduction_per_passenger
def speed_plane3 := base_speed - num_passengers_plane3 * speed_reduction_per_passenger

-- Calculate the total speed and average speed
def total_speed := speed_plane1 + speed_plane2 + speed_plane3
def average_speed := total_speed / 3

-- The theorem to prove the average speed is 500 MPH
theorem average_speed_of_planes : average_speed = 500 := by
  sorry

end average_speed_of_planes_l2175_217581


namespace point_quadrant_I_or_IV_l2175_217544

def is_point_on_line (x y : ℝ) : Prop := 4 * x + 3 * y = 12
def is_equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

def point_in_quadrant_I (x y : ℝ) : Prop := (x > 0 ∧ y > 0)
def point_in_quadrant_IV (x y : ℝ) : Prop := (x > 0 ∧ y < 0)

theorem point_quadrant_I_or_IV (x y : ℝ) 
  (h1 : is_point_on_line x y) 
  (h2 : is_equidistant_from_axes x y) :
  point_in_quadrant_I x y ∨ point_in_quadrant_IV x y :=
sorry

end point_quadrant_I_or_IV_l2175_217544


namespace solve_eq1_solve_eq2_l2175_217509

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l2175_217509


namespace no_rational_solutions_l2175_217557

theorem no_rational_solutions (a b c d : ℚ) (n : ℕ) :
  ¬ ((a + b * (Real.sqrt 2))^(2 * n) + (c + d * (Real.sqrt 2))^(2 * n) = 5 + 4 * (Real.sqrt 2)) :=
sorry

end no_rational_solutions_l2175_217557


namespace value_of_playstation_l2175_217543

theorem value_of_playstation (V : ℝ) (H1 : 700 + 200 = 900) (H2 : V - 0.2 * V = 0.8 * V) (H3 : 0.8 * V = 900 - 580) : V = 400 :=
by
  sorry

end value_of_playstation_l2175_217543


namespace polygon_sides_l2175_217572

theorem polygon_sides (n : ℕ) (f : ℕ) (h1 : f = n * (n - 3) / 2) (h2 : 2 * n = f) : n = 7 :=
  by
  sorry

end polygon_sides_l2175_217572


namespace arithmetic_expression_value_l2175_217530

theorem arithmetic_expression_value : 4 * (8 - 3) - 7 = 13 := by
  sorry

end arithmetic_expression_value_l2175_217530


namespace xiaoyu_money_left_l2175_217562

def box_prices (x y z : ℝ) : Prop :=
  2 * x + 5 * y = z + 3 ∧ 5 * x + 2 * y = z - 3

noncomputable def money_left (x y z : ℝ) : ℝ :=
  z - 7 * x
  
theorem xiaoyu_money_left (x y z : ℝ) (hx : box_prices x y z) :
  money_left x y z = 7 := by
  sorry

end xiaoyu_money_left_l2175_217562


namespace probability_of_choosing_red_base_l2175_217564

theorem probability_of_choosing_red_base (A B : Prop) (C D : Prop) : 
  let red_bases := 2
  let total_bases := 4
  let probability := red_bases / total_bases
  probability = 1 / 2 := 
by
  sorry

end probability_of_choosing_red_base_l2175_217564


namespace min_ball_count_required_l2175_217514

def is_valid_ball_count (n : ℕ) : Prop :=
  n >= 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_list (l : List ℕ) : Prop :=
  ∀ i j, i < l.length → j < l.length → i ≠ j → l.nthLe i sorry ≠ l.nthLe j sorry

def valid_ball_counts_list (l : List ℕ) : Prop :=
  (l.length = 10) ∧ distinct_list l ∧ (∀ n ∈ l, is_valid_ball_count n)

theorem min_ball_count_required : ∃ l, valid_ball_counts_list l ∧ l.sum = 174 := sorry

end min_ball_count_required_l2175_217514


namespace paint_price_max_boxes_paint_A_l2175_217550

theorem paint_price :
  ∃ x y : ℕ, (x + 2 * y = 56 ∧ 2 * x + y = 64) ∧ x = 24 ∧ y = 16 := by
  sorry

theorem max_boxes_paint_A (m : ℕ) :
  24 * m + 16 * (200 - m) ≤ 3920 → m ≤ 90 := by
  sorry

end paint_price_max_boxes_paint_A_l2175_217550


namespace find_radius_of_third_circle_l2175_217599

noncomputable def radius_of_third_circle_equals_shaded_region (r1 r2 r3 : ℝ) : Prop :=
  let area_large := Real.pi * (r2 ^ 2)
  let area_small := Real.pi * (r1 ^ 2)
  let area_shaded := area_large - area_small
  let area_third_circle := Real.pi * (r3 ^ 2)
  area_shaded = area_third_circle

theorem find_radius_of_third_circle (r1 r2 : ℝ) (r1_eq : r1 = 17) (r2_eq : r2 = 27) : ∃ r3 : ℝ, r3 = 10 * Real.sqrt 11 ∧ radius_of_third_circle_equals_shaded_region r1 r2 r3 := 
by
  sorry

end find_radius_of_third_circle_l2175_217599


namespace original_employee_count_l2175_217526

theorem original_employee_count (employees_operations : ℝ) 
                                (employees_sales : ℝ) 
                                (employees_finance : ℝ) 
                                (employees_hr : ℝ) 
                                (employees_it : ℝ) 
                                (h1 : employees_operations / 0.82 = 192)
                                (h2 : employees_sales / 0.75 = 135)
                                (h3 : employees_finance / 0.85 = 123)
                                (h4 : employees_hr / 0.88 = 66)
                                (h5 : employees_it / 0.90 = 90) : 
                                employees_operations + employees_sales + employees_finance + employees_hr + employees_it = 734 :=
sorry

end original_employee_count_l2175_217526


namespace sqrt_sum_of_fractions_l2175_217533

theorem sqrt_sum_of_fractions :
  (Real.sqrt ((25 / 36) + (16 / 9)) = (Real.sqrt 89) / 6) :=
by
  sorry

end sqrt_sum_of_fractions_l2175_217533


namespace problem_solution_A_problem_solution_C_l2175_217571

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l2175_217571


namespace speed_of_current_l2175_217578

theorem speed_of_current (d : ℝ) (c : ℝ) : 
  ∀ (h1 : ∀ (t : ℝ), d = (30 - c) * (40 / 60)) (h2 : ∀ (t : ℝ), d = (30 + c) * (25 / 60)), 
  c = 90 / 13 := by
  sorry

end speed_of_current_l2175_217578


namespace perimeter_of_regular_polygon_l2175_217541

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : n = 3) (h2 : side_length = 5) (h3 : exterior_angle = 120) : 
  n * side_length = 15 :=
by
  sorry

end perimeter_of_regular_polygon_l2175_217541


namespace distinct_three_digit_numbers_count_l2175_217588

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l2175_217588


namespace probability_of_second_ball_white_is_correct_l2175_217521

-- Definitions based on the conditions
def initial_white_balls : ℕ := 8
def initial_black_balls : ℕ := 7
def total_initial_balls : ℕ := initial_white_balls + initial_black_balls
def white_balls_after_first_draw : ℕ := initial_white_balls
def black_balls_after_first_draw : ℕ := initial_black_balls - 1
def total_balls_after_first_draw : ℕ := white_balls_after_first_draw + black_balls_after_first_draw
def probability_second_ball_white : ℚ := white_balls_after_first_draw / total_balls_after_first_draw

-- The proof problem
theorem probability_of_second_ball_white_is_correct :
  probability_second_ball_white = 4 / 7 :=
by
  sorry

end probability_of_second_ball_white_is_correct_l2175_217521


namespace parallel_lines_a_value_l2175_217523

theorem parallel_lines_a_value :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
      (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l2175_217523


namespace no_distinct_nat_numbers_eq_l2175_217518

theorem no_distinct_nat_numbers_eq (x y z t : ℕ) (hxy : x ≠ y) (hxz : x ≠ z) (hxt : x ≠ t) 
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) : x ^ x + y ^ y ≠ z ^ z + t ^ t := 
by 
  sorry

end no_distinct_nat_numbers_eq_l2175_217518


namespace total_baseball_fans_l2175_217561

theorem total_baseball_fans (Y M B : ℕ)
  (h1 : Y = 3 / 2 * M)
  (h2 : M = 88)
  (h3 : B = 5 / 4 * M) :
  Y + M + B = 330 :=
by
  sorry

end total_baseball_fans_l2175_217561


namespace union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l2175_217545

open Set

variable (a : ℝ)

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := univ

theorem union_A_B :
  A ∪ B = {x | 1 ≤ x ∧ x ≤ 8} := by
  sorry

theorem compl_A_inter_B :
  (U \ A) ∩ B = {x | 1 ≤ x ∧ x < 2} := by
  sorry

theorem intersection_A_C_not_empty :
  (A ∩ C a ≠ ∅) → a < 8 := by
  sorry

end union_A_B_compl_A_inter_B_intersection_A_C_not_empty_l2175_217545


namespace pandas_bamboo_consumption_l2175_217531

/-- Given:
  1. An adult panda can eat 138 pounds of bamboo each day.
  2. A baby panda can eat 50 pounds of bamboo a day.
Prove: the total pounds of bamboo eaten by both pandas in a week is 1316 pounds. -/
theorem pandas_bamboo_consumption :
  let adult_daily_bamboo := 138
  let baby_daily_bamboo := 50
  let days_in_week := 7
  (adult_daily_bamboo * days_in_week) + (baby_daily_bamboo * days_in_week) = 1316 := by
  sorry

end pandas_bamboo_consumption_l2175_217531


namespace adam_apples_l2175_217577

theorem adam_apples (x : ℕ) 
  (h1 : 15 + 75 * x = 240) : x = 3 :=
sorry

end adam_apples_l2175_217577


namespace dara_employment_wait_time_l2175_217591

theorem dara_employment_wait_time :
  ∀ (min_age current_jane_age years_later half_age_factor : ℕ), 
  min_age = 25 → 
  current_jane_age = 28 → 
  years_later = 6 → 
  half_age_factor = 2 →
  (min_age - (current_jane_age + years_later) / half_age_factor - years_later) = 14 :=
by
  intros min_age current_jane_age years_later half_age_factor 
  intros h_min_age h_current_jane_age h_years_later h_half_age_factor
  sorry

end dara_employment_wait_time_l2175_217591


namespace four_times_remaining_marbles_l2175_217596

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end four_times_remaining_marbles_l2175_217596


namespace not_exist_three_numbers_l2175_217583

theorem not_exist_three_numbers :
  ¬ ∃ (a b c : ℝ),
  (b^2 - 4 * a * c > 0 ∧ (-b / a > 0) ∧ (c / a > 0)) ∧
  (b^2 - 4 * a * c > 0 ∧ (-b / a < 0) ∧ (c / a > 0)) :=
by
  sorry

end not_exist_three_numbers_l2175_217583


namespace last_two_digits_of_sum_l2175_217551

noncomputable def last_two_digits_sum_factorials : ℕ :=
  let fac : List ℕ := List.map (fun n => Nat.factorial (n * 3)) [1, 2, 3, 4, 5, 6, 7]
  fac.foldl (fun acc x => (acc + x) % 100) 0

theorem last_two_digits_of_sum : last_two_digits_sum_factorials = 6 :=
by
  sorry

end last_two_digits_of_sum_l2175_217551


namespace finite_discrete_points_3_to_15_l2175_217503

def goldfish_cost (n : ℕ) : ℕ := 18 * n

theorem finite_discrete_points_3_to_15 : 
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → 
  ∃ (C : ℕ), C = goldfish_cost n ∧ ∃ (x : ℕ), (n, C) = (x, goldfish_cost x) :=
by
  sorry

end finite_discrete_points_3_to_15_l2175_217503


namespace limit_example_l2175_217502

open Real

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, (∀ x : ℝ, 0 < |x + 4| ∧ |x + 4| < δ → abs ((2 * x^2 + 6 * x - 8) / (x + 4) + 10) < ε) :=
by
  sorry

end limit_example_l2175_217502


namespace log2_sufficient_not_necessary_l2175_217549

noncomputable def baseTwoLog (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log2_sufficient_not_necessary (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (baseTwoLog a > baseTwoLog b) ↔ (a > b) :=
sorry

end log2_sufficient_not_necessary_l2175_217549


namespace integer_solutions_of_system_l2175_217568

theorem integer_solutions_of_system (x y z : ℤ) :
  x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10 ↔ 
  (x = 3 ∧ y = 3 ∧ z = -4) ∨ 
  (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_of_system_l2175_217568


namespace price_36kg_apples_l2175_217574

-- Definitions based on given conditions
def cost_per_kg_first_30 (l : ℕ) (n₁ : ℕ) (total₁ : ℕ) : Prop :=
  n₁ = 10 ∧ l = total₁ / n₁

def total_cost_33kg (l q : ℕ) (total₂ : ℕ) : Prop :=
  30 * l + 3 * q = total₂

-- Question to prove
def total_cost_36kg (l q : ℕ) (cost_36 : ℕ) : Prop :=
  30 * l + 6 * q = cost_36

theorem price_36kg_apples (l q cost_36 : ℕ) :
  (cost_per_kg_first_30 l 10 200) →
  (total_cost_33kg l q 663) →
  cost_36 = 726 :=
by
  intros h₁ h₂
  sorry

end price_36kg_apples_l2175_217574


namespace tan_sum_formula_eq_l2175_217580

theorem tan_sum_formula_eq {θ : ℝ} (h1 : ∃θ, θ ∈ Set.Ico 0 (2 * Real.pi) 
  ∧ ∃P, P = (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4)) 
  ∧ θ = (3 * Real.pi / 4)) : 
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := 
sorry

end tan_sum_formula_eq_l2175_217580


namespace arccos_cos_9_eq_2_717_l2175_217566

-- Statement of the proof problem
theorem arccos_cos_9_eq_2_717 : Real.arccos (Real.cos 9) = 2.717 :=
by
  sorry

end arccos_cos_9_eq_2_717_l2175_217566


namespace initial_mean_correctness_l2175_217585

variable (M : ℝ)

theorem initial_mean_correctness (h1 : 50 * M + 20 = 50 * 36.5) : M = 36.1 :=
by 
  sorry

end initial_mean_correctness_l2175_217585


namespace total_weight_new_group_l2175_217524

variable (W : ℝ) -- Total weight of the original group of 20 people
variable (weights_old : List ℝ) 
variable (weights_new : List ℝ)

-- Given conditions
def five_weights_old : List ℝ := [40, 55, 60, 75, 80]
def average_weight_increase : ℝ := 2
def group_size : ℕ := 20
def num_replaced : ℕ := 5

-- Define theorem
theorem total_weight_new_group :
(W - five_weights_old.sum + group_size * average_weight_increase) -
(W - five_weights_old.sum) = weights_new.sum → 
weights_new.sum = 350 := 
by
  sorry

end total_weight_new_group_l2175_217524


namespace domain_w_l2175_217512

noncomputable def w (y : ℝ) : ℝ := (y - 3)^(1/3) + (15 - y)^(1/3)

theorem domain_w : ∀ y : ℝ, ∃ x : ℝ, w y = x := by
  sorry

end domain_w_l2175_217512


namespace count_books_in_row_on_tuesday_l2175_217517

-- Define the given conditions
def tiles_count_monday : ℕ := 38
def books_count_monday : ℕ := 75
def total_count_tuesday : ℕ := 301
def tiles_count_tuesday := tiles_count_monday * 2

-- The Lean statement we need to prove
theorem count_books_in_row_on_tuesday (hcbooks : books_count_monday = 75) 
(hc1 : total_count_tuesday = 301) 
(hc2 : tiles_count_tuesday = tiles_count_monday * 2):
  (total_count_tuesday - tiles_count_tuesday) / books_count_monday = 3 :=
by
  sorry

end count_books_in_row_on_tuesday_l2175_217517


namespace union_A_B_complement_union_l2175_217567

-- Define \( U \), \( A \), and \( B \)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

-- Define complement in the universe \( U \)
def complement_U (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- Statements to prove
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
  sorry

theorem complement_union : complement_U A ∪ complement_U B = {x | x < 3 ∨ x ≥ 5} :=
  sorry

end union_A_B_complement_union_l2175_217567


namespace range_of_a_same_solution_set_l2175_217570

-- Define the inequality (x-2)(x-5) ≤ 0
def ineq1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the first inequality in the system (x-2)(x-5) ≤ 0
def ineq_system_1 (x : ℝ) : Prop :=
  (x - 2) * (x - 5) ≤ 0

-- Define the second inequality in the system x(x-a) ≥ 0
def ineq_system_2 (x a : ℝ) : Prop :=
  x * (x - a) ≥ 0

-- The final proof statement
theorem range_of_a_same_solution_set (a : ℝ) :
  (∀ x : ℝ, ineq_system_1 x ↔ ineq1 x) →
  (∀ x : ℝ, ineq_system_2 x a → ineq1 x) →
  a ≤ 2 :=
sorry

end range_of_a_same_solution_set_l2175_217570


namespace martha_knits_hat_in_2_hours_l2175_217546

-- Definitions based on given conditions
variables (H : ℝ)
def knit_times (H : ℝ) : ℝ := H + 3 + 2 + 3 + 6

def total_knitting_time (H : ℝ) : ℝ := 3 * knit_times H

-- The main statement to be proven
theorem martha_knits_hat_in_2_hours (H : ℝ) (h : total_knitting_time H = 48) : H = 2 := 
by
  sorry

end martha_knits_hat_in_2_hours_l2175_217546


namespace sum_base6_l2175_217552

theorem sum_base6 (a b c : ℕ) 
  (ha : a = 1 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 1 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hc : c = 1 * 6^1 + 5 * 6^0) :
  a + b + c = 2 * 6^3 + 2 * 6^2 + 0 * 6^1 + 3 * 6^0 :=
by 
  sorry

end sum_base6_l2175_217552


namespace solve_for_x_l2175_217525

theorem solve_for_x (x : ℝ) (h : 24 - 6 = 3 * x + 3) : x = 5 := by
  sorry

end solve_for_x_l2175_217525


namespace pentagon_same_parity_l2175_217511

open Classical

theorem pentagon_same_parity (vertices : Fin 5 → ℤ × ℤ) : 
  ∃ i j : Fin 5, i ≠ j ∧ (vertices i).1 % 2 = (vertices j).1 % 2 ∧ (vertices i).2 % 2 = (vertices j).2 % 2 :=
by
  sorry

end pentagon_same_parity_l2175_217511


namespace commission_8000_l2175_217556

variable (C k : ℝ)

def commission_5000 (C k : ℝ) : Prop := C + 5000 * k = 110
def commission_11000 (C k : ℝ) : Prop := C + 11000 * k = 230

theorem commission_8000 
  (h1 : commission_5000 C k) 
  (h2 : commission_11000 C k)
  : C + 8000 * k = 170 :=
sorry

end commission_8000_l2175_217556


namespace pictures_on_front_l2175_217538

-- Conditions
variable (total_pictures : ℕ)
variable (pictures_on_back : ℕ)

-- Proof obligation
theorem pictures_on_front (h1 : total_pictures = 15) (h2 : pictures_on_back = 9) : total_pictures - pictures_on_back = 6 :=
sorry

end pictures_on_front_l2175_217538


namespace jenna_interest_l2175_217510

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def interest_earned (P r : ℝ) (n : ℕ) : ℝ :=
  compound_interest P r n - P

theorem jenna_interest :
  interest_earned 1500 0.05 5 = 414.42 :=
by
  sorry

end jenna_interest_l2175_217510


namespace preparation_start_month_l2175_217529

variable (ExamMonth : ℕ)
def start_month (ExamMonth : ℕ) : ℕ :=
  (ExamMonth - 5) % 12

theorem preparation_start_month :
  ∀ (ExamMonth : ℕ), start_month ExamMonth = (ExamMonth - 5) % 12 :=
by
  sorry

end preparation_start_month_l2175_217529


namespace fourth_term_in_arithmetic_sequence_l2175_217500

theorem fourth_term_in_arithmetic_sequence (a d : ℝ) (h : 2 * a + 6 * d = 20) : a + 3 * d = 10 :=
sorry

end fourth_term_in_arithmetic_sequence_l2175_217500


namespace concert_cost_l2175_217592

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l2175_217592


namespace max_abs_asin_b_l2175_217593

theorem max_abs_asin_b (a b c : ℝ) (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) :
  ∃ M : ℝ, (∀ x : ℝ, |a * Real.sin x + b| ≤ M) ∧ M = 2 :=
sorry

end max_abs_asin_b_l2175_217593


namespace complex_pow_six_eq_eight_i_l2175_217587

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end complex_pow_six_eq_eight_i_l2175_217587


namespace LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l2175_217513

noncomputable section

-- Problem 1: Prove length ratios for simultaneous ignition
def LengthRatioSimultaneous (t : ℝ) : Prop :=
  let LA := 1 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosSimultaneous (t : ℝ) : LengthRatioSimultaneous t := sorry

-- Problem 2: Prove length ratios when one candle is lit 30 minutes earlier
def LengthRatioNonSimultaneous (t : ℝ) : Prop :=
  let LA := 5 / 6 - t / 3
  let LB := 1 - t / 5
  (LA / LB = 1 / 2 ∨ LA / LB = 1 / 3 ∨ LA / LB = 1 / 4)

theorem LengthRatiosNonSimultaneous (t : ℝ) : LengthRatioNonSimultaneous t := sorry

end LengthRatiosSimultaneous_LengthRatiosNonSimultaneous_l2175_217513


namespace cost_of_flight_XY_l2175_217590

theorem cost_of_flight_XY :
  let d_XY : ℕ := 4800
  let booking_fee : ℕ := 150
  let cost_per_km : ℚ := 0.12
  ∃ cost : ℚ, cost = d_XY * cost_per_km + booking_fee ∧ cost = 726 := 
by
  sorry

end cost_of_flight_XY_l2175_217590


namespace Lucy_retirement_month_l2175_217565

theorem Lucy_retirement_month (start_month : ℕ) (duration : ℕ) (March : ℕ) (May : ℕ) : 
  (start_month = March) ∧ (duration = 3) → (start_month + duration - 1 = May) :=
by
  intro h
  have h_start_month := h.1
  have h_duration := h.2
  sorry

end Lucy_retirement_month_l2175_217565


namespace compare_expressions_l2175_217515

theorem compare_expressions (x y : ℝ) (h1: x * y > 0) (h2: x ≠ y) : 
  x^4 + 6 * x^2 * y^2 + y^4 > 4 * x * y * (x^2 + y^2) :=
by
  sorry

end compare_expressions_l2175_217515


namespace range_of_m_for_roots_greater_than_2_l2175_217563

theorem range_of_m_for_roots_greater_than_2 :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + (m-2)*x + 5 - m = 0 → x > 2) ↔ (-5 < m ∧ m ≤ -4) :=
  sorry

end range_of_m_for_roots_greater_than_2_l2175_217563
