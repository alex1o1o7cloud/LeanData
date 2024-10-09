import Mathlib

namespace gcd_360_504_is_72_l918_91892

theorem gcd_360_504_is_72 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_is_72_l918_91892


namespace novice_experienced_parts_l918_91828

variables (x y : ℕ)

theorem novice_experienced_parts :
  (y - x = 30) ∧ (x + 2 * y = 180) :=
sorry

end novice_experienced_parts_l918_91828


namespace units_digit_sum_is_9_l918_91860

-- Define the units function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def x := 42 ^ 2
def y := 25 ^ 3

-- Define variables for the units digits of x and y
def units_digit_x := units_digit x
def units_digit_y := units_digit y

-- Define the problem statement to be proven
theorem units_digit_sum_is_9 : units_digit (x + y) = 9 :=
by sorry

end units_digit_sum_is_9_l918_91860


namespace fixed_point_and_max_distance_eqn_l918_91856

-- Define line l1
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y - 8 = 0

-- Define line l2 parallel to l1 passing through origin
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  (m + 1) * x - (m - 3) * y = 0

-- Define line y = x
def line_y_eq_x (x y : ℝ) : Prop :=
  y = x

-- Define line x + y = 0
def line_x_plus_y_eq_0 (x y : ℝ) : Prop :=
  x + y = 0

theorem fixed_point_and_max_distance_eqn :
  (∀ m : ℝ, l1 m 2 2) ∧ (∀ m : ℝ, (l2 m 2 2 → false)) →
  (∃ x y : ℝ, l2 m x y ∧ line_x_plus_y_eq_0 x y) :=
by sorry

end fixed_point_and_max_distance_eqn_l918_91856


namespace sum_of_A_H_l918_91818

theorem sum_of_A_H (A B C D E F G H : ℝ) (h1 : C = 10) 
  (h2 : A + B + C = 40) (h3 : B + C + D = 40) (h4 : C + D + E = 40) 
  (h5 : D + E + F = 40) (h6 : E + F + G = 40) (h7 : F + G + H = 40) :
  A + H = 30 := 
sorry

end sum_of_A_H_l918_91818


namespace tony_walking_speed_l918_91823

-- Define the conditions as hypotheses
def walking_speed_on_weekend (W : ℝ) : Prop := 
  let store_distance := 4 
  let run_speed := 10
  let day1_time := store_distance / W
  let day2_time := store_distance / run_speed
  let day3_time := store_distance / run_speed
  let avg_time := (day1_time + day2_time + day3_time) / 3
  avg_time = 56 / 60

-- State the theorem
theorem tony_walking_speed : ∃ W : ℝ, walking_speed_on_weekend W ∧ W = 2 := 
sorry

end tony_walking_speed_l918_91823


namespace geom_mean_does_not_exist_l918_91847

theorem geom_mean_does_not_exist (a b : Real) (h1 : a = 2) (h2 : b = -2) : ¬ ∃ g : Real, g^2 = a * b := 
by
  sorry

end geom_mean_does_not_exist_l918_91847


namespace ending_number_of_second_range_l918_91864

theorem ending_number_of_second_range :
  let avg100_400 := (100 + 400) / 2
  let avg_50_n := (50 + n) / 2
  avg100_400 = avg_50_n + 100 → n = 250 :=
by
  sorry

end ending_number_of_second_range_l918_91864


namespace tennis_tournament_rounds_needed_l918_91862

theorem tennis_tournament_rounds_needed (n : ℕ) (total_participants : ℕ) (win_points loss_points : ℕ) (get_point_no_pair : ℕ) (elimination_loss : ℕ) :
  total_participants = 1152 →
  win_points = 1 →
  loss_points = 0 →
  get_point_no_pair = 1 →
  elimination_loss = 2 →
  n = 14 :=
by
  sorry

end tennis_tournament_rounds_needed_l918_91862


namespace statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l918_91802

-- Statement A: Proving the solution set of the inequality
theorem statement_A_solution_set (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ (-1 / 2) < x ∧ x < 1 :=
sorry

-- Statement B: "ab > 1" is not a sufficient condition for "a > 1, b > 1"
theorem statement_B_insufficient_condition (a b : ℝ) :
  (a * b > 1) → ¬(a > 1 ∧ b > 1) :=
sorry

-- Statement C: The negation of p: ∀ x ∈ ℝ, x² > 0 is true
theorem statement_C_negation (x0 : ℝ) : 
  (∀ x : ℝ, x^2 > 0) → ¬ (∃ x0 : ℝ, x0^2 ≤ 0) :=
sorry

-- Statement D: "a < 2" is not a necessary condition for "a < 6"
theorem statement_D_not_necessary_condition (a : ℝ) :
  (a < 2) → ¬(a < 6) :=
sorry

end statement_A_solution_set_statement_B_insufficient_condition_statement_C_negation_statement_D_not_necessary_condition_l918_91802


namespace prob_both_calligraphy_is_correct_prob_one_each_is_correct_l918_91885

section ProbabilityOfVolunteerSelection

variable (C P : ℕ) -- C = number of calligraphy competition winners, P = number of painting competition winners
variable (total_pairs : ℕ := 6 * (6 - 1) / 2) -- Number of ways to choose 2 out of 6 participants, binomial coefficient (6 choose 2)

-- Condition variables
def num_calligraphy_winners : ℕ := 4
def num_painting_winners : ℕ := 2
def num_total_winners : ℕ := num_calligraphy_winners + num_painting_winners

-- Number of pairs of both calligraphy winners
def pairs_both_calligraphy : ℕ := 4 * (4 - 1) / 2
-- Number of pairs of one calligraphy and one painting winner
def pairs_one_each : ℕ := 4 * 2

-- Probability calculations
def prob_both_calligraphy : ℚ := pairs_both_calligraphy / total_pairs
def prob_one_each : ℚ := pairs_one_each / total_pairs

-- Theorem statements to prove the probabilities of selected types of volunteers
theorem prob_both_calligraphy_is_correct : 
  prob_both_calligraphy = 2/5 := sorry

theorem prob_one_each_is_correct : 
  prob_one_each = 8/15 := sorry

end ProbabilityOfVolunteerSelection

end prob_both_calligraphy_is_correct_prob_one_each_is_correct_l918_91885


namespace frank_peanuts_average_l918_91804

theorem frank_peanuts_average :
  let one_dollar := 7 * 1
  let five_dollar := 4 * 5
  let ten_dollar := 2 * 10
  let twenty_dollar := 1 * 20
  let total_money := one_dollar + five_dollar + ten_dollar + twenty_dollar
  let change := 4
  let money_spent := total_money - change
  let cost_per_pound := 3
  let total_pounds := money_spent / cost_per_pound
  let days := 7
  let average_per_day := total_pounds / days
  average_per_day = 3 :=
by
  sorry

end frank_peanuts_average_l918_91804


namespace average_of_remaining_five_l918_91825

open Nat Real

theorem average_of_remaining_five (avg9 avg4 : ℝ) (S S4 : ℝ) 
(h1 : avg9 = 18) (h2 : avg4 = 8) 
(h_sum9 : S = avg9 * 9) 
(h_sum4 : S4 = avg4 * 4) :
(S - S4) / 5 = 26 := by
  sorry

end average_of_remaining_five_l918_91825


namespace repeating_decimal_multiplication_l918_91865

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l918_91865


namespace greatest_possible_sum_of_squares_l918_91898

theorem greatest_possible_sum_of_squares (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 8 :=
by sorry

end greatest_possible_sum_of_squares_l918_91898


namespace mrs_sheridan_total_cats_l918_91841

-- Definitions from the conditions
def original_cats : Nat := 17
def additional_cats : Nat := 14

-- The total number of cats is the sum of the original and additional cats
def total_cats : Nat := original_cats + additional_cats

-- Statement to prove
theorem mrs_sheridan_total_cats : total_cats = 31 := by
  sorry

end mrs_sheridan_total_cats_l918_91841


namespace initial_markers_l918_91843

variable (markers_given : ℕ) (total_markers : ℕ)

theorem initial_markers (h_given : markers_given = 109) (h_total : total_markers = 326) :
  total_markers - markers_given = 217 :=
by
  sorry

end initial_markers_l918_91843


namespace compute_expression_l918_91874

theorem compute_expression : 12 * (1 / 26) * 52 * 4 = 96 :=
by
  sorry

end compute_expression_l918_91874


namespace determine_students_and_benches_l918_91876

theorem determine_students_and_benches (a b s : ℕ) :
  (s = a * b + 5) ∧ (s = 8 * b - 4) →
  ((a = 7 ∧ b = 9 ∧ s = 68) ∨ (a = 5 ∧ b = 3 ∧ s = 20)) :=
by
  sorry

end determine_students_and_benches_l918_91876


namespace find_first_term_l918_91890

theorem find_first_term
  (S : ℝ) (a r : ℝ)
  (h1 : S = 10)
  (h2 : a + a * r = 6)
  (h3 : a = 2 * r) :
  a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13 := by
  sorry

end find_first_term_l918_91890


namespace select_twins_in_grid_l918_91896

theorem select_twins_in_grid (persons : Fin 8 × Fin 8 → Fin 2) :
  ∃ (selection : Fin 8 × Fin 8 → Bool), 
    (∀ i : Fin 8, ∃ j : Fin 8, selection (i, j) = true) ∧ 
    (∀ j : Fin 8, ∃ i : Fin 8, selection (i, j) = true) :=
sorry

end select_twins_in_grid_l918_91896


namespace parity_of_expression_l918_91851

theorem parity_of_expression {a b c : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : 0 < c) :
  ∃ k : ℕ, 3 ^ a + (b - 1) ^ 2 * c = 2 * k + 1 :=
by
  sorry

end parity_of_expression_l918_91851


namespace age_of_person_l918_91814

theorem age_of_person (x : ℕ) (h : 3 * (x + 3) - 3 * (x - 3) = x) : x = 18 :=
  sorry

end age_of_person_l918_91814


namespace scheme_choice_l918_91855

variable (x y₁ y₂ : ℕ)

def cost_scheme_1 (x : ℕ) : ℕ := 12 * x + 40

def cost_scheme_2 (x : ℕ) : ℕ := 16 * x

theorem scheme_choice :
  ∀ (x : ℕ), 5 ≤ x → x ≤ 20 →
  (if x < 10 then cost_scheme_2 x < cost_scheme_1 x else
   if x = 10 then cost_scheme_2 x = cost_scheme_1 x else
   cost_scheme_1 x < cost_scheme_2 x) :=
by
  sorry

end scheme_choice_l918_91855


namespace john_pays_per_year_l918_91886

-- Define the costs and insurance parameters.
def cost_per_epipen : ℝ := 500
def insurance_coverage : ℝ := 0.75

-- Number of months in a year.
def months_in_year : ℕ := 12

-- Number of months each EpiPen lasts.
def months_per_epipen : ℕ := 6

-- Amount covered by insurance for each EpiPen.
def insurance_amount (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * coverage

-- Amount John pays after insurance for each EpiPen.
def amount_john_pays_per_epipen (cost : ℝ) (covered: ℝ) : ℝ :=
  cost - covered

-- Number of EpiPens John needs per year.
def epipens_per_year (months_in_year : ℕ) (months_per_epipen : ℕ) : ℕ :=
  months_in_year / months_per_epipen

-- Total amount John pays per year.
def total_amount_john_pays_per_year (amount_per_epipen : ℝ) (epipens_per_year : ℕ) : ℝ :=
  amount_per_epipen * epipens_per_year

-- Theorem to prove the correct answer.
theorem john_pays_per_year :
  total_amount_john_pays_per_year (amount_john_pays_per_epipen cost_per_epipen (insurance_amount cost_per_epipen insurance_coverage)) (epipens_per_year months_in_year months_per_epipen) = 250 := 
by
  sorry

end john_pays_per_year_l918_91886


namespace discount_percentage_l918_91875

theorem discount_percentage (shirts : ℕ) (total_cost : ℕ) (price_after_discount : ℕ) 
  (h1 : shirts = 3) (h2 : total_cost = 60) (h3 : price_after_discount = 12) : 
  ∃ discount_percentage : ℕ, discount_percentage = 40 := 
by 
  sorry

end discount_percentage_l918_91875


namespace find_f_prime_at_two_l918_91837

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

noncomputable def f' (x a b : ℝ) : ℝ := (a * x + 2) / (x^2)

theorem find_f_prime_at_two (a b : ℝ) (h₁ : f 1 a b = -2) (h₂ : f' 1 a b = 0) : 
  f' 2 a (-2) = -1 / 2 := 
by
  sorry

end find_f_prime_at_two_l918_91837


namespace total_pieces_of_candy_l918_91836

-- Define the given conditions
def students : ℕ := 43
def pieces_per_student : ℕ := 8

-- Define the goal, which is proving the total number of pieces of candy is 344
theorem total_pieces_of_candy : students * pieces_per_student = 344 :=
by
  sorry

end total_pieces_of_candy_l918_91836


namespace max_value_x_1_minus_3x_is_1_over_12_l918_91883

open Real

noncomputable def max_value_of_x_1_minus_3x (x : ℝ) : ℝ :=
  x * (1 - 3 * x)

theorem max_value_x_1_minus_3x_is_1_over_12 :
  ∀ x : ℝ, 0 < x ∧ x < 1 / 3 → max_value_of_x_1_minus_3x x ≤ 1 / 12 :=
by
  intros x h
  sorry

end max_value_x_1_minus_3x_is_1_over_12_l918_91883


namespace geometric_seq_relation_l918_91858

variables {α : Type*} [Field α]

-- Conditions for the arithmetic sequence (for reference)
def arithmetic_seq_sum (S : ℕ → α) (d : α) : Prop :=
∀ m n : ℕ, S (m + n) = S m + S n + (m * n) * d

-- Conditions for the geometric sequence
def geometric_seq_prod (T : ℕ → α) (q : α) : Prop :=
∀ m n : ℕ, T (m + n) = T m * T n * (q ^ (m * n))

-- Proving the desired relationship
theorem geometric_seq_relation {T : ℕ → α} {q : α} (h : geometric_seq_prod T q) (m n : ℕ) :
  T (m + n) = T m * T n * (q ^ (m * n)) :=
by
  apply h m n

end geometric_seq_relation_l918_91858


namespace student_community_arrangement_l918_91867

theorem student_community_arrangement :
  let students := 4
  let communities := 3
  (students.choose 2) * (communities.factorial / (communities - (students - 1)).factorial) = 36 :=
by
  have students := 4
  have communities := 3
  sorry

end student_community_arrangement_l918_91867


namespace balloons_floated_away_l918_91871

theorem balloons_floated_away (starting_balloons given_away grabbed_balloons final_balloons flattened_balloons : ℕ)
  (h1 : starting_balloons = 50)
  (h2 : given_away = 10)
  (h3 : grabbed_balloons = 11)
  (h4 : final_balloons = 39)
  : flattened_balloons = starting_balloons - given_away + grabbed_balloons - final_balloons → flattened_balloons = 12 :=
by
  sorry

end balloons_floated_away_l918_91871


namespace sin_cos_power_sum_l918_91827

theorem sin_cos_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := 
by
  sorry

end sin_cos_power_sum_l918_91827


namespace staircase_problem_l918_91873

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem staircase_problem (total_steps required_steps : ℕ) (num_two_steps : ℕ) :
  total_steps = 11 ∧ required_steps = 7 ∧ num_two_steps = 4 →
  C 7 4 = 35 :=
by
  intro h
  sorry

end staircase_problem_l918_91873


namespace reflex_angle_at_T_l918_91863

-- Assume points P, Q, R, and S are aligned
def aligned (P Q R S : ℝ × ℝ) : Prop :=
  ∃ a b, ∀ x, x = 0 * a + b + (P.1, Q.1, R.1, S.1)

-- Angles given in the problem
def PQT_angle : ℝ := 150
def RTS_angle : ℝ := 70

-- definition of the reflex angle at T
def reflex_angle (angle : ℝ) : ℝ := 360 - angle

theorem reflex_angle_at_T (P Q R S T : ℝ × ℝ) :
  aligned P Q R S → PQT_angle = 150 → RTS_angle = 70 →
  reflex_angle 40 = 320 :=
by
  sorry

end reflex_angle_at_T_l918_91863


namespace cos2_a_plus_sin2_b_eq_one_l918_91831

variable {a b c : ℝ}

theorem cos2_a_plus_sin2_b_eq_one
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  Real.cos a ^ 2 + Real.sin b ^ 2 = 1 := 
  sorry

end cos2_a_plus_sin2_b_eq_one_l918_91831


namespace find_a_l918_91830

theorem find_a (a x_0 : ℝ) (h_tangent: (ax_0^3 + 1 = x_0) ∧ (3 * a * x_0^2 = 1)) : a = 4 / 27 :=
sorry

end find_a_l918_91830


namespace gcd_of_expression_l918_91869

theorem gcd_of_expression 
  (a b c d : ℕ) :
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a - b) (c - d)) (a - c)) (b - d)) (a - d)) (b - c) = 12 :=
sorry

end gcd_of_expression_l918_91869


namespace proof_statement_l918_91880

-- Assume 5 * 3^x = 243
def condition (x : ℝ) : Prop := 5 * (3:ℝ)^x = 243

-- Define the log base 3 for use in the statement
noncomputable def log_base_3 (y : ℝ) : ℝ := Real.log y / Real.log 3

-- State that if the condition holds, then (x + 2)(x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2
theorem proof_statement (x : ℝ) (h : condition x) : (x + 2) * (x - 2) = 21 - 10 * log_base_3 5 + (log_base_3 5)^2 := sorry

end proof_statement_l918_91880


namespace max_digit_d_of_form_7d733e_multiple_of_33_l918_91884

theorem max_digit_d_of_form_7d733e_multiple_of_33 
  (d e : ℕ) (d_digit : d < 10) (e_digit : e < 10) 
  (multiple_of_33: ∃ k : ℕ, 7 * 10^5 + d * 10^4 + 7 * 10^3 + 33 * 10 + e = 33 * k) 
  : d ≤ 6 := 
sorry

end max_digit_d_of_form_7d733e_multiple_of_33_l918_91884


namespace find_d_l918_91852

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)
  (h1 : α = c)
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : c = 36) :
  d = 42 := 
sorry

end find_d_l918_91852


namespace fiona_probability_correct_l918_91838

def probability_to_reach_pad14 :=
  (1 / 27) + (1 / 3) = 13 / 27 ∧
  (13 / 27) * (1 / 3) = 13 / 81 ∧
  (13 / 81) * (1 / 3) = 13 / 243 ∧
  (13 / 243) * (1 / 3) = 13 / 729 ∧
  (1 / 81) + (1 / 27) + (1 / 27) = 4 / 81 ∧
  (13 / 729) * (4 / 81) = 52 / 59049

theorem fiona_probability_correct :
  (probability_to_reach_pad14 : Prop) := by
  sorry

end fiona_probability_correct_l918_91838


namespace find_a6_l918_91835

variable {a : ℕ → ℤ} -- Assume we have a sequence of integers
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Conditions
axiom h1 : a 3 = 7
axiom h2 : a 5 = a 2 + 6

-- Define arithmetic sequence property
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Theorem to prove
theorem find_a6 (h1 : a 3 = 7) (h2 : a 5 = a 2 + 6) (h3 : arithmetic_seq a d) : a 6 = 13 :=
by
  sorry

end find_a6_l918_91835


namespace number_of_students_scoring_above_90_l918_91807

theorem number_of_students_scoring_above_90
  (total_students : ℕ)
  (mean : ℝ)
  (variance : ℝ)
  (students_scoring_at_least_60 : ℕ)
  (h1 : total_students = 1200)
  (h2 : mean = 75)
  (h3 : ∃ (σ : ℝ), variance = σ^2)
  (h4 : students_scoring_at_least_60 = 960)
  : ∃ n, n = total_students - students_scoring_at_least_60 ∧ n = 240 :=
by {
  sorry
}

end number_of_students_scoring_above_90_l918_91807


namespace student_answers_all_correctly_l918_91854

/-- 
The exam tickets have 2 theoretical questions and 1 problem each. There are 28 tickets. 
A student is prepared for 50 theoretical questions out of 56 and 22 problems out of 28.
The probability that by drawing a ticket at random, and the student answers all questions 
correctly is 0.625.
-/
theorem student_answers_all_correctly :
  let total_theoretical := 56
  let total_problems := 28
  let prepared_theoretical := 50
  let prepared_problems := 22
  let p_correct_theoretical := (prepared_theoretical * (prepared_theoretical - 1)) / (total_theoretical * (total_theoretical - 1))
  let p_correct_problem := prepared_problems / total_problems
  let combined_probability := p_correct_theoretical * p_correct_problem
  combined_probability = 0.625 :=
  sorry

end student_answers_all_correctly_l918_91854


namespace clock_ticks_6_times_at_6_oclock_l918_91899

theorem clock_ticks_6_times_at_6_oclock
  (h6 : 5 * t = 25)
  (h12 : 11 * t = 55) :
  t = 5 ∧ 6 = 6 :=
by
  sorry

end clock_ticks_6_times_at_6_oclock_l918_91899


namespace find_m_max_value_l918_91879

noncomputable def f (x : ℝ) := |x - 1|

theorem find_m (m : ℝ) :
  (∀ x, f (x + 5) ≤ 3 * m) ∧ m > 0 ∧ (∀ x, -7 ≤ x ∧ x ≤ -1 → f (x + 5) ≤ 3 * m) →
  m = 1 :=
by
  sorry

theorem max_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h2 : 2 * a ^ 2 + b ^ 2 = 3) :
  ∃ x, (∀ a b, 2 * a * Real.sqrt (1 + b ^ 2) ≤ x) ∧ x = 2 * Real.sqrt 2 :=
by
  sorry

end find_m_max_value_l918_91879


namespace range_of_magnitudes_l918_91868

theorem range_of_magnitudes (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 5) :
  3 ≤ BC ∧ BC ≤ 13 :=
by
  sorry

end range_of_magnitudes_l918_91868


namespace hannah_money_left_l918_91833

variable (initial_amount : ℕ) (amount_spent_rides : ℕ) (amount_spent_dessert : ℕ)
  (remaining_after_rides : ℕ) (remaining_money : ℕ)

theorem hannah_money_left :
  initial_amount = 30 →
  amount_spent_rides = initial_amount / 2 →
  remaining_after_rides = initial_amount - amount_spent_rides →
  amount_spent_dessert = 5 →
  remaining_money = remaining_after_rides - amount_spent_dessert →
  remaining_money = 10 := by
  sorry

end hannah_money_left_l918_91833


namespace diagonal_in_parallelogram_l918_91887

-- Define the conditions of the problem
variable (A B C D M : Point)
variable (parallelogram : Parallelogram A B C D)
variable (height_bisects_side : Midpoint M A D)
variable (height_length : Distance B M = 2)
variable (acute_angle_30 : Angle A B D = 30)

-- Define the theorem based on the conditions
theorem diagonal_in_parallelogram (h1 : parallelogram) (h2 : height_bisects_side)
  (h3 : height_length) (h4 : acute_angle_30) : 
  ∃ (BD_length : ℝ) (angle1 angle2 : ℝ), BD_length = 4 ∧ angle1 = 30 ∧ angle2 = 120 := 
sorry

end diagonal_in_parallelogram_l918_91887


namespace wire_length_l918_91829

theorem wire_length (r_sphere r_cylinder : ℝ) (V_sphere_eq_V_cylinder : (4/3) * π * r_sphere^3 = π * r_cylinder^2 * 144) :
  r_sphere = 12 → r_cylinder = 4 → 144 = 144 := sorry

end wire_length_l918_91829


namespace time_for_first_half_is_15_l918_91872

-- Definitions of the conditions in Lean
def floors := 20
def time_per_floor_next_5 := 5
def time_per_floor_final_5 := 16
def total_time := 120

-- Theorem statement
theorem time_for_first_half_is_15 :
  ∃ T, (T + (5 * time_per_floor_next_5) + (5 * time_per_floor_final_5) = total_time) ∧ (T = 15) :=
by
  sorry

end time_for_first_half_is_15_l918_91872


namespace juan_european_stamps_total_cost_l918_91821

/-- Define the cost of European stamps collection for Juan -/
def total_cost_juan_stamps : ℝ := 
  -- Costs of stamps from the 1980s
  (15 * 0.07) + (11 * 0.06) + (14 * 0.08) +
  -- Costs of stamps from the 1990s
  (14 * 0.07) + (10 * 0.06) + (12 * 0.08)

/-- Prove that the total cost for European stamps from the 80s and 90s is $5.37 -/
theorem juan_european_stamps_total_cost : total_cost_juan_stamps = 5.37 :=
  by sorry

end juan_european_stamps_total_cost_l918_91821


namespace solve_quartic_eqn_l918_91826

noncomputable def solutionSet : Set ℂ :=
  {x | x^2 = 6 ∨ x^2 = -6}

theorem solve_quartic_eqn (x : ℂ) : (x^4 - 36 = 0) ↔ (x ∈ solutionSet) := 
sorry

end solve_quartic_eqn_l918_91826


namespace blue_more_than_white_l918_91845

theorem blue_more_than_white :
  ∃ (B R : ℕ), (B > 16) ∧ (R = 2 * B) ∧ (B + R + 16 = 100) ∧ (B - 16 = 12) :=
sorry

end blue_more_than_white_l918_91845


namespace xy_power_l918_91866

def x : ℚ := 3/4
def y : ℚ := 4/3

theorem xy_power : x^7 * y^8 = 4/3 := by
  sorry

end xy_power_l918_91866


namespace total_slices_l918_91805

theorem total_slices (pizzas : ℕ) (slices1 slices2 slices3 slices4 : ℕ)
  (h1 : pizzas = 4)
  (h2 : slices1 = 8)
  (h3 : slices2 = 8)
  (h4 : slices3 = 10)
  (h5 : slices4 = 12) :
  slices1 + slices2 + slices3 + slices4 = 38 := by
  sorry

end total_slices_l918_91805


namespace plan1_more_cost_effective_than_plan2_l918_91801

variable (x : ℝ)

def plan1_cost (x : ℝ) : ℝ :=
  36 + 0.1 * x

def plan2_cost (x : ℝ) : ℝ :=
  0.6 * x

theorem plan1_more_cost_effective_than_plan2 (h : x > 72) : 
  plan1_cost x < plan2_cost x :=
by
  sorry

end plan1_more_cost_effective_than_plan2_l918_91801


namespace large_bucket_capacity_l918_91870

variable (S L : ℕ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
sorry

end large_bucket_capacity_l918_91870


namespace negation_of_p_l918_91839

def p := ∀ x : ℝ, Real.sin x ≤ 1

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, Real.sin x > 1 := 
by 
  sorry

end negation_of_p_l918_91839


namespace city_mileage_per_tankful_l918_91895

theorem city_mileage_per_tankful :
  ∀ (T : ℝ), 
  ∃ (city_miles : ℝ),
    (462 = T * (32 + 12)) ∧
    (city_miles = 32 * T) ∧
    (city_miles = 336) :=
by
  sorry

end city_mileage_per_tankful_l918_91895


namespace find_fx_for_neg_x_l918_91897

-- Let f be an odd function defined on ℝ 
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)

-- Given condition for x > 0
variable (h_pos : ∀ x, 0 < x → f x = x^2 + x - 1)

-- Problem: Prove that f(x) = -x^2 + x + 1 for x < 0
theorem find_fx_for_neg_x (x : ℝ) (h_neg : x < 0) : f x = -x^2 + x + 1 :=
sorry

end find_fx_for_neg_x_l918_91897


namespace no_two_perfect_cubes_l918_91822

theorem no_two_perfect_cubes (n : ℕ) : ¬ (∃ a b : ℕ, a^3 = n + 2 ∧ b^3 = n^2 + n + 1) := by
  sorry

end no_two_perfect_cubes_l918_91822


namespace g_of_g_of_g_of_20_l918_91840

def g (x : ℕ) : ℕ :=
  if x < 10 then x^2 - 9 else x - 15

theorem g_of_g_of_g_of_20 : g (g (g 20)) = 1 := by
  -- Proof steps would go here
  sorry

end g_of_g_of_g_of_20_l918_91840


namespace arithmetic_sequence_sum_l918_91810

theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℕ) (h1 : S n = 54) (h2 : S (2 * n) = 72) :
  S (3 * n) = 78 :=
sorry

end arithmetic_sequence_sum_l918_91810


namespace second_range_is_18_l918_91859

variable (range1 range2 range3 : ℕ)

theorem second_range_is_18
  (h1 : range1 = 30)
  (h2 : range2 = 18)
  (h3 : range3 = 32) :
  range2 = 18 := by
  sorry

end second_range_is_18_l918_91859


namespace insurance_compensation_correct_l918_91848

def actual_damage : ℝ := 300000
def deductible_percent : ℝ := 0.01
def deductible_amount : ℝ := deductible_percent * actual_damage
def insurance_compensation : ℝ := actual_damage - deductible_amount

theorem insurance_compensation_correct : insurance_compensation = 297000 :=
by
  -- To be proved
  sorry

end insurance_compensation_correct_l918_91848


namespace consecutive_integers_sum_to_thirty_unique_sets_l918_91891

theorem consecutive_integers_sum_to_thirty_unique_sets :
  (∃ a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60) ↔ ∃! a n : ℕ, a ≥ 3 ∧ n ≥ 2 ∧ n * (2 * a + n - 1) = 60 :=
by
  sorry

end consecutive_integers_sum_to_thirty_unique_sets_l918_91891


namespace original_price_petrol_in_euros_l918_91849

theorem original_price_petrol_in_euros
  (P : ℝ) -- The original price of petrol in USD per gallon
  (h1 : 0.865 * P * 7.25 + 0.135 * 325 = 325) -- Condition derived from price reduction and additional gallons
  (h2 : P > 0) -- Ensure original price is positive
  (exchange_rate : ℝ) (h3 : exchange_rate = 1.15) : 
  P / exchange_rate = 38.98 :=
by 
  let price_in_euros := P / exchange_rate 
  have h4 : price_in_euros = 38.98 := sorry
  exact h4

end original_price_petrol_in_euros_l918_91849


namespace divisor_of_first_division_l918_91832

theorem divisor_of_first_division (n d : ℕ) (hn_pos : 0 < n)
  (h₁ : (n + 1) % d = 4) (h₂ : n % 2 = 1) : 
  d = 6 :=
sorry

end divisor_of_first_division_l918_91832


namespace depth_of_sand_l918_91842

theorem depth_of_sand (h : ℝ) (fraction_above_sand : ℝ) :
  h = 9000 → fraction_above_sand = 1/9 → depth = 342 :=
by
  -- height of the pyramid
  let height := 9000
  -- ratio of submerged height to the total height
  let ratio := (8 / 9)^(1 / 3)
  -- height of the submerged part
  let submerged_height := height * ratio
  -- depth of the sand
  let depth := height - submerged_height
  sorry

end depth_of_sand_l918_91842


namespace integer_solutions_system_ineq_l918_91815

theorem integer_solutions_system_ineq (x : ℤ) :
  (3 * x + 6 > x + 8 ∧ (x : ℚ) / 4 ≥ (x - 1) / 3) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  sorry

end integer_solutions_system_ineq_l918_91815


namespace harry_water_per_mile_l918_91893

noncomputable def water_per_mile_during_first_3_miles (initial_water : ℝ) (remaining_water : ℝ) (leak_rate : ℝ) (hike_time : ℝ) (water_drunk_last_mile : ℝ) (first_3_miles : ℝ) : ℝ :=
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  water_drunk_first_3_miles / first_3_miles

theorem harry_water_per_mile :
  water_per_mile_during_first_3_miles 10 2 1 2 3 3 = 1 / 3 :=
by
  have initial_water := 10
  have remaining_water := 2
  have leak_rate := 1
  have hike_time := 2
  have water_drunk_last_mile := 3
  have first_3_miles := 3
  let total_leak := leak_rate * hike_time
  let remaining_before_last_mile := remaining_water + water_drunk_last_mile
  let start_last_mile := remaining_before_last_mile + total_leak
  let water_before_leak := initial_water - total_leak
  let water_drunk_first_3_miles := water_before_leak - start_last_mile
  let result := water_drunk_first_3_miles / first_3_miles
  exact sorry

end harry_water_per_mile_l918_91893


namespace proportional_increase_l918_91846

theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) : y = (3 / 2) * x - 7 / 2 :=
by
  sorry

end proportional_increase_l918_91846


namespace general_formula_minimum_n_l918_91857

-- Definitions based on given conditions
def arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d
def sum_arith_seq (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- Conditions of the problem
def a2 : ℤ := -5
def S5 : ℤ := -20

-- Proving the general formula of the sequence
theorem general_formula :
  ∃ a₁ d, arith_seq a₁ d 2 = a2 ∧ sum_arith_seq a₁ d 5 = S5 ∧ (∀ n, arith_seq a₁ d n = n - 7) :=
by
  sorry

-- Proving the minimum value of n for which Sn > an
theorem minimum_n :
  ∃ n : ℕ, (n > 14) ∧ sum_arith_seq (-6) 1 n > arith_seq (-6) 1 n :=
by
  sorry

end general_formula_minimum_n_l918_91857


namespace exists_ordering_no_arithmetic_progression_l918_91878

theorem exists_ordering_no_arithmetic_progression (m : ℕ) (hm : 0 < m) :
  ∃ (a : Fin (2^m) → ℕ), (∀ i j k : Fin (2^m), i < j → j < k → a j - a i ≠ a k - a j) := sorry

end exists_ordering_no_arithmetic_progression_l918_91878


namespace total_amount_spent_l918_91894

def cost_of_haley_paper : ℝ := 3.75 + (3.75 * 0.5)
def cost_of_sister_paper : ℝ := (4.50 * 2) + (4.50 * 0.5)
def cost_of_haley_pens : ℝ := (1.45 * 5) - ((1.45 * 5) * 0.25)
def cost_of_sister_pens : ℝ := (1.65 * 7) - ((1.65 * 7) * 0.25)

def total_cost_of_supplies : ℝ := cost_of_haley_paper + cost_of_sister_paper + cost_of_haley_pens + cost_of_sister_pens

theorem total_amount_spent : total_cost_of_supplies = 30.975 :=
by
  sorry

end total_amount_spent_l918_91894


namespace batsman_average_after_17th_inning_l918_91888

theorem batsman_average_after_17th_inning (A : ℝ) :
  (16 * A + 87) / 17 = A + 3 → A + 3 = 39 :=
by
  intro h
  sorry

end batsman_average_after_17th_inning_l918_91888


namespace modulus_of_z_l918_91834

open Complex -- Open the Complex number namespace

-- Define the given condition as a hypothesis
def condition (z : ℂ) : Prop := (1 + I) * z = 3 + I

-- Statement of the theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = Real.sqrt 5 :=
sorry

end modulus_of_z_l918_91834


namespace final_laptop_price_l918_91853

theorem final_laptop_price :
  let original_price := 1000.00
  let first_discounted_price := original_price * (1 - 0.10)
  let second_discounted_price := first_discounted_price * (1 - 0.25)
  let recycling_fee := second_discounted_price * 0.05
  let final_price := second_discounted_price + recycling_fee
  final_price = 708.75 :=
by
  sorry

end final_laptop_price_l918_91853


namespace amare_needs_more_fabric_l918_91819

theorem amare_needs_more_fabric :
  let first_two_dresses_in_feet := 2 * 5.5 * 3
  let next_two_dresses_in_feet := 2 * 6 * 3
  let last_two_dresses_in_feet := 2 * 6.5 * 3
  let total_fabric_needed := first_two_dresses_in_feet + next_two_dresses_in_feet + last_two_dresses_in_feet
  let fabric_amare_has := 10
  total_fabric_needed - fabric_amare_has = 98 :=
by {
  sorry
}

end amare_needs_more_fabric_l918_91819


namespace peter_savings_l918_91824

noncomputable def calc_discounted_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    original_price * (1 - discount_percentage / 100)

noncomputable def calc_savings (original_price : ℝ) (external_price : ℝ) : ℝ :=
    original_price - external_price

noncomputable def total_savings : ℝ :=
    let math_original := 45.0
    let math_discount := 20.0
    let science_original := 60.0
    let science_discount := 25.0
    let literature_original := 35.0
    let literature_discount := 15.0
    let math_external := calc_discounted_price math_original math_discount
    let science_external := calc_discounted_price science_original science_discount
    let literature_external := calc_discounted_price literature_original literature_discount
    let math_savings := calc_savings math_original math_external
    let science_savings := calc_savings science_original science_external
    let literature_savings := calc_savings literature_original literature_external
    math_savings + science_savings + literature_savings

theorem peter_savings :
  total_savings = 29.25 :=
by
    sorry

end peter_savings_l918_91824


namespace diagonal_intersection_probability_decagon_l918_91861

noncomputable def probability_diagonal_intersection_in_decagon : ℚ :=
  let vertices := 10
  let total_diagonals := (vertices * (vertices - 3)) / 2
  let total_pairs_of_diagonals := total_diagonals * (total_diagonals - 1) / 2
  let total_intersecting_pairs := (vertices * (vertices - 1) * (vertices - 2) * (vertices - 3)) / 24
  total_intersecting_pairs / total_pairs_of_diagonals

theorem diagonal_intersection_probability_decagon (h : probability_diagonal_intersection_in_decagon = 42 / 119) : 
  probability_diagonal_intersection_in_decagon = 42 / 119 :=
sorry

end diagonal_intersection_probability_decagon_l918_91861


namespace complex_multiplication_quadrant_l918_91882

-- Given conditions
def complex_mul (z1 z2 : ℂ) : ℂ := z1 * z2

-- Proving point is in the fourth quadrant
theorem complex_multiplication_quadrant
  (a b : ℝ) (z : ℂ)
  (h1 : z = a + b * Complex.I)
  (h2 : z = complex_mul (1 + Complex.I) (3 - Complex.I)) :
  b < 0 ∧ a > 0 :=
by
  sorry

end complex_multiplication_quadrant_l918_91882


namespace egg_sales_l918_91806

/-- Two vendors together sell 110 eggs and both have equal revenues.
    Given the conditions about changing the number of eggs and corresponding revenues,
    the first vendor sells 60 eggs and the second vendor sells 50 eggs. -/
theorem egg_sales (x y : ℝ) (h1 : x + (110 - x) = 110) (h2 : 110 * (y / x) = 5) (h3 : 110 * (y / (110 - x)) = 7.2) :
  x = 60 ∧ (110 - x) = 50 :=
by sorry

end egg_sales_l918_91806


namespace carol_first_round_points_l918_91850

theorem carol_first_round_points (P : ℤ) (h1 : P + 6 - 16 = 7) : P = 17 :=
by
  sorry

end carol_first_round_points_l918_91850


namespace complement_intersection_l918_91881

-- Define the universal set U and sets A, B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x ∈ A | x ∈ B}

-- Define the complement of A_inter_B in U
def complement_U_A_inter_B : Set ℕ := {x ∈ U | x ∉ A_inter_B}

-- Prove that the complement of the intersection of A and B in U is {1, 4, 5}
theorem complement_intersection :
  complement_U_A_inter_B = {1, 4, 5} :=
by
  sorry

end complement_intersection_l918_91881


namespace no_real_solution_l918_91803

theorem no_real_solution (n : ℝ) : (∀ x : ℝ, (x+6)*(x-3) = n + 4*x → false) ↔ n < -73/4 := by
  sorry

end no_real_solution_l918_91803


namespace difference_max_min_is_7_l918_91844

-- Define the number of times Kale mowed his lawn during each season
def timesSpring : ℕ := 8
def timesSummer : ℕ := 5
def timesFall : ℕ := 12

-- Statement to prove
theorem difference_max_min_is_7 : 
  (max timesSpring (max timesSummer timesFall)) - (min timesSpring (min timesSummer timesFall)) = 7 :=
by
  -- Proof would go here
  sorry

end difference_max_min_is_7_l918_91844


namespace part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l918_91816

-- Part (1)
theorem part1_inequality (m : ℝ) : (∀ x : ℝ, (m^2 + 1)*x^2 - (2*m - 1)*x + 1 > 0) ↔ m > -3/4 := sorry

-- Part (2)
theorem part2_inequality_case1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ x < 1 ∨ x > 1/a) := sorry

theorem part2_inequality_case2 : 
  (∀ x : ℝ, (x - 1)*(0*x - 1) > 0 ↔ x < 1) := sorry

theorem part2_inequality_case3 (a : ℝ) (h : a < 0) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ 1/a < x ∧ x < 1) := sorry

end part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l918_91816


namespace smallest_possible_N_l918_91811

theorem smallest_possible_N (p q r s t : ℕ) (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0)
(h_sum : p + q + r + s + t = 2022) :
    ∃ N : ℕ, N = 506 ∧ N = max (p + q) (max (q + r) (max (r + s) (s + t))) :=
by
    sorry

end smallest_possible_N_l918_91811


namespace calculate_expression_l918_91889

theorem calculate_expression :
  (Real.sqrt 2 - 3)^0 - Real.sqrt 9 + |(-2: ℝ)| + ((-1/3: ℝ)⁻¹)^2 = 9 :=
by
  sorry

end calculate_expression_l918_91889


namespace area_of_garden_l918_91820

-- Define the garden properties
variables {l w : ℕ}

-- Calculate length from the condition of walking length 30 times
def length_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Calculate perimeter from the condition of walking perimeter 12 times
def perimeter_of_garden (total_distance : ℕ) (times : ℕ) := total_distance / times

-- Define the proof statement
theorem area_of_garden (total_distance : ℕ) (times_length_walk : ℕ) (times_perimeter_walk : ℕ)
  (h1 : length_of_garden total_distance times_length_walk = l)
  (h2 : perimeter_of_garden total_distance (2 * times_perimeter_walk) = 2 * (l + w)) :
  l * w = 400 := 
sorry

end area_of_garden_l918_91820


namespace true_propositions_count_l918_91809

theorem true_propositions_count
  (a b c : ℝ)
  (h : a > b) :
  ( (a > b → a * c^2 > b * c^2) ∧
    (a * c^2 > b * c^2 → a > b) ∧
    (a ≤ b → a * c^2 ≤ b * c^2) ∧
    (a * c^2 ≤ b * c^2 → a ≤ b) 
  ) ∧ 
  (¬(a > b → a * c^2 > b * c^2) ∧
   ¬(a * c^2 ≤ b * c^2 → a ≤ b)) →
  (a * c^2 > b * c^2 → a > b) ∧
  (a ≤ b → a * c^2 ≤ b * c^2) ∨
  (a > b → a * c^2 > b * c^2) ∨
  (a * c^2 ≤ b * c^2 → a ≤ b) :=
sorry

end true_propositions_count_l918_91809


namespace last_digit_of_1_div_3_pow_9_is_7_l918_91800

noncomputable def decimal_expansion_last_digit (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem last_digit_of_1_div_3_pow_9_is_7 :
  decimal_expansion_last_digit 1 (3^9) = 7 :=
by
  sorry

end last_digit_of_1_div_3_pow_9_is_7_l918_91800


namespace pumpkins_eaten_l918_91812

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l918_91812


namespace largest_prime_divisor_36_squared_plus_81_squared_l918_91813

-- Definitions of the key components in the problem
def a := 36
def b := 81
def expr := a^2 + b^2
def largest_prime_divisor (n : ℕ) : ℕ := sorry -- Assume this function can compute the largest prime divisor

-- Theorem stating the problem
theorem largest_prime_divisor_36_squared_plus_81_squared : largest_prime_divisor (36^2 + 81^2) = 53 := 
  sorry

end largest_prime_divisor_36_squared_plus_81_squared_l918_91813


namespace arithmetic_geometric_sequence_solution_l918_91817

theorem arithmetic_geometric_sequence_solution (u v : ℕ → ℝ) (a b u₀ : ℝ) :
  (∀ n, u (n + 1) = a * u n + b) ∧ (∀ n, v (n + 1) = a * v n + b) →
  u 0 = u₀ →
  v 0 = b / (1 - a) →
  ∀ n, u n = a ^ n * (u₀ - b / (1 - a)) + b / (1 - a) :=
by
  intros
  sorry

end arithmetic_geometric_sequence_solution_l918_91817


namespace no_solution_range_has_solution_range_l918_91808

open Real

theorem no_solution_range (a : ℝ) : (∀ x, ¬ (|x - 4| + |3 - x| < a)) ↔ a ≤ 1 := 
sorry

theorem has_solution_range (a : ℝ) : (∃ x, |x - 4| + |3 - x| < a) ↔ 1 < a :=
sorry

end no_solution_range_has_solution_range_l918_91808


namespace trig_identity_l918_91877

theorem trig_identity :
  (4 * (1 / 2) - Real.sqrt 2 * (Real.sqrt 2 / 2) - Real.sqrt 3 * (Real.sqrt 3 / 3) + 2 * (Real.sqrt 3 / 2)) = Real.sqrt 3 :=
by sorry

end trig_identity_l918_91877
