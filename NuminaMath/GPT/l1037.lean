import Mathlib

namespace radar_placement_problem_l1037_103720

noncomputable def max_distance (n : ℕ) (coverage_radius : ℝ) (central_angle : ℝ) : ℝ :=
  coverage_radius / Real.sin (central_angle / 2)

noncomputable def ring_area (inner_radius : ℝ) (outer_radius : ℝ) : ℝ :=
  Real.pi * (outer_radius ^ 2 - inner_radius ^ 2)

theorem radar_placement_problem (r : ℝ := 13) (n : ℕ := 5) (width : ℝ := 10) :
  let angle := 2 * Real.pi / n
  let max_dist := max_distance n r angle
  let inner_radius := (r ^ 2 - (r - width) ^ 2) / Real.tan (angle / 2)
  let outer_radius := inner_radius + width
  max_dist = 12 / Real.sin (angle / 2) ∧
  ring_area inner_radius outer_radius = 240 * Real.pi / Real.tan (angle / 2) :=
by
  sorry

end radar_placement_problem_l1037_103720


namespace evaluate_ceiling_neg_cubed_frac_l1037_103740

theorem evaluate_ceiling_neg_cubed_frac :
  (Int.ceil ((- (5 : ℚ) / 3) ^ 3 + 1) = -3) :=
sorry

end evaluate_ceiling_neg_cubed_frac_l1037_103740


namespace sandra_beignets_16_weeks_l1037_103783

-- Define the constants used in the problem
def beignets_per_morning : ℕ := 3
def days_per_week : ℕ := 7
def weeks : ℕ := 16

-- Define the number of beignets Sandra eats in 16 weeks
def beignets_in_16_weeks : ℕ := beignets_per_morning * days_per_week * weeks

-- State the theorem
theorem sandra_beignets_16_weeks : beignets_in_16_weeks = 336 :=
by
  -- Provide a placeholder for the proof
  sorry

end sandra_beignets_16_weeks_l1037_103783


namespace diagonal_crosses_700_cubes_l1037_103713

noncomputable def num_cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c

theorem diagonal_crosses_700_cubes :
  num_cubes_crossed 200 300 350 = 700 :=
sorry

end diagonal_crosses_700_cubes_l1037_103713


namespace evaluate_expression_l1037_103764

theorem evaluate_expression : 
  -((5: ℤ) ^ 2) - (-(3: ℤ) ^ 3) * ((2: ℚ) / 9) - 9 * |((-(2: ℚ)) / 3)| = -25 := by
  sorry

end evaluate_expression_l1037_103764


namespace sum_of_integers_l1037_103744

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end sum_of_integers_l1037_103744


namespace sector_arc_length_l1037_103797

theorem sector_arc_length (n : ℝ) (r : ℝ) (l : ℝ) (h1 : n = 90) (h2 : r = 3) (h3 : l = (n * Real.pi * r) / 180) :
  l = (3 / 2) * Real.pi := by
  rw [h1, h2] at h3
  sorry

end sector_arc_length_l1037_103797


namespace ratio_of_novels_read_l1037_103751

theorem ratio_of_novels_read (jordan_read : ℕ) (alexandre_read : ℕ)
  (h_jordan_read : jordan_read = 120) 
  (h_diff : jordan_read = alexandre_read + 108) :
  alexandre_read / jordan_read = 1 / 10 :=
by
  -- Proof skipped
  sorry

end ratio_of_novels_read_l1037_103751


namespace quiz_score_difference_l1037_103776

theorem quiz_score_difference :
  let percentage_70 := 0.10
  let percentage_80 := 0.35
  let percentage_90 := 0.30
  let percentage_100 := 0.25
  let mean_score := (percentage_70 * 70) + (percentage_80 * 80) + (percentage_90 * 90) + (percentage_100 * 100)
  let median_score := 90
  mean_score = 87 → median_score - mean_score = 3 :=
by
  sorry

end quiz_score_difference_l1037_103776


namespace solution_set_of_abs_x_gt_1_l1037_103771

theorem solution_set_of_abs_x_gt_1 (x : ℝ) : |x| > 1 ↔ x > 1 ∨ x < -1 := 
sorry

end solution_set_of_abs_x_gt_1_l1037_103771


namespace pyramid_base_edge_length_l1037_103796

-- Prove that the edge-length of the base of the pyramid is as specified
theorem pyramid_base_edge_length
  (r h : ℝ)
  (hemisphere_radius : r = 3)
  (pyramid_height : h = 8)
  (tangency_condition : true) : true :=
by
  sorry

end pyramid_base_edge_length_l1037_103796


namespace ethanol_in_tank_l1037_103793

theorem ethanol_in_tank (capacity fuel_a fuel_b : ℝ)
  (ethanol_a ethanol_b : ℝ)
  (h1 : capacity = 218)
  (h2 : fuel_a = 122)
  (h3 : fuel_b = capacity - fuel_a)
  (h4 : ethanol_a = 0.12)
  (h5 : ethanol_b = 0.16) :
  fuel_a * ethanol_a + fuel_b * ethanol_b = 30 := 
by {
  sorry
}

end ethanol_in_tank_l1037_103793


namespace number_one_fourth_less_than_25_percent_more_l1037_103769

theorem number_one_fourth_less_than_25_percent_more (x : ℝ) :
  (3 / 4) * x = 1.25 * 80 → x = 133.33 :=
by
  intros h
  sorry

end number_one_fourth_less_than_25_percent_more_l1037_103769


namespace time_spent_watching_movies_l1037_103724

def total_flight_time_minutes : ℕ := 11 * 60 + 20
def time_reading_minutes : ℕ := 2 * 60
def time_eating_dinner_minutes : ℕ := 30
def time_listening_radio_minutes : ℕ := 40
def time_playing_games_minutes : ℕ := 1 * 60 + 10
def time_nap_minutes : ℕ := 3 * 60

theorem time_spent_watching_movies :
  total_flight_time_minutes
  - time_reading_minutes
  - time_eating_dinner_minutes
  - time_listening_radio_minutes
  - time_playing_games_minutes
  - time_nap_minutes = 4 * 60 := by
  sorry

end time_spent_watching_movies_l1037_103724


namespace tournament_participants_l1037_103754

theorem tournament_participants (n : ℕ) (h₁ : 2 * (n * (n - 1) / 2 + 4) - (n - 2) * (n - 3) - 16 = 124) : n = 13 :=
sorry

end tournament_participants_l1037_103754


namespace average_transformation_l1037_103778

theorem average_transformation (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_avg : (a_1 + a_2 + a_3 + a_4 + a_5) / 5 = 8) : 
  ((a_1 + 10) + (a_2 - 10) + (a_3 + 10) + (a_4 - 10) + (a_5 + 10)) / 5 = 10 := 
by
  sorry

end average_transformation_l1037_103778


namespace count_4_digit_multiples_of_5_is_9_l1037_103752

noncomputable def count_4_digit_multiples_of_5 : Nat :=
  let digits := [2, 7, 4, 5]
  let last_digit := 5
  let remaining_digits := [2, 7, 4]
  let case_1 := 3
  let case_2 := 3 * 2
  case_1 + case_2

theorem count_4_digit_multiples_of_5_is_9 : count_4_digit_multiples_of_5 = 9 :=
by
  sorry

end count_4_digit_multiples_of_5_is_9_l1037_103752


namespace sum_of_sequence_correct_l1037_103798

def calculateSumOfSequence : ℚ :=
  (4 / 3) + (7 / 5) + (11 / 8) + (19 / 15) + (35 / 27) + (67 / 52) - 9

theorem sum_of_sequence_correct :
  calculateSumOfSequence = (-17312.5 / 7020) := by
  sorry

end sum_of_sequence_correct_l1037_103798


namespace age_problem_solution_l1037_103705

theorem age_problem_solution :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
  a1 + a2 + a3 = 54 ∧
  a5 - a4 = 5 ∧
  a3 + a4 + a5 = 78 ∧
  a2 - a1 = 7 ∧
  a1 + a5 = 44 ∧
  a1 = 13 ∧
  a2 = 20 ∧
  a3 = 21 ∧
  a4 = 26 ∧
  a5 = 31 :=
by
  -- We should skip the implementation because the solution is provided in the original problem.
  sorry

end age_problem_solution_l1037_103705


namespace right_handed_players_total_l1037_103785

-- Definitions of the given quantities
def total_players : ℕ := 70
def throwers : ℕ := 49
def non_throwers : ℕ := total_players - throwers
def one_third_non_throwers : ℕ := non_throwers / 3
def left_handed_non_throwers : ℕ := one_third_non_throwers
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def right_handed_throwers : ℕ := throwers
def total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers

-- The theorem stating the main proof goal
theorem right_handed_players_total (h1 : total_players = 70)
                                   (h2 : throwers = 49)
                                   (h3 : total_players - throwers = non_throwers)
                                   (h4 : non_throwers = 21) -- derived from the above
                                   (h5 : non_throwers / 3 = left_handed_non_throwers)
                                   (h6 : non_throwers - left_handed_non_throwers = right_handed_non_throwers)
                                   (h7 : right_handed_throwers = throwers)
                                   (h8 : total_right_handed = right_handed_throwers + right_handed_non_throwers) :
  total_right_handed = 63 := sorry

end right_handed_players_total_l1037_103785


namespace range_of_a_l1037_103721

-- Definitions of position conditions in the 4th quadrant
def PosInFourthQuad (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- Statement to prove
theorem range_of_a (a : ℝ) (h : PosInFourthQuad (2 * a + 4) (3 * a - 6)) : -2 < a ∧ a < 2 :=
  sorry

end range_of_a_l1037_103721


namespace simplify_fractions_l1037_103706

theorem simplify_fractions : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end simplify_fractions_l1037_103706


namespace no_rational_solution_l1037_103787

theorem no_rational_solution :
  ¬ ∃ (x y z : ℚ), 
  x + y + z = 0 ∧ x^2 + y^2 + z^2 = 100 := sorry

end no_rational_solution_l1037_103787


namespace find_A_when_A_clubsuit_7_equals_61_l1037_103702

-- Define the operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- Define the main problem statement
theorem find_A_when_A_clubsuit_7_equals_61 : 
  ∃ A : ℝ, clubsuit A 7 = 61 ∧ A = (2 * Real.sqrt 30) / 3 :=
by
  sorry

end find_A_when_A_clubsuit_7_equals_61_l1037_103702


namespace Angle_CNB_20_l1037_103728

theorem Angle_CNB_20 :
  ∀ (A B C N : Type) 
    (AC BC : Prop) 
    (angle_ACB : ℕ)
    (angle_NAC : ℕ)
    (angle_NCA : ℕ), 
    (AC ↔ BC) →
    angle_ACB = 98 →
    angle_NAC = 15 →
    angle_NCA = 21 →
    ∃ angle_CNB, angle_CNB = 20 :=
by
  sorry

end Angle_CNB_20_l1037_103728


namespace alton_weekly_profit_l1037_103742

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end alton_weekly_profit_l1037_103742


namespace golden_ratio_eqn_value_of_ab_value_of_pq_n_l1037_103717

-- Part (1): Finding the golden ratio
theorem golden_ratio_eqn {x : ℝ} (h1 : x^2 + x - 1 = 0) : x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

-- Part (2): Finding the value of ab
theorem value_of_ab {a b m : ℝ} (h1 : a^2 + m * a = 1) (h2 : b^2 - 2 * m * b = 4) (h3 : b ≠ -2 * a) : a * b = 2 :=
sorry

-- Part (3): Finding the value of pq - n
theorem value_of_pq_n {p q n : ℝ} (h1 : p ≠ q) (eq1 : p^2 + n * p - 1 = q) (eq2 : q^2 + n * q - 1 = p) : p * q - n = 0 :=
sorry

end golden_ratio_eqn_value_of_ab_value_of_pq_n_l1037_103717


namespace middle_guards_hours_l1037_103794

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end middle_guards_hours_l1037_103794


namespace tangent_line_at_zero_l1037_103750

noncomputable def curve (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, (curve x) = m * x + b) ∧
    m = 2 ∧ b = 1 :=
by 
  sorry

end tangent_line_at_zero_l1037_103750


namespace reciprocals_of_product_one_l1037_103722

theorem reciprocals_of_product_one (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x :=
by 
  sorry

end reciprocals_of_product_one_l1037_103722


namespace john_candies_correct_l1037_103791

variable (Bob_candies : ℕ) (Mary_candies : ℕ)
          (Sue_candies : ℕ) (Sam_candies : ℕ)
          (Total_candies : ℕ) (John_candies : ℕ)

axiom bob_has : Bob_candies = 10
axiom mary_has : Mary_candies = 5
axiom sue_has : Sue_candies = 20
axiom sam_has : Sam_candies = 10
axiom total_has : Total_candies = 50

theorem john_candies_correct : 
  Bob_candies + Mary_candies + Sue_candies + Sam_candies + John_candies = Total_candies → John_candies = 5 := by
sorry

end john_candies_correct_l1037_103791


namespace triple_hash_72_eq_7_25_l1037_103714

def hash (N : ℝ) : ℝ := 0.5 * N - 1

theorem triple_hash_72_eq_7_25 : hash (hash (hash 72)) = 7.25 :=
by
  sorry

end triple_hash_72_eq_7_25_l1037_103714


namespace find_common_difference_l1037_103748

-- Definitions based on conditions in a)
def common_difference_4_10 (a₁ d : ℝ) : Prop :=
  (a₁ + 3 * d) + (a₁ + 9 * d) = 0

def sum_relation (a₁ d : ℝ) : Prop :=
  2 * (12 * a₁ + 66 * d) = (2 * a₁ + d + 10)

-- Math proof problem statement
theorem find_common_difference (a₁ d : ℝ) 
  (h₁ : common_difference_4_10 a₁ d) 
  (h₂ : sum_relation a₁ d) : 
  d = -10 :=
sorry

end find_common_difference_l1037_103748


namespace find_stu_l1037_103784

open Complex

theorem find_stu (p q r s t u : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (h1 : p = (q + r) / (s - 3))
  (h2 : q = (p + r) / (t - 3))
  (h3 : r = (p + q) / (u - 3))
  (h4 : s * t + s * u + t * u = 8)
  (h5 : s + t + u = 4) :
  s * t * u = 10 := 
sorry

end find_stu_l1037_103784


namespace jane_percentage_bread_to_treats_l1037_103779

variable (T J_b W_b W_t : ℕ) (P : ℕ)

-- Conditions as stated
axiom h1 : J_b = (P * T) / 100
axiom h2 : W_t = T / 2
axiom h3 : W_b = 3 * W_t
axiom h4 : W_b = 90
axiom h5 : J_b + W_b + T + W_t = 225

theorem jane_percentage_bread_to_treats : P = 75 :=
by
-- Proof skeleton
sorry

end jane_percentage_bread_to_treats_l1037_103779


namespace product_modulo_7_l1037_103770

theorem product_modulo_7 : (1729 * 1865 * 1912 * 2023) % 7 = 6 :=
by
  sorry

end product_modulo_7_l1037_103770


namespace fixed_point_exists_l1037_103703

theorem fixed_point_exists : ∃ (x y : ℝ), (∀ k : ℝ, (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0) ∧ x = 2 ∧ y = 3 := 
by
  -- Placeholder for proof
  sorry

end fixed_point_exists_l1037_103703


namespace simplify_expression_l1037_103760

-- Define the expressions and the simplification statement
def expr1 (x : ℝ) := (3 * x - 6) * (x + 8)
def expr2 (x : ℝ) := (x + 6) * (3 * x - 2)
def simplified (x : ℝ) := 2 * x - 36

theorem simplify_expression (x : ℝ) : expr1 x - expr2 x = simplified x := by
  sorry

end simplify_expression_l1037_103760


namespace find_edge_lengths_sum_l1037_103757

noncomputable def sum_edge_lengths (a d : ℝ) (volume surface_area : ℝ) : ℝ :=
  if (a - d) * a * (a + d) = volume ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = surface_area then
    4 * ((a - d) + a + (a + d))
  else
    0

theorem find_edge_lengths_sum:
  (∃ a d : ℝ, (a - d) * a * (a + d) = 512 ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = 352) →
  sum_edge_lengths (Real.sqrt 59) 1 512 352 = 12 * Real.sqrt 59 :=
by
  sorry

end find_edge_lengths_sum_l1037_103757


namespace Ryan_stickers_l1037_103715

def Ryan_has_30_stickers (R S T : ℕ) : Prop :=
  S = 3 * R ∧ T = S + 20 ∧ R + S + T = 230 → R = 30

theorem Ryan_stickers : ∃ R S T : ℕ, Ryan_has_30_stickers R S T :=
sorry

end Ryan_stickers_l1037_103715


namespace total_calories_consumed_l1037_103799

-- Definitions for conditions
def calories_per_chip : ℕ := 60 / 10
def extra_calories_per_cheezit := calories_per_chip / 3
def calories_per_cheezit: ℕ := calories_per_chip + extra_calories_per_cheezit
def total_calories_chips : ℕ := 60
def total_calories_cheezits : ℕ := 6 * calories_per_cheezit

-- Main statement to be proved
theorem total_calories_consumed : total_calories_chips + total_calories_cheezits = 108 := by 
  sorry

end total_calories_consumed_l1037_103799


namespace f_nonneg_f_positive_f_zero_condition_l1037_103761

noncomputable def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) +
  B * (a^2 * b + b^2 * c + c^2 * a + a * b^2 + b * c^2 + c * a^2) +
  C * a * b * c

theorem f_nonneg (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 ≥ 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

theorem f_positive (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 > 0) 
  (h2 : f A B C 1 1 0 ≥ 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c > 0 :=
by sorry

theorem f_zero_condition (A B C a b c : ℝ) 
  (h1 : f A B C 1 1 1 = 0) 
  (h2 : f A B C 1 1 0 > 0)
  (h3 : f A B C 2 1 1 ≥ 0) : f A B C a b c ≥ 0 :=
by sorry

end f_nonneg_f_positive_f_zero_condition_l1037_103761


namespace probability_after_6_passes_l1037_103747

noncomputable section

-- We define people
inductive Person
| A | B | C

-- Probability that person A has the ball after n passes
def P : ℕ → Person → ℚ
| 0, Person.A => 1
| 0, _ => 0
| n+1, Person.A => (P n Person.B + P n Person.C) / 2
| n+1, Person.B => (P n Person.A + P n Person.C) / 2
| n+1, Person.C => (P n Person.A + P n Person.B) / 2

theorem probability_after_6_passes :
  P 6 Person.A = 11 / 32 := by
  sorry

end probability_after_6_passes_l1037_103747


namespace baseball_games_per_month_l1037_103729

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l1037_103729


namespace difference_between_first_and_third_l1037_103726

variable (x : ℕ)

-- Condition 1: The first number is twice the second.
def first_number : ℕ := 2 * x

-- Condition 2: The first number is three times the third.
def third_number : ℕ := first_number x / 3

-- Condition 3: The average of the three numbers is 88.
def average_condition : Prop := (first_number x + x + third_number x) / 3 = 88

-- Prove that the difference between first and third number is 96.
theorem difference_between_first_and_third 
  (h : average_condition x) : first_number x - third_number x = 96 :=
by
  sorry -- Proof omitted

end difference_between_first_and_third_l1037_103726


namespace from20To25_l1037_103775

def canObtain25 (start : ℕ) : Prop :=
  ∃ (steps : ℕ → ℕ), steps 0 = start ∧ (∃ n, steps n = 25) ∧ 
  (∀ i, steps (i+1) = (steps i * 2) ∨ (steps (i+1) = steps i / 10))

theorem from20To25 : canObtain25 20 :=
sorry

end from20To25_l1037_103775


namespace no_integer_solution_exists_l1037_103772

theorem no_integer_solution_exists : ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8 * t - 1 := 
by sorry

end no_integer_solution_exists_l1037_103772


namespace dimes_difference_l1037_103738

theorem dimes_difference (a b c : ℕ) :
  a + b + c = 120 →
  5 * a + 10 * b + 25 * c = 1265 →
  c ≥ 10 →
  (max (b) - min (b)) = 92 :=
sorry

end dimes_difference_l1037_103738


namespace highest_y_coordinate_l1037_103749

theorem highest_y_coordinate : 
  (∀ x y : ℝ, ((x - 4)^2 / 25 + y^2 / 49 = 0) → y = 0) := 
by
  sorry

end highest_y_coordinate_l1037_103749


namespace total_number_of_items_l1037_103725

-- Define the conditions as equations in Lean
def model_cars_price := 5
def model_trains_price := 8
def total_amount := 31

-- Initialize the variable definitions for number of cars and trains
variables (c t : ℕ)

-- The proof problem: Show that given the equation, the sum of cars and trains is 5
theorem total_number_of_items : (model_cars_price * c + model_trains_price * t = total_amount) → (c + t = 5) := by
  -- Proof steps would go here
  sorry

end total_number_of_items_l1037_103725


namespace sum_of_n_natural_numbers_l1037_103786

theorem sum_of_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 1035) : n = 46 :=
sorry

end sum_of_n_natural_numbers_l1037_103786


namespace prob_two_packs_tablets_at_10am_dec31_l1037_103739
noncomputable def prob_two_packs_tablets (n : ℕ) : ℝ :=
  let numer := (2^n - 1)
  let denom := 2^(n-1) * n
  numer / denom

theorem prob_two_packs_tablets_at_10am_dec31 :
  prob_two_packs_tablets 10 = 1023 / 5120 := by
  sorry

end prob_two_packs_tablets_at_10am_dec31_l1037_103739


namespace proportion_estimation_chi_squared_test_l1037_103745

-- Definitions based on the conditions
def total_elders : ℕ := 500
def not_vaccinated_male : ℕ := 20
def not_vaccinated_female : ℕ := 10
def vaccinated_male : ℕ := 230
def vaccinated_female : ℕ := 240

-- Calculations based on the problem conditions
noncomputable def proportion_vaccinated : ℚ := (vaccinated_male + vaccinated_female) / total_elders

def chi_squared_statistic (a b c d n : ℕ) : ℚ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

noncomputable def K2_value : ℚ :=
  chi_squared_statistic not_vaccinated_male not_vaccinated_female vaccinated_male vaccinated_female total_elders

-- Specify the critical value for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Theorem statements (problems to prove)
theorem proportion_estimation : proportion_vaccinated = 94 / 100 := by
  sorry

theorem chi_squared_test : K2_value < critical_value_99 := by
  sorry

end proportion_estimation_chi_squared_test_l1037_103745


namespace find_x_squared_plus_y_squared_l1037_103719

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end find_x_squared_plus_y_squared_l1037_103719


namespace Yvonne_probability_of_success_l1037_103765

theorem Yvonne_probability_of_success
  (P_X : ℝ) (P_Z : ℝ) (P_XY_notZ : ℝ) :
  P_X = 1 / 3 →
  P_Z = 5 / 8 →
  P_XY_notZ = 0.0625 →
  ∃ P_Y : ℝ, P_Y = 0.5 :=
by
  intros hX hZ hXY_notZ
  existsi (0.5 : ℝ)
  sorry

end Yvonne_probability_of_success_l1037_103765


namespace percent_defective_shipped_l1037_103766

-- Conditions given in the problem
def percent_defective (percent_total_defective: ℝ) : Prop := percent_total_defective = 0.08
def percent_shipped_defective (percent_defective_shipped: ℝ) : Prop := percent_defective_shipped = 0.04

-- The main theorem we want to prove
theorem percent_defective_shipped (percent_total_defective percent_defective_shipped : ℝ) 
  (h1 : percent_defective percent_total_defective) (h2 : percent_shipped_defective percent_defective_shipped) : 
  (percent_total_defective * percent_defective_shipped * 100) = 0.32 :=
by
  sorry

end percent_defective_shipped_l1037_103766


namespace mike_profit_l1037_103704

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l1037_103704


namespace units_digit_3m_squared_plus_2m_l1037_103707

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m_l1037_103707


namespace height_of_new_TV_l1037_103767

theorem height_of_new_TV 
  (width1 height1 cost1 : ℝ) 
  (width2 cost2 : ℝ) 
  (cost_diff_per_sq_inch : ℝ) 
  (h1 : width1 = 24) 
  (h2 : height1 = 16) 
  (h3 : cost1 = 672) 
  (h4 : width2 = 48) 
  (h5 : cost2 = 1152) 
  (h6 : cost_diff_per_sq_inch = 1) : 
  ∃ height2 : ℝ, height2 = 32 :=
by
  sorry

end height_of_new_TV_l1037_103767


namespace Lenny_pens_left_l1037_103753

theorem Lenny_pens_left :
  let boxes := 20
  let pens_per_box := 5
  let total_pens := boxes * pens_per_box
  let pens_given_to_friends := 0.4 * total_pens
  let pens_left_after_friends := total_pens - pens_given_to_friends
  let pens_given_to_classmates := (1/4) * pens_left_after_friends
  let pens_left := pens_left_after_friends - pens_given_to_classmates
  pens_left = 45 :=
by
  repeat { sorry }

end Lenny_pens_left_l1037_103753


namespace find_second_number_l1037_103736

theorem find_second_number (x : ℝ) (h : (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8) : x = 40 :=
sorry

end find_second_number_l1037_103736


namespace b_came_third_four_times_l1037_103727

variable (a b c N : ℕ)

theorem b_came_third_four_times
    (a_pos : a > 0) 
    (b_pos : b > 0) 
    (c_pos : c > 0)
    (a_gt_b : a > b) 
    (b_gt_c : b > c) 
    (a_b_c_sum : a + b + c = 8)
    (score_A : 4 * a + b = 26) 
    (score_B : a + 4 * c = 11) 
    (score_C : 3 * b + 2 * c = 11) 
    (B_won_first_event : a + b + c = 8) : 
    4 * c = 4 := 
sorry

end b_came_third_four_times_l1037_103727


namespace geometric_sequence_a5_l1037_103718

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r)
  (h_eqn : ∃ x : ℝ, (x^2 + 7*x + 9 = 0) ∧ (a 3 = x) ∧ (a 7 = x)) :
  a 5 = 3 ∨ a 5 = -3 := 
sorry

end geometric_sequence_a5_l1037_103718


namespace factor_expression_l1037_103710

variable (x : ℝ)

def e : ℝ := (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5)

theorem factor_expression : e x = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by
  sorry

end factor_expression_l1037_103710


namespace minimum_n_value_l1037_103732

def satisfies_terms_condition (n : ℕ) : Prop :=
  (n + 1) * (n + 1) ≥ 2021

theorem minimum_n_value :
  ∃ n : ℕ, n > 0 ∧ satisfies_terms_condition n ∧ ∀ m : ℕ, m > 0 ∧ satisfies_terms_condition m → n ≤ m := by
  sorry

end minimum_n_value_l1037_103732


namespace oscar_cookie_baking_time_l1037_103780

theorem oscar_cookie_baking_time : 
  (1 / 5) + (1 / 6) + (1 / o) - (1 / 4) = (1 / 8) → o = 120 := by
  sorry

end oscar_cookie_baking_time_l1037_103780


namespace R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l1037_103768

theorem R_H_nonneg_def (H : ℝ) (s t : ℝ) (hH : 0 < H ∧ H ≤ 1) :
  (1 / 2) * (|t| ^ (2 * H) + |s| ^ (2 * H) - |t - s| ^ (2 * H)) ≥ 0 := sorry

theorem R_K_nonneg_def (K : ℝ) (s t : ℝ) (hK : 0 < K ∧ K ≤ 2) :
  (1 / 2 ^ K) * (|t + s| ^ K - |t - s| ^ K) ≥ 0 := sorry

theorem R_HK_nonneg_def (H K : ℝ) (s t : ℝ) (hHK : 0 < H ∧ H ≤ 1 ∧ 0 < K ∧ K ≤ 1) :
  (1 / 2 ^ K) * ( (|t| ^ (2 * H) + |s| ^ (2 * H)) ^ K - |t - s| ^ (2 * H * K) ) ≥ 0 := sorry

end R_H_nonneg_def_R_K_nonneg_def_R_HK_nonneg_def_l1037_103768


namespace geometric_sequence_a6_l1037_103723

theorem geometric_sequence_a6 (a : ℕ → ℝ) (geometric_seq : ∀ n, a (n + 1) = a n * a 1)
  (h1 : (a 4) * (a 8) = 9) (h2 : (a 4) + (a 8) = -11) : a 6 = -3 := by
  sorry

end geometric_sequence_a6_l1037_103723


namespace smallest_gcd_12a_20b_l1037_103777

theorem smallest_gcd_12a_20b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : Nat.gcd a b = 18) :
  Nat.gcd (12 * a) (20 * b) = 72 := sorry

end smallest_gcd_12a_20b_l1037_103777


namespace pencil_price_units_l1037_103788

noncomputable def price_pencil (base_price: ℕ) (extra_cost: ℕ): ℝ :=
  (base_price + extra_cost) / 10000.0

theorem pencil_price_units (base_price: ℕ) (extra_cost: ℕ) (h_base: base_price = 5000) (h_extra: extra_cost = 20) : 
  price_pencil base_price extra_cost = 0.5 := by
  sorry

end pencil_price_units_l1037_103788


namespace initial_deposit_l1037_103733

theorem initial_deposit :
  ∀ (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ),
    r = 0.05 → n = 1 → t = 2 → P * (1 + r / n) ^ (n * t) = 6615 → P = 6000 :=
by
  intros P r n t h_r h_n h_t h_eq
  rw [h_r, h_n, h_t] at h_eq
  norm_num at h_eq
  sorry

end initial_deposit_l1037_103733


namespace reciprocal_of_x_l1037_103716

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2 * x^2 = 0) (h2 : x ≠ 0) : x = 2 → (1 / x = 1 / 2) :=
by {
  sorry
}

end reciprocal_of_x_l1037_103716


namespace alpha_tan_beta_gt_beta_tan_alpha_l1037_103735

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) 
: α * Real.tan β > β * Real.tan α := 
sorry

end alpha_tan_beta_gt_beta_tan_alpha_l1037_103735


namespace number_of_players_per_game_l1037_103789

def total_players : ℕ := 50
def total_games : ℕ := 1225

-- If each player plays exactly one game with each of the other players,
-- there are C(total_players, 2) = total_games games.
theorem number_of_players_per_game : ∃ k : ℕ, k = 2 ∧ (total_players * (total_players - 1)) / 2 = total_games := 
  sorry

end number_of_players_per_game_l1037_103789


namespace daisies_multiple_of_4_l1037_103711

def num_roses := 8
def num_daisies (D : ℕ) := D
def num_marigolds := 48
def num_arrangements := 4

theorem daisies_multiple_of_4 (D : ℕ) 
  (h_roses_div_4 : num_roses % num_arrangements = 0)
  (h_marigolds_div_4 : num_marigolds % num_arrangements = 0)
  (h_total_div_4 : (num_roses + num_daisies D + num_marigolds) % num_arrangements = 0) :
  D % 4 = 0 :=
sorry

end daisies_multiple_of_4_l1037_103711


namespace frank_whack_a_mole_tickets_l1037_103755

variable (W : ℕ)
variable (skee_ball_tickets : ℕ := 9)
variable (candy_cost : ℕ := 6)
variable (candies_bought : ℕ := 7)
variable (total_tickets : ℕ := W + skee_ball_tickets)
variable (required_tickets : ℕ := candy_cost * candies_bought)

theorem frank_whack_a_mole_tickets : W + skee_ball_tickets = required_tickets → W = 33 := by
  sorry

end frank_whack_a_mole_tickets_l1037_103755


namespace area_of_triangle_PQR_l1037_103792

noncomputable def point := ℝ × ℝ

def P : point := (1, 1)
def Q : point := (4, 1)
def R : point := (3, 4)

def triangle_area (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

theorem area_of_triangle_PQR :
  triangle_area P Q R = 9 / 2 :=
by
  sorry

end area_of_triangle_PQR_l1037_103792


namespace multiples_count_l1037_103781

theorem multiples_count (count_5 count_7 count_35 count_total : ℕ) :
  count_5 = 600 →
  count_7 = 428 →
  count_35 = 85 →
  count_total = count_5 + count_7 - count_35 →
  count_total = 943 :=
by
  sorry

end multiples_count_l1037_103781


namespace sin_alpha_trig_expression_l1037_103712

theorem sin_alpha {α : ℝ} (hα : ∃ P : ℝ × ℝ, P = (4/5, -3/5)) :
  Real.sin α = -3/5 :=
sorry

theorem trig_expression {α : ℝ} 
  (hα : Real.sin α = -3/5) : 
  (Real.sin (π / 2 - α) / Real.sin (α + π)) - 
  (Real.tan (α - π) / Real.cos (3 * π - α)) = 19 / 48 :=
sorry

end sin_alpha_trig_expression_l1037_103712


namespace tan_add_pi_over_3_l1037_103734

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end tan_add_pi_over_3_l1037_103734


namespace pair_green_shirts_l1037_103741

/-- In a regional math gathering, 83 students wore red shirts, and 97 students wore green shirts. 
The 180 students are grouped into 90 pairs. Exactly 35 of these pairs consist of students both 
wearing red shirts. Prove that the number of pairs consisting solely of students wearing green shirts is 42. -/
theorem pair_green_shirts (r g total pairs rr: ℕ) (h_r : r = 83) (h_g : g = 97) (h_total : total = 180) 
    (h_pairs : pairs = 90) (h_rr : rr = 35) : 
    (g - (r - rr * 2)) / 2 = 42 := 
by 
  /- The proof is omitted. -/
  sorry

end pair_green_shirts_l1037_103741


namespace billy_used_54_tickets_l1037_103758

-- Definitions
def ferris_wheel_rides := 7
def bumper_car_rides := 3
def ferris_wheel_cost := 6
def bumper_car_cost := 4

-- Theorem Statement
theorem billy_used_54_tickets : 
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := 
by
  sorry

end billy_used_54_tickets_l1037_103758


namespace face_value_stock_l1037_103743

-- Given conditions
variables (F : ℝ) (yield quoted_price dividend_rate : ℝ)
variables (h_yield : yield = 20) (h_quoted_price : quoted_price = 125)
variables (h_dividend_rate : dividend_rate = 0.25)

--Theorem to prove the face value of the stock is 100
theorem face_value_stock : (dividend_rate * F / quoted_price) * 100 = yield ↔ F = 100 :=
by
  sorry

end face_value_stock_l1037_103743


namespace factor_of_quadratic_l1037_103790

theorem factor_of_quadratic (m : ℝ) : (∀ x, (x + 6) * (x + a) = x ^ 2 - mx - 42) → m = 1 :=
by sorry

end factor_of_quadratic_l1037_103790


namespace ten_percent_of_x_l1037_103730

variable (certain_value : ℝ)
variable (x : ℝ)

theorem ten_percent_of_x (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = certain_value) :
  0.1 * x = 0.7 * (1.5 - certain_value) := sorry

end ten_percent_of_x_l1037_103730


namespace lines_in_4_by_4_grid_l1037_103782

-- Definition for the grid and the number of lattice points.
def grid : Nat := 16

-- Theorem stating that the number of different lines passing through at least two points in a 4-by-4 grid of lattice points.
theorem lines_in_4_by_4_grid : 
  (number_of_lines : Nat) → number_of_lines = 40 ↔ grid = 16 := 
by
  -- Calculating number of lines passing through at least two points in a 4-by-4 grid.
  sorry -- proof skipped

end lines_in_4_by_4_grid_l1037_103782


namespace bus_fare_one_way_cost_l1037_103746

-- Define the conditions
def zoo_entry (dollars : ℕ) : ℕ := dollars -- Zoo entry cost is $5 per person
def initial_money : ℕ := 40 -- They bring $40 with them
def money_left : ℕ := 24 -- They have $24 left after spending on zoo entry and bus fare

-- Given values
def noah_ava : ℕ := 2 -- Number of persons, Noah and Ava
def zoo_entry_cost : ℕ := 5 -- $5 per person for zoo entry
def total_money_spent := initial_money - money_left -- Money spent on zoo entry and bus fare

-- Function to calculate the total cost based on bus fare x
def total_cost (x : ℕ) : ℕ := noah_ava * zoo_entry_cost + 2 * noah_ava * x

-- Assertion to be proved
theorem bus_fare_one_way_cost : 
  ∃ (x : ℕ), total_cost x = total_money_spent ∧ x = 150 / 100 := sorry

end bus_fare_one_way_cost_l1037_103746


namespace josh_paid_6_dollars_l1037_103773

def packs : ℕ := 3
def cheesePerPack : ℕ := 20
def costPerCheese : ℕ := 10 -- cost in cents

theorem josh_paid_6_dollars :
  (packs * cheesePerPack * costPerCheese) / 100 = 6 :=
by
  sorry

end josh_paid_6_dollars_l1037_103773


namespace train_length_calculation_l1037_103762

noncomputable def length_of_train 
  (time : ℝ) (speed_train : ℝ) (speed_man : ℝ) : ℝ :=
  let speed_relative := speed_train - speed_man
  let speed_relative_mps := speed_relative * (5 / 18)
  speed_relative_mps * time

theorem train_length_calculation :
  length_of_train 29.997600191984642 63 3 = 1666.67 := 
by
  sorry

end train_length_calculation_l1037_103762


namespace Evan_dog_weight_l1037_103759

-- Define the weights of the dogs as variables
variables (E I : ℕ)

-- Conditions given in the problem
def Evan_dog_weight_wrt_Ivan (I : ℕ) : ℕ := 7 * I
def dogs_total_weight (E I : ℕ) : Prop := E + I = 72

-- Correct answer we need to prove
theorem Evan_dog_weight (h1 : Evan_dog_weight_wrt_Ivan I = E)
                          (h2 : dogs_total_weight E I)
                          (h3 : I = 9) : E = 63 :=
by
  sorry

end Evan_dog_weight_l1037_103759


namespace range_of_function_l1037_103700

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x + 2)) ↔ (y ∈ Set.Iio 1 ∨ y ∈ Set.Ioi 1) := 
sorry

end range_of_function_l1037_103700


namespace value_of_a3_plus_a5_l1037_103795

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a : ℕ → α) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem value_of_a3_plus_a5 (a : ℕ → α) (S : ℕ → α)
  (h_sequence : arithmetic_sequence a)
  (h_S7 : S 7 = 14)
  (h_sum_formula : ∀ n, S n = n * (a 1 + a n) / 2) :
  a 3 + a 5 = 4 :=
by
  sorry

end value_of_a3_plus_a5_l1037_103795


namespace possible_values_of_inverse_sum_l1037_103701

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end possible_values_of_inverse_sum_l1037_103701


namespace min_value_of_frac_l1037_103774

open Real

theorem min_value_of_frac (x : ℝ) (hx : x > 0) : 
  ∃ (t : ℝ), t = 2 * sqrt 5 + 2 ∧ (∀ y, y > 0 → (x^2 + 2 * x + 5) / x ≥ t) :=
by
  sorry

end min_value_of_frac_l1037_103774


namespace triangle_angles_are_30_60_90_l1037_103763

theorem triangle_angles_are_30_60_90
  (a b c OH R r : ℝ)
  (h1 : OH = c / 2)
  (h2 : OH = a)
  (h3 : a < b)
  (h4 : b < c)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  ∃ (A B C : ℝ), (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) :=
sorry

end triangle_angles_are_30_60_90_l1037_103763


namespace ratio_of_means_l1037_103709

-- Variables for means
variables (xbar ybar zbar : ℝ)
-- Variables for sample sizes
variables (m n : ℕ)

-- Given conditions
def mean_x (x : ℕ) (xbar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ x → xbar = xbar
def mean_y (y : ℕ) (ybar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ y → ybar = ybar
def combined_mean (m n : ℕ) (xbar ybar zbar : ℝ) := zbar = (1/4) * xbar + (3/4) * ybar

-- Assertion to be proved
theorem ratio_of_means (h1 : mean_x m xbar) (h2 : mean_y n ybar)
  (h3 : xbar ≠ ybar) (h4 : combined_mean m n xbar ybar zbar) :
  m / n = 1 / 3 := sorry

end ratio_of_means_l1037_103709


namespace max_abs_value_l1037_103737

theorem max_abs_value (x y : ℝ) (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : |x - 2 * y + 1| ≤ 6 :=
sorry

end max_abs_value_l1037_103737


namespace kamal_average_marks_l1037_103708

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks_l1037_103708


namespace pencil_partition_l1037_103731

theorem pencil_partition (total_length green_fraction green_length remaining_length white_fraction half_remaining white_length gold_length : ℝ)
  (h1 : green_fraction = 7 / 10)
  (h2 : total_length = 2)
  (h3 : green_length = green_fraction * total_length)
  (h4 : remaining_length = total_length - green_length)
  (h5 : white_fraction = 1 / 2)
  (h6 : white_length = white_fraction * remaining_length)
  (h7 : gold_length = remaining_length - white_length) :
  (gold_length / remaining_length) = 1 / 2 :=
sorry

end pencil_partition_l1037_103731


namespace curtain_length_correct_l1037_103756

-- Define the problem conditions in Lean
def room_height_feet : ℝ := 8
def feet_to_inches : ℝ := 12
def additional_material_inches : ℝ := 5

-- Define the target length of the curtains
def curtain_length_inches : ℝ :=
  (room_height_feet * feet_to_inches) + additional_material_inches

-- Statement to prove the length of the curtains is 101 inches.
theorem curtain_length_correct :
  curtain_length_inches = 101 := by
  sorry

end curtain_length_correct_l1037_103756
