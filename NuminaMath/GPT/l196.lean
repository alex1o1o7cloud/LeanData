import Mathlib

namespace probability_two_students_next_to_each_other_l196_196075

theorem probability_two_students_next_to_each_other : (2 * Nat.factorial 9) / Nat.factorial 10 = 1 / 5 :=
by
  sorry

end probability_two_students_next_to_each_other_l196_196075


namespace part1_part2_l196_196758

noncomputable def f (x a : ℝ) : ℝ := abs (x + 2 * a) + abs (x - 1)

noncomputable def g (a : ℝ) : ℝ := abs ((1 : ℝ) / a + 2 * a) + abs ((1 : ℝ) / a - 1)

theorem part1 (x : ℝ) : f x 1 ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 0) : g a ≤ 4 ↔ (1 / 2) ≤ a ∧ a ≤ (3 / 2) := by
  sorry

end part1_part2_l196_196758


namespace problem_solution_l196_196454

theorem problem_solution (k a b : ℝ) (h1 : k = a + Real.sqrt b) 
  (h2 : abs (Real.logb 5 k - Real.logb 5 (k^2 + 3)) = 0.6) : 
  a + b = 15 :=
sorry

end problem_solution_l196_196454


namespace expected_replanted_seeds_l196_196742

open MeasureTheory ProbabilityTheory

-- Define the conditions
def probability_of_germination : ℝ := 0.9
def total_seeds : ℕ := 1000
def seeds_to_replant_per_failure : ℕ := 2

-- Define the binomial distribution and the expected value for failures
noncomputable def probability_of_failure : ℝ := 1 - probability_of_germination
noncomputable def number_of_failures : BinomialDistribution where
  n := total_seeds
  p := probability_of_failure

-- Main theorem: Expected number of replanted seeds
theorem expected_replanted_seeds : 
  (2 * (total_seeds * probability_of_failure : ℝ)) = 200 :=
by
  -- The proper proof goes here, but we use sorry to complete the theorem statement
  sorry

end expected_replanted_seeds_l196_196742


namespace trigonometric_range_l196_196941

open Real

theorem trigonometric_range (α : ℝ) :
  (0 < α ∧ α < 2 * π) ∧ (sin α < sqrt 3 / 2) ∧ (cos α > 1 / 2) →
  (0 < α ∧ α < π / 3) ∨ (5 * π / 3 < α ∧ α < 2 * π) :=
by
  sorry

end trigonometric_range_l196_196941


namespace average_speed_first_part_l196_196253

noncomputable def speed_of_first_part (v : ℝ) : Prop :=
  let distance_first_part := 124
  let speed_second_part := 60
  let distance_second_part := 250 - distance_first_part
  let total_time := 5.2
  (distance_first_part / v) + (distance_second_part / speed_second_part) = total_time

theorem average_speed_first_part : speed_of_first_part 40 :=
  sorry

end average_speed_first_part_l196_196253


namespace new_cylinder_height_percentage_l196_196695

variables (r h h_new : ℝ)

theorem new_cylinder_height_percentage :
  (7 / 8) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new →
  (h_new / h) = 14 / 15 :=
by
  intro h_volume_eq
  sorry

end new_cylinder_height_percentage_l196_196695


namespace number_of_questions_in_test_l196_196644

-- Definitions based on the conditions:
def marks_per_question : ℕ := 2
def jose_wrong_questions : ℕ := 5  -- number of questions Jose got wrong
def total_combined_score : ℕ := 210  -- total score of Meghan, Jose, and Alisson combined

-- Let A be Alisson's score
variables (A Jose Meghan : ℕ)

-- Conditions
axiom joe_more_than_alisson : Jose = A + 40
axiom megh_less_than_jose : Meghan = Jose - 20
axiom combined_scores : A + Jose + Meghan = total_combined_score

-- Function to compute the total possible score for Jose without wrong answers:
noncomputable def jose_improvement_score : ℕ := Jose + (jose_wrong_questions * marks_per_question)

-- Proof problem statement
theorem number_of_questions_in_test :
  (jose_improvement_score Jose) / marks_per_question = 50 :=
by
  -- Sorry is used here to indicate that the proof is omitted.
  sorry

end number_of_questions_in_test_l196_196644


namespace find_xy_l196_196283

theorem find_xy (x y : ℝ) :
  (x - 8) ^ 2 + (y - 9) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ 
  (x = 25 / 3 ∧ y = 26 / 3) :=
by
  sorry

end find_xy_l196_196283


namespace minimum_value_g_l196_196156

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x > 0, g x ≥ 4 ∧ (∃ x > 0, g x = 4) :=
by
  sorry

end minimum_value_g_l196_196156


namespace change_in_mean_and_median_l196_196364

-- Original attendance data
def original_data : List ℕ := [15, 23, 17, 19, 17, 20]

-- Corrected attendance data
def corrected_data : List ℕ := [15, 23, 17, 19, 17, 25]

-- Function to compute mean
def mean (data: List ℕ) : ℚ := (data.sum : ℚ) / data.length

-- Function to compute median
def median (data: List ℕ) : ℚ :=
  let sorted := data.toArray.qsort (· ≤ ·) |>.toList
  if sorted.length % 2 == 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Lean statement verifying the expected change in mean and median
theorem change_in_mean_and_median :
  mean corrected_data - mean original_data = 1 ∧ median corrected_data = median original_data :=
by -- Note the use of 'by' to structure the proof
  sorry -- Proof omitted

end change_in_mean_and_median_l196_196364


namespace opposite_sides_range_l196_196050

theorem opposite_sides_range (a : ℝ) : (2 * 1 + 3 * a + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 := sorry

end opposite_sides_range_l196_196050


namespace calc_value_l196_196043

theorem calc_value (n : ℕ) (h : 1 ≤ n) : 
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := 
sorry

end calc_value_l196_196043


namespace period_of_cos_3x_l196_196832

theorem period_of_cos_3x :
  ∃ T : ℝ, (∀ x : ℝ, (Real.cos (3 * (x + T))) = Real.cos (3 * x)) ∧ (T = (2 * Real.pi) / 3) :=
sorry

end period_of_cos_3x_l196_196832


namespace floor_identity_l196_196800

theorem floor_identity (x : ℝ) : 
    (⌊(3 + x) / 6⌋ - ⌊(4 + x) / 6⌋ + ⌊(5 + x) / 6⌋ = ⌊(1 + x) / 2⌋ - ⌊(1 + x) / 3⌋) :=
by
  sorry

end floor_identity_l196_196800


namespace floor_abs_sum_eq_501_l196_196975

open Int

theorem floor_abs_sum_eq_501 (x : Fin 1004 → ℝ) (h : ∀ i, x i + (i : ℝ) + 1 = (Finset.univ.sum x) + 1005) : 
  Int.floor (abs (Finset.univ.sum x)) = 501 :=
by
  -- Proof steps will go here
  sorry

end floor_abs_sum_eq_501_l196_196975


namespace prob_heart_club_spade_l196_196554

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l196_196554


namespace rain_probability_weekend_l196_196983

theorem rain_probability_weekend :
  let p_rain_F := 0.60
  let p_rain_S := 0.70
  let p_rain_U := 0.40
  let p_no_rain_F := 1 - p_rain_F
  let p_no_rain_S := 1 - p_rain_S
  let p_no_rain_U := 1 - p_rain_U
  let p_no_rain_all_days := p_no_rain_F * p_no_rain_S * p_no_rain_U
  let p_rain_at_least_one_day := 1 - p_no_rain_all_days
  (p_rain_at_least_one_day * 100 = 92.8) := sorry

end rain_probability_weekend_l196_196983


namespace radius_of_sphere_l196_196702

theorem radius_of_sphere 
  (shadow_length_sphere : ℝ)
  (stick_height : ℝ)
  (stick_shadow : ℝ)
  (parallel_sun_rays : Prop) 
  (tan_θ : ℝ) 
  (h1 : tan_θ = stick_height / stick_shadow)
  (h2 : tan_θ = shadow_length_sphere / 20) :
  shadow_length_sphere / 20 = 1/4 → shadow_length_sphere = 5 := by
  sorry

end radius_of_sphere_l196_196702


namespace sum_is_402_3_l196_196120

def sum_of_numbers := 3 + 33 + 333 + 33.3

theorem sum_is_402_3 : sum_of_numbers = 402.3 := by
  sorry

end sum_is_402_3_l196_196120


namespace find_k_values_l196_196735

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2)

theorem find_k_values (k : ℝ) :
  (vector_norm (k * (3, -4) - (5, 8)) = 5 * sqrt 10) ∧
  (vector_norm (2 * k * (3, -4) + (10, 16)) = 10 * sqrt 10) ↔
  (k = 2.08) ∨ (k = -3.44) :=
by sorry

end find_k_values_l196_196735


namespace episodes_per_season_l196_196860

theorem episodes_per_season (S : ℕ) (E : ℕ) (H1 : S = 12) (H2 : 2/3 * E = 160) : E / S = 20 :=
by
  sorry

end episodes_per_season_l196_196860


namespace swimming_distance_l196_196004

theorem swimming_distance
  (t : ℝ) (d_up : ℝ) (d_down : ℝ) (v_man : ℝ) (v_stream : ℝ)
  (h1 : v_man = 5) (h2 : t = 5) (h3 : d_up = 20) 
  (h4 : d_up = (v_man - v_stream) * t) :
  d_down = (v_man + v_stream) * t :=
by
  sorry

end swimming_distance_l196_196004


namespace two_digit_factors_of_2_pow_18_minus_1_l196_196935

-- Define the main problem statement: 
-- How many two-digit factors does 2^18 - 1 have?

theorem two_digit_factors_of_2_pow_18_minus_1 : 
  ∃ n : ℕ, n = 5 ∧ ∀ f : ℕ, (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100) ↔ (f ∣ (2^18 - 1) ∧ 10 ≤ f ∧ f < 100 ∧ ∃ k : ℕ, (2^18 - 1) = k * f) :=
by sorry

end two_digit_factors_of_2_pow_18_minus_1_l196_196935


namespace evaluate_expression_l196_196731

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l196_196731


namespace area_ratio_l196_196321

noncomputable def initial_areas (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def misallocated_areas (a b : ℝ) :=
  let b' := b + 0.1 * a - 0.5 * b
  b' = 0.4 * (a + b)

noncomputable def final_ratios (a b c : ℝ) :=
  let a' := 0.9 * a + 0.5 * b
  let b' := b + 0.1 * a - 0.5 * b
  let c' := 0.5 * c
  a' + b' + c' = a + b + c ∧ a' / b' = 2 ∧ b' / c' = 1 

theorem area_ratio (a b c m : ℝ) (h1 : initial_areas a b c) 
  (h2 : misallocated_areas a b)
  (h3 : final_ratios a b c) : 
  (m = 0.4 * a) → (m / (a + b + c) = 1 / 20) :=
sorry

end area_ratio_l196_196321


namespace find_a_if_x_is_1_root_l196_196304

theorem find_a_if_x_is_1_root {a : ℝ} (h : (1 : ℝ)^2 + a * 1 - 2 = 0) : a = 1 :=
by sorry

end find_a_if_x_is_1_root_l196_196304


namespace remainder_consec_even_div12_l196_196387

theorem remainder_consec_even_div12 (n : ℕ) (h: n % 2 = 0)
  (h1: 11234 ≤ n ∧ n + 12 ≥ 11246) : 
  (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) + (n + 12)) % 12 = 6 :=
by 
  sorry

end remainder_consec_even_div12_l196_196387


namespace range_of_a_if_distinct_zeros_l196_196641

theorem range_of_a_if_distinct_zeros (a : ℝ) :
(∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ (x₁^3 - 3*x₁ + a = 0) ∧ (x₂^3 - 3*x₂ + a = 0) ∧ (x₃^3 - 3*x₃ + a = 0)) → -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_if_distinct_zeros_l196_196641


namespace ratio_platform_to_train_length_l196_196439

variable (L P t : ℝ)

-- Definitions based on conditions
def train_has_length (L : ℝ) : Prop := true
def train_constant_velocity : Prop := true
def train_passes_pole_in_t_seconds (L t : ℝ) : Prop := L / t = L
def train_passes_platform_in_4t_seconds (L P t : ℝ) : Prop := L / t = (L + P) / (4 * t)

-- Theorem statement: ratio of the length of the platform to the length of the train is 3:1
theorem ratio_platform_to_train_length (h1 : train_has_length L) 
                                      (h2 : train_constant_velocity) 
                                      (h3 : train_passes_pole_in_t_seconds L t)
                                      (h4 : train_passes_platform_in_4t_seconds L P t) :
  P / L = 3 := 
by sorry

end ratio_platform_to_train_length_l196_196439


namespace price_per_foot_of_fencing_l196_196563

theorem price_per_foot_of_fencing
  (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ)
  (h1 : area = 36) (h2 : total_cost = 1392) :
  price_per_foot = 58 :=
by
  sorry

end price_per_foot_of_fencing_l196_196563


namespace probability_no_adjacent_same_color_l196_196295

-- Define the problem space
def total_beads : ℕ := 9
def red_beads : ℕ := 4
def white_beads : ℕ := 3
def blue_beads : ℕ := 2

-- Define the total number of arrangements
def total_arrangements := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- State the probability computation theorem
theorem probability_no_adjacent_same_color :
  (∃ valid_arrangements : ℕ,
     valid_arrangements / total_arrangements = 1 / 63) := sorry

end probability_no_adjacent_same_color_l196_196295


namespace black_squares_in_45th_row_l196_196029

-- Definitions based on the conditions
def number_of_squares_in_row (n : ℕ) : ℕ := 2 * n + 1

def number_of_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

-- The theorem statement
theorem black_squares_in_45th_row : number_of_black_squares (number_of_squares_in_row 45) = 45 :=
by sorry

end black_squares_in_45th_row_l196_196029


namespace sum_inverse_terms_l196_196589

theorem sum_inverse_terms : 
  (∑' n : ℕ, if n = 0 then (0 : ℝ) else (1 / (n * (n + 3) : ℝ))) = 11 / 18 :=
by {
  -- proof to be filled in
  sorry
}

end sum_inverse_terms_l196_196589


namespace complementSetM_l196_196483

open Set Real

-- The universal set U is the set of all real numbers
def universalSet : Set ℝ := univ

-- The set M is defined as {x | |x - 1| ≤ 2}
def setM : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- We need to prove that the complement of M with respect to U is {x | x < -1 ∨ x > 3}
theorem complementSetM :
  (universalSet \ setM) = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end complementSetM_l196_196483


namespace total_water_needed_l196_196277

def adults : ℕ := 7
def children : ℕ := 3
def hours : ℕ := 24
def replenish_bottles : ℚ := 14
def water_per_hour_adult : ℚ := 1/2
def water_per_hour_child : ℚ := 1/3

theorem total_water_needed : 
  let total_water_per_hour := (adults * water_per_hour_adult) + (children * water_per_hour_child)
  let total_water := total_water_per_hour * hours 
  let initial_water_needed := total_water - replenish_bottles
  initial_water_needed = 94 := by 
  sorry

end total_water_needed_l196_196277


namespace no_integer_solutions_l196_196873

theorem no_integer_solutions :
  ∀ n m : ℤ, (n^2 + (n+1)^2 + (n+2)^2) ≠ m^2 :=
by
  intro n m
  sorry

end no_integer_solutions_l196_196873


namespace other_endpoint_l196_196233

theorem other_endpoint (M : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ) :
  M = (2, 3) ∧ A = (5, -1) ∧ (M = ((A.1 + x) / 2, (A.2 + y) / 2)) → (x, y) = (-1, 7) := by
  sorry

end other_endpoint_l196_196233


namespace f_f_minus_two_l196_196751

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (1 + x⁻¹))

theorem f_f_minus_two : f (f (-2)) = -8 / 3 := by
  sorry

end f_f_minus_two_l196_196751


namespace popularity_order_is_correct_l196_196642

noncomputable def fraction_liking_dodgeball := (13 : ℚ) / 40
noncomputable def fraction_liking_karaoke := (9 : ℚ) / 30
noncomputable def fraction_liking_magicshow := (17 : ℚ) / 60
noncomputable def fraction_liking_quizbowl := (23 : ℚ) / 120

theorem popularity_order_is_correct :
  (fraction_liking_dodgeball ≥ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≥ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≥ fraction_liking_quizbowl) ∧
  (fraction_liking_dodgeball ≠ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≠ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≠ fraction_liking_quizbowl) := by
  sorry

end popularity_order_is_correct_l196_196642


namespace circle_line_intersection_symmetric_l196_196495

theorem circle_line_intersection_symmetric (m n p x y : ℝ)
    (h_intersects : ∃ x y, x = m * y - 1 ∧ x^2 + y^2 + m * x + n * y + p = 0)
    (h_symmetric : ∀ A B : ℝ × ℝ, A = (x, y) ∧ B = (y, x) → y = x) :
    p < -3 / 2 :=
by
  sorry

end circle_line_intersection_symmetric_l196_196495


namespace gear_q_revolutions_per_minute_is_40_l196_196450

-- Definitions corresponding to conditions
def gear_p_revolutions_per_minute : ℕ := 10
def gear_q_revolutions_per_minute (r : ℕ) : Prop :=
  ∃ (r : ℕ), (r * 20 / 60) - (10 * 20 / 60) = 10

-- Statement we need to prove
theorem gear_q_revolutions_per_minute_is_40 :
  gear_q_revolutions_per_minute 40 :=
sorry

end gear_q_revolutions_per_minute_is_40_l196_196450


namespace weekly_cost_l196_196557

def cost_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7
def number_of_bodyguards : ℕ := 2

theorem weekly_cost :
  (cost_per_hour * hours_per_day * number_of_bodyguards * days_per_week) = 2240 := by
  sorry

end weekly_cost_l196_196557


namespace mixed_feed_total_pounds_l196_196825

theorem mixed_feed_total_pounds 
  (cheap_feed_cost : ℝ) (expensive_feed_cost : ℝ) (mix_cost : ℝ) 
  (cheap_feed_amount : ℕ) :
  cheap_feed_cost = 0.18 → 
  expensive_feed_cost = 0.53 → 
  mix_cost = 0.36 → 
  cheap_feed_amount = 17 → 
  (∃ (expensive_feed_amount : ℕ), 
    (cheap_feed_amount + expensive_feed_amount = 35)) :=
begin
  intros,
  use 18, -- We introduce 18 as the amount of more expensive feed
  sorry, -- Proof goes here
end

end mixed_feed_total_pounds_l196_196825


namespace incorrect_statement_g2_l196_196766

def g (x : ℚ) : ℚ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_g2 : g 2 ≠ 0 := by
  sorry

end incorrect_statement_g2_l196_196766


namespace card_probability_l196_196552

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l196_196552


namespace people_per_van_is_six_l196_196136

noncomputable def n_vans : ℝ := 6.0
noncomputable def n_buses : ℝ := 8.0
noncomputable def p_bus : ℝ := 18.0
noncomputable def people_difference : ℝ := 108

theorem people_per_van_is_six (x : ℝ) (h : n_buses * p_bus = n_vans * x + people_difference) : x = 6.0 := 
by
  sorry

end people_per_van_is_six_l196_196136


namespace range_of_m_l196_196620
-- Import the essential libraries

-- Define the problem conditions and state the theorem
theorem range_of_m (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_mono_dec : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x)
  (m : ℝ) (h_ineq : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l196_196620


namespace correct_value_division_l196_196183

theorem correct_value_division (x : ℕ) (h : 9 - x = 3) : 96 / x = 16 :=
by
  sorry

end correct_value_division_l196_196183


namespace longest_sequence_positive_integer_x_l196_196876

theorem longest_sequence_positive_integer_x :
  ∃ x : ℤ, 0 < x ∧ 34 * x - 10500 > 0 ∧ 17000 - 55 * x > 0 ∧ x = 309 :=
by
  use 309
  sorry

end longest_sequence_positive_integer_x_l196_196876


namespace sum_black_cells_even_l196_196537

-- Define a rectangular board with cells colored in a chess manner.

structure ChessBoard (m n : ℕ) :=
  (cells : Fin m → Fin n → Int)
  (row_sums_even : ∀ i : Fin m, (Finset.univ.sum (λ j => cells i j)) % 2 = 0)
  (column_sums_even : ∀ j : Fin n, (Finset.univ.sum (λ i => cells i j)) % 2 = 0)

def is_black_cell (i j : ℕ) : Bool :=
  (i + j) % 2 = 0

theorem sum_black_cells_even {m n : ℕ} (B : ChessBoard m n) :
    (Finset.univ.sum (λ (i : Fin m) =>
         Finset.univ.sum (λ (j : Fin n) =>
            if (is_black_cell i.val j.val) then B.cells i j else 0))) % 2 = 0 :=
by
  sorry

end sum_black_cells_even_l196_196537


namespace white_animals_count_l196_196793

-- Definitions
def total : ℕ := 13
def black : ℕ := 6
def white : ℕ := total - black

-- Theorem stating the number of white animals
theorem white_animals_count : white = 7 :=
by {
  -- The proof would go here, but we'll use sorry to skip it.
  sorry
}

end white_animals_count_l196_196793


namespace smallest_integer_relative_prime_to_2310_l196_196388

theorem smallest_integer_relative_prime_to_2310 (n : ℕ) : (2 < n → n ≤ 13 → ¬ (n ∣ 2310)) → n = 13 := by
  sorry

end smallest_integer_relative_prime_to_2310_l196_196388


namespace composite_numbers_characterization_l196_196166

noncomputable def is_sum_and_product_seq (n : ℕ) (seq : List ℕ) : Prop :=
  seq.sum = n ∧ seq.prod = n ∧ 2 ≤ seq.length ∧ ∀ x ∈ seq, 1 ≤ x

theorem composite_numbers_characterization (n : ℕ) :
  (∃ seq : List ℕ, is_sum_and_product_seq n seq) ↔ ¬Nat.Prime n ∧ 1 < n :=
sorry

end composite_numbers_characterization_l196_196166


namespace cameron_total_questions_l196_196447

theorem cameron_total_questions :
  let questions_per_tourist := 2
  let first_group := 6
  let second_group := 11
  let third_group := 8
  let third_group_special_tourist := 1
  let third_group_special_questions := 3 * questions_per_tourist
  let fourth_group := 7
  let first_group_total_questions := first_group * questions_per_tourist
  let second_group_total_questions := second_group * questions_per_tourist
  let third_group_total_questions := (third_group - third_group_special_tourist) * questions_per_tourist + third_group_special_questions
  let fourth_group_total_questions := fourth_group * questions_per_tourist
  in first_group_total_questions + second_group_total_questions + third_group_total_questions + fourth_group_total_questions = 68 := by
  sorry

end cameron_total_questions_l196_196447


namespace total_selling_amount_l196_196143

-- Defining the given conditions
def total_metres_of_cloth := 200
def loss_per_metre := 6
def cost_price_per_metre := 66

-- Theorem statement to prove the total selling amount
theorem total_selling_amount : 
    (cost_price_per_metre - loss_per_metre) * total_metres_of_cloth = 12000 := 
by 
    sorry

end total_selling_amount_l196_196143


namespace unbroken_seashells_left_l196_196219

-- Definitions based on given conditions
def total_seashells : ℕ := 6
def cone_shells : ℕ := 3
def conch_shells : ℕ := 3
def broken_cone_shells : ℕ := 2
def broken_conch_shells : ℕ := 2
def given_away_conch_shells : ℕ := 1

-- Mathematical statement to prove the final count of unbroken seashells
theorem unbroken_seashells_left : 
  (cone_shells - broken_cone_shells) + (conch_shells - broken_conch_shells - given_away_conch_shells) = 1 :=
by 
  -- Calculation (steps omitted per instructions)
  sorry

end unbroken_seashells_left_l196_196219


namespace ratio_larva_to_cocoon_l196_196078

theorem ratio_larva_to_cocoon (total_days : ℕ) (cocoon_days : ℕ)
  (h1 : total_days = 120) (h2 : cocoon_days = 30) :
  (total_days - cocoon_days) / cocoon_days = 3 := by
  sorry

end ratio_larva_to_cocoon_l196_196078


namespace twelfth_term_geometric_sequence_l196_196118

theorem twelfth_term_geometric_sequence :
  let a1 := 5
  let r := (2 / 5 : ℝ)
  (a1 * r ^ 11) = (10240 / 48828125 : ℝ) :=
by
  sorry

end twelfth_term_geometric_sequence_l196_196118


namespace amc_inequality_l196_196081

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l196_196081


namespace complement_union_A_B_in_U_l196_196345

open Set Nat

def U : Set ℕ := { x | x < 6 ∧ x > 0 }
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_union_A_B_in_U : (U \ (A ∪ B)) = {2, 4} := by
  sorry

end complement_union_A_B_in_U_l196_196345


namespace solve_cubic_equation_l196_196534

theorem solve_cubic_equation : 
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^3 - y^3 = 999 ∧ (x, y) = (12, 9) ∨ (x, y) = (10, 1) := 
  by
  sorry

end solve_cubic_equation_l196_196534


namespace intersection_M_N_l196_196906

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l196_196906


namespace inequality_four_a_cubed_sub_l196_196748

theorem inequality_four_a_cubed_sub (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  4 * a^3 * (a - b) ≥ a^4 - b^4 :=
sorry

end inequality_four_a_cubed_sub_l196_196748


namespace disjoint_subsets_same_sum_l196_196997

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l196_196997


namespace find_common_difference_l196_196773

variable (a an Sn d : ℚ)
variable (n : ℕ)

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a : ℚ) (an : ℚ) (n : ℕ) : ℚ :=
  n * (a + an) / 2

theorem find_common_difference
  (h1 : a = 3)
  (h2 : an = 50)
  (h3 : Sn = 318)
  (h4 : an = arithmetic_sequence a d n)
  (h5 : Sn = sum_arithmetic_sequence a an n) :
  d = 47 / 11 :=
by
  sorry

end find_common_difference_l196_196773


namespace probability_larger_than_two_thirds_l196_196801

noncomputable def prob_larger_than_two_thirds : ℝ :=
  let I : set ℝ := set.Icc 0 2
  let prob_interval (a b : ℝ) : ℝ := (b - a) / (2 - 0)
  let prob_less_than_two_thirds := prob_interval 0 (2 / 3)
  let prob_both_less_than_two_thirds := prob_less_than_two_thirds ^ 2
  1 - prob_both_less_than_two_thirds

theorem probability_larger_than_two_thirds :
  prob_larger_than_two_thirds = 8 / 9 :=
by sorry

end probability_larger_than_two_thirds_l196_196801


namespace M_inter_N_eq_neg2_l196_196896

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l196_196896


namespace problem_statement_l196_196805

-- Given: x, y, z are real numbers such that x < 0 and x < y < z
variables {x y z : ℝ} 

-- Conditions
axiom h1 : x < 0
axiom h2 : x < y
axiom h3 : y < z

-- Statement to prove: x + y < y + z
theorem problem_statement : x + y < y + z :=
by {
  sorry
}

end problem_statement_l196_196805


namespace sum_of_solutions_l196_196835

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l196_196835


namespace final_selling_price_correct_l196_196003

noncomputable def purchase_price_inr : ℝ := 8000
noncomputable def depreciation_rate_annual : ℝ := 0.10
noncomputable def profit_rate : ℝ := 0.10
noncomputable def discount_rate : ℝ := 0.05
noncomputable def sales_tax_rate : ℝ := 0.12
noncomputable def exchange_rate_at_purchase : ℝ := 80
noncomputable def exchange_rate_at_selling : ℝ := 75

noncomputable def depreciated_value_after_2_years (initial_value : ℝ) : ℝ :=
  initial_value * (1 - depreciation_rate_annual) * (1 - depreciation_rate_annual)

noncomputable def marked_price (initial_value : ℝ) : ℝ :=
  initial_value * (1 + profit_rate)

noncomputable def selling_price_before_tax (marked_price : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

noncomputable def final_selling_price_inr (selling_price_before_tax : ℝ) : ℝ :=
  selling_price_before_tax * (1 + sales_tax_rate)

noncomputable def final_selling_price_usd (final_selling_price_inr : ℝ) : ℝ :=
  final_selling_price_inr / exchange_rate_at_selling

theorem final_selling_price_correct :
  final_selling_price_usd (final_selling_price_inr (selling_price_before_tax (marked_price purchase_price_inr))) = 124.84 := 
sorry

end final_selling_price_correct_l196_196003


namespace roots_of_equation_l196_196987

theorem roots_of_equation :
  {x : ℝ | -x * (x + 3) = x * (x + 3)} = {0, -3} :=
by
  sorry

end roots_of_equation_l196_196987


namespace find_xy_l196_196875

variable (a : ℝ)

theorem find_xy (x y : ℝ) (k : ℤ) :
  x + y = a ∧ sin x ^ 2 + sin y ^ 2 = 1 - cos a ↔ 
  (x = a / 2 + k * Real.pi ∧ y = a / 2 - k * Real.pi) ∨
  (cos a = 0 ∧ x + y = (2 * k + 1) * Real.pi / 2) := 
sorry

end find_xy_l196_196875


namespace probability_odd_number_l196_196230

-- Defining the set of digits
def digits := {2, 3, 5, 7, 9}

-- Defining the condition that the number must be odd
def is_odd (n : Nat) : Prop := 
  ∃ x ∈ digits, n % 10 = x ∧ x % 2 = 1

-- Defining the total number of favorable outcomes for odd numbers
def favorable_outcomes : Nat := 4

-- Defining the total number of possible outcomes
def total_outcomes : Nat := 5

-- Statement of the theorem
theorem probability_odd_number : (favorable_outcomes : ℚ) / total_outcomes = 4 / 5 := 
by sorry

end probability_odd_number_l196_196230


namespace at_least_one_negative_l196_196796

theorem at_least_one_negative (a : Fin 7 → ℤ) :
  (∀ i j : Fin 7, i ≠ j → a i ≠ a j) ∧
  (∀ l1 l2 l3 : Fin 7, 
    a l1 + a l2 + a l3 = a l1 + a l2 + a l3) ∧
  (∃ i : Fin 7, a i = 0) →
  (∃ i : Fin 7, a i < 0) :=
  by
  sorry

end at_least_one_negative_l196_196796


namespace largest_of_three_consecutive_integers_l196_196371

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l196_196371


namespace min_value_a2_plus_b2_l196_196303

theorem min_value_a2_plus_b2 (a b : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 2 * b = 0 -> x = -2) : (∃ a b, a = 1 ∧ b = -1 ∧ ∀ a' b', a^2 + b^2 ≥ a'^2 + b'^2) := 
by {
  sorry
}

end min_value_a2_plus_b2_l196_196303


namespace volume_of_dug_out_earth_l196_196847

theorem volume_of_dug_out_earth
  (diameter depth : ℝ)
  (h_diameter : diameter = 2) 
  (h_depth : depth = 14) 
  : abs ((π * (1 / 2 * diameter / 2) ^ 2 * depth) - 44) < 0.1 :=
by
  -- Provide a placeholder for the proof
  sorry

end volume_of_dug_out_earth_l196_196847


namespace width_of_paving_stone_l196_196992

-- Given conditions as definitions
def length_of_courtyard : ℝ := 40
def width_of_courtyard : ℝ := 16.5
def number_of_stones : ℕ := 132
def length_of_stone : ℝ := 2.5

-- Define the total area of the courtyard
def area_of_courtyard := length_of_courtyard * width_of_courtyard

-- Define the equation we need to prove
theorem width_of_paving_stone :
  (length_of_stone * W * number_of_stones = area_of_courtyard) → W = 2 :=
by
  sorry

end width_of_paving_stone_l196_196992


namespace calculate_expression_l196_196960

def f (x : ℝ) := x^2 + 3
def g (x : ℝ) := 2 * x + 4

theorem calculate_expression : f (g 2) - g (f 2) = 49 := by
  sorry

end calculate_expression_l196_196960


namespace problem_ns_k_divisibility_l196_196728

theorem problem_ns_k_divisibility (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
  (∃ (a b : ℕ), (a = 1 ∨ a = 5) ∧ (b = 1 ∨ b = 5) ∧ a = n ∧ b = k) ↔ 
  n * k ∣ (2^(2^n) + 1) * (2^(2^k) + 1) := 
sorry

end problem_ns_k_divisibility_l196_196728


namespace arithmetic_seq_value_zero_l196_196948

theorem arithmetic_seq_value_zero (a b c : ℝ) (a_seq : ℕ → ℝ)
    (l m n : ℕ) (h_arith : ∀ k, a_seq (k + 1) - a_seq k = a_seq 1 - a_seq 0)
    (h_l : a_seq l = 1 / a)
    (h_m : a_seq m = 1 / b)
    (h_n : a_seq n = 1 / c) :
    (l - m) * a * b + (m - n) * b * c + (n - l) * c * a = 0 := 
sorry

end arithmetic_seq_value_zero_l196_196948


namespace pie_contest_l196_196556

def first_student_pie := 7 / 6
def second_student_pie := 4 / 3
def third_student_eats_from_first := 1 / 2
def third_student_eats_from_second := 1 / 3

theorem pie_contest :
  (first_student_pie - third_student_eats_from_first = 2 / 3) ∧
  (second_student_pie - third_student_eats_from_second = 1) ∧
  (third_student_eats_from_first + third_student_eats_from_second = 5 / 6) :=
by
  sorry

end pie_contest_l196_196556


namespace sum_reciprocal_eq_eleven_eighteen_l196_196587

noncomputable def sum_reciprocal (n : ℕ) : ℝ := ∑' (n : ℕ), 1 / (n * (n + 3))

theorem sum_reciprocal_eq_eleven_eighteen :
  sum_reciprocal = 11 / 18 :=
by
  sorry

end sum_reciprocal_eq_eleven_eighteen_l196_196587


namespace largest_of_three_consecutive_integers_l196_196372

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l196_196372


namespace problem_statement_l196_196312

theorem problem_statement (m : ℤ) (h : (m + 2)^2 = 64) : (m + 1) * (m + 3) = 63 :=
sorry

end problem_statement_l196_196312


namespace decrease_travel_time_l196_196779

variable (distance : ℕ) (initial_speed : ℕ) (speed_increase : ℕ)

def original_travel_time (distance initial_speed : ℕ) : ℕ :=
  distance / initial_speed

def new_travel_time (distance new_speed : ℕ) : ℕ :=
  distance / new_speed

theorem decrease_travel_time (h₁ : distance = 600) (h₂ : initial_speed = 50) (h₃ : speed_increase = 25) :
  original_travel_time distance initial_speed - new_travel_time distance (initial_speed + speed_increase) = 4 :=
by
  sorry

end decrease_travel_time_l196_196779


namespace initial_volume_proof_l196_196317

-- Definitions for initial mixture and ratios
variables (x : ℕ)

def initial_milk := 4 * x
def initial_water := x
def initial_volume := initial_milk x + initial_water x

def add_water (water_added : ℕ) := initial_water x + water_added

def resulting_ratio := initial_milk x / add_water x 9 = 2

theorem initial_volume_proof (h : resulting_ratio x) : initial_volume x = 45 :=
by sorry

end initial_volume_proof_l196_196317


namespace initial_number_of_people_l196_196325

theorem initial_number_of_people (P : ℕ) : P * 10 = (P + 1) * 5 → P = 1 :=
by sorry

end initial_number_of_people_l196_196325


namespace largest_integer_a_l196_196287

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l196_196287


namespace greatest_x_value_l196_196383

theorem greatest_x_value : 
  ∃ x : ℝ, (∀ y : ℝ, (y = (4 * x - 16) / (3 * x - 4)) → (y^2 + y = 12)) ∧ (x = 2) := by
  sorry

end greatest_x_value_l196_196383


namespace cubes_with_even_red_faces_l196_196583

theorem cubes_with_even_red_faces :
  let block_dimensions := (5, 5, 1)
  let painted_sides := 6
  let total_cubes := 25
  let cubes_with_2_red_faces := 16
  cubes_with_2_red_faces = 16 := by
  sorry

end cubes_with_even_red_faces_l196_196583


namespace Kyle_papers_delivered_each_week_proof_l196_196340

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l196_196340


namespace range_of_a_l196_196306

noncomputable def f (a x : ℝ) := Real.logb (1 / 2) (x^2 - a * x - a)

theorem range_of_a :
  (∀ a : ℝ, (∀ x : ℝ, f a x ∈ Set.univ) ∧ 
            (∀ x1 x2 : ℝ, -3 < x1 ∧ x1 < x2 ∧ x2 < 1 - Real.sqrt 3 → f a x1 < f a x2)) → 
  (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l196_196306


namespace andrew_age_l196_196871

theorem andrew_age (a g : ℝ) (h1 : g = 9 * a) (h2 : g - a = 63) : a = 7.875 :=
by
  sorry

end andrew_age_l196_196871


namespace inequality_solution_l196_196744

theorem inequality_solution (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3)
:=
sorry

end inequality_solution_l196_196744


namespace investment_ratio_l196_196112

theorem investment_ratio 
  (P Q : ℝ) 
  (profitP profitQ : ℝ)
  (h1 : profitP = 7 * (profitP + profitQ) / 17) 
  (h2 : profitQ = 10 * (profitP + profitQ) / 17)
  (tP : ℝ := 10)
  (tQ : ℝ := 20) 
  (h3 : profitP / profitQ = (P * tP) / (Q * tQ)) :
  P / Q = 7 / 5 := 
sorry

end investment_ratio_l196_196112


namespace intersection_M_N_l196_196901

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l196_196901


namespace overlap_region_area_l196_196701

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def overlap_area : ℝ := 
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  let P1 : ℝ × ℝ := (2, 2);
  let P2 : ℝ × ℝ := (4, 2);
  let P3 : ℝ × ℝ := (3, 3);
  let P4 : ℝ × ℝ := (2, 3);
  1/2 * abs (P1.1 * (P2.2 - P4.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P4.2 - P2.2) + P4.1 * (P1.2 - P3.2))

theorem overlap_region_area :
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  triangle_area A B C > 0 →
  triangle_area D E F > 0 →
  overlap_area = 0.5 :=
by { sorry }

end overlap_region_area_l196_196701


namespace convert_1623_to_base7_l196_196455

theorem convert_1623_to_base7 :
  ∃ a b c d : ℕ, 1623 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
  a = 4 ∧ b = 5 ∧ c = 0 ∧ d = 6 :=
by
  sorry

end convert_1623_to_base7_l196_196455


namespace team_A_wins_series_4_1_probability_l196_196241

noncomputable def probability_team_A_wins_series_4_1 : ℝ :=
  let home_win_prob : ℝ := 0.6
  let away_win_prob : ℝ := 0.5
  let home_loss_prob : ℝ := 1 - home_win_prob
  let away_loss_prob : ℝ := 1 - away_win_prob
  -- Scenario 1: L W W W W
  let p1 := home_loss_prob * home_win_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 2: W L W W W
  let p2 := home_win_prob * home_loss_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 3: W W L W W
  let p3 := home_win_prob * home_win_prob * away_loss_prob * away_win_prob * home_win_prob
  -- Scenario 4: W W W L W
  let p4 := home_win_prob * home_win_prob * away_win_prob * away_loss_prob * home_win_prob
  p1 + p2 + p3 + p4

theorem team_A_wins_series_4_1_probability : 
  probability_team_A_wins_series_4_1 = 0.18 :=
by
  -- This where the proof would go
  sorry

end team_A_wins_series_4_1_probability_l196_196241


namespace cost_per_quart_l196_196314

theorem cost_per_quart (paint_cost : ℝ) (coverage : ℝ) (cost_to_paint_cube : ℝ) (cube_edge : ℝ) 
    (h_coverage : coverage = 1200) (h_cost_to_paint_cube : cost_to_paint_cube = 1.60) 
    (h_cube_edge : cube_edge = 10) : paint_cost = 3.20 := by 
  sorry

end cost_per_quart_l196_196314


namespace polar_to_rectangular_l196_196200

noncomputable def curve_equation (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x = curve_equation θ * Real.cos θ ∧ y = curve_equation θ * Real.sin θ) :=
sorry

end polar_to_rectangular_l196_196200


namespace square_of_1024_l196_196719

theorem square_of_1024 : (1024 : ℤ)^2 = 1048576 := by
  let a := 1020
  let b := 4
  have h : (1024 : ℤ) = a + b := by
    norm_num
  rw [h] 
  norm_num
  sorry
  -- expand (a+b)^2 = a^2 + 2ab + b^2
  -- prove that 1020^2 = 1040400
  -- prove that 2 * 1020 * 4 = 8160
  -- prove that 4^2 = 16
  -- sum these results 
  -- result = 1048576

end square_of_1024_l196_196719


namespace remainder_of_division_l196_196607

noncomputable def P (x : ℝ) := x ^ 888
noncomputable def Q (x : ℝ) := (x ^ 2 - x + 1) * (x + 1)

theorem remainder_of_division :
  ∀ x : ℝ, (P x) % (Q x) = 1 :=
sorry

end remainder_of_division_l196_196607


namespace percentage_of_salt_in_second_solution_l196_196223

-- Define the data and initial conditions
def original_solution_salt_percentage := 0.15
def replaced_solution_salt_percentage (x: ℝ) := x
def resulting_solution_salt_percentage := 0.16

-- State the question as a theorem
theorem percentage_of_salt_in_second_solution (S : ℝ) (x : ℝ) :
  0.15 * S - 0.0375 * S + x * (S / 4) = 0.16 * S → x = 0.19 :=
by 
  sorry

end percentage_of_salt_in_second_solution_l196_196223


namespace solution_set_of_f_gt_2x_add_4_l196_196980

theorem solution_set_of_f_gt_2x_add_4 {f : ℝ → ℝ} (h_domain : ∀ x, true) (h_f_neq : f (-1) = 2)
  (h_deriv : ∀ x, deriv f x > 2) : 
  {x : ℝ | f x > 2 * x + 4} = Ioi (-1) := 
    by 
    sorry

end solution_set_of_f_gt_2x_add_4_l196_196980


namespace maximum_candy_leftover_l196_196629

theorem maximum_candy_leftover (x : ℕ) 
  (h1 : ∀ (bags : ℕ), bags = 12 → x ≥ bags * 10)
  (h2 : ∃ (leftover : ℕ), leftover < 12 ∧ leftover = (x - 120) % 12) : 
  ∃ (leftover : ℕ), leftover = 11 :=
by
  sorry

end maximum_candy_leftover_l196_196629


namespace smallest_divisible_by_15_11_12_l196_196041

theorem smallest_divisible_by_15_11_12 : ∃ n : ℕ, (n > 0) ∧ (15 ∣ n) ∧ (11 ∣ n) ∧ (12 ∣ n) ∧ (∀ m : ℕ, (m > 0) ∧ (15 ∣ m) ∧ (11 ∣ m) ∧ (12 ∣ m) → n ≤ m) ∧ n = 660 :=
by
  sorry

end smallest_divisible_by_15_11_12_l196_196041


namespace f_is_periodic_l196_196127

noncomputable def f (x : ℝ) : ℝ := x - ⌊x⌋

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x := by
  intro x
  sorry

end f_is_periodic_l196_196127


namespace discount_percentage_is_ten_l196_196703

-- Definitions based on given conditions
def cost_price : ℝ := 42
def markup (S : ℝ) : ℝ := 0.30 * S
def selling_price (S : ℝ) : Prop := S = cost_price + markup S
def profit : ℝ := 6

-- To prove the discount percentage
theorem discount_percentage_is_ten (S SP : ℝ) 
  (h_sell_price : selling_price S) 
  (h_SP : SP = S - profit) : 
  ((S - SP) / S) * 100 = 10 := 
by
  sorry

end discount_percentage_is_ten_l196_196703


namespace lost_card_l196_196402

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l196_196402


namespace inverse_variation_y_squared_sqrt_z_l196_196808

theorem inverse_variation_y_squared_sqrt_z (k : ℝ) :
  (∀ y z : ℝ, y^2 * sqrt z = k) →
  (∃ y z : ℝ, y = 3 ∧ z = 4 ∧ y^2 * sqrt z = k) →
  (∃ z : ℝ, (6 : ℝ)^2 * sqrt z = k ∧ z = 1/4) :=
by
  intros h₁ h₂
  sorry

end inverse_variation_y_squared_sqrt_z_l196_196808


namespace x_plus_y_eq_20_l196_196213

theorem x_plus_y_eq_20 (x y : ℝ) (hxy : x ≠ y) (hdet : (Matrix.det ![
  ![2, 3, 7],
  ![4, x, y],
  ![4, y, x]]) = 0) : x + y = 20 :=
by
  sorry

end x_plus_y_eq_20_l196_196213


namespace minimum_value_f_l196_196305

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end minimum_value_f_l196_196305


namespace probability_blue_or_green_face_l196_196429

def cube_faces: ℕ := 6
def blue_faces: ℕ := 3
def red_faces: ℕ := 2
def green_faces: ℕ := 1

theorem probability_blue_or_green_face (h1: blue_faces + red_faces + green_faces = cube_faces):
  (3 + 1) / 6 = 2 / 3 :=
by
  sorry

end probability_blue_or_green_face_l196_196429


namespace determine_n_l196_196271

theorem determine_n : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 % 8 = n := by
  use 2
  sorry

end determine_n_l196_196271


namespace boards_tested_l196_196661

-- Define the initial conditions and problem
def total_thumbtacks : ℕ := 450
def thumbtacks_remaining_each_can : ℕ := 30
def initial_thumbtacks_each_can := total_thumbtacks / 3
def thumbtacks_used_each_can := initial_thumbtacks_each_can - thumbtacks_remaining_each_can
def total_thumbtacks_used := thumbtacks_used_each_can * 3
def thumbtacks_per_board := 3

-- Define the proposition to prove 
theorem boards_tested (B : ℕ) : 
  (B = total_thumbtacks_used / thumbtacks_per_board) → B = 120 :=
by
  -- Proof skipped with sorry
  sorry

end boards_tested_l196_196661


namespace erasers_per_box_l196_196955

theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (erasers_per_box : ℕ) : total_erasers = 40 → num_boxes = 4 → erasers_per_box = total_erasers / num_boxes → erasers_per_box = 10 :=
by
  intros h_total h_boxes h_div
  rw [h_total, h_boxes] at h_div
  norm_num at h_div
  exact h_div

end erasers_per_box_l196_196955


namespace expand_expression_l196_196732

theorem expand_expression (x : ℝ) : (x + 3) * (2 * x ^ 2 - x + 4) = 2 * x ^ 3 + 5 * x ^ 2 + x + 12 :=
by
  sorry

end expand_expression_l196_196732


namespace young_people_in_sample_l196_196700

-- Define the conditions
def total_population (elderly middle_aged young : ℕ) : ℕ :=
  elderly + middle_aged + young

def sample_proportion (sample_size total_pop : ℚ) : ℚ :=
  sample_size / total_pop

def stratified_sample (group_size proportion : ℚ) : ℚ :=
  group_size * proportion

-- Main statement to prove
theorem young_people_in_sample (elderly middle_aged young : ℕ) (sample_size : ℚ) :
  total_population elderly middle_aged young = 108 →
  sample_size = 36 →
  stratified_sample (young : ℚ) (sample_proportion sample_size 108) = 17 :=
by
  intros h_total h_sample_size
  sorry -- proof omitted

end young_people_in_sample_l196_196700


namespace first_installment_amount_l196_196324

-- Define the conditions stated in the problem
def original_price : ℝ := 480
def discount_rate : ℝ := 0.05
def monthly_installment : ℝ := 102
def number_of_installments : ℕ := 3

-- The final price after discount
def final_price : ℝ := original_price * (1 - discount_rate)

-- The total amount of the 3 monthly installments
def total_of_3_installments : ℝ := monthly_installment * number_of_installments

-- The first installment paid
def first_installment : ℝ := final_price - total_of_3_installments

-- The main theorem to prove the first installment amount
theorem first_installment_amount : first_installment = 150 := by
  unfold first_installment
  unfold final_price
  unfold total_of_3_installments
  unfold original_price
  unfold discount_rate
  unfold monthly_installment
  unfold number_of_installments
  sorry

end first_installment_amount_l196_196324


namespace find_S30_l196_196502

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l196_196502


namespace proposition_3_true_proposition_4_true_l196_196311

def exp_pos (x : ℝ) : Prop := Real.exp x > 0

def two_power_gt_xsq (x : ℝ) : Prop := 2^x > x^2

def prod_gt_one (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop := a * b > 1

def geom_seq_nec_suff (a b c : ℝ) : Prop := ¬(b = Real.sqrt (a * c) ∨ (a * b = c * b ∧ b^2 = a * c))

theorem proposition_3_true (a b : ℝ) (ha : a > 1) (hb : b > 1) : prod_gt_one a b ha hb :=
sorry

theorem proposition_4_true (a b c : ℝ) : geom_seq_nec_suff a b c :=
sorry

end proposition_3_true_proposition_4_true_l196_196311


namespace geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l196_196176

variable (a b c : ℝ)

theorem geometric_implies_b_squared_eq_ac
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∃ r : ℝ, b = r * a ∧ c = r * b) :
  b^2 = a * c :=
by
  sorry

theorem not_geometric_if_all_zero 
  (hz : a = 0 ∧ b = 0 ∧ c = 0) : 
  ¬(∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

theorem sufficient_but_not_necessary_condition :
  (∃ r : ℝ, b = r * a ∧ c = r * b → b^2 = a * c) ∧ ¬(b^2 = a * c → ∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

end geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l196_196176


namespace problem_A_inter_B_empty_l196_196932

section

def set_A : Set ℝ := {x | |x| ≥ 2}
def set_B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_A_inter_B_empty : set_A ∩ set_B = ∅ := 
  sorry

end

end problem_A_inter_B_empty_l196_196932


namespace divisors_condition_l196_196601

theorem divisors_condition (n : ℕ) (hn : 1 < n) : 
  (∀ k l : ℕ, k ∣ n → l ∣ n → k < n → l < n → ((2 * k - l) ∣ n ∨ (2 * l - k) ∣ n)) →
  (nat.prime n ∨ n = 6 ∨ n = 9 ∨ n = 15) :=
by
  sorry

end divisors_condition_l196_196601


namespace solve_for_x_l196_196225

-- Let us state and prove that x = 495 / 13 is a solution to the equation 3x + 5 = 500 - (4x + 6x)
theorem solve_for_x (x : ℝ) : 3 * x + 5 = 500 - (4 * x + 6 * x) → x = 495 / 13 :=
by
  sorry

end solve_for_x_l196_196225


namespace train_length_l196_196145

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 90) 
  (h2 : time_sec = 11) 
  (h3 : length_m = 275) :
  length_m = (speed_km_hr * 1000 / 3600) * time_sec :=
sorry

end train_length_l196_196145


namespace range_of_a_l196_196248

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) → a < 1 :=
by
  sorry

end range_of_a_l196_196248


namespace cannot_achieve_1_5_percent_salt_solution_l196_196239

-- Define the initial concentrations and volumes
def initial_state (V1 V2 : ℝ) (C1 C2 : ℝ) : Prop :=
  V1 = 1 ∧ C1 = 0 ∧ V2 = 1 ∧ C2 = 0.02

-- Define the transfer and mixing operation
noncomputable def transfer_and_mix (V1_old V2_old C1_old C2_old : ℝ) (amount_to_transfer : ℝ)
  (new_V1 new_V2 new_C1 new_C2 : ℝ) : Prop :=
  amount_to_transfer ≤ V2_old ∧
  new_V1 = V1_old + amount_to_transfer ∧
  new_V2 = V2_old - amount_to_transfer ∧
  new_C1 = (V1_old * C1_old + amount_to_transfer * C2_old) / new_V1 ∧
  new_C2 = (V2_old * C2_old - amount_to_transfer * C2_old) / new_V2

-- Prove that it is impossible to achieve a 1.5% salt concentration in container 1
theorem cannot_achieve_1_5_percent_salt_solution :
  ∀ V1 V2 C1 C2, initial_state V1 V2 C1 C2 →
  ¬ ∃ V1' V2' C1' C2', transfer_and_mix V1 V2 C1 C2 0.5 V1' V2' C1' C2' ∧ C1' = 0.015 :=
by
  intros
  sorry

end cannot_achieve_1_5_percent_salt_solution_l196_196239


namespace solution_set_of_inequality_l196_196472

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Problem conditions
theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x, x < 0 → f x = x + 2) :
  { x : ℝ | 2 * f x - 1 < 0 } = { x : ℝ | x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) } :=
by
  sorry

end solution_set_of_inequality_l196_196472


namespace points_for_win_l196_196316

variable (W T : ℕ)

theorem points_for_win (W T : ℕ) (h1 : W * (T + 12) + T = 60) : W = 2 :=
by {
  sorry
}

end points_for_win_l196_196316


namespace total_rounds_played_l196_196124

/-- William and Harry played some rounds of tic-tac-toe.
    William won 5 more rounds than Harry.
    William won 10 rounds.
    Prove that the total number of rounds they played is 15. -/
theorem total_rounds_played (williams_wins : ℕ) (harrys_wins : ℕ)
  (h1 : williams_wins = 10)
  (h2 : williams_wins = harrys_wins + 5) :
  williams_wins + harrys_wins = 15 := 
by
  sorry

end total_rounds_played_l196_196124


namespace range_of_2a_sub_b_l196_196296

theorem range_of_2a_sub_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) : -4 < 2 * a - b ∧ 2 * a - b < 2 :=
by
  sorry

end range_of_2a_sub_b_l196_196296


namespace kyle_paper_delivery_l196_196337

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l196_196337


namespace area_of_square_l196_196110

theorem area_of_square (A_circle : ℝ) (hA_circle : A_circle = 39424) (cm_to_inch : ℝ) (hcm_to_inch : cm_to_inch = 2.54) :
  ∃ (A_square : ℝ), A_square = 121.44 := 
by
  sorry

end area_of_square_l196_196110


namespace total_opponents_points_is_36_l196_196128
-- Import the Mathlib library

-- Define the conditions as Lean definitions
def game_scores : List ℕ := [3, 5, 6, 7, 8, 9, 11, 12]

def lost_by_two (n : ℕ) : Prop := n + 2 ∈ game_scores

def three_times_as_many (n : ℕ) : Prop := n * 3 ∈ game_scores

-- State the problem
theorem total_opponents_points_is_36 : 
  (∃ l1 l2 l3 w1 w2 w3 w4 w5 : ℕ, 
    game_scores = [l1, l2, l3, w1, w2, w3, w4, w5] ∧
    lost_by_two l1 ∧ lost_by_two l2 ∧ lost_by_two l3 ∧
    three_times_as_many w1 ∧ three_times_as_many w2 ∧ 
    three_times_as_many w3 ∧ three_times_as_many w4 ∧ 
    three_times_as_many w5 ∧ 
    l1 + 2 + l2 + 2 + l3 + 2 + ((w1 / 3) + (w2 / 3) + (w3 / 3) + (w4 / 3) + (w5 / 3)) = 36) :=
sorry

end total_opponents_points_is_36_l196_196128


namespace determine_function_f_l196_196722

noncomputable def f (c x : ℝ) : ℝ := c ^ (1 / Real.log x)

theorem determine_function_f (f : ℝ → ℝ) (c : ℝ) (Hc : c > 1) :
  (∀ x, 1 < x → 1 < f x) →
  (∀ (x y : ℝ) (u v : ℝ), 1 < x → 1 < y → 0 < u → 0 < v →
    f (x ^ 4 * y ^ v) ≤ (f x) ^ (1 / (4 * u)) * (f y) ^ (1 / (4 * v))) →
  (∀ x : ℝ, 1 < x → f x = c ^ (1 / Real.log x)) :=
by
  sorry

end determine_function_f_l196_196722


namespace min_dominos_in_2x2_l196_196854

/-- A 100 × 100 square is divided into 2 × 2 squares.
Then it is divided into dominos (rectangles 1 × 2 and 2 × 1).
Prove that the minimum number of dominos within the 2 × 2 squares is 100. -/
theorem min_dominos_in_2x2 (N : ℕ) (hN : N = 100) :
  ∃ d : ℕ, d = 100 :=
sorry

end min_dominos_in_2x2_l196_196854


namespace binomial_expansion_terms_l196_196756

theorem binomial_expansion_terms (n : ℕ) (x : ℝ) (h : Nat.choose n (n-2) = 45) :
  (∃ c : ℝ, (sqrt (sqrt x) + sqrt (x ^ 3)) ^ n = c * x^5 ∧ c = 45) ∧
  (∃ c' : ℝ, (sqrt (sqrt x) + sqrt (x ^ 3)) ^ n = c' * x^(35 / 4) ∧ c' = 252) := by
  sorry

end binomial_expansion_terms_l196_196756


namespace air_quality_conditional_prob_l196_196441

theorem air_quality_conditional_prob :
  let p1 := 0.8
  let p2 := 0.68
  let p := p2 / p1
  p = 0.85 :=
by
  sorry

end air_quality_conditional_prob_l196_196441


namespace duty_pairing_impossible_l196_196471

theorem duty_pairing_impossible :
  ∀ (m n : ℕ), 29 * m + 32 * n ≠ 29 * 32 := 
by 
  sorry

end duty_pairing_impossible_l196_196471


namespace diagonal_length_l196_196228

noncomputable def convertHectaresToSquareMeters (hectares : ℝ) : ℝ :=
  hectares * 10000

noncomputable def sideLength (areaSqMeters : ℝ) : ℝ :=
  Real.sqrt areaSqMeters

noncomputable def diagonal (side : ℝ) : ℝ :=
  side * Real.sqrt 2

theorem diagonal_length (area : ℝ) (h : area = 1 / 2) :
  let areaSqMeters := convertHectaresToSquareMeters area
  let side := sideLength areaSqMeters
  let diag := diagonal side
  abs (diag - 100) < 1 :=
by
  sorry

end diagonal_length_l196_196228


namespace a_2018_value_l196_196754

theorem a_2018_value (S a : ℕ -> ℕ) (h₁ : S 1 = a 1) (h₂ : a 1 = 1) (h₃ : ∀ n : ℕ, n > 0 -> S (n + 1) = 3 * S n) :
  a 2018 = 2 * 3 ^ 2016 :=
sorry

end a_2018_value_l196_196754


namespace system_solutions_range_b_l196_196310

theorem system_solutions_range_b (b : ℝ) :
  (∀ x y : ℝ, x^2 - y^2 = 0 → x^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0 ∨ y = b) →
  b ≥ 2 ∨ b ≤ -2 :=
sorry

end system_solutions_range_b_l196_196310


namespace solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l196_196972

-- For the first polynomial equation
theorem solve_cubic_eq_a (x : ℝ) : x^3 - 3 * x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

-- For the second polynomial equation
theorem solve_cubic_eq_b (x : ℝ) : x^3 - 19 * x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
by sorry

-- For the third polynomial equation
theorem solve_cubic_eq_c (x : ℝ) : x^3 + 4 * x^2 + 6 * x + 4 = 0 ↔ x = -2 :=
by sorry

end solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l196_196972


namespace gcd_546_210_l196_196480

theorem gcd_546_210 : Nat.gcd 546 210 = 42 := by
  sorry -- Proof is required to solve

end gcd_546_210_l196_196480


namespace find_x1_l196_196656

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 :=
  sorry

end find_x1_l196_196656


namespace max_three_m_plus_four_n_l196_196114

theorem max_three_m_plus_four_n (m n : ℕ) 
  (h : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n ≤ 221 :=
sorry

end max_three_m_plus_four_n_l196_196114


namespace minimum_perimeter_triangle_l196_196193

noncomputable def minimum_perimeter (a b c : ℝ) (cos_C : ℝ) (ha : a + b = 10) (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0) 
  : ℝ :=
  a + b + c

theorem minimum_perimeter_triangle (a b c : ℝ) (cos_C : ℝ)
  (ha : a + b = 10)
  (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0)
  (cos_C_valid : cos_C = -1/2) :
  (minimum_perimeter a b c cos_C ha hroot) = 10 + 5 * Real.sqrt 3 :=
sorry

end minimum_perimeter_triangle_l196_196193


namespace arcsin_one_half_eq_pi_six_l196_196268

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = π / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l196_196268


namespace quadratic_zeros_interval_l196_196190

theorem quadratic_zeros_interval (a : ℝ) :
  (5 - 2 * a > 0) ∧ (4 * a^2 - 16 > 0) ∧ (a > 1) ↔ (2 < a ∧ a < 5 / 2) :=
by
  sorry

end quadratic_zeros_interval_l196_196190


namespace kyle_paper_delivery_l196_196336

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l196_196336


namespace residue_calculation_l196_196291

theorem residue_calculation 
  (h1 : 182 ≡ 0 [MOD 14])
  (h2 : 182 * 12 ≡ 0 [MOD 14])
  (h3 : 15 * 7 ≡ 7 [MOD 14])
  (h4 : 3 ≡ 3 [MOD 14]) :
  (182 * 12 - 15 * 7 + 3) ≡ 10 [MOD 14] :=
sorry

end residue_calculation_l196_196291


namespace train_length_l196_196013

theorem train_length (v_train_kmph : ℝ) (v_man_kmph : ℝ) (time_sec : ℝ) 
  (h1 : v_train_kmph = 25) 
  (h2 : v_man_kmph = 2) 
  (h3 : time_sec = 20) : 
  (150 : ℝ) = (v_train_kmph + v_man_kmph) * (1000 / 3600) * time_sec := 
by {
  -- sorry for the steps here
  sorry
}

end train_length_l196_196013


namespace smallest_next_divisor_of_m_l196_196330

theorem smallest_next_divisor_of_m (m : ℕ) (h1 : m % 2 = 0) (h2 : 10000 ≤ m ∧ m < 100000) (h3 : 523 ∣ m) : 
  ∃ d : ℕ, 523 < d ∧ d ∣ m ∧ ∀ e : ℕ, 523 < e ∧ e ∣ m → d ≤ e :=
by
  sorry

end smallest_next_divisor_of_m_l196_196330


namespace compute_ratio_l196_196753

variable {p q r u v w : ℝ}

theorem compute_ratio
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (h1 : p^2 + q^2 + r^2 = 49) 
  (h2 : u^2 + v^2 + w^2 = 64) 
  (h3 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 := 
sorry

end compute_ratio_l196_196753


namespace books_read_l196_196521

-- Given conditions
def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

-- Statement to prove
theorem books_read : (total_chapters_read / chapters_per_book) = 4 := 
by sorry

end books_read_l196_196521


namespace range_of_m_l196_196930

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + (m + 2) * x + (m + 5) = 0 → 0 < x) → (-5 < m ∧ m ≤ -4) :=
by
  sorry

end range_of_m_l196_196930


namespace ratio_of_girls_to_boys_l196_196497

theorem ratio_of_girls_to_boys (g b : ℕ) (h1 : g = b + 6) (h2 : g + b = 36) : g / b = 7 / 5 := by sorry

end ratio_of_girls_to_boys_l196_196497


namespace find_b_value_l196_196284

theorem find_b_value :
  ∃ b : ℕ, 70 = (2 * (b + 1)^2 + 3 * (b + 1) + 4) - (2 * (b - 1)^2 + 3 * (b - 1) + 4) ∧ b > 0 ∧ b < 1000 :=
by
  sorry

end find_b_value_l196_196284


namespace sum_of_legs_of_larger_triangle_l196_196829

theorem sum_of_legs_of_larger_triangle (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ) :
    (area_small = 8 ∧ area_large = 200 ∧ hypotenuse_small = 6) →
    ∃ sum_of_legs : ℝ, sum_of_legs = 41.2 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l196_196829


namespace difference_in_height_l196_196874

-- Define the heights of the sandcastles
def h_J : ℚ := 3.6666666666666665
def h_S : ℚ := 2.3333333333333335

-- State the theorem
theorem difference_in_height :
  h_J - h_S = 1.333333333333333 := by
  sorry

end difference_in_height_l196_196874


namespace find_number_l196_196428

variable (number x : ℝ)

theorem find_number (h1 : number * x = 1600) (h2 : x = -8) : number = -200 := by
  sorry

end find_number_l196_196428


namespace dog_partitioning_l196_196227

open Combinatorics

/-- Given 12 dogs, we want to partition them into groups of sizes 4, 6, and 2, such that:
  Rover is in the 4-dog group, and Spot is in the 6-dog group. The number
  of ways to achieve this partition is 2520. -/
theorem dog_partitioning :
  (∃ dogs : Finset ℕ, dogs.card = 12) →
  (∃ Rover Spot : ℕ, Rover ≠ Spot ∧ Rover ∈ dogs ∧ Spot ∈ dogs) →
  (∃ group1 group2 group3 : Finset ℕ, 
    group1.card = 4 ∧ group2.card = 6 ∧ group3.card = 2 ∧
    Rover ∈ group1 ∧ Spot ∈ group2 ∧ 
    group1 ∪ group2 ∪ group3 = dogs ∧ 
    group1 ∩ group2 = ∅ ∧ group2 ∩ group3 = ∅ ∧ group1 ∩ group3 = ∅) →
  nat.choose 10 3 * nat.choose 7 5 = 2520 := 
by
  -- Proof omitted
  sorry

end dog_partitioning_l196_196227


namespace greatest_integer_a_exists_l196_196286

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l196_196286


namespace remainder_3_pow_2000_mod_17_l196_196685

theorem remainder_3_pow_2000_mod_17 : (3^2000 % 17) = 1 := by
  sorry

end remainder_3_pow_2000_mod_17_l196_196685


namespace directrix_of_parabola_l196_196924

theorem directrix_of_parabola (a : ℝ) (P : ℝ × ℝ)
  (h1 : 3 * P.1 ^ 2 - P.2 ^ 2 = 3 * a ^ 2)
  (h2 : P.2 ^ 2 = 8 * a * P.1)
  (h3 : a > 0)
  (h4 : abs ((P.1 - 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) + abs ((P.1 + 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) = 12) :
  (a = 1) → P.1 = 6 - 3 * a → P.2 ^ 2 = 8 * a * (6 - 3 * a) → -2 * a = -2 := 
by
  sorry

end directrix_of_parabola_l196_196924


namespace y_in_terms_of_x_l196_196047

theorem y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := 
by
  sorry

end y_in_terms_of_x_l196_196047


namespace condition_M_intersect_N_N_l196_196048

theorem condition_M_intersect_N_N (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + (y - a)^2 ≤ 1 → y ≥ x^2)) ↔ (a ≥ 5 / 4) :=
sorry

end condition_M_intersect_N_N_l196_196048


namespace intersection_M_N_l196_196903

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l196_196903


namespace chickens_in_zoo_l196_196274

theorem chickens_in_zoo (c e : ℕ) (h_legs : 2 * c + 4 * e = 66) (h_heads : c + e = 24) : c = 15 :=
by
  sorry

end chickens_in_zoo_l196_196274


namespace average_visitors_30_day_month_l196_196848

def visitors_per_day (total_visitors : ℕ) (days : ℕ) : ℕ := total_visitors / days

theorem average_visitors_30_day_month (visitors_sunday : ℕ) (visitors_other_days : ℕ) 
  (total_days : ℕ) (sundays : ℕ) (other_days : ℕ) :
  visitors_sunday = 510 →
  visitors_other_days = 240 →
  total_days = 30 →
  sundays = 4 →
  other_days = 26 →
  visitors_per_day (sundays * visitors_sunday + other_days * visitors_other_days) total_days = 276 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end average_visitors_30_day_month_l196_196848


namespace total_value_is_76_percent_of_dollar_l196_196088

def coin_values : List Nat := [1, 5, 20, 50]

def total_value (coins : List Nat) : Nat :=
  List.sum coins

def percentage_of_dollar (value : Nat) : Nat :=
  value * 100 / 100

theorem total_value_is_76_percent_of_dollar :
  percentage_of_dollar (total_value coin_values) = 76 := by
  sorry

end total_value_is_76_percent_of_dollar_l196_196088


namespace prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l196_196855

noncomputable def prob_first_firing_A : ℚ := 4 / 5
noncomputable def prob_first_firing_B : ℚ := 3 / 4
noncomputable def prob_first_firing_C : ℚ := 2 / 3

noncomputable def prob_second_firing : ℚ := 3 / 5

noncomputable def prob_only_one_success_first_firing :=
  prob_first_firing_A * (1 - prob_first_firing_B) * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * prob_first_firing_B * (1 - prob_first_firing_C) +
  (1 - prob_first_firing_A) * (1 - prob_first_firing_B) * prob_first_firing_C

theorem prob_only_one_success_first_firing_is_correct :
  prob_only_one_success_first_firing = 3 / 20 :=
by sorry

noncomputable def prob_success_after_both_firings_A := prob_first_firing_A * prob_second_firing
noncomputable def prob_success_after_both_firings_B := prob_first_firing_B * prob_second_firing
noncomputable def prob_success_after_both_firings_C := prob_first_firing_C * prob_second_firing

noncomputable def prob_all_success_after_both_firings :=
  prob_success_after_both_firings_A * prob_success_after_both_firings_B * prob_success_after_both_firings_C

theorem prob_all_success_after_both_firings_is_correct :
  prob_all_success_after_both_firings = 54 / 625 :=
by sorry

end prob_only_one_success_first_firing_is_correct_prob_all_success_after_both_firings_is_correct_l196_196855


namespace f_n_2_l196_196726

def f (m n : ℕ) : ℝ :=
if h : m = 1 ∧ n = 1 then 1 else
if h : n > m then 0 else 
sorry -- This would be calculated based on the recursive definition

lemma f_2_2 : f 2 2 = 2 :=
sorry

theorem f_n_2 (n : ℕ) (hn : n ≥ 1) : f n 2 = 2^(n - 1) :=
sorry

end f_n_2_l196_196726


namespace geometric_sequence_min_value_l196_196510

theorem geometric_sequence_min_value
  (s : ℝ) (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * s)
  (h3 : b3 = 2 * s ^ 2) :
  ∃ (s : ℝ), 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end geometric_sequence_min_value_l196_196510


namespace solution_set_l196_196931

def f (x : ℝ) : ℝ := abs x - x + 1

theorem solution_set (x : ℝ) : f (1 - x^2) > f (1 - 2 * x) ↔ x > 2 ∨ x < -1 := by
  sorry

end solution_set_l196_196931


namespace expand_expression_l196_196036

theorem expand_expression (x y z : ℝ) :
  (2 * x + 15) * (3 * y + 20 * z + 25) = 
  6 * x * y + 40 * x * z + 50 * x + 45 * y + 300 * z + 375 :=
by
  sorry

end expand_expression_l196_196036


namespace solve_equation_l196_196353

theorem solve_equation (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^(2 * y - 1) + (x + 1)^(2 * y - 1) = (x + 2)^(2 * y - 1) ↔ (x = 1 ∧ y = 1) := by
  sorry

end solve_equation_l196_196353


namespace intersection_length_l196_196209

theorem intersection_length 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 1) 
  (hA_on_line : A.1 = A.2) 
  (hB_on_line : B.1 = B.2) 
  (hAB : A ≠ B) :
  dist A B = 2 :=
by sorry

end intersection_length_l196_196209


namespace students_in_grade6_l196_196280

noncomputable def num_students_total : ℕ := 100
noncomputable def num_students_grade4 : ℕ := 30
noncomputable def num_students_grade5 : ℕ := 35
noncomputable def num_students_grade6 : ℕ := num_students_total - (num_students_grade4 + num_students_grade5)

theorem students_in_grade6 : num_students_grade6 = 35 := by
  sorry

end students_in_grade6_l196_196280


namespace greatest_number_of_cool_cells_l196_196080

noncomputable def greatest_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) : ℕ :=
n^2 - 2 * n + 1

theorem greatest_number_of_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) (h : 0 < n) :
  ∃ m, m = (n - 1)^2 ∧ m = greatest_cool_cells n grid :=
sorry

end greatest_number_of_cool_cells_l196_196080


namespace find_a_l196_196054

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x >= 0 then a^x else a^(-x)

theorem find_a (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a)
(h_ge : ∀ x : ℝ, x >= 0 → f x a = a ^ x)
(h_a_gt_1 : a > 1)
(h_sol : ∀ x : ℝ, f x a ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) :
a = 2 :=
sorry

end find_a_l196_196054


namespace michael_has_more_flying_robots_l196_196663

theorem michael_has_more_flying_robots (tom_robots michael_robots : ℕ) (h_tom : tom_robots = 3) (h_michael : michael_robots = 12) :
  michael_robots / tom_robots = 4 :=
by
  sorry

end michael_has_more_flying_robots_l196_196663


namespace yellow_balls_count_l196_196990

theorem yellow_balls_count (R B G Y : ℕ) 
  (h1 : R = 2 * B) 
  (h2 : B = 2 * G) 
  (h3 : Y > 7) 
  (h4 : R + B + G + Y = 27) : 
  Y = 20 := by
  sorry

end yellow_balls_count_l196_196990


namespace no_solutions_in_natural_numbers_l196_196969

theorem no_solutions_in_natural_numbers (x y : ℕ) : x^2 + x * y + y^2 ≠ x^2 * y^2 :=
  sorry

end no_solutions_in_natural_numbers_l196_196969


namespace fifteen_percent_of_x_equals_sixty_l196_196884

theorem fifteen_percent_of_x_equals_sixty (x : ℝ) (h : 0.15 * x = 60) : x = 400 :=
by
  sorry

end fifteen_percent_of_x_equals_sixty_l196_196884


namespace find_x_l196_196687

theorem find_x (x : ℝ) (h : (40 / 80) = Real.sqrt (x / 80)) : x = 20 := 
by 
  sorry

end find_x_l196_196687


namespace sum_inverse_terms_l196_196588

theorem sum_inverse_terms : 
  (∑' n : ℕ, if n = 0 then (0 : ℝ) else (1 / (n * (n + 3) : ℝ))) = 11 / 18 :=
by {
  -- proof to be filled in
  sorry
}

end sum_inverse_terms_l196_196588


namespace conclusion_1_conclusion_3_l196_196458

def tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem conclusion_1 : tensor 2 (-2) = 6 :=
by sorry

theorem conclusion_3 (a b : ℝ) (h : a + b = 0) : tensor a a + tensor b b = 2 * a * b :=
by sorry

end conclusion_1_conclusion_3_l196_196458


namespace largest_sum_l196_196266

theorem largest_sum : max (1/4 + 1/9) (max (1/4 + 1/10) (max (1/4 + 1/2) (max (1/4 + 1/12) (1/4 + 1/11)))) = 3/4 :=
by {
  -- The proof goes here
  sorry
}

end largest_sum_l196_196266


namespace part_a_contradiction_l196_196852

theorem part_a_contradiction :
  ¬ (225 / 25 + 75 = 100 - 16 → 25 * (9 / (1 + 3)) = 84) :=
by
  sorry

end part_a_contradiction_l196_196852


namespace ant_at_C_after_4_minutes_l196_196712

open ProbabilityTheory

-- Definitions of points on the lattice
structure Point :=
  (x : Int)
  (y : Int)

-- Define movement on the lattice
def move (p : Point) (d : Point) : Point :=
  ⟨p.x + d.x, p.y + d.y⟩

-- Adjacent moves (up, down, left, right)
def adjacent_moves : Set Point :=
  {⟨0, 1⟩, ⟨0, -1⟩, ⟨1, 0⟩, ⟨-1, 0⟩}

-- Probabilistic function to determine reach
noncomputable def transition_prob (start : Point) (end : Point) (n : Nat) : ℝ := sorry

theorem ant_at_C_after_4_minutes (A C : Point) (H_A : A = ⟨0, 0⟩) (H_C : C = ⟨2, 0⟩) :
  transition_prob A C 4 = 1/3 := 
sorry

end ant_at_C_after_4_minutes_l196_196712


namespace find_number_l196_196073

theorem find_number (x : ℤ) (h : 5 * (x - 12) = 40) : x = 20 := 
by
  sorry

end find_number_l196_196073


namespace alice_travel_time_l196_196101

theorem alice_travel_time (distance_AB : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) (max_time_diff_hr : ℝ) (time_conversion : ℝ) :
  distance_AB = 60 →
  bob_speed = 40 →
  alice_speed = 60 →
  max_time_diff_hr = 0.5 →
  time_conversion = 60 →
  max_time_diff_hr * time_conversion = 30 :=
by
  intros
  sorry

end alice_travel_time_l196_196101


namespace perfect_squares_represented_as_diff_of_consecutive_cubes_l196_196761

theorem perfect_squares_represented_as_diff_of_consecutive_cubes : ∃ (count : ℕ), 
  count = 40 ∧ 
  ∀ n : ℕ, 
  (∃ a : ℕ, a^2 = ( ( n + 1 )^3 - n^3 ) ∧ a^2 < 20000) → count = 40 := by 
sorry

end perfect_squares_represented_as_diff_of_consecutive_cubes_l196_196761


namespace xiaoliang_steps_l196_196692

/-- 
  Xiaoping lives on the fifth floor and climbs 80 steps to get home every day.
  Xiaoliang lives on the fourth floor.
  Prove that the number of steps Xiaoliang has to climb is 60.
-/
theorem xiaoliang_steps (steps_per_floor : ℕ) (h_xiaoping : 4 * steps_per_floor = 80) : 3 * steps_per_floor = 60 :=
by {
  -- The proof is intentionally left out
  sorry
}

end xiaoliang_steps_l196_196692


namespace find_coordinates_of_symmetric_point_l196_196618

def point_on_parabola (A : ℝ × ℝ) : Prop :=
  A.2 = (A.1 - 1)^2 + 2

def symmetric_with_respect_to_axis (A A' : ℝ × ℝ) : Prop :=
  A'.1 = 2 * 1 - A.1 ∧ A'.2 = A.2

def correct_coordinates_of_A' (A' : ℝ × ℝ) : Prop :=
  A' = (3, 6)

theorem find_coordinates_of_symmetric_point (A A' : ℝ × ℝ)
  (hA : A = (-1, 6))
  (h_parabola : point_on_parabola A)
  (h_symmetric : symmetric_with_respect_to_axis A A') :
  correct_coordinates_of_A' A' :=
sorry

end find_coordinates_of_symmetric_point_l196_196618


namespace fraction_identity_l196_196633

theorem fraction_identity (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : (a + b) / a = 7 / 4 :=
by
  sorry

end fraction_identity_l196_196633


namespace find_a_b_find_solution_set_l196_196309

-- Conditions
variable {a b c x : ℝ}

-- Given inequality condition
def given_inequality (x : ℝ) (a b : ℝ) : Prop := a * x^2 + x + b > 0

-- Define the solution set
def solution_set (x : ℝ) (a b : ℝ) : Prop :=
  (x < -2 ∨ x > 1) ↔ given_inequality x a b

-- Part I: Prove values of a and b
theorem find_a_b
  (H : ∀ x, solution_set x a b) :
  a = 1 ∧ b = -2 := by sorry

-- Define the second inequality
def second_inequality (x : ℝ) (c : ℝ) : Prop := x^2 - (c - 2) * x - 2 * c < 0

-- Solution set for the second inequality
def second_solution_set (x : ℝ) (c : ℝ) : Prop :=
  (c = -2 → False) ∧
  (c > -2 → -2 < x ∧ x < c) ∧
  (c < -2 → c < x ∧ x < -2)

-- Part II: Prove the solution set
theorem find_solution_set
  (H : a = 1)
  (H1 : b = -2) :
  ∀ x, second_solution_set x c ↔ second_inequality x c := by sorry

end find_a_b_find_solution_set_l196_196309


namespace gwen_did_not_recycle_2_bags_l196_196486

def points_per_bag : ℕ := 8
def total_bags : ℕ := 4
def points_earned : ℕ := 16

theorem gwen_did_not_recycle_2_bags : total_bags - points_earned / points_per_bag = 2 := by
  sorry

end gwen_did_not_recycle_2_bags_l196_196486


namespace min_output_to_avoid_losses_l196_196683

theorem min_output_to_avoid_losses (x : ℝ) (y : ℝ) (h : y = 0.1 * x - 150) : y ≥ 0 → x ≥ 1500 :=
sorry

end min_output_to_avoid_losses_l196_196683


namespace trajectory_of_T_l196_196051

-- Define coordinates for points A, T, and M
variables {x x0 y y0 : ℝ}
def A (x0: ℝ) (y0: ℝ) := (x0, y0)
def T (x: ℝ) (y: ℝ) := (x, y)
def M : ℝ × ℝ := (-2, 0)

-- Conditions
def curve (x : ℝ) (y : ℝ) := 4 * x^2 - y + 1 = 0
def vector_condition (x x0 y y0 : ℝ) := (x - x0, y - y0) = 2 * (-2 - x, -y)

theorem trajectory_of_T (x y x0 y0 : ℝ) (hA : curve x0 y0) (hV : vector_condition x x0 y y0) :
  4 * (3 * x + 4)^2 - 3 * y + 1 = 0 :=
by
  sorry

end trajectory_of_T_l196_196051


namespace find_common_ratio_l196_196052

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

theorem find_common_ratio (h1 : a_n 1 = 2) (h2 : a_n 4 = 16) (h_geom : ∀ n, a_n n = a_n (n - 1) * q)
  : q = 2 := by
  sorry

end find_common_ratio_l196_196052


namespace vertical_asymptotes_A_plus_B_plus_C_l196_196544

noncomputable def A : ℤ := -6
noncomputable def B : ℤ := 5
noncomputable def C : ℤ := 12

theorem vertical_asymptotes_A_plus_B_plus_C :
  (x + 1) * (x - 3) * (x - 4) = x^3 + A*x^2 + B*x + C ∧ A + B + C = 11 := by
  sorry

end vertical_asymptotes_A_plus_B_plus_C_l196_196544


namespace rectangle_diagonal_length_l196_196359

theorem rectangle_diagonal_length (P L W k d : ℝ) 
  (h1 : P = 72) 
  (h2 : L / W = 3 / 2) 
  (h3 : L = 3 * k) 
  (h4 : W = 2 * k) 
  (h5 : P = 2 * (L + W))
  (h6 : d = Real.sqrt ((L^2) + (W^2))) :
  d = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l196_196359


namespace fruit_bowl_remaining_l196_196994

-- Define the initial conditions
def oranges : Nat := 3
def lemons : Nat := 6
def fruits_eaten : Nat := 3

-- Define the total count of fruits initially
def total_fruits : Nat := oranges + lemons

-- The goal is to prove remaining fruits == 6
theorem fruit_bowl_remaining : total_fruits - fruits_eaten = 6 := by
  sorry

end fruit_bowl_remaining_l196_196994


namespace integral_sup_bound_l196_196492

noncomputable theory

open Complex

variable {c : ℕ → ℂ} (hc : ∀ k : ℕ, abs (c k) ≤ 1)

def binary_repr (n : ℕ) : ℕ → ℕ
| i := if n / 2 ^ i % 2 = 1 then 1 else 0

def xor (k n : ℕ) : ℕ :=
nat.bits (λ i, nat.bits (binary_repr k i) (binary_repr n i))

theorem integral_sup_bound (N : ℕ) (hN : 0 < N) :
  ∃ C δ : ℝ, 0 < C ∧ 0 < δ ∧
    ∫⁻ (x y : ℝ) in set.prod (set.interval (-π) π) (set.interval (-π) π),
      ennreal.of_real (⨆ (n : ℕ) (hn : n < N), (1 / N : ℝ) *
        abs (∑ k in finset.range n, c k * (exp (I * ((k : ℝ) * x + (xor k n) * y)))))
    ≤ C * N ^ (-δ) :=
begin
  sorry
end

end integral_sup_bound_l196_196492


namespace paving_stone_proof_l196_196993

noncomputable def paving_stone_width (length_court : ℝ) (width_court : ℝ) 
                                      (num_stones: ℕ) (stone_length: ℝ) : ℝ :=
  let area_court := length_court * width_court
  let area_stone := stone_length * (area_court / (num_stones * stone_length))
  area_court / area_stone

theorem paving_stone_proof :
  paving_stone_width 50 16.5 165 2.5 = 2 :=
sorry

end paving_stone_proof_l196_196993


namespace expected_rolls_in_non_leap_year_l196_196016

-- Define the conditions and the expected value
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def stops_rolling (n : ℕ) : Prop := is_prime n ∨ is_multiple_of_4 n

def expected_rolls_one_day : ℚ := 6 / 7

def non_leap_year_days : ℕ := 365

def expected_rolls_one_year := expected_rolls_one_day * non_leap_year_days

theorem expected_rolls_in_non_leap_year : expected_rolls_one_year = 314 :=
by
  -- Verification of the mathematical model
  sorry

end expected_rolls_in_non_leap_year_l196_196016


namespace bernardo_larger_probability_l196_196584

open Finset

theorem bernardo_larger_probability :
  let B := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let prob_bernardo_larger := 217 / 336
  in
  (∑ x in powerset_len 3 B, 1) ≠ 0 →
  (∑ y in powerset_len 3 S, 1) ≠ 0 →
  let bernardo_picks := (λ x, x ∈ powerset_len 3 B)
  let silvia_picks := (λ y, y ∈ powerset_len 3 S)
  in (bernardo_larger_probability == prob_bernardo_larger) :=
by
  sorry

end bernardo_larger_probability_l196_196584


namespace work_rate_problem_l196_196425

theorem work_rate_problem
  (W : ℕ) -- total work
  (A_rate : ℕ) -- A's work rate in days
  (B_rate : ℕ) -- B's work rate in days
  (x : ℕ) -- days A worked alone
  (total_days : ℕ) -- days A and B worked together
  (hA : A_rate = 12) -- A can do the work in 12 days
  (hB : B_rate = 6) -- B can do the work in 6 days
  (hx : total_days = 3) -- remaining days they together work
  : x = 3 := 
by
  sorry

end work_rate_problem_l196_196425


namespace largest_of_three_consecutive_integers_l196_196369

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l196_196369


namespace ratio_of_money_spent_on_clothes_is_1_to_2_l196_196171

-- Definitions based on conditions
def allowance1 : ℕ := 5
def weeks1 : ℕ := 8
def allowance2 : ℕ := 6
def weeks2 : ℕ := 6
def cost_video : ℕ := 35
def remaining_money : ℕ := 3

-- Calculations
def total_saved : ℕ := (allowance1 * weeks1) + (allowance2 * weeks2)
def total_expended : ℕ := cost_video + remaining_money
def spent_on_clothes : ℕ := total_saved - total_expended

-- Prove the ratio of money spent on clothes to the total money saved is 1:2
theorem ratio_of_money_spent_on_clothes_is_1_to_2 :
  (spent_on_clothes : ℚ) / (total_saved : ℚ) = 1 / 2 :=
by
  sorry

end ratio_of_money_spent_on_clothes_is_1_to_2_l196_196171


namespace jackie_more_apples_oranges_l196_196711

-- Definitions of initial conditions
def adams_apples : ℕ := 25
def adams_oranges : ℕ := 34
def jackies_apples : ℕ := 43
def jackies_oranges : ℕ := 29

-- The proof statement
theorem jackie_more_apples_oranges :
  (jackies_apples - adams_apples) + (jackies_oranges - adams_oranges) = 13 :=
by
  sorry

end jackie_more_apples_oranges_l196_196711


namespace students_received_B_l196_196076

theorem students_received_B (x : ℕ) 
  (h1 : (0.8 * x : ℝ) + x + (1.2 * x : ℝ) = 28) : 
  x = 9 := 
by
  sorry

end students_received_B_l196_196076


namespace solution_set_inequality_l196_196763

variable (a x : ℝ)

-- Conditions
theorem solution_set_inequality (h₀ : 0 < a) (h₁ : a < 1) :
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) := 
by 
  sorry

end solution_set_inequality_l196_196763


namespace regular_hexagon_has_greatest_lines_of_symmetry_l196_196564

-- Definitions for the various shapes and their lines of symmetry.
def regular_pentagon_lines_of_symmetry : ℕ := 5
def parallelogram_lines_of_symmetry : ℕ := 0
def oval_ellipse_lines_of_symmetry : ℕ := 2
def right_triangle_lines_of_symmetry : ℕ := 0
def regular_hexagon_lines_of_symmetry : ℕ := 6

-- Theorem stating that the regular hexagon has the greatest number of lines of symmetry.
theorem regular_hexagon_has_greatest_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry > regular_pentagon_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > parallelogram_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > oval_ellipse_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > right_triangle_lines_of_symmetry :=
by
  sorry

end regular_hexagon_has_greatest_lines_of_symmetry_l196_196564


namespace range_of_m_for_basis_l196_196628

open Real

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 3 * m - 2)

theorem range_of_m_for_basis (m : ℝ) :
  vector_a ≠ vector_b m → m ≠ 2 :=
sorry

end range_of_m_for_basis_l196_196628


namespace lost_card_number_l196_196410

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l196_196410


namespace operation_positive_l196_196189

theorem operation_positive (op : ℤ → ℤ → ℤ) (is_pos : op 1 (-2) > 0) : op = Int.sub :=
by
  sorry

end operation_positive_l196_196189


namespace total_points_scored_l196_196781

theorem total_points_scored (layla_score nahima_score : ℕ)
  (h1 : layla_score = 70)
  (h2 : layla_score = nahima_score + 28) :
  layla_score + nahima_score = 112 :=
by
  sorry

end total_points_scored_l196_196781


namespace senior_tickets_count_l196_196242

-- Define variables and problem conditions
variables (A S : ℕ)

-- Total number of tickets equation
def total_tickets (A S : ℕ) : Prop := A + S = 510

-- Total receipts equation
def total_receipts (A S : ℕ) : Prop := 21 * A + 15 * S = 8748

-- Prove that the number of senior citizen tickets S is 327
theorem senior_tickets_count (A S : ℕ) (h1 : total_tickets A S) (h2 : total_receipts A S) : S = 327 :=
sorry

end senior_tickets_count_l196_196242


namespace squirrel_rise_per_circuit_l196_196257

noncomputable def rise_per_circuit
    (height : ℕ)
    (circumference : ℕ)
    (distance : ℕ) :=
    height / (distance / circumference)

theorem squirrel_rise_per_circuit : rise_per_circuit 25 3 15 = 5 :=
by
  sorry

end squirrel_rise_per_circuit_l196_196257


namespace min_absolute_difference_l196_196636

open Int

theorem min_absolute_difference (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 4 * x + 3 * y = 215) : |x - y| = 15 :=
sorry

end min_absolute_difference_l196_196636


namespace correct_incorrect_difference_l196_196396

variable (x : ℝ)

theorem correct_incorrect_difference : (x - 2152) - (x - 1264) = 888 := by
  sorry

end correct_incorrect_difference_l196_196396


namespace find_B_value_l196_196042

-- Define the polynomial and conditions
def polynomial (A B : ℤ) (z : ℤ) : ℤ := z^4 - 12 * z^3 + A * z^2 + B * z + 36

-- Define roots and their properties according to the conditions
def roots_sum_to_twelve (r1 r2 r3 r4 : ℕ) : Prop := r1 + r2 + r3 + r4 = 12

-- The final statement to prove
theorem find_B_value (r1 r2 r3 r4 : ℕ) (A B : ℤ) (h_sum : roots_sum_to_twelve r1 r2 r3 r4)
    (h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ r4 > 0) 
    (h_poly : polynomial A B = (z^4 - 12*z^3 + Az^2 + Bz + 36)) :
    B = -96 :=
    sorry

end find_B_value_l196_196042


namespace probability_of_drawing_different_colors_l196_196196

-- Condition: There are 5 balls in total, 3 white and 2 yellow, and two are drawn.
def total_balls : ℕ := 5
def white_balls : ℕ := 3
def yellow_balls : ℕ := 2
def total_drawn : ℕ := 2

-- The number of ways to choose two balls from total_balls
def total_combinations : ℕ := total_balls.choose total_drawn

-- The number of ways to choose one white and one yellow ball
def different_color_combinations : ℕ := white_balls.choose 1 * yellow_balls.choose 1

-- The probability of drawing two balls of different colors
def probability_different_colors : ℝ :=
  different_color_combinations.toReal / total_combinations.toReal

theorem probability_of_drawing_different_colors : probability_different_colors = 0.6 := by
  sorry

end probability_of_drawing_different_colors_l196_196196


namespace fraction_of_satisfactory_grades_is_24_over_35_l196_196098

def num_students_with_grade_A : ℕ := 6
def num_students_with_grade_B : ℕ := 5
def num_students_with_grade_C : ℕ := 4
def num_students_with_grade_D : ℕ := 4
def num_students_with_grade_E : ℕ := 3
def num_students_with_grade_G : ℕ := 2
def num_students_with_grade_F : ℕ := 8
def num_students_with_grade_H : ℕ := 3

def satisfactory_grades : ℕ := 
  num_students_with_grade_A + num_students_with_grade_B + 
  num_students_with_grade_C + num_students_with_grade_D + 
  num_students_with_grade_E + num_students_with_grade_G

def total_students : ℕ := 
  satisfactory_grades + num_students_with_grade_F + num_students_with_grade_H

def fraction_satisfactory_grades : Rat := satisfactory_grades / total_students

-- The theorem we need to prove
theorem fraction_of_satisfactory_grades_is_24_over_35 : fraction_satisfactory_grades = 24 / 35 := by
  sorry

end fraction_of_satisfactory_grades_is_24_over_35_l196_196098


namespace no_integer_solutions_for_sum_of_squares_l196_196463

theorem no_integer_solutions_for_sum_of_squares :
  ∀ a b c : ℤ, a^2 + b^2 + c^2 ≠ 20122012 := 
by sorry

end no_integer_solutions_for_sum_of_squares_l196_196463


namespace no_such_increasing_seq_exists_l196_196524

theorem no_such_increasing_seq_exists :
  ¬(∃ (a : ℕ → ℕ), (∀ m n : ℕ, a (m * n) = a m + a n) ∧ (∀ n : ℕ, a n < a (n + 1))) :=
by
  sorry

end no_such_increasing_seq_exists_l196_196524


namespace mike_total_games_l196_196091

theorem mike_total_games
  (non_working : ℕ)
  (price_per_game : ℕ)
  (total_earnings : ℕ)
  (h1 : non_working = 9)
  (h2 : price_per_game = 5)
  (h3 : total_earnings = 30) :
  non_working + (total_earnings / price_per_game) = 15 := 
by
  sorry

end mike_total_games_l196_196091


namespace question1_question2_question3_l196_196194

-- Define probabilities of renting and returning bicycles at different stations
def P (X Y : Char) : ℝ :=
  if X = 'A' ∧ Y = 'A' then 0.3 else
  if X = 'A' ∧ Y = 'B' then 0.2 else
  if X = 'A' ∧ Y = 'C' then 0.5 else
  if X = 'B' ∧ Y = 'A' then 0.7 else
  if X = 'B' ∧ Y = 'B' then 0.1 else
  if X = 'B' ∧ Y = 'C' then 0.2 else
  if X = 'C' ∧ Y = 'A' then 0.4 else
  if X = 'C' ∧ Y = 'B' then 0.5 else
  if X = 'C' ∧ Y = 'C' then 0.1 else 0

-- Question 1: Prove P(CC) = 0.1
theorem question1 : P 'C' 'C' = 0.1 := by
  sorry

-- Question 2: Prove P(AC) * P(CB) = 0.25
theorem question2 : P 'A' 'C' * P 'C' 'B' = 0.25 := by
  sorry

-- Question 3: Prove the probability P = 0.43
theorem question3 : P 'A' 'A' * P 'A' 'A' + P 'A' 'B' * P 'B' 'A' + P 'A' 'C' * P 'C' 'A' = 0.43 := by
  sorry

end question1_question2_question3_l196_196194


namespace description_of_T_l196_196783

def T : Set (ℝ × ℝ) := { p | ∃ c, (4 = p.1 + 3 ∨ 4 = p.2 - 2 ∨ p.1 + 3 = p.2 - 2) 
                           ∧ (p.1 + 3 ≤ c ∨ p.2 - 2 ≤ c ∨ 4 ≤ c) }

theorem description_of_T : 
  (∀ p ∈ T, (∃ x y : ℝ, p = (x, y) ∧ ((x = 1 ∧ y ≤ 6) ∨ (y = 6 ∧ x ≤ 1) ∨ (y = x + 5 ∧ x ≥ 1 ∧ y ≥ 6)))) :=
sorry

end description_of_T_l196_196783


namespace general_term_of_sequence_l196_196087

def A := {n : ℕ | ∃ k : ℕ, k + 1 = n }
def B := {m : ℕ | ∃ k : ℕ, 3 * k - 1 = m }

theorem general_term_of_sequence (k : ℕ) : 
  ∃ a_k : ℕ, a_k ∈ A ∩ B ∧ a_k = 9 * k^2 - 9 * k + 2 :=
sorry

end general_term_of_sequence_l196_196087


namespace train_length_approx_200_l196_196709

noncomputable def train_length (speed_kmph : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_kmph * 1000) / 3600 * time_sec

theorem train_length_approx_200
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (h_speed : speed_kmph = 120)
  (h_time : time_sec = 6) :
  train_length speed_kmph time_sec ≈ 200 := 
by sorry

end train_length_approx_200_l196_196709


namespace lost_card_number_l196_196405

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196405


namespace expected_yield_correct_l196_196664

-- Conditions
def garden_length_steps : ℕ := 18
def garden_width_steps : ℕ := 25
def step_length_ft : ℝ := 2.5
def yield_per_sqft_pounds : ℝ := 0.75

-- Related quantities
def garden_length_ft : ℝ := garden_length_steps * step_length_ft
def garden_width_ft : ℝ := garden_width_steps * step_length_ft
def garden_area_sqft : ℝ := garden_length_ft * garden_width_ft
def expected_yield_pounds : ℝ := garden_area_sqft * yield_per_sqft_pounds

-- Statement to prove
theorem expected_yield_correct : expected_yield_pounds = 2109.375 := by
  sorry

end expected_yield_correct_l196_196664


namespace additional_flour_minus_salt_l196_196089

structure CakeRecipe where
  flour    : ℕ
  sugar    : ℕ
  salt     : ℕ

def MaryHasAdded (cups_flour : ℕ) (cups_sugar : ℕ) (cups_salt : ℕ) : Prop :=
  cups_flour = 2 ∧ cups_sugar = 0 ∧ cups_salt = 0

variable (r : CakeRecipe)

theorem additional_flour_minus_salt (H : MaryHasAdded 2 0 0) : 
  (r.flour - 2) - r.salt = 3 :=
sorry

end additional_flour_minus_salt_l196_196089


namespace hexagon_unique_intersection_points_are_45_l196_196723

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end hexagon_unique_intersection_points_are_45_l196_196723


namespace tower_height_l196_196991

theorem tower_height (h : ℝ) (hd : ¬ (h ≥ 200)) (he : ¬ (h ≤ 150)) (hf : ¬ (h ≤ 180)) : 180 < h ∧ h < 200 := 
by 
  sorry

end tower_height_l196_196991


namespace eric_running_time_l196_196730

-- Define the conditions
variables (jog_time to_park_time return_time : ℕ)
axiom jog_time_def : jog_time = 10
axiom return_time_def : return_time = 90
axiom trip_relation : return_time = 3 * to_park_time

-- Define the question
def run_time : ℕ := to_park_time - jog_time

-- State the problem: Prove that given the conditions, the running time is 20 minutes.
theorem eric_running_time : run_time = 20 :=
by
  -- Proof goes here
  sorry

end eric_running_time_l196_196730


namespace find_interest_rate_l196_196739

theorem find_interest_rate
  (P : ℝ)  -- Principal amount
  (A : ℝ)  -- Final amount
  (T : ℝ)  -- Time period in years
  (H1 : P = 1000)
  (H2 : A = 1120)
  (H3 : T = 2.4)
  : ∃ R : ℝ, (A - P) = (P * R * T) / 100 ∧ R = 5 :=
by
  -- Proof with calculations to be provided here
  sorry

end find_interest_rate_l196_196739


namespace tangent_line_at_0_1_is_correct_l196_196357

theorem tangent_line_at_0_1_is_correct :
  let f := λ x : ℝ, x * Real.exp x + 2 * x + 1
  let f' := λ x : ℝ, (1 + x) * Real.exp x + 2
  ∀ x : ℝ, f 0 = 1 → f' 0 = 3 → (∀ x : ℝ, f' 0 * x + 1 = 3 * x + 1) :=
by
  intro f f'
  assume h₁ h₂
  intro x
  rw [← h₂, mul_comm]
  rfl

end tangent_line_at_0_1_is_correct_l196_196357


namespace four_digit_number_sum_l196_196281

theorem four_digit_number_sum (x y z w : ℕ) (h1 : 1001 * x + 101 * y + 11 * z + 2 * w = 2003)
  (h2 : x = 1) : (x = 1 ∧ y = 9 ∧ z = 7 ∧ w = 8) ↔ (1000 * x + 100 * y + 10 * z + w = 1978) :=
by sorry

end four_digit_number_sum_l196_196281


namespace number_of_jars_pasta_sauce_l196_196790

-- Conditions
def pasta_cost_per_kg := 1.5
def pasta_weight_kg := 2.0
def ground_beef_cost_per_kg := 8.0
def ground_beef_weight_kg := 1.0 / 4.0
def quesadilla_cost := 6.0
def jar_sauce_cost := 2.0
def total_money := 15.0

-- Helper definitions for total costs
def pasta_total_cost := pasta_weight_kg * pasta_cost_per_kg
def ground_beef_total_cost := ground_beef_weight_kg * ground_beef_cost_per_kg
def other_total_cost := quesadilla_cost + pasta_total_cost + ground_beef_total_cost
def remaining_money := total_money - other_total_cost

-- Proof statement
theorem number_of_jars_pasta_sauce :
  (remaining_money / jar_sauce_cost) = 2 := by
  sorry

end number_of_jars_pasta_sauce_l196_196790


namespace intersection_M_N_l196_196898

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196898


namespace initial_gift_card_value_l196_196802

-- The price per pound of coffee
def cost_per_pound : ℝ := 8.58

-- The number of pounds of coffee bought by Rita
def pounds_bought : ℝ := 4.0

-- The remaining balance on Rita's gift card after buying coffee
def remaining_balance : ℝ := 35.68

-- The total cost of the coffee Rita bought
def total_cost_of_coffee : ℝ := cost_per_pound * pounds_bought

-- The initial value of Rita's gift card
def initial_value_of_gift_card : ℝ := total_cost_of_coffee + remaining_balance

-- Statement of the proof problem
theorem initial_gift_card_value : initial_value_of_gift_card = 70.00 :=
by
  -- Placeholder for the proof
  sorry

end initial_gift_card_value_l196_196802


namespace pythagorean_triple_transformation_l196_196168

theorem pythagorean_triple_transformation
  (a b c α β γ s p q r : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : α^2 + β^2 - γ^2 = 2)
  (h₃ : s = a * α + b * β - c * γ)
  (h₄ : p = a - α * s)
  (h₅ : q = b - β * s)
  (h₆ : r = c - γ * s) :
  p^2 + q^2 = r^2 :=
by
  sorry

end pythagorean_triple_transformation_l196_196168


namespace f_odd_function_f_decreasing_f_max_min_values_l196_196893

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) (hx : 0 < x) : f x < 0
axiom f_value : f 3 = -2

theorem f_odd_function : ∀ (x : ℝ), f (-x) = - f x := sorry
theorem f_decreasing : ∀ (x y : ℝ), x < y → f x > f y := sorry
theorem f_max_min_values : ∀ (x : ℝ), -12 ≤ x ∧ x ≤ 12 → f x ≤ 8 ∧ f x ≥ -8 := sorry

end f_odd_function_f_decreasing_f_max_min_values_l196_196893


namespace cylinder_height_l196_196681

theorem cylinder_height (r h : ℝ) (SA : ℝ) 
  (hSA : SA = 2 * Real.pi * r ^ 2 + 2 * Real.pi * r * h) 
  (hr : r = 3) (hSA_val : SA = 36 * Real.pi) : 
  h = 3 :=
by
  sorry

end cylinder_height_l196_196681


namespace max_radius_of_circle_in_triangle_inscribed_l196_196508

theorem max_radius_of_circle_in_triangle_inscribed (ω : Set (ℝ × ℝ)) (hω : ∀ (P : ℝ × ℝ), P ∈ ω → P.1^2 + P.2^2 = 1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) (P : ℝ × ℝ) (hP : P ∈ ω) (A : ℝ × ℝ) 
  (hA : A = (P.1, 0)) : 
  (∃ r : ℝ, r = (Real.sqrt 2 - 1) / 2) :=
by
  sorry

end max_radius_of_circle_in_triangle_inscribed_l196_196508


namespace sin_of_double_angle_equals_neg_seventeen_over_twentyfive_l196_196184

theorem sin_of_double_angle_equals_neg_seventeen_over_twentyfive (α : ℝ) :
  sin (π / 4 + α) = 2 / 5 → sin (2 * α) = -17 / 25 :=
by
  intros h
  sorry

end sin_of_double_angle_equals_neg_seventeen_over_twentyfive_l196_196184


namespace range_of_m_l196_196788

open Set

def M (m : ℝ) : Set ℝ := {x | x ≤ m}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}

theorem range_of_m (m : ℝ) : (M m ∩ N).Nonempty ↔ m > 0 := sorry

end range_of_m_l196_196788


namespace max_y_coordinate_l196_196466

open Real

noncomputable def y_coordinate (θ : ℝ) : ℝ :=
  let k := sin θ in
  3 * k - 4 * k^4

theorem max_y_coordinate :
  ∃ θ : ℝ, y_coordinate θ = 3 * (3 / 16)^(1/3) - 4 * ((3 / 16)^(1/3))^4 :=
sorry

end max_y_coordinate_l196_196466


namespace minimum_positive_period_of_f_is_pi_l196_196813

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 4))^2 - (Real.sin (x + Real.pi / 4))^2

theorem minimum_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 ∧ (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end minimum_positive_period_of_f_is_pi_l196_196813


namespace calf_rope_length_l196_196706

noncomputable def new_rope_length (initial_length : ℝ) (additional_area : ℝ) : ℝ :=
  let A1 := Real.pi * initial_length ^ 2
  let A2 := A1 + additional_area
  let new_length_squared := A2 / Real.pi
  Real.sqrt new_length_squared

theorem calf_rope_length :
  new_rope_length 12 565.7142857142857 = 18 := by
  sorry

end calf_rope_length_l196_196706


namespace distance_to_pinedale_mall_l196_196420

-- Define the conditions given in the problem
def average_speed : ℕ := 60  -- km/h
def stops_interval : ℕ := 5   -- minutes
def number_of_stops : ℕ := 8

-- The distance from Yahya's house to Pinedale Mall
theorem distance_to_pinedale_mall : 
  (average_speed * (number_of_stops * stops_interval / 60) = 40) :=
by
  sorry

end distance_to_pinedale_mall_l196_196420


namespace smallest_even_integer_l196_196363

theorem smallest_even_integer (n : ℕ) (h_even : n % 2 = 0)
  (h_2digit : 10 ≤ n ∧ n ≤ 98)
  (h_property : (n - 2) * n * (n + 2) = 5 * ((n - 2) + n + (n + 2))) :
  n = 86 :=
by
  sorry

end smallest_even_integer_l196_196363


namespace problem_l196_196592

def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f x = -f (-x)
axiom periodic_g : ∀ x, g x = g (x + 2)
axiom f_at_neg1 : f (-1) = 3
axiom g_at_1 : g 1 = 3
axiom g_function : ∀ n : ℕ, g (2 * n * f 1) = n * f (f 1 + g (-1)) + 2

theorem problem : g (-6) + f 0 = 2 :=
by sorry

end problem_l196_196592


namespace work_completion_l196_196846

theorem work_completion (a b : ℝ) 
  (h1 : a + b = 6) 
  (h2 : a = 10) : 
  a + b = 6 :=
by sorry

end work_completion_l196_196846


namespace vector_parallel_solution_l196_196061

-- Define the vectors and the condition
def a (m : ℝ) := (2 * m + 1, 3)
def b (m : ℝ) := (2, m)

-- The proof problem statement
theorem vector_parallel_solution (m : ℝ) :
  (2 * m + 1) * m = 3 * 2 ↔ m = 3 / 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_solution_l196_196061


namespace gcd_of_720_120_168_is_24_l196_196812

theorem gcd_of_720_120_168_is_24 : Int.gcd (Int.gcd 720 120) 168 = 24 := 
by sorry

end gcd_of_720_120_168_is_24_l196_196812


namespace bead_arrangement_probability_l196_196566

def total_beads := 6
def red_beads := 2
def white_beads := 2
def blue_beads := 2

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

def valid_arrangements : ℕ := 6  -- Based on valid patterns RWBRWB, RWBWRB, and all other permutations for each starting color

def probability_valid := valid_arrangements / total_arrangements

theorem bead_arrangement_probability : probability_valid = 1 / 15 :=
  by
  -- The context and details of the solution steps are omitted as they are not included in the Lean theorem statement.
  -- This statement will skip the proof
  sorry

end bead_arrangement_probability_l196_196566


namespace license_plate_combinations_l196_196019

open Nat

theorem license_plate_combinations : 
  (∃ (choose_two_letters: ℕ) (place_first_letter: ℕ) (place_second_letter: ℕ) (choose_non_repeated: ℕ)
     (first_digit: ℕ) (second_digit: ℕ) (third_digit: ℕ),
    choose_two_letters = choose 26 2 ∧
    place_first_letter = choose 5 2 ∧
    place_second_letter = choose 3 2 ∧
    choose_non_repeated = 24 ∧
    first_digit = 10 ∧
    second_digit = 9 ∧
    third_digit = 8 ∧
    choose_two_letters * place_first_letter * place_second_letter * choose_non_repeated * first_digit * second_digit * third_digit = 56016000) :=
sorry

end license_plate_combinations_l196_196019


namespace intersection_M_N_l196_196913

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l196_196913


namespace intersection_M_N_l196_196907

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196907


namespace Mary_avg_speed_l196_196662

def Mary_uphill_distance := 1.5 -- km
def Mary_uphill_time := 45.0 / 60.0 -- hours
def Mary_downhill_distance := 1.5 -- km
def Mary_downhill_time := 15.0 / 60.0 -- hours

def total_distance := Mary_uphill_distance + Mary_downhill_distance
def total_time := Mary_uphill_time + Mary_downhill_time

theorem Mary_avg_speed : 
  (total_distance / total_time) = 3.0 := by
  sorry

end Mary_avg_speed_l196_196662


namespace hyperbola_equation_l196_196049

variable (a b : ℝ)
variable (c : ℝ) (h1 : c = 4)
variable (h2 : b / a = Real.sqrt 3)
variable (h3 : a ^ 2 + b ^ 2 = c ^ 2)

theorem hyperbola_equation : (a ^ 2 = 4) ∧ (b ^ 2 = 12) ↔ (∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → (x ^ 2 / 4) - (y ^ 2 / 12) = 1) := by
  sorry

end hyperbola_equation_l196_196049


namespace cost_plane_l196_196090

def cost_boat : ℝ := 254.00
def savings_boat : ℝ := 346.00

theorem cost_plane : cost_boat + savings_boat = 600 := 
by 
  sorry

end cost_plane_l196_196090


namespace simplify_logarithmic_expression_l196_196674

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem simplify_logarithmic_expression :
  simplify_expression = 4 / 3 :=
by
  sorry

end simplify_logarithmic_expression_l196_196674


namespace james_faster_than_john_l196_196651

theorem james_faster_than_john :
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds
  
  james_top_speed - john_top_speed = 2 :=
by
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds

  sorry

end james_faster_than_john_l196_196651


namespace find_common_difference_l196_196057

variable {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) = a n + d)
variable (a7_minus_2a4_eq_6 : a 7 - 2 * a 4 = 6)
variable (a3_eq_2 : a 3 = 2)

theorem find_common_difference (d : ℝ) : d = 4 :=
by
  -- Proof would go here
  sorry

end find_common_difference_l196_196057


namespace jason_investing_months_l196_196096

noncomputable def initial_investment (total_amount earned_amount_per_month : ℕ) := total_amount / 3
noncomputable def months_investing (initial_investment earned_amount_per_month : ℕ) := (2 * initial_investment) / earned_amount_per_month

theorem jason_investing_months (total_amount earned_amount_per_month : ℕ) 
  (h1 : total_amount = 90) 
  (h2 : earned_amount_per_month = 12) 
  : months_investing (initial_investment total_amount earned_amount_per_month) earned_amount_per_month = 5 := 
by
  sorry

end jason_investing_months_l196_196096


namespace solve_ordered_pair_l196_196595

theorem solve_ordered_pair : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^y + 3 = y^x ∧ 2 * x^y = y^x + 11 ∧ x = 14 ∧ y = 1 :=
by
  sorry

end solve_ordered_pair_l196_196595


namespace int_sol_many_no_int_sol_l196_196877

-- Part 1: If there is one integer solution, there are at least three integer solutions
theorem int_sol_many (n : ℤ) (hn : n > 0) (x y : ℤ) 
  (hxy : x^3 - 3 * x * y^2 + y^3 = n) : 
  ∃ a b c d e f : ℤ, 
    (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (e, f) ≠ (x, y) ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f) ∧ 
    a^3 - 3 * a * b^2 + b^3 = n ∧ 
    c^3 - 3 * c * d^2 + d^3 = n ∧ 
    e^3 - 3 * e * f^2 + f^3 = n :=
sorry

-- Part 2: When n = 2891, the equation has no integer solutions
theorem no_int_sol : ¬ ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = 2891 :=
sorry

end int_sol_many_no_int_sol_l196_196877


namespace power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l196_196570

theorem power_of_two_minus_one_divisible_by_seven (n : ℕ) (hn : 0 < n) : 
  (∃ k : ℕ, 0 < k ∧ n = k * 3) ↔ (7 ∣ 2^n - 1) :=
by sorry

theorem power_of_two_plus_one_not_divisible_by_seven (n : ℕ) (hn : 0 < n) :
  ¬(7 ∣ 2^n + 1) :=
by sorry

end power_of_two_minus_one_divisible_by_seven_power_of_two_plus_one_not_divisible_by_seven_l196_196570


namespace bmw_cars_sold_l196_196001

def percentage_non_bmw (ford_pct nissan_pct chevrolet_pct : ℕ) : ℕ :=
  ford_pct + nissan_pct + chevrolet_pct

def percentage_bmw (total_pct non_bmw_pct : ℕ) : ℕ :=
  total_pct - non_bmw_pct

def number_of_bmws (total_cars bmw_pct : ℕ) : ℕ :=
  (total_cars * bmw_pct) / 100

theorem bmw_cars_sold (total_cars ford_pct nissan_pct chevrolet_pct : ℕ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 20)
  (h_nissan_pct : nissan_pct = 25)
  (h_chevrolet_pct : chevrolet_pct = 10) :
  number_of_bmws total_cars (percentage_bmw 100 (percentage_non_bmw ford_pct nissan_pct chevrolet_pct)) = 135 := by
  sorry

end bmw_cars_sold_l196_196001


namespace base7_perfect_square_values_l196_196863

theorem base7_perfect_square_values (a b c : ℕ) (h1 : a ≠ 0) (h2 : b < 7) :
  ∃ (n : ℕ), (343 * a + 49 * c + 28 + b = n * n) → (b = 0 ∨ b = 1 ∨ b = 4) :=
by
  sorry

end base7_perfect_square_values_l196_196863


namespace trig_proof_l196_196747

theorem trig_proof (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_proof_l196_196747


namespace range_of_a_l196_196752

noncomputable def function_monotonicity (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Ico (-1 : ℝ) 2, (has_deriv_at f (2 * x * a + 2) x) → (2 * x * a + 2) ≥ 0

theorem range_of_a : 
  (∀ f, f = (λ x : ℝ, a * x^2 + 2 * x - 2 * a) → function_monotonicity f a) → 
  a ∈ Icc (-1 / 2 : ℝ) 1 :=
sorry

end range_of_a_l196_196752


namespace lost_card_number_l196_196409

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l196_196409


namespace subtraction_correctness_l196_196160

theorem subtraction_correctness : 25.705 - 3.289 = 22.416 := 
by
  sorry

end subtraction_correctness_l196_196160


namespace investment_ratio_l196_196654

theorem investment_ratio (total_investment Jim_investment : ℕ) (h₁ : total_investment = 80000) (h₂ : Jim_investment = 36000) :
  (total_investment - Jim_investment) / Nat.gcd (total_investment - Jim_investment) Jim_investment = 11 ∧ Jim_investment / Nat.gcd (total_investment - Jim_investment) Jim_investment = 9 :=
by
  sorry

end investment_ratio_l196_196654


namespace number_of_adults_had_meal_l196_196432

theorem number_of_adults_had_meal (A : ℝ) :
  let num_children_food : ℝ := 63
  let food_for_adults : ℝ := 70
  let food_for_children : ℝ := 90
  (food_for_children - A * (food_for_children / food_for_adults) = num_children_food) →
  A = 21 :=
by
  intros num_children_food food_for_adults food_for_children h
  have h2 : 90 - A * (90 / 70) = 63 := h
  sorry

end number_of_adults_had_meal_l196_196432


namespace increase_factor_l196_196221

-- Definition of parameters: number of letters, digits, and symbols.
def num_letters : ℕ := 26
def num_digits : ℕ := 10
def num_symbols : ℕ := 5

-- Definition of the number of old license plates and new license plates.
def num_old_plates : ℕ := num_letters ^ 2 * num_digits ^ 3
def num_new_plates : ℕ := num_letters ^ 3 * num_digits ^ 3 * num_symbols

-- The proof problem statement: Prove that the increase factor is 130.
theorem increase_factor : num_new_plates / num_old_plates = 130 := by
  sorry

end increase_factor_l196_196221


namespace solution_l196_196265

noncomputable def problem (x : ℝ) (h : x ≠ 3) : ℝ :=
  (3 * x / (x - 3)) + ((x + 6) / (3 - x))

theorem solution (x : ℝ) (h : x ≠ 3) : problem x h = 2 :=
by
  sorry

end solution_l196_196265


namespace yuri_lost_card_l196_196413

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l196_196413


namespace route_Y_saves_2_minutes_l196_196966

noncomputable def distance_X : ℝ := 8
noncomputable def speed_X : ℝ := 40

noncomputable def distance_Y1 : ℝ := 5
noncomputable def speed_Y1 : ℝ := 50
noncomputable def distance_Y2 : ℝ := 1
noncomputable def speed_Y2 : ℝ := 20
noncomputable def distance_Y3 : ℝ := 1
noncomputable def speed_Y3 : ℝ := 60

noncomputable def t_X : ℝ := (distance_X / speed_X) * 60
noncomputable def t_Y1 : ℝ := (distance_Y1 / speed_Y1) * 60
noncomputable def t_Y2 : ℝ := (distance_Y2 / speed_Y2) * 60
noncomputable def t_Y3 : ℝ := (distance_Y3 / speed_Y3) * 60
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3

noncomputable def time_saved : ℝ := t_X - t_Y

theorem route_Y_saves_2_minutes :
  time_saved = 2 := by
  sorry

end route_Y_saves_2_minutes_l196_196966


namespace smallest_angle_of_triangle_l196_196545

theorem smallest_angle_of_triangle (a b c : ℕ) 
    (h1 : a = 60) (h2 : b = 70) (h3 : a + b + c = 180) : 
    c = 50 ∧ min a (min b c) = 50 :=
by {
    sorry
}

end smallest_angle_of_triangle_l196_196545


namespace part_a_part_b_l196_196850

-- Part (a)
theorem part_a (x y z : ℤ) : (x^2 + y^2 + z^2 = 2 * x * y * z) → (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

-- Part (b)
theorem part_b : ∃ (x y z v : ℤ), (x^2 + y^2 + z^2 + v^2 = 2 * x * y * z * v) → (x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) :=
by
  sorry

end part_a_part_b_l196_196850


namespace sum_of_solutions_l196_196839

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l196_196839


namespace amc_inequality_l196_196082

theorem amc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := 
by 
  sorry

end amc_inequality_l196_196082


namespace find_a10_l196_196648

noncomputable def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d a₁, a 1 = a₁ ∧ ∀ n, a (n + 1) = a n + d

theorem find_a10 (a : ℕ → ℤ) (h_seq : arithmeticSequence a) 
  (h1 : a 1 + a 3 + a 5 = 9) 
  (h2 : a 3 * (a 4) ^ 2 = 27) :
  a 10 = -39 ∨ a 10 = 30 :=
sorry

end find_a10_l196_196648


namespace problem_statement_l196_196925

section

variable {f : ℝ → ℝ}

-- Conditions
axiom even_function (h : ∀ x : ℝ, f (-x) = f x) : ∀ x, f (-x) = f x 
axiom monotonically_increasing (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Goal
theorem problem_statement 
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  f (-Real.log 2 / Real.log 3) > f (Real.log 2 / Real.log 3) ∧ f (Real.log 2 / Real.log 3) > f 0 := 
sorry

end

end problem_statement_l196_196925


namespace circle_radius_tangents_l196_196443

theorem circle_radius_tangents
  (AB CD EF r : ℝ)
  (circle_tangent_AB : AB = 5)
  (circle_tangent_CD : CD = 11)
  (circle_tangent_EF : EF = 15) :
  r = 2.5 := by
  sorry

end circle_radius_tangents_l196_196443


namespace area_of_triangle_ABC_l196_196201

-- Definitions as per the conditions
noncomputable def triangle_ABC_right : Type := 
  {ABC : Triangle ℝ // right_triangle ABC ∧ angle_ABC ABC = 45 ∧ angle_ACB ABC = 90}

-- Altitude length from C to hypotenuse AB
def altitude_CD (ABC : triangle_ABC_right) : ℝ :=
  sqrt 2

-- Statement to express the area to be proved
theorem area_of_triangle_ABC (ABC : triangle_ABC_right) :
  Triangle.area ABC.to_subtype.val = 2 :=
by
  sorry

end area_of_triangle_ABC_l196_196201


namespace seven_in_M_l196_196515

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the set M complement with respect to U
def compl_U_M : Set ℕ := {1, 3, 5}

-- Define the set M
def M : Set ℕ := U \ compl_U_M

-- Prove that 7 is an element of M
theorem seven_in_M : 7 ∈ M :=
by {
  sorry
}

end seven_in_M_l196_196515


namespace A_n_plus_B_n_eq_2n_cubed_l196_196797

-- Definition of A_n given the grouping of positive integers
def A_n (n : ℕ) : ℕ :=
  let sum_first_n_squared := n * n * (n * n + 1) / 2
  let sum_first_n_minus_1_squared := (n - 1) * (n - 1) * ((n - 1) * (n - 1) + 1) / 2
  sum_first_n_squared - sum_first_n_minus_1_squared

-- Definition of B_n given the array of cubes of natural numbers
def B_n (n : ℕ) : ℕ := n * n * n - (n - 1) * (n - 1) * (n - 1)

-- The theorem to prove that A_n + B_n = 2n^3
theorem A_n_plus_B_n_eq_2n_cubed (n : ℕ) : A_n n + B_n n = 2 * n^3 := by
  sorry

end A_n_plus_B_n_eq_2n_cubed_l196_196797


namespace insurance_covers_80_percent_of_lenses_l196_196505

/--
James needs to get a new pair of glasses. 
His frames cost $200 and the lenses cost $500. 
Insurance will cover a certain percentage of the cost of lenses and he has a $50 off coupon for frames. 
Everything costs $250. 
Prove that the insurance covers 80% of the cost of the lenses.
-/

def frames_cost : ℕ := 200
def lenses_cost : ℕ := 500
def total_cost_after_discounts_and_insurance : ℕ := 250
def coupon : ℕ := 50

theorem insurance_covers_80_percent_of_lenses :
  ((frames_cost - coupon + lenses_cost - total_cost_after_discounts_and_insurance) * 100 / lenses_cost) = 80 := 
  sorry

end insurance_covers_80_percent_of_lenses_l196_196505


namespace maximum_ratio_x_over_y_l196_196214

theorem maximum_ratio_x_over_y {x y : ℕ} (hx : x > 9 ∧ x < 100) (hy : y > 9 ∧ y < 100)
  (hmean : x + y = 110) (hsquare : ∃ z : ℕ, z^2 = x * y) : x = 99 ∧ y = 11 := 
by
  -- mathematical proof
  sorry

end maximum_ratio_x_over_y_l196_196214


namespace last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l196_196872

theorem last_digit_of_sum_1_to_5 : 
  (1 ^ 2012 + 2 ^ 2012 + 3 ^ 2012 + 4 ^ 2012 + 5 ^ 2012) % 10 = 9 :=
  sorry

theorem last_digit_of_sum_1_to_2012 : 
  (List.sum (List.map (λ k => k ^ 2012) (List.range 2012).tail)) % 10 = 0 :=
  sorry

end last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l196_196872


namespace abc_eq_ab_bc_ca_l196_196964

variable {u v w A B C : ℝ}
variable (Huvw : u * v * w = 1)
variable (HA : A = u * v + u + 1)
variable (HB : B = v * w + v + 1)
variable (HC : C = w * u + w + 1)

theorem abc_eq_ab_bc_ca 
  (Huvw : u * v * w = 1)
  (HA : A = u * v + u + 1)
  (HB : B = v * w + v + 1)
  (HC : C = w * u + w + 1) : 
  A * B * C = A * B + B * C + C * A := 
by
  sorry

end abc_eq_ab_bc_ca_l196_196964


namespace total_spent_in_may_l196_196275

-- Conditions as definitions
def cost_per_weekday : ℕ := (2 * 15) + (2 * 18)
def cost_per_weekend_day : ℕ := (3 * 12) + (2 * 20)
def weekdays_in_may : ℕ := 22
def weekend_days_in_may : ℕ := 9

-- The statement to prove
theorem total_spent_in_may :
  cost_per_weekday * weekdays_in_may + cost_per_weekend_day * weekend_days_in_may = 2136 :=
by
  sorry

end total_spent_in_may_l196_196275


namespace max_reflections_l196_196240

theorem max_reflections (angle_increase : ℕ := 10) (max_angle : ℕ := 90) :
  ∃ n : ℕ, 10 * n ≤ max_angle ∧ ∀ m : ℕ, (10 * (m + 1) > max_angle → m < n) := 
sorry

end max_reflections_l196_196240


namespace participating_girls_l196_196571

theorem participating_girls (total_students boys_participation girls_participation participating_students : ℕ)
  (h1 : total_students = 800)
  (h2 : boys_participation = 2)
  (h3 : girls_participation = 3)
  (h4 : participating_students = 550) :
  (4 / total_students) * (boys_participation / 3) * total_students + (4 * girls_participation / 4) * total_students = 4 * 150 :=
by
  sorry

end participating_girls_l196_196571


namespace bridge_supports_88_ounces_l196_196204

-- Define the conditions
def weight_of_soda_per_can : ℕ := 12
def number_of_soda_cans : ℕ := 6
def weight_of_empty_can : ℕ := 2
def additional_empty_cans : ℕ := 2

-- Define the total weight the bridge must hold up
def total_weight_bridge_support : ℕ :=
  (number_of_soda_cans * weight_of_soda_per_can) + ((number_of_soda_cans + additional_empty_cans) * weight_of_empty_can)

-- Prove that the total weight is 88 ounces
theorem bridge_supports_88_ounces : total_weight_bridge_support = 88 := by
  sorry

end bridge_supports_88_ounces_l196_196204


namespace sqrt_9_eq_pos_neg_3_l196_196834

theorem sqrt_9_eq_pos_neg_3 : ∀ x : ℝ, x^2 = 9 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end sqrt_9_eq_pos_neg_3_l196_196834


namespace fraction_inequality_l196_196488

theorem fraction_inequality {a b : ℝ} (h1 : a < b) (h2 : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end fraction_inequality_l196_196488


namespace remaining_amount_is_16_l196_196527

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end remaining_amount_is_16_l196_196527


namespace relationship_of_y_l196_196491

theorem relationship_of_y (k y1 y2 y3 : ℝ)
  (hk : k < 0)
  (hy1 : y1 = k / -2)
  (hy2 : y2 = k / 1)
  (hy3 : y3 = k / 2) :
  y2 < y3 ∧ y3 < y1 := by
  -- Proof omitted
  sorry

end relationship_of_y_l196_196491


namespace quadrilateral_rectangle_ratio_l196_196170

theorem quadrilateral_rectangle_ratio
  (s x y : ℝ)
  (h_area : (s + 2 * x) ^ 2 = 4 * s ^ 2)
  (h_y : 2 * y = s) :
  y / x = 1 :=
by
  sorry

end quadrilateral_rectangle_ratio_l196_196170


namespace total_feed_amount_l196_196826

theorem total_feed_amount (x : ℝ) : 
  (17 * 0.18) + (x * 0.53) = (17 + x) * 0.36 → 17 + x = 35 :=
by
  intros h
  sorry

end total_feed_amount_l196_196826


namespace ivanov_voted_against_kuznetsov_l196_196423

theorem ivanov_voted_against_kuznetsov
    (members : List String)
    (vote : String → String)
    (majority_dismissed : (String × Nat))
    (petrov_statement : String)
    (ivanov_concluded : Bool) :
  members = ["Ivanov", "Petrov", "Sidorov", "Kuznetsov"] →
  (∀ x ∈ members, vote x ∈ members ∧ vote x ≠ x) →
  majority_dismissed = ("Ivanov", 3) →
  petrov_statement = "Petrov voted against Kuznetsov" →
  ivanov_concluded = True →
  vote "Ivanov" = "Kuznetsov" :=
by
  intros members_cond vote_cond majority_cond petrov_cond ivanov_cond
  sorry

end ivanov_voted_against_kuznetsov_l196_196423


namespace simplify_expression_l196_196094

theorem simplify_expression : 20 * (9 / 14) * (1 / 18) = 5 / 7 :=
by sorry

end simplify_expression_l196_196094


namespace find_number_l196_196689

theorem find_number (N : ℝ) (h : 6 + (1/2) * (1/3) * (1/5) * N = (1/15) * N) : N = 180 :=
by 
  sorry

end find_number_l196_196689


namespace exponential_inequality_l196_196487

variable (a b : ℝ)

theorem exponential_inequality (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b :=
by
  sorry

end exponential_inequality_l196_196487


namespace probability_multiple_4_or_15_l196_196704

-- Definitions of natural number range and a set of multiples
def first_30_nat_numbers : Finset ℕ := Finset.range 30
def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

-- Conditions
def multiples_of_4 := multiples_of 4 first_30_nat_numbers
def multiples_of_15 := multiples_of 15 first_30_nat_numbers

-- Proof that probability of selecting a multiple of 4 or 15 is 3 / 10
theorem probability_multiple_4_or_15 : 
  let favorable_outcomes := (multiples_of_4 ∪ multiples_of_15).card
  let total_outcomes := first_30_nat_numbers.card
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  -- correct answer based on the computation
  sorry

end probability_multiple_4_or_15_l196_196704


namespace arithmetic_sequence_nth_term_l196_196231

theorem arithmetic_sequence_nth_term (a b c n : ℕ) (x: ℕ)
  (h1: a = 3*x - 4)
  (h2: b = 6*x - 17)
  (h3: c = 4*x + 5)
  (h4: b - a = c - b)
  (h5: a + (n - 1) * (b - a) = 4021) : 
  n = 502 :=
by 
  sorry

end arithmetic_sequence_nth_term_l196_196231


namespace intersection_M_N_is_correct_l196_196919

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l196_196919


namespace integer_solutions_to_equation_l196_196736

theorem integer_solutions_to_equation :
  {p : ℤ × ℤ | (p.fst^2 - 2 * p.fst * p.snd - 3 * p.snd^2 = 5)} =
  {(4, 1), (2, -1), (-4, -1), (-2, 1)} :=
by {
  sorry
}

end integer_solutions_to_equation_l196_196736


namespace intersection_M_N_l196_196914

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l196_196914


namespace original_solution_sugar_percentage_l196_196866

theorem original_solution_sugar_percentage :
  ∃ x : ℚ, (∀ (y : ℚ), (y = 14) → (∃ (z : ℚ), (z = 26) → (3 / 4 * x + 1 / 4 * z = y))) → x = 10 := 
  sorry

end original_solution_sugar_percentage_l196_196866


namespace equation_solution_l196_196803

theorem equation_solution (x : ℚ) (h₁ : (5 * x^2 + 4 * x + 2) / (x + 2) = 5 * x - 3) : x = 8 / 3 :=
by
  sorry

end equation_solution_l196_196803


namespace no_hiphop_or_contemporary_not_in_slow_l196_196378

namespace DanceProblem

-- Defining the conditions
def total_kids : ℕ := 140
def fraction_dancers : ℚ := 1 / 4
def total_dancers : ℕ := (total_kids * fraction_dancers).natAbs
def ratio_slow_hiphop_contemporary : ℚ := 5 / 10 + 3 / 10 + 2 / 10
def slow_dance_ratio : ℚ := 5 / ratio_slow_hiphop_contemporary
def slow_dance_kids : ℕ := 25
def ratio_part_kids : ℕ := (slow_dance_kids / 5).natAbs
def hiphop_kids : ℕ := (3 * ratio_part_kids).natAbs
def contemporary_kids : ℕ := (2 * ratio_part_kids).natAbs
def total_dancers_minus_slow : ℕ := total_dancers - slow_dance_kids

-- Theorem statement
theorem no_hiphop_or_contemporary_not_in_slow :
  (hiphop_kids + contemporary_kids) - (total_dancers_minus_slow) = 0 :=
by
  sorry

end DanceProblem

end no_hiphop_or_contemporary_not_in_slow_l196_196378


namespace sum_of_all_possible_x_l196_196775

theorem sum_of_all_possible_x : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 8 ∨ x = 2)) → ( ∃ (x1 x2 : ℝ), (x1 = 8) ∧ (x2 = 2) ∧ (x1 + x2 = 10) ) :=
by
  admit

end sum_of_all_possible_x_l196_196775


namespace smallest_k_l196_196591

def v_seq (v : ℕ → ℝ) : Prop :=
  v 0 = 1/8 ∧ ∀ k, v (k + 1) = 3 * v k - 3 * (v k)^2

noncomputable def limit_M : ℝ := 0.5

theorem smallest_k 
  (v : ℕ → ℝ)
  (hv : v_seq v) :
  ∃ k : ℕ, |v k - limit_M| ≤ 1 / 2 ^ 500 ∧ ∀ n < k, ¬ (|v n - limit_M| ≤ 1 / 2 ^ 500) := 
sorry

end smallest_k_l196_196591


namespace george_elaine_ratio_l196_196445

-- Define the conditions
def time_jerry := 3
def time_elaine := 2 * time_jerry
def time_kramer := 0
def total_time := 11

-- Define George's time based on the given total time condition
def time_george := total_time - (time_jerry + time_elaine + time_kramer)

-- Prove the ratio of George's time to Elaine's time is 1:3
theorem george_elaine_ratio : time_george / time_elaine = 1 / 3 :=
by
  -- Lean proof would go here
  sorry

end george_elaine_ratio_l196_196445


namespace bird_average_l196_196519

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l196_196519


namespace hyperbola_range_m_l196_196069

theorem hyperbola_range_m (m : ℝ) : (m - 2) * (m - 6) < 0 ↔ 2 < m ∧ m < 6 :=
by sorry

end hyperbola_range_m_l196_196069


namespace angle_diff_complement_supplement_l196_196810

theorem angle_diff_complement_supplement (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end angle_diff_complement_supplement_l196_196810


namespace find_a_l196_196494

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) → 
    a^(2*x) + 2*a^x - 9 ≤ 6) ∧ 
  (∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ 
    a^(2*x) + 2*a^x - 9 = 6) → 
  a = 3 ∨ a = 1 / 3 :=
sorry

end find_a_l196_196494


namespace max_a_plus_b_min_a_squared_plus_b_squared_l196_196619

theorem max_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  a + b ≤ 2 := 
sorry

theorem min_a_squared_plus_b_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  2 ≤ a^2 + b^2 := 
sorry

end max_a_plus_b_min_a_squared_plus_b_squared_l196_196619


namespace Tom_search_cost_l196_196828

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end Tom_search_cost_l196_196828


namespace plates_not_adj_l196_196433

def num_ways_arrange_plates (blue red green orange : ℕ) (no_adj : Bool) : ℕ :=
  -- assuming this function calculates the desired number of arrangements
  sorry

theorem plates_not_adj (h : num_ways_arrange_plates 6 2 2 1 true = 1568) : 
  num_ways_arrange_plates 6 2 2 1 true = 1568 :=
  by exact h -- using the hypothesis directly for the theorem statement

end plates_not_adj_l196_196433


namespace abc_inequality_l196_196513

open Real

noncomputable def posReal (x : ℝ) : Prop := x > 0

theorem abc_inequality (a b c : ℝ) 
  (hCond1 : posReal a) 
  (hCond2 : posReal b) 
  (hCond3 : posReal c) 
  (hCond4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end abc_inequality_l196_196513


namespace gcd_of_polynomials_l196_196923

theorem gcd_of_polynomials (b : ℤ) (h : b % 1620 = 0) : Int.gcd (b^2 + 11 * b + 36) (b + 6) = 6 := 
by
  sorry

end gcd_of_polynomials_l196_196923


namespace find_A_time_l196_196536

noncomputable def work_rate_equations (W : ℝ) (A B C : ℝ) : Prop :=
  B + C = W / 2 ∧ A + B = W / 2 ∧ C = W / 3

theorem find_A_time {W A B C : ℝ} (h : work_rate_equations W A B C) :
  W / A = 3 :=
sorry

end find_A_time_l196_196536


namespace find_lost_card_number_l196_196397

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l196_196397


namespace moneyEarnedDuringHarvest_l196_196660

-- Define the weekly earnings, duration of harvest, and weekly rent.
def weeklyEarnings : ℕ := 403
def durationOfHarvest : ℕ := 233
def weeklyRent : ℕ := 49

-- Define total earnings and total rent.
def totalEarnings : ℕ := weeklyEarnings * durationOfHarvest
def totalRent : ℕ := weeklyRent * durationOfHarvest

-- Calculate the money earned after rent.
def moneyEarnedAfterRent : ℕ := totalEarnings - totalRent

-- The theorem to prove.
theorem moneyEarnedDuringHarvest : moneyEarnedAfterRent = 82482 :=
  by
  sorry

end moneyEarnedDuringHarvest_l196_196660


namespace ratio_neha_mother_age_12_years_ago_l196_196967

variables (N : ℕ) (M : ℕ) (X : ℕ)

theorem ratio_neha_mother_age_12_years_ago 
  (hM : M = 60)
  (h_future : M + 12 = 2 * (N + 12)) :
  (12 : ℕ) * (M - 12) = (48 : ℕ) * (N - 12) :=
by
  sorry

end ratio_neha_mother_age_12_years_ago_l196_196967


namespace number_increase_when_reversed_l196_196861

theorem number_increase_when_reversed :
  let n := 253
  let reversed_n := 352
  reversed_n - n = 99 :=
by
  let n := 253
  let reversed_n := 352
  sorry

end number_increase_when_reversed_l196_196861


namespace sequence_problem_l196_196755

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n - 1) = a 1 - a 0

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b n * b (n - 1) = b 1 * b 0

theorem sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : a 0 = -9) (ha1 : a 3 = -1) (ha_seq : arithmetic_sequence a)
  (hb : b 0 = -9) (hb4 : b 4 = -1) (hb_seq : geometric_sequence b) :
  b 2 * (a 2 - a 1) = -8 :=
sorry

end sequence_problem_l196_196755


namespace remaining_amount_to_be_paid_is_1080_l196_196067

noncomputable def deposit : ℕ := 120
noncomputable def total_price : ℕ := 10 * deposit
noncomputable def remaining_amount : ℕ := total_price - deposit

theorem remaining_amount_to_be_paid_is_1080 :
  remaining_amount = 1080 :=
by
  sorry

end remaining_amount_to_be_paid_is_1080_l196_196067


namespace probability_A_seventh_week_l196_196856

/-
Conditions:
1. There are four different passwords: A, B, C, and D.
2. Each week, one of these passwords is used.
3. Each week, the password is chosen at random and equally likely from the three passwords that were not used in the previous week.
4. Password A is used in the first week.

Goal:
Prove that the probability that password A will be used in the seventh week is 61/243.
-/

def prob_password_A_in_seventh_week : ℚ :=
  let Pk (k : ℕ) : ℚ := 
    if k = 1 then 1
    else if k >= 2 then ((-1 / 3)^(k - 1) * (3 / 4) + 1 / 4) else 0
  Pk 7

theorem probability_A_seventh_week : prob_password_A_in_seventh_week = 61 / 243 := by
  sorry

end probability_A_seventh_week_l196_196856


namespace ellipse_min_area_contains_circles_l196_196452

-- Define the ellipse and circles
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def circle1 (x y : ℝ) := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) := ((x + 2)^2 + y^2 = 4)

-- Proof statement: The smallest possible area of the ellipse containing the circles
theorem ellipse_min_area_contains_circles : 
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), 
    (circle1 x y → ellipse x y) ∧ 
    (circle2 x y → ellipse x y)) ∧
  (k = 12) := 
sorry

end ellipse_min_area_contains_circles_l196_196452


namespace question_statement_l196_196645

def line := Type
def plane := Type

-- Definitions for line lying in plane and planes being parallel 
def isIn (a : line) (α : plane) : Prop := sorry
def isParallel (α β : plane) : Prop := sorry
def isParallelLinePlane (a : line) (β : plane) : Prop := sorry

-- Conditions 
variables (a b : line) (α β : plane) 
variable (distinct_lines : a ≠ b)
variable (distinct_planes : α ≠ β)

-- Main statement to prove
theorem question_statement (h_parallel_planes : isParallel α β) (h_line_in_plane : isIn a α) : isParallelLinePlane a β := 
sorry

end question_statement_l196_196645


namespace find_principal_l196_196568

variable (P R : ℝ)
variable (condition1 : P + (P * R * 2) / 100 = 660)
variable (condition2 : P + (P * R * 7) / 100 = 1020)

theorem find_principal : P = 516 := by
  sorry

end find_principal_l196_196568


namespace gcd_solution_l196_196177

noncomputable def gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : ℤ :=
  Int.gcd (4 * b^2 + 35 * b + 72) (3 * b + 8)

theorem gcd_solution (b : ℤ) (h : ∃ k : ℤ, b = 2 * 5171 * k) : gcd_problem b h = 2 :=
by
  sorry

end gcd_solution_l196_196177


namespace horner_method_V3_correct_when_x_equals_2_l196_196243

-- Polynomial f(x)
noncomputable def f (x : ℝ) : ℝ :=
  2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

-- Horner's method for evaluating f(x)
noncomputable def V3 (x : ℝ) : ℝ :=
  (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Proof that V3 = 12 when x = 2
theorem horner_method_V3_correct_when_x_equals_2 : V3 2 = 12 := by
  sorry

end horner_method_V3_correct_when_x_equals_2_l196_196243


namespace brick_height_l196_196424

theorem brick_height (length width : ℕ) (num_bricks : ℕ) (wall_length wall_width wall_height : ℕ) (h : ℕ) :
  length = 20 ∧ width = 10 ∧ num_bricks = 25000 ∧ wall_length = 2500 ∧ wall_width = 200 ∧ wall_height = 75 ∧
  ( 20 * 10 * h = (wall_length * wall_width * wall_height) / 25000 ) -> 
  h = 75 :=
by
  sorry

end brick_height_l196_196424


namespace probability_red_ball_l196_196713

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l196_196713


namespace center_of_symmetry_l196_196245

theorem center_of_symmetry (k : ℤ) : ∀ (k : ℤ), ∃ x : ℝ, 
  (x = (k * Real.pi / 6 - Real.pi / 9) ∨ x = - (Real.pi / 18)) → False :=
by
  sorry

end center_of_symmetry_l196_196245


namespace train_cross_first_platform_l196_196258

noncomputable def time_to_cross_first_platform (L_t L_p1 L_p2 t2 : ℕ) : ℕ :=
  (L_t + L_p1) / ((L_t + L_p2) / t2)

theorem train_cross_first_platform :
  time_to_cross_first_platform 100 200 300 20 = 15 :=
by
  sorry

end train_cross_first_platform_l196_196258


namespace find_a_l196_196100

theorem find_a (a : ℝ) (h : 6 * a + 4 = 0) : a = -2 / 3 :=
by
  sorry

end find_a_l196_196100


namespace sum_distances_eq_6sqrt2_l196_196774

-- Define the curves C1 and C2 in Cartesian coordinates
def curve_C1 := { p : ℝ × ℝ | p.1 + p.2 = 3 }
def curve_C2 := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }

-- Defining the point P in ℝ²
def point_P : ℝ × ℝ := (1, 2)

-- Find the sum of distances |PA| + |PB|
theorem sum_distances_eq_6sqrt2 : 
  ∃ A B : ℝ × ℝ, A ∈ curve_C1 ∧ A ∈ curve_C2 ∧ 
                B ∈ curve_C1 ∧ B ∈ curve_C2 ∧ 
                (dist point_P A) + (dist point_P B) = 6 * Real.sqrt 2 := 
sorry

end sum_distances_eq_6sqrt2_l196_196774


namespace smallest_x_multiple_of_53_l196_196833

theorem smallest_x_multiple_of_53 :
  ∃ x : ℕ, (3 * x + 41) % 53 = 0 ∧ x > 0 ∧ x = 4 :=
by 
  sorry

end smallest_x_multiple_of_53_l196_196833


namespace largest_integer_a_l196_196288

theorem largest_integer_a (x a : ℤ) :
  ∃ x : ℤ, (x - a) * (x - 7) + 3 = 0 → a ≤ 11 :=
sorry

end largest_integer_a_l196_196288


namespace stephen_female_worker_ants_l196_196677

-- Define the conditions
def stephen_ants : ℕ := 110
def worker_ants (total_ants : ℕ) : ℕ := total_ants / 2
def male_worker_ants (workers : ℕ) : ℕ := (20 / 100) * workers

-- Define the question and correct answer
def female_worker_ants (total_ants : ℕ) : ℕ :=
  let workers := worker_ants total_ants
  workers - male_worker_ants workers

-- The theorem to prove
theorem stephen_female_worker_ants : female_worker_ants stephen_ants = 44 :=
  by sorry -- Skip the proof for now

end stephen_female_worker_ants_l196_196677


namespace curve_intersection_four_points_l196_196157

theorem curve_intersection_four_points (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 * a^2 ∧ y = a * x^2 - 2 * a) ∧ 
  (∃! (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧
    x1^2 + y1^2 = 4 * a^2 ∧ y1 = a * x1^2 - 2 * a ∧
    x2^2 + y2^2 = 4 * a^2 ∧ y2 = a * x2^2 - 2 * a ∧
    x3^2 + y3^2 = 4 * a^2 ∧ y3 = a * x3^2 - 2 * a ∧
    x4^2 + y4^2 = 4 * a^2 ∧ y4 = a * x4^2 - 2 * a) ↔ 
  a > 1 / 2 :=
by 
  sorry

end curve_intersection_four_points_l196_196157


namespace Kyle_papers_delivered_each_week_l196_196333

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l196_196333


namespace find_focus_with_larger_x_coordinate_l196_196453

noncomputable def focus_of_hyperbola_with_larger_x_coordinate : ℝ × ℝ :=
  let h := 5
  let k := 20
  let a := 7
  let b := 9
  let c := Real.sqrt (a^2 + b^2)
  (h + c, k)

theorem find_focus_with_larger_x_coordinate :
  focus_of_hyperbola_with_larger_x_coordinate = (5 + Real.sqrt 130, 20) := by
  sorry

end find_focus_with_larger_x_coordinate_l196_196453


namespace perpendicular_line_through_P_l196_196102

open Real

-- Define the point (1, 0)
def P : ℝ × ℝ := (1, 0)

-- Define the initial line x - 2y - 2 = 0
def initial_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the desired line 2x + y - 2 = 0
def desired_line (x y : ℝ) : Prop := 2 * x + y = 2

-- State that the desired line passes through the point (1, 0) and is perpendicular to the initial line
theorem perpendicular_line_through_P :
  (∃ m b, b ∈ Set.univ ∧ (∀ x y, desired_line x y → y = m * x + b)) ∧ ∀ x y, 
  initial_line x y → x ≠ 0 → desired_line y (-x / 2) :=
sorry

end perpendicular_line_through_P_l196_196102


namespace closest_point_on_line_l196_196040

theorem closest_point_on_line 
  (t : ℚ)
  (x y z : ℚ)
  (x_eq : x = 3 + t)
  (y_eq : y = 2 - 3 * t)
  (z_eq : z = -1 + 2 * t)
  (x_ortho_eq : (1 + t) = 0)
  (y_ortho_eq : (3 - 3 * t) = 0)
  (z_ortho_eq : (-3 + 2 * t) = 0) :
  (45/14, 16/14, -1/7) = (x, y, z) := by
  sorry

end closest_point_on_line_l196_196040


namespace ratio_volume_sphere_to_hemisphere_l196_196682

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (1/2) * volume_sphere r

theorem ratio_volume_sphere_to_hemisphere (p : ℝ) (hp : 0 < p) :
  (volume_sphere p) / (volume_hemisphere (2 * p)) = 1 / 4 :=
by
  sorry

end ratio_volume_sphere_to_hemisphere_l196_196682


namespace rank_from_right_l196_196437

theorem rank_from_right (n total rank_left : ℕ) (h1 : rank_left = 5) (h2 : total = 21) : n = total - (rank_left - 1) :=
by {
  sorry
}

end rank_from_right_l196_196437


namespace car_dealership_theorem_l196_196426

def car_dealership_problem : Prop :=
  let initial_cars := 100
  let new_shipment := 150
  let initial_silver_percentage := 0.20
  let new_silver_percentage := 0.40
  let initial_silver := initial_silver_percentage * initial_cars
  let new_silver := new_silver_percentage * new_shipment
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_shipment
  let silver_percentage := (total_silver / total_cars) * 100
  silver_percentage = 32

theorem car_dealership_theorem : car_dealership_problem :=
by {
  sorry
}

end car_dealership_theorem_l196_196426


namespace find_valid_pairs_l196_196122

-- Decalred the main definition for the problem.
def valid_pairs (x y : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99) ∧ ((x + y)^2 = 100 * x + y)

-- Stating the theorem without the proof.
theorem find_valid_pairs :
  valid_pairs 20 25 ∧ valid_pairs 30 25 :=
sorry

end find_valid_pairs_l196_196122


namespace completion_days_l196_196878

theorem completion_days (D : ℝ) :
  (1 / D + 1 / 9 = 1 / 3.2142857142857144) → D = 5 := by
  sorry

end completion_days_l196_196878


namespace express_inequality_l196_196733

theorem express_inequality (x : ℝ) : x + 4 ≥ -1 := sorry

end express_inequality_l196_196733


namespace max_plates_l196_196015

def cost_pan : ℕ := 3
def cost_pot : ℕ := 5
def cost_plate : ℕ := 11
def total_cost : ℕ := 100
def min_pans : ℕ := 2
def min_pots : ℕ := 2

theorem max_plates (p q r : ℕ) :
  p >= min_pans → q >= min_pots → (cost_pan * p + cost_pot * q + cost_plate * r = total_cost) → r = 7 :=
by
  intros h_p h_q h_cost
  sorry

end max_plates_l196_196015


namespace equation_of_line_AB_l196_196012

noncomputable def center_of_circle : (ℝ × ℝ) := (-4, -1)

noncomputable def point_P : (ℝ × ℝ) := (2, 3)

noncomputable def slope_OP : ℝ :=
  let (x₁, y₁) := center_of_circle
  let (x₂, y₂) := point_P
  (y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_AB : ℝ :=
  -1 / slope_OP

theorem equation_of_line_AB : (6 * x + 4 * y + 19 = 0) :=
  sorry

end equation_of_line_AB_l196_196012


namespace area_union_of_rectangle_and_circle_l196_196578

theorem area_union_of_rectangle_and_circle :
  let length := 12
  let width := 15
  let r := 15
  let area_rectangle := length * width
  let area_circle := Real.pi * r^2
  let area_overlap := (1/4) * area_circle
  let area_union := area_rectangle + area_circle - area_overlap
  area_union = 180 + 168.75 * Real.pi := by
    sorry

end area_union_of_rectangle_and_circle_l196_196578


namespace a7_of_arithmetic_seq_l196_196647

-- Defining the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

theorem a7_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : arithmetic_seq a d) 
  (h_a4 : a 4 = 5) 
  (h_a5_a6 : a 5 + a 6 = 11) : 
  a 7 = 6 :=
by
  sorry

end a7_of_arithmetic_seq_l196_196647


namespace avery_donation_l196_196021

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l196_196021


namespace gcd_qr_l196_196765

theorem gcd_qr (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 770) : Nat.gcd q r = 70 := sorry

end gcd_qr_l196_196765


namespace eq_expression_l196_196392

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l196_196392


namespace max_flowers_used_min_flowers_used_l196_196133

-- Part (a) Setup
def max_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def max_T (C M : ℕ) := C + M

-- Given conditions
theorem max_flowers_used :
  (∀ C M : ℕ, max_flowers C M 36 → max_T C M = 36) :=
by sorry

-- Part (b) Setup
def min_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def min_T (C M : ℕ) := C + M

-- Given conditions
theorem min_flowers_used :
  (∀ C M : ℕ, min_flowers C M 48 → min_T C M = 24) :=
by sorry

end max_flowers_used_min_flowers_used_l196_196133


namespace cameron_total_questions_answered_l196_196449

def questions_per_tourist : ℕ := 2
def group1_size : ℕ := 6
def group2_size : ℕ := 11
def group3_size_regular : ℕ := 7
def group3_inquisitive_size : ℕ := 1
def group4_size : ℕ := 7

theorem cameron_total_questions_answered :
  let group1_questions := questions_per_tourist * group1_size in
  let group2_questions := questions_per_tourist * group2_size in
  let group3_regular_questions := questions_per_tourist * group3_size_regular in
  let group3_inquisitive_questions := group3_inquisitive_size * (questions_per_tourist * 3) in
  let group3_questions := group3_regular_questions + group3_inquisitive_questions in
  let group4_questions := questions_per_tourist * group4_size in
  group1_questions + group2_questions + group3_questions + group4_questions = 68 :=
by
  sorry

end cameron_total_questions_answered_l196_196449


namespace exponentiation_problem_l196_196490

theorem exponentiation_problem (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 2^a * 2^b = 8) : (2^a)^b = 4 := 
sorry

end exponentiation_problem_l196_196490


namespace correct_sampling_method_l196_196140

structure SchoolPopulation :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)

-- Define the school population
def school : SchoolPopulation :=
  { senior := 10, intermediate := 50, junior := 75 }

-- Define the condition for sampling method
def total_school_teachers (s : SchoolPopulation) : ℕ :=
  s.senior + s.intermediate + s.junior

-- The desired sample size
def sample_size : ℕ := 30

-- The correct sampling method based on the population strata
def stratified_sampling (s : SchoolPopulation) : Prop :=
  s.senior + s.intermediate + s.junior > 0

theorem correct_sampling_method : stratified_sampling school :=
by { sorry }

end correct_sampling_method_l196_196140


namespace triangle_area_ABC_l196_196750

variable {A : Prod ℝ ℝ}
variable {B : Prod ℝ ℝ}
variable {C : Prod ℝ ℝ}

noncomputable def area_of_triangle (A B C : Prod ℝ ℝ ) : ℝ :=
  (1 / 2) * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))

theorem triangle_area_ABC : 
  ∀ {A B C : Prod ℝ ℝ}, 
  A = (2, 3) → 
  B = (5, 7) → 
  C = (6, 1) → 
  area_of_triangle A B C = 11 
:= by
  intros
  subst_vars
  simp [area_of_triangle]
  sorry

end triangle_area_ABC_l196_196750


namespace min_value_expression_l196_196294

theorem min_value_expression (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 21) ∧ 
           (∀ z : ℝ, (z = (x + 18) / Real.sqrt (x - 3)) → y ≤ z) := 
sorry

end min_value_expression_l196_196294


namespace three_digit_numbers_divisible_by_17_l196_196063

theorem three_digit_numbers_divisible_by_17 : ∃ (n : ℕ), n = 53 ∧ ∀ k, 100 <= 17 * k ∧ 17 * k <= 999 ↔ (6 <= k ∧ k <= 58) :=
by
  sorry

end three_digit_numbers_divisible_by_17_l196_196063


namespace yuri_lost_card_l196_196414

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l196_196414


namespace number_of_integer_pairs_satisfying_equation_l196_196605

theorem number_of_integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | let x := p.1, y := p.2 in 5 * x^2 - 6 * x * y + y^2 = 6^100 }.to_finset.card = 19594 := sorry

end number_of_integer_pairs_satisfying_equation_l196_196605


namespace relay_go_match_outcomes_l196_196354

theorem relay_go_match_outcomes : (Nat.choose 14 7) = 3432 := by
  sorry

end relay_go_match_outcomes_l196_196354


namespace required_bricks_l196_196246

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

theorem required_bricks : 
  let brick_length := 25
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 850
  let wall_width := 600
  let wall_height := 22.5
  (wall_volume wall_length wall_width wall_height) / 
  (brick_volume brick_length brick_width brick_height) = 6800 :=
by
  sorry

end required_bricks_l196_196246


namespace margaret_age_in_12_years_l196_196267

theorem margaret_age_in_12_years
  (brian_age : ℝ)
  (christian_age : ℝ)
  (margaret_age : ℝ)
  (h1 : christian_age = 3.5 * brian_age)
  (h2 : brian_age + 12 = 45)
  (h3 : margaret_age = christian_age - 10) :
  margaret_age + 12 = 117.5 :=
by
  sorry

end margaret_age_in_12_years_l196_196267


namespace common_rational_root_is_negative_non_integer_l196_196269

theorem common_rational_root_is_negative_non_integer 
    (a b c d e f g : ℤ)
    (p : ℚ)
    (h1 : 90 * p^4 + a * p^3 + b * p^2 + c * p + 15 = 0)
    (h2 : 15 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 90 = 0)
    (h3 : ¬ (∃ k : ℤ, p = k))
    (h4 : p < 0) : 
  p = -1 / 3 := 
sorry

end common_rational_root_is_negative_non_integer_l196_196269


namespace ratio_of_efficacy_l196_196262

-- Define original conditions
def original_sprigs_of_mint := 3
def green_tea_leaves_per_sprig := 2

-- Define new condition
def new_green_tea_leaves := 12

-- Calculate the number of sprigs of mint corresponding to the new green tea leaves in the new mud
def new_sprigs_of_mint := new_green_tea_leaves / green_tea_leaves_per_sprig

-- Statement of the theorem: ratio of the efficacy of new mud to original mud is 1:2
theorem ratio_of_efficacy : new_sprigs_of_mint = 2 * original_sprigs_of_mint :=
by
    sorry

end ratio_of_efficacy_l196_196262


namespace total_fish_l196_196216

-- Definition of the number of fish Lilly has
def lilly_fish : Nat := 10

-- Definition of the number of fish Rosy has
def rosy_fish : Nat := 8

-- Statement to prove
theorem total_fish : lilly_fish + rosy_fish = 18 := 
by
  -- The proof is omitted
  sorry

end total_fish_l196_196216


namespace scarves_per_box_l196_196154

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothes : ℕ)
  (h1 : boxes = 4)
  (h2 : mittens_per_box = 6)
  (h3 : total_clothes = 32)
  (total_mittens := boxes * mittens_per_box)
  (total_scarves := total_clothes - total_mittens) :
  total_scarves / boxes = 2 :=
by
  sorry

end scarves_per_box_l196_196154


namespace john_sixth_quiz_score_l196_196507

noncomputable def sixth_quiz_score_needed : ℤ :=
  let scores := [86, 91, 88, 84, 97]
  let desired_average := 95
  let number_of_quizzes := 6
  let total_score_needed := number_of_quizzes * desired_average
  let total_score_so_far := scores.sum
  total_score_needed - total_score_so_far

theorem john_sixth_quiz_score :
  sixth_quiz_score_needed = 124 := 
by
  sorry

end john_sixth_quiz_score_l196_196507


namespace third_angle_is_90_triangle_is_right_l196_196617

-- Define the given angles
def angle1 : ℝ := 56
def angle2 : ℝ := 34

-- Define the sum of angles in a triangle
def angle_sum : ℝ := 180

-- Define the third angle
def third_angle : ℝ := angle_sum - angle1 - angle2

-- Prove that the third angle is 90 degrees
theorem third_angle_is_90 : third_angle = 90 := by
  sorry

-- Define the type of the triangle based on the largest angle
def is_right_triangle : Prop := third_angle = 90

-- Prove that the triangle is a right triangle
theorem triangle_is_right : is_right_triangle := by
  sorry

end third_angle_is_90_triangle_is_right_l196_196617


namespace solution_sum_of_eq_zero_l196_196837

open Real

theorem solution_sum_of_eq_zero : 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  in (∀ x, f x = 0 → x = -3/2 ∨ x = 8/3) → 
     (-3/2 + 8/3 = 7/6) :=
by 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  intro h
  have h₁ : f(-3/2) = 0 := by sorry
  have h₂ : f(8/3) = 0 := by sorry
  have sum_of_solutions : -3/2 + 8/3 = 7/6 := by 
    sorry
  exact sum_of_solutions

end solution_sum_of_eq_zero_l196_196837


namespace base_conversion_and_addition_l196_196883

def C : ℕ := 12

def base9_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 9^2) + (d1 * 9^1) + (d0 * 9^0)

def base13_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 13^2) + (d1 * 13^1) + (d0 * 13^0)

def num1 := base9_to_nat 7 5 2
def num2 := base13_to_nat 6 C 3

theorem base_conversion_and_addition :
  num1 + num2 = 1787 :=
by
  sorry

end base_conversion_and_addition_l196_196883


namespace number_of_days_l196_196202

theorem number_of_days (d : ℝ) (h : 2 * d = 1.5 * d + 3) : d = 6 :=
by
  sorry

end number_of_days_l196_196202


namespace cube_root_of_neg8_l196_196978

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end cube_root_of_neg8_l196_196978


namespace kyle_paper_delivery_l196_196335

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l196_196335


namespace decompose_375_l196_196572

theorem decompose_375 : 375 = 3 * 100 + 7 * 10 + 5 * 1 :=
by
  sorry

end decompose_375_l196_196572


namespace find_2019th_integer_not_divisible_by_5_l196_196737

/-- Function to calculate the 5-adic valuation of a binomial coefficient -/
def binomial_5_adic_valuation (n : ℕ) : ℕ :=
  PadicVal.eval 5 (Nat.choose (2 * n) n)

/-- Find the 2019th positive integer n such that the binomial coefficient 
  (2n choose n) is not divisible by 5. -/
theorem find_2019th_integer_not_divisible_by_5 : 
  ∃ n : ℕ, (∀ i < 2019, ∃ m : ℕ, m > 0 ∧ binomial_5_adic_valuation m = 0 ∧ m < n) 
  ∧ binomial_5_adic_valuation n = 0 ∧ n = 37805 :=
sorry

end find_2019th_integer_not_divisible_by_5_l196_196737


namespace b7_value_l196_196786

theorem b7_value (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h₀a : a 0 = 3) (h₀b : b 0 = 4)
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 / b n)
  (h₂ : ∀ n, b (n + 1) = b n ^ 2 / a n) :
  b 7 = 4 ^ 730 / 3 ^ 1093 :=
by
  sorry

end b7_value_l196_196786


namespace forty_percent_of_number_l196_196418

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : (40/100) * N = 192 :=
by
  sorry

end forty_percent_of_number_l196_196418


namespace a2_add_a8_l196_196949

variable (a : ℕ → ℝ) -- a_n is an arithmetic sequence
variable (d : ℝ) -- common difference

-- Condition stating that a_n is an arithmetic sequence with common difference d
axiom arithmetic_sequence : ∀ n, a (n + 1) = a n + d

-- Given condition a_3 + a_4 + a_5 + a_6 + a_7 = 450
axiom given_condition : a 3 + a 4 + a 5 + a 6 + a 7 = 450

theorem a2_add_a8 : a 2 + a 8 = 180 :=
by
  sorry

end a2_add_a8_l196_196949


namespace find_m_range_l196_196290

noncomputable def quadratic_inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem find_m_range :
  { m : ℝ | quadratic_inequality_condition m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
sorry

end find_m_range_l196_196290


namespace inequality_proof_l196_196673

theorem inequality_proof (a : ℝ) : 
  2 * a^4 + 2 * a^2 - 1 ≥ (3 / 2) * (a^2 + a - 1) :=
by
  sorry

end inequality_proof_l196_196673


namespace Compute_fraction_power_l196_196451

theorem Compute_fraction_power :
  (81081 / 27027) ^ 4 = 81 :=
by
  -- We provide the specific condition as part of the proof statement
  have h : 27027 * 3 = 81081 := by norm_num
  sorry

end Compute_fraction_power_l196_196451


namespace price_per_working_game_l196_196791

theorem price_per_working_game 
  (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 16) (h2 : non_working_games = 8) (h3 : total_earnings = 56) :
  total_earnings / (total_games - non_working_games) = 7 :=
by {
  sorry
}

end price_per_working_game_l196_196791


namespace geometric_progression_fourth_term_l196_196637

theorem geometric_progression_fourth_term (x : ℚ)
  (h : (3 * x + 3) / x = (5 * x + 5) / (3 * x + 3)) :
  (5 / 3) * (5 * x + 5) = -125/12 :=
by
  sorry

end geometric_progression_fourth_term_l196_196637


namespace circle_equation_equivalence_l196_196457

theorem circle_equation_equivalence 
    (x y : ℝ) : 
    x^2 + y^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 :=
sorry

end circle_equation_equivalence_l196_196457


namespace total_distance_covered_l196_196567

theorem total_distance_covered :
  let t1 := 30 / 60 -- time in hours for first walking session
  let s1 := 3       -- speed in mph for first walking session
  let t2 := 20 / 60 -- time in hours for running session
  let s2 := 8       -- speed in mph for running session
  let t3 := 10 / 60 -- time in hours for second walking session
  let s3 := 2       -- speed in mph for second walking session
  let d1 := s1 * t1 -- distance for first walking session
  let d2 := s2 * t2 -- distance for running session
  let d3 := s3 * t3 -- distance for second walking session
  d1 + d2 + d3 = 4.5 :=
by
  sorry

end total_distance_covered_l196_196567


namespace max_sum_of_integer_pairs_on_circle_l196_196375

theorem max_sum_of_integer_pairs_on_circle : 
  ∃ (x y : ℤ), x^2 + y^2 = 169 ∧ ∀ (a b : ℤ), a^2 + b^2 = 169 → x + y ≥ a + b :=
sorry

end max_sum_of_integer_pairs_on_circle_l196_196375


namespace repeated_root_cubic_l196_196470

theorem repeated_root_cubic (p : ℝ) :
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ (9 * x^2 - 2 * (p + 1) * x + 4 = 0)) →
  (p = 5 ∨ p = -7) :=
by
  sorry

end repeated_root_cubic_l196_196470


namespace product_gcf_lcm_l196_196342

def gcf (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : Nat) : Nat := Nat.lcm (Nat.lcm a b) c

theorem product_gcf_lcm :
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  A * B = 432 :=
by
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  have hA : A = Nat.gcd (Nat.gcd 6 18) 24 := rfl
  have hB : B = Nat.lcm (Nat.lcm 6 18) 24 := rfl
  sorry

end product_gcf_lcm_l196_196342


namespace value_of_expression_l196_196940

theorem value_of_expression (a b : ℝ) (h : -3 * a - b = -1) : 3 - 6 * a - 2 * b = 1 :=
by
  sorry

end value_of_expression_l196_196940


namespace max_y_coordinate_is_three_fourths_l196_196467

noncomputable def max_y_coordinate : ℝ :=
  let y k := 3 * k^2 - 4 * k^4 in 
  y (Real.sqrt (3 / 8))

theorem max_y_coordinate_is_three_fourths:
  max_y_coordinate = 3 / 4 := 
by 
  sorry

end max_y_coordinate_is_three_fourths_l196_196467


namespace no_solution_l196_196890

theorem no_solution (x : ℝ) (h₁ : x ≠ -1/3) (h₂ : x ≠ -4/5) :
  (2 * x - 4) / (3 * x + 1) ≠ (2 * x - 10) / (5 * x + 4) := 
sorry

end no_solution_l196_196890


namespace arithmetic_sequence_8th_term_l196_196562

theorem arithmetic_sequence_8th_term :
  ∃ (a1 a15 n : ℕ) (d a8 : ℝ),
  a1 = 3 ∧ a15 = 48 ∧ n = 15 ∧
  d = (a15 - a1) / (n - 1) ∧
  a8 = a1 + 7 * d ∧
  a8 = 25.5 :=
by
  sorry

end arithmetic_sequence_8th_term_l196_196562


namespace puzzles_and_board_games_count_l196_196817

def num_toys : ℕ := 200
def num_action_figures : ℕ := num_toys / 4
def num_dolls : ℕ := num_toys / 3

theorem puzzles_and_board_games_count :
  num_toys - num_action_figures - num_dolls = 84 := 
  by
    -- TODO: Prove this theorem
    sorry

end puzzles_and_board_games_count_l196_196817


namespace find_a9_l196_196323

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions of the problem
variables {a : ℕ → ℝ}
axiom h_geom_seq : is_geometric_sequence a
axiom h_root1 : a 3 * a 15 = 1
axiom h_root2 : a 3 + a 15 = -4

-- The proof statement
theorem find_a9 : a 9 = 1 := 
by sorry

end find_a9_l196_196323


namespace measure_of_angle_C_l196_196953

variable (A B C : ℕ)

theorem measure_of_angle_C :
  (A = B - 20) →
  (C = A + 40) →
  (A + B + C = 180) →
  C = 80 :=
by
  intros h1 h2 h3
  sorry

end measure_of_angle_C_l196_196953


namespace min_value_fraction_l196_196743

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 19) ∧ (∀ z : ℝ, (z = (x + 15) / Real.sqrt (x - 4)) → z ≥ y) :=
by
  sorry

end min_value_fraction_l196_196743


namespace min_polyline_distance_between_circle_and_line_l196_196646

def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

def on_line (Q : ℝ × ℝ) : Prop :=
  2 * Q.1 + Q.2 = 2 * Real.sqrt 5

theorem min_polyline_distance_between_circle_and_line :
  ∃ P Q, on_circle P ∧ on_line Q ∧ polyline_distance P Q = (Real.sqrt 5) / 2 :=
by
  sorry

end min_polyline_distance_between_circle_and_line_l196_196646


namespace average_first_19_natural_numbers_l196_196697

theorem average_first_19_natural_numbers : 
  (1 + 19) / 2 = 10 := 
by 
  sorry

end average_first_19_natural_numbers_l196_196697


namespace Kyle_papers_delivered_each_week_l196_196332

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l196_196332


namespace solve_for_x_l196_196767

variable {a b c x : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3)

theorem solve_for_x (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c :=
sorry

end solve_for_x_l196_196767


namespace unique_ordered_triple_l196_196509

theorem unique_ordered_triple (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ab : Nat.lcm a b = 500) (h_bc : Nat.lcm b c = 2000) (h_ca : Nat.lcm c a = 2000) :
  (a = 100 ∧ b = 2000 ∧ c = 2000) :=
by
  sorry

end unique_ordered_triple_l196_196509


namespace tangent_line_eqn_extreme_values_l196_196481

/-- The tangent line to the function f at (0, 5) -/
theorem tangent_line_eqn (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ k b, (∀ x, f x = k * x + b) ∧ k = -2 ∧ b = 5) ∧ (2 * 0 + 5 - 5 = 0) := by
  sorry

/-- The function f has a local maximum at x = -1 and a local minimum at x = 2 -/
theorem extreme_values (f : ℝ → ℝ) (h : ∀ x, f x = (1 / 3) * x ^ 3 - (1 / 2) * x ^ 2 - 2 * x + 5) :
  (∃ x₁ x₂, x₁ = -1 ∧ f x₁ = 37 / 6 ∧ x₂ = 2 ∧ f x₂ = 5 / 3) := by
  sorry

end tangent_line_eqn_extreme_values_l196_196481


namespace Ginger_sold_10_lilacs_l196_196046

variable (R L G : ℕ)

def condition1 := R = 3 * L
def condition2 := G = L / 2
def condition3 := L + R + G = 45

theorem Ginger_sold_10_lilacs
    (h1 : condition1 R L)
    (h2 : condition2 G L)
    (h3 : condition3 L R G) :
  L = 10 := 
  sorry

end Ginger_sold_10_lilacs_l196_196046


namespace solve_first_equation_solve_second_equation_l196_196741

theorem solve_first_equation (x : ℝ) : 3 * (x - 2)^2 - 27 = 0 ↔ x = 5 ∨ x = -1 :=
by {
  sorry
}

theorem solve_second_equation (x : ℝ) : 2 * (x + 1)^3 + 54 = 0 ↔ x = -4 :=
by {
  sorry
}

end solve_first_equation_solve_second_equation_l196_196741


namespace sara_staircase_steps_l196_196770

-- Define the problem statement and conditions
theorem sara_staircase_steps (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → n = 12 := 
by
  intro h
  sorry

end sara_staircase_steps_l196_196770


namespace conor_chop_eggplants_l196_196721

theorem conor_chop_eggplants (E : ℕ) 
  (condition1 : E + 9 + 8 = (E + 17))
  (condition2 : 4 * (E + 9 + 8) = 116) :
  E = 12 :=
by {
  sorry
}

end conor_chop_eggplants_l196_196721


namespace surface_area_of_given_cube_l196_196548

-- Define the edge length condition
def edge_length_of_cube (sum_edge_lengths : ℕ) :=
  sum_edge_lengths / 12

-- Define the surface area of a cube given an edge length
def surface_area_of_cube (edge_length : ℕ) :=
  6 * (edge_length * edge_length)

-- State the theorem
theorem surface_area_of_given_cube : 
  edge_length_of_cube 36 = 3 ∧ surface_area_of_cube 3 = 54 :=
by
  -- We leave the proof as an exercise.
  sorry

end surface_area_of_given_cube_l196_196548


namespace train_length_l196_196708

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_ms : ℝ) (length_m : ℝ)
  (h1 : speed_kmph = 120) 
  (h2 : time_sec = 6)
  (h3 : speed_ms = 33.33)
  (h4 : length_m = 200) : 
  speed_kmph * 1000 / 3600 * time_sec = length_m :=
by
  sorry

end train_length_l196_196708


namespace find_biology_marks_l196_196030

variable (english mathematics physics chemistry average_marks : ℕ)

theorem find_biology_marks
  (h_english : english = 86)
  (h_mathematics : mathematics = 85)
  (h_physics : physics = 92)
  (h_chemistry : chemistry = 87)
  (h_average_marks : average_marks = 89) : 
  (english + mathematics + physics + chemistry + (445 - (english + mathematics + physics + chemistry))) / 5 = average_marks :=
by
  sorry

end find_biology_marks_l196_196030


namespace six_power_six_div_two_l196_196417

theorem six_power_six_div_two : 6 ^ (6 / 2) = 216 := by
  sorry

end six_power_six_div_two_l196_196417


namespace largest_possible_k_l196_196988

open Finset

-- Define the set X with 1983 elements
def X : Finset ℕ := range 1983

-- Define the family of subsets
variable {S : ℕ → Finset ℕ}

-- State the conditions as hypotheses
variable (h1 : ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → (S i) ∪ (S j) ∪ (S k) = X)
variable (h2 : ∀ (i j : ℕ), i ≠ j → ((S i) ∪ (S j)).card ≤ 1979)

-- Define the proof problem
theorem largest_possible_k : ∃ k : ℕ, (∀ (S : Finset ℕ → Type), 
  (∀ i, (S i).subset X) → 
  (∀ i j k, i ≠ j → j ≠ k → k ≠ i → (S i) ∪ (S j) ∪ (S k) = X) → 
  (∀ i j, i ≠ j → ((S i) ∪ (S j)).card ≤ 1979) → 
  k ≤ 31 ∧ ¬(k + 1 ≤ 31) ) :=
sorry

end largest_possible_k_l196_196988


namespace movies_shown_eq_twenty_four_l196_196007

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l196_196007


namespace max_min_f_l196_196782

-- Defining a and the set A
def a : ℤ := 2001

def A : Set (ℤ × ℤ) := {p | p.snd ≠ 0 ∧ p.fst < 2 * a ∧ (2 * p.snd) ∣ ((2 * a * p.fst) - (p.fst * p.fst) + (p.snd * p.snd)) ∧ ((p.snd * p.snd) - (p.fst * p.fst) + (2 * p.fst * p.snd) ≤ (2 * a * (p.snd - p.fst)))}

-- Defining the function f
def f (m n : ℤ): ℤ := (2 * a * m - m * m - m * n) / n

-- Main theorem: Proving that the maximum and minimum values of f over A are 3750 and 2 respectively
theorem max_min_f : 
  ∃ p ∈ A, f p.fst p.snd = 3750 ∧
  ∃ q ∈ A, f q.fst q.snd = 2 :=
sorry

end max_min_f_l196_196782


namespace complement_intersection_l196_196485

-- Definitions of sets and complements
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}
def C_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}
def C_U_B : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- The proof statement
theorem complement_intersection {U A B C_U_A C_U_B : Set ℕ} (h1 : U = {1, 2, 3, 4, 5}) (h2 : A = {1, 2, 3}) (h3 : B = {2, 5}) (h4 : C_U_A = {x | x ∈ U ∧ x ∉ A}) (h5 : C_U_B = {x | x ∈ U ∧ x ∉ B}) : 
  (C_U_A ∩ C_U_B) = {4} :=
by 
  sorry

end complement_intersection_l196_196485


namespace proof_problem_l196_196293

theorem proof_problem (a b A B : ℝ) (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (h_f_def : ∀ θ : ℝ, f θ = 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end proof_problem_l196_196293


namespace bound_on_points_l196_196197

variables (k n : ℕ) (A : Fin k → Fin k → ℕ)
          (C : Fin k → Fin k → ℕ → ℕ → Prop)

open Finset

-- Define conditions
def outgoing (i j : Fin k) : Prop := (i.val < j.val)
def incoming (i j : Fin k) : Prop := (i.val > j.val)

-- Condition that the color of all outgoing lines for A_i are different from that of incoming lines for A_i
def valid_coloring (A : Fin k → Fin k → ℕ) (C : Fin k → Fin k → ℕ → ℕ → Prop) : Prop :=
  ∀ i : Fin k, 
    let out_colors := {c | ∃ j : Fin k, outgoing i j ∧ C i j c A i j} in
    let in_colors := {c | ∃ j : Fin k, incoming i j ∧ C j i c A j i} in
    out_colors ∩ in_colors = ∅

theorem bound_on_points (A : Fin k → Fin k → ℕ) (C : Fin k → Fin k → ℕ → ℕ → Prop) (h : valid_coloring A C) :
  k ≤ 2^n :=
sorry

end bound_on_points_l196_196197


namespace cost_of_three_tshirts_l196_196581

-- Defining the conditions
def saving_per_tshirt : ℝ := 5.50
def full_price_per_tshirt : ℝ := 16.50
def number_of_tshirts : ℕ := 3
def number_of_paid_tshirts : ℕ := 2

-- Statement of the problem
theorem cost_of_three_tshirts :
  (number_of_paid_tshirts * full_price_per_tshirt) = 33 := 
by
  -- Proof steps go here (using sorry as a placeholder)
  sorry

end cost_of_three_tshirts_l196_196581


namespace Bran_remaining_payment_l196_196026

theorem Bran_remaining_payment :
  let tuition_fee : ℝ := 90
  let job_income_per_month : ℝ := 15
  let scholarship_percentage : ℝ := 0.30
  let months : ℕ := 3
  let scholarship_amount : ℝ := tuition_fee * scholarship_percentage
  let remaining_after_scholarship : ℝ := tuition_fee - scholarship_amount
  let total_job_income : ℝ := job_income_per_month * months
  let amount_to_pay : ℝ := remaining_after_scholarship - total_job_income
  amount_to_pay = 18 := sorry

end Bran_remaining_payment_l196_196026


namespace find_f_neg2_l196_196473

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a * f b
axiom f_pos (x : ℝ) : f x > 0
axiom f_one : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 :=
by {
  sorry
}

end find_f_neg2_l196_196473


namespace intersection_M_N_l196_196904

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l196_196904


namespace infinite_integer_and_noninteger_terms_l196_196343

theorem infinite_integer_and_noninteger_terms (m : Nat) (h_m : m > 0) :
  ∃ (infinite_int_terms : Nat → Prop) (infinite_nonint_terms : Nat → Prop),
  (∀ n, ∃ k, infinite_int_terms k ∧ ∀ k, infinite_int_terms k → ∃ N, k = n + N + 1) ∧
  (∀ n, ∃ k, infinite_nonint_terms k ∧ ∀ k, infinite_nonint_terms k → ∃ N, k = n + N + 1) :=
sorry

end infinite_integer_and_noninteger_terms_l196_196343


namespace probability_three_female_finalists_l196_196643

open Finset

noncomputable def probability_all_female (total : ℕ) (females : ℕ) (selected : ℕ) : ℚ :=
  (females.choose selected : ℚ) / (total.choose selected : ℚ)

theorem probability_three_female_finalists :
  probability_all_female 7 4 3 = 4 / 35 :=
by
  sorry

end probability_three_female_finalists_l196_196643


namespace percentage_saved_is_25_l196_196256

def monthly_salary : ℝ := 1000

def increase_percentage : ℝ := 0.10

def saved_amount_after_increase : ℝ := 175

def calculate_percentage_saved (x : ℝ) : Prop := 
  1000 - (1000 - (x / 100) * monthly_salary) * (1 + increase_percentage) = saved_amount_after_increase

theorem percentage_saved_is_25 :
  ∃ x : ℝ, x = 25 ∧ calculate_percentage_saved x :=
sorry

end percentage_saved_is_25_l196_196256


namespace dartboard_area_ratio_l196_196707

theorem dartboard_area_ratio
    (larger_square_side_length : ℝ)
    (inner_square_side_length : ℝ)
    (angle_division : ℝ)
    (s : ℝ)
    (p : ℝ)
    (h1 : larger_square_side_length = 4)
    (h2 : inner_square_side_length = 2)
    (h3 : angle_division = 45)
    (h4 : s = 1/4)
    (h5 : p = 3) :
    p / s = 12 :=
by
    sorry

end dartboard_area_ratio_l196_196707


namespace shaded_region_area_l196_196113

theorem shaded_region_area
  (n : ℕ) (d : ℝ) 
  (h₁ : n = 25) 
  (h₂ : d = 10) 
  (h₃ : n > 0) : 
  (d^2 / n = 2) ∧ (n * (d^2 / (2 * n)) = 50) :=
by 
  sorry

end shaded_region_area_l196_196113


namespace insphere_radius_of_tetrahedron_l196_196164

noncomputable def insphere_radius (H α : ℝ) : ℝ := 
  H / (4 * tan(α)^2) * (sqrt (4 * tan(α)^2 + 1) - 1)

theorem insphere_radius_of_tetrahedron (H α : ℝ) 
  (hH : 0 < H) (hα : 0 < α ∧ α < real.pi / 2) :
  insphere_radius H α = 
    (H / (4 * tan α ^ 2)) * ((sqrt (4 * tan α ^ 2 + 1)) - 1) :=
by 
  sorry

end insphere_radius_of_tetrahedron_l196_196164


namespace office_needs_24_pencils_l196_196862

noncomputable def number_of_pencils (total_cost : ℝ) (cost_per_pencil : ℝ) (cost_per_folder : ℝ) (number_of_folders : ℕ) : ℝ :=
  (total_cost - (number_of_folders * cost_per_folder)) / cost_per_pencil

theorem office_needs_24_pencils :
  number_of_pencils 30 0.5 0.9 20 = 24 :=
by
  sorry

end office_needs_24_pencils_l196_196862


namespace yards_mowed_by_christian_l196_196028

-- Definitions based on the provided conditions
def initial_savings := 5 + 7
def sue_earnings := 6 * 2
def total_savings := initial_savings + sue_earnings
def additional_needed := 50 - total_savings
def short_amount := 6
def christian_earnings := additional_needed - short_amount
def charge_per_yard := 5

theorem yards_mowed_by_christian : 
  (christian_earnings / charge_per_yard) = 4 :=
by
  sorry

end yards_mowed_by_christian_l196_196028


namespace sum_first_n_terms_l196_196746

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end sum_first_n_terms_l196_196746


namespace sum_of_k_values_l196_196356

theorem sum_of_k_values 
  (h : ∀ (k : ℤ), (∀ x y : ℤ, x * y = 15 → x + y = k) → k > 0 → false) : 
  ∃ k_values : List ℤ, 
  (∀ (k : ℤ), k ∈ k_values → (∀ x y : ℤ, x * y = 15 → x + y = k) ∧ k > 0) ∧ 
  k_values.sum = 24 := sorry

end sum_of_k_values_l196_196356


namespace range_of_a_plus_b_l196_196053

variable (a b : ℝ)
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (h : a + b + 1/a + 1/b = 5)

theorem range_of_a_plus_b : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end range_of_a_plus_b_l196_196053


namespace problem_l196_196632

variable {x y : ℝ}

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : (x - y)^2 = 41 := 
by
  sorry

end problem_l196_196632


namespace inequality_proof_l196_196299

-- Define the given conditions
def a : ℝ := Real.log 0.99
def b : ℝ := Real.exp 0.1
def c : ℝ := Real.exp (Real.log 0.99) ^ Real.exp 1

-- State the goal to be proved
theorem inequality_proof : a < c ∧ c < b := 
by
  sorry

end inequality_proof_l196_196299


namespace max_y_coordinate_l196_196469

theorem max_y_coordinate (θ : ℝ) : (∃ θ : ℝ, r = sin (3 * θ) → y = r * sin θ → y ≤ (2 * sqrt 3) / 3 - (2 * sqrt 3) / 9) :=
by
  have r := sin (3 * θ)
  have y := r * sin θ
  sorry

end max_y_coordinate_l196_196469


namespace rational_root_theorem_l196_196161

theorem rational_root_theorem :
  (∃ x : ℚ, 3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0)
  → (x = 1 ∨ x = 1/3) := by
  sorry

end rational_root_theorem_l196_196161


namespace probability_drawing_red_l196_196715

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l196_196715


namespace largest_n_property_l196_196659

open Set Nat

theorem largest_n_property :
  ∀ n : ℕ,
    (∀ S : Finset ℕ, S.card = 51 → ∃ a b ∈ S, a ≠ b ∧ a + b = 101) ↔ n = 100 :=
by
  sorry

end largest_n_property_l196_196659


namespace lost_card_l196_196400

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l196_196400


namespace frac_sum_property_l196_196064

theorem frac_sum_property (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end frac_sum_property_l196_196064


namespace distinct_trees_with_7_vertices_l196_196062

theorem distinct_trees_with_7_vertices : 
  count_distinct_trees 7 = 11 :=
sorry

end distinct_trees_with_7_vertices_l196_196062


namespace find_oranges_to_put_back_l196_196018

theorem find_oranges_to_put_back (A O x : ℕ) (h₁ : A + O = 15) (h₂ : 40 * A + 60 * O = 720) (h₃ : (360 + 360 - 60 * x) / (15 - x) = 45) : x = 3 := by
  sorry

end find_oranges_to_put_back_l196_196018


namespace intersection_M_N_eq_neg2_l196_196910

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l196_196910


namespace smallest_boxes_l196_196108

theorem smallest_boxes (n : Nat) (h₁ : n % 5 = 0) (h₂ : n % 24 = 0) : n = 120 := 
  sorry

end smallest_boxes_l196_196108


namespace eggs_distributed_equally_l196_196278

-- Define the total number of eggs
def total_eggs : ℕ := 8484

-- Define the number of baskets
def baskets : ℕ := 303

-- Define the expected number of eggs per basket
def eggs_per_basket : ℕ := 28

-- State the theorem
theorem eggs_distributed_equally :
  total_eggs / baskets = eggs_per_basket := sorry

end eggs_distributed_equally_l196_196278


namespace value_of_x_l196_196192

noncomputable def sum_integers_30_to_50 : ℕ :=
  (50 - 30 + 1) * (30 + 50) / 2

def even_count_30_to_50 : ℕ :=
  11

theorem value_of_x 
  (x := sum_integers_30_to_50)
  (y := even_count_30_to_50)
  (h : x + y = 851) : x = 840 :=
sorry

end value_of_x_l196_196192


namespace original_pencils_l196_196376

-- Definition of the conditions
def pencils_initial := 115
def pencils_added := 100
def pencils_total := 215

-- Theorem stating the problem to be proved
theorem original_pencils :
  pencils_initial + pencils_added = pencils_total :=
by
  sorry

end original_pencils_l196_196376


namespace geese_percentage_among_non_swan_birds_l196_196206

theorem geese_percentage_among_non_swan_birds :
  let total_birds := 100
  let geese := 0.40 * total_birds
  let swans := 0.20 * total_birds
  let non_swans := total_birds - swans
  let geese_percentage_among_non_swans := (geese / non_swans) * 100
  geese_percentage_among_non_swans = 50 := 
by sorry

end geese_percentage_among_non_swan_birds_l196_196206


namespace race_runners_l196_196009

theorem race_runners (k : ℕ) (h1 : 2*(k - 1) = k - 1) (h2 : 2*(2*(k + 9) - 12) = k + 9) : 3*k - 2 = 31 :=
by
  sorry

end race_runners_l196_196009


namespace total_sequins_correct_l196_196327

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l196_196327


namespace investor_receives_7260_l196_196561

-- Define the initial conditions
def principal : ℝ := 6000
def annual_rate : ℝ := 0.10
def compoundings_per_year : ℝ := 1
def years : ℝ := 2

-- Define the compound interest formula
noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

-- State the theorem: The investor will receive $7260 after two years
theorem investor_receives_7260 : compound_interest principal annual_rate compoundings_per_year years = 7260 := by
  sorry

end investor_receives_7260_l196_196561


namespace greatest_x_integer_l196_196831

theorem greatest_x_integer (x : ℤ) (h : ∃ n : ℤ, x^2 + 2 * x + 7 = (x - 4) * n) : x ≤ 35 :=
sorry

end greatest_x_integer_l196_196831


namespace solve_for_x_l196_196045

theorem solve_for_x (x : ℝ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l196_196045


namespace total_sequins_is_162_l196_196329

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l196_196329


namespace angles_count_geometric_seq_l196_196272

theorem angles_count_geometric_seq :
  let S := {θ : ℝ | 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ ∀ n : ℤ, θ ≠ n * Real.pi / 2} in
  {θ ∈ S | ∃ (f g h : ℝ), {f, g, h} = {Real.sin θ, Real.cos θ, Real.tan θ} ∧
    (g = f * (Real.sqrt ((h/g) * h) : ℝ) ∨ g = f / (Real.sqrt ((h/g) * h) : ℝ))}.toFinset.card = 4 :=
sorry

end angles_count_geometric_seq_l196_196272


namespace linear_relationship_correct_profit_160_max_profit_l196_196575

-- Define the conditions for the problem
def data_points : List (ℝ × ℝ) := [(3.5, 280), (5.5, 120)]

-- The linear function relationship between y and x
def linear_relationship (x : ℝ) : ℝ := -80 * x + 560

-- The equation for profit, given selling price and sales quantity
def profit (x : ℝ) : ℝ := (x - 3) * (linear_relationship x) - 80

-- Prove the relationship y = -80x + 560 from given data points
theorem linear_relationship_correct : 
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = linear_relationship x :=
sorry

-- Prove the selling price x = 4 results in a profit of $160 per day
theorem profit_160 (x : ℝ) (h : profit x = 160) : x = 4 :=
sorry

-- Prove the maximum profit and corresponding selling price
theorem max_profit : 
  ∃ x : ℝ, ∃ w : ℝ, 3.5 ≤ x ∧ x ≤ 5.5 ∧ profit x = w ∧ ∀ y, 3.5 ≤ y ∧ y ≤ 5.5 → profit y ≤ w ∧ w = 240 ∧ x = 5 :=
sorry

end linear_relationship_correct_profit_160_max_profit_l196_196575


namespace part_i_part_ii_l196_196658

-- Part (i)
theorem part_i : 
  let A := fun (a b : ℕ) => (1 <= a ∧ a <= 6) ∧ (1 <= b ∧ b <= 6) ∧ (a < b * Real.sqrt 3) in
  (PMF.filter A (PMF.uniform_of_finset (Finset.product (Finset.range 7) (Finset.range 7)))).toRealProb = 7 / 12 := 
sorry

-- Part (ii)
theorem part_ii :
  let A := fun (a b : ℝ) => (a - Real.sqrt 3)^2 + (b - 1)^2 <= 4 ∧ (a < b * Real.sqrt 3) in
  (MeasureTheory.measureSpace.restrict MeasureTheory.measureSpace.volume { p : ℝ × ℝ | A p.fst p.snd } 
   (FloatingMeasureTheory.volume {p : ℝ × ℝ | (a~-Real.sqrt 3)^2 +(b-1)^2<=4 }).toRealMeasure.toProbabilitySpace = 1 / 2 := 
sorry

end part_i_part_ii_l196_196658


namespace articleWords_l196_196010

-- Define the number of words per page for larger and smaller types
def wordsLargerType : Nat := 1800
def wordsSmallerType : Nat := 2400

-- Define the total number of pages and the number of pages in smaller type
def totalPages : Nat := 21
def smallerTypePages : Nat := 17

-- The number of pages in larger type
def largerTypePages : Nat := totalPages - smallerTypePages

-- Calculate the total number of words in the article
def totalWords : Nat := (largerTypePages * wordsLargerType) + (smallerTypePages * wordsSmallerType)

-- Prove that the total number of words in the article is 48,000
theorem articleWords : totalWords = 48000 := 
by
  sorry

end articleWords_l196_196010


namespace how_many_did_not_play_l196_196435

def initial_players : ℕ := 40
def first_half_starters : ℕ := 11
def first_half_substitutions : ℕ := 4
def second_half_extra_substitutions : ℕ := (first_half_substitutions * 3) / 4 -- 75% more substitutions
def injury_substitution : ℕ := 1
def total_second_half_substitutions : ℕ := first_half_substitutions + second_half_extra_substitutions + injury_substitution
def total_players_played : ℕ := first_half_starters + first_half_substitutions + total_second_half_substitutions
def players_did_not_play : ℕ := initial_players - total_players_played

theorem how_many_did_not_play : players_did_not_play = 17 := by
  sorry

end how_many_did_not_play_l196_196435


namespace ninth_square_more_than_eighth_l196_196141

noncomputable def side_length (n : ℕ) : ℕ := 3 + 2 * (n - 1)

noncomputable def tile_count (n : ℕ) : ℕ := (side_length n) ^ 2

theorem ninth_square_more_than_eighth : (tile_count 9 - tile_count 8) = 72 :=
by sorry

end ninth_square_more_than_eighth_l196_196141


namespace roots_subtraction_l196_196569

theorem roots_subtraction (a b : ℝ) (h_roots : a * b = 20 ∧ a + b = 12) (h_order : a > b) : a - b = 8 :=
sorry

end roots_subtraction_l196_196569


namespace arithmetic_sequence_angles_sum_l196_196650

theorem arithmetic_sequence_angles_sum (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : 2 * B = A + C) :
  A + C = 120 :=
by
  sorry

end arithmetic_sequence_angles_sum_l196_196650


namespace value_of_expression_l196_196444

variables {x1 x2 x3 x4 x5 x6 : ℝ}

theorem value_of_expression
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 = 1)
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 = 14)
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 = 135) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 = 832 :=
by
  sorry

end value_of_expression_l196_196444


namespace find_reflection_line_l196_196995

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line_l196_196995


namespace intersection_M_N_is_correct_l196_196921

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l196_196921


namespace passengers_final_count_l196_196374

structure BusStop :=
  (initial_passengers : ℕ)
  (first_stop_increase : ℕ)
  (other_stops_decrease : ℕ)
  (other_stops_increase : ℕ)

def passengers_at_last_stop (b : BusStop) : ℕ :=
  b.initial_passengers + b.first_stop_increase - b.other_stops_decrease + b.other_stops_increase

theorem passengers_final_count :
  passengers_at_last_stop ⟨50, 16, 22, 5⟩ = 49 := by
  rfl

end passengers_final_count_l196_196374


namespace power_function_at_point_l196_196926

theorem power_function_at_point (f : ℝ → ℝ) (h : ∃ α, ∀ x, f x = x^α) (hf : f 2 = 4) : f 3 = 9 :=
sorry

end power_function_at_point_l196_196926


namespace percentage_of_cars_on_monday_compared_to_tuesday_l196_196820

theorem percentage_of_cars_on_monday_compared_to_tuesday : 
  ∀ (cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun : ℕ),
    cars_mon + cars_tue + cars_wed + cars_thu + cars_fri + cars_sat + cars_sun = 97 →
    cars_tue = 25 →
    cars_wed = cars_mon + 2 →
    cars_thu = 10 →
    cars_fri = 10 →
    cars_sat = 5 →
    cars_sun = 5 →
    (cars_mon * 100 / cars_tue = 80) :=
by
  intros cars_mon cars_tue cars_wed cars_thu cars_fri cars_sat cars_sun
  intro h_total
  intro h_tue
  intro h_wed
  intro h_thu
  intro h_fri
  intro h_sat
  intro h_sun
  sorry

end percentage_of_cars_on_monday_compared_to_tuesday_l196_196820


namespace probability_at_least_one_defective_is_correct_l196_196129

/-- Define a box containing 21 bulbs, 4 of which are defective -/
def total_bulbs : ℕ := 21
def defective_bulbs : ℕ := 4
def non_defective_bulbs : ℕ := total_bulbs - defective_bulbs

/-- Define probabilities of choosing non-defective bulbs -/
def prob_first_non_defective : ℚ := non_defective_bulbs / total_bulbs
def prob_second_non_defective : ℚ := (non_defective_bulbs - 1) / (total_bulbs - 1)

/-- Calculate the probability of both bulbs being non-defective -/
def prob_both_non_defective : ℚ := prob_first_non_defective * prob_second_non_defective

/-- Calculate the probability of at least one defective bulb -/
def prob_at_least_one_defective : ℚ := 1 - prob_both_non_defective

theorem probability_at_least_one_defective_is_correct :
  prob_at_least_one_defective = 37 / 105 :=
by
  -- Sorry allows us to skip the proof
  sorry

end probability_at_least_one_defective_is_correct_l196_196129


namespace lollipops_initial_count_l196_196131

theorem lollipops_initial_count (L : ℕ) (k : ℕ) 
  (h1 : L % 42 ≠ 0) 
  (h2 : (L + 22) % 42 = 0) : 
  L = 62 :=
by
  sorry

end lollipops_initial_count_l196_196131


namespace power_function_monotonic_decreasing_l196_196393

theorem power_function_monotonic_decreasing (α : ℝ) (h : ∀ x y : ℝ, 0 < x → x < y → x^α > y^α) : α < 0 :=
sorry

end power_function_monotonic_decreasing_l196_196393


namespace probability_number_is_odd_l196_196229

def definition_of_odds : set ℕ := {3, 5, 7, 9}

def number_is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem probability_number_is_odd :
  let total_digits := {2, 3, 5, 7, 9}
  let odd_digits := definition_of_odds
  let favorable_outcomes := set.card odd_digits
  let total_outcomes := set.card total_digits
  in (total_outcomes = 5) -> (favorable_outcomes = 4) -> (favorable_outcomes / total_outcomes : ℚ) = 4 / 5 :=
by
  intro total_digits odd_digits favorable_outcomes total_outcomes total_outcomes_eq favorable_outcomes_eq
  have h1 : favorable_outcomes = 4 := favorable_outcomes_eq
  have h2 : total_outcomes = 5 := total_outcomes_eq
  norm_num
  sorry

end probability_number_is_odd_l196_196229


namespace intersection_M_N_l196_196918

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l196_196918


namespace tuesday_more_than_monday_l196_196222

variable (M T W Th x : ℕ)

-- Conditions
def monday_dinners : M = 40 := by sorry
def tuesday_dinners : T = M + x := by sorry
def wednesday_dinners : W = T / 2 := by sorry
def thursday_dinners : Th = W + 3 := by sorry
def total_dinners : M + T + W + Th = 203 := by sorry

-- Proof problem: How many more dinners were sold on Tuesday than on Monday?
theorem tuesday_more_than_monday : x = 32 :=
by
  sorry

end tuesday_more_than_monday_l196_196222


namespace largest_pos_int_divisor_l196_196604

theorem largest_pos_int_divisor:
  ∃ n : ℕ, (n + 10 ∣ n^3 + 2011) ∧ (∀ m : ℕ, (m + 10 ∣ m^3 + 2011) → m ≤ n) :=
sorry

end largest_pos_int_divisor_l196_196604


namespace largest_of_three_consecutive_integers_sum_18_l196_196365

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l196_196365


namespace value_of_f3_f10_l196_196179

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) : f (x + 4) = f x + f 2
axiom f_at_one : f 1 = 4

theorem value_of_f3_f10 : f 3 + f 10 = 4 := sorry

end value_of_f3_f10_l196_196179


namespace least_five_digit_congruent_to_7_mod_18_l196_196385

theorem least_five_digit_congruent_to_7_mod_18 : 
  ∃ n, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
  ∃ n, 10015 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 7 ∧ ∀ m, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 7 → n ≤ m :=
sorry

end least_five_digit_congruent_to_7_mod_18_l196_196385


namespace relationship_f_3x_ge_f_2x_l196_196104

/-- Given a quadratic function f(x) = ax^2 + bx + c with a > 0, and
    satisfying the symmetry condition f(1-x) = f(1+x) for any x ∈ ℝ,
    the relationship f(3^x) ≥ f(2^x) holds. -/
theorem relationship_f_3x_ge_f_2x (a b c : ℝ) (h_a : a > 0) (symm_cond : ∀ x : ℝ, (a * (1 - x)^2 + b * (1 - x) + c) = (a * (1 + x)^2 + b * (1 + x) + c)) :
  ∀ x : ℝ, (a * (3^x)^2 + b * 3^x + c) ≥ (a * (2^x)^2 + b * 2^x + c) :=
sorry

end relationship_f_3x_ge_f_2x_l196_196104


namespace Q_cannot_be_log_x_l196_196188

def P : Set ℝ := {y | y ≥ 0}

theorem Q_cannot_be_log_x (Q : Set ℝ) :
  (P ∩ Q = Q) → Q ≠ {y | ∃ x, y = Real.log x} :=
by
  sorry

end Q_cannot_be_log_x_l196_196188


namespace map_distance_8_cm_l196_196070

-- Define the conditions
def scale : ℕ := 5000000
def actual_distance_km : ℕ := 400
def actual_distance_cm : ℕ := 40000000
def map_distance_cm (x : ℕ) : Prop := x * scale = actual_distance_cm

-- The theorem to be proven
theorem map_distance_8_cm : ∃ x : ℕ, map_distance_cm x ∧ x = 8 :=
by
  use 8
  unfold map_distance_cm
  norm_num
  sorry

end map_distance_8_cm_l196_196070


namespace find_g_values_l196_196981

theorem find_g_values
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x * y) = x * g y)
  (h2 : g 1 = 30) :
  g 50 = 1500 ∧ g 0.5 = 15 :=
by
  sorry

end find_g_values_l196_196981


namespace avery_donation_l196_196020

theorem avery_donation (shirts pants shorts : ℕ)
  (h_shirts : shirts = 4)
  (h_pants : pants = 2 * shirts)
  (h_shorts : shorts = pants / 2) :
  shirts + pants + shorts = 16 := by
  sorry

end avery_donation_l196_196020


namespace prove_ordered_triple_l196_196961

theorem prove_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) : 
  (x, y, z) = (13, 11, 6) :=
sorry

end prove_ordered_triple_l196_196961


namespace pencil_and_pen_cost_l196_196680

theorem pencil_and_pen_cost
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 3.75)
  (h2 : 2 * p + 3 * q = 4.05) :
  p + q = 1.56 :=
by
  sorry

end pencil_and_pen_cost_l196_196680


namespace walking_west_10_neg_l196_196071

-- Define the condition that walking east for 20 meters is +20 meters
def walking_east_20 := 20

-- Assert that walking west for 10 meters is -10 meters given the east direction definition
theorem walking_west_10_neg : walking_east_20 = 20 → (-10 = -10) :=
by
  intro h
  sorry

end walking_west_10_neg_l196_196071


namespace tan_pi_add_alpha_eq_two_l196_196298

theorem tan_pi_add_alpha_eq_two
  (α : ℝ)
  (h : Real.tan (Real.pi + α) = 2) :
  (2 * Real.sin α - Real.cos α) / (3 * Real.sin α + 2 * Real.cos α) = 3 / 8 :=
sorry

end tan_pi_add_alpha_eq_two_l196_196298


namespace intersection_M_N_l196_196909

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196909


namespace max_result_of_operation_l196_196870

theorem max_result_of_operation : ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → 3 * (300 - m) ≤ 870) ∧ 3 * (300 - n) = 870 :=
by
  sorry

end max_result_of_operation_l196_196870


namespace arctan_addition_formula_l196_196887

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula_l196_196887


namespace geometric_sequence_sum_l196_196514

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (n : ℕ) (hS : ∀ n, S n = t - 3 * 2^n) (h_geom : ∀ n, a (n + 1) = a n * r) :
  t = 3 :=
by
  sorry

end geometric_sequence_sum_l196_196514


namespace permits_increase_l196_196099

theorem permits_increase :
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  new_permits = 67600 * old_permits :=
by
  let old_permits := 26^2 * 10^3
  let new_permits := 26^4 * 10^4
  exact sorry

end permits_increase_l196_196099


namespace tom_charges_per_lawn_l196_196558

theorem tom_charges_per_lawn (gas_cost earnings_from_weeding total_profit lawns_mowed : ℕ) (charge_per_lawn : ℤ) 
  (h1 : gas_cost = 17)
  (h2 : earnings_from_weeding = 10)
  (h3 : total_profit = 29)
  (h4 : lawns_mowed = 3)
  (h5 : total_profit = ((lawns_mowed * charge_per_lawn) + earnings_from_weeding) - gas_cost) :
  charge_per_lawn = 12 := 
by
  sorry

end tom_charges_per_lawn_l196_196558


namespace S_of_1_eq_8_l196_196616

variable (x : ℝ)

-- Definition of original polynomial R(x)
def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

-- Definition of new polynomial S(x) created by adding 2 to each coefficient of R(x)
def S (x : ℝ) : ℝ := 5 * x^3 - 3 * x + 6

-- The theorem we want to prove
theorem S_of_1_eq_8 : S 1 = 8 := by
  sorry

end S_of_1_eq_8_l196_196616


namespace find_g_l196_196341

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g (g : ℝ → ℝ)
  (H : ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 4) :
  g = fun x => x + 5 :=
by
  sorry

end find_g_l196_196341


namespace education_fund_growth_l196_196115

theorem education_fund_growth (x : ℝ) :
  2500 * (1 + x)^2 = 3600 :=
sorry

end education_fund_growth_l196_196115


namespace surface_area_of_brick_l196_196126

namespace SurfaceAreaProof

def brick_length : ℝ := 8
def brick_width : ℝ := 6
def brick_height : ℝ := 2

theorem surface_area_of_brick :
  2 * (brick_length * brick_width + brick_length * brick_height + brick_width * brick_height) = 152 :=
by
  sorry

end SurfaceAreaProof

end surface_area_of_brick_l196_196126


namespace find_numerator_l196_196186

theorem find_numerator (n : ℕ) : 
  (n : ℚ) / 22 = 9545 / 10000 → 
  n = 9545 * 22 / 10000 :=
by sorry

end find_numerator_l196_196186


namespace minimum_value_of_quadratic_l196_196119

def quadratic_polynomial (x : ℝ) : ℝ := 2 * x^2 - 16 * x + 22

theorem minimum_value_of_quadratic : ∃ x : ℝ, quadratic_polynomial x = -10 :=
by 
  use 4
  { sorry }

end minimum_value_of_quadratic_l196_196119


namespace Cherry_weekly_earnings_l196_196718

theorem Cherry_weekly_earnings :
  let charge_small_cargo := 2.50
  let charge_large_cargo := 4.00
  let daily_small_cargo := 4
  let daily_large_cargo := 2
  let days_in_week := 7
  let daily_earnings := (charge_small_cargo * daily_small_cargo) + (charge_large_cargo * daily_large_cargo)
  let weekly_earnings := daily_earnings * days_in_week
  weekly_earnings = 126 := sorry

end Cherry_weekly_earnings_l196_196718


namespace six_digit_palindromes_count_l196_196381

theorem six_digit_palindromes_count : 
  ∃ n : ℕ, n = 27 ∧ 
  (∀ (A B C : ℕ), 
       (A = 6 ∨ A = 7 ∨ A = 8) ∧ 
       (B = 6 ∨ B = 7 ∨ B = 8) ∧ 
       (C = 6 ∨ C = 7 ∨ C = 8) → 
       ∃ p : ℕ, 
         p = (A * 10^5 + B * 10^4 + C * 10^3 + C * 10^2 + B * 10 + A) ∧ 
         (6 ≤ p / 10^5 ∧ p / 10^5 ≤ 8) ∧ 
         (6 ≤ (p / 10^4) % 10 ∧ (p / 10^4) % 10 ≤ 8) ∧ 
         (6 ≤ (p / 10^3) % 10 ∧ (p / 10^3) % 10 ≤ 8)) :=
  by sorry

end six_digit_palindromes_count_l196_196381


namespace right_triangle_perimeter_l196_196434

theorem right_triangle_perimeter (n : ℕ) (hn : Nat.Prime n) (x y : ℕ) 
  (h1 : y^2 = x^2 + n^2) : n + x + y = n + n^2 := by
  sorry

end right_triangle_perimeter_l196_196434


namespace total_fruits_picked_l196_196512

variable (L M P B : Nat)

theorem total_fruits_picked (hL : L = 25) (hM : M = 32) (hP : P = 12) (hB : B = 18) : L + M + P = 69 :=
by
  sorry

end total_fruits_picked_l196_196512


namespace sum_reciprocal_eq_eleven_eighteen_l196_196586

noncomputable def sum_reciprocal (n : ℕ) : ℝ := ∑' (n : ℕ), 1 / (n * (n + 3))

theorem sum_reciprocal_eq_eleven_eighteen :
  sum_reciprocal = 11 / 18 :=
by
  sorry

end sum_reciprocal_eq_eleven_eighteen_l196_196586


namespace john_has_25_roommates_l196_196957

def roommates_of_bob := 10
def roommates_of_john := 2 * roommates_of_bob + 5

theorem john_has_25_roommates : roommates_of_john = 25 := 
by
  sorry

end john_has_25_roommates_l196_196957


namespace odd_integer_solution_l196_196734

theorem odd_integer_solution
  (y : ℤ) (hy_odd : y % 2 = 1)
  (h : ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1) :
  y = 1 :=
sorry

end odd_integer_solution_l196_196734


namespace length_of_each_stone_l196_196137

theorem length_of_each_stone {L : ℝ} (hall_length hall_breadth : ℝ) (stone_breadth : ℝ) (num_stones : ℕ) (area_hall : ℝ) (area_stone : ℝ) :
  hall_length = 36 * 10 ∧ hall_breadth = 15 * 10 ∧ stone_breadth = 5 ∧ num_stones = 3600 ∧
  area_hall = hall_length * hall_breadth ∧ area_stone = L * stone_breadth ∧
  area_stone * num_stones = area_hall →
  L = 3 :=
by
  sorry

end length_of_each_stone_l196_196137


namespace hummus_serving_amount_proof_l196_196821

/-- Given conditions: 
    one_can is the number of ounces of chickpeas in one can,
    total_cans is the number of cans Thomas buys,
    total_servings is the number of servings of hummus Thomas needs to make,
    to_produce_one_serving is the amount of chickpeas needed for one serving,
    we prove that to_produce_one_serving = 6.4 given the above conditions. -/
theorem hummus_serving_amount_proof 
  (one_can : ℕ) 
  (total_cans : ℕ) 
  (total_servings : ℕ) 
  (to_produce_one_serving : ℚ) 
  (h_one_can : one_can = 16) 
  (h_total_cans : total_cans = 8)
  (h_total_servings : total_servings = 20) 
  (h_total_ounces : total_cans * one_can = 128) : 
  to_produce_one_serving = 128 / 20 := 
by
  sorry

end hummus_serving_amount_proof_l196_196821


namespace range_of_b_plus_c_l196_196105

noncomputable def func (b c x : ℝ) : ℝ := x^2 + b*x + c * 3^x

theorem range_of_b_plus_c {b c : ℝ} (h1 : ∃ x, func b c x = 0)
  (h2 : ∀ x, (func b c x = 0 ↔ func b c (func b c x) = 0)) : 
  0 ≤ b + c ∧ b + c < 4 :=
by
  sorry

end range_of_b_plus_c_l196_196105


namespace fraction_simplify_l196_196159

variable (a b c : ℝ)

theorem fraction_simplify
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : a + 2 * b + 3 * c ≠ 0) :
  (a^2 + 4 * b^2 - 9 * c^2 + 4 * a * b) / (a^2 + 9 * c^2 - 4 * b^2 + 6 * a * c) =
  (a + 2 * b - 3 * c) / (a - 2 * b + 3 * c) := by
  sorry

end fraction_simplify_l196_196159


namespace min_n_Sn_greater_1020_l196_196361

theorem min_n_Sn_greater_1020 : ∃ n : ℕ, (n ≥ 0) ∧ (2^(n+1) - 2 - n > 1020) ∧ ∀ m : ℕ, (m ≥ 0) ∧ (m < n) → (2^(m+1) - 2 - m ≤ 1020) :=
by
  sorry

end min_n_Sn_greater_1020_l196_196361


namespace circle_tangent_radius_l196_196822

-- Define the radii of the three given circles
def radius1 : ℝ := 1.0
def radius2 : ℝ := 2.0
def radius3 : ℝ := 3.0

-- Define the problem statement: finding the radius of the fourth circle externally tangent to the given three circles
theorem circle_tangent_radius (r1 r2 r3 : ℝ) (cond1 : r1 = 1) (cond2 : r2 = 2) (cond3 : r3 = 3) : 
  ∃ R : ℝ, R = 6 := by
  sorry

end circle_tangent_radius_l196_196822


namespace garrison_reinforcement_l196_196135

/-- A garrison has initial provisions for 2000 men for 65 days. 
    After 15 days, reinforcement arrives and the remaining provisions last for 20 more days. 
    The size of the reinforcement is 3000 men.  -/
theorem garrison_reinforcement (P : ℕ) (M1 M2 D1 D2 D3 R : ℕ) 
  (h1 : M1 = 2000) (h2 : D1 = 65) (h3 : D2 = 15) (h4 : D3 = 20) 
  (h5 : P = M1 * D1) (h6 : P - M1 * D2 = (M1 + R) * D3) : 
  R = 3000 := 
sorry

end garrison_reinforcement_l196_196135


namespace parallel_vectors_x_value_l196_196934

theorem parallel_vectors_x_value :
  ∀ (x : ℝ), (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (2, x) → a.1 * b.2 = a.2 * b.1) → x = -4 :=
by
  intros x h
  have h_parallel := h (1, -2) (2, x) rfl rfl
  sorry

end parallel_vectors_x_value_l196_196934


namespace sum_of_roots_l196_196313

theorem sum_of_roots (m n : ℝ) (h1 : ∀ x, x^2 - 3 * x - 1 = 0 → x = m ∨ x = n) : m + n = 3 :=
sorry

end sum_of_roots_l196_196313


namespace abs_eq_cases_l196_196844

theorem abs_eq_cases (a b : ℝ) : (|a| = |b|) → (a = b ∨ a = -b) :=
sorry

end abs_eq_cases_l196_196844


namespace imaginary_unit_cubic_l196_196106

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

theorem imaginary_unit_cubic (i : ℂ) (h : imaginary_unit_property i) : 1 + i^3 = 1 - i :=
  sorry

end imaginary_unit_cubic_l196_196106


namespace cylinder_height_l196_196815

theorem cylinder_height (h : ℝ)
  (circumference : ℝ)
  (rectangle_diagonal : ℝ)
  (C_eq : circumference = 12)
  (d_eq : rectangle_diagonal = 20) :
  h = 16 :=
by
  -- We derive the result based on the given conditions and calculations
  sorry -- Skipping the proof part

end cylinder_height_l196_196815


namespace find_constant_a_find_ordinary_equation_of_curve_l196_196621

open Real

theorem find_constant_a (a t : ℝ) (h1 : 1 + 2 * t = 3) (h2 : a * t^2 = 1) : a = 1 :=
by
  -- Proof goes here
  sorry

theorem find_ordinary_equation_of_curve (x y t : ℝ) (h1 : x = 1 + 2 * t) (h2 : y = t^2) :
  (x - 1)^2 = 4 * y :=
by
  -- Proof goes here
  sorry

end find_constant_a_find_ordinary_equation_of_curve_l196_196621


namespace meal_cost_l196_196430

/-- 
    Define the cost of a meal consisting of one sandwich, one cup of coffee, and one piece of pie 
    given the costs of two different meals.
-/
theorem meal_cost (s c p : ℝ) (h1 : 2 * s + 5 * c + p = 5) (h2 : 3 * s + 8 * c + p = 7) :
    s + c + p = 3 :=
by
  sorry

end meal_cost_l196_196430


namespace fraction_of_A_or_B_l196_196195

def fraction_A : ℝ := 0.7
def fraction_B : ℝ := 0.2

theorem fraction_of_A_or_B : fraction_A + fraction_B = 0.9 := 
by
  sorry

end fraction_of_A_or_B_l196_196195


namespace intersection_M_N_l196_196900

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196900


namespace initial_ripe_peaches_l196_196252

theorem initial_ripe_peaches (P U R: ℕ) (H1: P = 18) (H2: 2 * 5 = 10) (H3: (U + 7) + U = 15 - 3) (H4: R + 10 = U + 7) : 
  R = 1 :=
by
  sorry

end initial_ripe_peaches_l196_196252


namespace disjoint_subsets_same_sum_l196_196998

/-- 
Given a set of 10 distinct integers between 1 and 100, 
there exist two disjoint subsets of this set that have the same sum.
-/
theorem disjoint_subsets_same_sum : ∃ (x : Finset ℤ), (x.card = 10) ∧ (∀ i ∈ x, 1 ≤ i ∧ i ≤ 100) → 
  ∃ (A B : Finset ℤ), (A ⊆ x) ∧ (B ⊆ x) ∧ (A ∩ B = ∅) ∧ (A.sum id = B.sum id) :=
by
  sorry

end disjoint_subsets_same_sum_l196_196998


namespace number_of_puppies_l196_196352

def total_portions : Nat := 105
def feeding_days : Nat := 5
def feedings_per_day : Nat := 3

theorem number_of_puppies (total_portions feeding_days feedings_per_day : Nat) : 
  (total_portions / feeding_days / feedings_per_day = 7) := 
by 
  sorry

end number_of_puppies_l196_196352


namespace solution_sum_of_eq_zero_l196_196838

open Real

theorem solution_sum_of_eq_zero : 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  in (∀ x, f x = 0 → x = -3/2 ∨ x = 8/3) → 
     (-3/2 + 8/3 = 7/6) :=
by 
  let f (x : ℝ) := (4*x + 6) * (3*x - 8)
  intro h
  have h₁ : f(-3/2) = 0 := by sorry
  have h₂ : f(8/3) = 0 := by sorry
  have sum_of_solutions : -3/2 + 8/3 = 7/6 := by 
    sorry
  exact sum_of_solutions

end solution_sum_of_eq_zero_l196_196838


namespace remainder_sum_mod_14_l196_196943

theorem remainder_sum_mod_14 
  (a b c : ℕ) 
  (ha : a % 14 = 5) 
  (hb : b % 14 = 5) 
  (hc : c % 14 = 5) :
  (a + b + c) % 14 = 1 := 
by
  sorry

end remainder_sum_mod_14_l196_196943


namespace largest_of_three_consecutive_integers_sum_18_l196_196367

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l196_196367


namespace ted_worked_hours_l196_196845

variable (t : ℝ)
variable (julie_rate ted_rate combined_rate : ℝ)
variable (julie_alone_time : ℝ)
variable (job_done : ℝ)

theorem ted_worked_hours :
  julie_rate = 1 / 10 →
  ted_rate = 1 / 8 →
  combined_rate = julie_rate + ted_rate →
  julie_alone_time = 0.9999999999999998 →
  job_done = combined_rate * t + julie_rate * julie_alone_time →
  t = 4 :=
by
  sorry

end ted_worked_hours_l196_196845


namespace prove_expression_value_l196_196596

theorem prove_expression_value (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 := by
  rw [h]
  sorry

end prove_expression_value_l196_196596


namespace second_month_sale_l196_196255

theorem second_month_sale (S : ℝ) :
  (S + 5420 + 6200 + 6350 + 6500 = 30000) → S = 5530 :=
by
  sorry

end second_month_sale_l196_196255


namespace max_largest_int_of_avg_and_diff_l196_196944

theorem max_largest_int_of_avg_and_diff (A B C D E : ℕ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : C ≤ D) (h4 : D ≤ E) 
  (h_avg : (A + B + C + D + E) / 5 = 70) (h_diff : E - A = 10) : E = 340 :=
by
  sorry

end max_largest_int_of_avg_and_diff_l196_196944


namespace committee_selection_correct_l196_196950

def num_ways_to_choose_committee : ℕ :=
  let total_people := 10
  let president_ways := total_people
  let vp_ways := total_people - 1
  let remaining_people := total_people - 2
  let committee_ways := Nat.choose remaining_people 2
  president_ways * vp_ways * committee_ways

theorem committee_selection_correct :
  num_ways_to_choose_committee = 2520 :=
by
  sorry

end committee_selection_correct_l196_196950


namespace radius_of_circle_l196_196235

theorem radius_of_circle (r : ℝ) (h : 6 * Real.pi * r + 6 = 2 * Real.pi * r^2) : 
  r = (3 + Real.sqrt 21) / 2 :=
by
  sorry

end radius_of_circle_l196_196235


namespace ways_to_sum_31_as_two_primes_l196_196498

theorem ways_to_sum_31_as_two_primes : 
  let p := 31 
  (p = 2 + 29 ∧ prime 2 ∧ prime 29) ∨ (p = 11 + 19 ∧ prime 11 ∧ prime 19) 
  → nat.num_possible_sums_of_two_primes p = 2 :=
by sorry

end ways_to_sum_31_as_two_primes_l196_196498


namespace day50_previous_year_is_Wednesday_l196_196954

-- Given conditions
variable (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)

-- Provided conditions stating specific days are Fridays
def day250_is_Friday : Prop := dayOfWeek 250 N = 5
def day150_is_Friday_next_year : Prop := dayOfWeek 150 (N+1) = 5

-- Proving the day of week for the 50th day of year N-1
def day50_previous_year : Prop := dayOfWeek 50 (N-1) = 3

-- Main theorem tying it together
theorem day50_previous_year_is_Wednesday (N : ℕ) (dayOfWeek : ℕ → ℕ → ℕ)
  (h1 : day250_is_Friday N dayOfWeek)
  (h2 : day150_is_Friday_next_year N dayOfWeek) :
  day50_previous_year N dayOfWeek :=
sorry -- Placeholder for actual proof

end day50_previous_year_is_Wednesday_l196_196954


namespace complement_intersection_l196_196060

open Set

-- Definitions based on conditions given
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- The mathematical proof problem
theorem complement_intersection :
  (U \ A) ∩ B = {1, 3, 7} :=
by
  sorry

end complement_intersection_l196_196060


namespace temperature_comparison_l196_196698

theorem temperature_comparison: ¬ (-3 > -0.3) :=
by
  sorry -- Proof goes here, skipped for now.

end temperature_comparison_l196_196698


namespace remainder_when_squared_l196_196843

theorem remainder_when_squared (n : ℕ) (h : n % 8 = 6) : (n * n) % 32 = 4 := by
  sorry

end remainder_when_squared_l196_196843


namespace devin_teaching_years_l196_196880

section DevinTeaching
variable (Calculus Algebra Statistics Geometry DiscreteMathematics : ℕ)

theorem devin_teaching_years :
  Calculus = 4 ∧
  Algebra = 2 * Calculus ∧
  Statistics = 5 * Algebra ∧
  Geometry = 3 * Statistics ∧
  DiscreteMathematics = Geometry / 2 ∧
  (Calculus + Algebra + Statistics + Geometry + DiscreteMathematics) = 232 :=
by
  sorry
end DevinTeaching

end devin_teaching_years_l196_196880


namespace total_points_each_team_l196_196819

def score_touchdown := 7
def score_field_goal := 3
def score_safety := 2

def team_hawks_first_match_score := 3 * score_touchdown + 2 * score_field_goal + score_safety
def team_eagles_first_match_score := 5 * score_touchdown + 4 * score_field_goal
def team_hawks_second_match_score := 4 * score_touchdown + 3 * score_field_goal
def team_falcons_second_match_score := 6 * score_touchdown + 2 * score_safety

def total_score_hawks := team_hawks_first_match_score + team_hawks_second_match_score
def total_score_eagles := team_eagles_first_match_score
def total_score_falcons := team_falcons_second_match_score

theorem total_points_each_team :
  total_score_hawks = 66 ∧ total_score_eagles = 47 ∧ total_score_falcons = 46 :=
by
  unfold total_score_hawks team_hawks_first_match_score team_hawks_second_match_score
           total_score_eagles team_eagles_first_match_score
           total_score_falcons team_falcons_second_match_score
           score_touchdown score_field_goal score_safety
  sorry

end total_points_each_team_l196_196819


namespace least_subtraction_to_divisible_by_prime_l196_196688

theorem least_subtraction_to_divisible_by_prime :
  ∃ k : ℕ, (k = 46) ∧ (856324 - k) % 101 = 0 :=
by
  sorry

end least_subtraction_to_divisible_by_prime_l196_196688


namespace largest_x_not_defined_l196_196384

theorem largest_x_not_defined : 
  (∀ x, (6 * x ^ 2 - 17 * x + 5 = 0) → x ≤ 2.5) ∧
  (∃ x, (6 * x ^ 2 - 17 * x + 5 = 0) ∧ x = 2.5) :=
by
  sorry

end largest_x_not_defined_l196_196384


namespace ratio_a_to_c_l196_196416

theorem ratio_a_to_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_to_c_l196_196416


namespace fraction_of_larger_part_l196_196224

theorem fraction_of_larger_part (x y : ℝ) (f : ℝ) (h1 : x = 50) (h2 : x + y = 66) (h3 : f * x = 0.625 * y + 10) : f = 0.4 :=
by
  sorry

end fraction_of_larger_part_l196_196224


namespace max_x1_x2_squares_l196_196615

noncomputable def x1_x2_squares_eq_max : Prop :=
  ∃ k : ℝ, (∀ x1 x2 : ℝ, (x1 + x2 = k - 2) ∧ (x1 * x2 = k^2 + 3 * k + 5) → x1^2 + x2^2 = 18)

theorem max_x1_x2_squares : x1_x2_squares_eq_max :=
by sorry

end max_x1_x2_squares_l196_196615


namespace quadratic_roots_l196_196282

theorem quadratic_roots (a b k : ℝ) (h₁ : a + b = -2) (h₂ : a * b = k / 3)
    (h₃ : |a - b| = 1/2 * (a^2 + b^2)) : k = 0 ∨ k = 6 :=
sorry

end quadratic_roots_l196_196282


namespace problem_l196_196939

noncomputable def K : ℕ := 36
noncomputable def L : ℕ := 147
noncomputable def M : ℕ := 56

theorem problem (h1 : 4 / 7 = K / 63) (h2 : 4 / 7 = 84 / L) (h3 : 4 / 7 = M / 98) :
  (K + L + M) = 239 :=
by
  sorry

end problem_l196_196939


namespace solve_inequality_l196_196608

noncomputable def solution_set (x : ℝ) : Prop :=
  (-(9/2) ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2)

theorem solve_inequality (x : ℝ) :
  (x ≠ -2 ∧ x ≠ 9/2) →
  ( (x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9) ) ↔ solution_set x :=
sorry

end solve_inequality_l196_196608


namespace find_lost_card_number_l196_196398

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l196_196398


namespace avery_donates_16_clothes_l196_196022

theorem avery_donates_16_clothes : 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
in Shirts + Pants + Shorts = 16 :=
by 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
  show Shirts + Pants + Shorts = 16
  sorry

end avery_donates_16_clothes_l196_196022


namespace instantaneous_velocity_at_2_l196_196187

def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

theorem instantaneous_velocity_at_2 : 
  (deriv s 2) = 29 :=
by
  -- The proof is skipped by using sorry
  sorry

end instantaneous_velocity_at_2_l196_196187


namespace trader_sold_meters_l196_196867

variable (x : ℕ) (SP P CP : ℕ)

theorem trader_sold_meters (h_SP : SP = 660) (h_P : P = 5) (h_CP : CP = 5) : x = 66 :=
  by
  sorry

end trader_sold_meters_l196_196867


namespace largest_of_three_consecutive_integers_l196_196370

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l196_196370


namespace pages_per_day_l196_196226

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (result : ℕ) :
  total_pages = 81 ∧ days = 3 → result = 27 :=
by
  sorry

end pages_per_day_l196_196226


namespace intersection_M_N_l196_196905

variable M : Set Int := {-2, -1, 0, 1, 2}
variable N : Set Int := {x | x^2 - x - 6 >= 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
by sorry

end intersection_M_N_l196_196905


namespace price_after_9_years_decreases_continuously_l196_196158

theorem price_after_9_years_decreases_continuously (price_current : ℝ) (price_after_9_years : ℝ) :
  (∀ k : ℕ, k % 3 = 0 → price_current = 8100 → price_after_9_years = 2400) :=
sorry

end price_after_9_years_decreases_continuously_l196_196158


namespace sqrt_nested_eq_x_pow_eleven_eighths_l196_196729

theorem sqrt_nested_eq_x_pow_eleven_eighths (x : ℝ) (hx : 0 ≤ x) : 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (11 / 8) :=
  sorry

end sqrt_nested_eq_x_pow_eleven_eighths_l196_196729


namespace problem_statement_l196_196484

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement_l196_196484


namespace distinct_negative_real_roots_l196_196462

def poly (p : ℝ) (x : ℝ) : ℝ := x^4 + 2*p*x^3 + x^2 + 2*p*x + 1

theorem distinct_negative_real_roots (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly p x1 = 0 ∧ poly p x2 = 0) ↔ p > 3/4 :=
sorry

end distinct_negative_real_roots_l196_196462


namespace solve_for_x_l196_196638

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end solve_for_x_l196_196638


namespace average_birds_seen_correct_l196_196517

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l196_196517


namespace roots_reciprocal_sum_l196_196211

theorem roots_reciprocal_sum
  (a b c : ℂ)
  (h : Polynomial.roots (Polynomial.C 1 + Polynomial.X - Polynomial.C 1 * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3) = {a, b, c}) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 :=
by
  sorry

end roots_reciprocal_sum_l196_196211


namespace tooth_fairy_left_amount_l196_196842

-- Define the values of the different types of coins
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50
def dime_value : ℝ := 0.10

-- Define the number of each type of coins Joan received
def num_quarters : ℕ := 14
def num_half_dollars : ℕ := 14
def num_dimes : ℕ := 14

-- Calculate the total values for each type of coin
def total_quarters_value : ℝ := num_quarters * quarter_value
def total_half_dollars_value : ℝ := num_half_dollars * half_dollar_value
def total_dimes_value : ℝ := num_dimes * dime_value

-- The total amount of money left by the tooth fairy
def total_amount_left := total_quarters_value + total_half_dollars_value + total_dimes_value

-- The theorem stating that the total amount is $11.90
theorem tooth_fairy_left_amount : total_amount_left = 11.90 := by 
  sorry

end tooth_fairy_left_amount_l196_196842


namespace floor_area_difference_l196_196599

noncomputable def area_difference (r_outer : ℝ) (n : ℕ) (r_inner : ℝ) : ℝ :=
  let outer_area := Real.pi * r_outer^2
  let inner_area := n * Real.pi * r_inner^2
  outer_area - inner_area

theorem floor_area_difference :
  ∀ (r_outer : ℝ) (n : ℕ) (r_inner : ℝ), 
  n = 8 ∧ r_outer = 40 ∧ r_inner = 40 / (2*Real.sqrt 2 + 1) →
  ⌊area_difference r_outer n r_inner⌋ = 1150 :=
by
  intros
  sorry

end floor_area_difference_l196_196599


namespace find_lost_card_number_l196_196399

theorem find_lost_card_number (n : ℕ) (S : ℕ) (x : ℕ) 
  (h1 : S = n * (n + 1) / 2) 
  (h2 : S - x = 101) 
  (h3 : n = 14) : 
  x = 4 := 
by sorry

end find_lost_card_number_l196_196399


namespace ned_shirts_problem_l196_196665

theorem ned_shirts_problem
  (long_sleeve_shirts : ℕ)
  (total_shirts_washed : ℕ)
  (total_shirts_had : ℕ)
  (h1 : long_sleeve_shirts = 21)
  (h2 : total_shirts_washed = 29)
  (h3 : total_shirts_had = total_shirts_washed + 1) :
  ∃ short_sleeve_shirts : ℕ, short_sleeve_shirts = total_shirts_had - total_shirts_washed - 1 :=
by
  sorry

end ned_shirts_problem_l196_196665


namespace least_possible_value_of_y_l196_196625

theorem least_possible_value_of_y
  (x y z : ℤ)
  (hx : Even x)
  (hy : Odd y)
  (hz : Odd z)
  (h1 : y - x > 5)
  (h2 : ∀ z', z' - x ≥ 9 → z' ≥ 9) :
  y ≥ 7 :=
by
  -- Proof is not required here
  sorry

end least_possible_value_of_y_l196_196625


namespace S_equals_l196_196079
noncomputable def S : Real :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_equals : S = 2 * Real.sqrt 23 - 2 :=
by
  sorry

end S_equals_l196_196079


namespace distance_to_fourth_side_l196_196175

theorem distance_to_fourth_side (s : ℕ) (d1 d2 d3 : ℕ) (x : ℕ) 
  (cond1 : d1 = 4) (cond2 : d2 = 7) (cond3 : d3 = 12)
  (h : d1 + d2 + d3 + x = s) : x = 9 ∨ x = 15 :=
  sorry

end distance_to_fourth_side_l196_196175


namespace find_percentage_l196_196065

theorem find_percentage (x p : ℝ) (h₀ : x = 780) (h₁ : 0.25 * x = (p / 100) * 1500 - 30) : p = 15 :=
by
  sorry

end find_percentage_l196_196065


namespace height_of_parallelogram_l196_196162

theorem height_of_parallelogram (Area Base : ℝ) (h1 : Area = 180) (h2 : Base = 18) : Area / Base = 10 :=
by
  sorry

end height_of_parallelogram_l196_196162


namespace initial_puppies_l196_196577

-- Definitions based on the conditions in the problem
def sold : ℕ := 21
def puppies_per_cage : ℕ := 9
def number_of_cages : ℕ := 9

-- The statement to prove
theorem initial_puppies : sold + (puppies_per_cage * number_of_cages) = 102 := by
  sorry

end initial_puppies_l196_196577


namespace inequality_proof_l196_196892

theorem inequality_proof (a b : Real) (h1 : (1 / a) < (1 / b)) (h2 : (1 / b) < 0) : 
  (b / a) + (a / b) > 2 :=
by
  sorry

end inequality_proof_l196_196892


namespace parabola_vertex_trajectory_eq_l196_196165

noncomputable def parabola_vertex_trajectory : Prop :=
  ∀ (m : ℝ), ∃ (x y : ℝ), (y = 2 * m) ∧ (x = -m^2) ∧ (y - 4 * x - 4 * m * y = 0)

theorem parabola_vertex_trajectory_eq :
  (∀ x y : ℝ, (∃ m : ℝ, y = 2 * m ∧ x = -m^2) → y^2 = -4 * x) :=
by
  sorry

end parabola_vertex_trajectory_eq_l196_196165


namespace wendys_brother_pieces_l196_196244

-- Definitions based on conditions
def number_of_boxes : ℕ := 2
def pieces_per_box : ℕ := 3
def total_pieces : ℕ := 12

-- Summarization of Wendy's pieces of candy
def wendys_pieces : ℕ := number_of_boxes * pieces_per_box

-- Lean statement: Prove the number of pieces Wendy's brother had
theorem wendys_brother_pieces : total_pieces - wendys_pieces = 6 :=
by
  sorry

end wendys_brother_pieces_l196_196244


namespace ratio_problem_l196_196764

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l196_196764


namespace sufficient_not_necessary_condition_l196_196250

open Complex

theorem sufficient_not_necessary_condition (a b : ℝ) (i := Complex.I) :
  (a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i) ∧ ¬((a + b * i)^2 = 2 * i → a = 1 ∧ b = 1) :=
by
  sorry

end sufficient_not_necessary_condition_l196_196250


namespace simplify_expression_l196_196095

-- Define the fractions involved
def frac1 : ℚ := 1 / 2
def frac2 : ℚ := 1 / 3
def frac3 : ℚ := 1 / 5
def frac4 : ℚ := 1 / 7

-- Define the expression to be simplified
def expr : ℚ := (frac1 - frac2 + frac3) / (frac2 - frac1 + frac4)

-- The goal is to show that the expression simplifies to -77 / 5
theorem simplify_expression : expr = -77 / 5 := by
  sorry

end simplify_expression_l196_196095


namespace hyperbola_equation_standard_form_l196_196927

noncomputable def point_on_hyperbola_asymptote (A : ℝ × ℝ) (C : ℝ) : Prop :=
  let x := A.1
  let y := A.2
  (4 * y^2 - x^2 = C) ∧
  (y = (1/2) * x ∨ y = -(1/2) * x)

theorem hyperbola_equation_standard_form
  (A : ℝ × ℝ)
  (hA : A = (2 * Real.sqrt 2, 2))
  (asymptote1 asymptote2 : ℝ → ℝ)
  (hasymptote1 : ∀ x, asymptote1 x = (1/2) * x)
  (hasymptote2 : ∀ x, asymptote2 x = -(1/2) * x) :
  (∃ C : ℝ, point_on_hyperbola_asymptote A C) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (4 * (A.2)^2 - (A.1)^2 = 8) ∧ 
    (∀ x y : ℝ, (4 * y^2 - x^2 = 8) ↔ ((y^2) / a - (x^2) / b = 1))) :=
by
  sorry

end hyperbola_equation_standard_form_l196_196927


namespace least_five_digit_congruent_to_7_mod_18_l196_196386

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 7 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < n → m % 18 ≠ 7 :=
by
  sorry

end least_five_digit_congruent_to_7_mod_18_l196_196386


namespace carrie_total_spend_l196_196415

def cost_per_tshirt : ℝ := 9.15
def number_of_tshirts : ℝ := 22

theorem carrie_total_spend : (cost_per_tshirt * number_of_tshirts) = 201.30 := by 
  sorry

end carrie_total_spend_l196_196415


namespace divide_circle_into_parts_l196_196598

theorem divide_circle_into_parts : 
    ∃ (divide : ℕ → ℕ), 
        (divide 3 = 4 ∧ divide 3 = 5 ∧ divide 3 = 6 ∧ divide 3 = 7) :=
by
  -- This illustrates that we require a proof to show that for 3 straight cuts ('n = 3'), 
  -- we can achieve 4, 5, 6, and 7 segments in different settings (circle with strategic line placements).
  sorry

end divide_circle_into_parts_l196_196598


namespace arithmetic_sequence_sum_square_l196_196690

theorem arithmetic_sequence_sum_square (a d : ℕ) :
  (∀ n : ℕ, ∃ k : ℕ, n * (a + (n-1) * d / 2) = k * k) ↔ (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) := 
by
  sorry

end arithmetic_sequence_sum_square_l196_196690


namespace sum_x_y_650_l196_196072

theorem sum_x_y_650 (x y : ℤ) (h1 : x - y = 200) (h2 : y = 225) : x + y = 650 :=
by
  sorry

end sum_x_y_650_l196_196072


namespace jodi_walks_days_l196_196652

section
variables {d : ℕ} -- d is the number of days Jodi walks per week

theorem jodi_walks_days (h : 1 * d + 2 * d + 3 * d + 4 * d = 60) : d = 6 := by
  sorry

end

end jodi_walks_days_l196_196652


namespace last_three_digits_2005_pow_2005_l196_196208

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_digits_2005_pow_2005 :
  last_three_digits (2005 ^ 2005) = 125 :=
sorry

end last_three_digits_2005_pow_2005_l196_196208


namespace members_didnt_show_up_l196_196710

theorem members_didnt_show_up (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  total_members = 14 →
  points_per_member = 5 →
  total_points = 35 →
  total_members - (total_points / points_per_member) = 7 :=
by
  intros
  sorry

end members_didnt_show_up_l196_196710


namespace power_function_passes_through_point_l196_196945

theorem power_function_passes_through_point (a : ℝ) : (2 ^ a = Real.sqrt 2) → (a = 1 / 2) :=
  by
  intro h
  sorry

end power_function_passes_through_point_l196_196945


namespace solve_for_x_l196_196804

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 8) * x = 14 ↔ x = 392 :=
by {
  sorry
}

end solve_for_x_l196_196804


namespace toy_production_difference_l196_196151

variables (w t : ℕ)
variable  (t_nonneg : 0 < t) -- assuming t is always non-negative for a valid working hour.
variable  (h : w = 3 * t)

theorem toy_production_difference : 
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end toy_production_difference_l196_196151


namespace log_xyz_eq_one_l196_196635

theorem log_xyz_eq_one {x y z : ℝ} (h1 : log (x^2 * y^2 * z) = 2) (h2 : log (x * y * z^3) = 2) :
  log (x * y * z) = 1 := by
  sorry

end log_xyz_eq_one_l196_196635


namespace Berengere_contribution_l196_196717

theorem Berengere_contribution (cake_cost_in_euros : ℝ) (emily_dollars : ℝ) (exchange_rate : ℝ)
  (h1 : cake_cost_in_euros = 6)
  (h2 : emily_dollars = 5)
  (h3 : exchange_rate = 1.25) :
  cake_cost_in_euros - emily_dollars * (1 / exchange_rate) = 2 := by
  sorry

end Berengere_contribution_l196_196717


namespace prime_diff_of_cubes_sum_of_square_and_three_times_square_l196_196799

theorem prime_diff_of_cubes_sum_of_square_and_three_times_square 
  (p : ℕ) (a b : ℕ) (h_prime : Nat.Prime p) (h_diff : p = a^3 - b^3) :
  ∃ c d : ℤ, p = c^2 + 3 * d^2 := 
  sorry

end prime_diff_of_cubes_sum_of_square_and_three_times_square_l196_196799


namespace calculate_pens_l196_196984

theorem calculate_pens (P : ℕ) (Students : ℕ) (Pencils : ℕ) (h1 : Students = 40) (h2 : Pencils = 920) (h3 : ∃ k : ℕ, Pencils = Students * k) 
(h4 : ∃ m : ℕ, P = Students * m) : ∃ k : ℕ, P = 40 * k := by
  sorry

end calculate_pens_l196_196984


namespace probability_of_selecting_one_of_each_color_l196_196611

noncomputable def number_of_ways_to_select_4_marbles_from_10 := Nat.choose 10 4
noncomputable def ways_to_select_1_red := Nat.choose 3 1
noncomputable def ways_to_select_1_blue := Nat.choose 3 1
noncomputable def ways_to_select_1_green := Nat.choose 2 1
noncomputable def ways_to_select_1_yellow := Nat.choose 2 1

theorem probability_of_selecting_one_of_each_color :
  (ways_to_select_1_red * ways_to_select_1_blue * ways_to_select_1_green * ways_to_select_1_yellow) / number_of_ways_to_select_4_marbles_from_10 = 6 / 35 :=
by
  sorry

end probability_of_selecting_one_of_each_color_l196_196611


namespace paris_total_study_hours_semester_l196_196540

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end paris_total_study_hours_semester_l196_196540


namespace largest_of_three_consecutive_integers_l196_196368

theorem largest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : x + 2 = 7 := 
sorry

end largest_of_three_consecutive_integers_l196_196368


namespace phoneExpences_l196_196891

structure PhonePlan where
  fixed_fee : ℝ
  free_minutes : ℕ
  excess_rate : ℝ -- rate per minute

def JanuaryUsage : ℕ := 15 * 60 + 17 -- 15 hours 17 minutes in minutes
def FebruaryUsage : ℕ := 9 * 60 + 55 -- 9 hours 55 minutes in minutes

def computeBill (plan : PhonePlan) (usage : ℕ) : ℝ :=
  let excess_minutes := (usage - plan.free_minutes).max 0
  plan.fixed_fee + (excess_minutes * plan.excess_rate)

theorem phoneExpences (plan : PhonePlan) :
  plan = { fixed_fee := 18.00, free_minutes := 600, excess_rate := 0.03 } →
  computeBill plan JanuaryUsage + computeBill plan FebruaryUsage = 45.51 := by
  sorry

end phoneExpences_l196_196891


namespace rope_length_91_4_l196_196238

noncomputable def total_rope_length (n : ℕ) (d : ℕ) (pi_val : Real) : Real :=
  let linear_segments := 6 * d
  let arc_length := (d * pi_val / 3) * 6
  let total_length_per_tie := linear_segments + arc_length
  total_length_per_tie * 2

theorem rope_length_91_4 :
  total_rope_length 7 5 3.14 = 91.4 :=
by
  sorry

end rope_length_91_4_l196_196238


namespace find_n_for_2013_in_expansion_l196_196092

/-- Define the pattern for the last term of the expansion of n^3 -/
def last_term (n : ℕ) : ℕ :=
  n^2 + n - 1

/-- The main problem statement -/
theorem find_n_for_2013_in_expansion :
  ∃ n : ℕ, last_term (n - 1) ≤ 2013 ∧ 2013 < last_term n ∧ n = 45 :=
by
  sorry

end find_n_for_2013_in_expansion_l196_196092


namespace minimum_value_expr_l196_196785

variable (a b : ℝ)

theorem minimum_value_expr (h1 : 0 < a) (h2 : 0 < b) : 
  (a + 1 / b) * (a + 1 / b - 1009) + (b + 1 / a) * (b + 1 / a - 1009) ≥ -509004.5 :=
sorry

end minimum_value_expr_l196_196785


namespace arithmetic_sequence_S7_eq_28_l196_196479

/--
Given the arithmetic sequence \( \{a_n\} \) and the sum of its first \( n \) terms is \( S_n \),
if \( a_3 + a_4 + a_5 = 12 \), then prove \( S_7 = 28 \).
-/
theorem arithmetic_sequence_S7_eq_28
  (a : ℕ → ℤ) -- Sequence a_n
  (S : ℕ → ℤ) -- Sum sequence S_n
  (h1 : a 3 + a 4 + a 5 = 12)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) -- Sum formula
  : S 7 = 28 :=
sorry

end arithmetic_sequence_S7_eq_28_l196_196479


namespace increasing_on_real_iff_a_range_l196_196059

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a / x

theorem increasing_on_real_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -3 ≤ a ∧ a ≤ -2 := 
by
  sorry

end increasing_on_real_iff_a_range_l196_196059


namespace solve_for_x_l196_196533

theorem solve_for_x (x : ℝ) (h : 10 - x = 15) : x = -5 :=
by
  sorry

end solve_for_x_l196_196533


namespace amy_lily_tie_probability_l196_196116

theorem amy_lily_tie_probability (P_Amy P_Lily : ℚ) (hAmy : P_Amy = 4/9) (hLily : P_Lily = 1/3) :
  1 - P_Amy - (↑P_Lily : ℚ) = 2 / 9 := by
  sorry

end amy_lily_tie_probability_l196_196116


namespace find_a_l196_196058

-- Define the domains of the functions f and g
def A : Set ℝ :=
  {x | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ :=
  {x | 2 * a < x ∧ x < a + 1}

-- Restate the problem as a Lean proposition
theorem find_a (a : ℝ) (h : a < 1) (hb : B a ⊆ A) :
  a ∈ {x | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
sorry

end find_a_l196_196058


namespace number_of_older_females_l196_196772

theorem number_of_older_females (total_population : ℕ) (num_groups : ℕ) (one_group_population : ℕ) :
  total_population = 1000 → num_groups = 5 → total_population = num_groups * one_group_population →
  one_group_population = 200 :=
by
  intro h1 h2 h3
  sorry

end number_of_older_females_l196_196772


namespace circle_value_of_m_l196_196538

theorem circle_value_of_m (m : ℝ) : (∃ a b r : ℝ, r > 0 ∧ (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2) ↔ m < 1/2 := by
  sorry

end circle_value_of_m_l196_196538


namespace avery_donates_16_clothes_l196_196023

theorem avery_donates_16_clothes : 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
in Shirts + Pants + Shorts = 16 :=
by 
  let Shirts := 4
  let Pants := 4 * 2
  let Shorts := (4 * 2) / 2
  show Shirts + Pants + Shorts = 16
  sorry

end avery_donates_16_clothes_l196_196023


namespace sam_money_left_l196_196528

theorem sam_money_left (initial_amount : ℕ) (book_cost : ℕ) (number_of_books : ℕ) (initial_amount_eq : initial_amount = 79) (book_cost_eq : book_cost = 7) (number_of_books_eq : number_of_books = 9) : initial_amount - book_cost * number_of_books = 16 :=
by
  rw [initial_amount_eq, book_cost_eq, number_of_books_eq]
  norm_num
  sorry

end sam_money_left_l196_196528


namespace no_exact_cover_l196_196725

theorem no_exact_cover (large_w : ℕ) (large_h : ℕ) (small_w : ℕ) (small_h : ℕ) (n : ℕ) :
  large_w = 13 → large_h = 7 → small_w = 2 → small_h = 3 → n = 15 →
  ¬ (small_w * small_h * n = large_w * large_h) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end no_exact_cover_l196_196725


namespace range_of_m_l196_196178

noncomputable def function_even_and_monotonic (f : ℝ → ℝ) := 
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x > f y)

variable (f : ℝ → ℝ)
variable (m : ℝ)

theorem range_of_m (h₁ : function_even_and_monotonic f) 
  (h₂ : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l196_196178


namespace hall_mat_expenditure_l196_196947

theorem hall_mat_expenditure
  (length width height cost_per_sq_meter : ℕ)
  (H_length : length = 20)
  (H_width : width = 15)
  (H_height : height = 5)
  (H_cost_per_sq_meter : cost_per_sq_meter = 50) :
  (2 * (length * width) + 2 * (length * height) + 2 * (width * height)) * cost_per_sq_meter = 47500 :=
by
  sorry

end hall_mat_expenditure_l196_196947


namespace cos_sum_of_arctan_roots_l196_196816

theorem cos_sum_of_arctan_roots (α β : ℝ) (hα : -π/2 < α ∧ α < 0) (hβ : -π/2 < β ∧ β < 0) 
  (h1 : Real.tan α + Real.tan β = -3 * Real.sqrt 3) 
  (h2 : Real.tan α * Real.tan β = 4) : 
  Real.cos (α + β) = - 1 / 2 :=
sorry

end cos_sum_of_arctan_roots_l196_196816


namespace divisibility_problem_l196_196639

theorem divisibility_problem (n : ℕ) : 2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := 
sorry

end divisibility_problem_l196_196639


namespace least_possible_value_of_y_l196_196623

theorem least_possible_value_of_y (x y z : ℤ) (hx : Even x) (hy : Odd y) (hz : Odd z) 
  (h1 : y - x > 5) (h2 : z - x ≥ 9) : y ≥ 7 :=
by {
  -- sorry allows us to skip the proof
  sorry
}

end least_possible_value_of_y_l196_196623


namespace flowers_in_each_basket_l196_196270

-- Definitions based on the conditions
def initial_flowers (d1 d2 : Nat) : Nat := d1 + d2
def grown_flowers (initial growth : Nat) : Nat := initial + growth
def remaining_flowers (grown dead : Nat) : Nat := grown - dead
def flowers_per_basket (remaining baskets : Nat) : Nat := remaining / baskets

-- Given conditions in Lean 4
theorem flowers_in_each_basket 
    (daughters_flowers : Nat) 
    (growth : Nat) 
    (dead : Nat) 
    (baskets : Nat) 
    (h_daughters : daughters_flowers = 5 + 5) 
    (h_growth : growth = 20) 
    (h_dead : dead = 10) 
    (h_baskets : baskets = 5) : 
    flowers_per_basket (remaining_flowers (grown_flowers (initial_flowers 5 5) growth) dead) baskets = 4 := 
sorry

end flowers_in_each_basket_l196_196270


namespace kittens_and_mice_count_l196_196780

theorem kittens_and_mice_count :
  let children := 12
  let baskets_per_child := 3
  let cats_per_basket := 1
  let kittens_per_cat := 12
  let mice_per_kitten := 4
  let total_kittens := children * baskets_per_child * cats_per_basket * kittens_per_cat
  let total_mice := total_kittens * mice_per_kitten
  total_kittens + total_mice = 2160 :=
by
  sorry

end kittens_and_mice_count_l196_196780


namespace michelle_scored_30_l196_196318

-- Define the total team points
def team_points : ℕ := 72

-- Define the number of other players
def num_other_players : ℕ := 7

-- Define the average points scored by the other players
def avg_points_other_players : ℕ := 6

-- Calculate the total points scored by the other players
def total_points_other_players : ℕ := num_other_players * avg_points_other_players

-- Define the points scored by Michelle
def michelle_points : ℕ := team_points - total_points_other_players

-- Prove that the points scored by Michelle is 30
theorem michelle_scored_30 : michelle_points = 30 :=
by
  -- Here would be the proof, but we skip it with sorry.
  sorry

end michelle_scored_30_l196_196318


namespace bird_average_l196_196518

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l196_196518


namespace final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l196_196117

-- Definitions of the driving records for trainee A and B
def driving_record_A : List Int := [15, -2, 5, -1, 10, -3, -2, 12, 4, -5, 6]
def driving_record_B : List Int := [-17, 9, -2, 8, 6, 9, -5, -1, 4, -7, -8]

-- Fuel consumption rate per kilometer
variable (a : ℝ)

-- Proof statements in Lean
theorem final_position_A : driving_record_A.sum = 39 := by sorry
theorem final_position_B : driving_record_B.sum = -4 := by sorry
theorem fuel_consumption_A : (driving_record_A.map (abs)).sum * a = 65 * a := by sorry
theorem fuel_consumption_B : (driving_record_B.map (abs)).sum * a = 76 * a := by sorry
theorem less_fuel_consumption : (driving_record_A.map (abs)).sum * a < (driving_record_B.map (abs)).sum * a := by sorry

end final_position_A_final_position_B_fuel_consumption_A_fuel_consumption_B_less_fuel_consumption_l196_196117


namespace parallelogram_proof_l196_196362

noncomputable def sin_angle_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem parallelogram_proof (x : ℝ) (A : ℝ) (r : ℝ) (side1 side2 : ℝ) (P : ℝ):
  (A = 972) → (r = 4 / 3) → (sin_angle_degrees 45 = Real.sqrt 2 / 2) →
  (side1 = 4 * x) → (side2 = 3 * x) →
  (A = side1 * (side2 * (Real.sqrt 2 / 2 / 3))) →
  x = 9 * 2^(3/4) →
  side1 = 36 * 2^(3/4) →
  side2 = 27 * 2^(3/4) →
  (P = 2 * (side1 + side2)) →
  (P = 126 * 2^(3/4)) :=
by
  intros
  sorry

end parallelogram_proof_l196_196362


namespace eq_expression_l196_196391

theorem eq_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
by
  sorry

end eq_expression_l196_196391


namespace largest_of_three_consecutive_integers_sum_18_l196_196366

theorem largest_of_three_consecutive_integers_sum_18 (n : ℤ) (h : n + (n + 1) + (n + 2) = 18) : n + 2 = 7 :=
by
  sorry

end largest_of_three_consecutive_integers_sum_18_l196_196366


namespace three_digit_numbers_divisible_by_11_are_550_or_803_l196_196727

theorem three_digit_numbers_divisible_by_11_are_550_or_803 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ ∃ (a b c : ℕ), N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 11 ∣ N ∧ (N / 11 = a^2 + b^2 + c^2)) → (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_divisible_by_11_are_550_or_803_l196_196727


namespace ab_sum_l196_196503

theorem ab_sum (A B C D : Nat) (h_digits: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_mult : A * (10 * C + D) = 1001 + 100 * A + 10 * B + A) : A + B = 1 := 
  sorry

end ab_sum_l196_196503


namespace container_fullness_calc_l196_196422

theorem container_fullness_calc (initial_percent : ℝ) (added_water : ℝ) (total_capacity : ℝ) (result_fraction : ℝ) :
  initial_percent = 0.3 →
  added_water = 27 →
  total_capacity = 60 →
  result_fraction = 3/4 →
  ((initial_percent * total_capacity + added_water) / total_capacity) = result_fraction :=
by
  intros h1 h2 h3 h4
  sorry

end container_fullness_calc_l196_196422


namespace valid_lineups_count_l196_196349

-- Definitions of the problem conditions
def num_players : ℕ := 18
def quadruplets : Finset ℕ := {0, 1, 2, 3} -- Indices of Benjamin, Brenda, Brittany, Bryan
def total_starters : ℕ := 8

-- Function to count lineups based on given constraints
noncomputable def count_valid_lineups : ℕ :=
  let others := num_players - quadruplets.card
  Nat.choose others total_starters + quadruplets.card * Nat.choose others (total_starters - 1)

-- The theorem to prove the count of valid lineups
theorem valid_lineups_count : count_valid_lineups = 16731 := by
  -- Placeholder for the actual proof
  sorry

end valid_lineups_count_l196_196349


namespace find_E_l196_196476

theorem find_E (A H S M E : ℕ) (h1 : A ≠ 0) (h2 : H ≠ 0) (h3 : S ≠ 0) (h4 : M ≠ 0) (h5 : E ≠ 0) 
  (cond1 : A + H = E)
  (cond2 : S + M = E)
  (cond3 : E = (A * M - S * H) / (M - H)) : 
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end find_E_l196_196476


namespace ferris_wheel_seats_l196_196977

theorem ferris_wheel_seats (S : ℕ) (h1 : ∀ (p : ℕ), p = 9) (h2 : ∀ (r : ℕ), r = 18) (h3 : 9 * S = 18) : S = 2 :=
by
  sorry

end ferris_wheel_seats_l196_196977


namespace imperative_sentence_structure_l196_196694

theorem imperative_sentence_structure (word : String) (is_base_form : word = "Surround") :
  (word = "Surround" ∨ word = "Surrounding" ∨ word = "Surrounded" ∨ word = "Have surrounded") →
  (∃ sentence : String, sentence = word ++ " yourself with positive people, and you will keep focused on what you can do instead of what you can’t.") →
  word = "Surround" :=
by
  intros H_choice H_sentence
  cases H_choice
  case inl H1 => assumption
  case inr H2_1 =>
    cases H2_1
    case inl H2_1_1 => sorry
    case inr H2_1_2 =>
      cases H2_1_2
      case inl H2_1_2_1 => sorry
      case inr H2_1_2_2 => sorry

end imperative_sentence_structure_l196_196694


namespace original_average_is_6_2_l196_196865

theorem original_average_is_6_2 (n : ℕ) (S : ℚ) (h1 : 6.2 = S / n) (h2 : 6.6 = (S + 4) / n) :
  6.2 = S / n :=
by
  sorry

end original_average_is_6_2_l196_196865


namespace ratio_of_area_l196_196776

noncomputable def area_of_triangle_ratio (AB CD height : ℝ) (h : CD = 2 * AB) : ℝ :=
  let ABCD_area := (AB + CD) * height / 2
  let EAB_area := ABCD_area / 3
  EAB_area / ABCD_area

theorem ratio_of_area (AB CD : ℝ) (height : ℝ) (h1 : AB = 10) (h2 : CD = 20) (h3 : height = 5) : 
  area_of_triangle_ratio AB CD height (by rw [h1, h2]; ring) = 1 / 3 :=
sorry

end ratio_of_area_l196_196776


namespace area_of_triangle_formed_by_tangent_line_l196_196757
-- Import necessary libraries from Mathlib

-- Set up the problem
theorem area_of_triangle_formed_by_tangent_line
  (f : ℝ → ℝ) (h_f : ∀ x, f x = x^2) :
  let slope := (deriv f 1)
  let tangent_line (x : ℝ) := slope * (x - 1) + f 1
  let x_intercept := (0 : ℝ)
  let y_intercept := tangent_line 0
  let area := 0.5 * abs x_intercept * abs y_intercept
  area = 1 / 4 :=
by
  sorry -- Proof to be completed

end area_of_triangle_formed_by_tangent_line_l196_196757


namespace lost_card_number_l196_196406

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196406


namespace problem_1_l196_196249

theorem problem_1 (α : ℝ) (k : ℤ) (n : ℕ) (hk : k > 0) (hα : α ≠ k * Real.pi) (hn : n > 0) :
  n = 1 → (0.5 + Real.cos α) = (0.5 + Real.cos α) :=
by
  sorry

end problem_1_l196_196249


namespace compare_neg_fractions_l196_196155

theorem compare_neg_fractions : (-3 / 4) > (-5 / 6) :=
sorry

end compare_neg_fractions_l196_196155


namespace quadratic_has_negative_root_condition_l196_196039

theorem quadratic_has_negative_root_condition (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, ax^2 + 2*x + 1 = 0 ∧ x < 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end quadratic_has_negative_root_condition_l196_196039


namespace m_ducks_l196_196237

variable (M C K : ℕ)

theorem m_ducks :
  (M = C + 4) ∧
  (M = 2 * C + K + 3) ∧
  (M + C + K = 90) →
  M = 89 := by
  sorry

end m_ducks_l196_196237


namespace lost_card_number_l196_196411

variable (n : ℕ)
variable (s : ℕ)

-- Axioms based on the given conditions.
axiom sum_of_first_n : s = n * (n + 1) / 2
axiom remaining_sum : s - 101 ∈ finset.singleton 101

-- The theorem we need to prove:
theorem lost_card_number (n : ℕ) (s : ℕ) (h₁ : s = n * (n + 1) / 2) (h₂ : s - 101 = 101) : n = 14 → (n * (n + 1) / 2 - 101 = 4) :=
by
  sorry

end lost_card_number_l196_196411


namespace intersection_M_N_l196_196917

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l196_196917


namespace exists_strictly_increasing_sequences_l196_196530

theorem exists_strictly_increasing_sequences :
  ∃ u v : ℕ → ℕ, (∀ n, u n < u (n + 1)) ∧ (∀ n, v n < v (n + 1)) ∧ (∀ n, 5 * u n * (u n + 1) = v n ^ 2 + 1) :=
sorry

end exists_strictly_increasing_sequences_l196_196530


namespace circle_integer_solution_max_sum_l196_196574

theorem circle_integer_solution_max_sum : ∀ (x y : ℤ), (x - 1)^2 + (y + 2)^2 = 16 → x + y ≤ 3 :=
by
  sorry

end circle_integer_solution_max_sum_l196_196574


namespace floor_sqrt_77_l196_196881

theorem floor_sqrt_77 : 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 → Int.floor (Real.sqrt 77) = 8 :=
by
  sorry

end floor_sqrt_77_l196_196881


namespace inequality_xyz_l196_196086

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 1/x + 1/y + 1/z = 3) : 
  (x - 1) * (y - 1) * (z - 1) ≤ (1/4) * (x * y * z - 1) := 
by 
  sorry

end inequality_xyz_l196_196086


namespace M_inter_N_eq_neg2_l196_196897

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l196_196897


namespace simplify_expression_l196_196531

theorem simplify_expression (a c d x y z : ℝ) :
  (cx * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + dz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cx + dz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cx * a^3 * y^3 / (cx + dz)) + (3 * dz * c^3 * x^3 / (cx + dz)) :=
by
  sorry

end simplify_expression_l196_196531


namespace arithmetic_sequence_S30_l196_196499

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l196_196499


namespace find_second_sum_l196_196849

def sum : ℕ := 2717
def interest_rate_first : ℚ := 3 / 100
def interest_rate_second : ℚ := 5 / 100
def time_first : ℕ := 8
def time_second : ℕ := 3

theorem find_second_sum (x : ℚ) (h : x * interest_rate_first * time_first = (sum - x) * interest_rate_second * time_second) : 
  sum - x = 2449 :=
by
  sorry

end find_second_sum_l196_196849


namespace similar_triangles_height_l196_196560

theorem similar_triangles_height (h₁ h₂ : ℝ) (a₁ a₂ : ℝ) 
  (ratio_area : a₁ / a₂ = 1 / 9) (height_small : h₁ = 4) :
  h₂ = 12 :=
sorry

end similar_triangles_height_l196_196560


namespace sum_of_digits_a_l196_196640

def a : ℕ := 10^10 - 47

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_a : sum_of_digits a = 81 := 
  by 
    sorry

end sum_of_digits_a_l196_196640


namespace sum_of_x_coordinates_l196_196667

def exists_common_point (x : ℕ) : Prop :=
  (3 * x + 5) % 9 = (7 * x + 3) % 9

theorem sum_of_x_coordinates :
  ∃ x : ℕ, exists_common_point x ∧ x % 9 = 5 := 
by
  sorry

end sum_of_x_coordinates_l196_196667


namespace bill_donuts_combinations_l196_196152

theorem bill_donuts_combinations :
  let kinds := 4 in
  let total_donuts := 7 in
  let required_each_kind := 1 in
  let remaining_donuts := total_donuts - kinds * required_each_kind in
  (remaining_donuts = 3) →
  (kinds = 4) →
  (finset.card (finset.powerset_len remaining_donuts (finset.range (remaining_donuts + kinds - 1))) = 20) :=
by
  intros
  sorry

end bill_donuts_combinations_l196_196152


namespace cameron_answers_l196_196448

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end cameron_answers_l196_196448


namespace value_of_ab_l196_196768

theorem value_of_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : ab = 8 :=
by
  sorry

end value_of_ab_l196_196768


namespace lemons_left_l196_196590

/--
Prove that Cristine has 9 lemons left, given that she initially bought 12 lemons and gave away 1/4 of them.
-/
theorem lemons_left {initial_lemons : ℕ} (h1 : initial_lemons = 12) (fraction_given : ℚ) (h2 : fraction_given = 1 / 4) : initial_lemons - initial_lemons * fraction_given = 9 := by
  sorry

end lemons_left_l196_196590


namespace paint_fraction_used_l196_196653

theorem paint_fraction_used (initial_paint: ℕ) (first_week_fraction: ℚ) (total_paint_used: ℕ) (remaining_paint_after_first_week: ℕ) :
  initial_paint = 360 →
  first_week_fraction = 1/3 →
  total_paint_used = 168 →
  remaining_paint_after_first_week = initial_paint - initial_paint * first_week_fraction →
  (total_paint_used - initial_paint * first_week_fraction) / remaining_paint_after_first_week = 1/5 := 
by
  sorry

end paint_fraction_used_l196_196653


namespace figure_Z_has_largest_shaded_area_l196_196379

noncomputable def shaded_area_X :=
  let rectangle_area := 4 * 2
  let circle_area := Real.pi * (1)^2
  rectangle_area - circle_area

noncomputable def shaded_area_Y :=
  let rectangle_area := 4 * 2
  let semicircle_area := (1 / 2) * Real.pi * (1)^2
  rectangle_area - semicircle_area

noncomputable def shaded_area_Z :=
  let outer_square_area := 4^2
  let inner_square_area := 2^2
  outer_square_area - inner_square_area

theorem figure_Z_has_largest_shaded_area :
  shaded_area_Z > shaded_area_X ∧ shaded_area_Z > shaded_area_Y :=
by
  sorry

end figure_Z_has_largest_shaded_area_l196_196379


namespace floor_sum_lemma_l196_196974

theorem floor_sum_lemma (x : Fin 1004 → ℝ) 
  (h : ∀ n : Fin 1004, x n + (n : ℝ) + 1 = ∑ i : Fin 1004, x i + 1005) 
  : ⌊|∑ i : Fin 1004, x i|⌋ = 501 :=
sorry

end floor_sum_lemma_l196_196974


namespace intersection_M_N_l196_196902

open Set

def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = {-2} :=
sorry

end intersection_M_N_l196_196902


namespace sandbag_weight_l196_196380

theorem sandbag_weight (s : ℝ) (f : ℝ) (h : ℝ) : 
  f = 0.75 ∧ s = 450 ∧ h = 0.65 → f * s + h * (f * s) = 556.875 :=
by
  intro hfs
  sorry

end sandbag_weight_l196_196380


namespace minimal_degree_g_l196_196626

theorem minimal_degree_g {f g h : Polynomial ℝ} 
  (h_eq : 2 * f + 5 * g = h)
  (deg_f : f.degree = 6)
  (deg_h : h.degree = 10) : 
  g.degree = 10 :=
sorry

end minimal_degree_g_l196_196626


namespace JuliaPlayedTuesday_l196_196205

variable (Monday : ℕ) (Wednesday : ℕ) (Total : ℕ)
variable (KidsOnTuesday : ℕ)

theorem JuliaPlayedTuesday :
  Monday = 17 →
  Wednesday = 2 →
  Total = 34 →
  KidsOnTuesday = Total - (Monday + Wednesday) →
  KidsOnTuesday = 15 :=
by
  intros hMon hWed hTot hTue
  rw [hTot, hMon, hWed] at hTue
  exact hTue

end JuliaPlayedTuesday_l196_196205


namespace inequality_proof_l196_196525

theorem inequality_proof (x y z : ℝ) (hx : 2 < x) (hx4 : x < 4) (hy : 2 < y) (hy4 : y < 4) (hz : 2 < z) (hz4 : z < 4) :
  (x / (y^2 - z) + y / (z^2 - x) + z / (x^2 - y)) > 1 :=
by
  sorry

end inequality_proof_l196_196525


namespace Kyle_papers_delivered_each_week_proof_l196_196339

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l196_196339


namespace tile_C_is_TileIV_l196_196824

-- Define the tiles with their respective sides
structure Tile :=
(top right bottom left : ℕ)

def TileI : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileII : Tile := { top := 6, right := 3, bottom := 1, left := 5 }
def TileIII : Tile := { top := 5, right := 7, bottom := 2, left := 3 }
def TileIV : Tile := { top := 3, right := 5, bottom := 7, left := 2 }

-- Define Rectangles for reasoning
inductive Rectangle
| A
| B
| C
| D

open Rectangle

-- Define the mathematical statement to prove
theorem tile_C_is_TileIV : ∃ tile, tile = TileIV :=
  sorry

end tile_C_is_TileIV_l196_196824


namespace gcd_lcm_45_150_l196_196982

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 :=
by
  sorry

end gcd_lcm_45_150_l196_196982


namespace prime_gt_10_exists_m_n_l196_196085

theorem prime_gt_10_exists_m_n (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m + n < p ∧ p ∣ (5^m * 7^n - 1) :=
by
  sorry

end prime_gt_10_exists_m_n_l196_196085


namespace equivalent_expression_l196_196389

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l196_196389


namespace scientific_notation_l196_196952

def given_number : ℝ := 632000

theorem scientific_notation : given_number = 6.32 * 10^5 :=
by sorry

end scientific_notation_l196_196952


namespace linear_function_m_value_l196_196489

theorem linear_function_m_value (m : ℝ) (h : abs (m + 1) = 1) : m = -2 :=
sorry

end linear_function_m_value_l196_196489


namespace distinct_real_c_f_ff_ff_five_l196_196212

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_c_f_ff_ff_five : 
  (∀ c : ℝ, f (f (f (f c))) = 5 → False) :=
by
  sorry

end distinct_real_c_f_ff_ff_five_l196_196212


namespace lost_card_number_l196_196408

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196408


namespace sum_opposite_abs_val_eq_neg_nine_l196_196989

theorem sum_opposite_abs_val_eq_neg_nine (a b : ℤ) (h1 : a = -15) (h2 : b = 6) : a + b = -9 := 
by
  -- conditions given
  rw [h1, h2]
  -- skip the proof
  sorry

end sum_opposite_abs_val_eq_neg_nine_l196_196989


namespace least_possible_value_of_y_l196_196624

theorem least_possible_value_of_y
  (x y z : ℤ)
  (hx : Even x)
  (hy : Odd y)
  (hz : Odd z)
  (h1 : y - x > 5)
  (h2 : ∀ z', z' - x ≥ 9 → z' ≥ 9) :
  y ≥ 7 :=
by
  -- Proof is not required here
  sorry

end least_possible_value_of_y_l196_196624


namespace base3_addition_proof_l196_196869

-- Define the base 3 numbers
def one_3 : ℕ := 1
def twelve_3 : ℕ := 1 * 3 + 2
def two_hundred_twelve_3 : ℕ := 2 * 3^2 + 1 * 3 + 2
def two_thousand_one_hundred_twenty_one_3 : ℕ := 2 * 3^3 + 1 * 3^2 + 2 * 3 + 1

-- Define the correct answer in base 3
def expected_sum_3 : ℕ := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3 + 0

-- The proof problem
theorem base3_addition_proof :
  one_3 + twelve_3 + two_hundred_twelve_3 + two_thousand_one_hundred_twenty_one_3 = expected_sum_3 :=
by
  -- Proof goes here
  sorry

end base3_addition_proof_l196_196869


namespace possible_values_l196_196459

def expression (m n : ℕ) : ℤ :=
  (m^2 + m * n + n^2) / (m * n - 1)

theorem possible_values (m n : ℕ) (h : m * n ≠ 1) : 
  ∃ (N : ℤ), N = expression m n → N = 0 ∨ N = 4 ∨ N = 7 :=
by
  sorry

end possible_values_l196_196459


namespace roman_remy_gallons_l196_196671

theorem roman_remy_gallons (R : ℕ) (Remy_uses : 3 * R + 1 = 25) :
  R + (3 * R + 1) = 33 :=
by
  sorry

end roman_remy_gallons_l196_196671


namespace cube_root_approx_l196_196979

open Classical

theorem cube_root_approx (n : ℤ) (x : ℝ) (h₁ : 2^n = x^3) (h₂ : abs (x - 50) <  1) : n = 17 := by
  sorry

end cube_root_approx_l196_196979


namespace intersection_M_N_eq_neg2_l196_196912

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l196_196912


namespace interest_difference_l196_196134

theorem interest_difference (P R T: ℝ) (hP: P = 2500) (hR: R = 8) (hT: T = 8) :
  let I := P * R * T / 100
  (P - I = 900) :=
by
  -- definition of I
  let I := P * R * T / 100
  -- proof goal
  sorry

end interest_difference_l196_196134


namespace distance_from_A_to_origin_l196_196308

open Real

theorem distance_from_A_to_origin 
  (x1 y1 : ℝ)
  (hx1 : y1^2 = 4 * x1)
  (hratio : (x1 + 1) / abs y1 = 5 / 4)
  (hAF_gt_2 : dist (x1, y1) (1, 0) > 2) : 
  dist (x1, y1) (0, 0) = 4 * sqrt 2 :=
sorry

end distance_from_A_to_origin_l196_196308


namespace LeibnizTriangleElement_l196_196247

theorem LeibnizTriangleElement (n k : ℕ) :
  L n k = 1 / ((n + 1) * nat.choose n k) := 
sorry

end LeibnizTriangleElement_l196_196247


namespace sum_of_solutions_l196_196836

theorem sum_of_solutions : ∀ x : ℚ, (4 * x + 6) * (3 * x - 8) = 0 → 
  (x = -3 / 2 ∨ x = 8 / 3) → 
  (-3 / 2 + 8 / 3) = 7 / 6 :=
by
  intros x h1 h2
  sorry

end sum_of_solutions_l196_196836


namespace friendship_groups_ways_l196_196034

noncomputable def count_friendship_configurations : Nat :=
  let n : Nat := 8
  let pairs := n.choose 2  -- Number of ways to choose 2 out of 8.
  pairs / 2  -- Dividing by 2 because each pair is counted twice.
  
theorem friendship_groups_ways :
  count_friendship_configurations = 210 := by
  sorry

end friendship_groups_ways_l196_196034


namespace mumu_identity_l196_196684

def f (m u : ℕ) : ℕ := 
  -- Assume f is correctly defined to match the number of valid Mumu words 
  -- involving m M's and u U's according to the problem's definition.
  sorry 

theorem mumu_identity (u m : ℕ) (h₁ : u ≥ 2) (h₂ : 3 ≤ m) (h₃ : m ≤ 2 * u) :
  f m u = f (2 * u - m + 1) u ↔ f m (u - 1) = f (2 * u - m + 1) (u - 1) :=
by
  sorry

end mumu_identity_l196_196684


namespace number_of_birds_seen_l196_196672

theorem number_of_birds_seen (dozens_seen : ℕ) (birds_per_dozen : ℕ) (h₀ : dozens_seen = 8) (h₁ : birds_per_dozen = 12) : dozens_seen * birds_per_dozen = 96 :=
by sorry

end number_of_birds_seen_l196_196672


namespace decimal_to_fraction_l196_196600

theorem decimal_to_fraction : (0.3 + (0.24 - 0.24 / 100)) = (19 / 33) :=
by
  sorry

end decimal_to_fraction_l196_196600


namespace positive_integer_solutions_of_inequality_l196_196111

theorem positive_integer_solutions_of_inequality : 
  {x : ℕ | 3 * x - 1 ≤ 2 * x + 3} = {1, 2, 3, 4} :=
by
  sorry

end positive_integer_solutions_of_inequality_l196_196111


namespace min_time_adult_worms_l196_196146

noncomputable def f : ℕ → ℝ
| 1 => 0
| n => (1 - 1 / (2 ^ (n - 1)))

theorem min_time_adult_worms (n : ℕ) (h : n ≥ 1) : 
  ∃ min_time : ℝ, 
  (min_time = 1 - 1 / (2 ^ (n - 1))) ∧ 
  (∀ t : ℝ, (t = 1 - 1 / (2 ^ (n - 1)))) := 
sorry

end min_time_adult_worms_l196_196146


namespace sector_angle_l196_196056

theorem sector_angle (R : ℝ) (S : ℝ) (α : ℝ) (hR : R = 2) (hS : S = 8) : 
  α = 4 := by
  sorry

end sector_angle_l196_196056


namespace length_of_road_l196_196976

-- Definitions based on conditions
def trees : Nat := 10
def interval : Nat := 10

-- Statement of the theorem
theorem length_of_road 
  (trees : Nat) (interval : Nat) (beginning_planting : Bool) (h_trees : trees = 10) (h_interval : interval = 10) (h_beginning : beginning_planting = true) 
  : (trees - 1) * interval = 90 := 
by 
  sorry

end length_of_road_l196_196976


namespace problem_statement_l196_196301

theorem problem_statement (a b : ℕ) (m n : ℕ)
  (h1 : 32 + (2 / 7 : ℝ) = 3 * (2 / 7 : ℝ))
  (h2 : 33 + (3 / 26 : ℝ) = 3 * (3 / 26 : ℝ))
  (h3 : 34 + (4 / 63 : ℝ) = 3 * (4 / 63 : ℝ))
  (h4 : 32014 + (m / n : ℝ) = 2014 * 3 * (m / n : ℝ))
  (h5 : 32016 + (a / b : ℝ) = 2016 * 3 * (a / b : ℝ)) :
  (b + 1) / (a * a) = 2016 :=
sorry

end problem_statement_l196_196301


namespace vasya_wins_game_l196_196612

/- Define the conditions of the problem -/

def grid_size : Nat := 9
def total_matchsticks : Nat := 2 * grid_size * (grid_size + 1)

/-- Given a game on a 9x9 matchstick grid with Petya going first, 
    Prove that Vasya can always win by ensuring that no whole 1x1 
    squares remain in the end. -/
theorem vasya_wins_game : 
  ∃ strategy_for_vasya : Nat → Nat → Prop, -- Define a strategy for Vasya
  ∀ (matchsticks_left : Nat),
  matchsticks_left % 2 = 1 →     -- Petya makes a move and the remaining matchsticks are odd
  strategy_for_vasya matchsticks_left total_matchsticks :=
sorry

end vasya_wins_game_l196_196612


namespace sufficient_but_not_necessary_l196_196679

variable (x : ℚ)

def is_integer (n : ℚ) : Prop := ∃ (k : ℤ), n = k

theorem sufficient_but_not_necessary :
  (is_integer x → is_integer (2 * x + 1)) ∧
  (¬ (is_integer (2 * x + 1) → is_integer x)) :=
by
  sorry

end sufficient_but_not_necessary_l196_196679


namespace find_S30_l196_196501

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l196_196501


namespace JacobProof_l196_196504

def JacobLadders : Prop :=
  let costPerRung : ℤ := 2
  let costPer50RungLadder : ℤ := 50 * costPerRung
  let num50RungLadders : ℤ := 10
  let totalPayment : ℤ := 3400
  let cost1 : ℤ := num50RungLadders * costPer50RungLadder
  let remainingAmount : ℤ := totalPayment - cost1
  let numRungs20Ladders : ℤ := remainingAmount / costPerRung
  numRungs20Ladders = 1200

theorem JacobProof : JacobLadders := by
  sorry

end JacobProof_l196_196504


namespace dice_number_divisible_by_7_l196_196532

theorem dice_number_divisible_by_7 :
  ∃ a b c : ℕ, (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) 
               ∧ (1001 * (100 * a + 10 * b + c)) % 7 = 0 :=
by
  sorry

end dice_number_divisible_by_7_l196_196532


namespace compare_a_b_l196_196173

def a := 1 / 3 + 1 / 4
def b := 1 / 5 + 1 / 6 + 1 / 7

theorem compare_a_b : a > b := 
  sorry

end compare_a_b_l196_196173


namespace infinite_geometric_series_sum_l196_196279

theorem infinite_geometric_series_sum :
  let a := (4 : ℚ) / 3
  let r := -(9 : ℚ) / 16
  (a / (1 - r)) = (64 : ℚ) / 75 :=
by
  sorry

end infinite_geometric_series_sum_l196_196279


namespace circle_Γ_contains_exactly_one_l196_196649

-- Condition definitions
variables (z1 z2 : ℂ) (Γ : ℂ → ℂ → Prop)
variable (hz1z2 : z1 * z2 = 1)
variable (hΓ_passes : Γ (-1) 1)
variable (hΓ_not_passes : ¬Γ z1 z2)

-- Math proof problem
theorem circle_Γ_contains_exactly_one (hz1z2 : z1 * z2 = 1)
    (hΓ_passes : Γ (-1) 1) (hΓ_not_passes : ¬Γ z1 z2) : 
  (Γ 0 z1 ↔ ¬Γ 0 z2) ∨ (Γ 0 z2 ↔ ¬Γ 0 z1) :=
sorry

end circle_Γ_contains_exactly_one_l196_196649


namespace ellens_initial_legos_l196_196276

-- Define the initial number of Legos as a proof goal
theorem ellens_initial_legos : ∀ (x y : ℕ), (y = x - 17) → (x = 2080) :=
by
  intros x y h
  sorry

end ellens_initial_legos_l196_196276


namespace find_k_value_l196_196068

-- Define the condition that point A(3, -5) lies on the graph of the function y = k / x
def point_on_inverse_proportion (k : ℝ) : Prop :=
  (3 : ℝ) ≠ 0 ∧ (-5) = k / (3 : ℝ)

-- The theorem to prove that k = -15 given the point on the graph
theorem find_k_value (k : ℝ) (h : point_on_inverse_proportion k) : k = -15 :=
by
  sorry

end find_k_value_l196_196068


namespace simplify_product_l196_196971

theorem simplify_product : 
  18 * (8 / 15) * (2 / 27) = 32 / 45 :=
by
  sorry

end simplify_product_l196_196971


namespace quotient_of_division_l196_196794

theorem quotient_of_division 
  (dividend divisor remainder : ℕ) 
  (h_dividend : dividend = 265) 
  (h_divisor : divisor = 22) 
  (h_remainder : remainder = 1) 
  (h_div : dividend = divisor * (dividend / divisor) + remainder) : 
  (dividend / divisor) = 12 := 
by
  sorry

end quotient_of_division_l196_196794


namespace extra_charge_per_wand_l196_196123

theorem extra_charge_per_wand
  (cost_per_wand : ℕ)
  (num_wands : ℕ)
  (total_collected : ℕ)
  (num_wands_sold : ℕ)
  (h_cost : cost_per_wand = 60)
  (h_num_wands : num_wands = 3)
  (h_total_collected : total_collected = 130)
  (h_num_wands_sold : num_wands_sold = 2) :
  ((total_collected / num_wands_sold) - cost_per_wand) = 5 :=
by
  -- Proof goes here
  sorry

end extra_charge_per_wand_l196_196123


namespace range_of_x_when_a_equals_1_range_of_a_l196_196784

variable {a x : ℝ}

-- Definitions for conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part (1): Prove the range of x when a = 1 and p ∨ q is true.
theorem range_of_x_when_a_equals_1 (h : a = 1) (h1 : p 1 x ∨ q x) : 1 < x ∧ x < 3 :=
by sorry

-- Part (2): Prove the range of a when p is a necessary but not sufficient condition for q.
theorem range_of_a (h2 : ∀ x, q x → p a x) (h3 : ¬ ∀ x, p a x → q x) : 1 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_x_when_a_equals_1_range_of_a_l196_196784


namespace total_movies_shown_l196_196005

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l196_196005


namespace inequality_proof_l196_196055

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hSum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1 / 2 :=
by
  sorry

end inequality_proof_l196_196055


namespace cotangent_positives_among_sequence_l196_196182

def cotangent_positive_count (n : ℕ) : ℕ :=
  if n ≤ 2019 then
    let count := (n / 4) * 3 + if n % 4 ≠ 0 then (3 + 1 - max 0 ((n % 4) - 1)) else 0
    count
  else 0

theorem cotangent_positives_among_sequence :
  cotangent_positive_count 2019 = 1515 := sorry

end cotangent_positives_among_sequence_l196_196182


namespace polygon_is_octahedron_l196_196478

theorem polygon_is_octahedron (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_is_octahedron_l196_196478


namespace complement_of_A_in_U_l196_196172

open Set

-- Define the sets U and A with their respective elements in the real numbers
def U : Set ℝ := Icc 0 1
def A : Set ℝ := Ico 0 1

-- State the theorem
theorem complement_of_A_in_U : (U \ A) = {1} := by
  sorry

end complement_of_A_in_U_l196_196172


namespace stratified_sampling_elderly_employees_l196_196254

-- Definitions for the conditions
def total_employees : ℕ := 430
def young_employees : ℕ := 160
def middle_aged_employees : ℕ := 180
def elderly_employees : ℕ := 90
def sample_young_employees : ℕ := 32

-- The property we want to prove
theorem stratified_sampling_elderly_employees :
  (sample_young_employees / young_employees) * elderly_employees = 18 :=
by
  sorry

end stratified_sampling_elderly_employees_l196_196254


namespace king_chessboard_strategy_king_chessboard_strategy_odd_l196_196138

theorem king_chessboard_strategy (m n : ℕ) : 
  (m * n) % 2 = 0 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) := 
sorry

theorem king_chessboard_strategy_odd (m n : ℕ) : 
  (m * n) % 2 = 1 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) :=
sorry

end king_chessboard_strategy_king_chessboard_strategy_odd_l196_196138


namespace geometric_sequence_sum_l196_196475

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r)
    (h2 : r = 2) (h3 : a 1 * 2 + a 3 * 8 + a 5 * 32 = 3) :
    a 4 * 16 + a 6 * 64 + a 8 * 256 = 24 :=
sorry

end geometric_sequence_sum_l196_196475


namespace percentage_reduction_correct_l196_196234

-- Define the initial conditions
def initial_conditions (P S : ℝ) (new_sales_increase_percentage net_sale_value_increase_percentage: ℝ) :=
  new_sales_increase_percentage = 0.72 ∧ net_sale_value_increase_percentage = 0.4104

-- Define the statement for the required percentage reduction
theorem percentage_reduction_correct (P S : ℝ) (x : ℝ) 
  (h : initial_conditions P S 0.72 0.4104) : 
  (S:ℝ) * (1 - x / 100) = 1.4104 * S := 
sorry

end percentage_reduction_correct_l196_196234


namespace ratio_volumes_tetrahedron_octahedron_l196_196740

theorem ratio_volumes_tetrahedron_octahedron (a b : ℝ) (h_eq_areas : a^2 * (Real.sqrt 3) = 2 * b^2 * (Real.sqrt 3)) :
  (a^3 * (Real.sqrt 2) / 12) / (b^3 * (Real.sqrt 2) / 3) = 1 / Real.sqrt 2 :=
by
  sorry

end ratio_volumes_tetrahedron_octahedron_l196_196740


namespace interval_x_2x_3x_l196_196888

theorem interval_x_2x_3x (x : ℝ) :
  (2 * x > 1) ∧ (2 * x < 2) ∧ (3 * x > 1) ∧ (3 * x < 2) ↔ (x > 1 / 2) ∧ (x < 2 / 3) :=
by
  sorry

end interval_x_2x_3x_l196_196888


namespace integral_fx_l196_196655

noncomputable def f : ℝ → ℝ := 
λ x, if (0 ≤ x ∧ x < 1) then x ^ 2 else if (1 < x ∧ x ≤ 2) then 2 - x else 0

theorem integral_fx : ∫ x in 0..2, f x = 5 / 6 :=
by {
  let g : ℝ → ℝ := λ x, if 0 ≤ x ∧ x < 1 then x ^ 2 else if 1 < x ∧ x ≤ 2 then 2 - x else 0,
  have h_g_eq_f : ∀ x, g x = f x := by simp [f, g],
  simp_rw [← h_g_eq_f],
  -- Then integrate g instead of f
  convert_to (∫ x in 0..1, x ^ 2 + ∫ x in 1..2, 2 - x = _) using 1,
  { apply interval_integrable.integral_add (interval_integrable_iff.mpr _),
    simp [g, interval_integrable, continuous_on] with integrable_simp }
  -- Then calculate the integral accordingly
  apply_dvision_partition,
  interval_integral, 
  sorry -- completes the proof accordingly
}

end integral_fx_l196_196655


namespace volume_of_regular_tetrahedron_with_edge_length_1_l196_196841

-- We define the concepts needed: regular tetrahedron, edge length, and volume.
open Real

noncomputable def volume_of_regular_tetrahedron (a : ℝ) : ℝ :=
  let base_area := (sqrt 3 / 4) * a^2
  let height := sqrt (a^2 - (a * (sqrt 3 / 3))^2)
  (1 / 3) * base_area * height

-- The problem statement and our goal to prove:
theorem volume_of_regular_tetrahedron_with_edge_length_1 :
  volume_of_regular_tetrahedron 1 = sqrt 2 / 12 := sorry

end volume_of_regular_tetrahedron_with_edge_length_1_l196_196841


namespace walking_speed_l196_196859

theorem walking_speed (d : ℝ) (w_speed r_speed : ℝ) (w_time r_time : ℝ)
    (h1 : d = r_speed * r_time)
    (h2 : r_speed = 24)
    (h3 : r_time = 1)
    (h4 : w_time = 3) :
    w_speed = 8 :=
by
  sorry

end walking_speed_l196_196859


namespace rectangles_equal_area_implies_value_l196_196547

theorem rectangles_equal_area_implies_value (x y : ℝ) (h1 : x < 9) (h2 : y < 4)
  (h3 : x * (4 - y) = y * (9 - x)) : 360 * x / y = 810 :=
by
  -- We only need to state the theorem, the proof is not required.
  sorry

end rectangles_equal_area_implies_value_l196_196547


namespace dist_points_l196_196885

-- Define the points p1 and p2
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, 1)

-- Define the distance formula between the points
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The theorem stating the distance between these points is 5
theorem dist_points : dist p1 p2 = 5 := by
  sorry

end dist_points_l196_196885


namespace min_platforms_needed_l196_196771

theorem min_platforms_needed :
  let slabs_7_tons := 120
  let slabs_9_tons := 80
  let weight_7_tons := 7
  let weight_9_tons := 9
  let max_weight_per_platform := 40
  let total_weight := slabs_7_tons * weight_7_tons + slabs_9_tons * weight_9_tons
  let platforms_needed_per_7_tons := slabs_7_tons / 3
  let platforms_needed_per_9_tons := slabs_9_tons / 2
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 ∧ 3 * platforms_needed_per_7_tons = slabs_7_tons ∧ 2 * platforms_needed_per_9_tons = slabs_9_tons →
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 :=
by
  sorry

end min_platforms_needed_l196_196771


namespace find_y_common_solution_l196_196273

theorem find_y_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 11 ∧ x^2 = 4*y - 7) ↔ (7/4 ≤ y ∧ y ≤ Real.sqrt 11) :=
by
  sorry

end find_y_common_solution_l196_196273


namespace train_length_correct_l196_196438

noncomputable def length_of_train (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let total_distance := speed_m_per_s * time_s
  total_distance - platform_length_m

theorem train_length_correct :
  length_of_train 55 520 43.196544276457885 = 140 :=
by
  unfold length_of_train
  -- The conversion and calculations would be verified here
  sorry

end train_length_correct_l196_196438


namespace part1_monotonic_intervals_part2_max_a_l196_196180

noncomputable def f1 (x : ℝ) := Real.log x - 2 * x^2

theorem part1_monotonic_intervals :
  (∀ x, 0 < x ∧ x < 0.5 → f1 x > 0) ∧ (∀ x, x > 0.5 → f1 x < 0) :=
by
  sorry

noncomputable def f2 (x a : ℝ) := Real.log x + a * x^2

theorem part2_max_a (a : ℤ) :
  (∀ x, x > 1 → f2 x a < Real.exp x) → a ≤ 1 :=
by
  sorry

end part1_monotonic_intervals_part2_max_a_l196_196180


namespace jerry_trays_l196_196506

theorem jerry_trays :
  ∀ (trays_from_table1 trays_from_table2 trips trays_per_trip : ℕ),
  trays_from_table1 = 9 →
  trays_from_table2 = 7 →
  trips = 2 →
  trays_from_table1 + trays_from_table2 = 16 →
  trays_per_trip = (trays_from_table1 + trays_from_table2) / trips →
  trays_per_trip = 8 :=
by
  intros
  sorry

end jerry_trays_l196_196506


namespace no_real_roots_iff_k_lt_neg_one_l196_196315

theorem no_real_roots_iff_k_lt_neg_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - k ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_iff_k_lt_neg_one_l196_196315


namespace figure_50_squares_l196_196215

open Nat

noncomputable def g (n : ℕ) : ℕ := 2 * n ^ 2 + 5 * n + 2

theorem figure_50_squares : g 50 = 5252 :=
by
  sorry

end figure_50_squares_l196_196215


namespace part1_part2_l196_196669

variable (a : ℝ)

-- Proposition A
def propA (a : ℝ) := ∀ x : ℝ, ¬ (x^2 + (2*a-1)*x + a^2 ≤ 0)

-- Proposition B
def propB (a : ℝ) := 0 < a^2 - 1 ∧ a^2 - 1 < 1

theorem part1 (ha : propA a ∨ propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) ∨ (a > 1/4) :=
  sorry

theorem part2 (ha : ¬ propA a) (hb : propB a) : 
  (-Real.sqrt 2 < a ∧ a < -1) → (a^3 + 1 < a^2 + a) :=
  sorry

end part1_part2_l196_196669


namespace square_of_1024_l196_196720

theorem square_of_1024 : 1024^2 = 1048576 :=
by
  sorry

end square_of_1024_l196_196720


namespace point_inside_circle_l196_196627

theorem point_inside_circle : 
  ∀ (x y : ℝ), 
  (x-2)^2 + (y-3)^2 = 4 → 
  (3-2)^2 + (2-3)^2 < 4 :=
by
  intro x y h
  sorry

end point_inside_circle_l196_196627


namespace find_d_given_n_eq_cda_div_a_minus_d_l196_196066

theorem find_d_given_n_eq_cda_div_a_minus_d (a c d n : ℝ) (h : n = c * d * a / (a - d)) :
  d = n * a / (c * d + n) := 
by
  sorry

end find_d_given_n_eq_cda_div_a_minus_d_l196_196066


namespace B_completion_time_l196_196147

-- Definitions based on the conditions
def A_work : ℚ := 1 / 24
def B_work : ℚ := 1 / 16
def C_work : ℚ := 1 / 32  -- Since C takes twice the time as B, C_work = B_work / 2

-- Combined work rates based on the conditions
def combined_ABC_work := A_work + B_work + C_work
def combined_AB_work := A_work + B_work

-- Question: How long does B take to complete the job alone?
-- Answer: 16 days

theorem B_completion_time : 
  (combined_ABC_work = 1 / 8) ∧ 
  (combined_AB_work = 1 / 12) ∧ 
  (A_work = 1 / 24) ∧ 
  (C_work = B_work / 2) → 
  (1 / B_work = 16) := 
by 
  sorry

end B_completion_time_l196_196147


namespace cricket_initial_overs_l196_196074

theorem cricket_initial_overs
  (x : ℕ)
  (hx1 : ∃ x : ℕ, 0 ≤ x)
  (initial_run_rate : ℝ)
  (remaining_run_rate : ℝ)
  (remaining_overs : ℕ)
  (target_runs : ℕ)
  (H1 : initial_run_rate = 3.2)
  (H2 : remaining_run_rate = 6.25)
  (H3 : remaining_overs = 40)
  (H4 : target_runs = 282) :
  3.2 * (x : ℝ) + 6.25 * 40 = 282 → x = 10 := 
by 
  simp only [H1, H2, H3, H4]
  sorry

end cricket_initial_overs_l196_196074


namespace count_bottom_right_arrows_l196_196033

/-!
# Problem Statement
Each blank cell on the edge is to be filled with an arrow. The number in each square indicates the number of arrows pointing to that number. The arrows can point in the following directions: up, down, left, right, top-left, top-right, bottom-left, and bottom-right. Each arrow must point to a number. Figure 3 is provided and based on this, determine the number of arrows pointing to the bottom-right direction.
-/

def bottom_right_arrows_count : Nat :=
  2

theorem count_bottom_right_arrows :
  bottom_right_arrows_count = 2 :=
by
  sorry

end count_bottom_right_arrows_l196_196033


namespace inequality_sum_squares_l196_196083

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l196_196083


namespace total_people_l196_196320

-- Given definitions
def students : ℕ := 37500
def ratio_students_professors : ℕ := 15
def professors : ℕ := students / ratio_students_professors

-- The statement to prove
theorem total_people : students + professors = 40000 := by
  sorry

end total_people_l196_196320


namespace find_divisor_value_l196_196121

theorem find_divisor_value
  (D : ℕ) 
  (h1 : ∃ k : ℕ, 242 = k * D + 6)
  (h2 : ∃ l : ℕ, 698 = l * D + 13)
  (h3 : ∃ m : ℕ, 940 = m * D + 5) : 
  D = 14 :=
by
  sorry

end find_divisor_value_l196_196121


namespace correct_propositions_l196_196260

-- Definitions based on conditions
def diameter_perpendicular_bisects_chord (d : ℝ) (c : ℝ) : Prop :=
  ∃ (r : ℝ), d = 2 * r ∧ c = r

def triangle_vertices_determine_circle (a b c : ℝ) : Prop :=
  ∃ (O : ℝ), O = (a + b + c) / 3

def cyclic_quadrilateral_diagonals_supplementary (a b c d : ℕ) : Prop :=
  a + b + c + d = 360 -- incorrect statement

def tangent_perpendicular_to_radius (r t : ℝ) : Prop :=
  r * t = 1 -- assuming point of tangency

-- Theorem based on the problem conditions
theorem correct_propositions :
  diameter_perpendicular_bisects_chord 2 1 ∧
  triangle_vertices_determine_circle 1 2 3 ∧
  ¬ cyclic_quadrilateral_diagonals_supplementary 90 90 90 90 ∧
  tangent_perpendicular_to_radius 1 1 :=
by
  sorry

end correct_propositions_l196_196260


namespace lost_card_number_l196_196404

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196404


namespace find_set_of_all_points_P_l196_196199

-- Define the problem conditions and proof statement
variable {c : Type} [circle c] {l : Type} [line l] (M : point) [on_line M l] [tangent_to l c]
variable (A : point) (a b : ℝ) [center_of_circle A c] [radius_of_circle c b]

theorem find_set_of_all_points_P:
  ∃ (P : point) (Q R : point), 
    on_line Q l ∧ on_line R l ∧ midpoint M Q R ∧ incircle_of_triangle c P Q R → 
    ∀ (x1 y1 : ℝ), y1 = b / a * x1 + b ∧ y1 > 2 * b :=
sorry

end find_set_of_all_points_P_l196_196199


namespace middle_pile_cards_l196_196421

theorem middle_pile_cards (x : Nat) (h : x ≥ 2) : 
    let left := x - 2
    let middle := x + 2
    let right := x
    let middle_after_step3 := middle + 1
    let final_middle := middle_after_step3 - left
    final_middle = 5 := 
by
  sorry

end middle_pile_cards_l196_196421


namespace find_x_squared_plus_y_squared_l196_196938

theorem find_x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = -12) : x^2 + y^2 = 25 := 
by 
  sorry

end find_x_squared_plus_y_squared_l196_196938


namespace disjoint_subsets_with_same_sum_l196_196999

theorem disjoint_subsets_with_same_sum :
  ∀ (S : Finset ℕ), S.card = 10 ∧ (∀ x ∈ S, x ∈ Finset.range 101) →
  ∃ A B : Finset ℕ, A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by
  sorry

end disjoint_subsets_with_same_sum_l196_196999


namespace equation_of_circle_unique_l196_196886

noncomputable def equation_of_circle := 
  ∃ (d e f : ℝ), 
    (4 + 4 + 2*d + 2*e + f = 0) ∧ 
    (25 + 9 + 5*d + 3*e + f = 0) ∧ 
    (9 + 1 + 3*d - e + f = 0) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 + d*x + e*y + f = 0 → (x = 2 ∧ y = 2) ∨ (x = 5 ∧ y = 3) ∨ (x = 3 ∧ y = -1))

theorem equation_of_circle_unique :
  equation_of_circle := sorry

end equation_of_circle_unique_l196_196886


namespace transform_fraction_l196_196103

theorem transform_fraction (x : ℝ) (h : x ≠ 1) : - (1 / (1 - x)) = 1 / (x - 1) :=
by
  sorry

end transform_fraction_l196_196103


namespace order_of_mnpq_l196_196614

theorem order_of_mnpq 
(m n p q : ℝ) 
(h1 : m < n)
(h2 : p < q)
(h3 : (p - m) * (p - n) < 0)
(h4 : (q - m) * (q - n) < 0) 
: m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_mnpq_l196_196614


namespace isosceles_triangle_largest_angle_l196_196319

theorem isosceles_triangle_largest_angle (A B C : Type) (α β γ : ℝ)
  (h_iso : α = β) (h_angles : α = 50) (triangle: α + β + γ = 180) : γ = 80 :=
sorry

end isosceles_triangle_largest_angle_l196_196319


namespace find_additional_payment_l196_196427

-- Definitions used from the conditions
def total_payments : ℕ := 52
def first_partial_payments : ℕ := 25
def second_partial_payments : ℕ := total_payments - first_partial_payments
def first_payment_amount : ℝ := 500
def average_payment : ℝ := 551.9230769230769

-- Condition in Lean
theorem find_additional_payment :
  let total_amount := average_payment * total_payments
  let first_payment_total := first_partial_payments * first_payment_amount
  ∃ x : ℝ, total_amount = first_payment_total + second_partial_payments * (first_payment_amount + x) → x = 100 :=
by
  sorry

end find_additional_payment_l196_196427


namespace probability_3a_minus_1_lt_0_l196_196830

noncomputable def uniform_random_variable (a : ℝ) : Prop :=
    0 ≤ a ∧ a ≤ 1

theorem probability_3a_minus_1_lt_0 (a : ℝ) 
    (ha : uniform_random_variable a) :
    (MeasureTheory.MeasureSpace.volume {x : ℝ | 3 * x - 1 < 0 ∧ 0 ≤ x ∧ x ≤ 1} / 
     MeasureTheory.MeasureSpace.volume {x : ℝ | 0 ≤ x ∧ x ≤ 1}) = 1 / 3 :=
  sorry

end probability_3a_minus_1_lt_0_l196_196830


namespace probability_fourth_term_integer_l196_196218

def generate_term (previous_term : ℚ) (coin_flip : Bool) : ℚ :=
  if coin_flip then 
    3 * previous_term + 2 
  else 
    previous_term / 3 - 2

def mia_sequence (initial_term : ℚ) (coin_flips : List Bool) : List ℚ :=
  coin_flips.foldl (fun acc flip => acc ++ [generate_term (acc.head!) flip]) [initial_term]

def count_integers (l: List ℚ): ℕ := 
  l.countp (fun x => x.denom = 1)

theorem probability_fourth_term_integer : 
  let initial_term := (10 : ℚ)
  let possible_flips := {ff, tt}
  let sequences := List.bind possible_flips (
    fun f1 => List.bind possible_flips (
      fun f2 => List.bind possible_flips (
        fun f3 => List.bind possible_flips (
          fun f4 => [mia_sequence initial_term [f1, f2, f3, f4]])))
  let fourth_terms := sequences.map (fun seq => seq.nth 4)
  count_integers fourth_terms = 1 / 8 := 
sorry

end probability_fourth_term_integer_l196_196218


namespace surface_area_of_larger_prism_l196_196017

def volume_of_brick := 288
def number_of_bricks := 11
def target_surface_area := 1368

theorem surface_area_of_larger_prism
    (vol: ℕ := volume_of_brick)
    (num: ℕ := number_of_bricks)
    (target: ℕ := target_surface_area)
    (exists_a_b_h : ∃ (a b h : ℕ), a = 12 ∧ b = 8 ∧ h = 3)
    (large_prism_dimensions : ∃ (L W H : ℕ), L = 24 ∧ W = 12 ∧ H = 11):
    2 * (24 * 12 + 24 * 11 + 12 * 11) = target :=
by
  sorry

end surface_area_of_larger_prism_l196_196017


namespace map_distance_correct_l196_196024

noncomputable def distance_on_map : ℝ :=
  let speed := 60  -- miles per hour
  let time := 6.5  -- hours
  let scale := 0.01282051282051282 -- inches per mile
  let actual_distance := speed * time -- in miles
  actual_distance * scale -- convert to inches

theorem map_distance_correct :
  distance_on_map = 5 :=
by 
  sorry

end map_distance_correct_l196_196024


namespace greatest_divisor_l196_196464

theorem greatest_divisor (n : ℕ) (h1 : 3461 % n = 23) (h2 : 4783 % n = 41) : n = 2 := by {
  sorry
}

end greatest_divisor_l196_196464


namespace total_passengers_landed_l196_196207

theorem total_passengers_landed 
  (passengers_on_time : ℕ) 
  (passengers_late : ℕ) 
  (passengers_connecting : ℕ) 
  (passengers_changed_plans : ℕ)
  (H1 : passengers_on_time = 14507)
  (H2 : passengers_late = 213)
  (H3 : passengers_connecting = 320)
  (H4 : passengers_changed_plans = 95) : 
  passengers_on_time + passengers_late + passengers_connecting = 15040 :=
by 
  sorry

end total_passengers_landed_l196_196207


namespace rise_in_height_of_field_l196_196419

theorem rise_in_height_of_field
  (field_length : ℝ)
  (field_width : ℝ)
  (pit_length : ℝ)
  (pit_width : ℝ)
  (pit_depth : ℝ)
  (field_area : ℝ := field_length * field_width)
  (pit_area : ℝ := pit_length * pit_width)
  (remaining_area : ℝ := field_area - pit_area)
  (pit_volume : ℝ := pit_length * pit_width * pit_depth)
  (rise_in_height : ℝ := pit_volume / remaining_area) :
  field_length = 20 →
  field_width = 10 →
  pit_length = 8 →
  pit_width = 5 →
  pit_depth = 2 →
  rise_in_height = 0.5 :=
by
  intros
  sorry

end rise_in_height_of_field_l196_196419


namespace point_A_outside_circle_iff_l196_196795

-- Define the conditions
def B : ℝ := 16
def radius : ℝ := 4
def A_position (t : ℝ) : ℝ := 2 * t

-- Define the theorem
theorem point_A_outside_circle_iff (t : ℝ) : (A_position t < B - radius) ∨ (A_position t > B + radius) ↔ (t < 6 ∨ t > 10) :=
by
  sorry

end point_A_outside_circle_iff_l196_196795


namespace triangle_count_in_circle_intersections_l196_196097

theorem triangle_count_in_circle_intersections
    (P : Finset.Points Circle) (hP : P.card = 10) (hInter : ∀ (C1 C2 C3 : Chord P), 
    ¬ Collinear (Intersection C1 C2) (Intersection C2 C3) (Intersection C1 C3)) : 
    (number_of_triangles P = 120) :=
sorry

end triangle_count_in_circle_intersections_l196_196097


namespace equivalent_expression_l196_196390

theorem equivalent_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 :=
sorry

end equivalent_expression_l196_196390


namespace Kyle_papers_delivered_each_week_proof_l196_196338

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end Kyle_papers_delivered_each_week_proof_l196_196338


namespace calc_expression_l196_196446

theorem calc_expression :
  15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 :=
by
  sorry

end calc_expression_l196_196446


namespace books_at_end_of_month_l196_196576

-- Definitions based on provided conditions
def initial_books : ℕ := 75
def loaned_books (x : ℕ) : ℕ := 40  -- Rounded from 39.99999999999999
def returned_books (x : ℕ) : ℕ := (loaned_books x * 70) / 100
def not_returned_books (x : ℕ) : ℕ := loaned_books x - returned_books x

-- The statement to be proved
theorem books_at_end_of_month (x : ℕ) : initial_books - not_returned_books x = 63 :=
by
  -- This will be filled in with the actual proof steps later
  sorry

end books_at_end_of_month_l196_196576


namespace tom_search_cost_l196_196827

theorem tom_search_cost (n : ℕ) (h1 : n = 10) :
  let first_5_days_cost := 5 * 100 in
  let remaining_days_cost := (n - 5) * 60 in
  let total_cost := first_5_days_cost + remaining_days_cost in
  total_cost = 800 :=
by
  -- conditions
  have h2 : first_5_days_cost = 500 := by rfl
  have h3 : remaining_days_cost = 300 := by
    have rem_day_count : n - 5 = 5 := by
      rw [h1]
      rfl
    rfl
  have h4 : total_cost = 500 + 300 :=
    by rfl
  -- conclusion
  rw [h2, h3] at h4
  exact h4

end tom_search_cost_l196_196827


namespace pizzas_difference_l196_196760

def pizzas (craig_first_day craig_second_day heather_first_day heather_second_day total_pizzas: ℕ) :=
  heather_first_day = 4 * craig_first_day ∧
  heather_second_day = craig_second_day - 20 ∧
  craig_first_day = 40 ∧
  craig_first_day + heather_first_day + craig_second_day + heather_second_day = total_pizzas

theorem pizzas_difference :
  ∀ (craig_first_day craig_second_day heather_first_day heather_second_day : ℕ),
  pizzas craig_first_day craig_second_day heather_first_day heather_second_day 380 →
  craig_second_day - craig_first_day = 60 :=
by
  intros craig_first_day craig_second_day heather_first_day heather_second_day h
  sorry

end pizzas_difference_l196_196760


namespace arithmetic_sequence_sum_l196_196965

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 : ℤ) (h1 : a_1 = -2017) 
  (h2 : (S 2009) / 2009 - (S 2007) / 2007 = 2) : 
  S 2017 = -2017 :=
by
  -- definitions and steps would go here
  sorry

end arithmetic_sequence_sum_l196_196965


namespace distance_between_centers_l196_196959

-- Define the points P, Q, R in the plane
variable (P Q R : ℝ × ℝ)

-- Define the lengths PQ, PR, and QR
variable (PQ PR QR : ℝ)
variable (is_right_triangle : ∃ (a b c : ℝ), PQ = a ∧ PR = b ∧ QR = c ∧ a^2 + b^2 = c^2)

-- Define the inradii r1, r2, r3 for triangles PQR, RST, and QUV respectively
variable (r1 r2 r3 : ℝ)

-- Assume PQ = 90, PR = 120, and QR = 150
axiom PQ_length : PQ = 90
axiom PR_length : PR = 120
axiom QR_length : QR = 150

-- Define the centers O2 and O3 of the circles C2 and C3 respectively
variable (O2 O3 : ℝ × ℝ)

-- Assume the inradius length is 30 for the initial triangle
axiom inradius_PQR : r1 = 30

-- Assume the positions of the centers of C2 and C3
axiom O2_position : O2 = (15, 75)
axiom O3_position : O3 = (70, 10)

-- Use the distance formula to express the final result
theorem distance_between_centers : ∃ n : ℕ, dist O2 O3 = Real.sqrt (10 * n) ∧ n = 725 :=
by
  sorry

end distance_between_centers_l196_196959


namespace intersection_M_N_l196_196899

-- Define the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x - 6 ≥ 0}

-- State the proof problem
theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196899


namespace find_a_values_l196_196493

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end find_a_values_l196_196493


namespace degree_polynomial_is_13_l196_196593

noncomputable def degree_polynomial (a b c d e f g h j : ℝ) : ℕ :=
  (7 + 4 + 2)

theorem degree_polynomial_is_13 (a b c d e f g h j : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) (hh : h ≠ 0) (hj : j ≠ 0) : 
  degree_polynomial a b c d e f g h j = 13 :=
by
  rfl

end degree_polynomial_is_13_l196_196593


namespace wipes_per_pack_l196_196526

theorem wipes_per_pack (days : ℕ) (wipes_per_day : ℕ) (packs : ℕ) (total_wipes : ℕ) (n : ℕ)
    (h1 : days = 360)
    (h2 : wipes_per_day = 2)
    (h3 : packs = 6)
    (h4 : total_wipes = wipes_per_day * days)
    (h5 : total_wipes = n * packs) : 
    n = 120 := 
by 
  sorry

end wipes_per_pack_l196_196526


namespace prob_heart_club_spade_l196_196555

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l196_196555


namespace trailing_zeros_30_factorial_l196_196936

-- Define the problem in Lean 4
theorem trailing_zeros_30_factorial : Nat.trailingZeroes (Nat.factorial 30) = 7 :=
by
  sorry

end trailing_zeros_30_factorial_l196_196936


namespace max_ab_min_reciprocal_sum_l196_196613

noncomputable section

-- Definitions for conditions
def is_positive_real (x : ℝ) : Prop := x > 0

def condition (a b : ℝ) : Prop := is_positive_real a ∧ is_positive_real b ∧ (a + 10 * b = 1)

-- Maximum value of ab
theorem max_ab (a b : ℝ) (h : condition a b) : a * b ≤ 1 / 40 :=
sorry

-- Minimum value of 1/a + 1/b
theorem min_reciprocal_sum (a b : ℝ) (h : condition a b) : 1 / a + 1 / b ≥ 11 + 2 * Real.sqrt 10 :=
sorry

end max_ab_min_reciprocal_sum_l196_196613


namespace bowling_ball_weight_l196_196792

theorem bowling_ball_weight :
  (∀ b c : ℝ, 9 * b = 2 * c → c = 35 → b = 70 / 9) :=
by
  intros b c h1 h2
  sorry

end bowling_ball_weight_l196_196792


namespace phone_answered_before_fifth_ring_l196_196002

theorem phone_answered_before_fifth_ring:
  (0.1 + 0.2 + 0.25 + 0.25 = 0.8) :=
by
  sorry

end phone_answered_before_fifth_ring_l196_196002


namespace ryan_learning_hours_l196_196882

theorem ryan_learning_hours :
  ∃ hours : ℕ, 
    (∀ e_hrs : ℕ, e_hrs = 2) → 
    (∃ c_hrs : ℕ, c_hrs = hours) → 
    (∀ s_hrs : ℕ, s_hrs = 4) → 
    hours = 4 + 1 :=
by
  sorry

end ryan_learning_hours_l196_196882


namespace least_possible_value_of_y_l196_196622

theorem least_possible_value_of_y (x y z : ℤ) (hx : Even x) (hy : Odd y) (hz : Odd z) 
  (h1 : y - x > 5) (h2 : z - x ≥ 9) : y ≥ 7 :=
by {
  -- sorry allows us to skip the proof
  sorry
}

end least_possible_value_of_y_l196_196622


namespace number_of_integer_pairs_l196_196606

theorem number_of_integer_pairs (n : ℕ) : 
  (∀ x y : ℤ, 5 * x^2 - 6 * x * y + y^2 = 6^100) → n = 19594 :=
sorry

end number_of_integer_pairs_l196_196606


namespace inequalities_always_true_l196_196347

variables {x y a b : Real}

/-- All given conditions -/
def conditions (x y a b : Real) :=
  x < a ∧ y < b ∧ x < 0 ∧ y < 0 ∧ a > 0 ∧ b > 0

theorem inequalities_always_true {x y a b : Real} (h : conditions x y a b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
sorry

end inequalities_always_true_l196_196347


namespace Kyle_papers_delivered_each_week_l196_196334

-- Definitions representing the conditions
def papers_per_day := 100
def days_Mon_to_Sat := 6
def regular_Sunday_customers := 100
def non_regular_Sunday_customers := 30
def no_delivery_customers_on_Sunday := 10

-- The total number of papers delivered each week
def total_papers_per_week : ℕ :=
  days_Mon_to_Sat * papers_per_day +
  regular_Sunday_customers - no_delivery_customers_on_Sunday + non_regular_Sunday_customers

-- Prove that Kyle delivers 720 papers each week
theorem Kyle_papers_delivered_each_week : total_papers_per_week = 720 :=
sorry

end Kyle_papers_delivered_each_week_l196_196334


namespace shortest_time_between_ships_l196_196142

theorem shortest_time_between_ships 
  (AB : ℝ) (speed_A : ℝ) (speed_B : ℝ) (angle_ABA' : ℝ) : (AB = 10) → (speed_A = 4) → (speed_B = 6) → (angle_ABA' = 60) →
  ∃ t : ℝ, (t = 150/7 / 60) :=
by
  intro hAB hSpeedA hSpeedB hAngle
  sorry

end shortest_time_between_ships_l196_196142


namespace solve_inequality_l196_196477

open Real

theorem solve_inequality (f : ℝ → ℝ)
  (h_cos : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f (cos x) ≥ 0) :
  ∀ k : ℤ, ∀ x, (2 * ↑k * π ≤ x ∧ x ≤ 2 * ↑k * π + π) → f (sin x) ≥ 0 :=
by
  intros k x hx
  sorry

end solve_inequality_l196_196477


namespace sum_of_solutions_l196_196840

-- Define the quadratic equation as a product of linear factors
def quadratic_eq (x : ℚ) : Prop := (4 * x + 6) * (3 * x - 8) = 0

-- Define the roots of the quadratic equation
def root1 : ℚ := -3 / 2
def root2 : ℚ := 8 / 3

-- Sum of the roots of the quadratic equation
def sum_of_roots : ℚ := root1 + root2

-- Theorem stating that the sum of the roots is 7/6
theorem sum_of_solutions : sum_of_roots = 7 / 6 := by
  sorry

end sum_of_solutions_l196_196840


namespace people_left_after_first_stop_l196_196551

def initial_people_on_train : ℕ := 48
def people_got_off_train : ℕ := 17

theorem people_left_after_first_stop : (initial_people_on_train - people_got_off_train) = 31 := by
  sorry

end people_left_after_first_stop_l196_196551


namespace negation_equivalence_l196_196546

theorem negation_equivalence {Triangle : Type} (has_circumcircle : Triangle → Prop) :
  ¬ (∃ (t : Triangle), ¬ has_circumcircle t) ↔ (∀ (t : Triangle), has_circumcircle t) :=
by
  sorry

end negation_equivalence_l196_196546


namespace crescent_perimeter_l196_196858

def radius_outer : ℝ := 10.5
def radius_inner : ℝ := 6.7

theorem crescent_perimeter : (radius_outer + radius_inner) * Real.pi = 54.037 :=
by
  sorry

end crescent_perimeter_l196_196858


namespace number_of_squirrels_l196_196676

/-
Problem: Some squirrels collected 575 acorns. If each squirrel needs 130 acorns to get through the winter, each squirrel needs to collect 15 more acorns. 
Question: How many squirrels are there?
Conditions:
 1. Some squirrels collected 575 acorns.
 2. Each squirrel needs 130 acorns to get through the winter.
 3. Each squirrel needs to collect 15 more acorns.
Answer: 5 squirrels
-/

theorem number_of_squirrels (acorns_total : ℕ) (acorns_needed : ℕ) (acorns_short : ℕ) (S : ℕ)
  (h1 : acorns_total = 575)
  (h2 : acorns_needed = 130)
  (h3 : acorns_short = 15)
  (h4 : S * (acorns_needed - acorns_short) = acorns_total) :
  S = 5 :=
by
  sorry

end number_of_squirrels_l196_196676


namespace remaining_yards_is_720_l196_196139

-- Definitions based on conditions:
def marathon_miles : Nat := 25
def marathon_yards : Nat := 500
def yards_in_mile : Nat := 1760
def num_of_marathons : Nat := 12

-- Total distance for one marathon in yards
def one_marathon_total_yards : Nat :=
  marathon_miles * yards_in_mile + marathon_yards

-- Total distance for twelve marathons in yards
def total_distance_yards : Nat :=
  num_of_marathons * one_marathon_total_yards

-- Remaining yards after converting the total distance into miles and yards
def y : Nat :=
  total_distance_yards % yards_in_mile

-- Condition ensuring y is the remaining yards and is within the bounds 0 ≤ y < 1760
theorem remaining_yards_is_720 : 
  y = 720 := sorry

end remaining_yards_is_720_l196_196139


namespace area_of_rectangle_l196_196360

theorem area_of_rectangle (a b : ℝ) (h1 : 2 * (a + b) = 16) (h2 : 2 * a^2 + 2 * b^2 = 68) :
  a * b = 15 :=
by
  have h3 : a + b = 8 := by sorry
  have h4 : a^2 + b^2 = 34 := by sorry
  have h5 : (a + b) ^ 2 = a^2 + b^2 + 2 * a * b := by sorry
  have h6 : 64 = 34 + 2 * a * b := by sorry
  have h7 : 2 * a * b = 30 := by sorry
  exact sorry

end area_of_rectangle_l196_196360


namespace card_probability_l196_196553

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l196_196553


namespace great_dane_more_than_triple_pitbull_l196_196857

variables (C P G : ℕ)
variables (h1 : G = 307) (h2 : P = 3 * C) (h3 : C + P + G = 439)

theorem great_dane_more_than_triple_pitbull
  : G - 3 * P = 10 :=
by
  sorry

end great_dane_more_than_triple_pitbull_l196_196857


namespace jenny_original_amount_half_l196_196956

-- Definitions based on conditions
def original_amount (x : ℝ) := x
def spent_fraction := 3 / 7
def left_after_spending (x : ℝ) := x * (1 - spent_fraction)

theorem jenny_original_amount_half (x : ℝ) (h : left_after_spending x = 24) : original_amount x / 2 = 21 :=
by
  -- Indicate the intention to prove the statement by sorry
  sorry

end jenny_original_amount_half_l196_196956


namespace greatest_b_max_b_value_l196_196539

theorem greatest_b (b y : ℤ) (h : b > 0) (hy : y^2 + b*y = -21) : b ≤ 22 :=
sorry

theorem max_b_value : ∃ b : ℤ, (∀ y : ℤ, y^2 + b*y = -21 → b > 0) ∧ (b = 22) :=
sorry

end greatest_b_max_b_value_l196_196539


namespace perpendicular_angles_l196_196946

theorem perpendicular_angles (α β : ℝ) (k : ℤ) : 
  (∃ k : ℤ, β - α = k * 360 + 90 ∨ β - α = k * 360 - 90) →
  β = k * 360 + α + 90 ∨ β = k * 360 + α - 90 :=
by
  sorry

end perpendicular_angles_l196_196946


namespace polynomial_coeff_fraction_eq_neg_122_div_121_l196_196963

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121_l196_196963


namespace neighbor_packs_l196_196520

theorem neighbor_packs (n : ℕ) :
  let milly_balloons := 3 * 6 -- Milly and Floretta use 3 packs of their own
  let neighbor_balloons := n * 6 -- some packs of the neighbor's balloons, each contains 6 balloons
  let total_balloons := milly_balloons + neighbor_balloons -- total balloons
  -- They split balloons evenly; Milly takes 7 extra, then Floretta has 8 left
  total_balloons / 2 + 7 = total_balloons - 15
  → n = 2 := sorry

end neighbor_packs_l196_196520


namespace g_at_10_l196_196344

noncomputable def g : ℕ → ℝ :=
sorry

axiom g_1 : g 1 = 2
axiom g_prop (m n : ℕ) (hmn : m ≥ n) : g (m + n) + g (m - n) = 2 * (g m + g n)

theorem g_at_10 : g 10 = 200 := 
sorry

end g_at_10_l196_196344


namespace samantha_original_cans_l196_196529

theorem samantha_original_cans : 
  ∀ (cans_per_classroom : ℚ),
  (cans_per_classroom = (50 - 38) / 5) →
  (50 / cans_per_classroom) = 21 := 
by
  sorry

end samantha_original_cans_l196_196529


namespace linear_function_third_quadrant_and_origin_l196_196942

theorem linear_function_third_quadrant_and_origin (k b : ℝ) (h1 : ∀ x < 0, k * x + b ≥ 0) (h2 : k * 0 + b ≠ 0) : k < 0 ∧ b > 0 :=
sorry

end linear_function_third_quadrant_and_origin_l196_196942


namespace sum_of_three_numbers_l196_196107

theorem sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : b = 10) 
  (h2 : (a + b + c) / 3 = a + 20) 
  (h3 : (a + b + c) / 3 = c - 25) : 
  a + b + c = 45 := 
by 
  sorry

end sum_of_three_numbers_l196_196107


namespace polynomial_sum_l196_196814

noncomputable def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_sum : ∃ a b c d : ℝ, 
  (g a b c d (-3 * Complex.I) = 0) ∧
  (g a b c d (1 + Complex.I) = 0) ∧
  (g a b c d (3 * Complex.I) = 0) ∧
  (g a b c d (1 - Complex.I) = 0) ∧ 
  (a + b + c + d = 9) := by
  sorry

end polynomial_sum_l196_196814


namespace cost_of_apples_and_bananas_l196_196550

variable (a b : ℝ) -- Assume a and b are real numbers.

theorem cost_of_apples_and_bananas (a b : ℝ) : 
  (3 * a + 2 * b) = 3 * a + 2 * b :=
by 
  sorry -- Proof placeholder

end cost_of_apples_and_bananas_l196_196550


namespace multiplication_pattern_correct_l196_196759

theorem multiplication_pattern_correct :
  (1 * 9 + 2 = 11) ∧
  (12 * 9 + 3 = 111) ∧
  (123 * 9 + 4 = 1111) ∧
  (1234 * 9 + 5 = 11111) ∧
  (12345 * 9 + 6 = 111111) →
  123456 * 9 + 7 = 1111111 :=
by
  sorry

end multiplication_pattern_correct_l196_196759


namespace robotics_club_students_l196_196348

theorem robotics_club_students (total cs e both neither : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : e = 38)
  (h4 : both = 25)
  (h5 : neither = total - (cs - both + e - both + both)) :
  neither = 15 :=
by
  sorry

end robotics_club_students_l196_196348


namespace num_of_sets_eq_four_l196_196358

open Finset

theorem num_of_sets_eq_four : ∀ B : Finset ℕ, (insert 1 (insert 2 B) = {1, 2, 3, 4, 5}) → (B = {3, 4, 5} ∨ B = {1, 3, 4, 5} ∨ B = {2, 3, 4, 5} ∨ B = {1, 2, 3, 4, 5}) := 
by
  sorry

end num_of_sets_eq_four_l196_196358


namespace inequality_sum_squares_l196_196084

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l196_196084


namespace opposite_of_one_third_l196_196149

theorem opposite_of_one_third : -(1/3) = -1/3 := by
  sorry

end opposite_of_one_third_l196_196149


namespace sum_of_angles_satisfying_condition_l196_196889

theorem sum_of_angles_satisfying_condition :
  (∑ x in (Finset.filter (λ x : ℝ, 0 ≤ x ∧ x ≤ 360
                              ∧ sin x ^ 3 + cos x ^ 3 = 1 / cos x + 1 / sin x 
                             ) (Finset.range 361)),
      x) = 450 := by
sorry

end sum_of_angles_satisfying_condition_l196_196889


namespace china_nhsm_league_2021_zhejiang_p15_l196_196798

variable (x y z : ℝ)

theorem china_nhsm_league_2021_zhejiang_p15 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x ^ 4 + y ^ 2 * z ^ 2) / (x ^ (5 / 2) * (y + z)) + 
  (y ^ 4 + z ^ 2 * x ^ 2) / (y ^ (5 / 2) * (z + x)) + 
  (z ^ 4 + y ^ 2 * x ^ 2) / (z ^ (5 / 2) * (y + x)) ≥ 1 := 
sorry

end china_nhsm_league_2021_zhejiang_p15_l196_196798


namespace chocolate_chips_per_cookie_l196_196331

theorem chocolate_chips_per_cookie
  (num_batches : ℕ)
  (cookies_per_batch : ℕ)
  (num_people : ℕ)
  (chocolate_chips_per_person : ℕ) :
  (num_batches = 3) →
  (cookies_per_batch = 12) →
  (num_people = 4) →
  (chocolate_chips_per_person = 18) →
  (chocolate_chips_per_person / (num_batches * cookies_per_batch / num_people) = 2) :=
by
  sorry

end chocolate_chips_per_cookie_l196_196331


namespace problem_statement_l196_196297

namespace ProofProblems

open Set

def U : Set ℕ := {2, 3, 4, 5, 6}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 4, 5, 6}

theorem problem_statement : M ∪ N = U := sorry

end ProofProblems

end problem_statement_l196_196297


namespace two_pair_probability_l196_196382

noncomputable def prob_two_pair : ℚ :=
  let total_outcomes := (Nat.choose 52 5 : ℚ) in
  let successful_outcomes := (13 * 6 * 12 * 6 * 11 * 4 : ℚ) in
  successful_outcomes / total_outcomes

theorem two_pair_probability :
  prob_two_pair = 108 / 1005 := by sorry

end two_pair_probability_l196_196382


namespace cube_difference_positive_l196_196185

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l196_196185


namespace total_bricks_l196_196355

theorem total_bricks (n1 n2 r1 r2 : ℕ) (w1 w2 : ℕ)
  (h1 : n1 = 60) (h2 : r1 = 100) (h3 : n2 = 80) (h4 : r2 = 120)
  (h5 : w1 = 5) (h6 : w2 = 5) :
  (w1 * (n1 * r1) + w2 * (n2 * r2)) = 78000 :=
by sorry

end total_bricks_l196_196355


namespace passengers_taken_second_station_l196_196440

def initial_passengers : ℕ := 288
def passengers_dropped_first_station : ℕ := initial_passengers / 3
def passengers_after_first_station : ℕ := initial_passengers - passengers_dropped_first_station
def passengers_taken_first_station : ℕ := 280
def total_passengers_after_first_station : ℕ := passengers_after_first_station + passengers_taken_first_station
def passengers_dropped_second_station : ℕ := total_passengers_after_first_station / 2
def passengers_left_after_second_station : ℕ := total_passengers_after_first_station - passengers_dropped_second_station
def passengers_at_third_station : ℕ := 248

theorem passengers_taken_second_station : 
  ∃ (x : ℕ), passengers_left_after_second_station + x = passengers_at_third_station ∧ x = 12 :=
by 
  sorry

end passengers_taken_second_station_l196_196440


namespace six_digit_number_multiple_of_7_l196_196169

theorem six_digit_number_multiple_of_7 (d : ℕ) (hd : d ≤ 9) :
  (∃ k : ℤ, 56782 + d * 10 = 7 * k) ↔ (d = 0 ∨ d = 7) := by
sorry

end six_digit_number_multiple_of_7_l196_196169


namespace total_apples_eq_l196_196818

-- Define the conditions for the problem
def baskets : ℕ := 37
def apples_per_basket : ℕ := 17

-- Define the theorem to prove the total number of apples
theorem total_apples_eq : baskets * apples_per_basket = 629 :=
by
  sorry

end total_apples_eq_l196_196818


namespace gcd_sum_of_cubes_l196_196044

-- Define the problem conditions
variables (n : ℕ) (h_pos : n > 27)

-- Define the goal to prove
theorem gcd_sum_of_cubes (h : n > 27) : 
  gcd (n^3 + 27) (n + 3) = n + 3 :=
by sorry

end gcd_sum_of_cubes_l196_196044


namespace intersection_M_N_eq_neg2_l196_196911

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end intersection_M_N_eq_neg2_l196_196911


namespace find_a_l196_196307

theorem find_a (a : ℝ) (extreme_at_neg_2 : ∀ x : ℝ, (3 * a * x^2 + 2 * x) = 0 → x = -2) :
    a = 1 / 3 :=
sorry

end find_a_l196_196307


namespace replacement_digit_is_9_l196_196769

theorem replacement_digit_is_9 :
  ∀ (d : ℕ), 
    (let num_sixes_units := 10 in
     let num_sixes_tens := 10 in
     let total_difference := num_sixes_units * (d - 6) + num_sixes_tens * 10 * (d - 6) in
     total_difference = 330) →
    d = 9 :=
by
  intros d h
  -- definitions and equations derived from conditions
  let num_sixes_units := 10
  let num_sixes_tens := 10
  let total_difference := num_sixes_units * (d - 6) + num_sixes_tens * 10 * (d - 6)
  -- proof would go here
  sorry

end replacement_digit_is_9_l196_196769


namespace max_value_of_y_l196_196985

open Real

noncomputable def y (x : ℝ) := 1 + 1 / (x^2 + 2*x + 2)

theorem max_value_of_y : ∃ x : ℝ, y x = 2 :=
sorry

end max_value_of_y_l196_196985


namespace second_certificate_interest_rate_l196_196261

noncomputable def find_ann_rate_after_second_period (initial_investment : ℝ) (first_rate : ℝ) (first_duration_months : ℝ)
  (first_value : ℝ) (second_value : ℝ) : ℝ :=
  let first_interest := first_rate / 12 * first_duration_months
  let intermediate_value := initial_investment * (1 + first_interest / 100)
  let second_duration_months := 3
  let s := ((second_value / intermediate_value) - 1) * 400 / (second_duration_months / 12)
  s

theorem second_certificate_interest_rate
  (initial_investment : ℝ) 
  (first_rate : ℝ) 
  (first_duration_months : ℝ) 
  (first_value : ℝ) 
  (second_value : ℝ) 
  (h_initial : initial_investment = 12000) 
  (h_first_rate : first_rate = 8) 
  (h_first_duration : first_duration_months = 3) 
  (h_first_value : first_value = 12240)
  (h_second_value : second_value = 12435) :
  find_ann_rate_after_second_period initial_investment first_rate first_duration_months first_value second_value ≈ 6.38 :=
by
  -- Using the given conditions to show the equality
  sorry

end second_certificate_interest_rate_l196_196261


namespace algebraic_expression_interpretation_l196_196951

def donations_interpretation (m n : ℝ) : ℝ := 5 * m + 2 * n
def plazas_area_interpretation (a : ℝ) : ℝ := 6 * a^2

theorem algebraic_expression_interpretation (m n a : ℝ) :
  donations_interpretation m n = 5 * m + 2 * n ∧ plazas_area_interpretation a = 6 * a^2 :=
by
  sorry

end algebraic_expression_interpretation_l196_196951


namespace real_solutions_of_polynomial_l196_196038

theorem real_solutions_of_polynomial (b : ℝ) :
  b < -4 → ∃! x : ℝ, x^3 - b * x^2 - 4 * b * x + b^2 - 4 = 0 :=
by
  sorry

end real_solutions_of_polynomial_l196_196038


namespace initial_red_marbles_l196_196631

theorem initial_red_marbles (r g : ℕ) (h1 : r * 3 = 7 * g) (h2 : 4 * (r - 14) = g + 30) : r = 24 := 
sorry

end initial_red_marbles_l196_196631


namespace probability_red_ball_l196_196714

def total_balls : ℕ := 3
def red_balls : ℕ := 1
def yellow_balls : ℕ := 2

theorem probability_red_ball : (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
by
  sorry

end probability_red_ball_l196_196714


namespace unique_element_in_set_l196_196174

theorem unique_element_in_set (A : Set ℝ) (h₁ : ∃ x, A = {x})
(h₂ : ∀ x ∈ A, (x + 3) / (x - 1) ∈ A) : ∃ x, x ∈ A ∧ (x = 3 ∨ x = -1) := by
  sorry

end unique_element_in_set_l196_196174


namespace walls_painted_purple_l196_196203

theorem walls_painted_purple :
  (10 - (3 * 10 / 5)) * 8 = 32 := by
  sorry

end walls_painted_purple_l196_196203


namespace solve_for_x_l196_196749

noncomputable def avg (a b : ℝ) := (a + b) / 2

noncomputable def B (t : List ℝ) : List ℝ :=
  match t with
  | [a, b, c, d, e] => [avg a b, avg b c, avg c d, avg d e]
  | _ => []

noncomputable def B_iter (m : ℕ) (t : List ℝ) : List ℝ :=
  match m with
  | 0 => t
  | k + 1 => B (B_iter k t)

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : B_iter 4 [1, x, x^2, x^3, x^4] = [1/4]) :
  x = Real.sqrt 2 - 1 :=
sorry

end solve_for_x_l196_196749


namespace no_solution_for_conditions_l196_196351

theorem no_solution_for_conditions :
  ∀ (x y : ℝ), 0 < x → 0 < y → x * y = 2^15 → (Real.log x / Real.log 2) * (Real.log y / Real.log 2) = 60 → False :=
by
  intro x y x_pos y_pos h1 h2
  sorry

end no_solution_for_conditions_l196_196351


namespace frank_whack_a_mole_tickets_l196_196691

variable (W : ℕ)
variable (skee_ball_tickets : ℕ := 9)
variable (candy_cost : ℕ := 6)
variable (candies_bought : ℕ := 7)
variable (total_tickets : ℕ := W + skee_ball_tickets)
variable (required_tickets : ℕ := candy_cost * candies_bought)

theorem frank_whack_a_mole_tickets : W + skee_ball_tickets = required_tickets → W = 33 := by
  sorry

end frank_whack_a_mole_tickets_l196_196691


namespace probability_drawing_red_l196_196716

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end probability_drawing_red_l196_196716


namespace roots_equation_l196_196962
-- We bring in the necessary Lean libraries

-- Define the conditions as Lean definitions
variable (x1 x2 : ℝ)
variable (h1 : x1^2 + x1 - 3 = 0)
variable (h2 : x2^2 + x2 - 3 = 0)

-- Lean 4 statement we need to prove
theorem roots_equation (x1 x2 : ℝ) (h1 : x1^2 + x1 - 3 = 0) (h2 : x2^2 + x2 - 3 = 0) : 
  x1^3 - 4 * x2^2 + 19 = 0 := 
sorry

end roots_equation_l196_196962


namespace sum_of_sequence_l196_196745

-- Definitions of sequence and difference sequence conditions
def a : ℕ → ℕ
| 0     := 2
| (n+1) := a n + 2^n

-- Sum of the first n terms of sequence a
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a i

-- Theorem statement to prove
theorem sum_of_sequence (n : ℕ) : S n = 2^(n+1) - 2 :=
by sorry

end sum_of_sequence_l196_196745


namespace triangle_area_solution_l196_196609

noncomputable def solve_for_x (x : ℝ) : Prop :=
  x > 0 ∧ (1 / 2 * x * 3 * x = 96) → x = 8

theorem triangle_area_solution : solve_for_x 8 :=
by
  sorry

end triangle_area_solution_l196_196609


namespace team_C_has_most_uniform_height_l196_196109

theorem team_C_has_most_uniform_height
  (S_A S_B S_C S_D : ℝ)
  (h_A : S_A = 0.13)
  (h_B : S_B = 0.11)
  (h_C : S_C = 0.09)
  (h_D : S_D = 0.15)
  (h_same_num_members : ∀ (a b c d : ℕ), a = b ∧ b = c ∧ c = d) 
  : S_C = min S_A (min S_B (min S_C S_D)) :=
by
  sorry

end team_C_has_most_uniform_height_l196_196109


namespace most_stable_city_l196_196032

def variance_STD : ℝ := 12.5
def variance_A : ℝ := 18.3
def variance_B : ℝ := 17.4
def variance_C : ℝ := 20.1

theorem most_stable_city : variance_STD < variance_A ∧ variance_STD < variance_B ∧ variance_STD < variance_C :=
by {
  -- Proof skipped
  sorry
}

end most_stable_city_l196_196032


namespace pool_capacity_l196_196864

theorem pool_capacity
  (pump_removes : ∀ (x : ℝ), x > 0 → (2 / 3) * x / 7.5 = (4 / 15) * x)
  (working_time : 0.15 * 60 = 9)
  (remaining_water : ∀ (x : ℝ), x > 0 → x - (0.8 * x) = 25) :
  ∃ x : ℝ, x = 125 :=
by
  sorry

end pool_capacity_l196_196864


namespace cost_of_show_dogs_l196_196153

noncomputable def cost_per_dog : ℕ → ℕ → ℕ → ℕ
| total_revenue, total_profit, number_of_dogs => (total_revenue - total_profit) / number_of_dogs

theorem cost_of_show_dogs {revenue_per_puppy number_of_puppies profit number_of_dogs : ℕ}
  (h_puppies: number_of_puppies = 6)
  (h_revenue_per_puppy : revenue_per_puppy = 350)
  (h_profit : profit = 1600)
  (h_number_of_dogs : number_of_dogs = 2)
:
  cost_per_dog (number_of_puppies * revenue_per_puppy) profit number_of_dogs = 250 :=
by
  sorry

end cost_of_show_dogs_l196_196153


namespace find_positive_integers_l196_196289

noncomputable def positive_integer_solutions_ineq (x : ℕ) : Prop :=
  x > 0 ∧ (x : ℝ) < 4

theorem find_positive_integers (x : ℕ) : 
  (x > 0 ∧ (↑x - 3)/3 < 7 - 5*(↑x)/3) ↔ positive_integer_solutions_ineq x :=
by
  sorry

end find_positive_integers_l196_196289


namespace car_a_speed_l196_196027

theorem car_a_speed (d_A d_B v_B t v_A : ℝ)
  (h1 : d_A = 10)
  (h2 : v_B = 50)
  (h3 : t = 2.25)
  (h4 : d_A + 8 - d_B = v_A * t)
  (h5 : d_B = v_B * t) :
  v_A = 58 :=
by
  -- Work on the proof here
  sorry

end car_a_speed_l196_196027


namespace percentage_of_children_allowed_to_draw_l196_196377

def total_jelly_beans := 100
def total_children := 40
def remaining_jelly_beans := 36
def jelly_beans_per_child := 2

theorem percentage_of_children_allowed_to_draw :
  ((total_jelly_beans - remaining_jelly_beans) / jelly_beans_per_child : ℕ) * 100 / total_children = 80 := by
  sorry

end percentage_of_children_allowed_to_draw_l196_196377


namespace A_salary_is_3000_l196_196251

theorem A_salary_is_3000 
    (x y : ℝ) 
    (h1 : x + y = 4000)
    (h2 : 0.05 * x = 0.15 * y) 
    : x = 3000 := by
  sorry

end A_salary_is_3000_l196_196251


namespace min_expression_value_l196_196474

variable {a : ℕ → ℝ}
variable (m n : ℕ)
variable (q : ℝ)

axiom pos_seq (n : ℕ) : a n > 0
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom seq_condition : a 7 = a 6 + 2 * a 5
axiom exists_terms :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1)

theorem min_expression_value : 
  (∃m n : ℕ, m > 0 ∧ n > 0 ∧ (Real.sqrt (a m * a n) = 4 * a 1) ∧ 
  a 7 = a 6 + 2 * a 5 ∧ 
  (∀ n, a n > 0 ∧ a (n + 1) = q * a n)) → 
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_expression_value_l196_196474


namespace sheets_of_paper_in_each_box_l196_196150

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 30)
  (h2 : 2 * E = S)
  (h3 : 3 * E = S - 10) :
  S = 40 :=
by
  sorry

end sheets_of_paper_in_each_box_l196_196150


namespace max_value_fraction_l196_196929

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x - 1 = 0

theorem max_value_fraction (a b : ℝ) (H : circle_eq a b) :
  ∃ t : ℝ, -1/2 ≤ t ∧ t ≤ 1/2 ∧ b = t * (a - 3) ∧ t = 1 / 2 :=
by sorry

end max_value_fraction_l196_196929


namespace groupB_median_and_excellent_rate_l196_196132

noncomputable def groupB_scores : List ℕ :=
  [7, 7, 7, 7, 8, 8, 8, 9, 9, 10]

def median (l : List ℕ) : ℚ :=
  let sorted := l.sorted
  let n := l.length
  if n % 2 = 0 then
    (sorted.get ⟨n / 2 - 1, sorry⟩ + sorted.get ⟨n / 2, sorry⟩) / 2
  else
    sorted.get ⟨n / 2, sorry⟩

def excellent_rate (l : List ℕ) (threshold : ℕ) : ℚ :=
  let excellent_count := l.filter (λ score => score ≥ threshold).length
  (excellent_count : ℚ) / (l.length : ℚ) * 100

theorem groupB_median_and_excellent_rate :
  median groupB_scores = 8 ∧ excellent_rate groupB_scores 8 = 60 := by
  sorry

end groupB_median_and_excellent_rate_l196_196132


namespace find_the_number_l196_196292

theorem find_the_number (x : ℕ) (h : 18396 * x = 183868020) : x = 9990 :=
by
  sorry

end find_the_number_l196_196292


namespace parity_of_function_parity_neither_odd_nor_even_l196_196031

def f (x p : ℝ) : ℝ := x * |x| + p * x^2

theorem parity_of_function (p : ℝ) :
  (∀ x : ℝ, f x p = - f (-x) p) ↔ p = 0 :=
by
  sorry

theorem parity_neither_odd_nor_even (p : ℝ) :
  (∀ x : ℝ, f x p ≠ f (-x) p) ∧ (∀ x : ℝ, f x p ≠ - f (-x) p) ↔ p ≠ 0 :=
by
  sorry

end parity_of_function_parity_neither_odd_nor_even_l196_196031


namespace question_1_question_2_question_3_question_4_l196_196496

-- Define each condition as a theorem
theorem question_1 (explanation: String) : explanation = "providing for the living" :=
  sorry

theorem question_2 (usage: String) : usage = "structural auxiliary word, placed between subject and predicate, negating sentence independence" :=
  sorry

theorem question_3 (explanation: String) : explanation = "The Shang dynasty called it 'Xu,' and the Zhou dynasty called it 'Xiang.'" :=
  sorry

theorem question_4 (analysis: String) : analysis = "The statement about the 'ultimate ideal' is incorrect; the original text states that 'enabling people to live and die without regret' is 'the beginning of the King's Way.'" :=
  sorry

end question_1_question_2_question_3_question_4_l196_196496


namespace least_number_subtracted_divisible_l196_196125

theorem least_number_subtracted_divisible (n : ℕ) (divisor : ℕ) (rem : ℕ) :
  n = 427398 → divisor = 15 → n % divisor = rem → rem = 3 → ∃ k : ℕ, n - k = 427395 :=
by
  intros
  use 3
  sorry

end least_number_subtracted_divisible_l196_196125


namespace solve_x_y_l196_196670

theorem solve_x_y (x y : ℝ) (h1 : x^2 + y^2 = 16 * x - 10 * y + 14) (h2 : x - y = 6) : 
  x + y = 3 := 
by 
  sorry

end solve_x_y_l196_196670


namespace find_g_five_l196_196543

theorem find_g_five 
  (g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : g 0 = 1) : g 5 = Real.exp 5 :=
sorry

end find_g_five_l196_196543


namespace intersection_P_Q_l196_196958

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0, 1} :=
sorry

end intersection_P_Q_l196_196958


namespace keith_spent_on_tires_l196_196778

noncomputable def money_spent_on_speakers : ℝ := 136.01
noncomputable def money_spent_on_cd_player : ℝ := 139.38
noncomputable def total_expenditure : ℝ := 387.85
noncomputable def total_spent_on_speakers_and_cd_player : ℝ := money_spent_on_speakers + money_spent_on_cd_player
noncomputable def money_spent_on_new_tires : ℝ := total_expenditure - total_spent_on_speakers_and_cd_player

theorem keith_spent_on_tires :
  money_spent_on_new_tires = 112.46 :=
by
  sorry

end keith_spent_on_tires_l196_196778


namespace solve_equation_l196_196535

theorem solve_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x * (x - 1)) = 0) ↔ x = 2 :=
by
  sorry

end solve_equation_l196_196535


namespace jasmine_stops_at_S_l196_196666

-- Definitions of the given conditions
def circumference : ℕ := 60
def total_distance : ℕ := 5400
def quadrants : ℕ := 4
def laps (distance circumference : ℕ) := distance / circumference
def isMultiple (a b : ℕ) := b ∣ a
def onSamePoint (distance circumference : ℕ) := (distance % circumference) = 0

-- The theorem to be proved: Jasmine stops at point S after running the total distance
theorem jasmine_stops_at_S 
  (circumference : ℕ) (total_distance : ℕ) (quadrants : ℕ)
  (h1 : circumference = 60) 
  (h2 : total_distance = 5400)
  (h3 : quadrants = 4)
  (h4 : laps total_distance circumference = 90)
  (h5 : isMultiple total_distance circumference)
  : onSamePoint total_distance circumference := 
  sorry

end jasmine_stops_at_S_l196_196666


namespace total_movies_shown_l196_196006

-- Define the conditions of the problem
def screens := 6
def open_hours := 8
def movie_duration := 2

-- Define the statement to prove
theorem total_movies_shown : screens * (open_hours / movie_duration) = 24 := 
by
  sorry

end total_movies_shown_l196_196006


namespace right_angled_triangle_l196_196762

-- Define the lengths of the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- State the theorem using the Pythagorean theorem
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l196_196762


namespace center_in_triangle_probability_l196_196823

theorem center_in_triangle_probability (n : ℕ) :
  let vertices := 2 * n + 1
  let total_ways := vertices.choose 3
  let no_center_ways := vertices * (n.choose 2) / 2
  let p_no_center := no_center_ways / total_ways
  let p_center := 1 - p_no_center
  p_center = (n + 1) / (4 * n - 2) := sorry

end center_in_triangle_probability_l196_196823


namespace cookie_boxes_condition_l196_196217

theorem cookie_boxes_condition (n : ℕ) (M A : ℕ) :
  M = n - 8 ∧ A = n - 2 ∧ M + A < n ∧ M ≥ 1 ∧ A ≥ 1 → n = 9 :=
by
  intro h
  sorry

end cookie_boxes_condition_l196_196217


namespace lost_card_number_l196_196403

theorem lost_card_number (n x : ℕ) (sum_n : ℕ) (h_sum : sum_n = n * (n + 1) / 2) (h_remaining_sum : sum_n - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196403


namespace binomial_theorem_fifth_term_l196_196603
-- Import the necessary library

-- Define the theorem as per the given conditions and required proof
theorem binomial_theorem_fifth_term
  (a x : ℝ) 
  (hx : x ≠ 0) 
  (ha : a ≠ 0) : 
  (Nat.choose 8 4 * (a / x)^4 * (x / a^3)^4 = 70 / a^8) :=
by
  -- Applying the binomial theorem and simplifying the expression
  rw [Nat.choose]
  sorry

end binomial_theorem_fifth_term_l196_196603


namespace yuri_lost_card_l196_196412

theorem yuri_lost_card (n : ℕ) (remaining_sum : ℕ) : 
  n = 14 ∧ remaining_sum = 101 → 
  ∃ x : ℕ, x = (n * (n + 1)) / 2 - remaining_sum ∧ x = 4 :=
by 
  intros h,
  cases h with hn hr,
  use (n * (n + 1)) / 2 - remaining_sum,
  split,
  {
    simp only [hn, hr],
    norm_num,
  },
  {
    exact eq.refl 4,
  }

end yuri_lost_card_l196_196412


namespace max_y_coordinate_l196_196468

noncomputable theory
open Classical

def r (θ : ℝ) := Real.sin (3 * θ)
def y (θ : ℝ) := r θ * Real.sin θ

theorem max_y_coordinate : ∃ θ : ℝ, y θ = 9/8 := sorry

end max_y_coordinate_l196_196468


namespace group_for_2019_is_63_l196_196986

def last_term_of_group (n : ℕ) : ℕ := (n * (n + 1)) / 2 + n

theorem group_for_2019_is_63 :
  ∃ n : ℕ, (2015 < 2019 ∧ 2019 ≤ 2079) :=
by
  sorry

end group_for_2019_is_63_l196_196986


namespace M_inter_N_eq_neg2_l196_196895

variable M : Set ℤ := { -2, -1, 0, 1, 2 }
variable N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem M_inter_N_eq_neg2 : (M ∩ N : Set ℝ) = { -2 } := by
  sorry

end M_inter_N_eq_neg2_l196_196895


namespace solve_for_x_l196_196634

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := 
sorry

end solve_for_x_l196_196634


namespace number_of_ways_to_choose_cooks_l196_196220

theorem number_of_ways_to_choose_cooks : ∀ n k : ℕ, n = 5 → k = 3 → Fintype.card {s : Finset (Fin 5) // s.card = k} = 10 :=
by
  intros n k hn hk,
  rw [hn, hk],
  norm_num,
  sorry

end number_of_ways_to_choose_cooks_l196_196220


namespace total_digits_in_book_l196_196851

open Nat

theorem total_digits_in_book (n : Nat) (h : n = 10000) : 
    let pages_1_9 := 9
    let pages_10_99 := 90 * 2
    let pages_100_999 := 900 * 3
    let pages_1000_9999 := 9000 * 4
    let page_10000 := 5
    pages_1_9 + pages_10_99 + pages_100_999 + pages_1000_9999 + page_10000 = 38894 :=
by
    sorry

end total_digits_in_book_l196_196851


namespace least_element_of_special_set_l196_196210

theorem least_element_of_special_set :
  ∃ T : Finset ℕ, T ⊆ Finset.range 16 ∧ T.card = 7 ∧
    (∀ {x y : ℕ}, x ∈ T → y ∈ T → x < y → ¬ (y % x = 0)) ∧ 
    (∀ {z : ℕ}, z ∈ T → ∀ {x y : ℕ}, x ≠ y → x ∈ T → y ∈ T → z ≠ x + y) ∧
    ∀ (x : ℕ), x ∈ T → x ≥ 4 :=
sorry

end least_element_of_special_set_l196_196210


namespace multiply_binomials_l196_196522

variable (x : ℝ)

theorem multiply_binomials :
  (4 * x - 3) * (x + 7) = 4 * x ^ 2 + 25 * x - 21 :=
by
  sorry

end multiply_binomials_l196_196522


namespace nonempty_solution_set_range_l196_196236

theorem nonempty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := sorry

end nonempty_solution_set_range_l196_196236


namespace train_speed_km_per_hr_l196_196014

-- Definitions for the conditions
def length_of_train_meters : ℕ := 250
def time_to_cross_pole_seconds : ℕ := 10

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℚ := m / 1000
def seconds_to_hours (s : ℕ) : ℚ := s / 3600

-- Theorem stating that the speed of the train is 90 km/hr
theorem train_speed_km_per_hr : 
  meters_to_kilometers length_of_train_meters / seconds_to_hours time_to_cross_pole_seconds = 90 := 
by 
  -- We skip the actual proof with sorry
  sorry

end train_speed_km_per_hr_l196_196014


namespace intersection_M_N_l196_196908

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l196_196908


namespace tempo_insured_fraction_l196_196144

theorem tempo_insured_fraction (premium : ℝ) (rate : ℝ) (original_value : ℝ) (h1 : premium = 300) (h2 : rate = 0.03) (h3 : original_value = 14000) : 
  premium / rate / original_value = 5 / 7 :=
by 
  sorry

end tempo_insured_fraction_l196_196144


namespace number_of_non_empty_proper_subsets_of_A_l196_196482

noncomputable def A : Set ℤ := { x : ℤ | -1 < x ∧ x ≤ 2 }

theorem number_of_non_empty_proper_subsets_of_A : 
  (∃ (A : Set ℤ), A = { x : ℤ | -1 < x ∧ x ≤ 2 }) → 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_non_empty_proper_subsets_of_A_l196_196482


namespace theater_seats_l196_196853

theorem theater_seats (x y t : ℕ) (h1 : x = 532) (h2 : y = 218) (h3 : t = x + y) : t = 750 := 
by 
  rw [h1, h2] at h3
  exact h3

end theater_seats_l196_196853


namespace lost_card_number_l196_196407

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l196_196407


namespace set_intersection_l196_196789

open Set

variable (x : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := { x | |x - 1| > 2 }
def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }

theorem set_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 2 < x ∧ x ≤ 3 := sorry

end set_intersection_l196_196789


namespace intersection_M_N_l196_196916

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l196_196916


namespace arithmetic_sequence_S30_l196_196500

theorem arithmetic_sequence_S30
  (S : ℕ → ℕ)
  (h_arith_seq: ∀ m : ℕ, 2 * (S (2 * m) - S m) = S m + S (3 * m) - S (2 * m))
  (h_S10: S 10 = 4)
  (h_S20: S 20 = 20) :
  S 30 = 48 := 
by
  sorry

end arithmetic_sequence_S30_l196_196500


namespace num_students_scoring_between_80_and_90_l196_196198

noncomputable def class_size : ℕ := 48
noncomputable def mean : ℝ := 80
noncomputable def variance : ℝ := 100

theorem num_students_scoring_between_80_and_90 :
  let σ := Real.sqrt variance,
      Z_80 := (80 - mean) / σ,
      Z_90 := (90 - mean) / σ,
      P_80 := Real.cdf_normal 0 σ mean 80,
      P_90 := Real.cdf_normal 0 σ mean 90,
      prob := P_90 - P_80,
      students := class_size * prob in
  students ≈ 16 := 
by
  sorry

end num_students_scoring_between_80_and_90_l196_196198


namespace total_study_time_l196_196541

theorem total_study_time
  (weeks : ℕ) (weekday_hours : ℕ) (weekend_saturday_hours : ℕ) (weekend_sunday_hours : ℕ)
  (H1 : weeks = 15)
  (H2 : ∀ i : ℕ, i < 5 → weekday_hours = 3)
  (H3 : weekend_saturday_hours = 4)
  (H4 : weekend_sunday_hours = 5) :
  let total_weekday_hours := 5 * weekday_hours in
  let total_weekend_hours := weekend_saturday_hours + weekend_sunday_hours in
  let total_week_hours := total_weekday_hours + total_weekend_hours in
  let total_semester_hours := total_week_hours * weeks in
  total_semester_hours = 360 := by
    sorry

end total_study_time_l196_196541


namespace squirrel_spiral_path_height_l196_196582

-- Define the conditions
def spiralPath (circumference rise totalDistance : ℝ) : Prop :=
  ∃ (numberOfCircuits : ℝ), numberOfCircuits = totalDistance / circumference ∧ numberOfCircuits * rise = totalDistance

-- Define the height of the post proof
theorem squirrel_spiral_path_height : 
  let circumference := 2 -- feet
  let rise := 4 -- feet
  let totalDistance := 8 -- feet
  let height := 16 -- feet
  spiralPath circumference rise totalDistance → height = (totalDistance / circumference) * rise :=
by
  intro h
  sorry

end squirrel_spiral_path_height_l196_196582


namespace constant_term_expansion_l196_196322

theorem constant_term_expansion (n : ℕ) (x : ℝ) (h1 : n = 6) (h2 : ∀ r : ℕ, 0 ≤ r ∧ r ≤ n → (nat.choose n r) ≤ (nat.choose n 4)) :
  let T := (nat.choose 6 4) * (x^2)^2 * (1/x)^4
  in T = 15 :=
by
  sorry

end constant_term_expansion_l196_196322


namespace unique_diff_subset_l196_196970

noncomputable def exists_unique_diff_subset : Prop :=
  ∃ S : Set ℕ, 
    (∀ n : ℕ, n > 0 → ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ n = a - b)

theorem unique_diff_subset : exists_unique_diff_subset :=
  sorry

end unique_diff_subset_l196_196970


namespace intersection_M_N_l196_196915

-- Definitions based on the conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 6 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = {-2} :=
by
  sorry

end intersection_M_N_l196_196915


namespace cubic_transform_l196_196968

theorem cubic_transform (A B C x z β : ℝ) (h₁ : z = x + β) (h₂ : 3 * β + A = 0) :
  z^3 + A * z^2 + B * z + C = 0 ↔ x^3 + (B - (A^2 / 3)) * x + (C - A * B / 3 + 2 * A^3 / 27) = 0 :=
sorry

end cubic_transform_l196_196968


namespace solve_equation_l196_196602

-- Define the conditions
def satisfies_equation (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1

-- Theorem statement
theorem solve_equation : ∀ n m : ℕ, satisfies_equation n m ↔ (n = 2 ∧ m = 2) := 
by { sorry }

end solve_equation_l196_196602


namespace area_difference_l196_196937

theorem area_difference (radius1 radius2 : ℝ) (pi : ℝ) (h1 : radius1 = 15) (h2 : radius2 = 14 / 2) :
  pi * radius1 ^ 2 - pi * radius2 ^ 2 = 176 * pi :=
by 
  sorry

end area_difference_l196_196937


namespace test_scores_ordering_l196_196346

variable (M Q S Z K : ℕ)
variable (M_thinks_lowest : M > K)
variable (Q_thinks_same : Q = K)
variable (S_thinks_not_highest : S < K)
variable (Z_thinks_not_middle : (Z < S ∨ Z > M))

theorem test_scores_ordering : (Z < S) ∧ (S < Q) ∧ (Q < M) := by
  -- proof
  sorry

end test_scores_ordering_l196_196346


namespace angle_of_elevation_proof_l196_196996

noncomputable def height_of_lighthouse : ℝ := 100

noncomputable def distance_between_ships : ℝ := 273.2050807568877

noncomputable def angle_of_elevation_second_ship : ℝ := 45

noncomputable def distance_from_second_ship := height_of_lighthouse

noncomputable def distance_from_first_ship := distance_between_ships - distance_from_second_ship

noncomputable def tanθ := height_of_lighthouse / distance_from_first_ship

noncomputable def angle_of_elevation_first_ship := Real.arctan tanθ

theorem angle_of_elevation_proof :
  angle_of_elevation_first_ship = 30 := by
    sorry

end angle_of_elevation_proof_l196_196996


namespace xunzi_statement_l196_196395

/-- 
Given the conditions:
  "If not accumulating small steps, then not reaching a thousand miles."
  Which can be represented as: ¬P → ¬q.
Prove that accumulating small steps (P) is a necessary but not sufficient condition for
reaching a thousand miles (q).
-/
theorem xunzi_statement (P q : Prop) (h : ¬P → ¬q) : (q → P) ∧ ¬(P → q) :=
by sorry

end xunzi_statement_l196_196395


namespace circle_radius_and_diameter_relations_l196_196181

theorem circle_radius_and_diameter_relations
  (r_x r_y r_z A_x A_y A_z d_x d_z : ℝ)
  (hx_circumference : 2 * π * r_x = 18 * π)
  (hx_area : A_x = π * r_x^2)
  (hy_area_eq : A_y = A_x)
  (hz_area_eq : A_z = 4 * A_x)
  (hy_area : A_y = π * r_y^2)
  (hz_area : A_z = π * r_z^2)
  (dx_def : d_x = 2 * r_x)
  (dz_def : d_z = 2 * r_z)
  : r_y = r_z / 2 ∧ d_z = 2 * d_x := 
by 
  sorry

end circle_radius_and_diameter_relations_l196_196181


namespace product_of_two_numbers_l196_196549

variable {x y : ℝ}

theorem product_of_two_numbers (h1 : x + y = 25) (h2 : x - y = 7) : x * y = 144 := by
  sorry

end product_of_two_numbers_l196_196549


namespace ratio_of_distances_l196_196565

-- Definitions based on conditions in a)
variables (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_w : 0 ≤ w)
variables (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0) (h_eq_times : y / w = x / w + (x + y) / (9 * w))

-- The proof statement
theorem ratio_of_distances (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y)
  (h_nonneg_w : 0 ≤ w) (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0)
  (h_eq_times : y / w = x / w + (x + y) / (9 * w)) :
  x / y = 4 / 5 :=
sorry

end ratio_of_distances_l196_196565


namespace polynomial_roots_arithmetic_progression_not_all_real_l196_196461

theorem polynomial_roots_arithmetic_progression_not_all_real :
  ∀ (a : ℝ), (∃ r d : ℂ, r - d ≠ r ∧ r ≠ r + d ∧ r - d + r + (r + d) = 9 ∧ (r - d) * r + (r - d) * (r + d) + r * (r + d) = 33 ∧ d ≠ 0) →
  a = -45 :=
by
  sorry

end polynomial_roots_arithmetic_progression_not_all_real_l196_196461


namespace dice_sum_not_11_l196_196777

/-- Jeremy rolls three standard six-sided dice, with each showing a different number and the product of the numbers on the upper faces is 72.
    Prove that the sum 11 is not possible. --/
theorem dice_sum_not_11 : 
  ∃ (a b c : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ 
    (1 ≤ b ∧ b ≤ 6) ∧ 
    (1 ≤ c ∧ c ≤ 6) ∧ 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
    (a * b * c = 72) ∧ 
    (a > 4 ∨ b > 4 ∨ c > 4) → 
    a + b + c ≠ 11 := 
by
  sorry

end dice_sum_not_11_l196_196777


namespace range_of_x_range_of_a_l196_196511

-- Definitions of the conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (1)
theorem range_of_x (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 :=
by sorry

-- Part (2)
theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (p x a) → ¬ (q x)) : 1 < a ∧ a ≤ 2 :=
by sorry

end range_of_x_range_of_a_l196_196511


namespace intersection_of_line_with_x_axis_l196_196811

theorem intersection_of_line_with_x_axis 
  (k : ℝ) 
  (h : ∀ x y : ℝ, y = k * x + 4 → (x = -1 ∧ y = 2)) 
  : ∃ x : ℝ, (2 : ℝ) * x + 4 = 0 ∧ x = -2 :=
by {
  sorry
}

end intersection_of_line_with_x_axis_l196_196811


namespace annual_increase_rate_l196_196035

theorem annual_increase_rate (r : ℝ) : 
  (6400 * (1 + r) * (1 + r) = 8100) → r = 0.125 :=
by sorry

end annual_increase_rate_l196_196035


namespace tan_add_pi_over_four_sin_cos_ratio_l196_196928

-- Definition of angle α with the condition that tanα = 2
def α : ℝ := sorry -- Define α such that tan α = 2

-- The first Lean statement for proving tan(α + π/4) = -3
theorem tan_add_pi_over_four (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- The second Lean statement for proving (sinα + cosα) / (2sinα - cosα) = 1
theorem sin_cos_ratio (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 :=
sorry

end tan_add_pi_over_four_sin_cos_ratio_l196_196928


namespace bonus_received_l196_196580

-- Definitions based on the conditions
def total_sales (S : ℝ) : Prop :=
  S > 10000

def commission (S : ℝ) : ℝ :=
  0.09 * S

def excess_amount (S : ℝ) : ℝ :=
  S - 10000

def additional_commission (S : ℝ) : ℝ :=
  0.03 * (S - 10000)

def total_commission (S : ℝ) : ℝ :=
  commission S + additional_commission S

-- Given the conditions
axiom total_sales_commission : ∀ S : ℝ, total_sales S → total_commission S = 1380

-- The goal is to prove the bonus
theorem bonus_received (S : ℝ) (h : total_sales S) : additional_commission S = 120 := 
by 
  sorry

end bonus_received_l196_196580


namespace energy_consumption_correct_l196_196011

def initial_wattages : List ℕ := [60, 80, 100, 120]

def increased_wattages : List ℕ := initial_wattages.map (λ x => x + (x * 25 / 100))

def combined_wattage (ws : List ℕ) : ℕ := ws.sum

def daily_energy_consumption (cw : ℕ) : ℕ := cw * 6 / 1000

def total_energy_consumption (dec : ℕ) : ℕ := dec * 30

-- Main theorem statement
theorem energy_consumption_correct :
  total_energy_consumption (daily_energy_consumption (combined_wattage increased_wattages)) = 81 := 
sorry

end energy_consumption_correct_l196_196011


namespace perfect_shells_l196_196630

theorem perfect_shells (P_spiral B_spiral P_total : ℕ) 
  (h1 : 52 = 2 * B_spiral)
  (h2 : B_spiral = P_spiral + 21)
  (h3 : P_total = P_spiral + 12) :
  P_total = 17 :=
by
  sorry

end perfect_shells_l196_196630


namespace missing_digit_is_0_l196_196542

/- Define the known digits of the number. -/
def digit1 : ℕ := 6
def digit2 : ℕ := 5
def digit3 : ℕ := 3
def digit4 : ℕ := 4

/- Define the condition that ensures the divisibility by 9. -/
def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

/- The main theorem to prove: the value of the missing digit d is 0. -/
theorem missing_digit_is_0 (d : ℕ) 
  (h : is_divisible_by_9 (digit1 + digit2 + digit3 + digit4 + d)) : 
  d = 0 :=
sorry

end missing_digit_is_0_l196_196542


namespace total_sequins_is_162_l196_196328

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end total_sequins_is_162_l196_196328


namespace sin_double_angle_second_quadrant_l196_196302

open Real

theorem sin_double_angle_second_quadrant (α : ℝ) (h₁ : π < α ∧ α < 2 * π) (h₂ : sin (π - α) = 3 / 5) : sin (2 * α) = -24 / 25 :=
by sorry

end sin_double_angle_second_quadrant_l196_196302


namespace student_marks_l196_196436

variable (max_marks : ℕ) (pass_percent : ℕ) (fail_by : ℕ)

theorem student_marks
  (h_max : max_marks = 400)
  (h_pass : pass_percent = 35)
  (h_fail : fail_by = 40)
  : max_marks * pass_percent / 100 - fail_by = 100 :=
by
  sorry

end student_marks_l196_196436


namespace part_a_part_b_l196_196167

open Finset

-- Define Map(n) as the set of all functions from {1, 2, ..., n} to {1, 2, ..., n}
def Map (n : ℕ) := {f : Fin n → Fin n // true}

-- Problem (a)
theorem part_a (n : ℕ) (f : Map n) (h : ∀ x, (f.val x) ≠ x) : (f.val ∘ f.val) ≠ f.val := 
by sorry

-- Problem (b)
theorem part_b (n : ℕ) : 
  ∃ S, S = ∑ k in range (n+1), (nat.choose n k) * k^(n-k) := 
by sorry

end part_a_part_b_l196_196167


namespace least_possible_value_of_z_minus_x_l196_196191

variables (x y z : ℤ)

-- Define the conditions
def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

-- State the theorem
theorem least_possible_value_of_z_minus_x (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) 
    (hx_even : even x) (hy_odd : odd y) (hz_odd : odd z) : z - x = 7 :=
sorry

end least_possible_value_of_z_minus_x_l196_196191


namespace cows_total_l196_196148

theorem cows_total (A M R : ℕ) (h1 : A = 4 * M) (h2 : M = 60) (h3 : A + M = R + 30) : 
  A + M + R = 570 := by
  sorry

end cows_total_l196_196148


namespace orange_is_faster_by_l196_196263

def forest_run_time (distance speed : ℕ) : ℕ := distance / speed
def beach_run_time (distance speed : ℕ) : ℕ := distance / speed
def mountain_run_time (distance speed : ℕ) : ℕ := distance / speed

def total_time_in_minutes (forest_distance forest_speed beach_distance beach_speed mountain_distance mountain_speed : ℕ) : ℕ :=
  (forest_run_time forest_distance forest_speed + beach_run_time beach_distance beach_speed + mountain_run_time mountain_distance mountain_speed) * 60

def apple_total_time := total_time_in_minutes 18 3 6 2 3 1
def mac_total_time := total_time_in_minutes 20 4 8 3 3 1
def orange_total_time := total_time_in_minutes 22 5 10 4 3 2

def combined_time := apple_total_time + mac_total_time
def orange_time_difference := combined_time - orange_total_time

theorem orange_is_faster_by :
  orange_time_difference = 856 := sorry

end orange_is_faster_by_l196_196263


namespace movies_shown_eq_twenty_four_l196_196008

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l196_196008


namespace multiplication_factor_l196_196585

theorem multiplication_factor (k : ℝ) :
  k = (∛(2 / 3 * (5 * sqrt 3 + 3 * sqrt 7))) → 
  k * ∛(5 * sqrt 3 - 3 * sqrt 7) = 2 := by
  sorry

end multiplication_factor_l196_196585


namespace lastNumberIsOneOverSeven_l196_196523

-- Definitions and conditions
def seq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 99 → a k = a (k - 1) * a (k + 1)

def nonZeroSeq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → a k ≠ 0

def firstSeq7 (a : ℕ → ℝ) : Prop :=
  a 1 = 7

-- Theorem statement
theorem lastNumberIsOneOverSeven (a : ℕ → ℝ) :
  seq a → nonZeroSeq a → firstSeq7 a → a 100 = 1 / 7 :=
by
  sorry

end lastNumberIsOneOverSeven_l196_196523


namespace min_red_chips_l196_196130

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ (1 / 3) * w)
  (h2 : b ≤ (1 / 4) * r)
  (h3 : w + b ≥ 70) : r ≥ 72 :=
by
  sorry

end min_red_chips_l196_196130


namespace expected_value_of_winnings_is_4_l196_196431

noncomputable def expected_value_of_winnings : ℕ := 
  let outcomes := [7, 6, 5, 4, 4, 3, 2, 1]
  let total_winnings := outcomes.sum
  total_winnings / 8

theorem expected_value_of_winnings_is_4 :
  expected_value_of_winnings = 4 :=
by
  sorry

end expected_value_of_winnings_is_4_l196_196431


namespace original_hourly_wage_l196_196259

theorem original_hourly_wage 
  (daily_wage_increase : ∀ W : ℝ, 1.60 * W + 10 = 45)
  (work_hours : ℝ := 8) : 
  ∃ W_hourly : ℝ, W_hourly = 2.73 :=
by 
  have W : ℝ := (45 - 10) / 1.60 
  have W_hourly : ℝ := W / work_hours
  use W_hourly 
  sorry

end original_hourly_wage_l196_196259


namespace hexagon_largest_angle_l196_196232

theorem hexagon_largest_angle (x : ℝ) (h : 3 * x + 3 * x + 3 * x + 4 * x + 5 * x + 6 * x = 720) : 
  6 * x = 180 :=
by
  sorry

end hexagon_largest_angle_l196_196232


namespace solution_correct_statements_count_l196_196610

variable (a b : ℚ)

def statement1 (a b : ℚ) : Prop := (a + b > 0) → (a > 0 ∧ b > 0)
def statement2 (a b : ℚ) : Prop := (a + b < 0) → ¬(a < 0 ∧ b < 0)
def statement3 (a b : ℚ) : Prop := (|a| > |b| ∧ (a < 0 ↔ b > 0)) → (a + b > 0)
def statement4 (a b : ℚ) : Prop := (|a| < b) → (a + b > 0)

theorem solution_correct_statements_count : 
  (statement1 a b ∧ statement4 a b ∧ ¬statement2 a b ∧ ¬statement3 a b) → 2 = 2 :=
by
  intro _s
  decide
  sorry

end solution_correct_statements_count_l196_196610


namespace red_light_probability_l196_196868

theorem red_light_probability :
  let red_duration := 30
  let yellow_duration := 5
  let green_duration := 40
  let total_duration := red_duration + yellow_duration + green_duration
  let probability_of_red := (red_duration:ℝ) / total_duration
  probability_of_red = 2 / 5 := by
    sorry

end red_light_probability_l196_196868


namespace max_y_on_graph_l196_196465

theorem max_y_on_graph (θ : ℝ) : ∃ θ, (3 * (sin θ)^2 - 4 * (sin θ)^4) ≤ (3 * (sin (arcsin (sqrt (3 / 8))))^2 - 4 * (sin (arcsin (sqrt (3 / 8))))^4) :=
by
  -- We express the function y
  let y := λ θ : ℝ, 3 * (sin θ)^2 - 4 * (sin θ)^4
  use arcsin (sqrt (3 / 8))
  have h1: y (arcsin (sqrt (3 / 8))) = 3 * (sqrt (3 / 8))^2 - 4 * (sqrt (3 / 8))^4 := sorry
  have h2: ∀ θ : ℝ, y θ ≤ y (arcsin (sqrt (3 / 8))) := sorry
  exact ⟨arcsin (sqrt (3 / 8)), h2 ⟩

end max_y_on_graph_l196_196465


namespace part1_part2_l196_196657

-- Part 1
theorem part1 (n : ℕ) (hn : n ≠ 0) (d : ℕ) (hd : d ∣ 2 * n^2) : 
  ∀ m : ℕ, ¬ (m ≠ 0 ∧ m^2 = n^2 + d) :=
by
  sorry 

-- Part 2
theorem part2 (n : ℕ) (hn : n ≠ 0) : 
  ∀ d : ℕ, (d ∣ 3 * n^2 ∧ ∃ m : ℕ, m ≠ 0 ∧ m^2 = n^2 + d) → d = 3 * n^2 :=
by
  sorry

end part1_part2_l196_196657


namespace mod_neg_result_l196_196879

-- Define the hypothesis as the residue equivalence and positive range constraint.
theorem mod_neg_result : 
  ∀ (a b : ℤ), (-1277 : ℤ) % 32 = 3 := by
  sorry

end mod_neg_result_l196_196879


namespace handshakes_at_gathering_l196_196675

-- Define the number of couples
def couples := 6

-- Define the total number of people
def total_people := 2 * couples

-- Each person shakes hands with 10 others (excluding their spouse)
def handshakes_per_person := 10

-- Total handshakes counted with pairs counted twice
def total_handshakes := total_people * handshakes_per_person / 2

-- The theorem to prove the number of handshakes
theorem handshakes_at_gathering : total_handshakes = 60 :=
by
  sorry

end handshakes_at_gathering_l196_196675


namespace bran_amount_to_pay_l196_196025

variable (tuition_fee scholarship_percentage monthly_income payment_duration : ℝ)

def amount_covered_by_scholarship : ℝ := scholarship_percentage * tuition_fee

def remaining_after_scholarship : ℝ := tuition_fee - amount_covered_by_scholarship

def total_earnings_part_time_job : ℝ := monthly_income * payment_duration

def amount_still_to_pay : ℝ := remaining_after_scholarship - total_earnings_part_time_job

theorem bran_amount_to_pay (h_tuition_fee : tuition_fee = 90)
                          (h_scholarship_percentage : scholarship_percentage = 0.30)
                          (h_monthly_income : monthly_income = 15)
                          (h_payment_duration : payment_duration = 3) :
  amount_still_to_pay tuition_fee scholarship_percentage monthly_income payment_duration = 18 := 
by
  sorry

end bran_amount_to_pay_l196_196025


namespace z_when_y_six_l196_196807

theorem z_when_y_six
    (k : ℝ)
    (h1 : ∀ y (z : ℝ), y^2 * Real.sqrt z = k)
    (h2 : ∃ (y : ℝ) (z : ℝ), y = 3 ∧ z = 4 ∧ y^2 * Real.sqrt z = k) :
  ∃ z : ℝ, y = 6 ∧ z = 1 / 4 := 
sorry

end z_when_y_six_l196_196807


namespace find_value_of_a_l196_196668

theorem find_value_of_a (a : ℝ) (h : 2 - a = 0) : a = 2 :=
by {
  sorry
}

end find_value_of_a_l196_196668


namespace prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l196_196809

noncomputable def prob_TeamA_wins_game : ℝ := 0.6
noncomputable def prob_TeamB_wins_game : ℝ := 0.4

-- Probability of Team A winning 2-1 in a best-of-three
noncomputable def prob_TeamA_wins_2_1 : ℝ := 2 * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game 

-- Probability of Team B winning in a best-of-three
noncomputable def prob_TeamB_wins_2_0 : ℝ := prob_TeamB_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins_2_1 : ℝ := 2 * prob_TeamB_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins : ℝ := prob_TeamB_wins_2_0 + prob_TeamB_wins_2_1

-- Probability of Team A winning in a best-of-three
noncomputable def prob_TeamA_wins_best_of_three : ℝ := 1 - prob_TeamB_wins

-- Probability of Team A winning in a best-of-five
noncomputable def prob_TeamA_wins_3_0 : ℝ := prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamA_wins_game
noncomputable def prob_TeamA_wins_3_1 : ℝ := 3 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)
noncomputable def prob_TeamA_wins_3_2 : ℝ := 6 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)

noncomputable def prob_TeamA_wins_best_of_five : ℝ := prob_TeamA_wins_3_0 + prob_TeamA_wins_3_1 + prob_TeamA_wins_3_2

theorem prob_TeamA_wins_2_1_proof :
  prob_TeamA_wins_2_1 = 0.288 :=
sorry

theorem prob_TeamB_wins_proof :
  prob_TeamB_wins = 0.352 :=
sorry

theorem best_of_five_increases_prob :
  prob_TeamA_wins_best_of_three < prob_TeamA_wins_best_of_five :=
sorry

end prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l196_196809


namespace longer_diagonal_is_116_l196_196579

-- Given conditions
def side_length : ℕ := 65
def short_diagonal : ℕ := 60

-- Prove that the length of the longer diagonal in the rhombus is 116 units.
theorem longer_diagonal_is_116 : 
  let s := side_length
  let d1 := short_diagonal / 2
  let d2 := (s^2 - d1^2).sqrt
  (2 * d2) = 116 :=
by
  sorry

end longer_diagonal_is_116_l196_196579


namespace line_pass_through_point_l196_196894

theorem line_pass_through_point (k b : ℝ) (x1 x2 : ℝ) (h1: b ≠ 0) (h2: x1^2 - k*x1 - b = 0) (h3: x2^2 - k*x2 - b = 0)
(h4: x1 + x2 = k) (h5: x1 * x2 = -b) 
(h6: (k^2 * (-b) + k * b * k + b^2 = b^2) = true) : 
  ∃ (x y : ℝ), (y = k * x + 1) ∧ (x, y) = (0, 1) :=
by
  sorry

end line_pass_through_point_l196_196894


namespace towel_percentage_decrease_l196_196696

theorem towel_percentage_decrease
  (L B: ℝ)
  (original_area : ℝ := L * B)
  (new_length : ℝ := 0.70 * L)
  (new_breadth : ℝ := 0.75 * B)
  (new_area : ℝ := new_length * new_breadth) :
  ((original_area - new_area) / original_area) * 100 = 47.5 := 
by 
  sorry

end towel_percentage_decrease_l196_196696


namespace min_value_of_expression_l196_196738

noncomputable def given_expression (x : ℝ) : ℝ := 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) + Real.sqrt ((2 + x)^2 + x^2)

theorem min_value_of_expression : ∃ x : ℝ, given_expression x = 6 * Real.sqrt 2 := 
by 
  use 0
  sorry

end min_value_of_expression_l196_196738


namespace range_of_b_l196_196300

theorem range_of_b (a b c : ℝ) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 24) : 
  1 ≤ b ∧ b ≤ 5 := 
sorry

end range_of_b_l196_196300


namespace max_cyclic_permutation_sum_l196_196693

open Finset

theorem max_cyclic_permutation_sum (n : ℕ) (h : n ≥ 2) :
  ∃ (x : Fin n → ℕ), (set.univ = { x i | i < n }).perm (set.univ : Finset (Fin n))
  ∧ (∑ i : Fin n, x i * x ((i+1) % n)) = (2 * n^3 + 3 * n^2 - 11 * n + 18) / 6 :=
by sorry

end max_cyclic_permutation_sum_l196_196693


namespace probability_of_qualification_l196_196077

-- Define the probability of hitting a target and the number of shots
def probability_hit : ℝ := 0.4
def number_of_shots : ℕ := 3

-- Define the probability of hitting a specific number of targets
noncomputable def binomial (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the event of qualifying by hitting at least 2 targets
noncomputable def probability_qualify (n : ℕ) (p : ℝ) : ℝ :=
  binomial n 2 p + binomial n 3 p

-- The theorem we want to prove
theorem probability_of_qualification : probability_qualify number_of_shots probability_hit = 0.352 :=
  by sorry

end probability_of_qualification_l196_196077


namespace sum_of_exterior_angles_of_convex_quadrilateral_l196_196686

theorem sum_of_exterior_angles_of_convex_quadrilateral:
  ∀ (α β γ δ : ℝ),
  (α + β + γ + δ = 360) → 
  (∀ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    (θ₁ = 180 - α ∧ θ₂ = 180 - β ∧ θ₃ = 180 - γ ∧ θ₄ = 180 - δ) → 
    θ₁ + θ₂ + θ₃ + θ₄ = 360) := 
by 
  intros α β γ δ h1 θ₁ θ₂ θ₃ θ₄ h2
  rcases h2 with ⟨hα, hβ, hγ, hδ⟩
  rw [hα, hβ, hγ, hδ]
  linarith

end sum_of_exterior_angles_of_convex_quadrilateral_l196_196686


namespace number_of_students_l196_196678

theorem number_of_students 
  (N : ℕ)
  (avg_age : ℕ → ℕ)
  (h1 : avg_age N = 15)
  (h2 : avg_age 5 = 12)
  (h3 : avg_age 9 = 16)
  (h4 : N = 15 ∧ avg_age 1 = 21) : 
  N = 15 :=
by
  sorry

end number_of_students_l196_196678


namespace curve_intersects_self_at_6_6_l196_196724

-- Definitions for the given conditions
def x (t : ℝ) : ℝ := t^2 - 3
def y (t : ℝ) : ℝ := t^4 - t^2 - 9 * t + 6

-- Lean statement stating that the curve intersects itself at the coordinate (6, 6)
theorem curve_intersects_self_at_6_6 :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ x t1 = x t2 ∧ y t1 = y t2 ∧ x t1 = 6 ∧ y t1 = 6 :=
by
  sorry

end curve_intersects_self_at_6_6_l196_196724


namespace smaller_angle_at_3_pm_l196_196264

-- Define the condition for minute hand position at 3:00 p.m.
def minute_hand_position_at_3_pm_deg : ℝ := 0

-- Define the condition for hour hand position at 3:00 p.m.
def hour_hand_position_at_3_pm_deg : ℝ := 90

-- Define the angle between the minute hand and hour hand
def angle_between_hands (minute_deg hour_deg : ℝ) : ℝ :=
  abs (hour_deg - minute_deg)

-- The main theorem we need to prove
theorem smaller_angle_at_3_pm :
  angle_between_hands minute_hand_position_at_3_pm_deg hour_hand_position_at_3_pm_deg = 90 :=
by
  sorry

end smaller_angle_at_3_pm_l196_196264


namespace binary_111_to_decimal_l196_196456

-- Define a function to convert binary list to decimal
def binaryToDecimal (bin : List ℕ) : ℕ :=
  bin.reverse.enumFrom 0 |>.foldl (λ acc ⟨i, b⟩ => acc + b * (2 ^ i)) 0

-- Assert the equivalence between the binary number [1, 1, 1] and its decimal representation 7
theorem binary_111_to_decimal : binaryToDecimal [1, 1, 1] = 7 :=
  by
  sorry

end binary_111_to_decimal_l196_196456


namespace determine_b_l196_196597

noncomputable def Q (x : ℝ) (b : ℝ) : ℝ := x^3 + 3 * x^2 + b * x + 20

theorem determine_b (b : ℝ) :
  (∃ x : ℝ, x = 4 ∧ Q x b = 0) → b = -33 :=
by
  intro h
  rcases h with ⟨_, rfl, hQ⟩
  sorry

end determine_b_l196_196597


namespace find_integers_for_perfect_square_l196_196037

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

theorem find_integers_for_perfect_square :
  {x : ℤ | is_perfect_square (x^4 + x^3 + x^2 + x + 1)} = {-1, 0, 3} :=
by
  sorry

end find_integers_for_perfect_square_l196_196037


namespace greatest_integer_a_exists_l196_196285

theorem greatest_integer_a_exists (a x : ℤ) (h : (x - a) * (x - 7) + 3 = 0) : a ≤ 11 := by
  sorry

end greatest_integer_a_exists_l196_196285


namespace greatest_multiple_of_four_l196_196806

theorem greatest_multiple_of_four (x : ℕ) (hx : x > 0) (h4 : x % 4 = 0) (hcube : x^3 < 800) : x ≤ 8 :=
by {
  sorry
}

end greatest_multiple_of_four_l196_196806


namespace find_x_value_l196_196705

theorem find_x_value :
  ∀ (x : ℝ), 0.3 + 0.1 + 0.4 + x = 1 → x = 0.2 :=
by
  intros x h
  sorry

end find_x_value_l196_196705


namespace rational_expression_equals_3_l196_196093

theorem rational_expression_equals_3 (x : ℝ) (hx : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := 
by
  sorry

end rational_expression_equals_3_l196_196093


namespace summer_has_150_degrees_l196_196973

-- Define the condition that Summer has five more degrees than Jolly,
-- and the combined number of degrees they both have is 295.
theorem summer_has_150_degrees (S J : ℕ) (h1 : S = J + 5) (h2 : S + J = 295) : S = 150 :=
by sorry

end summer_has_150_degrees_l196_196973


namespace total_amount_due_l196_196699

noncomputable def original_bill : ℝ := 500
noncomputable def late_charge_rate : ℝ := 0.02
noncomputable def annual_interest_rate : ℝ := 0.05

theorem total_amount_due (n : ℕ) (initial_amount : ℝ) (late_charge_rate : ℝ) (interest_rate : ℝ) : 
  initial_amount = 500 → 
  late_charge_rate = 0.02 → 
  interest_rate = 0.05 → 
  n = 3 → 
  (initial_amount * (1 + late_charge_rate)^n * (1 + interest_rate) = 557.13) :=
by
  intros h_initial_amount h_late_charge_rate h_interest_rate h_n
  sorry

end total_amount_due_l196_196699


namespace average_birds_seen_correct_l196_196516

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end average_birds_seen_correct_l196_196516


namespace lost_card_l196_196401

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

theorem lost_card (n : ℕ) (h1 : sum_natural n - 101 < n) (h2 : sum_natural (n - 1) ≤ 101) :
  ∃ x : ℕ, x = sum_natural n - 101 ∧ x = 4 :=
begin
  have h_sum : sum_natural 14 = 105,
  {
    unfold sum_natural,
    norm_num,
  },
  use 4,
  split,
  {
    unfold sum_natural at *,
    norm_num,
    rw h_sum,
  },
  {
    norm_num,
  },
  sorry,
end

end lost_card_l196_196401


namespace total_time_hover_layover_two_days_l196_196442

theorem total_time_hover_layover_two_days 
    (hover_pacific_day1 : ℝ)
    (hover_mountain_day1 : ℝ)
    (hover_central_day1 : ℝ)
    (hover_eastern_day1 : ℝ)
    (layover_time : ℝ)
    (speed_increase : ℝ)
    (time_decrease : ℝ) :
    hover_pacific_day1 = 2 →
    hover_mountain_day1 = 3 →
    hover_central_day1 = 4 →
    hover_eastern_day1 = 3 →
    layover_time = 1.5 →
    speed_increase = 0.2 →
    time_decrease = 1.6 →
    hover_pacific_day1 + hover_mountain_day1 + hover_central_day1 + hover_eastern_day1 + 4 * layover_time 
      + (hover_eastern_day1 - (speed_increase * hover_eastern_day1) + hover_central_day1 - (speed_increase * hover_central_day1) 
         + hover_mountain_day1 - (speed_increase * hover_mountain_day1) + hover_pacific_day1 - (speed_increase * hover_pacific_day1)) 
      + 4 * layover_time = 33.6 := 
by
  intros
  sorry

end total_time_hover_layover_two_days_l196_196442


namespace notes_count_l196_196394

theorem notes_count (x : ℕ) (num_2_yuan num_5_yuan num_10_yuan total_notes total_amount : ℕ) 
    (h1 : total_amount = 160)
    (h2 : total_notes = 25)
    (h3 : num_5_yuan = x)
    (h4 : num_10_yuan = x)
    (h5 : num_2_yuan = total_notes - 2 * x)
    (h6 : 2 * num_2_yuan + 5 * num_5_yuan + 10 * num_10_yuan = total_amount) :
    num_5_yuan = 10 ∧ num_10_yuan = 10 ∧ num_2_yuan = 5 :=
by
  sorry

end notes_count_l196_196394


namespace num_arrangement_options_l196_196000

def competition_events := ["kicking shuttlecocks", "jumping rope", "tug-of-war", "pushing the train", "multi-person multi-foot"]

def is_valid_arrangement (arrangement : List String) : Prop :=
  arrangement.length = 5 ∧
  arrangement.getLast? = some "tug-of-war" ∧
  arrangement.get? 0 ≠ some "multi-person multi-foot"

noncomputable def count_valid_arrangements : ℕ :=
  let positions := ["kicking shuttlecocks", "jumping rope", "pushing the train"]
  3 * positions.permutations.length

theorem num_arrangement_options : count_valid_arrangements = 18 :=
by
  sorry

end num_arrangement_options_l196_196000


namespace product_of_real_solutions_of_t_cubed_eq_216_l196_196163

theorem product_of_real_solutions_of_t_cubed_eq_216 : 
  (∃ t : ℝ, t^3 = 216) →
  (∀ t₁ t₂, (t₁ = t₂) → (t₁^3 = 216 → t₂^3 = 216) → (t₁ * t₂ = 6)) :=
by
  sorry

end product_of_real_solutions_of_t_cubed_eq_216_l196_196163


namespace radius_inscribed_circle_ABC_l196_196460

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_inscribed_circle_ABC (hAB : AB = 18) (hAC : AC = 18) (hBC : BC = 24) :
  radius_of_inscribed_circle 18 18 24 = 2 * Real.sqrt 6 := by
  sorry

end radius_inscribed_circle_ABC_l196_196460


namespace intersection_M_N_is_correct_l196_196920

def M := {-2, -1, 0, 1, 2}
def N := {x | x^2 - x - 6 >= 0}
def correct_intersection := {-2}
theorem intersection_M_N_is_correct : M ∩ N = correct_intersection := 
by
    sorry

end intersection_M_N_is_correct_l196_196920


namespace checkerboard_sum_is_328_l196_196573

def checkerboard_sum : Nat :=
  1 + 2 + 9 + 8 + 73 + 74 + 81 + 80

theorem checkerboard_sum_is_328 : checkerboard_sum = 328 := by
  sorry

end checkerboard_sum_is_328_l196_196573


namespace qs_length_l196_196559

theorem qs_length
  (PQR : Triangle)
  (PQ QR PR : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 8)
  (h3 : PR = 9)
  (bugs_meet_half_perimeter : PQ + QR + PR = 24)
  (bugs_meet_distance : PQ + qs = 12) :
  qs = 5 :=
by
  sorry

end qs_length_l196_196559


namespace pipe_Q_drain_portion_l196_196350

noncomputable def portion_liquid_drain_by_Q (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) : ℝ :=
  let rate_P := 1 / T_P
  let rate_Q := 1 / T_Q
  let rate_R := 1 / T_R
  let combined_rate := rate_P + rate_Q + rate_R
  (rate_Q / combined_rate)

theorem pipe_Q_drain_portion (T_Q T_P T_R : ℝ) (h1 : T_P = 3 / 4 * T_Q) (h2 : T_R = T_P) :
  portion_liquid_drain_by_Q T_Q T_P T_R h1 h2 = 3 / 11 :=
by
  sorry

end pipe_Q_drain_portion_l196_196350


namespace fraction_equal_decimal_l196_196594

theorem fraction_equal_decimal : (1 / 4) = 0.25 :=
sorry

end fraction_equal_decimal_l196_196594


namespace total_sequins_correct_l196_196326

def blue_rows : ℕ := 6
def blue_columns : ℕ := 8
def purple_rows : ℕ := 5
def purple_columns : ℕ := 12
def green_rows : ℕ := 9
def green_columns : ℕ := 6

def total_sequins : ℕ :=
  (blue_rows * blue_columns) + (purple_rows * purple_columns) + (green_rows * green_columns)

theorem total_sequins_correct : total_sequins = 162 := by
  sorry

end total_sequins_correct_l196_196326


namespace range_of_c_div_a_l196_196922

-- Define the conditions and variables
variables (a b c : ℝ)

-- Define the given conditions
def conditions : Prop :=
  (a ≥ b ∧ b ≥ c) ∧ (a + b + c = 0)

-- Define the range of values for c / a
def range_for_c_div_a : Prop :=
  -2 ≤ c / a ∧ c / a ≤ -1/2

-- The theorem statement to prove
theorem range_of_c_div_a (h : conditions a b c) : range_for_c_div_a a c := 
  sorry

end range_of_c_div_a_l196_196922


namespace largest_of_three_consecutive_integers_l196_196373

theorem largest_of_three_consecutive_integers (N : ℤ) (h : N + (N + 1) + (N + 2) = 18) : N + 2 = 7 :=
sorry

end largest_of_three_consecutive_integers_l196_196373


namespace inequality_solution_l196_196933

theorem inequality_solution (a b c : ℝ) :
  (∀ x : ℝ, -4 < x ∧ x < 7 → a * x^2 + b * x + c > 0) →
  (∀ x : ℝ, (x < -1/7 ∨ x > 1/4) ↔ c * x^2 - b * x + a > 0) :=
by
  sorry

end inequality_solution_l196_196933


namespace multiplicative_inverse_CD_mod_1000000_l196_196787

theorem multiplicative_inverse_CD_mod_1000000 :
  let C := 123456
  let D := 166666
  let M := 48
  M * (C * D) % 1000000 = 1 := by
  sorry

end multiplicative_inverse_CD_mod_1000000_l196_196787
