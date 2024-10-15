import Mathlib

namespace NUMINAMATH_GPT_greater_number_l828_82897

theorem greater_number (x y : ℕ) (h1 : x + y = 22) (h2 : x - y = 4) : x = 13 := 
by sorry

end NUMINAMATH_GPT_greater_number_l828_82897


namespace NUMINAMATH_GPT_factorize_expression_l828_82829

variable (a b c : ℝ)

theorem factorize_expression : 
  (a - 2 * b) * (a - 2 * b - 4) + 4 - c ^ 2 = ((a - 2 * b) - 2 + c) * ((a - 2 * b) - 2 - c) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l828_82829


namespace NUMINAMATH_GPT_root_in_interval_k_eq_2_l828_82874

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem root_in_interval_k_eq_2
  (k : ℤ)
  (h1 : 0 < f 2)
  (h2 : Real.log 2 + 2 * 2 - 5 < 0)
  (h3 : Real.log 3 + 2 * 3 - 5 > 0) 
  (h4 : f (k : ℝ) * f (k + 1 : ℝ) < 0) :
  k = 2 := 
sorry

end NUMINAMATH_GPT_root_in_interval_k_eq_2_l828_82874


namespace NUMINAMATH_GPT_smallest_sum_arith_geo_sequence_l828_82815

theorem smallest_sum_arith_geo_sequence :
  ∃ (X Y Z W : ℕ),
    X < Y ∧ Y < Z ∧ Z < W ∧
    (2 * Y = X + Z) ∧
    (Y ^ 2 = Z * X) ∧
    (Z / Y = 7 / 4) ∧
    (X + Y + Z + W = 97) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_arith_geo_sequence_l828_82815


namespace NUMINAMATH_GPT_volume_of_region_l828_82886

theorem volume_of_region : 
  ∀ (x y z : ℝ),
  abs (x + y + z) + abs (x - y + z) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
  ∃ V : ℝ, V = 62.5 :=
  sorry

end NUMINAMATH_GPT_volume_of_region_l828_82886


namespace NUMINAMATH_GPT_revenue_from_full_price_tickets_l828_82895

theorem revenue_from_full_price_tickets (f h p : ℝ) (total_tickets : f + h = 160) (total_revenue : f * p + h * (p / 2) = 2400) :
  f * p = 960 :=
sorry

end NUMINAMATH_GPT_revenue_from_full_price_tickets_l828_82895


namespace NUMINAMATH_GPT_sum_of_specific_terms_in_arithmetic_sequence_l828_82813

theorem sum_of_specific_terms_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_S11 : S 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_specific_terms_in_arithmetic_sequence_l828_82813


namespace NUMINAMATH_GPT_find_interest_rate_l828_82867

theorem find_interest_rate (P r : ℝ) 
  (h1 : 460 = P * (1 + 3 * r)) 
  (h2 : 560 = P * (1 + 8 * r)) : 
  r = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l828_82867


namespace NUMINAMATH_GPT_real_solutions_to_system_l828_82859

theorem real_solutions_to_system (x y : ℝ) (h1 : x^3 + y^3 = 1) (h2 : x^4 + y^4 = 1) :
  (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_real_solutions_to_system_l828_82859


namespace NUMINAMATH_GPT_cos_alpha_value_l828_82802

-- Definitions for conditions and theorem statement

def condition_1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def condition_2 (α : ℝ) : Prop :=
  Real.cos (Real.pi / 3 + α) = 1 / 3

theorem cos_alpha_value (α : ℝ) (h1 : condition_1 α) (h2 : condition_2 α) :
  Real.cos α = (1 + 2 * Real.sqrt 6) / 6 := sorry

end NUMINAMATH_GPT_cos_alpha_value_l828_82802


namespace NUMINAMATH_GPT_find_line_eqn_from_bisected_chord_l828_82856

noncomputable def line_eqn_from_bisected_chord (x y : ℝ) : Prop :=
  2 * x + y - 3 = 0

theorem find_line_eqn_from_bisected_chord (
  A B : ℝ × ℝ) 
  (hA :  (A.1^2) / 2 + (A.2^2) / 4 = 1)
  (hB :  (B.1^2) / 2 + (B.2^2) / 4 = 1)
  (h_mid : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  line_eqn_from_bisected_chord 1 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_line_eqn_from_bisected_chord_l828_82856


namespace NUMINAMATH_GPT_verify_calculations_l828_82872

theorem verify_calculations (m n x y a b : ℝ) :
  (2 * m - 3 * n) ^ 2 = 4 * m ^ 2 - 12 * m * n + 9 * n ^ 2 ∧
  (-x + y) ^ 2 = x ^ 2 - 2 * x * y + y ^ 2 ∧
  (a + 2 * b) * (a - 2 * b) = a ^ 2 - 4 * b ^ 2 ∧
  (-2 * x ^ 2 * y ^ 2) ^ 3 / (- x * y) ^ 3 ≠ -2 * x ^ 3 * y ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_verify_calculations_l828_82872


namespace NUMINAMATH_GPT_directrix_of_parabola_l828_82842

theorem directrix_of_parabola (x y : ℝ) (h : y = (1/4) * x^2) : y = -1 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l828_82842


namespace NUMINAMATH_GPT_exterior_angle_of_regular_pentagon_l828_82851

theorem exterior_angle_of_regular_pentagon : 
  (360 / 5) = 72 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_of_regular_pentagon_l828_82851


namespace NUMINAMATH_GPT_price_36kg_apples_l828_82845

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

end NUMINAMATH_GPT_price_36kg_apples_l828_82845


namespace NUMINAMATH_GPT_complex_pow_six_eq_eight_i_l828_82852

theorem complex_pow_six_eq_eight_i (i : ℂ) (h : i^2 = -1) : (1 - i) ^ 6 = 8 * i := by
  sorry

end NUMINAMATH_GPT_complex_pow_six_eq_eight_i_l828_82852


namespace NUMINAMATH_GPT_trajectory_no_intersection_distance_AB_l828_82890

variable (M : Type) [MetricSpace M]

-- Point M on the plane
variable (M : ℝ × ℝ)

-- Given conditions
def condition1 (M : ℝ × ℝ) : Prop := 
  (Real.sqrt ((M.1 - 8)^2 + M.2^2) = 2 * Real.sqrt ((M.1 - 2)^2 + M.2^2))

-- 1. Proving the trajectory C of M
theorem trajectory (M : ℝ × ℝ) (h : condition1 M) : M.1^2 + M.2^2 = 16 :=
by
  sorry

-- 2. Range of values for k such that y = kx - 5 does not intersect trajectory C
theorem no_intersection (k : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 16 → y ≠ k * x - 5) ↔ (-3 / 4 < k ∧ k < 3 / 4) :=
by
  sorry

-- 3. Distance between intersection points A and B of given circles
def intersection_condition (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16) ∧ (x^2 + y^2 - 8 * x - 8 * y + 16 = 0)

theorem distance_AB (A B : ℝ × ℝ) (hA : intersection_condition A.1 A.2) (hB : intersection_condition B.1 B.2) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_no_intersection_distance_AB_l828_82890


namespace NUMINAMATH_GPT_length_of_yellow_line_l828_82898

theorem length_of_yellow_line
  (w1 w2 w3 w4 : ℝ) (path_width : ℝ) (middle_line_dist : ℝ)
  (h1 : w1 = 40) (h2 : w2 = 10) (h3 : w3 = 20) (h4 : w4 = 30) (h5 : path_width = 5) (h6 : middle_line_dist = 2.5) :
  w1 - path_width * middle_line_dist/2 + w2 + w3 + w4 - path_width * middle_line_dist/2 = 95 :=
by sorry

end NUMINAMATH_GPT_length_of_yellow_line_l828_82898


namespace NUMINAMATH_GPT_two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l828_82877

-- Define the number of boys and girls
def boys : ℕ := 2
def girls : ℕ := 3
def total_people : ℕ := boys + girls

-- Define assumptions about arrangements
def arrangements_in_two_rows : ℕ := sorry
def arrangements_with_person_A_not_head_tail : ℕ := sorry
def arrangements_with_girls_together : ℕ := sorry
def arrangements_with_boys_not_adjacent : ℕ := sorry

-- State the mathematical equivalence proof problems
theorem two_rows_arrangement : arrangements_in_two_rows = 60 := 
  sorry

theorem person_A_not_head_tail_arrangement : arrangements_with_person_A_not_head_tail = 72 := 
  sorry

theorem girls_together_arrangement : arrangements_with_girls_together = 36 := 
  sorry

theorem boys_not_adjacent_arrangement : arrangements_with_boys_not_adjacent = 72 := 
  sorry

end NUMINAMATH_GPT_two_rows_arrangement_person_A_not_head_tail_arrangement_girls_together_arrangement_boys_not_adjacent_arrangement_l828_82877


namespace NUMINAMATH_GPT_generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l828_82807

-- Define the number five as 4, as we are using five 4s
def four := 4

-- Now prove that each number from 1 to 22 can be generated using the conditions
theorem generate_1 : 1 = (4 / 4) * (4 / 4) := sorry
theorem generate_2 : 2 = (4 / 4) + (4 / 4) := sorry
theorem generate_3 : 3 = ((4 + 4 + 4) / 4) - (4 / 4) := sorry
theorem generate_4 : 4 = 4 * (4 - 4) + 4 := sorry
theorem generate_5 : 5 = 4 + (4 / 4) := sorry
theorem generate_6 : 6 = 4 + 4 - (4 / 4) := sorry
theorem generate_7 : 7 = 4 + 4 - (4 / 4) := sorry
theorem generate_8 : 8 = 4 + 4 := sorry
theorem generate_9 : 9 = 4 + 4 + (4 / 4) := sorry
theorem generate_10 : 10 = 4 * (2 + 4 / 4) := sorry
theorem generate_11 : 11 = 4 * (3 - 1 / 4) := sorry
theorem generate_12 : 12 = 4 + 4 + 4 := sorry
theorem generate_13 : 13 = (4 * 4) - (4 / 4) - 4 := sorry
theorem generate_14 : 14 = 4 * (4 - 1 / 4) := sorry
theorem generate_15 : 15 = 4 * 4 - (4 / 4) - 1 := sorry
theorem generate_16 : 16 = 4 * (4 - (4 - 4) / 4) := sorry
theorem generate_17 : 17 = 4 * (4 + 4 / 4) := sorry
theorem generate_18 : 18 = 4 * 4 + 4 - 4 / 4 := sorry
theorem generate_19 : 19 = 4 + 4 + 4 + 4 + 3 := sorry
theorem generate_20 : 20 = 4 + 4 + 4 + 4 + 4 := sorry
theorem generate_21 : 21 = 4 * 4 + (4 - 1) / 4 := sorry
theorem generate_22 : 22 = (4 * 4 + 4) / 4 := sorry

end NUMINAMATH_GPT_generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l828_82807


namespace NUMINAMATH_GPT_arithmetic_sequence_a11_l828_82833

theorem arithmetic_sequence_a11 (a : ℕ → ℤ) (h_arithmetic : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_a3 : a 3 = 4) (h_a5 : a 5 = 8) : a 11 = 12 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a11_l828_82833


namespace NUMINAMATH_GPT_third_number_in_first_set_l828_82811

theorem third_number_in_first_set (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end NUMINAMATH_GPT_third_number_in_first_set_l828_82811


namespace NUMINAMATH_GPT_total_weight_of_peppers_l828_82857

theorem total_weight_of_peppers
  (green_peppers : ℝ) 
  (red_peppers : ℝ)
  (h_green : green_peppers = 0.33)
  (h_red : red_peppers = 0.33) :
  green_peppers + red_peppers = 0.66 := 
by
  sorry

end NUMINAMATH_GPT_total_weight_of_peppers_l828_82857


namespace NUMINAMATH_GPT_Mitya_age_l828_82826

noncomputable def Mitya_current_age (S M : ℝ) := 
  (S + 11 = M) ∧ (M - S = 2*(S - (M - S)))

theorem Mitya_age (S M : ℝ) (h : Mitya_current_age S M) : M = 27.5 := by
  sorry

end NUMINAMATH_GPT_Mitya_age_l828_82826


namespace NUMINAMATH_GPT_max_levels_passed_prob_pass_three_levels_l828_82819

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

end NUMINAMATH_GPT_max_levels_passed_prob_pass_three_levels_l828_82819


namespace NUMINAMATH_GPT_final_price_set_l828_82871

variable (c ch s : ℕ)
variable (dc dtotal : ℚ)
variable (p_final : ℚ)

def coffee_price : ℕ := 6
def cheesecake_price : ℕ := 10
def sandwich_price : ℕ := 8
def coffee_discount : ℚ := 0.25 * 6
def final_discount : ℚ := 3

theorem final_price_set :
  p_final = (coffee_price - coffee_discount) + cheesecake_price + sandwich_price - final_discount :=
by
  sorry

end NUMINAMATH_GPT_final_price_set_l828_82871


namespace NUMINAMATH_GPT_roots_sum_and_product_l828_82875

theorem roots_sum_and_product (k p : ℝ) (hk : (k / 3) = 9) (hp : (p / 3) = 10) : k + p = 57 := by
  sorry

end NUMINAMATH_GPT_roots_sum_and_product_l828_82875


namespace NUMINAMATH_GPT_framing_needed_l828_82887

def orig_width : ℕ := 5
def orig_height : ℕ := 7
def border_width : ℕ := 3
def doubling_factor : ℕ := 2
def inches_per_foot : ℕ := 12

-- Define the new dimensions after doubling
def new_width := orig_width * doubling_factor
def new_height := orig_height * doubling_factor

-- Define the dimensions after adding the border
def final_width := new_width + 2 * border_width
def final_height := new_height + 2 * border_width

-- Calculate the perimeter in inches
def perimeter := 2 * (final_width + final_height)

-- Convert perimeter to feet and round up if necessary
def framing_feet := (perimeter + inches_per_foot - 1) / inches_per_foot

theorem framing_needed : framing_feet = 6 := by
  sorry

end NUMINAMATH_GPT_framing_needed_l828_82887


namespace NUMINAMATH_GPT_attention_index_proof_l828_82868

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 10 then 100 * a ^ (x / 10) - 60
  else if 10 < x ∧ x ≤ 20 then 340
  else if 20 < x ∧ x ≤ 40 then 640 - 15 * x
  else 0

theorem attention_index_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f 5 a = 140) :
  a = 4 ∧ f 5 4 > f 35 4 ∧ (5 ≤ (x : ℝ) ∧ x ≤ 100 / 3 → f x 4 ≥ 140) :=
by
  sorry

end NUMINAMATH_GPT_attention_index_proof_l828_82868


namespace NUMINAMATH_GPT_average_speed_of_planes_l828_82817

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

end NUMINAMATH_GPT_average_speed_of_planes_l828_82817


namespace NUMINAMATH_GPT_total_animals_l828_82865

variable (rats chihuahuas : ℕ)
variable (h1 : rats = 60)
variable (h2 : rats = 6 * chihuahuas)

theorem total_animals (rats : ℕ) (chihuahuas : ℕ) (h1 : rats = 60) (h2 : rats = 6 * chihuahuas) : rats + chihuahuas = 70 := by
  sorry

end NUMINAMATH_GPT_total_animals_l828_82865


namespace NUMINAMATH_GPT_prove_inequality_l828_82892

theorem prove_inequality (x : ℝ) (h : 3 * x^2 + x - 8 < 0) : -2 < x ∧ x < 4 / 3 :=
sorry

end NUMINAMATH_GPT_prove_inequality_l828_82892


namespace NUMINAMATH_GPT_chess_tournament_games_l828_82861

def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games (n : ℕ) (h : n = 19) : games_played n = 171 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l828_82861


namespace NUMINAMATH_GPT_snail_max_distance_300_meters_l828_82870
-- Import required library

-- Define the problem statement
theorem snail_max_distance_300_meters 
  (n : ℕ) (left_turns : ℕ) (right_turns : ℕ) 
  (total_distance : ℕ)
  (h1 : n = 300)
  (h2 : left_turns = 99)
  (h3 : right_turns = 200)
  (h4 : total_distance = n) : 
  ∃ d : ℝ, d = 100 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_snail_max_distance_300_meters_l828_82870


namespace NUMINAMATH_GPT_find_minimum_value_l828_82855

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + a * |x - 1| + 1

-- The statement of the proof problem
theorem find_minimum_value (a : ℝ) (h : a ≥ 0) :
  (a = 0 → ∀ x, f x a ≥ 1 ∧ ∃ x, f x a = 1) ∧
  ((0 < a ∧ a < 2) → ∀ x, f x a ≥ -a^2 / 4 + a + 1 ∧ ∃ x, f x a = -a^2 / 4 + a + 1) ∧
  (a ≥ 2 → ∀ x, f x a ≥ 2 ∧ ∃ x, f x a = 2) := 
by
  sorry

end NUMINAMATH_GPT_find_minimum_value_l828_82855


namespace NUMINAMATH_GPT_jelly_bean_problem_l828_82806

theorem jelly_bean_problem 
  (x y : ℕ) 
  (h1 : x + y = 1200) 
  (h2 : x = 3 * y - 400) :
  x = 800 := 
sorry

end NUMINAMATH_GPT_jelly_bean_problem_l828_82806


namespace NUMINAMATH_GPT_rightmost_three_digits_of_3_pow_1987_l828_82883

theorem rightmost_three_digits_of_3_pow_1987 :
  3^1987 % 2000 = 187 :=
by sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_3_pow_1987_l828_82883


namespace NUMINAMATH_GPT_cats_needed_to_catch_100_mice_in_time_l828_82889

-- Define the context and given conditions
def cats_mice_catch_time (cats mice minutes : ℕ) : Prop :=
  cats = 5 ∧ mice = 5 ∧ minutes = 5

-- Define the goal
theorem cats_needed_to_catch_100_mice_in_time :
  cats_mice_catch_time 5 5 5 → (∃ t : ℕ, t = 500) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cats_needed_to_catch_100_mice_in_time_l828_82889


namespace NUMINAMATH_GPT_family_members_before_baby_l828_82866

theorem family_members_before_baby 
  (n T : ℕ)
  (h1 : T = 17 * n)
  (h2 : (T + 3 * n + 2) / (n + 1) = 17)
  (h3 : 2 = 2) : n = 5 :=
sorry

end NUMINAMATH_GPT_family_members_before_baby_l828_82866


namespace NUMINAMATH_GPT_not_exist_three_numbers_l828_82847

theorem not_exist_three_numbers :
  ¬ ∃ (a b c : ℝ),
  (b^2 - 4 * a * c > 0 ∧ (-b / a > 0) ∧ (c / a > 0)) ∧
  (b^2 - 4 * a * c > 0 ∧ (-b / a < 0) ∧ (c / a > 0)) :=
by
  sorry

end NUMINAMATH_GPT_not_exist_three_numbers_l828_82847


namespace NUMINAMATH_GPT_trig_identity_example_l828_82838

theorem trig_identity_example (α : Real) (h : Real.cos α = 3 / 5) : Real.cos (2 * α) + Real.sin α ^ 2 = 9 / 25 := by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l828_82838


namespace NUMINAMATH_GPT_triangular_number_difference_30_28_l828_82824

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_number_difference_30_28 : triangular_number 30 - triangular_number 28 = 59 := 
by
  sorry

end NUMINAMATH_GPT_triangular_number_difference_30_28_l828_82824


namespace NUMINAMATH_GPT_multiplicative_inverse_185_mod_341_l828_82841

theorem multiplicative_inverse_185_mod_341 :
  ∃ (b: ℕ), b ≡ 74466 [MOD 341] ∧ 185 * b ≡ 1 [MOD 341] :=
sorry

end NUMINAMATH_GPT_multiplicative_inverse_185_mod_341_l828_82841


namespace NUMINAMATH_GPT_tan_sum_formula_eq_l828_82840

theorem tan_sum_formula_eq {θ : ℝ} (h1 : ∃θ, θ ∈ Set.Ico 0 (2 * Real.pi) 
  ∧ ∃P, P = (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4)) 
  ∧ θ = (3 * Real.pi / 4)) : 
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_sum_formula_eq_l828_82840


namespace NUMINAMATH_GPT_polygon_sides_l828_82820

theorem polygon_sides (n : ℕ) (f : ℕ) (h1 : f = n * (n - 3) / 2) (h2 : 2 * n = f) : n = 7 :=
  by
  sorry

end NUMINAMATH_GPT_polygon_sides_l828_82820


namespace NUMINAMATH_GPT_cost_of_flight_XY_l828_82809

theorem cost_of_flight_XY :
  let d_XY : ℕ := 4800
  let booking_fee : ℕ := 150
  let cost_per_km : ℚ := 0.12
  ∃ cost : ℚ, cost = d_XY * cost_per_km + booking_fee ∧ cost = 726 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_flight_XY_l828_82809


namespace NUMINAMATH_GPT_false_disjunction_implies_both_false_l828_82894

theorem false_disjunction_implies_both_false (p q : Prop) (h : ¬ (p ∨ q)) : ¬ p ∧ ¬ q :=
sorry

end NUMINAMATH_GPT_false_disjunction_implies_both_false_l828_82894


namespace NUMINAMATH_GPT_power_function_increasing_m_eq_2_l828_82869

theorem power_function_increasing_m_eq_2 (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_power_function_increasing_m_eq_2_l828_82869


namespace NUMINAMATH_GPT_martha_butterflies_total_l828_82863

theorem martha_butterflies_total
  (B : ℕ) (Y : ℕ) (black : ℕ)
  (h1 : B = 4)
  (h2 : Y = B / 2)
  (h3 : black = 5) :
  B + Y + black = 11 :=
by {
  -- skip proof 
  sorry 
}

end NUMINAMATH_GPT_martha_butterflies_total_l828_82863


namespace NUMINAMATH_GPT_initial_mean_correctness_l828_82810

variable (M : ℝ)

theorem initial_mean_correctness (h1 : 50 * M + 20 = 50 * 36.5) : M = 36.1 :=
by 
  sorry

end NUMINAMATH_GPT_initial_mean_correctness_l828_82810


namespace NUMINAMATH_GPT_cooking_dishes_time_l828_82801

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

end NUMINAMATH_GPT_cooking_dishes_time_l828_82801


namespace NUMINAMATH_GPT_change_calculation_l828_82825

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple = 4.25) := by
  sorry

end NUMINAMATH_GPT_change_calculation_l828_82825


namespace NUMINAMATH_GPT_four_times_remaining_marbles_l828_82831

theorem four_times_remaining_marbles (initial total_given : ℕ) (remaining : ℕ := initial - total_given) :
  initial = 500 → total_given = 4 * 80 → 4 * remaining = 720 := by sorry

end NUMINAMATH_GPT_four_times_remaining_marbles_l828_82831


namespace NUMINAMATH_GPT_part1_part2_part3_l828_82891

-- Definition of the function
def linear_function (m : ℝ) (x : ℝ) : ℝ :=
  (2 * m + 1) * x + m - 3

-- Part 1: If the graph passes through the origin
theorem part1 (h : linear_function m 0 = 0) : m = 3 :=
by {
  sorry
}

-- Part 2: If the graph is parallel to y = 3x - 3
theorem part2 (h : ∀ x, linear_function m x = 3 * x - 3 → 2 * m + 1 = 3) : m = 1 :=
by {
  sorry
}

-- Part 3: If the graph intersects the y-axis below the x-axis
theorem part3 (h_slope : 2 * m + 1 ≠ 0) (h_intercept : m - 3 < 0) : m < 3 ∧ m ≠ -1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_part3_l828_82891


namespace NUMINAMATH_GPT_initial_books_eq_41_l828_82858

-- Definitions and conditions
def books_sold : ℕ := 33
def books_added : ℕ := 2
def books_remaining : ℕ := 10

-- Proof problem
theorem initial_books_eq_41 (B : ℕ) (h : B - books_sold + books_added = books_remaining) : B = 41 :=
by
  sorry

end NUMINAMATH_GPT_initial_books_eq_41_l828_82858


namespace NUMINAMATH_GPT_raft_time_l828_82879

-- Defining the conditions
def distance_between_villages := 1 -- unit: ed (arbitrary unit)

def steamboat_time := 1 -- unit: hours
def motorboat_time := 3 / 4 -- unit: hours
def motorboat_speed_ratio := 2 -- Motorboat speed is twice steamboat speed in still water

-- Speed equations with the current
def steamboat_speed_with_current (v_s v_c : ℝ) := v_s + v_c = 1 -- unit: ed/hr
def motorboat_speed_with_current (v_s v_c : ℝ) := 2 * v_s + v_c = 4 / 3 -- unit: ed/hr

-- Goal: Prove the time it takes for the raft to travel the same distance downstream
theorem raft_time : ∃ t : ℝ, t = 90 :=
by
  -- Definitions
  let v_s := 1 / 3 -- Speed of the steamboat in still water (derived)
  let v_c := 2 / 3 -- Speed of the current (derived)
  let raft_speed := v_c -- Raft speed equals the speed of the current
  
  -- Calculate the time for the raft to travel the distance
  let raft_time := distance_between_villages / raft_speed
  
  -- Convert time to minutes
  let raft_time_minutes := raft_time * 60
  
  -- Prove the raft time is 90 minutes
  existsi raft_time_minutes
  exact sorry

end NUMINAMATH_GPT_raft_time_l828_82879


namespace NUMINAMATH_GPT_find_y_l828_82860

theorem find_y (x y : ℝ) (h₁ : x = 51) (h₂ : x^3 * y - 2 * x^2 * y + x * y = 51000) : y = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_find_y_l828_82860


namespace NUMINAMATH_GPT_distinct_three_digit_numbers_count_l828_82818

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end NUMINAMATH_GPT_distinct_three_digit_numbers_count_l828_82818


namespace NUMINAMATH_GPT_balance_scale_weights_part_a_balance_scale_weights_part_b_l828_82832

-- Part (a)
theorem balance_scale_weights_part_a (w : List ℕ) (h : w = List.range (90 + 1) \ List.range 1) :
  ¬ ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

-- Part (b)
theorem balance_scale_weights_part_b (w : List ℕ) (h : w = List.range (99 + 1) \ List.range 1) :
  ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

end NUMINAMATH_GPT_balance_scale_weights_part_a_balance_scale_weights_part_b_l828_82832


namespace NUMINAMATH_GPT_equation_of_parallel_line_passing_through_point_l828_82880

variable (x y : ℝ)

def is_point_on_line (x_val y_val : ℝ) (a b c : ℝ) : Prop := a * x_val + b * y_val + c = 0

def is_parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem equation_of_parallel_line_passing_through_point :
  (is_point_on_line (-1) 3 1 (-2) 7) ∧ (is_parallel (1 / 2) (1 / 2)) → (∀ x y, is_point_on_line x y 1 (-2) 7) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_passing_through_point_l828_82880


namespace NUMINAMATH_GPT_pooja_speed_l828_82873

theorem pooja_speed (v : ℝ) 
  (roja_speed : ℝ := 5)
  (distance : ℝ := 32)
  (time : ℝ := 4)
  (h : distance = (roja_speed + v) * time) : v = 3 :=
by
  sorry

end NUMINAMATH_GPT_pooja_speed_l828_82873


namespace NUMINAMATH_GPT_flour_needed_l828_82828

theorem flour_needed (sugar flour : ℕ) (h1 : sugar = 50) (h2 : sugar / 10 = flour) : flour = 5 :=
by
  sorry

end NUMINAMATH_GPT_flour_needed_l828_82828


namespace NUMINAMATH_GPT_scientific_notation_36000_l828_82850

theorem scientific_notation_36000 : 36000 = 3.6 * (10^4) := 
by 
  -- Skipping the proof by adding sorry
  sorry

end NUMINAMATH_GPT_scientific_notation_36000_l828_82850


namespace NUMINAMATH_GPT_speed_of_current_l828_82805

theorem speed_of_current (d : ℝ) (c : ℝ) : 
  ∀ (h1 : ∀ (t : ℝ), d = (30 - c) * (40 / 60)) (h2 : ∀ (t : ℝ), d = (30 + c) * (25 / 60)), 
  c = 90 / 13 := by
  sorry

end NUMINAMATH_GPT_speed_of_current_l828_82805


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l828_82881

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (|x| + |y| ≤ 1 → x^2 + y^2 ≤ 1) ∧ (x^2 + y^2 ≤ 1 → ¬ (|x| + |y| ≤ 1)) :=
sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l828_82881


namespace NUMINAMATH_GPT_percentage_of_women_employees_l828_82882

variable (E W M : ℝ)

-- Introduce conditions
def total_employees_are_married : Prop := 0.60 * E = (1 / 3) * M + 0.6842 * W
def total_employees_count : Prop := W + M = E
def percentage_of_women : Prop := W = 0.7601 * E

-- State the theorem to prove
theorem percentage_of_women_employees :
  total_employees_are_married E W M ∧ total_employees_count E W M → percentage_of_women E W :=
by sorry

end NUMINAMATH_GPT_percentage_of_women_employees_l828_82882


namespace NUMINAMATH_GPT_min_value_of_squares_l828_82884

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧ (∃ (a' b' c' : ℝ), a' = b' ∧ b' = c' ∧ a' + b' + c' = t ∧ a'^2 + b'^2 + c'^2 = t^2 / 3) := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_squares_l828_82884


namespace NUMINAMATH_GPT_solve_floor_fractional_l828_82896

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - floor x

theorem solve_floor_fractional (x : ℝ) :
  floor x * fractional_part x = 2019 * x ↔ x = 0 ∨ x = -1 / 2020 :=
by
  sorry

end NUMINAMATH_GPT_solve_floor_fractional_l828_82896


namespace NUMINAMATH_GPT_football_outcomes_l828_82837

theorem football_outcomes : 
  ∃ (W D L : ℕ), (3 * W + D = 19) ∧ (W + D + L = 14) ∧ 
  ((W = 3 ∧ D = 10 ∧ L = 1) ∨ 
   (W = 4 ∧ D = 7 ∧ L = 3) ∨ 
   (W = 5 ∧ D = 4 ∧ L = 5) ∨ 
   (W = 6 ∧ D = 1 ∧ L = 7)) ∧
  (∀ W' D' L' : ℕ, (3 * W' + D' = 19) → (W' + D' + L' = 14) → 
    (W' = 3 ∧ D' = 10 ∧ L' = 1) ∨ 
    (W' = 4 ∧ D' = 7 ∧ L' = 3) ∨ 
    (W' = 5 ∧ D' = 4 ∧ L' = 5) ∨ 
    (W' = 6 ∧ D' = 1 ∧ L' = 7)) := 
sorry

end NUMINAMATH_GPT_football_outcomes_l828_82837


namespace NUMINAMATH_GPT_find_monday_temperature_l828_82862

theorem find_monday_temperature
  (M T W Th F : ℤ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 35) :
  M = 43 :=
by
  sorry

end NUMINAMATH_GPT_find_monday_temperature_l828_82862


namespace NUMINAMATH_GPT_least_integer_value_abs_l828_82844

theorem least_integer_value_abs (x : ℤ) : 
  (∃ x : ℤ, (abs (3 * x + 5) ≤ 20) ∧ (∀ y : ℤ, (abs (3 * y + 5) ≤ 20) → x ≤ y)) ↔ x = -8 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_abs_l828_82844


namespace NUMINAMATH_GPT_find_Y_l828_82823

theorem find_Y (Y : ℝ) (h : (100 + Y / 90) * 90 = 9020) : Y = 20 := 
by 
  sorry

end NUMINAMATH_GPT_find_Y_l828_82823


namespace NUMINAMATH_GPT_dara_employment_wait_time_l828_82853

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

end NUMINAMATH_GPT_dara_employment_wait_time_l828_82853


namespace NUMINAMATH_GPT_tan_function_constants_l828_82849

theorem tan_function_constants (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_period : b ≠ 0 ∧ ∃ k : ℤ, b * (3 / 2) = k * π) 
(h_pass : a * Real.tan (b * (π / 4)) = 3) : a * b = 2 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_tan_function_constants_l828_82849


namespace NUMINAMATH_GPT_mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l828_82827

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

end NUMINAMATH_GPT_mutually_exclusive_pair2_complementary_pair2_mutually_exclusive_pair4_complementary_pair4_l828_82827


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l828_82835

theorem arithmetic_sequence_common_difference 
    (a_2 : ℕ → ℕ) (S_4 : ℕ) (a_n : ℕ → ℕ → ℕ) (S_n : ℕ → ℕ → ℕ → ℕ)
    (h1 : a_2 2 = 3) (h2 : S_4 = 16) 
    (h3 : ∀ n a_1 d, a_n a_1 n = a_1 + (n-1)*d)
    (h4 : ∀ n a_1 d, S_n n a_1 d = n / 2 * (2*a_1 + (n-1)*d)) : ∃ d, d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l828_82835


namespace NUMINAMATH_GPT_find_a_b_l828_82878

noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def g (x : ℝ) : ℝ := 3 * x - 6

theorem find_a_b (a b : ℝ) (h : ∀ x : ℝ, g (f a b x) = 4 * x + 7) :
  a + b = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l828_82878


namespace NUMINAMATH_GPT_completed_shape_perimeter_602_l828_82839

noncomputable def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

noncomputable def total_perimeter_no_overlap (n : ℕ) (length width : ℝ) : ℝ :=
  n * rectangle_perimeter length width

noncomputable def total_reduction (n : ℕ) (overlap : ℝ) : ℝ :=
  (n - 1) * overlap

noncomputable def overall_perimeter (n : ℕ) (length width overlap : ℝ) : ℝ :=
  total_perimeter_no_overlap n length width - total_reduction n overlap

theorem completed_shape_perimeter_602 :
  overall_perimeter 100 3 1 2 = 602 :=
by
  sorry

end NUMINAMATH_GPT_completed_shape_perimeter_602_l828_82839


namespace NUMINAMATH_GPT_remainder_of_large_product_mod_17_l828_82848

theorem remainder_of_large_product_mod_17 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_large_product_mod_17_l828_82848


namespace NUMINAMATH_GPT_min_students_green_eyes_backpack_no_glasses_l828_82854

theorem min_students_green_eyes_backpack_no_glasses
  (S G B Gl : ℕ)
  (h_S : S = 25)
  (h_G : G = 15)
  (h_B : B = 18)
  (h_Gl : Gl = 6)
  : ∃ x, x ≥ 8 ∧ x + Gl ≤ S ∧ x ≤ min G B :=
sorry

end NUMINAMATH_GPT_min_students_green_eyes_backpack_no_glasses_l828_82854


namespace NUMINAMATH_GPT_evaluate_expression_l828_82843

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 8 = 43046724 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l828_82843


namespace NUMINAMATH_GPT_mutually_exclusive_scoring_l828_82864

-- Define conditions as types
def shoots_twice : Prop := true
def scoring_at_least_once : Prop :=
  ∃ (shot1 shot2 : Bool), shot1 || shot2
def not_scoring_both_times : Prop :=
  ∀ (shot1 shot2 : Bool), ¬(shot1 && shot2)

-- Statement of the problem: Prove the events are mutually exclusive.
theorem mutually_exclusive_scoring :
  shoots_twice → (scoring_at_least_once → not_scoring_both_times → false) :=
by
  intro h_shoots_twice
  intro h_scoring_at_least_once
  intro h_not_scoring_both_times
  sorry

end NUMINAMATH_GPT_mutually_exclusive_scoring_l828_82864


namespace NUMINAMATH_GPT_range_of_p_l828_82803

def A := {x : ℝ | x^2 - x - 2 > 0}
def B := {x : ℝ | (3 / x) - 1 ≥ 0}
def intersection := {x : ℝ | x ∈ A ∧ x ∈ B}
def C (p : ℝ) := {x : ℝ | 2 * x + p ≤ 0}

theorem range_of_p (p : ℝ) : (∀ x : ℝ, x ∈ intersection → x ∈ C p) → p < -6 := by
  sorry

end NUMINAMATH_GPT_range_of_p_l828_82803


namespace NUMINAMATH_GPT_find_radius_of_third_circle_l828_82812

noncomputable def radius_of_third_circle_equals_shaded_region (r1 r2 r3 : ℝ) : Prop :=
  let area_large := Real.pi * (r2 ^ 2)
  let area_small := Real.pi * (r1 ^ 2)
  let area_shaded := area_large - area_small
  let area_third_circle := Real.pi * (r3 ^ 2)
  area_shaded = area_third_circle

theorem find_radius_of_third_circle (r1 r2 : ℝ) (r1_eq : r1 = 17) (r2_eq : r2 = 27) : ∃ r3 : ℝ, r3 = 10 * Real.sqrt 11 ∧ radius_of_third_circle_equals_shaded_region r1 r2 r3 := 
by
  sorry

end NUMINAMATH_GPT_find_radius_of_third_circle_l828_82812


namespace NUMINAMATH_GPT_max_abs_asin_b_l828_82800

theorem max_abs_asin_b (a b c : ℝ) (h : ∀ x : ℝ, |a * (Real.cos x)^2 + b * Real.sin x + c| ≤ 1) :
  ∃ M : ℝ, (∀ x : ℝ, |a * Real.sin x + b| ≤ M) ∧ M = 2 :=
sorry

end NUMINAMATH_GPT_max_abs_asin_b_l828_82800


namespace NUMINAMATH_GPT_length_of_one_string_l828_82899

theorem length_of_one_string (total_length : ℕ) (num_strings : ℕ) (h_total_length : total_length = 98) (h_num_strings : num_strings = 7) : total_length / num_strings = 14 := by
  sorry

end NUMINAMATH_GPT_length_of_one_string_l828_82899


namespace NUMINAMATH_GPT_simplify_expression_l828_82822

theorem simplify_expression (m n : ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 3^(m * n / (m - n))) : 
  ((
    (x^(2 / m) - 9 * x^(2 / n)) *
    ((x^(1 - m))^(1 / m) - 3 * (x^(1 - n))^(1 / n))
  ) / (
    (x^(1 / m) + 3 * x^(1 / n))^2 - 12 * x^((m + n) / (m * n))
  ) = (x^(1 / m) + 3 * x^(1 / n)) / x) := 
sorry

end NUMINAMATH_GPT_simplify_expression_l828_82822


namespace NUMINAMATH_GPT_price_reduction_for_desired_profit_l828_82816

def profit_per_piece (x : ℝ) : ℝ := 40 - x
def pieces_sold_per_day (x : ℝ) : ℝ := 20 + 2 * x

theorem price_reduction_for_desired_profit (x : ℝ) :
  (profit_per_piece x) * (pieces_sold_per_day x) = 1200 ↔ (x = 10 ∨ x = 20) := by
  sorry

end NUMINAMATH_GPT_price_reduction_for_desired_profit_l828_82816


namespace NUMINAMATH_GPT_ratio_of_edges_l828_82804

noncomputable def cube_volume (edge : ℝ) : ℝ := edge^3

theorem ratio_of_edges 
  {a b : ℝ} 
  (h : cube_volume a / cube_volume b = 27) : 
  a / b = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_edges_l828_82804


namespace NUMINAMATH_GPT_arrange_books_l828_82814

noncomputable def numberOfArrangements : Nat :=
  4 * 3 * 6 * (Nat.factorial 9)

theorem arrange_books :
  numberOfArrangements = 26210880 := by
  sorry

end NUMINAMATH_GPT_arrange_books_l828_82814


namespace NUMINAMATH_GPT_race_distance_l828_82885

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l828_82885


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l828_82808

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_value (h : a 3 + a 5 + a 11 + a 13 = 80) : a 8 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l828_82808


namespace NUMINAMATH_GPT_determine_c_l828_82888

noncomputable def fib (n : ℕ) : ℕ :=
match n with
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem determine_c (c d : ℤ) (h1 : ∃ s : ℂ, s^2 - s - 1 = 0 ∧ (c : ℂ) * s^19 + (d : ℂ) * s^18 + 1 = 0) : 
  c = 1597 :=
by
  sorry

end NUMINAMATH_GPT_determine_c_l828_82888


namespace NUMINAMATH_GPT_correct_answers_count_l828_82834

theorem correct_answers_count
  (c w : ℕ)
  (h1 : 4 * c - 2 * w = 420)
  (h2 : c + w = 150) : 
  c = 120 :=
sorry

end NUMINAMATH_GPT_correct_answers_count_l828_82834


namespace NUMINAMATH_GPT_complement_inter_section_l828_82821

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

end NUMINAMATH_GPT_complement_inter_section_l828_82821


namespace NUMINAMATH_GPT_initial_number_of_students_l828_82830

theorem initial_number_of_students (S : ℕ) (h : S + 6 = 37) : S = 31 :=
sorry

end NUMINAMATH_GPT_initial_number_of_students_l828_82830


namespace NUMINAMATH_GPT_probability_kwoes_non_intersect_breads_l828_82876

-- Define the total number of ways to pick 3 points from 7
def total_combinations : ℕ := Nat.choose 7 3

-- Define the number of ways to pick 3 consecutive points from 7
def favorable_combinations : ℕ := 7

-- Define the probability of non-intersection
def non_intersection_probability : ℚ := favorable_combinations / total_combinations

-- Assert the final required probability
theorem probability_kwoes_non_intersect_breads :
  non_intersection_probability = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_kwoes_non_intersect_breads_l828_82876


namespace NUMINAMATH_GPT_prod_is_96_l828_82836

noncomputable def prod_of_nums (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : ℝ := x * y

theorem prod_is_96 (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : prod_of_nums x y h1 h2 = 96 :=
by
  sorry

end NUMINAMATH_GPT_prod_is_96_l828_82836


namespace NUMINAMATH_GPT_sara_initial_quarters_l828_82893

theorem sara_initial_quarters (borrowed quarters_current : ℕ) (q_initial : ℕ) :
  quarters_current = 512 ∧ quarters_borrowed = 271 → q_initial = 783 :=
by
  sorry

end NUMINAMATH_GPT_sara_initial_quarters_l828_82893


namespace NUMINAMATH_GPT_problem_solution_A_problem_solution_C_l828_82846

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

end NUMINAMATH_GPT_problem_solution_A_problem_solution_C_l828_82846
