import Mathlib

namespace NUMINAMATH_GPT_find_b_l768_76870

noncomputable def p (x : ℝ) : ℝ := 3 * x - 8
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) : p (q 3 b) = 10 → b = 6 :=
by
  unfold p q
  intro h
  sorry

end NUMINAMATH_GPT_find_b_l768_76870


namespace NUMINAMATH_GPT_bakery_storage_l768_76899

theorem bakery_storage (S F B : ℕ) 
  (h1 : S * 4 = F * 5) 
  (h2 : F = 10 * B) 
  (h3 : F * 1 = (B + 60) * 8) : S = 3000 :=
sorry

end NUMINAMATH_GPT_bakery_storage_l768_76899


namespace NUMINAMATH_GPT_polynomial_A_l768_76827

variables {a b : ℝ} (A : ℝ)
variables (h1 : 2 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem polynomial_A (h : A / (2 * a * b) = 1 - 4 * a ^ 2) : 
  A = 2 * a * b - 8 * a ^ 3 * b :=
by
  sorry

end NUMINAMATH_GPT_polynomial_A_l768_76827


namespace NUMINAMATH_GPT_good_carrots_total_l768_76842

-- Define the number of carrots picked by Carol and her mother
def carolCarrots := 29
def motherCarrots := 16

-- Define the number of bad carrots
def badCarrots := 7

-- Define the total number of carrots picked by Carol and her mother
def totalCarrots := carolCarrots + motherCarrots

-- Define the total number of good carrots
def goodCarrots := totalCarrots - badCarrots

-- The theorem to prove that the total number of good carrots is 38
theorem good_carrots_total : goodCarrots = 38 := by
  sorry

end NUMINAMATH_GPT_good_carrots_total_l768_76842


namespace NUMINAMATH_GPT_polynomial_sum_of_coefficients_l768_76859

theorem polynomial_sum_of_coefficients {v : ℕ → ℝ} (h1 : v 1 = 7)
  (h2 : ∀ n : ℕ, v (n + 1) - v n = 5 * n - 2) :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_of_coefficients_l768_76859


namespace NUMINAMATH_GPT_minimum_apples_l768_76890

theorem minimum_apples (x : ℕ) : 
  (x ≡ 10 [MOD 3]) ∧ (x ≡ 11 [MOD 4]) ∧ (x ≡ 12 [MOD 5]) → x = 67 :=
sorry

end NUMINAMATH_GPT_minimum_apples_l768_76890


namespace NUMINAMATH_GPT_gumballs_result_l768_76809

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end NUMINAMATH_GPT_gumballs_result_l768_76809


namespace NUMINAMATH_GPT_exists_circular_chain_of_four_l768_76815

-- Let A and B be the two teams, each with a set of players.
variable {A B : Type}
-- Assume there exists a relation "beats" that determines match outcomes.
variable (beats : A → B → Prop)

-- Each player in both teams has at least one win and one loss against the opposite team.
axiom each_has_win_and_loss (a : A) : ∃ b1 b2 : B, beats a b1 ∧ ¬beats a b2 ∧ b1 ≠ b2
axiom each_has_win_and_loss' (b : B) : ∃ a1 a2 : A, beats a1 b ∧ ¬beats a2 b ∧ a1 ≠ a2

-- Main theorem: Exist four players forming a circular chain of victories.
theorem exists_circular_chain_of_four :
  ∃ (a1 a2 : A) (b1 b2 : B), beats a1 b1 ∧ ¬beats a1 b2 ∧ beats a2 b2 ∧ ¬beats a2 b1 ∧ b1 ≠ b2 ∧ a1 ≠ a2 :=
sorry

end NUMINAMATH_GPT_exists_circular_chain_of_four_l768_76815


namespace NUMINAMATH_GPT_variance_stability_l768_76837

theorem variance_stability (S2_A S2_B : ℝ) (hA : S2_A = 1.1) (hB : S2_B = 2.5) : ¬(S2_B < S2_A) :=
by {
  sorry
}

end NUMINAMATH_GPT_variance_stability_l768_76837


namespace NUMINAMATH_GPT_train_speed_initial_l768_76850

variable (x : ℝ)
variable (v : ℝ)
variable (average_speed : ℝ := 40 / 3)
variable (initial_distance : ℝ := x)
variable (initial_speed : ℝ := v)
variable (next_distance : ℝ := 4 * x)
variable (next_speed : ℝ := 20)

theorem train_speed_initial : 
  (5 * x) / ((x / v) + (x / 5)) = 40 / 3 → v = 40 / 7 :=
by
  -- Definition of average speed in the context of the problem
  let t1 := x / v
  let t2 := (4 * x) / 20
  let total_distance := 5 * x
  let total_time := t1 + t2
  have avg_speed_eq : total_distance / total_time = 40 / 3 := by sorry
  sorry

end NUMINAMATH_GPT_train_speed_initial_l768_76850


namespace NUMINAMATH_GPT_correct_number_of_outfits_l768_76847

-- Define the number of each type of clothing
def num_red_shirts := 4
def num_green_shirts := 4
def num_blue_shirts := 4
def num_pants := 10
def num_red_hats := 6
def num_green_hats := 6
def num_blue_hats := 4

-- Define the total number of outfits that meet the conditions
def total_outfits : ℕ :=
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats)) +
  (num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) +
  (num_blue_shirts * num_pants * (num_red_hats + num_green_hats))

-- The proof statement asserting that the total number of valid outfits is 1280
theorem correct_number_of_outfits : total_outfits = 1280 := by
  sorry

end NUMINAMATH_GPT_correct_number_of_outfits_l768_76847


namespace NUMINAMATH_GPT_value_of_expression_l768_76817

noncomputable def line_does_not_pass_through_third_quadrant (k b : ℝ) : Prop :=
k < 0 ∧ b ≥ 0

theorem value_of_expression 
  (k b a e m n c d : ℝ) 
  (h_line : line_does_not_pass_through_third_quadrant k b)
  (h_a_gt_e : a > e)
  (hA : a * k + b = m)
  (hB : e * k + b = n)
  (hC : -m * k + b = c)
  (hD : -n * k + b = d) :
  (m - n) * (c - d) ^ 3 > 0 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l768_76817


namespace NUMINAMATH_GPT_fixed_monthly_charge_l768_76851

variables (F C_J : ℝ)

-- Conditions
def january_bill := F + C_J = 46
def february_bill := F + 2 * C_J = 76

-- The proof goal
theorem fixed_monthly_charge
  (h_jan : january_bill F C_J)
  (h_feb : february_bill F C_J)
  (h_calls : C_J = 30) : F = 16 :=
by sorry

end NUMINAMATH_GPT_fixed_monthly_charge_l768_76851


namespace NUMINAMATH_GPT_three_friends_expenses_l768_76836

theorem three_friends_expenses :
  let ticket_cost := 7
  let number_of_tickets := 3
  let popcorn_cost := 1.5
  let number_of_popcorn := 2
  let milk_tea_cost := 3
  let number_of_milk_tea := 3
  let total_expenses := (ticket_cost * number_of_tickets) + (popcorn_cost * number_of_popcorn) + (milk_tea_cost * number_of_milk_tea)
  let amount_per_friend := total_expenses / 3
  amount_per_friend = 11 := 
by
  sorry

end NUMINAMATH_GPT_three_friends_expenses_l768_76836


namespace NUMINAMATH_GPT_best_fitting_model_l768_76800

theorem best_fitting_model :
  ∀ (R1 R2 R3 R4 : ℝ), R1 = 0.976 → R2 = 0.776 → R3 = 0.076 → R4 = 0.351 →
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  sorry

end NUMINAMATH_GPT_best_fitting_model_l768_76800


namespace NUMINAMATH_GPT_max_x_y3_z4_l768_76844

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_x_y3_z4 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  max_value_expression x y z ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_x_y3_z4_l768_76844


namespace NUMINAMATH_GPT_minimum_even_N_for_A_2015_turns_l768_76821

noncomputable def a (n : ℕ) : ℕ :=
  6 * 2^n - 4

def A_minimum_even_moves_needed (k : ℕ) : ℕ :=
  2015 - 1

theorem minimum_even_N_for_A_2015_turns :
  ∃ N : ℕ, 2 ∣ N ∧ A_minimum_even_moves_needed 2015 ≤ N ∧ a 1007 = 6 * 2^1007 - 4 := by
  sorry

end NUMINAMATH_GPT_minimum_even_N_for_A_2015_turns_l768_76821


namespace NUMINAMATH_GPT_fill_in_blank_with_warning_l768_76830

-- Definitions corresponding to conditions
def is_noun (word : String) : Prop :=
  -- definition of being a noun
  sorry

def corresponds_to_chinese_hint (word : String) (hint : String) : Prop :=
  -- definition of corresponding to a Chinese hint
  sorry

-- The theorem we want to prove
theorem fill_in_blank_with_warning : ∀ word : String, 
  (is_noun word ∧ corresponds_to_chinese_hint word "警告") → word = "warning" :=
by {
  sorry
}

end NUMINAMATH_GPT_fill_in_blank_with_warning_l768_76830


namespace NUMINAMATH_GPT_fraction_of_age_l768_76871

theorem fraction_of_age (jane_age_current : ℕ) (years_since_babysit : ℕ) (age_oldest_babysat_current : ℕ) :
  jane_age_current = 32 →
  years_since_babysit = 12 →
  age_oldest_babysat_current = 23 →
  ∃ (f : ℚ), f = 11 / 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_of_age_l768_76871


namespace NUMINAMATH_GPT_workers_together_complete_work_in_14_days_l768_76892

noncomputable def efficiency (Wq : ℝ) := 1.4 * Wq

def work_done_in_one_day_p (Wp : ℝ) := Wp = 1 / 24

noncomputable def work_done_in_one_day_q (Wq : ℝ) := Wq = (1 / 24) / 1.4

noncomputable def combined_work_per_day (Wp Wq : ℝ) := Wp + Wq

noncomputable def days_to_complete_work (W : ℝ) := 1 / W

theorem workers_together_complete_work_in_14_days (Wp Wq : ℝ) 
  (h1 : Wp = efficiency Wq)
  (h2 : work_done_in_one_day_p Wp)
  (h3 : work_done_in_one_day_q Wq) :
  days_to_complete_work (combined_work_per_day Wp Wq) = 14 := 
sorry

end NUMINAMATH_GPT_workers_together_complete_work_in_14_days_l768_76892


namespace NUMINAMATH_GPT_decreasing_sufficient_condition_l768_76802

theorem decreasing_sufficient_condition {a : ℝ} (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (a-2)*x^3 > (a-2)*y^3) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_sufficient_condition_l768_76802


namespace NUMINAMATH_GPT_min_value_of_z_l768_76841

-- Define the conditions as separate hypotheses.
variable (x y : ℝ)

def condition1 : Prop := x - y + 1 ≥ 0
def condition2 : Prop := x + y - 1 ≥ 0
def condition3 : Prop := x ≤ 3

-- Define the objective function.
def z : ℝ := 2 * x - 3 * y

-- State the theorem to prove the minimum value of z given the conditions.
theorem min_value_of_z (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x) :
  ∃ x y, condition1 x y ∧ condition2 x y ∧ condition3 x ∧ z x y = -6 :=
sorry

end NUMINAMATH_GPT_min_value_of_z_l768_76841


namespace NUMINAMATH_GPT_lorelai_jellybeans_l768_76810

variable (Gigi Rory Luke Lane Lorelai : ℕ)
variable (h1 : Gigi = 15)
variable (h2 : Rory = Gigi + 30)
variable (h3 : Luke = 2 * Rory)
variable (h4 : Lane = Gigi + 10)
variable (h5 : Lorelai = 3 * (Gigi + Luke + Lane))

theorem lorelai_jellybeans : Lorelai = 390 := by
  sorry

end NUMINAMATH_GPT_lorelai_jellybeans_l768_76810


namespace NUMINAMATH_GPT_find_l_in_triangle_l768_76877

/-- In triangle XYZ, if XY = 5, YZ = 12, XZ = 13, and YM is the angle bisector from vertex Y with YM = l * sqrt 2, then l equals 60/17. -/
theorem find_l_in_triangle (XY YZ XZ : ℝ) (YM l : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (hYM : YM = l * Real.sqrt 2) : 
    l = 60 / 17 :=
sorry

end NUMINAMATH_GPT_find_l_in_triangle_l768_76877


namespace NUMINAMATH_GPT_triangle_angle_C_and_area_l768_76848

theorem triangle_angle_C_and_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * c * Real.cos B = 2 * a - b)
  (h2 : c = Real.sqrt 3)
  (h3 : b - a = 1) :
  (C = Real.pi / 3) ∧
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_C_and_area_l768_76848


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l768_76826

variable (a b x y : ℝ)

theorem simplify_expr1 : 6 * a + 7 * b^2 - 9 + 4 * a - b^2 + 6 = 10 * a + 6 * b^2 - 3 :=
by
  sorry

theorem simplify_expr2 : 5 * x - 2 * (4 * x + 5 * y) + 3 * (3 * x - 4 * y) = 6 * x - 22 * y :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l768_76826


namespace NUMINAMATH_GPT_ball_center_distance_traveled_l768_76891

theorem ball_center_distance_traveled (d : ℝ) (r1 r2 r3 r4 : ℝ) (R1 R2 R3 R4 : ℝ) :
  d = 6 → 
  R1 = 120 → 
  R2 = 50 → 
  R3 = 90 → 
  R4 = 70 → 
  r1 = R1 - 3 → 
  r2 = R2 + 3 → 
  r3 = R3 - 3 → 
  r4 = R4 + 3 → 
  (1/2) * 2 * π * r1 + (1/2) * 2 * π * r2 + (1/2) * 2 * π * r3 + (1/2) * 2 * π * r4 = 330 * π :=
by
  sorry

end NUMINAMATH_GPT_ball_center_distance_traveled_l768_76891


namespace NUMINAMATH_GPT_candle_lighting_time_l768_76807

theorem candle_lighting_time 
  (l : ℕ) -- initial length of the candles
  (t_diff : ℤ := 206) -- the time difference in minutes, correlating to 1:34 PM.
  : t_diff = 206 :=
by sorry

end NUMINAMATH_GPT_candle_lighting_time_l768_76807


namespace NUMINAMATH_GPT_white_tiles_in_square_l768_76894

theorem white_tiles_in_square (n S : ℕ) (hn : n * n = S) (black_tiles : ℕ) (hblack_tiles : black_tiles = 81) (diagonal_black_tiles : n = 9) :
  S - black_tiles = 72 :=
by
  sorry

end NUMINAMATH_GPT_white_tiles_in_square_l768_76894


namespace NUMINAMATH_GPT_number_of_complete_decks_l768_76854

theorem number_of_complete_decks (total_cards : ℕ) (additional_cards : ℕ) (cards_per_deck : ℕ) 
(h1 : total_cards = 319) (h2 : additional_cards = 7) (h3 : cards_per_deck = 52) : 
total_cards - additional_cards = (cards_per_deck * 6) :=
by
  sorry

end NUMINAMATH_GPT_number_of_complete_decks_l768_76854


namespace NUMINAMATH_GPT_find_A_l768_76883

theorem find_A (A B : ℝ) (h1 : B = 10 * A) (h2 : 211.5 = B - A) : A = 23.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_A_l768_76883


namespace NUMINAMATH_GPT_route_speeds_l768_76812

theorem route_speeds (x : ℝ) (hx : x > 0) :
  (25 / x) - (21 / (1.4 * x)) = (20 / 60) := by
  sorry

end NUMINAMATH_GPT_route_speeds_l768_76812


namespace NUMINAMATH_GPT_compute_expression_l768_76806

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l768_76806


namespace NUMINAMATH_GPT_must_be_odd_l768_76833

theorem must_be_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
sorry

end NUMINAMATH_GPT_must_be_odd_l768_76833


namespace NUMINAMATH_GPT_sum_of_positive_factors_36_l768_76819

def sum_of_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (fun m => n % m = 0) |>.sum

theorem sum_of_positive_factors_36 : sum_of_divisors 36 = 91 := by
  sorry

end NUMINAMATH_GPT_sum_of_positive_factors_36_l768_76819


namespace NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l768_76818

theorem trajectory_of_center_of_moving_circle
  (x y : ℝ)
  (C1 : (x + 4)^2 + y^2 = 2)
  (C2 : (x - 4)^2 + y^2 = 2) :
  ((x = 0) ∨ (x^2 / 2 - y^2 / 14 = 1)) :=
sorry

end NUMINAMATH_GPT_trajectory_of_center_of_moving_circle_l768_76818


namespace NUMINAMATH_GPT_range_a_ge_one_l768_76805

theorem range_a_ge_one (a : ℝ) (x : ℝ) 
  (p : Prop := |x + 1| > 2) 
  (q : Prop := x > a) 
  (suff_not_necess_cond : ¬p → ¬q) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_a_ge_one_l768_76805


namespace NUMINAMATH_GPT_proof_problem_l768_76862

-- Define the system of equations
def system_of_equations (x y a : ℝ) : Prop :=
  (3 * x + y = 2 + 3 * a) ∧ (x + 3 * y = 2 + a)

-- Define the condition x + y < 0
def condition (x y : ℝ) : Prop := x + y < 0

-- Prove that if the system of equations has a solution with x + y < 0, then a < -1 and |1 - a| + |a + 1 / 2| = 1 / 2 - 2 * a
theorem proof_problem (x y a : ℝ) (h1 : system_of_equations x y a) (h2 : condition x y) :
  a < -1 ∧ |1 - a| + |a + 1 / 2| = (1 / 2) - 2 * a := 
sorry

end NUMINAMATH_GPT_proof_problem_l768_76862


namespace NUMINAMATH_GPT_students_with_one_problem_l768_76825

theorem students_with_one_problem :
  ∃ (n_1 n_2 n_3 n_4 n_5 n_6 n_7 : ℕ) (k_1 k_2 k_3 k_4 k_5 k_6 k_7 : ℕ),
    (n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 = 39) ∧
    (n_1 * k_1 + n_2 * k_2 + n_3 * k_3 + n_4 * k_4 + n_5 * k_5 + n_6 * k_6 + n_7 * k_7 = 60) ∧
    (k_1 ≠ 0) ∧ (k_2 ≠ 0) ∧ (k_3 ≠ 0) ∧ (k_4 ≠ 0) ∧ (k_5 ≠ 0) ∧ (k_6 ≠ 0) ∧ (k_7 ≠ 0) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧ (k_1 ≠ k_5) ∧ (k_1 ≠ k_6) ∧ (k_1 ≠ k_7) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ (k_2 ≠ k_5) ∧ (k_2 ≠ k_6) ∧ (k_2 ≠ k_7) ∧
    (k_3 ≠ k_4) ∧ (k_3 ≠ k_5) ∧ (k_3 ≠ k_6) ∧ (k_3 ≠ k_7) ∧
    (k_4 ≠ k_5) ∧ (k_4 ≠ k_6) ∧ (k_4 ≠ k_7) ∧
    (k_5 ≠ k_6) ∧ (k_5 ≠ k_7) ∧
    (k_6 ≠ k_7) ∧
    (n_1 = 33) :=
sorry

end NUMINAMATH_GPT_students_with_one_problem_l768_76825


namespace NUMINAMATH_GPT_compare_mixed_decimal_l768_76895

def mixed_number_value : ℚ := -2 - 1 / 3  -- Representation of -2 1/3 as a rational number
def decimal_value : ℚ := -2.3             -- Representation of -2.3 as a rational number

theorem compare_mixed_decimal : mixed_number_value < decimal_value :=
sorry

end NUMINAMATH_GPT_compare_mixed_decimal_l768_76895


namespace NUMINAMATH_GPT_coefficient_of_x_is_nine_l768_76880

theorem coefficient_of_x_is_nine (x : ℝ) (c : ℝ) (h : x = 0.5) (eq : 2 * x^2 + c * x - 5 = 0) : c = 9 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x_is_nine_l768_76880


namespace NUMINAMATH_GPT_divide_oranges_into_pieces_l768_76853

-- Definitions for conditions
def oranges : Nat := 80
def friends : Nat := 200
def pieces_per_friend : Nat := 4

-- Theorem stating the problem and the answer
theorem divide_oranges_into_pieces :
    (oranges > 0) → (friends > 0) → (pieces_per_friend > 0) →
    ((friends * pieces_per_friend) / oranges = 10) :=
by
  intros
  sorry

end NUMINAMATH_GPT_divide_oranges_into_pieces_l768_76853


namespace NUMINAMATH_GPT_cyc_inequality_l768_76896

theorem cyc_inequality (x y z : ℝ) (hx : 0 < x ∧ x < 2) (hy : 0 < y ∧ y < 2) (hz : 0 < z ∧ z < 2) 
  (hxyz : x^2 + y^2 + z^2 = 3) : 
  3 / 2 < (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) ∧ 
  (1 + y^2) / (x + 2) + (1 + z^2) / (y + 2) + (1 + x^2) / (z + 2) < 3 := 
by
  sorry

end NUMINAMATH_GPT_cyc_inequality_l768_76896


namespace NUMINAMATH_GPT_roots_in_ap_difference_one_l768_76882

theorem roots_in_ap_difference_one :
  ∀ (r1 r2 r3 : ℝ), 
    64 * r1^3 - 144 * r1^2 + 92 * r1 - 15 = 0 ∧
    64 * r2^3 - 144 * r2^2 + 92 * r2 - 15 = 0 ∧
    64 * r3^3 - 144 * r3^2 + 92 * r3 - 15 = 0 ∧
    (r2 - r1 = r3 - r2) →
    max (max r1 r2) r3 - min (min r1 r2) r3 = 1 := 
by
  intros r1 r2 r3 h
  sorry

end NUMINAMATH_GPT_roots_in_ap_difference_one_l768_76882


namespace NUMINAMATH_GPT_number_of_valid_subsets_l768_76857

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def oddSet : Finset ℕ := {1, 3, 5, 7}
def evenSet : Finset ℕ := {2, 4, 6}

theorem number_of_valid_subsets : 
  (oddSet.powerset.card * (evenSet.powerset.card - 1) - oddSet.powerset.card) = 96 :=
by sorry

end NUMINAMATH_GPT_number_of_valid_subsets_l768_76857


namespace NUMINAMATH_GPT_max_n_leq_V_l768_76804

theorem max_n_leq_V (n : ℤ) (V : ℤ) (h1 : 102 * n^2 <= V) (h2 : ∀ k : ℤ, (102 * k^2 <= V) → k <= 8) : V >= 6528 :=
sorry

end NUMINAMATH_GPT_max_n_leq_V_l768_76804


namespace NUMINAMATH_GPT_probability_of_adjacent_rs_is_two_fifth_l768_76889

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def countArrangementsWithAdjacentRs : ℕ :=
factorial 4

noncomputable def countTotalArrangements : ℕ :=
factorial 5 / factorial 2

noncomputable def probabilityOfAdjacentRs : ℚ :=
(countArrangementsWithAdjacentRs : ℚ) / (countTotalArrangements : ℚ)

theorem probability_of_adjacent_rs_is_two_fifth :
  probabilityOfAdjacentRs = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_adjacent_rs_is_two_fifth_l768_76889


namespace NUMINAMATH_GPT_ellipse_properties_l768_76897

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x * x) / (a * a) + (y * y) / (b * b) = 1

theorem ellipse_properties (a b c k : ℝ) (h_ab : a > b) (h_b : b > 1) (h_c : 2 * c = 2) 
  (h_area : (2 * Real.sqrt 3 / 3)^2 = 4 / 3) (h_slope : k ≠ 0)
  (h_PD : |(c - 4 * k^2 / (3 + 4 * k^2))^2 + (-3 * k / (3 + 4 * k^2))^2| = 3 * Real.sqrt 2 / 7) :
  (ellipse_equation 1 0 a b ∧
   (a = 2 ∧ b = Real.sqrt 3) ∧
   k = 1 ∨ k = -1) :=
by
  -- Prove the standard equation of the ellipse C and the value of k
  sorry

end NUMINAMATH_GPT_ellipse_properties_l768_76897


namespace NUMINAMATH_GPT_total_milk_consumed_l768_76885

theorem total_milk_consumed (regular_milk : ℝ) (soy_milk : ℝ) (H1 : regular_milk = 0.5) (H2: soy_milk = 0.1) :
    regular_milk + soy_milk = 0.6 :=
  by
  sorry

end NUMINAMATH_GPT_total_milk_consumed_l768_76885


namespace NUMINAMATH_GPT_find_change_l768_76861

def initial_amount : ℝ := 1.80
def cost_of_candy_bar : ℝ := 0.45
def change : ℝ := 1.35

theorem find_change : initial_amount - cost_of_candy_bar = change :=
by sorry

end NUMINAMATH_GPT_find_change_l768_76861


namespace NUMINAMATH_GPT_pauline_spent_in_all_l768_76856

theorem pauline_spent_in_all
  (cost_taco_shells : ℝ := 5)
  (cost_bell_pepper : ℝ := 1.5)
  (num_bell_peppers : ℕ := 4)
  (cost_meat_per_pound : ℝ := 3)
  (num_pounds_meat : ℝ := 2) :
  (cost_taco_shells + num_bell_peppers * cost_bell_pepper + num_pounds_meat * cost_meat_per_pound = 17) :=
by
  sorry

end NUMINAMATH_GPT_pauline_spent_in_all_l768_76856


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_9_l768_76803

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_digit_divisible_by_9_l768_76803


namespace NUMINAMATH_GPT_subtraction_result_l768_76874

-- Define the condition as given: x - 46 = 15
def condition (x : ℤ) := x - 46 = 15

-- Define the theorem that gives us the equivalent mathematical statement we want to prove
theorem subtraction_result (x : ℤ) (h : condition x) : x - 29 = 32 :=
by
  -- Here we would include the proof steps, but as per instructions we will use 'sorry' to skip the proof
  sorry

end NUMINAMATH_GPT_subtraction_result_l768_76874


namespace NUMINAMATH_GPT_opposite_exprs_have_value_l768_76835

theorem opposite_exprs_have_value (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) → x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_opposite_exprs_have_value_l768_76835


namespace NUMINAMATH_GPT_tory_video_games_l768_76828

theorem tory_video_games (T J: ℕ) :
    (3 * J + 5 = 11) → (J = T / 3) → T = 6 :=
by
  sorry

end NUMINAMATH_GPT_tory_video_games_l768_76828


namespace NUMINAMATH_GPT_incorrect_transformation_when_c_zero_l768_76898

theorem incorrect_transformation_when_c_zero {a b c : ℝ} (h : a * c = b * c) (hc : c = 0) : a ≠ b :=
by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_when_c_zero_l768_76898


namespace NUMINAMATH_GPT_minimum_value_g_l768_76881

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - x - 2

def g (x : ℝ) : ℝ := (x + a)^2 - (x + a) - 2 + x

theorem minimum_value_g (a : ℝ) :
  (if 1 ≤ a then g a (-1) = a^2 - 3 * a - 1 else
   if -3 < a ∧ a < 1 then g a (-a) = -a - 2 else
   if a ≤ -3 then g a 3 = a^2 + 5 * a + 7 else false) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_g_l768_76881


namespace NUMINAMATH_GPT_range_for_a_l768_76886

def f (a : ℝ) (x : ℝ) := 2 * x^3 - a * x^2 + 1

def two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (1/2 ≤ x1 ∧ x1 ≤ 2) ∧ (1/2 ≤ x2 ∧ x2 ≤ 2) ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem range_for_a {a : ℝ} : (3/2 : ℝ) < a ∧ a ≤ (17/4 : ℝ) ↔ two_zeros_in_interval a :=
by sorry

end NUMINAMATH_GPT_range_for_a_l768_76886


namespace NUMINAMATH_GPT_prize_selection_count_l768_76845

theorem prize_selection_count :
  (Nat.choose 20 1) * (Nat.choose 19 2) * (Nat.choose 17 4) = 8145600 := 
by 
  sorry

end NUMINAMATH_GPT_prize_selection_count_l768_76845


namespace NUMINAMATH_GPT_find_sum_l768_76866

variable {a : ℕ → ℝ} {r : ℝ}

-- Conditions: a_n > 0 for all n
axiom pos : ∀ n : ℕ, a n > 0

-- Given equation: a_1 * a_5 + 2 * a_3 * a_5 + a_3 * a_7 = 25
axiom given_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

theorem find_sum : a 3 + a 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l768_76866


namespace NUMINAMATH_GPT_number_of_players_sold_eq_2_l768_76888

def initial_balance : ℕ := 100
def selling_price_per_player : ℕ := 10
def buying_cost_per_player : ℕ := 15
def number_of_players_bought : ℕ := 4
def final_balance : ℕ := 60

theorem number_of_players_sold_eq_2 :
  ∃ x : ℕ, (initial_balance + selling_price_per_player * x - buying_cost_per_player * number_of_players_bought = final_balance) ∧ (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_sold_eq_2_l768_76888


namespace NUMINAMATH_GPT_decrease_in_demand_l768_76832

theorem decrease_in_demand (init_price new_price demand : ℝ) (init_demand : ℕ) (price_increase : ℝ) (original_revenue new_demand : ℝ) :
  init_price = 20 ∧ init_demand = 500 ∧ price_increase = 5 ∧ demand = init_price + price_increase ∧ 
  original_revenue = init_price * init_demand ∧ new_demand ≤ init_demand ∧ 
  new_demand * demand ≥ original_revenue → 
  init_demand - new_demand = 100 :=
by 
  sorry

end NUMINAMATH_GPT_decrease_in_demand_l768_76832


namespace NUMINAMATH_GPT_solve_problem_l768_76846

noncomputable def problem_statement : Prop :=
  ∀ (T0 Ta T t1 T1 h t2 T2 : ℝ),
    T0 = 88 ∧ Ta = 24 ∧ T1 = 40 ∧ t1 = 20 ∧
    T1 - Ta = (T0 - Ta) * ((1/2)^(t1/h)) ∧
    T2 = 32 ∧ T2 - Ta = (T1 - Ta) * ((1/2)^(t2/h)) →
    t2 = 10

theorem solve_problem : problem_statement := sorry

end NUMINAMATH_GPT_solve_problem_l768_76846


namespace NUMINAMATH_GPT_maximum_value_of_reciprocals_l768_76867

theorem maximum_value_of_reciprocals (c b : ℝ) (h0 : 0 < b ∧ b < c)
  (e1 : ℝ) (e2 : ℝ)
  (h1 : e1 = c / (Real.sqrt (c^2 + (2 * b)^2)))
  (h2 : e2 = c / (Real.sqrt (c^2 - b^2)))
  (h3 : 1 / e1^2 + 4 / e2^2 = 5) :
  ∃ max_val, max_val = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_reciprocals_l768_76867


namespace NUMINAMATH_GPT_lila_stickers_correct_l768_76843

-- Defining the constants for number of stickers each has
def Kristoff_stickers : ℕ := 85
def Riku_stickers : ℕ := 25 * Kristoff_stickers
def Lila_stickers : ℕ := 2 * (Kristoff_stickers + Riku_stickers)

-- The theorem to prove
theorem lila_stickers_correct : Lila_stickers = 4420 := 
by {
  sorry
}

end NUMINAMATH_GPT_lila_stickers_correct_l768_76843


namespace NUMINAMATH_GPT_new_student_bmi_l768_76876

theorem new_student_bmi 
(average_weight_29 : ℚ)
(average_height_29 : ℚ)
(average_weight_30 : ℚ)
(average_height_30 : ℚ)
(new_student_height : ℚ)
(bmi : ℚ)
(h1 : average_weight_29 = 28)
(h2 : average_height_29 = 1.5)
(h3 : average_weight_30 = 27.5)
(h4 : average_height_30 = 1.5)
(h5 : new_student_height = 1.4)
: bmi = 6.63 := 
sorry

end NUMINAMATH_GPT_new_student_bmi_l768_76876


namespace NUMINAMATH_GPT_correct_proposition_l768_76864

theorem correct_proposition (a b : ℝ) (h : |a| < b) : a^2 < b^2 :=
sorry

end NUMINAMATH_GPT_correct_proposition_l768_76864


namespace NUMINAMATH_GPT_intersection_first_quadrant_l768_76887

theorem intersection_first_quadrant (a : ℝ) : 
  (∃ x y : ℝ, (ax + y = 4) ∧ (x - y = 2) ∧ (0 < x) ∧ (0 < y)) ↔ (-1 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_first_quadrant_l768_76887


namespace NUMINAMATH_GPT_classroom_desks_l768_76875

theorem classroom_desks (N y : ℕ) (h : 16 * y = 21 * N)
  (hN_le: N <= 30 * 16 / 21) (hMultiple: 3 * N % 4 = 0)
  (hy_le: y ≤ 30)
  : y = 21 := by
  sorry

end NUMINAMATH_GPT_classroom_desks_l768_76875


namespace NUMINAMATH_GPT_max_writers_at_conference_l768_76879

variables (T E W x : ℕ)

-- Defining the conditions
def conference_conditions (T E W x : ℕ) : Prop :=
  T = 90 ∧ E > 38 ∧ x ≤ 6 ∧ 2 * x + (W + E - x) = T ∧ W = T - E - x

-- Statement to prove the number of writers
theorem max_writers_at_conference : ∃ W, conference_conditions 90 39 W 1 :=
by
  sorry

end NUMINAMATH_GPT_max_writers_at_conference_l768_76879


namespace NUMINAMATH_GPT_least_distinct_values_l768_76858

theorem least_distinct_values (lst : List ℕ) (h_len : lst.length = 2023) (h_mode : ∃ m, (∀ n ≠ m, lst.count n < lst.count m) ∧ lst.count m = 13) : ∃ x, x = 169 :=
by
  sorry

end NUMINAMATH_GPT_least_distinct_values_l768_76858


namespace NUMINAMATH_GPT_sakshi_work_days_l768_76813

theorem sakshi_work_days (x : ℝ) (efficiency_tanya : ℝ) (days_tanya : ℝ) 
  (h_efficiency : efficiency_tanya = 1.25) 
  (h_days : days_tanya = 4)
  (h_relationship : x / efficiency_tanya = days_tanya) : 
  x = 5 :=
by 
  -- Lean proof would go here
  sorry

end NUMINAMATH_GPT_sakshi_work_days_l768_76813


namespace NUMINAMATH_GPT_average_weight_when_D_joins_is_53_l768_76893

noncomputable def new_average_weight (A B C D E : ℕ) : ℕ :=
  (73 + B + C + D) / 4

theorem average_weight_when_D_joins_is_53 :
  (A + B + C) / 3 = 50 →
  A = 73 →
  (B + C + D + E) / 4 = 51 →
  E = D + 3 →
  73 + B + C + D = 212 →
  new_average_weight A B C D E = 53 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_when_D_joins_is_53_l768_76893


namespace NUMINAMATH_GPT_probability_one_head_two_tails_l768_76849

-- Define an enumeration for Coin with two possible outcomes: heads and tails.
inductive Coin
| heads
| tails

-- Function to count the number of heads in a list of Coin.
def countHeads : List Coin → Nat
| [] => 0
| Coin.heads :: xs => 1 + countHeads xs
| Coin.tails :: xs => countHeads xs

-- Function to calculate the probability of a specific event given the total outcomes.
def probability (specific_events total_outcomes : Nat) : Rat :=
  (specific_events : Rat) / (total_outcomes : Rat)

-- The main theorem
theorem probability_one_head_two_tails : probability 3 8 = (3 / 8 : Rat) :=
sorry

end NUMINAMATH_GPT_probability_one_head_two_tails_l768_76849


namespace NUMINAMATH_GPT_complex_multiplication_l768_76811

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (2 + i) * (1 - 3 * i) = 5 - 5 * i := 
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l768_76811


namespace NUMINAMATH_GPT_ramola_rank_last_is_14_l768_76823

-- Define the total number of students
def total_students : ℕ := 26

-- Define Ramola's rank from the start
def ramola_rank_start : ℕ := 14

-- Define a function to calculate the rank from the last given the above conditions
def ramola_rank_from_last (total_students ramola_rank_start : ℕ) : ℕ :=
  total_students - ramola_rank_start + 1

-- Theorem stating that Ramola's rank from the last is 14th
theorem ramola_rank_last_is_14 :
  ramola_rank_from_last total_students ramola_rank_start = 14 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ramola_rank_last_is_14_l768_76823


namespace NUMINAMATH_GPT_kanul_total_amount_l768_76824

theorem kanul_total_amount (T : ℝ) (h1 : 35000 + 40000 + 0.2 * T = T) : T = 93750 := 
by
  sorry

end NUMINAMATH_GPT_kanul_total_amount_l768_76824


namespace NUMINAMATH_GPT_total_cost_of_puzzles_l768_76855

-- Definitions for the costs of large and small puzzles
def large_puzzle_cost : ℕ := 15
def small_puzzle_cost : ℕ := 23 - large_puzzle_cost

-- Theorem statement
theorem total_cost_of_puzzles :
  (large_puzzle_cost + 3 * small_puzzle_cost) = 39 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_cost_of_puzzles_l768_76855


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l768_76801

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l768_76801


namespace NUMINAMATH_GPT_range_of_a_l768_76808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1 / 2) * Real.log x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 + 1) / (2 * x)

def p (a : ℝ) : Prop := ∀ x, 1 ≤ x → f_prime (a) (x) ≤ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1

theorem range_of_a (a : ℝ) : (p a ∧ q a) → -1 < a ∧ a ≤ -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l768_76808


namespace NUMINAMATH_GPT_number_of_chocolates_l768_76820

-- Define the dimensions of the box
def W_box := 30
def L_box := 20
def H_box := 5

-- Define the dimensions of one chocolate
def W_chocolate := 6
def L_chocolate := 4
def H_chocolate := 1

-- Calculate the volume of the box
def V_box := W_box * L_box * H_box

-- Calculate the volume of one chocolate
def V_chocolate := W_chocolate * L_chocolate * H_chocolate

-- Lean theorem statement for the proof problem
theorem number_of_chocolates : V_box / V_chocolate = 125 := 
by
  sorry

end NUMINAMATH_GPT_number_of_chocolates_l768_76820


namespace NUMINAMATH_GPT_Kolya_can_form_triangles_l768_76884

theorem Kolya_can_form_triangles :
  ∃ (K1a K1b K1c K3a K3b K3c V1 V2 V3 : ℝ), 
  (K1a + K1b + K1c = 1) ∧
  (K3a + K3b + K3c = 1) ∧
  (V1 + V2 + V3 = 1) ∧
  (K1a = 0.5) ∧ (K1b = 0.25) ∧ (K1c = 0.25) ∧
  (K3a = 0.5) ∧ (K3b = 0.25) ∧ (K3c = 0.25) ∧
  (∀ (V1 V2 V3 : ℝ), V1 + V2 + V3 = 1 → 
  (
    (K1a + V1 > K3b ∧ K1a + K3b > V1 ∧ V1 + K3b > K1a) ∧ 
    (K1b + V2 > K3a ∧ K1b + K3a > V2 ∧ V2 + K3a > K1b) ∧ 
    (K1c + V3 > K3c ∧ K1c + K3c > V3 ∧ V3 + K3c > K1c)
  )) :=
sorry

end NUMINAMATH_GPT_Kolya_can_form_triangles_l768_76884


namespace NUMINAMATH_GPT_Luke_spent_money_l768_76838

theorem Luke_spent_money : ∀ (initial_money additional_money current_money x : ℕ),
  initial_money = 48 →
  additional_money = 21 →
  current_money = 58 →
  (initial_money + additional_money - current_money) = x →
  x = 11 :=
by
  intros initial_money additional_money current_money x h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_Luke_spent_money_l768_76838


namespace NUMINAMATH_GPT_intersection_points_count_l768_76869

noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 2)
noncomputable def f2 (x : ℝ) : ℝ := -abs (2 * x + 5)

theorem intersection_points_count : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ 
    (∀ x : ℝ, f1 x = f2 x → x = x1 ∨ x = x2)) :=
sorry

end NUMINAMATH_GPT_intersection_points_count_l768_76869


namespace NUMINAMATH_GPT_sample_size_correct_l768_76839

def population_size : Nat := 8000
def sampled_students : List Nat := List.replicate 400 1 -- We use 1 as a placeholder for the heights

theorem sample_size_correct : sampled_students.length = 400 := by
  sorry

end NUMINAMATH_GPT_sample_size_correct_l768_76839


namespace NUMINAMATH_GPT_product_of_five_consecutive_is_divisible_by_sixty_l768_76829

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end NUMINAMATH_GPT_product_of_five_consecutive_is_divisible_by_sixty_l768_76829


namespace NUMINAMATH_GPT_cos_alpha_beta_l768_76865

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x * (sin x) ^ 2 - (1 / 2)

theorem cos_alpha_beta :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π / 2) →
    (0 < β ∧ β < π / 2) →
    f (α / 2) = sqrt 5 / 5 →
    f (β / 2) = 3 * sqrt 10 / 10 →
    cos (α - β) = sqrt 2 / 2 :=
by
  intros α β hα hβ h1 h2
  sorry

end NUMINAMATH_GPT_cos_alpha_beta_l768_76865


namespace NUMINAMATH_GPT_average_branches_per_foot_l768_76816

theorem average_branches_per_foot :
  let b1 := 200
  let h1 := 50
  let b2 := 180
  let h2 := 40
  let b3 := 180
  let h3 := 60
  let b4 := 153
  let h4 := 34
  (b1 / h1 + b2 / h2 + b3 / h3 + b4 / h4) / 4 = 4 := by
  sorry

end NUMINAMATH_GPT_average_branches_per_foot_l768_76816


namespace NUMINAMATH_GPT_cans_for_credit_l768_76860

theorem cans_for_credit (P C R : ℕ) : 
  (3 * P = 2 * C) → (C ≠ 0) → (R ≠ 0) → P * R / C = (P * R / C : ℕ) :=
by
  intros h1 h2 h3
  -- proof required here
  sorry

end NUMINAMATH_GPT_cans_for_credit_l768_76860


namespace NUMINAMATH_GPT_smallest_N_exists_l768_76834

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_N_exists_l768_76834


namespace NUMINAMATH_GPT_expected_socks_pairs_l768_76868

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end NUMINAMATH_GPT_expected_socks_pairs_l768_76868


namespace NUMINAMATH_GPT_diff_of_squares_525_475_l768_76831

theorem diff_of_squares_525_475 : 525^2 - 475^2 = 50000 := by
  sorry

end NUMINAMATH_GPT_diff_of_squares_525_475_l768_76831


namespace NUMINAMATH_GPT_arithmetic_mean_of_distribution_l768_76878

-- Defining conditions
def stddev : ℝ := 2.3
def value : ℝ := 11.6

-- Proving the mean (μ) is 16.2
theorem arithmetic_mean_of_distribution : ∃ μ : ℝ, μ = 16.2 ∧ value = μ - 2 * stddev :=
by
  use 16.2
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_distribution_l768_76878


namespace NUMINAMATH_GPT_ratio_odd_even_divisors_l768_76840

def sum_of_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of divisors

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of odd divisors

def sum_of_even_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of even divisors

theorem ratio_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 210) :
  sum_of_odd_divisors M / sum_of_even_divisors M = 1 / 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_odd_even_divisors_l768_76840


namespace NUMINAMATH_GPT_product_of_x_and_y_l768_76814

theorem product_of_x_and_y :
  ∀ (x y : ℝ), (∀ p : ℝ × ℝ, (p = (x, 6) ∨ p = (10, y)) → p.2 = (1 / 2) * p.1) → x * y = 60 :=
by
  intros x y h
  have hx : 6 = (1 / 2) * x := by exact h (x, 6) (Or.inl rfl)
  have hy : y = (1 / 2) * 10 := by exact h (10, y) (Or.inr rfl)
  sorry

end NUMINAMATH_GPT_product_of_x_and_y_l768_76814


namespace NUMINAMATH_GPT_cos_2theta_plus_sin_2theta_l768_76852

theorem cos_2theta_plus_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) : 
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_2theta_plus_sin_2theta_l768_76852


namespace NUMINAMATH_GPT_solution_when_a_is_1_solution_for_arbitrary_a_l768_76863

-- Let's define the inequality and the solution sets
def inequality (a x : ℝ) : Prop :=
  ((a + 1) * x - 3) / (x - 1) < 1

def solutionSet_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2

def solutionSet_a_eq_0 (x : ℝ) : Prop :=
  1 < x
  
def solutionSet_a_lt_0 (a x : ℝ) : Prop :=
  x < (2 / a) ∨ 1 < x

def solutionSet_0_lt_a_lt_2 (a x : ℝ) : Prop :=
  1 < x ∧ x < (2 / a)

def solutionSet_a_eq_2 : Prop :=
  false

def solutionSet_a_gt_2 (a x : ℝ) : Prop :=
  (2 / a) < x ∧ x < 1

-- Prove the solution for a = 1
theorem solution_when_a_is_1 : ∀ (x : ℝ), inequality 1 x ↔ solutionSet_a_eq_1 x :=
by sorry

-- Prove the solution for arbitrary real number a
theorem solution_for_arbitrary_a : ∀ (a x : ℝ),
  (a < 0 → inequality a x ↔ solutionSet_a_lt_0 a x) ∧
  (a = 0 → inequality a x ↔ solutionSet_a_eq_0 x) ∧
  (0 < a ∧ a < 2 → inequality a x ↔ solutionSet_0_lt_a_lt_2 a x) ∧
  (a = 2 → inequality a x → solutionSet_a_eq_2) ∧
  (a > 2 → inequality a x ↔ solutionSet_a_gt_2 a x) :=
by sorry

end NUMINAMATH_GPT_solution_when_a_is_1_solution_for_arbitrary_a_l768_76863


namespace NUMINAMATH_GPT_circle_intersection_line_l768_76872

theorem circle_intersection_line (d : ℝ) :
  (∃ (x y : ℝ), (x - 5)^2 + (y + 2)^2 = 49 ∧ (x + 1)^2 + (y - 5)^2 = 25 ∧ x + y = d) ↔ d = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_circle_intersection_line_l768_76872


namespace NUMINAMATH_GPT_find_m_l768_76873

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) (h1 : ∀ n, S n = n^2)
  (h2 : S m = (a m + a (m + 1)) / 2)
  (h3 : ∀ n > 1, a n = S n - S (n - 1))
  (h4 : a 1 = 1) :
  m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_l768_76873


namespace NUMINAMATH_GPT_a2b_etc_ge_9a2b2c2_l768_76822

theorem a2b_etc_ge_9a2b2c2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 :=
by
  sorry

end NUMINAMATH_GPT_a2b_etc_ge_9a2b2c2_l768_76822
