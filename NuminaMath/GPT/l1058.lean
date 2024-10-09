import Mathlib

namespace symmetrical_point_correct_l1058_105884

variables (x₁ y₁ : ℝ)

def symmetrical_point_x_axis (x y : ℝ) : ℝ × ℝ :=
(x, -y)

theorem symmetrical_point_correct : symmetrical_point_x_axis 3 2 = (3, -2) :=
by
  -- This is where we would provide the proof
  sorry

end symmetrical_point_correct_l1058_105884


namespace investor_amount_after_two_years_l1058_105815

noncomputable def compound_interest
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem investor_amount_after_two_years :
  compound_interest 3000 0.10 1 2 = 3630 :=
by
  -- Calculation goes here
  sorry

end investor_amount_after_two_years_l1058_105815


namespace total_cost_pencils_and_pens_l1058_105852

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l1058_105852


namespace inverse_matrix_l1058_105822

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, 7], ![-1, -1]]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-(1/3 : ℚ), -(7/3 : ℚ)], ![1/3, 4/3]]

theorem inverse_matrix : A.det ≠ 0 → A⁻¹ = A_inv := by
  sorry

end inverse_matrix_l1058_105822


namespace find_number_l1058_105876

theorem find_number (x : ℝ) (h : x - (3/5) * x = 60) : x = 150 :=
by
  sorry

end find_number_l1058_105876


namespace green_peaches_eq_three_l1058_105873

theorem green_peaches_eq_three (p r g : ℕ) (h1 : p = r + g) (h2 : r + 2 * g = p + 3) : g = 3 := 
by 
  sorry

end green_peaches_eq_three_l1058_105873


namespace projectile_height_reaches_45_at_t_0_5_l1058_105875

noncomputable def quadratic (a b c : ℝ) : ℝ → ℝ :=
  λ t => a * t^2 + b * t + c

theorem projectile_height_reaches_45_at_t_0_5 :
  ∃ t : ℝ, quadratic (-16) 98.5 (-45) t = 45 ∧ 0 ≤ t ∧ t = 0.5 :=
by
  sorry

end projectile_height_reaches_45_at_t_0_5_l1058_105875


namespace sequence_term_is_square_l1058_105880

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end sequence_term_is_square_l1058_105880


namespace speed_ratio_l1058_105820

theorem speed_ratio (v_A v_B : ℝ) (h : 71 / v_B = 142 / v_A) : v_A / v_B = 2 :=
by
  sorry

end speed_ratio_l1058_105820


namespace no_descending_digits_multiple_of_111_l1058_105836

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l1058_105836


namespace surface_area_of_z_eq_xy_over_a_l1058_105864

noncomputable def surface_area (a : ℝ) (h : a > 0) : ℝ :=
  (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1)

theorem surface_area_of_z_eq_xy_over_a (a : ℝ) (h : a > 0) :
  surface_area a h = (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1) := 
sorry

end surface_area_of_z_eq_xy_over_a_l1058_105864


namespace half_angle_quadrant_l1058_105844

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 / 2 * Real.pi)
  : (k % 2 = 0 → k * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < k * Real.pi + 3 / 4 * Real.pi) ∨
    (k % 2 = 1 → (k + 1) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k + 1) * Real.pi + 3 / 4 * Real.pi) :=
by
  sorry

end half_angle_quadrant_l1058_105844


namespace interval_first_bell_l1058_105805

theorem interval_first_bell (x : ℕ) : (Nat.lcm (Nat.lcm (Nat.lcm x 10) 14) 18 = 630) → x = 1 := by
  sorry

end interval_first_bell_l1058_105805


namespace power_of_xy_l1058_105898

-- Problem statement: Given a condition on x and y, find x^y.
theorem power_of_xy (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 13 = 0) : x^y = -8 :=
by {
  -- Proof will be added here
  sorry
}

end power_of_xy_l1058_105898


namespace imo_42nd_inequality_l1058_105837

theorem imo_42nd_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 := by
  sorry

end imo_42nd_inequality_l1058_105837


namespace pies_sold_in_week_l1058_105891

def daily_pies : ℕ := 8
def days_in_week : ℕ := 7

theorem pies_sold_in_week : daily_pies * days_in_week = 56 := by
  sorry

end pies_sold_in_week_l1058_105891


namespace cos_270_eq_zero_l1058_105853

-- Defining the cosine value for the given angle
theorem cos_270_eq_zero : Real.cos (270 * Real.pi / 180) = 0 :=
by
  sorry

end cos_270_eq_zero_l1058_105853


namespace decrease_is_75_86_percent_l1058_105842

noncomputable def decrease_percent (x y z : ℝ) : ℝ :=
  let x' := 0.8 * x
  let y' := 0.75 * y
  let z' := 0.9 * z
  let original_value := x^2 * y^3 * z
  let new_value := (x')^2 * (y')^3 * z'
  let decrease_value := original_value - new_value
  decrease_value / original_value

theorem decrease_is_75_86_percent (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) :
  decrease_percent x y z = 0.7586 :=
sorry

end decrease_is_75_86_percent_l1058_105842


namespace quiz_true_false_questions_l1058_105847

theorem quiz_true_false_questions (n : ℕ) 
  (h1 : 2^n - 2 ≠ 0) 
  (h2 : (2^n - 2) * 16 = 224) : 
  n = 4 := 
sorry

end quiz_true_false_questions_l1058_105847


namespace duration_of_loan_l1058_105834

namespace SimpleInterest

variables (P SI R : ℝ) (T : ℝ)

-- Defining the conditions
def principal := P = 1500
def simple_interest := SI = 735
def rate := R = 7 / 100

-- The question: Prove the duration (T) of the loan
theorem duration_of_loan (hP : principal P) (hSI : simple_interest SI) (hR : rate R) :
  T = 7 :=
sorry

end SimpleInterest

end duration_of_loan_l1058_105834


namespace function_domain_exclusion_l1058_105857

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l1058_105857


namespace hyperbola_asymptotes_l1058_105885

theorem hyperbola_asymptotes : 
  (∀ x y : ℝ, x^2 / 16 - y^2 / 9 = -1 → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end hyperbola_asymptotes_l1058_105885


namespace friend_owns_10_bicycles_l1058_105840

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l1058_105840


namespace total_typing_cost_l1058_105883

def typingCost (totalPages revisedOncePages revisedTwicePages : ℕ) (firstTimeCost revisionCost : ℕ) : ℕ := 
  let initialCost := totalPages * firstTimeCost
  let firstRevisionCost := revisedOncePages * revisionCost
  let secondRevisionCost := revisedTwicePages * (revisionCost * 2)
  initialCost + firstRevisionCost + secondRevisionCost

theorem total_typing_cost : typingCost 200 80 20 5 3 = 1360 := 
  by 
    rfl

end total_typing_cost_l1058_105883


namespace card_probability_ratio_l1058_105823

theorem card_probability_ratio :
  let total_cards := 40
  let numbers := 10
  let cards_per_number := 4
  let choose (n k : ℕ) := Nat.choose n k
  let p := 10 / choose total_cards 4
  let q := 1440 / choose total_cards 4
  (q / p) = 144 :=
by
  sorry

end card_probability_ratio_l1058_105823


namespace cyclic_sequence_u_16_eq_a_l1058_105813

-- Sequence definition and recurrence relation
def cyclic_sequence (u : ℕ → ℝ) (a : ℝ) : Prop :=
  u 1 = a ∧ ∀ n : ℕ, u (n + 1) = -1 / (u n + 1)

-- Proof that u_{16} = a under given conditions
theorem cyclic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : ∃ (u : ℕ → ℝ), cyclic_sequence u a ∧ u 16 = a :=
by
  sorry

end cyclic_sequence_u_16_eq_a_l1058_105813


namespace find_k_l1058_105855

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 3 = 2 * (n + k) + 5) : k = 3 / 2 := 
by 
  sorry

end find_k_l1058_105855


namespace probability_no_absolute_winner_l1058_105835

def no_absolute_winner_prob (P_AB : ℝ) (P_BV : ℝ) (P_VA : ℝ) : ℝ :=
  0.24 * P_VA + 0.36 * (1 - P_VA)

theorem probability_no_absolute_winner :
  (∀ P_VA : ℝ, P_VA >= 0 ∧ P_VA <= 1 → no_absolute_winner_prob 0.6 0.4 P_VA == 0.24) :=
sorry

end probability_no_absolute_winner_l1058_105835


namespace total_fish_caught_l1058_105863

theorem total_fish_caught (C_trips : ℕ) (B_fish_per_trip : ℕ) (C_fish_per_trip : ℕ) (D_fish_per_trip : ℕ) (B_trips D_trips : ℕ) :
  C_trips = 10 →
  B_trips = 2 * C_trips →
  B_fish_per_trip = 400 →
  C_fish_per_trip = B_fish_per_trip * (1 + 2/5) →
  D_trips = 3 * C_trips →
  D_fish_per_trip = C_fish_per_trip * (1 + 50/100) →
  B_trips * B_fish_per_trip + C_trips * C_fish_per_trip + D_trips * D_fish_per_trip = 38800 := 
by
  sorry

end total_fish_caught_l1058_105863


namespace find_pairs_l1058_105826

/-
Define the conditions:
1. The number of three-digit phone numbers consisting of only odd digits.
2. The number of three-digit phone numbers consisting of only even digits excluding 0.
3. Revenue difference is given by a specific equation.
4. \(X\) and \(Y\) are integers less than 250.
-/
def N₁ : ℕ := 5 * 5 * 5  -- Number of combinations with odd digits (1, 3, 5, 7, 9)
def N₂ : ℕ := 4 * 4 * 4  -- Number of combinations with even digits (2, 4, 6, 8)

-- Main theorem: finding pairs (X, Y) that satisfy the given conditions.
theorem find_pairs (X Y : ℕ) (hX : X < 250) (hY : Y < 250) :
  N₁ * X - N₂ * Y = 5 ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) := 
by {
  sorry
}

end find_pairs_l1058_105826


namespace find_f_neg1_l1058_105825

theorem find_f_neg1 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 := 
by 
  -- skipping the proof: 
  sorry

end find_f_neg1_l1058_105825


namespace undefined_integer_count_l1058_105803

noncomputable def expression (x : ℤ) : ℚ := (x^2 - 16) / ((x^2 - x - 6) * (x - 4))

theorem undefined_integer_count : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x^2 - x - 6) * (x - 4) = 0) ∧ S.card = 3 :=
  sorry

end undefined_integer_count_l1058_105803


namespace Z_real_axis_Z_first_quadrant_Z_on_line_l1058_105828

-- Definitions based on the problem conditions
def Z_real (m : ℝ) : ℝ := m^2 + 5*m + 6
def Z_imag (m : ℝ) : ℝ := m^2 - 2*m - 15

-- Lean statement for the equivalent proof problem

theorem Z_real_axis (m : ℝ) :
  Z_imag m = 0 ↔ (m = -3 ∨ m = 5) := sorry

theorem Z_first_quadrant (m : ℝ) :
  (Z_real m > 0 ∧ Z_imag m > 0) ↔ (m > 5) := sorry

theorem Z_on_line (m : ℝ) :
  (Z_real m + Z_imag m + 5 = 0) ↔ (m = (-5 + Real.sqrt 41) / 2) := sorry

end Z_real_axis_Z_first_quadrant_Z_on_line_l1058_105828


namespace joan_video_game_spending_l1058_105894

theorem joan_video_game_spending:
  let basketball_game := 5.20
  let racing_game := 4.23
  basketball_game + racing_game = 9.43 := 
by
  sorry

end joan_video_game_spending_l1058_105894


namespace kimberly_bought_skittles_l1058_105865

-- Conditions
def initial_skittles : ℕ := 5
def total_skittles : ℕ := 12

-- Prove
theorem kimberly_bought_skittles : ∃ bought_skittles : ℕ, (total_skittles = initial_skittles + bought_skittles) ∧ bought_skittles = 7 :=
by
  sorry

end kimberly_bought_skittles_l1058_105865


namespace fruit_seller_apples_l1058_105800

theorem fruit_seller_apples (original_apples : ℝ) (sold_percent : ℝ) (remaining_apples : ℝ)
  (h1 : sold_percent = 0.40)
  (h2 : remaining_apples = 420)
  (h3 : original_apples * (1 - sold_percent) = remaining_apples) :
  original_apples = 700 :=
by
  sorry

end fruit_seller_apples_l1058_105800


namespace cakes_bought_l1058_105827

theorem cakes_bought (initial : ℕ) (left : ℕ) (bought : ℕ) :
  initial = 169 → left = 32 → bought = initial - left → bought = 137 :=
by
  intros h_initial h_left h_bought
  rw [h_initial, h_left] at h_bought
  exact h_bought

end cakes_bought_l1058_105827


namespace exists_x_quadratic_eq_zero_iff_le_one_l1058_105858

variable (a : ℝ)

theorem exists_x_quadratic_eq_zero_iff_le_one : (∃ x : ℝ, x^2 - 2 * x + a = 0) ↔ a ≤ 1 :=
sorry

end exists_x_quadratic_eq_zero_iff_le_one_l1058_105858


namespace sum_of_ages_3_years_ago_l1058_105839

noncomputable def siblings_age_3_years_ago (R D S J : ℕ) : Prop :=
  R = D + 6 ∧
  D = S + 8 ∧
  J = R - 5 ∧
  R + 8 = 2 * (S + 8) ∧
  J + 10 = (D + 10) / 2 + 4 ∧
  S + 24 + J = 60 →
  (R - 3) + (D - 3) + (S - 3) + (J - 3) = 43

theorem sum_of_ages_3_years_ago (R D S J : ℕ) :
  siblings_age_3_years_ago R D S J :=
by
  intros
  sorry

end sum_of_ages_3_years_ago_l1058_105839


namespace minions_mistake_score_l1058_105870

theorem minions_mistake_score :
  (minions_left_phone_on_untrusted_website ∧
   downloaded_file_from_untrusted_source ∧
   guidelines_by_cellular_operators ∧
   avoid_sharing_personal_info ∧
   unverified_files_may_be_harmful ∧
   double_extensions_signify_malicious_software) →
  score = 21 :=
by
  -- Here we would provide the proof steps which we skip with sorry
  sorry

end minions_mistake_score_l1058_105870


namespace Ivan_pays_1_point_5_times_more_l1058_105856

theorem Ivan_pays_1_point_5_times_more (x y : ℝ) (h : x = 2 * y) : 1.5 * (0.6 * x + 0.8 * y) = x + y :=
by
  sorry

end Ivan_pays_1_point_5_times_more_l1058_105856


namespace factor_expression_l1058_105888

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) :=
by
  sorry

end factor_expression_l1058_105888


namespace value_of_c7_l1058_105801

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem value_of_c7 : c 7 = 448 := by
  sorry

end value_of_c7_l1058_105801


namespace arithmetic_geometric_seq_l1058_105810

noncomputable def a (n : ℕ) : ℤ := 2 * n - 4 -- General form of the arithmetic sequence

def is_geometric_sequence (s : ℕ → ℤ) : Prop := 
  ∀ n : ℕ, (n > 1) → s (n+1) * s (n-1) = s n ^ 2

theorem arithmetic_geometric_seq:
  (∃ (d : ℤ) (a : ℕ → ℤ), a 5 = 6 ∧ 
  (∀ n, a n = 6 + (n - 5) * d) ∧ a (3) * a (11) = a (5) ^ 2 ∧
  (∀ k, 5 < k → is_geometric_sequence (fun n => a (k + n - 1)))) → 
  ∃ t : ℕ, ∀ n : ℕ, n <= 2015 → 
  (a n = 2 * n - 4 →  n = 7) := 
sorry

end arithmetic_geometric_seq_l1058_105810


namespace abs_lt_2_sufficient_not_necessary_l1058_105848

theorem abs_lt_2_sufficient_not_necessary (x : ℝ) :
  (|x| < 2 → x^2 - x - 6 < 0) ∧ ¬ (x^2 - x - 6 < 0 → |x| < 2) :=
by {
  sorry
}

end abs_lt_2_sufficient_not_necessary_l1058_105848


namespace remainder_is_correct_l1058_105818

def P (x : ℝ) : ℝ := x^6 + 2 * x^5 - 3 * x^4 + x^2 - 8
def D (x : ℝ) : ℝ := x^2 - 1

theorem remainder_is_correct : 
  ∃ q : ℝ → ℝ, ∀ x : ℝ, P x = D x * q x + (2.5 * x - 9.5) :=
by
  sorry

end remainder_is_correct_l1058_105818


namespace negation_of_existential_l1058_105846

def divisible_by (n x : ℤ) := ∃ k : ℤ, x = k * n
def odd (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

def P (x : ℤ) := divisible_by 7 x ∧ ¬ odd x

theorem negation_of_existential :
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, divisible_by 7 x → odd x :=
by
  sorry

end negation_of_existential_l1058_105846


namespace dante_final_coconuts_l1058_105882

theorem dante_final_coconuts
  (Paolo_coconuts : ℕ) (Dante_init_coconuts : ℝ)
  (Bianca_coconuts : ℕ) (Dante_final_coconuts : ℕ):
  Paolo_coconuts = 14 →
  Dante_init_coconuts = 1.5 * Real.sqrt Paolo_coconuts →
  Bianca_coconuts = 2 * (Paolo_coconuts + Int.floor Dante_init_coconuts) →
  Dante_final_coconuts = (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) - 
    (25 * (Int.floor (Dante_init_coconuts) - (Int.floor (Dante_init_coconuts) / 3)) / 100) →
  Dante_final_coconuts = 3 :=
by
  sorry

end dante_final_coconuts_l1058_105882


namespace length_of_scale_parts_l1058_105821

theorem length_of_scale_parts (total_length_ft : ℕ) (remaining_inches : ℕ) (parts : ℕ) : 
  total_length_ft = 6 ∧ remaining_inches = 8 ∧ parts = 2 →
  ∃ ft inches, ft = 3 ∧ inches = 4 :=
by
  sorry

end length_of_scale_parts_l1058_105821


namespace probability_at_least_two_tails_l1058_105809

def fair_coin_prob (n : ℕ) : ℚ :=
  (1 / 2 : ℚ)^n

def at_least_two_tails_in_next_three_flips : ℚ :=
  1 - (fair_coin_prob 3 + 3 * fair_coin_prob 3)

theorem probability_at_least_two_tails :
  at_least_two_tails_in_next_three_flips = 1 / 2 := 
by
  sorry

end probability_at_least_two_tails_l1058_105809


namespace find_magnitude_of_z_l1058_105831

open Complex

theorem find_magnitude_of_z
    (z : ℂ)
    (h : z^4 = 80 - 96 * I) : abs z = 5^(3/4) :=
by sorry

end find_magnitude_of_z_l1058_105831


namespace remainder_of_3_pow_244_mod_5_l1058_105879

theorem remainder_of_3_pow_244_mod_5 : (3^244) % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l1058_105879


namespace no_tangential_triangle_exists_l1058_105838

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C2
def C2 (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Additional condition that the point (1, 1) lies on C2
def point_on_C2 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (1^2) / (a^2) + (1^2) / (b^2) = 1

-- The theorem to prove
theorem no_tangential_triangle_exists (a b : ℝ) (h : a > b ∧ b > 0) :
  point_on_C2 a b h →
  ¬ ∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2) ∧ 
    (C2 a b A.1 A.2 h ∧ C2 a b B.1 B.2 h ∧ C2 a b C.1 C.2 h) :=
by sorry

end no_tangential_triangle_exists_l1058_105838


namespace length_of_DC_l1058_105899

noncomputable def AB : ℝ := 30
noncomputable def sine_A : ℝ := 4 / 5
noncomputable def sine_C : ℝ := 1 / 4
noncomputable def angle_ADB : ℝ := Real.pi / 2

theorem length_of_DC (h_AB : AB = 30) (h_sine_A : sine_A = 4 / 5) (h_sine_C : sine_C = 1 / 4) (h_angle_ADB : angle_ADB = Real.pi / 2) :
  ∃ DC : ℝ, DC = 24 * Real.sqrt 15 :=
by sorry

end length_of_DC_l1058_105899


namespace stratified_sampling_total_results_l1058_105841

theorem stratified_sampling_total_results :
  let junior_students := 400
  let senior_students := 200
  let total_students_to_sample := 60
  let junior_sample := 40
  let senior_sample := 20
  (Nat.choose junior_students junior_sample) * (Nat.choose senior_students senior_sample) = Nat.choose 400 40 * Nat.choose 200 20 :=
  sorry

end stratified_sampling_total_results_l1058_105841


namespace area_of_circle_r_is_16_percent_of_circle_s_l1058_105807

open Real

variables (Ds Dr Rs Rr As Ar : ℝ)

def circle_r_is_40_percent_of_circle_s (Ds Dr : ℝ) := Dr = 0.40 * Ds
def radius_of_circle (D : ℝ) (R : ℝ) := R = D / 2
def area_of_circle (R : ℝ) (A : ℝ) := A = π * R^2
def percentage_area (As Ar : ℝ) (P : ℝ) := P = (Ar / As) * 100

theorem area_of_circle_r_is_16_percent_of_circle_s :
  ∀ (Ds Dr Rs Rr As Ar : ℝ),
    circle_r_is_40_percent_of_circle_s Ds Dr →
    radius_of_circle Ds Rs →
    radius_of_circle Dr Rr →
    area_of_circle Rs As →
    area_of_circle Rr Ar →
    percentage_area As Ar 16 := by
  intros Ds Dr Rs Rr As Ar H1 H2 H3 H4 H5
  sorry

end area_of_circle_r_is_16_percent_of_circle_s_l1058_105807


namespace first_vessel_milk_water_l1058_105829

variable (V : ℝ)

def vessel_ratio (v1 v2 : ℝ) : Prop := 
  v1 / v2 = 3 / 5

def vessel1_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 1 / 2

def vessel2_milk_water_ratio (milk water : ℝ) : Prop :=
  milk / water = 3 / 2

def mix_ratio (milk1 water1 milk2 water2 : ℝ) : Prop :=
  (milk1 + milk2) / (water1 + water2) = 1

theorem first_vessel_milk_water (V : ℝ) (v1 v2 : ℝ) (m1 w1 m2 w2 : ℝ)
  (hv : vessel_ratio v1 v2)
  (hv1 : vessel1_milk_water_ratio m1 w1)
  (hv2 : vessel2_milk_water_ratio m2 w2)
  (hmix : mix_ratio m1 w1 m2 w2) :
  vessel1_milk_water_ratio m1 w1 :=
  sorry

end first_vessel_milk_water_l1058_105829


namespace books_in_final_category_l1058_105832

-- Define the number of initial books
def initial_books : ℕ := 400

-- Define the number of divisions
def num_divisions : ℕ := 4

-- Define the iterative division process
def final_books (initial : ℕ) (divisions : ℕ) : ℕ :=
  initial / (2 ^ divisions)

-- State the theorem
theorem books_in_final_category : final_books initial_books num_divisions = 25 := by
  sorry

end books_in_final_category_l1058_105832


namespace x_cubed_plus_y_cubed_l1058_105877

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := 
by 
  sorry

end x_cubed_plus_y_cubed_l1058_105877


namespace find_m_value_l1058_105861

theorem find_m_value (m : ℝ) (h₀ : m > 0) (h₁ : (4 - m) / (m - 2) = m) : m = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end find_m_value_l1058_105861


namespace fill_sink_time_l1058_105868

theorem fill_sink_time {R1 R2 R T: ℝ} (h1: R1 = 1 / 210) (h2: R2 = 1 / 214) (h3: R = R1 + R2) (h4: T = 1 / R):
  T = 105.75 :=
by 
  sorry

end fill_sink_time_l1058_105868


namespace sin_A_and_height_on_AB_l1058_105897

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l1058_105897


namespace sum_50_to_75_l1058_105890

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end sum_50_to_75_l1058_105890


namespace expectedValueProof_l1058_105895

-- Definition of the problem conditions
def veryNormalCoin {n : ℕ} : Prop :=
  ∀ t : ℕ, (5 < t → (t - 5) = n → (t+1 = t + 1)) ∧ (t ≤ 5 ∨ n = t)

-- Definition of the expected value calculation
def expectedValue (n : ℕ) : ℚ :=
  if n > 0 then (1/2)^n else 0

-- Expected value for the given problem
def expectedValueProblem : ℚ := 
  let a1 := -2/683
  let expectedFirstFlip := 1/2 - 1/(2 * 683)
  100 * 341 + 683

-- Main statement to prove
theorem expectedValueProof : expectedValueProblem = 34783 := 
  sorry -- Proof omitted

end expectedValueProof_l1058_105895


namespace calculate_decimal_l1058_105869

theorem calculate_decimal : 3.59 + 2.4 - 1.67 = 4.32 := 
  by
  sorry

end calculate_decimal_l1058_105869


namespace find_cost_price_l1058_105867

theorem find_cost_price (SP : ℤ) (profit_percent : ℚ) (CP : ℤ) (h1 : SP = CP + (profit_percent * CP)) (h2 : SP = 240) (h3 : profit_percent = 0.25) : CP = 192 :=
by
  sorry

end find_cost_price_l1058_105867


namespace intersection_M_N_l1058_105866

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l1058_105866


namespace volume_of_parallelepiped_l1058_105878

theorem volume_of_parallelepiped 
  (l w h : ℝ)
  (h1 : l * w / Real.sqrt (l^2 + w^2) = 2 * Real.sqrt 5)
  (h2 : h * w / Real.sqrt (h^2 + w^2) = 30 / Real.sqrt 13)
  (h3 : h * l / Real.sqrt (h^2 + l^2) = 15 / Real.sqrt 10) 
  : l * w * h = 750 :=
sorry

end volume_of_parallelepiped_l1058_105878


namespace smallest_s_triangle_l1058_105812

theorem smallest_s_triangle (s : ℕ) :
  (7 + s > 11) ∧ (7 + 11 > s) ∧ (11 + s > 7) → s = 5 :=
sorry

end smallest_s_triangle_l1058_105812


namespace range_of_a_l1058_105849

theorem range_of_a (a : ℝ) : (a < 0 → (∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) ∧ 
                              (∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0) ↔ (x < -4 ∨ x ≥ -2)) ∧ 
                              ((¬(∀ x : ℝ, 3 * a < x ∧ x < a → x^2 - 4 * a * x + 3 * a^2 < 0)) 
                                → (¬(∀ x : ℝ, (x^2 - x - 6 ≤ 0) ∨ (x^2 + 2 * x - 8 > 0))))
                            → (a ≤ -4 ∨ (a < 0 ∧ 3 * a >= -2)) :=
by
  intros h
  sorry

end range_of_a_l1058_105849


namespace tangent_addition_tangent_subtraction_l1058_105819

theorem tangent_addition (a b : ℝ) : 
  Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
sorry

theorem tangent_subtraction (a b : ℝ) : 
  Real.tan (a - b) = (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b) :=
sorry

end tangent_addition_tangent_subtraction_l1058_105819


namespace member_age_greater_than_zero_l1058_105808

def num_members : ℕ := 23
def avg_age : ℤ := 0
def age_range : Set ℤ := {x | x ≥ -20 ∧ x ≤ 20}
def num_negative_members : ℕ := 5

theorem member_age_greater_than_zero :
  ∃ n : ℕ, n ≤ 18 ∧ (avg_age = 0 ∧ num_members = 23 ∧ num_negative_members = 5 ∧ ∀ age ∈ age_range, age ≥ -20 ∧ age ≤ 20) :=
sorry

end member_age_greater_than_zero_l1058_105808


namespace toys_per_week_l1058_105817

-- Define the number of days the workers work in a week
def days_per_week : ℕ := 4

-- Define the number of toys produced each day
def toys_per_day : ℕ := 1140

-- State the proof problem: workers produce 4560 toys per week
theorem toys_per_week : (toys_per_day * days_per_week) = 4560 :=
by
  -- Proof goes here
  sorry

end toys_per_week_l1058_105817


namespace hash_op_is_100_l1058_105804

def hash_op (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_op_is_100 (a b : ℕ) (h1 : a + b = 5) : hash_op a b = 100 :=
sorry

end hash_op_is_100_l1058_105804


namespace opposite_of_one_third_l1058_105872

theorem opposite_of_one_third : -(1/3) = -1/3 := by
  sorry

end opposite_of_one_third_l1058_105872


namespace school_dance_boys_count_l1058_105881

theorem school_dance_boys_count :
  let total_attendees := 100
  let faculty_and_staff := total_attendees * 10 / 100
  let students := total_attendees - faculty_and_staff
  let girls := 2 * students / 3
  let boys := students - girls
  boys = 30 := by
  sorry

end school_dance_boys_count_l1058_105881


namespace problem_statement_l1058_105816

open Real

namespace MathProblem

def p₁ := ∃ x : ℝ, x^2 + x + 1 < 0
def p₂ := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_statement : (¬p₁) ∨ (¬p₂) :=
by
  sorry

end MathProblem

end problem_statement_l1058_105816


namespace merchant_loss_l1058_105887

theorem merchant_loss (n m : ℝ) (h₁ : n ≠ m) : 
  let x := n / m
  let y := m / n
  x + y > 2 := by
sorry

end merchant_loss_l1058_105887


namespace squares_below_16x_144y_1152_l1058_105811

noncomputable def count_squares_below_line (a b c : ℝ) (x_max y_max : ℝ) : ℝ :=
  let total_squares := x_max * y_max
  let line_slope := -a/b
  let squares_crossed_by_diagonal := x_max + y_max - 1
  (total_squares - squares_crossed_by_diagonal) / 2

theorem squares_below_16x_144y_1152 : 
  count_squares_below_line 16 144 1152 72 8 = 248.5 := 
by
  sorry

end squares_below_16x_144y_1152_l1058_105811


namespace part_I_part_II_l1058_105806

/-- (I) -/
theorem part_I (x : ℝ) (a : ℝ) (h_a : a = -1) :
  (|2 * x| + |x - 1| ≤ 4) → x ∈ Set.Icc (-1) (5 / 3) :=
by sorry

/-- (II) -/
theorem part_II (x : ℝ) (a : ℝ) (h_eq : |2 * x| + |x + a| = |x - a|) :
  (a > 0 → x ∈ Set.Icc (-a) 0) ∧ (a < 0 → x ∈ Set.Icc 0 (-a)) :=
by sorry

end part_I_part_II_l1058_105806


namespace concyclic_projections_of_concyclic_quad_l1058_105843

variables {A B C D A' B' C' D' : Type*}

def are_concyclic (p1 p2 p3 p4: Type*) : Prop :=
  sorry -- Assume we have a definition for concyclic property of points

def are_orthogonal_projection (x y : Type*) (l : Type*) : Type* :=
  sorry -- Assume we have a definition for orthogonal projection of a point on line

theorem concyclic_projections_of_concyclic_quad
  (hABCD : are_concyclic A B C D)
  (hA'_proj : are_orthogonal_projection A A' (BD))
  (hC'_proj : are_orthogonal_projection C C' (BD))
  (hB'_proj : are_orthogonal_projection B B' (AC))
  (hD'_proj : are_orthogonal_projection D D' (AC)) :
  are_concyclic A' B' C' D' :=
sorry

end concyclic_projections_of_concyclic_quad_l1058_105843


namespace toms_expense_l1058_105830

def cost_per_square_foot : ℝ := 5
def square_feet_per_seat : ℝ := 12
def number_of_seats : ℝ := 500
def partner_coverage : ℝ := 0.40

def total_square_feet : ℝ := square_feet_per_seat * number_of_seats
def land_cost : ℝ := cost_per_square_foot * total_square_feet
def construction_cost : ℝ := 2 * land_cost
def total_cost : ℝ := land_cost + construction_cost
def tom_coverage_percentage : ℝ := 1 - partner_coverage
def toms_share : ℝ := tom_coverage_percentage * total_cost

theorem toms_expense :
  toms_share = 54000 :=
by
  sorry

end toms_expense_l1058_105830


namespace inequality_for_average_daily_work_l1058_105886

-- Given
def total_earthwork : ℕ := 300
def completed_earthwork_first_day : ℕ := 60
def scheduled_days : ℕ := 6
def days_ahead : ℕ := 2

-- To Prove
theorem inequality_for_average_daily_work (x : ℕ) :
  scheduled_days - days_ahead - 1 > 0 →
  (total_earthwork - completed_earthwork_first_day) ≤ x * (scheduled_days - days_ahead - 1) :=
by
  sorry

end inequality_for_average_daily_work_l1058_105886


namespace N_cannot_be_sum_of_three_squares_l1058_105802

theorem N_cannot_be_sum_of_three_squares (K : ℕ) (L : ℕ) (N : ℕ) (h1 : N = 4^K * L) (h2 : L % 8 = 7) : ¬ ∃ (a b c : ℕ), N = a^2 + b^2 + c^2 := 
sorry

end N_cannot_be_sum_of_three_squares_l1058_105802


namespace train_speed_l1058_105814

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l1058_105814


namespace min_value_proof_l1058_105833

noncomputable def min_value (x y : ℝ) : ℝ :=
x^3 + y^3 - x^2 - y^2

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) : 
min_value x y ≥ 1 := by
  sorry

end min_value_proof_l1058_105833


namespace sufficient_but_not_necessary_l1058_105854

variable (x : ℝ)

def condition1 : Prop := x > 2
def condition2 : Prop := x^2 > 4

theorem sufficient_but_not_necessary :
  (condition1 x → condition2 x) ∧ (¬ (condition2 x → condition1 x)) :=
by 
  sorry

end sufficient_but_not_necessary_l1058_105854


namespace solveTheaterProblem_l1058_105893

open Nat

def theaterProblem : Prop :=
  ∃ (A C : ℕ), (A + C = 80) ∧ (12 * A + 5 * C = 519) ∧ (C = 63)

theorem solveTheaterProblem : theaterProblem :=
  by
  sorry

end solveTheaterProblem_l1058_105893


namespace problem1_problem2_l1058_105860

theorem problem1 : (1 : ℤ) - (2 : ℤ)^3 / 8 - ((1 / 4 : ℚ) * (-2)^2) = (-2 : ℤ) := by
  sorry

theorem problem2 : (-(1 / 12 : ℚ) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = (-21 : ℤ) := by
  sorry

end problem1_problem2_l1058_105860


namespace rectangle_area_l1058_105850

theorem rectangle_area
  (L B : ℕ)
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) : L * B = 2030 :=
sorry

end rectangle_area_l1058_105850


namespace derivative_at_zero_l1058_105851

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem with the given conditions and expected result
theorem derivative_at_zero :
  (deriv f 0 = -120) :=
by
  sorry

end derivative_at_zero_l1058_105851


namespace isosceles_triangle_aacute_l1058_105824

theorem isosceles_triangle_aacute (a b c : ℝ) (h1 : a = b) (h2 : a + b + c = 180) (h3 : c = 108)
  : ∃ x y z : ℝ, x + y + z = 180 ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by {
  sorry
}

end isosceles_triangle_aacute_l1058_105824


namespace sasha_added_num_l1058_105874

theorem sasha_added_num (a b c : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a / b = 5 * (a + c) / (b * c)) : c = 6 ∨ c = -20 := 
sorry

end sasha_added_num_l1058_105874


namespace passing_probability_l1058_105859

def probability_of_passing (p : ℝ) : ℝ :=
  p^3 + p^2 * (1 - p) + (1 - p) * p^2

theorem passing_probability :
  probability_of_passing 0.6 = 0.504 :=
by {
  sorry
}

end passing_probability_l1058_105859


namespace hypotenuse_is_2_l1058_105862

noncomputable def quadratic_trinomial_hypotenuse (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let xv := -b / (2 * a)
  let yv := a * xv^2 + b * xv + c
  if xv = (x1 + x2) / 2 then
    Real.sqrt 2 * abs (-b / a)
  else 0

theorem hypotenuse_is_2 {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  quadratic_trinomial_hypotenuse a b c = 2 := by
  sorry

end hypotenuse_is_2_l1058_105862


namespace multiples_of_4_between_88_and_104_l1058_105892

theorem multiples_of_4_between_88_and_104 : 
  ∃ n, (104 - 4 * 23 = n) ∧ n = 88 ∧ ( ∀ x, (x ≥ 88 ∧ x ≤ 104 ∧ x % 4 = 0) → ( x - 88) / 4 < 24) :=
by
  sorry

end multiples_of_4_between_88_and_104_l1058_105892


namespace speed_of_man_rowing_upstream_l1058_105871

theorem speed_of_man_rowing_upstream (Vm Vdownstream Vupstream : ℝ) (hVm : Vm = 40) (hVdownstream : Vdownstream = 45) : Vupstream = 35 :=
by
  sorry

end speed_of_man_rowing_upstream_l1058_105871


namespace find_y_value_l1058_105896

theorem find_y_value (x y : ℝ) (k : ℝ) 
  (h1 : 5 * y = k / x^2)
  (h2 : y = 4)
  (h3 : x = 2)
  (h4 : k = 80) :
  ( ∃ y : ℝ, 5 * y = k / 4^2 ∧ y = 1) :=
by
  sorry

end find_y_value_l1058_105896


namespace bob_age_is_725_l1058_105889

theorem bob_age_is_725 (n : ℕ) (h1 : ∃ k : ℤ, n - 3 = k^2) (h2 : ∃ j : ℤ, n + 4 = j^3) : n = 725 :=
sorry

end bob_age_is_725_l1058_105889


namespace rows_of_roses_l1058_105845

variable (rows total_roses_per_row roses_per_row_red roses_per_row_non_red roses_per_row_white roses_per_row_pink total_pink_roses : ℕ)
variable (half_two_fifth three_fifth : ℚ)

-- Assume the conditions
axiom h1 : total_roses_per_row = 20
axiom h2 : roses_per_row_red = total_roses_per_row / 2
axiom h3 : roses_per_row_non_red = total_roses_per_row - roses_per_row_red
axiom h4 : roses_per_row_white = (3 / 5 : ℚ) * roses_per_row_non_red
axiom h5 : roses_per_row_pink = (2 / 5 : ℚ) * roses_per_row_non_red
axiom h6 : total_pink_roses = 40

-- Prove the number of rows in the garden
theorem rows_of_roses : rows = total_pink_roses / (roses_per_row_pink) :=
by
  sorry

end rows_of_roses_l1058_105845
