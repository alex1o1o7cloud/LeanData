import Mathlib

namespace fraction_meaningful_l1465_146570

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l1465_146570


namespace roger_trays_l1465_146505

theorem roger_trays (trays_per_trip trips trays_first_table : ℕ) 
  (h1 : trays_per_trip = 4) 
  (h2 : trips = 3) 
  (h3 : trays_first_table = 10) : 
  trays_per_trip * trips - trays_first_table = 2 :=
by
  -- Step proofs are omitted
  sorry

end roger_trays_l1465_146505


namespace halfway_between_fractions_l1465_146531

theorem halfway_between_fractions : ( (1/8 : ℚ) + (1/3 : ℚ) ) / 2 = 11 / 48 :=
by
  sorry

end halfway_between_fractions_l1465_146531


namespace polygon_sides_arithmetic_progression_l1465_146563

theorem polygon_sides_arithmetic_progression
  (n : ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 172 - (i - 1) * 8 > 0) -- Each angle in the sequence is positive
  (h2 : (∀ i, 1 ≤ i → i ≤ n → (172 - (i - 1) * 8) < 180)) -- Each angle < 180 degrees
  (h3 : n * (172 - (n-1) * 4) = 180 * (n - 2)) -- Sum of interior angles formula
  : n = 10 :=
sorry

end polygon_sides_arithmetic_progression_l1465_146563


namespace iced_tea_cost_is_correct_l1465_146558

noncomputable def iced_tea_cost (cost_cappuccino cost_latte cost_espresso : ℝ) (num_cappuccino num_iced_tea num_latte num_espresso : ℕ) (bill_amount change_amount : ℝ) : ℝ :=
  let total_cappuccino_cost := cost_cappuccino * num_cappuccino
  let total_latte_cost := cost_latte * num_latte
  let total_espresso_cost := cost_espresso * num_espresso
  let total_spent := bill_amount - change_amount
  let total_other_cost := total_cappuccino_cost + total_latte_cost + total_espresso_cost
  let total_iced_tea_cost := total_spent - total_other_cost
  total_iced_tea_cost / num_iced_tea

theorem iced_tea_cost_is_correct:
  iced_tea_cost 2 1.5 1 3 2 2 2 20 3 = 3 :=
by
  sorry

end iced_tea_cost_is_correct_l1465_146558


namespace sandy_found_additional_money_l1465_146507

-- Define the initial amount of money Sandy had
def initial_amount : ℝ := 13.99

-- Define the cost of the shirt
def shirt_cost : ℝ := 12.14

-- Define the cost of the jacket
def jacket_cost : ℝ := 9.28

-- Define the remaining amount after buying the shirt
def remaining_after_shirt : ℝ := initial_amount - shirt_cost

-- Define the additional money found in Sandy's pocket
def additional_found_money : ℝ := jacket_cost - remaining_after_shirt

-- State the theorem to prove the amount of additional money found
theorem sandy_found_additional_money :
  additional_found_money = 11.13 :=
by sorry

end sandy_found_additional_money_l1465_146507


namespace max_gcd_value_l1465_146510

theorem max_gcd_value (n : ℕ) (hn : 0 < n) : ∃ k, k = gcd (13 * n + 4) (8 * n + 3) ∧ k <= 7 := sorry

end max_gcd_value_l1465_146510


namespace annie_purchases_l1465_146544

theorem annie_purchases (x y z : ℕ) 
  (h1 : x + y + z = 50) 
  (h2 : 20 * x + 400 * y + 500 * z = 5000) :
  x = 40 :=
by sorry

end annie_purchases_l1465_146544


namespace smallest_part_division_l1465_146539

theorem smallest_part_division (y : ℝ) (h1 : y > 0) :
  ∃ (x : ℝ), x = y / 9 ∧ (∃ (a b c : ℝ), a = x ∧ b = 3 * x ∧ c = 5 * x ∧ a + b + c = y) :=
sorry

end smallest_part_division_l1465_146539


namespace factor_expression_l1465_146586

theorem factor_expression (x y : ℝ) :
  75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) :=
by
  sorry

end factor_expression_l1465_146586


namespace percentage_selected_in_state_A_l1465_146535

-- Definitions
def num_candidates : ℕ := 8000
def percentage_selected_state_B : ℕ := 7
def extra_selected_candidates : ℕ := 80

-- Question
theorem percentage_selected_in_state_A :
  ∃ (P : ℕ), ((P / 100) * 8000 + 80 = 560) ∧ (P = 6) := sorry

end percentage_selected_in_state_A_l1465_146535


namespace john_total_distance_l1465_146591

theorem john_total_distance :
  let speed1 := 35
  let time1 := 2
  let distance1 := speed1 * time1

  let speed2 := 55
  let time2 := 3
  let distance2 := speed2 * time2

  let total_distance := distance1 + distance2

  total_distance = 235 := by
    sorry

end john_total_distance_l1465_146591


namespace robert_total_balls_l1465_146581

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l1465_146581


namespace product_of_second_and_fourth_term_l1465_146536

theorem product_of_second_and_fourth_term (a : ℕ → ℤ) (d : ℤ) (h₁ : a 10 = 25) (h₂ : d = 3)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * d) : a 2 * a 4 = 7 :=
by
  -- Assuming necessary conditions are defined
  sorry

end product_of_second_and_fourth_term_l1465_146536


namespace value_of_k_l1465_146528

theorem value_of_k (k : ℝ) : 
  (∃ P Q R : ℝ × ℝ, P = (5, 12) ∧ Q = (0, k) ∧ dist (0, 0) P = dist (0, 0) Q + 5) → 
  k = 8 := 
by
  sorry

end value_of_k_l1465_146528


namespace gilda_stickers_left_l1465_146538

variable (S : ℝ) (hS : S > 0)

def remaining_after_olga : ℝ := 0.70 * S
def remaining_after_sam : ℝ := 0.80 * remaining_after_olga S
def remaining_after_max : ℝ := 0.70 * remaining_after_sam S
def remaining_after_charity : ℝ := 0.90 * remaining_after_max S

theorem gilda_stickers_left :
  remaining_after_charity S / S * 100 = 35.28 := by
  sorry

end gilda_stickers_left_l1465_146538


namespace players_scores_l1465_146594

/-- Lean code to verify the scores of three players in a guessing game -/
theorem players_scores (H F S : ℕ) (h1 : H = 42) (h2 : F - H = 24) (h3 : S - F = 18) (h4 : H < F) (h5 : H < S) : 
  F = 66 ∧ S = 84 :=
by
  sorry

end players_scores_l1465_146594


namespace length_of_bridge_is_correct_l1465_146529

def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem length_of_bridge_is_correct : 
  length_of_bridge 170 45 30 = 205 :=
by
  -- we state the translation and prove here (proof omitted, just the structure is present)
  sorry

end length_of_bridge_is_correct_l1465_146529


namespace trig_expression_evaluation_l1465_146550

-- Define the given conditions
axiom sin_390 : Real.sin (390 * Real.pi / 180) = 1 / 2
axiom tan_neg_45 : Real.tan (-45 * Real.pi / 180) = -1
axiom cos_360 : Real.cos (360 * Real.pi / 180) = 1

-- Formulate the theorem
theorem trig_expression_evaluation : 
  2 * Real.sin (390 * Real.pi / 180) - Real.tan (-45 * Real.pi / 180) + 5 * Real.cos (360 * Real.pi / 180) = 7 :=
by
  rw [sin_390, tan_neg_45, cos_360]
  sorry

end trig_expression_evaluation_l1465_146550


namespace bicycle_final_price_l1465_146523

-- Define initial conditions
def original_price : ℝ := 200
def wednesday_discount : ℝ := 0.40
def friday_increase : ℝ := 0.20
def saturday_discount : ℝ := 0.25

-- Statement to prove that the final price, after all discounts and increases, is $108
theorem bicycle_final_price :
  (original_price * (1 - wednesday_discount) * (1 + friday_increase) * (1 - saturday_discount)) = 108 := by
  sorry

end bicycle_final_price_l1465_146523


namespace geometric_extraction_from_arithmetic_l1465_146513

theorem geometric_extraction_from_arithmetic (a b : ℤ) :
  ∃ k : ℕ → ℤ, (∀ n : ℕ, k n = a * (b + 1) ^ n) ∧ (∀ n : ℕ, ∃ m : ℕ, k n = a + b * m) :=
by sorry

end geometric_extraction_from_arithmetic_l1465_146513


namespace range_of_m_l1465_146572

def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), (1 ≤ x) → (x^2 - 2*m*x + 1/2 > 0)

def proposition_q (m : ℝ) : Prop :=
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (x^2 - m*x - 2 = 0)

theorem range_of_m (m : ℝ) (h1 : ¬ proposition_q m) (h2 : proposition_p m ∨ proposition_q m) :
  -1 < m ∧ m < 3/4 :=
  sorry

end range_of_m_l1465_146572


namespace area_under_f_l1465_146503

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x + 2 * x - 3

theorem area_under_f' : 
  - ∫ x in (1/2 : ℝ)..1, f' x = (3 / 4) - Real.log 2 := 
by
  sorry

end area_under_f_l1465_146503


namespace work_completion_l1465_146577

theorem work_completion (days_A : ℕ) (days_B : ℕ) (hA : days_A = 14) (hB : days_B = 35) :
  let rate_A := 1 / (days_A : ℚ)
  let rate_B := 1 / (days_B : ℚ)
  let combined_rate := rate_A + rate_B
  let days_AB := 1 / combined_rate
  days_AB = 10 := by
  sorry

end work_completion_l1465_146577


namespace value_of_f_at_2_l1465_146521

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  -- proof steps would go here
  sorry

end value_of_f_at_2_l1465_146521


namespace average_age_increase_l1465_146590

theorem average_age_increase
  (n : ℕ)
  (A : ℝ)
  (w : ℝ)
  (h1 : (n + 1) * (A + w) = n * A + 39)
  (h2 : (n + 1) * (A - 1) = n * A + 15)
  (hw : w = 7) :
  w = 7 := 
by
  sorry

end average_age_increase_l1465_146590


namespace equivalent_statements_l1465_146509
  
variables {A B : Prop}

theorem equivalent_statements :
  ((A ∧ B) → ¬ (A ∨ B)) ↔ ((A ∨ B) → ¬ (A ∧ B)) :=
sorry

end equivalent_statements_l1465_146509


namespace beta_max_two_day_ratio_l1465_146569

noncomputable def alpha_first_day_score : ℚ := 160 / 300
noncomputable def alpha_second_day_score : ℚ := 140 / 200
noncomputable def alpha_two_day_ratio : ℚ := 300 / 500

theorem beta_max_two_day_ratio :
  ∃ (p q r : ℕ), 
  p < 300 ∧
  q < (8 * p / 15) ∧
  r < ((3500 - 7 * p) / 10) ∧
  q + r = 299 ∧
  gcd 299 500 = 1 ∧
  (299 + 500) = 799 := 
sorry

end beta_max_two_day_ratio_l1465_146569


namespace stars_total_is_correct_l1465_146508

-- Define the given conditions
def number_of_stars_per_student : ℕ := 6
def number_of_students : ℕ := 210

-- Define total number of stars calculation
def total_number_of_stars : ℕ := number_of_stars_per_student * number_of_students

-- Proof statement that the total number of stars is correct
theorem stars_total_is_correct : total_number_of_stars = 1260 := by
  sorry

end stars_total_is_correct_l1465_146508


namespace largest_lcm_value_is_90_l1465_146584

def lcm_vals (a b : ℕ) : ℕ := Nat.lcm a b

theorem largest_lcm_value_is_90 :
  max (lcm_vals 18 3)
      (max (lcm_vals 18 9)
           (max (lcm_vals 18 6)
                (max (lcm_vals 18 12)
                     (max (lcm_vals 18 15)
                          (lcm_vals 18 18))))) = 90 :=
by
  -- Use the fact that the calculations of LCMs are as follows:
  -- lcm(18, 3) = 18
  -- lcm(18, 9) = 18
  -- lcm(18, 6) = 18
  -- lcm(18, 12) = 36
  -- lcm(18, 15) = 90
  -- lcm(18, 18) = 18
  -- therefore, the largest value among these is 90
  sorry

end largest_lcm_value_is_90_l1465_146584


namespace range_k_fx_greater_than_ln_l1465_146520

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem range_k (k : ℝ) : 0 ≤ k ∧ k ≤ Real.exp 1 ↔ ∀ x : ℝ, f x ≥ k * x := 
by 
  sorry

theorem fx_greater_than_ln (t : ℝ) (x : ℝ) : t ≤ 2 ∧ 0 < x → f x > t + Real.log x :=
by
  sorry

end range_k_fx_greater_than_ln_l1465_146520


namespace factorize_l1465_146551

theorem factorize (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by 
  sorry

end factorize_l1465_146551


namespace parallelogram_diagonal_square_l1465_146541

theorem parallelogram_diagonal_square (A B C D P Q R S : Type)
    (area_ABCD : ℝ) (proj_A_P_BD proj_C_Q_BD proj_B_R_AC proj_D_S_AC : Prop)
    (PQ RS : ℝ) (d_squared : ℝ) 
    (h_area : area_ABCD = 24)
    (h_proj_A_P : proj_A_P_BD) (h_proj_C_Q : proj_C_Q_BD)
    (h_proj_B_R : proj_B_R_AC) (h_proj_D_S : proj_D_S_AC)
    (h_PQ_length : PQ = 8) (h_RS_length : RS = 10)
    : d_squared = 62 + 20*Real.sqrt 61 := sorry

end parallelogram_diagonal_square_l1465_146541


namespace count_good_numbers_l1465_146564

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l1465_146564


namespace double_root_condition_l1465_146504

theorem double_root_condition (a : ℝ) : 
  (∃! x : ℝ, (x+2)^2 * (x+7)^2 + a = 0) ↔ a = -625 / 16 :=
sorry

end double_root_condition_l1465_146504


namespace sum_even_numbered_terms_l1465_146516

variable (n : ℕ)

def a_n (n : ℕ) : ℕ := 2 * 3^(n-1)

def new_sequence (n : ℕ) : ℕ := a_n (2 * n)

def Sn (n : ℕ) : ℕ := (6 * (1 - 9^n)) / (1 - 9)

theorem sum_even_numbered_terms (n : ℕ) : Sn n = 3 * (9^n - 1) / 4 :=
by sorry

end sum_even_numbered_terms_l1465_146516


namespace sebastian_age_correct_l1465_146595

-- Define the ages involved
def sebastian_age_now := 40
def sister_age_now (S : ℕ) := S - 10
def father_age_now := 85

-- Define the conditions
def age_difference_condition (S : ℕ) := (sister_age_now S) = S - 10
def father_age_condition := father_age_now = 85
def past_age_sum_condition (S : ℕ) := (S - 5) + (sister_age_now S - 5) = 3 / 4 * (father_age_now - 5)

theorem sebastian_age_correct (S : ℕ) 
  (h1 : age_difference_condition S) 
  (h2 : father_age_condition) 
  (h3 : past_age_sum_condition S) : 
  S = sebastian_age_now := 
  by sorry

end sebastian_age_correct_l1465_146595


namespace negation_of_universal_prop_l1465_146555

-- Define the conditions
variable (f : ℝ → ℝ)

-- Theorem statement
theorem negation_of_universal_prop : 
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by
  sorry

end negation_of_universal_prop_l1465_146555


namespace cat_count_l1465_146571

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l1465_146571


namespace sabrina_total_leaves_l1465_146579

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l1465_146579


namespace expansion_term_count_l1465_146574

theorem expansion_term_count 
  (A : Finset ℕ) (B : Finset ℕ) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  (Finset.card (A.product B)) = 12 :=
by {
  sorry
}

end expansion_term_count_l1465_146574


namespace scientific_notation_of_8_36_billion_l1465_146522

theorem scientific_notation_of_8_36_billion : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 8.36 * 10^9 = a * 10^n := 
by
  use 8.36
  use 9
  simp
  sorry

end scientific_notation_of_8_36_billion_l1465_146522


namespace max_cards_possible_l1465_146562

-- Define the dimensions for the cardboard and the card.
def cardboard_length : ℕ := 48
def cardboard_width : ℕ := 36
def card_length : ℕ := 16
def card_width : ℕ := 12

-- State the theorem to prove the maximum number of cards.
theorem max_cards_possible : (cardboard_length / card_length) * (cardboard_width / card_width) = 9 :=
by
  sorry -- Skip the proof, as only the statement is required.

end max_cards_possible_l1465_146562


namespace range_of_a_l1465_146525

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 1 / y = 1) : 
  (x + y + a > 0) ↔ (a > -3 - 2 * Real.sqrt 2) :=
sorry

end range_of_a_l1465_146525


namespace percent_swans_non_ducks_l1465_146502

def percent_ducks : ℝ := 35
def percent_swans : ℝ := 30
def percent_herons : ℝ := 20
def percent_geese : ℝ := 15
def percent_non_ducks := 100 - percent_ducks

theorem percent_swans_non_ducks : (percent_swans / percent_non_ducks) * 100 = 46.15 := 
by
  sorry

end percent_swans_non_ducks_l1465_146502


namespace sufficient_but_not_necessary_condition_for_intersections_l1465_146506

theorem sufficient_but_not_necessary_condition_for_intersections
  (k : ℝ) (h : 0 < k ∧ k < 3) :
  ∃ x y : ℝ, (x - y - k = 0) ∧ ((x - 1)^2 + y^2 = 2) :=
sorry

end sufficient_but_not_necessary_condition_for_intersections_l1465_146506


namespace find_line_eq_l1465_146517

theorem find_line_eq (l : ℝ → ℝ → Prop) :
  (∃ A B : ℝ × ℝ, l A.fst A.snd ∧ l B.fst B.snd ∧ ((A.fst + 1)^2 + (A.snd - 2)^2 = 100 ∧ (B.fst + 1)^2 + (B.snd - 2)^2 = 100)) ∧
  (∃ M : ℝ × ℝ, M = (-2, 3) ∧ (l M.fst M.snd)) →
  (∀ x y : ℝ, l x y ↔ x - y + 5 = 0) :=
by
  sorry

end find_line_eq_l1465_146517


namespace mean_proportion_of_3_and_4_l1465_146565

theorem mean_proportion_of_3_and_4 : ∃ x : ℝ, 3 / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end mean_proportion_of_3_and_4_l1465_146565


namespace find_x2_plus_y2_l1465_146514

theorem find_x2_plus_y2 
  (x y : ℕ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h1 : x * y + x + y = 117) 
  (h2 : x^2 * y + x * y^2 = 1512) : 
  x^2 + y^2 = 549 := 
sorry

end find_x2_plus_y2_l1465_146514


namespace no_such_function_exists_l1465_146549

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ n > 2, f (f (n - 1)) = f (n + 1) - f n :=
by {
  sorry
}

end no_such_function_exists_l1465_146549


namespace assign_teachers_to_classes_l1465_146519

-- Define the given conditions as variables and constants
theorem assign_teachers_to_classes :
  (∃ ways : ℕ, ways = 36) :=
by
  sorry

end assign_teachers_to_classes_l1465_146519


namespace part_I_part_II_l1465_146583

variable (f : ℝ → ℝ)

-- Condition 1: f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

-- Condition 2: f is symmetric about x = 1
axiom symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x)

-- Condition 3: f(x₁ + x₂) = f(x₁) * f(x₂) for x₁, x₂ ∈ [0, 1/2]
axiom multiplicative_on_interval : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1/2) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1/2) → f (x₁ + x₂) = f x₁ * f x₂

-- Given f(1) = 2
axiom f_one : f 1 = 2

-- Part I: Prove f(1/2) = √2 and f(1/4) = 2^(1/4).
theorem part_I : f (1 / 2) = Real.sqrt 2 ∧ f (1 / 4) = Real.sqrt (Real.sqrt 2) := by
  sorry

-- Part II: Prove that f(x) is a periodic function with period 2.
theorem part_II : ∀ x : ℝ, f x = f (x + 2) := by
  sorry

end part_I_part_II_l1465_146583


namespace proof_firstExpr_proof_secondExpr_l1465_146534

noncomputable def firstExpr : ℝ :=
  Real.logb 2 (Real.sqrt (7 / 48)) + Real.logb 2 12 - (1 / 2) * Real.logb 2 42 - 1

theorem proof_firstExpr :
  firstExpr = -3 / 2 :=
by
  sorry

noncomputable def secondExpr : ℝ :=
  (Real.logb 10 2) ^ 2 + Real.logb 10 (2 * Real.logb 10 50 + Real.logb 10 25)

theorem proof_secondExpr :
  secondExpr = 0.0906 + Real.logb 10 5.004 :=
by
  sorry

end proof_firstExpr_proof_secondExpr_l1465_146534


namespace max_sum_n_of_arithmetic_sequence_l1465_146573

/-- Let \( S_n \) be the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with 
a non-zero common difference, and \( a_1 > 0 \). If \( S_5 = S_9 \), then when \( S_n \) is maximum, \( n = 7 \). -/
theorem max_sum_n_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (a1_pos : a 1 > 0) (common_difference : ∀ n, a (n + 1) = a n + d)
  (s5_eq_s9 : S 5 = S 9) :
  ∃ n, (∀ m, m ≤ n → S m ≤ S n) ∧ n = 7 :=
sorry

end max_sum_n_of_arithmetic_sequence_l1465_146573


namespace vegetarian_eaters_l1465_146585

-- Define the conditions
theorem vegetarian_eaters : 
  ∀ (total family_size : ℕ) 
  (only_veg only_nonveg both_veg_nonveg eat_veg : ℕ), 
  family_size = 45 → 
  only_veg = 22 → 
  only_nonveg = 15 → 
  both_veg_nonveg = 8 → 
  eat_veg = only_veg + both_veg_nonveg → 
  eat_veg = 30 :=
by
  intros total family_size only_veg only_nonveg both_veg_nonveg eat_veg
  sorry

end vegetarian_eaters_l1465_146585


namespace temperature_fraction_l1465_146597

def current_temperature : ℤ := 84
def temperature_decrease : ℤ := 21

theorem temperature_fraction :
  (current_temperature - temperature_decrease) = (3 * current_temperature / 4) := 
by
  sorry

end temperature_fraction_l1465_146597


namespace multiply_by_5_l1465_146527

theorem multiply_by_5 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end multiply_by_5_l1465_146527


namespace lexie_crayons_count_l1465_146540

variable (number_of_boxes : ℕ) (crayons_per_box : ℕ)

theorem lexie_crayons_count (h1: number_of_boxes = 10) (h2: crayons_per_box = 8) :
  (number_of_boxes * crayons_per_box) = 80 := by
  sorry

end lexie_crayons_count_l1465_146540


namespace proof_problem_l1465_146543

-- Definitions based on the given conditions
def cond1 : Prop := 1 * 9 + 2 = 11
def cond2 : Prop := 12 * 9 + 3 = 111
def cond3 : Prop := 123 * 9 + 4 = 1111
def cond4 : Prop := 1234 * 9 + 5 = 11111
def cond5 : Prop := 12345 * 9 + 6 = 111111

-- Main statement to prove
theorem proof_problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) : 
  123456 * 9 + 7 = 1111111 :=
sorry

end proof_problem_l1465_146543


namespace probability_two_green_marbles_l1465_146524

open Classical

section
variable (num_red num_green num_white num_blue : ℕ)
variable (total_marbles : ℕ := num_red + num_green + num_white + num_blue)

def probability_green_two_draws (num_green : ℕ) (total_marbles : ℕ) : ℚ :=
  (num_green / total_marbles : ℚ) * ((num_green - 1) / (total_marbles - 1))

theorem probability_two_green_marbles :
  probability_green_two_draws 4 (3 + 4 + 8 + 5) = 3 / 95 := by
  sorry
end

end probability_two_green_marbles_l1465_146524


namespace zoo_animals_total_l1465_146533

-- Conditions as definitions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippopotamus_adopted : ℕ := 1
def rhinos_taken_in : ℕ := 3
def lion_cubs_born : ℕ := 8
def meerkats_per_cub : ℕ := 2

-- Theorem to prove the resulting number of animals
theorem zoo_animals_total :
  (initial_animals - gorillas_sent_away + hippopotamus_adopted + rhinos_taken_in + lion_cubs_born + meerkats_per_cub * lion_cubs_born) = 90 :=
by 
  sorry

end zoo_animals_total_l1465_146533


namespace acute_triangle_probability_correct_l1465_146511

noncomputable def acute_triangle_probability : ℝ :=
  let l_cube_vol := 1
  let quarter_cone_vol := (1/4) * (1/3) * Real.pi * (1^2) * 1
  let total_unfavorable_vol := 3 * quarter_cone_vol
  let favorable_vol := l_cube_vol - total_unfavorable_vol
  favorable_vol / l_cube_vol

theorem acute_triangle_probability_correct : abs (acute_triangle_probability - 0.2146) < 0.0001 :=
  sorry

end acute_triangle_probability_correct_l1465_146511


namespace price_of_bracelets_max_type_a_bracelets_l1465_146560

-- Part 1: Proving the prices of the bracelets
theorem price_of_bracelets :
  ∃ (x y : ℝ), (3 * x + y = 128 ∧ x + 2 * y = 76) ∧ (x = 36 ∧ y = 20) :=
sorry

-- Part 2: Proving the maximum number of type A bracelets they can buy within the budget
theorem max_type_a_bracelets :
  ∃ (m : ℕ), 36 * m + 20 * (100 - m) ≤ 2500 ∧ m = 31 :=
sorry

end price_of_bracelets_max_type_a_bracelets_l1465_146560


namespace quadratic_roots_identity_l1465_146589

theorem quadratic_roots_identity
  (a b c : ℝ)
  (x1 x2 : ℝ)
  (hx1 : x1 = Real.sin (42 * Real.pi / 180))
  (hx2 : x2 = Real.sin (48 * Real.pi / 180))
  (hx2_trig_identity : x2 = Real.cos (42 * Real.pi / 180))
  (hroots : ∀ x, a * x^2 + b * x + c = 0 ↔ (x = x1 ∨ x = x2)) :
  b^2 = a^2 + 2 * a * c :=
by
  sorry

end quadratic_roots_identity_l1465_146589


namespace find_equation_of_line_l1465_146561

theorem find_equation_of_line
  (midpoint : ℝ × ℝ)
  (ellipse : ℝ → ℝ → Prop)
  (l_eq : ℝ → ℝ → Prop)
  (H_mid : midpoint = (1, 2))
  (H_ellipse : ∀ (x y : ℝ), ellipse x y ↔ x^2 / 64 + y^2 / 16 = 1)
  (H_line : ∀ (x y : ℝ), l_eq x y ↔ y - 2 = - (1/8) * (x - 1))
  : ∃ (a b c : ℝ), (a, b, c) = (1, 8, -17) ∧ (∀ (x y : ℝ), l_eq x y ↔ a * x + b * y + c = 0) :=
by 
  sorry

end find_equation_of_line_l1465_146561


namespace total_bees_including_queen_at_end_of_14_days_l1465_146599

-- Conditions definitions
def bees_hatched_per_day : ℕ := 5000
def bees_lost_per_day : ℕ := 1800
def duration_days : ℕ := 14
def initial_bees : ℕ := 20000
def queen_bees : ℕ := 1

-- Question statement as Lean theorem
theorem total_bees_including_queen_at_end_of_14_days :
  (initial_bees + (bees_hatched_per_day - bees_lost_per_day) * duration_days + queen_bees) = 64801 := 
by
  sorry

end total_bees_including_queen_at_end_of_14_days_l1465_146599


namespace range_of_m_l1465_146501
-- Import the essential libraries

-- Define the problem conditions and state the theorem
theorem range_of_m (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_mono_dec : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x)
  (m : ℝ) (h_ineq : f m > f (1 - m)) : m < 1 / 2 :=
sorry

end range_of_m_l1465_146501


namespace craig_age_l1465_146598

theorem craig_age (C M : ℕ) (h1 : C = M - 24) (h2 : C + M = 56) : C = 16 := 
by
  sorry

end craig_age_l1465_146598


namespace units_digit_of_7_pow_y_plus_6_is_9_l1465_146546

theorem units_digit_of_7_pow_y_plus_6_is_9 (y : ℕ) (hy : 0 < y) : 
  (7^y + 6) % 10 = 9 ↔ ∃ k : ℕ, y = 4 * k + 3 := by
  sorry

end units_digit_of_7_pow_y_plus_6_is_9_l1465_146546


namespace exists_five_distinct_nat_numbers_l1465_146548

theorem exists_five_distinct_nat_numbers 
  (a b c d e : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_no_div_3 : ¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e))
  (h_no_div_4 : ¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e))
  (h_no_div_5 : ¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) :
  (∃ (a b c d e : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e)) ∧
    (¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e)) ∧
    (¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) ∧
    (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y + z = a + b + c + d + e → (x + y + z) % 3 = 0) ∧
    (∀ w x y z : ℕ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z → w + x + y + z = a + b + c + d + e → (w + x + y + z) % 4 = 0) ∧
    (a + b + c + d + e) % 5 = 0) :=
  sorry

end exists_five_distinct_nat_numbers_l1465_146548


namespace least_non_lucky_multiple_of_11_l1465_146557

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l1465_146557


namespace pizzas_served_during_lunch_l1465_146515

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem pizzas_served_during_lunch :
  ∃ lunch_pizzas : ℕ, lunch_pizzas = total_pizzas - dinner_pizzas :=
by
  use 9
  exact rfl

end pizzas_served_during_lunch_l1465_146515


namespace floor_area_not_greater_than_10_l1465_146554

theorem floor_area_not_greater_than_10 (L W H : ℝ) (h_height : H = 3)
  (h_more_paint_wall1 : L * 3 > L * W)
  (h_more_paint_wall2 : W * 3 > L * W) :
  L * W ≤ 9 :=
by
  sorry

end floor_area_not_greater_than_10_l1465_146554


namespace length_of_hypotenuse_l1465_146512

theorem length_of_hypotenuse (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 2450) (h2 : c = b + 10) (h3 : a^2 + b^2 = c^2) : c = 35 :=
by
  sorry

end length_of_hypotenuse_l1465_146512


namespace tiling_possible_if_and_only_if_one_dimension_is_integer_l1465_146588

-- Define our conditions: a, b are dimensions of the board and t is the positive dimension of the small rectangles
variable (a b : ℝ) (t : ℝ)

-- Define corresponding properties for these variables
axiom pos_t : t > 0

-- Theorem stating the condition for tiling
theorem tiling_possible_if_and_only_if_one_dimension_is_integer (a_non_int : ¬ ∃ z : ℤ, a = z) (b_non_int : ¬ ∃ z : ℤ, b = z) :
  ∃ n m : ℕ, n * 1 + m * t = a * b :=
sorry

end tiling_possible_if_and_only_if_one_dimension_is_integer_l1465_146588


namespace Yoque_monthly_payment_l1465_146556

theorem Yoque_monthly_payment :
  ∃ m : ℝ, m = 15 ∧ ∀ a t : ℝ, a = 150 ∧ t = 11 ∧ (a + 0.10 * a) / t = m :=
by
  sorry

end Yoque_monthly_payment_l1465_146556


namespace race_head_start_l1465_146530

variable (vA vB L h : ℝ)
variable (hva_vb : vA = (16 / 15) * vB)

theorem race_head_start (hL_pos : L > 0) (hvB_pos : vB > 0) 
    (h_times_eq : (L / vA) = ((L - h) / vB)) : h = L / 16 :=
by
  sorry

end race_head_start_l1465_146530


namespace ball_bounce_height_l1465_146559

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l1465_146559


namespace simplify_expression_l1465_146582

variables {K : Type*} [Field K]

theorem simplify_expression (a b c : K) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) : 
    (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
by
  sorry

end simplify_expression_l1465_146582


namespace percentage_taken_l1465_146578

theorem percentage_taken (P : ℝ) (h : (P / 100) * 150 - 40 = 50) : P = 60 :=
by
  sorry

end percentage_taken_l1465_146578


namespace fraction_equality_l1465_146566

theorem fraction_equality {x y : ℝ} (h : x + y ≠ 0) (h1 : x - y ≠ 0) : 
  (-x + y) / (-x - y) = (x - y) / (x + y) := 
sorry

end fraction_equality_l1465_146566


namespace part1_part2_l1465_146592

-- Definitions for the sets A and B
def A := {x : ℝ | x^2 - 2 * x - 8 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a * x + a^2 - 12 = 0}

-- Proof statements
theorem part1 (a : ℝ) : (A ∩ B a = A) → a = -2 :=
by
  sorry

theorem part2 (a : ℝ) : (A ∪ B a = A) → (a ≥ 4 ∨ a < -4 ∨ a = -2) :=
by
  sorry

end part1_part2_l1465_146592


namespace fibonacci_sequence_x_l1465_146526

theorem fibonacci_sequence_x {a : ℕ → ℕ} 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2) 
  (h3 : a 3 = 3) 
  (h_fib : ∀ n, n ≥ 3 → a (n + 1) = a n + a (n - 1)) : 
  a 5 = 8 := 
sorry

end fibonacci_sequence_x_l1465_146526


namespace geometric_seq_a4_l1465_146552

variable {a : ℕ → ℝ}

-- Definition: a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition
axiom h : a 2 * a 6 = 4

-- Theorem that needs to be proved
theorem geometric_seq_a4 (h_seq: is_geometric_sequence a) (h: a 2 * a 6 = 4) : a 4 = 2 ∨ a 4 = -2 := by
  sorry

end geometric_seq_a4_l1465_146552


namespace certain_event_l1465_146580

-- Definitions of the events as propositions
def EventA : Prop := ∃ n : ℕ, n ≥ 1 ∧ (n % 2 = 0)
def EventB : Prop := ∃ t : ℝ, t ≥ 0  -- Simplifying as the event of an advertisement airing
def EventC : Prop := ∃ w : ℕ, w ≥ 1  -- Simplifying as the event of rain in Weinan on a specific future date
def EventD : Prop := true  -- The sun rises from the east in the morning is always true

-- The statement that Event D is the only certain event among the given options
theorem certain_event : EventD ∧ ¬EventA ∧ ¬EventB ∧ ¬EventC :=
by
  sorry

end certain_event_l1465_146580


namespace max_value_of_function_f_l1465_146537

noncomputable def f (t : ℝ) : ℝ := (4^t - 2 * t) * t / 16^t

theorem max_value_of_function_f : ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 := sorry

end max_value_of_function_f_l1465_146537


namespace next_podcast_duration_l1465_146532

def minutes_in_an_hour : ℕ := 60

def first_podcast_minutes : ℕ := 45
def second_podcast_minutes : ℕ := 2 * first_podcast_minutes
def third_podcast_minutes : ℕ := 105
def fourth_podcast_minutes : ℕ := 60

def total_podcast_minutes : ℕ := first_podcast_minutes + second_podcast_minutes + third_podcast_minutes + fourth_podcast_minutes

def drive_minutes : ℕ := 6 * minutes_in_an_hour

theorem next_podcast_duration :
  (drive_minutes - total_podcast_minutes) / minutes_in_an_hour = 1 :=
by
  sorry

end next_podcast_duration_l1465_146532


namespace sphere_volume_l1465_146545

/-- A sphere is perfectly inscribed in a cube. 
If the edge of the cube measures 10 inches, the volume of the sphere in cubic inches is \(\frac{500}{3}\pi\). -/
theorem sphere_volume (a : ℝ) (h : a = 10) : 
  ∃ V : ℝ, V = (4 / 3) * Real.pi * (a / 2)^3 ∧ V = (500 / 3) * Real.pi :=
by
  use (4 / 3) * Real.pi * (a / 2)^3
  sorry

end sphere_volume_l1465_146545


namespace inequalities_region_quadrants_l1465_146587

theorem inequalities_region_quadrants:
  (∀ x y : ℝ, y > -2 * x + 3 → y > x / 2 + 1 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
sorry

end inequalities_region_quadrants_l1465_146587


namespace Linda_total_distance_is_25_l1465_146593

theorem Linda_total_distance_is_25 : 
  ∃ (x : ℤ), x > 0 ∧ 
  (60/x + 60/(x+5) + 60/(x+10) + 60/(x+15) = 25) :=
by 
  sorry

end Linda_total_distance_is_25_l1465_146593


namespace ella_spent_on_video_games_last_year_l1465_146547

theorem ella_spent_on_video_games_last_year 
  (new_salary : ℝ) 
  (raise : ℝ) 
  (percentage_spent_on_video_games : ℝ) 
  (h_new_salary : new_salary = 275) 
  (h_raise : raise = 0.10) 
  (h_percentage_spent : percentage_spent_on_video_games = 0.40) :
  (new_salary / (1 + raise) * percentage_spent_on_video_games = 100) :=
by
  sorry

end ella_spent_on_video_games_last_year_l1465_146547


namespace find_k_from_hexadecimal_to_decimal_l1465_146542

theorem find_k_from_hexadecimal_to_decimal 
  (k : ℕ) 
  (h : 1 * 6^3 + k * 6 + 5 = 239) : 
  k = 3 := by
  sorry

end find_k_from_hexadecimal_to_decimal_l1465_146542


namespace ribbon_original_length_l1465_146568

theorem ribbon_original_length (x : ℕ) (h1 : 11 * 35 = 7 * x) : x = 55 :=
by
  sorry

end ribbon_original_length_l1465_146568


namespace racers_final_segment_l1465_146567

def final_racer_count : Nat := 9

def segment_eliminations (init_count: Nat) : Nat :=
  let seg1 := init_count - Int.toNat (Nat.sqrt init_count)
  let seg2 := seg1 - seg1 / 3
  let seg3 := seg2 - (seg2 / 4 + (2 ^ 2))
  let seg4 := seg3 - seg3 / 3
  let seg5 := seg4 / 2
  let seg6 := seg5 - (seg5 * 3 / 4)
  seg6

theorem racers_final_segment
  (init_count: Nat)
  (h: init_count = 225) :
  segment_eliminations init_count = final_racer_count :=
  by
  rw [h]
  unfold segment_eliminations
  sorry

end racers_final_segment_l1465_146567


namespace veronica_photo_choices_l1465_146553

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem veronica_photo_choices : choose 5 3 + choose 5 4 = 15 := by
  sorry

end veronica_photo_choices_l1465_146553


namespace difference_of_fractions_l1465_146575

theorem difference_of_fractions (x y : ℝ) (h1 : x = 497) (h2 : y = 325) :
  (2/5) * (3 * x + 7 * y) - (3/5) * (x * y) = -95408.6 := by
  rw [h1, h2]
  sorry

end difference_of_fractions_l1465_146575


namespace range_of_b_l1465_146576

theorem range_of_b (b : ℤ) : 
  (∃ x1 x2 : ℤ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ x1 - b > 0 ∧ x2 - b > 0 ∧ (∀ x : ℤ, x < 0 ∧ x - b > 0 → (x = x1 ∨ x = x2))) ↔ (-3 ≤ b ∧ b < -2) :=
by sorry

end range_of_b_l1465_146576


namespace problem_statement_l1465_146518

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def setB : Set ℝ := {x : ℝ | x ≥ 2}
def setC (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a ≥ 0}

theorem problem_statement (a : ℝ):
  (setA ∩ setB = {x : ℝ | 2 ≤ x ∧ x < 3}) ∧ 
  (setA ∪ setB = {x : ℝ | x ≥ -1}) ∧ 
  (setB ⊆ setC a → a > -4) :=
by
  sorry

end problem_statement_l1465_146518


namespace smallest_m_for_integral_roots_l1465_146500

theorem smallest_m_for_integral_roots :
  ∃ (m : ℕ), (∃ (p q : ℤ), p * q = 30 ∧ m = 12 * (p + q)) ∧ m = 132 := by
  sorry

end smallest_m_for_integral_roots_l1465_146500


namespace Derek_is_42_l1465_146596

def Aunt_Anne_age : ℕ := 36

def Brianna_age : ℕ := (2 * Aunt_Anne_age) / 3

def Caitlin_age : ℕ := Brianna_age - 3

def Derek_age : ℕ := 2 * Caitlin_age

theorem Derek_is_42 : Derek_age = 42 := by
  sorry

end Derek_is_42_l1465_146596
