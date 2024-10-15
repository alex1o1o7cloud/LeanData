import Mathlib

namespace NUMINAMATH_GPT_proof_inequality_l1799_179989

noncomputable def problem (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1 → a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d

theorem proof_inequality (a b c d : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
sorry

end NUMINAMATH_GPT_proof_inequality_l1799_179989


namespace NUMINAMATH_GPT_Jason_spent_on_music_store_l1799_179941

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end NUMINAMATH_GPT_Jason_spent_on_music_store_l1799_179941


namespace NUMINAMATH_GPT_find_weight_of_a_l1799_179968

variables (a b c d e : ℕ)

-- Conditions
def cond1 : Prop := a + b + c = 252
def cond2 : Prop := a + b + c + d = 320
def cond3 : Prop := e = d + 7
def cond4 : Prop := b + c + d + e = 316

theorem find_weight_of_a (h1 : cond1 a b c) (h2 : cond2 a b c d) (h3 : cond3 d e) (h4 : cond4 b c d e) :
  a = 79 :=
by sorry

end NUMINAMATH_GPT_find_weight_of_a_l1799_179968


namespace NUMINAMATH_GPT_B_visible_from_A_l1799_179933

noncomputable def visibility_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x < 3 → 4 * x - 2 > 2 * x^2

theorem B_visible_from_A (a : ℝ) : visibility_condition a ↔ a < 10 :=
by
  -- sorry statement is used to skip the proof part.
  sorry

end NUMINAMATH_GPT_B_visible_from_A_l1799_179933


namespace NUMINAMATH_GPT_total_fireworks_l1799_179921

-- Definitions of the given conditions
def koby_boxes : Nat := 2
def koby_box_sparklers : Nat := 3
def koby_box_whistlers : Nat := 5
def cherie_boxes : Nat := 1
def cherie_box_sparklers : Nat := 8
def cherie_box_whistlers : Nat := 9

-- Statement to prove the total number of fireworks
theorem total_fireworks : 
  let koby_fireworks := koby_boxes * (koby_box_sparklers + koby_box_whistlers)
  let cherie_fireworks := cherie_boxes * (cherie_box_sparklers + cherie_box_whistlers)
  koby_fireworks + cherie_fireworks = 33 := by
  sorry

end NUMINAMATH_GPT_total_fireworks_l1799_179921


namespace NUMINAMATH_GPT_swimming_pool_width_l1799_179985

theorem swimming_pool_width 
  (V_G : ℝ) (G_CF : ℝ) (height_inch : ℝ) (L : ℝ) (V_CF : ℝ) (height_ft : ℝ) (A : ℝ) (W : ℝ) :
  V_G = 3750 → G_CF = 7.48052 → height_inch = 6 → L = 40 →
  V_CF = V_G / G_CF → height_ft = height_inch / 12 →
  A = L * W → V_CF = A * height_ft →
  W = 25.067 :=
by
  intros hV hG hH hL hVC hHF hA hVF
  sorry

end NUMINAMATH_GPT_swimming_pool_width_l1799_179985


namespace NUMINAMATH_GPT_mn_value_l1799_179905

variables {x m n : ℝ} -- Define variables x, m, n as real numbers

theorem mn_value (h : x^2 + m * x - 15 = (x + 3) * (x + n)) : m * n = 10 :=
by {
  -- Sorry for skipping the proof steps
  sorry
}

end NUMINAMATH_GPT_mn_value_l1799_179905


namespace NUMINAMATH_GPT_can_weight_is_two_l1799_179978

theorem can_weight_is_two (c : ℕ) (h1 : 100 = 20 * c + 6 * ((100 - 20 * c) / 6)) (h2 : 160 = 10 * ((100 - 20 * c) / 6) + 3 * 20) : c = 2 :=
by
  sorry

end NUMINAMATH_GPT_can_weight_is_two_l1799_179978


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l1799_179966

theorem fraction_to_terminating_decimal : (21 : ℚ) / 40 = 0.525 := 
by
  sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l1799_179966


namespace NUMINAMATH_GPT_mysterious_neighbor_is_13_l1799_179937

variable (x : ℕ) (h1 : x < 15) (h2 : 2 * x * 30 = 780)

theorem mysterious_neighbor_is_13 : x = 13 :=
by {
    sorry 
}

end NUMINAMATH_GPT_mysterious_neighbor_is_13_l1799_179937


namespace NUMINAMATH_GPT_avg_income_pr_l1799_179991

theorem avg_income_pr (P Q R : ℝ) 
  (h_avgPQ : (P + Q) / 2 = 5050) 
  (h_avgQR : (Q + R) / 2 = 6250)
  (h_P : P = 4000) 
  : (P + R) / 2 = 5200 := 
by 
  sorry

end NUMINAMATH_GPT_avg_income_pr_l1799_179991


namespace NUMINAMATH_GPT_johnson_family_seating_l1799_179961

-- Defining the total number of children:
def total_children := 8

-- Defining the number of sons and daughters:
def sons := 5
def daughters := 3

-- Factoring in the total number of unrestricted seating arrangements:
def total_seating_arrangements : ℕ := Nat.factorial total_children

-- Factoring in the number of non-adjacent seating arrangements for sons:
def non_adjacent_arrangements : ℕ :=
  (Nat.factorial daughters) * (Nat.factorial sons)

-- The lean proof statement to prove:
theorem johnson_family_seating :
  total_seating_arrangements - non_adjacent_arrangements = 39600 :=
by
  sorry

end NUMINAMATH_GPT_johnson_family_seating_l1799_179961


namespace NUMINAMATH_GPT_max_value_E_zero_l1799_179917

noncomputable def E (a b c : ℝ) : ℝ :=
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2)

theorem max_value_E_zero (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≥ b * c^2) (h2 : b ≥ c * a^2) (h3 : c ≥ a * b^2) :
  E a b c ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_max_value_E_zero_l1799_179917


namespace NUMINAMATH_GPT_greatest_consecutive_integers_sum_120_l1799_179953

def sum_of_consecutive_integers (n : ℤ) (a : ℤ) : ℤ :=
  n * (2 * a + n - 1) / 2

theorem greatest_consecutive_integers_sum_120 (N : ℤ) (a : ℤ) (h1 : sum_of_consecutive_integers N a = 120) : N ≤ 240 :=
by {
  -- Here we would provide the proof, but it's omitted with 'sorry'.
  sorry
}

end NUMINAMATH_GPT_greatest_consecutive_integers_sum_120_l1799_179953


namespace NUMINAMATH_GPT_probability_of_exactly_9_correct_matches_is_zero_l1799_179997

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∃ (P : ℕ → ℕ → ℕ), 
    (∀ (total correct : ℕ), 
      total = 10 → 
      correct = 9 → 
      P total correct = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_exactly_9_correct_matches_is_zero_l1799_179997


namespace NUMINAMATH_GPT_sum_of_two_consecutive_negative_integers_l1799_179987

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 2210) (hn : n < 0) : n + (n + 1) = -95 := 
sorry

end NUMINAMATH_GPT_sum_of_two_consecutive_negative_integers_l1799_179987


namespace NUMINAMATH_GPT_find_added_value_l1799_179901

theorem find_added_value (avg_15_numbers : ℤ) (new_avg : ℤ) (x : ℤ)
    (H1 : avg_15_numbers = 40) 
    (H2 : new_avg = 50) 
    (H3 : (600 + 15 * x) / 15 = new_avg) : 
    x = 10 := 
sorry

end NUMINAMATH_GPT_find_added_value_l1799_179901


namespace NUMINAMATH_GPT_nine_y_squared_eq_x_squared_z_squared_l1799_179975

theorem nine_y_squared_eq_x_squared_z_squared (x y z : ℝ) (h : x / y = 3 / z) : 9 * y ^ 2 = x ^ 2 * z ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_nine_y_squared_eq_x_squared_z_squared_l1799_179975


namespace NUMINAMATH_GPT_number_of_six_digit_palindromes_l1799_179964

def is_six_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ n = a * 100001 + b * 10010 + c * 1100

theorem number_of_six_digit_palindromes : ∃ p, p = 900 ∧ (∀ n, is_six_digit_palindrome n → n = p) :=
by
  sorry

end NUMINAMATH_GPT_number_of_six_digit_palindromes_l1799_179964


namespace NUMINAMATH_GPT_area_ratio_of_square_side_multiplied_by_10_l1799_179984

theorem area_ratio_of_square_side_multiplied_by_10 (s : ℝ) (A_original A_resultant : ℝ) 
  (h1 : A_original = s^2)
  (h2 : A_resultant = (10 * s)^2) :
  (A_original / A_resultant) = (1 / 100) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_square_side_multiplied_by_10_l1799_179984


namespace NUMINAMATH_GPT_regular_polygon_sides_l1799_179903

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1799_179903


namespace NUMINAMATH_GPT_all_three_white_probability_l1799_179993

noncomputable def box_probability : ℚ :=
  let total_white := 4
  let total_black := 7
  let total_balls := total_white + total_black
  let draw_count := 3
  let total_combinations := (total_balls.choose draw_count : ℕ)
  let favorable_combinations := (total_white.choose draw_count : ℕ)
  (favorable_combinations : ℚ) / (total_combinations : ℚ)

theorem all_three_white_probability :
  box_probability = 4 / 165 :=
by
  sorry

end NUMINAMATH_GPT_all_three_white_probability_l1799_179993


namespace NUMINAMATH_GPT_race_problem_l1799_179972

theorem race_problem 
  (A B C : ℝ) 
  (h1 : A = 100) 
  (h2 : B = 100 - x) 
  (h3 : C = 72) 
  (h4 : B = C + 4)
  : x = 24 := 
by 
  sorry

end NUMINAMATH_GPT_race_problem_l1799_179972


namespace NUMINAMATH_GPT_find_land_area_l1799_179919

variable (L : ℝ) -- cost of land per square meter
variable (B : ℝ) -- cost of bricks per 1000 bricks
variable (R : ℝ) -- cost of roof tiles per tile
variable (numBricks : ℝ) -- number of bricks needed
variable (numTiles : ℝ) -- number of roof tiles needed
variable (totalCost : ℝ) -- total construction cost

theorem find_land_area (h1 : L = 50) 
                       (h2 : B = 100)
                       (h3 : R = 10) 
                       (h4 : numBricks = 10000) 
                       (h5 : numTiles = 500) 
                       (h6 : totalCost = 106000) : 
                       ∃ x : ℝ, 50 * x + (numBricks / 1000) * B + numTiles * R = totalCost ∧ x = 2000 := 
by 
  use 2000
  simp [h1, h2, h3, h4, h5, h6]
  norm_num
  done

end NUMINAMATH_GPT_find_land_area_l1799_179919


namespace NUMINAMATH_GPT_jeremy_school_distance_l1799_179950

theorem jeremy_school_distance :
  ∃ d : ℝ, d = 9.375 ∧
  (∃ v : ℝ, (d = v * (15 / 60)) ∧ (d = (v + 25) * (9 / 60))) := by
  sorry

end NUMINAMATH_GPT_jeremy_school_distance_l1799_179950


namespace NUMINAMATH_GPT_value_of_k_l1799_179967

theorem value_of_k (k : ℝ) :
  (5 + ∑' n : ℕ, (5 + k * (2^n / 4^n))) / 4^n = 10 → k = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1799_179967


namespace NUMINAMATH_GPT_zongzi_unit_prices_max_type_A_zongzi_l1799_179907

theorem zongzi_unit_prices (x : ℝ) : 
  (800 / x - 1200 / (2 * x) = 50) → 
  (x = 4 ∧ 2 * x = 8) :=
by
  intro h
  sorry

theorem max_type_A_zongzi (m : ℕ) : 
  (m ≤ 200) → 
  (8 * m + 4 * (200 - m) ≤ 1150) → 
  (m ≤ 87) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_zongzi_unit_prices_max_type_A_zongzi_l1799_179907


namespace NUMINAMATH_GPT_sally_more_cards_than_dan_l1799_179983

theorem sally_more_cards_than_dan :
  let sally_initial := 27
  let sally_bought := 20
  let dan_cards := 41
  sally_initial + sally_bought - dan_cards = 6 :=
by
  sorry

end NUMINAMATH_GPT_sally_more_cards_than_dan_l1799_179983


namespace NUMINAMATH_GPT_no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l1799_179999

-- Part (i)
theorem no_solutions_for_a_ne_4 (a : ℕ) (h : a ≠ 4) :
  ¬∃ (u v : ℕ), (u > 0 ∧ v > 0 ∧ u^2 + v^2 - a * u * v + 2 = 0) :=
by sorry

-- Part (ii)
theorem solutions_for_a_eq_4_infinite :
  ∃ (a_seq : ℕ → ℕ),
    (a_seq 0 = 1 ∧ a_seq 1 = 3 ∧
     ∀ n, a_seq (n + 2) = 4 * a_seq (n + 1) - a_seq n ∧
    ∀ n, (a_seq n) > 0 ∧ (a_seq (n + 1)) > 0 ∧ (a_seq n)^2 + (a_seq (n + 1))^2 - 4 * (a_seq n) * (a_seq (n + 1)) + 2 = 0) :=
by sorry

end NUMINAMATH_GPT_no_solutions_for_a_ne_4_solutions_for_a_eq_4_infinite_l1799_179999


namespace NUMINAMATH_GPT_extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l1799_179952

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - (x + 1) ^ 2

-- Question 1: Extreme value when a = -1
theorem extreme_value_when_a_is_neg_one : 
  f (-1) (-1) = 1 / exp 1 := sorry

-- Question 2: Range of a such that ∀ x ∈ [-1, 1], f(x) ≤ 0
theorem range_of_a_for_f_non_positive :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 4 / exp 1 := sorry

end NUMINAMATH_GPT_extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l1799_179952


namespace NUMINAMATH_GPT_missing_fraction_is_73_div_60_l1799_179980

-- Definition of the given fractions
def fraction1 : ℚ := 1/3
def fraction2 : ℚ := 1/2
def fraction3 : ℚ := -5/6
def fraction4 : ℚ := 1/5
def fraction5 : ℚ := 1/4
def fraction6 : ℚ := -5/6

-- Total sum provided in the problem
def total_sum : ℚ := 50/60  -- 0.8333333333333334 in decimal form

-- The summation of given fractions
def sum_of_fractions : ℚ := fraction1 + fraction2 + fraction3 + fraction4 + fraction5 + fraction6

-- The statement to prove that the missing fraction is 73/60
theorem missing_fraction_is_73_div_60 : (total_sum - sum_of_fractions) = 73/60 := by
  sorry

end NUMINAMATH_GPT_missing_fraction_is_73_div_60_l1799_179980


namespace NUMINAMATH_GPT_bella_bracelets_l1799_179909

theorem bella_bracelets (h_beads_per_bracelet : Nat)
  (h_initial_beads : Nat) 
  (h_additional_beads : Nat) 
  (h_friends : Nat):
  h_beads_per_bracelet = 8 →
  h_initial_beads = 36 →
  h_additional_beads = 12 →
  h_friends = (h_initial_beads + h_additional_beads) / h_beads_per_bracelet →
  h_friends = 6 :=
by
  intros h_beads_per_bracelet_eq h_initial_beads_eq h_additional_beads_eq h_friends_eq
  subst_vars
  sorry

end NUMINAMATH_GPT_bella_bracelets_l1799_179909


namespace NUMINAMATH_GPT_original_surface_area_l1799_179918

theorem original_surface_area (R : ℝ) (h : 2 * π * R^2 = 4 * π) : 4 * π * R^2 = 8 * π :=
by
  sorry

end NUMINAMATH_GPT_original_surface_area_l1799_179918


namespace NUMINAMATH_GPT_thirteen_pow_seven_mod_nine_l1799_179912

theorem thirteen_pow_seven_mod_nine : (13^7 % 9 = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_thirteen_pow_seven_mod_nine_l1799_179912


namespace NUMINAMATH_GPT_total_pens_bought_l1799_179924

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end NUMINAMATH_GPT_total_pens_bought_l1799_179924


namespace NUMINAMATH_GPT_a1_a2_a3_sum_l1799_179927

-- Given conditions and hypothesis
variables (a0 a1 a2 a3 : ℝ)
axiom H : ∀ x : ℝ, 1 + x + x^2 + x^3 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3

-- Goal statement to be proven
theorem a1_a2_a3_sum : a1 + a2 + a3 = -3 :=
sorry

end NUMINAMATH_GPT_a1_a2_a3_sum_l1799_179927


namespace NUMINAMATH_GPT_profit_sharing_l1799_179946

theorem profit_sharing 
  (total_profit : ℝ) 
  (managing_share_percentage : ℝ) 
  (capital_a : ℝ) 
  (capital_b : ℝ) 
  (managing_partner_share : ℝ)
  (total_capital : ℝ) 
  (remaining_profit : ℝ) 
  (proportion_a : ℝ)
  (share_a_remaining : ℝ)
  (total_share_a : ℝ) : 
  total_profit = 8800 → 
  managing_share_percentage = 0.125 → 
  capital_a = 50000 → 
  capital_b = 60000 → 
  managing_partner_share = managing_share_percentage * total_profit → 
  total_capital = capital_a + capital_b → 
  remaining_profit = total_profit - managing_partner_share → 
  proportion_a = capital_a / total_capital → 
  share_a_remaining = proportion_a * remaining_profit → 
  total_share_a = managing_partner_share + share_a_remaining → 
  total_share_a = 4600 :=
by sorry

end NUMINAMATH_GPT_profit_sharing_l1799_179946


namespace NUMINAMATH_GPT_number_of_marbles_pat_keeps_l1799_179998

theorem number_of_marbles_pat_keeps 
  (x : ℕ) 
  (h1 : x / 6 = 9) 
  : x / 3 = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_marbles_pat_keeps_l1799_179998


namespace NUMINAMATH_GPT_part_i_l1799_179976

theorem part_i (n : ℕ) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 := by
  sorry

end NUMINAMATH_GPT_part_i_l1799_179976


namespace NUMINAMATH_GPT_books_bought_l1799_179931

def cost_price_of_books (n : ℕ) (C : ℝ) (S : ℝ) : Prop :=
  n * C = 16 * S

def gain_or_loss_percentage (gain_loss_percent : ℝ) : Prop :=
  gain_loss_percent = 0.5

def loss_selling_price (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) : Prop :=
  S = (1 - gain_loss_percent) * C
  
theorem books_bought (n : ℕ) (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) 
  (h1 : cost_price_of_books n C S) 
  (h2 : gain_or_loss_percentage gain_loss_percent) 
  (h3 : loss_selling_price C S gain_loss_percent) : 
  n = 8 := 
sorry 

end NUMINAMATH_GPT_books_bought_l1799_179931


namespace NUMINAMATH_GPT_interest_rate_increase_l1799_179925

-- Define the conditions
def principal (P : ℕ) := P = 1000
def time (t : ℕ) := t = 5
def original_amount (A : ℕ) := A = 1500
def new_amount (A' : ℕ) := A' = 1750

-- Prove that the interest rate increase is 50%
theorem interest_rate_increase
  (P : ℕ) (t : ℕ) (A A' : ℕ)
  (hP : principal P)
  (ht : time t)
  (hA : original_amount A)
  (hA' : new_amount A') :
  (((((A' - P) / (P * t)) - ((A - P) / (P * t))) / ((A - P) / (P * t))) * 100) = 50 := by
  sorry

end NUMINAMATH_GPT_interest_rate_increase_l1799_179925


namespace NUMINAMATH_GPT_simplify_expression_l1799_179911

variable (w : ℝ)

theorem simplify_expression : 3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1799_179911


namespace NUMINAMATH_GPT_solution_set_f_ge_1_l1799_179957

noncomputable def f (x : ℝ) (a : ℝ) :=
  if x >= 0 then |x - 2| + a else -(|-x - 2| + a)

theorem solution_set_f_ge_1 {a : ℝ} (ha : a = -2) :
  {x : ℝ | f x a ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 5} :=
by sorry

end NUMINAMATH_GPT_solution_set_f_ge_1_l1799_179957


namespace NUMINAMATH_GPT_triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l1799_179965

theorem triangle_side_square_sum_eq_three_times_centroid_dist_square_sum
  {A B C O : EuclideanSpace ℝ (Fin 2)}
  (h_centroid : O = (1/3 : ℝ) • (A + B + C)) :
  (dist A B)^2 + (dist B C)^2 + (dist C A)^2 =
  3 * ((dist O A)^2 + (dist O B)^2 + (dist O C)^2) :=
sorry

end NUMINAMATH_GPT_triangle_side_square_sum_eq_three_times_centroid_dist_square_sum_l1799_179965


namespace NUMINAMATH_GPT_problem_statement_l1799_179914

def P (m n : ℕ) : ℕ :=
  let coeff_x := Nat.choose 4 m
  let coeff_y := Nat.choose 6 n
  coeff_x * coeff_y

theorem problem_statement : P 2 1 + P 1 2 = 96 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1799_179914


namespace NUMINAMATH_GPT_bowling_ball_weight_l1799_179982

theorem bowling_ball_weight :
  (∃ (b c : ℝ), 8 * b = 4 * c ∧ 2 * c = 64) → ∃ b : ℝ, b = 16 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1799_179982


namespace NUMINAMATH_GPT_smallest_n_for_fraction_with_digits_439_l1799_179971

theorem smallest_n_for_fraction_with_digits_439 (m n : ℕ) (hmn : Nat.gcd m n = 1) (hmn_pos : 0 < m ∧ m < n) (digits_439 : ∃ X : ℕ, (m : ℚ) / n = (439 + 1000 * X) / 1000) : n = 223 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_fraction_with_digits_439_l1799_179971


namespace NUMINAMATH_GPT_two_digit_sum_divisible_by_17_l1799_179922

theorem two_digit_sum_divisible_by_17 :
  ∃ A : ℕ, A ≥ 10 ∧ A < 100 ∧ ∃ B : ℕ, B = (A % 10) * 10 + (A / 10) ∧ (A + B) % 17 = 0 ↔ A = 89 ∨ A = 98 := 
sorry

end NUMINAMATH_GPT_two_digit_sum_divisible_by_17_l1799_179922


namespace NUMINAMATH_GPT_find_some_number_l1799_179945

-- The conditions of the problem
variables (x y : ℝ)
axiom cond1 : 2 * x + y = 7
axiom cond2 : x + 2 * y = 5

-- The "some number" we want to prove exists
def some_number := 3

-- Statement of the problem: the value of 2xy / some_number should equal 2
theorem find_some_number (x y : ℝ) (cond1 : 2 * x + y = 7) (cond2 : x + 2 * y = 5) :
  2 * x * y / some_number = 2 :=
sorry

end NUMINAMATH_GPT_find_some_number_l1799_179945


namespace NUMINAMATH_GPT_n_squared_divides_2n_plus_1_l1799_179951

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end NUMINAMATH_GPT_n_squared_divides_2n_plus_1_l1799_179951


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_504_l1799_179992

theorem sum_of_three_consecutive_integers_product_504 : 
  ∃ n : ℤ, n * (n + 1) * (n + 2) = 504 ∧ n + (n + 1) + (n + 2) = 24 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_product_504_l1799_179992


namespace NUMINAMATH_GPT_certain_amount_l1799_179935

theorem certain_amount (x : ℝ) (h1 : 2 * x = 86 - 54) (h2 : 8 + 3 * 8 = 24) (h3 : 86 - 54 + 32 = 86) : x = 43 := 
by {
  sorry
}

end NUMINAMATH_GPT_certain_amount_l1799_179935


namespace NUMINAMATH_GPT_solve_quadratic_roots_l1799_179908

theorem solve_quadratic_roots : ∀ x : ℝ, (x - 1)^2 = 1 → (x = 2 ∨ x = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_roots_l1799_179908


namespace NUMINAMATH_GPT_Mayor_decision_to_adopt_model_A_l1799_179928

-- Define the conditions
def num_people := 17

def radicals_support_model_A := (0 : ℕ)

def socialists_support_model_B (y : ℕ) := y

def republicans_support_model_B (x y : ℕ) := x - y

def independents_support_model_B (x y : ℕ) := (y + (x - y)) / 2

-- The number of individuals supporting model A and model B
def support_model_B (x y : ℕ) := radicals_support_model_A + socialists_support_model_B y + republicans_support_model_B x y + independents_support_model_B x y

def support_model_A (x : ℕ) := 4 * x - support_model_B x x / 2

-- Statement to prove
theorem Mayor_decision_to_adopt_model_A (x : ℕ) (h : x = num_people) : 
  support_model_A x > support_model_B x x := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_Mayor_decision_to_adopt_model_A_l1799_179928


namespace NUMINAMATH_GPT_equivalent_problem_l1799_179943

theorem equivalent_problem : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 := by
  sorry

end NUMINAMATH_GPT_equivalent_problem_l1799_179943


namespace NUMINAMATH_GPT_find_values_of_a_l1799_179996

noncomputable def has_one_real_solution (a : ℝ) : Prop :=
  ∃ x: ℝ, (x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0) ∧ (∀ y: ℝ, (y^3 - a*y^2 - 3*a*y + a^2 - 1 = 0) → y = x)

theorem find_values_of_a : ∀ a: ℝ, has_one_real_solution a ↔ a < -(5 / 4) :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_l1799_179996


namespace NUMINAMATH_GPT_weight_of_each_bag_is_correct_l1799_179958

noncomputable def weightOfEachBag
    (days1 : ℕ := 60)
    (consumption1 : ℕ := 2)
    (days2 : ℕ := 305)
    (consumption2 : ℕ := 4)
    (ouncesPerPound : ℕ := 16)
    (numberOfBags : ℕ := 17) : ℝ :=
        let totalOunces := (days1 * consumption1) + (days2 * consumption2)
        let totalPounds := totalOunces / ouncesPerPound
        totalPounds / numberOfBags

theorem weight_of_each_bag_is_correct :
  weightOfEachBag = 4.93 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_bag_is_correct_l1799_179958


namespace NUMINAMATH_GPT_initial_trees_l1799_179981

theorem initial_trees (DeadTrees CutTrees LeftTrees : ℕ) (h1 : DeadTrees = 15) (h2 : CutTrees = 23) (h3 : LeftTrees = 48) :
  DeadTrees + CutTrees + LeftTrees = 86 :=
by
  sorry

end NUMINAMATH_GPT_initial_trees_l1799_179981


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l1799_179960

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)) : a 7 = 9 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l1799_179960


namespace NUMINAMATH_GPT_cube_surface_area_l1799_179942

noncomputable def volume_of_cube (s : ℝ) := s ^ 3
noncomputable def surface_area_of_cube (s : ℝ) := 6 * (s ^ 2)

theorem cube_surface_area (s : ℝ) (h : volume_of_cube s = 1728) : surface_area_of_cube s = 864 :=
  sorry

end NUMINAMATH_GPT_cube_surface_area_l1799_179942


namespace NUMINAMATH_GPT_find_cos_alpha_l1799_179932

theorem find_cos_alpha 
  (α : ℝ) 
  (h₁ : Real.tan (π - α) = 3/4) 
  (h₂ : α ∈ Set.Ioo (π/2) π) 
: Real.cos α = -4/5 :=
sorry

end NUMINAMATH_GPT_find_cos_alpha_l1799_179932


namespace NUMINAMATH_GPT_parabola_equation_l1799_179963

theorem parabola_equation 
  (vertex_x vertex_y : ℝ)
  (a b c : ℝ)
  (h_vertex : vertex_x = 3 ∧ vertex_y = 5)
  (h_point : ∃ x y: ℝ, x = 2 ∧ y = 2 ∧ y = a * (x - vertex_x)^2 + vertex_y)
  (h_vertical_axis : ∃ a b c, a = -3 ∧ b = 18 ∧ c = -22):
  ∀ x: ℝ, x ≠ vertex_x → b^2 - 4 * a * c > 0 := 
    sorry

end NUMINAMATH_GPT_parabola_equation_l1799_179963


namespace NUMINAMATH_GPT_smallest_number_l1799_179969

theorem smallest_number
  (A : ℕ := 2^3 + 2^2 + 2^1 + 2^0)
  (B : ℕ := 2 * 6^2 + 1 * 6)
  (C : ℕ := 1 * 4^3)
  (D : ℕ := 8 + 1) :
  A < B ∧ A < C ∧ A < D :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_l1799_179969


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l1799_179936

theorem arithmetic_geometric_sequence : 
  ∀ (a : ℤ), (∀ n : ℤ, a_n = a + (n-1) * 2) → 
  (a + 4)^2 = a * (a + 6) → 
  (a + 10 = 2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l1799_179936


namespace NUMINAMATH_GPT_find_point_P_l1799_179994

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end NUMINAMATH_GPT_find_point_P_l1799_179994


namespace NUMINAMATH_GPT_canary_possible_distances_l1799_179990

noncomputable def distance_from_bus_stop (bus_stop swallow sparrow canary : ℝ) : Prop :=
  swallow = 380 ∧
  sparrow = 450 ∧
  (sparrow - swallow) = (canary - sparrow) ∨
  (swallow - sparrow) = (sparrow - canary)

theorem canary_possible_distances (swallow sparrow canary : ℝ) :
  distance_from_bus_stop 0 swallow sparrow canary →
  canary = 520 ∨ canary = 1280 :=
by
  sorry

end NUMINAMATH_GPT_canary_possible_distances_l1799_179990


namespace NUMINAMATH_GPT_max_value_of_f_on_S_l1799_179915

noncomputable def S : Set ℝ := { x | x^4 - 13 * x^2 + 36 ≤ 0 }
noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem max_value_of_f_on_S : ∃ x ∈ S, ∀ y ∈ S, f y ≤ f x ∧ f x = 18 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_on_S_l1799_179915


namespace NUMINAMATH_GPT_car_travel_distance_l1799_179995

-- Define the conditions
def speed : ℝ := 23
def time : ℝ := 3

-- Define the formula for distance
def distance_traveled (s : ℝ) (t : ℝ) : ℝ := s * t

-- State the theorem to prove the distance the car traveled
theorem car_travel_distance : distance_traveled speed time = 69 :=
by
  -- The proof would normally go here, but we're skipping it as per the instructions
  sorry

end NUMINAMATH_GPT_car_travel_distance_l1799_179995


namespace NUMINAMATH_GPT_christine_stickers_needed_l1799_179906

-- Define the number of stickers Christine has
def stickers_has : ℕ := 11

-- Define the number of stickers required for the prize
def stickers_required : ℕ := 30

-- Define the formula to calculate the number of stickers Christine needs
def stickers_needed : ℕ := stickers_required - stickers_has

-- The theorem we need to prove
theorem christine_stickers_needed : stickers_needed = 19 :=
by
  sorry

end NUMINAMATH_GPT_christine_stickers_needed_l1799_179906


namespace NUMINAMATH_GPT_complement_intersection_complement_in_U_l1799_179940

universe u
open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Definitions based on the conditions
def universal_set : Set ℕ := { x ∈ (Set.univ : Set ℕ) | x ≤ 4 }
def set_A : Set ℕ := {1, 4}
def set_B : Set ℕ := {2, 4}

-- Problem to be proven
theorem complement_intersection_complement_in_U :
  (U = universal_set) → (A = set_A) → (B = set_B) →
  compl (A ∩ B) ∩ U = {1, 2, 3} :=
by
  intro hU hA hB
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_intersection_complement_in_U_l1799_179940


namespace NUMINAMATH_GPT_proof_second_number_is_30_l1799_179977

noncomputable def second_number_is_30 : Prop :=
  ∃ (a b c : ℕ), 
    a + b + c = 98 ∧ 
    (a / (gcd a b) = 2) ∧ (b / (gcd a b) = 3) ∧
    (b / (gcd b c) = 5) ∧ (c / (gcd b c) = 8) ∧
    b = 30

theorem proof_second_number_is_30 : second_number_is_30 :=
  sorry

end NUMINAMATH_GPT_proof_second_number_is_30_l1799_179977


namespace NUMINAMATH_GPT_product_a_b_l1799_179962

variable (a b c : ℝ)
variable (h_pos_a : a > 0)
variable (h_pos_b : b > 0)
variable (h_pos_c : c > 0)
variable (h_c : c = 3)
variable (h_a : a = b^2)
variable (h_bc : b + c = b * c)

theorem product_a_b : a * b = 27 / 8 :=
by
  -- We need to prove that given the above conditions, a * b = 27 / 8
  sorry

end NUMINAMATH_GPT_product_a_b_l1799_179962


namespace NUMINAMATH_GPT_root_division_7_pow_l1799_179900

theorem root_division_7_pow : 
  ( (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = 7 ^ (3 / 28) ) :=
sorry

end NUMINAMATH_GPT_root_division_7_pow_l1799_179900


namespace NUMINAMATH_GPT_g_60_l1799_179974

noncomputable def g : ℝ → ℝ :=
sorry

axiom g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

axiom g_45 : g 45 = 15

theorem g_60 : g 60 = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_g_60_l1799_179974


namespace NUMINAMATH_GPT_sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l1799_179934

-- 1. Sum of the interior angles in a triangle is 180 degrees.
theorem sum_of_angles_in_triangle : ∀ a : ℕ, (∀ x y z : ℕ, x + y + z = 180) → a = 180 := by
  intros a h
  have : a = 180 := sorry
  exact this

-- 2. Sum of interior angles of a regular b-sided polygon is 1080 degrees.
theorem sum_of_angles_in_polygon : ∀ b : ℕ, ((b - 2) * 180 = 1080) → b = 8 := by
  intros b h
  have : b = 8 := sorry
  exact this

-- 3. Exponential equation involving b.
theorem exponential_equation : ∀ p b : ℕ, (8 ^ b = p ^ 21) ∧ (b = 8) → p = 2 := by
  intros p b h
  have : p = 2 := sorry
  exact this

-- 4. Logarithmic equation involving p.
theorem logarithmic_equation : ∀ q p : ℕ, (p = Real.log 81 / Real.log q) ∧ (p = 2) → q = 9 := by
  intros q p h
  have : q = 9 := sorry
  exact this

end NUMINAMATH_GPT_sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l1799_179934


namespace NUMINAMATH_GPT_base_number_is_two_l1799_179938

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^18) (h2 : n = 17) : x = 2 :=
by sorry

end NUMINAMATH_GPT_base_number_is_two_l1799_179938


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1799_179916

open Real

theorem boat_speed_in_still_water (V_s d t : ℝ) (h1 : V_s = 6) (h2 : d = 72) (h3 : t = 3.6) :
  ∃ (V_b : ℝ), V_b = 14 := by
  have V_d := d / t
  have V_b := V_d - V_s
  use V_b
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l1799_179916


namespace NUMINAMATH_GPT_fifth_inequality_proof_l1799_179954

theorem fifth_inequality_proof :
  (1 + 1 / (2^2 : ℝ) + 1 / (3^2 : ℝ) + 1 / (4^2 : ℝ) + 1 / (5^2 : ℝ) + 1 / (6^2 : ℝ) < 11 / 6) 
  := 
sorry

end NUMINAMATH_GPT_fifth_inequality_proof_l1799_179954


namespace NUMINAMATH_GPT_smallest_value_of_N_l1799_179970

theorem smallest_value_of_N :
  ∃ N : ℕ, ∀ (P1 P2 P3 P4 P5 : ℕ) (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℕ),
    (P1 = 1 ∧ P2 = 2 ∧ P3 = 3 ∧ P4 = 4 ∧ P5 = 5) →
    (x1 = a_1 ∧ x2 = N + a_2 ∧ x3 = 2 * N + a_3 ∧ x4 = 3 * N + a_4 ∧ x5 = 4 * N + a_5) →
    (y1 = 5 * (a_1 - 1) + 1 ∧ y2 = 5 * (a_2 - 1) + 2 ∧ y3 = 5 * (a_3 - 1) + 3 ∧ y4 = 5 * (a_4 - 1) + 4 ∧ y5 = 5 * (a_5 - 1) + 5) →
    (x1 = y2 ∧ x2 = y1 ∧ x3 = y4 ∧ x4 = y5 ∧ x5 = y3) →
    N = 149 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_N_l1799_179970


namespace NUMINAMATH_GPT_joe_bath_shop_bottles_l1799_179948

theorem joe_bath_shop_bottles (b : ℕ) (n : ℕ) (m : ℕ) 
    (h1 : 5 * n = b * m)
    (h2 : 5 * n = 95)
    (h3 : b * m = 95)
    (h4 : b ≠ 1)
    (h5 : b ≠ 95): 
    b = 19 := 
by 
    sorry

end NUMINAMATH_GPT_joe_bath_shop_bottles_l1799_179948


namespace NUMINAMATH_GPT_speed_of_stream_l1799_179986

variable (b s : ℝ)

theorem speed_of_stream (h1 : 110 = (b + s + 3) * 5)
                        (h2 : 85 = (b - s + 2) * 6) : s = 3.4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l1799_179986


namespace NUMINAMATH_GPT_number_of_rabbits_l1799_179929

-- Defining the problem conditions
variables (x y : ℕ)
axiom heads_condition : x + y = 40
axiom legs_condition : 4 * x = 10 * 2 * y - 8

--  Prove the number of rabbits is 33
theorem number_of_rabbits : x = 33 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rabbits_l1799_179929


namespace NUMINAMATH_GPT_friends_division_l1799_179910

def num_ways_to_divide (total_friends teams : ℕ) : ℕ :=
  4^8 - (Nat.choose 4 1) * 3^8 + (Nat.choose 4 2) * 2^8 - (Nat.choose 4 3) * 1^8

theorem friends_division (total_friends teams : ℕ) (h_friends : total_friends = 8) (h_teams : teams = 4) :
  num_ways_to_divide total_friends teams = 39824 := by
  sorry

end NUMINAMATH_GPT_friends_division_l1799_179910


namespace NUMINAMATH_GPT_hyperbola_condition_l1799_179959

theorem hyperbola_condition (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) → m < 0 ∨ m > 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l1799_179959


namespace NUMINAMATH_GPT_problem_statement_l1799_179920

theorem problem_statement : 6 * (3/2 + 2/3) = 13 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1799_179920


namespace NUMINAMATH_GPT_gcd_of_256_180_600_l1799_179923

theorem gcd_of_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 12 :=
by
  -- The proof would be placed here
  sorry

end NUMINAMATH_GPT_gcd_of_256_180_600_l1799_179923


namespace NUMINAMATH_GPT_JackEmails_l1799_179947

theorem JackEmails (E : ℕ) (h1 : 10 = E + 7) : E = 3 :=
by
  sorry

end NUMINAMATH_GPT_JackEmails_l1799_179947


namespace NUMINAMATH_GPT_cos_sum_condition_l1799_179926

theorem cos_sum_condition {x y z : ℝ} (h1 : Real.cos x + Real.cos y + Real.cos z = 1) (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_cos_sum_condition_l1799_179926


namespace NUMINAMATH_GPT_inches_repaired_before_today_l1799_179979

-- Definitions and assumptions based on the conditions.
def total_inches_repaired : ℕ := 4938
def inches_repaired_today : ℕ := 805

-- Target statement that needs to be proven.
theorem inches_repaired_before_today : total_inches_repaired - inches_repaired_today = 4133 :=
by
  sorry

end NUMINAMATH_GPT_inches_repaired_before_today_l1799_179979


namespace NUMINAMATH_GPT_manuscript_pages_l1799_179913

theorem manuscript_pages (P : ℕ) (rate_first : ℕ) (rate_revision : ℕ) 
  (revised_once_pages : ℕ) (revised_twice_pages : ℕ) (total_cost : ℕ) :
  rate_first = 6 →
  rate_revision = 4 →
  revised_once_pages = 35 →
  revised_twice_pages = 15 →
  total_cost = 860 →
  6 * (P - 35 - 15) + 10 * 35 + 14 * 15 = total_cost →
  P = 100 :=
by
  intros h_first h_revision h_once h_twice h_cost h_eq
  sorry

end NUMINAMATH_GPT_manuscript_pages_l1799_179913


namespace NUMINAMATH_GPT_fg_square_diff_l1799_179955

open Real

noncomputable def f (x: ℝ) : ℝ := sorry
noncomputable def g (x: ℝ) : ℝ := sorry

axiom h1 (x: ℝ) (hx : -π / 2 < x ∧ x < π / 2) : f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x))
axiom h2 : ∀ x, f (-x) = -f x
axiom h3 : ∀ x, g (-x) = g x

theorem fg_square_diff (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end NUMINAMATH_GPT_fg_square_diff_l1799_179955


namespace NUMINAMATH_GPT_minimum_w_coincide_after_translation_l1799_179988

noncomputable def period_of_cosine (w : ℝ) : ℝ := (2 * Real.pi) / w

theorem minimum_w_coincide_after_translation
  (w : ℝ) (h_w_pos : 0 < w) :
  period_of_cosine w = (4 * Real.pi) / 3 → w = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_w_coincide_after_translation_l1799_179988


namespace NUMINAMATH_GPT_proof_f_of_2_add_g_of_3_l1799_179902

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x^2 + 2 * x - 1

theorem proof_f_of_2_add_g_of_3 : f (2 + g 3) = 44 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_of_2_add_g_of_3_l1799_179902


namespace NUMINAMATH_GPT_simplify_expression_l1799_179956

theorem simplify_expression : (2^8 + 4^5) * ((1^3 - (-1)^3)^8) = 327680 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1799_179956


namespace NUMINAMATH_GPT_savings_calculation_l1799_179930

-- Definitions of the given conditions
def window_price : ℕ := 100
def free_window_offer (purchased : ℕ) : ℕ := purchased / 4

-- Number of windows needed
def dave_needs : ℕ := 7
def doug_needs : ℕ := 8

-- Calculations based on the conditions
def individual_costs : ℕ :=
  (dave_needs - free_window_offer dave_needs) * window_price +
  (doug_needs - free_window_offer doug_needs) * window_price

def together_costs : ℕ :=
  let total_needs := dave_needs + doug_needs
  (total_needs - free_window_offer total_needs) * window_price

def savings : ℕ := individual_costs - together_costs

-- Proof statement
theorem savings_calculation : savings = 100 := by
  sorry

end NUMINAMATH_GPT_savings_calculation_l1799_179930


namespace NUMINAMATH_GPT_problem_statement_l1799_179904

variables {R : Type*} [LinearOrderedField R]

-- Definitions of f and its derivatives
variable (f : R → R)
variable (f' : R → R) 
variable (f'' : R → R)

-- Conditions given in the math problem
axiom decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2
axiom derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x

-- Lean 4 statement for the proof problem
theorem problem_statement (decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2)
    (derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x) :
    ∀ x : R, f x > 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1799_179904


namespace NUMINAMATH_GPT_value_of_c_infinite_solutions_l1799_179939

theorem value_of_c_infinite_solutions (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 18 * y + 15) ↔ (c = 3) :=
by
  sorry

end NUMINAMATH_GPT_value_of_c_infinite_solutions_l1799_179939


namespace NUMINAMATH_GPT_min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l1799_179973

section
variables {a x : ℝ}

/-- Define the function f(x) = ax^3 - 2x^2 + x + c where c = 1 -/
def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + x + 1

/-- Proposition 1: Minimum value of f when a = 1 and f passes through (0,1) is 1 -/
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 1) := 
by {
  -- Sorry for the full proof
  sorry
}

/-- Proposition 2: If f has no extremum points, then a ≥ 4/3 -/
theorem no_extrema_implies_a_ge_four_thirds (h : ∀ x : ℝ, 3 * a * x^2 - 4 * x + 1 ≠ 0) : 
  a ≥ (4 / 3) :=
by {
  -- Sorry for the full proof
  sorry
}

end

end NUMINAMATH_GPT_min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l1799_179973


namespace NUMINAMATH_GPT_arithmetic_sum_l1799_179944

variables {a d : ℝ}

theorem arithmetic_sum (h : 15 * a + 105 * d = 90) : 2 * a + 14 * d = 12 :=
sorry

end NUMINAMATH_GPT_arithmetic_sum_l1799_179944


namespace NUMINAMATH_GPT_pedestrian_travel_time_l1799_179949

noncomputable def travel_time (d : ℝ) (x y : ℝ) : ℝ :=
  d / x

theorem pedestrian_travel_time
  (d : ℝ)
  (x y : ℝ)
  (h1 : d = 1)
  (h2 : 3 * x = 1 - x - y)
  (h3 : (1 / 2) * (x + y) = 1 - x - y)
  : travel_time d x y = 9 := 
sorry

end NUMINAMATH_GPT_pedestrian_travel_time_l1799_179949
