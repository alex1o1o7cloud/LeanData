import Mathlib

namespace NUMINAMATH_GPT_youngest_age_is_29_l1038_103819

-- Define that the ages form an arithmetic sequence
def arithmetic_sequence (a1 a2 a3 a4 : ℕ) : Prop :=
  ∃ (d : ℕ), a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d

-- Define the problem statement
theorem youngest_age_is_29 (a1 a2 a3 a4 : ℕ) (h_seq : arithmetic_sequence a1 a2 a3 a4) (h_oldest : a4 = 50) (h_sum : a1 + a2 + a3 + a4 = 158) :
  a1 = 29 :=
by
  sorry

end NUMINAMATH_GPT_youngest_age_is_29_l1038_103819


namespace NUMINAMATH_GPT_base_of_first_term_l1038_103897

-- Define the necessary conditions
def equation (x s : ℝ) : Prop :=
  x^16 * 25^s = 5 * 10^16

-- The proof goal
theorem base_of_first_term (x s : ℝ) (h : equation x s) : x = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_base_of_first_term_l1038_103897


namespace NUMINAMATH_GPT_original_average_weight_l1038_103849

theorem original_average_weight 
  (W : ℝ)  -- Define W as the original average weight
  (h1 : 0 < W)  -- Define conditions
  (w_new1 : ℝ := 110)
  (w_new2 : ℝ := 60)
  (num_initial_players : ℝ := 7)
  (num_total_players : ℝ := 9)
  (new_average_weight : ℝ := 92)
  (total_weight_initial := num_initial_players * W)
  (total_weight_additional := w_new1 + w_new2)
  (total_weight_total := new_average_weight * num_total_players) : 
  total_weight_initial + total_weight_additional = total_weight_total → W = 94 :=
by 
  sorry

end NUMINAMATH_GPT_original_average_weight_l1038_103849


namespace NUMINAMATH_GPT_number_of_feet_on_branches_l1038_103815

def number_of_birds : ℕ := 46
def feet_per_bird : ℕ := 2

theorem number_of_feet_on_branches : number_of_birds * feet_per_bird = 92 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_feet_on_branches_l1038_103815


namespace NUMINAMATH_GPT_problem_inequality_l1038_103800

theorem problem_inequality (a : ℝ) (h_pos : 0 < a) : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a :=
by 
  sorry

end NUMINAMATH_GPT_problem_inequality_l1038_103800


namespace NUMINAMATH_GPT_integer_average_problem_l1038_103872

theorem integer_average_problem (a b c d : ℤ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
(h_max : max a (max b (max c d)) = 90) (h_min : min a (min b (min c d)) = 29) : 
(a + b + c + d) / 4 = 45 := 
sorry

end NUMINAMATH_GPT_integer_average_problem_l1038_103872


namespace NUMINAMATH_GPT_trader_gain_percentage_is_25_l1038_103852

noncomputable def trader_gain_percentage (C : ℝ) : ℝ :=
  ((22 * C) / (88 * C)) * 100

theorem trader_gain_percentage_is_25 (C : ℝ) (h : C ≠ 0) : trader_gain_percentage C = 25 := by
  unfold trader_gain_percentage
  field_simp [h]
  norm_num
  sorry

end NUMINAMATH_GPT_trader_gain_percentage_is_25_l1038_103852


namespace NUMINAMATH_GPT_find_length_d_l1038_103884

theorem find_length_d :
  ∀ (A B C P: Type) (AB AC BC : ℝ) (d : ℝ),
    AB = 425 ∧ BC = 450 ∧ AC = 510 ∧
    (∃ (JG FI HE : ℝ), JG = FI ∧ FI = HE ∧ JG = d ∧ 
      (d / BC + d / AC + d / AB = 2)) 
    → d = 306 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_length_d_l1038_103884


namespace NUMINAMATH_GPT_person_b_days_work_alone_l1038_103873

theorem person_b_days_work_alone (B : ℕ) (h1 : (1 : ℚ) / 40 + 1 / B = 1 / 24) : B = 60 := 
by
  sorry

end NUMINAMATH_GPT_person_b_days_work_alone_l1038_103873


namespace NUMINAMATH_GPT_bacteria_growth_rate_l1038_103836

theorem bacteria_growth_rate (B G : ℝ) (h : B * G^16 = 2 * B * G^15) : G = 2 :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l1038_103836


namespace NUMINAMATH_GPT_number_of_women_l1038_103848

theorem number_of_women (n_men n_women n_dances men_partners women_partners : ℕ) 
  (h_men_partners : men_partners = 4)
  (h_women_partners : women_partners = 3)
  (h_n_men : n_men = 15)
  (h_total_dances : n_dances = n_men * men_partners)
  (h_women_calc : n_women = n_dances / women_partners) :
  n_women = 20 :=
sorry

end NUMINAMATH_GPT_number_of_women_l1038_103848


namespace NUMINAMATH_GPT_f_g_2_eq_256_l1038_103801

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem f_g_2_eq_256 : f (g 2) = 256 := by
  sorry

end NUMINAMATH_GPT_f_g_2_eq_256_l1038_103801


namespace NUMINAMATH_GPT_find_p_probability_of_match_ending_after_4_games_l1038_103875

variables (p : ℚ)

-- Conditions translated to Lean definitions
def probability_first_game_win : ℚ := 1 / 2

def probability_consecutive_games_win : ℚ := 5 / 16

-- Definitions based on conditions
def prob_second_game_win_if_won_first : ℚ := (1 + p) / 2

def prob_winning_consecutive_games (prob_first_game : ℚ) (prob_second_game_if_won_first : ℚ) : ℚ :=
prob_first_game * prob_second_game_if_won_first

-- Main Theorem Statements to be proved
theorem find_p 
    (h_eq : prob_winning_consecutive_games probability_first_game_win (prob_second_game_win_if_won_first p) = probability_consecutive_games_win) :
    p = 1 / 4 :=
sorry

-- Given p = 1/4, probabilities for each scenario the match ends after 4 games
def prob_scenario1 : ℚ := (1 / 2) * ((1 + 1/4) / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2)
def prob_scenario2 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2)
def prob_scenario3 : ℚ := (1 / 2) * ((1 - 1/4) / 2) * ((1 + 1/4) / 2) * ((1 + 1/4) / 2)

def total_probability_ending_in_4_games : ℚ :=
2 * (prob_scenario1 + prob_scenario2 + prob_scenario3)

theorem probability_of_match_ending_after_4_games (hp : p = 1 / 4) :
    total_probability_ending_in_4_games = 165 / 512 :=
sorry

end NUMINAMATH_GPT_find_p_probability_of_match_ending_after_4_games_l1038_103875


namespace NUMINAMATH_GPT_determine_m_value_l1038_103878

theorem determine_m_value (m : ℤ) (A : Set ℤ) : 
  A = {1, m + 2, m^2 + 4} → 5 ∈ A → m = 3 ∨ m = 1 := 
by
  sorry

end NUMINAMATH_GPT_determine_m_value_l1038_103878


namespace NUMINAMATH_GPT_simplify_fraction_l1038_103827

theorem simplify_fraction :
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1038_103827


namespace NUMINAMATH_GPT_spent_on_basil_seeds_l1038_103890

-- Define the variables and conditions
variables (S cost_soil num_plants price_per_plant net_profit total_revenue total_expenses : ℝ)
variables (h1 : cost_soil = 8)
variables (h2 : num_plants = 20)
variables (h3 : price_per_plant = 5)
variables (h4 : net_profit = 90)

-- Definition of total revenue as the multiplication of number of plants and price per plant
def revenue_eq : Prop := total_revenue = num_plants * price_per_plant

-- Definition of total expenses as the sum of soil cost and cost of basil seeds
def expenses_eq : Prop := total_expenses = cost_soil + S

-- Definition of net profit
def profit_eq : Prop := net_profit = total_revenue - total_expenses

-- The theorem to prove
theorem spent_on_basil_seeds : S = 2 :=
by
  -- Since we define variables and conditions as inputs,
  -- the proof itself is omitted as per instructions
  sorry

end NUMINAMATH_GPT_spent_on_basil_seeds_l1038_103890


namespace NUMINAMATH_GPT_tangent_lines_parallel_to_line_l1038_103817

theorem tangent_lines_parallel_to_line (a : ℝ) (b : ℝ)
  (h1 : b = a^3 + a - 2)
  (h2 : 3 * a^2 + 1 = 4) :
  (b = 4 * a - 4 ∨ b = 4 * a) :=
sorry

end NUMINAMATH_GPT_tangent_lines_parallel_to_line_l1038_103817


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1038_103851

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (ha : 0 < a) : ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) := sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1038_103851


namespace NUMINAMATH_GPT_unique_triangle_with_consecutive_sides_and_angle_condition_l1038_103847

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end NUMINAMATH_GPT_unique_triangle_with_consecutive_sides_and_angle_condition_l1038_103847


namespace NUMINAMATH_GPT_correct_sum_of_integers_l1038_103857

theorem correct_sum_of_integers
  (x y : ℕ)
  (h1 : x - y = 5)
  (h2 : x * y = 84) :
  x + y = 19 :=
sorry

end NUMINAMATH_GPT_correct_sum_of_integers_l1038_103857


namespace NUMINAMATH_GPT_number_of_integers_satisfying_l1038_103840

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end NUMINAMATH_GPT_number_of_integers_satisfying_l1038_103840


namespace NUMINAMATH_GPT_total_teeth_cleaned_l1038_103845

/-
  Given:
   1. Dogs have 42 teeth.
   2. Cats have 30 teeth.
   3. Pigs have 28 teeth.
   4. There are 5 dogs.
   5. There are 10 cats.
   6. There are 7 pigs.
  Prove: The total number of teeth Vann will clean today is 706.
-/

theorem total_teeth_cleaned :
  let dogs: Nat := 5
  let cats: Nat := 10
  let pigs: Nat := 7
  let dog_teeth: Nat := 42
  let cat_teeth: Nat := 30
  let pig_teeth: Nat := 28
  (dogs * dog_teeth) + (cats * cat_teeth) + (pigs * pig_teeth) = 706 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_teeth_cleaned_l1038_103845


namespace NUMINAMATH_GPT_part_a_region_part_b_region_part_c_region_l1038_103842

-- Definitions for Part (a)
def surface1a (x y z : ℝ) := 2 * y = x ^ 2 + z ^ 2
def surface2a (x y z : ℝ) := x ^ 2 + z ^ 2 = 1
def region_a (x y z : ℝ) := surface1a x y z ∧ surface2a x y z

-- Definitions for Part (b)
def surface1b (x y z : ℝ) := z = 0
def surface2b (x y z : ℝ) := y + z = 2
def surface3b (x y z : ℝ) := y = x ^ 2
def region_b (x y z : ℝ) := surface1b x y z ∧ surface2b x y z ∧ surface3b x y z

-- Definitions for Part (c)
def surface1c (x y z : ℝ) := z = 6 - x ^ 2 - y ^ 2
def surface2c (x y z : ℝ) := x ^ 2 + y ^ 2 = z ^ 2
def region_c (x y z : ℝ) := surface1c x y z ∧ surface2c x y z

-- The formal theorem statements
theorem part_a_region : ∃x y z : ℝ, region_a x y z := by
  sorry

theorem part_b_region : ∃x y z : ℝ, region_b x y z := by
  sorry

theorem part_c_region : ∃x y z : ℝ, region_c x y z := by
  sorry

end NUMINAMATH_GPT_part_a_region_part_b_region_part_c_region_l1038_103842


namespace NUMINAMATH_GPT_problem_eval_at_x_eq_3_l1038_103853

theorem problem_eval_at_x_eq_3 : ∀ x : ℕ, x = 3 → (x^x)^(x^x) = 27^27 :=
by
  intros x hx
  rw [hx]
  sorry

end NUMINAMATH_GPT_problem_eval_at_x_eq_3_l1038_103853


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1038_103860

section Problem

-- Initial conditions
variable (a : ℕ → ℝ) (t m : ℝ)
def a_1 : ℝ := 3
def a_n (n : ℕ) (h : 2 ≤ n) : ℝ := 2 * a (n - 1) + (t + 1) * 2^n + 3 * m + t

-- Problem 1:
theorem problem_1 (h : t = 0) (h' : m = 0) :
  ∃ d, ∀ n, 2 ≤ n → (a n / 2^n) = (a (n - 1) / 2^(n-1)) + d := sorry

-- Problem 2:
theorem problem_2 (h : t = -1) (h' : m = 4/3) :
  ∃ r, ∀ n, 2 ≤ n → a n + 3 = r * (a (n - 1) + 3) := sorry

-- Problem 3:
theorem problem_3 (h : t = 0) (h' : m = 1) :
  (∀ n, 1 ≤ n → a n = (n + 2) * 2^n - 3) ∧
  (∃ S : ℕ → ℝ, ∀ n, S n = (n + 1) * 2^(n + 1) - 2 - 3 * n) := sorry

end Problem

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1038_103860


namespace NUMINAMATH_GPT_vanya_speed_l1038_103833

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_GPT_vanya_speed_l1038_103833


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1038_103887

theorem hyperbola_eccentricity : 
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt 5 / 2 := 
by
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1038_103887


namespace NUMINAMATH_GPT_cyclic_quad_angles_l1038_103828

theorem cyclic_quad_angles (A B C D : ℝ) (x : ℝ)
  (h_ratio : A = 5 * x ∧ B = 6 * x ∧ C = 4 * x)
  (h_cyclic : A + D = 180 ∧ B + C = 180):
  (B = 108) ∧ (C = 72) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quad_angles_l1038_103828


namespace NUMINAMATH_GPT_perimeter_of_region_l1038_103854

-- Define the conditions as Lean definitions
def area_of_region (a : ℝ) := a = 400
def number_of_squares (n : ℕ) := n = 8
def arrangement := "2x4 rectangle"

-- Define the statement we need to prove
theorem perimeter_of_region (a : ℝ) (n : ℕ) (s : ℝ) 
  (h_area_region : area_of_region a) 
  (h_number_of_squares : number_of_squares n) 
  (h_arrangement : arrangement = "2x4 rectangle")
  (h_area_one_square : a / n = s^2) :
  4 * 10 * (s) = 80 * 2^(1/2)  :=
by sorry

end NUMINAMATH_GPT_perimeter_of_region_l1038_103854


namespace NUMINAMATH_GPT_packs_used_after_6_weeks_l1038_103821

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_packs_used_after_6_weeks_l1038_103821


namespace NUMINAMATH_GPT_fraction_simplifiable_by_7_l1038_103809

theorem fraction_simplifiable_by_7 (a b c : ℤ) (h : (100 * a + 10 * b + c) % 7 = 0) : 
  ((10 * b + c + 16 * a) % 7 = 0) ∧ ((10 * b + c - 61 * a) % 7 = 0) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplifiable_by_7_l1038_103809


namespace NUMINAMATH_GPT_max_min_values_of_f_l1038_103885

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_min_values_of_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∃ x : ℝ, f x = 2) ∧
  (∀ x : ℝ, -2 ≤ f x) ∧ (∃ x : ℝ, f x = -2) :=
by 
  sorry

end NUMINAMATH_GPT_max_min_values_of_f_l1038_103885


namespace NUMINAMATH_GPT_relationship_among_mnr_l1038_103826

-- Definitions of the conditions
variables {a b c : ℝ}
variables (m n r : ℝ)

-- Assumption given by the conditions
def conditions (a b c : ℝ) := 0 < a ∧ a < b ∧ b < 1 ∧ 1 < c
def log_equations (a b c m n : ℝ) := m = Real.log c / Real.log a ∧ n = Real.log c / Real.log b
def r_definition (a c r : ℝ) := r = a^c

-- Statement: If the conditions are satisfied, then the relationship holds
theorem relationship_among_mnr (a b c m n r : ℝ)
  (h1 : conditions a b c)
  (h2 : log_equations a b c m n)
  (h3 : r_definition a c r) :
  n < m ∧ m < r := by
  sorry

end NUMINAMATH_GPT_relationship_among_mnr_l1038_103826


namespace NUMINAMATH_GPT_quadratic_nonneg_iff_m_in_range_l1038_103839

theorem quadratic_nonneg_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 * m + 5 ≥ 0) ↔ (-2 : ℝ) ≤ m ∧ m ≤ 10 :=
by sorry

end NUMINAMATH_GPT_quadratic_nonneg_iff_m_in_range_l1038_103839


namespace NUMINAMATH_GPT_oranges_purchase_cost_l1038_103877

/-- 
Oranges are sold at a rate of $3$ per three pounds.
If a customer buys 18 pounds and receives a discount of $5\%$ for buying more than 15 pounds,
prove that the total amount the customer pays is $17.10.
-/
theorem oranges_purchase_cost (rate : ℕ) (base_weight : ℕ) (discount_rate : ℚ)
  (total_weight : ℕ) (discount_threshold : ℕ) (final_cost : ℚ) :
  rate = 3 → base_weight = 3 → discount_rate = 0.05 → 
  total_weight = 18 → discount_threshold = 15 → final_cost = 17.10 := by
  sorry

end NUMINAMATH_GPT_oranges_purchase_cost_l1038_103877


namespace NUMINAMATH_GPT_find_x2_y2_and_xy_l1038_103820

-- Problem statement
theorem find_x2_y2_and_xy (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_find_x2_y2_and_xy_l1038_103820


namespace NUMINAMATH_GPT_rectangle_area_l1038_103841

theorem rectangle_area (w l d : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : d = 10)
  (h3 : d^2 = w^2 + l^2) : 
  l * w = 40 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1038_103841


namespace NUMINAMATH_GPT_total_attendance_l1038_103822

theorem total_attendance (A C : ℕ) (adult_ticket_price child_ticket_price total_revenue : ℕ) 
(h1 : adult_ticket_price = 11) (h2 : child_ticket_price = 10) (h3 : total_revenue = 246) 
(h4 : C = 7) (h5 : adult_ticket_price * A + child_ticket_price * C = total_revenue) : 
A + C = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_attendance_l1038_103822


namespace NUMINAMATH_GPT_man_speed_is_4_kmph_l1038_103835

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := train_length / time_to_pass_seconds
  let relative_speed_kmph := relative_speed_mps * 3600 / 1000
  relative_speed_kmph - train_speed_kmph

theorem man_speed_is_4_kmph : speed_of_man 140 50 9.332586726395222 = 4 := by
  sorry

end NUMINAMATH_GPT_man_speed_is_4_kmph_l1038_103835


namespace NUMINAMATH_GPT_base_k_representation_l1038_103807

theorem base_k_representation (k : ℕ) (hk : k > 0) (hk_exp : 7 / 51 = (2 * k + 3 : ℚ) / (k ^ 2 - 1 : ℚ)) : k = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_k_representation_l1038_103807


namespace NUMINAMATH_GPT_dogsled_race_time_difference_l1038_103891

theorem dogsled_race_time_difference :
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  T_W - T_A = 3 :=
by
  let D := 300  -- Distance in miles
  let V_W := 20  -- Team W's average speed in mph
  let V_A := 25  -- Team A's average speed in mph
  let T_W := D / V_W  -- Time taken by Team W
  let T_A := D / V_A  -- Time taken by Team A
  sorry

end NUMINAMATH_GPT_dogsled_race_time_difference_l1038_103891


namespace NUMINAMATH_GPT_percentage_decrease_l1038_103838

theorem percentage_decrease (P : ℝ) (new_price : ℝ) (x : ℝ) (h1 : new_price = 320) (h2 : P = 421.05263157894734) : x = 24 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l1038_103838


namespace NUMINAMATH_GPT_no_roots_of_form_one_over_n_l1038_103810

theorem no_roots_of_form_one_over_n (a b c : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_c : c % 2 = 1) :
  ∀ n : ℕ, ¬(a * (1 / (n:ℚ))^2 + b * (1 / (n:ℚ)) + c = 0) := by
  sorry

end NUMINAMATH_GPT_no_roots_of_form_one_over_n_l1038_103810


namespace NUMINAMATH_GPT_angle_sum_90_l1038_103846

theorem angle_sum_90 (A B : ℝ) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) : A + B = Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_angle_sum_90_l1038_103846


namespace NUMINAMATH_GPT_sufficient_cond_l1038_103892

theorem sufficient_cond (x : ℝ) (h : 1/x > 2) : x < 1/2 := 
by {
  sorry 
}

end NUMINAMATH_GPT_sufficient_cond_l1038_103892


namespace NUMINAMATH_GPT_toy_store_revenue_fraction_l1038_103830

theorem toy_store_revenue_fraction (N D J : ℝ) 
  (h1 : J = N / 3) 
  (h2 : D = 3.75 * (N + J) / 2) : 
  (N / D) = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_toy_store_revenue_fraction_l1038_103830


namespace NUMINAMATH_GPT_find_number_l1038_103881

theorem find_number (x : ℝ) (h : 2 = 0.04 * x) : x = 50 := 
sorry

end NUMINAMATH_GPT_find_number_l1038_103881


namespace NUMINAMATH_GPT_find_k_l1038_103868

theorem find_k (x : ℝ) (a h k : ℝ) (h1 : 9 * x^2 - 12 * x = a * (x - h)^2 + k) : k = -4 := by
  sorry

end NUMINAMATH_GPT_find_k_l1038_103868


namespace NUMINAMATH_GPT_expected_value_of_winnings_l1038_103837

noncomputable def winnings (n : ℕ) : ℕ := 2 * n - 1

theorem expected_value_of_winnings : 
  (1 / 6 : ℚ) * ((winnings 1) + (winnings 2) + (winnings 3) + (winnings 4) + (winnings 5) + (winnings 6)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_winnings_l1038_103837


namespace NUMINAMATH_GPT_exist_n_l1038_103813

theorem exist_n : ∃ n : ℕ, n > 1 ∧ ¬(Nat.Prime n) ∧ ∀ a : ℤ, (a^n - a) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_exist_n_l1038_103813


namespace NUMINAMATH_GPT_probability_red_blue_yellow_l1038_103843

-- Define the probabilities for white, green, and black marbles
def p_white : ℚ := 1/4
def p_green : ℚ := 1/6
def p_black : ℚ := 1/8

-- Define the problem: calculating the probability of drawing a red, blue, or yellow marble
theorem probability_red_blue_yellow : 
  p_white = 1/4 → p_green = 1/6 → p_black = 1/8 →
  (1 - (p_white + p_green + p_black)) = 11/24 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_probability_red_blue_yellow_l1038_103843


namespace NUMINAMATH_GPT_find_a_value_l1038_103874

def quadratic_vertex_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = 2) → (y = 5) →
  a * (x - 2)^2 + 5 = y

def quadratic_passing_point_condition (a : ℚ) : Prop :=
  ∀ x y : ℚ,
  (x = -1) → (y = -20) →
  a * (x - 2)^2 + 5 = y

theorem find_a_value : ∃ a : ℚ, quadratic_vertex_condition a ∧ quadratic_passing_point_condition a ∧ a = (-25)/9 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_value_l1038_103874


namespace NUMINAMATH_GPT_remainder_when_divided_l1038_103898

theorem remainder_when_divided (N : ℕ) (hN : N = 123456789012) : 
  (N % 252) = 228 := by
  -- The following conditions:
  have h1 : N % 4 = 0 := by sorry
  have h2 : N % 9 = 3 := by sorry
  have h3 : N % 7 = 4 := by sorry
  -- Proof that the remainder is 228 when divided by 252.
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l1038_103898


namespace NUMINAMATH_GPT_inequality_proof_l1038_103814

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1038_103814


namespace NUMINAMATH_GPT_range_of_2a_minus_b_l1038_103876

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) :
  2 < 2 * a - b ∧ 2 * a - b < 5 := 
sorry

end NUMINAMATH_GPT_range_of_2a_minus_b_l1038_103876


namespace NUMINAMATH_GPT_volume_of_hemisphere_l1038_103823

theorem volume_of_hemisphere (d : ℝ) (h : d = 10) : 
  let r := d / 2
  let V := (2 / 3) * π * r^3
  V = 250 / 3 * π := by
sorry

end NUMINAMATH_GPT_volume_of_hemisphere_l1038_103823


namespace NUMINAMATH_GPT_power_multiplication_l1038_103806

theorem power_multiplication :
  3^5 * 6^5 = 1889568 :=
by
  sorry

end NUMINAMATH_GPT_power_multiplication_l1038_103806


namespace NUMINAMATH_GPT_probability_diagonals_intersect_l1038_103861

theorem probability_diagonals_intersect (n : ℕ) : 
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  probability = n * (2 * n - 1) / (3 * (2 * n ^ 2 - n - 2)) :=
by
  let V := 2 * n + 1
  let total_diagonals := (V * (V - 3)) / 2
  let choose_pairs_diagonals := (total_diagonals * (total_diagonals - 1)) / 2
  let choose_four_vertices := (V * (V - 1) * (V - 2) * (V - 3)) / 24
  let probability := choose_four_vertices * 2 / (3 * choose_pairs_diagonals)
  sorry

end NUMINAMATH_GPT_probability_diagonals_intersect_l1038_103861


namespace NUMINAMATH_GPT_max_distance_origin_perpendicular_bisector_l1038_103858

theorem max_distance_origin_perpendicular_bisector :
  ∀ (k m : ℝ), k ≠ 0 → 
  (|m| = Real.sqrt (1 + k^2)) → 
  ∃ (d : ℝ), d = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_origin_perpendicular_bisector_l1038_103858


namespace NUMINAMATH_GPT_complex_z_modulus_l1038_103883

open Complex

theorem complex_z_modulus (z : ℂ) (h1 : (z + 2 * I).re = z + 2 * I) (h2 : (z / (2 - I)).re = z / (2 - I)) :
  (z = 4 - 2 * I) ∧ abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_complex_z_modulus_l1038_103883


namespace NUMINAMATH_GPT_average_score_l1038_103829

variable (u v A : ℝ)
variable (h1 : v / u = 1/3)
variable (h2 : A = (u + v) / 2)

theorem average_score : A = (2/3) * u := by
  sorry

end NUMINAMATH_GPT_average_score_l1038_103829


namespace NUMINAMATH_GPT_a_n_formula_l1038_103889

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n * (n + 1) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) 
  (S_n : ℕ → ℕ)
  (hS : ∀ n, S_n n = (n + 2) / 3 * a_n n) 
  : a_n n = n * (n + 1) / 2 := sorry

end NUMINAMATH_GPT_a_n_formula_l1038_103889


namespace NUMINAMATH_GPT_initial_group_size_l1038_103832

theorem initial_group_size (W : ℝ) : 
  (∃ n : ℝ, (W + 15) / n = W / n + 2.5) → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_group_size_l1038_103832


namespace NUMINAMATH_GPT_calculate_expression_l1038_103879

def seq (k : Nat) : Nat := 2^k + 3^k

def product_seq : Nat :=
  (2 + 3) * (2^3 + 3^3) * (2^6 + 3^6) * (2^12 + 3^12) * (2^24 + 3^24)

theorem calculate_expression :
  product_seq = (3^47 - 2^47) :=
sorry

end NUMINAMATH_GPT_calculate_expression_l1038_103879


namespace NUMINAMATH_GPT_f3_is_ideal_function_l1038_103859

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + f (-x) = 0

def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

noncomputable def f3 (x : ℝ) : ℝ :=
  if x < 0 then x ^ 2 else -x ^ 2

theorem f3_is_ideal_function : is_odd_function f3 ∧ is_strictly_decreasing f3 := 
  sorry

end NUMINAMATH_GPT_f3_is_ideal_function_l1038_103859


namespace NUMINAMATH_GPT_hyperbola_equation_l1038_103895

theorem hyperbola_equation {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0)
    (hfocal : 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5)
    (hslope : b / a = 1 / 8) :
    (∀ x y : ℝ, (x^2 / 4 - y^2 = 1) ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :=
by
  -- Goals and conditions to handle proof
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1038_103895


namespace NUMINAMATH_GPT_exactly_one_valid_N_l1038_103882

def four_digit_number (N : ℕ) : Prop := 1000 ≤ N ∧ N < 10000

def condition (N x a : ℕ) : Prop := 
  N = 1000 * a + x ∧ x = N / 7

theorem exactly_one_valid_N : 
  ∃! N : ℕ, ∃ x a : ℕ, four_digit_number N ∧ condition N x a :=
sorry

end NUMINAMATH_GPT_exactly_one_valid_N_l1038_103882


namespace NUMINAMATH_GPT_exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l1038_103880

theorem exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles
    (a b c d α β γ δ: ℝ) (h_conv: a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c)
    (h_angles: α < β + γ + δ ∧ β < α + γ + δ ∧ γ < α + β + δ ∧ δ < α + β + γ) :
    ∃ (a' b' c' d' α' β' γ' δ' : ℝ),
      (a' / b' = α / β) ∧ (b' / c' = β / γ) ∧ (c' / d' = γ / δ) ∧ (d' / a' = δ / α) ∧
      (a' < b' + c' + d') ∧ (b' < a' + c' + d') ∧ (c' < a' + b' + d') ∧ (d' < a' + b' + c') ∧
      (α' < β' + γ' + δ') ∧ (β' < α' + γ' + δ') ∧ (γ' < α' + β' + δ') ∧ (δ' < α' + β' + γ') :=
  sorry

end NUMINAMATH_GPT_exists_convex_quadrilateral_with_ratio_of_sides_eq_ratio_of_angles_l1038_103880


namespace NUMINAMATH_GPT_total_campers_correct_l1038_103865

-- Definitions for the conditions
def campers_morning : ℕ := 15
def campers_afternoon : ℕ := 17

-- Define total campers, question is to prove it is indeed 32
def total_campers : ℕ := campers_morning + campers_afternoon

theorem total_campers_correct : total_campers = 32 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_campers_correct_l1038_103865


namespace NUMINAMATH_GPT_find_d_l1038_103824

noncomputable def problem_condition :=
  ∃ (v d : ℝ × ℝ) (t : ℝ) (x y : ℝ),
  (y = (5 * x - 7) / 6) ∧ 
  ((x, y) = (v.1 + t * d.1, v.2 + t * d.2)) ∧ 
  (x ≥ 4) ∧ 
  (dist (x, y) (4, 2) = t)

noncomputable def correct_answer : ℝ × ℝ := ⟨6 / 7, 5 / 7⟩

theorem find_d 
  (h : problem_condition) : 
  ∃ (d : ℝ × ℝ), d = correct_answer :=
sorry

end NUMINAMATH_GPT_find_d_l1038_103824


namespace NUMINAMATH_GPT_fair_coin_second_head_l1038_103818

theorem fair_coin_second_head (P : ℝ) 
  (fair_coin : ∀ outcome : ℝ, outcome = 0.5) :
  P = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_fair_coin_second_head_l1038_103818


namespace NUMINAMATH_GPT_solve_for_x_l1038_103816

theorem solve_for_x (x : ℝ) (h : (3 / 4) + (1 / x) = 7 / 8) : x = 8 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1038_103816


namespace NUMINAMATH_GPT_average_is_5x_minus_10_implies_x_is_50_l1038_103896

theorem average_is_5x_minus_10_implies_x_is_50 (x : ℝ) 
  (h : (1 / 3) * ((3 * x + 8) + (7 * x + 3) + (4 * x + 9)) = 5 * x - 10) : 
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_average_is_5x_minus_10_implies_x_is_50_l1038_103896


namespace NUMINAMATH_GPT_even_sum_probability_l1038_103864

theorem even_sum_probability :
  let p_even_w1 := 3 / 4
  let p_even_w2 := 1 / 2
  let p_even_w3 := 1 / 4
  let p_odd_w1 := 1 - p_even_w1
  let p_odd_w2 := 1 - p_even_w2
  let p_odd_w3 := 1 - p_even_w3
  (p_even_w1 * p_even_w2 * p_even_w3) +
  (p_odd_w1 * p_odd_w2 * p_even_w3) +
  (p_odd_w1 * p_even_w2 * p_odd_w3) +
  (p_even_w1 * p_odd_w2 * p_odd_w3) = 1 / 2 := by
    sorry

end NUMINAMATH_GPT_even_sum_probability_l1038_103864


namespace NUMINAMATH_GPT_david_chemistry_marks_l1038_103808

theorem david_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℝ)
  (average_marks: ℝ) (marks_english_val: marks_english = 72) (marks_math_val: marks_math = 45)
  (marks_physics_val: marks_physics = 72) (marks_biology_val: marks_biology = 75)
  (average_marks_val: average_marks = 68.2) : 
  ∃ marks_chemistry : ℝ, (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) / 5 = average_marks ∧ 
    marks_chemistry = 77 := 
by
  sorry

end NUMINAMATH_GPT_david_chemistry_marks_l1038_103808


namespace NUMINAMATH_GPT_find_k_l1038_103834

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 3) (h2 : m + 2 = 2 * (n + k) + 3) : k = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_k_l1038_103834


namespace NUMINAMATH_GPT_smallest_possible_students_group_l1038_103825

theorem smallest_possible_students_group 
  (students : ℕ) :
  (∀ n, 2 ≤ n ∧ n ≤ 15 → ∃ k, students = k * n) ∧
  ¬∃ k, students = k * 10 ∧ ¬∃ k, students = k * 25 ∧ ¬∃ k, students = k * 50 ∧
  ∀ m n, 1 ≤ m ∧ m ≤ 15 ∧ 1 ≤ n ∧ n ≤ 15 ∧ (students ≠ m * n) → (m = n ∨ m ≠ n)
  → students = 120 := sorry

end NUMINAMATH_GPT_smallest_possible_students_group_l1038_103825


namespace NUMINAMATH_GPT_divide_circle_into_parts_l1038_103863

theorem divide_circle_into_parts : 
    ∃ (divide : ℕ → ℕ), 
        (divide 3 = 4 ∧ divide 3 = 5 ∧ divide 3 = 6 ∧ divide 3 = 7) :=
by
  -- This illustrates that we require a proof to show that for 3 straight cuts ('n = 3'), 
  -- we can achieve 4, 5, 6, and 7 segments in different settings (circle with strategic line placements).
  sorry

end NUMINAMATH_GPT_divide_circle_into_parts_l1038_103863


namespace NUMINAMATH_GPT_find_daily_wage_of_c_l1038_103855

def dailyWagesInRatio (a b c : ℕ) : Prop :=
  4 * a = 3 * b ∧ 5 * a = 3 * c

def totalEarnings (a b c : ℕ) (total : ℕ) : Prop :=
  6 * a + 9 * b + 4 * c = total

theorem find_daily_wage_of_c (a b c : ℕ) (total : ℕ) 
  (h1 : dailyWagesInRatio a b c) 
  (h2 : totalEarnings a b c total) 
  (h3 : total = 1406) : 
  c = 95 :=
by
  -- We assume the conditions and solve the required proof.
  sorry

end NUMINAMATH_GPT_find_daily_wage_of_c_l1038_103855


namespace NUMINAMATH_GPT_min_value_am_hm_inequality_l1038_103866

theorem min_value_am_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_am_hm_inequality_l1038_103866


namespace NUMINAMATH_GPT_no_two_digit_numbers_satisfy_condition_l1038_103899

theorem no_two_digit_numbers_satisfy_condition :
  ¬ ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end NUMINAMATH_GPT_no_two_digit_numbers_satisfy_condition_l1038_103899


namespace NUMINAMATH_GPT_find_sum_of_a_b_c_l1038_103894

theorem find_sum_of_a_b_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(h4 : (a + b + c) ^ 3 - a ^ 3 - b ^ 3 - c ^ 3 = 210) : a + b + c = 11 :=
sorry

end NUMINAMATH_GPT_find_sum_of_a_b_c_l1038_103894


namespace NUMINAMATH_GPT_bridge_length_problem_l1038_103862

noncomputable def length_of_bridge (num_carriages : ℕ) (length_carriage : ℕ) (length_engine : ℕ) (speed_kmph : ℕ) (crossing_time_min : ℕ) : ℝ :=
  let total_train_length := (num_carriages + 1) * length_carriage
  let speed_mps := (speed_kmph * 1000) / 3600
  let crossing_time_secs := crossing_time_min * 60
  let total_distance := speed_mps * crossing_time_secs
  let bridge_length := total_distance - total_train_length
  bridge_length

theorem bridge_length_problem :
  length_of_bridge 24 60 60 60 5 = 3501 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_problem_l1038_103862


namespace NUMINAMATH_GPT_area_inside_Z_outside_X_l1038_103831

structure Circle :=
  (center : Real × Real)
  (radius : ℝ)

def tangent (A B : Circle) : Prop :=
  dist A.center B.center = A.radius + B.radius

theorem area_inside_Z_outside_X (X Y Z : Circle)
  (hX : X.radius = 1) 
  (hY : Y.radius = 1) 
  (hZ : Z.radius = 1)
  (tangent_XY : tangent X Y)
  (tangent_XZ : tangent X Z)
  (non_intersect_YZ : dist Z.center Y.center > Z.radius + Y.radius) :
  π - 1/2 * π = 1/2 * π := 
by
  sorry

end NUMINAMATH_GPT_area_inside_Z_outside_X_l1038_103831


namespace NUMINAMATH_GPT_gcd_max_value_l1038_103803

noncomputable def max_gcd (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 1

theorem gcd_max_value :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → gcd (13 * m + 4) (7 * m + 2) ≤ max_gcd m) ∧
              (∀ m : ℕ, m > 0 → max_gcd m ≤ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_max_value_l1038_103803


namespace NUMINAMATH_GPT_remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l1038_103844

theorem remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0:
  (7 * 12^24 + 3^24) % 11 = 0 := sorry

end NUMINAMATH_GPT_remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l1038_103844


namespace NUMINAMATH_GPT_students_average_vegetables_l1038_103811

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end NUMINAMATH_GPT_students_average_vegetables_l1038_103811


namespace NUMINAMATH_GPT_binom_12_6_eq_924_l1038_103856

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end NUMINAMATH_GPT_binom_12_6_eq_924_l1038_103856


namespace NUMINAMATH_GPT_rowing_rate_in_still_water_l1038_103804

theorem rowing_rate_in_still_water (R C : ℝ) 
  (h1 : (R + C) * 2 = 26)
  (h2 : (R - C) * 4 = 26) : 
  R = 26 / 3 :=
by
  sorry

end NUMINAMATH_GPT_rowing_rate_in_still_water_l1038_103804


namespace NUMINAMATH_GPT_greatest_product_l1038_103893

theorem greatest_product (x : ℤ) (h : x + (2024 - x) = 2024) : 
  2024 * x - x^2 ≤ 1024144 :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_l1038_103893


namespace NUMINAMATH_GPT_line_plane_parallelism_l1038_103812

variables {Point : Type} [LinearOrder Point] -- Assuming Point is a Type with some linear order.

-- Definitions for line and plane
-- These definitions need further libraries or details depending on actual Lean geometry library support
@[ext] structure Line (P : Type) := (contains : P → Prop)
@[ext] structure Plane (P : Type) := (contains : P → Prop)

variables {a b : Line Point} {α β : Plane Point} {l : Line Point}

-- Conditions (as in part a)
axiom lines_are_different : a ≠ b
axiom planes_are_different : α ≠ β
axiom planes_intersect_in_line : ∃ l, α.contains l ∧ β.contains l
axiom a_parallel_l : ∀ p : Point, a.contains p → l.contains p
axiom b_within_plane : ∀ p : Point, b.contains p → β.contains p
axiom b_parallel_alpha : ∀ p q : Point, β.contains p → β.contains q → α.contains p → α.contains q

-- Define the theorem statement
theorem line_plane_parallelism : a ≠ b ∧ α ≠ β ∧ (∃ l, α.contains l ∧ β.contains l) 
  ∧ (∀ p, a.contains p → l.contains p) 
  ∧ (∀ p, b.contains p → β.contains p) 
  ∧ (∀ p q, β.contains p → β.contains q → α.contains p → α.contains q) → a = b :=
by sorry

end NUMINAMATH_GPT_line_plane_parallelism_l1038_103812


namespace NUMINAMATH_GPT_inequality_proof_l1038_103850

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1038_103850


namespace NUMINAMATH_GPT_initial_number_of_apples_l1038_103870

-- Definitions based on the conditions
def number_of_trees : ℕ := 3
def apples_picked_per_tree : ℕ := 8
def apples_left_on_trees : ℕ := 9

-- The theorem to prove
theorem initial_number_of_apples (t: ℕ := number_of_trees) (a: ℕ := apples_picked_per_tree) (l: ℕ := apples_left_on_trees) : t * a + l = 33 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_apples_l1038_103870


namespace NUMINAMATH_GPT_matrix_calculation_l1038_103869

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![0, 2]]

def B15_minus_3B14 : Matrix (Fin 2) (Fin 2) ℝ :=
  B^15 - 3 * B^14

theorem matrix_calculation : B15_minus_3B14 = ![![0, 4], ![0, -1]] := by
  sorry

end NUMINAMATH_GPT_matrix_calculation_l1038_103869


namespace NUMINAMATH_GPT_sum_mod_20_l1038_103886

/-- Define the elements that are summed. -/
def elements : List ℤ := [82, 83, 84, 85, 86, 87, 88, 89]

/-- The problem statement to prove. -/
theorem sum_mod_20 : (elements.sum % 20) = 15 := by
  sorry

end NUMINAMATH_GPT_sum_mod_20_l1038_103886


namespace NUMINAMATH_GPT_ax2_x_plus_1_positive_l1038_103805

theorem ax2_x_plus_1_positive (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ (a > 1/4) :=
by {
  sorry
}

end NUMINAMATH_GPT_ax2_x_plus_1_positive_l1038_103805


namespace NUMINAMATH_GPT_find_a_and_c_range_of_m_l1038_103802

theorem find_a_and_c (a c : ℝ) 
  (h : ∀ x, 1 < x ∧ x < 3 ↔ ax^2 + x + c > 0) 
  : a = -1/4 ∧ c = -3/4 := 
sorry

theorem range_of_m (m : ℝ) 
  (h : ∀ x, (-1/4)*x^2 + 2*x - 3 > 0 → x + m > 0) 
  : m ≥ -2 :=
sorry

end NUMINAMATH_GPT_find_a_and_c_range_of_m_l1038_103802


namespace NUMINAMATH_GPT_min_y_value_l1038_103867

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16 * x + 50 * y + 64) : y ≥ 0 :=
sorry

end NUMINAMATH_GPT_min_y_value_l1038_103867


namespace NUMINAMATH_GPT_toy_factory_days_per_week_l1038_103871

theorem toy_factory_days_per_week (toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : toys_per_week = 4560) (h₂ : toys_per_day = 1140) : toys_per_week / toys_per_day = 4 := 
by {
  -- Proof to be provided
  sorry
}

end NUMINAMATH_GPT_toy_factory_days_per_week_l1038_103871


namespace NUMINAMATH_GPT_maximum_marks_l1038_103888

theorem maximum_marks (M : ℝ) :
  (0.45 * M = 80) → (M = 180) :=
by
  sorry

end NUMINAMATH_GPT_maximum_marks_l1038_103888
