import Mathlib

namespace NUMINAMATH_GPT_parallel_lines_l569_56983

theorem parallel_lines (a : ℝ) : (2 * a = a * (a + 4)) → a = -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_l569_56983


namespace NUMINAMATH_GPT_initial_weight_l569_56939

theorem initial_weight (W : ℝ) (current_weight : ℝ) (future_weight : ℝ) (months : ℝ) (additional_months : ℝ) 
  (constant_rate : Prop) :
  current_weight = 198 →
  future_weight = 170 →
  months = 3 →
  additional_months = 3.5 →
  constant_rate →
  W = 222 :=
by
  intros h_current_weight h_future_weight h_months h_additional_months h_constant_rate
  -- proof would go here
  sorry

end NUMINAMATH_GPT_initial_weight_l569_56939


namespace NUMINAMATH_GPT_lee_sold_action_figures_l569_56942

-- Defining variables and conditions based on the problem
def sneaker_cost : ℕ := 90
def saved_money : ℕ := 15
def price_per_action_figure : ℕ := 10
def remaining_money : ℕ := 25

-- Theorem statement asserting that Lee sold 10 action figures
theorem lee_sold_action_figures : 
  (sneaker_cost - saved_money + remaining_money) / price_per_action_figure = 10  :=
by
  sorry

end NUMINAMATH_GPT_lee_sold_action_figures_l569_56942


namespace NUMINAMATH_GPT_evaluate_b3_l569_56968

variable (b1 q : ℤ)
variable (b1_cond : b1 = 5 ∨ b1 = -5)
variable (q_cond : q = 3 ∨ q = -3)
def b3 : ℤ := b1 * q^2

theorem evaluate_b3 (h : b1^2 * (1 + q^2 + q^4) = 2275) : b3 = 45 ∨ b3 = -45 :=
by sorry

end NUMINAMATH_GPT_evaluate_b3_l569_56968


namespace NUMINAMATH_GPT_rick_savings_ratio_proof_l569_56902

-- Define the conditions
def erika_savings : ℤ := 155
def cost_of_gift : ℤ := 250
def cost_of_cake : ℤ := 25
def amount_left : ℤ := 5

-- Define the total amount they have together
def total_amount : ℤ := cost_of_gift + cost_of_cake - amount_left

-- Define Rick's savings based on the conditions
def rick_savings : ℤ := total_amount - erika_savings

-- Define the ratio of Rick's savings to the cost of the gift
def rick_gift_ratio : ℚ := rick_savings / cost_of_gift

-- Prove the ratio is 23/50
theorem rick_savings_ratio_proof : rick_gift_ratio = 23 / 50 :=
  by
    have h1 : total_amount = 270 := by sorry
    have h2 : rick_savings = 115 := by sorry
    have h3 : rick_gift_ratio = 23 / 50 := by sorry
    exact h3

end NUMINAMATH_GPT_rick_savings_ratio_proof_l569_56902


namespace NUMINAMATH_GPT_maximum_tangency_circles_l569_56988

/-- Points \( P_1, P_2, \ldots, P_n \) are in the plane
    Real numbers \( r_1, r_2, \ldots, r_n \) are such that the distance between \( P_i \) and \( P_j \) is \( r_i + r_j \) for \( i \ne j \).
    -/
theorem maximum_tangency_circles (n : ℕ) (P : Fin n → ℝ × ℝ) (r : Fin n → ℝ)
  (h : ∀ i j : Fin n, i ≠ j → dist (P i) (P j) = r i + r j) : n ≤ 4 :=
sorry

end NUMINAMATH_GPT_maximum_tangency_circles_l569_56988


namespace NUMINAMATH_GPT_original_number_proof_l569_56936

-- Define the conditions
variables (x y : ℕ)
-- Given conditions
def condition1 : Prop := y = 13
def condition2 : Prop := 7 * x + 5 * y = 146

-- Goal: the original number (sum of the parts x and y)
def original_number : ℕ := x + y

-- State the problem as a theorem
theorem original_number_proof (x y : ℕ) (h1 : condition1 y) (h2 : condition2 x y) : original_number x y = 24 := by
  -- The proof will be written here
  sorry

end NUMINAMATH_GPT_original_number_proof_l569_56936


namespace NUMINAMATH_GPT_grapefruits_orchards_proof_l569_56914

/-- 
Given the following conditions:
1. There are 40 orchards in total.
2. 15 orchards are dedicated to lemons.
3. The number of orchards for oranges is two-thirds of the number of orchards for lemons.
4. Limes and grapefruits have an equal number of orchards.
5. Mandarins have half as many orchards as limes or grapefruits.
Prove that the number of citrus orchards growing grapefruits is 6.
-/
def num_grapefruit_orchards (TotalOrchards Lemons Oranges L G M : ℕ) : Prop :=
  TotalOrchards = 40 ∧
  Lemons = 15 ∧
  Oranges = 2 * Lemons / 3 ∧
  L = G ∧
  M = G / 2 ∧
  L + G + M = TotalOrchards - (Lemons + Oranges) ∧
  G = 6

theorem grapefruits_orchards_proof : ∃ (TotalOrchards Lemons Oranges L G M : ℕ), num_grapefruit_orchards TotalOrchards Lemons Oranges L G M :=
by
  sorry

end NUMINAMATH_GPT_grapefruits_orchards_proof_l569_56914


namespace NUMINAMATH_GPT_yard_length_is_correct_l569_56905

-- Definitions based on the conditions
def trees : ℕ := 26
def distance_between_trees : ℕ := 11

-- Theorem stating that the length of the yard is 275 meters
theorem yard_length_is_correct : (trees - 1) * distance_between_trees = 275 :=
by sorry

end NUMINAMATH_GPT_yard_length_is_correct_l569_56905


namespace NUMINAMATH_GPT_width_of_river_l569_56963

def ferry_problem (v1 v2 W t1 t2 : ℝ) : Prop :=
  v1 * t1 + v2 * t1 = W ∧
  v1 * t1 = 720 ∧
  v2 * t1 = W - 720 ∧
  (v1 * t2 + v2 * t2 = 3 * W) ∧
  v1 * t2 = 2 * W - 400 ∧
  v2 * t2 = W + 400

theorem width_of_river 
  (v1 v2 W t1 t2 : ℝ)
  (h : ferry_problem v1 v2 W t1 t2) :
  W = 1280 :=
by
  sorry

end NUMINAMATH_GPT_width_of_river_l569_56963


namespace NUMINAMATH_GPT_triangle_inequality_expression_non_negative_l569_56906

theorem triangle_inequality_expression_non_negative
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_expression_non_negative_l569_56906


namespace NUMINAMATH_GPT_find_third_number_l569_56933

noncomputable def third_number := 9.110300000000005

theorem find_third_number :
  12.1212 + 17.0005 - third_number = 20.011399999999995 :=
sorry

end NUMINAMATH_GPT_find_third_number_l569_56933


namespace NUMINAMATH_GPT_remainder_sum_division_by_9_l569_56934

theorem remainder_sum_division_by_9 :
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_division_by_9_l569_56934


namespace NUMINAMATH_GPT_train_length_is_250_l569_56926

noncomputable def train_length (speed_kmh : ℕ) (time_sec : ℕ) (station_length : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600 * time_sec) - station_length

theorem train_length_is_250 :
  train_length 36 45 200 = 250 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_250_l569_56926


namespace NUMINAMATH_GPT_equation_two_roots_iff_l569_56927

theorem equation_two_roots_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a) ↔ a > -1 :=
by
  sorry

end NUMINAMATH_GPT_equation_two_roots_iff_l569_56927


namespace NUMINAMATH_GPT_chocolate_chips_per_family_member_l569_56978

def total_family_members : ℕ := 4
def batches_choco_chip : ℕ := 3
def batches_double_choco_chip : ℕ := 2
def batches_white_choco_chip : ℕ := 1
def cookies_per_batch_choco_chip : ℕ := 12
def cookies_per_batch_double_choco_chip : ℕ := 10
def cookies_per_batch_white_choco_chip : ℕ := 15
def choco_chips_per_cookie_choco_chip : ℕ := 2
def choco_chips_per_cookie_double_choco_chip : ℕ := 4
def choco_chips_per_cookie_white_choco_chip : ℕ := 3

theorem chocolate_chips_per_family_member :
  (batches_choco_chip * cookies_per_batch_choco_chip * choco_chips_per_cookie_choco_chip +
   batches_double_choco_chip * cookies_per_batch_double_choco_chip * choco_chips_per_cookie_double_choco_chip +
   batches_white_choco_chip * cookies_per_batch_white_choco_chip * choco_chips_per_cookie_white_choco_chip) / 
   total_family_members = 49 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_chips_per_family_member_l569_56978


namespace NUMINAMATH_GPT_smallest_b_factor_2020_l569_56969

theorem smallest_b_factor_2020 :
  ∃ b : ℕ, b > 0 ∧
  (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ b = r + s) ∧
  (∀ c : ℕ, c > 0 → (∃ r s : ℕ, r > s ∧ r * s = 2020 ∧ c = r + s) → b ≤ c) ∧
  b = 121 :=
sorry

end NUMINAMATH_GPT_smallest_b_factor_2020_l569_56969


namespace NUMINAMATH_GPT_valid_x_for_sqrt_l569_56992

theorem valid_x_for_sqrt (x : ℝ) (hx : x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) : x ≥ 2 ↔ x = 3 := 
sorry

end NUMINAMATH_GPT_valid_x_for_sqrt_l569_56992


namespace NUMINAMATH_GPT_min_boys_needed_l569_56984

theorem min_boys_needed
  (T : ℕ) -- total apples
  (n : ℕ) -- total number of boys
  (x : ℕ) -- number of boys collecting 20 apples each
  (y : ℕ) -- number of boys collecting 20% of total apples each
  (h1 : n = x + y)
  (h2 : T = 20 * x + Nat.div (T * 20 * y) 100)
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : n ≥ 2 :=
sorry

end NUMINAMATH_GPT_min_boys_needed_l569_56984


namespace NUMINAMATH_GPT_lily_of_the_valley_bushes_needed_l569_56982

theorem lily_of_the_valley_bushes_needed 
  (r l : ℕ) (h_radius : r = 20) (h_length : l = 400) : 
  l / (2 * r) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_lily_of_the_valley_bushes_needed_l569_56982


namespace NUMINAMATH_GPT_number_of_representations_l569_56949

-- Definitions of the conditions
def is_valid_b (b : ℕ) : Prop :=
  b ≤ 99

def is_representation (b3 b2 b1 b0 : ℕ) : Prop :=
  3152 = b3 * 10^3 + b2 * 10^2 + b1 * 10 + b0

-- The theorem to prove
theorem number_of_representations : 
  ∃ (N' : ℕ), (N' = 316) ∧ 
  (∀ (b3 b2 b1 b0 : ℕ), is_representation b3 b2 b1 b0 → is_valid_b b0 → is_valid_b b1 → is_valid_b b2 → is_valid_b b3) :=
sorry

end NUMINAMATH_GPT_number_of_representations_l569_56949


namespace NUMINAMATH_GPT_Bryce_raisins_l569_56929

theorem Bryce_raisins (B C : ℚ) (h1 : B = C + 10) (h2 : C = B / 4) : B = 40 / 3 :=
by
 -- The proof goes here, but we skip it for now
 sorry

end NUMINAMATH_GPT_Bryce_raisins_l569_56929


namespace NUMINAMATH_GPT_find_certain_number_multiplied_by_24_l569_56931

-- Define the conditions
theorem find_certain_number_multiplied_by_24 :
  (∃ x : ℤ, 37 - x = 24) →
  ∀ x : ℤ, (37 - x = 24) → (x * 24 = 312) :=
by
  intros h x hx
  -- Here we will have the proof using the assumption and the theorem.
  sorry

end NUMINAMATH_GPT_find_certain_number_multiplied_by_24_l569_56931


namespace NUMINAMATH_GPT_find_g3_l569_56977

variable (g : ℝ → ℝ)

axiom condition_g :
  ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 2 * x

theorem find_g3 : g 3 = 9 / 2 :=
  by
    sorry

end NUMINAMATH_GPT_find_g3_l569_56977


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_t_for_b_l569_56966

variable (x y : ℝ)

def condition_t : Prop := x ≤ 12 ∨ y ≤ 16
def condition_b : Prop := x + y ≤ 28 ∨ x * y ≤ 192

theorem necessary_not_sufficient_condition_t_for_b (h : condition_b x y) : condition_t x y ∧ ¬ (condition_t x y → condition_b x y) := by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_t_for_b_l569_56966


namespace NUMINAMATH_GPT_reciprocal_of_complex_power_l569_56918

noncomputable def complex_num_reciprocal : ℂ :=
  (Complex.I) ^ 2023

theorem reciprocal_of_complex_power :
  ∀ z : ℂ, z = (Complex.I) ^ 2023 -> (1 / z) = Complex.I :=
by
  intro z
  intro hz
  have h_power : z = Complex.I ^ 2023 := by assumption
  sorry

end NUMINAMATH_GPT_reciprocal_of_complex_power_l569_56918


namespace NUMINAMATH_GPT_prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l569_56967

-- Definitions
def classes := 4
def students := 4
def total_distributions := classes ^ students

-- Problem 1
theorem prob_each_class_receives_one : 
  (A_4 ^ 4) / total_distributions = 3 / 32 := sorry

-- Problem 2
theorem prob_at_least_one_class_empty : 
  1 - (A_4 ^ 4) / total_distributions = 29 / 32 := sorry

-- Problem 3
theorem prob_exactly_one_class_empty :
  (C_4 ^ 1 * C_4 ^ 2 * C_3 ^ 1 * C_2 ^ 1) / total_distributions = 9 / 16 := sorry

end NUMINAMATH_GPT_prob_each_class_receives_one_prob_at_least_one_class_empty_prob_exactly_one_class_empty_l569_56967


namespace NUMINAMATH_GPT_speed_in_still_water_l569_56965

-- Given conditions
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 41

-- Question: Prove the speed of the man in still water is 33 kmph.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 33 := 
by 
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l569_56965


namespace NUMINAMATH_GPT_probability_range_inequality_l569_56956

theorem probability_range_inequality :
  ∀ p : ℝ, 0 ≤ p → p ≤ 1 →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2 → 0.4 ≤ p ∧ p < 1) := sorry

end NUMINAMATH_GPT_probability_range_inequality_l569_56956


namespace NUMINAMATH_GPT_pet_store_customers_buy_different_pets_l569_56928

theorem pet_store_customers_buy_different_pets :
  let puppies := 20
  let kittens := 10
  let hamsters := 12
  let rabbits := 5
  let customers := 4
  (puppies * kittens * hamsters * rabbits * Nat.factorial customers = 288000) := 
by
  sorry

end NUMINAMATH_GPT_pet_store_customers_buy_different_pets_l569_56928


namespace NUMINAMATH_GPT_incorrect_operation_B_l569_56954

variables (a b c : ℝ)

theorem incorrect_operation_B : (c - 2 * (a + b)) ≠ (c - 2 * a + 2 * b) := by
  sorry

end NUMINAMATH_GPT_incorrect_operation_B_l569_56954


namespace NUMINAMATH_GPT_fraction_of_married_men_l569_56912

-- We start by defining the conditions given in the problem.
def only_single_women_and_married_couples (total_women total_married_women : ℕ) :=
  total_women - total_married_women + total_married_women * 2

def probability_single_woman_single (total_women total_single_women : ℕ) :=
  total_single_women / total_women = 3 / 7

-- The main theorem we need to prove under the given conditions.
theorem fraction_of_married_men (total_women total_married_women : ℕ)
  (h1 : probability_single_woman_single total_women (total_women - total_married_women))
  : (total_married_women * 2) / (total_women + total_married_women) = 4 / 11 := sorry

end NUMINAMATH_GPT_fraction_of_married_men_l569_56912


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l569_56907

-- Statement for Problem 1
theorem problem1_solution (x : ℝ) : (1 / 2 * (x - 3) ^ 2 = 18) ↔ (x = 9 ∨ x = -3) :=
by sorry

-- Statement for Problem 2
theorem problem2_solution (x : ℝ) : (x ^ 2 + 6 * x = 5) ↔ (x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l569_56907


namespace NUMINAMATH_GPT_symmetric_graph_l569_56917

variable (f : ℝ → ℝ)
variable (c : ℝ)
variable (h_nonzero : c ≠ 0)
variable (h_fx_plus_y : ∀ (x y : ℝ), f (x + y) + f (x - y) = 2 * f x * f y)
variable (h_f_half_c : f (c / 2) = 0)
variable (h_f_zero : f 0 ≠ 0)

theorem symmetric_graph (k : ℤ) : 
  ∀ (x : ℝ), f (x) = f (2*k*c - x) :=
sorry

end NUMINAMATH_GPT_symmetric_graph_l569_56917


namespace NUMINAMATH_GPT_min_dot_product_l569_56913

noncomputable def ellipse_eq_p (x y : ℝ) : Prop :=
    x^2 / 9 + y^2 / 8 = 1

noncomputable def dot_product_op_fp (x y : ℝ) : ℝ :=
    x^2 + x + y^2

theorem min_dot_product : 
    (∀ x y : ℝ, ellipse_eq_p x y → dot_product_op_fp x y = 6) := 
sorry

end NUMINAMATH_GPT_min_dot_product_l569_56913


namespace NUMINAMATH_GPT_proof_two_digit_number_l569_56943

noncomputable def two_digit_number := {n : ℤ // 10 ≤ n ∧ n ≤ 99}

theorem proof_two_digit_number (n : two_digit_number) :
  (n.val % 2 = 0) ∧ 
  ((n.val + 1) % 3 = 0) ∧
  ((n.val + 2) % 4 = 0) ∧
  ((n.val + 3) % 5 = 0) →
  n.val = 62 :=
by sorry

end NUMINAMATH_GPT_proof_two_digit_number_l569_56943


namespace NUMINAMATH_GPT_donation_value_l569_56959

def donation_in_yuan (usd: ℝ) (exchange_rate: ℝ): ℝ :=
  usd * exchange_rate

theorem donation_value :
  donation_in_yuan 1.2 6.25 = 7.5 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_donation_value_l569_56959


namespace NUMINAMATH_GPT_interval_between_prizes_l569_56925

theorem interval_between_prizes (total_prize : ℝ) (first_place : ℝ) (interval : ℝ) :
  total_prize = 4800 ∧
  first_place = 2000 ∧
  (first_place - interval) + (first_place - 2 * interval) = total_prize - 2000 →
  interval = 400 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_interval_between_prizes_l569_56925


namespace NUMINAMATH_GPT_largest_term_at_k_31_l569_56951

noncomputable def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.15)^k

theorem largest_term_at_k_31 : 
  ∀ k : ℕ, (k ≤ 500) →
    (B_k 31 ≥ B_k k) :=
by
  intro k hk
  sorry

end NUMINAMATH_GPT_largest_term_at_k_31_l569_56951


namespace NUMINAMATH_GPT_range_of_m_l569_56930

def cond1 (x : ℝ) : Prop := x^2 - 4 * x + 3 < 0
def cond2 (x : ℝ) : Prop := x^2 - 6 * x + 8 < 0
def cond3 (x m : ℝ) : Prop := 2 * x^2 - 9 * x + m < 0

theorem range_of_m (m : ℝ) : (∀ x, cond1 x → cond2 x → cond3 x m) → m < 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l569_56930


namespace NUMINAMATH_GPT_trigonometric_identity_l569_56991

theorem trigonometric_identity (alpha : ℝ) (h : Real.tan alpha = 2 * Real.tan (π / 5)) :
  (Real.cos (alpha - 3 * π / 10) / Real.sin (alpha - π / 5)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l569_56991


namespace NUMINAMATH_GPT_find_g_neg_3_l569_56964

def g (x : ℤ) : ℤ :=
if x < 1 then 3 * x - 4 else x + 6

theorem find_g_neg_3 : g (-3) = -13 :=
by
  -- proof omitted: sorry
  sorry

end NUMINAMATH_GPT_find_g_neg_3_l569_56964


namespace NUMINAMATH_GPT_print_time_is_fifteen_l569_56998

noncomputable def time_to_print (total_pages rate : ℕ) := 
  (total_pages : ℚ) / rate

theorem print_time_is_fifteen :
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  round time = 15 := by
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  have time_val : time = (350 : ℚ) / 24 := by rfl
  let rounded_time := round time
  have rounded_time_val : rounded_time = 15 := by sorry
  exact rounded_time_val

end NUMINAMATH_GPT_print_time_is_fifteen_l569_56998


namespace NUMINAMATH_GPT_trapezoid_perimeter_l569_56921

theorem trapezoid_perimeter (AB CD BC DA : ℝ) (BCD_angle : ℝ)
  (h1 : AB = 60) (h2 : CD = 40) (h3 : BC = DA) (h4 : BCD_angle = 120) :
  AB + BC + CD + DA = 220 := 
sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l569_56921


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l569_56952

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_a5 (a₁ d : ℝ) (h1 : a 2 a₁ d = 2 * a 3 a₁ d + 1) (h2 : a 4 a₁ d = 2 * a 3 a₁ d + 7) :
  a 5 a₁ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l569_56952


namespace NUMINAMATH_GPT_range_of_a_l569_56908

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - a = 0) ↔ a ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l569_56908


namespace NUMINAMATH_GPT_smallest_m_l569_56940

-- Defining the remainder function
def r (m n : ℕ) : ℕ := m % n

-- Main theorem stating the problem needed to be proved
theorem smallest_m (m : ℕ) (h : m > 0) 
  (H : (r m 1 + r m 2 + r m 3 + r m 4 + r m 5 + r m 6 + r m 7 + r m 8 + r m 9 + r m 10) = 4) : 
  m = 120 :=
sorry

end NUMINAMATH_GPT_smallest_m_l569_56940


namespace NUMINAMATH_GPT_find_smaller_integer_l569_56999

theorem find_smaller_integer
  (x y : ℤ)
  (h1 : x + y = 30)
  (h2 : 2 * y = 5 * x - 10) :
  x = 10 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_find_smaller_integer_l569_56999


namespace NUMINAMATH_GPT_non_zero_digits_of_fraction_l569_56910

def fraction : ℚ := 80 / (2^4 * 5^9)

def decimal_expansion (x : ℚ) : String :=
  -- some function to compute the decimal expansion of a fraction as a string
  "0.00000256" -- placeholder

def non_zero_digits_to_right (s : String) : ℕ :=
  -- some function to count the number of non-zero digits to the right of the decimal point in the string
  3 -- placeholder

theorem non_zero_digits_of_fraction : non_zero_digits_to_right (decimal_expansion fraction) = 3 := by
  sorry

end NUMINAMATH_GPT_non_zero_digits_of_fraction_l569_56910


namespace NUMINAMATH_GPT_segment_problem_l569_56901

theorem segment_problem 
  (A C : ℝ) (B D : ℝ) (P Q : ℝ) (x y k : ℝ)
  (hA : A = 0) (hC : C = 0) 
  (hB : B = 6) (hD : D = 9)
  (hx : x = P - A) (hy : y = Q - C) 
  (hxk : x = 3 * k)
  (hxyk : x + y = 12 * k) :
  k = 2 :=
  sorry

end NUMINAMATH_GPT_segment_problem_l569_56901


namespace NUMINAMATH_GPT_grocer_display_rows_l569_56944

theorem grocer_display_rows (n : ℕ)
  (h1 : ∃ k, k = 2 + 3 * (n - 1))
  (h2 : ∃ s, s = (n / 2) * (2 + (3 * n - 1))):
  (3 * n^2 + n) / 2 = 225 → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_grocer_display_rows_l569_56944


namespace NUMINAMATH_GPT_paul_books_sold_l569_56904

theorem paul_books_sold:
  ∀ (initial_books friend_books sold_per_day days final_books sold_books: ℝ),
    initial_books = 284.5 →
    friend_books = 63.7 →
    sold_per_day = 16.25 →
    days = 8 →
    final_books = 112.3 →
    sold_books = initial_books - friend_books - final_books →
    sold_books = 108.5 :=
by intros initial_books friend_books sold_per_day days final_books sold_books
   sorry

end NUMINAMATH_GPT_paul_books_sold_l569_56904


namespace NUMINAMATH_GPT_rational_numbers_inequality_l569_56911

theorem rational_numbers_inequality (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 :=
sorry

end NUMINAMATH_GPT_rational_numbers_inequality_l569_56911


namespace NUMINAMATH_GPT_faster_car_distance_l569_56957

theorem faster_car_distance (d v : ℝ) (h_dist: d + 2 * d = 4) (h_faster: 2 * v = 2 * (d / v)) : 
  d = 4 / 3 → 2 * d = 8 / 3 :=
by sorry

end NUMINAMATH_GPT_faster_car_distance_l569_56957


namespace NUMINAMATH_GPT_sum_proper_divisors_81_l569_56970

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end NUMINAMATH_GPT_sum_proper_divisors_81_l569_56970


namespace NUMINAMATH_GPT_younger_brother_age_l569_56990

variable (x y : ℕ)

theorem younger_brother_age :
  x + y = 46 →
  y = x / 3 + 10 →
  y = 19 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_younger_brother_age_l569_56990


namespace NUMINAMATH_GPT_width_of_room_l569_56987

-- Define the givens
def length_of_room : ℝ := 5.5
def total_cost : ℝ := 20625
def rate_per_sq_meter : ℝ := 1000

-- Define the required proof statement
theorem width_of_room : (total_cost / rate_per_sq_meter) / length_of_room = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_width_of_room_l569_56987


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l569_56997

theorem solve_equation_1 (x : ℝ) : x^2 - 7 * x = 0 ↔ (x = 0 ∨ x = 7) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 6 * x + 1 = 0 ↔ (x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) :=
by sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l569_56997


namespace NUMINAMATH_GPT_rectangle_width_is_4_l569_56948

-- Definitions of conditions
variable (w : ℝ) -- width of the rectangle
def length := w + 2 -- length of the rectangle
def perimeter := 2 * w + 2 * (w + 2) -- perimeter of the rectangle, using given conditions

-- The theorem to be proved
theorem rectangle_width_is_4 (h : perimeter = 20) : w = 4 :=
by {
  sorry -- To be proved
}

end NUMINAMATH_GPT_rectangle_width_is_4_l569_56948


namespace NUMINAMATH_GPT_decimal_6_to_binary_is_110_l569_56919

def decimal_to_binary (n : ℕ) : ℕ :=
  -- This is just a placeholder definition. Adjust as needed for formalization.
  sorry

theorem decimal_6_to_binary_is_110 :
  decimal_to_binary 6 = 110 := 
sorry

end NUMINAMATH_GPT_decimal_6_to_binary_is_110_l569_56919


namespace NUMINAMATH_GPT_white_longer_than_blue_l569_56961

noncomputable def whiteLineInches : ℝ := 7.666666666666667
noncomputable def blueLineInches : ℝ := 3.3333333333333335
noncomputable def inchToCm : ℝ := 2.54
noncomputable def cmToMm : ℝ := 10

theorem white_longer_than_blue :
  let whiteLineCm := whiteLineInches * inchToCm
  let blueLineCm := blueLineInches * inchToCm
  let differenceCm := whiteLineCm - blueLineCm
  let differenceMm := differenceCm * cmToMm
  differenceMm = 110.05555555555553 := by
  sorry

end NUMINAMATH_GPT_white_longer_than_blue_l569_56961


namespace NUMINAMATH_GPT_cos_double_angle_l569_56971

theorem cos_double_angle (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l569_56971


namespace NUMINAMATH_GPT_inequality_le_one_equality_case_l569_56975

open Real

theorem inequality_le_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) :=
sorry

theorem equality_case (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_GPT_inequality_le_one_equality_case_l569_56975


namespace NUMINAMATH_GPT_polynomial_evaluation_l569_56924

theorem polynomial_evaluation (a : ℝ) (h : a^2 + 3 * a = 2) : 2 * a^2 + 6 * a - 10 = -6 := by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l569_56924


namespace NUMINAMATH_GPT_common_root_rational_l569_56915

variable (a b c d e f g : ℚ) -- coefficient variables

def poly1 (x : ℚ) : ℚ := 90 * x^4 + a * x^3 + b * x^2 + c * x + 18

def poly2 (x : ℚ) : ℚ := 18 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 90

theorem common_root_rational (k : ℚ) (h1 : poly1 a b c k = 0) (h2 : poly2 d e f g k = 0) 
  (hn : k < 0) (hi : ∀ (m n : ℤ), k ≠ m / n) : k = -1/3 := sorry

end NUMINAMATH_GPT_common_root_rational_l569_56915


namespace NUMINAMATH_GPT_appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l569_56937

-- Definitions for binomial coefficient and Pascal's triangle

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Check occurrences in Pascal's triangle more than three times
theorem appears_more_than_three_times_in_Pascal (n : ℕ) :
  n = 10 ∨ n = 15 ∨ n = 21 → ∃ a b c : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ 
    (binomial_coeff a 2 = n ∨ binomial_coeff a 3 = n) ∧
    (binomial_coeff b 2 = n ∨ binomial_coeff b 3 = n) ∧
    (binomial_coeff c 2 = n ∨ binomial_coeff c 3 = n) := 
by
  sorry

-- Check occurrences in Pascal's triangle more than four times
theorem appears_more_than_four_times_in_Pascal (n : ℕ) :
  n = 120 ∨ n = 210 ∨ n = 3003 → ∃ a b c d : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ (1 < d) ∧ 
    (binomial_coeff a 3 = n ∨ binomial_coeff a 4 = n) ∧
    (binomial_coeff b 3 = n ∨ binomial_coeff b 4 = n) ∧
    (binomial_coeff c 3 = n ∨ binomial_coeff c 4 = n) ∧
    (binomial_coeff d 3 = n ∨ binomial_coeff d 4 = n) := 
by
  sorry

end NUMINAMATH_GPT_appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l569_56937


namespace NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l569_56920

theorem minimum_value_of_sum_of_squares (x y z : ℝ) (h : 4 * x + 3 * y + 12 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 169 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_sum_of_squares_l569_56920


namespace NUMINAMATH_GPT_meeting_time_l569_56935

def time_Cassie_leaves : ℕ := 495 -- 8:15 AM in minutes past midnight
def speed_Cassie : ℕ := 12 -- mph
def break_Cassie : ℚ := 0.25 -- hours
def time_Brian_leaves : ℕ := 540 -- 9:00 AM in minutes past midnight
def speed_Brian : ℕ := 14 -- mph
def total_distance : ℕ := 74 -- miles

def time_in_minutes (h m : ℕ) : ℕ := h * 60 + m

theorem meeting_time : time_Cassie_leaves + (87 : ℚ) / 26 * 60 = time_in_minutes 11 37 := 
by sorry

end NUMINAMATH_GPT_meeting_time_l569_56935


namespace NUMINAMATH_GPT_total_weight_is_28_87_l569_56960

def blue_ball_weight : ℝ := 6
def brown_ball_weight : ℝ := 3.12
def green_ball_weight : ℝ := 4.25

def red_ball_weight : ℝ := 2 * green_ball_weight
def yellow_ball_weight : ℝ := red_ball_weight - 1.5

def total_weight : ℝ := blue_ball_weight + brown_ball_weight + green_ball_weight + red_ball_weight + yellow_ball_weight

theorem total_weight_is_28_87 : total_weight = 28.87 :=
by
  /- proof goes here -/
  sorry

end NUMINAMATH_GPT_total_weight_is_28_87_l569_56960


namespace NUMINAMATH_GPT_max_cookies_Andy_can_eat_l569_56909

theorem max_cookies_Andy_can_eat 
  (x y : ℕ) 
  (h1 : x + y = 36)
  (h2 : y ≥ 2 * x) : 
  x ≤ 12 := by
  sorry

end NUMINAMATH_GPT_max_cookies_Andy_can_eat_l569_56909


namespace NUMINAMATH_GPT_persons_in_boat_l569_56979

theorem persons_in_boat (W1 W2 new_person_weight : ℝ) (n : ℕ)
  (hW1 : W1 = 55)
  (h_new_person : new_person_weight = 50)
  (hW2 : W2 = W1 - 5) :
  (n * W1 + new_person_weight) / (n + 1) = W2 → false :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_persons_in_boat_l569_56979


namespace NUMINAMATH_GPT_parallelogram_construction_l569_56989

theorem parallelogram_construction 
  (α : ℝ) (hα : 0 ≤ α ∧ α < 180)
  (A B : (ℝ × ℝ))
  (in_angle : (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ α ∧ 
               ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ α))
  (C D : (ℝ × ℝ)) :
  ∃ O : (ℝ × ℝ), 
    O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) :=
sorry

end NUMINAMATH_GPT_parallelogram_construction_l569_56989


namespace NUMINAMATH_GPT_zeroes_in_base_81_l569_56946

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_zeroes_in_base_81_l569_56946


namespace NUMINAMATH_GPT_central_angle_measures_l569_56947

-- Definitions for the conditions
def perimeter_eq (r l : ℝ) : Prop := l + 2 * r = 6
def area_eq (r l : ℝ) : Prop := (1 / 2) * l * r = 2
def central_angle (r l α : ℝ) : Prop := α = l / r

-- The final proof statement
theorem central_angle_measures (r l α : ℝ) (h1 : perimeter_eq r l) (h2 : area_eq r l) :
  central_angle r l α → (α = 1 ∨ α = 4) :=
sorry

end NUMINAMATH_GPT_central_angle_measures_l569_56947


namespace NUMINAMATH_GPT_cat_mouse_position_after_300_moves_l569_56945

def move_pattern_cat_mouse :=
  let cat_cycle_length := 4
  let mouse_cycle_length := 8
  let cat_moves := 300
  let mouse_moves := (3 / 2) * cat_moves
  let cat_position := (cat_moves % cat_cycle_length)
  let mouse_position := (mouse_moves % mouse_cycle_length)
  (cat_position, mouse_position)

theorem cat_mouse_position_after_300_moves :
  move_pattern_cat_mouse = (0, 2) :=
by
  sorry

end NUMINAMATH_GPT_cat_mouse_position_after_300_moves_l569_56945


namespace NUMINAMATH_GPT_find_a_and_b_l569_56995

theorem find_a_and_b (a b : ℝ) 
  (curve : ∀ x : ℝ, y = x^2 + a * x + b) 
  (tangent : ∀ x : ℝ, y - b = a * x) 
  (tangent_line : ∀ x y : ℝ, x + y = 1) :
  a = -1 ∧ b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_and_b_l569_56995


namespace NUMINAMATH_GPT_find_integer_tuples_l569_56973

theorem find_integer_tuples (a b c x y z : ℤ) :
  a + b + c = x * y * z →
  x + y + z = a * b * c →
  a ≥ b → b ≥ c → c ≥ 1 →
  x ≥ y → y ≥ z → z ≥ 1 →
  (a, b, c, x, y, z) = (2, 2, 2, 6, 1, 1) ∨
  (a, b, c, x, y, z) = (5, 2, 1, 8, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 3, 1, 7, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 2, 1, 6, 2, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_tuples_l569_56973


namespace NUMINAMATH_GPT_unique_intersection_point_l569_56972

theorem unique_intersection_point (c : ℝ) :
  (∀ x : ℝ, (|x - 20| + |x + 18| = x + c) → (x = 18 - 2 \/ x = 38 - x \/ x = 2 - 3 * x)) →
  c = 18 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_point_l569_56972


namespace NUMINAMATH_GPT_max_value_fraction_l569_56962

theorem max_value_fraction (x : ℝ) : 
  (∃ x, (x^4 / (x^8 + 4 * x^6 - 8 * x^4 + 16 * x^2 + 64)) = (1 / 24)) := 
sorry

end NUMINAMATH_GPT_max_value_fraction_l569_56962


namespace NUMINAMATH_GPT_application_outcomes_l569_56950

theorem application_outcomes :
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  (choices_A * choices_B * choices_C) = 18 :=
by
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  show (choices_A * choices_B * choices_C = 18)
  sorry

end NUMINAMATH_GPT_application_outcomes_l569_56950


namespace NUMINAMATH_GPT_evaluate_f_at_3_over_4_l569_56996

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ := (1 - y) / y

theorem evaluate_f_at_3_over_4 (h : g (x : ℝ) = 1 - x^2) (x_ne_zero : x ≠ 0) :
  f (3 / 4) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_over_4_l569_56996


namespace NUMINAMATH_GPT_cubes_of_roots_l569_56900

theorem cubes_of_roots (a b c : ℝ) (h1 : a + b + c = 2) (h2 : ab + ac + bc = 2) (h3 : abc = 3) : 
  a^3 + b^3 + c^3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_cubes_of_roots_l569_56900


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l569_56903

noncomputable def arithmetic_sequence_terms (a d n : ℕ) : Prop :=
  let sum_first_three := 3 * a + 3 * d = 34
  let sum_last_three := 3 * a + 3 * (n - 1) * d = 146
  let sum_all := n * (2 * a + (n - 1) * d) / 2 = 390
  (sum_first_three ∧ sum_last_three ∧ sum_all) → n = 13

theorem number_of_terms_in_arithmetic_sequence (a d n : ℕ) : arithmetic_sequence_terms a d n → n = 13 := 
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l569_56903


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l569_56981

theorem arithmetic_sequence_terms (a d n : ℕ) 
  (h_sum_first_3 : 3 * a + 3 * d = 34)
  (h_sum_last_3 : 3 * a + 3 * d * (n - 1) = 146)
  (h_sum_all : n * (2 * a + (n - 1) * d) = 2 * 390) : 
  n = 13 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l569_56981


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_l569_56923

theorem arithmetic_sequence_a6 (a : ℕ → ℕ)
  (h_arith_seq : ∀ n, ∃ d, a (n+1) = a n + d)
  (h_sum : a 4 + a 8 = 16) : a 6 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_l569_56923


namespace NUMINAMATH_GPT_probability_reach_2C_l569_56993

noncomputable def f (x C : ℝ) : ℝ :=
  x / (2 * C)

theorem probability_reach_2C (x C : ℝ) (hC : 0 < C) (hx : 0 < x ∧ x < 2 * C) :
  f x C = x / (2 * C) := 
by
  sorry

end NUMINAMATH_GPT_probability_reach_2C_l569_56993


namespace NUMINAMATH_GPT_molecular_weight_of_N2O5_is_correct_l569_56976

-- Definitions for atomic weights
def atomic_weight_N : ℚ := 14.01
def atomic_weight_O : ℚ := 16.00

-- Define the molecular weight calculation for N2O5
def molecular_weight_N2O5 : ℚ := (2 * atomic_weight_N) + (5 * atomic_weight_O)

-- The theorem to prove
theorem molecular_weight_of_N2O5_is_correct : molecular_weight_N2O5 = 108.02 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_molecular_weight_of_N2O5_is_correct_l569_56976


namespace NUMINAMATH_GPT_ratio_of_bronze_to_silver_l569_56985

def total_gold_coins := 3500
def num_chests := 5
def total_silver_coins := 500
def coins_per_chest := 1000

-- Definitions based on the conditions to be used in the proof
def gold_coins_per_chest := total_gold_coins / num_chests
def silver_coins_per_chest := total_silver_coins / num_chests
def bronze_coins_per_chest := coins_per_chest - gold_coins_per_chest - silver_coins_per_chest
def bronze_to_silver_ratio := bronze_coins_per_chest / silver_coins_per_chest

theorem ratio_of_bronze_to_silver : bronze_to_silver_ratio = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_bronze_to_silver_l569_56985


namespace NUMINAMATH_GPT_option_c_same_function_l569_56922

-- Definitions based on conditions
def f_c (x : ℝ) : ℝ := x^2
def g_c (x : ℝ) : ℝ := 3 * x^6

-- Theorem statement that Option C f(x) and g(x) represent the same function
theorem option_c_same_function : ∀ x : ℝ, f_c x = g_c x := by
  sorry

end NUMINAMATH_GPT_option_c_same_function_l569_56922


namespace NUMINAMATH_GPT_Cl_invalid_electrons_l569_56941

noncomputable def Cl_mass_number : ℕ := 35
noncomputable def Cl_protons : ℕ := 17
noncomputable def Cl_neutrons : ℕ := Cl_mass_number - Cl_protons
noncomputable def Cl_electrons : ℕ := Cl_protons

theorem Cl_invalid_electrons : Cl_electrons ≠ 18 :=
by
  sorry

end NUMINAMATH_GPT_Cl_invalid_electrons_l569_56941


namespace NUMINAMATH_GPT_anayet_speed_is_61_l569_56953

-- Define the problem conditions
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_time : ℝ := 2
def total_distance : ℝ := 369
def remaining_distance : ℝ := 121

-- Calculate derived values
def amoli_distance : ℝ := amoli_speed * amoli_time
def covered_distance : ℝ := total_distance - remaining_distance
def anayet_distance : ℝ := covered_distance - amoli_distance

-- Define the theorem to prove Anayet's speed
theorem anayet_speed_is_61 : anayet_distance / anayet_time = 61 :=
by
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_anayet_speed_is_61_l569_56953


namespace NUMINAMATH_GPT_anne_cleaning_time_l569_56974

theorem anne_cleaning_time :
  ∃ (A B : ℝ), (4 * (B + A) = 1) ∧ (3 * (B + 2 * A) = 1) ∧ (1 / A = 12) :=
by
  sorry

end NUMINAMATH_GPT_anne_cleaning_time_l569_56974


namespace NUMINAMATH_GPT_sqrt_of_9_fact_over_84_eq_24_sqrt_15_l569_56955

theorem sqrt_of_9_fact_over_84_eq_24_sqrt_15 :
  Real.sqrt (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / (2^2 * 3 * 7)) = 24 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_9_fact_over_84_eq_24_sqrt_15_l569_56955


namespace NUMINAMATH_GPT_alice_age_2005_l569_56980

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end NUMINAMATH_GPT_alice_age_2005_l569_56980


namespace NUMINAMATH_GPT_system_of_equations_has_no_solution_l569_56986

theorem system_of_equations_has_no_solution
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y + z = 10)
  (h2 : 6 * x - 8 * y + 2 * z = 16)
  (h3 : x + y - z = 3) :
  false :=
by 
  sorry

end NUMINAMATH_GPT_system_of_equations_has_no_solution_l569_56986


namespace NUMINAMATH_GPT_library_visit_period_l569_56916

noncomputable def dance_class_days := 6
noncomputable def karate_class_days := 12
noncomputable def common_days := 36

theorem library_visit_period (library_days : ℕ) 
  (hdance : ∀ (n : ℕ), n * dance_class_days = common_days)
  (hkarate : ∀ (n : ℕ), n * karate_class_days = common_days)
  (hcommon : ∀ (n : ℕ), n * library_days = common_days) : 
  library_days = 18 := 
sorry

end NUMINAMATH_GPT_library_visit_period_l569_56916


namespace NUMINAMATH_GPT_negation_of_p_l569_56958

theorem negation_of_p :
  (∃ x : ℝ, x < 0 ∧ x + (1 / x) > -2) ↔ ¬ (∀ x : ℝ, x < 0 → x + (1 / x) ≤ -2) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_p_l569_56958


namespace NUMINAMATH_GPT_value_of_y_l569_56932

theorem value_of_y (y : ℝ) (α : ℝ) (h₁ : (-3, y) = (x, y)) (h₂ : Real.sin α = -3 / 4) : 
  y = -9 * Real.sqrt 7 / 7 := 
  sorry

end NUMINAMATH_GPT_value_of_y_l569_56932


namespace NUMINAMATH_GPT_inradius_of_triangle_l569_56938

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 15) (h2 : b = 16) (h3 : c = 17) : 
    let s := (a + b + c) / 2
    let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
    let r := area / s
    r = Real.sqrt 21 := by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l569_56938


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_solution_range_l569_56994

-- Part I: Defining the function and proving the solution set for m = 3
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part_I_solution_set (x : ℝ) :
  (f x 3 ≥ 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
sorry

-- Part II: Proving the range of values for m such that f(x) ≥ 8 for any real number x
theorem part_II_solution_range (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_solution_range_l569_56994
