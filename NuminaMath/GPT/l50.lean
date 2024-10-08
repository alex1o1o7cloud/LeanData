import Mathlib

namespace problem_statement_l50_50606

theorem problem_statement (k x₁ x₂ : ℝ) (hx₁x₂ : x₁ < x₂)
  (h_eq : ∀ x : ℝ, x^2 - (k - 3) * x + (k + 4) = 0) 
  (P : ℝ) (hP : P ≠ 0) 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (hacute : ∀ A B : ℝ, A = x₁ ∧ B = x₂ ∧ A < 0 ∧ B > 0) :
  k < -4 ∧ α ≠ β ∧ α < β := 
sorry

end problem_statement_l50_50606


namespace workers_in_workshop_l50_50052

theorem workers_in_workshop (W : ℕ) (h1 : W ≤ 100) (h2 : W % 3 = 0) (h3 : W % 25 = 0)
  : W = 75 ∧ W / 3 = 25 ∧ W * 8 / 100 = 6 :=
by
  sorry

end workers_in_workshop_l50_50052


namespace total_kids_played_l50_50125

theorem total_kids_played (kids_monday : ℕ) (kids_tuesday : ℕ) (h_monday : kids_monday = 4) (h_tuesday : kids_tuesday = 14) : 
  kids_monday + kids_tuesday = 18 := 
by
  -- proof steps here (for now, use sorry to skip the proof)
  sorry

end total_kids_played_l50_50125


namespace cost_price_toy_l50_50250

theorem cost_price_toy (selling_price_total : ℝ) (total_toys : ℕ) (gain_toys : ℕ) (sp_per_toy : ℝ) (general_cost : ℝ) :
  selling_price_total = 27300 →
  total_toys = 18 →
  gain_toys = 3 →
  sp_per_toy = selling_price_total / total_toys →
  general_cost = sp_per_toy * total_toys - (sp_per_toy * gain_toys / total_toys) →
    general_cost = 1300 := 
by 
  sorry

end cost_price_toy_l50_50250


namespace find_k_l50_50846

theorem find_k : 
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = 3 * x + 7 ∧ y = -4 * x + 1) ∧ 
    ∃ (x y : ℚ), y = 3 * x + 7 ∧ y = 2 * x + k ∧ k = 43 / 7 := 
sorry

end find_k_l50_50846


namespace evaluate_expr_l50_50945

theorem evaluate_expr (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 5 * x^(y+1) + 6 * y^(x+1) = 2751 :=
by
  rw [h₁, h₂]
  rfl

end evaluate_expr_l50_50945


namespace total_cost_l50_50260

def c_teacher : ℕ := 60
def c_student : ℕ := 40

theorem total_cost (x : ℕ) : ∃ y : ℕ, y = c_student * x + c_teacher := by
  sorry

end total_cost_l50_50260


namespace union_of_sets_l50_50729

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | x^2 - 3 * x ≤ 0}

theorem union_of_sets : M ∪ N = {x : ℝ | -1 < x ∧ x ≤ 3} :=
by sorry

end union_of_sets_l50_50729


namespace minimum_points_to_guarantee_highest_score_l50_50087

theorem minimum_points_to_guarantee_highest_score :
  ∃ (score1 score2 score3 : ℕ), 
   (score1 = 7 ∨ score1 = 4 ∨ score1 = 2) ∧ (score2 = 7 ∨ score2 = 4 ∨ score2 = 2) ∧
   (score3 = 7 ∨ score3 = 4 ∨ score3 = 2) ∧ 
   (∀ (score4 : ℕ), 
     (score4 = 7 ∨ score4 = 4 ∨ score4 = 2) → 
     (score1 + score2 + score3 + score4 < 25)) → 
  score1 + score2 + score3 + 7 ≥ 25 :=
   sorry

end minimum_points_to_guarantee_highest_score_l50_50087


namespace range_of_k_l50_50923

theorem range_of_k (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_ne : x ≠ 2) :
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) ↔ (k > -2 ∧ k ≠ 4) :=
by
  sorry

end range_of_k_l50_50923


namespace evaluate_expression_l50_50428

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -a - b^3 + a * b^2 = 59 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l50_50428


namespace cone_lateral_area_l50_50042

theorem cone_lateral_area (r l S: ℝ) (h1: r = 1 / 2) (h2: l = 1) (h3: S = π * r * l) : 
  S = π / 2 :=
by
  sorry

end cone_lateral_area_l50_50042


namespace find_y_parallel_l50_50140

-- Definitions
def a : ℝ × ℝ := (2, 3)
def b (y : ℝ) : ℝ × ℝ := (4, -1 + y)

-- Parallel condition implies proportional components
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- The proof problem
theorem find_y_parallel : ∀ y : ℝ, parallel_vectors a (b y) → y = 7 :=
by
  sorry

end find_y_parallel_l50_50140


namespace cylinder_volume_scaling_l50_50961

theorem cylinder_volume_scaling (r h : ℝ) (V : ℝ) (V' : ℝ) 
  (h_original : V = π * r^2 * h) 
  (h_new : V' = π * (1.5 * r)^2 * (3 * h)) :
  V' = 6.75 * V := by
  sorry

end cylinder_volume_scaling_l50_50961


namespace abs_diff_eq_sqrt_l50_50914

theorem abs_diff_eq_sqrt (x1 x2 a b : ℝ) (h1 : x1 + x2 = a) (h2 : x1 * x2 = b) : 
  |x1 - x2| = Real.sqrt (a^2 - 4 * b) :=
by
  sorry

end abs_diff_eq_sqrt_l50_50914


namespace corrected_mean_is_40_point_6_l50_50545

theorem corrected_mean_is_40_point_6 
  (mean_original : ℚ) (num_observations : ℕ) (wrong_observation : ℚ) (correct_observation : ℚ) :
  num_observations = 50 → mean_original = 40 → wrong_observation = 15 → correct_observation = 45 →
  ((mean_original * num_observations + (correct_observation - wrong_observation)) / num_observations = 40.6 : Prop) :=
by intros; sorry

end corrected_mean_is_40_point_6_l50_50545


namespace total_cards_proof_l50_50324

-- Define the standard size of a deck of playing cards
def standard_deck_size : Nat := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : Nat := 6

-- Define the number of additional cards the shopkeeper has
def additional_cards : Nat := 7

-- Define the total number of cards from the complete decks
def total_deck_cards : Nat := complete_decks * standard_deck_size

-- Define the total number of all cards the shopkeeper has
def total_cards : Nat := total_deck_cards + additional_cards

-- The theorem statement that we need to prove
theorem total_cards_proof : total_cards = 319 := by
  sorry

end total_cards_proof_l50_50324


namespace abc_sum_square_identity_l50_50185

theorem abc_sum_square_identity (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 941) (h2 : a + b + c = 31) :
  ab + bc + ca = 10 :=
by
  sorry

end abc_sum_square_identity_l50_50185


namespace train_length_l50_50449

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : v = (L + 130) / 15)
  (h2 : v = (L + 250) / 20) : 
  L = 230 :=
sorry

end train_length_l50_50449


namespace jerry_pool_depth_l50_50730

theorem jerry_pool_depth :
  ∀ (total_gallons : ℝ) (gallons_drinking_cooking : ℝ) (gallons_per_shower : ℝ)
    (number_of_showers : ℝ) (pool_length : ℝ) (pool_width : ℝ)
    (gallons_per_cubic_foot : ℝ),
    total_gallons = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    number_of_showers = 15 →
    pool_length = 10 →
    pool_width = 10 →
    gallons_per_cubic_foot = 1 →
    (total_gallons - (gallons_drinking_cooking + gallons_per_shower * number_of_showers)) / 
    (pool_length * pool_width) = 6 := 
by
  intros total_gallons gallons_drinking_cooking gallons_per_shower number_of_showers pool_length pool_width gallons_per_cubic_foot
  intros total_gallons_eq drinking_cooking_eq shower_eq showers_eq length_eq width_eq cubic_foot_eq
  sorry

end jerry_pool_depth_l50_50730


namespace bruce_purchased_mangoes_l50_50544

noncomputable def calculate_mango_quantity (grapes_quantity : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let cost_of_grapes := grapes_quantity * grapes_rate
  let cost_of_mangoes := total_paid - cost_of_grapes
  cost_of_mangoes / mango_rate

theorem bruce_purchased_mangoes :
  calculate_mango_quantity 8 70 55 1055 = 9 :=
by
  sorry

end bruce_purchased_mangoes_l50_50544


namespace find_m_plus_n_l50_50742

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + x

theorem find_m_plus_n (m n : ℝ) (h1 : m < n ∧ n ≤ 1) (h2 : ∀ (x : ℝ), m ≤ x ∧ x ≤ n → 3 * m ≤ f x ∧ f x ≤ 3 * n) : m + n = -4 :=
by
  have H1 : - (1 / 2) * m^2 + m = 3 * m := sorry
  have H2 : - (1 / 2) * n^2 + n = 3 * n := sorry
  sorry

end find_m_plus_n_l50_50742


namespace negation_of_proposition_l50_50398

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end negation_of_proposition_l50_50398


namespace probability_first_two_heads_l50_50950

-- The probability of getting heads in a single flip of a fair coin
def probability_heads_single_flip : ℚ := 1 / 2

-- Independence of coin flips
def independent_flips {α : Type} (p : α → Prop) := ∀ a b : α, a ≠ b → p a ∧ p b

-- The event of getting heads on a coin flip
def heads_event : Prop := true

-- Problem statement: The probability that the first two flips are both heads
theorem probability_first_two_heads : probability_heads_single_flip * probability_heads_single_flip = 1 / 4 :=
by
  sorry

end probability_first_two_heads_l50_50950


namespace correct_option_e_l50_50318

theorem correct_option_e : 15618 = 1 + 5^6 - 1 * 8 :=
by sorry

end correct_option_e_l50_50318


namespace solve_inequality_l50_50620

theorem solve_inequality (x: ℝ) : (25 - 5 * Real.sqrt 3) ≤ x ∧ x ≤ (25 + 5 * Real.sqrt 3) ↔ x ^ 2 - 50 * x + 575 ≤ 25 :=
by
  sorry

end solve_inequality_l50_50620


namespace mans_rate_in_still_water_l50_50693

theorem mans_rate_in_still_water : 
  ∀ (V_m V_s : ℝ), 
  V_m + V_s = 16 → 
  V_m - V_s = 4 → 
  V_m = 10 :=
by
  intros V_m V_s h1 h2
  sorry

end mans_rate_in_still_water_l50_50693


namespace max_cubes_fit_in_box_l50_50235

theorem max_cubes_fit_in_box :
  ∀ (h w l : ℕ) (cube_vol box_max_cubes : ℕ),
    h = 12 → w = 8 → l = 9 → cube_vol = 27 → 
    box_max_cubes = (h * w * l) / cube_vol → box_max_cubes = 32 :=
by
  intros h w l cube_vol box_max_cubes h_def w_def l_def cube_vol_def box_max_cubes_def
  sorry

end max_cubes_fit_in_box_l50_50235


namespace carnations_count_l50_50122

theorem carnations_count (total_flowers : ℕ) (fract_rose : ℚ) (num_tulips : ℕ) (h1 : total_flowers = 40) (h2 : fract_rose = 2 / 5) (h3 : num_tulips = 10) :
  total_flowers - ((fract_rose * total_flowers) + num_tulips) = 14 := 
by
  sorry

end carnations_count_l50_50122


namespace first_discount_percentage_l50_50895

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (additional_discount : ℝ) (first_discount : ℝ) : 
  original_price = 600 → final_price = 513 → additional_discount = 0.05 →
  600 * (1 - first_discount / 100) * (1 - 0.05) = 513 →
  first_discount = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end first_discount_percentage_l50_50895


namespace how_many_more_choc_chip_cookies_l50_50362

-- Define the given conditions
def choc_chip_cookies_yesterday := 19
def raisin_cookies_this_morning := 231
def choc_chip_cookies_this_morning := 237

-- Define the total chocolate chip cookies
def total_choc_chip_cookies : ℕ := choc_chip_cookies_this_morning + choc_chip_cookies_yesterday

-- Define the proof statement
theorem how_many_more_choc_chip_cookies :
  total_choc_chip_cookies - raisin_cookies_this_morning = 25 :=
by
  -- Proof will go here
  sorry

end how_many_more_choc_chip_cookies_l50_50362


namespace vertex_closest_point_l50_50785

theorem vertex_closest_point (a : ℝ) (x y : ℝ) :
  (x^2 = 2 * y) ∧ (y ≥ 0) ∧ ((y^2 + 2 * (1 - a) * y + a^2) ≤ 0) → a ≤ 1 :=
by 
  sorry

end vertex_closest_point_l50_50785


namespace upper_bound_neg_expr_l50_50752

theorem upper_bound_neg_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  - (1 / (2 * a) + 2 / b) ≤ - (9 / 2) := 
sorry

end upper_bound_neg_expr_l50_50752


namespace area_of_fourth_rectangle_l50_50612

theorem area_of_fourth_rectangle (a b c d : ℕ) (h1 : a = 18) (h2 : b = 27) (h3 : c = 12) :
d = 93 :=
by
  -- Problem reduces to showing that d equals 93 using the given h1, h2, h3
  sorry

end area_of_fourth_rectangle_l50_50612


namespace rhombus_area_l50_50867

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by
  sorry

end rhombus_area_l50_50867


namespace simple_interest_principal_l50_50646

theorem simple_interest_principal (R : ℝ) (P : ℝ) (h : P * 7 * (R + 2) / 100 = P * 7 * R / 100 + 140) : P = 1000 :=
by
  sorry

end simple_interest_principal_l50_50646


namespace length_of_bridge_l50_50929

theorem length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_to_cross : ℕ)
  (lt : length_of_train = 140)
  (st : speed_of_train_kmh = 45)
  (tc : time_to_cross = 30) : 
  ∃ length_of_bridge, length_of_bridge = 235 := 
by 
  sorry

end length_of_bridge_l50_50929


namespace symmetric_slope_angle_l50_50937

theorem symmetric_slope_angle (α₁ : ℝ)
  (hα₁ : 0 ≤ α₁ ∧ α₁ < Real.pi) :
  ∃ α₂ : ℝ, (α₁ < Real.pi / 2 → α₂ = Real.pi - α₁) ∧
            (α₁ = Real.pi / 2 → α₂ = 0) :=
sorry

end symmetric_slope_angle_l50_50937


namespace shorter_side_length_l50_50994

variables (x y : ℝ)
variables (h1 : 2 * x + 2 * y = 60)
variables (h2 : x * y = 200)

theorem shorter_side_length :
  min x y = 10 :=
by
  sorry

end shorter_side_length_l50_50994


namespace tangent_line_equation_l50_50142

open Real

noncomputable def circle_center : ℝ × ℝ := (2, 1)
noncomputable def tangent_point : ℝ × ℝ := (4, 3)

def circle_equation (x y : ℝ) : Prop := (x - circle_center.1)^2 + (y - circle_center.2)^2 = 1

theorem tangent_line_equation :
  ∀ (x y : ℝ), ( (x = 4 ∧ y = 3) ∨ circle_equation x y ) → 2 * x + 2 * y - 7 = 0 :=
sorry

end tangent_line_equation_l50_50142


namespace binom_10_2_eq_45_l50_50618

-- Definitions used in the conditions
def binom (n k : ℕ) := n.choose k

-- The statement that needs to be proven
theorem binom_10_2_eq_45 : binom 10 2 = 45 :=
by
  sorry

end binom_10_2_eq_45_l50_50618


namespace hexagon_angles_sum_l50_50242

theorem hexagon_angles_sum (mA mB mC : ℤ) (x y : ℤ)
  (hA : mA = 35) (hB : mB = 80) (hC : mC = 30)
  (hSum : (6 - 2) * 180 = 720)
  (hAdjacentA : 90 + 90 = 180)
  (hAdjacentC : 90 - mC = 60) :
  x + y = 95 := by
  sorry

end hexagon_angles_sum_l50_50242


namespace solve_for_x_l50_50182

theorem solve_for_x (x y : ℝ) (h₁ : y = (x^2 - 9) / (x - 3)) (h₂ : y = 3 * x - 4) : x = 7 / 2 :=
by sorry

end solve_for_x_l50_50182


namespace total_material_ordered_l50_50222

theorem total_material_ordered (c b s : ℝ) (hc : c = 0.17) (hb : b = 0.17) (hs : s = 0.5) :
  c + b + s = 0.84 :=
by sorry

end total_material_ordered_l50_50222


namespace suggested_bacon_students_l50_50451

-- Definitions based on the given conditions
def students_mashed_potatoes : ℕ := 330
def students_tomatoes : ℕ := 76
def difference_bacon_mashed_potatoes : ℕ := 61

-- Lean 4 statement to prove the correct answer
theorem suggested_bacon_students : ∃ (B : ℕ), students_mashed_potatoes = B + difference_bacon_mashed_potatoes ∧ B = 269 := 
by
  sorry

end suggested_bacon_students_l50_50451


namespace algebraic_expression_value_l50_50274

theorem algebraic_expression_value (p q : ℤ) 
  (h : 8 * p + 2 * q = -2023) : 
  (p * (-2) ^ 3 + q * (-2) + 1 = 2024) :=
by
  sorry

end algebraic_expression_value_l50_50274


namespace paving_cost_correct_l50_50302

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_m : ℝ := 300
def area (length : ℝ) (width : ℝ) : ℝ := length * width
def cost (area : ℝ) (rate : ℝ) : ℝ := area * rate

theorem paving_cost_correct :
  cost (area length width) rate_per_sq_m = 6187.50 :=
by
  sorry

end paving_cost_correct_l50_50302


namespace polynomial_product_l50_50682

noncomputable def sum_of_coefficients (g h : ℤ) : ℤ := g + h

theorem polynomial_product (g h : ℤ) :
  (9 * d^3 - 5 * d^2 + g) * (4 * d^2 + h * d - 9) = 36 * d^5 - 11 * d^4 - 49 * d^3 + 45 * d^2 - 9 * d →
  sum_of_coefficients g h = 18 :=
by
  intro
  sorry

end polynomial_product_l50_50682


namespace fg_at_3_equals_97_l50_50078

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_at_3_equals_97 : f (g 3) = 97 := by
  sorry

end fg_at_3_equals_97_l50_50078


namespace total_pencils_l50_50578

def pencils_in_rainbow_box : ℕ := 7
def total_people : ℕ := 8

theorem total_pencils : pencils_in_rainbow_box * total_people = 56 := by
  sorry

end total_pencils_l50_50578


namespace price_change_on_eggs_and_apples_l50_50482

theorem price_change_on_eggs_and_apples :
  let initial_egg_price := 1.00
  let initial_apple_price := 1.00
  let egg_drop_percent := 0.10
  let apple_increase_percent := 0.02
  let new_egg_price := initial_egg_price * (1 - egg_drop_percent)
  let new_apple_price := initial_apple_price * (1 + apple_increase_percent)
  let initial_total := initial_egg_price + initial_apple_price
  let new_total := new_egg_price + new_apple_price
  let percent_change := ((new_total - initial_total) / initial_total) * 100
  percent_change = -4 :=
by
  sorry

end price_change_on_eggs_and_apples_l50_50482


namespace negation_equiv_l50_50333

-- Given problem conditions
def exists_real_x_lt_0 : Prop := ∃ x : ℝ, x^2 + 1 < 0

-- Mathematically equivalent proof problem statement
theorem negation_equiv :
  ¬exists_real_x_lt_0 ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l50_50333


namespace ratio_of_sequences_is_5_over_4_l50_50724

-- Definitions of arithmetic sequences
def arithmetic_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Hypotheses
def sequence_1_sum : ℕ :=
  arithmetic_sum 5 5 16

def sequence_2_sum : ℕ :=
  arithmetic_sum 4 4 16

-- Main statement to be proven
theorem ratio_of_sequences_is_5_over_4 : sequence_1_sum / sequence_2_sum = 5 / 4 := sorry

end ratio_of_sequences_is_5_over_4_l50_50724


namespace rect_area_162_l50_50231

def rectangle_field_area (w l : ℝ) (A : ℝ) : Prop :=
  w = (1/2) * l ∧ 2 * (w + l) = 54 ∧ A = w * l

theorem rect_area_162 {w l A : ℝ} :
  rectangle_field_area w l A → A = 162 :=
by
  intro h
  sorry

end rect_area_162_l50_50231


namespace same_type_as_target_l50_50576

-- Definitions of the polynomials
def optionA (a b : ℝ) : ℝ := a^2 * b
def optionB (a b : ℝ) : ℝ := -2 * a * b^2
def optionC (a b : ℝ) : ℝ := a * b
def optionD (a b c : ℝ) : ℝ := a * b^2 * c

-- Definition of the target polynomial type
def target (a b : ℝ) : ℝ := a * b^2

-- Statement: Option B is of the same type as target
theorem same_type_as_target (a b : ℝ) : optionB a b = -2 * target a b := 
sorry

end same_type_as_target_l50_50576


namespace kiwis_to_apples_l50_50557

theorem kiwis_to_apples :
  (1 / 4) * 20 = 10 → (3 / 4) * 12 * (2 / 5) = 18 :=
by
  sorry

end kiwis_to_apples_l50_50557


namespace solve_for_x_l50_50582

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end solve_for_x_l50_50582


namespace least_positive_integer_y_l50_50456

theorem least_positive_integer_y (x k y: ℤ) (h1: 24 * x + k * y = 4) (h2: ∃ x: ℤ, ∃ y: ℤ, 24 * x + k * y = 4) : y = 4 :=
sorry

end least_positive_integer_y_l50_50456


namespace number_of_shirts_is_39_l50_50382

-- Define the conditions as Lean definitions.
def washing_machine_capacity : ℕ := 8
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 9

-- Define the total number of pieces of clothing based on the conditions.
def total_pieces_of_clothing : ℕ :=
  number_of_loads * washing_machine_capacity

-- Define the number of shirts.
noncomputable def number_of_shirts : ℕ :=
  total_pieces_of_clothing - number_of_sweaters

-- The actual proof problem statement.
theorem number_of_shirts_is_39 :
  number_of_shirts = 39 := by
  sorry

end number_of_shirts_is_39_l50_50382


namespace vectors_opposite_direction_l50_50147

noncomputable def a : ℝ × ℝ := (-2, 4)
noncomputable def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction : a = (-2 : ℝ) • b :=
by
  sorry

end vectors_opposite_direction_l50_50147


namespace average_of_multiples_l50_50090

theorem average_of_multiples :
  let sum_of_first_7_multiples_of_9 := 9 + 18 + 27 + 36 + 45 + 54 + 63
  let sum_of_first_5_multiples_of_11 := 11 + 22 + 33 + 44 + 55
  let sum_of_first_3_negative_multiples_of_13 := -13 + -26 + -39
  let total_sum := sum_of_first_7_multiples_of_9 + sum_of_first_5_multiples_of_11 + sum_of_first_3_negative_multiples_of_13
  let average := total_sum / 3
  average = 113 :=
by
  sorry

end average_of_multiples_l50_50090


namespace jenny_grade_l50_50209

theorem jenny_grade (J A B : ℤ) 
  (hA : A = J - 25) 
  (hB : B = A / 2) 
  (hB_val : B = 35) : 
  J = 95 :=
by
  sorry

end jenny_grade_l50_50209


namespace center_in_triangle_probability_l50_50036

theorem center_in_triangle_probability (n : ℕ) :
  let vertices := 2 * n + 1
  let total_ways := vertices.choose 3
  let no_center_ways := vertices * (n.choose 2) / 2
  let p_no_center := no_center_ways / total_ways
  let p_center := 1 - p_no_center
  p_center = (n + 1) / (4 * n - 2) := sorry

end center_in_triangle_probability_l50_50036


namespace min_value_inverse_sum_l50_50352

theorem min_value_inverse_sum {m n : ℝ} (h1 : -2 * m - 2 * n + 1 = 0) (h2 : m * n > 0) : 
  (1 / m + 1 / n) ≥ 8 :=
sorry

end min_value_inverse_sum_l50_50352


namespace maximize_revenue_l50_50083

noncomputable def revenue (p : ℝ) : ℝ :=
p * (145 - 7 * p)

theorem maximize_revenue : ∃ p : ℕ, p ≤ 30 ∧ p = 10 ∧ ∀ q ≤ 30, revenue (q : ℝ) ≤ revenue 10 :=
by
  sorry

end maximize_revenue_l50_50083


namespace solve_inequality_system_l50_50055

-- Define the inequalities as conditions.
def cond1 (x : ℝ) := 2 * x + 1 < 3 * x - 2
def cond2 (x : ℝ) := 3 * (x - 2) - x ≤ 4

-- Formulate the theorem to prove that these conditions give the solution 3 < x ≤ 5.
theorem solve_inequality_system (x : ℝ) : cond1 x ∧ cond2 x ↔ 3 < x ∧ x ≤ 5 := 
sorry

end solve_inequality_system_l50_50055


namespace H2CO3_formation_l50_50659

-- Define the given conditions
def one_to_one_reaction (a b : ℕ) := a = b

-- Define the reaction
theorem H2CO3_formation (m_CO2 m_H2O : ℕ) 
  (h : one_to_one_reaction m_CO2 m_H2O) : 
  m_CO2 = 2 → m_H2O = 2 → m_CO2 = 2 ∧ m_H2O = 2 := 
by 
  intros h1 h2
  exact ⟨h1, h2⟩

end H2CO3_formation_l50_50659


namespace find_values_of_m_and_n_l50_50739

theorem find_values_of_m_and_n (m n : ℝ) (h : m / (1 + I) = 1 - n * I) : 
  m = 2 ∧ n = 1 :=
sorry

end find_values_of_m_and_n_l50_50739


namespace max_area_triangle_l50_50365

theorem max_area_triangle (a b c S : ℝ) (h₁ : S = a^2 - (b - c)^2) (h₂ : b + c = 8) :
  S ≤ 64 / 17 :=
sorry

end max_area_triangle_l50_50365


namespace find_value_l50_50882

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom explicit_form : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem statement
theorem find_value : f (-5/2) = -1/2 :=
by
  -- Here would be the place to start the proof based on the above axioms
  sorry

end find_value_l50_50882


namespace simplify_expression_l50_50706

theorem simplify_expression : 8^5 + 8^5 + 8^5 + 8^5 = 8^(17/3) :=
by
  -- Proof will be completed here
  sorry

end simplify_expression_l50_50706


namespace inequality_of_f_l50_50939

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem inequality_of_f (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ :=
by
  -- sorry placeholder for the actual proof
  sorry

end inequality_of_f_l50_50939


namespace intersection_complement_l50_50007

open Set

variable (U : Set ℕ) (P Q : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (hP : P = {1, 2, 3, 4, 5})
variable (hQ : Q = {3, 4, 5, 6, 7})

theorem intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_l50_50007


namespace number_of_distinct_collections_l50_50981

def mathe_matical_letters : Multiset Char :=
  {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

def vowels : Multiset Char :=
  {'A', 'A', 'A', 'E', 'I'}

def consonants : Multiset Char :=
  {'M', 'T', 'H', 'M', 'T', 'C', 'L', 'C'}

def indistinguishable (s : Multiset Char) :=
  (s.count 'A' = s.count 'A' ∧
   s.count 'E' = 1 ∧
   s.count 'I' = 1 ∧
   s.count 'M' = 2 ∧
   s.count 'T' = 2 ∧
   s.count 'H' = 1 ∧
   s.count 'C' = 2 ∧
   s.count 'L' = 1)

theorem number_of_distinct_collections :
  5 * 16 = 80 :=
by
  -- proof would go here
  sorry

end number_of_distinct_collections_l50_50981


namespace solve_quadratic_equation_l50_50754

theorem solve_quadratic_equation : ∀ x : ℝ, x * (x - 14) = 0 ↔ x = 0 ∨ x = 14 :=
by
  sorry

end solve_quadratic_equation_l50_50754


namespace quadrilateral_tile_angles_l50_50296

theorem quadrilateral_tile_angles :
  ∃ a b c d : ℝ, a + b + c + d = 360 ∧ a = 45 ∧ b = 60 ∧ c = 105 ∧ d = 150 := 
by {
  sorry
}

end quadrilateral_tile_angles_l50_50296


namespace value_of_expression_at_x_eq_2_l50_50969

theorem value_of_expression_at_x_eq_2 :
  (2 * (2: ℕ)^2 - 3 * 2 + 4 = 6) := 
by sorry

end value_of_expression_at_x_eq_2_l50_50969


namespace solve_for_x_l50_50598

theorem solve_for_x (x : ℕ) : (8^3 + 8^3 + 8^3 + 8^3 = 2^x) → x = 11 :=
by
  intro h
  sorry

end solve_for_x_l50_50598


namespace photo_gallery_total_l50_50838

theorem photo_gallery_total (initial_photos: ℕ) (first_day_photos: ℕ) (second_day_photos: ℕ)
  (h_initial: initial_photos = 400) 
  (h_first_day: first_day_photos = initial_photos / 2)
  (h_second_day: second_day_photos = first_day_photos + 120) : 
  initial_photos + first_day_photos + second_day_photos = 920 :=
by
  sorry

end photo_gallery_total_l50_50838


namespace wire_length_l50_50224

theorem wire_length (S L W : ℝ) (h1 : S = 20) (h2 : S = (2 / 7) * L) (h3 : W = S + L) : W = 90 :=
by sorry

end wire_length_l50_50224


namespace real_root_solution_l50_50966

theorem real_root_solution (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ∃ x1 x2 : ℝ, 
    (x1 < b ∧ b < x2) ∧
    (1 / (x1 - a) + 1 / (x1 - b) + 1 / (x1 - c) = 0) ∧ 
    (1 / (x2 - a) + 1 / (x2 - b) + 1 / (x2 - c) = 0) :=
by
  sorry

end real_root_solution_l50_50966


namespace total_wheels_is_90_l50_50364

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end total_wheels_is_90_l50_50364


namespace worth_of_each_gift_is_4_l50_50277

noncomputable def worth_of_each_gift
  (workers_per_block : ℕ)
  (total_blocks : ℕ)
  (total_amount : ℝ) : ℝ :=
  total_amount / (workers_per_block * total_blocks)

theorem worth_of_each_gift_is_4 (workers_per_block total_blocks : ℕ) (total_amount : ℝ)
  (h1 : workers_per_block = 100)
  (h2 : total_blocks = 10)
  (h3 : total_amount = 4000) :
  worth_of_each_gift workers_per_block total_blocks total_amount = 4 :=
by
  sorry

end worth_of_each_gift_is_4_l50_50277


namespace count_whole_numbers_between_cuberoots_l50_50788

theorem count_whole_numbers_between_cuberoots : 
  ∃ (n : ℕ), n = 7 ∧ 
      ∀ x : ℝ, (3 < x ∧ x < 4 → ∃ k : ℕ, k = 4) ∧ 
                (9 < x ∧ x ≤ 10 → ∃ k : ℕ, k = 10) :=
sorry

end count_whole_numbers_between_cuberoots_l50_50788


namespace rockham_soccer_league_members_count_l50_50212

def cost_per_pair_of_socks : Nat := 4
def additional_cost_per_tshirt : Nat := 5
def cost_per_tshirt : Nat := cost_per_pair_of_socks + additional_cost_per_tshirt

def pairs_of_socks_per_member : Nat := 2
def tshirts_per_member : Nat := 2

def total_cost_per_member : Nat :=
  pairs_of_socks_per_member * cost_per_pair_of_socks + tshirts_per_member * cost_per_tshirt

def total_cost_all_members : Nat := 2366
def total_members : Nat := total_cost_all_members / total_cost_per_member

theorem rockham_soccer_league_members_count : total_members = 91 :=
by
  -- Given steps in the solution, verify each condition and calculation.
  sorry

end rockham_soccer_league_members_count_l50_50212


namespace find_a_plus_b_l50_50911

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 := 
by 
  sorry

end find_a_plus_b_l50_50911


namespace find_starting_number_l50_50057

theorem find_starting_number (num_even_ints: ℕ) (end_num: ℕ) (h_num: num_even_ints = 35) (h_end: end_num = 95) : 
  ∃ start_num: ℕ, start_num = 24 ∧ (∀ n: ℕ, (start_num + 2 * n ≤ end_num ∧ n < num_even_ints)) := by
  sorry

end find_starting_number_l50_50057


namespace stock_price_end_of_second_year_l50_50273

def initial_price : ℝ := 80
def first_year_increase_rate : ℝ := 1.2
def second_year_decrease_rate : ℝ := 0.3

theorem stock_price_end_of_second_year : 
  initial_price * (1 + first_year_increase_rate) * (1 - second_year_decrease_rate) = 123.2 := 
by sorry

end stock_price_end_of_second_year_l50_50273


namespace factory_hours_per_day_l50_50892

def hour_worked_forth_machine := 12
def production_rate_per_hour := 2
def selling_price_per_kg := 50
def total_earnings := 8100

def h := 23

theorem factory_hours_per_day
  (num_machines : ℕ)
  (num_machines := 3)
  (production_first_three : ℕ := num_machines * production_rate_per_hour * h)
  (production_fourth : ℕ := hour_worked_forth_machine * production_rate_per_hour)
  (total_production : ℕ := production_first_three + production_fourth)
  (total_earnings_eq : total_production * selling_price_per_kg = total_earnings) :
  h = 23 := by
  sorry

end factory_hours_per_day_l50_50892


namespace problem_l50_50652

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x : ℝ, f x a ≤ 2) : f (π / 6) a = -1 :=
by {
  sorry
}

end problem_l50_50652


namespace special_four_digit_numbers_l50_50604

noncomputable def count_special_four_digit_numbers : Nat :=
  -- The task is to define the number of four-digit numbers formed using the digits {0, 1, 2, 3, 4}
  -- that contain the digit 0 and have exactly two digits repeating
  144

theorem special_four_digit_numbers : count_special_four_digit_numbers = 144 := by
  sorry

end special_four_digit_numbers_l50_50604


namespace neg_one_exponent_difference_l50_50279

theorem neg_one_exponent_difference : (-1 : ℤ) ^ 2004 - (-1 : ℤ) ^ 2003 = 2 := by
  sorry

end neg_one_exponent_difference_l50_50279


namespace vans_for_field_trip_l50_50026

-- Definitions based on conditions
def students := 25
def adults := 5
def van_capacity := 5

-- Calculate total number of people
def total_people := students + adults

-- Calculate number of vans needed
def vans_needed := total_people / van_capacity

-- Theorem statement
theorem vans_for_field_trip : vans_needed = 6 := by
  -- Proof would go here
  sorry

end vans_for_field_trip_l50_50026


namespace area_of_paper_is_500_l50_50584

-- Define the width and length of the rectangular drawing paper
def width := 25
def length := 20

-- Define the formula for the area of a rectangle
def area (w : Nat) (l : Nat) : Nat := w * l

-- Prove that the area of the paper is 500 square centimeters
theorem area_of_paper_is_500 : area width length = 500 := by
  -- placeholder for the proof
  sorry

end area_of_paper_is_500_l50_50584


namespace goods_train_speed_l50_50806

theorem goods_train_speed (length_train length_platform distance time : ℕ) (conversion_factor : ℚ) : 
  length_train = 250 → 
  length_platform = 270 → 
  distance = length_train + length_platform → 
  time = 26 → 
  conversion_factor = 3.6 →
  (distance / time : ℚ) * conversion_factor = 72 :=
by
  intros h_lt h_lp h_d h_t h_cf
  rw [h_lt, h_lp] at h_d
  rw [h_t, h_cf]
  sorry

end goods_train_speed_l50_50806


namespace sales_in_second_month_l50_50993

-- Given conditions:
def sales_first_month : ℕ := 6400
def sales_third_month : ℕ := 6800
def sales_fourth_month : ℕ := 7200
def sales_fifth_month : ℕ := 6500
def sales_sixth_month : ℕ := 5100
def average_sales : ℕ := 6500

-- Statement to prove:
theorem sales_in_second_month :
  ∃ (sales_second_month : ℕ), 
    average_sales * 6 = sales_first_month + sales_second_month + sales_third_month 
    + sales_fourth_month + sales_fifth_month + sales_sixth_month 
    ∧ sales_second_month = 7000 :=
  sorry

end sales_in_second_month_l50_50993


namespace g_four_times_of_three_l50_50072

noncomputable def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_four_times_of_three :
  g (g (g (g 3))) = 3 := by
  sorry

end g_four_times_of_three_l50_50072


namespace count_lineups_not_last_l50_50347

theorem count_lineups_not_last (n : ℕ) (htallest_not_last : n = 5) :
  ∃ (k : ℕ), k = 96 :=
by { sorry }

end count_lineups_not_last_l50_50347


namespace sum_of_consecutive_integers_l50_50748

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 14) : a + b + c = 39 := 
by 
  sorry

end sum_of_consecutive_integers_l50_50748


namespace largest_divisor_of_expression_l50_50303

theorem largest_divisor_of_expression :
  ∃ k : ℕ, (∀ m : ℕ, (m > k → m ∣ (1991 ^ k * 1990 ^ (1991 ^ 1992) + 1992 ^ (1991 ^ 1990)) = false))
  ∧ k = 1991 := by
sorry

end largest_divisor_of_expression_l50_50303


namespace sum_of_first_n_natural_numbers_l50_50570

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers_l50_50570


namespace decreased_cost_proof_l50_50160

def original_cost : ℝ := 200
def percentage_decrease : ℝ := 0.5
def decreased_cost (original_cost : ℝ) (percentage_decrease : ℝ) : ℝ := 
  original_cost - (percentage_decrease * original_cost)

theorem decreased_cost_proof : decreased_cost original_cost percentage_decrease = 100 := 
by { 
  sorry -- Proof is not required
}

end decreased_cost_proof_l50_50160


namespace tom_needs_495_boxes_l50_50918

-- Define the conditions
def total_chocolate_bars : ℕ := 3465
def chocolate_bars_per_box : ℕ := 7

-- Define the proof statement
theorem tom_needs_495_boxes : total_chocolate_bars / chocolate_bars_per_box = 495 :=
by
  sorry

end tom_needs_495_boxes_l50_50918


namespace total_blocks_l50_50767

-- Conditions
def original_blocks : ℝ := 35.0
def added_blocks : ℝ := 65.0

-- Question and proof goal
theorem total_blocks : original_blocks + added_blocks = 100.0 := 
by
  -- The proof would be provided here
  sorry

end total_blocks_l50_50767


namespace ellipse_properties_l50_50645

noncomputable def a_square : ℝ := 2
noncomputable def b_square : ℝ := 9 / 8
noncomputable def c_square : ℝ := a_square - b_square
noncomputable def c : ℝ := Real.sqrt c_square
noncomputable def distance_between_foci : ℝ := 2 * c
noncomputable def eccentricity : ℝ := c / Real.sqrt a_square

theorem ellipse_properties :
  (distance_between_foci = Real.sqrt 14) ∧ (eccentricity = Real.sqrt 7 / 4) := by
  sorry

end ellipse_properties_l50_50645


namespace max_sum_red_green_balls_l50_50990

theorem max_sum_red_green_balls (total_balls : ℕ) (green_balls : ℕ) (max_red_balls : ℕ) 
  (h1 : total_balls = 28) (h2 : green_balls = 12) (h3 : max_red_balls ≤ 11) : 
  (max_red_balls + green_balls) = 23 := 
sorry

end max_sum_red_green_balls_l50_50990


namespace percentage_land_mr_william_l50_50840

noncomputable def tax_rate_arable := 0.01
noncomputable def tax_rate_orchard := 0.02
noncomputable def tax_rate_pasture := 0.005

noncomputable def subsidy_arable := 100
noncomputable def subsidy_orchard := 50
noncomputable def subsidy_pasture := 20

noncomputable def total_tax_village := 3840
noncomputable def tax_mr_william := 480

theorem percentage_land_mr_william : 
  (tax_mr_william / total_tax_village : ℝ) * 100 = 12.5 :=
by
  sorry

end percentage_land_mr_william_l50_50840


namespace complex_problem_l50_50068

theorem complex_problem (a b : ℝ) (h : (⟨a, 3⟩ : ℂ) + ⟨2, -1⟩ = ⟨5, b⟩) : a * b = 6 := by
  sorry

end complex_problem_l50_50068


namespace find_square_l50_50009

theorem find_square (q x : ℝ) 
  (h1 : x + q = 74) 
  (h2 : x + 2 * q^2 = 180) : 
  x = 66 :=
by {
  sorry
}

end find_square_l50_50009


namespace evaluate_exponent_l50_50885

theorem evaluate_exponent : (3^3)^2 = 729 := by
  sorry

end evaluate_exponent_l50_50885


namespace min_value_of_expression_l50_50794

noncomputable def minimum_value_expression : ℝ :=
  let f (a b : ℝ) := a^4 + b^4 + 16 / (a^2 + b^2)^2
  4

theorem min_value_of_expression (a b : ℝ) (h : 0 < a ∧ 0 < b) : 
  let f := a^4 + b^4 + 16 / (a^2 + b^2)^2
  ∃ c : ℝ, f = c ∧ c = 4 :=
sorry

end min_value_of_expression_l50_50794


namespace arctan_sum_l50_50394

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end arctan_sum_l50_50394


namespace tromino_covering_l50_50177

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def chessboard_black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

def minimum_trominos (n : ℕ) : ℕ := (n^2 + 1) / 6

theorem tromino_covering (n : ℕ) (h_odd : is_odd n) (h_ge7 : n ≥ 7) :
  ∃ k : ℕ, chessboard_black_squares n = 3 * k ∧ (k = minimum_trominos n) :=
sorry

end tromino_covering_l50_50177


namespace total_pennies_thrown_l50_50158

theorem total_pennies_thrown (R G X M T : ℝ) (hR : R = 1500)
  (hG : G = (2 / 3) * R) (hX : X = (3 / 4) * G) 
  (hM : M = 3.5 * X) (hT : T = (4 / 5) * M) : 
  R + G + X + M + T = 7975 :=
by
  sorry

end total_pennies_thrown_l50_50158


namespace find_w_l50_50000

theorem find_w (a w : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * w) : w = 49 :=
by
  sorry

end find_w_l50_50000


namespace grace_crayon_selection_l50_50168

def crayons := {i // 1 ≤ i ∧ i ≤ 15}
def red_crayons := {i // 1 ≤ i ∧ i ≤ 3}

def total_ways := Nat.choose 15 5
def non_favorable := Nat.choose 12 5

theorem grace_crayon_selection : total_ways - non_favorable = 2211 :=
by
  sorry

end grace_crayon_selection_l50_50168


namespace cannot_be_correct_average_l50_50148

theorem cannot_be_correct_average (a : ℝ) (h_pos : a > 0) (h_median : a ≤ 12) : 
  ∀ avg, avg = (12 + a + 8 + 15 + 23) / 5 → avg ≠ 71 / 5 := 
by
  intro avg h_avg
  sorry

end cannot_be_correct_average_l50_50148


namespace range_of_a_l50_50714

-- Defining the propositions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

-- Main theorem statement
theorem range_of_a (a : ℝ) : (¬(∃ x, p x) → ¬(∃ x, q x a)) → a < -3 :=
by
  sorry

end range_of_a_l50_50714


namespace fraction_power_rule_example_l50_50769

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l50_50769


namespace Mike_additional_money_needed_proof_l50_50624

-- Definitions of conditions
def phone_cost : ℝ := 1300
def smartwatch_cost : ℝ := 500
def phone_discount : ℝ := 0.10
def smartwatch_discount : ℝ := 0.15
def sales_tax : ℝ := 0.07
def mike_has_percentage : ℝ := 0.40

-- Definitions of intermediate calculations
def discounted_phone_cost : ℝ := phone_cost * (1 - phone_discount)
def discounted_smartwatch_cost : ℝ := smartwatch_cost * (1 - smartwatch_discount)
def total_cost_before_tax : ℝ := discounted_phone_cost + discounted_smartwatch_cost
def total_tax : ℝ := total_cost_before_tax * sales_tax
def total_cost_after_tax : ℝ := total_cost_before_tax + total_tax
def mike_has_amount : ℝ := total_cost_after_tax * mike_has_percentage
def additional_money_needed : ℝ := total_cost_after_tax - mike_has_amount

-- Theorem statement
theorem Mike_additional_money_needed_proof :
  additional_money_needed = 1023.99 :=
by sorry

end Mike_additional_money_needed_proof_l50_50624


namespace sedrach_divides_each_pie_l50_50129

theorem sedrach_divides_each_pie (P : ℕ) :
  (13 * P * 5 = 130) → P = 2 :=
by
  sorry

end sedrach_divides_each_pie_l50_50129


namespace bags_sold_in_afternoon_l50_50807

theorem bags_sold_in_afternoon (bags_morning : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) 
  (h1 : bags_morning = 29) (h2 : weight_per_bag = 7) (h3 : total_weight = 322) : 
  total_weight - bags_morning * weight_per_bag / weight_per_bag = 17 := 
by 
  sorry

end bags_sold_in_afternoon_l50_50807


namespace sum_of_coefficients_of_factorized_polynomial_l50_50845

theorem sum_of_coefficients_of_factorized_polynomial : 
  ∃ (a b c d e : ℕ), 
    (216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ 
    (a + b + c + d + e = 36) :=
sorry

end sum_of_coefficients_of_factorized_polynomial_l50_50845


namespace general_term_of_sequence_l50_50018

def S (n : ℕ) : ℕ := n^2 + 3 * n + 1

def a (n : ℕ) : ℕ := 
  if n = 1 then 5 
  else 2 * n + 2

theorem general_term_of_sequence (n : ℕ) : 
  a n = if n = 1 then 5 else (S n - S (n - 1)) := 
by 
  sorry

end general_term_of_sequence_l50_50018


namespace not_divisible_by_4_l50_50497

theorem not_divisible_by_4 (n : Int) : ¬ (1 + n + n^2 + n^3 + n^4) % 4 = 0 := by
  sorry

end not_divisible_by_4_l50_50497


namespace difference_students_l50_50408

variables {A B AB : ℕ}

theorem difference_students (h1 : A + AB + B = 800)
  (h2 : AB = 20 * (A + AB) / 100)
  (h3 : AB = 25 * (B + AB) / 100) :
  A - B = 100 :=
sorry

end difference_students_l50_50408


namespace hyperbola_satisfies_m_l50_50359

theorem hyperbola_satisfies_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - m * y^2 = 1)
  (h2 : ∀ a b : ℝ, (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2*a = 2 * 2*b)) : 
  m = 4 := 
sorry

end hyperbola_satisfies_m_l50_50359


namespace speed_with_current_l50_50132

-- Define the constants
def speed_of_current : ℝ := 2.5
def speed_against_current : ℝ := 20

-- Define the man's speed in still water
axiom speed_in_still_water : ℝ
axiom speed_against_current_eq : speed_in_still_water - speed_of_current = speed_against_current

-- The statement we need to prove
theorem speed_with_current : speed_in_still_water + speed_of_current = 25 := sorry

end speed_with_current_l50_50132


namespace cylinder_volume_ratio_l50_50498

theorem cylinder_volume_ratio (w h : ℝ) (w_pos : w = 6) (h_pos : h = 9) :
  let r1 := w / (2 * Real.pi)
  let h1 := h
  let V1 := Real.pi * r1^2 * h1
  let r2 := h / (2 * Real.pi)
  let h2 := w
  let V2 := Real.pi * r2^2 * h2
  V2 / V1 = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l50_50498


namespace S6_values_l50_50311

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

axiom geo_seq (q : ℝ) :
  ∀ n : ℕ, a n = a 0 * q ^ n

variable (a3_eq_4 : a 2 = 4) 
variable (S3_eq_7 : S 3 = 7)

theorem S6_values : S 6 = 63 ∨ S 6 = 133 / 27 := sorry

end S6_values_l50_50311


namespace increase_in_votes_l50_50234

noncomputable def initial_vote_for (y : ℝ) : ℝ := 500 - y
noncomputable def revote_for (y : ℝ) : ℝ := (10 / 9) * y

theorem increase_in_votes {x x' y m : ℝ}
  (H1 : x + y = 500)
  (H2 : y - x = m)
  (H3 : x' - y = 2 * m)
  (H4 : x' + y = 500)
  (H5 : x' = (10 / 9) * y)
  (H6 : y = 282) :
  revote_for y - initial_vote_for y = 95 :=
by sorry

end increase_in_votes_l50_50234


namespace find_first_number_l50_50525

theorem find_first_number (sum_is_33 : ∃ x y : ℕ, x + y = 33) (second_is_twice_first : ∃ x y : ℕ, y = 2 * x) (second_is_22 : ∃ y : ℕ, y = 22) : ∃ x : ℕ, x = 11 :=
by
  sorry

end find_first_number_l50_50525


namespace purchase_total_cost_l50_50973

theorem purchase_total_cost :
  (1 * 16) + (3 * 2) + (6 * 1) = 28 :=
sorry

end purchase_total_cost_l50_50973


namespace trapezoid_leg_length_l50_50409

theorem trapezoid_leg_length (S : ℝ) (h₁ : S > 0) : 
  ∃ x : ℝ, x = Real.sqrt (2 * S) ∧ x > 0 :=
by
  sorry

end trapezoid_leg_length_l50_50409


namespace factorize_a_cubed_minus_25a_l50_50033

variable {a : ℝ}

theorem factorize_a_cubed_minus_25a (a : ℝ) : a^3 - 25 * a = a * (a + 5) * (a - 5) := 
by sorry

end factorize_a_cubed_minus_25a_l50_50033


namespace number_of_arrangements_BANANA_l50_50474

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l50_50474


namespace factorize_expression_l50_50780

theorem factorize_expression (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l50_50780


namespace stephanie_oranges_l50_50709

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l50_50709


namespace total_wasted_time_is_10_l50_50967

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l50_50967


namespace arrangement_count_SUCCESS_l50_50605

-- Define the conditions for the problem
def letters : Finset String := {"S", "U", "C", "C", "E", "S", "S"}
def occurrences_S : Nat := 3
def occurrences_C : Nat := 2
def occurrences_other : Nat := 1 -- For 'U' and 'E'

-- State the theorem using these conditions
theorem arrangement_count_SUCCESS : 
  let N := letters.card
  N = 7 →
  occurrences_S = 3 →
  occurrences_C = 2 →
  occurrences_other = 1 →
  Nat.factorial N / (Nat.factorial occurrences_S * Nat.factorial occurrences_C * Nat.factorial occurrences_other * Nat.factorial occurrences_other) = 420 :=
by
  sorry

end arrangement_count_SUCCESS_l50_50605


namespace squares_difference_l50_50399

theorem squares_difference (x y z : ℤ) 
  (h1 : x + y = 10) 
  (h2 : x - y = 8) 
  (h3 : y + z = 15) : 
  x^2 - z^2 = -115 :=
by 
  sorry

end squares_difference_l50_50399


namespace vasya_correct_l50_50138

theorem vasya_correct (x : ℝ) (h : x^2 + x + 1 = 0) : 
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 :=
by 
  sorry

end vasya_correct_l50_50138


namespace fraction_equality_l50_50495

theorem fraction_equality
  (a b c d : ℝ) 
  (h1 : b ≠ c)
  (h2 : (a * c - b^2) / (a - 2 * b + c) = (b * d - c^2) / (b - 2 * c + d)) : 
  (a * c - b^2) / (a - 2 * b + c) = (a * d - b * c) / (a - b - c + d) ∧
  (b * d - c^2) / (b - 2 * c + d) = (a * d - b * c) / (a - b - c + d) := 
by
  sorry

end fraction_equality_l50_50495


namespace find_a6_l50_50422

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 1) = a (n + 2) + a n

theorem find_a6 (a : ℕ → ℤ) (h : seq a) : a 6 = -3 :=
by
  sorry

end find_a6_l50_50422


namespace target_annual_revenue_l50_50488

-- Given conditions as definitions
def monthly_sales : ℕ := 4000
def additional_sales : ℕ := 1000

-- The proof problem in Lean statement form
theorem target_annual_revenue : (monthly_sales + additional_sales) * 12 = 60000 := by
  sorry

end target_annual_revenue_l50_50488


namespace find_third_number_l50_50685

theorem find_third_number (x : ℝ) 
  (h : (20 + 40 + x) / 3 = (10 + 50 + 45) / 3 + 5) : x = 60 :=
sorry

end find_third_number_l50_50685


namespace expression_evaluation_l50_50255

theorem expression_evaluation (x y z : ℝ) (h : x = y + z) (h' : x = 2) :
  x^3 + 2 * y^3 + 2 * z^3 + 6 * x * y * z = 24 :=
by
  sorry

end expression_evaluation_l50_50255


namespace probability_blue_face_l50_50843

-- Define the total number of faces and the number of blue faces
def total_faces : ℕ := 4 + 2 + 6
def blue_faces : ℕ := 6

-- Calculate the probability of a blue face being up when rolled
theorem probability_blue_face :
  (blue_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end probability_blue_face_l50_50843


namespace wall_width_8_l50_50697

theorem wall_width_8 (w h l : ℝ) (V : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h) 
  (vol_eq : w * h * l = 129024) : 
  w = 8 := 
by 
  sorry

end wall_width_8_l50_50697


namespace somu_present_age_l50_50387

def Somu_Age_Problem (S F : ℕ) : Prop := 
  S = F / 3 ∧ S - 6 = (F - 6) / 5

theorem somu_present_age (S F : ℕ) 
  (h : Somu_Age_Problem S F) : S = 12 := 
by
  sorry

end somu_present_age_l50_50387


namespace alice_no_guarantee_win_when_N_is_18_l50_50103

noncomputable def alice_cannot_guarantee_win : Prop :=
  ∀ (B : ℝ × ℝ) (P : ℕ → ℝ × ℝ),
    (∀ k, 0 ≤ k → k ≤ 18 → 
         dist (P (k + 1)) B < dist (P k) B ∨ dist (P (k + 1)) B ≥ dist (P k) B) →
    ∀ A : ℝ × ℝ, dist A B > 1 / 2020

theorem alice_no_guarantee_win_when_N_is_18 : alice_cannot_guarantee_win :=
sorry

end alice_no_guarantee_win_when_N_is_18_l50_50103


namespace factor_is_three_l50_50749

theorem factor_is_three (x f : ℝ) (h1 : 2 * x + 5 = y) (h2 : f * y = 111) (h3 : x = 16):
  f = 3 :=
by
  sorry

end factor_is_three_l50_50749


namespace value_of_expression_l50_50799

theorem value_of_expression : (2 + 4 + 6) - (1 + 3 + 5) = 3 := 
by 
  sorry

end value_of_expression_l50_50799


namespace estimate_fish_in_pond_l50_50530

theorem estimate_fish_in_pond
  (n m k : ℕ)
  (h_pr: k = 200)
  (h_cr: k = 8)
  (h_m: n = 200):
  n / (m / k) = 5000 := sorry

end estimate_fish_in_pond_l50_50530


namespace sum_of_fourth_powers_l50_50049

theorem sum_of_fourth_powers (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 37 / 6 := 
sorry

end sum_of_fourth_powers_l50_50049


namespace angle_diff_l50_50531

-- Given conditions as definitions
def angle_A : ℝ := 120
def angle_B : ℝ := 50
def angle_D : ℝ := 60
def angle_E : ℝ := 140

-- Prove the difference between angle BCD and angle AFE is 10 degrees
theorem angle_diff (AB_parallel_DE : ∀ (A B D E : ℝ), AB_parallel_DE)
                 (angle_A_def : angle_A = 120)
                 (angle_B_def : angle_B = 50)
                 (angle_D_def : angle_D = 60)
                 (angle_E_def : angle_E = 140) :
    let angle_3 : ℝ := 180 - angle_A
    let angle_4 : ℝ := 180 - angle_E
    let angle_BCD : ℝ := angle_B + angle_D
    let angle_AFE : ℝ := angle_3 + angle_4
    angle_BCD - angle_AFE = 10 :=
by {
  sorry
}

end angle_diff_l50_50531


namespace D_72_value_l50_50053

-- Define D(n) as described
def D (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual function definition

-- Theorem statement
theorem D_72_value : D 72 = 97 :=
by sorry

end D_72_value_l50_50053


namespace root_constraints_between_zero_and_twoR_l50_50091

variable (R l a : ℝ)
variable (hR : R > 0) (hl : l > 0) (ha_nonzero : a ≠ 0)

theorem root_constraints_between_zero_and_twoR :
  ∀ (x : ℝ), (2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2 = 0) →
  (0 < x ∧ x < 2 * R) ↔
  (a > 0 ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (a < 0 ∧ -2 * R < a ∧ l^2 < (2 * R - a)^2) :=
sorry

end root_constraints_between_zero_and_twoR_l50_50091


namespace count_valid_N_l50_50888

theorem count_valid_N : ∃ (N : ℕ), N = 1174 ∧ ∀ (n : ℕ), (1 ≤ n ∧ n < 2000) → ∃ (x : ℝ), x ^ (⌊x⌋ + 1) = n :=
by
  sorry

end count_valid_N_l50_50888


namespace age_difference_l50_50678

variable (A B C : ℕ)

theorem age_difference : A + B = B + C + 11 → A - C = 11 := by
  sorry

end age_difference_l50_50678


namespace total_animals_l50_50580

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l50_50580


namespace inequality_solution_set_l50_50442

noncomputable def solution_set (a b : ℝ) := {x : ℝ | 2 < x ∧ x < 3}

theorem inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (ax^2 + 5 * x + b > 0)) →
  (∀ x : ℝ, (-6) * x^2 - 5 * x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by
  sorry

end inequality_solution_set_l50_50442


namespace tangent_slope_at_one_one_l50_50755

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem tangent_slope_at_one_one : (deriv curve 1) = 2 := 
sorry

end tangent_slope_at_one_one_l50_50755


namespace sum_of_fractions_to_decimal_l50_50517

theorem sum_of_fractions_to_decimal :
  ((2 / 40 : ℚ) + (4 / 80) + (6 / 120) + (9 / 180) : ℚ) = 0.2 :=
by
  sorry

end sum_of_fractions_to_decimal_l50_50517


namespace red_pencils_in_box_l50_50089

theorem red_pencils_in_box (B R G : ℕ) 
  (h1 : B + R + G = 20)
  (h2 : B = 6 * G)
  (h3 : R < B) : R = 6 := by
  sorry

end red_pencils_in_box_l50_50089


namespace sum_S19_is_190_l50_50417

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n + a m = a (n+1) + a (m-1)

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

-- Main theorem
theorem sum_S19_is_190 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S)
  (h_condition : a 6 + a 14 = 20) :
  S 19 = 190 :=
sorry

end sum_S19_is_190_l50_50417


namespace fuel_A_added_l50_50954

noncomputable def total_tank_capacity : ℝ := 218

noncomputable def ethanol_fraction_A : ℝ := 0.12
noncomputable def ethanol_fraction_B : ℝ := 0.16

noncomputable def total_ethanol : ℝ := 30

theorem fuel_A_added (x : ℝ) 
    (hA : 0 ≤ x) 
    (hA_le_capacity : x ≤ total_tank_capacity) 
    (h_eq : 0.12 * x + 0.16 * (total_tank_capacity - x) = total_ethanol) : 
    x = 122 := 
sorry

end fuel_A_added_l50_50954


namespace initial_cats_l50_50932

theorem initial_cats (C : ℕ) (h1 : 36 + 12 - 20 + C = 57) : C = 29 :=
by
  sorry

end initial_cats_l50_50932


namespace inequality_holds_l50_50248

variable {x y : ℝ}

theorem inequality_holds (x : ℝ) (y : ℝ) (hy : y ≥ 5) : 
  x^2 - 2 * x * Real.sqrt (y - 5) + y^2 + y - 30 ≥ 0 := 
sorry

end inequality_holds_l50_50248


namespace circle_tangent_line_standard_equation_l50_50827

-- Problem Statement:
-- Prove that the standard equation of the circle with center at (1,1)
-- and tangent to the line x + y = 4 is (x - 1)^2 + (y - 1)^2 = 2
theorem circle_tangent_line_standard_equation :
  (forall (x y : ℝ), (x + y = 4) -> (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

end circle_tangent_line_standard_equation_l50_50827


namespace gwen_books_collection_l50_50522

theorem gwen_books_collection :
  let mystery_books := 8 * 6
  let picture_books := 5 * 4
  let science_books := 4 * 7
  let non_fiction_books := 3 * 5
  let lent_mystery_books := 2
  let lent_science_books := 3
  let borrowed_picture_books := 5
  mystery_books - lent_mystery_books + picture_books - borrowed_picture_books + borrowed_picture_books + science_books - lent_science_books + non_fiction_books = 106 := by
  sorry

end gwen_books_collection_l50_50522


namespace fill_tank_time_l50_50541

theorem fill_tank_time (t_A t_B : ℕ) (hA : t_A = 20) (hB : t_B = t_A / 4) :
  t_B = 4 := by
  sorry

end fill_tank_time_l50_50541


namespace complete_the_square_1_complete_the_square_2_complete_the_square_3_l50_50317

theorem complete_the_square_1 (x : ℝ) : 
  (x^2 - 2 * x + 3) = (x - 1)^2 + 2 :=
sorry

theorem complete_the_square_2 (x : ℝ) : 
  (3 * x^2 + 6 * x - 1) = 3 * (x + 1)^2 - 4 :=
sorry

theorem complete_the_square_3 (x : ℝ) : 
  (-2 * x^2 + 3 * x - 2) = -2 * (x - 3 / 4)^2 - 7 / 8 :=
sorry

end complete_the_square_1_complete_the_square_2_complete_the_square_3_l50_50317


namespace certain_number_is_14_l50_50797

theorem certain_number_is_14 
  (a b n : ℕ) 
  (h1 : ∃ k1, a = k1 * n) 
  (h2 : ∃ k2, b = k2 * n) 
  (h3 : b = a + 11 * n) 
  (h4 : b = a + 22 * 7) : n = 14 := 
by 
  sorry

end certain_number_is_14_l50_50797


namespace sum_of_digits_0_to_2012_l50_50450

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end sum_of_digits_0_to_2012_l50_50450


namespace factor_z4_minus_81_l50_50478

theorem factor_z4_minus_81 :
  (z^4 - 81) = (z - 3) * (z + 3) * (z^2 + 9) :=
by
  sorry

end factor_z4_minus_81_l50_50478


namespace perfect_square_expression_l50_50290

theorem perfect_square_expression (x y : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
    (h : 4 * x^2 + 8 * y^2 + (2 * x - 3 * y) * p - 12 * x * y = 0) :
    ∃ (n : ℕ), 4 * y + 1 = n^2 :=
sorry

end perfect_square_expression_l50_50290


namespace circle_bisect_line_l50_50777

theorem circle_bisect_line (a : ℝ) :
  (∃ x y, (x - a) ^ 2 + (y + 1) ^ 2 = 3 ∧ 5 * x + 4 * y - a = 0) →
  a = 1 :=
by
  sorry

end circle_bisect_line_l50_50777


namespace prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l50_50679

-- Definitions for the conditions.

-- cube configuration
def num_dice : ℕ := 27
def num_visible_dice : ℕ := 26
def num_faces_per_die : ℕ := 6
def num_visible_faces : ℕ := 54

-- Given probabilities
def prob_six (face : ℕ) : ℚ := 1/6
def prob_not_six (face : ℕ) : ℚ := 5/6
def prob_not_one (face : ℕ) : ℚ := 5/6

-- Expected values given conditions
def expected_num_sixes : ℚ := 9
def expected_sum_faces : ℚ := 189
def expected_diff_digits : ℚ := 6 - (5^6) / (2 * 3^17)

-- Probabilities given conditions
def prob_25_sixes_on_surface : ℚ := (26 * 5) / (6^26)
def prob_at_least_one_one : ℚ := 1 - (5^6) / (2^2 * 3^18)

-- Lean statements for proof

theorem prob_of_25_sixes_on_surface :
  prob_25_sixes_on_surface = 31 / (2^13 * 3^18) := by
  sorry

theorem prob_of_at_least_one_one_on_surface :
  prob_at_least_one_one = 0.99998992 := by
  sorry

theorem expected_number_of_sixes_on_surface :
  expected_num_sixes = 9 := by
  sorry

theorem expected_sum_of_numbers_on_surface :
  expected_sum_faces = 189 := by
  sorry

theorem expected_value_of_diff_digits_on_surface :
  expected_diff_digits = 6 - (5^6) / (2 * 3^17) := by
  sorry

end prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l50_50679


namespace probability_r25_to_r35_l50_50184

theorem probability_r25_to_r35 (n : ℕ) (r : Fin n → ℕ) (h : n = 50) 
  (distinct : ∀ i j : Fin n, i ≠ j → r i ≠ r j) : 1 + 1260 = 1261 :=
by
  sorry

end probability_r25_to_r35_l50_50184


namespace find_n_l50_50738

theorem find_n (x : ℝ) (hx : x > 0) (h : x / n + x / 25 = 0.24000000000000004 * x) : n = 5 :=
sorry

end find_n_l50_50738


namespace man_speed_in_still_water_l50_50288

theorem man_speed_in_still_water
  (speed_of_current_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_meters : ℝ)
  (speed_of_current_ms : ℝ := speed_of_current_kmph * (1000 / 3600))
  (speed_downstream : ℝ := distance_meters / time_seconds) :
  speed_of_current_kmph = 3 →
  time_seconds = 13.998880089592832 →
  distance_meters = 70 →
  (speed_downstream = (25 / 6)) →
  (speed_downstream - speed_of_current_ms) * (3600 / 1000) = 15 :=
by
  intros h_speed_current h_time h_distance h_downstream
  sorry

end man_speed_in_still_water_l50_50288


namespace triangle_inequality_sqrt_sum_three_l50_50171

theorem triangle_inequality_sqrt_sum_three
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  3 ≤ (Real.sqrt (a / (-a + b + c)) + 
       Real.sqrt (b / (a - b + c)) + 
       Real.sqrt (c / (a + b - c))) := 
sorry

end triangle_inequality_sqrt_sum_three_l50_50171


namespace platform_length_l50_50607

noncomputable def train_length := 420 -- length of the train in meters
noncomputable def time_to_cross_platform := 60 -- time to cross the platform in seconds
noncomputable def time_to_cross_pole := 30 -- time to cross the signal pole in seconds

theorem platform_length :
  ∃ L, L = 420 ∧ train_length / time_to_cross_pole = train_length / time_to_cross_platform * (train_length + L) / time_to_cross_platform :=
by
  use 420
  sorry

end platform_length_l50_50607


namespace team_structure_ways_l50_50816

open Nat

noncomputable def combinatorial_structure (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem team_structure_ways :
  let total_members := 13
  let team_lead_choices := total_members
  let remaining_after_lead := total_members - 1
  let project_manager_choices := combinatorial_structure remaining_after_lead 3
  let remaining_after_pm1 := remaining_after_lead - 3
  let subordinate_choices_pm1 := combinatorial_structure remaining_after_pm1 3
  let remaining_after_pm2 := remaining_after_pm1 - 3
  let subordinate_choices_pm2 := combinatorial_structure remaining_after_pm2 3
  let remaining_after_pm3 := remaining_after_pm2 - 3
  let subordinate_choices_pm3 := combinatorial_structure remaining_after_pm3 3
  let total_ways := team_lead_choices * project_manager_choices * subordinate_choices_pm1 * subordinate_choices_pm2 * subordinate_choices_pm3
  total_ways = 4804800 :=
by
  sorry

end team_structure_ways_l50_50816


namespace complement_A_B_correct_l50_50056

open Set

-- Given sets A and B
def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

-- Define the complement of B with respect to A
def complement_A_B : Set ℕ := A \ B

-- Statement to prove
theorem complement_A_B_correct : complement_A_B = {0, 2, 6, 10} :=
  by sorry

end complement_A_B_correct_l50_50056


namespace prime_triplets_satisfy_condition_l50_50568

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_triplets_satisfy_condition :
  ∀ p q r : ℕ,
    is_prime p → is_prime q → is_prime r →
    (p * (r - 1) = q * (r + 7)) →
    (p = 3 ∧ q = 2 ∧ r = 17) ∨ 
    (p = 7 ∧ q = 3 ∧ r = 7) ∨
    (p = 5 ∧ q = 3 ∧ r = 13) :=
by
  sorry

end prime_triplets_satisfy_condition_l50_50568


namespace one_eighth_of_power_l50_50372

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l50_50372


namespace roots_exist_range_k_l50_50501

theorem roots_exist_range_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, (2 * k * x1^2 + (8 * k + 1) * x1 + 8 * k = 0) ∧ 
                 (2 * k * x2^2 + (8 * k + 1) * x2 + 8 * k = 0)) ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end roots_exist_range_k_l50_50501


namespace quarters_initially_l50_50655

theorem quarters_initially (quarters_borrowed : ℕ) (quarters_now : ℕ) (initial_quarters : ℕ) 
   (h1 : quarters_borrowed = 3) (h2 : quarters_now = 5) :
   initial_quarters = quarters_now + quarters_borrowed :=
by
  -- Proof goes here
  sorry

end quarters_initially_l50_50655


namespace American_carmakers_produce_l50_50702

theorem American_carmakers_produce :
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  total = 5650000 :=
by
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  show total = 5650000
  sorry

end American_carmakers_produce_l50_50702


namespace sum_exists_l50_50353

theorem sum_exists 
  (n : ℕ) 
  (hn : n ≥ 5) 
  (k : ℕ) 
  (hk : k > (n + 1) / 2) 
  (a : ℕ → ℕ) 
  (ha1 : ∀ i, 1 ≤ a i) 
  (ha2 : ∀ i, a i < n) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j):
  ∃ i j l, i ≠ j ∧ a i + a j = a l := 
by 
  sorry

end sum_exists_l50_50353


namespace train_crosses_pole_in_12_seconds_l50_50218

noncomputable def time_to_cross_pole (speed train_length : ℕ) : ℕ := 
  train_length / speed

theorem train_crosses_pole_in_12_seconds 
  (speed : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) (train_crossing_time : ℕ)
  (h_speed : speed = 10) 
  (h_platform_length : platform_length = 320) 
  (h_time_to_cross_platform : time_to_cross_platform = 44) 
  (h_train_crossing_time : train_crossing_time = 12) :
  time_to_cross_pole speed 120 = train_crossing_time := 
by 
  sorry

end train_crosses_pole_in_12_seconds_l50_50218


namespace sum_of_nonneg_real_numbers_inequality_l50_50170

open BigOperators

variables {α : Type*} [LinearOrderedField α]

theorem sum_of_nonneg_real_numbers_inequality 
  (a : ℕ → α) (n : ℕ)
  (h_nonneg : ∀ i : ℕ, 0 ≤ a i) : 
  (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j) * (∑ j in Finset.Icc i (n - 1), a j ^ 2))) 
  ≤ (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j)) ^ 2) :=
sorry

end sum_of_nonneg_real_numbers_inequality_l50_50170


namespace smallest_number_of_cubes_l50_50750

def box_length : ℕ := 49
def box_width : ℕ := 42
def box_depth : ℕ := 14
def gcd_box_dimensions : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes :
  (box_length / gcd_box_dimensions) *
  (box_width / gcd_box_dimensions) *
  (box_depth / gcd_box_dimensions) = 84 := by
  sorry

end smallest_number_of_cubes_l50_50750


namespace sunny_ahead_in_second_race_l50_50505

theorem sunny_ahead_in_second_race
  (s w : ℝ)
  (h1 : s / w = 8 / 7) :
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  450 - distance_windy_in_time_sunny = 12.5 :=
by
  let sunny_new_speed := 0.9 * s
  let distance_sunny_runs := 450
  let distance_windy_runs := 400
  let time_sunny := distance_sunny_runs / sunny_new_speed
  let distance_windy_in_time_sunny := w * time_sunny
  sorry

end sunny_ahead_in_second_race_l50_50505


namespace park_area_l50_50854

theorem park_area (length breadth : ℝ) (x : ℝ) 
  (h1 : length = 3 * x) 
  (h2 : breadth = x) 
  (h3 : 2 * length + 2 * breadth = 800) 
  (h4 : 12 * (4 / 60) * 1000 = 800) : 
  length * breadth = 30000 := by
sorry

end park_area_l50_50854


namespace bicycle_cost_l50_50304

theorem bicycle_cost (CP_A SP_B SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 225) : CP_A = 150 :=
by
  sorry

end bicycle_cost_l50_50304


namespace pow_div_pow_eq_result_l50_50183

theorem pow_div_pow_eq_result : 13^8 / 13^5 = 2197 := by
  sorry

end pow_div_pow_eq_result_l50_50183


namespace combined_total_capacity_l50_50609

theorem combined_total_capacity (A B C : ℝ) 
  (hA : 0.35 * A + 48 = 3 / 4 * A)
  (hB : 0.45 * B + 36 = 0.95 * B)
  (hC : 0.20 * C - 24 = 0.10 * C) :
  A + B + C = 432 := 
by 
  sorry

end combined_total_capacity_l50_50609


namespace inverse_proposition_false_l50_50479

theorem inverse_proposition_false (a b c : ℝ) : 
  ¬ (a > b → ((c ≠ 0) ∧ (a / (c * c)) > (b / (c * c))))
:= 
by 
  -- Outline indicating that the proof will follow from checking cases
  sorry

end inverse_proposition_false_l50_50479


namespace ratio_volumes_of_spheres_l50_50466

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l50_50466


namespace least_months_exceed_tripled_borrowed_l50_50157

theorem least_months_exceed_tripled_borrowed :
  ∃ t : ℕ, (1.03 : ℝ)^t > 3 ∧ ∀ n < t, (1.03 : ℝ)^n ≤ 3 :=
sorry

end least_months_exceed_tripled_borrowed_l50_50157


namespace petya_oranges_l50_50012

theorem petya_oranges (m o : ℕ) (h1 : m + 6 * m + o = 20) (h2 : 6 * m > o) : o = 6 :=
by 
  sorry

end petya_oranges_l50_50012


namespace find_x_l50_50878

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 :=
by sorry

end find_x_l50_50878


namespace length_of_first_platform_l50_50565

-- Definitions corresponding to conditions
def length_train := 310
def time_first_platform := 15
def length_second_platform := 250
def time_second_platform := 20

-- Time-speed relationship
def speed_first_platform (L : ℕ) : ℚ := (length_train + L) / time_first_platform
def speed_second_platform : ℚ := (length_train + length_second_platform) / time_second_platform

-- Theorem to prove length of first platform
theorem length_of_first_platform (L : ℕ) (h : speed_first_platform L = speed_second_platform) : L = 110 :=
by
  sorry

end length_of_first_platform_l50_50565


namespace candy_bar_cost_l50_50883

theorem candy_bar_cost :
  ∀ (members : ℕ) (avg_candy_bars : ℕ) (total_earnings : ℝ), 
  members = 20 →
  avg_candy_bars = 8 →
  total_earnings = 80 →
  total_earnings / (members * avg_candy_bars) = 0.50 :=
by
  intros members avg_candy_bars total_earnings h_mem h_avg h_earn
  sorry

end candy_bar_cost_l50_50883


namespace fraction_to_decimal_l50_50627

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l50_50627


namespace ratio_Rachel_Sara_l50_50537

-- Define Sara's spending
def Sara_shoes_spending : ℝ := 50
def Sara_dress_spending : ℝ := 200

-- Define Rachel's budget
def Rachel_budget : ℝ := 500

-- Calculate Sara's total spending
def Sara_total_spending : ℝ := Sara_shoes_spending + Sara_dress_spending

-- Define the theorem to prove the ratio
theorem ratio_Rachel_Sara : (Rachel_budget / Sara_total_spending) = 2 := by
  -- Proof is omitted (you would fill in the proof here)
  sorry

end ratio_Rachel_Sara_l50_50537


namespace remaining_files_l50_50658

def initial_music_files : ℕ := 16
def initial_video_files : ℕ := 48
def deleted_files : ℕ := 30

theorem remaining_files :
  initial_music_files + initial_video_files - deleted_files = 34 := 
by
  sorry

end remaining_files_l50_50658


namespace negation_statement_l50_50825

theorem negation_statement (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) : x^2 - x ≠ 0 :=
by sorry

end negation_statement_l50_50825


namespace tan_5pi_over_4_l50_50169

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l50_50169


namespace superhero_vs_supervillain_distance_l50_50263

-- Definitions expressing the conditions
def superhero_speed (miles : ℕ) (minutes : ℕ) := (10 : ℕ) / (4 : ℕ)
def supervillain_speed (miles_per_hour : ℕ) := (100 : ℕ)

-- Distance calculation in 60 minutes
def superhero_distance_in_hour := 60 * superhero_speed 10 4
def supervillain_distance_in_hour := supervillain_speed 100

-- Proof statement
theorem superhero_vs_supervillain_distance :
  superhero_distance_in_hour - supervillain_distance_in_hour = (50 : ℕ) :=
by
  sorry

end superhero_vs_supervillain_distance_l50_50263


namespace greatest_second_term_arithmetic_sequence_l50_50034

theorem greatest_second_term_arithmetic_sequence:
  ∃ a d : ℕ, (a > 0) ∧ (d > 0) ∧ (2 * a + 3 * d = 29) ∧ (4 * a + 6 * d = 58) ∧ (((a + d : ℤ) / 3 : ℤ) = 10) :=
sorry

end greatest_second_term_arithmetic_sequence_l50_50034


namespace total_expense_l50_50989

noncomputable def sandys_current_age : ℕ := 36 - 2
noncomputable def sandys_monthly_expense : ℕ := 10 * sandys_current_age
noncomputable def alexs_current_age : ℕ := sandys_current_age / 2
noncomputable def alexs_next_month_expense : ℕ := 2 * sandys_monthly_expense

theorem total_expense : 
  sandys_monthly_expense + alexs_next_month_expense = 1020 := 
by 
  sorry

end total_expense_l50_50989


namespace arctan_sum_in_right_triangle_l50_50149

theorem arctan_sum_in_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : 
  (Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4) :=
sorry

end arctan_sum_in_right_triangle_l50_50149


namespace polar_line_equation_l50_50725

theorem polar_line_equation (r θ: ℝ) (p : r = 3 ∧ θ = 0) : r = 3 := 
by 
  sorry

end polar_line_equation_l50_50725


namespace sum_a_b_neg1_l50_50549

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end sum_a_b_neg1_l50_50549


namespace smallest_multiplier_to_perfect_square_l50_50386

-- Definitions for the conditions
def y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6

-- The theorem statement itself
theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, (∀ m : ℕ, (y * m = k) → (∃ n : ℕ, (k * y) = n^2)) :=
by
  let y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6
  let smallest_k := 70
  have h : y = 2^33 * 3^20 * 5^3 * 7^5 := by sorry
  use smallest_k
  intros m hm
  use (2^17 * 3^10 * 5 * 7)
  sorry

end smallest_multiplier_to_perfect_square_l50_50386


namespace find_y_l50_50120

theorem find_y : ∃ y : ℕ, y > 0 ∧ (y + 3050) % 15 = 1234 % 15 ∧ y = 14 := 
by
  sorry

end find_y_l50_50120


namespace units_place_3_pow_34_l50_50452

theorem units_place_3_pow_34 : (3^34 % 10) = 9 :=
by
  sorry

end units_place_3_pow_34_l50_50452


namespace perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l50_50393

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Problem statement for part (I)
theorem perp_bisector_eq : ∃ (k m: ℝ), 3 * k - 4 * m - 23 = 0 :=
sorry

-- Problem statement for part (II)
theorem parallel_line_eq : ∃ (k m: ℝ), 4 * k + 3 * m + 1 = 0 :=
sorry

-- Problem statement for part (III)
theorem reflected_ray_eq : ∃ (k m: ℝ), 11 * k + 27 * m + 74 = 0 :=
sorry

end perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l50_50393


namespace dealer_selling_price_above_cost_l50_50992

variable (cost_price : ℝ := 100)
variable (discount_percent : ℝ := 20)
variable (profit_percent : ℝ := 20)

theorem dealer_selling_price_above_cost :
  ∀ (x : ℝ), 
  (0.8 * x = 1.2 * cost_price) → 
  x = cost_price * (1 + profit_percent / 100) :=
by
  sorry

end dealer_selling_price_above_cost_l50_50992


namespace correct_expression_l50_50747

theorem correct_expression (a b c m x y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b ≠ 0) (h5 : x ≠ y) : 
  ¬ ( (a + m) / (b + m) = a / b ) ∧
  ¬ ( (a + b) / (a + b) = 0 ) ∧ 
  ¬ ( (a * b - 1) / (a * c - 1) = (b - 1) / (c - 1) ) ∧ 
  ( (x - y) / (x^2 - y^2) = 1 / (x + y) ) :=
by
  sorry

end correct_expression_l50_50747


namespace complement_U_P_l50_50763

def U (y : ℝ) : Prop := y > 0
def P (y : ℝ) : Prop := 0 < y ∧ y < 1/3

theorem complement_U_P :
  {y : ℝ | U y} \ {y : ℝ | P y} = {y : ℝ | y ≥ 1/3} :=
by
  sorry

end complement_U_P_l50_50763


namespace unit_digit_2_pow_15_l50_50309

theorem unit_digit_2_pow_15 : (2^15) % 10 = 8 := by
  sorry

end unit_digit_2_pow_15_l50_50309


namespace bouquets_ratio_l50_50208

theorem bouquets_ratio (monday tuesday wednesday : ℕ) 
  (h1 : monday = 12) 
  (h2 : tuesday = 3 * monday) 
  (h3 : monday + tuesday + wednesday = 60) :
  wednesday / tuesday = 1 / 3 :=
by sorry

end bouquets_ratio_l50_50208


namespace anne_carries_16point5_kg_l50_50573

theorem anne_carries_16point5_kg :
  let w1 := 2
  let w2 := 1.5 * w1
  let w3 := 2 * w1
  let w4 := w1 + w2
  let w5 := (w1 + w2) / 2
  w1 + w2 + w3 + w4 + w5 = 16.5 :=
by {
  sorry
}

end anne_carries_16point5_kg_l50_50573


namespace find_sequence_l50_50601

noncomputable def seq (a : ℕ → ℝ) :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = (n / (n + 1)) * (a n + 1))

theorem find_sequence {a : ℕ → ℝ} (h : seq a) :
  ∀ n, a n = (n - 1) / 2 :=
sorry

end find_sequence_l50_50601


namespace laura_pants_count_l50_50760

def cost_of_pants : ℕ := 54
def cost_of_shirt : ℕ := 33
def number_of_shirts : ℕ := 4
def total_money_given : ℕ := 250
def change_received : ℕ := 10

def laura_spent : ℕ := total_money_given - change_received
def total_cost_shirts : ℕ := cost_of_shirt * number_of_shirts
def spent_on_pants : ℕ := laura_spent - total_cost_shirts
def pairs_of_pants_bought : ℕ := spent_on_pants / cost_of_pants

theorem laura_pants_count : pairs_of_pants_bought = 2 :=
by
  sorry

end laura_pants_count_l50_50760


namespace initial_population_l50_50254

/--
Suppose 5% of people in a village died by bombardment,
15% of the remaining population left the village due to fear,
and the population is now reduced to 3294.
Prove that the initial population was 4080.
-/
theorem initial_population (P : ℝ) 
  (H1 : 0.05 * P + 0.15 * (1 - 0.05) * P + 3294 = P) : P = 4080 :=
sorry

end initial_population_l50_50254


namespace speed_in_still_water_l50_50513

theorem speed_in_still_water (upstream_speed : ℝ) (downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 45) (h_downstream : downstream_speed = 55) : 
  (upstream_speed + downstream_speed) / 2 = 50 := 
by
  rw [h_upstream, h_downstream] 
  norm_num  -- simplifies the numeric expression
  done

end speed_in_still_water_l50_50513


namespace probability_target_hit_l50_50787

theorem probability_target_hit (P_A P_B : ℚ) (h1 : P_A = 1/2) (h2 : P_B = 1/3) : 
  (1 - (1 - P_A) * (1 - P_B)) = 2/3 :=
by
  sorry

end probability_target_hit_l50_50787


namespace total_distinguishable_triangles_l50_50540

-- Define number of colors
def numColors : Nat := 8

-- Define center colors
def centerColors : Nat := 3

-- Prove the total number of distinguishable large equilateral triangles
theorem total_distinguishable_triangles : 
  numColors * (numColors + numColors * (numColors - 1) + (numColors.choose 3)) * centerColors = 360 := by
  sorry

end total_distinguishable_triangles_l50_50540


namespace sum_gcd_lcm_60_429_l50_50136

theorem sum_gcd_lcm_60_429 : 
  let a := 60
  let b := 429
  gcd a b + lcm a b = 8583 :=
by
  -- Definitions of a and b
  let a := 60
  let b := 429
  
  -- The GCD and LCM calculations would go here
  
  -- Proof body (skipped with 'sorry')
  sorry

end sum_gcd_lcm_60_429_l50_50136


namespace Felix_distance_proof_l50_50587

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end Felix_distance_proof_l50_50587


namespace max_value_fraction_l50_50278

theorem max_value_fraction (a b : ℝ) (h1 : ab = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  ∃ C, C = 30 / 97 ∧ (∀ x y : ℝ, (xy = 1) → (x > y) → (y ≥ 2/3) → (x - y) / (x^2 + y^2) ≤ C) :=
sorry

end max_value_fraction_l50_50278


namespace B_pow_2021_eq_B_l50_50700

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1 / 2, 0, -Real.sqrt 3 / 2],
  ![0, -1, 0],
  ![Real.sqrt 3 / 2, 0, 1 / 2]
]

theorem B_pow_2021_eq_B : B ^ 2021 = B := 
by sorry

end B_pow_2021_eq_B_l50_50700


namespace man_son_age_ratio_l50_50890

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l50_50890


namespace exchange_rmb_ways_l50_50977

theorem exchange_rmb_ways : 
  {n : ℕ // ∃ (x y z : ℕ), x + 2 * y + 5 * z = 10 ∧ n = 10} :=
sorry

end exchange_rmb_ways_l50_50977


namespace original_bill_l50_50084

theorem original_bill (m : ℝ) (h1 : 10 * (m / 10) = m)
                      (h2 : 9 * ((m - 10) / 10 + 3) = m - 10) :
  m = 180 :=
  sorry

end original_bill_l50_50084


namespace smallest_possible_sum_l50_50704

theorem smallest_possible_sum :
  ∃ (B : ℕ) (c : ℕ), B + c = 34 ∧ 
    (B ≥ 0 ∧ B < 5) ∧ 
    (c > 7) ∧ 
    (31 * B = 4 * c + 4) := 
by
  sorry

end smallest_possible_sum_l50_50704


namespace time_to_get_to_lawrence_house_l50_50689

def distance : ℝ := 12
def speed : ℝ := 2

theorem time_to_get_to_lawrence_house : (distance / speed) = 6 :=
by
  sorry

end time_to_get_to_lawrence_house_l50_50689


namespace factorial_trailing_zeros_l50_50870

theorem factorial_trailing_zeros :
  ∃ (S : Finset ℕ), (∀ m ∈ S, 1 ≤ m ∧ m ≤ 30) ∧ (S.card = 24) ∧ (∀ m ∈ S, 
    ∃ n : ℕ, ∃ k : ℕ,  n ≥ k * 5 ∧ n ≤ (k + 1) * 5 - 1 ∧ 
      m = (n / 5) + (n / 25) + (n / 125) ∧ ((n / 5) % 5 = 0)) :=
sorry

end factorial_trailing_zeros_l50_50870


namespace alicia_local_tax_in_cents_l50_50021

theorem alicia_local_tax_in_cents (hourly_wage : ℝ) (tax_rate : ℝ)
  (h_hourly_wage : hourly_wage = 30) (h_tax_rate : tax_rate = 0.021) :
  (hourly_wage * tax_rate * 100) = 63 := by
  sorry

end alicia_local_tax_in_cents_l50_50021


namespace rectangle_length_l50_50744

theorem rectangle_length (s w : ℝ) (A : ℝ) (L : ℝ) (h1 : s = 9) (h2 : w = 3) (h3 : A = s * s) (h4 : A = w * L) : L = 27 :=
by
  sorry

end rectangle_length_l50_50744


namespace evaluate_expression_l50_50385

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end evaluate_expression_l50_50385


namespace circle_area_l50_50295

theorem circle_area (r : ℝ) (h : 6 / (2 * π * r) = r / 2) : π * r^2 = 3 :=
by
  sorry

end circle_area_l50_50295


namespace base_conversion_subtraction_l50_50001

theorem base_conversion_subtraction :
  let n1_base9 := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let n2_base7 := 1 * 7^2 + 6 * 7^1 + 5 * 7^0
  n1_base9 - n2_base7 = 169 :=
by
  sorry

end base_conversion_subtraction_l50_50001


namespace arithmetic_geometric_mean_inequality_l50_50551

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l50_50551


namespace speed_of_river_l50_50194

-- Definitions of the conditions
def rowing_speed_still_water := 9 -- kmph in still water
def total_time := 1 -- hour for a round trip
def total_distance := 8.84 -- km

-- Distance to the place the man rows to
def d := total_distance / 2

-- Problem statement in Lean 4
theorem speed_of_river (v : ℝ) : 
  rowing_speed_still_water = 9 ∧
  total_time = 1 ∧
  total_distance = 8.84 →
  (4.42 / (rowing_speed_still_water + v) + 4.42 / (rowing_speed_still_water - v) = 1) →
  v = 1.2 := 
by
  sorry

end speed_of_river_l50_50194


namespace theatre_lost_revenue_l50_50886

def ticket_price (category : String) : Float :=
  match category with
  | "general" => 10.0
  | "children" => 6.0
  | "senior" => 8.0
  | "veteran" => 8.0  -- $10.00 - $2.00 discount
  | _ => 0.0

def vip_price (base_price : Float) : Float :=
  base_price + 5.0

def calculate_revenue_sold : Float :=
  let general_revenue := 12 * ticket_price "general" + 3 * (vip_price $ ticket_price "general") / 2
  let children_revenue := 3 * ticket_price "children" + vip_price (ticket_price "children")
  let senior_revenue := 4 * ticket_price "senior" + (vip_price (ticket_price "senior")) / 2
  let veteran_revenue := 2 * ticket_price "veteran" + vip_price (ticket_price "veteran")
  general_revenue + children_revenue + senior_revenue + veteran_revenue

def potential_total_revenue : Float :=
  40 * ticket_price "general" + 10 * vip_price (ticket_price "general")

def potential_revenue_lost : Float :=
  potential_total_revenue - calculate_revenue_sold

theorem theatre_lost_revenue : potential_revenue_lost = 224.0 :=
  sorry

end theatre_lost_revenue_l50_50886


namespace abc_zero_l50_50447

theorem abc_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) :
  a * b * c = 0 := 
sorry

end abc_zero_l50_50447


namespace distance_metric_l50_50269

noncomputable def d (x y : ℝ) : ℝ :=
  (|x - y|) / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem distance_metric (x y z : ℝ) :
  (d x x = 0) ∧
  (d x y = d y x) ∧
  (d x y + d y z ≥ d x z) := by
  sorry

end distance_metric_l50_50269


namespace determine_digits_l50_50740

def product_eq_digits (A B C D x : ℕ) : Prop :=
  x * (x + 1) = 1000 * A + 100 * B + 10 * C + D

def product_minus_3_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D

def product_minus_30_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D

theorem determine_digits :
  ∃ (A B C D x : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  product_eq_digits A B C D x ∧
  product_minus_3_eq_digits A B C D x ∧
  product_minus_30_eq_digits A B C D x ∧
  A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by
  sorry

end determine_digits_l50_50740


namespace fraction_to_percentage_l50_50971

theorem fraction_to_percentage (x : ℝ) (hx : 0 < x) : 
  (x / 50 + x / 25) = 0.06 * x := 
sorry

end fraction_to_percentage_l50_50971


namespace change_received_l50_50193

variable (a : ℕ)

theorem change_received (h : a ≤ 30) : 100 - 3 * a = 100 - 3 * a := 
by 
  sorry

end change_received_l50_50193


namespace find_first_offset_l50_50920

theorem find_first_offset {area diagonal offset₁ offset₂ : ℝ}
  (h_area : area = 150)
  (h_diagonal : diagonal = 20)
  (h_offset₂ : offset₂ = 6) :
  2 * area = diagonal * (offset₁ + offset₂) → offset₁ = 9 := by
  sorry

end find_first_offset_l50_50920


namespace pairs_sum_gcd_l50_50694

theorem pairs_sum_gcd (a b : ℕ) (h_sum : a + b = 288) (h_gcd : Int.gcd a b = 36) :
  (a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108) :=
by {
   sorry
}

end pairs_sum_gcd_l50_50694


namespace profit_calculation_l50_50301

-- Definitions from conditions
def initial_shares := 20
def cost_per_share := 3
def sold_shares := 10
def sale_price_per_share := 4
def remaining_shares_value_multiplier := 2

-- Calculations based on conditions
def initial_cost := initial_shares * cost_per_share
def revenue_from_sold_shares := sold_shares * sale_price_per_share
def remaining_shares := initial_shares - sold_shares
def value_of_remaining_shares := remaining_shares * (cost_per_share * remaining_shares_value_multiplier)
def total_value := revenue_from_sold_shares + value_of_remaining_shares
def expected_profit := total_value - initial_cost

-- The problem statement to be proven
theorem profit_calculation : expected_profit = 40 := by
  -- Proof steps go here
  sorry

end profit_calculation_l50_50301


namespace solve_for_x_l50_50104

theorem solve_for_x (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 :=
by
  sorry

end solve_for_x_l50_50104


namespace max_notebooks_15_dollars_l50_50137

noncomputable def max_notebooks (money : ℕ) : ℕ :=
  let cost_individual   := 2
  let cost_pack_4       := 6
  let cost_pack_7       := 9
  let notebooks_budget  := 15
  if money >= 9 then 
    7 + max_notebooks (money - 9)
  else if money >= 6 then 
    4 + max_notebooks (money - 6)
  else 
    money / 2

theorem max_notebooks_15_dollars : max_notebooks 15 = 11 :=
by
  sorry

end max_notebooks_15_dollars_l50_50137


namespace sufficient_but_not_necessary_l50_50276

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1 → x^2 > 1) ∧ ¬(x^2 > 1 → x < -1) :=
by
  sorry

end sufficient_but_not_necessary_l50_50276


namespace minimum_trucks_required_l50_50360

-- Definitions for the problem
def total_weight_stones : ℝ := 10
def max_stone_weight : ℝ := 1
def truck_capacity : ℝ := 3

-- The theorem to prove
theorem minimum_trucks_required : ∃ (n : ℕ), n = 5 ∧ (n * truck_capacity) ≥ total_weight_stones := by
  sorry

end minimum_trucks_required_l50_50360


namespace weight_of_person_replaced_l50_50619

theorem weight_of_person_replaced (W : ℝ) (old_avg_weight : ℝ) (new_avg_weight : ℝ)
  (h_avg_increase : new_avg_weight = old_avg_weight + 1.5) (new_person_weight : ℝ) :
  ∃ (person_replaced_weight : ℝ), new_person_weight = 77 ∧ old_avg_weight = W / 8 ∧
  new_avg_weight = (W - person_replaced_weight + 77) / 8 ∧ person_replaced_weight = 65 := by
    sorry

end weight_of_person_replaced_l50_50619


namespace sum_of_f_l50_50998

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) / (3^x + (Real.sqrt 3))

theorem sum_of_f :
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6) + 
   f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f (0) + f (1) + f (2) + 
   f (3) + f (4) + f (5) + f (6) + f (7) + f (8) + f (9) + f (10) + 
   f (11) + f (12) + f (13)) = 13 :=
sorry

end sum_of_f_l50_50998


namespace line_passing_through_points_l50_50486

theorem line_passing_through_points (a_1 b_1 a_2 b_2 : ℝ) 
  (h1 : 2 * a_1 + 3 * b_1 + 1 = 0)
  (h2 : 2 * a_2 + 3 * b_2 + 1 = 0) : 
  ∃ (m n : ℝ), (∀ x y : ℝ, (y - b_1) * (x - a_2) = (y - b_2) * (x - a_1)) → (m = 2 ∧ n = 3) :=
by { sorry }

end line_passing_through_points_l50_50486


namespace symmetric_point_l50_50312

-- Define the given conditions
def pointP : (ℤ × ℤ) := (3, -2)
def symmetry_line (y : ℤ) := (y = 1)

-- Prove the assertion that point Q is (3, 4)
theorem symmetric_point (x y1 y2 : ℤ) (hx: x = 3) (hy1: y1 = -2) (hy : symmetry_line 1) :
  (x, 2 * 1 - y1) = (3, 4) :=
by
  sorry

end symmetric_point_l50_50312


namespace initial_deadline_l50_50027

theorem initial_deadline (W : ℕ) (R : ℕ) (D : ℕ) :
    100 * 25 * 8 = (1/3 : ℚ) * W →
    (2/3 : ℚ) * W = 160 * R * 10 →
    D = 25 + R →
    D = 50 := 
by
  intros h1 h2 h3
  sorry

end initial_deadline_l50_50027


namespace number_of_cookies_l50_50680

def candy : ℕ := 63
def brownies : ℕ := 21
def people : ℕ := 7
def dessert_per_person : ℕ := 18

theorem number_of_cookies : 
  (people * dessert_per_person) - (candy + brownies) = 42 := 
by
  sorry

end number_of_cookies_l50_50680


namespace find_line_equation_l50_50416

theorem find_line_equation (k m b : ℝ) :
  (∃ k, |(k^2 + 7*k + 10) - (m*k + b)| = 8) ∧ (8 = 2*m + b) ∧ (b ≠ 0) → (m = 5 ∧ b = 3) := 
by
  intro h
  sorry

end find_line_equation_l50_50416


namespace max_min_diff_of_c_l50_50830

-- Definitions and conditions
variables (a b c : ℝ)
def condition1 := a + b + c = 6
def condition2 := a^2 + b^2 + c^2 = 18

-- Theorem statement
theorem max_min_diff_of_c (h1 : condition1 a b c) (h2 : condition2 a b c) :
  ∃ (c_max c_min : ℝ), c_max = 6 ∧ c_min = -2 ∧ (c_max - c_min = 8) :=
by
  sorry

end max_min_diff_of_c_l50_50830


namespace length_of_train_a_l50_50371

theorem length_of_train_a
  (speed_train_a : ℝ) (speed_train_b : ℝ) 
  (clearing_time : ℝ) (length_train_b : ℝ)
  (h1 : speed_train_a = 42)
  (h2 : speed_train_b = 30)
  (h3 : clearing_time = 12.998960083193344)
  (h4 : length_train_b = 160) :
  ∃ length_train_a : ℝ, length_train_a = 99.9792016638669 :=
by 
  sorry

end length_of_train_a_l50_50371


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l50_50791

-- Problem 1: 1 / 0.25 = 4
theorem problem1 : 1 / 0.25 = 4 :=
by sorry

-- Problem 2: 0.25 / 0.1 = 2.5
theorem problem2 : 0.25 / 0.1 = 2.5 :=
by sorry

-- Problem 3: 1.2 / 1.2 = 1
theorem problem3 : 1.2 / 1.2 = 1 :=
by sorry

-- Problem 4: 4.01 * 1 = 4.01
theorem problem4 : 4.01 * 1 = 4.01 :=
by sorry

-- Problem 5: 0.25 * 2 = 0.5
theorem problem5 : 0.25 * 2 = 0.5 :=
by sorry

-- Problem 6: 0 / 2.76 = 0
theorem problem6 : 0 / 2.76 = 0 :=
by sorry

-- Problem 7: 0.8 / 1.25 = 0.64
theorem problem7 : 0.8 / 1.25 = 0.64 :=
by sorry

-- Problem 8: 3.5 * 2.7 = 9.45
theorem problem8 : 3.5 * 2.7 = 9.45 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l50_50791


namespace Carter_cards_l50_50071

variable (C : ℕ) -- Let C be the number of baseball cards Carter has.

-- Condition 1: Marcus has 210 baseball cards.
def Marcus_cards : ℕ := 210

-- Condition 2: Marcus has 58 more cards than Carter.
def Marcus_has_more (C : ℕ) : Prop := Marcus_cards = C + 58

theorem Carter_cards (C : ℕ) (h : Marcus_has_more C) : C = 152 :=
by
  -- Expand the condition
  unfold Marcus_has_more at h
  -- Simplify the given equation
  rw [Marcus_cards] at h
  -- Solve for C
  linarith

end Carter_cards_l50_50071


namespace Lizzie_group_difference_l50_50343

theorem Lizzie_group_difference
  (lizzie_group_members : ℕ)
  (total_members : ℕ)
  (lizzie_more_than_other : lizzie_group_members > total_members - lizzie_group_members)
  (lizzie_members_eq : lizzie_group_members = 54)
  (total_members_eq : total_members = 91)
  : lizzie_group_members - (total_members - lizzie_group_members) = 17 := 
sorry

end Lizzie_group_difference_l50_50343


namespace reduced_price_is_60_l50_50688

variable (P R: ℝ) -- Declare the variables P and R as real numbers.

-- Define the conditions as hypotheses.
axiom h1 : R = 0.7 * P
axiom h2 : 1800 / R = 1800 / P + 9

-- The theorem stating the problem to prove.
theorem reduced_price_is_60 (P R : ℝ) (h1 : R = 0.7 * P) (h2 : 1800 / R = 1800 / P + 9) : R = 60 :=
by sorry

end reduced_price_is_60_l50_50688


namespace min_sum_of_factors_l50_50463

theorem min_sum_of_factors (x y z : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x * y * z = 3920) : x + y + z = 70 :=
sorry

end min_sum_of_factors_l50_50463


namespace nutty_professor_mixture_weight_l50_50737

/-- The Nutty Professor's problem translated to Lean 4 -/
theorem nutty_professor_mixture_weight :
  let cashews_weight := 20
  let cashews_cost_per_pound := 6.75
  let brazil_nuts_cost_per_pound := 5.00
  let mixture_cost_per_pound := 5.70
  ∃ (brazil_nuts_weight : ℝ), cashews_weight * cashews_cost_per_pound + brazil_nuts_weight * brazil_nuts_cost_per_pound =
                             (cashews_weight + brazil_nuts_weight) * mixture_cost_per_pound ∧
                             (cashews_weight + brazil_nuts_weight = 50) := 
sorry

end nutty_professor_mixture_weight_l50_50737


namespace pow_evaluation_l50_50325

theorem pow_evaluation (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end pow_evaluation_l50_50325


namespace problem_l50_50681

theorem problem (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α^2 = 16 / 5 :=
sorry

end problem_l50_50681


namespace solve_equation_l50_50098

open Real

noncomputable def verify_solution (x : ℝ) : Prop :=
  1 / ((x - 3) * (x - 4)) +
  1 / ((x - 4) * (x - 5)) +
  1 / ((x - 5) * (x - 6)) = 1 / 8

theorem solve_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6) :
  verify_solution x ↔ (x = (9 + sqrt 57) / 2 ∨ x = (9 - sqrt 57) / 2) := 
by
  sorry

end solve_equation_l50_50098


namespace sum_is_odd_prob_l50_50499

-- A type representing the spinner results, which can be either 1, 2, 3 or 4.
inductive SpinnerResult
| one : SpinnerResult
| two : SpinnerResult
| three : SpinnerResult
| four : SpinnerResult

open SpinnerResult

-- Function to determine if a spinner result is odd.
def isOdd (r : SpinnerResult) : Bool :=
  match r with
  | one => true
  | three => true
  | two => false
  | four => false

-- Defining the spinners P, Q, R, and S.
noncomputable def P : SpinnerResult := SpinnerResult.one -- example, could vary
noncomputable def Q : SpinnerResult := SpinnerResult.two -- example, could vary
noncomputable def R : SpinnerResult := SpinnerResult.three -- example, could vary
noncomputable def S : SpinnerResult := SpinnerResult.four -- example, could vary

-- Probability calculation function
def probabilityOddSum : ℚ :=
  let probOdd := 1 / 2
  let probEven := 1 / 2
  let scenario1 := 4 * probOdd * probEven^3
  let scenario2 := 4 * probOdd^3 * probEven
  scenario1 + scenario2

-- The theorem to be stated
theorem sum_is_odd_prob :
  probabilityOddSum = 1 / 2 := by
  sorry

end sum_is_odd_prob_l50_50499


namespace fraction_conversion_l50_50283

theorem fraction_conversion :
  let A := 4.5
  let B := 0.8
  let C := 80.0
  let D := 0.08
  let E := 0.45
  (4 / 5) = B :=
by
  sorry

end fraction_conversion_l50_50283


namespace hens_to_roosters_multiplier_l50_50391

def totalChickens : ℕ := 75
def numHens : ℕ := 67

-- Given the total number of chickens and a certain relationship
theorem hens_to_roosters_multiplier
  (numRoosters : ℕ) (multiplier : ℕ)
  (h1 : totalChickens = numHens + numRoosters)
  (h2 : numHens = multiplier * numRoosters - 5) :
  multiplier = 9 :=
by sorry

end hens_to_roosters_multiplier_l50_50391


namespace prop_B_contrapositive_correct_l50_50958

/-
Proposition B: The contrapositive of the proposition 
"If x^2 < 1, then -1 < x < 1" is 
"If x ≥ 1 or x ≤ -1, then x^2 ≥ 1".
-/
theorem prop_B_contrapositive_correct :
  (∀ (x : ℝ), x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ (x : ℝ), (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
sorry

end prop_B_contrapositive_correct_l50_50958


namespace eggs_in_each_basket_l50_50656

theorem eggs_in_each_basket :
  ∃ (n : ℕ), (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧
    (∀ m : ℕ, (m ∣ 30) ∧ (m ∣ 45) ∧ (m ≥ 5) → m ≤ n) ∧ n = 15 :=
by
  -- Condition 1: n divides 30
  -- Condition 2: n divides 45
  -- Condition 3: n is greater than or equal to 5
  -- Condition 4: n is the largest such divisor
  -- Therefore, n = 15
  sorry

end eggs_in_each_basket_l50_50656


namespace max_cake_pieces_l50_50461

theorem max_cake_pieces (m n : ℕ) (h₁ : m ≥ 4) (h₂ : n ≥ 4)
    (h : (m-4)*(n-4) = m * n) :
    m * n = 72 :=
by
  sorry

end max_cake_pieces_l50_50461


namespace associate_professors_bring_one_chart_l50_50519

theorem associate_professors_bring_one_chart
(A B C : ℕ) (h1 : 2 * A + B = 7) (h2 : A * C + 2 * B = 11) (h3 : A + B = 6) : C = 1 :=
by sorry

end associate_professors_bring_one_chart_l50_50519


namespace predicted_temperature_l50_50781

-- Define the observation data points
def data_points : List (ℕ × ℝ) :=
  [(20, 25), (30, 27.5), (40, 29), (50, 32.5), (60, 36)]

-- Define the linear regression equation with constant k
def regression (x : ℕ) (k : ℝ) : ℝ :=
  0.25 * x + k

-- Proof statement
theorem predicted_temperature (k : ℝ) (h : regression 40 k = 30) : regression 80 k = 40 :=
by
  sorry

end predicted_temperature_l50_50781


namespace final_exam_mean_score_l50_50472

theorem final_exam_mean_score (μ σ : ℝ) 
  (h1 : 55 = μ - 1.5 * σ)
  (h2 : 75 = μ - 2 * σ)
  (h3 : 85 = μ + 1.5 * σ)
  (h4 : 100 = μ + 3.5 * σ) :
  μ = 115 :=
by
  sorry

end final_exam_mean_score_l50_50472


namespace integer_pairs_solution_l50_50979

theorem integer_pairs_solution (k : ℕ) (h : k ≠ 1) : 
  ∃ (m n : ℤ), 
    ((m - n) ^ 2 = 4 * m * n / (m + n - 1)) ∧ 
    (m = k^2 + k / 2 ∧ n = k^2 - k / 2) ∨ 
    (m = k^2 - k / 2 ∧ n = k^2 + k / 2) :=
sorry

end integer_pairs_solution_l50_50979


namespace dog_revs_l50_50406

theorem dog_revs (r₁ r₂ : ℝ) (n₁ : ℕ) (n₂ : ℕ) (h₁ : r₁ = 48) (h₂ : n₁ = 40) (h₃ : r₂ = 12) :
  n₂ = 160 := 
sorry

end dog_revs_l50_50406


namespace parabola_transform_correct_l50_50943

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the transformation of moving the parabola one unit to the right and one unit up
def transformed_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- The theorem to prove
theorem parabola_transform_correct :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 1 :=
by
  intros x
  sorry

end parabola_transform_correct_l50_50943


namespace iced_coffee_cost_correct_l50_50635

-- Definitions based on the conditions 
def coffee_cost_per_day (iced_coffee_cost : ℝ) : ℝ := 3 + iced_coffee_cost
def total_spent (days : ℕ) (iced_coffee_cost : ℝ) : ℝ := days * coffee_cost_per_day iced_coffee_cost

-- Proof statement
theorem iced_coffee_cost_correct (iced_coffee_cost : ℝ) (h : total_spent 20 iced_coffee_cost = 110) : iced_coffee_cost = 2.5 :=
by
  sorry

end iced_coffee_cost_correct_l50_50635


namespace jane_played_8_rounds_l50_50135

variable (points_per_round : ℕ)
variable (end_points : ℕ)
variable (lost_points : ℕ)
variable (total_points : ℕ)
variable (rounds_played : ℕ)

theorem jane_played_8_rounds 
  (h1 : points_per_round = 10) 
  (h2 : end_points = 60) 
  (h3 : lost_points = 20)
  (h4 : total_points = end_points + lost_points)
  (h5 : total_points = points_per_round * rounds_played) : 
  rounds_played = 8 := 
by 
  sorry

end jane_played_8_rounds_l50_50135


namespace choir_row_lengths_l50_50338

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end choir_row_lengths_l50_50338


namespace price_of_red_car_l50_50579

noncomputable def car_price (total_amount loan_amount interest_rate : ℝ) : ℝ :=
  loan_amount + (total_amount - loan_amount) / (1 + interest_rate)

theorem price_of_red_car :
  car_price 38000 20000 0.15 = 35000 :=
by sorry

end price_of_red_car_l50_50579


namespace problem1_problem2_l50_50970

-- Define the given sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x < a + 4 }
def setB : Set ℝ := { x | x < -1 ∨ x > 5 }

-- Problem 1: Prove A ∩ B = { x | -3 < x ∧ x < -1 } when a = 1
theorem problem1 (a : ℝ) (h : a = 1) : 
  (setA a ∩ setB) = { x : ℝ | -3 < x ∧ x < -1 } := sorry

-- Problem 2: Prove range of a given A ∪ B = ℝ is (1, 3)
theorem problem2 (a : ℝ) : 
  (forall x : ℝ, x ∈ (setA a ∪ setB)) ↔ (1 < a ∧ a < 3) := sorry

end problem1_problem2_l50_50970


namespace absolute_value_half_l50_50143

theorem absolute_value_half (a : ℝ) (h : |a| = 1/2) : a = 1/2 ∨ a = -1/2 :=
sorry

end absolute_value_half_l50_50143


namespace investment_ratio_l50_50868

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

end investment_ratio_l50_50868


namespace curves_intersect_at_l50_50236

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

theorem curves_intersect_at :
  (∃ x : ℝ, f x = g x) ↔ ([(0, 2), (6, 86)] = [(0, 2), (6, 86)]) :=
by
  sorry

end curves_intersect_at_l50_50236


namespace hunting_season_fraction_l50_50082

noncomputable def fraction_of_year_hunting_season (hunting_times_per_month : ℕ) 
    (deers_per_hunt : ℕ) (weight_per_deer : ℕ) (fraction_kept : ℚ) 
    (total_weight_kept : ℕ) : ℚ :=
  let total_yearly_weight := total_weight_kept * 2
  let weight_per_hunt := deers_per_hunt * weight_per_deer
  let total_hunts_per_year := total_yearly_weight / weight_per_hunt
  let total_months_hunting := total_hunts_per_year / hunting_times_per_month
  let fraction_of_year := total_months_hunting / 12
  fraction_of_year

theorem hunting_season_fraction : 
  fraction_of_year_hunting_season 6 2 600 (1 / 2 : ℚ) 10800 = 1 / 4 := 
by
  simp [fraction_of_year_hunting_season]
  sorry

end hunting_season_fraction_l50_50082


namespace problem1_problem2_l50_50567

-- Let's define the first problem statement in Lean
theorem problem1 : 2 - 7 * (-3) + 10 + (-2) = 31 := sorry

-- Let's define the second problem statement in Lean
theorem problem2 : -1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14 := sorry

end problem1_problem2_l50_50567


namespace value_of_fraction_l50_50621

variables {a_1 q : ℝ}

-- Define the conditions and the mathematical equivalent of the problem.
def geometric_sequence (a_1 q : ℝ) (h_pos : a_1 > 0 ∧ q > 0) :=
  2 * a_1 + a_1 * q = a_1 * q^2

theorem value_of_fraction (h_pos : a_1 > 0 ∧ q > 0) (h_geom : geometric_sequence a_1 q h_pos) :
  (a_1 * q^3 + a_1 * q^4) / (a_1 * q^2 + a_1 * q^3) = 2 :=
sorry

end value_of_fraction_l50_50621


namespace min_value_x4_y3_z2_l50_50902

theorem min_value_x4_y3_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 9) : 
  x^4 * y^3 * z^2 ≥ 1 / 9^9 :=
by 
  -- Proof goes here
  sorry

end min_value_x4_y3_z2_l50_50902


namespace fraction_increase_by_50_percent_l50_50336

variable (x y : ℝ)
variable (h1 : 0 < y)

theorem fraction_increase_by_50_percent (h2 : 0.6 * x / 0.4 * y = 1.5 * x / y) : 
  1.5 * (x / y) = 1.5 * (x / y) :=
by
  sorry

end fraction_increase_by_50_percent_l50_50336


namespace jill_water_jars_l50_50696

theorem jill_water_jars (x : ℕ) (h : x * (1 / 4 + 1 / 2 + 1) = 28) : 3 * x = 48 :=
by
  sorry

end jill_water_jars_l50_50696


namespace greatest_identical_snack_bags_l50_50341

-- Defining the quantities of each type of snack
def granola_bars : Nat := 24
def dried_fruit : Nat := 36
def nuts : Nat := 60

-- Statement of the problem: greatest number of identical snack bags Serena can make without any food left over.
theorem greatest_identical_snack_bags :
  Nat.gcd (Nat.gcd granola_bars dried_fruit) nuts = 12 :=
sorry

end greatest_identical_snack_bags_l50_50341


namespace area_change_l50_50287

variable (L B : ℝ)

def initial_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.20 * L

def new_breadth (B : ℝ) : ℝ := 0.95 * B

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

theorem area_change (L B : ℝ) : new_area L B = 1.14 * (initial_area L B) := by
  -- Proof goes here
  sorry

end area_change_l50_50287


namespace complement_union_eq_l50_50940

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {3, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {2, 4} := by
  sorry

end complement_union_eq_l50_50940


namespace dan_helmet_craters_l50_50836

namespace HelmetCraters

variables {Dan Daniel Rin : ℕ}

/-- Condition 1: Dan's skateboarding helmet has ten more craters than Daniel's ski helmet. -/
def condition1 (C_d C_daniel : ℕ) : Prop := C_d = C_daniel + 10

/-- Condition 2: Rin's snorkel helmet has 15 more craters than Dan's and Daniel's helmets combined. -/
def condition2 (C_r C_d C_daniel : ℕ) : Prop := C_r = C_d + C_daniel + 15

/-- Condition 3: Rin's helmet has 75 craters. -/
def condition3 (C_r : ℕ) : Prop := C_r = 75

/-- The main theorem: Dan's skateboarding helmet has 35 craters given the conditions. -/
theorem dan_helmet_craters (C_d C_daniel C_r : ℕ) 
    (h1 : condition1 C_d C_daniel) 
    (h2 : condition2 C_r C_d C_daniel) 
    (h3 : condition3 C_r) : C_d = 35 :=
by {
    -- We state that the answer is 35 based on the conditions
    sorry
}

end HelmetCraters

end dan_helmet_craters_l50_50836


namespace cos_arcsin_l50_50532

theorem cos_arcsin (h : (7:ℝ) / 25 ≤ 1) : Real.cos (Real.arcsin ((7:ℝ) / 25)) = (24:ℝ) / 25 := by
  -- Proof to be provided
  sorry

end cos_arcsin_l50_50532


namespace sum_of_perimeters_of_squares_l50_50597

theorem sum_of_perimeters_of_squares
  (x y : ℝ)
  (h1 : x^2 + y^2 = 130)
  (h2 : x^2 / y^2 = 4) :
  4*x + 4*y = 12*Real.sqrt 26 := by
  sorry

end sum_of_perimeters_of_squares_l50_50597


namespace Edward_money_left_l50_50849

theorem Edward_money_left {initial_amount item_cost sales_tax_rate sales_tax total_cost money_left : ℝ} 
    (h_initial : initial_amount = 18) 
    (h_item : item_cost = 16.35) 
    (h_rate : sales_tax_rate = 0.075) 
    (h_sales_tax : sales_tax = item_cost * sales_tax_rate) 
    (h_sales_tax_rounded : sales_tax = 1.23) 
    (h_total : total_cost = item_cost + sales_tax) 
    (h_money_left : money_left = initial_amount - total_cost) :
    money_left = 0.42 :=
by sorry

end Edward_money_left_l50_50849


namespace max_area_trapezoid_l50_50628

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid_l50_50628


namespace bacterium_descendants_in_range_l50_50671

theorem bacterium_descendants_in_range (total_bacteria : ℕ) (initial : ℕ) 
  (h_total : total_bacteria = 1000) (h_initial : initial = total_bacteria) 
  (descendants : ℕ → ℕ)
  (h_step : ∀ k, descendants (k+1) ≤ descendants k / 2) :
  ∃ k, 334 ≤ descendants k ∧ descendants k ≤ 667 :=
by
  sorry

end bacterium_descendants_in_range_l50_50671


namespace no_such_divisor_l50_50164

theorem no_such_divisor (n : ℕ) : 
  (n ∣ (823435 : ℕ)^15) ∧ (n^5 - n^n = 1) → false := 
by sorry

end no_such_divisor_l50_50164


namespace enclosed_area_l50_50016

theorem enclosed_area {x y : ℝ} (h : x^2 + y^2 = 2 * |x| + 2 * |y|) : ∃ (A : ℝ), A = 8 :=
sorry

end enclosed_area_l50_50016


namespace sum_of_first_1000_terms_l50_50291

def sequence_block_sum (n : ℕ) : ℕ :=
  1 + 3 * n

def sequence_sum_up_to (k : ℕ) : ℕ :=
  if k = 0 then 0 else (1 + 3 * (k * (k - 1) / 2)) + k

def nth_term_position (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + n

theorem sum_of_first_1000_terms : sequence_sum_up_to 43 + (1000 - nth_term_position 43) * 3 = 2912 :=
sorry

end sum_of_first_1000_terms_l50_50291


namespace number_of_correct_propositions_l50_50637

def f (x b c : ℝ) := x * |x| + b * x + c

def proposition1 (b : ℝ) : Prop :=
  ∀ (x : ℝ), f x b 0 = -f (-x) b 0

def proposition2 (c : ℝ) : Prop :=
  c > 0 → ∃ (x : ℝ), ∀ (y : ℝ), f y 0 c = 0 → y = x

def proposition3 (b c : ℝ) : Prop :=
  ∀ (x : ℝ), f x b c = f (-x) b c + 2 * c

def proposition4 (b c : ℝ) : Prop :=
  ∀ (x₁ x₂ x₃ : ℝ), f x₁ b c = 0 → f x₂ b c = 0 → f x₃ b c = 0 → x₁ = x₂ ∨ x₂ = x₃ ∨ x₁ = x₃

theorem number_of_correct_propositions (b c : ℝ) : 
  1 + (if c > 0 then 1 else 0) + 1 + 0 = 3 :=
  sorry

end number_of_correct_propositions_l50_50637


namespace rectangle_area_l50_50114

theorem rectangle_area (side_length width length : ℝ) (h_square_area : side_length^2 = 36)
  (h_width : width = side_length) (h_length : length = 2.5 * width) :
  width * length = 90 :=
by 
  sorry

end rectangle_area_l50_50114


namespace classA_classC_ratio_l50_50219

-- Defining the sizes of classes B and C as given in conditions
def classB_size : ℕ := 20
def classC_size : ℕ := 120

-- Defining the size of class A based on the condition that it is twice as big as class B
def classA_size : ℕ := 2 * classB_size

-- Theorem to prove that the ratio of the size of class A to class C is 1:3
theorem classA_classC_ratio : classA_size / classC_size = 1 / 3 := 
sorry

end classA_classC_ratio_l50_50219


namespace range_of_p_l50_50229

def sequence_sum (n : ℕ) : ℚ := (-1) ^ (n + 1) * (1 / 2 ^ n)

def a_n (n : ℕ) : ℚ :=
  if h : n = 0 then sequence_sum 1 else
  sequence_sum n - sequence_sum (n - 1)

theorem range_of_p (p : ℚ) : 
  (∃ n : ℕ, 0 < n ∧ (p - a_n n) * (p - a_n (n + 1)) < 0) ↔ 
  - 3 / 4 < p ∧ p < 1 / 2 :=
sorry

end range_of_p_l50_50229


namespace correct_statements_count_l50_50109

theorem correct_statements_count (x : ℝ) :
  let inverse := (x > 0) → (x^2 > 0)
  let converse := (x^2 ≤ 0) → (x ≤ 0)
  let contrapositive := (x ≤ 0) → (x^2 ≤ 0)
  (∃ p : Prop, p = inverse ∨ p = converse ∧ p) ↔ 
  ¬ contrapositive →
  2 = 2 :=
by
  sorry

end correct_statements_count_l50_50109


namespace unit_digit_product_7858_1086_4582_9783_l50_50025

theorem unit_digit_product_7858_1086_4582_9783 : 
  (7858 * 1086 * 4582 * 9783) % 10 = 8 :=
by
  -- Given that the unit digits of the numbers are 8, 6, 2, and 3.
  let d1 := 7858 % 10 -- This unit digit is 8
  let d2 := 1086 % 10 -- This unit digit is 6
  let d3 := 4582 % 10 -- This unit digit is 2
  let d4 := 9783 % 10 -- This unit digit is 3
  -- We need to prove that the unit digit of the product is 8
  sorry -- The actual proof steps are skipped

end unit_digit_product_7858_1086_4582_9783_l50_50025


namespace chord_segments_division_l50_50332

theorem chord_segments_division (O : Point) (r r0 : ℝ) (h : r0 < r) : 
  3 * r0 ≥ r :=
sorry

end chord_segments_division_l50_50332


namespace find_lesser_number_l50_50023

theorem find_lesser_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by sorry

end find_lesser_number_l50_50023


namespace min_expression_l50_50470

theorem min_expression 
  (a b c : ℝ)
  (ha : -1 < a ∧ a < 1)
  (hb : -1 < b ∧ b < 1)
  (hc : -1 < c ∧ c < 1) :
  ∃ m, m = 2 ∧ ∀ x y z, (-1 < x ∧ x < 1) → (-1 < y ∧ y < 1) → (-1 < z ∧ z < 1) → 
  ( 1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)) ) ≥ m :=
sorry

end min_expression_l50_50470


namespace inequality_solution_intervals_l50_50571

theorem inequality_solution_intervals (x : ℝ) (h : x > 2) : 
  (x-2)^(x^2 - 6 * x + 8) > 1 ↔ (2 < x ∧ x < 3) ∨ x > 4 := 
sorry

end inequality_solution_intervals_l50_50571


namespace degrees_of_remainder_is_correct_l50_50356

noncomputable def degrees_of_remainder (P D : Polynomial ℤ) : Finset ℕ :=
  if D.degree = 3 then {0, 1, 2} else ∅

theorem degrees_of_remainder_is_correct
(P : Polynomial ℤ) :
  degrees_of_remainder P (Polynomial.C 3 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = {0, 1, 2} :=
by
  -- Proof omitted
  sorry

end degrees_of_remainder_is_correct_l50_50356


namespace parabola_shift_units_l50_50285

theorem parabola_shift_units (h : ℝ) :
  (∃ h, (0 + 3 - h)^2 - 1 = 0) ↔ (h = 2 ∨ h = 4) :=
by 
  sorry

end parabola_shift_units_l50_50285


namespace find_f_neg4_l50_50718

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg4 (a b : ℝ) (h : f a b 4 = 0) : f a b (-4) = 2 := by
  -- sorry to skip the proof
  sorry

end find_f_neg4_l50_50718


namespace final_sale_price_l50_50778

def initial_price : ℝ := 450
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def third_discount : ℝ := 0.05

def price_after_first_discount (initial : ℝ) (discount : ℝ) : ℝ :=
  initial * (1 - discount)
  
def price_after_second_discount (price_first : ℝ) (discount : ℝ) : ℝ :=
  price_first * (1 - discount)
  
def price_after_third_discount (price_second : ℝ) (discount : ℝ) : ℝ :=
  price_second * (1 - discount)

theorem final_sale_price :
  price_after_third_discount
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount)
      second_discount)
    third_discount = 288.5625 := 
sorry

end final_sale_price_l50_50778


namespace discount_percentage_l50_50239

theorem discount_percentage (marked_price sale_price cost_price : ℝ) (gain1 gain2 : ℝ)
  (h1 : gain1 = 0.35)
  (h2 : gain2 = 0.215)
  (h3 : sale_price = 30)
  (h4 : cost_price = marked_price / (1 + gain1))
  (h5 : marked_price = cost_price * (1 + gain2)) :
  ((sale_price - marked_price) / sale_price) * 100 = 10.009 :=
sorry

end discount_percentage_l50_50239


namespace remainder_sum_mod_13_l50_50030

theorem remainder_sum_mod_13 (a b c d : ℕ) 
(h₁ : a % 13 = 3) (h₂ : b % 13 = 5) (h₃ : c % 13 = 7) (h₄ : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 :=
by sorry

end remainder_sum_mod_13_l50_50030


namespace magical_stack_card_count_l50_50764

theorem magical_stack_card_count :
  ∃ n, n = 157 + 78 ∧ 2 * n = 470 :=
by
  let n := 235
  use n
  have h1: n = 157 + 78 := by sorry
  have h2: 2 * n = 470 := by sorry
  exact ⟨h1, h2⟩

end magical_stack_card_count_l50_50764


namespace intersection_A_B_union_A_B_subset_C_A_l50_50667

def set_A : Set ℝ := { x | x^2 - x - 2 > 0 }
def set_B : Set ℝ := { x | 3 - abs x ≥ 0 }
def set_C (p : ℝ) : Set ℝ := { x | 4 * x + p < 0 }

theorem intersection_A_B : set_A ∩ set_B = { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) } :=
sorry

theorem union_A_B : set_A ∪ set_B = Set.univ :=
sorry

theorem subset_C_A (p : ℝ) : set_C p ⊆ set_A → p ≥ 4 :=
sorry

end intersection_A_B_union_A_B_subset_C_A_l50_50667


namespace class_heights_mode_median_l50_50802

def mode (l : List ℕ) : ℕ := sorry
def median (l : List ℕ) : ℕ := sorry

theorem class_heights_mode_median 
  (A : List ℕ) -- Heights of students from Class A
  (B : List ℕ) -- Heights of students from Class B
  (hA : A = [170, 170, 169, 171, 171, 171])
  (hB : B = [168, 170, 170, 172, 169, 170]) :
  mode A = 171 ∧ median B = 170 := sorry

end class_heights_mode_median_l50_50802


namespace share_of_a_120_l50_50526

theorem share_of_a_120 (A B C : ℝ) 
  (h1 : A = (2 / 3) * (B + C)) 
  (h2 : B = (6 / 9) * (A + C)) 
  (h3 : A + B + C = 300) : 
  A = 120 := 
by 
  sorry

end share_of_a_120_l50_50526


namespace necessary_but_not_sufficient_l50_50554

-- Define conditions P and Q
def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

-- Statement to prove
theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x ∧ ¬ (Q x → P x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l50_50554


namespace evaluate_g_at_neg3_l50_50433

def g (x : ℝ) : ℝ := 3 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 - 10 * x ^ 2 - 12 * x + 36

theorem evaluate_g_at_neg3 : g (-3) = -1341 := by
  sorry

end evaluate_g_at_neg3_l50_50433


namespace derivative_exp_l50_50367

theorem derivative_exp (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x) : 
    ∀ x, deriv f x = Real.exp x :=
by 
  sorry

end derivative_exp_l50_50367


namespace contest_end_time_l50_50121

def start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 765 -- duration of the contest in minutes

theorem contest_end_time : start_time + duration = 3 * 60 + 45 := by
  -- start_time is 15 * 60 (3:00 p.m. in minutes)
  -- duration is 765 minutes
  -- end_time should be 3:45 a.m. which is 3 * 60 + 45 minutes from midnight
  sorry

end contest_end_time_l50_50121


namespace alice_walks_distance_l50_50957

theorem alice_walks_distance :
  let blocks_south := 5
  let blocks_west := 8
  let distance_per_block := 1 / 4
  let total_blocks := blocks_south + blocks_west
  let total_distance := total_blocks * distance_per_block
  total_distance = 3.25 :=
by
  sorry

end alice_walks_distance_l50_50957


namespace metallic_sheet_width_l50_50814

theorem metallic_sheet_width 
  (length_of_cut_square : ℝ) (original_length_of_sheet : ℝ) (volume_of_box : ℝ) (w : ℝ)
  (h1 : length_of_cut_square = 5) 
  (h2 : original_length_of_sheet = 48) 
  (h3 : volume_of_box = 4940) : 
  (38 * (w - 10) * 5 = 4940) → w = 36 :=
by
  intros
  sorry

end metallic_sheet_width_l50_50814


namespace total_geese_l50_50107

/-- Definition of the number of geese that remain flying after each lake, 
    based on the given conditions. -/
def geese_after_lake (G : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then G else 2^(n : ℕ) - 1

/-- Main theorem stating the total number of geese in the flock. -/
theorem total_geese (n : ℕ) : ∃ (G : ℕ), geese_after_lake G n = 2^n - 1 :=
by
  sorry

end total_geese_l50_50107


namespace christopher_more_than_karen_l50_50959

-- Define the number of quarters Karen and Christopher have
def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64

-- Define the value of a quarter in dollars
def value_of_quarter : ℚ := 0.25

-- Define the amount of money Christopher has more than Karen in dollars
def christopher_more_money : ℚ := (christopher_quarters - karen_quarters) * value_of_quarter

-- Theorem to prove that Christopher has $8.00 more than Karen
theorem christopher_more_than_karen : christopher_more_money = 8 := by
  sorry

end christopher_more_than_karen_l50_50959


namespace interior_angles_sum_l50_50983

def sum_of_interior_angles (sides : ℕ) : ℕ :=
  180 * (sides - 2)

theorem interior_angles_sum (n : ℕ) (h : sum_of_interior_angles n = 1800) :
  sum_of_interior_angles (n + 4) = 2520 :=
sorry

end interior_angles_sum_l50_50983


namespace alpha_plus_beta_l50_50425

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π)
variable (hβ : 0 < β ∧ β < π)
variable (h1 : Real.sin (α - β) = 3 / 4)
variable (h2 : Real.tan α / Real.tan β = -5)

theorem alpha_plus_beta (h3 : α + β = 5 * π / 6) : α + β = 5 * π / 6 :=
by
  sorry

end alpha_plus_beta_l50_50425


namespace perimeter_ratio_l50_50252

/-- Suppose we have a square piece of paper, 6 inches on each side, folded in half horizontally. 
The paper is then cut along the fold, and one of the halves is subsequently cut again horizontally 
through all layers. This results in one large rectangle and two smaller identical rectangles. 
Find the ratio of the perimeter of one smaller rectangle to the perimeter of the larger rectangle. -/
theorem perimeter_ratio (side_length : ℝ) (half_side_length : ℝ) (double_half_side_length : ℝ) :
    side_length = 6 →
    half_side_length = side_length / 2 →
    double_half_side_length = 1.5 * 2 →
    (2 * (half_side_length / 2 + side_length)) / (2 * (half_side_length + side_length)) = (5 / 6) :=
by
    -- Declare the side lengths
    intros h₁ h₂ h₃
    -- Insert the necessary algebra (proven manually earlier)
    sorry

end perimeter_ratio_l50_50252


namespace ratio_jacob_edward_l50_50306

-- Definitions and conditions
def brian_shoes : ℕ := 22
def edward_shoes : ℕ := 3 * brian_shoes
def total_shoes : ℕ := 121
def jacob_shoes : ℕ := total_shoes - brian_shoes - edward_shoes

-- Statement of the problem
theorem ratio_jacob_edward (h_brian : brian_shoes = 22)
                          (h_edward : edward_shoes = 3 * brian_shoes)
                          (h_total : total_shoes = 121)
                          (h_jacob : jacob_shoes = total_shoes - brian_shoes - edward_shoes) :
                          jacob_shoes / edward_shoes = 1 / 2 :=
by sorry

end ratio_jacob_edward_l50_50306


namespace tan_pi_over_4_plus_alpha_eq_two_l50_50529

theorem tan_pi_over_4_plus_alpha_eq_two
  (α : ℂ) 
  (h : Complex.tan ((π / 4) + α) = 2) : 
  (1 / (2 * Complex.sin α * Complex.cos α + (Complex.cos α)^2)) = (2 / 3) :=
by
  sorry

end tan_pi_over_4_plus_alpha_eq_two_l50_50529


namespace inequality_proof_l50_50695

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l50_50695


namespace g_h_2_eq_2175_l50_50097

def g (x : ℝ) : ℝ := 2 * x^2 - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_2_eq_2175 : g (h 2) = 2175 := by
  sorry

end g_h_2_eq_2175_l50_50097


namespace larger_number_is_23_l50_50826

theorem larger_number_is_23 (x y : ℕ) (h_sum : x + y = 40) (h_diff : x - y = 6) : x = 23 :=
sorry

end larger_number_is_23_l50_50826


namespace problem1_problem2_problem3_l50_50375

-- Definitions and conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Question 1: Prove the function is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

-- Question 2: Prove the function is monotonically decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

-- Question 3: Solve the inequality given f(2) = 1
theorem problem3 (h3 : f 2 = 1) : ∀ x : ℝ, f (-x^2) + 2*f x + 4 < 0 ↔ -2 < x ∧ x < 4 := by
  sorry

end problem1_problem2_problem3_l50_50375


namespace terrell_weight_lifting_l50_50936

theorem terrell_weight_lifting (n : ℝ) : 
  (2 * 25 * 10 = 500) → (2 * 20 * n = 500) → n = 12.5 :=
by
  intros h1 h2
  sorry

end terrell_weight_lifting_l50_50936


namespace second_number_is_650_l50_50039

theorem second_number_is_650 (x : ℝ) (h1 : 0.20 * 1600 = 0.20 * x + 190) : x = 650 :=
by sorry

end second_number_is_650_l50_50039


namespace pair_not_equal_to_64_l50_50345

theorem pair_not_equal_to_64 :
  ¬(4 * (9 / 2) = 64) := by
  sorry

end pair_not_equal_to_64_l50_50345


namespace probability_at_least_one_female_l50_50866

open Nat

theorem probability_at_least_one_female :
  let males := 2
  let females := 3
  let total_students := males + females
  let select := 2
  let total_ways := choose total_students select
  let ways_at_least_one_female : ℕ := (choose females 1) * (choose males 1) + choose females 2
  (ways_at_least_one_female / total_ways : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_female_l50_50866


namespace all_girls_select_same_color_probability_l50_50721

theorem all_girls_select_same_color_probability :
  let white_marbles := 10
  let black_marbles := 10
  let red_marbles := 10
  let girls := 15
  ∀ (total_marbles : ℕ), total_marbles = white_marbles + black_marbles + red_marbles →
  (white_marbles < girls ∧ black_marbles < girls ∧ red_marbles < girls) →
  0 = 0 :=
by
  intros
  sorry

end all_girls_select_same_color_probability_l50_50721


namespace solve_cryptarithm_l50_50376

def cryptarithm_puzzle (K I C : ℕ) : Prop :=
  K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K + I + C < 30 ∧  -- Ensuring each is a single digit (0-9)
  (10 * K + I + C) + (10 * K + 10 * C + I) = 100 + 10 * I + 10 * C + K

theorem solve_cryptarithm :
  ∃ K I C, cryptarithm_puzzle K I C ∧ K = 4 ∧ I = 9 ∧ C = 5 :=
by
  use 4, 9, 5
  sorry 

end solve_cryptarithm_l50_50376


namespace find_a4_a5_l50_50919

variable {α : Type*} [LinearOrderedField α]

-- Variables representing the terms of the geometric sequence
variables (a₁ a₂ a₃ a₄ a₅ q : α)

-- Conditions given in the problem
-- Geometric sequence condition
def is_geometric_sequence (a₁ a₂ a₃ a₄ a₅ q : α) : Prop :=
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q

-- First condition
def condition1 : Prop := a₁ + a₂ = 3

-- Second condition
def condition2 : Prop := a₂ + a₃ = 6

-- Theorem stating that a₄ + a₅ = 24 given the conditions
theorem find_a4_a5
  (h1 : condition1 a₁ a₂)
  (h2 : condition2 a₂ a₃)
  (hg : is_geometric_sequence a₁ a₂ a₃ a₄ a₅ q) :
  a₄ + a₅ = 24 := 
sorry

end find_a4_a5_l50_50919


namespace subcommittees_with_at_least_one_teacher_l50_50013

-- Define the total number of members and the count of teachers
def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 5

-- Define binomial coefficient calculation
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Problem statement: number of five-person subcommittees with at least one teacher
theorem subcommittees_with_at_least_one_teacher :
  binom total_members subcommittee_size - binom (total_members - teacher_count) subcommittee_size = 771 := by
  sorry

end subcommittees_with_at_least_one_teacher_l50_50013


namespace rth_term_l50_50374

-- Given arithmetic progression sum formula
def Sn (n : ℕ) : ℕ := 3 * n^2 + 4 * n + 5

-- Prove that the r-th term of the sequence is 6r + 1
theorem rth_term (r : ℕ) : (Sn r) - (Sn (r - 1)) = 6 * r + 1 :=
by
  sorry

end rth_term_l50_50374


namespace garden_area_l50_50722

theorem garden_area (posts : Nat) (distance : Nat) (n_corners : Nat) (a b : Nat)
  (h_posts : posts = 20)
  (h_distance : distance = 4)
  (h_corners : n_corners = 4)
  (h_total_posts : 2 * (a + b) = posts)
  (h_side_relation : b + 1 = 2 * (a + 1)) :
  (distance * (a + 1 - 1)) * (distance * (b + 1 - 1)) = 336 := 
by 
  sorry

end garden_area_l50_50722


namespace positive_integers_of_m_n_l50_50719

theorem positive_integers_of_m_n (m n : ℕ) (p : ℕ) (a : ℕ) (k : ℕ) (h_m_ge_2 : m ≥ 2) (h_n_ge_2 : n ≥ 2) 
  (h_prime_q : Prime (m + 1)) (h_4k_1 : m + 1 = 4 * k - 1) 
  (h_eq : (m ^ (2 ^ n - 1) - 1) / (m - 1) = m ^ n + p ^ a) : 
  (m, n) = (p - 1, 2) ∧ Prime p ∧ ∃k, p = 4 * k - 1 := 
by {
  sorry
}

end positive_integers_of_m_n_l50_50719


namespace geometric_sequence_arithmetic_progression_l50_50462

theorem geometric_sequence_arithmetic_progression
  (q : ℝ) (h_q : q ≠ 1)
  (a : ℕ → ℝ) (m n p : ℕ)
  (h1 : ∃ a1, ∀ k, a k = a1 * q ^ (k - 1))
  (h2 : a n ^ 2 = a m * a p) :
  2 * n = m + p := 
by
  sorry

end geometric_sequence_arithmetic_progression_l50_50462


namespace lateral_surface_area_ratio_l50_50736

theorem lateral_surface_area_ratio (r h : ℝ) :
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  cylinder_area / cone_area = 2 :=
by
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  sorry

end lateral_surface_area_ratio_l50_50736


namespace average_first_50_even_numbers_l50_50810

-- Condition: The sequence starts from 2.
-- Condition: The sequence consists of the first 50 even numbers.
def first50EvenNumbers : List ℤ := List.range' 2 100

theorem average_first_50_even_numbers : (first50EvenNumbers.sum / 50 = 51) :=
by
  sorry

end average_first_50_even_numbers_l50_50810


namespace model_tower_height_l50_50713

theorem model_tower_height (h_real : ℝ) (vol_real : ℝ) (vol_model : ℝ) 
  (h_real_eq : h_real = 60) (vol_real_eq : vol_real = 150000) (vol_model_eq : vol_model = 0.15) :
  (h_real * (vol_model / vol_real)^(1/3) = 0.6) :=
by
  sorry

end model_tower_height_l50_50713


namespace incorrect_operation_in_list_l50_50420

open Real

theorem incorrect_operation_in_list :
  ¬ (abs ((-2)^2) = -2) :=
by
  -- Proof will be added here
  sorry

end incorrect_operation_in_list_l50_50420


namespace constant_term_l50_50768

theorem constant_term (n : ℕ) (h : (Nat.choose n 4 * 2^4) / (Nat.choose n 2 * 2^2) = (56 / 3)) :
  (∃ k : ℕ, k = 2 ∧ n = 10 ∧ Nat.choose 10 k * 2^k = 180) := by
  sorry

end constant_term_l50_50768


namespace quadrilateral_angle_E_l50_50975

theorem quadrilateral_angle_E (E F G H : ℝ)
  (h1 : E = 3 * F)
  (h2 : E = 4 * G)
  (h3 : E = 6 * H)
  (h_sum : E + F + G + H = 360) :
  E = 206 :=
by
  sorry

end quadrilateral_angle_E_l50_50975


namespace shooter_variance_l50_50759

def scores : List ℝ := [9.7, 9.9, 10.1, 10.2, 10.1] -- Defining the scores

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length -- Calculating the mean

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length -- Defining the variance

theorem shooter_variance :
  variance scores = 0.032 :=
by
  sorry -- Proof to be provided later

end shooter_variance_l50_50759


namespace price_of_pants_l50_50457

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end price_of_pants_l50_50457


namespace number_of_men_in_first_group_l50_50079

-- Define the conditions as hypotheses in Lean
def work_completed_in_25_days (x : ℕ) : Prop := x * 25 * (1 : ℚ) / (25 * x) = (1 : ℚ)
def twenty_men_complete_in_15_days : Prop := 20 * 15 * (1 : ℚ) / 15 = (1 : ℚ)

-- Define the theorem to prove the number of men in the first group
theorem number_of_men_in_first_group (x : ℕ) (h1 : work_completed_in_25_days x)
  (h2 : twenty_men_complete_in_15_days) : x = 20 :=
  sorry

end number_of_men_in_first_group_l50_50079


namespace intersection_A_B_l50_50401
open Set

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by sorry

end intersection_A_B_l50_50401


namespace quadratic_roots_problem_l50_50905

theorem quadratic_roots_problem 
  (x y : ℤ) 
  (h1 : x + y = 10)
  (h2 : |x - y| = 12) :
  (x - 11) * (x + 1) = 0 :=
sorry

end quadratic_roots_problem_l50_50905


namespace ratio_m_n_l50_50663

theorem ratio_m_n (m n : ℕ) (h : (n : ℚ) / m = 3 / 7) : (m + n : ℚ) / m = 10 / 7 := by 
  sorry

end ratio_m_n_l50_50663


namespace find_x0_l50_50040

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_x0 (x0 : ℝ) (h : deriv f x0 = 0) : x0 = Real.exp 1 :=
by 
  sorry

end find_x0_l50_50040


namespace sum_of_inscribed_sphere_volumes_l50_50115

theorem sum_of_inscribed_sphere_volumes :
  let height := 3
  let angle := Real.pi / 3
  let r₁ := height / 3 -- Radius of the first inscribed sphere
  let geometric_ratio := 1 / 3
  let volume (r : ℝ) := (4 / 3) * Real.pi * r^3
  let volumes : ℕ → ℝ := λ n => volume (r₁ * geometric_ratio^(n - 1))
  let total_volume := ∑' n, volumes n
  total_volume = (18 * Real.pi) / 13 :=
by
  sorry

end sum_of_inscribed_sphere_volumes_l50_50115


namespace greatest_third_side_of_triangle_l50_50978

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 15) :
  ∃ x : ℕ, 8 < x ∧ x < 22 ∧ (∀ y : ℕ, 8 < y ∧ y < 22 → y ≤ x) ∧ x = 21 :=
by
  sorry

end greatest_third_side_of_triangle_l50_50978


namespace sandy_took_200_l50_50834

variable (X : ℝ)

/-- Given that Sandy had $140 left after spending 30% of the money she took for shopping,
we want to prove that Sandy took $200 for shopping. -/
theorem sandy_took_200 (h : 0.70 * X = 140) : X = 200 :=
by
  sorry

end sandy_took_200_l50_50834


namespace cone_water_volume_percentage_l50_50564

theorem cone_water_volume_percentage
  (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  abs (percentage - 29.6296) < 0.0001 :=
by
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  sorry

end cone_water_volume_percentage_l50_50564


namespace equation_solution_l50_50558

variable (x y : ℝ)

theorem equation_solution
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66):
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 :=
  by sorry

end equation_solution_l50_50558


namespace min_value_range_of_a_l50_50434

noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp (-2 * x) + a * (2 * x + 1) * Real.exp (-x) + x^2 + x

theorem min_value_range_of_a (a : ℝ) (h : a > 0)
  (min_f : ∃ x : ℝ, f a x = Real.log a ^ 2 + 3 * Real.log a + 2) :
  a ∈ Set.Ici (Real.exp (-3 / 2)) :=
by
  sorry

end min_value_range_of_a_l50_50434


namespace complex_number_equality_l50_50809

-- Define the conditions a, b ∈ ℝ and a + i = 1 - bi
theorem complex_number_equality (a b : ℝ) (i : ℂ) (h : a + i = 1 - b * i) : (a + b * i) ^ 8 = 16 :=
  sorry

end complex_number_equality_l50_50809


namespace total_mangoes_l50_50875

-- Definitions of the entities involved
variables (Alexis Dilan Ashley Ben : ℚ)

-- Conditions given in the problem
def condition1 : Prop := Alexis = 4 * (Dilan + Ashley) ∧ Alexis = 60
def condition2 : Prop := Ashley = 2 * Dilan
def condition3 : Prop := Ben = (1/2) * (Dilan + Ashley)

-- The theorem we want to prove: total mangoes is 82.5
theorem total_mangoes (Alexis Dilan Ashley Ben : ℚ)
  (h1 : condition1 Alexis Dilan Ashley)
  (h2 : condition2 Dilan Ashley)
  (h3 : condition3 Dilan Ashley Ben) :
  Alexis + Dilan + Ashley + Ben = 82.5 :=
sorry

end total_mangoes_l50_50875


namespace expression_calculates_to_l50_50475

noncomputable def mixed_number : ℚ := 3 + 3 / 4

noncomputable def decimal_to_fraction : ℚ := 2 / 10

noncomputable def given_expression : ℚ := ((mixed_number * decimal_to_fraction) / 135) * 5.4

theorem expression_calculates_to : given_expression = 0.03 := by
  sorry

end expression_calculates_to_l50_50475


namespace total_marks_of_all_candidates_l50_50390

theorem total_marks_of_all_candidates 
  (average_marks : ℕ) 
  (num_candidates : ℕ) 
  (average : average_marks = 35) 
  (candidates : num_candidates = 120) : 
  average_marks * num_candidates = 4200 :=
by
  -- The proof will be written here
  sorry

end total_marks_of_all_candidates_l50_50390


namespace simplify_expression_l50_50334

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : ( (3 * x + 6 - 5 * x) / 3 ) = ( (-2 * x) / 3 + 2 ) :=
by
  sorry

end simplify_expression_l50_50334


namespace cannot_form_figureB_l50_50468

-- Define the pieces as terms
inductive Piece
| square : Piece
| rectangle : Π (h w : ℕ), Piece   -- h: height, w: width

-- Define the available pieces in a list (assuming these are predefined somewhere)
def pieces : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

-- Define the figures that can be formed
def figureA : List Piece := [Piece.square, Piece.square, Piece.square, Piece.square, Piece.square, 
                            Piece.square, Piece.square, Piece.square, Piece.square, Piece.square]

def figureC : List Piece := [Piece.rectangle 2 1, Piece.rectangle 1 2, Piece.square, Piece.square, 
                             Piece.square, Piece.square]

def figureD : List Piece := [Piece.rectangle 2 2, Piece.square, Piece.square, Piece.square,
                              Piece.square]

def figureE : List Piece := [Piece.rectangle 3 1, Piece.square, Piece.square, Piece.square]

-- Define the figure B that we need to prove cannot be formed
def figureB : List Piece := [Piece.rectangle 5 1, Piece.square, Piece.square, Piece.square,
                              Piece.square]

theorem cannot_form_figureB :
  ¬(∃ arrangement : List Piece, arrangement ⊆ pieces ∧ arrangement = figureB) :=
sorry

end cannot_form_figureB_l50_50468


namespace max_value_of_abc_expression_l50_50161

noncomputable def max_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : ℝ :=
  a^3 * b^2 * c^2

theorem max_value_of_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  max_abc_expression a b c h1 h2 h3 h4 ≤ 432 / 7^7 :=
sorry

end max_value_of_abc_expression_l50_50161


namespace sum_first_100_terms_is_l50_50191

open Nat

noncomputable def seq (a_n : ℕ → ℤ) : Prop :=
  a_n 2 = 2 ∧ ∀ n : ℕ, n > 0 → a_n (n + 2) + (-1)^(n + 1) * a_n n = 1 + (-1)^n

def sum_seq (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum f

theorem sum_first_100_terms_is :
  ∃ (a_n : ℕ → ℤ), seq a_n ∧ sum_seq a_n 100 = 2550 :=
by
  sorry

end sum_first_100_terms_is_l50_50191


namespace intersection_of_A_and_B_l50_50175

-- Definitions of the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

-- Statement to prove the intersection of sets A and B is {3}
theorem intersection_of_A_and_B : A ∩ B = {3} :=
sorry

end intersection_of_A_and_B_l50_50175


namespace root_zero_implies_m_eq_6_l50_50430

theorem root_zero_implies_m_eq_6 (m : ℝ) (h : ∃ x : ℝ, 3 * (x^2) + m * x + m - 6 = 0) : m = 6 := 
by sorry

end root_zero_implies_m_eq_6_l50_50430


namespace triangle_obtuse_at_most_one_l50_50281

open Real -- Work within the Real number system

-- Definitions and main proposition
def is_obtuse (angle : ℝ) : Prop := angle > 90

def triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem triangle_obtuse_at_most_one (a b c : ℝ) (h : triangle a b c) :
  is_obtuse a ∧ is_obtuse b → false :=
by
  sorry

end triangle_obtuse_at_most_one_l50_50281


namespace least_number_of_stamps_l50_50207

theorem least_number_of_stamps : ∃ c f : ℕ, 3 * c + 4 * f = 50 ∧ c + f = 13 :=
by
  sorry

end least_number_of_stamps_l50_50207


namespace cocktail_cost_per_litre_l50_50196

theorem cocktail_cost_per_litre :
  let mixed_fruit_cost := 262.85
  let acai_berry_cost := 3104.35
  let mixed_fruit_volume := 37
  let acai_berry_volume := 24.666666666666668
  let total_cost := mixed_fruit_volume * mixed_fruit_cost + acai_berry_volume * acai_berry_cost
  let total_volume := mixed_fruit_volume + acai_berry_volume
  total_cost / total_volume = 1400 :=
by
  sorry

end cocktail_cost_per_litre_l50_50196


namespace monday_dressing_time_l50_50651

theorem monday_dressing_time 
  (Tuesday_time Wednesday_time Thursday_time Friday_time Old_average_time : ℕ)
  (H_tuesday : Tuesday_time = 4)
  (H_wednesday : Wednesday_time = 3)
  (H_thursday : Thursday_time = 4)
  (H_friday : Friday_time = 2)
  (H_average : Old_average_time = 3) :
  ∃ Monday_time : ℕ, Monday_time = 2 :=
by
  let Total_time_5_days := Old_average_time * 5
  let Total_time := 4 + 3 + 4 + 2
  let Monday_time := Total_time_5_days - Total_time
  exact ⟨Monday_time, sorry⟩

end monday_dressing_time_l50_50651


namespace find_a_value_l50_50599

theorem find_a_value :
  (∀ (x y : ℝ), (x = 1.5 → y = 8 → x * y = 12) ∧ 
               (x = 2 → y = 6 → x * y = 12) ∧ 
               (x = 3 → y = 4 → x * y = 12)) →
  ∃ (a : ℝ), (5 * a = 12 ∧ a = 2.4) :=
by
  sorry

end find_a_value_l50_50599


namespace time_per_mask_after_first_hour_l50_50653

-- Define the conditions as given in the problem
def rate_in_first_hour := 1 / 4 -- Manolo makes one face-mask every four minutes
def total_face_masks := 45 -- Manolo makes 45 face-masks in four hours
def first_hour_duration := 60 -- The duration of the first hour in minutes
def total_duration := 4 * 60 -- The total duration in minutes (4 hours)

-- Define the number of face-masks made in the first hour
def face_masks_first_hour := first_hour_duration / 4 -- 60 minutes / 4 minutes per face-mask = 15 face-masks

-- Calculate the number of face-masks made in the remaining time
def face_masks_remaining_hours := total_face_masks - face_masks_first_hour -- 45 - 15 = 30 face-masks

-- Define the duration of the remaining hours
def remaining_duration := total_duration - first_hour_duration -- 180 minutes (3 hours)

-- The target is to prove that the rate after the first hour is 6 minutes per face-mask
theorem time_per_mask_after_first_hour : remaining_duration / face_masks_remaining_hours = 6 := by
  sorry

end time_per_mask_after_first_hour_l50_50653


namespace real_root_exists_l50_50811

theorem real_root_exists (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 :=
sorry

end real_root_exists_l50_50811


namespace time_period_principal_1000_amount_1120_interest_5_l50_50710

-- Definitions based on the conditions
def principal : ℝ := 1000
def amount : ℝ := 1120
def interest_rate : ℝ := 0.05

-- Lean 4 statement asserting the time period
theorem time_period_principal_1000_amount_1120_interest_5
  (P : ℝ) (A : ℝ) (r : ℝ) (T : ℝ) 
  (hP : P = principal)
  (hA : A = amount)
  (hr : r = interest_rate) :
  (A - P) * 100 / (P * r * 100) = 2.4 :=
by 
  -- The proof is filled in by 'sorry'
  sorry

end time_period_principal_1000_amount_1120_interest_5_l50_50710


namespace diminishing_allocation_proof_l50_50538

noncomputable def diminishing_allocation_problem : Prop :=
  ∃ (a b m : ℝ), 
  a = 0.2 ∧
  b * (1 - a)^2 = 80 ∧
  b * (1 - a) + b * (1 - a)^3 = 164 ∧
  b + 80 + 164 = m ∧
  m = 369

theorem diminishing_allocation_proof : diminishing_allocation_problem :=
by
  sorry

end diminishing_allocation_proof_l50_50538


namespace seating_arrangements_l50_50206

theorem seating_arrangements {n k : ℕ} (h1 : n = 8) (h2 : k = 6) :
  ∃ c : ℕ, c = (n - 1) * Nat.factorial k ∧ c = 20160 :=
by
  sorry

end seating_arrangements_l50_50206


namespace sum_of_squares_of_roots_eq_l50_50377

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq_l50_50377


namespace cubic_function_not_monotonically_increasing_l50_50857

theorem cubic_function_not_monotonically_increasing (b : ℝ) :
  ¬(∀ x y : ℝ, x ≤ y → (1/3)*x^3 + b*x^2 + (b+2)*x + 3 ≤ (1/3)*y^3 + b*y^2 + (b+2)*y + 3) ↔ b ∈ (Set.Iio (-1) ∪ Set.Ioi 2) :=
by sorry

end cubic_function_not_monotonically_increasing_l50_50857


namespace total_admission_cost_l50_50119

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l50_50119


namespace marbles_sum_l50_50539

variable {K M : ℕ}

theorem marbles_sum (hFabian_kyle : 15 = 3 * K) (hFabian_miles : 15 = 5 * M) :
  K + M = 8 :=
by
  sorry

end marbles_sum_l50_50539


namespace rise_in_height_of_field_l50_50405

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

end rise_in_height_of_field_l50_50405


namespace friends_recycled_pounds_l50_50346

-- Definitions for the given conditions
def pounds_per_point : ℕ := 4
def paige_recycled : ℕ := 14
def total_points : ℕ := 4

-- The proof statement
theorem friends_recycled_pounds :
  ∃ p_friends : ℕ, 
  (paige_recycled / pounds_per_point) + (p_friends / pounds_per_point) = total_points 
  → p_friends = 4 := 
sorry

end friends_recycled_pounds_l50_50346


namespace sector_area_max_sector_area_l50_50123

-- Definitions based on the given conditions
def perimeter : ℝ := 8
def central_angle (α : ℝ) : Prop := α = 2

-- Question 1: Find the area of the sector given the central angle is 2 rad
theorem sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) (h2 : l = 2 * r) : 
  (1/2) * r * l = 4 := 
by sorry

-- Question 2: Find the maximum area of the sector and the corresponding central angle
theorem max_sector_area (r l : ℝ) (h1 : 2 * r + l = perimeter) : 
  ∃ r, 0 < r ∧ r < 4 ∧ l = 8 - 2 * r ∧ 
  (1/2) * r * l = 4 ∧ l = 2 * r := 
by sorry

end sector_area_max_sector_area_l50_50123


namespace negation_of_existential_proposition_l50_50262

theorem negation_of_existential_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 > Real.exp x_0) ↔ ∀ (x : ℝ), x^2 ≤ Real.exp x :=
by
  sorry

end negation_of_existential_proposition_l50_50262


namespace steve_needs_28_feet_of_wood_l50_50559

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l50_50559


namespace monotonic_iff_m_ge_one_third_l50_50163

-- Define the function f(x) = x^3 + x^2 + mx + 1
def f (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

-- Define the derivative of the function f w.r.t x
def f' (x m : ℝ) : ℝ := 3 * x^2 + 2 * x + m

-- State the main theorem: f is monotonic on ℝ if and only if m ≥ 1/3
theorem monotonic_iff_m_ge_one_third (m : ℝ) :
  (∀ x y : ℝ, x < y → f x m ≤ f y m) ↔ (m ≥ 1 / 3) :=
sorry

end monotonic_iff_m_ge_one_third_l50_50163


namespace correct_options_l50_50873

-- Given condition
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2 = 0

-- Option B assertion
def option_B (x y : ℝ) : Prop := (x^2 + y^2 - 4)/((x - 1)^2 + y^2 + 1) ≤ 2 + Real.sqrt 6

-- Option D assertion
def option_D (x y : ℝ) : Prop := x - Real.sqrt 2 * y + 2 = 0

-- Theorem to prove both options B and D are correct under the given condition
theorem correct_options {x y : ℝ} (h : curve x y) : option_B x y ∧ option_D x y := by
  sorry

end correct_options_l50_50873


namespace prank_helpers_combinations_l50_50245

theorem prank_helpers_combinations :
  let Monday := 1
  let Tuesday := 2
  let Wednesday := 3
  let Thursday := 4
  let Friday := 1
  (Monday * Tuesday * Wednesday * Thursday * Friday = 24) :=
by
  intros
  sorry

end prank_helpers_combinations_l50_50245


namespace trapezoid_height_ratios_l50_50901

theorem trapezoid_height_ratios (A B C D O M N K L : ℝ) (h : ℝ) (h_AD : D = 2 * B) 
  (h_OK : K = h / 3) (h_OL : L = (2 * h) / 3) :
  (K / h = 1 / 3) ∧ (L / h = 2 / 3) := by
  sorry

end trapezoid_height_ratios_l50_50901


namespace sum_of_squares_l50_50801

def b1 : ℚ := 10 / 32
def b2 : ℚ := 0
def b3 : ℚ := -5 / 32
def b4 : ℚ := 0
def b5 : ℚ := 1 / 32

theorem sum_of_squares : b1^2 + b2^2 + b3^2 + b4^2 + b5^2 = 63 / 512 :=
by
  sorry

end sum_of_squares_l50_50801


namespace min_max_solution_A_l50_50284

theorem min_max_solution_A (x y z : ℕ) (h₁ : x + y + z = 100) (h₂ : 5 * x + 8 * y + 9 * z = 700) 
                           (h₃ : 0 ≤ x ∧ x ≤ 60) (h₄ : 0 ≤ y ∧ y ≤ 60) (h₅ : 0 ≤ z ∧ z ≤ 47) :
    35 ≤ x ∧ x ≤ 49 :=
by
  sorry

end min_max_solution_A_l50_50284


namespace find_m_l50_50896

noncomputable def f (x : ℝ) := 4 * x^2 - 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -14 :=
by
  sorry

end find_m_l50_50896


namespace students_attending_Harvard_l50_50402

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end students_attending_Harvard_l50_50402


namespace find_n_l50_50972

noncomputable def condition (n : ℕ) : Prop :=
  (1/5)^n * (1/4)^18 = 1 / (2 * 10^35)

theorem find_n (n : ℕ) (h : condition n) : n = 35 :=
by
  sorry

end find_n_l50_50972


namespace remaining_hair_length_is_1_l50_50188

-- Variables to represent the inches of hair
variable (initial_length cut_length : ℕ)

-- Given initial length and cut length
def initial_length_is_14 (initial_length : ℕ) := initial_length = 14
def cut_length_is_13 (cut_length : ℕ) := cut_length = 13

-- Definition of the remaining hair length
def remaining_length (initial_length cut_length : ℕ) := initial_length - cut_length

-- Main theorem: Proving the remaining hair length is 1 inch
theorem remaining_hair_length_is_1 : initial_length_is_14 initial_length → cut_length_is_13 cut_length → remaining_length initial_length cut_length = 1 := by
  intros h1 h2
  rw [initial_length_is_14, cut_length_is_13] at *
  simp [remaining_length]
  sorry

end remaining_hair_length_is_1_l50_50188


namespace qr_length_is_correct_l50_50249

/-- Define points and segments in the triangle. -/
structure Point :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(P Q R : Point)

def PQ_length (T : Triangle) : ℝ :=
(T.Q.x - T.P.x) * (T.Q.x - T.P.x) + (T.Q.y - T.P.y) * (T.Q.y - T.P.y)

def PR_length (T : Triangle) : ℝ :=
(T.R.x - T.P.x) * (T.R.x - T.P.x) + (T.R.y - T.P.y) * (T.R.y - T.P.y)

def QR_length (T : Triangle) : ℝ :=
(T.R.x - T.Q.x) * (T.R.x - T.Q.x) + (T.R.y - T.Q.y) * (T.R.y - T.Q.y)

noncomputable def XZ_length (T : Triangle) (X Y Z : Point) : ℝ :=
(PQ_length T)^(1/2) -- Assume the least length of XZ that follows the given conditions

theorem qr_length_is_correct (T : Triangle) :
  PQ_length T = 4*4 → 
  XZ_length T T.P T.Q T.R = 3.2 →
  QR_length T = 4*4 :=
sorry

end qr_length_is_correct_l50_50249


namespace sequence_general_formula_l50_50831

theorem sequence_general_formula (a : ℕ → ℚ) (h₀ : a 1 = 3 / 5)
    (h₁ : ∀ n : ℕ, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n : ℕ, a n = 3 / (6 * n - 1) := 
by sorry

end sequence_general_formula_l50_50831


namespace candy_problem_l50_50672

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l50_50672


namespace average_length_of_ropes_l50_50927

def length_rope_1 : ℝ := 2
def length_rope_2 : ℝ := 6

theorem average_length_of_ropes :
  (length_rope_1 + length_rope_2) / 2 = 4 :=
by
  sorry

end average_length_of_ropes_l50_50927


namespace integral_of_reciprocal_l50_50793

theorem integral_of_reciprocal (a b : ℝ) (h_eq : a = 1) (h_eb : b = Real.exp 1) : ∫ x in a..b, 1/x = 1 :=
by 
  rw [h_eq, h_eb]
  sorry

end integral_of_reciprocal_l50_50793


namespace zero_function_l50_50348

noncomputable def f : ℝ → ℝ := sorry

theorem zero_function :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro h
  sorry

end zero_function_l50_50348


namespace eq1_eq2_eq3_l50_50842

theorem eq1 (x : ℝ) : (x - 2)^2 - 5 = 0 → x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by 
  intro h
  sorry

theorem eq2 (x : ℝ) : x^2 + 4 * x = -3 → x = -1 ∨ x = -3 := 
by 
  intro h
  sorry
  
theorem eq3 (x : ℝ) : 4 * x * (x - 2) = x - 2 → x = 2 ∨ x = 1/4 := 
by 
  intro h
  sorry

end eq1_eq2_eq3_l50_50842


namespace retailer_mark_up_l50_50404

theorem retailer_mark_up (R C M S : ℝ) 
  (hC : C = 0.7 * R)
  (hS : S = C / 0.7)
  (hSm : S = 0.9 * M) : 
  M = 1.111 * R :=
by 
  sorry

end retailer_mark_up_l50_50404


namespace value_of_x_plus_2y_l50_50176

theorem value_of_x_plus_2y 
  (x y : ℝ) 
  (h : (x + 5)^2 = -(|y - 2|)) : 
  x + 2 * y = -1 :=
sorry

end value_of_x_plus_2y_l50_50176


namespace ratio_of_larger_to_smaller_l50_50032

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 5 * (a - b)) :
  a / b = 3 / 2 := by
sorry

end ratio_of_larger_to_smaller_l50_50032


namespace expression_evaluation_l50_50956

def e : Int := -(-1) + 3^2 / (1 - 4) * 2

theorem expression_evaluation : e = -5 := 
by
  unfold e
  sorry

end expression_evaluation_l50_50956


namespace symmetric_point_of_P_l50_50051

-- Let P be a point with coordinates (5, -3)
def P : ℝ × ℝ := (5, -3)

-- Definition of the symmetric point with respect to the x-axis
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Theorem stating that the symmetric point to P with respect to the x-axis is (5, 3)
theorem symmetric_point_of_P : symmetric_point P = (5, 3) := 
  sorry

end symmetric_point_of_P_l50_50051


namespace solution_inequalities_l50_50934

theorem solution_inequalities (x : ℝ) :
  (x^2 - 12 * x + 32 > 0) ∧ (x^2 - 13 * x + 22 < 0) → 2 < x ∧ x < 4 :=
by
  intro h
  sorry

end solution_inequalities_l50_50934


namespace fraction_simplification_l50_50044

theorem fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5 / 4 :=
by
  sorry

end fraction_simplification_l50_50044


namespace smallest_natural_b_for_root_exists_l50_50786

-- Define the problem's conditions
def quadratic_eqn (b : ℕ) := ∀ x : ℝ, x^2 + (b : ℝ) * x + 25 = 0

def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Define the main problem statement
theorem smallest_natural_b_for_root_exists :
  ∃ b : ℕ, (discriminant 1 b 25 ≥ 0) ∧ (∀ b' : ℕ, b' < b → discriminant 1 b' 25 < 0) ∧ b = 10 :=
by
  sorry

end smallest_natural_b_for_root_exists_l50_50786


namespace min_packs_to_buy_120_cans_l50_50784

/-- Prove that the minimum number of packs needed to buy exactly 120 cans of soda,
with packs available in sizes of 8, 15, and 30 cans, is 4. -/
theorem min_packs_to_buy_120_cans : 
  ∃ n, n = 4 ∧ ∀ x y z: ℕ, 8 * x + 15 * y + 30 * z = 120 → x + y + z ≥ n :=
sorry

end min_packs_to_buy_120_cans_l50_50784


namespace largest_divisor_of_product_of_consecutive_evens_l50_50022

theorem largest_divisor_of_product_of_consecutive_evens (n : ℤ) : 
  ∃ d, d = 8 ∧ ∀ n, d ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end largest_divisor_of_product_of_consecutive_evens_l50_50022


namespace charlie_extra_charge_l50_50527

-- Define the data plan and cost structure
def data_plan_limit : ℕ := 8  -- GB
def extra_cost_per_gb : ℕ := 10  -- $ per GB

-- Define Charlie's data usage over each week
def usage_week_1 : ℕ := 2  -- GB
def usage_week_2 : ℕ := 3  -- GB
def usage_week_3 : ℕ := 5  -- GB
def usage_week_4 : ℕ := 10  -- GB

-- Calculate the total data usage and the extra data used
def total_usage : ℕ := usage_week_1 + usage_week_2 + usage_week_3 + usage_week_4
def extra_usage : ℕ := if total_usage > data_plan_limit then total_usage - data_plan_limit else 0
def extra_charge : ℕ := extra_usage * extra_cost_per_gb

-- Theorem to prove the extra charge
theorem charlie_extra_charge : extra_charge = 120 := by
  -- Skipping the proof
  sorry

end charlie_extra_charge_l50_50527


namespace mean_days_jogged_l50_50708

open Real

theorem mean_days_jogged 
  (p1 : ℕ := 5) (d1 : ℕ := 1)
  (p2 : ℕ := 4) (d2 : ℕ := 3)
  (p3 : ℕ := 10) (d3 : ℕ := 5)
  (p4 : ℕ := 7) (d4 : ℕ := 10)
  (p5 : ℕ := 3) (d5 : ℕ := 15)
  (p6 : ℕ := 1) (d6 : ℕ := 20) : 
  ( (p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6) / (p1 + p2 + p3 + p4 + p5 + p6) : ℝ) = 6.73 :=
by
  sorry

end mean_days_jogged_l50_50708


namespace factor_expression_l50_50774

theorem factor_expression (y : ℝ) : 
  5 * y * (y + 2) + 8 * (y + 2) + 15 = (5 * y + 8) * (y + 2) + 15 := 
by
  sorry

end factor_expression_l50_50774


namespace option_c_correct_l50_50455

-- Statement of the problem: Prove that (x-3)^2 = x^2 - 6x + 9

theorem option_c_correct (x : ℝ) : (x - 3) ^ 2 = x ^ 2 - 6 * x + 9 :=
by
  sorry

end option_c_correct_l50_50455


namespace not_all_inequalities_hold_l50_50411

theorem not_all_inequalities_hold (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end not_all_inequalities_hold_l50_50411


namespace haley_marbles_l50_50690

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (h_boys : boys = 13) (h_marbles_per_boy : marbles_per_boy = 2) :
  boys * marbles_per_boy = 26 := 
by 
  sorry

end haley_marbles_l50_50690


namespace sum_of_three_integers_with_product_of_5_cubed_l50_50298

theorem sum_of_three_integers_with_product_of_5_cubed :
  ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  a * b * c = 5^3 ∧ 
  a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_with_product_of_5_cubed_l50_50298


namespace roll_two_dice_prime_sum_l50_50438

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end roll_two_dice_prime_sum_l50_50438


namespace no_repair_needed_l50_50370

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end no_repair_needed_l50_50370


namespace total_edge_length_of_parallelepiped_l50_50636

/-- Kolya has 440 identical cubes with a side length of 1 cm.
Kolya constructs a rectangular parallelepiped from these cubes 
and all edges have lengths of at least 5 cm. Prove 
that the total length of all edges of the rectangular parallelepiped is 96 cm. -/
theorem total_edge_length_of_parallelepiped {a b c : ℕ} 
  (h1 : a * b * c = 440) 
  (h2 : a ≥ 5) 
  (h3 : b ≥ 5) 
  (h4 : c ≥ 5) : 
  4 * (a + b + c) = 96 :=
sorry

end total_edge_length_of_parallelepiped_l50_50636


namespace knives_percentage_l50_50583

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l50_50583


namespace male_worker_ants_percentage_l50_50489

theorem male_worker_ants_percentage 
  (total_ants : ℕ) 
  (half_ants : ℕ) 
  (female_worker_ants : ℕ) 
  (h1 : total_ants = 110) 
  (h2 : half_ants = total_ants / 2) 
  (h3 : female_worker_ants = 44) :
  (half_ants - female_worker_ants) * 100 / half_ants = 20 := by
  sorry

end male_worker_ants_percentage_l50_50489


namespace snail_climbs_well_l50_50921

theorem snail_climbs_well (h : ℕ) (c : ℕ) (s : ℕ) (d : ℕ) (h_eq : h = 12) (c_eq : c = 3) (s_eq : s = 2) : d = 10 :=
by
  sorry

end snail_climbs_well_l50_50921


namespace form_of_reasoning_is_incorrect_l50_50201

-- Definitions from the conditions
def some_rational_numbers_are_fractions : Prop := 
  ∃ q : ℚ, ∃ f : ℚ, q = f / 1

def integers_are_rational_numbers : Prop :=
  ∀ z : ℤ, ∃ q : ℚ, q = z

-- The proposition to be proved
theorem form_of_reasoning_is_incorrect (h1 : some_rational_numbers_are_fractions) (h2 : integers_are_rational_numbers) : 
  ¬ ∀ z : ℤ, ∃ f : ℚ, f = z  := sorry

end form_of_reasoning_is_incorrect_l50_50201


namespace steve_speed_on_way_back_l50_50223

-- Let's define the variables and constants used in the problem.
def distance_to_work : ℝ := 30 -- in km
def total_time_on_road : ℝ := 6 -- in hours
def back_speed_ratio : ℝ := 2 -- Steve drives twice as fast on the way back

theorem steve_speed_on_way_back :
  ∃ v : ℝ, v > 0 ∧ (30 / v + 15 / v = 6) ∧ (2 * v = 15) := by
  sorry

end steve_speed_on_way_back_l50_50223


namespace c_divisible_by_a_l50_50195

theorem c_divisible_by_a {a b c : ℤ} (h1 : a ∣ b * c) (h2 : Int.gcd a b = 1) : a ∣ c :=
by
  sorry

end c_divisible_by_a_l50_50195


namespace div_polynomials_l50_50267

variable (a b : ℝ)

theorem div_polynomials :
  10 * a^3 * b^2 / (-5 * a^2 * b) = -2 * a * b := 
by sorry

end div_polynomials_l50_50267


namespace intersection_of_A_and_B_l50_50684

noncomputable def A : Set ℝ := { x | -1 < x - 3 ∧ x - 3 ≤ 2 }
noncomputable def B : Set ℝ := { x | 3 ≤ x ∧ x < 6 }

theorem intersection_of_A_and_B : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end intersection_of_A_and_B_l50_50684


namespace linear_function_quadrants_l50_50264

theorem linear_function_quadrants (m : ℝ) :
  (∀ (x : ℝ), y = -3 * x + m →
  (x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0 ∨ x < 0 ∧ y < 0)) → m < 0 :=
sorry

end linear_function_quadrants_l50_50264


namespace exists_linear_eq_solution_x_2_l50_50061

theorem exists_linear_eq_solution_x_2 : ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x : ℝ, a * x + b = 0 ↔ x = 2 :=
by
  sorry

end exists_linear_eq_solution_x_2_l50_50061


namespace find_real_x_l50_50615

theorem find_real_x (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end find_real_x_l50_50615


namespace solve_for_y_in_terms_of_x_l50_50075

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3 * x) : y = -2 * x - 2 :=
sorry

end solve_for_y_in_terms_of_x_l50_50075


namespace fraction_of_white_surface_area_l50_50460

def larger_cube_edge : ℕ := 4
def number_of_smaller_cubes : ℕ := 64
def number_of_white_cubes : ℕ := 8
def number_of_red_cubes : ℕ := 56
def total_surface_area : ℕ := 6 * (larger_cube_edge * larger_cube_edge)
def minimized_white_surface_area : ℕ := 7

theorem fraction_of_white_surface_area :
  minimized_white_surface_area % total_surface_area = 7 % 96 :=
by
  sorry

end fraction_of_white_surface_area_l50_50460


namespace ratio_five_to_one_l50_50165

theorem ratio_five_to_one (x : ℕ) : (5 : ℕ) * 13 = 1 * x → x = 65 := 
by 
  intro h
  linarith

end ratio_five_to_one_l50_50165


namespace xyz_positive_and_distinct_l50_50111

theorem xyz_positive_and_distinct (a b x y z : ℝ)
  (h₁ : x + y + z = a)
  (h₂ : x^2 + y^2 + z^2 = b^2)
  (h₃ : x * y = z^2)
  (ha_pos : a > 0)
  (hb_condition : b^2 < a^2 ∧ a^2 < 3*b^2) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end xyz_positive_and_distinct_l50_50111


namespace unique_real_solution_l50_50631

theorem unique_real_solution : ∃ x : ℝ, (∀ t : ℝ, x^2 - t * x + 36 = 0 ∧ x^2 - 8 * x + t = 0) ∧ x = 3 :=
by
  sorry

end unique_real_solution_l50_50631


namespace divisor_of_form_4k_minus_1_l50_50634

theorem divisor_of_form_4k_minus_1
  (n : ℕ) (hn1 : Odd n) (hn_pos : 0 < n)
  (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h_eq : (1 / (x : ℚ) + 1 / (y : ℚ) = 4 / n)) :
  ∃ k : ℕ, ∃ d, d ∣ n ∧ d = 4 * k - 1 ∧ k ∈ Set.Ici 1 :=
sorry

end divisor_of_form_4k_minus_1_l50_50634


namespace birds_more_than_half_sunflower_seeds_l50_50528

theorem birds_more_than_half_sunflower_seeds :
  ∃ (n : ℕ), n = 3 ∧ ((4 / 5)^n * (2 / 5) + (2 / 5) > 1 / 2) :=
by
  sorry

end birds_more_than_half_sunflower_seeds_l50_50528


namespace reporters_local_politics_percentage_l50_50592

theorem reporters_local_politics_percentage
  (T : ℕ) -- Total number of reporters
  (P : ℝ) -- Percentage of reporters covering politics
  (h1 : 30 / 100 * (P / 100) * T = (P / 100 - 0.7 * (P / 100)) * T)
  (h2 : 92.85714285714286 / 100 * T = (1 - P / 100) * T):
  (0.7 * (P / 100) * T) / T = 5 / 100 :=
by
  sorry

end reporters_local_politics_percentage_l50_50592


namespace moles_of_CO2_formed_l50_50251

-- Define the reaction
def reaction (HCl NaHCO3 CO2 : ℕ) : Prop :=
  HCl = NaHCO3 ∧ HCl + NaHCO3 = CO2

-- Given conditions
def given_conditions : Prop :=
  ∃ (HCl NaHCO3 CO2 : ℕ),
    reaction HCl NaHCO3 CO2 ∧ HCl = 3 ∧ NaHCO3 = 3

-- Prove the number of moles of CO2 formed is 3.
theorem moles_of_CO2_formed : given_conditions → ∃ CO2 : ℕ, CO2 = 3 :=
  by
    intros h
    sorry

end moles_of_CO2_formed_l50_50251


namespace range_of_m_l50_50344

def f (x : ℝ) := |x - 3|
def g (x : ℝ) (m : ℝ) := -|x - 7| + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 4 :=
by
  sorry

end range_of_m_l50_50344


namespace find_present_age_of_eldest_l50_50640

noncomputable def eldest_present_age (x : ℕ) : ℕ :=
  8 * x

theorem find_present_age_of_eldest :
  ∃ x : ℕ, 20 * x - 21 = 59 ∧ eldest_present_age x = 32 :=
by
  sorry

end find_present_age_of_eldest_l50_50640


namespace find_x_minus_y_l50_50307

variables (x y : ℚ)

theorem find_x_minus_y
  (h1 : 3 * x - 4 * y = 17)
  (h2 : x + 3 * y = 1) :
  x - y = 69 / 13 := 
sorry

end find_x_minus_y_l50_50307


namespace common_root_implies_remaining_roots_l50_50139

variables {R : Type*} [LinearOrderedField R]

theorem common_root_implies_remaining_roots
  (a b c x1 x2 x3 : R) 
  (h_non_zero_a : a ≠ 0)
  (h_non_zero_b : b ≠ 0)
  (h_non_zero_c : c ≠ 0)
  (h_a_ne_b : a ≠ b)
  (h_common_root1 : x1^2 + a*x1 + b*c = 0)
  (h_common_root2 : x1^2 + b*x1 + c*a = 0)
  (h_root2_eq : x2^2 + a*x2 + b*c = 0)
  (h_root3_eq : x3^2 + b*x3 + c*a = 0)
  : x2^2 + c*x2 + a*b = 0 ∧ x3^2 + c*x3 + a*b = 0 :=
sorry

end common_root_implies_remaining_roots_l50_50139


namespace least_sum_of_exponents_l50_50244

theorem least_sum_of_exponents (a b c d e : ℕ) (h : ℕ) (h_divisors : 225 ∣ h ∧ 216 ∣ h ∧ 847 ∣ h)
  (h_form : h = (2 ^ a) * (3 ^ b) * (5 ^ c) * (7 ^ d) * (11 ^ e)) : 
  a + b + c + d + e = 10 :=
sorry

end least_sum_of_exponents_l50_50244


namespace fill_buckets_lcm_l50_50727

theorem fill_buckets_lcm :
  (∀ (A B C : ℕ), (2 / 3 : ℚ) * A = 90 ∧ (1 / 2 : ℚ) * B = 120 ∧ (3 / 4 : ℚ) * C = 150 → lcm A (lcm B C) = 1200) :=
by
  sorry

end fill_buckets_lcm_l50_50727


namespace compute_c_plus_d_l50_50563

variable {c d : ℝ}

-- Define the given polynomial equations
def poly_c (c : ℝ) := c^3 - 21*c^2 + 28*c - 70
def poly_d (d : ℝ) := 10*d^3 - 75*d^2 - 350*d + 3225

theorem compute_c_plus_d (hc : poly_c c = 0) (hd : poly_d d = 0) : c + d = 21 / 2 := sorry

end compute_c_plus_d_l50_50563


namespace compute_z_pow_7_l50_50906

namespace ComplexProof

noncomputable def z : ℂ := (Real.sqrt 3 + Complex.I) / 2

theorem compute_z_pow_7 : z ^ 7 = - (Real.sqrt 3 / 2) - (1 / 2) * Complex.I :=
by
  sorry

end ComplexProof

end compute_z_pow_7_l50_50906


namespace proof_problem_l50_50368

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ f (-3) = -2

theorem proof_problem (f : ℝ → ℝ) (h : given_function f) : f 3 + f 0 = -2 :=
by sorry

end proof_problem_l50_50368


namespace correct_f_l50_50837

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom functional_equation (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem correct_f (x : ℝ) : f x = x + 1 := sorry

end correct_f_l50_50837


namespace option_A_option_B_option_C_option_D_verify_options_l50_50828

open Real

-- Option A: Prove the maximum value of x(6-x) given 0 < x < 6 is 9.
theorem option_A (x : ℝ) (h1 : 0 < x) (h2 : x < 6) : 
  ∃ (max_value : ℝ), max_value = 9 ∧ ∀(y : ℝ), 0 < y ∧ y < 6 → y * (6 - y) ≤ max_value :=
sorry

-- Option B: Prove the minimum value of x^2 + 1/(x^2 + 3) for x in ℝ is not -1.
theorem option_B (x : ℝ) : ¬(∃ (min_value : ℝ), min_value = -1 ∧ ∀(y : ℝ), (y ^ 2) + 1 / (y ^ 2 + 3) ≥ min_value) :=
sorry

-- Option C: Prove the maximum value of xy given x + 2y + xy = 6 and x, y > 0 is 2.
theorem option_C (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y + x * y = 6) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 2 * v + u * v = 6 → u * v ≤ max_value :=
sorry

-- Option D: Prove the minimum value of 2x + y given x + 4y + 4 = xy and x, y > 0 is 17.
theorem option_D (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 4 * y + 4 = x * y) : 
  ∃ (min_value : ℝ), min_value = 17 ∧ ∀(u v : ℝ), 0 < u ∧ 0 < v ∧ u + 4 * v + 4 = u * v → 2 * u + v ≥ min_value :=
sorry

-- Combine to verify which options are correct
theorem verify_options (A_correct B_correct C_correct D_correct : Prop) :
  A_correct = true ∧ B_correct = false ∧ C_correct = true ∧ D_correct = true :=
sorry

end option_A_option_B_option_C_option_D_verify_options_l50_50828


namespace area_of_fig_between_x1_and_x2_l50_50464

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end area_of_fig_between_x1_and_x2_l50_50464


namespace line_intersects_ellipse_slopes_l50_50315

theorem line_intersects_ellipse_slopes :
  {m : ℝ | ∃ x, 4 * x^2 + 25 * (m * x + 8)^2 = 100} = 
  {m : ℝ | m ≤ -Real.sqrt 2.4 ∨ Real.sqrt 2.4 ≤ m} := 
by
  sorry

end line_intersects_ellipse_slopes_l50_50315


namespace slope_of_l4_l50_50469

open Real

def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 6
def pointD : ℝ × ℝ := (0, -2)
def line2 (y : ℝ) : Prop := y = -1
def area_triangle_DEF := 4

theorem slope_of_l4 
  (l4_slope : ℝ)
  (H1 : ∃ x, line1 x (-1))
  (H2 : ∀ x y, 
         x ≠ 0 ∧
         y ≠ -2 ∧
         y ≠ -1 →
         line2 y →
         l4_slope = (y - (-2)) / (x - 0) →
         (1/2) * |(y + 1)| * (sqrt ((x-0) * (x-0) + (y-(-2)) * (y-(-2)))) = area_triangle_DEF ) :
  l4_slope = 1 / 8 :=
sorry

end slope_of_l4_l50_50469


namespace largest_4_digit_divisible_by_12_l50_50952

theorem largest_4_digit_divisible_by_12 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 12 ∣ n ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ 12 ∣ m → m ≤ n :=
sorry

end largest_4_digit_divisible_by_12_l50_50952


namespace company_fund_amount_l50_50669

theorem company_fund_amount (n : ℕ) (h : 70 * n + 160 = 80 * n - 8) : 
  80 * n - 8 = 1352 :=
sorry

end company_fund_amount_l50_50669


namespace andrew_ruined_planks_l50_50496

variable (b L k g h leftover plank_total ruin_bedroom ruin_guest : ℕ)

-- Conditions
def bedroom_planks := b
def living_room_planks := L
def kitchen_planks := k
def guest_bedroom_planks := g
def hallway_planks := h
def planks_leftover := leftover

-- Values
axiom bedroom_planks_val : bedroom_planks = 8
axiom living_room_planks_val : living_room_planks = 20
axiom kitchen_planks_val : kitchen_planks = 11
axiom guest_bedroom_planks_val : guest_bedroom_planks = bedroom_planks - 2
axiom hallway_planks_val : hallway_planks = 4
axiom planks_leftover_val : planks_leftover = 6

-- Total planks used and total planks had
def total_planks_used := bedroom_planks + living_room_planks + kitchen_planks + guest_bedroom_planks + (2 * hallway_planks)
def total_planks_had := total_planks_used + planks_leftover

-- Planks ruined
def planks_ruined_in_bedroom := ruin_bedroom
def planks_ruined_in_guest_bedroom := ruin_guest

-- Theorem to be proven
theorem andrew_ruined_planks :
  (planks_ruined_in_bedroom = total_planks_had - total_planks_used) ∧
  (planks_ruined_in_guest_bedroom = planks_ruined_in_bedroom) :=
by
  sorry

end andrew_ruined_planks_l50_50496


namespace find_x_squared_l50_50005

variable (a b x p q : ℝ)

theorem find_x_squared (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : q ≠ p) (h4 : (a^2 + x^2) / (b^2 + x^2) = p / q) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := 
by 
  sorry

end find_x_squared_l50_50005


namespace inequality_a_b_c_l50_50261

theorem inequality_a_b_c (a b c : ℝ) (h1 : 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a^2 + b^2 + c^2 = 3) : 
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_a_b_c_l50_50261


namespace matrix_product_is_zero_l50_50593

-- Define the two matrices
def A (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -c], ![-d, 0, b], ![c, -b, 0]]

def B (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![d^2, b * d, c * d], ![b * d, b^2, b * c], ![c * d, b * c, c^2]]

-- Define the zero matrix
def zero_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 0], ![0, 0, 0]]

-- The theorem to prove
theorem matrix_product_is_zero (b c d : ℝ) : A b c d * B b c d = zero_matrix :=
by sorry

end matrix_product_is_zero_l50_50593


namespace find_a2_l50_50907

noncomputable def geometric_sequence (n : ℕ) : ℝ := sorry -- Define the geometric sequence

variable (a1 : ℝ) (a3a5_eq : ℝ) -- Variables for given conditions

-- Main theorem statement
theorem find_a2 (h_geo : ∀ n, geometric_sequence n = a1 * (2 : ℝ) ^ (n - 1))
  (h_a1 : a1 = 1 / 4)
  (h_a3a5 : (geometric_sequence 3) * (geometric_sequence 5) = 4 * (geometric_sequence 4 - 1)) :
  geometric_sequence 2 = 1 / 2 :=
sorry  -- Proof is omitted

end find_a2_l50_50907


namespace s_of_4_l50_50355

noncomputable def t (x : ℚ) : ℚ := 5 * x - 14
noncomputable def s (y : ℚ) : ℚ := 
  let x := (y + 14) / 5
  x^2 + 5 * x - 4

theorem s_of_4 : s (4) = 674 / 25 := by
  sorry

end s_of_4_l50_50355


namespace solve_inequalities_l50_50144

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l50_50144


namespace sequence_pattern_l50_50616

theorem sequence_pattern (a b c : ℝ) (h1 : a = 19.8) (h2 : b = 18.6) (h3 : c = 17.4) 
  (h4 : ∀ n, n = a ∨ n = b ∨ n = c ∨ n = 16.2 ∨ n = 15) 
  (H : ∀ x y, (y = x - 1.2) → 
    (x = a ∨ x = b ∨ x = c ∨ y = 16.2 ∨ y = 15)) :
  (16.2 = c - 1.2) ∧ (15 = (c - 1.2) - 1.2) :=
by
  sorry

end sequence_pattern_l50_50616


namespace function_sqrt_plus_one_l50_50897

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem function_sqrt_plus_one (h1 : ∀ x : ℝ, f x = 3) (h2 : x ≥ 0) : f (Real.sqrt x) + 1 = 4 :=
by
  sorry

end function_sqrt_plus_one_l50_50897


namespace quadratic_unique_solution_l50_50798

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l50_50798


namespace three_pow_sub_cube_eq_two_l50_50047

theorem three_pow_sub_cube_eq_two (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 2 := 
sorry

end three_pow_sub_cube_eq_two_l50_50047


namespace train_avg_speed_l50_50574

variable (x : ℝ)

def avg_speed_of_train (x : ℝ) : ℝ := 3

theorem train_avg_speed (h : x > 0) : avg_speed_of_train x / (x / 7.5) = 22.5 :=
  sorry

end train_avg_speed_l50_50574


namespace greatest_divisor_4665_6905_l50_50035

def digits_sum (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem greatest_divisor_4665_6905 :
  ∃ n : ℕ, (n ∣ 4665) ∧ (n ∣ 6905) ∧ (digits_sum n = 4) ∧
  (∀ m : ℕ, ((m ∣ 4665) ∧ (m ∣ 6905) ∧ (digits_sum m = 4)) → (m ≤ n)) :=
sorry

end greatest_divisor_4665_6905_l50_50035


namespace boys_play_theater_with_Ocho_l50_50566

variables (Ocho_friends : ℕ) (half_girls : Ocho_friends / 2 = 4)

theorem boys_play_theater_with_Ocho : (Ocho_friends / 2) = 4 := by
  -- Ocho_friends is the total number of Ocho's friends
  -- half_girls is given as a condition that half of Ocho's friends are girls
  -- thus, we directly use this to conclude that the number of boys is 4
  sorry

end boys_play_theater_with_Ocho_l50_50566


namespace proof_problem_l50_50271

variable {a b c : ℝ}

theorem proof_problem (h_cond : 0 < a ∧ a < b ∧ b < c) : 
  a * c < b * c ∧ a + b < b + c ∧ c / a > c / b := by
  sorry

end proof_problem_l50_50271


namespace imaginary_part_of_complex_num_l50_50804

-- Define the complex number and the imaginary part condition
def complex_num : ℂ := ⟨1, 2⟩

-- Define the theorem to prove the imaginary part is 2
theorem imaginary_part_of_complex_num : complex_num.im = 2 :=
by
  -- The proof steps would go here
  sorry

end imaginary_part_of_complex_num_l50_50804


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l50_50490

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l50_50490


namespace sum_of_squares_of_consecutive_integers_l50_50314

theorem sum_of_squares_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x^2 + (x + 1)^2 = 1625 := by
  sorry

end sum_of_squares_of_consecutive_integers_l50_50314


namespace blackboard_problem_l50_50337

theorem blackboard_problem (n : ℕ) (h_pos : 0 < n) :
  ∃ x, (∀ (t : ℕ), t < n - 1 → ∃ a b : ℕ, a + b + 2 * (t + 1) = n + 1 ∧ a > 0 ∧ b > 0) → 
  x ≥ 2 ^ ((4 * n ^ 2 - 4) / 3) :=
by
  sorry

end blackboard_problem_l50_50337


namespace largest_number_is_sqrt_7_l50_50326

noncomputable def largest_root (d e f : ℝ) : ℝ :=
if d ≥ e ∧ d ≥ f then d else if e ≥ d ∧ e ≥ f then e else f

theorem largest_number_is_sqrt_7 :
  ∃ (d e f : ℝ), (d + e + f = 3) ∧ (d * e + d * f + e * f = -14) ∧ (d * e * f = 21) ∧ (largest_root d e f = Real.sqrt 7) :=
sorry

end largest_number_is_sqrt_7_l50_50326


namespace instructors_meeting_l50_50437

theorem instructors_meeting (R P E M : ℕ) (hR : R = 5) (hP : P = 8) (hE : E = 10) (hM : M = 9) :
  Nat.lcm (Nat.lcm R P) (Nat.lcm E M) = 360 :=
by
  rw [hR, hP, hE, hM]
  sorry

end instructors_meeting_l50_50437


namespace product_of_two_numbers_l50_50654

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l50_50654


namespace series_items_increase_l50_50561

theorem series_items_increase (n : ℕ) (hn : n ≥ 2) :
  (2^n + 1) - 2^(n-1) - 1 = 2^(n-1) :=
by
  sorry

end series_items_increase_l50_50561


namespace unique_zero_identity_l50_50909

theorem unique_zero_identity (n : ℤ) : (∀ z : ℤ, z + n = z ∧ z * n = 0) → n = 0 :=
by
  intro h
  have h1 : ∀ z : ℤ, z + n = z := fun z => (h z).left
  have h2 : ∀ z : ℤ, z * n = 0 := fun z => (h z).right
  sorry

end unique_zero_identity_l50_50909


namespace range_of_a_zero_value_of_a_minimum_l50_50216

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (7 * a) / x

-- Problem 1: Range of a where f(x) has exactly one zero in its domain
theorem range_of_a_zero (a : ℝ) : 
  (∃! x : ℝ, (0 < x) ∧ f x a = 0) ↔ (a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)}) := sorry

-- Problem 2: Value of a such that the minimum value of f(x) on [e, e^2] is 3
theorem value_of_a_minimum (a : ℝ) : 
  (∃ x : ℝ, (Real.exp 1 ≤ x ∧ x ≤ Real.exp 2) ∧ f x a = 3) ↔ (a = (Real.exp 2)^2 / 7) := sorry

end range_of_a_zero_value_of_a_minimum_l50_50216


namespace initial_temperature_l50_50459

theorem initial_temperature (T_initial : ℝ) 
  (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ) 
  (T_heat : ℝ) (T_cool : ℝ) (T_target : ℝ) (T_final : ℝ) 
  (h1 : heating_rate = 5) (h2 : cooling_rate = 7)
  (h3 : T_target = 240) (h4 : T_final = 170) 
  (h5 : total_time = 46)
  (h6 : T_cool = (T_target - T_final) / cooling_rate)
  (h7: total_time = T_heat + T_cool)
  (h8 : T_heat = (T_target - T_initial) / heating_rate) :
  T_initial = 60 :=
by
  -- Proof yet to be filled in
  sorry

end initial_temperature_l50_50459


namespace inequality_proof_l50_50613

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 := 
by
  sorry

end inequality_proof_l50_50613


namespace N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l50_50953

open Nat -- Natural numbers framework

-- Definitions for game conditions would go here. We assume them to be defined as:
-- structure GameCondition (N : ℕ) :=
-- (players_take_turns_to_circle_numbers_from_1_to_N : Prop)
-- (any_two_circled_numbers_must_be_coprime : Prop)
-- (a_number_cannot_be_circled_twice : Prop)
-- (player_who_cannot_move_loses : Prop)

inductive Player
| first
| second

-- Definitions indicating which player wins for a given N
def first_player_wins (N : ℕ) : Prop := sorry
def second_player_wins (N : ℕ) : Prop := sorry

-- For N = 10
theorem N_10_first_player_wins : first_player_wins 10 := sorry

-- For N = 12
theorem N_12_first_player_wins : first_player_wins 12 := sorry

-- For N = 15
theorem N_15_second_player_wins : second_player_wins 15 := sorry

-- For N = 30
theorem N_30_first_player_wins : first_player_wins 30 := sorry

end N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l50_50953


namespace clara_weight_l50_50509

theorem clara_weight (a c : ℝ) (h1 : a + c = 220) (h2 : c - a = c / 3) : c = 88 :=
by
  sorry

end clara_weight_l50_50509


namespace books_sold_at_overall_loss_l50_50861

-- Defining the conditions and values
def total_cost : ℝ := 540
def C1 : ℝ := 315
def loss_percentage_C1 : ℝ := 0.15
def gain_percentage_C2 : ℝ := 0.19
def C2 : ℝ := total_cost - C1
def loss_C1 := (loss_percentage_C1 * C1)
def SP1 := C1 - loss_C1
def gain_C2 := (gain_percentage_C2 * C2)
def SP2 := C2 + gain_C2
def total_selling_price := SP1 + SP2
def overall_loss := total_cost - total_selling_price

-- Formulating the theorem based on the conditions and required proof
theorem books_sold_at_overall_loss : overall_loss = 4.50 := 
by 
  sorry

end books_sold_at_overall_loss_l50_50861


namespace value_of_expression_l50_50647

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) : 4 / 7 + (2 * q - p) / (2 * q + p) = 1 := by
  sorry

end value_of_expression_l50_50647


namespace pirate_treasure_chest_coins_l50_50014

theorem pirate_treasure_chest_coins:
  ∀ (gold_coins silver_coins bronze_coins: ℕ) (chests: ℕ),
    gold_coins = 3500 →
    silver_coins = 500 →
    bronze_coins = 2 * silver_coins →
    chests = 5 →
    (gold_coins / chests + silver_coins / chests + bronze_coins / chests = 1000) :=
by
  intros gold_coins silver_coins bronze_coins chests gold_eq silv_eq bron_eq chest_eq
  sorry

end pirate_treasure_chest_coins_l50_50014


namespace school_travel_time_is_12_l50_50358

noncomputable def time_to_school (T : ℕ) : Prop :=
  let extra_time := 6
  let total_distance_covered := 2 * extra_time
  T = total_distance_covered

theorem school_travel_time_is_12 :
  ∃ T : ℕ, time_to_school T ∧ T = 12 :=
by
  sorry

end school_travel_time_is_12_l50_50358


namespace num_divisors_of_factorial_9_multiple_3_l50_50335

-- Define the prime factorization of 9!
def factorial_9 := 2^7 * 3^4 * 5 * 7

-- Define the conditions for the exponents a, b, c, d
def valid_exponents (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 7) ∧ (1 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1)

-- Define the number of valid exponent combinations
def num_valid_combinations : ℕ :=
  8 * 4 * 2 * 2

-- Theorem stating that the number of divisors of 9! that are multiples of 3 is 128
theorem num_divisors_of_factorial_9_multiple_3 : num_valid_combinations = 128 := by
  sorry

end num_divisors_of_factorial_9_multiple_3_l50_50335


namespace find_income_l50_50095

noncomputable def income_expenditure_proof : Prop := 
  ∃ (x : ℕ), (5 * x - 4 * x = 3600) ∧ (5 * x = 18000)

theorem find_income : income_expenditure_proof :=
  sorry

end find_income_l50_50095


namespace projectile_height_time_l50_50762

-- Define constants and the height function
def a : ℝ := -4.9
def b : ℝ := 29.75
def c : ℝ := -35
def y (t : ℝ) : ℝ := a * t^2 + b * t

-- Problem statement
theorem projectile_height_time (h : y t = 35) : ∃ t : ℝ, 0 < t ∧ abs (t - 1.598) < 0.001 := by
  -- Placeholder for actual proof
  sorry

end projectile_height_time_l50_50762


namespace fg_value_l50_50330

def g (x : ℤ) : ℤ := 4 * x - 3
def f (x : ℤ) : ℤ := 6 * x + 2

theorem fg_value : f (g 5) = 104 := by
  sorry

end fg_value_l50_50330


namespace capital_of_a_l50_50850

variable (P P' TotalCapital Ca : ℝ)

theorem capital_of_a 
  (h1 : a_income_5_percent = (2/3) * P)
  (h2 : a_income_7_percent = (2/3) * P')
  (h3 : a_income_7_percent - a_income_5_percent = 200)
  (h4 : P = 0.05 * TotalCapital)
  (h5 : P' = 0.07 * TotalCapital)
  : Ca = (2/3) * TotalCapital :=
by
  sorry

end capital_of_a_l50_50850


namespace polygon_sides_l50_50877

theorem polygon_sides (n : ℕ) (D : ℕ) (hD : D = 77) (hFormula : D = n * (n - 3) / 2) (hVertex : n = n) : n + 1 = 15 :=
by
  sorry

end polygon_sides_l50_50877


namespace stamps_per_page_l50_50687

def a : ℕ := 924
def b : ℕ := 1386
def c : ℕ := 1848

theorem stamps_per_page : gcd (gcd a b) c = 462 :=
sorry

end stamps_per_page_l50_50687


namespace tony_water_drink_l50_50928

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l50_50928


namespace income_exceeds_previous_l50_50316

noncomputable def a_n (a b : ℝ) (n : ℕ) : ℝ :=
if n = 1 then a
else a * (2 / 3)^(n - 1) + b * (3 / 2)^(n - 2)

theorem income_exceeds_previous (a b : ℝ) (h : b ≥ 3 * a / 8) (n : ℕ) (hn : n ≥ 2) : 
  a_n a b n ≥ a :=
sorry

end income_exceeds_previous_l50_50316


namespace min_value_of_f_l50_50414

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 3)

theorem min_value_of_f (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx1_distinct : x1 ≠ x2) (hx2_pos : 0 < x2)
  (h_f_eq : f x1 = f x2) : (1 / x1 + 9 / x2) = 2 / 3 :=
by
  sorry

end min_value_of_f_l50_50414


namespace repeating_decimal_as_fraction_l50_50190

-- Given conditions
def repeating_decimal : ℚ := 7 + 832 / 999

-- Goal: Prove that the repeating decimal 7.\overline{832} equals 70/9
theorem repeating_decimal_as_fraction : repeating_decimal = 70 / 9 := by
  unfold repeating_decimal
  sorry

end repeating_decimal_as_fraction_l50_50190


namespace candy_store_revenue_l50_50731

/-- A candy store sold 20 pounds of fudge for $2.50 per pound,
    5 dozen chocolate truffles for $1.50 each, 
    and 3 dozen chocolate-covered pretzels at $2.00 each.
    Prove that the total money made by the candy store is $212.00. --/
theorem candy_store_revenue :
  let fudge_pounds := 20
  let fudge_price_per_pound := 2.50
  let truffle_dozen := 5
  let truffle_price_each := 1.50
  let pretzel_dozen := 3
  let pretzel_price_each := 2.00
  (fudge_pounds * fudge_price_per_pound) + 
  (truffle_dozen * 12 * truffle_price_each) + 
  (pretzel_dozen * 12 * pretzel_price_each) = 212 :=
by
  sorry

end candy_store_revenue_l50_50731


namespace screws_weight_l50_50586

theorem screws_weight (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 319) 
  (h2 : 2 * x + 3 * y = 351) : 
  x = 51 ∧ y = 83 :=
by 
  sorry

end screws_weight_l50_50586


namespace area_of_intersection_of_two_circles_l50_50515

open Real

noncomputable def area_intersection (r : ℝ) (c1 c2 : ℝ × ℝ) : ℝ :=
  let quarter_circle_area := (1/4) * π * r^2
  let triangle_area := (1/2) * r^2
  let segment_area := quarter_circle_area - triangle_area
  2 * segment_area

theorem area_of_intersection_of_two_circles :
  area_intersection 3 (3, 0) (0, 3) = (9 * π - 18) / 2 :=
by
  -- This will be proven by the steps of the provided solution.
  sorry

end area_of_intersection_of_two_circles_l50_50515


namespace area_of_right_triangle_l50_50553

theorem area_of_right_triangle (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 
  ∃ A : ℝ, A = (k^2) / (2 * m) :=
by
  sorry

end area_of_right_triangle_l50_50553


namespace count_three_digit_perfect_squares_divisible_by_4_l50_50397

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l50_50397


namespace inequality_proof_l50_50521

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (2 * x * y) / (x + y) + Real.sqrt ((x ^ 2 + y ^ 2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) :=
by
  sorry

end inequality_proof_l50_50521


namespace smallest_positive_multiple_of_45_divisible_by_3_l50_50772

theorem smallest_positive_multiple_of_45_divisible_by_3 
  (x : ℕ) (hx: x > 0) : ∃ y : ℕ, y = 45 ∧ 45 ∣ y ∧ 3 ∣ y ∧ ∀ z : ℕ, (45 ∣ z ∧ 3 ∣ z ∧ z > 0) → z ≥ y :=
by
  sorry

end smallest_positive_multiple_of_45_divisible_by_3_l50_50772


namespace brandon_investment_percentage_l50_50050

noncomputable def jackson_initial_investment : ℕ := 500
noncomputable def brandon_initial_investment : ℕ := 500
noncomputable def jackson_final_investment : ℕ := 2000
noncomputable def difference_in_investments : ℕ := 1900
noncomputable def brandon_final_investment : ℕ := jackson_final_investment - difference_in_investments

theorem brandon_investment_percentage :
  (brandon_final_investment : ℝ) / (brandon_initial_investment : ℝ) * 100 = 20 := by
  sorry

end brandon_investment_percentage_l50_50050


namespace emily_catch_catfish_l50_50715

-- Definitions based on given conditions
def num_trout : ℕ := 4
def num_bluegills : ℕ := 5
def weight_trout : ℕ := 2
def weight_catfish : ℚ := 1.5
def weight_bluegill : ℚ := 2.5
def total_fish_weight : ℚ := 25

-- Lean statement to prove the number of catfish
theorem emily_catch_catfish : ∃ (num_catfish : ℕ), 
  num_catfish * weight_catfish = total_fish_weight - (num_trout * weight_trout + num_bluegills * weight_bluegill) ∧
  num_catfish = 3 := by
  sorry

end emily_catch_catfish_l50_50715


namespace ratio_of_shirt_to_pants_l50_50240

theorem ratio_of_shirt_to_pants
    (total_cost : ℕ)
    (price_pants : ℕ)
    (price_shoes : ℕ)
    (price_shirt : ℕ)
    (h1 : total_cost = 340)
    (h2 : price_pants = 120)
    (h3 : price_shoes = price_pants + 10)
    (h4 : price_shirt = total_cost - (price_pants + price_shoes)) :
    price_shirt * 4 = price_pants * 3 := sorry

end ratio_of_shirt_to_pants_l50_50240


namespace range_of_m_l50_50073

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end range_of_m_l50_50073


namespace train_crossing_time_l50_50641

def speed := 60 -- in km/hr
def length := 300 -- in meters
def speed_in_m_per_s := (60 * 1000) / 3600 -- converting speed from km/hr to m/s
def expected_time := 18 -- in seconds

theorem train_crossing_time :
  (300 / (speed_in_m_per_s)) = expected_time :=
sorry

end train_crossing_time_l50_50641


namespace complete_square_l50_50383

theorem complete_square (x : ℝ) : (x^2 + 4*x - 1 = 0) → ((x + 2)^2 = 5) :=
by
  intro h
  sorry

end complete_square_l50_50383


namespace value_of_expression_l50_50792

theorem value_of_expression : (0.3 : ℝ)^2 + 0.1 = 0.19 := 
by sorry

end value_of_expression_l50_50792


namespace type1_pieces_count_l50_50938

theorem type1_pieces_count (n : ℕ) (pieces : ℕ → ℕ)  (nonNegative : ∀ i, pieces i ≥ 0) :
  pieces 1 ≥ 4 * n - 1 :=
sorry

end type1_pieces_count_l50_50938


namespace grocer_initial_stock_l50_50572

theorem grocer_initial_stock 
  (x : ℝ) 
  (h1 : 0.20 * x + 70 = 0.30 * (x + 100)) : 
  x = 400 := by
  sorry

end grocer_initial_stock_l50_50572


namespace isabella_hair_length_l50_50300

theorem isabella_hair_length (h : ℕ) (g : ℕ) (future_length : ℕ) (hg : g = 4) (future_length_eq : future_length = 22) :
  h = future_length - g :=
by
  rw [future_length_eq, hg]
  exact sorry

end isabella_hair_length_l50_50300


namespace range_satisfying_f_inequality_l50_50662

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (1 + |x|) - (1 / (1 + x^2))

theorem range_satisfying_f_inequality : 
  ∀ x : ℝ, (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) :=
by
  intro x hx
  sorry

end range_satisfying_f_inequality_l50_50662


namespace determine_value_of_x_l50_50756

theorem determine_value_of_x {b x : ℝ} (hb : 1 < b) (hx : 0 < x) 
  (h_eq : (4 * x)^(Real.logb b 2) = (5 * x)^(Real.logb b 3)) : 
  x = (4 / 5)^(Real.logb (3 / 2) b) :=
by
  sorry

end determine_value_of_x_l50_50756


namespace integer_solutions_l50_50378

theorem integer_solutions (m n : ℤ) :
  m^3 - n^3 = 2 * m * n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
sorry

end integer_solutions_l50_50378


namespace pq_sum_eight_l50_50211

theorem pq_sum_eight
  (p q : ℤ)
  (hp1 : p > 1)
  (hq1 : q > 1)
  (hs1 : (2 * q - 1) % p = 0)
  (hs2 : (2 * p - 1) % q = 0) : p + q = 8 := 
sorry

end pq_sum_eight_l50_50211


namespace travel_distance_l50_50480

theorem travel_distance (x t : ℕ) (h : t = 14400) (h_eq : 12 * x + 12 * (2 * x) = t) : x = 400 :=
by
  sorry

end travel_distance_l50_50480


namespace calculate_minus_one_minus_two_l50_50675

theorem calculate_minus_one_minus_two : -1 - 2 = -3 := by
  sorry

end calculate_minus_one_minus_two_l50_50675


namespace win_sector_area_l50_50910

/-- Given a circular spinner with a radius of 8 cm and the probability of winning being 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (P_win : ℝ) (area_WIN : ℝ) :
  r = 8 → P_win = 3 / 8 → area_WIN = 24 * Real.pi := by
sorry

end win_sector_area_l50_50910


namespace clock_in_probability_l50_50903

-- Definitions
def start_time := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_start := 495 -- 8:15 in minutes from 00:00 (495 minutes)
def arrival_start := 470 -- 7:50 in minutes from 00:00 (470 minutes)
def arrival_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)

-- Conditions
def arrival_window := arrival_end - arrival_start -- 40 minutes window
def valid_clock_in_window := valid_clock_in_end - valid_clock_in_start -- 15 minutes window

-- Required proof statement
theorem clock_in_probability :
  (valid_clock_in_window : ℚ) / (arrival_window : ℚ) = 3 / 8 :=
by
  sorry

end clock_in_probability_l50_50903


namespace possible_measures_of_angle_X_l50_50105

theorem possible_measures_of_angle_X :
  ∃ (n : ℕ), n = 17 ∧ ∀ (X Y : ℕ), 
    (X > 0) → 
    (Y > 0) → 
    (∃ k : ℕ, k ≥ 1 ∧ X = k * Y) → 
    X + Y = 180 → 
    ∃ d : ℕ, d ∈ {d | d ∣ 180 } ∧ d ≥ 2 :=
by
  sorry

end possible_measures_of_angle_X_l50_50105


namespace least_multiple_of_15_greater_than_500_l50_50286

theorem least_multiple_of_15_greater_than_500 : 
  ∃ (n : ℕ), n > 500 ∧ (∃ (k : ℕ), n = 15 * k) ∧ (n = 510) :=
by
  sorry

end least_multiple_of_15_greater_than_500_l50_50286


namespace fraction_of_product_l50_50465

theorem fraction_of_product : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_product_l50_50465


namespace find_constants_l50_50020

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x + 1

noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ( (x - a + Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3) +
  ( (x - a - Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3)

theorem find_constants (a b c : ℝ) (h1 : f_inv (1:ℝ) a b c = 0)
  (ha : a = 1) (hb : b = 2) (hc : c = 5) : a + 10 * b + 100 * c = 521 :=
by
  rw [ha, hb, hc]
  norm_num

end find_constants_l50_50020


namespace sum_T_19_34_51_l50_50151

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2 : ℕ) else (n + 1) / 2

def T (n : ℕ) : ℤ :=
  2 + S n

theorem sum_T_19_34_51 : T 19 + T 34 + T 51 = 25 := 
by
  -- Add the steps here
  sorry

end sum_T_19_34_51_l50_50151


namespace Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l50_50614

structure Person :=
  (name : String)
  (age : Nat)
  (location : String)
  (occupation : String)

def kate : Person := {name := "Kate Hashimoto", age := 30, location := "New York", occupation := "CPA"}

-- Conditions
def lives_on_15_dollars_a_month (p : Person) : Prop := p = kate → true
def dumpster_diving (p : Person) : Prop := p = kate → true
def upscale_stores_discard_good_items : Prop := true
def frugal_habits (p : Person) : Prop := p = kate → true

-- Proof
theorem Kate_relies_on_dumpster_diving : lives_on_15_dollars_a_month kate ∧ dumpster_diving kate → true := 
by sorry

theorem Upscale_stores_discard_items : upscale_stores_discard_good_items → true := 
by sorry

theorem Kate_frugal_habits : frugal_habits kate → true := 
by sorry

end Kate_relies_on_dumpster_diving_Upscale_stores_discard_items_Kate_frugal_habits_l50_50614


namespace weight_of_empty_box_l50_50534

theorem weight_of_empty_box (w12 w8 w : ℝ) (h1 : w12 = 11.48) (h2 : w8 = 8.12) (h3 : ∀ b : ℕ, b > 0 → w = 0.84) :
  w8 - 8 * w = 1.40 :=
by
  sorry

end weight_of_empty_box_l50_50534


namespace wheel_distance_l50_50617

noncomputable def diameter : ℝ := 9
noncomputable def revolutions : ℝ := 18.683651804670912
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def circumference (d : ℝ) : ℝ := pi_approx * d
noncomputable def distance (r : ℝ) (c : ℝ) : ℝ := r * c

theorem wheel_distance : distance revolutions (circumference diameter) = 528.219 :=
by
  unfold distance circumference diameter revolutions pi_approx
  -- Here we would perform the calculation and show that the result is approximately 528.219
  sorry

end wheel_distance_l50_50617


namespace range_of_k_l50_50002

noncomputable def f (k x : ℝ) := k * x - Real.exp x
noncomputable def g (x : ℝ) := Real.exp x / x

theorem range_of_k (k : ℝ) (h : ∃ x : ℝ, x ≠ 0 ∧ f k x = 0) :
  k < 0 ∨ k ≥ Real.exp 1 := sorry

end range_of_k_l50_50002


namespace nat_divisible_by_five_l50_50642

theorem nat_divisible_by_five (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  have h₀ : ¬ ((5 ∣ a) ∨ (5 ∣ b)) → ¬ (5 ∣ (a * b)) := sorry
  -- Proof by contradiction steps go here
  sorry

end nat_divisible_by_five_l50_50642


namespace ab_cardinals_l50_50996

open Set

/-- a|A| = b|B| given the conditions.
1. a and b are positive integers.
2. A and B are finite sets of integers such that:
   a. A and B are disjoint.
   b. If an integer i belongs to A or to B, then i + a ∈ A or i - b ∈ B.
-/
theorem ab_cardinals 
  (a b : ℕ) (A B : Finset ℤ) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (disjoint_AB : Disjoint A B)
  (condition_2 : ∀ i : ℤ, i ∈ A ∪ B → (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := 
sorry

end ab_cardinals_l50_50996


namespace find_divisor_l50_50380

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161) 
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 16 :=
by
  sorry

end find_divisor_l50_50380


namespace find_x_l50_50232
-- The first priority is to ensure the generated Lean code can be built successfully.

theorem find_x (x : ℤ) (h : 9823 + x = 13200) : x = 3377 :=
by
  sorry

end find_x_l50_50232


namespace lucy_crayons_l50_50955

theorem lucy_crayons (W L : ℕ) (h1 : W = 1400) (h2 : W = L + 1110) : L = 290 :=
by {
  sorry
}

end lucy_crayons_l50_50955


namespace number_of_real_solutions_l50_50770

noncomputable def f (x : ℝ) : ℝ := 2^(-x) + x^2 - 3

theorem number_of_real_solutions :
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (∀ x : ℝ, f x = 0 → (x = x₁ ∨ x = x₂)) :=
by
  sorry

end number_of_real_solutions_l50_50770


namespace only_value_of_k_l50_50476

def A (k a b : ℕ) : ℚ := (a + b : ℚ) / (a^2 + k^2 * b^2 - k^2 * a * b : ℚ)

theorem only_value_of_k : (∀ a b : ℕ, 0 < a → 0 < b → ¬ (∃ c d : ℕ, 1 < c ∧ A 1 a b = (c : ℚ) / (d : ℚ))) → k = 1 := 
    by sorry  -- proof omitted

-- Note: 'only_value_of_k' states that given the conditions, there is no k > 1 that makes A(k, a, b) a composite number, hence k must be 1.

end only_value_of_k_l50_50476


namespace toms_total_score_l50_50225

def points_per_enemy : ℕ := 10
def enemies_killed : ℕ := 175

def base_score (enemies : ℕ) : ℝ := enemies * points_per_enemy

def bonus_percentage (enemies : ℕ) : ℝ :=
  if 100 ≤ enemies ∧ enemies < 150 then 0.50
  else if 150 ≤ enemies ∧ enemies < 200 then 0.75
  else if enemies ≥ 200 then 1.00
  else 0.0

def total_score (enemies : ℕ) : ℝ :=
  let base := base_score enemies
  let bonus := base * bonus_percentage enemies
  base + bonus

theorem toms_total_score :
  total_score enemies_killed = 3063 :=
by
  -- The proof will show the computed total score
  -- matches the expected value
  sorry

end toms_total_score_l50_50225


namespace symmetric_circle_equation_l50_50214

-- Define original circle equation
def original_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Define symmetric circle equation
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

theorem symmetric_circle_equation (x y : ℝ) : 
  symmetric_circle x y ↔ original_circle (-x) y :=
by sorry

end symmetric_circle_equation_l50_50214


namespace common_difference_l50_50703

theorem common_difference (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
    (h₁ : a 5 + a 6 = -10)
    (h₂ : S 14 = -14)
    (h₃ : ∀ n, S n = n * (a 1 + a n) / 2)
    (h₄ : ∀ n, a (n + 1) = a n + d) :
  d = 2 :=
sorry

end common_difference_l50_50703


namespace slope_of_line_eq_slope_of_line_l50_50159

theorem slope_of_line_eq (x y : ℝ) (h : 4 * x + 6 * y = 24) : (6 * y = -4 * x + 24) → (y = - (2 : ℝ) / 3 * x + 4) :=
by
  intro h1
  sorry

theorem slope_of_line (x y m : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = - (2 : ℝ) / 3 * x + 4) : m = - (2 : ℝ) / 3 :=
by
  sorry

end slope_of_line_eq_slope_of_line_l50_50159


namespace sufficient_not_necessary_l50_50215

theorem sufficient_not_necessary (a : ℝ) (h : a ≠ 0) : 
  (a > 1 → a > 1 / a) ∧ (¬ (a > 1) → a > 1 / a → -1 < a ∧ a < 0) :=
sorry

end sufficient_not_necessary_l50_50215


namespace san_francisco_superbowl_probability_l50_50506

theorem san_francisco_superbowl_probability
  (P_play P_not_play : ℝ)
  (k : ℝ)
  (h1 : P_play = k * P_not_play)
  (h2 : P_play + P_not_play = 1) :
  k > 0 :=
sorry

end san_francisco_superbowl_probability_l50_50506


namespace arithmetic_sequence_50th_term_l50_50917

-- Definitions based on the conditions stated
def first_term := 3
def common_difference := 5
def n := 50

-- Function to calculate the n-th term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- The theorem that needs to be proven
theorem arithmetic_sequence_50th_term : nth_term first_term common_difference n = 248 := 
by
  sorry

end arithmetic_sequence_50th_term_l50_50917


namespace complex_root_product_l50_50664

theorem complex_root_product (w : ℂ) (hw1 : w^3 = 1) (hw2 : w^2 + w + 1 = 0) :
(1 - w + w^2) * (1 + w - w^2) = 4 :=
sorry

end complex_root_product_l50_50664


namespace line_equation_l50_50812

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end line_equation_l50_50812


namespace graph_does_not_pass_second_quadrant_l50_50776

noncomputable def y_function (a b : ℝ) (x : ℝ) : ℝ := a^x + b

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) : 
  ∀ x y : ℝ, (y = y_function a b x) → ¬(x < 0 ∧ y > 0) := by
  sorry

end graph_does_not_pass_second_quadrant_l50_50776


namespace rebus_system_solution_l50_50145

theorem rebus_system_solution :
  ∃ (M A H P h : ℕ), 
  (M > 0) ∧ (P > 0) ∧ 
  (M ≠ A) ∧ (M ≠ H) ∧ (M ≠ P) ∧ (M ≠ h) ∧
  (A ≠ H) ∧ (A ≠ P) ∧ (A ≠ h) ∧ 
  (H ≠ P) ∧ (H ≠ h) ∧ (P ≠ h) ∧
  ((M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P) ∧ 
  ((A * 10 + M) * (A * 10 + M) = P * 100 + h * 10 + M) ∧ 
  (((M = 1) ∧ (A = 3) ∧ (H = 6) ∧ (P = 9) ∧ (h = 6)) ∨
   ((M = 3) ∧ (A = 1) ∧ (H = 9) ∧ (P = 6) ∧ (h = 9))) :=
by
  sorry

end rebus_system_solution_l50_50145


namespace max_homework_ratio_l50_50381

theorem max_homework_ratio 
  (H : ℕ) -- time spent on history tasks
  (biology_time : ℕ)
  (total_homework_time : ℕ)
  (geography_time : ℕ)
  (history_geography_relation : geography_time = 3 * H)
  (total_time_relation : total_homework_time = 180)
  (biology_time_known : biology_time = 20)
  (sum_time_relation : H + geography_time + biology_time = total_homework_time) :
  H / biology_time = 2 :=
by
  sorry

end max_homework_ratio_l50_50381


namespace imaginary_part_of_z_l50_50855

open Complex

-- Condition
def equation_z (z : ℂ) : Prop := (z * (1 + I) * I^3) / (1 - I) = 1 - I

-- Problem statement
theorem imaginary_part_of_z (z : ℂ) (h : equation_z z) : z.im = -1 := 
by 
  sorry

end imaginary_part_of_z_l50_50855


namespace find_dividend_l50_50058

variable (Divisor Quotient Remainder Dividend : ℕ)
variable (h₁ : Divisor = 15)
variable (h₂ : Quotient = 8)
variable (h₃ : Remainder = 5)

theorem find_dividend : Dividend = 125 ↔ Dividend = Divisor * Quotient + Remainder := by
  sorry

end find_dividend_l50_50058


namespace quadratic_real_roots_range_k_l50_50101

-- Define the quadratic function
def quadratic_eq (k x : ℝ) : ℝ := k * x^2 - 6 * x + 9

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for the quadratic equation to have distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_real_roots_range_k (k : ℝ) :
  has_two_distinct_real_roots k (-6) 9 ↔ k < 1 ∧ k ≠ 0 := 
by
  sorry

end quadratic_real_roots_range_k_l50_50101


namespace parallel_lines_slope_l50_50507

theorem parallel_lines_slope (m : ℝ) (h : (x + (1 + m) * y + m - 2 = 0) ∧ (m * x + 2 * y + 6 = 0)) :
  m = 1 ∨ m = -2 :=
  sorry

end parallel_lines_slope_l50_50507


namespace initial_walking_speed_l50_50976

theorem initial_walking_speed
  (t : ℝ) -- Time in minutes for bus to reach the bus stand from when the person starts walking
  (h₁ : 5 = 5 * ((t - 5) / 60)) -- When walking at 5 km/h, person reaches 5 minutes early
  (h₂ : 5 = v * ((t + 10) / 60)) -- At initial speed v, person misses the bus by 10 minutes
  : v = 4 := 
by
  sorry

end initial_walking_speed_l50_50976


namespace cube_root_simplification_l50_50602

theorem cube_root_simplification (c d : ℕ) (h1 : c = 3) (h2 : d = 100) : c + d = 103 :=
by
  sorry

end cube_root_simplification_l50_50602


namespace sum_of_real_solutions_l50_50502

theorem sum_of_real_solutions :
  ∀ x : ℝ, (x - 3) / (x^2 + 5 * x + 2) = (x - 6) / (x^2 - 11 * x) →
  (∃ r1 r2 : ℝ, r1 + r2 = 46 / 13) :=
by
  sorry

end sum_of_real_solutions_l50_50502


namespace fewer_mpg_in_city_l50_50805

theorem fewer_mpg_in_city
  (highway_miles : ℕ)
  (city_miles : ℕ)
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (tank_size : ℝ) :
  highway_miles = 462 →
  city_miles = 336 →
  city_mpg = 32 →
  tank_size = 336 / 32 →
  highway_mpg = 462 / tank_size →
  (highway_mpg - city_mpg) = 12 :=
by
  intros h_highway_miles h_city_miles h_city_mpg h_tank_size h_highway_mpg
  sorry

end fewer_mpg_in_city_l50_50805


namespace Evan_earnings_Markese_less_than_Evan_l50_50041

-- Definitions from conditions
def MarkeseEarnings : ℕ := 16
def TotalEarnings : ℕ := 37

-- Theorem statements
theorem Evan_earnings (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E = 21 :=
by {
  sorry
}

theorem Markese_less_than_Evan (E : ℕ) (h : E + MarkeseEarnings = TotalEarnings) : E - MarkeseEarnings = 5 :=
by {
  sorry
}

end Evan_earnings_Markese_less_than_Evan_l50_50041


namespace no_positive_integers_abc_l50_50113

theorem no_positive_integers_abc :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) :=
sorry

end no_positive_integers_abc_l50_50113


namespace intersection_eq_expected_result_l50_50856

def M := { x : ℝ | x - 2 > 0 }
def N := { x : ℝ | (x - 3) * (x - 1) < 0 }
def expected_result := { x : ℝ | 2 < x ∧ x < 3 }

theorem intersection_eq_expected_result : M ∩ N = expected_result := 
by
  sorry

end intersection_eq_expected_result_l50_50856


namespace length_of_ST_l50_50069

theorem length_of_ST (LM MN NL: ℝ) (LR : ℝ) (LT TR LS SR: ℝ) 
  (h1: LM = 8) (h2: MN = 10) (h3: NL = 6) (h4: LR = 6) 
  (h5: LT = 8 / 3) (h6: TR = 10 / 3) (h7: LS = 9 / 4) (h8: SR = 15 / 4) :
  LS - LT = -5 / 12 :=
by
  sorry

end length_of_ST_l50_50069


namespace m_greater_than_p_l50_50712

theorem m_greater_than_p (p m n : ℕ) (prime_p : Prime p) (pos_m : 0 < m) (pos_n : 0 < n)
    (eq : p^2 + m^2 = n^2) : m > p := 
by 
  sorry

end m_greater_than_p_l50_50712


namespace tangent_line_at_pi_over_4_l50_50088

noncomputable def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x * Real.tan x

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  (2 + Real.pi) * x - y - (Real.pi^2 / 4) = 0

theorem tangent_line_at_pi_over_4 :
  tangent_eq (Real.pi / 4) (Real.pi / 2) →
  tangent_line_eq (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end tangent_line_at_pi_over_4_l50_50088


namespace find_angle_CDE_l50_50440

-- Definition of the angles and their properties
variables {A B C D E : Type}

-- Hypotheses
def angleA_is_right (angleA: ℝ) : Prop := angleA = 90
def angleB_is_right (angleB: ℝ) : Prop := angleB = 90
def angleC_is_right (angleC: ℝ) : Prop := angleC = 90
def angleAEB_value (angleAEB : ℝ) : Prop := angleAEB = 40
def angleBED_eq_angleBDE (angleBED angleBDE : ℝ) : Prop := angleBED = angleBDE

-- The theorem to be proved
theorem find_angle_CDE 
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ) (angleAEB : ℝ) (angleBED angleBDE : ℝ) (angleCDE : ℝ) :
  angleA_is_right angleA → 
  angleB_is_right angleB → 
  angleC_is_right angleC → 
  angleAEB_value angleAEB → 
  angleBED_eq_angleBDE angleBED angleBDE →
  angleBED = 45 →
  angleCDE = 95 :=
by
  intros
  sorry


end find_angle_CDE_l50_50440


namespace dihedral_angle_sum_bounds_l50_50701

variable (α β γ : ℝ)

/-- The sum of the internal dihedral angles of a trihedral angle is greater than 180 degrees and less than 540 degrees. -/
theorem dihedral_angle_sum_bounds (hα: α < 180) (hβ: β < 180) (hγ: γ < 180) : 180 < α + β + γ ∧ α + β + γ < 540 :=
by
  sorry

end dihedral_angle_sum_bounds_l50_50701


namespace B_initial_investment_l50_50948

theorem B_initial_investment (B : ℝ) :
  let A_initial := 2000
  let A_months := 12
  let A_withdraw := 1000
  let B_advanced := 1000
  let months_before_change := 8
  let months_after_change := 4
  let total_profit := 630
  let A_share := 175
  let B_share := total_profit - A_share
  let A_investment := A_initial * A_months
  let B_investment := (B * months_before_change) + ((B + B_advanced) * months_after_change)
  (B_share / A_share = B_investment / A_investment) →
  B = 4866.67 :=
sorry

end B_initial_investment_l50_50948


namespace ski_price_l50_50968

variable {x y : ℕ}

theorem ski_price (h1 : 2 * x + y = 340) (h2 : 3 * x + 2 * y = 570) : x = 110 ∧ y = 120 := by
  sorry

end ski_price_l50_50968


namespace find_ab_l50_50487

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l50_50487


namespace value_of_a2_l50_50126

variable {R : Type*} [Ring R] (x a_0 a_1 a_2 a_3 : R)

theorem value_of_a2 
  (h : ∀ x : R, x^3 = a_0 + a_1 * (x - 2) + a_2 * (x - 2)^2 + a_3 * (x - 2)^3) :
  a_2 = 6 :=
sorry

end value_of_a2_l50_50126


namespace shortest_distance_between_circles_l50_50067

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

theorem shortest_distance_between_circles :
  let c₁ := (4, -3)
  let r₁ := 4
  let c₂ := (-5, 1)
  let r₂ := 1
  distance c₁ c₂ - (r₁ + r₂) = Real.sqrt 97 - 5 :=
by
  sorry

end shortest_distance_between_circles_l50_50067


namespace frog_weight_difference_l50_50258

theorem frog_weight_difference
  (large_frog_weight : ℕ)
  (small_frog_weight : ℕ)
  (h1 : large_frog_weight = 10 * small_frog_weight)
  (h2 : large_frog_weight = 120) :
  large_frog_weight - small_frog_weight = 108 :=
by
  sorry

end frog_weight_difference_l50_50258


namespace cost_of_600_pages_l50_50471

def cost_per_5_pages := 10 -- 10 cents for 5 pages
def pages_to_copy := 600
def expected_cost := 12 * 100 -- 12 dollars in cents

theorem cost_of_600_pages : pages_to_copy * (cost_per_5_pages / 5) = expected_cost := by
  sorry

end cost_of_600_pages_l50_50471


namespace exists_two_elements_l50_50982

variable (F : Finset (Finset ℕ))
variable (h1 : ∀ (A B : Finset ℕ), A ∈ F → B ∈ F → (A ∪ B) ∈ F)
variable (h2 : ∀ (A : Finset ℕ), A ∈ F → ¬ (3 ∣ A.card))

theorem exists_two_elements : ∃ (x y : ℕ), ∀ (A : Finset ℕ), A ∈ F → x ∈ A ∨ y ∈ A :=
by
  sorry

end exists_two_elements_l50_50982


namespace Joey_study_time_l50_50094

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end Joey_study_time_l50_50094


namespace max_coins_identifiable_l50_50421

theorem max_coins_identifiable (n : ℕ) : exists (c : ℕ), c = 2 * n^2 + 1 :=
by
  sorry

end max_coins_identifiable_l50_50421


namespace work_completion_l50_50783

theorem work_completion (original_men planned_days absent_men remaining_men completion_days : ℕ) :
  original_men = 180 → 
  planned_days = 55 →
  absent_men = 15 →
  remaining_men = original_men - absent_men →
  remaining_men * completion_days = original_men * planned_days →
  completion_days = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end work_completion_l50_50783


namespace cost_to_paint_cube_l50_50775

theorem cost_to_paint_cube :
  let cost_per_kg := 50
  let coverage_per_kg := 20
  let side_length := 20
  let surface_area := 6 * (side_length * side_length)
  let amount_of_paint := surface_area / coverage_per_kg
  let total_cost := amount_of_paint * cost_per_kg
  total_cost = 6000 :=
by
  sorry

end cost_to_paint_cube_l50_50775


namespace fruit_salad_total_l50_50965

def fruit_salad_problem (R_red G R_rasp total_fruit : ℕ) : Prop :=
  R_red = 67 ∧ (3 * G + 7 = 67) ∧ (R_rasp = G - 5) ∧ (total_fruit = R_red + G + R_rasp)

theorem fruit_salad_total (R_red G R_rasp : ℕ) (total_fruit : ℕ) :
  fruit_salad_problem R_red G R_rasp total_fruit → total_fruit = 102 :=
by
  intro h
  sorry

end fruit_salad_total_l50_50965


namespace tile_chessboard_2n_l50_50322

theorem tile_chessboard_2n (n : ℕ) (board : Fin (2^n) → Fin (2^n) → Prop) (i j : Fin (2^n)) 
  (h : board i j = false) : ∃ tile : Fin (2^n) → Fin (2^n) → Bool, 
  (∀ i j, board i j = true ↔ tile i j = true) :=
sorry

end tile_chessboard_2n_l50_50322


namespace sum_of_possible_values_of_y_l50_50432

-- Definitions of the conditions
variables (y : ℝ)
-- Angle measures in degrees
variables (a b c : ℝ)
variables (isosceles : Bool)

-- Given conditions
def is_isosceles_triangle (a b c : ℝ) (isosceles : Bool) : Prop :=
  isosceles = true ∧ (a = b ∨ b = c ∨ c = a)

-- Sum of angles in any triangle
def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180

-- Main statement to be proven
theorem sum_of_possible_values_of_y (y : ℝ) (a b c : ℝ) (isosceles : Bool) :
  is_isosceles_triangle a b c isosceles →
  sum_of_angles_in_triangle a b c →
  ((y = 60) → (a = y ∨ b = y ∨ c = y)) →
  isosceles = true → a = 60 ∨ b = 60 ∨ c = 60 →
  y + y + y = 180 :=
by
  intros h1 h2 h3 h4 h5
  sorry  -- Proof will be provided here

end sum_of_possible_values_of_y_l50_50432


namespace neg_product_B_l50_50015

def expr_A := (-1 / 3) * (1 / 4) * (-6)
def expr_B := (-9) * (1 / 8) * (-4 / 7) * 7 * (-1 / 3)
def expr_C := (-3) * (-1 / 2) * 7 * 0
def expr_D := (-1 / 5) * 6 * (-2 / 3) * (-5) * (-1 / 2)

theorem neg_product_B :
  expr_B < 0 :=
by
  sorry

end neg_product_B_l50_50015


namespace evaluate_F_2_f_3_l50_50313

def f (a : ℕ) : ℕ := a^2 - 2*a
def F (a b : ℕ) : ℕ := b^2 + a*b

theorem evaluate_F_2_f_3 : F 2 (f 3) = 15 := by
  sorry

end evaluate_F_2_f_3_l50_50313


namespace number_of_cars_l50_50226

theorem number_of_cars 
  (num_bikes : ℕ) (num_wheels_total : ℕ) (wheels_per_bike : ℕ) (wheels_per_car : ℕ)
  (h1 : num_bikes = 10) (h2 : num_wheels_total = 76) (h3 : wheels_per_bike = 2) (h4 : wheels_per_car = 4) :
  ∃ (C : ℕ), C = 14 := 
by
  sorry

end number_of_cars_l50_50226


namespace grandmother_age_l50_50683

theorem grandmother_age (minyoung_age_current : ℕ)
                         (minyoung_age_future : ℕ)
                         (grandmother_age_future : ℕ)
                         (h1 : minyoung_age_future = minyoung_age_current + 3)
                         (h2 : grandmother_age_future = 65)
                         (h3 : minyoung_age_future = 10) : grandmother_age_future - (minyoung_age_future -minyoung_age_current) = 62 := by
  sorry

end grandmother_age_l50_50683


namespace landscape_length_l50_50741

-- Define the conditions from the problem
def breadth (b : ℝ) := b > 0
def length_of_landscape (l b : ℝ) := l = 8 * b
def area_of_playground (pg_area : ℝ) := pg_area = 1200
def playground_fraction (A b : ℝ) := A = 8 * b^2
def fraction_of_landscape (pg_area A : ℝ) := pg_area = (1/6) * A

-- Main theorem statement
theorem landscape_length (b l A pg_area : ℝ) 
  (H_b : breadth b) 
  (H_length : length_of_landscape l b)
  (H_pg_area : area_of_playground pg_area)
  (H_pg_fraction : playground_fraction A b)
  (H_pg_landscape_fraction : fraction_of_landscape pg_area A) :
  l = 240 :=
by
  sorry

end landscape_length_l50_50741


namespace gear_ratios_l50_50716

variable (x y z w : ℝ)
variable (ω_A ω_B ω_C ω_D : ℝ)
variable (k : ℝ)

theorem gear_ratios (h : x * ω_A = y * ω_B ∧ y * ω_B = z * ω_C ∧ z * ω_C = w * ω_D) : 
    ω_A/ω_B = yzw/xzw ∧ ω_B/ω_C = xzw/xyw ∧ ω_C/ω_D = xyw/xyz ∧ ω_A/ω_C = yzw/xyw := 
sorry

end gear_ratios_l50_50716


namespace John_meeting_percentage_l50_50876

def hours_to_minutes (h : ℕ) : ℕ := 60 * h

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 60
def third_meeting_duration : ℕ := 2 * first_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

def total_workday_duration : ℕ := hours_to_minutes 12

def percentage_of_meetings (total_meeting_time total_workday_time : ℕ) : ℕ := 
  (total_meeting_time * 100) / total_workday_time

theorem John_meeting_percentage : 
  percentage_of_meetings total_meeting_duration total_workday_duration = 21 :=
by
  sorry

end John_meeting_percentage_l50_50876


namespace inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l50_50670

-- Definitions for the conditions
def inside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 < r^2 ∧ (M.1 ≠ 0 ∨ M.2 ≠ 0)

def on_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 = r^2

def outside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 > r^2

def line_l_intersects_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 < r^2 ∨ M.1 * M.1 + M.2 * M.2 = r^2

def line_l_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 = r^2

def line_l_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 > r^2

-- Propositions
theorem inside_circle_implies_line_intersects_circle (M : ℝ × ℝ) (r : ℝ) : 
  inside_circle M r → line_l_intersects_circle M r := 
sorry

theorem on_circle_implies_line_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) :
  on_circle M r → line_l_tangent_to_circle M r :=
sorry

theorem outside_circle_implies_line_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) :
  outside_circle M r → line_l_does_not_intersect_circle M r :=
sorry

end inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l50_50670


namespace work_completion_l50_50008

theorem work_completion (W : ℕ) (a_rate b_rate combined_rate : ℕ) 
  (h1: combined_rate = W/8) 
  (h2: a_rate = W/12) 
  (h3: combined_rate = a_rate + b_rate) 
  : combined_rate = W/8 :=
by
  sorry

end work_completion_l50_50008


namespace possible_polynomials_l50_50808

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l50_50808


namespace logarithmic_expression_max_value_l50_50270

theorem logarithmic_expression_max_value (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a / b = 3) :
  3 * Real.log (a / b) / Real.log a + 2 * Real.log (b / a) / Real.log b = -4 := 
sorry

end logarithmic_expression_max_value_l50_50270


namespace percentage_of_men_in_company_l50_50066

theorem percentage_of_men_in_company 
  (M W : ℝ) 
  (h1 : 0.60 * M + 0.35 * W = 50) 
  (h2 : M + W = 100) : 
  M = 60 :=
by
  sorry

end percentage_of_men_in_company_l50_50066


namespace compute_large_expression_l50_50988

theorem compute_large_expression :
  ( (11^4 + 484) * (23^4 + 484) * (35^4 + 484) * (47^4 + 484) * (59^4 + 484) ) / 
  ( (5^4 + 484) * (17^4 + 484) * (29^4 + 484) * (41^4 + 484) * (53^4 + 484) ) = 552.42857 := 
sorry

end compute_large_expression_l50_50988


namespace string_cuts_l50_50874

theorem string_cuts (L S : ℕ) (h_diff : L - S = 48) (h_sum : L + S = 64) : 
  (L / S) = 7 :=
by
  sorry

end string_cuts_l50_50874


namespace speed_of_current_l50_50233

theorem speed_of_current (m c : ℝ) (h1 : m + c = 20) (h2 : m - c = 18) : c = 1 :=
by
  sorry

end speed_of_current_l50_50233


namespace quadratic_rewrite_constants_l50_50822

theorem quadratic_rewrite_constants (a b c : ℤ) 
    (h1 : -4 * (x - 2) ^ 2 + 144 = -4 * x ^ 2 + 16 * x + 128) 
    (h2 : a = -4)
    (h3 : b = -2)
    (h4 : c = 144) 
    : a + b + c = 138 := by
  sorry

end quadratic_rewrite_constants_l50_50822


namespace missy_total_watching_time_l50_50410

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l50_50410


namespace alberto_spent_2457_l50_50257

-- Define the expenses by Samara on each item
def oil_expense : ℕ := 25
def tires_expense : ℕ := 467
def detailing_expense : ℕ := 79

-- Define the additional amount Alberto spent more than Samara
def additional_amount : ℕ := 1886

-- Total amount spent by Samara
def samara_total_expense : ℕ := oil_expense + tires_expense + detailing_expense

-- The amount spent by Alberto
def alberto_expense := samara_total_expense + additional_amount

-- Theorem stating the amount spent by Alberto
theorem alberto_spent_2457 :
  alberto_expense = 2457 :=
by {
  -- Include the actual proof here if necessary
  sorry
}

end alberto_spent_2457_l50_50257


namespace average_molecular_weight_benzoic_acid_l50_50581

def atomic_mass_C : ℝ := (12 * 0.9893) + (13 * 0.0107)
def atomic_mass_H : ℝ := (1 * 0.99985) + (2 * 0.00015)
def atomic_mass_O : ℝ := (16 * 0.99762) + (17 * 0.00038) + (18 * 0.00200)

theorem average_molecular_weight_benzoic_acid :
  (7 * atomic_mass_C) + (6 * atomic_mass_H) + (2 * atomic_mass_O) = 123.05826 :=
by {
  sorry
}

end average_molecular_weight_benzoic_acid_l50_50581


namespace tennis_ball_ratio_problem_solution_l50_50596

def tennis_ball_ratio_problem (total_balls ordered_white ordered_yellow dispatched_yellow extra_yellow : ℕ) : Prop :=
  total_balls = 114 ∧ 
  ordered_white = total_balls / 2 ∧ 
  ordered_yellow = total_balls / 2 ∧ 
  dispatched_yellow = ordered_yellow + extra_yellow → 
  (ordered_white / dispatched_yellow = 57 / 107)

theorem tennis_ball_ratio_problem_solution :
  tennis_ball_ratio_problem 114 57 57 107 50 := by 
  sorry

end tennis_ball_ratio_problem_solution_l50_50596


namespace max_product_of_sum_2016_l50_50949

theorem max_product_of_sum_2016 (x y : ℤ) (h : x + y = 2016) : x * y ≤ 1016064 :=
by
  -- Proof goes here, but is not needed as per instructions
  sorry

end max_product_of_sum_2016_l50_50949


namespace second_smallest_packs_of_hot_dogs_l50_50230

theorem second_smallest_packs_of_hot_dogs
    (n : ℤ) 
    (h1 : ∃ m : ℤ, 12 * n = 8 * m + 6) :
    ∃ k : ℤ, n = 4 * k + 7 :=
sorry

end second_smallest_packs_of_hot_dogs_l50_50230


namespace simplest_common_denominator_fraction_exist_l50_50106

variable (x y : ℝ)

theorem simplest_common_denominator_fraction_exist :
  let d1 := x + y
  let d2 := x - y
  let d3 := x^2 - y^2
  (d3 = d1 * d2) → 
    ∀ n, (n = d1 * d2) → 
      (∃ m, (d1 * m = n) ∧ (d2 * m = n) ∧ (d3 * m = n)) :=
by
  sorry

end simplest_common_denominator_fraction_exist_l50_50106


namespace arithmetic_mean_six_expressions_l50_50960

theorem arithmetic_mean_six_expressions (x : ℝ) :
  (x + 10 + 17 + 2 * x + 15 + 2 * x + 6 + 3 * x - 5) / 6 = 30 →
  x = 137 / 8 :=
by
  sorry

end arithmetic_mean_six_expressions_l50_50960


namespace unique_sums_count_l50_50707

open Set

-- Defining the sets of chips in bags C and D
def BagC : Set ℕ := {1, 3, 7, 9}
def BagD : Set ℕ := {4, 6, 8}

-- The proof problem: show there are 7 unique sums
theorem unique_sums_count : (BagC ×ˢ BagD).image (λ p => p.1 + p.2) = {5, 7, 9, 11, 13, 15, 17} :=
by
  -- Proof omitted; complete proof would go here
  sorry

end unique_sums_count_l50_50707


namespace johnPaysPerYear_l50_50894

-- Define the conditions
def epiPenCost : ℝ := 500
def insuranceCoverage : ℝ := 0.75
def epiPenFrequencyPerYear : ℝ := 2 -- Twice a year since 12 months / 6 months per EpiPen

-- Calculate the cost after insurance
def costAfterInsurance (cost : ℝ) (coverage : ℝ) : ℝ :=
  cost * (1 - coverage)

-- Calculate the annual cost
def annualCost (freq : ℝ) (cost : ℝ) : ℝ :=
  freq * cost

-- The statement asserting the proof
theorem johnPaysPerYear (cost : ℝ) (coverage : ℝ) (freq : ℝ) : 
  epiPenCost = cost → 
  insuranceCoverage = coverage → 
  epiPenFrequencyPerYear = freq → 
  annualCost freq (costAfterInsurance cost coverage) = 250 := 
by 
  intros _ _ _ 
  sorry

end johnPaysPerYear_l50_50894


namespace total_fruits_picked_l50_50203

theorem total_fruits_picked (g_oranges g_apples a_oranges a_apples o_oranges o_apples : ℕ) :
  g_oranges = 45 →
  g_apples = a_apples + 5 →
  a_oranges = g_oranges - 18 →
  a_apples = 15 →
  o_oranges = 6 * 3 →
  o_apples = 6 * 2 →
  g_oranges + g_apples + a_oranges + a_apples + o_oranges + o_apples = 137 :=
by
  intros
  sorry

end total_fruits_picked_l50_50203


namespace greatest_brownies_produced_l50_50043

theorem greatest_brownies_produced (p side_length a b brownies : ℕ) :
  (4 * side_length = p) →
  (p = 40) →
  (brownies = side_length * side_length) →
  ((side_length - a - 2) * (side_length - b - 2) = 2 * (2 * (side_length - a) + 2 * (side_length - b) - 4)) →
  (a = 4) →
  (b = 4) →
  brownies = 100 :=
by
  intros h_perimeter h_perimeter_value h_brownies h_eq h_a h_b
  sorry

end greatest_brownies_produced_l50_50043


namespace num_pos_three_digit_div_by_seven_l50_50454

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l50_50454


namespace seventh_term_of_arithmetic_sequence_l50_50629

variable (a d : ℕ)

theorem seventh_term_of_arithmetic_sequence (h1 : 5 * a + 10 * d = 15) (h2 : a + 3 * d = 4) : a + 6 * d = 7 := 
by
  sorry

end seventh_term_of_arithmetic_sequence_l50_50629


namespace find_a_minus_b_l50_50852

def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 5
def h (a b x : ℝ) : ℝ := f a b (g x)
def h_inv (x : ℝ) : ℝ := x + 7

theorem find_a_minus_b (a b : ℝ) :
  (∀ x : ℝ, h a b x = -3 * a * x + 5 * a + b) ∧
  (∀ x : ℝ, h_inv (h a b x) = x) ∧
  (∀ x : ℝ, h a b x = x - 7) →
  a - b = 5 :=
by
  sorry

end find_a_minus_b_l50_50852


namespace part1_part2_l50_50991

theorem part1 (x : ℝ) : 3 + 2 * x > - x - 6 ↔ x > -3 := by
  sorry

theorem part2 (x : ℝ) : 2 * x + 1 ≤ x + 3 ∧ (2 * x + 1) / 3 > 1 ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end part1_part2_l50_50991


namespace no_n_exists_l50_50552

theorem no_n_exists (n : ℕ) : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
by {
  sorry
}

end no_n_exists_l50_50552


namespace quadratic_inequality_solution_l50_50448

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 - 6 * x - 16 > 0) ↔ (x < -2 ∨ x > 8) :=
sorry

end quadratic_inequality_solution_l50_50448


namespace quadrilateral_trapezoid_or_parallelogram_l50_50181

theorem quadrilateral_trapezoid_or_parallelogram
  (s1 s2 s3 s4 : ℝ)
  (hs : s1^2 = s2 * s4) :
  (exists (is_trapezoid : Prop), is_trapezoid) ∨ (exists (is_parallelogram : Prop), is_parallelogram) :=
by
  sorry

end quadrilateral_trapezoid_or_parallelogram_l50_50181


namespace dog_food_bags_count_l50_50674

-- Define the constants based on the problem statement
def CatFoodBags := 327
def DogFoodMore := 273

-- Define the total number of dog food bags based on the given conditions
def DogFoodBags : ℤ := CatFoodBags + DogFoodMore

-- State the theorem we want to prove
theorem dog_food_bags_count : DogFoodBags = 600 := by
  sorry

end dog_food_bags_count_l50_50674


namespace greatest_possible_difference_l50_50010

theorem greatest_possible_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  ∃ d, d = y - x ∧ d = 6 := 
by
  sorry

end greatest_possible_difference_l50_50010


namespace initial_ratio_of_milk_to_water_l50_50492

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + W = 60) (h2 : 2 * M = W + 60) : M / W = 2 :=
by
  sorry

end initial_ratio_of_milk_to_water_l50_50492


namespace ratio_of_AC_to_BD_l50_50595

theorem ratio_of_AC_to_BD (A B C D : ℝ) (AB BC AD AC BD : ℝ) 
  (h1 : AB = 2) (h2 : BC = 5) (h3 : AD = 14) (h4 : AC = AB + BC) (h5 : BD = AD - AB) :
  AC / BD = 7 / 12 := by
  sorry

end ratio_of_AC_to_BD_l50_50595


namespace problem_solution_l50_50569

theorem problem_solution (x : ℝ) :
    (x^2 / (x - 2) ≥ (3 / (x + 2)) + (7 / 5)) →
    (x ∈ Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi (2 : ℝ)) :=
by
  intro h
  sorry

end problem_solution_l50_50569


namespace min_expression_value_l50_50419

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end min_expression_value_l50_50419


namespace mean_of_solutions_l50_50413

theorem mean_of_solutions (x : ℝ) (h : x^3 + x^2 - 14 * x = 0) : 
  let a := (0 : ℝ)
  let b := (-1 + Real.sqrt 57) / 2
  let c := (-1 - Real.sqrt 57) / 2
  (a + b + c) / 3 = -2 / 3 :=
sorry

end mean_of_solutions_l50_50413


namespace age_problem_l50_50305

-- Defining the conditions and the proof problem
variables (B A : ℕ) -- B and A are natural numbers

-- Given conditions
def B_age : ℕ := 38
def A_age (B : ℕ) : ℕ := B + 8
def age_in_10_years (A : ℕ) : ℕ := A + 10
def years_ago (B : ℕ) (X : ℕ) : ℕ := B - X

-- Lean statement of the problem
theorem age_problem (X : ℕ) (hB : B = B_age) (hA : A = A_age B):
  age_in_10_years A = 2 * (years_ago B X) → X = 10 :=
by
  sorry

end age_problem_l50_50305


namespace log_base_5_domain_correct_l50_50817

def log_base_5_domain : Set ℝ := {x : ℝ | x > 0}

theorem log_base_5_domain_correct : (∀ x : ℝ, x > 0 ↔ x ∈ log_base_5_domain) :=
by sorry

end log_base_5_domain_correct_l50_50817


namespace right_triangle_isosceles_l50_50913

-- Define the conditions for a right-angled triangle inscribed in a circle
variables (a b : ℝ)

-- Conditions provided in the problem
def right_triangle_inscribed (a b : ℝ) : Prop :=
  ∃ h : a ≠ 0 ∧ b ≠ 0, 2 * (a^2 + b^2) = (a + 2*b)^2 + b^2 ∧ 2 * (a^2 + b^2) = (2 * a + b)^2 + a^2

-- The theorem to prove based on the conditions
theorem right_triangle_isosceles (a b : ℝ) (h : right_triangle_inscribed a b) : a = b :=
by 
  sorry

end right_triangle_isosceles_l50_50913


namespace Steve_bakes_more_apple_pies_l50_50477

def Steve_bakes (days_apple days_cherry pies_per_day : ℕ) : ℕ :=
  (days_apple * pies_per_day) - (days_cherry * pies_per_day)

theorem Steve_bakes_more_apple_pies :
  Steve_bakes 3 2 12 = 12 :=
by
  sorry

end Steve_bakes_more_apple_pies_l50_50477


namespace find_Y_payment_l50_50912

theorem find_Y_payment 
  (P X Z : ℝ)
  (total_payment : ℝ)
  (h1 : P + X + Z = total_payment)
  (h2 : X = 1.2 * P)
  (h3 : Z = 0.96 * P) :
  P = 332.28 := by
  sorry

end find_Y_payment_l50_50912


namespace movie_screening_guests_l50_50550

theorem movie_screening_guests
  (total_guests : ℕ)
  (women_percentage : ℝ)
  (men_count : ℕ)
  (men_left_fraction : ℝ)
  (children_left_percentage : ℝ)
  (children_count : ℕ)
  (people_left : ℕ) :
  total_guests = 75 →
  women_percentage = 0.40 →
  men_count = 25 →
  men_left_fraction = 1/3 →
  children_left_percentage = 0.20 →
  children_count = total_guests - (round (women_percentage * total_guests) + men_count) →
  people_left = (round (men_left_fraction * men_count)) + (round (children_left_percentage * children_count)) →
  (total_guests - people_left) = 63 :=
by
  intros ht hw hm hf hc hc_count hl
  sorry

end movie_screening_guests_l50_50550


namespace find_ages_l50_50820

-- Define that f is a polynomial with integer coefficients
noncomputable def f : ℤ → ℤ := sorry

-- Given conditions
axiom f_at_7 : f 7 = 77
axiom f_at_b : ∃ b : ℕ, f b = 85
axiom f_at_c : ∃ c : ℕ, f c = 0

-- Define what we need to prove
theorem find_ages : ∃ b c : ℕ, (b - 7 ∣ 8) ∧ (c - b ∣ 85) ∧ (c - 7 ∣ 77) ∧ (b = 9) ∧ (c = 14) :=
sorry

end find_ages_l50_50820


namespace find_n_from_equation_l50_50173

theorem find_n_from_equation (n m : ℕ) (h1 : (1^m / 5^m) * (1^n / 4^n) = 1 / (2 * 10^31)) (h2 : m = 31) : n = 16 := 
by
  sorry

end find_n_from_equation_l50_50173


namespace triangle_inequality_squared_l50_50585

theorem triangle_inequality_squared {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := sorry

end triangle_inequality_squared_l50_50585


namespace new_elephants_entry_rate_l50_50665

-- Definitions
def initial_elephants := 30000
def exodus_rate := 2880
def exodus_duration := 4
def final_elephants := 28980
def new_elephants_duration := 7

-- Prove that the rate of new elephants entering the park is 1500 elephants per hour
theorem new_elephants_entry_rate :
  let elephants_left_after_exodus := initial_elephants - exodus_rate * exodus_duration
  let new_elephants := final_elephants - elephants_left_after_exodus
  let new_entry_rate := new_elephants / new_elephants_duration
  new_entry_rate = 1500 :=
by
  sorry

end new_elephants_entry_rate_l50_50665


namespace overall_loss_amount_l50_50395

theorem overall_loss_amount 
    (S : ℝ)
    (hS : S = 12499.99)
    (profit_percent : ℝ)
    (loss_percent : ℝ)
    (sold_at_profit : ℝ)
    (sold_at_loss : ℝ) 
    (condition1 : profit_percent = 0.2)
    (condition2 : loss_percent = -0.1)
    (condition3 : sold_at_profit = 0.2 * S * (1 + profit_percent))
    (condition4 : sold_at_loss = 0.8 * S * (1 + loss_percent))
    :
    S - (sold_at_profit + sold_at_loss) = 500 := 
by 
  sorry

end overall_loss_amount_l50_50395


namespace unique_solution_for_lines_intersection_l50_50520

theorem unique_solution_for_lines_intersection (n : ℕ) (h : n * (n - 1) / 2 = 2) : n = 2 :=
by
  sorry

end unique_solution_for_lines_intersection_l50_50520


namespace solution_set_of_abs_inequality_l50_50117

theorem solution_set_of_abs_inequality (x : ℝ) : 
  (x < 5 ↔ |x - 8| - |x - 4| > 2) :=
sorry

end solution_set_of_abs_inequality_l50_50117


namespace marley_total_fruits_l50_50134

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits_l50_50134


namespace no_generating_combination_l50_50728

-- Representing Rubik's Cube state as a type (assume a type exists)
axiom CubeState : Type

-- A combination of turns represented as a function on states
axiom A : CubeState → CubeState

-- Simple rotations
axiom P : CubeState → CubeState
axiom Q : CubeState → CubeState

-- Rubik's Cube property of generating combination (assuming generating implies all states achievable)
def is_generating (A : CubeState → CubeState) :=
  ∀ X : CubeState, ∃ m n : ℕ, P X = A^[m] X ∧ Q X = A^[n] X

-- Non-commutativity condition
axiom non_commutativity : ∀ X : CubeState, P (Q X) ≠ Q (P X)

-- Formal statement of the problem
theorem no_generating_combination : ¬ ∃ A : CubeState → CubeState, is_generating A :=
by sorry

end no_generating_combination_l50_50728


namespace rectangular_field_area_eq_l50_50997

-- Definitions based on the problem's conditions
def length (x : ℝ) := x
def width (x : ℝ) := 60 - x
def area (x : ℝ) := x * (60 - x)

-- The proof statement
theorem rectangular_field_area_eq (x : ℝ) (h₀ : x + (60 - x) = 60) (h₁ : area x = 864) :
  x * (60 - x) = 864 :=
by
  -- Using the provided conditions and definitions, we aim to prove the equation.
  sorry

end rectangular_field_area_eq_l50_50997


namespace bushes_needed_for_60_zucchinis_l50_50503

-- Each blueberry bush yields 10 containers of blueberries.
def containers_per_bush : ℕ := 10

-- 6 containers of blueberries can be traded for 3 zucchinis.
def containers_to_zucchinis (containers zucchinis : ℕ) : Prop := containers = 6 ∧ zucchinis = 3

theorem bushes_needed_for_60_zucchinis (bushes containers zucchinis : ℕ) :
  containers_per_bush = 10 →
  containers_to_zucchinis 6 3 →
  zucchinis = 60 →
  bushes = 12 :=
by
  intros h1 h2 h3
  sorry

end bushes_needed_for_60_zucchinis_l50_50503


namespace find_r_l50_50859

theorem find_r (a b m p r : ℝ) (h_roots1 : a * b = 6) 
  (h_eq1 : ∀ x, x^2 - m*x + 6 = 0) 
  (h_eq2 : ∀ x, x^2 - p*x + r = 0) :
  r = 32 / 3 :=
by
  sorry

end find_r_l50_50859


namespace ratio_of_money_with_Gopal_and_Krishan_l50_50796

theorem ratio_of_money_with_Gopal_and_Krishan 
  (R G K : ℕ) 
  (h1 : R = 735) 
  (h2 : K = 4335) 
  (h3 : R * 17 = G * 7) :
  G * 4335 = 1785 * K :=
by
  sorry

end ratio_of_money_with_Gopal_and_Krishan_l50_50796


namespace greatest_two_digit_product_12_l50_50947

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l50_50947


namespace fillets_per_fish_l50_50081

-- Definitions for the conditions
def fish_caught_per_day := 2
def days := 30
def total_fish_caught : Nat := fish_caught_per_day * days
def total_fish_fillets := 120

-- The proof problem statement
theorem fillets_per_fish (h1 : total_fish_caught = 60) (h2 : total_fish_fillets = 120) : 
  (total_fish_fillets / total_fish_caught) = 2 := sorry

end fillets_per_fish_l50_50081


namespace parabola_properties_l50_50518

-- Define the conditions
def vertex (f : ℝ → ℝ) (v : ℝ × ℝ) : Prop :=
  ∀ (x : ℝ), f (v.1) ≤ f x

def vertical_axis_of_symmetry (f : ℝ → ℝ) (h : ℝ) : Prop :=
  ∀ (x : ℝ), f x = f (2 * h - x)

def contains_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- Define f as the given parabola equation
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 16

-- The main statement to prove
theorem parabola_properties :
  vertex f (3, -2) ∧ vertical_axis_of_symmetry f 3 ∧ contains_point f (6, 16) := sorry

end parabola_properties_l50_50518


namespace son_l50_50891

theorem son's_age (S F : ℕ) (h1 : F = S + 26) (h2 : F + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_l50_50891


namespace geometric_sequence_value_l50_50726

theorem geometric_sequence_value (a : ℝ) (h₁ : 280 ≠ 0) (h₂ : 35 ≠ 0) : 
  (∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8 ∧ a > 0) → a = 35 :=
by {
  sorry
}

end geometric_sequence_value_l50_50726


namespace cows_count_24_l50_50339

-- Declare the conditions as given in the problem.
variables (D C : Nat)

-- Define the total number of legs and heads and the given condition.
def total_legs := 2 * D + 4 * C
def total_heads := D + C
axiom condition : total_legs = 2 * total_heads + 48

-- The goal is to prove that the number of cows C is 24.
theorem cows_count_24 : C = 24 :=
by
  sorry

end cows_count_24_l50_50339


namespace sum_last_two_digits_of_powers_l50_50931

theorem sum_last_two_digits_of_powers (h₁ : 9 = 10 - 1) (h₂ : 11 = 10 + 1) :
  (9^20 + 11^20) % 100 / 10 + (9^20 + 11^20) % 10 = 2 :=
by
  sorry

end sum_last_two_digits_of_powers_l50_50931


namespace find_g_function_l50_50321

noncomputable def g : ℝ → ℝ :=
  sorry

theorem find_g_function (x y : ℝ) (h1 : g 1 = 2) (h2 : ∀ (x y : ℝ), g (x + y) = 5^y * g x + 3^x * g y) :
  g x = 5^x - 3^x :=
by
  sorry

end find_g_function_l50_50321


namespace maple_tree_taller_than_pine_tree_l50_50887

def improper_fraction (a b : ℕ) : ℚ := a + (b : ℚ) / 4
def mixed_number_to_improper_fraction (n m : ℕ) : ℚ := improper_fraction n m

def pine_tree_height : ℚ := mixed_number_to_improper_fraction 12 1
def maple_tree_height : ℚ := mixed_number_to_improper_fraction 18 3

theorem maple_tree_taller_than_pine_tree :
  maple_tree_height - pine_tree_height = 6 + 1 / 2 :=
by sorry

end maple_tree_taller_than_pine_tree_l50_50887


namespace nancy_total_money_l50_50904

def total_money (n_five n_ten n_one : ℕ) : ℕ :=
  (n_five * 5) + (n_ten * 10) + (n_one * 1)

theorem nancy_total_money :
  total_money 9 4 7 = 92 :=
by
  sorry

end nancy_total_money_l50_50904


namespace range_m_l50_50535

open Set

noncomputable def A : Set ℝ := { x : ℝ | -5 ≤ x ∧ x ≤ 3 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m + 3 }

theorem range_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ m ≤ 0 :=
by
  sorry

end range_m_l50_50535


namespace point_in_plane_region_l50_50771

-- Defining the condition that the inequality represents a region on the plane
def plane_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

-- Stating that the point (0, 1) lies within the plane region represented by the inequality
theorem point_in_plane_region : plane_region 0 1 :=
by {
    sorry
}

end point_in_plane_region_l50_50771


namespace zacharys_bus_ride_length_l50_50732

theorem zacharys_bus_ride_length (Vince Zachary : ℝ) (hV : Vince = 0.62) (hDiff : Vince = Zachary + 0.13) : Zachary = 0.49 :=
by
  sorry

end zacharys_bus_ride_length_l50_50732


namespace sequence_properties_l50_50388

theorem sequence_properties :
  ∀ {a : ℕ → ℝ} {b : ℕ → ℝ},
  a 1 = 1 ∧ 
  (∀ n, b n > 4 / 3) ∧ 
  (∀ n, (∀ x, x^2 - b n * x + a n = 0 → (x = a (n + 1) ∨ x = 1 + a n))) →
  (a 2 = 1 / 2 ∧ ∃ n, b n > 4 / 3 ∧ n = 5) := by
  sorry

end sequence_properties_l50_50388


namespace intersection_A_B_union_A_B_diff_A_B_diff_B_A_l50_50473

def A : Set Real := {x | -1 < x ∧ x < 2}
def B : Set Real := {x | 0 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x < 2} :=
sorry

theorem union_A_B :
  A ∪ B = {x | -1 < x ∧ x < 4} :=
sorry

theorem diff_A_B :
  A \ B = {x | -1 < x ∧ x ≤ 0} :=
sorry

theorem diff_B_A :
  B \ A = {x | 2 ≤ x ∧ x < 4} :=
sorry

end intersection_A_B_union_A_B_diff_A_B_diff_B_A_l50_50473


namespace power_function_increasing_l50_50668

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 := 
by 
  sorry

end power_function_increasing_l50_50668


namespace sqrt_inequality_sum_inverse_ge_9_l50_50178

-- (1) Prove that \(\sqrt{3} + \sqrt{8} < 2 + \sqrt{7}\)
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := sorry

-- (2) Prove that given \(a > 0, b > 0, c > 0\) and \(a + b + c = 1\), \(\frac{1}{a} + \frac{1}{b} + \frac{1}{c} \geq 9\)
theorem sum_inverse_ge_9 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) : 
    1 / a + 1 / b + 1 / c ≥ 9 := sorry

end sqrt_inequality_sum_inverse_ge_9_l50_50178


namespace custom_op_evaluation_l50_50512

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : (custom_op 9 6) - (custom_op 6 9) = -12 := by
  sorry

end custom_op_evaluation_l50_50512


namespace maximum_F_value_l50_50031

open Real

noncomputable def F (a b c x : ℝ) := abs ((a * x^2 + b * x + c) * (c * x^2 + b * x + a))

theorem maximum_F_value (a b c : ℝ) (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1)
    (hfx : abs (a * x^2 + b * x + c) ≤ 1) :
    ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ F a b c x = 2 := 
  sorry

end maximum_F_value_l50_50031


namespace tangent_line_b_value_l50_50453

noncomputable def b_value : ℝ := Real.log 2 - 1

theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x > 0, (fun x => Real.log x) x = (1/2) * x + b → ∃ c : ℝ, c = b) → b = Real.log 2 - 1 :=
by
  sorry

end tangent_line_b_value_l50_50453


namespace range_of_a_for_inequality_l50_50608

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end range_of_a_for_inequality_l50_50608


namespace find_abc_l50_50319

theorem find_abc (a b c : ℕ) (h1 : c = b^2) (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 3 := 
by
  sorry

end find_abc_l50_50319


namespace ratio_area_perimeter_eq_sqrt3_l50_50745

theorem ratio_area_perimeter_eq_sqrt3 :
  let side_length := 12
  let altitude := side_length * (Real.sqrt 3) / 2
  let area := (1 / 2) * side_length * altitude
  let perimeter := 3 * side_length
  let ratio := area / perimeter
  ratio = Real.sqrt 3 := 
by
  sorry

end ratio_area_perimeter_eq_sqrt3_l50_50745


namespace henry_age_l50_50124

theorem henry_age (H J : ℕ) 
  (sum_ages : H + J = 40) 
  (age_relation : H - 11 = 2 * (J - 11)) : 
  H = 23 := 
sorry

end henry_age_l50_50124


namespace min_value_2a_3b_6c_l50_50815

theorem min_value_2a_3b_6c (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (habc : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 :=
sorry

end min_value_2a_3b_6c_l50_50815


namespace grid_path_theorem_l50_50412

open Nat

variables (m n : ℕ)
variables (A B C : ℕ)

def conditions (m n : ℕ) : Prop := m ≥ 4 ∧ n ≥ 4

noncomputable def grid_path_problem (m n A B C : ℕ) : Prop :=
  conditions m n ∧
  ((m - 1) * (n - 1) = A + (B + C)) ∧
  A = B - C + m + n - 1

theorem grid_path_theorem (m n A B C : ℕ) (h : grid_path_problem m n A B C) : 
  A = B - C + m + n - 1 :=
  sorry

end grid_path_theorem_l50_50412


namespace circle_line_chord_length_l50_50803

theorem circle_line_chord_length :
  ∀ (k m : ℝ), (∀ x y : ℝ, x^2 + y^2 = 4 → y = k * x + m → ∃ (a : ℝ), a = 2) →
    |m| = Real.sqrt 3 :=
by 
  intros k m h
  sorry

end circle_line_chord_length_l50_50803


namespace area_of_sector_l50_50037

/-- The area of a sector of a circle with radius 10 meters and central angle 42 degrees is 35/3 * pi square meters. -/
theorem area_of_sector (r θ : ℕ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360 : ℝ) * (Real.pi : ℝ) * (r : ℝ)^2 = (35 / 3 : ℝ) * (Real.pi : ℝ) :=
by {
  sorry
}

end area_of_sector_l50_50037


namespace height_of_platform_l50_50379

variables (l w h : ℕ)

theorem height_of_platform (hl1 : l + h - 2 * w = 36) (hl2 : w + h - l = 30) (hl3 : h = 2 * w) : h = 44 := 
sorry

end height_of_platform_l50_50379


namespace dot_product_min_value_in_triangle_l50_50575

noncomputable def dot_product_min_value (a b c : ℝ) (angleA : ℝ) : ℝ :=
  b * c * Real.cos angleA

theorem dot_product_min_value_in_triangle (b c : ℝ) (hyp1 : 0 ≤ b) (hyp2 : 0 ≤ c) 
  (hyp3 : b^2 + c^2 + b * c = 16) (hyp4 : Real.cos (2 * Real.pi / 3) = -1 / 2) : 
  ∃ (p : ℝ), p = dot_product_min_value 4 b c (2 * Real.pi / 3) ∧ p = -8 / 3 :=
by
  sorry

end dot_product_min_value_in_triangle_l50_50575


namespace base_n_representation_of_b_l50_50686

theorem base_n_representation_of_b (n a b : ℕ) (hn : n > 8) 
  (h_n_solution : ∃ m, m ≠ n ∧ n * m = b ∧ n + m = a) 
  (h_a_base_n : 1 * n + 8 = a) :
  (b = 8 * n) :=
by
  sorry

end base_n_representation_of_b_l50_50686


namespace delores_money_left_l50_50633

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l50_50633


namespace ball_bounce_height_l50_50795

theorem ball_bounce_height :
  ∃ k : ℕ, 800 * (1 / 2 : ℝ)^k < 2 ∧ k ≥ 9 :=
by
  sorry

end ball_bounce_height_l50_50795


namespace period_of_3sin_minus_4cos_l50_50484

theorem period_of_3sin_minus_4cos (x : ℝ) : 
  ∃ T : ℝ, T = 2 * Real.pi ∧ (∀ x, 3 * Real.sin x - 4 * Real.cos x = 3 * Real.sin (x + T) - 4 * Real.cos (x + T)) :=
sorry

end period_of_3sin_minus_4cos_l50_50484


namespace f_value_at_3_l50_50054

theorem f_value_at_3 (a b : ℝ) (h : (a * (-3)^3 - b * (-3) + 2 = -1)) : a * (3)^3 - b * 3 + 2 = 5 :=
sorry

end f_value_at_3_l50_50054


namespace outdoor_chairs_count_l50_50824

theorem outdoor_chairs_count (indoor_tables outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) 
  (total_chairs : ℕ) (h1: indoor_tables = 9) (h2: outdoor_tables = 11) 
  (h3: chairs_per_indoor_table = 10) (h4: total_chairs = 123) : 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 :=
by 
  sorry

end outdoor_chairs_count_l50_50824


namespace strawberries_eaten_l50_50004

-- Definitions based on the conditions
def strawberries_picked : ℕ := 35
def strawberries_remaining : ℕ := 33

-- Statement of the proof problem
theorem strawberries_eaten :
  strawberries_picked - strawberries_remaining = 2 :=
by
  sorry

end strawberries_eaten_l50_50004


namespace solve_for_n_l50_50028

theorem solve_for_n (n : ℝ) (h : 1 / (2 * n) + 1 / (4 * n) = 3 / 12) : n = 3 :=
sorry

end solve_for_n_l50_50028


namespace total_practice_hours_l50_50491

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l50_50491


namespace smallest_n_for_sum_condition_l50_50844

theorem smallest_n_for_sum_condition :
  ∃ n, n ≥ 4 ∧ (∀ S : Finset ℤ, S.card = n → ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (a + b - c - d) % 20 = 0) ∧ n = 9 :=
by
  sorry

end smallest_n_for_sum_condition_l50_50844


namespace sale_in_first_month_is_5000_l50_50130

def sales : List ℕ := [6524, 5689, 7230, 6000, 12557]
def avg_sales : ℕ := 7000
def total_months : ℕ := 6

theorem sale_in_first_month_is_5000 :
  (avg_sales * total_months) - sales.sum = 5000 :=
by sorry

end sale_in_first_month_is_5000_l50_50130


namespace min_value_x_plus_2_div_x_l50_50880

theorem min_value_x_plus_2_div_x (x : ℝ) (hx : x > 0) : x + 2 / x ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2_div_x_l50_50880


namespace speech_competition_score_l50_50893

theorem speech_competition_score :
  let speech_content := 90
  let speech_skills := 80
  let speech_effects := 85
  let content_ratio := 4
  let skills_ratio := 2
  let effects_ratio := 4
  (speech_content * content_ratio + speech_skills * skills_ratio + speech_effects * effects_ratio) / (content_ratio + skills_ratio + effects_ratio) = 86 := by
  sorry

end speech_competition_score_l50_50893


namespace number_of_correct_propositions_l50_50590

variable (Ω : Type) (R : Type) [Nonempty Ω] [Nonempty R]

-- Definitions of the conditions
def carsPassingIntersection (t : ℝ) : Ω → ℕ := sorry
def passengersInWaitingRoom (t : ℝ) : Ω → ℕ := sorry
def maximumFlowRiverEachYear : Ω → ℝ := sorry
def peopleExitingTheater (t : ℝ) : Ω → ℕ := sorry

-- Statement to prove the number of correct propositions
theorem number_of_correct_propositions : 4 = 4 := sorry

end number_of_correct_propositions_l50_50590


namespace storks_initially_l50_50006

-- Definitions for conditions
variable (S : ℕ) -- initial number of storks
variable (B : ℕ) -- initial number of birds

theorem storks_initially (h1 : B = 2) (h2 : S = B + 3 + 1) : S = 6 := by
  -- proof goes here
  sorry

end storks_initially_l50_50006


namespace no_solution_for_A_to_make_47A8_div_by_5_l50_50366

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem no_solution_for_A_to_make_47A8_div_by_5 (A : ℕ) :
  ¬ (divisible_by_5 (47 * 1000 + A * 100 + 8)) :=
by
  sorry

end no_solution_for_A_to_make_47A8_div_by_5_l50_50366


namespace sin_div_one_minus_tan_eq_neg_three_fourths_l50_50011

variable (α : ℝ)

theorem sin_div_one_minus_tan_eq_neg_three_fourths
  (h : Real.sin (α - Real.pi / 4) = Real.sqrt 2 / 4) :
  (Real.sin α) / (1 - Real.tan α) = -3 / 4 := sorry

end sin_div_one_minus_tan_eq_neg_three_fourths_l50_50011


namespace number_of_games_l50_50924

theorem number_of_games (total_points points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) : total_points / points_per_game = 3 := by
  sorry

end number_of_games_l50_50924


namespace work_rate_calculate_l50_50841

theorem work_rate_calculate (A_time B_time C_time total_time: ℕ) 
  (hA : A_time = 4) 
  (hB : B_time = 8)
  (hTotal : total_time = 2) : 
  C_time = 8 :=
by
  sorry

end work_rate_calculate_l50_50841


namespace four_people_fill_pool_together_in_12_minutes_l50_50632

def combined_pool_time (j s t e : ℕ) : ℕ := 
  1 / ((1 / j) + (1 / s) + (1 / t) + (1 / e))

theorem four_people_fill_pool_together_in_12_minutes : 
  ∀ (j s t e : ℕ), j = 30 → s = 45 → t = 90 → e = 60 → combined_pool_time j s t e = 12 := 
by 
  intros j s t e h_j h_s h_t h_e
  unfold combined_pool_time
  rw [h_j, h_s, h_t, h_e]
  have r1 : 1 / 30 = 1 / 30 := rfl
  have r2 : 1 / 45 = 1 / 45 := rfl
  have r3 : 1 / 90 = 1 / 90 := rfl
  have r4 : 1 / 60 = 1 / 60 := rfl
  rw [r1, r2, r3, r4]
  norm_num
  sorry

end four_people_fill_pool_together_in_12_minutes_l50_50632


namespace fraction_inequality_solution_l50_50237

open Set

theorem fraction_inequality_solution :
  {x : ℝ | 7 * x - 3 ≥ x^2 - x - 12 ∧ x ≠ 3 ∧ x ≠ -4} = Icc (-1 : ℝ) 3 ∪ Ioo (3 : ℝ) 4 ∪ Icc 4 9 :=
by
  sorry

end fraction_inequality_solution_l50_50237


namespace sixth_term_of_arithmetic_sequence_l50_50514

noncomputable def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ :=
  n * a + (n * (n - 1) / 2) * d

theorem sixth_term_of_arithmetic_sequence
  (a d : ℕ)
  (h₁ : sum_first_n_terms a d 4 = 10)
  (h₂ : a + 4 * d = 5) :
  a + 5 * d = 6 :=
by {
  sorry
}

end sixth_term_of_arithmetic_sequence_l50_50514


namespace least_positive_whole_number_divisible_by_five_primes_l50_50962

theorem least_positive_whole_number_divisible_by_five_primes :
  ∃ n : ℕ, (n = 2 * 3 * 5 * 7 * 11) ∧
           ∀ p : ℕ, p ∣ n → p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11 :=
by
  sorry

end least_positive_whole_number_divisible_by_five_primes_l50_50962


namespace Winnie_the_Pooh_stationary_escalator_steps_l50_50415

theorem Winnie_the_Pooh_stationary_escalator_steps
  (u v L : ℝ)
  (cond1 : L * u / (u + v) = 55)
  (cond2 : L * u / (u - v) = 1155) :
  L = 105 := by
  sorry

end Winnie_the_Pooh_stationary_escalator_steps_l50_50415


namespace question_1_question_2_l50_50357

def f (x a : ℝ) := |x - a|

theorem question_1 :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

theorem question_2 (a : ℝ) (h : a = 2) :
  (∀ x, f x a + f (x + 5) a ≥ m) → m ≤ 5 :=
by
  sorry

end question_1_question_2_l50_50357


namespace simplify_fraction_l50_50848

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l50_50848


namespace privateer_overtakes_at_6_08_pm_l50_50560

noncomputable def time_of_overtake : Bool :=
  let initial_distance := 12 -- miles
  let initial_time := 10 -- 10:00 a.m.
  let privateer_speed_initial := 10 -- mph
  let merchantman_speed := 7 -- mph
  let time_to_sail_initial := 3 -- hours
  let distance_covered_privateer := privateer_speed_initial * time_to_sail_initial
  let distance_covered_merchantman := merchantman_speed * time_to_sail_initial
  let relative_distance_after_three_hours := initial_distance + distance_covered_merchantman - distance_covered_privateer
  let privateer_speed_modified := 13 -- new speed
  let merchantman_speed_modified := 12 -- corresponding merchantman speed

  -- Calculating the new relative speed after the privateer's speed is reduced
  let privateer_new_speed := (13 / 12) * merchantman_speed
  let relative_speed_after_damage := privateer_new_speed - merchantman_speed
  let time_to_overtake_remainder := relative_distance_after_three_hours / relative_speed_after_damage
  let total_time := time_to_sail_initial + time_to_overtake_remainder -- in hours

  let final_time := initial_time + total_time -- converting into the final time of the day
  final_time == 18.1333 -- This should convert to 6:08 p.m., approximately 18 hours and 8 minutes in a 24-hour format

theorem privateer_overtakes_at_6_08_pm : time_of_overtake = true :=
  by
    -- Proof will be provided here
    sorry

end privateer_overtakes_at_6_08_pm_l50_50560


namespace find_number_l50_50331

noncomputable def some_number : ℝ :=
  0.27712 / 9.237333333333334

theorem find_number :
  (69.28 * 0.004) / some_number = 9.237333333333334 :=
by 
  sorry

end find_number_l50_50331


namespace subset_implies_a_geq_4_l50_50445

open Set

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + 3 ≤ 0}

theorem subset_implies_a_geq_4 (a : ℝ) :
  A ⊆ B a → a ≥ 4 := sorry

end subset_implies_a_geq_4_l50_50445


namespace transport_cost_correct_l50_50085

-- Defining the weights of the sensor unit and communication module in grams
def weight_sensor_grams : ℕ := 500
def weight_comm_module_grams : ℕ := 1500

-- Defining the transport cost per kilogram
def cost_per_kg_sensor : ℕ := 25000
def cost_per_kg_comm_module : ℕ := 20000

-- Converting weights to kilograms
def weight_sensor_kg : ℚ := weight_sensor_grams / 1000
def weight_comm_module_kg : ℚ := weight_comm_module_grams / 1000

-- Calculating the transport costs
def cost_sensor : ℚ := weight_sensor_kg * cost_per_kg_sensor
def cost_comm_module : ℚ := weight_comm_module_kg * cost_per_kg_comm_module

-- Total cost of transporting both units
def total_cost : ℚ := cost_sensor + cost_comm_module

-- Proving that the total cost is $42500
theorem transport_cost_correct : total_cost = 42500 := by
  sorry

end transport_cost_correct_l50_50085


namespace both_shots_hit_target_exactly_one_shot_hits_target_l50_50294

variable (p q : Prop)

theorem both_shots_hit_target : (p ∧ q) := sorry

theorem exactly_one_shot_hits_target : ((p ∧ ¬ q) ∨ (¬ p ∧ q)) := sorry

end both_shots_hit_target_exactly_one_shot_hits_target_l50_50294


namespace evaluate_expression_l50_50942

theorem evaluate_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (y : ℝ) (h3 : y = 1 / x + z) : 
    (x - 1 / x) * (y + 1 / y) = (x^2 - 1) * (1 + 2 * x * z + x^2 * z^2 + x^2) / (x^2 * (1 + x * z)) := by
  sorry

end evaluate_expression_l50_50942


namespace cards_eaten_by_hippopotamus_l50_50086

-- Defining the initial and remaining number of cards
def initial_cards : ℕ := 72
def remaining_cards : ℕ := 11

-- Statement of the proof problem
theorem cards_eaten_by_hippopotamus (initial_cards remaining_cards : ℕ) : initial_cards - remaining_cards = 61 :=
by
  sorry

end cards_eaten_by_hippopotamus_l50_50086


namespace amount_borrowed_from_bank_l50_50493

-- Definitions of the conditions
def car_price : ℝ := 35000
def total_payment : ℝ := 38000
def interest_rate : ℝ := 0.15

theorem amount_borrowed_from_bank :
  total_payment - car_price = interest_rate * (total_payment - car_price) / interest_rate := sorry

end amount_borrowed_from_bank_l50_50493


namespace b_plus_c_is_square_l50_50074

-- Given the conditions:
variables (a b c : ℕ)
variable (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Condition 1: Positive integers
variable (h2 : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)  -- Condition 2: Pairwise relatively prime
variable (h3 : a % 2 = 1 ∧ c % 2 = 1)  -- Condition 3: a and c are odd
variable (h4 : a^2 + b^2 = c^2)  -- Condition 4: Pythagorean triple equation

-- Prove that b + c is the square of an integer
theorem b_plus_c_is_square : ∃ k : ℕ, b + c = k^2 :=
by
  sorry

end b_plus_c_is_square_l50_50074


namespace distance_between_sasha_and_kolya_when_sasha_finished_l50_50626

-- Definitions based on the problem conditions
def distance_sasha : ℝ := 100
def distance_lesha_when_sasha_finished : ℝ := 90
def distance_kolya_when_lesha_finished : ℝ := 90

def velocity_lesha (v_s : ℝ) : ℝ := 0.9 * v_s
def velocity_kolya (v_s : ℝ) : ℝ := 0.81 * v_s

-- Theorem statement
theorem distance_between_sasha_and_kolya_when_sasha_finished (v_s : ℝ) :
  distance_sasha - (velocity_kolya v_s * (distance_sasha / v_s)) = 19 :=
  by sorry

end distance_between_sasha_and_kolya_when_sasha_finished_l50_50626


namespace least_number_to_subtract_l50_50174

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h : n = 427398) (k : d = 13) (r_val : r = 2) : 
  ∃ x : ℕ, (n - x) % d = 0 ∧ r = x :=
by sorry

end least_number_to_subtract_l50_50174


namespace victor_decks_l50_50860

theorem victor_decks (V : ℕ) (cost_per_deck total_spent friend_decks : ℕ) 
  (h1 : cost_per_deck = 8)
  (h2 : total_spent = 64)
  (h3 : friend_decks = 2) 
  (h4 : 8 * V + 8 * friend_decks = total_spent) : 
  V = 6 :=
by sorry

end victor_decks_l50_50860


namespace rectangle_x_value_l50_50622

theorem rectangle_x_value (x : ℝ) (h : (4 * x) * (x + 7) = 2 * (4 * x) + 2 * (x + 7)) : x = 0.675 := 
sorry

end rectangle_x_value_l50_50622


namespace find_k_l50_50542

-- Define the number and compute the sum of its digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem find_k :
  ∃ k : ℕ, sum_of_digits (9 * (10^k - 1)) = 1111 ∧ k = 124 :=
sorry

end find_k_l50_50542


namespace relationship_t_s_l50_50899

variable {a b : ℝ}

theorem relationship_t_s (a b : ℝ) (t : ℝ) (s : ℝ) (ht : t = a + 2 * b) (hs : s = a + b^2 + 1) :
  t ≤ s := 
sorry

end relationship_t_s_l50_50899


namespace unique_function_property_l50_50524

theorem unique_function_property (f : ℕ → ℕ) (h : ∀ m n : ℕ, f m + f n ∣ m + n) :
  ∀ m : ℕ, f m = m :=
by
  sorry

end unique_function_property_l50_50524


namespace slope_of_line_l50_50102

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : ∀ m : ℝ, (y = m * x + 3) → m = -3/4 :=
by
  sorry

end slope_of_line_l50_50102


namespace algebra_statements_correct_l50_50835

theorem algebra_statements_correct (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (ac < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (ab > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
sorry

end algebra_statements_correct_l50_50835


namespace exists_integers_x_y_z_l50_50516

theorem exists_integers_x_y_z (n : ℕ) : 
  ∃ x y z : ℤ, (x^2 + y^2 + z^2 = 3^(2^n)) ∧ (Int.gcd x (Int.gcd y z) = 1) :=
sorry

end exists_integers_x_y_z_l50_50516


namespace numbers_identification_l50_50926

-- Definitions
def is_natural (n : ℤ) : Prop := n ≥ 0
def is_integer (n : ℤ) : Prop := True

-- Theorem
theorem numbers_identification :
  (is_natural 0 ∧ is_natural 2 ∧ is_natural 6 ∧ is_natural 7) ∧
  (is_integer (-15) ∧ is_integer (-3) ∧ is_integer 0 ∧ is_integer 4) :=
by
  sorry

end numbers_identification_l50_50926


namespace sphere_radius_l50_50698

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
sorry

end sphere_radius_l50_50698


namespace total_water_intake_l50_50241

def theo_weekday := 8
def mason_weekday := 7
def roxy_weekday := 9
def zara_weekday := 10
def lily_weekday := 6

def theo_weekend := 10
def mason_weekend := 8
def roxy_weekend := 11
def zara_weekend := 12
def lily_weekend := 7

def total_cups_in_week (weekday_cups weekend_cups : ℕ) : ℕ :=
  5 * weekday_cups + 2 * weekend_cups

theorem total_water_intake :
  total_cups_in_week theo_weekday theo_weekend +
  total_cups_in_week mason_weekday mason_weekend +
  total_cups_in_week roxy_weekday roxy_weekend +
  total_cups_in_week zara_weekday zara_weekend +
  total_cups_in_week lily_weekday lily_weekend = 296 :=
by sorry

end total_water_intake_l50_50241


namespace find_m_direct_proportion_l50_50253

theorem find_m_direct_proportion (m : ℝ) (h1 : m^2 - 3 = 1) (h2 : m ≠ 2) : m = -2 :=
by {
  -- here would be the proof, but it's omitted as per instructions
  sorry
}

end find_m_direct_proportion_l50_50253


namespace find_j_l50_50059

theorem find_j (n j : ℕ) (h_n_pos : n > 0) (h_j_pos : j > 0) (h_rem : n % j = 28) (h_div : n / j = 142 ∧ (↑n / ↑j : ℝ) = 142.07) : j = 400 :=
by {
  sorry
}

end find_j_l50_50059


namespace units_digit_of_product_is_eight_l50_50847

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l50_50847


namespace units_digit_of_first_four_composite_numbers_l50_50092

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l50_50092


namespace road_unrepaired_is_42_percent_statement_is_false_l50_50003

def road_length : ℝ := 1
def phase1_completion : ℝ := 0.40
def phase2_remaining_factor : ℝ := 0.30

def remaining_road (road : ℝ) (phase1 : ℝ) (phase2_factor : ℝ) : ℝ :=
  road - phase1 - (road - phase1) * phase2_factor

theorem road_unrepaired_is_42_percent (road_length : ℝ) (phase1_completion : ℝ) (phase2_remaining_factor : ℝ) :
  remaining_road road_length phase1_completion phase2_remaining_factor = 0.42 :=
by
  sorry

theorem statement_is_false : ¬(remaining_road road_length phase1_completion phase2_remaining_factor = 0.30) :=
by
  sorry

end road_unrepaired_is_42_percent_statement_is_false_l50_50003


namespace symmetric_point_l50_50881

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def plane_eq (M : Point3D) : Prop :=
  2 * M.x - 4 * M.y - 4 * M.z - 13 = 0

-- Given Point M
def M : Point3D := { x := 3, y := -3, z := -1 }

-- Symmetric Point M'
def M' : Point3D := { x := 2, y := -1, z := 1 }

theorem symmetric_point (H : plane_eq M) : plane_eq M' ∧ 
  (M'.x = 2 * (3 + 2 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.x) ∧ 
  (M'.y = 2 * (-3 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.y) ∧ 
  (M'.z = 2 * (-1 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.z) :=
sorry

end symmetric_point_l50_50881


namespace rectangle_area_l50_50481

-- Define the problem in Lean
theorem rectangle_area (l w : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end rectangle_area_l50_50481


namespace problem_statement_l50_50063

def y_and (y : ℤ) : ℤ := 9 - y
def and_y (y : ℤ) : ℤ := y - 9

theorem problem_statement : and_y (y_and 15) = -15 := 
by
  sorry

end problem_statement_l50_50063


namespace fair_hair_women_percentage_l50_50046

-- Definitions based on conditions
def total_employees (E : ℝ) := E
def women_with_fair_hair (E : ℝ) := 0.28 * E
def fair_hair_employees (E : ℝ) := 0.70 * E

-- Theorem to prove
theorem fair_hair_women_percentage (E : ℝ) (hE : E > 0) :
  (women_with_fair_hair E) / (fair_hair_employees E) * 100 = 40 :=
by 
  -- Sorry denotes the proof is omitted
  sorry

end fair_hair_women_percentage_l50_50046


namespace min_value_of_f_l50_50511

noncomputable def f (x : ℝ) : ℝ := 4 * x + 2 / x

theorem min_value_of_f (x : ℝ) (hx : x > 0) : ∃ y : ℝ, (∀ z : ℝ, z > 0 → f z ≥ y) ∧ y = 4 * Real.sqrt 2 :=
sorry

end min_value_of_f_l50_50511


namespace trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l50_50172

theorem trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13 :
  (Real.cos (58 * Real.pi / 180) * Real.cos (13 * Real.pi / 180) +
   Real.sin (58 * Real.pi / 180) * Real.sin (13 * Real.pi / 180) =
   Real.cos (45 * Real.pi / 180)) :=
sorry

end trigonometric_identity_cos_58_cos_13_plus_sin_58_sin_13_l50_50172


namespace planned_pencils_is_49_l50_50869

def pencils_planned (x : ℕ) : ℕ := x
def pencils_bought (x : ℕ) : ℕ := x + 12
def total_pencils_bought (x : ℕ) : ℕ := 61

theorem planned_pencils_is_49 (x : ℕ) :
  pencils_bought (pencils_planned x) = total_pencils_bought x → x = 49 :=
sorry

end planned_pencils_is_49_l50_50869


namespace equation_solutions_l50_50610

theorem equation_solutions (m n x y : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  x^n + y^n = 3^m ↔ (x = 1 ∧ y = 2 ∧ n = 3 ∧ m = 2) ∨ (x = 2 ∧ y = 1 ∧ n = 3 ∧ m = 2) :=
by
  sorry -- proof to be implemented

end equation_solutions_l50_50610


namespace equations_have_different_graphs_l50_50070

theorem equations_have_different_graphs :
  (∃ (x : ℝ), ∀ (y₁ y₂ y₃ : ℝ),
    (y₁ = x - 2) ∧
    (y₂ = (x^2 - 4) / (x + 2) ∧ x ≠ -2) ∧
    (y₃ = (x^2 - 4) / (x + 2) ∧ x ≠ -2 ∨ (x = -2 ∧ ∀ y₃ : ℝ, (x+2) * y₃ = x^2 - 4)))
  → (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∨ y₁ ≠ y₃ ∨ y₂ ≠ y₃) := sorry

end equations_have_different_graphs_l50_50070


namespace problem1_extr_vals_l50_50862

-- Definitions from conditions
def f (x : ℝ) := x^3 - 2 * x^2 + x
def g (x : ℝ) := f x + 1

theorem problem1_extr_vals :
  ∃ a b : ℝ, a = g (1/3) ∧ b = g 1 ∧ a = 31/27 ∧ b = 1 :=
by
  sorry

end problem1_extr_vals_l50_50862


namespace border_material_correct_l50_50823

noncomputable def pi_approx := (22 : ℚ) / 7

def circle_radius (area : ℚ) (pi_value : ℚ) : ℚ :=
  (area * (7 / 22)).sqrt

def circumference (radius : ℚ) (pi_value : ℚ) : ℚ :=
  2 * pi_value * radius

def total_border_material (area : ℚ) (pi_value : ℚ) (extra : ℚ) : ℚ :=
  circumference (circle_radius area pi_value) pi_value + extra

theorem border_material_correct :
  total_border_material 616 pi_approx 3 = 91 :=
by
  sorry

end border_material_correct_l50_50823


namespace track_time_is_80_l50_50839

noncomputable def time_to_complete_track
  (a b : ℕ) 
  (meetings : a = 15 ∧ b = 25) : ℕ :=
a + b

theorem track_time_is_80 (a b : ℕ) (meetings : a = 15 ∧ b = 25) : time_to_complete_track a b meetings = 80 := by
  sorry

end track_time_is_80_l50_50839


namespace work_ratio_of_man_to_boy_l50_50110

theorem work_ratio_of_man_to_boy 
  (M B : ℝ) 
  (work : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = work)
  (h2 : (13 * M + 24 * B) * 4 = work) :
  M / B = 2 :=
by 
  sorry

end work_ratio_of_man_to_boy_l50_50110


namespace boyfriend_picks_up_correct_l50_50987

-- Define the initial condition
def init_pieces : ℕ := 60

-- Define the amount swept by Anne
def swept_pieces (n : ℕ) : ℕ := n / 2

-- Define the number of pieces stolen by the cat
def stolen_pieces : ℕ := 3

-- Define the remaining pieces after the cat steals
def remaining_pieces (n : ℕ) : ℕ := n - stolen_pieces

-- Define how many pieces the boyfriend picks up
def boyfriend_picks_up (n : ℕ) : ℕ := n / 3

-- The main theorem
theorem boyfriend_picks_up_correct : boyfriend_picks_up (remaining_pieces (init_pieces - swept_pieces init_pieces)) = 9 :=
by
  sorry

end boyfriend_picks_up_correct_l50_50987


namespace total_ladders_climbed_in_inches_l50_50133

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l50_50133


namespace union_A_B_intersection_complementA_B_range_of_a_l50_50790

-- Definition of the universal set U, sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Complement of A in the universal set U
def complement_A : Set ℝ := {x | x < 1 ∨ x ≥ 5}

-- Definition of set C parametrized by a
def C (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Prove that A ∪ B is {x | 1 ≤ x < 8}
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 8} :=
sorry

-- Prove that (complement_U A) ∩ B = {x | 5 ≤ x < 8}
theorem intersection_complementA_B : (complement_A ∩ B) = {x | 5 ≤ x ∧ x < 8} :=
sorry

-- Prove the range of values for a if C ∩ A = C
theorem range_of_a (a : ℝ) : (C a ∩ A = C a) → a ≤ -1 :=
sorry

end union_A_B_intersection_complementA_B_range_of_a_l50_50790


namespace range_of_a_l50_50676

variables (a : ℝ)

def prop_p : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 16 > 0
def prop_q : Prop := (2 * a - 2)^2 - 8 * (3 * a - 7) ≥ 0
def combined : Prop := prop_p a ∧ prop_q a

theorem range_of_a (a : ℝ) : combined a ↔ -4 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l50_50676


namespace sin_double_angle_l50_50789

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 3 / 4) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_l50_50789


namespace total_apartment_units_l50_50202

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end total_apartment_units_l50_50202


namespace abs_eq_necessary_but_not_sufficient_l50_50439

theorem abs_eq_necessary_but_not_sufficient (x y : ℝ) :
  (|x| = |y|) → (¬(x = y) → x = -y) :=
by
  sorry

end abs_eq_necessary_but_not_sufficient_l50_50439


namespace simple_interest_two_years_l50_50650
-- Import the necessary Lean library for mathematical concepts

-- Define the problem conditions and the proof statement
theorem simple_interest_two_years (P r t : ℝ) (CI SI : ℝ)
  (hP : P = 17000) (ht : t = 2) (hCI : CI = 11730) : SI = 5100 :=
by
  -- Principal (P), Rate (r), and Time (t) definitions
  let P := 17000
  let t := 2

  -- Given Compound Interest (CI)
  let CI := 11730

  -- Correct value for Simple Interest (SI) that we need to prove
  let SI := 5100

  -- Formalize the assumptions
  have h1 : P = 17000 := rfl
  have h2 : t = 2 := rfl
  have h3 : CI = 11730 := rfl

  -- Crucial parts of the problem are used here
  sorry  -- This is a placeholder for the actual proof steps

end simple_interest_two_years_l50_50650


namespace tablet_battery_life_l50_50256

theorem tablet_battery_life :
  ∀ (active_usage_hours idle_usage_hours : ℕ),
  active_usage_hours + idle_usage_hours = 12 →
  active_usage_hours = 3 →
  ((active_usage_hours / 2) + (idle_usage_hours / 10)) > 1 →
  idle_usage_hours = 9 →
  0 = 0 := 
by
  intros active_usage_hours idle_usage_hours h1 h2 h3 h4
  sorry

end tablet_battery_life_l50_50256


namespace min_value_expr_l50_50757

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x + 2 * y = 5) :
  (1 / (x - 1) + 1 / (y - 1)) = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_expr_l50_50757


namespace largest_k_consecutive_sum_l50_50292

theorem largest_k_consecutive_sum (k n : ℕ) :
  (5^7 = (k * (2 * n + k + 1)) / 2) → 1 ≤ k → k * (2 * n + k + 1) = 2 * 5^7 → k = 250 :=
sorry

end largest_k_consecutive_sum_l50_50292


namespace max_value_of_M_l50_50152

theorem max_value_of_M (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)) ≤ 1 :=
sorry -- Proof placeholder

end max_value_of_M_l50_50152


namespace find_double_pieces_l50_50187

theorem find_double_pieces (x : ℕ) 
  (h1 : 100 + 2 * x + 150 + 660 = 1000) : x = 45 :=
by sorry

end find_double_pieces_l50_50187


namespace prime_square_remainder_l50_50657

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l50_50657


namespace find_point_N_l50_50508

-- Definition of symmetrical reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Given condition
def point_M : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem find_point_N : reflect_x point_M = (1, -3) :=
by
  sorry

end find_point_N_l50_50508


namespace solve_equation_l50_50964

theorem solve_equation (x : ℚ) : 3 * (x - 2) = 2 - 5 * (x - 2) ↔ x = 9 / 4 := by
  sorry

end solve_equation_l50_50964


namespace distance_between_x_intercepts_l50_50024

theorem distance_between_x_intercepts (x1 y1 : ℝ) 
  (m1 m2 : ℝ)
  (hx1 : x1 = 10) (hy1 : y1 = 15)
  (hm1 : m1 = 3) (hm2 : m2 = 5) :
  let x_intercept1 := (y1 - m1 * x1) / -m1
  let x_intercept2 := (y1 - m2 * x1) / -m2
  dist (x_intercept1, 0) (x_intercept2, 0) = 2 :=
by
  sorry

end distance_between_x_intercepts_l50_50024


namespace sum_of_pqrstu_l50_50099

theorem sum_of_pqrstu (p q r s t : ℤ) (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72) 
  (hpqrs : p ≠ q) (hnpr : p ≠ r) (hnps : p ≠ s) (hnpt : p ≠ t) (hnqr : q ≠ r) 
  (hnqs : q ≠ s) (hnqt : q ≠ t) (hnrs : r ≠ s) (hnrt : r ≠ t) (hnst : s ≠ t) : 
  p + q + r + s + t = 25 := 
by
  sorry

end sum_of_pqrstu_l50_50099


namespace find_a_l50_50630

theorem find_a (a : ℝ) (h_pos : 0 < a) 
  (prob : (2 / a) = (1 / 3)) : a = 6 :=
by sorry

end find_a_l50_50630


namespace value_of_expression_l50_50100

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2006 = 2007 :=
sorry

end value_of_expression_l50_50100


namespace problem_1_problem_2_problem_3_l50_50865

-- The sequence S_n and its given condition
def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 2 * n

-- Definitions for a_1, a_2, and a_3 based on S_n conditions
theorem problem_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  a 1 = 2 ∧ a 2 = 6 ∧ a 3 = 14 :=
sorry

-- Definition of sequence b_n and its property of being geometric
def b (n : ℕ) (a : ℕ → ℕ) : ℕ := a n + 2

theorem problem_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n ≥ 1, b n a = 2 * b (n - 1) a :=
sorry

-- The sum of the first n terms of the sequence {na_n}, denoted by T_n
def T (n : ℕ) (a : ℕ → ℕ) : ℕ := (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1)

theorem problem_3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, S n = 2 * a n - 2 * n) :
  ∀ n, T n a = (n + 1) * 2 ^ (n + 2) + 4 - n * (n + 1) :=
sorry

end problem_1_problem_2_problem_3_l50_50865


namespace machine_A_produces_40_percent_l50_50591

theorem machine_A_produces_40_percent (p : ℝ) : 
  (0 < p ∧ p < 1 ∧
  (0.0156 = p * 0.009 + (1 - p) * 0.02)) → 
  p = 0.4 :=
by 
  intro h
  sorry

end machine_A_produces_40_percent_l50_50591


namespace sum_of_angles_is_540_l50_50871

variables (angle1 angle2 angle3 angle4 angle5 angle6 angle7 : ℝ)

theorem sum_of_angles_is_540
  (h : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540 :=
sorry

end sum_of_angles_is_540_l50_50871


namespace parallelogram_theorem_l50_50900

noncomputable def parallelogram (A B C D O : Type) (θ : ℝ) :=
  let DBA := θ
  let DBC := 3 * θ
  let CAB := 9 * θ
  let ACB := 180 - (9 * θ + 3 * θ)
  let AOB := 180 - 12 * θ
  let s := ACB / AOB
  s = 4 / 5

theorem parallelogram_theorem (A B C D O : Type) (θ : ℝ) 
  (h1: θ > 0): parallelogram A B C D O θ := by
  sorry

end parallelogram_theorem_l50_50900


namespace min_value_of_quadratic_l50_50588

theorem min_value_of_quadratic (x y s : ℝ) (h : x + y = s) : 
  ∃ x y, 3 * x^2 + 2 * y^2 = 6 * s^2 / 5 := sorry

end min_value_of_quadratic_l50_50588


namespace set_notation_nat_lt_3_l50_50423

theorem set_notation_nat_lt_3 : {x : ℕ | x < 3} = {0, 1, 2} := 
sorry

end set_notation_nat_lt_3_l50_50423


namespace negation_proposition_l50_50272

theorem negation_proposition (x : ℝ) (hx : 0 < x) : x + 4 / x ≥ 4 :=
sorry

end negation_proposition_l50_50272


namespace totalNumberOfBalls_l50_50925

def numberOfBoxes : ℕ := 3
def numberOfBallsPerBox : ℕ := 5

theorem totalNumberOfBalls : numberOfBoxes * numberOfBallsPerBox = 15 := 
by
  sorry

end totalNumberOfBalls_l50_50925


namespace derivative_of_y_l50_50504

noncomputable def y (x : ℝ) : ℝ :=
  (4 * x + 1) / (16 * x^2 + 8 * x + 3) + (1 / Real.sqrt 2) * Real.arctan ((4 * x + 1) / Real.sqrt 2)

theorem derivative_of_y (x : ℝ) : 
  (deriv y x) = 16 / (16 * x^2 + 8 * x + 3)^2 :=
by 
  sorry

end derivative_of_y_l50_50504


namespace octagon_diagonals_l50_50623

def num_sides := 8

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diagonals : num_diagonals num_sides = 20 :=
by
  sorry

end octagon_diagonals_l50_50623


namespace geometric_sequence_S8_l50_50204

theorem geometric_sequence_S8 (S : ℕ → ℝ) (hs2 : S 2 = 4) (hs4 : S 4 = 16) : 
  S 8 = 160 := by
  sorry

end geometric_sequence_S8_l50_50204


namespace erika_walked_distance_l50_50282

/-- Erika traveled to visit her cousin. She started on a scooter at an average speed of 
22 kilometers per hour. After completing three-fifths of the distance, the scooter's battery died, 
and she walked the rest of the way at 4 kilometers per hour. The total time it took her to reach her cousin's 
house was 2 hours. How far, in kilometers rounded to the nearest tenth, did Erika walk? -/
theorem erika_walked_distance (d : ℝ) (h1 : d > 0)
  (h2 : (3 / 5 * d) / 22 + (2 / 5 * d) / 4 = 2) : 
  (2 / 5 * d) = 6.3 :=
sorry

end erika_walked_distance_l50_50282


namespace dropping_more_than_eating_l50_50766

theorem dropping_more_than_eating (n : ℕ) : n = 20 → (n * (n + 1)) / 2 > 10 * n := by
  intros h
  rw [h]
  sorry

end dropping_more_than_eating_l50_50766


namespace greatest_difference_l50_50131

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l50_50131


namespace circle_radius_five_d_value_l50_50228

theorem circle_radius_five_d_value :
  ∀ (d : ℝ), (∃ (x y : ℝ), (x - 4)^2 + (y + 5)^2 = 41 - d) → d = 16 :=
by
  intros d h
  sorry

end circle_radius_five_d_value_l50_50228


namespace longer_side_of_new_rectangle_l50_50986

theorem longer_side_of_new_rectangle {z : ℕ} (h : ∃x : ℕ, 9 * 16 = 144 ∧ x * z = 144 ∧ z ≠ 9 ∧ z ≠ 16) : z = 18 :=
sorry

end longer_side_of_new_rectangle_l50_50986


namespace value_of_expr_l50_50361

theorem value_of_expr (a : Int) (h : a = -2) : a + 1 = -1 := by
  -- Placeholder for the proof, assuming it's correct
  sorry

end value_of_expr_l50_50361


namespace p_sufficient_not_necessary_for_q_l50_50639

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- State the theorem that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l50_50639


namespace indolent_student_probability_l50_50898

-- Define the constants of the problem
def n : ℕ := 30  -- total number of students
def k : ℕ := 3   -- number of students selected each lesson
def m : ℕ := 10  -- number of students from the previous lesson

-- Define the probabilities
def P_asked_in_one_lesson : ℚ := 1 / k
def P_asked_twice_in_a_row : ℚ := 1 / n
def P_overall : ℚ := P_asked_in_one_lesson + P_asked_in_one_lesson - P_asked_twice_in_a_row
def P_avoid_reciting : ℚ := 1 - P_overall

theorem indolent_student_probability : P_avoid_reciting = 11 / 30 := 
  sorry

end indolent_student_probability_l50_50898


namespace boys_meet_time_is_correct_l50_50821

structure TrackMeetProblem where
  (track_length : ℕ) -- Track length in meters
  (speed_first_boy_kmh : ℚ) -- Speed of the first boy in km/hr
  (speed_second_boy_kmh : ℚ) -- Speed of the second boy in km/hr

noncomputable def time_to_meet (p : TrackMeetProblem) : ℚ :=
  let speed_first_boy_ms := (p.speed_first_boy_kmh * 1000) / 3600
  let speed_second_boy_ms := (p.speed_second_boy_kmh * 1000) / 3600
  let relative_speed := speed_first_boy_ms + speed_second_boy_ms
  (p.track_length : ℚ) / relative_speed

theorem boys_meet_time_is_correct (p : TrackMeetProblem) : 
  p.track_length = 4800 → 
  p.speed_first_boy_kmh = 61.3 → 
  p.speed_second_boy_kmh = 97.5 → 
  time_to_meet p = 108.8 := by
  intros
  sorry  

end boys_meet_time_is_correct_l50_50821


namespace second_investment_value_l50_50980

theorem second_investment_value
  (a : ℝ) (r1 r2 rt : ℝ) (x : ℝ)
  (h1 : a = 500)
  (h2 : r1 = 0.07)
  (h3 : r2 = 0.09)
  (h4 : rt = 0.085)
  (h5 : r1 * a + r2 * x = rt * (a + x)) :
  x = 1500 :=
by 
  -- The proof will go here
  sorry

end second_investment_value_l50_50980


namespace expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l50_50661

section
variables (a b : ℚ)

theorem expansion_of_a_plus_b_pow_4 :
  (a + b) ^ 4 = a ^ 4 + 4 * a ^ 3 * b + 6 * a ^ 2 * b ^ 2 + 4 * a * b ^ 3 + b ^ 4 :=
sorry

theorem expansion_of_a_plus_b_pow_5 :
  (a + b) ^ 5 = a ^ 5 + 5 * a ^ 4 * b + 10 * a ^ 3 * b ^ 2 + 10 * a ^ 2 * b ^ 3 + 5 * a * b ^ 4 + b ^ 5 :=
sorry

theorem computation_of_formula :
  2^4 + 4*2^3*(-1/3) + 6*2^2*(-1/3)^2 + 4*2*(-1/3)^3 + (-1/3)^4 = 625 / 81 :=
sorry
end

end expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l50_50661


namespace ratio_of_sides_of_rectangles_l50_50198

theorem ratio_of_sides_of_rectangles (s x y : ℝ) 
  (hsx : x + s = 2 * s) 
  (hsy : s + 2 * y = 2 * s)
  (houter_inner_area : (2 * s) ^ 2 = 4 * s ^ 2) : 
  x / y = 2 :=
by
  -- Assuming the conditions hold, we are interested in proving that the ratio x / y = 2
  -- The proof will be provided here
  sorry

end ratio_of_sides_of_rectangles_l50_50198


namespace max_modulus_l50_50930

open Complex

theorem max_modulus (z : ℂ) (h : abs z = 1) : ∃ M, M = 6 ∧ ∀ w, abs (z - w) ≤ M :=
by
  use 6
  sorry

end max_modulus_l50_50930


namespace walt_part_time_job_l50_50995

theorem walt_part_time_job (x : ℝ) 
  (h1 : 0.09 * x + 0.08 * 4000 = 770) : 
  x + 4000 = 9000 := by
  sorry

end walt_part_time_job_l50_50995


namespace bags_filled_l50_50782

def bags_filled_on_certain_day (x : ℕ) : Prop :=
  let bags := x + 3
  let total_cans := 8 * bags
  total_cans = 72

theorem bags_filled {x : ℕ} (h : bags_filled_on_certain_day x) : x = 6 :=
  sorry

end bags_filled_l50_50782


namespace inverse_proportion_quadrants_l50_50167

theorem inverse_proportion_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, x = -2 ∧ y = 3 ∧ y = k / x) →
  (∀ x : ℝ, (x < 0 → k / x > 0) ∧ (x > 0 → k / x < 0)) :=
sorry

end inverse_proportion_quadrants_l50_50167


namespace ab_product_l50_50310

theorem ab_product (a b : ℝ) (h_sol : ∀ x, -1 < x ∧ x < 4 → x^2 + a * x + b < 0) 
  (h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = -1 ∨ x = 4) : 
  a * b = 12 :=
sorry

end ab_product_l50_50310


namespace train_tunnel_length_l50_50373

theorem train_tunnel_length 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (time_for_tail_to_exit : ℝ) 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 90) 
  (h_time_for_tail_to_exit : time_for_tail_to_exit = 2 / 60) :
  ∃ tunnel_length : ℝ, tunnel_length = 1 := 
by
  sorry

end train_tunnel_length_l50_50373


namespace div_by_5_factor_l50_50467

theorem div_by_5_factor {x y z : ℤ} (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * 5 * (y - z) * (z - x) * (x - y) :=
sorry

end div_by_5_factor_l50_50467


namespace small_beaker_salt_fraction_l50_50444

theorem small_beaker_salt_fraction
  (S L : ℝ) 
  (h1 : L = 5 * S)
  (h2 : L * (1 / 5) = S)
  (h3 : L * 0.3 = S * 1.5)
  : (S * 0.5) / S = 0.5 :=
by 
  sorry

end small_beaker_salt_fraction_l50_50444


namespace largest_4_digit_congruent_15_mod_22_l50_50227

theorem largest_4_digit_congruent_15_mod_22 :
  ∃ (x : ℤ), x < 10000 ∧ x % 22 = 15 ∧ (∀ (y : ℤ), y < 10000 ∧ y % 22 = 15 → y ≤ x) → x = 9981 :=
sorry

end largest_4_digit_congruent_15_mod_22_l50_50227


namespace teammates_score_l50_50446

def Lizzie_score := 4
def Nathalie_score := Lizzie_score + 3
def combined_Lizzie_Nathalie := Lizzie_score + Nathalie_score
def Aimee_score := 2 * combined_Lizzie_Nathalie
def total_team_score := 50
def total_combined_score := Lizzie_score + Nathalie_score + Aimee_score

theorem teammates_score : total_team_score - total_combined_score = 17 :=
by
  sorry

end teammates_score_l50_50446


namespace sum_of_number_and_its_radical_conjugate_l50_50017

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l50_50017


namespace f_2015_eq_neg_2014_l50_50141

variable {f : ℝ → ℝ}

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f1_value : f 1 = 2014 := sorry

-- Theorem to prove
theorem f_2015_eq_neg_2014 :
  isOddFunction f → isPeriodic f 3 → (f 1 = 2014) → f 2015 = -2014 :=
by
  intros hOdd hPeriodic hF1
  sorry

end f_2015_eq_neg_2014_l50_50141


namespace sandwiches_final_count_l50_50485

def sandwiches_left (initial : ℕ) (eaten_by_ruth : ℕ) (given_to_brother : ℕ) (eaten_by_first_cousin : ℕ) (eaten_by_other_cousins : ℕ) : ℕ :=
  initial - (eaten_by_ruth + given_to_brother + eaten_by_first_cousin + eaten_by_other_cousins)

theorem sandwiches_final_count :
  sandwiches_left 10 1 2 2 2 = 3 := by
  sorry

end sandwiches_final_count_l50_50485


namespace petrol_expense_l50_50166

theorem petrol_expense 
  (rent milk groceries education misc savings petrol total_salary : ℝ)
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : education = 2500)
  (H5 : misc = 6100)
  (H6 : savings = 2400)
  (H7 : total_salary = savings / 0.10)
  (H8 : total_salary = rent + milk + groceries + education + misc + petrol + savings) :
  petrol = 2000 :=
by
  sorry

end petrol_expense_l50_50166


namespace hal_paul_difference_l50_50396

def halAnswer : Int := 12 - (3 * 2) + 4
def paulAnswer : Int := (12 - 3) * 2 + 4

theorem hal_paul_difference :
  halAnswer - paulAnswer = -12 := by
  sorry

end hal_paul_difference_l50_50396


namespace exterior_angle_regular_octagon_l50_50660

theorem exterior_angle_regular_octagon : 
  ∀ {θ : ℝ}, 
  (8 - 2) * 180 / 8 = θ →
  180 - θ = 45 := 
by 
  intro θ hθ
  sorry

end exterior_angle_regular_octagon_l50_50660


namespace chomp_game_configurations_l50_50761

/-- Number of valid configurations such that 0 ≤ a_1 ≤ a_2 ≤ ... ≤ a_5 ≤ 7 is 330 -/
theorem chomp_game_configurations :
  let valid_configs := {a : Fin 6 → Fin 8 // (∀ i j, i ≤ j → a i ≤ a j)}
  Fintype.card valid_configs = 330 :=
sorry

end chomp_game_configurations_l50_50761


namespace interval_solution_l50_50851

theorem interval_solution :
  { x : ℝ | 2 < 3 * x ∧ 3 * x < 3 ∧ 2 < 4 * x ∧ 4 * x < 3 } =
  { x : ℝ | (2 / 3) < x ∧ x < (3 / 4) } :=
by
  sorry

end interval_solution_l50_50851


namespace direct_proportion_function_l50_50735

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 3 * x^2 + 2

-- Definition of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, f x = k * x)

-- Theorem statement
theorem direct_proportion_function : is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l50_50735


namespace inscribed_sphere_radius_l50_50743

theorem inscribed_sphere_radius {V S1 S2 S3 S4 R : ℝ} :
  (1/3) * R * (S1 + S2 + S3 + S4) = V → 
  R = 3 * V / (S1 + S2 + S3 + S4) :=
by
  intro h
  sorry

end inscribed_sphere_radius_l50_50743


namespace area_of_kite_l50_50705

theorem area_of_kite (A B C D : ℝ × ℝ) (hA : A = (2, 3)) (hB : B = (6, 7)) (hC : C = (10, 3)) (hD : D = (6, 0)) : 
  let base := (C.1 - A.1)
  let height := (B.2 - D.2)
  let area := 2 * (1 / 2 * base * height)
  area = 56 := 
by
  sorry

end area_of_kite_l50_50705


namespace probability_of_dice_outcome_l50_50436

theorem probability_of_dice_outcome : 
  let p_one_digit := 3 / 4
  let p_two_digit := 1 / 4
  let comb := Nat.choose 5 3
  (comb * (p_one_digit^3) * (p_two_digit^2)) = 135 / 512 := 
by
  sorry

end probability_of_dice_outcome_l50_50436


namespace rosa_peaches_more_than_apples_l50_50265

def steven_peaches : ℕ := 17
def steven_apples  : ℕ := 16
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples  : ℕ := steven_apples + 8
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples  : ℕ := steven_apples / 2

theorem rosa_peaches_more_than_apples : rosa_peaches - rosa_apples = 25 := by
  sorry

end rosa_peaches_more_than_apples_l50_50265


namespace cos_identity_15_30_degrees_l50_50118

theorem cos_identity_15_30_degrees (a b : ℝ) (h : b = 2 * a^2 - 1) : 2 * a^2 - b = 1 :=
by
  sorry

end cos_identity_15_30_degrees_l50_50118


namespace ram_pairs_sold_correct_l50_50392

-- Define the costs
def graphics_card_cost := 600
def hard_drive_cost := 80
def cpu_cost := 200
def ram_pair_cost := 60

-- Define the number of items sold
def graphics_cards_sold := 10
def hard_drives_sold := 14
def cpus_sold := 8
def total_earnings := 8960

-- Calculate earnings from individual items
def earnings_graphics_cards := graphics_cards_sold * graphics_card_cost
def earnings_hard_drives := hard_drives_sold * hard_drive_cost
def earnings_cpus := cpus_sold * cpu_cost

-- Calculate total earnings from graphics cards, hard drives, and CPUs
def earnings_other_items := earnings_graphics_cards + earnings_hard_drives + earnings_cpus

-- Calculate earnings from RAM
def earnings_from_ram := total_earnings - earnings_other_items

-- Calculate number of RAM pairs sold
def ram_pairs_sold := earnings_from_ram / ram_pair_cost

-- The theorem to be proven
theorem ram_pairs_sold_correct : ram_pairs_sold = 4 :=
by
  sorry

end ram_pairs_sold_correct_l50_50392


namespace fill_tank_time_l50_50180

/-- 
If pipe A fills a tank in 30 minutes, pipe B fills the same tank in 20 minutes, 
and pipe C empties it in 40 minutes, then the time it takes to fill the tank 
when all three pipes are working together is 120/7 minutes.
-/
theorem fill_tank_time 
  (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (combined_rate : ℝ) (T : ℝ) :
  rate_A = 1/30 ∧ rate_B = 1/20 ∧ rate_C = -1/40 ∧ combined_rate = rate_A + rate_B + rate_C
  → T = 1 / combined_rate
  → T = 120 / 7 :=
by
  intros
  sorry

end fill_tank_time_l50_50180


namespace largest_int_with_remainder_5_lt_100_l50_50289

theorem largest_int_with_remainder_5_lt_100 (x : ℤ) : (x < 100) ∧ (x % 8 = 5) → x = 93 := 
sorry

end largest_int_with_remainder_5_lt_100_l50_50289


namespace crayons_total_correct_l50_50765

-- Definitions from the conditions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Expected total crayons as per the conditions and the correct answer
def total_crayons_expected : ℕ := 12

-- The proof statement
theorem crayons_total_correct :
  initial_crayons + added_crayons = total_crayons_expected :=
by
  -- Proof details here
  sorry

end crayons_total_correct_l50_50765


namespace speed_including_stoppages_l50_50299

theorem speed_including_stoppages : 
  ∀ (speed_excluding_stoppages : ℝ) (stoppage_minutes_per_hour : ℝ), 
  speed_excluding_stoppages = 65 → 
  stoppage_minutes_per_hour = 15.69 → 
  (speed_excluding_stoppages * (1 - stoppage_minutes_per_hour / 60)) = 47.9025 := 
by intros speed_excluding_stoppages stoppage_minutes_per_hour h1 h2
   sorry

end speed_including_stoppages_l50_50299


namespace ellipse_hyperbola_tangent_m_eq_l50_50340

variable (x y m : ℝ)

def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 2)^2 = 1
def curves_tangent (x m : ℝ) : Prop := ∃ y, ellipse x y ∧ hyperbola x y m

theorem ellipse_hyperbola_tangent_m_eq :
  (∃ x, curves_tangent x (12/13)) ↔ true := 
by
  sorry

end ellipse_hyperbola_tangent_m_eq_l50_50340


namespace number_of_oddly_powerful_integers_lt_500_l50_50247

noncomputable def count_oddly_powerful_integers_lt_500 : ℕ :=
  let count_cubes := 7 -- we counted cubes: 1^3, 2^3, 3^3, 4^3, 5^3, 6^3, 7^3
  let count_fifth_powers := 1 -- the additional fifth power not a cube: 3^5
  count_cubes + count_fifth_powers

theorem number_of_oddly_powerful_integers_lt_500 : count_oddly_powerful_integers_lt_500 = 8 :=
  sorry

end number_of_oddly_powerful_integers_lt_500_l50_50247


namespace hourly_wage_l50_50556

theorem hourly_wage (reps : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_payment : ℕ) :
  reps = 50 →
  hours_per_day = 8 →
  days = 5 →
  total_payment = 28000 →
  (total_payment / (reps * hours_per_day * days) : ℕ) = 14 :=
by
  intros h_reps h_hours_per_day h_days h_total_payment
  -- Now the proof steps can be added here
  sorry

end hourly_wage_l50_50556


namespace word_identification_l50_50096

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end word_identification_l50_50096


namespace son_l50_50342

theorem son's_age (S M : ℕ) (h1 : M = S + 20) (h2 : M + 2 = 2 * (S + 2)) : S = 18 := by
  sorry

end son_l50_50342


namespace simplify_expression_l50_50213

theorem simplify_expression : (2^4 * 2^4 * 2^4) = 2^12 :=
by
  sorry

end simplify_expression_l50_50213


namespace minimum_distance_AB_l50_50999

-- Definitions of the curves C1 and C2
def C1 (x y : ℝ) : Prop := x^2 - y + 1 = 0
def C2 (x y : ℝ) : Prop := y^2 - x + 1 = 0

theorem minimum_distance_AB :
  ∃ (A B : ℝ × ℝ), C1 A.1 A.2 ∧ C2 B.1 B.2 ∧ dist A B = 3*Real.sqrt 2 / 4 := sorry

end minimum_distance_AB_l50_50999


namespace simplify_and_evaluate_l50_50293

theorem simplify_and_evaluate (a : ℝ) (h : a = -3 / 2) : 
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := 
by 
  sorry

end simplify_and_evaluate_l50_50293


namespace salary_increase_after_five_years_l50_50872

theorem salary_increase_after_five_years :
  ∀ (S : ℝ), (S * (1.15)^5 - S) / S * 100 = 101.14 := by
sorry

end salary_increase_after_five_years_l50_50872


namespace safe_security_system_l50_50546

theorem safe_security_system (commission_members : ℕ) 
                            (majority_access : ℕ)
                            (max_inaccess_members : ℕ) 
                            (locks : ℕ)
                            (keys_per_member : ℕ) :
  commission_members = 11 →
  majority_access = 6 →
  max_inaccess_members = 5 →
  locks = (Nat.choose 11 5) →
  keys_per_member = (locks * 6) / 11 →
  locks = 462 ∧ keys_per_member = 252 :=
by
  intros
  sorry

end safe_security_system_l50_50546


namespace slope_of_line_l50_50984

theorem slope_of_line :
  ∃ (m : ℝ), (∃ b : ℝ, ∀ x y : ℝ, y = m * x + b) ∧
             (b = 2 ∧ ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ = 0 ∧ x₂ = 269 ∧ y₁ = 2 ∧ y₂ = 540 ∧ 
             m = (y₂ - y₁) / (x₂ - x₁)) ∧
             m = 2 :=
by {
  sorry
}

end slope_of_line_l50_50984


namespace ratio_of_spinsters_to_cats_l50_50155

theorem ratio_of_spinsters_to_cats (S C : ℕ) (hS : S = 12) (hC : C = S + 42) : S / gcd S C = 2 ∧ C / gcd S C = 9 :=
by
  -- skip proof (use sorry)
  sorry

end ratio_of_spinsters_to_cats_l50_50155


namespace ratio_of_products_l50_50029

variable (a b c d : ℚ) -- assuming a, b, c, d are rational numbers

theorem ratio_of_products (h1 : a = 3 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) :
  a * c / (b * d) = 15 := by
  sorry

end ratio_of_products_l50_50029


namespace reflection_coefficient_l50_50065

theorem reflection_coefficient (I_0 : ℝ) (I_4 : ℝ) (k : ℝ) 
  (h1 : I_4 = I_0 * (1 - k)^4) 
  (h2 : I_4 = I_0 / 256) : 
  k = 0.75 :=
by 
  -- Proof omitted
  sorry

end reflection_coefficient_l50_50065


namespace initial_solution_weight_100kg_l50_50974

theorem initial_solution_weight_100kg
  (W : ℝ)
  (initial_salt_percentage : ℝ)
  (added_salt : ℝ)
  (final_salt_percentage : ℝ)
  (H1 : initial_salt_percentage = 0.10)
  (H2 : added_salt = 12.5)
  (H3 : final_salt_percentage = 0.20)
  (H4 : 0.20 * (W + 12.5) = 0.10 * W + 12.5) :
  W = 100 :=   
by 
  sorry

end initial_solution_weight_100kg_l50_50974


namespace find_x_squared_plus_inverse_squared_l50_50711

theorem find_x_squared_plus_inverse_squared (x : ℝ) 
(h : x^4 + (1 / x^4) = 2398) : 
  x^2 + (1 / x^2) = 20 * Real.sqrt 6 :=
sorry

end find_x_squared_plus_inverse_squared_l50_50711


namespace find_triangle_angles_l50_50800

theorem find_triangle_angles (a b h_a h_b : ℝ) (A B C : ℝ) :
  a ≤ h_a → b ≤ h_b →
  h_a ≤ b → h_b ≤ a →
  ∃ x y z : ℝ, (x = 90 ∧ y = 45 ∧ z = 45) ∧ 
  (x + y + z = 180) :=
by
  sorry

end find_triangle_angles_l50_50800


namespace length_of_tube_l50_50562

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end length_of_tube_l50_50562


namespace find_ec_l50_50946

theorem find_ec (angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
  (angle_DBC_2_angle_ECB : Prop) :
  angle_A = 45 ∧ 
  BC = 8 ∧
  BD_perp_AC ∧
  CE_perp_AB ∧
  angle_DBC_2_angle_ECB → 
  ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 2 ∧ a + b + c = 7 :=
sorry

end find_ec_l50_50946


namespace ratio_lcm_gcf_280_476_l50_50329

theorem ratio_lcm_gcf_280_476 : 
  let a := 280
  let b := 476
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  lcm_ab / gcf_ab = 170 := by
  sorry

end ratio_lcm_gcf_280_476_l50_50329


namespace weight_of_new_person_l50_50548

theorem weight_of_new_person 
  (avg_weight_increase : ℝ)
  (old_weight : ℝ) 
  (num_people : ℕ)
  (new_weight_increase : ℝ)
  (total_weight_increase : ℝ)  
  (W : ℝ)
  (h1 : avg_weight_increase = 1.8)
  (h2 : old_weight = 69)
  (h3 : num_people = 6) 
  (h4 : new_weight_increase = num_people * avg_weight_increase) 
  (h5 : total_weight_increase = new_weight_increase)
  (h6 : W = old_weight + total_weight_increase)
  : W = 79.8 := 
by
  sorry

end weight_of_new_person_l50_50548


namespace sugar_mixture_problem_l50_50189

theorem sugar_mixture_problem :
  ∃ x : ℝ, (9 * x + 7 * (63 - x) = 0.9 * (9.24 * 63)) ∧ x = 41.724 :=
by
  sorry

end sugar_mixture_problem_l50_50189


namespace collapsed_buildings_l50_50323

theorem collapsed_buildings (initial_collapse : ℕ) (collapse_one : initial_collapse = 4)
                            (collapse_double : ∀ n m, m = 2 * n) : (4 + 8 + 16 + 32 = 60) :=
by
  sorry

end collapsed_buildings_l50_50323


namespace difference_between_percent_and_fraction_l50_50603

-- Define the number
def num : ℕ := 140

-- Define the percentage and fraction calculations
def percent_65 (n : ℕ) : ℕ := (65 * n) / 100
def fraction_4_5 (n : ℕ) : ℕ := (4 * n) / 5

-- Define the problem's conditions and the required proof
theorem difference_between_percent_and_fraction : 
  percent_65 num ≤ fraction_4_5 num ∧ (fraction_4_5 num - percent_65 num = 21) :=
by
  sorry

end difference_between_percent_and_fraction_l50_50603


namespace max_radius_of_circle_l50_50199

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l50_50199


namespace one_number_greater_than_one_l50_50813

theorem one_number_greater_than_one 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > (1 / a) + (1 / b) + (1 / c)) 
  : (1 < a ∨ 1 < b ∨ 1 < c) ∧ ¬(1 < a ∧ 1 < b ∧ 1 < c) :=
by
  sorry

end one_number_greater_than_one_l50_50813


namespace girl_walking_speed_l50_50192

-- Definitions of the conditions
def distance := 30 -- in kilometers
def time := 6 -- in hours

-- Definition of the walking speed function
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- The theorem we want to prove
theorem girl_walking_speed : speed distance time = 5 := by
  sorry

end girl_walking_speed_l50_50192


namespace value_a2_plus_b2_l50_50915

noncomputable def a_minus_b : ℝ := 8
noncomputable def ab : ℝ := 49.99999999999999

theorem value_a2_plus_b2 (a b : ℝ) (h1 : a - b = a_minus_b) (h2 : a * b = ab) :
  a^2 + b^2 = 164 := by
  sorry

end value_a2_plus_b2_l50_50915


namespace age_weight_not_proportional_l50_50879

theorem age_weight_not_proportional (age weight : ℕ) : ¬(∃ k, ∀ (a w : ℕ), w = k * a → age / weight = k) :=
by
  sorry

end age_weight_not_proportional_l50_50879


namespace triangle_area_l50_50389

theorem triangle_area {r : ℝ} (h_r : r = 6) {x : ℝ} 
  (h1 : 5 * x = 2 * r)
  (h2 : x = 12 / 5) : 
  (1 / 2 * (3 * x) * (4 * x) = 34.56) :=
by
  sorry

end triangle_area_l50_50389


namespace billy_boxes_of_candy_l50_50217

theorem billy_boxes_of_candy (pieces_per_box total_pieces : ℕ) (h1 : pieces_per_box = 3) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 7 := 
by
  sorry

end billy_boxes_of_candy_l50_50217


namespace partial_fraction_decomposition_l50_50829

noncomputable def polynomial : Polynomial ℝ := Polynomial.X^3 - 24 * Polynomial.X^2 + 143 * Polynomial.X - 210

theorem partial_fraction_decomposition (A B C p q r : ℝ) (h1 : Polynomial.roots polynomial = {p, q, r}) 
  (h2 : ∀ s : ℝ, 1 / (s^3 - 24 * s^2 + 143 * s - 210) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 243 :=
by
  sorry

end partial_fraction_decomposition_l50_50829


namespace inequality_part1_inequality_part2_l50_50062

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l50_50062


namespace seokjin_rank_l50_50751

-- Define the ranks and the people between them as given conditions in the problem
def jimin_rank : Nat := 4
def people_between : Nat := 19

-- The goal is to prove that Seokjin's rank is 24
theorem seokjin_rank : jimin_rank + people_between + 1 = 24 := 
by
  sorry

end seokjin_rank_l50_50751


namespace fraction_value_l50_50818

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l50_50818


namespace surface_area_of_sphere_l50_50832

theorem surface_area_of_sphere (l w h : ℝ) (s t : ℝ) :
  l = 3 ∧ w = 2 ∧ h = 1 ∧ (s = (l^2 + w^2 + h^2).sqrt / 2) → t = 4 * Real.pi * s^2 → t = 14 * Real.pi :=
by
  intros
  sorry

end surface_area_of_sphere_l50_50832


namespace coefficient_a6_l50_50186

def expand_equation (x a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) : Prop :=
  x * (x - 2) ^ 8 =
    a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 +
    a5 * (x - 1) ^ 5 + a6 * (x - 1) ^ 6 + a7 * (x - 1) ^ 7 + a8 * (x - 1) ^ 8 + 
    a9 * (x - 1) ^ 9

theorem coefficient_a6 (x a0 a1 a2 a3 a4 a5 a7 a8 a9 : ℝ) (h : expand_equation x a0 a1 a2 a3 a4 a5 (-28) a7 a8 a9) :
  a6 = -28 :=
sorry

end coefficient_a6_l50_50186


namespace expression_at_x_equals_2_l50_50162

theorem expression_at_x_equals_2 (a b : ℝ) (h : 2 * a - b = -1) : (2 * b - 4 * a) = 2 :=
by {
  sorry
}

end expression_at_x_equals_2_l50_50162


namespace geometric_sequence_eleventh_term_l50_50146

theorem geometric_sequence_eleventh_term (a₁ : ℚ) (r : ℚ) (n : ℕ) (hₐ : a₁ = 5) (hᵣ : r = 2 / 3) (hₙ : n = 11) :
  (a₁ * r^(n - 1) = 5120 / 59049) :=
by
  -- conditions of the problem
  rw [hₐ, hᵣ, hₙ]
  sorry

end geometric_sequence_eleventh_term_l50_50146


namespace greatest_a_inequality_l50_50426

theorem greatest_a_inequality :
  ∃ a : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) ∧
          (∀ b : ℝ, (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁ * x₂ + x₂ * x₃ + x₃ * x₄ + x₄ * x₅)) → b ≤ a) ∧
          a = 2 / Real.sqrt 3 :=
sorry

end greatest_a_inequality_l50_50426


namespace total_guests_at_least_one_reunion_l50_50691

-- Definitions used in conditions
def attendeesOates := 42
def attendeesYellow := 65
def attendeesBoth := 7

-- Definition of the total number of guests attending at least one of the reunions
def totalGuests := attendeesOates + attendeesYellow - attendeesBoth

-- Theorem stating that the total number of guests is equal to 100
theorem total_guests_at_least_one_reunion : totalGuests = 100 :=
by
  -- skipping the proof with sorry
  sorry

end total_guests_at_least_one_reunion_l50_50691


namespace lcm_16_35_l50_50666

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end lcm_16_35_l50_50666


namespace variance_transformation_example_l50_50153

def variance (X : List ℝ) : ℝ := sorry -- Assuming some definition of variance

theorem variance_transformation_example {n : ℕ} (X : List ℝ) (h_len : X.length = 2021) (h_var : variance X = 3) :
  variance (X.map (fun x => 3 * (x - 2))) = 27 := 
sorry

end variance_transformation_example_l50_50153


namespace inequality_solution_set_l50_50944

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 1) / (1 - 2 * x) ≥ 0} = {x : ℝ | -1 / 3 ≤ x ∧ x < 1 / 2} := by
  sorry

end inequality_solution_set_l50_50944


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l50_50443

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l50_50443


namespace compute_expression_l50_50108

theorem compute_expression :
  (-9 * 5) - (-7 * -2) + (11 * -4) = -103 :=
by
  sorry

end compute_expression_l50_50108


namespace complement_intersection_l50_50441

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3, 4}
def A_complement : Set ℕ := U \ A

theorem complement_intersection :
  (A_complement ∩ B) = {2, 4} :=
by 
  sorry

end complement_intersection_l50_50441


namespace parabola_and_line_sum_l50_50150

theorem parabola_and_line_sum (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y : ℝ, (y^2 = 4 * x) ↔ (x, y) = A ∨ (x, y) = B)
  (h_line : ∀ x y : ℝ, (2 * x + y - 4 = 0) ↔ (x, y) = A ∨ (x, y) = B)
  (h_focus : F = (1, 0))
  : |F - A| + |F - B| = 7 := 
sorry

end parabola_and_line_sum_l50_50150


namespace find_rate_of_interest_l50_50458

/-- At what rate percent on simple interest will Rs. 25,000 amount to Rs. 34,500 in 5 years? 
    Given Principal (P) = Rs. 25,000, Amount (A) = Rs. 34,500, Time (T) = 5 years. 
    We need to find the Rate (R). -/
def principal : ℝ := 25000
def amount : ℝ := 34500
def time : ℝ := 5

theorem find_rate_of_interest (P A T : ℝ) : 
  P = principal → 
  A = amount → 
  T = time → 
  ∃ R : ℝ, R = 7.6 :=
by
  intros hP hA hT
  -- proof goes here
  sorry

end find_rate_of_interest_l50_50458


namespace cylinder_surface_area_l50_50723

noncomputable def total_surface_area_cylinder (r h : ℝ) : ℝ :=
  let base_area := 64 * Real.pi
  let lateral_surface_area := 2 * Real.pi * r * h
  let total_surface_area := 2 * base_area + lateral_surface_area
  total_surface_area

theorem cylinder_surface_area (r h : ℝ) (hr : Real.pi * r^2 = 64 * Real.pi) (hh : h = 2 * r) : 
  total_surface_area_cylinder r h = 384 * Real.pi := by
  sorry

end cylinder_surface_area_l50_50723


namespace sum_of_a_and_b_l50_50543

-- Define conditions
def population_size : ℕ := 55
def sample_size : ℕ := 5
def interval : ℕ := population_size / sample_size
def sample_indices : List ℕ := [6, 28, 50]

-- Assume a and b are such that the systematic sampling is maintained
variable (a b : ℕ)
axiom a_idx : a = sample_indices.head! + interval
axiom b_idx : b = sample_indices.getLast! - interval

-- Define Lean 4 statement to prove
theorem sum_of_a_and_b :
  (a + b) = 56 :=
by
  -- This will be the place where the proof is inserted
  sorry

end sum_of_a_and_b_l50_50543


namespace fundraiser_total_money_l50_50427

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l50_50427


namespace expression_zero_iff_x_eq_three_l50_50611

theorem expression_zero_iff_x_eq_three (x : ℝ) :
  (4 * x - 8 ≠ 0) → ((x^2 - 6 * x + 9 = 0) ↔ (x = 3)) :=
by
  sorry

end expression_zero_iff_x_eq_three_l50_50611


namespace find_b_l50_50221

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (527816429 - b) % 17 = 0 ∧ b = 8 := 
by 
  sorry

end find_b_l50_50221


namespace intersection_A_B_l50_50577

-- Definition of sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y > 0 }

-- The proof goal
theorem intersection_A_B : A ∩ B = { x | x > 1 } :=
by sorry

end intersection_A_B_l50_50577


namespace percentage_increase_l50_50673

theorem percentage_increase (C S : ℝ) (h1 : S = 4.2 * C) 
  (h2 : ∃ X : ℝ, (S - (C + (X / 100) * C) = (2 / 3) * S)) : 
  ∃ X : ℝ, (C + (X / 100) * C - C)/(C) = 40 / 100 := 
by
  sorry

end percentage_increase_l50_50673


namespace boys_attended_dance_l50_50308

variables (d b : ℕ)

theorem boys_attended_dance (h1 : b = 2 * d) (h2 : b = (d - 1) + 8) : b = 14 :=
by sorry

end boys_attended_dance_l50_50308


namespace arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l50_50429

-- Definitions based on conditions in A)
def students : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def A : Char := 'A'
def B : Char := 'B'
def C : Char := 'C'
def D : Char := 'D'
def E : Char := 'E'
def F : Char := 'F'
def G : Char := 'G'

-- Holistic theorem statements for each question derived from the correct answers in B)
theorem arrangement_A_and_B_adjacent :
  ∃ (n : ℕ), n = 1440 := sorry

theorem arrangement_A_B_and_C_adjacent :
  ∃ (n : ℕ), n = 720 := sorry

theorem arrangement_A_and_B_adjacent_C_not_ends :
  ∃ (n : ℕ), n = 960 := sorry

theorem arrangement_ABC_and_DEFG_units :
  ∃ (n : ℕ), n = 288 := sorry

end arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l50_50429


namespace inequality_proof_l50_50197

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c :=
sorry

end inequality_proof_l50_50197


namespace greatest_integer_third_side_l50_50677

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l50_50677


namespace number_thought_of_eq_95_l50_50753

theorem number_thought_of_eq_95 (x : ℝ) (h : (x / 5) + 23 = 42) : x = 95 := 
by
  sorry

end number_thought_of_eq_95_l50_50753


namespace find_dividend_l50_50327

-- Definitions based on conditions from the problem
def divisor : ℕ := 13
def quotient : ℕ := 17
def remainder : ℕ := 1

-- Statement of the proof problem
theorem find_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

-- Proof statement ensuring dividend is as expected
example : find_dividend divisor quotient remainder = 222 :=
by 
  sorry

end find_dividend_l50_50327


namespace polynomial_value_at_3_l50_50243

theorem polynomial_value_at_3 :
  ∃ (P : ℕ → ℚ), 
    (∀ (x : ℕ), P x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6) ∧ 
    (∀ (i : ℕ), i ≤ 6 → 0 ≤ b_i ∧ b_i < 5) ∧ 
    P (Nat.sqrt 5) = 35 + 26 * Nat.sqrt 5 -> 
    P 3 = 437 := 
by
  simp
  sorry

end polynomial_value_at_3_l50_50243


namespace quadratic_completion_l50_50935

theorem quadratic_completion (x : ℝ) : 
  (2 * x^2 + 3 * x - 1) = 2 * (x + 3 / 4)^2 - 17 / 8 := 
by 
  -- Proof isn't required, we just state the theorem.
  sorry

end quadratic_completion_l50_50935


namespace math_problem_l50_50589

theorem math_problem (n : ℕ) (h : n > 0) : 
  1957 ∣ (1721^(2*n) - 73^(2*n) - 521^(2*n) + 212^(2*n)) :=
sorry

end math_problem_l50_50589


namespace domain_of_f_l50_50048

noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : 
  {x : ℝ | Real.sqrt (x^2 - 5 * x + 6) ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l50_50048


namespace proportion_exists_x_l50_50156

theorem proportion_exists_x : ∃ x : ℕ, 1 * x = 3 * 4 :=
by
  sorry

end proportion_exists_x_l50_50156


namespace sum_alternating_sequence_l50_50259

theorem sum_alternating_sequence : (Finset.range 2012).sum (λ k => (-1 : ℤ)^(k + 1)) = 0 :=
by
  sorry

end sum_alternating_sequence_l50_50259


namespace work_done_by_gravity_l50_50407

noncomputable def work_by_gravity (m g z_A z_B : ℝ) : ℝ :=
  m * g * (z_B - z_A)

theorem work_done_by_gravity (m g z_A z_B : ℝ) :
  work_by_gravity m g z_A z_B = m * g * (z_B - z_A) :=
by
  sorry

end work_done_by_gravity_l50_50407


namespace count_white_balls_l50_50536

variable (W B : ℕ)

theorem count_white_balls
  (h_total : W + B = 30)
  (h_white : ∀ S : Finset ℕ, S.card = 12 → ∃ w ∈ S, w < W)
  (h_black : ∀ S : Finset ℕ, S.card = 20 → ∃ b ∈ S, b < B) :
  W = 19 :=
sorry

end count_white_balls_l50_50536


namespace complete_square_form_l50_50649

theorem complete_square_form {a h k : ℝ} :
  ∀ x, (x^2 - 5 * x) = a * (x - h)^2 + k → k = -25 / 4 :=
by
  intro x
  intro h_eq
  sorry

end complete_square_form_l50_50649


namespace sum_of_ages_l50_50853

variable (S F : ℕ)

-- Conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F + 6 = 2 * (S + 6)

-- Theorem Statement
theorem sum_of_ages (h1 : condition1 S F) (h2 : condition2 S F) : S + 6 + (F + 6) = 36 := by
  sorry

end sum_of_ages_l50_50853


namespace stage_order_permutations_l50_50733

-- Define the problem in Lean terms
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem stage_order_permutations :
  let total_students := 6
  let predetermined_students := 3
  (permutations total_students) / (permutations predetermined_students) = 120 := by
  sorry

end stage_order_permutations_l50_50733


namespace find_x_l50_50093

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Definition of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem statement
theorem find_x (x : ℝ) (h_parallel : parallel a (b x)) : x = 6 :=
sorry

end find_x_l50_50093


namespace monthly_income_of_P_l50_50320

theorem monthly_income_of_P (P Q R : ℝ) 
    (h1 : (P + Q) / 2 = 2050) 
    (h2 : (Q + R) / 2 = 5250) 
    (h3 : (P + R) / 2 = 6200) : 
    P = 3000 :=
by
  sorry

end monthly_income_of_P_l50_50320


namespace cos_two_sum_l50_50112

theorem cos_two_sum {α β : ℝ} 
  (h1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α) ^ 2 - 2 * (Real.sin β + Real.cos β) ^ 2 = 1) :
  Real.cos (2 * (α + β)) = -1 / 3 :=
sorry

end cos_two_sum_l50_50112


namespace coefficient_ratio_is_4_l50_50127

noncomputable def coefficient_x3 := 
  let a := 60 -- Coefficient of x^3 in the expansion
  let b := Nat.choose 6 2 -- Binomial coefficient \binom{6}{2}
  a / b

theorem coefficient_ratio_is_4 : coefficient_x3 = 4 := by
  sorry

end coefficient_ratio_is_4_l50_50127


namespace enrique_speed_l50_50916

theorem enrique_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (E : ℝ) :
  distance = 200 ∧ time = 8 ∧ speed_diff = 7 ∧ 
  (2 * E + speed_diff) * time = distance → 
  E = 9 :=
by
  sorry

end enrique_speed_l50_50916


namespace find_m_l50_50268

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * x + m) 
  (h2 : ∀ x ≥ (3 : ℝ), f x ≥ 1) : m = -2 := 
sorry

end find_m_l50_50268


namespace find_value_of_y_l50_50154

noncomputable def angle_sum_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

noncomputable def triangle_ABC : angle_sum_triangle 80 60 x := by
  sorry

noncomputable def triangle_CDE (x y : ℝ) : Prop :=
(x = 40) ∧ (90 + x + y = 180)

theorem find_value_of_y (x y : ℝ) 
  (h1 : angle_sum_triangle 80 60 x)
  (h2 : triangle_CDE x y) : 
  y = 50 := 
by
  sorry

end find_value_of_y_l50_50154


namespace integer_roots_of_polynomial_l50_50076

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = 2 ∨ x = -3 ∨ x = 4 := 
by 
  sorry

end integer_roots_of_polynomial_l50_50076


namespace time_between_train_arrivals_l50_50424

-- Define the conditions as given in the problem statement
def passengers_per_train : ℕ := 320 + 200
def total_passengers_per_hour : ℕ := 6240
def minutes_per_hour : ℕ := 60

-- Declare the statement to be proven
theorem time_between_train_arrivals: 
  (total_passengers_per_hour / passengers_per_train) = (minutes_per_hour / 5) := by 
  sorry

end time_between_train_arrivals_l50_50424


namespace prism_visibility_percentage_l50_50922

theorem prism_visibility_percentage
  (base_edge : ℝ)
  (height : ℝ)
  (cell_side : ℝ)
  (wraps : ℕ)
  (lateral_surface_area : ℝ)
  (transparent_area : ℝ) :
  base_edge = 3.2 →
  height = 5 →
  cell_side = 1 →
  wraps = 2 →
  lateral_surface_area = base_edge * height * 3 →
  transparent_area = 13.8 →
  (transparent_area / lateral_surface_area) * 100 = 28.75 :=
by
  intros h_base_edge h_height h_cell_side h_wraps h_lateral_surface_area h_transparent_area
  sorry

end prism_visibility_percentage_l50_50922


namespace checker_arrangements_five_digit_palindromes_l50_50200

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem checker_arrangements :
  comb 32 12 * comb 20 12 = Nat.choose 32 12 * Nat.choose 20 12 := by
  sorry

theorem five_digit_palindromes :
  9 * 10 * 10 = 900 := by
  sorry

end checker_arrangements_five_digit_palindromes_l50_50200


namespace makeup_set_cost_l50_50435

theorem makeup_set_cost (initial : ℕ) (gift : ℕ) (needed : ℕ) (total_cost : ℕ) :
  initial = 35 → gift = 20 → needed = 10 → total_cost = initial + gift + needed → total_cost = 65 :=
by
  intros h_init h_gift h_needed h_cost
  sorry

end makeup_set_cost_l50_50435


namespace lydia_ate_24_ounces_l50_50963

theorem lydia_ate_24_ounces (total_fruit_pounds : ℕ) (mario_oranges_ounces : ℕ) (nicolai_peaches_pounds : ℕ) (total_fruit_ounces mario_oranges_ounces_in_ounces nicolai_peaches_ounces_in_ounces : ℕ) :
  total_fruit_pounds = 8 →
  mario_oranges_ounces = 8 →
  nicolai_peaches_pounds = 6 →
  total_fruit_ounces = total_fruit_pounds * 16 →
  mario_oranges_ounces_in_ounces = mario_oranges_ounces →
  nicolai_peaches_ounces_in_ounces = nicolai_peaches_pounds * 16 →
  (total_fruit_ounces - mario_oranges_ounces_in_ounces - nicolai_peaches_ounces_in_ounces) = 24 :=
by
  sorry

end lydia_ate_24_ounces_l50_50963


namespace work_days_l50_50779

theorem work_days (m r d : ℕ) (h : 2 * m * d = 2 * (m + r) * (md / (m + r))) : d = md / (m + r) :=
by
  sorry

end work_days_l50_50779


namespace central_angle_radian_measure_l50_50354

-- Define the unit circle radius
def unit_circle_radius : ℝ := 1

-- Given an arc of length 1
def arc_length : ℝ := 1

-- Problem Statement: Prove that the radian measure of the central angle α is 1
theorem central_angle_radian_measure :
  ∀ (r : ℝ) (l : ℝ), r = unit_circle_radius → l = arc_length → |l / r| = 1 :=
by
  intros r l hr hl
  rw [hr, hl]
  sorry

end central_angle_radian_measure_l50_50354


namespace smallest_prime_divides_polynomial_l50_50908

theorem smallest_prime_divides_polynomial : 
  ∃ n : ℤ, n^2 + 5 * n + 23 = 17 := 
sorry

end smallest_prime_divides_polynomial_l50_50908


namespace measure_of_angle_l50_50638

theorem measure_of_angle (x : ℝ) 
  (h₁ : 180 - x = 3 * x - 10) : x = 47.5 :=
by 
  sorry

end measure_of_angle_l50_50638


namespace units_digit_of_150_factorial_is_zero_l50_50734

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l50_50734


namespace total_points_is_400_l50_50179

-- Define the conditions as definitions in Lean 4 
def pointsPerEnemy : ℕ := 15
def bonusPoints : ℕ := 50
def totalEnemies : ℕ := 25
def enemiesLeftUndestroyed : ℕ := 5
def bonusesEarned : ℕ := 2

-- Calculate the total number of enemies defeated
def enemiesDefeated : ℕ := totalEnemies - enemiesLeftUndestroyed

-- Calculate the points from defeating enemies
def pointsFromEnemies := enemiesDefeated * pointsPerEnemy

-- Calculate the total bonus points
def totalBonusPoints := bonusesEarned * bonusPoints

-- The total points earned is the sum of points from enemies and bonus points
def totalPointsEarned := pointsFromEnemies + totalBonusPoints

-- Prove that the total points earned is equal to 400
theorem total_points_is_400 : totalPointsEarned = 400 := by
    sorry

end total_points_is_400_l50_50179


namespace number_of_arrangements_l50_50060

theorem number_of_arrangements (teams : Finset ℕ) (sites : Finset ℕ) :
  (∀ team, team ∈ teams → (team ∈ sites)) ∧ ((Finset.card sites = 3) ∧ (Finset.card teams = 6)) ∧ 
  (∃ (a b c : ℕ), a + b + c = 6 ∧ a >= 2 ∧ b >= 1 ∧ c >= 1) →
  ∃ (n : ℕ), n = 360 :=
sorry

end number_of_arrangements_l50_50060


namespace edward_lawns_forgotten_l50_50941

theorem edward_lawns_forgotten (dollars_per_lawn : ℕ) (total_lawns : ℕ) (total_earned : ℕ) (lawns_mowed : ℕ) (lawns_forgotten : ℕ) :
  dollars_per_lawn = 4 →
  total_lawns = 17 →
  total_earned = 32 →
  lawns_mowed = total_earned / dollars_per_lawn →
  lawns_forgotten = total_lawns - lawns_mowed →
  lawns_forgotten = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end edward_lawns_forgotten_l50_50941


namespace area_of_given_triangle_l50_50884

noncomputable def area_of_triangle (a A B : ℝ) : ℝ :=
  let C := Real.pi - A - B
  let b := a * (Real.sin B / Real.sin A)
  let S := (1 / 2) * a * b * Real.sin C
  S

theorem area_of_given_triangle : area_of_triangle 4 (Real.pi / 4) (Real.pi / 3) = 6 + 2 * Real.sqrt 3 := 
by 
  sorry

end area_of_given_triangle_l50_50884


namespace cement_total_l50_50625

-- Defining variables for the weights of cement
def weight_self : ℕ := 215
def weight_son : ℕ := 137

-- Defining the function that calculates the total weight of the cement
def total_weight (a b : ℕ) : ℕ := a + b

-- Theorem statement: Proving the total cement weight is 352 lbs
theorem cement_total : total_weight weight_self weight_son = 352 :=
by
  sorry

end cement_total_l50_50625


namespace moores_law_l50_50220

theorem moores_law (initial_transistors : ℕ) (doubling_period : ℕ) (t1 t2 : ℕ) 
  (initial_year : t1 = 1985) (final_year : t2 = 2010) (transistors_in_1985 : initial_transistors = 300000) 
  (doubles_every_two_years : doubling_period = 2) : 
  (initial_transistors * 2 ^ ((t2 - t1) / doubling_period) = 1228800000) := 
by
  sorry

end moores_law_l50_50220


namespace isosceles_perimeter_l50_50600

noncomputable def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_perimeter
  (k : ℝ)
  (a b : ℝ)
  (h1 : 4 = a)
  (h2 : k * b^2 - (k + 8) * b + 8 = 0)
  (h3 : k ≠ 0)
  (h4 : is_triangle 4 a a) : a + 4 + a = 9 :=
sorry

end isosceles_perimeter_l50_50600


namespace valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l50_50833

-- Definition: City is divided by roads, and there are initial and additional currency exchange points

structure City := 
(exchange_points : ℕ)   -- Number of exchange points in the city
(parts : ℕ)             -- Number of parts the city is divided into

-- Given: Initial conditions with one existing exchange point and divided parts
def initialCity : City :=
{ exchange_points := 1, parts := 2 }

-- Function to add exchange points in the city
def addExchangePoints (c : City) (new_points : ℕ) : City :=
{ exchange_points := c.exchange_points + new_points, parts := c.parts }

-- Function to verify that each part has exactly two exchange points
def isValidConfiguration (c : City) : Prop :=
c.exchange_points = 2 * c.parts

-- Theorem: Prove that each configuration of new points is valid
theorem valid_first_configuration : 
  isValidConfiguration (addExchangePoints initialCity 3) := 
sorry

theorem valid_second_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_third_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

theorem valid_fourth_configuration : 
  isValidConfiguration (addExchangePoints { exchange_points := 1, parts := 2 } 3) :=
sorry

end valid_first_configuration_valid_second_configuration_valid_third_configuration_valid_fourth_configuration_l50_50833


namespace find_h_at_2_l50_50863

noncomputable def h (x : ℝ) : ℝ := x^4 + 2 * x^3 - 12 * x^2 - 14 * x + 24

lemma poly_value_at_minus_2 : h (-2) = -4 := by
  sorry

lemma poly_value_at_1 : h 1 = -1 := by
  sorry

lemma poly_value_at_minus_4 : h (-4) = -16 := by
  sorry

lemma poly_value_at_3 : h 3 = -9 := by
  sorry

theorem find_h_at_2 : h 2 = -20 := by
  sorry

end find_h_at_2_l50_50863


namespace first_week_tickets_calc_l50_50717

def total_tickets : ℕ := 90
def second_week_tickets : ℕ := 17
def tickets_left : ℕ := 35

theorem first_week_tickets_calc : total_tickets - (second_week_tickets + tickets_left) = 38 := by
  sorry

end first_week_tickets_calc_l50_50717


namespace oblique_asymptote_l50_50275

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end oblique_asymptote_l50_50275


namespace integer_implies_perfect_square_l50_50594

theorem integer_implies_perfect_square (n : ℕ) (h : ∃ m : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = m) :
  ∃ k : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = (k ^ 2) :=
by
  sorry

end integer_implies_perfect_square_l50_50594


namespace max_area_of_triangle_l50_50246

open Real

theorem max_area_of_triangle (a b c : ℝ) 
  (ha : 9 ≥ a) 
  (ha1 : a ≥ 8) 
  (hb : 8 ≥ b) 
  (hb1 : b ≥ 4) 
  (hc : 4 ≥ c) 
  (hc1 : c ≥ 3) : 
  ∃ A : ℝ, ∃ S : ℝ, S ≤ 16 ∧ S = max (1/2 * b * c * sin A) 16 := 
sorry

end max_area_of_triangle_l50_50246


namespace inradius_of_triangle_l50_50500

/-- Given conditions for the triangle -/
def perimeter : ℝ := 32
def area : ℝ := 40

/-- The theorem to prove the inradius of the triangle -/
theorem inradius_of_triangle (h : area = (r * perimeter) / 2) : r = 2.5 :=
by
  sorry

end inradius_of_triangle_l50_50500


namespace final_price_correct_l50_50889

-- Define the initial price of the iPhone
def initial_price : ℝ := 1000

-- Define the discount rates for the first and second month
def first_month_discount : ℝ := 0.10
def second_month_discount : ℝ := 0.20

-- Calculate the price after the first month's discount
def price_after_first_month (price : ℝ) : ℝ := price * (1 - first_month_discount)

-- Calculate the price after the second month's discount
def price_after_second_month (price : ℝ) : ℝ := price * (1 - second_month_discount)

-- Final price calculation after both discounts
def final_price : ℝ := price_after_second_month (price_after_first_month initial_price)

-- Proof statement
theorem final_price_correct : final_price = 720 := by
  sorry

end final_price_correct_l50_50889


namespace solve_quadratic_eq_l50_50350

theorem solve_quadratic_eq (x y : ℝ) :
  (x = 3 ∧ y = 1) ∨ (x = -1 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) ∨ (x = -1 ∧ y = -5) ↔
  x ^ 2 - x * y + y ^ 2 - x + 3 * y - 7 = 0 := sorry

end solve_quadratic_eq_l50_50350


namespace number_of_persons_l50_50280

theorem number_of_persons (P : ℕ) : 
  (P * 12 * 5 = 30 * 13 * 6) → P = 39 :=
by
  sorry

end number_of_persons_l50_50280


namespace monthly_income_P_l50_50205

theorem monthly_income_P (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : (P + R) / 2 = 5200) :
  P = 4000 := 
sorry

end monthly_income_P_l50_50205


namespace smallest_n_divisible_l50_50128

open Nat

theorem smallest_n_divisible (n : ℕ) : (∃ (n : ℕ), n > 0 ∧ 45 ∣ n^2 ∧ 720 ∣ n^3) → n = 60 :=
by
  sorry

end smallest_n_divisible_l50_50128


namespace smallest_number_l50_50349

-- Define the conditions
def is_divisible_by (n d : ℕ) : Prop := d ∣ n

def conditions (n : ℕ) : Prop := 
  (n > 12) ∧ 
  is_divisible_by (n - 12) 12 ∧ 
  is_divisible_by (n - 12) 24 ∧
  is_divisible_by (n - 12) 36 ∧
  is_divisible_by (n - 12) 48 ∧
  is_divisible_by (n - 12) 56

-- State the theorem
theorem smallest_number : ∃ n : ℕ, conditions n ∧ n = 1020 :=
by
  sorry

end smallest_number_l50_50349


namespace total_cost_of_one_pencil_and_eraser_l50_50384

/-- Lila buys 15 pencils and 7 erasers for 170 cents. A pencil costs less than an eraser, 
neither item costs exactly half as much as the other, and both items cost a whole number of cents. 
Prove that the total cost of one pencil and one eraser is 16 cents. -/
theorem total_cost_of_one_pencil_and_eraser (p e : ℕ) (h1 : 15 * p + 7 * e = 170)
  (h2 : p < e) (h3 : p ≠ e / 2) : p + e = 16 :=
sorry

end total_cost_of_one_pencil_and_eraser_l50_50384


namespace map_line_segments_l50_50418

def point : Type := ℝ × ℝ

def transformation (f : point → point) (p q : point) : Prop := f p = q

def counterclockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

def clockwise_rotation_180 (p : point) : point := (-p.1, -p.2)

theorem map_line_segments :
  (transformation counterclockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation counterclockwise_rotation_180 (2, -5) (-2, 5)) ∨
  (transformation clockwise_rotation_180 (3, -2) (-3, 2) ∧
   transformation clockwise_rotation_180 (2, -5) (-2, 5)) :=
by
  sorry

end map_line_segments_l50_50418


namespace projection_is_correct_l50_50933

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end projection_is_correct_l50_50933


namespace fourth_person_height_l50_50985

theorem fourth_person_height (H : ℝ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 77) : 
  H + 10 = 83 :=
sorry

end fourth_person_height_l50_50985


namespace div_a2_plus_2_congr_mod8_l50_50431

variable (a d : ℤ)
variable (h_odd : a % 2 = 1)
variable (h_pos : a > 0)

theorem div_a2_plus_2_congr_mod8 :
  (d ∣ (a ^ 2 + 2)) → (d % 8 = 1 ∨ d % 8 = 3) :=
by
  sorry

end div_a2_plus_2_congr_mod8_l50_50431


namespace max_and_next_max_values_l50_50210

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log a) / b

theorem max_and_next_max_values :
  let values := [4.0^(1/4), 5.0^(1/5), 16.0^(1/16), 25.0^(1/25)]
  ∃ max2 max1, 
    max1 = 4.0^(1/4) ∧ max2 = 5.0^(1/5) ∧ 
    (∀ x ∈ values, x <= max1) ∧ 
    (∀ x ∈ values, x < max1 → x <= max2) :=
by
  sorry

end max_and_next_max_values_l50_50210


namespace eval_f_at_two_eval_f_at_neg_two_l50_50297

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x

theorem eval_f_at_two : f 2 = 14 :=
by
  sorry

theorem eval_f_at_neg_two : f (-2) = 2 :=
by
  sorry

end eval_f_at_two_eval_f_at_neg_two_l50_50297


namespace find_apron_cost_l50_50547

-- Definitions used in the conditions
variables (hand_mitts cost small_knife utensils apron : ℝ)
variables (nieces : ℕ)
variables (total_cost_before_discount total_cost_after_discount : ℝ)

-- Conditions given
def conditions := 
  hand_mitts = 14 ∧ 
  utensils = 10 ∧ 
  small_knife = 2 * utensils ∧
  (total_cost_before_discount : ℝ) = (3 * hand_mitts + 3 * utensils + 3 * small_knife + 3 * apron) ∧
  (total_cost_after_discount : ℝ) = 135 ∧
  total_cost_before_discount * 0.75 = total_cost_after_discount ∧
  nieces = 3

-- Theorem statement (proof problem)
theorem find_apron_cost (h : conditions hand_mitts utensils small_knife apron nieces total_cost_before_discount total_cost_after_discount) : 
  apron = 16 :=
by 
  sorry

end find_apron_cost_l50_50547


namespace max_contestants_l50_50045

theorem max_contestants (n : ℕ) (h1 : n = 55) (h2 : ∀ (i j : ℕ), i < j → j < n → (j - i) % 5 ≠ 4) : ∃(k : ℕ), k = 30 := 
  sorry

end max_contestants_l50_50045


namespace area_of_triangle_PQR_l50_50266

structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -4, y := 2 }
def Q : Point := { x := 8, y := 2 }
def R : Point := { x := 6, y := -4 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
  (1 / 2) * abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))

theorem area_of_triangle_PQR : triangle_area P Q R = 36 := by
  sorry

end area_of_triangle_PQR_l50_50266


namespace net_pay_is_correct_l50_50644

-- Define the gross pay and taxes paid as constants
def gross_pay : ℕ := 450
def taxes_paid : ℕ := 135

-- Define net pay as a function of gross pay and taxes paid
def net_pay (gross : ℕ) (taxes : ℕ) : ℕ := gross - taxes

-- The proof statement
theorem net_pay_is_correct : net_pay gross_pay taxes_paid = 315 := by
  sorry -- The proof goes here

end net_pay_is_correct_l50_50644


namespace find_radius_of_inscribed_sphere_l50_50351

variables (a b c s : ℝ)

theorem find_radius_of_inscribed_sphere
  (h1 : a + b + c = 18)
  (h2 : 2 * (a * b + b * c + c * a) = 216)
  (h3 : a^2 + b^2 + c^2 = 108) :
  s = 3 * Real.sqrt 3 :=
by
  sorry

end find_radius_of_inscribed_sphere_l50_50351


namespace trapezoid_area_correct_l50_50080

noncomputable def trapezoid_area (x : ℝ) : ℝ :=
  let base1 := 3 * x
  let base2 := 5 * x + 2
  (base1 + base2) / 2 * x

theorem trapezoid_area_correct (x : ℝ) : trapezoid_area x = 4 * x^2 + x :=
  by
  sorry

end trapezoid_area_correct_l50_50080


namespace geometric_sequence_condition_neither_necessary_nor_sufficient_l50_50238

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

noncomputable def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_condition_neither_necessary_nor_sufficient (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q → ¬( (is_monotonically_increasing a ↔ q > 1) ) :=
by sorry

end geometric_sequence_condition_neither_necessary_nor_sufficient_l50_50238


namespace range_of_k_l50_50494

open BigOperators

theorem range_of_k
  {f : ℝ → ℝ}
  (k : ℝ)
  (h : ∀ x : ℝ, f x = 32 * x - (k + 1) * 3^x + 2)
  (H : ∀ x : ℝ, f x > 0) :
  k < 1 /2 := 
sorry

end range_of_k_l50_50494


namespace showUpPeopleFirstDay_l50_50951

def cansFood := 2000
def people1stDay (cansTaken_1stDay : ℕ) := cansFood - 1500 = cansTaken_1stDay
def peopleSnapped_1stDay := 500

theorem showUpPeopleFirstDay :
  (people1stDay peopleSnapped_1stDay) → (peopleSnapped_1stDay / 1) = 500 := 
by 
  sorry

end showUpPeopleFirstDay_l50_50951


namespace correct_option_is_B_l50_50116

-- Define the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (m : ℝ) : Prop := (-2 * m^2)^3 = -8 * m^6
def optionC (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def optionD (a b : ℝ) : Prop := 2 * a * b + 3 * a^2 * b = 5 * a^3 * b^2

-- The proof problem: which option is correct
theorem correct_option_is_B (m : ℝ) : optionB m := by
  sorry

end correct_option_is_B_l50_50116


namespace old_man_gold_coins_l50_50019

theorem old_man_gold_coins (x y : ℕ) (h1 : x - y = 1) (h2 : x^2 - y^2 = 25 * (x - y)) : x + y = 25 := 
sorry

end old_man_gold_coins_l50_50019


namespace action_figure_ratio_l50_50720

variable (initial : ℕ) (sold : ℕ) (remaining : ℕ) (left : ℕ)
variable (h1 : initial = 24)
variable (h2 : sold = initial / 4)
variable (h3 : remaining = initial - sold)
variable (h4 : remaining - left = left)

theorem action_figure_ratio
  (h1 : initial = 24)
  (h2 : sold = initial / 4)
  (h3 : remaining = initial - sold)
  (h4 : remaining - left = left) :
  (remaining - left) * 3 = left :=
by
  sorry

end action_figure_ratio_l50_50720


namespace total_present_ages_l50_50483

theorem total_present_ages (P Q : ℕ) 
    (h1 : P - 12 = (1 / 2) * (Q - 12))
    (h2 : P = (3 / 4) * Q) : P + Q = 42 :=
by
  sorry

end total_present_ages_l50_50483


namespace cos_C_of_triangle_l50_50555

theorem cos_C_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (h_sine_relation : 3 * Real.sin A = 2 * Real.sin B)
  (h_cosine_law : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.cos C = -1/4 :=
by
  sorry

end cos_C_of_triangle_l50_50555


namespace combined_volume_cone_hemisphere_cylinder_l50_50533

theorem combined_volume_cone_hemisphere_cylinder (r h : ℝ)
  (vol_cylinder : ℝ) (vol_cone : ℝ) (vol_hemisphere : ℝ)
  (H1 : vol_cylinder = 72 * π)
  (H2 : vol_cylinder = π * r^2 * h)
  (H3 : vol_cone = (1/3) * π * r^2 * h)
  (H4 : vol_hemisphere = (2/3) * π * r^3)
  (H5 : vol_cylinder = vol_cone + vol_hemisphere) :
  vol_cylinder = 72 * π :=
by
  sorry

end combined_volume_cone_hemisphere_cylinder_l50_50533


namespace intersection_complement_eq_l50_50858

def A : Set ℝ := { x | 1 ≤ x ∧ x < 3 }

def B : Set ℝ := { x | x^2 ≥ 4 }

def complementB : Set ℝ := { x | -2 < x ∧ x < 2 }

def intersection (A : Set ℝ) (B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_eq : 
  intersection A complementB = { x | 1 ≤ x ∧ x < 2 } := 
sorry

end intersection_complement_eq_l50_50858


namespace triangle_expression_l50_50523

open Real

variable (D E F : ℝ)
variable (DE DF EF : ℝ)

-- conditions
def triangleDEF : Prop := DE = 7 ∧ DF = 9 ∧ EF = 8

theorem triangle_expression (h : triangleDEF DE DF EF) :
  (cos ((D - E)/2) / sin (F/2) - sin ((D - E)/2) / cos (F/2)) = 81/28 :=
by
  have h1 : DE = 7 := h.1
  have h2 : DF = 9 := h.2.1
  have h3 : EF = 8 := h.2.2
  sorry

end triangle_expression_l50_50523


namespace initial_eggs_proof_l50_50328

-- Definitions based on the conditions provided
def initial_eggs := 7
def added_eggs := 4
def total_eggs := 11

-- The statement to be proved
theorem initial_eggs_proof : initial_eggs + added_eggs = total_eggs :=
by
  -- Placeholder for proof
  sorry

end initial_eggs_proof_l50_50328


namespace alpha_plus_beta_l50_50400

theorem alpha_plus_beta (α β : ℝ) (h : ∀ x, (x - α) / (x + β) = (x^2 - 116 * x + 2783) / (x^2 + 99 * x - 4080)) 
: α + β = 115 := 
sorry

end alpha_plus_beta_l50_50400


namespace ryan_time_learning_l50_50038

variable (t : ℕ) (c : ℕ)

/-- Ryan spends a total of 3 hours on both languages every day. Assume further that he spends 1 hour on learning Chinese every day, and you need to find how many hours he spends on learning English. --/
theorem ryan_time_learning (h_total : t = 3) (h_chinese : c = 1) : (t - c) = 2 := 
by
  -- Proof goes here
  sorry

end ryan_time_learning_l50_50038


namespace train_speed_kmph_l50_50648

def train_length : ℝ := 360
def bridge_length : ℝ := 140
def time_to_pass : ℝ := 40
def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

theorem train_speed_kmph : mps_to_kmph ((train_length + bridge_length) / time_to_pass) = 45 := 
by {
  sorry
}

end train_speed_kmph_l50_50648


namespace right_triangle_width_l50_50758

theorem right_triangle_width (height : ℝ) (side_square : ℝ) (width : ℝ) (n_triangles : ℕ) 
  (triangle_right : height = 2)
  (fit_inside_square : side_square = 2)
  (number_triangles : n_triangles = 2) :
  width = 2 :=
sorry

end right_triangle_width_l50_50758


namespace part1_solution_set_part2_range_of_m_l50_50510

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) * abs (x - 3)

theorem part1_solution_set :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} :=
sorry

theorem part2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≤ abs (3 * m - 2)) → m ∈ Set.Iic (-1) ∪ Set.Ici (7 / 3) :=
sorry

end part1_solution_set_part2_range_of_m_l50_50510


namespace maximum_value_x2_add_3xy_add_y2_l50_50064

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2_l50_50064


namespace adam_age_l50_50643

variable (E A : ℕ)

namespace AgeProof

theorem adam_age (h1 : A = E - 5) (h2 : E + 1 = 3 * (A - 4)) : A = 9 :=
by
  sorry
end AgeProof

end adam_age_l50_50643


namespace cos_alpha_given_tan_alpha_and_quadrant_l50_50699

theorem cos_alpha_given_tan_alpha_and_quadrant 
  (α : ℝ) 
  (h1 : Real.tan α = -1/3)
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -3*Real.sqrt 10 / 10 :=
by
  sorry

end cos_alpha_given_tan_alpha_and_quadrant_l50_50699


namespace square_area_l50_50864

theorem square_area (x : ℝ) (h1 : x = 60) : x^2 = 1200 :=
by
  sorry

end square_area_l50_50864


namespace find_unit_prices_l50_50363

-- Define the prices of brush and chess set
variables (x y : ℝ)

-- Condition 1: Buying 5 brushes and 12 chess sets costs 315 yuan
def condition1 : Prop := 5 * x + 12 * y = 315

-- Condition 2: Buying 8 brushes and 6 chess sets costs 240 yuan
def condition2 : Prop := 8 * x + 6 * y = 240

-- Prove that the unit price of each brush is 15 yuan and each chess set is 20 yuan
theorem find_unit_prices (hx : condition1 x y) (hy : condition2 x y) :
  x = 15 ∧ y = 20 := 
sorry

end find_unit_prices_l50_50363


namespace books_count_is_8_l50_50403

theorem books_count_is_8
  (k a p_k p_a : ℕ)
  (h1 : k = a + 6)
  (h2 : k * p_k = 1056)
  (h3 : a * p_a = 56)
  (h4 : p_k > p_a + 100) :
  k = 8 := 
sorry

end books_count_is_8_l50_50403


namespace probability_scrapped_l50_50746

variable (P_A P_B_given_not_A : ℝ)
variable (prob_scrapped : ℝ)

def fail_first_inspection (P_A : ℝ) := 1 - P_A
def fail_second_inspection_given_fails_first (P_B_given_not_A : ℝ) := 1 - P_B_given_not_A

theorem probability_scrapped (h1 : P_A = 0.8) (h2 : P_B_given_not_A = 0.9) (h3 : prob_scrapped = fail_first_inspection P_A * fail_second_inspection_given_fails_first P_B_given_not_A) :
  prob_scrapped = 0.02 := by
  sorry

end probability_scrapped_l50_50746


namespace prime_solution_l50_50819

theorem prime_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 5 * p + 3 * q = 91) : p = 17 ∧ q = 2 :=
by
  sorry

end prime_solution_l50_50819


namespace race_length_l50_50369

theorem race_length (members : ℕ) (member_distance : ℕ) (ralph_multiplier : ℕ) 
    (h1 : members = 4) (h2 : member_distance = 3) (h3 : ralph_multiplier = 2) : 
    members * member_distance + ralph_multiplier * member_distance = 18 :=
by
  -- Start the proof with sorry to denote missing steps.
  sorry

end race_length_l50_50369


namespace r_squared_is_one_l50_50692

theorem r_squared_is_one (h : ∀ (x : ℝ), ∃ (y : ℝ), ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ y = m * x + b) : R_squared = 1 :=
sorry

end r_squared_is_one_l50_50692


namespace sum_of_consecutive_numbers_with_lcm_168_l50_50077

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l50_50077


namespace sin_cos_15_eq_quarter_l50_50773

theorem sin_cos_15_eq_quarter :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
by 
  sorry

end sin_cos_15_eq_quarter_l50_50773
