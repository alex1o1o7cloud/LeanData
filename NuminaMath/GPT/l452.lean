import Mathlib

namespace NUMINAMATH_GPT_probability_of_C_l452_45253

theorem probability_of_C (P_A P_B P_C P_D P_E : ℚ)
  (hA : P_A = 2/5)
  (hB : P_B = 1/5)
  (hCD : P_C = P_D)
  (hE : P_E = 2 * P_C)
  (h_total : P_A + P_B + P_C + P_D + P_E = 1) : P_C = 1/10 :=
by
  -- To prove this theorem, you will use the conditions provided in the hypotheses.
  -- Here's how you start the proof:
  sorry

end NUMINAMATH_GPT_probability_of_C_l452_45253


namespace NUMINAMATH_GPT_f_f_0_eq_zero_number_of_zeros_l452_45214

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x > 0 then 1 - 1/x else (a - 1) * x + 1

theorem f_f_0_eq_zero (a : ℝ) : f a (f a 0) = 0 := by
  sorry

theorem number_of_zeros (a : ℝ) : 
  if a = 1 then ∃! x, f a x = 0 else
  if a > 1 then ∃! x1, ∃! x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 else
  ∃! x, f a x = 0 := by sorry

end NUMINAMATH_GPT_f_f_0_eq_zero_number_of_zeros_l452_45214


namespace NUMINAMATH_GPT_abs_value_solutions_l452_45299

theorem abs_value_solutions (x : ℝ) : abs x = 6.5 ↔ x = 6.5 ∨ x = -6.5 :=
by
  sorry

end NUMINAMATH_GPT_abs_value_solutions_l452_45299


namespace NUMINAMATH_GPT_solution_set_characterization_l452_45226

noncomputable def satisfies_inequality (x : ℝ) : Bool :=
  (3 / (x + 2) + 4 / (x + 6)) > 1

theorem solution_set_characterization :
  ∀ x : ℝ, (satisfies_inequality x) ↔ (x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2) :=
by
  intro x
  unfold satisfies_inequality
  -- here we would provide the proof
  sorry

end NUMINAMATH_GPT_solution_set_characterization_l452_45226


namespace NUMINAMATH_GPT_smallest_number_divisible_by_conditions_l452_45271

theorem smallest_number_divisible_by_conditions:
  ∃ n : ℕ, (∀ d ∈ [8, 12, 22, 24], d ∣ (n - 12)) ∧ (n = 252) :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_conditions_l452_45271


namespace NUMINAMATH_GPT_average_speed_train_l452_45259

theorem average_speed_train (x : ℝ) (h1 : x ≠ 0) :
  let t1 := x / 40
  let t2 := 2 * x / 20
  let t3 := 3 * x / 60
  let total_time := t1 + t2 + t3
  let total_distance := 6 * x
  let average_speed := total_distance / total_time
  average_speed = 240 / 7 := by
  sorry

end NUMINAMATH_GPT_average_speed_train_l452_45259


namespace NUMINAMATH_GPT_equivalent_multipliers_l452_45217

variable (a b c : ℝ)

theorem equivalent_multipliers :
  (a - 0.07 * a + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c :=
sorry

end NUMINAMATH_GPT_equivalent_multipliers_l452_45217


namespace NUMINAMATH_GPT_percent_of_y_l452_45229

theorem percent_of_y (y : ℝ) (hy : y > 0) : (6 * y / 20) + (3 * y / 10) = 0.6 * y :=
by
  sorry

end NUMINAMATH_GPT_percent_of_y_l452_45229


namespace NUMINAMATH_GPT_find_a_squared_plus_b_squared_and_ab_l452_45279

theorem find_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 7)
  (h2 : (a - b) ^ 2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_squared_plus_b_squared_and_ab_l452_45279


namespace NUMINAMATH_GPT_value_of_k_l452_45233

theorem value_of_k (k : ℤ) : 
  (∃ a b : ℤ, x^2 + k * x + 81 = a^2 * x^2 + 2 * a * b * x + b^2) → (k = 18 ∨ k = -18) :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l452_45233


namespace NUMINAMATH_GPT_number_of_zeros_l452_45280

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + a + 1
noncomputable def g (b : ℝ) (x : ℝ) := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

theorem number_of_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (x : ℝ), g b (f a x) = 0 := sorry

end NUMINAMATH_GPT_number_of_zeros_l452_45280


namespace NUMINAMATH_GPT_unique_positive_real_solution_l452_45277

def f (x : ℝ) := x^11 + 5 * x^10 + 20 * x^9 + 1000 * x^8 - 800 * x^7

theorem unique_positive_real_solution :
  ∃! (x : ℝ), 0 < x ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_unique_positive_real_solution_l452_45277


namespace NUMINAMATH_GPT_parallelogram_area_twice_quadrilateral_area_l452_45264

theorem parallelogram_area_twice_quadrilateral_area (S : ℝ) (LMNP_area : ℝ) 
  (h : LMNP_area = 2 * S) : LMNP_area = 2 * S := 
by {
  sorry
}

end NUMINAMATH_GPT_parallelogram_area_twice_quadrilateral_area_l452_45264


namespace NUMINAMATH_GPT_sqrt_neg9_squared_l452_45209

theorem sqrt_neg9_squared : Real.sqrt ((-9: ℝ)^2) = 9 := by
  sorry

end NUMINAMATH_GPT_sqrt_neg9_squared_l452_45209


namespace NUMINAMATH_GPT_problem_statement_l452_45222

def f (x : ℤ) : ℤ := x^2 + 3
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem_statement : f (g 4) - g (f 4) = 129 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l452_45222


namespace NUMINAMATH_GPT_share_of_C_l452_45208

variable (A B C x : ℝ)

theorem share_of_C (hA : A = (2/3) * B) 
(hB : B = (1/4) * C) 
(hTotal : A + B + C = 595) 
(hC : C = x) : x = 420 :=
by
  -- Proof will follow here
  sorry

end NUMINAMATH_GPT_share_of_C_l452_45208


namespace NUMINAMATH_GPT_find_parallel_line_l452_45268

/-- 
Given a line l with equation 3x - 2y + 1 = 0 and a point A(1,1).
Find the equation of a line that passes through A and is parallel to l.
-/
theorem find_parallel_line (a b c : ℝ) (p_x p_y : ℝ) 
    (h₁ : 3 * p_x - 2 * p_y + c = 0) 
    (h₂ : p_x = 1 ∧ p_y = 1)
    (h₃ : a = 3 ∧ b = -2) :
    3 * x - 2 * y - 1 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_find_parallel_line_l452_45268


namespace NUMINAMATH_GPT_distance_to_center_square_l452_45205

theorem distance_to_center_square (x y : ℝ) (h : x*x + y*y = 72) (h1 : x*x + (y + 8)*(y + 8) = 72) (h2 : (x + 4)*(x + 4) + y*y = 72) :
  x*x + y*y = 9 ∨ x*x + y*y = 185 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_center_square_l452_45205


namespace NUMINAMATH_GPT_nonoverlapping_unit_squares_in_figure_50_l452_45297

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem nonoverlapping_unit_squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 :=
by
  sorry

end NUMINAMATH_GPT_nonoverlapping_unit_squares_in_figure_50_l452_45297


namespace NUMINAMATH_GPT_simple_interest_rate_l452_45291

theorem simple_interest_rate :
  ∀ (P R : ℝ), 
  (R * 25 / 100 = 1) → 
  R = 4 := 
by
  intros P R h
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l452_45291


namespace NUMINAMATH_GPT_division_of_decimals_l452_45292

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := 
sorry

end NUMINAMATH_GPT_division_of_decimals_l452_45292


namespace NUMINAMATH_GPT_dalton_needs_more_money_l452_45256

theorem dalton_needs_more_money :
  let jump_rope_cost := 9
  let board_game_cost := 15
  let playground_ball_cost := 5
  let puzzle_cost := 8
  let saved_allowance := 7
  let uncle_gift := 14
  let total_cost := jump_rope_cost + board_game_cost + playground_ball_cost + puzzle_cost
  let total_money := saved_allowance + uncle_gift
  (total_cost - total_money) = 16 :=
by
  sorry

end NUMINAMATH_GPT_dalton_needs_more_money_l452_45256


namespace NUMINAMATH_GPT_club_last_names_l452_45294

theorem club_last_names :
  ∃ A B C D E F : ℕ,
    A + B + C + D + E + F = 21 ∧
    A^2 + B^2 + C^2 + D^2 + E^2 + F^2 = 91 :=
by {
  sorry
}

end NUMINAMATH_GPT_club_last_names_l452_45294


namespace NUMINAMATH_GPT_shopkeeper_discount_and_selling_price_l452_45238

theorem shopkeeper_discount_and_selling_price :
  let CP := 100
  let MP := CP + 0.5 * CP
  let SP := CP + 0.15 * CP
  let Discount := (MP - SP) / MP * 100
  Discount = 23.33 ∧ SP = 115 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_discount_and_selling_price_l452_45238


namespace NUMINAMATH_GPT_final_amounts_total_l452_45231

variable {Ben_initial Tom_initial Max_initial: ℕ}
variable {Ben_final Tom_final Max_final: ℕ}

theorem final_amounts_total (h1: Ben_initial = 48) 
                           (h2: Max_initial = 48) 
                           (h3: Ben_final = ((Ben_initial - Tom_initial - Max_initial) * 3 / 2))
                           (h4: Max_final = ((Max_initial * 3 / 2))) 
                           (h5: Tom_final = (Tom_initial * 2 - ((Ben_initial - Tom_initial - Max_initial) / 2) - 48))
                           (h6: Max_final = 48) :
  Ben_final + Tom_final + Max_final = 144 := 
by 
  sorry

end NUMINAMATH_GPT_final_amounts_total_l452_45231


namespace NUMINAMATH_GPT_chlorine_moles_l452_45249

theorem chlorine_moles (methane_used chlorine_used chloromethane_formed : ℕ)
  (h_combined_methane : methane_used = 3)
  (h_formed_chloromethane : chloromethane_formed = 3)
  (balanced_eq : methane_used = chloromethane_formed) :
  chlorine_used = 3 :=
by
  have h : chlorine_used = methane_used := by sorry
  rw [h_combined_methane] at h
  exact h

end NUMINAMATH_GPT_chlorine_moles_l452_45249


namespace NUMINAMATH_GPT_union_complement_correct_l452_45263

open Set

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem union_complement_correct : A ∪ (compl B) = Ioo (-1 : ℝ) 3 ∪ {3} := by
  sorry

end NUMINAMATH_GPT_union_complement_correct_l452_45263


namespace NUMINAMATH_GPT_find_smallest_x_satisfying_condition_l452_45261

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_x_satisfying_condition_l452_45261


namespace NUMINAMATH_GPT_zs_share_in_profit_l452_45230

noncomputable def calculateProfitShare (x_investment y_investment z_investment z_months total_profit : ℚ) : ℚ :=
  let x_invest_months := x_investment * 12
  let y_invest_months := y_investment * 12
  let z_invest_months := z_investment * z_months
  let total_invest_months := x_invest_months + y_invest_months + z_invest_months
  let z_share := z_invest_months / total_invest_months
  total_profit * z_share

theorem zs_share_in_profit :
  calculateProfitShare 36000 42000 48000 8 14190 = 2580 :=
by
  sorry

end NUMINAMATH_GPT_zs_share_in_profit_l452_45230


namespace NUMINAMATH_GPT_major_premise_is_false_l452_45224

-- Define the major premise
def major_premise (a : ℝ) : Prop := a^2 > 0

-- Define the minor premise
def minor_premise (a : ℝ) := true

-- Define the conclusion based on the premises
def conclusion (a : ℝ) : Prop := a^2 > 0

-- Show that the major premise is false by finding a counterexample
theorem major_premise_is_false : ¬ ∀ a : ℝ, major_premise a := by
  sorry

end NUMINAMATH_GPT_major_premise_is_false_l452_45224


namespace NUMINAMATH_GPT_min_abs_sum_l452_45213

theorem min_abs_sum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

end NUMINAMATH_GPT_min_abs_sum_l452_45213


namespace NUMINAMATH_GPT_range_of_a_plus_abs_b_l452_45239

theorem range_of_a_plus_abs_b (a b : ℝ)
  (h1 : -1 ≤ a) (h2 : a ≤ 3)
  (h3 : -5 < b) (h4 : b < 3) :
  -1 ≤ a + |b| ∧ a + |b| < 8 := by
sorry

end NUMINAMATH_GPT_range_of_a_plus_abs_b_l452_45239


namespace NUMINAMATH_GPT_pizza_topping_slices_l452_45290

theorem pizza_topping_slices 
  (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ)
  (pepperoni_slices_has_at_least_one_topping : pepperoni_slices = 8)
  (mushroom_slices_has_at_least_one_topping : mushroom_slices = 12)
  (olive_slices_has_at_least_one_topping : olive_slices = 14)
  (total_slices_has_one_topping : total_slices = 16)
  (slices_with_at_least_one_topping : 8 + 12 + 14 - 2 * x = 16) :
  x = 9 :=
by
  sorry

end NUMINAMATH_GPT_pizza_topping_slices_l452_45290


namespace NUMINAMATH_GPT_tim_kittens_l452_45200

theorem tim_kittens (initial_kittens : ℕ) (given_to_jessica_fraction : ℕ) (saras_kittens : ℕ) (adopted_fraction : ℕ) 
  (h_initial : initial_kittens = 12)
  (h_fraction_to_jessica : given_to_jessica_fraction = 3)
  (h_saras_kittens : saras_kittens = 14)
  (h_adopted_fraction : adopted_fraction = 2) :
  let kittens_after_jessica := initial_kittens - initial_kittens / given_to_jessica_fraction
  let total_kittens_after_sara := kittens_after_jessica + saras_kittens
  let adopted_kittens := saras_kittens / adopted_fraction
  let final_kittens := total_kittens_after_sara - adopted_kittens
  final_kittens = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_tim_kittens_l452_45200


namespace NUMINAMATH_GPT_number_of_males_choosing_malt_l452_45258

-- Definitions of conditions as provided in the problem
def total_males : Nat := 10
def total_females : Nat := 16

def total_cheerleaders : Nat := total_males + total_females

def females_choosing_malt : Nat := 8
def females_choosing_coke : Nat := total_females - females_choosing_malt

noncomputable def cheerleaders_choosing_malt (M_males : Nat) : Nat :=
  females_choosing_malt + M_males

noncomputable def cheerleaders_choosing_coke (M_males : Nat) : Nat :=
  females_choosing_coke + (total_males - M_males)

theorem number_of_males_choosing_malt : ∃ (M_males : Nat), 
  cheerleaders_choosing_malt M_males = 2 * cheerleaders_choosing_coke M_males ∧
  cheerleaders_choosing_malt M_males + cheerleaders_choosing_coke M_males = total_cheerleaders ∧
  M_males = 9 := 
by
  sorry

end NUMINAMATH_GPT_number_of_males_choosing_malt_l452_45258


namespace NUMINAMATH_GPT_annual_decrease_rate_l452_45278

theorem annual_decrease_rate (P : ℕ) (P2 : ℕ) (r : ℝ) : 
  (P = 10000) → (P2 = 8100) → (P2 = P * (1 - r / 100)^2) → (r = 10) :=
by
  intro hP hP2 hEq
  sorry

end NUMINAMATH_GPT_annual_decrease_rate_l452_45278


namespace NUMINAMATH_GPT_jamestown_theme_parks_l452_45298

theorem jamestown_theme_parks (J : ℕ) (Venice := J + 25) (MarinaDelRay := J + 50) (total := J + Venice + MarinaDelRay) (h : total = 135) : J = 20 :=
by
  -- proof step to be done here
  sorry

end NUMINAMATH_GPT_jamestown_theme_parks_l452_45298


namespace NUMINAMATH_GPT_sum_of_15_terms_l452_45223

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_15_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3) + (a 4 + a 5 + a 6) + (a 7 + a 8 + a 9) +
  (a 10 + a 11 + a 12) + (a 13 + a 14 + a 15) = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_15_terms_l452_45223


namespace NUMINAMATH_GPT_sequence_divisibility_condition_l452_45244

theorem sequence_divisibility_condition (t a b x1 : ℕ) (x : ℕ → ℕ)
  (h1 : a = 1) (h2 : b = t) (h3 : x1 = t) (h4 : x 1 = x1)
  (h5 : ∀ n, n ≥ 2 → x n = a * x (n - 1) + b) :
  (∀ m n, m ∣ n → x m ∣ x n) ↔ (a = 1 ∧ b = t ∧ x1 = t) := sorry

end NUMINAMATH_GPT_sequence_divisibility_condition_l452_45244


namespace NUMINAMATH_GPT_german_team_goals_l452_45219

theorem german_team_goals :
  ∃ (x : ℕ), 10 < x ∧ x < 17 ∧ 11 < x ∧ x < 18 ∧ x % 2 = 1 ∧ 
             ((10 < x ∧ x < 17) ∧ (11 < x ∧ x < 18) ↔ (x % 2 = 0)) :=
sorry

end NUMINAMATH_GPT_german_team_goals_l452_45219


namespace NUMINAMATH_GPT_percent_of_x_l452_45273

variable {x y z : ℝ}

-- Define the given conditions
def cond1 (z y : ℝ) : Prop := 0.45 * z = 0.9 * y
def cond2 (z x : ℝ) : Prop := z = 1.5 * x

-- State the theorem to prove
theorem percent_of_x (h1 : cond1 z y) (h2 : cond2 z x) : y = 0.75 * x :=
sorry

end NUMINAMATH_GPT_percent_of_x_l452_45273


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l452_45201

namespace SimplificationProof

variable (x : ℝ)

theorem simplify_and_evaluate_expr (h : x = Real.sqrt 5 - 1) :
  ((x / (x - 1) - 1) / ((x ^ 2 - 1) / (x ^ 2 - 2 * x + 1))) = Real.sqrt 5 / 5 :=
by
  sorry

end SimplificationProof

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l452_45201


namespace NUMINAMATH_GPT_willie_bananas_remain_same_l452_45225

variable (Willie_bananas Charles_bananas Charles_loses : ℕ)

theorem willie_bananas_remain_same (h_willie : Willie_bananas = 48) (h_charles_initial : Charles_bananas = 14) (h_charles_loses : Charles_loses = 35) :
  Willie_bananas = 48 :=
by
  sorry

end NUMINAMATH_GPT_willie_bananas_remain_same_l452_45225


namespace NUMINAMATH_GPT_mohan_cookies_l452_45241

theorem mohan_cookies :
  ∃ a : ℕ, 
    a % 4 = 3 ∧
    a % 5 = 2 ∧
    a % 7 = 4 ∧
    a = 67 :=
by
  -- The proof will be written here.
  sorry

end NUMINAMATH_GPT_mohan_cookies_l452_45241


namespace NUMINAMATH_GPT_cos_double_angle_l452_45282

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7 / 25 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_l452_45282


namespace NUMINAMATH_GPT_benny_number_of_kids_l452_45274

-- Define the conditions
def benny_has_dollars (d: ℕ): Prop := d = 360
def cost_per_apple (c: ℕ): Prop := c = 4
def apples_shared (num_kids num_apples: ℕ): Prop := num_apples = 5 * num_kids

-- State the main theorem
theorem benny_number_of_kids : 
  ∀ (d c k a : ℕ), benny_has_dollars d → cost_per_apple c → apples_shared k a → k = 18 :=
by
  intros d c k a hd hc ha
  -- The goal is to prove k = 18; use the provided conditions
  sorry

end NUMINAMATH_GPT_benny_number_of_kids_l452_45274


namespace NUMINAMATH_GPT_num_integer_solutions_prime_l452_45255

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 ∧ m < n → n % m ≠ 0

def integer_solutions : List ℤ := [-1, 3]

theorem num_integer_solutions_prime :
  (∀ x ∈ integer_solutions, is_prime (|15 * x^2 - 32 * x - 28|)) ∧ (integer_solutions.length = 2) :=
by
  sorry

end NUMINAMATH_GPT_num_integer_solutions_prime_l452_45255


namespace NUMINAMATH_GPT_Janka_bottle_caps_l452_45240

theorem Janka_bottle_caps (n : ℕ) :
  (∃ k1 : ℕ, n = 3 * k1) ∧ (∃ k2 : ℕ, n = 4 * k2) ↔ n = 12 ∨ n = 24 :=
by
  sorry

end NUMINAMATH_GPT_Janka_bottle_caps_l452_45240


namespace NUMINAMATH_GPT_bowling_ball_weight_l452_45212

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l452_45212


namespace NUMINAMATH_GPT_distance_range_l452_45276

variable (x : ℝ)
variable (starting_fare : ℝ := 6) -- fare in yuan for up to 2 kilometers
variable (surcharge : ℝ := 1) -- yuan surcharge per ride
variable (additional_fare : ℝ := 1) -- fare for every additional 0.5 kilometers
variable (additional_distance : ℝ := 0.5) -- distance in kilometers for every additional fare

theorem distance_range (h_total_fare : 9 = starting_fare + (x - 2) / additional_distance * additional_fare + surcharge) :
  2.5 < x ∧ x ≤ 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distance_range_l452_45276


namespace NUMINAMATH_GPT_range_of_c_l452_45269

theorem range_of_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a + b = a * b) (habc : a + b + c = a * b * c) : 1 < c ∧ c ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_c_l452_45269


namespace NUMINAMATH_GPT_find_certain_number_l452_45203

def certain_number (x : ℚ) : Prop := 5 * 1.6 - (1.4 * x) / 1.3 = 4

theorem find_certain_number : certain_number (-(26/7)) :=
by 
  simp [certain_number]
  sorry

end NUMINAMATH_GPT_find_certain_number_l452_45203


namespace NUMINAMATH_GPT_segment_lengths_l452_45283

noncomputable def radius : ℝ := 5
noncomputable def diameter : ℝ := 2 * radius
noncomputable def chord_length : ℝ := 8

-- The lengths of the segments AK and KB
theorem segment_lengths (x : ℝ) (y : ℝ) 
  (hx : 0 < x ∧ x < diameter) 
  (hy : 0 < y ∧ y < diameter) 
  (h1 : x + y = diameter) 
  (h2 : x * y = (diameter^2) / 4 - 16 / 4) : 
  x = 2.5 ∧ y = 7.5 := 
sorry

end NUMINAMATH_GPT_segment_lengths_l452_45283


namespace NUMINAMATH_GPT_kolya_or_leva_wins_l452_45275

-- Definitions for segment lengths
variables (k l : ℝ)

-- Definition of the condition when Kolya wins
def kolya_wins (k l : ℝ) : Prop :=
  k > l

-- Definition of the condition when Leva wins
def leva_wins (k l : ℝ) : Prop :=
  k ≤ l

-- Theorem statement for the proof problem
theorem kolya_or_leva_wins (k l : ℝ) : kolya_wins k l ∨ leva_wins k l :=
sorry

end NUMINAMATH_GPT_kolya_or_leva_wins_l452_45275


namespace NUMINAMATH_GPT_num_of_original_numbers_l452_45247

theorem num_of_original_numbers
    (n : ℕ) 
    (S : ℤ) 
    (incorrect_avg correct_avg : ℤ)
    (incorrect_num correct_num : ℤ)
    (h1 : incorrect_avg = 46)
    (h2 : correct_avg = 51)
    (h3 : incorrect_num = 25)
    (h4 : correct_num = 75)
    (h5 : S + correct_num = correct_avg * n)
    (h6 : S + incorrect_num = incorrect_avg * n) :
  n = 10 := by
  sorry

end NUMINAMATH_GPT_num_of_original_numbers_l452_45247


namespace NUMINAMATH_GPT_a11_is_1_l452_45254

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Condition 1: The sum of the first n terms S_n satisfies S_n + S_m = S_{n+m}
axiom sum_condition (n m : ℕ) : S n + S m = S (n + m)

-- Condition 2: a_1 = 1
axiom a1_condition : a 1 = 1

-- Question: prove a_{11} = 1
theorem a11_is_1 : a 11 = 1 :=
sorry


end NUMINAMATH_GPT_a11_is_1_l452_45254


namespace NUMINAMATH_GPT_side_length_of_regular_pentagon_l452_45215

theorem side_length_of_regular_pentagon (perimeter : ℝ) (number_of_sides : ℕ) (h1 : perimeter = 23.4) (h2 : number_of_sides = 5) : 
  perimeter / number_of_sides = 4.68 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_regular_pentagon_l452_45215


namespace NUMINAMATH_GPT_smallest_number_greater_300_with_remainder_24_l452_45289

theorem smallest_number_greater_300_with_remainder_24 :
  ∃ n : ℕ, n > 300 ∧ n % 25 = 24 ∧ ∀ k : ℕ, k > 300 ∧ k % 25 = 24 → n ≤ k :=
sorry

end NUMINAMATH_GPT_smallest_number_greater_300_with_remainder_24_l452_45289


namespace NUMINAMATH_GPT_integral_f_equals_neg_third_l452_45246

def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * c

theorem integral_f_equals_neg_third :
  (∫ x in (0 : ℝ)..(1 : ℝ), f x (∫ t in (0 : ℝ)..(1 : ℝ), f t (∫ t in (0 : ℝ)..(1 : ℝ), f t 0))) = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_integral_f_equals_neg_third_l452_45246


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l452_45245

theorem sum_of_first_15_terms (S : ℕ → ℕ) (h1 : S 5 = 48) (h2 : S 10 = 60) : S 15 = 72 :=
sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l452_45245


namespace NUMINAMATH_GPT_rhombus_other_diagonal_l452_45250

theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
  (h1 : d1 = 50) 
  (h2 : area = 625) 
  (h3 : area = (d1 * d2) / 2) : 
  d2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_other_diagonal_l452_45250


namespace NUMINAMATH_GPT_less_than_reciprocal_l452_45227

theorem less_than_reciprocal (n : ℚ) : 
  n = -3 ∨ n = 3/4 ↔ (n = -1/2 → n >= 1/(-1/2)) ∧
                           (n = -3 → n < 1/(-3)) ∧
                           (n = 3/4 → n < 1/(3/4)) ∧
                           (n = 3 → n > 1/3) ∧
                           (n = 0 → false) := sorry

end NUMINAMATH_GPT_less_than_reciprocal_l452_45227


namespace NUMINAMATH_GPT_sum_ratio_is_nine_l452_45248

open Nat

-- Predicate to define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

axiom a : ℕ → ℝ -- The arithmetic sequence
axiom h_arith : is_arithmetic_sequence a
axiom a5_eq_5a3 : a 4 = 5 * a 2

-- Statement of the problem
theorem sum_ratio_is_nine : S 9 a / S 5 a = 9 :=
sorry

end NUMINAMATH_GPT_sum_ratio_is_nine_l452_45248


namespace NUMINAMATH_GPT_nested_sqrt_eq_two_l452_45218

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by {
    -- Proof skipped
    sorry
}

end NUMINAMATH_GPT_nested_sqrt_eq_two_l452_45218


namespace NUMINAMATH_GPT_clean_house_time_l452_45236

theorem clean_house_time (B A: ℝ) (h1: A = 1/12) (h2: B + A = 1/4):
  (B + 2 * A) = 1/3 → 1 / (B + 2 * A) = 3 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_clean_house_time_l452_45236


namespace NUMINAMATH_GPT_monotonic_increase_interval_l452_45281

noncomputable def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem monotonic_increase_interval 
    (ω : ℝ)
    (hω : 0 < ω)
    (hperiod : Real.pi = 2 * Real.pi / ω) :
    ∀ k : ℤ, ∃ I : Set ℝ, I = interval_of_monotonic_increase k := 
by
  sorry

end NUMINAMATH_GPT_monotonic_increase_interval_l452_45281


namespace NUMINAMATH_GPT_books_before_addition_l452_45242

-- Let b be the initial number of books on the shelf
variable (b : ℕ)

theorem books_before_addition (h : b + 10 = 19) : b = 9 := by
  sorry

end NUMINAMATH_GPT_books_before_addition_l452_45242


namespace NUMINAMATH_GPT_no_nat_n_divisible_by_169_l452_45221

theorem no_nat_n_divisible_by_169 (n : ℕ) : ¬ ∃ k : ℕ, n^2 + 5 * n + 16 = 169 * k :=
sorry

end NUMINAMATH_GPT_no_nat_n_divisible_by_169_l452_45221


namespace NUMINAMATH_GPT_mrs_hilt_current_rocks_l452_45237

-- Definitions based on conditions
def total_rocks_needed : ℕ := 125
def more_rocks_needed : ℕ := 61

-- Lean statement proving the required amount of currently held rocks
theorem mrs_hilt_current_rocks : (total_rocks_needed - more_rocks_needed) = 64 :=
by
  -- proof will be here
  sorry

end NUMINAMATH_GPT_mrs_hilt_current_rocks_l452_45237


namespace NUMINAMATH_GPT_find_strawberry_jelly_amount_l452_45220

noncomputable def strawberry_jelly (t b : ℕ) : ℕ := t - b

theorem find_strawberry_jelly_amount (h₁ : 6310 = 4518 + s) : s = 1792 := by
  sorry

end NUMINAMATH_GPT_find_strawberry_jelly_amount_l452_45220


namespace NUMINAMATH_GPT_probability_four_heads_l452_45270

-- Definitions for use in the conditions
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def biased_coin (h : ℚ) (n k : ℕ) : ℚ :=
  binomial_coefficient n k * (h ^ k) * ((1 - h) ^ (n - k))

-- Condition: probability of getting heads exactly twice is equal to getting heads exactly three times.
def condition (h : ℚ) : Prop :=
  biased_coin h 5 2 = biased_coin h 5 3

-- Theorem to be proven: probability of getting heads exactly four times out of five is 5/32.
theorem probability_four_heads (h : ℚ) (cond : condition h) : biased_coin h 5 4 = 5 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_four_heads_l452_45270


namespace NUMINAMATH_GPT_length_of_plot_57_meters_l452_45286

section RectangleProblem

variable (b : ℝ) -- breadth of the plot
variable (l : ℝ) -- length of the plot
variable (cost_per_meter : ℝ) -- cost per meter
variable (total_cost : ℝ) -- total cost

-- Given conditions
def length_eq_breadth_plus_14 (b l : ℝ) : Prop := l = b + 14
def cost_eq_perimeter_cost_per_meter (cost_per_meter total_cost perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

-- Definition of perimeter
def perimeter (b l : ℝ) : ℝ := 2 * l + 2 * b

-- Problem statement
theorem length_of_plot_57_meters
  (h1 : length_eq_breadth_plus_14 b l)
  (h2 : cost_eq_perimeter_cost_per_meter cost_per_meter total_cost (perimeter b l))
  (h3 : cost_per_meter = 26.50)
  (h4 : total_cost = 5300) :
  l = 57 :=
by
  sorry

end RectangleProblem

end NUMINAMATH_GPT_length_of_plot_57_meters_l452_45286


namespace NUMINAMATH_GPT_pyramid_transport_volume_l452_45267

-- Define the conditions of the problem
def pyramid_height : ℝ := 15
def pyramid_base_side_length : ℝ := 8
def box_length : ℝ := 10
def box_width : ℝ := 10
def box_height : ℝ := 15

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- State the theorem
theorem pyramid_transport_volume : box_volume = 1500 := by
  sorry

end NUMINAMATH_GPT_pyramid_transport_volume_l452_45267


namespace NUMINAMATH_GPT_inequality_holds_iff_m_lt_2_l452_45262

theorem inequality_holds_iff_m_lt_2 :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → x^2 - m * x + m > 0) ↔ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_iff_m_lt_2_l452_45262


namespace NUMINAMATH_GPT_quadratic_vertex_l452_45216

noncomputable def quadratic_vertex_max (c d : ℝ) (h : -x^2 + c * x + d ≤ 0) : (ℝ × ℝ) :=
sorry

theorem quadratic_vertex 
  (c d : ℝ)
  (h : -x^2 + c * x + d ≤ 0)
  (root1 root2 : ℝ)
  (h_roots : root1 = -5 ∧ root2 = 3) :
  quadratic_vertex_max c d h = (4, 1) ∧ (∀ x: ℝ, (x - 4)^2 ≤ 1) :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_l452_45216


namespace NUMINAMATH_GPT_min_abs_x1_x2_l452_45235

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

theorem min_abs_x1_x2 
  (a x1 x2 : ℝ)
  (h_symmetry : ∃ c : ℝ, c = -Real.pi / 6 ∧ (∀ x, f a (x - c) = f a x))
  (h_product : f a x1 * f a x2 = -16) :
  ∃ m : ℝ, m = abs (x1 + x2) ∧ m = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_GPT_min_abs_x1_x2_l452_45235


namespace NUMINAMATH_GPT_morgan_total_pens_l452_45243

def initial_red_pens : Nat := 65
def initial_blue_pens : Nat := 45
def initial_black_pens : Nat := 58
def initial_green_pens : Nat := 36
def initial_purple_pens : Nat := 27

def red_pens_given_away : Nat := 15
def blue_pens_given_away : Nat := 20
def green_pens_given_away : Nat := 10

def black_pens_bought : Nat := 12
def purple_pens_bought : Nat := 5

def final_red_pens : Nat := initial_red_pens - red_pens_given_away
def final_blue_pens : Nat := initial_blue_pens - blue_pens_given_away
def final_black_pens : Nat := initial_black_pens + black_pens_bought
def final_green_pens : Nat := initial_green_pens - green_pens_given_away
def final_purple_pens : Nat := initial_purple_pens + purple_pens_bought

def total_pens : Nat := final_red_pens + final_blue_pens + final_black_pens + final_green_pens + final_purple_pens

theorem morgan_total_pens : total_pens = 203 := 
by
  -- final_red_pens = 50
  -- final_blue_pens = 25
  -- final_black_pens = 70
  -- final_green_pens = 26
  -- final_purple_pens = 32
  -- Therefore, total_pens = 203
  sorry

end NUMINAMATH_GPT_morgan_total_pens_l452_45243


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l452_45265

theorem perfect_square_trinomial_m (m : ℤ) : (∀ x : ℤ, ∃ k : ℤ, x^2 + 2*m*x + 9 = (x + k)^2) ↔ m = 3 ∨ m = -3 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l452_45265


namespace NUMINAMATH_GPT_fraction_one_bedroom_apartments_l452_45287

theorem fraction_one_bedroom_apartments :
  ∃ x : ℝ, (x + 0.33 = 0.5) ∧ x = 0.17 :=
by
  sorry

end NUMINAMATH_GPT_fraction_one_bedroom_apartments_l452_45287


namespace NUMINAMATH_GPT_problem_solution_l452_45251

theorem problem_solution
  (a b c : ℕ)
  (h_pos_a : 0 < a ∧ a ≤ 10)
  (h_pos_b : 0 < b ∧ b ≤ 10)
  (h_pos_c : 0 < c ∧ c ≤ 10)
  (h1 : abc % 11 = 2)
  (h2 : 7 * c % 11 = 3)
  (h3 : 8 * b % 11 = 4 + b % 11) : 
  (a + b + c) % 11 = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_solution_l452_45251


namespace NUMINAMATH_GPT_algebraic_expression_value_l452_45207

variable {R : Type} [CommRing R]

theorem algebraic_expression_value (m n : R) (h1 : m - n = -2) (h2 : m * n = 3) :
  -m^3 * n + 2 * m^2 * n^2 - m * n^3 = -12 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l452_45207


namespace NUMINAMATH_GPT_det_A_zero_l452_45206

theorem det_A_zero
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : a11 = Real.sin (x1 - y1)) (h2 : a12 = Real.sin (x1 - y2)) (h3 : a13 = Real.sin (x1 - y3))
  (h4 : a21 = Real.sin (x2 - y1)) (h5 : a22 = Real.sin (x2 - y2)) (h6 : a23 = Real.sin (x2 - y3))
  (h7 : a31 = Real.sin (x3 - y1)) (h8 : a32 = Real.sin (x3 - y2)) (h9 : a33 = Real.sin (x3 - y3)) :
  (Matrix.det ![![a11, a12, a13], ![a21, a22, a23], ![a31, a32, a33]]) = 0 := sorry

end NUMINAMATH_GPT_det_A_zero_l452_45206


namespace NUMINAMATH_GPT_problem_proof_l452_45204

variable (P Q M N : ℝ)

axiom hp1 : M = 0.40 * Q
axiom hp2 : Q = 0.30 * P
axiom hp3 : N = 1.20 * P

theorem problem_proof : (M / N) = (1 / 10) := by
  sorry

end NUMINAMATH_GPT_problem_proof_l452_45204


namespace NUMINAMATH_GPT_goods_train_speed_l452_45210

theorem goods_train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ) :
  length_train = 280.04 →
  length_platform = 240 →
  time_seconds = 26 →
  speed_kmph = (length_train + length_platform) / time_seconds * 3.6 →
  speed_kmph = 72 :=
by
  intros h_train h_platform h_time h_speed
  rw [h_train, h_platform, h_time] at h_speed
  sorry

end NUMINAMATH_GPT_goods_train_speed_l452_45210


namespace NUMINAMATH_GPT_no_rational_roots_l452_45260

theorem no_rational_roots (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = p) (h2 : Prime p) (h3: Nat.digits 10 p = [a, b, c, d]) : 
  ¬ ∃ x : ℚ, a * x^3 + b * x^2 + c * x + d = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_rational_roots_l452_45260


namespace NUMINAMATH_GPT_geometric_sequence_properties_l452_45234

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ)
    (h1 : a = -2 * r)
    (h2 : b = a * r)
    (h3 : c = b * r)
    (h4 : -8 = c * r) :
    b = -4 ∧ a * c = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l452_45234


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l452_45202

def p (x : ℝ) : Prop := x > 0
def q (x : ℝ) : Prop := |x| > 0

theorem sufficient_but_not_necessary (x : ℝ) : 
  (p x → q x) ∧ (¬(q x → p x)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l452_45202


namespace NUMINAMATH_GPT_inner_prod_sum_real_inner_prod_modulus_l452_45211

open Complex

-- Define the given mathematical expressions
noncomputable def pair (α β : ℂ) : ℝ := (1 / 4) * (norm (α + β) ^ 2 - norm (α - β) ^ 2)

noncomputable def inner_prod (α β : ℂ) : ℂ := pair α β + Complex.I * pair α (Complex.I * β)

-- Prove the given mathematical statements

-- 1. Prove that ⟨α, β⟩ + ⟨β, α⟩ is a real number
theorem inner_prod_sum_real (α β : ℂ) : (inner_prod α β + inner_prod β α).im = 0 := sorry

-- 2. Prove that |⟨α, β⟩| = |α| * |β|
theorem inner_prod_modulus (α β : ℂ) : Complex.abs (inner_prod α β) = Complex.abs α * Complex.abs β := sorry

end NUMINAMATH_GPT_inner_prod_sum_real_inner_prod_modulus_l452_45211


namespace NUMINAMATH_GPT_equation_infinitely_many_solutions_l452_45296

theorem equation_infinitely_many_solutions (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27 / 4 :=
sorry

end NUMINAMATH_GPT_equation_infinitely_many_solutions_l452_45296


namespace NUMINAMATH_GPT_graph_fixed_point_l452_45228

theorem graph_fixed_point {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ ∀ x : ℝ, y = a^(x + 2) - 2 ↔ (x, y) = A := 
by 
  sorry

end NUMINAMATH_GPT_graph_fixed_point_l452_45228


namespace NUMINAMATH_GPT_brad_running_speed_l452_45257

-- Definitions based on the given conditions
def distance_between_homes : ℝ := 24
def maxwell_walking_speed : ℝ := 4
def maxwell_time_to_meet : ℝ := 3

/-- Brad's running speed is 6 km/h given the conditions of the problem. -/
theorem brad_running_speed : (distance_between_homes - (maxwell_walking_speed * maxwell_time_to_meet)) / (maxwell_time_to_meet - 1) = 6 := by
  sorry

end NUMINAMATH_GPT_brad_running_speed_l452_45257


namespace NUMINAMATH_GPT_patrick_age_l452_45295

theorem patrick_age (r_age_future : ℕ) (years_future : ℕ) (half_age : ℕ → ℕ) 
  (h1 : r_age_future = 30) (h2 : years_future = 2) 
  (h3 : ∀ n, half_age n = n / 2) :
  half_age (r_age_future - years_future) = 14 :=
by
  sorry

end NUMINAMATH_GPT_patrick_age_l452_45295


namespace NUMINAMATH_GPT_unique_digit_sum_l452_45293

theorem unique_digit_sum (X Y M Z F : ℕ) (H1 : X ≠ 0) (H2 : Y ≠ 0) (H3 : M ≠ 0) (H4 : Z ≠ 0) (H5 : F ≠ 0)
  (H6 : X ≠ Y) (H7 : X ≠ M) (H8 : X ≠ Z) (H9 : X ≠ F)
  (H10 : Y ≠ M) (H11 : Y ≠ Z) (H12 : Y ≠ F)
  (H13 : M ≠ Z) (H14 : M ≠ F)
  (H15 : Z ≠ F)
  (H16 : 10 * X + Y ≠ 0) (H17 : 10 * M + Z ≠ 0)
  (H18 : 111 * F = (10 * X + Y) * (10 * M + Z)) :
  X + Y + M + Z + F = 28 := by
  sorry

end NUMINAMATH_GPT_unique_digit_sum_l452_45293


namespace NUMINAMATH_GPT_rate_percent_per_annum_l452_45266

theorem rate_percent_per_annum (P : ℝ) (SI_increase : ℝ) (T_increase : ℝ) (R : ℝ) 
  (hP : P = 2000) (hSI_increase : SI_increase = 40) (hT_increase : T_increase = 4) 
  (h : SI_increase = P * R * T_increase / 100) : R = 0.5 :=
by  
  sorry

end NUMINAMATH_GPT_rate_percent_per_annum_l452_45266


namespace NUMINAMATH_GPT_quadrilateral_EFGH_inscribed_in_circle_l452_45288

theorem quadrilateral_EFGH_inscribed_in_circle 
  (a b c : ℝ)
  (angle_EFG : ℝ := 60)
  (angle_EHG : ℝ := 50)
  (EH : ℝ := 5)
  (FG : ℝ := 7)
  (EG : ℝ := a)
  (EF : ℝ := b)
  (GH : ℝ := c)
  : EG = 7 * (Real.sin (70 * Real.pi / 180)) / (Real.sin (50 * Real.pi / 180)) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_EFGH_inscribed_in_circle_l452_45288


namespace NUMINAMATH_GPT_line_parallel_to_plane_line_perpendicular_to_plane_l452_45232

theorem line_parallel_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  A * m + B * n + C * p = 0 ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

theorem line_perpendicular_to_plane (A B C D x1 y1 z1 m n p : ℝ) :
  (A / m = B / n ∧ B / n = C / p) ↔ 
  ∀ x y z, ((A * x + B * y + C * z + D = 0) → 
  (∃ t, x = x1 + m * t ∧ y = y1 + n * t ∧ z = z1 + p * t)) :=
sorry

end NUMINAMATH_GPT_line_parallel_to_plane_line_perpendicular_to_plane_l452_45232


namespace NUMINAMATH_GPT_sqrt_expression_evaluation_l452_45252

theorem sqrt_expression_evaluation : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_evaluation_l452_45252


namespace NUMINAMATH_GPT_min_value_of_expression_l452_45284

noncomputable def min_expression := 4 * (Real.rpow 5 (1/4) - 1)^2

theorem min_value_of_expression (a b c : ℝ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = min_expression :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l452_45284


namespace NUMINAMATH_GPT_sandys_average_price_l452_45285

noncomputable def average_price_per_book (priceA : ℝ) (discountA : ℝ) (booksA : ℕ) (priceB : ℝ) (discountB : ℝ) (booksB : ℕ) (conversion_rate : ℝ) : ℝ :=
  let costA := priceA / (1 - discountA)
  let priceB_in_usd := priceB / conversion_rate
  let costB := priceB_in_usd / (1 - discountB)
  let total_cost := costA + costB
  let total_books := booksA + booksB
  total_cost / total_books

theorem sandys_average_price :
  average_price_per_book 1380 0.15 65 900 0.10 55 0.85 = 23.33 :=
by
  sorry

end NUMINAMATH_GPT_sandys_average_price_l452_45285


namespace NUMINAMATH_GPT_inequality_solution_l452_45272

theorem inequality_solution (z : ℝ) : 
  z^2 - 40 * z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l452_45272
