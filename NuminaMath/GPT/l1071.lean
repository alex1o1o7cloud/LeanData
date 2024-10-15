import Mathlib

namespace NUMINAMATH_GPT_registration_methods_l1071_107154

theorem registration_methods :
  ∀ (interns : ℕ) (companies : ℕ), companies = 4 → interns = 5 → companies^interns = 1024 :=
by intros interns companies h1 h2; rw [h1, h2]; exact rfl

end NUMINAMATH_GPT_registration_methods_l1071_107154


namespace NUMINAMATH_GPT_original_selling_price_is_800_l1071_107106

-- Let CP denote the cost price
variable (CP : ℝ)

-- Condition 1: Selling price with a profit of 25%
def selling_price_with_profit (CP : ℝ) : ℝ := 1.25 * CP

-- Condition 2: Selling price with a loss of 35%
def selling_price_with_loss (CP : ℝ) : ℝ := 0.65 * CP

-- Given selling price with loss is Rs. 416
axiom loss_price_is_416 : selling_price_with_loss CP = 416

-- We need to prove the original selling price (with profit) is Rs. 800
theorem original_selling_price_is_800 : selling_price_with_profit CP = 800 :=
by sorry

end NUMINAMATH_GPT_original_selling_price_is_800_l1071_107106


namespace NUMINAMATH_GPT_min_value_fraction_sum_l1071_107131

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (4 / (x + 2) + 1 / (y + 1)) ≥ 9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l1071_107131


namespace NUMINAMATH_GPT_complex_quadrant_l1071_107139

theorem complex_quadrant (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : 1 + a * i = (b + i) * (1 + i)) : 
  (a - b * i).re > 0 ∧ (a - b * i).im < 0 :=
by
  have h1 : 1 + a * i = (b - 1) + (b + 1) * i := by sorry
  have h2 : a = b + 1 := by sorry
  have h3 : b - 1 = 1 := by sorry
  have h4 : b = 2 := by sorry
  have h5 : a = 3 := by sorry
  have h6 : (a - b * i).re = 3 := by sorry
  have h7 : (a - b * i).im = -2 := by sorry
  exact ⟨by linarith, by linarith⟩

end NUMINAMATH_GPT_complex_quadrant_l1071_107139


namespace NUMINAMATH_GPT_find_duplicate_page_l1071_107171

theorem find_duplicate_page (n p : ℕ) (h : (n * (n + 1) / 2) + p = 3005) : p = 2 := 
sorry

end NUMINAMATH_GPT_find_duplicate_page_l1071_107171


namespace NUMINAMATH_GPT_sara_bought_cards_l1071_107126

-- Definition of the given conditions
def initial_cards : ℕ := 39
def torn_cards : ℕ := 9
def remaining_cards_after_sale : ℕ := 15

-- Derived definition: Number of good cards before selling to Sara
def good_cards_before_selling : ℕ := initial_cards - torn_cards

-- The statement we need to prove
theorem sara_bought_cards : good_cards_before_selling - remaining_cards_after_sale = 15 :=
by
  sorry

end NUMINAMATH_GPT_sara_bought_cards_l1071_107126


namespace NUMINAMATH_GPT_woman_speed_in_still_water_l1071_107177

noncomputable def speed_in_still_water (V_c : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_downstream := d / (t / 3600)
  V_downstream - V_c

theorem woman_speed_in_still_water :
  let V_c := 60
  let t := 9.99920006399488
  let d := 0.5 -- 500 meters converted to kilometers
  speed_in_still_water V_c t d = 120.01800180018 :=
by
  unfold speed_in_still_water
  sorry

end NUMINAMATH_GPT_woman_speed_in_still_water_l1071_107177


namespace NUMINAMATH_GPT_intersection_eq_expected_l1071_107183

def setA := { x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def setB := { x : ℝ | 1 ≤ x ∧ x < 4 }
def expectedSet := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq_expected :
  {x : ℝ | x ∈ setA ∧ x ∈ setB} = expectedSet :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_expected_l1071_107183


namespace NUMINAMATH_GPT_find_number_l1071_107115

theorem find_number (y : ℝ) (h : (30 / 100) * y = (25 / 100) * 40) : y = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1071_107115


namespace NUMINAMATH_GPT_hotdogs_remainder_zero_l1071_107101

theorem hotdogs_remainder_zero :
  25197624 % 6 = 0 :=
by
  sorry -- Proof not required

end NUMINAMATH_GPT_hotdogs_remainder_zero_l1071_107101


namespace NUMINAMATH_GPT_quarters_count_l1071_107108

theorem quarters_count (total_money : ℝ) (value_of_quarter : ℝ) (h1 : total_money = 3) (h2 : value_of_quarter = 0.25) : total_money / value_of_quarter = 12 :=
by sorry

end NUMINAMATH_GPT_quarters_count_l1071_107108


namespace NUMINAMATH_GPT_problem1_problem2_l1071_107117

variables {p x1 x2 y1 y2 : ℝ} (h₁ : p > 0) (h₂ : x1 * x2 ≠ 0) (h₃ : y1^2 = 2 * p * x1) (h₄ : y2^2 = 2 * p * x2)

theorem problem1 (h₅ : x1 * x2 + y1 * y2 = 0) :
    ∀ (x y : ℝ), (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 → 
        x^2 + y^2 - (x1 + x2) * x - (y1 + y2) * y = 0 := sorry

theorem problem2 (h₀ : ∀ x y, x = (x1 + x2) / 2 → y = (y1 + y2) / 2 → 
    |((x1 + x2) / 2) - 2 * ((y1 + y2) / 2)| / (Real.sqrt 5) = 2 * (Real.sqrt 5) / 5) :
    p = 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1071_107117


namespace NUMINAMATH_GPT_mod_pow_sum_7_l1071_107134

theorem mod_pow_sum_7 :
  (45 ^ 1234 + 27 ^ 1234) % 7 = 5 := by
  sorry

end NUMINAMATH_GPT_mod_pow_sum_7_l1071_107134


namespace NUMINAMATH_GPT_father_l1071_107175

theorem father's_age :
  ∃ (S F : ℕ), 2 * S + F = 70 ∧ S + 2 * F = 95 ∧ F = 40 :=
by
  sorry

end NUMINAMATH_GPT_father_l1071_107175


namespace NUMINAMATH_GPT_find_c_l1071_107146

theorem find_c (a c : ℝ) (h1 : x^2 + 80 * x + c = (x + a)^2) (h2 : 2 * a = 80) : c = 1600 :=
sorry

end NUMINAMATH_GPT_find_c_l1071_107146


namespace NUMINAMATH_GPT_ratio_of_expenditures_l1071_107132

-- Let us define the conditions and rewrite the proof problem statement.
theorem ratio_of_expenditures
  (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℝ)
  (H1 : income_P1 / income_P2 = 5 / 4)
  (H2 : income_P1 = 5000)
  (H3 : income_P1 - expenditure_P1 = 2000)
  (H4 : income_P2 - expenditure_P2 = 2000) :
  expenditure_P1 / expenditure_P2 = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_expenditures_l1071_107132


namespace NUMINAMATH_GPT_five_algorithmic_statements_l1071_107147

-- Define the five types of algorithmic statements in programming languages
inductive AlgorithmicStatement : Type
| input : AlgorithmicStatement
| output : AlgorithmicStatement
| assignment : AlgorithmicStatement
| conditional : AlgorithmicStatement
| loop : AlgorithmicStatement

-- Theorem: Every programming language contains these five basic types of algorithmic statements
theorem five_algorithmic_statements : 
  ∃ (s : List AlgorithmicStatement), 
    (s.length = 5) ∧ 
    ∀ x, x ∈ s ↔
    x = AlgorithmicStatement.input ∨
    x = AlgorithmicStatement.output ∨
    x = AlgorithmicStatement.assignment ∨
    x = AlgorithmicStatement.conditional ∨
    x = AlgorithmicStatement.loop :=
by
  sorry

end NUMINAMATH_GPT_five_algorithmic_statements_l1071_107147


namespace NUMINAMATH_GPT_reduced_flow_rate_is_correct_l1071_107163

-- Define the original flow rate
def original_flow_rate : ℝ := 5.0

-- Define the function for the reduced flow rate
def reduced_flow_rate (x : ℝ) : ℝ := 0.6 * x - 1

-- Prove that the reduced flow rate is 2.0 gallons per minute
theorem reduced_flow_rate_is_correct : reduced_flow_rate original_flow_rate = 2.0 := by
  sorry

end NUMINAMATH_GPT_reduced_flow_rate_is_correct_l1071_107163


namespace NUMINAMATH_GPT_complement_U_A_eq_l1071_107160
noncomputable def U := {x : ℝ | x ≥ -2}
noncomputable def A := {x : ℝ | x > -1}
noncomputable def comp_U_A := {x ∈ U | x ∉ A}

theorem complement_U_A_eq : comp_U_A = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by sorry

end NUMINAMATH_GPT_complement_U_A_eq_l1071_107160


namespace NUMINAMATH_GPT_Laura_won_5_games_l1071_107143

-- Define the number of wins and losses for each player
def Peter_wins : ℕ := 5
def Peter_losses : ℕ := 3
def Peter_games : ℕ := Peter_wins + Peter_losses

def Emma_wins : ℕ := 4
def Emma_losses : ℕ := 4
def Emma_games : ℕ := Emma_wins + Emma_losses

def Kyler_wins : ℕ := 2
def Kyler_losses : ℕ := 6
def Kyler_games : ℕ := Kyler_wins + Kyler_losses

-- Define the total number of games played in the tournament
def total_games_played : ℕ := (Peter_games + Emma_games + Kyler_games + 8) / 2

-- Define total wins and losses
def total_wins_losses : ℕ := total_games_played

-- Prove the number of games Laura won
def Laura_wins : ℕ := total_wins_losses - (Peter_wins + Emma_wins + Kyler_wins)

theorem Laura_won_5_games : Laura_wins = 5 := by
  -- The proof will be completed here
  sorry

end NUMINAMATH_GPT_Laura_won_5_games_l1071_107143


namespace NUMINAMATH_GPT_depth_of_melted_sauce_l1071_107194

theorem depth_of_melted_sauce
  (r_sphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) (volume_conserved : Bool) :
  r_sphere = 3 ∧ r_cylinder = 10 ∧ volume_conserved → h_cylinder = 9/25 :=
by
  -- Explanation of the condition: 
  -- r_sphere is the radius of the original spherical globe (3 inches)
  -- r_cylinder is the radius of the cylindrical puddle (10 inches)
  -- h_cylinder is the depth we need to prove is 9/25 inches
  -- volume_conserved indicates that the volume is conserved
  sorry

end NUMINAMATH_GPT_depth_of_melted_sauce_l1071_107194


namespace NUMINAMATH_GPT_king_total_payment_l1071_107155

theorem king_total_payment
  (crown_cost : ℕ)
  (architect_cost : ℕ)
  (chef_cost : ℕ)
  (crown_tip_percent : ℕ)
  (architect_tip_percent : ℕ)
  (chef_tip_percent : ℕ)
  (crown_tip : ℕ)
  (architect_tip : ℕ)
  (chef_tip : ℕ)
  (total_crown_cost : ℕ)
  (total_architect_cost : ℕ)
  (total_chef_cost : ℕ)
  (total_paid : ℕ) :
  crown_cost = 20000 →
  architect_cost = 50000 →
  chef_cost = 10000 →
  crown_tip_percent = 10 →
  architect_tip_percent = 5 →
  chef_tip_percent = 15 →
  crown_tip = crown_cost * crown_tip_percent / 100 →
  architect_tip = architect_cost * architect_tip_percent / 100 →
  chef_tip = chef_cost * chef_tip_percent / 100 →
  total_crown_cost = crown_cost + crown_tip →
  total_architect_cost = architect_cost + architect_tip →
  total_chef_cost = chef_cost + chef_tip →
  total_paid = total_crown_cost + total_architect_cost + total_chef_cost →
  total_paid = 86000 := by
  sorry

end NUMINAMATH_GPT_king_total_payment_l1071_107155


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l1071_107161

theorem geometric_progression_common_ratio (a r : ℝ) 
(h_pos: a > 0)
(h_condition: ∀ n : ℕ, a * r^(n-1) = (a * r^n + a * r^(n+1))^2):
  r = 0.618 :=
sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l1071_107161


namespace NUMINAMATH_GPT_intersection_A_B_l1071_107164

open Set

variable (l : ℝ)

def A := {x : ℝ | x > l}
def B := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_A_B (h₁ : l = 1) :
  A l ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1071_107164


namespace NUMINAMATH_GPT_range_of_a_l1071_107130

noncomputable def satisfiesInequality (a : ℝ) (x : ℝ) : Prop :=
  x > 1 → a * Real.log x > 1 - 1/x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 1 → satisfiesInequality a x) ↔ a ∈ Set.Ici 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1071_107130


namespace NUMINAMATH_GPT_find_d_l1071_107140

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) : d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1071_107140


namespace NUMINAMATH_GPT_solve_for_d_l1071_107100

variable (n c b d : ℚ)  -- Alternatively, specify the types if they are required to be specific
variable (H : n = d * c * b / (c - d))

theorem solve_for_d :
  d = n * c / (c * b + n) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_d_l1071_107100


namespace NUMINAMATH_GPT_rectangle_width_decrease_l1071_107167

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_decrease_l1071_107167


namespace NUMINAMATH_GPT_lunks_needed_for_bananas_l1071_107176

theorem lunks_needed_for_bananas :
  (7 : ℚ) / 4 * (20 * 3 / 5) = 21 :=
by
  sorry

end NUMINAMATH_GPT_lunks_needed_for_bananas_l1071_107176


namespace NUMINAMATH_GPT_third_box_weight_l1071_107186

def b1 : ℕ := 2
def difference := 11

def weight_third_box (b1 b3 difference : ℕ) : Prop :=
  b3 - b1 = difference

theorem third_box_weight : weight_third_box b1 13 difference :=
by
  simp [b1, difference]
  sorry

end NUMINAMATH_GPT_third_box_weight_l1071_107186


namespace NUMINAMATH_GPT_find_bigger_number_l1071_107178

noncomputable def common_factor (x : ℕ) : Prop :=
  8 * x + 3 * x = 143

theorem find_bigger_number (x : ℕ) (h : common_factor x) : 8 * x = 104 :=
by
  sorry

end NUMINAMATH_GPT_find_bigger_number_l1071_107178


namespace NUMINAMATH_GPT_average_temperature_week_l1071_107159

theorem average_temperature_week 
  (T_sun : ℝ := 40)
  (T_mon : ℝ := 50)
  (T_tue : ℝ := 65)
  (T_wed : ℝ := 36)
  (T_thu : ℝ := 82)
  (T_fri : ℝ := 72)
  (T_sat : ℝ := 26) :
  (T_sun + T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) / 7 = 53 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_week_l1071_107159


namespace NUMINAMATH_GPT_inequality_condition_l1071_107120

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the main theorem to be proven
theorem inequality_condition (a b : ℝ) (h_a : a > 11 / 4) (h_b : b > 3 / 2) :
  (∀ x : ℝ, |x + 1| < b → |f x + 3| < a) :=
by
  -- We state the required proof without providing the steps
  sorry

end NUMINAMATH_GPT_inequality_condition_l1071_107120


namespace NUMINAMATH_GPT_extremum_point_of_f_l1071_107138

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem extremum_point_of_f : ∃ x, x = 1 ∧ (∀ y ≠ 1, f y ≥ f x) := 
sorry

end NUMINAMATH_GPT_extremum_point_of_f_l1071_107138


namespace NUMINAMATH_GPT_sum_of_center_coords_l1071_107179

theorem sum_of_center_coords (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : 2 + (-3) = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_center_coords_l1071_107179


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l1071_107118

theorem cost_price_of_computer_table
  (C : ℝ) 
  (S : ℝ := 1.20 * C)
  (S_eq : S = 8600) : 
  C = 7166.67 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l1071_107118


namespace NUMINAMATH_GPT_proofSmallestM_l1071_107142

def LeanProb (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 2512 →
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧ (0 < e) ∧ (0 < f) →
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f)))))

theorem proofSmallestM (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 2512) 
(h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 0 < d) (h6 : 0 < e) (h7 : 0 < f) : 
  ∃ M, (M = 1005) ∧ (M = max (a+b) (max (b+c) (max (c+d) (max (d+e) (e+f))))):=
by
  sorry

end NUMINAMATH_GPT_proofSmallestM_l1071_107142


namespace NUMINAMATH_GPT_eric_containers_l1071_107135

theorem eric_containers (initial_pencils : ℕ) (additional_pencils : ℕ) (pencils_per_container : ℕ) 
  (h1 : initial_pencils = 150) (h2 : additional_pencils = 30) (h3 : pencils_per_container = 36) :
  (initial_pencils + additional_pencils) / pencils_per_container = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_eric_containers_l1071_107135


namespace NUMINAMATH_GPT_minimum_value_inequality_l1071_107180

theorem minimum_value_inequality (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) : m + (n^2 - mn + 4)/(m - n) ≥ 4 :=
  sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1071_107180


namespace NUMINAMATH_GPT_theo_selling_price_l1071_107123

theorem theo_selling_price:
  ∀ (maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell: ℕ),
    maddox_price = 20 → 
    theo_cost = 20 → 
    maddox_sell = 28 →
    maddox_profit = (maddox_sell - maddox_price) * 3 →
    (theo_sell - theo_cost) * 3 = (maddox_profit - 15) →
    theo_sell = 23 := by
  intros maddox_price theo_cost maddox_sell theo_profit maddox_profit theo_sell
  intros maddox_price_eq theo_cost_eq maddox_sell_eq maddox_profit_eq theo_profit_eq

  -- Use given assumptions
  rw [maddox_price_eq, theo_cost_eq, maddox_sell_eq] at *
  simp at *

  -- Final goal
  sorry

end NUMINAMATH_GPT_theo_selling_price_l1071_107123


namespace NUMINAMATH_GPT_Louisa_total_travel_time_l1071_107185

theorem Louisa_total_travel_time :
  ∀ (v : ℝ), v > 0 → (200 / v) + 4 = (360 / v) → (200 / v) + (360 / v) = 14 :=
by
  intros v hv eqn
  sorry

end NUMINAMATH_GPT_Louisa_total_travel_time_l1071_107185


namespace NUMINAMATH_GPT_gcd_12345_6789_eq_3_l1071_107198

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_12345_6789_eq_3_l1071_107198


namespace NUMINAMATH_GPT_chosen_number_l1071_107192

theorem chosen_number (x : ℝ) (h : 2 * x - 138 = 102) : x = 120 := by
  sorry

end NUMINAMATH_GPT_chosen_number_l1071_107192


namespace NUMINAMATH_GPT_cubic_roots_l1071_107197

theorem cubic_roots (a x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 1) (h₃ : x₃ = a)
  (cond : (2 / x₁) + (2 / x₂) = (3 / x₃)) :
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = a ∧ (a = 2 ∨ a = 3 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_l1071_107197


namespace NUMINAMATH_GPT_total_people_ride_l1071_107105

theorem total_people_ride (people_per_carriage : ℕ) (num_carriages : ℕ) (h1 : people_per_carriage = 12) (h2 : num_carriages = 15) : 
    people_per_carriage * num_carriages = 180 := by
  sorry

end NUMINAMATH_GPT_total_people_ride_l1071_107105


namespace NUMINAMATH_GPT_A_inter_B_l1071_107114

open Set Real

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { y | ∃ x, y = exp x }

theorem A_inter_B :
  A ∩ B = { z | 0 < z ∧ z < 3 } :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_l1071_107114


namespace NUMINAMATH_GPT_domain_of_f_l1071_107107

noncomputable def f (x : ℝ) : ℝ := (Real.log (x^2 - 1)) / (Real.sqrt (x^2 - x - 2))

theorem domain_of_f :
  {x : ℝ | x^2 - 1 > 0 ∧ x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1071_107107


namespace NUMINAMATH_GPT_shadow_length_l1071_107157

variable (H h d : ℝ) (h_pos : h > 0) (H_pos : H > 0) (H_neq_h : H ≠ h)

theorem shadow_length (x : ℝ) (hx : x = d * h / (H - h)) :
  x = d * h / (H - h) :=
sorry

end NUMINAMATH_GPT_shadow_length_l1071_107157


namespace NUMINAMATH_GPT_point_b_not_inside_circle_a_l1071_107165

theorem point_b_not_inside_circle_a (a : ℝ) : a < 5 → ¬ (1 < a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_GPT_point_b_not_inside_circle_a_l1071_107165


namespace NUMINAMATH_GPT_minimum_cuts_l1071_107151

theorem minimum_cuts (n : Nat) : n >= 50 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cuts_l1071_107151


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l1071_107109

theorem bus_speed_including_stoppages :
  ∀ (s t : ℝ), s = 75 → t = 24 → (s * ((60 - t) / 60)) = 45 :=
by
  intros s t hs ht
  rw [hs, ht]
  sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l1071_107109


namespace NUMINAMATH_GPT_symmetric_points_power_l1071_107190

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_points_power_l1071_107190


namespace NUMINAMATH_GPT_total_volume_of_cubes_l1071_107169

theorem total_volume_of_cubes 
  (Carl_cubes : ℕ)
  (Carl_side_length : ℕ)
  (Kate_cubes : ℕ)
  (Kate_side_length : ℕ)
  (h1 : Carl_cubes = 8)
  (h2 : Carl_side_length = 2)
  (h3 : Kate_cubes = 3)
  (h4 : Kate_side_length = 3) :
  Carl_cubes * Carl_side_length ^ 3 + Kate_cubes * Kate_side_length ^ 3 = 145 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_cubes_l1071_107169


namespace NUMINAMATH_GPT_practice_hours_l1071_107174

-- Define the starting and ending hours, and the break duration
def start_hour : ℕ := 8
def end_hour : ℕ := 16
def break_duration : ℕ := 2

-- Compute the total practice hours
def total_practice_time : ℕ := (end_hour - start_hour) - break_duration

-- State that the computed practice time is equal to 6 hours
theorem practice_hours :
  total_practice_time = 6 := 
by
  -- Using the definitions provided to state the proof
  sorry

end NUMINAMATH_GPT_practice_hours_l1071_107174


namespace NUMINAMATH_GPT_cos_7_theta_l1071_107193

variable (θ : Real)

namespace CosineProof

theorem cos_7_theta (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -5669 / 16384 := by
  sorry

end CosineProof

end NUMINAMATH_GPT_cos_7_theta_l1071_107193


namespace NUMINAMATH_GPT_smallest_student_count_l1071_107196

theorem smallest_student_count (x y z w : ℕ) 
  (ratio12to10 : x / y = 3 / 2) 
  (ratio12to11 : x / z = 7 / 4) 
  (ratio12to9 : x / w = 5 / 3) : 
  x + y + z + w = 298 :=
by
  sorry

end NUMINAMATH_GPT_smallest_student_count_l1071_107196


namespace NUMINAMATH_GPT_minimum_surface_area_l1071_107170

def small_cuboid_1_length := 3 -- Edge length of small cuboid
def small_cuboid_2_length := 4 -- Edge length of small cuboid
def small_cuboid_3_length := 5 -- Edge length of small cuboid

def num_small_cuboids := 24 -- Number of small cuboids used to build the large cuboid

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def large_cuboid_length := 15 -- Corrected length dimension
def large_cuboid_width := 10  -- Corrected width dimension
def large_cuboid_height := 16 -- Corrected height dimension

theorem minimum_surface_area : surface_area large_cuboid_length large_cuboid_width large_cuboid_height = 788 := by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_minimum_surface_area_l1071_107170


namespace NUMINAMATH_GPT_transmission_time_estimation_l1071_107153

noncomputable def number_of_blocks := 80
noncomputable def chunks_per_block := 640
noncomputable def transmission_rate := 160 -- chunks per second
noncomputable def seconds_per_minute := 60
noncomputable def total_chunks := number_of_blocks * chunks_per_block
noncomputable def total_time_seconds := total_chunks / transmission_rate
noncomputable def total_time_minutes := total_time_seconds / seconds_per_minute

theorem transmission_time_estimation : total_time_minutes = 5 := 
  sorry

end NUMINAMATH_GPT_transmission_time_estimation_l1071_107153


namespace NUMINAMATH_GPT_value_of_expression_l1071_107119

theorem value_of_expression : (3023 - 2990) ^ 2 / 121 = 9 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1071_107119


namespace NUMINAMATH_GPT_number_of_subsets_of_set_A_l1071_107168

theorem number_of_subsets_of_set_A : 
  (setOfSubsets : Finset (Finset ℕ)) = Finset.powerset {2, 4, 5} → 
  setOfSubsets.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_set_A_l1071_107168


namespace NUMINAMATH_GPT_intervals_of_increase_of_f_l1071_107182

theorem intervals_of_increase_of_f :
  ∀ k : ℤ,
  ∀ x y : ℝ,
  k * π - (5 / 8) * π ≤ x ∧ x ≤ y ∧ y ≤ k * π - (1 / 8) * π →
  3 * Real.sin ((π / 4) - 2 * x) - 2 ≤ 3 * Real.sin ((π / 4) - 2 * y) - 2 :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_increase_of_f_l1071_107182


namespace NUMINAMATH_GPT_recurring_decimal_sum_l1071_107113

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end NUMINAMATH_GPT_recurring_decimal_sum_l1071_107113


namespace NUMINAMATH_GPT_Tom_age_ratio_l1071_107112

variable (T N : ℕ)
variable (a : ℕ)
variable (c3 c4 : ℕ)

-- conditions
def condition1 : Prop := T = 4 * a + 5
def condition2 : Prop := T - N = 3 * (4 * a + 5 - 4 * N)

theorem Tom_age_ratio (h1 : condition1 T a) (h2 : condition2 T N a) : (T = 6 * N) :=
by sorry

end NUMINAMATH_GPT_Tom_age_ratio_l1071_107112


namespace NUMINAMATH_GPT_johns_share_l1071_107144

theorem johns_share (total_amount : ℕ) (r1 r2 r3 : ℕ) (h : total_amount = 6000) (hr1 : r1 = 2) (hr2 : r2 = 4) (hr3 : r3 = 6) :
  let total_ratio := r1 + r2 + r3
  let johns_ratio := r1
  let johns_share := (johns_ratio * total_amount) / total_ratio
  johns_share = 1000 :=
by
  sorry

end NUMINAMATH_GPT_johns_share_l1071_107144


namespace NUMINAMATH_GPT_car_distance_l1071_107166

/-- A car takes 4 hours to cover a certain distance. We are given that the car should maintain a speed of 90 kmph to cover the same distance in (3/2) of the previous time (which is 6 hours). We need to prove that the distance the car needs to cover is 540 km. -/
theorem car_distance (time_initial : ℝ) (speed : ℝ) (time_new : ℝ) (distance : ℝ) 
  (h1 : time_initial = 4) 
  (h2 : speed = 90)
  (h3 : time_new = (3/2) * time_initial)
  (h4 : distance = speed * time_new) : 
  distance = 540 := 
sorry

end NUMINAMATH_GPT_car_distance_l1071_107166


namespace NUMINAMATH_GPT_LCM_of_numbers_l1071_107137

theorem LCM_of_numbers (a b : ℕ) (h1 : a = 20) (h2 : a / b = 5 / 4): Nat.lcm a b = 80 :=
by
  sorry

end NUMINAMATH_GPT_LCM_of_numbers_l1071_107137


namespace NUMINAMATH_GPT_bella_eats_six_apples_a_day_l1071_107156

variable (A : ℕ) -- Number of apples Bella eats per day
variable (G : ℕ) -- Total number of apples Grace picks in 6 weeks
variable (B : ℕ) -- Total number of apples Bella eats in 6 weeks

-- Definitions for the conditions 
def condition1 := B = 42 * A
def condition2 := B = (1 / 3) * G
def condition3 := (2 / 3) * G = 504

-- Final statement that needs to be proved
theorem bella_eats_six_apples_a_day (A G B : ℕ) 
  (h1 : condition1 A B) 
  (h2 : condition2 G B) 
  (h3 : condition3 G) 
  : A = 6 := by sorry

end NUMINAMATH_GPT_bella_eats_six_apples_a_day_l1071_107156


namespace NUMINAMATH_GPT_angle_relationship_l1071_107181

open Real

variables (A B C D : Point)
variables (AB AC AD : ℝ)
variables (CAB DAC BDC DBC : ℝ)
variables (k : ℝ)

-- Given conditions
axiom h1 : AB = AC
axiom h2 : AC = AD
axiom h3 : DAC = k * CAB

-- Proof to be shown
theorem angle_relationship : DBC = k * BDC :=
  sorry

end NUMINAMATH_GPT_angle_relationship_l1071_107181


namespace NUMINAMATH_GPT_calculate_total_revenue_l1071_107150

-- Definitions based on conditions
def apple_pie_slices := 8
def peach_pie_slices := 6
def cherry_pie_slices := 10

def apple_pie_price := 3
def peach_pie_price := 4
def cherry_pie_price := 5

def apple_pie_customers := 88
def peach_pie_customers := 78
def cherry_pie_customers := 45

-- Definition of total revenue
def total_revenue := 
  (apple_pie_customers * apple_pie_price) + 
  (peach_pie_customers * peach_pie_price) + 
  (cherry_pie_customers * cherry_pie_price)

-- Target theorem to prove: total revenue equals 801
theorem calculate_total_revenue : total_revenue = 801 := by
  sorry

end NUMINAMATH_GPT_calculate_total_revenue_l1071_107150


namespace NUMINAMATH_GPT_coin_difference_l1071_107127

-- Define the coin denominations
def coin_denominations : List ℕ := [5, 10, 25, 50]

-- Define the target amount Paul needs to pay
def target_amount : ℕ := 60

-- Define the function to compute the minimum number of coins required
noncomputable def min_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the function to compute the maximum number of coins required
noncomputable def max_coins (target : ℕ) (denominations : List ℕ) : ℕ :=
  sorry -- Implementation of the function is not essential for this statement

-- Define the theorem to state the difference between max and min coins is 10
theorem coin_difference : max_coins target_amount coin_denominations - min_coins target_amount coin_denominations = 10 :=
  sorry

end NUMINAMATH_GPT_coin_difference_l1071_107127


namespace NUMINAMATH_GPT_find_k_l1071_107148

theorem find_k (a k : ℝ) (h : a ≠ 0) (h1 : 3 * a + a = -12)
  (h2 : (3 * a) * a = k) : k = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1071_107148


namespace NUMINAMATH_GPT_max_common_initial_segment_l1071_107133

theorem max_common_initial_segment (m n : ℕ) (h_coprime : Nat.gcd m n = 1) : 
  ∃ L, L = m + n - 2 := 
sorry

end NUMINAMATH_GPT_max_common_initial_segment_l1071_107133


namespace NUMINAMATH_GPT_first_number_in_list_is_55_l1071_107121

theorem first_number_in_list_is_55 : 
  ∀ (x : ℕ), (55 + 57 + 58 + 59 + 62 + 62 + 63 + 65 + x) / 9 = 60 → x = 65 → 55 = 55 :=
by
  intros x avg_cond x_is_65
  rfl

end NUMINAMATH_GPT_first_number_in_list_is_55_l1071_107121


namespace NUMINAMATH_GPT_polygon_sum_of_sides_l1071_107128

-- Define the problem conditions and statement
theorem polygon_sum_of_sides :
  ∀ (A B C D E F : ℝ)
    (area_polygon : ℝ)
    (AB BC FA DE horizontal_distance_DF : ℝ),
    area_polygon = 75 →
    AB = 7 →
    BC = 10 →
    FA = 6 →
    DE = AB →
    horizontal_distance_DF = 8 →
    (DE + (2 * area_polygon - AB * BC) / (2 * horizontal_distance_DF) = 8.25) := 
by
  intro A B C D E F area_polygon AB BC FA DE horizontal_distance_DF
  intro h_area_polygon h_AB h_BC h_FA h_DE h_horizontal_distance_DF
  sorry

end NUMINAMATH_GPT_polygon_sum_of_sides_l1071_107128


namespace NUMINAMATH_GPT_tap_emptying_time_l1071_107110

theorem tap_emptying_time
  (F : ℝ := 1 / 3)
  (T_combined : ℝ := 7.5):
  ∃ x : ℝ, x = 5 ∧ (F - (1 / x) = 1 / T_combined) := 
sorry

end NUMINAMATH_GPT_tap_emptying_time_l1071_107110


namespace NUMINAMATH_GPT_valerie_needs_21_stamps_l1071_107145

def thank_you_cards : ℕ := 3
def bills : ℕ := 2
def mail_in_rebates : ℕ := bills + 3
def job_applications : ℕ := 2 * mail_in_rebates
def water_bill_stamps : ℕ := 1
def electric_bill_stamps : ℕ := 2

def stamps_for_thank_you_cards : ℕ := thank_you_cards * 1
def stamps_for_bills : ℕ := 1 * water_bill_stamps + 1 * electric_bill_stamps
def stamps_for_rebates : ℕ := mail_in_rebates * 1
def stamps_for_job_applications : ℕ := job_applications * 1

def total_stamps : ℕ :=
  stamps_for_thank_you_cards +
  stamps_for_bills +
  stamps_for_rebates +
  stamps_for_job_applications

theorem valerie_needs_21_stamps : total_stamps = 21 := by
  sorry

end NUMINAMATH_GPT_valerie_needs_21_stamps_l1071_107145


namespace NUMINAMATH_GPT_set_inclusion_interval_l1071_107136

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_GPT_set_inclusion_interval_l1071_107136


namespace NUMINAMATH_GPT_sixth_grade_boys_l1071_107141

theorem sixth_grade_boys (x : ℕ) :
    (1 / 11) * x + (147 - x) = 147 - x → 
    (152 - (x - (1 / 11) * x + (147 - x) - (152 - x - 5))) = x
    → x = 77 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sixth_grade_boys_l1071_107141


namespace NUMINAMATH_GPT_sum_of_nine_consecutive_parity_l1071_107149

theorem sum_of_nine_consecutive_parity (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) + (n + 8)) % 2 = n % 2 := 
  sorry

end NUMINAMATH_GPT_sum_of_nine_consecutive_parity_l1071_107149


namespace NUMINAMATH_GPT_remainder_when_200_divided_by_k_l1071_107162

theorem remainder_when_200_divided_by_k (k : ℕ) (hk_pos : 0 < k)
  (h₁ : 125 % (k^3) = 5) : 200 % k = 0 :=
sorry

end NUMINAMATH_GPT_remainder_when_200_divided_by_k_l1071_107162


namespace NUMINAMATH_GPT_number_of_red_balls_l1071_107124

theorem number_of_red_balls (m : ℕ) (h1 : ∃ m : ℕ, (3 / (m + 3) : ℚ) = 1 / 4) : m = 9 :=
by
  obtain ⟨m, h1⟩ := h1
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l1071_107124


namespace NUMINAMATH_GPT_increasing_range_of_a_l1071_107129

noncomputable def f (x : ℝ) (a : ℝ) := 
  if x ≤ 1 then -x^2 + 4*a*x 
  else (2*a + 3)*x - 4*a + 5

theorem increasing_range_of_a :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (1 / 2 ≤ a ∧ a ≤ 3 / 2) :=
sorry

end NUMINAMATH_GPT_increasing_range_of_a_l1071_107129


namespace NUMINAMATH_GPT_find_m_l1071_107102

section
variables {R : Type*} [CommRing R]

def f (x : R) : R := 4 * x^2 - 3 * x + 5
def g (x : R) (m : R) : R := x^2 - m * x - 8

theorem find_m (m : ℚ) : 
  f (5 : ℚ) - g (5 : ℚ) m = 20 → m = -53 / 5 :=
by {
  sorry
}

end

end NUMINAMATH_GPT_find_m_l1071_107102


namespace NUMINAMATH_GPT_pipe_drain_rate_l1071_107188

theorem pipe_drain_rate 
(T r_A r_B r_C : ℕ) 
(h₁ : T = 950) 
(h₂ : r_A = 40) 
(h₃ : r_B = 30) 
(h₄ : ∃ m : ℕ, m = 57 ∧ (T = (m / 3) * (r_A + r_B - r_C))) : 
r_C = 20 :=
sorry

end NUMINAMATH_GPT_pipe_drain_rate_l1071_107188


namespace NUMINAMATH_GPT_initial_hours_per_day_l1071_107104

-- Definitions capturing the conditions
def num_men_initial : ℕ := 100
def num_men_total : ℕ := 160
def portion_completed : ℚ := 1 / 3
def num_days_total : ℕ := 50
def num_days_half : ℕ := 25
def work_performed_portion : ℚ := 2 / 3
def hours_per_day_additional : ℕ := 10

-- Lean statement to prove the initial number of hours per day worked by the initial employees
theorem initial_hours_per_day (H : ℚ) :
  (num_men_initial * H * num_days_total = work_performed_portion) ∧
  (num_men_total * hours_per_day_additional * num_days_half = portion_completed) →
  H = 1.6 := 
sorry

end NUMINAMATH_GPT_initial_hours_per_day_l1071_107104


namespace NUMINAMATH_GPT_product_of_solutions_l1071_107152

theorem product_of_solutions :
  (∀ x : ℝ, |3 * x - 2| + 5 = 23 → x = 20 / 3 ∨ x = -16 / 3) →
  (20 / 3 * -16 / 3 = -320 / 9) :=
by
  intros h
  have h₁ : 20 / 3 * -16 / 3 = -320 / 9 := sorry
  exact h₁

end NUMINAMATH_GPT_product_of_solutions_l1071_107152


namespace NUMINAMATH_GPT_simplify_expression_l1071_107116

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1071_107116


namespace NUMINAMATH_GPT_length_AF_l1071_107111

def CE : ℝ := 40
def ED : ℝ := 50
def AE : ℝ := 120
def area_ABCD : ℝ := 7200

theorem length_AF (AF : ℝ) :
  CE = 40 → ED = 50 → AE = 120 → area_ABCD = 7200 →
  AF = 128 :=
by
  intros hCe hEd hAe hArea
  sorry

end NUMINAMATH_GPT_length_AF_l1071_107111


namespace NUMINAMATH_GPT_find_F_58_59_60_l1071_107195

def F : ℤ → ℤ → ℤ → ℝ := sorry

axiom F_scaling (a b c n : ℤ) : F (n * a) (n * b) (n * c) = n * F a b c
axiom F_shift (a b c n : ℤ) : F (a + n) (b + n) (c + n) = F a b c + n
axiom F_symmetry (a b c : ℤ) : F a b c = F c b a

theorem find_F_58_59_60 : F 58 59 60 = 59 :=
sorry

end NUMINAMATH_GPT_find_F_58_59_60_l1071_107195


namespace NUMINAMATH_GPT_A_finishes_remaining_work_in_2_days_l1071_107191

/-- 
Given that A's daily work rate is 1/6 of the work and B's daily work rate is 1/15 of the work,
and B has already completed 2/3 of the work, 
prove that A can finish the remaining work in 2 days.
-/
theorem A_finishes_remaining_work_in_2_days :
  let A_work_rate := (1 : ℝ) / 6
  let B_work_rate := (1 : ℝ) / 15
  let B_work_in_10_days := (10 : ℝ) * B_work_rate
  let remaining_work := (1 : ℝ) - B_work_in_10_days
  let days_for_A := remaining_work / A_work_rate
  B_work_in_10_days = 2 / 3 → 
  remaining_work = 1 / 3 → 
  days_for_A = 2 :=
by
  sorry

end NUMINAMATH_GPT_A_finishes_remaining_work_in_2_days_l1071_107191


namespace NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l1071_107122

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ (x y : ℝ), (x - y)^2 = 3 * x^2 - y^2 ↔ 
  (x = (1 - Real.sqrt 5) / 2 * y) ∨ (x = (1 + Real.sqrt 5) / 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l1071_107122


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1071_107172

-- Define N
def N := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

-- Define function to calculate sum of digits
def sum_of_digits(n: Nat) : Nat :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_of_N : sum_of_digits N = 7 :=
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1071_107172


namespace NUMINAMATH_GPT_divisor_of_44404_l1071_107103

theorem divisor_of_44404: ∃ k : ℕ, 2 * 11101 = k ∧ k ∣ (44402 + 2) :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_44404_l1071_107103


namespace NUMINAMATH_GPT_minimum_value_hyperbola_l1071_107125

noncomputable def min_value (a b : ℝ) (h : a > 0) (k : b > 0)
  (eccentricity_eq_two : (2:ℝ) = Real.sqrt (1 + (b/a)^2)) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem minimum_value_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2:ℝ) = Real.sqrt (1 + (b/a)^2) ∧
  min_value a b (by sorry) (by sorry) (by sorry) = (2 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_hyperbola_l1071_107125


namespace NUMINAMATH_GPT_total_volume_collection_l1071_107189

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def cost_per_box : ℚ := 0.5
def minimum_total_cost : ℚ := 255

-- Define the volume of one box
def volume_of_one_box : ℕ := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes : ℚ := minimum_total_cost / cost_per_box

-- Define the total volume of the collection
def total_volume : ℚ := volume_of_one_box * number_of_boxes

-- The goal is to prove that the total volume of the collection is as calculated
theorem total_volume_collection :
  total_volume = 3060000 := by
  sorry

end NUMINAMATH_GPT_total_volume_collection_l1071_107189


namespace NUMINAMATH_GPT_find_k_l1071_107199

-- Define the conditions and the question
theorem find_k (t k : ℝ) (h1 : t = 50) (h2 : t = (5 / 9) * (k - 32)) : k = 122 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_find_k_l1071_107199


namespace NUMINAMATH_GPT_even_function_inequality_l1071_107173

variable {α : Type*} [LinearOrderedField α]

def is_even_function (f : α → α) : Prop := ∀ x, f x = f (-x)

-- The hypothesis and the assertion in Lean
theorem even_function_inequality
  (f : α → α)
  (h_even : is_even_function f)
  (h3_gt_1 : f 3 > f 1)
  : f (-1) < f 3 :=
sorry

end NUMINAMATH_GPT_even_function_inequality_l1071_107173


namespace NUMINAMATH_GPT_determine_a_l1071_107158

theorem determine_a 
(h : ∃x, x = -1 ∧ 2 * x ^ 2 + a * x - a ^ 2 = 0) : a = -2 ∨ a = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_determine_a_l1071_107158


namespace NUMINAMATH_GPT_composite_sum_of_squares_l1071_107187

theorem composite_sum_of_squares (a b : ℤ) (h_roots : ∃ x1 x2 : ℕ, (x1 + x2 : ℤ) = -a ∧ (x1 * x2 : ℤ) = b + 1) :
  ∃ m n : ℕ, a^2 + b^2 = m * n ∧ 1 < m ∧ 1 < n :=
sorry

end NUMINAMATH_GPT_composite_sum_of_squares_l1071_107187


namespace NUMINAMATH_GPT_intersection_of_sets_l1071_107184

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1071_107184
