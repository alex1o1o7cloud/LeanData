import Mathlib

namespace change_in_expression_is_correct_l531_53142

def change_in_expression (x a : ℝ) : ℝ :=
  if increases : true then (x + a)^2 - 3 - (x^2 - 3)
  else (x - a)^2 - 3 - (x^2 - 3)

theorem change_in_expression_is_correct (x a : ℝ) :
  a > 0 → change_in_expression x a = 2 * a * x + a^2 ∨ change_in_expression x a = -(2 * a * x) + a^2 :=
by
  sorry

end change_in_expression_is_correct_l531_53142


namespace rajan_income_l531_53179

theorem rajan_income (x y : ℝ) 
  (h₁ : 7 * x - 6 * y = 1000) 
  (h₂ : 6 * x - 5 * y = 1000) : 
  7 * x = 7000 :=
by 
  sorry

end rajan_income_l531_53179


namespace problem_l531_53188

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x ^ 2

noncomputable def f (gx : ℝ) (x : ℝ) : ℝ := (2 - 3 * x ^ 2) / x ^ 2

theorem problem (x : ℝ) (hx : x ≠ 0) : f (g x) x = 3 / 2 :=
  sorry

end problem_l531_53188


namespace inequality_example_l531_53190

theorem inequality_example (a b m : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_m_pos : 0 < m) (h_ba : b = 2) (h_aa : a = 1) :
  (b + m) / (a + m) < b / a :=
sorry

end inequality_example_l531_53190


namespace set_of_x_satisfying_inequality_l531_53183

theorem set_of_x_satisfying_inequality : 
  {x : ℝ | (x - 2)^2 < 9} = {x : ℝ | -1 < x ∧ x < 5} :=
by
  sorry

end set_of_x_satisfying_inequality_l531_53183


namespace find_x_correct_l531_53141

theorem find_x_correct (x : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * 180 = 360 * x - 480)
  (h2 : (x + 4) + x + (x - 4) = 3 * x)
  (h3 : 100 = (360 * x - 480) / (3 * x)) : 
  x = 8 := 
sorry

end find_x_correct_l531_53141


namespace decreasing_function_range_l531_53132

theorem decreasing_function_range {a : ℝ} (h1 : ∀ x y : ℝ, x < y → (1 - 2 * a)^x > (1 - 2 * a)^y) : 
    0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l531_53132


namespace mary_initial_pokemon_cards_l531_53197

theorem mary_initial_pokemon_cards (x : ℕ) (torn_cards : ℕ) (new_cards : ℕ) (current_cards : ℕ) 
  (h1 : torn_cards = 6) 
  (h2 : new_cards = 23) 
  (h3 : current_cards = 56) 
  (h4 : current_cards = x - torn_cards + new_cards) : 
  x = 39 := 
by
  sorry

end mary_initial_pokemon_cards_l531_53197


namespace undefined_values_of_expression_l531_53146

theorem undefined_values_of_expression (a : ℝ) :
  a^2 - 9 = 0 ↔ a = -3 ∨ a = 3 := 
sorry

end undefined_values_of_expression_l531_53146


namespace pieces_given_by_brother_l531_53105

-- Given conditions
def original_pieces : ℕ := 18
def total_pieces_now : ℕ := 62

-- The statement to prove
theorem pieces_given_by_brother : total_pieces_now - original_pieces = 44 := by
  -- Starting with the given conditions
  unfold original_pieces total_pieces_now
  -- Place to insert the proof
  sorry

end pieces_given_by_brother_l531_53105


namespace overall_profit_refrigerator_mobile_phone_l531_53125

theorem overall_profit_refrigerator_mobile_phone
  (purchase_price_refrigerator : ℕ)
  (purchase_price_mobile_phone : ℕ)
  (loss_percentage_refrigerator : ℕ)
  (profit_percentage_mobile_phone : ℕ)
  (selling_price_refrigerator : ℕ)
  (selling_price_mobile_phone : ℕ)
  (total_cost_price : ℕ)
  (total_selling_price : ℕ)
  (overall_profit : ℕ) :
  purchase_price_refrigerator = 15000 →
  purchase_price_mobile_phone = 8000 →
  loss_percentage_refrigerator = 4 →
  profit_percentage_mobile_phone = 10 →
  selling_price_refrigerator = purchase_price_refrigerator - (purchase_price_refrigerator * loss_percentage_refrigerator / 100) →
  selling_price_mobile_phone = purchase_price_mobile_phone + (purchase_price_mobile_phone * profit_percentage_mobile_phone / 100) →
  total_cost_price = purchase_price_refrigerator + purchase_price_mobile_phone →
  total_selling_price = selling_price_refrigerator + selling_price_mobile_phone →
  overall_profit = total_selling_price - total_cost_price →
  overall_profit = 200 :=
  by sorry

end overall_profit_refrigerator_mobile_phone_l531_53125


namespace max_ab_l531_53177

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 40) : 
  ab ≤ 400 :=
sorry

end max_ab_l531_53177


namespace shaded_region_area_l531_53140

theorem shaded_region_area (ABCD: Type) (D B: Type) (AD CD: ℝ) 
  (h1: (AD = 5)) (h2: (CD = 12)):
  let radiusD := Real.sqrt (AD^2 + CD^2)
  let quarter_circle_area := Real.pi * radiusD^2 / 4
  let radiusC := CD / 2
  let semicircle_area := Real.pi * radiusC^2 / 2
  quarter_circle_area - semicircle_area = 97 * Real.pi / 4 :=
by sorry

end shaded_region_area_l531_53140


namespace at_most_one_existence_l531_53103

theorem at_most_one_existence
  (p : ℕ) (hp : Nat.Prime p)
  (A B : Finset (Fin p))
  (h_non_empty_A : A.Nonempty) (h_non_empty_B : B.Nonempty)
  (h_union : A ∪ B = Finset.univ) (h_disjoint : A ∩ B = ∅) :
  ∃! a : Fin p, ¬ (∃ x y : Fin p, (x ∈ A ∧ y ∈ B ∧ x + y = a) ∨ (x + y = a + p)) :=
sorry

end at_most_one_existence_l531_53103


namespace butterfat_mixture_l531_53168

theorem butterfat_mixture (x : ℝ) :
  (0.10 * x + 0.30 * 8 = 0.20 * (x + 8)) → x = 8 :=
by
  intro h
  sorry

end butterfat_mixture_l531_53168


namespace value_of_x_l531_53130

theorem value_of_x (x : ℝ) : 
  (x ≤ 0 → x^2 + 1 = 5 → x = -2) ∧ 
  (0 < x → -2 * x = 5 → false) := 
sorry

end value_of_x_l531_53130


namespace average_weight_increase_l531_53121

theorem average_weight_increase (A : ℝ) (hA : 8 * A + 20 = (80 : ℝ) + (8 * (A - (60 - 80) / 8))) :
  ((8 * A + 20) / 8) - A = (2.5 : ℝ) :=
by
  sorry

end average_weight_increase_l531_53121


namespace Ariella_has_more_savings_l531_53182

variable (Daniella_savings: ℝ) (Ariella_future_savings: ℝ) (interest_rate: ℝ) (time_years: ℝ)
variable (initial_Ariella_savings: ℝ)

-- Conditions
axiom h1 : Daniella_savings = 400
axiom h2 : Ariella_future_savings = 720
axiom h3 : interest_rate = 0.10
axiom h4 : time_years = 2

-- Assume simple interest formula for future savings
axiom simple_interest : Ariella_future_savings = initial_Ariella_savings * (1 + interest_rate * time_years)

-- Show the difference in savings
theorem Ariella_has_more_savings : initial_Ariella_savings - Daniella_savings = 200 :=
by sorry

end Ariella_has_more_savings_l531_53182


namespace smallest_value_n_l531_53180

def factorial_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125

theorem smallest_value_n
  (a b c m n : ℕ)
  (h1 : a + b + c = 2003)
  (h2 : a = 2 * b)
  (h3 : a.factorial * b.factorial * c.factorial = m * 10 ^ n)
  (h4 : ¬ (10 ∣ m)) :
  n = 400 :=
by
  sorry

end smallest_value_n_l531_53180


namespace probability_of_rain_at_least_once_l531_53194

theorem probability_of_rain_at_least_once 
  (P_sat : ℝ) (P_sun : ℝ) (P_mon : ℝ)
  (h_sat : P_sat = 0.30)
  (h_sun : P_sun = 0.60)
  (h_mon : P_mon = 0.50) :
  (1 - (1 - P_sat) * (1 - P_sun) * (1 - P_mon)) * 100 = 86 :=
by
  rw [h_sat, h_sun, h_mon]
  sorry

end probability_of_rain_at_least_once_l531_53194


namespace rhombus_area_l531_53150

theorem rhombus_area (side d1 d2 : ℝ) (h_side : side = 25) (h_d1 : d1 = 30) (h_diag : d2 = 40) :
  (d1 * d2) / 2 = 600 :=
by
  rw [h_d1, h_diag]
  norm_num

end rhombus_area_l531_53150


namespace least_x_value_l531_53186

theorem least_x_value : ∀ x : ℝ, (4 * x^2 + 7 * x + 3 = 5) → x = -2 ∨ x >= -2 := by 
    intro x
    intro h
    sorry

end least_x_value_l531_53186


namespace find_t_l531_53134

theorem find_t (t : ℝ) :
  let P := (t - 5, -2)
  let Q := (-3, t + 4)
  let M := ((t - 8) / 2, (t + 2) / 2)
  (dist M P) ^ 2 = t^2 / 3 →
  t = -12 + 2 * Real.sqrt 21 ∨ t = -12 - 2 * Real.sqrt 21 := sorry

end find_t_l531_53134


namespace part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l531_53111

-- Proof Problem for Part (a):
theorem part_a_probability_three_unused_rockets (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ q : ℝ, q = 10 * p^3 * (1-p)^2) := sorry

-- Proof Problem for Part (b):
theorem part_b_expected_targets_hit (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ e : ℝ, e = 10 * p - p^10) := sorry

end part_a_probability_three_unused_rockets_part_b_expected_targets_hit_l531_53111


namespace x_ge_y_l531_53198

variable (a : ℝ)

def x : ℝ := 2 * a * (a + 3)
def y : ℝ := (a - 3) * (a + 3)

theorem x_ge_y : x a ≥ y a := 
by 
  sorry

end x_ge_y_l531_53198


namespace value_of_p_l531_53181

noncomputable def p_value_condition (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) : Prop :=
  (9 * p^8 * q = 36 * p^7 * q^2)

theorem value_of_p (p q : ℝ) (h1 : p + q = 1) (h2 : p > 0) (h3 : q > 0) (h4 : p_value_condition p q h1 h2 h3) :
  p = 4 / 5 :=
by
  sorry

end value_of_p_l531_53181


namespace machine_working_time_l531_53147

theorem machine_working_time (total_shirts_made : ℕ) (shirts_per_minute : ℕ)
  (h1 : total_shirts_made = 196) (h2 : shirts_per_minute = 7) :
  (total_shirts_made / shirts_per_minute = 28) :=
by
  sorry

end machine_working_time_l531_53147


namespace xy_yz_zx_over_x2_y2_z2_l531_53122

theorem xy_yz_zx_over_x2_y2_z2 (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h_sum : x + y + z = 1) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end xy_yz_zx_over_x2_y2_z2_l531_53122


namespace measure_of_angle_A_l531_53116

theorem measure_of_angle_A {A B C : ℝ} (hC : C = 2 * B) (hB : B = 21) :
  A = 180 - B - C := 
by 
  sorry

end measure_of_angle_A_l531_53116


namespace total_area_of_removed_triangles_l531_53174

theorem total_area_of_removed_triangles : 
  ∀ (side_length_of_square : ℝ) (hypotenuse_length_of_triangle : ℝ),
  side_length_of_square = 20 →
  hypotenuse_length_of_triangle = 10 →
  4 * (1/2 * (hypotenuse_length_of_triangle^2 / 2)) = 100 :=
by
  intros side_length_of_square hypotenuse_length_of_triangle h_side_length h_hypotenuse_length
  -- Proof would go here, but we add "sorry" to complete the statement
  sorry

end total_area_of_removed_triangles_l531_53174


namespace calculate_land_tax_l531_53192

def plot_size : ℕ := 15
def cadastral_value_per_sotka : ℕ := 100000
def tax_rate : ℝ := 0.003

theorem calculate_land_tax :
  plot_size * cadastral_value_per_sotka * tax_rate = 4500 := 
by 
  sorry

end calculate_land_tax_l531_53192


namespace length_LL1_l531_53161

theorem length_LL1 (XZ : ℝ) (XY : ℝ) (YZ : ℝ) (X1Y : ℝ) (X1Z : ℝ) (LM : ℝ) (LN : ℝ) (MN : ℝ) (L1N : ℝ) (LL1 : ℝ) : 
  XZ = 13 → XY = 5 → 
  YZ = Real.sqrt (XZ^2 - XY^2) → 
  X1Y = 60 / 17 → 
  X1Z = 84 / 17 → 
  LM = X1Z → LN = X1Y → 
  MN = Real.sqrt (LM^2 - LN^2) → 
  (∀ k, L1N = 5 * k ∧ (7 * k + 5 * k) = MN → LL1 = 5 * k) →
  LL1 = 20 / 17 :=
by sorry

end length_LL1_l531_53161


namespace closest_multiple_of_17_to_2502_is_2499_l531_53104

def isNearestMultipleOf17 (m n : ℤ) : Prop :=
  ∃ k : ℤ, 17 * k = n ∧ abs (m - n) ≤ abs (m - 17 * (k + 1)) ∧ abs (m - n) ≤ abs (m - 17 * (k - 1))

theorem closest_multiple_of_17_to_2502_is_2499 :
  isNearestMultipleOf17 2502 2499 :=
sorry

end closest_multiple_of_17_to_2502_is_2499_l531_53104


namespace binomial_coefficient_12_4_l531_53112

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

theorem binomial_coefficient_12_4 : binomial_coefficient 12 4 = 495 := by
  sorry

end binomial_coefficient_12_4_l531_53112


namespace part_a_part_b_part_c_l531_53170

-- Part (a) Lean Statement
theorem part_a (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ k : ℝ, k = 2 * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (b) Lean Statement
theorem part_b (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) :
  (∃ q : ℝ, q = 1 - p ∧ ∃ r : ℝ, r = 2 * p / (2 * p + (1 - p) ^ 2)) :=
by
  -- Definitions and conditions would go here
  sorry

-- Part (c) Lean Statement
theorem part_c (N : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  (∃ S : ℝ, S = N * p / (p + 1)) :=
by
  -- Definitions and conditions would go here
  sorry

end part_a_part_b_part_c_l531_53170


namespace child_grandmother_ratio_l531_53163

def grandmother_weight (G D C : ℝ) : Prop :=
  G + D + C = 160

def daughter_child_weight (D C : ℝ) : Prop :=
  D + C = 60

def daughter_weight (D : ℝ) : Prop :=
  D = 40

theorem child_grandmother_ratio (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_child_weight D C) (h3 : daughter_weight D) :
  C / G = 1 / 5 :=
sorry

end child_grandmother_ratio_l531_53163


namespace base_b_representation_1987_l531_53184

theorem base_b_representation_1987 (x y z b : ℕ) (h1 : x + y + z = 25) (h2 : x ≥ 1)
  (h3 : 1987 = x * b^2 + y * b + z) (h4 : 12 < b) (h5 : b < 45) :
  x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
sorry

end base_b_representation_1987_l531_53184


namespace sin_double_angle_of_tan_pi_sub_alpha_eq_two_l531_53149

theorem sin_double_angle_of_tan_pi_sub_alpha_eq_two 
  (α : Real) 
  (h : Real.tan (Real.pi - α) = 2) : 
  Real.sin (2 * α) = -4 / 5 := 
  by sorry

end sin_double_angle_of_tan_pi_sub_alpha_eq_two_l531_53149


namespace tanker_fill_rate_l531_53107

theorem tanker_fill_rate :
  let barrels_per_min := 2
  let liters_per_barrel := 159
  let cubic_meters_per_liter := 0.001
  let minutes_per_hour := 60
  let liters_per_min := barrels_per_min * liters_per_barrel
  let liters_per_hour := liters_per_min * minutes_per_hour
  let cubic_meters_per_hour := liters_per_hour * cubic_meters_per_liter
  cubic_meters_per_hour = 19.08 :=
  by {
    sorry
  }

end tanker_fill_rate_l531_53107


namespace tan_pi_seven_product_eq_sqrt_seven_l531_53151

theorem tan_pi_seven_product_eq_sqrt_seven :
  (Real.tan (Real.pi / 7)) * (Real.tan (2 * Real.pi / 7)) * (Real.tan (3 * Real.pi / 7)) = Real.sqrt 7 :=
by
  sorry

end tan_pi_seven_product_eq_sqrt_seven_l531_53151


namespace eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l531_53145

noncomputable def relative_speed_moon_sun := (17/16 : ℝ) - (1/12 : ℝ)
noncomputable def initial_distance := (47/10 : ℝ)
noncomputable def time_coincide := initial_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_time_coincide : 
  (time_coincide - 12 : ℝ) = (2 + 1/60 : ℝ) :=
sorry

noncomputable def start_distance := (37/10 : ℝ)
noncomputable def time_start := start_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_start_time : 
  (time_start - 12 : ℝ) = (1 + 59/60 : ℝ) :=
sorry

noncomputable def end_distance := (57/10 : ℝ)
noncomputable def time_end := end_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_end_time : 
  (time_end - 12 : ℝ) = (3 + 2/60 : ℝ) :=
sorry

end eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l531_53145


namespace total_cost_is_correct_l531_53110

noncomputable def total_cost : ℝ :=
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let elephant_ear_cost := 7.00
  let purple_fountain_grass_cost := 6.00
  let pots := 6
  let sales_tax := 0.07
  let cost_one_pot := palm_fern_cost 
                   + 4 * creeping_jenny_cost 
                   + 4 * geranium_cost 
                   + 2 * elephant_ear_cost 
                   + 3 * purple_fountain_grass_cost
  let total_pots_cost := pots * cost_one_pot
  let tax := total_pots_cost * sales_tax
  total_pots_cost + tax

theorem total_cost_is_correct : total_cost = 494.34 :=
by
  -- This is where the proof would go, but we are adding sorry to skip the proof
  sorry

end total_cost_is_correct_l531_53110


namespace largest_n_exists_l531_53124

theorem largest_n_exists (n x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6 → 
  n ≤ 8 :=
sorry

end largest_n_exists_l531_53124


namespace AD_mutually_exclusive_not_complementary_l531_53129

-- Define the sets representing the outcomes of the events
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {2, 4, 6}
def D : Set ℕ := {2, 4}

-- Define mutually exclusive
def mutually_exclusive (X Y : Set ℕ) : Prop := X ∩ Y = ∅

-- Define complementary
def complementary (X Y : Set ℕ) : Prop := X ∪ Y = {1, 2, 3, 4, 5, 6}

-- The statement to prove that events A and D are mutually exclusive but not complementary
theorem AD_mutually_exclusive_not_complementary :
  mutually_exclusive A D ∧ ¬ complementary A D :=
by
  sorry

end AD_mutually_exclusive_not_complementary_l531_53129


namespace ordered_triples_count_l531_53164

theorem ordered_triples_count :
  {n : ℕ // n = 4} :=
sorry

end ordered_triples_count_l531_53164


namespace mack_writing_time_tuesday_l531_53117

variable (minutes_per_page_mon : ℕ := 30)
variable (time_mon : ℕ := 60)
variable (pages_wed : ℕ := 5)
variable (total_pages : ℕ := 10)
variable (minutes_per_page_tue : ℕ := 15)

theorem mack_writing_time_tuesday :
  (time_mon / minutes_per_page_mon) + pages_wed + (3 * minutes_per_page_tue / minutes_per_page_tue) = total_pages →
  (3 * minutes_per_page_tue) = 45 := by
  intros h
  sorry

end mack_writing_time_tuesday_l531_53117


namespace intersection_M_N_l531_53165

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_l531_53165


namespace solution_set_system_of_inequalities_l531_53123

theorem solution_set_system_of_inequalities :
  { x : ℝ | x - 2 > 1 ∧ x < 4 } = { x : ℝ| 3 < x ∧ x < 4 } :=
by
  sorry

end solution_set_system_of_inequalities_l531_53123


namespace CannotDetermineDraculaStatus_l531_53106

variable (Transylvanian_is_human : Prop)
variable (Dracula_is_alive : Prop)
variable (Statement : Transylvanian_is_human → Dracula_is_alive)

theorem CannotDetermineDraculaStatus : ¬ (∃ (H : Prop), H = Dracula_is_alive) :=
by
  sorry

end CannotDetermineDraculaStatus_l531_53106


namespace hexagon_perimeter_l531_53159

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (perimeter : ℕ) 
  (h1 : num_sides = 6)
  (h2 : side_length = 7)
  (h3 : perimeter = side_length * num_sides) : perimeter = 42 := by
  sorry

end hexagon_perimeter_l531_53159


namespace polynomial_value_at_neg2_l531_53156

def polynomial (x : ℝ) : ℝ :=
  x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2

theorem polynomial_value_at_neg2 :
  polynomial (-2) = 325.4 :=
by
  sorry

end polynomial_value_at_neg2_l531_53156


namespace range_of_m_l531_53196

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) ↔ -5 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l531_53196


namespace value_of_x_l531_53138

-- Define variables and conditions
def consecutive (x y z : ℤ) : Prop := x = z + 2 ∧ y = z + 1

-- Main proposition
theorem value_of_x (x y z : ℤ) (h1 : consecutive x y z) (h2 : z = 2) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) : x = 4 :=
by
  sorry

end value_of_x_l531_53138


namespace s_6_of_30_eq_146_over_175_l531_53191

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem s_6_of_30_eq_146_over_175 : s (s (s (s (s (s 30))))) = 146 / 175 := sorry

end s_6_of_30_eq_146_over_175_l531_53191


namespace three_digit_even_sum_12_l531_53102

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l531_53102


namespace john_initial_bench_weight_l531_53118

variable (B : ℕ)

theorem john_initial_bench_weight (B : ℕ) (HNewTotal : 1490 = 490 + B + 600) : B = 400 :=
by
  sorry

end john_initial_bench_weight_l531_53118


namespace a_and_b_together_complete_in_10_days_l531_53113

noncomputable def a_works_twice_as_fast_as_b (a b : ℝ) : Prop :=
  a = 2 * b

noncomputable def b_can_complete_work_in_30_days (b : ℝ) : Prop :=
  b = 1/30

theorem a_and_b_together_complete_in_10_days (a b : ℝ) 
  (h₁ : a_works_twice_as_fast_as_b a b)
  (h₂ : b_can_complete_work_in_30_days b) : 
  (1 / (a + b)) = 10 := 
sorry

end a_and_b_together_complete_in_10_days_l531_53113


namespace zeros_not_adjacent_probability_l531_53187

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l531_53187


namespace new_prism_volume_l531_53128

-- Define the original volume
def original_volume : ℝ := 12

-- Define the dimensions modification factors
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 3

-- Define the volume of the new prism
def new_volume := (length_factor * width_factor * height_factor) * original_volume

-- State the theorem to prove
theorem new_prism_volume : new_volume = 144 := 
by sorry

end new_prism_volume_l531_53128


namespace arithmetic_sequence_formula_l531_53114

theorem arithmetic_sequence_formula :
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n, a (n + 1) = a n + 2) → ∀ n, a n = 2 * n :=
by
  intro a
  intro h1
  intro hdiff
  sorry

end arithmetic_sequence_formula_l531_53114


namespace circle_properties_l531_53199

theorem circle_properties (C : ℝ) (hC : C = 36) :
  let r := 18 / π
  let d := 36 / π
  let A := 324 / π
  2 * π * r = 36 ∧ d = 2 * r ∧ A = π * r^2 :=
by
  sorry

end circle_properties_l531_53199


namespace nonagon_diagonals_count_eq_27_l531_53172

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l531_53172


namespace triangle_probability_l531_53175

open Classical

theorem triangle_probability :
  let a := 5
  let b := 6
  let lengths := [1, 2, 6, 11]
  let valid_third_side x := 1 < x ∧ x < 11
  let valid_lengths := lengths.filter valid_third_side
  let probability := valid_lengths.length / lengths.length
  probability = 1 / 2 :=
by {
  sorry
}

end triangle_probability_l531_53175


namespace evaluate_expression_l531_53143

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 9) : 
  2 * x^(y / 2 : ℕ) + 5 * y^(x / 2 : ℕ) = 1429 := by
  sorry

end evaluate_expression_l531_53143


namespace oldest_child_age_l531_53171

theorem oldest_child_age (x : ℕ) (h_avg : (5 + 7 + 10 + x) / 4 = 8) : x = 10 :=
by
  sorry

end oldest_child_age_l531_53171


namespace area_of_gray_region_l531_53109

def center_C : ℝ × ℝ := (4, 6)
def radius_C : ℝ := 6
def center_D : ℝ × ℝ := (14, 6)
def radius_D : ℝ := 6

theorem area_of_gray_region :
  let area_of_rectangle := (14 - 4) * 6
  let quarter_circle_area := (π * 6 ^ 2) / 4
  let area_to_subtract := 2 * quarter_circle_area
  area_of_rectangle - area_to_subtract = 60 - 18 * π := 
by {
  sorry
}

end area_of_gray_region_l531_53109


namespace exists_100_distinct_sums_l531_53155

theorem exists_100_distinct_sums : ∃ (a : Fin 100 → ℕ), (∀ i j k l : Fin 100, i ≠ j → k ≠ l → (i, j) ≠ (k, l) → a i + a j ≠ a k + a l) ∧ (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 25000) :=
by
  sorry

end exists_100_distinct_sums_l531_53155


namespace cos_8_degree_l531_53160

theorem cos_8_degree (m : ℝ) (h : Real.sin (74 * Real.pi / 180) = m) :
  Real.cos (8 * Real.pi / 180) = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_degree_l531_53160


namespace units_digit_7_pow_6_pow_5_l531_53154

def units_digit_of_power (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_6_pow_5 :
  units_digit_of_power (7 ^ (6 ^ 5)) = 1 :=
by
  -- Insert proof steps here
  sorry

end units_digit_7_pow_6_pow_5_l531_53154


namespace closest_point_to_line_l531_53127

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end closest_point_to_line_l531_53127


namespace arithmetic_sequence_first_term_l531_53120

theorem arithmetic_sequence_first_term :
  ∃ a₁ a₂ d : ℤ, a₂ = -5 ∧ d = 3 ∧ a₂ = a₁ + d ∧ a₁ = -8 :=
by
  sorry

end arithmetic_sequence_first_term_l531_53120


namespace fixed_point_of_function_l531_53189

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^(2-2) - 3) = -2 :=
by
  sorry

end fixed_point_of_function_l531_53189


namespace solve_container_capacity_l531_53126

noncomputable def container_capacity (C : ℝ) :=
  (0.75 * C - 0.35 * C = 48)

theorem solve_container_capacity : ∃ C : ℝ, container_capacity C ∧ C = 120 :=
by
  use 120
  constructor
  {
    -- Proof that 0.75 * 120 - 0.35 * 120 = 48
    sorry
  }
  -- Proof that C = 120
  sorry

end solve_container_capacity_l531_53126


namespace mowing_lawn_time_l531_53137

theorem mowing_lawn_time (pay_mow : ℝ) (rate_hour : ℝ) (time_plant : ℝ) (charge_flowers : ℝ) :
  pay_mow = 15 → rate_hour = 20 → time_plant = 2 → charge_flowers = 45 → 
  (charge_flowers + pay_mow) / rate_hour - time_plant = 1 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- This is an outline, so the actual proof steps are omitted
  sorry

end mowing_lawn_time_l531_53137


namespace square_of_third_side_l531_53119

theorem square_of_third_side (a b : ℕ) (h1 : a = 4) (h2 : b = 5) 
    (h_right_triangle : (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2)) : 
    (c = 9) ∨ (c = 41) :=
sorry

end square_of_third_side_l531_53119


namespace inequality_proof_l531_53178

theorem inequality_proof {x y : ℝ} (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
by sorry

end inequality_proof_l531_53178


namespace percentage_increase_from_boys_to_total_l531_53148

def DamesSchoolBoys : ℕ := 2000
def DamesSchoolGirls : ℕ := 5000
def TotalAttendance : ℕ := DamesSchoolBoys + DamesSchoolGirls
def PercentageIncrease (initial final : ℕ) : ℚ := ((final - initial) / initial) * 100

theorem percentage_increase_from_boys_to_total :
  PercentageIncrease DamesSchoolBoys TotalAttendance = 250 :=
by
  sorry

end percentage_increase_from_boys_to_total_l531_53148


namespace cos_five_theta_l531_53136

theorem cos_five_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : 
  Real.cos (5 * θ) = (125 * Real.sqrt 15 - 749) / 1024 := 
  sorry

end cos_five_theta_l531_53136


namespace sufficient_condition_for_inequality_l531_53157

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a > 4 :=
by 
  sorry

end sufficient_condition_for_inequality_l531_53157


namespace total_fruit_count_l531_53153

-- Define the number of oranges
def oranges : ℕ := 6

-- Define the number of apples based on the number of oranges
def apples : ℕ := oranges - 2

-- Define the number of bananas based on the number of apples
def bananas : ℕ := 3 * apples

-- Define the number of peaches based on the number of bananas
def peaches : ℕ := bananas / 2

-- Define the total number of fruits in the basket
def total_fruits : ℕ := oranges + apples + bananas + peaches

-- Prove that the total number of pieces of fruit in the basket is 28
theorem total_fruit_count : total_fruits = 28 := by
  sorry

end total_fruit_count_l531_53153


namespace abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l531_53169

theorem abs_x_minus_one_sufficient_but_not_necessary_for_quadratic (x : ℝ) :
  (|x - 1| < 2) → (x^2 - 4 * x - 5 < 0) ∧ ¬(x^2 - 4 * x - 5 < 0 → |x - 1| < 2) :=
by
  sorry

end abs_x_minus_one_sufficient_but_not_necessary_for_quadratic_l531_53169


namespace unique_non_zero_in_rows_and_cols_l531_53158

variable (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)

theorem unique_non_zero_in_rows_and_cols
  (non_neg_A : ∀ i j, 0 ≤ A i j)
  (non_sing_A : Invertible A)
  (non_neg_A_inv : ∀ i j, 0 ≤ (A⁻¹) i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end unique_non_zero_in_rows_and_cols_l531_53158


namespace second_candidate_votes_l531_53193

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) (first_candidate_votes: ℕ)
    (h1 : total_votes = 2400)
    (h2 : first_candidate_percentage = 0.80)
    (h3 : first_candidate_votes = total_votes * first_candidate_percentage) :
    total_votes - first_candidate_votes = 480 := by
    sorry

end second_candidate_votes_l531_53193


namespace price_of_lemonade_l531_53167

def costOfIngredients : ℝ := 20
def numberOfCups : ℕ := 50
def desiredProfit : ℝ := 80

theorem price_of_lemonade (price_per_cup : ℝ) :
  (costOfIngredients + desiredProfit) / numberOfCups = price_per_cup → price_per_cup = 2 :=
by
  sorry

end price_of_lemonade_l531_53167


namespace class_gpa_l531_53185

theorem class_gpa (n : ℕ) (hn : n > 0) (gpa1 : ℝ := 30) (gpa2 : ℝ := 33) : 
    (gpa1 * (n:ℝ) + gpa2 * (2 * n : ℝ)) / (3 * n : ℝ) = 32 :=
by
  sorry

end class_gpa_l531_53185


namespace find_numerical_value_l531_53131

-- Define the conditions
variables {x y z : ℝ}
axiom h1 : 3 * x - 4 * y - 2 * z = 0
axiom h2 : x + 4 * y - 20 * z = 0
axiom h3 : z ≠ 0

-- State the goal
theorem find_numerical_value : (x^2 + 4 * x * y) / (y^2 + z^2) = 2.933 :=
by
  sorry

end find_numerical_value_l531_53131


namespace product_of_integers_l531_53166

theorem product_of_integers (x y : ℕ) (h_gcd : Nat.gcd x y = 10) (h_lcm : Nat.lcm x y = 60) : x * y = 600 := by
  sorry

end product_of_integers_l531_53166


namespace cost_price_of_table_l531_53108

theorem cost_price_of_table 
  (SP : ℝ) 
  (CP : ℝ) 
  (h1 : SP = 1.24 * CP) 
  (h2 : SP = 8215) :
  CP = 6625 :=
by
  sorry

end cost_price_of_table_l531_53108


namespace solution_of_valve_problem_l531_53162

noncomputable def valve_filling_problem : Prop :=
  ∃ (x y z : ℝ), 
    (x + y + z = 1 / 2) ∧    -- Condition when all three valves are open
    (x + z = 1 / 3) ∧        -- Condition when valves X and Z are open
    (y + z = 1 / 4) ∧        -- Condition when valves Y and Z are open
    (1 / (x + y) = 2.4)      -- Required condition for valves X and Y

theorem solution_of_valve_problem : valve_filling_problem :=
sorry

end solution_of_valve_problem_l531_53162


namespace largest_whole_number_l531_53115

theorem largest_whole_number :
  ∃ x : ℕ, 9 * x - 8 < 130 ∧ (∀ y : ℕ, 9 * y - 8 < 130 → y ≤ x) ∧ x = 15 :=
sorry

end largest_whole_number_l531_53115


namespace range_of_m_l531_53144

theorem range_of_m (m x : ℝ) : 
  (2 / (x - 3) + (x + m) / (3 - x) = 2) 
  ∧ (x ≥ 0) →
  (m ≤ 8 ∧ m ≠ -1) :=
by 
  sorry

end range_of_m_l531_53144


namespace maximilian_wealth_greater_than_national_wealth_l531_53152

theorem maximilian_wealth_greater_than_national_wealth (x y z : ℝ) (h1 : 2 * x > z) (h2 : y < z) :
    x > (2 * x + y) - (x + z) :=
by
  sorry

end maximilian_wealth_greater_than_national_wealth_l531_53152


namespace student_ticket_cost_l531_53195

theorem student_ticket_cost (cost_per_student_ticket : ℝ) :
  (12 * cost_per_student_ticket + 4 * 3 = 24) → cost_per_student_ticket = 1 :=
by
  intros h
  -- We should provide a complete proof here, but for illustration, we use sorry.
  sorry

end student_ticket_cost_l531_53195


namespace sum_of_squares_not_divisible_by_13_l531_53139

theorem sum_of_squares_not_divisible_by_13
  (x y z : ℤ)
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_xz : Int.gcd x z = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_sum : (x + y + z) % 13 = 0)
  (h_prod : (x * y * z) % 13 = 0) :
  (x^2 + y^2 + z^2) % 13 ≠ 0 := by
  sorry

end sum_of_squares_not_divisible_by_13_l531_53139


namespace product_of_roots_l531_53133

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end product_of_roots_l531_53133


namespace number_of_bottle_caps_l531_53176

-- Condition: 7 bottle caps weigh exactly one ounce
def weight_of_7_caps : ℕ := 1 -- ounce

-- Condition: Josh's collection weighs 18 pounds, and 1 pound = 16 ounces
def weight_of_collection_pounds : ℕ := 18 -- pounds
def weight_of_pound_in_ounces : ℕ := 16 -- ounces per pound

-- Condition: Question translated to proof statement
theorem number_of_bottle_caps :
  (weight_of_collection_pounds * weight_of_pound_in_ounces * 7 = 2016) :=
by
  sorry

end number_of_bottle_caps_l531_53176


namespace discount_percentage_is_25_l531_53100

def piano_cost := 500
def lessons_count := 20
def lesson_price := 40
def total_paid := 1100

def lessons_cost := lessons_count * lesson_price
def total_cost := piano_cost + lessons_cost
def discount_amount := total_cost - total_paid
def discount_percentage := (discount_amount / lessons_cost) * 100

theorem discount_percentage_is_25 : discount_percentage = 25 := by
  sorry

end discount_percentage_is_25_l531_53100


namespace positive_integer_x_l531_53101

theorem positive_integer_x (x : ℕ) (hx : 15 * x = x^2 + 56) : x = 8 := by
  sorry

end positive_integer_x_l531_53101


namespace nancy_pictures_left_l531_53173

-- Given conditions stated in the problem
def picturesZoo : Nat := 49
def picturesMuseum : Nat := 8
def picturesDeleted : Nat := 38

-- The statement of the problem, proving Nancy still has 19 pictures after deletions
theorem nancy_pictures_left : (picturesZoo + picturesMuseum) - picturesDeleted = 19 := by
  sorry

end nancy_pictures_left_l531_53173


namespace age_of_B_is_23_l531_53135

-- Definitions of conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 87)
variable (h2 : A + C = 64)

-- Statement of the problem
theorem age_of_B_is_23 : B = 23 :=
by { sorry }

end age_of_B_is_23_l531_53135
