import Mathlib

namespace NUMINAMATH_GPT_percentage_increase_expenditure_l1718_171869

variable (I : ℝ) -- original income
variable (E : ℝ) -- original expenditure
variable (I_new : ℝ) -- new income
variable (S : ℝ) -- original savings
variable (S_new : ℝ) -- new savings

-- a) Conditions
def initial_spend (I : ℝ) : ℝ := 0.75 * I
def income_increased (I : ℝ) : ℝ := 1.20 * I
def savings_increased (S : ℝ) : ℝ := 1.4999999999999996 * S

-- b) Definitions relating formulated conditions
def new_expenditure (I : ℝ) : ℝ := 1.20 * I - 0.3749999999999999 * I
def original_expenditure (I : ℝ) : ℝ := 0.75 * I

-- c) Proof statement
theorem percentage_increase_expenditure :
  initial_spend I = E →
  income_increased I = I_new →
  savings_increased (0.25 * I) = S_new →
  ((new_expenditure I - original_expenditure I) / original_expenditure I) * 100 = 10 := 
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_increase_expenditure_l1718_171869


namespace NUMINAMATH_GPT_solve_for_x_l1718_171843

theorem solve_for_x (x : ℤ) (h : 20 * 14 + x = 20 + 14 * x) : x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1718_171843


namespace NUMINAMATH_GPT_sum_reciprocals_roots_l1718_171858

theorem sum_reciprocals_roots :
  (∃ p q : ℝ, p + q = 10 ∧ p * q = 3) →
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 → (1 / p) + (1 / q) = 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_roots_l1718_171858


namespace NUMINAMATH_GPT_helium_min_cost_l1718_171872

noncomputable def W (x : ℝ) : ℝ :=
  if x < 4 then 40 * (4 * x + 16 / x + 100)
  else 40 * (9 / (x * x) - 3 / x + 117)

theorem helium_min_cost :
  (∀ x, W x ≥ 4640) ∧ (W 2 = 4640) :=
by {
  sorry
}

end NUMINAMATH_GPT_helium_min_cost_l1718_171872


namespace NUMINAMATH_GPT_number_of_mixed_groups_l1718_171846

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end NUMINAMATH_GPT_number_of_mixed_groups_l1718_171846


namespace NUMINAMATH_GPT_geese_count_l1718_171892

variables (k n : ℕ)

theorem geese_count (h1 : k * n = (k + 20) * (n - 75)) (h2 : k * n = (k - 15) * (n + 100)) : n = 300 :=
by
  sorry

end NUMINAMATH_GPT_geese_count_l1718_171892


namespace NUMINAMATH_GPT_intersection_M_N_l1718_171859

open Set Real

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | ∃ α : ℝ, x = sin α}
def IntersectSet := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = IntersectSet := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1718_171859


namespace NUMINAMATH_GPT_card_distribution_count_l1718_171813

theorem card_distribution_count : 
  ∃ (methods : ℕ), methods = 18 ∧ 
  ∃ (cards : Finset ℕ),
  ∃ (envelopes : Finset (Finset ℕ)), 
  cards = {1, 2, 3, 4, 5, 6} ∧ 
  envelopes.card = 3 ∧ 
  (∀ e ∈ envelopes, (e.card = 2) ∧ ({1, 2} ⊆ e → ∃ e1 e2, {e1, e2} ∈ envelopes ∧ {e1, e2} ⊆ cards \ {1, 2})) ∧ 
  (∀ c1 ∈ cards, ∃ e ∈ envelopes, c1 ∈ e) :=
by
  sorry

end NUMINAMATH_GPT_card_distribution_count_l1718_171813


namespace NUMINAMATH_GPT_simplify_expression_l1718_171896

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂: x ≠ -3) :
  (x - 1 - 8 / (x + 1)) / ( (x + 3) / (x + 1) ) = x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1718_171896


namespace NUMINAMATH_GPT_inequality_arith_geo_mean_l1718_171874

variable (a k : ℝ)
variable (h1 : 1 ≤ k)
variable (h2 : k ≤ 3)
variable (h3 : 0 < k)

theorem inequality_arith_geo_mean (h1 : 1 ≤ k) (h2 : k ≤ 3) (h3 : 0 < k):
    ( (a + k * a) / 2 ) ^ 2 ≥ ( (a * (k * a)) ^ (1/2) ) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_arith_geo_mean_l1718_171874


namespace NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt_2_l1718_171832

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt_2_l1718_171832


namespace NUMINAMATH_GPT_water_added_l1718_171804

theorem water_added (W : ℝ) : 
  (15 + W) * 0.20833333333333336 = 3.75 → W = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_water_added_l1718_171804


namespace NUMINAMATH_GPT_positive_int_sum_square_l1718_171848

theorem positive_int_sum_square (M : ℕ) (h_pos : 0 < M) (h_eq : M^2 + M = 12) : M = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_int_sum_square_l1718_171848


namespace NUMINAMATH_GPT_food_consumption_reduction_l1718_171852

noncomputable def reduction_factor (n p : ℝ) : ℝ :=
  (n * p) / ((n - 0.05 * n) * (p + 0.2 * p))

theorem food_consumption_reduction (n p : ℝ) (h : n > 0 ∧ p > 0) :
  (1 - reduction_factor n p) * 100 = 12.28 := by
  sorry

end NUMINAMATH_GPT_food_consumption_reduction_l1718_171852


namespace NUMINAMATH_GPT_no_such_integers_exist_l1718_171800

theorem no_such_integers_exist :
  ¬ ∃ (a b : ℕ), a ≥ 1 ∧ b ≥ 1 ∧ ∃ k₁ k₂ : ℕ, (a^5 * b + 3 = k₁^3) ∧ (a * b^5 + 3 = k₂^3) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_exist_l1718_171800


namespace NUMINAMATH_GPT_certain_number_is_310_l1718_171828

theorem certain_number_is_310 (x : ℤ) (h : 3005 - x + 10 = 2705) : x = 310 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_310_l1718_171828


namespace NUMINAMATH_GPT_find_x_l1718_171895

def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

theorem find_x (x : ℝ) (h : otimes (x + 1) (x - 2) = 5) : x = 0 ∨ x = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1718_171895


namespace NUMINAMATH_GPT_no_parallelepiped_exists_l1718_171836

theorem no_parallelepiped_exists 
  (xyz_half_volume: ℝ)
  (xy_plus_yz_plus_zx_half_surface_area: ℝ) 
  (sum_of_squares_eq_4: ℝ) : 
  ¬(∃ x y z : ℝ, (x * y * z = xyz_half_volume) ∧ 
                 (x * y + y * z + z * x = xy_plus_yz_plus_zx_half_surface_area) ∧ 
                 (x^2 + y^2 + z^2 = sum_of_squares_eq_4)) := 
by
  let xyz_half_volume := 2 * Real.pi / 3
  let xy_plus_yz_plus_zx_half_surface_area := Real.pi
  let sum_of_squares_eq_4 := 4
  sorry

end NUMINAMATH_GPT_no_parallelepiped_exists_l1718_171836


namespace NUMINAMATH_GPT_abs_inequality_solution_l1718_171880

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1718_171880


namespace NUMINAMATH_GPT_simplify_expression_l1718_171867

theorem simplify_expression :
  (20^4 + 625) * (40^4 + 625) * (60^4 + 625) * (80^4 + 625) /
  (10^4 + 625) * (30^4 + 625) * (50^4 + 625) * (70^4 + 625) = 7 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1718_171867


namespace NUMINAMATH_GPT_dentist_age_considered_years_ago_l1718_171837

theorem dentist_age_considered_years_ago (A : ℕ) (X : ℕ) (H1 : A = 32) (H2 : (1/6 : ℚ) * (A - X) = (1/10 : ℚ) * (A + 8)) : X = 8 :=
sorry

end NUMINAMATH_GPT_dentist_age_considered_years_ago_l1718_171837


namespace NUMINAMATH_GPT_find_m_b_l1718_171808

theorem find_m_b (m b : ℚ) :
  (3 * m - 14 = 2) ∧ (m ^ 2 - 6 * m + 15 = b) →
  m = 16 / 3 ∧ b = 103 / 9 := by
  intro h
  rcases h with ⟨h1, h2⟩
  -- proof steps here
  sorry

end NUMINAMATH_GPT_find_m_b_l1718_171808


namespace NUMINAMATH_GPT_proposition_is_false_l1718_171893

noncomputable def false_proposition : Prop :=
¬(∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), Real.sin x + Real.cos x ≥ 2)

theorem proposition_is_false : false_proposition :=
by
  sorry

end NUMINAMATH_GPT_proposition_is_false_l1718_171893


namespace NUMINAMATH_GPT_measure_of_angle_A_l1718_171847

variables (A B C a b c : ℝ)
variables (triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variables (sides_relation : (a^2 + b^2 - c^2) * tan A = a * b)

theorem measure_of_angle_A :
  A = π / 6 :=
by 
  sorry

end NUMINAMATH_GPT_measure_of_angle_A_l1718_171847


namespace NUMINAMATH_GPT_greatest_possible_mean_BC_l1718_171866

theorem greatest_possible_mean_BC :
  ∀ (A_n B_n C_weight C_n : ℕ),
    (A_n > 0) ∧ (B_n > 0) ∧ (C_n > 0) ∧
    (40 * A_n + 50 * B_n) / (A_n + B_n) = 43 ∧
    (40 * A_n + C_weight) / (A_n + C_n) = 44 →
    ∃ k : ℕ, ∃ n : ℕ, 
      A_n = 7 * k ∧ B_n = 3 * k ∧ 
      C_weight = 28 * k + 44 * n ∧ 
      44 + 46 * k / (3 * k + n) ≤ 59 :=
sorry

end NUMINAMATH_GPT_greatest_possible_mean_BC_l1718_171866


namespace NUMINAMATH_GPT_cos_angle_sum_eq_negative_sqrt_10_div_10_l1718_171812

theorem cos_angle_sum_eq_negative_sqrt_10_div_10 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (α + π / 4) = - Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_GPT_cos_angle_sum_eq_negative_sqrt_10_div_10_l1718_171812


namespace NUMINAMATH_GPT_range_of_x_l1718_171860

theorem range_of_x (x : ℝ) : (x^2 - 9*x + 14 < 0) ∧ (2*x + 3 > 0) ↔ (2 < x) ∧ (x < 7) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l1718_171860


namespace NUMINAMATH_GPT_Kate_has_223_pennies_l1718_171821

-- Definition of the conditions
variables (J K : ℕ)
variable (h1 : J = 388)
variable (h2 : J = K + 165)

-- Prove the question equals the answer
theorem Kate_has_223_pennies : K = 223 :=
by
  sorry

end NUMINAMATH_GPT_Kate_has_223_pennies_l1718_171821


namespace NUMINAMATH_GPT_combination_addition_l1718_171851

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem combination_addition :
  combination 13 11 + 3 = 81 :=
by
  sorry

end NUMINAMATH_GPT_combination_addition_l1718_171851


namespace NUMINAMATH_GPT_simplify_and_sum_coefficients_l1718_171891

theorem simplify_and_sum_coefficients :
  (∃ A B C D : ℤ, (∀ x : ℝ, x ≠ D → (x^3 + 6 * x^2 + 11 * x + 6) / (x + 1) = A * x^2 + B * x + C) ∧ A + B + C + D = 11) :=
sorry

end NUMINAMATH_GPT_simplify_and_sum_coefficients_l1718_171891


namespace NUMINAMATH_GPT_max_band_members_l1718_171806

variable (r x m : ℕ)

noncomputable def band_formation (r x m: ℕ) :=
  m = r * x + 4 ∧
  m = (r - 3) * (x + 2) ∧
  m < 100

theorem max_band_members (r x m : ℕ) (h : band_formation r x m) : m = 88 :=
by
  sorry

end NUMINAMATH_GPT_max_band_members_l1718_171806


namespace NUMINAMATH_GPT_problem1_problem2_l1718_171830

open Set Real

-- Given A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {y | y > -1}

-- Problem 1: If A = B, then a = -1
theorem problem1 (a : ℝ) (h : A a = B) : a = -1 := by
  sorry

-- Problem 2: If (complement of A) ∩ B ≠ ∅, find the range of a
theorem problem2 (a : ℝ) (h : (compl (A a)) ∩ B ≠ ∅) : a ∈ Ioi (-1) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1718_171830


namespace NUMINAMATH_GPT_domain_log2_x_minus_1_l1718_171839

theorem domain_log2_x_minus_1 (x : ℝ) : (1 < x) ↔ (∃ y : ℝ, y = Real.logb 2 (x - 1)) := by
  sorry

end NUMINAMATH_GPT_domain_log2_x_minus_1_l1718_171839


namespace NUMINAMATH_GPT_average_weight_women_l1718_171819

variable (average_weight_men : ℕ) (number_of_men : ℕ)
variable (average_weight : ℕ) (number_of_women : ℕ)
variable (average_weight_all : ℕ) (total_people : ℕ)

theorem average_weight_women (h1 : average_weight_men = 190) 
                            (h2 : number_of_men = 8)
                            (h3 : average_weight_all = 160)
                            (h4 : total_people = 14) 
                            (h5 : number_of_women = 6):
  average_weight = 120 := 
by
  sorry

end NUMINAMATH_GPT_average_weight_women_l1718_171819


namespace NUMINAMATH_GPT_evaluate_expr_l1718_171831

noncomputable def expr : ℚ :=
  2013 * (5.7 * 4.2 + (21 / 5) * 4.3) / ((14 / 73) * 15 + (5 / 73) * 177 + 656)

theorem evaluate_expr : expr = 126 := by
  sorry

end NUMINAMATH_GPT_evaluate_expr_l1718_171831


namespace NUMINAMATH_GPT_wedge_product_correct_l1718_171855

variables {a1 a2 b1 b2 : ℝ}
def a : ℝ × ℝ := (a1, a2)
def b : ℝ × ℝ := (b1, b2)

def wedge_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.2 - v.2 * w.1

theorem wedge_product_correct (a b : ℝ × ℝ) :
  wedge_product a b = a.1 * b.2 - a.2 * b.1 :=
by
  -- Proof is omitted, theorem statement only
  sorry

end NUMINAMATH_GPT_wedge_product_correct_l1718_171855


namespace NUMINAMATH_GPT_cost_of_each_box_is_8_33_l1718_171840

noncomputable def cost_per_box (boxes pens_per_box pens_packaged price_per_packaged price_per_set profit_total : ℕ) : ℝ :=
  let total_pens := boxes * pens_per_box
  let packaged_pens := pens_packaged * pens_per_box
  let packages := packaged_pens / 6
  let revenue_packages := packages * price_per_packaged
  let remaining_pens := total_pens - packaged_pens
  let sets := remaining_pens / 3
  let revenue_sets := sets * price_per_set
  let total_revenue := revenue_packages + revenue_sets
  let cost_total := total_revenue - profit_total
  cost_total / boxes

theorem cost_of_each_box_is_8_33 :
  cost_per_box 12 30 5 3 2 115 = 100 / 12 :=
by
  unfold cost_per_box
  sorry

end NUMINAMATH_GPT_cost_of_each_box_is_8_33_l1718_171840


namespace NUMINAMATH_GPT_rational_expression_equals_3_l1718_171841

theorem rational_expression_equals_3 (x : ℝ) (hx : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := 
by
  sorry

end NUMINAMATH_GPT_rational_expression_equals_3_l1718_171841


namespace NUMINAMATH_GPT_kelly_needs_to_give_away_l1718_171865

-- Definition of initial number of Sony games and desired number of Sony games left
def initial_sony_games : ℕ := 132
def desired_remaining_sony_games : ℕ := 31

-- The main theorem: The number of Sony games Kelly needs to give away to have 31 left
theorem kelly_needs_to_give_away : initial_sony_games - desired_remaining_sony_games = 101 := by
  sorry

end NUMINAMATH_GPT_kelly_needs_to_give_away_l1718_171865


namespace NUMINAMATH_GPT_dacid_average_l1718_171857

noncomputable def average (a b : ℕ) : ℚ :=
(a + b) / 2

noncomputable def overall_average (a b c d e : ℕ) : ℚ :=
(a + b + c + d + e) / 5

theorem dacid_average :
  ∀ (english mathematics physics chemistry biology : ℕ),
  english = 86 →
  mathematics = 89 →
  physics = 82 →
  chemistry = 87 →
  biology = 81 →
  (average english mathematics < 90) ∧
  (average english physics < 90) ∧
  (average english chemistry < 90) ∧
  (average english biology < 90) ∧
  (average mathematics physics < 90) ∧
  (average mathematics chemistry < 90) ∧
  (average mathematics biology < 90) ∧
  (average physics chemistry < 90) ∧
  (average physics biology < 90) ∧
  (average chemistry biology < 90) ∧
  overall_average english mathematics physics chemistry biology = 85 := by
  intros english mathematics physics chemistry biology
  intros h_english h_mathematics h_physics h_chemistry h_biology
  simp [average, overall_average]
  rw [h_english, h_mathematics, h_physics, h_chemistry, h_biology]
  sorry

end NUMINAMATH_GPT_dacid_average_l1718_171857


namespace NUMINAMATH_GPT_calculate_drift_l1718_171898

theorem calculate_drift (w v t : ℝ) (hw : w = 400) (hv : v = 10) (ht : t = 50) : v * t - w = 100 :=
by
  sorry

end NUMINAMATH_GPT_calculate_drift_l1718_171898


namespace NUMINAMATH_GPT_david_remaining_money_l1718_171888

-- Given conditions
def hourly_rate : ℕ := 14
def hours_per_day : ℕ := 2
def days_in_week : ℕ := 7
def weekly_earnings : ℕ := hourly_rate * hours_per_day * days_in_week
def cost_of_shoes : ℕ := weekly_earnings / 2
def remaining_after_shoes : ℕ := weekly_earnings - cost_of_shoes
def given_to_mom : ℕ := remaining_after_shoes / 2
def remaining_after_gift : ℕ := remaining_after_shoes - given_to_mom

-- Theorem
theorem david_remaining_money : remaining_after_gift = 49 := by
  sorry

end NUMINAMATH_GPT_david_remaining_money_l1718_171888


namespace NUMINAMATH_GPT_cost_per_pack_l1718_171879

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end NUMINAMATH_GPT_cost_per_pack_l1718_171879


namespace NUMINAMATH_GPT_cone_base_radius_l1718_171899

/--
Given a cone with the following properties:
1. The surface area of the cone is \(3\pi\).
2. The lateral surface of the cone unfolds into a semicircle (which implies the slant height is twice the radius of the base).
Prove that the radius of the base of the cone is \(1\).
-/
theorem cone_base_radius 
  (S : ℝ)
  (r l : ℝ)
  (h1 : S = 3 * Real.pi)
  (h2 : l = 2 * r)
  : r = 1 := 
  sorry

end NUMINAMATH_GPT_cone_base_radius_l1718_171899


namespace NUMINAMATH_GPT_cirrus_to_cumulus_is_four_l1718_171856

noncomputable def cirrus_to_cumulus_ratio (Ci Cu Cb : ℕ) : ℕ :=
  Ci / Cu

theorem cirrus_to_cumulus_is_four :
  ∀ (Ci Cu Cb : ℕ), (Cb = 3) → (Cu = 12 * Cb) → (Ci = 144) → cirrus_to_cumulus_ratio Ci Cu Cb = 4 :=
by
  intros Ci Cu Cb hCb hCu hCi
  sorry

end NUMINAMATH_GPT_cirrus_to_cumulus_is_four_l1718_171856


namespace NUMINAMATH_GPT_below_sea_level_notation_l1718_171824

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end NUMINAMATH_GPT_below_sea_level_notation_l1718_171824


namespace NUMINAMATH_GPT_distinct_values_f_in_interval_l1718_171884

noncomputable def f (x : ℝ) : ℤ :=
  ⌊x⌋ + ⌊2 * x⌋ + ⌊(5 * x) / 3⌋ + ⌊3 * x⌋ + ⌊4 * x⌋

theorem distinct_values_f_in_interval : 
  ∃ n : ℕ, n = 734 ∧ 
    ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 100 → 
      f x = f y → x = y :=
sorry

end NUMINAMATH_GPT_distinct_values_f_in_interval_l1718_171884


namespace NUMINAMATH_GPT_factor_expression_l1718_171881

variable (x : ℕ)

theorem factor_expression : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l1718_171881


namespace NUMINAMATH_GPT_right_triangle_345_l1718_171861

theorem right_triangle_345 : 
  (∃ a b c : ℕ, a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 2 ∧ b = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 4 ∧ b = 5 ∧ c = 6 ∧ a^2 + b^2 = c^2) ∧ 
  ¬(∃ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 9 ∧ a^2 + b^2 = c^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_345_l1718_171861


namespace NUMINAMATH_GPT_tetrahedron_volume_l1718_171897

theorem tetrahedron_volume (h_1 h_2 h_3 : ℝ) (V : ℝ)
  (h1_pos : 0 < h_1) (h2_pos : 0 < h_2) (h3_pos : 0 < h_3)
  (V_nonneg : 0 ≤ V) : 
  V ≥ (1 / 3) * h_1 * h_2 * h_3 := sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1718_171897


namespace NUMINAMATH_GPT_quadratic_inequality_l1718_171810

theorem quadratic_inequality (x : ℝ) : x^2 - x + 1 ≥ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l1718_171810


namespace NUMINAMATH_GPT_carnations_percentage_l1718_171811

-- Definition of the total number of flowers
def total_flowers (F : ℕ) : Prop := 
  F > 0

-- Definition of the pink roses condition
def pink_roses_condition (F : ℕ) : Prop := 
  (1 / 2) * (3 / 5) * F = (3 / 10) * F

-- Definition of the red carnations condition
def red_carnations_condition (F : ℕ) : Prop := 
  (1 / 3) * (2 / 5) * F = (2 / 15) * F

-- Definition of the total pink flowers
def pink_flowers_condition (F : ℕ) : Prop :=
  (3 / 5) * F > 0

-- Proof that the percentage of the flowers that are carnations is 50%
theorem carnations_percentage (F : ℕ) (h_total : total_flowers F) (h_pink_roses : pink_roses_condition F) (h_red_carnations : red_carnations_condition F) (h_pink_flowers : pink_flowers_condition F) :
  (1 / 2) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_carnations_percentage_l1718_171811


namespace NUMINAMATH_GPT_floor_function_solution_l1718_171803

def floor_eq_x_solutions : Prop :=
  ∀ x : ℤ, (⌊(x : ℝ) / 2⌋ + ⌊(x : ℝ) / 4⌋ = x) ↔ x = 0 ∨ x = -3 ∨ x = -2 ∨ x = -5

theorem floor_function_solution: floor_eq_x_solutions :=
by
  intro x
  sorry

end NUMINAMATH_GPT_floor_function_solution_l1718_171803


namespace NUMINAMATH_GPT_cheese_stick_problem_l1718_171894

theorem cheese_stick_problem (cheddar pepperjack mozzarella : ℕ) (total : ℕ)
    (h1 : cheddar = 15)
    (h2 : pepperjack = 45)
    (h3 : 2 * pepperjack = total)
    (h4 : total = cheddar + pepperjack + mozzarella) :
    mozzarella = 30 :=
by
    sorry

end NUMINAMATH_GPT_cheese_stick_problem_l1718_171894


namespace NUMINAMATH_GPT_changed_answers_percentage_l1718_171814

variables (n : ℕ) (a b c d : ℕ)

theorem changed_answers_percentage (h1 : a + b + c + d = 100)
  (h2 : a + d + c = 50)
  (h3 : a + c = 60)
  (h4 : b + d = 40) :
  10 ≤ c + d ∧ c + d ≤ 90 :=
  by sorry

end NUMINAMATH_GPT_changed_answers_percentage_l1718_171814


namespace NUMINAMATH_GPT_trapezoid_area_l1718_171809

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end NUMINAMATH_GPT_trapezoid_area_l1718_171809


namespace NUMINAMATH_GPT_inscribed_circle_theta_l1718_171834

/-- Given that a circle inscribed in triangle ABC is tangent to sides BC, CA, and AB at points
    where the tangential angles are 120 degrees, 130 degrees, and theta degrees respectively,
    we need to prove that theta is 110 degrees. -/
theorem inscribed_circle_theta 
  (ABC : Type)
  (A B C : ABC)
  (theta : ℝ)
  (tangent_angle_BC : ℝ)
  (tangent_angle_CA : ℝ) 
  (tangent_angle_AB : ℝ) 
  (h1 : tangent_angle_BC = 120)
  (h2 : tangent_angle_CA = 130) 
  (h3 : tangent_angle_AB = theta) : 
  theta = 110 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_theta_l1718_171834


namespace NUMINAMATH_GPT_xiao_xuan_wins_l1718_171853

def cards_game (n : ℕ) (min_take : ℕ) (max_take : ℕ) (initial_turn : String) : String :=
  if initial_turn = "Xiao Liang" then "Xiao Xuan" else "Xiao Liang"

theorem xiao_xuan_wins :
  cards_game 17 1 2 "Xiao Liang" = "Xiao Xuan" :=
sorry

end NUMINAMATH_GPT_xiao_xuan_wins_l1718_171853


namespace NUMINAMATH_GPT_line_through_point_intersects_yaxis_triangular_area_l1718_171817

theorem line_through_point_intersects_yaxis_triangular_area 
  (a T : ℝ) 
  (h : 0 < a) 
  (line_eqn : ∀ x y : ℝ, x = -a * y + a → 2 * T * x + a^2 * y - 2 * a * T = 0) 
  : ∃ (m b : ℝ), (forall x y : ℝ, y = m * x + b) := 
by
  sorry

end NUMINAMATH_GPT_line_through_point_intersects_yaxis_triangular_area_l1718_171817


namespace NUMINAMATH_GPT_infinitely_many_n_l1718_171825

theorem infinitely_many_n (h : ℤ) : ∃ (S : Set ℤ), S ≠ ∅ ∧ ∀ n ∈ S, ∃ k : ℕ, ⌊n * Real.sqrt (h^2 + 1)⌋ = k^2 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_n_l1718_171825


namespace NUMINAMATH_GPT_mod_inverse_9_mod_23_l1718_171823

theorem mod_inverse_9_mod_23 : ∃ (a : ℤ), 0 ≤ a ∧ a < 23 ∧ (9 * a) % 23 = 1 :=
by
  use 18
  sorry

end NUMINAMATH_GPT_mod_inverse_9_mod_23_l1718_171823


namespace NUMINAMATH_GPT_original_inhabitants_proof_l1718_171876

noncomputable def original_inhabitants (final_population : ℕ) : ℝ :=
  final_population / (0.75 * 0.9)

theorem original_inhabitants_proof :
  original_inhabitants 5265 = 7800 :=
by
  sorry

end NUMINAMATH_GPT_original_inhabitants_proof_l1718_171876


namespace NUMINAMATH_GPT_longest_side_of_rectangular_solid_l1718_171882

theorem longest_side_of_rectangular_solid 
  (x y z : ℝ) 
  (h1 : x * y = 20) 
  (h2 : y * z = 15) 
  (h3 : x * z = 12) 
  (h4 : x * y * z = 60) : 
  max (max x y) z = 10 := 
by sorry

end NUMINAMATH_GPT_longest_side_of_rectangular_solid_l1718_171882


namespace NUMINAMATH_GPT_belt_length_sufficient_l1718_171827

theorem belt_length_sufficient (r O_1O_2 O_1O_3 O_3_plane : ℝ) 
(O_1O_2_eq : O_1O_2 = 12) (O_1O_3_eq : O_1O_3 = 10) (O_3_plane_eq : O_3_plane = 8) (r_eq : r = 2) : 
(∃ L₁ L₂, L₁ = 32 + 4 * Real.pi ∧ L₂ = 22 + 2 * Real.sqrt 97 + 4 * Real.pi ∧ 
L₁ ≠ 54 ∧ L₂ > 54) := 
by 
  sorry

end NUMINAMATH_GPT_belt_length_sufficient_l1718_171827


namespace NUMINAMATH_GPT_genetic_recombination_does_not_occur_during_dna_replication_l1718_171844

-- Definitions based on conditions
def dna_replication_spermatogonial_cells : Prop := 
  ∃ dna_interphase: Prop, ∃ dna_unwinding: Prop, 
    ∃ gene_mutation: Prop, ∃ protein_synthesis: Prop,
      dna_interphase ∧ dna_unwinding ∧ gene_mutation ∧ protein_synthesis

def genetic_recombination_not_occur : Prop :=
  ¬ ∃ genetic_recombination: Prop, genetic_recombination

-- Proof problem statement
theorem genetic_recombination_does_not_occur_during_dna_replication : 
  dna_replication_spermatogonial_cells → genetic_recombination_not_occur :=
by sorry

end NUMINAMATH_GPT_genetic_recombination_does_not_occur_during_dna_replication_l1718_171844


namespace NUMINAMATH_GPT_calculate_teena_speed_l1718_171802

noncomputable def Teena_speed (t c t_ahead_in_1_5_hours : ℝ) : ℝ :=
  let distance_initial_gap := 7.5
  let coe_speed := 40
  let time_in_hours := 1.5
  let distance_coe_travels := coe_speed * time_in_hours
  let total_distance_teena_needs := distance_coe_travels + distance_initial_gap + t_ahead_in_1_5_hours
  total_distance_teena_needs / time_in_hours

theorem calculate_teena_speed :
  (Teena_speed 7.5 40 15) = 55 :=
  by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_calculate_teena_speed_l1718_171802


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l1718_171820

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l1718_171820


namespace NUMINAMATH_GPT_sum_of_ages_now_l1718_171875

variable (D A Al B : ℝ)

noncomputable def age_condition (D : ℝ) : Prop :=
  D = 16

noncomputable def alex_age_condition (A : ℝ) : Prop :=
  A = 60 - (30 - 16)

noncomputable def allison_age_condition (Al : ℝ) : Prop :=
  Al = 15 - (30 - 16)

noncomputable def bernard_age_condition (B A Al : ℝ) : Prop :=
  B = (A + Al) / 2

noncomputable def sum_of_ages (A Al B : ℝ) : ℝ :=
  A + Al + B

theorem sum_of_ages_now :
  age_condition D →
  alex_age_condition A →
  allison_age_condition Al →
  bernard_age_condition B A Al →
  sum_of_ages A Al B = 70.5 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_now_l1718_171875


namespace NUMINAMATH_GPT_remainders_mod_m_l1718_171816

theorem remainders_mod_m {m n b : ℤ} (h_coprime : Int.gcd m n = 1) :
    (∀ r : ℤ, 0 ≤ r ∧ r < m → ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ ((b + k * n) % m = r)) :=
by
  sorry

end NUMINAMATH_GPT_remainders_mod_m_l1718_171816


namespace NUMINAMATH_GPT_calories_consumed_l1718_171870

-- Define the conditions
def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

-- Define the theorem to be proved
theorem calories_consumed (pages_written : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  (pages_written / pages_per_donut) * calories_per_donut

-- Ensure the theorem corresponds to the correct answer
example : calories_consumed 12 2 150 = 900 := by
  sorry

end NUMINAMATH_GPT_calories_consumed_l1718_171870


namespace NUMINAMATH_GPT_f_f_1_equals_4_l1718_171807

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2) else x^2 + 2

theorem f_f_1_equals_4 : f (f 1) = 4 := by sorry

end NUMINAMATH_GPT_f_f_1_equals_4_l1718_171807


namespace NUMINAMATH_GPT_count_polynomials_with_three_integer_roots_l1718_171890

def polynomial_with_roots (n: ℕ) : Nat :=
  have h: n = 8 := by
    sorry
  if n = 8 then
    -- Apply the combinatorial argument as discussed
    52
  else
    -- Case for other n
    0

theorem count_polynomials_with_three_integer_roots:
  polynomial_with_roots 8 = 52 := 
  sorry

end NUMINAMATH_GPT_count_polynomials_with_three_integer_roots_l1718_171890


namespace NUMINAMATH_GPT_num_circles_rectangle_l1718_171838

structure Rectangle (α : Type*) [Field α] :=
  (A B C D : α × α)
  (AB_parallel_CD : B.1 = A.1 ∧ D.1 = C.1)
  (AD_parallel_BC : D.2 = A.2 ∧ C.2 = B.2)

def num_circles_with_diameter_vertices (R : Rectangle ℝ) : ℕ :=
  sorry

theorem num_circles_rectangle (R : Rectangle ℝ) : num_circles_with_diameter_vertices R = 5 :=
  sorry

end NUMINAMATH_GPT_num_circles_rectangle_l1718_171838


namespace NUMINAMATH_GPT_intersection_point_of_lines_l1718_171805

theorem intersection_point_of_lines : 
  (∃ x y : ℚ, (8 * x - 3 * y = 5) ∧ (5 * x + 2 * y = 20)) ↔ (x = 70 / 31 ∧ y = 135 / 31) :=
sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l1718_171805


namespace NUMINAMATH_GPT_simplify_expression_l1718_171850

theorem simplify_expression : 
  (Real.sqrt 12) + (Real.sqrt 4) * ((Real.sqrt 5 - Real.pi) ^ 0) - (abs (-2 * Real.sqrt 3)) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1718_171850


namespace NUMINAMATH_GPT_problem_sol_l1718_171889

theorem problem_sol (a b : ℝ) (h : ∀ x, (x > -1 ∧ x < 1/3) ↔ (ax^2 + bx + 1 > 0)) : a * b = 6 :=
sorry

end NUMINAMATH_GPT_problem_sol_l1718_171889


namespace NUMINAMATH_GPT_surveys_from_retired_is_12_l1718_171887

-- Define the given conditions
def ratio_retired : ℕ := 2
def ratio_current : ℕ := 8
def ratio_students : ℕ := 40
def total_surveys : ℕ := 300
def total_ratio : ℕ := ratio_retired + ratio_current + ratio_students

-- Calculate the expected number of surveys from retired faculty
def number_of_surveys_retired : ℕ := total_surveys * ratio_retired / total_ratio

-- Lean 4 statement for proof
theorem surveys_from_retired_is_12 :
  number_of_surveys_retired = 12 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_surveys_from_retired_is_12_l1718_171887


namespace NUMINAMATH_GPT_amounts_are_correct_l1718_171815

theorem amounts_are_correct (P Q R S : ℕ) 
    (h1 : P + Q + R + S = 10000)
    (h2 : R = 2 * P)
    (h3 : R = 3 * Q)
    (h4 : S = P + Q) :
    P = 1875 ∧ Q = 1250 ∧ R = 3750 ∧ S = 3125 := by
  sorry

end NUMINAMATH_GPT_amounts_are_correct_l1718_171815


namespace NUMINAMATH_GPT_rearrange_digits_to_perfect_square_l1718_171873

theorem rearrange_digits_to_perfect_square :
  ∃ n : ℤ, 2601 = n ^ 2 ∧ (∃ (perm : List ℤ), perm = [2, 0, 1, 6] ∧ perm.permutations ≠ List.nil) :=
by
  sorry

end NUMINAMATH_GPT_rearrange_digits_to_perfect_square_l1718_171873


namespace NUMINAMATH_GPT_find_constants_for_B_l1718_171826
open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2, 4], ![2, 0, 2], ![4, 2, 0]]

def I3 : Matrix (Fin 3) (Fin 3) ℝ := 1

def zeros : Matrix (Fin 3) (Fin 3) ℝ := 0

theorem find_constants_for_B : 
  ∃ (s t u : ℝ), s = 0 ∧ t = -36 ∧ u = -48 ∧ (B^3 + s • B^2 + t • B + u • I3 = zeros) :=
sorry

end NUMINAMATH_GPT_find_constants_for_B_l1718_171826


namespace NUMINAMATH_GPT_derek_percentage_difference_l1718_171871

-- Definitions and assumptions based on conditions
def average_score_first_test (A : ℝ) : ℝ := A

def derek_score_first_test (D1 : ℝ) (A : ℝ) : Prop := D1 = 0.5 * A

def derek_score_second_test (D2 : ℝ) (D1 : ℝ) : Prop := D2 = 1.5 * D1

-- Theorem statement
theorem derek_percentage_difference (A D1 D2 : ℝ)
  (h1 : derek_score_first_test D1 A)
  (h2 : derek_score_second_test D2 D1) :
  (A - D2) / A * 100 = 25 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_derek_percentage_difference_l1718_171871


namespace NUMINAMATH_GPT_gain_percent_l1718_171818

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 115) : 
  ((SP - CP) / CP) * 100 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_gain_percent_l1718_171818


namespace NUMINAMATH_GPT_retail_price_before_discount_l1718_171835

variable (R : ℝ) -- Let R be the retail price of each machine before the discount

theorem retail_price_before_discount :
    let wholesale_price := 126
    let machines := 10
    let bulk_discount_rate := 0.05
    let profit_margin := 0.20
    let sales_tax_rate := 0.07
    let discount_rate := 0.10

    -- Calculate wholesale total price
    let wholesale_total := machines * wholesale_price

    -- Calculate bulk purchase discount
    let bulk_discount := bulk_discount_rate * wholesale_total

    -- Calculate total amount paid
    let amount_paid := wholesale_total - bulk_discount

    -- Calculate profit per machine
    let profit_per_machine := profit_margin * wholesale_price
    
    -- Calculate total profit
    let total_profit := machines * profit_per_machine

    -- Calculate sales tax on profit
    let tax_on_profit := sales_tax_rate * total_profit

    -- Calculate total amount after paying tax
    let total_amount_after_tax := (amount_paid + total_profit) - tax_on_profit

    -- Express total selling price after discount
    let total_selling_after_discount := machines * (0.90 * R)

    -- Total selling price after discount is equal to total amount after tax
    (9 * R = total_amount_after_tax) →
    R = 159.04 :=
by
  sorry

end NUMINAMATH_GPT_retail_price_before_discount_l1718_171835


namespace NUMINAMATH_GPT_find_complex_Z_l1718_171877

open Complex

theorem find_complex_Z (Z : ℂ) (h : (2 + 4 * I) / Z = 1 - I) : 
  Z = -1 + 3 * I :=
by
  sorry

end NUMINAMATH_GPT_find_complex_Z_l1718_171877


namespace NUMINAMATH_GPT_savings_percentage_correct_l1718_171801

variables (price_jacket : ℕ) (price_shirt : ℕ) (price_hat : ℕ)
          (discount_jacket : ℕ) (discount_shirt : ℕ) (discount_hat : ℕ)

def original_total_cost (price_jacket price_shirt price_hat : ℕ) : ℕ :=
  price_jacket + price_shirt + price_hat

def savings (price : ℕ) (discount : ℕ) : ℕ :=
  price * discount / 100

def total_savings (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  (savings price_jacket discount_jacket) + (savings price_shirt discount_shirt) + (savings price_hat discount_hat)

def total_savings_percentage (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  total_savings price_jacket price_shirt price_hat discount_jacket discount_shirt discount_hat * 100 /
  original_total_cost price_jacket price_shirt price_hat

theorem savings_percentage_correct :
  total_savings_percentage 100 50 30 30 60 50 = 4167 / 100 :=
sorry

end NUMINAMATH_GPT_savings_percentage_correct_l1718_171801


namespace NUMINAMATH_GPT_volume_of_prism_l1718_171854

theorem volume_of_prism (a : ℝ) (h_pos : 0 < a) (h_lat : ∀ S_lat, S_lat = a ^ 2) : 
  ∃ V, V = (a ^ 3 * (Real.sqrt 2 - 1)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1718_171854


namespace NUMINAMATH_GPT_functional_eq_solution_l1718_171845

theorem functional_eq_solution (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m + f n) = f m + n) : ∀ n, f n = n := 
by
  sorry

end NUMINAMATH_GPT_functional_eq_solution_l1718_171845


namespace NUMINAMATH_GPT_cost_price_marked_price_ratio_l1718_171862

theorem cost_price_marked_price_ratio (x : ℝ) (hx : x > 0) :
  let selling_price := (2 / 3) * x
  let cost_price := (3 / 4) * selling_price 
  cost_price / x = 1 / 2 := 
by
  let selling_price := (2 / 3) * x 
  let cost_price := (3 / 4) * selling_price 
  have hs : selling_price = (2 / 3) * x := rfl 
  have hc : cost_price = (3 / 4) * selling_price := rfl 
  have ratio := hc.symm 
  simp [ratio, hs]
  sorry

end NUMINAMATH_GPT_cost_price_marked_price_ratio_l1718_171862


namespace NUMINAMATH_GPT_problem_statement_l1718_171822

-- Definition of operation nabla
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- Main theorem statement
theorem problem_statement : nabla 2 (nabla 0 (nabla 1 7)) = 71859 :=
by
  -- Computational proof
  sorry

end NUMINAMATH_GPT_problem_statement_l1718_171822


namespace NUMINAMATH_GPT_SammyFinishedProblems_l1718_171878

def initial : ℕ := 9 -- number of initial math problems
def remaining : ℕ := 7 -- number of remaining math problems
def finished (init rem : ℕ) : ℕ := init - rem -- defining number of finished problems

theorem SammyFinishedProblems : finished initial remaining = 2 := by
  sorry -- placeholder for proof

end NUMINAMATH_GPT_SammyFinishedProblems_l1718_171878


namespace NUMINAMATH_GPT_count_five_digit_numbers_with_digit_8_l1718_171868

theorem count_five_digit_numbers_with_digit_8 : 
    let total_numbers := 99999 - 10000 + 1
    let without_8 := 8 * (9 ^ 4)
    90000 - without_8 = 37512 := by
    let total_numbers := 99999 - 10000 + 1 -- Total number of five-digit numbers
    let without_8 := 8 * (9 ^ 4) -- Number of five-digit numbers without any '8'
    show total_numbers - without_8 = 37512
    sorry

end NUMINAMATH_GPT_count_five_digit_numbers_with_digit_8_l1718_171868


namespace NUMINAMATH_GPT_orange_marbles_l1718_171886

-- Definitions based on the given conditions
def total_marbles : ℕ := 24
def blue_marbles : ℕ := total_marbles / 2
def red_marbles : ℕ := 6

-- The statement to prove: the number of orange marbles is 6
theorem orange_marbles : (total_marbles - (blue_marbles + red_marbles)) = 6 := 
  by 
  sorry

end NUMINAMATH_GPT_orange_marbles_l1718_171886


namespace NUMINAMATH_GPT_lino_shells_l1718_171833

theorem lino_shells (picked_up : ℝ) (put_back : ℝ) (remaining_shells : ℝ) :
  picked_up = 324.0 → 
  put_back = 292.0 → 
  remaining_shells = picked_up - put_back → 
  remaining_shells = 32.0 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_lino_shells_l1718_171833


namespace NUMINAMATH_GPT_inverse_proportion_function_l1718_171885

theorem inverse_proportion_function (m x : ℝ) (h : (m ≠ 0)) (A : (m, m / 8) ∈ {p : ℝ × ℝ | p.snd = (m / p.fst)}) :
    ∃ f : ℝ → ℝ, (∀ x, f x = 8 / x) :=
by
  use (fun x => 8 / x)
  intros x
  rfl

end NUMINAMATH_GPT_inverse_proportion_function_l1718_171885


namespace NUMINAMATH_GPT_average_of_a_and_b_l1718_171864

theorem average_of_a_and_b (a b c : ℝ) 
  (h₁ : (b + c) / 2 = 90)
  (h₂ : c - a = 90) :
  (a + b) / 2 = 45 :=
sorry

end NUMINAMATH_GPT_average_of_a_and_b_l1718_171864


namespace NUMINAMATH_GPT_repeating_decimals_sum_l1718_171863

theorem repeating_decimals_sum :
  let x := (246 : ℚ) / 999
  let y := (135 : ℚ) / 999
  let z := (579 : ℚ) / 999
  x - y + z = (230 : ℚ) / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_sum_l1718_171863


namespace NUMINAMATH_GPT_sum_diameters_eq_sum_legs_l1718_171883

theorem sum_diameters_eq_sum_legs 
  (a b c R r : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_circum_radius : R = c / 2)
  (h_incircle_radius : r = (a + b - c) / 2) :
  2 * R + 2 * r = a + b :=
by 
  sorry

end NUMINAMATH_GPT_sum_diameters_eq_sum_legs_l1718_171883


namespace NUMINAMATH_GPT_line_perpendicular_slope_l1718_171842

theorem line_perpendicular_slope (m : ℝ) :
  let slope1 := (1 / 2) 
  let slope2 := (-2 / m)
  slope1 * slope2 = -1 → m = 1 := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_line_perpendicular_slope_l1718_171842


namespace NUMINAMATH_GPT_greatest_prime_factor_of_154_l1718_171849

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ (∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p) :=
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_154_l1718_171849


namespace NUMINAMATH_GPT_present_age_of_R_l1718_171829

variables (P_p Q_p R_p : ℝ)

-- Conditions from the problem
axiom h1 : P_p - 8 = 1/2 * (Q_p - 8)
axiom h2 : Q_p - 8 = 2/3 * (R_p - 8)
axiom h3 : Q_p = 2 * Real.sqrt R_p
axiom h4 : P_p = 3/5 * Q_p

theorem present_age_of_R : R_p = 400 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_R_l1718_171829
