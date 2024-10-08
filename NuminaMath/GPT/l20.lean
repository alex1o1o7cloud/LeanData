import Mathlib

namespace find_x_value_l20_20920

open Real

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
(h3 : tan (150 * π / 180 - x * π / 180) = (sin (150 * π / 180) - sin (x * π / 180)) / (cos (150 * π / 180) - cos (x * π / 180))) :
x = 120 :=
sorry

end find_x_value_l20_20920


namespace number_of_teachers_students_possible_rental_plans_economical_plan_l20_20552

-- Definitions of conditions

def condition1 (x y : ℕ) : Prop := y - 30 * x = 7
def condition2 (x y : ℕ) : Prop := 31 * x - y = 1
def capacity_condition (m : ℕ) : Prop := 35 * m + 30 * (8 - m) ≥ 255
def rental_fee_condition (m : ℕ) : Prop := 400 * m + 320 * (8 - m) ≤ 3000

-- Problems to prove

theorem number_of_teachers_students (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 8 ∧ y = 247 := 
by sorry

theorem possible_rental_plans (m : ℕ) (h_cap : capacity_condition m) (h_fee : rental_fee_condition m) : m = 3 ∨ m = 4 ∨ m = 5 := 
by sorry

theorem economical_plan (m : ℕ) (h_fee : rental_fee_condition 3) (h_fee_alt1 : rental_fee_condition 4) (h_fee_alt2 : rental_fee_condition 5) : m = 3 := 
by sorry

end number_of_teachers_students_possible_rental_plans_economical_plan_l20_20552


namespace percentage_needed_to_pass_l20_20077

-- Definitions for conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def total_marks : ℕ := 500
def passing_marks := obtained_marks + failed_by

-- Assertion to prove
theorem percentage_needed_to_pass : (passing_marks : ℕ) * 100 / total_marks = 33 := by
  sorry

end percentage_needed_to_pass_l20_20077


namespace last_two_digits_of_7_pow_2015_l20_20931

theorem last_two_digits_of_7_pow_2015 : ((7 ^ 2015) % 100) = 43 := 
by
  sorry

end last_two_digits_of_7_pow_2015_l20_20931


namespace total_area_of_field_l20_20506

theorem total_area_of_field (A1 A2 : ℝ) (h1 : A1 = 225)
    (h2 : A2 - A1 = (1 / 5) * ((A1 + A2) / 2)) :
  A1 + A2 = 500 := by
  sorry

end total_area_of_field_l20_20506


namespace find_roots_l20_20890

-- Given the conditions:
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points (x, y)
def points := [(-5, 6), (-4, 0), (-2, -6), (0, -4), (2, 6)] 

-- Prove that the roots of the quadratic equation are -4 and 1
theorem find_roots (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : quadratic_function a b c (-5) = 6)
  (h₂ : quadratic_function a b c (-4) = 0)
  (h₃ : quadratic_function a b c (-2) = -6)
  (h₄ : quadratic_function a b c (0) = -4)
  (h₅ : quadratic_function a b c (2) = 6) :
  ∃ x₁ x₂ : ℝ, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = -4 ∧ x₂ = 1 := 
sorry

end find_roots_l20_20890


namespace fraction_of_remaining_birds_left_l20_20051

theorem fraction_of_remaining_birds_left (B : ℕ) (F : ℚ) (hB : B = 60)
  (H : (1/3) * (2/3 : ℚ) * B * (1 - F) = 8) :
  F = 4/5 := 
sorry

end fraction_of_remaining_birds_left_l20_20051


namespace intersection_of_sets_l20_20280

def set_A (x : ℝ) := x + 1 ≤ 3
def set_B (x : ℝ) := 4 - x^2 ≤ 0

theorem intersection_of_sets : {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | x ≤ -2} ∪ {2} :=
by
  sorry

end intersection_of_sets_l20_20280


namespace train_passing_platform_time_l20_20708

theorem train_passing_platform_time
  (L_train : ℝ) (L_plat : ℝ) (time_to_cross_tree : ℝ) (time_to_pass_platform : ℝ)
  (H1 : L_train = 2400) 
  (H2 : L_plat = 800)
  (H3 : time_to_cross_tree = 60) :
  time_to_pass_platform = 80 :=
by
  -- add proof here
  sorry

end train_passing_platform_time_l20_20708


namespace min_c_value_l20_20845

theorem min_c_value 
  (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + 1 = b)
  (h6 : b + 1 = c)
  (h7 : c + 1 = d)
  (h8 : d + 1 = e)
  (h9 : ∃ k : ℕ, k ^ 2 = b + c + d)
  (h10 : ∃ m : ℕ, m ^ 3 = a + b + c + d + e) : 
  c = 675 := 
sorry

end min_c_value_l20_20845


namespace supplemental_tank_time_l20_20724

-- Define the given conditions as assumptions
def primary_tank_time : Nat := 2
def total_time_needed : Nat := 8
def supplemental_tanks : Nat := 6
def additional_time_needed : Nat := total_time_needed - primary_tank_time

-- Define the theorem to prove
theorem supplemental_tank_time :
  additional_time_needed / supplemental_tanks = 1 :=
by
  -- Here we would provide the proof, but it is omitted with "sorry"
  sorry

end supplemental_tank_time_l20_20724


namespace maggie_sold_2_subscriptions_to_neighbor_l20_20620

-- Definition of the problem conditions
def maggie_pays_per_subscription : Int := 5
def maggie_subscriptions_to_parents : Int := 4
def maggie_subscriptions_to_grandfather : Int := 1
def maggie_earned_total : Int := 55

-- Define the function to be proven
def subscriptions_sold_to_neighbor (x : Int) : Prop :=
  maggie_pays_per_subscription * (maggie_subscriptions_to_parents + maggie_subscriptions_to_grandfather + x + 2*x) = maggie_earned_total

-- The statement we need to prove
theorem maggie_sold_2_subscriptions_to_neighbor :
  subscriptions_sold_to_neighbor 2 :=
sorry

end maggie_sold_2_subscriptions_to_neighbor_l20_20620


namespace total_onions_grown_l20_20734

theorem total_onions_grown :
  let onions_per_day_nancy := 3
  let days_worked_nancy := 4
  let onions_per_day_dan := 4
  let days_worked_dan := 6
  let onions_per_day_mike := 5
  let days_worked_mike := 5
  let onions_per_day_sasha := 6
  let days_worked_sasha := 4
  let onions_per_day_becky := 2
  let days_worked_becky := 6

  let total_onions_nancy := onions_per_day_nancy * days_worked_nancy
  let total_onions_dan := onions_per_day_dan * days_worked_dan
  let total_onions_mike := onions_per_day_mike * days_worked_mike
  let total_onions_sasha := onions_per_day_sasha * days_worked_sasha
  let total_onions_becky := onions_per_day_becky * days_worked_becky

  let total_onions := total_onions_nancy + total_onions_dan + total_onions_mike + total_onions_sasha + total_onions_becky

  total_onions = 97 :=
by
  -- proof goes here
  sorry

end total_onions_grown_l20_20734


namespace angus_tokens_l20_20744

theorem angus_tokens (x : ℕ) (h1 : x = 60 - (25 / 100) * 60) : x = 45 :=
by
  sorry

end angus_tokens_l20_20744


namespace part_1_part_2_part_3_l20_20466

/-- Defining a structure to hold the values of x and y as given in the problem --/
structure PhoneFeeData (α : Type) :=
  (x : α) (y : α)

def problem_data : List (PhoneFeeData ℝ) :=
  [
    ⟨1, 18.4⟩, ⟨2, 18.8⟩, ⟨3, 19.2⟩, ⟨4, 19.6⟩, ⟨5, 20⟩, ⟨6, 20.4⟩
  ]

noncomputable def phone_fee_equation (x : ℝ) : ℝ := 0.4 * x + 18

theorem part_1 :
  ∀ data ∈ problem_data, phone_fee_equation data.x = data.y :=
by
  sorry

theorem part_2 : phone_fee_equation 10 = 22 :=
by
  sorry

theorem part_3 : ∀ x : ℝ, phone_fee_equation x = 26 → x = 20 :=
by
  sorry

end part_1_part_2_part_3_l20_20466


namespace arithmetic_sequence_sum_six_l20_20864

open Nat

noncomputable def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  3 * (2 * a1 + 5 * d) / 3

theorem arithmetic_sequence_sum_six (a : ℕ → ℚ) (h : a 2 + a 5 = 2 / 3) : sum_first_six_terms a = 2 :=
by
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  have eq1 : a 5 = a1 + 4 * d := by sorry
  have eq2 : 3 * (2 * a1 + 5 * d) / 3 = (2 : ℚ) := by sorry
  sorry

end arithmetic_sequence_sum_six_l20_20864


namespace total_acorns_l20_20824

theorem total_acorns (x y : ℝ) :
  let sheila_acorns := 5.3 * x
  let danny_acorns := sheila_acorns + y
  x + sheila_acorns + danny_acorns = 11.6 * x + y :=
by
  sorry

end total_acorns_l20_20824


namespace product_three_consecutive_not_power_l20_20172

theorem product_three_consecutive_not_power (n k m : ℕ) (hn : n > 0) (hm : m ≥ 2) : 
  (n-1) * n * (n+1) ≠ k^m :=
by sorry

end product_three_consecutive_not_power_l20_20172


namespace weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l20_20195

noncomputable def Wa : ℕ := 
  let volume_a := (3/5) * 4
  let volume_b := (2/5) * 4
  let weight_b := 700
  let total_weight := 3280
  (total_weight - (weight_b * volume_b)) / volume_a

theorem weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900 :
  Wa = 900 := 
by
  sorry

end weight_of_one_liter_vegetable_ghee_packet_of_brand_a_is_900_l20_20195


namespace factory_correct_decision_prob_l20_20567

def prob_correct_decision (p : ℝ) : ℝ :=
  let prob_all_correct := p * p * p
  let prob_two_correct_one_incorrect := 3 * p * p * (1 - p)
  prob_all_correct + prob_two_correct_one_incorrect

theorem factory_correct_decision_prob : prob_correct_decision 0.8 = 0.896 :=
by
  sorry

end factory_correct_decision_prob_l20_20567


namespace father_age_difference_l20_20852

variables (F S X : ℕ)
variable (h1 : F = 33)
variable (h2 : F = 3 * S + X)
variable (h3 : F + 3 = 2 * (S + 3) + 10)

theorem father_age_difference : X = 3 :=
by
  sorry

end father_age_difference_l20_20852


namespace car_daily_rental_cost_l20_20695

theorem car_daily_rental_cost 
  (x : ℝ)
  (cost_per_mile : ℝ)
  (budget : ℝ)
  (miles : ℕ)
  (h1 : cost_per_mile = 0.18)
  (h2 : budget = 75)
  (h3 : miles = 250)
  (h4 : x + (miles * cost_per_mile) = budget) : 
  x = 30 := 
sorry

end car_daily_rental_cost_l20_20695


namespace farthest_vertex_coordinates_l20_20314

noncomputable def image_vertex_coordinates_farthest_from_origin 
    (center_EFGH : ℝ × ℝ) (area_EFGH : ℝ) (dilation_center : ℝ × ℝ) 
    (scale_factor : ℝ) : ℝ × ℝ := sorry

theorem farthest_vertex_coordinates 
    (center_EFGH : ℝ × ℝ := (10, -6)) (area_EFGH : ℝ := 16) 
    (dilation_center : ℝ × ℝ := (2, 2)) (scale_factor : ℝ := 3) : 
    image_vertex_coordinates_farthest_from_origin center_EFGH area_EFGH dilation_center scale_factor = (32, -28) := 
sorry

end farthest_vertex_coordinates_l20_20314


namespace find_x_l20_20363

def custom_op (a b : ℝ) : ℝ :=
  a^2 - 3 * b

theorem find_x (x : ℝ) : 
  (custom_op (custom_op 7 x) 3 = 18) ↔ (x = 17.71 ∨ x = 14.96) := 
by
  sorry

end find_x_l20_20363


namespace total_tv_show_cost_correct_l20_20322

noncomputable def total_cost_of_tv_show : ℕ :=
  let cost_per_episode_first_season := 100000
  let episodes_first_season := 12
  let episodes_seasons_2_to_4 := 18
  let cost_per_episode_other_seasons := 2 * cost_per_episode_first_season
  let episodes_last_season := 24
  let number_of_other_seasons := 4
  let total_cost_first_season := episodes_first_season * cost_per_episode_first_season
  let total_cost_other_seasons := (episodes_seasons_2_to_4 * 3 + episodes_last_season) * cost_per_episode_other_seasons
  total_cost_first_season + total_cost_other_seasons

theorem total_tv_show_cost_correct : total_cost_of_tv_show = 16800000 := by
  sorry

end total_tv_show_cost_correct_l20_20322


namespace gcd_of_abcd_dcba_l20_20900

theorem gcd_of_abcd_dcba : 
  ∀ (a : ℕ), 0 ≤ a ∧ a ≤ 3 → 
  gcd (2332 * a + 7112) (2332 * (a + 1) + 7112) = 2 ∧ 
  gcd (2332 * (a + 1) + 7112) (2332 * (a + 2) + 7112) = 2 ∧ 
  gcd (2332 * (a + 2) + 7112) (2332 * (a + 3) + 7112) = 2 := 
by 
  sorry

end gcd_of_abcd_dcba_l20_20900


namespace change_calculation_l20_20197

/-!
# Problem
Adam has $5 to buy an airplane that costs $4.28. How much change will he get after buying the airplane?

# Conditions
Adam has $5.
The airplane costs $4.28.

# Statement
Prove that the change Adam will get is $0.72.
-/

theorem change_calculation : 
  let amount := 5.00
  let cost := 4.28
  let change := 0.72
  amount - cost = change :=
by 
  sorry

end change_calculation_l20_20197


namespace tangent_line_sum_l20_20672

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end tangent_line_sum_l20_20672


namespace quadratic_function_vertex_and_comparison_l20_20949

theorem quadratic_function_vertex_and_comparison
  (a b c : ℝ)
  (A_conds : 4 * a - 2 * b + c = 9)
  (B_conds : c = 3)
  (C_conds : 16 * a + 4 * b + c = 3) :
  (a = 1/2 ∧ b = -2 ∧ c = 3) ∧
  (∀ (m : ℝ) (y₁ y₂ : ℝ),
     y₁ = 1/2 * m^2 - 2 * m + 3 ∧
     y₂ = 1/2 * (m + 1)^2 - 2 * (m + 1) + 3 →
     (m < 3/2 → y₁ > y₂) ∧
     (m = 3/2 → y₁ = y₂) ∧
     (m > 3/2 → y₁ < y₂)) :=
by
  sorry

end quadratic_function_vertex_and_comparison_l20_20949


namespace incorrect_statement_C_l20_20575

theorem incorrect_statement_C (a b : ℤ) (h : |a| = |b|) : (a ≠ b ∧ a = -b) :=
by
  sorry

end incorrect_statement_C_l20_20575


namespace solve_N1N2_identity_l20_20189

theorem solve_N1N2_identity :
  (∃ N1 N2 : ℚ,
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 3 →
      (42 * x - 37) / (x^2 - 4 * x + 3) =
      N1 / (x - 1) + N2 / (x - 3)) ∧ 
      N1 * N2 = -445 / 4) :=
by
  sorry

end solve_N1N2_identity_l20_20189


namespace simplify_fraction_product_l20_20911

theorem simplify_fraction_product :
  (2 / 3) * (4 / 7) * (9 / 13) = 24 / 91 := by
  sorry

end simplify_fraction_product_l20_20911


namespace triangular_number_30_sum_of_first_30_triangular_numbers_l20_20776

theorem triangular_number_30 
  (T : ℕ → ℕ)
  (hT : ∀ n : ℕ, T n = n * (n + 1) / 2) : 
  T 30 = 465 :=
by
  -- Skipping proof with sorry
  sorry

theorem sum_of_first_30_triangular_numbers 
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = n * (n + 1) * (n + 2) / 6) : 
  S 30 = 4960 :=
by
  -- Skipping proof with sorry
  sorry

end triangular_number_30_sum_of_first_30_triangular_numbers_l20_20776


namespace convert_quadratic_l20_20107

theorem convert_quadratic :
  ∀ x : ℝ, (x^2 + 2*x + 4) = ((x + 1)^2 + 3) :=
by
  sorry

end convert_quadratic_l20_20107


namespace range_of_a_l20_20199

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x^2 + 2 * a * x + 1) → -1 ≤ a ∧ a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l20_20199


namespace max_value_x_y2_z3_l20_20960

theorem max_value_x_y2_z3 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  x + y^2 + z^3 ≤ 1 :=
by
  sorry

end max_value_x_y2_z3_l20_20960


namespace complement_of_16deg51min_is_73deg09min_l20_20966

def complement_angle (A : ℝ) : ℝ := 90 - A

theorem complement_of_16deg51min_is_73deg09min :
  complement_angle 16.85 = 73.15 := by
  sorry

end complement_of_16deg51min_is_73deg09min_l20_20966


namespace minimum_value_exists_l20_20473

theorem minimum_value_exists (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) ∧ (1 / m + 1 / n ≥ min_val) :=
by {
  -- Proof will be provided here.
  sorry
}

end minimum_value_exists_l20_20473


namespace force_for_wrenches_l20_20161

open Real

theorem force_for_wrenches (F : ℝ) (k : ℝ) :
  (F * 12 = 3600) → 
  (k = 3600) →
  (3600 / 8 = 450) →
  (3600 / 18 = 200) →
  true :=
by
  intro hF hk h8 h18
  trivial

end force_for_wrenches_l20_20161


namespace find_sum_l20_20110

variable (a b : ℝ)

theorem find_sum (h1 : 2 = b - 1) (h2 : -1 = a + 3) : a + b = -1 :=
by
  sorry

end find_sum_l20_20110


namespace xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l20_20449

theorem xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) :
  (x * y^2 + 2 * y) ∣ (2 * x^2 * y + x * y^2 + 8 * x) ↔ 
  (∃ a : ℕ, 0 < a ∧ x = a ∧ y = 2 * a) ∨ (x = 3 ∧ y = 1) ∨ (x = 8 ∧ y = 1) :=
by
  sorry

end xy2_plus_2y_divides_2x2y_plus_xy2_plus_8x_l20_20449


namespace garden_area_increase_l20_20361

-- Definitions derived directly from the conditions
def length := 50
def width := 10
def perimeter := 2 * (length + width)
def side_length_square := perimeter / 4
def area_rectangle := length * width
def area_square := side_length_square * side_length_square

-- The proof statement
theorem garden_area_increase :
  area_square - area_rectangle = 400 := 
by
  sorry

end garden_area_increase_l20_20361


namespace product_of_roots_l20_20779

theorem product_of_roots :
  ∃ x₁ x₂ : ℝ, (x₁ * x₂ = -4) ∧ (x₁ ^ 2 + 2 * x₁ - 4 = 0) ∧ (x₂ ^ 2 + 2 * x₂ - 4 = 0) := by
  sorry

end product_of_roots_l20_20779


namespace solve_for_k_and_j_l20_20683

theorem solve_for_k_and_j (k j : ℕ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end solve_for_k_and_j_l20_20683


namespace value_range_f_l20_20240

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x - Real.sin (2 * x) + 1

theorem value_range_f :
  ∀ x ∈ Set.Ico (-(5 * Real.pi) / 12) (Real.pi / 3), 
  f x ∈ Set.Icc ((3 : ℝ) / 2 - Real.sqrt 2) 3 :=
by
  sorry

end value_range_f_l20_20240


namespace hands_straight_line_time_l20_20013

noncomputable def time_when_hands_straight_line : List (ℕ × ℚ) :=
  let x₁ := 21 + 9 / 11
  let x₂ := 54 + 6 / 11
  [(4, x₁), (4, x₂)]

theorem hands_straight_line_time :
  time_when_hands_straight_line = [(4, 21 + 9 / 11), (4, 54 + 6 / 11)] :=
by
  sorry

end hands_straight_line_time_l20_20013


namespace add_fraction_l20_20825

theorem add_fraction (x : ℚ) (h : x - 7/3 = 3/2) : x + 7/3 = 37/6 :=
by
  sorry

end add_fraction_l20_20825


namespace mean_weight_is_70_357_l20_20431

def weights_50 : List ℕ := [57]
def weights_60 : List ℕ := [60, 64, 64, 66, 69]
def weights_70 : List ℕ := [71, 73, 73, 75, 77, 78, 79, 79]

def weights := weights_50 ++ weights_60 ++ weights_70

def total_weight : ℕ := List.sum weights
def total_players : ℕ := List.length weights
def mean_weight : ℚ := (total_weight : ℚ) / total_players

theorem mean_weight_is_70_357 :
  mean_weight = 70.357 := 
sorry

end mean_weight_is_70_357_l20_20431


namespace intersecting_lines_l20_20822

-- Definitions based on conditions
def line1 (m : ℝ) (x : ℝ) : ℝ := m * x + 4
def line2 (b : ℝ) (x : ℝ) : ℝ := 3 * x + b

-- Lean 4 Statement of the problem
theorem intersecting_lines (m b : ℝ) (h1 : line1 m 6 = 10) (h2 : line2 b 6 = 10) : b + m = -7 :=
by
  sorry

end intersecting_lines_l20_20822


namespace count_sequences_of_length_15_l20_20202

def countingValidSequences (n : ℕ) : ℕ := sorry

theorem count_sequences_of_length_15 :
  countingValidSequences 15 = 266 :=
  sorry

end count_sequences_of_length_15_l20_20202


namespace technician_round_trip_percentage_l20_20830

theorem technician_round_trip_percentage (D: ℝ) (hD: D ≠ 0): 
  let round_trip_distance := 2 * D
  let distance_to_center := D
  let distance_back_10_percent := 0.10 * D
  let total_distance_completed := distance_to_center + distance_back_10_percent
  let percentage_completed := (total_distance_completed / round_trip_distance) * 100
  percentage_completed = 55 := 
by
  simp
  sorry -- Proof is not required per instructions

end technician_round_trip_percentage_l20_20830


namespace intersection_A_B_range_m_l20_20998

-- Define set A when m = 3 as given
def A_set (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0
def A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define set B when m = 3 as given
def B_set (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

-- The intersection of A and B should be: -2 ≤ x ≤ 1
theorem intersection_A_B : ∀ (x : ℝ), A x ∧ B x ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

-- Define A for general m > 0
def A_set_general (x : ℝ) : Prop := 3 - 2 * x - x^2 ≥ 0

-- Define B for general m
def B_set_general (x : ℝ) (m : ℝ) : Prop := (x - 1)^2 ≤ m^2

-- Prove the range for m such that A ⊆ B
theorem range_m (m : ℝ) (h : m > 0) : (∀ x, A_set_general x → B_set_general x m) ↔ m ≥ 4 := sorry

end intersection_A_B_range_m_l20_20998


namespace peg_stickers_total_l20_20152

def stickers_in_red_folder : ℕ := 10 * 3
def stickers_in_green_folder : ℕ := 10 * 2
def stickers_in_blue_folder : ℕ := 10 * 1

def total_stickers : ℕ := stickers_in_red_folder + stickers_in_green_folder + stickers_in_blue_folder

theorem peg_stickers_total : total_stickers = 60 := by
  sorry

end peg_stickers_total_l20_20152


namespace net_income_correct_l20_20707

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l20_20707


namespace min_x2_y2_z2_l20_20266

open Real

theorem min_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_x2_y2_z2_l20_20266


namespace tangent_line_equation_at_point_l20_20142

-- Defining the function and the point
def f (x : ℝ) : ℝ := x^2 + 2 * x
def point : ℝ × ℝ := (1, 3)

-- Main theorem stating the tangent line equation at the given point
theorem tangent_line_equation_at_point : 
  ∃ m b, (m = (2 * 1 + 2)) ∧ 
         (b = (3 - m * 1)) ∧ 
         (∀ x y, y = f x → y = m * x + b → 4 * x - y - 1 = 0) :=
by
  -- Proof is omitted and can be filled in later
  sorry

end tangent_line_equation_at_point_l20_20142


namespace johns_cocktail_not_stronger_l20_20675

-- Define the percentage of alcohol in each beverage
def beer_percent_alcohol : ℝ := 0.05
def liqueur_percent_alcohol : ℝ := 0.10
def vodka_percent_alcohol : ℝ := 0.40
def whiskey_percent_alcohol : ℝ := 0.50

-- Define the weights of the liquids used in the cocktails
def john_liqueur_weight : ℝ := 400
def john_whiskey_weight : ℝ := 100
def ivan_vodka_weight : ℝ := 400
def ivan_beer_weight : ℝ := 100

-- Calculate the alcohol contents in each cocktail
def johns_cocktail_alcohol : ℝ := john_liqueur_weight * liqueur_percent_alcohol + john_whiskey_weight * whiskey_percent_alcohol
def ivans_cocktail_alcohol : ℝ := ivan_vodka_weight * vodka_percent_alcohol + ivan_beer_weight * beer_percent_alcohol

-- The proof statement to prove that John's cocktail is not stronger than Ivan's cocktail
theorem johns_cocktail_not_stronger :
  johns_cocktail_alcohol ≤ ivans_cocktail_alcohol :=
by
  -- Add proof steps here
  sorry

end johns_cocktail_not_stronger_l20_20675


namespace mother_to_grandfather_age_ratio_l20_20831

theorem mother_to_grandfather_age_ratio
  (rachel_age : ℕ)
  (grandfather_ratio : ℕ)
  (father_mother_gap : ℕ) 
  (future_rachel_age: ℕ) 
  (future_father_age : ℕ)
  (current_father_age current_mother_age current_grandfather_age : ℕ) 
  (h1 : rachel_age = 12)
  (h2 : grandfather_ratio = 7)
  (h3 : father_mother_gap = 5)
  (h4 : future_rachel_age = 25)
  (h5 : future_father_age = 60)
  (h6 : current_father_age = future_father_age - (future_rachel_age - rachel_age))
  (h7 : current_mother_age = current_father_age - father_mother_gap)
  (h8 : current_grandfather_age = grandfather_ratio * rachel_age) :
  current_mother_age = current_grandfather_age / 2 :=
by
  sorry

end mother_to_grandfather_age_ratio_l20_20831


namespace cos_a2_plus_a8_eq_neg_half_l20_20021

noncomputable def a_n (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem cos_a2_plus_a8_eq_neg_half 
  (a₁ d : ℝ) 
  (h : a₁ + a_n 5 a₁ d + a_n 9 a₁ d = 5 * Real.pi)
  : Real.cos (a_n 2 a₁ d + a_n 8 a₁ d) = -1 / 2 :=
by
  sorry

end cos_a2_plus_a8_eq_neg_half_l20_20021


namespace books_sold_on_tuesday_l20_20141

theorem books_sold_on_tuesday (total_stock : ℕ) (monday_sold : ℕ) (wednesday_sold : ℕ)
  (thursday_sold : ℕ) (friday_sold : ℕ) (percent_unsold : ℚ) (tuesday_sold : ℕ) :
  total_stock = 1100 →
  monday_sold = 75 →
  wednesday_sold = 64 →
  thursday_sold = 78 →
  friday_sold = 135 →
  percent_unsold = 63.45 →
  tuesday_sold = total_stock - (monday_sold + wednesday_sold + thursday_sold + friday_sold + (total_stock * percent_unsold / 100)) :=
by sorry

end books_sold_on_tuesday_l20_20141


namespace preimage_of_8_is_5_image_of_8_is_64_l20_20887

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2)

theorem preimage_of_8_is_5 : ∃ x, f x = 8 := by
  use 5
  sorry

theorem image_of_8_is_64 : f 8 = 64 := by
  sorry

end preimage_of_8_is_5_image_of_8_is_64_l20_20887


namespace shaded_area_l20_20669

-- Definitions based on given conditions
def Rectangle (A B C D : ℝ) := True -- Placeholder for the geometric definition of a rectangle

-- Total area of the non-shaded part
def non_shaded_area : ℝ := 10

-- Problem statement in Lean
theorem shaded_area (A B C D : ℝ) :
  Rectangle A B C D →
  (exists shaded_area : ℝ, shaded_area = 14 ∧ non_shaded_area + shaded_area = A * B) :=
by
  sorry

end shaded_area_l20_20669


namespace no_valid_road_network_l20_20513

theorem no_valid_road_network
  (k_A k_B k_C : ℕ)
  (h_kA : k_A ≥ 2)
  (h_kB : k_B ≥ 2)
  (h_kC : k_C ≥ 2) :
  ¬ ∃ (t : ℕ) (d : ℕ → ℕ), t ≥ 7 ∧ 
    (∀ i j, i ≠ j → d i ≠ d j) ∧
    (∀ i, i < 4 * (k_A + k_B + k_C) + 4 → d i = i + 1) :=
sorry

end no_valid_road_network_l20_20513


namespace average_income_A_B_l20_20364

theorem average_income_A_B (A B C : ℝ)
  (h1 : (B + C) / 2 = 5250)
  (h2 : (A + C) / 2 = 4200)
  (h3 : A = 3000) : (A + B) / 2 = 4050 :=
by
  sorry

end average_income_A_B_l20_20364


namespace no_solution_exists_l20_20005

theorem no_solution_exists :
  ¬ ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧ n % 8 = 0 ∧ n % 10 = 6 ∧ n % 7 = 6 := 
by
  sorry

end no_solution_exists_l20_20005


namespace horner_method_poly_at_neg2_l20_20224

-- Define the polynomial using the given conditions and Horner's method transformation
def polynomial : ℤ → ℤ := fun x => (((((x - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2

-- State the theorem
theorem horner_method_poly_at_neg2 : polynomial (-2) = -40 := by
  sorry

end horner_method_poly_at_neg2_l20_20224


namespace solutions_of_quadratic_l20_20896

theorem solutions_of_quadratic 
  (p q : ℚ) 
  (h₁ : 2 * p * p + 11 * p - 21 = 0) 
  (h₂ : 2 * q * q + 11 * q - 21 = 0) : 
  (p - q) * (p - q) = 289 / 4 := 
sorry

end solutions_of_quadratic_l20_20896


namespace value_of_x2_plus_inv_x2_l20_20674

theorem value_of_x2_plus_inv_x2 (x : ℝ) (hx : x ≠ 0) (h : x^4 + 1 / x^4 = 47) : x^2 + 1 / x^2 = 7 :=
sorry

end value_of_x2_plus_inv_x2_l20_20674


namespace diamond_45_15_eq_3_l20_20657

noncomputable def diamond (x y : ℝ) : ℝ := x / y

theorem diamond_45_15_eq_3 :
  ∀ (x y : ℝ), 
    (∀ x y : ℝ, (x * y) / y = x * (x / y)) ∧
    (∀ x : ℝ, (x / 1) / x = x / 1) ∧
    (∀ x y : ℝ, x / y = x / y) ∧
    1 / 1 = 1
    → diamond 45 15 = 3 :=
by
  intros x y H
  sorry

end diamond_45_15_eq_3_l20_20657


namespace value_of_x_l20_20451

theorem value_of_x (x : ℝ) (h : x = 80 + 0.2 * 80) : x = 96 :=
sorry

end value_of_x_l20_20451


namespace smallest_integer_solution_l20_20680

theorem smallest_integer_solution : ∀ x : ℤ, (x < 2 * x - 7) → (8 = x) :=
by
  sorry

end smallest_integer_solution_l20_20680


namespace ellipse_foci_cond_l20_20559

theorem ellipse_foci_cond (m n : ℝ) (h_cond : m > n ∧ n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1 → (m > n ∧ n > 0)) ∧ ((m > n ∧ n > 0) → ∀ x y : ℝ, mx^2 + ny^2 = 1) :=
sorry

end ellipse_foci_cond_l20_20559


namespace range_of_m_l20_20726

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m+1)*x^2 + (m+1)*x + (m+2) ≥ 0) ↔ m ≥ -1 := by
  sorry

end range_of_m_l20_20726


namespace monotonic_intervals_range_of_k_l20_20111

noncomputable def f (x a : ℝ) : ℝ :=
  (x - a - 1) * Real.exp x - (1/2) * x^2 + a * x

-- Conditions: a > 0
variables (a : ℝ) (h_a : 0 < a)

-- Part (1): Monotonic Intervals
theorem monotonic_intervals :
  (∀ x, f x a < f (x + 1) a ↔ x < 0 ∨ a < x) ∧
  (∀ x, f (x + 1) a < f x a ↔ 0 < x ∧ x < a) :=
  sorry

-- Part (2): Range of k
theorem range_of_k (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  (f x1 a - f x2 a < k * a^3) ↔ k ≥ -1/6 :=
  sorry

end monotonic_intervals_range_of_k_l20_20111


namespace circle_circumference_l20_20795

theorem circle_circumference (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi :=
by
  -- Use the given dimensions to find the diagonal (which is the diameter).
  -- Calculate the circumference using the calculated diameter.
  sorry

end circle_circumference_l20_20795


namespace cos_double_angle_l20_20476

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 2) = 1 / 2) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l20_20476


namespace collinear_points_on_curve_sum_zero_l20_20587

theorem collinear_points_on_curve_sum_zero
  {x1 y1 x2 y2 x3 y3 : ℝ}
  (h_curve1 : y1^2 = x1^3)
  (h_curve2 : y2^2 = x2^3)
  (h_curve3 : y3^2 = x3^3)
  (h_collinear : ∃ (a b c k : ℝ), k ≠ 0 ∧ 
    a * x1 + b * y1 + c = 0 ∧
    a * x2 + b * y2 + c = 0 ∧
    a * x3 + b * y3 + c = 0) :
  x1 / y1 + x2 / y2 + x3 / y3 = 0 :=
sorry

end collinear_points_on_curve_sum_zero_l20_20587


namespace constant_for_odd_m_l20_20886

theorem constant_for_odd_m (constant : ℝ) (f : ℕ → ℝ)
  (h1 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k + 1) → f m = constant * m)
  (h2 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k) → f m = (1/2 : ℝ) * m)
  (h3 : f 5 * f 6 = 15) : constant = 1 :=
by
  sorry

end constant_for_odd_m_l20_20886


namespace max_distance_from_origin_to_line_l20_20259

variable (k : ℝ)

def line (x y : ℝ) : Prop := k * x + y + 1 = 0

theorem max_distance_from_origin_to_line :
  ∃ k : ℝ, ∀ x y : ℝ, line k x y -> dist (0, 0) (x, y) ≤ 1 := 
sorry

end max_distance_from_origin_to_line_l20_20259


namespace work_completion_time_l20_20774

theorem work_completion_time (P W : ℕ) (h : P * 8 = W) : 2 * P * 2 = W / 2 := by
  sorry

end work_completion_time_l20_20774


namespace watch_cost_l20_20087

-- Definitions based on conditions
def initial_money : ℤ := 1
def money_from_david : ℤ := 12
def money_needed : ℤ := 7

-- Indicating the total money Evan has after receiving money from David
def total_money := initial_money + money_from_david

-- The cost of the watch based on total money Evan has and additional money needed
def cost_of_watch := total_money + money_needed

-- Proving the cost of the watch
theorem watch_cost : cost_of_watch = 20 := by
  -- We are skipping the proof steps here
  sorry

end watch_cost_l20_20087


namespace bob_friends_l20_20759

-- Define the total price and the amount paid by each person
def total_price : ℕ := 40
def amount_per_person : ℕ := 8

-- Define the total number of people who paid
def total_people : ℕ := total_price / amount_per_person

-- Define Bob's presence and require proving the number of his friends
theorem bob_friends (total_price amount_per_person total_people : ℕ) (h1 : total_price = 40)
  (h2 : amount_per_person = 8) (h3 : total_people = total_price / amount_per_person) : 
  total_people - 1 = 4 :=
by
  sorry

end bob_friends_l20_20759


namespace point_in_first_quadrant_l20_20902

-- Define the system of equations
def equations (x y : ℝ) : Prop :=
  x + y = 2 ∧ x - y = 1

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : equations x y) : x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l20_20902


namespace train_length_is_sixteenth_mile_l20_20872

theorem train_length_is_sixteenth_mile
  (train_speed : ℕ)
  (bridge_length : ℕ)
  (man_speed : ℕ)
  (cross_time : ℚ)
  (man_distance : ℚ)
  (length_of_train : ℚ)
  (h1 : train_speed = 80)
  (h2 : bridge_length = 1)
  (h3 : man_speed = 5)
  (h4 : cross_time = bridge_length / train_speed)
  (h5 : man_distance = man_speed * cross_time)
  (h6 : length_of_train = man_distance) :
  length_of_train = 1 / 16 :=
by sorry

end train_length_is_sixteenth_mile_l20_20872


namespace remainder_of_7_9_power_2008_mod_64_l20_20796

theorem remainder_of_7_9_power_2008_mod_64 :
  (7^2008 + 9^2008) % 64 = 2 := 
sorry

end remainder_of_7_9_power_2008_mod_64_l20_20796


namespace exactly_one_is_multiple_of_5_l20_20936

theorem exactly_one_is_multiple_of_5 (a b : ℤ) (h: 24 * a^2 + 1 = b^2) : 
  (∃ k : ℤ, a = 5 * k) ∧ (∀ l : ℤ, b ≠ 5 * l) ∨ (∃ m : ℤ, b = 5 * m) ∧ (∀ n : ℤ, a ≠ 5 * n) :=
sorry

end exactly_one_is_multiple_of_5_l20_20936


namespace sum_of_distinct_integers_l20_20994

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
(h_prod : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120) : 
a + b + c + d + e = 33 := 
sorry

end sum_of_distinct_integers_l20_20994


namespace maximum_sum_is_42_l20_20137

-- Definitions according to the conditions in the problem

def initial_faces : ℕ := 7 -- 2 pentagonal + 5 rectangular
def initial_vertices : ℕ := 10 -- 5 at the top and 5 at the bottom
def initial_edges : ℕ := 15 -- 5 for each pentagon and 5 linking them

def added_faces : ℕ := 5 -- 5 new triangular faces
def added_vertices : ℕ := 1 -- 1 new vertex at the apex of the pyramid
def added_edges : ℕ := 5 -- 5 new edges connecting the new vertex to the pentagon's vertices

-- New quantities after adding the pyramid
def new_faces : ℕ := initial_faces - 1 + added_faces
def new_vertices : ℕ := initial_vertices + added_vertices
def new_edges : ℕ := initial_edges + added_edges

-- Sum of the new shape's characteristics
def sum_faces_vertices_edges : ℕ := new_faces + new_vertices + new_edges

-- Statement to be proved
theorem maximum_sum_is_42 : sum_faces_vertices_edges = 42 := by
  sorry

end maximum_sum_is_42_l20_20137


namespace find_natural_number_A_l20_20269

theorem find_natural_number_A (A : ℕ) : 
  (A * 1000 ≤ (A * (A + 1)) / 2 ∧ (A * (A + 1)) / 2 ≤ A * 1000 + 999) → A = 1999 :=
by
  sorry

end find_natural_number_A_l20_20269


namespace krish_remaining_money_l20_20270

variable (initial_amount sweets stickers friends each_friend charity : ℝ)

theorem krish_remaining_money :
  initial_amount = 200.50 →
  sweets = 35.25 →
  stickers = 10.75 →
  friends = 4 →
  each_friend = 25.20 →
  charity = 15.30 →
  initial_amount - (sweets + stickers + friends * each_friend + charity) = 38.40 :=
by
  intros h_initial h_sweets h_stickers h_friends h_each_friend h_charity
  sorry

end krish_remaining_money_l20_20270


namespace max_profit_l20_20581

noncomputable def profit (x : ℝ) : ℝ :=
  20 * x - 3 * x^2 + 96 * Real.log x - 90

theorem max_profit :
  ∃ x : ℝ, 4 ≤ x ∧ x ≤ 12 ∧ 
  (∀ y : ℝ, 4 ≤ y ∧ y ≤ 12 → profit y ≤ profit x) ∧ profit x = 96 * Real.log 6 - 78 :=
by
  sorry

end max_profit_l20_20581


namespace part1_part2_part3_l20_20534

variable {x : ℝ}

def A := {x : ℝ | x^2 + 3 * x - 4 > 0}
def B := {x : ℝ | x^2 - x - 6 < 0}
def C_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem part1 : (A ∩ B) = {x : ℝ | 1 < x ∧ x < 3} := sorry

theorem part2 : (C_R (A ∩ B)) = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := sorry

theorem part3 : (A ∪ (C_R B)) = {x : ℝ | x ≤ -2 ∨ x > 1} := sorry

end part1_part2_part3_l20_20534


namespace mean_of_two_means_eq_l20_20068

theorem mean_of_two_means_eq (z : ℚ) (h : (5 + 10 + 20) / 3 = (15 + z) / 2) : z = 25 / 3 :=
by
  sorry

end mean_of_two_means_eq_l20_20068


namespace circle_area_l20_20024

-- Let r be the radius of the circle
-- The circumference of the circle is given by 2 * π * r, which is 36 cm
-- We need to prove that given this condition, the area of the circle is 324/π square centimeters

theorem circle_area (r : Real) (h : 2 * Real.pi * r = 36) : Real.pi * r^2 = 324 / Real.pi :=
by
  sorry

end circle_area_l20_20024


namespace find_r_amount_l20_20891

theorem find_r_amount (p q r : ℝ) (h_total : p + q + r = 8000) (h_r_fraction : r = 2 / 3 * (p + q)) : r = 3200 :=
by 
  -- Proof is not required, hence we use sorry
  sorry

end find_r_amount_l20_20891


namespace mark_more_than_kate_l20_20611

variables {K P M : ℕ}

-- Conditions
def total_hours (P K M : ℕ) : Prop := P + K + M = 189
def pat_as_kate (P K : ℕ) : Prop := P = 2 * K
def pat_as_mark (P M : ℕ) : Prop := P = M / 3

-- Statement
theorem mark_more_than_kate (K P M : ℕ) (h1 : total_hours P K M)
  (h2 : pat_as_kate P K) (h3 : pat_as_mark P M) : M - K = 105 :=
by {
  sorry
}

end mark_more_than_kate_l20_20611


namespace polygon_sides_sum_l20_20147

theorem polygon_sides_sum :
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  -- The sides of shapes that are adjacent on one side each (triangle and nonagon)
  let adjacent_triangle_nonagon := triangle_sides + nonagon_sides - 2
  -- The sides of the intermediate shapes that are each adjacent on two sides
  let adjacent_other_shapes := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - 5 * 2
  -- Summing up all the sides exposed to the outside
  adjacent_triangle_nonagon + adjacent_other_shapes = 30 := by
sorry

end polygon_sides_sum_l20_20147


namespace kids_played_on_tuesday_l20_20749

-- Definitions of the conditions
def kids_played_on_wednesday (julia : Type) : Nat := 4
def kids_played_on_monday (julia : Type) : Nat := 6
def difference_monday_wednesday (julia : Type) : Nat := 2

-- Define the statement to prove
theorem kids_played_on_tuesday (julia : Type) :
  (kids_played_on_monday julia - difference_monday_wednesday julia) = kids_played_on_wednesday julia :=
by
  sorry

end kids_played_on_tuesday_l20_20749


namespace lcm_of_numbers_l20_20571

theorem lcm_of_numbers (a b lcm hcf : ℕ) (h_prod : a * b = 45276) (h_hcf : hcf = 22) (h_relation : a * b = hcf * lcm) : lcm = 2058 :=
by sorry

end lcm_of_numbers_l20_20571


namespace part_II_l20_20264

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 + (a - 1) * x - Real.log x

theorem part_II (a : ℝ) (h : a > 0) :
  ∀ x > 0, f a x ≥ 2 - (3 / (2 * a)) :=
sorry

end part_II_l20_20264


namespace Janice_earnings_after_deductions_l20_20607

def dailyEarnings : ℕ := 30
def daysWorked : ℕ := 6
def weekdayOvertimeRate : ℕ := 15
def weekendOvertimeRate : ℕ := 20
def weekdayOvertimeShifts : ℕ := 2
def weekendOvertimeShifts : ℕ := 1
def tipsReceived : ℕ := 10
def taxRate : ℝ := 0.10

noncomputable def calculateEarnings : ℝ :=
  let regularEarnings := dailyEarnings * daysWorked
  let overtimeEarnings := (weekdayOvertimeRate * weekdayOvertimeShifts) + (weekendOvertimeRate * weekendOvertimeShifts)
  let totalEarningsBeforeTax := regularEarnings + overtimeEarnings + tipsReceived
  let taxAmount := totalEarningsBeforeTax * taxRate
  totalEarningsBeforeTax - taxAmount

theorem Janice_earnings_after_deductions :
  calculateEarnings = 216 := by
  sorry

end Janice_earnings_after_deductions_l20_20607


namespace ellipse_eccentricity_l20_20418

theorem ellipse_eccentricity (a b : ℝ) (c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2) : (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end ellipse_eccentricity_l20_20418


namespace jaylen_dog_food_consumption_l20_20122

theorem jaylen_dog_food_consumption :
  ∀ (morning evening daily_consumption total_food : ℕ)
  (days : ℕ),
  (morning = evening) →
  (total_food = 32) →
  (days = 16) →
  (daily_consumption = total_food / days) →
  (morning + evening = daily_consumption) →
  morning = 1 := by
  intros morning evening daily_consumption total_food days h_eq h_total h_days h_daily h_sum
  sorry

end jaylen_dog_food_consumption_l20_20122


namespace circle_radius_of_tangent_parabolas_l20_20339

theorem circle_radius_of_tangent_parabolas :
  ∃ r : ℝ, 
  (∀ (x : ℝ), (x^2 + r = x)) →
  r = 1 / 4 :=
by
  sorry

end circle_radius_of_tangent_parabolas_l20_20339


namespace meaningful_fraction_x_range_l20_20124

theorem meaningful_fraction_x_range (x : ℝ) : (x-2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end meaningful_fraction_x_range_l20_20124


namespace trapezium_other_parallel_side_l20_20992

theorem trapezium_other_parallel_side (a : ℝ) (b d : ℝ) (area : ℝ) 
  (h1 : a = 18) (h2 : d = 15) (h3 : area = 285) : b = 20 :=
by
  sorry

end trapezium_other_parallel_side_l20_20992


namespace max_value_a_l20_20543

theorem max_value_a (a : ℝ) : (∀ x : ℝ, |x - 2| + |x - 8| ≥ a) ↔ a ≤ 6 := by
  sorry

end max_value_a_l20_20543


namespace stratified_sampling_number_l20_20063

noncomputable def students_in_grade_10 : ℕ := 150
noncomputable def students_in_grade_11 : ℕ := 180
noncomputable def students_in_grade_12 : ℕ := 210
noncomputable def total_students : ℕ := students_in_grade_10 + students_in_grade_11 + students_in_grade_12
noncomputable def sample_size : ℕ := 72
noncomputable def selection_probability : ℚ := sample_size / total_students
noncomputable def combined_students_grade_10_11 : ℕ := students_in_grade_10 + students_in_grade_11

theorem stratified_sampling_number :
  combined_students_grade_10_11 * selection_probability = 44 := 
by
  sorry

end stratified_sampling_number_l20_20063


namespace jims_speed_l20_20970

variable (x : ℝ)

theorem jims_speed (bob_speed : ℝ) (bob_head_start : ℝ) (time : ℝ) (bob_distance : ℝ) :
  bob_speed = 6 →
  bob_head_start = 1 →
  time = 1 / 3 →
  bob_distance = bob_speed * time →
  (x * time = bob_distance + bob_head_start) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jims_speed_l20_20970


namespace total_number_of_glasses_l20_20335

open scoped Nat

theorem total_number_of_glasses (x y : ℕ) (h1 : y = x + 16) (h2 : (12 * x + 16 * y) / (x + y) = 15) : 12 * x + 16 * y = 480 := by
  sorry

end total_number_of_glasses_l20_20335


namespace fractional_difference_l20_20201

def recurring72 : ℚ := 72 / 99
def decimal72 : ℚ := 72 / 100

theorem fractional_difference : recurring72 - decimal72 = 2 / 275 := by
  sorry

end fractional_difference_l20_20201


namespace find_t_l20_20319

-- Define the utility on both days
def utility_monday (t : ℝ) := t * (10 - t)
def utility_tuesday (t : ℝ) := (4 - t) * (t + 5)

-- Define the total hours spent on activities condition for both days
def total_hours_monday (t : ℝ) := t + (10 - t)
def total_hours_tuesday (t : ℝ) := (4 - t) + (t + 5)

theorem find_t : ∃ t : ℝ, t * (10 - t) = (4 - t) * (t + 5) ∧ 
                            total_hours_monday t ≥ 8 ∧ 
                            total_hours_tuesday t ≥ 8 :=
by
  sorry

end find_t_l20_20319


namespace total_cookies_l20_20004

-- Define the number of bags and the number of cookies per bag
def bags : ℕ := 37
def cookies_per_bag : ℕ := 19

-- State the theorem
theorem total_cookies : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l20_20004


namespace combined_area_of_removed_triangles_l20_20629

theorem combined_area_of_removed_triangles (s : ℝ) (x : ℝ) (h : 15 = ((s - 2 * x) ^ 2 + (s - 2 * x) ^ 2) ^ (1/2)) :
  2 * x ^ 2 = 28.125 :=
by
  -- The necessary proof will go here
  sorry

end combined_area_of_removed_triangles_l20_20629


namespace find_number_l20_20959

theorem find_number (x : ℤ) (h : x * 9999 = 806006795) : x = 80601 :=
sorry

end find_number_l20_20959


namespace twelve_sided_die_expected_value_l20_20617

theorem twelve_sided_die_expected_value : 
  ∃ (E : ℝ), (E = 6.5) :=
by
  let n := 12
  let numerator := n * (n + 1) / 2
  let expected_value := numerator / n
  use expected_value
  sorry

end twelve_sided_die_expected_value_l20_20617


namespace max_voters_after_T_l20_20682

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l20_20682


namespace steve_travel_time_l20_20616

noncomputable def total_travel_time (distance: ℕ) (speed_to_work: ℕ) (speed_back: ℕ) : ℕ :=
  (distance / speed_to_work) + (distance / speed_back)

theorem steve_travel_time : 
  ∀ (distance speed_back speed_to_work : ℕ), 
  (speed_to_work = speed_back / 2) → 
  speed_back = 15 → 
  distance = 30 → 
  total_travel_time distance speed_to_work speed_back = 6 := 
by
  intros
  rw [total_travel_time]
  sorry

end steve_travel_time_l20_20616


namespace lines_intersection_l20_20082

theorem lines_intersection :
  ∃ (t u : ℚ), 
    (∃ (x y : ℚ),
    (x = 2 - t ∧ y = 3 + 4 * t) ∧ 
    (x = -1 + 3 * u ∧ y = 6 + 5 * u) ∧ 
    (x = 28 / 17 ∧ y = 75 / 17)) := sorry

end lines_intersection_l20_20082


namespace arithmetic_sequence_sum_ratio_l20_20378

noncomputable def S (n : ℕ) (a_1 : ℚ) (d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_ratio (a_1 d : ℚ) (h : d ≠ 0) (h_ratio : (a_1 + 5 * d) / (a_1 + 2 * d) = 2) :
  S 6 a_1 d / S 3 a_1 d = 7 / 2 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l20_20378


namespace angle_bisector_segment_conditional_equality_l20_20661

theorem angle_bisector_segment_conditional_equality
  (a1 b1 a2 b2 : ℝ)
  (h1 : ∃ (P : ℝ), ∃ (e1 e2 : ℝ → ℝ), (e1 P = a1 ∧ e2 P = b1) ∧ (e1 P = a2 ∧ e2 P = b2)) :
  (1 / a1 + 1 / b1 = 1 / a2 + 1 / b2) :=
by 
  sorry

end angle_bisector_segment_conditional_equality_l20_20661


namespace households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l20_20839

namespace VehicleHouseholds

-- Definitions for the conditions
def totalHouseholds : ℕ := 250
def householdsNoVehicles : ℕ := 25
def householdsAllVehicles : ℕ := 36
def householdsCarOnly : ℕ := 62
def householdsBikeOnly : ℕ := 45
def householdsScooterOnly : ℕ := 30

-- Proof Statements
theorem households_with_two_types_of_vehicles :
  (totalHouseholds - householdsNoVehicles - householdsAllVehicles - 
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly)) = 52 := by
  sorry

theorem households_with_exactly_one_type_of_vehicle :
  (householdsCarOnly + householdsBikeOnly + householdsScooterOnly) = 137 := by
  sorry

theorem households_with_at_least_one_type_of_vehicle :
  (totalHouseholds - householdsNoVehicles) = 225 := by
  sorry

end VehicleHouseholds

end households_with_two_types_of_vehicles_households_with_exactly_one_type_of_vehicle_households_with_at_least_one_type_of_vehicle_l20_20839


namespace not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l20_20034

def vector_a : ℝ × ℝ := (3, 2)
def vector_vA : ℝ × ℝ := (3, -2)
def vector_vB : ℝ × ℝ := (2, 3)
def vector_vD : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem not_perpendicular_to_vA : dot_product vector_a vector_vA ≠ 0 := by sorry
theorem not_perpendicular_to_vB : dot_product vector_a vector_vB ≠ 0 := by sorry
theorem not_perpendicular_to_vD : dot_product vector_a vector_vD ≠ 0 := by sorry

end not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l20_20034


namespace correct_exponentiation_l20_20813

theorem correct_exponentiation : ∀ (x : ℝ), (x^(4/5))^(5/4) = x :=
by
  intro x
  sorry

end correct_exponentiation_l20_20813


namespace graph_intersection_l20_20083

noncomputable def log : ℝ → ℝ := sorry

lemma log_properties (a b : ℝ) (ha : 0 < a) (hb : 0 < b): log (a * b) = log a + log b := sorry

theorem graph_intersection :
  ∃! x : ℝ, 2 * log x = log (2 * x) :=
by
  sorry

end graph_intersection_l20_20083


namespace find_xy_value_l20_20222

theorem find_xy_value (x y z w : ℕ) (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) (h4 : y = w)
    (h5 : w + w = w * w) (h6 : z = 3) : x * y = 4 := by
  -- Given that w = 2 based on the conditions
  sorry

end find_xy_value_l20_20222


namespace rectangle_dimensions_l20_20990

theorem rectangle_dimensions (l w : ℝ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 2880) :
  (l = 86.833 ∧ w = 33.167) ∨ (l = 33.167 ∧ w = 86.833) :=
by
  sorry

end rectangle_dimensions_l20_20990


namespace quadratic_equation_m_l20_20743

theorem quadratic_equation_m (m : ℝ) (h1 : |m| + 1 = 2) (h2 : m + 1 ≠ 0) : m = 1 :=
sorry

end quadratic_equation_m_l20_20743


namespace age_of_B_l20_20012

theorem age_of_B (A B C : ℕ) 
  (h1 : (A + B + C) / 3 = 22)
  (h2 : (A + B) / 2 = 18)
  (h3 : (B + C) / 2 = 25) : 
  B = 20 := 
by
  sorry

end age_of_B_l20_20012


namespace cost_of_ABC_book_l20_20165

theorem cost_of_ABC_book (x : ℕ) 
  (h₁ : 8 = 8)  -- Cost of "TOP" book is 8 dollars
  (h₂ : 13 * 8 = 104)  -- Thirteen "TOP" books sold last week
  (h₃ : 104 - 4 * x = 12)  -- Difference in earnings is $12
  : x = 23 :=
sorry

end cost_of_ABC_book_l20_20165


namespace remainder_sum_div_40_l20_20787

variable (k m n : ℤ)
variables (a b c : ℤ)
variable (h1 : a % 80 = 75)
variable (h2 : b % 120 = 115)
variable (h3 : c % 160 = 155)

theorem remainder_sum_div_40 : (a + b + c) % 40 = 25 :=
by
  -- Use sorry as we are not required to fill in the proof
  sorry

end remainder_sum_div_40_l20_20787


namespace find_a_l20_20874

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 13 ∧ (51^2015 + a) % 13 = 0 → a = 1 :=
by { sorry }

end find_a_l20_20874


namespace speed_of_train_A_is_90_kmph_l20_20166

-- Definitions based on the conditions
def train_length_A := 225 -- in meters
def train_length_B := 150 -- in meters
def crossing_time := 15 -- in seconds

-- The total distance covered by train A to cross train B
def total_distance := train_length_A + train_length_B

-- The speed of train A in m/s
def speed_in_mps := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def mps_to_kmph (mps: ℕ) := mps * 36 / 10

-- The speed of train A in km/hr
def speed_in_kmph := mps_to_kmph speed_in_mps

-- The theorem to be proved
theorem speed_of_train_A_is_90_kmph : speed_in_kmph = 90 := by
  -- Proof steps go here
  sorry

end speed_of_train_A_is_90_kmph_l20_20166


namespace january_roses_l20_20277

theorem january_roses (r_october r_november r_december r_february r_january : ℕ)
  (h_october_november : r_november = r_october + 12)
  (h_november_december : r_december = r_november + 12)
  (h_december_january : r_january = r_december + 12)
  (h_january_february : r_february = r_january + 12) :
  r_january = 144 :=
by {
  -- The proof would go here.
  sorry
}

end january_roses_l20_20277


namespace derivative_at_2_l20_20640

noncomputable def f (x : ℝ) : ℝ := x

theorem derivative_at_2 : (deriv f 2) = 1 :=
by
  -- sorry, proof not included
  sorry

end derivative_at_2_l20_20640


namespace large_box_total_chocolate_bars_l20_20595

def number_of_small_boxes : ℕ := 15
def chocolate_bars_per_small_box : ℕ := 20
def total_chocolate_bars (n : ℕ) (m : ℕ) : ℕ := n * m

theorem large_box_total_chocolate_bars :
  total_chocolate_bars number_of_small_boxes chocolate_bars_per_small_box = 300 :=
by
  sorry

end large_box_total_chocolate_bars_l20_20595


namespace f_minus_ten_l20_20758

noncomputable def f : ℝ → ℝ := sorry

theorem f_minus_ten :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  (f 1 = 2) →
  f (-10) = 90 :=
by
  intros h1 h2
  sorry

end f_minus_ten_l20_20758


namespace gumballs_per_package_correct_l20_20751

-- Define the conditions
def total_gumballs_eaten : ℕ := 20
def number_of_boxes_finished : ℕ := 4

-- Define the target number of gumballs in each package
def gumballs_in_each_package := 5

theorem gumballs_per_package_correct :
  total_gumballs_eaten / number_of_boxes_finished = gumballs_in_each_package :=
by
  sorry

end gumballs_per_package_correct_l20_20751


namespace cost_price_radio_l20_20359

theorem cost_price_radio (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1305) 
  (h2 : loss_percentage = 0.13) 
  (h3 : SP = C * (1 - loss_percentage)) :
  C = 1500 := 
by 
  sorry

end cost_price_radio_l20_20359


namespace symmetric_graph_inverse_l20_20747

def f (x : ℝ) : ℝ := sorry -- We assume f is defined accordingly somewhere, as the inverse of ln.

theorem symmetric_graph_inverse (h : ∀ x, f (f x) = x) : f 2 = Real.exp 2 := by
  sorry

end symmetric_graph_inverse_l20_20747


namespace wise_men_correct_guesses_l20_20739

noncomputable def max_correct_guesses (n k : ℕ) : ℕ :=
  if n > k + 1 then n - k - 1 else 0

theorem wise_men_correct_guesses (n k : ℕ) :
  ∃ (m : ℕ), m = max_correct_guesses n k ∧ m ≤ n - k - 1 :=
by {
  sorry
}

end wise_men_correct_guesses_l20_20739


namespace temperature_conversion_correct_l20_20257

noncomputable def f_to_c (T : ℝ) : ℝ := (T - 32) * (5 / 9)

theorem temperature_conversion_correct :
  f_to_c 104 = 40 :=
by
  sorry

end temperature_conversion_correct_l20_20257


namespace john_pin_discount_l20_20522

theorem john_pin_discount :
  ∀ (n_pins price_per_pin amount_spent discount_rate : ℝ),
    n_pins = 10 →
    price_per_pin = 20 →
    amount_spent = 170 →
    discount_rate = ((n_pins * price_per_pin - amount_spent) / (n_pins * price_per_pin)) * 100 →
    discount_rate = 15 :=
by
  intros n_pins price_per_pin amount_spent discount_rate h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end john_pin_discount_l20_20522


namespace absolute_sum_l20_20771

def S (n : ℕ) : ℤ := n^2 - 4 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem absolute_sum : 
    (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|) = 68 :=
by
  sorry

end absolute_sum_l20_20771


namespace difference_of_values_l20_20829

theorem difference_of_values (num : Nat) : 
  (num = 96348621) →
  let face_value := 8
  let local_value := 8 * 10000
  local_value - face_value = 79992 := 
by
  intros h_eq
  have face_value := 8
  have local_value := 8 * 10000
  sorry

end difference_of_values_l20_20829


namespace most_stable_machine_l20_20327

noncomputable def var_A : ℝ := 10.3
noncomputable def var_B : ℝ := 6.9
noncomputable def var_C : ℝ := 3.5

theorem most_stable_machine :
  (var_C < var_B) ∧ (var_C < var_A) :=
by
  sorry

end most_stable_machine_l20_20327


namespace students_circle_no_regular_exists_zero_regular_school_students_l20_20190

noncomputable def students_circle_no_regular (n : ℕ) 
    (student : ℕ → String)
    (neighbor_right : ℕ → ℕ)
    (lies_to : ℕ → ℕ → Bool) : Prop :=
  ∀ i, student i = "Gymnasium student" →
    (if lies_to i (neighbor_right i)
     then (student (neighbor_right i) ≠ "Gymnasium student")
     else student (neighbor_right i) = "Gymnasium student") →
    (if lies_to (neighbor_right i) i
     then (student i ≠ "Gymnasium student")
     else student i = "Gymnasium student")

theorem students_circle_no_regular_exists_zero_regular_school_students
  (n : ℕ) 
  (student : ℕ → String)
  (neighbor_right : ℕ → ℕ)
  (lies_to : ℕ → ℕ → Bool)
  (h : students_circle_no_regular n student neighbor_right lies_to)
  : (∀ i, student i ≠ "Regular school student") :=
sorry

end students_circle_no_regular_exists_zero_regular_school_students_l20_20190


namespace find_divisor_l20_20947

-- Definitions of the conditions
def dividend : ℕ := 15968
def quotient : ℕ := 89
def remainder : ℕ := 37

-- The theorem stating the proof problem
theorem find_divisor (D : ℕ) (h : dividend = D * quotient + remainder) : D = 179 :=
sorry

end find_divisor_l20_20947


namespace average_fuel_consumption_correct_l20_20157

def distance_to_x : ℕ := 150
def distance_to_y : ℕ := 220
def fuel_to_x : ℕ := 20
def fuel_to_y : ℕ := 15

def total_distance : ℕ := distance_to_x + distance_to_y
def total_fuel_used : ℕ := fuel_to_x + fuel_to_y
def avg_fuel_consumption : ℚ := total_fuel_used / total_distance

theorem average_fuel_consumption_correct :
  avg_fuel_consumption = 0.0946 := by
  sorry

end average_fuel_consumption_correct_l20_20157


namespace hyperbola_eccentricity_l20_20746

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (c : ℝ) (h3 : a^2 + b^2 = c^2) 
  (h4 : ∃ M : ℝ × ℝ, (M.fst^2 / a^2 - M.snd^2 / b^2 = 1) ∧ (M.snd^2 = 8 * M.fst)
    ∧ (|M.fst - 2| + |M.snd| = 5)) : 
  (c / a = 2) :=
by
  sorry

end hyperbola_eccentricity_l20_20746


namespace average_annual_growth_rate_equation_l20_20081

variable (x : ℝ)
axiom seventh_to_ninth_reading_increase : (1 : ℝ) * (1 + x) * (1 + x) = 1.21

theorem average_annual_growth_rate_equation :
  100 * (1 + x) ^ 2 = 121 :=
by
  have h : (1 : ℝ) * (1 + x) * (1 + x) = 1.21 := seventh_to_ninth_reading_increase x
  sorry

end average_annual_growth_rate_equation_l20_20081


namespace isabella_canadian_dollars_sum_l20_20453

def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + ((n / 10) % 10)

theorem isabella_canadian_dollars_sum (d : Nat) (H: 10 * d = 7 * d + 280) : sum_of_digits d = 12 :=
by
  sorry

end isabella_canadian_dollars_sum_l20_20453


namespace sqrt_pow_simplification_l20_20532

theorem sqrt_pow_simplification :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 125 * (5 ^ (3 / 4)) :=
by
  sorry

end sqrt_pow_simplification_l20_20532


namespace team_A_wins_2_1_team_B_wins_l20_20817

theorem team_A_wins_2_1 (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (2 * p_a * p_b) * p_a = 0.288 := by
  sorry

theorem team_B_wins (p_a p_b : ℝ)
  (h1 : p_a = 0.6)
  (h2 : p_b = 0.4)
  (h3 : ∀ {x y: ℝ}, x + y = 1)
  (h4 : ∃ n : ℕ, n = 3) : (p_b * p_b) + (2 * p_a * p_b * p_b) = 0.352 := by
  sorry

end team_A_wins_2_1_team_B_wins_l20_20817


namespace vasya_max_points_l20_20287

theorem vasya_max_points (cards : Finset (Fin 36)) 
  (petya_hand vasya_hand : Finset (Fin 36)) 
  (h_disjoint : Disjoint petya_hand vasya_hand)
  (h_union : petya_hand ∪ vasya_hand = cards)
  (h_card : cards.card = 36)
  (h_half : petya_hand.card = 18 ∧ vasya_hand.card = 18) : 
  ∃ max_points : ℕ, max_points = 15 := 
sorry

end vasya_max_points_l20_20287


namespace company_employees_count_l20_20155

theorem company_employees_count :
  ∃ E : ℕ, E = 80 + 100 - 30 + 20 := 
sorry

end company_employees_count_l20_20155


namespace fluorescent_tubes_count_l20_20882

theorem fluorescent_tubes_count 
  (x y : ℕ)
  (h1 : x + y = 13)
  (h2 : x / 3 + y / 2 = 5) : x = 9 :=
by
  sorry

end fluorescent_tubes_count_l20_20882


namespace find_ellipse_equation_l20_20974

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ c : ℝ, a > b ∧ b > 0 ∧ 4 * a = 16 ∧ |c| = 2 ∧ a^2 = b^2 + c^2

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b) → (∃ b : ℝ, (a = 4) ∧ (b > 0) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, (x^2 / 16) + (y^2 / 12) = 1)) :=
by {
  sorry
}

end find_ellipse_equation_l20_20974


namespace swimming_pool_paint_area_l20_20650

theorem swimming_pool_paint_area :
  let length := 20 -- The pool is 20 meters long
  let width := 12  -- The pool is 12 meters wide
  let depth := 2   -- The pool is 2 meters deep
  let area_longer_walls := 2 * length * depth
  let area_shorter_walls := 2 * width * depth
  let total_side_wall_area := area_longer_walls + area_shorter_walls
  let floor_area := length * width
  let total_area_to_paint := total_side_wall_area + floor_area
  total_area_to_paint = 368 :=
by
  sorry

end swimming_pool_paint_area_l20_20650


namespace g_cross_horizontal_asymptote_at_l20_20188

noncomputable def g (x : ℝ) : ℝ :=
  (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 6)

theorem g_cross_horizontal_asymptote_at (x : ℝ) : g x = 3 ↔ x = 13 / 4 :=
by
  sorry

end g_cross_horizontal_asymptote_at_l20_20188


namespace mow_lawn_payment_l20_20997

theorem mow_lawn_payment (bike_cost weekly_allowance babysitting_rate babysitting_hours money_saved target_savings mowing_payment : ℕ) 
  (h1 : bike_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : babysitting_rate = 7)
  (h4 : babysitting_hours = 2)
  (h5 : money_saved = 65)
  (h6 : target_savings = 6) :
  mowing_payment = 10 :=
sorry

end mow_lawn_payment_l20_20997


namespace total_books_in_school_l20_20944

theorem total_books_in_school (tables_A tables_B tables_C : ℕ)
  (books_per_table_A books_per_table_B books_per_table_C : ℕ → ℕ)
  (hA : tables_A = 750)
  (hB : tables_B = 500)
  (hC : tables_C = 850)
  (h_books_per_table_A : ∀ n, books_per_table_A n = 3 * n / 5)
  (h_books_per_table_B : ∀ n, books_per_table_B n = 2 * n / 5)
  (h_books_per_table_C : ∀ n, books_per_table_C n = n / 3) :
  books_per_table_A tables_A + books_per_table_B tables_B + books_per_table_C tables_C = 933 :=
by sorry

end total_books_in_school_l20_20944


namespace total_initial_passengers_l20_20207

theorem total_initial_passengers (M W : ℕ) 
  (h1 : W = M / 3) 
  (h2 : M - 24 = W + 12) : 
  M + W = 72 :=
sorry

end total_initial_passengers_l20_20207


namespace petya_cannot_have_equal_coins_l20_20846

def petya_initial_two_kopeck_coins : Nat := 1
def petya_initial_ten_kopeck_coins : Nat := 0
def petya_use_ten_kopeck (T G : Nat) : Nat := G - 1 + T + 5
def petya_use_two_kopeck (T G : Nat) : Nat := T - 1 + G + 5

theorem petya_cannot_have_equal_coins : ¬ (∃ n : Nat, 
  ∃ T G : Nat, 
    T = G ∧ 
    (n = petya_use_ten_kopeck T G ∨ n = petya_use_two_kopeck T G ∨ n = petya_initial_two_kopeck_coins + petya_initial_ten_kopeck_coins)) := 
by
  sorry

end petya_cannot_have_equal_coins_l20_20846


namespace two_n_minus_one_lt_n_plus_one_squared_l20_20925

theorem two_n_minus_one_lt_n_plus_one_squared (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1) ^ 2 := 
by
  sorry

end two_n_minus_one_lt_n_plus_one_squared_l20_20925


namespace maxwell_walking_speed_l20_20009

theorem maxwell_walking_speed :
  ∀ (distance_between_homes : ℕ)
    (brad_speed : ℕ)
    (middle_travel_maxwell : ℕ)
    (middle_distance : ℕ),
    distance_between_homes = 36 →
    brad_speed = 4 →
    middle_travel_maxwell = 12 →
    middle_distance = 18 →
    (middle_travel_maxwell : ℕ) / (8 : ℕ) = (middle_distance - middle_travel_maxwell) / brad_speed :=
  sorry

end maxwell_walking_speed_l20_20009


namespace find_day_for_balance_l20_20050

-- Define the initial conditions and variables
def initialEarnings : ℤ := 20
def secondDaySpending : ℤ := 15
variables (X Y : ℤ)

-- Define the function for net balance on day D
def netBalance (D : ℤ) : ℤ :=
  initialEarnings + (D - 1) * X - (secondDaySpending + (D - 2) * Y)

-- The main theorem proving the day D for net balance of Rs. 60
theorem find_day_for_balance (X Y : ℤ) : ∃ D : ℤ, netBalance X Y D = 60 → 55 = (D + 1) * (X - Y) :=
by
  sorry

end find_day_for_balance_l20_20050


namespace abs_eq_zero_solve_l20_20681

theorem abs_eq_zero_solve (a b : ℚ) (h : |a - (1/2 : ℚ)| + |b + 5| = 0) : a + b = -9 / 2 := 
by
  sorry

end abs_eq_zero_solve_l20_20681


namespace club_president_vice_president_combinations_144_l20_20725

variables (boys_total girls_total : Nat)
variables (senior_boys junior_boys senior_girls junior_girls : Nat)
variables (choose_president_vice_president : Nat)

-- Define the conditions
def club_conditions : Prop :=
  boys_total = 12 ∧
  girls_total = 12 ∧
  senior_boys = 6 ∧
  junior_boys = 6 ∧
  senior_girls = 6 ∧
  junior_girls = 6

-- Define the proof problem
def president_vice_president_combinations (boys_total girls_total senior_boys junior_boys senior_girls junior_girls : Nat) : Nat :=
  2 * senior_boys * junior_boys + 2 * senior_girls * junior_girls

-- The main theorem to prove
theorem club_president_vice_president_combinations_144 :
  club_conditions boys_total girls_total senior_boys junior_boys senior_girls junior_girls →
  president_vice_president_combinations boys_total girls_total senior_boys junior_boys senior_girls junior_girls = 144 :=
sorry

end club_president_vice_president_combinations_144_l20_20725


namespace log_expression_eval_find_m_from_conditions_l20_20094

-- (1) Prove that lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3.
theorem log_expression_eval : 
  Real.logb 10 (5^2) + (2 / 3) * Real.logb 10 8 + Real.logb 10 5 * Real.logb 10 20 + (Real.logb 10 2)^2 = 3 := 
sorry

-- (2) Given 2^a = 5^b = m and 1/a + 1/b = 2, prove that m = sqrt(10).
theorem find_m_from_conditions (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 :=
sorry

end log_expression_eval_find_m_from_conditions_l20_20094


namespace choir_members_l20_20679

theorem choir_members (n : ℕ) :
  (150 < n) ∧ (n < 250) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 8 = 5) → n = 159 :=
by
  sorry

end choir_members_l20_20679


namespace inequality_x_n_l20_20022

theorem inequality_x_n (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : n ≥ 2) : (1 - x)^n + (1 + x)^n < 2^n := 
sorry

end inequality_x_n_l20_20022


namespace problem_solution_l20_20972

open Real

/-- If (y / 6) / 3 = 6 / (y / 3), then y is ±18. -/
theorem problem_solution (y : ℝ) (h : (y / 6) / 3 = 6 / (y / 3)) : y = 18 ∨ y = -18 :=
by
  sorry

end problem_solution_l20_20972


namespace percentage_of_non_honda_red_cars_l20_20101

/-- 
Total car population in Chennai is 9000.
Honda cars in Chennai is 5000.
Out of every 100 Honda cars, 90 are red.
60% of the total car population is red.
Prove that the percentage of non-Honda cars that are red is 22.5%.
--/
theorem percentage_of_non_honda_red_cars 
  (total_cars : ℕ) (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) (total_red_ratio : ℚ) 
  (h : total_cars = 9000) 
  (h1 : honda_cars = 5000) 
  (h2 : red_honda_ratio = 90 / 100) 
  (h3 : total_red_ratio = 60 / 100) : 
  (900 / (9000 - 5000) * 100 = 22.5) := 
sorry

end percentage_of_non_honda_red_cars_l20_20101


namespace positive_difference_of_two_numbers_l20_20912

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l20_20912


namespace population_present_l20_20529

variable (P : ℝ)

theorem population_present (h1 : P * 0.90 = 450) : P = 500 :=
by
  sorry

end population_present_l20_20529


namespace evaluate_expression_l20_20878

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem evaluate_expression : spadesuit 3 (spadesuit 6 5) = -112 := by
  sorry

end evaluate_expression_l20_20878


namespace find_k_for_min_value_zero_l20_20175

theorem find_k_for_min_value_zero :
  (∃ k : ℝ, ∀ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0 ∧
                         ∃ x y : ℝ, 9 * x^2 - 12 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) →
  k = 3 / 2 :=
sorry

end find_k_for_min_value_zero_l20_20175


namespace geometric_condition_l20_20833

def Sn (p : ℤ) (n : ℕ) : ℤ := p * 2^n + 2

def an (p : ℤ) (n : ℕ) : ℤ :=
  if n = 1 then Sn p n
  else Sn p n - Sn p (n - 1)

def is_geometric_progression (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ r : ℤ, an p n = an p (n - 1) * r

theorem geometric_condition (p : ℤ) :
  is_geometric_progression p ↔ p = -2 :=
sorry

end geometric_condition_l20_20833


namespace cost_price_of_cupboard_l20_20059

theorem cost_price_of_cupboard (C S S_profit : ℝ) (h1 : S = 0.88 * C) (h2 : S_profit = 1.12 * C) (h3 : S_profit - S = 1650) :
  C = 6875 := by
  sorry

end cost_price_of_cupboard_l20_20059


namespace smallest_value_of_x_l20_20143

theorem smallest_value_of_x (x : ℝ) (h : 4 * x^2 - 20 * x + 24 = 0) : x = 2 :=
    sorry

end smallest_value_of_x_l20_20143


namespace train_length_is_200_l20_20374

noncomputable def train_length 
  (speed_kmh : ℕ) 
  (time_s: ℕ) : ℕ := 
  ((speed_kmh * 1000) / 3600) * time_s

theorem train_length_is_200
  (h_speed : 40 = 40)
  (h_time : 18 = 18) :
  train_length 40 18 = 200 :=
sorry

end train_length_is_200_l20_20374


namespace perfect_square_expression_l20_20741

theorem perfect_square_expression (x y z : ℤ) :
    9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3 * x * y * z) =
      ((x + y + z)^2 - 6 * (x * y + y * z + z * x))^2 := 
by 
  sorry

end perfect_square_expression_l20_20741


namespace find_x_and_verify_l20_20114

theorem find_x_and_verify (x : ℤ) (h : (x - 14) / 10 = 4) : (x - 5) / 7 = 7 := 
by 
  sorry

end find_x_and_verify_l20_20114


namespace faye_country_albums_l20_20336

theorem faye_country_albums (C : ℕ) (h1 : 6 * C + 18 = 30) : C = 2 :=
by
  -- This is the theorem statement with the necessary conditions and question
  sorry

end faye_country_albums_l20_20336


namespace translate_parabola_l20_20950

-- Translating the parabola y = (x-2)^2 - 8 three units left and five units up
theorem translate_parabola (x y : ℝ) :
  y = (x - 2) ^ 2 - 8 →
  y = ((x + 3) - 2) ^ 2 - 8 + 5 →
  y = (x + 1) ^ 2 - 3 := by
sorry

end translate_parabola_l20_20950


namespace trig_identity_example_l20_20085

theorem trig_identity_example :
  (2 * (Real.sin (Real.pi / 6)) - Real.tan (Real.pi / 4)) = 0 :=
by
  -- Definitions from conditions
  have h1 : Real.sin (Real.pi / 6) = 1/2 := Real.sin_pi_div_six
  have h2 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [h1, h2]
  sorry -- The proof is omitted as per instructions

end trig_identity_example_l20_20085


namespace feasibility_orderings_l20_20274

theorem feasibility_orderings (a : ℝ) :
  (a ≠ 0) →
  (∀ a > 0, a < 2 * a ∧ 2 * a < 3 * a + 1) ∧
  ¬∃ a, a < 3 * a + 1 ∧ 3 * a + 1 < 2 * a ∧ 2 * a < 3 * a + 1 ∧ a ≠ 0 ∧ a > 0 ∧ a < -1 / 2 ∧ a < 0 ∧ a < -1 ∧ a < -1 / 2 ∧ a < -1 / 2 ∧ a < 0 :=
sorry

end feasibility_orderings_l20_20274


namespace not_perfect_square_l20_20862

theorem not_perfect_square : ¬ ∃ x : ℝ, x^2 = 7^2025 := by
  sorry

end not_perfect_square_l20_20862


namespace river_bank_bottom_width_l20_20407

/-- 
The cross-section of a river bank is a trapezium with a 12 m wide top and 
a certain width at the bottom. The area of the cross-section is 500 sq m 
and the depth is 50 m. Prove that the width at the bottom is 8 m.
-/
theorem river_bank_bottom_width (area height top_width : ℝ) (h_area: area = 500) 
(h_height : height = 50) (h_top_width : top_width = 12) : ∃ b : ℝ, (1 / 2) * (top_width + b) * height = area ∧ b = 8 :=
by
  use 8
  sorry

end river_bank_bottom_width_l20_20407


namespace circle_tangent_x_axis_at_origin_l20_20262

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0) ↔ (D = 0 ∧ F = 0 ∧ E ≠ 0) :=
sorry

end circle_tangent_x_axis_at_origin_l20_20262


namespace carousel_problem_l20_20962

theorem carousel_problem (n : ℕ) : 
  (∃ (f : Fin n → Fin n), 
    (∀ i, f (f i) = i) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ i, f i < n)) ↔ 
  (Even n) := 
sorry

end carousel_problem_l20_20962


namespace speed_of_train_is_correct_l20_20719

-- Given conditions
def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_to_cross_bridge : ℝ := 20

-- Derived definition
def total_distance : ℝ := length_of_train + length_of_bridge

-- Goal to be proved
theorem speed_of_train_is_correct : total_distance / time_to_cross_bridge = 18.5 := 
by
  sorry

end speed_of_train_is_correct_l20_20719


namespace best_fitting_model_l20_20061

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.25) 
  (h2 : R2_2 = 0.50) 
  (h3 : R2_3 = 0.80) 
  (h4 : R2_4 = 0.98) : 
  (R2_4 = max (max R2_1 (max R2_2 R2_3)) R2_4) :=
by
  sorry

end best_fitting_model_l20_20061


namespace problem_statement_l20_20897

variable (a b c : ℝ)

theorem problem_statement
  (h1 : a + b = 100)
  (h2 : b + c = 140) :
  c - a = 40 :=
sorry

end problem_statement_l20_20897


namespace find_number_250_l20_20555

theorem find_number_250 (N : ℤ)
  (h1 : 5 * N = 8 * 156 + 2): N = 250 :=
sorry

end find_number_250_l20_20555


namespace trajectory_of_A_l20_20042

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A_l20_20042


namespace right_triangle_area_and_perimeter_l20_20458

theorem right_triangle_area_and_perimeter (a c : ℕ) (h₁ : c = 13) (h₂ : a = 5) :
  ∃ (b : ℕ), b^2 = c^2 - a^2 ∧
             (1/2 : ℝ) * (a : ℝ) * (b : ℝ) = 30 ∧
             (a + b + c : ℕ) = 30 :=
by
  sorry

end right_triangle_area_and_perimeter_l20_20458


namespace sqrt_seven_lt_three_l20_20686

theorem sqrt_seven_lt_three : Real.sqrt 7 < 3 :=
by
  sorry

end sqrt_seven_lt_three_l20_20686


namespace decode_division_problem_l20_20093

theorem decode_division_problem :
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  dividend / divisor = quotient :=
by {
  -- Definitions of given and derived values
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  -- The statement to prove
  sorry
}

end decode_division_problem_l20_20093


namespace sqrt_of_25_l20_20439

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l20_20439


namespace circumcircle_eq_l20_20185

-- Definitions and conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_P : (ℝ × ℝ) := (4, 2)
def is_tangent_point (x y : ℝ) : Prop := sorry -- You need a proper definition for tangency

theorem circumcircle_eq :
  ∃ (hA : is_tangent_point 0 2) (hB : ∃ x y, is_tangent_point x y),
  ∃ (x y : ℝ), (circle_eq 0 2 ∧ circle_eq x y) ∧ (x-2)^2 + (y-1)^2 = 5 :=
  sorry

end circumcircle_eq_l20_20185


namespace scientific_notation_650000_l20_20484

theorem scientific_notation_650000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 650000 = a * 10 ^ n ∧ a = 6.5 ∧ n = 5 :=
  sorry

end scientific_notation_650000_l20_20484


namespace find_number_l20_20347

theorem find_number (x : ℕ) (h : 15 * x = x + 196) : 15 * x = 210 :=
by
  sorry

end find_number_l20_20347


namespace simplify_division_l20_20975

theorem simplify_division (a b c d : ℕ) (h1 : a = 27) (h2 : b = 10^12) (h3 : c = 9) (h4 : d = 10^4) :
  ((a * b) / (c * d) = 300000000) :=
by {
  sorry
}

end simplify_division_l20_20975


namespace john_fixes_8_computers_l20_20987

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l20_20987


namespace area_of_new_triangle_geq_twice_sum_of_areas_l20_20136

noncomputable def area_of_triangle (a b c : ℝ) (alpha : ℝ) : ℝ :=
  0.5 * a * b * (Real.sin alpha)

theorem area_of_new_triangle_geq_twice_sum_of_areas
  (a1 b1 c a2 b2 alpha : ℝ)
  (h1 : a1 <= b1) (h2 : b1 <= c) (h3 : a2 <= b2) (h4 : b2 <= c) :
  let α_1 := Real.arcsin ((a1 + a2) / (2 * c))
  let area1 := area_of_triangle a1 b1 c alpha
  let area2 := area_of_triangle a2 b2 c alpha
  let area_new := area_of_triangle (a1 + a2) (b1 + b2) (2 * c) α_1
  area_new >= 2 * (area1 + area2) :=
sorry

end area_of_new_triangle_geq_twice_sum_of_areas_l20_20136


namespace total_eggs_found_l20_20612

def eggs_from_club_house : ℕ := 40
def eggs_from_park : ℕ := 25
def eggs_from_town_hall : ℕ := 15

theorem total_eggs_found : eggs_from_club_house + eggs_from_park + eggs_from_town_hall = 80 := by
  -- Proof of this theorem
  sorry

end total_eggs_found_l20_20612


namespace part1_part2_l20_20088

variable (x : ℝ)
def A : ℝ := 2 * x^2 - 3 * x + 2
def B : ℝ := x^2 - 3 * x - 2

theorem part1 : A x - B x = x^2 + 4 := sorry

theorem part2 (h : x = -2) : A x - B x = 8 := sorry

end part1_part2_l20_20088


namespace solve_y_l20_20995

theorem solve_y (y : ℝ) (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
by
  sorry

end solve_y_l20_20995


namespace evaluate_expression_l20_20558

variable (a b c d e : ℝ)

-- The equivalent proof problem statement
theorem evaluate_expression 
  (h : (a / b * c - d + e = a / (b * c - d - e))) : 
  a / b * c - d + e = a / (b * c - d - e) :=
by 
  exact h

-- Placeholder for the proof
#check evaluate_expression

end evaluate_expression_l20_20558


namespace max_xy_l20_20977

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 9 * y = 12) : xy ≤ 4 :=
by
sorry

end max_xy_l20_20977


namespace cost_comparison_l20_20647

def cost_function_A (x : ℕ) : ℕ := 450 * x + 1000
def cost_function_B (x : ℕ) : ℕ := 500 * x

theorem cost_comparison (x : ℕ) : 
  if x = 20 then cost_function_A x = cost_function_B x 
  else if x < 20 then cost_function_A x > cost_function_B x 
  else cost_function_A x < cost_function_B x :=
sorry

end cost_comparison_l20_20647


namespace product_xyz_l20_20952

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l20_20952


namespace abc_positive_l20_20965

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end abc_positive_l20_20965


namespace inequality_must_hold_l20_20367

section
variables {a b c : ℝ}

theorem inequality_must_hold (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry
end

end inequality_must_hold_l20_20367


namespace simplify_expression_l20_20921

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l20_20921


namespace exists_multiple_of_power_of_two_non_zero_digits_l20_20515

open Nat

theorem exists_multiple_of_power_of_two_non_zero_digits (k : ℕ) (h : 0 < k) : 
  ∃ m : ℕ, (2^k ∣ m) ∧ (∀ d ∈ digits 10 m, d ≠ 0) :=
sorry

end exists_multiple_of_power_of_two_non_zero_digits_l20_20515


namespace sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l20_20636

noncomputable def sec (x : ℝ) := 1 / Real.cos x
noncomputable def csc (x : ℝ) := 1 / Real.sin x

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := by
  sorry

theorem csc_150_eq_2 : csc (150 * Real.pi / 180) = 2 := by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_csc_150_eq_2_l20_20636


namespace count_arithmetic_sequence_l20_20468

theorem count_arithmetic_sequence :
  ∃ n, 195 - (n - 1) * 3 = 12 ∧ n = 62 :=
by {
  sorry
}

end count_arithmetic_sequence_l20_20468


namespace diving_classes_on_weekdays_l20_20489

theorem diving_classes_on_weekdays 
  (x : ℕ) 
  (weekend_classes_per_day : ℕ := 4)
  (people_per_class : ℕ := 5)
  (total_people_3_weeks : ℕ := 270)
  (weekend_days : ℕ := 2)
  (total_weeks : ℕ := 3)
  (weekend_total_classes : ℕ := weekend_classes_per_day * weekend_days * total_weeks) 
  (total_people_weekends : ℕ := weekend_total_classes * people_per_class) 
  (total_people_weekdays : ℕ := total_people_3_weeks - total_people_weekends)
  (weekday_classes_needed : ℕ := total_people_weekdays / people_per_class)
  (weekly_weekday_classes : ℕ := weekday_classes_needed / total_weeks)
  (h : weekly_weekday_classes = x)
  : x = 10 := sorry

end diving_classes_on_weekdays_l20_20489


namespace rational_sum_zero_l20_20922

theorem rational_sum_zero (x1 x2 x3 x4 : ℚ)
  (h1 : x1 = x2 + x3 + x4)
  (h2 : x2 = x1 + x3 + x4)
  (h3 : x3 = x1 + x2 + x4)
  (h4 : x4 = x1 + x2 + x3) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 := 
sorry

end rational_sum_zero_l20_20922


namespace smallest_n_l20_20786

-- Define the costs of candies
def cost_purple := 24
def cost_yellow := 30

-- Define the number of candies Lara can buy
def pieces_red := 10
def pieces_green := 16
def pieces_blue := 18
def pieces_yellow := 22

-- Define the total money Lara has equivalently expressed by buying candies
def lara_total_money (n : ℕ) := n * cost_purple

-- Prove the smallest value of n that satisfies the conditions stated
theorem smallest_n : ∀ n : ℕ, 
  (lara_total_money n = 10 * pieces_red * cost_purple) ∧
  (lara_total_money n = 16 * pieces_green * cost_purple) ∧
  (lara_total_money n = 18 * pieces_blue * cost_purple) ∧
  (lara_total_money n = pieces_yellow * cost_yellow) → 
  n = 30 :=
by
  intro
  sorry

end smallest_n_l20_20786


namespace inequality_holds_for_real_numbers_l20_20414

theorem inequality_holds_for_real_numbers (a1 a2 a3 a4 : ℝ) (h1 : 1 < a1) 
  (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) : 
  8 * (a1 * a2 * a3 * a4 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) :=
by sorry

end inequality_holds_for_real_numbers_l20_20414


namespace percent_increase_in_maintenance_time_l20_20365

theorem percent_increase_in_maintenance_time (original_time new_time : ℝ) (h1 : original_time = 25) (h2 : new_time = 30) : 
  ((new_time - original_time) / original_time) * 100 = 20 :=
by
  sorry

end percent_increase_in_maintenance_time_l20_20365


namespace simplify_polynomial_l20_20373

variable {R : Type} [CommRing R] (s : R)

theorem simplify_polynomial :
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 4) = -4 * s + 1 :=
by
  sorry

end simplify_polynomial_l20_20373


namespace usual_time_eq_three_l20_20584

variable (S T : ℝ)
variable (usual_speed : S > 0)
variable (usual_time : T > 0)
variable (reduced_speed : S' = 6/7 * S)
variable (reduced_time : T' = T + 0.5)

theorem usual_time_eq_three (h : 7/6 = T' / T) : T = 3 :=
by
  -- proof to be filled in
  sorry

end usual_time_eq_three_l20_20584


namespace fraction_relationship_l20_20170

theorem fraction_relationship (a b c : ℚ)
  (h1 : a / b = 3 / 5)
  (h2 : b / c = 2 / 7) :
  c / a = 35 / 6 :=
by
  sorry

end fraction_relationship_l20_20170


namespace arithmetic_seq_sum_a4_a6_l20_20942

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum_a4_a6 (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_root1 : a 3 ^ 2 - 3 * a 3 + 1 = 0)
  (h_root2 : a 7 ^ 2 - 3 * a 7 + 1 = 0) :
  a 4 + a 6 = 3 :=
sorry

end arithmetic_seq_sum_a4_a6_l20_20942


namespace jills_daily_earnings_first_month_l20_20766

-- Definitions based on conditions
variable (x : ℕ) -- daily earnings in the first month
def total_earnings_first_month := 30 * x
def total_earnings_second_month := 30 * (2 * x)
def total_earnings_third_month := 15 * (2 * x)
def total_earnings_three_months := total_earnings_first_month x + total_earnings_second_month x + total_earnings_third_month x

-- The theorem we need to prove
theorem jills_daily_earnings_first_month
  (h : total_earnings_three_months x = 1200) : x = 10 :=
sorry

end jills_daily_earnings_first_month_l20_20766


namespace balance_five_diamonds_bullets_l20_20460

variables (a b c : ℝ)

-- Conditions
def condition1 : Prop := 4 * a + 2 * b = 12 * c
def condition2 : Prop := 2 * a = b + 4 * c

-- Theorem statement
theorem balance_five_diamonds_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 5 * b = 5 * c :=
by
  sorry

end balance_five_diamonds_bullets_l20_20460


namespace unique_polynomial_l20_20849

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem unique_polynomial 
  (a b c : ℝ) 
  (extremes : f' a b c 1 = 0 ∧ f' a b c (-1) = 0) 
  (tangent_slope : f' a b c 0 = -3)
  : f a b c = f 1 0 (-3) := sorry

end unique_polynomial_l20_20849


namespace price_per_glass_second_day_l20_20429

theorem price_per_glass_second_day 
  (O W : ℕ)  -- O is the amount of orange juice used on each day, W is the amount of water used on the first day
  (V : ℕ)   -- V is the volume of one glass
  (P₁ : ℚ)  -- P₁ is the price per glass on the first day
  (P₂ : ℚ)  -- P₂ is the price per glass on the second day
  (h1 : W = O)  -- First day, water is equal to orange juice
  (h2 : V > 0)  -- Volume of one glass > 0
  (h3 : P₁ = 0.48)  -- Price per glass on the first day
  (h4 : (2 * O / V) * P₁ = (3 * O / V) * P₂)  -- Revenue's are the same
  : P₂ = 0.32 :=  -- Prove that price per glass on the second day is 0.32
by
  sorry

end price_per_glass_second_day_l20_20429


namespace area_of_square_field_l20_20526

-- Define side length
def sideLength : ℕ := 14

-- Define the area function for a square
def area_of_square (side : ℕ) : ℕ := side * side

-- Prove that the area of the square with side length 14 meters is 196 square meters
theorem area_of_square_field : area_of_square sideLength = 196 := by
  sorry

end area_of_square_field_l20_20526


namespace monochromatic_triangle_l20_20461

def R₃ (n : ℕ) : ℕ := sorry

theorem monochromatic_triangle {n : ℕ} (h1 : R₃ 2 = 6)
  (h2 : ∀ n, R₃ (n + 1) ≤ (n + 1) * R₃ n - n + 1) :
  R₃ n ≤ 3 * Nat.factorial n :=
by
  induction n with
  | zero => sorry -- base case proof
  | succ n ih => sorry -- inductive step proof

end monochromatic_triangle_l20_20461


namespace algebraic_expression_identity_l20_20812

theorem algebraic_expression_identity (a b x : ℕ) (h : x * 3 * a * b = 3 * a * a * b) : x = a :=
sorry

end algebraic_expression_identity_l20_20812


namespace find_width_of_plot_l20_20447

def length : ℕ := 90
def poles : ℕ := 52
def distance_between_poles : ℕ := 5
def perimeter : ℕ := poles * distance_between_poles

theorem find_width_of_plot (perimeter_eq : perimeter = 2 * (length + width)) : width = 40 := by
  sorry

end find_width_of_plot_l20_20447


namespace relationship_x2_ax_bx_l20_20731

variable {x a b : ℝ}

theorem relationship_x2_ax_bx (h1 : x < a) (h2 : a < 0) (h3 : b > 0) : x^2 > ax ∧ ax > bx :=
by
  sorry

end relationship_x2_ax_bx_l20_20731


namespace fraction_non_throwers_left_handed_l20_20427

theorem fraction_non_throwers_left_handed (total_players : ℕ) (num_throwers : ℕ) (total_right_handed : ℕ) (all_throwers_right_handed : ∀ x, x < num_throwers → true) (num_right_handed := total_right_handed - num_throwers) (non_throwers := total_players - num_throwers) (num_left_handed := non_throwers - num_right_handed) : 
    total_players = 70 → 
    num_throwers = 40 → 
    total_right_handed = 60 → 
    (∃ f: ℚ, f = num_left_handed / non_throwers ∧ f = 1/3) := 
by {
  sorry
}

end fraction_non_throwers_left_handed_l20_20427


namespace sufficient_y_wages_l20_20419

noncomputable def days_sufficient_for_y_wages (Wx Wy : ℝ) (total_money : ℝ) : ℝ :=
  total_money / Wy

theorem sufficient_y_wages
  (Wx Wy : ℝ)
  (H1 : ∀(D : ℝ), total_money = D * Wx → D = 36 )
  (H2 : total_money = 20 * (Wx + Wy)) :
  days_sufficient_for_y_wages Wx Wy total_money = 45 := by
  sorry

end sufficient_y_wages_l20_20419


namespace solve_for_a_l20_20847

theorem solve_for_a (a : ℝ) (y : ℝ) (h1 : 4 * 2 + y = a) (h2 : 2 * 2 + 5 * y = 3 * a) : a = 18 :=
  sorry

end solve_for_a_l20_20847


namespace total_good_vegetables_l20_20850

theorem total_good_vegetables :
  let carrots_day1 := 23
  let carrots_day2 := 47
  let tomatoes_day1 := 34
  let cucumbers_day1 := 42
  let tomatoes_day2 := 50
  let cucumbers_day2 := 38
  let rotten_carrots_day1 := 10
  let rotten_carrots_day2 := 15
  let rotten_tomatoes_day1 := 5
  let rotten_cucumbers_day1 := 7
  let rotten_tomatoes_day2 := 7
  let rotten_cucumbers_day2 := 12
  let good_carrots := (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2)
  let good_tomatoes := (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2)
  let good_cucumbers := (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2)
  good_carrots + good_tomatoes + good_cucumbers = 178 := 
  sorry

end total_good_vegetables_l20_20850


namespace exist_integers_not_div_by_7_l20_20610

theorem exist_integers_not_div_by_7 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (¬ (7 ∣ x)) ∧ (¬ (7 ∣ y)) ∧ (x^2 + 6 * y^2 = 7^k) :=
sorry

end exist_integers_not_div_by_7_l20_20610


namespace problem_statement_l20_20732

open Complex

theorem problem_statement :
  (3 - I) / (2 + I) = 1 - I :=
by
  sorry

end problem_statement_l20_20732


namespace second_discarded_number_l20_20722

theorem second_discarded_number (S : ℝ) (X : ℝ) :
  (S = 50 * 44) →
  ((S - 45 - X) / 48 = 43.75) →
  X = 55 :=
by
  intros h1 h2
  -- The proof steps would go here, but we leave it unproved
  sorry

end second_discarded_number_l20_20722


namespace find_k_l20_20311

-- Given: The polynomial x^2 - 3k * x * y - 3y^2 + 6 * x * y - 8
-- We want to prove the value of k such that the polynomial does not contain the term "xy".

theorem find_k (k : ℝ) : 
  (∀ x y : ℝ, (x^2 - 3 * k * x * y - 3 * y^2 + 6 * x * y - 8) = x^2 - 3 * y^2 - 8) → 
  k = 2 := 
by
  intro h
  have h_coeff := h 1 1
  -- We should observe that the polynomial should not contain the xy term
  sorry

end find_k_l20_20311


namespace steak_weight_in_ounces_l20_20767

-- Definitions from conditions
def pounds : ℕ := 15
def ounces_per_pound : ℕ := 16
def steaks : ℕ := 20

-- The theorem to prove
theorem steak_weight_in_ounces : 
  (pounds * ounces_per_pound) / steaks = 12 := by
  sorry

end steak_weight_in_ounces_l20_20767


namespace calculation_result_l20_20154

theorem calculation_result : 
  2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := 
by 
  sorry

end calculation_result_l20_20154


namespace real_part_zero_implies_x3_l20_20918

theorem real_part_zero_implies_x3 (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ∧ (x + 1 ≠ 0) → x = 3 :=
by
  sorry

end real_part_zero_implies_x3_l20_20918


namespace find_S10_l20_20901

noncomputable def S (n : ℕ) : ℤ := 2 * (-2 ^ (n - 1)) + 1

theorem find_S10 : S 10 = -1023 :=
by
  sorry

end find_S10_l20_20901


namespace value_of_x_l20_20041

theorem value_of_x (x y : ℕ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
sorry

end value_of_x_l20_20041


namespace find_exponent_l20_20236

theorem find_exponent (x : ℕ) (h : 2^x + 2^x + 2^x + 2^x + 2^x = 2048) : x = 9 :=
sorry

end find_exponent_l20_20236


namespace farmer_spending_l20_20665

theorem farmer_spending (X : ℝ) (hc : 0.80 * X + 0.60 * X = 49) : X = 35 := 
by
  sorry

end farmer_spending_l20_20665


namespace candy_bar_calories_l20_20810

theorem candy_bar_calories
  (miles_walked : ℕ)
  (calories_per_mile : ℕ)
  (net_calorie_deficit : ℕ)
  (total_calories_burned : ℕ)
  (candy_bar_calories : ℕ)
  (h1 : miles_walked = 3)
  (h2 : calories_per_mile = 150)
  (h3 : net_calorie_deficit = 250)
  (h4 : total_calories_burned = miles_walked * calories_per_mile)
  (h5 : candy_bar_calories = total_calories_burned - net_calorie_deficit) :
  candy_bar_calories = 200 := 
by
  sorry

end candy_bar_calories_l20_20810


namespace randy_quiz_goal_l20_20302

def randy_scores : List ℕ := [90, 98, 92, 94]
def randy_next_score : ℕ := 96
def randy_goal_average : ℕ := 94

theorem randy_quiz_goal :
  let total_score := randy_scores.sum
  let required_total_score := 470
  total_score + randy_next_score = required_total_score →
  required_total_score / randy_goal_average = 5 :=
by
  intro h
  sorry

end randy_quiz_goal_l20_20302


namespace weight_of_b_l20_20914

variable {a b c : ℝ}

theorem weight_of_b (h1 : (a + b + c) / 3 = 45)
                    (h2 : (a + b) / 2 = 40)
                    (h3 : (b + c) / 2 = 43) :
                    b = 31 := by
  sorry

end weight_of_b_l20_20914


namespace two_lines_in_3d_space_l20_20687

theorem two_lines_in_3d_space : 
  ∀ x y z : ℝ, x^2 + 2 * x * (y + z) + y^2 = z^2 + 2 * z * (y + x) + x^2 → 
  (∃ a : ℝ, y = -z ∧ x = 0) ∨ (∃ b : ℝ, z = - (2 / 3) * x) :=
  sorry

end two_lines_in_3d_space_l20_20687


namespace addition_result_l20_20226

theorem addition_result : 148 + 32 + 18 + 2 = 200 :=
by
  sorry

end addition_result_l20_20226


namespace fish_remain_approximately_correct_l20_20450

noncomputable def remaining_fish : ℝ :=
  let west_initial := 1800
  let east_initial := 3200
  let north_initial := 500
  let south_initial := 2300
  let a := 3
  let b := 4
  let c := 2
  let d := 5
  let e := 1
  let f := 3
  let west_caught := (a / b) * west_initial
  let east_caught := (c / d) * east_initial
  let south_caught := (e / f) * south_initial
  let west_left := west_initial - west_caught
  let east_left := east_initial - east_caught
  let south_left := south_initial - south_caught
  let north_left := north_initial
  west_left + east_left + south_left + north_left

theorem fish_remain_approximately_correct :
  abs (remaining_fish - 4403) < 1 := 
  sorry

end fish_remain_approximately_correct_l20_20450


namespace determine_h_l20_20973

def h (x : ℝ) := -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1

theorem determine_h (x : ℝ) : 
  (12 * x^4 + 9 * x^3 - 3 * x + 1 + h x = 5 * x^3 - 8 * x^2 + 3) →
  h x = -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1 :=
by
  sorry

end determine_h_l20_20973


namespace union_dues_proof_l20_20794

noncomputable def h : ℕ := 42
noncomputable def r : ℕ := 10
noncomputable def tax_rate : ℝ := 0.20
noncomputable def insurance_rate : ℝ := 0.05
noncomputable def take_home_pay : ℝ := 310

noncomputable def gross_earnings : ℝ := h * r
noncomputable def tax_deduction : ℝ := tax_rate * gross_earnings
noncomputable def insurance_deduction : ℝ := insurance_rate * gross_earnings
noncomputable def total_deductions : ℝ := tax_deduction + insurance_deduction
noncomputable def net_earnings_before_union_dues : ℝ := gross_earnings - total_deductions
noncomputable def union_dues_deduction : ℝ := net_earnings_before_union_dues - take_home_pay

theorem union_dues_proof : union_dues_deduction = 5 := 
by sorry

end union_dues_proof_l20_20794


namespace plastering_cost_correct_l20_20943

def length : ℕ := 40
def width : ℕ := 18
def depth : ℕ := 10
def cost_per_sq_meter : ℚ := 1.25

def area_bottom (L W : ℕ) : ℕ := L * W
def perimeter_bottom (L W : ℕ) : ℕ := 2 * (L + W)
def area_walls (P D : ℕ) : ℕ := P * D
def total_area (A_bottom A_walls : ℕ) : ℕ := A_bottom + A_walls
def total_cost (A_total : ℕ) (cost_per_sq_meter : ℚ) : ℚ := A_total * cost_per_sq_meter

theorem plastering_cost_correct :
  total_cost (total_area (area_bottom length width)
                        (area_walls (perimeter_bottom length width) depth))
             cost_per_sq_meter = 2350 :=
by 
  sorry

end plastering_cost_correct_l20_20943


namespace simplification_evaluation_l20_20357

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 :=
by
  sorry

end simplification_evaluation_l20_20357


namespace soccer_camp_afternoon_kids_l20_20430

def num_kids_in_camp : ℕ := 2000
def fraction_going_to_soccer_camp : ℚ := 1 / 2
def fraction_going_to_soccer_camp_in_morning : ℚ := 1 / 4

noncomputable def num_kids_going_to_soccer_camp := num_kids_in_camp * fraction_going_to_soccer_camp
noncomputable def num_kids_going_to_soccer_camp_in_morning := num_kids_going_to_soccer_camp * fraction_going_to_soccer_camp_in_morning
noncomputable def num_kids_going_to_soccer_camp_in_afternoon := num_kids_going_to_soccer_camp - num_kids_going_to_soccer_camp_in_morning

theorem soccer_camp_afternoon_kids : num_kids_going_to_soccer_camp_in_afternoon = 750 :=
by
  sorry

end soccer_camp_afternoon_kids_l20_20430


namespace exists_small_triangle_l20_20443

-- Definitions and conditions based on the identified problem points
def square_side_length : ℝ := 1
def total_points : ℕ := 53
def vertex_points : ℕ := 4
def interior_points : ℕ := 49
def total_area : ℝ := square_side_length ^ 2
def max_triangle_area : ℝ := 0.01

-- The main theorem statement
theorem exists_small_triangle
  (sq_side : ℝ := square_side_length)
  (total_pts : ℕ := total_points)
  (vertex_pts : ℕ := vertex_points)
  (interior_pts : ℕ := interior_points)
  (total_ar : ℝ := total_area)
  (max_area : ℝ := max_triangle_area)
  (h_side : sq_side = 1)
  (h_pts : total_pts = 53)
  (h_vertex : vertex_pts = 4)
  (h_interior : interior_pts = 49)
  (h_total_area : total_ar = 1) :
  ∃ (t : ℝ), t ≤ max_area :=
sorry

end exists_small_triangle_l20_20443


namespace equation_of_trisection_line_l20_20870

/-- Let P be the point (1, 2) and let A and B be the points (2, 3) and (-3, 0), respectively. 
    One of the lines through point P and a trisection point of the line segment joining A and B has 
    the equation 3x + 7y = 17. -/
theorem equation_of_trisection_line :
  let P : ℝ × ℝ := (1, 2)
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, 0)
  -- Definition of the trisection points
  let T1 : ℝ × ℝ := ((2 + (-3 - 2) / 3) / 1, (3 + (0 - 3) / 3) / 1) -- First trisection point
  let T2 : ℝ × ℝ := ((2 + 2 * (-3 - 2) / 3) / 1, (3 + 2 * (0 - 3) / 3) / 1) -- Second trisection point
  -- Equation of the line through P and T2 is 3x + 7y = 17
  3 * (P.1 + P.2) + 7 * (P.2 + T2.2) = 17 :=
sorry

end equation_of_trisection_line_l20_20870


namespace frosting_cupcakes_l20_20826

theorem frosting_cupcakes :
  let r1 := 1 / 15
  let r2 := 1 / 25
  let r3 := 1 / 40
  let t := 600
  t * (r1 + r2 + r3) = 79 :=
by
  sorry

end frosting_cupcakes_l20_20826


namespace problem_statement_l20_20293

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, f x = 4 * x + 3) (h₁ : a > 0) (h₂ : b > 0) :
  (∀ x, |f x + 5| < a ↔ |x + 3| < b) ↔ b ≤ a / 4 :=
sorry

end problem_statement_l20_20293


namespace three_digit_perfect_squares_div_by_4_count_l20_20208

theorem three_digit_perfect_squares_div_by_4_count : 
  (∃ count : ℕ, count = 11 ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 31 → n^2 ≥ 100 ∧ n^2 ≤ 999 ∧ n^2 % 4 = 0)) :=
by
  sorry

end three_digit_perfect_squares_div_by_4_count_l20_20208


namespace true_product_of_two_digit_number_l20_20366

theorem true_product_of_two_digit_number (a b : ℕ) (h1 : b = 2 * a) (h2 : 136 * (10 * b + a) = 136 * (10 * a + b) + 1224) : 136 * (10 * a + b) = 1632 := 
by sorry

end true_product_of_two_digit_number_l20_20366


namespace number_of_connections_l20_20956

-- Definitions based on conditions
def switches : ℕ := 15
def connections_per_switch : ℕ := 4

-- Theorem statement proving the correct number of connections
theorem number_of_connections : switches * connections_per_switch / 2 = 30 := by
  sorry

end number_of_connections_l20_20956


namespace indoor_table_chairs_l20_20917

theorem indoor_table_chairs (x : ℕ) :
  (9 * x) + (11 * 3) = 123 → x = 10 :=
by
  intro h
  sorry

end indoor_table_chairs_l20_20917


namespace price_of_one_rose_l20_20525

theorem price_of_one_rose
  (tulips1 tulips2 tulips3 roses1 roses2 roses3 : ℕ)
  (price_tulip : ℕ)
  (total_earnings : ℕ)
  (R : ℕ) :
  tulips1 = 30 →
  roses1 = 20 →
  tulips2 = 2 * tulips1 →
  roses2 = 2 * roses1 →
  tulips3 = 10 * tulips2 / 100 →  -- simplification of 0.1 * tulips2
  roses3 = 16 →
  price_tulip = 2 →
  total_earnings = 420 →
  (96 * price_tulip + 76 * R) = total_earnings →
  R = 3 :=
by
  intros
  -- Proof will go here
  sorry

end price_of_one_rose_l20_20525


namespace overlapping_rectangles_perimeter_l20_20755

namespace RectangleOverlappingPerimeter

def length := 7
def width := 3

/-- Prove that the perimeter of the shape formed by overlapping two rectangles,
    each measuring 7 cm by 3 cm, is 28 cm. -/
theorem overlapping_rectangles_perimeter : 
  let total_perimeter := 2 * (length + (2 * width))
  total_perimeter = 28 :=
by
  sorry

end RectangleOverlappingPerimeter

end overlapping_rectangles_perimeter_l20_20755


namespace vector_operation_l20_20602

open Matrix

def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-6]]
def v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![-9]]
def w : Matrix (Fin 2) (Fin 1) ℝ := ![![-1], ![4]]

--\mathbf{u} - 5\mathbf{v} + \mathbf{w} = \begin{pmatrix} = \begin{pmatrix} -3 \\ 43 \end{pmatrix}
theorem vector_operation : u - (5 : ℝ) • v + w = ![![-3], ![43]] :=
by
  sorry

end vector_operation_l20_20602


namespace find_principal_l20_20659

theorem find_principal :
  ∃ P r : ℝ, (8820 = P * (1 + r) ^ 2) ∧ (9261 = P * (1 + r) ^ 3) → (P = 8000) :=
by
  sorry

end find_principal_l20_20659


namespace expenditure_of_negative_l20_20853

def income := 5000
def expenditure (x : Int) : Int := -x

theorem expenditure_of_negative (x : Int) : expenditure (-x) = x :=
by
  sorry

example : expenditure (-400) = 400 :=
by 
  exact expenditure_of_negative 400

end expenditure_of_negative_l20_20853


namespace x_coordinate_of_P_l20_20436

noncomputable section

open Real

-- Define the standard properties of the parabola and point P
def parabola (p : ℝ) (x y : ℝ) := (y ^ 2 = 4 * x)

def distance (P F : ℝ × ℝ) : ℝ := 
  sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- Position of the focus for the given parabola y^2 = 4x; Focus F(1, 0)
def focus : ℝ × ℝ := (1, 0)

-- The given conditions translated into Lean form
def on_parabola (x y : ℝ) := parabola 2 x y ∧ distance (x, y) focus = 5

-- The theorem we need to prove: If point P satisfies these conditions, then its x-coordinate is 4
theorem x_coordinate_of_P (P : ℝ × ℝ) (h : on_parabola P.1 P.2) : P.1 = 4 :=
by
  sorry

end x_coordinate_of_P_l20_20436


namespace sum_of_digits_eq_11_l20_20880

-- Define the problem conditions
variables (p q r : ℕ)
variables (h1 : 1 ≤ p ∧ p ≤ 9)
variables (h2 : 1 ≤ q ∧ q ≤ 9)
variables (h3 : 1 ≤ r ∧ r ≤ 9)
variables (h4 : p ≠ q ∧ p ≠ r ∧ q ≠ r)
variables (h5 : (10 * p + q) * (10 * p + r) = 221)

-- Define the theorem
theorem sum_of_digits_eq_11 : p + q + r = 11 :=
by
  sorry

end sum_of_digits_eq_11_l20_20880


namespace polynomial_root_condition_l20_20899

noncomputable def polynomial_q (q x : ℝ) : ℝ :=
  x^6 + 3 * q * x^4 + 3 * x^4 + 3 * q * x^2 + x^2 + 3 * q + 1

theorem polynomial_root_condition (q : ℝ) :
  (∃ x > 0, polynomial_q q x = 0) ↔ (q ≥ 3 / 2) :=
sorry

end polynomial_root_condition_l20_20899


namespace min_value_x2y3z_l20_20156

theorem min_value_x2y3z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 2 / x + 3 / y + 1 / z = 12) :
  x^2 * y^3 * z ≥ 1 / 64 :=
by
  sorry

end min_value_x2y3z_l20_20156


namespace mean_score_seniors_138_l20_20286

def total_students : ℕ := 200
def mean_score_all : ℕ := 120

variable (s n : ℕ) -- number of seniors and non-seniors
variable (ms mn : ℚ) -- mean score of seniors and non-seniors

def non_seniors_twice_seniors := n = 2 * s
def mean_score_non_seniors := mn = 0.8 * ms
def total_students_eq := s + n = total_students

def total_score := (s : ℚ) * ms + (n : ℚ) * mn = (total_students : ℚ) * mean_score_all

theorem mean_score_seniors_138 :
  ∃ s n ms mn,
    non_seniors_twice_seniors s n ∧
    mean_score_non_seniors ms mn ∧
    total_students_eq s n ∧
    total_score s n ms mn → 
    ms = 138 :=
sorry

end mean_score_seniors_138_l20_20286


namespace part_a_l20_20735

theorem part_a (n : ℕ) (h_condition : n < 135) : ∃ r, r = 239 % n ∧ r ≤ 119 := 
sorry

end part_a_l20_20735


namespace delta_value_l20_20539

-- Define the variables and the hypothesis
variable (Δ : Int)
variable (h : 5 * (-3) = Δ - 3)

-- State the theorem
theorem delta_value : Δ = -12 := by
  sorry

end delta_value_l20_20539


namespace salt_solution_percentage_l20_20182

theorem salt_solution_percentage
  (x : ℝ)
  (y : ℝ)
  (h1 : 600 + y = 1000)
  (h2 : 600 * x + y * 0.12 = 1000 * 0.084) :
  x = 0.06 :=
by
  -- The proof goes here.
  sorry

end salt_solution_percentage_l20_20182


namespace disjoint_sets_l20_20727

def P : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => 4 * x^3 + 3 * x
| (n + 1), x => (4 * x^2 + 2) * P n x - P (n - 1) x

def A (m : ℝ) : Set ℝ := {x | ∃ n : ℕ, P n m = x }

theorem disjoint_sets (m : ℝ) : Disjoint (A m) (A (m + 4)) :=
by
  -- Proof goes here
  sorry

end disjoint_sets_l20_20727


namespace temperature_representation_l20_20596

theorem temperature_representation (a : ℤ) (b : ℤ) (h1 : a = 8) (h2 : b = -5) :
    b < 0 → b = -5 :=
by
  sorry

end temperature_representation_l20_20596


namespace find_b_of_perpendicular_bisector_l20_20233

theorem find_b_of_perpendicular_bisector :
  (∃ b : ℝ, (∀ x y : ℝ, x + y = b → x + y = 4 + 6)) → b = 10 :=
by
  sorry

end find_b_of_perpendicular_bisector_l20_20233


namespace partA_partB_partC_l20_20718
noncomputable section

def n : ℕ := 100
def p : ℝ := 0.8
def q : ℝ := 1 - p

def binomial_prob (k1 k2 : ℕ) : ℝ := sorry

theorem partA : binomial_prob 70 85 = 0.8882 := sorry
theorem partB : binomial_prob 70 100 = 0.9938 := sorry
theorem partC : binomial_prob 0 69 = 0.0062 := sorry

end partA_partB_partC_l20_20718


namespace vivian_mail_june_l20_20151

theorem vivian_mail_june :
  ∀ (m_apr m_may m_jul m_aug : ℕ),
  m_apr = 5 →
  m_may = 10 →
  m_jul = 40 →
  ∃ m_jun : ℕ,
  ∃ pattern : ℕ → ℕ,
  (pattern m_apr = m_may) →
  (pattern m_may = m_jun) →
  (pattern m_jun = m_jul) →
  (pattern m_jul = m_aug) →
  (m_aug = 80) →
  pattern m_may = m_may * 2 →
  pattern m_jun = m_jun * 2 →
  pattern m_jun = 20 :=
by
  sorry

end vivian_mail_june_l20_20151


namespace intersection_A_B_l20_20104

-- Definition of set A
def A : Set ℝ := { x | x ≤ 3 }

-- Definition of set B
def B : Set ℝ := {2, 3, 4, 5}

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {2, 3} :=
sorry

end intersection_A_B_l20_20104


namespace series_sum_equals_9_over_4_l20_20010

noncomputable def series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

theorem series_sum_equals_9_over_4 :
  series_sum = 9 / 4 :=
sorry

end series_sum_equals_9_over_4_l20_20010


namespace evaluate_expression_l20_20098

variable (x : ℝ)
variable (hx : x^3 - 3 * x = 6)

theorem evaluate_expression : x^7 - 27 * x^2 = 9 * (x + 1) * (x + 6) :=
by
  sorry

end evaluate_expression_l20_20098


namespace vector_parallel_unique_solution_l20_20570

def is_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

theorem vector_parallel_unique_solution (m : ℝ) :
  let a := (m^2 - 1, m + 1)
  let b := (1, -2)
  a ≠ (0, 0) → is_parallel a b → m = 1/2 := by
  sorry

end vector_parallel_unique_solution_l20_20570


namespace line_through_point_and_isosceles_triangle_l20_20528

def is_line_eq (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def is_isosceles_right_triangle_with_axes (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0

theorem line_through_point_and_isosceles_triangle (a b c : ℝ) (hx : ℝ) (hy : ℝ) :
  is_line_eq a b c hx hy ∧ is_isosceles_right_triangle_with_axes a b → 
  ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 1 ∧ b = -1 ∧ c = -1)) :=
by
  sorry

end line_through_point_and_isosceles_triangle_l20_20528


namespace log_add_property_l20_20865

theorem log_add_property (log : ℝ → ℝ) (h1 : ∀ a b : ℝ, 0 < a → 0 < b → log a + log b = log (a * b)) (h2 : log 10 = 1) :
  log 5 + log 2 = 1 :=
by
  sorry

end log_add_property_l20_20865


namespace red_notebooks_count_l20_20406

variable (R B : ℕ)

-- Conditions
def cost_condition : Prop := 4 * R + 4 + 3 * B = 37
def count_condition : Prop := R + 2 + B = 12
def blue_notebooks_expr : Prop := B = 10 - R

-- Prove the number of red notebooks
theorem red_notebooks_count : cost_condition R B ∧ count_condition R B ∧ blue_notebooks_expr R B → R = 3 := by
  sorry

end red_notebooks_count_l20_20406


namespace race_course_length_proof_l20_20537

def race_course_length (L : ℝ) (v_A v_B : ℝ) : Prop :=
  v_A = 4 * v_B ∧ (L / v_A = (L - 66) / v_B) → L = 88

theorem race_course_length_proof (v_A v_B : ℝ) : race_course_length 88 v_A v_B :=
by 
  intros
  sorry

end race_course_length_proof_l20_20537


namespace total_votes_cast_l20_20654

theorem total_votes_cast (total_votes : ℕ) (brenda_votes : ℕ) (percentage_brenda : ℚ) 
  (h1 : brenda_votes = 40) (h2 : percentage_brenda = 0.25) 
  (h3 : brenda_votes = percentage_brenda * total_votes) : total_votes = 160 := 
by sorry

end total_votes_cast_l20_20654


namespace smallest_three_digit_multiple_of_17_l20_20753

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l20_20753


namespace box_volume_is_correct_l20_20804

noncomputable def box_volume (length width cut_side : ℝ) : ℝ :=
  (length - 2 * cut_side) * (width - 2 * cut_side) * cut_side

theorem box_volume_is_correct : box_volume 48 36 5 = 9880 := by
  sorry

end box_volume_is_correct_l20_20804


namespace count_routes_from_P_to_Q_l20_20676

variable (P Q R S T : Type)
variable (roadPQ roadPS roadPT roadQR roadQS roadRS roadST : Prop)

theorem count_routes_from_P_to_Q :
  ∃ (routes : ℕ), routes = 16 :=
by
  sorry

end count_routes_from_P_to_Q_l20_20676


namespace problem_l20_20649

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f (-x) = f x)  -- f is an even function
variable (h_mono : ∀ x y : ℝ, 0 < x → x < y → f y < f x)  -- f is monotonically decreasing on (0, +∞)

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l20_20649


namespace roses_in_vase_now_l20_20885

-- Definitions of initial conditions and variables
def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def orchids_cut : ℕ := 19
def orchids_now : ℕ := 21

-- The proof problem to show that the number of roses now is still the same as initially.
theorem roses_in_vase_now : initial_roses = 12 :=
by
  -- The proof itself is left as an exercise (add proof here)
  sorry

end roses_in_vase_now_l20_20885


namespace sum_three_circles_l20_20520

theorem sum_three_circles (a b : ℚ) 
  (h1 : 5 * a + 2 * b = 27)
  (h2 : 2 * a + 5 * b = 29) :
  3 * b = 13 :=
by
  sorry

end sum_three_circles_l20_20520


namespace largest_power_of_two_dividing_7_pow_2048_minus_1_l20_20488

theorem largest_power_of_two_dividing_7_pow_2048_minus_1 :
  ∃ n : ℕ, 2^n ∣ (7^2048 - 1) ∧ n = 14 :=
by
  use 14
  sorry

end largest_power_of_two_dividing_7_pow_2048_minus_1_l20_20488


namespace mike_washed_cars_l20_20265

theorem mike_washed_cars 
    (total_work_time : ℕ := 4 * 60) 
    (wash_time : ℕ := 10)
    (oil_change_time : ℕ := 15) 
    (tire_change_time : ℕ := 30) 
    (num_oil_changes : ℕ := 6) 
    (num_tire_changes : ℕ := 2) 
    (remaining_time : ℕ := total_work_time - (num_oil_changes * oil_change_time + num_tire_changes * tire_change_time))
    (num_cars_washed : ℕ := remaining_time / wash_time) :
    num_cars_washed = 9 := by
  sorry

end mike_washed_cars_l20_20265


namespace pipe_q_fills_cistern_in_15_minutes_l20_20035

theorem pipe_q_fills_cistern_in_15_minutes :
  ∃ T : ℝ, 
    (1/12 * 2 + 1/T * 2 + 1/T * 10.5 = 1) → 
    T = 15 :=
by {
  -- Assume the conditions and derive T = 15
  sorry
}

end pipe_q_fills_cistern_in_15_minutes_l20_20035


namespace speed_of_current_l20_20798

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l20_20798


namespace smallest_next_smallest_sum_l20_20592

-- Defining the set of numbers as constants
def nums : Set ℕ := {10, 11, 12, 13}

-- Define the smallest number in the set
def smallest : ℕ := 10

-- Define the next smallest number in the set
def next_smallest : ℕ := 11

-- The main theorem statement
theorem smallest_next_smallest_sum : smallest + next_smallest = 21 :=
by 
  sorry

end smallest_next_smallest_sum_l20_20592


namespace probability_of_6_consecutive_heads_l20_20933

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l20_20933


namespace consumption_increase_percentage_l20_20206

theorem consumption_increase_percentage
  (T C : ℝ)
  (H1 : 0.90 * (1 + X/100) = 0.9999999999999858) :
  X = 11.11111111110953 :=
by
  sorry

end consumption_increase_percentage_l20_20206


namespace consecutive_sum_150_l20_20699

theorem consecutive_sum_150 : ∃ (n : ℕ), n ≥ 2 ∧ (∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 150) :=
sorry

end consecutive_sum_150_l20_20699


namespace count_perfect_cubes_l20_20801

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1600) :
  ∃ (n : ℕ), n = 6 :=
by
  sorry

end count_perfect_cubes_l20_20801


namespace rectangular_solid_volume_l20_20805

theorem rectangular_solid_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : a * b * c = 60 :=
by
  sorry

end rectangular_solid_volume_l20_20805


namespace trigonometric_identities_l20_20670

theorem trigonometric_identities
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (sinα : Real.sin α = 4 / 5)
  (cosβ : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry

end trigonometric_identities_l20_20670


namespace arithmetic_sequence_fifth_term_l20_20857

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15) 
  (h2 : a + 11 * d = 21) :
  a + 4 * d = 0 :=
by
  sorry

end arithmetic_sequence_fifth_term_l20_20857


namespace distinct_ordered_pairs_count_l20_20838

theorem distinct_ordered_pairs_count :
  ∃ (n : ℕ), n = 29 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b → a + b = 30 → ∃! p : ℕ × ℕ, p = (a, b)) :=
sorry

end distinct_ordered_pairs_count_l20_20838


namespace michael_average_speed_l20_20253

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed_l20_20253


namespace evaluate_expression_l20_20609

theorem evaluate_expression :
  (3025^2 : ℝ) / ((305^2 : ℝ) - (295^2 : ℝ)) = 1525.10417 :=
by
  sorry

end evaluate_expression_l20_20609


namespace sufficient_but_not_necessary_condition_l20_20273

variable {α : Type*} (A B : Set α)

theorem sufficient_but_not_necessary_condition (h₁ : A ∩ B = A) (h₂ : A ≠ B) :
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l20_20273


namespace students_play_football_l20_20904

theorem students_play_football 
  (total : ℕ) (C : ℕ) (B : ℕ) (Neither : ℕ) (F : ℕ) 
  (h_total : total = 420) 
  (h_C : C = 175) 
  (h_B : B = 130) 
  (h_Neither : Neither = 50) 
  (h_inclusion_exclusion : F + C - B = total - Neither) :
  F = 325 := 
sorry

end students_play_football_l20_20904


namespace find_point_on_parabola_l20_20421

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def positive_y (y : ℝ) : Prop := y > 0
def distance_to_focus (x y : ℝ) : Prop := (x - 3/2)^2 + y^2 = (5/2)^2 

theorem find_point_on_parabola (x y : ℝ) :
  parabola x y ∧ positive_y y ∧ distance_to_focus x y → (x = 1 ∧ y = Real.sqrt 6) :=
by
  sorry

end find_point_on_parabola_l20_20421


namespace range_for_a_l20_20884

theorem range_for_a (f : ℝ → ℝ) (a : ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 8 = 1/4 →
  f (a+1) < f 2 →
  a < -3 ∨ a > 1 :=
by
  intros h1 h2 h3
  sorry

end range_for_a_l20_20884


namespace find_k_l20_20615

noncomputable def polynomial1 : Polynomial Int := sorry

theorem find_k :
  ∃ P : Polynomial Int,
  (P.eval 1 = 2013) ∧
  (P.eval 2013 = 1) ∧
  (∃ k : Int, P.eval k = k) →
  ∃ k : Int, P.eval k = k ∧ k = 1007 :=
by
  sorry

end find_k_l20_20615


namespace ratio_pat_mark_l20_20268

-- Conditions (as definitions)
variables (K P M : ℕ)
variables (h1 : P = 2 * K)  -- Pat charged twice as much time as Kate
variables (h2 : M = K + 80) -- Mark charged 80 more hours than Kate
variables (h3 : K + P + M = 144) -- Total hours charged is 144

theorem ratio_pat_mark (h1 : P = 2 * K) (h2 : M = K + 80) (h3 : K + P + M = 144) : 
  P / M = 1 / 3 :=
by
  sorry -- to be proved

end ratio_pat_mark_l20_20268


namespace empty_pencil_cases_l20_20422

theorem empty_pencil_cases (total_cases pencil_cases pen_cases both_cases : ℕ) 
  (h1 : total_cases = 10)
  (h2 : pencil_cases = 5)
  (h3 : pen_cases = 4)
  (h4 : both_cases = 2) : total_cases - (pencil_cases + pen_cases - both_cases) = 3 := by
  sorry

end empty_pencil_cases_l20_20422


namespace badger_hid_35_l20_20100

-- Define the variables
variables (h_b h_f x : ℕ)

-- Define the conditions based on the problem
def badger_hides : Prop := 5 * h_b = x
def fox_hides : Prop := 7 * h_f = x
def fewer_holes : Prop := h_b = h_f + 2

-- The main theorem to prove the badger hid 35 walnuts
theorem badger_hid_35 (h_b h_f x : ℕ) :
  badger_hides h_b x ∧ fox_hides h_f x ∧ fewer_holes h_b h_f → x = 35 :=
by sorry

end badger_hid_35_l20_20100


namespace mr_lee_harvested_apples_l20_20946

theorem mr_lee_harvested_apples :
  let number_of_baskets := 19
  let apples_per_basket := 25
  (number_of_baskets * apples_per_basket = 475) :=
by
  sorry

end mr_lee_harvested_apples_l20_20946


namespace count_of_sequence_l20_20549

theorem count_of_sequence : 
  let a := 156
  let d := -6
  let final_term := 36
  (∃ n, a + (n - 1) * d = final_term) -> n = 21 := 
by
  sorry

end count_of_sequence_l20_20549


namespace average_student_age_before_leaving_l20_20393

theorem average_student_age_before_leaving
  (A : ℕ)
  (student_count : ℕ := 30)
  (leaving_student_age : ℕ := 11)
  (teacher_age : ℕ := 41)
  (new_avg_age : ℕ := 11)
  (new_total_students : ℕ := 30)
  (initial_total_age : ℕ := 30 * A)
  (remaining_students : ℕ := 29)
  (total_age_after_leaving : ℕ := initial_total_age - leaving_student_age)
  (total_age_including_teacher : ℕ := total_age_after_leaving + teacher_age) :
  total_age_including_teacher / new_total_students = new_avg_age → A = 10 := 
  by
    intros h
    sorry

end average_student_age_before_leaving_l20_20393


namespace line_passes_through_fixed_point_l20_20073

-- Define the condition that represents the family of lines
def family_of_lines (k : ℝ) (x y : ℝ) : Prop := k * x + y + 2 * k + 1 = 0

-- Formulate the theorem stating that (-2, -1) always lies on the line
theorem line_passes_through_fixed_point (k : ℝ) : family_of_lines k (-2) (-1) :=
by
  -- Proof skipped with sorry.
  sorry

end line_passes_through_fixed_point_l20_20073


namespace avg_time_stopped_per_hour_l20_20382

-- Definitions and conditions
def avgSpeedInMotion : ℝ := 75
def overallAvgSpeed : ℝ := 40

-- Statement to prove
theorem avg_time_stopped_per_hour :
  (1 - overallAvgSpeed / avgSpeedInMotion) * 60 = 28 := 
by
  sorry

end avg_time_stopped_per_hour_l20_20382


namespace determinant_scaled_l20_20145

-- Define the initial determinant condition
def init_det (x y z w : ℝ) : Prop :=
  x * w - y * z = -3

-- Define the scaled determinant
def scaled_det (x y z w : ℝ) : ℝ :=
  3 * x * (3 * w) - 3 * y * (3 * z)

-- State the theorem we want to prove
theorem determinant_scaled (x y z w : ℝ) (h : init_det x y z w) :
  scaled_det x y z w = -27 :=
by
  sorry

end determinant_scaled_l20_20145


namespace range_of_x_l20_20635

noncomputable def A (x : ℝ) : ℤ := Int.ceil x

theorem range_of_x (x : ℝ) (h₁ : x > 0) (h₂ : A (2 * x * A x) = 5) : x ∈ Set.Ioc 1 (5 / 4 : ℝ) :=
sorry

end range_of_x_l20_20635


namespace grasshopper_jumps_rational_angle_l20_20337

noncomputable def alpha_is_rational (α : ℝ) (jump : ℕ → ℝ × ℝ) : Prop :=
  ∃ k n : ℕ, (n ≠ 0) ∧ (jump n = (0, 0)) ∧ (α = (k : ℝ) / (n : ℝ) * 360)

theorem grasshopper_jumps_rational_angle :
  ∀ (α : ℝ) (jump : ℕ → ℝ × ℝ),
    (∀ n : ℕ, dist (jump (n + 1)) (jump n) = 1) →
    (jump 0 = (0, 0)) →
    (∃ n : ℕ, n ≠ 0 ∧ jump n = (0, 0)) →
    alpha_is_rational α jump :=
by
  intros α jump jumps_eq_1 start_exists returns_to_start
  sorry

end grasshopper_jumps_rational_angle_l20_20337


namespace math_problem_l20_20307

theorem math_problem (d r : ℕ) (hd : d > 1)
  (h1 : 1259 % d = r) 
  (h2 : 1567 % d = r) 
  (h3 : 2257 % d = r) : d - r = 1 :=
by
  sorry

end math_problem_l20_20307


namespace mrs_jane_total_coins_l20_20329

theorem mrs_jane_total_coins (Jayden_coins Jason_coins : ℕ) (h1 : Jayden_coins = 300) (h2 : Jason_coins = Jayden_coins + 60) :
  Jayden_coins + Jason_coins = 660 :=
sorry

end mrs_jane_total_coins_l20_20329


namespace correct_cost_per_piece_l20_20496

-- Definitions for the given conditions
def totalPaid : ℝ := 20700
def reimbursement : ℝ := 600
def numberOfPieces : ℝ := 150
def correctTotal := totalPaid - reimbursement

-- Theorem stating the correct cost per piece of furniture
theorem correct_cost_per_piece : correctTotal / numberOfPieces = 134 := 
by
  sorry

end correct_cost_per_piece_l20_20496


namespace pizzas_in_park_l20_20855

-- Define the conditions and the proof problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100  -- in meters
def building_distance : ℕ := 2000  -- in meters
def pizzas_delivered_to_building : ℕ := 2
def total_payment_received : ℕ := 64

-- Prove the number of pizzas delivered in the park
theorem pizzas_in_park : (64 - (pizzas_delivered_to_building * pizza_cost + delivery_charge)) / pizza_cost = 3 :=
by
  sorry -- Proof not required

end pizzas_in_park_l20_20855


namespace betty_age_l20_20179

theorem betty_age {A M B : ℕ} (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 14) : B = 7 :=
sorry

end betty_age_l20_20179


namespace common_chord_length_l20_20590

theorem common_chord_length (r d : ℝ) (hr : r = 12) (hd : d = 16) : 
  ∃ l : ℝ, l = 8 * Real.sqrt 5 := 
by
  sorry

end common_chord_length_l20_20590


namespace geometric_sequence_property_l20_20362

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 3 * 2^n - 3

noncomputable def a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1

theorem geometric_sequence_property (n : ℕ) (hn : n ≥ 0) :
  T_n n < b_n (n+1) :=
by
  sorry

end geometric_sequence_property_l20_20362


namespace man_l20_20791

theorem man's_speed_with_stream
  (V_m V_s : ℝ)
  (h1 : V_m = 6)
  (h2 : V_m - V_s = 4) :
  V_m + V_s = 8 :=
sorry

end man_l20_20791


namespace express_in_scientific_notation_l20_20435

theorem express_in_scientific_notation : (250000 : ℝ) = 2.5 * 10^5 := 
by {
  -- proof
  sorry
}

end express_in_scientific_notation_l20_20435


namespace total_tickets_sold_l20_20057

-- Define the conditions
variables (V G : ℕ)

-- Condition 1: Total revenue from VIP and general admission
def total_revenue_eq : Prop := 40 * V + 15 * G = 7500

-- Condition 2: There are 212 fewer VIP tickets than general admission
def vip_tickets_eq : Prop := V = G - 212

-- Main statement to prove: the total number of tickets sold
theorem total_tickets_sold (h1 : total_revenue_eq V G) (h2 : vip_tickets_eq V G) : V + G = 370 :=
sorry

end total_tickets_sold_l20_20057


namespace tire_price_l20_20211

theorem tire_price (x : ℝ) (h1 : 2 * x + 5 = 185) : x = 90 :=
by
  sorry

end tire_price_l20_20211


namespace exists_func_satisfies_condition_l20_20321

theorem exists_func_satisfies_condition :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (x^2 + 2*x) = abs (x + 1) :=
sorry

end exists_func_satisfies_condition_l20_20321


namespace initial_red_balls_l20_20045

-- Define all the conditions as given in part (a)
variables (R : ℕ)  -- Initial number of red balls
variables (B : ℕ)  -- Number of blue balls
variables (Y : ℕ)  -- Number of yellow balls

-- The conditions
def conditions (R B Y total : ℕ) : Prop :=
  B = 2 * R ∧
  Y = 32 ∧
  total = (R - 6) + B + Y

-- The target statement proving R = 16 given the conditions
theorem initial_red_balls (R: ℕ) (B: ℕ) (Y: ℕ) (total: ℕ) 
  (h : conditions R B Y total): 
  total = 74 → R = 16 :=
by 
  sorry

end initial_red_balls_l20_20045


namespace braden_total_amount_after_winning_l20_20498

noncomputable def initial_amount := 400
noncomputable def multiplier := 2

def total_amount_after_winning (initial: ℕ) (mult: ℕ) : ℕ := initial + (mult * initial)

theorem braden_total_amount_after_winning : total_amount_after_winning initial_amount multiplier = 1200 := by
  sorry

end braden_total_amount_after_winning_l20_20498


namespace product_of_smallest_primes_l20_20856

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l20_20856


namespace aleena_vs_bob_distance_l20_20113

theorem aleena_vs_bob_distance :
  let AleenaDistance := 75
  let BobDistance := 60
  AleenaDistance - BobDistance = 15 :=
by
  let AleenaDistance := 75
  let BobDistance := 60
  show AleenaDistance - BobDistance = 15
  sorry

end aleena_vs_bob_distance_l20_20113


namespace simplify_trig_expression_l20_20194

theorem simplify_trig_expression : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := 
sorry

end simplify_trig_expression_l20_20194


namespace tire_miles_used_l20_20477

theorem tire_miles_used (total_miles : ℕ) (number_of_tires : ℕ) (tires_in_use : ℕ)
  (h_total_miles : total_miles = 40000) (h_number_of_tires : number_of_tires = 6)
  (h_tires_in_use : tires_in_use = 4) : 
  (total_miles * tires_in_use) / number_of_tires = 26667 := 
by 
  sorry

end tire_miles_used_l20_20477


namespace total_questions_attempted_l20_20053

/-- 
In an examination, a student scores 3 marks for every correct answer and loses 1 mark for
every wrong answer. He attempts some questions and secures 180 marks. The number of questions
he attempts correctly is 75. Prove that the total number of questions he attempts is 120. 
-/
theorem total_questions_attempted
  (marks_per_correct : ℕ := 3)
  (marks_lost_per_wrong : ℕ := 1)
  (total_marks : ℕ := 180)
  (correct_answers : ℕ := 75) :
  ∃ (wrong_answers total_questions : ℕ), 
    total_marks = (marks_per_correct * correct_answers) - (marks_lost_per_wrong * wrong_answers) ∧
    total_questions = correct_answers + wrong_answers ∧
    total_questions = 120 := 
by {
  sorry -- proof omitted
}

end total_questions_attempted_l20_20053


namespace trapezoid_area_condition_l20_20608

theorem trapezoid_area_condition
  (a x y z : ℝ)
  (h_sq  : ∀ (ABCD : ℝ), ABCD = a * a)
  (h_trap: ∀ (EBCF : ℝ), EBCF = x * a)
  (h_rec : ∀ (JKHG : ℝ), JKHG = y * z)
  (h_sum : y + z = a)
  (h_area : x * a = a * a - 2 * y * z) :
  x = a / 2 :=
by
  sorry

end trapezoid_area_condition_l20_20608


namespace inequality_sum_l20_20889

theorem inequality_sum
  (x y z : ℝ)
  (h : abs (x * y * z) = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 := 
sorry

end inequality_sum_l20_20889


namespace sum_of_squares_l20_20493

theorem sum_of_squares (R r r1 r2 r3 d d1 d2 d3 : ℝ) 
  (h1 : d^2 = R^2 - 2 * R * r)
  (h2 : d1^2 = R^2 + 2 * R * r1)
  (h3 : d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2) :
  d^2 + d1^2 + d2^2 + d3^2 = 12 * R^2 :=
by
  sorry

end sum_of_squares_l20_20493


namespace necessary_and_sufficient_condition_l20_20816

theorem necessary_and_sufficient_condition 
  (a b c : ℝ) :
  (a^2 = b^2 + c^2) ↔
  (∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0) := 
sorry

end necessary_and_sufficient_condition_l20_20816


namespace anne_initial_sweettarts_l20_20510

variable (x : ℕ)
variable (num_friends : ℕ := 3)
variable (sweettarts_per_friend : ℕ := 5)
variable (total_sweettarts_given : ℕ := num_friends * sweettarts_per_friend)

theorem anne_initial_sweettarts 
  (h1 : ∀ person, person < num_friends → sweettarts_per_friend = 5)
  (h2 : total_sweettarts_given = 15) : 
  total_sweettarts_given = 15 := 
by 
  sorry

end anne_initial_sweettarts_l20_20510


namespace find_p_l20_20752

theorem find_p (x y : ℝ) (h : y = 1.15 * x * (1 - p / 100)) : p = 15 :=
sorry

end find_p_l20_20752


namespace sin_beta_value_l20_20011

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < β ∧ β < π / 2)
variable (h₂ : Real.sin α = 3 / 5)
variable (h₃ : Real.cos (β - α) = 12 / 13)

theorem sin_beta_value : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_value_l20_20011


namespace exponent_is_23_l20_20238

theorem exponent_is_23 (k : ℝ) : (1/2: ℝ) ^ 23 * (1/81: ℝ) ^ k = (1/18: ℝ) ^ 23 → 23 = 23 := by
  intro h
  sorry

end exponent_is_23_l20_20238


namespace smallest_angle_measure_l20_20803

-- Define the conditions
def is_spherical_triangle (a b c : ℝ) : Prop :=
  a + b + c > 180 ∧ a + b + c < 540

def angles (k : ℝ) : Prop :=
  let a := 3 * k
  let b := 4 * k
  let c := 5 * k
  is_spherical_triangle a b c ∧ a + b + c = 270

-- Statement of the theorem
theorem smallest_angle_measure (k : ℝ) (h : angles k) : 3 * k = 67.5 :=
sorry

end smallest_angle_measure_l20_20803


namespace avg_of_9_numbers_l20_20349

theorem avg_of_9_numbers (a b c d e f g h i : ℕ)
  (h1 : (a + b + c + d + e) / 5 = 99)
  (h2 : (e + f + g + h + i) / 5 = 100)
  (h3 : e = 59) : 
  (a + b + c + d + e + f + g + h + i) / 9 = 104 := 
sorry

end avg_of_9_numbers_l20_20349


namespace sum_is_composite_l20_20036

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (h : a^2 - a * b + b^2 = c^2 - c * d + d^2) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ k * l = a + b + c + d :=
by sorry

end sum_is_composite_l20_20036


namespace find_b_l20_20714

def nabla (a b : ℤ) (h : a ≠ b) : ℤ := (a + b) / (a - b)

theorem find_b (b : ℤ) (h : 3 ≠ b) (h_eq : nabla 3 b h = -4) : b = 5 :=
sorry

end find_b_l20_20714


namespace bottles_left_l20_20721

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l20_20721


namespace cost_per_mile_l20_20282

variable (x : ℝ)
variable (monday_miles : ℝ) (thursday_miles : ℝ) (base_cost : ℝ) (total_spent : ℝ)

-- Given conditions
def car_rental_conditions : Prop :=
  monday_miles = 620 ∧
  thursday_miles = 744 ∧
  base_cost = 150 ∧
  total_spent = 832 ∧
  total_spent = base_cost + (monday_miles + thursday_miles) * x

-- Theorem to prove the cost per mile
theorem cost_per_mile (h : car_rental_conditions x 620 744 150 832) : x = 0.50 :=
  by
    sorry

end cost_per_mile_l20_20282


namespace women_decreased_by_3_l20_20341

noncomputable def initial_men := 12
noncomputable def initial_women := 27

theorem women_decreased_by_3 
  (ratio_men_women : 4 / 5 = initial_men / initial_women)
  (men_after_enter : initial_men + 2 = 14)
  (women_after_leave : initial_women - 3 = 24) :
  (24 - 27 = -3) :=
by
  sorry

end women_decreased_by_3_l20_20341


namespace g_f_g_1_equals_82_l20_20228

def f (x : ℤ) : ℤ := 2 * x + 2
def g (x : ℤ) : ℤ := 5 * x + 2
def x : ℤ := 1

theorem g_f_g_1_equals_82 : g (f (g x)) = 82 := by
  sorry

end g_f_g_1_equals_82_l20_20228


namespace mary_money_after_purchase_l20_20428

def mary_initial_money : ℕ := 58
def pie_cost : ℕ := 6
def mary_friend_money : ℕ := 43  -- This is an extraneous condition, included for completeness.

theorem mary_money_after_purchase : mary_initial_money - pie_cost = 52 := by
  sorry

end mary_money_after_purchase_l20_20428


namespace polar_to_cartesian_l20_20711

theorem polar_to_cartesian (p θ : ℝ) (x y : ℝ) (hp : p = 8 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 + y^2 = 8 * x := 
sorry

end polar_to_cartesian_l20_20711


namespace student_losses_one_mark_l20_20446

def number_of_marks_lost_per_wrong_answer (correct_ans marks_attempted total_questions total_marks correct_questions : ℤ) : ℤ :=
  (correct_ans * correct_questions - total_marks) / (total_questions - correct_questions)

theorem student_losses_one_mark
  (correct_ans : ℤ)
  (marks_attempted : ℤ)
  (total_questions : ℤ)
  (total_marks : ℤ)
  (correct_questions : ℤ)
  (total_wrong : ℤ):
  correct_ans = 4 →
  total_questions = 80 →
  total_marks = 120 →
  correct_questions = 40 →
  total_wrong = total_questions - correct_questions →
  number_of_marks_lost_per_wrong_answer correct_ans marks_attempted total_questions total_marks correct_questions = 1 :=
by
  sorry

end student_losses_one_mark_l20_20446


namespace system_sum_of_squares_l20_20784

theorem system_sum_of_squares :
  (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    9*y1^2 - 4*x1^2 = 144 - 48*x1 ∧ 9*y1^2 + 4*x1^2 = 144 + 18*x1*y1 ∧
    9*y2^2 - 4*x2^2 = 144 - 48*x2 ∧ 9*y2^2 + 4*x2^2 = 144 + 18*x2*y2 ∧
    9*y3^2 - 4*x3^2 = 144 - 48*x3 ∧ 9*y3^2 + 4*x3^2 = 144 + 18*x3*y3 ∧
    (x1^2 + x2^2 + x3^2 + y1^2 + y2^2 + y3^2 = 68)) :=
by sorry

end system_sum_of_squares_l20_20784


namespace billy_sleep_total_hours_l20_20129

theorem billy_sleep_total_hours : 
    let first_night := 6
    let second_night := 2 * first_night
    let third_night := second_night - 3
    let fourth_night := 3 * third_night
    first_night + second_night + third_night + fourth_night = 54
  := by
    sorry

end billy_sleep_total_hours_l20_20129


namespace train_speed_l20_20638

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l20_20638


namespace sum_of_ages_l20_20163

/--
Given:
- Beckett's age is 12.
- Olaf is 3 years older than Beckett.
- Shannen is 2 years younger than Olaf.
- Jack is 5 more than twice as old as Shannen.

Prove that the sum of the ages of Beckett, Olaf, Shannen, and Jack is 71 years.
-/
theorem sum_of_ages :
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  Beckett + Olaf + Shannen + Jack = 71 :=
by
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  show Beckett + Olaf + Shannen + Jack = 71
  sorry

end sum_of_ages_l20_20163


namespace divisor_of_p_l20_20815

-- Define the necessary variables and assumptions
variables (p q r s : ℕ)

-- State the conditions
def conditions := gcd p q = 28 ∧ gcd q r = 45 ∧ gcd r s = 63 ∧ 80 < gcd s p ∧ gcd s p < 120 

-- State the proposition to prove: 11 divides p
theorem divisor_of_p (h : conditions p q r s) : 11 ∣ p := 
sorry

end divisor_of_p_l20_20815


namespace speed_of_stream_l20_20229

theorem speed_of_stream 
  (b s : ℝ) 
  (h1 : 78 = (b + s) * 2) 
  (h2 : 50 = (b - s) * 2) 
  : s = 7 := 
sorry

end speed_of_stream_l20_20229


namespace part1_part2_l20_20710

variables (x y z : ℝ)

-- Conditions
def conditions := (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 1)

-- Part 1: Prove 2(x^2 + y^2 + z^2) + 9xyz >= 1
theorem part1 (h : conditions x y z) : 2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 :=
sorry

-- Part 2: Prove xy + yz + zx - 3xyz ≤ 1/4
theorem part2 (h : conditions x y z) : x * y + y * z + z * x - 3 * x * y * z ≤ 1 / 4 :=
sorry

end part1_part2_l20_20710


namespace intersection_of_M_and_N_l20_20658

def M : Set ℕ := {0, 2, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N :
  {x | x ∈ M ∧ x ∈ N} = {0, 4} := by
  sorry

end intersection_of_M_and_N_l20_20658


namespace nate_age_when_ember_is_14_l20_20664

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end nate_age_when_ember_is_14_l20_20664


namespace lightning_distance_l20_20405

/--
Linus observed a flash of lightning and then heard the thunder 15 seconds later.
Given:
- speed of sound: 1088 feet/second
- 1 mile = 5280 feet
Prove that the distance from Linus to the lightning strike is 3.25 miles.
-/
theorem lightning_distance (time_seconds : ℕ) (speed_sound : ℕ) (feet_per_mile : ℕ) (distance_miles : ℚ) :
  time_seconds = 15 →
  speed_sound = 1088 →
  feet_per_mile = 5280 →
  distance_miles = 3.25 :=
by
  sorry

end lightning_distance_l20_20405


namespace problem_1_problem_2_problem_3_l20_20218

-- Definitions of assumptions and conditions.
structure Problem :=
  (boys : ℕ) -- number of boys
  (girls : ℕ) -- number of girls
  (subjects : ℕ) -- number of subjects
  (boyA_not_math : Prop) -- Boy A can't be a representative of the mathematics course
  (girlB_chinese : Prop) -- Girl B must be a representative of the Chinese language course

-- Problem 1: Calculate the number of ways satisfying condition (1)
theorem problem_1 (p : Problem) (h1 : p.girls < p.boys) :
  ∃ n : ℕ, n = 5520 := sorry

-- Problem 2: Calculate the number of ways satisfying condition (2)
theorem problem_2 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) :
  ∃ n : ℕ, n = 3360 := sorry

-- Problem 3: Calculate the number of ways satisfying condition (3)
theorem problem_3 (p : Problem) (h1 : p.boys ≥ 1) (h2 : p.boyA_not_math) (h3 : p.girlB_chinese) :
  ∃ n : ℕ, n = 360 := sorry

end problem_1_problem_2_problem_3_l20_20218


namespace total_toothpicks_needed_l20_20219

/-- The number of toothpicks needed to construct both a large and smaller equilateral triangle 
    side by side, given the large triangle has a base of 100 small triangles and the smaller triangle 
    has a base of 50 small triangles -/
theorem total_toothpicks_needed 
  (base_large : ℕ) (base_small : ℕ) (shared_boundary : ℕ) 
  (h1 : base_large = 100) (h2 : base_small = 50) (h3 : shared_boundary = base_small) :
  3 * (100 * 101 / 2) / 2 + 3 * (50 * 51 / 2) / 2 - shared_boundary = 9462 := 
sorry

end total_toothpicks_needed_l20_20219


namespace worker_assignment_l20_20642

theorem worker_assignment :
  ∃ (x y : ℕ), x + y = 85 ∧
  (16 * x) / 2 = (10 * y) / 3 ∧
  x = 25 ∧ y = 60 :=
by
  sorry

end worker_assignment_l20_20642


namespace john_height_in_feet_l20_20249

theorem john_height_in_feet (initial_height : ℕ) (growth_rate : ℕ) (months : ℕ) (inches_per_foot : ℕ) :
  initial_height = 66 → growth_rate = 2 → months = 3 → inches_per_foot = 12 → 
  (initial_height + growth_rate * months) / inches_per_foot = 6 := by
  intros h1 h2 h3 h4
  sorry

end john_height_in_feet_l20_20249


namespace Buffy_whiskers_l20_20283

def whiskers_Juniper : ℕ := 12
def whiskers_Puffy : ℕ := 3 * whiskers_Juniper
def whiskers_Scruffy : ℕ := 2 * whiskers_Puffy
def whiskers_Buffy : ℕ := (whiskers_Puffy + whiskers_Scruffy + whiskers_Juniper) / 3

theorem Buffy_whiskers : whiskers_Buffy = 40 := by
  sorry

end Buffy_whiskers_l20_20283


namespace visibility_beach_to_hill_visibility_ferry_to_tree_l20_20223

noncomputable def altitude_lake : ℝ := 104
noncomputable def altitude_hill_tree : ℝ := 154
noncomputable def map_distance_1 : ℝ := 70 / 100 -- Convert cm to meters
noncomputable def map_distance_2 : ℝ := 38.5 / 100 -- Convert cm to meters
noncomputable def map_scale : ℝ := 95000
noncomputable def earth_circumference : ℝ := 40000000 -- Convert km to meters

noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

noncomputable def visible_distance (height : ℝ) : ℝ :=
  Real.sqrt (2 * earth_radius * height)

noncomputable def actual_distance_1 : ℝ := map_distance_1 * map_scale
noncomputable def actual_distance_2 : ℝ := map_distance_2 * map_scale

theorem visibility_beach_to_hill :
  actual_distance_1 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

theorem visibility_ferry_to_tree :
  actual_distance_2 > visible_distance (altitude_hill_tree - altitude_lake) :=
by
  sorry

end visibility_beach_to_hill_visibility_ferry_to_tree_l20_20223


namespace no_playful_two_digit_numbers_l20_20215

def is_playful (a b : ℕ) : Prop := 10 * a + b = a^3 + b^2

theorem no_playful_two_digit_numbers :
  (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → ¬ is_playful a b) :=
by {
  sorry
}

end no_playful_two_digit_numbers_l20_20215


namespace solve_fractional_equation_l20_20276

theorem solve_fractional_equation {x : ℝ} (h1 : x ≠ -1) (h2 : x ≠ 0) :
  6 / (x + 1) = (x + 5) / (x * (x + 1)) ↔ x = 1 :=
by
  -- This proof is left as an exercise.
  sorry

end solve_fractional_equation_l20_20276


namespace baker_sold_cakes_l20_20929

theorem baker_sold_cakes (S : ℕ) (h1 : 154 = S + 63) : S = 91 :=
by
  sorry

end baker_sold_cakes_l20_20929


namespace multiplication_equivalence_l20_20916

theorem multiplication_equivalence :
    44 * 22 = 88 * 11 :=
by
  sorry

end multiplication_equivalence_l20_20916


namespace average_of_ratios_l20_20250

theorem average_of_ratios (a b c : ℕ) (h1 : 2 * b = 3 * a) (h2 : 3 * c = 4 * a) (h3 : a = 28) : (a + b + c) / 3 = 42 := by
  -- skipping the proof
  sorry

end average_of_ratios_l20_20250


namespace gift_combinations_l20_20031

theorem gift_combinations (wrapping_paper_count ribbon_count card_count : ℕ)
  (restricted_wrapping : ℕ)
  (restricted_ribbon : ℕ)
  (total_combinations := wrapping_paper_count * ribbon_count * card_count)
  (invalid_combinations := card_count)
  (valid_combinations := total_combinations - invalid_combinations) :
  wrapping_paper_count = 10 →
  ribbon_count = 4 →
  card_count = 5 →
  restricted_wrapping = 10 →
  restricted_ribbon = 1 →
  valid_combinations = 195 :=
by
  intros
  sorry

end gift_combinations_l20_20031


namespace total_students_at_competition_l20_20932
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end total_students_at_competition_l20_20932


namespace comb_sum_C8_2_C8_3_l20_20239

open Nat

theorem comb_sum_C8_2_C8_3 : (Nat.choose 8 2) + (Nat.choose 8 3) = 84 :=
by
  sorry

end comb_sum_C8_2_C8_3_l20_20239


namespace problem1_problem2_l20_20697

-- Problem (1) Lean Statement
theorem problem1 (c a b : ℝ) (hc : c > a) (ha : a > b) (hb : b > 0) : 
  a / (c - a) > b / (c - b) :=
sorry

-- Problem (2) Lean Statement
theorem problem2 (x : ℝ) (hx : x > 2) : 
  ∃ (xmin : ℝ), xmin = 6 ∧ (x = 6 → (x + 16 / (x - 2)) = 10) :=
sorry

end problem1_problem2_l20_20697


namespace range_of_a_l20_20643

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → (x < a ∨ x > a + 4)) ∧ ¬(∀ x : ℝ, (x < a ∨ x > a + 4) → -2 ≤ x ∧ x ≤ 1) ↔
  a > 1 ∨ a < -6 :=
by {
  sorry
}

end range_of_a_l20_20643


namespace solve_inequality_l20_20440

theorem solve_inequality (a x : ℝ) : 
  (ax^2 + (a - 1) * x - 1 < 0) ↔ (
  (a = 0 ∧ x > -1) ∨ 
  (a > 0 ∧ -1 < x ∧ x < 1/a) ∨
  (-1 < a ∧ a < 0 ∧ (x < 1/a ∨ x > -1)) ∨ 
  (a = -1 ∧ x ≠ -1) ∨ 
  (a < -1 ∧ (x < -1 ∨ x > 1/a))
) := sorry

end solve_inequality_l20_20440


namespace people_per_car_l20_20619

theorem people_per_car (total_people : ℕ) (total_cars : ℕ) (h_people : total_people = 63) (h_cars : total_cars = 3) : 
  total_people / total_cars = 21 := by
  sorry

end people_per_car_l20_20619


namespace scooter_value_depreciation_l20_20523

theorem scooter_value_depreciation (V0 Vn : ℝ) (rate : ℝ) (n : ℕ) 
  (hV0 : V0 = 40000) 
  (hVn : Vn = 9492.1875) 
  (hRate : rate = 3 / 4) 
  (hValue : Vn = V0 * rate ^ n) : 
  n = 5 := 
by 
  -- Conditions are set up, proof needs to be constructed.
  sorry

end scooter_value_depreciation_l20_20523


namespace total_eggs_examined_l20_20457

def trays := 7
def eggs_per_tray := 10

theorem total_eggs_examined : trays * eggs_per_tray = 70 :=
by 
  sorry

end total_eggs_examined_l20_20457


namespace profit_calculation_l20_20346

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l20_20346


namespace max_collisions_l20_20356

-- Define the problem
theorem max_collisions (n : ℕ) (hn : n > 0) : 
  ∃ C : ℕ, C = (n * (n - 1)) / 2 := 
sorry

end max_collisions_l20_20356


namespace price_per_glass_second_day_l20_20221

theorem price_per_glass_second_day (O : ℝ) (P : ℝ) 
  (V1 : ℝ := 2 * O) -- Volume on the first day
  (V2 : ℝ := 3 * O) -- Volume on the second day
  (price_first_day : ℝ := 0.30) -- Price per glass on the first day
  (revenue_equal : V1 * price_first_day = V2 * P) :
  P = 0.20 := 
by
  -- skipping the proof
  sorry

end price_per_glass_second_day_l20_20221


namespace union_of_sets_l20_20605

def A : Set ℝ := {x | x < -1 ∨ x > 3}
def B : Set ℝ := {x | x ≥ 2}

theorem union_of_sets : A ∪ B = {x | x < -1 ∨ x ≥ 2} :=
by
  sorry

end union_of_sets_l20_20605


namespace solve_for_q_l20_20071

theorem solve_for_q (x y q : ℚ) 
  (h1 : 7 / 8 = x / 96) 
  (h2 : 7 / 8 = (x + y) / 104) 
  (h3 : 7 / 8 = (q - y) / 144) : 
  q = 133 := 
sorry

end solve_for_q_l20_20071


namespace dima_always_wins_l20_20814

theorem dima_always_wins (n : ℕ) (P : Prop) : 
  (∀ (gosha dima : ℕ → Prop), 
    (∀ k : ℕ, k < n → (gosha k ∨ dima k))
    ∧ (∀ i : ℕ, i < 14 → (gosha i ∨ dima i))
    ∧ (∃ j : ℕ, j ≤ n ∧ (∃ k ≤ j + 7, dima k))
    ∧ (∃ l : ℕ, l ≤ 14 ∧ (∃ m ≤ l + 7, dima m))
    → P) → P := sorry

end dima_always_wins_l20_20814


namespace investment_of_c_l20_20383

theorem investment_of_c (P_b P_a P_c C_a C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_a - P_c = 799.9999999999998)
  (h3 : C_a = 8000)
  (h4 : C_b = 10000)
  (h5 : P_b / C_b = P_a / C_a)
  (h6 : P_c / C_c = P_a / C_a)
  : C_c = 4000 :=
by 
  sorry

end investment_of_c_l20_20383


namespace reduce_repeating_decimal_l20_20463

noncomputable def repeating_decimal_to_fraction (a : ℚ) (n : ℕ) : ℚ :=
  a + (n / 99)

theorem reduce_repeating_decimal : repeating_decimal_to_fraction 2 7 = 205 / 99 := by
  -- proof omitted
  sorry

end reduce_repeating_decimal_l20_20463


namespace expression_C_eq_seventeen_l20_20848

theorem expression_C_eq_seventeen : (3 + 4 * 5 - 6) = 17 := 
by 
  sorry

end expression_C_eq_seventeen_l20_20848


namespace third_row_number_l20_20562

-- Define the conditions to fill the grid
def grid (n : Nat) := Fin 4 → Fin 4 → Fin n

-- Ensure each number 1-4 in each cell such that numbers do not repeat
def unique_in_row (g : grid 4) : Prop :=
  ∀ i j1 j2, j1 ≠ j2 → g i j1 ≠ g i j2

def unique_in_col (g : grid 4) : Prop :=
  ∀ j i1 i2, i1 ≠ i2 → g i1 j ≠ g i1 j

-- Define the external hints condition, encapsulating the provided hints.
def hints_condition (g : grid 4) : Prop :=
  -- Example placeholders for hint conditions that would be expanded accordingly.
  g 0 0 = 3 ∨ g 0 1 = 2 -- First row hints interpreted constraints
  -- Additional hint conditions to be added accordingly

-- Prove the correct number formed by the numbers in the third row is 4213
theorem third_row_number (g : grid 4) :
  unique_in_row g ∧ unique_in_col g ∧ hints_condition g →
  (g 2 0 = 4 ∧ g 2 1 = 2 ∧ g 2 2 = 1 ∧ g 2 3 = 3) :=
by
  sorry

end third_row_number_l20_20562


namespace largest_visits_is_four_l20_20103

noncomputable def largest_num_visits (stores people visits : ℕ) (eight_people_two_stores : ℕ) 
  (one_person_min : ℕ) : ℕ := 4 -- This represents the largest number of stores anyone could have visited.

theorem largest_visits_is_four 
  (stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ) 
  (eight_people_two_stores : ℕ) (each_one_store : ℕ) 
  (H1 : stores = 8) 
  (H2 : total_visits = 23) 
  (H3 : people_shopping = 12) 
  (H4 : eight_people_two_stores = 8)
  (H5 : each_one_store = 1) :
  largest_num_visits stores people_shopping total_visits eight_people_two_stores each_one_store = 4 :=
by
  sorry

end largest_visits_is_four_l20_20103


namespace functional_ineq_l20_20781

noncomputable def f : ℝ → ℝ := sorry

theorem functional_ineq (h1 : ∀ x > 1400^2021, x * f x ≤ 2021) (h2 : ∀ x : ℝ, 0 < x → f x = f (x + 2) + 2 * f (x * (x + 2))) : 
  ∀ x : ℝ, 0 < x → x * f x ≤ 2021 :=
sorry

end functional_ineq_l20_20781


namespace solve_equation_l20_20898

theorem solve_equation (x : ℝ) (h : 2 * x + 6 = 2 + 3 * x) : x = 4 :=
by
  sorry

end solve_equation_l20_20898


namespace trains_pass_time_l20_20412

def length_train1 : ℕ := 200
def length_train2 : ℕ := 280

def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * 1000 / 3600

def relative_speed_mps : ℚ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_length : ℕ :=
  length_train1 + length_train2

def time_to_pass_trains : ℚ :=
  total_length / relative_speed_mps

theorem trains_pass_time :
  time_to_pass_trains = 24 := by
  sorry

end trains_pass_time_l20_20412


namespace sales_tax_difference_l20_20425

theorem sales_tax_difference
  (item_price : ℝ)
  (rate1 rate2 : ℝ)
  (h_rate1 : rate1 = 0.0725)
  (h_rate2 : rate2 = 0.0675)
  (h_item_price : item_price = 40) :
  item_price * rate1 - item_price * rate2 = 0.20 :=
by
  -- Since we are required to skip the proof, we put sorry here.
  sorry

end sales_tax_difference_l20_20425


namespace max_distance_bicycle_l20_20064

theorem max_distance_bicycle (front_tire_last : ℕ) (rear_tire_last : ℕ) :
  front_tire_last = 5000 ∧ rear_tire_last = 3000 →
  ∃ (max_distance : ℕ), max_distance = 3750 :=
by
  sorry

end max_distance_bicycle_l20_20064


namespace intersection_point_on_y_axis_l20_20368

theorem intersection_point_on_y_axis (k : ℝ) :
  ∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0 ↔ k = 6 ∨ k = -6 :=
by
  sorry

end intersection_point_on_y_axis_l20_20368


namespace first_even_number_of_8_sum_424_l20_20132

theorem first_even_number_of_8_sum_424 (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + 
                   (x + 8) + (x + 10) + (x + 12) + (x + 14) = 424) : x = 46 :=
by sorry

end first_even_number_of_8_sum_424_l20_20132


namespace male_listeners_l20_20490

structure Survey :=
  (males_dont_listen : Nat)
  (females_listen : Nat)
  (total_listeners : Nat)
  (total_dont_listen : Nat)

def number_of_females_dont_listen (s : Survey) : Nat :=
  s.total_dont_listen - s.males_dont_listen

def number_of_males_listen (s : Survey) : Nat :=
  s.total_listeners - s.females_listen

theorem male_listeners (s : Survey) (h : s = { males_dont_listen := 85, females_listen := 75, total_listeners := 180, total_dont_listen := 160 }) :
  number_of_males_listen s = 105 :=
by
  sorry

end male_listeners_l20_20490


namespace functional_expression_selling_price_for_profit_l20_20312

-- Define the initial conditions
def cost_price : ℚ := 8
def initial_selling_price : ℚ := 10
def initial_sales_volume : ℚ := 200
def sales_decrement_per_yuan_increase : ℚ := 20

-- Functional expression between y (items) and x (yuan)
theorem functional_expression (x : ℚ) : 
  (200 - 20 * (x - 10) = -20 * x + 400) :=
sorry

-- Determine the selling price to achieve a daily profit of 640 yuan
theorem selling_price_for_profit (x : ℚ) (h1 : 8 ≤ x) (h2 : x ≤ 15) : 
  ((x - 8) * (400 - 20 * x) = 640) → (x = 12) :=
sorry

end functional_expression_selling_price_for_profit_l20_20312


namespace combined_average_pieces_lost_l20_20263

theorem combined_average_pieces_lost
  (audrey_losses : List ℕ) (thomas_losses : List ℕ)
  (h_audrey : audrey_losses = [6, 8, 4, 7, 10])
  (h_thomas : thomas_losses = [5, 6, 3, 7, 11]) :
  (audrey_losses.sum + thomas_losses.sum : ℚ) / 5 = 13.4 := by 
  sorry

end combined_average_pieces_lost_l20_20263


namespace first_player_wins_l20_20625

-- Define the set of points S
def S : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) ∧ x^2 + y^2 ≤ 1010 }

-- Define the game properties and conditions
def game_property :=
  ∀ (p : ℤ × ℤ), p ∈ S →
  ∀ (q : ℤ × ℤ), q ∈ S →
  p ≠ q →
  -- Forbidden to move to a point symmetric to the current one relative to the origin
  q ≠ (-p.fst, -p.snd) →
  -- Distances of moves must strictly increase
  dist p q > dist q (q.fst, q.snd)

-- The first player always guarantees a win
theorem first_player_wins : game_property → true :=
by
  sorry

end first_player_wins_l20_20625


namespace lower_right_is_one_l20_20677

def initial_grid : Matrix (Fin 5) (Fin 5) (Option (Fin 5)) :=
![![some 0, none, some 1, none, none],
  ![some 1, some 3, none, none, none],
  ![none, none, none, some 4, none],
  ![none, some 4, none, none, none],
  ![none, none, none, none, none]]

theorem lower_right_is_one 
  (complete_grid : Matrix (Fin 5) (Fin 5) (Fin 5)) 
  (unique_row_col : ∀ i j k, 
      complete_grid i j = complete_grid i k ↔ j = k ∧ 
      complete_grid i j = complete_grid k j ↔ i = k)
  (matches_partial : ∀ i j, ∃ x, 
      initial_grid i j = some x → complete_grid i j = x) :
  complete_grid 4 4 = 0 := 
sorry

end lower_right_is_one_l20_20677


namespace quadractic_b_value_l20_20883

def quadratic_coefficients (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem quadractic_b_value :
  ∀ (a b c : ℝ), quadratic_coefficients 1 (-2) (-3) (x : ℝ) → 
  b = -2 := by
  sorry

end quadractic_b_value_l20_20883


namespace train_length_is_correct_l20_20873

noncomputable def convert_speed (speed_kmh : ℕ) : ℝ :=
  (speed_kmh : ℝ) * 5 / 18

noncomputable def relative_speed (train_speed_kmh man's_speed_kmh : ℕ) : ℝ :=
  convert_speed train_speed_kmh + convert_speed man's_speed_kmh

noncomputable def length_of_train (train_speed_kmh man's_speed_kmh : ℕ) (time_seconds : ℝ) : ℝ := 
  relative_speed train_speed_kmh man's_speed_kmh * time_seconds

theorem train_length_is_correct :
  length_of_train 60 6 29.997600191984645 = 550 :=
by
  sorry

end train_length_is_correct_l20_20873


namespace simplify_expr_l20_20308

def A (a b : ℝ) := b^2 - a^2 + 5 * a * b
def B (a b : ℝ) := 3 * a * b + 2 * b^2 - a^2

theorem simplify_expr (a b : ℝ) : 2 * (A a b) - (B a b) = -a^2 + 7 * a * b := by
  -- actual proof omitted
  sorry

example : (2 * (A 1 2) - (B 1 2)) = 13 := by
  -- actual proof omitted
  sorry

end simplify_expr_l20_20308


namespace total_people_at_zoo_l20_20237

theorem total_people_at_zoo (A K : ℕ) (ticket_price_adult : ℕ := 28) (ticket_price_kid : ℕ := 12) (total_sales : ℕ := 3864) (number_of_kids : ℕ := 203) :
  (ticket_price_adult * A + ticket_price_kid * number_of_kids = total_sales) → 
  (A + number_of_kids = 254) :=
by
  sorry

end total_people_at_zoo_l20_20237


namespace subtraction_of_negatives_l20_20903

theorem subtraction_of_negatives : (-7) - (-5) = -2 := 
by {
  -- sorry replaces the actual proof steps.
  sorry
}

end subtraction_of_negatives_l20_20903


namespace inequality_solution_l20_20953

theorem inequality_solution (x : ℝ) : 
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 :=
by
  sorry

end inequality_solution_l20_20953


namespace compute_expression_l20_20244

theorem compute_expression (p q r : ℝ) 
  (h1 : p + q + r = 6) 
  (h2 : pq + qr + rp = 11) 
  (h3 : pqr = 12) : 
  (pq / r) + (qr / p) + (rp / q) = -23 / 12 := 
sorry

end compute_expression_l20_20244


namespace empty_set_l20_20360

def setA := {x : ℝ | x^2 - 4 = 0}
def setB := {x : ℝ | x > 9 ∨ x < 3}
def setC := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}
def setD := {x : ℝ | x > 9 ∧ x < 3}

theorem empty_set : setD = ∅ := 
  sorry

end empty_set_l20_20360


namespace dice_probability_sum_18_l20_20454

theorem dice_probability_sum_18 : 
  (∃ d1 d2 d3 : ℕ, 1 ≤ d1 ∧ d1 ≤ 8 ∧ 1 ≤ d2 ∧ d2 ≤ 8 ∧ 1 ≤ d3 ∧ d3 ≤ 8 ∧ d1 + d2 + d3 = 18) →
  (1/8 : ℚ) * (1/8) * (1/8) * 9 = 9 / 512 :=
by 
  sorry

end dice_probability_sum_18_l20_20454


namespace circle_equation_tangent_x_axis_l20_20641

theorem circle_equation_tangent_x_axis (x y : ℝ) (center : ℝ × ℝ) (r : ℝ) 
  (h_center : center = (-1, 2)) 
  (h_tangent : r = |2 - 0|) :
  (x + 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_tangent_x_axis_l20_20641


namespace approx_log_base_5_10_l20_20016

noncomputable def log_base (b a : ℝ) : ℝ := (Real.log a) / (Real.log b)

theorem approx_log_base_5_10 :
  let lg2 := 0.301
  let lg3 := 0.477
  let lg10 := 1
  let lg5 := lg10 - lg2
  log_base 5 10 = 10 / 7 :=
  sorry

end approx_log_base_5_10_l20_20016


namespace two_point_questions_l20_20733

theorem two_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
by
  sorry

end two_point_questions_l20_20733


namespace minimum_value_l20_20338

variable (m n x y : ℝ)

theorem minimum_value (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (my_nx : ℝ), my_nx = my + nx → my_nx ≥ min_val :=
by
  sorry

end minimum_value_l20_20338


namespace tangent_line_at_point_l20_20531

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x - f 0 + 2 = 0

theorem tangent_line_at_point (f : ℝ → ℝ)
  (h_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  (h_eq : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  tangent_line_equation f 0 :=
by
  sorry

end tangent_line_at_point_l20_20531


namespace max_red_socks_l20_20631

theorem max_red_socks (x y : ℕ) 
  (h1 : x + y ≤ 2017) 
  (h2 : (x * (x - 1) + y * (y - 1)) = (x + y) * (x + y - 1) / 2) : 
  x ≤ 990 := 
sorry

end max_red_socks_l20_20631


namespace cherries_cost_l20_20109

def cost_per_kg (total_cost kilograms : ℕ) : ℕ :=
  total_cost / kilograms

theorem cherries_cost 
  (genevieve_amount : ℕ) 
  (short_amount : ℕ)
  (total_kilograms : ℕ) 
  (total_cost : ℕ := genevieve_amount + short_amount) 
  (cost : ℕ := cost_per_kg total_cost total_kilograms) : 
  cost = 8 :=
by
  have h1 : genevieve_amount = 1600 := by sorry
  have h2 : short_amount = 400 := by sorry
  have h3 : total_kilograms = 250 := by sorry
  sorry

end cherries_cost_l20_20109


namespace percentage_of_motorists_speeding_l20_20573

-- Definitions based on the conditions
def total_motorists : Nat := 100
def percent_motorists_receive_tickets : Real := 0.20
def percent_speeders_no_tickets : Real := 0.20

-- Define the variables for the number of speeders
variable (x : Real) -- the percentage of total motorists who speed 

-- Lean statement to formalize the problem
theorem percentage_of_motorists_speeding 
  (h1 : 20 = (0.80 * x) * (total_motorists / 100)) : 
  x = 25 :=
sorry

end percentage_of_motorists_speeding_l20_20573


namespace hair_cut_second_day_l20_20630

variable (hair_first_day : ℝ) (total_hair_cut : ℝ)

theorem hair_cut_second_day (h1 : hair_first_day = 0.375) (h2 : total_hair_cut = 0.875) :
  total_hair_cut - hair_first_day = 0.500 :=
by sorry

end hair_cut_second_day_l20_20630


namespace unique_solution_l20_20582

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (f x)) + f (f y) = f y + x

-- Define the main theorem
theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y, functional_eq f x y) → (∀ x, f x = x) :=
by
  intros h x
  -- Proof steps would go here
  sorry

end unique_solution_l20_20582


namespace sum_of_n_values_l20_20047

theorem sum_of_n_values : ∃ n1 n2 : ℚ, (abs (3 * n1 - 4) = 5) ∧ (abs (3 * n2 - 4) = 5) ∧ n1 + n2 = 8 / 3 :=
by
  sorry

end sum_of_n_values_l20_20047


namespace laser_total_distance_l20_20116

noncomputable def laser_path_distance : ℝ :=
  let A := (2, 4)
  let B := (2, -4)
  let C := (-2, -4)
  let D := (8, 4)
  let distance (p q : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B + distance B C + distance C D

theorem laser_total_distance :
  laser_path_distance = 12 + 2 * Real.sqrt 41 :=
by sorry

end laser_total_distance_l20_20116


namespace sum_reciprocals_factors_12_l20_20793

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l20_20793


namespace isosceles_base_angle_l20_20426

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end isosceles_base_angle_l20_20426


namespace dodecahedron_interior_diagonals_count_l20_20729

-- Define a dodecahedron structure
structure Dodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (faces_per_vertex : ℕ)

-- Define the property of a dodecahedron
def dodecahedron_property : Dodecahedron :=
{
  vertices := 20,
  edges_per_vertex := 3,
  faces_per_vertex := 3
}

-- The theorem statement
theorem dodecahedron_interior_diagonals_count (d : Dodecahedron)
  (h1 : d.vertices = 20)
  (h2 : d.edges_per_vertex = 3)
  (h3 : d.faces_per_vertex = 3) : 
  (d.vertices * (d.vertices - d.edges_per_vertex)) / 2 = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_count_l20_20729


namespace evaluate_expression_l20_20546

variable (m n p : ℝ)

theorem evaluate_expression 
  (h : m / (140 - m) + n / (210 - n) + p / (180 - p) = 9) :
  10 / (140 - m) + 14 / (210 - n) + 12 / (180 - p) = 40 := 
by 
  sorry

end evaluate_expression_l20_20546


namespace intersection_of_M_and_N_l20_20540

def M : Set ℤ := { x | -3 < x ∧ x < 3 }
def N : Set ℤ := { x | x < 1 }

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := by
  sorry

end intersection_of_M_and_N_l20_20540


namespace total_revenue_correct_l20_20126

noncomputable def total_revenue : ℚ := 
  let revenue_v1 := 23 * 5 * 0.50
  let revenue_v2 := 28 * 6 * 0.60
  let revenue_v3 := 35 * 7 * 0.50
  let revenue_v4 := 43 * 8 * 0.60
  let revenue_v5 := 50 * 9 * 0.50
  let revenue_v6 := 64 * 10 * 0.60
  revenue_v1 + revenue_v2 + revenue_v3 + revenue_v4 + revenue_v5 + revenue_v6

theorem total_revenue_correct : total_revenue = 1096.20 := 
by
  sorry

end total_revenue_correct_l20_20126


namespace sum_excluded_values_domain_l20_20705

theorem sum_excluded_values_domain (x : ℝ) :
  (3 * x^2 - 9 * x + 6 = 0) → (x = 1 ∨ x = 2) ∧ (1 + 2 = 3) :=
by {
  -- given that 3x² - 9x + 6 = 0, we need to show that x = 1 or x = 2, and that their sum is 3
  sorry
}

end sum_excluded_values_domain_l20_20705


namespace total_legs_arms_tentacles_correct_l20_20656

-- Define the counts of different animals
def num_horses : Nat := 2
def num_dogs : Nat := 5
def num_cats : Nat := 7
def num_turtles : Nat := 3
def num_goat : Nat := 1
def num_snakes : Nat := 4
def num_spiders : Nat := 2
def num_birds : Nat := 3
def num_starfish : Nat := 1
def num_octopus : Nat := 1
def num_three_legged_dogs : Nat := 1

-- Define the legs, arms, and tentacles for each type of animal
def legs_per_horse : Nat := 4
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def legs_per_turtle : Nat := 4
def legs_per_goat : Nat := 4
def legs_per_snake : Nat := 0
def legs_per_spider : Nat := 8
def legs_per_bird : Nat := 2
def arms_per_starfish : Nat := 5
def tentacles_per_octopus : Nat := 6
def legs_per_three_legged_dog : Nat := 3

-- Define the total number of legs, arms, and tentacles
def total_legs_arms_tentacles : Nat := 
  (num_horses * legs_per_horse) + 
  (num_dogs * legs_per_dog) + 
  (num_cats * legs_per_cat) + 
  (num_turtles * legs_per_turtle) + 
  (num_goat * legs_per_goat) + 
  (num_snakes * legs_per_snake) + 
  (num_spiders * legs_per_spider) + 
  (num_birds * legs_per_bird) + 
  (num_starfish * arms_per_starfish) + 
  (num_octopus * tentacles_per_octopus) + 
  (num_three_legged_dogs * legs_per_three_legged_dog)

-- The theorem to prove
theorem total_legs_arms_tentacles_correct :
  total_legs_arms_tentacles = 108 := by
  -- Proof goes here
  sorry

end total_legs_arms_tentacles_correct_l20_20656


namespace compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l20_20689

theorem compare_pi_314 : Real.pi > 3.14 :=
by sorry

theorem compare_neg_sqrt3_neg_sqrt2 : -Real.sqrt 3 < -Real.sqrt 2 :=
by sorry

theorem compare_2_sqrt5 : 2 < Real.sqrt 5 :=
by sorry

end compare_pi_314_compare_neg_sqrt3_neg_sqrt2_compare_2_sqrt5_l20_20689


namespace set_intersection_l20_20507

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

noncomputable def complement_U_A := U \ A
noncomputable def intersection := B ∩ complement_U_A

theorem set_intersection :
  intersection = ({3, 4} : Set ℕ) := by
  sorry

end set_intersection_l20_20507


namespace fraction_problem_l20_20423

theorem fraction_problem (x : ℝ) (h₁ : x * 180 = 18) (h₂ : x < 0.15) : x = 1/10 :=
by sorry

end fraction_problem_l20_20423


namespace average_of_k_l20_20598

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l20_20598


namespace sara_total_money_eq_640_l20_20212

def days_per_week : ℕ := 5
def cakes_per_day : ℕ := 4
def price_per_cake : ℕ := 8
def weeks : ℕ := 4

theorem sara_total_money_eq_640 :
  (days_per_week * cakes_per_day * price_per_cake * weeks) = 640 := 
sorry

end sara_total_money_eq_640_l20_20212


namespace correct_model_l20_20247

def average_homework_time_decrease (x : ℝ) : Prop :=
  100 * (1 - x) ^ 2 = 70

theorem correct_model (x : ℝ) : average_homework_time_decrease x := 
  sorry

end correct_model_l20_20247


namespace parallel_vectors_tan_l20_20919

theorem parallel_vectors_tan (θ : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₀ : a = (2, Real.sin θ))
  (h₁ : b = (1, Real.cos θ))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  Real.tan θ = 2 := 
sorry

end parallel_vectors_tan_l20_20919


namespace geometric_sequence_third_term_and_sum_l20_20514

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ :=
  b1 * r^(n - 1)

theorem geometric_sequence_third_term_and_sum (b2 b5 : ℝ) (h1 : b2 = 24.5) (h2 : b5 = 196) :
  (∃ b1 r : ℝ, r ≠ 0 ∧ geometric_sequence b1 r 2 = b2 ∧ geometric_sequence b1 r 5 = b5 ∧
  geometric_sequence b1 r 3 = 49 ∧
  b1 * (r^4 - 1) / (r - 1) = 183.75) :=
by sorry

end geometric_sequence_third_term_and_sum_l20_20514


namespace boaster_guarantee_distinct_balls_l20_20505

noncomputable def canGuaranteeDistinctBallCounts (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)) : Prop :=
  ∀ i j : Fin 2018, i ≠ j → boxes i ≠ boxes j

theorem boaster_guarantee_distinct_balls :
  ∃ (boxes : Fin 2018 → ℕ) (pairs : Fin 4032 → (Fin 2018 × Fin 2018)),
  canGuaranteeDistinctBallCounts boxes pairs :=
sorry

end boaster_guarantee_distinct_balls_l20_20505


namespace findCorrectAnswer_l20_20318

-- Definitions
variable (x : ℕ)
def mistakenCalculation : Prop := 3 * x = 90
def correctAnswer : ℕ := x - 30

-- Theorem statement
theorem findCorrectAnswer (h : mistakenCalculation x) : correctAnswer x = 0 :=
sorry

end findCorrectAnswer_l20_20318


namespace jonah_raisins_l20_20323

variable (y : ℝ)

theorem jonah_raisins :
  (y + 0.4 = 0.7) → (y = 0.3) :=
  by
  intro h
  sorry

end jonah_raisins_l20_20323


namespace compound_interest_comparison_l20_20783

theorem compound_interest_comparison :
  (1 + 0.04) < (1 + 0.04 / 12) ^ 12 := sorry

end compound_interest_comparison_l20_20783


namespace range_of_a_l20_20203

theorem range_of_a (a : ℝ) (h1 : ∃ x : ℝ, x > 0 ∧ |x| = a * x - a) (h2 : ∀ x : ℝ, x < 0 → |x| ≠ a * x - a) : a > 1 :=
sorry

end range_of_a_l20_20203


namespace floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l20_20379

theorem floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2 (n : ℕ) (hn : n > 0) :
  Int.floor (Real.sqrt n + Real.sqrt (n + 1)) = Int.floor (Real.sqrt (4 * n + 2)) := 
  sorry

end floor_sqrt_n_sqrt_n_plus1_eq_floor_sqrt_4n_plus_2_l20_20379


namespace isosceles_triangle_circles_distance_l20_20049

theorem isosceles_triangle_circles_distance (h α : ℝ) (hα : α ≤ π / 6) :
    let R := h / (2 * (Real.cos α)^2)
    let r := h * (Real.tan α) * (Real.tan (π / 4 - α / 2))
    let OO1 := h * (1 - 1 / (2 * (Real.cos α)^2) - (Real.tan α) * (Real.tan (π / 4 - α / 2)))
    OO1 = (2 * h * Real.sin (π / 12 - α / 2) * Real.cos (π / 12 + α / 2)) / (Real.cos α)^2 :=
    sorry

end isosceles_triangle_circles_distance_l20_20049


namespace simplify_expression_l20_20843

theorem simplify_expression (a b c d : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) :
  -5 * a + 2017 * c * d - 5 * b = 2017 :=
by
  sorry

end simplify_expression_l20_20843


namespace landmark_distance_l20_20773

theorem landmark_distance (d : ℝ) : 
  (d >= 7 → d < 7) ∨ (d <= 8 → d > 8) ∨ (d <= 10 → d > 10) → d > 10 :=
by
  sorry

end landmark_distance_l20_20773


namespace president_and_committee_combination_l20_20235

theorem president_and_committee_combination (n : ℕ) (k : ℕ) (total : ℕ) :
  n = 10 ∧ k = 3 ∧ total = (10 * Nat.choose 9 3) → total = 840 :=
by
  intros
  sorry

end president_and_committee_combination_l20_20235


namespace simplify_expression_l20_20828

-- Define the hypotheses and the expression.
variables (x : ℚ)
def expr := (1 + 1 / x) * (1 - 2 / (x + 1)) * (1 + 2 / (x - 1))

-- Define the conditions.
def valid_x : Prop := (x ≠ 0) ∧ (x ≠ -1) ∧ (x ≠ 1)

-- State the main theorem.
theorem simplify_expression (h : valid_x x) : expr x = (x + 1) / x := 
sorry

end simplify_expression_l20_20828


namespace only_positive_odd_integer_dividing_3n_plus_1_l20_20150

theorem only_positive_odd_integer_dividing_3n_plus_1 : 
  ∀ (n : ℕ), (0 < n) → (n % 2 = 1) → (n ∣ (3 ^ n + 1)) → n = 1 := by
  sorry

end only_positive_odd_integer_dividing_3n_plus_1_l20_20150


namespace speed_equivalence_l20_20910

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end speed_equivalence_l20_20910


namespace focus_of_parabola_l20_20487

theorem focus_of_parabola : 
  (∃ p : ℝ, y^2 = 4 * p * x ∧ p = 1 ∧ ∃ c : ℝ × ℝ, c = (1, 0)) :=
sorry

end focus_of_parabola_l20_20487


namespace abs_neg_six_l20_20140

theorem abs_neg_six : |(-6)| = 6 := by
  sorry

end abs_neg_six_l20_20140


namespace smallest_value_inequality_l20_20186

variable (a b c d : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem smallest_value_inequality :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
sorry

end smallest_value_inequality_l20_20186


namespace ratio_of_Jordyn_age_to_Zrinka_age_is_2_l20_20968

variable (Mehki_age : ℕ) (Jordyn_age : ℕ) (Zrinka_age : ℕ)

-- Conditions
def Mehki_is_10_years_older_than_Jordyn := Mehki_age = Jordyn_age + 10
def Zrinka_age_is_6 := Zrinka_age = 6
def Mehki_age_is_22 := Mehki_age = 22

-- Theorem statement: the ratio of Jordyn's age to Zrinka's age is 2.
theorem ratio_of_Jordyn_age_to_Zrinka_age_is_2
  (h1 : Mehki_is_10_years_older_than_Jordyn Mehki_age Jordyn_age)
  (h2 : Zrinka_age_is_6 Zrinka_age)
  (h3 : Mehki_age_is_22 Mehki_age) : Jordyn_age / Zrinka_age = 2 :=
by
  -- The proof would go here
  sorry

end ratio_of_Jordyn_age_to_Zrinka_age_is_2_l20_20968


namespace sum_mod_17_l20_20756

theorem sum_mod_17 :
  (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) % 17 = 6 :=
by
  sorry

end sum_mod_17_l20_20756


namespace probability_of_correct_digit_in_two_attempts_l20_20564

theorem probability_of_correct_digit_in_two_attempts : 
  let num_possible_digits := 10
  let num_attempts := 2
  let total_possible_outcomes := num_possible_digits * (num_possible_digits - 1)
  let total_favorable_outcomes := (num_possible_digits - 1) + (num_possible_digits - 1)
  let probability := (total_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)
  probability = (1 / 5 : ℚ) :=
by
  sorry

end probability_of_correct_digit_in_two_attempts_l20_20564


namespace train_length_correct_l20_20058

noncomputable def train_length (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * time

theorem train_length_correct :
  train_length 17.998560115190784 36 = 179.98560115190784 :=
by
  sorry

end train_length_correct_l20_20058


namespace other_group_less_garbage_l20_20954

theorem other_group_less_garbage :
  387 + (735 - 387) = 735 :=
by
  sorry

end other_group_less_garbage_l20_20954


namespace lego_tower_levels_l20_20991

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end lego_tower_levels_l20_20991


namespace measure_of_angle_C_l20_20410

variable (C D : ℕ)
variable (h1 : C + D = 180)
variable (h2 : C = 5 * D)

theorem measure_of_angle_C : C = 150 :=
by
  sorry

end measure_of_angle_C_l20_20410


namespace total_chocolate_bar_count_l20_20827

def large_box_count : ℕ := 150
def small_box_count_per_large_box : ℕ := 45
def chocolate_bar_count_per_small_box : ℕ := 35

theorem total_chocolate_bar_count :
  large_box_count * small_box_count_per_large_box * chocolate_bar_count_per_small_box = 236250 :=
by
  sorry

end total_chocolate_bar_count_l20_20827


namespace arithmetic_seq_sum_l20_20745

variable {a_n : ℕ → ℕ}
variable (S_n : ℕ → ℕ)
variable (q : ℕ)
variable (a_1 : ℕ)

axiom h1 : a_n 2 = 2
axiom h2 : a_n 6 = 32
axiom h3 : ∀ n, S_n n = a_1 * (1 - q ^ n) / (1 - q)

theorem arithmetic_seq_sum : S_n 100 = 2^100 - 1 :=
by
  sorry

end arithmetic_seq_sum_l20_20745


namespace fg_of_3_eq_neg5_l20_20432

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Lean statement to prove question == answer
theorem fg_of_3_eq_neg5 : f (g 3) = -5 := by
  sorry

end fg_of_3_eq_neg5_l20_20432


namespace necessary_but_not_sufficient_l20_20644

-- Define the conditions as seen in the problem statement
def condition_x (x : ℝ) : Prop := x < 0
def condition_ln (x : ℝ) : Prop := Real.log (x + 1) < 0

-- State that the condition "x < 0" is necessary but not sufficient for "ln(x + 1) < 0"
theorem necessary_but_not_sufficient :
  ∀ (x : ℝ), (condition_ln x → condition_x x) ∧ ¬(condition_x x → condition_ln x) :=
by
  sorry

end necessary_but_not_sufficient_l20_20644


namespace total_bill_l20_20039

theorem total_bill (total_friends : ℕ) (extra_payment : ℝ) (total_bill : ℝ) (paid_by_friends : ℝ) :
  total_friends = 8 → extra_payment = 2.50 →
  (7 * ((total_bill / total_friends) + extra_payment)) = total_bill →
  total_bill = 140 :=
by
  intros h1 h2 h3
  sorry

end total_bill_l20_20039


namespace julien_swims_50_meters_per_day_l20_20842

-- Definitions based on given conditions
def distance_julien_swims_per_day : ℕ := 50
def distance_sarah_swims_per_day (J : ℕ) : ℕ := 2 * J
def distance_jamir_swims_per_day (J : ℕ) : ℕ := distance_sarah_swims_per_day J + 20
def combined_distance_per_day (J : ℕ) : ℕ := J + distance_sarah_swims_per_day J + distance_jamir_swims_per_day J
def combined_distance_per_week (J : ℕ) : ℕ := 7 * combined_distance_per_day J

-- Proof statement 
theorem julien_swims_50_meters_per_day :
  combined_distance_per_week distance_julien_swims_per_day = 1890 :=
by
  -- We are formulating the proof without solving it, to be proven formally in Lean
  sorry

end julien_swims_50_meters_per_day_l20_20842


namespace triangle_inequality_a2_lt_ab_ac_l20_20503

theorem triangle_inequality_a2_lt_ab_ac {a b c : ℝ} (h1 : a < b + c) (h2 : 0 < a) : a^2 < a * b + a * c := 
sorry

end triangle_inequality_a2_lt_ab_ac_l20_20503


namespace distance_between_lines_l20_20242

-- Definitions from conditions in (a)
def l1 (x y : ℝ) := 3 * x + 4 * y - 7 = 0
def l2 (x y : ℝ) := 6 * x + 8 * y + 1 = 0

-- The proof goal from (c)
theorem distance_between_lines : 
  ∀ (x y : ℝ),
    (l1 x y) → 
    (l2 x y) →
      -- Distance between the lines is 3/2
      ( (|(-14) - 1| : ℝ) / (Real.sqrt (6^2 + 8^2)) ) = 3 / 2 :=
by
  sorry

end distance_between_lines_l20_20242


namespace remainder_3042_div_29_l20_20001

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end remainder_3042_div_29_l20_20001


namespace sequence_term_divisible_by_n_l20_20256

theorem sequence_term_divisible_by_n (n : ℕ) (hn1 : 1 < n) (hn_odd : n % 2 = 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ n ∣ (2^k - 1) :=
by
  sorry

end sequence_term_divisible_by_n_l20_20256


namespace zoo_feeding_ways_l20_20684

-- Noncomputable is used for definitions that are not algorithmically computable
noncomputable def numFeedingWays : Nat :=
  4 * 3 * 3 * 2 * 2

theorem zoo_feeding_ways :
  ∀ (pairs : Fin 4 → (String × String)), -- Representing pairs of animals
  numFeedingWays = 144 :=
by
  sorry

end zoo_feeding_ways_l20_20684


namespace correct_calculation_l20_20569

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end correct_calculation_l20_20569


namespace snow_at_Brecknock_l20_20750

theorem snow_at_Brecknock (hilt_snow brecknock_snow : ℕ) (h1 : hilt_snow = 29) (h2 : hilt_snow = brecknock_snow + 12) : brecknock_snow = 17 :=
by
  sorry

end snow_at_Brecknock_l20_20750


namespace min_disks_needed_l20_20305

/-- 
  Sandhya must save 35 files onto disks, each with 1.44 MB space. 
  5 of the files take up 0.6 MB, 18 of the files take up 0.5 MB, 
  and the rest take up 0.3 MB. Files cannot be split across disks.
  Prove that the smallest number of disks needed to store all 35 files is 12.
--/
theorem min_disks_needed 
  (total_files : ℕ)
  (disk_capacity : ℝ)
  (file_sizes : ℕ → ℝ)
  (files_0_6_MB : ℕ)
  (files_0_5_MB : ℕ)
  (files_0_3_MB : ℕ)
  (remaining_files : ℕ)
  (storage_per_disk : ℝ)
  (smallest_disks_needed : ℕ) 
  (h1 : total_files = 35)
  (h2 : disk_capacity = 1.44)
  (h3 : file_sizes 0 = 0.6)
  (h4 : file_sizes 1 = 0.5)
  (h5 : file_sizes 2 = 0.3)
  (h6 : files_0_6_MB = 5)
  (h7 : files_0_5_MB = 18)
  (h8 : remaining_files = total_files - files_0_6_MB - files_0_5_MB)
  (h9 : remaining_files = 12)
  (h10 : storage_per_disk = file_sizes 0 * 2 + file_sizes 1 + file_sizes 2)
  (h11 : smallest_disks_needed = 12) :
  total_files = 35 ∧ disk_capacity = 1.44 ∧ storage_per_disk <= 1.44 ∧ smallest_disks_needed = 12 :=
by
  sorry

end min_disks_needed_l20_20305


namespace grain_output_l20_20040

-- Define the condition regarding grain output.
def premier_goal (x : ℝ) : Prop :=
  x > 1.3

-- The mathematical statement that needs to be proved, given the condition.
theorem grain_output (x : ℝ) (h : premier_goal x) : x > 1.3 :=
by
  sorry

end grain_output_l20_20040


namespace bills_average_speed_l20_20167

theorem bills_average_speed :
  ∃ v t : ℝ, 
      (v + 5) * (t + 2) + v * t = 680 ∧ 
      (t + 2) + t = 18 ∧ 
      v = 35 :=
by
  sorry

end bills_average_speed_l20_20167


namespace solution_l20_20358

variable (x y z : ℝ)

noncomputable def problem := 
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
  x^2 + x * y + y^2 = 48 →
  y^2 + y * z + z^2 = 25 →
  z^2 + z * x + x^2 = 73 →
  x * y + y * z + z * x = 40

theorem solution : problem := by
  intros
  sorry

end solution_l20_20358


namespace one_elephant_lake_empty_in_365_days_l20_20720

variables (C K V : ℝ)
variables (t : ℝ)

noncomputable def lake_empty_one_day (C K V : ℝ) := 183 * C = V + K
noncomputable def lake_empty_five_days (C K V : ℝ) := 185 * C = V + 5 * K

noncomputable def elephant_time (C K V t : ℝ) : Prop :=
  (t * C = V + t * K) → (t = 365)

theorem one_elephant_lake_empty_in_365_days (C K V t : ℝ) :
  (lake_empty_one_day C K V) →
  (lake_empty_five_days C K V) →
  (elephant_time C K V t) := by
  intros h1 h2 h3
  sorry

end one_elephant_lake_empty_in_365_days_l20_20720


namespace find_2n_plus_m_l20_20452

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 2 * n + m = 36 :=
sorry

end find_2n_plus_m_l20_20452


namespace circle_radius_integer_l20_20251

theorem circle_radius_integer (r : ℤ)
  (center : ℝ × ℝ)
  (inside_point : ℝ × ℝ)
  (outside_point : ℝ × ℝ)
  (h1 : center = (-2, -3))
  (h2 : inside_point = (-2, 2))
  (h3 : outside_point = (5, -3))
  (h4 : (dist center inside_point : ℝ) < r)
  (h5 : (dist center outside_point : ℝ) > r) 
  : r = 6 :=
sorry

end circle_radius_integer_l20_20251


namespace largest_four_digit_number_l20_20134

theorem largest_four_digit_number
  (n : ℕ) (hn1 : n % 8 = 2) (hn2 : n % 7 = 4) (hn3 : 1000 ≤ n) (hn4 : n ≤ 9999) :
  n = 9990 :=
sorry

end largest_four_digit_number_l20_20134


namespace negation_of_exists_gt_1_l20_20958

theorem negation_of_exists_gt_1 :
  (∀ x : ℝ, x ≤ 1) ↔ ¬ (∃ x : ℝ, x > 1) :=
sorry

end negation_of_exists_gt_1_l20_20958


namespace number_of_people_in_group_l20_20241

-- Definitions and conditions
def total_cost : ℕ := 94
def mango_juice_cost : ℕ := 5
def pineapple_juice_cost : ℕ := 6
def pineapple_cost_total : ℕ := 54

-- Theorem statement to prove
theorem number_of_people_in_group : 
  ∃ M P : ℕ, 
    mango_juice_cost * M + pineapple_juice_cost * P = total_cost ∧ 
    pineapple_juice_cost * P = pineapple_cost_total ∧ 
    M + P = 17 := 
by 
  sorry

end number_of_people_in_group_l20_20241


namespace evaluate_expression_l20_20158

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end evaluate_expression_l20_20158


namespace distinct_real_roots_of_quadratic_l20_20174

/-
Given a quadratic equation x^2 + 4x = 0,
prove that the equation has two distinct real roots.
-/

theorem distinct_real_roots_of_quadratic : 
  ∀ (a b c : ℝ), a = 1 → b = 4 → c = 0 → (b^2 - 4 * a * c) > 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ (r₁^2 + 4 * r₁ = 0) ∧ (r₂^2 + 4 * r₂ = 0) := 
by
  intros a b c ha hb hc hΔ
  sorry -- Proof to be provided later

end distinct_real_roots_of_quadratic_l20_20174


namespace milkman_total_profit_l20_20445

-- Declare the conditions
def initialMilk : ℕ := 50
def initialWater : ℕ := 15
def firstMixtureMilk : ℕ := 30
def firstMixtureWater : ℕ := 8
def remainingMilk : ℕ := initialMilk - firstMixtureMilk
def secondMixtureMilk : ℕ := remainingMilk
def secondMixtureWater : ℕ := 7
def costOfMilkPerLiter : ℕ := 20
def sellingPriceFirstMixturePerLiter : ℕ := 17
def sellingPriceSecondMixturePerLiter : ℕ := 15
def totalCostOfMilk := (firstMixtureMilk + secondMixtureMilk) * costOfMilkPerLiter
def totalRevenueFirstMixture := (firstMixtureMilk + firstMixtureWater) * sellingPriceFirstMixturePerLiter
def totalRevenueSecondMixture := (secondMixtureMilk + secondMixtureWater) * sellingPriceSecondMixturePerLiter
def totalRevenue := totalRevenueFirstMixture + totalRevenueSecondMixture
def totalProfit := totalRevenue - totalCostOfMilk

-- Proof statement
theorem milkman_total_profit : totalProfit = 51 := by
  sorry

end milkman_total_profit_l20_20445


namespace geometric_series_sum_l20_20313

theorem geometric_series_sum :
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  S₅ = 61 / 243 := by
  let a := (1 : ℚ) / 3
  let r := -(1 / 3)
  let n := 5
  let S₅ := (a * (1 - r ^ n)) / (1 - r)
  sorry

end geometric_series_sum_l20_20313


namespace smallest_c_for_3_in_range_l20_20125

theorem smallest_c_for_3_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, (x^2 - 6 * x + c) = 3) ↔ (c ≥ 12) :=
by {
  sorry
}

end smallest_c_for_3_in_range_l20_20125


namespace meaningful_expression_iff_l20_20485

theorem meaningful_expression_iff (x : ℝ) : (∃ y, y = (2 : ℝ) / (2*x - 1)) ↔ x ≠ (1 / 2 : ℝ) :=
by
  sorry

end meaningful_expression_iff_l20_20485


namespace min_value_of_expression_l20_20377

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 27) 
  : x + 3 * y + 9 * z ≥ 27 :=
sorry

end min_value_of_expression_l20_20377


namespace beta_max_success_ratio_l20_20033

theorem beta_max_success_ratio :
  ∃ (a b c d : ℕ),
    0 < a ∧ a < b ∧
    0 < c ∧ c < d ∧
    b + d ≤ 550 ∧
    (15 * a < 8 * b) ∧ (10 * c < 7 * d) ∧
    (21 * a + 16 * c < 4400) ∧
    ((a + c) / (b + d : ℚ) = 274 / 550) :=
sorry

end beta_max_success_ratio_l20_20033


namespace min_value_of_f_in_D_l20_20180

noncomputable def f (x y : ℝ) : ℝ := 6 * (x^2 + y^2) * (x + y) - 4 * (x^2 + x * y + y^2) - 3 * (x + y) + 5

def D (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem min_value_of_f_in_D : ∃ (x y : ℝ), D x y ∧ f x y = 2 ∧ (∀ (u v : ℝ), D u v → f u v ≥ 2) :=
by
  sorry

end min_value_of_f_in_D_l20_20180


namespace math_problem_l20_20389

/-
Two mathematicians take a morning coffee break each day.
They arrive at the cafeteria independently, at random times between 9 a.m. and 10:30 a.m.,
and stay for exactly m minutes.
Given the probability that either one arrives while the other is in the cafeteria is 30%,
and m = a - b√c, where a, b, and c are positive integers, and c is not divisible by the square of any prime,
prove that a + b + c = 127.

-/

noncomputable def is_square_free (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p * p ∣ c → False

theorem math_problem
  (m a b c : ℕ)
  (h1 : 0 < m)
  (h2 : m = a - b * Real.sqrt c)
  (h3 : is_square_free c)
  (h4 : 30 * (90 * 90) / 100 = (90 - m) * (90 - m)) :
  a + b + c = 127 :=
sorry

end math_problem_l20_20389


namespace bones_in_beef_l20_20424

def price_of_beef_with_bones : ℝ := 78
def price_of_boneless_beef : ℝ := 90
def price_of_bones : ℝ := 15
def fraction_of_bones_in_kg : ℝ := 0.16
def grams_per_kg : ℝ := 1000

theorem bones_in_beef :
  (fraction_of_bones_in_kg * grams_per_kg = 160) :=
by
  sorry

end bones_in_beef_l20_20424


namespace cell_survival_after_6_hours_l20_20927

def cell_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 2) ∧ (∀ n, a (n + 1) = 2 * a n - 1)

theorem cell_survival_after_6_hours :
  ∃ a : ℕ → ℕ, cell_sequence a ∧ a 6 = 65 :=
by
  sorry

end cell_survival_after_6_hours_l20_20927


namespace probability_of_different_colors_l20_20527

theorem probability_of_different_colors :
  let total_chips := 12
  let prob_blue_then_yellow_red := ((6 / total_chips) * ((4 + 2) / total_chips))
  let prob_yellow_then_blue_red := ((4 / total_chips) * ((6 + 2) / total_chips))
  let prob_red_then_blue_yellow := ((2 / total_chips) * ((6 + 4) / total_chips))
  prob_blue_then_yellow_red + prob_yellow_then_blue_red + prob_red_then_blue_yellow = 11 / 18 := by
    sorry

end probability_of_different_colors_l20_20527


namespace function_domain_l20_20480

theorem function_domain (x : ℝ) :
  (x + 5 ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≥ -5) ∧ (x ≠ -2) :=
by
  sorry

end function_domain_l20_20480


namespace increasing_function_a_l20_20976

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then
    x^2
  else
    x^3 - (a-1)*x + a^2 - 3*a - 4

theorem increasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end increasing_function_a_l20_20976


namespace triangle_angle_contradiction_l20_20372

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (h_sum : A + B + C = 180) :
  false :=
by
  -- Here "A > 60, B > 60, C > 60 and A + B + C = 180" leads to a contradiction
  sorry

end triangle_angle_contradiction_l20_20372


namespace instantaneous_velocity_at_t2_l20_20876

noncomputable def s (t : ℝ) : ℝ := t^3 - t^2 + 2 * t

theorem instantaneous_velocity_at_t2 : 
  deriv s 2 = 10 := 
by
  sorry

end instantaneous_velocity_at_t2_l20_20876


namespace chess_tournament_games_l20_20127

def number_of_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_games :
  number_of_games 20 = 190 :=
by
  sorry

end chess_tournament_games_l20_20127


namespace eggs_not_eaten_is_6_l20_20306

noncomputable def eggs_not_eaten_each_week 
  (trays_purchased : ℕ) 
  (eggs_per_tray : ℕ) 
  (eggs_morning : ℕ) 
  (days_in_week : ℕ) 
  (eggs_night : ℕ) : ℕ :=
  let total_eggs := trays_purchased * eggs_per_tray
  let eggs_eaten_son_daughter := eggs_morning * days_in_week
  let eggs_eaten_rhea_husband := eggs_night * days_in_week
  let eggs_eaten_total := eggs_eaten_son_daughter + eggs_eaten_rhea_husband
  total_eggs - eggs_eaten_total

theorem eggs_not_eaten_is_6 
  (trays_purchased : ℕ := 2) 
  (eggs_per_tray : ℕ := 24) 
  (eggs_morning : ℕ := 2) 
  (days_in_week : ℕ := 7) 
  (eggs_night : ℕ := 4) : 
  eggs_not_eaten_each_week trays_purchased eggs_per_tray eggs_morning days_in_week eggs_night = 6 :=
by
  -- Here should be proof steps, but we use sorry to skip it as per instruction
  sorry

end eggs_not_eaten_is_6_l20_20306


namespace not_product_of_consecutives_l20_20135

theorem not_product_of_consecutives (n k : ℕ) : 
  ¬ (∃ a b: ℕ, a + 1 = b ∧ (2 * n^(3 * k) + 4 * n^k + 10 = a * b)) :=
by sorry

end not_product_of_consecutives_l20_20135


namespace simplify_fraction_l20_20660

theorem simplify_fraction :
  ∀ (x : ℝ),
    (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) /
    (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) =
    (2 * x + 3) / (2 * x - 3) :=
by sorry

end simplify_fraction_l20_20660


namespace sum_of_integers_between_neg20_5_and_10_5_l20_20504

noncomputable def sum_arithmetic_series (a l n : ℤ) : ℤ :=
  n * (a + l) / 2

theorem sum_of_integers_between_neg20_5_and_10_5 :
  (sum_arithmetic_series (-20) 10 31) = -155 := by
  sorry

end sum_of_integers_between_neg20_5_and_10_5_l20_20504


namespace polar_bear_daily_food_l20_20330

-- Definitions based on the conditions
def bucketOfTroutDaily : ℝ := 0.2
def bucketOfSalmonDaily : ℝ := 0.4

-- The proof statement
theorem polar_bear_daily_food : bucketOfTroutDaily + bucketOfSalmonDaily = 0.6 := by
  sorry

end polar_bear_daily_food_l20_20330


namespace cannot_form_shape_B_l20_20055

-- Define the given pieces
def pieces : List (List (Nat × Nat)) :=
  [ [(1, 1)],
    [(1, 2)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 3)],
    [(1, 3)] ]

-- Define shape B requirement
def shapeB : List (Nat × Nat) := [(1, 6)]

theorem cannot_form_shape_B :
  ¬ (∃ (combinations : List (List (Nat × Nat))), combinations ⊆ pieces ∧ 
     (List.foldr (λ x acc => acc + x) 0 (combinations.map (List.foldr (λ y acc => acc + (y.1 * y.2)) 0)) = 6)) :=
sorry

end cannot_form_shape_B_l20_20055


namespace menu_choices_l20_20809

theorem menu_choices :
  let lunchChinese := 5 
  let lunchJapanese := 4 
  let dinnerChinese := 3 
  let dinnerJapanese := 5 
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  lunchOptions * dinnerOptions = 72 :=
by
  let lunchChinese := 5
  let lunchJapanese := 4
  let dinnerChinese := 3
  let dinnerJapanese := 5
  let lunchOptions := lunchChinese + lunchJapanese
  let dinnerOptions := dinnerChinese + dinnerJapanese
  have h : lunchOptions * dinnerOptions = 72 :=
    by 
      sorry
  exact h

end menu_choices_l20_20809


namespace water_park_children_l20_20811

theorem water_park_children (cost_adult cost_child total_cost : ℝ) (c : ℕ) 
  (h1 : cost_adult = 1)
  (h2 : cost_child = 0.75)
  (h3 : total_cost = 3.25) :
  c = 3 :=
by
  sorry

end water_park_children_l20_20811


namespace correct_option_l20_20007

variable (a : ℝ)

theorem correct_option (h1 : 5 * a^2 - 4 * a^2 = a^2)
                       (h2 : a^7 / a^4 = a^3)
                       (h3 : (a^3)^2 = a^6)
                       (h4 : a^2 * a^3 = a^5) : 
                       a^7 / a^4 = a^3 := 
by
  exact h2

end correct_option_l20_20007


namespace inequality_abc_l20_20413

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / (a^2 + b^2) + (b + c) / (b^2 + c^2) + (c + a) / (c^2 + a^2) ≤ 1/a + 1/b + 1/c := 
by
  sorry

end inequality_abc_l20_20413


namespace min_expr_l20_20840

theorem min_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = 1) :
  ∃ s : ℝ, (s = a + b) ∧ (s ≥ 2) ∧ (a^2 + b^2 + 4/(s^2) = 3) :=
by sorry

end min_expr_l20_20840


namespace johns_elevation_after_travel_l20_20938

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l20_20938


namespace final_price_percentage_l20_20159

theorem final_price_percentage (P : ℝ) (h₀ : P > 0)
  (h₁ : ∃ P₁, P₁ = 0.80 * P)
  (h₂ : ∃ P₂, P₁ = 0.80 * P ∧ P₂ = P₁ - 0.10 * P₁) :
  P₂ = 0.72 * P :=
by
  sorry

end final_price_percentage_l20_20159


namespace geometric_progression_product_l20_20325

variables {n : ℕ} {b q S S' P : ℝ} 

theorem geometric_progression_product (hb : b ≠ 0) (hq : q ≠ 1)
  (hP : P = b^n * q^(n*(n-1)/2))
  (hS : S = b * (1 - q^n) / (1 - q))
  (hS' : S' = (q^n - 1) / (b * (q - 1)))
  : P = (S * S')^(n/2) := 
sorry

end geometric_progression_product_l20_20325


namespace cost_per_crayon_l20_20597

-- Definitions for conditions
def half_dozen := 6
def total_crayons := 4 * half_dozen
def total_cost := 48

-- Problem statement
theorem cost_per_crayon :
  (total_cost / total_crayons) = 2 := 
  by
    sorry

end cost_per_crayon_l20_20597


namespace mean_combined_set_l20_20408

noncomputable def mean (s : Finset ℚ) : ℚ :=
  (s.sum id) / s.card

theorem mean_combined_set :
  ∀ (s1 s2 : Finset ℚ),
  s1.card = 7 →
  s2.card = 8 →
  mean s1 = 15 →
  mean s2 = 18 →
  mean (s1 ∪ s2) = 249 / 15 :=
by
  sorry

end mean_combined_set_l20_20408


namespace point_A_in_third_quadrant_l20_20530

-- Defining the point A with its coordinates
structure Point :=
  (x : Int)
  (y : Int)

def A : Point := ⟨-1, -3⟩

-- The definition of quadrants in Cartesian coordinate system
def quadrant (p : Point) : String :=
  if p.x > 0 ∧ p.y > 0 then "first"
  else if p.x < 0 ∧ p.y > 0 then "second"
  else if p.x < 0 ∧ p.y < 0 then "third"
  else if p.x > 0 ∧ p.y < 0 then "fourth"
  else "boundary"

-- The theorem we want to prove
theorem point_A_in_third_quadrant : quadrant A = "third" :=
by 
  sorry

end point_A_in_third_quadrant_l20_20530


namespace solve_fraction_l20_20387

theorem solve_fraction : (3.242 * 10) / 100 = 0.3242 := by
  sorry

end solve_fraction_l20_20387


namespace polynomial_expansion_l20_20380

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 :=
by
  sorry

end polynomial_expansion_l20_20380


namespace calculate_expression_l20_20797

theorem calculate_expression :
  (2^3 * 3 * 5) + (18 / 2) = 129 := by
  -- Proof skipped
  sorry

end calculate_expression_l20_20797


namespace ship_length_correct_l20_20742

noncomputable def ship_length : ℝ :=
  let speed_kmh := 24
  let speed_mps := speed_kmh * 1000 / 3600
  let time := 202.48
  let bridge_length := 900
  let total_distance := speed_mps * time
  total_distance - bridge_length

theorem ship_length_correct : ship_length = 450.55 :=
by
  -- This is where the proof would be written, but we're skipping the proof as per instructions
  sorry

end ship_length_correct_l20_20742


namespace mod_50_remainder_of_b86_l20_20096

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem mod_50_remainder_of_b86 : (b 86) % 50 = 40 := 
by 
-- Given definition of b and the problem is to prove the remainder of b_86 when divided by 50 is 40
sorry

end mod_50_remainder_of_b86_l20_20096


namespace problem_l20_20200

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

def f_prime (a b x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * b

theorem problem (a b c : ℝ) (h1 : f_prime a b 2 = 0) (h2 : f_prime a b 1 = -3) :
  a = -1 ∧ b = 0 ∧ (let f_min := f (-1) 0 c 2 
                   let f_max := 0 
                   f_max - f_min = 4) :=
by
  sorry

end problem_l20_20200


namespace solve_equation_l20_20508

theorem solve_equation : ∀ x : ℝ, (2 * x - 8 = 0) ↔ (x = 4) :=
by sorry

end solve_equation_l20_20508


namespace solution_set_of_quadratic_inequality_l20_20893

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | 2 - x - x^2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_quadratic_inequality_l20_20893


namespace nat_add_ge_3_implies_at_least_one_ge_2_l20_20521

theorem nat_add_ge_3_implies_at_least_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by {
  sorry
}

end nat_add_ge_3_implies_at_least_one_ge_2_l20_20521


namespace colin_speed_l20_20089

variable (B T Bn C : ℝ)
variable (m : ℝ)

-- Given conditions
def condition1 := B = 1
def condition2 := T = m * B
def condition3 := Bn = T / 3
def condition4 := C = 6 * Bn
def condition5 := C = 4

-- We need to prove C = 4 given the conditions
theorem colin_speed :
  (B = 1) →
  (T = m * B) →
  (Bn = T / 3) →
  (C = 6 * Bn) →
  C = 4 :=
by
  intros _ _ _ _
  sorry

end colin_speed_l20_20089


namespace expected_value_of_win_l20_20243

noncomputable def win_amount (n : ℕ) : ℕ :=
  2 * n^2

noncomputable def expected_value : ℝ :=
  (1/8) * (win_amount 1 + win_amount 2 + win_amount 3 + win_amount 4 + win_amount 5 + win_amount 6 + win_amount 7 + win_amount 8)

theorem expected_value_of_win :
  expected_value = 51 := by
  sorry

end expected_value_of_win_l20_20243


namespace odd_function_fixed_point_l20_20996

noncomputable def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_fixed_point 
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f) :
  f (0) = 0 → f (-1 + 1) - 2 = -2 :=
by
  sorry

end odd_function_fixed_point_l20_20996


namespace f_at_count_l20_20400

def f (a b c : ℕ) : ℕ := (a * b * c) / (Nat.gcd (Nat.gcd a b) c * Nat.lcm (Nat.lcm a b) c)

def is_f_at (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≤ 60 ∧ y ≤ 60 ∧ z ≤ 60 ∧ f x y z = n

theorem f_at_count : ∃ (n : ℕ), n = 70 ∧ ∀ k, is_f_at k → k ≤ 70 := 
sorry

end f_at_count_l20_20400


namespace f_g_2_eq_1_l20_20171

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -2 * x + 5

theorem f_g_2_eq_1 : f (g 2) = 1 :=
by
  sorry

end f_g_2_eq_1_l20_20171


namespace mean_tasks_b_l20_20495

variable (a b : ℕ)
variable (m_a m_b : ℕ)
variable (h1 : a + b = 260)
variable (h2 : a = 3 * b / 10 + b)
variable (h3 : m_a = 80)
variable (h4 : m_b = 12 * m_a / 10)

theorem mean_tasks_b :
  m_b = 96 := by
  -- This is where the proof would go
  sorry

end mean_tasks_b_l20_20495


namespace add_fractions_l20_20288

theorem add_fractions : (1 / 4 : ℚ) + (3 / 5) = 17 / 20 := 
by
  sorry

end add_fractions_l20_20288


namespace jimmy_points_l20_20343

theorem jimmy_points (eng_pts init_eng_pts : ℕ) (math_pts init_math_pts : ℕ) 
  (sci_pts init_sci_pts : ℕ) (hist_pts init_hist_pts : ℕ) 
  (phy_pts init_phy_pts : ℕ) (eng_penalty math_penalty sci_penalty hist_penalty phy_penalty : ℕ)
  (passing_points : ℕ) (total_points_required : ℕ):
  init_eng_pts = 60 →
  init_math_pts = 55 →
  init_sci_pts = 40 →
  init_hist_pts = 70 →
  init_phy_pts = 50 →
  eng_penalty = 5 →
  math_penalty = 3 →
  sci_penalty = 8 →
  hist_penalty = 2 →
  phy_penalty = 6 →
  passing_points = 250 →
  total_points_required = (init_eng_pts - eng_penalty) + (init_math_pts - math_penalty) + 
                         (init_sci_pts - sci_penalty) + (init_hist_pts - hist_penalty) + 
                         (init_phy_pts - phy_penalty) →
  ∀ extra_loss, (total_points_required - extra_loss ≥ passing_points) → extra_loss ≤ 1 :=
by {
  sorry
}

end jimmy_points_l20_20343


namespace ratio_of_sister_to_Aaron_l20_20002

noncomputable def Aaron_age := 15
variable (H S : ℕ)
axiom Henry_age_relation : H = 4 * S
axiom combined_age : H + S + Aaron_age = 240

theorem ratio_of_sister_to_Aaron : (S : ℚ) / Aaron_age = 3 := 
by
  -- Proof omitted
  sorry

end ratio_of_sister_to_Aaron_l20_20002


namespace final_amoeba_is_blue_l20_20600

theorem final_amoeba_is_blue
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ)
  (merge : ∀ (a b : ℕ), a ≠ b → ∃ c, a + b - c = a ∧ a + b - c = b ∧ a + b - c = c)
  (initial_counts : n1 = 26 ∧ n2 = 31 ∧ n3 = 16)
  (final_count : ∃ a, a = 1) :
  ∃ color, color = "blue" := sorry

end final_amoeba_is_blue_l20_20600


namespace area_of_50th_ring_l20_20591

-- Definitions based on conditions:
def garden_area : ℕ := 9
def ring_area (n : ℕ) : ℕ := 9 * ((2 * n + 1) ^ 2 - (2 * (n - 1) + 1) ^ 2) / 2

-- Theorem to prove:
theorem area_of_50th_ring : ring_area 50 = 1800 := by sorry

end area_of_50th_ring_l20_20591


namespace parking_lot_wheels_l20_20821

noncomputable def total_car_wheels (guest_cars : Nat) (guest_car_wheels : Nat) (parent_cars : Nat) (parent_car_wheels : Nat) : Nat :=
  guest_cars * guest_car_wheels + parent_cars * parent_car_wheels

theorem parking_lot_wheels :
  total_car_wheels 10 4 2 4 = 48 :=
by
  sorry

end parking_lot_wheels_l20_20821


namespace quadratic_rewrite_l20_20685

theorem quadratic_rewrite (d e f : ℤ) (h1 : d^2 = 25) (h2 : 2 * d * e = -40) (h3 : e^2 + f = -75) : d * e = -20 := 
by 
  sorry

end quadratic_rewrite_l20_20685


namespace three_digit_number_divisible_by_eleven_l20_20988

theorem three_digit_number_divisible_by_eleven
  (x : ℕ) (n : ℕ)
  (units_digit_is_two : n % 10 = 2)
  (hundreds_digit_is_seven : n / 100 = 7)
  (tens_digit : n = 700 + x * 10 + 2)
  (divisibility_condition : (7 - x + 2) % 11 = 0) :
  n = 792 := by
  sorry

end three_digit_number_divisible_by_eleven_l20_20988


namespace alice_sales_goal_l20_20806

def price_adidas := 45
def price_nike := 60
def price_reeboks := 35
def price_puma := 50
def price_converse := 40

def num_adidas := 10
def num_nike := 12
def num_reeboks := 15
def num_puma := 8
def num_converse := 14

def quota := 2000

def total_sales :=
  (num_adidas * price_adidas) +
  (num_nike * price_nike) +
  (num_reeboks * price_reeboks) +
  (num_puma * price_puma) +
  (num_converse * price_converse)

def exceed_amount := total_sales - quota

theorem alice_sales_goal : exceed_amount = 655 := by
  -- calculation steps would go here
  sorry

end alice_sales_goal_l20_20806


namespace solid2_solid4_views_identical_l20_20945

-- Define the solids and their orthographic views
structure Solid :=
  (top_view : String)
  (front_view : String)
  (side_view : String)

-- Given solids as provided by the problem
def solid1 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid2 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid3 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid4 : Solid := { top_view := "...", front_view := "...", side_view := "..." }

-- Function to compare two solids' views
def views_identical (s1 s2 : Solid) : Prop :=
  (s1.top_view = s2.top_view ∧ s1.front_view = s2.front_view) ∨
  (s1.top_view = s2.top_view ∧ s1.side_view = s2.side_view) ∨
  (s1.front_view = s2.front_view ∧ s1.side_view = s2.side_view)

-- Theorem statement
theorem solid2_solid4_views_identical : views_identical solid2 solid4 := 
sorry

end solid2_solid4_views_identical_l20_20945


namespace simplify_expression_l20_20762

theorem simplify_expression : 
  (81 ^ (1 / Real.logb 5 9) + 3 ^ (3 / Real.logb (Real.sqrt 6) 3)) / 409 * 
  ((Real.sqrt 7) ^ (2 / Real.logb 25 7) - 125 ^ (Real.logb 25 6)) = 1 :=
by 
  sorry

end simplify_expression_l20_20762


namespace Theresa_helper_hours_l20_20563

theorem Theresa_helper_hours :
  ∃ x : ℕ, (7 + 10 + 8 + 11 + 9 + 7 + x) / 7 = 9 ∧ x ≥ 10 := by
  sorry

end Theresa_helper_hours_l20_20563


namespace googoo_total_buttons_l20_20548

noncomputable def button_count_shirt_1 : ℕ := 3
noncomputable def button_count_shirt_2 : ℕ := 5
noncomputable def quantity_shirt_1 : ℕ := 200
noncomputable def quantity_shirt_2 : ℕ := 200

theorem googoo_total_buttons :
  (quantity_shirt_1 * button_count_shirt_1) + (quantity_shirt_2 * button_count_shirt_2) = 1600 := by
  sorry

end googoo_total_buttons_l20_20548


namespace hilary_regular_toenails_in_jar_l20_20376

-- Conditions
def jar_capacity : Nat := 100
def big_toenail_size : Nat := 2
def num_big_toenails : Nat := 20
def remaining_regular_toenails_space : Nat := 20

-- Question & Answer
theorem hilary_regular_toenails_in_jar : 
  (jar_capacity - remaining_regular_toenails_space - (num_big_toenails * big_toenail_size)) = 40 :=
by
  sorry

end hilary_regular_toenails_in_jar_l20_20376


namespace smallest_b_in_ap_l20_20627

-- Definition of an arithmetic progression
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

-- Problem statement in Lean
theorem smallest_b_in_ap (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h_ap : is_arithmetic_progression a b c) 
  (h_prod : a * b * c = 216) : 
  b ≥ 6 :=
by
  sorry

end smallest_b_in_ap_l20_20627


namespace sufficient_but_not_necessary_l20_20148

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1) : a^2 > a :=
by
  sorry

end sufficient_but_not_necessary_l20_20148


namespace Alan_age_is_29_l20_20385

/-- Alan and Chris ages problem -/
theorem Alan_age_is_29
    (A C : ℕ)
    (h1 : A + C = 52)
    (h2 : C = A / 3 + 2 * (A - C)) :
    A = 29 :=
by
  sorry

end Alan_age_is_29_l20_20385


namespace triangle_angle_split_l20_20394

-- Conditions
variables (A B C C1 C2 : ℝ)
-- Axioms/Assumptions
axiom angle_order : A < B
axiom angle_partition : A + C1 = 90 ∧ B + C2 = 90

-- The theorem to prove
theorem triangle_angle_split : C1 - C2 = B - A :=
by {
  sorry
}

end triangle_angle_split_l20_20394


namespace last_three_digits_of_8_pow_105_l20_20289

theorem last_three_digits_of_8_pow_105 : (8 ^ 105) % 1000 = 992 :=
by
  sorry

end last_three_digits_of_8_pow_105_l20_20289


namespace problem_part1_problem_part2_problem_part3_l20_20981

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then S n else S n - S (n - 1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (h : n > 0) : n > 0 → a_n n = 4 * n - 1 := by
  sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : n > 0 → b_n n = 2^(n - 1) := by
  sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : n > 0 → T_n n = (4 * n - 5) * 2^n + 5 := by
  sorry

end problem_part1_problem_part2_problem_part3_l20_20981


namespace bags_of_cookies_l20_20700

theorem bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) : total_cookies / cookies_per_bag = 3 :=
by
  sorry

end bags_of_cookies_l20_20700


namespace pyramid_volume_l20_20025

theorem pyramid_volume
  (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5)
  (angle_lateral : ℝ) (h4 : angle_lateral = 45) :
  ∃ (V : ℝ), V = 6 :=
by
  -- the proof steps would be included here
  sorry

end pyramid_volume_l20_20025


namespace annika_total_east_hike_distance_l20_20517

def annika_flat_rate : ℝ := 10 -- minutes per kilometer on flat terrain
def annika_initial_distance : ℝ := 2.75 -- kilometers already hiked east
def total_time : ℝ := 45 -- minutes
def uphill_rate : ℝ := 15 -- minutes per kilometer on uphill
def downhill_rate : ℝ := 5 -- minutes per kilometer on downhill
def uphill_distance : ℝ := 0.5 -- kilometer of uphill section
def downhill_distance : ℝ := 0.5 -- kilometer of downhill section

theorem annika_total_east_hike_distance :
  let total_uphill_time := uphill_distance * uphill_rate
  let total_downhill_time := downhill_distance * downhill_rate
  let time_for_uphill_and_downhill := total_uphill_time + total_downhill_time
  let time_available_for_outward_hike := total_time / 2
  let remaining_time_after_up_down := time_available_for_outward_hike - time_for_uphill_and_downhill
  let additional_flat_distance := remaining_time_after_up_down / annika_flat_rate
  (annika_initial_distance + additional_flat_distance) = 4 :=
by
  sorry

end annika_total_east_hike_distance_l20_20517


namespace minimum_value_of_reciprocals_l20_20048

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : a - b = 1) :
  (1 / a) + (1 / b) ≥ 4 :=
sorry

end minimum_value_of_reciprocals_l20_20048


namespace min_value_xy_expression_l20_20448

theorem min_value_xy_expression (x y : ℝ) : ∃ c : ℝ, (∀ x y : ℝ, (xy - 1)^2 + (x + y)^2 ≥ c) ∧ c = 1 :=
by {
  -- Placeholder for proof
  sorry
}

end min_value_xy_expression_l20_20448


namespace determine_contents_l20_20481

inductive Color
| White
| Black

open Color

-- Definitions of the mislabeled boxes
def mislabeled (box : Nat → List Color) : Prop :=
  ¬ (box 1 = [Black, Black] ∧ box 2 = [Black, White]
     ∧ box 3 = [White, White])

-- Draw a ball from a box revealing its content
def draw_ball (box : Nat → List Color) (i : Nat) (c : Color) : Prop :=
  c ∈ box i

-- theorem statement
theorem determine_contents (box : Nat → List Color) (c : Color) (h : draw_ball box 3 c) (hl : mislabeled box) :
  (c = White → box 3 = [White, White] ∧ box 2 = [Black, White] ∧ box 1 = [Black, Black]) ∧
  (c = Black → box 3 = [Black, Black] ∧ box 2 = [Black, White] ∧ box 1 = [White, White]) :=
by
  sorry

end determine_contents_l20_20481


namespace rectangle_dimensions_l20_20951

-- Define the known shapes and their dimensions
def square (s : ℝ) : ℝ := s^2
def rectangle1 : ℝ := 10 * 24
def rectangle2 (a b : ℝ) : ℝ := a * b

-- The total area must match the area of a square of side length 24 cm
def total_area (s a b : ℝ) : ℝ := (2 * square s) + rectangle1 + rectangle2 a b

-- The problem statement
theorem rectangle_dimensions
  (s a b : ℝ)
  (h0 : a ∈ [2, 19, 34, 34, 14, 14, 24])
  (h1 : b ∈ [24, 17.68, 10, 44, 24, 17, 38])
  : (total_area s a b = 24^2) :=
by
  sorry

end rectangle_dimensions_l20_20951


namespace triangle_circle_fill_l20_20351

theorem triangle_circle_fill (A B C D : ℕ) : 
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9) →
  (B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9) →
  (C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9) →
  (D = 6 ∨ D = 7 ∨ D = 8 ∨ D = 9) →
  (A + B + 1 + 8 =  A + 4 + 3 + 7) →  (D + 4 + 2 + 5 = 5 + 1 + 8 + B) →
  (5 + 1 + 8 + 6 = 5 + C + 7 + 4 ) →
  (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) := by
  sorry

end triangle_circle_fill_l20_20351


namespace fourth_number_ninth_row_eq_206_l20_20717

-- Define the first number in a given row
def first_number_in_row (i : Nat) : Nat :=
  2 + 4 * 6 * (i - 1)

-- Define the number in the j-th position in the i-th row
def number_in_row (i j : Nat) : Nat :=
  first_number_in_row i + 4 * (j - 1)

-- Define the 9th row and fourth number in it
def fourth_number_ninth_row : Nat :=
  number_in_row 9 4

-- The theorem to prove the fourth number in the 9th row is 206
theorem fourth_number_ninth_row_eq_206 : fourth_number_ninth_row = 206 := by
  sorry

end fourth_number_ninth_row_eq_206_l20_20717


namespace g_ln_1_over_2017_l20_20832

theorem g_ln_1_over_2017 (a : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (f g : ℝ → ℝ)
  (h_f_add : ∀ m n : ℝ, f (m + n) = f m + f n - 1)
  (h_g : ∀ x : ℝ, g x = f x + a^x / (a^x + 1))
  (h_g_ln_2017 : g (Real.log 2017) = 2018) :
  g (Real.log (1 / 2017)) = -2015 :=
sorry

end g_ln_1_over_2017_l20_20832


namespace find_a5_and_sum_l20_20353

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) > a n

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
is_geometric_sequence a q ∧ is_increasing_sequence a ∧ a 2 = 3 ∧ a 4 - a 3 = 18

-- Theorem to prove
theorem find_a5_and_sum {a : ℕ → ℝ} {q : ℝ} (h : given_conditions a q) :
  a 5 = 81 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
by
  -- Placeholder for the actual proof
  sorry

end find_a5_and_sum_l20_20353


namespace investment_Q_correct_l20_20178

-- Define the investments of P and Q
def investment_P : ℝ := 40000
def investment_Q : ℝ := 60000

-- Define the profit share ratio
def profit_ratio_PQ : ℝ × ℝ := (2, 3)

-- State the theorem to prove
theorem investment_Q_correct :
  (investment_P / investment_Q = (profit_ratio_PQ.1 / profit_ratio_PQ.2)) → 
  investment_Q = 60000 := 
by 
  sorry

end investment_Q_correct_l20_20178


namespace no_intersect_x_axis_intersection_points_m_minus3_l20_20153

-- Define the quadratic function y = x^2 - 6x + 2m - 1
def quadratic_function (x m : ℝ) : ℝ := x^2 - 6 * x + 2 * m - 1

-- Theorem for Question 1: The function does not intersect the x-axis if and only if m > 5
theorem no_intersect_x_axis (m : ℝ) : (∀ x : ℝ, quadratic_function x m ≠ 0) ↔ m > 5 := sorry

-- Specific case when m = -3
def quadratic_function_m_minus3 (x : ℝ) : ℝ := x^2 - 6 * x - 7

-- Theorem for Question 2: Intersection points with coordinate axes for m = -3
theorem intersection_points_m_minus3 :
  ((∃ x : ℝ, quadratic_function_m_minus3 x = 0 ∧ (x = -1 ∨ x = 7)) ∧
   quadratic_function_m_minus3 0 = -7) := sorry

end no_intersect_x_axis_intersection_points_m_minus3_l20_20153


namespace ratio_of_pieces_l20_20622

theorem ratio_of_pieces (total_length : ℝ) (short_piece : ℝ) (total_length_eq : total_length = 70) (short_piece_eq : short_piece = 27.999999999999993) :
  let long_piece := total_length - short_piece
  let ratio := short_piece / long_piece
  ratio = 2 / 3 :=
by
  sorry

end ratio_of_pieces_l20_20622


namespace housewife_money_left_l20_20037

theorem housewife_money_left (total : ℕ) (spent_fraction : ℚ) (spent : ℕ) (left : ℕ) :
  total = 150 → spent_fraction = 2 / 3 → spent = spent_fraction * total → left = total - spent → left = 50 :=
by
  intros
  sorry

end housewife_money_left_l20_20037


namespace possible_values_of_k_l20_20688

theorem possible_values_of_k (k : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, x > 0 → k * x > 0) ∧ (∀ x : ℝ, x < 0 → k * x > 0) → k > 0 :=
by
  sorry

end possible_values_of_k_l20_20688


namespace intersection_M_N_l20_20967

noncomputable def M : Set ℝ := { x | -1 < x ∧ x < 3 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.log (x - x^2) }
noncomputable def intersection (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_M_N : intersection M N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l20_20967


namespace remainder_n_plus_2023_mod_7_l20_20260

theorem remainder_n_plus_2023_mod_7 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 :=
by
  sorry

end remainder_n_plus_2023_mod_7_l20_20260


namespace slope_of_line_through_A_B_l20_20303

theorem slope_of_line_through_A_B :
  let A := (2, 1)
  let B := (-1, 3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = -2/3 :=
by
  have A_x : Int := 2
  have A_y : Int := 1
  have B_x : Int := -1
  have B_y : Int := 3
  sorry

end slope_of_line_through_A_B_l20_20303


namespace sum_in_range_l20_20768

theorem sum_in_range :
  let a := (27 : ℚ) / 8
  let b := (22 : ℚ) / 5
  let c := (67 : ℚ) / 11
  13 < a + b + c ∧ a + b + c < 14 :=
by
  sorry

end sum_in_range_l20_20768


namespace problem1_problem2_l20_20131

open Real

theorem problem1 : sin (420 * π / 180) * cos (330 * π / 180) + sin (-690 * π / 180) * cos (-660 * π / 180) = 1 := by
  sorry

theorem problem2 (α : ℝ) : 
  (sin (π / 2 + α) * cos (π / 2 - α) / cos (π + α)) + 
  (sin (π - α) * cos (π / 2 + α) / sin (π + α)) = 0 := by
  sorry

end problem1_problem2_l20_20131


namespace correct_operation_l20_20470

theorem correct_operation (a b : ℝ) : 
  (3 * Real.sqrt 7 + 7 * Real.sqrt 3 ≠ 10 * Real.sqrt 10) ∧ 
  (Real.sqrt (2 * a) * Real.sqrt (3) * a = Real.sqrt (6) * a) ∧ 
  (Real.sqrt a - Real.sqrt b ≠ Real.sqrt (a - b)) ∧ 
  (Real.sqrt (20 / 45) ≠ 4 / 9) :=
by
  sorry

end correct_operation_l20_20470


namespace find_5_minus_c_l20_20396

theorem find_5_minus_c (c d : ℤ) (h₁ : 5 + c = 6 - d) (h₂ : 3 + d = 8 + c) : 5 - c = 7 := by
  sorry

end find_5_minus_c_l20_20396


namespace sequence_contains_at_most_one_square_l20_20281

theorem sequence_contains_at_most_one_square 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∀ m n, (m ≠ n) → ¬ (∃ k, a m = k^2 ∧ a n = k^2) :=
sorry

end sequence_contains_at_most_one_square_l20_20281


namespace slices_per_birthday_l20_20331

-- Define the conditions: 
-- k is the age, the number of candles, starting from 3.
variable (k : ℕ) (h : k ≥ 3)

-- Define the function for the number of triangular slices
def number_of_slices (k : ℕ) : ℕ := 2 * k - 5

-- State the theorem to prove that the number of slices is 2k - 5
theorem slices_per_birthday (k : ℕ) (h : k ≥ 3) : 
    number_of_slices k = 2 * k - 5 := 
by
  sorry

end slices_per_birthday_l20_20331


namespace complement_of_union_l20_20713

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2}) (hN : N = {2, 3}) :
  (U \ (M ∪ N)) = {4} :=
by
  sorry

end complement_of_union_l20_20713


namespace tv_price_with_tax_l20_20117

-- Define the original price of the TV
def originalPrice : ℝ := 1700

-- Define the value-added tax rate
def taxRate : ℝ := 0.15

-- Calculate the total price including tax
theorem tv_price_with_tax : originalPrice * (1 + taxRate) = 1955 :=
by
  sorry

end tv_price_with_tax_l20_20117


namespace sector_area_l20_20019

theorem sector_area (theta r : ℝ) (h1 : theta = 2 * Real.pi / 3) (h2 : r = 2) :
  (1 / 2 * r ^ 2 * theta) = 4 * Real.pi / 3 := by
  sorry

end sector_area_l20_20019


namespace age_difference_l20_20678

theorem age_difference (A B C : ℕ) (h1 : B = 20) (h2 : C = B / 2) (h3 : A + B + C = 52) : A - B = 2 := by
  sorry

end age_difference_l20_20678


namespace triangles_from_decagon_l20_20512

theorem triangles_from_decagon (vertices : Fin 10 → Prop) 
  (h : ∀ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c → Prop) :
  ∃ triangles : ℕ, triangles = 120 :=
by
  sorry

end triangles_from_decagon_l20_20512


namespace erased_number_is_30_l20_20046

-- Definitions based on conditions
def consecutiveNumbers (start n : ℕ) : List ℕ :=
  List.range' start n

def erase (l : List ℕ) (x : ℕ) : List ℕ :=
  List.filter (λ y => y ≠ x) l

def average (l : List ℕ) : ℚ :=
  l.sum / l.length

-- Statement to prove
theorem erased_number_is_30 :
  ∃ n x, average (erase (consecutiveNumbers 11 n) x) = 23 ∧ x = 30 := by
  sorry

end erased_number_is_30_l20_20046


namespace coefficient_a2_in_expansion_l20_20554

theorem coefficient_a2_in_expansion:
  let a := (x - 1)^4
  let expansion := a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4
  a2 = 6 :=
by
  sorry

end coefficient_a2_in_expansion_l20_20554


namespace circle_equation_l20_20969

theorem circle_equation (x y : ℝ) :
    (x - 1) ^ 2 + (y - 1) ^ 2 = 1 ↔ (∃ (C : ℝ × ℝ), C = (1, 1) ∧ ∃ (r : ℝ), r = 1 ∧ (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2) :=
by
  sorry

end circle_equation_l20_20969


namespace vinegar_ratio_to_total_capacity_l20_20084

theorem vinegar_ratio_to_total_capacity (bowl_capacity : ℝ) (oil_fraction : ℝ) 
  (oil_density : ℝ) (vinegar_density : ℝ) (total_weight : ℝ) :
  bowl_capacity = 150 ∧ oil_fraction = 2/3 ∧ oil_density = 5 ∧ vinegar_density = 4 ∧ total_weight = 700 →
  (total_weight - (bowl_capacity * oil_fraction * oil_density)) / vinegar_density / bowl_capacity = 1/3 :=
by
  sorry

end vinegar_ratio_to_total_capacity_l20_20084


namespace larger_square_side_length_l20_20915

theorem larger_square_side_length :
  ∃ (a : ℕ), ∃ (b : ℕ), a^2 = b^2 + 2001 ∧ (a = 1001 ∨ a = 335 ∨ a = 55 ∨ a = 49) :=
by
  sorry

end larger_square_side_length_l20_20915


namespace find_radius_l20_20906

theorem find_radius (a : ℝ) :
  (∃ (x y : ℝ), (x + 2) ^ 2 + (y - 2) ^ 2 = a ∧ x + y + 2 = 0) ∧
  (∃ (l : ℝ), l = 6 ∧ 2 * Real.sqrt (a - 2) = l) →
  a = 11 :=
by
  sorry

end find_radius_l20_20906


namespace students_per_group_l20_20192

-- Define the conditions:
def total_students : ℕ := 120
def not_picked_students : ℕ := 22
def groups : ℕ := 14

-- Calculate the picked students:
def picked_students : ℕ := total_students - not_picked_students

-- Statement of the problem:
theorem students_per_group : picked_students / groups = 7 :=
  by sorry

end students_per_group_l20_20192


namespace not_sophomores_percentage_l20_20632

theorem not_sophomores_percentage (total_students : ℕ)
    (juniors_percentage : ℚ) (juniors : ℕ)
    (seniors : ℕ) (freshmen sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : juniors_percentage = 0.22)
    (h3 : juniors = juniors_percentage * total_students)
    (h4 : seniors = 160)
    (h5 : freshmen = sophomores + 48)
    (h6 : freshmen + sophomores + juniors + seniors = total_students) :
    ((total_students - sophomores : ℚ) / total_students) * 100 = 74 := by
  sorry

end not_sophomores_percentage_l20_20632


namespace m_not_equal_n_possible_l20_20384

-- Define the touching relation on an infinite chessboard
structure Chessboard :=
(colored_square : ℤ × ℤ → Prop)
(touches : ℤ × ℤ → ℤ × ℤ → Prop)

-- Define the properties
def colors_square (board : Chessboard) : Prop :=
∃ i j : ℤ, board.colored_square (i, j) ∧ board.colored_square (i + 1, j + 1)

def black_square_touches_m_black_squares (board : Chessboard) (m : ℕ) : Prop :=
∀ i j : ℤ, board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly m black squares are touched

def white_square_touches_n_white_squares (board : Chessboard) (n : ℕ) : Prop :=
∀ i j : ℤ, ¬board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → ¬board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → ¬board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → ¬board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → ¬board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly n white squares are touched

theorem m_not_equal_n_possible (board : Chessboard) (m n : ℕ) :
colors_square board →
black_square_touches_m_black_squares board m →
white_square_touches_n_white_squares board n →
m ≠ n :=
by {
    sorry
}

end m_not_equal_n_possible_l20_20384


namespace cistern_water_breadth_l20_20894

theorem cistern_water_breadth (length width total_area : ℝ) (h : ℝ) 
  (h_length : length = 10) 
  (h_width : width = 6) 
  (h_area : total_area = 103.2) : 
  (60 + 20*h + 12*h = total_area) → h = 1.35 :=
by
  intros
  sorry

end cistern_water_breadth_l20_20894


namespace number_of_birds_flew_up_correct_l20_20583

def initial_number_of_birds : ℕ := 29
def final_number_of_birds : ℕ := 42
def number_of_birds_flew_up : ℕ := final_number_of_birds - initial_number_of_birds

theorem number_of_birds_flew_up_correct :
  number_of_birds_flew_up = 13 := sorry

end number_of_birds_flew_up_correct_l20_20583


namespace travel_time_reduction_impossible_proof_l20_20586

noncomputable def travel_time_reduction_impossible : Prop :=
  ∀ (x : ℝ), x > 60 → ¬ (1 / x * 60 = 1 - 1)

theorem travel_time_reduction_impossible_proof : travel_time_reduction_impossible :=
sorry

end travel_time_reduction_impossible_proof_l20_20586


namespace equation_squares_l20_20556

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l20_20556


namespace arithmetic_sequence_a5_value_l20_20877

variable {a_n : ℕ → ℝ}

theorem arithmetic_sequence_a5_value
  (h : a_n 2 + a_n 8 = 15 - a_n 5) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l20_20877


namespace part1_part2_part3_l20_20588

-- Conditions
def A : Set ℝ := { x : ℝ | 2 < x ∧ x < 6 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m }

-- Proof statements
theorem part1 : A ∪ B 2 = { x : ℝ | 2 < x ∧ x < 6 } := by
  sorry

theorem part2 (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) → m ≤ 3 := by
  sorry

theorem part3 (m : ℝ) : (∃ x, x ∈ B m) ∧ (∀ x, x ∉ A ∩ B m) → m ≥ 5 := by
  sorry

end part1_part2_part3_l20_20588


namespace find_numerator_l20_20589

theorem find_numerator (n : ℕ) : 
  (n : ℚ) / 22 = 9545 / 10000 → 
  n = 9545 * 22 / 10000 :=
by sorry

end find_numerator_l20_20589


namespace cards_distribution_l20_20939

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ) (people_with_extra_cards : ℕ) (people_with_fewer_cards : ℕ) :
  total_cards = 100 →
  total_people = 15 →
  total_cards / total_people = cards_per_person →
  total_cards % total_people = extra_cards →
  people_with_extra_cards = extra_cards →
  people_with_fewer_cards = total_people - people_with_extra_cards →
  people_with_fewer_cards = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cards_distribution_l20_20939


namespace simplify_fraction_l20_20782

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x + 1/y ≠ 0) (h2 : y + 1/x ≠ 0) : 
  (x + 1/y) / (y + 1/x) = x / y :=
sorry

end simplify_fraction_l20_20782


namespace seventh_grader_count_l20_20438

variables {x n : ℝ}

noncomputable def number_of_seventh_graders (x n : ℝ) :=
  10 * x = 10 * x ∧  -- Condition 1
  4.5 * n = 4.5 * n ∧  -- Condition 2
  11 * x = 11 * x ∧  -- Condition 3
  5.5 * n = 5.5 * n ∧  -- Condition 4
  5.5 * n = (11 * x * (11 * x - 1)) / 2 ∧  -- Condition 5
  n = x * (11 * x - 1)  -- Condition 6

theorem seventh_grader_count (x n : ℝ) (h : number_of_seventh_graders x n) : x = 1 :=
  sorry

end seventh_grader_count_l20_20438


namespace find_m_l20_20808

noncomputable def g (d e f x : ℤ) : ℤ := d * x * x + e * x + f

theorem find_m (d e f m : ℤ) (h₁ : g d e f 2 = 0)
    (h₂ : 60 < g d e f 6 ∧ g d e f 6 < 70) 
    (h₃ : 80 < g d e f 9 ∧ g d e f 9 < 90)
    (h₄ : 10000 * m < g d e f 100 ∧ g d e f 100 < 10000 * (m + 1)) :
  m = -1 :=
sorry

end find_m_l20_20808


namespace carla_sheep_l20_20139

theorem carla_sheep (T : ℝ) (pen_sheep wilderness_sheep : ℝ) 
(h1: 0.90 * T = 81) (h2: pen_sheep = 81) 
(h3: wilderness_sheep = 0.10 * T) : wilderness_sheep = 9 :=
sorry

end carla_sheep_l20_20139


namespace problem_solution_l20_20146

theorem problem_solution (a b c d : ℝ) 
  (h1 : 3 * a + 2 * b + 4 * c + 8 * d = 40)
  (h2 : 4 * (d + c) = b)
  (h3 : 2 * b + 2 * c = a)
  (h4 : c + 1 = d) :
  a * b * c * d = 0 :=
sorry

end problem_solution_l20_20146


namespace positive_integers_of_inequality_l20_20168

theorem positive_integers_of_inequality (x : ℕ) (h : 9 - 3 * x > 0) : x = 1 ∨ x = 2 :=
sorry

end positive_integers_of_inequality_l20_20168


namespace abs_expr_evaluation_l20_20730

theorem abs_expr_evaluation : abs (abs (-abs (-1 + 2) - 2) + 3) = 6 := by
  sorry

end abs_expr_evaluation_l20_20730


namespace inequality_proof_l20_20160

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b :=
by
  sorry

end inequality_proof_l20_20160


namespace max_rectangle_area_max_rectangle_area_exists_l20_20837

theorem max_rectangle_area (l w : ℕ) (h : l + w = 20) : l * w ≤ 100 :=
by sorry

-- Alternatively, to also show the existence of the maximum value.
theorem max_rectangle_area_exists : ∃ l w : ℕ, l + w = 20 ∧ l * w = 100 :=
by sorry

end max_rectangle_area_max_rectangle_area_exists_l20_20837


namespace rate_of_work_l20_20066

theorem rate_of_work (A : ℝ) (h1: 0 < A) (h_eq : 1 / A + 1 / 6 = 1 / 2) : A = 3 := sorry

end rate_of_work_l20_20066


namespace area_of_fourth_rectangle_l20_20420

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle_l20_20420


namespace sports_day_results_l20_20371

-- Conditions and questions
variables (a b c : ℕ)
variables (class1_score class2_score class3_score class4_score : ℕ)

-- Conditions given in the problem
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom a_gt_b_gt_c : a > b ∧ b > c
axiom no_ties : (class1_score ≠ class2_score) ∧ (class2_score ≠ class3_score) ∧ (class3_score ≠ class4_score) ∧ (class1_score ≠ class3_score) ∧ (class1_score ≠ class4_score) ∧ (class2_score ≠ class4_score)
axiom class_scores : class1_score + class2_score + class3_score + class4_score = 40

-- To prove
theorem sports_day_results : a + b + c = 8 ∧ a = 5 :=
by
  sorry

end sports_day_results_l20_20371


namespace consecutive_page_numbers_sum_l20_20709

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) = 19881) : n + (n + 1) = 283 :=
sorry

end consecutive_page_numbers_sum_l20_20709


namespace ratio_XY_7_l20_20355

variable (Z : ℕ)
variable (population_Z : ℕ := Z)
variable (population_Y : ℕ := 2 * Z)
variable (population_X : ℕ := 14 * Z)

theorem ratio_XY_7 :
  population_X / population_Y = 7 := by
  sorry

end ratio_XY_7_l20_20355


namespace compare_negatives_l20_20444

theorem compare_negatives : -4 < -2.1 := 
sorry

end compare_negatives_l20_20444


namespace f_comp_g_eq_g_comp_f_has_solution_l20_20934

variable {R : Type*} [Field R]

def f (a b x : R) : R := a * x + b
def g (c d x : R) : R := c * x ^ 2 + d

theorem f_comp_g_eq_g_comp_f_has_solution (a b c d : R) :
  (∃ x : R, f a b (g c d x) = g c d (f a b x)) ↔ (c = 0 ∨ a * b = 0) ∧ (a * d - c * b ^ 2 + b - d = 0) := by
  sorry

end f_comp_g_eq_g_comp_f_has_solution_l20_20934


namespace width_of_door_l20_20986

theorem width_of_door 
  (L W H : ℕ) 
  (cost_per_sq_ft : ℕ) 
  (door_height window_height window_width : ℕ) 
  (num_windows total_cost : ℕ) 
  (door_width : ℕ) 
  (total_wall_area area_door area_windows area_to_whitewash : ℕ)
  (raw_area_door raw_area_windows total_walls_to_paint : ℕ) 
  (cost_per_sq_ft_eq : cost_per_sq_ft = 9)
  (total_cost_eq : total_cost = 8154)
  (room_dimensions_eq : L = 25 ∧ W = 15 ∧ H = 12)
  (door_dimensions_eq : door_height = 6)
  (window_dimensions_eq : window_height = 3 ∧ window_width = 4)
  (num_windows_eq : num_windows = 3)
  (total_wall_area_eq : total_wall_area = 2 * (L * H) + 2 * (W * H))
  (raw_area_door_eq : raw_area_door = door_height * door_width)
  (raw_area_windows_eq : raw_area_windows = num_windows * (window_width * window_height))
  (total_walls_to_paint_eq : total_walls_to_paint = total_wall_area - raw_area_door - raw_area_windows)
  (area_to_whitewash_eq : area_to_whitewash = 924 - 6 * door_width)
  (total_cost_eq_calc : total_cost = area_to_whitewash * cost_per_sq_ft) :
  door_width = 3 := sorry

end width_of_door_l20_20986


namespace pell_solution_unique_l20_20310

theorem pell_solution_unique 
  (x_0 y_0 x y : ℤ) 
  (h_fundamental : x_0^2 - 2003 * y_0^2 = 1)
  (h_pos_x : 0 < x) 
  (h_pos_y : 0 < y)
  (h_prime_div : ∀ p, Prime p → p ∣ x → p ∣ x_0) :
  x^2 - 2003 * y^2 = 1 → (x, y) = (x_0, y_0) :=
sorry

end pell_solution_unique_l20_20310


namespace find_x_value_l20_20324

noncomputable def log (a b: ℝ): ℝ := Real.log a / Real.log b

theorem find_x_value (a n : ℝ) (t y: ℝ):
  1 < a →
  1 < t →
  y = 8 →
  log n (a^t) - 3 * log a (a^t) * log y 8 = 3 →
  x = a^t →
  x = a^2 :=
by
  sorry

end find_x_value_l20_20324


namespace total_eggs_collected_l20_20482

-- Define the variables given in the conditions
def Benjamin_eggs := 6
def Carla_eggs := 3 * Benjamin_eggs
def Trisha_eggs := Benjamin_eggs - 4

-- State the theorem using the conditions and correct answer in the equivalent proof problem
theorem total_eggs_collected :
  Benjamin_eggs + Carla_eggs + Trisha_eggs = 26 := by
  -- Proof goes here.
  sorry

end total_eggs_collected_l20_20482


namespace total_cupcakes_l20_20108

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (h1 : children = 8) (h2 : cupcakes_per_child = 12) : children * cupcakes_per_child = 96 :=
by
  sorry

end total_cupcakes_l20_20108


namespace sum_of_fractions_l20_20204

theorem sum_of_fractions : (1 / 6) + (2 / 9) + (1 / 3) = 13 / 18 := by
  sorry

end sum_of_fractions_l20_20204


namespace product_gcd_lcm_150_90_l20_20441

theorem product_gcd_lcm_150_90 (a b : ℕ) (h1 : a = 150) (h2 : b = 90): Nat.gcd a b * Nat.lcm a b = a * b := by
  rw [h1, h2]
  sorry

end product_gcd_lcm_150_90_l20_20441


namespace fraction_of_number_l20_20518

theorem fraction_of_number (x f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 :=
by
  sorry

end fraction_of_number_l20_20518


namespace sum_of_squares_diagonals_cyclic_quadrilateral_l20_20957

theorem sum_of_squares_diagonals_cyclic_quadrilateral 
(a b c d : ℝ) (α : ℝ) 
(hc : c^2 = a^2 + b^2 + 2 * a * b * Real.cos α)
(hd : d^2 = a^2 + b^2 - 2 * a * b * Real.cos α) :
  c^2 + d^2 = 2 * a^2 + 2 * b^2 :=
by
  sorry

end sum_of_squares_diagonals_cyclic_quadrilateral_l20_20957


namespace rational_mul_example_l20_20225

theorem rational_mul_example : ((19 + 15 / 16) * (-8)) = (-159 - 1 / 2) :=
by
  sorry

end rational_mul_example_l20_20225


namespace find_a_l20_20214

open Set Real

-- Defining sets A and B, and the condition A ∩ B = {3}
def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- Mathematically equivalent proof statement
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
  sorry

end find_a_l20_20214


namespace sin_double_angle_l20_20130

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end sin_double_angle_l20_20130


namespace elijah_total_cards_l20_20502

-- Define the conditions
def num_decks : ℕ := 6
def cards_per_deck : ℕ := 52

-- The main statement that we need to prove
theorem elijah_total_cards : num_decks * cards_per_deck = 312 := by
  -- We skip the proof
  sorry

end elijah_total_cards_l20_20502


namespace total_widgets_sold_after_15_days_l20_20008

def widgets_sold_day_n (n : ℕ) : ℕ :=
  2 + (n - 1) * 3

def sum_of_widgets (n : ℕ) : ℕ :=
  n * (2 + widgets_sold_day_n n) / 2

theorem total_widgets_sold_after_15_days : 
  sum_of_widgets 15 = 345 :=
by
  -- Prove the arithmetic sequence properties and sum.
  sorry

end total_widgets_sold_after_15_days_l20_20008


namespace ratio_of_John_to_Mary_l20_20209

-- Definitions based on conditions
variable (J M T : ℕ)
variable (hT : T = 60)
variable (hJ : J = T / 2)
variable (hAvg : (J + M + T) / 3 = 35)

-- Statement to prove
theorem ratio_of_John_to_Mary : J / M = 2 := by
  -- Proof goes here
  sorry

end ratio_of_John_to_Mary_l20_20209


namespace ratio_of_x_to_y_l20_20834

theorem ratio_of_x_to_y (x y : ℝ) (R : ℝ) (h1 : x = R * y) (h2 : x - y = 0.909090909090909 * x) : R = 11 := by
  sorry

end ratio_of_x_to_y_l20_20834


namespace square_of_binomial_l20_20248

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l20_20248


namespace minimum_value_l20_20993

theorem minimum_value (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 1) : 
  (∃ (x : ℝ), x = a + 2*b) → (∃ (y : ℝ), y = 2*a + b) → 
  (∀ (x y : ℝ), x + y = 3 → (1/x + 4/y) ≥ 3) :=
by
  sorry

end minimum_value_l20_20993


namespace find_x_l20_20295

variables {K J : ℝ} {A B C A_star B_star C_star : Type*}

-- Define the triangles and areas
def triangle_area (K : ℝ) : Prop := K > 0

-- We know the fractions of segments in triangle
def segment_ratios (x : ℝ) : Prop :=
  0 < x ∧ x < 1 ∧
  ∀ (AA_star AB BB_star BC CC_star CA : ℝ),
    AA_star / AB = x ∧ BB_star / BC = x ∧ CC_star / CA = x

-- Area of the smaller inner triangle
def inner_triangle_area (x : ℝ) (K : ℝ) (J : ℝ) : Prop :=
  J = x * K

-- The theorem combining all to show x = 1/3
theorem find_x (x : ℝ) (K J : ℝ) (triangleAreaK : triangle_area K)
    (ratios : segment_ratios x)
    (innerArea : inner_triangle_area x K J) :
  x = 1 / 3 :=
by
  sorry

end find_x_l20_20295


namespace pure_gold_to_add_eq_46_67_l20_20326

-- Define the given conditions
variable (initial_alloy_weight : ℝ) (initial_gold_percentage : ℝ) (final_gold_percentage : ℝ)
variable (added_pure_gold : ℝ)

-- State the proof problem
theorem pure_gold_to_add_eq_46_67 :
  initial_alloy_weight = 20 ∧
  initial_gold_percentage = 0.50 ∧
  final_gold_percentage = 0.85 ∧
  (10 + added_pure_gold) / (20 + added_pure_gold) = 0.85 →
  added_pure_gold = 46.67 :=
by
  sorry

end pure_gold_to_add_eq_46_67_l20_20326


namespace time_to_cover_length_l20_20761

-- Definitions from conditions
def escalator_speed : Real := 15 -- ft/sec
def escalator_length : Real := 180 -- feet
def person_speed : Real := 3 -- ft/sec

-- Combined speed definition
def combined_speed : Real := escalator_speed + person_speed

-- Lean theorem statement proving the time taken
theorem time_to_cover_length : escalator_length / combined_speed = 10 := by
  sorry

end time_to_cover_length_l20_20761


namespace sqrt_six_lt_a_lt_cubic_two_l20_20728

theorem sqrt_six_lt_a_lt_cubic_two (a : ℝ) (h : a^5 - a^3 + a = 2) : (Real.sqrt 3)^6 < a ∧ a < 2^(1/3) :=
sorry

end sqrt_six_lt_a_lt_cubic_two_l20_20728


namespace find_probability_of_B_l20_20999

-- Define the conditions and the problem
def system_A_malfunction_prob := 1 / 10
def at_least_one_not_malfunction_prob := 49 / 50

/-- The probability that System B malfunctions given that 
  the probability of at least one system not malfunctioning 
  is 49/50 and the probability of System A malfunctioning is 1/10 -/
theorem find_probability_of_B (p : ℝ) 
  (h1 : system_A_malfunction_prob = 1 / 10) 
  (h2 : at_least_one_not_malfunction_prob = 49 / 50) 
  (h3 : 1 - (system_A_malfunction_prob * p) = at_least_one_not_malfunction_prob) : 
  p = 1 / 5 :=
sorry

end find_probability_of_B_l20_20999


namespace positive_difference_of_sums_l20_20483

def sum_first_n (n : Nat) : Nat := n * (n + 1) / 2

def sum_first_n_even (n : Nat) : Nat := 2 * sum_first_n n

def sum_first_n_odd (n : Nat) : Nat := n * n

theorem positive_difference_of_sums :
  let S1 := sum_first_n_even 25
  let S2 := sum_first_n_odd 20
  S1 - S2 = 250 := by
  sorry

end positive_difference_of_sums_l20_20483


namespace thief_speed_l20_20191

theorem thief_speed (v : ℝ) (hv : v > 0) : 
  let head_start_duration := (1/2 : ℝ)  -- 30 minutes, converted to hours
  let owner_speed := (75 : ℝ)  -- speed of owner in kmph
  let chase_duration := (2 : ℝ)  -- duration of the chase in hours
  let distance_by_owner := owner_speed * chase_duration  -- distance covered by the owner
  let total_distance_thief := head_start_duration * v + chase_duration * v  -- total distance covered by the thief
  distance_by_owner = 150 ->  -- given that owner covers 150 km
  total_distance_thief = 150  -- and so should the thief
  -> v = 60 := sorry

end thief_speed_l20_20191


namespace skill_position_players_waiting_l20_20298

def linemen_drink : ℕ := 8
def skill_position_player_drink : ℕ := 6
def num_linemen : ℕ := 12
def num_skill_position_players : ℕ := 10
def cooler_capacity : ℕ := 126

theorem skill_position_players_waiting :
  num_skill_position_players - (cooler_capacity - num_linemen * linemen_drink) / skill_position_player_drink = 5 :=
by
  -- Calculation is needed to be filled in here
  sorry

end skill_position_players_waiting_l20_20298


namespace tangent_intersect_x_axis_l20_20342

-- Defining the conditions based on the given problem
def radius1 : ℝ := 3
def center1 : ℝ × ℝ := (0, 0)

def radius2 : ℝ := 5
def center2 : ℝ × ℝ := (12, 0)

-- Stating what needs to be proved
theorem tangent_intersect_x_axis : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (∀ (x1 x2 : ℝ), 
    (x1 = x) ∧ 
    (x2 = 12 - x) ∧ 
    (radius1 / (center2.1 - x) = radius2 / x2) → 
    (x = 9 / 2)) := 
sorry

end tangent_intersect_x_axis_l20_20342


namespace tom_tim_typing_ratio_l20_20982

variable (T M : ℝ)

theorem tom_tim_typing_ratio (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
sorry

end tom_tim_typing_ratio_l20_20982


namespace horner_method_V3_correct_when_x_equals_2_l20_20835

-- Polynomial f(x)
noncomputable def f (x : ℝ) : ℝ :=
  2 * x^5 - 3 * x^3 + 2 * x^2 + x - 3

-- Horner's method for evaluating f(x)
noncomputable def V3 (x : ℝ) : ℝ :=
  (((((2 * x + 0) * x - 3) * x + 2) * x + 1) * x - 3)

-- Proof that V3 = 12 when x = 2
theorem horner_method_V3_correct_when_x_equals_2 : V3 2 = 12 := by
  sorry

end horner_method_V3_correct_when_x_equals_2_l20_20835


namespace joe_bought_books_l20_20292

theorem joe_bought_books (money_given : ℕ) (notebook_cost : ℕ) (num_notebooks : ℕ) (book_cost : ℕ) (leftover_money : ℕ) (total_spent := money_given - leftover_money) (spent_on_notebooks := num_notebooks * notebook_cost) (spent_on_books := total_spent - spent_on_notebooks) (num_books := spent_on_books / book_cost) : money_given = 56 → notebook_cost = 4 → num_notebooks = 7 → book_cost = 7 → leftover_money = 14 → num_books = 2 := by
  intros
  sorry

end joe_bought_books_l20_20292


namespace central_angle_of_sector_l20_20614

/-- The central angle of the sector obtained by unfolding the lateral surface of a cone with
    base radius 1 and slant height 2 is \(\pi\). -/
theorem central_angle_of_sector (r_base : ℝ) (r_slant : ℝ) (α : ℝ)
  (h1 : r_base = 1) (h2 : r_slant = 2) (h3 : 2 * π = α * r_slant) : α = π :=
by
  sorry

end central_angle_of_sector_l20_20614


namespace probability_not_overcoming_is_half_l20_20860

/-- Define the five elements. -/
inductive Element
| metal | wood | water | fire | earth

open Element

/-- Define the overcoming relation. -/
def overcomes : Element → Element → Prop
| metal, wood => true
| wood, earth => true
| earth, water => true
| water, fire => true
| fire, metal => true
| _, _ => false

/-- Define the probability calculation. -/
def probability_not_overcoming : ℚ :=
  let total_combinations := 10    -- C(5, 2)
  let overcoming_combinations := 5
  let not_overcoming_combinations := total_combinations - overcoming_combinations
  not_overcoming_combinations / total_combinations

/-- The proof problem statement. -/
theorem probability_not_overcoming_is_half : probability_not_overcoming = 1 / 2 :=
by
  sorry

end probability_not_overcoming_is_half_l20_20860


namespace dave_paid_4_more_than_doug_l20_20701

theorem dave_paid_4_more_than_doug :
  let slices := 8
  let plain_cost := 8
  let anchovy_additional_cost := 2
  let total_cost := plain_cost + anchovy_additional_cost
  let cost_per_slice := total_cost / slices
  let dave_slices := 5
  let doug_slices := slices - dave_slices
  -- Calculate payments
  let dave_payment := dave_slices * cost_per_slice
  let doug_payment := doug_slices * cost_per_slice
  dave_payment - doug_payment = 4 :=
by
  sorry

end dave_paid_4_more_than_doug_l20_20701


namespace number_of_customers_before_lunch_rush_l20_20173

-- Defining the total number of customers during the lunch rush
def total_customers_during_lunch_rush : ℕ := 49 + 2

-- Defining the number of additional customers during the lunch rush
def additional_customers : ℕ := 12

-- Target statement to prove
theorem number_of_customers_before_lunch_rush : total_customers_during_lunch_rush - additional_customers = 39 :=
  by sorry

end number_of_customers_before_lunch_rush_l20_20173


namespace negation_of_proposition_l20_20370

theorem negation_of_proposition (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (∃ x y z : ℝ, (x < 0) ∧ (y < 0) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (x ≠ y)) →
  ¬(∀ x y z : ℝ, (x < 0 ∨ y < 0 ∨ z < 0) → (x ≠ y → x ≠ z → y ≠ z → (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ (z = a ∨ z = b ∨ z = c))) :=
sorry

end negation_of_proposition_l20_20370


namespace largest_n_multiple_of_7_l20_20354

theorem largest_n_multiple_of_7 (n : ℕ) (h1 : n < 50000) (h2 : (5*(n-3)^5 - 3*n^2 + 20*n - 35) % 7 = 0) : n = 49999 :=
sorry

end largest_n_multiple_of_7_l20_20354


namespace complement_of_A_in_U_l20_20403

-- Define the universal set U as the set of integers
def U : Set ℤ := Set.univ

-- Define the set A as the set of odd integers
def A : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set ℤ := U \ A

-- State the equivalence to be proved
theorem complement_of_A_in_U :
  complement_A = {x : ℤ | ∃ k : ℤ, x = 2 * k} :=
by
  sorry

end complement_of_A_in_U_l20_20403


namespace total_food_consumed_l20_20169

theorem total_food_consumed (n1 n2 f1 f2 : ℕ) (h1 : n1 = 4000) (h2 : n2 = n1 - 500) (h3 : f1 = 10) (h4 : f2 = f1 - 2) : 
    n1 * f1 + n2 * f2 = 68000 := by 
  sorry

end total_food_consumed_l20_20169


namespace least_value_a_plus_b_l20_20551

theorem least_value_a_plus_b (a b : ℕ) (h : 20 / 19 = 1 + 1 / (1 + a / b)) : a + b = 19 :=
sorry

end least_value_a_plus_b_l20_20551


namespace intersection_y_axis_parabola_l20_20820

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end intersection_y_axis_parabola_l20_20820


namespace p_neither_sufficient_nor_necessary_l20_20560

theorem p_neither_sufficient_nor_necessary (x y : ℝ) :
  (x > 1 ∧ y > 1) ↔ ¬((x > 1 ∧ y > 1) → (x + y > 3)) ∧ ¬((x + y > 3) → (x > 1 ∧ y > 1)) :=
by
  sorry

end p_neither_sufficient_nor_necessary_l20_20560


namespace inequality_proof_l20_20072

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := 
by sorry

end inequality_proof_l20_20072


namespace factorization_correct_l20_20344

-- Define the given expression
def expression (a b : ℝ) : ℝ := 9 * a^2 * b - b

-- Define the factorized form
def factorized_form (a b : ℝ) : ℝ := b * (3 * a + 1) * (3 * a - 1)

-- Theorem stating that the factorization is correct
theorem factorization_correct (a b : ℝ) : expression a b = factorized_form a b := by
  sorry

end factorization_correct_l20_20344


namespace number_of_men_in_first_group_l20_20802

/-
Given the initial conditions:
1. Some men can color a 48 m long cloth in 2 days.
2. 6 men can color a 36 m long cloth in 1 day.

We need to prove that the number of men in the first group is equal to 9.
-/

theorem number_of_men_in_first_group (M : ℕ)
    (h1 : ∃ (x : ℕ), x * 48 = M * 2)
    (h2 : 6 * 36 = 36 * 1) :
    M = 9 :=
by
sorry

end number_of_men_in_first_group_l20_20802


namespace johns_photo_world_sitting_fee_l20_20572

variable (J : ℝ)

theorem johns_photo_world_sitting_fee
  (h1 : ∀ n : ℝ, n = 12 → 2.75 * n + J = 1.50 * n + 140) : J = 125 :=
by
  -- We will skip the proof since it is not required by the problem statement.
  sorry

end johns_photo_world_sitting_fee_l20_20572


namespace crease_length_l20_20499

theorem crease_length (A B C : ℝ) (h1 : A = 5) (h2 : B = 12) (h3 : C = 13) : ∃ D, D = 6.5 :=
by
  sorry

end crease_length_l20_20499


namespace monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l20_20538

noncomputable def f (x : ℝ) : ℝ := 3 * (Real.sin (2 * x + (Real.pi / 4)))

theorem monotonic_intervals_increasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + Real.pi / 8) → f x ≤ f y :=
sorry

theorem monotonic_intervals_decreasing (k : ℤ) :
  ∀ x y : ℝ, (k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ y ∧ y ≤ k * Real.pi + 5 * Real.pi / 8) → f x ≥ f y :=
sorry

theorem maximum_value (k : ℤ) :
  f (k * Real.pi + Real.pi / 8) = 3 :=
sorry

theorem minimum_value (k : ℤ) :
  f (k * Real.pi - 3 * Real.pi / 8) = -3 :=
sorry

end monotonic_intervals_increasing_monotonic_intervals_decreasing_maximum_value_minimum_value_l20_20538


namespace angle_of_inclination_l20_20317

theorem angle_of_inclination (t : ℝ) (x y : ℝ) :
  (x = 1 + t * (Real.sin (Real.pi / 6))) ∧ 
  (y = 2 + t * (Real.cos (Real.pi / 6))) →
  ∃ α : ℝ, α = Real.arctan (Real.sqrt 3) ∧ (0 ≤ α ∧ α < Real.pi) := 
by 
  sorry

end angle_of_inclination_l20_20317


namespace expression_independent_of_alpha_l20_20176

theorem expression_independent_of_alpha
  (α : Real) (n : ℤ) (h : α ≠ (n * (π / 2)) + (π / 12)) :
  (1 - 2 * Real.sin (α - (3 * π / 2))^2 + (Real.sqrt 3) * Real.cos (2 * α + (3 * π / 2))) /
  (Real.sin (π / 6 - 2 * α)) = -2 := 
sorry

end expression_independent_of_alpha_l20_20176


namespace closest_fraction_to_team_alpha_medals_l20_20054

theorem closest_fraction_to_team_alpha_medals :
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 5) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 6) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 7) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 9) := 
by
  sorry

end closest_fraction_to_team_alpha_medals_l20_20054


namespace james_profit_l20_20524

theorem james_profit
  (tickets_bought : ℕ)
  (cost_per_ticket : ℕ)
  (percentage_winning : ℕ)
  (winning_tickets_percentage_5dollars : ℕ)
  (grand_prize : ℕ)
  (average_other_prizes : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (winning_tickets : ℕ)
  (tickets_prize_5dollars : ℕ)
  (amount_won_5dollars : ℕ)
  (other_winning_tickets : ℕ)
  (other_tickets_prize : ℕ)
  (total_winning_amount : ℕ)
  (profit : ℕ) :

  tickets_bought = 200 →
  cost_per_ticket = 2 →
  percentage_winning = 20 →
  winning_tickets_percentage_5dollars = 80 →
  grand_prize = 5000 →
  average_other_prizes = 10 →
  total_tickets = tickets_bought →
  total_cost = total_tickets * cost_per_ticket →
  winning_tickets = (percentage_winning * total_tickets) / 100 →
  tickets_prize_5dollars = (winning_tickets_percentage_5dollars * winning_tickets) / 100 →
  amount_won_5dollars = tickets_prize_5dollars * 5 →
  other_winning_tickets = winning_tickets - 1 →
  other_tickets_prize = (other_winning_tickets - tickets_prize_5dollars) * average_other_prizes →
  total_winning_amount = amount_won_5dollars + grand_prize + other_tickets_prize →
  profit = total_winning_amount - total_cost →
  profit = 4830 := 
sorry

end james_profit_l20_20524


namespace general_formula_compare_Tn_l20_20099

open scoped BigOperators

-- Define the sequence {a_n} and its sum S_n
noncomputable def aSeq (n : ℕ) : ℕ := n + 1
noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range n, aSeq (k + 1)

-- Given condition
axiom given_condition (n : ℕ) : 2 * S n = (aSeq n - 1) * (aSeq n + 2)

-- Prove the general formula of the sequence
theorem general_formula (n : ℕ) : aSeq n = n + 1 :=
by
  sorry  -- proof

-- Define T_n sequence
noncomputable def T (n : ℕ) : ℕ := ∑ k in Finset.range n, (k - 1) * 2^k / (k * aSeq k)

-- Compare T_n with the given expression
theorem compare_Tn (n : ℕ) : 
  if n < 17 then T n < (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else if n = 17 then T n = (2^(n+1)*(18-n)-2*n-2)/(n+1)
  else T n > (2^(n+1)*(18-n)-2*n-2)/(n+1) :=
by
  sorry  -- proof

end general_formula_compare_Tn_l20_20099


namespace maximum_angle_B_in_triangle_l20_20577

theorem maximum_angle_B_in_triangle
  (A B C M : ℝ × ℝ)
  (hM : midpoint ℝ A B = M)
  (h_angle_MAC : ∃ angle_MAC : ℝ, angle_MAC = 15) :
  ∃ angle_B : ℝ, angle_B = 105 := 
by
  sorry

end maximum_angle_B_in_triangle_l20_20577


namespace xyz_problem_l20_20579

/-- Given x = 36^2 + 48^2 + 64^3 + 81^2, prove the following:
    - x is a multiple of 3. 
    - x is a multiple of 4.
    - x is a multiple of 9.
    - x is not a multiple of 16. 
-/
theorem xyz_problem (x : ℕ) (h_x : x = 36^2 + 48^2 + 64^3 + 81^2) :
  (x % 3 = 0) ∧ (x % 4 = 0) ∧ (x % 9 = 0) ∧ ¬(x % 16 = 0) := 
by
  have h1 : 36^2 = 1296 := by norm_num
  have h2 : 48^2 = 2304 := by norm_num
  have h3 : 64^3 = 262144 := by norm_num
  have h4 : 81^2 = 6561 := by norm_num
  have hx : x = 1296 + 2304 + 262144 + 6561 := by rw [h_x, h1, h2, h3, h4]
  sorry

end xyz_problem_l20_20579


namespace add_zero_eq_self_l20_20867

theorem add_zero_eq_self (n x : ℤ) (h : n + x = n) : x = 0 := 
sorry

end add_zero_eq_self_l20_20867


namespace tangent_line_eq_f_positive_find_a_l20_20599

noncomputable def f (x a : ℝ) : ℝ := 1 - (a * x^2) / (Real.exp x)
noncomputable def f' (x a : ℝ) : ℝ := (a * x * (x - 2)) / (Real.exp x)

-- Part 1: equation of tangent line
theorem tangent_line_eq (a : ℝ) (h1 : f' 1 a = 1) (hx : f 1 a = 2) : ∀ x, f 1 a + f' 1 a * (x - 1) = x + 1 :=
sorry

-- Part 2: f(x) > 0 for x > 0 when a = 1
theorem f_positive (x : ℝ) (h : x > 0) : f x 1 > 0 :=
sorry

-- Part 3: minimum value of f(x) is -3, find a
theorem find_a (a : ℝ) (h : ∀ x, f x a ≥ -3) : a = Real.exp 2 :=
sorry

end tangent_line_eq_f_positive_find_a_l20_20599


namespace definite_integral_abs_poly_l20_20465

theorem definite_integral_abs_poly :
  ∫ x in (-2 : ℝ)..(2 : ℝ), |x^2 - 2*x| = 8 :=
by
  sorry

end definite_integral_abs_poly_l20_20465


namespace coconut_transport_l20_20653

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l20_20653


namespace ways_to_start_writing_l20_20187

def ratio_of_pens_to_notebooks (pens notebooks : ℕ) : Prop := 
    pens * 4 = notebooks * 5

theorem ways_to_start_writing 
    (pens notebooks : ℕ) 
    (h_ratio : ratio_of_pens_to_notebooks pens notebooks) 
    (h_pens : pens = 50)
    (h_notebooks : notebooks = 40) : 
    ∃ ways : ℕ, ways = 40 :=
by
  sorry

end ways_to_start_writing_l20_20187


namespace sum_of_circle_areas_l20_20213

theorem sum_of_circle_areas 
    (r s t : ℝ)
    (h1 : r + s = 6)
    (h2 : r + t = 8)
    (h3 : s + t = 10) : 
    (π * r^2 + π * s^2 + π * t^2) = 36 * π := 
by
    sorry

end sum_of_circle_areas_l20_20213


namespace number_of_folds_l20_20544

theorem number_of_folds (n : ℕ) :
  (3 * (8 * 8)) / n = 48 → n = 4 :=
by
  sorry

end number_of_folds_l20_20544


namespace prod_mod7_eq_zero_l20_20472

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l20_20472


namespace analytic_expression_and_symmetry_l20_20074

noncomputable def f (A : ℝ) (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem analytic_expression_and_symmetry {A ω φ : ℝ}
  (hA : A > 0) 
  (hω : ω > 0)
  (h_period : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h_max : f A ω φ (1 / 3) = 2) :
  (f 2 π (π / 6) = fun x => 2 * Real.sin (π * x + π / 6)) ∧
  (∃ k : ℤ, k = 5 ∧ (1 / 3 + k = 16 / 3) ∧ (21 / 4 ≤ 1 / 3 + ↑k) ∧ (1 / 3 + ↑k ≤ 23 / 4)) :=
  sorry

end analytic_expression_and_symmetry_l20_20074


namespace remainder_of_polynomial_l20_20823

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end remainder_of_polynomial_l20_20823


namespace problem_statement_l20_20930

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l20_20930


namespace range_of_a_l20_20231

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) (q : 0 < 2 * a - 1 ∧ 2 * a - 1 < 1) : 
  (1 / 2) < a ∧ a ≤ (2 / 3) :=
sorry

end range_of_a_l20_20231


namespace inequality_proof_l20_20181

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := 
sorry

end inequality_proof_l20_20181


namespace absolute_value_inequality_l20_20516

theorem absolute_value_inequality (x : ℝ) : 
  (|3 * x + 1| > 2) ↔ (x > 1/3 ∨ x < -1) := by
  sorry

end absolute_value_inequality_l20_20516


namespace find_a_value_l20_20519

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l20_20519


namespace smallest_cars_number_l20_20698

theorem smallest_cars_number :
  ∃ N : ℕ, N > 2 ∧ (N % 5 = 2) ∧ (N % 6 = 2) ∧ (N % 7 = 2) ∧ N = 212 := by
  sorry

end smallest_cars_number_l20_20698


namespace pairs_solution_l20_20115

theorem pairs_solution (x y : ℝ) :
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := 
by
  sorry

end pairs_solution_l20_20115


namespace total_travel_time_l20_20712

theorem total_travel_time (x y : ℝ) : 
  (x / 50 + y / 70 + 0.5) = (7 * x + 5 * y) / 350 + 0.5 :=
by
  sorry

end total_travel_time_l20_20712


namespace statue_of_liberty_model_height_l20_20486

theorem statue_of_liberty_model_height :
  let scale_ratio : Int := 30
  let actual_height : Int := 305
  round (actual_height / scale_ratio) = 10 := by
  sorry

end statue_of_liberty_model_height_l20_20486


namespace product_of_numbers_l20_20566

theorem product_of_numbers (x y : ℝ) 
  (h₁ : x + y = 8 * (x - y)) 
  (h₂ : x * y = 40 * (x - y)) : x * y = 4032 := 
by
  sorry

end product_of_numbers_l20_20566


namespace necessary_but_not_sufficient_l20_20246

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l20_20246


namespace negation_of_universal_proposition_l20_20062

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l20_20062


namespace max_electronic_thermometers_l20_20275

theorem max_electronic_thermometers :
  ∀ (x : ℕ), 10 * x + 3 * (53 - x) ≤ 300 → x ≤ 20 :=
by
  sorry

end max_electronic_thermometers_l20_20275


namespace fraction_satisfactory_is_two_thirds_l20_20252

-- Total number of students with satisfactory grades
def satisfactory_grades : ℕ := 3 + 7 + 4 + 2

-- Total number of students with unsatisfactory grades
def unsatisfactory_grades : ℕ := 4

-- Total number of students
def total_students : ℕ := satisfactory_grades + unsatisfactory_grades

-- Fraction of satisfactory grades
def fraction_satisfactory : ℚ := satisfactory_grades / total_students

theorem fraction_satisfactory_is_two_thirds :
  fraction_satisfactory = 2 / 3 := by
  sorry

end fraction_satisfactory_is_two_thirds_l20_20252


namespace nat_condition_l20_20941

theorem nat_condition (n : ℕ) (h : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  (∃ p : ℕ, n = 2^p - 2) :=
sorry

end nat_condition_l20_20941


namespace watermelon_seeds_l20_20895

theorem watermelon_seeds (n_slices : ℕ) (total_seeds : ℕ) (B W : ℕ) 
  (h1: n_slices = 40) 
  (h2: B = W) 
  (h3 : n_slices * B + n_slices * W = total_seeds)
  (h4 : total_seeds = 1600) : B = 20 :=
by {
  sorry
}

end watermelon_seeds_l20_20895


namespace complementary_angles_ratio_4_to_1_smaller_angle_l20_20261

theorem complementary_angles_ratio_4_to_1_smaller_angle :
  ∃ (θ : ℝ), (4 * θ + θ = 90) ∧ (θ = 18) :=
by
  sorry

end complementary_angles_ratio_4_to_1_smaller_angle_l20_20261


namespace ratio_large_to_small_l20_20869

-- Definitions of the conditions
def total_fries_sold : ℕ := 24
def small_fries_sold : ℕ := 4
def large_fries_sold : ℕ := total_fries_sold - small_fries_sold

-- The proof goal
theorem ratio_large_to_small : large_fries_sold / small_fries_sold = 5 :=
by
  -- Mathematical steps would go here, but we skip with sorry
  sorry

end ratio_large_to_small_l20_20869


namespace parallel_lines_suff_cond_not_necess_l20_20149

theorem parallel_lines_suff_cond_not_necess (a : ℝ) :
  a = -2 → 
  (∀ x y : ℝ, (2 * x + y - 3 = 0) ∧ (2 * x + y + 4 = 0) → 
    (∃ a : ℝ, a = -2 ∨ a = 1)) ∧
    (a = -2 → ∃ a : ℝ, a = -2 ∨ a = 1) :=
by {
  sorry
}

end parallel_lines_suff_cond_not_necess_l20_20149


namespace quadratic_solutions_l20_20494

theorem quadratic_solutions:
  (2 * (x : ℝ)^2 - 5 * x + 2 = 0) ↔ (x = 2 ∨ x = 1 / 2) :=
sorry

end quadratic_solutions_l20_20494


namespace geometric_sequence_sum_l20_20593

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1 / 4) :
  a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 = 341 / 32 :=
by sorry

end geometric_sequence_sum_l20_20593


namespace problem_solution_l20_20217

variable (U : Set Real) (a b : Real) (t : Real)
variable (A B : Set Real)

-- Conditions
def condition1 : U = Set.univ := sorry

def condition2 : ∀ x, a ≠ 0 → ax^2 + 2 * x + b > 0 ↔ x ≠ -1 / a := sorry

def condition3 : a > b := sorry

def condition4 : t = (a^2 + b^2) / (a - b) := sorry

def condition5 : ∀ m, (∀ x, |x + 1| - |x - 3| ≤ m^2 - 3 * m) → m ∈ B := sorry

-- To Prove
theorem problem_solution : A ∩ (Set.univ \ B) = {m : Real | 2 * Real.sqrt 2 ≤ m ∧ m < 4} := sorry

end problem_solution_l20_20217


namespace find_a_l20_20052

theorem find_a (x y a : ℕ) (h1 : ((10 : ℕ) ^ ((32 : ℕ) / y)) ^ a - (64 : ℕ) = (279 : ℕ))
                 (h2 : a > 0)
                 (h3 : x * y = 32) :
  a = 1 :=
sorry

end find_a_l20_20052


namespace value_of_expression_l20_20090

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l20_20090


namespace product_not_perfect_power_l20_20807

theorem product_not_perfect_power (n : ℕ) : ¬∃ (k : ℕ) (a : ℤ), k > 1 ∧ n * (n + 1) = a^k := by
  sorry

end product_not_perfect_power_l20_20807


namespace angle_bisector_form_l20_20553

noncomputable def P : ℝ × ℝ := (-8, 5)
noncomputable def Q : ℝ × ℝ := (-15, -19)
noncomputable def R : ℝ × ℝ := (1, -7)

-- Function to check if the given equation can be in the form ax + 2y + c = 0
-- and that a + c equals 89.
theorem angle_bisector_form (a c : ℝ) : a + c = 89 :=
by
   sorry

end angle_bisector_form_l20_20553


namespace flowers_total_l20_20075

theorem flowers_total (yoojung_flowers : ℕ) (namjoon_flowers : ℕ)
 (h1 : yoojung_flowers = 32)
 (h2 : yoojung_flowers = 4 * namjoon_flowers) :
  yoojung_flowers + namjoon_flowers = 40 := by
  sorry

end flowers_total_l20_20075


namespace radius_of_inner_circle_l20_20772

def right_triangle_legs (AC BC : ℝ) : Prop :=
  AC = 3 ∧ BC = 4

theorem radius_of_inner_circle (AC BC : ℝ) (h : right_triangle_legs AC BC) :
  ∃ r : ℝ, r = 2 :=
by
  sorry

end radius_of_inner_circle_l20_20772


namespace imaginary_part_of_conjugate_l20_20078

def z : Complex := Complex.mk 1 2

def z_conj : Complex := Complex.mk 1 (-2)

theorem imaginary_part_of_conjugate :
  z_conj.im = -2 := by
  sorry

end imaginary_part_of_conjugate_l20_20078


namespace cube_edge_length_l20_20980

def radius := 2
def edge_length (r : ℕ) := 4 + 2 * r

theorem cube_edge_length :
  ∀ r : ℕ, r = radius → edge_length r = 8 :=
by
  intros r h
  rw [h, edge_length]
  rfl

end cube_edge_length_l20_20980


namespace c_share_of_profit_l20_20065

theorem c_share_of_profit (a b c total_profit : ℕ) 
  (h₁ : a = 5000) (h₂ : b = 8000) (h₃ : c = 9000) (h₄ : total_profit = 88000) :
  c * total_profit / (a + b + c) = 36000 :=
by
  sorry

end c_share_of_profit_l20_20065


namespace sum_of_m_and_n_l20_20780

theorem sum_of_m_and_n (m n : ℚ) (h : (m - 3) * (Real.sqrt 5) + 2 - n = 0) : m + n = 5 :=
sorry

end sum_of_m_and_n_l20_20780


namespace find_number_l20_20375

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
by {
  sorry
}

end find_number_l20_20375


namespace hh3_value_l20_20095

noncomputable def h (x : ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x - 1

theorem hh3_value : h (h 3) = 3406935 := by
  sorry

end hh3_value_l20_20095


namespace right_triangle_area_l20_20112

theorem right_triangle_area (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) (h4 : a^2 + b^2 = c^2) :
  (1/2) * (a : ℝ) * b = 30 :=
by
  sorry

end right_triangle_area_l20_20112


namespace max_naive_number_l20_20979

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end max_naive_number_l20_20979


namespace arvin_first_day_km_l20_20232

theorem arvin_first_day_km :
  ∀ (x : ℕ), (∀ i : ℕ, (i < 5 → (i + x) < 6) → (x + 4 = 6)) → x = 2 :=
by sorry

end arvin_first_day_km_l20_20232


namespace find_z_given_conditions_l20_20299

variable (x y z : ℤ)

theorem find_z_given_conditions :
  (x + y) / 2 = 4 →
  x + y + z = 0 →
  z = -8 := by
  sorry

end find_z_given_conditions_l20_20299


namespace boris_neighbors_l20_20284

-- Define the people
inductive Person
| Arkady | Boris | Vera | Galya | Danya | Egor
deriving DecidableEq

open Person

-- Define the circular arrangement
def next_to (p1 p2 : Person) : Prop :=
p1 = Vera ∧ p2 = Danya ∨
p1 = Danya ∧ p2 = Egor ∨
p1 = Egor ∧ p2 = Vera ∨
p1 = Boris ∧ p2 = Galya ∨
p1 = Galya ∧ p2 = Boris ∨
p1 = Boris ∧ p2 = Arkady ∨
p1 = Arkady ∧ p2 = Boris

axiom danya_next_to_vera : next_to Danya Vera
axiom galya_opposite_egor : ∀ p, (p = Galya) = (p ≠ Egor) ∧ (next_to Egor Danya)  
axiom egor_next_to_danya : next_to Egor Danya
axiom arkady_not_next_to_galya : ¬ next_to Arkady Galya

theorem boris_neighbors : next_to Boris Arkady ∧ next_to Boris Galya :=
by {
  sorry
}

end boris_neighbors_l20_20284


namespace range_of_f_l20_20655

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 4^x

def domain_M (x : ℝ) : Prop := 1 < x ∧ x < 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain_M x ∧ f x = y) ↔ -32 < y ∧ y < 4 :=
sorry

end range_of_f_l20_20655


namespace min_days_required_l20_20162

theorem min_days_required (n : ℕ) (h1 : n ≥ 1) (h2 : 2 * (2^n - 1) ≥ 100) : n = 6 :=
sorry

end min_days_required_l20_20162


namespace max_value_of_PQ_l20_20748

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 12)
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

theorem max_value_of_PQ (t : ℝ) : abs (f t - g t) ≤ 2 :=
by sorry

end max_value_of_PQ_l20_20748


namespace line_through_M_intersects_lines_l20_20340

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def line1 (t : ℝ) : Point3D :=
  {x := 2 - t, y := 3, z := -2 + t}

def plane1 (p : Point3D) : Prop :=
  2 * p.x - 2 * p.y - p.z - 4 = 0

def plane2 (p : Point3D) : Prop :=
  p.x + 3 * p.y + 2 * p.z + 1 = 0

def param_eq (t : ℝ) : Point3D :=
  {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t}

theorem line_through_M_intersects_lines : 
  ∀ (t : ℝ), plane1 (param_eq t) ∧ plane2 (param_eq t) -> 
  ∃ t, param_eq t = {x := -2 + 13 * t, y := -3 * t, z := 3 - 12 * t} :=
by
  intros t h
  sorry

end line_through_M_intersects_lines_l20_20340


namespace prove_nat_number_l20_20254

theorem prove_nat_number (p : ℕ) (hp : Nat.Prime p) (n : ℕ) :
  n^2 = p^2 + 3*p + 9 → n = 7 :=
sorry

end prove_nat_number_l20_20254


namespace water_settles_at_34_cm_l20_20851

-- Conditions definitions
def h : ℝ := 40 -- Initial height of the liquids in cm
def ρ_w : ℝ := 1000 -- Density of water in kg/m^3
def ρ_o : ℝ := 700  -- Density of oil in kg/m^3

-- Given the conditions provided above,
-- prove that the new height level of water in the first vessel is 34 cm
theorem water_settles_at_34_cm :
  (40 / (1 + (ρ_o / ρ_w))) = 34 := 
sorry

end water_settles_at_34_cm_l20_20851


namespace length_of_row_of_small_cubes_l20_20417

/-!
# Problem: Calculate the length of a row of smaller cubes

A cube with an edge length of 0.5 m is cut into smaller cubes, each with an edge length of 2 mm.
Prove that the length of the row formed by arranging the smaller cubes in a continuous line 
is 31 km and 250 m.
-/

noncomputable def large_cube_edge_length_m : ℝ := 0.5
noncomputable def small_cube_edge_length_mm : ℝ := 2

theorem length_of_row_of_small_cubes :
  let length_mm := 31250000
  (31 : ℝ) * 1000 + (250 : ℝ) = length_mm / 1000 + 250 := 
sorry

end length_of_row_of_small_cubes_l20_20417


namespace ending_number_l20_20003

theorem ending_number (h : ∃ n, 3 * n = 99 ∧ n = 33) : ∃ m, m = 99 :=
by
  sorry

end ending_number_l20_20003


namespace op_plus_18_plus_l20_20120

def op_plus (y: ℝ) : ℝ := 9 - y
def plus_op (y: ℝ) : ℝ := y - 9

theorem op_plus_18_plus :
  plus_op (op_plus 18) = -18 := by
  sorry

end op_plus_18_plus_l20_20120


namespace lauren_total_earnings_l20_20736

-- Define earnings conditions
def mondayCommercialEarnings (views : ℕ) : ℝ := views * 0.40
def mondaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 0.80

def tuesdayCommercialEarnings (views : ℕ) : ℝ := views * 0.50
def tuesdaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 1.00

def weekendMerchandiseEarnings (sales : ℝ) : ℝ := 0.10 * sales

-- Specific conditions for each day
def mondayTotalEarnings : ℝ := mondayCommercialEarnings 80 + mondaySubscriptionEarnings 20
def tuesdayTotalEarnings : ℝ := tuesdayCommercialEarnings 100 + tuesdaySubscriptionEarnings 27
def weekendTotalEarnings : ℝ := weekendMerchandiseEarnings 150

-- Total earnings for the period
def totalEarnings : ℝ := mondayTotalEarnings + tuesdayTotalEarnings + weekendTotalEarnings

-- Examining the final value
theorem lauren_total_earnings : totalEarnings = 140.00 := by
  sorry

end lauren_total_earnings_l20_20736


namespace value_of_expression_l20_20216

theorem value_of_expression (x y : ℝ) (h : x + 2 * y = 30) : (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 :=
by
  sorry

end value_of_expression_l20_20216


namespace min_value_x_squared_plus_y_squared_l20_20196

theorem min_value_x_squared_plus_y_squared {x y : ℝ} 
  (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) : 
  ∃ m : ℝ, m = 14 - 2 * Real.sqrt 13 ∧ ∀ u v : ℝ, (u^2 + v^2 - 4*u - 6*v + 12 = 0) → (u^2 + v^2 ≥ m) :=
by
  sorry

end min_value_x_squared_plus_y_squared_l20_20196


namespace totalPayment_l20_20032

def totalNumberOfTrees : Nat := 850
def pricePerDouglasFir : Nat := 300
def pricePerPonderosaPine : Nat := 225
def numberOfDouglasFirPurchased : Nat := 350
def numberOfPonderosaPinePurchased := totalNumberOfTrees - numberOfDouglasFirPurchased

def costDouglasFir := numberOfDouglasFirPurchased * pricePerDouglasFir
def costPonderosaPine := numberOfPonderosaPinePurchased * pricePerPonderosaPine

def totalCost := costDouglasFir + costPonderosaPine

theorem totalPayment : totalCost = 217500 := by
  sorry

end totalPayment_l20_20032


namespace find_a_l20_20469

theorem find_a (a : ℝ) (x : ℝ) : (a - 1) * x^|a| + 4 = 0 → |a| = 1 → a ≠ 1 → a = -1 :=
by
  intros
  sorry

end find_a_l20_20469


namespace total_cost_two_rackets_l20_20740

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l20_20740


namespace number_of_children_correct_l20_20924

def total_spectators : ℕ := 25000
def men_spectators : ℕ := 15320
def ratio_children_women : ℕ × ℕ := (7, 3)
def remaining_spectators : ℕ := total_spectators - men_spectators
def total_ratio_parts : ℕ := ratio_children_women.1 + ratio_children_women.2
def spectators_per_part : ℕ := remaining_spectators / total_ratio_parts

def children_spectators : ℕ := spectators_per_part * ratio_children_women.1

theorem number_of_children_correct : children_spectators = 6776 := by
  sorry

end number_of_children_correct_l20_20924


namespace find_b_l20_20634

variable (a b : Prod ℝ ℝ)
variable (x y : ℝ)

theorem find_b (h1 : (Prod.fst a + Prod.fst b = 0) ∧
                    (Real.sqrt ((Prod.snd a + Prod.snd b) ^ 2) = 1))
                    (h2 : a = (2, -1)) :
                    b = (-2, 2) ∨ b = (-2, 0) :=
by sorry

end find_b_l20_20634


namespace parallel_condition_coincide_condition_perpendicular_condition_l20_20955

-- Define the equations of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y = 8

-- Parallel lines condition
theorem parallel_condition (m : ℝ) : (l1 m = l2 m ↔ m = -7) →
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y) → False := sorry

-- Coincidence condition
theorem coincide_condition (m : ℝ) : 
  (l1 (-1) = l2 (-1)) :=
sorry

-- Perpendicular lines condition
theorem perpendicular_condition (m : ℝ) : 
  (m = - 13 / 3 ↔ (2 * (m + 3) + 4 * (m + 5) = 0)) :=
sorry

end parallel_condition_coincide_condition_perpendicular_condition_l20_20955


namespace inheritance_amount_l20_20754

theorem inheritance_amount (x : ℝ) 
  (h1 : x * 0.25 + (x - x * 0.25) * 0.12 = 13600) : x = 40000 :=
by
  -- This is where the proof would go
  sorry

end inheritance_amount_l20_20754


namespace min_abs_diff_l20_20800

theorem min_abs_diff (a b c d : ℝ) (h1 : |a - b| = 5) (h2 : |b - c| = 8) (h3 : |c - d| = 10) : 
  ∃ m, m = |a - d| ∧ m = 3 := 
by 
  sorry

end min_abs_diff_l20_20800


namespace tiles_needed_l20_20399

/-- 
Given:
- The cafeteria is tiled with the same floor tiles.
- It takes 630 tiles to cover an area of 18 square decimeters of tiles.
- We switch to square tiles with a side length of 6 decimeters.

Prove:
- The number of new tiles needed to cover the same area is 315.
--/
theorem tiles_needed (n_tiles : ℕ) (area_per_tile : ℕ) (new_tile_side_length : ℕ) 
  (h1 : n_tiles = 630) (h2 : area_per_tile = 18) (h3 : new_tile_side_length = 6) :
  (630 * 18) / (6 * 6) = 315 :=
by
  sorry

end tiles_needed_l20_20399


namespace trivia_team_students_l20_20652

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (total_students : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  total_students = not_picked + groups * students_per_group →
  total_students = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end trivia_team_students_l20_20652


namespace find_x_if_delta_phi_eq_3_l20_20392

variable (x : ℚ)

def delta (x : ℚ) := 4 * x + 9
def phi (x : ℚ) := 9 * x + 6

theorem find_x_if_delta_phi_eq_3 : 
  delta (phi x) = 3 → x = -5 / 6 := by 
  sorry

end find_x_if_delta_phi_eq_3_l20_20392


namespace polygon_side_count_l20_20017

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l20_20017


namespace no_prime_satisfies_condition_l20_20304

theorem no_prime_satisfies_condition :
  ¬ ∃ p : ℕ, p > 1 ∧ 10 * (p : ℝ) = (p : ℝ) + 5.4 := by {
  sorry
}

end no_prime_satisfies_condition_l20_20304


namespace train_crosses_in_26_seconds_l20_20119

def speed_km_per_hr := 72
def length_of_train := 250
def length_of_platform := 270

def total_distance := length_of_train + length_of_platform

noncomputable def speed_m_per_s := (speed_km_per_hr * 1000 / 3600)  -- Convert km/hr to m/s

noncomputable def time_to_cross := total_distance / speed_m_per_s

theorem train_crosses_in_26_seconds :
  time_to_cross = 26 := 
sorry

end train_crosses_in_26_seconds_l20_20119


namespace petya_second_race_finishes_first_l20_20964

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end petya_second_race_finishes_first_l20_20964


namespace period_of_trig_sum_l20_20693

theorem period_of_trig_sum : ∀ x : ℝ, 2 * Real.sin x + 3 * Real.cos x = 2 * Real.sin (x + 2 * Real.pi) + 3 * Real.cos (x + 2 * Real.pi) := 
sorry

end period_of_trig_sum_l20_20693


namespace vector_projection_unique_l20_20296

theorem vector_projection_unique (a : ℝ) (c d : ℝ) (h : c + 3 * d = 0) :
    ∃ p : ℝ × ℝ, (∀ a : ℝ, ∀ (v : ℝ × ℝ) (w : ℝ × ℝ), 
      v = (a, 3 * a - 2) → 
      w = (c, d) → 
      ∃ p : ℝ × ℝ, p = (3 / 5, -1 / 5)) :=
sorry

end vector_projection_unique_l20_20296


namespace min_value_expr_l20_20561

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 :=
sorry

end min_value_expr_l20_20561


namespace race_distance_difference_l20_20706

theorem race_distance_difference
  (d : ℕ) (tA tB : ℕ)
  (h_d: d = 80) 
  (h_tA: tA = 20) 
  (h_tB: tB = 25) :
  (d / tA) * tA = d ∧ (d - (d / tB) * tA) = 16 := 
by
  sorry

end race_distance_difference_l20_20706


namespace lines_intersect_at_point_l20_20028

/-
Given two lines parameterized as:
Line 1: (x, y) = (2, 0) + s * (3, -4)
Line 2: (x, y) = (6, -10) + v * (5, 3)
Prove that these lines intersect at (242/29, -248/29).
-/

def parametric_line_1 (s : ℚ) : ℚ × ℚ :=
  (2 + 3 * s, -4 * s)

def parametric_line_2 (v : ℚ) : ℚ × ℚ :=
  (6 + 5 * v, -10 + 3 * v)

theorem lines_intersect_at_point :
  ∃ (s v : ℚ), parametric_line_1 s = parametric_line_2 v ∧ parametric_line_1 s = (242 / 29, -248 / 29) :=
sorry

end lines_intersect_at_point_l20_20028


namespace solve_equation_l20_20391

theorem solve_equation (a b n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (a > 0) → (b > 0) → (n > 0) → (a ^ 2013 + b ^ 2013 = p ^ n) ↔ 
  ∃ k : ℕ, a = 2 ^ k ∧ b = 2 ^ k ∧ p = 2 ∧ n = 2013 * k + 1 :=
by
  sorry

end solve_equation_l20_20391


namespace work_days_for_A_l20_20434

theorem work_days_for_A (x : ℕ) : 
  (∀ a b, 
    (a = 1 / (x : ℚ)) ∧ 
    (b = 1 / 20) ∧ 
    (8 * (a + b) = 14 / 15) → 
    x = 15) :=
by
  intros a b h
  have ha : a = 1 / (x : ℚ) := h.1
  have hb : b = 1 / 20 := h.2.1
  have hab : 8 * (a + b) = 14 / 15 := h.2.2
  sorry

end work_days_for_A_l20_20434


namespace log_expression_is_zero_l20_20044

noncomputable def log_expr : ℝ := (Real.logb 2 3 + Real.logb 2 27) * (Real.logb 4 4 + Real.logb 4 (1/4))

theorem log_expression_is_zero : log_expr = 0 :=
by
  sorry

end log_expression_is_zero_l20_20044


namespace untouched_shapes_after_moves_l20_20673

-- Definitions
def num_shapes : ℕ := 12
def num_triangles : ℕ := 3
def num_squares : ℕ := 4
def num_pentagons : ℕ := 5
def total_moves : ℕ := 10
def petya_moves_first : Prop := True
def vasya_strategy : Prop := True  -- Vasya's strategy to minimize untouched shapes
def petya_strategy : Prop := True  -- Petya's strategy to maximize untouched shapes

-- Theorem
theorem untouched_shapes_after_moves : num_shapes = 12 ∧ num_triangles = 3 ∧ num_squares = 4 ∧ num_pentagons = 5 ∧
                                        total_moves = 10 ∧ petya_moves_first ∧ vasya_strategy ∧ petya_strategy → 
                                        num_shapes - 5 = 6 :=
by
  sorry

end untouched_shapes_after_moves_l20_20673


namespace willie_stickers_l20_20106

theorem willie_stickers (initial_stickers : ℕ) (given_stickers : ℕ) (remaining_stickers : ℕ) :
  initial_stickers = 124 → given_stickers = 23 → remaining_stickers = initial_stickers - given_stickers → remaining_stickers = 101 :=
by
  intros h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining.trans rfl

end willie_stickers_l20_20106


namespace micrometer_conversion_l20_20401

theorem micrometer_conversion :
  (0.01 * (1 * 10 ^ (-6))) = (1 * 10 ^ (-8)) :=
by 
  -- sorry is used to skip the actual proof but ensure the theorem is recognized
  sorry

end micrometer_conversion_l20_20401


namespace tom_chocolates_l20_20715

variable (n : ℕ)

-- Lisa's box holds 64 chocolates and has unit dimensions (1^3 = 1 cubic unit)
def lisa_chocolates := 64
def lisa_volume := 1

-- Tom's box has dimensions thrice Lisa's and hence its volume (3^3 = 27 cubic units)
def tom_volume := 27

-- Number of chocolates Tom's box holds
theorem tom_chocolates : lisa_chocolates * tom_volume = 1728 := by
  -- calculations with known values
  sorry

end tom_chocolates_l20_20715


namespace solve_for_y_l20_20006

theorem solve_for_y : ∀ (y : ℝ), (3 / 4 - 5 / 8 = 1 / y) → y = 8 :=
by
  intros y h
  sorry

end solve_for_y_l20_20006


namespace system_no_solution_l20_20788

theorem system_no_solution (n : ℝ) :
  ∃ x y z : ℝ, (n * x + y = 1) ∧ (1 / 2 * n * y + z = 1) ∧ (x + 1 / 2 * n * z = 2) ↔ n = -1 := 
sorry

end system_no_solution_l20_20788


namespace percentage_decrease_l20_20770

theorem percentage_decrease (x : ℝ) 
  (h1 : 400 * (1 - x / 100) * 1.40 = 476) : 
  x = 15 := 
by 
  sorry

end percentage_decrease_l20_20770


namespace speed_of_man_in_still_water_l20_20574

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : 5 * (v_m + v_s) = 45) (h2 : 5 * (v_m - v_s) = 25) : v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l20_20574


namespace fathers_age_multiple_l20_20984

theorem fathers_age_multiple 
  (Johns_age : ℕ)
  (sum_of_ages : ℕ)
  (additional_years : ℕ)
  (m : ℕ)
  (h1 : Johns_age = 15)
  (h2 : sum_of_ages = 77)
  (h3 : additional_years = 32)
  (h4 : sum_of_ages = Johns_age + (Johns_age * m + additional_years)) :
  m = 2 := 
by 
  sorry

end fathers_age_multiple_l20_20984


namespace compare_logs_l20_20294

noncomputable def a := Real.log 3
noncomputable def b := Real.log 3 / Real.log 2 / 2
noncomputable def c := Real.log 2 / Real.log 3 / 2

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l20_20294


namespace determine_k_l20_20541

theorem determine_k (k : ℝ) : 
  (2 * k * (-1/2) - 3 = -7 * 3) → k = 18 :=
by
  intro h
  sorry

end determine_k_l20_20541


namespace tony_lego_sets_l20_20633

theorem tony_lego_sets
  (price_lego price_sword price_dough : ℕ)
  (num_sword num_dough total_cost : ℕ)
  (L : ℕ)
  (h1 : price_lego = 250)
  (h2 : price_sword = 120)
  (h3 : price_dough = 35)
  (h4 : num_sword = 7)
  (h5 : num_dough = 10)
  (h6 : total_cost = 1940)
  (h7 : total_cost = price_lego * L + price_sword * num_sword + price_dough * num_dough) :
  L = 3 := 
by
  sorry

end tony_lego_sets_l20_20633


namespace total_books_l20_20409

def books_per_shelf : ℕ := 78
def number_of_shelves : ℕ := 15

theorem total_books : books_per_shelf * number_of_shelves = 1170 := 
by
  sorry

end total_books_l20_20409


namespace multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l20_20663

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem multiple_of_4 : ∃ k : ℕ, y = 4 * k := by
  sorry

theorem multiple_of_8 : ∃ k : ℕ, y = 8 * k := by
  sorry

theorem not_multiple_of_16 : ¬ ∃ k : ℕ, y = 16 * k := by
  sorry

theorem multiple_of_24 : ∃ k : ℕ, y = 24 * k := by
  sorry

end multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l20_20663


namespace q_computation_l20_20844

def q : ℤ → ℤ → ℤ :=
  λ x y =>
    if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
    else if x < 0 ∧ y < 0 then x - 3 * y
    else 2 * x + y

theorem q_computation : q (q 2 (-2)) (q (-4) (-1)) = 3 :=
by {
  sorry
}

end q_computation_l20_20844


namespace isosceles_triangle_base_length_l20_20079

theorem isosceles_triangle_base_length
  (perimeter_eq_triangle : ℕ)
  (perimeter_isosceles_triangle : ℕ)
  (side_eq_triangle_isosceles : ℕ)
  (side_eq : side_eq_triangle_isosceles = perimeter_eq_triangle / 3)
  (perimeter_eq : perimeter_isosceles_triangle = 2 * side_eq_triangle_isosceles + 15) :
  15 = perimeter_isosceles_triangle - 2 * side_eq_triangle_isosceles :=
sorry

end isosceles_triangle_base_length_l20_20079


namespace quad_form_unique_solution_l20_20778

theorem quad_form_unique_solution (d e f : ℤ) (h1 : d * d = 16) (h2 : 2 * d * e = -40) (h3 : e * e + f = -56) : d * e = -20 :=
by sorry

end quad_form_unique_solution_l20_20778


namespace elvis_squares_count_l20_20819

theorem elvis_squares_count :
  ∀ (total : ℕ) (Elvis_squares Ralph_squares squares_used_by_Ralph matchsticks_left : ℕ)
  (uses_by_Elvis_per_square uses_by_Ralph_per_square : ℕ),
  total = 50 →
  uses_by_Elvis_per_square = 4 →
  uses_by_Ralph_per_square = 8 →
  Ralph_squares = 3 →
  matchsticks_left = 6 →
  squares_used_by_Ralph = Ralph_squares * uses_by_Ralph_per_square →
  total = (Elvis_squares * uses_by_Elvis_per_square) + squares_used_by_Ralph + matchsticks_left →
  Elvis_squares = 5 :=
by
  sorry

end elvis_squares_count_l20_20819


namespace number_of_children_l20_20227

-- Define conditions
variable (A C : ℕ) (h1 : A + C = 280) (h2 : 60 * A + 25 * C = 14000)

-- Lean statement to prove the number of children
theorem number_of_children : C = 80 :=
by
  sorry

end number_of_children_l20_20227


namespace inequality_proof_l20_20352

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := 
by 
  sorry

end inequality_proof_l20_20352


namespace john_paid_correct_amount_l20_20757

theorem john_paid_correct_amount : 
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  john_share = 8400 :=
by
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  show john_share = 8400
  sorry

end john_paid_correct_amount_l20_20757


namespace spherical_coordinate_cone_l20_20703

-- Define spherical coordinates
structure SphericalCoordinate :=
  (ρ : ℝ)
  (θ : ℝ)
  (φ : ℝ)

-- Definition to describe the cone condition
def isCone (d : ℝ) (p : SphericalCoordinate) : Prop :=
  p.φ = d

-- The main theorem to state the problem
theorem spherical_coordinate_cone (d : ℝ) :
  ∀ (p : SphericalCoordinate), isCone d p → ∃ (ρ : ℝ), ∃ (θ : ℝ), (p = ⟨ρ, θ, d⟩) := sorry

end spherical_coordinate_cone_l20_20703


namespace triangle_rectangle_ratio_l20_20785

theorem triangle_rectangle_ratio (s b w l : ℕ) 
(h1 : 2 * s + b = 60) 
(h2 : 2 * (w + l) = 60) 
(h3 : 2 * w = l) 
(h4 : b = w) 
: s / w = 5 / 2 := 
by 
  sorry

end triangle_rectangle_ratio_l20_20785


namespace Anne_wander_time_l20_20332

theorem Anne_wander_time (distance speed : ℝ) (h1 : distance = 3.0) (h2 : speed = 2.0) : distance / speed = 1.5 := by
  -- Given conditions
  sorry

end Anne_wander_time_l20_20332


namespace distance_between_first_and_last_tree_l20_20183

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 100) 
  (h3 : d / 5 = 20) :
  (20 * 9 = 180) :=
by
  sorry

end distance_between_first_and_last_tree_l20_20183


namespace range_of_m_l20_20285

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, f' x < x)
variable (h_ineq : ∀ m, f (4 - m) - f m ≥ 8 - 4 * m)

theorem range_of_m (m : ℝ) : m ≥ 2 :=
sorry

end range_of_m_l20_20285


namespace aaronFoundCards_l20_20350

-- Given conditions
def initialCardsAaron : ℕ := 5
def finalCardsAaron : ℕ := 67

-- Theorem statement
theorem aaronFoundCards : finalCardsAaron - initialCardsAaron = 62 :=
by
  sorry

end aaronFoundCards_l20_20350


namespace number_of_animal_books_l20_20983

variable (A : ℕ)

theorem number_of_animal_books (h1 : 6 * 6 + 3 * 6 + A * 6 = 102) : A = 8 :=
sorry

end number_of_animal_books_l20_20983


namespace car_division_ways_l20_20716

/-- 
Prove that the number of ways to divide 6 people 
into two different cars, with each car holding 
a maximum of 4 people, is equal to 50. 
-/
theorem car_division_ways : 
  (∃ s1 s2 : Finset ℕ, s1.card = 2 ∧ s2.card = 4) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 3 ∧ s2.card = 3) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 4 ∧ s2.card = 2) →
  (15 + 20 + 15 = 50) := 
by 
  sorry

end car_division_ways_l20_20716


namespace consecutive_numbers_count_l20_20397

theorem consecutive_numbers_count (n x : ℕ) (h_avg : (2 * n * 20 = n * (2 * x + n - 1))) (h_largest : x + n - 1 = 23) : n = 7 :=
by
  sorry

end consecutive_numbers_count_l20_20397


namespace line_CD_area_triangle_equality_line_CD_midpoint_l20_20069

theorem line_CD_area_triangle_equality :
  ∃ k : ℝ, 4 * k - 1 = 1 - k := sorry

theorem line_CD_midpoint :
  ∃ k : ℝ, 9 * k - 2 = 1 := sorry

end line_CD_area_triangle_equality_line_CD_midpoint_l20_20069


namespace division_correct_l20_20777

theorem division_correct (x : ℝ) (h : 10 / x = 2) : 20 / x = 4 :=
by
  sorry

end division_correct_l20_20777


namespace prism_volume_l20_20184

theorem prism_volume (x y z : ℝ) (h1 : x * y = 24) (h2 : y * z = 8) (h3 : x * z = 3) : 
  x * y * z = 24 :=
sorry

end prism_volume_l20_20184


namespace tangent_line_circle_l20_20102

theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*y = 0 → y = a) → (a = 0 ∨ a = 2) :=
by
  sorry

end tangent_line_circle_l20_20102


namespace mean_points_scored_l20_20015

def Mrs_Williams_points : ℝ := 50
def Mr_Adams_points : ℝ := 57
def Mrs_Browns_points : ℝ := 49
def Mrs_Daniels_points : ℝ := 57

def total_points : ℝ := Mrs_Williams_points + Mr_Adams_points + Mrs_Browns_points + Mrs_Daniels_points
def number_of_classes : ℝ := 4

theorem mean_points_scored :
  (total_points / number_of_classes) = 53.25 :=
by
  sorry

end mean_points_scored_l20_20015


namespace jenn_has_five_jars_l20_20456

/-- Each jar can hold 160 quarters, the bike costs 180 dollars, 
    Jenn will have 20 dollars left over, 
    and a quarter is worth 0.25 dollars.
    Prove that Jenn has 5 jars full of quarters. -/
theorem jenn_has_five_jars :
  let quarters_per_jar := 160
  let bike_cost := 180
  let money_left := 20
  let total_money_needed := bike_cost + money_left
  let quarter_value := 0.25
  let total_quarters_needed := total_money_needed / quarter_value
  let jars := total_quarters_needed / quarters_per_jar
  
  jars = 5 :=
by
  sorry

end jenn_has_five_jars_l20_20456


namespace trains_crossing_time_l20_20390

theorem trains_crossing_time
  (L1 : ℕ) (L2 : ℕ) (T1 : ℕ) (T2 : ℕ)
  (H1 : L1 = 150) (H2 : L2 = 180)
  (H3 : T1 = 10) (H4 : T2 = 15) :
  (L1 + L2) / ((L1 / T1) + (L2 / T2)) = 330 / 27 := sorry

end trains_crossing_time_l20_20390


namespace avg_of_x_y_is_41_l20_20474

theorem avg_of_x_y_is_41 
  (x y : ℝ) 
  (h : (4 + 6 + 8 + x + y) / 5 = 20) 
  : (x + y) / 2 = 41 := 
by 
  sorry

end avg_of_x_y_is_41_l20_20474


namespace Sam_has_most_pages_l20_20023

theorem Sam_has_most_pages :
  let pages_per_inch_miles := 5
  let inches_miles := 240
  let pages_per_inch_daphne := 50
  let inches_daphne := 25
  let pages_per_inch_sam := 30
  let inches_sam := 60

  let pages_miles := inches_miles * pages_per_inch_miles
  let pages_daphne := inches_daphne * pages_per_inch_daphne
  let pages_sam := inches_sam * pages_per_inch_sam
  pages_sam = 1800 ∧ pages_sam > pages_miles ∧ pages_sam > pages_daphne :=
by
  sorry

end Sam_has_most_pages_l20_20023


namespace maximize_hotel_profit_l20_20606

theorem maximize_hotel_profit :
  let rooms := 50
  let base_price := 180
  let increase_per_vacancy := 10
  let maintenance_cost := 20
  ∃ (x : ℕ), ((base_price + increase_per_vacancy * x) * (rooms - x) 
    - maintenance_cost * (rooms - x) = 10890) ∧ (base_price + increase_per_vacancy * x = 350) :=
by
  sorry

end maximize_hotel_profit_l20_20606


namespace original_fraction_l20_20694

theorem original_fraction (n d : ℝ) (h1 : n + d = 5.25) (h2 : (n + 3) / (2 * d) = 1 / 3) : n / d = 2 / 33 :=
by
  sorry

end original_fraction_l20_20694


namespace dad_caught_more_trouts_l20_20879

-- Definitions based on conditions
def caleb_trouts : ℕ := 2
def dad_trouts : ℕ := 3 * caleb_trouts

-- The proof problem: proving dad caught 4 more trouts than Caleb
theorem dad_caught_more_trouts : dad_trouts = caleb_trouts + 4 :=
by
  sorry

end dad_caught_more_trouts_l20_20879


namespace john_minimum_pizzas_l20_20464

theorem john_minimum_pizzas (car_cost bag_cost earnings_per_pizza gas_cost p : ℕ) 
  (h_car : car_cost = 6000)
  (h_bag : bag_cost = 200)
  (h_earnings : earnings_per_pizza = 12)
  (h_gas : gas_cost = 4)
  (h_p : 8 * p >= car_cost + bag_cost) : p >= 775 := 
sorry

end john_minimum_pizzas_l20_20464


namespace prove_p_false_and_q_true_l20_20411

variables (p q : Prop)

theorem prove_p_false_and_q_true (h1 : p ∨ q) (h2 : ¬p) : ¬p ∧ q :=
by {
  -- proof placeholder
  sorry
}

end prove_p_false_and_q_true_l20_20411


namespace sphere_diameter_l20_20177

theorem sphere_diameter 
  (shadow_sphere : ℝ)
  (height_pole : ℝ)
  (shadow_pole : ℝ)
  (parallel_rays : Prop)
  (vertical_objects : Prop)
  (tan_theta : ℝ) :
  shadow_sphere = 12 →
  height_pole = 1.5 →
  shadow_pole = 3 →
  (tan_theta = height_pole / shadow_pole) →
  parallel_rays →
  vertical_objects →
  2 * (shadow_sphere * tan_theta) = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sphere_diameter_l20_20177


namespace sum_of_decimals_is_one_l20_20908

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end sum_of_decimals_is_one_l20_20908


namespace total_overtime_hours_worked_l20_20300

def gary_wage : ℕ := 12
def mary_wage : ℕ := 14
def john_wage : ℕ := 16
def alice_wage : ℕ := 18
def michael_wage : ℕ := 20

def regular_hours : ℕ := 40
def overtime_rate : ℚ := 1.5

def total_paycheck : ℚ := 3646

theorem total_overtime_hours_worked :
  let gary_overtime := gary_wage * overtime_rate
  let mary_overtime := mary_wage * overtime_rate
  let john_overtime := john_wage * overtime_rate
  let alice_overtime := alice_wage * overtime_rate
  let michael_overtime := michael_wage * overtime_rate
  let regular_total := (gary_wage + mary_wage + john_wage + alice_wage + michael_wage) * regular_hours
  let total_overtime_pay := total_paycheck - regular_total
  let total_overtime_rate := gary_overtime + mary_overtime + john_overtime + alice_overtime + michael_overtime
  let overtime_hours := total_overtime_pay / total_overtime_rate
  overtime_hours.floor = 3 := 
by
  sorry

end total_overtime_hours_worked_l20_20300


namespace f_25_over_11_neg_l20_20978

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end f_25_over_11_neg_l20_20978


namespace temperature_at_6_km_l20_20624

-- Define the initial conditions
def groundTemperature : ℝ := 25
def temperatureDropPerKilometer : ℝ := 5

-- Define the question which is the temperature at a height of 6 kilometers
def temperatureAtHeight (height : ℝ) : ℝ :=
  groundTemperature - temperatureDropPerKilometer * height

-- Prove that the temperature at 6 kilometers is -5 degrees Celsius
theorem temperature_at_6_km : temperatureAtHeight 6 = -5 := by
  -- Use expected proof  
  simp [temperatureAtHeight, groundTemperature, temperatureDropPerKilometer]
  sorry

end temperature_at_6_km_l20_20624


namespace sum_and_gap_l20_20604

-- Define the gap condition
def gap_condition (x : ℝ) : Prop :=
  |5.46 - x| = 3.97

-- Define the main theorem to be proved 
theorem sum_and_gap :
  ∀ (x : ℝ), gap_condition x → x < 5.46 → x + 5.46 = 6.95 := 
by 
  intros x hx hlt
  sorry

end sum_and_gap_l20_20604


namespace cost_of_pencils_l20_20067

def cost_of_notebooks : ℝ := 3 * 1.2
def cost_of_pens : ℝ := 1.7
def total_spent : ℝ := 6.8

theorem cost_of_pencils :
  total_spent - (cost_of_notebooks + cost_of_pens) = 1.5 :=
by
  sorry

end cost_of_pencils_l20_20067


namespace least_number_to_add_l20_20971

theorem least_number_to_add (m n : ℕ) (h₁ : m = 1052) (h₂ : n = 23) : 
  ∃ k : ℕ, (m + k) % n = 0 ∧ k = 6 :=
by
  sorry

end least_number_to_add_l20_20971


namespace pastries_sold_l20_20255

def initial_pastries : ℕ := 148
def pastries_left : ℕ := 45

theorem pastries_sold : initial_pastries - pastries_left = 103 := by
  sorry

end pastries_sold_l20_20255


namespace relationship_between_a_and_b_l20_20909

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end relationship_between_a_and_b_l20_20909


namespace induction_inequality_term_added_l20_20618

theorem induction_inequality_term_added (k : ℕ) (h : k > 0) :
  let termAdded := (1 / (2 * (k + 1) - 1 : ℝ)) + (1 / (2 * (k + 1) : ℝ)) - (1 / (k + 1 : ℝ))
  ∃ h : ℝ, termAdded = h :=
by
  sorry

end induction_inequality_term_added_l20_20618


namespace proportional_function_property_l20_20875

theorem proportional_function_property :
  (∀ x, ∃ y, y = -3 * x ∧
  (x = 0 → y = 0) ∧
  (x > 0 → y < 0) ∧
  (x < 0 → y > 0) ∧
  (x = 1 → y = -3) ∧
  (∀ x, y = -3 * x → (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0))) :=
by
  sorry

end proportional_function_property_l20_20875


namespace total_weight_correct_l20_20594

-- Definitions of the given weights of materials
def weight_concrete : ℝ := 0.17
def weight_bricks : ℝ := 0.237
def weight_sand : ℝ := 0.646
def weight_stone : ℝ := 0.5
def weight_steel : ℝ := 1.73
def weight_wood : ℝ := 0.894

-- Total weight of all materials
def total_weight : ℝ := 
  weight_concrete + weight_bricks + weight_sand + weight_stone + weight_steel + weight_wood

-- The proof statement
theorem total_weight_correct : total_weight = 4.177 := by
  sorry

end total_weight_correct_l20_20594


namespace shape_area_is_36_l20_20580

def side_length : ℝ := 3
def num_squares : ℕ := 4
def area_square : ℝ := side_length ^ 2
def total_area : ℝ := num_squares * area_square

theorem shape_area_is_36 :
  total_area = 36 := by
  sorry

end shape_area_is_36_l20_20580


namespace proportion_problem_l20_20437

theorem proportion_problem 
  (x : ℝ) 
  (third_number : ℝ) 
  (h1 : 0.75 / x = third_number / 8) 
  (h2 : x = 0.6) 
  : third_number = 10 := 
by 
  sorry

end proportion_problem_l20_20437


namespace fraction_mistake_l20_20320

theorem fraction_mistake (n : ℕ) (h : n = 288) (student_answer : ℕ) 
(h_student : student_answer = 240) : student_answer / n = 5 / 6 := 
by 
  -- Given that n = 288 and the student's answer = 240;
  -- we need to prove that 240/288 = 5/6
  sorry

end fraction_mistake_l20_20320


namespace tina_husband_brownies_days_l20_20769

variable (d : Nat)

theorem tina_husband_brownies_days : 
  (exists (d : Nat), 
    let total_brownies := 24
    let tina_daily := 2
    let husband_daily := 1
    let total_daily := tina_daily + husband_daily
    let shared_with_guests := 4
    let remaining_brownies := total_brownies - shared_with_guests
    let final_leftover := 5
    let brownies_eaten := remaining_brownies - final_leftover
    brownies_eaten = d * total_daily) → d = 5 := 
by
  sorry

end tina_husband_brownies_days_l20_20769


namespace work_completion_time_l20_20871

-- Definitions for work rates
def work_rate_B : ℚ := 1 / 7
def work_rate_A : ℚ := 1 / 10

-- Statement to prove
theorem work_completion_time (W : ℚ) : 
  (1 / work_rate_A + 1 / work_rate_B) = 70 / 17 := 
by 
  sorry

end work_completion_time_l20_20871


namespace code_transformation_l20_20000

def old_to_new_encoding (s : String) : String := sorry

theorem code_transformation :
  old_to_new_encoding "011011010011" = "211221121" := sorry

end code_transformation_l20_20000


namespace average_weight_increase_l20_20297

variable (A N X : ℝ)

theorem average_weight_increase (hN : N = 135.5) (h_avg : A + X = (9 * A - 86 + N) / 9) : 
  X = 5.5 :=
by
  sorry

end average_weight_increase_l20_20297


namespace find_f_neg2_l20_20645

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

theorem find_f_neg2 : f (-2) = -9 :=
by sorry

end find_f_neg2_l20_20645


namespace cyclists_equal_distance_l20_20267

theorem cyclists_equal_distance (v1 v2 v3 : ℝ) (t1 t2 t3 : ℝ) (d : ℝ)
  (h_v1 : v1 = 12) (h_v2 : v2 = 16) (h_v3 : v3 = 24)
  (h_one_riding : t1 + t2 + t3 = 3) 
  (h_dist_equal : v1 * t1 = v2 * t2 ∧ v2 * t2 = v3 * t3 ∧ v1 * t1 = d) :
  d = 16 :=
by
  sorry

end cyclists_equal_distance_l20_20267


namespace average_minutes_heard_l20_20961

theorem average_minutes_heard :
  let total_audience := 200
  let duration := 90
  let percent_entire := 0.15
  let percent_slept := 0.15
  let percent_half := 0.25
  let percent_one_fourth := 0.75
  let total_entire := total_audience * percent_entire
  let total_slept := total_audience * percent_slept
  let remaining := total_audience - total_entire - total_slept
  let total_half := remaining * percent_half
  let total_one_fourth := remaining * percent_one_fourth
  let minutes_entire := total_entire * duration
  let minutes_half := total_half * (duration / 2)
  let minutes_one_fourth := total_one_fourth * (duration / 4)
  let total_minutes_heard := minutes_entire + 0 + minutes_half + minutes_one_fourth
  let average_minutes := total_minutes_heard / total_audience
  average_minutes = 33 :=
by
  sorry

end average_minutes_heard_l20_20961


namespace ab2c_value_l20_20386

theorem ab2c_value (a b c : ℚ) (h₁ : |a + 1| + (b - 2)^2 = 0) (h₂ : |c| = 3) :
  a + b + 2 * c = 7 ∨ a + b + 2 * c = -5 := sorry

end ab2c_value_l20_20386


namespace problem_statement_l20_20404

-- Defining the properties of the function
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def symmetric_about_2 (f : ℝ → ℝ) := ∀ x, f (2 + (2 - x)) = f x

-- Given the function f, even function, and symmetric about line x = 2,
-- and given that f(3) = 3, we need to prove f(-1) = 3.
theorem problem_statement (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : symmetric_about_2 f) 
  (h3 : f 3 = 3) : 
  f (-1) = 3 := 
sorry

end problem_statement_l20_20404


namespace jasper_drinks_more_than_hot_dogs_l20_20415

-- Definition of conditions based on the problem
def bags_of_chips := 27
def fewer_hot_dogs_than_chips := 8
def drinks_sold := 31

-- Definition to compute the number of hot dogs
def hot_dogs_sold := bags_of_chips - fewer_hot_dogs_than_chips

-- Lean 4 statement to prove the final result
theorem jasper_drinks_more_than_hot_dogs : drinks_sold - hot_dogs_sold = 12 :=
by
  -- skipping the proof
  sorry

end jasper_drinks_more_than_hot_dogs_l20_20415


namespace points_on_line_relation_l20_20092

theorem points_on_line_relation (b y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-2) + b) 
  (h2 : y2 = -3 * (-1) + b) 
  (h3 : y3 = -3 * 1 + b) : 
  y1 > y2 ∧ y2 > y3 :=
sorry

end points_on_line_relation_l20_20092


namespace discount_rate_for_1000_min_price_for_1_3_discount_l20_20279

def discounted_price (original_price : ℕ) : ℕ := 
  original_price * 80 / 100

def voucher_amount (discounted_price : ℕ) : ℕ :=
  if discounted_price < 400 then 30
  else if discounted_price < 500 then 60
  else if discounted_price < 700 then 100
  else if discounted_price < 900 then 130
  else 0 -- Can extend the rule as needed

def discount_rate (original_price : ℕ) : ℚ := 
  let total_discount := original_price * 20 / 100 + voucher_amount (discounted_price original_price)
  (total_discount : ℚ) / (original_price : ℚ)

theorem discount_rate_for_1000 : 
  discount_rate 1000 = 0.33 := 
by
  sorry

theorem min_price_for_1_3_discount :
  ∀ (x : ℕ), 500 ≤ x ∧ x ≤ 800 → 0.33 ≤ discount_rate x ↔ (625 ≤ x ∧ x ≤ 750) :=
by
  sorry

end discount_rate_for_1000_min_price_for_1_3_discount_l20_20279


namespace solve_x_l20_20334

def δ (x : ℝ) : ℝ := 4 * x + 6
def φ (x : ℝ) : ℝ := 5 * x + 4

theorem solve_x : ∃ x: ℝ, δ (φ x) = 3 → x = -19 / 20 := by
  sorry

end solve_x_l20_20334


namespace solution_l20_20863

open Set

theorem solution (A B : Set ℤ) :
  (∀ x, x ∈ A ∨ x ∈ B) →
  (∀ x, x ∈ A → (x - 1) ∈ B) →
  (∀ x y, x ∈ B ∧ y ∈ B → (x + y) ∈ A) →
  A = { z | ∃ n, z = 2 * n } ∧ B = { z | ∃ n, z = 2 * n + 1 } :=
by
  sorry

end solution_l20_20863


namespace rick_iron_hours_l20_20118

def can_iron_dress_shirts (h : ℕ) : ℕ := 4 * h

def can_iron_dress_pants (hours : ℕ) : ℕ := 3 * hours

def total_clothes_ironed (h : ℕ) : ℕ := can_iron_dress_shirts h + can_iron_dress_pants 5

theorem rick_iron_hours (h : ℕ) (H : total_clothes_ironed h = 27) : h = 3 :=
by sorry

end rick_iron_hours_l20_20118


namespace unique_subset_empty_set_l20_20290

def discriminant (a : ℝ) : ℝ := 4 - 4 * a^2

theorem unique_subset_empty_set (a : ℝ) :
  (∀ (x : ℝ), ¬(a * x^2 + 2 * x + a = 0)) ↔ (a > 1 ∨ a < -1) :=
by
  sorry

end unique_subset_empty_set_l20_20290


namespace negation_proposition_l20_20854

theorem negation_proposition :
  (¬(∀ x : ℝ, x^2 - x + 2 < 0) ↔ ∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
sorry

end negation_proposition_l20_20854


namespace gabriel_forgot_days_l20_20542

def days_in_july : ℕ := 31
def days_taken : ℕ := 28

theorem gabriel_forgot_days : days_in_july - days_taken = 3 := by
  sorry

end gabriel_forgot_days_l20_20542


namespace area_of_rectangular_garden_l20_20479

-- Definition of conditions
def width : ℕ := 14
def length : ℕ := 3 * width

-- Statement for proof of the area of the rectangular garden
theorem area_of_rectangular_garden :
  length * width = 588 := 
by
  sorry

end area_of_rectangular_garden_l20_20479


namespace green_eyes_students_l20_20692

def total_students := 45
def brown_hair_condition (green_eyes : ℕ) := 3 * green_eyes
def both_attributes := 9
def neither_attributes := 5

theorem green_eyes_students (green_eyes : ℕ) :
  (total_students = (green_eyes - both_attributes) + both_attributes
    + (brown_hair_condition green_eyes - both_attributes) + neither_attributes) →
  green_eyes = 10 :=
by
  sorry

end green_eyes_students_l20_20692


namespace negation_of_p_correct_l20_20639

def p := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p_correct :
  (¬ p) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_of_p_correct_l20_20639


namespace part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l20_20511

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem part1_f0_f1 : f 0 + f 1 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg1_f2 : f (-1) + f 2 = Real.sqrt 3 / 3 := sorry

theorem part1_f_neg2_f3 : f (-2) + f 3 = Real.sqrt 3 / 3 := sorry

theorem part2_conjecture (x1 x2 : ℝ) (h : x1 + x2 = 1) : f x1 + f x2 = Real.sqrt 3 / 3 := sorry

end part1_f0_f1_part1_f_neg1_f2_part1_f_neg2_f3_part2_conjecture_l20_20511


namespace larger_square_side_length_l20_20648

theorem larger_square_side_length (x y H : ℝ) 
  (smaller_square_perimeter : 4 * x = H - 20)
  (larger_square_perimeter : 4 * y = H) :
  y = x + 5 :=
by
  sorry

end larger_square_side_length_l20_20648


namespace initial_kittens_l20_20792

-- Define the number of kittens given to Jessica and Sara, and the number of kittens currently Tim has.
def kittens_given_to_Jessica : ℕ := 3
def kittens_given_to_Sara : ℕ := 6
def kittens_left_with_Tim : ℕ := 9

-- Define the theorem to prove the initial number of kittens Tim had.
theorem initial_kittens (kittens_given_to_Jessica kittens_given_to_Sara kittens_left_with_Tim : ℕ) 
    (h1 : kittens_given_to_Jessica = 3)
    (h2 : kittens_given_to_Sara = 6)
    (h3 : kittens_left_with_Tim = 9) :
    (kittens_given_to_Jessica + kittens_given_to_Sara + kittens_left_with_Tim) = 18 := 
    sorry

end initial_kittens_l20_20792


namespace license_plate_combinations_l20_20398

theorem license_plate_combinations :
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits = 110250 :=
by
  let num_consonants := 21
  let num_vowels := 5
  let num_digits := 10
  sorry

end license_plate_combinations_l20_20398


namespace problem_statement_l20_20738

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

noncomputable def f' (x : ℝ) := (deriv f) x

theorem problem_statement :
  f 2018 + f (-2018) + f' 2019 - f' (-2019) = 2 :=
by {
  sorry
}

end problem_statement_l20_20738


namespace triangle_C_squared_eq_b_a_plus_b_l20_20963

variables {A B C a b : ℝ}

theorem triangle_C_squared_eq_b_a_plus_b
  (h1 : C = 2 * B)
  (h2 : A ≠ B) :
  C^2 = b * (a + b) :=
sorry

end triangle_C_squared_eq_b_a_plus_b_l20_20963


namespace positive_integer_as_sum_of_distinct_factors_l20_20210

-- Defining that all elements of a list are factors of a given number
def AllFactorsOf (factors : List ℕ) (n : ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Defining that the sum of elements in the list equals a given number
def SumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- Theorem statement
theorem positive_integer_as_sum_of_distinct_factors (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m ∧ m ≤ n!) :
  ∃ factors : List ℕ, factors.length ≤ n ∧ AllFactorsOf factors n! ∧ SumList factors = m := 
sorry

end positive_integer_as_sum_of_distinct_factors_l20_20210


namespace cos_8_minus_sin_8_l20_20935

theorem cos_8_minus_sin_8 (α m : ℝ) (h : Real.cos (2 * α) = m) :
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 :=
by
  sorry

end cos_8_minus_sin_8_l20_20935


namespace find_b_l20_20881

theorem find_b (a b : ℝ) (f : ℝ → ℝ) (df : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : ∀ x, f x = a * x + Real.log x)
  (h₂ : ∀ x, f x = 2 * x + b)
  (h₃ : x₀ = 1)
  (h₄ : f x₀ = a) :
  b = -1 := 
by
  sorry

end find_b_l20_20881


namespace pencils_per_row_l20_20841

-- Define the conditions
def total_pencils := 25
def number_of_rows := 5

-- Theorem statement: The number of pencils per row is 5 given the conditions
theorem pencils_per_row : total_pencils / number_of_rows = 5 :=
by
  -- The proof should go here
  sorry

end pencils_per_row_l20_20841


namespace carson_clouds_l20_20666

theorem carson_clouds (C D : ℕ) (h1 : D = 3 * C) (h2 : C + D = 24) : C = 6 :=
by
  sorry

end carson_clouds_l20_20666


namespace quadratic_no_real_solutions_l20_20123

theorem quadratic_no_real_solutions (k : ℝ) :
  k < -9 / 4 ↔ ∀ x : ℝ, ¬ (x^2 - 3 * x - k = 0) :=
by
  sorry

end quadratic_no_real_solutions_l20_20123


namespace jackson_running_l20_20691

variable (x : ℕ)

theorem jackson_running (h : x + 4 = 7) : x = 3 := by
  sorry

end jackson_running_l20_20691


namespace tan_a4_a12_eq_neg_sqrt3_l20_20603

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a)
          (h_sum : a 1 + a 8 + a 15 = Real.pi)

-- The main statement to prove
theorem tan_a4_a12_eq_neg_sqrt3 : 
  Real.tan (a 4 + a 12) = -Real.sqrt 3 :=
sorry

end tan_a4_a12_eq_neg_sqrt3_l20_20603


namespace f_at_10_l20_20928

variable (f : ℕ → ℝ)

-- Conditions
axiom f_1 : f 1 = 2
axiom f_relation : ∀ m n : ℕ, m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2 + 2 * n

-- Prove f(10) = 361
theorem f_at_10 : f 10 = 361 :=
by
  sorry

end f_at_10_l20_20928


namespace inequality_proof_l20_20198

theorem inequality_proof (a b c A α : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (h_sum : a + b + c = A) (hA : A ≤ 1) (hα : α > 0) :
  ( (1 / a - a) ^ α + (1 / b - b) ^ α + (1 / c - c) ^ α ) ≥ 3 * ( (3 / A) - (A / 3) ) ^ α :=
by
  sorry

end inequality_proof_l20_20198


namespace smallest_repeating_block_7_over_13_l20_20086

theorem smallest_repeating_block_7_over_13 : 
  ∃ n : ℕ, (∀ d : ℕ, d < n → 
  (∃ (q r : ℕ), r < 13 ∧ 10 ^ (d + 1) * 7 % 13 = q * 10 ^ n + r)) ∧ n = 6 := sorry

end smallest_repeating_block_7_over_13_l20_20086


namespace max_g_8_l20_20133

noncomputable def g (x : ℝ) : ℝ := sorry -- To be filled with the specific polynomial

theorem max_g_8 (g : ℝ → ℝ)
  (h_nonneg : ∀ x, 0 ≤ g x)
  (h4 : g 4 = 16)
  (h16 : g 16 = 1024) : g 8 ≤ 128 :=
sorry

end max_g_8_l20_20133


namespace largest_k_for_right_triangle_l20_20888

noncomputable def k : ℝ := (3 * Real.sqrt 2 - 4) / 2

theorem largest_k_for_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) :
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3 :=
sorry

end largest_k_for_right_triangle_l20_20888


namespace length_of_jordans_rectangle_l20_20646

theorem length_of_jordans_rectangle
  (carol_length : ℕ) (carol_width : ℕ) (jordan_width : ℕ) (equal_area : (carol_length * carol_width) = (jordan_width * 2)) :
  (2 = 120 / 60) := by
  sorry

end length_of_jordans_rectangle_l20_20646


namespace find_ice_cream_cost_l20_20623

def chapatis_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def rice_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def mixed_vegetable_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soup_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def dessert_cost (num: ℕ) (price: ℝ) : ℝ := num * price
def soft_drink_cost (num: ℕ) (price: ℝ) (discount: ℝ) : ℝ := num * price * (1 - discount)
def total_cost (chap: ℝ) (rice: ℝ) (veg: ℝ) (soup: ℝ) (dessert: ℝ) (drink: ℝ) : ℝ := chap + rice + veg + soup + dessert + drink
def total_cost_with_tax (base_cost: ℝ) (tax_rate: ℝ) : ℝ := base_cost * (1 + tax_rate)

theorem find_ice_cream_cost :
  let chapatis := chapatis_cost 16 6
  let rice := rice_cost 5 45
  let veg := mixed_vegetable_cost 7 70
  let soup := soup_cost 4 30
  let dessert := dessert_cost 3 85
  let drinks := soft_drink_cost 2 50 0.1
  let base_cost := total_cost chapatis rice veg soup dessert drinks
  let final_cost := total_cost_with_tax base_cost 0.18
  final_cost + 6 * 108.89 = 2159 := 
  by sorry

end find_ice_cream_cost_l20_20623


namespace power_relationship_l20_20030

variable (a b : ℝ)

theorem power_relationship (h : 0 < a ∧ a < b ∧ b < 2) : a^b < b^a :=
sorry

end power_relationship_l20_20030


namespace rate_per_kg_mangoes_l20_20601

theorem rate_per_kg_mangoes (kg_apples kg_mangoes total_cost rate_apples total_payment rate_mangoes : ℕ) 
  (h1 : kg_apples = 8) 
  (h2 : rate_apples = 70)
  (h3 : kg_mangoes = 9)
  (h4 : total_payment = 965) :
  rate_mangoes = 45 := 
by
  sorry

end rate_per_kg_mangoes_l20_20601


namespace delete_middle_divides_l20_20433

def digits (n : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ :=
  let a := n / 10000
  let b := (n % 10000) / 1000
  let c := (n % 1000) / 100
  let d := (n % 100) / 10
  let e := n % 10
  (a, b, c, d, e)

def delete_middle_digit (n : ℕ) : ℕ :=
  let (a, b, c, d, e) := digits n
  1000 * a + 100 * b + 10 * d + e

theorem delete_middle_divides (n : ℕ) (hn : 10000 ≤ n ∧ n < 100000) :
  (delete_middle_digit n) ∣ n :=
sorry

end delete_middle_divides_l20_20433


namespace functional_equation_solution_l20_20789

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * y * f x) :
  ∀ x : ℝ, f x = 0 := 
sorry

end functional_equation_solution_l20_20789


namespace common_integer_root_l20_20704

theorem common_integer_root (a x : ℤ) : (a * x + a = 7) ∧ (3 * x - a = 17) → a = 1 :=
by
    sorry

end common_integer_root_l20_20704


namespace boys_and_girls_in_class_l20_20471

theorem boys_and_girls_in_class (b g : ℕ) (h1 : b + g = 21) (h2 : 5 * b + 2 * g = 69) 
: b = 9 ∧ g = 12 := by
  sorry

end boys_and_girls_in_class_l20_20471


namespace karen_has_32_quarters_l20_20613

variable (k : ℕ)  -- the number of quarters Karen has

-- Define the number of quarters Christopher has
def christopher_quarters : ℕ := 64

-- Define the value of a single quarter in dollars
def quarter_value : ℚ := 0.25

-- Define the amount of money Christopher has
def christopher_money : ℚ := christopher_quarters * quarter_value

-- Define the monetary difference between Christopher and Karen
def money_difference : ℚ := 8

-- Define the amount of money Karen has
def karen_money : ℚ := christopher_money - money_difference

-- Define the number of quarters Karen has
def karen_quarters := karen_money / quarter_value

-- The theorem we need to prove
theorem karen_has_32_quarters : k = 32 :=
by
  sorry

end karen_has_32_quarters_l20_20613


namespace toll_for_18_wheel_truck_l20_20868

-- Define the number of wheels on the front axle and the other axles
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def total_wheels : ℕ := 18

-- Define the toll formula
def toll (x : ℕ) : ℝ := 3.50 + 0.50 * (x - 2)

-- Calculate the number of axles for the 18-wheel truck
def num_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the expected toll for the given number of axles
def expected_toll : ℝ := 5.00

-- State the theorem
theorem toll_for_18_wheel_truck : toll num_axles = expected_toll := by
    sorry

end toll_for_18_wheel_truck_l20_20868


namespace cubed_gt_if_gt_l20_20535

theorem cubed_gt_if_gt {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cubed_gt_if_gt_l20_20535


namespace equality_of_costs_l20_20989

variable (x : ℝ)
def C1 : ℝ := 50 + 0.35 * (x - 500)
def C2 : ℝ := 75 + 0.45 * (x - 1000)

theorem equality_of_costs : C1 x = C2 x → x = 2500 :=
by
  intro h
  sorry

end equality_of_costs_l20_20989


namespace annie_spent_on_candies_l20_20128

theorem annie_spent_on_candies : 
  ∀ (num_classmates : ℕ) (candies_per_classmate : ℕ) (candies_left : ℕ) (cost_per_candy : ℚ),
  num_classmates = 35 →
  candies_per_classmate = 2 →
  candies_left = 12 →
  cost_per_candy = 0.1 →
  (num_classmates * candies_per_classmate + candies_left) * cost_per_candy = 8.2 :=
by
  intros num_classmates candies_per_classmate candies_left cost_per_candy
         h_classmates h_candies_per_classmate h_candies_left h_cost_per_candy
  simp [h_classmates, h_candies_per_classmate, h_candies_left, h_cost_per_candy]
  sorry

end annie_spent_on_candies_l20_20128


namespace locus_of_P_l20_20491

theorem locus_of_P
  (a b x y : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (x ≠ 0 ∧ y ≠ 0))
  (h4 : x^2 / a^2 - y^2 / b^2 = 1) :
  (x / a)^2 - (y / b)^2 = ((a^2 + b^2) / (a^2 - b^2))^2 := by
  sorry

end locus_of_P_l20_20491


namespace dawson_marks_l20_20662

theorem dawson_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (M : ℕ),
  max_marks = 220 →
  passing_percentage = 30 →
  failed_by = 36 →
  M = (passing_percentage * max_marks / 100) - failed_by →
  M = 30 := by
  intros max_marks passing_percentage failed_by M h_max h_percent h_failed h_M
  rw [h_max, h_percent, h_failed] at h_M
  norm_num at h_M
  exact h_M

end dawson_marks_l20_20662


namespace tim_change_l20_20651

theorem tim_change :
  ∀ (initial_amount : ℕ) (amount_paid : ℕ),
  initial_amount = 50 →
  amount_paid = 45 →
  initial_amount - amount_paid = 5 :=
by
  intros
  sorry

end tim_change_l20_20651


namespace find_first_set_length_l20_20076

def length_of_second_set : ℤ := 20
def ratio := 5

theorem find_first_set_length (x : ℤ) (h1 : length_of_second_set = ratio * x) : x = 4 := 
sorry

end find_first_set_length_l20_20076


namespace bailing_rate_bailing_problem_l20_20328

theorem bailing_rate (distance : ℝ) (rate_in : ℝ) (sink_limit : ℝ) (speed : ℝ) : ℝ :=
  let time_to_shore := distance / speed * 60 -- convert hours to minutes
  let total_intake := rate_in * time_to_shore
  let excess_water := total_intake - sink_limit
  excess_water / time_to_shore

theorem bailing_problem : bailing_rate 2 12 40 3 = 11 := by
  sorry

end bailing_rate_bailing_problem_l20_20328


namespace sum_of_powers_divisible_by_6_l20_20760

theorem sum_of_powers_divisible_by_6 (a1 a2 a3 a4 : ℤ)
  (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) (k : ℕ) (hk : k % 2 = 1) :
  6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
sorry

end sum_of_powers_divisible_by_6_l20_20760


namespace average_age_new_students_l20_20765

theorem average_age_new_students (A : ℚ)
    (avg_original_age : ℚ := 48)
    (num_new_students : ℚ := 120)
    (new_avg_age : ℚ := 44)
    (total_students : ℚ := 160) :
    let num_original_students := total_students - num_new_students
    let total_age_original := num_original_students * avg_original_age
    let total_age_all := total_students * new_avg_age
    total_age_original + (num_new_students * A) = total_age_all → A = 42.67 := 
by
  intros
  sorry

end average_age_new_students_l20_20765


namespace evaluate_f_at_neg_three_l20_20316

def f (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_f_at_neg_three : f (-3) = -14 := by
  sorry

end evaluate_f_at_neg_three_l20_20316


namespace tommy_profit_l20_20043

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit_l20_20043


namespace f_neg_a_eq_neg_2_l20_20220

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 1

-- Given condition: f(a) = 4
variable (a : ℝ)
axiom h_f_a : f a = 4

-- We need to prove that: f(-a) = -2
theorem f_neg_a_eq_neg_2 (a : ℝ) (h_f_a : f a = 4) : f (-a) = -2 :=
by
  sorry

end f_neg_a_eq_neg_2_l20_20220


namespace mike_payments_total_months_l20_20861

-- Definitions based on conditions
def lower_rate := 295
def higher_rate := 310
def lower_payments := 5
def higher_payments := 7
def total_paid := 3615

-- The statement to prove
theorem mike_payments_total_months : lower_payments + higher_payments = 12 := by
  -- Proof goes here
  sorry

end mike_payments_total_months_l20_20861


namespace work_earnings_t_l20_20309

theorem work_earnings_t (t : ℤ) (h1 : (t + 2) * (4 * t - 4) = (4 * t - 7) * (t + 3) + 3) : t = 10 := 
by
  sorry

end work_earnings_t_l20_20309


namespace average_of_rest_l20_20301

theorem average_of_rest (A : ℝ) (total_students scoring_95 scoring_0 : ℕ) (total_avg : ℝ)
  (h_total_students : total_students = 25)
  (h_scoring_95 : scoring_95 = 3)
  (h_scoring_0 : scoring_0 = 3)
  (h_total_avg : total_avg = 45.6)
  (h_sum_eq : total_students * total_avg = 3 * 95 + 3 * 0 + (total_students - scoring_95 - scoring_0) * A) :
  A = 45 := sorry

end average_of_rest_l20_20301


namespace gabor_can_cross_l20_20858

open Real

-- Definitions based on conditions
def river_width : ℝ := 100
def total_island_perimeter : ℝ := 800
def banks_parallel : Prop := true

theorem gabor_can_cross (w : ℝ) (p : ℝ) (bp : Prop) : 
  w = river_width → 
  p = total_island_perimeter → 
  bp = banks_parallel → 
  ∃ d : ℝ, d ≤ 300 := 
by
  sorry

end gabor_can_cross_l20_20858


namespace range_of_a_l20_20105

variable {a b c d : ℝ}

theorem range_of_a (h1 : a + b + c + d = 3) (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l20_20105


namespace no_sqrt_negative_number_l20_20478

theorem no_sqrt_negative_number (a b c d : ℝ) (hA : a = (-3)^2) (hB : b = 0) (hC : c = 1/8) (hD : d = -6^3) : 
  ¬ (∃ x : ℝ, x^2 = d) :=
by
  sorry

end no_sqrt_negative_number_l20_20478


namespace find_x_l20_20557

theorem find_x (x : ℝ) (h : 40 * x - 138 = 102) : x = 6 :=
by 
  sorry

end find_x_l20_20557


namespace relationship_between_f_x1_and_f_x2_l20_20907

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

-- Conditions:
variable (h_even : ∀ x, f x = f (-x))          -- f is even
variable (h_increasing : ∀ a b, 0 < a → a < b → f a < f b)  -- f is increasing on (0, +∞)
variable (h_x1_neg : x1 < 0)                   -- x1 < 0
variable (h_x2_pos : 0 < x2)                   -- x2 > 0
variable (h_abs : |x1| > |x2|)                 -- |x1| > |x2|

-- Goal:
theorem relationship_between_f_x1_and_f_x2 : f x1 > f x2 :=
by
  sorry

end relationship_between_f_x1_and_f_x2_l20_20907


namespace inequality_bounds_l20_20578

noncomputable def f (a b A B : ℝ) (θ : ℝ) : ℝ :=
  1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)

theorem inequality_bounds (a b A B : ℝ) (h : ∀ θ : ℝ, f a b A B θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_bounds_l20_20578


namespace count_valid_integers_1_to_999_l20_20475

-- Define a function to count the valid integers
def count_valid_integers : Nat :=
  let digits := [1, 2, 6, 7, 9]
  let one_digit_count := 5
  let two_digit_count := 5 * 5
  let three_digit_count := 5 * 5 * 5
  one_digit_count + two_digit_count + three_digit_count

-- The theorem we want to prove
theorem count_valid_integers_1_to_999 : count_valid_integers = 155 := by
  sorry

end count_valid_integers_1_to_999_l20_20475


namespace remainder_is_83_l20_20723

-- Define the condition: the values for the division
def value1 : ℤ := 2021
def value2 : ℤ := 102

-- State the theorem: remainder when 2021 is divided by 102 is 83
theorem remainder_is_83 : value1 % value2 = 83 := by
  sorry

end remainder_is_83_l20_20723


namespace xyz_value_l20_20091

-- Define real numbers x, y, z
variables {x y z : ℝ}

-- Define the theorem with the given conditions and conclusion
theorem xyz_value 
  (h1 : (x + y + z) * (xy + xz + yz) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := 
sorry

end xyz_value_l20_20091


namespace find_x_l20_20536

theorem find_x (x : ℝ) (h : 1 - 1 / (1 - x) = 1 / (1 - x)) : x = -1 :=
by
  sorry

end find_x_l20_20536


namespace slope_of_line_inclination_l20_20395

theorem slope_of_line_inclination (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 180) 
  (h3 : Real.tan (α * Real.pi / 180) = Real.sqrt 3 / 3) : α = 30 :=
by
  sorry

end slope_of_line_inclination_l20_20395


namespace largest_value_WY_cyclic_quadrilateral_l20_20070

theorem largest_value_WY_cyclic_quadrilateral :
  ∃ WZ ZX ZY YW : ℕ, 
    WZ ≠ ZX ∧ WZ ≠ ZY ∧ WZ ≠ YW ∧ ZX ≠ ZY ∧ ZX ≠ YW ∧ ZY ≠ YW ∧ 
    WZ < 20 ∧ ZX < 20 ∧ ZY < 20 ∧ YW < 20 ∧ 
    WZ * ZY = ZX * YW ∧
    (∀ WY', (∃ WY : ℕ, WY' < WY → WY <= 19 )) :=
sorry

end largest_value_WY_cyclic_quadrilateral_l20_20070


namespace parallel_and_equidistant_line_l20_20905

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 6 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 3 = 0

-- Define the desired property: a line parallel to line1 and line2, and equidistant from both
theorem parallel_and_equidistant_line :
  ∃ b : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + b = 0) ∧
  (|-6 - b| / Real.sqrt (9 + 4) = |-3/2 - b| / Real.sqrt (9 + 4)) →
  (12 * x + 8 * y - 15 = 0) :=
by
  sorry

end parallel_and_equidistant_line_l20_20905


namespace exists_convex_polygon_diagonals_l20_20027

theorem exists_convex_polygon_diagonals :
  ∃ n : ℕ, n * (n - 3) / 2 = 54 :=
by
  sorry

end exists_convex_polygon_diagonals_l20_20027


namespace union_of_P_and_Q_l20_20056

noncomputable def P : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_of_P_and_Q :
  P ∪ Q = {x | -1 < x ∧ x < 2} :=
sorry

end union_of_P_and_Q_l20_20056


namespace total_arrangements_excluding_zhang_for_shooting_event_l20_20565

theorem total_arrangements_excluding_zhang_for_shooting_event
  (students : Fin 5) 
  (events : Fin 3)
  (shooting : events ≠ 0) : 
  ∃ arrangements, arrangements = 48 := 
sorry

end total_arrangements_excluding_zhang_for_shooting_event_l20_20565


namespace algebraic_identity_l20_20345

theorem algebraic_identity (x y : ℝ) (h₁ : x * y = 4) (h₂ : x - y = 5) : 
  x^2 + 5 * x * y + y^2 = 53 := 
by 
  sorry

end algebraic_identity_l20_20345


namespace original_cost_proof_l20_20245

/-!
# Prove that the original cost of the yearly subscription to professional magazines is $940.
# Given conditions:
# 1. The company must make a 20% cut in the magazine budget.
# 2. After the cut, the company will spend $752.
-/

theorem original_cost_proof (x : ℝ)
  (h1 : 0.80 * x = 752) :
  x = 940 :=
by
  sorry

end original_cost_proof_l20_20245


namespace tan_theta_value_l20_20278

open Real

theorem tan_theta_value
  (theta : ℝ)
  (h_quad : 3 * pi / 2 < theta ∧ theta < 2 * pi)
  (h_sin : sin theta = -sqrt 6 / 3) :
  tan theta = -sqrt 2 := by
  sorry

end tan_theta_value_l20_20278


namespace kitty_cleaning_weeks_l20_20550

def time_spent_per_week (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust_furniture: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust_furniture

def total_weeks (total_time: ℕ) (time_per_week: ℕ) : ℕ :=
  total_time / time_per_week

theorem kitty_cleaning_weeks
  (pick_up_time : ℕ := 5)
  (vacuum_time : ℕ := 20)
  (clean_windows_time : ℕ := 15)
  (dust_furniture_time : ℕ := 10)
  (total_cleaning_time : ℕ := 200)
  : total_weeks total_cleaning_time (time_spent_per_week pick_up_time vacuum_time clean_windows_time dust_furniture_time) = 4 :=
by
  sorry

end kitty_cleaning_weeks_l20_20550


namespace sum_of_21st_set_l20_20545

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

def first_element_of_set (n : ℕ) : ℕ := triangular_number n - n + 1

def sum_of_elements_in_set (n : ℕ) : ℕ := 
  n * ((first_element_of_set n + triangular_number n) / 2)

theorem sum_of_21st_set : sum_of_elements_in_set 21 = 4641 := by 
  sorry

end sum_of_21st_set_l20_20545


namespace percentage_blue_and_red_l20_20764

theorem percentage_blue_and_red (F : ℕ) (h_even: F % 2 = 0)
  (h1: ∃ C, 50 / 100 * C = F / 2)
  (h2: ∃ C, 60 / 100 * C = F / 2)
  (h3: ∃ C, 40 / 100 * C = F / 2) :
  ∃ C, (50 / 100 * C + 60 / 100 * C - 100 / 100 * C) = 10 / 100 * C :=
sorry

end percentage_blue_and_red_l20_20764


namespace solve_for_x_l20_20459

theorem solve_for_x (x : ℚ) (h : x > 0) (hx : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
by 
  sorry

end solve_for_x_l20_20459


namespace number_of_smaller_cubes_in_larger_cube_l20_20926

-- Defining the conditions
def volume_large_cube : ℝ := 125
def volume_small_cube : ℝ := 1
def surface_area_difference : ℝ := 600

-- Translating the question into a math proof problem
theorem number_of_smaller_cubes_in_larger_cube : 
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3) ^ 2) = surface_area_difference :=
by
  sorry

end number_of_smaller_cubes_in_larger_cube_l20_20926


namespace find_number_l20_20671

theorem find_number (x : ℕ) (h : (x / 5) - 154 = 6) : x = 800 := by
  sorry

end find_number_l20_20671


namespace value_of_f_csc_squared_l20_20144

noncomputable def f (x : ℝ) : ℝ := if x ≠ 0 ∧ x ≠ 1 then 1 / x else 0

lemma csc_sq_identity (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  (f (x / (x - 1)) = 1 / x) := 
  by sorry

theorem value_of_f_csc_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π / 2) :
  f ((1 / (Real.sin t) ^ 2)) = - (Real.cos t) ^ 2 :=
  by sorry

end value_of_f_csc_squared_l20_20144


namespace fraction_a_over_b_l20_20205

theorem fraction_a_over_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_a_over_b_l20_20205


namespace arrange_books_l20_20381

-- Given conditions
def math_books_count := 4
def history_books_count := 6

-- Question: How many ways can the books be arranged given the conditions?
theorem arrange_books (math_books_count history_books_count : ℕ) :
  math_books_count = 4 → 
  history_books_count = 6 →
  ∃ ways : ℕ, ways = 51840 :=
by
  sorry

end arrange_books_l20_20381


namespace rebecca_groups_eq_l20_20291

-- Definitions
def total_eggs : ℕ := 15
def eggs_per_group : ℕ := 5
def expected_groups : ℕ := 3

-- Theorem to prove
theorem rebecca_groups_eq :
  total_eggs / eggs_per_group = expected_groups :=
by
  sorry

end rebecca_groups_eq_l20_20291


namespace find_x_values_l20_20080

noncomputable def condition (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4

theorem find_x_values : 
  {x : ℝ | condition  x} = {x : ℝ | x < -3} ∪ {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end find_x_values_l20_20080


namespace algebraic_expression_equivalence_l20_20775

theorem algebraic_expression_equivalence (x : ℝ) : 
  x^2 - 6*x + 10 = (x - 3)^2 + 1 := 
by 
  sorry

end algebraic_expression_equivalence_l20_20775


namespace function_value_l20_20637

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value_l20_20637


namespace minimum_value_shifted_function_l20_20271

def f (x a : ℝ) : ℝ := x^2 + 4 * x + 7 - a

theorem minimum_value_shifted_function (a : ℝ) (h : ∃ x, f x a = 2) :
  ∃ y, (∃ x, y = f (x - 2015) a) ∧ y = 2 :=
sorry

end minimum_value_shifted_function_l20_20271


namespace min_number_of_girls_l20_20272

theorem min_number_of_girls (d : ℕ) (students : ℕ) (boys : ℕ → ℕ) : 
  students = 20 ∧ ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → boys i ≠ boys j ∨ boys j ≠ boys k ∨ boys i ≠ boys k → d = 6 :=
by
  sorry

end min_number_of_girls_l20_20272


namespace calculate_value_l20_20455

def a : ℕ := 2500
def b : ℕ := 2109
def d : ℕ := 64

theorem calculate_value : (a - b) ^ 2 / d = 2389 := by
  sorry

end calculate_value_l20_20455


namespace price_of_75_cans_l20_20138

/-- The price of 75 cans of a certain brand of soda purchased in 24-can cases,
    given the regular price per can is $0.15 and a 10% discount is applied when
    purchased in 24-can cases, is $10.125.
-/
theorem price_of_75_cans (regular_price : ℝ) (discount : ℝ) (cases_needed : ℕ) (remaining_cans : ℕ) 
  (discounted_price : ℝ) (total_price : ℝ) :
  regular_price = 0.15 →
  discount = 0.10 →
  discounted_price = regular_price - (discount * regular_price) →
  cases_needed = 75 / 24 ∧ remaining_cans = 75 % 24 →
  total_price = (cases_needed * 24 + remaining_cans) * discounted_price →
  total_price = 10.125 :=
by
  sorry

end price_of_75_cans_l20_20138


namespace jordan_wins_two_games_l20_20892

theorem jordan_wins_two_games 
  (Peter_wins : ℕ) 
  (Peter_losses : ℕ)
  (Emma_wins : ℕ) 
  (Emma_losses : ℕ)
  (Jordan_losses : ℕ) 
  (hPeter : Peter_wins = 5)
  (hPeterL : Peter_losses = 4)
  (hEmma : Emma_wins = 4)
  (hEmmaL : Emma_losses = 5)
  (hJordanL : Jordan_losses = 2) : ∃ (J : ℕ), J = 2 :=
by
  -- The proof will go here
  sorry

end jordan_wins_two_games_l20_20892


namespace black_ink_cost_l20_20702

theorem black_ink_cost (B : ℕ) 
  (h1 : 2 * B + 3 * 15 + 2 * 13 = 50 + 43) : B = 11 :=
by
  sorry

end black_ink_cost_l20_20702


namespace contradiction_assumption_l20_20038

theorem contradiction_assumption (a : ℝ) (h : a < |a|) : ¬(a ≥ 0) :=
by 
  sorry

end contradiction_assumption_l20_20038


namespace average_salary_of_all_workers_l20_20497

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l20_20497


namespace product_representation_count_l20_20018

theorem product_representation_count :
  let n := 1000000
  let distinct_ways := 139
  (∃ (a b c d e f : ℕ), 2^(a+b+c) * 5^(d+e+f) = n ∧ 
    a + b + c = 6 ∧ d + e + f = 6 ) → 
    139 = distinct_ways := 
by {
  sorry
}

end product_representation_count_l20_20018


namespace overtime_hours_l20_20585

theorem overtime_hours (x y : ℕ) 
  (h1 : 60 * x + 90 * y = 3240) 
  (h2 : x + y = 50) : 
  y = 8 :=
by
  sorry

end overtime_hours_l20_20585


namespace solution_set_of_inequality_l20_20315

theorem solution_set_of_inequality : 
  { x : ℝ | x^2 - 3*x - 4 < 0 } = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end solution_set_of_inequality_l20_20315


namespace minimum_rectangles_needed_l20_20467

def type1_corners := 12
def type2_corners := 12
def group_size := 3

theorem minimum_rectangles_needed (cover_type1: ℕ) (cover_type2: ℕ)
  (type1_corners coverable_by_one: ℕ) (type2_groups_num: ℕ) :
  type1_corners = 12 → type2_corners = 12 → type2_groups_num = 4 →
  group_size = 3 → cover_type1 + cover_type2 = 12 :=
by
  intros h1 h2 h3 h4 
  sorry

end minimum_rectangles_needed_l20_20467


namespace product_eq_5832_l20_20667

-- Define the integers A, B, C, D that satisfy the given conditions.
variables (A B C D : ℕ)

-- Define the conditions in the problem.
def conditions : Prop :=
  (A + B + C + D = 48) ∧
  (A + 3 = B - 3) ∧
  (A + 3 = C * 3) ∧
  (A + 3 = D / 3)

-- State the final theorem we want to prove.
theorem product_eq_5832 : conditions A B C D → A * B * C * D = 5832 :=
by 
  sorry

end product_eq_5832_l20_20667


namespace one_eighth_percent_of_160_plus_half_l20_20923

theorem one_eighth_percent_of_160_plus_half :
  ((1 / 8) / 100 * 160) + 0.5 = 0.7 :=
  sorry

end one_eighth_percent_of_160_plus_half_l20_20923


namespace find_natural_number_l20_20014

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_natural_number (n : ℕ) : sum_of_digits (2 ^ n) = 5 ↔ n = 5 := by
  sorry

end find_natural_number_l20_20014


namespace find_number_l20_20866

theorem find_number (n : ℝ) :
  (n + 2 * 1.5)^5 = (1 + 3 * 1.5)^4 → n = 0.72 :=
sorry

end find_number_l20_20866


namespace divisible_by_120_l20_20416

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end divisible_by_120_l20_20416


namespace equilibrium_temperature_l20_20737

theorem equilibrium_temperature 
  (c_B : ℝ) (c_m : ℝ)
  (m_B : ℝ) (m_m : ℝ)
  (T₁ : ℝ) (T_eq₁ : ℝ) (T_metal : ℝ) 
  (T_eq₂ : ℝ)
  (h₁ : T₁ = 80)
  (h₂ : T_eq₁ = 60)
  (h₃ : T_metal = 20)
  (h₄ : T₂ = 50)
  (h_ratio : c_B * m_B = 2 * c_m * m_m) :
  T_eq₂ = 50 :=
by
  sorry

end equilibrium_temperature_l20_20737


namespace possible_values_f2001_l20_20442

noncomputable def f : ℕ → ℝ := sorry

lemma functional_equation (a b d : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : d = Nat.gcd a b) :
  f (a * b) = f d * (f (a / d) + f (b / d)) :=
sorry

theorem possible_values_f2001 :
  f 2001 = 0 ∨ f 2001 = 1 / 2 :=
sorry

end possible_values_f2001_l20_20442


namespace sine_of_smaller_angle_and_k_domain_l20_20940

theorem sine_of_smaller_angle_and_k_domain (α : ℝ) (k : ℝ) (AD : ℝ) (h0 : 1 < k) 
  (h1 : CD = AD * Real.tan (2 * α)) (h2 : BD = AD * Real.tan α) 
  (h3 : k = CD / BD) :
  k > 2 ∧ Real.sin (Real.pi / 2 - 2 * α) = 1 / (k - 1) := by
  sorry

end sine_of_smaller_angle_and_k_domain_l20_20940


namespace binomial_expansion_coefficient_l20_20193

theorem binomial_expansion_coefficient :
  let a_0 : ℚ := (1 + 2 * (0:ℚ))^5
  (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_3 = 80 :=
by 
  sorry

end binomial_expansion_coefficient_l20_20193


namespace division_multiplication_relation_l20_20568

theorem division_multiplication_relation (h: 7650 / 306 = 25) :
  25 * 306 = 7650 ∧ 7650 / 25 = 306 := 
by 
  sorry

end division_multiplication_relation_l20_20568


namespace sum_first_12_terms_l20_20763

variable (S : ℕ → ℝ)

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  S n

theorem sum_first_12_terms (h₁ : sum_of_first_n_terms 4 = 30) (h₂ : sum_of_first_n_terms 8 = 100) :
  sum_of_first_n_terms 12 = 210 := 
sorry

end sum_first_12_terms_l20_20763


namespace geometric_sum_4500_l20_20097

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l20_20097


namespace unique_k_for_prime_roots_of_quadratic_l20_20348

/-- Function to check primality of a natural number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Theorem statement with the given conditions -/
theorem unique_k_for_prime_roots_of_quadratic :
  ∃! k : ℕ, ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 50 ∧ p * q = k :=
sorry

end unique_k_for_prime_roots_of_quadratic_l20_20348


namespace greatest_possible_n_l20_20258

theorem greatest_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 8100) : n ≤ 8 :=
by
  -- Intentionally left uncommented.
  sorry

end greatest_possible_n_l20_20258


namespace unoccupied_volume_l20_20369

/--
Given:
1. Three congruent cones, each with a radius of 8 cm and a height of 8 cm.
2. The cones are enclosed within a cylinder such that the bases of two cones are at each base of the cylinder, and one cone is inverted in the middle touching the other two cones at their vertices.
3. The height of the cylinder is 16 cm.

Prove:
The volume of the cylinder not occupied by the cones is 512π cubic cm.
-/
theorem unoccupied_volume 
  (r h : ℝ) 
  (hr : r = 8) 
  (hh_cone : h = 8) 
  (hh_cyl : h_cyl = 16) 
  : (π * r^2 * h_cyl) - (3 * (1/3 * π * r^2 * h)) = 512 * π := 
by 
  sorry

end unoccupied_volume_l20_20369


namespace brayan_hourly_coffee_l20_20576

theorem brayan_hourly_coffee (I B : ℕ) (h1 : B = 2 * I) (h2 : I + B = 30) : B / 5 = 4 :=
by
  sorry

end brayan_hourly_coffee_l20_20576


namespace division_pow_zero_l20_20509

theorem division_pow_zero (a b : ℝ) (hb : b ≠ 0) : ((a / b) ^ 0 = (1 : ℝ)) :=
by
  sorry

end division_pow_zero_l20_20509


namespace number_of_dimes_paid_l20_20501

theorem number_of_dimes_paid (cost_in_dollars : ℕ) (value_of_dime_in_cents : ℕ) (value_of_dollar_in_cents : ℕ) 
  (h_cost : cost_in_dollars = 9) (h_dime : value_of_dime_in_cents = 10) (h_dollar : value_of_dollar_in_cents = 100) : 
  (cost_in_dollars * value_of_dollar_in_cents) / value_of_dime_in_cents = 90 := by
  -- Proof to be provided here
  sorry

end number_of_dimes_paid_l20_20501


namespace number_of_valid_m_l20_20060

def is_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  (Qx - Px) * (Qx - Px) + (Qy - Py) * (Qy - Py) + (Rx - Qx) * (Rx - Qx) + (Ry - Qy) * (Ry - Qy) ==
  (Px - Rx) * (Px - Rx) + (Py - Ry) * (Py - Ry) + 2 * ((Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy))

def legs_parallel_to_axes (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  Px = Qx ∨ Px = Rx ∨ Qx = Rx ∧ Py = Qy ∨ Py = Ry ∨ Qy = Ry

def medians_condition (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  let M_PQ := ((Px + Qx) / 2, (Py + Qy) / 2);
  let M_PR := ((Px + Rx) / 2, (Py + Ry) / 2);
  (M_PQ.2 = 3 * M_PQ.1 + 1) ∧ (M_PR.2 = 2)

theorem number_of_valid_m (a b c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (P := (a, b)) (Q := (a, b+2*c)) (R := (a-2*d, b)) :
  is_right_triangle P Q R →
  legs_parallel_to_axes P Q R →
  medians_condition P Q R →
  ∃ m, m = 1 :=
sorry

end number_of_valid_m_l20_20060


namespace original_rectangle_perimeter_l20_20690

theorem original_rectangle_perimeter (l w : ℝ) (h1 : w = l / 2)
  (h2 : 2 * (w + l / 3) = 40) : 2 * l + 2 * w = 72 :=
by
  sorry

end original_rectangle_perimeter_l20_20690


namespace roots_of_quadratic_l20_20500

open Real

theorem roots_of_quadratic (r s : ℝ) (h1 : r + s = 2 * sqrt 3) (h2 : r * s = 2) :
  r^6 + s^6 = 3104 :=
sorry

end roots_of_quadratic_l20_20500


namespace garden_perimeter_l20_20462

-- We are given:
variables (a b : ℝ)
variables (h1 : b = 3 * a)
variables (h2 : a^2 + b^2 = 34^2)
variables (h3 : a * b = 240)

-- We must prove:
theorem garden_perimeter (h4 : a^2 + 9 * a^2 = 1156) (h5 : 10 * a^2 = 1156) (h6 : a^2 = 115.6) 
  (h7 : 3 * a^2 = 240) (h8 : a^2 = 80) :
  2 * (a + b) = 72 := 
by
  sorry

end garden_perimeter_l20_20462


namespace billboard_dimensions_l20_20859

theorem billboard_dimensions (photo_width_cm : ℕ) (photo_length_dm : ℕ) (billboard_area_m2 : ℕ)
  (h1 : photo_width_cm = 30) (h2 : photo_length_dm = 4) (h3 : billboard_area_m2 = 48) :
  ∃ photo_length_cm : ℕ, photo_length_cm = 40 ∧
  ∃ k : ℕ, k = 20 ∧
  ∃ billboard_width_m billboard_length_m : ℕ,
    billboard_width_m = photo_width_cm * k / 100 ∧ 
    billboard_length_m = photo_length_cm * k / 100 ∧ 
    billboard_width_m = 6 ∧ 
    billboard_length_m = 8 := by
  sorry

end billboard_dimensions_l20_20859


namespace hexagonal_prism_min_cut_l20_20621

-- We formulate the problem conditions and the desired proof
def minimum_edges_to_cut (total_edges : ℕ) (uncut_edges : ℕ) : ℕ :=
  total_edges - uncut_edges

theorem hexagonal_prism_min_cut :
  minimum_edges_to_cut 18 7 = 11 :=
by
  sorry

end hexagonal_prism_min_cut_l20_20621


namespace train_length_proof_l20_20628

-- Definitions based on the conditions given in the problem
def speed_km_per_hr := 45 -- speed of the train in km/hr
def time_seconds := 60 -- time taken to pass the platform in seconds
def length_platform_m := 390 -- length of the platform in meters

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Calculate the speed in m/s
def speed_m_per_s : ℕ := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance covered by the train while passing the platform
def total_distance_m : ℕ := speed_m_per_s * time_seconds

-- Total distance is the sum of the length of the train and the length of the platform
def length_train_m := total_distance_m - length_platform_m

-- The statement to prove the length of the train
theorem train_length_proof : length_train_m = 360 :=
by
  sorry

end train_length_proof_l20_20628


namespace solution_verification_l20_20020

-- Define the differential equation
def diff_eq (y y' y'': ℝ → ℝ) : Prop :=
  ∀ x, y'' x - 4 * y' x + 5 * y x = 2 * Real.cos x + 6 * Real.sin x

-- General solution form
def general_solution (C₁ C₂ : ℝ) (y: ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.exp (2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) + Real.cos x + 1/2 * Real.sin x

-- Proof problem statement
theorem solution_verification (C₁ C₂ : ℝ) (y y' y'': ℝ → ℝ) :
  (∀ x, y' x = deriv y x) →
  (∀ x, y'' x = deriv (deriv y) x) →
  diff_eq y y' y'' →
  general_solution C₁ C₂ y :=
by
  intros h1 h2 h3
  sorry

end solution_verification_l20_20020


namespace spongebob_earnings_l20_20230

-- Define the conditions as variables and constants
def burgers_sold : ℕ := 30
def price_per_burger : ℝ := 2
def fries_sold : ℕ := 12
def price_per_fries : ℝ := 1.5

-- Define total earnings calculation
def earnings_from_burgers := burgers_sold * price_per_burger
def earnings_from_fries := fries_sold * price_per_fries
def total_earnings := earnings_from_burgers + earnings_from_fries

-- State the theorem we need to prove
theorem spongebob_earnings :
  total_earnings = 78 := by
    sorry

end spongebob_earnings_l20_20230


namespace ratio_of_areas_l20_20234

noncomputable def area (a b : ℕ) : ℚ := (a * b : ℚ) / 2

theorem ratio_of_areas :
  let GHI := (7, 24, 25)
  let JKL := (9, 40, 41)
  area 7 24 / area 9 40 = (7 : ℚ) / 15 :=
by
  sorry

end ratio_of_areas_l20_20234


namespace student_percentage_in_math_l20_20668

theorem student_percentage_in_math (M H T : ℝ) (H_his : H = 84) (H_third : T = 69) (H_avg : (M + H + T) / 3 = 75) : M = 72 :=
by
  sorry

end student_percentage_in_math_l20_20668


namespace rooms_in_second_wing_each_hall_l20_20626

theorem rooms_in_second_wing_each_hall
  (floors_first_wing : ℕ)
  (halls_per_floor_first_wing : ℕ)
  (rooms_per_hall_first_wing : ℕ)
  (floors_second_wing : ℕ)
  (halls_per_floor_second_wing : ℕ)
  (total_rooms : ℕ)
  (h1 : floors_first_wing = 9)
  (h2 : halls_per_floor_first_wing = 6)
  (h3 : rooms_per_hall_first_wing = 32)
  (h4 : floors_second_wing = 7)
  (h5 : halls_per_floor_second_wing = 9)
  (h6 : total_rooms = 4248) :
  (total_rooms - floors_first_wing * halls_per_floor_first_wing * rooms_per_hall_first_wing) / 
  (floors_second_wing * halls_per_floor_second_wing) = 40 :=
  by {
  sorry
}

end rooms_in_second_wing_each_hall_l20_20626


namespace three_f_x_expression_l20_20029

variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f (3 * x) = 3 / (3 + 2 * x))

theorem three_f_x_expression (x : ℝ) (hx : x > 0) : 3 * f x = 27 / (9 + 2 * x) :=
by sorry

end three_f_x_expression_l20_20029


namespace polynomial_product_is_square_l20_20402

theorem polynomial_product_is_square (x a : ℝ) :
  (x + a) * (x + 2 * a) * (x + 3 * a) * (x + 4 * a) + a^4 = (x^2 + 5 * a * x + 5 * a^2)^2 :=
by
  sorry

end polynomial_product_is_square_l20_20402


namespace tetrahedron_edge_square_sum_l20_20937

variable (A B C D : Point)
variable (AB AC AD BC BD CD : ℝ) -- Lengths of the edges
variable (m₁ m₂ m₃ : ℝ) -- Distances between the midpoints of the opposite edges

theorem tetrahedron_edge_square_sum:
  (AB ^ 2 + AC ^ 2 + AD ^ 2 + BC ^ 2 + BD ^ 2 + CD ^ 2) =
  4 * (m₁ ^ 2 + m₂ ^ 2 + m₃ ^ 2) :=
  sorry

end tetrahedron_edge_square_sum_l20_20937


namespace number_of_students_l20_20985

theorem number_of_students (S N : ℕ) (h1 : S = 15 * N)
                           (h2 : (8 * 14) = 112)
                           (h3 : (6 * 16) = 96)
                           (h4 : 17 = 17)
                           (h5 : S = 225) : N = 15 :=
by sorry

end number_of_students_l20_20985


namespace nate_distance_after_resting_l20_20818

variables (length_of_field total_distance : ℕ)

def distance_before_resting (length_of_field : ℕ) := 4 * length_of_field

def distance_after_resting (total_distance length_of_field : ℕ) : ℕ := 
  total_distance - distance_before_resting length_of_field

theorem nate_distance_after_resting
  (length_of_field_val : length_of_field = 168)
  (total_distance_val : total_distance = 1172) :
  distance_after_resting total_distance length_of_field = 500 :=
by
  -- Proof goes here
  sorry

end nate_distance_after_resting_l20_20818


namespace rotation_problem_l20_20388

-- Define the coordinates of the points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles with given vertices
def P : Point := {x := 0, y := 0}
def Q : Point := {x := 0, y := 13}
def R : Point := {x := 17, y := 0}

def P' : Point := {x := 34, y := 26}
def Q' : Point := {x := 46, y := 26}
def R' : Point := {x := 34, y := 0}

-- Rotation parameters
variables (n : ℝ) (x y : ℝ) (h₀ : 0 < n) (h₁ : n < 180)

-- The mathematical proof problem
theorem rotation_problem :
  n + x + y = 180 := by
  sorry

end rotation_problem_l20_20388


namespace sasha_tree_planting_cost_l20_20121

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l20_20121


namespace find_sequence_l20_20696

def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 
    a (n + 1) = (a n * a (n - 1)) / 
               Real.sqrt (a n^2 + a (n - 1)^2 + 1)

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 5

def sequence_property (F : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = Real.sqrt (1 / (Real.exp (F n * Real.log 10) - 1))

theorem find_sequence (a : ℕ → ℝ) (F : ℕ → ℝ) :
  initial_conditions a →
  recurrence_relation a →
  (∀ n : ℕ, n ≥ 2 →
    F (n + 1) = F n + F (n - 1)) →
  sequence_property F a :=
by
  intros h_initial h_recur h_F
  sorry

end find_sequence_l20_20696


namespace at_least_one_inequality_holds_l20_20948

theorem at_least_one_inequality_holds
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l20_20948


namespace sum_first_twelve_arithmetic_divisible_by_6_l20_20026

theorem sum_first_twelve_arithmetic_divisible_by_6 
  (a d : ℕ) (h1 : a > 0) (h2 : d > 0) : 
  6 ∣ (12 * a + 66 * d) := 
by
  sorry

end sum_first_twelve_arithmetic_divisible_by_6_l20_20026


namespace right_triangle_inscribed_circle_inequality_l20_20533

theorem right_triangle_inscribed_circle_inequality 
  {a b c r : ℝ} (h : a^2 + b^2 = c^2) (hr : r = (a + b - c) / 2) : 
  r ≤ (c / 2) * (Real.sqrt 2 - 1) :=
sorry

end right_triangle_inscribed_circle_inequality_l20_20533


namespace yellow_fraction_after_changes_l20_20164

theorem yellow_fraction_after_changes (y : ℕ) :
  let green_initial := (4 / 7 : ℚ) * y
  let yellow_initial := (3 / 7 : ℚ) * y
  let yellow_new := 3 * yellow_initial
  let green_new := green_initial + (1 / 2) * green_initial
  let total_new := green_new + yellow_new
  yellow_new / total_new = (3 / 5 : ℚ) :=
by
  sorry

end yellow_fraction_after_changes_l20_20164


namespace coconut_trees_per_sqm_l20_20913

def farm_area : ℕ := 20
def harvests : ℕ := 2
def total_earnings : ℝ := 240
def coconut_price : ℝ := 0.50
def coconuts_per_tree : ℕ := 6

theorem coconut_trees_per_sqm : 
  let total_coconuts := total_earnings / coconut_price / harvests
  let total_trees := total_coconuts / coconuts_per_tree 
  let trees_per_sqm := total_trees / farm_area 
  trees_per_sqm = 2 :=
by
  sorry

end coconut_trees_per_sqm_l20_20913


namespace missing_number_l20_20836

theorem missing_number (x : ℝ) : (306 / x) * 15 + 270 = 405 ↔ x = 34 := 
by
  sorry

end missing_number_l20_20836


namespace dot_but_not_straight_line_l20_20333

theorem dot_but_not_straight_line :
  let total := 80
  let D_n_S := 28
  let S_n_D := 47
  ∃ (D : ℕ), D - D_n_S = 5 ∧ D + S_n_D = total :=
by
  sorry

end dot_but_not_straight_line_l20_20333


namespace find_ending_number_l20_20492

theorem find_ending_number (N : ℕ) :
  (∃ k : ℕ, N = 3 * k) ∧ (∀ x,  40 < x ∧ x ≤ N → x % 3 = 0) ∧ (∃ avg, avg = (N + 42) / 2 ∧ avg = 60) → N = 78 :=
by
  sorry

end find_ending_number_l20_20492


namespace problem_solution_l20_20547

theorem problem_solution {a b : ℝ} (h : a * b + b^2 = 12) : (a + b)^2 - (a + b) * (a - b) = 24 :=
by sorry

end problem_solution_l20_20547


namespace amount_C_l20_20799

theorem amount_C (A_amt B_amt C_amt : ℚ)
  (h1 : A_amt + B_amt + C_amt = 527)
  (h2 : A_amt = (2 / 3) * B_amt)
  (h3 : B_amt = (1 / 4) * C_amt) :
  C_amt = 372 :=
sorry

end amount_C_l20_20799


namespace complex_exp_cos_l20_20790

theorem complex_exp_cos (z : ℂ) (α : ℂ) (n : ℕ) (h : z + z⁻¹ = 2 * Complex.cos α) : 
  z^n + z⁻¹^n = 2 * Complex.cos (n * α) :=
by
  sorry

end complex_exp_cos_l20_20790
