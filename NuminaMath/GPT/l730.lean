import Mathlib

namespace NUMINAMATH_GPT_equivalent_single_increase_l730_73063

-- Defining the initial price of the mobile
variable (P : ℝ)
-- Condition stating the price after a 40% increase
def increased_price := 1.40 * P
-- Condition stating the new price after a further 15% decrease
def final_price := 0.85 * increased_price P

-- The mathematically equivalent statement to prove
theorem equivalent_single_increase:
  final_price P = 1.19 * P :=
sorry

end NUMINAMATH_GPT_equivalent_single_increase_l730_73063


namespace NUMINAMATH_GPT_smallest_n_for_sum_is_24_l730_73098

theorem smallest_n_for_sum_is_24 :
  ∃ (n : ℕ), (0 < n) ∧ 
    (∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n = k) ∧
    ∀ (m : ℕ), ((0 < m) ∧ 
                (∃ (k' : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / m = k') → n ≤ m) := sorry

end NUMINAMATH_GPT_smallest_n_for_sum_is_24_l730_73098


namespace NUMINAMATH_GPT_Vasya_numbers_l730_73031

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end NUMINAMATH_GPT_Vasya_numbers_l730_73031


namespace NUMINAMATH_GPT_sum_first_n_arithmetic_sequence_l730_73040

theorem sum_first_n_arithmetic_sequence (a1 d : ℝ) (S : ℕ → ℝ) :
  (S 3 + S 6 = 18) → 
  S 3 = 3 * a1 + 3 * d → 
  S 6 = 6 * a1 + 15 * d → 
  S 5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_n_arithmetic_sequence_l730_73040


namespace NUMINAMATH_GPT_compare_decimal_to_fraction_l730_73091

theorem compare_decimal_to_fraction : (0.650 - (1 / 8) = 0.525) :=
by
  /- We need to prove that 0.650 - 1/8 = 0.525 -/
  sorry

end NUMINAMATH_GPT_compare_decimal_to_fraction_l730_73091


namespace NUMINAMATH_GPT_complement_U_M_correct_l730_73071

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 4 * x + 3 = 0}
def complement_U_M : Set ℕ := U \ M

theorem complement_U_M_correct : complement_U_M = {2, 4} :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_complement_U_M_correct_l730_73071


namespace NUMINAMATH_GPT_mark_more_than_kate_by_100_l730_73044

variable (Pat Kate Mark : ℕ)
axiom total_hours : Pat + Kate + Mark = 180
axiom pat_twice_as_kate : Pat = 2 * Kate
axiom pat_third_of_mark : Pat = Mark / 3

theorem mark_more_than_kate_by_100 : Mark - Kate = 100 :=
by
  sorry

end NUMINAMATH_GPT_mark_more_than_kate_by_100_l730_73044


namespace NUMINAMATH_GPT_min_value_of_f_l730_73052

noncomputable def f (x : ℝ) : ℝ := max (2 * x + 1) (5 - x)

theorem min_value_of_f : ∃ y, (∀ x : ℝ, f x ≥ y) ∧ y = 11 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_f_l730_73052


namespace NUMINAMATH_GPT_evaluate_expression_l730_73018

theorem evaluate_expression : 
  ( (5 ^ 2014) ^ 2 - (5 ^ 2012) ^ 2 ) / ( (5 ^ 2013) ^ 2 - (5 ^ 2011) ^ 2 ) = 25 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l730_73018


namespace NUMINAMATH_GPT_student_weekly_allowance_l730_73076

theorem student_weekly_allowance (A : ℝ) 
  (h1 : ∃ spent_arcade, spent_arcade = (3 / 5) * A)
  (h2 : ∃ spent_toy, spent_toy = (1 / 3) * ((2 / 5) * A))
  (h3 : ∃ spent_candy, spent_candy = 0.60)
  (h4 : ∃ remaining_after_toy, remaining_after_toy = ((6 / 15) * A - (2 / 15) * A))
  (h5 : remaining_after_toy = 0.60) : 
  A = 2.25 := by
  sorry

end NUMINAMATH_GPT_student_weekly_allowance_l730_73076


namespace NUMINAMATH_GPT_sum_of_largest_and_smallest_four_digit_numbers_is_11990_l730_73077

theorem sum_of_largest_and_smallest_four_digit_numbers_is_11990 (A B C D : ℕ) 
    (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D)
    (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
    (h_eq : 1001 * A + 110 * B + 110 * C + 1001 * D = 11990) :
    (min (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 1999) ∧
    (max (1000 * A + 100 * B + 10 * C + D) (1000 * D + 100 * C + 10 * B + A) = 9991) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_largest_and_smallest_four_digit_numbers_is_11990_l730_73077


namespace NUMINAMATH_GPT_total_cards_l730_73096

theorem total_cards (Brenda_card Janet_card Mara_card Michelle_card : ℕ)
  (h1 : Janet_card = Brenda_card + 9)
  (h2 : Mara_card = 7 * Janet_card / 4)
  (h3 : Michelle_card = 4 * Mara_card / 5)
  (h4 : Mara_card = 210 - 60) :
  Janet_card + Brenda_card + Mara_card + Michelle_card = 432 :=
by
  sorry

end NUMINAMATH_GPT_total_cards_l730_73096


namespace NUMINAMATH_GPT_largest_number_of_hcf_lcm_l730_73023

theorem largest_number_of_hcf_lcm (HCF : ℕ) (factor1 factor2 : ℕ) (n1 n2 : ℕ) (largest : ℕ) 
  (h1 : HCF = 52) 
  (h2 : factor1 = 11) 
  (h3 : factor2 = 12) 
  (h4 : n1 = HCF * factor1) 
  (h5 : n2 = HCF * factor2) 
  (h6 : largest = max n1 n2) : 
  largest = 624 := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_of_hcf_lcm_l730_73023


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l730_73087

theorem min_value_x_plus_2y (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 * y - x * y = 0) : x + 2 * y = 8 := 
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l730_73087


namespace NUMINAMATH_GPT_jo_bob_pulled_chain_first_time_l730_73024

/-- Given the conditions of the balloon ride, prove that Jo-Bob pulled the chain
    for the first time for 15 minutes. --/
theorem jo_bob_pulled_chain_first_time (x : ℕ) : 
  (50 * x - 100 + 750 = 1400) → (x = 15) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_jo_bob_pulled_chain_first_time_l730_73024


namespace NUMINAMATH_GPT_expression_for_A_plus_2B_A_plus_2B_independent_of_b_l730_73014

theorem expression_for_A_plus_2B (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * b - 1
  let B := -a^2 - a * b + 1
  A + 2 * B = a * b - 2 * b + 1 :=
by
  sorry

theorem A_plus_2B_independent_of_b (a : ℝ) :
  (∀ b : ℝ, let A := 2 * a^2 + 3 * a * b - 2 * b - 1
            let B := -a^2 - a * b + 1
            A + 2 * B = a * b - 2 * b + 1) →
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_for_A_plus_2B_A_plus_2B_independent_of_b_l730_73014


namespace NUMINAMATH_GPT_average_speed_home_l730_73037

theorem average_speed_home
  (s_to_retreat : ℝ)
  (d_to_retreat : ℝ)
  (total_round_trip_time : ℝ)
  (t_retreat : d_to_retreat / s_to_retreat = 6)
  (t_total : d_to_retreat / s_to_retreat + 4 = total_round_trip_time) :
  (d_to_retreat / 4 = 75) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_home_l730_73037


namespace NUMINAMATH_GPT_part1_part2_l730_73097

-- Define A and B according to given expressions
def A (a b : ℚ) : ℚ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) : ℚ := -a^2 + a * b - 1

-- Prove the first statement
theorem part1 (a b : ℚ) : 4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 :=
by sorry

-- Prove the second statement
theorem part2 (F : ℚ) (b : ℚ) : (∀ a, A a b + 2 * B a b = F) → b = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l730_73097


namespace NUMINAMATH_GPT_hemisphere_surface_area_l730_73067

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l730_73067


namespace NUMINAMATH_GPT_find_common_ratio_limit_SN_over_TN_l730_73092

noncomputable def S (q : ℚ) (n : ℕ) : ℚ := (1 - q^n) / (1 - q)
noncomputable def T (q : ℚ) (n : ℕ) : ℚ := (1 - q^(2 * n)) / (1 - q^2)

theorem find_common_ratio
  (S3 : S q 3 = 3)
  (S6 : S q 6 = -21) :
  q = -2 :=
sorry

theorem limit_SN_over_TN
  (q_pos : 0 < q)
  (Tn_def : ∀ n, T q n = 1) :
  (q > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 0| < ε) ∧
  (0 < q ∧ q < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - (1 + q)| < ε) ∧
  (q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 1| < ε) :=
sorry

end NUMINAMATH_GPT_find_common_ratio_limit_SN_over_TN_l730_73092


namespace NUMINAMATH_GPT_inequality_div_half_l730_73095

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end NUMINAMATH_GPT_inequality_div_half_l730_73095


namespace NUMINAMATH_GPT_perimeter_eq_20_l730_73088

-- Define the lengths of the sides
def horizontal_sides := [2, 3]
def vertical_sides := [2, 3, 3, 2]

-- Define the perimeter calculation
def perimeter := horizontal_sides.sum + vertical_sides.sum

theorem perimeter_eq_20 : perimeter = 20 :=
by
  -- We assert that the calculations do hold
  sorry

end NUMINAMATH_GPT_perimeter_eq_20_l730_73088


namespace NUMINAMATH_GPT_students_both_l730_73085

noncomputable def students_total : ℕ := 32
noncomputable def students_go : ℕ := 18
noncomputable def students_chess : ℕ := 23

theorem students_both : students_go + students_chess - students_total = 9 := by
  sorry

end NUMINAMATH_GPT_students_both_l730_73085


namespace NUMINAMATH_GPT_angle_sum_l730_73007

theorem angle_sum (x : ℝ) (h1 : 2 * x + x = 90) : x = 30 := 
sorry

end NUMINAMATH_GPT_angle_sum_l730_73007


namespace NUMINAMATH_GPT_equal_profit_for_Robi_and_Rudy_l730_73058

theorem equal_profit_for_Robi_and_Rudy
  (robi_contrib : ℕ)
  (rudy_extra_contrib : ℕ)
  (profit_percent : ℚ)
  (share_profit_equally : Prop)
  (total_profit: ℚ)
  (each_share: ℕ) :
  robi_contrib = 4000 →
  rudy_extra_contrib = (1/4) * robi_contrib →
  profit_percent = 0.20 →
  share_profit_equally →
  total_profit = profit_percent * (robi_contrib + robi_contrib + rudy_extra_contrib) →
  each_share = (total_profit / 2) →
  each_share = 900 :=
by {
  sorry
}

end NUMINAMATH_GPT_equal_profit_for_Robi_and_Rudy_l730_73058


namespace NUMINAMATH_GPT_cos_beta_of_acute_angles_l730_73050

theorem cos_beta_of_acute_angles (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = Real.sqrt 5 / 5)
  (hsin_alpha_minus_beta : Real.sin (α - β) = 3 * Real.sqrt 10 / 10) :
  Real.cos β = 7 * Real.sqrt 2 / 10 :=
sorry

end NUMINAMATH_GPT_cos_beta_of_acute_angles_l730_73050


namespace NUMINAMATH_GPT_find_x_l730_73074

-- conditions
variable (k : ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- proportional relationship
def proportional_relationship (k x y z : ℝ) : Prop := 
  x = (k * y^2) / z

-- initial conditions
def initial_conditions (k : ℝ) : Prop := 
  proportional_relationship k 6 1 3

-- prove x = 24 when y = 2 and z = 3 under given conditions
theorem find_x (k : ℝ) (h : initial_conditions k) : 
  proportional_relationship k 24 2 3 :=
sorry

end NUMINAMATH_GPT_find_x_l730_73074


namespace NUMINAMATH_GPT_simplest_form_fraction_C_l730_73013

def fraction_A (x : ℤ) (y : ℤ) : ℚ := (2 * x + 4) / (6 * x + 8)
def fraction_B (x : ℤ) (y : ℤ) : ℚ := (x + y) / (x^2 - y^2)
def fraction_C (x : ℤ) (y : ℤ) : ℚ := (x^2 + y^2) / (x + y)
def fraction_D (x : ℤ) (y : ℤ) : ℚ := (x^2 - y^2) / (x^2 - 2 * x * y + y^2)

theorem simplest_form_fraction_C (x y : ℤ) :
  ¬ (∃ (A : ℚ), A ≠ fraction_C x y ∧ (A = fraction_C x y)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_simplest_form_fraction_C_l730_73013


namespace NUMINAMATH_GPT_quadratic_expression_value_l730_73073

theorem quadratic_expression_value :
  ∀ x1 x2 : ℝ, (x1^2 - 4 * x1 - 2020 = 0) ∧ (x2^2 - 4 * x2 - 2020 = 0) →
  (x1^2 - 2 * x1 + 2 * x2 = 2028) :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_quadratic_expression_value_l730_73073


namespace NUMINAMATH_GPT_complementary_angles_ratio_l730_73089

theorem complementary_angles_ratio (x : ℝ) (hx : 5 * x = 90) : abs (4 * x - x) = 54 :=
by
  have h₁ : x = 18 := by 
    linarith [hx]
  rw [h₁]
  norm_num

end NUMINAMATH_GPT_complementary_angles_ratio_l730_73089


namespace NUMINAMATH_GPT_least_number_to_add_to_4499_is_1_l730_73069

theorem least_number_to_add_to_4499_is_1 (x : ℕ) : (4499 + x) % 9 = 0 → x = 1 := sorry

end NUMINAMATH_GPT_least_number_to_add_to_4499_is_1_l730_73069


namespace NUMINAMATH_GPT_Jenna_total_cost_l730_73080

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end NUMINAMATH_GPT_Jenna_total_cost_l730_73080


namespace NUMINAMATH_GPT_households_in_city_l730_73064

theorem households_in_city (x : ℕ) (h1 : x < 100) (h2 : x + x / 3 = 100) : x = 75 :=
sorry

end NUMINAMATH_GPT_households_in_city_l730_73064


namespace NUMINAMATH_GPT_vanya_exam_scores_l730_73030

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end NUMINAMATH_GPT_vanya_exam_scores_l730_73030


namespace NUMINAMATH_GPT_geralds_average_speed_l730_73015

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end NUMINAMATH_GPT_geralds_average_speed_l730_73015


namespace NUMINAMATH_GPT_no_solution_system_l730_73065

theorem no_solution_system (v : ℝ) :
  (∀ x y z : ℝ, ¬(x + y + z = v ∧ x + v * y + z = v ∧ x + y + v^2 * z = v^2)) ↔ (v = -1) :=
  sorry

end NUMINAMATH_GPT_no_solution_system_l730_73065


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l730_73034

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d2 = 17) (h2 : A = 127.5) 
  (h3 : A = (d1 * d2) / 2) : d1 = 15 := 
by 
  -- Definitions
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l730_73034


namespace NUMINAMATH_GPT_height_of_triangle_l730_73056

variables (a b h' : ℝ)

theorem height_of_triangle (h : (1/2) * a * h' = a * b) : h' = 2 * b :=
sorry

end NUMINAMATH_GPT_height_of_triangle_l730_73056


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l730_73033

noncomputable def repeating_decimal := 1357 / 9999
def product_with_7 (x : ℚ) := 7 * x

theorem product_of_repeating_decimal :
  product_with_7 repeating_decimal = 9499 / 9999 :=
by sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_l730_73033


namespace NUMINAMATH_GPT_total_tennis_balls_used_l730_73047

theorem total_tennis_balls_used :
  let rounds := [1028, 514, 257, 128, 64, 32, 16, 8, 4]
  let cans_per_game_A := 6
  let cans_per_game_B := 8
  let balls_per_can_A := 3
  let balls_per_can_B := 4
  let games_A_to_B := rounds.splitAt 4
  let total_A := games_A_to_B.1.sum * cans_per_game_A * balls_per_can_A
  let total_B := games_A_to_B.2.sum * cans_per_game_B * balls_per_can_B
  total_A + total_B = 37573 := 
by
  sorry

end NUMINAMATH_GPT_total_tennis_balls_used_l730_73047


namespace NUMINAMATH_GPT_smallest_positive_period_intervals_of_monotonicity_max_min_values_l730_73081

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Prove the smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x := sorry

-- Prove the intervals of monotonicity
theorem intervals_of_monotonicity (k : ℤ) : 
  ∀ x y, (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
         (k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ k * Real.pi + Real.pi / 6) → 
         (x < y → f x < f y) ∨ (y < x → f y < f x) := sorry

-- Prove the maximum and minimum values on [0, π/2]
theorem max_min_values : ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧ 
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val ∧ f x ≥ min_val := sorry

end NUMINAMATH_GPT_smallest_positive_period_intervals_of_monotonicity_max_min_values_l730_73081


namespace NUMINAMATH_GPT_sphere_radius_l730_73061

-- Define the conditions
variable (r : ℝ) -- Radius of the sphere
variable (sphere_shadow : ℝ) (stick_height : ℝ) (stick_shadow : ℝ)

-- Given conditions
axiom sphere_shadow_equals_10 : sphere_shadow = 10
axiom stick_height_equals_1 : stick_height = 1
axiom stick_shadow_equals_2 : stick_shadow = 2

-- Using similar triangles and tangent relations, we want to prove the radius of sphere.
theorem sphere_radius (h1 : sphere_shadow = 10)
    (h2 : stick_height = 1)
    (h3 : stick_shadow = 2) : r = 5 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sphere_radius_l730_73061


namespace NUMINAMATH_GPT_jake_snake_length_l730_73099

theorem jake_snake_length (j p : ℕ) (h1 : j = p + 12) (h2 : j + p = 70) : j = 41 := by
  sorry

end NUMINAMATH_GPT_jake_snake_length_l730_73099


namespace NUMINAMATH_GPT_number_of_elephants_l730_73094

theorem number_of_elephants (giraffes penguins total_animals elephants : ℕ)
  (h1 : giraffes = 5)
  (h2 : penguins = 2 * giraffes)
  (h3 : penguins = total_animals / 5)
  (h4 : elephants = total_animals * 4 / 100) :
  elephants = 2 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_elephants_l730_73094


namespace NUMINAMATH_GPT_scientific_notation_of_74850000_l730_73075

theorem scientific_notation_of_74850000 : 74850000 = 7.485 * 10^7 :=
  by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_74850000_l730_73075


namespace NUMINAMATH_GPT_compute_expression_l730_73043

theorem compute_expression : (3 + 7)^3 + 2 * (3^2 + 7^2) = 1116 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l730_73043


namespace NUMINAMATH_GPT_temperature_difference_is_correct_l730_73001

def highest_temperature : ℤ := -9
def lowest_temperature : ℤ := -22
def temperature_difference : ℤ := highest_temperature - lowest_temperature

theorem temperature_difference_is_correct :
  temperature_difference = 13 := by
  -- We need to prove this statement is correct
  sorry

end NUMINAMATH_GPT_temperature_difference_is_correct_l730_73001


namespace NUMINAMATH_GPT_average_temperature_l730_73078

theorem average_temperature (T_tue T_wed T_thu : ℝ) 
  (h1 : (42 + T_tue + T_wed + T_thu) / 4 = 48)
  (T_fri : ℝ := 34) :
  ((T_tue + T_wed + T_thu + T_fri) / 4 = 46) :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_l730_73078


namespace NUMINAMATH_GPT_boys_girls_difference_l730_73016

/--
If there are 550 students in a class and the ratio of boys to girls is 7:4, 
prove that the number of boys exceeds the number of girls by 150.
-/
theorem boys_girls_difference : 
  ∀ (students boys_ratio girls_ratio : ℕ),
  students = 550 →
  boys_ratio = 7 →
  girls_ratio = 4 →
  (students * boys_ratio) % (boys_ratio + girls_ratio) = 0 ∧
  (students * girls_ratio) % (boys_ratio + girls_ratio) = 0 →
  (students * boys_ratio - students * girls_ratio) / (boys_ratio + girls_ratio) = 150 :=
by
  intros students boys_ratio girls_ratio h_students h_boys_ratio h_girls_ratio h_divisibility
  -- The detailed proof would follow here, but we add 'sorry' to bypass it.
  sorry

end NUMINAMATH_GPT_boys_girls_difference_l730_73016


namespace NUMINAMATH_GPT_sum_n_k_of_binomial_coefficient_ratio_l730_73083

theorem sum_n_k_of_binomial_coefficient_ratio :
  ∃ (n k : ℕ), (n = (7 * k + 5) / 2) ∧ (2 * (n - k) = 5 * (k + 1)) ∧ 
    ((k % 2 = 1) ∧ (n + k = 7 ∨ n + k = 16)) ∧ (23 = 7 + 16) :=
by
  sorry

end NUMINAMATH_GPT_sum_n_k_of_binomial_coefficient_ratio_l730_73083


namespace NUMINAMATH_GPT_tangent_line_equation_l730_73017

noncomputable def f (x : ℝ) : ℝ := (2 + Real.sin x) / Real.cos x

theorem tangent_line_equation :
  let x0 : ℝ := 0
  let y0 : ℝ := f x0
  let m : ℝ := (2 * x0 + 1) / (Real.cos x0 ^ 2)
  ∃ (a b c : ℝ), a * x0 + b * y0 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l730_73017


namespace NUMINAMATH_GPT_sum_of_remaining_six_numbers_l730_73000

theorem sum_of_remaining_six_numbers :
  ∀ (S T U : ℕ), 
    S = 20 * 500 → T = 14 * 390 → U = S - T → U = 4540 :=
by
  intros S T U hS hT hU
  sorry

end NUMINAMATH_GPT_sum_of_remaining_six_numbers_l730_73000


namespace NUMINAMATH_GPT_triangle_area_specific_l730_73048

noncomputable def vector2_area_formula (u v : ℝ × ℝ) : ℝ :=
|u.1 * v.2 - u.2 * v.1|

noncomputable def triangle_area (u v : ℝ × ℝ) : ℝ :=
(vector2_area_formula u v) / 2

theorem triangle_area_specific :
  let A := (1, 3)
  let B := (5, -1)
  let C := (9, 4)
  let u := (1 - 9, 3 - 4)
  let v := (5 - 9, -1 - 4)
  triangle_area u v = 18 := 
by sorry

end NUMINAMATH_GPT_triangle_area_specific_l730_73048


namespace NUMINAMATH_GPT_twin_brothers_age_l730_73002

theorem twin_brothers_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 17) : x = 8 := 
  sorry

end NUMINAMATH_GPT_twin_brothers_age_l730_73002


namespace NUMINAMATH_GPT_part_a_part_b_l730_73068

-- Define the cost variables for chocolates, popsicles, and lollipops
variables (C P L : ℕ)

-- Given conditions
axiom cost_relation1 : 3 * C = 2 * P
axiom cost_relation2 : 2 * L = 5 * C

-- Part (a): Prove that Mário can buy 5 popsicles with the money for 3 lollipops
theorem part_a : 
  (3 * L) / P = 5 :=
by sorry

-- Part (b): Prove that Mário can buy 11 chocolates with the money for 3 chocolates, 2 popsicles, and 2 lollipops combined
theorem part_b : 
  (3 * C + 2 * P + 2 * L) / C = 11 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l730_73068


namespace NUMINAMATH_GPT_hyperbola_asymptote_a_value_l730_73027

theorem hyperbola_asymptote_a_value (a : ℝ) (h : 0 < a) 
  (asymptote_eq : y = (3 / 5) * x) :
  (x^2 / a^2 - y^2 / 9 = 1) → a = 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_a_value_l730_73027


namespace NUMINAMATH_GPT_friends_recycled_pounds_l730_73008

theorem friends_recycled_pounds (total_points chloe_points each_points pounds_per_point : ℕ)
  (h1 : each_points = pounds_per_point / 6)
  (h2 : total_points = 5)
  (h3 : chloe_points = pounds_per_point / 6)
  (h4 : pounds_per_point = 28) 
  (h5 : total_points - chloe_points = 1) :
  pounds_per_point = 6 :=
by
  sorry

end NUMINAMATH_GPT_friends_recycled_pounds_l730_73008


namespace NUMINAMATH_GPT_previous_day_visitors_l730_73054

-- Define the number of visitors on the day Rachel visited
def visitors_on_day_rachel_visited : ℕ := 317

-- Define the difference in the number of visitors between the day Rachel visited and the previous day
def extra_visitors : ℕ := 22

-- Prove that the number of visitors on the previous day is 295
theorem previous_day_visitors : visitors_on_day_rachel_visited - extra_visitors = 295 :=
by
  sorry

end NUMINAMATH_GPT_previous_day_visitors_l730_73054


namespace NUMINAMATH_GPT_area_of_isosceles_right_triangle_l730_73009

def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b) ∧ (a^2 + b^2 = c^2)

theorem area_of_isosceles_right_triangle (a : ℝ) (hypotenuse : ℝ) (h_isosceles : is_isosceles_right_triangle a a hypotenuse) (h_hypotenuse : hypotenuse = 6) :
  (1 / 2) * a * a = 9 :=
by
  sorry

end NUMINAMATH_GPT_area_of_isosceles_right_triangle_l730_73009


namespace NUMINAMATH_GPT_timeSpentReading_l730_73020

def totalTime : ℕ := 120
def timeOnPiano : ℕ := 30
def timeWritingMusic : ℕ := 25
def timeUsingExerciser : ℕ := 27

theorem timeSpentReading :
  totalTime - timeOnPiano - timeWritingMusic - timeUsingExerciser = 38 := by
  sorry

end NUMINAMATH_GPT_timeSpentReading_l730_73020


namespace NUMINAMATH_GPT_winning_candidate_votes_percentage_l730_73051

theorem winning_candidate_votes_percentage (majority : ℕ) (total_votes : ℕ) (winning_percentage : ℚ) :
  majority = 174 ∧ total_votes = 435 ∧ winning_percentage = 70 → 
  ∃ P : ℚ, (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority ∧ P = 70 :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_percentage_l730_73051


namespace NUMINAMATH_GPT_solve_quadratic_eq_l730_73066

theorem solve_quadratic_eq (x : ℝ) : (x^2 + 4 * x = 5) ↔ (x = 1 ∨ x = -5) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l730_73066


namespace NUMINAMATH_GPT_sequence_missing_number_l730_73041

theorem sequence_missing_number : 
  ∃ x, (x - 21 = 7 ∧ 37 - x = 9) ∧ x = 28 := by
  sorry

end NUMINAMATH_GPT_sequence_missing_number_l730_73041


namespace NUMINAMATH_GPT_frog_jump_correct_l730_73028

def grasshopper_jump : ℤ := 25
def additional_distance : ℤ := 15
def frog_jump : ℤ := grasshopper_jump + additional_distance

theorem frog_jump_correct : frog_jump = 40 := by
  sorry

end NUMINAMATH_GPT_frog_jump_correct_l730_73028


namespace NUMINAMATH_GPT_atomic_weight_Br_correct_l730_73032

def atomic_weight_Ba : ℝ := 137.33
def molecular_weight_compound : ℝ := 297
def atomic_weight_Br : ℝ := 79.835

theorem atomic_weight_Br_correct :
  molecular_weight_compound = atomic_weight_Ba + 2 * atomic_weight_Br :=
by
  sorry

end NUMINAMATH_GPT_atomic_weight_Br_correct_l730_73032


namespace NUMINAMATH_GPT_trigonometric_identity_l730_73039

variable {α β γ n : Real}

-- Condition:
axiom h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)

-- Statement to be proved:
theorem trigonometric_identity : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l730_73039


namespace NUMINAMATH_GPT_degree_measure_supplement_complement_l730_73035

theorem degree_measure_supplement_complement : 
  let alpha := 63 -- angle value
  let theta := 90 - alpha -- complement of the angle
  let phi := 180 - theta -- supplement of the complement
  phi = 153 := -- prove the final step
by
  sorry

end NUMINAMATH_GPT_degree_measure_supplement_complement_l730_73035


namespace NUMINAMATH_GPT_perpendicular_vectors_k_zero_l730_73012

theorem perpendicular_vectors_k_zero
  (k : ℝ)
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ := (k, 2)) 
  (h : (a.1 - c.1, a.2 - c.2).1 * b.1 + (a.1 - c.1, a.2 - c.2).2 * b.2 = 0) :
  k = 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_k_zero_l730_73012


namespace NUMINAMATH_GPT_fermat_little_theorem_l730_73019

theorem fermat_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a ^ p ≡ a [MOD p] :=
sorry

end NUMINAMATH_GPT_fermat_little_theorem_l730_73019


namespace NUMINAMATH_GPT_students_chose_greek_food_l730_73038
  
theorem students_chose_greek_food (total_students : ℕ) (percentage_greek : ℝ) (h1 : total_students = 200) (h2 : percentage_greek = 0.5) :
  (percentage_greek * total_students : ℝ) = 100 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_students_chose_greek_food_l730_73038


namespace NUMINAMATH_GPT_finite_decimals_are_rational_l730_73004

-- Conditions as definitions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_infinite_decimal (x : ℝ) : Prop := ¬∃ (n : ℤ), x = ↑n
def is_finite_decimal (x : ℝ) : Prop := ∃ (a b : ℕ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Equivalence to statement C: Finite decimals are rational numbers
theorem finite_decimals_are_rational : ∀ (x : ℝ), is_finite_decimal x → is_rational x := by
  sorry

end NUMINAMATH_GPT_finite_decimals_are_rational_l730_73004


namespace NUMINAMATH_GPT_albert_snakes_count_l730_73036

noncomputable def garden_snake_length : ℝ := 10.0
noncomputable def boa_ratio : ℝ := 1 / 7.0
noncomputable def boa_length : ℝ := 1.428571429

theorem albert_snakes_count : 
  garden_snake_length = 10.0 ∧ 
  boa_ratio = 1 / 7.0 ∧ 
  boa_length = 1.428571429 → 
  2 = 2 :=
by
  intro h
  sorry   -- Proof will go here

end NUMINAMATH_GPT_albert_snakes_count_l730_73036


namespace NUMINAMATH_GPT_random_events_l730_73053

def is_random_event_1 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a + d < 0 ∨ b + c > 0

def is_random_event_2 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a - d > 0 ∨ b - c < 0

def is_impossible_event_3 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a * b > 0

def is_certain_event_4 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a / b < 0

theorem random_events (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  is_random_event_1 a b ha hb ∧ is_random_event_2 a b ha hb :=
by
  sorry

end NUMINAMATH_GPT_random_events_l730_73053


namespace NUMINAMATH_GPT_meals_distinct_pairs_l730_73090

theorem meals_distinct_pairs :
  let entrees := 4
  let drinks := 3
  let desserts := 3
  let total_meals := entrees * drinks * desserts
  total_meals * (total_meals - 1) = 1260 :=
by 
  sorry

end NUMINAMATH_GPT_meals_distinct_pairs_l730_73090


namespace NUMINAMATH_GPT_expression_eval_l730_73082

theorem expression_eval :
    (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
    (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * 5040 = 
    (5^128 - 4^128) * 5040 := by
  sorry

end NUMINAMATH_GPT_expression_eval_l730_73082


namespace NUMINAMATH_GPT_rank_from_start_l730_73003

theorem rank_from_start (n r_l : ℕ) (h_n : n = 31) (h_r_l : r_l = 15) : n - (r_l - 1) = 17 := by
  sorry

end NUMINAMATH_GPT_rank_from_start_l730_73003


namespace NUMINAMATH_GPT_mary_rental_hours_l730_73093

-- Definitions of the given conditions
def fixed_fee : ℝ := 17
def hourly_rate : ℝ := 7
def total_paid : ℝ := 80

-- Goal: Prove that the number of hours Mary paid for is 9
theorem mary_rental_hours : (total_paid - fixed_fee) / hourly_rate = 9 := 
by
  sorry

end NUMINAMATH_GPT_mary_rental_hours_l730_73093


namespace NUMINAMATH_GPT_base5_product_is_correct_l730_73084

-- Definitions for the problem context
def base5_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

def base10_to_base5 (n : ℕ) : List ℕ :=
  if n = 528 then [4, 1, 0, 0, 3] else []

-- Theorem to prove the base-5 multiplication result
theorem base5_product_is_correct :
  base10_to_base5 (base5_to_base10 2 3 1 * base5_to_base10 1 3 0) = [4, 1, 0, 0, 3] :=
by
  sorry

end NUMINAMATH_GPT_base5_product_is_correct_l730_73084


namespace NUMINAMATH_GPT_intersection_point_of_lines_l730_73062

theorem intersection_point_of_lines : 
  ∃ x y : ℝ, (3 * x + 4 * y - 2 = 0) ∧ (2 * x + y + 2 = 0) ∧ (x = -2) ∧ (y = 2) := 
by 
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l730_73062


namespace NUMINAMATH_GPT_ratio_of_speeds_correct_l730_73055

noncomputable def ratio_speeds_proof_problem : Prop :=
  ∃ (v_A v_B : ℝ),
    (∀ t : ℝ, 0 ≤ t ∧ t = 3 → 3 * v_A = abs (-800 + 3 * v_B)) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t = 15 → 15 * v_A = abs (-800 + 15 * v_B)) ∧
    (3 * 15 * v_A / (15 * v_B) = 3 / 4)

theorem ratio_of_speeds_correct : ratio_speeds_proof_problem :=
sorry

end NUMINAMATH_GPT_ratio_of_speeds_correct_l730_73055


namespace NUMINAMATH_GPT_train_speed_l730_73059

/-- Given: 
1. A train travels a distance of 80 km in 40 minutes. 
2. We need to prove that the speed of the train is 120 km/h.
-/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h_distance : distance = 80) 
  (h_time_minutes : time_minutes = 40) 
  (h_time_hours : time_hours = 40 / 60) 
  (h_speed : speed = distance / time_hours) : 
  speed = 120 :=
sorry

end NUMINAMATH_GPT_train_speed_l730_73059


namespace NUMINAMATH_GPT_schur_theorem_l730_73070

theorem schur_theorem {n : ℕ} (P : Fin n → Set ℕ) (h_partition : ∀ x : ℕ, ∃ i : Fin n, x ∈ P i) :
  ∃ (i : Fin n) (x y : ℕ), x ∈ P i ∧ y ∈ P i ∧ x + y ∈ P i :=
sorry

end NUMINAMATH_GPT_schur_theorem_l730_73070


namespace NUMINAMATH_GPT_num_unpainted_cubes_l730_73021

theorem num_unpainted_cubes (n : ℕ) (h1 : n ^ 3 = 125) : (n - 2) ^ 3 = 27 :=
by
  sorry

end NUMINAMATH_GPT_num_unpainted_cubes_l730_73021


namespace NUMINAMATH_GPT_base_6_to_10_conversion_l730_73079

theorem base_6_to_10_conversion : 
  ∀ (n : ℕ), n = 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0 → n = 1295 :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_base_6_to_10_conversion_l730_73079


namespace NUMINAMATH_GPT_sin_square_range_l730_73046

def range_sin_square_values (α β : ℝ) : Prop :=
  3 * (Real.sin α) ^ 2 - 2 * Real.sin α + 2 * (Real.sin β) ^ 2 = 0

theorem sin_square_range (α β : ℝ) (h : range_sin_square_values α β) :
  0 ≤ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ∧ 
  (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ≤ 4 / 9 :=
sorry

end NUMINAMATH_GPT_sin_square_range_l730_73046


namespace NUMINAMATH_GPT_angle_y_equals_90_l730_73006

/-- In a geometric configuration, if ∠CBD = 120° and ∠ABE = 30°, 
    then the measure of angle y is 90°. -/
theorem angle_y_equals_90 (angle_CBD angle_ABE : ℝ) 
  (h1 : angle_CBD = 120) 
  (h2 : angle_ABE = 30) : 
  ∃ y : ℝ, y = 90 := 
by
  sorry

end NUMINAMATH_GPT_angle_y_equals_90_l730_73006


namespace NUMINAMATH_GPT_sam_weight_l730_73057

theorem sam_weight (Tyler Sam Peter : ℕ) : 
  (Peter = 65) →
  (Peter = Tyler / 2) →
  (Tyler = Sam + 25) →
  Sam = 105 :=
  by
  intros hPeter1 hPeter2 hTyler
  sorry

end NUMINAMATH_GPT_sam_weight_l730_73057


namespace NUMINAMATH_GPT_milo_skateboarding_speed_l730_73086

theorem milo_skateboarding_speed (cory_speed milo_skateboarding_speed : ℝ) 
  (h1 : cory_speed = 12) 
  (h2 : cory_speed = 2 * milo_skateboarding_speed) : 
  milo_skateboarding_speed = 6 :=
by sorry

end NUMINAMATH_GPT_milo_skateboarding_speed_l730_73086


namespace NUMINAMATH_GPT_part1_part2_l730_73025

noncomputable def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 ≤ x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := sorry

theorem part2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (hmin : ∀ x, f x m ≥ 5 - n - t) :
  1 / (m + n) + 1 / t ≥ 2 := sorry

end NUMINAMATH_GPT_part1_part2_l730_73025


namespace NUMINAMATH_GPT_parabola_relative_positions_l730_73005

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 3
def parabola3 (x : ℝ) : ℝ := x^2 + 2*x + 3

noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem parabola_relative_positions :
  vertex_x 1 (-1) 3 < vertex_x 1 1 3 ∧ vertex_x 1 1 3 < vertex_x 1 2 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_relative_positions_l730_73005


namespace NUMINAMATH_GPT_train_length_is_300_l730_73010

noncomputable def length_of_train (V L : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 500 = V * 48)

theorem train_length_is_300
  (V : ℝ) (L : ℝ) (h : length_of_train V L) : L = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_300_l730_73010


namespace NUMINAMATH_GPT_complement_union_l730_73049

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  (U \ M) ∪ N = {2, 3, 4} :=
sorry

end NUMINAMATH_GPT_complement_union_l730_73049


namespace NUMINAMATH_GPT_cosine_of_eight_times_alpha_l730_73029

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end NUMINAMATH_GPT_cosine_of_eight_times_alpha_l730_73029


namespace NUMINAMATH_GPT_final_apples_count_l730_73060

def initial_apples : ℝ := 5708
def apples_given_away : ℝ := 2347.5
def additional_apples_harvested : ℝ := 1526.75

theorem final_apples_count :
  initial_apples - apples_given_away + additional_apples_harvested = 4887.25 :=
by
  sorry

end NUMINAMATH_GPT_final_apples_count_l730_73060


namespace NUMINAMATH_GPT_tangent_with_min_slope_has_given_equation_l730_73042

-- Define the given function f(x)
def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the coordinates of the tangent point
def tangent_point : ℝ × ℝ := (-1, f (-1))

-- Define the equation of the tangent line at the point with the minimum slope
def tangent_line_equation (x y : ℝ) : Prop := 3 * x - y - 11 = 0

-- Main theorem statement that needs to be proved
theorem tangent_with_min_slope_has_given_equation :
  tangent_line_equation (-1) (f (-1)) :=
sorry

end NUMINAMATH_GPT_tangent_with_min_slope_has_given_equation_l730_73042


namespace NUMINAMATH_GPT_solution_criteria_l730_73045

def is_solution (M : ℕ) : Prop :=
  5 ∣ (1989^M + M^1989)

theorem solution_criteria (M : ℕ) (h : M < 10) : is_solution M ↔ (M = 1 ∨ M = 4) :=
sorry

end NUMINAMATH_GPT_solution_criteria_l730_73045


namespace NUMINAMATH_GPT_ratio_of_bottles_l730_73026

theorem ratio_of_bottles
  (initial_money : ℤ)
  (initial_bottles : ℕ)
  (cost_per_bottle : ℤ)
  (cost_per_pound_cheese : ℤ)
  (cheese_pounds : ℚ)
  (remaining_money : ℤ) :
  initial_money = 100 →
  initial_bottles = 4 →
  cost_per_bottle = 2 →
  cost_per_pound_cheese = 10 →
  cheese_pounds = 0.5 →
  remaining_money = 71 →
  (2 * initial_bottles) / initial_bottles = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_bottles_l730_73026


namespace NUMINAMATH_GPT_roots_depend_on_k_l730_73072

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem roots_depend_on_k (k : ℝ) :
  let a := 1
  let b := -3
  let c := 2 - k
  discriminant a b c = 1 + 4 * k :=
by
  sorry

end NUMINAMATH_GPT_roots_depend_on_k_l730_73072


namespace NUMINAMATH_GPT_Jeremy_age_l730_73011

noncomputable def A : ℝ := sorry
noncomputable def J : ℝ := sorry
noncomputable def C : ℝ := sorry

-- Conditions
axiom h1 : A + J + C = 132
axiom h2 : A = (1/3) * J
axiom h3 : C = 2 * A

-- The goal is to prove J = 66
theorem Jeremy_age : J = 66 :=
sorry

end NUMINAMATH_GPT_Jeremy_age_l730_73011


namespace NUMINAMATH_GPT_finite_decimal_fractions_l730_73022

theorem finite_decimal_fractions (a b c d : ℕ) (n : ℕ) 
  (h1 : n = 2^a * 5^b)
  (h2 : n + 1 = 2^c * 5^d) :
  n = 1 ∨ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_finite_decimal_fractions_l730_73022
