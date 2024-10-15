import Mathlib

namespace NUMINAMATH_GPT_unique_a_for_set_A_l2218_221818

def A (a : ℝ) : Set ℝ := {a^2, 2 - a, 4}

theorem unique_a_for_set_A (a : ℝ) : A a = {x : ℝ // x = a^2 ∨ x = 2 - a ∨ x = 4} → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_unique_a_for_set_A_l2218_221818


namespace NUMINAMATH_GPT_curve_is_parabola_l2218_221869

theorem curve_is_parabola (r θ : ℝ) : (r = 1 / (1 - Real.cos θ)) ↔ ∃ x y : ℝ, y^2 = 2 * x + 1 :=
by 
  sorry

end NUMINAMATH_GPT_curve_is_parabola_l2218_221869


namespace NUMINAMATH_GPT_tom_hours_per_week_l2218_221843

-- Define the conditions
def summer_hours_per_week := 40
def summer_weeks := 8
def summer_total_earnings := 3200
def semester_weeks := 24
def semester_total_earnings := 2400
def hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
def total_hours_needed := semester_total_earnings / hourly_wage

-- Define the theorem to prove
theorem tom_hours_per_week :
  (total_hours_needed / semester_weeks) = 10 :=
sorry

end NUMINAMATH_GPT_tom_hours_per_week_l2218_221843


namespace NUMINAMATH_GPT_circle_symmetric_equation_l2218_221855

noncomputable def circle1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

noncomputable def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

noncomputable def symmetric_condition (x y : ℝ) (L : ℝ × ℝ → Prop) : Prop := 
  L (y + 1, x - 1)

theorem circle_symmetric_equation :
  ∀ (x y : ℝ),
  circle1 (y + 1) (x - 1) →
  (x-2)^2 + (y+2)^2 = 1 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_circle_symmetric_equation_l2218_221855


namespace NUMINAMATH_GPT_evaluate_expression_l2218_221831

theorem evaluate_expression : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2218_221831


namespace NUMINAMATH_GPT_distance_to_focus_2_l2218_221841

-- Definition of the ellipse and the given distance to one focus
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2)/25 + (P.2^2)/16 = 1
def distance_to_focus_1 (P : ℝ × ℝ) : Prop := dist P (5, 0) = 3

-- Proof problem statement
theorem distance_to_focus_2 (P : ℝ × ℝ) (h₁ : ellipse P) (h₂ : distance_to_focus_1 P) :
  dist P (-5, 0) = 7 :=
sorry

end NUMINAMATH_GPT_distance_to_focus_2_l2218_221841


namespace NUMINAMATH_GPT_number_of_members_l2218_221858

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end NUMINAMATH_GPT_number_of_members_l2218_221858


namespace NUMINAMATH_GPT_flagstaff_height_l2218_221894

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_flagstaff_height_l2218_221894


namespace NUMINAMATH_GPT_onions_total_l2218_221805

theorem onions_total (Sara_onions : ℕ) (Sally_onions : ℕ) (Fred_onions : ℕ)
  (h1 : Sara_onions = 4) (h2 : Sally_onions = 5) (h3 : Fred_onions = 9) :
  Sara_onions + Sally_onions + Fred_onions = 18 := by
  sorry

end NUMINAMATH_GPT_onions_total_l2218_221805


namespace NUMINAMATH_GPT_Mary_age_is_10_l2218_221832

-- Define the parameters for the ages of Rahul and Mary
variables (Rahul Mary : ℕ)

-- Conditions provided in the problem
def condition1 := Rahul = Mary + 30
def condition2 := Rahul + 20 = 2 * (Mary + 20)

-- Stating the theorem to be proved
theorem Mary_age_is_10 (Rahul Mary : ℕ) 
  (h1 : Rahul = Mary + 30) 
  (h2 : Rahul + 20 = 2 * (Mary + 20)) : 
  Mary = 10 :=
by 
  sorry

end NUMINAMATH_GPT_Mary_age_is_10_l2218_221832


namespace NUMINAMATH_GPT_correct_operation_l2218_221817

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^3 ≠ 2 * a^5) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  (a^3 * a^5 ≠ a^15) ∧
  ((ab^2)^2 = a^2 * b^4) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2218_221817


namespace NUMINAMATH_GPT_evaluate_fraction_l2218_221863

theorem evaluate_fraction : (3 : ℚ) / (2 - (3 / 4)) = (12 / 5) := 
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l2218_221863


namespace NUMINAMATH_GPT_determine_a_l2218_221840

theorem determine_a :
  ∃ a : ℝ, (∀ x : ℝ, y = -((x - a) / (x - a - 1)) ↔ x = (3 - a) / (3 - a - 1)) → a = 2 :=
sorry

end NUMINAMATH_GPT_determine_a_l2218_221840


namespace NUMINAMATH_GPT_slow_population_growth_before_ir_l2218_221803

-- Define the conditions
def low_level_social_productivity_before_ir : Prop := sorry
def high_birth_rate_before_ir : Prop := sorry
def high_mortality_rate_before_ir : Prop := sorry

-- The correct answer
def low_natural_population_growth_rate_before_ir : Prop := sorry

-- The theorem to prove
theorem slow_population_growth_before_ir 
  (h1 : low_level_social_productivity_before_ir) 
  (h2 : high_birth_rate_before_ir) 
  (h3 : high_mortality_rate_before_ir) : low_natural_population_growth_rate_before_ir := 
sorry

end NUMINAMATH_GPT_slow_population_growth_before_ir_l2218_221803


namespace NUMINAMATH_GPT_combined_population_l2218_221807

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end NUMINAMATH_GPT_combined_population_l2218_221807


namespace NUMINAMATH_GPT_sum_of_reciprocals_six_l2218_221867

theorem sum_of_reciprocals_six {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x) + (1 / y) = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_six_l2218_221867


namespace NUMINAMATH_GPT_large_square_area_l2218_221847

theorem large_square_area (a b c : ℕ) (h1 : 4 * a < b) (h2 : c^2 = a^2 + b^2 + 10) : c^2 = 36 :=
  sorry

end NUMINAMATH_GPT_large_square_area_l2218_221847


namespace NUMINAMATH_GPT_remainder_is_three_l2218_221883

def P (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem remainder_is_three : P 1 = 3 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_remainder_is_three_l2218_221883


namespace NUMINAMATH_GPT_golden_section_MP_length_l2218_221823

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem golden_section_MP_length (MN : ℝ) (hMN : MN = 2) (P : ℝ) 
  (hP : P > 0 ∧ P < MN ∧ P / (MN - P) = (MN - P) / P)
  (hMP_NP : MN - P < P) :
  P = Real.sqrt 5 - 1 :=
by
  sorry

end NUMINAMATH_GPT_golden_section_MP_length_l2218_221823


namespace NUMINAMATH_GPT_tangent_line_to_circle_l2218_221864

open Real

theorem tangent_line_to_circle (x y : ℝ) :
  ((x - 2) ^ 2 + (y + 1) ^ 2 = 9) ∧ ((x = -1) → (x = -1 ∧ y = 3) ∨ (y = (37 - 8*x) / 15)) :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_line_to_circle_l2218_221864


namespace NUMINAMATH_GPT_convert_fraction_to_decimal_l2218_221842

theorem convert_fraction_to_decimal : (3 / 40 : ℝ) = 0.075 := 
by
  sorry

end NUMINAMATH_GPT_convert_fraction_to_decimal_l2218_221842


namespace NUMINAMATH_GPT_geom_seq_b_value_l2218_221850

variable (r : ℝ) (b : ℝ)

-- b is the second term of the geometric sequence with first term 180 and third term 36/25
-- condition 1
def geom_sequence_cond1 := 180 * r = b
-- condition 2
def geom_sequence_cond2 := b * r = 36 / 25

-- Prove b = 16.1 given the conditions
theorem geom_seq_b_value (hb_pos : b > 0) (h1 : geom_sequence_cond1 r b) (h2 : geom_sequence_cond2 r b) : b = 16.1 :=
by sorry

end NUMINAMATH_GPT_geom_seq_b_value_l2218_221850


namespace NUMINAMATH_GPT_runners_meet_fractions_l2218_221895

theorem runners_meet_fractions (l V₁ V₂ : ℝ)
  (h1 : l / V₂ - l / V₁ = 10)
  (h2 : 720 * V₁ - 720 * V₂ = l) :
  (1 / V₁ = 1 / 80 ∧ 1 / V₂ = 1 / 90) ∨ (1 / V₁ = 1 / 90 ∧ 1 / V₂ = 1 / 80) :=
sorry

end NUMINAMATH_GPT_runners_meet_fractions_l2218_221895


namespace NUMINAMATH_GPT_percentage_of_men_is_55_l2218_221898

-- Define the percentage of men among all employees
def percent_of_men (M : ℝ) := M

-- Define the percentage of women among all employees
def percent_of_women (M : ℝ) := 1 - M

-- Define the contribution to picnic attendance by men
def attendance_by_men (M : ℝ) := 0.20 * M

-- Define the contribution to picnic attendance by women
def attendance_by_women (M : ℝ) := 0.40 * (percent_of_women M)

-- Define the total attendance
def total_attendance (M : ℝ) := attendance_by_men M + attendance_by_women M

theorem percentage_of_men_is_55 : ∀ M : ℝ, total_attendance M = 0.29 → M = 0.55 :=
by
  intro M
  intro h
  sorry

end NUMINAMATH_GPT_percentage_of_men_is_55_l2218_221898


namespace NUMINAMATH_GPT_length_of_diagonal_l2218_221826

theorem length_of_diagonal (h1 h2 area : ℝ) (h1_val : h1 = 7) (h2_val : h2 = 3) (area_val : area = 50) :
  ∃ d : ℝ, d = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_of_diagonal_l2218_221826


namespace NUMINAMATH_GPT_base_conversion_sum_l2218_221857

def A := 10
def B := 11

def convert_base11_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 11^2
  let d1 := (n % 11^2) / 11
  let d0 := n % 11
  d2 * 11^2 + d1 * 11 + d0

def convert_base12_to_base10 (n : ℕ) : ℕ :=
  let d2 := n / 12^2
  let d1 := (n % 12^2) / 12
  let d0 := n % 12
  d2 * 12^2 + d1 * 12 + d0

def n1 := 2 * 11^2 + 4 * 11 + 9    -- = 249_11 in base 10
def n2 := 3 * 12^2 + A * 12 + B   -- = 3AB_12 in base 10

theorem base_conversion_sum :
  (convert_base11_to_base10 294 + convert_base12_to_base10 563 = 858) := by
  sorry

end NUMINAMATH_GPT_base_conversion_sum_l2218_221857


namespace NUMINAMATH_GPT_g_triple_of_10_l2218_221836

def g (x : Int) : Int :=
  if x < 4 then x^2 - 9 else x + 7

theorem g_triple_of_10 : g (g (g 10)) = 31 := by
  sorry

end NUMINAMATH_GPT_g_triple_of_10_l2218_221836


namespace NUMINAMATH_GPT_original_number_is_80_l2218_221802

theorem original_number_is_80 (x : ℝ) (h1 : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_80_l2218_221802


namespace NUMINAMATH_GPT_exp_base_lt_imp_cube_l2218_221882

theorem exp_base_lt_imp_cube (a x y : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_exp : a^x > a^y) : x^3 < y^3 :=
by
  sorry

end NUMINAMATH_GPT_exp_base_lt_imp_cube_l2218_221882


namespace NUMINAMATH_GPT_sales_tax_difference_l2218_221830

def item_price : ℝ := 20
def sales_tax_rate1 : ℝ := 0.065
def sales_tax_rate2 : ℝ := 0.06

theorem sales_tax_difference :
  (item_price * sales_tax_rate1) - (item_price * sales_tax_rate2) = 0.1 := 
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l2218_221830


namespace NUMINAMATH_GPT_range_of_a_l2218_221853

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 8 ∧ a ≠ 4) ↔
  (a > 1 ∧ a < 8) ∧ (a > -4 ∧ a ≠ 4) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2218_221853


namespace NUMINAMATH_GPT_square_field_side_length_l2218_221884

theorem square_field_side_length (t : ℕ) (v : ℕ) 
  (run_time : t = 56) 
  (run_speed : v = 9) : 
  ∃ l : ℝ, l = 35 := 
sorry

end NUMINAMATH_GPT_square_field_side_length_l2218_221884


namespace NUMINAMATH_GPT_pasha_mistake_l2218_221875

theorem pasha_mistake :
  ¬ (∃ (K R O S C T P : ℕ), K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ P < 10 ∧
    K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ P ∧
    R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ P ∧
    O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ P ∧
    S ≠ C ∧ S ≠ T ∧ S ≠ P ∧
    C ≠ T ∧ C ≠ P ∧ T ≠ P ∧
    10000 * K + 1000 * R + 100 * O + 10 * S + S + 2011 = 10000 * C + 1000 * T + 100 * A + 10 * P + T) :=
sorry

end NUMINAMATH_GPT_pasha_mistake_l2218_221875


namespace NUMINAMATH_GPT_specific_five_card_order_probability_l2218_221893

open Classical

noncomputable def prob_five_cards_specified_order : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49) * (9 / 48)

theorem specific_five_card_order_probability :
  prob_five_cards_specified_order = 2304 / 31187500 :=
by
  sorry

end NUMINAMATH_GPT_specific_five_card_order_probability_l2218_221893


namespace NUMINAMATH_GPT_f_even_l2218_221845

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_const : ¬ (∀ x y : ℝ, f x = f y)
axiom f_equiv1 : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_equiv2 : ∀ x : ℝ, f (1 + x) = -f x

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  sorry

end NUMINAMATH_GPT_f_even_l2218_221845


namespace NUMINAMATH_GPT_train_length_correct_l2218_221827

noncomputable def length_of_train (train_speed_kmh : ℝ) (cross_time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * cross_time_s
  total_distance - bridge_length_m

theorem train_length_correct :
  length_of_train 45 30 205 = 170 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l2218_221827


namespace NUMINAMATH_GPT_remainder_of_m_l2218_221886

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_of_m_l2218_221886


namespace NUMINAMATH_GPT_value_of_k_parallel_vectors_l2218_221838

theorem value_of_k_parallel_vectors :
  (a : ℝ × ℝ) → (b : ℝ × ℝ) → (k : ℝ) →
  a = (2, 1) → b = (-1, k) → 
  (a.1 * b.2 - a.2 * b.1 = 0) →
  k = -(1/2) :=
by
  intros a b k ha hb hab_det
  sorry

end NUMINAMATH_GPT_value_of_k_parallel_vectors_l2218_221838


namespace NUMINAMATH_GPT_domain_of_function_l2218_221809

theorem domain_of_function :
  {x : ℝ | x > 4 ∧ x ≠ 5} = (Set.Ioo 4 5 ∪ Set.Ioi 5) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2218_221809


namespace NUMINAMATH_GPT_triangle_area_MEQF_l2218_221810

theorem triangle_area_MEQF
  (radius_P : ℝ)
  (chord_EF : ℝ)
  (par_EF_MN : Prop)
  (MQ : ℝ)
  (collinear_MQPN : Prop)
  (P MEF : ℝ × ℝ)
  (segment_P_Q : ℝ)
  (EF_length : ℝ)
  (radius_value : radius_P = 10)
  (EF_value : chord_EF = 12)
  (MQ_value : MQ = 20)
  (MN_parallel : par_EF_MN)
  (collinear : collinear_MQPN) :
  ∃ (area : ℝ), area = 48 := 
sorry

end NUMINAMATH_GPT_triangle_area_MEQF_l2218_221810


namespace NUMINAMATH_GPT_transform_negation_l2218_221848

variable (a b c : ℝ)

theorem transform_negation (a b c : ℝ) : - (a - b + c) = -a + b - c :=
by sorry

end NUMINAMATH_GPT_transform_negation_l2218_221848


namespace NUMINAMATH_GPT_kitty_cleaning_time_l2218_221856

def weekly_cleaning_time (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  pick_up + vacuum + clean_windows + dust

def total_cleaning_time (weeks: ℕ) (pick_up: ℕ) (vacuum: ℕ) (clean_windows: ℕ) (dust: ℕ) : ℕ :=
  weeks * weekly_cleaning_time pick_up vacuum clean_windows dust

theorem kitty_cleaning_time :
  total_cleaning_time 4 5 20 15 10 = 200 := by
  sorry

end NUMINAMATH_GPT_kitty_cleaning_time_l2218_221856


namespace NUMINAMATH_GPT_exists_a_lt_0_l2218_221849

noncomputable def f : ℝ → ℝ :=
sorry

theorem exists_a_lt_0 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, 0 < x → 0 < y → f (Real.sqrt (x * y)) = (f x + f y) / 2)
  (h2 : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :
  ∃ a : ℝ, 0 < a ∧ f a < 0 :=
sorry

end NUMINAMATH_GPT_exists_a_lt_0_l2218_221849


namespace NUMINAMATH_GPT_percentage_change_area_right_triangle_l2218_221816

theorem percentage_change_area_right_triangle
  (b h : ℝ)
  (hb : b = 0.5 * h)
  (A_original A_new : ℝ)
  (H_original : A_original = (1 / 2) * b * h)
  (H_new : A_new = (1 / 2) * (1.10 * b) * (1.10 * h)) :
  ((A_new - A_original) / A_original) * 100 = 21 := by
  sorry

end NUMINAMATH_GPT_percentage_change_area_right_triangle_l2218_221816


namespace NUMINAMATH_GPT_problem_a_problem_b_l2218_221820

-- Part (a)
theorem problem_a (n: Nat) : ∃ k: ℤ, (32^ (3 * n) - 1312^ n) = 1966 * k := sorry

-- Part (b)
theorem problem_b (n: Nat) : ∃ m: ℤ, (843^ (2 * n + 1) - 1099^ (2 * n + 1) + 16^ (4 * n + 2)) = 1967 * m := sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2218_221820


namespace NUMINAMATH_GPT_eval_expression_eq_54_l2218_221859

theorem eval_expression_eq_54 : (3 * 4 * 6) * ((1/3 : ℚ) + 1/4 + 1/6) = 54 := 
by
  sorry

end NUMINAMATH_GPT_eval_expression_eq_54_l2218_221859


namespace NUMINAMATH_GPT_intersection_M_N_l2218_221800

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l2218_221800


namespace NUMINAMATH_GPT_part1_part2_l2218_221825

-- Condition for exponents of x to be equal
def condition1 (a : ℤ) : Prop := (3 : ℤ) = 2 * a - 3

-- Condition for exponents of y to be equal
def condition2 (b : ℤ) : Prop := b = 1

noncomputable def a_value : ℤ := 3
noncomputable def b_value : ℤ := 1

-- Theorem for part (1): values of a and b
theorem part1 : condition1 3 ∧ condition2 1 :=
by
  have ha : condition1 3 := by sorry
  have hb : condition2 1 := by sorry
  exact And.intro ha hb

-- Theorem for part (2): value of (7a - 22)^2024 given a = 3
theorem part2 : (7 * a_value - 22) ^ 2024 = 1 :=
by
  have hx : 7 * a_value - 22 = -1 := by sorry
  have hres : (-1) ^ 2024 = 1 := by sorry
  exact Eq.trans (congrArg (fun x => x ^ 2024) hx) hres

end NUMINAMATH_GPT_part1_part2_l2218_221825


namespace NUMINAMATH_GPT_total_amount_from_grandparents_l2218_221881

theorem total_amount_from_grandparents (amount_from_grandpa : ℕ) (multiplier : ℕ) (amount_from_grandma : ℕ) (total_amount : ℕ) 
  (h1 : amount_from_grandpa = 30) 
  (h2 : multiplier = 3) 
  (h3 : amount_from_grandma = multiplier * amount_from_grandpa) 
  (h4 : total_amount = amount_from_grandpa + amount_from_grandma) :
  total_amount = 120 := 
by 
  sorry

end NUMINAMATH_GPT_total_amount_from_grandparents_l2218_221881


namespace NUMINAMATH_GPT_find_a_of_complex_eq_l2218_221851

theorem find_a_of_complex_eq (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * (⟨1, -a⟩ : ℂ) = 2) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_complex_eq_l2218_221851


namespace NUMINAMATH_GPT_unique_solution_m_l2218_221811

theorem unique_solution_m :
  ∃! m : ℝ, ∀ x y : ℝ, (y = x^2 ∧ y = 4*x + m) → m = -4 :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_m_l2218_221811


namespace NUMINAMATH_GPT_converse_proposition_l2218_221897

theorem converse_proposition (x : ℝ) (h : x = 1 → x^2 = 1) : x^2 = 1 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_converse_proposition_l2218_221897


namespace NUMINAMATH_GPT_proof_l2218_221837

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x ≥ 1
def q : Prop := ∀ x : ℝ, 0 < x → Real.exp x > Real.log x

-- The theorem statement
theorem proof : p ∧ q := by sorry

end NUMINAMATH_GPT_proof_l2218_221837


namespace NUMINAMATH_GPT_Douglas_won_in_county_Y_l2218_221879

def total_percentage (x y t r : ℝ) : Prop :=
  (0.74 * 2 + y * 1 = 0.66 * (2 + 1))

theorem Douglas_won_in_county_Y :
  ∀ (x y t r : ℝ), x = 0.74 → t = 0.66 → r = 2 →
  total_percentage x y t r → y = 0.50 := 
by
  intros x y t r hx ht hr H
  rw [hx, hr, ht] at H
  sorry

end NUMINAMATH_GPT_Douglas_won_in_county_Y_l2218_221879


namespace NUMINAMATH_GPT_coeff_x4_expansion_l2218_221874

def binom_expansion (a : ℚ) : ℚ :=
  let term1 : ℚ := a * 28
  let term2 : ℚ := -56
  term1 + term2

theorem coeff_x4_expansion (a : ℚ) : (binom_expansion a = -42) → a = 1/2 := 
by 
  intro h
  -- continuation of proof will go here.
  sorry

end NUMINAMATH_GPT_coeff_x4_expansion_l2218_221874


namespace NUMINAMATH_GPT_num_pos_int_values_l2218_221887

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end NUMINAMATH_GPT_num_pos_int_values_l2218_221887


namespace NUMINAMATH_GPT_construct_pairwise_tangent_circles_l2218_221834

-- Define the three points A, B, and C in a 2D plane.
variables (A B C : EuclideanSpace ℝ (Fin 2))

/--
  Given three points A, B, and C in the plane, 
  it is possible to construct three circles that are pairwise tangent at these points.
-/
theorem construct_pairwise_tangent_circles (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃ (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) (r1 r2 r3 : ℝ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    dist O1 O2 = r1 + r2 ∧
    dist O2 O3 = r2 + r3 ∧
    dist O3 O1 = r3 + r1 ∧
    dist O1 A = r1 ∧ dist O2 B = r2 ∧ dist O3 C = r3 :=
sorry

end NUMINAMATH_GPT_construct_pairwise_tangent_circles_l2218_221834


namespace NUMINAMATH_GPT_eraser_crayon_difference_l2218_221829

def initial_crayons : Nat := 601
def initial_erasers : Nat := 406
def final_crayons : Nat := 336
def final_erasers : Nat := initial_erasers

theorem eraser_crayon_difference :
  final_erasers - final_crayons = 70 :=
by
  sorry

end NUMINAMATH_GPT_eraser_crayon_difference_l2218_221829


namespace NUMINAMATH_GPT_john_candies_on_fourth_day_l2218_221892

theorem john_candies_on_fourth_day (c : ℕ) (h1 : 5 * c + 80 = 150) : c + 24 = 38 :=
by 
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_john_candies_on_fourth_day_l2218_221892


namespace NUMINAMATH_GPT_min_speed_x_l2218_221861

theorem min_speed_x (V_X : ℝ) : 
  let relative_speed_xy := V_X + 40;
  let relative_speed_xz := V_X - 30;
  (500 / relative_speed_xy) > (300 / relative_speed_xz) → 
  V_X ≥ 136 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_min_speed_x_l2218_221861


namespace NUMINAMATH_GPT_div_of_power_diff_div_l2218_221828

theorem div_of_power_diff_div (a b n : ℕ) (h : a ≠ b) (h₀ : n ∣ (a^n - b^n)) : n ∣ (a^n - b^n) / (a - b) :=
  sorry

end NUMINAMATH_GPT_div_of_power_diff_div_l2218_221828


namespace NUMINAMATH_GPT_determine_q_l2218_221821

-- Lean 4 statement
theorem determine_q (a : ℝ) (q : ℝ → ℝ) :
  (∀ x, q x = a * (x + 2) * (x - 3)) ∧ q 1 = 8 →
  q x = - (4 / 3) * x ^ 2 + (4 / 3) * x + 8 := 
sorry

end NUMINAMATH_GPT_determine_q_l2218_221821


namespace NUMINAMATH_GPT_secret_known_on_monday_l2218_221865

def students_know_secret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem secret_known_on_monday :
  ∃ n : ℕ, students_know_secret n = 3280 ∧ (n + 1) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_secret_known_on_monday_l2218_221865


namespace NUMINAMATH_GPT_find_e_l2218_221889

def P (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

-- Conditions
variables (d e f : ℝ)
-- Mean of zeros, twice product of zeros, and sum of coefficients are equal
variables (mean_of_zeros equals twice_product_of_zeros equals sum_of_coefficients equals: ℝ)
-- y-intercept is 9
axiom intercept_eq_nine : f = 9

-- Vieta's formulas for cubic polynomial
axiom product_of_zeros : twice_product_of_zeros = 2 * (- (f / 3))
axiom mean_of_zeros_sum : mean_of_zeros = -18/3  -- 3 times the mean of the zeros
axiom sum_of_coef : 3 + d + e + f = sum_of_coefficients

-- All these quantities are equal to the same value
axiom triple_equality : mean_of_zeros = twice_product_of_zeros
axiom triple_equality_coefs : mean_of_zeros = sum_of_coefficients

-- Lean statement we need to prove
theorem find_e : e = -72 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l2218_221889


namespace NUMINAMATH_GPT_ball_hits_ground_time_l2218_221878

theorem ball_hits_ground_time (t : ℚ) :
  (-4.9 * (t : ℝ)^2 + 5 * (t : ℝ) + 10 = 0) → t = 10 / 7 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l2218_221878


namespace NUMINAMATH_GPT_length_of_train_l2218_221812

theorem length_of_train (speed_kmh : ℕ) (time_s : ℕ) (length_bridge_m : ℕ) (length_train_m : ℕ) :
  speed_kmh = 45 → time_s = 30 → length_bridge_m = 275 → length_train_m = 475 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_length_of_train_l2218_221812


namespace NUMINAMATH_GPT_rectangular_field_area_l2218_221888

noncomputable def length (c : ℚ) : ℚ := 3 * c / 2
noncomputable def width (c : ℚ) : ℚ := 4 * c / 2
noncomputable def area (c : ℚ) : ℚ := (length c) * (width c)
noncomputable def field_area (c1 : ℚ) (c2 : ℚ) : ℚ :=
  let l := length c1
  let w := width c1
  if 25 * c2 = 101.5 * 100 then
    area c1
  else
    0

theorem rectangular_field_area :
  ∃ (c : ℚ), field_area c 25 = 10092 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l2218_221888


namespace NUMINAMATH_GPT_exist_one_common_ball_l2218_221814

theorem exist_one_common_ball (n : ℕ) (h_n : 5 ≤ n) (A : Fin (n+1) → Finset (Fin n))
  (hA_card : ∀ i, (A i).card = 3)
  (h_distinct : ∀ i j, i ≠ j → A i ≠ A j) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (A i ∩ A j).card = 1 :=
sorry

end NUMINAMATH_GPT_exist_one_common_ball_l2218_221814


namespace NUMINAMATH_GPT_minimum_value_g_l2218_221822

noncomputable def g (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem minimum_value_g : ∀ x : ℝ, x > 0 → g x ≥ 7 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_minimum_value_g_l2218_221822


namespace NUMINAMATH_GPT_find_maximum_marks_l2218_221846

variable (percent_marks : ℝ := 0.92)
variable (obtained_marks : ℝ := 368)
variable (max_marks : ℝ := obtained_marks / percent_marks)

theorem find_maximum_marks : max_marks = 400 := by
  sorry

end NUMINAMATH_GPT_find_maximum_marks_l2218_221846


namespace NUMINAMATH_GPT_convert_angle_degrees_to_radians_l2218_221870

theorem convert_angle_degrees_to_radians :
  ∃ (k : ℤ) (α : ℝ), -1125 * (Real.pi / 180) = 2 * k * Real.pi + α ∧ 0 ≤ α ∧ α < 2 * Real.pi ∧ (-8 * Real.pi + 7 * Real.pi / 4) = 2 * k * Real.pi + α :=
by {
  sorry
}

end NUMINAMATH_GPT_convert_angle_degrees_to_radians_l2218_221870


namespace NUMINAMATH_GPT_konjok_gorbunok_should_act_l2218_221844

def magical_power_retention (eat : ℕ → Prop) (sleep : ℕ → Prop) (seven_days : ℕ) : Prop :=
  ∀ t : ℕ, (0 ≤ t ∧ t ≤ seven_days) → ¬(eat t ∨ sleep t)

def retains_power (need_action : Prop) : Prop :=
  need_action

theorem konjok_gorbunok_should_act
  (eat : ℕ → Prop) (sleep : ℕ → Prop)
  (seven_days : ℕ)
  (h : magical_power_retention eat sleep seven_days)
  (before_start : ℕ → Prop) :
  retains_power (before_start seven_days) :=
by
  sorry

end NUMINAMATH_GPT_konjok_gorbunok_should_act_l2218_221844


namespace NUMINAMATH_GPT_blue_red_area_ratio_l2218_221868

theorem blue_red_area_ratio (d_small d_large : ℕ) (h1 : d_small = 2) (h2 : d_large = 6) :
    let r_small := d_small / 2
    let r_large := d_large / 2
    let A_red := Real.pi * (r_small : ℝ) ^ 2
    let A_large := Real.pi * (r_large : ℝ) ^ 2
    let A_blue := A_large - A_red
    A_blue / A_red = 8 :=
by
  sorry

end NUMINAMATH_GPT_blue_red_area_ratio_l2218_221868


namespace NUMINAMATH_GPT_christian_age_in_eight_years_l2218_221873

theorem christian_age_in_eight_years (b c : ℕ)
  (h1 : c = 2 * b)
  (h2 : b + 8 = 40) :
  c + 8 = 72 :=
sorry

end NUMINAMATH_GPT_christian_age_in_eight_years_l2218_221873


namespace NUMINAMATH_GPT_sarith_laps_l2218_221860

theorem sarith_laps 
  (k_speed : ℝ) (s_speed : ℝ) (k_laps : ℝ) (s_laps : ℝ) (distance_ratio : ℝ) :
  k_speed = 3 * s_speed →
  distance_ratio = 1 / 2 →
  k_laps = 12 →
  s_laps = (k_laps * 2 / 3) →
  s_laps = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sarith_laps_l2218_221860


namespace NUMINAMATH_GPT_circle_center_l2218_221804

theorem circle_center : 
  ∃ (h k : ℝ), (h, k) = (1, -2) ∧ 
    ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y - 4 = 0 ↔ (x - h)^2 + (y - k)^2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_l2218_221804


namespace NUMINAMATH_GPT_area_of_abs_inequality_l2218_221891

theorem area_of_abs_inequality :
  ∀ (x y : ℝ), |x + 2 * y| + |2 * x - y| ≤ 6 → 
  ∃ (area : ℝ), area = 12 := 
by
  -- This skips the proofs
  sorry

end NUMINAMATH_GPT_area_of_abs_inequality_l2218_221891


namespace NUMINAMATH_GPT_max_students_can_be_equally_distributed_l2218_221801

def num_pens : ℕ := 2730
def num_pencils : ℕ := 1890

theorem max_students_can_be_equally_distributed : Nat.gcd num_pens num_pencils = 210 := by
  sorry

end NUMINAMATH_GPT_max_students_can_be_equally_distributed_l2218_221801


namespace NUMINAMATH_GPT_point_A_coordinates_l2218_221854

-- Given conditions
def point_A (a : ℝ) : ℝ × ℝ := (a + 1, a^2 - 4)
def negative_half_x_axis (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 = 0

-- Theorem statement
theorem point_A_coordinates (a : ℝ) (h : negative_half_x_axis (point_A a)) :
  point_A a = (-1, 0) :=
sorry

end NUMINAMATH_GPT_point_A_coordinates_l2218_221854


namespace NUMINAMATH_GPT_gcd_of_840_and_1764_l2218_221835

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_GPT_gcd_of_840_and_1764_l2218_221835


namespace NUMINAMATH_GPT_percentage_increase_is_50_l2218_221852

def initial : ℝ := 110
def final : ℝ := 165

theorem percentage_increase_is_50 :
  ((final - initial) / initial) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_50_l2218_221852


namespace NUMINAMATH_GPT_max_z_val_l2218_221885

theorem max_z_val (x y : ℝ) (h1 : x + y ≤ 4) (h2 : y - 2 * x + 2 ≤ 0) (h3 : y ≥ 0) :
  ∃ x y, z = x + 2 * y ∧ z = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_z_val_l2218_221885


namespace NUMINAMATH_GPT_root_in_interval_l2218_221877

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - 2 / x

variable (h_monotonic : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)
variable (h_f_half : f (1 / 2) < 0)
variable (h_f_one : f 1 < 0)
variable (h_f_three_half : f (3 / 2) < 0)
variable (h_f_two : f 2 > 0)

theorem root_in_interval : ∃ c : ℝ, c ∈ Set.Ioo (3 / 2) 2 ∧ f c = 0 :=
sorry

end NUMINAMATH_GPT_root_in_interval_l2218_221877


namespace NUMINAMATH_GPT_polynomial_identity_l2218_221871

theorem polynomial_identity 
  (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  a^2 * ((x - b) * (x - c) / ((a - b) * (a - c))) +
  b^2 * ((x - c) * (x - a) / ((b - c) * (b - a))) +
  c^2 * ((x - a) * (x - b) / ((c - a) * (c - b))) = x^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l2218_221871


namespace NUMINAMATH_GPT_calculate_expression_l2218_221813
open Complex

-- Define the given values for a and b
def a := 3 + 2 * Complex.I
def b := 2 - 3 * Complex.I

-- Define the target expression
def target := 3 * a + 4 * b

-- The statement asserts that the target expression equals the expected result
theorem calculate_expression : target = 17 - 6 * Complex.I := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2218_221813


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l2218_221839

theorem arithmetic_sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = 5 * n^2 + 3 * n)
  (hS₁ : a 1 = S 1)
  (hS₂ : ∀ n, a (n + 1) = S (n + 1) - S n) :
  ∀ n, a n = 10 * n - 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l2218_221839


namespace NUMINAMATH_GPT_isosceles_largest_angle_eq_60_l2218_221899

theorem isosceles_largest_angle_eq_60 :
  ∀ (A B C : ℝ), (
    -- Condition: A triangle is isosceles with two equal angles of 60 degrees.
    ∀ (x y : ℝ), A = x ∧ B = x ∧ C = y ∧ x = 60 →
    -- Prove that
    max A (max B C) = 60 ) :=
by
  intros A B C h
  -- Sorry denotes skipping the proof.
  sorry

end NUMINAMATH_GPT_isosceles_largest_angle_eq_60_l2218_221899


namespace NUMINAMATH_GPT_quadratic_range_l2218_221876

theorem quadratic_range (x y : ℝ) (h1 : y = -(x - 5) ^ 2 + 1) (h2 : 2 < x ∧ x < 6) :
  -8 < y ∧ y ≤ 1 := 
sorry

end NUMINAMATH_GPT_quadratic_range_l2218_221876


namespace NUMINAMATH_GPT_A_and_D_mut_exclusive_not_complementary_l2218_221833

-- Define the events based on the conditions
inductive Die
| one | two | three | four | five | six

def is_odd (d : Die) : Prop :=
  d = Die.one ∨ d = Die.three ∨ d = Die.five

def is_even (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_multiple_of_2 (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four ∨ d = Die.six

def is_two_or_four (d : Die) : Prop :=
  d = Die.two ∨ d = Die.four

-- Define the predicate for mutually exclusive but not complementary
def mutually_exclusive_but_not_complementary (P Q : Die → Prop) : Prop :=
  (∀ d, ¬ (P d ∧ Q d)) ∧ ¬ (∀ d, P d ∨ Q d)

-- Verify that "A and D" are mutually exclusive but not complementary
theorem A_and_D_mut_exclusive_not_complementary :
  mutually_exclusive_but_not_complementary is_odd is_two_or_four :=
  by
    sorry

end NUMINAMATH_GPT_A_and_D_mut_exclusive_not_complementary_l2218_221833


namespace NUMINAMATH_GPT_expression_value_l2218_221866

theorem expression_value (a b c : ℝ) (h : a + b + c = 0) : (a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b)) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_expression_value_l2218_221866


namespace NUMINAMATH_GPT_find_x_l2218_221824

-- Defining the conditions
def angle_PQR : ℝ := 180
def angle_PQS : ℝ := 125
def angle_QSR (x : ℝ) : ℝ := x
def SQ_eq_SR : Prop := true -- Assuming an isosceles triangle where SQ = SR.

-- The theorem to be proved
theorem find_x (x : ℝ) :
  angle_PQR = 180 → angle_PQS = 125 → SQ_eq_SR → angle_QSR x = 70 :=
by
  intros _ _ _
  sorry

end NUMINAMATH_GPT_find_x_l2218_221824


namespace NUMINAMATH_GPT_sushi_eating_orders_l2218_221896

/-- Define a 2 x 3 grid with sushi pieces being distinguishable -/
inductive SushiPiece : Type
| A | B | C | D | E | F

open SushiPiece

/-- A function that counts the valid orders to eat sushi pieces satisfying the given conditions -/
noncomputable def countValidOrders : Nat :=
  sorry -- This is where the proof would go, stating the number of valid orders

theorem sushi_eating_orders :
  countValidOrders = 360 :=
sorry -- Skipping proof details

end NUMINAMATH_GPT_sushi_eating_orders_l2218_221896


namespace NUMINAMATH_GPT_problem_solution_l2218_221808

open Real

def system_satisfied (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ k : ℕ, k < n → 
    (a (2 * k + 1) = (1 / a (2 * (k + n) - 1) + 1 / a (2 * k + 2))) ∧ 
    (a (2 * k + 2) = a (2 * k + 1) + a (2 * k + 3))

theorem problem_solution (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ k, 0 ≤ k → k < 2 * n → a k > 0)
  (h3 : system_satisfied a n) :
  ∀ k, 0 ≤ k ∧ k < n → a (2 * k + 1) = 1 ∧ a (2 * k + 2) = 2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2218_221808


namespace NUMINAMATH_GPT_problem_I_problem_II_l2218_221815

-- Define the function f(x) = |x+1| + |x+m+1|
def f (x : ℝ) (m : ℝ) : ℝ := |x+1| + |x+(m+1)|

-- Define the problem (Ⅰ): f(x) ≥ |m-2| for all x implies m ≥ 1
theorem problem_I (m : ℝ) (h : ∀ x : ℝ, f x m ≥ |m-2|) : m ≥ 1 := sorry

-- Define the problem (Ⅱ): Find the solution set for f(-x) < 2m
theorem problem_II (m : ℝ) :
  (m ≤ 0 → ∀ x : ℝ, ¬ (f (-x) m < 2 * m)) ∧
  (m > 0 → ∀ x : ℝ, (1 - m / 2 < x ∧ x < 3 * m / 2 + 1) ↔ f (-x) m < 2 * m) := sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2218_221815


namespace NUMINAMATH_GPT_no_solution_ineq_system_l2218_221819

def inequality_system (x : ℝ) : Prop :=
  (x / 6 + 7 / 2 > (3 * x + 29) / 5) ∧
  (x + 9 / 2 > x / 8) ∧
  (11 / 3 - x / 6 < (34 - 3 * x) / 5)

theorem no_solution_ineq_system : ¬ ∃ x : ℝ, inequality_system x :=
  sorry

end NUMINAMATH_GPT_no_solution_ineq_system_l2218_221819


namespace NUMINAMATH_GPT_find_cos_A_l2218_221890

variable {A : Real}

theorem find_cos_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.tan A = 2 / 3) : Real.cos A = 3 * Real.sqrt 13 / 13 :=
by
  sorry

end NUMINAMATH_GPT_find_cos_A_l2218_221890


namespace NUMINAMATH_GPT_min_value_point_on_line_l2218_221872

theorem min_value_point_on_line (m n : ℝ) (h : m + 2 * n = 1) : 
  2^m + 4^n ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_point_on_line_l2218_221872


namespace NUMINAMATH_GPT_maximal_product_at_12_l2218_221862

noncomputable def geometric_sequence (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
a₁ * q^(n - 1)

noncomputable def product_first_n_terms (a₁ : ℕ) (q : ℚ) (n : ℕ) : ℚ :=
(a₁ ^ n) * (q ^ ((n - 1) * n / 2))

theorem maximal_product_at_12 :
  ∀ (a₁ : ℕ) (q : ℚ), 
  a₁ = 1536 → 
  q = -1/2 → 
  ∀ (n : ℕ), n ≠ 12 → 
  (product_first_n_terms a₁ q 12) > (product_first_n_terms a₁ q n) :=
by
  sorry

end NUMINAMATH_GPT_maximal_product_at_12_l2218_221862


namespace NUMINAMATH_GPT_oranges_equiv_frac_bananas_l2218_221806

theorem oranges_equiv_frac_bananas :
  (3 / 4) * 16 * (1 / 3) * 9 = (3 / 2) * 6 :=
by
  sorry

end NUMINAMATH_GPT_oranges_equiv_frac_bananas_l2218_221806


namespace NUMINAMATH_GPT_equation_has_two_distinct_real_roots_l2218_221880

open Real

theorem equation_has_two_distinct_real_roots (m : ℝ) :
  (∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < 16 ∧ 0 < x2 ∧ x2 < 16 ∧ x1 ≠ x2 ∧ exp (m * x1) = x1^2 ∧ exp (m * x2) = x2^2) ↔
  (log 2 / 2 < m ∧ m < 2 / exp 1) :=
by sorry

end NUMINAMATH_GPT_equation_has_two_distinct_real_roots_l2218_221880
