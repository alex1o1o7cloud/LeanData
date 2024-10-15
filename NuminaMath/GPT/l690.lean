import Mathlib

namespace NUMINAMATH_GPT_intersection_of_A_and_B_l690_69065

def setA : Set ℝ := {y | ∃ x : ℝ, y = 2 * x}
def setB : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem intersection_of_A_and_B : setA ∩ setB = {y | y ≥ 0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l690_69065


namespace NUMINAMATH_GPT_polygon_sides_l690_69086

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l690_69086


namespace NUMINAMATH_GPT_trajectory_equation_line_slope_is_constant_l690_69053

/-- Definitions for points A, B, and the moving point P -/ 
def pointA : ℝ × ℝ := (-2, 0)
def pointB : ℝ × ℝ := (2, 0)

/-- The condition that the product of the slopes is -3/4 -/
def slope_condition (P : ℝ × ℝ) : Prop :=
  let k_PA := P.2 / (P.1 + 2)
  let k_PB := P.2 / (P.1 - 2)
  k_PA * k_PB = -3 / 4

/-- The trajectory equation as a theorem to be proved -/
theorem trajectory_equation (P : ℝ × ℝ) (h : slope_condition P) : 
  P.2 ≠ 0 ∧ (P.1^2 / 4 + P.2^2 / 3 = 1) := 
sorry

/-- Additional conditions for the line l and points M, N -/ 
def line_l (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def intersect_conditions (P M N : ℝ × ℝ) (k m : ℝ) : Prop :=
  (M.2 = line_l k m M.1) ∧ (N.2 = line_l k m N.1) ∧ 
  (P ≠ M ∧ P ≠ N) ∧ ((P.1 = 1) ∧ (P.2 = 3 / 2)) ∧ 
  (let k_PM := (M.2 - P.2) / (M.1 - P.1)
  let k_PN := (N.2 - P.2) / (N.1 - P.1)
  k_PM + k_PN = 0)

/-- The theorem to prove that the slope of line l is 1/2 -/
theorem line_slope_is_constant (P M N : ℝ × ℝ) (k m : ℝ) 
  (h1 : slope_condition P) 
  (h2 : intersect_conditions P M N k m) : 
  k = 1 / 2 := 
sorry

end NUMINAMATH_GPT_trajectory_equation_line_slope_is_constant_l690_69053


namespace NUMINAMATH_GPT_ram_leela_piggy_bank_l690_69049

theorem ram_leela_piggy_bank (final_amount future_deposits weeks: ℕ) 
  (initial_deposit common_diff: ℕ) (total_deposits : ℕ) 
  (h_total : total_deposits = (weeks * (initial_deposit + (initial_deposit + (weeks - 1) * common_diff)) / 2)) 
  (h_final : final_amount = 1478) 
  (h_weeks : weeks = 52) 
  (h_future_deposits : future_deposits = total_deposits) 
  (h_initial_deposit : initial_deposit = 1) 
  (h_common_diff : common_diff = 1) 
  : final_amount - future_deposits = 100 :=
sorry

end NUMINAMATH_GPT_ram_leela_piggy_bank_l690_69049


namespace NUMINAMATH_GPT_remainder_division_by_8_is_6_l690_69012

theorem remainder_division_by_8_is_6 (N Q2 R1 : ℤ) (h1 : N = 64 + R1) (h2 : N % 5 = 4) : R1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_division_by_8_is_6_l690_69012


namespace NUMINAMATH_GPT_evaluate_expression_l690_69000

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l690_69000


namespace NUMINAMATH_GPT_cos_double_angle_l690_69028

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l690_69028


namespace NUMINAMATH_GPT_initial_bushes_count_l690_69087

theorem initial_bushes_count (n : ℕ) (h : 2 * (27 * n - 26) + 26 = 190 + 26) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_bushes_count_l690_69087


namespace NUMINAMATH_GPT_total_bills_proof_l690_69056

variable (a : ℝ) (total_may : ℝ) (total_june_may_june : ℝ)

-- The total bill in May is 140 yuan.
def total_bill_may (a : ℝ) := 140

-- The water bill increases by 10% in June.
def water_bill_june (a : ℝ) := 1.1 * a

-- The electricity bill in May.
def electricity_bill_may (a : ℝ) := 140 - a

-- The electricity bill increases by 20% in June.
def electricity_bill_june (a : ℝ) := (140 - a) * 1.2

-- Total electricity bills in June.
def total_electricity_june (a : ℝ) := (140 - a) + 0.2 * (140 - a)

-- Total water and electricity bills in June.
def total_water_electricity_june (a : ℝ) := 1.1 * a + 168 - 1.2 * a

-- Total water and electricity bills for May and June.
def total_water_electricity_may_june (a : ℝ) := a + (1.1 * a) + (140 - a) + ((140 - a) * 1.2)

-- When a = 40, the total water and electricity bills for May and June.
theorem total_bills_proof : ∀ a : ℝ, a = 40 → total_water_electricity_may_june a = 304 := 
by
  intros a ha
  rw [ha]
  sorry

end NUMINAMATH_GPT_total_bills_proof_l690_69056


namespace NUMINAMATH_GPT_pie_eating_contest_l690_69054

theorem pie_eating_contest :
  (7 / 8 : ℚ) - (5 / 6 : ℚ) = (1 / 24 : ℚ) :=
sorry

end NUMINAMATH_GPT_pie_eating_contest_l690_69054


namespace NUMINAMATH_GPT_big_al_bananas_l690_69061

theorem big_al_bananas (total_bananas : ℕ) (a : ℕ)
  (h : total_bananas = 150)
  (h1 : a + (a + 7) + (a + 14) + (a + 21) + (a + 28) = total_bananas) :
  a + 14 = 30 :=
by
  -- Using the given conditions to prove the statement
  sorry

end NUMINAMATH_GPT_big_al_bananas_l690_69061


namespace NUMINAMATH_GPT_non_talking_birds_count_l690_69088

def total_birds : ℕ := 77
def talking_birds : ℕ := 64

theorem non_talking_birds_count : total_birds - talking_birds = 13 := by
  sorry

end NUMINAMATH_GPT_non_talking_birds_count_l690_69088


namespace NUMINAMATH_GPT_angle_ZAX_pentagon_triangle_common_vertex_l690_69038

theorem angle_ZAX_pentagon_triangle_common_vertex :
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  common_angle_A = 192 := by
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  sorry

end NUMINAMATH_GPT_angle_ZAX_pentagon_triangle_common_vertex_l690_69038


namespace NUMINAMATH_GPT_min_ab_value_l690_69036

theorem min_ab_value (a b : Real) (h_a : 1 < a) (h_b : 1 < b)
  (h_geom_seq : ∀ (x₁ x₂ x₃ : Real), x₁ = (1/4) * Real.log a → x₂ = 1/4 → x₃ = Real.log b →  x₂^2 = x₁ * x₃) : 
  a * b ≥ Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_min_ab_value_l690_69036


namespace NUMINAMATH_GPT_train_takes_longer_l690_69096

-- Definitions for the conditions
def train_speed : ℝ := 48
def ship_speed : ℝ := 60
def distance : ℝ := 480

-- Theorem statement for the proof
theorem train_takes_longer : (distance / train_speed) - (distance / ship_speed) = 2 := by
  sorry

end NUMINAMATH_GPT_train_takes_longer_l690_69096


namespace NUMINAMATH_GPT_commute_time_abs_diff_l690_69024

theorem commute_time_abs_diff (x y : ℝ) 
  (h1 : (x + y + 10 + 11 + 9)/5 = 10) 
  (h2 : (1/5 : ℝ) * ((x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2) = 2) : 
  |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_abs_diff_l690_69024


namespace NUMINAMATH_GPT_smallest_divisible_1_to_10_l690_69055

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end NUMINAMATH_GPT_smallest_divisible_1_to_10_l690_69055


namespace NUMINAMATH_GPT_parabolas_equation_l690_69006

theorem parabolas_equation (vertex_origin : (0, 0) ∈ {(x, y) | y = x^2} ∨ (0, 0) ∈ {(x, y) | x = -y^2})
  (focus_on_axis : ∀ F : ℝ × ℝ, (F ∈ {(x, y) | y = x^2} ∨ F ∈ {(x, y) | x = -y^2}) → (F.1 = 0 ∨ F.2 = 0))
  (through_point : (-2, 4) ∈ {(x, y) | y = x^2} ∨ (-2, 4) ∈ {(x, y) | x = -y^2}) :
  {(x, y) | y = x^2} ∪ {(x, y) | x = -y^2} ≠ ∅ :=
by
  sorry

end NUMINAMATH_GPT_parabolas_equation_l690_69006


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l690_69048

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ 

-- Define sets A and B within the universal set U
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }
def B : Set ℝ := { x | x < -3 ∨ x > 1 }

-- Define the complements of A and B within U
def complement_A : Set ℝ := U \ A
def complement_B : Set ℝ := U \ B

-- Define the results as goals to be proved
theorem problem_part1 : A ∩ B = { x | 1 < x ∧ x ≤ 2 } := 
by
  sorry

theorem problem_part2 : complement_A ∩ complement_B = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_part3 : U \ (A ∪ B) = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l690_69048


namespace NUMINAMATH_GPT_max_unmarried_women_l690_69022

theorem max_unmarried_women (total_people : ℕ) (frac_women : ℚ) (frac_married : ℚ)
  (h_total : total_people = 80) (h_frac_women : frac_women = 1 / 4) (h_frac_married : frac_married = 3 / 4) :
  ∃ (max_unmarried_women : ℕ), max_unmarried_women = 20 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_max_unmarried_women_l690_69022


namespace NUMINAMATH_GPT_ab_value_l690_69018

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : ab = 2 ∨ ab = -2 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l690_69018


namespace NUMINAMATH_GPT_smallest_n_with_divisors_2020_l690_69082

theorem smallest_n_with_divisors_2020 :
  ∃ n : ℕ, (∃ α1 α2 α3 : ℕ, 
  n = 2^α1 * 3^α2 * 5^α3 ∧
  (α1 + 1) * (α2 + 1) * (α3 + 1) = 2020) ∧
  n = 2^100 * 3^4 * 5 * 7 := by
  sorry

end NUMINAMATH_GPT_smallest_n_with_divisors_2020_l690_69082


namespace NUMINAMATH_GPT_find_x_for_g_inv_l690_69078

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 4 * x + 1

theorem find_x_for_g_inv (x : ℝ) (h : g 3 = x) : g⁻¹ 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_g_inv_l690_69078


namespace NUMINAMATH_GPT_slower_whale_length_is_101_25_l690_69013

def length_of_slower_whale (v_i_f v_i_s a_f a_s t : ℝ) : ℝ :=
  let D_f := v_i_f * t + 0.5 * a_f * t^2
  let D_s := v_i_s * t + 0.5 * a_s * t^2
  D_f - D_s

theorem slower_whale_length_is_101_25
  (v_i_f v_i_s a_f a_s t L : ℝ)
  (h1 : v_i_f = 18)
  (h2 : v_i_s = 15)
  (h3 : a_f = 1)
  (h4 : a_s = 0.5)
  (h5 : t = 15)
  (h6 : length_of_slower_whale v_i_f v_i_s a_f a_s t = L) :
  L = 101.25 :=
by
  sorry

end NUMINAMATH_GPT_slower_whale_length_is_101_25_l690_69013


namespace NUMINAMATH_GPT_value_of_E_l690_69021

variable {D E F : ℕ}

theorem value_of_E (h1 : D + E + F = 16) (h2 : F + D + 1 = 16) (h3 : E - 1 = D) : E = 1 :=
sorry

end NUMINAMATH_GPT_value_of_E_l690_69021


namespace NUMINAMATH_GPT_angle_is_10_l690_69003

theorem angle_is_10 (x : ℕ) (h1 : 180 - x = 2 * (90 - x) + 10) : x = 10 := 
by sorry

end NUMINAMATH_GPT_angle_is_10_l690_69003


namespace NUMINAMATH_GPT_reciprocal_of_neg_three_l690_69020

theorem reciprocal_of_neg_three : (1 / (-3 : ℝ)) = (-1 / 3) := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_three_l690_69020


namespace NUMINAMATH_GPT_total_population_of_Springfield_and_Greenville_l690_69039

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end NUMINAMATH_GPT_total_population_of_Springfield_and_Greenville_l690_69039


namespace NUMINAMATH_GPT_profit_in_may_highest_monthly_profit_and_max_value_l690_69033

def f (x : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28 else 200 - 14 * x

theorem profit_in_may :
  f 5 = 88 :=
by sorry

theorem highest_monthly_profit_and_max_value :
  ∃ x, 1 ≤ x ∧ x ≤ 12 ∧ f x = 102 :=
by sorry

end NUMINAMATH_GPT_profit_in_may_highest_monthly_profit_and_max_value_l690_69033


namespace NUMINAMATH_GPT_problem1_l690_69043

theorem problem1 (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, 2 * |x + 3| ≥ m - 2 * |x + 7|) →
  (m ≤ 20) :=
by
  sorry

end NUMINAMATH_GPT_problem1_l690_69043


namespace NUMINAMATH_GPT_problem_solution_l690_69074

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -5) :
  (1 / a) + (1 / b) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l690_69074


namespace NUMINAMATH_GPT_base_b_square_l690_69050

-- Given that 144 in base b can be written as b^2 + 4b + 4 in base 10,
-- prove that it is a perfect square if and only if b > 4

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, b^2 + 4 * b + 4 = k^2 := by
  sorry

end NUMINAMATH_GPT_base_b_square_l690_69050


namespace NUMINAMATH_GPT_relationship_p_q_l690_69085

noncomputable def p (α β : ℝ) : ℝ := Real.cos α * Real.cos β
noncomputable def q (α β : ℝ) : ℝ := Real.cos ((α + β) / 2) ^ 2

theorem relationship_p_q (α β : ℝ) : p α β ≤ q α β :=
by
  sorry

end NUMINAMATH_GPT_relationship_p_q_l690_69085


namespace NUMINAMATH_GPT_find_n_l690_69035

-- Definitions and conditions
def painted_total_faces (n : ℕ) : ℕ := 6 * n^2
def total_faces_of_unit_cubes (n : ℕ) : ℕ := 6 * n^3
def fraction_of_red_faces (n : ℕ) : ℚ := (painted_total_faces n : ℚ) / (total_faces_of_unit_cubes n : ℚ)

-- Statement to be proven
theorem find_n (n : ℕ) (h : fraction_of_red_faces n = 1 / 4) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l690_69035


namespace NUMINAMATH_GPT_mobot_coloring_six_colorings_l690_69068

theorem mobot_coloring_six_colorings (n m : ℕ) (h : n ≥ 3 ∧ m ≥ 3) :
  (∃ mobot, mobot = (1, 1)) ↔ (∃ colorings : ℕ, colorings = 6) :=
sorry

end NUMINAMATH_GPT_mobot_coloring_six_colorings_l690_69068


namespace NUMINAMATH_GPT_Jake_needs_to_lose_12_pounds_l690_69091

theorem Jake_needs_to_lose_12_pounds (J S : ℕ) (h1 : J + S = 156) (h2 : J = 108) : J - 2 * S = 12 := by
  sorry

end NUMINAMATH_GPT_Jake_needs_to_lose_12_pounds_l690_69091


namespace NUMINAMATH_GPT_expression_value_l690_69008

theorem expression_value (x : ℝ) (h : x = 3 + 5 / (2 + 5 / x)) : x = 5 :=
sorry

end NUMINAMATH_GPT_expression_value_l690_69008


namespace NUMINAMATH_GPT_num_integer_pairs_satisfying_m_plus_n_eq_mn_l690_69034

theorem num_integer_pairs_satisfying_m_plus_n_eq_mn : 
  ∃ (m n : ℤ), (m + n = m * n) ∧ ∀ (m n : ℤ), (m + n = m * n) → 
  (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_num_integer_pairs_satisfying_m_plus_n_eq_mn_l690_69034


namespace NUMINAMATH_GPT_number_of_female_students_l690_69044

theorem number_of_female_students 
  (average_all : ℝ)
  (num_males : ℝ) 
  (average_males : ℝ)
  (average_females : ℝ) 
  (h_avg_all : average_all = 88)
  (h_num_males : num_males = 15)
  (h_avg_males : average_males = 80)
  (h_avg_females : average_females = 94) :
  ∃ F : ℝ, 1200 + 94 * F = 88 * (15 + F) ∧ F = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_number_of_female_students_l690_69044


namespace NUMINAMATH_GPT_set_relationship_l690_69037

def set_M : Set ℚ := {x : ℚ | ∃ m : ℤ, x = m + 1/6}
def set_N : Set ℚ := {x : ℚ | ∃ n : ℤ, x = n/2 - 1/3}
def set_P : Set ℚ := {x : ℚ | ∃ p : ℤ, x = p/2 + 1/6}

theorem set_relationship : set_M ⊆ set_N ∧ set_N = set_P := by
  sorry

end NUMINAMATH_GPT_set_relationship_l690_69037


namespace NUMINAMATH_GPT_older_brother_is_14_l690_69084

theorem older_brother_is_14 {Y O : ℕ} (h1 : Y + O = 26) (h2 : O = Y + 2) : O = 14 :=
by
  sorry

end NUMINAMATH_GPT_older_brother_is_14_l690_69084


namespace NUMINAMATH_GPT_shopkeeper_loss_percentage_l690_69030

theorem shopkeeper_loss_percentage
  (total_stock_value : ℝ)
  (overall_loss : ℝ)
  (first_part_percentage : ℝ)
  (first_part_profit_percentage : ℝ)
  (remaining_part_loss : ℝ)
  (total_worth_first_part : ℝ)
  (first_part_profit : ℝ)
  (remaining_stock_value : ℝ)
  (remaining_stock_loss : ℝ)
  (loss_percentage : ℝ) :
  total_stock_value = 16000 →
  overall_loss = 400 →
  first_part_percentage = 0.10 →
  first_part_profit_percentage = 0.20 →
  total_worth_first_part = total_stock_value * first_part_percentage →
  first_part_profit = total_worth_first_part * first_part_profit_percentage →
  remaining_stock_value = total_stock_value * (1 - first_part_percentage) →
  remaining_stock_loss = overall_loss + first_part_profit →
  loss_percentage = (remaining_stock_loss / remaining_stock_value) * 100 →
  loss_percentage = 5 :=
by intros; sorry

end NUMINAMATH_GPT_shopkeeper_loss_percentage_l690_69030


namespace NUMINAMATH_GPT_evaluate_expression_l690_69092

theorem evaluate_expression :
  (2 / 10 + 3 / 100 + 5 / 1000 + 7 / 10000)^2 = 0.05555649 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l690_69092


namespace NUMINAMATH_GPT_max_non_div_by_3_l690_69057

theorem max_non_div_by_3 (s : Finset ℕ) (h_len : s.card = 7) (h_prod : 3 ∣ s.prod id) : 
  ∃ n, n ≤ 6 ∧ ∀ x ∈ s, ¬ (3 ∣ x) → n = 6 :=
sorry

end NUMINAMATH_GPT_max_non_div_by_3_l690_69057


namespace NUMINAMATH_GPT_factor_expression_l690_69016

variable (x y : ℝ)

theorem factor_expression : 3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := 
by 
  sorry

end NUMINAMATH_GPT_factor_expression_l690_69016


namespace NUMINAMATH_GPT_correct_multiplication_value_l690_69063

theorem correct_multiplication_value (N : ℝ) (x : ℝ) : 
  (0.9333333333333333 = (N * x - N / 5) / (N * x)) → 
  x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_correct_multiplication_value_l690_69063


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l690_69059

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l690_69059


namespace NUMINAMATH_GPT_brownies_count_l690_69099

theorem brownies_count (pan_length : ℕ) (pan_width : ℕ) (piece_side : ℕ) 
  (h1 : pan_length = 24) (h2 : pan_width = 15) (h3 : piece_side = 3) : 
  (pan_length * pan_width) / (piece_side * piece_side) = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_brownies_count_l690_69099


namespace NUMINAMATH_GPT_solution_set_M_minimum_value_expr_l690_69051

-- Define the function f(x)
def f (x : ℝ) : ℝ := abs (x + 1) - 2 * abs (x - 2)

-- Proof problem (1): Prove that the solution set M of the inequality f(x) ≥ -1 is {x | 2/3 ≤ x ≤ 6}.
theorem solution_set_M : 
  { x : ℝ | f x ≥ -1 } = { x : ℝ | 2/3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Define the requirement for t and the expression to minimize
noncomputable def t : ℝ := 6
noncomputable def expr (a b c : ℝ) : ℝ := 1 / (2 * a + b) + 1 / (2 * a + c)

-- Proof problem (2): Given t = 6 and 4a + b + c = 6, prove that the minimum value of expr is 2/3.
theorem minimum_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : 4 * a + b + c = t) :
  expr a b c ≥ 2/3 :=
sorry

end NUMINAMATH_GPT_solution_set_M_minimum_value_expr_l690_69051


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l690_69058

theorem batsman_average_after_12th_innings (A : ℕ) (total_runs_11 : ℕ) (total_runs_12 : ℕ ) : 
  total_runs_11 = 11 * A → 
  total_runs_12 = total_runs_11 + 55 → 
  (total_runs_12 / 12 = A + 1) → 
  (A + 1) = 44 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l690_69058


namespace NUMINAMATH_GPT_relationship_between_abc_l690_69064

-- Definitions based on the conditions
def a : ℕ := 3^44
def b : ℕ := 4^33
def c : ℕ := 5^22

-- The theorem to prove the relationship a > b > c
theorem relationship_between_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_relationship_between_abc_l690_69064


namespace NUMINAMATH_GPT_minimum_value_of_f_l690_69097

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 4 * y * z + 2 * z * x - 6 * x - 10 * y - 12 * z

theorem minimum_value_of_f : ∃ x y z : ℝ, f x y z = -14 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l690_69097


namespace NUMINAMATH_GPT_gcd_80_36_l690_69025

theorem gcd_80_36 : Nat.gcd 80 36 = 4 := by
  -- Using the method of successive subtraction algorithm
  sorry

end NUMINAMATH_GPT_gcd_80_36_l690_69025


namespace NUMINAMATH_GPT_set_intersection_l690_69023

open Set Real

theorem set_intersection (A : Set ℝ) (hA : A = {-1, 0, 1}) (B : Set ℝ) (hB : B = {y | ∃ x ∈ A, y = cos (π * x)}) :
  A ∩ B = {-1, 1} :=
by
  rw [hA, hB]
  -- remaining proof should go here
  sorry

end NUMINAMATH_GPT_set_intersection_l690_69023


namespace NUMINAMATH_GPT_inequality_correct_l690_69076

variable {a b : ℝ}

theorem inequality_correct (h₁ : a < 1) (h₂ : b > 1) : ab < a + b :=
sorry

end NUMINAMATH_GPT_inequality_correct_l690_69076


namespace NUMINAMATH_GPT_fraction_zero_implies_a_eq_neg2_l690_69032

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_a_eq_neg2_l690_69032


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_0_6_l690_69047

theorem line_intersects_y_axis_at_0_6 : ∃ y : ℝ, 4 * y + 3 * (0 : ℝ) = 24 ∧ (0, y) = (0, 6) :=
by
  use 6
  simp
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_0_6_l690_69047


namespace NUMINAMATH_GPT_gel_pen_price_ratio_l690_69019

variable (x y b g T : ℝ)

-- Conditions from the problem
def condition1 : Prop := T = x * b + y * g
def condition2 : Prop := (x + y) * g = 4 * T
def condition3 : Prop := (x + y) * b = (1 / 2) * T

theorem gel_pen_price_ratio (h1 : condition1 x y b g T) (h2 : condition2 x y g T) (h3 : condition3 x y b T) :
  g = 8 * b :=
sorry

end NUMINAMATH_GPT_gel_pen_price_ratio_l690_69019


namespace NUMINAMATH_GPT_intersection_point_of_lines_l690_69040

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), x + 2 * y - 4 = 0 ∧ 2 * x - y + 2 = 0 ∧ (x, y) = (0, 2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l690_69040


namespace NUMINAMATH_GPT_horse_food_needed_l690_69005

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end NUMINAMATH_GPT_horse_food_needed_l690_69005


namespace NUMINAMATH_GPT_radius_ratio_eq_inv_sqrt_5_l690_69010

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_radius_ratio_eq_inv_sqrt_5_l690_69010


namespace NUMINAMATH_GPT_sarah_shampoo_and_conditioner_usage_l690_69079

-- Condition Definitions
def shampoo_daily_oz := 1
def conditioner_daily_oz := shampoo_daily_oz / 2
def total_daily_usage := shampoo_daily_oz + conditioner_daily_oz

def days_in_two_weeks := 14

-- Assertion: Total volume used in two weeks.
theorem sarah_shampoo_and_conditioner_usage :
  (days_in_two_weeks * total_daily_usage) = 21 := by
  sorry

end NUMINAMATH_GPT_sarah_shampoo_and_conditioner_usage_l690_69079


namespace NUMINAMATH_GPT_arrange_x_y_z_l690_69041

theorem arrange_x_y_z (x : ℝ) (hx : 0.9 < x ∧ x < 1) :
  let y := x^(1/x)
  let z := x^y
  x < z ∧ z < y :=
by
  let y := x^(1/x)
  let z := x^y
  have : 0.9 < x ∧ x < 1 := hx
  sorry

end NUMINAMATH_GPT_arrange_x_y_z_l690_69041


namespace NUMINAMATH_GPT_students_in_class_l690_69009

theorem students_in_class
  (B : ℕ) (E : ℕ) (G : ℕ)
  (h1 : B = 12)
  (h2 : G + B = 22)
  (h3 : E = 10) :
  G + E + B = 32 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l690_69009


namespace NUMINAMATH_GPT_true_statement_count_l690_69083

def reciprocal (n : ℕ) : ℚ := 1 / n

def statement_i := (reciprocal 4 + reciprocal 8 = reciprocal 12)
def statement_ii := (reciprocal 9 - reciprocal 3 = reciprocal 6)
def statement_iii := (reciprocal 3 * reciprocal 9 = reciprocal 27)
def statement_iv := (reciprocal 15 / reciprocal 3 = reciprocal 5)

theorem true_statement_count :
  (¬statement_i ∧ ¬statement_ii ∧ statement_iii ∧ statement_iv) ↔ (2 = 2) :=
by sorry

end NUMINAMATH_GPT_true_statement_count_l690_69083


namespace NUMINAMATH_GPT_johns_total_cost_l690_69042

-- Definitions for the prices and quantities
def price_shirt : ℝ := 15.75
def price_tie : ℝ := 9.40
def quantity_shirts : ℕ := 3
def quantity_ties : ℕ := 2

-- Definition for the total cost calculation
def total_cost (price_shirt price_tie : ℝ) (quantity_shirts quantity_ties : ℕ) : ℝ :=
  (price_shirt * quantity_shirts) + (price_tie * quantity_ties)

-- Theorem stating the total cost calculation for John's purchase
theorem johns_total_cost : total_cost price_shirt price_tie quantity_shirts quantity_ties = 66.05 :=
by
  sorry

end NUMINAMATH_GPT_johns_total_cost_l690_69042


namespace NUMINAMATH_GPT_gcd_min_b_c_l690_69004

theorem gcd_min_b_c (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  Nat.gcd b c = 21 :=
sorry

end NUMINAMATH_GPT_gcd_min_b_c_l690_69004


namespace NUMINAMATH_GPT_average_infection_rate_infected_computers_exceed_700_l690_69014

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end NUMINAMATH_GPT_average_infection_rate_infected_computers_exceed_700_l690_69014


namespace NUMINAMATH_GPT_find_x_l690_69011

def seq : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 11
| 3 => 20
| 4 => 32
| 5 => 47
| (n+6) => seq (n+5) + 3 * (n + 1)

theorem find_x : seq 6 = 65 := by
  sorry

end NUMINAMATH_GPT_find_x_l690_69011


namespace NUMINAMATH_GPT_prove_range_of_a_l690_69094

noncomputable def f (x a : ℝ) : ℝ := (x + a - 1) * Real.exp x

def problem_condition1 (x a : ℝ) : Prop := 
  f x a ≥ (x^2 / 2 + a * x)

def problem_condition2 (x : ℝ) : Prop := 
  x ∈ Set.Ici 0 -- equivalent to [0, +∞)

theorem prove_range_of_a (a : ℝ) :
  (∀ x : ℝ, problem_condition2 x → problem_condition1 x a) → a ∈ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_prove_range_of_a_l690_69094


namespace NUMINAMATH_GPT_zoe_distance_more_than_leo_l690_69031

theorem zoe_distance_more_than_leo (d t s : ℝ)
  (maria_driving_time : ℝ := t + 2)
  (maria_speed : ℝ := s + 15)
  (zoe_driving_time : ℝ := t + 3)
  (zoe_speed : ℝ := s + 20)
  (leo_distance : ℝ := s * t)
  (maria_distance : ℝ := (s + 15) * (t + 2))
  (zoe_distance : ℝ := (s + 20) * (t + 3))
  (maria_leo_distance_diff : ℝ := 110)
  (h1 : maria_distance = leo_distance + maria_leo_distance_diff)
  : zoe_distance - leo_distance = 180 :=
by
  sorry

end NUMINAMATH_GPT_zoe_distance_more_than_leo_l690_69031


namespace NUMINAMATH_GPT_gain_percent_calculation_l690_69069

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  (S - C) / C * 100

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 46 * S) : 
  gain_percent C S = 100 / 11.5 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_calculation_l690_69069


namespace NUMINAMATH_GPT_total_books_arithmetic_sequence_l690_69066

theorem total_books_arithmetic_sequence :
  ∃ (n : ℕ) (a₁ a₂ aₙ d S : ℤ), 
    n = 11 ∧
    a₁ = 32 ∧
    a₂ = 29 ∧
    aₙ = 2 ∧
    d = -3 ∧
    S = (n * (a₁ + aₙ)) / 2 ∧
    S = 187 :=
by sorry

end NUMINAMATH_GPT_total_books_arithmetic_sequence_l690_69066


namespace NUMINAMATH_GPT_circle_area_ratio_l690_69077

theorem circle_area_ratio (O X P : ℝ) (rOx rOp : ℝ) (h1 : rOx = rOp / 3) :
  (π * rOx^2) / (π * rOp^2) = 1 / 9 :=
by 
  -- Import required theorems and add assumptions as necessary
  -- Continue the proof based on Lean syntax
  sorry

end NUMINAMATH_GPT_circle_area_ratio_l690_69077


namespace NUMINAMATH_GPT_bobby_gasoline_left_l690_69045

theorem bobby_gasoline_left
  (initial_gasoline : ℕ) (supermarket_distance : ℕ) 
  (travel_distance : ℕ) (turn_back_distance : ℕ)
  (trip_fuel_efficiency : ℕ) : 
  initial_gasoline = 12 →
  supermarket_distance = 5 →
  travel_distance = 6 →
  turn_back_distance = 2 →
  trip_fuel_efficiency = 2 →
  ∃ remaining_gasoline,
    remaining_gasoline = initial_gasoline - 
    ((supermarket_distance * 2 + 
    turn_back_distance * 2 + 
    travel_distance) / trip_fuel_efficiency) ∧ 
    remaining_gasoline = 2 :=
by sorry

end NUMINAMATH_GPT_bobby_gasoline_left_l690_69045


namespace NUMINAMATH_GPT_number_condition_l690_69060

theorem number_condition (x : ℝ) (h : 45 - 3 * x^2 = 12) : x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_number_condition_l690_69060


namespace NUMINAMATH_GPT_teams_worked_together_days_l690_69027

noncomputable def first_team_rate : ℝ := 1 / 12
noncomputable def second_team_rate : ℝ := 1 / 9
noncomputable def first_team_days : ℕ := 5
noncomputable def total_work : ℝ := 1
noncomputable def work_first_team_alone := first_team_rate * first_team_days

theorem teams_worked_together_days (x : ℝ) : work_first_team_alone + (first_team_rate + second_team_rate) * x = total_work → x = 3 := 
by
  sorry

end NUMINAMATH_GPT_teams_worked_together_days_l690_69027


namespace NUMINAMATH_GPT_red_ball_probability_correct_l690_69026

theorem red_ball_probability_correct (R B : ℕ) (hR : R = 3) (hB : B = 3) :
  (R / (R + B) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_red_ball_probability_correct_l690_69026


namespace NUMINAMATH_GPT_cheddar_cheese_slices_l690_69093

-- Define the conditions
def cheddar_slices (C : ℕ) := ∃ (packages : ℕ), packages * C = 84
def swiss_slices := 28
def randy_bought_same_slices (C : ℕ) := swiss_slices = 28 ∧ 84 = 84

-- Lean theorem statement to prove the number of slices per package of cheddar cheese equals 28.
theorem cheddar_cheese_slices {C : ℕ} (h1 : cheddar_slices C) (h2 : randy_bought_same_slices C) : C = 28 :=
sorry

end NUMINAMATH_GPT_cheddar_cheese_slices_l690_69093


namespace NUMINAMATH_GPT_fraction_pow_four_result_l690_69002

theorem fraction_pow_four_result (x : ℚ) (h : x = 1 / 4) : x ^ 4 = 390625 / 100000000 :=
by sorry

end NUMINAMATH_GPT_fraction_pow_four_result_l690_69002


namespace NUMINAMATH_GPT_value_of_expression_l690_69098

theorem value_of_expression (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : |m| = 3) :
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l690_69098


namespace NUMINAMATH_GPT_magician_earning_correct_l690_69071

def magician_earning (initial_decks : ℕ) (remaining_decks : ℕ) (price_per_deck : ℕ) : ℕ :=
  (initial_decks - remaining_decks) * price_per_deck

theorem magician_earning_correct :
  magician_earning 5 3 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_magician_earning_correct_l690_69071


namespace NUMINAMATH_GPT_union_A_B_l690_69007

-- Definitions based on the conditions
def A := { x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3) }
def B := { x : ℝ | -2 ≤ x ∧ x < 4 }

-- The proof goal
theorem union_A_B : A ∪ B = { x : ℝ | x < 4 } :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_union_A_B_l690_69007


namespace NUMINAMATH_GPT_clock_strikes_twelve_l690_69001

def clock_strike_interval (strikes : Nat) (time : Nat) : Nat :=
  if strikes > 1 then time / (strikes - 1) else 0

def total_time_for_strikes (strikes : Nat) (interval : Nat) : Nat :=
  if strikes > 1 then (strikes - 1) * interval else 0

theorem clock_strikes_twelve (interval_six : Nat) (time_six : Nat) (time_twelve : Nat) :
  interval_six = clock_strike_interval 6 time_six →
  time_twelve = total_time_for_strikes 12 interval_six →
  time_six = 30 →
  time_twelve = 66 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_clock_strikes_twelve_l690_69001


namespace NUMINAMATH_GPT_solve_fractional_equation_l690_69017

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x^2 + 4 * x - 4) = (3 * x) / (3 * x - 2) ↔ x = 1 / 3 ∨ x = -2 := by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l690_69017


namespace NUMINAMATH_GPT_positive_difference_even_odd_sums_l690_69095

noncomputable def sum_first_n_even (n : ℕ) : ℕ :=
  2 * (n * (n + 1)) / 2

noncomputable def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem positive_difference_even_odd_sums :
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sum_even - sum_odd = 250 :=
by
  let sum_even := sum_first_n_even 25
  let sum_odd := sum_first_n_odd 20
  sorry

end NUMINAMATH_GPT_positive_difference_even_odd_sums_l690_69095


namespace NUMINAMATH_GPT_snowman_volume_l690_69046

theorem snowman_volume
  (r1 r2 r3 : ℝ)
  (volume : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h_volume : volume = (4.0 / 3.0) * Real.pi * (r1 ^ 3 + r2 ^ 3 + r3 ^ 3)) :
  volume = (1124.0 / 3.0) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_snowman_volume_l690_69046


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_l690_69075

noncomputable def a_8 : ℕ :=
  let a₁ := 8
  let r := 2
  a₁ * r^(8-1)

theorem geometric_sequence_eighth_term : a_8 = 1024 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_l690_69075


namespace NUMINAMATH_GPT_vance_family_stamp_cost_difference_l690_69089

theorem vance_family_stamp_cost_difference :
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    cost_daffodil - cost_rooster = 0.75 :=
by
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    show cost_daffodil - cost_rooster = 0.75
    sorry

end NUMINAMATH_GPT_vance_family_stamp_cost_difference_l690_69089


namespace NUMINAMATH_GPT_percentage_increase_in_length_l690_69072

theorem percentage_increase_in_length (L B : ℝ) (hB : 0 < B) (hL : 0 < L) :
  (1 + x / 100) * 1.22 = 1.3542 -> x = 11.016393 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_length_l690_69072


namespace NUMINAMATH_GPT_xy_sum_correct_l690_69015

theorem xy_sum_correct (x y : ℝ) 
  (h : (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3) : 
  x + y = 26.5 :=
by
  sorry

end NUMINAMATH_GPT_xy_sum_correct_l690_69015


namespace NUMINAMATH_GPT_increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l690_69090

noncomputable def f (x : ℝ) := x ^ 2 * Real.exp x - Real.log x

theorem increasing_f_for_x_ge_1 : ∀ (x : ℝ), x ≥ 1 → ∀ y > x, f y > f x :=
by
  sorry

theorem f_gt_1_for_x_gt_0 : ∀ (x : ℝ), x > 0 → f x > 1 :=
by
  sorry

end NUMINAMATH_GPT_increasing_f_for_x_ge_1_f_gt_1_for_x_gt_0_l690_69090


namespace NUMINAMATH_GPT_elements_in_set_C_l690_69052

-- Definitions and main theorem
variables (C D : Finset ℕ)  -- Define sets C and D as finite sets of natural numbers
open BigOperators    -- Opens notation for finite sums

-- Given conditions as premises
def condition1 (c d : ℕ) : Prop := c = 3 * d
def condition2 (C D : Finset ℕ) : Prop := (C ∪ D).card = 4500
def condition3 (C D : Finset ℕ) : Prop := (C ∩ D).card = 1200

-- Theorem statement to be proven
theorem elements_in_set_C (c d : ℕ) (h1 : condition1 c d)
  (h2 : ∀ (C D : Finset ℕ), condition2 C D)
  (h3 : ∀ (C D : Finset ℕ), condition3 C D) :
  c = 4275 :=
sorry  -- proof to be completed

end NUMINAMATH_GPT_elements_in_set_C_l690_69052


namespace NUMINAMATH_GPT_find_initial_men_l690_69080

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end NUMINAMATH_GPT_find_initial_men_l690_69080


namespace NUMINAMATH_GPT_margie_change_l690_69067

theorem margie_change :
  let num_apples := 5
  let cost_per_apple := 0.30
  let discount := 0.10
  let amount_paid := 10.00
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount)
  let change_received := amount_paid - discounted_cost
  change_received = 8.65 := sorry

end NUMINAMATH_GPT_margie_change_l690_69067


namespace NUMINAMATH_GPT_add_base_12_l690_69029

def a_in_base_10 := 10
def b_in_base_10 := 11
def c_base := 12

theorem add_base_12 : 
  let a := 10
  let b := 11
  (3 * c_base ^ 2 + 12 * c_base + 5) + (2 * c_base ^ 2 + a * c_base + b) = 6 * c_base ^ 2 + 3 * c_base + 4 :=
by
  sorry

end NUMINAMATH_GPT_add_base_12_l690_69029


namespace NUMINAMATH_GPT_max_value_of_x_times_one_minus_2x_l690_69073

theorem max_value_of_x_times_one_minus_2x : 
  ∀ x : ℝ, 0 < x ∧ x < 1 / 2 → x * (1 - 2 * x) ≤ 1 / 8 :=
by
  intro x 
  intro hx
  sorry

end NUMINAMATH_GPT_max_value_of_x_times_one_minus_2x_l690_69073


namespace NUMINAMATH_GPT_zero_polynomial_is_solution_l690_69062

noncomputable def polynomial_zero (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2))) → p = 0

theorem zero_polynomial_is_solution : ∀ p : Polynomial ℝ, (∀ x : ℝ, x ≠ 0 → (p.eval x)^2 + (p.eval (1/x))^2 = (p.eval (x^2)) * (p.eval (1/(x^2)))) → p = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_polynomial_is_solution_l690_69062


namespace NUMINAMATH_GPT_positive_difference_proof_l690_69081

noncomputable def solve_system : Prop :=
  ∃ (x y : ℝ), 
  (x + y = 40) ∧ 
  (3 * y - 4 * x = 10) ∧ 
  abs (y - x) = 8.58

theorem positive_difference_proof : solve_system := 
  sorry

end NUMINAMATH_GPT_positive_difference_proof_l690_69081


namespace NUMINAMATH_GPT_determinant_scaled_l690_69070

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end NUMINAMATH_GPT_determinant_scaled_l690_69070
