import Mathlib

namespace NUMINAMATH_GPT_solution_exists_l725_72539

theorem solution_exists (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (gcd_ca : Nat.gcd c a = 1) (gcd_cb : Nat.gcd c b = 1) : 
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^a + y^b = z^c :=
sorry

end NUMINAMATH_GPT_solution_exists_l725_72539


namespace NUMINAMATH_GPT_fruit_vendor_total_l725_72590

theorem fruit_vendor_total (lemons_dozen avocados_dozen : ℝ) (dozen_size : ℝ) 
  (lemons : ℝ) (avocados : ℝ) (total_fruits : ℝ) 
  (h1 : lemons_dozen = 2.5) (h2 : avocados_dozen = 5) 
  (h3 : dozen_size = 12) (h4 : lemons = lemons_dozen * dozen_size) 
  (h5 : avocados = avocados_dozen * dozen_size) 
  (h6 : total_fruits = lemons + avocados) : 
  total_fruits = 90 := 
sorry

end NUMINAMATH_GPT_fruit_vendor_total_l725_72590


namespace NUMINAMATH_GPT_units_digit_of_a_l725_72541

theorem units_digit_of_a :
  (2003^2004 - 2004^2003) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_a_l725_72541


namespace NUMINAMATH_GPT_percent_of_d_is_e_l725_72504

variable (a b c d e : ℝ)
variable (h1 : d = 0.40 * a)
variable (h2 : d = 0.35 * b)
variable (h3 : e = 0.50 * b)
variable (h4 : e = 0.20 * c)
variable (h5 : c = 0.30 * a)
variable (h6 : c = 0.25 * b)

theorem percent_of_d_is_e : (e / d) * 100 = 15 :=
by sorry

end NUMINAMATH_GPT_percent_of_d_is_e_l725_72504


namespace NUMINAMATH_GPT_neg_p_necessary_not_sufficient_neg_q_l725_72573

def p (x : ℝ) := abs x < 1
def q (x : ℝ) := x^2 + x - 6 < 0

theorem neg_p_necessary_not_sufficient_neg_q :
  (¬ (∃ x, p x)) → (¬ (∃ x, q x)) ∧ ¬ ((¬ (∃ x, p x)) → (¬ (∃ x, q x))) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_necessary_not_sufficient_neg_q_l725_72573


namespace NUMINAMATH_GPT_neg_p_is_correct_l725_72518

def is_positive_integer (x : ℕ) : Prop := x > 0

def proposition_p (x : ℕ) : Prop := (1 / 2 : ℝ) ^ x ≤ 1 / 2

def negation_of_p : Prop := ∃ x : ℕ, is_positive_integer x ∧ ¬ proposition_p x

theorem neg_p_is_correct : negation_of_p :=
sorry

end NUMINAMATH_GPT_neg_p_is_correct_l725_72518


namespace NUMINAMATH_GPT_sum_of_squares_pattern_l725_72554

theorem sum_of_squares_pattern (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_pattern_l725_72554


namespace NUMINAMATH_GPT_friedEdgeProb_l725_72547

-- Define a data structure for positions on the grid
inductive Pos
| A1 | A2 | A3 | A4
| B1 | B2 | B3 | B4
| C1 | C2 | C3 | C4
| D1 | D2 | D3 | D4
deriving DecidableEq, Repr

-- Define whether a position is an edge square (excluding corners)
def isEdge : Pos → Prop
| Pos.A2 | Pos.A3 | Pos.B1 | Pos.B4 | Pos.C1 | Pos.C4 | Pos.D2 | Pos.D3 => True
| _ => False

-- Define the initial state and max hops
def initialState := Pos.B2
def maxHops := 5

-- Define the recursive probability function (details omitted for brevity)
noncomputable def probabilityEdge (p : Pos) (hops : Nat) : ℚ := sorry

-- The proof problem statement
theorem friedEdgeProb :
  probabilityEdge initialState maxHops = 94 / 256 := sorry

end NUMINAMATH_GPT_friedEdgeProb_l725_72547


namespace NUMINAMATH_GPT_expand_expression_l725_72568

theorem expand_expression : 
  ∀ (x : ℝ), (7 * x^3 - 5 * x + 2) * 4 * x^2 = 28 * x^5 - 20 * x^3 + 8 * x^2 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_expand_expression_l725_72568


namespace NUMINAMATH_GPT_percentage_increase_l725_72516

variable {x y : ℝ}
variable {P : ℝ} -- percentage

theorem percentage_increase (h1 : y = x * (1 + P / 100)) (h2 : x = y * 0.5882352941176471) : P = 70 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l725_72516


namespace NUMINAMATH_GPT_range_of_a_for_negative_root_l725_72561

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 7^(x + 1) - 7^x * a - a - 5 = 0) ↔ -5 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_negative_root_l725_72561


namespace NUMINAMATH_GPT_inv_38_mod_53_l725_72555

theorem inv_38_mod_53 (h : 15 * 31 % 53 = 1) : ∃ x : ℤ, 38 * x % 53 = 1 ∧ (x % 53 = 22) :=
by
  sorry

end NUMINAMATH_GPT_inv_38_mod_53_l725_72555


namespace NUMINAMATH_GPT_distance_between_vertices_hyperbola_l725_72505

-- Defining the hyperbola equation and necessary constants
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2) / 64 - (y^2) / 81 = 1

-- Proving the distance between the vertices is 16
theorem distance_between_vertices_hyperbola : ∀ x y : ℝ, hyperbola_eq x y → 16 = 16 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_distance_between_vertices_hyperbola_l725_72505


namespace NUMINAMATH_GPT_people_got_on_at_third_stop_l725_72545

theorem people_got_on_at_third_stop :
  let people_1st_stop := 10
  let people_off_2nd_stop := 3
  let twice_people_1st_stop := 2 * people_1st_stop
  let people_off_3rd_stop := 18
  let people_after_3rd_stop := 12

  let people_after_1st_stop := people_1st_stop
  let people_after_2nd_stop := (people_after_1st_stop - people_off_2nd_stop) + twice_people_1st_stop
  let people_after_3rd_stop_but_before_new_ones := people_after_2nd_stop - people_off_3rd_stop
  let people_on_at_3rd_stop := people_after_3rd_stop - people_after_3rd_stop_but_before_new_ones

  people_on_at_3rd_stop = 3 := 
by
  sorry

end NUMINAMATH_GPT_people_got_on_at_third_stop_l725_72545


namespace NUMINAMATH_GPT_evaluate_expression_l725_72581

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l725_72581


namespace NUMINAMATH_GPT_remainder_142_to_14_l725_72598

theorem remainder_142_to_14 (N k : ℤ) 
  (h : N = 142 * k + 110) : N % 14 = 8 :=
sorry

end NUMINAMATH_GPT_remainder_142_to_14_l725_72598


namespace NUMINAMATH_GPT_sum_of_digits_in_base_7_l725_72576

theorem sum_of_digits_in_base_7 (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (hA7 : A < 7) (hB7 : B < 7) (hC7 : C < 7)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eqn : A * 49 + B * 7 + C + (B * 7 + C) = A * 49 + C * 7 + A) : 
  (A + B + C) = 14 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_in_base_7_l725_72576


namespace NUMINAMATH_GPT_find_fourth_term_in_sequence_l725_72592

theorem find_fourth_term_in_sequence (x: ℤ) (h1: 86 - 8 = 78) (h2: 2 - 86 = -84) (h3: x - 2 = -90) (h4: -12 - x = 76):
  x = -88 :=
sorry

end NUMINAMATH_GPT_find_fourth_term_in_sequence_l725_72592


namespace NUMINAMATH_GPT_volume_tetrahedron_PXYZ_l725_72589

noncomputable def volume_of_tetrahedron_PXYZ (x y z : ℝ) : ℝ :=
  (1 / 6) * x * y * z

theorem volume_tetrahedron_PXYZ :
  ∃ (x y z : ℝ), (x^2 + y^2 = 49) ∧ (y^2 + z^2 = 64) ∧ (z^2 + x^2 = 81) ∧
  volume_of_tetrahedron_PXYZ (Real.sqrt x) (Real.sqrt y) (Real.sqrt z) = 4 * Real.sqrt 11 := 
by {
  sorry
}

end NUMINAMATH_GPT_volume_tetrahedron_PXYZ_l725_72589


namespace NUMINAMATH_GPT_max_minus_min_all_three_languages_l725_72519

def student_population := 1500

def english_students (e : ℕ) : Prop := 1050 ≤ e ∧ e ≤ 1125
def spanish_students (s : ℕ) : Prop := 750 ≤ s ∧ s ≤ 900
def german_students (g : ℕ) : Prop := 300 ≤ g ∧ g ≤ 450

theorem max_minus_min_all_three_languages (e s g e_s e_g s_g e_s_g : ℕ) 
    (he : english_students e)
    (hs : spanish_students s)
    (hg : german_students g)
    (pie : e + s + g - e_s - e_g - s_g + e_s_g = student_population) 
    : (M - m = 450) :=
sorry

end NUMINAMATH_GPT_max_minus_min_all_three_languages_l725_72519


namespace NUMINAMATH_GPT_line_parallel_to_x_axis_l725_72553

variable (k : ℝ)

theorem line_parallel_to_x_axis :
  let point1 := (3, 2 * k + 1)
  let point2 := (8, 4 * k - 5)
  (point1.2 = point2.2) ↔ (k = 3) :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_to_x_axis_l725_72553


namespace NUMINAMATH_GPT_product_diff_squares_l725_72558

theorem product_diff_squares (a b c d x1 y1 x2 y2 x3 y3 x4 y4 : ℕ) 
  (ha : a = x1^2 - y1^2) 
  (hb : b = x2^2 - y2^2) 
  (hc : c = x3^2 - y3^2) 
  (hd : d = x4^2 - y4^2)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  ∃ X Y : ℕ, a * b * c * d = X^2 - Y^2 :=
by
  sorry

end NUMINAMATH_GPT_product_diff_squares_l725_72558


namespace NUMINAMATH_GPT_absent_children_l725_72572

-- Definitions
def total_children := 840
def bananas_per_child_present := 4
def bananas_per_child_if_all_present := 2
def total_bananas_if_all_present := total_children * bananas_per_child_if_all_present

-- The theorem to prove
theorem absent_children (A : ℕ) (P : ℕ) :
  P = total_children - A →
  total_bananas_if_all_present = P * bananas_per_child_present →
  A = 420 :=
by
  sorry

end NUMINAMATH_GPT_absent_children_l725_72572


namespace NUMINAMATH_GPT_mean_equality_and_find_y_l725_72542

theorem mean_equality_and_find_y : 
  (8 + 9 + 18) / 3 = (15 + (25 / 3)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_mean_equality_and_find_y_l725_72542


namespace NUMINAMATH_GPT_sum_of_c_and_d_l725_72591

theorem sum_of_c_and_d (c d : ℝ) :
  (∀ x : ℝ, x ≠ 2 → x ≠ -3 → (x - 2) * (x + 3) = x^2 + c * x + d) →
  c + d = -5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_c_and_d_l725_72591


namespace NUMINAMATH_GPT_solution_concentration_l725_72521

theorem solution_concentration (C : ℝ) :
  (0.16 + 0.01 * C * 2 = 0.36) ↔ (C = 10) :=
by
  sorry

end NUMINAMATH_GPT_solution_concentration_l725_72521


namespace NUMINAMATH_GPT_perfect_squares_in_interval_l725_72560

theorem perfect_squares_in_interval (s : Set Int) (h1 : ∃ a : Nat, ∀ x ∈ s, a^4 ≤ x ∧ x ≤ (a+9)^4)
                                     (h2 : ∃ b : Nat, ∀ x ∈ s, b^3 ≤ x ∧ x ≤ (b+99)^3) :
  ∃ c : Nat, c ≥ 2000 ∧ ∀ x ∈ s, x = c^2 :=
sorry

end NUMINAMATH_GPT_perfect_squares_in_interval_l725_72560


namespace NUMINAMATH_GPT_average_study_diff_l725_72550

theorem average_study_diff (diff : List ℤ) (h_diff : diff = [15, -5, 25, -10, 5, 20, -15]) :
  (List.sum diff) / (List.length diff) = 5 := by
  sorry

end NUMINAMATH_GPT_average_study_diff_l725_72550


namespace NUMINAMATH_GPT_evaluate_f_of_composed_g_l725_72595

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 2

theorem evaluate_f_of_composed_g :
  f (2 + g 3) = 17 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_of_composed_g_l725_72595


namespace NUMINAMATH_GPT_geom_mean_4_16_l725_72578

theorem geom_mean_4_16 (x : ℝ) (h : x^2 = 4 * 16) : x = 8 ∨ x = -8 :=
by
  sorry

end NUMINAMATH_GPT_geom_mean_4_16_l725_72578


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l725_72559

noncomputable def smallest_angle : ℝ := 25
noncomputable def largest_angle : ℝ := 105
noncomputable def num_angles : ℕ := 5

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, (smallest_angle + (num_angles - 1) * d = largest_angle) ∧ d = 20 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l725_72559


namespace NUMINAMATH_GPT_total_value_of_bills_in_cash_drawer_l725_72584

-- Definitions based on conditions
def total_bills := 54
def five_dollar_bills := 20
def twenty_dollar_bills := total_bills - five_dollar_bills
def value_of_five_dollar_bills := 5
def value_of_twenty_dollar_bills := 20
def total_value_of_five_dollar_bills := five_dollar_bills * value_of_five_dollar_bills
def total_value_of_twenty_dollar_bills := twenty_dollar_bills * value_of_twenty_dollar_bills

-- Statement to prove
theorem total_value_of_bills_in_cash_drawer :
  total_value_of_five_dollar_bills + total_value_of_twenty_dollar_bills = 780 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_value_of_bills_in_cash_drawer_l725_72584


namespace NUMINAMATH_GPT_percent_of_value_l725_72513

theorem percent_of_value (decimal_form : Real) (value : Nat) (expected_result : Real) : 
  decimal_form = 0.25 ∧ value = 300 ∧ expected_result = 75 → 
  decimal_form * value = expected_result := by
  sorry

end NUMINAMATH_GPT_percent_of_value_l725_72513


namespace NUMINAMATH_GPT_graph_properties_l725_72531

theorem graph_properties (k b : ℝ) (hk : k ≠ 0) (hb : b ≠ 0) (positive_kb : k * b > 0) :
  (∃ (f g : ℝ → ℝ),
    (∀ x, f x = k * x + b) ∧
    (∀ x (hx : x ≠ 0), g x = k * b / x) ∧
    -- Under the given conditions, the graphs must match option (B)
    (True)) := sorry

end NUMINAMATH_GPT_graph_properties_l725_72531


namespace NUMINAMATH_GPT_function_range_l725_72564

def function_defined (x : ℝ) : Prop := x ≠ 5

theorem function_range (x : ℝ) : x ≠ 5 → function_defined x :=
by
  intro h
  exact h

end NUMINAMATH_GPT_function_range_l725_72564


namespace NUMINAMATH_GPT_quadratic_root_q_value_l725_72579

theorem quadratic_root_q_value
  (p q : ℝ)
  (h1 : ∃ r : ℝ, r = -3 ∧ 3 * r^2 + p * r + q = 0)
  (h2 : ∃ s : ℝ, -3 + s = -2) :
  q = -9 :=
sorry

end NUMINAMATH_GPT_quadratic_root_q_value_l725_72579


namespace NUMINAMATH_GPT_ratio_of_inverse_l725_72552

theorem ratio_of_inverse (a b c d : ℝ) (h : ∀ x, (3 * (a * x + b) / (c * x + d) - 2) / ((a * x + b) / (c * x + d) + 4) = x) : 
  a / c = -4 :=
sorry

end NUMINAMATH_GPT_ratio_of_inverse_l725_72552


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l725_72594

theorem common_ratio_of_geometric_series (a r : ℝ) (r_pos : 0 < r) (r_lt_one : r < 1) 
(h : (a / (1 - r)) = 81 * (a * r^4 / (1 - r))) : r = 1 / 3 :=
by
  have h_simplified : r^4 = 1 / 81 :=
    by
      sorry
  have r_value : r = (1 / 3) := by
      sorry
  exact r_value

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l725_72594


namespace NUMINAMATH_GPT_adult_tickets_sold_l725_72511

open Nat

theorem adult_tickets_sold (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) :
  A = 130 :=
by
  sorry

end NUMINAMATH_GPT_adult_tickets_sold_l725_72511


namespace NUMINAMATH_GPT_functional_equation_solution_l725_72549

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) → (∀ x : ℝ, f x = x) :=
by
  intros f h x
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l725_72549


namespace NUMINAMATH_GPT_fraction_to_decimal_l725_72571

theorem fraction_to_decimal : (5 : ℝ) / 16 = 0.3125 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l725_72571


namespace NUMINAMATH_GPT_gcd_g102_g103_eq_one_l725_72508

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_g102_g103_eq_one_l725_72508


namespace NUMINAMATH_GPT_pagoda_lanterns_l725_72507

-- Definitions
def top_layer_lanterns (a₁ : ℕ) : ℕ := a₁
def bottom_layer_lanterns (a₁ : ℕ) : ℕ := a₁ * 2^6
def sum_of_lanterns (a₁ : ℕ) : ℕ := (a₁ * (1 - 2^7)) / (1 - 2)
def total_lanterns : ℕ := 381
def layers : ℕ := 7
def common_ratio : ℕ := 2

-- Problem Statement
theorem pagoda_lanterns (a₁ : ℕ) (h : sum_of_lanterns a₁ = total_lanterns) : 
  top_layer_lanterns a₁ + bottom_layer_lanterns a₁ = 195 := sorry

end NUMINAMATH_GPT_pagoda_lanterns_l725_72507


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l725_72500

theorem largest_angle_of_triangle (x : ℝ) (h_ratio : (5 * x) + (6 * x) + (7 * x) = 180) :
  7 * x = 70 := 
sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l725_72500


namespace NUMINAMATH_GPT_An_is_integer_for_all_n_l725_72512

noncomputable def sin_theta (a b : ℕ) : ℝ :=
  if h : a^2 + b^2 ≠ 0 then (2 * a * b) / (a^2 + b^2) else 0

theorem An_is_integer_for_all_n (a b : ℕ) (n : ℕ) (h₁ : a > b) (h₂ : 0 < sin_theta a b) (h₃ : sin_theta a b < 1) :
  ∃ k : ℤ, ∀ n : ℕ, ((a^2 + b^2)^n * sin_theta a b) = k :=
sorry

end NUMINAMATH_GPT_An_is_integer_for_all_n_l725_72512


namespace NUMINAMATH_GPT_number_a_eq_223_l725_72563

theorem number_a_eq_223 (A B : ℤ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 :=
by
  sorry

end NUMINAMATH_GPT_number_a_eq_223_l725_72563


namespace NUMINAMATH_GPT_part1_part2_l725_72577

def A : Set ℝ := {x | (x + 4) * (x - 2) > 0}
def B : Set ℝ := {y | ∃ x : ℝ, y = (x - 1)^2 + 1}
def C (a : ℝ) : Set ℝ := {x | -4 ≤ x ∧ x ≤ a}

theorem part1 : A ∩ B = {x : ℝ | x > 2} := 
by sorry

theorem part2 (a : ℝ) (h : (C a \ A) ⊆ C a) : 2 ≤ a :=
by sorry

end NUMINAMATH_GPT_part1_part2_l725_72577


namespace NUMINAMATH_GPT_third_consecutive_even_sum_52_l725_72570

theorem third_consecutive_even_sum_52
  (x : ℤ)
  (h : x + (x + 2) + (x + 4) + (x + 6) = 52) :
  x + 4 = 14 :=
by
  sorry

end NUMINAMATH_GPT_third_consecutive_even_sum_52_l725_72570


namespace NUMINAMATH_GPT_value_does_not_appear_l725_72567

theorem value_does_not_appear : 
  let f : ℕ → ℕ := fun x => 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
  let x := 2
  let values := [14, 31, 64, 129, 259]
  127 ∉ values :=
by
  sorry

end NUMINAMATH_GPT_value_does_not_appear_l725_72567


namespace NUMINAMATH_GPT_quadratic_equation_no_real_roots_l725_72502

theorem quadratic_equation_no_real_roots :
  ∀ (x : ℝ), ¬ (x^2 - 2 * x + 3 = 0) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_quadratic_equation_no_real_roots_l725_72502


namespace NUMINAMATH_GPT_complex_expression_equality_l725_72588

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_expression_equality_l725_72588


namespace NUMINAMATH_GPT_arithmetic_sequence_a3a6_l725_72525

theorem arithmetic_sequence_a3a6 (a : ℕ → ℤ)
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_inc : ∀ n, a n < a (n + 1))
  (h_eq : a 3 * a 4 = 45): 
  a 2 * a 5 = 13 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3a6_l725_72525


namespace NUMINAMATH_GPT_bruce_bhishma_meet_again_l725_72517

theorem bruce_bhishma_meet_again (L S_B S_H : ℕ) (hL : L = 600) (hSB : S_B = 30) (hSH : S_H = 20) : 
  ∃ t : ℕ, t = 60 ∧ (t * S_B - t * S_H) % L = 0 :=
by
  sorry

end NUMINAMATH_GPT_bruce_bhishma_meet_again_l725_72517


namespace NUMINAMATH_GPT_binkie_gemstones_l725_72515

variables (F B S : ℕ)

theorem binkie_gemstones :
  (B = 4 * F) →
  (S = (1 / 2 : ℝ) * F - 2) →
  (S = 1) →
  B = 24 :=
by
  sorry

end NUMINAMATH_GPT_binkie_gemstones_l725_72515


namespace NUMINAMATH_GPT_choose_bar_length_l725_72527

theorem choose_bar_length (x : ℝ) (h1 : 1 < x) (h2 : x < 4) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_choose_bar_length_l725_72527


namespace NUMINAMATH_GPT_object_speed_approx_l725_72596

theorem object_speed_approx :
  ∃ (speed : ℝ), abs (speed - 27.27) < 0.01 ∧
  (∀ (d : ℝ) (t : ℝ)
    (m : ℝ), 
    d = 80 ∧ t = 2 ∧ m = 5280 →
    speed = (d / m) / (t / 3600)) :=
by 
  sorry

end NUMINAMATH_GPT_object_speed_approx_l725_72596


namespace NUMINAMATH_GPT_exists_monochromatic_rectangle_l725_72540

theorem exists_monochromatic_rectangle 
  (coloring : ℤ × ℤ → Prop)
  (h : ∀ p : ℤ × ℤ, coloring p = red ∨ coloring p = blue)
  : ∃ (a b c d : ℤ × ℤ), (a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∧ (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end NUMINAMATH_GPT_exists_monochromatic_rectangle_l725_72540


namespace NUMINAMATH_GPT_problem_statement_l725_72551

theorem problem_statement (p q r : ℝ) (h1 : p + q + r = 5) (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l725_72551


namespace NUMINAMATH_GPT_students_with_B_l725_72556

theorem students_with_B (students_jacob : ℕ) (students_B_jacob : ℕ) (students_smith : ℕ) (ratio_same : (students_B_jacob / students_jacob : ℚ) = 2 / 5) : 
  ∃ y : ℕ, (y / students_smith : ℚ) = 2 / 5 ∧ y = 12 :=
by 
  use 12
  sorry

end NUMINAMATH_GPT_students_with_B_l725_72556


namespace NUMINAMATH_GPT_minimum_components_needed_l725_72575

-- Define the parameters of the problem
def production_cost_per_component : ℝ := 80
def shipping_cost_per_component : ℝ := 7
def fixed_monthly_cost : ℝ := 16500
def selling_price_per_component : ℝ := 198.33

-- Define the total cost as a function of the number of components
def total_cost (x : ℝ) : ℝ :=
  fixed_monthly_cost + (production_cost_per_component + shipping_cost_per_component) * x

-- Define the revenue as a function of the number of components
def revenue (x : ℝ) : ℝ :=
  selling_price_per_component * x

-- Define the theorem to be proved
theorem minimum_components_needed (x : ℝ) : x = 149 ↔ total_cost x ≤ revenue x := sorry

end NUMINAMATH_GPT_minimum_components_needed_l725_72575


namespace NUMINAMATH_GPT_team_selection_l725_72537

theorem team_selection :
  let teachers := 5
  let students := 10
  (teachers * students = 50) :=
by
  sorry

end NUMINAMATH_GPT_team_selection_l725_72537


namespace NUMINAMATH_GPT_arc_length_l725_72574

-- Define the conditions
def radius (r : ℝ) := 2 * r + 2 * r = 8
def central_angle (θ : ℝ) := θ = 2 -- Given the central angle

-- Define the length of the arc
def length_of_arc (l r : ℝ) := l = r * 2

-- Theorem stating that given the sector conditions, the length of the arc is 4 cm
theorem arc_length (r l : ℝ) (h1 : central_angle 2) (h2 : radius r) (h3 : length_of_arc l r) : l = 4 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_l725_72574


namespace NUMINAMATH_GPT_find_p_series_l725_72536

theorem find_p_series (p : ℝ) (h : 5 + (5 + p) / 5 + (5 + 2 * p) / 5^2 + (5 + 3 * p) / 5^3 + ∑' (n : ℕ), (5 + (n + 1) * p) / 5^(n + 1) = 10) : p = 16 :=
sorry

end NUMINAMATH_GPT_find_p_series_l725_72536


namespace NUMINAMATH_GPT_maximize_revenue_l725_72566

theorem maximize_revenue (p : ℝ) (hp : p ≤ 30) :
  (p = 12 ∨ p = 13) → (∀ p : ℤ, p ≤ 30 → 200 * p - 8 * p * p ≤ 1248) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_maximize_revenue_l725_72566


namespace NUMINAMATH_GPT_find_number_subtracted_l725_72530

-- Given a number x, where the ratio of the two natural numbers is 6:5,
-- and another number y is subtracted to both numbers such that the new ratio becomes 5:4,
-- and the larger number exceeds the smaller number by 5,
-- prove that y = 5.
theorem find_number_subtracted (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
by sorry

end NUMINAMATH_GPT_find_number_subtracted_l725_72530


namespace NUMINAMATH_GPT_subset_proof_l725_72546

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (x^2 + 2*x + 1)}

-- The problem statement
theorem subset_proof : M ⊆ N ∧ ∃ y ∈ N, y ∉ M :=
by
  sorry

end NUMINAMATH_GPT_subset_proof_l725_72546


namespace NUMINAMATH_GPT_largest_positive_integer_n_l725_72510

def binary_operation (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_n (x : ℤ) (h : x = -15) : 
  ∃ (n : ℤ), n > 0 ∧ binary_operation n < x ∧ ∀ m > 0, binary_operation m < x → m ≤ n :=
by
  sorry

end NUMINAMATH_GPT_largest_positive_integer_n_l725_72510


namespace NUMINAMATH_GPT_find_abs_product_l725_72514

noncomputable def distinct_nonzero_real (a b c : ℝ) : Prop :=
a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

theorem find_abs_product (a b c : ℝ) (h1 : distinct_nonzero_real a b c) 
(h2 : a + 1/(b^2) = b + 1/(c^2))
(h3 : b + 1/(c^2) = c + 1/(a^2)) :
  |a * b * c| = 1 :=
sorry

end NUMINAMATH_GPT_find_abs_product_l725_72514


namespace NUMINAMATH_GPT_negation_seated_l725_72528

variable (Person : Type) (in_room : Person → Prop) (seated : Person → Prop)

theorem negation_seated :
  ¬ (∀ x, in_room x → seated x) ↔ ∃ x, in_room x ∧ ¬ seated x :=
by sorry

end NUMINAMATH_GPT_negation_seated_l725_72528


namespace NUMINAMATH_GPT_xy_equals_252_l725_72526

-- Definitions and conditions
variables (x y : ℕ) -- positive integers
variable (h1 : x + y = 36)
variable (h2 : 4 * x * y + 12 * x = 5 * y + 390)

-- Statement of the problem
theorem xy_equals_252 (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by 
  sorry

end NUMINAMATH_GPT_xy_equals_252_l725_72526


namespace NUMINAMATH_GPT_identify_INPUT_statement_l725_72538

/-- Definition of the PRINT statement --/
def is_PRINT_statement (s : String) : Prop := s = "PRINT"

/-- Definition of the INPUT statement --/
def is_INPUT_statement (s : String) : Prop := s = "INPUT"

/-- Definition of the IF statement --/
def is_IF_statement (s : String) : Prop := s = "IF"

/-- Definition of the WHILE statement --/
def is_WHILE_statement (s : String) : Prop := s = "WHILE"

/-- Proof statement that the INPUT statement is the one for input --/
theorem identify_INPUT_statement (s : String) (h1 : is_PRINT_statement "PRINT") (h2: is_INPUT_statement "INPUT") (h3 : is_IF_statement "IF") (h4 : is_WHILE_statement "WHILE") : s = "INPUT" :=
sorry

end NUMINAMATH_GPT_identify_INPUT_statement_l725_72538


namespace NUMINAMATH_GPT_min_value_f_l725_72535

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * Real.arcsin x + 3

theorem min_value_f (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) (hmax : ∃ x, f a b x = 10) : ∃ y, f a b y = -4 := by
  sorry

end NUMINAMATH_GPT_min_value_f_l725_72535


namespace NUMINAMATH_GPT_find_a_l725_72543

theorem find_a (a x y : ℝ) 
  (h1 : (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0) 
  (h2 : (x + 2)^2 + (y + 4)^2 = a) 
  (h3 : ∃! x y, (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0 ∧ (x + 2)^2 + (y + 4)^2 = a) :
  a = 9 ∨ a = 23 + 4 * Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_find_a_l725_72543


namespace NUMINAMATH_GPT_area_at_stage_7_l725_72544

-- Define the size of one square added at each stage
def square_size : ℕ := 4

-- Define the area of one square
def area_of_one_square : ℕ := square_size * square_size

-- Define the number of stages
def number_of_stages : ℕ := 7

-- Define the total area at a given stage
def total_area (n : ℕ) : ℕ := n * area_of_one_square

-- The theorem which proves the area of the rectangle at Stage 7
theorem area_at_stage_7 : total_area number_of_stages = 112 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_area_at_stage_7_l725_72544


namespace NUMINAMATH_GPT_multiples_of_7_between_15_and_200_l725_72593

theorem multiples_of_7_between_15_and_200 : ∃ n : ℕ, n = 26 ∧ ∃ (a₁ a_n d : ℕ), 
  a₁ = 21 ∧ a_n = 196 ∧ d = 7 ∧ (a₁ + (n - 1) * d = a_n) := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_7_between_15_and_200_l725_72593


namespace NUMINAMATH_GPT_chickens_after_9_years_l725_72529

-- Definitions from the conditions
def annual_increase : ℕ := 150
def current_chickens : ℕ := 550
def years : ℕ := 9

-- Lean statement for the proof
theorem chickens_after_9_years : current_chickens + annual_increase * years = 1900 :=
by
  sorry

end NUMINAMATH_GPT_chickens_after_9_years_l725_72529


namespace NUMINAMATH_GPT_range_of_composite_function_l725_72534

noncomputable def range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, y = (1/2) ^ (|x + 1|)}

theorem range_of_composite_function : range_of_function = Set.Ioc 0 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_composite_function_l725_72534


namespace NUMINAMATH_GPT_marks_lost_per_wrong_answer_l725_72506

theorem marks_lost_per_wrong_answer 
  (marks_per_correct : ℕ)
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (wrong_answers : ℕ)
  (score_from_correct : ℕ := correct_answers * marks_per_correct)
  (remaining_marks : ℕ := score_from_correct - total_marks)
  (marks_lost_per_wrong : ℕ) :
  total_questions = correct_answers + wrong_answers →
  total_marks = 130 →
  correct_answers = 38 →
  total_questions = 60 →
  marks_per_correct = 4 →
  marks_lost_per_wrong * wrong_answers = remaining_marks →
  marks_lost_per_wrong = 1 := 
sorry

end NUMINAMATH_GPT_marks_lost_per_wrong_answer_l725_72506


namespace NUMINAMATH_GPT_find_functional_equation_solutions_l725_72582

theorem find_functional_equation_solutions :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → f x * f (y * f x) = f (x + y)) →
    (∃ a > 0, ∀ x > 0, f x = 1 / (1 + a * x) ∨ ∀ x > 0, f x = 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_functional_equation_solutions_l725_72582


namespace NUMINAMATH_GPT_new_person_weight_increase_avg_l725_72583

theorem new_person_weight_increase_avg
  (W : ℝ) -- total weight of the original 20 people
  (new_person_weight : ℝ) -- weight of the new person
  (h1 : (W - 80 + new_person_weight) = W + 20 * 15) -- condition given in the problem
  : new_person_weight = 380 := 
sorry

end NUMINAMATH_GPT_new_person_weight_increase_avg_l725_72583


namespace NUMINAMATH_GPT_function_identity_l725_72533

theorem function_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_function_identity_l725_72533


namespace NUMINAMATH_GPT_T_simplified_l725_72557

-- Define the polynomial expression T
def T (x : ℝ) : ℝ := (x-2)^4 - 4*(x-2)^3 + 6*(x-2)^2 - 4*(x-2) + 1

-- Prove that T simplifies to (x-3)^4
theorem T_simplified (x : ℝ) : T x = (x - 3)^4 := by
  sorry

end NUMINAMATH_GPT_T_simplified_l725_72557


namespace NUMINAMATH_GPT_part_I_part_II_l725_72580

noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) - (2 * x) / (x + 2)
noncomputable def g (x : ℝ) : ℝ := f x - (4 / (x + 2))

theorem part_I (x : ℝ) (h₀ : 0 < x) : f x > 0 := sorry

theorem part_II (a : ℝ) (h : ∀ x, g x < x + a) : -2 < a := sorry

end NUMINAMATH_GPT_part_I_part_II_l725_72580


namespace NUMINAMATH_GPT_mock_exam_girls_count_l725_72524

theorem mock_exam_girls_count
  (B G Bc Gc : ℕ)
  (h1: B + G = 400)
  (h2: Bc = 60 * B / 100)
  (h3: Gc = 80 * G / 100)
  (h4: Bc + Gc = 65 * 400 / 100)
  : G = 100 :=
sorry

end NUMINAMATH_GPT_mock_exam_girls_count_l725_72524


namespace NUMINAMATH_GPT_smallest_a_for_quadratic_poly_l725_72548

theorem smallest_a_for_quadratic_poly (a : ℕ) (a_pos : 0 < a) :
  (∃ b c : ℤ, ∀ x : ℝ, 0 < x ∧ x < 1 → a*x^2 + b*x + c = 0 → (2 : ℝ)^2 - (4 : ℝ)*(a * c) < 0 ∧ b^2 - 4*a*c ≥ 1) → a ≥ 5 := 
sorry

end NUMINAMATH_GPT_smallest_a_for_quadratic_poly_l725_72548


namespace NUMINAMATH_GPT_find_number_of_sides_l725_72597

theorem find_number_of_sides (n : ℕ) (h : n - (n * (n - 3)) / 2 = 3) : n = 3 := 
sorry

end NUMINAMATH_GPT_find_number_of_sides_l725_72597


namespace NUMINAMATH_GPT_elena_total_pens_l725_72532

theorem elena_total_pens 
  (cost_X : ℝ) (cost_Y : ℝ) (total_spent : ℝ) (num_brand_X : ℕ) (num_brand_Y : ℕ) (total_pens : ℕ)
  (h1 : cost_X = 4.0) 
  (h2 : cost_Y = 2.8) 
  (h3 : total_spent = 40.0) 
  (h4 : num_brand_X = 8) 
  (h5 : total_pens = num_brand_X + num_brand_Y) 
  (h6 : total_spent = num_brand_X * cost_X + num_brand_Y * cost_Y) :
  total_pens = 10 :=
sorry

end NUMINAMATH_GPT_elena_total_pens_l725_72532


namespace NUMINAMATH_GPT_axis_of_symmetry_imp_cond_l725_72509

-- Necessary definitions
variables {p q r s x y : ℝ}

-- Given conditions
def curve_eq (x y p q r s : ℝ) : Prop := y = (2 * p * x + q) / (r * x + 2 * s)
def axis_of_symmetry (x y : ℝ) : Prop := y = x

-- Main statement
theorem axis_of_symmetry_imp_cond (h1 : curve_eq x y p q r s) (h2 : axis_of_symmetry x y) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : p = -2 * s :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_imp_cond_l725_72509


namespace NUMINAMATH_GPT_compute_abc_l725_72587

theorem compute_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_sum : a + b + c = 30) (h_frac : (1 : ℚ) / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) : a * b * c = 1920 :=
by sorry

end NUMINAMATH_GPT_compute_abc_l725_72587


namespace NUMINAMATH_GPT_solve_system_l725_72586

theorem solve_system (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y = 4 * z) (h2 : x / y = 81) (h3 : x * z = 36) :
  x = 36 ∧ y = 4 / 9 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l725_72586


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l725_72569

variable (a : ℕ → ℝ) -- The geometric sequence {a_n}
variable (q : ℝ)     -- The common ratio

-- Conditions
axiom h1 : a 2 = 18
axiom h2 : a 4 = 8

theorem common_ratio_of_geometric_sequence :
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ q^2 = 4/9 → q = 2/3 ∨ q = -2/3 := by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l725_72569


namespace NUMINAMATH_GPT_exists_real_m_l725_72585

noncomputable def f (a : ℝ) (x : ℝ) := 4 * x + a * x ^ 2 - (2 / 3) * x ^ 3
noncomputable def g (x : ℝ) := 2 * x + (1 / 3) * x ^ 3

theorem exists_real_m (a : ℝ) (t : ℝ) (x1 x2 : ℝ) :
  (-1 : ℝ) ≤ a ∧ a ≤ 1 →
  (-1 : ℝ) ≤ t ∧ t ≤ 1 →
  f a x1 = g x1 ∧ f a x2 = g x2 →
  x1 ≠ 0 ∧ x2 ≠ 0 →
  x1 ≠ x2 →
  ∃ m : ℝ, (m ≥ 2 ∨ m ≤ -2) ∧ m^2 + t * m + 1 ≥ |x1 - x2| :=
sorry

end NUMINAMATH_GPT_exists_real_m_l725_72585


namespace NUMINAMATH_GPT_left_handed_rock_lovers_l725_72501

theorem left_handed_rock_lovers (total_people left_handed rock_music right_dislike_rock x : ℕ) :
  total_people = 30 →
  left_handed = 14 →
  rock_music = 20 →
  right_dislike_rock = 5 →
  (x + (left_handed - x) + (rock_music - x) + right_dislike_rock = total_people) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_left_handed_rock_lovers_l725_72501


namespace NUMINAMATH_GPT_machine_initial_value_l725_72565

-- Conditions
def initial_value (P : ℝ) : Prop := P * (0.75 ^ 2) = 4000

noncomputable def initial_market_value : ℝ := 4000 / (0.75 ^ 2)

-- Proof problem statement
theorem machine_initial_value (P : ℝ) (h : initial_value P) : P = 4000 / (0.75 ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_machine_initial_value_l725_72565


namespace NUMINAMATH_GPT_calc_15_op_and_op2_l725_72503

def op1 (x : ℤ) : ℤ := 10 - x
def op2 (x : ℤ) : ℤ := x - 10

theorem calc_15_op_and_op2 :
  op2 (op1 15) = -15 :=
by
  sorry

end NUMINAMATH_GPT_calc_15_op_and_op2_l725_72503


namespace NUMINAMATH_GPT_total_toys_l725_72522

theorem total_toys (m a t : ℕ) (h1 : a = m + 3 * m) (h2 : t = a + 2) (h3 : m = 6) : m + a + t = 56 := by
  sorry

end NUMINAMATH_GPT_total_toys_l725_72522


namespace NUMINAMATH_GPT_find_t_l725_72523

-- Definitions from the given conditions
def earning (hours : ℕ) (rate : ℕ) : ℕ := hours * rate

-- The main theorem based on the translated problem
theorem find_t
  (t : ℕ)
  (h1 : earning (t - 4) (3 * t - 7) = earning (3 * t - 12) (t - 3)) :
  t = 4 := 
sorry

end NUMINAMATH_GPT_find_t_l725_72523


namespace NUMINAMATH_GPT_selling_price_l725_72562

theorem selling_price (profit_percent : ℝ) (cost_price : ℝ) (h_profit : profit_percent = 5) (h_cp : cost_price = 2400) :
  let profit := (profit_percent / 100) * cost_price 
  let selling_price := cost_price + profit
  selling_price = 2520 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_l725_72562


namespace NUMINAMATH_GPT_smallest_collection_l725_72599

def Yoongi_collected : ℕ := 4
def Jungkook_collected : ℕ := 6 * 3
def Yuna_collected : ℕ := 5

theorem smallest_collection : Yoongi_collected = 4 ∧ Yoongi_collected ≤ Jungkook_collected ∧ Yoongi_collected ≤ Yuna_collected := by
  sorry

end NUMINAMATH_GPT_smallest_collection_l725_72599


namespace NUMINAMATH_GPT_nabla_2_3_2_eq_4099_l725_72520

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem nabla_2_3_2_eq_4099 : nabla (nabla 2 3) 2 = 4099 :=
by
  sorry

end NUMINAMATH_GPT_nabla_2_3_2_eq_4099_l725_72520
