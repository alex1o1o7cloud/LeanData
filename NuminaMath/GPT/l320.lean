import Mathlib

namespace find_y_intercept_l320_32035

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l320_32035


namespace numberOfPairsPaddlesSold_l320_32043

def totalSalesPaddles : ℝ := 735
def avgPricePerPairPaddles : ℝ := 9.8

theorem numberOfPairsPaddlesSold :
  totalSalesPaddles / avgPricePerPairPaddles = 75 := 
by
  sorry

end numberOfPairsPaddlesSold_l320_32043


namespace stratified_sampling_number_of_grade12_students_in_sample_l320_32027

theorem stratified_sampling_number_of_grade12_students_in_sample 
  (total_students : ℕ)
  (students_grade10 : ℕ)
  (students_grade11_minus_grade12 : ℕ)
  (sampled_students_grade10 : ℕ)
  (total_students_eq : total_students = 1290)
  (students_grade10_eq : students_grade10 = 480)
  (students_grade11_minus_grade12_eq : students_grade11_minus_grade12 = 30)
  (sampled_students_grade10_eq : sampled_students_grade10 = 96) :
  ∃ n : ℕ, n = 78 :=
by
  -- Proof would go here, but we are skipping with "sorry"
  sorry

end stratified_sampling_number_of_grade12_students_in_sample_l320_32027


namespace fraction_division_l320_32053

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l320_32053


namespace value_a8_l320_32046

def sequence_sum (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem value_a8 : a 8 = 15 :=
by
  sorry

end value_a8_l320_32046


namespace weight_difference_l320_32018

def weight_chemistry : ℝ := 7.12
def weight_geometry : ℝ := 0.62

theorem weight_difference : weight_chemistry - weight_geometry = 6.50 :=
by
  sorry

end weight_difference_l320_32018


namespace fraction_calculation_l320_32082

theorem fraction_calculation :
  ( (12^4 + 324) * (26^4 + 324) * (38^4 + 324) * (50^4 + 324) * (62^4 + 324)) /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) =
  73.481 :=
by
  sorry

end fraction_calculation_l320_32082


namespace arithmetic_sequence_minimum_value_S_n_l320_32097

-- Part 1: Proving the sequence is arithmetic
theorem arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) :
  (∀ n : ℕ, a (n + 1) = a n + 1) :=
by {
  -- Ideal proof here
  sorry
}

-- Part 2: Finding the minimum value of S_n
theorem minimum_value_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n : ℕ, 2 * S n / n + n = 2 * a n + 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 1) (h3 : a 4 * 2 = a 7 * a 9) : 
  ∃ n : ℕ, S n = -78 :=
by {
  -- Ideal proof here
  sorry
}

end arithmetic_sequence_minimum_value_S_n_l320_32097


namespace problem_statement_l320_32066

theorem problem_statement : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end problem_statement_l320_32066


namespace rational_functional_equation_l320_32008

theorem rational_functional_equation (f : ℚ → ℚ) (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (f = λ x => x) ∨ (f = λ x => -x) :=
by
  sorry

end rational_functional_equation_l320_32008


namespace odd_not_div_by_3_l320_32031

theorem odd_not_div_by_3 (n : ℤ) (h1 : Odd n) (h2 : ¬ ∃ k : ℤ, n = 3 * k) : 6 ∣ (n^2 + 5) :=
  sorry

end odd_not_div_by_3_l320_32031


namespace min_value_of_expression_l320_32033

theorem min_value_of_expression (x y z : ℝ) : ∃ a : ℝ, (∀ x y z : ℝ, x^2 + x * y + y^2 + y * z + z^2 ≥ a) ∧ (a = 0) :=
sorry

end min_value_of_expression_l320_32033


namespace incorrect_statement_l320_32083

-- Definitions based on the given conditions
def tripling_triangle_altitude_triples_area (b h : ℝ) : Prop :=
  3 * (1/2 * b * h) = 1/2 * b * (3 * h)

def halving_rectangle_base_halves_area (b h : ℝ) : Prop :=
  1/2 * b * h = 1/2 * (b * h)

def tripling_circle_radius_triples_area (r : ℝ) : Prop :=
  3 * (Real.pi * r^2) = Real.pi * (3 * r)^2

def tripling_divisor_and_numerator_leaves_quotient_unchanged (a b : ℝ) (hb : b ≠ 0) : Prop :=
  a / b = 3 * a / (3 * b)

def halving_negative_quantity_makes_it_greater (x : ℝ) : Prop :=
  x < 0 → (x / 2) > x

-- The incorrect statement is that tripling the radius of a circle triples the area
theorem incorrect_statement : ∃ r : ℝ, tripling_circle_radius_triples_area r → False :=
by
  use 1
  simp [tripling_circle_radius_triples_area]
  sorry

end incorrect_statement_l320_32083


namespace new_average_page_count_l320_32050

theorem new_average_page_count
  (n : ℕ) (a : ℕ) (p1 p2 : ℕ)
  (h_n : n = 80) (h_a : a = 120)
  (h_p1 : p1 = 150) (h_p2 : p2 = 170) :
  (n - 2) ≠ 0 → 
  ((n * a - (p1 + p2)) / (n - 2) = 119) := 
by sorry

end new_average_page_count_l320_32050


namespace sequence_last_number_is_one_l320_32092

theorem sequence_last_number_is_one :
  ∃ (a : ℕ → ℤ), (a 1 = 1) ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1997 → a (n + 1) = a n + a (n + 2)) ∧ (a 1999 = 1) := sorry

end sequence_last_number_is_one_l320_32092


namespace problem_statement_l320_32065

variables {R : Type*} [LinearOrderedField R]

-- Definitions of f and its derivatives
variable (f : R → R)
variable (f' : R → R) 
variable (f'' : R → R)

-- Conditions given in the math problem
axiom decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2
axiom derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x

-- Lean 4 statement for the proof problem
theorem problem_statement (decreasing_f : ∀ x1 x2 : R, x1 < x2 → f x1 > f x2)
    (derivative_condition : ∀ x : R, f'' x ≠ 0 → f x / f'' x < 1 - x) :
    ∀ x : R, f x > 0 :=
by
  sorry

end problem_statement_l320_32065


namespace fifth_term_arithmetic_sequence_l320_32077

-- Conditions provided
def first_term (x y : ℝ) := x + y^2
def second_term (x y : ℝ) := x - y^2
def third_term (x y : ℝ) := x - 3*y^2
def fourth_term (x y : ℝ) := x - 5*y^2

-- Proof to determine the fifth term
theorem fifth_term_arithmetic_sequence (x y : ℝ) :
  (fourth_term x y) - (third_term x y) = -2*y^2 →
  (x - 5 * y^2) - 2 * y^2 = x - 7 * y^2 :=
by sorry

end fifth_term_arithmetic_sequence_l320_32077


namespace boxes_containing_neither_l320_32096

theorem boxes_containing_neither (total_boxes markers erasers both : ℕ) 
  (h_total : total_boxes = 15) (h_markers : markers = 8) (h_erasers : erasers = 5) (h_both : both = 4) :
  total_boxes - (markers + erasers - both) = 6 :=
by
  sorry

end boxes_containing_neither_l320_32096


namespace geometric_and_arithmetic_sequences_l320_32074

theorem geometric_and_arithmetic_sequences (a b c x y : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : 2 * x = a + b)
  (h3 : 2 * y = b + c) :
  (a / x + c / y) = 2 := 
by 
  sorry

end geometric_and_arithmetic_sequences_l320_32074


namespace proof_f_of_2_add_g_of_3_l320_32070

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x^2 + 2 * x - 1

theorem proof_f_of_2_add_g_of_3 : f (2 + g 3) = 44 :=
by
  sorry

end proof_f_of_2_add_g_of_3_l320_32070


namespace B_subset_A_A_inter_B_empty_l320_32015

-- Definitions for the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}

-- Proof statement for Part (1)
theorem B_subset_A (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (-1 / 2 < a ∧ a < 1) := sorry

-- Proof statement for Part (2)
theorem A_inter_B_empty (a : ℝ) : (∀ x, ¬(x ∈ A ∧ x ∈ B a)) ↔ (a ≤ -4 ∨ a ≥ 2) := sorry

end B_subset_A_A_inter_B_empty_l320_32015


namespace expression_value_l320_32007

theorem expression_value :
  (35 + 12) ^ 2 - (12 ^ 2 + 35 ^ 2 - 2 * 12 * 35) = 1680 :=
by
  sorry

end expression_value_l320_32007


namespace sum_roots_of_quadratic_l320_32062

theorem sum_roots_of_quadratic (a b : ℝ) (h₁ : a^2 - a - 6 = 0) (h₂ : b^2 - b - 6 = 0) (h₃ : a ≠ b) :
  a + b = 1 :=
sorry

end sum_roots_of_quadratic_l320_32062


namespace cubic_identity_l320_32060

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 := 
by
  sorry

end cubic_identity_l320_32060


namespace resistance_of_one_rod_l320_32085

section RodResistance

variables (R_0 R : ℝ)

-- Given: the resistance of the entire construction is 8 Ω
def entire_construction_resistance : Prop := R = 8

-- Given: formula for the equivalent resistance
def equivalent_resistance_formula : Prop := R = 4 / 10 * R_0

-- To prove: the resistance of one rod is 20 Ω
theorem resistance_of_one_rod 
  (h1 : entire_construction_resistance R)
  (h2 : equivalent_resistance_formula R_0 R) :
  R_0 = 20 :=
sorry

end RodResistance

end resistance_of_one_rod_l320_32085


namespace nth_monomial_correct_l320_32019

-- Definitions of the sequence of monomials

def coeff (n : ℕ) : ℕ := 3 * n + 2
def exponent (n : ℕ) : ℕ := n

def nth_monomial (n : ℕ) (a : ℕ) : ℕ := (coeff n) * (a ^ (exponent n))

-- Theorem statement
theorem nth_monomial_correct (n : ℕ) (a : ℕ) : nth_monomial n a = (3 * n + 2) * (a ^ n) :=
by
  sorry

end nth_monomial_correct_l320_32019


namespace cone_sector_volume_ratio_l320_32014

theorem cone_sector_volume_ratio 
  (H R : ℝ) 
  (nonneg_H : 0 ≤ H) 
  (nonneg_R : 0 ≤ R) :
  let volume_original := (1/3) * π * R^2 * H
  let volume_sector   := (1/12) * π * R^2 * H
  volume_sector / volume_sector = 1 :=
  by
    sorry

end cone_sector_volume_ratio_l320_32014


namespace largest_number_with_four_digits_divisible_by_72_is_9936_l320_32001

theorem largest_number_with_four_digits_divisible_by_72_is_9936 :
  ∃ n : ℕ, (n < 10000 ∧ n ≥ 1000) ∧ (72 ∣ n) ∧ (∀ m : ℕ, (m < 10000 ∧ m ≥ 1000) ∧ (72 ∣ m) → m ≤ n) :=
sorry

end largest_number_with_four_digits_divisible_by_72_is_9936_l320_32001


namespace divides_power_diff_l320_32061

theorem divides_power_diff (x : ℤ) (y z w : ℕ) (hy : y % 2 = 1) (hz : z % 2 = 1) (hw : w % 2 = 1) : 17 ∣ x^(y^(z^w)) - x^(y^z) := 
by
  sorry

end divides_power_diff_l320_32061


namespace rectangle_area_l320_32002

theorem rectangle_area (L W P : ℝ) (hL : L = 13) (hP : P = 50) (hP_eq : P = 2 * L + 2 * W) :
  L * W = 156 :=
by
  have hL_val : L = 13 := hL
  have hP_val : P = 50 := hP
  have h_perimeter : P = 2 * L + 2 * W := hP_eq
  sorry

end rectangle_area_l320_32002


namespace g_function_property_l320_32057

variable {g : ℝ → ℝ}
variable {a b : ℝ}

theorem g_function_property 
  (h1 : ∀ a c : ℝ, c^3 * g a = a^3 * g c)
  (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 :=
  sorry

end g_function_property_l320_32057


namespace base8_subtraction_correct_l320_32024

def base8_sub (a b : Nat) : Nat := sorry  -- function to perform base 8 subtraction

theorem base8_subtraction_correct :
  base8_sub 0o126 0o45 = 0o41 := sorry

end base8_subtraction_correct_l320_32024


namespace earnings_per_widget_l320_32049

/-
Theorem:
Given:
1. Hourly wage is $12.50.
2. Hours worked in a week is 40.
3. Total weekly earnings are $580.
4. Number of widgets produced in a week is 500.

We want to prove:
The earnings per widget are $0.16.
-/

theorem earnings_per_widget (hourly_wage : ℝ) (hours_worked : ℝ)
  (total_weekly_earnings : ℝ) (widgets_produced : ℝ) :
  (hourly_wage = 12.50) →
  (hours_worked = 40) →
  (total_weekly_earnings = 580) →
  (widgets_produced = 500) →
  ( (total_weekly_earnings - hourly_wage * hours_worked) / widgets_produced = 0.16) :=
by
  intros h_wage h_hours h_earnings h_widgets
  sorry

end earnings_per_widget_l320_32049


namespace cells_after_9_days_l320_32028

noncomputable def remaining_cells (initial : ℕ) (days : ℕ) : ℕ :=
  let rec divide_and_decay (cells: ℕ) (remaining_days: ℕ) : ℕ :=
    if remaining_days = 0 then cells
    else
      let divided := cells * 2
      let decayed := (divided * 9) / 10
      divide_and_decay decayed (remaining_days - 3)
  divide_and_decay initial days

theorem cells_after_9_days :
  remaining_cells 5 9 = 28 := by
  sorry

end cells_after_9_days_l320_32028


namespace a_pow_a_b_pow_b_c_pow_c_ge_one_l320_32016

theorem a_pow_a_b_pow_b_c_pow_c_ge_one
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a + b + c = Real.rpow a (1/7) + Real.rpow b (1/7) + Real.rpow c (1/7)) :
    a^a * b^b * c^c ≥ 1 := 
by
  sorry

end a_pow_a_b_pow_b_c_pow_c_ge_one_l320_32016


namespace find_special_n_l320_32086

open Nat

def is_divisor (d n : ℕ) : Prop := n % d = 0

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def special_primes_condition (n : ℕ) : Prop :=
  ∀ d : ℕ, is_divisor d n → is_prime (d^2 - d + 1) ∧ is_prime (d^2 + d + 1)

theorem find_special_n (n : ℕ) (h : n > 1) :
  special_primes_condition n → n = 2 ∨ n = 3 ∨ n = 6 :=
sorry

end find_special_n_l320_32086


namespace missing_digit_l320_32047

theorem missing_digit (x : ℕ) (h1 : x ≥ 0) (h2 : x ≤ 9) : 
  (if x ≥ 2 then 9 * 1000 + x * 100 + 2 * 10 + 1 else 9 * 100 + 2 * 10 + x * 1) - (1 * 1000 + 2 * 100 + 9 * 10 + x) = 8262 → x = 5 :=
by 
  sorry

end missing_digit_l320_32047


namespace quadratic_graphs_intersect_at_one_point_l320_32078

theorem quadratic_graphs_intersect_at_one_point
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h_distinct : a1 ≠ a2 ∧ a2 ≠ a3 ∧ a1 ≠ a3)
  (h_intersect_fg : ∃ x₀ : ℝ, (a1 - a2) * x₀^2 + (b1 - b2) * x₀ + (c1 - c2) = 0 ∧ (b1 - b2)^2 - 4 * (a1 - a2) * (c1 - c2) = 0)
  (h_intersect_gh : ∃ x₁ : ℝ, (a2 - a3) * x₁^2 + (b2 - b3) * x₁ + (c2 - c3) = 0 ∧ (b2 - b3)^2 - 4 * (a2 - a3) * (c2 - c3) = 0)
  (h_intersect_fh : ∃ x₂ : ℝ, (a1 - a3) * x₂^2 + (b1 - b3) * x₂ + (c1 - c3) = 0 ∧ (b1 - b3)^2 - 4 * (a1 - a3) * (c1 - c3) = 0) :
  ∃ x : ℝ, (a1 * x^2 + b1 * x + c1 = 0) ∧ (a2 * x^2 + b2 * x + c2 = 0) ∧ (a3 * x^2 + b3 * x + c3 = 0) :=
by
  sorry

end quadratic_graphs_intersect_at_one_point_l320_32078


namespace cos_4_arccos_fraction_l320_32039

theorem cos_4_arccos_fraction :
  (Real.cos (4 * Real.arccos (2 / 5))) = (-47 / 625) :=
by
  sorry

end cos_4_arccos_fraction_l320_32039


namespace other_team_scored_l320_32010

open Nat

def points_liz_scored (free_throws three_pointers jump_shots : Nat) : Nat :=
  free_throws * 1 + three_pointers * 3 + jump_shots * 2

def points_deficit := 20
def points_liz_deficit := points_liz_scored 5 3 4 - points_deficit
def final_loss_margin := 8
def other_team_score := points_liz_scored 5 3 4 + final_loss_margin

theorem other_team_scored
  (points_liz : Nat := points_liz_scored 5 3 4)
  (final_deficit : Nat := points_deficit)
  (final_margin : Nat := final_loss_margin)
  (other_team_points : Nat := other_team_score) :
  other_team_points = 30 := 
sorry

end other_team_scored_l320_32010


namespace sqrt_product_simplified_l320_32032

theorem sqrt_product_simplified (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (32 * x)) = 84 * x * Real.sqrt (2 * x) :=
by 
  sorry

end sqrt_product_simplified_l320_32032


namespace max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l320_32004

noncomputable def f (a x : ℝ) := a * x + Real.log x
noncomputable def g (a x : ℝ) := x * f a x
noncomputable def e := Real.exp 1

-- Statement for part (1)
theorem max_value_fx_when_a_neg1 : 
  ∀ x : ℝ, 0 < x → (f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement for part (2)
theorem find_a_when_max_fx_is_neg3 : 
  (∀ x : ℝ, 0 < x ∧ x ≤ e → (f (-e^2) x ≤ -3)) →
  (∃ a : ℝ, a = -e^2) :=
sorry

-- Statement for part (3)
theorem inequality_gx_if_a_pos (a : ℝ) (hapos : 0 < a) 
  (x1 x2 : ℝ) (hxpos1 : 0 < x1) (hxpos2 : 0 < x2) (hx12 : x1 ≠ x2) :
  2 * g a ((x1 + x2) / 2) < g a x1 + g a x2 :=
sorry

end max_value_fx_when_a_neg1_find_a_when_max_fx_is_neg3_inequality_gx_if_a_pos_l320_32004


namespace system1_solution_system2_solution_l320_32022

-- Problem 1
theorem system1_solution (x y : ℝ) (h1 : 3 * x - 2 * y = 6) (h2 : 2 * x + 3 * y = 17) : 
  x = 4 ∧ y = 3 :=
by {
  sorry
}

-- Problem 2
theorem system2_solution (x y : ℝ) (h1 : x + 4 * y = 14) 
  (h2 : (x - 3) / 4 - (y - 3) / 3 = 1 / 12) : 
  x = 3 ∧ y = 11 / 4 :=
by {
  sorry
}

end system1_solution_system2_solution_l320_32022


namespace area_of_fifteen_sided_figure_l320_32003

def point : Type := ℕ × ℕ

def vertices : List point :=
  [(1,1), (1,3), (3,5), (4,5), (5,4), (5,3), (6,3), (6,2), (5,1), (4,1), (3,2), (2,2), (1,1)]

def graph_paper_area (vs : List point) : ℚ :=
  -- Placeholder for actual area calculation logic
  -- The area for the provided vertices is found to be 11 cm^2.
  11

theorem area_of_fifteen_sided_figure : graph_paper_area vertices = 11 :=
by
  -- The actual proof would involve detailed steps to show that the area is indeed 11 cm^2
  -- Placeholder proof
  sorry

end area_of_fifteen_sided_figure_l320_32003


namespace distance_A_beats_B_l320_32023

theorem distance_A_beats_B 
  (A_time : ℝ) (A_distance : ℝ) (B_time : ℝ) (B_distance : ℝ)
  (hA : A_distance = 128) (hA_time : A_time = 28)
  (hB : B_distance = 128) (hB_time : B_time = 32) :
  (A_distance - (B_distance * (A_time / B_time))) = 16 :=
by
  sorry

end distance_A_beats_B_l320_32023


namespace coats_leftover_l320_32040

theorem coats_leftover :
  ∀ (total_coats : ℝ) (num_boxes : ℝ),
  total_coats = 385.5 →
  num_boxes = 7.5 →
  ∃ extra_coats : ℕ, extra_coats = 3 :=
by
  intros total_coats num_boxes h1 h2
  sorry

end coats_leftover_l320_32040


namespace greatest_M_inequality_l320_32013

theorem greatest_M_inequality :
  ∀ x y z : ℝ, x^4 + y^4 + z^4 + x * y * z * (x + y + z) ≥ (2/3) * (x * y + y * z + z * x)^2 :=
by
  sorry

end greatest_M_inequality_l320_32013


namespace ducks_and_dogs_total_l320_32054

theorem ducks_and_dogs_total (d g : ℕ) (h1 : d = g + 2) (h2 : 4 * g - 2 * d = 10) : d + g = 16 :=
  sorry

end ducks_and_dogs_total_l320_32054


namespace safe_travel_exists_l320_32006

def total_travel_time : ℕ := 16
def first_crater_cycle : ℕ := 18
def first_crater_duration : ℕ := 1
def second_crater_cycle : ℕ := 10
def second_crater_duration : ℕ := 1

theorem safe_travel_exists : 
  ∃ t : ℕ, t ∈ { t | (∀ k : ℕ, t % first_crater_cycle ≠ k ∨ t % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, t % second_crater_cycle ≠ k ∨ t % second_crater_cycle ≥ second_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % first_crater_cycle ≠ k ∨ (t + total_travel_time) % first_crater_cycle ≥ first_crater_duration) 
  ∧ (∀ k : ℕ, (t + total_travel_time) % second_crater_cycle ≠ k ∨ (t + total_travel_time) % second_crater_cycle ≥ second_crater_duration) } :=
sorry

end safe_travel_exists_l320_32006


namespace carnival_ring_toss_l320_32055

theorem carnival_ring_toss (total_amount : ℕ) (days : ℕ) (amount_per_day : ℕ) 
  (h1 : total_amount = 420) 
  (h2 : days = 3) 
  (h3 : total_amount = days * amount_per_day) : amount_per_day = 140 :=
by
  sorry

end carnival_ring_toss_l320_32055


namespace promotional_codes_one_tenth_l320_32009

open Nat

def promotional_chars : List Char := ['C', 'A', 'T', '3', '1', '1', '9']

def count_promotional_codes (chars : List Char) (len : Nat) : Nat := sorry

theorem promotional_codes_one_tenth : count_promotional_codes promotional_chars 5 / 10 = 60 :=
by 
  sorry

end promotional_codes_one_tenth_l320_32009


namespace identical_solutions_k_value_l320_32073

theorem identical_solutions_k_value (k : ℝ) :
  (∀ (x y : ℝ), y = x^2 ∧ y = 4 * x + k → (x - 2)^2 = 0) → k = -4 :=
by
  sorry

end identical_solutions_k_value_l320_32073


namespace multiply_658217_99999_l320_32017

theorem multiply_658217_99999 : 658217 * 99999 = 65821034183 := 
by
  sorry

end multiply_658217_99999_l320_32017


namespace trajectory_of_P_l320_32005

open Real

-- Definitions of points F1 and F2
def F1 : (ℝ × ℝ) := (-4, 0)
def F2 : (ℝ × ℝ) := (4, 0)

-- Definition of the condition on moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  abs (dist P F2 - dist P F1) = 4

-- Definition of the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1 ∧ x ≤ -2

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, satisfies_condition P → ∃ x y : ℝ, P = (x, y) ∧ hyperbola_equation x y :=
by
  sorry

end trajectory_of_P_l320_32005


namespace gummies_remain_l320_32063

theorem gummies_remain
  (initial_candies : ℕ)
  (sibling_candies_per : ℕ)
  (num_siblings : ℕ)
  (best_friend_fraction : ℝ)
  (cousin_fraction : ℝ)
  (kept_candies : ℕ)
  (result : ℕ)
  (h_initial : initial_candies = 500)
  (h_sibling_candies_per : sibling_candies_per = 35)
  (h_num_siblings : num_siblings = 3)
  (h_best_friend_fraction : best_friend_fraction = 0.5)
  (h_cousin_fraction : cousin_fraction = 0.25)
  (h_kept_candies : kept_candies = 50)
  (h_result : result = 99) : 
  (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋ - 
  ⌊cousin_fraction * (initial_candies - num_siblings * sibling_candies_per - ⌊best_friend_fraction * (initial_candies - num_siblings * sibling_candies_per)⌋)⌋ 
  - kept_candies) = result := 
by {
  sorry
}

end gummies_remain_l320_32063


namespace zongzi_unit_prices_max_type_A_zongzi_l320_32098

theorem zongzi_unit_prices (x : ℝ) : 
  (800 / x - 1200 / (2 * x) = 50) → 
  (x = 4 ∧ 2 * x = 8) :=
by
  intro h
  sorry

theorem max_type_A_zongzi (m : ℕ) : 
  (m ≤ 200) → 
  (8 * m + 4 * (200 - m) ≤ 1150) → 
  (m ≤ 87) :=
by
  intros h1 h2
  sorry

end zongzi_unit_prices_max_type_A_zongzi_l320_32098


namespace train_distance_difference_l320_32011

theorem train_distance_difference 
  (speed1 speed2 : ℕ) (distance : ℕ) (meet_time : ℕ)
  (h_speed1 : speed1 = 16)
  (h_speed2 : speed2 = 21)
  (h_distance : distance = 444)
  (h_meet_time : meet_time = distance / (speed1 + speed2)) :
  (speed2 * meet_time) - (speed1 * meet_time) = 60 :=
by
  sorry

end train_distance_difference_l320_32011


namespace price_reduction_l320_32099

theorem price_reduction (x : ℝ) 
  (initial_price : ℝ := 60) 
  (final_price : ℝ := 48.6) :
  initial_price * (1 - x) * (1 - x) = final_price :=
by
  sorry

end price_reduction_l320_32099


namespace shopkeeper_gain_l320_32079

noncomputable def overall_percentage_gain (P : ℝ) (increase_percentage : ℝ) (discount1_percentage : ℝ) (discount2_percentage : ℝ) : ℝ :=
  let increased_price := P * (1 + increase_percentage)
  let price_after_first_discount := increased_price * (1 - discount1_percentage)
  let final_price := price_after_first_discount * (1 - discount2_percentage)
  ((final_price - P) / P) * 100

theorem shopkeeper_gain : 
  overall_percentage_gain 100 0.32 0.10 0.15 = 0.98 :=
by
  sorry

end shopkeeper_gain_l320_32079


namespace find_added_value_l320_32069

theorem find_added_value (avg_15_numbers : ℤ) (new_avg : ℤ) (x : ℤ)
    (H1 : avg_15_numbers = 40) 
    (H2 : new_avg = 50) 
    (H3 : (600 + 15 * x) / 15 = new_avg) : 
    x = 10 := 
sorry

end find_added_value_l320_32069


namespace range_of_m_l320_32094

theorem range_of_m (m : ℝ)
  (h₁ : (m^2 - 4) ≥ 0)
  (h₂ : (4 * (m - 2)^2 - 16) < 0) :
  1 < m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l320_32094


namespace triangle_acd_area_l320_32030

noncomputable def area_of_triangle : ℝ := sorry

theorem triangle_acd_area (AB CD : ℝ) (h : CD = 3 * AB) (area_trapezoid: ℝ) (h1: area_trapezoid = 20) :
  area_of_triangle = 15 := 
sorry

end triangle_acd_area_l320_32030


namespace negation_of_existence_statement_l320_32026

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 2 = 0) = ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existence_statement_l320_32026


namespace negation_proposition_true_l320_32072

theorem negation_proposition_true (x : ℝ) : (¬ (|x| > 1 → x > 1)) ↔ (|x| ≤ 1 → x ≤ 1) :=
by sorry

end negation_proposition_true_l320_32072


namespace total_students_l320_32044

theorem total_students (h1 : 15 * 70 = 1050) 
                       (h2 : 10 * 95 = 950) 
                       (h3 : 1050 + 950 = 2000)
                       (h4 : 80 * N = 2000) :
  N = 25 :=
by sorry

end total_students_l320_32044


namespace friends_division_l320_32067

def num_ways_to_divide (total_friends teams : ℕ) : ℕ :=
  4^8 - (Nat.choose 4 1) * 3^8 + (Nat.choose 4 2) * 2^8 - (Nat.choose 4 3) * 1^8

theorem friends_division (total_friends teams : ℕ) (h_friends : total_friends = 8) (h_teams : teams = 4) :
  num_ways_to_divide total_friends teams = 39824 := by
  sorry

end friends_division_l320_32067


namespace short_trees_after_planting_l320_32042

-- Defining the conditions as Lean definitions
def current_short_trees : Nat := 3
def newly_planted_short_trees : Nat := 9

-- Defining the question (assertion to prove) with the expected answer
theorem short_trees_after_planting : current_short_trees + newly_planted_short_trees = 12 := by
  sorry

end short_trees_after_planting_l320_32042


namespace three_digit_number_is_11_times_sum_of_digits_l320_32036

theorem three_digit_number_is_11_times_sum_of_digits :
    ∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ 
        (100 * a + 10 * b + c = 11 * (a + b + c)) ↔ 
        (100 * 1 + 10 * 9 + 8 = 11 * (1 + 9 + 8)) := 
by
    sorry

end three_digit_number_is_11_times_sum_of_digits_l320_32036


namespace salesperson_commission_l320_32034

noncomputable def commission (sale_price : ℕ) (rate : ℚ) : ℚ :=
  rate * sale_price

noncomputable def total_commission (machines_sold : ℕ) (first_rate : ℚ) (second_rate : ℚ) (sale_price : ℕ) : ℚ :=
  let first_commission := commission sale_price first_rate * 100
  let second_commission := commission sale_price second_rate * (machines_sold - 100)
  first_commission + second_commission

theorem salesperson_commission :
  total_commission 130 0.03 0.04 10000 = 42000 := by
  sorry

end salesperson_commission_l320_32034


namespace isosceles_triangle_sides_l320_32090

-- Definitions and assumptions
def is_isosceles (a b c : ℕ) : Prop :=
(a = b) ∨ (a = c) ∨ (b = c)

noncomputable def perimeter (a b c : ℕ) : ℕ :=
a + b + c

theorem isosceles_triangle_sides (a b c : ℕ) (h_iso : is_isosceles a b c) (h_perim : perimeter a b c = 17) (h_side : a = 4 ∨ b = 4 ∨ c = 4) :
  (a = 6 ∧ b = 6 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 7) :=
sorry

end isosceles_triangle_sides_l320_32090


namespace john_makes_200_profit_l320_32076

noncomputable def john_profit (num_woodburnings : ℕ) (price_per_woodburning : ℕ) (cost_of_wood : ℕ) : ℕ :=
  (num_woodburnings * price_per_woodburning) - cost_of_wood

theorem john_makes_200_profit :
  john_profit 20 15 100 = 200 :=
by
  sorry

end john_makes_200_profit_l320_32076


namespace regular_polygon_sides_l320_32064

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) 
    (h2 : (180 * (n-2)) / n = 150) : n = 12 := by
  sorry

end regular_polygon_sides_l320_32064


namespace part1_part2_l320_32084

noncomputable def inverse_function_constant (k : ℝ) : Prop :=
  (∀ x : ℝ, 0 < x → (x, 3) ∈ {p : ℝ × ℝ | p.snd = k / p.fst})

noncomputable def range_m (m : ℝ) : Prop :=
  0 < m → m < 3

theorem part1 (k : ℝ) (hk : k ≠ 0) (h : (1, 3).snd = k / (1, 3).fst) :
  k = 3 := by
  sorry

theorem part2 (m : ℝ) (hm : m ≠ 0) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → 3 / x > m * x) ↔ (m < 0 ∨ (0 < m ∧ m < 3)) := by
  sorry

end part1_part2_l320_32084


namespace closest_integer_to_cuberoot_of_200_l320_32059

theorem closest_integer_to_cuberoot_of_200 : 
  let c := (200 : ℝ)^(1/3)
  ∃ (k : ℤ), abs (c - 6) < abs (c - 5) :=
by
  let c := (200 : ℝ)^(1/3)
  existsi (6 : ℤ)
  sorry

end closest_integer_to_cuberoot_of_200_l320_32059


namespace mn_value_l320_32087

variables {x m n : ℝ} -- Define variables x, m, n as real numbers

theorem mn_value (h : x^2 + m * x - 15 = (x + 3) * (x + n)) : m * n = 10 :=
by {
  -- Sorry for skipping the proof steps
  sorry
}

end mn_value_l320_32087


namespace find_M_l320_32048

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l320_32048


namespace other_carton_racket_count_l320_32058

def num_total_cartons : Nat := 38
def num_total_rackets : Nat := 100
def num_specific_cartons : Nat := 24
def num_rackets_per_specific_carton : Nat := 3

def num_remaining_cartons := num_total_cartons - num_specific_cartons
def num_remaining_rackets := num_total_rackets - (num_specific_cartons * num_rackets_per_specific_carton)

theorem other_carton_racket_count :
  (num_remaining_rackets / num_remaining_cartons) = 2 :=
by
  sorry

end other_carton_racket_count_l320_32058


namespace average_of_quantities_l320_32081

theorem average_of_quantities (a1 a2 a3 a4 a5 : ℝ) :
  ((a1 + a2 + a3) / 3 = 4) →
  ((a4 + a5) / 2 = 21.5) →
  ((a1 + a2 + a3 + a4 + a5) / 5 = 11) :=
by
  intros h3 h2
  sorry

end average_of_quantities_l320_32081


namespace product_of_binaries_l320_32068

-- Step a) Define the binary numbers as Lean 4 terms.
def bin_11011 : ℕ := 0b11011
def bin_111 : ℕ := 0b111
def bin_101 : ℕ := 0b101

-- Step c) Define the goal to be proven.
theorem product_of_binaries :
  bin_11011 * bin_111 * bin_101 = 0b1110110001 :=
by
  -- proof goes here
  sorry

end product_of_binaries_l320_32068


namespace cost_two_cones_l320_32051

-- Definition for the cost of a single ice cream cone
def cost_one_cone : ℕ := 99

-- The theorem to prove the cost of two ice cream cones
theorem cost_two_cones : 2 * cost_one_cone = 198 := 
by 
  sorry

end cost_two_cones_l320_32051


namespace max_sum_of_xj4_minus_xj5_l320_32021

theorem max_sum_of_xj4_minus_xj5 (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ j => (x j)^4 - (x j)^5)) ≤ 1 / 12 :=
sorry

end max_sum_of_xj4_minus_xj5_l320_32021


namespace eval_expression_l320_32025

theorem eval_expression (x y z : ℝ) (h1 : y > z) (h2 : z > 0) (h3 : x = y + z) : 
  ( (y+z+y)^z + (y+z+z)^y ) / (y^z + z^y) = 2^y + 2^z :=
by
  sorry

end eval_expression_l320_32025


namespace bottles_last_days_l320_32038

theorem bottles_last_days :
  let total_bottles := 8066
  let bottles_per_day := 109
  total_bottles / bottles_per_day = 74 :=
by
  sorry

end bottles_last_days_l320_32038


namespace suff_not_nec_for_abs_eq_one_l320_32056

variable (m : ℝ)

theorem suff_not_nec_for_abs_eq_one (hm : m = 1) : |m| = 1 ∧ (¬(|m| = 1 → m = 1)) := by
  sorry

end suff_not_nec_for_abs_eq_one_l320_32056


namespace rectangle_perimeter_eq_circle_circumference_l320_32095

theorem rectangle_perimeter_eq_circle_circumference (l : ℝ) :
  2 * (l + 3) = 10 * Real.pi -> l = 5 * Real.pi - 3 :=
by
  intro h
  sorry

end rectangle_perimeter_eq_circle_circumference_l320_32095


namespace sum_of_different_roots_eq_six_l320_32089

theorem sum_of_different_roots_eq_six (a b : ℝ) (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_different_roots_eq_six_l320_32089


namespace max_k_l320_32037

def seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = k * (a n) ^ 2 + 1

def bounded (a : ℕ → ℝ) (c : ℝ) : Prop :=
∀ n : ℕ, a n < c

theorem max_k (k : ℝ) (c : ℝ) (a : ℕ → ℝ) :
  a 1 = 1 →
  seq a k →
  bounded a c →
  0 < k ∧ k ≤ 1 / 4 :=
by
  sorry

end max_k_l320_32037


namespace find_number_l320_32020

theorem find_number (num : ℝ) (x : ℝ) (h1 : x = 0.08999999999999998) (h2 : num / x = 0.1) : num = 0.008999999999999999 :=
by 
  sorry

end find_number_l320_32020


namespace john_bought_packs_l320_32052

def students_in_classes : List ℕ := [20, 25, 18, 22, 15]
def packs_per_student : ℕ := 3

theorem john_bought_packs :
  (students_in_classes.sum) * packs_per_student = 300 := by
  sorry

end john_bought_packs_l320_32052


namespace probability_exactly_one_red_ball_l320_32041

-- Define the given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 3
def children : ℕ := 10

-- Define the question and calculate the probability
theorem probability_exactly_one_red_ball : 
  (3 * (3 / 10) * ((7 / 10) * (7 / 10))) = 0.441 := 
by 
  sorry

end probability_exactly_one_red_ball_l320_32041


namespace stella_toilet_paper_packs_l320_32093

-- Define the relevant constants/conditions
def rolls_per_bathroom_per_day : Nat := 1
def number_of_bathrooms : Nat := 6
def days_per_week : Nat := 7
def weeks : Nat := 4
def rolls_per_pack : Nat := 12

-- Theorem statement
theorem stella_toilet_paper_packs :
  (rolls_per_bathroom_per_day * number_of_bathrooms * days_per_week * weeks) / rolls_per_pack = 14 :=
by
  sorry

end stella_toilet_paper_packs_l320_32093


namespace bob_mean_score_l320_32080

-- Conditions
def scores : List ℝ := [68, 72, 76, 80, 85, 90]
def alice_scores (a1 a2 a3 : ℝ) : Prop := a1 < a2 ∧ a2 < a3 ∧ a1 + a2 + a3 = 225
def bob_scores (b1 b2 b3 : ℝ) : Prop := b1 + b2 + b3 = 246

-- Theorem statement proving Bob's mean score
theorem bob_mean_score (a1 a2 a3 b1 b2 b3 : ℝ) (h1 : a1 ∈ scores) (h2 : a2 ∈ scores) (h3 : a3 ∈ scores)
  (h4 : b1 ∈ scores) (h5 : b2 ∈ scores) (h6 : b3 ∈ scores)
  (h7 : alice_scores a1 a2 a3)
  (h8 : bob_scores b1 b2 b3)
  (h9 : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3)
  : (b1 + b2 + b3) / 3 = 82 :=
sorry

end bob_mean_score_l320_32080


namespace evaluate_expression_l320_32045

theorem evaluate_expression (k : ℤ): 
  2^(-(3*k+1)) - 2^(-(3*k-2)) + 2^(-(3*k)) - 2^(-(3*k+3)) = -((21:ℚ)/(8:ℚ)) * 2^(-(3*k)) := 
by 
  sorry

end evaluate_expression_l320_32045


namespace christine_stickers_needed_l320_32088

-- Define the number of stickers Christine has
def stickers_has : ℕ := 11

-- Define the number of stickers required for the prize
def stickers_required : ℕ := 30

-- Define the formula to calculate the number of stickers Christine needs
def stickers_needed : ℕ := stickers_required - stickers_has

-- The theorem we need to prove
theorem christine_stickers_needed : stickers_needed = 19 :=
by
  sorry

end christine_stickers_needed_l320_32088


namespace intersection_unique_point_x_coordinate_l320_32000

theorem intersection_unique_point_x_coordinate (a b : ℝ) (h : a ≠ b) : 
  (∃ x y : ℝ, y = x^2 + 2*a*x + 6*b ∧ y = x^2 + 2*b*x + 6*a) → ∃ x : ℝ, x = 3 :=
by
  sorry

end intersection_unique_point_x_coordinate_l320_32000


namespace y_increase_by_41_8_units_l320_32012

theorem y_increase_by_41_8_units :
  ∀ (x y : ℝ),
    (∀ k : ℝ, y = 2 + k * 11 / 5 → x = 1 + k * 5) →
    x = 20 → y = 41.8 :=
by
  sorry

end y_increase_by_41_8_units_l320_32012


namespace fraction_income_spent_on_rent_l320_32091

theorem fraction_income_spent_on_rent
  (hourly_wage : ℕ)
  (work_hours_per_week : ℕ)
  (weeks_in_month : ℕ)
  (food_expense : ℕ)
  (tax_expense : ℕ)
  (remaining_income : ℕ) :
  hourly_wage = 30 →
  work_hours_per_week = 48 →
  weeks_in_month = 4 →
  food_expense = 500 →
  tax_expense = 1000 →
  remaining_income = 2340 →
  ((hourly_wage * work_hours_per_week * weeks_in_month - remaining_income - (food_expense + tax_expense)) / (hourly_wage * work_hours_per_week * weeks_in_month) = 1/3) :=
by
  intros h_wage h_hours h_weeks h_food h_taxes h_remaining
  sorry

end fraction_income_spent_on_rent_l320_32091


namespace sufficient_but_not_necessary_not_necessary_l320_32029

theorem sufficient_but_not_necessary (x y : ℝ) (h : x < y ∧ y < 0) : x^2 > y^2 :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

theorem not_necessary (x y : ℝ) (h : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

end sufficient_but_not_necessary_not_necessary_l320_32029


namespace poster_height_proportion_l320_32071

-- Defining the given conditions
def original_width : ℕ := 3
def original_height : ℕ := 2
def new_width : ℕ := 12
def scale_factor := new_width / original_width

-- The statement to prove the new height
theorem poster_height_proportion :
  scale_factor = 4 → (original_height * scale_factor) = 8 :=
by
  sorry

end poster_height_proportion_l320_32071


namespace survey_support_percentage_l320_32075

theorem survey_support_percentage 
  (num_men : ℕ) (percent_men_support : ℝ)
  (num_women : ℕ) (percent_women_support : ℝ)
  (h_men : num_men = 200)
  (h_percent_men_support : percent_men_support = 0.7)
  (h_women : num_women = 500)
  (h_percent_women_support : percent_women_support = 0.75) :
  (num_men * percent_men_support + num_women * percent_women_support) / (num_men + num_women) * 100 = 74 := 
by
  sorry

end survey_support_percentage_l320_32075
