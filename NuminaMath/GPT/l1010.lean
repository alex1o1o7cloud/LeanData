import Mathlib

namespace min_value_of_polynomial_l1010_101072

theorem min_value_of_polynomial (a : ℝ) : 
  (∀ x : ℝ, (2 * x^3 - 3 * x^2 + a) ≥ 5) → a = 6 :=
by
  sorry   -- Proof omitted

end min_value_of_polynomial_l1010_101072


namespace expenditure_recording_l1010_101082

def income : ℕ := 200
def recorded_income : ℤ := 200
def expenditure (e : ℕ) : ℤ := -(e : ℤ)

theorem expenditure_recording (e : ℕ) :
  expenditure 150 = -150 := by
  sorry

end expenditure_recording_l1010_101082


namespace problem1_problem2_l1010_101030

-- Problem 1: If a is parallel to b, then x = 4
theorem problem1 (x : ℝ) (u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  (a.1 / b.1 = a.2 / b.2) → x = 4 := 
by 
  intros a b h
  dsimp [a, b] at h
  sorry

-- Problem 2: If (u - 2 * v) is perpendicular to (u + v), then x = -6
theorem problem2 (x : ℝ) (a u v : ℝ × ℝ) : 
  let a := (1, 1)
  let b := (4, x)
  let u := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let v := (2 * a.1 + b.1, 2 * a.2 + b.2)
  ((u.1 - 2 * v.1) * (u.1 + v.1) + (u.2 - 2 * v.2) * (u.2 + v.2) = 0) → x = -6 := 
by 
  intros a b u v h
  dsimp [a, b, u, v] at h
  sorry

end problem1_problem2_l1010_101030


namespace total_production_l1010_101041

theorem total_production (S : ℝ) 
  (h1 : 4 * S = 4400) : 
  4400 + S = 5500 := 
by
  sorry

end total_production_l1010_101041


namespace books_remaining_after_second_day_l1010_101010

theorem books_remaining_after_second_day :
  let initial_books := 100
  let first_day_borrowed := 5 * 2
  let second_day_borrowed := 20
  let total_borrowed := first_day_borrowed + second_day_borrowed
  let remaining_books := initial_books - total_borrowed
  remaining_books = 70 :=
by
  sorry

end books_remaining_after_second_day_l1010_101010


namespace count_integers_log_condition_l1010_101086

theorem count_integers_log_condition :
  (∃! n : ℕ, n = 54 ∧ (∀ x : ℕ, x > 30 ∧ x < 90 ∧ ((x - 30) * (90 - x) < 1000) ↔ (31 <= x ∧ x <= 84))) :=
sorry

end count_integers_log_condition_l1010_101086


namespace right_triangle_of_medians_l1010_101096

theorem right_triangle_of_medians
  (a b c m1 m2 m3 : ℝ)
  (h1 : 4 * m1^2 = 2 * (b^2 + c^2) - a^2)
  (h2 : 4 * m2^2 = 2 * (a^2 + c^2) - b^2)
  (h3 : 4 * m3^2 = 2 * (a^2 + b^2) - c^2)
  (h4 : m1^2 + m2^2 = 5 * m3^2) :
  c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_of_medians_l1010_101096


namespace inequality_proof_l1010_101052

theorem inequality_proof (a b : ℝ) (h_a : a > 0) (h_b : 3 + b = a) : 
  3 / b + 1 / a >= 3 :=
sorry

end inequality_proof_l1010_101052


namespace abs_val_problem_l1010_101088

variable (a b : ℝ)

theorem abs_val_problem (h_abs_a : |a| = 2) (h_abs_b : |b| = 4) (h_sum_neg : a + b < 0) : a - b = 2 ∨ a - b = 6 :=
sorry

end abs_val_problem_l1010_101088


namespace decimal_to_base5_equiv_l1010_101091

def base5_representation (n : ℕ) : ℕ := -- Conversion function (implementation to be filled later)
  sorry

theorem decimal_to_base5_equiv : base5_representation 88 = 323 :=
by
  -- Proof steps go here.
  sorry

end decimal_to_base5_equiv_l1010_101091


namespace inequality_proof_l1010_101099

theorem inequality_proof {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1) :
    (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 := sorry

end inequality_proof_l1010_101099


namespace complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l1010_101012

theorem complete_even_square_diff_eqn : (10^2 - 8^2 = 4 * 9) :=
by sorry

theorem even_square_diff_multiple_of_four (n : ℕ) : (4 * (n + 1) * (n + 1) - 4 * n * n) % 4 = 0 :=
by sorry

theorem odd_square_diff_multiple_of_eight (m : ℕ) : ((2 * m + 1)^2 - (2 * m - 1)^2) % 8 = 0 :=
by sorry

end complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l1010_101012


namespace area_of_rhombus_l1010_101044

theorem area_of_rhombus (x y : ℝ) (d1 d2 : ℝ) (hx : x^2 + y^2 = 130) (hy : d1 = 2 * x) (hz : d2 = 2 * y) (h_diff : abs (d1 - d2) = 4) : 
  4 * 0.5 * x * y = 126 :=
by
  sorry

end area_of_rhombus_l1010_101044


namespace intersecting_lines_l1010_101027

theorem intersecting_lines (n c : ℝ) 
  (h1 : (15 : ℝ) = n * 5 + 5)
  (h2 : (15 : ℝ) = 4 * 5 + c) : 
  c + n = -3 := 
by
  sorry

end intersecting_lines_l1010_101027


namespace jade_cal_difference_l1010_101015

def Mabel_transactions : ℕ := 90

def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)

def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3

def Jade_transactions : ℕ := 85

theorem jade_cal_difference : Jade_transactions - Cal_transactions = 19 := by
  sorry

end jade_cal_difference_l1010_101015


namespace combined_perimeter_l1010_101032

theorem combined_perimeter (side_square : ℝ) (a b c : ℝ) (diameter : ℝ) 
  (h_square : side_square = 7) 
  (h_triangle : a = 5 ∧ b = 6 ∧ c = 7) 
  (h_diameter : diameter = 4) : 
  4 * side_square + (a + b + c) + (2 * Real.pi * (diameter / 2) + diameter) = 50 + 2 * Real.pi := 
by 
  sorry

end combined_perimeter_l1010_101032


namespace horner_value_at_3_l1010_101008

noncomputable def horner (x : ℝ) : ℝ :=
  ((((0.5 * x + 4) * x + 0) * x - 3) * x + 1) * x - 1

theorem horner_value_at_3 : horner 3 = 5.5 :=
by
  sorry

end horner_value_at_3_l1010_101008


namespace number_of_white_tiles_l1010_101014

-- Definition of conditions in the problem
def side_length_large_square := 81
def area_large_square := side_length_large_square * side_length_large_square
def area_black_tiles := 81
def num_red_tiles := 154
def area_red_tiles := num_red_tiles * 4
def area_covered_by_black_and_red := area_black_tiles + area_red_tiles
def remaining_area_for_white_tiles := area_large_square - area_covered_by_black_and_red
def area_of_one_white_tile := 2
def expected_num_white_tiles := 2932

-- The theorem to prove
theorem number_of_white_tiles :
  remaining_area_for_white_tiles / area_of_one_white_tile = expected_num_white_tiles :=
by
  sorry

end number_of_white_tiles_l1010_101014


namespace rhombus_diagonals_l1010_101039

theorem rhombus_diagonals (p d1 d2 : ℝ) (h1 : p = 100) (h2 : abs (d1 - d2) = 34) :
  ∃ d1 d2 : ℝ, d1 = 14 ∧ d2 = 48 :=
by
  -- proof omitted
  sorry

end rhombus_diagonals_l1010_101039


namespace bus_average_speed_excluding_stoppages_l1010_101024

theorem bus_average_speed_excluding_stoppages :
  ∀ v : ℝ, (32 / 60) * v = 40 → v = 75 :=
by
  intro v
  intro h
  sorry

end bus_average_speed_excluding_stoppages_l1010_101024


namespace root_equivalence_l1010_101059

theorem root_equivalence (a_1 a_2 a_3 b : ℝ) :
  (∃ c_1 c_2 c_3 : ℝ, c_1 ≠ c_2 ∧ c_2 ≠ c_3 ∧ c_1 ≠ c_3 ∧
    (∀ x : ℝ, (x - a_1) * (x - a_2) * (x - a_3) = b ↔ (x = c_1 ∨ x = c_2 ∨ x = c_3))) →
  (∀ x : ℝ, (x + c_1) * (x + c_2) * (x + c_3) = b ↔ (x = -a_1 ∨ x = -a_2 ∨ x = -a_3)) :=
by 
  sorry

end root_equivalence_l1010_101059


namespace average_remaining_two_numbers_l1010_101023

theorem average_remaining_two_numbers 
    (a b c d e f : ℝ)
    (h1 : (a + b + c + d + e + f) / 6 = 3.95)
    (h2 : (a + b) / 2 = 4.4)
    (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 3.6 := 
sorry

end average_remaining_two_numbers_l1010_101023


namespace count_valid_sequences_l1010_101020

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (x : ℕ → ℕ) : Prop :=
  (x 7 % 2 = 0) ∧ (∀ i < 7, (x i % 2 = 0 → x (i + 1) % 2 = 1) ∧ (x i % 2 = 1 → x (i + 1) % 2 = 0))

theorem count_valid_sequences : ∃ n, 
  n = 78125 ∧ 
  ∃ x : ℕ → ℕ, 
    (∀ i < 8, 0 ≤ x i ∧ x i ≤ 9) ∧ valid_sequence x :=
sorry

end count_valid_sequences_l1010_101020


namespace find_b_l1010_101093

theorem find_b 
  (a b c x : ℝ)
  (h : (3 * x^2 - 4 * x + 5 / 2) * (a * x^2 + b * x + c) 
       = 6 * x^4 - 17 * x^3 + 11 * x^2 - 7 / 2 * x + 5 / 3) 
  (ha : 3 * a = 6) : b = -3 := 
by 
  sorry

end find_b_l1010_101093


namespace income_of_deceased_member_l1010_101089

theorem income_of_deceased_member
  (A B C : ℝ) -- Incomes of the three members
  (h1 : (A + B + C) / 3 = 735)
  (h2 : (A + B) / 2 = 650) :
  C = 905 :=
by
  sorry

end income_of_deceased_member_l1010_101089


namespace largest_is_B_l1010_101078

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_is_B_l1010_101078


namespace solve_equations_l1010_101046

theorem solve_equations :
  (∃ x1 x2 : ℝ, (x1 = 1 ∧ x2 = 3) ∧ (x1^2 - 4 * x1 + 3 = 0) ∧ (x2^2 - 4 * x2 + 3 = 0)) ∧
  (∃ y1 y2 : ℝ, (y1 = 9 ∧ y2 = 11 / 7) ∧ (4 * (2 * y1 - 5)^2 = (3 * y1 - 1)^2) ∧ (4 * (2 * y2 - 5)^2 = (3 * y2 - 1)^2)) :=
by
  sorry

end solve_equations_l1010_101046


namespace find_g_neg_five_l1010_101003

-- Given function and its properties
variables (g : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom ax2 : ∀ (x : ℝ), g x ≠ 0
axiom ax3 : g 5 = 2

-- Theorem to prove
theorem find_g_neg_five : g (-5) = 8 :=
sorry

end find_g_neg_five_l1010_101003


namespace fraction_equality_l1010_101004

variables {a b : ℝ}

theorem fraction_equality (h : ab * (a + b) = 1) (ha : a > 0) (hb : b > 0) : 
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := 
sorry

end fraction_equality_l1010_101004


namespace Laticia_knitted_socks_l1010_101029

theorem Laticia_knitted_socks (x : ℕ) (cond1 : x ≥ 0)
  (cond2 : ∃ y, y = x + 4)
  (cond3 : ∃ z, z = (x + (x + 4)) / 2)
  (cond4 : ∃ w, w = z - 3)
  (cond5 : x + (x + 4) + z + w = 57) : x = 13 := by
  sorry

end Laticia_knitted_socks_l1010_101029


namespace find_expression_l1010_101026

theorem find_expression (x y : ℝ) (h1 : 4 * x + y = 17) (h2 : x + 4 * y = 23) :
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 :=
by
  sorry

end find_expression_l1010_101026


namespace determine_h_l1010_101045

theorem determine_h (x : ℝ) : 
  ∃ h : ℝ → ℝ, (4*x^4 + 11*x^3 + h x = 10*x^3 - x^2 + 4*x - 7) ↔ (h x = -4*x^4 - x^3 - x^2 + 4*x - 7) :=
by
  sorry

end determine_h_l1010_101045


namespace future_cup_defensive_analysis_l1010_101070

variables (avg_A : ℝ) (std_dev_A : ℝ) (avg_B : ℝ) (std_dev_B : ℝ)

-- Statement translations:
-- A: On average, Class B has better defensive skills than Class A.
def stat_A : Prop := avg_B < avg_A

-- C: Class B sometimes performs very well in defense, while other times it performs relatively poorly.
def stat_C : Prop := std_dev_B > std_dev_A

-- D: Class A rarely concedes goals.
def stat_D : Prop := avg_A <= 1.9 -- It's implied that 'rarely' indicates consistency and a lower average threshold, so this represents that.

theorem future_cup_defensive_analysis (h_avg_A : avg_A = 1.9) (h_std_dev_A : std_dev_A = 0.3) 
  (h_avg_B : avg_B = 1.3) (h_std_dev_B : std_dev_B = 1.2) :
  stat_A avg_A avg_B ∧ stat_C std_dev_A std_dev_B ∧ stat_D avg_A :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end future_cup_defensive_analysis_l1010_101070


namespace number_of_teachers_l1010_101042

theorem number_of_teachers
    (number_of_students : ℕ)
    (classes_per_student : ℕ)
    (classes_per_teacher : ℕ)
    (students_per_class : ℕ)
    (total_teachers : ℕ)
    (h1 : number_of_students = 2400)
    (h2 : classes_per_student = 5)
    (h3 : classes_per_teacher = 4)
    (h4 : students_per_class = 30)
    (h5 : total_teachers * classes_per_teacher * students_per_class = number_of_students * classes_per_student) :
    total_teachers = 100 :=
by
  sorry

end number_of_teachers_l1010_101042


namespace extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l1010_101092

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x + x^2 / 2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

theorem extreme_values_for_f_when_a_is_one :
  (∀ x : ℝ, (f 1 x) ≤ 0) ∧ f 1 0 = 0 ∧ f 1 1 = (1 / Real.exp 1) - 1 / 2 :=
sorry

theorem number_of_zeros_of_h (a : ℝ) :
  (0 ≤ a → 
   if 1 < a ∧ a < Real.exp 1 / 2 then
     ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ h a x1 = 0 ∧ h a x2 = 0
   else if 0 ≤ a ∧ a ≤ 1 ∨ a = Real.exp 1 / 2 then
     ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h a x = 0
   else
     ∀ x : ℝ, x > 0 → h a x ≠ 0) :=
sorry

end extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l1010_101092


namespace sum_of_possible_values_l1010_101097

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 10) = -7) :
  ∃ N1 N2 : ℝ, (N1 * (N1 - 10) = -7 ∧ N2 * (N2 - 10) = -7) ∧ (N1 + N2 = 10) :=
sorry

end sum_of_possible_values_l1010_101097


namespace find_x_l1010_101043

theorem find_x 
  (x : ℝ) 
  (angle_PQS angle_QSR angle_SRQ : ℝ) 
  (h1 : angle_PQS = 2 * x)
  (h2 : angle_QSR = 50)
  (h3 : angle_SRQ = x) :
  x = 50 :=
sorry

end find_x_l1010_101043


namespace A_eq_B_l1010_101058

variables (α : Type) (Q : α → Prop)
variables (A B C : α → Prop)

-- Conditions
-- 1. For the questions where both B and C answered "yes", A also answered "yes".
axiom h1 : ∀ q, B q ∧ C q → A q
-- 2. For the questions where A answered "yes", B also answered "yes".
axiom h2 : ∀ q, A q → B q
-- 3. For the questions where B answered "yes", at least one of A and C answered "yes".
axiom h3 : ∀ q, B q → (A q ∨ C q)

-- Prove that A and B gave the same answer to all questions
theorem A_eq_B : ∀ q, A q ↔ B q :=
sorry

end A_eq_B_l1010_101058


namespace paid_more_than_free_l1010_101087

def num_men : ℕ := 194
def num_women : ℕ := 235
def free_admission : ℕ := 68
def total_people (num_men num_women : ℕ) : ℕ := num_men + num_women
def paid_admission (total_people free_admission : ℕ) : ℕ := total_people - free_admission
def paid_over_free (paid_admission free_admission : ℕ) : ℕ := paid_admission - free_admission

theorem paid_more_than_free :
  paid_over_free (paid_admission (total_people num_men num_women) free_admission) free_admission = 293 := 
by
  sorry

end paid_more_than_free_l1010_101087


namespace factorize_l1010_101040

variables (a b x y : ℝ)

theorem factorize : (a * x - b * y)^2 + (a * y + b * x)^2 = (x^2 + y^2) * (a^2 + b^2) :=
by
  sorry

end factorize_l1010_101040


namespace speed_of_rest_distance_l1010_101069

theorem speed_of_rest_distance (D V : ℝ) (h1 : D = 26.67)
                                (h2 : (D / 2) / 5 + (D / 2) / V = 6) : 
  V = 20 :=
by
  sorry

end speed_of_rest_distance_l1010_101069


namespace larger_number_is_299_l1010_101098

theorem larger_number_is_299 {a b : ℕ} (hcf : Nat.gcd a b = 23) (lcm_factors : ∃ k1 k2 : ℕ, Nat.lcm a b = 23 * k1 * k2 ∧ k1 = 12 ∧ k2 = 13) :
  max a b = 299 :=
by
  sorry

end larger_number_is_299_l1010_101098


namespace hardware_contract_probability_l1010_101048

noncomputable def P_S' : ℚ := 3 / 5
noncomputable def P_at_least_one : ℚ := 5 / 6
noncomputable def P_H_and_S : ℚ := 0.31666666666666654 -- 19 / 60 in fraction form
noncomputable def P_S : ℚ := 1 - P_S'

theorem hardware_contract_probability :
  (P_at_least_one = P_H + P_S - P_H_and_S) →
  P_H = 0.75 :=
by
  sorry

end hardware_contract_probability_l1010_101048


namespace shifted_polynomial_roots_are_shifted_l1010_101022

noncomputable def original_polynomial : Polynomial ℝ := Polynomial.X ^ 3 - 5 * Polynomial.X + 7
noncomputable def shifted_polynomial : Polynomial ℝ := Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 7 * Polynomial.X + 5

theorem shifted_polynomial_roots_are_shifted :
  (∀ (a b c : ℝ), (original_polynomial.eval a = 0) ∧ (original_polynomial.eval b = 0) ∧ (original_polynomial.eval c = 0) 
    → (shifted_polynomial.eval (a - 2) = 0) ∧ (shifted_polynomial.eval (b - 2) = 0) ∧ (shifted_polynomial.eval (c - 2) = 0)) :=
by
  sorry

end shifted_polynomial_roots_are_shifted_l1010_101022


namespace min_m_plus_n_l1010_101057

theorem min_m_plus_n (m n : ℕ) (h₁ : m > n) (h₂ : 4^m + 4^n % 100 = 0) : m + n = 7 :=
sorry

end min_m_plus_n_l1010_101057


namespace boxcar_capacity_ratio_l1010_101055

-- The known conditions translated into Lean definitions
def red_boxcar_capacity (B : ℕ) : ℕ := 3 * B
def blue_boxcar_count : ℕ := 4
def red_boxcar_count : ℕ := 3
def black_boxcar_count : ℕ := 7
def black_boxcar_capacity : ℕ := 4000
def total_capacity : ℕ := 132000

-- The mathematical condition as a Lean theorem statement.
theorem boxcar_capacity_ratio 
  (B : ℕ)
  (h_condition : (red_boxcar_count * red_boxcar_capacity B + 
                  blue_boxcar_count * B + 
                  black_boxcar_count * black_boxcar_capacity = 
                  total_capacity)) : 
  black_boxcar_capacity / B = 1 / 2 := 
sorry

end boxcar_capacity_ratio_l1010_101055


namespace product_of_repeating_decimal_l1010_101019

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end product_of_repeating_decimal_l1010_101019


namespace ducks_joined_l1010_101001

theorem ducks_joined (initial_ducks total_ducks ducks_joined : ℕ) 
  (h_initial: initial_ducks = 13)
  (h_total: total_ducks = 33) :
  initial_ducks + ducks_joined = total_ducks → ducks_joined = 20 :=
by
  intros h_equation
  rw [h_initial, h_total] at h_equation
  sorry

end ducks_joined_l1010_101001


namespace larger_triangle_perimeter_l1010_101068

theorem larger_triangle_perimeter 
    (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h1 : a = 6) (h2 : b = 8)
    (hypo_large : ∀ c : ℝ, c = 20) : 
    (2 * a + 2 * b + 20 = 48) :=
by {
  sorry
}

end larger_triangle_perimeter_l1010_101068


namespace common_divisors_count_l1010_101065

-- Define the numbers
def num1 := 9240
def num2 := 13860

-- Define the gcd of the numbers
def gcdNum := Nat.gcd num1 num2

-- Prove the number of divisors of the gcd is 48
theorem common_divisors_count : (Nat.divisors gcdNum).card = 48 :=
by
  -- Normally we would provide a detailed proof here
  sorry

end common_divisors_count_l1010_101065


namespace son_age_l1010_101051

theorem son_age {M S : ℕ} 
  (h1 : M = S + 18) 
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 16 := 
by
  sorry

end son_age_l1010_101051


namespace construct_rhombus_l1010_101033

-- Define data structure representing a point in a 2-dimensional Euclidean space.
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for four points to form a rhombus.
def isRhombus (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2

-- Define circumradius condition for triangle ABC
def circumradius (A B C : Point) (R : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- Define inradius condition for triangle BCD
def inradius (B C D : Point) (r : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- The proposition to be proved: We can construct the rhombus ABCD given R and r.
theorem construct_rhombus (A B C D : Point) (R r : ℝ) :
  (circumradius A B C R) →
  (inradius B C D r) →
  isRhombus A B C D :=
by
  sorry

end construct_rhombus_l1010_101033


namespace fraction_of_married_men_is_two_fifths_l1010_101085

noncomputable def fraction_of_married_men (W : ℕ) (p : ℚ) (h : p = 1 / 3) : ℚ :=
  let W_s := p * W
  let W_m := W - W_s
  let M_m := W_m
  let T := W + M_m
  M_m / T

theorem fraction_of_married_men_is_two_fifths (W : ℕ) (p : ℚ) (h : p = 1 / 3) (hW : W = 6) : fraction_of_married_men W p h = 2 / 5 :=
by
  sorry

end fraction_of_married_men_is_two_fifths_l1010_101085


namespace train_pass_jogger_in_41_seconds_l1010_101028

-- Definitions based on conditions
def jogger_speed_kmh := 9 -- in km/hr
def train_speed_kmh := 45 -- in km/hr
def initial_distance_jogger := 200 -- in meters
def train_length := 210 -- in meters

-- Converting speeds from km/hr to m/s
def kmh_to_ms (kmh: ℕ) : ℕ := (kmh * 1000) / 3600

def jogger_speed_ms := kmh_to_ms jogger_speed_kmh -- in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh -- in m/s

-- Relative speed of the train with respect to the jogger
def relative_speed := train_speed_ms - jogger_speed_ms -- in m/s

-- Total distance to be covered by the train to pass the jogger
def total_distance := initial_distance_jogger + train_length -- in meters

-- Time taken to pass the jogger
def time_to_pass (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem train_pass_jogger_in_41_seconds : time_to_pass total_distance relative_speed = 41 :=
by
  sorry

end train_pass_jogger_in_41_seconds_l1010_101028


namespace green_valley_ratio_l1010_101006

variable (j s : ℕ)

theorem green_valley_ratio (h : (3 / 4 : ℚ) * j = (1 / 2 : ℚ) * s) : s = 3 / 2 * j :=
by
  sorry

end green_valley_ratio_l1010_101006


namespace tire_price_l1010_101090

-- Definitions based on given conditions
def tire_cost (T : ℝ) (n : ℕ) : Prop :=
  n * T + 56 = 224

-- The equivalence we want to prove
theorem tire_price (T : ℝ) (n : ℕ) (h : tire_cost T n) : n * T = 168 :=
by
  sorry

end tire_price_l1010_101090


namespace elderly_in_sample_l1010_101084

variable (A E M : ℕ)
variable (total_employees : ℕ)
variable (total_young : ℕ)
variable (sample_size_young : ℕ)
variable (sampling_ratio : ℚ)
variable (sample_elderly : ℕ)

axiom condition_1 : total_young = 160
axiom condition_2 : total_employees = 430
axiom condition_3 : M = 2 * E
axiom condition_4 : A + M + E = total_employees
axiom condition_5 : sampling_ratio = sample_size_young / total_young
axiom sampling : sample_size_young = 32
axiom elderly_employees : sample_elderly = 18

theorem elderly_in_sample : sample_elderly = sampling_ratio * E := by
  -- Proof steps are not provided
  sorry

end elderly_in_sample_l1010_101084


namespace num_more_green_l1010_101021

noncomputable def num_people : ℕ := 150
noncomputable def more_blue : ℕ := 90
noncomputable def both_green_blue : ℕ := 40
noncomputable def neither_green_blue : ℕ := 20

theorem num_more_green :
  (num_people + more_blue + both_green_blue + neither_green_blue) ≤ 150 →
  (more_blue - both_green_blue) + both_green_blue + neither_green_blue ≤ num_people →
  (num_people - 
  ((more_blue - both_green_blue) + both_green_blue + neither_green_blue)) + both_green_blue = 80 :=
by
    intros h1 h2
    sorry

end num_more_green_l1010_101021


namespace perfect_square_a_value_l1010_101005

theorem perfect_square_a_value (x y a : ℝ) :
  (∃ k : ℝ, x^2 + 2 * x * y + y^2 - a * (x + y) + 25 = k^2) →
  a = 10 ∨ a = -10 :=
sorry

end perfect_square_a_value_l1010_101005


namespace right_triangle_angles_l1010_101013

theorem right_triangle_angles (c : ℝ) (t : ℝ) (h : t = c^2 / 8) :
  ∃(A B: ℝ), A = 90 ∧ (B = 75 ∨ B = 15) :=
by
  sorry

end right_triangle_angles_l1010_101013


namespace solution_set_l1010_101047

theorem solution_set (x : ℝ) : (3 ≤ |5 - 2 * x| ∧ |5 - 2 * x| < 9) ↔ (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) :=
sorry

end solution_set_l1010_101047


namespace a_2n_is_square_l1010_101062

def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else a_n (n - 1) + a_n (n - 3) + a_n (n - 4)

theorem a_2n_is_square (n : ℕ) : ∃ k : ℕ, a_n (2 * n) = k * k := by
  sorry

end a_2n_is_square_l1010_101062


namespace large_rectangle_perimeter_l1010_101056

-- Definitions from the conditions
def side_length_of_square (perimeter_square : ℕ) : ℕ := perimeter_square / 4
def width_of_small_rectangle (perimeter_rect : ℕ) (side_length : ℕ) : ℕ := (perimeter_rect / 2) - side_length

-- Given conditions
def perimeter_square := 24
def perimeter_rect := 16
def side_length := side_length_of_square perimeter_square
def rect_width := width_of_small_rectangle perimeter_rect side_length
def large_rectangle_height := side_length + rect_width
def large_rectangle_width := 3 * side_length

-- Perimeter calculation
def perimeter_large_rectangle (width height : ℕ) : ℕ := 2 * (width + height)

-- Proof problem statement
theorem large_rectangle_perimeter : 
  perimeter_large_rectangle large_rectangle_width large_rectangle_height = 52 :=
sorry

end large_rectangle_perimeter_l1010_101056


namespace initial_people_in_line_l1010_101079

theorem initial_people_in_line (X : ℕ) 
  (h1 : X - 6 + 3 = 18) : X = 21 :=
  sorry

end initial_people_in_line_l1010_101079


namespace remaining_days_to_finish_coke_l1010_101037

def initial_coke_in_ml : ℕ := 2000
def daily_consumption_in_ml : ℕ := 200
def days_already_drunk : ℕ := 3

theorem remaining_days_to_finish_coke : 
  (initial_coke_in_ml / daily_consumption_in_ml) - days_already_drunk = 7 := 
by
  sorry -- Proof placeholder

end remaining_days_to_finish_coke_l1010_101037


namespace regular_tire_price_l1010_101000

theorem regular_tire_price 
  (x : ℝ) 
  (h1 : 3 * x + x / 2 = 300) 
  : x = 600 / 7 := 
sorry

end regular_tire_price_l1010_101000


namespace washing_machine_capacity_l1010_101071

theorem washing_machine_capacity 
  (shirts : ℕ) (sweaters : ℕ) (loads : ℕ) (total_clothing : ℕ) (n : ℕ)
  (h1 : shirts = 43) (h2 : sweaters = 2) (h3 : loads = 9)
  (h4 : total_clothing = shirts + sweaters)
  (h5 : total_clothing / loads = n) :
  n = 5 :=
sorry

end washing_machine_capacity_l1010_101071


namespace crate_stacking_probability_l1010_101016

theorem crate_stacking_probability :
  ∃ (p q : ℕ), (p.gcd q = 1) ∧ (p : ℚ) / q = 170 / 6561 ∧ (total_height = 50) ∧ (number_of_crates = 12) ∧ (orientation_probability = 1 / 3) :=
sorry

end crate_stacking_probability_l1010_101016


namespace john_pays_total_l1010_101073

-- Definitions based on conditions
def total_cans : ℕ := 30
def price_per_can : ℝ := 0.60

-- Main statement to be proven
theorem john_pays_total : (total_cans / 2) * price_per_can = 9 := 
by
  sorry

end john_pays_total_l1010_101073


namespace ratio_of_dancers_l1010_101074

theorem ratio_of_dancers (total_kids total_dancers slow_dance non_slow_dance : ℕ)
  (h1 : total_kids = 140)
  (h2 : slow_dance = 25)
  (h3 : non_slow_dance = 10)
  (h4 : total_dancers = slow_dance + non_slow_dance) :
  (total_dancers : ℚ) / total_kids = 1 / 4 :=
by
  sorry

end ratio_of_dancers_l1010_101074


namespace number_of_pencil_boxes_l1010_101034

-- Define the total number of pencils and pencils per box as given conditions
def total_pencils : ℝ := 2592
def pencils_per_box : ℝ := 648.0

-- Problem statement: To prove the number of pencil boxes is 4
theorem number_of_pencil_boxes : total_pencils / pencils_per_box = 4 := by
  sorry

end number_of_pencil_boxes_l1010_101034


namespace second_year_students_sampled_l1010_101002

def total_students (f s t : ℕ) : ℕ := f + s + t

def proportion_second_year (s total_stu : ℕ) : ℚ := s / total_stu

def sampled_second_year_students (p : ℚ) (n : ℕ) : ℚ := p * n

theorem second_year_students_sampled
  (f s t : ℕ) (n : ℕ)
  (h1 : f = 600)
  (h2 : s = 780)
  (h3 : t = 720)
  (h4 : n = 35) :
  sampled_second_year_students (proportion_second_year s (total_students f s t)) n = 13 := 
sorry

end second_year_students_sampled_l1010_101002


namespace problem_solution_l1010_101064

noncomputable def root1 : ℝ := (3 + Real.sqrt 105) / 4
noncomputable def root2 : ℝ := (3 - Real.sqrt 105) / 4

theorem problem_solution :
  (∀ x : ℝ, x ≠ -2 → x ≠ -3 → (x^3 - x^2 - 4 * x) / (x^2 + 5 * x + 6) + x = -4
    → x = root1 ∨ x = root2) := 
by
  sorry

end problem_solution_l1010_101064


namespace shauna_min_test_score_l1010_101018

theorem shauna_min_test_score (score1 score2 score3 : ℕ) (h1 : score1 = 82) (h2 : score2 = 88) (h3 : score3 = 95) 
  (max_score : ℕ) (h4 : max_score = 100) (desired_avg : ℕ) (h5 : desired_avg = 85) :
  ∃ (score4 score5 : ℕ), score4 ≥ 75 ∧ score5 ≥ 75 ∧ (score1 + score2 + score3 + score4 + score5) / 5 = desired_avg :=
by
  -- proof here
  sorry

end shauna_min_test_score_l1010_101018


namespace number_of_students_l1010_101054

theorem number_of_students (n S : ℕ) 
  (h1 : S = 15 * n) 
  (h2 : (S + 36) / (n + 1) = 16) : 
  n = 20 :=
by 
  sorry

end number_of_students_l1010_101054


namespace simplify_and_evaluate_l1010_101061

theorem simplify_and_evaluate (x : ℝ) (h : x = 2 + Real.sqrt 2) :
  (1 - 3 / (x + 1)) / ((x^2 - 4*x + 4) / (x + 1)) = Real.sqrt 2 / 2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_l1010_101061


namespace median_and_mode_l1010_101060

theorem median_and_mode (data : List ℝ) (h : data = [6, 7, 4, 7, 5, 2]) :
  ∃ median mode, median = 5.5 ∧ mode = 7 := 
by {
  sorry
}

end median_and_mode_l1010_101060


namespace find_range_of_k_l1010_101076

-- Define the conditions and the theorem
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

theorem find_range_of_k :
  {k : ℝ | is_ellipse k} = {k : ℝ | (-3 < k ∧ k < -1/2) ∨ (-1/2 < k ∧ k < 2)} :=
by
  sorry

end find_range_of_k_l1010_101076


namespace b_2018_eq_5043_l1010_101035

def b (n : Nat) : Nat :=
  if n % 2 = 1 then 5 * ((n + 1) / 2) - 3 else 5 * (n / 2) - 2

theorem b_2018_eq_5043 : b 2018 = 5043 := by
  sorry

end b_2018_eq_5043_l1010_101035


namespace sandy_paint_area_l1010_101038

-- Define the dimensions of the wall
def wall_height : ℕ := 10
def wall_length : ℕ := 15

-- Define the dimensions of the decorative region
def deco_height : ℕ := 3
def deco_length : ℕ := 5

-- Calculate the areas and prove the required area to paint
theorem sandy_paint_area :
  wall_height * wall_length - deco_height * deco_length = 135 := by
  sorry

end sandy_paint_area_l1010_101038


namespace remainder_five_n_minus_eleven_l1010_101063

theorem remainder_five_n_minus_eleven (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := 
    sorry

end remainder_five_n_minus_eleven_l1010_101063


namespace circle_radius_l1010_101066

theorem circle_radius (r : ℝ) (hr : 3 * (2 * Real.pi * r) = 2 * Real.pi * r^2) : r = 3 :=
by 
  sorry

end circle_radius_l1010_101066


namespace daria_weeks_needed_l1010_101049

-- Defining the parameters and conditions
def initial_amount : ℕ := 20
def weekly_savings : ℕ := 10
def cost_of_vacuum_cleaner : ℕ := 120

-- Defining the total money Daria needs to add to her initial amount
def additional_amount_needed : ℕ := cost_of_vacuum_cleaner - initial_amount

-- Defining the number of weeks needed to save the additional amount, given weekly savings
def weeks_needed : ℕ := additional_amount_needed / weekly_savings

-- The theorem stating that Daria needs exactly 10 weeks to cover the expense of the vacuum cleaner
theorem daria_weeks_needed : weeks_needed = 10 := by
  sorry

end daria_weeks_needed_l1010_101049


namespace find_x_l1010_101036

-- Given condition that x is 11 percent greater than 90
def eleven_percent_greater (x : ℝ) : Prop := x = 90 + (11 / 100) * 90

-- Theorem statement
theorem find_x (x : ℝ) (h: eleven_percent_greater x) : x = 99.9 :=
  sorry

end find_x_l1010_101036


namespace ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l1010_101067

theorem ones_digit_largest_power_of_three_divides_factorial_3_pow_3 :
  (3 ^ 13) % 10 = 3 := by
  sorry

end ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l1010_101067


namespace part_I_part_II_l1010_101094

noncomputable def f (x : ℝ) : ℝ := 5 + Real.log x
noncomputable def g (x k : ℝ) : ℝ := k * x / (x + 1)

theorem part_I (k : ℝ) : 
  (∃ x0, g x0 k = x0 + 4 ∧ (k / (x0 + 1)^2) = 1) ↔ (k = 1 ∨ k = 9) :=
by
  sorry

theorem part_II (k : ℕ) : (∀ x : ℝ, 1 < x → f x > g x k) → k ≤ 7 :=
by
  sorry

end part_I_part_II_l1010_101094


namespace gcf_lcm_15_l1010_101017

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_15 : 
  GCF (LCM 9 15) (LCM 10 21) = 15 :=
by 
  sorry

end gcf_lcm_15_l1010_101017


namespace not_divisible_by_81_l1010_101077

theorem not_divisible_by_81 (n : ℤ) : ¬ (81 ∣ n^3 - 9 * n + 27) :=
sorry

end not_divisible_by_81_l1010_101077


namespace multiplier_is_five_l1010_101009

-- condition 1: n = m * (n - 4)
-- condition 2: n = 5
-- question: prove m = 5

theorem multiplier_is_five (n m : ℝ) 
  (h1 : n = m * (n - 4)) 
  (h2 : n = 5) : m = 5 := 
  sorry

end multiplier_is_five_l1010_101009


namespace cake_divided_into_equal_parts_l1010_101031

theorem cake_divided_into_equal_parts (cake_weight : ℕ) (pierre : ℕ) (nathalie : ℕ) (parts : ℕ) 
  (hw_eq : cake_weight = 400)
  (hp_eq : pierre = 100)
  (pn_eq : pierre = 2 * nathalie)
  (parts_eq : cake_weight / nathalie = parts)
  (hparts_eq : parts = 8) :
  cake_weight / nathalie = 8 := 
by
  sorry

end cake_divided_into_equal_parts_l1010_101031


namespace fraction_value_unchanged_l1010_101053

theorem fraction_value_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / (x + y) = (2 * x) / (2 * (x + y))) :=
by sorry

end fraction_value_unchanged_l1010_101053


namespace min_wins_required_l1010_101095

theorem min_wins_required 
  (total_matches initial_matches remaining_matches : ℕ)
  (points_for_win points_for_draw points_for_defeat current_points target_points : ℕ)
  (matches_played_points : ℕ)
  (h_total : total_matches = 20)
  (h_initial : initial_matches = 5)
  (h_remaining : remaining_matches = total_matches - initial_matches)
  (h_win_points : points_for_win = 3)
  (h_draw_points : points_for_draw = 1)
  (h_defeat_points : points_for_defeat = 0)
  (h_current_points : current_points = 8)
  (h_target_points : target_points = 40)
  (h_matches_played_points : matches_played_points = current_points)
  :
  (∃ min_wins : ℕ, min_wins * points_for_win + (remaining_matches - min_wins) * points_for_defeat >= target_points - matches_played_points ∧ min_wins ≤ remaining_matches) ∧
  (∀ other_wins : ℕ, other_wins < min_wins → (other_wins * points_for_win + (remaining_matches - other_wins) * points_for_defeat < target_points - matches_played_points)) :=
sorry

end min_wins_required_l1010_101095


namespace rhombus_area_l1010_101080

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 5) (h_d2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 20 :=
by
  sorry

end rhombus_area_l1010_101080


namespace factorize_poly1_factorize_poly2_l1010_101025

theorem factorize_poly1 (x : ℝ) : 2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4) :=
by
  sorry

theorem factorize_poly2 (x : ℝ) : x^2 - 14 * x + 49 = (x - 7) ^ 2 :=
by
  sorry

end factorize_poly1_factorize_poly2_l1010_101025


namespace total_apples_l1010_101007

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l1010_101007


namespace Jerry_weekly_earnings_l1010_101081

def hours_per_task : ℕ := 2
def pay_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

theorem Jerry_weekly_earnings : pay_per_task * (hours_per_day / hours_per_task) * days_per_week = 1400 :=
by
  -- Carry out the proof here
  sorry

end Jerry_weekly_earnings_l1010_101081


namespace sin_inequality_l1010_101083

theorem sin_inequality (x : ℝ) (hx1 : 0 < x) (hx2 : x < Real.pi / 4) : 
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by 
  sorry

end sin_inequality_l1010_101083


namespace find_the_number_l1010_101011

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end find_the_number_l1010_101011


namespace mice_population_l1010_101050

theorem mice_population :
  ∃ (mice_initial : ℕ) (pups_per_mouse : ℕ) (survival_rate_first_gen : ℕ → ℕ) 
    (survival_rate_second_gen : ℕ → ℕ) (num_dead_first_gen : ℕ) (pups_eaten_per_adult : ℕ)
    (total_mice : ℕ),
    mice_initial = 8 ∧ pups_per_mouse = 7 ∧
    (∀ n, survival_rate_first_gen n = (n * 80) / 100) ∧
    (∀ n, survival_rate_second_gen n = (n * 60) / 100) ∧
    num_dead_first_gen = 2 ∧ pups_eaten_per_adult = 3 ∧
    total_mice = mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse)) - num_dead_first_gen + (survival_rate_second_gen ((mice_initial + (survival_rate_first_gen (mice_initial * pups_per_mouse))) * pups_per_mouse)) - ((mice_initial - num_dead_first_gen) * pups_eaten_per_adult) :=
  sorry

end mice_population_l1010_101050


namespace five_letter_words_with_at_least_one_vowel_l1010_101075

theorem five_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F']
  let vowels := ['A', 'E', 'F']
  (6 ^ 5) - (3 ^ 5) = 7533 := by 
  sorry

end five_letter_words_with_at_least_one_vowel_l1010_101075
