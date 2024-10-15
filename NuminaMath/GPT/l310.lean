import Mathlib

namespace NUMINAMATH_GPT_parabola_slopes_l310_31087

theorem parabola_slopes (k : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hC : C = (0, -2)) (hA : A.1^2 = 2 * A.2) (hB : B.1^2 = 2 * B.2) 
    (hA_eq : A.2 = k * A.1 + 2) (hB_eq : B.2 = k * B.1 + 2) :
  ((C.2 - A.2) / (C.1 - A.1))^2 + ((C.2 - B.2) / (C.1 - B.1))^2 - 2 * k^2 = 8 := 
sorry

end NUMINAMATH_GPT_parabola_slopes_l310_31087


namespace NUMINAMATH_GPT_quotient_of_large_div_small_l310_31044

theorem quotient_of_large_div_small (L S : ℕ) (h1 : L - S = 1365)
  (h2 : L = S * (L / S) + 20) (h3 : L = 1634) : (L / S) = 6 := by
  sorry

end NUMINAMATH_GPT_quotient_of_large_div_small_l310_31044


namespace NUMINAMATH_GPT_john_purchased_large_bottles_l310_31016

noncomputable def large_bottle_cost : ℝ := 1.75
noncomputable def small_bottle_cost : ℝ := 1.35
noncomputable def num_small_bottles : ℝ := 690
noncomputable def avg_price_paid : ℝ := 1.6163438256658595
noncomputable def total_small_cost : ℝ := num_small_bottles * small_bottle_cost
noncomputable def total_cost (L : ℝ) : ℝ := large_bottle_cost * L + total_small_cost
noncomputable def total_bottles (L : ℝ) : ℝ := L + num_small_bottles

theorem john_purchased_large_bottles : ∃ L : ℝ, 
  (total_cost L / total_bottles L = avg_price_paid) ∧ 
  (L = 1380) := 
sorry

end NUMINAMATH_GPT_john_purchased_large_bottles_l310_31016


namespace NUMINAMATH_GPT_half_radius_circle_y_l310_31067

-- Conditions
def circle_x_circumference (C : ℝ) : Prop :=
  C = 20 * Real.pi

def circle_x_and_y_same_area (r R : ℝ) : Prop :=
  Real.pi * r^2 = Real.pi * R^2

-- Problem statement: Prove that half the radius of circle y is 5
theorem half_radius_circle_y (r R : ℝ) (hx : circle_x_circumference (2 * Real.pi * r)) (hy : circle_x_and_y_same_area r R) : R / 2 = 5 :=
by sorry

end NUMINAMATH_GPT_half_radius_circle_y_l310_31067


namespace NUMINAMATH_GPT_positive_number_l310_31046

theorem positive_number (n : ℕ) (h : n^2 + 2 * n = 170) : n = 12 :=
sorry

end NUMINAMATH_GPT_positive_number_l310_31046


namespace NUMINAMATH_GPT_true_statement_for_f_l310_31071

variable (c : ℝ) (f : ℝ → ℝ)

theorem true_statement_for_f :
  (∀ x : ℝ, f x = x^2 - 2 * x + c) → (∀ x : ℝ, f x ≥ c - 1) :=
by
  sorry

end NUMINAMATH_GPT_true_statement_for_f_l310_31071


namespace NUMINAMATH_GPT_usual_travel_time_l310_31006

theorem usual_travel_time
  (S : ℝ) (T : ℝ) 
  (h0 : S > 0)
  (h1 : (S / T) = (4 / 5 * S / (T + 6))) : 
  T = 30 :=
by sorry

end NUMINAMATH_GPT_usual_travel_time_l310_31006


namespace NUMINAMATH_GPT_probability_white_ball_is_two_fifths_l310_31042

-- Define the total number of each type of balls.
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Calculate the total number of balls in the bag.
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability calculation.
noncomputable def probability_of_white_ball : ℚ := white_balls / total_balls

-- The theorem statement asserting the probability of drawing a white ball.
theorem probability_white_ball_is_two_fifths :
  probability_of_white_ball = 2 / 5 :=
sorry

end NUMINAMATH_GPT_probability_white_ball_is_two_fifths_l310_31042


namespace NUMINAMATH_GPT_problem_sum_congruent_mod_11_l310_31013

theorem problem_sum_congruent_mod_11 : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := 
by
  -- Proof needed here
  sorry

end NUMINAMATH_GPT_problem_sum_congruent_mod_11_l310_31013


namespace NUMINAMATH_GPT_base7_to_base10_l310_31001

theorem base7_to_base10 (a b c d e : ℕ) (h : 45321 = a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0)
  (ha : a = 4) (hb : b = 5) (hc : c = 3) (hd : d = 2) (he : e = 1) : 
  a * 7^4 + b * 7^3 + c * 7^2 + d * 7^1 + e * 7^0 = 11481 := 
by 
  sorry

end NUMINAMATH_GPT_base7_to_base10_l310_31001


namespace NUMINAMATH_GPT_complex_sum_is_2_l310_31003

theorem complex_sum_is_2 
  (a b c d e f : ℂ) 
  (hb : b = 4) 
  (he : e = 2 * (-a - c)) 
  (hr : a + c + e = 0) 
  (hi : b + d + f = 6) 
  : d + f = 2 := 
  by
  sorry

end NUMINAMATH_GPT_complex_sum_is_2_l310_31003


namespace NUMINAMATH_GPT_exists_polynomial_for_divisors_l310_31004

open Polynomial

theorem exists_polynomial_for_divisors (n : ℕ) :
  (∃ P : ℤ[X], ∀ d : ℕ, d ∣ n → P.eval (d : ℤ) = (n / d : ℤ)^2) ↔
  (Nat.Prime n ∨ n = 1 ∨ n = 6) := by
  sorry

end NUMINAMATH_GPT_exists_polynomial_for_divisors_l310_31004


namespace NUMINAMATH_GPT_proof_problem_l310_31058

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < (π / 2))
variable (htan : tan α = (1 + sin β) / cos β)

theorem proof_problem : 2 * α - β = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l310_31058


namespace NUMINAMATH_GPT_find_d_minus_b_l310_31062

theorem find_d_minus_b (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 19) : d - b = 757 := 
by sorry

end NUMINAMATH_GPT_find_d_minus_b_l310_31062


namespace NUMINAMATH_GPT_reflect_A_across_x_axis_l310_31082

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the reflection function across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Theorem statement: The reflection of point A across the x-axis should be (-3, -2)
theorem reflect_A_across_x_axis : reflect_x A = (-3, -2) := by
  sorry

end NUMINAMATH_GPT_reflect_A_across_x_axis_l310_31082


namespace NUMINAMATH_GPT_sequence_term_500_l310_31025

theorem sequence_term_500 (a : ℕ → ℤ) (h1 : a 1 = 3009) (h2 : a 2 = 3010) 
  (h3 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 500 = 3341 := 
sorry

end NUMINAMATH_GPT_sequence_term_500_l310_31025


namespace NUMINAMATH_GPT_cookies_left_over_l310_31066

def abigail_cookies : Nat := 53
def beatrice_cookies : Nat := 65
def carson_cookies : Nat := 26
def pack_size : Nat := 10

theorem cookies_left_over : (abigail_cookies + beatrice_cookies + carson_cookies) % pack_size = 4 := 
by
  sorry

end NUMINAMATH_GPT_cookies_left_over_l310_31066


namespace NUMINAMATH_GPT_sin_cos_relation_l310_31093

theorem sin_cos_relation 
  (α β : Real) 
  (h : 2 * Real.sin α - Real.cos β = 2) 
  : Real.sin α + 2 * Real.cos β = 1 ∨ Real.sin α + 2 * Real.cos β = -1 := 
sorry

end NUMINAMATH_GPT_sin_cos_relation_l310_31093


namespace NUMINAMATH_GPT_determinant_zero_implies_sum_neg_nine_l310_31091

theorem determinant_zero_implies_sum_neg_nine
  (x y : ℝ)
  (h1 : x ≠ y)
  (h2 : x * y = 1)
  (h3 : (Matrix.det ![
    ![1, 5, 8], 
    ![3, x, y], 
    ![3, y, x]
  ]) = 0) : 
  x + y = -9 := 
sorry

end NUMINAMATH_GPT_determinant_zero_implies_sum_neg_nine_l310_31091


namespace NUMINAMATH_GPT_initial_yards_lost_l310_31020

theorem initial_yards_lost (x : ℤ) (h : -x + 7 = 2) : x = 5 := by
  sorry

end NUMINAMATH_GPT_initial_yards_lost_l310_31020


namespace NUMINAMATH_GPT_area_of_triangle_from_line_l310_31005

-- Define the conditions provided in the problem
def line_eq (B : ℝ) (x y : ℝ) := B * x + 9 * y = 18
def B_val := (36 : ℝ)

theorem area_of_triangle_from_line (B : ℝ) (hB : B = B_val) : 
  (∃ C : ℝ, C = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_from_line_l310_31005


namespace NUMINAMATH_GPT_arithmetic_sequence_proof_l310_31027

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_proof
  (a₁ d : ℝ)
  (h : a 4 a₁ d + a 6 a₁ d + a 8 a₁ d + a 10 a₁ d + a 12 a₁ d = 120) :
  a 7 a₁ d - (1 / 3) * a 5 a₁ d = 16 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_proof_l310_31027


namespace NUMINAMATH_GPT_pairwise_sums_modulo_l310_31098

theorem pairwise_sums_modulo (n : ℕ) (h : n = 2011) :
  ∃ (sums_div_3 sums_rem_1 : ℕ),
  (sums_div_3 = (n * (n - 1)) / 6) ∧
  (sums_rem_1 = (n * (n - 1)) / 6) := by
  sorry

end NUMINAMATH_GPT_pairwise_sums_modulo_l310_31098


namespace NUMINAMATH_GPT_range_of_a_sq_l310_31083

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ m n : ℕ, a (m + n) = a m + a n

theorem range_of_a_sq {n : ℕ}
  (h_arith : arithmetic_sequence a)
  (h_cond : a 1 ^ 2 + a (2 * n + 1) ^ 2 = 1) :
  ∃ (L R : ℝ), (L = 2) ∧ (∀ k : ℕ, a (n+1) ^ 2 + a (3*n+1) ^ 2 ≥ L) := sorry

end NUMINAMATH_GPT_range_of_a_sq_l310_31083


namespace NUMINAMATH_GPT_sum_ge_six_l310_31068

theorem sum_ge_six (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a ≥ 12) : a + b + c ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_ge_six_l310_31068


namespace NUMINAMATH_GPT_sin_3x_over_4_period_l310_31078

noncomputable def sine_period (b : ℝ) : ℝ :=
  (2 * Real.pi) / b

theorem sin_3x_over_4_period :
  sine_period (3/4) = (8 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_3x_over_4_period_l310_31078


namespace NUMINAMATH_GPT_distribute_tickets_among_people_l310_31070

noncomputable def distribution_ways : ℕ := 84

theorem distribute_tickets_among_people (tickets : Fin 5 → ℕ) (persons : Fin 4 → ℕ)
  (h1 : ∀ p : Fin 4, ∃ t : Fin 5, tickets t = persons p)
  (h2 : ∀ p : Fin 4, ∀ t1 t2 : Fin 5, tickets t1 = persons p ∧ tickets t2 = persons p → (t1.val + 1 = t2.val ∨ t2.val + 1 = t1.val)) :
  ∃ n : ℕ, n = distribution_ways := by
  use 84
  trivial

end NUMINAMATH_GPT_distribute_tickets_among_people_l310_31070


namespace NUMINAMATH_GPT_find_f_729_l310_31009

variable (f : ℕ+ → ℕ+) -- Define the function f on the positive integers.

-- Conditions of the problem.
axiom h1 : ∀ n : ℕ+, f (f n) = 3 * n
axiom h2 : ∀ n : ℕ+, f (3 * n + 1) = 3 * n + 2 

-- Proof statement.
theorem find_f_729 : f 729 = 729 :=
by
  sorry -- Placeholder for the proof.

end NUMINAMATH_GPT_find_f_729_l310_31009


namespace NUMINAMATH_GPT_initial_fish_l310_31053

-- Define the conditions of the problem
def fish_bought : Float := 280.0
def current_fish : Float := 492.0

-- Define the question to be proved
theorem initial_fish (x : Float) (h : x + fish_bought = current_fish) : x = 212 :=
by 
  sorry

end NUMINAMATH_GPT_initial_fish_l310_31053


namespace NUMINAMATH_GPT_line_equation_with_equal_intercepts_l310_31047

theorem line_equation_with_equal_intercepts 
  (a : ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h : ∀ x y, l x y ↔ (a+1)*x + y + 2 - a = 0) 
  (intercept_condition : ∀ x y, l x 0 = l 0 y) : 
  (∀ x y, l x y ↔ x + y + 2 = 0) ∨ (∀ x y, l x y ↔ 3*x + y = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_with_equal_intercepts_l310_31047


namespace NUMINAMATH_GPT_determine_N_l310_31030

/-- 
Each row and two columns in the grid forms distinct arithmetic sequences.
Given:
- First column values: 10 and 18 (arithmetic sequence).
- Second column top value: N, bottom value: -23 (arithmetic sequence).
Prove that N = -15.
 -/
theorem determine_N : ∃ N : ℤ, (∀ n : ℕ, 10 + n * 8 = 10 ∨ 10 + n * 8 = 18) ∧ (∀ m : ℕ, N + m * 8 = N ∨ N + m * 8 = -23) ∧ N = -15 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_N_l310_31030


namespace NUMINAMATH_GPT_xy_max_value_l310_31031

theorem xy_max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 12) :
  xy <= 9 := by
  sorry

end NUMINAMATH_GPT_xy_max_value_l310_31031


namespace NUMINAMATH_GPT_odd_expression_is_odd_l310_31051

theorem odd_expression_is_odd (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) : (4 * p * q + 1) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_odd_expression_is_odd_l310_31051


namespace NUMINAMATH_GPT_range_of_t_l310_31018

theorem range_of_t (a b c : ℝ) (t : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_inequality : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (1 / a^2) + (4 / b^2) + (t / c^2) ≥ 0) :
  t ≥ -9 :=
sorry

end NUMINAMATH_GPT_range_of_t_l310_31018


namespace NUMINAMATH_GPT_max_value_of_sequence_l310_31011

theorem max_value_of_sequence :
  ∃ a : ℕ → ℕ, (∀ i, 1 ≤ i ∧ i ≤ 101 → 0 < a i) →
              (∀ i, 1 ≤ i ∧ i < 101 → (a i + 1) % a (i + 1) = 0) →
              (a 102 = a 1) →
              (∀ n, (1 ≤ n ∧ n ≤ 101) → a n ≤ 201) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_sequence_l310_31011


namespace NUMINAMATH_GPT_Rockham_Soccer_League_members_l310_31039

theorem Rockham_Soccer_League_members (sock_cost tshirt_cost cap_cost total_cost members : ℕ) (h1 : sock_cost = 6) (h2 : tshirt_cost = sock_cost + 10) (h3 : cap_cost = 3) (h4 : total_cost = 4620) (h5 : total_cost = 50 * members) : members = 92 :=
by
  sorry

end NUMINAMATH_GPT_Rockham_Soccer_League_members_l310_31039


namespace NUMINAMATH_GPT_dinosaur_count_l310_31089

theorem dinosaur_count (h : ℕ) (l : ℕ) (H1 : h = 1) (H2 : l = 3) (total_hl : ℕ) (H3 : total_hl = 20) :
  ∃ D : ℕ, 4 * D = total_hl := 
by
  use 5
  sorry

end NUMINAMATH_GPT_dinosaur_count_l310_31089


namespace NUMINAMATH_GPT_Fermat_numbers_are_not_cubes_l310_31059

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem Fermat_numbers_are_not_cubes : ∀ n : ℕ, ¬ ∃ k : ℕ, F n = k^3 :=
by
  sorry

end NUMINAMATH_GPT_Fermat_numbers_are_not_cubes_l310_31059


namespace NUMINAMATH_GPT_find_a4_l310_31055

theorem find_a4 (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n - 3) : a 4 = -8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a4_l310_31055


namespace NUMINAMATH_GPT_smallest_n_terminating_contains_9_l310_31034

def isTerminatingDecimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2 ^ a * 5 ^ b

def containsDigit9 (n : ℕ) : Prop :=
  (Nat.digits 10 n).contains 9

theorem smallest_n_terminating_contains_9 : ∃ n : ℕ, 
  isTerminatingDecimal n ∧
  containsDigit9 n ∧
  (∀ m : ℕ, isTerminatingDecimal m ∧ containsDigit9 m → n ≤ m) ∧
  n = 5120 :=
  sorry

end NUMINAMATH_GPT_smallest_n_terminating_contains_9_l310_31034


namespace NUMINAMATH_GPT_determine_v6_l310_31032

variable (v : ℕ → ℝ)

-- Given initial conditions: v₄ = 12 and v₇ = 471
def initial_conditions := v 4 = 12 ∧ v 7 = 471

-- Recurrence relation definition: vₙ₊₂ = 3vₙ₊₁ + vₙ
def recurrence_relation := ∀ n : ℕ, v (n + 2) = 3 * v (n + 1) + v n

-- The target is to prove that v₆ = 142.5
theorem determine_v6 (h1 : initial_conditions v) (h2 : recurrence_relation v) : 
  v 6 = 142.5 :=
sorry

end NUMINAMATH_GPT_determine_v6_l310_31032


namespace NUMINAMATH_GPT_inequality_solution_l310_31057

-- Define the inequality condition
def fraction_inequality (x : ℝ) : Prop :=
  (3 * x - 1) / (x - 2) ≤ 0

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 / 3 ≤ x ∧ x < 2

-- The theorem to prove that the inequality's solution matches the given solution set
theorem inequality_solution (x : ℝ) (h : fraction_inequality x) : solution_set x :=
  sorry

end NUMINAMATH_GPT_inequality_solution_l310_31057


namespace NUMINAMATH_GPT_melting_point_of_ice_in_Celsius_l310_31048

theorem melting_point_of_ice_in_Celsius :
  ∀ (boiling_point_F boiling_point_C melting_point_F temperature_C temperature_F : ℤ),
    (boiling_point_F = 212) →
    (boiling_point_C = 100) →
    (melting_point_F = 32) →
    (temperature_C = 60) →
    (temperature_F = 140) →
    (5 * melting_point_F = 9 * 0 + 160) →         -- Using the given equation F = (9/5)C + 32 and C = 0
    melting_point_F = 32 ∧ 0 = 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_melting_point_of_ice_in_Celsius_l310_31048


namespace NUMINAMATH_GPT_delta_minus2_3_eq_minus14_l310_31081

def delta (a b : Int) : Int := a * b^2 + b + 1

theorem delta_minus2_3_eq_minus14 : delta (-2) 3 = -14 :=
by
  sorry

end NUMINAMATH_GPT_delta_minus2_3_eq_minus14_l310_31081


namespace NUMINAMATH_GPT_inequality_satisfaction_l310_31072

theorem inequality_satisfaction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / y + 1 / x + y ≥ y / x + 1 / y + x) ↔ 
  ((x = y) ∨ (x = 1 ∧ y ≠ 0) ∨ (y = 1 ∧ x ≠ 0)) ∧ (x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_satisfaction_l310_31072


namespace NUMINAMATH_GPT_addition_of_decimals_l310_31095

theorem addition_of_decimals :
  0.9 + 0.99 = 1.89 :=
by
  sorry

end NUMINAMATH_GPT_addition_of_decimals_l310_31095


namespace NUMINAMATH_GPT_team_lineup_count_l310_31002

theorem team_lineup_count (total_members specialized_kickers remaining_players : ℕ) 
  (captain_assignments : specialized_kickers = 2) 
  (available_members : total_members = 20) 
  (choose_players : remaining_players = 8) : 
  (2 * (Nat.choose 19 remaining_players)) = 151164 := 
by
  sorry

end NUMINAMATH_GPT_team_lineup_count_l310_31002


namespace NUMINAMATH_GPT_smallest_int_remainder_two_l310_31092

theorem smallest_int_remainder_two (m : ℕ) (hm : m > 1)
  (h3 : m % 3 = 2)
  (h4 : m % 4 = 2)
  (h5 : m % 5 = 2)
  (h6 : m % 6 = 2)
  (h7 : m % 7 = 2) :
  m = 422 :=
sorry

end NUMINAMATH_GPT_smallest_int_remainder_two_l310_31092


namespace NUMINAMATH_GPT_number_multiplies_p_plus_1_l310_31019

theorem number_multiplies_p_plus_1 (p q x : ℕ) 
  (hp : 1 < p) (hq : 1 < q)
  (hEq : x * (p + 1) = 25 * (q + 1))
  (hSum : p + q = 40) :
  x = 325 :=
sorry

end NUMINAMATH_GPT_number_multiplies_p_plus_1_l310_31019


namespace NUMINAMATH_GPT_tailwind_speed_rate_of_change_of_ground_speed_l310_31012

-- Define constants and variables
variables (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ)

-- Define conditions
def conditions := Vg1 = Vp + Vw ∧ altitude = 10000 ∧ Vg1 = 460 ∧
                  Vg2 = Vp - Vw ∧ altitude = 5000 ∧ Vg2 = 310

-- Define theorems to prove
theorem tailwind_speed (Vp Vw : ℝ) (altitude Vg1 Vg2 : ℝ) :
  conditions Vp Vw altitude Vg1 Vg2 → Vw = 75 :=
by
  sorry

theorem rate_of_change_of_ground_speed (altitude1 altitude2 Vg1 Vg2 : ℝ) :
  altitude1 = 10000 → altitude2 = 5000 → Vg1 = 460 → Vg2 = 310 →
  (Vg2 - Vg1) / (altitude2 - altitude1) = 0.03 :=
by
  sorry

end NUMINAMATH_GPT_tailwind_speed_rate_of_change_of_ground_speed_l310_31012


namespace NUMINAMATH_GPT_bags_of_chips_count_l310_31074

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_bags_of_chips_count_l310_31074


namespace NUMINAMATH_GPT_remainder_modulo_l310_31061

theorem remainder_modulo (N k q r : ℤ) (h1 : N = 1423 * k + 215) (h2 : N = 109 * q + r) : 
  (N - q ^ 2) % 109 = 106 := by
  sorry

end NUMINAMATH_GPT_remainder_modulo_l310_31061


namespace NUMINAMATH_GPT_circles_condition_l310_31010

noncomputable def circles_intersect_at (p1 p2 : ℝ × ℝ) (m c : ℝ) : Prop :=
  p1 = (1, 3) ∧ p2 = (m, 1) ∧ (∃ (x y : ℝ), (x - y + c / 2 = 0) ∧ 
    (p1.1 - x)^2 + (p1.2 - y)^2 = (p2.1 - x)^2 + (p2.2 - y)^2)

theorem circles_condition (m c : ℝ) (h : circles_intersect_at (1, 3) (m, 1) m c) : m + c = 3 :=
sorry

end NUMINAMATH_GPT_circles_condition_l310_31010


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l310_31060

variable (a₁ d : ℝ)

def S₄ := 4 * a₁ + 6 * d
def S₅ := 5 * a₁ + 10 * d
def S₆ := 6 * a₁ + 15 * d

theorem sufficient_but_not_necessary_condition (h : d > 1) :
  S₄ a₁ d + S₆ a₁ d > 2 * S₅ a₁ d :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l310_31060


namespace NUMINAMATH_GPT_remaining_sessions_l310_31099

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_remaining_sessions_l310_31099


namespace NUMINAMATH_GPT_cube_volume_in_pyramid_and_cone_l310_31028

noncomputable def volume_of_cube
  (base_side : ℝ)
  (pyramid_height : ℝ)
  (cone_radius : ℝ)
  (cone_height : ℝ)
  (cube_side_length : ℝ) : ℝ := 
  cube_side_length^3

theorem cube_volume_in_pyramid_and_cone :
  let base_side := 2
  let pyramid_height := Real.sqrt 3
  let cone_radius := Real.sqrt 2
  let cone_height := Real.sqrt 3
  let cube_side_length := (Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3)
  volume_of_cube base_side pyramid_height cone_radius cone_height cube_side_length = (6 * Real.sqrt 6) / 17 :=
by sorry

end NUMINAMATH_GPT_cube_volume_in_pyramid_and_cone_l310_31028


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_lower_bound_l310_31014

def f (x a b : ℝ) : ℝ := abs (x - a) + abs (x + b)

-- Part (I)
theorem part_one_solution_set (a b x : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (f x a b ≤ 5) ↔ -3 ≤ x ∧ x ≤ 2 := by
  rw [h1, h2]
  sorry

-- Part (II)
theorem part_two_lower_bound (a b x : ℝ) (h : a > 0) (h' : b > 0) (h'' : a + 4 * b = 2 * a * b) :
  f x a b ≥ 9 / 2 := by
  sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_lower_bound_l310_31014


namespace NUMINAMATH_GPT_discount_rate_l310_31015

variable (P P_b P_s D : ℝ)

-- Conditions
variable (h1 : P_s = 1.24 * P)
variable (h2 : P_s = 1.55 * P_b)
variable (h3 : P_b = P * (1 - D))

theorem discount_rate :
  D = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_l310_31015


namespace NUMINAMATH_GPT_monotonically_decreasing_intervals_max_and_min_values_on_interval_l310_31065

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem monotonically_decreasing_intervals (a : ℝ) : 
  ∀ x : ℝ, (x < -1 ∨ x > 3) → f x a < f (x+1) a :=
sorry

theorem max_and_min_values_on_interval : 
  (f (-1) (-2) = -7) ∧ (max (f (-2) (-2)) (f 2 (-2)) = 20) :=
sorry

end NUMINAMATH_GPT_monotonically_decreasing_intervals_max_and_min_values_on_interval_l310_31065


namespace NUMINAMATH_GPT_change_is_13_82_l310_31054

def sandwich_cost : ℝ := 5
def num_sandwiches : ℕ := 3
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05
def payment : ℝ := 20 + 5 + 3

def total_cost_before_discount : ℝ := num_sandwiches * sandwich_cost
def discount_amount : ℝ := total_cost_before_discount * discount_rate
def discounted_cost : ℝ := total_cost_before_discount - discount_amount
def tax_amount : ℝ := discounted_cost * tax_rate
def total_cost_after_tax : ℝ := discounted_cost + tax_amount

def change (payment total_cost : ℝ) : ℝ := payment - total_cost

theorem change_is_13_82 : change payment total_cost_after_tax = 13.82 := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_change_is_13_82_l310_31054


namespace NUMINAMATH_GPT_circle_equation_center_line_l310_31064

theorem circle_equation_center_line (x y : ℝ) :
  -- Conditions
  (∀ (x1 y1 : ℝ), x1 + y1 - 2 = 0 → (x = 1 ∧ y = 1)) ∧
  ((x - 1)^2 + (y - 1)^2 = 4) ∧
  -- Points A and B
  (∀ (xA yA : ℝ), xA = 1 ∧ yA = -1 ∨ xA = -1 ∧ yA = 1 →
    ((xA - x)^2 + (yA - y)^2 = 4)) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_center_line_l310_31064


namespace NUMINAMATH_GPT_rahul_savings_is_correct_l310_31021

def Rahul_Savings_Problem : Prop :=
  ∃ (NSC PPF : ℝ), 
    (1/3) * NSC = (1/2) * PPF ∧ 
    NSC + PPF = 180000 ∧ 
    PPF = 72000

theorem rahul_savings_is_correct : Rahul_Savings_Problem :=
  sorry

end NUMINAMATH_GPT_rahul_savings_is_correct_l310_31021


namespace NUMINAMATH_GPT_pool_capacity_percentage_l310_31033

noncomputable def hose_rate := 60 -- cubic feet per minute
noncomputable def pool_width := 80 -- feet
noncomputable def pool_length := 150 -- feet
noncomputable def pool_depth := 10 -- feet
noncomputable def drainage_time := 2000 -- minutes
noncomputable def pool_volume := pool_width * pool_length * pool_depth -- cubic feet
noncomputable def removed_water_volume := hose_rate * drainage_time -- cubic feet

theorem pool_capacity_percentage :
  (removed_water_volume / pool_volume) * 100 = 100 :=
by
  -- the proof steps would go here
  sorry

end NUMINAMATH_GPT_pool_capacity_percentage_l310_31033


namespace NUMINAMATH_GPT_value_of_expression_l310_31097

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 3*x^2 + 9*x - 2 = 4 :=
by
  -- The proof will be filled here; it's currently skipped using 'sorry'
  sorry

end NUMINAMATH_GPT_value_of_expression_l310_31097


namespace NUMINAMATH_GPT_units_digit_of_sum_of_squares_2010_odds_l310_31079

noncomputable def sum_units_digit_of_squares (n : ℕ) : ℕ :=
  let units_digits := [1, 9, 5, 9, 1]
  List.foldl (λ acc x => (acc + x) % 10) 0 (List.map (λ i => units_digits.get! (i % 5)) (List.range (2 * n)))

theorem units_digit_of_sum_of_squares_2010_odds : sum_units_digit_of_squares 2010 = 0 := sorry

end NUMINAMATH_GPT_units_digit_of_sum_of_squares_2010_odds_l310_31079


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l310_31085

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 72) : a = 4.5 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l310_31085


namespace NUMINAMATH_GPT_find_value_of_d_l310_31049

theorem find_value_of_d
  (a b c d : ℕ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : ab + bc + ac = abc) 
  (h6 : abc = d) : 
  d = 36 := 
sorry

end NUMINAMATH_GPT_find_value_of_d_l310_31049


namespace NUMINAMATH_GPT_hexagonal_H5_find_a_find_t_find_m_l310_31038

section problem1

-- Define the hexagonal number formula
def hexagonal_number (n : ℕ) : ℕ :=
  2 * n^2 - n

-- Define that H_5 should equal 45
theorem hexagonal_H5 : hexagonal_number 5 = 45 := sorry

end problem1

section problem2

variables (a b c : ℕ)

-- Given hexagonal number equations
def H1 := a + b + c
def H2 := 4 * a + 2 * b + c
def H3 := 9 * a + 3 * b + c

-- Conditions given in problem
axiom H1_def : H1 = 1
axiom H2_def : H2 = 7
axiom H3_def : H3 = 19

-- Prove that a = 3
theorem find_a : a = 3 := sorry

end problem2

section problem3

variables (p q r t : ℕ)

-- Given ratios in problem
axiom ratio1 : p * 3 = 2 * q
axiom ratio2 : q * 5 = 4 * r

-- Prove that t = 12
theorem find_t : t = 12 := sorry

end problem3

section problem4

variables (x y m : ℕ)

-- Given proportional conditions
axiom ratio3 : x * 3 = y * 4
axiom ratio4 : (x + y) * 3 = x * m

-- Prove that m = 7
theorem find_m : m = 7 := sorry

end problem4

end NUMINAMATH_GPT_hexagonal_H5_find_a_find_t_find_m_l310_31038


namespace NUMINAMATH_GPT_chips_recoloring_impossible_l310_31036

theorem chips_recoloring_impossible :
  (∀ a b c : ℕ, a = 2008 ∧ b = 2009 ∧ c = 2010 →
   ¬(∃ k : ℕ, a + b + c = k ∧ (a = k ∨ b = k ∨ c = k))) :=
by sorry

end NUMINAMATH_GPT_chips_recoloring_impossible_l310_31036


namespace NUMINAMATH_GPT_trapezoid_division_areas_l310_31080

open Classical

variable (area_trapezoid : ℝ) (base1 base2 : ℝ)
variable (triangle1 triangle2 triangle3 triangle4 : ℝ)

theorem trapezoid_division_areas 
  (h1 : area_trapezoid = 3) 
  (h2 : base1 = 1) 
  (h3 : base2 = 2) 
  (h4 : triangle1 = 1 / 3)
  (h5 : triangle2 = 2 / 3)
  (h6 : triangle3 = 2 / 3)
  (h7 : triangle4 = 4 / 3) :
  triangle1 + triangle2 + triangle3 + triangle4 = area_trapezoid :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_division_areas_l310_31080


namespace NUMINAMATH_GPT_total_value_of_assets_l310_31050

variable (value_expensive_stock : ℕ)
variable (shares_expensive_stock : ℕ)
variable (shares_other_stock : ℕ)
variable (value_other_stock : ℕ)

theorem total_value_of_assets
    (h1: value_expensive_stock = 78)
    (h2: shares_expensive_stock = 14)
    (h3: shares_other_stock = 26)
    (h4: value_other_stock = value_expensive_stock / 2) :
    shares_expensive_stock * value_expensive_stock + shares_other_stock * value_other_stock = 2106 := by
    sorry

end NUMINAMATH_GPT_total_value_of_assets_l310_31050


namespace NUMINAMATH_GPT_amount_over_budget_l310_31052

-- Define the prices of each item
def cost_necklace_A : ℕ := 34
def cost_necklace_B : ℕ := 42
def cost_necklace_C : ℕ := 50
def cost_first_book := cost_necklace_A + 20
def cost_second_book := cost_necklace_C - 10

-- Define Bob's budget
def budget : ℕ := 100

-- Define the total cost
def total_cost := cost_necklace_A + cost_necklace_B + cost_necklace_C + cost_first_book + cost_second_book

-- Prove the amount over budget
theorem amount_over_budget : total_cost - budget = 120 := by
  sorry

end NUMINAMATH_GPT_amount_over_budget_l310_31052


namespace NUMINAMATH_GPT_max_a_plus_2b_l310_31017

theorem max_a_plus_2b (a b : ℝ) (h : a^2 + 2 * b^2 = 1) : a + 2 * b ≤ Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_max_a_plus_2b_l310_31017


namespace NUMINAMATH_GPT_negation_of_exactly_one_even_l310_31026

variable (a b c : ℕ)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

theorem negation_of_exactly_one_even :
  ¬ exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
                                 (is_even a ∧ is_even b) ∨
                                 (is_even a ∧ is_even c) ∨
                                 (is_even b ∧ is_even c) :=
by sorry

end NUMINAMATH_GPT_negation_of_exactly_one_even_l310_31026


namespace NUMINAMATH_GPT_price_of_A_correct_l310_31086

noncomputable def A_price : ℝ := 25

theorem price_of_A_correct (H1 : 6000 / A_price - 4800 / (1.2 * A_price) = 80) 
                           (H2 : ∀ B_price : ℝ, B_price = 1.2 * A_price) : A_price = 25 := 
by
  sorry

end NUMINAMATH_GPT_price_of_A_correct_l310_31086


namespace NUMINAMATH_GPT_fare_collected_from_I_class_l310_31040

theorem fare_collected_from_I_class (x y : ℝ) 
  (h1 : ∀i, i = x → ∀ii, ii = 4 * x)
  (h2 : ∀f1, f1 = 3 * y)
  (h3 : ∀f2, f2 = y)
  (h4 : x * 3 * y + 4 * x * y = 224000) : 
  x * 3 * y = 96000 :=
by
  sorry

end NUMINAMATH_GPT_fare_collected_from_I_class_l310_31040


namespace NUMINAMATH_GPT_simplify_and_evaluate_l310_31043

noncomputable def simplifyExpression (a : ℚ) : ℚ :=
  (a - 3 + (1 / (a - 1))) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2))

theorem simplify_and_evaluate
  (h : ∀ a, a ∈ [-2, -1, 0, 1, 2]) :
  ∀ a, (a - 1) ≠ 0 → a ≠ 0 → a ≠ 2  →
  simplifyExpression a = a / (a - 1) ∧ simplifyExpression (-1) = 1 / 2 :=
by
  intro a ha_ne_zero ha_ne_two
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l310_31043


namespace NUMINAMATH_GPT_multiply_correct_l310_31035

theorem multiply_correct : 2.4 * 0.2 = 0.48 := by
  sorry

end NUMINAMATH_GPT_multiply_correct_l310_31035


namespace NUMINAMATH_GPT_inner_cube_surface_area_l310_31063

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_inner_cube_surface_area_l310_31063


namespace NUMINAMATH_GPT_x_in_interval_l310_31077

theorem x_in_interval (x : ℝ) (h : x = (1 / x) * (-x) + 2) : 0 < x ∧ x ≤ 2 :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_x_in_interval_l310_31077


namespace NUMINAMATH_GPT_C_increases_with_n_l310_31056

noncomputable def C (e n R r : ℝ) : ℝ := (e * n) / (R + n * r)

theorem C_increases_with_n (e R r : ℝ) (h_e : 0 < e) (h_R : 0 < R) (h_r : 0 < r) :
  ∀ {n₁ n₂ : ℝ}, 0 < n₁ → n₁ < n₂ → C e n₁ R r < C e n₂ R r :=
by
  sorry

end NUMINAMATH_GPT_C_increases_with_n_l310_31056


namespace NUMINAMATH_GPT_polygon_sides_l310_31000

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_sides_l310_31000


namespace NUMINAMATH_GPT_eval_expr_l310_31088

theorem eval_expr : 2 + 3 * 4 - 5 / 5 + 7 = 20 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l310_31088


namespace NUMINAMATH_GPT_sin_eq_cos_example_l310_31024

theorem sin_eq_cos_example 
  (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180)
  (h_eq : Real.sin (n * Real.pi / 180) = Real.cos (682 * Real.pi / 180)) :
  n = 128 :=
sorry

end NUMINAMATH_GPT_sin_eq_cos_example_l310_31024


namespace NUMINAMATH_GPT_rate_of_grapes_l310_31090

theorem rate_of_grapes (G : ℝ) (H : 8 * G + 9 * 50 = 1010) : G = 70 := by
  sorry

end NUMINAMATH_GPT_rate_of_grapes_l310_31090


namespace NUMINAMATH_GPT_r_n_m_smallest_m_for_r_2006_l310_31008

def euler_totient (n : ℕ) : ℕ := 
  n * (1 - (1 / 2)) * (1 - (1 / 17)) * (1 - (1 / 59))

def r (n m : ℕ) : ℕ :=
  m * euler_totient n

theorem r_n_m (n m : ℕ) : r n m = m * euler_totient n := 
  by sorry

theorem smallest_m_for_r_2006 (n m : ℕ) (h : n = 2006) (h2 : r n m = 841 * 928) : 
  ∃ m, r n m = 841^2 := 
  by sorry

end NUMINAMATH_GPT_r_n_m_smallest_m_for_r_2006_l310_31008


namespace NUMINAMATH_GPT_not_possible_perimeter_l310_31041

theorem not_possible_perimeter :
  ∀ (x : ℝ), 13 < x ∧ x < 37 → ¬ (37 + x = 50) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_not_possible_perimeter_l310_31041


namespace NUMINAMATH_GPT_g_inverse_sum_l310_31029

-- Define the function g and its inverse
def g (x : ℝ) : ℝ := x ^ 3
noncomputable def g_inv (y : ℝ) : ℝ := y ^ (1/3 : ℝ)

-- State the theorem to be proved
theorem g_inverse_sum : g_inv 8 + g_inv (-64) = -2 := by 
  sorry

end NUMINAMATH_GPT_g_inverse_sum_l310_31029


namespace NUMINAMATH_GPT_syllogism_example_l310_31073

-- Definitions based on the conditions
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_2 (n : ℕ) := n % 2 = 0

-- Given conditions:
axiom even_implies_divisible_by_2 : ∀ n : ℕ, is_even n → is_divisible_by_2 n
axiom h2012_is_even : is_even 2012

-- Proving the conclusion and the syllogism pattern
theorem syllogism_example : is_divisible_by_2 2012 :=
by
  apply even_implies_divisible_by_2
  apply h2012_is_even

end NUMINAMATH_GPT_syllogism_example_l310_31073


namespace NUMINAMATH_GPT_conditional_probability_l310_31084

/-
We define the probabilities of events A and B.
-/
variables (P : Set (Set α) → ℝ)
variable {α : Type*}

-- Event A: the animal lives up to 20 years old
def A : Set α := {x | true}   -- placeholder definition

-- Event B: the animal lives up to 25 years old
def B : Set α := {x | true}   -- placeholder definition

/-
Given conditions
-/
axiom P_A : P A = 0.8
axiom P_B : P B = 0.4

/-
Proof problem to show P(B | A) = 0.5
-/
theorem conditional_probability : P (B ∩ A) / P A = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_l310_31084


namespace NUMINAMATH_GPT_z_max_plus_z_min_l310_31045

theorem z_max_plus_z_min {x y z : ℝ} 
  (h1 : x^2 + y^2 + z^2 = 3) 
  (h2 : x + 2 * y - 2 * z = 4) : 
  z + z = -4 :=
by 
  sorry

end NUMINAMATH_GPT_z_max_plus_z_min_l310_31045


namespace NUMINAMATH_GPT_total_chocolate_bars_l310_31069

theorem total_chocolate_bars :
  let num_large_boxes := 45
  let num_small_boxes_per_large_box := 36
  let num_chocolate_bars_per_small_box := 72
  num_large_boxes * num_small_boxes_per_large_box * num_chocolate_bars_per_small_box = 116640 :=
by
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_l310_31069


namespace NUMINAMATH_GPT_product_of_prs_l310_31022

theorem product_of_prs
  (p r s : ℕ)
  (H1 : 4 ^ p + 4 ^ 3 = 272)
  (H2 : 3 ^ r + 27 = 54)
  (H3 : 2 ^ (s + 2) + 10 = 42) : 
  p * r * s = 27 :=
sorry

end NUMINAMATH_GPT_product_of_prs_l310_31022


namespace NUMINAMATH_GPT_initial_number_of_children_l310_31076

-- Define the initial conditions
variables {X : ℕ} -- Initial number of children on the bus
variables (got_off got_on children_after : ℕ)
variables (H1 : got_off = 10)
variables (H2 : got_on = 5)
variables (H3 : children_after = 16)

-- Define the theorem to be proved
theorem initial_number_of_children (H : X - got_off + got_on = children_after) : X = 21 :=
by sorry

end NUMINAMATH_GPT_initial_number_of_children_l310_31076


namespace NUMINAMATH_GPT_garden_area_l310_31037

theorem garden_area (w l : ℕ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end NUMINAMATH_GPT_garden_area_l310_31037


namespace NUMINAMATH_GPT_minimum_value_expression_l310_31096

theorem minimum_value_expression 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_eq : 1 / a + 1 / b = 1) : 
  (∃ (x : ℝ), x = (1 / (a-1) + 9 / (b-1)) ∧ x = 6) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l310_31096


namespace NUMINAMATH_GPT_find_a4_l310_31094

noncomputable def S : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * S n + 2^(n+1) - 3

def a : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * a n + 2^n

theorem find_a4 (h1 : ∀ n ≥ 2, S n = 3 * S (n - 1) + 2^n - 3) (h2 : a 1 = -1) : a 4 = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_l310_31094


namespace NUMINAMATH_GPT_range_of_a_minus_abs_b_l310_31023

theorem range_of_a_minus_abs_b (a b : ℝ) (h₁ : 1 < a ∧ a < 3) (h₂ : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_minus_abs_b_l310_31023


namespace NUMINAMATH_GPT_sonny_received_45_boxes_l310_31007

def cookies_received (cookies_given_brother : ℕ) (cookies_given_sister : ℕ) (cookies_given_cousin : ℕ) (cookies_left : ℕ) : ℕ :=
  cookies_given_brother + cookies_given_sister + cookies_given_cousin + cookies_left

theorem sonny_received_45_boxes :
  cookies_received 12 9 7 17 = 45 :=
by
  sorry

end NUMINAMATH_GPT_sonny_received_45_boxes_l310_31007


namespace NUMINAMATH_GPT_range_of_a_l310_31075

noncomputable def g (x : ℝ) : ℝ := abs (x-1) - abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (g x ≥ a^2 + a + 1)) ↔ (a < -1 ∨ a > 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l310_31075
