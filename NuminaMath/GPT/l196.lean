import Complex
import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Functions
import Mathlib.Algebra.Order
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.IVT
import Mathlib.Analysis.Convex.Basic
import Mathlib.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Statistics.Regression
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace incorrect_statements_l196_196588

def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n + d = a (n + 1)

def is_geometric_seq (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a n * q = a (n + 1)

noncomputable def sum_of_seq (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range(n + 1), a i

theorem incorrect_statements {a : ℕ → ℤ} :
  (∀ a, is_arithmetic_seq a → ¬is_arithmetic_seq (λ n, abs (a n))) ∧
  (∀ q a, is_geometric_seq a → q = 1 → ¬is_geometric_seq (λ n, a (n + 1) - a n)) ∧
  (∀ q a, is_geometric_seq a → q = -1 → ¬is_geometric_seq (λ k, sum_of_seq a k - sum_of_seq a (k - 1) if k > 0 else sum_of_seq a k)) :=
by
  sorry

end incorrect_statements_l196_196588


namespace correct_number_of_statements_l196_196373

-- Definitions based on conditions
def point_on_x_axis (x y z : ℝ) : Prop := (x ≠ 0) ∧ (y = 0) ∧ (z = 0)
def point_on_yOz_plane (x y z : ℝ) : Prop := (x = 0)
def point_on_z_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0)
def point_on_xOz_plane (x y z : ℝ) : Prop := (y = 0)

-- Correctness of statements
def statement_1_correct : Prop := ¬point_on_x_axis 0 b c
def statement_2_correct : Prop := point_on_yOz_plane 0 b c
def statement_3_correct : Prop := point_on_z_axis 0 0 c
def statement_4_correct : Prop := point_on_xOz_plane a 0 c

-- Proving the correct number of statements
theorem correct_number_of_statements (a b c : ℝ) :
  (statement_1_correct = ff ∧ statement_2_correct = tt ∧ statement_3_correct = tt ∧
   statement_4_correct = tt → nat.of_bool statement_1_correct +
                                     nat.of_bool statement_2_correct +
                                     nat.of_bool statement_3_correct +
                                     nat.of_bool statement_4_correct = 3) := sorry

end correct_number_of_statements_l196_196373


namespace sqrt_of_16_is_4_l196_196406

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l196_196406


namespace j_nonzero_l196_196639

noncomputable def Q (x : ℝ) (f g h i j : ℝ) : ℝ :=
  x^6 + f * x^5 + g * x^4 + h * x^3 + i * x^2 + j * x

-- Polynomial with six distinct roots, two of which are 0 and 1
def polynomial_has_six_distinct_roots (Q : ℝ → ℝ) (f g h i j : ℝ) : Prop :=
  ∃ m n o p : ℝ, m ≠ n ∧ n ≠ o ∧ o ≠ p ∧ p ≠ m ∧ m ≠ 1 ∧ n ≠ 1 ∧ o ≠ 1 ∧ p ≠ 1 ∧
  Q(x) = x^2 * (x - 1) * (x - m) * (x - n) * (x - o) * (x - p)

-- Prove that coefficient j cannot be zero
theorem j_nonzero (f g h i j : ℝ) :
  polynomial_has_six_distinct_roots (Q f g h i j) f g h i j → j ≠ 0 :=
by { sorry }

end j_nonzero_l196_196639


namespace base9_minus_base6_to_decimal_l196_196840

theorem base9_minus_base6_to_decimal :
  let b9 := 3 * 9^2 + 2 * 9^1 + 1 * 9^0
  let b6 := 2 * 6^2 + 5 * 6^1 + 4 * 6^0
  b9 - b6 = 156 := by
sorry

end base9_minus_base6_to_decimal_l196_196840


namespace proof_problem_l196_196966

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l196_196966


namespace union_sets_l196_196045

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 6}

theorem union_sets : A ∪ B = {1, 2, 4, 6} := by
  sorry

end union_sets_l196_196045


namespace part1_part2_l196_196501

noncomputable def f (x a : ℝ) := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

noncomputable def g (x a : ℝ) := f x a + Real.log (x + 1) + 1/2 * x

theorem part1 (a : ℝ) (x : ℝ) (h : x > 0) : 
  (a ≤ 2 → ∀ x, g x a > 0) ∧ 
  (a > 2 → ∀ x, x < Real.exp (a - 2) - 1 → g x a < 0) ∧
  (a > 2 → ∀ x, x > Real.exp (a - 2) - 1 → g x a > 0) :=
sorry

theorem part2 (a : ℤ) : 
  (∃ x ≥ 0, f x a < 0) → a ≥ 3 :=
sorry

end part1_part2_l196_196501


namespace find_angle_B_find_dot_product_AB_BC_l196_196049

variable {a b c : ℝ}
variable {C : ℝ}
variable (A B : ℝ) -- Angles opposite to sides a, b, respectively

-- Question 1
theorem find_angle_B (h1 : sqrt 3 * cos C + sin C = sqrt 3 * a / b) :
  B = π / 3 :=
sorry

-- Question 2
theorem find_dot_product_AB_BC (h1 : sqrt 3 * cos C + sin C = sqrt 3 * a / b)
  (h2 : a + c = 5 * sqrt 7) (h3 : b = 7) (h4 : B = π / 3) :
  (a * c) * (-cos B) = -21 :=
sorry

end find_angle_B_find_dot_product_AB_BC_l196_196049


namespace incorrect_statements_l196_196321

-- Definitions for the points
def A := (-2, -3) 
def P := (1, 1)
def pt := (1, 3)

-- Definitions for the equations in the statements
def equationA (x y : ℝ) := x + y + 5 = 0
def equationB (m x y : ℝ) := 2*(m+1)*x + (m-3)*y + 7 - 5*m = 0
def equationC (θ x y : ℝ) := y - 1 = Real.tan θ * (x - 1)
def equationD (x₁ y₁ x₂ y₂ x y : ℝ) := (x₂ - x₁)*(y - y₁) = (y₂ - y₁)*(x - x₁)

-- Points of interest
def xA : ℝ := -2
def yA : ℝ := -3
def xP : ℝ := 1
def yP : ℝ := 1
def pt_x : ℝ := 1
def pt_y : ℝ := 3

-- Main proof to show which statements are incorrect
theorem incorrect_statements :
  ¬ equationA xA yA ∨ ¬ (∀ m, equationB m pt_x pt_y) ∨ (θ = (Real.pi / 2) → ¬ equationC θ xP yP) ∨
  ∀ x₁ y₁ x₂ y₂ x y, equationD x₁ y₁ x₂ y₂ x y :=
by {
  sorry
}

end incorrect_statements_l196_196321


namespace angle_BPC_is_105_l196_196990

-- Given conditions as definitions
def side_length : ℝ := 6
def ABE_equilateral : Prop := ∀ A B E : ℝ, A = B ∧ B = E
def intersection_point (A C B E P : ℝ) : Prop := line_through A C ∧ line_through B E ∧ intersect A C B E = P
def Q_on_BC (Q B C x P : ℝ) : Prop := collinear Q B C ∧ perpendicular PQ BC ∧ PQ = x

-- Target angle to prove
def measure_angle_BPC (B P C : ℝ) : Prop := ∠BPC = 105

-- Statement of the goal:
theorem angle_BPC_is_105 (A B C D E P Q : ℝ) (P_intersect: intersection_point A C B E P) (Q_on_BC : Q_on_BC Q B C PQ P): 
  measure_angle_BPC B P C :=
by
  sorry

end angle_BPC_is_105_l196_196990


namespace ratio_of_coefficients_rational_terms_largest_coefficient_term_sum_expression_l196_196490

theorem ratio_of_coefficients (n : ℕ) (x : ℚ) : 
  (C(n, 4) * (-2)^4) / (C(n, 2) * (-2)^2) = 56 / 3 := sorry

noncomputable def general_term (n r : ℕ) (x : ℚ) : ℚ := 
  C(n, r) * (-2)^r * x^(5 - 5 * r / 6)

theorem rational_terms (x : ℚ) : 
  (general_term 10 0 x = x^5) ∧ (general_term 10 6 x = 13440) := sorry

theorem largest_coefficient_term (x : ℚ) : 
  general_term 10 7 x = -15360 * x^(-5 / 6) := sorry

theorem sum_expression : 
  10 + 9 * C(10, 2) + 81 * C(10, 3) + ∑ i in {4..10}, 9^(i-1) * C(10, i) = (10^10 - 1) / 9 := sorry

end ratio_of_coefficients_rational_terms_largest_coefficient_term_sum_expression_l196_196490


namespace exist_2022_good_numbers_with_good_sum_l196_196575

def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

theorem exist_2022_good_numbers_with_good_sum :
  ∃ (a : Fin 2022 → ℕ), (∀ i j : Fin 2022, i ≠ j → a i ≠ a j) ∧ (∀ i : Fin 2022, is_good (a i)) ∧ is_good (Finset.univ.sum a) :=
sorry

end exist_2022_good_numbers_with_good_sum_l196_196575


namespace sphere_surface_area_ratios_l196_196778

theorem sphere_surface_area_ratios
  (s : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (r3 : ℝ)
  (h1 : r1 = s / 4 * Real.sqrt 6)
  (h2 : r2 = s / 4 * Real.sqrt 2)
  (h3 : r3 = s / 12 * Real.sqrt 6) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r3^2) = 9 ∧
  (4 * Real.pi * r2^2) / (4 * Real.pi * r3^2) = 3 ∧
  (4 * Real.pi * r3^2) / (4 * Real.pi * r3^2) = 1 := 
by
  sorry

end sphere_surface_area_ratios_l196_196778


namespace is_parallel_ID_AC_l196_196129

noncomputable def is_isosceles (A B C : Point) : Prop := AC = BC
noncomputable def is_circumcenter (O A B C : Point) : Prop := 
  dist O A = dist O B ∧ dist O B = dist O C
noncomputable def is_incenter (I A B C : Point) : Prop := 
  dist I (side AB) = dist I (side BC) = dist I (side CA)
noncomputable def perpendicular (l1 l2 : Line) : Prop := 
  angle_between l1 l2 = π / 2

/- Lean statement corresponding to the proof problem -/
theorem is_parallel_ID_AC {A B C O I D : Point} 
  (h1 : is_isosceles A B C)
  (h2 : is_circumcenter O A B C)
  (h3 : is_incenter I A B C)
  (h4 : D ∈ side BC)
  (h5 : perpendicular (line_through O D) (line_through B I)) :
  parallel (line_through I D) (line_through A C) :=
sorry

end is_parallel_ID_AC_l196_196129


namespace g_m_form_l196_196175

-- Define the function f(n)
def f (n : ℕ) : ℕ :=
(n.toNatDigits 2).count 1

-- Define the function g(m)
def g (m : ℕ) : ℤ :=
∑ k in Finset.range (2^m), (-1) ^ (f k) * k^m

-- Define the function h(n) as in the problem
def h (n : ℕ) : ℤ :=
(-1) ^ f n

theorem g_m_form (m : ℕ) :
  ∃ a : ℤ, ∃ p q : Polynomial ℤ, g m = (-1)^m * a * p.eval m * (q.eval m)! :=
sorry

end g_m_form_l196_196175


namespace total_weight_of_apples_l196_196188

/-- Define the weight of an apple and an orange -/
def apple_weight := 4
def orange_weight := 3

/-- Define the maximum weight a bag can hold -/
def max_bag_weight := 49

/-- Define the number of bags Marta buys -/
def num_bags := 3

/-- Prove the total weight of apples Marta should buy -/
theorem total_weight_of_apples : 
    ∀ (A : ℕ), 4 * A + 3 * A ≤ 49 → A = 7 → 4 * A * 3 = 84 :=
by 
    intros A h1 h2
    rw [h2]
    norm_num 
    sorry

end total_weight_of_apples_l196_196188


namespace fraction_area_above_line_l196_196361

/-- Define the square as a set of four vertices -/
structure Point where
  x : ℝ
  y : ℝ

def square_vertices : List Point :=
  [ { x := 4, y := 1 },
    { x := 7, y := 1 },
    { x := 7, y := 4 },
    { x := 4, y := 4 }]

/-- Define the line passing through the points (4,3) and (7,1) -/
def line_through_points (p1 p2 : Point) : ℝ → ℝ :=
  λ x => ((p2.y - p1.y) / (p2.x - p1.x)) * (x - p1.x) + p1.y

def line : ℝ → ℝ := line_through_points { x := 4, y := 3 } { x := 7, y := 1 }

/-- Proof that the fraction of the area of the square that lies above the line is 1/2 -/
theorem fraction_area_above_line : 
  let A_square := 3 ^ 2 in
  let A_triangle := (1 / 2) * 3 * 3 in
  A_triangle / A_square = 1 / 2 := by
  let A_square := 3 ^ 2
  let A_triangle := (1 / 2) * 3 * 3
  show A_triangle / A_square = 1 / 2
  sorry

end fraction_area_above_line_l196_196361


namespace matrix_pow_101_eq_B_l196_196927

open Matrix

variables {α : Type*} [DecidableEq α] [Fintype α] {n : ℕ} [DecidableEq α] [Fintype n]

/-- Define matrix B -/
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [0, 0, 1],
    [0, 0, 0],
    [-1, 0, 0]
  ]

/-- Prove that B^101 = B -/
theorem matrix_pow_101_eq_B : B ^ 101 = B := by
  sorry

end matrix_pow_101_eq_B_l196_196927


namespace sum_of_ages_l196_196159

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l196_196159


namespace find_intersection_l196_196884

noncomputable def f (n : ℕ) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def f_set (s : Set ℕ) : Set ℕ := {n | f n ∈ s}

theorem find_intersection : f_set A ∩ f_set B = {1, 2} := 
by {
  sorry
}

end find_intersection_l196_196884


namespace fixed_point_P_exists_l196_196508

open_locale classical
noncomputable theory

variables {O1 O2 A P M1 M2 : Type*}
variables [metric_space O1] [metric_space O2] [metric_space A] [metric_space P] [metric_space M1] [metric_space M2]
variables (r1 r2 : ℝ)
variables (O1_radius : ∀ x, dist O1 x = r1)
variables (O2_radius : ∀ x, dist O2 x = r2)
variables (same_direction : ∀ x y, x ≠ y → x ∈ O1 → y ∈ O2 → true) -- Placeholder for direction consistency
variables (full_revolution : ∀ x, x ∈ O1 → dist x A = r1 → dist x M1 = r1)
variables (simultaneous_return : ∀ x y, x ∈ O1 → y ∈ O2 → dist x A = r1 → dist y A = r2 → dist x y = 0)

-- Main theorem to prove
theorem fixed_point_P_exists :
  ∃ (P : O1), ∀ (M1 M2 : O1), (dist P M1 = dist P M2) :=
sorry

end fixed_point_P_exists_l196_196508


namespace perpendicular_line_slope_l196_196530

theorem perpendicular_line_slope (k : ℝ) :
  (∀ x : ℝ, y = 3 * x) → (∀ x : ℝ, y = k * x - 2) → k = -1 / 3 :=
by
  assume h1 h2,
  sorry

end perpendicular_line_slope_l196_196530


namespace value_of_x2_inv2_l196_196018

variable {x : ℝ}

theorem value_of_x2_inv2 (h : x + x⁻¹ = 2) : x^2 + x⁻² = 2 := by
  sorry

end value_of_x2_inv2_l196_196018


namespace non_powers_of_a_meet_condition_l196_196470

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end non_powers_of_a_meet_condition_l196_196470


namespace least_n_exceeds_million_l196_196075

noncomputable def product_terms (n : ℕ) : ℝ :=
  ∏ k in finset.range (n + 1), 20 ^ (k / 11 : ℝ)

theorem least_n_exceeds_million :
  ∀ n : ℕ, product_terms n > 1000000 → n ≥ 12 :=
begin
  sorry,
end

end least_n_exceeds_million_l196_196075


namespace range_of_m_l196_196962

variable (x m : ℝ)
hypothesis : (x + m) / (x - 2) + (2 * m) / (2 - x) = 3
hypothesis_pos : 0 < x

theorem range_of_m :
  m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l196_196962


namespace find_natural_numbers_l196_196844

theorem find_natural_numbers (n : ℕ) :
  (∀ (a : Fin (n + 2) → ℝ), (a (Fin.last _) * (a (Fin.last _))
   - 2 * (a (Fin.last _)) * Real.sqrt (Finset.univ.sum (λ i, (a i) ^ 2)) 
   + (Finset.univ.erase (Fin.last _)).sum (λ i, a i) = 0) → 
   (a (Fin.last _) ≠ 0) → 
   ∃ x : ℝ, (a (Fin.last _) * x^2
   - 2 * x * Real.sqrt (Finset.univ.sum (λ i, (a i) ^ 2))
   + (Finset.univ.erase (Fin.last _)).sum (λ i, a i) = 0)) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end find_natural_numbers_l196_196844


namespace N_is_composite_l196_196808

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  have h_mod : N % 2027 = 0 := 
    sorry
  intro h_prime
  have h_div : 2027 ∣ N := by
    rw [Nat.dvd_iff_mod_eq_zero, h_mod]
  exact Nat.Prime.not_dvd_one h_prime h_div

end N_is_composite_l196_196808


namespace find_b_and_extreme_value_l196_196925

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x - 10) / (x - 4)

theorem find_b_and_extreme_value :
  (∀ x : ℝ, x ≠ 0 → x ≠ 4 → (derivative (λ x, x^2 + b * Real.log x) x = derivative (λ x, (x - 10) / (x - 4)) x) ∧ x = 5) →
  b = -20 ∧ ∃ min_x : ℝ, min_x = Real.sqrt 10 ∧ f (Real.sqrt 10) (-20) = 10 - 10 * Real.log 10 := by
  intros h
  sorry

end find_b_and_extreme_value_l196_196925


namespace intersection_of_complements_l196_196182

open Set

variable (U A B : Set Nat)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 4, 5})
variable (hB : B = {2, 4, 6, 8})

theorem intersection_of_complements :
  A ∩ (U \ B) = {3, 5} :=
by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l196_196182


namespace simplify_and_evaluate_result_l196_196615

noncomputable def simplify_and_evaluate (a : ℚ) : ℚ :=
  (1 / a) + Real.sqrt((1 / a^2) + a^2 - 2)

theorem simplify_and_evaluate_result : simplify_and_evaluate (1/5) = 49 / 5 := by
  sorry

end simplify_and_evaluate_result_l196_196615


namespace sum_of_ages_l196_196157

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l196_196157


namespace hexagon_largest_angle_l196_196125

theorem hexagon_largest_angle
  (A B : ℝ)
  (C : ℝ)
  (D : ℝ := C)
  (E : ℝ := 2 * C + 20)
  (F : ℝ := 720 - (A + B + C + C + E))
  (sum_angles : A + B + C + D + E + F = 720)
  (hA : A = 60)
  (hB : B = 95) :
  ∃ (M : ℝ), M = 292.5 ∧ (M = E ∨ M = F) :=
by
  use 292.5
  split
  . rfl
  . left
    sorry

end hexagon_largest_angle_l196_196125


namespace increasing_exponential_is_necessary_condition_l196_196336

variable {a : ℝ}

theorem increasing_exponential_is_necessary_condition (h : ∀ x y : ℝ, x < y → a ^ x < a ^ y) :
    (a > 1) ∧ (¬ (a > 2 → a > 1)) :=
by
  sorry

end increasing_exponential_is_necessary_condition_l196_196336


namespace angle_BPC_is_105_l196_196991

-- Given conditions as definitions
def side_length : ℝ := 6
def ABE_equilateral : Prop := ∀ A B E : ℝ, A = B ∧ B = E
def intersection_point (A C B E P : ℝ) : Prop := line_through A C ∧ line_through B E ∧ intersect A C B E = P
def Q_on_BC (Q B C x P : ℝ) : Prop := collinear Q B C ∧ perpendicular PQ BC ∧ PQ = x

-- Target angle to prove
def measure_angle_BPC (B P C : ℝ) : Prop := ∠BPC = 105

-- Statement of the goal:
theorem angle_BPC_is_105 (A B C D E P Q : ℝ) (P_intersect: intersection_point A C B E P) (Q_on_BC : Q_on_BC Q B C PQ P): 
  measure_angle_BPC B P C :=
by
  sorry

end angle_BPC_is_105_l196_196991


namespace single_bill_value_l196_196779

theorem single_bill_value 
  (total_amount : ℕ) 
  (num_5_dollar_bills : ℕ) 
  (amount_5_dollar_bills : ℕ) 
  (single_bill : ℕ) : 
  total_amount = 45 → 
  num_5_dollar_bills = 7 → 
  amount_5_dollar_bills = 5 → 
  total_amount = num_5_dollar_bills * amount_5_dollar_bills + single_bill → 
  single_bill = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end single_bill_value_l196_196779


namespace find_value_of_b_l196_196064

theorem find_value_of_b (a b : ℕ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 :=
sorry

end find_value_of_b_l196_196064


namespace derivative_eq_limit_l196_196050

variable {f : ℝ → ℝ}
variable {x : ℝ}

theorem derivative_eq_limit (h : deriv f x = 3) : 
  (tendsto (λ Δx : ℝ, (f (x + Δx) - f x) / Δx) (𝓝 0) (𝓝 3)) :=
sorry

end derivative_eq_limit_l196_196050


namespace min_value_s2_minus_t2_l196_196910

noncomputable def s (x y z : ℝ) : ℝ := real.sqrt (x + 2) + real.sqrt (y + 5) + real.sqrt (z + 10)
noncomputable def t (x y z : ℝ) : ℝ := real.sqrt (x + 1) + real.sqrt (y + 1) + real.sqrt (z + 1)

theorem min_value_s2_minus_t2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (s x y z) ^ 2 - (t x y z) ^ 2 ≥ 36 :=
sorry

end min_value_s2_minus_t2_l196_196910


namespace correct_statement_l196_196083

variables {Line Plane : Type}
variable (a b c : Line)
variable (M N : Plane)

/- Definitions for the conditions -/
def lies_on_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

/- Conditions -/
axiom h1 : lies_on_plane a M
axiom h2 : lies_on_plane b N
axiom h3 : intersection M N = c

/- The correct statement to be proved -/
theorem correct_statement : parallel a b → parallel a c :=
by sorry

end correct_statement_l196_196083


namespace number_of_pairs_exterior_angles_l196_196940

theorem number_of_pairs_exterior_angles (m n : ℕ) :
  (3 ≤ m ∧ 3 ≤ n ∧ 360 = m * n) ↔ 20 = 20 := 
by sorry

end number_of_pairs_exterior_angles_l196_196940


namespace probability_final_roll_six_l196_196736

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l196_196736


namespace product_of_roots_l196_196805

theorem product_of_roots (a b c : ℝ) (h_eq : 24 * a^2 + 36 * a - 648 = 0) : a * c = -27 := 
by
  have h_root_product : (24 * a^2 + 36 * a - 648) = 0 ↔ a = -27 := sorry
  exact sorry

end product_of_roots_l196_196805


namespace age_of_B_l196_196752

noncomputable def A : ℕ := 2 + B
noncomputable def B : ℕ := 2 * C
noncomputable def C : ℕ := 4  -- Derived from solving the equation

theorem age_of_B : B = 8 :=
by
  have h1 : A = 2 + B := by sorry
  have h2 : B = 2 * C := by sorry
  have h3 : A + B + C = 22 := by sorry
  sorry

end age_of_B_l196_196752


namespace hyperbola_triangle_perimeter_l196_196073

/-
Given the hyperbola \(C: \frac{x^2}{3} - y^2 = 1\) with its left and right foci denoted as \(F_1\) and \(F_2\) respectively.
A line passing through point \(F_2\) intersects the right branch of the hyperbola \(C\) at points \(P\) and \(Q\),
and the x-coordinate of point \(P\) is \(2\).
Prove that the perimeter of triangle \(\triangle PF_1Q\) is \(\frac{16\sqrt{3}}{3}\).
-/

def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

def foci_positions (F1 F2 : ℝ × ℝ) : Prop := F1 = (-2, 0) ∧ F2 = (2, 0)

noncomputable def perimeter_triangle (P F1 Q : ℝ × ℝ) : ℝ :=
  dist P F1 + dist Q F1 + dist P Q

theorem hyperbola_triangle_perimeter :
  ∀ (F1 F2 P Q : ℝ × ℝ),
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧ foci_positions F1 F2 ∧
  P.1 = 2 ∧ P.2 = Q.2 ∧ Q.1 = 2 ∧
  ∃ c, P = (2, c) ∧ Q = (2, -c) →
  perimeter_triangle P F1 Q = 16 * real.sqrt 3 / 3 :=
by sorry

end hyperbola_triangle_perimeter_l196_196073


namespace RX_XQ_ratio_l196_196533

variables (P Q R X Y Z : Type)
variables [InsideTriangle P Q R X] [OnSegment QR X] [OnSegment PR Y] [Intersect PX QY Z]
variables (PZ ZX : ℝ) (QZ ZY : ℝ)

def ratio_PZ_ZX : Prop := PZ / ZX = 5
def ratio_QZ_ZY : Prop := QZ / ZY = 3
def ratio_RX_XQ : Prop := RX / XQ = 5 / 19

theorem RX_XQ_ratio :
  ratio_PZ_ZX PZ ZX →
  ratio_QZ_ZY QZ ZY →
  ratio_RX_XQ :=
by
  intros hPZ hQZ
  sorry

end RX_XQ_ratio_l196_196533


namespace mia_spent_per_parent_l196_196191

theorem mia_spent_per_parent (amount_sibling : ℕ) (num_siblings : ℕ) (total_spent : ℕ) 
  (num_parents : ℕ) : 
  amount_sibling = 30 → num_siblings = 3 → total_spent = 150 → num_parents = 2 → 
  (total_spent - num_siblings * amount_sibling) / num_parents = 30 :=
by
  sorry

end mia_spent_per_parent_l196_196191


namespace average_weight_correct_l196_196261

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l196_196261


namespace calculate_3_nabla_neg4_l196_196019

/-- Definition of the operation ∇ -/
def nabla (x y : ℝ) : ℝ := (x + y) / (1 + x * y)

theorem calculate_3_nabla_neg4 :
  nabla 3 (-4) = 1 / 11 :=
by
  sorry

end calculate_3_nabla_neg4_l196_196019


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196285

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196285


namespace determine_a_l196_196931

variable {a : ℝ}

def M : Set ℝ := {0, 1, a + 1}

theorem determine_a (h : -1 ∈ M) : a = -2 := by
  sorry

end determine_a_l196_196931


namespace area_of_inscribed_rectangle_l196_196613

theorem area_of_inscribed_rectangle 
    (DA : ℝ) 
    (GD HD : ℝ) 
    (rectangle_inscribed : ∀ (A B C D G H : Type), true) 
    (radius : ℝ) 
    (GH : ℝ):
    DA = 20 ∧ GD = 5 ∧ HD = 5 ∧ GH = GD + DA + HD ∧ radius = GH / 2 → 
    200 * Real.sqrt 2 = DA * (Real.sqrt (radius^2 - (GD^2))) :=
by
  sorry

end area_of_inscribed_rectangle_l196_196613


namespace simplify_expression_l196_196210

theorem simplify_expression (a b : ℤ) : 
  (18 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 40 * b) = 21 * a + 41 * b := 
by
  sorry

end simplify_expression_l196_196210


namespace average_weight_men_women_l196_196266

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l196_196266


namespace negation_statement_l196_196644

-- Define the initial statement
def mom_loves_me : Prop := ∃ x : String, (x = "Mom" ∧ loves x "me")

-- Define the negation of a conditional statement
def negation (P Q : Prop) (h : P) : Prop := ∀ x : String, (¬ (x = "Mom") → ¬ (loves x "me"))

-- The problem statement as a theorem to be proved
theorem negation_statement :
  (mom_loves_me → ∃ x, (¬ (x = "Mom") → ¬ (loves x "me"))) :=
sorry

end negation_statement_l196_196644


namespace max_value_f_l196_196181

def f (x a : ℝ) : ℝ := - (1 / 3) * x^3 + (1 / 2) * x^2 + 2 * a * x

theorem max_value_f (a : ℝ) (h1 : 0 < a) (h2 : a < 2)
  (hmin : ∀ x ∈ set.Icc (1 : ℝ) 4, ∀ y ∈ set.Icc (1 : ℝ) 4, 
    f y a = -16 / 3 → f x a ≥ f y a):
  ∃ x ∈ set.Icc (1 : ℝ) 4, f x a = 10 / 3 :=
by
  sorry

end max_value_f_l196_196181


namespace num_zeros_in_fraction_decimal_l196_196098

theorem num_zeros_in_fraction_decimal :
  (let x := (1 : ℚ) / (2^3 * 5^6) in
   ∃ k : ℕ, x = 8 / 10^6 ∧ k = 5) :=
by
  -- The proof would involve showing x can be written as '0.000008',
  -- which has exactly 5 zeros between the decimal point and the first non-zero digit.
  sorry

end num_zeros_in_fraction_decimal_l196_196098


namespace sqrt_inequality_neg_l196_196413

theorem sqrt_inequality_neg {x y : ℝ} (h : x > y) : -real.sqrt x < -real.sqrt y :=
by sorry

lemma problem_sqrt_13_neg_lt_neg_3 : -real.sqrt 13 < -3 :=
by {
  have h : 13 > 9 := by norm_num,
  exact sqrt_inequality_neg h,
  sorry
}

end sqrt_inequality_neg_l196_196413


namespace tank_fills_in_56_minutes_l196_196197

theorem tank_fills_in_56_minutes : 
  (∃ A B C : ℕ, (A = 40 ∧ B = 30 ∧ C = 20) ∧ 
                 ∃ capacity : ℕ, capacity = 950 ∧ 
                 ∃ time : ℕ, time = 56 ∧
                 ∀ cycle_time : ℕ, cycle_time = 3 ∧ 
                 ∀ net_water_per_cycle : ℕ, net_water_per_cycle = A + B - C ∧
                 ∀ total_cycles : ℕ, total_cycles = capacity / net_water_per_cycle ∧
                 ∀ total_time : ℕ, total_time = total_cycles * cycle_time - 1 ∧
                 total_time = time) :=
sorry

end tank_fills_in_56_minutes_l196_196197


namespace line_intersects_circle_l196_196890

variable (x0 y0 R : ℝ)

theorem line_intersects_circle (h : x0^2 + y0^2 > R^2) :
  ∃ (x y : ℝ), (x^2 + y^2 = R^2) ∧ (x0 * x + y0 * y = R^2) :=
sorry

end line_intersects_circle_l196_196890


namespace definite_integral_solution_l196_196386

noncomputable def integral_problem : ℝ := 
  by 
    sorry

theorem definite_integral_solution :
  integral_problem = (1/6 : ℝ) + Real.log 2 - Real.log 3 := 
by
  sorry

end definite_integral_solution_l196_196386


namespace positional_relationship_l196_196984

variable {α : Type*} [normed_field α]

/-- Definition of coplanar lines -/
def coplanar (a b : set (euclidean_space α 3)) : Prop :=
∃ (P : euclidean_space α 3), ∃ (N : euclidean_space α 3), 
∀ (x ∈ a) (y ∈ b), (x - P) • N = 0 ∧ (y - P) • N = 0

/-- Definition of parallel lines -/
def parallel (a b : set (euclidean_space α 3)) : Prop :=
coplanar a b ∧ ∃ (v : euclidean_space α 3), ∀ (x ∈ a) (y ∈ b), (x - y) • v = 0

/-- Definition of skew lines -/
def skew (a b : set (euclidean_space α 3)) : Prop :=
¬ coplanar a b

/-- Main theorem: proving that lines with no common points are either parallel or skew. -/
theorem positional_relationship (a b : set (euclidean_space α 3)) (h : ∀ (x ∈ a), x ∉ b) : 
  parallel a b ∨ skew a b :=
by sorry

end positional_relationship_l196_196984


namespace maggie_earnings_l196_196590

def subscriptions_to_parents := 4
def subscriptions_to_grandfather := 1
def subscriptions_to_next_door_neighbor := 2
def subscriptions_to_another_neighbor := 2 * subscriptions_to_next_door_neighbor
def subscription_rate := 5

theorem maggie_earnings : 
  (subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door_neighbor + subscriptions_to_another_neighbor) * subscription_rate = 55 := 
by
  sorry

end maggie_earnings_l196_196590


namespace average_time_correct_l196_196332

-- Define the times for each runner
def y_time : ℕ := 58
def z_time : ℕ := 26
def w_time : ℕ := 2 * z_time

-- Define the number of runners
def num_runners : ℕ := 3

-- Calculate the summed time of all runners
def total_time : ℕ := y_time + z_time + w_time

-- Calculate the average time
def average_time : ℚ := total_time / num_runners

-- Statement of the proof problem
theorem average_time_correct : average_time = 45.33 := by
  -- The proof would go here
  sorry

end average_time_correct_l196_196332


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196298

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196298


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196290

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196290


namespace correct_formula_for_xy_l196_196031

theorem correct_formula_for_xy :
  (∀ x y, (x = 1 ∧ y = 5) ∨ (x = 2 ∧ y = 11) ∨ (x = 3 ∧ y = 19) ∨ 
  (x = 4 ∧ y = 29) ∨ (x = 5 ∧ y = 41) →
  y = x^2 + 3*x + 1) :=
by
  intro x y h
  cases h with h₁ h
  { rw [h₁.1, h₁.2], simp }
  cases h with h₂ h
  { rw [h₂.1, h₂.2], simp }
  cases h with h₃ h
  { rw [h₃.1, h₃.2], simp }
  cases h with h₄ h
  { rw [h₄.1, h₄.2], simp }
  { rw [h.1, h.2], simp }
  sorry

end correct_formula_for_xy_l196_196031


namespace find_unit_vector_l196_196005

theorem find_unit_vector (a b : ℝ) : 
  a^2 + b^2 = 1 ∧ 3 * a + 4 * b = 0 →
  (a = 4 / 5 ∧ b = -3 / 5) ∨ (a = -4 / 5 ∧ b = 3 / 5) :=
by sorry

end find_unit_vector_l196_196005


namespace solve_professions_l196_196160

-- Define people
inductive Person
| Kondratyev
| Davydov
| Fedorov

open Person

-- Define professions
inductive Profession
| Carpenter
| Painter
| Plumber

open Profession

-- Define older relationship
inductive Older
| older : Person → Person → Prop

open Older

-- Define knowledge relationship
inductive Know
| know : Person → Person → Prop

open Know

-- Conditions
def conditions : Prop :=
  Kondratyev ≠ Davydov ∧ Davydov ≠ Fedorov ∧ Fedorov ≠ Kondratyev ∧
  (∀ p : Profession, p ≠ Carpenter → p ≠ Plumber) ∧
  (∃ p1 p2 : Person, (Older.older p1 p2) ∧ (∀ p3, p3 ≠ painter → p3 ≠ plumber)) ∧
  Older.older Davydov Kondratyev ∧
  ¬Know.know Fedorov Davydov

-- Final answer
def final_assignment : Prop :=
  ∃ (Kondratyev_profession Davydov_profession Fedorov_profession : Profession),
    Kondratyev_profession = Carpenter ∧
    Davydov_profession = Painter ∧
    Fedorov_profession = Plumber

-- Lean 4 statement to prove the final assignment given the conditions
theorem solve_professions : conditions → final_assignment :=
by sorry

end solve_professions_l196_196160


namespace intersection_of_sets_l196_196080

def setA : Set ℝ := {x | x^2 ≤ 4 * x}
def setB : Set ℝ := {x | x < 1}

theorem intersection_of_sets : setA ∩ setB = {x | x < 1} := by
  sorry

end intersection_of_sets_l196_196080


namespace min_vertex_remove_eq_max_disjoint_paths_l196_196174

def directed_graph : Type := sorry -- Placeholder for actual graph definition
def vertices (G : directed_graph) : Type := sorry
def edges (G : directed_graph) : Type := sorry
def A : set (vertices G) := sorry
def B : set (vertices G) := sorry
def minimal_vertex_removal_number (G : directed_graph) (A B : set (vertices G)): Nat := sorry
def max_vertex_disjoint_paths (G : directed_graph) (A B : set (vertices G)): Nat := sorry

theorem min_vertex_remove_eq_max_disjoint_paths
  (G : directed_graph) (A B : set (vertices G)):
  minimal_vertex_removal_number G A B = max_vertex_disjoint_paths G A B := 
sorry

end min_vertex_remove_eq_max_disjoint_paths_l196_196174


namespace least_crawl_distance_l196_196764

noncomputable def cone_base_radius : ℝ := 500
noncomputable def cone_height : ℝ := 300 * Real.sqrt 3
noncomputable def start_distance_from_vertex : ℝ := 150
noncomputable def end_distance_from_vertex : ℝ := 450 * Real.sqrt 2

theorem least_crawl_distance :
  ∃ D : ℝ, D = 
    let R := Real.sqrt (cone_base_radius^2 + cone_height^2),
        θ := (2 * Real.pi * cone_base_radius) / R,
        A_x := start_distance_from_vertex,
        A_y := 0,
        B_x := end_distance_from_vertex * Real.cos(θ / 2),
        B_y := end_distance_from_vertex * Real.sin(θ / 2)
    in Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2) :=
sorry

end least_crawl_distance_l196_196764


namespace value_of_a_plus_b_l196_196021

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l196_196021


namespace lock_combinations_l196_196690

theorem lock_combinations :
  let digits := {4, 6, 8, 9}
  ∃ a b c d ∈ digits, a + b + c + d = 20 ∧
    finset.card (finset.filter (λ (comb : ℕ × ℕ × ℕ × ℕ), let (a, b, c, d) := comb in a + b + c + d = 20) (finset.product (finset.product (finset.product digits digits) digits) digits)) = 10 := 
sorry

end lock_combinations_l196_196690


namespace surface_area_of_rotation_l196_196721

theorem surface_area_of_rotation (d : ℝ) (d_pos : d > 0) :
  ∃ A1 An : ℝ × ℝ, 
  ∃ (A : list (ℝ × ℝ)) (A_valid : ∀ A_i ∈ A, convex A_i), 
  (list.sum (list.map (λ (A_i : (ℝ × ℝ)), dist A_i.1 A_i.2) A) = d) 
  → (∃ (surface_area : ℝ), surface_area ≤ (π * (d^2) / 2)) :=
sorry

end surface_area_of_rotation_l196_196721


namespace simplify_complex_expression_l196_196616

theorem simplify_complex_expression (x y : ℝ) : 
  (let i : ℂ := complex.I in (2 * x + 3 * i * y) * (2 * x - 3 * i * y)) = 
  (4 * x^2 + 9 * y^2) :=
by
  have h : complex.I * complex.I = -1 := complex.I_mul_I
  sorry

end simplify_complex_expression_l196_196616


namespace sum_of_ages_l196_196152

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196152


namespace capacity_of_vessel_b_l196_196369

theorem capacity_of_vessel_b :
  ∀ (x : ℝ), 
  (∀ (V_a V_b : ℝ), -- Capacities of the two vessels
    V_a = 2 → -- Vessel A has a capacity of 2 litres
    V_b = x → -- Vessel B has a capacity of x litres
    ∀ (conc_a conc_b : ℝ), -- Concentrations of alcohol in the vessels
      conc_a = 0.4 → -- 40% alcohol in Vessel A
      conc_b = 0.6 → -- 60% alcohol in Vessel B
      ∀ (total_liquid total_conc : ℝ), -- Total properties of the mixture
        total_liquid = 8 → -- Total liquid from both vessels is 8 litres
        total_conc = 0.44 → -- Total concentration in the 10-litre vessel is 44%
        V_b = 6) → -- Then Vessel B must have a capacity of 6 litres
sorry

end capacity_of_vessel_b_l196_196369


namespace find_point_C_l196_196044

open_locale real

def point := ℝ × ℝ

variables (A B C : point) (AB AC : point)

noncomputable def vec (P Q : point) : point := (Q.1 - P.1, Q.2 - P.2)

theorem find_point_C (hA : A = (1, 1)) (hB : B = (-1, 5)) (hAC_AB : vec A C = (2 * (vec A B))) :
  C = (-3, 9) :=
by { sorry }

end find_point_C_l196_196044


namespace min_value_f_l196_196878

def f (x : ℝ) : ℝ := x + 1 / (x - 4)

theorem min_value_f : ∃ (x : ℝ), x > 4 ∧ (∀ y > 4, f y ≥ 6) ∧ f x = 6 :=
by
  use 5
  sorry

end min_value_f_l196_196878


namespace problem_y_value_l196_196983

theorem problem_y_value (x y : ℝ)
  (h_right_triangle : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (BC AC : ℝ)
  (hBC : BC = 5)
  (hAC : AC = 12)
  (AM : ℝ)
  (h_AM : AM = x)
  (MN NP : ℝ)
  (hMN_perp_AC : ∀ (MN AC : ℝ), is_perpendicular MN AC)
  (hNP_perp_BC : ∀ (NP BC : ℝ), is_perpendicular NP BC)
  (N_on_AB : point_on_line N AB)
  (h_half_perimeter : y = (1/2) * (2 * (MN + NP + (12 - x) + (5 * (12 - x) / 12))))
  : y = (144 - 7*x) / 12 :=
sorry

end problem_y_value_l196_196983


namespace parabola_vertex_l196_196634

theorem parabola_vertex :
  ∃ a k : ℝ, (∀ x y : ℝ, y^2 - 4*y + 2*x + 7 = 0 ↔ y = k ∧ x = a - (1/2)*(y - k)^2) ∧ a = -3/2 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l196_196634


namespace log_five_fraction_l196_196835

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196835


namespace log_base5_of_inverse_sqrt5_l196_196828

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196828


namespace find_x_coordinate_l196_196362

theorem find_x_coordinate (m b x y : ℝ) (h1: m = 4) (h2: b = 100) (h3: y = 300) (line_eq: y = m * x + b) : x = 50 :=
by {
  sorry
}

end find_x_coordinate_l196_196362


namespace has_solution_in_interval_l196_196914

def f : ℝ → ℝ := λ x, -x^3 - 3*x + 5

lemma continuous_f : continuous f :=
by continuity

theorem has_solution_in_interval : ∃ c ∈ set.Ioo (1:ℝ) 2, f c = 0 :=
sorry

end has_solution_in_interval_l196_196914


namespace part1_tan_x_eq_1_l196_196985

noncomputable def m : ℝ × ℝ := (real.sqrt 2 / 2, -real.sqrt 2 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (real.sin x, real.cos x)

theorem part1_tan_x_eq_1 (h : m.1 * real.sin x + m.2 * real.cos x = 0)
  (hx : 0 < x ∧ x < real.pi / 2) : real.tan x = 1 := by sorry

end part1_tan_x_eq_1_l196_196985


namespace factorize_expression_l196_196438

theorem factorize_expression (m n : ℝ) :
  2 * m^3 * n - 32 * m * n = 2 * m * n * (m + 4) * (m - 4) :=
by
  sorry

end factorize_expression_l196_196438


namespace trigonometric_identity_l196_196392

theorem trigonometric_identity :
  let cos60 := (1 / 2)
  let sin30 := (1 / 2)
  let tan45 := (1 : ℝ)
  4 * cos60 + 8 * sin30 - 5 * tan45 = 1 :=
by
  let cos60 := (1 / 2 : ℝ)
  let sin30 := (1 / 2 : ℝ)
  let tan45 := (1 : ℝ)
  show 4 * cos60 + 8 * sin30 - 5 * tan45 = 1
  sorry

end trigonometric_identity_l196_196392


namespace count_interesting_numbers_l196_196589

def is_interesting (n : ℕ) : Prop :=
  let digits := (List.ofFn (λ i, (Nat.digit n i))).eraseDuplicates
  n ≥ 10^9 ∧ n < 10^10 ∧ digits.length = 10 ∧ n % 11111 = 0

theorem count_interesting_numbers : ∃ (count : ℕ), count = 3456 ∧
  ∀ n, is_interesting n ↔ n ∈ (Finset.range (10^10)).filter is_interesting :=
by
  exists 3456
  sorry

end count_interesting_numbers_l196_196589


namespace min_cos_C_l196_196917

theorem min_cos_C {A B C : ℝ} (h : sin A + sqrt 2 * sin B = 2 * sin C) : cos C ≥ (sqrt 6 - sqrt 2) / 4 :=
sorry

end min_cos_C_l196_196917


namespace ten_row_triangle_total_l196_196366

theorem ten_row_triangle_total:
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  rods + connectors = 231 :=
by
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  sorry

end ten_row_triangle_total_l196_196366


namespace sum_of_first_n_terms_l196_196891

-- Definitions for the sequences and the problem conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

-- The theorem statement
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range n).sum c = T n :=
  sorry

end sum_of_first_n_terms_l196_196891


namespace polygon_diagonals_formula_l196_196759

theorem polygon_diagonals_formula (n : ℕ) (h₁ : n = 5) (h₂ : 2 * n = (n * (n - 3)) / 2) :
  ∃ D : ℕ, D = n * (n - 3) / 2 :=
by
  sorry

end polygon_diagonals_formula_l196_196759


namespace p_sufficient_but_not_necessary_for_q_l196_196905

open Classical

variables {x y : ℝ}

def p := x > 0 ∧ y > 0
def q := x * y > 0

theorem p_sufficient_but_not_necessary_for_q : (p → q) ∧ ¬(q → p) := by
  split
  { intro h
    cases h with h1 h2
    exact mul_pos h1 h2 }
  { intro h
    by_cases hx : x > 0
    { by_cases hy : y > 0
      { exact ⟨hx, hy⟩ }
      { have hfalse : x * y ≤ 0 := mul_nonpos_of_nonneg_of_nonpos (le_of_lt hx) (le_of_not_gt hy)
        linarith } }
    { have hfalse : ¬ (x = 0 ∨ ¬ (x = 0 ∧ y ≠ 0))
      { split; intro hnx; linarith }
      exact ⟨hx, hfalse.elim⟩ } }

end p_sufficient_but_not_necessary_for_q_l196_196905


namespace sequence_diverges_l196_196200

theorem sequence_diverges (a : ℕ → ℝ) (s : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ k, 0 < a k) →
  (∀ n, s n = ∑ i in Finset.range (n + 1), a i) →
  filter.tendsto s filter.at_top filter.at_top →
  filter.tendsto S filter.at_top filter.at_top :=
begin
  sorry
end

end sequence_diverges_l196_196200


namespace coefficient_x_105_l196_196850

noncomputable def P (x : ℕ) : ℤ[X] := ∏ k in Finset.range 1 16, (X^k - k)

-- The theorem that states the resultant coefficient of x^105 is 134 given the polynomial P(x)
theorem coefficient_x_105 : (P x).coeff 105 = 134 :=
sorry

end coefficient_x_105_l196_196850


namespace max_distance_from_circle_to_line_l196_196165

-- Definitions for the given conditions
def circle_center : ℝ × ℝ := (2,2)
def circle_radius : ℝ := Real.sqrt 2
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2
def line_equation (x y : ℝ) : Prop := x - y - 4 = 0

-- The main theorem statement
theorem max_distance_from_circle_to_line :
  ∀ (x y : ℝ), circle_equation x y → ∃ d : ℝ, d = 3 * Real.sqrt 2 := 
by
  sorry -- Proof to be completed

end max_distance_from_circle_to_line_l196_196165


namespace ratio_DK_AB_l196_196763

-- Definitions of points and lengths
variables (A B C D C₁ K : Type) [geometry.realAE A B C D C₁ K]

-- Given conditions
def is_midpoint (C₁ : Type) (A D : Type) [geometry.realAE C₁ A D] : Prop :=
  sorry

def is_rectangle (A B C D : Type) [geometry.realAE A B C D] : Prop :=
  sorry

def length_AD (x : ℝ) (A D : Type) [geometry.realAE A D] : Prop :=
  sorry

-- Prove that the ratio DK / AB is 1 / 3
theorem ratio_DK_AB (A B C D C₁ K : Type) [geometry.realAE A B C D C₁ K]
    (h_midpoint : is_midpoint C₁ A D) 
    (h_rectangle : is_rectangle A B C D)
    (h_length_AD : length_AD x A D) : 
    (length K / length A B) = 1 / 3 :=
  sorry

end ratio_DK_AB_l196_196763


namespace max_prime_difference_l196_196571

theorem max_prime_difference (a b c d : ℕ) 
  (p1 : Prime a) (p2 : Prime b) (p3 : Prime c) (p4 : Prime d)
  (p5 : Prime (a + b + c + 18 + d)) (p6 : Prime (a + b + c + 18 - d))
  (p7 : Prime (b + c)) (p8 : Prime (c + d))
  (h1 : a + b + c = 2010) (h2 : a ≠ 3) (h3 : b ≠ 3) (h4 : c ≠ 3) (h5 : d ≠ 3) (h6 : d ≤ 50)
  (distinct_primes : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ (a + b + c + 18 + d)
                    ∧ a ≠ (a + b + c + 18 - d) ∧ a ≠ (b + c) ∧ a ≠ (c + d)
                    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ (a + b + c + 18 + d)
                    ∧ b ≠ (a + b + c + 18 - d) ∧ b ≠ (b + c) ∧ b ≠ (c + d)
                    ∧ c ≠ d ∧ c ≠ (a + b + c + 18 + d)
                    ∧ c ≠ (a + b + c + 18 - d) ∧ c ≠ (b + c) ∧ c ≠ (c + d)
                    ∧ d ≠ (a + b + c + 18 + d) ∧ d ≠ (a + b + c + 18 - d)
                    ∧ d ≠ (b + c) ∧ d ≠ (c + d)
                    ∧ (a + b + c + 18 + d) ≠ (a + b + c + 18 - d)
                    ∧ (a + b + c + 18 + d) ≠ (b + c) ∧ (a + b + c + 18 + d) ≠ (c + d)
                    ∧ (a + b + c + 18 - d) ≠ (b + c) ∧ (a + b + c + 18 - d) ≠ (c + d)
                    ∧ (b + c) ≠ (c + d)) :
  ∃ max_diff : ℕ, max_diff = 2067 := sorry

end max_prime_difference_l196_196571


namespace f_is_even_f_range_proof_h_min_value_l196_196901

-- Define the given functions and the conditions
def f (x : ℝ) : ℝ := sqrt (1 + x) + sqrt (1 - x)
def g (x : ℝ) : ℝ := sqrt (1 - x^2)

-- Assumptions and definitions
def f_even : Prop := ∀ x : ℝ, f (-x) = f x
def f_range : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → sqrt 2 ≤ f x ∧ f x ≤ 2

noncomputable def F (x a : ℝ) : ℝ := f x + 2 * a * g x
def a_neg (a : ℝ) : Prop := a < 0

-- Maximum value of F(x) with given conditions
def h (a : ℝ) : ℝ := sorry
def h_min (v : ℝ) : Prop := ∀ a : ℝ, a < 0 → h a ≥ v

-- Prove that f is even and its range
theorem f_is_even : f_even := sorry
theorem f_range_proof : f_range := sorry

-- Prove the minimum value of h(a)
theorem h_min_value (a : ℝ) : a_neg a → h_min (sqrt 2) := sorry

end f_is_even_f_range_proof_h_min_value_l196_196901


namespace find_m_l196_196496

noncomputable def f (x m : ℝ) : ℝ := 2 * m * real.sin x - 2 * (real.cos x)^2 + m^2 / 2 - 4 * m + 3

theorem find_m (m : ℝ) :
  (∀ x, f x m ≥ -7) → (∃ m, m = 10) :=
sorry

end find_m_l196_196496


namespace arithmetic_sequence_suff_nec_straight_line_l196_196173

variable (n : ℕ) (P_n : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, a (m + 1) = a m + d

def lies_on_straight_line (P : ℕ → ℝ) : Prop :=
  ∃ m b, ∀ n, P n = m * n + b

theorem arithmetic_sequence_suff_nec_straight_line
  (h_n : 0 < n)
  (h_arith : arithmetic_sequence P_n) :
  lies_on_straight_line P_n ↔ arithmetic_sequence P_n :=
sorry

end arithmetic_sequence_suff_nec_straight_line_l196_196173


namespace sqrt_sixteen_equals_four_l196_196404

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l196_196404


namespace probability_final_roll_six_l196_196738

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l196_196738


namespace odd_function_min_periodic_3_l196_196051

noncomputable def f : ℝ → ℝ
| x => if -3/2 < x ∧ x < 0 then real.logb 2 (-3 * x + 1) else sorry

theorem odd_function_min_periodic_3 (f : ℝ → ℝ)
  (hf_odd: ∀ x, f (-x) = -f (x))
  (hf_period: ∀ x, f (x + 3) = f x)
  (hf_def: ∀ x : ℝ, -3/2 < x ∧ x < 0 → f x = real.logb 2 (-3 * x + 1)) :
  f 2011 = -2 := 
sorry

end odd_function_min_periodic_3_l196_196051


namespace finished_in_6th_l196_196975

variable (p : ℕ → Prop)
variable (Sana Max Omar Jonah Leila : ℕ)

-- Conditions
def condition1 : Prop := Omar = Jonah - 7
def condition2 : Prop := Sana = Max - 2
def condition3 : Prop := Leila = Jonah + 3
def condition4 : Prop := Max = Omar + 1
def condition5 : Prop := Sana = 4

-- Conclusion
theorem finished_in_6th (h1 : condition1 Omar Jonah)
                         (h2 : condition2 Sana Max)
                         (h3 : condition3 Leila Jonah)
                         (h4 : condition4 Max Omar)
                         (h5 : condition5 Sana) :
  Max = 6 := by
  sorry

end finished_in_6th_l196_196975


namespace log_base_5_of_inv_sqrt_5_l196_196822

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196822


namespace total_wet_surface_area_of_cistern_l196_196692

-- The Lean theorem using the given conditions and question translating to proof

theorem total_wet_surface_area_of_cistern :
  ∀ (length width depth : ℝ),
    length = 4 → width = 8 → depth = 1.25 →
    let bottom_area := length * width in
    let short_wall_area := 2 * (length * depth) in
    let long_wall_area := 2 * (width * depth) in
    let total_wet_surface_area := bottom_area + short_wall_area + long_wall_area in
    total_wet_surface_area = 62 :=
by
  -- introduce the variables length, width, and depth
  intros length width depth length_eq width_eq depth_eq,
  -- define bottom_area, short_wall_area, and long_wall_area
  let bottom_area := length * width,
  let short_wall_area := 2 * (length * depth),
  let long_wall_area := 2 * (width * depth),
  -- define total_wet_surface_area
  let total_wet_surface_area := bottom_area + short_wall_area + long_wall_area,
  -- simplify and use the given condition
  have h1 : bottom_area = 32 := by rw [length_eq, width_eq]; exact rfl,
  have h2 : short_wall_area = 10 := by rw [length_eq, depth_eq]; exact rfl,
  have h3 : long_wall_area = 20 := by rw [width_eq, depth_eq]; exact rfl,
  have h4 : total_wet_surface_area = 62 := by
    rw [bottom_area, short_wall_area, long_wall_area, h1, h2, h3]; exact rfl,
  -- assertion
  exact h4

end total_wet_surface_area_of_cistern_l196_196692


namespace find_number_of_ordered_pairs_l196_196858

noncomputable def num_pairs : ℂ → ℂ → Prop :=
  λ a b, a^5 * b^3 + a^2 * b^7 = 0 ∧ a^3 * b^2 = 1

theorem find_number_of_ordered_pairs :
  (∃ a b : ℂ, num_pairs a b) ∧ 
  (set.card ({p : ℂ × ℂ | num_pairs p.1 p.2}.to_finset) = 40) :=
sorry

end find_number_of_ordered_pairs_l196_196858


namespace xiao_ming_brother_age_l196_196256

def first_year_with_no_repeated_digits (y: ℕ) : Prop :=
  let digit_set := (y.digits 10).to_finset
  in digit_set.card = y.digits 10.length

def is_multiple_of_19 (y: ℕ) : Prop := y % 19 = 0

theorem xiao_ming_brother_age (brother_birth_year: ℕ) (h1: is_multiple_of_19 brother_birth_year) 
  (h2: ∀ y < 2013, y ≥ brother_birth_year → ¬ first_year_with_no_repeated_digits y)
  (h3: first_year_with_no_repeated_digits 2013) :
  2013 - brother_birth_year = 18 :=
by
  sorry

end xiao_ming_brother_age_l196_196256


namespace set_equality_l196_196081

theorem set_equality (M P : Set (ℝ × ℝ))
  (hM : M = {p : ℝ × ℝ | p.1 + p.2 < 0 ∧ p.1 * p.2 > 0})
  (hP : P = {p : ℝ × ℝ | p.1 < 0 ∧ p.2 < 0}) : M = P :=
by
  sorry

end set_equality_l196_196081


namespace additional_songs_added_l196_196511

theorem additional_songs_added (original_songs : ℕ) (song_duration : ℕ) (total_duration : ℕ) :
  original_songs = 25 → song_duration = 3 → total_duration = 105 → 
  (total_duration - original_songs * song_duration) / song_duration = 10 :=
by
  intros h1 h2 h3
  sorry

end additional_songs_added_l196_196511


namespace x_1000_bounds_l196_196164

def x : ℕ → ℝ
| 0       := 5
| (n + 1) := x n + 1 / x n

theorem x_1000_bounds :
  45 < x 1000 ∧ x 1000 < 45.1 := by
  sorry

end x_1000_bounds_l196_196164


namespace carbonate_ions_in_Al2_CO3_3_l196_196452

theorem carbonate_ions_in_Al2_CO3_3 (total_weight : ℕ) (formula : String) 
  (molecular_weight : ℕ) (ions_in_formula : String) : 
  formula = "Al2(CO3)3" → molecular_weight = 234 → ions_in_formula = "CO3" → total_weight = 3 := 
by
  intros formula_eq weight_eq ions_eq
  sorry

end carbonate_ions_in_Al2_CO3_3_l196_196452


namespace grade_assignment_ways_l196_196352

-- Define the number of students and the number of grade choices
def students : ℕ := 12
def grade_choices : ℕ := 4

-- Define the number of ways to assign grades
def num_ways_to_assign_grades : ℕ := grade_choices ^ students

-- Prove that the number of ways to assign grades is 16777216
theorem grade_assignment_ways :
  num_ways_to_assign_grades = 16777216 :=
by
  -- Calculation validation omitted (proof step)
  sorry

end grade_assignment_ways_l196_196352


namespace angle_bisector_theorem_l196_196329

variable (A B C D : Type)
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
variable [InnerProductSpace ℝ D]

def angle_bisector (BD : B → D) (ABC : Triangle A B C) : Prop := sorry

theorem angle_bisector_theorem (A B C D : Type) 
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
  [InnerProductSpace ℝ D]
  (ABC : Triangle A B C) (BD : B → D)
  (h : angle_bisector BD ABC) :
  ∀ (AD DC AB BC : ℝ), AD / DC = AB / BC := sorry

end angle_bisector_theorem_l196_196329


namespace max_expression_value_l196_196179

theorem max_expression_value (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_eq : x^2 + y^2 + z^2 = 1) : 
  ∃ (M : ℝ), M = sqrt 13 ∧ (3 * x * y * sqrt 4 + 9 * y * z ≤ M) :=
by
  sorry

end max_expression_value_l196_196179


namespace parametric_C2_max_distance_P_l196_196133

open Real

/-- Define the curves C1 and C2. -/
def curve_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def curve_C2 (x y : ℝ) : Prop := ∃ θ : ℝ, x = √2 * cos θ ∧ y = √3 * sin θ

/-- Define the line l. -/
def line_l (x y : ℝ) : Prop := x + y - 4 * √5 = 0

/-- Parametric equations for the curve C2. -/
theorem parametric_C2 : ∀ θ : ℝ, curve_C2 (√2 * cos θ) (√3 * sin θ) :=
by
  intro θ
  use θ
  simp

/-- Maximum distance from point P on curve C2 to line l. -/
theorem max_distance_P : ∃ (x y : ℝ), 
  curve_C2 x y ∧ line_l x y ∧ 
  x = -2 * √5 / 5 ∧ y = -3 * √5 / 5 ∧ 
  ∀ (x' y' : ℝ), curve_C2 x' y' → 
  abs (x' + y' - 4 * √5) / √2 ≤ 5 * √10 / 2 :=
by
  have h : ∃ θ : ℝ, 
    curve_C2 (√2 * cos θ) (√3 * sin θ) ∧ 
    ∃ (x y : ℝ),
      x = √2 * cos θ ∧ y = √3 * sin θ ∧ 
      abs (x + y - 4 * √5) / √2 = 5 * √10 / 2
  {
    use -atan (√(2 / 3))
    use -2 * √5 / 5
    use -3 * √5 / 5
    -- Proof details omitted.
    sorry
  }
  cases h with θ h_aug
  exact ⟨-2 * √5 / 5, -3 * √5 / 5, h_aug⟩

end parametric_C2_max_distance_P_l196_196133


namespace convex_polygon_max_sides_l196_196124

theorem convex_polygon_max_sides (n : ℕ) (h_convex : convex_polygon n) (h_obtuse : ∃ (k : ℕ), k = 5 ∧ obtuse_interior_angles k) : n ≤ 8 := by
  sorry

-- Definitions used in the theorem
def convex_polygon (n : ℕ) := n ≥ 3 -- A polygon must have at least 3 sides
def obtuse_interior_angles (k : ℕ) := k > 0 ∧ k < n

end convex_polygon_max_sides_l196_196124


namespace angle_bisector_of_LBK_l196_196580

-- Definitions of the problem elements based on given conditions
variables {A B C A1 B1 C1 L K : Type*}

-- Assume we already have a definition of angle bisectors and intersection points
axiom is_angle_bisector_of (X Y Z : Type*) (X1 Y1 Z1 : Type*) : Prop
axiom intersection_point (X Y X1 Y1 : Type*) : Type*

-- Given conditions
def angle_bisectors (A A1 B B1 C C1 : Type*) : Prop :=
  is_angle_bisector_of A B C A1 B1 C1

def intersection_points (A A1 B B1 C C1 : Type*) (L K : Type*) : Prop :=
  L = intersection_point A A1 B1 C1 ∧
  K = intersection_point C C1 A1 B1

-- The Equivalent Proof Problem as a Lean Theorem Statement
theorem angle_bisector_of_LBK
  (A A1 B B1 C C1 L K : Type*)
  (h1 : angle_bisectors A A1 B B1 C C1)
  (h2 : intersection_points A A1 B B1 C C1 L K) :
  is_angle_bisector_of B L K B1 :=
sorry

end angle_bisector_of_LBK_l196_196580


namespace power_function_passes_point_l196_196074

noncomputable def f (k α x : ℝ) : ℝ := k * x^α

theorem power_function_passes_point (k α : ℝ) (h1 : f k α (1/2) = (Real.sqrt 2)/2) : 
  k + α = 3/2 :=
sorry

end power_function_passes_point_l196_196074


namespace zeros_before_first_nonzero_digit_l196_196090

theorem zeros_before_first_nonzero_digit 
  (h : ∀ n : ℕ, n = 2^3 * 5^6) : 
  (zeros_before_first_nonzero_digit_decimal \(\frac{1}{n}) = 6) :=
sorry

end zeros_before_first_nonzero_digit_l196_196090


namespace find_perpendicular_line_through_intersection_l196_196854

open Real

def line1 (x y : ℝ) : Prop := x - 6 * y + 4 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y = 5
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y = 0

theorem find_perpendicular_line_through_intersection :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∃ x y : ℝ, line2 (2 * x) y ∧ perpendicular_line x y) :=
by
  intro h,
  obtain ⟨x, y, hx1, hy1⟩ := h,
  sorry

end find_perpendicular_line_through_intersection_l196_196854


namespace trigonometric_identity_l196_196920

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  2 * (3 / -5) + 4 / -5 = -2 / 5 :=
by
  sorry

end trigonometric_identity_l196_196920


namespace tourists_count_l196_196347

theorem tourists_count (n k : ℤ) (h1 : 2 * k % n = 1) (h2 : 3 * k % n = 13) : n = 23 := 
by
-- Proof is omitted
sorry

end tourists_count_l196_196347


namespace count_zeros_decimal_representation_l196_196104

theorem count_zeros_decimal_representation (n m : ℕ) (h : n = 3) (h₁ : m = 6) : 
  ∃ k : ℕ, k = 5 ∧ 
    let d := (1 : ℚ) / (2^n * 5^m) in 
    let s := d.repr.mantissa in -- Assuming a function repr.mantissa to represent decimal digits
    s.indexOf '1' - s.indexOf '.' - 1 = k := sorry

end count_zeros_decimal_representation_l196_196104


namespace sum_of_ages_l196_196151

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196151


namespace surface_area_of_sphere_eq_3pi_l196_196194

theorem surface_area_of_sphere_eq_3pi
  (P A B C : Point)
  (PA PB PC : ℝ)
  (h1 : PA = dist P A)
  (h2 : PB = dist P B)
  (h3 : PC = dist P C)
  (h4 : PA = 1)
  (h5 : PB = 1)
  (h6 : PC = 1)
  (h_perpendicular : PA * PB * sin (angle P A B) = PA * PC * sin (angle P A C) = PB * PC * sin (angle P B C) = 1) :
  surface_area (Sphere P 1) = 3 * π := sorry

end surface_area_of_sphere_eq_3pi_l196_196194


namespace value_of_expression_l196_196906

theorem value_of_expression (x y : ℝ) (h1 : 4 * x + y = 20) (h2 : x + 4 * y = 16) : 
  17 * x ^ 2 + 20 * x * y + 17 * y ^ 2 = 656 :=
sorry

end value_of_expression_l196_196906


namespace rest_area_milepost_l196_196635

theorem rest_area_milepost (milepost_first : ℕ) (milepost_seventh : ℕ) (h_first : milepost_first = 20) (h_seventh : milepost_seventh = 140) : 
  ∃ milepost_rest : ℕ, milepost_rest = (milepost_first + milepost_seventh) / 2 ∧ milepost_rest = 80 :=
by
  sorry

end rest_area_milepost_l196_196635


namespace compute_a_plus_b_l196_196553

noncomputable def side_length_square : ℝ := 4
noncomputable def perimeter_triangle_MNP (a b : ℤ) : ℝ := a + b * Real.sqrt 3

theorem compute_a_plus_b :
  ∃ a b : ℤ, 
  let perimeter := 3 * (side_length_square + (side_length_square / 2) * Real.sqrt 3)
  in perimeter_triangle_MNP a b = perimeter ∧ a + b = 20 :=
by {
  sorry
}

end compute_a_plus_b_l196_196553


namespace value_of_a_plus_b_l196_196025

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l196_196025


namespace average_weight_correct_l196_196262

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l196_196262


namespace sum_of_ages_l196_196154

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196154


namespace benny_added_march_l196_196783

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end benny_added_march_l196_196783


namespace probability_six_on_final_roll_l196_196742

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l196_196742


namespace debby_drink_bottles_per_day_l196_196423

theorem debby_drink_bottles_per_day
  (initial_bottles : ℕ)
  (days : ℕ)
  (remaining_bottles : ℕ)
  (bottles_per_day : ℕ)
  (h1 : initial_bottles = 264)
  (h2 : days = 11)
  (h3 : remaining_bottles = 99) :
  bottles_per_day = (initial_bottles - remaining_bottles) / days :=
by {
  intros,
  -- proof should be going here
  sorry
}

end debby_drink_bottles_per_day_l196_196423


namespace arrange_x_y_z_l196_196934

theorem arrange_x_y_z (x : ℝ) (hx : 0.9 < x ∧ x < 1) :
  let y := x^(1/x)
  let z := x^y
  x < z ∧ z < y :=
by
  let y := x^(1/x)
  let z := x^y
  have : 0.9 < x ∧ x < 1 := hx
  sorry

end arrange_x_y_z_l196_196934


namespace isosceles_right_triangle_shaded_area_l196_196375

def area_of_isosceles_right_triangle (leg_length : ℝ) : ℝ :=
  0.5 * leg_length * leg_length

def area_of_small_triangle (total_area : ℝ) (num_triangles : ℕ) : ℝ :=
  total_area / num_triangles

def shaded_area (area_per_triangle : ℝ) (num_shaded : ℕ) : ℝ :=
  area_per_triangle * num_shaded

theorem isosceles_right_triangle_shaded_area :
  let leg_length := 12
  let total_triangles := 18
  let shaded_triangles := 12
  let total_area := area_of_isosceles_right_triangle leg_length
  let small_triangle_area := area_of_small_triangle total_area total_triangles
  shaded_area small_triangle_area shaded_triangles = 48 :=
by 
  sorry

end isosceles_right_triangle_shaded_area_l196_196375


namespace find_m_l196_196237

theorem find_m (m : ℕ) : 
  m ≥ 50 ∧ m ≤ 180 ∧ 
  m % 9 = 0 ∧ 
  m % 10 = 7 ∧ 
  m % 7 = 5 → 
  m = 117 :=
by 
  intros h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end find_m_l196_196237


namespace space_shuttle_speed_l196_196359

theorem space_shuttle_speed (speed_km_per_hr : ℕ) (seconds_per_hour : ℕ) (h_speed : speed_km_per_hr = 7200) (h_time : seconds_per_hour = 3600) :
  (speed_km_per_hr / seconds_per_hour) = 2 :=
by
  rw [h_speed, h_time]
  norm_num

end space_shuttle_speed_l196_196359


namespace choir_meets_every_5_days_l196_196633

theorem choir_meets_every_5_days (n : ℕ) (h1 : n = 15) (h2 : ∃ k : ℕ, 15 = 3 * k) : ∃ x : ℕ, 15 = x * 3 ∧ x = 5 := 
by
  sorry

end choir_meets_every_5_days_l196_196633


namespace probability_final_roll_six_l196_196737

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l196_196737


namespace one_red_ball_probability_l196_196921

/-- Given that there is 1 red ball and 2 black balls in box A, 
all of the same shape and texture, and there are 2 red balls 
and 2 black balls in box B, also of the same shape and texture. 
Now, one ball is randomly drawn from each of the two boxes.
Prove that the probability that exactly one of the two balls drawn is red is 1/2. -/
theorem one_red_ball_probability : 
  let ballA := (1, 2)  -- 1 red, 2 black in box A
      ballB := (2, 2)  -- 2 red, 2 black in box B
  in (prob_red_A_black_B ballA ballB + prob_red_B_black_A ballA ballB) = 1 / 2 :=
by
  sorry

/-- Probability of drawing a red ball from box A and a black ball 
   from box B -/
def prob_red_A_black_B : (ℕ × ℕ) → (ℕ × ℕ) → ℚ
| (redA, blackA), (redB, blackB) =>
  ((redA / (redA + blackA : ℕ)) * (blackB / (redB + blackB : ℕ) : ℚ))

/-- Probability of drawing a red ball from box B and a black ball 
   from box A -/
def prob_red_B_black_A : (ℕ × ℕ) → (ℕ × ℕ) → ℚ
| (redA, blackA), (redB, blackB) =>
  ((redB / (redB + blackB : ℕ)) * (blackA / (redA + blackA : ℕ) : ℚ))

end one_red_ball_probability_l196_196921


namespace calculate_pool_volume_l196_196391

theorem calculate_pool_volume :
  ∀ (d h1 h2: ℝ), d = 20 ∧ h1 = 3 ∧ h2 = 5 → 
  let r := d / 2 in 
  let V := π * r^2 * h1 in
  V = 300 * π :=
by 
  intro d h1 h2 h_cond
  cases h_cond with h_d h_rest
  cases h_rest with h_h1 h_h2
  let r := d / 2
  let V := π * r^2 * h1
  have h_r : r = 10 := by rw [h_d, div_eq_mul_one_div, mul_comm, mul_one_div, inv_of_eq_inv (2 : ℝ)]
  have h_V : V = 300 * π := by rw [←h_h1, ←h_r, pow_two, div_pow, ←mul_assoc]
  exact h_V

end calculate_pool_volume_l196_196391


namespace speed_of_second_part_of_trip_l196_196728

-- Given conditions
def total_distance : Real := 50
def first_part_distance : Real := 25
def first_part_speed : Real := 66
def average_speed : Real := 44.00000000000001

-- The statement we want to prove
theorem speed_of_second_part_of_trip :
  ∃ second_part_speed : Real, second_part_speed = 33 :=
by
  sorry

end speed_of_second_part_of_trip_l196_196728


namespace jill_third_month_days_l196_196566

theorem jill_third_month_days :
  ∀ (days : ℕ),
    (earnings_first_month : ℕ) = 10 * 30 →
    (earnings_second_month : ℕ) = 20 * 30 →
    (total_earnings : ℕ) = 1200 →
    (total_earnings_two_months : ℕ) = earnings_first_month + earnings_second_month →
    (earnings_third_month : ℕ) = total_earnings - total_earnings_two_months →
    earnings_third_month = 300 →
    days = earnings_third_month / 20 →
    days = 15 := 
sorry

end jill_third_month_days_l196_196566


namespace angle_BPC_theorem_l196_196993

structure Square (A B C D : Type) :=
  (side_length : ℝ)
  (AB : A = B)
  (BC : B = C)
  (CD : C = D)
  (DA : D = A)
  (length : AB = 6)

structure EquilateralTriangle (A B E : Type) :=
  (length : AB = BE ∧ BE = AE ∧ AE = AB)

structure Perpendicular (PQ BC : Type) :=
  (perp : PQ ⊥ BC)

def determine_angle_BPC (A B C D E P Q : Type) 
  (sq : Square A B C D) (tri : EquilateralTriangle A E B)
  (intersect : BE ∩ AC = P) (perp : Perpendicular PQ BC) : ℝ :=
  105

theorem angle_BPC_theorem (A B C D E P Q : Type) 
  (sq : Square A B C D) (tri : EquilateralTriangle A E B) 
  (intersect : BE ∩ AC = P) (perp : Perpendicular PQ BC) : determine_angle_BPC A B C D E P Q sq tri intersect perp = 105 :=
  sorry

end angle_BPC_theorem_l196_196993


namespace sufficient_condition_for_B_subset_A_l196_196166

def A : set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : set ℝ := {x | x * m + 1 = 0}

theorem sufficient_condition_for_B_subset_A (m : ℝ) : 
  B m ⊆ A → m ∈ {0, 1/3} :=
sorry

end sufficient_condition_for_B_subset_A_l196_196166


namespace number_of_boys_in_class_l196_196978

theorem number_of_boys_in_class :
  ∃ B : ℕ,
    let avg_score_boys := 84 in
    let avg_score_girls := 92 in
    let num_girls := 4 in
    let avg_score_class := 86 in
    (avg_score_boys * B + avg_score_girls * num_girls) / (B + num_girls) = avg_score_class ∧
    B = 12 :=
by
  sorry

end number_of_boys_in_class_l196_196978


namespace not_square_l196_196950

open Nat

theorem not_square (p : ℕ) (hp : Prime p) : ¬ ∃ a : ℤ, (7 * p : ℤ) + (3 : ℤ)^p - 4 = a^2 := 
sorry

end not_square_l196_196950


namespace first_four_digits_of_pow_l196_196275

noncomputable def a : ℝ := 5^(1001) + 2
noncomputable def b : ℝ := 5 / 3

theorem first_four_digits_of_pow (x : ℝ) (h : x = a^b) : 
  floor (10^4 * (x - floor x)) = 3333 := 
sorry

end first_four_digits_of_pow_l196_196275


namespace roots_of_polynomial_l196_196205

theorem roots_of_polynomial (x1 x2 x3 x4 : ℝ) (W : ℝ → ℝ) 
  (h1 : ∀ x, W x = 0 ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4))
  (h2 : ∀ n : ℕ, W n ∈ ℚ)
  (h3 : x3 + x4 ∈ ℚ)
  (h4 : x3 * x4 ∈ ℝ \ ℚ) : x1 + x2 = x3 + x4 :=
sorry

end roots_of_polynomial_l196_196205


namespace boys_brought_the_same_car_l196_196603

-- Definitions for the properties of toy cars:
structure ToyCar :=
(size : string)    -- size can be "small" or "big"
(color : string)   -- color can be "green", "blue" etc.
(trailer : bool)   -- trailer can be true (with trailer) or false (without trailer)

-- Conditions
def M1 : ToyCar := { size := "unknown", color := "unknown", trailer := true }
def M2 : ToyCar := { size := "small", color := "unknown", trailer := false }
def M3 : ToyCar := { size := "unknown", color := "green", trailer := false }

def V1 : ToyCar := { size := "unknown", color := "unknown", trailer := false }
def V2 : ToyCar := { size := "small", color := "green", trailer := true }

def K1 : ToyCar := { size := "big", color := "unknown", trailer := false }
def K2 : ToyCar := { size := "small", color := "blue", trailer := true }

-- The final answer
def common_car : ToyCar := { size := "big", color := "green", trailer := false }

-- Proof statement
theorem boys_brought_the_same_car : 
  (∃ c : ToyCar, c = M1 ∨ c = M2 ∨ c = M3) ∧
  (∃ c : ToyCar, c = V1 ∨ c = V2) ∧
  (∃ c : ToyCar, c = K1 ∨ c = K2) ∧
  (common_car = c) :=
sorry

end boys_brought_the_same_car_l196_196603


namespace length_BC_in_triangle_ABC_l196_196998

theorem length_BC_in_triangle_ABC :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
  let AB : ℝ := 2,
      AC : ℝ := 3,
      median_from_A : ℝ := BC / 2 -- assumption that median from A to BC is half of BC
  in (BC = ∥midpoint (B, C) - A∥) → 
     (BC = 2 * median_from_A) → 
  BC = (√130) / 5 :=
by
  intro A B C _ _ _
  let AB := 2 : ℝ
  let AC := 3 : ℝ
  let median_from_A := BC / 2
  intro h1 h2
  sorry

end length_BC_in_triangle_ABC_l196_196998


namespace sum_of_extrema_of_f_l196_196251

noncomputable def f (x : ℝ) : ℝ := 1 + (Real.sin x) / (2 + (Real.cos x))

theorem sum_of_extrema_of_f : 
  let ymax := (3 + Real.sqrt 3) / 3,
      ymin := (3 - Real.sqrt 3) / 3
  in ymax + ymin = 2 := by 
  sorry

end sum_of_extrema_of_f_l196_196251


namespace no_digit_make_1C4_multiple_of_5_l196_196456

theorem no_digit_make_1C4_multiple_of_5 : ∀ C : ℕ, C < 10 → ¬ (∃ k : ℕ, 100 + C * 10 + 4 = 5 * k) :=
by
  intros C hC
  intro h
  cases h with k hk
  have h_last_digit : (100 + C * 10 + 4) % 10 = 4 := by sorry
  have h_five_multiple : 5 * k % 10 = 0 := by sorry
  contradiction

end no_digit_make_1C4_multiple_of_5_l196_196456


namespace sum_of_factors_l196_196315

theorem sum_of_factors (n : ℕ) (h : n = 60) : 
  ∑ d in (finset.filter (λ x => x ∣ n) (finset.range (n+1))), d = 168 := 
by
  sorry

end sum_of_factors_l196_196315


namespace arithmetic_sequence_general_formula_product_bound_l196_196587

noncomputable def seq (n : ℕ) : ℝ :=
  if n = 0 then 1 / 2 else
    let rec seq' (n : ℕ) : ℝ :=
      if n = 0 then 1 / 2 else
        let a_next = 2 * (seq' (n-1) - 1) * (seq' (n-1) - 1) + seq' (n-1)
        a_next
    seq' n

theorem arithmetic_sequence :
  let seq' (n : ℕ) := 1 / (seq n - 1)
  ∀ n : ℕ, seq' (n+1) - seq' n = -2 :=
sorry

theorem general_formula :
  ∀ n : ℕ, seq (n+1) = (2*(n+1)-1) / (2*(n+1)) :=
sorry

theorem product_bound :
  ∀ n : ℕ, (list.prod (list.of_fn seq (n+1)))
    < 1 / real.sqrt (2*(n+1)) :=
sorry

end arithmetic_sequence_general_formula_product_bound_l196_196587


namespace log_base_5_sqrt_inverse_l196_196817

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196817


namespace koolaid_amount_l196_196146

variable (K : ℚ)

def initial_water := 16
def evaporated_water := 4
def quadruple_factor := 4

def remaining_water := initial_water - evaporated_water
def quadrupled_water := quadruple_factor * remaining_water

def total_liquid := K + quadrupled_water

def koolaid_percentage := (4 : ℚ) / 100

def percentage_condition := (K / total_liquid) = koolaid_percentage

theorem koolaid_amount :
  percentage_condition K → K = 2 :=
by
  intro h
  -- Proof goes here
  sorry

end koolaid_amount_l196_196146


namespace twelve_sided_polygon_l196_196135

noncomputable theory

def square_vertices : List ℂ :=
  [1, Complex.I, -1, -Complex.I]

def u : ℂ := exp (Complex.I * (Real.pi / 6))
def u2 : ℂ := u ^ 2

def vertices_of_equilateral_triangles : List ℂ :=
  [ 1 + (Complex.I - 1) * u2,
    Complex.I - (1 + Complex.I) * u2,
    -1 + (1 - Complex.I) * u2,
    -Complex.I + (1 + Complex.I) * u2 ]

def midpoints_of_segments : List ℂ :=
  [ (vertices_of_equilateral_triangles.head! + vertices_of_equilateral_triangles.tail!.head!) / 2,
    (vertices_of_equilateral_triangles.tail!.head! + vertices_of_equilateral_triangles.tail!.tail!.head!) / 2,
    (vertices_of_equilateral_triangles.tail!.tail!.head! + vertices_of_equilateral_triangles.tail!.tail!.tail!.head!) / 2,
    (vertices_of_equilateral_triangles.tail!.tail!.tail!.head! + vertices_of_equilateral_triangles.head!) / 2 ]

/-- Prove that the midpoints of the segments KL, LM, MN, and NK,
    as well as the midpoints of the segments AK, BK, BL, CL, CM, DM, DN, and AN,
    form the vertices of a regular 12-sided polygon. -/
theorem twelve_sided_polygon :
  ∃ (λ : ℂ), 
    let polygon_vertices := 
      [(λ * Complex.exp (11 * Complex.I * (Real.pi / 6))),
       (λ * Complex.exp (2 * Complex.I * (Real.pi / 3))),
       (λ * Complex.exp (Complex.I * (Real.pi / 3))),
       (λ * Complex.exp (7 * Complex.I * (Real.pi / 6))),
       (λ * Complex.exp (5 * Complex.I * (Real.pi / 6))),
       (λ * Complex.exp (5 * Complex.I * (2 * Real.pi / 3))),
       (λ * Complex.exp (4 * Complex.I * (Real.pi / 3))),
       (λ * Complex.exp (Complex.I * (Real.pi / 6))),
       (-λ * Complex.I),
       (λ),
       (λ * Complex.I),
       (-λ)] 
    in polygon_vertices.all (λ x, x.norm = λ) := 
sorry

end twelve_sided_polygon_l196_196135


namespace projection_of_a_on_b_l196_196052

variables (a b : ℝ) (u v : ℝ^3)
variables (ab_dot : ℝ) (b_mag : ℝ)

-- Conditions
hypothesis h1 : ab_dot = 12
hypothesis h2 : b_mag = 5
hypothesis h3 : b ≠ 0

-- Prove that projection of a onto b is 12/5
theorem projection_of_a_on_b : (ab_dot / b_mag) = 12 / 5 :=
by
  sorry

end projection_of_a_on_b_l196_196052


namespace find_interest_rate_per_annum_l196_196351

noncomputable def interest_rate (A P : ℝ) (n t : ℝ) : ℝ :=
  (A / P)^(1 / (n * t)) - 1

theorem find_interest_rate_per_annum :
  interest_rate 4913 4096 1 3 ≈ 0.059463094 :=
by 
  sorry

end find_interest_rate_per_annum_l196_196351


namespace count_zeros_decimal_representation_l196_196103

theorem count_zeros_decimal_representation (n m : ℕ) (h : n = 3) (h₁ : m = 6) : 
  ∃ k : ℕ, k = 5 ∧ 
    let d := (1 : ℚ) / (2^n * 5^m) in 
    let s := d.repr.mantissa in -- Assuming a function repr.mantissa to represent decimal digits
    s.indexOf '1' - s.indexOf '.' - 1 = k := sorry

end count_zeros_decimal_representation_l196_196103


namespace smallest_distance_l196_196219

-- Definitions for Rational Woman's path
def RationalWomanPath (t : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos t, 2 * Real.sin t)

-- Definitions for Rational Man's path
def RationalManPath (t : ℝ) : ℝ × ℝ := (Real.cos (t / 2), Real.sin (t / 2))

-- Define the distance function between two points in the plane
def dist (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Statement: The smallest possible distance between any point A on Rational Woman's track 
-- and B on Rational Man's track is 1
theorem smallest_distance : ∃ A B (t₁ t₂ : ℝ), 
  A = RationalWomanPath t₁ ∧ B = RationalManPath t₂ ∧ dist A B = 1 :=
sorry

end smallest_distance_l196_196219


namespace truck_driver_net_rate_of_pay_l196_196368

-- Conditions
def hours := 3
def speed := 45 -- miles per hour
def miles_per_gallon := 15
def pay_per_mile := 0.75 -- dollars per mile
def cost_per_gallon := 3.00 -- dollars per gallon

-- Define the various calculations
def total_distance := hours * speed -- miles
def diesel_usage := total_distance / miles_per_gallon -- gallons
def earnings := pay_per_mile * total_distance -- dollars
def diesel_cost := cost_per_gallon * diesel_usage -- dollars
def net_earnings := earnings - diesel_cost -- dollars
def net_rate_of_pay := net_earnings / hours -- dollars per hour

-- Theorem to prove the correct answer
theorem truck_driver_net_rate_of_pay : net_rate_of_pay = 24.75 := by
  sorry

end truck_driver_net_rate_of_pay_l196_196368


namespace area_constant_OPMN_l196_196491

-- Define the ellipse with the given equation.
def ellipse (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

-- Define the points on the ellipse.
def point_on_ellipse (x y : ℝ) : Prop :=
  ellipse x y

-- Define the specific points O, P, M, N.
def O : ℝ × ℝ := (0, 0)

-- Variables for points P, M, N on the ellipse
variables {P M N : ℝ × ℝ}

-- Condition for points P, M, N on the ellipse.
axiom P_on_ellipse : point_on_ellipse P.1 P.2
axiom M_on_ellipse : point_on_ellipse M.1 M.2
axiom N_on_ellipse : point_on_ellipse N.1 N.2

-- Definition of a parallelogram OPMN
def parallelogram_OPMN : Prop :=
  (M.1 = P.1 + N.1) ∧ (M.2 = P.2 + N.2)

-- The main theorem
theorem area_constant_OPMN
  (h1 : P_on_ellipse)
  (h2 : M_on_ellipse)
  (h3 : N_on_ellipse)
  (h4 : parallelogram_OPMN) :
  ∃ S : ℝ, S = 2 * sqrt 6 :=
sorry

end area_constant_OPMN_l196_196491


namespace fencing_cost_approx_122_52_l196_196853

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def fencing_cost (d rate : ℝ) : ℝ := circumference d * rate

theorem fencing_cost_approx_122_52 :
  let d := 26
  let rate := 1.50
  abs (fencing_cost d rate - 122.52) < 1 :=
by
  let d : ℝ := 26
  let rate : ℝ := 1.50
  let cost := fencing_cost d rate
  sorry

end fencing_cost_approx_122_52_l196_196853


namespace parallelogram_vertices_l196_196440

open Set Finset

noncomputable def is_parallelogram_vertices (S : Finset (ℝ × ℝ)) : Prop :=
∀ (A B C : (ℝ × ℝ)), A ∈ S → B ∈ S → C ∈ S → A ≠ B → A ≠ C → B ≠ C → 
  ∃ D : (ℝ × ℝ), D ∈ S ∧ (vector_span ℝ ({A,B,C,D} : set (ℝ × ℝ))).dim = 2 ∧ 
  (2 : ℝ) ∈ ({dist A B, dist B C, dist C D, dist D A})

theorem parallelogram_vertices (S : Finset (ℝ × ℝ)) :
  (∀ (A B C : (ℝ × ℝ)), A ∈ S → B ∈ S → C ∈ S → A ≠ B → A ≠ C → B ≠ C → 
      ∃ D : (ℝ × ℝ), D ∈ S ∧ (vector_span ℝ ({A,B,C,D} : set (ℝ × ℝ))).dim = 2 ∧ 
      (2 : ℝ) ∈ ({dist A B, dist B C, dist C D, dist D A})) →
  ∃ (A B C D : (ℝ × ℝ)), S = {A, B, C, D} ∧ 
    (vector_span ℝ ({A,B,C,D} : set (ℝ × ℝ))).dim = 2 ∧ 
    (∀ X Y : (ℝ × ℝ), X ∈ S → Y ∈ S → X ≠ Y → (X - Y).norm = (A - B).norm) :=
begin
  sorry
end

end parallelogram_vertices_l196_196440


namespace total_fish_in_lake_l196_196872

theorem total_fish_in_lake:
  let 
    white_ducks := 3 
    black_ducks := 7 
    multico_ducks := 6 
    fish_per_white_duck := 5 
    fish_per_black_duck := 10 
    fish_per_multico_duck := 12 
  in 
  white_ducks * fish_per_white_duck + 
  black_ducks * fish_per_black_duck + 
  multico_ducks * fish_per_multico_duck = 157 := 
by 
  sorry

end total_fish_in_lake_l196_196872


namespace total_area_approx_l196_196670

noncomputable def combined_area : Float :=
  let rect1_area := 4 * 5
  let rect2_area := 3 * 6
  let tri_area := (5 * 8) / 2
  let trap_area := ((6 + 3) / 2) * 4
  let circ_area := Float.pi * (3.5 * 3.5)
  let para_area := 4 * 6
  rect1_area + rect2_area + tri_area + trap_area + circ_area + para_area

theorem total_area_approx : combined_area ≈ 138.4845 := by
  sorry

end total_area_approx_l196_196670


namespace red_candies_count_l196_196661

theorem red_candies_count :
  ∀ (total_candies blue_candies : ℕ),
  total_candies = 3409 → 
  blue_candies = 3264 →
  total_candies - blue_candies = 145 :=
by
  intros total_candies blue_candies h_total h_blue
  rw [h_total, h_blue]
  exact rfl

end red_candies_count_l196_196661


namespace sqrt_nested_expression_l196_196437

theorem sqrt_nested_expression (N : ℝ) (h : 1 < N) : (sqrt (N * cbrt (N * sqrt N))) = N^(3/4) :=
by
  sorry

end sqrt_nested_expression_l196_196437


namespace quadratic_sum_l196_196216

theorem quadratic_sum (r s : ℝ) (x : ℝ) : (15 * x^2 + 30 * x - 450 = 0) ∧ ((x + r)^2 = s) → (r + s = 32) :=
begin
  sorry
end

end quadratic_sum_l196_196216


namespace parabola_standard_equation_l196_196469

theorem parabola_standard_equation (vertex : (ℝ × ℝ)) (symmetry_axis : ℝ × ℝ → Prop) (focus_line : ℝ × ℝ → Prop) :
    vertex = (0, 0) ∧ symmetry_axis = (λ p, p.1 = 0 ∨ p.2 = 0) ∧ focus_line = (λ p, 2 * p.1 - p.2 - 4 = 0) →
    (∃ x, ∀ (y : ℝ), y^2 = 8 * x) ∨ (∃ y, ∀ (x : ℝ), x^2 = -16 * y) := 
by 
  sorry

end parabola_standard_equation_l196_196469


namespace count_zeros_in_fraction_l196_196096

theorem count_zeros_in_fraction : 
  ∃ n : ℕ, (to_decimal_representation (1 / (2 ^ 3 * 5 ^ 6)) = 0.000008) ∧ (n = 5) :=
by
  sorry

end count_zeros_in_fraction_l196_196096


namespace length_of_segment_AB_is_2_l196_196903

noncomputable def distance (P Q : euclidean_space ℝ (fin 3)) : ℝ :=
  real.sqrt ((P - Q).sum_of_squares)

def symmetric_to_xoy (P : euclidean_space ℝ (fin 3)) : euclidean_space ℝ (fin 3) :=
  ![P 0, P 1, -P 2]

def A : euclidean_space ℝ (fin 3) :=
  ![1, 2, -1]

def B : euclidean_space ℝ (fin 3) :=
  symmetric_to_xoy A

theorem length_of_segment_AB_is_2 : distance A B = 2 := by
  sorry

end length_of_segment_AB_is_2_l196_196903


namespace sqrt_of_sixteen_l196_196395

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l196_196395


namespace monotonous_integers_count_l196_196410

-- Define what it means to be a monotonous integer
def is_monotonous (n : Nat) : Prop :=
  (n < 10) ∨ (∃ (digits : List Nat), 
    (List.reverse digits = digits) ∧ 
    (digits ≠ List.nil) ∧
    (List.all digits fun d => d > 0) ∧
    (List.chain' (<) digits ∨ List.chain' (>) digits))

-- Define the number of monotonous integers
def num_monotonous_integers : Nat :=
  9 + -- one-digit numbers
  2 * (∑ n in Finset.range (9 + 1), Nat.choose 9 n) - 9 -- adjusting for zero-digit placement

theorem monotonous_integers_count : num_monotonous_integers = 1524 :=
by sorry

end monotonous_integers_count_l196_196410


namespace problem1_problem2_l196_196409

theorem problem1 : 2 * Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  -- Proof omitted
  sorry

theorem problem2 : (Real.sqrt 12 - Real.sqrt 24) / Real.sqrt 6 - 2 * Real.sqrt (1/2) = -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l196_196409


namespace marj_money_left_l196_196592

theorem marj_money_left (twenty_bills : ℕ) (five_bills : ℕ) (loose_coins : ℝ) (cake_cost : ℝ) :
  twenty_bills = 2 → five_bills = 3 → loose_coins = 4.5 → cake_cost = 17.5 →
  (20 * twenty_bills + 5 * five_bills + loose_coins - cake_cost = 42) :=
by {
  intros h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  norm_num,
  sorry
}

end marj_money_left_l196_196592


namespace min_distance_eq_5_l196_196957

-- Define the conditions
def condition1 (a b : ℝ) : Prop := b = 4 * Real.log a - a^2
def condition2 (c d : ℝ) : Prop := d = 2 * c + 2

-- Define the function to prove the minimum value
def minValue (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- The main theorem statement
theorem min_distance_eq_5 (a b c d : ℝ) (ha : a > 0) (h1: condition1 a b) (h2: condition2 c d) : 
  ∃ a c b d, minValue a b c d = 5 := 
sorry

end min_distance_eq_5_l196_196957


namespace part1_part2_l196_196584

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def parabola (a b x : ℝ) : ℝ := x^2 + a * x + b

def A1 : ℝ × ℝ := (1, 0)
def x1 : ℝ := 1
def a1 : ℝ := -7

-- Define the point P2(x_2, 2) such that P2 is on the parabola C1 and distance is minimized
def C1 (x : ℝ) : ℝ := parabola (-7) 14 x
def P2 : ℝ × ℝ := sorry

theorem part1 :
  P2 = (3, 2) ∧ C1 = λ x, x^2 - 7 * x + 14 :=
sorry

-- Define recursive conditions for x_n
def a_n (n : ℕ) : ℝ := -2 - 4 * n - 1 / (2^((n - 1 : ℕ)))
def x_n : ℕ → ℝ
| 1 => 1
| n + 2 => sorry

-- Define the point P_(n+1)(x_(n+1), 2^n) such that P_(n+1) is on the parabola C_n and distance is minimized
def P_nplus1 (n : ℕ) : ℝ × ℝ := sorry

theorem part2 (n : ℕ) (h : n ≥ 1) :
  x_n = λ n, 2 * n - 1 := 
sorry

end part1_part2_l196_196584


namespace lines_intersect_perpendicularly_l196_196244

theorem lines_intersect_perpendicularly (α : ℝ) (h1 : sin α ≠ 0) (h2 : cos α ≠ 0) : 
  ∃ P : ℝ × ℝ, (P.1 * sin α + P.2 * cos α + 1 = 0) ∧ (P.1 * cos α - P.2 * sin α + 2 = 0) ∧ 
  (sin α ≠ 0) ∧ (cos α ≠ 0) ─> 
  ((P.1 * (-tan α) + P.2 * (cot α))  = 0) := 
by 
  sorry

end lines_intersect_perpendicularly_l196_196244


namespace total_triangles_correct_l196_196414

-- Define the rectangle and additional constructions
structure Rectangle :=
  (A B C D : Type)
  (midpoint_AB midpoint_BC midpoint_CD midpoint_DA : Type)
  (AC BD diagonals : Type)

-- Hypothesize the structure
variables (rect : Rectangle)

-- Define the number of triangles
def number_of_triangles (r : Rectangle) : Nat := 16

-- The theorem statement
theorem total_triangles_correct : number_of_triangles rect = 16 :=
by
  sorry

end total_triangles_correct_l196_196414


namespace log_five_fraction_l196_196833

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196833


namespace max_sum_n_value_l196_196898

open Nat

-- Definitions for the problem
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

-- Statement of the theorem
theorem max_sum_n_value (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : arithmetic_sequence a d) 
  (h_initial : a 0 > 0) (h_condition : 8 * a 4 = 13 * a 10) : 
  ∃ n, sum_of_first_n_terms a n = max (sum_of_first_n_terms a n) ∧ n = 20 :=
sorry

end max_sum_n_value_l196_196898


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196294

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196294


namespace find_BF_l196_196691

noncomputable theory

-- Definitions for the given conditions
structure Point := (x : ℝ) (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨9, 5.4⟩
def C : Point := ⟨13, 0⟩
def D : Point := ⟨4, -6.5⟩
def E : Point := ⟨4, 0⟩
def F : Point := ⟨9, 0⟩

-- Right angles at A and C
def right_angle (p q r : Point) : Prop :=
  (q.y - p.y) * (r.y - q.y) + (q.x - p.x) * (r.x - q.x) = 0

def quadrilateral_right_angles (A B C D : Point) : Prop :=
  right_angle B A D ∧ right_angle A C D

-- Perpendicular DE and BF to AC
def perpendicular_to_line (p q r : Point) : Prop :=
  (q.x - p.x) * (r.x - p.x) + (q.y - p.y) * (r.y - p.y) = 0

def points_on_line (collinear : list Point) : Prop :=
  ∀ p1 p2, p1 ∈ collinear → p2 ∈ collinear → p1.x = p2.x ∨ p1.y = p2.y

-- Handle the conditions for this specific problem
def conditions : Prop :=
  quadrilateral_right_angles A B C D ∧
  points_on_line [A, E, F, C] ∧
  perpendicular_to_line D E C ∧
  perpendicular_to_line B F A ∧
  (E.x = 4 ∧ E.y = 0) ∧
  (D.x = 4 ∧ D.y = -6.5) ∧
  (C.x = 13 ∧ C.y = 0) ∧
  (F.x = 9 ∧ F.y = 0)

-- The theorem we need to prove
theorem find_BF : conditions → dist B F = 5.4 :=
begin
  sorry
end

end find_BF_l196_196691


namespace num_valid_A_values_l196_196800

theorem num_valid_A_values :
  let valid_solutions := {A : ℤ | 1 ≤ A ∧ A ≤ 9 ∧ ∃ r s : ℕ, 
                                         r ≠ 0 ∧ s ≠ 0 ∧ 
                                         r + s = 10 + A ∧ 
                                         r * s = A * (A - 1)}
  in valid_solutions.to_finset.card = 8 :=
by
  sorry

end num_valid_A_values_l196_196800


namespace distinct_primes_divide_sequence_l196_196574

theorem distinct_primes_divide_sequence (n M : ℕ) (hM : M > n^(n-1)) :
  ∃ (p : ℕ → ℕ), (∀ j, 1 ≤ j ∧ j ≤ n → nat.prime (p j)) ∧
                 (∀ j, 1 ≤ j ∧ j ≤ n → p j ∣ (M + j)) ∧
                 function.injective p := 
sorry

end distinct_primes_divide_sequence_l196_196574


namespace total_bricks_required_l196_196722

/-
Definitions:
- Courtyard dimensions: length and width in meters.
- Brick dimensions: length and width in meters.
- Calculate the area of the courtyard and a single brick.
- Verify that the total number of bricks required is 41,667.
-/

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 20
def brick_length_cm : ℝ := 15
def brick_width_cm : ℝ := 8
def cm_to_m (l : ℝ) : ℝ := l / 100
def brick_length : ℝ := cm_to_m brick_length_cm
def brick_width : ℝ := cm_to_m brick_width_cm
def courtyard_area : ℝ := courtyard_length * courtyard_width
def brick_area : ℝ := brick_length * brick_width
def num_bricks : ℝ := courtyard_area / brick_area

theorem total_bricks_required : num_bricks.ceil = 41667 := by
  sorry

end total_bricks_required_l196_196722


namespace cell_division_l196_196912

theorem cell_division (initial_cells : ℕ) (divisions : ℕ) : initial_cells = 1 → divisions = 3 → 2 ^ divisions = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end cell_division_l196_196912


namespace total_percentage_increase_l196_196150

noncomputable def initialSalary : ℝ := 60
noncomputable def firstRaisePercent : ℝ := 10
noncomputable def secondRaisePercent : ℝ := 15
noncomputable def promotionRaisePercent : ℝ := 20

theorem total_percentage_increase :
  let finalSalary := initialSalary * (1 + firstRaisePercent / 100) * (1 + secondRaisePercent / 100) * (1 + promotionRaisePercent / 100)
  let increase := finalSalary - initialSalary
  let percentageIncrease := (increase / initialSalary) * 100
  percentageIncrease = 51.8 := by
  sorry

end total_percentage_increase_l196_196150


namespace dot_product_AA1_BC1_l196_196554

-- Define the cube data
structure Cube (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
(edge_length : ℝ)
(A A1 B C : V)
(AA1 BC : V)

-- Given the cube with the given conditions
def cube_ABCD_A1B1C1D1 : Cube (EuclideanSpace ℝ (Fin 3)) :=
{ edge_length  := 2,
  A            := ![0, 0, 0],
  A1           := ![0, 0, 2],
  B            := ![2, 0, 0],
  C            := ![2, 2, 0],
  AA1          := ![0, 0, 2],
  BC           := ![0, 2, 0] }

-- Define the dot product calculation to be proven
theorem dot_product_AA1_BC1 : (cube_ABCD_A1B1C1D1.AA1) • (![0, 2, 2] : EuclideanSpace ℝ (Fin 3)) = 4 := by
  sorry

end dot_product_AA1_BC1_l196_196554


namespace digit_in_ten_thousandths_place_of_fraction_l196_196281

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196281


namespace perimeter_of_region_l196_196360

theorem perimeter_of_region (side : ℝ) (r : ℝ) (h_side: side = 4 / real.pi) (h_radius: r = (4 / real.pi) / 2) :
  let quarter_circle_perimeter := (1 / 4) * 2 * real.pi * r,
  total_perimeter := 4 * quarter_circle_perimeter
  in total_perimeter = 4 :=
by
  sorry

end perimeter_of_region_l196_196360


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196297

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196297


namespace max_area_triangle_ABC_l196_196474

theorem max_area_triangle_ABC {a b c : ℝ} (h : a^2 + b^2 + 3 * c^2 = 7) :
  ∃ S, S = (Real.sqrt 7) / 4 ∧ ∀ A, triangle_area A a b c ≤ S :=
sorry

end max_area_triangle_ABC_l196_196474


namespace distance_AB_sqrt_6_l196_196138

-- Define the points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Distance formula between two points in 3D
def distance (A B : Point3D) : ℝ := 
  Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2 + (B.z - A.z) ^ 2)

noncomputable def pointA : Point3D := { x := 1, y := 0, z := 1 }
noncomputable def pointB : Point3D := { x := -1, y := 1, z := 2 }

-- Theorem statement: the distance between pointA and pointB is sqrt(6)
theorem distance_AB_sqrt_6 : distance pointA pointB = Real.sqrt 6 := by
  sorry

end distance_AB_sqrt_6_l196_196138


namespace ihsan_children_l196_196535

theorem ihsan_children :
  ∃ n : ℕ, (n + n^2 + n^3 + n^4 = 2800) ∧ (n = 7) :=
sorry

end ihsan_children_l196_196535


namespace units_digit_calculation_l196_196254

theorem units_digit_calculation : 
  ((33 * (83 ^ 1001) * (7 ^ 1002) * (13 ^ 1003)) % 10) = 9 :=
by
  sorry

end units_digit_calculation_l196_196254


namespace find_c_l196_196330

-- Let a, b, c, d, and e be positive consecutive integers.
variables {a b c d e : ℕ}

-- Conditions: 
def conditions (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a + b = e - 1 ∧
  a * b = d + 1

-- Proof statement
theorem find_c (h : conditions a b c d e) : c = 4 :=
by sorry

end find_c_l196_196330


namespace problem_statement_l196_196876

section problem

variables {n k : ℕ} 
variables {a : fin n → ℕ} (hn : 1 < n)
variables (ha : ∀ i j : fin n, i ≠ j → a i ≠ a j)

def p (i : fin n) : ℤ :=
  ∏ j in (finset.univ.filter (λ j, j ≠ i)),
    (a i - a j)

theorem problem_statement (k : ℕ) 
  (hk : 0 < k) :
  ∑ i, (p i) ^ k / (p i) ∈ ℤ :=
sorry

end problem

end problem_statement_l196_196876


namespace range_of_m_l196_196915

variable {x m : ℝ}

def quadratic (x m : ℝ) : ℝ := x^2 + (m - 1) * x + (m^2 - 3 * m + 1)

def absolute_quadratic (x m : ℝ) : ℝ := abs (quadratic x m)

theorem range_of_m (h : ∀ x ∈ Set.Icc (-1 : ℝ) 0, absolute_quadratic x m ≥ absolute_quadratic (x - 1) m) :
  m = 1 ∨ m ≥ 3 :=
sorry

end range_of_m_l196_196915


namespace range_of_y_l196_196517

theorem range_of_y (y : ℝ) (h₁ : y < 0) (h₂ : ⌈y⌉ * ⌊y⌋ = 110) : -11 < y ∧ y < -10 := 
sorry

end range_of_y_l196_196517


namespace largest_k_of_tree_l196_196935

theorem largest_k_of_tree (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k, (∀ G : SimpleGraph ℕ, G.is_tree ∧ G.order = k → 
       ∃ u v : ℕ, (∀ w : ℕ, w ∈ G.verts → 
       (G.distance u w ≤ m ∨ G.distance v w ≤ n))) ∧ 
       k = min (2 * n + 2 * m + 2) (3 * n + 2) :=
sorry

end largest_k_of_tree_l196_196935


namespace domain_of_f_l196_196067

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

def is_domain (x : ℝ) : Prop := ¬ ∃ k : ℤ, x = Real.pi / 12 + k * (Real.pi / 2)

theorem domain_of_f : ∀ x : ℝ, is_domain x ↔ (f x ≠ ℝ) :=
by
  sorry

end domain_of_f_l196_196067


namespace find_coordinates_B_l196_196797

-- Condition Definitions
def isosceles_triangle (O A B : ℝ × ℝ) : Prop :=
let (xA, yA) := A in O = (0, 0) ∧ xA = 4 ∧ yA = 2 ∧ 
(∃ (xB yB : ℝ), B = (xB, yB) ∧ (xB ≠ 0 ∨ yB ≠ 0) ∧
  (O.1 - B.1)^2 + (O.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2)

def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

def angle_OBA_90 (O A B : ℝ × ℝ) : Prop :=
let OB := (B.1 - O.1, B.2 - O.2) in
let AB := (B.1 - A.1, B.2 - A.2) in
orthogonal OB AB

-- Theorem Statement
theorem find_coordinates_B (B : ℝ × ℝ) :
  ∃ (x y : ℝ), B = (x, y) ∧ let O := (0, 0) in let A := (4, 2) in 
  isosceles_triangle O A B ∧ angle_OBA_90 O A B ∧ 
  (B = (1, 3) ∨ B = (3, -1)) :=
sorry

end find_coordinates_B_l196_196797


namespace values_of_a_l196_196458

theorem values_of_a (α : ℝ) (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), (cos (x₁ - a) - sin (x₁ + 2 * α) = 0) ∧ 
   (cos (x₂ - a) - sin (x₂ + 2 * α) = 0) ∧ 
   (∃ k : ℤ, x₁ - x₂ ≠ k * π)) ↔ 
  (∃ t : ℤ, a = (π * (4 * t + 1) / 6)) := 
sorry

end values_of_a_l196_196458


namespace prob_not_rain_correct_l196_196789

noncomputable def prob_not_rain_each_day (prob_rain : ℚ) : ℚ :=
  1 - prob_rain

noncomputable def prob_not_rain_four_days (prob_not_rain : ℚ) : ℚ :=
  prob_not_rain ^ 4

theorem prob_not_rain_correct :
  prob_not_rain_four_days (prob_not_rain_each_day (2/3)) = 1 / 81 :=
by 
  sorry

end prob_not_rain_correct_l196_196789


namespace find_triangle_area_l196_196564

open Real

-- Definitions for the conditions
variable (a b c : ℝ) -- side lengths
variable (A B C : ℝ) -- angles
variable (S : ℝ) -- area

-- Triangle ABC with angles A, B, C and opposite sides a, b, c
-- Given conditions
axiom cond1 : b / (a + c) = 1 - (sin C / (sin A + sin B))
axiom cond2 : b = 5
axiom cond3 : (a * c) * cos A = 5

-- Query: Find the area S of triangle ABC
theorem find_triangle_area : S = 5 * sqrt(3) / 2 := sorry

end find_triangle_area_l196_196564


namespace sqrt_sixteen_is_four_l196_196397

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l196_196397


namespace inequality_proof_l196_196029

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (2 * (a^3 + b^3 + c^3)) / (a * b * c) + 3 :=
by
  sorry

end inequality_proof_l196_196029


namespace unique_positive_integer_solution_count_l196_196008

theorem unique_positive_integer_solution_count :
  (∀ x : ℕ, (2 * x + 1 > 3 * x - 2) → (4 * x - a > -11) → x = 3) ↔
  (∃! a : ℕ, 19 ≤ a ∧ a ≤ 22): 
sorry

end unique_positive_integer_solution_count_l196_196008


namespace angle_in_second_quadrant_l196_196115

theorem angle_in_second_quadrant (α : ℝ) (h₁ : -2 * Real.pi < α) (h₂ : α < -Real.pi) : 
  α = -4 → (α > -3 * Real.pi / 2 ∧ α < -Real.pi / 2) :=
by
  intros hα
  sorry

end angle_in_second_quadrant_l196_196115


namespace center_of_circle_l196_196804

theorem center_of_circle (x y : ℝ) : 
    (∃ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9) → (x, y) = (2, -3) := 
by sorry

end center_of_circle_l196_196804


namespace student_selection_problem_l196_196010

noncomputable def total_selections : ℕ :=
  let C := Nat.choose
  let A := Nat.factorial
  (C 3 1 * C 3 2 + C 3 2 * C 3 1 + C 3 3) * A 3

theorem student_selection_problem :
  total_selections = 114 :=
by
  sorry

end student_selection_problem_l196_196010


namespace n_pow4_sub_n_pow2_divisible_by_12_l196_196607

theorem n_pow4_sub_n_pow2_divisible_by_12 (n : ℤ) (h : n > 1) : 12 ∣ (n^4 - n^2) :=
by sorry

end n_pow4_sub_n_pow2_divisible_by_12_l196_196607


namespace imo_1990_q31_l196_196697

def A (n : ℕ) : ℕ := sorry -- definition of A(n)
def B (n : ℕ) : ℕ := sorry -- definition of B(n)
def f (n : ℕ) : ℕ := if B n = 1 then 1 else -- largest prime factor of B(n)
  sorry -- logic to find the largest prime factor of B(n)

theorem imo_1990_q31 :
  ∃ (M : ℕ), (∀ n : ℕ, f n ≤ M) ∧ (∀ N, (∀ n, f n ≤ N) → M ≤ N) ∧ M = 1999 :=
by sorry

end imo_1990_q31_l196_196697


namespace odd_function_a_eq_neg1_l196_196516

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2^x + a * 2^(-x)

theorem odd_function_a_eq_neg1 (h : ∀ x : ℝ, f a x = - f a (-x)) : a = -1 :=
sorry

end odd_function_a_eq_neg1_l196_196516


namespace domain_of_function_l196_196802

def domain_conditions (x : ℝ) : Prop :=
  (1 - x ≥ 0) ∧ (x + 2 > 0)

theorem domain_of_function :
  {x : ℝ | domain_conditions x} = {x : ℝ | -2 < x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l196_196802


namespace zeros_in_fraction_representation_l196_196388

theorem zeros_in_fraction_representation : 
  ∀ (x y : ℕ) (h : x = 15 ∧ y = 3), 
  let n := x^15 * y in 
  (number_of_zeros_after_decimal (1 / n) = 15) :=
by
  intros x y h
  let n := x ^ 15 * y
  -- The function number_of_zeros_after_decimal isn't a real Lean function.
  -- You might need a custom definition of number_of_zeros_after_decimal
  -- appropriate for your use case.
  -- We use sorry to leave the proof incomplete.
  sorry

-- Custom placeholder definition; actual implementation would be required.
def number_of_zeros_after_decimal (r : ℝ) : ℕ := sorry

end zeros_in_fraction_representation_l196_196388


namespace remainder_of_2_pow_33_mod_9_l196_196247

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end remainder_of_2_pow_33_mod_9_l196_196247


namespace benny_added_march_l196_196784

theorem benny_added_march :
  let january := 19 
  let february := 19
  let march_total := 46
  (march_total - (january + february) = 8) :=
by
  let january := 19
  let february := 19
  let march_total := 46
  sorry

end benny_added_march_l196_196784


namespace sqrt_sixteen_is_four_l196_196398

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l196_196398


namespace boys_brought_the_same_car_l196_196604

-- Definitions for the properties of toy cars:
structure ToyCar :=
(size : string)    -- size can be "small" or "big"
(color : string)   -- color can be "green", "blue" etc.
(trailer : bool)   -- trailer can be true (with trailer) or false (without trailer)

-- Conditions
def M1 : ToyCar := { size := "unknown", color := "unknown", trailer := true }
def M2 : ToyCar := { size := "small", color := "unknown", trailer := false }
def M3 : ToyCar := { size := "unknown", color := "green", trailer := false }

def V1 : ToyCar := { size := "unknown", color := "unknown", trailer := false }
def V2 : ToyCar := { size := "small", color := "green", trailer := true }

def K1 : ToyCar := { size := "big", color := "unknown", trailer := false }
def K2 : ToyCar := { size := "small", color := "blue", trailer := true }

-- The final answer
def common_car : ToyCar := { size := "big", color := "green", trailer := false }

-- Proof statement
theorem boys_brought_the_same_car : 
  (∃ c : ToyCar, c = M1 ∨ c = M2 ∨ c = M3) ∧
  (∃ c : ToyCar, c = V1 ∨ c = V2) ∧
  (∃ c : ToyCar, c = K1 ∨ c = K2) ∧
  (common_car = c) :=
sorry

end boys_brought_the_same_car_l196_196604


namespace inv_f_zero_l196_196579

noncomputable def f (a b x : Real) : Real := 1 / (2 * a * x + 3 * b)

theorem inv_f_zero (a b : Real) (ha : a ≠ 0) (hb : b ≠ 0) : f a b (1 / (3 * b)) = 0 :=
by 
  sorry

end inv_f_zero_l196_196579


namespace unique_integer_sequence_exists_l196_196612

theorem unique_integer_sequence_exists
  (a : ℕ → ℕ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 > 1)
  (h₃ : ∀ (n : ℕ), n > 0 →
    a (n + 1) * (a (n + 1) - 1) = 
    (a n * a (n + 2)) / (real.cbrt (a n * a (n + 2) - 1) + 1) - 1) :
  ∃! (a : ℕ → ℕ) (h₁' : a 1 = 1) (h₂' : a 2 > 1), 
    ∀ (n : ℕ), n > 0 → 
    a (n + 1) * (a (n + 1) - 1) = 
    (a n * a (n + 2)) / (real.cbrt (a n * a (n + 2) - 1) + 1) - 1 := 
sorry

end unique_integer_sequence_exists_l196_196612


namespace contrapositive_example_l196_196225

theorem contrapositive_example (x : ℝ) : (x = 1 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 1) :=
by
  sorry

end contrapositive_example_l196_196225


namespace mutually_exclusive_necessary_for_complementary_l196_196509

variables {Ω : Type} -- Define the sample space type
variables (A1 A2 : Ω → Prop) -- Define the events as predicates over the sample space

-- Define mutually exclusive events
def mutually_exclusive (A1 A2 : Ω → Prop) : Prop :=
∀ ω, A1 ω → ¬ A2 ω

-- Define complementary events
def complementary (A1 A2 : Ω → Prop) : Prop :=
∀ ω, (A1 ω ↔ ¬ A2 ω)

-- The proof problem: Statement 1 is a necessary but not sufficient condition for Statement 2
theorem mutually_exclusive_necessary_for_complementary (A1 A2 : Ω → Prop) :
  (mutually_exclusive A1 A2) → (complementary A1 A2) → (mutually_exclusive A1 A2) ∧ ¬ (complementary A1 A2 → mutually_exclusive A1 A2) :=
sorry

end mutually_exclusive_necessary_for_complementary_l196_196509


namespace power_addition_l196_196016

theorem power_addition {a m n : ℝ} (h1 : a^m = 2) (h2 : a^n = 8) : a^(m + n) = 16 :=
sorry

end power_addition_l196_196016


namespace difference_eq_neg_subtrahend_implies_minuend_zero_l196_196953

theorem difference_eq_neg_subtrahend_implies_minuend_zero {x y : ℝ} (h : x - y = -y) : x = 0 :=
sorry

end difference_eq_neg_subtrahend_implies_minuend_zero_l196_196953


namespace radius_of_larger_circle_l196_196647

theorem radius_of_larger_circle (r : ℝ) (large_radius : ℝ) :
  (large_radius = 4 * r) ∧ (18² + ((large_radius / 2) - r)² = (large_radius / 2)² + (large_radius / 2)²) → large_radius = 36 := 
by
  sorry

end radius_of_larger_circle_l196_196647


namespace eccentricity_range_find_a_value_l196_196070

-- Define the given hyperbola and line intersection problem
variables {a x y : ℝ}
variables {A B P : ℝ × ℝ}

-- Given conditions
def is_hyperbola (a : ℝ) : Prop := a > 0 ∧ (∃ x y, x^2 / a^2 - y^2 = 1)
def is_line (x y : ℝ) : Prop := x + y = 1
def distinct_intersection (a : ℝ) : Prop := ∃ A B : ℝ × ℝ, A ≠ B ∧ (∃ x₁ y₁ x₂ y₂, (x₁, y₁) = A ∧ (x₂, y₂) = B ∧ (x₁^2 / a^2 - y₁^2 = 1) ∧ (x₂^2 / a^2 - y₂^2 = 1) ∧ (x₁ + y₁ = 1) ∧ (x₂ + y₂ = 1))

-- Define eccentricity
def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 / a^2 + 1)

-- Theorem (I): Range of the eccentricity of the hyperbola
theorem eccentricity_range (a : ℝ) (h : is_hyperbola a) (h' : distinct_intersection a) : (eccentricity a) > Real.sqrt 6 / 2 ∧ (eccentricity a) ≠ Real.sqrt 2 := 
sorry

-- Given \(\overrightarrow{PA} = \frac{5}{12}\overrightarrow{PB}\), prove that \(a = \frac{17}{13}\)
def vector_relationship (P A B : ℝ × ℝ) : Prop := 
  P.2 = 1 ∧ P.1 = 0 ∧ ((A.1, A.2 - 1) = (5 / 12) • (B.1, B.2 - 1))

theorem find_a_value (A B : ℝ × ℝ) (h : vector_relationship (0, 1) A B) (h_eq : is_hyperbola a) :
  a = 17 / 13 :=
sorry

end eccentricity_range_find_a_value_l196_196070


namespace clock_time_at_entry_l196_196700

theorem clock_time_at_entry :
  (∀ t : ℕ, ∃ chimes : ℕ, 
    (t % 30 = 0 → chimes = 1) ∧
    (t % 60 = 0 → chimes = t / 60 ∈ {1, 2, 3, ..., 12})) ∧
  ((∃ t : ℕ, t % 30 = 0 ∧ t / 30 ∈ {0, 1, 2, 3}) →
   // The actual chime pattern:
   (1 chime at entering) ∧
   (1 chime after 30 mins) ∧
   (1 chime after another 30 mins) ∧
   (1 chime after another 30 mins)) →
  (∃ t : ℕ, t = 12 * 60) := -- Time at entry is 12:00
sorry

end clock_time_at_entry_l196_196700


namespace tan_family_total_cost_l196_196625

-- Define the number of people in each age group and respective discounts
def num_children : ℕ := 2
def num_adults : ℕ := 2
def num_seniors : ℕ := 2

def price_adult_ticket : ℝ := 10
def discount_senior : ℝ := 0.30
def discount_child : ℝ := 0.20
def group_discount : ℝ := 0.10

-- Calculate the cost for each group with discounts applied
def price_senior_ticket := price_adult_ticket * (1 - discount_senior)
def price_child_ticket := price_adult_ticket * (1 - discount_child)

-- Calculate the total cost of tickets before group discount
def total_cost_before_group_discount :=
  (price_senior_ticket * num_seniors) +
  (price_child_ticket * num_children) +
  (price_adult_ticket * num_adults)

-- Check if the family qualifies for group discount and apply if necessary
def total_cost_after_group_discount :=
  if (num_children + num_adults + num_seniors > 5)
  then total_cost_before_group_discount * (1 - group_discount)
  else total_cost_before_group_discount

-- Main theorem statement
theorem tan_family_total_cost : total_cost_after_group_discount = 45 := by
  sorry

end tan_family_total_cost_l196_196625


namespace function_passes_through_point_l196_196986

theorem function_passes_through_point :
  (∃ (a : ℝ), a = 1 ∧ (∀ (x y : ℝ), y = a * x + a → y = x + 1)) →
  ∃ x y : ℝ, x = -2 ∧ y = -1 ∧ y = x + 1 :=
by
  sorry

end function_passes_through_point_l196_196986


namespace batsman_average_increase_l196_196713

theorem batsman_average_increase :
  ∀ (A : ℝ), (10 * A + 110 = 11 * 60) → (60 - A = 5) :=
by
  intros A h
  -- Proof goes here
  sorry

end batsman_average_increase_l196_196713


namespace cos_triple_angle_l196_196948

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end cos_triple_angle_l196_196948


namespace book_cost_l196_196694

theorem book_cost (C_1 C_2 : ℝ)
  (h1 : C_1 + C_2 = 420)
  (h2 : C_1 * 0.85 = C_2 * 1.19) :
  C_1 = 245 :=
by
  -- We skip the proof here using sorry.
  sorry

end book_cost_l196_196694


namespace quadratic_root_and_coeff_l196_196109

theorem quadratic_root_and_coeff (x m : ℝ) (h : x^2 + 2 * x + m = 0) (hx : x = -2) : 
  m = 0 ∧ ∃ (r : ℝ), r ≠ -2 ∧ (x + r = -2) :=
by {
  -- assume the given root and substitute to find m
  have hm : (x = -2) → (4 - 4 + m = 0) := by {
    intro hx,
    rw [hx],
    norm_num,
  },
  -- prove m = 0
  have hm_val : m = 0 := by {
    rw [hx] at h,
    simp * at *,
  },
  -- now see that the sum of the roots simplifies accordingly
  use [0],
  split,
  { -- show the root is not -2 itself
    linarith,
  },
  { -- show the sum of the roots is -2
    linarith,
  },
  sorry -- proof completeness is left as an exercise for the reader
}

end quadratic_root_and_coeff_l196_196109


namespace angle_DAB_EAB_l196_196761

open Real EuclideanGeometry

-- Definitions based on the conditions
variables (A B C D E : Point)
variables (x : ℝ)
variables (AB CD AE : Line)

-- Conditions
def quadrilateral_condition (AB CD AE : Line) : Prop :=
  parallel AB CD ∧ ∃ E, line_contains CD E ∧ perpendicular AE CD

-- Theorem statement
theorem angle_DAB_EAB :
  quadrilateral_condition AB CD AE →
  angle A B C D = (90:ℝ) ∧ angle A E B = (90:ℝ) :=
by sorry

end angle_DAB_EAB_l196_196761


namespace minimum_phi_for_symmetry_l196_196956

def f (x : ℝ) : ℝ := sin (2 * x) + cos (2 * x)

def shifted_f (x : ℝ) (φ : ℝ) : ℝ := sqrt 2 * sin (2 * x + 2 * φ + π / 4)

def is_symmetric_about_y_axis (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

theorem minimum_phi_for_symmetry :
  ∃ φ > 0, is_symmetric_about_y_axis (shifted_f x φ) ∧ (∀ φ' > 0, is_symmetric_about_y_axis (shifted_f x φ') → φ' ≥ φ) ∧ φ = π / 8 :=
sorry

end minimum_phi_for_symmetry_l196_196956


namespace first_player_wins_l196_196600

theorem first_player_wins :
  let numbers := List.range' 1 20 1 
  let result := numbers.foldl (+) 0
  result % 2 = 0 :=
by
  let numbers := List.range' 1 20 1 
  let result := numbers.foldl (+) 0
  sorry

end first_player_wins_l196_196600


namespace smallest_tangent_circle_l196_196337

def line1 : ℝ → ℝ → Prop := λ x y, x - y - 4 = 0
def circle1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 2*x - 2*y = 0

def circle2 : ℝ → ℝ → Prop := λ x y, (x + 1)^2 + (y + 1)^2 = 2

theorem smallest_tangent_circle:
  (∀ x y, circle2 x y → line1 x y ∨ circle1 x y) →
  (∃ x y, circle2 x y) :=
by
  intros h
  sorry

end smallest_tangent_circle_l196_196337


namespace pens_needed_to_achieve_profit_l196_196749

noncomputable def cost_price_per_pen : ℝ := 7 / 4
noncomputable def selling_price_per_pen : ℝ := 12 / 5
noncomputable def profit_per_pen : ℝ := selling_price_per_pen - cost_price_per_pen
noncomputable def desired_profit : ℝ := 50
noncomputable def number_of_pens_required : ℝ := desired_profit / profit_per_pen

theorem pens_needed_to_achieve_profit :
  ceil number_of_pens_required = 77 := by
  sorry

end pens_needed_to_achieve_profit_l196_196749


namespace N_is_composite_l196_196810

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end N_is_composite_l196_196810


namespace probability_final_roll_six_l196_196735

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l196_196735


namespace white_fraction_of_large_cube_l196_196724

-- Conditions
def largeCubeEdge : Nat := 4
def smallCubeEdge : Nat := 1
def totalCubes : Nat := 64
def whiteCubes : Nat := 48
def blackCubes : Nat := 16
def blackCorners : Nat := 8 -- This is inferred: 8 corners, each with a black cube
def blackEdges : Nat := 12 -- This is inferred: 12 edges

-- Surface area of a cube with given edge length
def surfaceArea (edge : Nat) : Nat := 6 * (edge * edge)

-- Number of black cubic faces exposed
def blackFacesExposed : Nat :=
  blackCorners * 3 + (blackEdges - blackCorners) -- 3 faces per cube at corners and 1 face per cube on edges excluding corners

-- Number of white cubic faces exposed
def whiteFacesExposed (totalSurfaceArea : Nat) (blackFaces : Nat) : Nat :=
  totalSurfaceArea - blackFaces

-- Fraction of white surface area
def whiteSurfaceFraction (totalSurfaceArea whiteSurfaceArea : Nat) : Rat :=
  whiteSurfaceArea (totalSurfaceArea : ℚ)

theorem white_fraction_of_large_cube :
  whiteSurfaceFraction (surfaceArea largeCubeEdge) (whiteFacesExposed (surfaceArea largeCubeEdge) blackFacesExposed) = 5 8 :=
by
  sorry

end white_fraction_of_large_cube_l196_196724


namespace area_of_triangle_HFG_l196_196547

noncomputable def calculate_area_of_triangle (A B C : (ℝ × ℝ)) :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_HFG :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 4)
  let D := (0, 4)
  let E := (2, 2)
  let F := (1, 4)
  let G := (0, 2)
  let H := ((2 + 1 + 0) / 3, (2 + 4 + 2) / 3)
  calculate_area_of_triangle H F G = 2/3 :=
by
  sorry

end area_of_triangle_HFG_l196_196547


namespace probability_six_greater_than_five_over_six_l196_196746

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l196_196746


namespace solution_interval_l196_196618

theorem solution_interval (x : ℝ) : 
  (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 7) ∨ (7 < x) ↔ 
  ((x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0) := sorry

end solution_interval_l196_196618


namespace binomial_expansion_calculation_l196_196106

theorem binomial_expansion_calculation :
  let a : Fin 2014 → ℝ := (λ i, (1 - 2 * x)^2013.coeff i)
  a 0 = 1 →
  (∑ i in Finset.range 2014, a (i + 1) / 2^(i + 2)) = -1/2 := 
sorry

end binomial_expansion_calculation_l196_196106


namespace heavyTailedPermutationsCount_l196_196757

def isHeavyTailed (a : List ℕ) : Prop :=
  a.length = 6 ∧
  (a.take 3).sum < (a.drop 3).sum

theorem heavyTailedPermutationsCount :
  (Finset.univ.filter (λ a : List ℕ, a.perm [1, 2, 3, 4, 5, 6] ∧ isHeavyTailed a)).card = 72 :=
by
  sorry

end heavyTailedPermutationsCount_l196_196757


namespace total_fish_in_lake_l196_196870

theorem total_fish_in_lake:
  let 
    white_ducks := 3 
    black_ducks := 7 
    multico_ducks := 6 
    fish_per_white_duck := 5 
    fish_per_black_duck := 10 
    fish_per_multico_duck := 12 
  in 
  white_ducks * fish_per_white_duck + 
  black_ducks * fish_per_black_duck + 
  multico_ducks * fish_per_multico_duck = 157 := 
by 
  sorry

end total_fish_in_lake_l196_196870


namespace monotonicity_and_extrema_l196_196923

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f(x1) < f(x2)) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end monotonicity_and_extrema_l196_196923


namespace compare_negatives_l196_196412

theorem compare_negatives : -2 > -3 :=
by
  sorry

end compare_negatives_l196_196412


namespace sequence_general_term_eq_l196_196457

-- Defining the sequence based on given conditions
noncomputable def a : ℕ → ℝ
| 0     := 2
| 1     := 5 / 2
| (n+2) := a (n + 1) * (a n ^ 2 - 2) - 5 / 2

-- The closed-form/general term of the sequence
noncomputable def a_closed (n : ℕ) :=
  2 ^ ((2 ^ n - (-1) ^ n) / 3 : ℝ) + 2 ^ (-(2 ^ n - (-1) ^ n) / 3 : ℝ)

-- The theorem stating equivalence of the sequence and its general term
theorem sequence_general_term_eq :
  ∀ n : ℕ, a n = a_closed n :=
by
  -- Proof goes here
  sorry

end sequence_general_term_eq_l196_196457


namespace tank_ratio_l196_196769

theorem tank_ratio (V1 V2 : ℝ) (h1 : 0 < V1) (h2 : 0 < V2) (h1_full : 3 / 4 * V1 - 7 / 20 * V2 = 0) (h2_full : 1 / 4 * V2 + 7 / 20 * V2 = 3 / 5 * V2) :
  V1 / V2 = 7 / 9 :=
by
  sorry

end tank_ratio_l196_196769


namespace geometric_sequence_general_formula_l196_196889

section
variable (a : ℕ+ → ℝ)
variable (q : ℝ)
variable (b : ℕ+ → ℝ)
variable (S : ℕ+ → ℝ)

def is_geometric_sequence (a : ℕ+ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = a n * q

def positive_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, 0 < a n

def geometric_condition1 (a : ℕ+ → ℝ) (q : ℝ) : Prop :=
  a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25

def geometric_mean_condition (a : ℕ+ → ℝ) : Prop :=
  sqrt (a 3 * a 5) = 2

def general_formula (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n = 2 ^ (6 - 2 * n)

def b_definition (a : ℕ+ → ℝ) (b : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, b n = Real.log (2, a n)

def S_formula (S : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, S n = n * (5 - n)

theorem geometric_sequence_general_formula
  (a : ℕ+ → ℝ) (q : ℝ) (b : ℕ+ → ℝ) (S : ℕ+ → ℝ)
  (h0 : positive_geometric_sequence a)
  (h1 : q ∈ Set.Ioo 0 1)
  (h2 : geometric_condition1 a q)
  (h3 : geometric_mean_condition a)
  (h4 : is_geometric_sequence a q)
  (h5 : b_definition a b) :
  general_formula a ∧ S_formula S :=
by
  sorry
end

end geometric_sequence_general_formula_l196_196889


namespace log_base_5_of_inv_sqrt_5_l196_196825

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196825


namespace print_shop_X_charge_l196_196865

-- Define the given conditions
def cost_per_copy_X (x : ℝ) : Prop := x > 0
def cost_per_copy_Y : ℝ := 2.75
def total_copies : ℕ := 40
def extra_cost_Y : ℝ := 60

-- Define the main problem
theorem print_shop_X_charge (x : ℝ) (h : cost_per_copy_X x) :
  total_copies * cost_per_copy_Y = total_copies * x + extra_cost_Y → x = 1.25 :=
by
  sorry

end print_shop_X_charge_l196_196865


namespace average_weight_l196_196264

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l196_196264


namespace area_of_given_triangle_l196_196276

def point := (ℝ × ℝ)

def triangle := {a b c : point // a ≠ b ∧ b ≠ c ∧ a ≠ c}

def height (a b : point) := real.abs (a.2 - b.2)

def base (b c : point) := real.abs (b.1 - c.1)

def area_of_triangle (b h : ℝ) := 1 / 2 * b * h

theorem area_of_given_triangle :
  let a := (2, 3) in
  let b := (2, -4) in
  let c := (7, -4) in
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  area_of_triangle (base b c) (height a b) = 17.5 := 
by
  intros _ 
  simp [a, b, c, height, base, area_of_triangle]
  sorry

end area_of_given_triangle_l196_196276


namespace shape_with_circular_cross_sections_is_sphere_l196_196681

-- Define the shapes as types
inductive Shape
| cylinder
| cone
| sphere
| cone_with_circular_base

-- Define the property that a shape has circular cross-sections
def has_circular_cross_sections (s : Shape) : Prop :=
  match s with
  | Shape.sphere => true
  | _ => false

-- The theorem statement claiming that only Sphere has constant circular cross-sections
theorem shape_with_circular_cross_sections_is_sphere :
  ∀ s : Shape, has_circular_cross_sections s ↔ s = Shape.sphere :=
by 
  intro s
  cases s
  case cylinder { simp [has_circular_cross_sections] }
  case cone { simp [has_circular_cross_sections] }
  case sphere { simp [has_circular_cross_sections] }
  case cone_with_circular_base { simp [has_circular_cross_sections] }
  sorry

end shape_with_circular_cross_sections_is_sphere_l196_196681


namespace num_distinct_exponentiations_l196_196795

-- Define the custom up-arrow operator for exponentiation
def up (a b : ℕ) : ℕ := a ^ b

-- All possible parenthesizations of 3 up 3 up 3 up 2
def exp1 := up 3 (up 3 (up 3 2))
def exp2 := up 3 (up (up 3 3) 2)
def exp3 := up (up (up 3 3) 3) 2
def exp4 := up (up 3 (up 3 3)) 2
def exp5 := up (up 3 3) (up 3 2)

-- The original expression (for reference)
def original := up 3 (up 3 (up 3 2))

-- The proof problem equivalent to the original problem statement
theorem num_distinct_exponentiations : 
  {exp1, exp2, exp3, exp4, exp5}.erase original = 4 :=
by
  sorry

end num_distinct_exponentiations_l196_196795


namespace find_y_l196_196994

theorem find_y (y : ℚ) (h : 6 * y + 3 * y + 4 * y + 2 * y + 1 * y + 5 * y = 360) : y = 120 / 7 := 
sorry

end find_y_l196_196994


namespace fraction_power_equals_l196_196788

theorem fraction_power_equals :
  (5 / 7) ^ 7 = (78125 : ℚ) / 823543 := 
by
  sorry

end fraction_power_equals_l196_196788


namespace intersection_S_T_l196_196933

def U := ℝ
def S := { y : ℝ | ∃ x : ℝ, y = 2^x }
def T := { x : ℝ | log (x - 1) < 0 }

theorem intersection_S_T : S ∩ T = { x : ℝ | 1 < x ∧ x < 2 } :=
by 
  -- Proof goes here
  sorry

end intersection_S_T_l196_196933


namespace min_value_of_sum_squares_l196_196107

theorem min_value_of_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 := sorry

end min_value_of_sum_squares_l196_196107


namespace frequency_converges_to_probability_l196_196632

-- Definitions of frequency and probability.
def frequency (n : ℕ) (occurrences : ℕ) : ℝ := (occurrences : ℝ) / (n : ℝ)
def probability (event : Prop) : ℝ := sorry -- Assume we have a definition of probability.

-- Proposition: As the number of trials increases, frequency converges to probability.
theorem frequency_converges_to_probability (event : Prop) (n : ℕ) (occurrences : ℕ) :
  (frequency n occurrences) = (probability event) := 
sorry

end frequency_converges_to_probability_l196_196632


namespace consecutive_sum_divisible_l196_196610

theorem consecutive_sum_divisible (a : ℕ → ℤ) (n : ℕ) (h: n > 0): 
  ∃ p q : ℕ, p < q ∧ p < n ∧ q ≤ n ∧ (∑ i in finset.range(q) \ finset.range(p), a i) % n = 0 := sorry

end consecutive_sum_divisible_l196_196610


namespace value_of_a_value_of_sin_A_plus_pi_over_4_l196_196559

section TriangleABC

variables {a b c A B : ℝ}
variables (h_b : b = 3) (h_c : c = 1) (h_A_eq_2B : A = 2 * B)

theorem value_of_a : a = 2 * Real.sqrt 3 :=
sorry

theorem value_of_sin_A_plus_pi_over_4 : Real.sin (A + π / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end TriangleABC

end value_of_a_value_of_sin_A_plus_pi_over_4_l196_196559


namespace total_fish_in_lake_l196_196871

theorem total_fish_in_lake:
  let 
    white_ducks := 3 
    black_ducks := 7 
    multico_ducks := 6 
    fish_per_white_duck := 5 
    fish_per_black_duck := 10 
    fish_per_multico_duck := 12 
  in 
  white_ducks * fish_per_white_duck + 
  black_ducks * fish_per_black_duck + 
  multico_ducks * fish_per_multico_duck = 157 := 
by 
  sorry

end total_fish_in_lake_l196_196871


namespace increase_in_circumference_l196_196581

variable {d : ℝ}  -- original diameter

theorem increase_in_circumference {Q : ℝ} 
(h : 2 * π) 
: Q = (π * (d + 2 * π)) - π * d → Q = 2 * π^2 := 
by sorry

end increase_in_circumference_l196_196581


namespace perimeter_of_nonagon_l196_196943

-- Definitions based on the conditions
def sides := 9
def side_length : ℝ := 2

-- The problem statement in Lean
theorem perimeter_of_nonagon : sides * side_length = 18 := 
by sorry

end perimeter_of_nonagon_l196_196943


namespace solve_for_x_l196_196214

theorem solve_for_x (x : ℚ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 :=
sorry

end solve_for_x_l196_196214


namespace angle_at_7_30_l196_196379

def angle_between_hands (h m : ℕ) : ℝ :=
  abs ((60 * h - 11 * m) / 2)

theorem angle_at_7_30 : angle_between_hands 7 30 = 45 :=
by
  sorry

end angle_at_7_30_l196_196379


namespace work_completion_l196_196717

theorem work_completion (days_A : ℕ) (days_B : ℕ) (hA : days_A = 14) (hB : days_B = 35) :
  let rate_A := 1 / (days_A : ℚ)
  let rate_B := 1 / (days_B : ℚ)
  let combined_rate := rate_A + rate_B
  let days_AB := 1 / combined_rate
  days_AB = 10 := by
  sorry

end work_completion_l196_196717


namespace probability_at_least_four_same_is_correct_l196_196863

noncomputable def probability_at_least_four_same (dice : Fin 5 → Fin 6) : ℚ :=
  -- Probability that all five dice show the same value
  (1 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ) +
  -- Probability that exact four dice show the same value and the fifth is different
  5 * ((1/6 : ℚ) * (1/6 : ℚ) * (1/6 : ℚ)) * (5/6 : ℚ)

theorem probability_at_least_four_same_is_correct :
  ∀ (dice : Fin 5 → Fin 6), probability_at_least_four_same dice = 13/648 :=
by
  intro dice
  -- The proof would go here
  sorry

end probability_at_least_four_same_is_correct_l196_196863


namespace intersection_points_between_C1_and_C2_are_zero_l196_196987

def curve_C1 (t : ℝ) : ℝ × ℝ := (t + 1 / t, 2)
def curve_C2 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

theorem intersection_points_between_C1_and_C2_are_zero :
  ∃! (P : ℝ × ℝ), (∃ t, P = curve_C1 t) ∧ (∃ θ, P = curve_C2 θ) → P = 0 :=
by
  sorry

end intersection_points_between_C1_and_C2_are_zero_l196_196987


namespace sum_of_ages_l196_196158

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end sum_of_ages_l196_196158


namespace bus_distance_time_relation_l196_196716

theorem bus_distance_time_relation (t : ℝ) :
    (0 ≤ t ∧ t ≤ 1 → s = 60 * t) ∧
    (1 < t ∧ t ≤ 1.5 → s = 60) ∧
    (1.5 < t ∧ t ≤ 2.5 → s = 80 * (t - 1.5) + 60) :=
sorry

end bus_distance_time_relation_l196_196716


namespace log_base5_of_inverse_sqrt5_l196_196831

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196831


namespace fgf_3_equals_108_l196_196178

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem fgf_3_equals_108 : f (g (f 3)) = 108 := 
by
  sorry

end fgf_3_equals_108_l196_196178


namespace selection_exists_l196_196203

theorem selection_exists (s : Finset ℕ) (hs : s.card = 100) (h_positive : ∀ x ∈ s, 0 < x) :
  ∃ S T : Finset ℕ, S ∪ T = s ∧ S ∩ T = ∅ ∧ S.card = 98 ∧ T.card = 2 ∧ (∑ x in S, x) % (∑ y in T, y) ≠ 0 :=
by sorry

end selection_exists_l196_196203


namespace count_zeros_in_fraction_l196_196095

theorem count_zeros_in_fraction : 
  ∃ n : ℕ, (to_decimal_representation (1 / (2 ^ 3 * 5 ^ 6)) = 0.000008) ∧ (n = 5) :=
by
  sorry

end count_zeros_in_fraction_l196_196095


namespace last_locker_opened_l196_196714

theorem last_locker_opened :
  (∃ lockers : Finset ℕ,
     (∀ locker ∈ lockers, 1 ≤ locker ∧ locker ≤ 1024) ∧
     lockers.nonempty ∧
     lockers = (Finset.range 1025).filter (λ n, n % 512 = 342))
  → ∃ locker, locker = 854 :=
by 
  sorry

end last_locker_opened_l196_196714


namespace digit_in_ten_thousandths_place_l196_196309

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196309


namespace dot_product_is_six_l196_196460

def a : ℝ × ℝ := (-2, 4)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_is_six : (a.1 * b.1 + a.2 * b.2) = 6 := 
by 
  -- definition and proof logic follows
  sorry

end dot_product_is_six_l196_196460


namespace determine_x_l196_196426

theorem determine_x (x : ℕ) 
  (hx1 : x % 6 = 0) 
  (hx2 : x^2 > 196) 
  (hx3 : x < 30) : 
  x = 18 ∨ x = 24 := 
sorry

end determine_x_l196_196426


namespace total_number_of_valid_guesses_l196_196344

noncomputable def valid_guesses (digits : Multiset ℕ) (prizes : list ℕ) : ℕ :=
  (Multiset.card digits).choose 3 * 12

theorem total_number_of_valid_guesses :
  valid_guesses {2, 2, 2, 2, 4, 4, 4} [D, E, F] = 420 :=
by {
  sorry
}

end total_number_of_valid_guesses_l196_196344


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196301

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196301


namespace repeating_decimal_fraction_l196_196450

theorem repeating_decimal_fraction : (real.mk (rat.mk_pnat (nat.succ 3 * (1 + 9 * 10)))) (nat.succ 27) = rat.mk 4 11 :=
by
-- proof can be filled here using Calc and necessary steps, but currently skipped
sorry

end repeating_decimal_fraction_l196_196450


namespace frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l196_196640

-- Definitions of conditions
def grasshopper_jump : ℕ := 19
def mouse_jump_frog (frog_jump : ℕ) : ℕ := frog_jump + 20
def mouse_jump_grasshopper : ℕ := grasshopper_jump + 30

-- The proof problem statement
theorem frog_jumps_10_inches_more_than_grasshopper (frog_jump : ℕ) :
  mouse_jump_frog frog_jump = mouse_jump_grasshopper → frog_jump = 29 :=
by
  sorry

-- The ultimate question in the problem
theorem frog_jumps_10_inches_farther_than_grasshopper : 
  (∃ (frog_jump : ℕ), frog_jump = 29) → (frog_jump - grasshopper_jump = 10) :=
by
  sorry

end frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l196_196640


namespace max_profit_achieved_when_x_is_1_l196_196340

noncomputable def revenue (x : ℕ) : ℝ := 30 * x - 0.2 * x^2
noncomputable def fixed_costs : ℝ := 40
noncomputable def material_cost (x : ℕ) : ℝ := 5 * x
noncomputable def profit (x : ℕ) : ℝ := revenue x - (fixed_costs + material_cost x)
noncomputable def marginal_profit (x : ℕ) : ℝ := profit (x + 1) - profit x

theorem max_profit_achieved_when_x_is_1 :
  marginal_profit 1 = 24.40 :=
by
  -- Skip the proof
  sorry

end max_profit_achieved_when_x_is_1_l196_196340


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196296

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196296


namespace tangent_line_at_zero_no_zeros_r_geq_one_l196_196926

-- Definition f(x) = e^x
def f (x : ℝ) := Real.exp x

-- Definition g(x) = mx + n
def g (m n x : ℝ) := m * x + n

-- Definition h(x) = f(x) - g(x)
def h (m n x : ℝ) := f x - g m n x

-- Prove m + n = 2 given the tangent line of h(x) at x = 0 passes through (1, 0)
theorem tangent_line_at_zero (m n : ℝ) (h_passes : h m n 0 = 1 - n) :
    m + n = 2 := by sorry

-- Prove the range of m given n = 0 and no zeros in (-1, +∞)
theorem no_zeros (m : ℝ) (h_zeros : ∀ x > -1, h m 0 x ≠ 0) :
    m ≥ -Real.exp (-1) ∧ m < Real.exp 1 := by sorry

-- Definition r(x) = 1 / f(x) + nx / g(x)
def r (m x : ℝ) := (1 / f x) + ((4 * m * x) / g m (4 * m) x)

-- Prove r(x) ≥ 1 for x ≥ 0 given n = 4m and m > 0
theorem r_geq_one (m : ℝ) (m_pos : m > 0) (x : ℝ) (x_geq_zero : x ≥ 0) :
    r m x ≥ 1 := by sorry

end tangent_line_at_zero_no_zeros_r_geq_one_l196_196926


namespace ratio_of_perimeters_l196_196048

-- Definitions and conditions for the problem
namespace TriangleSimilarity

variables {A B C A1 B1 C1 : Type} [MetricSpace A] [MetricSpace B] 
  [MetricSpace C] [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]

variables (ΔABC ΔA1B1C1 : Triangle A) -- Triangle ABC and A1B1C1

-- Similarity condition with ratio 1:2
def is_similar (ΔABC ΔA1B1C1 : Triangle A) : Prop :=
  (Triangle.is_similar ΔABC ΔA1B1C1 ∧ 
   ∃ (r : ℝ), r = 1/2 ∧ 
   (∀ (a b c : ℝ), 
      Triangle.side_len ΔABC a b c → 
      Triangle.side_len ΔA1B1C1 (2*a) (2*b) (2*c)))

-- Theorem statement
theorem ratio_of_perimeters (ΔABC ΔA1B1C1 : Triangle A)
  (h : is_similar ΔABC ΔA1B1C1) :
  Triangle.perimeter ΔABC / Triangle.perimeter ΔA1B1C1 = 1 / 2 :=
sorry -- Proof to be filled

end TriangleSimilarity

end ratio_of_perimeters_l196_196048


namespace Q1_Q2_l196_196930

open Set

-- Definitions of sets A and B and the "length" of an interval
def setA (t : ℝ) : Set ℝ := {2, real.log2 t}
def setB : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}

-- Length of an interval
def interval_length (a b : ℝ) : ℝ := b - a

-- Question 1: Determine the value of t when the length of set A is 3
theorem Q1 (t : ℝ) (H : interval_length 2 (real.log2 t) = 3) : t = 32 := by
  sorry

-- Question 2: Determine the range of values of t such that A is a subset of B
theorem Q2 (t : ℝ) (H : setA t ⊆ setB) : 4 < t ∧ t < 32 := by
  sorry

end Q1_Q2_l196_196930


namespace skew_lines_angle_distance_l196_196224

-- Define points in 3D space
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

-- Define the points as per the problem statement
def C : Point3D := ⟨0, 0, 0⟩
def S : Point3D := ⟨0, 0, 2⟩
def M : Point3D := ⟨√2, √6, 0⟩ -- Midpoint of BC
def K : Point3D := ⟨0, 2√6, 0⟩ -- Midpoint of AB

-- Define the vectors
def vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def SM := vector S M
def CK := vector C K

-- Define the dot product of vectors
def dot_product (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Define the magnitude of a vector
def magnitude (v : Point3D) : ℝ :=
  sqrt (v.x^2 + v.y^2 + v.z^2)

-- Define the cosine of the angle between two vectors
def cos_theta (u v : Point3D) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

-- Define the distance formula from a point to a plane
def distance_to_plane (P : Point3D) (a b c d : ℝ) : ℝ :=
  abs (a * P.x + b * P.y + c * P.z + d) / sqrt (a^2 + b^2 + c^2)

-- Statement in Lean 4
theorem skew_lines_angle_distance :
  cos_theta SM CK = 1 / sqrt 2 ∧
  distance_to_plane C (1 / √2) 0 (1 / 2) (-1) = 2 / sqrt 3 :=
by
  sorry

end skew_lines_angle_distance_l196_196224


namespace sum_of_fractions_l196_196790

theorem sum_of_fractions :
  (3 / 9) + (7 / 12) = (11 / 12) :=
by 
  sorry

end sum_of_fractions_l196_196790


namespace sqrt_seq_ineq_l196_196162

noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| (n+1) := 1 + ∑ k in Finset.range(n+1), (k+1) * a_seq k

theorem sqrt_seq_ineq (n : ℕ) (h : n > 1) : Real.sqrt[a_seq n] < (n + 1) / 2 :=
sorry

end sqrt_seq_ineq_l196_196162


namespace largest_angle_is_120_degrees_l196_196560

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Conditions given in the problem
def cond1 (a b c : ℝ) : Prop := a + 2b + 2c = a^2
def cond2 (a b c : ℝ) : Prop := a + 2b - 2c = -3

-- Hypothesis: The conditions are satisfied
axiom h_cond1 : cond1 a b c
axiom h_cond2 : cond2 a b c

-- Proof goal: The largest angle C is 120 degrees
theorem largest_angle_is_120_degrees (h1 : cond1 a b c) (h2 : cond2 a b c) : C = 120 :=
by
  sorry

end largest_angle_is_120_degrees_l196_196560


namespace eccentricity_of_ellipse_l196_196478

theorem eccentricity_of_ellipse :
  ∀ (A B : ℝ × ℝ) (has_axes_intersection : A.2 = 0 ∧ B.2 = 0) 
    (product_of_slopes : ∀ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B → (P.2 / (P.1 - A.1)) * (P.2 / (P.1 + B.1)) = -1/2),
  ∃ (e : ℝ), e = 1 / Real.sqrt 2 :=
by
  sorry

end eccentricity_of_ellipse_l196_196478


namespace minimum_distance_l196_196350

def parabola (x : ℝ) : ℝ := x^2

def line_through_point (k x : ℝ) : ℝ := k * (x - 1) + 3

def circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

theorem minimum_distance (k x1 x2 : ℝ) :
  (line_through_point k x1 = parabola x1) →
  (line_through_point k x2 = parabola x2) →
  x1 ≠ x2 →
  let Q_x := k / 2 in
  let Q_y := k - 3 in
  ∃ d : ℝ, d = real.sqrt 5 - 2 ∧ ∀ x y, circle x y → dist (Q_x, Q_y) (x, y) = d := by
  sorry

end minimum_distance_l196_196350


namespace range_of_m_l196_196502

open Real

noncomputable def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

lemma log_decreasing (a : ℝ) (ha : 0 < a ∧ a < 1) (x y : ℝ) (h : x < y) : log a y < log a x :=
begin
  sorry -- omitted proof
end

noncomputable def t (x m : ℝ) : ℝ := x^2 - 2 * m * x + 3

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Ioo ( -∞ : ℝ) (1 : ℝ), tendsto_on (log (1/2)) (t x m) (Ioo ( -∞ : ℝ) (1 : ℝ))) →
  (1 ≤ m ∧ m < 2) :=
begin
  sorry -- omitted proof
end

end range_of_m_l196_196502


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196303

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196303


namespace fraction_zero_iff_x_neg_one_l196_196119

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h : 1 - |x| = 0) (h_non_zero : 1 - x ≠ 0) : x = -1 :=
sorry

end fraction_zero_iff_x_neg_one_l196_196119


namespace evaluate_expression_l196_196435

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 1 - 2 * (Real.sin θ) ^ 2
noncomputable def cofunction_identity (θ : ℝ) : ℝ := Real.sin (Real.pi / 2 - θ)

theorem evaluate_expression :
    (let cos_10 := cos_double_angle (5 * Real.pi / 180) in
    let cos_85 := cofunction_identity (5 * Real.pi / 180) in
    let expr := (Real.sqrt (1 - cos_10)) / cos_85
    in expr = Real.sqrt 2) := by
    sorry

end evaluate_expression_l196_196435


namespace log_base5_of_inverse_sqrt5_l196_196832

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196832


namespace remainder_N_mod_1000_l196_196007

def base_three_digit_sum (n : ℕ) : ℕ :=
  (n.digits 3).sum

def base_eight_digit_sum (n : ℕ) : ℕ :=
  (n.digits 8).sum

def f (n : ℕ) : ℕ := base_three_digit_sum n

def g (n : ℕ) : ℕ := base_eight_digit_sum (f n)

def N : ℕ := (Nat.find (λ n => 
  let g_val := g n
  g_val.digits 16 ∃d ∈ g_val.digits 16, 9 < d))

theorem remainder_N_mod_1000 : N % 1000 = 862 :=
by
  sorry

end remainder_N_mod_1000_l196_196007


namespace tan_cot_eq_num_solutions_l196_196803

theorem tan_cot_eq_num_solutions :
  (∀ θ ∈ Ioo 0 (2 * Real.pi), tan (3 * Real.pi * cos θ) = cot (3 * Real.pi * cos (Real.pi / 3 - θ))) →
  finset.card {θ ∈ Ioo (0 : ℝ) (2 * Real.pi) | tan (3 * Real.pi * cos θ) = cot (3 * Real.pi * cos (Real.pi / 3 - θ))}.to_finset = 14 :=
by
  sorry

end tan_cot_eq_num_solutions_l196_196803


namespace equivalent_determinant_l196_196177

-- Define vectors and dot/cross product operations in Lean
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)

-- Definition of determinant D
def det_D : ℝ := ⟪a, b × c⟫

-- Definition of determinant E
def det_E (a b c d : V) : ℝ := matrix.det ![![⟪d, a × b⟫, ⟪d, b × c⟫, ⟪d, c × a⟫]]

-- The proof statement
theorem equivalent_determinant (a b c d : V) :
  det_E a b c d = (∥d∥)^3 * (det_D a b c)^3 :=
sorry

end equivalent_determinant_l196_196177


namespace coefficient_of_reciprocal_x_l196_196444

theorem coefficient_of_reciprocal_x :
  let expr := ((1 - x^2)^4 * (x + 1)^5 / x^5) in
  (coefficient_of (1/x) expr) = -29 :=
by
  sorry

end coefficient_of_reciprocal_x_l196_196444


namespace max_profit_sum_correct_l196_196565

-- Define the conditions
def city_count : Nat := 100000000
def route_count : Nat := 5050

-- Define the prosperity of a city as the number of routes originating from it
def prosperity (cities: Fin city_count) : Nat :=
  sorry -- The prosperity function should properly assign the number of routes (degree) for each city.

-- Define the profit of a flight route as the product of the prosperity of the two cities it connects
def profit (city1 city2: Fin city_count) : Nat :=
  prosperity city1 * prosperity city2

-- Define the maximum possible sum of profits of the 5050 flight routes
def max_profit_sum : Nat :=
  50500000

-- The theorem to prove the maximum possible sum of the profits
theorem max_profit_sum_correct :
  ∃ (routes : Fin route_count → (Fin city_count × Fin city_count)),
    (∀ i j, routes i ≠ routes j) ∧
    (∀ r, r ∈ routes → r.1 ≠ r.2) ∧
    (∑ r in routes, profit r.1 r.2) = max_profit_sum :=
begin
  sorry
end

end max_profit_sum_correct_l196_196565


namespace find_angle_C_find_max_value_l196_196893

-- Defining the variables and conditions
variables {a b c A B C : Real}
-- Sine function for angles
variable [Sine A, Sine B, Sine C]

-- Assuming the given equation
axiom cond : a * sin A + b * sin B = c * sin C + sqrt 2 * a * sin B

-- (I) Proving angle C
theorem find_angle_C : C = π / 4 :=
by
  sorry

-- (II) Proving the maximum value of the given expression
theorem find_max_value : 
  (√3 * sin A - cos (B + π / 4)) ≤ 2 :=
by
  sorry

end find_angle_C_find_max_value_l196_196893


namespace complex_multiplication_l196_196885

theorem complex_multiplication 
  (i : ℂ) 
  (h : i = complex.I) :
  (\frac{1}{2} + \frac{\sqrt{3}}{2} * i) * (\frac{\sqrt{3}}{2} + \frac{1}{2} * i) = i :=
by
  sorry

end complex_multiplication_l196_196885


namespace coeff_of_x_105_in_P_l196_196849

-- Definition of the polynomial P(x)
def P (x : ℝ) : ℝ :=
  (x - 1) * (x^2 - 2) * (x^3 - 3) * (x^4 - 4) * (x^5 - 5) * (x^6 - 6) * (x^7 - 7) * 
  (x^8 - 8) * (x^9 - 9) * (x^10 - 10) * (x^11 - 11) * (x^12 - 12) * (x^13 - 13) * 
  (x^14 - 14) * (x^15 - 15)

-- Goal: find the coefficient of x^105 in P(x)
theorem coeff_of_x_105_in_P :
  coefficient_of (x^105) (P x) = c :=
sorry

end coeff_of_x_105_in_P_l196_196849


namespace proof_problem_l196_196968

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l196_196968


namespace smallest_draws_correct_l196_196377

noncomputable def smallest_draws (k : ℕ) (m : ℕ) (n : ℕ → ℕ) : ℕ :=
  let s := (List.range k).find (λ i, n i ≥ m) | 0
  1 + (m - 1) * (k - s + 1) + ((List.range s).map n).sum

theorem smallest_draws_correct (k m : ℕ) (n : ℕ → ℕ) (h : ∀ i, i < k → n i ≥ 0) (hm : n (List.range k).find (λ i, n i ≥ m) ≥ m) :
  ∃ s, n s ≥ m ∧ smallest_draws k m n = 1 + (m - 1) * (k - s + 1) + ∑ i in (List.range s), n i :=
by
  sorry

end smallest_draws_correct_l196_196377


namespace largest_among_options_l196_196322

theorem largest_among_options :
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  D > A ∧ D > B ∧ D > C ∧ D > E := by
{
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  sorry
}

end largest_among_options_l196_196322


namespace sqrt_of_16_is_4_l196_196407

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l196_196407


namespace price_of_second_box_l196_196777

noncomputable def price_of_first_box : ℝ := 25
noncomputable def contacts_in_first_box : ℕ := 50
noncomputable def contacts_in_second_box : ℕ := 99
noncomputable def price_per_contact_first_box : ℝ := price_of_first_box / contacts_in_first_box
noncomputable def chosen_price_per_contact : ℝ := 1 / 3

theorem price_of_second_box :
  chosen_price_per_contact < price_per_contact_first_box →
  let price_per_contact_second_box := chosen_price_per_contact in
  let total_price_second_box := price_per_contact_second_box * contacts_in_second_box in
  total_price_second_box = 32.67 :=
by
  intros h
  let price_per_contact_second_box := chosen_price_per_contact
  let total_price_second_box := price_per_contact_second_box * contacts_in_second_box
  have : total_price_second_box = 32.67 := sorry
  exact this

end price_of_second_box_l196_196777


namespace number_of_valid_pairs_l196_196995

theorem number_of_valid_pairs : 
  let pairs := [(x, y) | x <- [1..1000], y <- [1..1000], (x^2 + y^2) % 7 = 0] in
    pairs.length = 20164 := 
by 
  sorry

end number_of_valid_pairs_l196_196995


namespace student_test_score_l196_196365

variable (C I : ℕ)

theorem student_test_score  
  (h1 : C + I = 100)
  (h2 : C - 2 * I = 64) :
  C = 88 :=
by
  -- Proof steps should go here
  sorry

end student_test_score_l196_196365


namespace probability_of_specific_balls_drawn_l196_196536

/--
In a box, there are 15 red, 9 blue, and 6 green balls. Six balls are drawn at random. 
Prove that the probability of drawing 1 green, 2 blue, and 3 red balls is approximately 24/145.
-/
theorem probability_of_specific_balls_drawn (total_red total_blue total_green total_balls drawn_balls : ℕ) 
    (red_drawn blue_drawn green_drawn : ℕ)
    (h_red : total_red = 15) 
    (h_blue : total_blue = 9) 
    (h_green : total_green = 6) 
    (h_total : total_balls = 30) 
    (h_drawn : drawn_balls = 6) 
    (h_red_drawn : red_drawn = 3) 
    (h_blue_drawn : blue_drawn = 2) 
    (h_green_drawn : green_drawn = 1) :
    let total_outcomes := Nat.choose total_balls drawn_balls,
        favorable_red := Nat.choose total_red red_drawn,
        favorable_blue := Nat.choose total_blue blue_drawn,
        favorable_green := Nat.choose total_green green_drawn,
        favorable_outcomes := favorable_red * favorable_blue * favorable_green,
        prob := (favorable_outcomes : ℚ) / total_outcomes
    in prob ≈ (24 : ℚ) / 145 :=
by
  sorry

end probability_of_specific_balls_drawn_l196_196536


namespace sum_of_ages_l196_196156

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196156


namespace digit_in_ten_thousandths_place_of_fraction_l196_196282

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196282


namespace find_k_collinear_l196_196577

variable (e1 e2 : Vector ℝ)
variable (h_noncollinear : ¬ collinear ℝ {e1, e2})
variable (k : ℝ)
variable (h_collinear : collinear ℝ {e1 - 4 • e2, k • e1 + e2})

theorem find_k_collinear :
  k = -1 / 4 :=
sorry

end find_k_collinear_l196_196577


namespace difference_of_squares_l196_196683

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end difference_of_squares_l196_196683


namespace smallest_m_for_T_divisibility_l196_196866

def h (x : ℕ) : ℕ := nat.find_greatest (λ i, 2^i ∣ x) x

def T (m : ℕ) : ℕ := ∏ i in (finset.range (2^m + 1)), h i

-- Prove that T_m = ∏_{j=1}^{2^m} h(j) is divisible by 2^1000 when m = 6.
theorem smallest_m_for_T_divisibility :
  ∀ m > 0, (T m ∣ 2^1000) → m = 6 :=
sorry

end smallest_m_for_T_divisibility_l196_196866


namespace proof_problem_l196_196492

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin (2 * x - (Real.pi / 3)) + b

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := b * cos (a * x + (Real.pi / 6))

theorem proof_problem :
  (∃ a b : ℝ, a > 0 ∧ 
     (∀ x : ℝ, -a + b = -5 ∧ a + b = 1) ∧ 
     g a b (λ x: ℝ, x = 0) = 2 ∧ 
     (∀ k : ℤ, ∃ x : ℝ, x = (5 * Real.pi / 18) + (2 * k * Real.pi / 3)))
  :=
  sorry

end proof_problem_l196_196492


namespace sqrt_sixteen_equals_four_l196_196403

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l196_196403


namespace championship_team_groups_exists_l196_196718

noncomputable def n := 1000
noncomputable def k := 7

theorem championship_team_groups_exists (teams : Finₓ n → Finₓ n → Prop):
  (∀ i j, i ≠ j → (teams i j ∨ teams j i)) →
  (∃ (A B : Finₓ n → Prop), 
    (set.card (set_of A) = k) ∧
    (set.card (set_of B) = k) ∧
    (∀ i j, A i → B j → teams j i)) := 
by {
  sorry
}

end championship_team_groups_exists_l196_196718


namespace quadratic_root_condition_l196_196951

theorem quadratic_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
by 
  sorry

end quadratic_root_condition_l196_196951


namespace uniqueSumEqualNumber_l196_196679

noncomputable def sumPreceding (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem uniqueSumEqualNumber :
  ∃! n : ℕ, sumPreceding n = n := by
  sorry

end uniqueSumEqualNumber_l196_196679


namespace triangle_right_triangle_l196_196918

theorem triangle_right_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 3) (h2 : b = 4) (h3 : c^2 - 10 * c + 25 = 0) : 
  a^2 + b^2 = c^2 :=
by
  -- We know the values of a, b, and c by the conditions
  sorry

end triangle_right_triangle_l196_196918


namespace cost_price_of_apple_l196_196357

-- Define the given conditions SP = 20, and the relation between SP and CP.
variables (SP CP : ℝ)
axiom h1 : SP = 20
axiom h2 : SP = CP - (1/6) * CP

-- Statement to be proved.
theorem cost_price_of_apple : CP = 24 :=
by
  sorry

end cost_price_of_apple_l196_196357


namespace log_simplification_l196_196705

theorem log_simplification : log 4 + 2 * log 5 = 2 := 
  sorry

end log_simplification_l196_196705


namespace problem1_problem2_problem3_l196_196141

open_locale big_operators

-- Define the sequence of points
variables {P : ℕ → (ℤ × ℤ)} (n : ℕ)

-- Problem 1: Prove coordinates of P1
theorem problem1
  (h0 : P 0 = (0, 1))
  (h1 : ∃ Δx Δy, x1' = (fst P 0) + Δx ∧ y1' = (snd P 0) + Δy ∧ 
      0 < Δx ∧ Δx < Δy ∧ |Δx| * |Δy| = 2) :
  P 1 = (1, 3) :=
sorry

-- Problem 2: Prove value of n when Pn is on the line y = 3x - 8
theorem problem2
  (h0 : P 0 = (0, 1))
  (h1 : ∀ k, 1 ≤ k → k ≤ n → Δx k = 1)
  (h2 : ∀ k, 0 ≤ k → k ≤ n → y (k + 1) > y k)
  (h3 : P n = (fst (P n), 3 * fst (P n) - 8)) :
  n = 9 :=
sorry

-- Problem 3: Prove the maximum value of the sum of x coordinates
theorem problem3
  (h0 : P 0 = (0, 0))
  (h1 : ∃ Δy, (sum (λ k, Δy k) 1 2016) = 100) :
  (sum (λ k, fst (P k)) 0 2016) = 4066272 :=
sorry

end problem1_problem2_problem3_l196_196141


namespace fish_in_lake_l196_196873

theorem fish_in_lake (white_ducks black_ducks multicolor_ducks : ℕ) 
                     (fish_per_white fish_per_black fish_per_multicolor : ℕ)
                     (h1 : fish_per_white = 5)
                     (h2 : fish_per_black = 10)
                     (h3 : fish_per_multicolor = 12)
                     (h4 : white_ducks = 3)
                     (h5 : black_ducks = 7)
                     (h6 : multicolor_ducks = 6) :
                     (white_ducks * fish_per_white + black_ducks * fish_per_black + multicolor_ducks * fish_per_multicolor) = 157 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end fish_in_lake_l196_196873


namespace adults_attended_l196_196371

def adult_ticket_cost : ℕ := 25
def children_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400

theorem adults_attended (A C: ℕ) (h1 : adult_ticket_cost * A + children_ticket_cost * C = total_receipts)
                       (h2 : A + C = total_attendance) : A = 120 :=
by
  sorry

end adults_attended_l196_196371


namespace scientific_notation_of_384000_l196_196222

theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l196_196222


namespace parallelogram_ABCD_area_l196_196043

noncomputable def f (x : ℝ) : ℝ := log (x + 1) / log 2 - log (x - 1) / log 2

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 31, y := f 31}
def B : Point := {x := 5 / 3, y := 2}

def is_on_f (p : Point) : Prop := p.y = f p.x

def parallelogram_area (A B C D : Point) : ℝ :=
  let u : Point := {x := B.x - A.x, y := B.y - A.y}
  let v : Point := {x := D.x - A.x, y := D.y - A.y}
  abs (u.x * v.y - u.y * v.x)

theorem parallelogram_ABCD_area :
  ∀ C D : Point, is_on_f A ∧ is_on_f B ∧ is_on_f C ∧ is_on_f D →
                parallelogram_area A B C D = 26 / 3 :=
by
  intros
  sorry


end parallelogram_ABCD_area_l196_196043


namespace batsman_average_l196_196327

theorem batsman_average (A : ℕ) (total_runs_before : ℕ) (new_score : ℕ) (increase : ℕ)
  (h1 : total_runs_before = 11 * A)
  (h2 : new_score = 70)
  (h3 : increase = 3)
  (h4 : 11 * A + new_score = 12 * (A + increase)) :
  (A + increase) = 37 :=
by
  -- skipping the proof with sorry
  sorry

end batsman_average_l196_196327


namespace remainder_when_divided_by_x_minus_1_remainder_when_divided_by_x_squared_minus_1_l196_196003

noncomputable def f (x : ℝ) : ℝ := x^243 + x^81 + x^27 + x^9 + x^3 + 1

theorem remainder_when_divided_by_x_minus_1 : (eval 1 f) = 6 := 
by sorry

theorem remainder_when_divided_by_x_squared_minus_1 : (x : ℝ) :=
  (f (x) % (x^2 - 1)) = (5 * x + 1) :=
by sorry

end remainder_when_divided_by_x_minus_1_remainder_when_divided_by_x_squared_minus_1_l196_196003


namespace probability_six_greater_than_five_over_six_l196_196748

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l196_196748


namespace problem_solution_l196_196108

-- Define the imaginary unit with its periodic properties
axiom i : ℂ
axiom i_squared : i^2 = -1

-- Define the expression to be proved
def expression : ℂ := (1 / i) + (1 / (i^3)) + (1 / (i^5)) + (1 / (i^7)) + (1 / (i^9))

-- State the theorem
theorem problem_solution : expression = -i :=
by
  sorry

end problem_solution_l196_196108


namespace find_subtracted_value_l196_196768

-- Define the conditions
def chosen_number := 124
def result := 110

-- Lean statement to prove
theorem find_subtracted_value (x : ℕ) (y : ℕ) (h1 : x = chosen_number) (h2 : 2 * x - y = result) : y = 138 :=
by
  sorry

end find_subtracted_value_l196_196768


namespace length_of_BC_in_triangle_l196_196142

theorem length_of_BC_in_triangle (
  A B C: ℝ)
  (cos_eq_one : real.cos (2 * A - B) = 1)
  (sin_eq_one : real.sin (A + B) = 1)
  (AB_pos : 0 ≤ AB)
  (AB_eq_four : AB = 4) 
: real.dist B C = 2 :=
sorry

end length_of_BC_in_triangle_l196_196142


namespace probability_final_roll_six_l196_196734

def roll_die : Int → Bool
| n => n >= 1 ∧ n <= 6

theorem probability_final_roll_six
    (p : Fin 6 → ℝ)
    (h : p 0 + p 1 + p 2 + p 3 + p 4 + p 5 = 1)
    (S : Fin 6 → ℝ)
    (n : ℕ)
    (Y : ℕ → ℝ)
    (H : Y n + S 6 >= 2019) :
  (∑ k in (Finset.range 6).map Fin.mk, (p k) / (7 - (k + 1))) > 1/6 :=
by
  sorry

end probability_final_roll_six_l196_196734


namespace sum_of_surface_areas_l196_196033

-- Defining the conditions
variables (AB AC AA₁ : ℝ)
variables (AB_perp_AC : AB ⊥ AC)

-- Given the specific values for the right prism
def prism_conditions : Prop :=
  AB = 3 ∧ AC = 4 ∧ AA₁ = 2 ∧ AB_perp_AC

-- The theorem statement
theorem sum_of_surface_areas (h : prism_conditions AB AC AA₁ AB_perp_AC) : 
  sum_surface_areas = 33 * π :=
sorry

end sum_of_surface_areas_l196_196033


namespace circle_condition_l196_196954

theorem circle_condition (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2*x - 4*y + m = 0) → m < 5 :=
by
  -- Define constants and equation representation
  let d : ℝ := -2
  let e : ℝ := -4
  let f : ℝ := m
  -- Use the condition for the circle equation
  have h : d^2 + e^2 - 4*f > 0 := sorry
  -- Prove the inequality
  sorry

end circle_condition_l196_196954


namespace intersection_AB_CD_l196_196032

open real

variables {P : Type} [euclidean_space P ℝ]

-- Definitions for the points on the parabola
def parabola (x : ℝ) : P := ⟨x, x^2⟩

-- Definitions for points M, A, B, C, D
def M : P := ⟨1, 1⟩
def A (x1 : ℝ) : P := parabola x1
def B (x2 : ℝ) : P := parabola x2
def C (x3 : ℝ) : P := parabola x3
def D (x4 : ℝ) : P := parabola x4

-- Slope function between two points
def slope (p1 p2 : P) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Perpendicular condition
def perpendicular (p1 p2 p3 : P) : Prop := slope p1 p2 * slope p1 p3 = -1

-- Define the equations of AB and CD based on points A, B, C, D
def line_eq (p1 p2 : P) (x : ℝ) : ℝ := slope p1 p2 * (x - p1.1) + p1.2

-- Prove the intersection points of AB and CD
theorem intersection_AB_CD (x1 x2 x3 x4 : ℝ) 
  (h1 : perpendicular M (A x1) (B x2)) 
  (h2 : perpendicular M (C x3) (D x4)) : 
  ∃ E : P, E.1 = -1 ∧ E.2 = 2 :=
begin
  use ⟨-1, 2⟩,
  split; refl,
end

end intersection_AB_CD_l196_196032


namespace probability_of_forming_triangle_l196_196656

theorem probability_of_forming_triangle :
  let lengths := [1, 3, 5, 7, 9] in
  let valid_combinations := [[3, 5, 7], [3, 7, 9], [5, 7, 9]] in
  let total_combinations := 10 in
  length valid_combinations / total_combinations = 3 / 10 :=
by
  -- Definitions
  let lengths := [1, 3, 5, 7, 9]
  let valid_combinations := [[3, 5, 7], [3, 7, 9], [5, 7, 9]]
  let total_combinations := 10
  
  -- Main assertion
  have proportion : length valid_combinations / total_combinations = 3 / 10 := sorry

  exact proportion

end probability_of_forming_triangle_l196_196656


namespace number_of_ways_correct_l196_196030

def S : Set ℕ := {1, 2, 3, 4, 5, 6}

def grid_6x6 (diamondsuit : ℕ → ℕ → ℕ) :=
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 → diamondsuit i j ∈ S

def condition1 (diamondsuit : ℕ → ℕ → ℕ) :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 6 → diamondsuit i i = i

def condition2 (diamondsuit : ℕ → ℕ → ℕ) :=
  ∀ (i j k l : ℕ), 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ 1 ≤ l ∧ l ≤ 6 →
  diamondsuit (diamondsuit i j) (diamondsuit k l) = diamondsuit i l

noncomputable def number_of_ways : ℕ := 
  if ∃ (diamondsuit : ℕ → ℕ → ℕ), grid_6x6 diamondsuit ∧ condition1 diamondsuit ∧ condition2 diamondsuit 
  then 42 
  else 0

theorem number_of_ways_correct : number_of_ways = 42 :=
sorry

end number_of_ways_correct_l196_196030


namespace arithmetic_sequence_a4_possible_values_l196_196036

theorem arithmetic_sequence_a4_possible_values (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 * a 5 = 9)
  (h3 : a 2 = 3) : 
  a 4 = 3 ∨ a 4 = 7 := 
by 
  sorry

end arithmetic_sequence_a4_possible_values_l196_196036


namespace kelsey_pie_chart_l196_196120

noncomputable def cherry_pie_angle (total_students chocolate_apple_blueberry cherry_fraction : ℕ) : ℕ :=
  let remaining_students := total_students - chocolate_apple_blueberry
  let cherry_students := (remaining_students * cherry_fraction) / 2
  (cherry_students * 360) / total_students

theorem kelsey_pie_chart :
  (let total_students := 45 in
   let chocolate := 15 in
   let apple := 10 in
   let blueberry := 9 in
   let chocolate_apple_blueberry := chocolate + apple + blueberry in
   let cherry_fraction := 1 in
   cherry_pie_angle total_students chocolate_apple_blueberry cherry_fraction = 40) :=
by
  sorry

end kelsey_pie_chart_l196_196120


namespace sum_factors_60_l196_196317

theorem sum_factors_60 : ∑ i in (finset.filter (| i | ∃ (a b c : ℕ), (2^a * 3^b * 5^c = i ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) (finset.range 61)), i = 168 :=
by
  sorry

end sum_factors_60_l196_196317


namespace largest_two_numbers_l196_196372

def a : Real := 2^(1/2)
def b : Real := 3^(1/3)
def c : Real := 8^(1/8)
def d : Real := 9^(1/9)

theorem largest_two_numbers : 
  (max (max (max a b) c) d = b) ∧ 
  (max (max a c) d = a) := 
sorry

end largest_two_numbers_l196_196372


namespace sqrt_sixteen_is_four_l196_196400

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l196_196400


namespace incorrect_statement_about_R2_l196_196877

variables {n : ℕ} (x y : Fin n → ℝ)
-- Assume residuals are zero if and only if R² is 1.
def residuals_zero_iff_R2_one (residuals : Fin n → ℝ) (R2 : ℝ) : Prop :=
  (∀ i, residuals i = 0) ↔ R2 = 1

-- Assume the model with a smaller sum of squared residuals has better fitting.
def better_fitting_iff_smaller_sum_squares (s1 s2 : Fin n → ℝ) : Prop :=
  ∑ i, (s1 i)^2 < ∑ i, (s2 i)^2

-- Assume if r = -0.9362, there's a linear correlation R² > 0.75.
def linear_correlation_if_r_high (r : ℝ) : Prop :=
  r = -0.9362 → R2 > 0.75

-- Prove that the statement regarding R² and model fitting is incorrect.
theorem incorrect_statement_about_R2 (residuals : Fin n → ℝ) (R2 : ℝ) (r : ℝ) :
  residuals_zero_iff_R2_one residuals R2 →
  (∑ i, (residuals i)^2 < ∑ i, (residuals i)^2) →
  linear_correlation_if_r_high r →
  ¬ (∀ R2, R2 < 1 → better_fitting_iff_smaller_sum_squares residuals residuals) :=
by sorry

end incorrect_statement_about_R2_l196_196877


namespace least_positive_integer_n_l196_196855

theorem least_positive_integer_n :
  (∑ k in finset.range (144 - 35), 1 / (Real.sin (35 + k * (1 : ℝ)) * Real.sin (35 + (k + 1) * (1 : ℝ)))) =
  1 / Real.sin 71 :=
by {
  sorry
}

end least_positive_integer_n_l196_196855


namespace smallest_abs_diff_l196_196229

theorem smallest_abs_diff (a b : ℕ) (h_distinct: a ≠ b)
  (h1 : (a + b) % 2 = 0) (h2 : Nat.sqrt (a * b) ^ 2 = a * b)
  (h3: (2 * a * b) % (a + b) = 0) :
  |a - b| = 3 :=
sorry

end smallest_abs_diff_l196_196229


namespace sum_tetrahedral_formula_l196_196201

def tetrahedral_number (n : ℕ) : ℕ :=
  Nat.choose (n + 2) 3

def sum_tetrahedral (k : ℕ) : ℕ :=
  (Finset.range k).sum (λ i => tetrahedral_number (i + 1))

theorem sum_tetrahedral_formula (k : ℕ) : sum_tetrahedral (k + 1) = Nat.choose (k + 3) 4 :=
by
  sorry

end sum_tetrahedral_formula_l196_196201


namespace karen_average_speed_correct_l196_196568

def karen_time_duration : ℚ := (22 : ℚ) / 3
def karen_distance : ℚ := 230

def karen_average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed_correct :
  karen_average_speed karen_distance karen_time_duration = (31 + 4/11 : ℚ) :=
by
  sorry

end karen_average_speed_correct_l196_196568


namespace move_factors_inside_sqrt_l196_196597

theorem move_factors_inside_sqrt (x : ℝ) (hx : x < 0) : x * real.sqrt (-1 / x) = -real.sqrt (-x) :=
sorry

end move_factors_inside_sqrt_l196_196597


namespace sqrt_sixteen_equals_four_l196_196401

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l196_196401


namespace find_x_l196_196484

variables {x y z d e f : ℝ}
variables (h1 : xy / (x + 2 * y) = d)
variables (h2 : xz / (2 * x + z) = e)
variables (h3 : yz / (y + 2 * z) = f)

theorem find_x :
  x = 3 * d * e * f / (d * e - 2 * d * f + e * f) :=
sorry

end find_x_l196_196484


namespace fractions_integer_or_fractional_distinct_l196_196213

theorem fractions_integer_or_fractional_distinct (a b : Fin 6 → ℕ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_irreducible : ∀ i, Nat.gcd (a i) (b i) = 1)
  (h_sum_eq : (Finset.univ : Finset (Fin 6)).sum a = (Finset.univ : Finset (Fin 6)).sum b) :
  ¬ ∀ i j : Fin 6, i ≠ j → ((a i / b i = a j / b j) ∨ (a i % b i / b i = a j % b j / b j)) :=
sorry

end fractions_integer_or_fractional_distinct_l196_196213


namespace mn_value_l196_196012

theorem mn_value (m n : ℤ) (h1 : 3 ^ m = 1 / 27) (h2 : (1 / 2) ^ n = 16) : m ^ n = 1 / 81 := 
by sorry

end mn_value_l196_196012


namespace problem_statement_l196_196171

noncomputable def f : ℕ → ℕ := sorry

theorem problem_statement (n s : ℕ) (h1 : ∀ a b : ℕ, 3 * f (a^2 + b^2 + a) = (f a)^2 + (f b)^2 + 3 * f a) 
  (hn : n = {x : ℕ | ∃ a : ℕ, f 49 = a}.toFinset.card)
  (hs : s = {x : ℕ | ∃ a : ℕ, f 49 = a}.toFinset.sum (λ x, x)) : 
  n * s = 444 :=
sorry

end problem_statement_l196_196171


namespace limit_zero_l196_196900

open Filter

variable (a : ℕ → ℝ)

-- Given condition
def condition := tendsto (fun n => a (n + 1) - (a n) / 2) atTop (𝓝 0)

-- Prove the statement
theorem limit_zero (h : condition a) : tendsto a atTop (𝓝 0) :=
sorry

end limit_zero_l196_196900


namespace log_base_5_sqrt_inverse_l196_196815

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196815


namespace isos_trap_sum_sides_eq_sum_diags_l196_196208

noncomputable def is_isosceles_trapezoid {A B C D : Type} (AB CD AD BC AC BD : ℝ) (cos_ABC cos_DAB : ℝ) : Prop :=
  (AB = CD) ∧ (AD = BC) ∧ (cos_ABC = -cos_DAB) ∧
  (AC^2 + BD^2 = AB^2 + BC^2 + CD^2 + DA^2)

theorem isos_trap_sum_sides_eq_sum_diags {A B C D : Type} (AB CD AD BC AC BD : ℝ) (cos_ABC cos_DAB : ℝ)
  (h : is_isosceles_trapezoid AB CD AD BC AC BD cos_ABC cos_DAB) : 
  AC^2 + BD^2 = AB^2 + BC^2 + CD^2 + AD^2 := 
sorry

end isos_trap_sum_sides_eq_sum_diags_l196_196208


namespace negation_equivalence_l196_196643

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end negation_equivalence_l196_196643


namespace average_weight_l196_196265

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l196_196265


namespace count_ways_to_get_multiple_of_2_l196_196546

noncomputable def count_arrangements (digits : List ℕ) (n : ℕ) : ℕ :=
  let factorial (x : ℕ) : ℕ := if x = 0 then 1 else List.prod (List.range x).map(.succ)
  let permutations (l : List ℕ) : ℕ :=
    let counts := l.foldr (λ x acc, x :: acc) List.nil |>.insertionSortN |>.attach
    factorial l.length / List.prod (counts.map (λ p => factorial (p.1.getD 1)))
  permutations (digits.erase n)

theorem count_ways_to_get_multiple_of_2 : 
  let digits := [1, 1, 2, 5, 0]
  count_arrangements digits 2 + count_arrangements digits 0 = 24 := 
by
  let digits := [1, 1, 2, 5, 0]
  sorry

end count_ways_to_get_multiple_of_2_l196_196546


namespace example_problem_l196_196382

theorem example_problem : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end example_problem_l196_196382


namespace expected_variance_Y_l196_196076

variable {E : Type → ℝ} -- E represents expectation
variable {D : Type → ℝ} -- D represents variance

def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem expected_variance_Y {X Y : ℕ → ℝ} (h1 : ∀ t, X t + Y t = 8) 
  (h2 : ∀ t, X t ∼ B(10, 0.6)) : 
  E (λ t, Y t) = 2 ∧ D (λ t, Y t) = 2.4 := 
by
  sorry

end expected_variance_Y_l196_196076


namespace find_f_2_l196_196463

-- Condition: f(x + 1) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Statement to prove
theorem find_f_2 : f 2 = -1 := by
  sorry

end find_f_2_l196_196463


namespace no_positive_x_alpha_exists_l196_196430

open Set

theorem no_positive_x_alpha_exists :
  ¬ ∃ (x α : ℝ), (0 < x) ∧ (0 < α) ∧ ∀ (S : Finset ℕ), S.Nonempty → 
  ∃ max_S : ℕ, max_S = S.max' S.Nonempty ∧ 
  |x - ∑ s in S, (1 : ℝ) / s| > 1 / (max_S : ℝ) ^ α :=
begin
  -- To be proved
  sorry
end

end no_positive_x_alpha_exists_l196_196430


namespace range_arcsin_x_squared_minus_x_l196_196246

noncomputable def range_of_arcsin : Set ℝ :=
  {x | -Real.arcsin (1 / 4) ≤ x ∧ x ≤ Real.pi / 2}

theorem range_arcsin_x_squared_minus_x :
  ∀ x : ℝ, ∃ y ∈ range_of_arcsin, y = Real.arcsin (x^2 - x) :=
by
  sorry

end range_arcsin_x_squared_minus_x_l196_196246


namespace ways_to_divide_day_l196_196726

theorem ways_to_divide_day (n m : ℕ) (h : n * m = 86400) : 
  (∃ k : ℕ, k = 96) :=
  sorry

end ways_to_divide_day_l196_196726


namespace probability_defective_by_A_is_correct_l196_196658

noncomputable def defect_rate_A := 0.06
noncomputable def defect_rate_B := 0.05
noncomputable def market_share_A := 0.45
noncomputable def market_share_B := 0.55

noncomputable def total_defective_probability : ℚ :=
  market_share_A * defect_rate_A + market_share_B * defect_rate_B

noncomputable def bayes_theorem_probability : ℚ :=
  (market_share_A * defect_rate_A) / total_defective_probability

noncomputable def result : ℚ :=
  54 / 109

theorem probability_defective_by_A_is_correct :
  bayes_theorem_probability = result := by
  sorry

end probability_defective_by_A_is_correct_l196_196658


namespace parallel_lines_l196_196902

def line1 (a : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + 2 * a * p.2 - 1 = 0
def line2 (a : ℝ) : ℝ × ℝ → Prop := λ p, (2 * a - 1) * p.1 - a * p.2 - 1 = 0

theorem parallel_lines (a : ℝ) :
  (∀ p1 : ℝ × ℝ, line1 a p1) ∧ (∀ p2 : ℝ × ℝ, line2 a p2) →
  (a = 0 ∨ a = 1 / 4) :=
sorry

end parallel_lines_l196_196902


namespace find_c_l196_196462

noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 5
noncomputable def S : ℝ := 5 * Real.sqrt 3

theorem find_c (c : ℝ) (h : ∃ (C : ℝ), S = 1 / 2 * a * b * Real.sin C ∧ (C = π / 3 ∨ C = 2 * π / 3)) :
  c = Real.sqrt 21 ∨ c = Real.sqrt 61 := 
by 
   obtain ⟨C, hS, hC⟩ := h
   cases hC
   sorry

end find_c_l196_196462


namespace sum_of_factors_l196_196316

theorem sum_of_factors (n : ℕ) (h : n = 60) : 
  ∑ d in (finset.filter (λ x => x ∣ n) (finset.range (n+1))), d = 168 := 
by
  sorry

end sum_of_factors_l196_196316


namespace captain_age_problem_l196_196813

theorem captain_age_problem
    (sailor_age : ℕ)
    (bosun_sailor_age_diff : ℕ)
    (bosun_engineer_age_diff : ℕ)
    (helmsman_age_is_double_cabin_boy : ℕ → Prop) 
    (crew_members : ℕ)
    (average_age : ℕ)
    (age_sum : ℕ)
    (bosun_age : ℕ)
    (engineer_age : ℕ)
    (helmsman_age : ℕ)
    (cabin_boy_age : ℕ)
    (junior_sailor_age : ℕ)
    (captain_age : ℕ) : 
    sailor_age = 20 ∧
    bosun_sailor_age_diff = 4 ∧
    (bosun_is_4_years_older_than_sailor : bosun_age = sailor_age + bosun_sailor_age_diff) ∧
    bosun_engineer_age_diff = 6 ∧
    (helmsman_is_double_cabin_boy_and_6_years_older_than_engineer : ∀ cabin_boy_age, helmsman_age = 2 * cabin_boy_age ∧ helmsman_age = engineer_age + bosun_engineer_age_diff) ∧
    (bosun_age_eq : ∀ x, bosun_age = 24) ∧
    (bosun_is_older_than_cabin_boy_and_younger_than_engineer_by_same_amount : ∀ x,  bosun_age - cabin_boy_age = engineer_age - bosun_age) ∧
    crew_members = 6 ∧
    average_age = 28 →
    age_sum = (sailor_age + bosun_age + engineer_age + helmsman_age + cabin_boy_age + junior_sailor_age + captain_age) →
    captain_age = 40 :=
by
  intros h
  sorry

end captain_age_problem_l196_196813


namespace power_function_k_values_l196_196118

theorem power_function_k_values (k : ℝ) :
  (∃ (a : ℝ), (k^2 - k - 5) = a ∧ (∀ x : ℝ, (k^2 - k - 5) * x^3 = a * x^3)) →
  (k = 3 ∨ k = -2) :=
by
  intro h
  sorry

end power_function_k_values_l196_196118


namespace birth_date_of_older_friend_l196_196674

/-- Lean 4 statement for the proof problem --/
theorem birth_date_of_older_friend
  (d m y : ℕ)
  (h1 : y ≥ 1900 ∧ y < 2000)
  (h2 : d + 7 < 32) -- Assuming the month has at most 31 days
  (h3 : ((d+7) * 10^4 + m * 10^2 + y % 100) = 6 * (d * 10^4 + m * 10^2 + y % 100))
  (h4 : m > 0 ∧ m < 13)  -- Months are between 1 and 12
  (h5 : (d * 10^4 + m * 10^2 + y % 100) < (d+7) * 10^4 + m * 10^2 + y % 100) -- d < d+7 so older means smaller number
  : d = 1 ∧ m = 4 ∧ y = 1900 :=
by
  sorry -- Proof omitted

end birth_date_of_older_friend_l196_196674


namespace power_of_0_99_power_of_0_999_power_of_0_999999_l196_196143

theorem power_of_0_99 (n : ℕ) (h : n ≥ 1389) : (0.99 ^ n < 0.000001) :=
sorry

theorem power_of_0_999 (n : ℕ) (h : n ≥ 13825) : (0.999 ^ n < 0.000001) :=
sorry

theorem power_of_0_999999 (n : ℕ) (h : n ≥ 6000000) : (0.999999 ^ n < 0.000001) :=
sorry

end power_of_0_99_power_of_0_999_power_of_0_999999_l196_196143


namespace symmetric_points_subtraction_l196_196054

theorem symmetric_points_subtraction (a b : ℝ) (h1 : -2 = -a) (h2 : b = -3) : a - b = 5 :=
by {
  sorry
}

end symmetric_points_subtraction_l196_196054


namespace isosceles_triangle_perimeter_l196_196037

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l196_196037


namespace equation_equivalence_and_rst_l196_196231

theorem equation_equivalence_and_rst 
  (a x y c : ℝ) 
  (r s t : ℤ) 
  (h1 : r = 3) 
  (h2 : s = 1) 
  (h3 : t = 5)
  (h_eq1 : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^3) = a^5 * c^5 ∧ r * s * t = 15 :=
by
  sorry

end equation_equivalence_and_rst_l196_196231


namespace digit_in_ten_thousandths_place_of_fraction_l196_196280

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196280


namespace calculate_total_profit_l196_196325

def investment_and_profit (X Y : ℝ) (b_profit : ℝ) (total_profit : ℝ) :=
  let A_investment := 3 * X
  let A_period := 2 * Y
  let B_share := X * Y
  let A_share := A_investment * A_period
  (A_share / B_share = 6) ∧ (b_profit = 3000) ∧ (total_profit = 7 * b_profit)

theorem calculate_total_profit :
  Σ' (X Y : ℝ) (b_profit := 3000 : ℝ),
  ∃ total_profit : ℝ,
  investment_and_profit X Y b_profit total_profit →
  total_profit = 21000 :=
begin
  sorry
end

end calculate_total_profit_l196_196325


namespace find_radius_of_circumcircle_l196_196544

noncomputable def circumcircle_radius 
  (ABC : Type)
  [triangle ABC]
  (acute_angled : ABC → Prop)
  (height_AP : A → P → BC)
  (height_CQ : C → Q → AB)
  (area_ABC : ℕ := 18)
  (area_BPQ : ℕ := 2)
  (length_PQ : ℝ := 2 * sqrt 2) :
  Prop :=
  radius_circumcircle ABC = 9 / 2

theorem find_radius_of_circumcircle
    (ABC : Type)
    [triangle ABC]
    (acute_angled : ABC → Prop)
    (height_AP : A → P → BC)
    (height_CQ : C → Q → AB)
    (area_ABC : ℕ := 18)
    (area_BPQ : ℕ := 2)
    (length_PQ : ℝ := 2 * sqrt 2) :
    circumcircle_radius ABC acute_angled height_AP height_CQ area_ABC area_BPQ length_PQ :=
    by sorry

end find_radius_of_circumcircle_l196_196544


namespace problem_inequality_l196_196245

theorem problem_inequality (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 < a * b) (h_n : 2 ≤ n) :
  (a + b)^n > a^n + b^n + 2^n - 2 :=
sorry

end problem_inequality_l196_196245


namespace solution_f_neg_2_5_l196_196056

noncomputable def f : ℝ → ℝ := 
  λ x, if 0 < x ∧ x < 1 then 4 ^ x else sorry -- using "sorry" as we don't have full details for other ranges

theorem solution_f_neg_2_5 :
  (∀ x, f (x + 2) = f x) → -- periodicity with period 2
  (∀ x, f (-x) = -f x) →  -- odd function property
  (∀ x, (0 < x ∧ x < 1) → f x = 4 ^ x) → -- given condition on (0, 1)
  f (-2.5) = -2 := 
by
  intros periodic odd_func cond_0_1
  have h1: f (2.5) = f (0.5), from by
    rw [← sub_eq_zero, show 2.5 - 0.5 = 2, by norm_num] -- periodicity applies
    exact periodic 0.5,
  have h2: f (0.5) = 4 ^ 0.5, from cond_0_1 0.5 (by norm_num), -- using given condition
  rw [odd_func 2.5, h1, h2, real.sqrt_eq_rpow] -- apply odd function property and solve
  norm_num,
end

end solution_f_neg_2_5_l196_196056


namespace tan_theta_eq_neg_four_thirds_l196_196487

-- Conditions as definitions in Lean
def imaginary_unit : ℂ := complex.I
def z (θ : ℝ) : ℂ := (3 + 4 * imaginary_unit) * (complex.cos θ + imaginary_unit * complex.sin θ)
def real_part (θ : ℝ) : Prop := z θ = complex.ofReal (z θ).re
def not_multiple_of_pi (θ : ℝ) (k : ℤ) : Prop := θ ≠ k * real.pi + real.pi / 2

-- The Lean statement to be proved
theorem tan_theta_eq_neg_four_thirds (θ : ℝ) (k : ℤ) (hz : real_part θ) (hθ : not_multiple_of_pi θ k) : 
  real.tan θ = -4 / 3 := by
  sorry

end tan_theta_eq_neg_four_thirds_l196_196487


namespace red_candies_count_l196_196662

theorem red_candies_count : 
  let total_candies := 3409 in
  let blue_candies := 3264 in
  let red_candies := total_candies - blue_candies in
  red_candies = 145 :=
by
  sorry

end red_candies_count_l196_196662


namespace proof_problem_1_proof_problem_2_l196_196904

/-- Proof Problem 1 -/
theorem proof_problem_1 (α : ℝ) :
  let A := ⟨3, 0⟩
  let B := ⟨0, 3⟩
  let C := ⟨Real.cos α, Real.sin α⟩
  (C.1 - A.1) * C.1 + C.2 * (C.2 - A.2) 
  = -1 → 
  Real.sin (α + π / 4) = (sqrt 2) / 3 :=
by
  sorry

/-- Proof Problem 2 -/
theorem proof_problem_2 (α : ℝ) :
  let A := ⟨3, 0⟩
  let B := ⟨0, 3⟩
  let C := ⟨Real.cos α, Real.sin α⟩
  (|⟨A.1 - C.1, A.2 - C.2⟩| = sqrt 13) 
  ∧ (0 < α) 
  ∧ (α < π) → 
  (Real.acos ((C.1 ⬝ B.1 + C.2 ⬝ B.2) 
   / ((sqrt (C.1 ^ 2 + C.2 ^ 2)) * (sqrt (B.1 ^ 2 + B.2 ^ 2)))) 
  = π / 6) :=
by
  sorry

end proof_problem_1_proof_problem_2_l196_196904


namespace pink_cookies_l196_196193

theorem pink_cookies (total_cookies : ℕ) (percentage_pink : ℝ) (pink_cookies : ℕ) 
  (h1 : total_cookies = 150) 
  (h2 : percentage_pink = 30 / 100) :
  pink_cookies = (percentage_pink * total_cookies).to_nat :=
by
  sorry

end pink_cookies_l196_196193


namespace width_of_hall_l196_196539

variable (L W H : ℕ) -- Length, Width, Height of the hall
variable (expenditure cost : ℕ) -- Expenditure and cost per square meter

-- Given conditions
def hall_length : L = 20 := by sorry
def hall_height : H = 5 := by sorry
def total_expenditure : expenditure = 28500 := by sorry
def cost_per_sq_meter : cost = 30 := by sorry

-- Derived value
def total_area_to_cover (W : ℕ) : ℕ :=
  (2 * (L * W) + 2 * (L * H) + 2 * (W * H))

theorem width_of_hall (W : ℕ) (h: total_area_to_cover L W H * cost = expenditure) : W = 15 := by
  sorry

end width_of_hall_l196_196539


namespace N_is_composite_l196_196811

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  sorry

end N_is_composite_l196_196811


namespace fish_in_lake_l196_196875

theorem fish_in_lake (white_ducks black_ducks multicolor_ducks : ℕ) 
                     (fish_per_white fish_per_black fish_per_multicolor : ℕ)
                     (h1 : fish_per_white = 5)
                     (h2 : fish_per_black = 10)
                     (h3 : fish_per_multicolor = 12)
                     (h4 : white_ducks = 3)
                     (h5 : black_ducks = 7)
                     (h6 : multicolor_ducks = 6) :
                     (white_ducks * fish_per_white + black_ducks * fish_per_black + multicolor_ducks * fish_per_multicolor) = 157 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end fish_in_lake_l196_196875


namespace probability_roll_6_final_l196_196732

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l196_196732


namespace black_number_as_sum_of_white_numbers_l196_196183

def is_white_number (x : Real) : Prop :=
  ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ x = Real.sqrt (a + b * Real.sqrt 2)

def is_black_number (x : Real) : Prop :=
  ∃ (c d : ℤ), c ≠ 0 ∧ d ≠ 0 ∧ x = Real.sqrt (c + d * Real.sqrt 7)

theorem black_number_as_sum_of_white_numbers :
  ∃ (c d : ℤ), c ≠ 0 ∧ d ≠ 0 ∧ ∃ (k : ℕ) (white_numbers : Fin k → Real), 
    (∀ i, is_white_number (white_numbers i)) ∧ 
    is_black_number (Finset.univ.fold (+) 0 white_numbers) :=
sorry

end black_number_as_sum_of_white_numbers_l196_196183


namespace Martha_should_buy_84oz_of_apples_l196_196187

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end Martha_should_buy_84oz_of_apples_l196_196187


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196292

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196292


namespace exponentiation_equality_l196_196791

theorem exponentiation_equality :
  3^12 * 8^12 * 3^3 * 8^8 = 24 ^ 15 * 32768 := by
  sorry

end exponentiation_equality_l196_196791


namespace max_matching_pairs_l196_196591

theorem max_matching_pairs 
  (total_pairs : ℕ := 23) 
  (total_colors : ℕ := 6) 
  (total_sizes : ℕ := 3) 
  (lost_shoes : ℕ := 9)
  (shoes_per_pair : ℕ := 2) 
  (total_shoes := total_pairs * shoes_per_pair) 
  (remaining_shoes := total_shoes - lost_shoes) :
  ∃ max_pairs : ℕ, max_pairs = total_pairs - lost_shoes / shoes_per_pair :=
sorry

end max_matching_pairs_l196_196591


namespace table_columns_sum_non_decreasing_l196_196979

theorem table_columns_sum_non_decreasing {m n : ℕ} (x : ℕ → ℕ → ℕ)
  (rearranged_x : ℕ → ℕ → ℕ)
  (h_rearranged : ∀ i, rearrange_row (x i) = rearranged_x i) :
  ∑ j in finset.range n, (∏ i in finset.range m, rearranged_x i j) ≥
  ∑ j in finset.range n, (∏ i in finset.range m, x i j) :=
sorry

end table_columns_sum_non_decreasing_l196_196979


namespace find_a_l196_196116

theorem find_a (x y a : ℤ) (h1 : a * x + y = 40) (h2 : 2 * x - y = 20) (h3 : 3 * y^2 = 48) : a = 3 :=
sorry

end find_a_l196_196116


namespace circumcenter_BCD_l196_196465

variables {A B C D E F K R : Type}

-- Given conditions
variables [Incenter E A B C] [Incenter F A C D]
variables (h1 : isAngleBisector AC (Angle BAD))
variables (h2 : AC^2 = AB * AD)
variables (h3 : IntersectionPoint AD Circumcircle(C D F) K)
variables (h4 : IntersectionPoint FC Circumcircle(B C E) R)
variables (h5 : Parallel RK EF)

-- Desired conclusion
theorem circumcenter_BCD (h1 : isAngleBisector AC (Angle BAD)) 
                         (h2 : AC^2 = AB * AD) 
                         (h3 : IntersectionPoint AD Circumcircle(C D F) K)
                         (h4 : IntersectionPoint FC Circumcircle(B C E) R)
                         (h5 : Parallel RK EF) : IsCircumcenter A (Triangle B C D) :=
begin
  sorry -- proof goes here
end

end circumcenter_BCD_l196_196465


namespace find_n_for_circles_tangent_l196_196672

theorem find_n_for_circles_tangent
  {C1 C2 : Circle}
  (intersect_pt : Point)
  (radii_product : ℝ)
  (tangent_line_slope : ℝ)
  (h_intersect : intersect_pt = (6, 8))
  (h_radii_product : ∀ r₁ r₂ : ℝ, r₁ * r₂ = 45)
  (h_tangent_x_axis : ∀ c : Circle, tangent c x_axis)
  (h_tangent_line : ∀ c : Circle, tangent c (line_through_origin tangent_line_slope))
  (h_positive_slope : tangent_line_slope > 0) :
  tangent_line_slope = 3 / 7 := 
sorry

end find_n_for_circles_tangent_l196_196672


namespace zeros_of_g_l196_196497

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else |Real.log x / Real.log 2|

def g (x : ℝ) : ℝ := f x - 1 / 2

theorem zeros_of_g : {x : ℝ | g x = 0} = {-1, Real.sqrt 2 / 2, Real.sqrt 2} :=
by
  sorry

end zeros_of_g_l196_196497


namespace max_value_of_y_no_min_value_l196_196636

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem max_value_of_y_no_min_value :
  (∃ x, -2 < x ∧ x < 2 ∧ function_y x = 5) ∧
  (∀ y, ∃ x, -2 < x ∧ x < 2 ∧ function_y x >= y) :=
by
  sorry

end max_value_of_y_no_min_value_l196_196636


namespace solve_inequality_l196_196653

theorem solve_inequality (x : ℝ) : (2 * x - 3) / (x + 2) ≤ 1 ↔ (-2 < x ∧ x ≤ 5) :=
  sorry

end solve_inequality_l196_196653


namespace odd_function_h_l196_196518

noncomputable def f (x h k : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

theorem odd_function_h (k : ℝ) (h : ℝ) (H : ∀ x : ℝ, x ≠ -1 → f x h k = -f (-x) h k) :
  h = Real.log 2 :=
sorry

end odd_function_h_l196_196518


namespace log_base5_of_inverse_sqrt5_l196_196829

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196829


namespace digit_in_ten_thousandths_place_of_fraction_l196_196279

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196279


namespace log_base_5_sqrt_inverse_l196_196819

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196819


namespace area_of_triangle_l196_196277

theorem area_of_triangle : 
  let line1 := λ x : ℝ, 3 * x - 6
  let line2 := λ x : ℝ, -2 * x + 14
  let y_axis_intercept1 := (0, -6)
  let y_axis_intercept2 := (0, 14)
  let intersection := (4, 6)
  let base := y_axis_intercept2.snd - y_axis_intercept1.snd
  let height := intersection.fst
  let area := 1 / 2 * base * height
  area = 40 := 
by
  sorry

end area_of_triangle_l196_196277


namespace total_class_arrangements_l196_196355

-- Define the constraints as provided in the problem description.
def chinese : Type := unit
def mathematics : Type := unit
def english : Type := unit
def physics : Type := unit
def chemistry : Type := unit
def elective : Type := unit
def self_study : Type := unit

-- First period must be one of Chinese, Mathematics, or English.
def first_period_options : set (unit) := set.insert () (set.insert () (set.insert () set.empty))

-- Eighth period can be either an elective or a self-study period.
def eighth_period_options : set (unit) := set.insert () (set.insert () set.empty)

-- Define the constraints for non-adjacency conditions.
def non_adjacent (a b : unit) : Prop := (a = () ∧ b = ()) ∨ (a = () ∧ b = ())

-- Define the total number of different arrangements in Lean.
theorem total_class_arrangements : (nat :=
  -- Add the correct number of different ways to handle each constraint.
sorry

end total_class_arrangements_l196_196355


namespace volume_of_circumscribed_sphere_l196_196348

noncomputable def hex_prism_sphere_volume 
  (height : ℝ) 
  (perimeter : ℝ) 
  (base_side_length : ℝ) 
  (diagonal_length : ℝ) 
  (sphere_radius : ℝ) 
  (volume : ℝ) : Prop :=
  height = sqrt 3 ∧ 
  perimeter = 3 ∧ 
  base_side_length = 1 / 2 ∧ 
  diagonal_length = 2 ∧  -- due to sqrt(3 + 1)
  sphere_radius = 1 ∧ 
  volume = (4 * Real.pi) / 3

theorem volume_of_circumscribed_sphere :
  ∃ V, hex_prism_sphere_volume (sqrt 3) 3 (1 / 2) 2 1 V := 
begin
  use (4 * Real.pi) / 3,
  repeat { split },
  { refl },
  { refl },
  { norm_num },
  { norm_num },
  { norm_num },
end

end volume_of_circumscribed_sphere_l196_196348


namespace probability_six_on_final_roll_l196_196740

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l196_196740


namespace health_risk_factor_prob_l196_196812

noncomputable def find_p_q_sum (p q: ℕ) : ℕ :=
if h1 : p.gcd q = 1 then
  31
else 
  sorry

theorem health_risk_factor_prob (p q : ℕ) (h1 : p.gcd q = 1) 
                                (h2 : (p : ℚ) / q = 5 / 26) :
  find_p_q_sum p q = 31 :=
sorry

end health_risk_factor_prob_l196_196812


namespace find_line_l_equation_l196_196481

theorem find_line_l_equation 
  (A B: (ℝ × ℝ)) (l₁ l₂: (ℝ × ℝ → Prop)) (l: ℝ × ℝ → Prop)
  (H1 : A = (3, 3))
  (H2 : B = (5, 2))
  (H3 : l₁ = λ p, 3 * p.1 - p.2 - 1 = 0)
  (H4 : l₂ = λ p, p.1 + p.2 - 3 = 0)
  (H5 : ∃ p : ℝ × ℝ, l₁ p ∧ l₂ p ∧ l p)
  (H6 : ∃ k, ∀ p, k = dist p A ↔ k = dist p B):
  (l = λ p, p.1 + 2 * p.2 - 5 = 0) ∨ (l = λ p, p.1 - 6 * p.2 + 11 = 0) :=
sorry

end find_line_l_equation_l196_196481


namespace part1_part2_part3_l196_196585

def A : Set ℝ := {x | x^2 - x - 2 > 0}
def B : Set ℝ := {x | 3 - |x| ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - (m - 1)) * (x - (2*m + 1)) < 0}

theorem part1 : A ∩ B = (Iio (-1) ∪ Ioi 2) ∩ Icc (-3) 3 := by
  sorry

theorem part2 : (Set.Univ \ A) ∪ B = Icc (-3) 3 := by
  sorry

theorem part3 : ∀ m : ℝ, C m ⊆ B → m ∈ Icc (-2) 1 := by
  sorry

end part1_part2_part3_l196_196585


namespace average_price_per_book_l196_196204

theorem average_price_per_book
  (science_books_first: ℕ := 25) (math_books_first: ℕ := 20) (lit_books_first: ℕ := 20)
  (cost_science_first: ℕ := 1500) (cost_math_first: ℕ := 2500) (cost_lit_first: ℕ := 2500)
  (first_shop_discount: ℕ := 500)
  (history_books_second: ℕ := 15) (geo_books_second: ℕ := 10) (lang_books_second: ℕ := 10)
  (cost_hist_second: ℕ := 1000) (cost_geo_second: ℕ := 500) (cost_lang_second: ℕ := 750)
  (geo_discount: ℕ := 3-2) (second_shop_discount: ℕ := 250) :
  (let total_books := science_books_first + math_books_first + lit_books_first +
                       history_books_second + (geo_books_second * geo_discount) + lang_books_second in
  let total_cost := (cost_science_first + cost_math_first + cost_lit_first - first_shop_discount) +
                    (cost_hist_second + cost_geo_second + cost_lang_second - second_shop_discount) in
  (total_cost / total_books : ℝ)) = 76.19 := 
by 
  sorry

end average_price_per_book_l196_196204


namespace range_of_a_l196_196924

open Real

noncomputable def f (x : ℝ) : ℝ := ln (x + 1) + x^2
noncomputable def g (x : ℝ) : ℝ := sqrt 2 / 2 * sin x - x

noncomputable def f_deriv (x : ℝ) : ℝ := 1 / (x + 1) + 2 * x
noncomputable def g_deriv (x : ℝ) : ℝ := sqrt 2 / 2 * cos x - 1

theorem range_of_a : (∀ (x₁ : ℝ), ∃ (x₂ : ℝ), ( f_deriv x₁ ) * ( g_deriv x₂ ) = -1) → 
  ∀ (a : ℝ), |a| ≥ sqrt 2 :=
sorry

end range_of_a_l196_196924


namespace no_such_k_l196_196168

theorem no_such_k (u : ℕ → ℝ) (v : ℕ → ℝ)
  (h1 : u 0 = 6) (h2 : v 0 = 4)
  (h3 : ∀ n, u (n + 1) = (3 / 5) * u n - (4 / 5) * v n)
  (h4 : ∀ n, v (n + 1) = (4 / 5) * u n + (3 / 5) * v n) :
  ¬ ∃ k, u k = 7 ∧ v k = 2 :=
by
  sorry

end no_such_k_l196_196168


namespace range_of_m_l196_196017

noncomputable def f (m x : ℝ) : ℝ := m * (x - 2 * m) * (x + m + 3)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ∧ (∃ x : ℝ, x < -4 ∧ f m x * g x < 0) → (-4 < m ∧ m < -2) :=
by
  sorry

end range_of_m_l196_196017


namespace maximum_product_of_two_digit_numbers_l196_196676

theorem maximum_product_of_two_digit_numbers : 
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ {a, b, c, d} ⊆ {3, 4, 7, 8}) ∧ 
  ((10 * a + b) * (10 * c + d) = 6142) := 
sorry

end maximum_product_of_two_digit_numbers_l196_196676


namespace problem_l196_196695

theorem problem (k : ℕ) (h1 : 30^k ∣ 929260) : 3^k - k^3 = 2 :=
sorry

end problem_l196_196695


namespace polynomial_is_first_degree_l196_196472

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ i, i ≥ 1 → a (i - 1) + a (i + 1) = 2 * a i

def polynomial_P (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (nat.choose n k) * (x^k) * ((1 - x) ^ (n - k))

theorem polynomial_is_first_degree (a : ℕ → ℝ) (n : ℕ) (h : sequence_condition a) :
  ∃ A B : ℝ, ∀ x : ℝ, polynomial_P a n x = A + B * x :=
sorry

end polynomial_is_first_degree_l196_196472


namespace second_derivative_at_x₀_l196_196180

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₀ a b : ℝ)

-- Condition: f(x₀ + Δx) - f(x₀) = a * Δx + b * (Δx)^2
axiom condition : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * (Δx)^2

theorem second_derivative_at_x₀ : deriv (deriv f) x₀ = 2 * b :=
sorry

end second_derivative_at_x₀_l196_196180


namespace probability_roll_6_final_l196_196731

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l196_196731


namespace contradiction_prop_l196_196482

theorem contradiction_prop (p : Prop) : 
  (∃ x : ℝ, x < -1 ∧ x^2 - x + 1 < 0) → (∀ x : ℝ, x < -1 → x^2 - x + 1 ≥ 0) :=
sorry

end contradiction_prop_l196_196482


namespace find_a_plus_b_plus_c_l196_196416

-- Define the polynomial and its roots α, β, γ
variables {α β γ : ℂ}
variables {a b c : ℂ}

-- Define the polynomial having α, β, γ as roots
def polynomial_has_roots := (∀ x : ℂ, x^3 - 7 * x^2 + 12 * x - 18 = 0 
  ↔ x = α ∨ x = β ∨ x = γ)

-- Define the s_k series and its initial given values
def s_0 : ℂ := 3
def s_1 : ℂ := 7
def s_2 : ℂ := 13

-- Define the recursive relationship for s_k
def recursive_relation (s_k s_k1 s_k2 : ℂ) :=
  s_k1 = α^k + β^k + γ^k ∧ s_k2 = α^(k-1) + β^(k-1) + γ^(k-1)

-- Statement that we need to prove
theorem find_a_plus_b_plus_c (h_poly : polynomial_has_roots)
  (h_s0 : s_0 = 3)
  (h_s1 : s_1 = 7)
  (h_s2 : s_2 = 13)
  (h_rec : ∀ k : ℕ, k ≥ 2 → recursive_relation s_k s_(k-1) s_(k-2)) : 
  a + b + c = -3 := sorry

end find_a_plus_b_plus_c_l196_196416


namespace weight_of_second_piece_l196_196771

-- Define the uniform density of the metal.
def density : ℝ := 0.5  -- ounces per square inch

-- Define the side lengths of the two pieces of metal.
def side_length1 : ℝ := 4  -- inches
def side_length2 : ℝ := 7  -- inches

-- Define the weights of the first piece of metal.
def weight1 : ℝ := 8  -- ounces

-- Define the areas of the pieces of metal.
def area1 : ℝ := side_length1^2  -- square inches
def area2 : ℝ := side_length2^2  -- square inches

-- The theorem to prove: the weight of the second piece of metal.
theorem weight_of_second_piece : (area2 * density) = 24.5 :=
by
  sorry

end weight_of_second_piece_l196_196771


namespace log_n_ge_k_log_2_l196_196199

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem log_n_ge_k_log_2 (n : ℕ) (k : ℕ) (h : n > 1) (hk : ∃ (p : ℕ → ℕ), (∀ i, p i ∈ Prime ∧ n = ∏ i in (Finset.range k), (p i) ^ (mult p i)) ∧ (mult : ℕ → ℕ) ∧ (∀ i, (p i) > 1)) : log n ≥ k * log 2 :=
by
  sorry

end log_n_ge_k_log_2_l196_196199


namespace arithmetic_geometric_sequence_l196_196899

theorem arithmetic_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) 
(h1 : S 3 = 2) 
(h2 : S 6 = 18) 
(h3 : ∀ n, S n = a 1 * (1 - q^n) / (1 - q)) 
: S 10 / S 5 = 33 := 
sorry

end arithmetic_geometric_sequence_l196_196899


namespace simplify_expression_l196_196211

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : (x^2)⁻¹ - 2 = (1 - 2 * x^2) / (x^2) :=
by
  -- proof here
  sorry

end simplify_expression_l196_196211


namespace angle_C_measure_l196_196896

theorem angle_C_measure 
  (a b c : ℝ) -- side lengths of the triangle
  (h : a^2 + b^2 - c^2 = a * b) -- given condition
  : real.arccos (1 / 2) = real.pi / 3 :=
by
  sorry

end angle_C_measure_l196_196896


namespace company_blocks_l196_196123

noncomputable def number_of_blocks (workers_per_block total_budget gift_cost : ℕ) : ℕ :=
  (total_budget / gift_cost) / workers_per_block

theorem company_blocks :
  number_of_blocks 200 6000 2 = 15 :=
by
  sorry

end company_blocks_l196_196123


namespace num_zeros_in_fraction_decimal_l196_196101

theorem num_zeros_in_fraction_decimal :
  (let x := (1 : ℚ) / (2^3 * 5^6) in
   ∃ k : ℕ, x = 8 / 10^6 ∧ k = 5) :=
by
  -- The proof would involve showing x can be written as '0.000008',
  -- which has exactly 5 zeros between the decimal point and the first non-zero digit.
  sorry

end num_zeros_in_fraction_decimal_l196_196101


namespace sufficient_but_not_necessary_condition_l196_196432

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → log (1/2) (x + 2) < 0) ∧ (¬ (x > 1) → log (1/2) (x + 2) < 0 → x > -1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l196_196432


namespace max_n_divisor_l196_196059

theorem max_n_divisor (k n : ℕ) (h1 : 81849 % n = k) (h2 : 106392 % n = k) (h3 : 124374 % n = k) : n = 243 := by
  sorry

end max_n_divisor_l196_196059


namespace hansel_album_duration_l196_196086

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l196_196086


namespace divisor_and_remainder_l196_196645

theorem divisor_and_remainder
  (a : ℕ) (q : ℕ) (b r : ℕ) (h : a = b * q + r) (h_r : 0 ≤ r ∧ r < b) :
  a = 1270 ∧ q = 74 → b = 17 ∧ r = 12 :=
by
  intro h₁ h₂
  sorry

end divisor_and_remainder_l196_196645


namespace find_tan_z_l196_196561

theorem find_tan_z (X Y Z : ℝ) (h1 : Real.cot X * Real.cot Z = 1 / 3) (h2 : Real.cot Y * Real.cot Z = 1 / 8) : 
  Real.tan Z = 12 + Real.sqrt 136 := 
sorry

end find_tan_z_l196_196561


namespace digit_in_ten_thousandths_place_l196_196310

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196310


namespace decreasing_interval_of_f_l196_196642

def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x - 8)

theorem decreasing_interval_of_f :
  ∀ x, x < -2 → monotone_decreasing_on (λ x, Real.log (x^2 - 2*x - 8)) (Iio x) :=
begin
  sorry
end

end decreasing_interval_of_f_l196_196642


namespace decreasing_interval_and_extrema_cos_2x0_of_f_x0_l196_196066

def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

-- Part (1)
theorem decreasing_interval_and_extrema :
  (∀ x : ℝ, x ∈ Icc (0:ℝ) (Real.pi / 2) → f(x) = 2 * Real.sin (2*x + Real.pi / 6)) ∧
  (∀ x : ℝ, x ∈ Icc (Real.pi / 6) (2 * Real.pi / 3) → f(x)) ∧
  (f (0:ℝ) = -1) ∧ (f (Real.pi / 2) = 2) :=
sorry

-- Part (2)
theorem cos_2x0_of_f_x0 :
  (∃ x0 : ℝ, f(x0) = 6 / 5 ∧ x0 ∈ Icc (Real.pi / 4) (Real.pi / 2)) →
  ∃ x0 : ℝ, cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

end decreasing_interval_and_extrema_cos_2x0_of_f_x0_l196_196066


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196293

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196293


namespace can_be_profitable_lowest_average_cost_l196_196669

-- Condition definitions
def processing_cost (x : ℝ) (y: ℝ) : Prop :=
  if 120 ≤ x ∧ x < 144 then
    y = (1/3) * x^3 - 80 * x^2 + 5040 * x
  else if 144 ≤ x ∧ x < 500 then
    y = (1/2) * x^2 - 200 * x + 80000
  else
    false

def value_per_ton : ℝ := 200

-- Problem statement
theorem can_be_profitable (x y: ℝ) : 200 ≤ x ∧ x ≤ 300 → processing_cost x y →
  let S := value_per_ton * x - y in S < 0 ∧ max_profit := -5000 ∧ minimum_subsidy := 5000  :=
sorry

theorem lowest_average_cost (x y : ℝ) : processing_cost x y →
  x = 400 :=
sorry

end can_be_profitable_lowest_average_cost_l196_196669


namespace probability_sum_8_9_10_l196_196234

/-- The faces of the first die -/
def first_die := [2, 2, 3, 3, 5, 5]

/-- The faces of the second die -/
def second_die := [1, 3, 4, 5, 6, 7]

/-- Predicate that checks if the sum of two numbers is either 8, 9, or 10 -/
def valid_sum (a b : ℕ) : Prop := a + b = 8 ∨ a + b = 9 ∨ a + b = 10

/-- Calculate the probability of a sum being 8, 9, or 10 according to the given dice setup -/
def calc_probability : ℚ := 
  let valid_pairs := [(2, 6), (2, 7), (3, 5), (3, 6), (3, 7), (5, 3), (5, 4), (5, 5)] 
  (valid_pairs.length : ℚ) / (first_die.length * second_die.length : ℚ)

theorem probability_sum_8_9_10 : calc_probability = 4 / 9 :=
by
  sorry

end probability_sum_8_9_10_l196_196234


namespace max_prime_product_l196_196621

theorem max_prime_product : 
  ∃ (x y z : ℕ), 
    Prime x ∧ Prime y ∧ Prime z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x + y + z = 49 ∧ 
    x * y * z = 4199 := 
by
  sorry

end max_prime_product_l196_196621


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196300

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196300


namespace zeros_before_first_nonzero_digit_l196_196093

theorem zeros_before_first_nonzero_digit 
  (h : ∀ n : ℕ, n = 2^3 * 5^6) : 
  (zeros_before_first_nonzero_digit_decimal \(\frac{1}{n}) = 6) :=
sorry

end zeros_before_first_nonzero_digit_l196_196093


namespace log_base_5_sqrt_inverse_l196_196816

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196816


namespace value_of_x_squared_minus_y_squared_l196_196523

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l196_196523


namespace int_solutions_fraction_l196_196442

theorem int_solutions_fraction :
  ∀ n : ℤ, (∃ k : ℤ, (n - 2) / (n + 1) = k) ↔ n = 0 ∨ n = -2 ∨ n = 2 ∨ n = -4 :=
by
  intro n
  sorry

end int_solutions_fraction_l196_196442


namespace significant_digits_of_square_side_length_l196_196243

noncomputable def side_length (A : ℝ) : ℝ := Real.sqrt A

def significant_digits (x : ℝ) : ℕ :=
  if x = 0 then 0
  else
    let s := x.abs.toString(false) in
    s.toList.filter (fun c => c.isDigit ∨ c ≠ '0').length

theorem significant_digits_of_square_side_length (A : ℝ) (hA : A = 3.0625) :
  significant_digits (side_length A) = 3 :=
by
  sorry

end significant_digits_of_square_side_length_l196_196243


namespace Harkamal_purchase_grapes_l196_196938

theorem Harkamal_purchase_grapes
  (G : ℕ) -- The number of kilograms of grapes
  (cost_grapes_per_kg : ℕ := 70)
  (kg_mangoes : ℕ := 9)
  (cost_mangoes_per_kg : ℕ := 55)
  (total_paid : ℕ := 1195) :
  70 * G + 55 * 9 = 1195 → G = 10 := 
by
  sorry

end Harkamal_purchase_grapes_l196_196938


namespace cos_x_values_eq_045_count_l196_196515

theorem cos_x_values_eq_045_count (x : ℝ) (h1 : -360 ≤ x) (h2 : x < 360) (h3 : Real.cos (Real.pi * x / 180) = 0.45) : 
  4 := sorry

end cos_x_values_eq_045_count_l196_196515


namespace rowing_speed_downstream_l196_196755

theorem rowing_speed_downstream (V_m V_u V_d : ℕ) (hVm : V_m = 45) (hVu : V_u = 25) :
  V_d = 65 :=
by
  have Vs : ℕ := V_m - V_u
  have hVs : Vs = 20 := by rw [hVm, hVu]; norm_num
  have hVd : V_d = V_m + Vs := by rw [hVm, hVs]; norm_num
  sorry

end rowing_speed_downstream_l196_196755


namespace multiplication_result_l196_196946

theorem multiplication_result
  (h : 16 * 21.3 = 340.8) :
  213 * 16 = 3408 :=
sorry

end multiplication_result_l196_196946


namespace work_completion_days_l196_196774

theorem work_completion_days :
  (∀ B_time C_time : ℕ, B_time > 0 ∧ C_time > 0 ∧ B_time = 18 ∧ C_time = 12 → 
  let W_B := 1 / (B_time : ℝ) in
  let W_A := 2 * W_B in
  let W_C := 1 / (C_time : ℝ) in
  let W_ABC := W_A + W_B + W_C in
  (1 / W_ABC) = 4 ) :=
by
  intros B_time C_time h
  let W_B := 1 / (B_time : ℝ)
  let W_A := 2 * W_B
  let W_C := 1 / (C_time : ℝ)
  let W_ABC := W_A + W_B + W_C
  have : (1 / W_ABC = 4) := sorry
  exact this

end work_completion_days_l196_196774


namespace determine_base_solution_l196_196801

theorem determine_base_solution :
  ∃ (h : ℕ), 
  h > 8 ∧ 
  (8 * h^3 + 6 * h^2 + 7 * h + 4) + (4 * h^3 + 3 * h^2 + 2 * h + 9) = 1 * h^4 + 3 * h^3 + 0 * h^2 + 0 * h + 3 ∧
  (9 + 4) = 13 ∧
  1 * h + 3 = 13 ∧
  (7 + 2 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (6 + 3 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (8 + 4 + 1) = 13 ∧
  1 * h + 3 = 13 ∧
  h = 10 :=
by
  sorry

end determine_base_solution_l196_196801


namespace total_minutes_to_finish_album_l196_196088

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l196_196088


namespace intersect_complement_l196_196963

open Finset

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def A := {1, 3, 6} : Finset ℕ
def B := {2, 3, 4} : Finset ℕ

-- Define the complement of B in U
def complement_U_B := U \ B

-- The statement to prove
theorem intersect_complement : A ∩ complement_U_B = {1, 6} :=
by sorry

end intersect_complement_l196_196963


namespace find_fraction_l196_196677

noncomputable def a := 5100
noncomputable def b := (2 : ℝ) / 5
noncomputable def c := (1 : ℝ) / 2
noncomputable def d := 765.0000000000001

theorem find_fraction (x : ℝ) : x * (c * b * a) = d -> x = 0.75 :=
by
  intro h
  sorry

end find_fraction_l196_196677


namespace log_five_fraction_l196_196834

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196834


namespace max_sin_tan_arctan_diff_l196_196053

theorem max_sin_tan_arctan_diff (x : ℝ) (hx : 0 < x) : 
  ∃ a b : ℝ, tan(a)= x / 9 ∧ tan(b)= x / 16 ∧ sin (a - b) = 7 / 25 := sorry

end max_sin_tan_arctan_diff_l196_196053


namespace range_of_m_l196_196960

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l196_196960


namespace valid_arrangements_modulo_1000_is_596_l196_196269

variable (flagpoles : Type) [DecidableEq flagpoles] (flags : Type) [DecidableEq flags]

/-- There are two distinguishable flagpoles. -/
def flagpoles_count : ℕ := 2

/-- There are 12 identical blue flags and 9 identical green flags (21 flags in total). -/
def blue_flags_count : ℕ := 12
def green_flags_count : ℕ := 9

/-- Valid arrangements must have each flagpole with at least three flags,
and no two green flags on either pole can be adjacent. -/
def valid_arrangements : ℕ := sorry

/-- Compute the number of valid arrangements modulo 1000. -/
def arrangements_modulo_1000 : ℕ :=
  valid_arrangements % 1000

/-- Given the conditions, the number of valid arrangements modulo 1000 is 596.—/
theorem valid_arrangements_modulo_1000_is_596 :
  arrangements_modulo_1000 = 596 := sorry

end valid_arrangements_modulo_1000_is_596_l196_196269


namespace value_of_x_squared_minus_y_squared_l196_196528

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l196_196528


namespace inequalities_neither_necessary_nor_sufficient_l196_196253

theorem inequalities_neither_necessary_nor_sufficient 
  (x a b y: ℝ) :
  (x > a ∧ y > b) ↔ (x + y > a + b ∧ xy > ab) = false := 
sorry

end inequalities_neither_necessary_nor_sufficient_l196_196253


namespace leftover_money_l196_196324

def cost_of_bread := 2.25
def cost_of_peanut_butter := 2.0
def number_of_loaves := 3
def total_money := 14.0

theorem leftover_money : total_money - (number_of_loaves * cost_of_bread + cost_of_peanut_butter) = 5.25 :=
by
  sorry

end leftover_money_l196_196324


namespace marj_money_left_l196_196593

theorem marj_money_left (twenty_bills : ℕ) (five_bills : ℕ) (loose_coins : ℝ) (cake_cost : ℝ) :
  twenty_bills = 2 → five_bills = 3 → loose_coins = 4.5 → cake_cost = 17.5 →
  (20 * twenty_bills + 5 * five_bills + loose_coins - cake_cost = 42) :=
by {
  intros h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  norm_num,
  sorry
}

end marj_money_left_l196_196593


namespace area_equalities_l196_196196

noncomputable def area_of_triangle (A B C : ℝ) := (1/2) * ((B - A).cross (C - A)).z

theorem area_equalities
  (A B C P D E F : ℝ)
  (hP_outside : ¬ ((∃ t : ℝ, P = A + t*(B - A)) ∨ (∃ t : ℝ, P = B + t*(C - B)) ∨ (∃ t : ℝ, P = C + t*(A - C))))
  (hD_perpendicular : (P - D).dot (B - C) = 0)
  (hE_perpendicular : (P - E).dot (C - A) = 0)
  (hF_perpendicular : (P - F).dot (A - B) = 0)
  (h_area_equal_1 : area_of_triangle P A F = area_of_triangle P B D)
  (h_area_equal_2 : area_of_triangle P B D = area_of_triangle P C E) :
  area_of_triangle A B C = area_of_triangle P A F := 
sorry

end area_equalities_l196_196196


namespace range_of_a_l196_196955

def f (a x : ℝ) : ℝ := 2 * x ^ 3 - 3 * a * x ^ 2 + a

theorem range_of_a (a : ℝ) (h : ∃ x₁ x₂ x₃ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) : a > 1 ∨ a < -1 :=
begin
  sorry
end

end range_of_a_l196_196955


namespace sum_d_sq_eq_550_l196_196798

noncomputable def d (k : ℕ) : ℝ := k + 1 / (3 * k + d k)

theorem sum_d_sq_eq_550 :
  ∑ k in Finset.range 10, (k + 1)^2 + 3 * (k + 1) = 550 :=
by
  sorry

end sum_d_sq_eq_550_l196_196798


namespace unique_a_l196_196132

noncomputable theory

open Real

def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

def is_intersection (l : ℝ → ℝ) (x y : ℝ) : Prop := l y = x ∧ hyperbola x y

def is_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

-- Given the hyperbola x^2 - y^2 = 1
-- Find the real number a > 1 that satisfies the condition

theorem unique_a (a : ℝ) (h : a > 1) :
  (∀ (l1 l2 : ℝ → ℝ),
    -- l1 and l2 are lines through (a,0) and are perpendicular
    (l1 0 = a ∧ l2 0 = a ∧ is_perpendicular (l1 1) (l2 1)) →
    -- l1 intersects the hyperbola at P and Q
    ∃ (P Q : ℝ × ℝ), is_intersection l1 P.1 P.2 ∧ is_intersection l1 Q.1 Q.2 →
    -- l2 intersects the hyperbola at R and S
    ∃ (R S : ℝ × ℝ), is_intersection l2 R.1 R.2 ∧ is_intersection l2 S.1 S.2 →
    -- |PQ| = |RS|
    dist P Q = dist R S
  ) → a = sqrt 2 :=
sorry

end unique_a_l196_196132


namespace correct_sunset_time_l196_196542

-- Definitions corresponding to the conditions
def length_of_daylight : ℕ × ℕ := (10, 30) -- (hours, minutes)
def sunrise_time : ℕ × ℕ := (6, 50) -- (hours, minutes)

-- The reaching goal is to prove the sunset time
def sunset_time (sunrise : ℕ × ℕ) (daylight : ℕ × ℕ) : ℕ × ℕ :=
  let (sh, sm) := sunrise
  let (dh, dm) := daylight
  let total_minutes := sm + dm
  let extra_hour := total_minutes / 60
  let final_minutes := total_minutes % 60
  (sh + dh + extra_hour, final_minutes)

-- The theorem to prove
theorem correct_sunset_time :
  sunset_time sunrise_time length_of_daylight = (17, 20) := sorry

end correct_sunset_time_l196_196542


namespace value_of_x_squared_minus_y_squared_l196_196525

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l196_196525


namespace range_a_l196_196972

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∃ x : ℝ, |x - a| + |x - 1| ≤ 3

theorem range_a (a : ℝ) : range_of_a a → -2 ≤ a ∧ a ≤ 4 :=
sorry

end range_a_l196_196972


namespace counting_intersections_l196_196419

noncomputable def focus := (1, 0 : ℝ × ℝ)
def directrix_lines (a b : ℤ) : (ℤ × ℤ) → Prop := λ x, 
  (a = x.1 ∧ x.1 ∈ {-1, 0, 1, 2}) ∧ (b = x.2 ∧ x.2 ∈ {-2, -1, 0, 1, 2})

def unique_pair_intersections (parabolas : set (ℤ × ℤ)) : Prop :=
  ∀ p₁ p₂ p₃ ∈ parabolas, p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃ →
  ¬ ∃ pt : ℝ × ℝ, pt ∈ (intersection p₁ p₂) ∧
                   pt ∈ (intersection p₂ p₃) ∧
                   pt ∈ (intersection p₃ p₁)
                   
theorem counting_intersections : 
  ∀ (parabolas : set(ℤ × ℤ)),
  (∀ p ∈ parabolas, directrix_lines p.1 p.2 p) →
  (parabolas.card = 25) →
  unique_pair_intersections parabolas →
  count_intersections parabolas = 568 :=
by
  sorry

end counting_intersections_l196_196419


namespace locus_of_P_l196_196072

theorem locus_of_P
  (a b x y : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : (x ≠ 0 ∧ y ≠ 0))
  (h4 : x^2 / a^2 - y^2 / b^2 = 1) :
  (x / a)^2 - (y / b)^2 = ((a^2 + b^2) / (a^2 - b^2))^2 := by
  sorry

end locus_of_P_l196_196072


namespace find_d_l196_196041

theorem find_d (c d : ℤ) (h : ∃ k : ℤ, (λ x : ℤ, x^3 - 2 * x^2 - x + 2) * (λ x : ℤ, c * x + k) = (λ x : ℤ, c * x^4 + d * x^3 - 2 * x^2 + 2)) : d = -1 :=
sorry

end find_d_l196_196041


namespace ellipse_equation_l196_196477

variables (a b c : ℝ)
variables (a_pos : a > b) (b_pos : b > 0)
variables (ell_eq : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ ((a^2 = b^2 + c^2) ∧ (2 * abs (b * sqrt (4 * a^2 - c^2)) / (2 * a) * (c / 2) = sqrt (3))))
variables (area_eq : 2 * abs (b * sqrt (4 * a^2 - c^2)) / (2 * a) * (c / 2) = sqrt (3))
variables (foci_cond : a^2 = b^2 + c^2)

theorem ellipse_equation : (2 * sqrt(3)) + 4 = a^2 ∧ 2 * sqrt(3) = b^2 → 
  ∀ (x y : ℝ), (x^2 / ((2 * sqrt (3)) + 4) + y^2 / (2 * sqrt (3)) = 1) := 
by
  sorry

end ellipse_equation_l196_196477


namespace sum_of_ages_l196_196153

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196153


namespace find_x_values_l196_196428

theorem find_x_values (x : ℝ) :
  (x^2 - 3 * x > 8) ∧ (|x| > 2) ↔ x ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 4 ∞ :=
by
  sorry

end find_x_values_l196_196428


namespace binary_to_hexadecimal_l196_196420

-- Definition of the conversion function from binary to decimal
def bin_to_dec (n : nat) : nat :=
if n = 0 then 0 else (if n % 10 != 0 then 1 else 0) * 2 ^ ((nat.log 2 n)) + bin_to_dec (n / 10)

-- Auxiliary definition for base 10
def dec_to_base (n b : nat) : list nat :=
if n < b then [n] else (n % b) :: dec_to_base (n / b) b

-- Define the main proof problem
theorem binary_to_hexadecimal : 
  (dec_to_base (bin_to_dec 1011001) 6 = [2, 2, 5]) :=
by
  -- Put the proof code here
  sorry

end binary_to_hexadecimal_l196_196420


namespace benny_march_savings_l196_196786

theorem benny_march_savings :
  (january_add : ℕ) (february_add : ℕ) (march_total : ℕ) 
  (H1 : january_add = 19) (H2 : february_add = 19) (H3 : march_total = 46) :
  march_total - (january_add + february_add) = 8 := 
by
  sorry

end benny_march_savings_l196_196786


namespace digit_in_ten_thousandths_place_of_fraction_l196_196284

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196284


namespace min_distance_l196_196055

theorem min_distance (P Q : ℝ × ℝ) (hP : ∃ x y : ℝ, P = (x, y) ∧ sqrt 3 * x - y + 2 = 0)
    (hQ : ∃ x y : ℝ, Q = (x, y) ∧ x^2 + y^2 + 2 * y = 0) :
    ∃ d : ℝ, d = 1 / 2 :=
by
  sorry

end min_distance_l196_196055


namespace betty_berries_july_five_l196_196787
open Nat

def betty_bear_berries : Prop :=
  ∃ (b : ℕ), (5 * b + 100 = 150) ∧ (b + 40 = 50)

theorem betty_berries_july_five : betty_bear_berries :=
  sorry

end betty_berries_july_five_l196_196787


namespace problem_A_problem_B_problem_C_problem_D_l196_196883

variables {a b : Real}

theorem problem_A (h : a > 0) (h1 : b > 0) (h2 : ab - a - 2b = 0) : a + 2b ≥ 8 := 
sorry

theorem problem_B (h : a > 0) (h1 : b > 0) : ¬(a^2 + b^2 ≥ 2 * (a + b + 1)) := 
sorry

theorem problem_C (h : a > 0) (h1 : b > 0) : (a^2 / b + b^2 / a) ≥ a + b := 
sorry

theorem problem_D (h : a > 0) (h1 : b > 0) (h2 : 1 / (a + 1) + 1 / (b + 2) = 1 / 3) : ab + a + b ≥ 14 + 6 * Real.sqrt 6 := 
sorry

end problem_A_problem_B_problem_C_problem_D_l196_196883


namespace value_of_a_plus_b_l196_196024

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l196_196024


namespace complex_number_z_satisfies_l196_196046

theorem complex_number_z_satisfies (z : ℂ) : 
  (z * (1 + I) + (-I) * (1 - I) = 0) → z = -1 := 
by {
  sorry
}

end complex_number_z_satisfies_l196_196046


namespace number_of_carving_methods_l196_196473

-- Definitions for conditions
def isOppositeFaces (c1 c2 : ℕ) : Prop :=
  (c1, c2) ∈ [(1, 6), (2, 5), (3, 4), (6, 1), (5, 2), (4, 3)]

def validCarving (carving : ℕ → ℕ) : Prop :=
  carving 1 = 1 ∧ carving 6 = 6 ∧ 
  carving 2 = 2 ∧ carving 5 = 5 ∧
  carving 3 = 3 ∧ carving 4 = 4 ∧
  carving 6 = 1 ∧ carving 1 = 6 ∧
  carving 5 = 2 ∧ carving 2 = 5 ∧
  carving 4 = 3 ∧ carving 3 = 4

-- The theorem statement
theorem number_of_carving_methods :
  (Σ (f : ℕ → ℕ), validCarving f) ≃_presheaf_to_presheaf (8 Data:8 Kind:Eq) 48 := sorry

end number_of_carving_methods_l196_196473


namespace quadratic_inequality_solution_set_l196_196004

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end quadratic_inequality_solution_set_l196_196004


namespace roundSumAndMultiply_l196_196598

noncomputable def addAndMultiply (x y : ℝ) : ℝ := (x + y) * 2

noncomputable def roundToNearestTenth (n : ℝ) : ℝ :=
  let scaled := n * 10
  let floored := Real.floor (scaled + 0.5)
  floored / 10

theorem roundSumAndMultiply :
  roundToNearestTenth (addAndMultiply 158.23 47.869) = 412.2 :=
by
  sorry

end roundSumAndMultiply_l196_196598


namespace find_length_MN_l196_196557

variables (A B C D M N : Type) [geometry_space A B C D M N]

-- Definitions based on the conditions
def trapezoid (A B C D : Type) [geometry_space A B C D] : Prop :=
  parallel (B C) (A D)

def length (A B : Type) [geometry_space A B] (l : ℝ) : Prop :=
  distance A B = l

def angle (A B C : Type) [geometry_space A B C] (θ : ℝ) : Prop :=
  measure_angle A B C = θ

def midpoint (M : Type) [geometry_space M] {A B : Type} [geometry_space A B] : Prop :=
  distance A M = distance M B ∧
  collinear A M B

-- Problem statement
theorem find_length_MN
  (ABCD_is_trapezoid : trapezoid A B C D)
  (BC_length : length B C 1100)
  (AD_length : length A D 2200)
  (angle_A_45 : angle D A B 45)
  (angle_D_45 : angle A D C 45)
  (M_midpoint : midpoint M B C)
  (N_midpoint : midpoint N A D) :
  length M N 550 :=
sorry

end find_length_MN_l196_196557


namespace arithmetic_sequence_mod_l196_196415

theorem arithmetic_sequence_mod :
  let a := 2
  let d := 5
  let l := 137
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  n = 28 ∧ S = 1946 →
  S % 20 = 6 :=
by
  intros h
  sorry

end arithmetic_sequence_mod_l196_196415


namespace value_of_a_add_b_l196_196028

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l196_196028


namespace isosceles_triangle_perpendicular_l196_196130

/-- In an isosceles triangle ABC with |AB| = |BC|, 
  D is the midpoint of AC,
  E is the projection of D onto BC,
  F is the midpoint of DE, 
  then the lines BF and AE are perpendicular. -/
theorem isosceles_triangle_perpendicular (A B C D E F : ℝ → ℝ → Prop)
    (h_iso: ∃ (A B C: ℝ → ℝ → Prop), is_isosceles A B C)
    (h_mid_D: midpoint A C D)
    (h_proj_E: projection D B C E)
    (h_mid_F: midpoint D E F)
    : perpendicular B F A E := 
  sorry
  
/-- Definitions used in the theorem -/
def is_isosceles (A B C : ℝ → ℝ → Prop) : Prop :=
  ∃ (AB BC : ℝ), AB = BC

def midpoint (P Q R : ℝ → ℝ → Prop) : Prop :=
  P (λ x y, (Q (λ a b, a), Q (λ a b, b) / 2)) ∧ P (λ x y, (R (λ a b, a), R (λ a b, b) / 2))

def projection (P Q Q' R: ℝ → ℝ → Prop) : Prop :=
  ∀ (x y: ℝ), R (λ a b, a - b = (x - y) / (Q (λ c d, d) - Q' (λ e f, f)))

def perpendicular (P Q R S: ℝ → ℝ → Prop) : Prop :=
  ∃ (m n : ℝ), m * n = -1

end isosceles_triangle_perpendicular_l196_196130


namespace log_b_1024_number_of_positive_integers_b_l196_196513

theorem log_b_1024 (b : ℕ) : (∃ n : ℕ, b^n = 1024) ↔ b ∈ {2, 4, 32, 1024} :=
by sorry

theorem number_of_positive_integers_b : (∃ b : ℕ, ∃ n : ℕ, b^n = 1024 ∧ n > 0) ↔ 4 :=
by {
  have h := log_b_1024,
  sorry
}

end log_b_1024_number_of_positive_integers_b_l196_196513


namespace solve_mod_problem_l196_196678

theorem solve_mod_problem :
  ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [MOD 9] ∧ n = 8 :=
by
  sorry

end solve_mod_problem_l196_196678


namespace correct_intersection_l196_196932

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem correct_intersection : M ∩ N = {2, 3} := by sorry

end correct_intersection_l196_196932


namespace percentage_increase_of_allowance_l196_196569

-- Define the allowances as described in the conditions
def middle_school_allowance := 8 + 2
def senior_year_allowance := (2 * middle_school_allowance) + 5

-- % increase function
def percentage_increase (old new : ℕ) : ℝ := ((new - old) / old.toReal) * 100

-- The theorem stating the proof problem
theorem percentage_increase_of_allowance : 
  let old := middle_school_allowance in
  let new := senior_year_allowance in
  percentage_increase old new = 150 := by 
  sorry

end percentage_increase_of_allowance_l196_196569


namespace hyperbola_eccentricity_l196_196529

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > b) (hb : b > 0)
  (h_ellipse : (a^2 - b^2) / a^2 = 3 / 4) :
  (a^2 + b^2) / a^2 = 5 / 4 :=
by
  -- We start with the given conditions and need to show the result
  sorry  -- Proof omitted

end hyperbola_eccentricity_l196_196529


namespace nine_chapters_correct_statements_l196_196704

theorem nine_chapters_correct_statements
  (a b c : ℝ)
  (S: ℝ)
  (h1: a/b = 2/3)
  (h2: b/c = 3/4)
  (h3: S = 3 * real.sqrt 15)
  (incircle_area_correct: ℝ)
  (dot_product_correct: ℝ):
  (incircle_area_correct = (5/3) * real.pi) ∧ (dot_product_correct = -22) :=
by
  sorry

end nine_chapters_correct_statements_l196_196704


namespace find_other_integer_l196_196206

theorem find_other_integer (x y : ℤ) (h1 : 3*x + 4*y = 103) (h2 : x = 19 ∨ y = 19) : x = 9 ∨ y = 9 :=
by sorry

end find_other_integer_l196_196206


namespace median_is_31_l196_196534

def data_set : List ℕ := [31, 35, 31, 33, 30, 33, 31]

theorem median_is_31 : (data_set.toFinset.median = 31) :=
sorry

end median_is_31_l196_196534


namespace prove_range_of_a_l196_196495

noncomputable def f (x a : ℝ) : ℝ := (x + a - 1) * Real.exp x

def problem_condition1 (x a : ℝ) : Prop := 
  f x a ≥ (x^2 / 2 + a * x)

def problem_condition2 (x : ℝ) : Prop := 
  x ∈ Set.Ici 0 -- equivalent to [0, +∞)

theorem prove_range_of_a (a : ℝ) :
  (∀ x : ℝ, problem_condition2 x → problem_condition1 x a) → a ∈ Set.Ici 1 :=
sorry

end prove_range_of_a_l196_196495


namespace find_hyperbola_equation_l196_196504

noncomputable def hyperbola_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a = 2 ∧ b^2 = 3

theorem find_hyperbola_equation (a b : ℝ) : 
  hyperbola_equation a b → 
  (a = 2 ∧ b = sqrt 3) → 
  (∀ x y : ℝ, (x^2 / 4 - y^2 / 3 = 1)) :=
by
  intro h_eq h_vals
  sorry

end find_hyperbola_equation_l196_196504


namespace hyperbola_slope_reciprocals_l196_196071

theorem hyperbola_slope_reciprocals (P : ℝ × ℝ) (t : ℝ) :
  (P.1 = t ∧ P.2 = - (8 / 9) * t ∧ t ≠ 0 ∧  
    ∃ k1 k2: ℝ, k1 = - (8 * t) / (9 * (t + 3)) ∧ k2 = - (8 * t) / (9 * (t - 3)) ∧
    (1 / k1) + (1 / k2) = -9 / 4) ∧
    ((P = (9/5, -(8/5)) ∨ P = (-(9/5), 8/5)) →
        ∃ kOA kOB kOC kOD : ℝ, (kOA + kOB + kOC + kOD = 0)) := 
sorry

end hyperbola_slope_reciprocals_l196_196071


namespace score_difference_l196_196770

theorem score_difference 
  (x y z w : ℝ)
  (h1 : x = 2 + (y + z + w) / 3)
  (h2 : y = (x + z + w) / 3 - 3)
  (h3 : z = 3 + (x + y + w) / 3) :
  (x + y + z) / 3 - w = 2 :=
by {
  sorry
}

end score_difference_l196_196770


namespace soccer_lineup_count_l196_196599

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem soccer_lineup_count : 
  let total_players := 18
  let goalies := 1
  let defenders := 6
  let forwards := 4
  18 * choose 17 6 * choose 11 4 = 73457760 :=
by
  sorry

end soccer_lineup_count_l196_196599


namespace max_profit_and_investment_l196_196659

variables (x : ℝ) (t : ℝ)

/-- The profit functions for goods A and B. -/
def profit_A (c : ℝ) := c / 4
def profit_B (c : ℝ) := (3 / 4) * real.sqrt (c - 1)

/-- The total profit function considering the constraints. -/
noncomputable def total_profit (x : ℝ) : ℝ :=
  (1 / 4) * (8 - x) + (3 / 4) * real.sqrt (x - 1)

/-- The rewritten total profit function in terms of t. -/
noncomputable def total_profit_t (t : ℝ) : ℝ :=
  (1 / 4) * (7 - t^2) + (3 / 4) * t

theorem max_profit_and_investment :
  1 ≤ x ∧ x ≤ 8 ∧ x = t^2 + 1 → 
  (total_profit x = (37 / 16) ∧ 
  (8 - x = 19 / 4) ∧ 
  (x = 13 / 4)) :=
sorry

end max_profit_and_investment_l196_196659


namespace problem_1_problem_2_l196_196063

noncomputable def f (a b x : ℝ) := a * (x - 1)^2 + b * Real.log x

theorem problem_1 (a : ℝ) (h_deriv : ∀ x ≥ 2, (2 * a * x^2 - 2 * a * x + 1) / x ≤ 0) : 
  a ≤ -1 / 4 :=
sorry

theorem problem_2 (a : ℝ) (h_ineq : ∀ x ≥ 1, a * (x - 1)^2 + Real.log x ≤ x - 1) : 
  a ≤ 0 :=
sorry

end problem_1_problem_2_l196_196063


namespace centers_of_circles_in_relation_l196_196572

open EuclideanGeometry

def Circle := {center : Point ℝ × ℝ, radius : ℝ}

def point_in_or_on_circle (P : Point ℝ × ℝ) (c : Circle) : Prop :=
  let (O, r) := c
  dist P O ≤ r

theorem centers_of_circles_in_relation (P : Point ℝ × ℝ) (ω : Fin 6 → Circle)
  (h₀ : ∀ i, point_in_or_on_circle P (ω i)) :
  ∃ i j : Fin 6, i ≠ j ∧ point_in_or_on_circle (ω i).center (ω j) :=
begin
  sorry
end

end centers_of_circles_in_relation_l196_196572


namespace num_zeros_in_fraction_decimal_l196_196099

theorem num_zeros_in_fraction_decimal :
  (let x := (1 : ℚ) / (2^3 * 5^6) in
   ∃ k : ℕ, x = 8 / 10^6 ∧ k = 5) :=
by
  -- The proof would involve showing x can be written as '0.000008',
  -- which has exactly 5 zeros between the decimal point and the first non-zero digit.
  sorry

end num_zeros_in_fraction_decimal_l196_196099


namespace a5_is_9_l196_196424

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1989 ^ 1989
  else (a_seq (n - 1)).digits.sum

theorem a5_is_9 : a_seq 5 = 9 := by
  sorry

end a5_is_9_l196_196424


namespace tangent_line_at_P_range_of_y_compare_exp_m1_m_e1_l196_196500

noncomputable def f (m : ℝ) (x : ℝ) := m * Real.exp x - x - 1
noncomputable def f_prime (m : ℝ) (x : ℝ) := m * Real.exp x - 1

theorem tangent_line_at_P :
  (f 2 0 = 1) → 
  (f_prime 2 0 = 1) → 
  ∀ x y : ℝ, y = x + 1 ↔ x - y + 1 = 0 := 
by {
  sorry
}

theorem range_of_y 
  (x1 x2 t : ℝ) (m : ℝ) 
  (h_zero_points: x1 < x2) 
  (h_zero1: f m x1 = 0) (h_zero2: f m x2 = 0)
  (h_t: t = x2 - x1):
  (e ^ x2 - e ^ x1) * (1 / (e ^ x2 + e ^ x1) - m) ∈ (-∞, 0) := 
by {
  sorry
}

theorem compare_exp_m1_m_e1 
  (m : ℝ) (h_positive : ∀ x : ℝ, f m x > 0):
  if (1 < m ∧ m < Real.exp 1) then (e ^ (m - 1) < m ^ (Real.exp 1 - 1))
  else if (m = Real.exp 1) then (e ^ (m - 1) = m ^ (Real.exp 1 - 1))
  else if (m > Real.exp 1) then (e ^ (m - 1) > m ^ (Real.exp 1 - 1)) :=
by {
  sorry
}

end tangent_line_at_P_range_of_y_compare_exp_m1_m_e1_l196_196500


namespace extremum_of_function_l196_196446

theorem extremum_of_function (k : ℝ) (h₀ : k ≠ 1) :
  (k > 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≤ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) ∧
  (k < 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≥ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) :=
by
  sorry

end extremum_of_function_l196_196446


namespace hansel_album_duration_l196_196087

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l196_196087


namespace intersect_complement_l196_196964

open Finset

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def A := {1, 3, 6} : Finset ℕ
def B := {2, 3, 4} : Finset ℕ

-- Define the complement of B in U
def complement_U_B := U \ B

-- The statement to prove
theorem intersect_complement : A ∩ complement_U_B = {1, 6} :=
by sorry

end intersect_complement_l196_196964


namespace reflection_squared_is_identity_l196_196169

variable {α : Type*} [Field α] [Module α (Matrix (Fin 2) (Fin 2) α)]

-- Define the reflection matrix over a given vector
def reflection_matrix (v : Vector α) : Matrix (Fin 2) (Fin 2) α :=
  let ⟨a, b⟩ := v in 
  let norm_sq := a * a + b * b in
  Matrix.ofList 2 2 [[(a * a - b * b) / norm_sq, 2 * a * b / norm_sq],
                     [2 * a * b / norm_sq, (b * b - a * a) / norm_sq]]

-- Given vector (4, 2)
def S : Vector α := ![4, 2]

-- Prove that the square of the reflection matrix is the identity matrix
theorem reflection_squared_is_identity : reflection_matrix S * reflection_matrix S = 1 :=
  sorry

end reflection_squared_is_identity_l196_196169


namespace Martha_should_buy_84oz_of_apples_l196_196186

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end Martha_should_buy_84oz_of_apples_l196_196186


namespace specify_points_l196_196217

/-- Specify 1997 points in the plane such that:
  1. The distance between any two points is an integer.
  2. Every line contains at most 100 of these points.
-/
theorem specify_points : ∃ (P : fin 1997 → ℝ × ℝ), 
  (∀ i j, i ≠ j → ∃ n : ℕ, (P i - P j).norm = n) ∧
  (∀ l : ℝ × ℝ → Prop, ((∀ i, l (P i)) → finset.univ.card ≤ 100)) := sorry

end specify_points_l196_196217


namespace birthday_day_of_week_l196_196411

def day_of_week_after_days (starting_day : ℕ) (days_after : ℕ) : ℕ :=
  (starting_day + days_after) % 7

theorem birthday_day_of_week :
    day_of_week_after_days 0 75 = 5 := 
begin
  -- where 0 represents Sunday,
  -- and 5 represents Friday according to problem's context
  sorry
end

end birthday_day_of_week_l196_196411


namespace number_of_solutions_l196_196429

theorem number_of_solutions (x : ℤ) : 
  set.count {x : ℤ | (x - 3) ^ 2 < 9} = 5 :=
sorry

end number_of_solutions_l196_196429


namespace orange_juice_fraction_l196_196675

theorem orange_juice_fraction 
    (capacity1 capacity2 : ℕ)
    (orange_fraction1 orange_fraction2 : ℚ)
    (h_capacity1 : capacity1 = 800)
    (h_capacity2 : capacity2 = 700)
    (h_orange_fraction1 : orange_fraction1 = 1/4)
    (h_orange_fraction2 : orange_fraction2 = 1/3) :
    (capacity1 * orange_fraction1 + capacity2 * orange_fraction2) / (capacity1 + capacity2) = 433.33 / 1500 :=
by sorry

end orange_juice_fraction_l196_196675


namespace escalator_times_comparison_l196_196758

variable (v v1 v2 l : ℝ)
variable (h_v_lt_v1 : v < v1)
variable (h_v1_lt_v2 : v1 < v2)

theorem escalator_times_comparison
  (h_cond : v < v1 ∧ v1 < v2) :
  (l / (v1 + v) + l / (v2 - v)) < (l / (v1 - v) + l / (v2 + v)) :=
  sorry

end escalator_times_comparison_l196_196758


namespace micheal_item_count_l196_196192

theorem micheal_item_count : ∃ a b c : ℕ, a + b + c = 50 ∧ 60 * a + 500 * b + 400 * c = 10000 ∧ a = 30 :=
  by
    sorry

end micheal_item_count_l196_196192


namespace Robert_salary_loss_l196_196614

theorem Robert_salary_loss (S : ℝ) (x : ℝ) (h : x ≠ 0) (h1 : (S - (x/100) * S + (x/100) * (S - (x/100) * S) = (96/100) * S)) : x = 20 :=
by sorry

end Robert_salary_loss_l196_196614


namespace total_minutes_to_finish_album_l196_196089

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l196_196089


namespace total_weight_of_apples_l196_196189

/-- Define the weight of an apple and an orange -/
def apple_weight := 4
def orange_weight := 3

/-- Define the maximum weight a bag can hold -/
def max_bag_weight := 49

/-- Define the number of bags Marta buys -/
def num_bags := 3

/-- Prove the total weight of apples Marta should buy -/
theorem total_weight_of_apples : 
    ∀ (A : ℕ), 4 * A + 3 * A ≤ 49 → A = 7 → 4 * A * 3 = 84 :=
by 
    intros A h1 h2
    rw [h2]
    norm_num 
    sorry

end total_weight_of_apples_l196_196189


namespace function_has_zero_in_interval_l196_196238

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x - 2

theorem function_has_zero_in_interval (x : ℝ) (h1 : 2 < x) (h2 : x < 3) : ∃ c ∈ (2,3), f c = 0 :=
by
  sorry

end function_has_zero_in_interval_l196_196238


namespace projection_3_4_matrix_l196_196856

open Matrix -- to make matrix operations more straightforward

def projection_matrix (v : Vector ℝ 2) : Matrix (Fin 2) (Fin 2) ℝ :=
  let w := ![3, 4]
  (outer_product w w) • (1 / (dot_product w w))

theorem projection_3_4_matrix :
  projection_matrix ![3, 4] = !!( 9/25, 12/25; 12/25, 16/25 ) :=
by
  sorry

end projection_3_4_matrix_l196_196856


namespace combined_rise_in_water_level_is_12_58_cm_l196_196723

def cube_volume (a : ℝ) : ℝ := a^3
def box_volume (l w h : ℝ) : ℝ := l * w * h
def base_area (length width : ℝ) : ℝ := length * width
def rise_in_water_level (v : ℝ) (base_area : ℝ) : ℝ := v / base_area

theorem combined_rise_in_water_level_is_12_58_cm :
  let cube_edge := 15
  let box_length := 10
  let box_width := 5
  let box_height := 8
  let vessel_base_length := 20
  let vessel_base_width := 15
  let cube_vol := cube_volume cube_edge
  let box_vol := box_volume box_length box_width box_height
  let total_volume := cube_vol + box_vol
  let area := base_area vessel_base_length vessel_base_width
  let rise := rise_in_water_level total_volume area
  in rise ≈ 12.58 := 
by
  -- Definitions and conditions provided
  let cube_edge := 15
  let box_length := 10
  let box_width := 5
  let box_height := 8
  let vessel_base_length := 20
  let vessel_base_width := 15
  let cube_vol := cube_volume cube_edge
  let box_vol := box_volume box_length box_width box_height
  let total_volume := cube_vol + box_vol
  let area := base_area vessel_base_length vessel_base_width
  let rise := rise_in_water_level total_volume area
  -- We expect the rise in water level to be approximately 12.58 cm
  have h_approx : rise ≈ 12.58 := by sorry
  exact h_approx

end combined_rise_in_water_level_is_12_58_cm_l196_196723


namespace asymptotes_of_hyperbola_l196_196485

theorem asymptotes_of_hyperbola
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (eccentricity_product : (sqrt (a^2 - b^2) / a) * (sqrt (a^2 + b^2) / a) = sqrt 15 / 4)
  : (∀ x y : ℝ, y = (1/2) * x → x - 2 * y = 0) ∧ (∀ x y : ℝ, y = -(1/2) * x → x + 2 * y = 0) := 
sorry

end asymptotes_of_hyperbola_l196_196485


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196289

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196289


namespace John_l196_196567

variables {j d s : ℕ}

theorem John's_age : 
  (j = d - 20) → 
  (j + d = 80) → 
  (s = 1 / 2 * j) → 
  j = 30 :=
by {
  assume h1 h2 h3,
  sorry
}

end John_l196_196567


namespace unique_solution_l196_196441

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (f : ℝ → ℝ) : ∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y

theorem unique_solution (f : ℝ → ℝ) (hf : ∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y) :
  (f = λ x, x - 1) ∨ (f = λ x, -x - 1) := 
sorry

end unique_solution_l196_196441


namespace sum_of_coefficients_l196_196233

theorem sum_of_coefficients :
  ∃ a b c d e : ℤ, 
    27 * (x : ℝ)^3 + 64 = (a * x + b) * (c * x^2 + d * x + e) ∧ 
    a + b + c + d + e = 20 :=
by
  sorry

end sum_of_coefficients_l196_196233


namespace diff_of_squares_l196_196685

theorem diff_of_squares (x y : ℝ) :
  (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x + y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x - y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (x + 2) * (2 + x)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (2x + 3) * (3x - 2)) := 
by 
  sorry

end diff_of_squares_l196_196685


namespace g_at_10_l196_196425

noncomputable def g (n : ℕ) : ℝ := sorry

axiom g_definition : g 2 = 4
axiom g_recursive : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (3 * g (2 * m) + g (2 * n)) / 4

theorem g_at_10 : g 10 = 64 := sorry

end g_at_10_l196_196425


namespace probability_six_greater_than_five_over_six_l196_196744

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l196_196744


namespace projection_3_4_matrix_l196_196857

open Matrix -- to make matrix operations more straightforward

def projection_matrix (v : Vector ℝ 2) : Matrix (Fin 2) (Fin 2) ℝ :=
  let w := ![3, 4]
  (outer_product w w) • (1 / (dot_product w w))

theorem projection_3_4_matrix :
  projection_matrix ![3, 4] = !!( 9/25, 12/25; 12/25, 16/25 ) :=
by
  sorry

end projection_3_4_matrix_l196_196857


namespace find_a_plus_c_l196_196367

noncomputable def angle_bisector_coefs (a c : ℤ) : Prop :=
  ∃ P Q R : ℤ × ℤ, P = (-7, 6) ∧ Q = (-12, -20) ∧ R = (2, -8) ∧
  (angle_bisector_eq (P, Q, R) = (a, 3, c))

theorem find_a_plus_c : ∃ a c : ℤ, angle_bisector_coefs a c ∧ a + c = 123 :=
by {
  sorry
}

end find_a_plus_c_l196_196367


namespace equation_of_tangent_line_l196_196549

theorem equation_of_tangent_line
  (c1 c2 : ℝ → ℝ → Prop)
  (P : ℝ × ℝ) (l : ℝ → Prop)
  (h1 : P = (3, 2))
  (h2 : ∃ r1 r2, r1 * r2 = 13 / 2 ∧ c1 (λ x, (x - r1 * cot(α))^2 + (P.snd - r1)^2 = r1^2) ∧ c2 (λ x, (x - r2 * cot(α))^2 + (P.snd - r2)^2 = r2^2))
  (h3 : l = (λ x, x * tan(2 * α)))
  : l = (λ x, 2 * √2 * x) :=
sorry

end equation_of_tangent_line_l196_196549


namespace zookeeper_configurations_l196_196776

theorem zookeeper_configurations :
  ∃ (configs : ℕ), configs = 3 ∧ 
  (∀ (r p : ℕ), 
    30 * r + 35 * p = 1400 ∧ p ≥ r → 
    ((r, p) = (7, 34) ∨ (r, p) = (14, 28) ∨ (r, p) = (21, 22))) :=
sorry

end zookeeper_configurations_l196_196776


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196286

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196286


namespace num_zeros_in_fraction_decimal_l196_196100

theorem num_zeros_in_fraction_decimal :
  (let x := (1 : ℚ) / (2^3 * 5^6) in
   ∃ k : ℕ, x = 8 / 10^6 ∧ k = 5) :=
by
  -- The proof would involve showing x can be written as '0.000008',
  -- which has exactly 5 zeros between the decimal point and the first non-zero digit.
  sorry

end num_zeros_in_fraction_decimal_l196_196100


namespace probability_six_greater_than_five_over_six_l196_196747

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l196_196747


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196299

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196299


namespace problem_ineq_l196_196486

theorem problem_ineq (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := 
by 
  sorry

end problem_ineq_l196_196486


namespace max_value_of_quadratic_l196_196861

theorem max_value_of_quadratic : ∃ x : ℝ, (∀ y : ℝ, (-3 * y^2 + 9 * y - 1) ≤ (-3 * (3/2)^2 + 9 * (3/2) - 1)) ∧ x = 3/2 :=
by
  sorry

end max_value_of_quadratic_l196_196861


namespace missing_number_is_odd_l196_196084

open Finset

noncomputable def SetA_and_SetBSum_even_probability_0_5 (x : ℕ) : Prop :=
  let a := {11, 44, x}
  let b := {1}
  (∃ n ∈ a, (n + 1) % 2 = 0) ∧ (∃ m ∈ a, (m + 1) % 2 ≠ 0)

theorem missing_number_is_odd : ∀ x : ℕ, SetA_and_SetBSum_even_probability_0_5 x → x % 2 ≠ 0 ∧ x ≠ 11 :=
  sorry

end missing_number_is_odd_l196_196084


namespace arithmetic_sequence_a5_l196_196545

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_l196_196545


namespace true_propositions_l196_196512

-- Define the conditions in Lean
section

variable (tetrahedron : Type) 
variable (is_regular : tetrahedron → Prop)
variable (is_equilateral_base : tetrahedron → Prop)
variable (equal_dihedral_angles : tetrahedron → Prop)
variable (isosceles_lateral_faces : tetrahedron → Prop)
variable (equal_areas_lateral_faces : tetrahedron → Prop)
variable (equal_angles_lateral_edges : tetrahedron → Prop)

-- Propositions corresponding to the conditions
def prop1 : tetrahedron → Prop :=
λ t, is_equilateral_base t ∧ equal_dihedral_angles t → is_regular t

def prop2 : tetrahedron → Prop :=
λ t, is_equilateral_base t ∧ isosceles_lateral_faces t → is_regular t

def prop3 : tetrahedron → Prop :=
λ t, is_equilateral_base t ∧ equal_areas_lateral_faces t → is_regular t

def prop4 : tetrahedron → Prop :=
λ t, equal_angles_lateral_edges t ∧ equal_dihedral_angles t → is_regular t

-- The theorem statement indicating which propositions are true
theorem true_propositions (t : tetrahedron) : 
prop1 t ∧ prop4 t ∧ ¬ (prop2 t ∧ prop3 t) :=
by
  sorry

end

end true_propositions_l196_196512


namespace sqrt_of_16_is_4_l196_196408

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l196_196408


namespace digit_in_ten_thousandths_place_l196_196312

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196312


namespace compare_points_on_quadratic_graph_l196_196489

theorem compare_points_on_quadratic_graph :
  let y1 := (3 - 1) ^ 2,
      y2 := (1 - 1) ^ 2
  in y1 > y2 :=
by
  let y1 := (3 - 1) ^ 2
  let y2 := (1 - 1) ^ 2
  sorry

end compare_points_on_quadratic_graph_l196_196489


namespace log_product_identity_l196_196702

theorem log_product_identity :
  log 3 2 * log 4 3 * log 5 4 * log 6 5 * log 7 6 * log 8 7 = 1 / 3 :=
by
  sorry

end log_product_identity_l196_196702


namespace find_a_perpendicular_lines_l196_196042

theorem find_a_perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, ax + (a + 2) * y + 1 = 0 ∧ x + a * y + 2 = 0) → a = -3 :=
sorry

end find_a_perpendicular_lines_l196_196042


namespace tangent_and_normal_at_t_eq_pi_div4_l196_196879

def tangent_line_equation (t: ℝ) := - (4 / 3) * t + 4 * Real.sqrt 2
def normal_line_equation (t: ℝ) := (3 / 4) * t + (7 * Real.sqrt 2) / 8

theorem tangent_and_normal_at_t_eq_pi_div4 :
  (tangent_line_equation (3 * Real.cos (Real.pi / 4)) = 4 * Real.sqrt 2) ∧
  (normal_line_equation (3 * Real.cos (Real.pi / 4)) = (7 * Real.sqrt 2) / 8) :=
by
  sorry

end tangent_and_normal_at_t_eq_pi_div4_l196_196879


namespace intersection_of_C1_and_C2_l196_196062

noncomputable def C1_rect_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def C2_param_eqn (x y t : ℝ) : Prop := x = 3 - (1/2) * t ∧ y = (sqrt 3 / 2) * t
noncomputable def C2_gen_eqn (x y : ℝ) : Prop := sqrt 3 * x + y - 3 * sqrt 3 = 0
def point_A : ℝ × ℝ := (3, 0)
noncomputable def dist (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_of_C1_and_C2 :
  ∃ (t1 t2 : ℝ), 
  (C1_rect_eqn (3 - (1/2) * t1) ((sqrt 3 / 2) * t1) ∧ C1_rect_eqn (3 - (1/2) * t2) ((sqrt 3 / 2) * t2)) ∧
  (dist point_A (3 - (1/2) * t1, (sqrt 3 / 2) * t1) * dist point_A (3 - (1/2) * t2, (sqrt 3 / 2) * t2) = 3) :=
sorry

end intersection_of_C1_and_C2_l196_196062


namespace triangle_angle_relation_l196_196334

noncomputable def triangle_condition (a b c : ℝ) : Prop :=
  b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3

noncomputable def angle_relation (α β γ : ℝ) : Prop :=
  (1 / (Real.sqrt α + Real.sqrt β)) + (1 / (Real.sqrt β + Real.sqrt γ)) = 2 / (Real.sqrt γ + Real.sqrt α)

theorem triangle_angle_relation (a b c α β γ : ℝ)
  (h1 : triangle_condition a b c)
  (h2 : α = measure_of_angle a b c)
  (h3 : β = measure_of_angle b c a)
  (h4 : γ = measure_of_angle c a b) :
  angle_relation α β γ :=
sorry

end triangle_angle_relation_l196_196334


namespace place_crosses_in_grid_l196_196121

theorem place_crosses_in_grid :
  ∃ (ways : ℕ), ways = 240 ∧ 
    (∀ (r : ℕ) (c : ℕ), r < 4 → c < 5 → ∃ (x : set (Fin 4 × Fin 5)), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), (i, j) ∈ x) ∧ 
      (∀ (j : Fin 5), ∃ (i : Fin 4), (i, j) ∈ x) ∧ 
      x.card = 5) :=
by
  sorry

end place_crosses_in_grid_l196_196121


namespace sum_of_cosines_bounds_l196_196252

theorem sum_of_cosines_bounds (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π / 2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π / 2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ π / 2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ π / 2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ π / 2)
  (sum_sines_eq : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) : 
  2 ≤ Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ∧ 
      Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ≤ 4 :=
by
  sorry

end sum_of_cosines_bounds_l196_196252


namespace best_fitting_model_is_model_3_l196_196137

-- Define models with their corresponding R^2 values
def R_squared_model_1 : ℝ := 0.72
def R_squared_model_2 : ℝ := 0.64
def R_squared_model_3 : ℝ := 0.98
def R_squared_model_4 : ℝ := 0.81

-- Define a proposition that model 3 has the best fitting effect
def best_fitting_model (R1 R2 R3 R4 : ℝ) : Prop :=
  R3 = max (max R1 R2) (max R3 R4)

-- State the theorem that we need to prove
theorem best_fitting_model_is_model_3 :
  best_fitting_model R_squared_model_1 R_squared_model_2 R_squared_model_3 R_squared_model_4 :=
by
  sorry

end best_fitting_model_is_model_3_l196_196137


namespace part_a_part_b_l196_196936

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ :=
  let ⟨a1, a2⟩ := a x
  let ⟨b1, b2⟩ := b x
  a1 * b1 + a2 * b2

noncomputable def g (x : ℝ) : ℝ :=
  Real.cos (2 * x - Real.pi / 3) + 1

theorem part_a (x : ℝ) (h₁ : x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 3)) (h₂ : f x = 0) : x = -Real.pi / 6 :=
sorry

theorem part_b (k : ℤ) : (Set.range g = Set.Icc 0 2) ∧ 
  (∀ x, x ∈ Set.Icc (↑k * Real.pi - Real.pi / 3) (↑k * Real.pi + Real.pi / 6) →
  Real.deriv g x ≥ 0) :=
sorry

end part_a_part_b_l196_196936


namespace probability_roll_6_final_l196_196730

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l196_196730


namespace number_of_correct_propositions_is_2_l196_196242

-- Conditions
def three_points_determine_plane : Prop := false
def trapezoid_determines_plane : Prop := true
def three_lines_intersecting_pairs_max_three_planes : Prop := true
def two_planes_three_common_points_coincide : Prop := false

-- Prove that the number of correct propositions is 2
theorem number_of_correct_propositions_is_2 :
  (three_points_determine_plane → 1) + (trapezoid_determines_plane → 1) +
  (three_lines_intersecting_pairs_max_three_planes → 1) + (two_planes_three_common_points_coincide → 1) = 2 :=
by
  sorry

end number_of_correct_propositions_is_2_l196_196242


namespace area_triangle_CNK_l196_196701

open Classical

variable (ABC : Type) [AffineGeometry ABC] 
variable (A B C M K N : ABC)
variable [hABC : Triangle ABC]
variable [h1 : A ≠ B] [h2 : B ≠ C] [h3 : A ≠ C]
variable (areaABC : ℝ)
variable (areaBMN : ℝ)
variable (areaAMK : ℝ)
variable (midpoint_M : Midpoint M A B)
variable (h_areaABC : areaABC = 75)
variable (h_areaBMN : areaBMN = 15)
variable (h_areaAMK : areaAMK = 25)

theorem area_triangle_CNK :
  ∃ (areaCNK : ℝ), areaCNK = 15 :=
sorry

end area_triangle_CNK_l196_196701


namespace diving_assessment_l196_196433

theorem diving_assessment (total_athletes : ℕ) (selected_athletes : ℕ) (not_meeting_standard : ℕ) 
  (first_level_sample : ℕ) (first_level_total : ℕ) (athletes : Set ℕ) :
  total_athletes = 56 → 
  selected_athletes = 8 → 
  not_meeting_standard = 2 → 
  first_level_sample = 3 → 
  (∀ (A B C D E : ℕ), athletes = {A, B, C, D, E} → first_level_total = 5 → 
  (∃ proportion_standard number_first_level probability_E, 
    proportion_standard = (8 - 2) / 8 ∧  -- first part: proportion of athletes who met the standard
    number_first_level = 56 * (3 / 8) ∧ -- second part: number of first-level athletes
    probability_E = 4 / 10))           -- third part: probability of athlete E being chosen
:= sorry

end diving_assessment_l196_196433


namespace smartphone_cost_decrease_l196_196627

theorem smartphone_cost_decrease :
  ∀ (cost2010 cost2020 : ℝ),
  cost2010 = 600 →
  cost2020 = 450 →
  ((cost2010 - cost2020) / cost2010) * 100 = 25 :=
by
  intros cost2010 cost2020 h1 h2
  sorry

end smartphone_cost_decrease_l196_196627


namespace log_base_5_sqrt_inverse_l196_196818

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196818


namespace sqrt_sixteen_equals_four_l196_196402

theorem sqrt_sixteen_equals_four : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_sixteen_equals_four_l196_196402


namespace average_weight_men_women_l196_196268

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l196_196268


namespace coeff_of_x_105_in_P_l196_196848

-- Definition of the polynomial P(x)
def P (x : ℝ) : ℝ :=
  (x - 1) * (x^2 - 2) * (x^3 - 3) * (x^4 - 4) * (x^5 - 5) * (x^6 - 6) * (x^7 - 7) * 
  (x^8 - 8) * (x^9 - 9) * (x^10 - 10) * (x^11 - 11) * (x^12 - 12) * (x^13 - 13) * 
  (x^14 - 14) * (x^15 - 15)

-- Goal: find the coefficient of x^105 in P(x)
theorem coeff_of_x_105_in_P :
  coefficient_of (x^105) (P x) = c :=
sorry

end coeff_of_x_105_in_P_l196_196848


namespace num_ordered_pairs_l196_196453

theorem num_ordered_pairs (a b : ℤ) (h_dvd_a : a ∣ 720) (h_dvd_b : b ∣ 720) (h_not_dvd_ab : ¬ (a * b ∣ 720)) : 
  (number_of_pairs : ℕ) := 2520 := sorry

end num_ordered_pairs_l196_196453


namespace coordinates_of_B_l196_196358

-- Define the initial conditions
def A : ℝ × ℝ := (-2, 1)
def jump_units : ℝ := 4

-- Define the function to compute the new coordinates after the jump
def new_coordinates (start : ℝ × ℝ) (jump : ℝ) : ℝ × ℝ :=
  let (x, y) := start
  (x + jump, y)

-- State the theorem to be proved
theorem coordinates_of_B
  (A : ℝ × ℝ) (jump_units : ℝ)
  (hA : A = (-2, 1))
  (h_jump : jump_units = 4) :
  new_coordinates A jump_units = (2, 1) := 
by
  -- Placeholder for the actual proof
  sorry

end coordinates_of_B_l196_196358


namespace arithmetic_sequence_properties_l196_196476

variables {a_1 d : ℝ}

def arithmetic_sequence (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

def sum_of_first_n_terms (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

def geometric_progression (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_properties (h1 : d ≠ 0)
  (h2 : geometric_progression (arithmetic_sequence a_1 d 3) (arithmetic_sequence a_1 d 5) (arithmetic_sequence a_1 d 10)) :
  a_1 * d < 0 ∧ d * (sum_of_first_n_terms a_1 d 4) > 0 :=
by
  sorry

end arithmetic_sequence_properties_l196_196476


namespace exist_four_optimal_sequences_l196_196981

theorem exist_four_optimal_sequences (n : ℕ) (h : n > 1) :
  ∃ (xs : Fin 2n → ℤ), 
    (∑ i, |xs i| ≠ 0) ∧  -- Constraint (i)
    (∀ i, |xs i| ≤ n) ∧  -- Constraint (ii)
    (∃ (grid : Fin n → Fin (2 * n) → ℤ),
      (∀ j : Fin n, ∑ i, grid j i = 0) ∧  -- Constraint (iii)
      (∀ i : Fin (2 * n), grid (⟨ i.val / 2, sorry ⟩) i = xs i) 
    ) ∧  -- Grid construction, ensuring each xs maps correctly
    ∃ seqs : Fin 4 → (Fin (2 * n) → ℤ), 
      (∀ k : Fin 4, 
        (∀ i, seqs k i = xs i) ∨ 
        (∃ (k' : Fin 4), seqs k ≠ seqs k')
      )

end exist_four_optimal_sequences_l196_196981


namespace ben_daily_spending_l196_196381

variable (S : ℕ)

def daily_savings (S : ℕ) : ℕ := 50 - S

def total_savings (S : ℕ) : ℕ := 7 * daily_savings S

def final_amount (S : ℕ) : ℕ := 2 * total_savings S + 10

theorem ben_daily_spending :
  final_amount 15 = 500 :=
by
  unfold final_amount
  unfold total_savings
  unfold daily_savings
  sorry

end ben_daily_spending_l196_196381


namespace positivity_of_fraction_l196_196909

theorem positivity_of_fraction
  (a b c d x1 x2 x3 x4 : ℝ)
  (h_neg_a : a < 0)
  (h_neg_b : b < 0)
  (h_neg_c : c < 0)
  (h_neg_d : d < 0)
  (h_abs : |x1 - a| + |x2 + b| + |x3 - c| + |x4 + d| = 0) :
  (x1 * x2 / (x3 * x4) > 0) := by
  sorry

end positivity_of_fraction_l196_196909


namespace value_of_x_squared_minus_y_squared_l196_196527

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l196_196527


namespace log_base_5_sqrt_inverse_l196_196820

theorem log_base_5_sqrt_inverse (x : ℝ) (hx : 5 ^ x = 1 / real.sqrt 5) : 
  x = -1 / 2 := 
by
  sorry

end log_base_5_sqrt_inverse_l196_196820


namespace necessary_but_not_sufficient_condition_l196_196708

variable (a b : ℝ)

theorem necessary_but_not_sufficient_condition : (a > b) → ((a > b) ↔ ((a - b) * b^2 > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l196_196708


namespace johns_cloth_cost_per_metre_l196_196149

noncomputable def calculate_cost_per_metre (total_cost : ℝ) (total_metres : ℝ) : ℝ :=
  total_cost / total_metres

def johns_cloth_purchasing_data : Prop :=
  calculate_cost_per_metre 444 9.25 = 48

theorem johns_cloth_cost_per_metre : johns_cloth_purchasing_data :=
  sorry

end johns_cloth_cost_per_metre_l196_196149


namespace compute_M_l196_196794

noncomputable def M : ℕ :=
  let f (n : ℕ) : ℤ := if n % 4 = 0 ∨ n % 4 = 3 then (n:ℤ)^2 else -(n:ℤ)^2
  ∑ i in range 1 51, f i

theorem compute_M : M = 2550 := by
  sorry

end compute_M_l196_196794


namespace log_five_fraction_l196_196838

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196838


namespace line_passing_point_parallel_l196_196001

-- Definitions of point and line
def Point := ℝ × ℝ
def Line := ℝ → ℝ

-- Given a point (1, 2)
def A : Point := (1, 2)

-- Given a line (represented by 2x - 3y + 5 = 0)
def original_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∀ (x y : ℝ), l1 x y = 0 → l2 x y = 0

-- Line to be proved: 2x - 3y + 4 = 0
def target_line (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

-- We need to prove that the line passing through (1, 2) and parallel to the original line is given by the target line
theorem line_passing_point_parallel (A : Point) (original_line target_line : ℝ → ℝ → Prop) 
  (h1 : ∀ x y, original_line x y → 2 * x - 3 * y + 5 = 0)
  (h2 : are_parallel original_line target_line)
  (hA : original_line 1 2):
  target_line 1 2 :=
sorry

end line_passing_point_parallel_l196_196001


namespace intersection_with_y_axis_l196_196641

theorem intersection_with_y_axis :
  ∀ (y : ℝ), (∃ x : ℝ, y = 2 * x + 2 ∧ x = 0) → y = 2 :=
by
  sorry

end intersection_with_y_axis_l196_196641


namespace arc_MTN_constant_l196_196376

open Set

/-- The geometric setup of the problem -/
structure IsoscelesTriangle (A B C : Point) :=
(isosceles : dist A B = dist A C + dist B C)
(altitude : ∃ P, line A B ⊥ line C P ∧ dist C P = 4)

/-- A circle rolling along AB with radius 2, tangent at T, intersects AC at M and BC at N -/
structure RollingCircle (C T M N : Point) :=
(radius_two : ∀ N', N' ∈ circle T 2 → dist T N' = 2)
(tangent_at_T : line T ⟨1,0⟩ tangent circle T 2)
(intersects_AC : ∃ M, M ∈ line A C ∧ M ∈ circle T 2)
(intersects_BC : ∃ N, N ∈ line B C ∧ N ∈ circle T 2)

/-- Prove that the arc MTN (as an angle subtended) is always 120 degrees -/
theorem arc_MTN_constant :
  ∀ (A B C T M N : Point), IsoscelesTriangle A B C → RollingCircle C T M N → arc_measure M T N = 120 :=
by sorry

end arc_MTN_constant_l196_196376


namespace probability_roll_6_final_l196_196733

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l196_196733


namespace find_number_l196_196227

theorem find_number : ∃ x : ℝ, 3 * x - 1 = 2 * x ∧ x = 1 := sorry

end find_number_l196_196227


namespace figure_area_l196_196555

-- Given conditions
def right_angles (α β γ δ: ℕ): Prop :=
  α = 90 ∧ β = 90 ∧ γ = 90 ∧ δ = 90

def segment_lengths (a b c d e f g: ℕ): Prop :=
  a = 15 ∧ b = 8 ∧ c = 7 ∧ d = 3 ∧ e = 4 ∧ f = 2 ∧ g = 5

-- Define the problem
theorem figure_area :
  ∀ (α β γ δ a b c d e f g: ℕ),
    right_angles α β γ δ →
    segment_lengths a b c d e f g →
    a * b - (g * 1 + (d * f)) = 109 :=
by
  sorry

end figure_area_l196_196555


namespace proof_p_or_q_l196_196483

variables {a b c : ℝ^3}

def p : Prop := (a ⬝ c = b ⬝ c) → (a = b)

def q : Prop := (∥a∥ + ∥b∥ = 2 ∧ ∥a∥ < ∥b∥) → (∥b∥^2 > 1)

theorem proof_p_or_q : p ∨ q :=
by
  sorry

end proof_p_or_q_l196_196483


namespace cone_lateral_to_base_area_ratio_l196_196952

theorem cone_lateral_to_base_area_ratio (r : ℝ) (h : r > 0) :
  ∀ (S_lateral S_base : ℝ), 
  let l := 2 * r in
  S_lateral = π * r * l → S_base = π * r ^ 2 → 
  S_lateral / S_base = 2 := 
by
  intros S_lateral S_base l_eq r_eq
  sorry

end cone_lateral_to_base_area_ratio_l196_196952


namespace general_solution_of_diff_eq_l196_196451

theorem general_solution_of_diff_eq (C1 C2 : ℝ) :
  ∀ x : ℝ, 
  let y := λ x, (C1 + C2 * x) * exp (-5 * x) + 2 * x^2 * exp (-5 * x) in 
  y'' x + 10 * y' x + 25 * y x = 4 * exp (-5 * x) :=
sorry

end general_solution_of_diff_eq_l196_196451


namespace digit_in_ten_thousandths_place_l196_196308

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196308


namespace digit_in_ten_thousandths_place_l196_196311

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196311


namespace count_even_numbers_with_adjacent_4_5_l196_196374

theorem count_even_numbers_with_adjacent_4_5 :
  let digits := {1, 2, 3, 4, 5}
  ∃ (xs : List ℕ), xs.length = 4 ∧ (∀ x ∈ xs, x ∈ digits) ∧ (xs.nodup) ∧ 
                   (List.last (4) xs = some 2) ∧ (List.indexOf 4 xs < List.indexOf 5 xs + 1 ∨ List.indexOf 5 xs < List.indexOf 4 xs + 1)
                   ∧ xs.perm ([2, 4, 5, 1] ∨ xs.perm ([2, 1, 4, 5]) ∨ xs.perm ([2, 5, 4, 1]) ∨ xs.perm ([2, 1, 5, 4])) 
                   → xs.perm.permutations.count = 14 :=
by
  sorry

end count_even_numbers_with_adjacent_4_5_l196_196374


namespace solve_equation_1_solve_equation_2_l196_196215

theorem solve_equation_1 :
  ∀ x : ℝ, 2 * x^2 - 4 * x = 0 ↔ (x = 0 ∨ x = 2) :=
by
  intro x
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) :=
by
  intro x
  sorry

end solve_equation_1_solve_equation_2_l196_196215


namespace medical_team_combination_l196_196459

noncomputable def choose : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, n + 1 => 0
| m + 1, 0 => 1
| m + 1, n + 1 => choose m n + choose m (n + 1)

theorem medical_team_combination :
  let total_ways := choose 9 5
  let all_male_ways := choose 6 5
  (total_ways - all_male_ways) = 120 :=
by
  let total_ways := choose 9 5
  let all_male_ways := choose 6 5
  have h1 : total_ways = 126 := by sorry
  have h2 : all_male_ways = 6 := by sorry
  show (total_ways - all_male_ways) = 120 from
    calc
      total_ways - all_male_ways = 126 - 6 := by rw [h1, h2]
                              ... = 120   := by norm_num

end medical_team_combination_l196_196459


namespace infinite_bad_numbers_l196_196479

-- Define types for natural numbers
variables {a b : ℕ}

-- The theorem statement
theorem infinite_bad_numbers (a b : ℕ) : ∃ᶠ (n : ℕ) in at_top, n > 0 ∧ ¬ (n^b + 1 ∣ a^n + 1) :=
sorry

end infinite_bad_numbers_l196_196479


namespace distance_between_homes_l196_196596

variable (MaxwellSpeed : ℝ) (BradSpeed : ℝ) (MaxwellDistance : ℝ)

theorem distance_between_homes (h1 : MaxwellSpeed = 3) 
                               (h2 : BradSpeed = 6)
                               (h3 : MaxwellDistance = 12) : 
  12 + (4 * 6) = 36 :=
by 
  -- Maxwell walks 12 km, at 3 km/h, so time = 12 / 3 = 4 hours
  have time := MaxwellDistance / MaxwellSpeed,
  -- Brad runs for the same time, 4 hours, so distance = 6 km/h * 4 hours = 24 km
  have BradDistance := BradSpeed * time,
  -- Total distance = Maxwell's distance + Brad's distance
  calc
    12 + (4 * 6) = 12 + 24 : by sorry
    ... = 36 : by sorry

end distance_between_homes_l196_196596


namespace problems_per_worksheet_l196_196775

theorem problems_per_worksheet (total_worksheets graded_worksheets remaining_problems : ℕ) (h1 : total_worksheets = 14) (h2 : graded_worksheets = 7) (h3 : remaining_problems = 14) :
  (remaining_problems / (total_worksheets - graded_worksheets)) = 2 :=
by {
  rw [h1, h2, h3],
  norm_num,
}

end problems_per_worksheet_l196_196775


namespace ratio_of_areas_l196_196973

noncomputable def area_of_triangle (A B C : Point) : ℝ := sorry

theorem ratio_of_areas (A B C D E : Point)
  (hD_on_AB : collinear A D B)
  (hE_on_AC : collinear A E C)
  (hDE_parallel_BC : parallel DE BC)
  (AD_eq_1 : distance A D = 1)
  (DB_eq_2 : distance D B = 2) :
  area_of_triangle A D E / area_of_triangle A B C = 1 / 9 :=
sorry

end ratio_of_areas_l196_196973


namespace rearrangements_divisible_by_11_l196_196228

noncomputable def countValidRearrangements : Nat :=
  31680

theorem rearrangements_divisible_by_11 :
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}.toFinset in
  let total_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 in
  total_sum = 45 →
  (∃ (X Y : Nat), 
    X + Y = 45 ∧ 
    (X ≠ Y) ∧ 
    (11 ∣ abs (X - Y)) ∧ 
    (X = 28 ∨ X = 17)) →
  countValidRearrangements = 31680 :=
by intros; sorry

end rearrangements_divisible_by_11_l196_196228


namespace derivative_x_sqrt_x_derivative_x_square_over_sin_tangent_line_ln_x_at_exp_l196_196709

-- (Ⅰ)(1) Proving the derivative of y = x * sqrt(x) is (3/2) * sqrt(x)
theorem derivative_x_sqrt_x (x : ℝ) (hx : x ≥ 0) : 
  deriv (λ x : ℝ, x * real.sqrt x) x = (3 / 2) * real.sqrt x := 
sorry

-- (Ⅰ)(2) Proving the derivative of y = x^2 / sin(x) is (2x sin(x) - x^2 cos(x)) / sin^2(x)
theorem derivative_x_square_over_sin (x : ℝ) (hx : x ≠ 0 ∧ sin x ≠ 0) : 
  deriv (λ x : ℝ, x^2 / sin x) x = (2 * x * sin x - x^2 * cos x) / (sin x)^2 := 
sorry

-- (Ⅱ) Proving the tangent line of f(x) = ln(x) at x = e is y = (1/e) * x
theorem tangent_line_ln_x_at_exp (x : ℝ) (hx : x = real.exp 1) : 
  (λ x : ℝ, real.log x) x = 1 / real.exp 1 * x := 
sorry

end derivative_x_sqrt_x_derivative_x_square_over_sin_tangent_line_ln_x_at_exp_l196_196709


namespace max_tuesday_13_in_year_l196_196543

def days_in_month (month : ℕ) : ℕ :=
  if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else if month = 2 then 28
  else 31

def day_of_week (day: ℕ) : ℕ :=
  day % 7

noncomputable def day_of_13th (month : ℕ) (d_jan_13 : ℕ) : ℕ :=
  let days_from_jan : ℕ := (List.range (month - 1)).map days_in_month.sum
  day_of_week (d_jan_13 + days_from_jan + 12)

def max_tuesday_13 (d_jan_13 : ℕ) : ℕ :=
  (List.range 12).count (λ month, day_of_13th month d_jan_13 = 2)

theorem max_tuesday_13_in_year (d_jan_13 : ℕ) : max_tuesday_13 d_jan_13 ≤ 3 := by
  sorry

end max_tuesday_13_in_year_l196_196543


namespace find_g_neg3_l196_196218

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_neg3 (h : ∀ (x : ℝ), x ≠ 0 → 2 * g (3 / x) - (3 * g x / x) = x^3) :
  g (-3) = -328 / 945 :=
begin
  sorry
end

end find_g_neg3_l196_196218


namespace zeros_before_first_nonzero_digit_l196_196092

theorem zeros_before_first_nonzero_digit 
  (h : ∀ n : ℕ, n = 2^3 * 5^6) : 
  (zeros_before_first_nonzero_digit_decimal \(\frac{1}{n}) = 6) :=
sorry

end zeros_before_first_nonzero_digit_l196_196092


namespace polynomial_inequality_l196_196750

theorem polynomial_inequality (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, (r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3) ∧ 
    (∀ t : ℝ, (t - r1) * (t - r2) * (t - r3) = t^3 + a*t^2 + b*t + c))
  (h2 : ¬ ∃ x : ℝ, (x^2 + x + 2013)^3 + a*(x^2 + x + 2013)^2 + b*(x^2 + x + 2013) + c = 0) :
  t^3 + a*2013^2 + b*2013 + c > 1 / 64 :=
sorry

end polynomial_inequality_l196_196750


namespace coeff_x_50_is_comb_1001_50_l196_196000

def S (x : ℕ) : ℕ := (1 + x)^1000 + ∑ k in range(1, 1000 + 1), (x^k * (1 + x)^(1000 - k))

theorem coeff_x_50_is_comb_1001_50 :
  (coeff (S x) 50) = nat.choose 1001 50 :=
sorry

end coeff_x_50_is_comb_1001_50_l196_196000


namespace least_gumballs_to_four_same_color_l196_196751

theorem least_gumballs_to_four_same_color :
  ∀ (red white blue yellow : ℕ), red = 10 → white = 9 → blue = 8 → yellow = 7 → (∃ n : ℕ, n = 13 ∧
  ∀ (r w b y : ℕ), r + w + b + y = n → (r ≥ 4 ∨ w ≥ 4 ∨ b ≥ 4 ∨ y ≥ 4)) :=
by
  intros red white blue yellow hred hwhite hblue hyellow
  use 13
  split
  · refl
  · intros r w b y hn
    sorry

end least_gumballs_to_four_same_color_l196_196751


namespace value_of_x_squared_minus_y_squared_l196_196524

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l196_196524


namespace curve_transformation_l196_196997

theorem curve_transformation (x y x' y' : ℝ)
  (h1 : x' = 5 * x)
  (h2 : y' = 3 * y)
  (curve : x' ^ 2 + 4 * y' ^ 2 = 1) :
  25 * x ^ 2 + 36 * y ^ 2 = 1 :=
by {
  sorry,
}

end curve_transformation_l196_196997


namespace f_at_1_f_add_f_inv_x_sum_f_and_f_inv_l196_196494

/-- Define the function f -/
def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

/-- Prove that f(1) = 2 -/
theorem f_at_1 (a b : ℝ) : f 1 a b = 2 := by
  unfold f
  have log_1_eq : ∀ b, b * Real.log 1 = 0 := by sorry
  rw [log_1_eq (a / Real.log 2), log_1_eq (b / Real.log 3)]
  norm_num

/-- Prove that f(x) + f(1 / x) = 4 -/
theorem f_add_f_inv_x (x a b : ℝ) (hx : x ≠ 0) : 
  f x a b + f (1 / x) a b = 4 := by
  unfold f
  have log_inv_eq : ∀ (b y : ℝ), b * Real.log (y / 1) = - b * Real.log y := by sorry
  rw [log_inv_eq (a / Real.log 2) x, log_inv_eq (b / Real.log 3) x]
  norm_num

/-- Prove that the sum from 1 to 2013 and their inverses is 8050 -/
theorem sum_f_and_f_inv (a b : ℝ) :
  (∑ k in Finset.range 2013 \ {0}, f (↑(k + 1)) a b) + 
  (∑ k in Finset.range 2013 \ {0}, f (1 / (↑(k + 1))) a b) + f 1 a b = 8050 := by
  unfold f
  have sum_eq : ∀ (k a b : ℝ), f k a b + f (1 / k) a b = 4 := by sorry
  rw [Finset.sum_congr (by simp) (by simp [sum_eq]), f_at_1 a b]
  norm_num


end f_at_1_f_add_f_inv_x_sum_f_and_f_inv_l196_196494


namespace coordinate_scaling_l196_196629

noncomputable def scaling_transformation (x y : ℝ) : (ℝ × ℝ) :=
  let x' := x / 2
  let y' := 3 * y
  (x', y')

theorem coordinate_scaling :
  ∀ x y : ℝ,
  let (x', y') := scaling_transformation x y in
  y = Real.sin x → y' = 3 * Real.sin (2 * x') :=
begin
  intros x y h,
  simp [scaling_transformation] at *,
  sorry
end

end coordinate_scaling_l196_196629


namespace complex_point_quadrant_l196_196319

   theorem complex_point_quadrant (m : ℝ) (h : -1 < m ∧ m < 1) :
     (∃ z : ℂ, z = (1 : ℂ) - (1 : ℂ) * Complex.I + m * (1 : ℂ + Complex.I) ∧
     0 < z.re ∧ z.im < 0) :=
   sorry
   
end complex_point_quadrant_l196_196319


namespace red_candies_count_l196_196663

theorem red_candies_count : 
  let total_candies := 3409 in
  let blue_candies := 3264 in
  let red_candies := total_candies - blue_candies in
  red_candies = 145 :=
by
  sorry

end red_candies_count_l196_196663


namespace digit_in_ten_thousandths_place_l196_196306

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196306


namespace xiaoLi_scored_full_marks_l196_196977

-- Define the statements made by each student
def xiaoLi_statement (xiaoXin_full : Bool) : Bool := ¬ xiaoXin_full
def xiaoDong_statement (xiaoDong_full : Bool) : Bool := xiaoDong_full
def xiaoXin_statement (xiaoLi_statement : Bool) : Bool := xiaoLi_statement

-- Constants for the full mark status of each student
constant xiaoLi_full : Bool
constant xiaoDong_full : Bool
constant xiaoXin_full : Bool

-- Full mark status corresponds with the truthfulness of their statements
def xiaoLi_truth : Bool := xiaoLi_statement xiaoXin_full
def xiaoDong_truth : Bool := xiaoDong_statement xiaoDong_full
def xiaoXin_truth : Bool := xiaoXin_statement (xiaoLi_statement xiaoXin_full)

-- Only one person lied
def one_lied : Bool :=
  ¬(xiaoLi_truth ∧ xiaoDong_truth ∧ xiaoXin_truth ∨
  ¬xiaoLi_truth ∧ ¬xiaoDong_truth ∧ ¬xiaoXin_truth ∨
  xiaoLi_truth ∧ xiaoDong_truth ∧ ¬xiaoXin_truth ∨
  xiaoLi_truth ∧ ¬xiaoDong_truth ∧ xiaoXin_truth ∨
  ¬xiaoLi_truth ∧ xiaoDong_truth ∧ xiaoXin_truth ∨
  ¬xiaoLi_truth ∧ xiaoDong_truth ∧ ¬xiaoXin_truth)

-- The proof statement: prove that Xiao Li scored full marks
theorem xiaoLi_scored_full_marks : one_lied → xiaoLi_full = true :=
by sorry

end xiaoLi_scored_full_marks_l196_196977


namespace solution_problem_l196_196161

theorem solution_problem (a : ℕ → ℚ) : 
  (a 1 = 1) ∧ (a 2 = 1) ∧ (∀ n : ℕ, a (n + 2) = (n * (n + 1) * a (n + 1) + n^2 * a n + 5) / (n + 2) - 2) → 
  (∀ n : ℕ, a n ∈ ℕ ↔ n = 1 ∨ n = 2) :=
by
  intros
  sorry

end solution_problem_l196_196161


namespace Daniela_buys_2_pairs_of_shoes_l196_196421

theorem Daniela_buys_2_pairs_of_shoes
  (original_price_shoes : ℝ := 50)
  (discount_shoes : ℝ := 0.4)
  (original_price_dress : ℝ := 100)
  (discount_dress : ℝ := 0.2)
  (total_spent : ℝ := 140)
  (num_dresses : ℕ := 1) :
  (let discounted_price_shoes := original_price_shoes * (1 - discount_shoes),
      discounted_price_dress := original_price_dress * (1 - discount_dress),
      amount_spent_on_dresses := discounted_price_dress * num_dresses,
      amount_spent_on_shoes := total_spent - amount_spent_on_dresses,
      num_shoes := amount_spent_on_shoes / discounted_price_shoes
  in num_shoes) = 2 := by
  sorry

end Daniela_buys_2_pairs_of_shoes_l196_196421


namespace D_N_O_collinear_l196_196140

-- Definitions and conditions as given in the problem
variables {A B C D U V K E F T M N O : Point}

-- Assuming various conditions and setups as described
-- Triangle ABC inscribed in circle O
-- D is a point on arc BC not containing A

-- Perpendicular relationships
def orthogonal_AB : ⦃U : Point⦄ → Line A B → Line D U
def orthogonal_AC : ⦃V : Point⦄ → Line A C → Line D V

-- Parallel lines
def parallel_UV : Line UV → Line D K
def parallel_EF_UV : Line E F → Line U V

-- Circumcircle of triangle AEF intersecting circle O at T
def circumcircle_AEF : Circle
def intersect_circles : Circle → Circle → Point → Line T K → Line E F → Point M

-- Isogonal conjugate of M wrt triangle ABC
def isogonal_conjugate : Triangle ABC → Point M → Point N

-- Required to prove D, N, O are collinear
theorem D_N_O_collinear 
    (circ_AEF : circumcircle_AEF = Circle)
    (U_perp : orthogonal_AB U (Line.mk A B) (Line.mk D U))
    (V_perp : orthogonal_AC V (Line.mk A C) (Line.mk D V))
    (D_on_arc : D ∈ arc (Commute.mk B C O) ∧ A ∉ arc (Commute.mk B C O))
    (parallel_condition : parallel_UV (Line.mk U V) (Line.mk D K))
    (parallel_EF_UV : parallel_EF_UV (Line.mk E F) (Line.mk U V))
    (T_on_circ : T ∈ circumcircle_AEF ∧ T ∈ Circle)
    (intersect_circles_two : intersect_circles circ_AEF Circle T (Line.mk T K) (Line.mk E F) M)
    (iso_conj_N : isogonal_conjugate (Triangle.mk A B C) M N) : 
    collinear D N O := sorry

end D_N_O_collinear_l196_196140


namespace age_transition_l196_196271

theorem age_transition (initial_ages : List ℕ) : 
  initial_ages = [19, 34, 37, 42, 48] →
  (∃ x, 0 < x ∧ x < 10 ∧ 
  new_ages = List.map (fun age => age + x) initial_ages ∧ 
  new_ages = [25, 40, 43, 48, 54]) →
  x = 6 :=
by
  intros h_initial_ages h_exist_x
  sorry

end age_transition_l196_196271


namespace cos_2theta_l196_196114

theorem cos_2theta (θ : ℝ) (h : exp (θ * complex.I) = (3 + complex.I * real.sqrt 8) / 4) :
  real.cos (2 * θ) = 1 / 8 :=
by sorry

end cos_2theta_l196_196114


namespace find_a6_l196_196892

-- Define the sequence {a_n} with the given conditions
noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := 2
| (n + 1) := real.sqrt ((2 * (a n) ^ 2) + (a (n - 1)) ^ 2)

-- State the problem: Given the conditions, we need to find a_6 = 4
theorem find_a6 :
  ∀ (a : ℕ → ℝ),
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ n ≥ 2, 2 * (a n) ^ 2 = (a (n + 1)) ^ 2 + (a (n - 1)) ^ 2)
  → a 6 = 4 :=
begin
  intro a,
  intros h1 h2 h3,
  sorry
end

end find_a6_l196_196892


namespace grains_in_gray_parts_l196_196434

theorem grains_in_gray_parts (total1 total2 shared : ℕ) (h1 : total1 = 87) (h2 : total2 = 110) (h_shared : shared = 68) :
  (total1 - shared) + (total2 - shared) = 61 :=
by sorry

end grains_in_gray_parts_l196_196434


namespace square_of_sum_of_roots_l196_196314

theorem square_of_sum_of_roots (a b c : ℝ) (h_eq : a = 1 ∧ b = -6 ∧ c = 8) :
  (let sum_roots := -b / a in sum_roots^2 = 36) :=
by
  have h1 : a = 1 := h_eq.1
  have h2 : b = -6 := h_eq.2.1
  have h3 : c = 8 := h_eq.2.2
  let sum_roots := -b / a
  have hs : sum_roots = 6 := by
    rw [h1, h2]
    simp only [neg_neg, one_div_one, neg_neg_eq_pos]
  rw hs
  norm_num
  exact eq.refl 36

end square_of_sum_of_roots_l196_196314


namespace log_base5_of_inverse_sqrt5_l196_196830

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196830


namespace spot_reachable_area_l196_196620

-- Define the conditions
def doghouse_side_length : ℝ := 2
def rope_length : ℝ := 3

-- Define the angles
def sector_angle_main : ℝ := 240
def sector_angle_additional : ℝ := 60

-- Define the expected answer
def expected_area : ℝ := (22 / 3) * Real.pi

-- Lean statement to prove the area Spot can reach
theorem spot_reachable_area :
  let area_main_sector := Real.pi * rope_length^2 * (sector_angle_main / 360)
  let area_additional_sectors := 2 * (Real.pi * doghouse_side_length^2 * (sector_angle_additional / 360))
  area_main_sector + area_additional_sectors = expected_area := by
  let area_main_sector := Real.pi * rope_length^2 * (sector_angle_main / 360)
  let area_additional_sectors := 2 * (Real.pi * doghouse_side_length^2 * (sector_angle_additional / 360))
  sorry

end spot_reachable_area_l196_196620


namespace sqrt_of_sixteen_l196_196394

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l196_196394


namespace final_answer_for_m_l196_196061

noncomputable def proof_condition_1 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

noncomputable def proof_condition_2 (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

noncomputable def proof_condition_perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem final_answer_for_m :
  (∀ (x y m : ℝ), proof_condition_1 x y m) →
  (∀ (x y : ℝ), proof_condition_2 x y) →
  (∀ (x1 y1 x2 y2 : ℝ), proof_condition_perpendicular x1 y1 x2 y2) →
  m = 12 / 5 :=
sorry

end final_answer_for_m_l196_196061


namespace absolute_value_half_l196_196519

theorem absolute_value_half (a : ℝ) (h : |a| = 1/2) : a = 1/2 ∨ a = -1/2 :=
sorry

end absolute_value_half_l196_196519


namespace Chad_saves_40_percent_of_his_earnings_l196_196792

theorem Chad_saves_40_percent_of_his_earnings :
  let earnings_mow := 600
  let earnings_birthday := 250
  let earnings_games := 150
  let earnings_oddjobs := 150
  let amount_saved := 460
  (amount_saved / (earnings_mow + earnings_birthday + earnings_games + earnings_oddjobs) * 100) = 40 :=
by
  sorry

end Chad_saves_40_percent_of_his_earnings_l196_196792


namespace valid_arrangements_correct_l196_196882

def selections (α β : Type) := (set α) × (set β)

noncomputable def valid_arrangements : ℕ :=
  let letters := {A, B, C, D, E}
  let numbers := {1, 3, 5, 7, 9}
  let total_rows := (finset.card (finset.choose 2 letters)) *
                    (finset.card (finset.choose 2 numbers)) *
                    (finset.factorial 4)
  let invalid_rows := (finset.card (finset.choose 1 (letters \ {A}))) *
                      (finset.card (finset.choose 1 (numbers \ {9}))) *
                      (finset.factorial 4)
  total_rows - invalid_rows

theorem valid_arrangements_correct : valid_arrangements = 2016 := 
  by
  sorry

end valid_arrangements_correct_l196_196882


namespace expressions_equal_l196_196807

variable (a b c : ℝ)

theorem expressions_equal (h : a + 2 * b + 2 * c = 0) : a + 2 * b * c = (a + 2 * b) * (a + 2 * c) := 
by 
  sorry

end expressions_equal_l196_196807


namespace taco_variants_count_l196_196671

theorem taco_variants_count :
  let toppings := 8
  let meat_variants := 3
  let shell_variants := 2
  2 ^ toppings * meat_variants * shell_variants = 1536 := by
sorry

end taco_variants_count_l196_196671


namespace value_of_a_plus_b_l196_196020

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l196_196020


namespace triangle_obtuse_inequality_l196_196131

theorem triangle_obtuse_inequality (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) (h₃ : a^2 + b^2 < c^2) : 
  sqrt 5 < c ∧ c < 3 :=
by 
  sorry

end triangle_obtuse_inequality_l196_196131


namespace count_zeros_decimal_representation_l196_196102

theorem count_zeros_decimal_representation (n m : ℕ) (h : n = 3) (h₁ : m = 6) : 
  ∃ k : ℕ, k = 5 ∧ 
    let d := (1 : ℚ) / (2^n * 5^m) in 
    let s := d.repr.mantissa in -- Assuming a function repr.mantissa to represent decimal digits
    s.indexOf '1' - s.indexOf '.' - 1 = k := sorry

end count_zeros_decimal_representation_l196_196102


namespace divide_angle_into_parts_l196_196897

-- Definitions based on the conditions
def given_angle : ℝ := 19

/-- 
Theorem: An angle of 19 degrees can be divided into 19 equal parts using a compass and a ruler,
and each part will measure 1 degree.
-/
theorem divide_angle_into_parts (angle : ℝ) (n : ℕ) (h1 : angle = given_angle) (h2 : n = 19) : angle / n = 1 :=
by
  -- Proof to be filled out
  sorry

end divide_angle_into_parts_l196_196897


namespace more_than_half_square_perimeter_inside_triangle_l196_196719

variables {R : Type*} [LinearOrderedField R]

-- Definitions based on the conditions
def circle_inscribed_in_triangle (C : Circle R) (T : Triangle R) : Prop := sorry
def square_circumscribed_around_circle (S : Square R) (C : Circle R) : Prop := sorry

-- Given conditions
variables (C : Circle R) (T : Triangle R) (S : Square R)
variables (hcirc : circle_inscribed_in_triangle C T)
variables (hsquare : square_circumscribed_around_circle S C)

-- Theorem to prove more than half the perimeter of the square lies inside the triangle
theorem more_than_half_square_perimeter_inside_triangle : 
  ∃ p > 0, p > (1 / 2) * S.perimeter ∧ p ≤ S.perimeter ∧ inside_triangle (S.perimeter_inside T) :=
sorry

end more_than_half_square_perimeter_inside_triangle_l196_196719


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196287

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196287


namespace solid_with_square_views_is_cube_l196_196651

-- Define the conditions and the solid type
def is_square_face (view : Type) : Prop := 
  -- Definition to characterize a square view. This is general,
  -- as the detailed characterization of a 'square' in Lean would depend
  -- on more advanced geometry modules, assuming a simple predicate here.
  sorry

structure Solid := (front_view : Type) (top_view : Type) (left_view : Type)

-- Conditions indicating that all views are squares
def all_views_square (S : Solid) : Prop :=
  is_square_face S.front_view ∧ is_square_face S.top_view ∧ is_square_face S.left_view

-- The theorem we are aiming to prove
theorem solid_with_square_views_is_cube (S : Solid) (h : all_views_square S) : S = {front_view := ℝ, top_view := ℝ, left_view := ℝ} := sorry

end solid_with_square_views_is_cube_l196_196651


namespace find_the_value_l196_196911

open Real

noncomputable def m (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

noncomputable def n (θ : ℝ) : ℝ × ℝ :=
  (sqrt 2 - sin θ, cos θ)

def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_the_value (θ : ℝ) (hθ : θ ∈ Ioo π (2 * π)) 
  (h : magnitude (m θ + n θ) = (8 * sqrt 2) / 5) : 
  5 * cos (θ / 2 + π / 8) + 5 = 1 :=
sorry

end find_the_value_l196_196911


namespace log_base_5_of_inv_sqrt_5_l196_196823

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196823


namespace range_of_m_l196_196959

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (x + m) / (x - 2) + 2 * m / (2 - x) = 3) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l196_196959


namespace sum_of_ages_l196_196155

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l196_196155


namespace find_base_a_l196_196922

theorem find_base_a :
  (∃ a > 1, log a π - log a 2 = 1) ∨ (∃ a, 0 < a ∧ a < 1 ∧ log a 2 - log a π = 1) :=
sorry

end find_base_a_l196_196922


namespace angle_PSQ_eq_160_l196_196531

-- This is a noncomputable theory statement because it involves degree measures
noncomputable theory

-- Define the variables and the context of the triangle and points
variables {P Q R S : Type} [EuclideanGeometry P] [EuclideanGeometry Q] [EuclideanGeometry R] [EuclideanGeometry S]

-- Define the given conditions
def triangle_PQR : Triangle P Q R := sorry   -- The definition of a triangle with vertices P, Q, R
def point_S_on_PR : S ∈ segment P R := sorry -- S is on the segment PR
def QS_eq_SR : distance Q S = distance S R := sorry -- QS = SR
def angle_QSR : angle Q S R = 80° := sorry -- ∠QSR = 80°

-- Define the theorem to be proved
theorem angle_PSQ_eq_160 :
  angle P S Q = 160° :=
sorry

end angle_PSQ_eq_160_l196_196531


namespace log_base5_of_inverse_sqrt5_l196_196827

theorem log_base5_of_inverse_sqrt5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
begin
  sorry
end

end log_base5_of_inverse_sqrt5_l196_196827


namespace total_bill_l196_196380

def number_of_adults := 2
def number_of_children := 5
def meal_cost := 3

theorem total_bill : number_of_adults * meal_cost + number_of_children * meal_cost = 21 :=
by
  sorry

end total_bill_l196_196380


namespace probability_six_on_final_roll_l196_196743

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l196_196743


namespace primes_sq_l196_196040

theorem primes_sq (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (h_div : p * q * r ∣ (p + q + r)) :
  ∃ n : ℕ, (p - 1) * (q - 1) * (r - 1) + 1 = n^2 :=
by sorry

end primes_sq_l196_196040


namespace find_f_neg_a_l196_196498

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end find_f_neg_a_l196_196498


namespace monotonicity_of_g_l196_196057

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (log a x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (log a (3 - 2 * x - x^2))

theorem monotonicity_of_g (a : ℝ) (ha : 1 < a) :
  ∀ x, -3 < x ∧ x < -1 →
    3 - 2 * x - x^2 > 0 → ( g a ) x < ( g a ) ( x + ϵ ) for some small positive ϵ :=
begin
  sorry
end

end monotonicity_of_g_l196_196057


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196304

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196304


namespace inverse_of_exponential_minus_three_l196_196239

theorem inverse_of_exponential_minus_three :
  (∀ x, f (f_inv x) = x) ∧ (∀ x, f_inv (f x) = x) :=
sorry

def f (x : ℝ) : ℝ :=
2^x - 3

def f_inv (x : ℝ) : ℝ :=
Real.log (x + 3) / Real.log 2

end inverse_of_exponential_minus_three_l196_196239


namespace find_price_per_backpack_l196_196937

noncomputable def original_price_of_each_backpack
  (total_backpacks : ℕ)
  (monogram_cost : ℕ)
  (total_cost : ℕ)
  (backpacks_cost_before_discount : ℕ) : ℕ :=
total_cost - (total_backpacks * monogram_cost)

theorem find_price_per_backpack
  (total_backpacks : ℕ := 5)
  (monogram_cost : ℕ := 12)
  (total_cost : ℕ := 140)
  (expected_price_per_backpack : ℕ := 16) :
  original_price_of_each_backpack total_backpacks monogram_cost total_cost / total_backpacks = expected_price_per_backpack :=
by
  sorry

end find_price_per_backpack_l196_196937


namespace base_7_representation_of_85_has_three_non_consecutive_digits_l196_196136

theorem base_7_representation_of_85_has_three_non_consecutive_digits:
  ∃ (n : ℕ), n = 85 ∧ (n.base_repr 7 = "151" ∧ ¬ ∃ a b c : ℕ, a + 1 = b ∧ b + 1 = c ∧ "151" = repr (a * 49 + b * 7 + c * 1)) :=
by sorry

end base_7_representation_of_85_has_three_non_consecutive_digits_l196_196136


namespace total_coins_l196_196184

theorem total_coins (piles_quarters piles_dimes piles_nickels piles_pennies : ℕ)
  (coins_per_pile_quarters coins_per_pile_dimes coins_per_pile_nickels coins_per_pile_pennies : ℕ) :
  piles_quarters = 5 →
  piles_dimes = 5 →
  piles_nickels = 3 →
  piles_pennies = 4 →
  coins_per_pile_quarters = 3 →
  coins_per_pile_dimes = 3 →
  coins_per_pile_nickels = 4 →
  coins_per_pile_pennies = 5 →
  piles_quarters * coins_per_pile_quarters +
  piles_dimes * coins_per_pile_dimes +
  piles_nickels * coins_per_pile_nickels +
  piles_pennies * coins_per_pile_pennies = 62 :=
begin
  sorry
end

end total_coins_l196_196184


namespace repayment_installments_l196_196148

theorem repayment_installments :
  ∃ n : ℕ, 
    let PV := 993,
        PMT := 399.30,
        r := 0.10 in
    n = Real.ceil (Real.log (1 - (PV * r) / PMT) / Real.log (1 + r)) := by
  use 3
  sorry

end repayment_installments_l196_196148


namespace area_of_S_l196_196079

-- Define the conditions
def in_circle (z : ℂ) (center : ℂ) (r : ℝ) : Prop := 
  abs (z - center) = r

def on_unit_circle (z : ℂ) : Prop :=
  abs z = 1

-- Given set and conditions
def S : set ℂ := 
  { z | ∃ z₁ : ℂ, on_unit_circle z₁ ∧ abs (z - (7 + 8 * complex.I)) = abs (z₁^4 + 1 - 2 * z₁^2) }

-- Area calculation
theorem area_of_S :
  let A := π * 4^2 in
  ∀ z : ℂ, z ∈ S ↔ in_circle z (7 + 8 * complex.I) 4 :=
by 
  intros,
  unfold A,
  sorry

end area_of_S_l196_196079


namespace smallest_n_exceeding_10_pow_80_l196_196343

def f : ℕ+ → ℕ
| 1 := 0
| (n + 1) := 2 ^ (f n)

theorem smallest_n_exceeding_10_pow_80 :
  ∃ n : ℕ+, n = 7 ∧ (f n > 2^240) ∧ ∀ m : ℕ+, m < n → f m ≤ 2^240 :=
by {
  use 7,
  split,
  { refl, },
  split,
  { sorry, },
  { sorry, }
}

end smallest_n_exceeding_10_pow_80_l196_196343


namespace N_is_composite_l196_196809

def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

theorem N_is_composite : ¬ (Nat.Prime N) :=
by
  have h_mod : N % 2027 = 0 := 
    sorry
  intro h_prime
  have h_div : 2027 ∣ N := by
    rw [Nat.dvd_iff_mod_eq_zero, h_mod]
  exact Nat.Prime.not_dvd_one h_prime h_div

end N_is_composite_l196_196809


namespace rectangle_area_l196_196637

/-- A figure is formed by a triangle and a rectangle, using 60 equal sticks.
Each side of the triangle uses 6 sticks, and each stick measures 5 cm in length.
Prove that the area of the rectangle is 2250 cm². -/
theorem rectangle_area (sticks_total : ℕ) (sticks_per_side_triangle : ℕ) (stick_length_cm : ℕ)
    (sticks_used_triangle : ℕ) (sticks_left_rectangle : ℕ) (sticks_per_width_rectangle : ℕ)
    (width_sticks_rectangle : ℕ) (length_sticks_rectangle : ℕ) (width_cm : ℕ) (length_cm : ℕ)
    (area_rectangle : ℕ) 
    (h_sticks_total : sticks_total = 60)
    (h_sticks_per_side_triangle : sticks_per_side_triangle = 6)
    (h_stick_length_cm : stick_length_cm = 5)
    (h_sticks_used_triangle  : sticks_used_triangle = sticks_per_side_triangle * 3)
    (h_sticks_left_rectangle : sticks_left_rectangle = sticks_total - sticks_used_triangle)
    (h_sticks_per_width_rectangle : sticks_per_width_rectangle = 6 * 2) 
    (h_width_sticks_rectangle : width_sticks_rectangle = 6)
    (h_length_sticks_rectangle : length_sticks_rectangle = (sticks_left_rectangle - sticks_per_width_rectangle) / 2)
    (h_width_cm : width_cm = width_sticks_rectangle * stick_length_cm)
    (h_length_cm : length_cm = length_sticks_rectangle * stick_length_cm)
    (h_area_rectangle : area_rectangle = width_cm * length_cm) :
    area_rectangle = 2250 := 
by sorry

end rectangle_area_l196_196637


namespace solve_system_of_equations_l196_196619

theorem solve_system_of_equations 
  (p q r s t : ℝ)
  (h1 : p^2 + q^2 + r^2 = 6)
  (h2 : pq - s^2 - t^2 = 3) : 
  (p, q, r, s, t) = (sqrt 3, sqrt 3, 0, 0, 0) ∨ 
  (p, q, r, s, t) = (-sqrt 3, -sqrt 3, 0, 0, 0) :=
by
  sorry

end solve_system_of_equations_l196_196619


namespace rhombus_unique_property_l196_196760

theorem rhombus_unique_property (P : Type) [EuclideanGeometry P] :
  (∃ R : Rhombus P, ∀ S : Rectangle P, (∀ r : Rhombus P, (all_four_sides_equal r) → (¬ all_four_sides_equal S))) :=
begin
  sorry
end

end rhombus_unique_property_l196_196760


namespace integral_solutions_l196_196427

theorem integral_solutions (a b c : ℤ) (h : a^2 + b^2 + c^2 = a^2 * b^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integral_solutions_l196_196427


namespace quadratic_real_solutions_l196_196847

namespace Proof

theorem quadratic_real_solutions (n : ℕ) 
    (a : Fin (n + 2) → ℝ) : 
    (∀ x : ℝ, (a n + 1) * x^2 - 2 * x * sqrt (Σ i, (a i)^2) + (Σ i in Finset.range (n + 1), a i) = 0 → 
    ∃ x : ℝ, (a n + 1) * x^2 - 2 * x * sqrt (Σ i, (a i)^2) + (Σ i in Finset.range (n + 1), a i) = 0) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
begin
    sorry
end

end Proof

end quadratic_real_solutions_l196_196847


namespace weight_of_a_l196_196255

-- Define conditions
def weight_of_b : ℕ := 750 -- weight of one liter of ghee packet of brand 'b' in grams
def ratio_a_to_b : ℕ × ℕ := (3, 2)
def total_volume_liters : ℕ := 4 -- total volume of the mixture in liters
def total_weight_grams : ℕ := 3360 -- total weight of the mixture in grams

-- Target proof statement
theorem weight_of_a (W_a : ℕ) 
  (h_ratio : (ratio_a_to_b.1 + ratio_a_to_b.2) = 5)
  (h_mix_vol_a : (ratio_a_to_b.1 * total_volume_liters) = 12)
  (h_mix_vol_b : (ratio_a_to_b.2 * total_volume_liters) = 8)
  (h_weight_eq : (ratio_a_to_b.1 * W_a * total_volume_liters + ratio_a_to_b.2 * weight_of_b * total_volume_liters) = total_weight_grams * 5) : 
  W_a = 900 :=
by {
  sorry
}

end weight_of_a_l196_196255


namespace probability_roll_6_final_l196_196729

variable {Ω : Type*} [ProbabilitySpace Ω]

/-- Define the outcomes of rolls -/
noncomputable def diceRollPMF : PMF (Fin 6) :=
  PMF.ofFinset (finset.univ) (by simp; exact λ i, 1/6)

/-- Define the scenario and prove the required probability -/
theorem probability_roll_6_final {sum : ℕ} (h_sum : sum ≥ 2019) :
  (PMF.cond diceRollPMF (λ x, x = 5)).prob > 5/6 :=
sorry

end probability_roll_6_final_l196_196729


namespace probability_three_heads_l196_196710

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability (n : ℕ) (k : ℕ) : ℚ :=
  (binom n k) / (2 ^ n)

theorem probability_three_heads : probability 12 3 = 55 / 1024 := 
by
  sorry

end probability_three_heads_l196_196710


namespace digit_in_ten_thousandths_place_of_fraction_l196_196283

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196283


namespace coefficient_x_105_l196_196851

noncomputable def P (x : ℕ) : ℤ[X] := ∏ k in Finset.range 1 16, (X^k - k)

-- The theorem that states the resultant coefficient of x^105 is 134 given the polynomial P(x)
theorem coefficient_x_105 : (P x).coeff 105 = 134 :=
sorry

end coefficient_x_105_l196_196851


namespace calculate_expression_l196_196387

theorem calculate_expression :
  0.027 ^ (1 / 3) - (-1 / 7) ^ (-2) + 256 ^ (3 / 4) - 3 ^ (-1) + (Real.sqrt 2 - 1) ^ 0 - (Real.log 2 / Real.log 6 + Real.log 3 / Real.log 6) = 449 / 30 :=
by
  sorry

end calculate_expression_l196_196387


namespace sum_of_fractions_l196_196389

-- Definitions (Conditions)
def frac1 : ℚ := 5 / 13
def frac2 : ℚ := 9 / 11

-- Theorem (Equivalent Proof Problem)
theorem sum_of_fractions : frac1 + frac2 = 172 / 143 := 
by
  -- Proof skipped
  sorry

end sum_of_fractions_l196_196389


namespace compound_interest_period_l196_196445

theorem compound_interest_period
  (P : ℝ) (r : ℝ) (I : ℝ) (n : ℕ) 
  (principal : P = 6000)
  (annual_rate : r = 0.15)
  (interest : I = 2331.75)
  (compounded_annually : n = 1)
  (A : ℝ := P + I) :
  A = 8331.75 ∧ (∃ t : ℝ, (P * (1 + r / n)^(n * t) = A) ∧ (t ≈ 2)) :=
by
  sorry

end compound_interest_period_l196_196445


namespace sum_factors_60_l196_196318

theorem sum_factors_60 : ∑ i in (finset.filter (| i | ∃ (a b c : ℕ), (2^a * 3^b * 5^c = i ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) (finset.range 61)), i = 168 :=
by
  sorry

end sum_factors_60_l196_196318


namespace arc_length_correct_l196_196538

-- Define the given conditions
def radius : ℝ := 3
def centralAngle : ℝ := π / 7

-- Define the arc length formula
def arc_length (r : ℝ) (angle : ℝ) : ℝ := r * angle

-- The theorem statement
theorem arc_length_correct :
  arc_length radius centralAngle = 3 * π / 7 :=
by
  sorry

end arc_length_correct_l196_196538


namespace min_max_angle_numbers_l196_196601

theorem min_max_angle_numbers (n : ℕ) (h : n > 2) :
  ∃ min max, min = 3 ∧ max = n :=
by
  sorry

end min_max_angle_numbers_l196_196601


namespace arithmetic_sequence_common_difference_l196_196919

theorem arithmetic_sequence_common_difference 
  (a_n : ℕ → ℕ)
  (h : ∀ n : ℕ, a_n = 2 * n) : 
  ∀ n : ℕ, a_n - a_n.pred = 2 := 
sorry

end arithmetic_sequence_common_difference_l196_196919


namespace area_triangle_ABC_l196_196060

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_triangle_ABC :
  area_of_triangle (2, 4) (-1, 1) (1, -1) = 6 :=
by
  sorry

end area_triangle_ABC_l196_196060


namespace polynomial_identity_l196_196608

theorem polynomial_identity 
  (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  a^2 * ((x - b) * (x - c) / ((a - b) * (a - c))) +
  b^2 * ((x - c) * (x - a) / ((b - c) * (b - a))) +
  c^2 * ((x - a) * (x - b) / ((c - a) * (c - b))) = x^2 :=
by
  sorry

end polynomial_identity_l196_196608


namespace sqrt_sixteen_is_four_l196_196399

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l196_196399


namespace orthocenter_on_circumcircle_l196_196573

noncomputable def circumcircle (A B C : Point) : Circle := sorry

noncomputable def orthoCenter (A B C : Point) : Point := sorry

variable (A B C : Point)

-- Given conditions
variable (circumcircleABC : Circle) (h1 : circumcircleABC = circumcircle A B C)
variable (B1 : Point) (h2 : PointOnLineExtended B B1 (Segment A B))
variable (h3 : dist A B1 = dist A C)
variable (W : Point) (h4 : IsAngleBisector W (Angle A B C) A B circumcircleABC)

-- Prove
theorem orthocenter_on_circumcircle : PointOnCircle (orthoCenter A W B1) circumcircleABC := by
  sorry

end orthocenter_on_circumcircle_l196_196573


namespace second_derivative_y_wrt_x_l196_196698

variable (t : ℝ)

def x (t : ℝ) : ℝ := Mathlib.sinh t
def y (t : ℝ) : ℝ := Mathlib.tanh t ^ 2

theorem second_derivative_y_wrt_x (t : ℝ) :
  (deriv (deriv (y t / x t)) / deriv (x t)) =
  (2 - 6 * (Mathlib.sinh t) ^ 2) / (Mathlib.cosh t) ^ 6 :=
sorry

end second_derivative_y_wrt_x_l196_196698


namespace number_of_monochromatic_triangles_l196_196540

-- Given conditions in the problem:
variables (Members : Type) [Fintype Members] [DecidableEq Members] (friend enemy : Members → Members → Prop)
variables [Symmetric friend] [Symmetric enemy]
variables (n : Nat) (H1 : Fintype.card Members = 30)
variables (H2 : ∀ x : Members, Fintype.card ({y // friend x y}) = 6)
variables (H3 : ∀ x y : Members, x ≠ y → (friend x y ∨ enemy x y))

-- Definition of the main statement to be proved:
theorem number_of_monochromatic_triangles : 
  ∑ x y z : Members, if friend x y ∧ friend y z ∧ friend z x ∨ enemy x y ∧ enemy y z ∧ enemy z x then 1 else 0 = 1990 :=
sorry

end number_of_monochromatic_triangles_l196_196540


namespace roots_expression_value_l196_196907

theorem roots_expression_value {m n : ℝ} (h₁ : m^2 - 3 * m - 2 = 0) (h₂ : n^2 - 3 * n - 2 = 0) : 
  (7 * m^2 - 21 * m - 3) * (3 * n^2 - 9 * n + 5) = 121 := 
by 
  sorry

end roots_expression_value_l196_196907


namespace sawyer_total_octopus_legs_l196_196207

-- Formalization of the problem conditions
def num_octopuses : Nat := 5
def legs_per_octopus : Nat := 8

-- Formalization of the question and answer
def total_legs : Nat := num_octopuses * legs_per_octopus

-- The proof statement
theorem sawyer_total_octopus_legs : total_legs = 40 :=
by
  sorry

end sawyer_total_octopus_legs_l196_196207


namespace apex_angle_l196_196980

variables (a : ℝ) (P A B C : ℝ → ℝ → ℝ → Prop)

-- Given conditions of the problem
def regular_pyramid (P A B C : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a : ℝ, (side_base ABC = a) ∧ (lateral_edge PA = 2 * a) ∧ 
  (on_cone A P B C) 

-- To prove: angle at apex is 2 * arcsin (3 / (2 * sqrt(5))) given the above conditions
theorem apex_angle (a : ℝ) 
  (h₁ : side_base ABC = a) 
  (h₂ : lateral_edge PA = 2 * a) 
  (h₃ : on_cone A P B C) : 
  angle_apex (cross_section (cone A P B C)) = 2 * arcsin (3 / (2 * sqrt 5)) :=
sorry

end apex_angle_l196_196980


namespace maximum_height_l196_196353

def height (t : ℝ) : ℝ :=
  -20 * t^2 + 50 * t + 10

theorem maximum_height :
  ∃ t : ℝ, height t = 41.25 :=
by
  sorry

end maximum_height_l196_196353


namespace find_divisor_l196_196958

def remainder : Nat := 1
def quotient : Nat := 54
def dividend : Nat := 217

theorem find_divisor : ∃ divisor : Nat, (dividend = divisor * quotient + remainder) ∧ divisor = 4 :=
by
  sorry

end find_divisor_l196_196958


namespace permutations_count_l196_196982

def valid_permutations (l : List ℕ) : Prop :=
  l ~=[1, 2, 3, 4] ∧ -- l is a permutation of [1, 2, 3, 4]
  (∀ a b c : ℕ, List.Sublist 3 [a, b, c] l → (a < b < c) → False) ∧ -- No three consecutive terms are increasing
  (∀ a b c : ℕ, List.Sublist 3 [a, b, c] l → (a > b > c) → False) ∧ -- No three consecutive terms are decreasing
  l.head! < l.getLast' -- The first term is less than the last term

theorem permutations_count : ∃ l : List ℕ, valid_permutations l ∧ (l = [1, 2, 3, 4] ∨ 0: Finset.Range = 1) :=
begin
sorry
end

end permutations_count_l196_196982


namespace perpendicular_lines_l196_196009

theorem perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔
  let L1 := λ x y : ℝ, a * x + (1 - a) * y = 3 in
  let L2 := λ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 in
  let slope1_exists := (a ≠ 1) in
  let slope2_exists := (2 * a + 3 ≠ 0) in
  if ¬ slope1_exists ∧ slope2_exists then
    true -- when slope of L1 is undefined and slope of L2 is zero
  else if slope1_exists ∧ ¬ slope2_exists then
    a = -3 / 2 -- when slope of L2 is undefined (special case check outside the main proof)
  else if slope1_exists ∧ slope2_exists then
    (a / (1 - a)) * ((1 - a) / (2 * a + 3)) = -1 -- product of slopes is -1
  else
    false := sorry

end perpendicular_lines_l196_196009


namespace problem_solution_l196_196232

theorem problem_solution 
  (c d : ℝ)
  (hcd : c ≥ d)
  (hsol : ∀ x : ℝ, x^2 - 6 * x + 11 = 23 ↔ x = c ∨ x = d) :
  3 * c + 2 * d = 15 + sqrt 21 :=
by sorry

end problem_solution_l196_196232


namespace min_value_expr_l196_196230

noncomputable def minimum_value_of_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : sqrt (1 - (b^2 / a^2)) = 1 / 2) : ℝ :=
  (b^2 + 1) / (3 * a)

theorem min_value_expr (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : sqrt (1 - (b^2 / a^2)) = 1 / 2) : 
  minimum_value_of_expression a b h1 h2 h3 = sqrt 3 / 3 :=
  sorry

end min_value_expr_l196_196230


namespace find_constant_a_l196_196862

theorem find_constant_a (a : ℝ) :
    (1 - 4 * a + 6) = -6 → a = 3 := 
by
  intro h
  calc
    1 - 4 * a + 6 = -6 := h
    1 + 6 - 6 = 4 * a := sorry
    0 = 4 * a := sorry
    a = 3 := sorry

end find_constant_a_l196_196862


namespace extreme_point_properties_l196_196624

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_point_properties (a x₁ x₂ : ℝ) (h₁ : 0 < a) (h₂ : a < 1 / 4) 
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : x₁ < x₂) :
  f x₁ a < 0 ∧ f x₂ a > (-1 / 2) := 
sorry

end extreme_point_properties_l196_196624


namespace smallest_n_for_f_n_eq_4_l196_196578

def f (n : ℕ) : ℕ :=
  finset.card {p : ℕ × ℕ | let a := p.1; let b := p.2 in a ≠ b ∧ a^2 + b^2 = n}
  
theorem smallest_n_for_f_n_eq_4 : ∃ n : ℕ, n = 65 ∧ f(n) = 4 :=
by
  existsi 65
  split
  rfl
  -- f(65) = 4 needs to be proven
  sorry

end smallest_n_for_f_n_eq_4_l196_196578


namespace sqrt_of_sixteen_l196_196393

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l196_196393


namespace smallest_number_l196_196687

def smallest_of_five_numbers (a b c d e : ℝ) : ℝ :=
if a < b ∧ a < c ∧ a < d ∧ a < e then a
else if b < a ∧ b < c ∧ b < d ∧ b < e then b
else if c < a ∧ c < b ∧ c < d ∧ c < e then c
else if d < a ∧ d < b ∧ d < c ∧ d < e then d
else e

theorem smallest_number
  (a b c d e : ℝ)
  (ha : a = 0.803)
  (hb : b = 0.8003)
  (hc : c = 0.8)
  (hd : d = 0.8039)
  (he : e = 0.809) : 
  smallest_of_five_numbers a b c d e = 0.8 :=
by {
  rw [ha, hb, hc, hd, he],
  -- The assertion is that c is 0.8 and it's indeed smallest, but proof is omitted for now.
  have h : ∀ x ∈ [a, b, d, e], x > c, sorry,
  rw smallest_of_five_numbers,
  simp [hc, ha, hb, hd, he, h],
  }

end smallest_number_l196_196687


namespace relationship_among_a_b_c_l196_196015

theorem relationship_among_a_b_c :
  let a := 2^0.3
  let b := 0.3 ^ 2
  let c := Real.log 2 / Real.log (Real.sqrt 2)
  in b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l196_196015


namespace find_function_proof_l196_196843

noncomputable theory

open Nat

theorem find_function_proof (f : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → f (n!) = (f n)!) ∧
  (∀ m n : ℕ, m > 0 ∧ n > 0 ∧ m ≠ n → (m - n) ∣ (f m - f n)) →
  (∀ n : ℕ, f n = 1 ∨ f n = 2 ∨ f n = n) :=
by
  sorry

end find_function_proof_l196_196843


namespace find_y_l196_196887

variables {a b c x : ℝ}
variables {p q r y : ℝ}

theorem find_y 
  (h₀ : (log a) / p = (log b) / q)
  (h₁ : (log b) / q = (log c) / r)
  (h₂ : (log c) / r = log x)
  (hx : x ≠ 1)
  (h₃ : a^2 * c / b^3 = x^y) :
  y = 2 * p + r - 3 * q :=
by {
  sorry
}

end find_y_l196_196887


namespace number_of_diagonals_in_prism_l196_196762

theorem number_of_diagonals_in_prism (w : ℝ) : 
  let width := w
  let height := 2 * w
  let depth := 3 * w
  in total_diagonals :=
    face_diagonals + space_diagonals = 16 :=
begin
  -- Let width, height, and depth be defined as above
  let face_diagonals := 6 * 2, -- 2 diagonals per face * 6 faces
  let space_diagonals := 4, -- diagonally opposite vertices
  let total_diagonals := face_diagonals + space_diagonals,
  show total_diagonals = 16,
  sorry
end

end number_of_diagonals_in_prism_l196_196762


namespace max_blocks_fit_l196_196313

-- Define the dimensions of the boxes
def block_dim := (3, 1, 1 : ℕ × ℕ × ℕ)
def box_dim := (3, 4, 3 : ℕ × ℕ × ℕ)

-- Calculate the volume of the block and the box
def block_volume := (block_dim.1 * block_dim.2 * block_dim.3 : ℕ)
def box_volume := (box_dim.1 * box_dim.2 * box_dim.3 : ℕ)

-- Statement of the maximum number of blocks that can fit inside the box
theorem max_blocks_fit (block_dim : ℕ × ℕ × ℕ) (box_dim : ℕ × ℕ × ℕ) :
  let block_volume := block_dim.1 * block_dim.2 * block_dim.3,
      box_volume := box_dim.1 * box_dim.2 * box_dim.3 in
  block_dim = (3, 1, 1) → box_dim = (3, 4, 3) → box_volume / block_volume = 12 :=
by
  intros h1 h2
  rw [h1, h2]
  dsimp [block_volume, box_volume]
  norm_num
  sorry

end max_blocks_fit_l196_196313


namespace triangle_obtuse_count_l196_196248

theorem triangle_obtuse_count :
  ∃ k_set : set ℕ, (∀ k ∈ k_set, 4 < k ∧ k < 26 ∧ (∃ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2))) ∧
  k_set.card = 13 :=
begin
  sorry
end

end triangle_obtuse_count_l196_196248


namespace average_weight_l196_196263

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l196_196263


namespace proof_problem_l196_196967

-- Definitions of the sets U, A, B
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {2, 3, 4}

-- The complement of B with respect to U
def complement_U_B : Set ℕ := U \ B

-- The intersection of A and the complement of B with respect to U
def intersection_A_complement_U_B : Set ℕ := A ∩ complement_U_B

-- The statement we want to prove
theorem proof_problem : intersection_A_complement_U_B = {1, 6} :=
by
  sorry

end proof_problem_l196_196967


namespace number_of_pitchers_l196_196664

theorem number_of_pitchers (glasses_per_pitcher : ℕ) (total_glasses : ℕ) (hpitcher : glasses_per_pitcher = 6) (htotal : total_glasses = 54) : (total_glasses / glasses_per_pitcher) = 9 :=
by
  rw [htotal, hpitcher]
  norm_num
  sorry

end number_of_pitchers_l196_196664


namespace value_of_a_add_b_l196_196026

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l196_196026


namespace fraction_of_repeating_decimal_l196_196447

theorem fraction_of_repeating_decimal :
  let a := 36 / 100
      r := 1 / 100 
      series_sum := a / (1 - r)
  in series_sum = (4 / 11) := 
by
  let a : ℚ := 36 / 100 
  let r : ℚ := 1 / 100
  let series_sum : ℚ := a / (1 - r)
  have h₁ : series_sum = (9 / 25) / (99 / 100) := by sorry
  have h₂ : (9 / 25) / (99 / 100) = (9 / 25) * (100 / 99) := by sorry
  have h₃ : (9 / 25) * (100 / 99) = 900 / 2475 := by sorry
  have h₄ : 900 / 2475 = 36 / 99 := by sorry
  have h₅ : 36 / 99 = 4 / 11 := by sorry
  show series_sum = 4 / 11 from Eq.trans h₁ (Eq.trans h₂ (Eq.trans h₃ (Eq.trans h₄ h₅)))

end fraction_of_repeating_decimal_l196_196447


namespace sin_theta_phi_l196_196949

theorem sin_theta_phi (θ φ : ℝ) (h1 : complex.exp (complex.I * θ) = (4 / 5) + (3 / 5) * complex.I)
  (h2 : complex.exp (complex.I * φ) = (-5 / 13) + (12 / 13) * complex.I) : 
  real.sin (θ + φ) = 84 / 65 := 
by
  sorry

end sin_theta_phi_l196_196949


namespace zeros_before_first_nonzero_digit_l196_196091

theorem zeros_before_first_nonzero_digit 
  (h : ∀ n : ℕ, n = 2^3 * 5^6) : 
  (zeros_before_first_nonzero_digit_decimal \(\frac{1}{n}) = 6) :=
sorry

end zeros_before_first_nonzero_digit_l196_196091


namespace time_until_meet_l196_196772

-- Define conditions
def train1_length : ℝ := 300
def train1_time_crossing_pole : ℝ := 20
def train2_length : ℝ := 450
def train2_speed_kmph : ℝ := 90

-- Convert units and define speeds
def train1_speed : ℝ := train1_length / train1_time_crossing_pole
def train2_speed : ℝ := (train2_speed_kmph * 1000) / 3600

-- Calculate relative speed
def relative_speed : ℝ := train1_speed + train2_speed

-- Calculate total distance to be covered
def total_distance : ℝ := train1_length + train2_length

-- Statement proving the time it will take for the trains to meet
theorem time_until_meet : total_distance / relative_speed = 18.75 := by
  sorry

end time_until_meet_l196_196772


namespace chord_length_l196_196240

def line_parametric (t : ℝ) : ℝ × ℝ := (1 + 2 * t, 2 + t)

def circle_eq (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 = 9

theorem chord_length :
  ∀ (t : ℝ),  -- this represents the parametric form where t ranges over all reals 
  let l := 2 * real.sqrt (9 - (3 / real.sqrt 5)^2) in
  l = 12 / 5 * real.sqrt 5 :=
by
  sorry

end chord_length_l196_196240


namespace angle_bisector_le_altitude_l196_196333

variable {α : Type} [LinearOrderedField α]

-- Variables representing sides of the triangle
variables (a b c : α)

-- Variables for altitude and angle bisectors
variables (ha lc : α)

-- Hypotheses/conditions
hypothesis side_lengths : c > b ∧ b > a
hypothesis alt_ha : ha = (2 * (sqrt (s * (s - a) * (s - b )* (s - c)))) / a

hypothesis angle_bisector_lc:
  lc = (2 * b * c / (b + c)) * (cos ((angle_of_sides a b c) / 2))

theorem angle_bisector_le_altitude :
  lc ≤ ha :=
sorry

end angle_bisector_le_altitude_l196_196333


namespace greatest_distance_proof_l196_196673

-- Define the rectangle and circle properties
structure Rectangle where
  width : ℝ
  height : ℝ

structure Circle where
  diameter : ℝ

-- Define the conditions given in the problem
def rect := Rectangle.mk 15 10
def circle := Circle.mk 5
def radius := circle.diameter / 2

-- Definition of the greatest distance between the centers of two circles in the given rectangle
def greatestDistanceBetweenCenters (r : Rectangle) (c : Circle) : ℝ :=
  let horizontal_dist := r.width - 2 * (c.diameter / 2)
  let vertical_dist := r.height - 2 * (c.diameter / 2)
  Real.sqrt (horizontal_dist^2 + vertical_dist^2)

-- Theorem stating the answer to the problem
theorem greatest_distance_proof :
  greatestDistanceBetweenCenters rect circle = 5 * Real.sqrt 5 :=
by
  -- Skip the proof for now
  sorry

end greatest_distance_proof_l196_196673


namespace frustum_relationship_l196_196342

theorem frustum_relationship (r R h : ℝ) (h_pos: h > 0) (r_pos : r > 0) (R_pos : R > 0)
  (eq_cond : π * (r^2 + R^2) = π * (r + R) * sqrt(h^2 + (R - r)^2)) :
  2 / h = 1 / R + 1 / r :=
begin
  sorry
end

end frustum_relationship_l196_196342


namespace prove_incorrect_statement_l196_196688

-- Definitions based on given conditions
def isIrrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b ∧ b ≠ 0
def isSquareRoot (x y : ℝ) : Prop := x * x = y
def hasSquareRoot (x : ℝ) : Prop := ∃ y : ℝ, isSquareRoot y x

-- Options translated into Lean
def optionA : Prop := ∀ x : ℝ, isIrrational x → ¬ hasSquareRoot x
def optionB (x : ℝ) : Prop := 0 < x → ∃ y : ℝ, y * y = x ∧ (-y) * (-y) = x
def optionC : Prop := isSquareRoot 0 0
def optionD (a : ℝ) : Prop := ∀ x : ℝ, x = -a → (x ^ 3 = - (a ^ 3))

-- The incorrect statement according to the solution
def incorrectStatement : Prop := optionA

-- The theorem to be proven
theorem prove_incorrect_statement : incorrectStatement :=
by
  -- Replace with the actual proof, currently a placeholder using sorry
  sorry

end prove_incorrect_statement_l196_196688


namespace points_not_on_x_axis_l196_196881

theorem points_not_on_x_axis : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let count_points (x y : ℕ) := x ∈ digits ∧ y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ x ≠ y in
  (finset.univ.filter (λ p : ℕ × ℕ, count_points p.1 p.2)).card = 81 :=
sorry

end points_not_on_x_axis_l196_196881


namespace fraction_of_repeating_decimal_l196_196448

theorem fraction_of_repeating_decimal :
  let a := 36 / 100
      r := 1 / 100 
      series_sum := a / (1 - r)
  in series_sum = (4 / 11) := 
by
  let a : ℚ := 36 / 100 
  let r : ℚ := 1 / 100
  let series_sum : ℚ := a / (1 - r)
  have h₁ : series_sum = (9 / 25) / (99 / 100) := by sorry
  have h₂ : (9 / 25) / (99 / 100) = (9 / 25) * (100 / 99) := by sorry
  have h₃ : (9 / 25) * (100 / 99) = 900 / 2475 := by sorry
  have h₄ : 900 / 2475 = 36 / 99 := by sorry
  have h₅ : 36 / 99 = 4 / 11 := by sorry
  show series_sum = 4 / 11 from Eq.trans h₁ (Eq.trans h₂ (Eq.trans h₃ (Eq.trans h₄ h₅)))

end fraction_of_repeating_decimal_l196_196448


namespace share_of_a_in_profit_l196_196326

variable (rs6300 rs4200 rs10500 total_profit : ℝ)
variable (share_of_a : ℝ)
variable (gcd_ratio investment_a_ratio : ℝ)

-- Conditions
def investments := rs6300 = 6300 ∧ rs4200 = 4200 ∧ rs10500 = 10500
def total_profit_amount := total_profit = 12500
def gcd_value := gcd_ratio = 2100
def investment_ratios := investment_a_ratio = (rs6300 / gcd_ratio)

-- Goal
theorem share_of_a_in_profit : 
  investments ∧ total_profit_amount ∧ gcd_value ∧ (investment_a_ratio = 3) →
  share_of_a = 3750 := 
by
  sorry

end share_of_a_in_profit_l196_196326


namespace trigonometric_identity_l196_196999

theorem trigonometric_identity 
  (A B C : ℝ) (a b c : ℝ) (h1 : c = 1) 
  (h2 : cos B * sin C - (a - sin B) * cos C = 0)
  (C_eq_pi_div_4 : C = π / 4)
  (range_a_times_b : -1/2 ≤ a * b ∧ a * b ≤ sqrt 2 / 2) : 
   (C = π / 4) ∧ (-1/2 ≤ a * b ∧ a * b ≤ sqrt 2 / 2) :=
  sorry

end trigonometric_identity_l196_196999


namespace regression_and_income_l196_196668

-- Define the given data points
def months : List ℝ := [1, 2, 3, 4, 5]
def income : List ℝ := [0.3, 0.3, 0.5, 0.9, 1]

-- Define the means of x and y, and sums needed
def x_mean := (months.sum) / 5
def y_mean := (income.sum) / 5
def xy_sum := (List.zipWith (*) months income).sum
def x2_sum := (months.map (λ x => x * x)).sum

-- Define the regression coefficients
def b := (xy_sum - 5 * x_mean * y_mean) / (x2_sum - 5 * x_mean ^ 2)
def a := y_mean - b * x_mean

-- Define the regression line
def regression_line (t : ℝ) : ℝ := a + b * t

-- Define the prediction for September (month 9)
def income_september := regression_line 9

theorem regression_and_income : 
  regression_line = λ t, 0.2 * t 
  ∧ income_september <= 2 := by
sorry

end regression_and_income_l196_196668


namespace trapezoid_area_l196_196139

variable (A B C D K : Type)
variable [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty K]

-- Define the lengths as given in the conditions
def AK : ℝ := 16
def DK : ℝ := 4
def CD : ℝ := 6

-- Define the property that the trapezoid ABCD has an inscribed circle
axiom trapezoid_with_inscribed_circle (ABCD : Prop) : Prop

-- The Lean theorem statement
theorem trapezoid_area (ABCD : Prop) (AK DK CD : ℝ) 
  (H1 : trapezoid_with_inscribed_circle ABCD)
  (H2 : AK = 16)
  (H3 : DK = 4)
  (H4 : CD = 6) : 
  ∃ (area : ℝ), area = 432 :=
by
  sorry

end trapezoid_area_l196_196139


namespace probability_is_one_fourteenth_l196_196270

-- Define the set of numbers
def num_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the function to determine smallest difference >= 3
def valid_triplet (a b c : ℕ) : Prop :=
  a ∈ num_set ∧ b ∈ num_set ∧ c ∈ num_set ∧
  a < b ∧ b < c ∧ (b - a) ≥ 3 ∧ (c - b) ≥ 3

-- Count the number of valid triplets
noncomputable def count_valid_triplets : ℕ :=
  (num_set.to_list.comb 3).countp (λ t, match t with
                                        | [a, b, c] => valid_triplet a b c
                                        | _         => false
                                        end)

-- Total combinations of three numbers
def total_combinations : ℕ := num_set.card.choose 3

-- Define the probability
noncomputable def probability : ℚ :=
  count_valid_triplets / total_combinations

-- Theorem statement
theorem probability_is_one_fourteenth :
  probability = 1 / 14 := by
    sorry

end probability_is_one_fourteenth_l196_196270


namespace heidi_zoe_paint_wall_l196_196947

theorem heidi_zoe_paint_wall :
  let heidi_rate := (1 : ℚ) / 60
  let zoe_rate := (1 : ℚ) / 90
  let combined_rate := heidi_rate + zoe_rate
  let painted_fraction_15_minutes := combined_rate * 15
  painted_fraction_15_minutes = (5 : ℚ) / 12 :=
by
  sorry

end heidi_zoe_paint_wall_l196_196947


namespace count_zeros_in_fraction_l196_196097

theorem count_zeros_in_fraction : 
  ∃ n : ℕ, (to_decimal_representation (1 / (2 ^ 3 * 5 ^ 6)) = 0.000008) ∧ (n = 5) :=
by
  sorry

end count_zeros_in_fraction_l196_196097


namespace vector_projection_correct_l196_196860

def vec_a : ℝ × ℝ × ℝ := (4, -1, 3)
def dir_vec : ℝ × ℝ × ℝ := (-2, 3, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def scalar_mult (k : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (k * v.1, k * v.2, k * v.3)

def projection (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  scalar_mult (dot_product a b / dot_product b b) b

theorem vector_projection_correct :
  projection vec_a dir_vec = (8/7, -12/7, -4/7) :=
  sorry

end vector_projection_correct_l196_196860


namespace P_has_roots_P_eval_at_7_l196_196582

-- Define the polynomial P with the given conditions
def P (x : ℝ) : ℝ :=
  (3*x^4 - 45*x^3 + a*x^2 + b*x + c) * (4*x^4 - 64*x^3 + d*x^2 + e*x + f)

-- State that P(x) has the given roots with multiplicities
theorem P_has_roots :
  ∃ (a b c d e f : ℝ),
    {1, 2, 3, 4, 6} ⊆ {z : ℂ | (P(z.re) : ℝ) = 0} ∧
    multiset.count 1 (multiset.of_list [1, 2, 3, 3, 4, 6, 6, 6]) = 1 ∧
    multiset.count 2 (multiset.of_list [1, 2, 3, 3, 4, 6, 6, 6]) = 1 ∧
    multiset.count 3 (multiset.of_list [1, 2, 3, 3, 4, 6, 6, 6]) = 2 ∧
    multiset.count 4 (multiset.of_list [1, 2, 3, 3, 4, 6, 6, 6]) = 1 ∧
    multiset.count 6 (multiset.of_list [1, 2, 3, 3, 4, 6, 6, 6]) = 3 

-- The final statement to prove
theorem P_eval_at_7 : P 7 = 17280 :=
  sorry

end P_has_roots_P_eval_at_7_l196_196582


namespace average_of_first_201_terms_l196_196418

def a (n : ℕ) : ℤ := (-1)^(n + 1) * n

theorem average_of_first_201_terms :
  (1 / 201 * (∑ i in finset.range 201, a (i + 1)) : ℚ) = 101 / 201 := by
  sorry

end average_of_first_201_terms_l196_196418


namespace matrix_has_identical_rows_l196_196163

variable {M : Matrix ℕ ℕ ℕ} (r c : ℕ) (a : ℚ)
variable (H1 : ∀ i j, 0 ≤ M i j) (H2 : ∑ i j, M i j = r * c * a) (H3 : r > (10 * a + 10)^c)

theorem matrix_has_identical_rows
    (Hdistinct : ∀ i j, i < r → j < r → i ≠ j → M i = M j)
    : False :=
by
  intro h
  sorry


end matrix_has_identical_rows_l196_196163


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196302

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196302


namespace evaluate_fractions_l196_196814

theorem evaluate_fractions : (7 / 3 : ℚ) + (11 / 5) + (19 / 9) + (37 / 17) - 8 = 628 / 765 := by
  sorry

end evaluate_fractions_l196_196814


namespace vasya_incorrect_l196_196657

theorem vasya_incorrect :
  ¬(∀ (L R : set ℕ), (∀ x ∈ L, x ∈ {1..27}) ∧ (∀ x ∈ R, x ∈ {1..27}) ∧ L ∪ R = {1..27} ∧ L ∩ R = ∅ → (∃ (w_1 w_2 w_3 : ℕ), w_1 = 1 ∧ w_1 ∈ L ∧ w_2 ∈ L ∧ w_3 ∈ L ∧ L \ {w_1, w_2, w_3} = R)) :=
sorry

end vasya_incorrect_l196_196657


namespace probability_picking_pair_l196_196655

theorem probability_picking_pair : 
  let left_shoes := {A1, A2, A3} in
  let right_shoes := {B1, B2, B3} in
  let pairs := [(A1, B1), (A1, B2), (A1, B3), (A2, B1), (A2, B2), (A2, B3), (A3, B1), (A3, B2), (A3, B3)] in
  let desired_pairs := [(A1, B1), (A2, B2), (A3, B3)] in
  (desired_pairs.length / pairs.length) = (1 / 3) := 
by
  sorry

end probability_picking_pair_l196_196655


namespace locus_points_tangents_M_N_lines_through_various_R_l196_196699

-- Given problem conditions
variables (S : set Point)  -- Circle S
variables (P A B K M N R : Point)  -- Points P, A, B, K, M, N, R
variables (l : Line)  -- Line l
variables (tangent_to_S : Point → Line)  -- (lambda) function to get tangent line to the circle at a given point
variables (intersects : Line → Circle → list Point)
-- (Assume intersects finds intersection points of a line and a circle)
variables [∀ (X : Type), decidable_eq X]  -- Decision procedure 

-- Conditions given in the problem 
axiom circle_S : circle S 
axiom point_P_not_in_S : P ∉ S 
axiom line_l : line l 
axiom line_l_intersects_S_at_A_B : intersects l S = [A, B]
axiom tangents_A_B_intersect_at_K : tangent_to_S A ∩ tangent_to_S B = {K}

axiom lines_through_P_intersect_AK_BK : ∀ (L : Line), (P ∈ L) → 
  intersects L (line_through A K) = [M] ∧ intersects L (line_through B K) = [N]

-- Part (a) statement
theorem locus_points_tangents_M_N : 
    (∃ L : Line, ∀ M N : Point, 
        M ∈ intersects L (line_through A K) ∧ N ∈ intersects L (line_through B K) →
        ∀ T : Point, (T ∉ line_through A K) → (T ∉ line_through B K) → 
        T ∈ (tangent_to_S M ∩ tangent_to_S N) → T ∈ line_through K P) :=
sorry

-- Part (b) statement
theorem lines_through_various_R :
    (∀ R : Point, R ∈ S → ∃ Z : Point, Z ∈ l ∧ 
    ∀ p1 p2 : Point, p1 ∈ intersects (line_through R K) S → 
    p2 ∈ intersects (line_through R P) S → p1 ≠ R → p2 ≠ R →
    line_through p1 p2 = Z) :=
sorry

end locus_points_tangents_M_N_lines_through_various_R_l196_196699


namespace parallel_planes_mn_l196_196047

theorem parallel_planes_mn (m n : ℝ) (a b : ℝ × ℝ × ℝ) (α β : Type) (h1 : a = (0, 1, m)) (h2 : b = (0, n, -3)) 
  (h3 : ∃ k : ℝ, a = (k • b)) : m * n = -3 :=
by
  -- Proof would be here
  sorry

end parallel_planes_mn_l196_196047


namespace students_difference_l196_196766

variables (x1 x2 y1 y2 : ℤ)

def fewer_students_between_classes (x1 x2 y1 y2 : ℤ) : ℤ :=
  y2 - x1

lemma problem_condition_one (hx1 : x1 = x2 + 4) : Prop := hx1

lemma problem_condition_two (hy1 : y1 = y2 - 5) : Prop := hy1

lemma problem_condition_three (h_total : x1 + x2 = y1 + y2 - 17) : Prop := h_total

theorem students_difference
  (hx1 : x1 = x2 + 4)
  (hy1 : y1 = y2 - 5)
  (h_total : x1 + x2 = y1 + y2 - 17) :
  fewer_students_between_classes x1 x2 y1 y2 = 9 := by
  sorry

end students_difference_l196_196766


namespace length_of_chord_equals_16_l196_196913

-- Define the given conditions as Lean definitions and the final theorem
variables {p : ℝ} (hp : p > 0)

noncomputable def parabola_focus : ℝ := -2

def ellipse_foci : ℝ × ℝ := (0, -2)

theorem length_of_chord_equals_16
  (hp : p > 0)
  (h_parabola : ∀ x y : ℝ, C x y ↔ x ^ 2 = -2 * p * y)
  (h_focus : ellipse_foci = (0, parabola_focus))
  (h_tangents_abscissa : ∀ (A B : ℝ × ℝ), intersects_tangents A B = 4) :
  chord_length A B = 16 := sorry

end length_of_chord_equals_16_l196_196913


namespace value_of_x_squared_minus_y_squared_l196_196526

theorem value_of_x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x_squared_minus_y_squared_l196_196526


namespace center_of_symmetry_f_l196_196638

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin(2 * x + Real.pi / 6))^2 - Real.sin(4 * x + Real.pi / 3)

-- Define the center of symmetry function
def center_of_symmetry (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, f (2 * c.1 - x) = 2 * c.2 - f x

-- The theorem the center of symmetry is (-7π/48, 1)
theorem center_of_symmetry_f : center_of_symmetry f (-7 * Real.pi / 48, 1) :=
  sorry

end center_of_symmetry_f_l196_196638


namespace supplementary_angle_measure_l196_196241

theorem supplementary_angle_measure (a b : ℝ) 
  (h1 : a + b = 180) 
  (h2 : a / 5 = b / 4) : b = 80 :=
by
  sorry

end supplementary_angle_measure_l196_196241


namespace compound_ratio_is_one_fourteenth_l196_196331

theorem compound_ratio_is_one_fourteenth :
  (2 / 3) * (6 / 7) * (1 / 3) * (3 / 8) = 1 / 14 :=
by sorry

end compound_ratio_is_one_fourteenth_l196_196331


namespace lightest_pumpkin_weight_l196_196666

theorem lightest_pumpkin_weight 
  (A B C : ℕ)
  (h₁ : A + B = 12)
  (h₂ : B + C = 15)
  (h₃ : A + C = 13) :
  A = 5 :=
by
  sorry

end lightest_pumpkin_weight_l196_196666


namespace angle_BPC_theorem_l196_196992

structure Square (A B C D : Type) :=
  (side_length : ℝ)
  (AB : A = B)
  (BC : B = C)
  (CD : C = D)
  (DA : D = A)
  (length : AB = 6)

structure EquilateralTriangle (A B E : Type) :=
  (length : AB = BE ∧ BE = AE ∧ AE = AB)

structure Perpendicular (PQ BC : Type) :=
  (perp : PQ ⊥ BC)

def determine_angle_BPC (A B C D E P Q : Type) 
  (sq : Square A B C D) (tri : EquilateralTriangle A E B)
  (intersect : BE ∩ AC = P) (perp : Perpendicular PQ BC) : ℝ :=
  105

theorem angle_BPC_theorem (A B C D E P Q : Type) 
  (sq : Square A B C D) (tri : EquilateralTriangle A E B) 
  (intersect : BE ∩ AC = P) (perp : Perpendicular PQ BC) : determine_angle_BPC A B C D E P Q sq tri intersect perp = 105 :=
  sorry

end angle_BPC_theorem_l196_196992


namespace cut_figure_into_triangles_and_form_square_l196_196195

theorem cut_figure_into_triangles_and_form_square
  (A : ℕ) -- assume the area of the original figure is a natural number
  (original_figure : { x : ℕ // x = A }) -- the figure has area A
  (triangles : Fin 5 → { t : ℕ // t = A / 5 }) -- 5 triangles each contributing to the total area A
  (triangle_shape : ∀ i, is_triangle (triangles i)) -- ensures each part is a triangle
  (can_rearrange : can_form_square (triangles)) -- checks if we can rearrange triangles to form a square
  : original_figure = A := 
sorry

end cut_figure_into_triangles_and_form_square_l196_196195


namespace diagonals_of_square_are_equal_l196_196908

theorem diagonals_of_square_are_equal
  (H1 : ∀ (P : Type) [parallelogram P], diagonals_equal P)
  (H2 : ∀ (S : Type) [square S], parallelogram S) :
  ∀ (S : Type) [square S], diagonals_equal S :=
by
  sorry

end diagonals_of_square_are_equal_l196_196908


namespace find_interest_rate_l196_196364

noncomputable def interest_rate
  (total_sum : ℝ)
  (second_sum : ℝ)
  (num_years_1 : ℝ)
  (num_years_2 : ℝ)
  (rate_2 : ℝ) : ℝ :=
  (rate_2 * num_years_2) / num_years_1

theorem find_interest_rate
  (total_sum : ℝ)
  (second_sum : ℝ)
  (num_years_1 : ℝ)
  (num_years_2 : ℝ)
  (rate_2 : ℝ)
  (first_part : ℝ) :
  total_sum = first_part + second_sum →
  (first_part * interest_rate total_sum second_sum num_years_1 num_years_2 rate_2 * num_years_1 = second_sum * rate_2 * num_years_2) →
  interest_rate total_sum second_sum num_years_1 num_years_2 rate_2 = 0.03 :=
by
  intros h_sum h_interest
  sorry

-- Constants from the problem
#eval find_interest_rate 2665 1332.5 5 3 0.05 1332.5 -- Expected to evaluate to 3%

end find_interest_rate_l196_196364


namespace absolute_value_expression_l196_196390

theorem absolute_value_expression : 
  (abs ((-abs (-1 + 2))^2 - 1) = 0) :=
sorry

end absolute_value_expression_l196_196390


namespace remaining_volume_after_pours_l196_196725

-- Definitions based on the problem conditions
def initial_volume_liters : ℝ := 2
def initial_volume_milliliters : ℝ := initial_volume_liters * 1000
def pour_amount (x : ℝ) : ℝ := x

-- Statement of the problem as a theorem in Lean 4
theorem remaining_volume_after_pours (x : ℝ) : 
  ∃ remaining_volume : ℝ, remaining_volume = initial_volume_milliliters - 4 * pour_amount x :=
by
  -- To be filled with the proof
  sorry

end remaining_volume_after_pours_l196_196725


namespace annual_population_growth_l196_196626

noncomputable def annual_percentage_increase := 
  let P0 := 15000
  let P2 := 18150  
  exists (r : ℝ), (P0 * (1 + r)^2 = P2) ∧ (r = 0.1)

theorem annual_population_growth : annual_percentage_increase :=
by
  -- Placeholder proof
  sorry

end annual_population_growth_l196_196626


namespace probability_six_on_final_roll_l196_196739

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l196_196739


namespace white_tshirts_per_pack_l196_196422

-- Define the given conditions
def packs_white := 5
def packs_blue := 3
def t_shirts_per_blue_pack := 9
def total_t_shirts := 57

-- Define the total number of blue t-shirts
def total_blue_t_shirts := packs_blue * t_shirts_per_blue_pack

-- Define the variable W for the number of white t-shirts per pack
variable (W : ℕ)

-- Define the total number of white t-shirts
def total_white_t_shirts := packs_white * W

-- State the theorem to prove
theorem white_tshirts_per_pack :
    total_white_t_shirts + total_blue_t_shirts = total_t_shirts → W = 6 :=
by
  sorry

end white_tshirts_per_pack_l196_196422


namespace solid_is_cube_l196_196767

/-- 
If a solid's front view, side view, and top view are all congruent plane figures, 
then the solid is a cube.
-/
theorem solid_is_cube (solid : Type) (front_view side_view top_view : solid → plane_figure)
  (congruent_views : ∀ s : solid, front_view s = side_view s ∧ side_view s = top_view s) :
  (∃ s : solid, is_cube s) :=
sorry

end solid_is_cube_l196_196767


namespace points_concyclic_l196_196880

variable (P : Point) 
variable (ellipse : Curve) 
variable (A B C D : Point)
variable (l1 l2 : Line)
variable (α β : ℝ)
variable (a b : ℝ)

-- assumptions
axiom not_on_ellipse (hP : ¬ ellipse.contains P)
axiom ellipse_eq : ellipse = {pt : Point | pt.x^2 / a^2 + pt.y^2 / b^2 = 1}
axiom intersect_ellipse_l1 : ∀ t : ℝ, ∃ x y : ℝ, A = (x, y) ∨ B = (x, y) ∧ x = P.x + t * cos α ∧ y = P.y + t * sin α
axiom intersect_ellipse_l2 : ∀ p : ℝ, ∃ x y : ℝ, C = (x, y) ∨ D = (x, y) ∧ x = P.x + p * cos β ∧ y = P.y + p * sin β
axiom angles_sum_pi : α + β = π

-- to prove
theorem points_concyclic : ∃ circle : Curve, (circle.contains A) ∧ (circle.contains B) ∧ (circle.contains C) ∧ (circle.contains D) := 
sorry

end points_concyclic_l196_196880


namespace strawberry_unit_prices_l196_196689

theorem strawberry_unit_prices (x y : ℝ) (h1 : x = 1.5 * y) (h2 : 2 * x - 2 * y = 10) : x = 15 ∧ y = 10 :=
by
  sorry

end strawberry_unit_prices_l196_196689


namespace sin_A_value_c_value_from_area_l196_196562

-- Question (Ⅰ)
theorem sin_A_value (a c C : ℝ) (ha : a = 6) (hc : c = 14) (hC : C = (2 * Real.pi) / 3) : 
  Real.sin (∀ A : ℝ, ∃ hA : A = Real.asin((a / c) * Real.sin C), A) = (3 * Real.sqrt 3) / 14 :=
by
  -- provide the assumptions
  have A := Real.asin((a / c) * Real.sin C)
  -- goal
  sorry

-- Question (Ⅱ)
theorem c_value_from_area (a S C : ℝ) (ha : a = 6) (hS : S = 3 * Real.sqrt 3) (hC : C = (2 * Real.pi) / 3) :
  ∃ c, c = 2 * Real.sqrt 13 :=
by
  -- provide the assumptions
  have b := 2
  -- goal
  sorry

end sin_A_value_c_value_from_area_l196_196562


namespace radius_sphere_through_ABCD_l196_196548
open Real EuclideanGeometry

theorem radius_sphere_through_ABCD (A B C D : Point)
  (m n : Line) (hAB : A ≠ B) (hAC : A ≠ C)
  (hBD : B ≠ D) (hAB_length : dist A B = a)
  (hCD_length : dist C D = b) (theta : ℝ)
  (hm : m.through A) (hn : n.through B)
  (perp_m_AB : ⟂ m (line_through A B))
  (perp_n_AB : ⟂ n (line_through A B))
  (angle_mn : angle_between m n = θ) :
  ∃ r : ℝ, r = sqrt (a^2 + (b^2 / (4 * (sin θ)^2))) :=
by
  sorry

end radius_sphere_through_ABCD_l196_196548


namespace min_mouse_clicks_to_one_color_l196_196341

-- Define the size of the chessboard.
def chessboard_size : ℕ := 98

-- Define the chessboard as a structure with specific characteristics.
structure Chessboard :=
  (rows : ℕ)
  (columns : ℕ)
  (is_colored : rows = chessboard_size ∧ columns = chessboard_size)

-- Define the condition for a mouse click toggle on the chessboard.
def toggle (cb : Chessboard) (r1 c1 r2 c2 : ℕ) : Chessboard :=
  cb -- Placeholder for the actual implementation of toggle.

-- Define the target condition: entire chessboard is of one color.
def is_one_color (cb : Chessboard) : Prop :=
  ∀ i j, i < cb.rows ∧ j < cb.columns → cb.is_colored

-- Define the main theorem to prove the minimum number of mouse clicks.
theorem min_mouse_clicks_to_one_color (cb : Chessboard) : ∃ k, k = chessboard_size ∧
  (∀ n, n < chessboard_size → ∃ r1 c1 r2 c2, toggle cb r1 c1 r2 c2 = cb ∧ is_one_color (toggle cb r1 c1 r2 c2)) :=
sorry

end min_mouse_clicks_to_one_color_l196_196341


namespace coefficient_x2_of_product_l196_196680

def pol1 : Polynomial ℤ := -3*X^4 - 2*X^3 - 4*X^2 - 8*X + 2
def pol2 : Polynomial ℤ := 2*X^3 - 5*X^2 + 3*X - 1

theorem coefficient_x2_of_product :
  (pol1 * pol2).coeff 2 = -30 := by
  sorry

end coefficient_x2_of_product_l196_196680


namespace value_at_half_l196_196069

def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem value_at_half {f : ℝ → ℝ} (h₁ : f = power_function (-2)) 
                        (h₂ : f 2 = 1/4) : f (1/2) = 4 := 
by 
  sorry

end value_at_half_l196_196069


namespace multiply_469160_999999_l196_196383

theorem multiply_469160_999999 :
  469160 * 999999 = 469159530840 :=
by
  sorry

end multiply_469160_999999_l196_196383


namespace find_a_l196_196654

theorem find_a (a : ℝ) 
  (tangent_at_half : ∀ x, x = 1/2 → y = x^2 + a → y' = 2*x)
  (tangent_to_exp : ∀ x, y = exp x → y' = exp x)
  (tangent_points : ∃ x0, (∀ y = exp x0, y' = 1) ∧ (x0 = 0) ∧ (y = 1) ∧ (1 = a - 1/4)) :
  a = 5/4 :=
begin
  sorry
end

end find_a_l196_196654


namespace ratio_of_large_to_small_cup_approx_1_167_l196_196753

variable (S L : ℝ)
variable (C : ℝ)
variable (h1 : (1 / 5 : ℝ) * C * S)
variable (h2 : (4 / 5 : ℝ) * C * L)
variable (h3 : ((4 / 5 : ℝ) * L * C) / ((1 / 5 : ℝ) * S * C + (4 / 5 : ℝ) * L * C) = 0.8235294117647058)

theorem ratio_of_large_to_small_cup_approx_1_167 : L / S ≈ 1.1666666666666667 := by
  sorry

end ratio_of_large_to_small_cup_approx_1_167_l196_196753


namespace proof_general_formula_proof_sum_sequence_l196_196586

-- Define the sequence condition as a predicate
def seq_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a 1 + (finset.sum (finset.range (n-1))) (λ k, 2^k * a (k + 2)) = n

-- Define the general formula
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 1 / 2^(n-1)

-- Define the sum of the first n terms of the new sequence
noncomputable def sum_sequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, finset.sum (finset.range n) (λ k, a (k + 1) + real.log (a (k + 1)) / real.log 2)

-- Define the formula for the sum of the modified sequence
def sum_formula (n : ℕ) : ℝ :=
  2 - 1 / 2^(n-1) - n^2 / 2 + n / 2

-- State the two proof problems
theorem proof_general_formula (a : ℕ → ℝ) : seq_condition a → general_formula a :=
by 
  sorry

theorem proof_sum_sequence (a : ℕ → ℝ) (n : ℕ) : 
  seq_condition a → general_formula a → sum_sequence a n = sum_formula n :=
by 
  sorry

end proof_general_formula_proof_sum_sequence_l196_196586


namespace impact_point_coordinate_l196_196667

variables (R g α : ℝ)
noncomputable def V : ℝ := real.sqrt (2 * g * R * real.cos α)
noncomputable def T : ℝ := real.sqrt (2 * R / g) * (real.sin α * real.sqrt (real.cos α) + real.sqrt (1 - real.cos α ^ 3))
noncomputable def x_T : ℝ := R * (real.sin α + real.sin (2 * α) + real.sqrt (real.cos α * (1 - real.cos α ^ 3)))

theorem impact_point_coordinate :
  let x := λ t, R * real.sin α + V R g α * real.cos α * t,
      y := λ t, R * (1 - real.cos α) + V R g α * real.sin α * t - (g * t^2 / 2) in
  x T R g α = R * (real.sin α + real.sin (2 * α) + real.sqrt (real.cos α * (1 - real.cos α ^ 3))) :=
by sorry

end impact_point_coordinate_l196_196667


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196288

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196288


namespace simple_interest_total_l196_196696

theorem simple_interest_total {P R : ℝ} (h : (P * R * 10) / 100 = 1200) : 
  let P' := 3 * P in
  let R' := R in
  let T1 := 5 in
  let T2 := 5 in
  let SI := (P * R * T1) / 100 in
  let SI' := (P' * R' * T2) / 100 in
  SI + SI' = 3000 :=
begin
  sorry
end

end simple_interest_total_l196_196696


namespace intersection_complement_l196_196969

variable U A B : Set ℕ
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (A_def : A = {1, 3, 6})
variable (B_def : B = {2, 3, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 6} :=
by
  rw [U_def, A_def, B_def]
  simp
  sorry

end intersection_complement_l196_196969


namespace problem_statement_l196_196006

noncomputable def M (x y : ℝ) : ℝ := max x y
noncomputable def m (x y : ℝ) : ℝ := min x y

theorem problem_statement {p q r s t : ℝ} (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) :
  M (M p (m q r)) (m s (m p t)) = q :=
by
  sorry

end problem_statement_l196_196006


namespace not_prime_for_any_n_l196_196202

theorem not_prime_for_any_n (k : ℕ) (hk : 1 < k) (n : ℕ) : 
  ¬ Prime (n^4 + 4 * k^4) :=
sorry

end not_prime_for_any_n_l196_196202


namespace sum_series_l196_196929

open Complex

theorem sum_series :
  (∀ n : ℕ, ∃ (a_n b_n : ℝ), (2 + Complex.i)^n = a_n + b_n * Complex.i) →
  (∑ n : ℕ, (a_n * b_n) / (7:ℝ)^n) = (7:ℝ) / 16 :=
by
  intro h
  sorry

end sum_series_l196_196929


namespace total_fish_in_lake_l196_196868

theorem total_fish_in_lake :
  let w_ducks := 3
  let b_ducks := 7
  let m_ducks := 6
  let fish_per_white := 5
  let fish_per_black := 10
  let fish_per_multi := 12
  in w_ducks * fish_per_white + b_ducks * fish_per_black + m_ducks * fish_per_multi = 157 := by sorry

end total_fish_in_lake_l196_196868


namespace isosceles_triangle_equal_sides_length_l196_196223

theorem isosceles_triangle_equal_sides_length {
  base : ℝ,
  median : ℝ,
  a b c : Point } 
  (H1 : base = 4 * Real.sqrt 2)
  (H2 : median = 5)
  (isosceles : is_isosceles_triangle a b c)
  (median_property : is_median (Segment b c) (Segment a b) median) :
  distance a b = 6 ∧ distance b c = 6 := 
sorry

end isosceles_triangle_equal_sides_length_l196_196223


namespace area_of_region_l196_196385

noncomputable def integral_example : ℝ :=
  ∫ x in 1..2, exp (1 / x) / x^2

theorem area_of_region :
  integral_example = Real.exp 1 - Real.exp (1/2) := 
by
  sorry

end area_of_region_l196_196385


namespace find_p_value_l196_196928

-- Given conditions
def parabola (p : ℝ) : Set (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line_through_M (M : ℝ × ℝ) (slope : ℝ) : Set (ℝ × ℝ) := { (x, y) | y = slope * (x - M.1) + M.2 }

-- Prove that the value of p which satisfies the conditions is 2
theorem find_p_value (p : ℝ) (M : ℝ × ℝ) (slope : ℝ) 
  (hM : M = (1, 0)) (hSlope : slope = sqrt 3)
  (A B : ℝ × ℝ)
  (hLineA : A ∈ line_through_M M slope)
  (hLineB : B ∈ parabola p)
  (hMidpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  : p = 2 :=
sorry

end find_p_value_l196_196928


namespace evaluate_g_at_3_l196_196945

def g (x : ℤ) : ℤ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem evaluate_g_at_3 : g 3 = 113 := by
  -- Proof of g(3) = 113 skipped
  sorry

end evaluate_g_at_3_l196_196945


namespace find_angle_A_max_sin_product_l196_196974

variables {a b c : ℝ}
variables {A B C : ℝ}

-- Conditions from the problem
axiom cond1 : 2 * (b * c) * real.cos A = a ^ 2 - (b + c) ^ 2
axiom triangle_angles : A + B + C = real.pi
axiom angles_positive : A > 0 ∧ B > 0 ∧ C > 0

-- Problem 1: Find angle A
theorem find_angle_A : A = 2 * real.pi / 3 :=
sorry

-- Problem 2: Find maximum value of sin(A) * sin(B) * sin(C) and corresponding B and C
theorem max_sin_product : 
  ∃ (B C : ℝ), B = real.pi / 6 ∧ C = real.pi / 6 ∧ sin A * sin B * sin C = real.sqrt 3 / 8 :=
sorry

end find_angle_A_max_sin_product_l196_196974


namespace problem_statement_l196_196499

noncomputable def f (a x : ℝ) : ℝ := Real.log x - (a * (x + 1) / (x - 1))

theorem problem_statement (a : ℝ) :
  (a > 0 → ∀ x ∈ Ioi 1, deriv (f a) x > 0) ∧
  ((deriv (f a) 2 = 2) → a = 3/4) ∧
  (a > 0 → (∃ x1 x2 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ x1 ≠ x2 ∧ x1 * x2 = 1)) :=
by
  sorry

end problem_statement_l196_196499


namespace T_b_T_neg_b_eq_4800_l196_196864

-- Definitions based on conditions
noncomputable def T (r : ℝ) : ℝ := 20 / (1 - r)

-- Problem statement in Lean 4
theorem T_b_T_neg_b_eq_4800 (b : ℝ) (h_b : -1 < b ∧ b < 1) (h_Tb_Tneg_b : T b * T (-b) = 4800) :
  T b + T (-b) = 480 :=
by
  have H1 : T b = 20 / (1 - b) := rfl
  have H2 : T (-b) = 20 / (1 + b) := rfl
  sorry

end T_b_T_neg_b_eq_4800_l196_196864


namespace racecourse_min_distance_l196_196126

noncomputable def min_distance (d_A_wall : ℝ) (d_B_wall : ℝ) (wall_length : ℝ) : ℝ :=
  let d_total := d_A_wall + d_B_wall in
  Real.sqrt (wall_length^2 + d_total^2)

theorem racecourse_min_distance : min_distance 500 700 1400 = 1843 := by
  sorry

end racecourse_min_distance_l196_196126


namespace area_ratio_oblique_axonometric_l196_196320

theorem area_ratio_oblique_axonometric 
(base original_height perspective_height : ℝ) 
(h_base : base = base) 
(h_height : perspective_height = original_height * (sqrt 2 / 2)) :
(perspective_height * base / 2) / (original_height * base / 2) = sqrt 2 / 4 := 
sorry

end area_ratio_oblique_axonometric_l196_196320


namespace number_of_ways_to_get_off_l196_196258

theorem number_of_ways_to_get_off (n_passengers : ℕ) (n_stations : ℕ) :
  n_passengers = 10 → n_stations = 5 → (n_stations ^ n_passengers = 5 ^ 10) :=
by
  intros h_passengers h_stations
  rw [h_passengers, h_stations]
  exact rfl

end number_of_ways_to_get_off_l196_196258


namespace sum_of_inscribed_circle_radii_l196_196611

theorem sum_of_inscribed_circle_radii 
  (A B C D : Type) [cyclic_quad A B C D]
  (r_ABC r_ACD r_BCD r_BDA : ℝ)
  (h1 : inscribed_circle_radius A B C = r_ABC)
  (h2 : inscribed_circle_radius A C D = r_ACD)
  (h3 : inscribed_circle_radius B C D = r_BCD)
  (h4 : inscribed_circle_radius B D A = r_BDA) :
  r_ABC + r_ACD = r_BCD + r_BDA := 
  sorry

end sum_of_inscribed_circle_radii_l196_196611


namespace sqrt_of_sixteen_l196_196396

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_sixteen_l196_196396


namespace angle_between_lateral_and_base_is_45_l196_196236

noncomputable def regular_hexagonal_pyramid_angle (a : ℝ) : ℝ := 
  let M := (0 : ℝ, 0 : ℝ, 0 : ℝ)   -- Center of the hexagon
  let P := (0 : ℝ, 0 : ℝ, a)        -- Apex of the pyramid
  let A := (a : ℝ, 0 : ℝ, 0 : ℝ)    -- Vertex of the hexagon at the base
  real_angle P A M

theorem angle_between_lateral_and_base_is_45 :
  ∀ (a : ℝ), a > 0 → regular_hexagonal_pyramid_angle a = 45 :=
by
  intros
  unfold regular_hexagonal_pyramid_angle
  sorry

end angle_between_lateral_and_base_is_45_l196_196236


namespace range_of_f_l196_196455

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan x

theorem range_of_f :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x ∈ set.Icc (π / 4) (3 * π / 4) :=
by
  intro x
  sorry

end range_of_f_l196_196455


namespace rice_bags_proof_l196_196257

-- Define the condition
def weights := [50 + 0.5, 50 + 0.3, 50 + 0, 50 - 0.2, 50 - 0.3, 50 + 1.1, 50 - 0.7, 50 - 0.2, 50 + 0.6, 50 + 0.7]

-- Total weight to be proven
def total_weight : ℝ := 501.7

-- Total excess to be proven
def total_excess : ℝ := 1.7

-- Average weight per bag to be proven
def average_weight : ℝ := 50.17

-- Proof goal
theorem rice_bags_proof :
  (List.sum weights = total_weight) ∧
  (total_weight - 50 * 10 = total_excess) ∧
  (total_weight / 10 = average_weight) :=
by
  sorry

end rice_bags_proof_l196_196257


namespace find_center_of_circle_l196_196852

noncomputable def polar_center_eq : Prop :=
  let ρ := λ θ, Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  ∃ θ, ρ θ = Real.sqrt 2 * (Real.cos θ + Real.sin θ) ∧ (ρ θ, θ) = (1, Real.pi / 4)

theorem find_center_of_circle :
  polar_center_eq :=
by
  sorry

end find_center_of_circle_l196_196852


namespace seq_150th_term_l196_196796

def seq_element (n : ℕ) : ℕ := 
  if n = 0 then 1
  else let bin := nat.to_digits 2 n in
    list.foldl (λ acc p, acc + if (list.get_or_else bin p 0) = 1 
                                then 3 ^ (p / 2) * 5 ^ (p % 2) 
                                else 0) 0 (list.range bin.length)

theorem seq_150th_term : seq_element 150 = 2840 := 
by {
  -- Here will be the proof, left as sorry for now
  sorry
}

end seq_150th_term_l196_196796


namespace triangle_area_condition_l196_196058

theorem triangle_area_condition (m : ℝ) 
  (H_line : ∀ (x y : ℝ), x - m*y + 1 = 0)
  (H_circle : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 4)
  (H_area : ∃ (A B C : (ℝ × ℝ)), (x - my + 1 = 0) ∧ (∃ C : (ℝ × ℝ), (x1 - 1)^2 + y1^2 = 4 ∨ (x2 - 1)^2 + y2^2 = 4))
  : m = 2 :=
sorry

end triangle_area_condition_l196_196058


namespace find_ab_l196_196113

theorem find_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_area_9 : (1/2) * (12 / a) * (12 / b) = 9) : 
  a * b = 8 := 
by 
  sorry

end find_ab_l196_196113


namespace complex_multiplication_l196_196886

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- The theorem stating the equality
theorem complex_multiplication : (2 + i) * (3 + i) = 5 + 5 * i := 
sorry

end complex_multiplication_l196_196886


namespace ten_thousandths_digit_of_five_over_thirty_two_l196_196291

theorem ten_thousandths_digit_of_five_over_thirty_two : 
  (Rat.ofInt 5 / Rat.ofInt 32).toDecimalString 5 = "0.15625" := 
by 
  sorry

end ten_thousandths_digit_of_five_over_thirty_two_l196_196291


namespace log_base_5_of_inv_sqrt_5_l196_196821

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196821


namespace marj_money_left_l196_196595

def marj_two_twenty_bills : ℝ := 2 * 20
def marj_three_five_bills : ℝ := 3 * 5
def marj_loose_coins : ℝ := 4.50
def cake_cost : ℝ := 17.50

theorem marj_money_left : marj_two_twenty_bills + marj_three_five_bills + marj_loose_coins - cake_cost = 42 :=
by
  calc
    marj_two_twenty_bills + marj_three_five_bills + marj_loose_coins - cake_cost
    = (2 * 20) + (3 * 5) + 4.50 - 17.50 : by rfl
    ... = 40 + 15 + 4.50 - 17.50 : by rfl
    ... = 59.50 - 17.50 : by rfl
    ... = 42 : by rfl

end marj_money_left_l196_196595


namespace problem_statement_l196_196082

def U := Set ℝ
def M := { x : ℝ | x^2 - 4 * x - 5 < 0 }
def N := { x : ℝ | 1 ≤ x }
def comp_U_N := { x : ℝ | x < 1 }
def intersection := { x : ℝ | -1 < x ∧ x < 1 }

theorem problem_statement : M ∩ comp_U_N = intersection := sorry

end problem_statement_l196_196082


namespace difference_of_squares_l196_196684

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end difference_of_squares_l196_196684


namespace intersection_point_sum_l196_196916

theorem intersection_point_sum {h j : ℝ → ℝ} 
    (h3: h 3 = 3) (j3: j 3 = 3) 
    (h6: h 6 = 9) (j6: j 6 = 9)
    (h9: h 9 = 18) (j9: j 9 = 18)
    (h12: h 12 = 18) (j12: j 12 = 18) :
    ∃ a b, (h (3 * a) = 3 * j a ∧ a + b = 22) := 
sorry

end intersection_point_sum_l196_196916


namespace max_dot_product_exists_theta_for_sum_max_magnitude_difference_l196_196013

noncomputable def a : ℝ × ℝ := (1, real.sqrt 3)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sin θ)

theorem max_dot_product (θ : ℝ): 
  2 * real.sin (θ + (real.pi / 6)) ≤ 2 :=
by sorry

theorem exists_theta_for_sum : 
  ∃ θ, real.sqrt ((1 + real.cos θ)^2 + (real.sqrt 3 + real.sin θ)^2) = 
       (real.sqrt (1^2 + (real.sqrt 3)^2) + real.sqrt ((real.cos θ)^2 + (real.sin θ)^2)) :=
by sorry

theorem max_magnitude_difference (θ : ℝ) : 
  real.sqrt ((1 - real.cos θ)^2 + (real.sqrt 3 - real.sin θ)^2) ≤ 3 :=
by sorry

end max_dot_product_exists_theta_for_sum_max_magnitude_difference_l196_196013


namespace n_cube_plus_5n_divisible_by_6_l196_196198

theorem n_cube_plus_5n_divisible_by_6 (n : ℤ) : 6 ∣ (n^3 + 5 * n) := 
sorry

end n_cube_plus_5n_divisible_by_6_l196_196198


namespace probability_six_greater_than_five_over_six_l196_196745

noncomputable def sumBeforeLastRoll (n : ℕ) (Y : ℕ → ℕ) : Prop :=
  Y (n - 1) + 6 >= 2019

noncomputable def probabilityLastRollSix (n : ℕ) (S : ℕ) : Prop :=
  S = 6

theorem probability_six_greater_than_five_over_six (n : ℕ) :
  ∀ (Y : ℕ → ℕ) (S : ℕ), sumBeforeLastRoll n Y →
  probabilityLastRollSix n S →
  (∑ k in range 1 7, probabilityLastRollSix (n - k)) > (5 / 6) :=
begin
  -- Proof would go here
  sorry
end

end probability_six_greater_than_five_over_six_l196_196745


namespace simplify_expression_l196_196212

theorem simplify_expression (x : ℝ) :
  (sqrt (x^2 - 4 * x + 4) + sqrt (x^2 + 4 * x + 4)) = abs (x - 2) + abs (x + 2) :=
by
  sorry

end simplify_expression_l196_196212


namespace discount_percentage_correct_l196_196185

-- Define the conditions
def cost_price : ℝ := 540
def mark_up_percentage : ℝ := 0.15
def selling_price : ℝ := 459
def marked_price : ℝ := cost_price * (1 + mark_up_percentage)
def discount : ℝ := marked_price - selling_price
def discount_percentage : ℝ := (discount / marked_price) * 100

-- Statement to prove
theorem discount_percentage_correct : discount_percentage ≈ 26.09 :=
by
  -- This proof step would be filled in by actually proving the theorem.
  sorry

end discount_percentage_correct_l196_196185


namespace ellipse_parameters_l196_196781

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def point_on_ellipse (x y : ℝ) (h k a b : ℝ) :=
(x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

theorem ellipse_parameters
  (a b h k : ℝ)
  (f1 := (3, 3))
  (f2 := (3, 9))
  (p := (16, -2))
  (h_eq : h = 3)
  (k_eq : k = 6)
  (a_eq : a = (real.sqrt 194 + real.sqrt 290) / 2)
  (b_eq : b = real.sqrt ((real.sqrt 194 + real.sqrt 290) / 2) ^ 2 - 9)
  (pos_a : 0 < a)
  (pos_b : 0 < b) :
  point_on_ellipse 16 (-2) h k a b :=
sorry

end ellipse_parameters_l196_196781


namespace platform_length_l196_196345

theorem platform_length
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (train_length_m : ℕ)
  (h_speed : speed_kmph = 72)
  (h_time : time_sec = 26)
  (h_train_length : train_length_m = 440) :
  ∃ platform_length_m, platform_length_m = 80 := 
by
  -- Here we represent the conversion factor from km/hr to m/s as a constant.
  let conversion_factor : ℝ := 5.0 / 18.0
  -- Speed in m/s
  let speed_mps : ℝ := speed_kmph * conversion_factor
  -- Distance covered by the train while crossing the platform in meters.
  let distance_covered_m : ℝ := speed_mps * time_sec
  -- Using the equation distance_covered = length_of_train + length_of_platform
  -- We need to prove that length_of_platform = 80 meters.
  have h_distance_covered : distance_covered_m = 520 := by
    calc
      distance_covered_m = 72 * conversion_factor * 26 : by rw [h_speed, h_time] 
      ... = 20 * 26 : by norm_num
      ... = 520 : by norm_num

  have : distance_covered_m = train_length_m + 80 := by
    rw [h_train_length, ← add_assoc]
    exact h_distance_covered

  use 80
  exact this.


end platform_length_l196_196345


namespace infinite_product_l196_196799

noncomputable def sequence_a : ℕ → ℚ
| 0       := 1/3
| (n + 1) := 1 + (sequence_a n - 1)^2

theorem infinite_product (prod : ℚ) (H : prod = (1 / 3)) :
  (∏ (i:ℕ) (h: i < 100), sequence_a i) = prod := sorry

end infinite_product_l196_196799


namespace quadratic_real_solutions_l196_196846

namespace Proof

theorem quadratic_real_solutions (n : ℕ) 
    (a : Fin (n + 2) → ℝ) : 
    (∀ x : ℝ, (a n + 1) * x^2 - 2 * x * sqrt (Σ i, (a i)^2) + (Σ i in Finset.range (n + 1), a i) = 0 → 
    ∃ x : ℝ, (a n + 1) * x^2 - 2 * x * sqrt (Σ i, (a i)^2) + (Σ i in Finset.range (n + 1), a i) = 0) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
begin
    sorry
end

end Proof

end quadratic_real_solutions_l196_196846


namespace deal_or_no_deal_l196_196556

theorem deal_or_no_deal :
  let boxes : List ℝ := [0.01, 1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000, 5000000]
  let good_boxes : Finset ℝ := {200000, 300000, 400000, 500000, 750000, 1000000, 5000000}.to_finset
  in ∃ n : Nat, n ≥ 7 ∧ n ≤ 30 - 23 ∧
    (List.length (List.filter (λ x : ℝ, x ∉ good_boxes) (List.drop n boxes)) < 3 * good_boxes.card) :=
sorry

end deal_or_no_deal_l196_196556


namespace benny_march_savings_l196_196785

theorem benny_march_savings :
  (january_add : ℕ) (february_add : ℕ) (march_total : ℕ) 
  (H1 : january_add = 19) (H2 : february_add = 19) (H3 : march_total = 46) :
  march_total - (january_add + february_add) = 8 := 
by
  sorry

end benny_march_savings_l196_196785


namespace log_base_5_of_inv_sqrt_5_l196_196824

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196824


namespace exists_circles_with_perpendicular_tangents_l196_196507

variables (A B C A' B' C' : Point)
variables (α β γ : ℝ)

def isosceles_triangle (P Q R : Point) : Prop :=
  distance P Q = distance P R

def mutually_perpendicular_tangents (α β γ : ℝ) : Prop :=
  ∃ (angle_A angle_B angle_C : ℝ), 
    (β + γ + angle_A = 90) ∧
    (γ + α + angle_B = 90) ∧
    (α + β + angle_C = 90)

theorem exists_circles_with_perpendicular_tangents
    (h_non_collinear : ¬ collinear A B C)
    (h_circles_centers : center_of_circle A' B C ∧ center_of_circle B' A C ∧ center_of_circle C' A B)
    (h_isosceles : isosceles_triangle B A' C ∧ isosceles_triangle A B' C ∧ isosceles_triangle A C' B)
    : ∃ (A' B' C' : Point),
        (circle_through B C A') ∧ (circle_through A C B') ∧ (circle_through A B C') ∧ 
        mutually_perpendicular_tangents α β γ := 
      sorry

end exists_circles_with_perpendicular_tangents_l196_196507


namespace merchant_discount_percentage_l196_196756

theorem merchant_discount_percentage :
  ∀ (CP MP SP : ℝ) (M_percent P_percent : ℝ),
  CP = 100 →
  M_percent = 0.40 →
  P_percent = 0.12 →
  MP = CP + (M_percent * CP) →
  SP = CP + (P_percent * CP) →
  ∃ (D_percent : ℝ), D_percent = 20 :=
by
  intro CP MP SP M_percent P_percent hCP hM_percent hP_percent hMP hSP
  use 20
  sorry

end merchant_discount_percentage_l196_196756


namespace log_base_5_of_inv_sqrt_5_l196_196826

theorem log_base_5_of_inv_sqrt_5 : log 5 (1 / real.sqrt 5) = -1 / 2 := 
sorry

end log_base_5_of_inv_sqrt_5_l196_196826


namespace marj_money_left_l196_196594

def marj_two_twenty_bills : ℝ := 2 * 20
def marj_three_five_bills : ℝ := 3 * 5
def marj_loose_coins : ℝ := 4.50
def cake_cost : ℝ := 17.50

theorem marj_money_left : marj_two_twenty_bills + marj_three_five_bills + marj_loose_coins - cake_cost = 42 :=
by
  calc
    marj_two_twenty_bills + marj_three_five_bills + marj_loose_coins - cake_cost
    = (2 * 20) + (3 * 5) + 4.50 - 17.50 : by rfl
    ... = 40 + 15 + 4.50 - 17.50 : by rfl
    ... = 59.50 - 17.50 : by rfl
    ... = 42 : by rfl

end marj_money_left_l196_196594


namespace a_n_plus_1_geometric_and_sum_b_n_l196_196471

theorem a_n_plus_1_geometric_and_sum_b_n (n : ℕ) : 
  (∀ n, (a : ℕ → ℕ) (a 1 = 1) (a (n + 1) = 2 * a n + 1)) ∧
  (b : ℕ → ℕ) (b n = n * a n) ∧
  (S : ℕ → ℕ) (S n = ∑ i in range n, b i)
  → (∀ n, a n + 1 = 2 * 2 ^ (n - 1)) ∧ (a n = 2 ^ n - 1) ∧
  (S n = (n - 2) * 2 ^ (n + 1) + 2 - n * (n + 1) / 2) :=
by
  sorry

end a_n_plus_1_geometric_and_sum_b_n_l196_196471


namespace num_distinct_differences_l196_196939

def differences (s : Set ℕ) : Set ℕ :=
  {d | ∃ a b ∈ s, a > b ∧ d = a - b}

theorem num_distinct_differences : 
  let s := {1, 2, 3, ..., 25}
  (differences s).card = 24 :=
by
  let s := {1 ≤ x | x ≤ 25}
  have h : ∀ d ∈ differences s, 1 ≤ d ∧ d ≤ 24 :=
  sorry
  have max_diffs : {d ∈ differences s | 1 ≤ d ∧ d ≤ 24}.card = 24 :=
  sorry
  exact max_diffs

end num_distinct_differences_l196_196939


namespace total_outfits_l196_196623

theorem total_outfits 
  (shirts : ℕ) 
  (ties : ℕ) 
  (pants : ℕ) 
  (belts : ℕ) 
  (extra_tie_option : ℕ)
  (extra_belt_option : ℕ) :
  shirts = 8 → 
  ties = 5 → 
  pants = 4 → 
  belts = 2 → 
  extra_tie_option = 1 →
  extra_belt_option = 1 →
  shirts * pants * (ties + extra_tie_option) * (belts + extra_belt_option) = 576 := 
by 
  intros h_shirts h_ties h_pants h_belts h_extra_tie_option h_extra_belt_option
  rw [h_shirts, h_ties, h_pants, h_belts, h_extra_tie_option, h_extra_belt_option]
  norm_num
  sorry

end total_outfits_l196_196623


namespace diff_of_squares_l196_196686

theorem diff_of_squares (x y : ℝ) :
  (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x + y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (-x + y) * (x - y)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (x + 2) * (2 + x)) ∧ 
  ¬ (∃ a b : ℝ, (a - b) * (a + b) = (2x + 3) * (3x - 2)) := 
by 
  sorry

end diff_of_squares_l196_196686


namespace count_zeros_in_fraction_l196_196094

theorem count_zeros_in_fraction : 
  ∃ n : ℕ, (to_decimal_representation (1 / (2 ^ 3 * 5 ^ 6)) = 0.000008) ∧ (n = 5) :=
by
  sorry

end count_zeros_in_fraction_l196_196094


namespace value_of_a_plus_b_l196_196023

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l196_196023


namespace evaluate_expression_l196_196839

theorem evaluate_expression : 
  (1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))) = 5 / 7 :=
by
  sorry

end evaluate_expression_l196_196839


namespace surface_area_eighth_block_l196_196773

theorem surface_area_eighth_block {A B C D E F G H : ℕ} 
  (blockA : A = 148) 
  (blockB : B = 46) 
  (blockC : C = 72) 
  (blockD : D = 28) 
  (blockE : E = 88) 
  (blockF : F = 126) 
  (blockG : G = 58) 
  : H = 22 :=
by 
  sorry

end surface_area_eighth_block_l196_196773


namespace total_fish_in_lake_l196_196867

theorem total_fish_in_lake :
  let w_ducks := 3
  let b_ducks := 7
  let m_ducks := 6
  let fish_per_white := 5
  let fish_per_black := 10
  let fish_per_multi := 12
  in w_ducks * fish_per_white + b_ducks * fish_per_black + m_ducks * fish_per_multi = 157 := by sorry

end total_fish_in_lake_l196_196867


namespace log_base_2_of_1_l196_196706

theorem log_base_2_of_1 : log 2 1 = 0 := by
  sorry

end log_base_2_of_1_l196_196706


namespace digit_in_ten_thousandths_place_of_fraction_l196_196278

theorem digit_in_ten_thousandths_place_of_fraction (n d : ℕ) (h1 : n = 5) (h2 : d = 32) :
  (∀ {x : ℕ}, (decimalExpansion n d x 4 = 5) ↔ (x = 10^4)) :=
sorry

end digit_in_ten_thousandths_place_of_fraction_l196_196278


namespace sequence_general_formula_l196_196034

theorem sequence_general_formula (a : ℕ → ℕ) :
  (∀ n : ℕ, (∑ i in finset.range (n + 1), (i + 1) * a (i + 1)) = n * (n + 1) * (n + 2)) →
  (∀ n : ℕ, a n = 3 * n + 3) :=
by
  sorry

end sequence_general_formula_l196_196034


namespace lions_deers_15_minutes_l196_196111

theorem lions_deers_15_minutes :
  ∀ (n : ℕ), (15 * n = 15 * 15 → n = 15 → ∀ t, t = 15) := by
  sorry

end lions_deers_15_minutes_l196_196111


namespace enemies_left_undefeated_l196_196127

theorem enemies_left_undefeated (points_per_enemy : ℕ) (total_enemies : ℕ) (total_points_earned : ℕ) 
  (h1: points_per_enemy = 9) (h2: total_enemies = 11) (h3: total_points_earned = 72):
  total_enemies - (total_points_earned / points_per_enemy) = 3 :=
by
  sorry

end enemies_left_undefeated_l196_196127


namespace value_of_x_squared_minus_y_squared_l196_196522

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l196_196522


namespace money_spent_on_games_l196_196942

noncomputable def total_allowance : ℕ := 40
noncomputable def fraction_movies : ℚ := 1/4
noncomputable def fraction_burgers : ℚ := 1/8
noncomputable def fraction_ice_cream : ℚ := 1/5
noncomputable def fraction_music : ℚ := 1/4
noncomputable def fraction_games : ℚ := 3/20

theorem money_spent_on_games :
  let
    spent_movies := fraction_movies * total_allowance
    spent_burgers := fraction_burgers * total_allowance
    spent_ice_cream := fraction_ice_cream * total_allowance
    spent_music := fraction_music * total_allowance
    remaining_money := total_allowance - (spent_movies + spent_burgers + spent_ice_cream + spent_music)
  in remaining_money = 7 := 
by
  sorry

end money_spent_on_games_l196_196942


namespace count_valid_three_digit_numbers_l196_196011

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem count_valid_three_digit_numbers : 
  ∃ (count : ℕ), count = 54 ∧
  ∀ (even_digits : Finset ℕ) (odd_digits : Finset ℕ),
    even_digits = {2, 4, 6} ∧ odd_digits = {1, 3, 5} ∨ odd_digits = {1, 3, 5} ∧ even_digits = {2, 4, 6} →
    let valid_combinations := (even_digits.product (odd_digits.product odd_digits)).filter (λ n, is_three_digit (n.1 * 100 + n.2.1 * 10 + n.2.2)) in
    count = valid_combinations.card :=
  begin
    use 54,
    intros even_digits odd_digits h,
    sorry
  end

end count_valid_three_digit_numbers_l196_196011


namespace no_right_angled_triangle_in_cube_cross_section_l196_196682

theorem no_right_angled_triangle_in_cube_cross_section
  (P : Plane) (C : Cube) :
  ¬ (right_angled_triangle (P ∩ C)) :=
sorry

end no_right_angled_triangle_in_cube_cross_section_l196_196682


namespace perpendicular_condition_l196_196085

-- Definitions for the given vectors and their perpendicular condition
def a : Vect2 := ⟨1, 2⟩
def b (x : ℝ) : Vect2 := ⟨2, x⟩

-- Definition of perpendicular vectors in terms of dot product
def perpendicular (v w : Vect2) : Prop :=
  dot_product v w = 0

-- The theorem to be proved
-- If vectors a and b(x) are perpendicular, then x equals -1
theorem perpendicular_condition (x : ℝ) :
  perpendicular a (b x) → x = -1 := sorry

end perpendicular_condition_l196_196085


namespace range_of_m_l196_196961

variable (x m : ℝ)
hypothesis : (x + m) / (x - 2) + (2 * m) / (2 - x) = 3
hypothesis_pos : 0 < x

theorem range_of_m :
  m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_l196_196961


namespace line_intersects_circle_min_length_chord_l196_196888

-- Given the circle and the line as described
def circle (x y : ℝ) : Prop := (x-1)^2 + (y-2)^2 = 25

def line (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Prove that the line always intersects the circle at two points for any real m
theorem line_intersects_circle (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, circle x1 y1 ∧ circle x2 y2 ∧ line m x1 y1 ∧ line m x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) := 
sorry

-- Prove the equation of the line when the chord cut by the circle is at its minimum length
noncomputable def line_with_min_length : Prop :=
  ∃ x y : ℝ, line 2 x y ∧ y - 1 = 2 * (x - 3)

theorem min_length_chord : 
  line_with_min_length ∧ (∃ x y : ℝ, line 2 x y ∧ y - 1 = 2 * (x - 3) ∧ 2 * x - y - 5 = 0) :=
sorry

end line_intersects_circle_min_length_chord_l196_196888


namespace intersection_complement_l196_196970

variable U A B : Set ℕ
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (A_def : A = {1, 3, 6})
variable (B_def : B = {2, 3, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 6} :=
by
  rw [U_def, A_def, B_def]
  simp
  sorry

end intersection_complement_l196_196970


namespace fish_in_lake_l196_196874

theorem fish_in_lake (white_ducks black_ducks multicolor_ducks : ℕ) 
                     (fish_per_white fish_per_black fish_per_multicolor : ℕ)
                     (h1 : fish_per_white = 5)
                     (h2 : fish_per_black = 10)
                     (h3 : fish_per_multicolor = 12)
                     (h4 : white_ducks = 3)
                     (h5 : black_ducks = 7)
                     (h6 : multicolor_ducks = 6) :
                     (white_ducks * fish_per_white + black_ducks * fish_per_black + multicolor_ducks * fish_per_multicolor) = 157 :=
by
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end fish_in_lake_l196_196874


namespace intersection_complement_l196_196971

variable U A B : Set ℕ
variable (U_def : U = {1, 2, 3, 4, 5, 6})
variable (A_def : A = {1, 3, 6})
variable (B_def : B = {2, 3, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {1, 6} :=
by
  rw [U_def, A_def, B_def]
  simp
  sorry

end intersection_complement_l196_196971


namespace find_a_plus_b_plus_c_l196_196439

-- Definitions of conditions
def is_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) := 
  ∀ x : ℝ, vertex_y = (a * (vertex_x ^ 2)) + (b * vertex_x) + c

def contains_point (a b c : ℝ) (x y : ℝ) := 
  y = (a * (x ^ 2)) + (b * x) + c

theorem find_a_plus_b_plus_c
  (a b c : ℝ)
  (h_vertex : is_vertex a b c 3 4)
  (h_symmetry : ∃ h : ℝ, ∀ x : ℝ, a * (x - h) ^ 2 = a * (h - x) ^ 2)
  (h_contains : contains_point a b c 1 0)
  : a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l196_196439


namespace count_zeros_decimal_representation_l196_196105

theorem count_zeros_decimal_representation (n m : ℕ) (h : n = 3) (h₁ : m = 6) : 
  ∃ k : ℕ, k = 5 ∧ 
    let d := (1 : ℚ) / (2^n * 5^m) in 
    let s := d.repr.mantissa in -- Assuming a function repr.mantissa to represent decimal digits
    s.indexOf '1' - s.indexOf '.' - 1 = k := sorry

end count_zeros_decimal_representation_l196_196105


namespace jameson_badminton_medals_l196_196145

theorem jameson_badminton_medals :
  ∃ (b : ℕ),  (∀ (t s : ℕ), t = 5 → s = 2 * t → t + s + b = 20) ∧ b = 5 :=
by {
sorry
}

end jameson_badminton_medals_l196_196145


namespace angle_between_vectors_proof_l196_196510

noncomputable def angle_between_vectors (a b : ℝ) : Real :=
  if a + b = 0 then π else
  let cos_theta := -1 / 2
  Real.acos cos_theta

theorem angle_between_vectors_proof
  (a b : ℝ) 
  (ha : ‖a‖ = 2) 
  (hb : ‖b‖ = 2)
  (h_perp : b ⋅ (2 • a + b) = 0) : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end angle_between_vectors_proof_l196_196510


namespace sphere_radius_l196_196273

variable {a : ℝ}

def is_equal_triangles (K L M N : ℝ³) : Prop :=
  dist K L = dist K N ∧ dist L M = dist L N ∧ dist K M = dist K N

def common_side (KL : ℝ³) : Prop := KL ≠ 0

def angle_klm_ln (K L M N : ℝ³) : Prop := 
  let ∠KLM = π / 3
  let ∠LKN = π / 3
  true

def side_lengths (KL LM KN : ℝ) : Prop := 
  KL = a ∧ LM = 6 * a ∧ KN = 6 * a

def planes_perpendicular (planeKLM planeKLN : ℝ³ → ℝ) : Prop := 
  true

def sphere_touch_segments_midpoint (sphere : ℝ³ → ℝ) (LM_kn : ℝ³) : Prop :=
  true

theorem sphere_radius (K L M N : ℝ³) (KL LM KN : ℝ) (sphere : ℝ³ → ℝ) :
  is_equal_triangles K L M N →
  common_side KL → 
  angle_klm_ln K L M N →
  side_lengths KL LM KN → 
  planes_perpendicular K L M K L N → 
  sphere_touch_segments_midpoint sphere (LM + KN) →
  sphere.radius = a / 2 * sqrt (137 / 3) :=
sorry

end sphere_radius_l196_196273


namespace area_of_figure_M_l196_196576

def point := ℝ × ℝ

def satisfies_first_inequality (x y a b : ℝ) : Prop :=
  (x - a) ^ 2 + (y - b) ^ 2 ≤ 25

def satisfies_second_inequality (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 ≤ min (-8 * a - 6 * b) 25

def figure_M : set point :=
  {p | ∃ a b : ℝ, satisfies_first_inequality p.1 p.2 a b ∧ satisfies_second_inequality a b}

theorem area_of_figure_M : measure_theory.measure_area.figure_M = 75 * real.pi - 25 * real.sqrt 3 / 2 :=
sorry

end area_of_figure_M_l196_196576


namespace vertex_angle_of_identical_cones_l196_196665

theorem vertex_angle_of_identical_cones :
  ∀ (A : Point)
    (θ : ℝ) 
    (cones : Fin 4 Cone)
    (h_common_vertex : ∀ i, cone.vertex cones[i] = A)
    (h_identical : cone.vertex_angle cones[0] = cone.vertex_angle cones[1])
    (h_identical_angle : ∀ i, (i = 2 → cone.vertex_angle cones[i] = π / 4) ∧ (i = 3 → cone.vertex_angle cones[i] = 3 * π / 4))
    (h_tangent : ∀ i, ∀ j, (j ≠ i → tangent_external cones[i] cones[j]))
    (h_internal_tangent : ∀ i, (i ≤ 2 → tangent_internal cones[i] cones[3])), 
    cone.vertex_angle cones[0] = 2 * arctan (2 / 3) :=
by
  intros
  sorry

end vertex_angle_of_identical_cones_l196_196665


namespace value_of_a_add_b_l196_196027

theorem value_of_a_add_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
sorry

end value_of_a_add_b_l196_196027


namespace cos_seven_pi_over_six_l196_196842

theorem cos_seven_pi_over_six :
  Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
sorry

end cos_seven_pi_over_six_l196_196842


namespace inequality_a_b_c_d_l196_196520

theorem inequality_a_b_c_d 
  (a b c d : ℝ) 
  (h0 : 0 ≤ a) 
  (h1 : a ≤ b) 
  (h2 : b ≤ c) 
  (h3 : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
by
  sorry

end inequality_a_b_c_d_l196_196520


namespace percentage_increase_of_allowance_l196_196570

-- Define the allowances as described in the conditions
def middle_school_allowance := 8 + 2
def senior_year_allowance := (2 * middle_school_allowance) + 5

-- % increase function
def percentage_increase (old new : ℕ) : ℝ := ((new - old) / old.toReal) * 100

-- The theorem stating the proof problem
theorem percentage_increase_of_allowance : 
  let old := middle_school_allowance in
  let new := senior_year_allowance in
  percentage_increase old new = 150 := by 
  sorry

end percentage_increase_of_allowance_l196_196570


namespace total_fish_in_lake_l196_196869

theorem total_fish_in_lake :
  let w_ducks := 3
  let b_ducks := 7
  let m_ducks := 6
  let fish_per_white := 5
  let fish_per_black := 10
  let fish_per_multi := 12
  in w_ducks * fish_per_white + b_ducks * fish_per_black + m_ducks * fish_per_multi = 157 := by sorry

end total_fish_in_lake_l196_196869


namespace isosceles_triangle_perimeter_l196_196039

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l196_196039


namespace isosceles_triangle_perimeter_l196_196038

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l196_196038


namespace parabola_tangent_inclination_l196_196443

theorem parabola_tangent_inclination :
  ∀ (x y : ℝ), y = x^2 → M = (1/2, 1/4) → 
  let k := 2 * 1/2 in
  k = tan (real.pi / 4) := sorry

end parabola_tangent_inclination_l196_196443


namespace isosceles_triangle_FGH_l196_196221

-- Definitions for the conditions
variables {A B C H F G : Point}
variables (triangle_ABC : Triangle A B C)
variables (altitude_A : Line A H)
variables (altitude_C : Line C H)
variables (angle_bisector_B : Line B _)
variables (intersection_F : F ∈ altitude_A ∧ F ∈ angle_bisector_B)
variables (intersection_G : G ∈ altitude_C ∧ G ∈ angle_bisector_B)

-- The goal to prove
theorem isosceles_triangle_FGH 
  (acute_triangle: isAcuteTri triangle_ABC)
  (scalene_triangle: isScalene triangle_ABC)
  (altitudes_intersect: H = altitude_A ∩ altitude_C)
  (F_on_bisector: intersection_F)
  (G_on_bisector: intersection_G)
  : isIsoscelesTriangle (Triangle F G H) :=
by 
  sorry

end isosceles_triangle_FGH_l196_196221


namespace digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196295

def fractional_part_to_decimal (n d : ℕ) : ℚ := n / d

def ten_thousandths_place_digit (q : ℚ) : ℕ :=
  let decimal_str := (q * 10^5).round.to_string
  decimal_str.get_or_else (decimal_str.find_idx (>= ".") + 5) '0' - '0'

theorem digit_in_ten_thousandths_place_of_five_over_thirty_two :
  let q := fractional_part_to_decimal 5 32
  ten_thousandths_place_digit q = 2 :=
by
  -- The proof that computes the ten-thousandths place digit will be here.
  sorry

end digit_in_ten_thousandths_place_of_five_over_thirty_two_l196_196295


namespace man_speed_3_kmph_l196_196715

noncomputable def bullet_train_length : ℝ := 200 -- The length of the bullet train in meters
noncomputable def bullet_train_speed_kmph : ℝ := 69 -- The speed of the bullet train in km/h
noncomputable def time_to_pass_man : ℝ := 10 -- The time taken to pass the man in seconds
noncomputable def conversion_factor_kmph_to_mps : ℝ := 1000 / 3600 -- Conversion factor from km/h to m/s
noncomputable def bullet_train_speed_mps : ℝ := bullet_train_speed_kmph * conversion_factor_kmph_to_mps -- Speed of the bullet train in m/s
noncomputable def relative_speed : ℝ := bullet_train_length / time_to_pass_man -- Relative speed at which train passes the man
noncomputable def speed_of_man_mps : ℝ := relative_speed - bullet_train_speed_mps -- Speed of the man in m/s
noncomputable def conversion_factor_mps_to_kmph : ℝ := 3.6 -- Conversion factor from m/s to km/h
noncomputable def speed_of_man_kmph : ℝ := speed_of_man_mps * conversion_factor_mps_to_kmph -- Speed of the man in km/h

theorem man_speed_3_kmph :
  speed_of_man_kmph = 3 :=
by
  sorry

end man_speed_3_kmph_l196_196715


namespace Area_Triangle_MDA_l196_196122

variable (r : ℝ)
variable (O A B M D : Type)

-- Definition of the circle with center O and radius r
def Circle (O : Type) (r : ℝ) := sorry

-- Chord AB has length 2r
def Chord (A B : Type) (r : ℝ) := sorry

-- OM is perpendicular to AB at point M
def Perpendicular (O M : Type) (A B : Type) := sorry

-- MD is perpendicular to OA at point D
def PerpendicularM (M D : Type) (O A : Type) := sorry

theorem Area_Triangle_MDA
  (circle : Circle O r)
  (chord : Chord A B r)
  (perpendicular1 : Perpendicular O M A B)
  (perpendicular2: PerpendicularM M D O A) :
  area_triangle_MDA = r^2 / 4 :=
sorry

end Area_Triangle_MDA_l196_196122


namespace hotel_accommodation_l196_196349

theorem hotel_accommodation :
  ∃ (arrangements : ℕ), arrangements = 27 :=
by
  -- problem statement
  let triple_room := 1
  let double_room := 1
  let single_room := 1
  let adults := 3
  let children := 2
  
  -- use the given conditions and properties of combinations to calculate arrangements
  sorry

end hotel_accommodation_l196_196349


namespace equation_solutions_count_l196_196384

theorem equation_solutions_count :
  (Finset.univ.filter (λ x : ℕ, ∃ k : ℕ, k * k = x) 
    <| Finset.range 1 51).card = 43 := 
  sorry

end equation_solutions_count_l196_196384


namespace valid_triangle_constructions_l196_196693

theorem valid_triangle_constructions (c d e : ℕ) (h : |2 * c - e| < 3 * d ∧ 3 * d < 2 * c + e) :
  (({6, 4, 1} : set ℕ) = {c, d, e} → 
  ({c, d, e} = ({6, 4, 1} : set ℕ) → d ∈ {1} → (|2 * c - e| < 3 * d ∧ 3 * d < 2 * c + e) ∨ 
   (|2 * d - e| < 3 * c ∧ 3 * c < 2 * d + e)) →
  ((|2 * 6 - 1| < 3 * 4 ∧ 3 * 4 < 2 * 6 + 1) ∨
   (|2 * 4 - 6| < 3 * 1 ∧ 3 * 1 < 2 * 4 + 6)) → 
   2 := sorry

end valid_triangle_constructions_l196_196693


namespace Eugene_buys_four_t_shirts_l196_196541

noncomputable def t_shirt_price : ℝ := 20
noncomputable def pants_price : ℝ := 80
noncomputable def shoes_price : ℝ := 150
noncomputable def discount : ℝ := 0.10

noncomputable def discounted_t_shirt_price : ℝ := t_shirt_price - (t_shirt_price * discount)
noncomputable def discounted_pants_price : ℝ := pants_price - (pants_price * discount)
noncomputable def discounted_shoes_price : ℝ := shoes_price - (shoes_price * discount)

noncomputable def num_pants : ℝ := 3
noncomputable def num_shoes : ℝ := 2
noncomputable def total_paid : ℝ := 558

noncomputable def total_cost_of_pants_and_shoes : ℝ := (num_pants * discounted_pants_price) + (num_shoes * discounted_shoes_price)
noncomputable def remaining_cost_for_t_shirts : ℝ := total_paid - total_cost_of_pants_and_shoes

noncomputable def num_t_shirts : ℝ := remaining_cost_for_t_shirts / discounted_t_shirt_price

theorem Eugene_buys_four_t_shirts : num_t_shirts = 4 := by
  sorry

end Eugene_buys_four_t_shirts_l196_196541


namespace reciprocal_neg_one_six_abs_neg_six_l196_196649

/-- Let x be -1/6, then the reciprocal of x is -6. -/
theorem reciprocal_neg_one_six : let x := (-1 : ℚ) / 6 in x⁻¹ = -6 := by
  sorry

/-- The absolute value of -6 is 6. -/
theorem abs_neg_six : abs (-6 : ℤ) = 6 := by
  sorry

end reciprocal_neg_one_six_abs_neg_six_l196_196649


namespace values_of_x_l196_196506

def P (x : ℝ) : ℝ := x^3 - 5 * x^2 + 8 * x

theorem values_of_x (x : ℝ) :
  P x = P (x + 1) ↔ (x = 1 ∨ x = 4 / 3) :=
by sorry

end values_of_x_l196_196506


namespace set_equality_power_sum_l196_196035

theorem set_equality_power_sum (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) : a^2016 + b^2017 = 1 :=
sorry

end set_equality_power_sum_l196_196035


namespace frequency_of_a_is_3_l196_196134

def sentence : String := "Happy Teachers'Day!"

def frequency_of_a_in_sentence (s : String) : Nat :=
  s.foldl (λ acc c => if c = 'a' then acc + 1 else acc) 0

theorem frequency_of_a_is_3 : frequency_of_a_in_sentence sentence = 3 :=
  by
    sorry

end frequency_of_a_is_3_l196_196134


namespace committee_count_8_choose_4_l196_196346

theorem committee_count_8_choose_4 : (Nat.choose 8 4) = 70 :=
  by
  -- proof skipped
  sorry

end committee_count_8_choose_4_l196_196346


namespace smallest_non_consecutive_product_not_factor_of_48_l196_196272

def is_factor (a b : ℕ) : Prop := b % a = 0

def non_consecutive_pairs (x y : ℕ) : Prop := (x ≠ y) ∧ (x + 1 ≠ y) ∧ (y + 1 ≠ x)

theorem smallest_non_consecutive_product_not_factor_of_48 :
  ∃ x y, x ∣ 48 ∧ y ∣ 48 ∧ non_consecutive_pairs x y ∧ ¬ (x * y ∣ 48) ∧ (∀ x' y', x' ∣ 48 ∧ y' ∣ 48 ∧ non_consecutive_pairs x' y' ∧ ¬ (x' * y' ∣ 48) → x' * y' ≥ 18) :=
by
  sorry

end smallest_non_consecutive_product_not_factor_of_48_l196_196272


namespace parabola_equation_l196_196505

theorem parabola_equation (p : ℝ) (hp : 0 < p) (F : ℝ × ℝ) (Q : ℝ × ℝ) (PQ QF : ℝ)
  (hPQ : PQ = 8 / p) (hQF : QF = 8 / p + p / 2) (hDist : QF = 5 / 4 * PQ) : 
  ∃ x, y^2 = 4 * x :=
by
  sorry

end parabola_equation_l196_196505


namespace find_x_condition_l196_196110

theorem find_x_condition (x : ℝ) (h : 0.75 / x = 5 / 11) : x = 1.65 := 
by
  sorry

end find_x_condition_l196_196110


namespace part1_1_part1_2_part2_l196_196065

-- Given function definition
def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 
          x - x^2 
       else if 1 < x ∧ x ≤ 3 then 
          - (Real.sqrt 5 / 5) * f(x - 1)
       else 
          0

-- Part (I)
theorem part1_1 : f (5 / 2) = 1 / 20 := 
sorry

theorem part1_2 (x : ℝ) (hx : 2 ≤ x ∧ x ≤ 3) : f x = (1 / 5) * (x - 2) * (3 - x) := 
sorry

-- Part (II)
theorem part2 (k : ℝ) (h : ∀ x, 0 < x ∧ x ≤ 3 → f(x) ≤ k / x) : k = 0 := 
sorry

end part1_1_part1_2_part2_l196_196065


namespace symmetric_point_to_origin_l196_196631

theorem symmetric_point_to_origin (a b : ℝ) :
  (∃ (a b : ℝ), (a / 2) - 2 * (b / 2) + 2 = 0 ∧ (b / a) * (1 / 2) = -1) →
  (a = -4 / 5 ∧ b = 8 / 5) :=
sorry

end symmetric_point_to_origin_l196_196631


namespace parabola_passes_through_fixed_point_l196_196176

theorem parabola_passes_through_fixed_point:
  ∀ t : ℝ, ∃ x y : ℝ, (y = 4 * x^2 + 2 * t * x - 3 * t ∧ (x = 3 ∧ y = 36)) :=
by
  intro t
  use 3
  use 36
  sorry

end parabola_passes_through_fixed_point_l196_196176


namespace exercise_l196_196552

noncomputable def intersection_points_of_C1_and_C2 :
    (set.Point ℝ × set.Point ℝ) := sorry

theorem exercise :
    intersection_points_of_C1_and_C2 = ((2, -Real.pi / 6), (2, 7 * Real.pi / 6)) ∧ 
    (∀ θ : ℝ, (x : ℝ), (y : ℝ), ((x = 2 * Real.cos θ) ∧ (y = -2 + 2 * Real.sin θ)) -> 
    maximal_distance_from_curve_C2_to_line_l θ = 2 * Real.sqrt(2) + 2 :=
begin
    -- Proof is not required, only the statement
    sorry,
end

-- Definitions
def C1 (θ : ℝ) : ℝ := -1 / Real.sin θ

def C2 (θ : ℝ) : (ℝ × ℝ) := (2 * Real.cos θ, -2 + 2 * Real.sin θ)

def line_l (x y : ℝ) : Prop := (x - y + 2 = 0)

def maximal_distance_from_curve_C2_to_line_l (θ : ℝ) : ℝ := sorry

end exercise_l196_196552


namespace odd_integer_has_ab_l196_196609

def Q (x : ℤ) (a : ℤ) (b : ℤ) : ℤ :=
  (x + a)^2 + b

theorem odd_integer_has_ab (n : ℤ) (h : odd n) (h1 : n > 1) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ Nat.gcd a n.natAbs = 1 ∧ Nat.gcd b n.natAbs = 1 ∧
    Q 0 a b % n = 0 ∧ (∀ x, x > 0 → ∃ p : ℤ, p.prime ∧ p ∣ Q x a b ∧ p ∣ n → false) :=
by
  sorry

end odd_integer_has_ab_l196_196609


namespace find_m_value_l196_196117

noncomputable def complex_number_imaginary (m : ℂ) : Prop :=
    let i := complex.I in
    let z := (1 + m * i) * (2 - i) in
    ∀ real_part z = 0, Im(z) = z

theorem find_m_value (m : ℝ) (h : complex_number_imaginary m) : m = -2 :=
sorry

end find_m_value_l196_196117


namespace boys_brought_the_same_car_l196_196605

structure Car :=
  (size       : string)  -- e.g., "small", "big"
  (color      : string)  -- e.g., "green", "blue"
  (hasTrailer : bool)    -- true if the car has a trailer, false otherwise

def mishaCars : List Car :=
  [ { size := "small", color := "any", hasTrailer := true },
    { size := "small", color := "any", hasTrailer := false },
    { size := "any", color := "green", hasTrailer := false }]

def vityaCars : List Car :=
  [ { size := "any", color := "any", hasTrailer := false },
    { size := "small", color := "green", hasTrailer := true }]

def kolyaCars : List Car :=
  [ { size := "big", color := "any", hasTrailer := false },
    { size := "small", color := "blue", hasTrailer := true }]

theorem boys_brought_the_same_car :
  ∃ c : Car, c ∈ mishaCars ∧ c ∈ vityaCars ∧ c ∈ kolyaCars ∧ 
             c.size = "big" ∧ c.color = "green" ∧ c.hasTrailer = false :=
by
  sorry

end boys_brought_the_same_car_l196_196605


namespace problem_conditions_l196_196461

noncomputable def f (a b x : ℝ) : ℝ := abs (x + a) + abs (2 * x - b)

theorem problem_conditions (ha : 0 < a) (hb : 0 < b) 
  (hmin : ∃ x : ℝ, f a b x = 1) : 
  2 * a + b = 2 ∧ 
  ∀ (t : ℝ), (∀ a b : ℝ, 
    (0 < a) → (0 < b) → (a + 2 * b ≥ t * a * b)) → 
  t ≤ 9 / 2 :=
by
  sorry

end problem_conditions_l196_196461


namespace partI_partII_l196_196996

def pointM : ℝ × ℝ := (4 * sqrt 2 * Real.cos (π / 4), 4 * sqrt 2 * Real.sin (π / 4))
def curveC : ℝ → ℝ × ℝ := λ α => (1 + sqrt 2 * Real.cos α, sqrt 2 * Real.sin α)

theorem partI : ∃ (M : ℝ × ℝ), M = (4, 4) ∧ ∀ (t : ℝ), M = (4, 4) → (prod.snd M = prod.fst M) :=
by
  sorry

theorem partII : ∀ (M : ℝ × ℝ), M = (4, 4) → ∀ (A : ℝ × ℝ), A = (1, 0) → ∀ (r : ℝ), r = sqrt 2 →
  let distance := Real.sqrt ((prod.fst M - prod.fst A)^2 + (prod.snd M - prod.snd A)^2) - r
  distance = 5 - sqrt 2 :=
by
  sorry

end partI_partII_l196_196996


namespace consecutive_sum_l196_196249

theorem consecutive_sum (m k : ℕ) (h : (k + 1) * (2 * m + k) = 2000) :
  (m = 1000 ∧ k = 0) ∨ 
  (m = 198 ∧ k = 4) ∨ 
  (m = 28 ∧ k = 24) ∨ 
  (m = 55 ∧ k = 15) :=
by sorry

end consecutive_sum_l196_196249


namespace count_valid_programs_l196_196363

def list_courses : List String := ["English", "Algebra", "Geometry", "History", "Art", "Latin"]

def condition1 := ∃ (s : Set String), s ⊆ list_courses.to_finset ∧ s.card = 4
def condition2 (s : Set String) := "English" ∈ s
def condition3 (s : Set String) := ("Algebra" ∈ s ∨ "Geometry" ∈ s)

theorem count_valid_programs :
  ∃ (n : ℕ), n = 9 ∧ 
  (∀ s, s ⊆ list_courses.to_finset ∧ s.card = 4 ∧ "English" ∈ s ∧ ("Algebra" ∈ s ∨ "Geometry" ∈ s) → s.count() = n) :=
by
  sorry

end count_valid_programs_l196_196363


namespace repeating_decimal_fraction_l196_196449

theorem repeating_decimal_fraction : (real.mk (rat.mk_pnat (nat.succ 3 * (1 + 9 * 10)))) (nat.succ 27) = rat.mk 4 11 :=
by
-- proof can be filled here using Calc and necessary steps, but currently skipped
sorry

end repeating_decimal_fraction_l196_196449


namespace average_weight_correct_l196_196260

-- Define the number of men and women
def number_of_men : ℕ := 8
def number_of_women : ℕ := 6

-- Define the average weights of men and women
def average_weight_men : ℕ := 190
def average_weight_women : ℕ := 120

-- Define the total weight of men and women
def total_weight_men : ℕ := number_of_men * average_weight_men
def total_weight_women : ℕ := number_of_women * average_weight_women

-- Define the total number of individuals
def total_individuals : ℕ := number_of_men + number_of_women

-- Define the combined total weight
def total_weight : ℕ := total_weight_men + total_weight_women

-- Define the average weight of all individuals
def average_weight_all : ℕ := total_weight / total_individuals

theorem average_weight_correct :
  average_weight_all = 160 :=
  by sorry

end average_weight_correct_l196_196260


namespace sequence_sum_l196_196475

noncomputable def T_n (n : ℕ) : ℚ :=
  n / (4 * n + 4)

theorem sequence_sum (a : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ) (b : ℕ → ℚ) (T : ℕ → ℚ) (k : ℚ)
  (h1 : ∀ n, a n = 4 * n - 3)
  (h2 : ∀ n, S n = 2 * n ^ 2 - n)
  (h3 : k ≠ 0)
  (h4 : ∀ n, b n = S n / (n + k))
  (h5: ∃ k, ∀ n, ∃ d, (S (n+1) - S n = d))
  (h6 : ∀ n, T n = n / (4 * n + 4)) :
  T n = T_n n := by
  sorry

end sequence_sum_l196_196475


namespace red_candies_count_l196_196660

theorem red_candies_count :
  ∀ (total_candies blue_candies : ℕ),
  total_candies = 3409 → 
  blue_candies = 3264 →
  total_candies - blue_candies = 145 :=
by
  intros total_candies blue_candies h_total h_blue
  rw [h_total, h_blue]
  exact rfl

end red_candies_count_l196_196660


namespace collinearity_of_BER_l196_196378

theorem collinearity_of_BER
    (P B D C A Q R E : Point)
    (O : Circle)
    (h1 : IsTangent P B O)
    (h2 : IsTangent P D O)
    (h3 : IsSecant P C A O)
    (h4 : IsTangentAt C O R Q)
    (h5 : IntersectsAt A Q E O)
  : Collinear B E R := by
  sorry

end collinearity_of_BER_l196_196378


namespace part1_part2_l196_196068

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * x^2 - Real.log x

theorem part1 (a : ℝ) (h : 0 < a) (hf'1 : (1 - 2 * a * 1 - 1) = -2) :
  a = 1 ∧ (∀ x y : ℝ, y = -2 * (x - 1) → 2 * x + y - 2 = 0) :=
by
  sorry

theorem part2 {a : ℝ} (ha : a ≥ 1 / 8) :
  ∀ x : ℝ, (1 - 2 * a * x - 1 / x) ≤ 0 :=
by
  sorry

end part1_part2_l196_196068


namespace value_of_a_plus_b_l196_196022

theorem value_of_a_plus_b (a b : ℤ) (h1 : |a| = 1) (h2 : b = -2) :
  a + b = -1 ∨ a + b = -3 :=
sorry

end value_of_a_plus_b_l196_196022


namespace average_of_averages_is_6_l196_196147

-- Definitions for initial conditions
def jesse_first_3_days_distance := (2 / 3 : ℝ) * 3  -- miles
def jesse_day4_distance := 10  -- miles
def mia_4_days_distance := 3 * 4  -- miles
def total_distance := 30  -- miles

-- Derived total run after 4 days for Jesse and Mia
def jesse_total_4_days := jesse_first_3_days_distance + jesse_day4_distance
def mia_total_4_days := mia_4_days_distance

-- Remaining distances to be run in the final 3 days
def jesse_remaining_3_days := total_distance - jesse_total_4_days
def mia_remaining_3_days := total_distance - mia_total_4_days

-- Average distances per final 3 days
def jesse_average_3_days := jesse_remaining_3_days / 3
def mia_average_3_days := mia_remaining_3_days / 3

-- Theorem to prove: the average of their final 3 days averages is 6
theorem average_of_averages_is_6 : 
  (jesse_average_3_days + mia_average_3_days) / 2 = 6 := by
  -- We skip the actual proof with sorry
  sorry

end average_of_averages_is_6_l196_196147


namespace sum_of_f_l196_196172

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.cos (30 * Real.pi / 180 - x))

theorem sum_of_f :
  let degrees := (1:ℝ) : (59:ℝ→ _),
  (finset.range 59).sum (λ i, f ((i + 1) * Real.pi / 180)) = 59 * (Real.sqrt 3) / 2 :=
sorry

end sum_of_f_l196_196172


namespace apollonius_circle_equation_l196_196235

theorem apollonius_circle_equation (x y : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (8, 0))
  (h : dist (x, y) A / dist (x, y) B = 1 / 2) : x^2 + y^2 = 16 := 
sorry

end apollonius_circle_equation_l196_196235


namespace area_ratio_of_triangles_l196_196190

open_locale classical

variables {A B C D E M N : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space E] [metric_space M] [metric_space N]

-- Variables representing points in the plane
variables {A B C D E M N : point ℝ}

-- Definitions representing the given conditions
def is_median (A B C D : point ℝ) : Prop := is_midpoint (A D) ∧ is_midpoint (B D)
def is_centroid (A B C M : point ℝ) : Prop := centroid (triangle A B C) = M
def is_midpoint (X Y Z : point ℝ) : Prop := midpoint Y Z = midpoint X Z

-- Main theorem definition
theorem area_ratio_of_triangles {A B C D E M N : point ℝ} :
  (is_median A B C D) ∧ (is_median A B C E) ∧ (is_centroid A B C M) ∧
  (is_midpoint A E N) → 
  area_of_triangle M N E = (1 : ℝ) / 12 * area_of_triangle A B C :=
begin
  sorry
end

end area_ratio_of_triangles_l196_196190


namespace at_least_one_expression_is_leq_neg_two_l196_196583

variable (a b c : ℝ)

theorem at_least_one_expression_is_leq_neg_two 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1 / b ≤ -2) ∨ (b + 1 / c ≤ -2) ∨ (c + 1 / a ≤ -2) :=
sorry

end at_least_one_expression_is_leq_neg_two_l196_196583


namespace boys_brought_the_same_car_l196_196606

structure Car :=
  (size       : string)  -- e.g., "small", "big"
  (color      : string)  -- e.g., "green", "blue"
  (hasTrailer : bool)    -- true if the car has a trailer, false otherwise

def mishaCars : List Car :=
  [ { size := "small", color := "any", hasTrailer := true },
    { size := "small", color := "any", hasTrailer := false },
    { size := "any", color := "green", hasTrailer := false }]

def vityaCars : List Car :=
  [ { size := "any", color := "any", hasTrailer := false },
    { size := "small", color := "green", hasTrailer := true }]

def kolyaCars : List Car :=
  [ { size := "big", color := "any", hasTrailer := false },
    { size := "small", color := "blue", hasTrailer := true }]

theorem boys_brought_the_same_car :
  ∃ c : Car, c ∈ mishaCars ∧ c ∈ vityaCars ∧ c ∈ kolyaCars ∧ 
             c.size = "big" ∧ c.color = "green" ∧ c.hasTrailer = false :=
by
  sorry

end boys_brought_the_same_car_l196_196606


namespace max_x_y_l196_196002

theorem max_x_y (x y : ℝ) (h : x^2 + y^2 - 3 * y - 1 = 0) : 
  (∃ θ : ℝ, x = (√13 / 2) * Real.cos θ ∧ y = (√13 / 2) * Real.sin θ + 3/2) →
  x + y ≤ (3 + √26) / 2 := 
by
  intro hθ
  sorry

end max_x_y_l196_196002


namespace probability_all_black_after_rotation_l196_196712

theorem probability_all_black_after_rotation :
  let p := 1/2 in
  let center_black_prob := p^4 in
  let pair_black_prob := (p * p)^8 in
  center_black_prob * pair_black_prob = 1/1048576 :=
sorry

end probability_all_black_after_rotation_l196_196712


namespace right_handed_players_l196_196259

-- Define preliminary data and conditions
def total_players : ℕ := 120
def throwers : ℕ := 55
def fraction_left_handed_non_throwers : ℚ := 2 / 5

-- Use sorry to skip the proof part
theorem right_handed_players :
  let non_throwers := total_players - throwers in
  let left_handed_non_throwers := (fraction_left_handed_non_throwers * non_throwers) in
  let right_handed_non_throwers := non_throwers - left_handed_non_throwers in
  let total_right_handed := throwers + right_handed_non_throwers in
  total_right_handed = 94 :=
by
  sorry

end right_handed_players_l196_196259


namespace intersection_sum_zero_l196_196646

theorem intersection_sum_zero :
  let p1 := λ x : ℝ, (x - 2) * (x - 2)
  let p2 := λ y : ℝ, ((y + 2) * (y + 2)) - 7
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ), 
    (y1 = p1 x1) ∧ (x1 + 7 = (y1 + 2) * (y1 + 2)) ∧ 
    (y2 = p1 x2) ∧ (x2 + 7 = (y2 + 2) * (y2 + 2)) ∧ 
    (y3 = p1 x3) ∧ (x3 + 7 = (y3 + 2) * (y3 + 2)) ∧ 
    (y4 = p1 x4) ∧ (x4 + 7 = (y4 + 2) * (y4 + 2)) ∧ 
    (x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 0) := by
    sorry

end intersection_sum_zero_l196_196646


namespace remainder_of_greatest_multiple_of_16_with_unique_digits_div_by_1000_l196_196167

noncomputable def is_unique_digits (n : ℕ) : Prop :=
  let digits := List.dedup (Nat.digits 10 n)
  digits.length = Nat.digits 10 n |>.length

noncomputable def greatest_integer_multiple_of_16_with_unique_digits : ℕ :=
  let candidates := List.filter is_unique_digits (List.range (10000))
  List.maximum (List.filter (λ x, x % 16 == 0) candidates) |>.get_or_else 0

theorem remainder_of_greatest_multiple_of_16_with_unique_digits_div_by_1000 :
  let M := greatest_integer_multiple_of_16_with_unique_digits
  M % 1000 = 864 := by
  sorry

end remainder_of_greatest_multiple_of_16_with_unique_digits_div_by_1000_l196_196167


namespace probability_of_picking_letter_from_mathematics_l196_196112

-- Definition of the problem conditions
def extended_alphabet_size := 30
def distinct_letters_in_mathematics := 8

-- Theorem statement
theorem probability_of_picking_letter_from_mathematics :
  (distinct_letters_in_mathematics / extended_alphabet_size : ℚ) = 4 / 15 := 
by 
  sorry

end probability_of_picking_letter_from_mathematics_l196_196112


namespace socks_combination_correct_l196_196617

noncomputable def socks_combination : ℕ :=
nat.choose 6 4

theorem socks_combination_correct : socks_combination = 15 :=
by
  sorry

end socks_combination_correct_l196_196617


namespace center_of_circle_l196_196630

theorem center_of_circle :
  ∀ (x y: ℝ), x^2 + y^2 - 4 * x + 8 * y + 5 = 0 → (2, -4) := sorry

end center_of_circle_l196_196630


namespace digit_in_ten_thousandths_place_of_fraction_is_two_l196_196305

theorem digit_in_ten_thousandths_place_of_fraction_is_two :
  (∃ d : ℕ, d = (Int.floor (5 / 32 * 10^4) % 10) ∧ d = 2) :=
by
  sorry

end digit_in_ten_thousandths_place_of_fraction_is_two_l196_196305


namespace final_milk_volume_l196_196370

def initial_milk_volume : ℝ := 30
def removed_volume : ℝ := 9
def replace_milk_with_water (milk : ℝ) (volume : ℝ) : ℝ := milk - volume
def remaining_milk_after_first_removal (initial : ℝ) (removed : ℝ) : ℝ := initial - removed
def ratio_milk_to_total (milk : ℝ) (total : ℝ) : ℝ := milk / total
def milk_in_removed_mixture (ratio : ℝ) (removed : ℝ) : ℝ := ratio * removed
def milk_left_after_second_removal (remaining : ℝ) (removed_milk : ℝ) : ℝ := remaining - removed_milk

theorem final_milk_volume :
  let initial := initial_milk_volume,
      removed := removed_volume,
      first_removal := replace_milk_with_water initial removed,
      remaining_milk_first := remaining_milk_after_first_removal initial removed,
      ratio := ratio_milk_to_total remaining_milk_first initial,
      milk_removed := milk_in_removed_mixture ratio removed,
      final_milk := milk_left_after_second_removal remaining_milk_first milk_removed
  in final_milk = 14.7 :=
by
  sorry

end final_milk_volume_l196_196370


namespace construct_chord_with_three_equal_segments_l196_196467

-- Given a circle with chords AB and AC of equal length, we aim to prove the existence
-- of a chord FG which is divided into three equal segments by AB and AC.

section ThreeEqualSegments
-- Assume we have a circle with center O and radius r.
variables {O : Type} {r : ℝ} 
-- Assume we have two chords AB and AC with given equal lengths.
variables {A B C : Type} [MetricSpace A O] [MetricSpace B O] [MetricSpace C O]
          (h_eq_length : dist A B = dist A C)

noncomputable def exists_three_equal_segments_chord : Prop :=
  ∃ (F G : A), (dist F G / 3) = dist F (line_through (A,B) ∩ line_through (A, C)) ∧
               (dist F G / 3) = dist G (line_through (A,B) ∩ line_through (A, C)) ∧
               (dist F G / 3) > 0 

theorem construct_chord_with_three_equal_segments :
  exists_three_equal_segments_chord :=
sorry

end ThreeEqualSegments

end construct_chord_with_three_equal_segments_l196_196467


namespace log_five_fraction_l196_196836

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196836


namespace simplify_expression_l196_196209

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^3 + 2 * b^2) - 2 * b^2 + 5 = 9 * b^4 + 6 * b^3 - 2 * b^2 + 5 := sorry

end simplify_expression_l196_196209


namespace largest_k_divides_N_l196_196711

theorem largest_k_divides_N :
  let N := (Nat.factorial 9) * ((Nat.factorial 6) ^ 3) in
  ∃ (k : ℕ), (2 ^ k ∣ N ∧ (∀ m : ℕ, (2 ^ m ∣ N) → m ≤ k)) ∧ k = 19 :=
by
  sorry

end largest_k_divides_N_l196_196711


namespace evaluate_floor_ceiling_sum_l196_196436

theorem evaluate_floor_ceiling_sum : 
  (Int.floor 1.99) + (Int.ceil 3.02) = 5 := 
by
  sorry

end evaluate_floor_ceiling_sum_l196_196436


namespace sequence_twice_square_l196_196703

theorem sequence_twice_square (n : ℕ) (a : ℕ → ℕ) :
    (∀ i : ℕ, a i = 0) →
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
        ∀ i : ℕ, i % (2 * m) = 0 → 
            a i = if a i = 0 then 1 else 0) →
    (∀ i : ℕ, a i = 1 ↔ ∃ k : ℕ, i = 2 * k^2) :=
by
  sorry

end sequence_twice_square_l196_196703


namespace circle_and_tangent_l196_196551

-- Define points in Cartesian Coordinate System
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define circle through three points and tangent line through a fourth point
def circle_through_three_points (A B C : Point) : Prop :=
∃ a b r,
(a^2 + (b - A.y)^2 = r^2) ∧
((a - B.x)^2 + (b - B.y)^2 = r^2) ∧
((a - C.x)^2 + (b - C.y)^2 = r^2) ∧
((x - a)^2 + (y - b)^2 = r^2)

def tangent_line_through_point (M : Point → Prop) (D : Point) : Prop :=
∃ m c, M (λ P, P.x = m * P.y + c) ∧
P((2 * D.x + D.y = 0))

axiom A : Point := ⟨0, 1⟩
axiom B : Point := ⟨2, 1⟩
axiom C : Point := ⟨3, 4⟩
axiom D : Point := ⟨-1, 2⟩

theorem circle_and_tangent :
(circle_through_three_points A B C) ∧
(tangent_line_through_point (λ P, (P.x - 1)^2 + (P.y - 3)^2 = 5) D)
:=
by
sory

end circle_and_tangent_l196_196551


namespace sum_of_sequence_l196_196650

-- Definitions and conditions
def sequence_x (a : ℝ) : ℕ → ℝ
| 1     := 1
| 2     := a
| (n + 1) := if (∃ k, (n + 1) = 2^k) then a * (sequence_x a (n + 1 - 2^nat.log2(n + 1))) else sequence_x a 1 -- This sequence definition may need further adjustment to ensure it captures periodicity correctly

def S (a : ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range (n + 1), sequence_x a i

-- Statement
theorem sum_of_sequence (a : ℝ) (n : ℕ): 
  ∀ k : ℕ, k ∈ (2^nat.log2(n) :: finset.range (nat.log2 (n))) → 
  S n = ∑ j in finset.range (nat.log2 n + 1), a ^ j * (1 + a) ^ (nat.bit1 (nat.log2 n).val) := sorry

end sum_of_sequence_l196_196650


namespace problem_l196_196078

theorem problem (a b : ℝ) (h_nonzero : a ≠ 0) (h_nonone : a ≠ 1) 
  (h_set_eq : {1, a, b / a} = {0, a^2, a + b}) : (a + b) ^ 2023 = -1 := 
by
  sorry

end problem_l196_196078


namespace euclidean_algorithm_steps_l196_196859

theorem euclidean_algorithm_steps (a b : ℕ) (ha : a = 360) (hb : b = 504) : 
  ∃ n, n = 3 ∧ 
  let gcd_alg : ℕ → ℕ → ℕ × ℕ := λ a b, if b = 0 then (a, 0) else (b, a % b) in
  let steps : ℕ → ℕ → ℕ := λ a b, 
    if b = 0 then 0 else 1 + steps b (a % b) in
  steps a b = n := 
sorry

end euclidean_algorithm_steps_l196_196859


namespace condition_for_a_pow_zero_eq_one_l196_196628

theorem condition_for_a_pow_zero_eq_one (a : Real) : a ≠ 0 ↔ a^0 = 1 :=
by
  sorry

end condition_for_a_pow_zero_eq_one_l196_196628


namespace arithmetic_sequence_ninth_term_l196_196250

variable {α : Type*} [Add α] [Mul α] [HasOne α] [One α] [Sub α]

def a (n : ℕ) (a₁ : α) (d : α) : α := a₁ + n * d
def S (n : ℕ) (a₁ : α) (d : α) : α := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_ninth_term :
  ∃ (a₁ d : α), a 4 a₁ d = 8 ∧ S 3 a₁ d = 6 ∧ a 8 a₁ d = 16 :=
sorry

end arithmetic_sequence_ninth_term_l196_196250


namespace probability_six_on_final_roll_l196_196741

theorem probability_six_on_final_roll (n : ℕ) (h : n ≥ 2019) :
  (∃ p : ℚ, p > 5 / 6 ∧ 
  (∀ roll : ℕ, roll <= n → roll mod 6 = 0 → roll / n > p)) :=
sorry

end probability_six_on_final_roll_l196_196741


namespace systematic_sampling_sequence_l196_196464

theorem systematic_sampling_sequence :
  ∃ k : ℕ, ∃ b : ℕ, (∀ n : ℕ, n < 6 → (3 + n * k = b + n * 10)) ∧ (b = 3 ∨ b = 13 ∨ b = 23 ∨ b = 33 ∨ b = 43 ∨ b = 53) :=
sorry

end systematic_sampling_sequence_l196_196464


namespace cone_dimensions_l196_196339

noncomputable def cone_height (r_sector : ℝ) (r_cone_base : ℝ) : ℝ :=
  Real.sqrt (r_sector^2 - r_cone_base^2)

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * radius^2 * height

theorem cone_dimensions 
  (r_circle : ℝ) (num_sectors : ℕ) (r_cone_base : ℝ) :
  r_circle = 12 → num_sectors = 4 → r_cone_base = 3 → 
  cone_height r_circle r_cone_base = 3 * Real.sqrt 15 ∧ 
  cone_volume r_cone_base (cone_height r_circle r_cone_base) = 9 * Real.pi * Real.sqrt 15 :=
by
  intros
  sorry

end cone_dimensions_l196_196339


namespace purple_chips_selected_is_one_l196_196537

noncomputable def chips_selected (B G P R x : ℕ) : Prop :=
  (1^B) * (5^G) * (x^P) * (11^R) = 140800 ∧ 5 < x ∧ x < 11

theorem purple_chips_selected_is_one :
  ∃ B G P R x, chips_selected B G P R x ∧ P = 1 :=
by {
  sorry
}

end purple_chips_selected_is_one_l196_196537


namespace intersect_complement_l196_196965

open Finset

-- Define the universal set U, set A, and set B
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def A := {1, 3, 6} : Finset ℕ
def B := {2, 3, 4} : Finset ℕ

-- Define the complement of B in U
def complement_U_B := U \ B

-- The statement to prove
theorem intersect_complement : A ∩ complement_U_B = {1, 6} :=
by sorry

end intersect_complement_l196_196965


namespace regression_line_is_correct_l196_196128

theorem regression_line_is_correct :
  ∃ (m b : ℝ), (∀ (x y : ℝ), ((x,y) ∈ {(1,3), (2,3.8), (3,5.2), (4,6)}) → y = m * x + b) ∧ (m = 1.04) ∧ (b = 1.9) :=
by {
  let points := [(1, 3), (2, 3.8), (3, 5.2), (4, 6)],
  let mean_x := (1 + 2 + 3 + 4) / 4,
  let mean_y := (3 + 3.8 + 5.2 + 6) / 4,
  existsi (1.04 : ℝ), existsi (1.9 : ℝ),
  refine ⟨_, rfl, rfl⟩,
  intros x y h,
  cases h;
  simp [mean_x, mean_y],
  exact sorry -- Proof part to be carried out here if needed
}

end regression_line_is_correct_l196_196128


namespace range_of_x_when_m_is_4_range_of_m_l196_196466

-- Define the conditions for p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0
def neg_p (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 5
def neg_q (x m : ℝ) : Prop := x ≤ m ∨ x ≥ 3 * m

-- Define the conditions for the values of m
def cond_m_pos (m : ℝ) : Prop := m > 0
def cond_sufficient (m : ℝ) : Prop := cond_m_pos m ∧ m ≤ 2 ∧ 3 * m ≥ 5

-- Problem 1
theorem range_of_x_when_m_is_4 (x : ℝ) : p x ∧ q x 4 → 4 < x ∧ x < 5 :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_q x m → neg_p x) → 5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_x_when_m_is_4_range_of_m_l196_196466


namespace fewer_cans_l196_196323

theorem fewer_cans (sarah_yesterday lara_more alex_yesterday sarah_today lara_today alex_today : ℝ)
  (H1 : sarah_yesterday = 50.5)
  (H2 : lara_more = 30.3)
  (H3 : alex_yesterday = 90.2)
  (H4 : sarah_today = 40.7)
  (H5 : lara_today = 70.5)
  (H6 : alex_today = 55.3) :
  (sarah_yesterday + (sarah_yesterday + lara_more) + alex_yesterday) - (sarah_today + lara_today + alex_today) = 55 :=
by {
  -- Sorry to skip the proof
  sorry
}

end fewer_cans_l196_196323


namespace contractor_absent_days_l196_196328

variable (x y : ℕ)  -- Number of days worked and absent, both are natural numbers

-- Conditions from the problem
def total_days (x y : ℕ) : Prop := x + y = 30
def total_payment (x y : ℕ) : Prop := 25 * x - 75 * y / 10 = 360

-- Main statement
theorem contractor_absent_days (h1 : total_days x y) (h2 : total_payment x y) : y = 12 :=
by
  sorry

end contractor_absent_days_l196_196328


namespace sqrt_of_16_is_4_l196_196405

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_of_16_is_4_l196_196405


namespace find_angle_B_find_perimeter_l196_196895

variables (a b A B C : ℝ)
noncomputable def vector_m : ℝ × ℝ := (-real.cos (A / 2), real.sin (A / 2))
noncomputable def vector_n : ℝ × ℝ := (real.cos (A / 2), real.sin (A / 2))
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Condition on dot product
axiom dot_product_condition : dot_product (vector_m A) (vector_n A) = 0.5
-- Condition: sqrt(2)*a = sqrt(3)*b
axiom relation_a_b : sqrt 2 * a = sqrt 3 * b

-- Question 1: Prove B = π/4 given the conditions
theorem find_angle_B (h1 : dot_product (vector_m A) (vector_n A) = 0.5) 
(h2 : sqrt 2 * a = sqrt 3 * b) : B = π / 4 := 
sorry

-- Additional variables and conditions for second question
variables (c area : ℝ)
axiom side_a : a = 2 * sqrt 3
axiom area_condition : area = sqrt 3
axiom area_relation : sqrt 3 = 0.5 * b * c * (real.sin (2 * π / 3))
axiom bc_relation : b * c = 4
axiom sides_sum : (b + c) ^ 2 = 16

-- Question 2: Prove the perimeter of the triangle given the conditions
theorem find_perimeter (h3 : a = 2 * sqrt 3) 
(h4 : area = sqrt 3) (h5 : b * c = 4) 
(h6 : (b + c) ^ 2 = 16) : a + b + c = 4 + 2 * sqrt 3 :=
sorry

end find_angle_B_find_perimeter_l196_196895


namespace find_b_range_l196_196077

def A (x : ℝ) : Prop := log (x + 2) / log (1/2) < 0
def B (x a b : ℝ) : Prop := (x - a) * (x - b) < 0

theorem find_b_range (a : ℝ) (b : ℝ) (h : a = -3) : 
  (∃ x, A x ∧ B x a b) → b > -1 :=
sorry

end find_b_range_l196_196077


namespace log_b_1024_number_of_positive_integers_b_l196_196514

theorem log_b_1024 (b : ℕ) : (∃ n : ℕ, b^n = 1024) ↔ b ∈ {2, 4, 32, 1024} :=
by sorry

theorem number_of_positive_integers_b : (∃ b : ℕ, ∃ n : ℕ, b^n = 1024 ∧ n > 0) ↔ 4 :=
by {
  have h := log_b_1024,
  sorry
}

end log_b_1024_number_of_positive_integers_b_l196_196514


namespace smallest_sum_of_three_diff_numbers_l196_196806

def numbers_set : set ℤ := {0, 10, -4, 2, -6}

theorem smallest_sum_of_three_diff_numbers :
  ∃ a b c ∈ numbers_set, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -10) :=
by
  sorry

end smallest_sum_of_three_diff_numbers_l196_196806


namespace enclosing_polygons_of_dodecagon_l196_196354

theorem enclosing_polygons_of_dodecagon (n : ℕ) :
  (∃ P : ℕ → ℕ, P 12 = n ∧ 
  (∀ k : ℕ, k = 12 → 
    let interior_angle_dodecagon := (10 * 180 : ℚ) / 12 in
    let exterior_angle_dodecagon := 180 - interior_angle_dodecagon in
    let exterior_angle_n_polygon := 360 / n in
    2 * exterior_angle_n_polygon = exterior_angle_dodecagon)) → n = 12 :=
by
  sorry

end enclosing_polygons_of_dodecagon_l196_196354


namespace inequality_solution_set_l196_196652

theorem inequality_solution_set (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end inequality_solution_set_l196_196652


namespace area_of_ABCD_l196_196226

/-- Define the quadrilateral and geometrical constraints given in the problem. -/
structure Quadrilateral :=
  (A B C D P : Point)
  (CD : ℝ)  /-- Length of side CD -/
  (a b p : ℝ) /-- Distances from points A, B, and P to the line CD -/

/-- Area of the quadrilateral ABCD -/
noncomputable def area_of_quadrilateral (quad : Quadrilateral) : ℝ :=
  (quad.a * quad.b * quad.CD) / (2 * quad.p)

theorem area_of_ABCD (quad : Quadrilateral) : 
  quad.area = (quad.a * quad.b * quad.CD) / (2 * quad.p) := sorry

end area_of_ABCD_l196_196226


namespace minimum_distance_l196_196550

theorem minimum_distance (a : ℝ) (ha_pos : a ≠ 0) : 
  xy_eq_sqrt3: (a * (sqrt 3 / a) = sqrt 3) → 
  minimum_distance: (dist (a, sqrt 3 / a) l = sqrt 3) :=
begin
  sorry,
end

end minimum_distance_l196_196550


namespace equal_segments_YX_ZX_l196_196563

/-- Given triangle KIA, point O is the midpoint of the median from K to IA, 
point Y is the foot of the perpendicular from I to the bisector of angle IOK, 
point Z is the foot of the perpendicular from A to the bisector of angle AOK, 
and point X is the intersection of KO and YZ. Prove that YX = ZX. -/
theorem equal_segments_YX_ZX
  (K I A O Y Z X : Type)
  [triangle K I A]
  (mid_O : is_midpoint O K I A)
  (perp_Y : is_perpendicular Y I (angle_bisector I O K))
  (perp_Z : is_perpendicular Z A (angle_bisector A O K))
  (int_X : is_intersection X (line_segment K O) (line_segment Y Z)) :
  distance Y X = distance Z X := sorry

end equal_segments_YX_ZX_l196_196563


namespace cylinder_volumes_difference_l196_196793

-- Define the radius and volume for Charlie's cylinder
def charlie_radius := 5 / Real.pi
def charlie_height := 12
def charlie_volume := Real.pi * (charlie_radius ^ 2) * charlie_height

-- Define the radius and volume for Dana's cylinder
def dana_radius := 6 / Real.pi
def dana_height := 10
def dana_volume := Real.pi * (dana_radius ^ 2) * dana_height

-- Define the absolute difference of the volumes
def volume_difference := abs (dana_volume - charlie_volume)

-- Define the desired result to prove
theorem cylinder_volumes_difference :
  Real.pi * volume_difference = 60 := by
  -- Here we would continue with the proof steps
  sorry

end cylinder_volumes_difference_l196_196793


namespace find_natural_numbers_l196_196845

theorem find_natural_numbers (n : ℕ) :
  (∀ (a : Fin (n + 2) → ℝ), (a (Fin.last _) * (a (Fin.last _))
   - 2 * (a (Fin.last _)) * Real.sqrt (Finset.univ.sum (λ i, (a i) ^ 2)) 
   + (Finset.univ.erase (Fin.last _)).sum (λ i, a i) = 0) → 
   (a (Fin.last _) ≠ 0) → 
   ∃ x : ℝ, (a (Fin.last _) * x^2
   - 2 * x * Real.sqrt (Finset.univ.sum (λ i, (a i) ^ 2))
   + (Finset.univ.erase (Fin.last _)).sum (λ i, a i) = 0)) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
sorry

end find_natural_numbers_l196_196845


namespace problem_1_problem_2_l196_196468

-- Problem 1:
theorem problem_1 (g : ℝ → ℝ) (x : ℝ) (hx_pos : 0 < x ∨ x < -2 ∨ x > 2) :
  g(x + x⁻¹) = x^2 + x⁻² → ∀ y, (y = x + x⁻¹ → y ≥ 2 ∨ y ≤ -2 → g(y) = y^2 - 2) := by
  sorry

-- Problem 2:
theorem problem_2 (h : ℝ → ℝ) (f : ℝ → ℝ) :
  (∀ x, h(x) = (4*x^2 - 12*x - 3) / (2*x + 1))
  → (∀ t, t = 2*x + 1 → 1 ≤ t ∧ t ≤ 3 → f(t) = t + 4/t - 8)
  → ((∀ y, y ∈ [1, 2] → f(y) ≥ f(2))
     ∧ (∀ y, y ∈ [2, 3] → f(y) ≥ f(2))
     ∧ f(1) = -3 ∧ f(2) = -4 ∧ f(3) = -11/3)
  → (range (λ x : ℝ, h(x)) ∩ set.Icc 0 1 = set.Icc (-4) (-3)) := by
  sorry

-- Note: The theorems include assumptions on conditions and respective properties derived from the problem.

end problem_1_problem_2_l196_196468


namespace find_x_l196_196622

variable (x : ℝ)

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x : delta (phi x) = 23 → x = -1 / 6 := by
  intro h
  sorry

end find_x_l196_196622


namespace total_time_spent_l196_196144

variable (B I E M EE ST ME : ℝ)

def learn_basic_rules : ℝ := B
def learn_intermediate_level : ℝ := I
def learn_expert_level : ℝ := E
def learn_master_level : ℝ := M
def endgame_exercises : ℝ := EE
def middle_game_strategy_tactics : ℝ := ST
def mentoring : ℝ := ME

theorem total_time_spent :
  B = 2 →
  I = 75 * B →
  E = 50 * (B + I) →
  M = 30 * E →
  EE = 0.25 * I →
  ST = 2 * EE →
  ME = 0.5 * E →
  B + I + E + M + EE + ST + ME = 235664.5 :=
by
  intros hB hI hE hM hEE hST hME
  rw [hB, hI, hE, hM, hEE, hST, hME]
  sorry

end total_time_spent_l196_196144


namespace expected_score_of_basketball_player_l196_196338

theorem expected_score_of_basketball_player :
  let p_inside : ℝ := 0.7
  let p_outside : ℝ := 0.4
  let attempts_inside : ℕ := 10
  let attempts_outside : ℕ := 5
  let points_inside : ℕ := 2
  let points_outside : ℕ := 3
  let E_inside : ℝ := attempts_inside * p_inside * points_inside
  let E_outside : ℝ := attempts_outside * p_outside * points_outside
  E_inside + E_outside = 20 :=
by
  sorry

end expected_score_of_basketball_player_l196_196338


namespace real_part_of_z_2008_l196_196648

-- Define the complex number z
def z : ℂ := 1 - I

-- Prove that the real part of z ^ 2008 is 2 ^ 1004
theorem real_part_of_z_2008 : (z ^ 2008).re = 2 ^ 1004 :=
by
  sorry

end real_part_of_z_2008_l196_196648


namespace proof_1_proof_2_l196_196894

noncomputable def problem_1 (a b c : ℝ) (A B C : ℝ) (h1 : (√2 * a - b) / c = cos B / cos C) : Prop :=
  C = π / 4

noncomputable def problem_2 (f g : ℝ → ℝ) (h2 : f = λ x, cos (2 * x + π / 4))
 (h3 : g = λ x, cos (2 * x - π / 4)) (I : set.Icc 0 (π / 3)) : set.Icc (√6 - √2 / 4) 1 :=
  set.image g I

theorem proof_1 (a b c A B C : ℝ) (h1 : (√2 * a - b) / c = cos B / cos C) : C = π / 4 :=
  sorry

theorem proof_2 (x : ℝ) (f g : ℝ → ℝ) (h2 : f = λ x, cos (2 * x + π / 4))
 (h3 : g = λ x, cos (2 * x - π / 4)) (I : set.Icc 0 (π / 3)) (y : ℝ) (hy : y ∈ set.image g I) : 
 I ∈ set.Icc (√6 - √2 / 4) 1 :=
  sorry

end proof_1_proof_2_l196_196894


namespace slope_of_line_with_sine_of_angle_l196_196754

theorem slope_of_line_with_sine_of_angle (α : ℝ) 
  (hα₁ : 0 ≤ α) (hα₂ : α < Real.pi) 
  (h_sin : Real.sin α = Real.sqrt 3 / 2) : 
  ∃ k : ℝ, k = Real.tan α ∧ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end slope_of_line_with_sine_of_angle_l196_196754


namespace angle_BAE_is_22_5_degrees_l196_196532

-- Define the conditions of the problem
variables (A B C D E : Point)
variables (CA CB : LineSegment)
variables (BC : LineSegment)

-- Given conditions
def is_isosceles_triangle (A B C : Point) : Prop :=
  CA = CB

def is_rhombus (B C D E : Point) : Prop :=
  BC = CD ∧ CD = DE ∧ DE = EB ∧ (BD.is_perpendicular_to DE)

-- The proof problem
theorem angle_BAE_is_22_5_degrees
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_rhombus B C D E)
  : angle B A E = 22.5 :=
sorry

end angle_BAE_is_22_5_degrees_l196_196532


namespace log_increasing_condition_log_increasing_not_necessary_l196_196335

theorem log_increasing_condition (a : ℝ) (h : a > 2) : a > 1 :=
by sorry

theorem log_increasing_not_necessary (a : ℝ) : ∃ b, (b > 1 ∧ ¬(b > 2)) :=
by sorry

end log_increasing_condition_log_increasing_not_necessary_l196_196335


namespace room_ratio_calculations_l196_196765

theorem room_ratio_calculations (length width : ℝ) (h_length : length = 20.5) (h_width : width = 12.3) :
  let perimeter_feet := 2 * (length + width)
  let ratio_feet := length / perimeter_feet
  let perimeter_yards := perimeter_feet / 3
  let ratio_yards := length / perimeter_yards
  ratio_feet = 20.5 / 65.6 ∧ ratio_yards = 20.5 / 21.8667 :=
by
  rw [h_length, h_width]
  have h1 : perimeter_feet = 2 * (20.5 + 12.3) := by norm_num
  have h2 : ratio_feet = 20.5 / perimeter_feet := by norm_num
  have h3 : ratio_feet = 20.5 / 65.6 := by norm_num [h1]
  have h4 : perimeter_yards = 65.6 / 3 := by norm_num [h1]
  have h5 : ratio_yards = 20.5 / perimeter_yards := by norm_num
  have h6 : ratio_yards = 20.5 / 21.8667 := by norm_num [h4]
  exact ⟨h3, h6⟩

end room_ratio_calculations_l196_196765


namespace perfect_cubes_between_50_and_1000_l196_196941

theorem perfect_cubes_between_50_and_1000 : 
  {n : ℕ | 50 ≤ n^3 ∧ n^3 ≤ 1000}.finite.toFinset.card = 7 := 
by
  sorry

end perfect_cubes_between_50_and_1000_l196_196941


namespace coordinates_of_complex_number_l196_196988

theorem coordinates_of_complex_number : ∃ x y : ℝ, (i * (2 - i) = x + y * i) ∧ (x = 1 ∧ y = 2) := by
  use 1, 2
  split
  { simp [Complex.ext_iff] }
  { exact And.intro rfl rfl }

end coordinates_of_complex_number_l196_196988


namespace median_angle_equality_l196_196558

-- Definitions and variables
variables {A B C K L M N : Point} -- Points A, B, C, K, L, M, N
variables [Triangle Α B C] -- Triangle ABC

-- Conditions
def is_median (P Q R : Point) (M : Point) := M ∈ segment P Q ∧ (distance M P = distance M Q)
def is_centroid (G : Point) {P Q R : Point} (mPQ mPR mQR : Point) :=
  G = intersection (line mPQ) (line mPR) ∧ G = intersection (line mPQ) (line mQR)

def lies_on_circumcircle (C : Point) (P Q R : Point) :=
  circle_circumcenter C P Q R

-- Equivalent proof problem
theorem median_angle_equality
  {A B C K L M N : Point}
  [triangle ABC : Triangle A B C]
  (h_medians : is_median A L C L ∧ is_median B M A M)
  (h_centroid : K = centroid A B C)
  (h_circumcircle : lies_on_circumcircle C K L M):
  angle B C H = angle N A K :=
sorry

end median_angle_equality_l196_196558


namespace distance_PF_l196_196488

-- Definitions of the parabola, points, and properties
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def directrix_x : ℝ := -1
def point_on_parabola (P : ℝ × ℝ) : Prop := ∃ y0, P = (y0^2 / 4, y0)
def perpendicular (P A F : ℝ × ℝ) : Prop := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PF := (F.1 - P.1, F.2 - P.2)
  PA.1 * PF.1 + PA.2 * PF.2 = 0

-- Given conditions
variables {F : ℝ × ℝ} (A P : ℝ × ℝ)
  (hF : F = focus)
  (hA : A = (directrix_x, 0))
  (hP : point_on_parabola P)
  (hPerpendicular : perpendicular P A F)

-- Statement to prove
theorem distance_PF :
  let PF := (F.1 - P.1)^2 + (F.2 - P.2)^2
  sqrt PF = sqrt(5) - 1 :=
by
  sorry

end distance_PF_l196_196488


namespace graph_y_eq_ffx_points_ab_cd_l196_196220

-- Define the function and its values
def f : ℕ → ℕ :=
  λ x, 
    if x = 1 then 5 
    else if x = 2 then 3 
    else if x = 3 then 1 
    else 0 -- default value for other inputs not needed

-- Define the points on the graph of y = f(f(x))
def point_a := (2, f(f 2))
def point_b := (3, f(f 3))

-- Proof statement for Lean
theorem graph_y_eq_ffx_points_ab_cd :
  let a := point_a.1
  let b := point_a.2
  let c := point_b.1
  let d := point_b.2
  a * b + c * d = 17 :=
by
  simp [point_a, point_b, f]
  simp [f]
  sorry

end graph_y_eq_ffx_points_ab_cd_l196_196220


namespace max_value_set_l196_196503

noncomputable def given_function (x : ℝ) : ℝ :=
  (1 / 2) * (cos x)^2 + (sqrt 3 / 2) * (sin x) * (cos x) + 1

theorem max_value_set : 
  {x : ℝ | ∃ k : ℤ, x = (k : ℝ) * π + π / 3} = {x | (1 / 2) * (cos x)^2 + (sqrt 3 / 2) * (sin x) * (cos x) + 1 = 5 / 4} :=
sorry

end max_value_set_l196_196503


namespace probability_fav_song_not_fully_played_l196_196727

theorem probability_fav_song_not_fully_played :
  let song_lengths := List.range 12 |>.map (λ n => 40 * (n + 1))
  let fav_song_idx := 7 -- index of the favourite song (8th song)
  60 * 6 = 360 -- total seconds in 6 minutes
  fav_song_length = 300 -- length of the favourite song in seconds (5 minutes)
  num_songs := 12
  in song_lengths.nth fav_song_idx = some fav_song_length →
      (1 - (1 / (12 * real.to_rat (num_songs.factorial)) *
        ((num_songs - 1).factorial + 3 * (num_songs - 2).factorial))) = 43 / 48 :=
by sorry

end probability_fav_song_not_fully_played_l196_196727


namespace bc_gt_ad_l196_196014

open Real

theorem bc_gt_ad (a b c d : ℝ) (H1 : ab > 0) (H2 : - (c / a) < - (d / b)) : bc > ad := by
  sorry

end bc_gt_ad_l196_196014


namespace average_weight_men_women_l196_196267

theorem average_weight_men_women (n_men n_women : ℕ) (avg_weight_men avg_weight_women : ℚ)
  (h_men : n_men = 8) (h_women : n_women = 6) (h_avg_weight_men : avg_weight_men = 190)
  (h_avg_weight_women : avg_weight_women = 120) :
  (n_men * avg_weight_men + n_women * avg_weight_women) / (n_men + n_women) = 160 := 
by
  sorry

end average_weight_men_women_l196_196267


namespace find_c_f_monotonic_on_interval_g_is_odd_l196_196493

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (c * x - 1) / (x + 1)
def f1 (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def g (x : ℝ) : ℝ := f1 (Real.exp x)

-- Problem 1: Prove that c = 1 given f(1) = 0
theorem find_c (c : ℝ) : f c 1 = 0 → c = 1 := by
  sorry

-- Problem 2: Prove that f(x) = (x-1)/(x+1) is monotonically increasing on [0, 2]
theorem f_monotonic_on_interval : MonotoneOn f1 (Set.Icc 0 2) := by
  sorry

-- Problem 3: Prove that g(x) = f(e^x) is odd
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end find_c_f_monotonic_on_interval_g_is_odd_l196_196493


namespace no_54_after_one_minute_l196_196602

theorem no_54_after_one_minute :
  let initial := 12
  let operations := [2, 3]
  let time := 60
  ∀ operations_performed : List ℕ,
    (∀ op ∈ operations_performed, (op = 2 ∨ op = 3 ∨ op = 1/2 ∨ op = 1/3)) →
    (operations_performed.length = time) →
    (List.foldl (λ acc op, acc * (if op > 1 then op else 1/op)) initial operations_performed ≠ 54) :=
by
  -- proof goes here
  sorry

end no_54_after_one_minute_l196_196602


namespace closest_whole_area_of_shaded_region_l196_196720

theorem closest_whole_area_of_shaded_region :
  let d := 1 in
  let r := d / 2 in
  let area_rectangle := 2 * 3 in
  let area_circle := π * r^2 in
  let area_shaded := area_rectangle - area_circle in
  (6 - π / 4 : ℝ).round = 5 := by
  sorry

end closest_whole_area_of_shaded_region_l196_196720


namespace investment_amount_is_correct_l196_196782

-- Definition of the conditions
def monthly_interest : ℕ := 231
def annual_interest_rate : ℝ := 0.09
def I_annual : ℕ := monthly_interest * 12 -- Annual interest from monthly interest payments

-- Definition of the principal amount calculation
def principal_amount (I : ℕ) (r : ℝ) (t : ℝ) : ℝ := I / (r * t)

-- The target amount of the investment
def target_principal_amount : ℕ := 30800

-- The theorem stating the amount of the investment
theorem investment_amount_is_correct : principal_amount I_annual annual_interest_rate 1 = target_principal_amount := by
  sorry

end investment_amount_is_correct_l196_196782


namespace root_complex_solution_l196_196944

theorem root_complex_solution (a b : ℝ) (h : (1 - 2 * complex.I) * (1 - 2 * complex.I) + a * (1 - 2 * complex.I) + b = 0) : 
  a = -2 ∧ b = 5 :=
by 
  -- Prove that a = -2 and b = 5 from the given condition
  sorry

end root_complex_solution_l196_196944


namespace min_value_exists_max_value_exists_l196_196431

noncomputable def y (x : ℝ) : ℝ := 3 - 4 * Real.sin x - 4 * (Real.cos x)^2

theorem min_value_exists :
  (∃ k : ℤ, y (π / 6 + 2 * k * π) = -2) ∧ (∃ k : ℤ, y (5 * π / 6 + 2 * k * π) = -2) :=
by 
  sorry

theorem max_value_exists :
  ∃ k : ℤ, y (-π / 2 + 2 * k * π) = 7 :=
by 
  sorry

end min_value_exists_max_value_exists_l196_196431


namespace burger_share_per_person_l196_196274

-- Definitions based on conditions
def foot_to_inches : ℕ := 12
def burger_length_foot : ℕ := 1
def burger_length_inches : ℕ := burger_length_foot * foot_to_inches

theorem burger_share_per_person : (burger_length_inches / 2) = 6 := by
  sorry

end burger_share_per_person_l196_196274


namespace log_five_fraction_l196_196837

theorem log_five_fraction : log 5 (1 / real.sqrt 5) = -1/2 := by
  sorry

end log_five_fraction_l196_196837


namespace amy_red_balloons_l196_196780

theorem amy_red_balloons (total_balloons green_balloons blue_balloons : ℕ) (h₁ : total_balloons = 67) (h₂: green_balloons = 17) (h₃ : blue_balloons = 21) : (total_balloons - (green_balloons + blue_balloons)) = 29 :=
by
  sorry

end amy_red_balloons_l196_196780


namespace problem1_problem2_l196_196707

-- Problem 1
theorem problem1 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x > 1) :
  let y := x + 4 / (x - 1)
  in y ≥ 5 ∧ (y = 5 → x = 3) :=
sorry

end problem1_problem2_l196_196707


namespace value_of_x_squared_minus_y_squared_l196_196521

theorem value_of_x_squared_minus_y_squared (x y : ℚ)
  (h1 : x + y = 8 / 15)
  (h2 : x - y = 2 / 15) :
  x^2 - y^2 = 16 / 225 := by
  sorry

end value_of_x_squared_minus_y_squared_l196_196521


namespace cross_country_meet_winning_scores_l196_196976

theorem cross_country_meet_winning_scores :
  ∃ (scores : Finset ℕ), scores.card = 13 ∧
    ∀ s ∈ scores, s ≥ 15 ∧ s ≤ 27 :=
by
  sorry

end cross_country_meet_winning_scores_l196_196976


namespace grid_diagonal_numbers_l196_196841

theorem grid_diagonal_numbers (n : ℕ) (hn : Odd n)
  (G : Matrix (Fin n) (Fin n) ℕ)
  (hG1 : ∀ i, Multiset.ofFn (λ j, G i j) = Finset.univ ∧ Multiset.ofFn (λ j, G j i) = Finset.univ)
  (hG2 : ∀ i j, G i j = G j i) :
  ∀ k : ℕ, k ∈ Finset.range n → ∃ i, G i i = k :=
  sorry

end grid_diagonal_numbers_l196_196841


namespace triangle_perimeter_correct_l196_196454

variable {α : ℝ} (R r : ℝ)

noncomputable def perimeter_triangle (α R r : ℝ) : ℝ :=
  2 * (r * Real.cot (α / 2) + 2 * R * Real.sin α)

theorem triangle_perimeter_correct :
  ∀ α R r : ℝ, perimeter_triangle α R r = 2 * (r * Real.cot (α / 2) + 2 * R * Real.sin α) :=
by 
  intros
  sorry

end triangle_perimeter_correct_l196_196454


namespace shaded_region_area_l196_196989

-- Define the distances between points (given conditions)
def AB : ℝ := 3
def BC : ℝ := 4
def CD : ℝ := 4
def DE : ℝ := 4
def EF : ℝ := 5
def AF : ℝ := AB + BC + CD + DE + EF

-- Function to calculate the area of a semicircle based on its diameter
def semicircle_area (d : ℝ) : ℝ := (π * (d^2)) / 8

-- Sum of semicircles' areas
def small_semicircles_area : ℝ :=
  semicircle_area AB + 3 * semicircle_area BC + semicircle_area EF

-- Area of the larger semicircle with diameter AF
def large_semicircle_area : ℝ := semicircle_area AF

-- Proof statement: Shaded area is 43.75 * π
theorem shaded_region_area : large_semicircle_area - small_semicircles_area = 43.75 * π :=
by
  -- Insert the mathematical proof here (omitted)
  sorry

end shaded_region_area_l196_196989


namespace prob1_prob2_prob3_l196_196480

structure Point2D where
  x : ℝ
  y : ℝ

def origin : Point2D := ⟨0, 0⟩

def distance_point_line (P : Point2D) (a b c : ℝ) : ℝ :=
  abs (a * P.x + b * P.y + c) / math.sqrt (a * a + b * b)

def max_distance_point_line (P : Point2D) (a b c : ℝ) : ℝ :=
  abs (a * P.x + b * P.y + c) / math.sqrt (a * a + b * b)

theorem prob1 (P : Point2D) (hP : P = ⟨2, -1⟩) :
  (∃ a b c : ℝ, distance_point_line P a b c = 2 ∧ 
    (a * 2 + b * (-1) + c = 0)) :=
sorry

theorem prob2 (P : Point2D) (hP : P = ⟨2, -1⟩) :
  (∃ a b c : ℝ, max_distance_point_line P a b c = math.sqrt 5 ∧ 
    (a * 2 + b * (-1) + c = 0)) :=
sorry

theorem prob3 (P : Point2D) (hP : P = ⟨2, -1⟩) :
  ¬(∃ a b c : ℝ, distance_point_line P a b c = 6 ∧ 
    (a * 2 + b * (-1) + c = 0)) :=
sorry

end prob1_prob2_prob3_l196_196480


namespace stratified_sampling_students_l196_196356

-- Define the total number of students in each grade
def grade10_students : ℕ := 150
def grade11_students : ℕ := 180
def grade12_students : ℕ := 210

-- Define the total number of students to be selected
def total_selected : ℕ := 72

-- Define the total number of students
def total_students : ℕ := grade10_students + grade11_students + grade12_students

-- Calculate the selection probability
def selection_probability : ℚ := total_selected / total_students

-- Calculate the total number of students in grades 10 and 11
def grade10_and_11_students : ℕ := grade10_students + grade11_students

-- Define the theorem statement
theorem stratified_sampling_students : 
  (grade10_and_11_students * selection_probability).natAbs = 44 :=
by
  -- The selection probability must be in the rational number field
  have h : (grade10_and_11_students : ℚ) * selection_probability = (330 : ℕ) * (2 / 15) := by
    sorry
  exact Nat.eq_of_gcd_eq _ _ (by
    have : 330 * (2 / 15) = 44 := by
      linarith
    rwa [Nat.cast_self, Nat.cast_two, Nat.cast_mul, Nat.cast_div] at this)

end stratified_sampling_students_l196_196356


namespace digit_in_ten_thousandths_place_l196_196307

theorem digit_in_ten_thousandths_place (n : ℕ) (hn : n = 5) : 
    (decimal_expansion 5/32).get (4) = some 2 :=
by
  sorry

end digit_in_ten_thousandths_place_l196_196307


namespace proposition_4_l196_196170

variables {Line Plane : Type}
variables {a b : Line} {α β : Plane}

-- Definitions of parallel and perpendicular relationships
class Parallel (l : Line) (p : Plane) : Prop
class Perpendicular (l : Line) (p : Plane) : Prop
class Contains (p : Plane) (l : Line) : Prop

theorem proposition_4
  (h1: Perpendicular a β)
  (h2: Parallel a b)
  (h3: Contains α b) : Perpendicular α β :=
sorry

end proposition_4_l196_196170


namespace a_100_gt_14_l196_196417

noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+1) := a n + 1 / a n

theorem a_100_gt_14 : 14 < a 100 :=
by {
  sorry
}

end a_100_gt_14_l196_196417
