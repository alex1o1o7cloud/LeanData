import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.ArithmeticSeries
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Log
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Ring
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.EuclideanSpace
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Matching
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Digit
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.EuclideanSpace.Basic

namespace Liam_contribution_l575_575576

theorem Liam_contribution 
  (pastry_cost_euros : ℝ) 
  (yumi_yen : ℝ) 
  (exchange_rate : ℝ) 
  (liam_contribution : ℝ) :
  pastry_cost_euros = 8 → 
  yumi_yen = 1000 → 
  exchange_rate = 140 → 
  liam_contribution = (pastry_cost_euros - yumi_yen / exchange_rate) → 
  liam_contribution = 0.86 :=
by
  intros h_cost h_yen h_rate h_liam
  rw [h_cost, h_yen, h_rate] at h_liam
  exact h_liam

end Liam_contribution_l575_575576


namespace initial_num_files_l575_575413

-- Define the conditions: number of files organized in the morning, files to organize in the afternoon, and missing files.
def num_files_organized_in_morning (X : ℕ) : ℕ := X / 2
def num_files_to_organize_in_afternoon : ℕ := 15
def num_files_missing : ℕ := 15

-- Theorem to prove the initial number of files is 60.
theorem initial_num_files (X : ℕ) 
  (h1 : num_files_organized_in_morning X = X / 2)
  (h2 : num_files_to_organize_in_afternoon = 15)
  (h3 : num_files_missing = 15) :
  X = 60 :=
by
  sorry

end initial_num_files_l575_575413


namespace pants_and_tshirts_l575_575655

noncomputable theory

variables (P T : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750
def condition3 : Prop := 8 * T = 400

-- Prove that P = 450 and T = 50 given the conditions
theorem pants_and_tshirts (h1 : condition1 P T) (h2 : condition2 P T) (h3 : condition3 P T) : 
  P = 450 ∧ T = 50 :=
by
  -- Proof will be provided here
  sorry

end pants_and_tshirts_l575_575655


namespace simplify_trig_identity_l575_575171

theorem simplify_trig_identity (x y : ℝ) : 
    sin (x + y) * sin (x - y) - cos (x + y) * cos (x - y) = -cos (2 * x) := 
sorry

end simplify_trig_identity_l575_575171


namespace distinct_remainders_permutation_l575_575998

theorem distinct_remainders_permutation (n : ℕ) (hn : n.prime) :
  ∃ (a : Fin n → Fin n), ∀ (i j : Fin n), i ≠ j → 
    ∃ (k : Fin (n+1)), (finset.univ.image (λ k : Fin (n+1), 
    (list.prod (list.map (λ i, a i) ((list.fin_range (k+1)).map coe)) : ℕ) % n)).nodup := 
sorry

end distinct_remainders_permutation_l575_575998


namespace sequence_integers_l575_575662

theorem sequence_integers 
    (a : ℕ → ℝ) 
    (h : ∀ k, 1 ≤ k → k ≤ 2015 → (∑ i in finset.range k, (a i)^3) = (∑ i in finset.range k, a i)^2) 
    : ∀ k, 1 ≤ k → k ≤ 2015 → ∃ n : ℤ, a k = n := 
sorry

end sequence_integers_l575_575662


namespace sequence_general_term_l575_575215

-- Define the sequence based on the given conditions
def seq (n : ℕ) : ℚ := if n = 0 then 1 else (n : ℚ) / (2 * n - 1)

theorem sequence_general_term (n : ℕ) :
  seq (n + 1) = (n + 1) / (2 * (n + 1) - 1) :=
by
  sorry

end sequence_general_term_l575_575215


namespace distance_from_B_to_orthocenter_l575_575544

theorem distance_from_B_to_orthocenter (A B C : Point) (hAB : dist A B = 2) (hAC : dist A C = 5) (hBC : dist B C = 6) :
  dist B (orthocenter A B C) = 50 / sqrt 39 := 
sorry

end distance_from_B_to_orthocenter_l575_575544


namespace cos_A_value_l575_575971

-- Conditions given in the problem
variables {a b c : ℝ} -- sides opposite to ∠A, ∠B, and ∠C
variables {A B C : ℝ} -- angles of the triangle
variables (h1 : (√3 * b - c) * real.cos A = a * real.cos C)

-- The theorem to prove
theorem cos_A_value : real.cos A = (√3 / 3) :=
begin
  -- Placeholder for proof
  sorry
end

end cos_A_value_l575_575971


namespace three_digit_numbers_with_product_30_l575_575497

theorem three_digit_numbers_with_product_30 : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
   d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d1 * d2 * d3 = 30) ↔
  12 := 
sorry

end three_digit_numbers_with_product_30_l575_575497


namespace temperature_difference_l575_575147

theorem temperature_difference (T_south T_north : ℤ) (h1 : T_south = -7) (h2 : T_north = -15) :
  T_south - T_north = 8 :=
by
  sorry

end temperature_difference_l575_575147


namespace union_set_correct_l575_575017

-- Define the sets
noncomputable def M := { x : ℝ | x^2 - x - 2 > 0 }
noncomputable def N := { x : ℝ | log (x + 2) + log (1 - x) = some (log x) }

-- Define the complement of M in R
noncomputable def complement_M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

-- The union of N and the complement of M
noncomputable def union_N_complement_M := { x : ℝ | (-2 < x ∧ x < 1) ∨ (-1 ≤ x ∧ x ≤ 2) }

-- The final statement to prove
theorem union_set_correct :
  union_N_complement_M = { x : ℝ | -2 < x ∧ x ≤ 2 } :=
sorry

end union_set_correct_l575_575017


namespace identical_numbers_in_geometric_progression_set_l575_575682

theorem identical_numbers_in_geometric_progression_set (n : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 4 * n → 0 < a i) 
  (h2 : ∀ (i j k l : ℕ), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → 
    (a j) / (a i) = (a k) / (a j) ∧ (a k) / (a j) = (a l) / (a k)) : 
  ∃ m, (∀ i, 1 ≤ i ∧ i ≤ n → a (4 * m) = a (4 * i)) :=
begin
  sorry
end

end identical_numbers_in_geometric_progression_set_l575_575682


namespace product_of_sequence_is_243_l575_575333

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l575_575333


namespace song_arrangement_l575_575763

/-- The total number of ways to arrange five different songs such that:
  1) No song is liked by all three girls (Amy, Beth, Jo),
  2) For each of the three pairs of the girls, there is at least one song liked by those two girls but disliked by the third,
  3) At least one song is liked only by one of the girls,
  is exactly 261. -/
theorem song_arrangement :
  ∃ (s: Finset (Fin 5)), 
    (∀ song, song ∉ s) → 
    (∀ (pair: Finset (Fin 3)), pair.card = 2 → ∃ song, song ∉ s) → 
    (∃! song, song ∈ s) → 
    s.card = 261 :=
sorry

end song_arrangement_l575_575763


namespace simplify_expression_l575_575167

theorem simplify_expression : 
  (cos (40 * real.pi / 180)) / (cos (25 * real.pi / 180) * sqrt(1 - sin (40 * real.pi / 180))) = sqrt 2 :=
sorry

end simplify_expression_l575_575167


namespace find_k_l575_575740

theorem find_k (k : ℚ) :
  (∃ (P Q : ℚ × ℚ), 
    P = (1, -7) ∧ 
    Q = (k, 19) ∧ 
    (λ P Q, (19 - (-7)) / (k - 1) = -3 / 4) P Q ∧ 
    ∃ (a b c : ℚ), 
      a = 3 ∧ 
      b = 4 ∧ 
      c = 12 ∧ 
      ∀ (x y : ℚ), a * x + b * y = c → y = -3 / 4 * x + 3) → 
  k = -101 / 3 := 
sorry

end find_k_l575_575740


namespace geometric_seq_not_sufficient_necessary_l575_575560

theorem geometric_seq_not_sufficient_necessary (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a_n (n+1) = a_n n * q) : 
  ¬ ((∃ q > 1, ∀ n, a_n (n+1) > a_n n) ∧ (∀ q > 1, ∀ n, a_n (n+1) > a_n n)) := 
sorry

end geometric_seq_not_sufficient_necessary_l575_575560


namespace circle_intersection_unique_point_l575_575533

theorem circle_intersection_unique_point (k : ℝ) :
  (∃! z : ℂ, |z - 4| = 3 * |z + 2| ∧ |z| = k) ↔ k = 5.5 := by
  sorry

end circle_intersection_unique_point_l575_575533


namespace good_number_constant_exists_number_of_bad_numbers_l575_575941

def is_good_number (n p q : ℕ) : Prop :=
  ∃ (x y : ℕ), n = p * x + q * y

theorem good_number_constant_exists (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] :
  ∃ c, ∀ (n : ℕ), (is_good_number n p q ∧ ¬ is_good_number (c - n) p q) ∨ (¬ is_good_number n p q ∧ is_good_number (c - n) p q) :=
sorry

theorem number_of_bad_numbers (p q : ℕ) [Fact (Nat.Prime p)] [Fact (Nat.Prime q)] :
  ∃ c, c = p * q - p - q ∧ 
  let bad_numbers := { n | n ≤ c ∧ ¬ is_good_number n p q } in
  bad_numbers.to_finset.card = (p - 1) * (q - 1) / 2 :=
sorry

end good_number_constant_exists_number_of_bad_numbers_l575_575941


namespace percentage_difference_and_commission_value_l575_575306

def income_of_C : ℝ := 100
def income_of_A : ℝ := 1.2 * income_of_C
def salary_of_B : ℝ := 1.25 * income_of_A
def commission_of_B : ℝ := 0.05 * (income_of_A + income_of_C)
def total_income_of_B : ℝ := salary_of_B + commission_of_B
def income_of_D : ℝ := 0.85 * total_income_of_B
def income_of_E : ℝ := 1.1 * income_of_C
def income_of_F : ℝ := (total_income_of_B + income_of_E) / 2
def combined_income : ℝ := income_of_A + income_of_C + income_of_D + income_of_E + income_of_F

theorem percentage_difference_and_commission_value :
  (total_income_of_B - combined_income) / combined_income * 100 = -73.27 ∧ |commission_of_B| = 11 :=
  by
  sorry

end percentage_difference_and_commission_value_l575_575306


namespace nonzero_even_exists_from_step_2_l575_575536

def initial_sequence (i : ℤ) : ℤ :=
  if i = 0 then 1 else 0

def update_sequence (seq : ℤ → ℤ) (i : ℤ) : ℤ :=
  seq (i - 1) + seq (i + 1)

def n_step_sequence (n : ℕ) (i : ℤ) : ℤ :=
  nat.rec_on n
    (initial_sequence i)
    (λ n' seq, update_sequence seq i)

theorem nonzero_even_exists_from_step_2 :
  ∀ (n : ℕ), ∃ i, 2 ≤ n ∧ n_step_sequence n i % 2 = 0 ∧ n_step_sequence n i ≠ 0 :=
begin
  sorry
end

end nonzero_even_exists_from_step_2_l575_575536


namespace value_of_expression_l575_575292

def g (x : ℝ) (p q r s t : ℝ) : ℝ :=
  p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_of_expression (p q r s t : ℝ) (h : g (-1) p q r s t = 4) :
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 :=
sorry

end value_of_expression_l575_575292


namespace find_b_from_extreme_value_l575_575481

theorem find_b_from_extreme_value :
  ∃ b c : ℝ, ∀ f : ℝ → ℝ,
    (f = λ x, - (1/3) * x^3 + b * x^2 + c * x + b * c) →
    (f 1 = -(4/3)) →
    (∀ f' : ℝ → ℝ, f' = λ x, - x^2 + 2 * b * x + c → f'(1) = 0) →
    b = -1 :=
by
  sorry

end find_b_from_extreme_value_l575_575481


namespace mangoes_harvested_l575_575137

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) (total_mangoes_distributed : ℕ) (total_mangoes : ℕ) :
  neighbors = 8 ∧ mangoes_per_neighbor = 35 ∧ total_mangoes_distributed = neighbors * mangoes_per_neighbor ∧ total_mangoes = 2 * total_mangoes_distributed →
  total_mangoes = 560 :=
by {
  sorry
}

end mangoes_harvested_l575_575137


namespace log_2025_lt_A_lt_log_2026_l575_575370

def f (n : ℕ) : ℝ :=
if n = 3 then log 3 
else log (n + f (n - 1))

theorem log_2025_lt_A_lt_log_2026 :
  let A := f 2022
  in log 2025 < A ∧ A < log 2026 :=
by
  let A := f 2022
  sorry

end log_2025_lt_A_lt_log_2026_l575_575370


namespace james_winnings_l575_575102

variables (W : ℝ)

theorem james_winnings :
  (W / 2 - 2 = 55) → W = 114 :=
by
  intros h,
  sorry

end james_winnings_l575_575102


namespace solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l575_575257

open Real

theorem solve_diff_eq_for_k_ne_zero (k : ℝ) (h : k ≠ 0) (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x * (f x + g x) ^ k)
  (hg : ∀ x, deriv g x = f x * (f x + g x) ^ k)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) + (1 - k * x) ^ (1 / k)) ∧ g x = (1 / 2) * ((1 / (1 - k * x)) ^ (1 / k) - (1 - k * x) ^ (1 / k))) :=
sorry

theorem solve_diff_eq_for_k_eq_zero (f g : ℝ → ℝ) 
  (hf : ∀ x, deriv f x = g x)
  (hg : ∀ x, deriv g x = f x)
  (hf0 : f 0 = 1)
  (hg0 : g 0 = 0) :
  (∀ x, f x = cosh x ∧ g x = sinh x) :=
sorry

end solve_diff_eq_for_k_ne_zero_solve_diff_eq_for_k_eq_zero_l575_575257


namespace people_at_first_table_l575_575211

theorem people_at_first_table (N x : ℕ) 
  (h1 : 20 < N) 
  (h2 : N < 50)
  (h3 : (N - x) % 42 = 0)
  (h4 : N % 8 = 7) : 
  x = 5 :=
sorry

end people_at_first_table_l575_575211


namespace binomial_12_6_eq_924_l575_575783

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575783


namespace max_visible_unit_cubes_from_single_point_l575_575726

theorem max_visible_unit_cubes_from_single_point (n : ℕ) (h_cube_dim : n = 12) :
    let total_faces := 3 * (n * n),
        edges_intersect := 3 * n - 2,
        max_visible := total_faces - edges_intersect
    in max_visible = 398 :=
by
    let total_faces := 3 * (12 * 12);
    let edges_intersect := 3 * 12 - 2;
    let max_visible := total_faces - edges_intersect;
    have : max_visible = 398 := by norm_num;
    exact this

#eval max_visible_unit_cubes_from_single_point 12 (rfl)

end max_visible_unit_cubes_from_single_point_l575_575726


namespace balls_in_jar_l575_575684

theorem balls_in_jar (total_balls initial_blue_balls balls_after_taking_out : ℕ) (probability_blue : ℚ) :
  initial_blue_balls = 6 →
  balls_after_taking_out = initial_blue_balls - 3 →
  probability_blue = 1 / 5 →
  (balls_after_taking_out : ℚ) / (total_balls - 3 : ℚ) = probability_blue →
  total_balls = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end balls_in_jar_l575_575684


namespace number_of_valid_pairs_l575_575627

theorem number_of_valid_pairs :
  ∃ (n : Nat), n = 8 ∧ 
  (∃ (a b : Int), 4 < a ∧ a < b ∧ b < 22 ∧ (4 + a + b + 22) / 4 = 13) :=
sorry

end number_of_valid_pairs_l575_575627


namespace parabola_tangent_mul_min_l575_575935

def parabola (p : ℝ) : set (ℝ × ℝ) := {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def on_line (M : ℝ × ℝ) : Prop := M.1 - M.2 - 2 = 0

def tangents (M : ℝ × ℝ) (p : ℝ) (C : set (ℝ × ℝ)) : Prop :=
∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ (A ≠ B) ∧ 
  (∃ x1 x2: ℝ, M ∈ {(x, (1 / (2 * p)) * x^2) | x = x1 ∨ x = x2})

def tangents_product_min (AF BF : ℝ) : Prop :=
∀ p > 0, @min ℝ _ (abs (AF * BF)) = 8

theorem parabola_tangent_mul_min (p : ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ) :
  ∀ AF BF, parabola p ∧ on_line M ∧ tangents M p (parabola p) 
  → (tangents_product_min AF BF → p = 4) := 
sorry

end parabola_tangent_mul_min_l575_575935


namespace problem_function_f_half_problem_function_ff_three_l575_575044

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x else 2 * f (x - 1)

theorem problem_function_f_half :
  f (3 / 2) = 1 :=
by
  sorry

theorem problem_function_ff_three :
  f (f 3) = 8 :=
by
  sorry

end problem_function_f_half_problem_function_ff_three_l575_575044


namespace inequalities_hold_l575_575865

theorem inequalities_hold (b : ℝ) :
  (b ∈ Set.Ioo (-(1 : ℝ) - Real.sqrt 2 / 4) (0 : ℝ) ∨ b < -(1 : ℝ) - Real.sqrt 2 / 4) →
  (∀ x y : ℝ, 2 * b * Real.cos (2 * (x - y)) + 8 * b^2 * Real.cos (x - y) + 8 * b^2 * (b + 1) + 5 * b < 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 + 1 > 2 * b * x + 2 * y + b - b^2) :=
by 
  intro h
  sorry

end inequalities_hold_l575_575865


namespace binary_110011_to_decimal_l575_575395

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575395


namespace find_k_l575_575450

noncomputable def S (n : ℕ) : ℤ := n^2 - 8 * n
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_k (k : ℕ) (h : a k = 5) : k = 7 := by
  sorry

end find_k_l575_575450


namespace find_e_m_l575_575207

noncomputable def B (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![4, 5], ![7, e]]
noncomputable def B_inv (e : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := (1 / (4 * e - 35)) • ![![e, -5], ![-7, 4]]

theorem find_e_m (e m : ℝ) (B_inv_eq_mB : B_inv e = m • B e) : e = -4 ∧ m = 1 / 51 :=
sorry

end find_e_m_l575_575207


namespace domain_of_composite_function_l575_575199

theorem domain_of_composite_function (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 9 → (1 ≤ x + 1 ∧ x + 1 ≤ 10)) →
  (∀ x, 0 ≤ log (x + 1) ∧ log (x + 1) ≤ 9) →
  (∀ x, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 ∧ x^2 ≤ 1) →
  (∃ x, -1 ≤ x ∧ x ≤ 1) :=
by
  sorry

end domain_of_composite_function_l575_575199


namespace arithmetic_sequence_l575_575011

open Nat

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_S : S (2 * n + 1) - S (2 * n - 1) + S 2 = 24) 
  (h_S_def : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2 ∧ d = a 2 - a 1) : 
  a (n + 1) = 6 :=
by
  sorry

end arithmetic_sequence_l575_575011


namespace stratified_sampling_probability_l575_575973

open Finset Nat

noncomputable def combin (n k : ℕ) : ℕ := choose n k

theorem stratified_sampling_probability :
  let total_balls := 40
  let red_balls := 16
  let blue_balls := 12
  let white_balls := 8
  let yellow_balls := 4
  let n_draw := 10
  let red_draw := 4
  let blue_draw := 3
  let white_draw := 2
  let yellow_draw := 1
  
  combin yellow_balls yellow_draw * combin white_balls white_draw * combin blue_balls blue_draw * combin red_balls red_draw = combin total_balls n_draw :=
sorry

end stratified_sampling_probability_l575_575973


namespace digit_150_of_7_over_29_l575_575696

theorem digit_150_of_7_over_29 : 
  (let frac := 7 / 29 in 
   let decimal_expansion := "0.2413793103448275862068965517" in
   nat.mod 150 28 = 22 ∧ (decimal_expansion.get 21).to_nat = 5) :=
by sorry

end digit_150_of_7_over_29_l575_575696


namespace radii_of_cylinder_and_cone_are_equal_l575_575734

theorem radii_of_cylinder_and_cone_are_equal
  (h : ℝ)
  (r : ℝ)
  (V_cylinder : ℝ := π * r^2 * h)
  (V_cone : ℝ := (1/3) * π * r^2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
  r = r :=
by
  sorry

end radii_of_cylinder_and_cone_are_equal_l575_575734


namespace part_one_proof_part_two_proof_l575_575545

variables (A B C a b c : ℝ)
variables (m n : ℝ × ℝ)

-- Conditions
def triangle_sides := b = 5
def vectors_parallel := m = (a, c) ∧ n = (1 - 2 * Real.cos A, 2 * Real.cos C - 1) ∧ (Real.cos A * Real.cos C - 1 = 0)
def tan_half_B := Real.tan (B / 2) = 1 / 2

-- Part (I): Prove a + c = 10
theorem part_one_proof (h1 : triangle_sides) (h2 : vectors_parallel) : a + c = 10 := sorry

-- Part (II): Prove cos A = 3/5
theorem part_two_proof (h1 : tan_half_B) : Real.cos A = 3 / 5 := sorry

end part_one_proof_part_two_proof_l575_575545


namespace largest_degree_condition_l575_575373

noncomputable def largest_possible_degree (p_den : Polynomial ℝ) : ℕ :=
sorry

theorem largest_degree_condition (p : Polynomial ℝ) :
  (p.degree ≤ (Polynomial.X ^ 6 + Polynomial.C 2 * Polynomial.X ^ 3 -
    Polynomial.X + Polynomial.C 4).degree) → (largest_possible_degree (Polynomial.X ^ 6 + Polynomial.C 2 * Polynomial.X ^ 3 - Polynomial.X + Polynomial.C 4) = 6) :=
sorry

end largest_degree_condition_l575_575373


namespace find_a_plus_b_l575_575042

noncomputable def f (a b x : ℝ) := a ^ x + b

theorem find_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (dom1 : f a b (-2) = -2) (dom2 : f a b 0 = 0) :
  a + b = (Real.sqrt 3) / 3 - 3 :=
by
  unfold f at dom1 dom2
  sorry

end find_a_plus_b_l575_575042


namespace triangle_third_side_l575_575512

theorem triangle_third_side (a b x : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  x = 5 ∨ x = Real.sqrt 7 :=
sorry

end triangle_third_side_l575_575512


namespace number_of_boys_l575_575494

variable {total_marbles : ℕ} (marbles_per_boy : ℕ := 10)
variable (H_total_marbles : total_marbles = 20)

theorem number_of_boys (total_marbles_marbs_eq_20 : total_marbles = 20) (marbles_per_boy_eq_10 : marbles_per_boy = 10) :
  total_marbles / marbles_per_boy = 2 :=
by {
  sorry
}

end number_of_boys_l575_575494


namespace sequence_expression_l575_575539

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n + 2^n

theorem sequence_expression (n : ℕ) : a n = 2^n := by
  sorry

end sequence_expression_l575_575539


namespace smallest_x_in_domain_of_ffx_l575_575506

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 3)

theorem smallest_x_in_domain_of_ffx :
  ∃ x : ℝ, (∀ y : ℝ, y < x → ¬(Real.sqrt ((f y)^2 - 3) ≥ 0)) ∧ f(f(x)) ≥ 0 :=
begin
  sorry
end

end smallest_x_in_domain_of_ffx_l575_575506


namespace points_do_not_form_triangle_l575_575940

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def is_collinear (a b c : Point3D) : Prop :=
  let AB := distance a b
  let AC := distance a c
  let BC := distance b c
  AB + BC = AC ∨ AB + AC = BC ∨ AC + BC = AB

theorem points_do_not_form_triangle :
  let A := Point3D.mk (-1) 0 1
  let B := Point3D.mk 2 4 3
  let C := Point3D.mk 5 8 5
  is_collinear A B C :=
by {
  let A := Point3D.mk (-1) 0 1,
  let B := Point3D.mk 2 4 3,
  let C := Point3D.mk 5 8 5,
  sorry -- proof steps would go here
}

end points_do_not_form_triangle_l575_575940


namespace sub_base8_l575_575327

theorem sub_base8 (x y : ℕ) (hx : x = 4765) (hy : y = 2314) :
  nat.sub_repr 8 x y = 2447 :=
by {
  -- We could place the actual demonstration here
  sorry
}

end sub_base8_l575_575327


namespace digit_2_count_divisible_by_3_l575_575068
  
/--
The number of positive integers less than or equal to 3000
that contain at least one digit '2' and are divisible by 3 is 384.
-/
theorem digit_2_count_divisible_by_3 : 
  ∃ count : ℕ, count = nat.count (λ n : ℕ, n ≤ 3000 ∧
                                   (∃ d : ℕ, d ∈ n.digits 10 ∧ d = 2) ∧ 
                                    n % 3 = 0) (range 1 3001) = 384 :=
begin
  sorry
end

end digit_2_count_divisible_by_3_l575_575068


namespace combination_8_5_l575_575355

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575355


namespace binom_8_5_eq_56_l575_575368

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575368


namespace AB_vector_eq_l575_575447

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C D : V)
variables (a b : V)
variable (ABCD_parallelogram : is_parallelogram A B C D)

-- Definition of the diagonals
def AC_vector : V := C - A
def BD_vector : V := D - B

-- The given condition that diagonals AC and BD are equal to a and b respectively
axiom AC_eq_a : AC_vector A C = a
axiom BD_eq_b : BD_vector B D = b

-- Proof problem
theorem AB_vector_eq : (B - A) = (1/2) • (a - b) :=
sorry

end AB_vector_eq_l575_575447


namespace water_remaining_l575_575323

theorem water_remaining (initial_water : ℝ) (poured_out : ℝ) (remaining_water : ℝ) 
  (h1 : initial_water = 0.8) 
  (h2 : poured_out = 0.2) 
  (h3 : remaining_water = 0.6) : 
  initial_water - poured_out = remaining_water :=
by
  rw [h1, h2, h3]
  exact rfl

end water_remaining_l575_575323


namespace product_of_sequence_is_243_l575_575335

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l575_575335


namespace binary_110011_to_decimal_l575_575398

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575398


namespace binomial_evaluation_l575_575824

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575824


namespace hyperbola_asymptote_l575_575050

def hyperbola_eqn (m x y : ℝ) := m * x^2 - y^2 = 1

def vertex_distance_condition (m : ℝ) := 2 * Real.sqrt (1 / m) = 4

theorem hyperbola_asymptote (m : ℝ) (h_eq : hyperbola_eqn m x y) (h_dist : vertex_distance_condition m) :
  ∃ k, y = k * x ∧ k = 1 / 2 ∨ k = -1 / 2 := by
  sorry

end hyperbola_asymptote_l575_575050


namespace minimum_tangent_length_l575_575744

theorem minimum_tangent_length : 
  let line := (y = x + 1)
  let circle := ((x - 3)^2 + y^2 = 1)
  ∃ m : Real, 
  ( ∀ p on line, 
      distance p center_of_circle 
      ≥ √7 
    ) := sorry

end minimum_tangent_length_l575_575744


namespace solve_for_x_l575_575258

theorem solve_for_x :
  ∃ x : ℝ, x = 4 ∧ 2^(x-1) + 2^(x-4) + 2^(x-2) = 13 :=
by
  sorry

end solve_for_x_l575_575258


namespace binom_8_5_eq_56_l575_575351

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575351


namespace binary_110011_to_decimal_l575_575399

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575399


namespace prove_ineq_l575_575405

-- Define the quadratic equation
def quadratic_eqn (a b x : ℝ) : Prop :=
  3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0

-- Define the root relation
def root_relation (x1 x2 : ℝ) : Prop :=
  x1 * (x1 + 1) + x2 * (x2 + 1) = (x1 + 1) * (x2 + 1)

-- State the theorem
theorem prove_ineq (a b : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eqn a b x1 ∧ quadratic_eqn a b x2 ∧ root_relation x1 x2) →
  (a + b)^2 ≤ 4 :=
by
  sorry

end prove_ineq_l575_575405


namespace min_sticks_to_avoid_rectangles_l575_575141

noncomputable def min_stick_deletions (n : ℕ) : ℕ :=
  if n = 8 then 43 else 0 -- we define 43 as the minimum for an 8x8 chessboard

theorem min_sticks_to_avoid_rectangles : min_stick_deletions 8 = 43 :=
  by
    sorry

end min_sticks_to_avoid_rectangles_l575_575141


namespace convex_quadrilateral_max_two_obtuse_l575_575067

theorem convex_quadrilateral_max_two_obtuse (a b c d : ℝ)
  (h1 : a + b + c + d = 360)
  (h2 : a < 180) (h3 : b < 180) (h4 : c < 180) (h5 : d < 180)
  : (∃ A1 A2, a = A1 ∧ b = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ c < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ c < 90) ∨
    (∃ A1 A2, b = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ d < 90) ∨
    (∃ A1 A2, b = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ c < 90) ∨
    (∃ A1 A2, c = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ b < 90) ∨
    (¬∃ x y z, (x > 90) ∧ (y > 90) ∧ (z > 90) ∧ x + y + z ≤ 360) := sorry

end convex_quadrilateral_max_two_obtuse_l575_575067


namespace order_of_magnitudes_l575_575178

variable (x : ℝ)
variable (a : ℝ)

theorem order_of_magnitudes (h1 : x < 0) (h2 : a = 2 * x) : x^2 < a * x ∧ a * x < a^2 := 
by
  sorry

end order_of_magnitudes_l575_575178


namespace tourist_tax_l575_575750

theorem tourist_tax (total_value : ℝ) (non_taxable_amount : ℝ) (tax_rate : ℝ) 
  (h1 : total_value = 1720) (h2 : non_taxable_amount = 600) (h3 : tax_rate = 0.08) : 
  ((total_value - non_taxable_amount) * tax_rate = 89.60) :=
by 
  sorry

end tourist_tax_l575_575750


namespace sequence_general_term_l575_575096

namespace SequenceSum

def Sn (n : ℕ) : ℕ :=
  2 * n^2 + n

def a₁ (n : ℕ) : ℕ :=
  if n = 1 then Sn n else (Sn n - Sn (n - 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) : 
  a₁ n = 4 * n - 1 :=
sorry

end SequenceSum

end sequence_general_term_l575_575096


namespace total_bronze_needed_l575_575585

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l575_575585


namespace lucy_jumps_2100_times_l575_575580

-- Define the total number of songs and the duration of each song in minutes
def numSongs : Nat := 10
def durationPerSong : Real := 3.5

-- Duration of the album in seconds
def durationInSeconds : Real := durationPerSong * 60 * numSongs

-- Lucy's jump rate in jumps per second
def jumpsPerSecond : Nat := 1

-- Total jumps
def totalJumps : Nat := durationInSeconds.to_nat * jumpsPerSecond

-- The theorem stating Lucy will jump rope 2100 times
theorem lucy_jumps_2100_times :
  totalJumps = 2100 := 
by
  sorry

end lucy_jumps_2100_times_l575_575580


namespace probabilities_inequalities_l575_575531

variables (M N : Prop) (P : Prop → ℝ)

axiom P_pos_M : P M > 0
axiom P_pos_N : P N > 0
axiom P_cond_N_M : P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)

theorem probabilities_inequalities :
    (P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)) ∧
    (P (N ∧ M) > P N * P M) ∧
    (P (M ∧ N) / P N > P (M ∧ ¬N) / P (¬N)) :=
by
    sorry

end probabilities_inequalities_l575_575531


namespace find_sequence_l575_575661

noncomputable def seq (a : ℕ → ℝ) :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = (n / (n + 1)) * (a n + 1))

theorem find_sequence {a : ℕ → ℝ} (h : seq a) :
  ∀ n, a n = (n - 1) / 2 :=
sorry

end find_sequence_l575_575661


namespace binary_to_decimal_110011_l575_575378

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575378


namespace inequality_x2_y4_z6_l575_575617

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l575_575617


namespace tan_theta_sqrt3_l575_575891

theorem tan_theta_sqrt3 (θ : ℝ) 
  (h : Real.cos (40 * (π / 180) - θ) 
     + Real.cos (40 * (π / 180) + θ) 
     + Real.cos (80 * (π / 180) - θ) = 0) 
  : Real.tan θ = -Real.sqrt 3 := 
by
  sorry

end tan_theta_sqrt3_l575_575891


namespace original_proposition_converse_negation_contrapositive_l575_575463

variable {a b : ℝ}

-- Original Proposition: If \( x^2 + ax + b \leq 0 \) has a non-empty solution set, then \( a^2 - 4b \geq 0 \)
theorem original_proposition (h : ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b ≥ 0 := sorry

-- Converse: If \( a^2 - 4b \geq 0 \), then \( x^2 + ax + b \leq 0 \) has a non-empty solution set
theorem converse (h : a^2 - 4 * b ≥ 0) : ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

-- Negation: If \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set, then \( a^2 - 4b < 0 \)
theorem negation (h : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b < 0 := sorry

-- Contrapositive: If \( a^2 - 4b < 0 \), then \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set
theorem contrapositive (h : a^2 - 4 * b < 0) : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

end original_proposition_converse_negation_contrapositive_l575_575463


namespace incorrect_step_l575_575894

-- Given conditions
variables {a b : ℝ} (hab : a < b)

-- Proof statement of the incorrect step ③
theorem incorrect_step : ¬ (2 * (a - b) ^ 2 < (a - b) ^ 2) :=
by sorry

end incorrect_step_l575_575894


namespace sum_of_logs_l575_575116

open Real

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a n = a 0 * r ^ n

axiom geom_seq_pos (a : ℕ → ℝ) : (geometric_sequence a) ∧ (∀ n, 0 < a n)

axiom a5a6_eq_81 (a : ℕ → ℝ) : a 5 * a 6 = 81

-- Prove the sum of logarithms
theorem sum_of_logs (a : ℕ → ℝ) (r : ℝ) (h_seq : geometric_sequence a) (h_pos : ∀ n, 0 < a n ) (h_a5a6 : a 5 * a 6 = 81)
    : ∑ i in finset.range 10, log 3 (a (i+1)) = 20 := 
sorry

end sum_of_logs_l575_575116


namespace max_perfect_square_gifts_l575_575521

theorem max_perfect_square_gifts (n : ℤ) :
  let S := (list.range 10).map (λ i, 9 * n + 36 + i) in
  (list.filter (λ x, ∃ m : ℕ, x = m * m) S).length ≤ 4 :=
by
  sorry

end max_perfect_square_gifts_l575_575521


namespace inequality_solution_l575_575173

theorem inequality_solution (x : ℝ) (h1 : 2 * x + 1 > x + 3) (h2 : 2 * x - 4 < x) : 2 < x ∧ x < 4 := sorry

end inequality_solution_l575_575173


namespace remainder_div_l575_575427

-- Definition of the polynomial f
def f (r : ℕ) : ℕ := r ^ 17 + 1

-- Theorem statement: Proving the remainder when f(r) is divided by r + 1 is 0
theorem remainder_div (r : ℕ) : (f (-1) % (r + 1) = 0) := by
  -- The remainder when f(-1) is divided by r+1 is 0
  sorry

end remainder_div_l575_575427


namespace train_speed_in_kmh_l575_575269

theorem train_speed_in_kmh 
  (train_length : ℕ) 
  (crossing_time : ℕ) 
  (conversion_factor : ℕ) 
  (hl : train_length = 120) 
  (ht : crossing_time = 6) 
  (hc : conversion_factor = 36) :
  train_length / crossing_time * conversion_factor / 10 = 72 := by
  sorry

end train_speed_in_kmh_l575_575269


namespace a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l575_575267

-- For Question 1
theorem a_squared_plus_b_squared_gt_one_over_four (a b : ℝ) (h : a + b = 1) : a^2 + b^2 > 1/4 :=
sorry

-- For Question 2
theorem sequence_is_arithmetic (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = 2 * (n:ℝ)^2 - 3 * (n:ℝ) - 2) :
  ∃ d, ∀ n, (S n / (2 * (n:ℝ) + 1)) = (S (n + 1) / (2 * (n + 1:ℝ) + 1)) + d :=
sorry

end a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l575_575267


namespace log5_one_div_25_eq_neg2_l575_575416

theorem log5_one_div_25_eq_neg2 : log 5 (1/25) = -2 := 
by
  -- Let's denote the logarithm by x for convenience
  let x := log 5 (1/25)
  -- By definition of logarithm, we have 5^x = 1/25
  have h1 : 5^x = (1/25) := by sorry
  -- We also know that 1/25 = 5^-2
  have h2 : (1/25) = 5^(-2) := by sorry
  -- Thus, 5^x = 5^-2
  have h3 : 5^x = 5^(-2) := by
    rw [←h2] at h1
    exact h1
  -- Equating the exponents of the same base, we get x = -2
  have h4 : x = -2 := by
    apply eq_of_pow_eq_pow
    exact h3
  -- Therefore, log 5 (1/25) = -2
  exact h4
  sorry

end log5_one_div_25_eq_neg2_l575_575416


namespace find_b_l575_575571

theorem find_b (α β b : ℤ)
  (h1: α > 1)
  (h2: β < -1)
  (h3: ∃ x : ℝ, α * x^2 + β * x - 2 = 0)
  (h4: ∃ x : ℝ, x^2 + bx - 2 = 0)
  (hb: ∀ root1 root2 : ℝ, root1 * root2 = -2 ∧ root1 + root2 = -b) :
  b = 0 := 
sorry

end find_b_l575_575571


namespace point_not_in_first_quadrant_l575_575476

theorem point_not_in_first_quadrant (a : ℝ) : 
  let z := (a - complex.I) / (1 + complex.I) in
  ¬(z.re > 0 ∧ z.im > 0) := 
by
  sorry

end point_not_in_first_quadrant_l575_575476


namespace avg_salary_increase_l575_575184

theorem avg_salary_increase 
  (avg_salary_18 : ℝ)
  (num_employees : ℕ)
  (manager_salary : ℝ)
  (old_total_salary : avg_salary_18 * num_employees = 36000)
  (new_total_salary : old_total_salary + manager_salary = 41800)
  (new_num_employees : 19)
  (old_avg_salary : avg_salary_18 = 2000)
  (manager_salary_value : manager_salary = 5800) :
  (41800 / 19) - 2000 = 200 :=
by
  sorry

end avg_salary_increase_l575_575184


namespace evaluate_g_neg8_l575_575132

def g (x : ℝ) : ℝ :=
if x < 3 then 3 * x - 4 else x^2 + 1

theorem evaluate_g_neg8 : g (-8) = -28 := by
  -- proof would go here
  sorry

end evaluate_g_neg8_l575_575132


namespace domain_g_eq_l575_575179

noncomputable def domain_f : Set ℝ := {x | -8 ≤ x ∧ x ≤ 4}

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

theorem domain_g_eq (f : ℝ → ℝ) (h : ∀ x, x ∈ domain_f → f x ∈ domain_f) :
  {x | x ∈ [-2, 4]} = {x | -2 ≤ x ∧ x ≤ 4} :=
by {
  sorry
}

end domain_g_eq_l575_575179


namespace sequence_periodic_l575_575095

def sequence (n : ℕ) : ℚ :=
  if n = 0 then -1/4
  else 1 - 1 / sequence (n - 1)

theorem sequence_periodic (a_n : ℕ) (hn : a_n > 0) : 
  sequence 2018 = 5 :=
by sorry

end sequence_periodic_l575_575095


namespace triangle_AD_DU_l575_575009

noncomputable def point := ℝ × ℝ
noncomputable def angle (A B C : point) := -- (Angle at B formed by points A, B, and C)
sorry

def Triangle (A B C : point) : Prop := -- Placeholder definition of a triangle
sorry

variables {A B C U D : point}

-- Conditions of the problem
axiom circumcenter (U : point) (A B C : point) : Triangle A B C ∧ 
  -- U is the circumcenter of triangle ABC
  (∀ X Y, (X = U ∧ Y = A) → angle B A Y = angle C B U) ∧  -- Circumcenter properties relationships

axiom angle_CBA_60 : angle C B A = 60
axiom angle_CBU_45 : angle C B U = 45
axiom intersect_BU_AC_at_D : -- BU and AC intersect at point D
  ∃ D : point, -- D exists such that it is the intersection of BU and AC
    (∃ U : point, ∃ B : point, ∃ AC : set point, AC = {x | x = (A,C)}) ∧ 
    D ∈ (line_segment B U) ∧ D ∈ (line_segment A C)

-- Goal: AD = DU
theorem triangle_AD_DU (h1 : circumcenter U A B C)
                        (h2 : angle C B A = 60)
                        (h3 : angle C B U = 45)
                        (h4 : ∃ D, D ∈ (line_segment B U) ∧ D ∈ (line_segment A C)) :
  dist A D = dist D U :=
sorry

end triangle_AD_DU_l575_575009


namespace digit_150_of_seven_over_twenty_nine_l575_575698

theorem digit_150_of_seven_over_twenty_nine : 
  (decimal_expansion 7 29).nth 150 = 8 :=
sorry

end digit_150_of_seven_over_twenty_nine_l575_575698


namespace longest_segment_CD_l575_575443

variables (A B C D : Type)
variables (angle_ABD angle_ADB angle_BDC angle_CBD : ℝ)

axiom angle_ABD_eq : angle_ABD = 30
axiom angle_ADB_eq : angle_ADB = 65
axiom angle_BDC_eq : angle_BDC = 60
axiom angle_CBD_eq : angle_CBD = 80

theorem longest_segment_CD
  (h_ABD : angle_ABD = 30)
  (h_ADB : angle_ADB = 65)
  (h_BDC : angle_BDC = 60)
  (h_CBD : angle_CBD = 80) : false :=
sorry

end longest_segment_CD_l575_575443


namespace non_intersecting_segments_possible_l575_575694

theorem non_intersecting_segments_possible
  (n : ℕ)
  (red_points blue_points : Fin n → Point ℕ)
  (h_no_three_collinear : ∀ (a b c : Point ℕ), a ∈ red_points ∧ b ∈ blue_points ∧ c ∈ (red_points ∪ blue_points) → ¬Collinear a b c) :
  ∃ (p : SymmetricPairs (Fin n)) (h : ∀ i, (p i).1 ∈ red_points ∧ (p i).2 ∈ blue_points), 
    (∀ i j, i ≠ j → ¬SegmentsIntersect (p i) (p j)) :=
sorry

end non_intersecting_segments_possible_l575_575694


namespace total_persons_l575_575149

theorem total_persons (n : ℕ) (A : ℝ) (h1 : 8 * 30 = 240)
  (h2 : (292.5 / n) + 20 = A)
  (h3 : 240 + A = 292.5) : n = 9 :=
begin
  sorry
end

end total_persons_l575_575149


namespace even_number_exists_from_step_2_l575_575537

noncomputable def update_sequence (seq : ℤ → ℤ) : ℤ → ℤ :=
  λ n, seq (n - 1) + seq (n + 1)

theorem even_number_exists_from_step_2
  (seq : ℤ → ℤ)
  (h_initial : seq 0 = 1 ∧ (∀ n ≠ 0, seq n = 0))
  (∀ n, ∃ k, k ≥ 0 → seq k = update_sequence (update_sequence seq) k)
  : ∃ (k : ℤ), k ≥ 2 ∧ even (seq k) ∧ seq k ≠ 0 := 
sorry

end even_number_exists_from_step_2_l575_575537


namespace log_base_5_of_1_div_25_l575_575417

theorem log_base_5_of_1_div_25 : log 5 (1 / 25) = -2 := by
  sorry

end log_base_5_of_1_div_25_l575_575417


namespace complex_number_in_third_quadrant_l575_575212

def is_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_in_third_quadrant : is_third_quadrant (1 - 2 * Complex.I) / (1 + Complex.I) :=
by
  sorry -- The proof would go here.

end complex_number_in_third_quadrant_l575_575212


namespace width_of_field_l575_575204

variable (w : ℕ)

def length (w : ℕ) : ℕ := (7 * w) / 5

def perimeter (w : ℕ) : ℕ := 2 * (length w) + 2 * w

theorem width_of_field (h : perimeter w = 240) : w = 50 :=
sorry

end width_of_field_l575_575204


namespace solve_for_x_l575_575709

theorem solve_for_x : ∃ x : ℝ, (9 - x) ^ 2 = x ^ 2 ∧ x = 4.5 :=
by
  sorry

end solve_for_x_l575_575709


namespace intersection_M_N_eq_M_l575_575554

-- Definition of M
def M := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Definition of N
def N := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_M_N_eq_M : (M ∩ N) = M :=
  sorry

end intersection_M_N_eq_M_l575_575554


namespace root_in_interval_l575_575710

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem root_in_interval : ∃ x ∈ Ioo (0 : ℝ) (1 : ℝ), f x = 0 := 
by
  sorry

end root_in_interval_l575_575710


namespace calc_product_eq_243_l575_575330

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l575_575330


namespace prove_one_certain_event_exists_l575_575314

def certain_event_condition_4 : Prop := ∀ (a b : ℝ), a * b = b * a
def impossible_event_condition_1 : Prop := pure_water_freezes_at_20_degrees_under_1_atm = false
def impossible_event_condition_2 : Prop := ∀ (score : ℝ), score ≠ 105
def random_event_condition_3 : Prop := ∃ (result : char), result = 'H' ∨ result = 'T'

theorem prove_one_certain_event_exists 
  (cond1 : impossible_event_condition_1)
  (cond2 : impossible_event_condition_2)
  (cond3 : random_event_condition_3)
  (cond4 : certain_event_condition_4) : 
  ∃ e, e = cond1 ∨ e = cond2 ∨ e = cond3 ∨ e = cond4 ∧ e = cond4 :=
by
  sorry

end prove_one_certain_event_exists_l575_575314


namespace distinct_values_of_expressions_l575_575038

theorem distinct_values_of_expressions : 
  let x := 3
  let expr1 := x^(x^(x^x))
  let expr2 := x^((x^x)^x)
  let expr3 := ((x^x)^x)^x
  let expr4 := (x^(x^x))^x
  let expr5 := (x^x)^(x^x)
  let values := {expr1, expr2, expr3, expr4, expr5}
  set.size values = 3 := by
  sorry

end distinct_values_of_expressions_l575_575038


namespace compute_x_squared_first_compute_x_squared_second_l575_575052

variable (x : ℝ)
variable (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1)

theorem compute_x_squared_first : 
  1 / (1 / x - 1 / (x + 1)) - x = x^2 :=
by
  sorry

theorem compute_x_squared_second : 
  1 / (1 / (x - 1) - 1 / x) + x = x^2 :=
by
  sorry

end compute_x_squared_first_compute_x_squared_second_l575_575052


namespace rectangle_cannot_be_decomposed_into_squares_of_distinct_fibonacci_lengths_l575_575293

inductive Fibonacci : ℕ → ℕ
| F1 : Fibonacci 1
| F2 : Fibonacci 1
| Fn (n : ℕ) (F_n : Fibonacci n) (F_n1 : Fibonacci (n - 1)) : Fibonacci (n + 1) := 
  F_n + F_n1

theorem rectangle_cannot_be_decomposed_into_squares_of_distinct_fibonacci_lengths : 
  ∀ (rect : ℝ × ℝ) (l : list ℝ), (∃ s : list ℝ, (∀ x ∈ s, x ∈ l) ∧ 
  ∃ n_k m_k : ℕ, n_k < m_k ∧ 
  (∀ k, s[k] ∈ (set.univ : set ℝ) → ∃ (n : ℕ), s[k] = Fibonacci n) ∧ 
  ¬ (rect = l.sum * l.head + l.tail.sum)) sorry

end rectangle_cannot_be_decomposed_into_squares_of_distinct_fibonacci_lengths_l575_575293


namespace exists_m₀_l575_575607

-- Define the function f with its given constraints
def f : ℕ → ℕ
| 1 := 0
| n := sorry  -- f is recursively defined based on the problem's rules; this definition is incomplete without implementation

-- The target theorem statement
theorem exists_m₀ (n : ℕ) : ∃ m₀ : ℕ, ∀ m : ℕ, m > m₀ → f m < 2 ^ (α₁ + α₂ + ... + αᵢ - n) :=
sorry

end exists_m₀_l575_575607


namespace special_number_digits_sum_l575_575875

def repeated_digits (digit : ℕ) (count : ℕ) : ℕ :=
  foldr (λ _ acc, acc * 10 + digit) 0 (list.range count)

def special_number_part1 : ℕ := repeated_digits 1 2017
def special_number_part2 : ℕ := repeated_digits 2 2018
def special_number : ℕ := special_number_part1 * (10 ^ 2019) + special_number_part2 * 10 + 5

noncomputable def integer_part_sqrt (n : ℕ) : ℕ :=
  nat.floor (real.sqrt ↑n)

def digit_sum (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

theorem special_number_digits_sum :
  digit_sum (integer_part_sqrt special_number) = 6056 :=
sorry

end special_number_digits_sum_l575_575875


namespace binomial_evaluation_l575_575826

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575826


namespace incorrect_conclusions_l575_575887

theorem incorrect_conclusions :
  let eq1 := (2.347, -6.423, -0.9284)
  let eq2 := (-3.476, 5.648, -0.9533)
  let eq3 := (5.437, 8.493, 0.9830)
  let eq4 := (-4.326, -4.578, 0.8997)
  (eq1.2 >= 0 ∧ eq1.3 < 0) ∨ (eq4.2 < 0 ∧ eq4.3 >= 0) :=
by
  sorry

end incorrect_conclusions_l575_575887


namespace part_one_part_two_part_three_l575_575308

-- Definition of the operation ⊕
def op⊕ (a b : ℚ) : ℚ := a * b + 2 * a

-- Part (1): Prove that 2 ⊕ (-1) = 2
theorem part_one : op⊕ 2 (-1) = 2 :=
by
  sorry

-- Part (2): Prove that -3 ⊕ (-4 ⊕ 1/2) = 24
theorem part_two : op⊕ (-3) (op⊕ (-4) (1 / 2)) = 24 :=
by
  sorry

-- Part (3): Prove that ⊕ is not commutative
theorem part_three : ∃ (a b : ℚ), op⊕ a b ≠ op⊕ b a :=
by
  use 2, -1
  sorry

end part_one_part_two_part_three_l575_575308


namespace simplify_trig_expr_l575_575168

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem we need to prove
theorem simplify_trig_expr : 
  sin (x + y) * sin (x - y) - cos (x + y) * cos (x - y) = - cos (2 * x) :=
sorry

end simplify_trig_expr_l575_575168


namespace three_digit_integers_product_30_l575_575498

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end three_digit_integers_product_30_l575_575498


namespace hyperbola_asymptotes_l575_575757

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 - (y^2 / 4) = 1) ↔ (y = 2 * x ∨ y = -2 * x) := by
  sorry

end hyperbola_asymptotes_l575_575757


namespace pencil_length_after_sharpening_l575_575987

-- Definition of the initial length of the pencil
def initial_length : ℕ := 22

-- Definition of the amount sharpened each day
def sharpened_each_day : ℕ := 2

-- Final length of the pencil after sharpening on Monday and Tuesday
def final_length (initial_length : ℕ) (sharpened_each_day : ℕ) : ℕ :=
  initial_length - sharpened_each_day * 2

-- Theorem stating that the final length is 18 inches
theorem pencil_length_after_sharpening : final_length initial_length sharpened_each_day = 18 := by
  sorry

end pencil_length_after_sharpening_l575_575987


namespace complex_numbers_product_l575_575917

theorem complex_numbers_product (z1 z2 : ℂ) (h1 : complex.abs z1 = 1) (h2 : complex.abs z2 = 1) 
  (h3 : z1 + z2 = -7/5 + 1/5 * complex.i) : z1 * z2 = 24/25 - 7/25 * complex.i :=
sorry

end complex_numbers_product_l575_575917


namespace proposition_B_is_correct_l575_575711

variable {p q : Prop}

theorem proposition_B_is_correct :
  (¬ (∃ x ∈ ℝ, sin x > 1)) ↔ (∀ x ∈ ℝ, sin x ≤ 1) := by
sorry

end proposition_B_is_correct_l575_575711


namespace inequality_example_l575_575620

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l575_575620


namespace greatest_third_term_of_arithmetic_sequence_l575_575671

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l575_575671


namespace binom_12_6_l575_575797

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575797


namespace james_winnings_l575_575105

theorem james_winnings (W : ℝ)
  (donated : W / 2)
  (spent : W / 2 - 2)
  (remaining : W / 2 - 2 = 55) : 
  W = 114 :=
by sorry

end james_winnings_l575_575105


namespace total_amount_spent_l575_575773

def cost_of_tshirt : ℕ := 100
def cost_of_pants : ℕ := 250
def num_of_tshirts : ℕ := 5
def num_of_pants : ℕ := 4

theorem total_amount_spent : (num_of_tshirts * cost_of_tshirt) + (num_of_pants * cost_of_pants) = 1500 := by
  sorry

end total_amount_spent_l575_575773


namespace distribute_water_l575_575317

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h
noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem distribute_water :
  let r_cone := 6 in
  let h_cone := 15 in
  let r_cylinder := 6 in
  let V_cone := volume_of_cone r_cone h_cone in
  let h_cylinder := V_cone / (π * r_cylinder^2 * 2) in
  h_cylinder = 2.5 :=
by
  have r_cone := 6
  have h_cone := 15
  have r_cylinder := 6
  have V_cone := volume_of_cone r_cone h_cone
  have h_cylinder := V_cone / (π * r_cylinder^2 * 2)
  calc h_cylinder
    = (volume_of_cone 6 15) / (π * 6^2 * 2) : by sorry
    ... = 2.5 : by sorry

end distribute_water_l575_575317


namespace cost_of_tomatoes_l575_575856

theorem cost_of_tomatoes (N C T : ℕ) (h_gt_one : N > 1) (hN5: N = 5) (h_cost : N * (3 * C + 4 * T) = 305) (h_eq : 3 * C + 4 * T = 61) : 
  4 * T * N = 200 := by
  rw [hN5] at *
  have h_tomatoes_cost : 4 * T * 5 = 200 := by
  sorry
  exact h_tomatoes_cost

end cost_of_tomatoes_l575_575856


namespace sin_a_2025_l575_575929

def f (x : ℝ) := x + 2 * Real.sin x

def a_n (n : ℕ) : ℝ := (4 * Real.pi / 3) + 2 * (n : ℝ) * Real.pi

theorem sin_a_2025 : Real.sin (a_n 2025) = - (Real.sqrt 3 / 2) := 
by sorry

end sin_a_2025_l575_575929


namespace min_dist_value_l575_575609

open Real

-- Defining the parabola y^2 = 4x
def on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 4 * P.1

-- Defining the line x - y + 5 = 0
def on_line (Q : ℝ × ℝ) : Prop :=
  Q.1 - Q.2 + 5 = 0

-- Defining the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Function to compute the Euclidean distance between two points
def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def min_value (P Q : ℝ × ℝ) : ℝ :=
  dist P focus + dist P Q

theorem min_dist_value :
  ∃ P Q : ℝ × ℝ, on_parabola P ∧ on_line Q ∧ (min_value P Q = 3 * sqrt 2) :=
by
  sorry

end min_dist_value_l575_575609


namespace exists_infinitely_many_n_with_prime_divisor_gt_2n_sqrt_5n_2011_l575_575612

open Nat Real

theorem exists_infinitely_many_n_with_prime_divisor_gt_2n_sqrt_5n_2011 :
  ∃∞ n : ℕ, ∃ p : ℕ, Prime p ∧ p ∣ (n^2 + 1) ∧ (p > 2 * n + sqrt (5 * n + 2011)) :=
sorry

end exists_infinitely_many_n_with_prime_divisor_gt_2n_sqrt_5n_2011_l575_575612


namespace remainder_of_polynomial_division_l575_575854

-- Define the polynomial f(r)
def f (r : ℝ) : ℝ := r ^ 15 + 1

-- Define the polynomial divisor g(r)
def g (r : ℝ) : ℝ := r + 1

-- State the theorem about the remainder when f(r) is divided by g(r)
theorem remainder_of_polynomial_division : 
  (f (-1)) = 0 := by
  -- Skipping the proof for now
  sorry

end remainder_of_polynomial_division_l575_575854


namespace most_convincing_method_for_relationship_l575_575277

-- Definitions from conditions
def car_owners : ℕ := 300
def car_owners_opposed_policy : ℕ := 116
def non_car_owners : ℕ := 200
def non_car_owners_opposed_policy : ℕ := 121

-- The theorem statement
theorem most_convincing_method_for_relationship : 
  (owning_a_car_related_to_opposing_policy : Bool) :=
by
  -- Proof of the statement
  sorry

end most_convincing_method_for_relationship_l575_575277


namespace hike_duration_l575_575063

def initial_water := 11
def final_water := 2
def leak_rate := 1
def water_drunk := 6

theorem hike_duration (time_hours : ℕ) :
  initial_water - final_water = water_drunk + time_hours * leak_rate →
  time_hours = 3 :=
by intro h; sorry

end hike_duration_l575_575063


namespace right_triangle_median_sum_eq_semi_perimeter_l575_575604

theorem right_triangle_median_sum_eq_semi_perimeter :
  ∀ (A B C : ℝ) (a b c : ℝ), 
  a = 6 → b = 4 → 
  c = (2 : ℝ) * Real.sqrt 13 → 
  let median_from_C := Real.sqrt 13 in
  let median_from_B := 5 in
  (median_from_B + median_from_C) = (a + b + c) / 2 :=
by 
  -- First, let’s define the length of the hypotenuse \(AB\)
  have hypotenuse_length : c = 2 * Real.sqrt 13, by sorry,
  -- Then let's assert the length of the medians
  have median_from_C : median_from_C = Real.sqrt 13, by sorry,
  have median_from_B : median_from_B = 5, by sorry,
  -- Finally, we prove the required equation
  sorry

end right_triangle_median_sum_eq_semi_perimeter_l575_575604


namespace hyperbola_eccentricity_l575_575899

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c² = a² + b²)
    (h4 : b / a = sqrt 3 ∨ a / b = sqrt 3) : c / a = 2 ∨ c / a = 2 * sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l575_575899


namespace Part_a_surface_area_Part_b_surface_area_Part_c_surface_area_Part_d_surface_area_l575_575329

-- Part (a)
theorem Part_a_surface_area (R : ℝ) :
  (∫ x in -R..R, ∫ z in 0..(sqrt (R^2 - x^2)), R) = 2 * R^2 :=
by sorry

-- Part (b)
theorem Part_b_surface_area (R : ℝ) :
  (∫ x in -R..R, ∫ y in -sqrt (R^2 - x^2)..sqrt (R^2 - x^2), 1) = 4 * R^2 :=
by sorry

-- Part (c)
theorem Part_c_surface_area (R : ℝ) :
  (36 * π * (1/3(4 - 1))) = 42 * π :=
by sorry

-- Part (d)
theorem Part_d_surface_area (a : ℝ) (h : a > 0) :
  (∫ x in -sqrt(2)*a .. sqrt(2)*a, ∫ y in -sqrt(2)*a .. sqrt(2)*a, sqrt(3*a^2 - x^2 - y^2)) = 2 * sqrt(3) * a^2 * (sqrt(3) - 1) :=
by sorry

end Part_a_surface_area_Part_b_surface_area_Part_c_surface_area_Part_d_surface_area_l575_575329


namespace find_a_of_conditions_l575_575024

theorem find_a_of_conditions 
  (a : ℕ) 
  (h1: 1000 ≤ 4 * a^2 ∧ 4 * a^2 < 10000) 
  (h2: 1000 ≤ (4 / 3) * a^3 ∧ (4 / 3) * a^3 < 10000)
  (h3: (nat.floor ((4:ℚ) / 3 * (a:ℚ)^3) : ℕ) = (4:ℕ) * a^3 / 3):
  a = 18 :=
by
  sorry

end find_a_of_conditions_l575_575024


namespace eccentricity_of_ellipse_l575_575918

-- Define the eccentricity e
def eccentricity (a b : ℝ) : ℝ := real.sqrt (1 - (b / a) ^ 2)

-- Conditions:
variables (a b : ℝ)
variable (h1 : a > b > 0)
variable (h2 : ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 )
variable (h3 : 2 * abs (sqrt (a^2 - b^2)) = (2 * b^2 / a))

-- Question as proof statement:
theorem eccentricity_of_ellipse : 
  eccentricity a b = -1 + real.sqrt 2 :=
sorry

end eccentricity_of_ellipse_l575_575918


namespace binom_12_6_eq_924_l575_575806

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575806


namespace savings_plan_l575_575597

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l575_575597


namespace derivative_at_one_l575_575025

noncomputable def f : ℝ → ℝ := λ x, 2^x + 3 * x * deriv f 0

theorem derivative_at_one :
  (deriv f 1) = (ℕ.ln 2) / 2 := sorry

end derivative_at_one_l575_575025


namespace focus_of_parabola_proof_l575_575193

noncomputable def focus_of_parabola (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

theorem focus_of_parabola_proof (a : ℝ) (h : a ≠ 0) :
  focus_of_parabola a h = (1 / (4 * a), 0) :=
sorry

end focus_of_parabola_proof_l575_575193


namespace perpendicular_bisector_eq_l575_575200

theorem perpendicular_bisector_eq (A B: (ℝ × ℝ)) (hA: A = (1, 3)) (hB: B = (-5, 1)) :
  ∃ m c, (m = -3) ∧ (c = 4) ∧ (∀ x y, y = m * x + c ↔ 3 * x + y + 4 = 0) := 
by
  sorry

end perpendicular_bisector_eq_l575_575200


namespace isosceles_triangles_in_configuration_l575_575542

noncomputable def isosceles_count {A B C D E F : Type} [linear_ordered_field A] [module B A] 
  (A B C : Point A) (D E F : Point A) (AB AC : Segment A) (ABC : Triangle A) 
  (BD : Angle A) (BE DE AB : Line A) (EF CF : Segment A) (ACB : Angle A) :
  Prop :=
  let is_isosceles (t : Triangle A) : Prop :=  (t.side_a = t.side_b) ∨ (t.side_a = t.side_c) ∨ (t.side_b = t.side_c)
  in
  (isosceles_count ABC + isosceles_count ABD + isosceles_count BDC +
  isosceles_count BDE + isosceles_count DEF + isosceles_count FEC + isosceles_count DEC) = 7


theorem isosceles_triangles_in_configuration {A B C D E F : Type} [linear_ordered_field A] [module B A] 
  (A B C : Point A) (D E F : Point A) (AB AC : Segment A) (ABC : Triangle A) 
  (BD : Angle A) (BE DE AB : Line A) (EF CF : Segment A) (ACB : Angle A) :
  
  let 
    congruent_segments := AB = AC -- AB is congruent to AC
    angle_ABC := ∠B = 60 -- measure of angle ABC is 60 degrees
    angle_bisector := (BD ∈ bisects ABC) -- BD bisects angle ABC
    point_on_side_DE_parallel_AB := DE.parallel AB -- DE is parallel to AB
    point_on_side_EF_parallel_BD := EF.parallel BD -- EF is parallel to BD
    angle_bisector_CF := (CF ∈ bisects ACB) -- CF bisects angle ACB

  in

  congruent_segments ∧ angle_ABC ∧ angle_bisector ∧ point_on_side_DE_parallel_AB ∧
  point_on_side_EF_parallel_BD ∧ angle_bisector_CF →
  isosceles_count A B C D E F AB AC ABC BD BE DE AB EF CF ACB = 7 := 
sorry

end isosceles_triangles_in_configuration_l575_575542


namespace binom_8_5_l575_575341

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575341


namespace evaporation_days_l575_575271

def InitialWater : ℝ := 10
def DailyEvaporation : ℝ := 0.004
def TotalEvaporation : ℝ := (2 / 100) * InitialWater
def NumberOfDays : ℝ := TotalEvaporation / DailyEvaporation

theorem evaporation_days :
  NumberOfDays = 50 := 
by
  sorry

end evaporation_days_l575_575271


namespace number_of_sophomores_l575_575322

-- Definition of the conditions
variables (J S P j s p : ℕ)

-- Condition: Equal number of students in debate team
def DebateTeam_Equal : Prop := j = s ∧ s = p

-- Condition: Total number of students
def TotalStudents : Prop := J + S + P = 45

-- Condition: Percentage relationships
def PercentRelations_J : Prop := j = J / 5
def PercentRelations_S : Prop := s = 3 * S / 20
def PercentRelations_P : Prop := p = P / 10

-- The main theorem to prove
theorem number_of_sophomores : DebateTeam_Equal j s p 
                               → TotalStudents J S P 
                               → PercentRelations_J J j 
                               → PercentRelations_S S s 
                               → PercentRelations_P P p 
                               → P = 21 :=
by 
  sorry

end number_of_sophomores_l575_575322


namespace find_special_numbers_l575_575860

theorem find_special_numbers (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) :=
by
  sorry

end find_special_numbers_l575_575860


namespace binomial_evaluation_l575_575827

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575827


namespace binary_to_decimal_110011_l575_575383

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575383


namespace no_day_for_statements_l575_575606

def Day := ℕ -- Assume days are represented by natural numbers 0 through 6.

def yesterday (d : Day) : Day := if d = 0 then 6 else d - 1
def tomorrow (d : Day) : Day := (d + 1) % 7

def tells_truth (d : Day) : Prop := sorry -- Prop that Lev tells the truth on day d.
def lies (d : Day) : Prop := sorry -- Prop that Lev lies on day d.

axiom truth_or_lie (d : Day) : tells_truth d ∨ lies d
axiom not_both (d : Day) : ¬ (tells_truth d ∧ lies d)

theorem no_day_for_statements : 
  ∀ d : Day, ¬ (tells_truth d ∧ lies (yesterday d) ∧ lies d ∧ tells_truth (tomorrow d)) :=
sorry

end no_day_for_statements_l575_575606


namespace a_2016_eq_1_l575_575491

noncomputable def a : ℕ → ℕ
| 0 := 2
| 1 := 3
| (n + 2) := |a (n + 1) - a n|

-- Prove that a 2016 = 1 given the defined sequence
theorem a_2016_eq_1 : a 2015 = 1 :=
by
  sorry

end a_2016_eq_1_l575_575491


namespace abs_diff_count_S_1000_l575_575886

def τ (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, d ∣ n).card

def S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).sum τ

def is_odd (n : ℕ) : bool := 
  n % 2 = 1

def count_odd_S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).count (λ x, is_odd (S x))

def count_even_S (n : ℕ) : ℕ := 
  (n + 1) - count_odd_S n

theorem abs_diff_count_S_1000 : 
  |(count_odd_S 1000) - (count_even_S 1000)| = 7 :=
by
  sorry

end abs_diff_count_S_1000_l575_575886


namespace binomial_12_6_l575_575820

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575820


namespace screen_number_2018_l575_575733

theorem screen_number_2018 :
  let sequence : ℕ → ℤ := λ n,
    let rec f : ℕ → ℤ
      | 0      := 1  -- Initial number
      | 1      := 2  -- Second number
      | (n+2)  := (f n) + (f (n+1)) - 1
    in
    f n
  in
  sequence 2017 = 0 := 
by
  sorry

end screen_number_2018_l575_575733


namespace binom_12_6_l575_575794

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575794


namespace problem_conditions_l575_575845

noncomputable def f (x : ℝ) := Real.tan (x / 2 - Real.pi / 3)

theorem problem_conditions :
  (∀ x, f x ≠ f x → x ∉ { k : ℤ | x = 5 * Real.pi / 3 + 2 * k * Real.pi }) ∧
  (∀ ε > 0, ∃ p, p > 0 ∧ (λ t, f t = f (t + p)) ∧ p = 2 * Real.pi) ∧
  (∀ k : ℤ, monotone_on f (Ioo (-Real.pi / 3 + 2 * k * Real.pi) (5 * Real.pi / 3 + 2 * k * Real.pi))) ∧
  (∀ k : ℤ, ∃ x : ℝ, f x = 0 ∧ x = 2 * Real.pi / 3 + k * Real.pi) ∧
  (∀ k : ℤ, ∀ x, x ∈ Icc (Real.pi / 6 + 2 * k * Real.pi) (4 * Real.pi / 3 + 2 * k * Real.pi) → -1 ≤ f x ∧ f x ≤ Real.sqrt 3) :=
sorry

end problem_conditions_l575_575845


namespace construct_triangle_ABC_l575_575059

variables {Point : Type} [MetricSpace Point]

-- Definitions of Points A and D
variables (A D : Point)

-- Distances p and q
variables (p q : ℝ)

-- Definitions for points B and C
variables (B C : Point)

-- Definition of the function that checks angle between plane of triangle and projection plane
def plane_angle_with_projection_plane (triangle_plane : Set Point) (projection_plane : Set Point) : Prop :=
  sorry

-- Definitions for distances involving points A, B, C, and D
def distance (x y : Point) : ℝ := sorry

-- Assumptions related to the problem
axiom point_distance_1 : distance A B - distance B D = p
axiom point_distance_2 : distance A C - distance C D = q

-- Equivalent to plane forming the same angle with first image plane as the line AD
axiom plane_angle_condition : plane_angle_with_projection_plane ({A, B, C} : Set Point) ({D} : Set Point) = plane_angle_with_projection_plane ({A, D} : Set Point) ({D} : Set Point)

-- Triangle property that AD is the angle bisector at vertex A
axiom angle_bisector_property : is_angle_bisector A D B C

-- The theorem statement to be proven
theorem construct_triangle_ABC :
  ∃ (B C : Point),
    plane_angle_condition ∧
    angle_bisector_property ∧
    (distance A B - distance B D = p) ∧
    (distance A C - distance C D = q) :=
sorry

end construct_triangle_ABC_l575_575059


namespace greatest_integer_less_than_neg_seventeen_thirds_l575_575701

theorem greatest_integer_less_than_neg_seventeen_thirds : floor (-17 / 3) = -6 := by
  sorry

end greatest_integer_less_than_neg_seventeen_thirds_l575_575701


namespace binomial_12_6_eq_924_l575_575782

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575782


namespace valuing_fraction_l575_575892

variable {x y : ℚ}

theorem valuing_fraction (h : x / y = 1 / 2) : (x - y) / (x + y) = -1 / 3 :=
by
  sorry

end valuing_fraction_l575_575892


namespace binomial_12_6_eq_924_l575_575784

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575784


namespace calc_f_g_3_minus_g_f_3_l575_575117

def f (x : ℝ) : ℝ := 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2

theorem calc_f_g_3_minus_g_f_3 :
  (f (g 3) - g (f 3)) = -96 :=
by
  sorry

end calc_f_g_3_minus_g_f_3_l575_575117


namespace extremum_f_when_a_eq_1_div_e_range_of_a_when_equality_holds_l575_575957

-- Define the functions f and g
def f (x a : ℝ) := x - 1 - a * Real.log x
def g (x : ℝ) := Real.exp x / x

-- Statement for the first problem
theorem extremum_f_when_a_eq_1_div_e :
  ∃ x : ℝ, f x (1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

-- Statement for the second problem
theorem range_of_a_when_equality_holds :
  4 - (3 / 4) * Real.exp 4 ≤ (a : ℝ) → a < 0 → 
  (∀ (x1 x2 : ℝ), 4 ≤ x1 → x1 ≤ 5 → 4 ≤ x2 → x2 ≤ 5 → x1 ≠ x2 → 
  |f x1 a - f x2 a| < |g x1 - g x2|) :=
sorry

end extremum_f_when_a_eq_1_div_e_range_of_a_when_equality_holds_l575_575957


namespace henry_total_payment_l575_575432

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l575_575432


namespace binary_num_to_decimal_eq_51_l575_575389

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575389


namespace eccentricity_of_perpendicular_asymptotes_hyperbola_l575_575647

/--
  Given the hyperbola equation (x^2 / a^2 - y^2 / b^2 = 1)
  and the condition that its asymptotes are perpendicular to each other,
  prove that its eccentricity is sqrt(2).
-/
theorem eccentricity_of_perpendicular_asymptotes_hyperbola (a b : ℝ) (h : a = b) :
  let e := (sqrt (2)) in e = sqrt(2) :=
by
  sorry

end eccentricity_of_perpendicular_asymptotes_hyperbola_l575_575647


namespace monthly_savings_correct_l575_575595

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l575_575595


namespace real_m_of_complex_number_l575_575076

theorem real_m_of_complex_number (m : ℝ) : (m^2 + complex.i) * (1 + m * complex.i) ∈ ℝ → m = -1 :=
by
  sorry

end real_m_of_complex_number_l575_575076


namespace general_term_arithmetic_sequence_l575_575451

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end general_term_arithmetic_sequence_l575_575451


namespace exists_continuous_density_divergent_integral_l575_575722

noncomputable def characteristic_function (F : ℝ → ℝ) (t : ℝ) : ℂ :=
∫ x in ℝ, complex.exp (complex.I * t * x) * (dF x)

theorem exists_continuous_density_divergent_integral (F : ℝ → ℝ) (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, 0 ≤ F x) →
  (∀ x, ∃ f, continuous f ∧ F' = f) →
  (∃ t, |∫ x in ℝ, complex.exp (complex.I * t * x) * (dF x)| = ∞) :=
sorry

end exists_continuous_density_divergent_integral_l575_575722


namespace minimum_u_l575_575911

theorem minimum_u (a b c d : ℝ) (h : a * d + b * c = 1) : 
  let u := a^2 + b^2 + c^2 + d^2 + (a + c)^2 + (b - d)^2 in
  u >= 2 * real.sqrt 3 :=
sorry

end minimum_u_l575_575911


namespace kramer_vote_percentage_l575_575527

def percentage_of_votes_cast (K : ℕ) (V : ℕ) : ℕ :=
  (K * 100) / V

theorem kramer_vote_percentage (K : ℕ) (V : ℕ) (h1 : K = 942568) 
  (h2 : V = 4 * K) : percentage_of_votes_cast K V = 25 := 
by 
  rw [h1, h2, percentage_of_votes_cast]
  sorry

end kramer_vote_percentage_l575_575527


namespace projectiles_meet_in_90_minutes_l575_575969

theorem projectiles_meet_in_90_minutes
  (d : ℝ) (v1 : ℝ) (v2 : ℝ) (time_in_minutes : ℝ)
  (h_d : d = 1455)
  (h_v1 : v1 = 470)
  (h_v2 : v2 = 500)
  (h_time : time_in_minutes = 90) :
  d / (v1 + v2) * 60 = time_in_minutes :=
by
  sorry

end projectiles_meet_in_90_minutes_l575_575969


namespace next_perfect_square_l575_575960

theorem next_perfect_square (x : ℝ) (hx : ∃ k : ℤ, x = k^2) : ∃ y : ℝ, y = x + 4 * real.sqrt x + 4 :=
by
  sorry

end next_perfect_square_l575_575960


namespace eighty_th_number_is_18_l575_575202

theorem eighty_th_number_is_18 :
  (exists f : ℕ → ℕ, (∀ n, f n = 2 * n) ∧ ∃ m, ∑ i in range m, 2 * i + (80 - ∑ i in range m, 2 * i) = 18) :=
sorry

end eighty_th_number_is_18_l575_575202


namespace triangle_side_lengths_l575_575895

noncomputable def f (x m : ℝ) := x^3 - 3 * x + m

theorem triangle_side_lengths (m : ℝ) :
  (∀ a b c ∈ set.Icc (0 : ℝ) 2, a ≠ b ∧ b ≠ c ∧ c ≠ a →
    let fa := f a m; let fb := f b m; let fc := f c m in
    (fa + fb > fc) ∧ (fa + fc > fb) ∧ (fb + fc > fa)) ↔ m > 6 :=
by
  sorry

end triangle_side_lengths_l575_575895


namespace triangle_area_isosceles_right_l575_575986

theorem triangle_area_isosceles_right (BC : ℝ) (AB : ℝ) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) 
  (h1 : angle_B = 90) (h2 : angle_A = angle_C) (h3 : BC = 8) (h4 : AB = BC) :
  let area := (1 / 2) * AB * BC in
  area = 32 := 
by
  let area := (1 / 2) * AB * BC
  have h5 : AB = 8 := h3 ▸ h4
  have h6 : area = (1 / 2) * 8 * 8 := by rw [h5, h3]
  have h7 : area = 32 := by norm_num at h6
  exact h7

end triangle_area_isosceles_right_l575_575986


namespace find_divisor_l575_575717

theorem find_divisor :
  ∃ D, (23 = 5 * D + 3) ∧ (∃ N, N = 7 * D + 5) ∧ D = 4 :=
begin
  sorry
end

end find_divisor_l575_575717


namespace integer_solutions_of_inequality_system_l575_575624

theorem integer_solutions_of_inequality_system :
  {x : ℤ // 1 ≤ x ∧ x < 4} = ({1, 2, 3} : set ℤ) :=
by 
  sorry

end integer_solutions_of_inequality_system_l575_575624


namespace alice_walking_speed_l575_575310

theorem alice_walking_speed:
  ∃ v : ℝ, 
  (∀ t : ℝ, t = 1 → ∀ d_a d_b : ℝ, d_a = 25 → d_b = 41 - d_a → 
  ∀ s_b : ℝ, s_b = 4 → 
  d_b / s_b + t = d_a / v) ∧ v = 5 :=
by
  sorry

end alice_walking_speed_l575_575310


namespace infinite_terms_in_interval_l575_575569

-- Harmonic series definition
def harmonicSeries (n : ℕ) : ℝ := (List.range (n + 1)).sum (λ i => 1 / (i + 1 : ℝ))

-- Floor function definition
def floor (x : ℝ) : ℤ := Int.floor x

-- Statement of the theorem
theorem infinite_terms_in_interval (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : a < b) (h₂ : b ≤ 1) :
  ∃∞ n : ℕ, let S_n := harmonicSeries n in a < S_n - (floor S_n) ∧ S_n - (floor S_n) < b :=
sorry

end infinite_terms_in_interval_l575_575569


namespace log_problem_l575_575621

noncomputable def log_expression : ℝ := log 2 (1/4) + log 2 32

theorem log_problem :
  log_expression = 3 := by
  sorry

end log_problem_l575_575621


namespace find_leak_rate_l575_575581

-- Conditions in Lean 4
def pool_capacity : ℝ := 60
def hose_rate : ℝ := 1.6
def fill_time : ℝ := 40

-- Define the leak rate calculation
def leak_rate (L : ℝ) : Prop :=
  pool_capacity = (hose_rate - L) * fill_time

-- The main theorem we want to prove
theorem find_leak_rate : ∃ L, leak_rate L ∧ L = 0.1 := by
  sorry

end find_leak_rate_l575_575581


namespace memo_processing_orders_l575_575974

theorem memo_processing_orders :
  let T := {1, 2, 3, 4, 5, 6, 7, 8}
  let memos := T ∪ {11}
  ∑ j in finset.range 9, nat.choose 8 j * (j + 2) = 1536 :=
by
  sorry

end memo_processing_orders_l575_575974


namespace domain_shift_l575_575469

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- State the problem in Lean
theorem domain_shift (hf : ∀ x, f x ∈ domain_f) : 
    { x | 1 ≤ x ∧ x ≤ 2 } = { x | ∃ y, y ∈ domain_f ∧ x = y + 1 } :=
by
  sorry

end domain_shift_l575_575469


namespace problem_20_l575_575724

theorem problem_20 : 
  let sum := (∑ i in (Finset.range 2022), i)  -- Sum of integers from 0 to 2021
  in sum % 5 = 1 :=
by
  let sum := (∑ i in (Finset.range 2022), i)  -- Sum of integers from 0 to 2021
  have h1 : Fermat_little_theorem : ∀ (a : ℤ), a^5 ≡ a [ZMOD 5] := 
    sorry  -- Fermat's Little Theorem 

  -- Directly using property by Fermat's little theorem
  have h2 : ∑ i in (Finset.range 2022), (i^5 % 5) = ∑ i in (Finset.range 2022), (i % 5) := 
    sorry  -- From Fermat's Little Theorem

  -- The sum as per the problem's simplification in modular arithmetic
  calc
    sum ≡ 2021 * (2022 / 2) [MOD 5] : sorry -- The sum of the first 2021 positive integers
    ... ≡ 1 * 1011 [MOD 5]        : sorry  -- Simplification of individual terms modulo 5
    ... ≡ 1                       : by norm_num -- Final equivalent value

end problem_20_l575_575724


namespace beautiful_numbers_bound_l575_575999

/--
Let n > 1 be an integer.
A number is called beautiful if its square leaves an odd remainder upon division by n.
We need to prove that the number of consecutive beautiful numbers is less than or equal to 1 + ⌊sqrt(3n)⌋.
-/
theorem beautiful_numbers_bound (n : ℕ) (h1 : n > 1) :
  ∀ (beautiful : ℕ → Prop), 
    (∀ k, beautiful k ↔ (k ^ 2) % n % 2 = 1) → 
    ∀ seq, (∀ i, seq i → beautiful i) → 
      ∃ m ≤ 1 + nat.floor (real.sqrt (3 * n)), 
        ∀ (i j : ℕ), i ≠ j → seq i ≤ m → seq j ≤ m → false :=
sorry

end beautiful_numbers_bound_l575_575999


namespace quarter_circle_intersection_distance_l575_575088

theorem quarter_circle_intersection_distance (s : ℝ) :
  let A := (0, 0 : ℝ × ℝ)
  let B := (s, 0 : ℝ × ℝ)
  let eq1 := λ (p : ℝ × ℝ), p.1^2 + p.2^2 = s^2
  let eq2 := λ (p : ℝ × ℝ), (p.1 - s)^2 + p.2^2 = (2 * s)^2
  ∃ Y : ℝ × ℝ, eq1 Y ∧ eq2 Y ∧ Y.1 = s ∧ Y.2 = 0 :=
begin
  sorry,
end

end quarter_circle_intersection_distance_l575_575088


namespace initial_students_count_l575_575978

theorem initial_students_count (x : ℕ) 
  (h1 : ∃ y : ℕ, 0.4 * x = y)
  (h2 : ∃ z : ℕ, 0.25 * h1.some = z ∧ z = 30) :
  x = 300 := 
sorry

end initial_students_count_l575_575978


namespace value_of_x_sum_of_perimeters_sum_of_areas_l575_575721

open real

-- Define basic square structure and points of intersection
def square (a : ℝ) := {A1 A2 A3 A4 : ℝ × ℝ // true}

-- Define transformation point construction logic
def points_on_square (a x : ℝ) : (ℝ × ℝ) := sorry -- Define the exact transformation logic based on x

-- Define the octagon D1...D8 from intersections
def regular_octagon (a x : ℝ) (B C : (ℝ × ℝ)) := 
  let D := points_on_square a x in
  true -- logical transformation

-- Main theorem parts without proof
theorem value_of_x (a : ℝ) : ∃ x, regular_octagon a x B C ∧ x = a / 2 * (2 - sqrt 2) :=
by
  sorry

theorem sum_of_perimeters (a : ℝ) : 
  ∃ k, k = 8 * a * sqrt (2 - sqrt 2) :=
by
  sorry

theorem sum_of_areas (a : ℝ) : 
  ∃ t, t = (a^2 * (8 - 2 * sqrt 2)) / 7 :=
by
  sorry

end value_of_x_sum_of_perimeters_sum_of_areas_l575_575721


namespace binomial_12_6_l575_575814

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575814


namespace value_of_a2011_to_a2020_l575_575078

open Real

-- Define the sequence a_n
def a : ℕ → ℝ
  := sorry -- placeholder for the sequence definition

-- The first condition: logarithmic difference being 1
axiom log_diff (n : ℕ) : log (a (n + 1)) - log (a n) = 1

-- The second condition: sum from a_2001 to a_2010 equals 2015
axiom sum_condition : (∑ n in finset.range 10, a (2001 + n)) = 2015

-- The final statement to prove
theorem value_of_a2011_to_a2020 : 
  (∑ n in finset.range 10, a (2011 + n)) = 2015 * 10^10 :=
sorry

end value_of_a2011_to_a2020_l575_575078


namespace geom_prog_identical_l575_575679

theorem geom_prog_identical (n : ℕ) (h : ∀ (s : Finset ℝ), s.card = 4 → ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (b / a = c / b ∧ c / b = d / c)) :
  ∃ t : Finset ℝ, t.card ≥ n ∧ ∀ x ∈ t, ∃ k : ℕ, (Finset.filter (λ y, y = x) (Finset.range (4*n)) ≠ ∅) :=
by
  sorry

end geom_prog_identical_l575_575679


namespace henry_total_fee_8_bikes_l575_575433

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l575_575433


namespace area_of_OBEC_is_25_l575_575287

noncomputable def area_OBEC : ℝ :=
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let O := (0, 0)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  area_triangle O B E - area_triangle O E C

theorem area_of_OBEC_is_25 :
  area_OBEC = 25 := 
by
  sorry

end area_of_OBEC_is_25_l575_575287


namespace binomial_12_6_eq_924_l575_575833

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575833


namespace range_of_a_l575_575934

variable (a : ℝ)
variable (x : ℝ)

theorem range_of_a (h : ∀ x ∈ set.Icc 1 2, abs (2 ^ x - a) < abs (5 - 2 ^ x)) : 3 < a ∧ a < 5 :=
sorry

end range_of_a_l575_575934


namespace number_of_lists_l575_575997

-- Definitions for compositions and inversion number
def composition (n : ℕ) : Type := { l : List ℕ // l.sum = n ∧ ∀ a ∈ l, a > 0}

def inversion_number (l : List ℕ) : ℕ :=
  l.enum.filter (λ ⟨i, ai⟩, l.enum.filter (λ ⟨j, aj⟩, i < j ∧ ai > aj).length).length

-- Definitions for sets A and B
def A (n : ℕ) : finset (composition n) :=
  {l : composition n | (inversion_number l.val % 2 = 0)}

def B (n : ℕ) : finset (composition n) :=
  {l : composition n | (inversion_number l.val % 2 = 1)}

-- Theorem statement
theorem number_of_lists (n : ℕ) (h : n > 0) : 
  ∑ l in A n, 1 + ∑ l in B n, 1 = 2^(n-1) → 
  ∑ l in A n, 1 = 2^(n-2) + 2^(n/2 - 1):=
sorry

end number_of_lists_l575_575997


namespace unique_solution_f_satisfies_inequality_l575_575422

theorem unique_solution_f_satisfies_inequality :
  ∃ f : ℝ → ℝ, (∀ x y z : ℝ, (f (x * y) + f (x * z)) / 2 - f x * f (y * z) ≥ 1/4) ∧ (∀ x : ℝ, f x = 1/2) :=
by
  let f := (λ x : ℝ, 1 / 2)
  use f
  split
  { intros x y z
    sorry },
  { intro x
    sorry }

end unique_solution_f_satisfies_inequality_l575_575422


namespace not_divisible_by_121_l575_575156

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 3 * n + 5)) :=
by
  sorry

end not_divisible_by_121_l575_575156


namespace missing_number_in_proportion_l575_575094

/-- Given the proportion 2 : 5 = x : 3.333333333333333, prove that the missing number x is 1.3333333333333332 -/
theorem missing_number_in_proportion : ∃ x, (2 / 5 = x / 3.333333333333333) ∧ x = 1.3333333333333332 :=
  sorry

end missing_number_in_proportion_l575_575094


namespace approximate_root_l575_575511

def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

theorem approximate_root :
  f 1 = -2 ∧
  f 1.5 = 0.625 ∧
  f 1.25 = -0.984 ∧
  f 1.375 = -0.26 ∧
  f 1.4375 = 0.162 ∧
  f 1.40625 = -0.054 →
  abs (1.42 - 1.40625) < 0.05 :=
by sorry

end approximate_root_l575_575511


namespace value_of_a_sq_sub_b_sq_l575_575504

theorem value_of_a_sq_sub_b_sq (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 :=
by
  sorry

end value_of_a_sq_sub_b_sq_l575_575504


namespace find_acute_angle_l575_575534

theorem find_acute_angle
  (m n : Line)
  (T1 T2 : Transversal)
  (h_parallel : m ∥ n)
  (h_T1_m : angle_with_line T1 m = 40)
  (h_T2_n : angle_with_line T2 n = 50) :
  acute_angle_between_transversals T1 T2 = 10 := 
sorry

end find_acute_angle_l575_575534


namespace savings_plan_l575_575598

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l575_575598


namespace proof_problem_l575_575061

variables (x y C : ℝ) (a b c A B: ℝ)

def m : ℝ × ℝ := (2 * cos x, y - 2 * sqrt 3 * sin x * cos x)
def n : ℝ × ℝ := (1, cos x)

-- Condition: m is parallel to n
def parallel (u v : ℝ × ℝ) : Prop := ∃ (k : ℝ), u = (k * v.1, k * v.2)

-- Function f(x) definition based on the solution step (I)
def f (x : ℝ) : ℝ := 2 * cos x * cos x + 2 * sqrt 3 * sin x * cos x

-- Given condition for (II)
def g (C : ℝ) : ℝ := f (C / 2)

-- Law of sines relationship
def sine_law_range (A C : ℝ) : set ℝ := (λ a b c : ℝ, 2 * sin (A + π / 6)) '' {1, 2}

theorem proof_problem :
  parallel m n → 
  g C = 3 → 
  (g (π / 3) = 3 ∧ 
  ∀ a b c A B, C = π / 3 → (A ∈ (0, 2 * π / 3)) → ((a + b) / c ∈ (1,2))) :=
by sorry

end proof_problem_l575_575061


namespace production_rate_l575_575260

theorem production_rate (machines_produce_240_bottles_per_minute : 6 * x = 240)
                        (constant_rate : ∀ (n : ℕ), machines_produce_n_bottles_per_minute (n * x)):
                        10 * x * 4 = 1600 :=
by
  -- Given that 6 machines produce 240 bottles per minute
  have h1 : x = 40 := by
    (calc
      6 * x = 240 : machines_produce_240_bottles_per_minute
      x = 240 / 6 : by field_simp
      x = 40 : by norm_num)

  -- Since the machines work at a constant rate, 10 machines produce
  -- 10 * 40 = 400 bottles per minute
  have h2 : 10 * x = 400 := by
    (calc
      10 * x = 10 * 40 : by rw h1
      10 * 40 = 400 : by norm_num)

  -- In 4 minutes, 10 machines produce 400 * 4 = 1600 bottles
  show 10 * x * 4 = 1600 from by
    (calc
      10 * x * 4 = 400 * 4 : by rw h2
      400 * 4 = 1600 : by norm_num)

end production_rate_l575_575260


namespace abs_eq_solutions_l575_575864

theorem abs_eq_solutions (x : ℝ) (hx : |x - 5| = 3 * x + 6) :
  x = -11 / 2 ∨ x = -1 / 4 :=
sorry

end abs_eq_solutions_l575_575864


namespace perpendicular_vectors_l575_575235

theorem perpendicular_vectors (b : ℝ) :
  let v1 := ![-6, 2]
      v2 := ![b, 3]
  in v1 ⬝ v2 = 0 → b = 1 := 
by
  sorry

end perpendicular_vectors_l575_575235


namespace tank_fraction_l575_575303

theorem tank_fraction (hA : 15 > 0) (hB : 6 > 0) (hAB : 2 > 0) (net_rate : (1/15 - 1/6) = -1/10) : 
  let amount := -1 * -1/10 * 2 in 
  amount = (1 - 1/5) := 
by
  let x := 1 - 1/5
  have hx : x = 4/5 := by 
    calc 
      1 - 1/5 = (5 - 1)/5  : by norm_num
      ... = 4/5             : by norm_num
  sorry

end tank_fraction_l575_575303


namespace wendy_total_glasses_l575_575244

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l575_575244


namespace jake_total_earnings_l575_575988

-- Define the number of snakes
def num_vipers := 2
def num_cobras := 3
def num_pythons := 1

-- Define the number of eggs laid by each type of snake
def eggs_per_viper := 3
def eggs_per_cobra := 2
def eggs_per_python := 4

-- Define sale prices for the baby snakes
def price_viper := 300
def price_cobra := 250
def price_python := 450

-- Define the discounts for the baby snakes
def discount_viper := 0.10
def discount_cobra := 0.05
def discount_python := 0.00

-- Sum the adjusted earnings
noncomputable def total_earnings : ℤ :=
  let num_baby_vipers := num_vipers * eggs_per_viper
  let num_baby_cobras := num_cobras * eggs_per_cobra
  let num_baby_pythons := num_pythons * eggs_per_python
  let earnings_vipers := (price_viper - price_viper * discount_viper) * num_baby_vipers
  let earnings_cobras := (price_cobra - price_cobra * discount_cobra) * num_baby_cobras
  let earnings_pythons := price_python * num_baby_pythons
  (earnings_vipers + earnings_cobras + earnings_pythons).toInt

theorem jake_total_earnings :
  total_earnings = 4845 :=
by
  sorry

end jake_total_earnings_l575_575988


namespace find_smallest_n_l575_575216

def a : ℕ → ℕ
| 1 := 1
| (2 * n) := if even n then a n else 2 * a n
| (2 * n + 1) := if even n then 2 * a n + 1 else a n

-- Define the smallest function to find the smallest n where the requirement holds.
def smallest (P : ℕ → Prop) : ℕ := 
  Nat.find (exists.elim (exists_unique.elim_left (ExistsUnique.intro 1 P sorry)))

-- The actual proof problem
theorem find_smallest_n : smallest (λ n, a n = a 2017) = 5 := 
sorry

end find_smallest_n_l575_575216


namespace problem_statement_l575_575093

-- Definitions of sequences {a_n} and {b_n}
def a (n : ℕ) : ℕ := 2 ^ n

def b (n : ℕ) : ℕ := 12 * n - 28

-- Sum of the sequence {a_n * b_n}
def S (n : ℕ) : ℕ := (3 * n - 10) * 2^(n + 3) - 80

-- Main theorem to prove the formulated proof problem
theorem problem_statement (n : ℕ) :
  ∑ k in finset.range n, (a k * b k) = S n := 
sorry

end problem_statement_l575_575093


namespace calculate_product_l575_575336

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l575_575336


namespace log5_one_div_25_eq_neg2_l575_575415

theorem log5_one_div_25_eq_neg2 : log 5 (1/25) = -2 := 
by
  -- Let's denote the logarithm by x for convenience
  let x := log 5 (1/25)
  -- By definition of logarithm, we have 5^x = 1/25
  have h1 : 5^x = (1/25) := by sorry
  -- We also know that 1/25 = 5^-2
  have h2 : (1/25) = 5^(-2) := by sorry
  -- Thus, 5^x = 5^-2
  have h3 : 5^x = 5^(-2) := by
    rw [←h2] at h1
    exact h1
  -- Equating the exponents of the same base, we get x = -2
  have h4 : x = -2 := by
    apply eq_of_pow_eq_pow
    exact h3
  -- Therefore, log 5 (1/25) = -2
  exact h4
  sorry

end log5_one_div_25_eq_neg2_l575_575415


namespace cost_price_l575_575749

-- Define the conditions: selling price (SP) and profit percentage
variable (SP : ℝ) (profit_percentage : ℝ)

-- Given conditions
def SP_value := SP = 1110
def profit_percentage_value := profit_percentage = 0.20

-- The main statement to be proved: cost price (CP)
def find_CP (SP profit_percentage : ℝ) : ℝ := SP / (1 + profit_percentage)

theorem cost_price (h1 : SP_value) (h2 : profit_percentage_value) : find_CP SP profit_percentage = 925 := 
  by 
  rw [SP_value, profit_percentage_value]
  -- Proof omitted
  sorry

end cost_price_l575_575749


namespace spiral_grid_third_row_sum_l575_575972

theorem spiral_grid_third_row_sum :
  let n := 12 
  let grid := array n (array n ℕ)
  -- Assume a function to generate the spiral grid
  let generate_spiral_grid : ℕ → array n (array n ℕ) := sorry
  -- Fill the grid with numbers 1 to n*n in a spiral order
  let spiral := generate_spiral_grid n
  -- Extract the third row
  let third_row := spiral[2]
  -- Find the least and greatest numbers in the third row
  let least_number_in_third_row := min third_row
  let greatest_number_in_third_row := max third_row
  -- Calculate the sum of these two numbers
  let sum := least_number_in_third_row + greatest_number_in_third_row
  sum = 55 :=
by
  sorry

end spiral_grid_third_row_sum_l575_575972


namespace min_students_in_class_l575_575518

-- Define the problem conditions
variables {b g : ℕ} -- number of boys (b) and number of girls (g)

-- Condition: three-fourths of the boys passed the test equals half of the girls passed the test
def condition1 := (3 / 4) * b = (1 / 2) * g

-- The proof problem statement in Lean
theorem min_students_in_class : condition1 → b + g = 5 :=
begin
  sorry
end

end min_students_in_class_l575_575518


namespace binom_12_6_l575_575790

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575790


namespace min_value_of_x_plus_y_l575_575939

theorem min_value_of_x_plus_y (x y : ℝ) (h1: y ≠ 0) (h2: 1 / y = (x - 1) / 2) : x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_of_x_plus_y_l575_575939


namespace divisible_by_7_and_either_2_or_5_l575_575065

theorem divisible_by_7_and_either_2_or_5 (S A B : Set ℕ) (h_S : S = { n | n ∈ Finset.range 301 }) 
  (h_A : A = { x | x ∈ S ∧ x % 14 = 0 }) (h_B : B = { x | x ∈ S ∧ x % 35 = 0 }) : 
  Finset.card (A ∪ B) = 25 := by
  let S := Finset.range 301
  let A := S.filter (λ x, x % 14 = 0)
  let B := S.filter (λ x, x % 35 = 0)
  let lcm_A_B := S.filter (λ x, x % 70 = 0)
  have h_A_card : A.card = 21 := by norm_num; sorry
  have h_B_card : B.card = 8 := by norm_num; sorry
  have h_lcm_A_B_card : lcm_A_B.card = 4 := by norm_num; sorry
  rw [Finset.card_union_eq h_A h_B lcm_A_B, h_A_card, h_B_card, h_lcm_A_B_card]
  have lcm_divisible : Finset.card (A ∪ B) = A.card + B.card - lcm_A_B.card := by sorry
  rw lcm_divisible
  norm_num


end divisible_by_7_and_either_2_or_5_l575_575065


namespace binom_12_6_l575_575787

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575787


namespace wendy_total_glasses_l575_575245

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end wendy_total_glasses_l575_575245


namespace sarah_score_l575_575062

theorem sarah_score
  (hunter_score : ℕ)
  (john_score : ℕ)
  (grant_score : ℕ)
  (sarah_score : ℕ)
  (h1 : hunter_score = 45)
  (h2 : john_score = 2 * hunter_score)
  (h3 : grant_score = john_score + 10)
  (h4 : sarah_score = grant_score - 5) :
  sarah_score = 95 :=
by
  sorry

end sarah_score_l575_575062


namespace even_f_min_g_of_interval_l575_575966

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x else x^2 - 2*x

def g (x a : ℝ) : ℝ :=
  f x + (4 - 2 * a) * x + 2

def h (a : ℝ) : ℝ :=
  if a ≤ 2 then 5 - 2 * a
  else if a < 3 then -a^2 + 2*a + 1
  else 10 - 4*a

theorem even_f :
  ∀ x : ℝ, f x = f (-x) := 
sorry

theorem min_g_of_interval :
  ∀ a : ℝ, ∃ x ∈ set.Icc (1 : ℝ) (2 : ℝ), g x a = h a :=
sorry

end even_f_min_g_of_interval_l575_575966


namespace volume_of_cut_and_fold_box_l575_575319

theorem volume_of_cut_and_fold_box (y : ℝ) (h_pos_y : y > 0) (h_dim_y : y < 6):
  let length := 15 - 2 * y,
      width := 12 - 2 * y,
      height := y,
      volume := length * width * height 
  in volume = 4 * y^3 - 54 * y^2 + 180 * y := 
by 
  let length := 15 - 2 * y
  let width := 12 - 2 * y
  let height := y
  let volume := length * width * height
  have h1 : volume = y * (15 - 2 * y) * (12 - 2 * y), by 
    sorry 
  have h2 : volume = 4 * y^3 - 54 * y^2 + 180 * y, by 
    sorry
  exact h2

end volume_of_cut_and_fold_box_l575_575319


namespace well_depth_and_rope_length_l575_575631

variables (h x : ℝ)

theorem well_depth_and_rope_length :
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) → True := 
by
  intro h_eq x_eq
  sorry

end well_depth_and_rope_length_l575_575631


namespace parallelogram_property_l575_575547

variables {A B C D K M P : Type} [AddCommGroup K] [Module ℝ K]
variables (A B C D K M P : K)

-- Defining parrallelogram ABCD,  with point K inside it
def is_parallelogram (A B C D : K) : Prop := 
(∀ t : ℝ, t • (B - A) + A = t • (C - D) + D) ∧ (A - B = D - C)

-- M is the midpoint of BC
def is_midpoint (M B C : K) : Prop := 
(2 : ℝ) • M = B + C

-- P is the midpoint of KM
def is_midpoint_KM (P K M : K) : Prop := 
(2 : ℝ) • P = K + M

-- Given angle conditions
def angle_condition (A B C D P : K) : Prop := 
(∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * A + b * B = 0 ∧ a * C + b * D = 0)

theorem parallelogram_property
  (A B C D K M P : K)
  (h_parallelogram : is_parallelogram A B C D)
  (h_midpoint_BC : is_midpoint M B C)
  (h_midpoint_KM : is_midpoint_KM P K M)
  (h_angle_condition: angle_condition A B C D P)
  : A - K = D - K :=
sorry

end parallelogram_property_l575_575547


namespace range_of_omega_l575_575046

open Real

noncomputable def sine_function (ω x : ℝ) : ℝ := sin (ω * x + (π / 4))

theorem range_of_omega 
  (h : ∀ x ∈ Set.Ioo (π / 12) (π / 3), ∃ M : ℝ, is_max (sine_function ω x) M ∧ ¬is_min (sine_function ω x) M) 
  (ω_pos : ω > 0) : 
  ω > 3 / 4 ∧ ω < 3 :=
sorry

end range_of_omega_l575_575046


namespace baba_yagas_savings_plan_l575_575602

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l575_575602


namespace triangle_area_l575_575033

noncomputable def area_of_triangle := 
  let a := 4
  let b := 5
  let c := 6
  let cosA := 3 / 4
  let sinA := Real.sqrt (1 - cosA ^ 2)
  (1 / 2) * b * c * sinA

theorem triangle_area :
  ∃ (a b c : ℝ), a = 4 ∧ b = 5 ∧ c = 6 ∧ 
  a < b ∧ b < c ∧ 
  -- Additional conditions
  (∃ A B C : ℝ, C = 2 * A ∧ 
   Real.cos A = 3 / 4 ∧ 
   Real.sin A * Real.cos A = sinA * cosA ∧ 
   0 < A ∧ A < Real.pi ∧ 
   (1 / 2) * b * c * sinA = (15 * Real.sqrt 7) / 4) :=
by
  sorry

end triangle_area_l575_575033


namespace at_least_two_primes_are_equal_l575_575111

open Nat

theorem at_least_two_primes_are_equal (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (prime_k : Prime (b^c + a)) (prime_l : Prime (a^b + c)) (prime_m : Prime (c^a + b)) : 
  (b^c + a = a^b + c) ∨ (a^b + c = c^a + b) ∨ (c^a + b = b^c + a) :=
by
  sorry

end at_least_two_primes_are_equal_l575_575111


namespace max_third_term_is_16_l575_575667

-- Define the arithmetic sequence conditions
def arithmetic_seq (a d : ℕ) : list ℕ := [a, a + d, a + 2 * d, a + 3 * d]

-- Define the sum condition
def sum_of_sequence_is_50 (a d : ℕ) : Prop :=
  (a + a + d + a + 2 * d + a + 3 * d) = 50

-- Define the third term of the sequence
def third_term (a d : ℕ) : ℕ := a + 2 * d

-- Prove that the greatest possible third term is 16
theorem max_third_term_is_16 : ∃ (a d : ℕ), sum_of_sequence_is_50 a d ∧ third_term a d = 16 :=
by
  sorry

end max_third_term_is_16_l575_575667


namespace part1_min_value_y_part2_compare_part3_min_value_M_l575_575264

theorem part1_min_value_y (x : ℝ) (h1 : x > -1):
  ∀ x, (y = (x+2)*(x+3) / (x+1) → y ≥ 2 * real.sqrt(2) + 3)
  ∧ (x = real.sqrt(2)-1 → y = 2 * real.sqrt(2) + 3) :=
sorry

theorem part2_compare (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) 
    (h_eq : x^2 / a^2 - y^2 / b^2 = 1) :
  a^2 - b^2 ≤ (x - y)^2 ∧ (a^2 - b^2 = (x - y)^2 ↔ (b^2 * x^2 / a^2 = a^2 * y^2 / b^2 ∧ x * y ≥ 0)) :=
sorry

theorem part3_min_value_M (m : ℝ) (h1 : m ≥ 1) :
  ∀ x = real.sqrt(4 * m - 3),
  ∀ y = real.sqrt(m - 1),
  (M = x - y → M ≥ real.sqrt(3) / 2) ∧ (m = 13 / 12 → M = real.sqrt(3) / 2) :=
sorry

end part1_min_value_y_part2_compare_part3_min_value_M_l575_575264


namespace binom_12_6_l575_575804

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575804


namespace correct_survey_is_B_correct_survey_is_B_from_options_l575_575316

def isCensusSurvey (description : String) : Bool :=
  description = "Understanding the math unit test scores of classmates"

theorem correct_survey_is_B :
  ∃ d, d = "Understanding the math unit test scores of classmates" ∧ isCensusSurvey d := by
  exists "Understanding the math unit test scores of classmates"
  split
  . refl
  . exact rfl

-- Additional conditions describing the options, though not directly used in the proof
def optionA := "Understanding the weekly allowance situation of classmates"
def optionB := "Understanding the math unit test scores of classmates"
def optionC := "Understanding how much TV classmates watch each week"
def optionD := "Understanding how much extracurricular reading classmates do each week"

-- Final link to ensure consistency from conditions to proofs
theorem correct_survey_is_B_from_options :
  optionB = "Understanding the math unit test scores of classmates" → 
  optionB = "Understanding the math unit test scores of classmates" := by
  intro h
  exact h
  
#check correct_survey_is_B
#check correct_survey_is_B_from_options

end correct_survey_is_B_correct_survey_is_B_from_options_l575_575316


namespace bronze_needed_l575_575583

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l575_575583


namespace count_multiples_of_4_between_100_and_350_l575_575946

theorem count_multiples_of_4_between_100_and_350 : 
  (∃ n : ℕ, 104 + (n - 1) * 4 = 348) ∧ (∀ k : ℕ, (104 + k * 4 ∈ set.Icc 100 350) ↔ (k ≤ 61)) → 
  n = 62 :=
by
  sorry

end count_multiples_of_4_between_100_and_350_l575_575946


namespace KLMN_is_square_l575_575605

-- Assuming a 2D point structure
structure Point2D where
  x : ℝ
  y : ℝ

-- Definition of a square by its vertices
structure Square where
  A B C D : Point2D
  is_square : (B.x - A.x = D.x - C.x) ∧ (B.y - A.y = D.y - C.y) ∧ (B.x - A.x) * (B.x - A.x) + (B.y - A.y) * (B.y - A.y) = (C.x - B.x) * (C.x - B.x) + (C.y - B.y) * (C.y - B.y)

-- Definition of points N, K, L, M dividing sides AB, BC, CD, DA in the same ratio p : (1 - p)
structure PointsOnSquare where
  AB BC CD DA : Square
  N K L M : Point2D
  p : ℝ
  ratio_cond : (0 < p) ∧ (p < 1) ∧ 
    (N.x = A.x + p * (B.x - A.x)) ∧ (N.y = A.y + p * (B.y - A.y)) ∧
    (K.x = B.x + p * (C.x - B.x)) ∧ (K.y = B.y + p * (C.y - B.y)) ∧
    (L.x = C.x + p * (D.x - C.x)) ∧ (L.y = C.y + p * (D.y - C.y)) ∧
    (M.x = D.x + p * (A.x - D.x)) ∧ (M.y = D.y + p * (A.y - D.y))

-- Definition to check if quadrilateral KLMN is a square
def isSquare (N K L M : Point2D) : Prop :=
  let distance := (N.x - K.x) * (N.x - K.x) + (N.y - K.y) * (N.y - K.y)
  (N.x - K.x = M.x - L.x) ∧ (N.y - K.y = M.y - L.y) ∧
  (K.x - L.x = N.x - M.x) ∧ (K.y - L.y = N.y - M.y) ∧
  (distance = (K.x - L.x) * (K.x - L.x) + (K.y - L.y) * (K.y - L.y))

theorem KLMN_is_square (sq : Square) (pts : PointsOnSquare sq) : isSquare pts.N pts.K pts.L pts.M := sorry

end KLMN_is_square_l575_575605


namespace inequality_example_l575_575619

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l575_575619


namespace inequality_of_function_l575_575165

theorem inequality_of_function (x : ℝ) : 
  (1 / 2 : ℝ) ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 / 2 : ℝ) :=
sorry

end inequality_of_function_l575_575165


namespace net_weekly_income_change_l575_575770

-- Definition of constants and conditions
def raise_per_hour : ℝ := 0.50
def hours_per_week : ℝ := 40
def additional_gross_income_per_week : ℝ := raise_per_hour * hours_per_week
def federal_tax_rate_before : ℝ := 0.15
def federal_tax_rate_after : ℝ := 0.20
def state_tax_rate : ℝ := 0.05
def social_security_deduction : ℝ := 0.062
def medicare_deduction : ℝ := 0.0145
def 401k_contribution_before : ℝ := 0.03
def 401k_contribution_after : ℝ := 0.04
def housing_benefit_reduction_per_month : ℝ := 60
def housing_benefit_reduction_per_week : ℝ := housing_benefit_reduction_per_month / 4

-- Prove the net weekly income change
theorem net_weekly_income_change :
  let
    new_federal_tax_deduction_per_week := federal_tax_rate_after * additional_gross_income_per_week
    new_state_tax_deduction_per_week := state_tax_rate * additional_gross_income_per_week
    new_social_security_deduction_per_week := social_security_deduction * additional_gross_income_per_week
    new_medicare_deduction_per_week := medicare_deduction * additional_gross_income_per_week
    new_401k_contribution_per_week := 401k_contribution_after * additional_gross_income_per_week
    total_new_deductions_per_week := new_federal_tax_deduction_per_week + 
                                    new_state_tax_deduction_per_week + 
                                    new_social_security_deduction_per_week + 
                                    new_medicare_deduction_per_week + 
                                    new_401k_contribution_per_week
    net_increase_in_weekly_income := additional_gross_income_per_week - total_new_deductions_per_week - housing_benefit_reduction_per_week
  in net_increase_in_weekly_income = -2.33 :=
by
  sorry

end net_weekly_income_change_l575_575770


namespace find_MN_l575_575985

variables (V : Type) [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (A B C D N M : V)

def is_parallelogram (A B C D : V) : Prop :=
  B - A = D - C ∧ C - B = D - A

-- Given conditions
variables (h_parallelogram : is_parallelogram A B C D)
variables (h_AB : B - A = a)
variables (h_AC : C - A = b)
variables (h_NC : N - C = (1/4) • b)
variables (h_BM : B - M = (1/2) • (b - a))

-- The target vector MN
def MN : V := M - N

-- Statement to be proved
theorem find_MN : MN a b B M N = (1/2) • a - (3/4) • b :=
sorry

end find_MN_l575_575985


namespace find_weekday_rate_l575_575181

-- Definitions of given conditions
def num_people : ℕ := 6
def days_weekdays : ℕ := 2
def days_weekend : ℕ := 2
def weekend_rate : ℕ := 540
def payment_per_person : ℕ := 320

-- Theorem to prove the weekday rental rate
theorem find_weekday_rate (W : ℕ) :
  (num_people * payment_per_person) = (days_weekdays * W) + (days_weekend * weekend_rate) →
  W = 420 :=
by 
  intros h
  sorry

end find_weekday_rate_l575_575181


namespace negation_of_proposition_l575_575910

theorem negation_of_proposition:
  (∀ x : ℝ, x ≥ 0 → x - 2 > 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x - 2 ≤ 0) := 
sorry

end negation_of_proposition_l575_575910


namespace simplify_trig_expr_l575_575169

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- State the theorem we need to prove
theorem simplify_trig_expr : 
  sin (x + y) * sin (x - y) - cos (x + y) * cos (x - y) = - cos (2 * x) :=
sorry

end simplify_trig_expr_l575_575169


namespace compute_AD_squared_l575_575532

-- Definitions for the conditions
variables (A B C D E F : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F]
variables (AD AE ED BC : ℝ) (α β γ : ℝ)

-- Provided conditions in the problem
def ratio_AE_ED : Prop := AE / ED = 1 / 9
def right_angle_BEC : Prop := α = 90 
def area_relation : Prop := (1 / 2 * ED * BC * sin β) = 27 * (1 / 2 * AE * BC * sin γ)
def equal_angles_EBC_EAB : Prop := α = γ
def equal_angles_ECB_EDC : Prop := β = γ
def BC_value : Prop := BC = 6

-- Final statement to compute AD^2
theorem compute_AD_squared (h1 : ratio_AE_ED AE ED)
                           (h2 : right_angle_BEC α)
                           (h3 : area_relation AE ED β γ)
                           (h4 : equal_angles_EBC_EAB α γ)
                           (h5 : equal_angles_ECB_EDC β γ)
                           (h6 : BC_value BC) :
                           AD^2 = 1620 :=
sorry -- Proof omitted

end compute_AD_squared_l575_575532


namespace triangle_orthocenter_angles_l575_575129

theorem triangle_orthocenter_angles (A B C H D E F X M : Point)
  (h1 : is_triangle ABC)
  (h2 : orthocenter H ABC)
  (h3 : foot D A BC)
  (h4 : foot E B CA)
  (h5 : foot F C AB)
  (h6 : parallel (line_through A C) (line_through B X))
  (h7 : intersects (line_through E F) (line_through B X) X)
  (h8 : midpoint M A B) :
  ∠ACM = ∠XDB := 
sorry

end triangle_orthocenter_angles_l575_575129


namespace monthly_savings_correct_l575_575592

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l575_575592


namespace binom_12_6_eq_924_l575_575808

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575808


namespace find_ratio_m_plus_2n_l575_575195

-- Given definitions: triangle ABC is equilateral with side length 2
def triangleABC := ∃ (A B C : ℝ×ℝ),
  let dist := λ (p q : ℝ×ℝ), Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  dist A B = 2 ∧ dist B C = 2 ∧ dist A C = 2

-- Point D lies on ray BC such that CD = 4
def pointD (B C D : ℝ×ℝ) : Prop :=
  let dist := λ (p q : ℝ×ℝ), Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  D.1 > C.1 ∧ D.2 = C.2 ∧ dist C D = 4

-- Points E and F lie on AB and AC respectively and are such that E, F and D are collinear
def pointsEF (A B C D E F : ℝ×ℝ) : Prop :=
  let on_line := λ (p q r : ℝ×ℝ), (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1) in
  on_line E D F ∧
  let dist := λ (p q : ℝ×ℝ), Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  dist A E ≤ dist A B ∧ dist A F ≤ dist A C

-- Area of triangle AEF is half the area of triangle ABC
def half_area (A B C E F : ℝ×ℝ) (areaABC areaAEF : ℝ) : Prop :=
  let area := λ (P Q R : ℝ×ℝ), Real.abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2 in
  area A B C = areaABC ∧ area A E F = areaAEF ∧ areaAEF = areaABC / 2

theorem find_ratio_m_plus_2n : ∃ (A B C D E F : ℝ×ℝ), 
  triangleABC ∧ pointD B C D ∧ pointsEF A B C D E F ∧
  ∃ (areaABC areaAEF : ℝ), half_area A B C E F areaABC areaAEF ∧ 
  ∃ (m n : ℕ), 
    let AE := λ (dist : ℝ → ℝ → ℝ), dist A E in
    let AF := λ (dist : ℝ → ℝ → ℝ), dist A F in
    (m > 0 ∧ n > 0 ∧ Nat.gcd m n = 1 ∧ ratio AE AF = m / n ∧ m + 2 * n = 4) :=
  sorry

end find_ratio_m_plus_2n_l575_575195


namespace percentage_increase_in_consumption_l575_575222

-- Define the conditions
variables {T C : ℝ}  -- T: original tax, C: original consumption
variables (P : ℝ)    -- P: percentage increase in consumption

-- Non-zero conditions
variables (hT : T ≠ 0) (hC : C ≠ 0)

-- Define the Lean theorem
theorem percentage_increase_in_consumption 
  (h : 0.8 * (1 + P / 100) = 0.96) : 
  P = 20 :=
by
  sorry

end percentage_increase_in_consumption_l575_575222


namespace binom_12_6_l575_575791

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575791


namespace trigonometric_expression_value_l575_575022

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  2 * (Real.sin α)^2 + 4 * Real.sin α * Real.cos α - 9 * (Real.cos α)^2 = 21 / 10 :=
by
  sorry

end trigonometric_expression_value_l575_575022


namespace chord_length_l575_575446

def ellipse : set (ℝ × ℝ) :=
  { p | let (x, y) := p in (x^2 / 4) + y^2 = 1 }

def line (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p in (x, x - sqrt 3)

theorem chord_length :
  let p1 := (4 * sqrt 3 + 2 * sqrt 2) / 5
  let p2 := (4 * sqrt 3 - 2 * sqrt 2) / 5
  let a : ℝ × ℝ := (p1, p1 - sqrt 3)
  let b : ℝ × ℝ := (p2, p2 - sqrt 3)
  a ∈ ellipse ∧ b ∈ ellipse →
  dist a b = 8 / 5 :=
by
  sorry

end chord_length_l575_575446


namespace find_cos_A_l575_575980

theorem find_cos_A
  (A C : ℝ)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (α : ℝ)
  (h1 : A = C)
  (h2 : AB = 150)
  (h3 : CD = 150)
  (h4 : AD ≠ BC)
  (h5 : AB + BC + CD + AD = 560)
  (h6 : A = α)
  (h7 : C = α)
  (BD₁ BD₂ : ℝ)
  (h8 : BD₁^2 = AD^2 + 150^2 - 2 * 150 * AD * Real.cos α)
  (h9 : BD₂^2 = BC^2 + 150^2 - 2 * 150 * BC * Real.cos α)
  (h10 : BD₁ = BD₂) :
  Real.cos A = 13 / 15 := 
sorry

end find_cos_A_l575_575980


namespace unique_twice_diff_fun_l575_575863

variable {f : ℝ → ℝ}

theorem unique_twice_diff_fun (h1 : ∀ x, deriv (deriv f) x = 0)
  (h2 : f 0 = 19)
  (h3 : f 1 = 99) :
  f = λ x, 80 * x + 19 := 
by 
  sorry

end unique_twice_diff_fun_l575_575863


namespace triangle_side_a_l575_575514

theorem triangle_side_a (a : ℝ) : 2 < a ∧ a < 8 → a = 7 :=
by
  sorry

end triangle_side_a_l575_575514


namespace binom_12_6_l575_575795

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575795


namespace tom_paid_264_l575_575231

noncomputable def discounted_price (original_price : ℤ) (discount_percentage : ℤ) : ℤ :=
  original_price - (original_price * discount_percentage / 100)

noncomputable def total_cost_before_tax (number_of_shirts : ℤ) (price_per_shirt : ℤ) : ℤ :=
  number_of_shirts * price_per_shirt

noncomputable def tax_amount (total_cost : ℤ) (tax_percentage : ℤ) : ℤ :=
  total_cost * tax_percentage / 100

theorem tom_paid_264 :
  let shirts_per_fandom := 5 in
  let number_of_fandoms := 4 in
  let original_price := 15 in
  let discount_percentage := 20 in
  let tax_percentage := 10 in
  let number_of_shirts := shirts_per_fandom * number_of_fandoms in
  let price_per_shirt := discounted_price original_price discount_percentage in
  let total_cost := total_cost_before_tax number_of_shirts price_per_shirt in
  let total := total_cost + tax_amount total_cost tax_percentage in
  total = 264 :=
by
  sorry

end tom_paid_264_l575_575231


namespace most_suitable_candidate_l575_575760

def S_A^2 : ℝ := 2.25
def S_B^2 : ℝ := 1.81
def S_C^2 : ℝ := 3.42

theorem most_suitable_candidate :
  S_B^2 < S_A^2 ∧ S_B^2 < S_C^2 :=
by
  -- Proof will be handled here
  sorry

end most_suitable_candidate_l575_575760


namespace curve_is_hyperbola_l575_575870

-- We define the given polar equation as a function.
noncomputable def polar_curve (r θ : ℝ) : ℝ :=
  3 * (Real.cot θ) * (Real.csc θ)

-- We state that the curve defined by 'polar_curve' is a hyperbola.
theorem curve_is_hyperbola (r θ : ℝ) : 
  polar_curve r θ = r 
  → -- Proving that this curve corresponds to a hyperbola in Cartesian coordinates
  is_hyperbola (Some_transform_to_Cartesian r θ) := 
sorry

end curve_is_hyperbola_l575_575870


namespace calculate_product_l575_575338

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l575_575338


namespace solution_count_in_interval_l575_575072

noncomputable def equation_solution_count : ℝ → ℝ := 
  λ θ, 4 + 2 * Real.cos θ - 3 * Real.sin (2 * θ)

theorem solution_count_in_interval : sorry := 
by
  sorry
  ⟨ 0 < θ, θ ≤ 2 * Real.pi ⟩
  sorry
  -- Conclude with: ∃ c (c.is_correct_answer=2)

end solution_count_in_interval_l575_575072


namespace smallest_student_count_l575_575768

theorem smallest_student_count (x y z w : ℕ) 
  (ratio12to10 : x / y = 3 / 2) 
  (ratio12to11 : x / z = 7 / 4) 
  (ratio12to9 : x / w = 5 / 3) : 
  x + y + z + w = 298 :=
by
  sorry

end smallest_student_count_l575_575768


namespace no_distinct_power_sums_on_board_l575_575528

theorem no_distinct_power_sums_on_board :
  ¬(∃ (a b c d p q r s : ℕ) (M : fin 4 → fin 4 → ℤ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧ 
    (a ≠ p ∧ a ≠ q ∧ a ≠ r ∧ a ≠ s ∧ b ≠ p ∧ b ≠ q ∧ b ≠ r ∧ b ≠ s ∧ 
     c ≠ p ∧ c ≠ q ∧ c ≠ r ∧ c ≠ s ∧ d ≠ p ∧ d ≠ q ∧ d ≠ r ∧ d ≠ s) ∧ 
    (2^a + 2^b + 2^c + 2^d = 2^p + 2^q + 2^r + 2^s) ∧ 
    (∀ i, (2^a = ∑ j, M i j) ∨ (2^b = ∑ j, M i j) ∨ (2^c = ∑ j, M i j) ∨ (2^d = ∑ j, M i j)) ∧
    (∀ j, (2^p = ∑ i, M i j) ∨ (2^q = ∑ i, M i j) ∨ (2^r = ∑ i, M i j) ∨ (2^s = ∑ i, M i j))) := 
sorry

end no_distinct_power_sums_on_board_l575_575528


namespace cone_cross_section_equilateral_triangle_l575_575648

theorem cone_cross_section_equilateral_triangle
  (R r : ℝ)
  (h1 : 2 * π * r = (1 / 2) * 2 * π * R) :
  ∃ h : ℝ, h = 2 * r ∧ cross_section R r = equilateral_triangle :=
by
  sorry

end cone_cross_section_equilateral_triangle_l575_575648


namespace find_m_l575_575738

theorem find_m (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_f_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x)
  (h_g_domain : a ≠ b ∧ a ≠ 0 ∧ b ≠ 0) 
  (h_g_range : ∀ x, x ∈ set.Icc a b → f x = g x)
  (h_g_intersection : ∀ (m : ℝ), set.finite {x | x ∈ set.Icc a b ∧ g x = x^2 + m} ∧ 
                                  set.card ({x | x ∈ set.Icc a b ∧ g x = x^2 + m}) = 2) :
  m = -2 := 
begin
  sorry
end

end find_m_l575_575738


namespace sector_angle_solution_l575_575468

-- Define the conditions
variables {R α : ℝ}
variable circ_eq : 2 * R + α * R = 6
variable area_eq : (1 / 2) * R^2 * α = 2

-- State the theorem
theorem sector_angle_solution : α = 1 ∨ α = 4 :=
by
  intro R α
  have h1 : 2 * R + α * R = 6, from circ_eq
  have h2 : (1 / 2) * R^2 * α = 2, from area_eq
  sorry

end sector_angle_solution_l575_575468


namespace enclosed_area_value_l575_575636

noncomputable def enclosed_area : ℝ :=
  ∫ x in 0..1, x - (2 * x - x^2)

theorem enclosed_area_value : enclosed_area = 1 / 6 :=
by
  sorry

end enclosed_area_value_l575_575636


namespace password_count_l575_575654

theorem password_count : ∃ s : Finset ℕ, s.card = 4 ∧ s.sum id = 27 ∧ 
  (s = {9, 8, 7, 3} ∨ s = {9, 8, 6, 4} ∨ s = {9, 7, 6, 5}) ∧ 
  (s.toList.permutations.length = 72) := sorry

end password_count_l575_575654


namespace log_base_5_of_1_div_25_l575_575419

theorem log_base_5_of_1_div_25 : log 5 (1 / 25) = -2 := by
  sorry

end log_base_5_of_1_div_25_l575_575419


namespace ratio_equality_l575_575154

variable {A B C A1 A2 B1 B2 C1 C2 : Type}
variable [has_coe ℝ ℝ A B C A1 A2 B1 B2 C1 C2 : ℝ]

noncomputable def equal_ratios (A B C A1 A2 B1 B2 C1 C2 : Type) [has_coe ℝ ℝ A B C A1 A2 B1 B2 C1 C2 : ℝ] :=
∃ (A1 A2 B1 B2 C1 C2 : ℝ), 
-- Point choosing on sides
(let A1 = 0; A2 = 1; B1 = 0.5; B2 = 1; C1 = 0; C2 = 1) 
-- Segments being equal length
∧ (dist A1 B2 = dist B1 C2 ∧ dist B1 C2 = dist C1 A2 ∧ dist C1 A2 = dist A1 B2)
-- Intersecting at a single point
∧ (exists (O : ℝ), (O = 0) ∧ (O = 1))
-- Forming 60° angles
∧ (ang A1 B2 C1 = ang B1 C2 C1 ∧ ang B1 C2 C1 = ang C1 A2 A1 ∧ ang C1 A2 A1 = ang A1 B2 C1)

theorem ratio_equality (A B C A1 A2 B1 B2 C1 C2 : ℝ) [condition: equal_ratios A B C A1 A2 B1 B2 C1 C2]:
\ (A B C A1 A2 B1 B2 C1 C2 : ℝ),
(A ≠ B) → (A ≠ C) → (B ≠ C) → 
(\frac (A1 - A2) (B - C) = \frac (B1 - B2) (C - A) ∧ \frac (B1 - B2) (C - A) = \frac (C1 - C2) (A - B) ∧ \frac (C1 - C2) (A - B))

end ratio_equality_l575_575154


namespace find_smallest_n_l575_575217

def a : ℕ → ℕ
| 1 := 1
| (2 * n) := if even n then a n else 2 * a n
| (2 * n + 1) := if even n then 2 * a n + 1 else a n

-- Define the smallest function to find the smallest n where the requirement holds.
def smallest (P : ℕ → Prop) : ℕ := 
  Nat.find (exists.elim (exists_unique.elim_left (ExistsUnique.intro 1 P sorry)))

-- The actual proof problem
theorem find_smallest_n : smallest (λ n, a n = a 2017) = 5 := 
sorry

end find_smallest_n_l575_575217


namespace commutating_matrices_l575_575115

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end commutating_matrices_l575_575115


namespace quadratic_eq_coeff_l575_575192

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_eq_coeff_l575_575192


namespace percentage_of_x_l575_575658

theorem percentage_of_x (x y : ℝ) (h1 : y = x / 4) (p : ℝ) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 :=
by sorry

end percentage_of_x_l575_575658


namespace percent_increase_bike_helmet_gloves_l575_575550

theorem percent_increase_bike_helmet_gloves :
  let bicycle_original := 200
  let helmet_original := 50
  let gloves_original := 30
  let bicycle_increase_rate := 0.10
  let helmet_increase_rate := 0.15
  let gloves_increase_rate := 0.20
  let bicycle_new := bicycle_original * (1 + bicycle_increase_rate)
  let helmet_new := helmet_original * (1 + helmet_increase_rate)
  let gloves_new := gloves_original * (1 + gloves_increase_rate)
  let total_original := bicycle_original + helmet_original + gloves_original
  let total_new := bicycle_new + helmet_new + gloves_new
  (total_new - total_original) / total_original * 100 ≈ 11.96 :=
by 
  sorry  

end percent_increase_bike_helmet_gloves_l575_575550


namespace inequality_pq_l575_575126

theorem inequality_pq (p q n : ℕ) (a : Fin (n+2) → ℝ)
  (hpq : q ≥ p) (hp0 : p ≥ 0) (hn : n ≥ 2) (a0 : a 0 = 0) (an : a n.succ = 1)
  (ha : ∀ k : Fin (n+1), a k ≤ a k.succ) 
  (cond : ∀ k : Fin n, 2 * a (k + 1) ≤ a k + a (k + 2)) :
  (p + 1) * ∑ k in Finset.range (n+1), a k.succ ^ p ≥ (q + 1) * ∑ k in Finset.range (n+1), a k.succ ^ q :=
by sorry

end inequality_pq_l575_575126


namespace digits_in_book_pages_l575_575718

theorem digits_in_book_pages : 
  (∑ n in (Finset.range 9), 1) + (∑ n in (Finset.range (99 - 10 + 1)), 2) + (∑ n in (Finset.range (366 - 100 + 1)), 3) = 990 :=
by
  sorry

end digits_in_book_pages_l575_575718


namespace binom_12_6_eq_924_l575_575807

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575807


namespace sum_vertical_asymptotes_l575_575644

theorem sum_vertical_asymptotes :
  let f := λ x => (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3),
      p := -1 / 2,
      q := -3 / 2 in
  (4 * p^2 + 6 * p + 3 = 0) ∧ (4 * q^2 + 6 * q + 3 = 0) ∧ (p + q = -2) :=
by
  -- definitions of f, p, and q
  let f := λ x => (6 * x^2 - 11) / (4 * x^2 + 6 * x + 3)
  let p := -1 / 2
  let q := -3 / 2
  -- prove the conditions and the result
  sorry

end sum_vertical_asymptotes_l575_575644


namespace binary_to_decimal_110011_l575_575377

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575377


namespace find_cartesian_equation_of_C_find_polar_equation_of_perpendicular_line_l575_575441

-- Given conditions translated to Lean
def transformed_curve (theta : ℝ) : ℝ × ℝ :=
  (2 * Real.cos theta, Real.sin theta)

-- Cartesian equation derived from the transformed coordinates
def cartesian_equation_C (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

-- Points of intersection of the line and the curve
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 2

def point_P1 : ℝ × ℝ := (2, 0)
def point_P2 : ℝ × ℝ := (0, 1)

def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def perpendicular_line_through_midpoint (midpoint : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ → Prop
| (x, y) => y = slope * (x - midpoint.1) + midpoint.2

def polar_eq (rho theta : ℝ) : ℝ :=
  rho = 3 / (4 * Real.cos theta - 2 * Real.sin theta)

-- Mathematical problem statements in Lean
theorem find_cartesian_equation_of_C :
  ∀ (theta x y : ℝ),
    transformed_curve theta = (x, y) →
    cartesian_equation_C x y :=
  sorry

theorem find_polar_equation_of_perpendicular_line :
  ∀ (x y : ℝ),
    line_eq x y →
    (x = point_P1.1 ∧ y = point_P1.2) ∨ (x = point_P2.1 ∧ y = point_P2.2) →
    let mid := midpoint point_P1 point_P2 in
    let perp_line := perpendicular_line_through_midpoint mid 2 in
    polar_eq (4 * (mid.1) - 2 * (mid.2)) (Real.atan mid.1/ mid.2) :=
  sorry

end find_cartesian_equation_of_C_find_polar_equation_of_perpendicular_line_l575_575441


namespace combination_8_5_l575_575353

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575353


namespace Wendy_total_glasses_l575_575246

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l575_575246


namespace chord_intersect_eq_l575_575777

-- Define the geometric entities
variable {O K M N : Type}

-- Define conditions
def center_of_circle (O : Type) : Prop := sorry
def intersect_at (AC BD K : Type) : Prop := sorry
def circumcenter (triangle circumcenter : Type) : Prop := sorry

-- The problem statement
theorem chord_intersect_eq (O AC BD K M N : Type)
  (hO : center_of_circle O)
  (hIntersect : intersect_at AC BD K)
  (hCircum1 : circumcenter (triangle AKB) M)
  (hCircum2 : circumcenter (triangle CKD) N)
  : OM = KN := 
begin
  sorry
end

end chord_intersect_eq_l575_575777


namespace negation_of_proposition_l575_575210

theorem negation_of_proposition (x : ℝ) :
  ¬ (x > 1 → x ^ 2 > x) ↔ (x ≤ 1 → x ^ 2 ≤ x) :=
by 
  sorry

end negation_of_proposition_l575_575210


namespace binom_12_6_l575_575796

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575796


namespace proof_problem_l575_575028

variable {R : Type} [LinearOrder R] [OrderTopology R]
variable (f g : R → R)
variables (a b : R)

-- Conditions
def is_odd (f : R → R) : Prop := ∀ x, f (-x) = -f x
def is_even (g : R → R) : Prop := ∀ x, g (-x) = g x
def is_increasing_on (h : R → R) (s : Set R) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → h x < h y

-- Proof problem
theorem proof_problem (ho : is_odd f) (he : is_even g) [hdom_f : ∀ x, f x ∈ R] [hdom_g : ∀ x, g x ∈ R]
  (h_incr_f : is_increasing_on f (Set.Ioo a b)) (h_incr_g : is_increasing_on g (Set.Ioo a b)) (hab : a < b) :
  (is_increasing_on f (Set.Ioo (-b) (-a))) ∧ ¬(is_odd (λ x, f x + g x) ∨ is_even (λ x, f x + g x) ∧ is_increasing_on (λ x, f x + g x) (Set.Ioo a b)) :=
sorry

end proof_problem_l575_575028


namespace perpendicular_line_through_M_l575_575916

def line_l(x y : ℝ) : Prop := 2 * x - y = 4

def point_M (x : ℝ) : ℝ × ℝ := (x, 0)

def perpendicular_line (a b c x y : ℝ) : Prop := a * x + b * y = c ∧ (a, b) ≠ (0, 0)

theorem perpendicular_line_through_M :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    perpendicular_line a b c x y ∧ 
    (point_M x y) ∧ 
    (line_l x y) ∧ 
    a * 2 + b * 1 = 0 ∧
    c = -2 := 
  sorry

end perpendicular_line_through_M_l575_575916


namespace four_digit_numbers_count_l575_575500

-- Define the problem statement as a theorem
theorem four_digit_numbers_count : 
  ∃ (n : ℕ), n = 168 ∧ 
    (∀ (num : ℕ), 
      1000 ≤ num ∧ num < 10000 → 
      let digits := (num / 1000 % 10, num / 100 % 10, num / 10 % 10, num % 10) in 
        (digits.1 ≠ digits.2 ∧ digits.1 ≠ digits.3 ∧ digits.1 ≠ digits.4 ∧
         digits.2 ≠ digits.3 ∧ digits.2 ≠ digits.4 ∧ digits.3 ≠ digits.4) →
         ((digits.1 + digits.2) / 2 = digits.3 ∨ 
          (digits.1 + digits.3) / 2 = digits.2 ∨ 
          (digits.1 + digits.4) / 2 = digits.2 ∨ 
          (digits.2 + digits.3) / 2 = digits.1 ∨ 
          (digits.2 + digits.4) / 2 = digits.1 ∨ 
          (digits.3 + digits.4) / 2 = digits.1)) :=
by sorry

end four_digit_numbers_count_l575_575500


namespace ratio_alcohol_to_water_l575_575507

theorem ratio_alcohol_to_water (vol_alcohol vol_water : ℚ) 
  (h_alcohol : vol_alcohol = 2/7) 
  (h_water : vol_water = 3/7) : 
  vol_alcohol / vol_water = 2 / 3 := 
by
  sorry

end ratio_alcohol_to_water_l575_575507


namespace binomial_evaluation_l575_575829

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575829


namespace maximum_perfect_matchings_in_triangulation_l575_575006

-- Let's define the context of the problem:
-- A convex 20-gon P, Triangulations T, Perfect matching conditions
open SimpleGraph

def convex_20_gon : Type := List (Fin 20)
def triangulation (P : convex_20_gon) : Type := { T : List (Fin 20 × Fin 20) // ∀ (d₁ d₂ ∈ T), d₁ ≠ d₂ → ¬intersect P d₁ d₂ }

def is_perfect_matching (T : List (Fin 20 × Fin 20)) (M : List (Fin 20 × Fin 20)) : Prop :=
M ⊆ T ∧ ∀ (e₁ e₂ ∈ M), (e₁ ≠ e₂ → shares_endpoint e₁ e₂ = false)

def max_perfect_matchings (P : convex_20_gon) : Nat :=
Nat.fib 10

theorem maximum_perfect_matchings_in_triangulation (P : convex_20_gon) (T : triangulation P) :
  ∃ M : List (Fin 20 × Fin 20), is_perfect_matching T.val M ∧
  ∀ (T' : triangulation P), count_perfect_matchings T' ≤ max_perfect_matchings P := by sorry

end maximum_perfect_matchings_in_triangulation_l575_575006


namespace probability_white_ball_l575_575272

def num_white_balls : ℕ := 5
def num_black_balls : ℕ := 6
def total_balls : ℕ := num_white_balls + num_black_balls

theorem probability_white_ball : (num_white_balls : ℚ) / total_balls = 5 / 11 := by
  sorry

end probability_white_ball_l575_575272


namespace line_parallel_l575_575509

theorem line_parallel (a : ℝ) : (l1_parallel : (2 : ℝ) * x - a * y + 1 = 0)
  (l2_parallel : (4 : ℝ) * x + (6 : ℝ) * y - 7 = 0) : a = -3 :=
by
  sorry

end line_parallel_l575_575509


namespace binomial_12_6_l575_575815

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575815


namespace binom_eight_five_l575_575363

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575363


namespace divisible_by_m_factorial_not_always_divisible_by_m_factorial_n_plus_1_l575_575611

open Nat

def S (m n : ℕ) : ℤ :=
  1 + (Finset.range m).sum (λ k =>
    (-1)^(k+1) * ((factorial (n + k + 2)) / ((factorial n) * (n + k + 1))))

theorem divisible_by_m_factorial (m n : ℕ) : 
  factorial m ∣ S m n :=
  sorry

theorem not_always_divisible_by_m_factorial_n_plus_1 : 
  ∃ m n : ℕ, ¬(factorial m * (n + 1) ∣ S m n) :=
  sorry

end divisible_by_m_factorial_not_always_divisible_by_m_factorial_n_plus_1_l575_575611


namespace total_surveyed_people_l575_575085

theorem total_surveyed_people (x y : ℕ) (h1 : 0.525 * x = 31) (h2 : 0.784 * y = x) : y = 75 :=
sorry

end total_surveyed_people_l575_575085


namespace limit_series_l575_575739

theorem limit_series (a : ℕ → ℝ) : 
  (a 0 = 1) ∧ (a 1 = 1/3 * sqrt 3) ∧ 
  (∀ n, a (2*n + 2) = n + 2 * (1 / (3 ^ (n + 1)))) ∧ 
  (∀ n, a (2*n + 1) = (2 * n + 3) * sqrt 3 * (1 / (3 ^ (n + 2)))) → 
  (real.tendsto (λ n, ∑ i in finset.range n, a i) at_top (𝓝 (1/3 * (4 + sqrt 3)))) :=
sorry

end limit_series_l575_575739


namespace three_digit_integers_product_30_l575_575499

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end three_digit_integers_product_30_l575_575499


namespace product_of_all_t_l575_575426

theorem product_of_all_t (t : ℤ → Prop) (t_values : Finset ℤ) :
  (∀ a b : ℤ, a * b = -36 → t (a + b)) →
  t_values = (Finset.univ.filter t) →
  t_values.prod id = 0 :=
by
  sorry

end product_of_all_t_l575_575426


namespace log5_one_div_25_eq_neg2_l575_575414

theorem log5_one_div_25_eq_neg2 : log 5 (1/25) = -2 := 
by
  -- Let's denote the logarithm by x for convenience
  let x := log 5 (1/25)
  -- By definition of logarithm, we have 5^x = 1/25
  have h1 : 5^x = (1/25) := by sorry
  -- We also know that 1/25 = 5^-2
  have h2 : (1/25) = 5^(-2) := by sorry
  -- Thus, 5^x = 5^-2
  have h3 : 5^x = 5^(-2) := by
    rw [←h2] at h1
    exact h1
  -- Equating the exponents of the same base, we get x = -2
  have h4 : x = -2 := by
    apply eq_of_pow_eq_pow
    exact h3
  -- Therefore, log 5 (1/25) = -2
  exact h4
  sorry

end log5_one_div_25_eq_neg2_l575_575414


namespace binom_12_6_eq_924_l575_575811

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575811


namespace approx_sqrt_inequality_l575_575087

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt x

noncomputable def f_prime (x : ℝ) : ℝ := 1 / (2 * sqrt x)

theorem approx_sqrt_inequality :
  let x0 := 4
  let delta := 4.001 - x0
  let f_x0 := f x0
  let f'_x0 := f_prime x0
  (f_x0 + f'_x0 * delta > sqrt 4.001) :=
by
  -- Definitions from the problem statement
  let x0 := 4
  let delta := 4.001 - x0
  let f_x0 := f 4
  let f'_x0 := f_prime 4
  have h := calc
    f_x0 + f'_x0 * delta
    = 2 + (1 / (2 * sqrt 4)) * 0.001 : by
      simp [f, f_prime, f_x0, f'_x0, delta, sqrt]
    ... > sqrt 4.001 : sorry
  exact h

end approx_sqrt_inequality_l575_575087


namespace product_sum_independence_l575_575877

-- Define the point data structure for Triangle
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the Triangle structure
structure Triangle :=
(A B C : Point)

-- Definition of reflection of a point M with respect to the midpoint of AB
def reflection (A B M: Point) : Point := 
  let midAB := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)
  Point.mk (2 * midAB.x - M.x) (2 * midAB.y - M.y)

-- Define the angle equality condition
def angle_equal (A B C M: Point) : Prop :=
  ∠ M A C = ∠ M B C

-- Main theorem statement with required hypothesis
theorem product_sum_independence (ABC : Triangle) (M : Point)
  (h_angle_equal : angle_equal ABC.A ABC.B ABC.C M) :
  let N := reflection ABC.A ABC.B M in
  | (dist ABC.A M) * (dist ABC.B M) + (dist ABC.C M) * (dist ABC.C N) | = 
    (dist ABC.B ABC.C) * (dist ABC.A ABC.C) :=
sorry

end product_sum_independence_l575_575877


namespace binom_eight_five_l575_575359

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575359


namespace digit_difference_l575_575261

theorem digit_difference (X Y : ℕ) (h1 : 10 * X + Y - (10 * Y + X) = 36) : X - Y = 4 := by
  sorry

end digit_difference_l575_575261


namespace binom_12_6_l575_575803

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575803


namespace most_suitable_candidate_l575_575759

def S_A^2 : ℝ := 2.25
def S_B^2 : ℝ := 1.81
def S_C^2 : ℝ := 3.42

theorem most_suitable_candidate :
  S_B^2 < S_A^2 ∧ S_B^2 < S_C^2 :=
by
  -- Proof will be handled here
  sorry

end most_suitable_candidate_l575_575759


namespace k_value_for_polynomial_l575_575961

theorem k_value_for_polynomial (k : ℤ) :
  (3 : ℤ)^3 + k * (3 : ℤ) - 18 = 0 → k = -3 :=
by
  sorry

end k_value_for_polynomial_l575_575961


namespace max_distance_to_line_l_l575_575090

section problem

-- Define conditions
def curve_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

def line_l_polar (ρ θ : ℝ) : Prop := ρ * (2 * cos θ - sin θ) = 6

def line_l_cartesian (x y : ℝ) : Prop := 2 * x - y = 6

def curve_C2_cartesian (x y : ℝ) : Prop := (x^2)/3 + (y^2)/4 = 1

def curve_C2_param (θ: ℝ) : ℝ × ℝ := (sqrt 3 * cos θ, 2 * sin θ)

-- Define maximum distance function
def distance_to_line_l (x y : ℝ) : ℝ := 
  abs (2 * x - y - 6) / sqrt 5

-- Main theorem to be proven
theorem max_distance_to_line_l (θ : ℝ) (P : ℝ × ℝ) :
  P = curve_C2_param θ → 
  curve_C2_cartesian (P.1) (P.2) → 
  line_l_cartesian P.1 P.2 → 
  (∃ θ, distance_to_line_l (P.1) (P.2) = 2 * sqrt 5) :=
sorry

end problem

end max_distance_to_line_l_l575_575090


namespace birthday_celebration_l575_575412

def total_guests : ℕ := 750
def percentage_women : ℝ := 0.432
def percentage_men : ℝ := 0.314

def women (total : ℕ) (p_women : ℝ) : ℕ := (p_women * total).round.to_nat
def men (total : ℕ) (p_men : ℝ) : ℕ := (p_men * total).round.to_nat
def children (total_women total_men total_guests : ℕ) : ℕ := total_guests - total_women - total_men

def fraction_left (total : ℕ) (fraction : ℝ) : ℕ := (fraction * total).round.to_nat
def num_left (total : ℕ) (fraction : ℝ) : ℕ := (fraction * total).round.to_nat

def people_stayed (total : ℕ) (left : ℕ) : ℕ := total - left

theorem birthday_celebration : 
  let total_women := women total_guests percentage_women,
      total_men := men total_guests percentage_men,
      total_children := children total_women total_men total_guests,
      women_left := fraction_left total_women (7 / 15),
      men_left := fraction_left total_men (5 / 12),
      children_left := 19,
      women_stayed := people_stayed total_women women_left,
      men_stayed := people_stayed total_men men_left,
      children_stayed := people_stayed total_children children_left
  in 
  women_stayed + men_stayed + children_stayed = 482 :=
by
  let total_women := women total_guests percentage_women
  let total_men := men total_guests percentage_men
  let total_children := children total_women total_men total_guests
  let women_left := fraction_left total_women (7 / 15)
  let men_left := fraction_left total_men (5 / 12)
  let children_left := 19
  let women_stayed := people_stayed total_women women_left
  let men_stayed := people_stayed total_men men_left
  let children_stayed := people_stayed total_children children_left
  have : women_stayed + men_stayed + children_stayed = 482 := by sorry
  exact this

end birthday_celebration_l575_575412


namespace trig_function_extrema_l575_575913

theorem trig_function_extrema (a b : ℝ) (h_a_gt_0 : a > 0) (h_range_x : ∀ x, 0 ≤ x ∧ x < 2 * π) 
    (h_max : ∀ x, y = cos x ^ 2 - a * sin x + b → y ≤ 0)
    (h_min : ∀ x, y = cos x ^ 2 - a * sin x + b → y ≥ -4) :
    a = 2 ∧ b = -2 ∧ (∀ x, y = cos x ^ 2 - a * sin x + b → if x = 3 * π / 2 then y = 0 else if x = π / 2 then y = -4 else True) :=
sorry

end trig_function_extrema_l575_575913


namespace stratified_sampling_male_students_l575_575084

theorem stratified_sampling_male_students (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ)
  (h1 : total_students = 900) (h2 : female_students = 0) (h3 : sample_size = 45) : 
  ((total_students - female_students) * sample_size / total_students) = 25 := 
by {
  sorry
}

end stratified_sampling_male_students_l575_575084


namespace nora_third_tree_oranges_l575_575138

theorem nora_third_tree_oranges (a b c total : ℕ)
  (h_a : a = 80)
  (h_b : b = 60)
  (h_total : total = 260)
  (h_sum : total = a + b + c) :
  c = 120 :=
by
  -- The proof should go here
  sorry

end nora_third_tree_oranges_l575_575138


namespace digit_7_count_inclusive_range_l575_575066

theorem digit_7_count_inclusive_range :
  ∃ n : ℕ, n = 133 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 700 → (k.toString.contains '7' → n = nat.pred (700 - 567)) :=
by {
  sorry
}

end digit_7_count_inclusive_range_l575_575066


namespace solve_for_x_l575_575962

theorem solve_for_x (x : ℝ) : 2 * (2 ^ (2 * x)) = 4 ^ x + 64 → x = 3 :=
by
  intro h
  have h1 : 4 ^ x = (2 ^ 2) ^ x := by rw [pow_mul, pow_mul]
  have h2 : 4 ^ x = 2 ^ (2 * x) := by rw [pow_two, ← h1]
  have h3 : 64 = 2 ^ 6 := by norm_num
  rw [h2, h3] at h
  sorry

end solve_for_x_l575_575962


namespace parabola_equation_and_line_intersection_l575_575488

theorem parabola_equation_and_line_intersection
  (p : ℝ) (hp : p > 0)
  (QF PQ : ℝ)
  (h1 : x = 4)
  (h2 : QF = 5 / 4 * PQ) :
  (∃ y : ℝ, x^2 = 2 * p * y) ∧ (∃ k : ℝ, point_A_on_parabola : (-4, 4)
  → right_angled_triangle x^2 4y ∧ line_equation (y = k * x + 4)) :=
sorry

end parabola_equation_and_line_intersection_l575_575488


namespace special_number_digits_sum_l575_575876

def repeated_digits (digit : ℕ) (count : ℕ) : ℕ :=
  foldr (λ _ acc, acc * 10 + digit) 0 (list.range count)

def special_number_part1 : ℕ := repeated_digits 1 2017
def special_number_part2 : ℕ := repeated_digits 2 2018
def special_number : ℕ := special_number_part1 * (10 ^ 2019) + special_number_part2 * 10 + 5

noncomputable def integer_part_sqrt (n : ℕ) : ℕ :=
  nat.floor (real.sqrt ↑n)

def digit_sum (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

theorem special_number_digits_sum :
  digit_sum (integer_part_sqrt special_number) = 6056 :=
sorry

end special_number_digits_sum_l575_575876


namespace find_lambda_l575_575020

variable (e1 e2 : ℝ → ℝ → ℝ)
variable (A B C D : ℝ)
variable (λ k : ℝ)

-- Given that e1 and e2 are two non-collinear vectors
axiom h_non_collinear : ∀ a b : Real, (a ≠ 0 ∨ b ≠ 0) → e1 a b ≠ e2 a b

-- Given conditions
def AB := 3 • e1 + 2 • e2
def CB := 2 • e1 - 5 • e2
def CD := λ • e1 - e2
def BD := CD - CB

-- Points A, B, and D are collinear
axiom collinear : AB = k • BD

-- We need to show that λ = 8 under these conditions
theorem find_lambda (h_AB : AB) (h_CB : CB) (h_CD : CD) (h_collinear : collinear) : 
  λ = 8 := sorry

end find_lambda_l575_575020


namespace well_depth_and_rope_length_l575_575633

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end well_depth_and_rope_length_l575_575633


namespace fraction_of_B_grades_l575_575082

theorem fraction_of_B_grades 
  (total_students : ℕ)
  (fraction_A : ℚ)
  (fraction_C : ℚ)
  (num_D : ℕ)
  (h_total : total_students = 100)
  (h_A : fraction_A = 1/5)
  (h_C : fraction_C = 1/2)
  (h_D : num_D = 5) :
  (fraction_B : ℚ) (h_B : fraction_B = (total_students - (fraction_A*total_students + fraction_C*total_students + num_D)) / total_students) :
  fraction_B = 1/4 :=
sorry

end fraction_of_B_grades_l575_575082


namespace ones_digit_sum_mod_2021_l575_575251

theorem ones_digit_sum_mod_2021 (n : ℕ) (h : n = 2021) :
  (∑ k in finset.range (n + 1), k^n) % 10 = 1 :=
by
  sorry

end ones_digit_sum_mod_2021_l575_575251


namespace sunday_race_result_l575_575688

def initial_order : list nat := [1, 2, 3]  -- This indicates [Primus, Secundus, Tertius]

def num_changes_primus_secundus : nat := 9
def num_changes_secundus_tertius : nat := 10
def num_changes_primus_tertius : nat := 11

theorem sunday_race_result :
  ∃ final_order : list nat,
    final_order = [2, 3, 1]  -- This indicates [Secundus, Tertius, Primus]
    ∧ initial_order = [1, 2, 3]
    ∧ num_changes_primus_secundus = 9
    ∧ num_changes_secundus_tertius = 10
    ∧ num_changes_primus_tertius = 11 :=
by {
  sorry
}

end sunday_race_result_l575_575688


namespace count_perfect_squares_l575_575953

def S1 := 11^2 + 13^2 + 17^2
def S2 := 24^2 + 25^2 + 26^2
def S3 := 12^2 + 24^2 + 36^2
def S4 := 11^2 + 12^2 + 132^2

theorem count_perfect_squares :
  (∃ n : ℕ, S1 = n^2) ↔ False ∧
  (∃ n : ℕ, S2 = n^2) ↔ False ∧
  (∃ n : ℕ, S3 = n^2) ↔ False ∧
  (∃ n : ℕ, S4 = n^2) ↔ True ∧
  (list.count (λ (S : ℕ), ∃ n : ℕ, S = n^2) [S1, S2, S3, S4] = 1) := by
  sorry

end count_perfect_squares_l575_575953


namespace vector_dot_product_and_magnitude_l575_575942

variables (a b : ℝ → ℝ → ℝ → ℝ)
variables (h1 : |a| = 3)
variables (h2 : |b| = ℝ.sqrt 2)
variables (h3 : (a + b) • (a - 2*b) = 4)

theorem vector_dot_product_and_magnitude (a b : ℝ → ℝ → ℝ → ℝ) (h1 : | a | = 3) (h2 : | b | = ℝ.sqrt 2) (h3 : (a + b) • (a - 2*b) = 4) : a • b = 1 ∧ |a - b| = 3 := 
by
  sorry

end vector_dot_product_and_magnitude_l575_575942


namespace save_plan_l575_575589

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l575_575589


namespace smallest_n_l575_575409

theorem smallest_n :
  ∃ (n : ℕ), (n > 0) ∧ (∀ m, (m > 0 ∧ m < n) → 
  ∑ k in finset.range (n + 1), log 3 (1 + 1 / 3^(3^k)) < 1 + log 3 (4030 / 4031)) ∧ 
  ∑ k in finset.range (n + 1), log 3 (1 + 1 / 3^(3^k)) ≥ 1 + log 3 (4030 / 4031)
  :=
begin
   sorry
end

end smallest_n_l575_575409


namespace james_winnings_l575_575104

theorem james_winnings (W : ℝ)
  (donated : W / 2)
  (spent : W / 2 - 2)
  (remaining : W / 2 - 2 = 55) : 
  W = 114 :=
by sorry

end james_winnings_l575_575104


namespace max_third_term_is_16_l575_575665

-- Define the arithmetic sequence conditions
def arithmetic_seq (a d : ℕ) : list ℕ := [a, a + d, a + 2 * d, a + 3 * d]

-- Define the sum condition
def sum_of_sequence_is_50 (a d : ℕ) : Prop :=
  (a + a + d + a + 2 * d + a + 3 * d) = 50

-- Define the third term of the sequence
def third_term (a d : ℕ) : ℕ := a + 2 * d

-- Prove that the greatest possible third term is 16
theorem max_third_term_is_16 : ∃ (a d : ℕ), sum_of_sequence_is_50 a d ∧ third_term a d = 16 :=
by
  sorry

end max_third_term_is_16_l575_575665


namespace log_power_function_l575_575049

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

theorem log_power_function :
  f(4) = 2 →
  log (1/4) (f 2) = -1 / 4 :=
by
  intro h
  rw [f, logb_pow] at *
  sorry

end log_power_function_l575_575049


namespace binom_8_5_eq_56_l575_575366

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575366


namespace probability_sum_seven_l575_575515

theorem probability_sum_seven :
  let A := {1, 2, 3, 4, 5, 6}
  let B := {2, 3, 4, 5, 6, 7}
  let total_outcomes := 6 * 6
  let favorable_outcomes := finset.card {x | (x ∈ A) ∧ ((7 - x) ∈ B)}
  in favorable_outcomes / total_outcomes = 1 / 6 :=
by
  let A := {1, 2, 3, 4, 5, 6}
  let B := {2, 3, 4, 5, 6, 7}
  let total_outcomes := 6 * 6
  let favorable_outcomes := finset.card {x | (x ∈ A) ∧ ((7 - x) ∈ B)}
  show favorable_outcomes / total_outcomes = 1 / 6
  sorry

end probability_sum_seven_l575_575515


namespace equivalent_single_percentage_increase_l575_575513

noncomputable def calculate_final_price (p : ℝ) : ℝ :=
  let p1 := p * (1 + 0.15)
  let p2 := p1 * (1 + 0.20)
  let p_final := p2 * (1 - 0.10)
  p_final

theorem equivalent_single_percentage_increase (p : ℝ) : 
  calculate_final_price p = p * 1.242 :=
by
  sorry

end equivalent_single_percentage_increase_l575_575513


namespace curve_transformation_C1_C2_l575_575908

open Matrix
open_locale Matrix

variables {R : Type*} [CommRing R]

def M : Matrix (Fin 2) (Fin 2) R := ![![0, -1], ![1, 0]]

def N : Matrix (Fin 2) (Fin 2) R := ![![2, 1], ![-1, -2]]

def MN := M ⬝ N = ![![1, 2], ![2, 1]]

theorem curve_transformation_C1_C2 :
  (∀ x y : R, (x^2 - y^2 = 1) → (let x_new := (x + 2 * y),
                                   y_new := (2 * x + y) in
                                   y_new^2 - x_new^2 = 3)) :=
by
  sorry

end curve_transformation_C1_C2_l575_575908


namespace tangent_line_at_point_P_l575_575440

-- Definitions from Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_circle : Prop := circle_eq 1 2

-- Statement to Prove
theorem tangent_line_at_point_P : 
  point_on_circle → ∃ (m : ℝ) (b : ℝ), (m = -1/2) ∧ (b = 5/2) ∧ (∀ x y : ℝ, y = m * x + b ↔ x + 2 * y - 5 = 0) :=
by
  sorry

end tangent_line_at_point_P_l575_575440


namespace A_is_7056_l575_575523

-- Define the variables and conditions
def D : ℕ := 4 * 3
def E : ℕ := 7 * 3
def B : ℕ := 4 * D
def C : ℕ := 7 * E
def A : ℕ := B * C

-- Prove that A = 7056 given the conditions
theorem A_is_7056 : A = 7056 := by
  -- We will skip the proof steps with 'sorry'
  sorry

end A_is_7056_l575_575523


namespace fraction_of_red_knights_with_magical_swords_l575_575525

def total_knights := 1
def red_fraction := 3 / 8
def blue_fraction := 1 / 4
def green_fraction := 1 - red_fraction - blue_fraction
def magical_fraction := 1 / 5
def red_to_blue_ratio := 1.5
def red_to_green_ratio := 2

-- Required to prove
def fraction_red_magical := 48 / 175

theorem fraction_of_red_knights_with_magical_swords:
  ∃ (r b g : ℝ), 
    r = red_fraction * total_knights ∧
    b = blue_fraction * total_knights ∧
    g = green_fraction * total_knights ∧
    15 * (fraction_red_magical : ℝ) + 
    10 * (fraction_red_magical / red_to_blue_ratio) + 
    15 * (fraction_red_magical / red_to_green_ratio) = 8 :=
sorry

end fraction_of_red_knights_with_magical_swords_l575_575525


namespace find_HCF_of_two_numbers_l575_575629

theorem find_HCF_of_two_numbers (a b H : ℕ) 
  (H_HCF : Nat.gcd a b = H) 
  (H_LCM_Factors : Nat.lcm a b = H * 13 * 14) 
  (H_largest_number : 322 = max a b) : 
  H = 14 :=
sorry

end find_HCF_of_two_numbers_l575_575629


namespace part_one_part_two_part_three_l575_575309

-- Definition of the operation ⊕
def op⊕ (a b : ℚ) : ℚ := a * b + 2 * a

-- Part (1): Prove that 2 ⊕ (-1) = 2
theorem part_one : op⊕ 2 (-1) = 2 :=
by
  sorry

-- Part (2): Prove that -3 ⊕ (-4 ⊕ 1/2) = 24
theorem part_two : op⊕ (-3) (op⊕ (-4) (1 / 2)) = 24 :=
by
  sorry

-- Part (3): Prove that ⊕ is not commutative
theorem part_three : ∃ (a b : ℚ), op⊕ a b ≠ op⊕ b a :=
by
  use 2, -1
  sorry

end part_one_part_two_part_three_l575_575309


namespace general_formula_l575_575901

noncomputable def a : ℕ → ℕ
| 1 := 1
| n+1 := (if n = 0 then 1 else sorry)

noncomputable def S : ℕ → ℕ
| 1 := 1
| n+1 := S n + a (n+1)

theorem general_formula (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n, a n, S n) is_arithmetic_sequence) :
  a n = 2^n - 1 :=
sorry

end general_formula_l575_575901


namespace baba_yagas_savings_plan_l575_575603

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l575_575603


namespace sum_zeros_transformed_parabola_l575_575848

theorem sum_zeros_transformed_parabola :
  let original_parabola := λ x => (x+3)^2 - 2,
      rotated_parabola := λ x => -(x+3)^2 - 2,
      shifted_right_parabola := λ x => -(x-2)^2 - 2,
      final_parabola := λ x => -(x-2)^2 + 2,
      zeros := {x | final_parabola x = 0} in
  (zeros.head + zeros.tail.head = 4) :=
by
  sorry

end sum_zeros_transformed_parabola_l575_575848


namespace negative_angle_in_fourth_quadrant_l575_575268

theorem negative_angle_in_fourth_quadrant :
  let angle := -3290
  let degrees_in_circle := 360
  let quadrant := if (angle mod degrees_in_circle) < 0 then (angle mod degrees_in_circle) + degrees_in_circle else (angle mod degrees_in_circle)
  270 < quadrant ∧ quadrant < 360 :=
by
  let angle := -3290
  let degrees_in_circle := 360
  let quadrant := if (angle mod degrees_in_circle) < 0 then (angle mod degrees_in_circle) + degrees_in_circle else (angle mod degrees_in_circle)
  have h : quadrant = 310 := sorry
  show 270 < quadrant ∧ quadrant < 360 from sorry

end negative_angle_in_fourth_quadrant_l575_575268


namespace part1_part2_part3_l575_575056

open Set

variable (U : Set ℝ) (A B : Set ℝ)
hypothesis A_def : A = {x : ℝ | 2 * x - 4 < 0}
hypothesis B_def : B = {x : ℝ | 0 < x ∧ x < 5}
hypothesis U_def : U = univ

theorem part1 : A ∩ B = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

theorem part2 : U \ A = {x : ℝ | x ≥ 2} :=
sorry

theorem part3 : (U \ A) ∩ B = {x : ℝ | 2 ≤ x ∧ x < 5} :=
sorry

end part1_part2_part3_l575_575056


namespace binomial_12_6_eq_924_l575_575832

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575832


namespace proof_of_problem_l575_575566

noncomputable def lean_problem_statement : Prop :=
  ∃ (x y z : ℕ), 
    (x < 10 ∧ y < 10 ∧ z < 10) ∧ 
    (xyz ≡ 1 [MOD 9]) ∧ 
    (7 * z ≡ 4 [MOD 9]) ∧ 
    (8 * y ≡ 5 + y [MOD 9]) ∧ 
    ((x + y + z) % 9 = 2)

theorem proof_of_problem : lean_problem_statement :=
sorry

end proof_of_problem_l575_575566


namespace sum_possible_values_a_total_sum_possible_values_a_l575_575772

theorem sum_possible_values_a (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) 
  (h4 : a + b + c + d = 62) 
  (h5 : {p | p = a - b ∨ p = a - c ∨ p = a - d ∨ p = b - c ∨ p = b - d ∨ p = c - d} = {2, 3, 5, 6, 7, 9}) :
  a ∈ {19.75, 21} :=
sorry

theorem total_sum_possible_values_a : (19.75 + 21 : ℝ) = 40.75 :=
by norm_num

end sum_possible_values_a_total_sum_possible_values_a_l575_575772


namespace probability_of_elementary_teachers_correct_l575_575275

open Finset

noncomputable def probability_of_selecting_two_elementary_teachers : ℚ :=
  let total_teachers := 21 + 14 + 7
  let elementary_teachers := 3
  let total_selected := 6
  let selected_elementary := 2
  (choose elementary_teachers selected_elementary : ℚ) / (choose total_selected selected_elementary : ℚ)

theorem probability_of_elementary_teachers_correct :
  probability_of_selecting_two_elementary_teachers = 1 / 5 :=
by
  sorry

end probability_of_elementary_teachers_correct_l575_575275


namespace distance_AC_eq_3_l575_575097

-- Defining Points A, B, and C with Cartesian coordinates
def Point := (ℝ × ℝ × ℝ)

def A : Point := (-1, 2, 0)
def B : Point := (-1, 1, 2)
def C : Point := (1, 1, 2)

-- Distance function in 3D space
def distance (p1 p2 : Point) : ℝ :=
  ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2).sqrt

-- Statement to prove
theorem distance_AC_eq_3 : distance A C = 3 := by
  sorry

end distance_AC_eq_3_l575_575097


namespace hyperbola_eccentricity_range_l575_575920

theorem hyperbola_eccentricity_range
  (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_circle : ∃ k : ℝ, k = sqrt 3 ∨ k = -sqrt 3)
  (h_intersect : ∀ k : ℝ, k = sqrt 3 ∨ k = -sqrt 3 → ∀ x y : ℝ, (x-1)^2 + y^2 = 3 / 4 → y = k * x → (x^2 / a^2) - (y^2 / b^2) = 1) :
  ∃ e : ℝ, e > 2 :=
by
  sorry

end hyperbola_eccentricity_range_l575_575920


namespace angle_relation_l575_575635

open EuclideanGeometry

theorem angle_relation (A B C K: Point)
  (h_triangle: Triangle A B C)
  (h_angle_condition: 2 * ∠ A B C + ∠ B A C = ∠ C A B)
  (h_K_on_bisector: bisects K A B A C)
  (h_BK_eq_BC: dist A B = dist A C) :
  ∠ K B C = 2 * ∠ K B A := 
sorry

end angle_relation_l575_575635


namespace bunyakovsky_same_sector_more_likely_l575_575753

variable {n : ℕ}
variable {p : Fin n → ℝ}

theorem bunyakovsky_same_sector_more_likely : 
  (∑ i, (p i)^2) ≥ (∑ i, p i * p ((i + 1) % n)) :=
sorry

end bunyakovsky_same_sector_more_likely_l575_575753


namespace range_cosB_plus_sinC_l575_575455

theorem range_cosB_plus_sinC (a b c A B C : ℝ) (htriangle : B > 0 ∧ B < π / 2) 
  (htriangle' : C > 0 ∧ C < π / 2) (hsum_angles : A + B + C = π) (hb_eq : b = 2 * a * sin B) : 
  ∃ x, (cos B + sin C = x ∧ x > sqrt 3 / 2 ∧ x < 3 / 2) :=
sorry

end range_cosB_plus_sinC_l575_575455


namespace digit_150_of_7_over_29_l575_575697

theorem digit_150_of_7_over_29 : 
  (let frac := 7 / 29 in 
   let decimal_expansion := "0.2413793103448275862068965517" in
   nat.mod 150 28 = 22 ∧ (decimal_expansion.get 21).to_nat = 5) :=
by sorry

end digit_150_of_7_over_29_l575_575697


namespace distinct_values_of_expressions_l575_575037

theorem distinct_values_of_expressions : 
  let x := 3
  let expr1 := x^(x^(x^x))
  let expr2 := x^((x^x)^x)
  let expr3 := ((x^x)^x)^x
  let expr4 := (x^(x^x))^x
  let expr5 := (x^x)^(x^x)
  let values := {expr1, expr2, expr3, expr4, expr5}
  set.size values = 3 := by
  sorry

end distinct_values_of_expressions_l575_575037


namespace find_vector_b_magnitude_l575_575027

noncomputable def vector_magnitudes (a b : ℝ) : Prop :=
  let angle := 120
  let a_magnitude := (3 : ℝ)
  let c_magnitude := real.sqrt 13
  let cos_angle := real.cos (2 * real.pi / 3) -- cos(120 degrees) in radians
  (a = a_magnitude) ∧
  (c_magnitude = real.sqrt (a*a + b*b - 2*a*b*cos_angle)) →
  b = 4

theorem find_vector_b_magnitude : vector_magnitudes 3 4 :=
  by
  sorry

end find_vector_b_magnitude_l575_575027


namespace karel_sum_impossible_l575_575549

theorem karel_sum_impossible :
  ∀ (n m : ℕ), n = 14 ∧ m = 105 ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → i ∈ {1, 2, 3, ..., 14}) → 
  (∃ k : ℕ, m % k = 0 → False) ∧ 
  (∀ config : List (List ℕ), sum (join config) = m → (∀ line : List ℕ, line ∈ config → sum line = m / length config) → False) := 
begin
  -- Parameters of the problem
  assume n m,
  assume h,
  existsi 4, -- Assuming 4 lines for the configuration
  intro hmod,
  have hnot_divisible : ¬ (105 % 4 = 0), by sorry,
  exact hnot_divisible hmod,
  assume config hsum hequal,
  have hconfig_length : length config <> 4, by sorry,
  exact hconfig_length
end

end karel_sum_impossible_l575_575549


namespace min_distance_parabola_midpoint_l575_575023

theorem min_distance_parabola_midpoint 
  (a : ℝ) (m : ℝ) (h_pos_a : a > 0) :
  (m ≥ 1 / a → ∃ M_y : ℝ, M_y = (2 * m * a - 1) / (4 * a)) ∧ 
  (m < 1 / a → ∃ M_y : ℝ, M_y = a * m^2 / 4) := 
by 
  sorry

end min_distance_parabola_midpoint_l575_575023


namespace find_angle_l575_575064

theorem find_angle (a b c d e : ℝ) (sum_of_hexagon_angles : ℝ) (h_sum : a = 135 ∧ b = 120 ∧ c = 105 ∧ d = 150 ∧ e = 110 ∧ sum_of_hexagon_angles = 720) : 
  ∃ P : ℝ, a + b + c + d + e + P = sum_of_hexagon_angles ∧ P = 100 :=
by
  sorry

end find_angle_l575_575064


namespace tony_fish_after_ten_years_l575_575689

theorem tony_fish_after_ten_years :
  let initial_fish := 6
  let x := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  let y := [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
  (List.foldl (fun acc ⟨add, die⟩ => acc + add - die) initial_fish (List.zip x y)) = 34 := 
by
  sorry

end tony_fish_after_ten_years_l575_575689


namespace math_competition_results_l575_575976

theorem math_competition_results {A B C D E F : Type} 
  (guesses_A : (A ∨ B ∨ D))
  (guesses_B : (A ∨ C ∨ E))
  (guesses_C : (A ∨ D ∨ E))
  (guesses_D : (B ∨ C ∨ E))
  (guesses_E : (B ∨ D ∨ E))
  (guesses_F : (C ∨ D ∨ E))
  (no_all_correct : ¬ ((A ∨ B ∨ D) ∧ (A ∨ C ∨ E) ∧ (A ∨ D ∨ E) ∧ (B ∨ C ∨ E) ∧ (B ∨ D ∨ E) ∧ (C ∨ D ∨ E)))
  (three_two_correct : 
    (((count (2=count_correct_guesses A (A ∨ B ∨ D)) + 
    (count (2=count_correct_guesses B (A ∨ C ∨ E)) +
    (count (2=count_correct_guesses C (A ∨ D ∨ E)) +
    (count (2=count_correct_guesses D (B ∨ C ∨ E)) +
    (count (2=count_correct_guesses E (B ∨ D ∨ E)) +
    (count (2=count_correct_guesses F (C ∨ D ∨ E)) = 3)))
  (two_one_correct : 
    (((count (1=count_correct_guesses A (A ∨ B ∨ D)) + 
    (count (1=count_correct_guesses B (A ∨ C ∨ E)) +
    (count (1=count_correct_guesses C (A ∨ D ∨ E)) +
    (count (1=count_correct_guesses D (B ∨ C ∨ E)) +
    (count (1=count_correct_guesses E (B ∨ D ∨ E)) +
    (count (1=count_correct_guesses F (C ∨ D ∨ E)) = 2)))
  (one_zero_correct: 
    (((count (0=count_correct_guesses A (A ∨ B ∨ D)) + 
    (count (0=count_correct_guesses B (A ∨ C ∨ E)) +
    (count (0=count_correct_guesses C (A ∨ D ∨ E})) +
    (count (0=count_correct_guesses D (B ∨ C ∨ E)) +
    (count (0=count_correct_guesses E (B ∨ D ∨ E)) +
    (count (0=count_correct_guesses F (C ∨ D ∨ E)) = 1)))
: (C ∧ E ∧ F) := 
by {
  sorry
  }

end math_competition_results_l575_575976


namespace wheat_flour_one_third_l575_575731

theorem wheat_flour_one_third (recipe_cups: ℚ) (third_recipe: ℚ) 
  (h1: recipe_cups = 5 + 2 / 3) (h2: third_recipe = recipe_cups / 3) :
  third_recipe = 1 + 8 / 9 :=
by
  sorry

end wheat_flour_one_third_l575_575731


namespace alice_minimum_speed_exceed_l575_575262

-- Define the conditions

def distance_ab : ℕ := 30  -- Distance from city A to city B is 30 miles
def speed_bob : ℕ := 40    -- Bob's constant speed is 40 miles per hour
def bob_travel_time := distance_ab / speed_bob  -- Bob's travel time in hours
def alice_travel_time := bob_travel_time - (1 / 2)  -- Alice leaves 0.5 hours after Bob

-- Theorem stating the minimum speed Alice must exceed
theorem alice_minimum_speed_exceed : ∃ v : Real, v > 60 ∧ distance_ab / alice_travel_time ≤ v := sorry

end alice_minimum_speed_exceed_l575_575262


namespace num_possible_values_of_exponentiations_l575_575035

theorem num_possible_values_of_exponentiations : 
  ∃ n : ℕ, n = 3 ∧ ∃ k : ℕ, k = 3 ∧ ∃ m : ℕ, m = 3 ∧ ∃ l : ℕ, l = 3 → 
  (n ^ (k ^ (m ^ l))).num_possible_values = 4 :=
sorry

end num_possible_values_of_exponentiations_l575_575035


namespace parabola_ellipse_focus_l575_575965

-- Define the focus of the ellipse
def ellipse_focus_right : (ℝ × ℝ) := (2, 0)

-- Define the focus of the parabola y² = 2mx
def parabola_focus (m : ℝ) : (ℝ × ℝ) := (m / 2, 0)

-- Main theorem to prove that m = 4 given the conditions
theorem parabola_ellipse_focus (m : ℝ) (H : parabola_focus m = ellipse_focus_right) : m = 4 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end parabola_ellipse_focus_l575_575965


namespace partI_solution_set_partII_range_of_a_l575_575484

namespace MathProof

-- Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 3)

-- Part (Ⅰ) Proof Problem
theorem partI_solution_set (x : ℝ) : 
  f x (-1) ≤ 1 ↔ -5/2 ≤ x :=
sorry

-- Part (Ⅱ) Proof Problem
theorem partII_range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 4) ↔ -7 ≤ a ∧ a ≤ 7 :=
sorry

end MathProof

end partI_solution_set_partII_range_of_a_l575_575484


namespace coefficient_and_terms_l575_575903

noncomputable def average (a b c d e : ℕ) : ℚ :=
  (a + b + c + d + e) / 5

def median (a b c d e : ℕ) : ℕ :=
  7  -- Since it is already proved that the median would be 7 based on the problem constraints

def f (x : ℚ) (n : ℕ) : ℚ :=
  (1 / x - x^2) ^ n

theorem coefficient_and_terms (s t : ℕ) (h1 : average 4 7 10 s t = 7) (h2 : median 4 7 10 s t = 7) :
  let n := median 4 7 10 s t
  in
  (coeff (f x n) (-1) = 21) ∧
  (max_term_coefficient (f x n) = 35x^5) ∧
  (min_term_coefficient (f x n) = -35x^2) :=
by {
  sorry
}

end coefficient_and_terms_l575_575903


namespace greatest_possible_third_term_l575_575668

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l575_575668


namespace sandbox_area_correct_l575_575298

noncomputable def meterToCm (meters: ℝ) : ℝ :=
  meters * 100

def lengthInMeters : ℝ := 3.12
def widthInCm : ℝ := 146

def lengthInCm : ℝ := meterToCm lengthInMeters

def areaInSquareCm (length: ℝ) (width: ℝ) : ℝ :=
  length * width

theorem sandbox_area_correct :
  areaInSquareCm lengthInCm widthInCm = 45552 :=
by
  -- Placeholder for the proof
  sorry

end sandbox_area_correct_l575_575298


namespace binomial_12_6_eq_924_l575_575779

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575779


namespace number_of_valid_codes_l575_575615

theorem number_of_valid_codes : 
  let codes := (List.range 1000).filter (λ n => n / 100 ≠ 0) in
  let valid_codes := codes.filter (λ n => 
    n ≠ 132 ∧ 
    (n / 100, (n / 10) % 10, n % 10) ≠ (1, 3, 2) ∧ 
    (n / 100 = 1 ∧ (n / 10) % 10 ≠ 3 ∧ n % 10 ≠ 2) ∧ 
    (n / 100 ≠ 1 ∧ (n / 10) % 10 = 3 ∧ n % 10 ≠ 2) ∧ 
    (n / 100 ≠ 1 ∧ (n / 10) % 10 ≠ 3 ∧ n % 10 = 2) ∧ 
    (n / 100 = 3 ∧ (n / 10) % 10 = 1 ∧ n % 10 = 2) ∧ 
    (n / 100 = 2 ∧ (n / 10) % 10 = 3 ∧ n % 10 = 1)
  ) in
  valid_codes.length = 870 :=
sorry

end number_of_valid_codes_l575_575615


namespace b4_minus_a4_l575_575101

-- Given quadratic equation and specified root, prove the difference of fourth powers.
theorem b4_minus_a4 (a b : ℝ) (h_root : (a^2 - b^2)^2 = x) (h_equation : x^2 + 4 * a^2 * b^2 * x = 4) : b^4 - a^4 = 2 ∨ b^4 - a^4 = -2 :=
sorry

end b4_minus_a4_l575_575101


namespace tangent_line_correct_l575_575201

noncomputable def derivative (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  deriv f x

def curve (x : ℝ) : ℝ :=
  x^3 - x + 3

def tangent_line_slope (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  derivative f x

def tangent_line_equation (m : ℝ) (p : ℝ × ℝ) : ℝ → ℝ :=
  λ x, m * (x - p.1) + p.2

theorem tangent_line_correct :
  tangent_line_equation 2 (1, 3) = λ x, 2 * x - y + 1 := sorry

end tangent_line_correct_l575_575201


namespace num_pairs_of_edges_l575_575071

noncomputable def num_pairs_edges_determine_plane (edges : Finset (Fin 12)) (distinct_lengths : Bool) : Nat :=
  42

theorem num_pairs_of_edges (distinct_lengths : Bool) (h : distinct_lengths = tt) : 
  num_pairs_edges_determine_plane {i // i < 12} distinct_lengths = 42 :=
by
  sorry

end num_pairs_of_edges_l575_575071


namespace part_one_part_two_l575_575931

noncomputable def f (x : ℝ) : ℝ := abs (Real.sin x)

theorem part_one (x : ℝ) : 
  Real.sin 1 ≤ f(x) + f(x + 1) ∧ f(x) + f(x + 1) ≤ 2 * Real.cos (1 / 2) :=
by
  sorry

theorem part_two (n : ℕ) (h : n > 0) : 
  Finset.sum (Finset.range (2 * n)) (λ k, f (n + k) / (n + k)) > Real.sin 1 / 2 :=
by
  sorry

end part_one_part_two_l575_575931


namespace number_of_triangles_at_least_15_l575_575907

noncomputable def number_of_triangles_in_equilateral_division (n : ℕ) (a : ℕ) : ℕ :=
  2 * (2 * log_base (φ) n + f(a / n))

theorem number_of_triangles_at_least_15 :
  ∀ (n a : ℕ), n = 32 → a = 1 → log_base (φ) n > 15 - 2 :
  number_of_triangles_in_equilateral_division n a ≥ 15 :=
begin
  intros n a h₁ h₂ h₃,
  rw [h₁, h₂, number_of_triangles_in_equilateral_division, mul_add, f],
  sorry
end

-- Definitions for helpers: logarithm with base φ and function f
def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

-- Defining function f as per the conditions, adjusting for Lean framework
def f : ℝ → ℝ
| x if 0 ≤ x ∧ x ≤ 2 - φ := 2 * log_base φ (1 - x) + 2
| _ := 0


end number_of_triangles_at_least_15_l575_575907


namespace product_sum_l575_575897

theorem product_sum (y x z: ℕ) 
  (h1: 2014 + y = 2015 + x) 
  (h2: 2015 + x = 2016 + z) 
  (h3: y * x * z = 504): 
  y * x + x * z = 128 := 
by 
  sorry

end product_sum_l575_575897


namespace density_of_ordered_vector_l575_575127

noncomputable def density_fn {n : ℕ} (f : (fin n → ℝ) → ℝ) (x : fin n → ℝ) : ℝ :=
  ∑ (σ : Equiv.Perm (fin n)), f (σ.to_fun ∘ x) * (if (∀ i j, i < j → x i < x j) then 1 else 0)

theorem density_of_ordered_vector
  (n : ℕ)
  (f : (fin n → ℝ) → ℝ)
  (ξ : fin n → ℝ)
  (h_permutation : ∀ (i j : fin n), i ≠ j → ξ i ≠ ξ j)
  (h_order : ∀ i j : fin n, i < j → ξ i < ξ j) :
  let X_n : fin n → ℝ := ξ in
  density_fn f X_n = ∑ (σ : Equiv.Perm (fin n)), f (σ.to_fun ∘ ξ) * (if (∀ i j, i < j → ξ i < ξ j) then 1 else 0) :=
by
  sorry

end density_of_ordered_vector_l575_575127


namespace count_multiples_of_four_between_100_and_350_l575_575947

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l575_575947


namespace greatest_third_term_of_arithmetic_sequence_l575_575673

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l575_575673


namespace function_defined_for_all_x_l575_575851

noncomputable def f (k x : ℝ) : ℝ := (k * x ^ 2 - 3 * x + 4) / (3 * x ^ 2 - 4 * x + k)

theorem function_defined_for_all_x (k : ℝ) : (∀ x : ℝ, 3 * x ^ 2 - 4 * x + k ≠ 0) ↔ k > 4 / 3 :=
begin
  sorry
end

end function_defined_for_all_x_l575_575851


namespace alex_silver_tokens_l575_575754

theorem alex_silver_tokens (R B : ℕ) 
  (initial_R : R = 100) 
  (initial_B : B = 100) 
  (exchange1 : ∀ (x : ℕ), 3 * x ≤ R → R - 3 * x + 2 * x ≥ 2) 
  (exchange2 : ∀ (y : ℕ), 4 * y ≤ B → B - 4 * y + 2 * R ≥ 3) 
  (tokens_exchanged: R = initial_R - 3 * exchange1 + 2 * exchange2 ∧ B = initial_B + 2 * exchange1 - 4 * exchange2)
: (exchange1 + exchange2) = 88 :=
by {
  sorry
}

end alex_silver_tokens_l575_575754


namespace binomial_12_6_eq_924_l575_575778

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575778


namespace max_sum_s_expression_l575_575570

noncomputable def max_S (n : ℕ) (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i) ^ 4 - (x i) ^ 5

theorem max_sum_s_expression (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n)
  (h2 : ∀ i, 0 ≤ x i) (h3 : ∑ i, x i = 1) : max_S n x ≤ 1 / 12 :=
sorry

end max_sum_s_expression_l575_575570


namespace calc_product_eq_243_l575_575332

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l575_575332


namespace problem1_problem2_problem3_l575_575614

theorem problem1 (a b : ℝ) : a^2 - b^2 + a - b = (a - b) * (a + b + 1) :=
sorry

theorem problem2 (a b : ℝ) : a^2 + 4 * a * b - 5 * b^2 = (a + 5 * b) * (a - b) :=
sorry

theorem problem3 : ∃ x : ℝ, ∀ y : ℝ, y^2 - 6 * y + 1 ≥ -8 ∧ (y^2 - 6 * y + 1 = -8 ↔ y = 3) :=
exists.intro 3 (by
  intro y
  split
  case mp =>
    sorry
  case mpr =>
    sorry)

end problem1_problem2_problem3_l575_575614


namespace area_of_triangle_l575_575100

-- Definitions of the sides
def AC : ℝ := 8
def BC : ℝ := 10

-- Given condition
def cos_A_minus_B : ℝ := 31 / 32

-- Theorem statement (we need to prove this)
theorem area_of_triangle : 
  ∃ (area : ℝ), 
  (cos (A - B) = cos_A_minus_B ∧ AC = 8 ∧ BC = 10) → 
  area = 15 * real.sqrt 7 :=
by 
  sorry

end area_of_triangle_l575_575100


namespace henry_books_l575_575495

def books_after_donation (initial_count : ℕ) 
    (novels science cookbooks philosophy history self_help : ℕ) 
    (donation_percentages : (ℚ × ℚ × ℚ × ℚ × ℚ × ℚ))
    (new_acquisitions : (ℕ × ℕ × ℕ))
    (reject_percentage : ℚ) : ℕ :=
  let donated := (novels * donation_percentages.1 +.to_nat) +
                 (science * donation_percentages.2.to_rat.to_nat) +
                 (cookbooks * donation_percentages.3.to_rat.to_nat) +
                 (philosophy * donation_percentages.4.to_rat.to_nat) +
                 (history * donation_percentages.5.to_rat.to_nat) +
                 self_help -- assuming all self-help books are donated
  let recycled := donated * reject_percentage.to_rat.to_nat
  let remaining_books := initial_count - donated + new_acquisitions.1 + new_acquisitions.2 + new_acquisitions.3
  remaining_books

theorem henry_books (initial_count : ℕ)
    (novels science cookbooks philosophy history self_help : ℕ)
    (donation_percentages : (ℚ × ℚ × ℚ × ℚ × ℚ × ℚ))
    (new_acquisitions : (ℕ × ℕ × ℕ))
    (reject_percentage : ℚ) :
  initial_count = 250 →
  novels = 75 →
  science = 55 →
  cookbooks = 40 →
  philosophy = 35 →
  history = 25 →
  self_help = 20 →
  donation_percentages = (60/100, 75/100, 1/2, 30/100, 1 / 4, 1) →
  new_acquisitions = (6, 10, 8) →
  reject_percentage = 5/100 →
  books_after_donation initial_count novels science cookbooks philosophy history self_help donation_percentages new_acquisitions reject_percentage = 139 :=
by
  intros 
  sorry

end henry_books_l575_575495


namespace find_a_l575_575054

variable (ξ : Type) (P : ξ → ℝ) (a : ℝ)
variable (h1 : P a + P 7 + P 9 = 1)
variable (h2 : P a * a + P 7 * 7 + P 9 * 9 = 6.3)

theorem find_a (b : ℝ) (h1 : 0.4 + 0.1 + b = 1) (h2 : 0.5 * a + 7 * 0.1 + 9 * 0.4 = 6.3) : a = 4 := 
by
  sorry

end find_a_l575_575054


namespace geom_prog_identical_l575_575680

theorem geom_prog_identical (n : ℕ) (h : ∀ (s : Finset ℝ), s.card = 4 → ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (b / a = c / b ∧ c / b = d / c)) :
  ∃ t : Finset ℝ, t.card ≥ n ∧ ∀ x ∈ t, ∃ k : ℕ, (Finset.filter (λ y, y = x) (Finset.range (4*n)) ≠ ∅) :=
by
  sorry

end geom_prog_identical_l575_575680


namespace general_formula_a_minimum_value_lambda_l575_575902

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 0 else 3^n

noncomputable def S (n : ℕ) : ℕ :=
if n = 0 then 0 else (3 * (3^n - 1)) / 2

noncomputable def b (n : ℕ) : ℝ :=
if n = 0 then 0 else (3^(n+1)) / (3^n - 1) / (3^(n+1) - 1)

noncomputable def T (n : ℕ) : ℝ :=
(3:ℝ)/2 * (1/2 - 1/(3^(n+1) - 1))

theorem general_formula_a (n : ℕ) : 
  a n = 3^n :=
sorry

theorem minimum_value_lambda (λ : ℝ) :
  (∀ n : ℕ, T n ≤ λ * (a n - 1)) → λ ≥ 9 / 32 :=
sorry

end general_formula_a_minimum_value_lambda_l575_575902


namespace relay_race_solution_l575_575163

variable (Sadie_time : ℝ) (Sadie_speed : ℝ)
variable (Ariana_time : ℝ) (Ariana_speed : ℝ)
variable (Sarah_speed : ℝ)
variable (total_distance : ℝ)

def relay_race_time : Prop :=
  let Sadie_distance := Sadie_time * Sadie_speed
  let Ariana_distance := Ariana_time * Ariana_speed
  let Sarah_distance := total_distance - Sadie_distance - Ariana_distance
  let Sarah_time := Sarah_distance / Sarah_speed
  Sadie_time + Ariana_time + Sarah_time = 4.5

theorem relay_race_solution (h1: Sadie_time = 2) (h2: Sadie_speed = 3)
  (h3: Ariana_time = 0.5) (h4: Ariana_speed = 6)
  (h5: Sarah_speed = 4) (h6: total_distance = 17) :
  relay_race_time Sadie_time Sadie_speed Ariana_time Ariana_speed Sarah_speed total_distance :=
by
  sorry

end relay_race_solution_l575_575163


namespace brogan_red_apples_total_l575_575326

theorem brogan_red_apples_total :
  let apples_tree1 := 15
  let red_percentage_tree1 := 0.40
  let red_apples_tree1 := (red_percentage_tree1 * apples_tree1).to_int -- 6
  
  let apples_tree2 := 20
  let red_percentage_tree2 := 0.50
  let red_apples_tree2 := (red_percentage_tree2 * apples_tree2).to_int -- 10
  
  let apples_tree3 := 25
  let red_percentage_tree3 := 0.30
  let red_apples_tree3 := (red_percentage_tree3 * apples_tree3).to_int -- 7 (rounded down)
  
  let apples_tree4 := 30
  let red_percentage_tree4 := 0.60
  let red_apples_tree4 := (red_percentage_tree4 * apples_tree4).to_int -- 18
in
  red_apples_tree1 + red_apples_tree2 + red_apples_tree3 + red_apples_tree4 = 41 := 
by
  sorry

end brogan_red_apples_total_l575_575326


namespace binary_to_decimal_110011_l575_575385

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575385


namespace average_age_of_other_9_students_l575_575638

variable (total_students : ℕ) (total_average_age : ℝ) (group1_students : ℕ) (group1_average_age : ℝ) (age_student12 : ℝ) (group2_students : ℕ)

theorem average_age_of_other_9_students 
  (h1 : total_students = 16) 
  (h2 : total_average_age = 16) 
  (h3 : group1_students = 5) 
  (h4 : group1_average_age = 14) 
  (h5 : age_student12 = 42) 
  (h6 : group2_students = 9) : 
  (group1_students * group1_average_age + group2_students * 16 + age_student12) / total_students = total_average_age := by
  sorry

end average_age_of_other_9_students_l575_575638


namespace find_angle_C_l575_575098

noncomputable def angle_C (a b c : ℝ) (p q : ℝ × ℝ) (A B C : ℝ) :=
  let R := a / (2 * Real.sin A) in
  let is_parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1 in
  p = (1, -Real.sqrt 3) ∧ q = (Real.cos B, Real.sin B) ∧
  is_parallel p q ∧
  b * Real.cos C + c * Real.cos B = 2 * a * Real.sin A

theorem find_angle_C (a b c : ℝ) (p q : ℝ × ℝ) (A B C : ℝ)
  (h : angle_C a b c p q A B C) : 
  C = Real.pi / 6 :=
by
  sorry

end find_angle_C_l575_575098


namespace percent_employed_females_l575_575540

-- Define the conditions as variables in Lean 4
variables (population : ℕ) -- Total population of town X
variables (employed : population → Prop) -- Predicate for being employed
variables (male : population → Prop) -- Predicate for being male
variables (female : population → Prop) -- Predicate for being female

-- 72% of the population are employed
variables (h1 : ↑(population.count employed) = 0.72 * population)

-- 36% of the population are employed males
variables (h2 : ↑(population.count (λ x, employed x ∧ male x)) = 0.36 * population)

-- Proof statement
theorem percent_employed_females :
  ∃ percent_females : ℝ, 
    percent_females = 0.5 ∧ 
    ∀ employed_population, employed_population = 0.72 * population →
      percent_females = ↑(population.count (λ x, employed x ∧ female x)) / employed_population :=
sorry

end percent_employed_females_l575_575540


namespace unique_g_l575_575112

noncomputable def S := { x : ℝ // x ≠ 0 }

def g (x : S) : ℝ := sorry

axiom g_prop1 : g 1 = 2
axiom g_prop2 : ∀ (x y : S), x.val + y.val ≠ 0 → g ⟨1 / (x.val + y.val), sorry⟩ = g ⟨1 / x.val, sorry⟩ + g ⟨1 / y.val, sorry⟩
axiom g_prop3 : ∀ (x y : S), x.val + y.val ≠ 0 → (x.val + y.val) * g ⟨x.val + y.val, sorry⟩ = x.val^2 * y.val^2 * g x * g y

theorem unique_g : ∃! (g : S → ℝ), (g 1 = 2) ∧
  (∀ (x y : S), x.val + y.val ≠ 0 → g ⟨1 / (x.val + y.val), sorry⟩ = g ⟨1 / x.val, sorry⟩ + g ⟨1 / y.val, sorry⟩) ∧
  (∀ (x y : S), x.val + y.val ≠ 0 → (x.val + y.val) * g ⟨x.val + y.val, sorry⟩ = x.val^2 * y.val^2 * g x * g y) :=
sorry

end unique_g_l575_575112


namespace commutating_matrices_l575_575114

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=  ![![2, 3], ![4, 5]]
noncomputable def B (x y z w : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![x, y], ![z, w]]

theorem commutating_matrices (x y z w : ℝ) (h1 : A * (B x y z w) = (B x y z w) * A) (h2 : 4 * y ≠ z) : 
  (x - w) / (z - 4 * y) = 1 / 2 := 
by
  sorry

end commutating_matrices_l575_575114


namespace binary_to_decimal_110011_l575_575379

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575379


namespace combinations_of_4_choose_3_l575_575226

theorem combinations_of_4_choose_3 : (nat.choose 4 3) = 4 := 
by
  -- Use combination formula directly
  sorry

end combinations_of_4_choose_3_l575_575226


namespace henry_total_payment_l575_575431

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l575_575431


namespace binary_to_decimal_110011_l575_575387

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575387


namespace locus_center_and_slope_l575_575457

-- Define circles M and N
def circleM (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circleN (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the equation of the ellipse C
def curveC (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define the point Q
def pointQ := (1 : ℝ, 8 / 3 : ℝ)

-- Main theorem to be proven in parts (I) and (II)
theorem locus_center_and_slope :
  (∀ (x y : ℝ), circleM x y → circleN x y → curveC x y)
  ∧ 
  (∀ (A B : ℝ × ℝ) (k : ℝ), 
    curveC (fst A) (snd A) → curveC (fst B) (snd B) 
    → (snd (pointQ) - snd A) / (fst (pointQ) - fst A) = k 
    → (snd (pointQ) - snd B) / (fst (pointQ) - fst B) = -k
    → ((snd B - snd A) / (fst B - fst A)) = 1 / 3) :=
sorry

end locus_center_and_slope_l575_575457


namespace electricity_consumption_l575_575221

variable (x y : ℝ)

-- y = 0.55 * x
def electricity_fee := 0.55 * x

-- if y = 40.7 then x should be 74
theorem electricity_consumption :
  (∃ x, electricity_fee x = 40.7) → (x = 74) :=
by
  sorry

end electricity_consumption_l575_575221


namespace range_of_a_l575_575000

-- Given function
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- Derivative of the function
def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

-- Discriminant of the derivative
def discriminant (a : ℝ) : ℝ := 4*a^2 - 12*(a + 6)

-- Proof that the range of 'a' is 'a < -3 or a > 6' for f(x) to have both maximum and minimum values
theorem range_of_a (a : ℝ) : discriminant a > 0 ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l575_575000


namespace Mike_JOHNSON_ratio_l575_575657

-- Definitions
variables (j m_r c m_t : ℕ)

-- Given conditions
def Johnson_share := j = 2500
def Mike_remaining_after_shirt := m_r = 800
def Shirt_cost := c = 200
def Mike_total := m_t = m_r + c

-- Statement of the problem
theorem Mike_JOHNSON_ratio (h1 : Johnson_share j) (h2 : Mike_remaining_after_shirt m_r) (h3 : Shirt_cost c) :
  m_t / j = 2 / 5 :=
by
  rw [Johnson_share, Mike_remaining_after_shirt, Shirt_cost, Mike_total]
  sorry

end Mike_JOHNSON_ratio_l575_575657


namespace min_value_of_expression_l575_575124

theorem min_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 3) :
  ∃ x : ℝ, (∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a + b + c = 3 → \frac{a + b}{abc} ≥ x) ∧ x = \frac{16}{9} :=
sorry

end min_value_of_expression_l575_575124


namespace sum_fractions_lt_two_l575_575909

theorem sum_fractions_lt_two (n : ℕ) (a : ℕ → ℝ) 
  (hpos : ∀ i, 1 ≤ i ∧ i ≤ n → a i > 0) 
  (hprod : ∀ k, 1 ≤ k ∧ k ≤ n → a 1 * a 2 * ... * a k ≥ 1) : 
  ∑ k in finset.range n, k.succ / ((list.range k.succ).map (λ i, 1 + a (i + 1))).prod < 2 :=
sorry

end sum_fractions_lt_two_l575_575909


namespace downstream_speed_l575_575288

variable (V_u V_s V_d : ℝ)

theorem downstream_speed (h1 : V_u = 22) (h2 : V_s = 32) (h3 : V_s = (V_u + V_d) / 2) : V_d = 42 :=
sorry

end downstream_speed_l575_575288


namespace angle_bisector_inequality_l575_575160

theorem angle_bisector_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_perimeter : (x + y + z) = 6) :
  (1 / x^2) + (1 / y^2) + (1 / z^2) ≥ 1 := by
  sorry

end angle_bisector_inequality_l575_575160


namespace binomial_12_6_l575_575819

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575819


namespace number_of_employees_l575_575630

noncomputable def cost_per_person (n : ℕ) :  ℝ :=
if n ≤ 25 then 1000 else 1500 - 20 * n

theorem number_of_employees
  (total_cost : ℝ) (total_cost_eq : total_cost = 27000) :
  ∃ x : ℕ, total_cost = x * cost_per_person x ∧ 700 ≤ cost_per_person x :=
begin
  use 30,
  split,
  { -- prove the total cost equals 27000 for x = 30
    calc total_cost
        = 30 * cost_per_person 30 : by rw total_cost_eq
    ... = 30 * (1500 - 20 * 30) : by rw if_neg (nat.lt_of_succ_lt_succ (nat.lt_succ_self 24))
    ... = 30 * 900 : by norm_num
    ... = 27000 : by norm_num },
  { -- prove the cost per person is at least 700 for x = 30
    calc cost_per_person 30
        = 1500 - 20 * 30 : by rw if_neg (nat.lt_of_succ_lt_succ (nat.lt_succ_self 24))
    ... = 900 : by norm_num
    ... ≥ 700 : by norm_num }
end

end number_of_employees_l575_575630


namespace solution_l575_575674

noncomputable def determine_numbers (x y : ℚ) : Prop :=
  x^2 + y^2 = 45 / 4 ∧ x - y = x * y

theorem solution (x y : ℚ) :
  determine_numbers x y → (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) :=
-- We state the main theorem that relates the determine_numbers predicate to the specific pairs of numbers
sorry

end solution_l575_575674


namespace binom_8_5_eq_56_l575_575365

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575365


namespace number_of_multiples_of_4_between_100_and_350_l575_575950

theorem number_of_multiples_of_4_between_100_and_350 :
  (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≥ 104 ∧ (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≤ 348 →
  (set.filter (λ x, x % 4 = 0) (finset.Icc 100 350).to_set).card = 62 :=
by
  sorry

end number_of_multiples_of_4_between_100_and_350_l575_575950


namespace number_of_correct_propositions_l575_575479

-- Define the propositions using Lean logical constructs
def prop1 (p q : Prop) : Prop := (¬(p ∧ q) → ¬p ∧ ¬q)
def prop2 (a b : ℝ) : Prop := (¬(a > b → 2^a > 2^b - 1) = (a ≤ b → 2^a ≤ 2^b - 1))
def prop3 : Prop := (¬(∀ x : ℝ, x^2 + 1 ≥ 1) = ∀ x : ℝ, x^2 + 1 < 1)
def prop4 (A B : ℝ) : Prop := (A > B) ↔ (Real.sin A > Real.sin B)

-- The main statement
theorem number_of_correct_propositions : 
  (∃ (p q : Prop), ¬(prop1 p q) ∧ ∃ (a b : ℝ), prop2 a b ∧ ¬prop3 ∧ ∃ (A B : ℝ), prop4 A B) → 
  ∃ (n : ℕ), n = 2 := 
by
  sorry

end number_of_correct_propositions_l575_575479


namespace trigonometric_identity_l575_575420

theorem trigonometric_identity :
  (cos (10 * Real.pi / 180) / tan (20 * Real.pi / 180) + 
   sqrt 3 * sin (10 * Real.pi / 180) * tan (70 * Real.pi / 180) - 
   2 * cos (40 * Real.pi / 180) = 2) :=
by sorry

end trigonometric_identity_l575_575420


namespace binom_eight_five_l575_575362

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575362


namespace range_of_a_l575_575464

open Real

theorem range_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 3 = x * y) :
  ∀ a : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + y + 3 = x * y → (x + y)^2 - a * (x + y) + 1 ≥ 0)
    ↔ a ≤ 37 / 6 :=
by { sorry }

end range_of_a_l575_575464


namespace math_problem_l575_575229

noncomputable def a : ℝ := Real.log 0.4 / Real.log 2
noncomputable def b : ℝ := 0.4 ^ 2
noncomputable def c : ℝ := 2 ^ 0.4

theorem math_problem :
  a < b ∧ b < c :=
by
  sorry

end math_problem_l575_575229


namespace total_percent_sample_candy_l575_575081

theorem total_percent_sample_candy (total_customers : ℕ) (percent_caught : ℝ) (percent_not_caught : ℝ)
  (h1 : percent_caught = 0.22)
  (h2 : percent_not_caught = 0.20)
  (h3 : total_customers = 100) :
  percent_caught + percent_not_caught = 0.28 :=
by
  sorry

end total_percent_sample_candy_l575_575081


namespace players_taking_physics_l575_575767

-- Definitions based on the conditions
def total_players : ℕ := 30
def players_taking_math : ℕ := 15
def players_taking_both : ℕ := 6

-- The main theorem to prove
theorem players_taking_physics : total_players - players_taking_math + players_taking_both = 21 := by
  sorry

end players_taking_physics_l575_575767


namespace greatest_integer_less_than_neg_seventeen_thirds_l575_575700

theorem greatest_integer_less_than_neg_seventeen_thirds : floor (-17 / 3) = -6 := by
  sorry

end greatest_integer_less_than_neg_seventeen_thirds_l575_575700


namespace find_y_l575_575878

noncomputable def v (y : ℝ) : ℝ × ℝ := (1, y)
def w : ℝ × ℝ := (8, 4)
def proj_w_v (y : ℝ) : ℝ × ℝ := let dot_v_w := 1 * 8 + y * 4 in let dot_w_w := 8 * 8 + 4 * 4 in let scalar := dot_v_w / dot_w_w in (scalar * 8, scalar * 4)

theorem find_y : ∃ y : ℝ, proj_w_v y = (7, 3.5) :=
by
  use 15.5
  sorry

end find_y_l575_575878


namespace work_completion_in_common_days_l575_575107

variable (john_days : ℕ) (rose_days : ℕ) (ethan_days : ℕ)
variable (common_days : ℚ)

# Definitions of work rates for John, Rose, and Ethan
def john_work_rate : ℚ := 1 / john_days
def rose_work_rate : ℚ := 1 / rose_days
def ethan_work_rate : ℚ := 1 / ethan_days

-- Their combined work rate
def combined_work_rate : ℚ := john_work_rate john_days + rose_work_rate rose_days + ethan_work_rate ethan_days

-- Prove that the combined result matches the known outcome
theorem work_completion_in_common_days : 
  john_days = 320 ∧ rose_days = 480 ∧ ethan_days = 240 ∧ common_days = 106.67 →
  (1 / combined_work_rate john_days rose_days ethan_days) = common_days := 
sorry

end work_completion_in_common_days_l575_575107


namespace find_numbers_l575_575014

theorem find_numbers (a b c d : ℕ)
  (h1 : a + b + c = 21)
  (h2 : a + b + d = 28)
  (h3 : a + c + d = 29)
  (h4 : b + c + d = 30) : 
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 :=
sorry

end find_numbers_l575_575014


namespace total_profit_proof_l575_575752

noncomputable def total_profit (B_share : ℝ) (B_ratio : ℝ) (total_ratio : ℝ) : ℝ :=
  B_share * (total_ratio / B_ratio)

theorem total_profit_proof (A_inv_ratio B_inv_ratio C_inv_ratio : ℝ) (B_share : ℝ) (total_ratio : ℝ) :
  A_inv_ratio = 3 * B_inv_ratio →
  B_inv_ratio = (2 / 3) * C_inv_ratio →
  B_ratio = 2 →
  total_ratio = 11 →
  B_share = 600 →
  total_profit B_share B_ratio total_ratio = 3300 :=
by
  -- We assume the statements and conditions.
  intros
  -- We prove the result directly by setting our non-computable definition for total profit.
  show total_profit 600 2 11 = 3300
  -- The rest of the proof is skipped.
  sorry

end total_profit_proof_l575_575752


namespace geometric_sequence_sum_log_l575_575984

theorem geometric_sequence_sum_log 
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ) 
  (h1 : a 2 = 3) 
  (h2 : a 5 = 81) 
  (h3 : ∀ n, a n = a 1 * (3 ^ (n - 1))) :
  (∀ n, a n = 3 ^ (n - 1)) ∧ (∀ n, b n = log 3 (a n)) ∧ (S n = (n * (n - 1)) / 2) :=
  sorry

end geometric_sequence_sum_log_l575_575984


namespace soda_price_l575_575659

-- We define the conditions as given in the problem
def regular_price (P : ℝ) : Prop :=
  -- Regular price per can is P
  ∃ P, 
  -- 25 percent discount on regular price when purchased in 24-can cases
  (∀ (discounted_price_per_can : ℝ), discounted_price_per_can = 0.75 * P) ∧
  -- Price of 70 cans at the discounted price is $28.875
  (70 * 0.75 * P = 28.875)

-- We state the theorem to prove that the regular price per can is $0.55
theorem soda_price (P : ℝ) (h : regular_price P) : P = 0.55 :=
by
  sorry

end soda_price_l575_575659


namespace three_digit_numbers_with_product_30_l575_575496

theorem three_digit_numbers_with_product_30 : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
   d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d1 * d2 * d3 = 30) ↔
  12 := 
sorry

end three_digit_numbers_with_product_30_l575_575496


namespace find_solution_l575_575558

variable {a A : ℝ}

def is_solution (a A : ℝ) : Prop :=
  let R := 2 in
  let area_circumcircle := 4 * Real.pi in
  4 * Real.pi = Real.pi * R^2 ∧ a = 4 * Real.sin A

theorem find_solution :
  is_solution 2 (Real.pi / 6) :=
by
  let R := 2
  have h1 : 4 * Real.pi = Real.pi * R^2 := by
    calc
      4 * Real.pi = Real.pi * 4 : by sorry
      _ = Real.pi * R^2 : by sorry
  have h2 : 2 = 4 * Real.sin (Real.pi / 6) := by
    calc
      2 = 4 * (1/2) : by sorry
      _ = 4 * Real.sin (Real.pi / 6) : by sorry
  exact ⟨h1, h2⟩

end find_solution_l575_575558


namespace henry_total_fee_8_bikes_l575_575434

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l575_575434


namespace quadratic_eq_coeff_l575_575191

theorem quadratic_eq_coeff (x : ℝ) : 
  (x^2 + 2 = 3 * x) = (∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 2 ∧ (a * x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_eq_coeff_l575_575191


namespace function_properties_l575_575898

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x > 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  f 1 = 3 →
  Set.range f = Set.Icc (-6) 6 :=
sorry

end function_properties_l575_575898


namespace ratio_of_poets_to_novelists_l575_575305

-- Define the conditions
def total_people : ℕ := 24
def novelists : ℕ := 15
def poets := total_people - novelists

-- Theorem asserting the ratio of poets to novelists
theorem ratio_of_poets_to_novelists (h1 : poets = total_people - novelists) : poets / novelists = 3 / 5 := by
  sorry

end ratio_of_poets_to_novelists_l575_575305


namespace proposition_truthfulness_l575_575161

-- Definitions
def is_positive (n : ℕ) : Prop := n > 0
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Original proposition
def original_prop (n : ℕ) : Prop := is_positive n ∧ is_even n → ¬ is_prime n

-- Converse proposition
def converse_prop (n : ℕ) : Prop := ¬ is_prime n → is_positive n ∧ is_even n

-- Inverse proposition
def inverse_prop (n : ℕ) : Prop := ¬ (is_positive n ∧ is_even n) → is_prime n

-- Contrapositive proposition
def contrapositive_prop (n : ℕ) : Prop := is_prime n → ¬ (is_positive n ∧ is_even n)

-- Proof problem statement
theorem proposition_truthfulness (n : ℕ) :
  (original_prop n = False) ∧
  (converse_prop n = False) ∧
  (inverse_prop n = False) ∧
  (contrapositive_prop n = True) :=
sorry

end proposition_truthfulness_l575_575161


namespace coeff_x5_y2_in_expansion_l575_575187

theorem coeff_x5_y2_in_expansion : 
  (expand_binomial_coeff (λ x : ℕ, cond (x = 5) 1 0) (λ y : ℕ, cond (y = 2) 1 0) (5) (x^2 + x + y)) = 30 :=
by 
  sorry

end coeff_x5_y2_in_expansion_l575_575187


namespace initial_potatoes_count_l575_575282

theorem initial_potatoes_count (initial_tomatoes picked_tomatoes total_remaining : ℕ) 
    (h_initial_tomatoes : initial_tomatoes = 177)
    (h_picked_tomatoes : picked_tomatoes = 53)
    (h_total_remaining : total_remaining = 136) :
  (initial_tomatoes - picked_tomatoes + x = total_remaining) → 
  x = 12 :=
by 
  sorry

end initial_potatoes_count_l575_575282


namespace divides_both_numerator_and_denominator_l575_575259

theorem divides_both_numerator_and_denominator (x m : ℤ) :
  (x ∣ (5 * m + 6)) ∧ (x ∣ (8 * m + 7)) → (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13) :=
by
  sorry

end divides_both_numerator_and_denominator_l575_575259


namespace max_sqrt_expression_l575_575131

open Real

theorem max_sqrt_expression (x y z : ℝ) (h_sum : x + y + z = 3)
  (hx : x ≥ -1) (hy : y ≥ -(2/3)) (hz : z ≥ -2) :
  sqrt (3 * x + 3) + sqrt (3 * y + 2) + sqrt (3 * z + 6) ≤ 2 * sqrt 15 := by
  sorry

end max_sqrt_expression_l575_575131


namespace philip_paintings_l575_575608

variable paintingsPerDay : ℕ := 2
variable initialPaintings : ℕ := 20
variable days : ℕ := 30

theorem philip_paintings (painting_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ) :
  painting_per_day = 2 →
  initial_paintings = 20 →
  days = 30 →
  initial_paintings + painting_per_day * days = 80 :=
by
  intros paintings_per_day_eq initial_paintings_eq days_eq
  rw [paintings_per_day_eq, initial_paintings_eq, days_eq]
  sorry

end philip_paintings_l575_575608


namespace drum_sticks_per_show_l575_575776

-- Defining the conditions as variables
variables 
  (give_away : ℕ)  -- sets given away per show
  (nights : ℕ)     -- number of nights
  (total_drum_sticks : ℕ)  -- total sets of drum sticks used

-- Given specific values for conditions
def give_away : ℕ := 6
def nights : ℕ := 30
def total_drum_sticks : ℕ := 330

-- Statement of the proof problem
theorem drum_sticks_per_show : (total_drum_sticks - give_away * nights) / nights = 5 :=
by
  sorry

end drum_sticks_per_show_l575_575776


namespace triangle_circumradius_l575_575690

open Classical

/-- Statement of the math problem -/
theorem triangle_circumradius :
  ∃ (a b : ℕ), let PQ := 20; let QR := 21; let PR := 29; 
  let a := 29; let b := 5183; 
  a = 29 ∧ b = 5183 ∧ 
  let x := ⌊(a: ℝ) + (Real.sqrt b)⌋ in x = 100 :=
sorry

end triangle_circumradius_l575_575690


namespace percentage_of_rejected_products_l575_575110

variable (P : ℝ) (John_reject_rate Jane_reject_rate Jane_fraction : ℝ)
variable (John_inspected Jane_inspected total_rejected : ℝ)

-- given conditions
def John_rejects_correct (John_inspected : ℝ) : ℝ := 0.007 * John_inspected
def Jane_rejects_correct (Jane_inspected : ℝ) : ℝ := 0.008 * Jane_inspected
def rejected_total (John_rejected Jane_rejected : ℝ) : ℝ := John_rejected + Jane_rejected

theorem percentage_of_rejected_products
  (P > 0 ) -- total products should be greater than 0
  (John_reject_rate = 0.007)
  (Jane_reject_rate = 0.008)
  (Jane_fraction = 0.5)
  (John_inspected = (1 - Jane_fraction) * P)
  (Jane_inspected = Jane_fraction * P)
  (John_rejected = John_rejects_correct John_inspected)
  (Jane_rejected = Jane_rejects_correct Jane_inspected)
  (total_rejected = rejected_total John_rejected Jane_rejected) :
  (total_rejected / P) * 100 = 0.75 :=
  sorry

end percentage_of_rejected_products_l575_575110


namespace star_sum_larger_than_emilio_sum_l575_575175

open List

-- Define the range of numbers from 1 through 50
def star_numbers : List ℕ := [1, 2, ..., 50]

-- Define a function to replace the digit 3 with the digit 2
def replace_3_with_2 (n : ℕ) : ℕ :=
  let digits := repr n |>.data.map (λ c, if c = '3' then '2' else c)
  String.toNat digits

-- Define the list of Emilio's numbers after replacement
def emilio_numbers : List ℕ := star_numbers.map replace_3_with_2

-- Statement that needs to be proven
theorem star_sum_larger_than_emilio_sum :
  star_numbers.sum = emilio_numbers.sum + 105 :=
sorry

end star_sum_larger_than_emilio_sum_l575_575175


namespace save_plan_l575_575590

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l575_575590


namespace variance_of_set_l575_575013

theorem variance_of_set {x : ℝ} (h_avg : (31 + 38 + 34 + 35 + x) / 5 = 34) :
  (1 / 5) * ((31 - 34) ^ 2 + (38 - 34) ^ 2 + (34 - 34) ^ 2 + (35 - 34) ^ 2 + (32 - 34) ^ 2) = 6 :=
by
  have hₓ : x = 32 := 
    -- Solve the equation
    sorry
  rw [hₓ]
  -- Compute the variance
  sorry

end variance_of_set_l575_575013


namespace prob3_9_l575_575937

variable {X : ℝ → ℝ}
variable {P : Set ℝ → ℝ}
variables (a b : ℝ)

-- Assume X follows a normal distribution with mean 7 and variance 4
axiom normal_dist: ∀ x, X x = PDF (Normal 7 2) x

-- Assume given probabilities
axiom prob5_9 : P {x | 5 < X x ∧ X x < 9} = a
axiom prob3_11 : P {x | 3 < X x ∧ X x < 11} = b

-- Prove the required probability
theorem prob3_9 : P {x | 3 < X x ∧ X x < 9} = (a + b) / 2 := by
  sorry

end prob3_9_l575_575937


namespace construct_triangle_l575_575375

-- Define the initial points O, A1, and A2
variable (O A1 A2 A B C : Point)
variable (Triangle : Type) [Incenter : Incenter O Triangle]
variable (FootAltitude : FootAltitude A1 A Triangle)
variable (AngleBisectorIntersect : AngleBisectorIntersect A2 A Triangle)

-- Define the conditions as hypotheses
axiom (h1 : Incenter O Triangle)
axiom (h2 : FootAltitude A1 A Triangle)
axiom (h3 : AngleBisectorIntersect A2 A Triangle)

-- State the theorem
theorem construct_triangle
  (O A1 A2 : Point) (A B C : Point) (Triangle : Type)
  [Incenter : Incenter O Triangle]
  [FootAltitude : FootAltitude A1 A Triangle]
  [AngleBisectorIntersect : AngleBisectorIntersect A2 A Triangle] :
  exists (Triangle : Type) (A B C : Point), 
    ((Incenter O Triangle) ∧ (FootAltitude A1 A Triangle) ∧ (AngleBisectorIntersect A2 A Triangle)) :=
  sorry

end construct_triangle_l575_575375


namespace binary_110011_to_decimal_l575_575400

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575400


namespace binom_8_5_eq_56_l575_575350

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575350


namespace binomial_12_6_l575_575818

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575818


namespace binom_8_5_l575_575343

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575343


namespace amount_spent_on_plain_fabric_is_45_l575_575242

-- Definitions based on given conditions
def cost_per_yard : ℝ := 7.50
def total_yards : ℝ := 16
def checkered_cost : ℝ := 75

-- Definition derived from conditions
def checkered_yards : ℝ := checkered_cost / cost_per_yard

-- Conclusion we intend to prove
def plain_yards : ℝ := total_yards - checkered_yards
def amount_spent_on_plain_fabric : ℝ := plain_yards * cost_per_yard

theorem amount_spent_on_plain_fabric_is_45 :
  amount_spent_on_plain_fabric = 45 := by
sorry

end amount_spent_on_plain_fabric_is_45_l575_575242


namespace fabric_delivered_on_monday_amount_l575_575401

noncomputable def cost_per_yard : ℝ := 2
noncomputable def earnings : ℝ := 140

def fabric_delivered_on_monday (x : ℝ) : Prop :=
  let tuesday := 2 * x
  let wednesday := (1 / 4) * tuesday
  let total_yards := x + tuesday + wednesday
  let total_earnings := total_yards * cost_per_yard
  total_earnings = earnings

theorem fabric_delivered_on_monday_amount : ∃ x : ℝ, fabric_delivered_on_monday x ∧ x = 20 :=
by sorry

end fabric_delivered_on_monday_amount_l575_575401


namespace find_angle_C_find_side_c_l575_575517
   
   variable {A B C : ℝ} {a b c : ℝ}

   -- First part
   def find_angle_condition1 : Prop :=
     a = 2 * a * Real.cos A * Real.cos B - 2 * b * (Real.sin A) ^ 2

   theorem find_angle_C (h : find_angle_condition1) : C = 2 * Real.pi / 3 :=
   sorry

   -- Second part
   def find_side_condition1 : Prop :=
     a = 2 * a * Real.cos A * Real.cos B - 2 * b * (Real.sin A) ^ 2

   def find_side_condition2 : Prop :=
     (∑ (s : ℝ), s = a + b + c) = 15

   def find_side_condition3 : Prop :=
     Real.sin A * b * c / 4 = 15 * Real.sqrt 3 / 4

   theorem find_side_c (h1 : find_side_condition1) (h2 : find_side_condition2) (h3 : find_side_condition3) : c = 7 :=
   sorry
   
end find_angle_C_find_side_c_l575_575517


namespace semicircle_geometry_problem_l575_575092

open euclidean_geometry

noncomputable def semicircle (O : Point) (A B : Point) : Circle :=
  Circle.mk_center_radius O (OA / 2)

theorem semicircle_geometry_problem
  (O A B C D : Point)
  (hAB : line_segment A B = diameter (semicircle O A B))
  (hC : on_sector C (semicircle O A B))
  (hD : on_sector D (semicircle O A B))
  (P : Point)
  (hTangent : tangent_to_semicircle P (semicircle O A B) B)
  (P_CD_intersection : intersects_line_segment C D P)
  (E F : Point)
  (PO_line : line_through P O intersects_line_segment C A E)
  (PO_line_AF_intersect: line_through P O intersects_line_segment A D F):
  distance O E = distance O F :=
sorry

end semicircle_geometry_problem_l575_575092


namespace factor_expression_l575_575859

theorem factor_expression (a b c : ℝ) : 
  ( (a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4 ) / 
  ( (a - b)^4 + (b - c)^4 + (c - a)^4 ) = 1 := 
by sorry

end factor_expression_l575_575859


namespace binom_12_6_l575_575788

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575788


namespace angle_of_parallel_l575_575508

-- Define a line and a plane
variable {L : Type} (l : L)
variable {P : Type} (β : P)

-- Define the parallel condition
def is_parallel (l : L) (β : P) : Prop := sorry

-- Define the angle function between a line and a plane
def angle (l : L) (β : P) : ℝ := sorry

-- The theorem stating that if l is parallel to β, then the angle is 0
theorem angle_of_parallel (h : is_parallel l β) : angle l β = 0 := sorry

end angle_of_parallel_l575_575508


namespace ratio_of_areas_is_five_l575_575143

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end ratio_of_areas_is_five_l575_575143


namespace positive_difference_of_solutions_eqn_l575_575425

theorem positive_difference_of_solutions_eqn :
  ∀ (x : ℝ), (∃ a b : ℝ, (sqrt[3](2 - x^2 / 5) = -1) ∧ (a = sqrt 15 ∨ a = - sqrt 15) ∧ (b = sqrt 15 ∨ b = - sqrt 15) → abs (a - b) = 2 * sqrt 15) := by
  sorry

end positive_difference_of_solutions_eqn_l575_575425


namespace min_squared_distance_l575_575924

open Real

theorem min_squared_distance : ∀ (x y : ℝ), (3 * x + y = 10) → (x^2 + y^2) ≥ 10 :=
by
  intros x y hxy
  -- Insert the necessary steps or key elements here
  sorry

end min_squared_distance_l575_575924


namespace projection_of_b_onto_a_is_neg_one_l575_575466

-- Define the conditions
variables (a b : ℝ^3) -- Assume vectors lie in 3D space
variable (magnitude_b : ℝ)
variable (angle_ab : ℝ)

-- Assume the given conditions
axiom magnitude_b_condition : magnitude_b = 2
axiom angle_ab_condition : angle_ab = (2/3) * Real.pi

-- Define the projection of b onto a
noncomputable def proj_onto (a b : ℝ^3) (angle_ab : ℝ) (magnitude_b : ℝ) : ℝ := 
  magnitude_b * Real.cos angle_ab

-- The projection of b onto a is -1
theorem projection_of_b_onto_a_is_neg_one : proj_onto a b angle_ab magnitude_b = -1 :=
by
  rw [magnitude_b_condition, angle_ab_condition]
  -- Substitute the known values and simplify
  sorry

end projection_of_b_onto_a_is_neg_one_l575_575466


namespace binomial_12_6_eq_924_l575_575785

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575785


namespace rhombus_area_540_l575_575869

variable (EFG EGH : Type)
variable (R_EFG : circumscribed_radius EFG = 15)
variable (R_EGH : circumscribed_radius EGH = 45)

theorem rhombus_area_540 : rhombus EFGH → area(EFGH) = 540 := 
by
  sorry

end rhombus_area_540_l575_575869


namespace maximum_distance_on_curve_C_l575_575089

-- Define curve C
def curve_C (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Parametric equations for curve C
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Cartesian coordinate equation of line l
def line_l (x y : ℝ) : Prop := x - 2 * y - 6 = 0

-- Polar form of line l
def polar_line_l (ρ θ : ℝ) : Prop := ρ * (Real.cos θ - 2 * Real.sin θ) = 6

-- Maximum distance from point (x, y) to the line (x - 2y - 6 = 0)
noncomputable def distance (x y : ℝ) : ℝ :=
  (abs (x - 2 * y - 6)) / (Real.sqrt 5)

-- Prove the parametric form and the maximum distance
theorem maximum_distance_on_curve_C 
  (θ : ℝ) (θ_range : 0 ≤ θ ∧ θ < Real.pi) : 
  distance (2 * Real.cos θ) (Real.sqrt 3 * Real.sin θ) ≤ 2 * Real.sqrt 5 ∧ 
  (distance (2 * Real.cos θ) (Real.sqrt 3 * Real.sin θ) = 2 * Real.sqrt 5 ↔ Real.cos (θ + Real.pi / 3) = -1) := by
  sorry

end maximum_distance_on_curve_C_l575_575089


namespace sean_net_profit_l575_575164

def cost_per_patch := 1.25
def shipping_fee_per_unit := 20
def patches_per_unit := 100

def price_tier_1 := 12.00
def price_tier_2 := 11.50
def price_tier_3 := 11.00
def price_tier_4 := 10.50

def customers_tier_1 := 4
def customers_tier_2 := 3
def customers_tier_3 := 2
def customers_tier_4 := 1

def patches_tier_1 := 5
def patches_tier_2_a := 20
def patches_tier_2_b := 15
def patches_tier_2_c := 12
def patches_tier_3 := 35
def patches_tier_4 := 100

noncomputable def total_patches := 
  (customers_tier_1 * patches_tier_1) + 
  (patches_tier_2_a + patches_tier_2_b + patches_tier_2_c) + 
  (customers_tier_3 * patches_tier_3) + 
  (customers_tier_4 * patches_tier_4)

noncomputable def units_needed := (total_patches + patches_per_unit - 1) / patches_per_unit

noncomputable def total_cost := 
  (units_needed * patches_per_unit * cost_per_patch) + 
  (units_needed * shipping_fee_per_unit)

noncomputable def total_revenue :=
  (customers_tier_1 * patches_tier_1 * price_tier_1) + 
  (patches_tier_2_a * price_tier_2 + patches_tier_2_b * price_tier_2 + patches_tier_2_c * price_tier_2) + 
  (customers_tier_3 * patches_tier_3 * price_tier_3) + 
  (customers_tier_4 * patches_tier_4 * price_tier_4)

noncomputable def net_profit := total_revenue - total_cost

theorem sean_net_profit : net_profit = 2165.50 := by
  sorry

end sean_net_profit_l575_575164


namespace unique_inverse_exponential_l575_575486

theorem unique_inverse_exponential :
  ∀ (x1 : ℝ),
  ∃! (x2 : ℝ), (Real.exp x1 * Real.exp x2 = 1) :=
by
  intro x1
  use -x1
  split
  · unfold Real.exp
    rw [exp_add]
    norm_num
  · intro y hy
    have : Real.exp (-x1) = 1 / Real.exp x1 := Real.exp_neg x1
    rw [this, ←hy]
    field_simp
    ring

end unique_inverse_exponential_l575_575486


namespace incenter_projection_of_tetrahedron_l575_575075

variables {α : Type*} [EuclideanSpace α]

/-- Given a scalene triangle ABC and a point S such that all dihedral angles between each lateral face and the base are equal, and the projection of S onto ABC falls inside the triangle, the point O, the projection of S, is the incenter of triangle ABC. -/
theorem incenter_projection_of_tetrahedron
  {A B C S : α} (hABC_scalene : ¬ ∃ (σ : Perm (Fin 3)), σ •! [A, B, C] = [A, B, C]) 
  (h_dihedral_equal : ∀ (D : Type*), DihedralAngleOfTypeA S A B C = DihedralAngleOfTypeB S A B C) 
  (h_projection_inside : ∃ t1 t2 t3 : ℝ, t1 + t2 + t3 = 1 ∧ 0 < t1 ∧ 0 < t2 ∧ 0 < t3 ∧ S = t1 • A + t2 • B + t3 • C) 
  : is_incenter_of_projection S A B C :=
sorry

end incenter_projection_of_tetrahedron_l575_575075


namespace last_digit_of_expression_l575_575250

-- Define the expression to be analyzed
def expression : ℚ := 1 / (3^15 * 2^5 : ℚ)

-- Define the property of interest: the last digit of the decimal expansion
def last_digit_decimal_expansion (x : ℚ) : ℕ := 
  (x.digits 10).reverse.head

-- Prove that the last digit of the decimal expansion of the expression is 5
theorem last_digit_of_expression : last_digit_decimal_expansion expression = 5 := 
  sorry

end last_digit_of_expression_l575_575250


namespace regular_polygon_symmetry_l575_575294

theorem regular_polygon_symmetry (n : ℕ) (h : 360 % (n * 45) = 0) : 
  (n = 8) ∧ (axial_symmetric n) ∧ (central_symmetric n) :=
by {
  sorry
}

end regular_polygon_symmetry_l575_575294


namespace sum_of_multiplicative_inverses_of_7_modulo_15_in_range_l575_575572

theorem sum_of_multiplicative_inverses_of_7_modulo_15_in_range :
  let s := Nat.Range.filter (λ x => x > 100 ∧ x < 200) in
  let inverses := s.filter (λ x => 7 * x % 15 = 1) in
  (inverses.sum : ℤ) = 1036 :=
by
  sorry

end sum_of_multiplicative_inverses_of_7_modulo_15_in_range_l575_575572


namespace parallel_resistance_l575_575716

theorem parallel_resistance (R1 R2 : ℝ) (hR1 : R1 = 8) (hR2 : R2 = 9) : 
  (1 / (1 / R1 + 1 / R2)) = 72 / 17 :=
  by
  rw [hR1, hR2]
  norm_num
  sorry

end parallel_resistance_l575_575716


namespace repeating_decimal_denominators_l575_575180

def valid_digit (x : ℕ) : Prop := x ≤ 9

def valid_digits (a b c : ℕ) : Prop :=
  valid_digit a ∧ valid_digit b ∧ valid_digit c ∧
  (a ≠ 9 ∨ b ≠ 9 ∨ c ≠ 9) ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  a ≠ b ∧ b ≠ c

def to_fraction (a b c : ℕ) : ℚ :=
  (a * 100 + b * 10 + c) / 999

noncomputable def reduced_denominator (n : ℚ) : ℕ :=
  let d := nat.gcd n.num n.den
  n.den / d

theorem repeating_decimal_denominators :
  ∃ N : finset ℕ, N.card = 8 ∧ ∀ (a b c : ℕ), valid_digits a b c → reduced_denominator (to_fraction a b c) ∈ N :=
sorry

end repeating_decimal_denominators_l575_575180


namespace trig_identity_simplified_l575_575774

open Real

theorem trig_identity_simplified :
  (sin (15 * π / 180) + cos (15 * π / 180)) * (sin (15 * π / 180) - cos (15 * π / 180)) = - (sqrt 3 / 2) :=
by
  sorry

end trig_identity_simplified_l575_575774


namespace velma_more_than_veronica_l575_575693

-- Defining the distances each flashlight can be seen
def veronica_distance : ℕ := 1000
def freddie_distance : ℕ := 3 * veronica_distance
def velma_distance : ℕ := 5 * freddie_distance - 2000

-- The proof problem: Prove that Velma's flashlight can be seen 12000 feet farther than Veronica's flashlight.
theorem velma_more_than_veronica : velma_distance - veronica_distance = 12000 := by
  sorry

end velma_more_than_veronica_l575_575693


namespace remainder_when_divided_by_x_minus_2_l575_575252

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 33 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l575_575252


namespace value_of_x_l575_575626

-- Define the custom operation * for the problem
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Define the main problem statement
theorem value_of_x (x : ℝ) (h : star 3 (star 7 x) = 5) : x = 49 / 4 :=
by
  have h7x : star 7 x = 28 - 2 * x := by sorry  -- Derived from the definitions
  have h3star7x : star 3 (28 - 2 * x) = -44 + 4 * x := by sorry  -- Derived from substituting star 7 x
  sorry

end value_of_x_l575_575626


namespace range_of_m_l575_575660

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + m^2 - 1 = 0) → (-2 < x)) ↔ m > -1 :=
by
  sorry

end range_of_m_l575_575660


namespace arithmetic_sequence_min_sum_l575_575906

theorem arithmetic_sequence_min_sum (x : ℝ) (d : ℝ) (h₁ : d > 0) :
  (∃ n : ℕ, n > 0 ∧ (n^2 - 4 * n < 0) ∧ (n = 6 ∨ n = 7)) :=
by
  sorry

end arithmetic_sequence_min_sum_l575_575906


namespace surface_distance_eq_l575_575919

-- Define the Earth's radius
variable (R : ℝ)

-- Define the coordinates of locations A and B
def locationA_latitude : ℝ := 45 -- 45°N
def locationB_latitude : ℝ := -75 -- 75°S
def longitude : ℝ := 120 -- 120°E, same for both locations

-- The proof problem statement
theorem surface_distance_eq : 
  let delta_lat := (locationA_latitude - locationB_latitude).abs in
  let fraction_circle := delta_lat / 360 in
  let distance := fraction_circle * (2 * Real.pi * R) in
  distance = (2 * Real.pi * R) * (1 / 3) := 
by 
  -- omitting the proof
  sorry

end surface_distance_eq_l575_575919


namespace number_of_correct_conclusions_l575_575921

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + x + 6

def point_on_graph (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in quadratic_function x = y

def conclusions (n : ℕ) : Prop :=
  (¬ point_on_graph (-1, 3)) ∧
  (quadratic_function (1 / 2) = 25 / 4) ∧
  (∃ (a b : ℝ), point_on_graph (-4, a) ∧ point_on_graph (-2, b) ∧ a < b) ∧
  (∀ c : ℝ, quadratic_function (c - 1/2) = (quadratic_function c - c + 1/4))

theorem number_of_correct_conclusions : conclusions 2 :=
by
  sorry

end number_of_correct_conclusions_l575_575921


namespace cosine_product_identity_l575_575613

theorem cosine_product_identity :
  (cos (π / 15)) * (cos (2 * π / 15)) * (cos (3 * π / 15)) *
  (cos (4 * π / 15)) * (cos (5 * π / 15)) * (cos (6 * π / 15)) *
  (cos (7 * π / 15)) = (1/2)^7 :=
by
  sorry

end cosine_product_identity_l575_575613


namespace diameter_of_circle_l575_575185

variables {S1 S2 : Type} [circle S1] [circle S2]
variables {P O A B C K L : Type} [point P] [point O] [point A] [point B] [point C] [point K] [point L]
variables [tangent_to S1 A] [tangent_to S1 B] [radius_of S2 O A] [radius_of S2 O B] [inner_arc S1 C] [is_line K B] [is_line L A]

theorem diameter_of_circle (h_intersect_1: intersects S1 S2 A) (h_intersect_2: intersects S1 S2 B)
(h_tangent_A : tangent_at S1 A)
(h_tangent_B : tangent_at S1 B) 
(h_radii_S2_A : radius S2 O A)
(h_radii_S2_B : radius S2 O B)
(h_inner_arc : inner_arc S1 C)
(h_line_CA : line_through C A K)
(h_line_CB : line_through C B L)
: diameter S2 K L :=
sorry

end diameter_of_circle_l575_575185


namespace intersection_cardinality_l575_575485

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem intersection_cardinality {a b : ℝ} {f : ℝ → ℝ} :
  (∃! y, (0, y) ∈ ({ (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b } ∩ { (x, y) | x = 0 })) ∨
  ¬ (∃ y, (0, y) ∈ { (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b }) :=
by
  sorry

end intersection_cardinality_l575_575485


namespace binary_to_decimal_110011_l575_575382

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575382


namespace binary_num_to_decimal_eq_51_l575_575392

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575392


namespace mass_percentage_C_in_ATP_l575_575328

noncomputable def molar_mass_C := 12.01
noncomputable def molar_mass_H := 1.008
noncomputable def molar_mass_N := 14.01
noncomputable def molar_mass_O := 16
noncomputable def molar_mass_P := 30.97

noncomputable def molar_mass_ATP :=
  10 * molar_mass_C + 16 * molar_mass_H + 5 * molar_mass_N + 13 * molar_mass_O + 3 * molar_mass_P

theorem mass_percentage_C_in_ATP :
  (10 * molar_mass_C / molar_mass_ATP) * 100 = 23.68 :=
by
  sorry

end mass_percentage_C_in_ATP_l575_575328


namespace binomial_12_6_eq_924_l575_575837

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575837


namespace problem_l575_575574

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := 2 * x - Real.cos x

-- Define the arithmetic sequence {an} with a common difference of π/8
def a (n : ℕ) : ℝ := a 0 + n * (Real.pi / 8)

-- State the given conditions as hypotheses
def conditions (a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  a_2 = a_1 + Real.pi / 8 ∧
  a_3 = a_1 + 2 * Real.pi / 8 ∧
  a_4 = a_1 + 3 * Real.pi / 8 ∧
  a_5 = a_1 + 4 * Real.pi / 8 ∧
  f a_1 + f a_2 + f a_3 + f a_4 + f a_5 = 5 * Real.pi

-- State the theorem to prove
theorem problem (a_1 a_2 a_3 a_4 a_5 : ℝ) 
  (h : conditions a_1 a_2 a_3 a_4 a_5) : [f a_3]^2 - a_2 * a_3 = (13 / 16) * Real.pi^2 := 
sorry

end problem_l575_575574


namespace binary_num_to_decimal_eq_51_l575_575390

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575390


namespace constant_function_of_quadratic_bound_l575_575766

noncomputable theory
open Classical Topology Filter

variables {f : ℝ → ℝ}

theorem constant_function_of_quadratic_bound (hf : ∀ x y : ℝ, |f x - f y| ≤ (x - y) ^ 2) : ∃ c : ℝ, ∀ x, f x = c :=
by  
  sorry

end constant_function_of_quadratic_bound_l575_575766


namespace binom_12_6_l575_575793

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575793


namespace abs_pi_expression_l575_575842

theorem abs_pi_expression : abs (π - abs (π - 9)) = 9 - 2 * π := 
by 
  -- insert proof steps here
  sorry

end abs_pi_expression_l575_575842


namespace exists_P_Q_l575_575556

variable {X : Type} [Fintype X]
variable (f : Finset X → ℝ)
variable {D : Finset X} 

-- Conditions
variable (h₁ : D.card % 2 = 0 ∧ f D > 1990)
variable (h₂ : ∀ {A B : Finset X}, A ∩ B = ∅ → A.card % 2 = 0 → B.card % 2 = 0 → f (A ∪ B) = f A + f B - 1990)

-- Proof statement
theorem exists_P_Q (h₁ : D.card % 2 = 0 ∧ f D > 1990) (h₂ : ∀ {A B : Finset X}, A ∩ B = ∅ → A.card % 2 = 0 → B.card % 2 = 0 → f (A ∪ B) = f A + f B - 1990) : 
  ∃ (P Q : Finset X), P ∩ Q = ∅ ∧ P ∪ Q = Finset.univ ∧ 
                      (∀ S, S ⊆ P → S ≠ ∅ → S.card % 2 = 0 → f S > 1990) ∧
                      (∀ T, T ⊆ Q → T.card % 2 = 0 → f T ≤ 1990)
:= by
  sorry

end exists_P_Q_l575_575556


namespace all_chords_are_diameters_l575_575520

open Classical

variables {C : Type} [MetricSpace C] {O : Point C} {r : ℝ} [Circle C O r]
variables {Ch : Finset (Chord C)}
variables (H1 : ∀ c ∈ Ch, ∃ c' ∈ Ch, midpoint c c' ∈ c' )

theorem all_chords_are_diameters (H1 : ∀ c ∈ Ch, ∃ c' ∈ Ch, midpoint c c' ∈ c') : 
  ∀ c ∈ Ch, ∃ O, c = diameter C O :=
sorry

end all_chords_are_diameters_l575_575520


namespace a4_mod_n_eq_a_l575_575120

theorem a4_mod_n_eq_a (n : ℕ) (a : ℤ) (h : Nat.gcd a n = 1) (h1 : a ^ 3 ≡ 1 [MOD n]) : a ^ 4 ≡ a [MOD n] :=
by
  sorry

end a4_mod_n_eq_a_l575_575120


namespace question1_question2_l575_575927

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.exp x - (x - a)^2 + 3

theorem question1 (a : ℝ) : (∀ f', f' (0 : ℝ) = 0) -> a = -1 :=
by
  sorry

theorem question2 (a : ℝ) : (∀ x ≥ 0, f x a ≥ 0) -> ln 3 - 3 ≤ a ∧ a ≤ sqrt 5 :=
by
  sorry

end question1_question2_l575_575927


namespace shirt_cost_correct_l575_575152

-- Definitions based on the conditions
def initial_amount : ℕ := 109
def pants_cost : ℕ := 13
def remaining_amount : ℕ := 74
def total_spent : ℕ := initial_amount - remaining_amount
def shirts_cost : ℕ := total_spent - pants_cost
def number_of_shirts : ℕ := 2

-- Statement to be proved
theorem shirt_cost_correct : shirts_cost / number_of_shirts = 11 := by
  sorry

end shirt_cost_correct_l575_575152


namespace value_of_ab_l575_575254

theorem value_of_ab (a b : ℝ) (x : ℝ) 
  (h : ∀ x, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : a * b = 0 :=
sorry

end value_of_ab_l575_575254


namespace geometric_sequence_arithmetic_sequence_common_ratio_l575_575975

theorem geometric_sequence_arithmetic_sequence_common_ratio
  (a_1 a_2 a_3 : ℝ) (h_pos : a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0)
  (h_geom_seq : ∃ q, a_2 = a_1 * q ∧ a_3 = a_1 * q ^ 2)
  (h_arith_seq : a_2 + a_1 = 2 * (1/2 * a_3)) :
  ∃ q, q * q - q - 1 = 0 ∧ q = (Real.sqrt 5 + 1) / 2 := 
begin
  sorry,
end

end geometric_sequence_arithmetic_sequence_common_ratio_l575_575975


namespace digit_150_of_seven_over_twenty_nine_l575_575699

theorem digit_150_of_seven_over_twenty_nine : 
  (decimal_expansion 7 29).nth 150 = 8 :=
sorry

end digit_150_of_seven_over_twenty_nine_l575_575699


namespace even_two_digit_numbers_count_l575_575437

open Nat

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_set : set ℕ := {0, 1, 2, 3, 4}

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def different_digits_form_even (d1 d2 : ℕ) : Prop :=
  d1 ≠ d2 ∧ is_two_digit_number (10 * d1 + d2) ∧ is_even (10 * d1 + d2)

theorem even_two_digit_numbers_count : 
  card {n | ∃ a b ∈ valid_set, different_digits_form_even a b ∧ (10 * a + b = n)} = 10 :=
by
  sorry

end even_two_digit_numbers_count_l575_575437


namespace ellipse_equation_range_of_m_l575_575922

-- Given Conditions
variables {a b : ℝ}
variable (h_ellipse : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
variable (h_notes : a > b)
variable (h_eccentricity : (sqrt 3) / 2)
variable (h_tangent : ∀ x y : ℝ, x - y + 2 = 0)

-- Proof Statements
theorem ellipse_equation
  (h1 : b = sqrt 2) 
  (h2 : ∀ a c : ℝ, (c / a) = (sqrt 3) / 2 ∧ a^2 = b^2 + c^2) :
  ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1 :=
sorry

theorem range_of_m
  (h1 : ∃ k : ℝ, ∀ x y : ℝ, y = k * x + m)
  (h2 : ∀ x1 x2 y1 y2, ((4 * k^2 + 1) * x^2 + 8 * k * m * x + 4 * m^2 - 8 = 0) →
    Δ = 16 * (8 * k^2 - m^2 + 2) > 0)
  (h3 : ∀ x1 x2, x1 + x2 = -(8 * k * m) / (4 * k^2 + 1) ∧ x1 * x2 = (4 * m^2 - 8) / (4 * k^2 + 1))
  (h4 : ∀ y1 y2, y1 * y2 = ((k * x1 + m) * (k * x2 + m)) = k^2 * x1 * x2 + k * m * (x1 + x2) + m^2 ∧
    y1 * y2 = (m^2 - 8 * k^2) / (4 * k^2 + 1))
  (h5 : ∀ x1 x2 y1 y2, |\vec{OA} + 2 * \vec{OB}| = |\vec{OA} - 2 * \vec{OB}| := vec{OA} * -1 –vec{OB} -1 ) :
  m > (2 * sqrt 10) / 5 ∨ m < (-2 * sqrt 10) / 5 :=
sorry

end ellipse_equation_range_of_m_l575_575922


namespace inequality_am_gm_l575_575157

open Real

noncomputable def sum (a : Fin n → ℝ) : ℝ :=
  ∑ i, a i

noncomputable def prod (a : Fin n → ℝ) : ℝ :=
  ∏ i, a i

theorem inequality_am_gm 
  (n : ℕ) 
  (a : Fin n → ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_n_pos : 0 < n) : 
  ( (sum a.toℝ) / n.toℝ ) ^ (sum a.toℝ) ≤ (prod (λ i => (a i).toℝ ^ (a i).toℝ)) 
  ∧ (prod (λ i => (a i).toℝ ^ (a i).toℝ)) ≤ ( (sum (λ i => (a i).toℝ^2)) / (sum a.toℝ) ) ^ (sum a.toℝ) :=
sorry

end inequality_am_gm_l575_575157


namespace psychiatrist_problem_l575_575746

theorem psychiatrist_problem 
  (x : ℕ)
  (h_total : 4 * 8 + x + (x + 5) = 25)
  : x = 2 := by
  sorry

end psychiatrist_problem_l575_575746


namespace limit_f_div_ln_l575_575995
open Real

noncomputable def u (t L : ℝ) : ℝ := t * cos (L / t)
noncomputable def v (t L : ℝ) : ℝ := t * sin (L / t)

noncomputable def u' (t L : ℝ) : ℝ := cos (L / t) - (L / t) * sin (L / t)
noncomputable def v' (t L : ℝ) : ℝ := sin (L / t) + (L / t) * cos (L / t)

noncomputable def f (a L : ℝ) (h : 0 < a ∧ a < 1) : ℝ :=
  ∫ (t : ℝ) in a..1, sqrt (u'(t, L)^2 + v'(t, L)^2)

theorem limit_f_div_ln (L : ℝ) (hL : 0 < L) : 
  ∀ a, (0 < a ∧ a < 1) → 
  ∃ l, filter.tendsto (λ a, f(a, L) / log a) (nhds_within 0 (Ioi 0)) (nhds l) := 
begin
  sorry
end

end limit_f_div_ln_l575_575995


namespace ratio_of_areas_is_five_l575_575144

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end ratio_of_areas_is_five_l575_575144


namespace even_decreasing_function_l575_575625

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_decreasing_function :
  is_even f →
  is_decreasing_on_nonneg f →
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end even_decreasing_function_l575_575625


namespace g_value_at_f_neg5_l575_575118

-- Definitions
def f (x : ℝ) : ℝ := 4 * x^2 - 8
def g : ℝ → ℝ

-- Given conditions
axiom g_value_at_f5 : g (f 5) = 10

-- Statement to prove
theorem g_value_at_f_neg5 : g (f (-5)) = 10 := by
  sorry

end g_value_at_f_neg5_l575_575118


namespace _l575_575438

noncomputable def angle_C (a b c : ℝ) : ℝ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

noncomputable theorem problem_1 (a b c C : ℝ) (h1 : a^2 + b^2 = c^2 + a * b)
  (h2 : sqrt 3 * c = 14 * Real.sin C) (h3 : a + b = 13) (h4 : C = π / 3) : 
  C = π / 3 :=
begin
  sorry
end

noncomputable theorem problem_2 (a b c C : ℝ) (h1 : a^2 + b^2 = c^2 + a * b)
  (h2 : sqrt 3 * c = 14 * Real.sin C) (h3 : a + b = 13) (Hc : c = 7) : 
  c = 7 :=
begin
  sorry
end

noncomputable theorem problem_3 (a b c : ℝ) (h1 : a^2 + b^2 = c^2 + a * b)
  (h2 : sqrt 3 * c = 14 * Real.sin (π / 3)) (h3 : a + b = 13) (h4 : a * b = 40) : 
  (1 / 2) * a * b * (Real.sin (π / 3)) = 10 * sqrt 3 :=
begin
  sorry
end

end _l575_575438


namespace circumradius_of_XYZ_l575_575996

-- Define the sides of the triangle
def side_AB := 85
def side_BC := 125
def side_CA := 140

-- Define the semi-perimeter
def semi_perimeter (a b c : ℕ) : ℝ := (a + b + c) / 2

-- Heron's formula to calculate the area of the triangle
def heron_area (a b c : ℕ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the inradius
def inradius (a b c : ℕ) : ℝ :=
  let s := semi_perimeter a b c
  let K := heron_area a b c
  K / s

-- Using the conditions and Fact 5 to find the required circumradius
theorem circumradius_of_XYZ :
  let r := inradius side_BC side_CA side_AB in
  r = 30 :=
by
  -- provide the proof here
  sorry

end circumradius_of_XYZ_l575_575996


namespace theater_casting_roles_l575_575299

theorem theater_casting_roles :
  let men := 6
  let women := 7
  let lead_female_candidates := 3
  let remaining_female_roles := 2
  let neutral_roles := 3
  (men * lead_female_candidates * (women - lead_female_candidates) * (women - lead_female_candidates - 1) * fact neutral_roles) = 108864 := sorry

end theater_casting_roles_l575_575299


namespace connie_initial_marbles_l575_575843

theorem connie_initial_marbles (marbles_given : ℝ) (marbles_left : ℝ) : 
  marbles_given = 183 → marbles_left = 593 → marbles_given + marbles_left = 776 :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end connie_initial_marbles_l575_575843


namespace binomial_12_6_eq_924_l575_575840

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575840


namespace problem_inequality_problem_equality_cases_l575_575128

theorem problem_inequality (a b c d : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 1) 
    (h2 : 0 ≤ b) (h3 : b ≤ 1) 
    (h4 : 0 ≤ c) (h5 : c ≤ 1) 
    (h6 : 0 ≤ d) (h7 : d ≤ 1) : 
    ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8 / 27 :=
by sorry

theorem problem_equality_cases (a b c d : ℝ) : 
  (ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) = 8 / 27) ↔ 
  ((a = 0 ∧ b = 1 ∧ c = 2 / 3 ∧ d = 1 / 3) ∨ 
   (a = 0 ∧ b = 1 / 3 ∧ c = 1 / 3 ∧ d = 2 / 3) ∨ 
   (a = 1 / 3 ∧ b = 1 / 3 ∧ c = 0 ∧ d = 2 / 3) ∨ 
   permutations thereof) :=
by sorry

end problem_inequality_problem_equality_cases_l575_575128


namespace total_stickers_used_l575_575153

theorem total_stickers_used :
  let red_sheets := 10,
      green_sheets := 8,
      blue_sheets := 6,
      yellow_sheets := 4,
      purple_sheets := 2,
      red_star := 3, red_circle := 4,
      green_star := 5, green_circle := 1,
      blue_star := 2, blue_circle := 1,
      yellow_star_pair := 4,
      purple_star := 6, purple_circle_shared := 2 in
  (red_sheets * (red_star + red_circle) +
   green_sheets * (green_star + green_circle) +
   blue_sheets * (blue_star + blue_circle) +
   yellow_sheets * yellow_star_pair * 2 +
   purple_sheets * purple_star + purple_circle_shared) = 182 :=
by
  sorry

end total_stickers_used_l575_575153


namespace line_intersect_curve_l575_575489

noncomputable def polar_to_cartesian_curve (rho theta : ℝ) : Prop :=
  (rho = 2 * real.cos theta) → (rho ^ 2 = 2 * rho * real.cos theta) → ((rho * cos theta) ^ 2 + (rho * sin theta) ^ 2 = 2 * (rho * cos theta))

theorem line_intersect_curve (m : ℝ) (t : ℝ) :
  (∀ x y, x = sqrt(3) / 2 * t + m ∧ y = 1 / 2 * t) →
  let x := sqrt(3) / 2 * t + m,
      y := 1 / 2 * t in
  (x^2 + y^2 = 2 * x) →
  (|m^2 - 2 * m| = 1) →
  (m = 1 + sqrt(2) ∨ m = 1 - sqrt(2)) :=
sorry

end line_intersect_curve_l575_575489


namespace range_of_a_l575_575967

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3^(a * x - 1) < (1 / 3)^(a * x^2)) ↔ a ∈ Icc (-4 : ℝ) 0 := 
sorry

end range_of_a_l575_575967


namespace at_least_two_equal_l575_575004

theorem at_least_two_equal (x y z : ℝ) (h : x / y + y / z + z / x = z / y + y / x + x / z) : 
  x = y ∨ y = z ∨ z = x := 
  sorry

end at_least_two_equal_l575_575004


namespace correct_statement_is_2_l575_575452

-- Definition of a closed set
def is_closed_set (A : Set ℤ) : Prop :=
  ∀ a b ∈ A, a + b ∈ A ∧ a - b ∈ A

-- Problem statements
def statement1 : Prop := is_closed_set ({-4, -2, 0, 2, 4} : Set ℤ)
def statement2 : Prop := is_closed_set ({n | ∃ k : ℤ, n = 3 * k} : Set ℤ)
def statement3 (A1 A2 : Set ℤ) : Prop := is_closed_set A1 → is_closed_set A2 → is_closed_set (A1 ∪ A2)
def statement4 (A1 A2 : Set ℝ) : Prop := is_closed_set (A1 : Set ℤ) → is_closed_set (A2 : Set ℤ) → ∃ c : ℝ, c ∉ (A1 ∪ A2)

-- Main problem statement: proving correctness of each statement
theorem correct_statement_is_2 :
  ¬statement1 ∧ statement2 ∧ ∀ A1 A2 : Set ℤ, ¬statement3 A1 A2 ∧ ∀ A1 A2 : Set ℝ, ¬statement4 A1 A2 :=
by sorry

end correct_statement_is_2_l575_575452


namespace perimeter_circumradius_ratio_only_sometimes_l575_575233

variables (A B a b : ℝ) (α α' β β' : ℝ)

-- Conditions: isosceles triangles with specified sides and angles
def is_isosceles_triangleI : Prop := B ≠ A ∧ α = β
def is_isosceles_triangleII : Prop := b ≠ a ∧ α' = β'
def angle_conditions : Prop := α = α' ∧ β = β'

-- Result: P:p = R:r only sometimes
theorem perimeter_circumradius_ratio_only_sometimes :
  is_isosceles_triangleI A B α β → is_isosceles_triangleII a b α' β' →
  angle_conditions α α' β β' → 
  (∀ P p R r, (P = 2 * A + B) ∧ (p = 2 * a + b) ∧ (R = A / (2 * sin α)) ∧ (r = a / (2 * sin α')) → 
  (P / p = R / r ↔ B ≠ b)) :=
by
  sorry

end perimeter_circumradius_ratio_only_sometimes_l575_575233


namespace relationship_between_x_y_l575_575439

theorem relationship_between_x_y (x y m : ℝ) (h₁ : x + m = 4) (h₂ : y - 5 = m) : x + y = 9 := 
sorry

end relationship_between_x_y_l575_575439


namespace tangent_line_eq_range_of_a_l575_575928

-- Part 1
theorem tangent_line_eq (x : ℝ) (a : ℝ) (h : a = 2 / Real.exp 1) (h2 : x = Real.exp 1) : 
  let f : ℝ → ℝ := fun x => Real.log x - a * x
  let df := (fun x => (1 / x) - a)
  let y : ℝ := f x + df x * (x - x)
  y + 1 = -1 / Real.exp 1 * (x - x + Real.exp 1) := 
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∃! n : ℤ, 0 < Real.log n - a * n) :
  (½ * Real.log 2 ≤ a ∧ a < ⅓ * Real.log 3) :=
sorry

end tangent_line_eq_range_of_a_l575_575928


namespace max_right_angles_in_pyramid_l575_575758

-- Define the conditions of the problem
def pyramid (n : ℕ) : Prop := n = 4

-- Define the property we want to prove
def max_right_angled_triangles (m : ℕ) : Prop := m = 1

-- The theorem stating the problem
theorem max_right_angles_in_pyramid :
  pyramid 4 →
  max_right_angled_triangles 1 :=
by
  intro h
  rw [pyramid] at h
  rw [max_right_angled_triangles]
  sorry -- proof to be completed

end max_right_angles_in_pyramid_l575_575758


namespace nested_custom_op_eval_l575_575403

def custom_op (x y z : ℝ) (hz : z ≠ 0) : ℝ :=
  (x + y) / z

theorem nested_custom_op_eval :
  custom_op
    (custom_op 45 15 60 (by norm_num))
    (custom_op 3 3 6 (by norm_num))
    (custom_op 24 6 30 (by norm_num))
    (by norm_num) = 2 := 
sorry

end nested_custom_op_eval_l575_575403


namespace valid_combinations_count_l575_575304

theorem valid_combinations_count (h : Nat) (g : Nat) (i : Nat) (s : Nat) : 
    h = 4 → g = 6 → i = 3 → s = 1 →
    (h * g - i - (h - s) = 18) :=
by
  intros h_eq g_eq i_eq s_eq
  rw [h_eq, g_eq, i_eq, s_eq]
  sorry

end valid_combinations_count_l575_575304


namespace find_VZ_l575_575982

/-
  Goal: To prove VZ = 20/3 given the conditions.
-/

variables {W X Y Z P Q U V R : Type}
variables [point W] [point X] [point Y] [point Z] [point P] [point Q] [point U] [point V] [point R]

-- Condition: WXYZ is a rectangle.
variable (rect_WXYZ : rectangle W X Y Z)

-- Condition: P is a point on WY such that ∠WPZ = 90°
variable (P_on_WY : P ∈ line W Y)
variable (angle_WPZ_90 : ∠ W P Z = 90)

-- Condition: UV is perpendicular to WY with WU = UP.
variable (perp_UV_WY : is_perpendicular U V W Y)
variable (WU_eq_UP : distance W U = distance U P)

-- Condition: PZ intersects UV at Q.
variable (PZ_intersects_UV_at_Q : Q ∈ line P Z ∧ Q ∈ line U V)

-- Condition: WR passes through Q.
variable (WR_passes_through_Q : Q ∈ line W R)

-- Given distances in △PQW: PW = 15, WQ = 20, QP = 25.
variables (d_PW : distance P W = 15)
variables (d_WQ : distance W Q = 20)
variables (d_QP : distance Q P = 25)

/-- To prove: VZ = 20/3. -/
theorem find_VZ : distance V Z = 20 / 3 := by 
  sorry

end find_VZ_l575_575982


namespace solve_for_x_l575_575622

theorem solve_for_x : 
  (∃ x : ℚ, x = 45 / (9 - 3 / 7) ∧ x = 21 / 4) :=
by
  use (21 / 4)
  split
  sorry

end solve_for_x_l575_575622


namespace tangent_line_slope_l575_575706

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the center of the circle and the point of tangency
def center : Point := ⟨1, 3⟩
def point_of_tangency : Point := ⟨4, 7⟩

-- Definition of slopes
def slope (p1 p2 : Point) : ℝ := (p2.y - p1.y) / (p2.x - p1.x)

-- Definition of perpendicular slope
def perpendicular_slope (m : ℝ) : ℝ := -1 / m

-- Theorem statement: slope of the tangent line
theorem tangent_line_slope :
  perpendicular_slope (slope center point_of_tangency) = -3 / 4 :=
by
  sorry

end tangent_line_slope_l575_575706


namespace sum_f_over_divisors_H_eq_zero_l575_575051

def is_even (n : ℕ) : Prop := n % 2 = 0
def num_prime_factors (n : ℕ) : ℕ := (nat.factorization n).to_finset.card

def f (d : ℕ) : ℤ :=
  if is_even (num_prime_factors d) then 1 else -1

theorem sum_f_over_divisors_H_eq_zero :
  let H := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 in
  (finset.sum (finset.filter (λ d, d ∣ H) (finset.range (H+1))) f) = 0 :=
sorry

end sum_f_over_divisors_H_eq_zero_l575_575051


namespace greatest_int_2_7_l575_575650

def greatest_int (x : ℝ) : ℤ :=
  Int.floor x

theorem greatest_int_2_7 : greatest_int 2.7 = 2 :=
by
  sorry

end greatest_int_2_7_l575_575650


namespace binomial_evaluation_l575_575831

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575831


namespace num_possible_values_of_exponentiations_l575_575036

theorem num_possible_values_of_exponentiations : 
  ∃ n : ℕ, n = 3 ∧ ∃ k : ℕ, k = 3 ∧ ∃ m : ℕ, m = 3 ∧ ∃ l : ℕ, l = 3 → 
  (n ^ (k ^ (m ^ l))).num_possible_values = 4 :=
sorry

end num_possible_values_of_exponentiations_l575_575036


namespace range_of_a_l575_575490

variable (a : ℝ)

def p : Prop := a - 4 < 0
def q : Prop := 2^a < 1

theorem range_of_a :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_l575_575490


namespace polynomial_range_a_l575_575958

/-- 
If f(x) = x^3 + 2ax^2 + 3(a+2)x + 1 has both a maximum and a minimum value,
prove that 16a^2 - 36(a+2) > 0 implies a > 2 or a < -1
-/
theorem polynomial_range_a (a : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ,
    let f := λ x, x^3 + 2*a*x^2 + 3*(a+2)*x + 1 in
    is_local_max f x ∧ is_local_min f y) ↔ (a > 2 ∨ a < -1) :=
by
  sorry

end polynomial_range_a_l575_575958


namespace well_depth_and_rope_length_l575_575634

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end well_depth_and_rope_length_l575_575634


namespace total_different_books_l575_575232

def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def tony_dean_shared_books : ℕ := 3
def all_three_shared_book : ℕ := 1

theorem total_different_books :
  tony_books + dean_books + breanna_books - tony_dean_shared_books - 2 * all_three_shared_book = 47 := 
by
  sorry 

end total_different_books_l575_575232


namespace graph_properties_of_x_pow_a_l575_575471

theorem graph_properties_of_x_pow_a (a : ℝ) (h1 : (3 : ℝ)^a = (1 / 3 : ℝ)) :
  (9 : ℝ)^a = (1 / 9 : ℝ) ∧ set.range (λ x : ℝ, x^a) = set.Ioi (0 : ℝ) :=
by
  -- We assume a = -1 based on (3 ^ a = 1 / 3)
  have a_eq_neg1 : a = -1 := by sorry
  -- Substitute a = -1 in both parts of the theorem
  rw [a_eq_neg1]
  split
  -- Part 1: (9 : ℝ) ^ (-1) = (1 / 9 : ℝ)
  calc (9 : ℝ) ^ (-1) = 1 / (9 : ℝ) : by sorry
  -- Part 2: Function range verification
  dsimp
  rw [set.ext_iff]
  intro y
  simp only [set.mem_set_of_eq, set.mem_Ioi]
  split
  intro hy
  rcases hy with ⟨x, hx⟩
  have hx_pos : 0 < x := by sorry
  linarith
  intro hy
  use 1 / y
  rw [abs_eq_self.mpr]
  rw [eq_div_iff_mul_eq (ne_of_gt hy)]
  rw [mul_comm]
  simpa using hy
  linarith

end graph_properties_of_x_pow_a_l575_575471


namespace dot_product_a_b_norm_a_plus_b_l575_575053

-- Conditions as given in part a)
variables (a b : ℝ × ℝ) (h_a_norm : ‖a‖ = 1) (h_b : b = (1,0)) 
variables (h_orth : a ∙ (a - (2:b)) = 0)

-- Question 1: Prove that a ∙ b = 1/2
theorem dot_product_a_b : a ∙ b = 1/2 :=
sorry

-- Question 2: Prove that the norm of (a + b) is sqrt(3)
theorem norm_a_plus_b : ‖a + b‖ = sqrt 3 :=
sorry

end dot_product_a_b_norm_a_plus_b_l575_575053


namespace profit_in_third_year_option_1_more_cost_effective_l575_575283

-- Define the initial cost of the boat
def initial_cost : ℕ := 980000

-- Define the first year expenses
def first_year_expenses : ℕ := 120000

-- Define the annual increase in expenses
def annual_expense_increase : ℕ := 40000

-- Define the annual fishing income
def annual_income : ℕ := 500000

-- Proof that the company starts to make a profit in the third year
theorem profit_in_third_year : ∃ n : ℕ, n = 3 ∧ 
  let expenses (year : ℕ) := first_year_expenses + annual_expense_increase * (year - 1) in
  let cumulative_income (year : ℕ) := annual_income * year in
  let cumulative_expenses (year : ℕ) := initial_cost + (first_year_expenses + annual_expense_increase * (year - 1)) * year / 2 in
  cumulative_income n > cumulative_expenses n :=
sorry

-- Proof that the first option is more cost-effective
theorem option_1_more_cost_effective :
  let total_net_income (year : ℕ) := annual_income * year - (initial_cost + (first_year_expenses + annual_expense_increase * (year - 1)) * year / 2) in
  let net_income_option_1 (year : ℕ) := total_net_income year + 260000 in
  let net_income_option_2 (year : ℕ) := total_net_income year + 80000 in
  ∃ year : ℕ, net_income_option_1 year > net_income_option_2 year :=
sorry

end profit_in_third_year_option_1_more_cost_effective_l575_575283


namespace area_HKC_over_ABC_l575_575099

variable (ABC : Triangle)

/-- Given a triangle ABC,
    altitude AF from vertex A to side BC,
    median BE from vertex B to side AC,
    points H (intersection of AF and BE) and K (midpoint of AF). -/
open Triangle

theorem area_HKC_over_ABC (ABC : Triangle)
  (A B C F E H K : Point)
  (h_A : A ∈ ABC.vertices) (h_B : B ∈ ABC.vertices) (h_C : C ∈ ABC.vertices)
  (h_FF : altitude_from_to A B C F) (h_EE : median_from_to B A C E)
  (h_H : intersection AF BE H) (h_K : midpoint_of AF K) :
  area (Triangle.mk H K C) = (3/8) * area ABC :=
sorry

end area_HKC_over_ABC_l575_575099


namespace binom_eight_five_l575_575358

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575358


namespace sequence_inequality_l575_575557

theorem sequence_inequality
  (a : Nat → ℝ)
  (pos : ∀ n, 0 < a n)
  (k : ℝ)
  (h : ∀ n, (∑ i in Finset.range (n + 1), a i ^ 2) < k * a (n + 1) ^ 2) :
  ∃ c : ℝ, ∀ n, (∑ i in Finset.range (n + 1), a i) < c * a (n + 1) :=
by
  let c := k + sqrt (k ^ 2 + k)
  existsi c
  intros n
  sorry

end sequence_inequality_l575_575557


namespace quadratic_vertex_coordinates_l575_575194

theorem quadratic_vertex_coordinates :
  (∃ v : ℝ × ℝ, v = (1, -1) ∧ ∀ x, (x^2 - 2 * x) = (x - 1)^2 - 1) :=
begin
  use (1, -1),
  split,
  { -- Prove the vertex coordinates are (1, -1)
    refl,
  },
  { -- Prove ∀ x, x^2 - 2x = (x - 1)^2 - 1
    intro x,
    calc
      x^2 - 2 * x   = (x - 1)^2 - 1 : by { sorry }
  }
end

end quadratic_vertex_coordinates_l575_575194


namespace isosceles_triangle_equilateral_l575_575751

theorem isosceles_triangle_equilateral 
  (T : Type) [triangle T]
  {a b c : angle T} :
  (a = 60° ∨ b = 60° ∨ c = 60°) ∧ is_isosceles T a b c → is_equilateral T a b c :=
by
  sorry

end isosceles_triangle_equilateral_l575_575751


namespace group_size_oranges_l575_575678

theorem group_size_oranges (oranges : ℕ) (groups : ℕ) (h1 : oranges = 356) (h2 : groups = 178) : oranges / groups = 2 :=
by
  unfold oranges groups
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left (Nat.zero_lt_succ _) (Nat.mul_comm _ _) rfl

end group_size_oranges_l575_575678


namespace find_xy_l575_575058

variables (x y : ℝ)

def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (2 * x, y)
def c : ℝ × ℝ := (x + y, 1)

theorem find_xy (h1 : a.1 / b.1 = a.2 / b.2)
                (h2 : a.1 * c.1 + a.2 * c.2 = 0) :
  x = -3/2 ∧ y = 9/4 :=
by
  sorry

end find_xy_l575_575058


namespace height_of_trapezium_l575_575423

-- Define the lengths of the parallel sides
def length_side1 : ℝ := 10
def length_side2 : ℝ := 18

-- Define the given area of the trapezium
def area_trapezium : ℝ := 210

-- The distance between the parallel sides (height) we want to prove
def height_between_sides : ℝ := 15

-- State the problem as a theorem in Lean: prove that the height is correct
theorem height_of_trapezium :
  (1 / 2) * (length_side1 + length_side2) * height_between_sides = area_trapezium :=
by
  sorry

end height_of_trapezium_l575_575423


namespace count_multiples_of_four_between_100_and_350_l575_575949

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l575_575949


namespace quadratic_expression_always_positive_l575_575881

theorem quadratic_expression_always_positive (x y : ℝ) : 
  x^2 - 4 * x * y + 6 * y^2 - 4 * y + 3 > 0 :=
by 
  sorry

end quadratic_expression_always_positive_l575_575881


namespace binomial_evaluation_l575_575828

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575828


namespace complex_number_addition_identity_l575_575914

-- Definitions of the conditions
def imaginary_unit (i : ℂ) := i^2 = -1

def complex_fraction_decomposition (a b : ℝ) (i : ℂ) := 
  (1 + i) / (1 - i) = a + b * i

-- The statement of the problem
theorem complex_number_addition_identity :
  ∃ (a b : ℝ) (i : ℂ), imaginary_unit i ∧ complex_fraction_decomposition a b i ∧ (a + b = 1) :=
sorry

end complex_number_addition_identity_l575_575914


namespace quadratic_inequality_l575_575428

theorem quadratic_inequality (a x : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) := 
sorry

end quadratic_inequality_l575_575428


namespace population_reaches_max_in_90_years_l575_575530

-- Define the initial conditions
def initial_year := 1998
def initial_population := 200
def acres := 30000
def acres_per_person := 1.5
def population_growth_rate := 4
def years_per_growth_period := 30

-- Compute the maximum number of people the island can support
def max_population := acres / acres_per_person

-- Prove the population will approximately reach the max capacity in 90 years
theorem population_reaches_max_in_90_years :
  ∃ t : ℕ, t ≈ 90 ∧ initial_population * population_growth_rate ^ (t / years_per_growth_period) ≥ max_population := 
sorry

end population_reaches_max_in_90_years_l575_575530


namespace smallest_non_multiple_of_5_abundant_l575_575764

def proper_divisors (n : ℕ) : List ℕ := List.filter (fun d => d ∣ n ∧ d < n) (List.range (n + 1))

def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_non_multiple_of_5_abundant : ∃ n, is_abundant n ∧ is_not_multiple_of_5 n ∧ 
  ∀ m, is_abundant m ∧ is_not_multiple_of_5 m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l575_575764


namespace smallest_square_with_5_lattice_points_l575_575408

-- Definitions based on conditions
def is_lattice_point (p : (ℤ × ℤ)) : Bool :=
  true -- any point with integer coordinates is a lattice point

def lattice_points_in_square (s : ℤ) : Finset (ℤ × ℤ) :=
  (Finset.Icc (0, 0) (s, s)).val

def is_on_boundary (p : (ℤ × ℤ)) (s : ℤ) : Bool :=
  p.1 = 0 ∨ p.1 = s ∨ p.2 = 0 ∨ p.2 = s

-- Proof statement
theorem smallest_square_with_5_lattice_points (s : ℤ) :
  (∃ s, lattice_points_in_square s.card = 5 ∧ (∃ B : Finset (ℤ × ℤ), B.card ≥ 3 ∧ B ⊆ lattice_points_in_square s ∧ ∀ b ∈ B, is_on_boundary b s)) → s = 2 :=
sorry

end smallest_square_with_5_lattice_points_l575_575408


namespace solution_set_of_inequality_l575_575872

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x + 1) * (1 - 2 * x) > 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} := 
by 
  sorry

end solution_set_of_inequality_l575_575872


namespace symmetric_circle_eq_l575_575641

-- Define the original circle equation
def originalCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the equation of the circle symmetric to the original with respect to the y-axis
def symmetricCircle (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y - 2) ^ 2 = 4

-- Theorem to prove that the symmetric circle equation is correct
theorem symmetric_circle_eq :
  ∀ x y : ℝ, originalCircle x y → symmetricCircle (-x) y := 
by
  sorry

end symmetric_circle_eq_l575_575641


namespace slower_pipe_time_to_fill_l575_575148

-- Define times for slower and faster pipe to fill the tank
def slower_pipe_time := ℕ
def faster_pipe_time := slower_pipe_time / 4

-- Combined rate of two pipes filling the tank
def combined_rate (slower_pipe_time : ℕ) := (1 / slower_pipe_time) + (4 / slower_pipe_time)

-- Rate when both pipes are working together
def combined_rate_known := 1 / 36

-- The proof statement we need to prove
theorem slower_pipe_time_to_fill (t : ℕ) (h1 : slower_pipe_time = t)
    (h2 : faster_pipe_time = t / 4)
    (h3 : combined_rate t = combined_rate_known) : 
    slower_pipe_time = 180 :=
by
  sorry

end slower_pipe_time_to_fill_l575_575148


namespace binomial_12_6_eq_924_l575_575835

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575835


namespace number_of_correct_propositions_l575_575458

theorem number_of_correct_propositions
  (P1 : ∀ (a b : Vector), collinear a b → (a = b ∨ a = -b))
  (P2 : ∀ (x : ℝ), |x| ≤ 3 → x ≤ 3)
  (P3 : ∀ (x : ℝ), (0 < x ∧ x < 2) → x^2 - 2*x - 3 ≥ 0)
  (P4 : ∀ (a : ℝ), a > 0 → ∀ (x : ℝ), (a = 1/2) → increasing (exp x) → ¬ increasing ((1/2)^x)) :
  num_correct (P1, P2, P3, P4) = 3 := sorry

end number_of_correct_propositions_l575_575458


namespace abs_difference_count_l575_575884

noncomputable def tau (n : ℕ) : ℕ := (finset.range n).filter (λ d, n % d = 0).card

noncomputable def S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).sum (λ k, tau (k + 1))

noncomputable def count_odd (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ k, S (k + 1) % 2 = 1).card

noncomputable def count_even (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ k, S (k + 1) % 2 = 0).card

theorem abs_difference_count (m : ℕ) : 
  |count_odd m - count_even m| = 61 :=
sorry

end abs_difference_count_l575_575884


namespace hyperbola_standard_equation_l575_575445

theorem hyperbola_standard_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (real.sqrt (a^2 + b^2)) / a = (real.sqrt 6) / 2)
  (h4 : (a * b) / (real.sqrt (a^2 + b^2)) = (2 * real.sqrt 6) / 3) :
  (∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ (a = 2*real.sqrt 2) ∧ (b = 2)) ∧ 
  (∀ (x y : ℝ), (x^2 / (a^2)) - (y^2 / (b^2)) = 1 ↔ (x^2 / 8) - (y^2 / 4) = 1) :=
by
  sorry

end hyperbola_standard_equation_l575_575445


namespace circleC_equation_minimum_tangent_length_l575_575018

noncomputable def F1 := (-1, 0 : ℝ)
noncomputable def F2 := (1, 0 : ℝ)
noncomputable def line : ℝ → ℝ → Prop := λ x y, x + y - 2 = 0

-- Definition for the circle C centered at (2, 2) with radius 1
noncomputable def circleC_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 1

-- Definition for the distance between a point and circle center (2, 2)
noncomputable def distance_to_C (m : ℝ) : ℝ := Real.sqrt ((m - 2)^2 + 2^2)

-- Statement to find the equation of circle C
theorem circleC_equation : ∃ (x y : ℝ), (circleC_eqn x y) :=
  sorry

-- Statement for the minimum length of the tangent line
theorem minimum_tangent_length : 
  ∃ (m : ℝ), ∃ (length : ℝ), distance_to_C m - 1 = Real.sqrt 3 ∧ (m = 2) :=
  sorry

end circleC_equation_minimum_tangent_length_l575_575018


namespace sum_of_digits_sqrt_N_l575_575874

theorem sum_of_digits_sqrt_N :
  let N := (10^2017 - 1) * 10^2019 / 9 + (10^2018 - 1) * 20 / 9 + 5 in
  let sqrt_N := (10^2018 + 5) / 3 in
  let sum_of_digits_of_sqrt_N := 2017 * 3 + 5 in
  sum_of_digits_of_sqrt_N = 6056 := 
sorry

end sum_of_digits_sqrt_N_l575_575874


namespace coefficient_x3_term_l575_575188

theorem coefficient_x3_term (n k : ℕ) (a b x : ℤ) (h_n : n = 7) (h_k : k = 3) (h_a : a = 1) (h_b : b = -2) : 
  (binomial n k) * b^k = -280 := by
  sorry

end coefficient_x3_term_l575_575188


namespace simplify_trig_identity_l575_575170

theorem simplify_trig_identity (x y : ℝ) : 
    sin (x + y) * sin (x - y) - cos (x + y) * cos (x - y) = -cos (2 * x) := 
sorry

end simplify_trig_identity_l575_575170


namespace force_required_8_inch_l575_575643

theorem force_required_8_inch
  (F L k : ℝ)
  (h_inv : F * L = k)
  (h_initial : F = 200)
  (h_length : L = 12) :
  (∃ F' : ℝ, F' * 8 = k ∧ F' = 300) :=
by
  -- Let k = F * L
  have h_k : k = 200 * 12 := by rw [h_initial, h_length, mul_comm]
  -- Simplifying the value of k
  have h_k_simp : k = 2400 := by norm_num [h_k]
  use 300
  -- Verify the calculated force satisfies the inverse relationship at 8 inches
  split
  { 
    change 300 * 8 = 2400, 
    norm_num 
  }
  { 
    refl 
  }

end force_required_8_inch_l575_575643


namespace binom_12_6_eq_924_l575_575813

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575813


namespace prob_at_least_3_correct_l575_575214

-- Define the probability of one patient being cured
def prob_cured : ℝ := 0.9

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of exactly 3 out of 4 patients being cured
def prob_exactly_3 : ℝ :=
  binomial 4 3 * prob_cured^3 * (1 - prob_cured)

-- Define the probability of all 4 patients being cured
def prob_all_4 : ℝ :=
  prob_cured^4

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_3 : ℝ :=
  prob_exactly_3 + prob_all_4

-- The theorem to prove
theorem prob_at_least_3_correct : prob_at_least_3 = 0.9477 :=
  by
  sorry

end prob_at_least_3_correct_l575_575214


namespace sum_of_possible_b_eq_60_l575_575281

theorem sum_of_possible_b_eq_60 :
  let rs := [(1, 24), (2, 12), (3, 8), (4, 6)] in
  ∑ (p : ℕ × ℕ) in rs, p.1 + p.2 = 60 :=
by
  sorry

end sum_of_possible_b_eq_60_l575_575281


namespace biggest_number_from_digits_l575_575692

theorem biggest_number_from_digits (a b c d : ℕ) (h1 : a = 1) (h2 : b = 4) (h3 : c = 9) (h4 : d = 5) :
  ∃ n : ℕ, n = 9541 ∧ (∀ x y z w : ℕ, x ∈ {a, b, c, d} ∧ y ∈ {a, b, c, d} ∧ z ∈ {a, b, c, d} ∧ w ∈ {a, b, c, d} ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w → 1000 * x + 100 * y + 10 * z + w ≤ n) :=
begin
  use 9541,
  split,
  { refl, },
  { intros x y z w hx hy hz hw hxy hxz hxw hyz hyw hzw,
    repeat { cases hx <|> cases hy <|> cases hz <|> cases hw;
      simp [h1, h2, h3, h4, hxy, hxz, hxw, hyz, hyw, hzw] },
    all_goals { norm_num,} }
end

end biggest_number_from_digits_l575_575692


namespace permutation_non_adjacent_formula_l575_575125

theorem permutation_non_adjacent_formula (n : ℕ) :
  let g_n := finset.card {σ : equiv.perm (fin (2 * n)) | ∀ k < n, σ k ≠ k + n ∧ σ (k + n) ≠ k} in
  g_n = ∑ k in finset.range (n + 1), (-1 : ℤ) ^ k * (nat.choose n k) * (2 ^ k) * (nat.factorial (2 * n - k)) :=
by
  sorry

end permutation_non_adjacent_formula_l575_575125


namespace nonzero_even_exists_from_step_2_l575_575535

def initial_sequence (i : ℤ) : ℤ :=
  if i = 0 then 1 else 0

def update_sequence (seq : ℤ → ℤ) (i : ℤ) : ℤ :=
  seq (i - 1) + seq (i + 1)

def n_step_sequence (n : ℕ) (i : ℤ) : ℤ :=
  nat.rec_on n
    (initial_sequence i)
    (λ n' seq, update_sequence seq i)

theorem nonzero_even_exists_from_step_2 :
  ∀ (n : ℕ), ∃ i, 2 ≤ n ∧ n_step_sequence n i % 2 = 0 ∧ n_step_sequence n i ≠ 0 :=
begin
  sorry
end

end nonzero_even_exists_from_step_2_l575_575535


namespace surface_area_of_given_cube_l575_575230

def point := (ℤ × ℤ × ℤ)

def distance (p1 p2 : point) : ℚ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)

def is_equilateral_triangle (p q r : point) : Prop :=
  distance p q = distance p r ∧ distance p r = distance q r

def is_cube (p q r : point) : Prop := 
  is_equilateral_triangle p q r ∧ distance p q = real.sqrt(98)

def cube_side_length_from_diagonal (d : ℚ) : ℚ := d / real.sqrt 2

noncomputable def surface_area_of_cube (side_length : ℚ) : ℚ := 6 * side_length ^ 2

theorem surface_area_of_given_cube :
  let P := (7,12,10) 
  let Q := (8,8,1)
  let R := (11,3,9)
  is_cube P Q R → surface_area_of_cube 7 = 294 :=
by
  intros
  sorry

end surface_area_of_given_cube_l575_575230


namespace value_of_f_neg_a_l575_575436

noncomputable def f (x : ℝ) : ℝ := x^3 + sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 1 - a^3 :=
by
  sorry

end value_of_f_neg_a_l575_575436


namespace suitable_candidate_l575_575762

noncomputable def S_A_squared : ℝ := 2.25
noncomputable def S_B_squared : ℝ := 1.81
noncomputable def S_C_squared : ℝ := 3.42

theorem suitable_candidate :
  S_B_squared < S_A_squared ∧ S_B_squared < S_C_squared :=
by {
  have h₁ : S_B_squared < S_A_squared := by sorry,
  have h₂ : S_B_squared < S_C_squared := by sorry,
  exact ⟨h₁, h₂⟩,
}

end suitable_candidate_l575_575762


namespace third_box_weight_l575_575730

def box1_height := 1 -- inches
def box1_width := 2 -- inches
def box1_length := 4 -- inches
def box1_weight := 30 -- grams

def box2_height := 3 * box1_height
def box2_width := 2 * box1_width
def box2_length := box1_length

def box3_height := box2_height
def box3_width := box2_width / 2
def box3_length := box2_length

def volume (height : ℕ) (width : ℕ) (length : ℕ) : ℕ := height * width * length

def weight (box1_weight : ℕ) (box1_volume : ℕ) (box3_volume : ℕ) : ℕ := 
  box3_volume / box1_volume * box1_weight

theorem third_box_weight :
  weight box1_weight (volume box1_height box1_width box1_length) 
  (volume box3_height box3_width box3_length) = 90 :=
by
  sorry

end third_box_weight_l575_575730


namespace binom_8_5_eq_56_l575_575369

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575369


namespace quadratic_real_roots_probability_l575_575080

theorem quadratic_real_roots_probability :
  (∫ (a : ℝ) in 0..1, ∫ (b : ℝ) in 0..1, indicator (λ p, p.1 ≥ 4 * p.2) ((1 : ℝ) : ℝ) (a, b)) = (1 / 8) :=
sorry

end quadratic_real_roots_probability_l575_575080


namespace correct_transformation_l575_575255

def transformationA := sqrt (1 + 7/9) = 4/3
def transformationB := real.cbrt 27 = 3
def transformationC := sqrt ((-4)^2) = 4
def transformationD := abs (sqrt 121) = 11

theorem correct_transformation : transformationD :=
by sorry

end correct_transformation_l575_575255


namespace increasing_intervals_of_f_max_value_of_g_l575_575482

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem increasing_intervals_of_f (k : ℤ) :
  StrictMonoOn f (Icc (k * Real.pi - Real.pi / 8) (k * Real.pi + 3 * Real.pi / 8)) := 
sorry

noncomputable def g (x : ℝ) : ℝ := sqrt 2 * Real.sin (x + Real.pi / 2)

theorem max_value_of_g : 
  (∀ x, g x ≤ sqrt 2) ∧ (∃ k : ℤ, ∀ x, g x = sqrt 2 ↔ x = 2 * k * Real.pi) := 
sorry

end increasing_intervals_of_f_max_value_of_g_l575_575482


namespace sale_saving_percentage_l575_575715

theorem sale_saving_percentage (P : ℝ) : 
  let original_price := 8 * P
  let sale_price := 6 * P
  let amount_saved := original_price - sale_price
  let percentage_saved := (amount_saved / original_price) * 100
  percentage_saved = 25 :=
by
  sorry

end sale_saving_percentage_l575_575715


namespace cafe_combination_l575_575273

theorem cafe_combination (dishes : ℕ) (choices_Alex : ℕ) (choices_Jordan : ℕ) :
  dishes = 12 → choices_Alex = dishes → choices_Jordan = dishes - 1 → choices_Alex * choices_Jordan = 132 :=
by
  intros h_dishes h_choices_Alex h_choices_Jordan
  rw [h_choices_Alex, h_choices_Jordan, h_dishes]
  sorry

end cafe_combination_l575_575273


namespace triangle_is_equilateral_l575_575649

-- Definitions and setup for the proof problem
variables {T : Type*} [EuclideanGeometry T] 
variables (a b c : ℝ) (m_a m_b m_c : ℝ) -- side lengths and medians
variables (t1 t2 t3 t4 t5 t6 : ℝ) -- areas of the smaller triangles
variables (r1 r2 r3 r4 : ℝ) -- radii of the inscribed circles

-- Condition: The medians divide the triangle into six smaller triangles of equal area
axiom equal_areas : t1 = t2 ∧ t3 = t4 ∧ t5 = t6

-- Condition: The areas of the smaller triangles satisfy the distribution
axiom area_distribution : 
  t1 + t2 + t3 = t4 + t5 + t6 ∧ 
  t1 + t5 + t6 = t2 + t3 + t4

-- Condition: The inscribed circles have the same radii
axiom equal_radii : r1 = r2 ∧ r3 = r4

-- Theorem to be proven
theorem triangle_is_equilateral (h1 : equal_areas) (h2 : area_distribution) (h3 : equal_radii) : 
  a = b ∧ b = c ∧ c = a := 
sorry

end triangle_is_equilateral_l575_575649


namespace vec_parallel_x_value_l575_575493

theorem vec_parallel_x_value :
  ∀ (x : ℝ), (∀ (k1 k2 : ℝ), k1 * 4 = k2 * -4 -> k1 = k2 * x -> x = -4): 
sorry

end vec_parallel_x_value_l575_575493


namespace product_in_A_l575_575003

def A : Set ℤ := { z | ∃ a b : ℤ, z = a^2 + 4 * a * b + b^2 }

theorem product_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := 
by
  sorry

end product_in_A_l575_575003


namespace num_ways_to_group_l575_575981

-- Defining the number of men and women
def num_men : ℕ := 4
def num_women : ℕ := 5

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Condition: at least one man and one woman in the group of four
def group_of_four (men women : ℕ) : Prop := men + women = 4 ∧ men ≥ 1 ∧ women ≥ 1

-- Condition: each group of two must contain one man and one woman
def valid_grouping (group1 group2 : set (ℕ × ℕ)) : Prop :=
  (∀ (g ∈ group1), g.1 = 1 ∧ g.2 = 1) ∧ (∀ (g ∈ group2), g.1 = 1 ∧ g.2 = 1)

-- Statement: number of ways to arrange the groups under given conditions
theorem num_ways_to_group :
  let ways_choose_2_men := binomial num_men 2 in
  let ways_choose_2_women := binomial num_women 2 in
  let ways_pair_remaining := binomial 3 2 in
  ways_choose_2_men * ways_choose_2_women * ways_pair_remaining = 180 := by
  sorry

end num_ways_to_group_l575_575981


namespace abs_diff_count_S_1000_l575_575885

def τ (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, d ∣ n).card

def S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).sum τ

def is_odd (n : ℕ) : bool := 
  n % 2 = 1

def count_odd_S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).count (λ x, is_odd (S x))

def count_even_S (n : ℕ) : ℕ := 
  (n + 1) - count_odd_S n

theorem abs_diff_count_S_1000 : 
  |(count_odd_S 1000) - (count_even_S 1000)| = 7 :=
by
  sorry

end abs_diff_count_S_1000_l575_575885


namespace range_of_g_l575_575932

def g (x : ℝ) : ℝ := (Real.sin x)^6 + (Real.cos x)^4

theorem range_of_g : Set.Icc 0.65 1 = {y : ℝ | ∃ x : ℝ, g(x) = y} := 
by 
  sorry

end range_of_g_l575_575932


namespace coefficients_proof_l575_575190

-- Define the given quadratic equation
def quadratic_eq := ∀ x : ℝ, x^2 + 2 = 3x

-- Define the standard form coefficients
def coefficients_quadratic (a b c : ℝ) :=
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2 = 3x

-- The proof problem
theorem coefficients_proof : coefficients_quadratic 1 (-3) 2 :=
by
  sorry

end coefficients_proof_l575_575190


namespace sphere_surface_area_l575_575289

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_area (r_circle r_distance : ℝ) :
  (Real.pi * r_circle^2 = 16 * Real.pi) →
  (r_distance = 3) →
  (surface_area_of_sphere (Real.sqrt (r_distance^2 + r_circle^2)) = 100 * Real.pi) := by
sorry

end sphere_surface_area_l575_575289


namespace both_reunions_attendance_l575_575234

theorem both_reunions_attendance (total_guests oates_guests hall_guests : ℕ) (h1 : total_guests = 100) (h2 : oates_guests = 50) (h3 : hall_guests = 62) : ∃ x, (oates_guests + hall_guests - x = total_guests) ∧ x = 12 :=
by
  use 12
  split
  sorry
  sorry

end both_reunions_attendance_l575_575234


namespace binom_12_6_eq_924_l575_575805

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575805


namespace problem_implies_l575_575371

variables (p q r : Prop)

def prop1 := ¬p ∧ q ∧ ¬r
def prop2 := p ∧ ¬q ∧ ¬r
def prop3 := ¬p ∧ ¬q ∧ r
def prop4 := p ∧ q ∧ ¬r

def implies_truth (prop : Prop) := ((¬p → ¬q) → ¬r)

theorem problem_implies : 
  (implies_truth prop1) ∧ 
  (implies_truth prop2) ∧ 
  (¬implies_truth prop3) ∧ 
  (implies_truth prop4) :=
by {
  sorry
}

lemma count_satisfying_propositions : 
  (count ([prop1, prop2, prop3, prop4].map implies_truth) (λb, b = true) = 3) :=
by {
  sorry
}

end problem_implies_l575_575371


namespace complex_quadrant_l575_575477

theorem complex_quadrant (h : (1:ℂ) / (1 + (1:ℂ)*complex.I) + complex.I = (1/2 + 1/2*complex.I : ℂ)) :
  (0 : ℝ) < 1/2 ∧ (0 : ℝ) < 1/2 :=
by
  sorry

end complex_quadrant_l575_575477


namespace collinear_EFN_l575_575279

-- Problem Definitions and Conditions
structure CyclicQuadrilateral (A B C D : Type _) := 
  (cyclic : ∃ (circumcircle : Circle), A ∈ circumcircle ∧ B ∈ circumcircle ∧ C ∈ circumcircle ∧ D ∈ circumcircle)

variables {A B C D E F M N : Type _}

structure ProblemConditions (A B C D E F M N : Type _) [CyclicQuadrilateral A B C D] :=
  (AD_inter_BC_at_E : ∃ E : Type _, AD.inter BC = E ∧ C ∈ (B..E))
  (AC_inter_BD_at_F : ∃ F : Type _, AC.inter BD = F)
  (midpoint_M_CD : M = midpoint C D)
  (N_on_circumcircle : ∃ circumcircle : Circle, N ∈ circumcircle)
  (N_AM_BM_ratio : (AN / BN) = (AM / BM))

-- The goal is to show E, F, N are collinear
theorem collinear_EFN (A B C D E F M N : Type _) [CyclicQuadrilateral A B C D]
  (conditions : ProblemConditions A B C D E F M N) : Collinear E F N :=
sorry

end collinear_EFN_l575_575279


namespace tan_theta_expr_l575_575912

theorem tan_theta_expr (θ : ℝ) (h : Real.tan θ = 4) : 
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + (Real.sin θ ^ 2) / 4 = 21 / 68 := 
by sorry

end tan_theta_expr_l575_575912


namespace binary_to_decimal_110011_l575_575381

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575381


namespace problem_statement_l575_575461

theorem problem_statement
  (x y : ℝ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 20) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 :=
  sorry

end problem_statement_l575_575461


namespace intersection_of_lines_l575_575853

theorem intersection_of_lines :
  ∃ (x y : ℝ), (y = 3 * x) ∧ (y + 3 = -9 * x) ∧ x = -1/4 ∧ y = -3/4 :=
by {
  use [-1/4, -3/4],
  split, 
  { calc
    -3/4 = 3 * (-1/4) : by norm_num
  },
  split,
  { calc
    -3/4 + 3 = -9 * (-1/4) : by norm_num
  },
  split; norm_num
}

end intersection_of_lines_l575_575853


namespace shortest_diagonal_probability_l575_575719

theorem shortest_diagonal_probability : 
  let n := 11 in
  let total_diagonals := n * (n - 3) / 2 in
  let shortest_diagonals := n in
  (shortest_diagonals / total_diagonals : ℚ) = 1 / 4 :=
by
  sorry

end shortest_diagonal_probability_l575_575719


namespace schur_problem_l575_575994

noncomputable def A : Finset ℕ := Finset.range 41 \ {0}

def is_valid_partition (parts : Finset (Finset ℕ)) : Prop :=
  (Finset.bUnion parts id = A) ∧ (∀ S ∈ parts, ∀ a b c ∈ S, a ≠ b + c)

theorem schur_problem :
  ∃ k : ℕ, k > 0 ∧ ∃ parts : Finset (Finset ℕ), parts.card = k ∧ is_valid_partition parts ∧ k = 4 :=
begin
  sorry
end

end schur_problem_l575_575994


namespace find_a_l575_575663

theorem find_a (x a : ℝ) (h : 2 * x + a - 8 = 0) (hx : x = 2) : a = 4 := 
by
  subst hx
  simp at h
  linarith
  sorry

end find_a_l575_575663


namespace problem1_solution_problem2_solution_l575_575266

-- Problem 1: f(x-2) = 3x - 5 implies f(x) = 3x + 1
def problem1 (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x - 2) = 3 * x - 5 → f x = 3 * x + 1

-- Problem 2: Quadratic function satisfying specific conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a*x^2 + b*x + c

def problem2 (f : ℝ → ℝ) : Prop :=
  is_quadratic f ∧
  (f 0 = 4) ∧
  (∀ x : ℝ, f (3 - x) = f x) ∧
  (∀ x : ℝ, f x ≥ 7/4) →
  (∀ x : ℝ, f x = x^2 - 3*x + 4)

-- Statements to be proved
theorem problem1_solution : ∀ f : ℝ → ℝ, problem1 x f := sorry
theorem problem2_solution : ∀ f : ℝ → ℝ, problem2 f := sorry

end problem1_solution_problem2_solution_l575_575266


namespace f_decreasing_intervals_g_range_l575_575483

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + 2 * sqrt 3 * cos x ^ 2 - sqrt 3

def decreasing_interval (k : ℤ) : set ℝ :=
  {x | k * π + π / 12 ≤ x ∧ x ≤ (7 * π / 12) + k * π}

noncomputable def g (x : ℝ) : ℝ := 2 * sin (4 * x + 2 * π / 3)

def interval : set ℝ := {x | -π / 12 < x ∧ x < π / 8}

def range_g : set ℝ := {y | -1 < y ∧ y ≤ 2}

theorem f_decreasing_intervals (k : ℤ) :
  ∀ x ∈ decreasing_interval k, deriv f x < 0 := sorry

theorem g_range :
  set.image g interval = range_g := sorry

end f_decreasing_intervals_g_range_l575_575483


namespace angle_B_acute_fraction_increase_l575_575725

-- Problem 1 in Lean 4
theorem angle_B_acute (A B C : Type) [add_group A] [linear_order B] 
  (angle_C_right : ∠C = 90°) (triangle_ABC : ∠A + ∠B + ∠C = 180°) 
  (angle_sum (a b c : B) [add_group A] [linear_order B] (a : ∠A) (b : ∠B) (c : ∠C) : ∠A + ∠B + ∠C = 180°) : 
  (∠B < 90°) := 
by 
  sorry

-- Problem 2 in Lean 4
theorem fraction_increase (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
  (b + m) / (a + m) > b / a :=
by 
  sorry

end angle_B_acute_fraction_increase_l575_575725


namespace binom_eight_five_l575_575361

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575361


namespace solve_for_x_l575_575955

theorem solve_for_x (x : ℝ) : (sqrt(3 + 2 * sqrt(x)) = 4) → x = 169 / 4 :=
by
  intro h
  -- to complete the proof
  sorry

end solve_for_x_l575_575955


namespace card_numbers_l575_575142

noncomputable def card_arrangement : list ℕ := [A, B, C]

def conditions (l : list ℕ) : Prop :=
  l.length = 9 ∧
  ∀ i, i < 7 → ((l[i] < l[i+1] ∧ l[i+1] < l[i+2]) → false) ∧
               ((l[i] > l[i+1] ∧ l[i+1] > l[i+2]) → false)

theorem card_numbers :
  ∃ (A B C : ℕ),
    conditions [1, 3, 4, A, 6, 8, B, 7, C] ∧
    A = 5 ∧ B = 2 ∧ C = 9 :=
sorry

end card_numbers_l575_575142


namespace max_y_for_f_eq_0_l575_575133

-- Define f(x, y, z) as the remainder when (x - y)! is divided by (x + z).
def f (x y z : ℕ) : ℕ :=
  Nat.factorial (x - y) % (x + z)

-- Conditions given in the problem
variable (x y z : ℕ)
variable (hx : x = 100)
variable (hz : z = 50)

theorem max_y_for_f_eq_0 : 
  f x y z = 0 → y ≤ 75 :=
by
  rw [hx, hz]
  sorry

end max_y_for_f_eq_0_l575_575133


namespace ticket_prices_count_l575_575979

theorem ticket_prices_count :
  let y := 30
  let divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  ∀ (k : ℕ), (k ∈ divisors) ↔ (60 % k = 0 ∧ 90 % k = 0) → 
  (∃ n : ℕ, n = 8) :=
by
  sorry

end ticket_prices_count_l575_575979


namespace number_of_multiples_of_4_between_100_and_350_l575_575951

theorem number_of_multiples_of_4_between_100_and_350 :
  (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≥ 104 ∧ (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≤ 348 →
  (set.filter (λ x, x % 4 = 0) (finset.Icc 100 350).to_set).card = 62 :=
by
  sorry

end number_of_multiples_of_4_between_100_and_350_l575_575951


namespace sequence_1000th_term_l575_575429

theorem sequence_1000th_term :
  (∀ n : ℕ, 0 < n → (∑ i in finset.range n, (λ i, a i)) / n = n + 2)
  → a 1000 = 1001 := 
begin
  sorry
end

end sequence_1000th_term_l575_575429


namespace jose_is_21_l575_575991

-- Define the ages of the individuals based on the conditions
def age_of_inez := 12
def age_of_zack := age_of_inez + 4
def age_of_jose := age_of_zack + 5

-- State the proposition we want to prove
theorem jose_is_21 : age_of_jose = 21 := 
by 
  sorry

end jose_is_21_l575_575991


namespace find_k_l575_575057

-- Definitions for vectors and their operations
structure Vector2 where
  x : ℝ
  y : ℝ

instance : Add Vector2 where
  add := λ a b, Vector2.mk (a.x + b.x) (a.y + b.y)

instance : Neg Vector2 where
  neg := λ a, Vector2.mk (-a.x) (-a.y)

instance : Sub Vector2 where
  sub := λ a b, a + -b

def dot (a b : Vector2) : ℝ := a.x * b.x + a.y * b.y

def scalarMul (r : ℝ) (v : Vector2) : Vector2 := Vector2.mk (r * v.x) (r * v.y)

-- Conditions as definitions
def a : Vector2 := Vector2.mk 3 (-4)
def b (k : ℝ) : Vector2 := Vector2.mk ((k + 1 - 3)/2) ((k - 4 + 4)/2)

def isOrthogonal (v1 v2 : Vector2) : Prop := dot v1 v2 = 0

-- Theorem to prove k = -6
theorem find_k (k : ℝ) (h : isOrthogonal a (b k)) : k = -6 :=
by
  unfold isOrthogonal at h
  unfold dot at h
  unfold a at h
  unfold b at h
  sorry

end find_k_l575_575057


namespace find_b_of_tangent_circle_l575_575936

theorem find_b_of_tangent_circle
  (b : ℝ)
  (C : ∀ (x y : ℝ), y^2 = 2 * x)
  (l : ∀ (x y : ℝ), y = - (1 / 2) * x + b)
  (tangent_x_axis : ∀ (x1 y1 x2 y2 : ℝ), 
    C x1 y1 → l x1 y1 → C x2 y2 → l x2 y2 → 
    ∃ (x0 y0 : ℝ), 
      (x0 = (x1 + x2) / 2) ∧ 
      (y0 = (y1 + y2) / 2) ∧ 
      (y0 = - 2) ∧ 
      (4 * Real.sqrt 5 * Real.sqrt (1 + b) = 4)) : 
  b = -4 / 5 :=
sorry

end find_b_of_tangent_circle_l575_575936


namespace binom_8_5_eq_56_l575_575347

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575347


namespace plane_exists_with_given_conditions_l575_575846

def plane1 (x y z : ℝ) : Prop := x + 3 * y + 4 * z = 3
def plane2 (x y z : ℝ) : Prop := 2 * x - 2 * y + 3 * z = 6
def lineM (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

noncomputable def planeQ (x y z : ℝ) : Prop := 10 * x - 18 * y + z - 21 = 0

theorem plane_exists_with_given_conditions :
  ∃ (a b c d : ℤ), (∀ (x y z : ℝ), (planeQ x y z ↔
    (a * x + b * y + c * z + d = 0)) ∧ a = 10 ∧ b = -18 ∧ c = 1 ∧ d = -21) ∧
    let point : ℝ × ℝ × ℝ := (4, 2, -2) in
    abs (10 * (point.1) - 18 * (point.2) + (point.3) - 21) / 
    sqrt (10^2 + (-18)^2 + 1^2) = 3 / sqrt 5 := 
by sorry

end plane_exists_with_given_conditions_l575_575846


namespace num_even_functions_correct_l575_575933

def f1 (x : ℝ) : ℝ := x^2
def f2 (x : ℝ) : ℝ := Real.log x
def f3 (x : ℝ) : ℝ := 2^x - 2^(-x)
def f4 (x : ℝ) : ℝ := 2^x + 2^(-x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def num_even_functions : Nat :=
  (if is_even_function f1 then 1 else 0) +
  (if is_even_function f2 then 1 else 0) +
  (if is_even_function f3 then 1 else 0) +
  (if is_even_function f4 then 1 else 0)

theorem num_even_functions_correct : num_even_functions = 2 := 
by 
  sorry

end num_even_functions_correct_l575_575933


namespace problem1_problem2_l575_575546

theorem problem1 (A B C : ℝ) (a b c : ℝ) (D : ℝ) (hA : A = π / 3) (hb : b = 2) (hc : c = 3) (area : ℝ):
  (triangle.area a b c A = 6 * real.sqrt 3) →
  (a * tan A⁻¹ + b * sin B = real.sqrt 21) :=
sorry

theorem problem2 (A B C : ℝ) (a b c : ℝ) (D : ℝ) (hA : A = π / 3) (hb : b = 2) (hc : c = 3) (hD : D * D = 1 / 4 * (c^2 + b^2 + b * c)) :
  (D ≥ 3 * real.sqrt 2) :=
sorry

end problem1_problem2_l575_575546


namespace integer_values_of_n_l575_575882

theorem integer_values_of_n :
  {n : ℤ | 16000 * (2 / 7 : ℚ)^n ∈ ℤ}.finite.card = 10 :=
by
  sorry

end integer_values_of_n_l575_575882


namespace binom_8_5_l575_575340

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575340


namespace number_of_integers_in_range_l575_575069

theorem number_of_integers_in_range : 
  let count_satisfying (f : ℕ → Prop) (a b : ℕ) := ((list.range' a (b - a)).filter f).card in
  let condition n := (150 * n) ^ 25 > n ^ 50 ∧ n ^ 50 > 3 ^ 150 in
  count_satisfying condition 28 150 = 122 := by
sorry

end number_of_integers_in_range_l575_575069


namespace binom_12_6_eq_924_l575_575812

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575812


namespace frustum_slant_height_l575_575290

theorem frustum_slant_height
  (ratio_area : ℝ)
  (slant_height_removed : ℝ)
  (sf_ratio : ratio_area = 1/16)
  (shr : slant_height_removed = 3) :
  ∃ (slant_height_frustum : ℝ), slant_height_frustum = 9 :=
by
  sorry

end frustum_slant_height_l575_575290


namespace cooking_pottery_only_l575_575284

noncomputable def num_yoga : ℕ := 50
noncomputable def num_cooking : ℕ := 30
noncomputable def num_weaving : ℕ := 20
noncomputable def num_pottery : ℕ := 15
noncomputable def num_dancing : ℕ := 10

noncomputable def num_yoga_cooking : ℕ := 20
noncomputable def num_yoga_weaving : ℕ := 13
noncomputable def num_yoga_pottery : ℕ := 9
noncomputable def num_yoga_dancing : ℕ := 7
noncomputable def num_cooking_weaving : ℕ := 10
noncomputable def num_cooking_pottery : ℕ := 4
noncomputable def num_cooking_dancing : ℕ := 5
noncomputable def num_weaving_pottery : ℕ := 3
noncomputable def num_weaving_dancing : ℕ := 2
noncomputable def num_pottery_dancing : ℕ := 6

noncomputable def num_yoga_cooking_weaving : ℕ := 9
noncomputable def num_yoga_cooking_pottery : ℕ := 3
noncomputable def num_yoga_cooking_dancing : ℕ := 2
noncomputable def num_yoga_weaving_pottery : ℕ := 4
noncomputable def num_yoga_weaving_dancing : ℕ := 1
noncomputable def num_cooking_weaving_pottery : ℕ := 2
noncomputable def num_cooking_weaving_dancing : ℕ := 1
noncomputable def num_cooking_pottery_dancing : ℕ := 3

noncomputable def num_all_activities : ℕ := 5

theorem cooking_pottery_only :
  (num_cooking_pottery - num_yoga_cooking_pottery - num_cooking_weaving_pottery - num_cooking_pottery_dancing) = 0 :=
by
  have h1 := num_cooking_pottery - num_yoga_cooking_pottery
  have h2 := h1 - num_cooking_weaving_pottery
  have h3 := h2 - num_cooking_pottery_dancing
  exact eq.trans h3 0 rfl

end cooking_pottery_only_l575_575284


namespace find_interest_rate_approximation_l575_575964

noncomputable def interest_rate (initial final : ℝ) (years : ℕ) : ℝ :=
  (70 / ((years / 2) : ℝ))

theorem find_interest_rate_approximation :
  interest_rate 8000 32000 18 ≈ 7.78 := 
sorry

end find_interest_rate_approximation_l575_575964


namespace area_of_triangle_l575_575158

theorem area_of_triangle (m1 m2 m3 : ℝ) 
  (h1 : m1 > 0) 
  (h2 : m2 > 0) 
  (h3 : m3 > 0) : 
  let 
    u := (1/2) * (1/m1 + 1/m2 + 1/m3),
    u1 := u - 1/m1,
    u2 := u - 1/m2,
    u3 := u - 1/m3
  in 
  4 * Real.sqrt(u * u1 * u2 * u3) = (4 / Real.sqrt(u * u1 * u2 * u3)) := 
sorry

end area_of_triangle_l575_575158


namespace prism_height_relation_l575_575454

theorem prism_height_relation (a b c h : ℝ) 
  (h_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_height : 0 < h) 
  (h_right_angles : true) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 :=
by 
  sorry 

end prism_height_relation_l575_575454


namespace cos_double_angle_l575_575893

open Real

theorem cos_double_angle {α β : ℝ} (h1 : sin α = sqrt 5 / 5)
                         (h2 : sin (α - β) = - sqrt 10 / 10)
                         (h3 : 0 < α ∧ α < π / 2)
                         (h4 : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 0 :=
  sorry

end cos_double_angle_l575_575893


namespace binom_12_6_eq_924_l575_575809

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575809


namespace x_n_is_integer_l575_575001

def x : ℕ → ℤ
| 0       := 0
| (n + 1) := 5 * x n + Int.sqrt (24 * (x n) ^ 2 + 1)

theorem x_n_is_integer (n : ℕ) : ∀ n, x n ∈ ℤ := sorry

end x_n_is_integer_l575_575001


namespace table_ways_even_ones_l575_575844

theorem table_ways_even_ones (m n : ℕ) : 
  (∃ (T : Fin m → Fin n → ℕ), 
    (∀ i, (∑ j, T i j) % 2 = 0) ∧ 
    (∀ j, (∑ i, T i j) % 2 = 0)) ↔
  (2 ^ ((m-1) * (n-1))) :=
by
  sorry

end table_ways_even_ones_l575_575844


namespace beatrix_installations_l575_575879

def installation := list (list string) -- Each cell can be "black", "grey" or "spotted"
def n_spotted_cells := 1
def n_grey_cells := 1
def n_black_cells := 2

def count_installations : ℕ :=
4 * 3

theorem beatrix_installations :
  count_installations = 12 :=
by {
  unfold count_installations,
  -- Each position for the spotted cell (4 positions)
  -- For each position of the spotted cell, there are 3 possibilities for the grey cell, giving us 4*3=12 arrangements
  sorry
}

end beatrix_installations_l575_575879


namespace water_depth_in_cylindrical_tub_l575_575280

theorem water_depth_in_cylindrical_tub
  (tub_diameter : ℝ) (tub_depth : ℝ) (pail_angle : ℝ)
  (h_diam : tub_diameter = 40)
  (h_depth : tub_depth = 50)
  (h_angle : pail_angle = 45) :
  ∃ water_depth : ℝ, water_depth = 30 :=
by
  sorry

end water_depth_in_cylindrical_tub_l575_575280


namespace circumcircle_midpoint_DE_l575_575411

-- Define the geometric conditions
variables {α : Type} [inner_product_space ℝ α]
variables {A B C D E : α}

-- Right-angled triangle ABC with right angle at C
variable (hABC : ∠ A C B = real.pi / 2)

-- Squares constructed on legs BC and CA
variables (hBC_square : ∃ D, is_square B C D (C + (B - C).perpendicular))
variables (hCA_square : ∃ E, is_square C A E (A + (A - C).perpendicular))

-- Midpoint M of segment DE
def M (D E : α) : α := midpoint ℝ D E

-- Circumcircle passes through M
theorem circumcircle_midpoint_DE (hABC : ∠ A C B = real.pi / 2)
  (hBC_square : ∃ D, is_square B C D (C + (B - C).perpendicular))
  (hCA_square : ∃ E, is_square C A E (A + (A - C).perpendicular)) :
  let M := M D E in
  point_on_circumcircle {A B C} M :=
begin
  sorry
end

end circumcircle_midpoint_DE_l575_575411


namespace finds_sv_l575_575060

noncomputable def vectors_fixed_distance (a b p : ℝ^n) (s v : ℝ) : Prop :=
∥p - a∥ = 3 * ∥p - b∥ → 
∥p - (s • a + v • b)∥ = ∥p∥ - ( (∥p∥^2 = (18/8) * (b • p) - (2/8) * (a • p) - (1/8) * ∥a∥^2 + (9/8) * ∥b∥^2) / 2)

theorem finds_sv (a b : ℝ^n) :
  ∃ s v : ℝ, vectors_fixed_distance a b (λ s, 1/8) (λ v, 9/8) :=
begin
  sorry,
end

end finds_sv_l575_575060


namespace selling_price_correct_l575_575748

def cost_price : ℝ := 20
def loss_fraction : ℝ := 1 / 6
def selling_price : ℝ := cost_price * (1 - loss_fraction)

theorem selling_price_correct :
  selling_price = 16.67 := by
  sorry

end selling_price_correct_l575_575748


namespace quotient_is_six_l575_575197

def larger_number (L : ℕ) : Prop := L = 1620
def difference (L S : ℕ) : Prop := L - S = 1365
def division_remainder (L S Q : ℕ) : Prop := L = S * Q + 15

theorem quotient_is_six (L S Q : ℕ) 
  (hL : larger_number L) 
  (hdiff : difference L S) 
  (hdiv : division_remainder L S Q) : Q = 6 :=
sorry

end quotient_is_six_l575_575197


namespace total_amount_spent_by_jim_is_50_l575_575519

-- Definitions for conditions
def cost_per_gallon_nc : ℝ := 2.00  -- Cost per gallon in North Carolina
def gallons_nc : ℕ := 10  -- Gallons bought in North Carolina
def additional_cost_per_gallon_va : ℝ := 1.00  -- Additional cost per gallon in Virginia
def gallons_va : ℕ := 10  -- Gallons bought in Virginia

-- Definition for total cost in North Carolina
def total_cost_nc : ℝ := gallons_nc * cost_per_gallon_nc

-- Definition for cost per gallon in Virginia
def cost_per_gallon_va : ℝ := cost_per_gallon_nc + additional_cost_per_gallon_va

-- Definition for total cost in Virginia
def total_cost_va : ℝ := gallons_va * cost_per_gallon_va

-- Definition for total amount spent
def total_spent : ℝ := total_cost_nc + total_cost_va

-- Theorem to prove
theorem total_amount_spent_by_jim_is_50 : total_spent = 50.00 :=
by
  -- Place proof here
  sorry

end total_amount_spent_by_jim_is_50_l575_575519


namespace quadratic_root_a_l575_575448

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 = 0 ∧ x = 1) → a = -5 :=
by
  intro h
  have h1 : (1:ℝ)^2 + a * (1:ℝ) + 4 = 0 := sorry
  linarith

end quadratic_root_a_l575_575448


namespace speed_of_man_l575_575727

theorem speed_of_man (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_km_per_hr : ℝ) : 
  length_of_train = 1200 →
  time_to_cross = 71.99424046076314 →
  speed_of_train_km_per_hr = 63 →
  let speed_of_train := speed_of_train_km_per_hr * 1000 / 3600 in
  let distance_covered_by_train := speed_of_train * time_to_cross in
  let speed_of_man := (distance_covered_by_train - length_of_train) / time_to_cross in
  let speed_of_man_km_per_hr := speed_of_man * 3600 / 1000 in
  speed_of_man_km_per_hr = 3 :=
by
  intros
  sorry

end speed_of_man_l575_575727


namespace binomial_12_6_l575_575816

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575816


namespace monthly_savings_correct_l575_575594

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l575_575594


namespace parallel_implies_equal_slope_equal_slope_implies_possibly_parallel_or_coincide_parallel_sufficient_not_necessary_l575_575564

-- Definitions for the slopes of the lines
variables {k1 k2 : Real}
variables {l1 l2 : Line}

-- Definitions for parallelism and equality of slopes
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_implies_equal_slope (h : parallel l1 l2) : k1 = k2 :=
  sorry

theorem equal_slope_implies_possibly_parallel_or_coincide (h : k1 = k2) :
  l1.slope = k1 ∧ l2.slope = k2 → (parallel l1 l2 ∨ l1 = l2) :=
  sorry

theorem parallel_sufficient_not_necessary (h1 : parallel l1 l2) (h2 : k1 = k2) :
  (parallel l1 l2 → k1 = k2) ∧ ¬(k1 = k2 → parallel l1 l2) :=
  sorry

end parallel_implies_equal_slope_equal_slope_implies_possibly_parallel_or_coincide_parallel_sufficient_not_necessary_l575_575564


namespace n_digit_numbers_with_1_2_3_l575_575002

theorem n_digit_numbers_with_1_2_3 (n : ℕ) (h : 2 * n ≥ 3) : 
  let total := 3^n in
  let exclusion_set := 3 * 2^n - 3 in
  total - exclusion_set = 3^n - 3 * 2^n + 3 :=
by sorry

end n_digit_numbers_with_1_2_3_l575_575002


namespace henry_total_fee_8_bikes_l575_575435

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l575_575435


namespace lee_can_make_cookies_l575_575551

def cookies_per_cup_of_flour (cookies : ℕ) (flour_cups : ℕ) : ℕ :=
  cookies / flour_cups

def flour_needed (sugar_cups : ℕ) (flour_to_sugar_ratio : ℕ) : ℕ :=
  sugar_cups * flour_to_sugar_ratio

def total_cookies (cookies_per_cup : ℕ) (total_flour : ℕ) : ℕ :=
  cookies_per_cup * total_flour

theorem lee_can_make_cookies
  (cookies : ℕ)
  (flour_cups : ℕ)
  (sugar_cups : ℕ)
  (flour_to_sugar_ratio : ℕ)
  (h1 : cookies = 24)
  (h2 : flour_cups = 4)
  (h3 : sugar_cups = 3)
  (h4 : flour_to_sugar_ratio = 2) :
  total_cookies (cookies_per_cup_of_flour cookies flour_cups)
    (flour_needed sugar_cups flour_to_sugar_ratio) = 36 :=
by
  sorry

end lee_can_make_cookies_l575_575551


namespace george_price_bound_by_2_dave_price_l575_575720

def price (seq : List ℝ) : ℝ :=
  (List.inits seq).map (List.sum ∘ abs).maximum'

def dave_price (seq : List ℝ) : ℝ :=
  (List.permutations seq).map price).minimum'

def george_price (seq : List ℝ) : ℝ :=
  seq.foldl (λ acc x, max acc (abs (acc + x))) 0

theorem george_price_bound_by_2_dave_price {n : ℕ} (seq : List ℝ) :
  george_price seq ≤ 2 * dave_price seq := 
sorry

end george_price_bound_by_2_dave_price_l575_575720


namespace sticks_needed_for_4x4_square_largest_square_with_100_sticks_l575_575091

-- Problem a)
def sticks_needed_for_square (n: ℕ) : ℕ := 2 * n * (n + 1)

theorem sticks_needed_for_4x4_square : sticks_needed_for_square 4 = 40 :=
by
  sorry

-- Problem b)
def max_square_side_length (total_sticks : ℕ) : ℕ × ℕ :=
  let n := Nat.sqrt (total_sticks / 2)
  if 2*n*(n+1) <= total_sticks then (n, total_sticks - 2*n*(n+1)) else (n-1, total_sticks - 2*(n-1)*n)

theorem largest_square_with_100_sticks : max_square_side_length 100 = (6, 16) :=
by
  sorry

end sticks_needed_for_4x4_square_largest_square_with_100_sticks_l575_575091


namespace coefficients_proof_l575_575189

-- Define the given quadratic equation
def quadratic_eq := ∀ x : ℝ, x^2 + 2 = 3x

-- Define the standard form coefficients
def coefficients_quadratic (a b c : ℝ) :=
  ∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2 = 3x

-- The proof problem
theorem coefficients_proof : coefficients_quadratic 1 (-3) 2 :=
by
  sorry

end coefficients_proof_l575_575189


namespace find_x_when_z_is_48_l575_575256

constant m n : ℝ
constant x y z : ℝ
constant k : ℝ

axiom proportionality_conditions :
  (x = m * y^3) ∧ (y = n / z^2) ∧ (x = 2 ∧ z = 4)

noncomputable def value_of_x (z : ℝ) : ℝ :=
  k / z^6

theorem find_x_when_z_is_48 :
  (k = 8192) → (z = 48) → (value_of_x z = 1 / 1492992) :=
by
  intros h_k h_z
  rw [h_k, h_z]
  sorry

end find_x_when_z_is_48_l575_575256


namespace fare_collected_from_I_class_l575_575325

theorem fare_collected_from_I_class (x y : ℕ) 
  (h_ratio_passengers : 4 * x = 4 * x) -- ratio of passengers 1:4
  (h_ratio_fare : 3 * y = 3 * y) -- ratio of fares 3:1
  (h_total_fare : 7 * 3 * x * y = 224000) -- total fare Rs. 224000
  : 3 * x * y = 96000 := 
by
  sorry

end fare_collected_from_I_class_l575_575325


namespace semiperimeter_ratio_eq_circumradius_inradius_ratio_l575_575130

open Real

section
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (triangle_ABC : Triangle A B C)
variables (p q R r : ℝ)
variables (is_acute : Triangle.isAcute triangle_ABC)
variables (h1 : semiperimeter triangle_ABC = p)
variables (h2 : semiperimeter (triangle_formed_by_feet_of_altitudes triangle_ABC) = q)
variables (h3 : circumradius triangle_ABC = R)
variables (h4 : inradius triangle_ABC = r)

theorem semiperimeter_ratio_eq_circumradius_inradius_ratio :
  p / q = R / r :=
sorry
end

end semiperimeter_ratio_eq_circumradius_inradius_ratio_l575_575130


namespace slope_of_arithmetic_sequence_l575_575905

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (a_1 d n : α) : α := n * a_1 + n * (n-1) / 2 * d

theorem slope_of_arithmetic_sequence (a_1 d n : α) 
  (hS2 : S a_1 d 2 = 10)
  (hS5 : S a_1 d 5 = 55)
  : (a_1 + 2 * d - a_1) / 2 = 4 :=
by
  sorry

end slope_of_arithmetic_sequence_l575_575905


namespace binom_12_6_l575_575792

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575792


namespace smallest_y_value_l575_575253

theorem smallest_y_value (y : ℚ) (h : y / 7 + 2 / (7 * y) = 1 / 3) : y = 2 / 3 :=
sorry

end smallest_y_value_l575_575253


namespace binomial_evaluation_l575_575830

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575830


namespace part1_part2_part3_l575_575047

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2
noncomputable def h (x : ℝ) : ℝ := f (x + 1) - g' x

theorem part1 : ∃ x : ℝ, (h x ≤ 2) := sorry

theorem part2 (a b : ℝ) (h1 : 0 < b) (h2 : b < a) : 
  f (a + b) - f (2 * a) < (b - a) / (2 * a) := sorry

theorem part3 (k : ℤ) : (∀ x : ℝ, x > 1 → k * (x - 1) < x * f x + 3 * g' x + 4) ↔ k ≤ 5 := sorry

end part1_part2_part3_l575_575047


namespace probability_log_product_neg_less_than_zero_l575_575055

theorem probability_log_product_neg_less_than_zero :
  let elems := {0.3, 0.5, 3, 4, 5, 6}
  let choose_three_distinct (set : Set ℝ) : Finset (ℝ × ℝ × ℝ) :=
    {p | p.1 ≠ p.2 ∧ p.1 ≠ p.3 ∧ p.2 ≠ p.3 ∧ p.1 ∈ set ∧ p.2 ∈ set ∧ p.3 ∈ set}.to_finset
  let condition (p : ℝ × ℝ × ℝ) := log p.1 * log p.2 * log p.3 < 0
  let probability (set : Set ℝ) : ℚ :=
    (choose_three_distinct set).count condition / (choose_three_distinct set).card

  probability elems = 3 / 5 :=
sorry

end probability_log_product_neg_less_than_zero_l575_575055


namespace function_count_mod_103_l575_575552

noncomputable def countFunctions (n : ℕ) : ℕ :=
  101^100 + 100!

theorem function_count_mod_103 :
  (countFunctions 101) % 103 = 77 :=
sorry

end function_count_mod_103_l575_575552


namespace polynomial_inequality_l575_575745

variable {ℤ : Type*} [Int.valued ℤ]

def polynomial (p : ℤ → ℤ) : Prop :=
∀ (a b : ℤ), p a - p b = (a - b) * (p (a + b))

theorem polynomial_inequality (p : ℤ → ℤ) (n : ℤ)
  (hp : polynomial p)
  (h : p (-n) < p n ∧ p n < n) : 
  p (-n) < -n := 
sorry

end polynomial_inequality_l575_575745


namespace part_a_part_b_l575_575198

open Real

variables {A B C G : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup G]
variables (a b c g_a g_b g_c r : ℝ)

-- Distances from centroid G to sides a, b, and c
def distances_from_centroid := g_a + g_b + g_c

-- Inradius of the triangle
def inradius := r

-- Part (a) proof goal: g_i ≥ 2/3 * r for i = a, b, c
theorem part_a (h₁ : ∀ g, g ∈ {g_a, g_b, g_c}) : g_a ≥ (2 / 3) * r ∧ g_b ≥ (2 / 3) * r ∧ g_c ≥ (2 / 3) * r :=
  sorry

-- Part (b) proof goal: g_a + g_b + g_c ≥ 3r
theorem part_b (h₂ : distances_from_centroid = g_a + g_b + g_c) : distances_from_centroid ≥ 3 * r :=
  sorry

end part_a_part_b_l575_575198


namespace sequence_general_formula_l575_575449

noncomputable def a (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n + 1) - 3

theorem sequence_general_formula (n : ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 3) → a n = 2 ^ (n + 1) - 3 :=
by 
  intros h1 h2,
  sorry

end sequence_general_formula_l575_575449


namespace part1_part2_l575_575265

def problem1 : ℝ:= 
  (9 / 4 : ℝ) ^ (1 / 2) - 1 - (27 / 8 : ℝ) ^ (-2 / 3) + (2 / 3 : ℝ) ^ 2 + (1 / 8 : ℝ) ^ (1 / 3) 

theorem part1 : problem1 = 1 := sorry

def problem2 (log : ℝ → ℝ → ℝ) : ℝ := 
  (log 9 (sqrt 3)) + 2 ^ (1 / log 3 2)

theorem part2 {log : ℝ → ℝ → ℝ} (log_div_log : ∀ b n : ℝ, b > 0 → log b n = (Real.log n) / (Real.log b)) : 
  problem2 log = 7 := sorry

end part1_part2_l575_575265


namespace find_b_find_theta_l575_575019

namespace VectorProof

def vec_a : ℝ × ℝ := (1, 2)
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def is_parallel (u v : ℝ × ℝ) := ∃ k : ℝ, v = (k * u.1, k * u.2)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_b
  (vec_b : ℝ × ℝ)
  (h₁ : magnitude vec_b = 2 * real.sqrt 5)
  (h₂ : is_parallel vec_a vec_b) :
  vec_b = (2, 4) ∨ vec_b = (-2, -4) :=
begin
  sorry
end

theorem find_theta
  (vec_c : ℝ × ℝ)
  (h₁ : magnitude vec_c = real.sqrt 10)
  (h₂ : is_perpendicular (2 * vec_a.1, 2 * vec_a.2 + vec_c.2) (4 * vec_a.1 - 3 * vec_c.1, 4 * vec_a.2 - 3 * vec_c.2))
  (h₃ : is_perpendicular (2 * vec_a.1 + vec_c.1, 2 * vec_a.2 + vec_c.2) (4 * vec_a.1 - 3 * vec_c.1, 4 * vec_a.2 - 3 * vec_c.2)) :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ real.pi ∧ real.cos θ = (real.sqrt 2 / 2) :=
begin
  sorry
end

end VectorProof

end find_b_find_theta_l575_575019


namespace calculate_difference_l575_575312

def total_paid (A B C D : ℝ) : ℝ := A + B + C + D

def evenly_split (total people : ℝ) : ℝ := total / people

def owes (should_pay paid : ℝ) : ℝ := should_pay - paid

theorem calculate_difference 
    (A B C D : ℝ) -- Amounts paid by Alice, Bob, Charlie, and Dana
    (total : ℝ := total_paid A B C D) -- Total amount paid
    (people : ℝ := 4) -- Number of people
    (each_share : ℝ := evenly_split total people) -- Amount each person should have paid
    (a : ℝ := owes each_share A) -- Alice's owed amount
    (b : ℝ := owes each_share B) -- Bob's owed amount
    (c : ℝ := owes each_share C) -- Charlie's owed amount (not relevant in the final calculation)
    : a - b = 30 :=
by
  sorry

end calculate_difference_l575_575312


namespace combination_8_5_l575_575357

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575357


namespace binom_8_5_l575_575342

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575342


namespace red_then_blue_probability_l575_575705

-- Define the given conditions
def total_marbles : ℕ := 5 + 4 + 6
def red_marbles : ℕ := 5
def blue_marbles : ℕ := 4
def green_marbles : ℕ := 6
def total_after_one_red : ℕ := total_marbles - 1

-- Define the desired probability
def probability_red_then_blue : ℚ := (red_marbles / total_marbles) * (blue_marbles / total_after_one_red)

-- Assert the target probability
theorem red_then_blue_probability :
  probability_red_then_blue = 2 / 21 :=
begin
  sorry
end

end red_then_blue_probability_l575_575705


namespace total_kayaks_built_by_end_of_May_l575_575324

theorem total_kayaks_built_by_end_of_May : ∀ a r n,
  a = 9 → r = 3 → n = 5 → ∑ i in finset.range n, nat.rec_on i a (λ n acc, acc * r) = 1089 :=
by
  intros a r n ha hr hn
  rw [ha, hr, hn]
  calc ∑ i in finset.range n, nat.rec_on i a (λ n acc, acc * r)
      = ∑ i in finset.range n, 9 * (3 ^ i) : by sorry -- (The proof steps would go here, omitted as per instructions)
      ... = 1089 : by sorry

end total_kayaks_built_by_end_of_May_l575_575324


namespace baba_yagas_savings_plan_l575_575600

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l575_575600


namespace number_of_axes_of_symmetry_isosceles_l575_575318

variable (T : Type) [IsoscelesTriangle T]

theorem number_of_axes_of_symmetry_isosceles :
  ∃ n, (n = 1 ∨ n = 3) :=
sorry

end number_of_axes_of_symmetry_isosceles_l575_575318


namespace polyhedron_volume_is_correct_l575_575228

noncomputable def volume_of_polyhedron : ℕ :=
  let side_length := 12
  let num_squares := 3
  let square_area := side_length * side_length
  let cube_volume := side_length ^ 3
  let polyhedron_volume := cube_volume / 2
  polyhedron_volume

theorem polyhedron_volume_is_correct :
  volume_of_polyhedron = 864 :=
by
  sorry

end polyhedron_volume_is_correct_l575_575228


namespace solve_for_x_l575_575119

def g (t : ℝ) : ℝ := 2 * t / (1 - t)

theorem solve_for_x (x y : ℝ) (h : y = g x) (h1 : x ≠ 1) (h2 : y ≠ 1) : x = -g (-y) :=
by 
-- Proof of the theorem (omitted)
sorry

end solve_for_x_l575_575119


namespace inequalities_hold_l575_575503

variables (a b : Real)
variables (h : 0 < a ∧ a < b)

theorem inequalities_hold : 
  a^3 < b^3 ∧ sqrt b - sqrt a < sqrt (b - a) := by
  sorry

end inequalities_hold_l575_575503


namespace problem_statements_l575_575174

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin x * cos x - cos x ^ 2 + (1 / 2)

theorem problem_statements : 
  (f (x + π/3) = cos(2 * x) ∧ is_zero_count f 20 (0, 10 * π))
  ∧ is_even_function (fun x => f (x + π / 3)) :=
by
  sorry

end problem_statements_l575_575174


namespace no_such_polynomials_exists_l575_575320

theorem no_such_polynomials_exists :
  ¬ ∃ (f g : Polynomial ℚ), (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) := 
by 
  sorry

end no_such_polynomials_exists_l575_575320


namespace binom_eight_five_l575_575360

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l575_575360


namespace combination_8_5_l575_575354

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575354


namespace basketball_students_count_l575_575083

def students_play_basketball (C : ℕ) (B_inter_C : ℕ) (B_union_C : ℕ) : ℕ :=
  B_union_C - (C - B_inter_C)

theorem basketball_students_count (C B_inter_C B_union_C : ℕ) (hC : C = 8) (hB_inter_C : B_inter_C = 4) (hB_union_C : B_union_C = 14) : students_play_basketball C B_inter_C B_union_C = 10 :=
by
  rw [hC, hB_inter_C, hB_union_C]
  unfold students_play_basketball
  simp
  sorry

end basketball_students_count_l575_575083


namespace binom_12_6_l575_575798

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575798


namespace sin_angle_CAB_l575_575653

def parabola (x : ℝ) : ℝ := -x^2 + 2*x - 3

def shifted_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def point_A : ℝ × ℝ := (-3, 0)
def point_B : ℝ × ℝ := (1, 0)
def vertex_C : ℝ × ℝ := (-1, 4)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def AC_length := distance point_A vertex_C

theorem sin_angle_CAB : ∃ θ : ℝ, sin θ = 2 * real.sqrt 5 / 5 :=
by
  let AC := distance (-3, 0) (-1, 4)
  let sin_θ := 4 / AC
  have h : AC = 2 * real.sqrt 5 := by sorry
  use acos (4 / (2 * real.sqrt 5))
  rw [sin_of_real, h]
  norm_num

end sin_angle_CAB_l575_575653


namespace production_bottles_l575_575162

-- Definitions from the problem conditions
def machines_production_rate (machines : ℕ) (rate : ℕ) : ℕ := rate / machines
def total_production (machines rate minutes : ℕ) : ℕ := machines * rate * minutes

-- Theorem to prove the solution
theorem production_bottles :
  machines_production_rate 6 300 = 50 →
  total_production 10 50 4 = 2000 :=
by
  intro h
  have : 10 * 50 * 4 = 2000 := by norm_num
  exact this

end production_bottles_l575_575162


namespace range_of_a_l575_575041

noncomputable def f (a x : ℝ) := Real.log (x^2 - 2*a*x) / Real.log a

theorem range_of_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Icc (3:ℝ) (4:ℝ) → x₂ ∈ Icc (3:ℝ) (4:ℝ) → x₁ ≠ x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ (1 < a ∧ a < 3 / 2) := 
sorry

end range_of_a_l575_575041


namespace minimum_tickets_needed_l575_575263

noncomputable def min_tickets {α : Type*} (winning_permutation : Fin 50 → α) (tickets : List (Fin 50 → α)) : ℕ :=
  List.length tickets

theorem minimum_tickets_needed
  (winning_permutation : Fin 50 → ℕ)
  (tickets : List (Fin 50 → ℕ))
  (h_tickets_valid : ∀ t ∈ tickets, Function.Surjective t)
  (h_at_least_one_match : ∀ winning_permutation : Fin 50 → ℕ,
      ∃ t ∈ tickets, ∃ i : Fin 50, t i = winning_permutation i) : 
  min_tickets winning_permutation tickets ≥ 26 :=
sorry

end minimum_tickets_needed_l575_575263


namespace correct_option_is_D_l575_575526

noncomputable def data : List ℕ := [7, 5, 3, 5, 10]

theorem correct_option_is_D :
  let mean := (7 + 5 + 3 + 5 + 10) / 5
  let variance := (1 / 5 : ℚ) * ((7 - mean) ^ 2 + (5 - mean) ^ 2 + (5 - mean) ^ 2 + (3 - mean) ^ 2 + (10 - mean) ^ 2)
  let mode := 5
  let median := 5
  mean = 6 ∧ variance ≠ 3.6 ∧ mode ≠ 10 ∧ median ≠ 3 :=
by
  sorry

end correct_option_is_D_l575_575526


namespace number_of_three_digit_multiples_of_6_l575_575070

theorem number_of_three_digit_multiples_of_6 : 
  let lower_bound := 100
  let upper_bound := 999
  let multiple := 6
  let smallest_n := Nat.ceil (100 / multiple)
  let largest_n := Nat.floor (999 / multiple)
  let count_multiples := largest_n - smallest_n + 1
  count_multiples = 150 := by
  sorry

end number_of_three_digit_multiples_of_6_l575_575070


namespace identical_numbers_in_geometric_progression_set_l575_575681

theorem identical_numbers_in_geometric_progression_set (n : ℕ) (a : ℕ → ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 4 * n → 0 < a i) 
  (h2 : ∀ (i j k l : ℕ), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → 
    (a j) / (a i) = (a k) / (a j) ∧ (a k) / (a j) = (a l) / (a k)) : 
  ∃ m, (∀ i, 1 ≤ i ∧ i ≤ n → a (4 * m) = a (4 * i)) :=
begin
  sorry
end

end identical_numbers_in_geometric_progression_set_l575_575681


namespace ratio_boys_to_girls_l575_575522

theorem ratio_boys_to_girls (total_students girls : ℕ) (h_total : total_students = 520) (h_girls : girls = 200) :
  let boys := total_students - girls in
  let gcd := Nat.gcd boys girls in
  (boys / gcd) = 8 ∧ (girls / gcd) = 5 :=
by
  sorry

end ratio_boys_to_girls_l575_575522


namespace find_two_numbers_l575_575238

theorem find_two_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 5) (harmonic_mean : 2 * a * b / (a + b) = 5 / 3) :
  (a = (15 + Real.sqrt 145) / 4 ∧ b = (15 - Real.sqrt 145) / 4) ∨
  (a = (15 - Real.sqrt 145) / 4 ∧ b = (15 + Real.sqrt 145) / 4) :=
by
  sorry

end find_two_numbers_l575_575238


namespace time_to_pass_platform_l575_575301

def lengthOfTrain : ℝ := 480
def lengthOfPlatform : ℝ := 250
def speedKmHr : ℝ := 60
def speedMs : ℝ := speedKmHr * (1000 / 3600) -- converting km/hr to m/s

def totalDistance : ℝ := lengthOfTrain + lengthOfPlatform
def expectedTime : ℝ := totalDistance / speedMs

theorem time_to_pass_platform :
  expectedTime ≈ 43.8 := 
by 
  sorry

end time_to_pass_platform_l575_575301


namespace greatest_int_less_than_neg_17_div_3_l575_575703

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end greatest_int_less_than_neg_17_div_3_l575_575703


namespace angle_between_AB_CD_l575_575867

def point := (ℝ × ℝ × ℝ)

def A : point := (-3, 0, 1)
def B : point := (2, 1, -1)
def C : point := (-2, 2, 0)
def D : point := (1, 3, 2)

noncomputable def angle_between_lines (p1 p2 p3 p4 : point) : ℝ := sorry

theorem angle_between_AB_CD :
  angle_between_lines A B C D = Real.arccos (2 * Real.sqrt 105 / 35) :=
sorry

end angle_between_AB_CD_l575_575867


namespace trisected_triangle_side_l575_575543

theorem trisected_triangle_side
  (A B C F G : Point)
  (h_triangle: is_triangle A B C)
  (h_segment1 : is_segment A B)
  (h_segment2 : is_segment B C)
  (h_segment3 : is_segment C A)
  (h_f_g_on_ab : is_on_line_segment F A B)
  (h_f_closer_a : closer_to F A G B)
  (h_angle_trisect : trisects_angle F G C)
  (h_AF_FB_CG_CF : ∀ (AF FB CG CF: ℝ), (AF / FB = CG / CF)):
  AF / AB = 1 / 3 ∧ FG / AB = 1 / 3 ∧ GB / AB = 1 / 3 := sorry

end trisected_triangle_side_l575_575543


namespace circumradius_inradius_inequality_l575_575307

-- Definitions related to the triangle and requested conditions.
variables {ABC : Type} [triangle ABC]
variables {R r a h : ℝ} -- Real-valued variables for circumradius, inradius, longest side, shortest altitude.
variables (h_cond1 : a ≤ 2 * R) (h_cond2 : h > 2 * r) -- Conditions from the problem.

-- The theorem statement
theorem circumradius_inradius_inequality (h_cond1 : a ≤ 2 * R) (h_cond2 : h > 2 * r) :
  R / r > a / h := 
  sorry

end circumradius_inradius_inequality_l575_575307


namespace alice_probability_at_A_or_C_after_35_moves_l575_575311

-- Define the regular hexagon and its vertices
inductive Vertex
| A | B | C | D | E | F
deriving DecidableEq

-- Define the movement rules
def move (v : Vertex) (direction : Bool) : Vertex :=
  match v, direction with
  | Vertex.A, true  => Vertex.B
  | Vertex.A, false => Vertex.F
  | Vertex.B, true  => Vertex.C
  | Vertex.B, false => Vertex.A
  | Vertex.C, true  => Vertex.D
  | Vertex.C, false => Vertex.B
  | Vertex.D, true  => Vertex.E
  | Vertex.D, false => Vertex.C
  | Vertex.E, true  => Vertex.F
  | Vertex.E, false => Vertex.D
  | Vertex.F, true  => Vertex.A
  | Vertex.F, false => Vertex.E

-- Define the problem statement
theorem alice_probability_at_A_or_C_after_35_moves : 
  let moves := 35 in let start := Vertex.A in
  -- Probability that she is at vertex A or C after 35 moves is zero
  (nat.choose moves (moves / 2)) = 0 :=
by 
-- The required proof goes here
sorry

end alice_probability_at_A_or_C_after_35_moves_l575_575311


namespace father_cannot_see_boy_more_than_half_time_l575_575150

def speed_boy := 10 -- speed in km/h
def speed_father := 5 -- speed in km/h

def cannot_see_boy_more_than_half_time (school_perimeter : ℝ) : Prop :=
  ¬(∃ T : ℝ, T > school_perimeter / (2 * speed_boy) ∧ T < school_perimeter / speed_boy)

theorem father_cannot_see_boy_more_than_half_time (school_perimeter : ℝ) (h_school_perimeter : school_perimeter > 0) :
  cannot_see_boy_more_than_half_time school_perimeter :=
by
  sorry

end father_cannot_see_boy_more_than_half_time_l575_575150


namespace jason_safe_combinations_l575_575989

theorem jason_safe_combinations :
  let digits := {0, 1, 2, 3, 4}
  let is_even n := n % 2 = 0
  let is_odd n := n % 2 = 1
  (∀ (combo : Vector ℕ 7),
    (∀ i, combo[i] ∈ digits) →
    (∀ i < 6, is_even (combo[i]) ↔ is_odd (combo[i+1])) →
    (∀ i < 6, is_odd (combo[i]) ↔ is_even (combo[i+1])) →
    cardinality { combo | (∀ i, combo[i] ∈ digits) ∧
                           (∀ i < 6, is_even (combo[i]) → is_odd (combo[i+1])) ∧
                           (∀ i < 6, is_odd (combo[i]) → is_even (combo[i+1])) 
              } = 1080) :=
sorry

end jason_safe_combinations_l575_575989


namespace combination_8_5_l575_575352

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575352


namespace number_of_hens_l575_575713

theorem number_of_hens (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 136) : H = 24 :=
by
  sorry

end number_of_hens_l575_575713


namespace ratio_of_second_dog_food_l575_575943

theorem ratio_of_second_dog_food (x : ℝ) (h1 : 3 * (1.5 + x + (x + 2.5)) = 30) : x / 1.5 = 2 :=
by 
  have h2: 1.5 + x + (x + 2.5) = 10 := 
    by linarith
  have h3: 1.5 + 2 * x + 2.5 = 10 := 
    by linarith
  have h4: 2 * x + 4 = 10 := 
    by linarith
  have x_eq: x = 3 := 
    by linarith
  have ratio: 3 / 1.5 = 2 := 
    by norm_num
  exact ratio

end ratio_of_second_dog_food_l575_575943


namespace savings_plan_l575_575596

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l575_575596


namespace cos_squared_minus_sin_squared_15_eq_sqrt3_div2_cos_30_eq_sqrt3_div2_cos_squared_minus_sin_squared_15_express_l575_575775

open Real

theorem cos_squared_minus_sin_squared_15_eq_sqrt3_div2 :
  cos (15 * π / 180) ^ 2 - sin (15 * π / 180) ^ 2 = cos (30 * π / 180) :=
by sorry

theorem cos_30_eq_sqrt3_div2 :
  cos (30 * π / 180) = sqrt 3 / 2 :=
by sorry

theorem cos_squared_minus_sin_squared_15_express : 
  cos (15 * π / 180) ^ 2 - sin (15 * π / 180) ^ 2 = sqrt 3 / 2 :=
by
  calc
    cos (15 * π / 180) ^ 2 - sin (15 * π / 180) ^ 2 = cos (30 * π / 180) : by exact cos_squared_minus_sin_squared_15_eq_sqrt3_div2
    ... = sqrt 3 / 2 : by exact cos_30_eq_sqrt3_div2.sorry

end cos_squared_minus_sin_squared_15_eq_sqrt3_div2_cos_30_eq_sqrt3_div2_cos_squared_minus_sin_squared_15_express_l575_575775


namespace fixed_point_l575_575586

theorem fixed_point (m : ℝ) : (2 * m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by {
  sorry
}

end fixed_point_l575_575586


namespace odd_and_monotonically_increasing_on_lg_2x_l575_575756

def odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

def monotonically_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

open Set

theorem odd_and_monotonically_increasing_on_lg_2x :
  (∀ x ∈ (Ioi 0 : Set ℝ), 
    odd (λ x : ℝ, real.log (2 ^ x))
    ∧ monotonically_increasing (λ x : ℝ, real.log (2 ^ x)) (Ioi 0)) :=
by
  sorry

end odd_and_monotonically_increasing_on_lg_2x_l575_575756


namespace range_of_a_l575_575968

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) ↔ -8 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l575_575968


namespace binomial_12_6_eq_924_l575_575838

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575838


namespace num_triangles_l575_575453

theorem num_triangles (n : ℕ) (h_n : n = 15) : 
  let vertices := 3
  let total_points := vertices + n
  let initial_triangles := 3
  let additional_triangles := 2 * (n - 1)
  total_points = 18 ∧ (∃ (points : Finset.Point), (∀ three_points : Finset.Point (points : Finset), ¬Collinear three_points)) →
  initial_triangles + additional_triangles = 31 :=
by
  intros vertices_total non_collinear_points
  sorry

end num_triangles_l575_575453


namespace product_of_five_consecutive_integers_not_square_l575_575166

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (ha : 0 < a) : ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) := sorry

end product_of_five_consecutive_integers_not_square_l575_575166


namespace find_volume_of_pure_alcohol_l575_575575

variable (V1 Vf V2 : ℝ)
variable (P1 Pf : ℝ)

theorem find_volume_of_pure_alcohol
  (h : V2 = Vf * Pf / 100 - V1 * P1 / 100) : 
  V2 = Vf * (Pf / 100) - V1 * (P1 / 100) :=
by
  -- This is the theorem statement. The proof is omitted.
  sorry

end find_volume_of_pure_alcohol_l575_575575


namespace count_multiples_of_four_between_100_and_350_l575_575948

-- Define the problem conditions
def is_multiple_of_four (n : ℕ) : Prop := n % 4 = 0
def in_range (n : ℕ) : Prop := 100 < n ∧ n < 350

-- Problem statement
theorem count_multiples_of_four_between_100_and_350 : 
  ∃ (k : ℕ), k = 62 ∧ ∀ n : ℕ, is_multiple_of_four n ∧ in_range n ↔ (100 < n ∧ n < 350 ∧ is_multiple_of_four n)
:= sorry

end count_multiples_of_four_between_100_and_350_l575_575948


namespace divisibility_equiv_l575_575121

-- Definition of the functions a(n) and b(n)
def a (n : ℕ) := n^5 + 5^n
def b (n : ℕ) := n^5 * 5^n + 1

-- Define a positive integer
variables (n : ℕ) (hn : n > 0)

-- The theorem stating the equivalence
theorem divisibility_equiv : (a n) % 11 = 0 ↔ (b n) % 11 = 0 :=
sorry
 
end divisibility_equiv_l575_575121


namespace problem_correctness_l575_575012

noncomputable def ellipse_equation : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧ a^2 - b^2 = 1 ∧ (1:ℝ)^2 / a^2 + (3/2)^2 / b^2 = 1

noncomputable def circle_equation : Prop :=
  ∀ l : ℝ → ℝ, (∀ y, l y = y*sqrt 2 + 1) → 
  let F1 := (-1, 0) in
  let r := 2 / sqrt 3 in
  ∃ (x y : ℝ), (x + 1)^2 + y^2 = (r^2 : ℝ)

theorem problem_correctness : ellipse_equation ∧ circle_equation :=
  sorry

end problem_correctness_l575_575012


namespace evaporate_water_l575_575501

theorem evaporate_water :
  ∀ (M : ℝ) (initial_water_content final_water_content : ℝ),
  M = 500 → 
  initial_water_content = 0.85 → 
  final_water_content = 0.75 →
  ∃ (x : ℝ), 425 - x = final_water_content * (M - x) ∧
              x = 200 :=
by
  intros M initial_water_content final_water_content hM hInitial hFinal
  use 200
  rw [hM, hInitial] at *,
  split
  · calc
      425 - 200 = 225  : by norm_num
      _ = 0.75 * (500 - 200) : by norm_num
  · norm_num
  sorry

end evaporate_water_l575_575501


namespace binom_8_5_eq_56_l575_575367

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575367


namespace trapezoid_largest_area_l575_575686

theorem trapezoid_largest_area (a b c : ℝ) (ha : a = 2.12) (hb : b = 2.71) (hc : c = 3.53) :
  let area₁ := (a + c) * b,
      area₂ := (b + c) * a,
      area₃ := (a + b) * c in
  max (max area₁ area₂) area₃ = area₃ :=
by 
  -- We know lengths of segments
  have ha' : a = 2.12 := ha,
  have hb' : b = 2.71 := hb,
  have hc' : c = 3.53 := hc,
      
  -- Express areas of the trapezoids
  let area1 := (2.12 + 3.53) * 2.71,
  let area2 := (2.71 + 3.53) * 2.12,
  let area3 := (2.12 + 2.71) * 3.53,

  -- areas calculation from general to specific values for clarity
  have h1 : area₁ = (a + c) * b := rfl,
  have h2 : area2 = (b + c) * a := rfl,
  have h3 : area3 = (a + b) * c := rfl,

  -- Calculation of specific area1, area2, area3 values needed
  have harea1 : area1 = (2.12 + 3.53) * 2.71 := rfl,
  have harea2 : area2 = (2.71 + 3.53) * 2.12 := rfl,
  have harea3 : area3 = (2.12 + 2.71) * 3.53 := rfl,

  -- Check that this is equal to calculated value in the conclusion
  have harea1_val : area1 = 15.3115 := rfl,
  have harea2_val : area2 = 13.2288 := rfl,
  have harea3_val : area3 = 17.0599 := rfl,

  -- Use computations to conclude the final proof.
  show max (max 15.3115 13.2288) 17.0599 = 17.0599 from rfl,
  sorry

end trapezoid_largest_area_l575_575686


namespace binom_12_6_l575_575802

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575802


namespace symmetric_center_of_g_l575_575926

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x + Real.pi / 3)) / (Real.sin (2 * x + 2 * Real.pi / 3))

theorem symmetric_center_of_g :
  (∃ g : ℝ → ℝ, (∀ x y, g x = y ↔ f (Real.pi / 6 - x) = y) ∧ g (Real.pi / 4) = 0) :=
begin
  sorry
end

end symmetric_center_of_g_l575_575926


namespace sequence_2010_eq_4040099_l575_575139

def sequence_term (n : Nat) : Int :=
  if n % 2 = 0 then 
    (n^2 - 1 : Int) 
  else 
    -(n^2 - 1 : Int)

theorem sequence_2010_eq_4040099 : sequence_term 2010 = 4040099 := 
  by 
    sorry

end sequence_2010_eq_4040099_l575_575139


namespace angle_in_first_quadrant_of_complex_condition_l575_575510

theorem angle_in_first_quadrant_of_complex_condition (θ : ℝ) (h : ∃ z : ℂ, z = complex.cos θ - complex.sin θ * complex.I ∧ z.im < 0 ∧ z.re > 0) : 
  0 < θ ∧ θ < π / 2 :=
sorry

end angle_in_first_quadrant_of_complex_condition_l575_575510


namespace find_three_numbers_l575_575742

-- Define the conditions
def condition1 (X : ℝ) : Prop := X = 0.35 * X + 60
def condition2 (X Y : ℝ) : Prop := X = 0.7 * (1 / 2) * Y + (1 / 2) * Y
def condition3 (Y Z : ℝ) : Prop := Y = 2 * Z ^ 2

-- Define the final result that we need to prove
def final_result (X Y Z : ℝ) : Prop := X = 92 ∧ Y = 108 ∧ Z = 7

-- The main theorem statement
theorem find_three_numbers :
  ∃ (X Y Z : ℝ), condition1 X ∧ condition2 X Y ∧ condition3 Y Z ∧ final_result X Y Z :=
by
  sorry

end find_three_numbers_l575_575742


namespace binary_to_decimal_l575_575376

theorem binary_to_decimal (b : list ℕ) (h_x : b = [1, 0, 1, 1, 0, 1, 1]) : 
  list.sum (list.map (λ (p : ℕ × ℕ), p.1 * 2^p.2) (list.zip b ([0, 1, 2, 3, 4, 5, 6] : list ℕ))) = 91 := 
by 
  sorry

end binary_to_decimal_l575_575376


namespace construct_parallelogram_l575_575015

variables {P : Type*} [AffinePlane P]

open AffinePlane

-- Definitions from conditions in a)
variables (l1 l2 l3 l4 : Line P) (O : Point P)
-- Conditions in a)
variable (h1 : ¬ (l1 ∥ l2 ∨ l1 ∥ l3 ∨ l1 ∥ l4 ∨ l2 ∥ l3 ∨ l2 ∥ l4 ∨ l3 ∥ l4)) 
variable (h2 : ∀ i ∈ {l1, l2, l3, l4}, O ∉ i)

-- The theorem statement we need to prove
theorem construct_parallelogram :
  ∃ A B C D : Point P, 
    (A ∈ l1) ∧ 
    (B ∈ l3) ∧ 
    (C ∈ l2) ∧ 
    (D ∈ l4) ∧ 
    ∃ (p1 : Segment P) (p2 : Segment P), 
      is_diag O p1 ∧ 
      is_diag O p2 :=
sorry

end construct_parallelogram_l575_575015


namespace joshua_justin_ratio_l575_575992

theorem joshua_justin_ratio (total_share joshua_share justin_share : ℕ) 
  (h₁ : total_share = 40) 
  (h₂ : joshua_share = 30) 
  (h₃ : joshua_share + justin_share = total_share) 
  (h₄ : ∃ k : ℕ, joshua_share = k * justin_share) : 
  joshua_share / justin_share = 3 :=
by
  rw [h₂, h₁, add_comm] at h₃
  have h₅ : justin_share = 10 := by linarith
  rw [h₅, h₂]
  norm_num

end joshua_justin_ratio_l575_575992


namespace magnitude_of_z_is_correct_l575_575134

noncomputable def z := Complex.i - Complex.i / (1 - Complex.i)

theorem magnitude_of_z_is_correct : Complex.abs z = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_is_correct_l575_575134


namespace binom_8_5_eq_56_l575_575346

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575346


namespace time_to_cross_l575_575240

noncomputable def speed_of_faster_train : ℝ := 72 * (1000 / 3600) -- meters/second
noncomputable def speed_of_slower_train : ℝ := 36 * (1000 / 3600) -- meters/second
noncomputable def length_of_faster_train : ℝ := 150 -- meters

theorem time_to_cross :
  let relative_speed := speed_of_faster_train - speed_of_slower_train in
  let time_to_cross := length_of_faster_train / relative_speed in
  time_to_cross = 15 := 
by
  sorry

end time_to_cross_l575_575240


namespace percentage_of_female_employees_l575_575086

theorem percentage_of_female_employees 
  (E : ℕ) 
  (H_E : E = 1100)
  (pct_comp_literate : ℚ)
  (H_pct_cl : pct_comp_literate = 0.62)
  (female_comp_literate : ℕ)
  (H_female_cl : female_comp_literate = 462)
  (pct_male_comp_literate : ℚ)
  (H_pct_male_cl : pct_male_comp_literate = 0.5) :
  let total_cl := (pct_comp_literate * E : ℚ)
      CLM := (female_comp_literate + 0.5 * (E - 1100 + female_comp_literate) : ℚ)
      M := E - (female_comp_literate + 0.5 * (E - 1100 + female_comp_literate))
      F := E - M
      pct_female := (F / E : ℚ) * 100 in
  pct_female = 60 :=
by
  -- Leaving the proof as an exercise
  sorry

end percentage_of_female_employees_l575_575086


namespace gift_wrap_problem_l575_575286

theorem gift_wrap_problem :
  ∃ S P : ℕ,
  (S + P = 480) ∧
  (4 * S + 6 * P = 2340) ∧
  (P = 210) :=
by
  let P := 210
  let S := 480 - P
  have eq1 : S + P = 480 := by
    calc
      S + P
          = (480 - P) + P : by rw S
      ... = 480 : by rw Nat.sub_add_cancel (Nat.le_of_lt (Nat.lt_of_lt_of_le (Nat.zero_lt_of_ne_zero (Nat.ne_of_ne_zero_rfl)) (Nat.le_of_lt (Nat.lt_add_of_pos_right 210 Nat.zero_lt_of_ne_zero (Nat.ne_of_ne_zero_rfl))))))
  have eq2 : 4 * S + 6 * P = 2340 := by
    calc
      4 * S + 6 * P
          = 4 * (480 - P) + 6 * P : by rw S
      ... = 4 * 480 - 4 * P + 6 * P : by rw [← Nat.mul_sub_left_distrib, ← Nat.mul_sub_right_distrib]
      ... = 1920 - 4 * P + 6 * P : by rw Nat.mul_comm
      ... = 1920 + (6 * P - 4 * P) : by rw Nat.mul_sub_right_distrib
      ... = 1920 + 2 * P : by rw Nat.sub_add
      ... = 1920 + 420 : by rw P
      ... = 2340 : by rw [Nat.add_comm, Nat.add_assoc]
  exact ⟨S, P, eq1, eq2, rfl⟩

end gift_wrap_problem_l575_575286


namespace pyramid_volume_and_base_edge_l575_575186

theorem pyramid_volume_and_base_edge:
  ∀ (r: ℝ) (h: ℝ) (_: r = 5) (_: h = 10), 
  ∃ s V: ℝ,
    s = (10 * Real.sqrt 6) / 3 ∧ 
    V = (2000 / 9) :=
by
    sorry

end pyramid_volume_and_base_edge_l575_575186


namespace circle_and_tangent_lines_correct_l575_575732

-- Define the conditions
def radius : ℝ := 3
def pointA : ℝ × ℝ := (1, -1)
def lineC (x : ℝ) : ℝ := 2 * x
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the circle
def is_circle (x y h k r : ℝ) : Prop := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Define the equation for the circle center
def center_on_line (h k : ℝ) : Prop := k = lineC h

-- Define the tangent lines conditions
def is_tangent_line (x y : ℝ) (h k r : ℝ) (m b : ℝ) : Prop :=
  (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ∧ formal_distance (h k) (line m b)

-- The main theorem
theorem circle_and_tangent_lines_correct :
  ∃ h k, first_quadrant h k ∧ center_on_line h k ∧ is_circle 1 (-1) h k 3 ∧
  (tangent_line 4 3 h k 3 0 ∨ tangent_line 4 3 h k 3 (-4 / 3)) ∧
  (eq_tangent_line_x 4 h k ∨ eq_tangent_line_slope h k (-4 / 3))
:= sorry

end circle_and_tangent_lines_correct_l575_575732


namespace min_value_squared_sum_l575_575465

noncomputable def f (a b x : ℝ) : ℝ :=
  2 * a * Real.sqrt x + b - Real.exp (x / 2)

theorem min_value_squared_sum 
  (a b x₀ : ℝ) 
  (h₀ : f a b x₀ = 0) 
  (h₁ : x₀ ∈ set.Icc (1 / 4) Real.exp 1) :
  (a^2 + b^2) = Real.exp (3 / 4) / 4 :=
sorry

end min_value_squared_sum_l575_575465


namespace number_of_subsets_of_A_l575_575652

noncomputable def set_A : Set ℝ := { x | x^2 - 1 = 0 }

theorem number_of_subsets_of_A : Fintype.card (Set.powerset set_A) = 4 := by
  sorry

end number_of_subsets_of_A_l575_575652


namespace trigonometric_proof_l575_575888

theorem trigonometric_proof 
  (θ : Real)
  (h : (2 * cos (3/2 * Real.pi + θ) + cos (Real.pi + θ)) / 
       (3 * sin (Real.pi - θ) + 2 * sin (5/2 * Real.pi + θ)) = 1/5) :
  (tan θ = 3/13)
  ∧
  (sin(θ)^2 + 3 * sin(θ) * cos(θ) = 20160 / 28561) :=
by
  sorry

end trigonometric_proof_l575_575888


namespace find_x_minus_y_l575_575016

theorem find_x_minus_y (x y : ℝ) (h : sqrt (x - 3) + (y + 2) ^ 2 = 0) : x - y = 5 :=
sorry

end find_x_minus_y_l575_575016


namespace discount_percentage_l575_575741

theorem discount_percentage (CP MP SP D : ℝ) (cp_value : CP = 100) 
(markup : MP = CP + 0.5 * CP) (profit : SP = CP + 0.35 * CP) 
(discount : D = MP - SP) : (D / MP) * 100 = 10 := 
by 
  sorry

end discount_percentage_l575_575741


namespace henry_total_payment_l575_575430

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l575_575430


namespace problem_a_problem_b_l575_575172

section ProblemA

variable (x : ℝ)

theorem problem_a :
  x ≠ 0 ∧ x ≠ -3/8 ∧ x ≠ 3/7 →
  2 + 5 / (4 * x) - 15 / (4 * x * (8 * x + 3)) = 2 * (7 * x + 1) / (7 * x - 3) →
  x = 9 := by
  sorry

end ProblemA

section ProblemB

variable (x : ℝ)

theorem problem_b :
  x ≠ 0 →
  2 / x + 1 / x^2 - (7 + 10 * x) / (x^2 * (x^2 + 7)) = 2 / (x + 3 / (x + 4 / x)) →
  x = 4 := by
  sorry

end ProblemB

end problem_a_problem_b_l575_575172


namespace lucy_jumps_2100_times_l575_575579

-- Define the total number of songs and the duration of each song in minutes
def numSongs : Nat := 10
def durationPerSong : Real := 3.5

-- Duration of the album in seconds
def durationInSeconds : Real := durationPerSong * 60 * numSongs

-- Lucy's jump rate in jumps per second
def jumpsPerSecond : Nat := 1

-- Total jumps
def totalJumps : Nat := durationInSeconds.to_nat * jumpsPerSecond

-- The theorem stating Lucy will jump rope 2100 times
theorem lucy_jumps_2100_times :
  totalJumps = 2100 := 
by
  sorry

end lucy_jumps_2100_times_l575_575579


namespace sum_sheets_at_least_four_pow_m_l575_575243

theorem sum_sheets_at_least_four_pow_m (m : ℕ) :
  let step := m * 2^(m-1)
  ∑ (i in finset.range (2^m), 1) = 2^m →
  ∑ (i in finset.range step, (λ i:ℕ, 1 + 1)) = 2^m →
  (∀ i j, i ≠ j → ∀ (a₁ a₂ : ℕ), a₁ = 1 → a₂ = 1 → 
    let t := a₁ + a₂ in ∑ (i in finset.range (2^m), t) ≥ 4^m) :=
begin
  sorry
end

end sum_sheets_at_least_four_pow_m_l575_575243


namespace line_through_focus_of_ellipse_l575_575034

theorem line_through_focus_of_ellipse
  (a b c : ℝ)
  (h_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → true)
  (h_focus : ∀ (x y : ℝ) (l : ℝ), y = l * (x + c) → x = -c ∧ y = 0)
  (h_intersect : ∀ (x y : ℝ), (x, y) ∈ ({p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) →
    ∃ A B : ℝ × ℝ, A ≠ B ∧
    (y - (0 : ℝ)) / (x - (-c)) = (y - l * (x + c)) / (x - (-c)) ∧
    l * (x + c) ≠ 0 ∧
    (Eq (y / x) x) ∧ (A, B) ∈ ({p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1})) :
  ∀ k : ℝ, k^2 = 1 → ∃ l : ℝ, y = l (x + c) := by
  sorry

end line_through_focus_of_ellipse_l575_575034


namespace find_p_l575_575213

theorem find_p (m n p : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < p) 
  (h : 3 * m + 3 / (n + 1 / p) = 17) : p = 2 := 
sorry

end find_p_l575_575213


namespace area_quadrilateral_extension_l575_575146

variable (A B C D B1 C1 D1 A1 : Type) 
          [AddCommGroup A] [TopologicalSpace A]
          [AddCommGroup B] [TopologicalSpace B]
          [AddCommGroup C] [TopologicalSpace C]
          [AddCommGroup D] [TopologicalSpace D]
          [AddCommGroup B1] [TopologicalSpace B1]
          [AddCommGroup C1] [TopologicalSpace C1]
          [AddCommGroup D1] [TopologicalSpace D1]
          [AddCommGroup A1] [TopologicalSpace A1]

-- Given conditions
def convex_quadrilateral (ABCD : Set A) : Prop := sorry
def point_extension (P Q PQ1 : Set A) (cond : P.length = Q.length) : Prop := sorry

-- Hypotheses
axiom h1 : convex_quadrilateral {A, B, C, D}
axiom h2 : point_extension A B B1
axiom h3 : point_extension B C C1
axiom h4 : point_extension C D D1
axiom h5 : point_extension D A A1

-- Lemma
theorem area_quadrilateral_extension :
  ∀ (S ABCD : ℝ), S ABCD > 0 → S (A1 B1 C1 D1) = 5 * S ABCD := 
sorry

end area_quadrilateral_extension_l575_575146


namespace stratified_sampling_l575_575285

theorem stratified_sampling (S F So J n : ℕ) (hS : S = 900) (hF : F = 300) 
(hSo : So = 200) (hJ : J = 400) (hn : n = 45) : 
  let sampling_fraction := n / S in
  let sampled_freshmen := sampling_fraction * F in
  let sampled_sophomores := sampling_fraction * So in
  let sampled_juniors := sampling_fraction * J in
  sampled_freshmen = 15 ∧ sampled_sophomores = 10 ∧ sampled_juniors = 20 :=
by
  let sampling_fraction := n / S
  let sampled_freshmen := sampling_fraction * F
  let sampled_sophomores := sampling_fraction * So
  let sampled_juniors := sampling_fraction * J
  have := sampling_fraction = 1 / 20
  have h_sampling_fraction: sampling_fraction = 1 / 20 := sorry
  have h_sampled_freshmen: sampled_freshmen = 15 := sorry
  have h_sampled_sophomores: sampled_sophomores = 10 := sorry
  have h_sampled_juniors: sampled_juniors = 20 := sorry
  exact ⟨h_sampled_freshmen, h_sampled_sophomores, h_sampled_juniors⟩
  sorry

end stratified_sampling_l575_575285


namespace sequence_arithmetic_progression_l575_575861

theorem sequence_arithmetic_progression (n : ℕ) (h1 : n ≥ 4) :
  (∃ (x : Fin n → ℝ),
    Function.Injective x ∧
    (∀ i : Fin n, let a := x i; let b := x (i + 1) % n; let c := x (i + 2) % n in
      a + c = 2 * b)) ↔ ∃ (k : ℕ), k ≥ 3 ∧ n = 3 * k :=
by
  sorry

end sequence_arithmetic_progression_l575_575861


namespace find_c_l575_575459

theorem find_c (a b c : ℝ) (h_line : 4 * a - 3 * b + c = 0) 
  (h_min : (a - 1)^2 + (b - 1)^2 = 4) : c = 9 ∨ c = -11 := 
    sorry

end find_c_l575_575459


namespace algebraic_identity_l575_575505

theorem algebraic_identity (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) :
    a^2 - b^2 = -8 := by
  sorry

end algebraic_identity_l575_575505


namespace num_integers_satisfying_inequalities_l575_575866

theorem num_integers_satisfying_inequalities : 
  let conditions_1 (x : ℤ) := -5 * x ≥ 2 * x + 10
  ∧ let conditions_2 (x : ℤ) := -x ≤ 14
  ∧ let conditions_3 (x : ℤ) := -3 * x ≥ x + 8
  ∃ x, conditions_1 x ∧ conditions_2 x ∧ conditions_3 x →
  ∃ (S : Finset ℤ), (∀ x ∈ S, conditions_1 x ∧ conditions_2 x ∧ conditions_3 x) 
                    ∧ S.card = 13 := 
by 
  sorry

end num_integers_satisfying_inequalities_l575_575866


namespace participant_guesses_needed_l575_575183

theorem participant_guesses_needed (n k : ℕ) (hk : n > k) : (if k = n / 2 then 2 else 1) = 
  (if k = n / 2 then 2 else 1) := 
begin
  sorry
end

end participant_guesses_needed_l575_575183


namespace FN_eq_DE_l575_575765

theorem FN_eq_DE {O A B C D E F M N : Point} {DEFM_square : Square DE FM}
    (h1 : C ∈ circle O) 
    (h2 : ∀ (D : Point), D ∈ line_through A B ∧ perpendicular CD D) 
    (h3 : E ∈ segment BD) 
    (h4 : AE = AC) 
    (h5 : intersect_extension AM (circle O) = N) 
: FN = DE := 
sorry

end FN_eq_DE_l575_575765


namespace total_donation_l575_575769

-- Define the number of stuffed animals each person has
def barbara_stuffed_animals := 9
def trish_stuffed_animals := 2 * barbara_stuffed_animals
def sam_stuffed_animals := barbara_stuffed_animals + 5

-- Define the selling prices for each person
def barbara_price := 2
def trish_price := 1.5
def sam_price := 2.5

-- Define the money each person makes
def barbara_money := barbara_stuffed_animals * barbara_price
def trish_money := trish_stuffed_animals * trish_price
def sam_money := sam_stuffed_animals * sam_price

-- The total money donated
def total_donated := barbara_money + trish_money + sam_money

-- Prove the total money donated is $80
theorem total_donation : total_donated = 80 := by
  sorry

end total_donation_l575_575769


namespace problem_1_problem_2_l575_575113

noncomputable def a_n : ℕ → ℝ := sorry

def b_n (a_n : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (1 - (a_n (k-1)) / (a_n k)) * (1 / Real.sqrt (a_n k))

theorem problem_1 (a_n : ℕ → ℝ) (h1 : a_n 0 = 1)
  (h2 : ∀ n, a_n n ≤ a_n (n + 1)) :
  ∀ n, 0 ≤ b_n a_n n ∧ b_n a_n n < 2 :=
sorry

theorem problem_2 :
  ∀ c : ℝ, 0 ≤ c ∧ c < 2 →
  ∃ (a_n : ℕ → ℝ), (∀ n, a_n 0 = 1 ∧ (∀ n, a_n n ≤ a_n (n + 1))) ∧ 
  ∃ᶠ n in Filter.atTop, b_n a_n n > c :=
sorry

end problem_1_problem_2_l575_575113


namespace xy_yz_zx_value_l575_575135

namespace MathProof

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 147) 
  (h2 : y^2 + y * z + z^2 = 16) 
  (h3 : z^2 + x * z + x^2 = 163) :
  x * y + y * z + z * x = 56 := 
sorry      

end MathProof

end xy_yz_zx_value_l575_575135


namespace LucyJumps_l575_575578

theorem LucyJumps (jump_per_second : ℕ) (seconds_per_minute : ℕ) (songs : ℕ) (length_per_song_minutes : ℕ → ℚ) (total_jumps : ℕ) :
  jump_per_second = 1 →
  seconds_per_minute = 60 →
  songs = 10 →
  (∀ (n : ℕ), length_per_song_minutes n = 3.5) →
  total_jumps = songs * (length_per_song_minutes 1 * seconds_per_minute * jump_per_second).toNat →
  total_jumps = 2100 :=
by
  intros h_jump_per_second h_seconds_per_minute h_songs h_length_per_song h_total_jumps
  rw [h_jump_per_second, h_seconds_per_minute, h_songs, h_length_per_song (1 : ℕ)] at h_total_jumps
  exact h_total_jumps

end LucyJumps_l575_575578


namespace number_of_students_l575_575223

theorem number_of_students (total_pizzas : ℕ) (slices_per_pizza : ℕ) (cheese_leftover : ℕ) (onion_leftover : ℕ) (cheese_per_student : ℕ) (onion_per_student : ℕ) : ℕ :=
  let total_slices := total_pizzas * slices_per_pizza
  let used_cheese_slices := total_slices - cheese_leftover
  let used_onion_slices := total_slices - onion_leftover
  let total_used_cheese_slices := cheese_per_student * used_cheese_slices
  let total_used_onion_slices := onion_per_student * used_onion_slices
  used_cheese_slices + used_onion_slices = total_used_cheese_slices + total_used_onion_slices →
  used_cheese_slices + used_onion_slices = 204 →
  let students := 204 / 3 in students = 68

end number_of_students_l575_575223


namespace triangle_side_c_value_l575_575970

-- Define the conditions and prove the conclusion.
theorem triangle_side_c_value
  (a b c : ℝ)
  (h1 : a^2 - b^2 = c)
  (h2 : ∃ (A B C : ℝ), sin A * cos B = 2 * cos A * sin B):
  c = 3 :=
sorry

end triangle_side_c_value_l575_575970


namespace binom_8_5_eq_56_l575_575364

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l575_575364


namespace fraction_simplification_l575_575708

theorem fraction_simplification :
  (3100 - 3037)^2 / 81 = 49 := by
  sorry

end fraction_simplification_l575_575708


namespace seated_ways_alice_between_bob_and_carol_l575_575529

-- Define the necessary entities and conditions for the problem.
def num_people : Nat := 7
def alice := "Alice"
def bob := "Bob"
def carol := "Carol"

-- The main theorem
theorem seated_ways_alice_between_bob_and_carol :
  ∃ (ways : Nat), ways = 48 := by
  sorry

end seated_ways_alice_between_bob_and_carol_l575_575529


namespace chess_tournament_participants_l575_575977

-- Define the number of grandmasters
variables (x : ℕ)

-- Define the number of masters as three times the number of grandmasters
def num_masters : ℕ := 3 * x

-- Condition on total points scored: Master's points is 1.2 times the Grandmaster's points
def points_condition (g m : ℕ) : Prop := m = 12 * g / 10

-- Proposition that the total number of participants is 12
theorem chess_tournament_participants (x_nonnegative: 0 < x) (g m : ℕ)
  (masters_points: points_condition g m) : 
  4 * x = 12 := 
sorry

end chess_tournament_participants_l575_575977


namespace fixed_interval_range_t_l575_575472

theorem fixed_interval_range_t :
  ∀ (f F : ℝ → ℝ) (t : ℝ), 
  (∀ x, F x = f (-x)) →
  (∀ x, f x = |2^x - t|) →
  interval_fixed [1, 2] f F →
  (∀ x ∈ [1, 2], (2^x - t) * (2^(-x) - t) ≤ 0) →
  (t ∈ Icc (1/2:ℝ) 2) :=
by
  sorry

end fixed_interval_range_t_l575_575472


namespace intersection_point_unique_l575_575031

theorem intersection_point_unique {a : ℝ} :
  (∃ (x y : ℝ), x + y + 2 = 0 ∧ 2 * x - y + 1 = 0 ∧ a * x + y + 3 = 0) →
  Matrix.det ![
    ![a, 1],
    ![1, 1]
  ] = 1 :=
begin
  sorry,
end

end intersection_point_unique_l575_575031


namespace smallest_positive_period_f_max_min_values_f_l575_575480

open Real

/-- The function f(x) = cos^2(x) + cos^2(x - π/6) -/
def f (x : ℝ) : ℝ := cos x ^ 2 + cos (x - π / 6) ^ 2

/-- The smallest positive period of the function f is π -/
theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π :=
sorry

/-- The maximum and minimum values of f(x) in the interval [-π/3, π/4] -/
theorem max_min_values_f : 
  ∃ a b, a = -π / 3 ∧ b = π / 4 ∧ 
  (∀ x, x ∈ Icc a b → f x ≤ f b) ∧ 
  (∀ x, x ∈ Icc a b → f x ≥ f a) ∧ 
  f a = 1 / 4 ∧ 
  f b = sqrt 3 / 2 + 1 :=
sorry

end smallest_positive_period_f_max_min_values_f_l575_575480


namespace savings_plan_l575_575599

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l575_575599


namespace ratio_equivalence_l575_575073

theorem ratio_equivalence (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h : y / (x - z) = (x + 2 * y) / z ∧ (x + 2 * y) / z = x / (y + z)) :
  x / (y + z) = (2 * y - z) / (y + z) :=
by
  sorry

end ratio_equivalence_l575_575073


namespace frog_arrangement_count_l575_575683

-- Defining the total number of frogs and their colors
def total_frogs := 7
def green_frogs := 3
def red_frogs := 3
def blue_frog := 1

-- Defining the condition that green frogs refuse to sit next to red frogs
def frog_arrangement_valid (arrangement : List ℕ) : Bool :=
  let pairs := arrangement.zip arrangement.tail
  ∀ (p : ℕ × ℕ), p ∈ pairs → 
    (p.1 = 1 ∧ p.2 = 2) ∨ (p.1 = 2 ∧ p.2 = 1) → False

noncomputable def num_valid_arrangements : ℕ :=
  -- Placeholder for the actual calculation
  7 * 2 * nat.factorial 3 * nat.factorial 3

theorem frog_arrangement_count : num_valid_arrangements = 504 :=
by
  -- Calculation steps show the total is 504
  exact rfl

end frog_arrangement_count_l575_575683


namespace save_plan_l575_575591

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l575_575591


namespace area_quadrilateral_extension_l575_575145

variable (A B C D B1 C1 D1 A1 : Type) 
          [AddCommGroup A] [TopologicalSpace A]
          [AddCommGroup B] [TopologicalSpace B]
          [AddCommGroup C] [TopologicalSpace C]
          [AddCommGroup D] [TopologicalSpace D]
          [AddCommGroup B1] [TopologicalSpace B1]
          [AddCommGroup C1] [TopologicalSpace C1]
          [AddCommGroup D1] [TopologicalSpace D1]
          [AddCommGroup A1] [TopologicalSpace A1]

-- Given conditions
def convex_quadrilateral (ABCD : Set A) : Prop := sorry
def point_extension (P Q PQ1 : Set A) (cond : P.length = Q.length) : Prop := sorry

-- Hypotheses
axiom h1 : convex_quadrilateral {A, B, C, D}
axiom h2 : point_extension A B B1
axiom h3 : point_extension B C C1
axiom h4 : point_extension C D D1
axiom h5 : point_extension D A A1

-- Lemma
theorem area_quadrilateral_extension :
  ∀ (S ABCD : ℝ), S ABCD > 0 → S (A1 B1 C1 D1) = 5 * S ABCD := 
sorry

end area_quadrilateral_extension_l575_575145


namespace proposition_D_correct_l575_575315

theorem proposition_D_correct :
  ∀ x : ℝ, x^2 + x + 2 > 0 :=
by
  sorry

end proposition_D_correct_l575_575315


namespace product_with_a_equals_3_l575_575858

theorem product_with_a_equals_3 (a : ℤ) (h : a = 3) : 
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 :=
by
  sorry

end product_with_a_equals_3_l575_575858


namespace binomial_12_6_eq_924_l575_575836

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575836


namespace ratio_BD_BO_l575_575610

-- Define the geometric setup with given conditions
variables {A B C O D : Type}
variables [EuclideanGeometry O]
variables {R : ℝ} -- Radius of the circle
variables (A C O : Point)

def A_on_circle (O : Point) (A : Point) (R : ℝ) : Prop :=
dist O A = R

def C_on_circle (O : Point) (C : Point) (R : ℝ) : Prop :=
dist O C = R

def tangent_line_through_B (O A B : Point) : Prop :=
∃ L, Line_tangent (circle O R) L ∧ passes_through B L ∧ passes_through A L

def right_triangle_ABC (A B C : Point) : Prop :=
angle B A C = π / 2

def D_on_BO (B O D : Point) : Prop :=
lies_on D (line B O)

-- The theorem to be proved
theorem ratio_BD_BO (h1 : A_on_circle O A R)
  (h2 : C_on_circle O C R)
  (h3 : tangent_line_through_B O A B)
  (h4 : tangent_line_through_B O C B)
  (h5 : right_triangle_ABC A B C)
  (h6 : D_on_BO B O D) :
  dist B D / dist B O = 1 - real.sqrt 2 / 2 :=
begin
  sorry
end

end ratio_BD_BO_l575_575610


namespace rate_of_filling_during_fourth_hour_l575_575339

theorem rate_of_filling_during_fourth_hour :
  ∀ (x : ℝ),
    let h1 := 8
    let h2 := 10 * 2
    let h3 := x
    let h4 := -8
    let total := h1 + h2 + h3 + h4
    total = 34 → x = 14 :=
by
  assume x : ℝ
  have h1 : 8 := 8
  have h2 : 10 * 2 := 20
  have h3 : ℝ := x
  have h4 : -8 := -8
  let total := h1 + h2 + h3 + h4
  show total = 34 → x = 14
  sorry

end rate_of_filling_during_fourth_hour_l575_575339


namespace greatest_third_term_of_arithmetic_sequence_l575_575672

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l575_575672


namespace combination_8_5_l575_575356

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l575_575356


namespace problem1_problem2_problem3_l575_575925

variable {α : Type*} [LinearOrderedField α]
variables (a : α) (f : α → α)

-- The function definition
def f (x : α) : α := x^2 - 2 * a * x + 1

-- Problem 1
theorem problem1 (H : ∀ x : α, f(1 + x) = f(1 - x)) : a = 1 := sorry

-- Problem 2
theorem problem2 (H : ∀ x y : α, x ≤ y → f x ≤ f y) (H_mono : ∀ x : α, 1 ≤ x → f(x) ≤ f(x + 1)) : a ≤ 1 := sorry

-- Problem 3
theorem problem3 (H : ∀ x : α, -1 ≤ x ∧ x ≤ 1 → f x ≤ 2) : ∀ x : α, -1 ≤ x ∧ x ≤ 1 → f x ≤ 2 := sorry

end problem1_problem2_problem3_l575_575925


namespace part_a_part_b_part_c_l575_575159

-- Define the sequence a_n
def a (n : ℕ) : ℝ := 1 / ((n:ℝ) * (n + 1))

-- Summation from n to infinity of the sequence
def sum_from (n : ℕ) : ℝ := ∑' (k : ℕ) in (Set.Ici n), a k

-- First part: ∑_{n=1}^∞ a_n = 1
theorem part_a : sum_from 1 = 1 := 
sorry

-- Second part: ∑_{n=2}^∞ a_n = 1/2
theorem part_b : sum_from 2 = 1/2 :=
sorry

-- Third part: ∑_{n=3}^∞ a_n = 1/3
theorem part_c : sum_from 3 = 1/3 := 
sorry

end part_a_part_b_part_c_l575_575159


namespace james_winnings_l575_575103

variables (W : ℝ)

theorem james_winnings :
  (W / 2 - 2 = 55) → W = 114 :=
by
  intros h,
  sorry

end james_winnings_l575_575103


namespace no_win_with_finite_spells_l575_575236

def finite_spells (S : Set (ℝ × ℝ)) : Prop :=
  ∀ a b ∈ S, 0 < a ∧ a < b

theorem no_win_with_finite_spells
  (S : Set (ℝ × ℝ))
  (finite_S : finite S)
  (condition : finite_spells S) :
  ¬ (∃ (win_strategy : Π (n : ℕ), (ℕ × ℝ) × (ℕ × ℝ)),
    ∀ (played_moves_self played_moves_opp : ℕ → ℝ) (i : ℕ),
    played_moves_self 0 = 100 ∧ played_moves_opp 0 = 100 ∧
    (∀ j, j < i → (played_moves_self (j+1) = played_moves_self j - (win_strategy j).fst.snd) ∧
          (played_moves_opp   (j+1) = played_moves_opp j   - (win_strategy j).snd.snd)) ∧
    played_moves_self (i + 1) > 0 ∧ played_moves_opp (i + 1) ≤ 0) :=
sorry

end no_win_with_finite_spells_l575_575236


namespace ratio_x2_x1_local_max_value_l575_575444

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

-- Definitions for conditions
variables {x1 x0 x2 : ℝ} (a b c : ℝ)

-- Condition that the function has exactly two zeros and one local maximum
hypothesis (h1 : ∃ x1 x2, is_zero (f x1 a b c) ∧ is_zero (f x2 a b c) ∧ x1 ≠ x2
                    ∧ ∃ x0, ∃ x1 x2, x1 < x0 ∧ x0 < x2 ∧ is_local_max (f x0 a b c))

-- Condition that x₁, x₀, x₂ form a geometric sequence
hypothesis (h2 : ∃ r, x0 = x1 * r ∧ x2 = x0 * r)

-- Condition that the solution set of f(x) > f(x0) is (5, ∞)
hypothesis (h3 : ∀ x, x > 5 → f x a b c > f x0 a b c)

-- Lean 4 statement for the first part: Proving the ratio x2 / x1 is 4
theorem ratio_x2_x1 : x2 / x1 = 4 :=
sorry

-- Lean 4 statement for the second part: Proving the local maximum value is 4
theorem local_max_value : f x0 a b c = 4 :=
sorry

end ratio_x2_x1_local_max_value_l575_575444


namespace prove_n_value_prove_odd_sum_binomial_sum_l575_575475

-- Problem 1
theorem prove_n_value (n : ℕ) (h1 : nCk n 4 = 3 * nCk n 2 * (sqrt 2) ^ 2) : n = 11 := sorry

-- Problem 2
theorem prove_odd_sum (n : ℕ) (h1 : even n) :
  ((∑ i in (finset.range (n / 2 + 1)).map (λ i, 2 * i), nCk n (2 * i) * 2 ^ i) + 1) % 2 = 1 := sorry

-- Problem 3
theorem binomial_sum (n : ℕ) :
  ∑ k in finset.range n, k * nCk n k * 2^(k - 1) = n * 3^(n - 1) := sorry

end prove_n_value_prove_odd_sum_binomial_sum_l575_575475


namespace domain_of_g_l575_575406

noncomputable def g (x : ℝ) : ℝ := real.sqrt (4 - real.sqrt (6 - real.sqrt x))

theorem domain_of_g :
  {x : ℝ | 0 ≤ x ∧ x ≤ 36} = {x : ℝ | ∃ y : ℝ, y = g x} := by
  sorry

end domain_of_g_l575_575406


namespace total_surface_area_of_tower_l575_575695

theorem total_surface_area_of_tower :
  let s : ℕ → ℕ := λ n, match n with
    | 1 => 4
    | 2 => 5
    | 3 => 6
    | 4 => 7
    | 5 => 8
    | 6 => 9
    | 7 => 10
    | _ => 0
  in
  (6 * (s 1)^2) +
  (6 * (s 2)^2 - (s 2)^2) +
  (6 * (s 3)^2 - (s 3)^2) +
  (6 * (s 4)^2 - (s 4)^2) +
  (6 * (s 5)^2 - (s 5)^2) +
  (6 * (s 6)^2 - (s 6)^2) +
  (6 * (s 7)^2 - (s 7)^2) = 1871 :=
by
  sorry

end total_surface_area_of_tower_l575_575695


namespace total_stoppage_time_per_hour_l575_575685

variables (speed_ex_stoppages_1 speed_in_stoppages_1 : ℕ)
variables (speed_ex_stoppages_2 speed_in_stoppages_2 : ℕ)
variables (speed_ex_stoppages_3 speed_in_stoppages_3 : ℕ)

-- Definitions of the speeds given in the problem's conditions.
def speed_bus_1_ex_stoppages := 54
def speed_bus_1_in_stoppages := 36
def speed_bus_2_ex_stoppages := 60
def speed_bus_2_in_stoppages := 40
def speed_bus_3_ex_stoppages := 72
def speed_bus_3_in_stoppages := 48

-- The main theorem to be proved.
theorem total_stoppage_time_per_hour :
  ((1 - speed_bus_1_in_stoppages / speed_bus_1_ex_stoppages : ℚ)
   + (1 - speed_bus_2_in_stoppages / speed_bus_2_ex_stoppages : ℚ)
   + (1 - speed_bus_3_in_stoppages / speed_bus_3_ex_stoppages : ℚ)) = 1 := by
  sorry

end total_stoppage_time_per_hour_l575_575685


namespace binary_110011_to_decimal_l575_575397

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575397


namespace find_xy_l575_575208

theorem find_xy (x y : ℤ) 
  (h1 : (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3) : 
  x = -35 ∧ y = -35 :=
by 
  sorry

end find_xy_l575_575208


namespace map_distance_representation_l575_575140

theorem map_distance_representation
  (d_map : ℕ) (d_actual : ℕ) (conversion_factor : ℕ) (final_length_map : ℕ):
  d_map = 10 →
  d_actual = 80 →
  conversion_factor = d_actual / d_map →
  final_length_map = 18 →
  (final_length_map * conversion_factor) = 144 :=
by
  intros h1 h2 h3 h4
  sorry

end map_distance_representation_l575_575140


namespace find_a_when_tangent_l575_575206

theorem find_a_when_tangent:
  ∃ (a : ℝ), (∀ (x y : ℝ), y = ln (x + a) → y = x + 1) ↔ (a = 2) :=
begin
  sorry

end find_a_when_tangent_l575_575206


namespace max_sum_of_products_l575_575032

theorem max_sum_of_products (a b c d : ℕ) (h : a ∈ {5, 6, 7, 8} ∧ b ∈ {5, 6, 7, 8} ∧ c ∈ {5, 6, 7, 8} ∧ d ∈ {5, 6, 7, 8} ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  ab + bc + cd + da ≤ 169 := by
  sorry

end max_sum_of_products_l575_575032


namespace number_of_multiples_of_4_between_100_and_350_l575_575952

theorem number_of_multiples_of_4_between_100_and_350 :
  (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≥ 104 ∧ (set.filter (λ x, x % 4 = 0) (set.range 351)).count ≤ 348 →
  (set.filter (λ x, x % 4 = 0) (finset.Icc 100 350).to_set).card = 62 :=
by
  sorry

end number_of_multiples_of_4_between_100_and_350_l575_575952


namespace complex_number_solution_l575_575005

theorem complex_number_solution (z : ℂ) (h : (z - 1) / (z + 1) = complex.I) : z = complex.I :=
by 
  -- Adding the necessary definitions to setup the theorem
  sorry

end complex_number_solution_l575_575005


namespace monotonicity_minimum_value_harmonic_series_inequality_l575_575573

noncomputable def f (x a : ℝ) := 2 * Real.exp x - 2 * a * x + 3 * a

theorem monotonicity (a : ℝ) : 
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a)
∧ (a > 0 → ∀ x : ℝ, (x < Real.log a → ∀ y : ℝ, x < y → f x a > f y a) 
∧ (x > Real.log a → ∀ y : ℝ, x < y → f x a < f y a)) :=
sorry

theorem minimum_value (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, f x a ≥ 5 * a) ↔ (0 < a ∧ a ≤ 1) := 
sorry

theorem harmonic_series_inequality (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range (n+1), 1 / (k + 1)) > Real.log (n + 1) := 
sorry

end monotonicity_minimum_value_harmonic_series_inequality_l575_575573


namespace password_probability_l575_575771

def is_prime_single_digit : Fin 10 → Prop
| 2 | 3 | 5 | 7 => true
| _ => false

def is_vowel : Char → Prop
| 'A' | 'E' | 'I' | 'O' | 'U' => true
| _ => false

def is_positive_even_single_digit : Fin 9 → Prop
| 2 | 4 | 6 | 8 => true
| _ => false

def prime_probability : ℚ := 4 / 10
def vowel_probability : ℚ := 5 / 26
def even_pos_digit_probability : ℚ := 4 / 9

theorem password_probability :
  prime_probability * vowel_probability * even_pos_digit_probability = 8 / 117 := by
  sorry

end password_probability_l575_575771


namespace number_of_functions_l575_575424

theorem number_of_functions : 
  (∃ (f : ℝ → ℝ), (∃ (a b c d : ℝ), 
          (f = λ x, a * x^3 + b * x^2 + c * x + d) ∧ 
          (∀ (x : ℝ), (f (x^2) = f(x) * f(-x))) ) ∧ 
       ([ a, b, c, d ] ∈ { [0, 1], [0, -1], [-1, 0], [0, 0], [1, 0], [0, 1], [1, -1] })) :=
sorry

end number_of_functions_l575_575424


namespace segment_properties_l575_575704

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem segment_properties :
  distance (1, 2) (9, 14) = 16 ∧ midpoint (1, 2) (9, 14) = (5, 8) :=
by
  sorry

end segment_properties_l575_575704


namespace positional_relationship_c_b_l575_575074

variables {a b c : Type} [linear_structure a] [linear_structure b] [linear_structure c]

-- Assume a, b are skew lines
axiom skew_lines (a b : Type) [linear_structure a] [linear_structure b] : 
  ¬( ∃ p : intersect a b, p) ∧ ¬(a ∥ b)

-- Assume c is parallel to a
axiom parallel_lines (a c : Type) [linear_structure a] [linear_structure c] : a ∥ c

-- Statement to be proved
theorem positional_relationship_c_b (a b c : Type) [linear_structure a] [linear_structure b] [linear_structure c] :
  skew_lines a b → parallel_lines a c → (¬( ∃ p : intersect c b, p) ∨ ( ∃ p : intersect c b, p)) :=
by
  sorry

end positional_relationship_c_b_l575_575074


namespace bronze_needed_l575_575582

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l575_575582


namespace probability_red_or_white_ball_l575_575728

theorem probability_red_or_white_ball :
  let red_balls := 3
  let yellow_balls := 2
  let white_balls := 1
  let total_balls := red_balls + yellow_balls + white_balls
  let favorable_outcomes := red_balls + white_balls
  (favorable_outcomes / total_balls : ℚ) = 2 / 3 := by
  sorry

end probability_red_or_white_ball_l575_575728


namespace pasha_game_solvable_l575_575723

def pasha_game : Prop :=
∃ (a : Fin 2017 → ℕ), 
  (∀ i, a i > 0) ∧
  (∃ (moves : ℕ), moves = 43 ∧
   (∀ (box_contents : Fin 2017 → ℕ), 
    (∀ j, box_contents j = 0) →
    (∃ (equal_count : ℕ),
      (∀ j, box_contents j = equal_count)
      ∧
      (∀ m < 43,
        ∃ j, box_contents j ≠ equal_count))))

theorem pasha_game_solvable : pasha_game :=
by
  sorry

end pasha_game_solvable_l575_575723


namespace find_ab_l575_575562

variables {a b : ℝ}

theorem find_ab
  (h : ∀ x : ℝ, 0 ≤ x → 0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) :
  a * b = -1 :=
sorry

end find_ab_l575_575562


namespace Wendy_total_glasses_l575_575247

theorem Wendy_total_glasses (small large : ℕ)
  (h1 : small = 50)
  (h2 : large = small + 10) :
  small + large = 110 :=
by
  sorry

end Wendy_total_glasses_l575_575247


namespace Jack_and_Jill_same_speed_l575_575548

theorem Jack_and_Jill_same_speed (x : ℕ) (h : x = 18) :
  (x^2 - 13*x - 30) = (x^2 - 5*x - 84) / (x + 7) -> (x - 12) = 6 :=
by
  intro h_speed_equal
  have h1 : Jack_speed = x^2 - 13*x - 30, by sorry
  have h2 : Jill_speed = (x^2 - 5*x - 84) / (x + 7), by sorry
  have h3 : (x - 12) = Jill_speed, by sorry
  have h4 : Jack_speed = Jill_speed, from h,
  rw h4 at h3,
  rw h1 at h3,
  rw h2 at h3,
  refine eq.symm h3,
  sorry

end Jack_and_Jill_same_speed_l575_575548


namespace sum_of_roots_eq_five_l575_575026

-- Define the function f with the given properties
variable {f : ℝ → ℝ}

/-- Given the symmetry property of the function f -/
axiom f_symmetry : ∀ x : ℝ, f(x) = f(2 - x)

/-- Given that the function f has 5 distinct roots -/
axiom f_has_five_distinct_roots : ∃ a b c d e : ℝ, (∀ (x : ℝ), f(x) = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

-- The proof statement we aim to prove
theorem sum_of_roots_eq_five :
  ∃ a b c d e : ℝ, (∀ (x : ℝ), f(x) = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e)) ∧
  a + b + c + d + e = 5 :=
sorry

end sum_of_roots_eq_five_l575_575026


namespace smallest_n_for_a_2017_l575_575219

def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (2*n) := if n % 2 = 0 then a n else 2 * a n
| (2*n+1) := if n % 2 = 0 then 2 * a n + 1 else a n

theorem smallest_n_for_a_2017 : a 2017 = a 5 := 
sorry

end smallest_n_for_a_2017_l575_575219


namespace verify_statements_l575_575736

-- Definitions from the problem
def injective (f : α → β) : Prop := ∀ x1 x2, f x1 = f x2 → x1 = x2

-- Definition of statements to be assessed
def statement2 (f : α → β) (A : set α) : Prop :=
  injective f → ∀ x1 x2 ∈ A, x1 ≠ x2 → f x1 ≠ f x2

def statement3 (f : α → β) (B : set β) : Prop :=
  injective f → ∀ b ∈ B, ∃ at_most_one (λ x, f x = b)

-- Main proof problem: Verify which statements are true
theorem verify_statements (f : α → β) (A : set α) (B : set β) :
  (statement2 f A) ∧ (statement3 f B) :=
by
  sorry

end verify_statements_l575_575736


namespace minimum_pizzas_needed_l575_575990

variables (p : ℕ)

def income_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4
def maintenance_cost_per_pizza : ℕ := 1
def car_cost : ℕ := 6500

theorem minimum_pizzas_needed :
  p ≥ 929 ↔ (income_per_pizza * p - (gas_cost_per_pizza + maintenance_cost_per_pizza) * p) ≥ car_cost :=
sorry

end minimum_pizzas_needed_l575_575990


namespace range_of_a_l575_575889

noncomputable def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3 ∨ (-1 / 2 ≤ a ∧ a ≤ 2)) := 
  sorry

end range_of_a_l575_575889


namespace save_plan_l575_575588

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l575_575588


namespace point_Q_on_circle_l575_575456

def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = 2
def symmetric_about_line (x1 y1 x2 y2 : ℝ) : Prop := y1 + x1 = 0 ∧ y2 + x2 = 0

variables {a b c k m x y : ℝ}
variables {P Q M : ℝ × ℝ}

-- Given conditions
def conditions : Prop :=
  ellipse a b x y ∧ a > 0 ∧ b > 0 ∧ a > b ∧
  symmetric_about_line (-c) 0 0 b ∧
  P = (sqrt 6 / 2, 1 / 2) ∧ M = (1, 0) ∧
  (∃ l : ℝ → ℝ, ∀ x, l x = k * x + m ∧
   (line_ellipse_intersect E l).card = 1 ∧ k ≠ 0)

-- The proof goal
theorem point_Q_on_circle :
  conditions →
  (∃ Q : ℝ × ℝ, foot_of_perpendicular Q M l ∧ on_circle Q.1 Q.2) :=
sorry

end point_Q_on_circle_l575_575456


namespace solution_x_y_z_squared_sum_l575_575847

def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 3 * y, 2 * z],
    ![2 * x, y, -z],
    ![2 * x, -y, z]
  ]

def N_transpose_N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.mul (N x y z)ᵀ (N x y z)

theorem solution_x_y_z_squared_sum (x y z : ℝ) (h : N_transpose_N x y z = 1) : x^2 + y^2 + z^2 = 47 / 120 :=
sorry

end solution_x_y_z_squared_sum_l575_575847


namespace triangle_area_eq_204_l575_575248

-- Define the side lengths of the triangle
def a := 17
def b := 25
def c := 26

-- Define the semi-perimeter
def s := (a + b + c) / 2

-- Define the area using Heron's formula
def area := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- The theorem we want to prove
theorem triangle_area_eq_204 : area = 204 := 
by 
  sorry

end triangle_area_eq_204_l575_575248


namespace carpet_area_l575_575297

-- Definitions
def Rectangle1 (length1 width1 : ℕ) : Prop :=
  length1 = 12 ∧ width1 = 9

def Rectangle2 (length2 width2 : ℕ) : Prop :=
  length2 = 6 ∧ width2 = 9

def feet_to_yards (feet : ℕ) : ℕ :=
  feet / 3

-- Statement to prove
theorem carpet_area (length1 width1 length2 width2 : ℕ) (h1 : Rectangle1 length1 width1) (h2 : Rectangle2 length2 width2) :
  feet_to_yards (length1 * width1) / 3 + feet_to_yards (length2 * width2) / 3 = 18 :=
by
  sorry

end carpet_area_l575_575297


namespace parallelogram_am_h_l575_575900

noncomputable def parallelogramAngleAMH (A B C D H M : Point) (angleB : ℝ) (BC_eq_BD : BC = BD)
  (BHD_perpendicular : ∠B H D = 90) (M_midpoint_AB : M = midpoint A B) : Prop :=
  ∠A M H = 132

-- Now we define the conditions and goal in Lean
theorem parallelogram_am_h (A B C D H M : Point)
  (h1 : is_parallelogram A B C D)
  (h2 : ∠B = 111)
  (h3 : BC = BD)
  (h4 : is_midpoint H B C)
  (h5 : ∠B H D = 90)
  (h6 : midpoint A B M) :
  ∠A M H = 132 := sorry

end parallelogram_am_h_l575_575900


namespace total_weight_l575_575109

def w1 : ℝ := 9.91
def w2 : ℝ := 4.11

theorem total_weight : w1 + w2 = 14.02 := by 
  sorry

end total_weight_l575_575109


namespace greatest_possible_third_term_l575_575670

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l575_575670


namespace abc_sum_eq_11_sqrt_6_l575_575176

variable {a b c : ℝ}

theorem abc_sum_eq_11_sqrt_6 : 
  0 < a → 0 < b → 0 < c → 
  a * b = 36 → 
  a * c = 72 → 
  b * c = 108 → 
  a + b + c = 11 * Real.sqrt 6 :=
by sorry

end abc_sum_eq_11_sqrt_6_l575_575176


namespace baba_yagas_savings_plan_l575_575601

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l575_575601


namespace midpoints_of_chords_form_circle_l575_575743

theorem midpoints_of_chords_form_circle {K : Type*} [metric_space K] [normed_group K]
  {O P : K} {r : ℝ} (hK : ∀ (x : K), dist O x ≤ r) (hO : dist O P = 4) : 
  ∃ C : K, ∀ Q : K, (∃ (A B : K), dist O A = r ∧ dist O B = r ∧ dist A B = dist A P + dist P B ∧ midpoint A B = Q) → dist C Q = 2 :=
by sorry

end midpoints_of_chords_form_circle_l575_575743


namespace calculate_product_l575_575337

theorem calculate_product : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := 
by
  sorry

end calculate_product_l575_575337


namespace instantaneous_velocity_at_2_l575_575209

def s (t : ℝ) : ℝ := 1 / t + 2 * t

theorem instantaneous_velocity_at_2 :
  (deriv s 2) = 7 / 4 :=
sorry

end instantaneous_velocity_at_2_l575_575209


namespace auditorium_rows_l575_575291

noncomputable def number_of_rows_in_auditorium (ticket_cost : ℕ) (seats_per_row : ℕ) (fraction_sold : ℚ) (total_earned : ℕ) : ℕ :=
  let R := (total_earned / (ticket_cost * fraction_sold * seats_per_row))
  R

theorem auditorium_rows (h_ticket_cost : ℕ)
                       (h_seats_per_row : ℕ)
                       (h_fraction_sold : ℚ)
                       (h_total_earned : ℕ)
                       (h_ticket_cost_eq : h_ticket_cost = 10)
                       (h_seats_per_row_eq : h_seats_per_row = 10)
                       (h_fraction_sold_eq : h_fraction_sold = 3 / 4)
                       (h_total_earned_eq : h_total_earned = 1500) : 
  number_of_rows_in_auditorium h_ticket_cost h_seats_per_row h_fraction_sold h_total_earned = 20 := 
by
  sorry

end auditorium_rows_l575_575291


namespace inequality_x2_y4_z6_l575_575618

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l575_575618


namespace asymptote_slope_l575_575203

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 81 = 1

-- Lean statement to prove slope of asymptotes
theorem asymptote_slope :
  (∀ x y : ℝ, hyperbola x y → (y/x) = 3/4 ∨ (y/x) = -(3/4)) :=
by
  sorry

end asymptote_slope_l575_575203


namespace binary_110011_to_decimal_l575_575396

-- Function to convert a binary list to a decimal number
def binary_to_decimal (bin : List ℕ) : ℕ :=
  bin.reverse.enum.sum (λ ⟨i, b⟩, b * (2 ^ i))

-- Theorem statement
theorem binary_110011_to_decimal : binary_to_decimal [1, 1, 0, 0, 1, 1] = 51 := by
  sorry

end binary_110011_to_decimal_l575_575396


namespace range_of_a_l575_575470

noncomputable def is_function_valid (a : ℝ) : Prop :=
  ∀ x ∈ Ioo (0 : ℝ) (1 / 2), log a (log (2*a) x - x^2) > 0

theorem range_of_a : (∀ x ∈ Ioo (0 : ℝ) (1 / 2), log a (log (2*a) x - x^2) > 0) ↔ (1/32 : ℝ) ≤ a ∧ a < (1 / 2) :=
sorry

end range_of_a_l575_575470


namespace probability_not_finish_l575_575656

theorem probability_not_finish (p : ℝ) (h : p = 5 / 8) : 1 - p = 3 / 8 := 
by 
  rw [h]
  norm_num

end probability_not_finish_l575_575656


namespace time_for_Q_to_finish_job_alone_l575_575712

theorem time_for_Q_to_finish_job_alone (T_Q : ℝ) 
  (h1 : 0 < T_Q)
  (rate_P : ℝ := 1 / 4) 
  (rate_Q : ℝ := 1 / T_Q)
  (combined_work_rate : ℝ := 3 * (rate_P + rate_Q))
  (remaining_work : ℝ := 0.1) -- 0.4 * rate_P
  (total_work_done : ℝ := 0.9) -- 1 - remaining_work
  (h2 : combined_work_rate = total_work_done) : T_Q = 20 :=
by sorry

end time_for_Q_to_finish_job_alone_l575_575712


namespace frequency_interval_20_to_inf_l575_575007

theorem frequency_interval_20_to_inf (sample_size : ℕ)
  (freq_5_10 : ℕ) (freq_10_15 : ℕ) (freq_15_20 : ℕ)
  (freq_20_25 : ℕ) (freq_25_30 : ℕ) (freq_30_35 : ℕ) :
  sample_size = 35 ∧
  freq_5_10 = 5 ∧
  freq_10_15 = 12 ∧
  freq_15_20 = 7 ∧
  freq_20_25 = 5 ∧
  freq_25_30 = 4 ∧
  freq_30_35 = 2 →
  (1 - (freq_5_10 + freq_10_15 + freq_15_20 : ℕ) / (sample_size : ℕ) : ℝ) = 11 / 35 :=
by sorry

end frequency_interval_20_to_inf_l575_575007


namespace chosen_number_is_121_l575_575714

theorem chosen_number_is_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := 
by 
  sorry

end chosen_number_is_121_l575_575714


namespace total_bronze_needed_l575_575584

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l575_575584


namespace complex_ratio_pure_imaginary_l575_575077

theorem complex_ratio_pure_imaginary (a b : ℝ) (h : (∀ z : ℂ, z = a + b * complex.I → (z / (2 + complex.I)).im = z / (2 + complex.I))) : b / a = -2 := sorry

end complex_ratio_pure_imaginary_l575_575077


namespace log2_fraction_l575_575462

variable (a b : ℝ)

-- Conditions given in the problem
axiom log2_3_eq_a : Real.log 2 3 = a
axiom log2_5_eq_b : Real.log 2 5 = b

-- The main theorem to be proved
theorem log2_fraction (a b : ℝ) (log2_3_eq_a : Real.log 2 3 = a) (log2_5_eq_b : Real.log 2 5 = b) :
  Real.log 2 (9 / 5) = 2 * a - b :=
by
  sorry

end log2_fraction_l575_575462


namespace gcd_power_sub_one_eq_gcd_power_diff_sub_one_gcd_power_sub_one_eq_power_gcd_sub_one_power_sub_one_dvd_iff_dvd_l575_575561

-- Conditions: a ≥ 2, m ≥ n ≥ 1
variables (a m n : ℕ) (ha : 2 ≤ a) (hm : 1 ≤ m) (hn : n ≤ m) (hn1 : 1 ≤ n)

-- 1. Prove that: \(\operatorname{PGCD}(a^m - 1, a^n - 1) = \operatorname{PGCD}(a^{m-n} - 1, a^n - 1)\)
theorem gcd_power_sub_one_eq_gcd_power_diff_sub_one (h : 0 < m - n) :
  (nat.gcd (a^m - 1) (a^n - 1) = nat.gcd (a^(m-n) - 1) (a^n - 1)) :=
sorry

-- 2. Prove that: \(\operatorname{PGCD}(a^m - 1, a^n - 1) = a^{\operatorname{PGCD}(m, n)} - 1\)
theorem gcd_power_sub_one_eq_power_gcd_sub_one :
  (nat.gcd (a^m - 1) (a^n - 1) = a^(nat.gcd m n) - 1) :=
sorry

-- 3. Prove that: \(a^m - 1 \mid a^n - 1 \Leftrightarrow m \mid n\)
theorem power_sub_one_dvd_iff_dvd :
  (a^m - 1 ∣ a^n - 1 ↔ m ∣ n) :=
sorry

end gcd_power_sub_one_eq_gcd_power_diff_sub_one_gcd_power_sub_one_eq_power_gcd_sub_one_power_sub_one_dvd_iff_dvd_l575_575561


namespace triangle_area_l575_575205

theorem triangle_area
  (a b : ℝ) (h₁ : a = 3) (h₂ : b = 5)
  (θ : ℝ) (h₃ : 5*θ^2 - 7*θ - 6 = 0)
  (h₄ : θ = -0.6) :
  let sinθ := real.sqrt (1 - θ^2) in
  1/2 * a * b * sinθ = 6 :=
by 
  sorry

end triangle_area_l575_575205


namespace binomial_12_6_l575_575817

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575817


namespace binom_8_5_eq_56_l575_575349

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575349


namespace value_of_x_sq_plus_inv_x_sq_l575_575474

theorem value_of_x_sq_plus_inv_x_sq (x : ℝ) (h : x + 1/x = 1.5) : x^2 + (1/x)^2 = 0.25 := 
by 
  sorry

end value_of_x_sq_plus_inv_x_sq_l575_575474


namespace binom_12_6_l575_575789

theorem binom_12_6 : Nat.choose 12 6 = 924 :=
by
  sorry

end binom_12_6_l575_575789


namespace estimate_students_excellence_l575_575300

noncomputable def estimate_excellent_students : ℝ :=
  let mu : ℝ := 70
  let sigma : ℝ := 10
  let threshold : ℝ := 90
  let total_students : ℕ := 1000
  let p_excellent : ℝ := (1 - 0.9545) / 2
  let expected_excellent_students : ℝ := total_students * p_excellent
  expected_excellent_students

theorem estimate_students_excellence :
  22 ≤ Int.round (estimate_excellent_students) ∧ Int.round (estimate_excellent_students) ≤ 23 :=
by {
  sorry
}

end estimate_students_excellence_l575_575300


namespace binomial_evaluation_l575_575825

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575825


namespace abs_difference_count_l575_575883

noncomputable def tau (n : ℕ) : ℕ := (finset.range n).filter (λ d, n % d = 0).card

noncomputable def S (n : ℕ) : ℕ := 
  (finset.range (n + 1)).sum (λ k, tau (k + 1))

noncomputable def count_odd (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ k, S (k + 1) % 2 = 1).card

noncomputable def count_even (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ k, S (k + 1) % 2 = 0).card

theorem abs_difference_count (m : ℕ) : 
  |count_odd m - count_even m| = 61 :=
sorry

end abs_difference_count_l575_575883


namespace blue_bird_chess_team_arrangements_l575_575182

theorem blue_bird_chess_team_arrangements :
  let boys := 3
  let girls := 2
  let end_positions := 2
  let middle_positions := 3
  let arrangements_girls := factorial end_positions
  let arrangements_boys := factorial middle_positions
  arrangements_girls * arrangements_boys = 12 :=
by
  have boys_count : boys = 3 := rfl
  have girls_count : girls = 2 := rfl
  have end_pos_count : end_positions = 2 := rfl
  have middle_pos_count : middle_positions = 3 := rfl
  have factorial_2 : factorial 2 = 2 := by norm_num
  have factorial_3 : factorial 3 = 6 := by norm_num
  have arrangements_girls_def : arrangements_girls = 2 := by rw [←end_pos_count, factorial_2]
  have arrangements_boys_def : arrangements_boys = 6 := by rw [←middle_pos_count, factorial_3]
  show arrangements_girls * arrangements_boys = 12,
  rw [arrangements_girls_def, arrangements_boys_def],
  norm_num

end blue_bird_chess_team_arrangements_l575_575182


namespace range_m_of_inequality_l575_575896

noncomputable def f (x : ℝ) : ℝ := x / (4 - x^2)

theorem range_m_of_inequality :
  (∀ x ∈ Ioo (-2 : ℝ) 2, f (-x) = -f x) →
  (∀ x₁ x₂ ∈ Ioo (-2 : ℝ) 2, x₁ < x₂ → f x₁ < f x₂) →
  (∀ m : ℝ, f (1 + m) + f (1 - m^2) < 0 ↔ m ∈ Ioo (-Real.sqrt 3) (-1)) :=
by
  sorry

end range_m_of_inequality_l575_575896


namespace smallest_n_for_a_2017_l575_575218

def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (2*n) := if n % 2 = 0 then a n else 2 * a n
| (2*n+1) := if n % 2 = 0 then 2 * a n + 1 else a n

theorem smallest_n_for_a_2017 : a 2017 = a 5 := 
sorry

end smallest_n_for_a_2017_l575_575218


namespace rectangle_diagonal_angles_l575_575642

theorem rectangle_diagonal_angles (x y : ℝ) 
  (is_rectangle : Prop) 
  (alternate_interior_angle : Prop) 
  (diagonal_intersection : Prop) 
  (parallel_segment : Prop) 
  (angles_at_intersection : ∀ (O : Point), acquainted x y): 
  x = 2 * y :=
sorry

end rectangle_diagonal_angles_l575_575642


namespace binom_8_5_l575_575345

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575345


namespace unique_solution_l575_575623

theorem unique_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x * y + y * z + z * x = 12) (eq2 : x * y * z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by 
  sorry

end unique_solution_l575_575623


namespace bikes_added_per_week_l575_575729

variables (x : ℕ) -- bikes added per week

-- Conditions
def original_bikes := 51
def bikes_sold := 18
def end_month_bikes := 45
def weeks_in_month := 4

-- Prove that the number of bikes added per week is 3
theorem bikes_added_per_week : 
  (original_bikes - bikes_sold + weeks_in_month * x = end_month_bikes) → 
  x = 3 := by
  sorry

end bikes_added_per_week_l575_575729


namespace binary_to_decimal_110011_l575_575380

theorem binary_to_decimal_110011 :
  let b := 110011
  ∑ i in [0, 1, 4, 5], (b.digits 2)[i] * 2^i = 51 := by
  sorry

end binary_to_decimal_110011_l575_575380


namespace travel_time_correct_l575_575755

-- Define the variables
variables (x y T : ℝ) (total_distance car_speed bike_speed hike_speed : ℝ)

-- Set the given conditions
def given_conditions : Prop :=
  total_distance = 150 ∧
  car_speed = 30 ∧
  bike_speed = 10 ∧
  hike_speed = 3

-- Define the equations from the problem
def charlie_time : ℝ := x / car_speed + (total_distance - x) / hike_speed
def alice_time : ℝ := x / car_speed + y / car_speed + (total_distance - (x - y)) / car_speed
def bob_time : ℝ := (x - y) / bike_speed + (total_distance - (x - y)) / car_speed

-- Define the equality of the total time
def total_travel_time : Prop :=
  T = charlie_time ∧
  T = alice_time ∧
  T = bob_time

-- The theorem stating the problem to prove
theorem travel_time_correct (conds : given_conditions) : total_travel_time :=
sorry

end travel_time_correct_l575_575755


namespace solve_for_n_l575_575410

theorem solve_for_n (n : ℤ) (h : (1 : ℤ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) : n = 2 :=
sorry

end solve_for_n_l575_575410


namespace sum_of_integers_l575_575196

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 18) (h3 : x * y = 72) : x + y = 2 * real.sqrt 153 :=
by
  sorry

end sum_of_integers_l575_575196


namespace bear_eats_victors_l575_575270

variable (x : ℕ)

theorem bear_eats_victors (x : ℕ) : (5 * x) = food_in_victors x := 
sorry

end bear_eats_victors_l575_575270


namespace trigonometric_identity_l575_575467

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : 2 * real.sin θ + real.cos θ = 0) : 
  real.sin θ * real.cos θ - real.cos θ ^ 2 = -6 / 5 := 
by 
  sorry

end trigonometric_identity_l575_575467


namespace measure_angle_PQT_l575_575155

/-- PQRSTU is a regular octagon. -/
def is_regular_octagon (PQRSTU : Fin 8 → Point) :=
  ∀ i, (dist (PQRSTU i) (PQRSTU ((i + 1) % 8))) = (dist (PQRSTU 0) (PQRSTU 1)) ∧
       (interior_angle (polygon.rotate PQRSTU i % 8) = 135)

/-- Regular octagon PQRSTU -/
variable {PQRSTU : Fin 8 → typePoint}

/-- PQT is isosceles triangle within the regular octagon -/
def is_isosceles_triangle (P : typePoint) (Q : typePoint) (T : typePoint) :=
  dist P Q = dist P T

theorem measure_angle_PQT (h : is_regular_octagon PQRSTU) (hPQT : is_isosceles_triangle (PQRSTU 0) (PQRSTU 1) (PQRSTU 2)):
  measure_angle (PQRSTU 0) (PQRSTU 1) (PQRSTU 2) = 22.5 :=
sorry

end measure_angle_PQT_l575_575155


namespace binary_to_decimal_conversion_l575_575849

theorem binary_to_decimal_conversion : 
  (let n := 10 in ∑ i in Finset.range n, 2^i) = 1023 := by
sorry

end binary_to_decimal_conversion_l575_575849


namespace binomial_12_6_l575_575821

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575821


namespace candy_bars_total_l575_575227

theorem candy_bars_total :
  let people : ℝ := 3.0;
  let candy_per_person : ℝ := 1.66666666699999;
  people * candy_per_person = 5.0 :=
by
  let people : ℝ := 3.0
  let candy_per_person : ℝ := 1.66666666699999
  show people * candy_per_person = 5.0
  sorry

end candy_bars_total_l575_575227


namespace determine_g_l575_575177

noncomputable def f (x : ℝ) : ℝ := x ^ 2

noncomputable def g (x : ℝ) : ℝ

theorem determine_g :
  (f(g(x)) = 9 * x ^ 2 - 6 * x + 1) →
  (g(x) = 3 * x - 1 ∨ g(x) = -3 * x + 1) :=
sorry

end determine_g_l575_575177


namespace moles_of_CO2_required_l575_575502

theorem moles_of_CO2_required (n_H2O n_H2CO3 : ℕ) (h1 : n_H2O = n_H2CO3) (h2 : n_H2O = 2): 
  (n_H2O = 2) → (∃ n_CO2 : ℕ, n_CO2 = n_H2O) :=
by
  sorry

end moles_of_CO2_required_l575_575502


namespace binomial_12_6_eq_924_l575_575780

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575780


namespace binomial_12_6_eq_924_l575_575839

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575839


namespace smallest_two_digit_multiple_of_3_l575_575707

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem smallest_two_digit_multiple_of_3 : ∃ n : ℕ, is_two_digit n ∧ is_multiple_of_3 n ∧ ∀ m : ℕ, is_two_digit m ∧ is_multiple_of_3 m → n <= m :=
sorry

end smallest_two_digit_multiple_of_3_l575_575707


namespace log_base_5_of_1_div_25_l575_575418

theorem log_base_5_of_1_div_25 : log 5 (1 / 25) = -2 := by
  sorry

end log_base_5_of_1_div_25_l575_575418


namespace line_increase_is_110_l575_575651

noncomputable def original_lines (increased_lines : ℕ) (percentage_increase : ℚ) : ℚ :=
  increased_lines / (1 + percentage_increase)

theorem line_increase_is_110
  (L' : ℕ)
  (percentage_increase : ℚ)
  (hL' : L' = 240)
  (hp : percentage_increase = 0.8461538461538461) :
  L' - original_lines L' percentage_increase = 110 :=
by
  sorry

end line_increase_is_110_l575_575651


namespace tetrahedron_volume_inequality_l575_575904

noncomputable def volume_bound (a b c d e f V : ℝ) : Prop :=
  3 * V ≤ (1 / 12 * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2))^(3 / 2)

theorem tetrahedron_volume_inequality (a b c d e f V : ℝ) (hV : V = volume_of_tetrahedron a b c d e f) :
  volume_bound a b c d e f V :=
sorry

end tetrahedron_volume_inequality_l575_575904


namespace matias_fewer_cards_l575_575108

theorem matias_fewer_cards (J M C : ℕ) (h1 : J = M) (h2 : C = 20) (h3 : C + M + J = 48) : C - M = 6 :=
by
-- To be proven
  sorry

end matias_fewer_cards_l575_575108


namespace distance_from_E_to_B_is_1_5_l575_575442

open_locale classical
noncomputable theory

-- Define Points A, B, C1, D1, and E in 3D space
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C1 : ℝ × ℝ × ℝ := (1, 1, 1)
def D1 : ℝ × ℝ × ℝ := (0, 1, 1)
def E : ℝ × ℝ × ℝ := (0.5, 1, 1)

-- Define Euclidean distance
def euclidean_dist (p q : ℝ × ℝ × ℝ) : ℝ :=
  (real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2))

-- Problem statement: Prove the distance from E to B is 1.5
theorem distance_from_E_to_B_is_1_5 : euclidean_dist E B = 1.5 :=
by {
  sorry
}

end distance_from_E_to_B_is_1_5_l575_575442


namespace eccentricity_range_l575_575923

-- Define the major and minor axes lengths and the foci distances
variables (a b : ℝ) (h : a > b) (h_pos : b > 0)

-- Define the eccentricity e
variable (e : ℝ)

-- Define the focus points and the condition on PF1 and PF2
variables (F1 F2 P : (ℝ × ℝ))
variables (h_ellipse : P ∈ set_of (λ p, (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1))
variable (h_ratio : dist P F1 / dist P F2 = e)

-- Goal: prove that the range of the eccentricity e is between sqrt(2) - 1 and 1
theorem eccentricity_range : sqrt 2 - 1 ≤ e ∧ e < 1 :=
sorry

end eccentricity_range_l575_575923


namespace binary_num_to_decimal_eq_51_l575_575394

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575394


namespace part1_part2_l575_575048

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (a x : ℝ) := a * |x - 1|

theorem part1 (a : ℝ) :
  (∀ x : ℝ, |f x| = g a x → x = 1) → a < 0 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

end part1_part2_l575_575048


namespace angle_at_3_25_l575_575407

def hoursToDegrees (hours : ℕ) : ℝ :=
  30 * hours

def minutesToHours (minutes : ℕ) : ℝ :=
  (minutes : ℝ) / 60

def clockDegreesAt (hours minutes : ℕ) : ℝ :=
  hoursToDegrees hours + 30 * (minutesToHours minutes)

def minutesToDegrees (minutes : ℕ) : ℝ :=
  (minutes : ℝ) / 60 * 360

def angleBetweenHands (hours minutes : ℕ) : ℝ :=
  let hourDegrees := clockDegreesAt hours minutes
  let minuteDegrees := minutesToDegrees minutes
  let diff := |minuteDegrees - hourDegrees|
  if diff > 180 then 360 - diff else diff

theorem angle_at_3_25 : angleBetweenHands 3 25 = 47.5 := sorry

end angle_at_3_25_l575_575407


namespace shortest_path_length_l575_575296

theorem shortest_path_length :
  ∀ (floor_x floor_y ceiling_height : ℝ)
    (fly_x fly_y fly_z : ℝ)
    (spider_x spider_y spider_z : ℝ),
    floor_x = 7 ∧ floor_y = 8 ∧ ceiling_height = 4 ∧
    fly_x = 0 ∧ fly_y = 0 ∧ fly_z = 4 ∧
    spider_x = 7 ∧ spider_y = 8 ∧ spider_z = 4 →
    ∃ d : ℝ, d = real.sqrt (4^2 + 7^2 + 8^2) ∧ d = real.sqrt 129 :=
sorry

end shortest_path_length_l575_575296


namespace dot_product_AB_AC_l575_575040

noncomputable def f (x : ℝ) : ℝ :=
  √3 * sin (π - x) * cos (-x) + sin (π + x) * cos (π / 2 - x)

def Point : Type := ℝ × ℝ

def A : Point := (2 * π / 3, -3 / 2)
def B : Point := (π / 6, 1 / 2)
def C : Point := (7 * π / 6, 1 / 2)

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_AB_AC : 
  dot_product (vector A B) (vector A C) = 4 - π^2 / 4 :=
by
  sorry

end dot_product_AB_AC_l575_575040


namespace biology_marks_correct_l575_575616

-- Define the known marks in other subjects
def math_marks : ℕ := 76
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 62

-- Define the total number of subjects
def total_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℕ := 74

-- Calculate the total marks of the known four subjects
def total_known_marks : ℕ := math_marks + science_marks + social_studies_marks + english_marks

-- Define a variable to represent the marks in biology
def biology_marks : ℕ := 370 - total_known_marks

-- Statement to prove
theorem biology_marks_correct : biology_marks = 85 := by
  sorry

end biology_marks_correct_l575_575616


namespace not_T1_not_T2_not_T3_l575_575008

section problem

-- Definitions of the sets and elements
inductive pib
| p1 | p2 | p3 | p4 | p5

inductive maa

-- Conditions
def collection_of_maas (P : pib) : set maa := sorry

axiom P1 : ∀ (P : pib), ∃ (M : set maa), collection_of_maas P = M

axiom P2 : ∀ (P Q : pib), P ≠ Q → (collection_of_maas P ∩ collection_of_maas Q).to_finset.card ≤ 2

axiom P3 : ∀ (m : maa), 2 ≤ {P : pib | m ∈ collection_of_maas P}.to_finset.card ∧
                       {P : pib | m ∈ collection_of_maas P}.to_finset.card ≤ 3

axiom P4 : {P : pib | true}.to_finset.card = 5

-- Theorems to be investigated
def T1 : Prop := {m : maa | true}.to_finset.card = 10
def T2 : Prop := ∀ (P : pib), (collection_of_maas P).to_finset.card = 4
def T3 : Prop := ∀ (m : maa), ∃! (m' : maa), ∀ (P : pib), m' ∉ collection_of_maas P ∨ m ∉ collection_of_maas P

-- Proof goals stating that these theorems cannot be confirmed
theorem not_T1 : ¬ T1 :=
by sorry

theorem not_T2 : ¬ T2 :=
by sorry

theorem not_T3 : ¬ T3 :=
by sorry

end problem

end not_T1_not_T2_not_T3_l575_575008


namespace suitable_candidate_l575_575761

noncomputable def S_A_squared : ℝ := 2.25
noncomputable def S_B_squared : ℝ := 1.81
noncomputable def S_C_squared : ℝ := 3.42

theorem suitable_candidate :
  S_B_squared < S_A_squared ∧ S_B_squared < S_C_squared :=
by {
  have h₁ : S_B_squared < S_A_squared := by sorry,
  have h₂ : S_B_squared < S_C_squared := by sorry,
  exact ⟨h₁, h₂⟩,
}

end suitable_candidate_l575_575761


namespace well_depth_and_rope_length_l575_575632

variables (h x : ℝ)

theorem well_depth_and_rope_length :
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) → True := 
by
  intro h_eq x_eq
  sorry

end well_depth_and_rope_length_l575_575632


namespace even_number_exists_from_step_2_l575_575538

noncomputable def update_sequence (seq : ℤ → ℤ) : ℤ → ℤ :=
  λ n, seq (n - 1) + seq (n + 1)

theorem even_number_exists_from_step_2
  (seq : ℤ → ℤ)
  (h_initial : seq 0 = 1 ∧ (∀ n ≠ 0, seq n = 0))
  (∀ n, ∃ k, k ≥ 0 → seq k = update_sequence (update_sequence seq) k)
  : ∃ (k : ℤ), k ≥ 2 ∧ even (seq k) ∧ seq k ≠ 0 := 
sorry

end even_number_exists_from_step_2_l575_575538


namespace number_of_triples_satisfying_conditions_l575_575871

theorem number_of_triples_satisfying_conditions :
  let gcd := (a b c : ℕ) → Nat.gcd (Nat.gcd a b) c
  let lcm := (a b c : ℕ) → Nat.lcm (Nat.lcm a b) c
  ∃ (a b c : ℕ), gcd a b c = 22 ∧ lcm a b c = 2^16 * 11^19 ∧
  ( 
  num_of_triples : 
    nat.of_int ((min ![a, b, c] ![a, b, c].map (λ x, (x = 1).card)) = 1) ∨ 1 ) * 
    nat.of_int (( max ![a, b, c] ![a, b, c].map (λ x, (x == 16).card)) == 16 ) ∨ 100 ∀ int.num_of_triples =  9720) 
= num_of_triples sorry 

end number_of_triples_satisfying_conditions_l575_575871


namespace greatest_possible_area_difference_l575_575239

theorem greatest_possible_area_difference :
  ∀ (l1 w1 l2 w2 : ℕ), 
  (2 * l1 + 2 * w1 = 200) → 
  (2 * l2 + 2 * w2 = 200) → 
  (l1 ≥ 25 ∨ w1 ≥ 25) → 
  (l2 ≥ 25 ∨ w2 ≥ 25) → 
  (0 ≤ l1 * w1 - l2 * w2 ∧ l1 * w1 - l2 * w2 ≤ 0) :=
by
  intros l1 w1 l2 w2 h1 h2 h3 h4
  have hw1: w1 = 100 - l1, from (eq_sub_of_add_eq (by linarith [h1])),
  have hw2: w2 = 100 - l2, from (eq_sub_of_add_eq (by linarith [h2])),
  have ha1: l1 * (100 - l1) = 2500 - (l1 - 50)^2 := by sorry,
  have ha2: l2 * (100 - l2) = 2500 - (l2 - 50)^2 := by sorry,
  have h_max_area: l1 * (100 - l1) ≤ 1875, from (by linarith [ha1]),
  have h_min_area: l1 * (100 - l1) ≥ 1875, from (by linarith [ha1]),
  have h_max_area2: l2 * (100 - l2) ≤ 1875, from (by linarith [ha2]),
  have h_min_area2: l2 * (100 - l2) ≥ 1875, from (by linarith [ha2]),
  have h5: l1 * (100 - l1) = 1875, from by linarith [h_max_area, h_min_area],
  have h6: l2 * (100 - l2) = 1875, from by linarith [h_max_area2, h_min_area2],
  rw [h5, h6],
  linarith

end greatest_possible_area_difference_l575_575239


namespace binom_8_5_l575_575344

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l575_575344


namespace fill_pipe_fraction_l575_575735

theorem fill_pipe_fraction (C : Type*) (por : C) (cistern : C) (t : ℕ)
  (h1: t = 35) (h2: fills_in_35_minutes (por)) (h3: t = 35 ∧ por = cistern) :
  fraction_filled por cistern = 1 := by
  sorry

end fill_pipe_fraction_l575_575735


namespace value_of_cd_l575_575954

variables (α β a b c d : ℝ)
variable (h_sin_roots : ∀ x : ℝ, x^2 - a * x + b = 0 → x = sin α ∨ x = sin β)
variable (h_cos_roots : ∀ x : ℝ, x^2 - c * x + d = 0 → x = cos α ∨ x = cos β)

theorem value_of_cd (α β a b c d : ℝ) 
  (h_sin_roots : ∀ x : ℝ, x^2 - a * x + b = 0 → x = sin α ∨ x = sin β)
  (h_cos_roots : ∀ x : ℝ, x^2 - c * x + d = 0 → x = cos α ∨ x = cos β) :
  c * d = 1 / 2 :=
sorry

end value_of_cd_l575_575954


namespace value_range_f_in_0_to_4_l575_575676

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem value_range_f_in_0_to_4 :
  ∀ (x : ℝ), (0 < x ∧ x ≤ 4) → (1 ≤ f x ∧ f x ≤ 10) :=
sorry

end value_range_f_in_0_to_4_l575_575676


namespace handshakes_at_gathering_l575_575321

/--
At a gathering of 40 people, there are 25 people in Group A who all know each other, and 15 people in Group B. Out 
of these 15, each person knows exactly 5 people from Group A. People who know each other hug, and people who do 
not know each other shake hands. Prove that the total number of handshakes that occur within the group is 330.
-/
theorem handshakes_at_gathering (total_people : ℕ) (group_A : ℕ) (group_B : ℕ) 
  (know_each_other_in_A : ∀ x y, x ∈ group_A → y ∈ group_A → x ≠ y → true) 
  (know_exactly_five_in_B : ∀ b, b ∈ group_B → fintype.card ({a // a ∈ group_A ∧ knows b a}) = 5) 
  (doesnt_know_shakes_hand : ∀ x y, x ≠ y → (¬ knows x y) → shakes_hand x y) 
  : number_of_handshakes = 330 := 
sorry

end handshakes_at_gathering_l575_575321


namespace mean_of_side_lengths_l575_575637

def area1 : ℝ := 25
def area2 : ℝ := 64
def area3 : ℝ := 100
def area4 : ℝ := 144

def side_length (area : ℝ) : ℝ := Real.sqrt area

def mean (l₁ l₂ l₃ l₄ : ℝ) : ℝ := (l₁ + l₂ + l₃ + l₄) / 4

theorem mean_of_side_lengths :
  mean (side_length area1) (side_length area2) (side_length area3) (side_length area4) = 8.75 :=
by
  sorry

end mean_of_side_lengths_l575_575637


namespace solution_set_l575_575404

noncomputable def f : ℝ → ℝ
variable (x : ℝ)

axiom f_derivative (x : ℝ) : has_deriv_at f (f' x) x

theorem solution_set :
  (∀ x ∈ ℝ, f x + f' x < real.exp 1) →
  (f 0 = real.exp 1 + 2) →
  {x : ℝ | real.exp x * f x > real.exp (x + 1) + 2} = set.Iio 0 :=
by
  sorry

end solution_set_l575_575404


namespace centroid_triangle_A_l575_575122

open Set

variable {A B C D M N P Q A_1 D_1 B_1 C_1 : Point}
variable {BCM CDM BDM BCD : Plane}
variable (AD AB AC AM A1N A1P A1Q : Line)
variable [h1 : intersects_plane_line BCM AD N]
variable [h2 : intersects_plane_line CDM AB P]
variable [h3 : intersects_plane_line BDM AC Q]
variable [h4 : intersects_plane_line BCD AM A_1]
variable [h5 : parallel_plane BCD (Plane.of_points A D_1 B_1)]
variable [h6 : parallel_plane BCD (Plane.of_points A C_1 D_1)]
variable [h7 : intersects_lines {A_1N} {A_1P} {A_1Q} D_1 B_1 C_1]

theorem centroid_triangle_A :
  centroid (Triangle.mk B_1 C_1 D_1) = A := 
sorry

end centroid_triangle_A_l575_575122


namespace binomial_12_6_eq_924_l575_575786

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575786


namespace find_m_and_equation_of_l2_l575_575930

theorem find_m_and_equation_of_l2 (a : ℝ) (M: ℝ × ℝ) (m : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hM : M = (-5, 1)) 
  (hl1 : ∀ {x y : ℝ}, 2 * x - y + 2 = 0) 
  (hl : ∀ {x y : ℝ}, x + y + m = 0) 
  (hl2 : ∀ {x y : ℝ}, (∃ p : ℝ × ℝ, p = M → x - 2 * y + 7 = 0)) : 
  m = -5 ∧ ∀ {x y : ℝ}, x - 2 * y + 7 = 0 :=
by
  sorry

end find_m_and_equation_of_l2_l575_575930


namespace set_union_example_l575_575492

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem set_union_example : M ∪ N = {1, 2, 3, 4} := by
  sorry

end set_union_example_l575_575492


namespace find_a1_l575_575010

variable {a_n : ℕ → ℤ}
variable (common_difference : ℤ) (a1 : ℤ)

-- Define that a_n is an arithmetic sequence with common difference of 2
def is_arithmetic_seq (a_n : ℕ → ℤ) (common_difference : ℤ) : Prop :=
  ∀ n, a_n (n + 1) - a_n n = common_difference

-- State the condition that a1, a2, a4 form a geometric sequence
def forms_geometric_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ a1 a2 a4, a2 * a2 = a1 * a4 ∧ a_n 1 = a1 ∧ a_n 2 = a2 ∧ a_n 4 = a4

-- Define the problem statement
theorem find_a1 (h_arith : is_arithmetic_seq a_n 2) (h_geom : forms_geometric_seq a_n) :
  a_n 1 = 2 :=
by
  sorry

end find_a1_l575_575010


namespace porch_width_l575_575646

theorem porch_width (L_house W_house total_area porch_length W : ℝ)
  (h1 : L_house = 20.5) (h2 : W_house = 10) (h3 : total_area = 232) (h4 : porch_length = 6) (h5 : total_area = (L_house * W_house) + (porch_length * W)) :
  W = 4.5 :=
by 
  sorry

end porch_width_l575_575646


namespace greatest_even_integer_leq_z_l575_575880

theorem greatest_even_integer_leq_z (z : ℝ) (z_star : ℝ → ℝ)
  (h1 : ∀ z, z_star z = z_star (z - (z - z_star z))) -- (This is to match the definition given)
  (h2 : 6.30 - z_star 6.30 = 0.2999999999999998) : z_star 6.30 ≤ 6.30 := by
sorry

end greatest_even_integer_leq_z_l575_575880


namespace kelsey_total_distance_l575_575993

-- Define the constants and variables involved
def total_distance (total_time : ℕ) (speed1 speed2 half_dist1 half_dist2 : ℕ) : ℕ :=
  let T1 := half_dist1 / speed1
  let T2 := half_dist2 / speed2
  let T := T1 + T2
  total_time

-- Prove the equivalency given the conditions
theorem kelsey_total_distance (total_time : ℕ) (speed1 speed2 : ℕ) : 
  (total_time = 10) ∧ (speed1 = 25) ∧ (speed2 = 40)  →
  ∃ D, D = 307 ∧ (10 = D / 50 + D / 80) :=
by 
  intro h
  have h_total_time := h.1
  have h_speed1 := h.2.1
  have h_speed2 := h.2.2
  -- Need to prove the statement using provided conditions
  let D := 307
  sorry

end kelsey_total_distance_l575_575993


namespace binomial_12_6_l575_575822

theorem binomial_12_6 : nat.choose 12 6 = 924 :=
by
  sorry

end binomial_12_6_l575_575822


namespace sum_of_digits_N_l575_575302

noncomputable def triangularArraySumOfDigits (N : ℕ) : ℕ :=
  (N + 1) * N / 2

theorem sum_of_digits_N (N : ℕ) (h : triangularArraySumOfDigits N = 3003) :
  Nat.digits 10 N |>.sum = 14 :=
sorry

end sum_of_digits_N_l575_575302


namespace max_third_term_is_16_l575_575666

-- Define the arithmetic sequence conditions
def arithmetic_seq (a d : ℕ) : list ℕ := [a, a + d, a + 2 * d, a + 3 * d]

-- Define the sum condition
def sum_of_sequence_is_50 (a d : ℕ) : Prop :=
  (a + a + d + a + 2 * d + a + 3 * d) = 50

-- Define the third term of the sequence
def third_term (a d : ℕ) : ℕ := a + 2 * d

-- Prove that the greatest possible third term is 16
theorem max_third_term_is_16 : ∃ (a d : ℕ), sum_of_sequence_is_50 a d ∧ third_term a d = 16 :=
by
  sorry

end max_third_term_is_16_l575_575666


namespace element_correspondence_l575_575473

variable (A B : Type) [Inhabited A] [Inhabited B]
variable (f : A → B) (x : A) (y : B)

def A := ℝ
def B := ℝ
def f : ℝ → ℝ := λ x, 2 * x - 1

theorem element_correspondence :
  (∃ x : ℝ, f x = 3) ↔ (2 = 2) :=
  by
    sorry

end element_correspondence_l575_575473


namespace product_of_last_two_digits_l575_575963

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 11) (h2 : ∃ (n : ℕ), 10 * A + B = 6 * n) : A * B = 24 :=
sorry

end product_of_last_two_digits_l575_575963


namespace binary_to_decimal_110011_l575_575388

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575388


namespace sum_of_repeating_decimals_l575_575421

noncomputable def repeat_decimal_to_fraction (n d : ℕ) : ℚ :=
  (n : ℚ) / (d - 1)

theorem sum_of_repeating_decimals :
  let x := repeat_decimal_to_fraction 4 10,
  let y := repeat_decimal_to_fraction 6 10 in
  (x + y : ℚ) = 10 / 9 :=
by
  let x := repeat_decimal_to_fraction 4 10
  let y := repeat_decimal_to_fraction 6 10
  have hx : x = 4 / 9 := by sorry
  have hy : y = 2 / 3 := by sorry
  calc
    x + y = 4 / 9 + 2 / 3         : by rw [hx, hy]
    ...   = 4 / 9 + 6 / 9         : by norm_num
    ...   = (4 + 6) / 9           : by ring
    ...   = 10 / 9                : by norm_num

end sum_of_repeating_decimals_l575_575421


namespace binary_num_to_decimal_eq_51_l575_575391

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575391


namespace count_multiples_of_4_between_100_and_350_l575_575945

theorem count_multiples_of_4_between_100_and_350 : 
  (∃ n : ℕ, 104 + (n - 1) * 4 = 348) ∧ (∀ k : ℕ, (104 + k * 4 ∈ set.Icc 100 350) ↔ (k ≤ 61)) → 
  n = 62 :=
by
  sorry

end count_multiples_of_4_between_100_and_350_l575_575945


namespace find_xyz_l575_575563

-- Let a, b, c, x, y, z be nonzero complex numbers
variables (a b c x y z : ℂ)
-- Given conditions
variables (h1 : a = (b + c) / (x - 2))
variables (h2 : b = (a + c) / (y - 2))
variables (h3 : c = (a + b) / (z - 2))
variables (h4 : x * y + x * z + y * z = 5)
variables (h5 : x + y + z = 3)

-- Prove that xyz = 5
theorem find_xyz : x * y * z = 5 :=
by
  sorry

end find_xyz_l575_575563


namespace sum_even_integers_102_to_200_l575_575220

theorem sum_even_integers_102_to_200 : 
  let sum_series (n : ℕ) (a₁ aₙ : ℕ) := n / 2 * (a₁ + aₙ) in
  sum_series 50 102 200 = 7550 :=
by
  sorry

end sum_even_integers_102_to_200_l575_575220


namespace evaluate_expression_l575_575857

theorem evaluate_expression :
  (∀ x : ℂ, (x^3 + 1 = (x + 1) * (x^2 - x + 1)) ∧
            (x^3 - 1 = (x - 1) * (x^2 + x + 1))) →
  ∀ x : ℂ,
  ((x^2 + 2*x + 2)^2 * (x^4 - x^2 + 1)^2 / (x^3 + 1)^3) *
  ((x^2 - 2*x + 2)^2 * (x^4 + x^2 + 1)^2 / (x^3 - 1)^3) = 1 :=
begin
  sorry
end

end evaluate_expression_l575_575857


namespace inequality_solution_l575_575862

theorem inequality_solution 
  (x : ℝ) : 
  (x^2 / (x+2)^2 ≥ 0) ↔ x ≠ -2 := 
by
  sorry

end inequality_solution_l575_575862


namespace cylinder_in_sphere_volume_difference_is_correct_l575_575295

noncomputable def volume_difference (base_radius_cylinder : ℝ) (radius_sphere : ℝ) : ℝ :=
  let height_cylinder := Real.sqrt (radius_sphere^2 - base_radius_cylinder^2)
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere^3
  let volume_cylinder := Real.pi * base_radius_cylinder^2 * height_cylinder
  volume_sphere - volume_cylinder

theorem cylinder_in_sphere_volume_difference_is_correct :
  volume_difference 4 7 = (1372 - 48 * Real.sqrt 33) / 3 * Real.pi :=
by
  sorry

end cylinder_in_sphere_volume_difference_is_correct_l575_575295


namespace binary_num_to_decimal_eq_51_l575_575393

-- Define the binary number as a list of bits
def binary_num : List ℕ := [1, 1, 0, 0, 1, 1]

-- Define the function to convert a binary number to a decimal number
def binary_to_decimal (bits : List ℕ) : ℕ :=
  bits.foldr (λ (bit acc : ℕ), bit + 2 * acc) 0

-- Prove that converting 110011 from binary to decimal equals 51
theorem binary_num_to_decimal_eq_51 : binary_to_decimal binary_num = 51 :=
by
  -- The proof is intentionally omitted
  sorry

end binary_num_to_decimal_eq_51_l575_575393


namespace min_omega_l575_575045

theorem min_omega (f : Real → Real) (ω φ : Real) (φ_bound : |φ| < π / 2) 
  (h1 : ω > 0) (h2 : f = fun x => Real.sin (ω * x + φ)) 
  (h3 : f 0 = 1/2) 
  (h4 : ∀ x, f x ≤ f (π / 12)) : ω = 4 := 
by
  sorry

end min_omega_l575_575045


namespace eggs_left_over_l575_575402

theorem eggs_left_over (David_eggs Ella_eggs Fiona_eggs : ℕ)
  (hD : David_eggs = 45)
  (hE : Ella_eggs = 58)
  (hF : Fiona_eggs = 29) :
  (David_eggs + Ella_eggs + Fiona_eggs) % 10 = 2 :=
by
  sorry

end eggs_left_over_l575_575402


namespace solve_expression_l575_575855

theorem solve_expression (x : ℝ) :
  (x^2 - x - 6 = 0) → (5x - 15 ≠ 0) → (x = -2) :=
by
  intros h1 h2
  /- proof steps here -/
  sorry

end solve_expression_l575_575855


namespace car_speed_increase_l575_575664

theorem car_speed_increase (x : ℕ) 
  (h1 : ∑ i in finset.range 12, (55 + i * x) = 792) : 
  x = 2 := 
sorry

end car_speed_increase_l575_575664


namespace monthly_savings_correct_l575_575593

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l575_575593


namespace part1_part2_part3_l575_575565

noncomputable def floor (x: ℝ) : ℤ := Int.floor x

noncomputable def f1 (x: ℝ) : ℤ := floor x
noncomputable def f2 (x: ℝ) : ℤ := floor ((x + 1) / 2) - floor (x / 2)

variable (x: ℝ) (a: ℝ)

axiom is_Omega_function (f: ℝ → ℝ) : Prop :=
∃ m ∈ ℝ, m ∉ ℤ ∧ f m = f (floor m)

theorem part1 (x: ℝ) : f1 1.2 = 1 ∧ f1 (-1.2) = -2 :=
by
  split
  sorry
  sorry

theorem part2 (x: ℝ) : ∀ x ∈ ℝ, f2 x ∈ {0, 1} :=
by
  intros
  sorry

theorem part3 (x: ℝ) (a: ℝ) : 
  is_Omega_function (λ x, x + a / x) →
  ∀ a ∈ ℝ, a > 0 → ¬∃ k ∈ ℕ*, (a = (k ^ 2) ∨ a = k * (k + 1)) :=
by
  intros
  sorry

end part1_part2_part3_l575_575565


namespace binary_to_decimal_110011_l575_575384

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575384


namespace log_eq_half_union_sets_l575_575938

variable (p q : ℤ)
def S := {x : ℤ | x^2 - p * x + q = 0}
def T := {x : ℤ | x^2 - (p + 3) * x + 6 = 0}

theorem log_eq_half (h : S ∩ T = {3}) : Real.logb 9 (3 * p + q) = 1 / 2 :=
sorry

theorem union_sets (h : S ∩ T = {3}) : S ∪ T = {-1, 2, 3} :=
sorry

end log_eq_half_union_sets_l575_575938


namespace coefficient_of_x6_in_expansion_l575_575249

theorem coefficient_of_x6_in_expansion :
  (∃ (c : ℕ), (∀ (x : ℕ), (coeff (3 * x + 2) 9 x^6) = c) ∧ c = 486144) :=
sorry

end coefficient_of_x6_in_expansion_l575_575249


namespace insect_population_calculations_l575_575524

theorem insect_population_calculations :
  (let ants_1 := 100
   let ants_2 := ants_1 - 20 * ants_1 / 100
   let ants_3 := ants_2 - 25 * ants_2 / 100
   let bees_1 := 150
   let bees_2 := bees_1 - 30 * bees_1 / 100
   let termites_1 := 200
   let termites_2 := termites_1 - 10 * termites_1 / 100
   ants_3 = 60 ∧ bees_2 = 105 ∧ termites_2 = 180) :=
by
  sorry

end insect_population_calculations_l575_575524


namespace sum_of_digits_sqrt_N_l575_575873

theorem sum_of_digits_sqrt_N :
  let N := (10^2017 - 1) * 10^2019 / 9 + (10^2018 - 1) * 20 / 9 + 5 in
  let sqrt_N := (10^2018 + 5) / 3 in
  let sum_of_digits_of_sqrt_N := 2017 * 3 + 5 in
  sum_of_digits_of_sqrt_N = 6056 := 
sorry

end sum_of_digits_sqrt_N_l575_575873


namespace sum_of_values_for_f_eq_2004_l575_575737

variable (f : ℝ → ℝ)

theorem sum_of_values_for_f_eq_2004 
  (h : ∀ (x : ℝ), x ≠ 0 → 2 * f x + f (1 / x) = 5 * x + 4) :
  ∑ x in {x : ℝ | f x = 2004} = 601 := 
sorry

end sum_of_values_for_f_eq_2004_l575_575737


namespace α_perpendicular_β_l575_575915

-- Definitions of the given conditions
variable (l m : Line)
variable (α β : Plane)

-- Conditions
axiom l_perpendicular_α : IsPerpendicular l α
axiom m_in_β : Contains β m
axiom l_parallel_m : IsParallel l m

-- Statement to prove
theorem α_perpendicular_β : IsPerpendicular α β :=
sorry

end α_perpendicular_β_l575_575915


namespace binomial_evaluation_l575_575823

-- Defining the binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.choose k

-- Theorem stating our problem
theorem binomial_evaluation : binomial 12 6 = 924 := 
by sorry

end binomial_evaluation_l575_575823


namespace sin_product_l575_575841

theorem sin_product (theta : ℝ) :
  (θ = π / 9) →
  (sin (3 * θ) = 3 * sin θ - 4 * sin θ ^ 3) →
  sin (π / 9) * sin (2 * π / 9) * sin (4 * π / 9) = (√3) / 8 :=
  sorry

end sin_product_l575_575841


namespace smallest_n_for_gn_gt_15_l575_575959

def sum_of_digits (n : Nat) : Nat :=
  n.digits.sum

def integer_part_base (a b : Nat) : Nat :=
  (a.toRational / b.toRational).floor

noncomputable def g (n : Nat) : Nat :=
  sum_of_digits (integer_part_base 10 3 ^ n)

theorem smallest_n_for_gn_gt_15 : ∃ n : Nat, n > 0 ∧ g n > 15 ∧ ∀ m : Nat, m > 0 ∧ m < n → g m ≤ 15 :=
by
  use 7
  sorry

end smallest_n_for_gn_gt_15_l575_575959


namespace product_of_sequence_is_243_l575_575334

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l575_575334


namespace surface_area_unchanged_l575_575313

theorem surface_area_unchanged (l w h : ℝ) (hl : l > 1) (hw : w > 1) (hh : h > 1) :
  let original_surface_area := 2 * (l * w + l * h + w * h)
  let new_surface_area := original_surface_area -- (3 faces removed + 3 new faces = net 0 change)
  original_surface_area = new_surface_area :=
by
  -- definitions for original surface area and new surface area
  let original_surface_area := 2 * (l * w + l * h + w * h)
  let new_face_area := 3 * (1 * 1)
  let removed_face_area := 3 * (1 * 1)
  let new_surface_area := original_surface_area - removed_face_area + new_face_area
  -- proof that the surface area is the same
  have h : original_surface_area = new_surface_area := by
    simp [original_surface_area, new_surface_area, new_face_area, removed_face_area]
    sorry -- steps to show simplification if necessary
  exact h

end surface_area_unchanged_l575_575313


namespace trapezoid_length_properties_l575_575541

-- Definition for the problem
def trapezoid_EFGH (EF GH EH FG : ℝ) := EFGH EFGH.mk'' (
  sorry /-
    The definitions and properties of the given "*conditions*" should be considered.
    For instance:
      * EF_parallel_GH: EF \parallel GH
      * EG_and_GH_have_equal_length: EG = GH = 39
      * EH_perpendicular_FG: EH \perp FG
    are its given properties.
  -/
)

-- Definitions for points
structure Point :=
  (x y : ℝ)

def point_I (E G F H : Point) : Point :=
sorry -- Intersection of diagonals EG and FH, specifics would be problem-dependent

def point_J (F H : Point) : Point :=
sorry -- Midpoint of FH, specifics would be problem-dependent

-- Define a noncomputable version since we are dealing with real numbers
noncomputable def length (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Declaration of given propositions as definitions
def IJ_length (I J : Point) : Prop :=
  length I J = 9

def EH_length (E H : Point) (p q : ℕ) : Prop :=
  length E H = p * real.sqrt q

-- The main theorem statement with given conditions
theorem trapezoid_length_properties
  (E F G H I J : Point)
  (EF GH EH FG : ℝ)
  (p q : ℕ) :
  trapezoid_EFGH EF GH EH FG →
  (IJ_length I J) →
  (EH_length E H p q) →
  p = 8 ∧ q = 198 :=
sorry

end trapezoid_length_properties_l575_575541


namespace α_in_quadrants_l575_575478

def α (k : ℤ) : ℝ := k * 180 + 45

theorem α_in_quadrants (k : ℤ) : 
  (0 ≤ α k ∧ α k < 90) ∨ (180 < α k ∧ α k ≤ 270) :=
sorry

end α_in_quadrants_l575_575478


namespace greatest_int_less_than_neg_17_div_3_l575_575702

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end greatest_int_less_than_neg_17_div_3_l575_575702


namespace find_m_find_line_eq_l575_575567

open Real

section geometry_problem

variables {P Q O : Type} [plane_group O P Q]: P Q
variables (x y b m : ℝ)
variables (P Q : point) (curve : ℝ → ℝ → Prop)
variables (line : ℝ → ℝ → ℝ → Prop)

/-- The curve equation -/
def curve_eq (x y : ℝ) :=
  (x + 1)^2 + (y - 3)^2 = 9

/-- The symmetry line -/
def symmetry_line (x y m : ℝ) :=
  x + m * y + 4 = 0

/-- Condition on the dot product of OP and OQ -/
def orthogonality_condition (x1 y1 x2 y2 : ℝ) :=
  x1 * x2 + y1 * y2 = 0

/-- Problem statement (1): Find m -/
theorem find_m (hm : ∃ P Q, curve_eq P.x P.y ∧ curve_eq Q.x Q.y ∧ symmetry_line P.x P.y m ∧ symmetry_line Q.x Q.y m ∧ orthogonality_condition P.x P.y Q.x Q.y) :
   m = -1 :=
sorry

/-- Problem statement (2): Find the equation of the line PQ -/
theorem find_line_eq (m_eq : m = -1) :
  ∃ b : ℝ, ∀ (P Q : Type) [plane_group O P Q],
  line P Q = sorry :=
sorry

end geometry_problem

end find_m_find_line_eq_l575_575567


namespace find_angle_A_min_area_triangle_l575_575559

variables {a b c A B C : ℝ}
variables (triangle_inequality : A ≤ B)
variables (angle_bisector : b ≠ 0 ∧ c ≠ 0 ∧ AD = 1)
variables (law_of_cosines : c + b * cos (2 * A) = 2 * a * cos A * cos B)

-- Part 1: Prove A = π / 3
theorem find_angle_A : A = π / 3 :=
  sorry

-- Part 2: Prove the minimum area of triangle is sqrt(3) / 3
theorem min_area_triangle (AD : ℝ) (hAD : AD = 1) : 
  let S := (sqrt 3 / 4) * b * c in
  S >= sqrt 3 / 3 :=
  sorry

end find_angle_A_min_area_triangle_l575_575559


namespace hyperbola_focus_l575_575852

theorem hyperbola_focus :
  let a := real.sqrt 19,
      b := real.sqrt (19 / 3),
      c := real.sqrt (a^2 + b^2),
      h := 4,
      k := -3,
      focus1 := (h + c, k) in
  focus1 = (4 + real.sqrt (76 / 3), -3) :=
by
  let a := real.sqrt 19
  let b := real.sqrt (19 / 3)
  let c := real.sqrt (a^2 + b^2)
  let h := 4
  let k := -3
  let focus1 := (h + c, k)
  show focus1 = (4 + real.sqrt (76 / 3), -3)
  sorry

end hyperbola_focus_l575_575852


namespace binom_12_6_l575_575799

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575799


namespace distribute_ice_cream_l575_575241

theorem distribute_ice_cream (n m : ℕ) (h_n : n = 11) (h_m : m = 143) : m / n = 13 := by
  rw [h_n, h_m]
  norm_num
  sorry

end distribute_ice_cream_l575_575241


namespace greatest_possible_third_term_l575_575669

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l575_575669


namespace count_exquisite_polynomials_l575_575553

theorem count_exquisite_polynomials (p : ℕ) (h_prime : p.prime) (h_gt_3 : p > 3) (d : ℕ) :
  let F_p := Finset.range p
  let S_d := {P : polynomial (polynomial F_p) // P.degree <= d ∧ ∀ x y, P.eval x y = P.eval y (-x - y)} in
  |S_d| = p ^ ⌈(d + 1) * (d + 2) / 6⌉ :=
sorry

end count_exquisite_polynomials_l575_575553


namespace binom_12_6_l575_575800

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575800


namespace electricity_bill_september_l575_575278

-- Definitions of the tiered pricing system
def price_first_tier : ℕ → ℝ := λ deg, if deg ≤ 200 then deg * 0.5 else 100
def price_second_tier : ℕ → ℝ := λ deg, if deg ≤ 400 then (deg - 200) * 0.6 else 120
def price_third_tier : ℕ → ℝ := λ deg, if deg > 400 then (deg - 400) * 0.8 else 0

-- The total price function considering tiered prices
def total_price (deg : ℕ) : ℝ :=
  price_first_tier deg + price_second_tier deg + price_third_tier deg

-- The proof statement that should be proven
theorem electricity_bill_september (deg : ℕ) (h : deg = 420) :
  total_price deg = 236 := by
  sorry

end electricity_bill_september_l575_575278


namespace polyhedron_surface_area_l575_575224

-- Define that the polyhedron has three views as a condition
def polyhedron_views : Type := sorry  -- placeholder type for the views of the polyhedron

-- States the theorem to prove the surface area, given these views
theorem polyhedron_surface_area (views : polyhedron_views) : surface_area views = 8 := sorry

end polyhedron_surface_area_l575_575224


namespace largest_value_of_a_l575_575123

theorem largest_value_of_a
  (a b c d e : ℕ)
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : e = d - 10)
  (h5 : e < 105) :
  a ≤ 6824 :=
by {
  -- Proof omitted
  sorry
}

end largest_value_of_a_l575_575123


namespace mark_charged_more_hours_l575_575151

theorem mark_charged_more_hours (P K M : ℕ) 
  (h1 : P + K + M = 135)
  (h2 : P = 2 * K)
  (h3 : P = M / 3) :
  M - K = 75 := by {

sorry
}

end mark_charged_more_hours_l575_575151


namespace binom_12_6_eq_924_l575_575810

theorem binom_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binom_12_6_eq_924_l575_575810


namespace checkerboard_disc_coverage_l575_575276

-- Define the context of the problem.
def D : ℝ := sorry -- Define D as a real number.

theorem checkerboard_disc_coverage :
  ∃ n, n = 16 ∧ (∀ (x y : ℝ), x^2 + y^2 ≤ (D / 2)^2 → (0 ≤ x) ∧ (x <= 6 * D) ∧ (0 ≤ y) ∧ (y <= 6 * D)) :=
begin
  -- Provide proof details here.
  sorry
end

end checkerboard_disc_coverage_l575_575276


namespace LucyJumps_l575_575577

theorem LucyJumps (jump_per_second : ℕ) (seconds_per_minute : ℕ) (songs : ℕ) (length_per_song_minutes : ℕ → ℚ) (total_jumps : ℕ) :
  jump_per_second = 1 →
  seconds_per_minute = 60 →
  songs = 10 →
  (∀ (n : ℕ), length_per_song_minutes n = 3.5) →
  total_jumps = songs * (length_per_song_minutes 1 * seconds_per_minute * jump_per_second).toNat →
  total_jumps = 2100 :=
by
  intros h_jump_per_second h_seconds_per_minute h_songs h_length_per_song h_total_jumps
  rw [h_jump_per_second, h_seconds_per_minute, h_songs, h_length_per_song (1 : ℕ)] at h_total_jumps
  exact h_total_jumps

end LucyJumps_l575_575577


namespace KP_perp_AE_l575_575983

-- Definitions of the points and lines
variables (A B C O D P T L K E : Point)
variables (circle_ABC : Circle)
variables (hO : Circumcenter O A B C)
variables (hAD_diameter : Diameter D O)
variables (hAP_perp_BC : Perpendicular A P B C)
variables (hBP_intersect_circle : SecondIntersection B P T circle_ABC)
variables (hCP_intersect_circle : SecondIntersection C P L circle_ABC)
variables (hDK_parallel_BC : Parallel D K B C)
variables (hDK_intersect_LT : Intersect D K L T)

-- The main theorem to prove
theorem KP_perp_AE : Perpendicular K P A E := 
by sorry

end KP_perp_AE_l575_575983


namespace polynomial_real_roots_l575_575372

noncomputable def desired_c : ℝ := 7.56

theorem polynomial_real_roots : 
  ∃ (c : ℝ), 
  (∀ (z : ℂ), polynomial.eval z (polynomial.X ^ 3 - 4 * polynomial.X ^ 2 + c * polynomial.X - 4) = 0 → z.im = 0) 
  ∧ (c = desired_c) :=
by
  sorry

end polynomial_real_roots_l575_575372


namespace trip_duration_with_stop_l575_575225

-- Define the initial conditions
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time
def time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Initial trip conditions
def initial_time := 4.5
def initial_speed := 70
def stop_time := 0.5
def new_speed := 60

-- The proof problem statement
theorem trip_duration_with_stop : 
  let dist := distance initial_speed initial_time in
  time dist new_speed + stop_time = 5.75 :=
by
  let dist := distance initial_speed initial_time
  have : dist = 315 := by
    rw [distance, mul_comm]
    norm_num
  have : time dist new_speed = 5.25 := by
    rw [time, this]
    norm_num
  have : time dist new_speed + stop_time = 5.75 := by
    rw [this]
    norm_num
  exact this

end trip_duration_with_stop_l575_575225


namespace average_of_natural_numbers_l575_575487

theorem average_of_natural_numbers (a1 an : ℕ) (h1 : a1 = 12) (h2 : an = 53) :
  (a1 + an) / 2 = 32.5 := by
  sorry

end average_of_natural_numbers_l575_575487


namespace avg_weight_of_a_b_c_l575_575639

theorem avg_weight_of_a_b_c (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by
  sorry

end avg_weight_of_a_b_c_l575_575639


namespace polygon_sides_eq_seven_l575_575079

theorem polygon_sides_eq_seven (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) → n = 7 :=
by
  sorry

end polygon_sides_eq_seven_l575_575079


namespace retail_price_percentage_l575_575747

variable (P : ℝ)
variable (wholesale_cost : ℝ)
variable (employee_price : ℝ)

axiom wholesale_cost_def : wholesale_cost = 200
axiom employee_price_def : employee_price = 192
axiom employee_discount_def : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))

theorem retail_price_percentage (P : ℝ) (wholesale_cost : ℝ) (employee_price : ℝ)
    (H1 : wholesale_cost = 200)
    (H2 : employee_price = 192)
    (H3 : employee_price = 0.80 * (wholesale_cost + (P / 100 * wholesale_cost))) :
    P = 20 :=
  sorry

end retail_price_percentage_l575_575747


namespace no_valid_distribution_l575_575850

-- Define the problem domain
def Cube (α : Type) := (faces : List (List (List α)))
def Shape := Cross | Triangle | Circle

-- Define the vertices condition requirements
def vertex_condition [DecidableEq α] (cube : Cube Shape) : Prop :=
  ∀ v : ℕ, 
  v < 8 → -- If v is within the 8 vertices in the cube
  (crosses v + triangles v + circles v = 3) -- There's one cross, one triangle, and one circle at each vertex

-- Define the possible distributions
def distribution_option (crosses triangles : Nat) := 
(crosses, triangles)

-- Candidate options to check
def optionA := distribution_option 6 8
def optionB := distribution_option 7 8
def optionC := distribution_option 5 8
def optionD := distribution_option 7 7

-- Proof that given distributions are not valid 
theorem no_valid_distribution : 
¬(vertex_condition Cube) optionA ∧ 
¬(vertex_condition Cube) optionB ∧ 
¬(vertex_condition Cube) optionC ∧ 
¬(vertex_condition Cube) optionD := by
  sorry

end no_valid_distribution_l575_575850


namespace not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l575_575039

theorem not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C (A B C : ℝ) (h1 : A = 2 * C) (h2 : B = 2 * C) (h3 : A + B + C = 180) : A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := 
by 
  sorry

end not_right_triangle_if_angle_A_eq_angle_B_eq_2_angle_C_l575_575039


namespace count_multiples_of_4_between_100_and_350_l575_575944

theorem count_multiples_of_4_between_100_and_350 : 
  (∃ n : ℕ, 104 + (n - 1) * 4 = 348) ∧ (∀ k : ℕ, (104 + k * 4 ∈ set.Icc 100 350) ↔ (k ≤ 61)) → 
  n = 62 :=
by
  sorry

end count_multiples_of_4_between_100_and_350_l575_575944


namespace calc_product_eq_243_l575_575331

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calc_product_eq_243_l575_575331


namespace niva_overtakes_toyota_l575_575640

theorem niva_overtakes_toyota :
  ∀ (S : ℝ), 
  let niva_dirt_speed := 80, niva_asphalt_speed := 90,
      toyota_dirt_speed := 40, toyota_asphalt_speed := 120,
      niva_lap_time := S / niva_dirt_speed + 3 * S / niva_asphalt_speed,
      toyota_lap_time := S / toyota_dirt_speed + 3 * S / toyota_asphalt_speed in
  ∃ (n : ℕ) (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1), 
    n = 10 ∧ x = 1 / 3 → n + 1 = 11 := sorry

end niva_overtakes_toyota_l575_575640


namespace binomial_12_6_eq_924_l575_575781

noncomputable def binomial (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem binomial_12_6_eq_924 : binomial 12 6 = 924 :=
by
  sorry

end binomial_12_6_eq_924_l575_575781


namespace negative_column_exists_l575_575687

theorem negative_column_exists
  (table : Fin 1999 → Fin 2001 → ℤ)
  (H : ∀ i : Fin 1999, (∏ j : Fin 2001, table i j) < 0) :
  ∃ j : Fin 2001, (∏ i : Fin 1999, table i j) < 0 :=
sorry

end negative_column_exists_l575_575687


namespace max_garden_area_20000_l575_575106

/-
  Jennifer wants to fence a rectangular garden next to her house, using the house 
  as one of the longer sides of the rectangle. She has 400 feet of fencing available 
  to cover the two shorter sides and the side opposite the house. 
  Prove that the maximum area of the garden is 20000 square feet.
-/

def maximum_garden_area (l w : ℝ) (fencing : ℝ) : ℝ :=
  if l + 2 * w = fencing then l * w else 0

theorem max_garden_area_20000 (w : ℝ) (h : 0 ≤ w ∧ 400 - 2 * w ≥ 0) :
  maximum_garden_area (400 - 2 * w) w 400 = 20000 :=
sorry

end max_garden_area_20000_l575_575106


namespace roots_of_quadratic_l575_575645

theorem roots_of_quadratic (b c : ℝ) (h_eq : ∀ x, (2 / real.sqrt 3) * x^2 + b * x + c = 0) 
  (h_geom : ∃ K L M : ℝ × ℝ, L.1 = K.1 + 1 ∧ L.2 = K.2 + 2 ∧ ∠ K L M = 120) : 
  ∃ p q : ℝ, p = 0.5 ∧ q = 1.5 ∧ (∂ p ≠ q) := 
sorry

end roots_of_quadratic_l575_575645


namespace find_a_of_inequality_solution_l575_575030

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -3 < ax - 2 ∧ ax - 2 < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 := by
  sorry

end find_a_of_inequality_solution_l575_575030


namespace binom_12_6_l575_575801

theorem binom_12_6 : Nat.choose 12 6 = 924 := by sorry

end binom_12_6_l575_575801


namespace zeros_f_range_of_a_l575_575043

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - a
noncomputable def g (x : ℝ) : ℝ := (1 / 3) * x^3 - 2 * x^2 + 3 * x + (16 / 3)

theorem zeros_f (a : ℝ) :
  (a < 0 ∨ a = 1 → (∃! x, f x a = 0)) ∧
  (0 ≤ a ∧ a < 1 → ¬∃ x, f x a = 0) ∧
  (a > 1 → (∃ x1 x2, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 2, ∃ x2 ∈ set.Icc (-1 : ℝ) 2, f x1 a ≥ g x2) ↔ a ∈ set.Iic 1 :=
sorry

end zeros_f_range_of_a_l575_575043


namespace circle_construction_l575_575374

-- Definitions
variables {P O : Type} {L : set (P × O)} {R : ℝ}

-- The statement of the theorem
theorem circle_construction (P : Type) (L : set (P × P)) (R : ℝ) :
  ∃ O : P, dist O L = R ∧ dist O P = R := sorry

end circle_construction_l575_575374


namespace P_2_lt_X_lt_4_l575_575029

-- Given definitions
variable (X : ℝ)
variable (σ : ℝ) (hσ : σ > 0)
def normal_dist := MeasureTheory.Normal 2 σ^2
def X_is_normal : MeasureTheory.AEStronglyMeasurable (MeasureTheory.RealMeasLe normal_dist) := sorry
def P_X_lt_0 : ℝ := MeasureTheory.Probability (MeasureTheory.RealMeasLe normal_dist {x | x < 0})
#eval assert (P_X_lt_0 = 0.1)

-- Prove statement
theorem P_2_lt_X_lt_4 : MeasureTheory.Probability (MeasureTheory.RealMeasLe normal_dist {x | 2 < x ∧ x < 4}) = 0.4 :=
by sorry

end P_2_lt_X_lt_4_l575_575029


namespace banknotes_sum_divisible_by_101_l575_575677

theorem banknotes_sum_divisible_by_101 (a b : ℕ) (h₀ : a ≠ b % 101) : 
  ∃ (m n : ℕ), m + n = 100 ∧ ∃ k l : ℕ, k ≤ m ∧ l ≤ n ∧ (k * a + l * b) % 101 = 0 :=
sorry

end banknotes_sum_divisible_by_101_l575_575677


namespace percentage_basketballs_lucien_l575_575136

-- Definitions derived from the problem conditions
def total_balls_lucca : ℕ := 100
def percent_basketballs_lucca : ℝ := 10 / 100
def basketballs_lucca : ℕ := total_balls_lucca * percent_basketballs_lucca
def total_balls_lucien : ℕ := 200
def total_basketballs : ℕ := 50
def basketballs_lucien : ℕ := total_basketballs - basketballs_lucca

-- Prove that the percentage of Lucien's balls that are basketballs is 20%
theorem percentage_basketballs_lucien : 
  (basketballs_lucien : ℝ) / total_balls_lucien * 100 = 20 := 
by {
  -- sorry to indicate objective of proof
  sorry
}

end percentage_basketballs_lucien_l575_575136


namespace distance_to_building_materials_l575_575691

theorem distance_to_building_materials (D : ℝ) 
  (h1 : 2 * 10 * 4 * D = 8000) : 
  D = 100 := 
by
  sorry

end distance_to_building_materials_l575_575691


namespace intersect_eq_l575_575555

variable (M N : Set Int)
def M_def : Set Int := { m | -3 < m ∧ m < 2 }
def N_def : Set Int := { n | -1 ≤ n ∧ n ≤ 3 }

theorem intersect_eq : M_def ∩ N_def = { -1, 0, 1 } := by
  sorry

end intersect_eq_l575_575555


namespace _l575_575568

noncomputable def tangency_theorem : ∀ (O O1 O2 : Circle) (A B C P Q : Point),
  (A ≠ B) →
  (P = midpoint A B) →
  (O1.is_tangent_Line_at P (Line[AB])) →
  (O1.is_tangent_to_circle O) →
  (C ∈ intersection_of (tangent_at_point O1 (Line[A])) O) →
  (Q = midpoint B C) →
  (O2.is_tangent_Line_at Q (Line[BC])) →
  (O2.is_tangent_to_segment A C) →
  (O2.is_tangent_to_circle O) :=
sorry

end _l575_575568


namespace sum_of_sequence_2017_l575_575890

def sequence_a : ℕ → ℤ
| 0 := 0  -- Define sequence starting with index 1
| 1 := 1
| 2 := 3
| (n+2) := sequence_a (n+1) - sequence_a n

noncomputable def sum_first_n_terms (n : ℕ) : ℤ :=
(nat.antidiagonal n).sum (λ p, sequence_a p.1)

theorem sum_of_sequence_2017 :
  sum_first_n_terms 2017 = 1 :=
sorry

end sum_of_sequence_2017_l575_575890


namespace number_of_therapy_hours_l575_575274

theorem number_of_therapy_hours (A F H : ℝ) (h1 : F = A + 35) 
  (h2 : F + (H - 1) * A = 350) (h3 : F + A = 161) :
  H = 5 :=
by
  sorry

end number_of_therapy_hours_l575_575274


namespace binom_8_5_eq_56_l575_575348

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l575_575348


namespace binomial_12_6_eq_924_l575_575834

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := by
  sorry

end binomial_12_6_eq_924_l575_575834


namespace intersection_points_at_most_one_l575_575956

open Classical

noncomputable def skew_lines (a b : Line) : Prop :=
  ¬ (a = b) ∧ ∃ p, p ∈ a ∧ p ∉ b ∧ ∃ q, q ∈ b ∧ q ∉ a ∧ ∀ r ∈ a, ∀ s ∈ b, r ≠ s

noncomputable def common_perpendicular (a b c : Line) : Prop :=
  ¬ (a = b) ∧ skew_lines a b ∧ ∀ p ∈ c, p ⊥ a ∧ p ⊥ b

noncomputable def parallel (a b : Line) : Prop :=
  ∃ v, ∀ (p ∈ a) (q ∈ b), q = p + v

theorem intersection_points_at_most_one 
  (a b l : Line) 
  (h_skew : skew_lines a b) 
  (h_perpendicular : ∃ AB, common_perpendicular a b AB) 
  (h_parallel : ∃ AB, parallel l AB) : 
  ∃ n, n ≤ 1 ∧ 
    (∀ p, p ∈ l → (p ∈ a ∨ p ∈ b) → ∃ q, (q ∈ a ∨ q ∈ b) ∧ p = q) → 
    cardinal #(p | p ∈ l ∧ (p ∈ a ∨ p ∈ b)) = n :=
sorry

end intersection_points_at_most_one_l575_575956


namespace trapezium_area_l575_575868

def find_trapezium_area (a b h : ℝ) : ℝ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area :
  find_trapezium_area 20 18 11 = 209 :=
by
  sorry

end trapezium_area_l575_575868


namespace tan_half_angle_l575_575021

variables {α : ℝ}

theorem tan_half_angle (h1 : sin α + cos α = -3 / real.sqrt 5) (h2 : |sin α| > |cos α|) : 
  tan (α / 2) = -(real.sqrt 5 + 1) / 2 :=
sorry

end tan_half_angle_l575_575021


namespace binary_to_decimal_110011_l575_575386

theorem binary_to_decimal_110011 : (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 51 :=
by
  -- Explicit computation for clarity in the assertion
  have h₁ : 1 * 2^5 = 32 := by norm_num
  have h₂ : 1 * 2^4 = 16 := by norm_num
  have h₃ : 0 * 2^3 = 0 := by norm_num
  have h₄ : 0 * 2^2 = 0 := by norm_num
  have h₅ : 1 * 2^1 = 2 := by norm_num
  have h₆ : 1 * 2^0 = 1 := by norm_num
  calc
    (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)
        = (32 + 16 + 0 + 0 + 2 + 1) : by rw [h₁, h₂, h₃, h₄, h₅, h₆]
    ... = 51 : by norm_num

end binary_to_decimal_110011_l575_575386


namespace intersection_A_B_l575_575460

open Set

noncomputable def A : Set ℝ := { x | 1 / 2 ≤ 2^x ∧ 2^x ≤ 4 }
def B : Set ℕ := {0, 1, 2, 3}
def A_cap_B : Set ℝ := { x | x ∈ A ∧ x ∈ B}

theorem intersection_A_B : A_cap_B = {0, 1, 2} := sorry

end intersection_A_B_l575_575460


namespace tangent_line_eq_l575_575675

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - 2 * x - 1)

theorem tangent_line_eq :
  let x := 1
  let y := f x
  ∃ (m : ℝ), m = -2 * Real.exp 1 ∧ (∀ (x y : ℝ), y = m * (x - 1) + f 1) :=
by
  sorry

end tangent_line_eq_l575_575675


namespace coin_piles_not_lighter_l575_575237

open Nat List

theorem coin_piles_not_lighter 
    (a b : List ℕ) (x : ℕ) 
    (h_sum : a.sum = b.sum)
    (h_cond : ∀ k : ℕ, k ≤ min a.length b.length → a.take k.sum ≤ b.take k.sum)
    (h_sorted_a : a.sorted (· ≥ ·))
    (h_sorted_b : b.sorted (· ≥ ·)) :
    (a.map (λ y => min y x)).sum ≥ (b.map (λ y => min y x)).sum := 
sorry

end coin_piles_not_lighter_l575_575237


namespace max_area_triangle_l575_575516

-- Definitions and conditions
variables (a c A B : ℝ)
variables (a_plus_c_eq_6 : a + c = 6)
variables (cos_a : ℝ)
variables (sin_a : ℝ)
variables (tan_half_b : ℝ)
variables (sin_a_cos_half_b_eq : (3 - cos_a) * tan_half_b = sin_a)

-- The maximum area of triangle ABC
theorem max_area_triangle : 
  ∃ (S_max : ℝ), S_max = 2 * Real.sqrt 2 ∧
                  ∀ (a c A B : ℝ),
                    (a + c = 6) → 
                    ((3 - cos A) * tan (B / 2) = sin A) →
                    let b := 2 in
                    let S := (1/2) * a * c * sin B in
                    S ≤ 2 * Real.sqrt 2 :=
begin
  sorry
end

end max_area_triangle_l575_575516


namespace Sergey_wins_with_optimal_play_l575_575587

theorem Sergey_wins_with_optimal_play :
  ∀ n : ℕ, (n ≥ 9) →
  ∃ sequence : list ℕ, 
    (sequence.length = 9) ∧
    (∀ i: ℕ, i ∈ sequence → (1 ≤ i ∧ i ≤ 9)) ∧
    (∃ Oleg_start : ℕ, Oleg_start = sequence.head ∧
     ∃ Sergey_last_digit : ℕ, Sergey_last_digit = sequence.reverse.head) ∧
    ¬ (sequence.reverse.head :: sequence.reverse.tail.head :: list.nil).foldr (λ a b, a + 10 * b) 0 % 4 = 0 :=
by
  sorry

end Sergey_wins_with_optimal_play_l575_575587


namespace total_money_is_220_l575_575628

-- Define the amounts on Table A, B, and C
def tableA := 40
def tableC := tableA + 20
def tableB := 2 * tableC

-- Define the total amount of money on all tables
def total_money := tableA + tableB + tableC

-- The main theorem to prove
theorem total_money_is_220 : total_money = 220 :=
by
  sorry

end total_money_is_220_l575_575628
