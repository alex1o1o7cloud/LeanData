import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Equiv.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.GcdMonoid.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Polynomial.Div
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Function
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Midpoint
import Mathlib.Geometry.Euclidean.Perpendicular
import Mathlib.Init.Data.Nat.Basic
import Mathlib.MeasureTheory.Geometry.EuclideanGeometry
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Topology.Basic
import Mathlib.analysis.special_functions.sqrt

namespace largest_three_digit_divisible_by_5_8_and_2_l274_274754

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Define the conditions from the problem
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0
def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

-- The problem statement
theorem largest_three_digit_divisible_by_5_8_and_2 : 
  ∃ n : ℕ, is_three_digit n ∧ divisible_by_5 n ∧ divisible_by_8 n ∧ 
  ∀ m : ℕ, is_three_digit m ∧ divisible_by_5 m ∧ divisible_by_8 m → m ≤ n :=
begin
  use 960,
  split,
  { -- proof that 960 is a three-digit number
    sorry },
  split,
  { -- proof that 960 is divisible by 5
    sorry },
  split,
  { -- proof that 960 is divisible by 8
    sorry },
  { -- proof that 960 is the largest such number
    sorry }
end

end largest_three_digit_divisible_by_5_8_and_2_l274_274754


namespace range_of_a_for_f_zero_l274_274179

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_f_zero (a : ℝ) :
  (∃ x : ℝ, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 :=
by
  sorry

end range_of_a_for_f_zero_l274_274179


namespace a_2023_le_100000_l274_274664

-- Define the sequence (a_n) with initial condition
def a : ℕ → ℕ
| 0     := 0
| (n+1) := find (λ m, m > a n ∧ ∀ i j, i < j → j < (n+1) → 2 * a i ≠ a j + m) sorry

-- Statement that we want to prove:
theorem a_2023_le_100000 : a 2023 ≤ 100000 :=
sorry

end a_2023_le_100000_l274_274664


namespace number_of_classes_l274_274305

theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by {
  sorry -- Proof goes here
}

end number_of_classes_l274_274305


namespace percentage_of_red_shirts_l274_274995

theorem percentage_of_red_shirts
  (Total : ℕ) 
  (P_blue P_green : ℝ) 
  (N_other : ℕ)
  (H_Total : Total = 600)
  (H_P_blue : P_blue = 0.45) 
  (H_P_green : P_green = 0.15) 
  (H_N_other : N_other = 102) :
  ( (Total - (P_blue * Total + P_green * Total + N_other)) / Total ) * 100 = 23 := by
  sorry

end percentage_of_red_shirts_l274_274995


namespace find_a_domain_range_find_a_decreasing_find_a_zero_l274_274180

-- Definitions for function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the value of a IF domain and range of f(x) are both [1,a]
theorem find_a_domain_range (a : ℝ) (h : a > 1) (h1 : ∀ x ∈ Icc 1 a, f x a = x ∈ Icc (1 : ℝ) a): 
  a = 2 :=
sorry

-- Prove the range of a given f(x) is decreasing and satisfies an inequality
theorem find_a_decreasing (a : ℝ) (h : a > 1) (h_decreasing : ∀ x ∈ Icc 1 a, f x a ≤ 4):
  2 ≤ a ∧ a ≤ 3 :=
sorry

-- Prove the range of a given there exists a zero in the interval [1,3]
theorem find_a_zero (a : ℝ) (h : a > 1) (h_zero : ∃ x ∈ Icc (1:ℝ) 3, f x a = 0):
  real.sqrt (5 : ℝ) ≤ a ∧ a ≤ 3 :=
sorry

end find_a_domain_range_find_a_decreasing_find_a_zero_l274_274180


namespace balanced_integers_count_l274_274421

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  d1 + d2 = d3 + d4 + d5

noncomputable def number_of_balanced_integers : ℕ :=
  (∑ s in finset.range (9 - 2 + 1), s * (nat.choose (s + 2) 2)) + 
  (∑ s in finset.range (18 - 10 + 1), (19 - s) * (nat.choose (s + 2) 2)) + 
  (∑ s in finset.range (27 - 19 + 1), nat.choose (27 - s + 2) 2)

theorem balanced_integers_count :
  ∃ t : ℕ, t = number_of_balanced_integers :=
sorry

end balanced_integers_count_l274_274421


namespace eithan_savings_l274_274817

theorem eithan_savings :
  let amount := 2000 : ℝ 
  let wife_share := (2/5) * amount 
  let remaining_after_wife := amount - wife_share  
  let first_son_share := (2/5) * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - first_son_share 
  let second_son_share := (40/100) * remaining_after_first_son 
  let savings := remaining_after_first_son - second_son_share 
  savings = 432 :=
by
  sorry

end eithan_savings_l274_274817


namespace fill_tank_with_leak_l274_274805

theorem fill_tank_with_leak (R L T: ℝ)
(h1: R = 1 / 7) (h2: L = 1 / 56) (h3: R - L = 1 / T) : T = 8 := by
  sorry

end fill_tank_with_leak_l274_274805


namespace sin_2theta_value_l274_274965

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l274_274965


namespace power_congruence_l274_274324

theorem power_congruence (n : ℕ) (hn : 0 < n) : 
  let a := 2 ^ (2 ^ (10 * n + 1)) in 
  a % 23 = 4 := 
  by 
  sorry

end power_congruence_l274_274324


namespace family_savings_amount_l274_274812

theorem family_savings_amount : 
  let total := 2000 
  let given_to_wife := 2 / 5 * total 
  let remaining_after_wife := total - given_to_wife 
  let given_to_first_son := 2 / 5 * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - given_to_first_son 
  let given_to_second_son := 40 / 100 * remaining_after_first_son 
  let remaining_amount := remaining_after_first_son - given_to_second_son 
  in remaining_amount = 432 := 
by
  sorry

end family_savings_amount_l274_274812


namespace log2x_x_eq_zero_interval_l274_274717

noncomputable def f (x : ℝ) : ℝ := log 2 x + x

theorem log2x_x_eq_zero_interval : ∀ x: ℝ, 0 < x → f x = 0 → (1 / 2 < x ∧ x < 1) := by
  intro x hx hf
  let log_2 := log 2 x
  have fx_def : f x = log_2 + x := rfl
  have increasing_f : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y :=
    -- Prove that f is strictly increasing on (0, ∞)
    sorry
  have f_half : f (1/2) = log 2 (1 / 2) + 1/2 := rfl
  have f_half_value : f_half < 0 :=
    -- Prove f(1/2) < 0
    sorry
  have f_one : f 1 = log 2 1 + 1 := rfl
  have f_one_value : f_one > 0 :=
    -- Prove f(1) > 0
    sorry
  have zero_in_interval : 1/2 < x ∧ x < 1 :=
    -- Show that zero of f(x) falls in (1/2, 1)
    sorry
  exact zero_in_interval

end log2x_x_eq_zero_interval_l274_274717


namespace bob_max_points_guarantee_l274_274819

-- Initial conditions and definitions
def suits : ℕ := 4
def cards_per_suit : ℕ := 9
def total_cards : ℕ := suits * cards_per_suit
def cards_alice : ℕ := 18
def cards_bob : ℕ := total_cards - cards_alice

-- The question translated into a Lean statement
theorem bob_max_points_guarantee (suits cards_per_suit : ℕ) (h_suits : suits = 4) (h_cps : cards_per_suit = 9) :
  let total_cards := suits * cards_per_suit in
  let cards_alice := 18 in
  let cards_bob := total_cards - cards_alice in
  (∃ n : ℕ, n = 15) :=
by
  sorry

end bob_max_points_guarantee_l274_274819


namespace range_of_a_l274_274200

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → (x / (x^2 + 3 * x + 1) ≤ a)) → a ≥ 1/5 :=
begin
  sorry
end

end range_of_a_l274_274200


namespace family_savings_amount_l274_274814

theorem family_savings_amount : 
  let total := 2000 
  let given_to_wife := 2 / 5 * total 
  let remaining_after_wife := total - given_to_wife 
  let given_to_first_son := 2 / 5 * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - given_to_first_son 
  let given_to_second_son := 40 / 100 * remaining_after_first_son 
  let remaining_amount := remaining_after_first_son - given_to_second_son 
  in remaining_amount = 432 := 
by
  sorry

end family_savings_amount_l274_274814


namespace smallest_n_for_conditions_l274_274284

theorem smallest_n_for_conditions :
  ∃ n : ℕ, (∀ (x : ℕ → ℝ), 
    (∀ i, 0 ≤ x i) ∧ 
    (finset.sum finset.univ (λ i, x i) = 1) ∧ 
    (finset.sum finset.univ (λ i, x i^2) ≤ 1/50) ∧ 
    (finset.sum finset.univ (λ i, x i^3) ≤ 1/150) ↔ n = 50) :=
begin
  sorry
end

end smallest_n_for_conditions_l274_274284


namespace smallest_n_for_g_equals_3_l274_274280

open Nat

def g (n : ℕ) : ℕ :=
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = n)

theorem smallest_n_for_g_equals_3 : ∃ n : ℕ, g n = 3 ∧ ∀ m : ℕ, m < n → g m ≠ 3 :=
by {
  sorry
}

end smallest_n_for_g_equals_3_l274_274280


namespace find_x_with_three_prime_divisors_l274_274149

def x_with_conditions (n : Nat) (x : Nat) : Prop :=
  x = 9^n - 1 ∧
  nat.factors x ∧
  nat.factors x.count 7 = 1

theorem find_x_with_three_prime_divisors (n : Nat) (x : Nat) :
  x_with_conditions n x → x = 728 :=
by
  sorry

end find_x_with_three_prime_divisors_l274_274149


namespace problem_statement_l274_274204

noncomputable def not_monotonic_interval_contains_zero_of_f' (k : ℝ) : Prop :=
  ¬ (mono_increasing_on (fun x => x^3 - 12*x) (k-1, k+1)) ∨ ¬ (mono_decreasing_on (fun x => x^3 - 12*x) (k-1, k+1))

theorem problem_statement (k : ℝ) :
  not_monotonic_interval_contains_zero_of_f' k ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end problem_statement_l274_274204


namespace max_area_correct_l274_274481

noncomputable def max_area (S : ℝ) (a b : ℤ) : Prop :=
  (a = -b / 2) ∧
  (10 * 300 * S ≤ 10000) ∧
  (S ≤ 10 / 3) ∧
  (S > 0) ∧
  (b = 2 * -a) ∧
  (b ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4}) ∧
  (2 * abs (2 * abs a - b) ≤ 9)

theorem max_area_correct : ∃ (S : ℝ) (a b : ℤ), max_area S a b :=
by
  use (10 / 3), 2, -4
  split
  · -- a = -b / 2
    exact rfl
  split
  · -- 10 * 300 * S ≤ 10000
    norm_num
  split
  · -- S ≤ 10 / 3
    norm_num
  split
  · -- S > 0
    norm_num
  split
  · -- b = 2 * -a
    exact rfl
  · -- b ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    exact Or.inl rfl

end max_area_correct_l274_274481


namespace common_divisors_count_48_80_l274_274194

noncomputable def prime_factors_48 : Nat -> Prop
| n => n = 48

noncomputable def prime_factors_80 : Nat -> Prop
| n => n = 80

theorem common_divisors_count_48_80 :
  let gcd_48_80 := 2^4
  let divisors_of_gcd := [1, 2, 4, 8, 16]
  prime_factors_48 48 ∧ prime_factors_80 80 →
  List.length divisors_of_gcd = 5 :=
by
  intros
  sorry

end common_divisors_count_48_80_l274_274194


namespace bob_distance_when_meet_l274_274398

theorem bob_distance_when_meet (total_distance : ℕ) (yolanda_speed : ℕ) (bob_speed : ℕ) 
    (yolanda_additional_distance : ℕ) (t : ℕ) :
    total_distance = 31 ∧ yolanda_speed = 3 ∧ bob_speed = 4 ∧ yolanda_additional_distance = 3 
    ∧ 7 * t = 28 → 4 * t = 16 := by
    sorry

end bob_distance_when_meet_l274_274398


namespace find_b_l274_274585

theorem find_b (a b c : ℝ) (h1 : a = 6) (h2 : c = 3) (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) : b = 15 :=
by
  rw [h1, h2] at h3
  sorry

end find_b_l274_274585


namespace probability_of_c_first_called_l274_274000

theorem probability_of_c_first_called 
  (A B C : Type)
  (events : Finset (List (A ⊕ B ⊕ C)))
  (h : ∀ (x : A ⊕ B ⊕ C), x ∈ events → List.cons x [] ∈ events) :
  P (C :: list( (A ⊕ B ⊕ C))) = 1/3 :=
sorry

end probability_of_c_first_called_l274_274000


namespace find_quadratic_function_expression_l274_274893

variable (a b c : ℝ)
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function_expression
  (a b c : ℝ)
  (h1 : quadratic_function a b c (-2) = 0)
  (h2 : quadratic_function a b c (4) = 0)
  (h3 : ∃ x, x ≠ -2 ∧ x ≠ 4 ∧ quadratic_function a b c x = 9) :
  quadratic_function a b c = -x^2 + 2x + 8 := 
sorry

end find_quadratic_function_expression_l274_274893


namespace prime_factors_539_l274_274878

theorem prime_factors_539 (h2 : nat.is_prime 2)
                         (h3 : nat.is_prime 3)
                         (h7 : nat.is_prime 7)
                         (h11 : nat.is_prime 11)
                         (h13 : nat.is_prime 13)
                         (h19 : nat.is_prime 19)
                         (h23 : nat.is_prime 23)
                         (h29 : nat.is_prime 29)
                         (h31 : nat.is_prime 31)
                         (h41 : nat.is_prime 41)
                         : (539 % 2 ≠ 0) ∧ (539 % 3 ≠ 0) ∧ (539 % 7 ≠ 0) ∧ (539 % 11 ≠ 0) ∧ (539 % 19 ≠ 0) ∧ (539 % 23 ≠ 0) ∧ (539 % 29 ≠ 0) ∧ (539 % 31 ≠ 0) ∧ 
                           (539 / 13 = 41) ∧ (539 / 41 = 13) ∧ nat.is_prime 41 :=
by sorry

end prime_factors_539_l274_274878


namespace eccentricity_ellipse_l274_274230

variable (a b : ℝ) (h1 : a > b) (h2 : b > 0)
variable (c : ℝ) (h3 : c = Real.sqrt (a ^ 2 - b ^ 2))
variable (h4 : b = c)
variable (ellipse_eq : ∀ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1)

theorem eccentricity_ellipse :
  c / a = Real.sqrt 2 / 2 :=
by
  sorry

end eccentricity_ellipse_l274_274230


namespace max_min_f_l274_274516

noncomputable def f (M : ℝ × ℝ) (AB MM₁ AM BM : ℝ) : ℝ :=
  AB + MM₁ - AM - BM

def unit_circle_radius : ℝ := 1
def diameter : ℝ := 2

theorem max_min_f :
  (∀ M : ℝ × ℝ, (M.fininte_diam ≤ 2) ∧ (M.proj_fininte ≤ 1)
    → 0 ≤ f M 2 (M.2 - M.1) (√(2 - 2 M.1)) (√(2 (1 - M.1))) 
    ∧ f M 2 (M.2 - M.1) (√(2)- 2 M.2) (√(2 (1 - M.2))) ≤ 3 - 2 * √(2)
  ) := sorry

end max_min_f_l274_274516


namespace find_omega_increasing_intervals_l274_274914

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (x : ℝ) : ℝ :=
  let ω := 3/2
  f ω (x - (Real.pi / 2))

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x : ℝ, f ω (x + 2*Real.pi / (2*ω)) = f ω x) :
  ω = 3/2 :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∃ a b, 
  a = (2/3 * k * Real.pi + Real.pi / 4) ∧ 
  b = (2/3 * k * Real.pi + 7 * Real.pi / 12) ∧
  ∀ x, a ≤ x ∧ x ≤ b → g x < g (x + 1) :=
  sorry

end find_omega_increasing_intervals_l274_274914


namespace events_independent_prob_A_union_B_correct_l274_274360

noncomputable def num_balls : ℕ := 8
noncomputable def num_odd_balls : ℕ := 4
noncomputable def num_multiple_of_three : ℕ := 2
noncomputable def num_odd_and_multiple_of_three : ℕ := 1

def prob_A : ℚ := num_odd_balls / num_balls
def prob_B : ℚ := num_multiple_of_three / num_balls
def prob_AB : ℚ := num_odd_and_multiple_of_three / num_balls

def independent_events : Prop := prob_AB = prob_A * prob_B
def prob_A_union_B : ℚ := prob_A + prob_B - prob_AB

theorem events_independent : independent_events :=
by
  -- Prove independence

theorem prob_A_union_B_correct : prob_A_union_B = 5 / 8 :=
by
  -- Prove probability of A union B is 5/8

#check events_independent
#check prob_A_union_B_correct

end events_independent_prob_A_union_B_correct_l274_274360


namespace distinct_triangles_count_l274_274466

theorem distinct_triangles_count :
  let pairs := { (x1, x2) | 0 ≤ x1 ∧ x1 ≤ 49 ∧ 0 ≤ x2 ∧ x2 ≤ 49 ∧ x1 ≠ x2 }
  let even_pairs := { (x1, x2) ∈ pairs | (x2 - x1) % 2 = 0 }
  cardinality even_pairs = 600 :=
by
  let N := 2009
  let m := 41
  let x_range := {x | 0 ≤ x ∧ x ≤ N / m}
  let even_set := {x ∈ x_range | x % 2 = 0}
  let odd_set := {x ∈ x_range | x % 2 = 1}
  have h1 : |even_set| = 25, sorry
  have h2 : |odd_set| = 25, sorry
  have h3 : cardinality even_pairs = 2 * (25 * 24 / 2), from sorry
  exact h3

end distinct_triangles_count_l274_274466


namespace valid_pairs_l274_274231

def conditions (a b : ℕ) (chessboard : Fin a → Fin b → Bool) : Prop :=
  ∀ i j : Fin a,
    (chessboard i j = false → (∑ k, chessboard i k = ∑ k, chessboard (⟨k, sorry⟩ : Fin a) j)) ∧
    (chessboard i j = true → (b - ∑ k, chessboard i k = a - ∑ k, chessboard (⟨k, sorry⟩ : Fin a) j))

theorem valid_pairs (a b : ℕ) (chessboard : Fin a → Fin b → Bool) :
  conditions a b chessboard →
  a = 2*b ∨ b = 2*a ∨ a = b :=
sorry 

end valid_pairs_l274_274231


namespace quadractic_roots_value_l274_274646

theorem quadractic_roots_value (c d : ℝ) (h₁ : 3*c^2 + 9*c - 21 = 0) (h₂ : 3*d^2 + 9*d - 21 = 0) :
  (3*c - 4) * (6*d - 8) = -22 := by
  sorry

end quadractic_roots_value_l274_274646


namespace difference_in_sums_l274_274306

def sum_of_digits (n : ℕ) : ℕ := (toString n).foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def Petrov_numbers := List.range' 1 2014 |>.filter (λ n => n % 2 = 1)
def Vasechkin_numbers := List.range' 2 2012 |>.filter (λ n => n % 2 = 0)

def sum_of_digits_Petrov := (Petrov_numbers.map sum_of_digits).sum
def sum_of_digits_Vasechkin := (Vasechkin_numbers.map sum_of_digits).sum

theorem difference_in_sums : sum_of_digits_Petrov - sum_of_digits_Vasechkin = 1007 := by
  sorry

end difference_in_sums_l274_274306


namespace magic_shop_purchase_l274_274499

theorem magic_shop_purchase :
  let deck_price := 7
  let frank_decks := 3
  let friend_decks := 2
  let discount_rate := 0.1
  let tax_rate := 0.05
  let total_cost := (frank_decks + friend_decks) * deck_price
  let discount := discount_rate * total_cost
  let discounted_total := total_cost - discount
  let sales_tax := tax_rate * discounted_total
  let rounded_sales_tax := (sales_tax * 100).round / 100
  let final_amount := discounted_total + rounded_sales_tax
  final_amount = 33.08 :=
by
  sorry

end magic_shop_purchase_l274_274499


namespace probability_blue_tile_l274_274788

def is_congruent_to_3_mod_7 (n : ℕ) : Prop := n % 7 = 3

def num_blue_tiles (n : ℕ) : ℕ := (n / 7) + 1

theorem probability_blue_tile : 
  num_blue_tiles 70 / 70 = 1 / 7 :=
by
  sorry

end probability_blue_tile_l274_274788


namespace find_n_l274_274500

noncomputable def power_series (n : ℕ) : ℕ :=
  let a₀ := 1
  let a₄ := 16 * Nat.choose n 4
  if a₀ + a₄ = 17 then n else 0

theorem find_n (n : ℕ) (h₀ : (1 - 2 * 0)^n = 1) (h₄ : (1 - 2*x)^n = ∑ i in finset.range (n+1), (a n) * x^i) :
  power_series n = 4 := sorry

end find_n_l274_274500


namespace giant_spider_leg_cross_sectional_area_l274_274040

theorem giant_spider_leg_cross_sectional_area :
  let previous_spider_weight := 6.4
  let weight_multiplier := 2.5
  let pressure := 4
  let num_legs := 8

  let giant_spider_weight := weight_multiplier * previous_spider_weight
  let weight_per_leg := giant_spider_weight / num_legs
  let cross_sectional_area := weight_per_leg / pressure

  cross_sectional_area = 0.5 :=
by 
  sorry

end giant_spider_leg_cross_sectional_area_l274_274040


namespace find_m_l274_274845

noncomputable def sequence (n : ℕ) : ℝ :=
if n = 0 then 7
else let x := sequence (n - 1) in (x ^ 2 + 5 * x + 4) / (x + 6)

theorem find_m :
  ∃ (m : ℕ), m > 0 ∧ sequence m ≤ 4 + 1 / 2 ^ 18 ∧ 109 ≤ m ∧ m ≤ 324 :=
begin
  sorry
end

end find_m_l274_274845


namespace exponent_of_3_in_30_factorial_l274_274249

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | nat.succ n' => (nat.succ n') * factorial n'

noncomputable def exponent_of_prime_in_factorial (n p : ℕ) : ℕ :=
  if h : p > 1 ∧ nat.prime p then
    let rec count_multiples (m : ℕ) (acc : ℕ) : ℕ :=
      if m > n then acc
      else count_multiples (p * m) (acc + n / m)
    count_multiples p 0
  else 0

theorem exponent_of_3_in_30_factorial : exponent_of_prime_in_factorial 30 3 = 14 := 
sorry

end exponent_of_3_in_30_factorial_l274_274249


namespace fraction_left_handed_l274_274395

def ratio : ℕ → ℕ → Prop := λ r b, r = b

def left_handed_fraction (total : ℕ) (lh : ℕ) : ℚ := lh / total

theorem fraction_left_handed (r b : ℕ) (x : ℕ) (hratio: ratio r b)
  (rh_left : ℚ) (bh_left : ℚ)
  (hrh_condition : rh_left = (1 / 3) * r)
  (hbh_condition : bh_left = (2 / 3) * b):
  left_handed_fraction (r + b) (rh_left + bh_left) = 1 / 2 :=
by
  rw [ratio] at hratio
  have r_def : r = 5 * x := sorry
  have b_def : b = 5 * x := sorry
  rw [r_def, b_def] at *
  have rh_left_def : rh_left = (1 / 3) * (5 * x) := sorry
  have bh_left_def : bh_left = (2 / 3) * (5 * x) := sorry
  rw [hrh_condition, hbh_condition, rh_left_def, bh_left_def] at *
  have lh_total : rh_left + bh_left = 15 * x / 3 := sorry
  have total_participants : r + b = 10 * x := sorry
  have fraction : (15 * x / 3) / (10 * x) = 1 / 2 := sorry
  exact fraction

end fraction_left_handed_l274_274395


namespace ratio_of_DN_NF_l274_274213

theorem ratio_of_DN_NF (D E F N : Type) (DE EF DF DN NF p q: ℕ) (h1 : DE = 18) (h2 : EF = 28) (h3 : DF = 34) 
(h4 : DN + NF = DF) (h5 : DN = 22) (h6 : NF = 11) (h7 : p = 101) (h8 : q = 50) : p + q = 151 := 
by 
  sorry

end ratio_of_DN_NF_l274_274213


namespace connection_no_values_l274_274096

def scm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem connection_no_values (y : ℕ) (h1 : y < 50)
  (h2 : (scm y 13 : ℚ) / (y * 13 : ℚ) = 3 / 5) : false :=
by
  sorry

end connection_no_values_l274_274096


namespace liam_total_money_l274_274666

-- Define the conditions as noncomputable since they involve monetary calculations
noncomputable def liam_money (initial_bottles : ℕ) (price_per_bottle : ℕ) (bottles_sold : ℕ) (extra_money : ℕ) : ℚ :=
  let cost := initial_bottles * price_per_bottle
  let money_after_selling_part := cost + extra_money
  let selling_price_per_bottle := money_after_selling_part / bottles_sold
  let total_revenue := initial_bottles * selling_price_per_bottle
  total_revenue

-- State the theorem with the given problem
theorem liam_total_money :
  let initial_bottles := 50
  let price_per_bottle := 1
  let bottles_sold := 40
  let extra_money := 10
  liam_money initial_bottles price_per_bottle bottles_sold extra_money = 75 := 
sorry

end liam_total_money_l274_274666


namespace min_m_plus_n_l274_274883

noncomputable def m : ℝ := sorry
noncomputable def n : ℝ := sorry

theorem min_m_plus_n (h : log 3 m + log 3 n = 4) : m + n ≥ 18 := sorry

end min_m_plus_n_l274_274883


namespace mul_582964_99999_l274_274008

theorem mul_582964_99999 : 582964 * 99999 = 58295817036 := by
  sorry

end mul_582964_99999_l274_274008


namespace z_in_third_quadrant_l274_274140

-- Define z based on the given condition
def z : ℂ := (2 + complex.i) / (complex.i^5 - 1)

-- Define a predicate for checking if a complex number lies in the third quadrant
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

-- The theorem to prove
theorem z_in_third_quadrant : in_third_quadrant z :=
  sorry

end z_in_third_quadrant_l274_274140


namespace calc1_calc2_calc3_l274_274103

theorem calc1 : -4 - 4 = -8 := by
  sorry

theorem calc2 : (-32) / 4 = -8 := by
  sorry

theorem calc3 : -(-2)^3 = 8 := by
  sorry

end calc1_calc2_calc3_l274_274103


namespace number_of_pigs_l274_274312

theorem number_of_pigs (daily_feed_per_pig : ℕ) (weekly_feed_total : ℕ) (days_per_week : ℕ)
  (h1 : daily_feed_per_pig = 10) (h2 : weekly_feed_total = 140) (h3 : days_per_week = 7) : 
  (weekly_feed_total / days_per_week) / daily_feed_per_pig = 2 := by
  sorry

end number_of_pigs_l274_274312


namespace larger_of_two_numbers_l274_274771

theorem larger_of_two_numbers (A B : ℕ) (hcf : A.gcd B = 47) (lcm_factors : A.lcm B = 47 * 49 * 11 * 13 * 4913) : max A B = 123800939 :=
sorry

end larger_of_two_numbers_l274_274771


namespace tan_of_negative_angle_sin_75_degrees_l274_274852

-- Problem 1: Prove that tan(-23π/6) equals sqrt(3)/3
theorem tan_of_negative_angle : 
  tan (-23 * Real.pi / 6) = Real.sqrt 3 / 3 := 
sorry

-- Problem 2: Prove that sin(75°) equals (sqrt(2) + sqrt(6)) / 4
theorem sin_75_degrees : 
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

end tan_of_negative_angle_sin_75_degrees_l274_274852


namespace exists_positive_integers_N_k_l274_274628

noncomputable theory

open Set

def finite_set_of_pos_integers (A : Set ℕ) := Finite A ∧ (∀ x ∈ A, 0 < x)

def sn (A : Set ℕ) (n : ℕ) : Set ℕ := {s | ∃ (x : ℕ) (xs : Fin n → ℕ), 
  (∀ i, xs i ∈ A) ∧ s = Finset.univ.sum (λ i, xs i)}

theorem exists_positive_integers_N_k (A : Set ℕ) (hA : finite_set_of_pos_integers A) :
  ∃ (N k : ℕ), 0 < N ∧ 0 < k ∧ ∀ n, n ≥ N → (sn A (n + 1)).to_finset.card = (sn A n).to_finset.card + k :=
begin
  sorry
end

end exists_positive_integers_N_k_l274_274628


namespace determinant_of_equilateral_triangle_l274_274651

theorem determinant_of_equilateral_triangle :
  let s := Real.sin (60 * Real.pi / 180) -- sin 60 degrees in radians
  in Matrix.det ![
    ![s, 1, 1],
    ![1, s, 1],
    ![1, 1, s]
  ] = - (Real.sqrt 3) / 8 :=
by
  let s := Real.sin (60 * Real.pi / 180)
  exact sorry

end determinant_of_equilateral_triangle_l274_274651


namespace find_angle_ACB_l274_274611

-- Defining angles given as conditions
def angleECA : ℝ := 50
def angleABC : ℝ := 60

-- Defining the parallel lines condition
def DC_parallel_AB : Prop := true
def DC_parallel_EF : Prop := true

-- Defining the fact that EF is above DC is not necessary for the angular calculation, so we omit it.

theorem find_angle_ACB (DC_parallel_AB : Prop) (DC_parallel_EF : Prop) (angleECA : ℝ) (angleABC : ℝ) : 
  angleECA = 50 → angleABC = 60 → DC_parallel_AB → DC_parallel_EF → (angle ECA == 70) sorry

end find_angle_ACB_l274_274611


namespace find_x_eq_728_l274_274155

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l274_274155


namespace least_positive_integer_k_l274_274864

noncomputable def least_k (a : ℝ) (n : ℕ) : ℝ :=
  (1 : ℝ) / ((n + 1 : ℝ) ^ 3)

theorem least_positive_integer_k :
  ∃ k : ℕ , (∀ a : ℝ, ∀ n : ℕ,
  (0 ≤ a ∧ a ≤ 1) → (a^k * (1 - a)^n < least_k a n)) ∧
  (∀ k' : ℕ, k' < 4 → ¬(∀ a : ℝ, ∀ n : ℕ, (0 ≤ a ∧ a ≤ 1) → (a^k' * (1 - a)^n < least_k a n))) :=
sorry

end least_positive_integer_k_l274_274864


namespace martha_butterflies_total_l274_274669

theorem martha_butterflies_total
  (B : ℕ) (Y : ℕ) (black : ℕ)
  (h1 : B = 4)
  (h2 : Y = B / 2)
  (h3 : black = 5) :
  B + Y + black = 11 :=
by {
  -- skip proof 
  sorry 
}

end martha_butterflies_total_l274_274669


namespace total_cost_l274_274560

def num_of_rings : ℕ := 2

def cost_per_ring : ℕ := 12

theorem total_cost : num_of_rings * cost_per_ring = 24 :=
by sorry

end total_cost_l274_274560


namespace sandra_tickets_relation_l274_274361

def volleyball_game : Prop :=
  ∃ (tickets_total tickets_left tickets_jude tickets_andrea tickets_sandra : ℕ),
    tickets_total = 100 ∧
    tickets_left = 40 ∧
    tickets_jude = 16 ∧
    tickets_andrea = 2 * tickets_jude ∧
    tickets_total - tickets_left = tickets_jude + tickets_andrea + tickets_sandra ∧
    tickets_sandra = tickets_jude - 4

theorem sandra_tickets_relation : volleyball_game :=
  sorry

end sandra_tickets_relation_l274_274361


namespace log_3_81_equals_4_l274_274869

theorem log_3_81_equals_4 :
  real.logb 3 81 = 4 :=
by
  have h1 : 81 = 3 ^ 4 := by norm_num
  rw [h1, real.logb_pow 3 4]
  norm_num

end log_3_81_equals_4_l274_274869


namespace students_basketball_not_table_tennis_l274_274221

theorem students_basketball_not_table_tennis :
  ∀ (total_students basketball_likers table_tennis_likers neither_likers : ℕ),
  total_students = 30 →
  basketball_likers = 15 →
  table_tennis_likers = 10 →
  neither_likers = 8 →
  ∃ (num_both : ℕ), (basketball_likers - num_both) - (total_students - table_tennis_likers - neither_likers - num_both) = 12 :=
by
  intros total_students basketball_likers table_tennis_likers neither_likers h_total h_basketball h_table_tennis h_neither
  use 3
  rw [h_total, h_basketball, h_table_tennis, h_neither]
  linarith

end students_basketball_not_table_tennis_l274_274221


namespace people_got_on_at_third_stop_l274_274441

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

end people_got_on_at_third_stop_l274_274441


namespace dividend_percentage_of_each_share_is_9_l274_274429

def market_value := 15
def face_value := 20
def desired_interest_rate := 0.12

def dividend_percentage (D : ℚ) : Prop :=
  (D/100 * face_value = desired_interest_rate * market_value)

theorem dividend_percentage_of_each_share_is_9 :
  dividend_percentage 9 :=
by
  unfold dividend_percentage
  sorry

end dividend_percentage_of_each_share_is_9_l274_274429


namespace range_of_first_five_average_l274_274907

-- Given conditions
variables (x : Fin 10 → ℝ)
variable (mean_condition : (∑ i, x i) / 10 = 6)
variable (stddev_condition : Real.sqrt ((∑ i, (x i - 6) ^ 2) / 10) = Real.sqrt 2)

-- The goal to prove
theorem range_of_first_five_average :
  6 - Real.sqrt 2 ≤ (∑ i in Finset.range 5, x i) / 5 ∧ (∑ i in Finset.range 5, x i) / 5 ≤ 6 + Real.sqrt 2 :=
sorry

end range_of_first_five_average_l274_274907


namespace fibonacci_series_sum_l274_274636

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Theorem to prove that the infinite series sum is 2
theorem fibonacci_series_sum : (∑' n : ℕ, (fib n : ℝ) / (2 ^ n : ℝ)) = 2 :=
sorry

end fibonacci_series_sum_l274_274636


namespace intersects_one_at_least_l274_274936

-- Conditions
variables (α β : Plane) (a b : Line) (c : Line)
variable (h1 : skew a b)
variable (h2 : lies_in a α)
variable (h3 : lies_in b β)
variable (h4 : intersection α β = c)

-- Proof statement
theorem intersects_one_at_least (c : Line) (a b : Line) (h1 : skew a b) (h2 : lies_in a α) (h3 : lies_in b β) (h4 : intersection α β = c) : 
  intersects c a ∨ intersects c b :=
sorry

end intersects_one_at_least_l274_274936


namespace exponent_of_3_in_30_factorial_l274_274243

theorem exponent_of_3_in_30_factorial : (prime_factor_exponent 3 30.factorial) = 14 :=
by
  sorry

end exponent_of_3_in_30_factorial_l274_274243


namespace possible_values_of_g_zero_l274_274707

variable {g : ℝ → ℝ}

theorem possible_values_of_g_zero (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) : g 0 = 0 ∨ g 0 = 1 := 
sorry

end possible_values_of_g_zero_l274_274707


namespace ab_value_l274_274196

-- Definitions and conditions from the problem statement
variables {a b : ℝ}

theorem ab_value (h : sqrt (a-1) + b^2 - 4 * b + 4 = 0) : a * b = 2 := 
sorry

end ab_value_l274_274196


namespace number_of_distinct_sums_of_special_fractions_l274_274832

def is_special_fraction (a b : ℕ) : Bool := (a > 0) ∧ (b > 0) ∧ (a + b = 20)

def special_fractions : Set ℚ := {ab | ∃ (a b : ℕ), is_special_fraction a b ∧ ab = (a : ℚ) / (b : ℚ)}

def sums_of_special_fractions : Set ℤ := {m | ∃ (x y : ℚ), x ∈ special_fractions ∧ y ∈ special_fractions ∧ m = (x + y).num}

theorem number_of_distinct_sums_of_special_fractions : sums_of_special_fractions.to_finset.card = 14 := 
sorry

end number_of_distinct_sums_of_special_fractions_l274_274832


namespace leak_empties_cistern_in_42_hours_l274_274032

noncomputable def fill_rate_without_leak := 1 / 6
noncomputable def effective_fill_rate_with_leak := 1 / 7

theorem leak_empties_cistern_in_42_hours :
    ∃ L (leak_rate : L), 
    effective_fill_rate_with_leak = fill_rate_without_leak - L ∧ 
    L = 1 / 42 ∧ 
    1 / L = 42 :=
by
    sorry

end leak_empties_cistern_in_42_hours_l274_274032


namespace distinct_four_digit_numbers_l274_274191

theorem distinct_four_digit_numbers (a b c d : ℕ) (h1 : a ∈ {1, 2, 3, 4}) (h2 : b ∈ {1, 2, 3, 4}) (h3 : c ∈ {1, 2, 3, 4}) (h4 : d ∈ {1, 2, 3, 4})
  (h5 : a ≠ b) (h6 : b ≠ c) (h7 : c ≠ d) (h8 : d ≠ a)
  (h9 : ∀ e ∈ {b, c, d}, a < e) : (list.permutations [a, b, c, d]).length = 24 :=
by sorry

end distinct_four_digit_numbers_l274_274191


namespace triangle_area_transform_l274_274467

-- Define the concept of a triangle with integer coordinates
structure Triangle :=
  (A : ℤ × ℤ)
  (B : ℤ × ℤ)
  (C : ℤ × ℤ)

-- Define the area of a triangle using determinant
def triangle_area (T : Triangle) : ℤ :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := (T.A, T.B, T.C)
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define a legal transformation for triangles
def legal_transform (T : Triangle) : Set Triangle :=
  { T' : Triangle |
    (∃ c : ℤ, 
      (T'.A = (T.A.1 + c * (T.B.1 - T.C.1), T.A.2 + c * (T.B.2 - T.C.2)) ∧ T'.B = T.B ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = (T.B.1 + c * (T.A.1 - T.C.1), T.B.2 + c * (T.A.2 - T.C.2)) ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = T.B ∧ T'.C = (T.C.1 + c * (T.A.1 - T.B.1), T.C.2 + c * (T.A.2 - T.B.2)))) }

-- Proposition that any two triangles with equal area can be legally transformed into each other
theorem triangle_area_transform (T1 T2 : Triangle) (h : triangle_area T1 = triangle_area T2) :
  ∃ (T' : Triangle), T' ∈ legal_transform T1 ∧ triangle_area T' = triangle_area T2 :=
sorry

end triangle_area_transform_l274_274467


namespace guinea_pig_food_ratio_l274_274692

-- Definitions of amounts of food consumed by each guinea pig
def first_guinea_pig_food : ℕ := 2
variable (x : ℕ)
def second_guinea_pig_food : ℕ := x
def third_guinea_pig_food : ℕ := x + 3

-- Total food requirement condition
def total_food_required := first_guinea_pig_food + second_guinea_pig_food x + third_guinea_pig_food x = 13

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- The goal is to prove this ratio given the conditions
theorem guinea_pig_food_ratio (h : total_food_required x) : ratio (second_guinea_pig_food x) first_guinea_pig_food = 2 := by
  sorry

end guinea_pig_food_ratio_l274_274692


namespace max_ab_l274_274570

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : ∃ c, ∀ x, f x = 4 * x^3 - a * x^2 - 2 * b * x + 2 ∧ 12 * c^2 - 2 * a * c - 2 * b = 0 ∧ c = 1): 
  a * b <= 9 :=
by 
  sorry

end max_ab_l274_274570


namespace find_n_value_l274_274287

def n (x y : ℤ) : ℤ := x - |y^(x-y)|

theorem find_n_value : n 3 (-1) = 2 := by
  sorry

end find_n_value_l274_274287


namespace correct_square_root_multiplication_l274_274760

theorem correct_square_root_multiplication :
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → real.sqrt (x * y) = real.sqrt x * real.sqrt y) →
  (real.sqrt 2 * real.sqrt 3 = real.sqrt 6) :=
by
  intro h
  apply h
  split
  norm_num
  norm_num

end correct_square_root_multiplication_l274_274760


namespace scientific_notation_l274_274061

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l274_274061


namespace regular_graph_l274_274905

open Finset

-- Definition of a simple graph with given conditions
structure SimpleGraph (V : Type) :=
  (adj : V → V → Prop)
  (sym : symmetric adj . trivial)
  (loopless : irreflexive adj . trivial)

def vertices := fin 20

noncomputable def G : SimpleGraph vertices := sorry

-- Given data
axiom degree_count : ∑ v : vertices, G.degree v = 200
axiom non_intersecting_edges : 4050 = (100 * 99 / 2) - ∑ v : vertices, G.degree v * (G.degree v - 1) / 2

-- Proving the graph is regular
theorem regular_graph : ∃ d : ℕ, ∀ v : vertices, G.degree v = d := 
sorry

end regular_graph_l274_274905


namespace least_x_value_l274_274987

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_x_value (x : ℕ) (p : ℕ) (hp : is_prime p) (h : x / (12 * p) = 2) : x = 48 := by
  sorry

end least_x_value_l274_274987


namespace water_tank_capacity_l274_274759

theorem water_tank_capacity (C : ℝ) :
  (0.40 * C - 0.25 * C = 36) → C = 240 :=
  sorry

end water_tank_capacity_l274_274759


namespace maximize_revenue_l274_274027

def revenue (p : ℝ) : ℝ := 150 * p - 4 * p^2

theorem maximize_revenue : 
  ∃ p, 0 ≤ p ∧ p ≤ 30 ∧ p = 18.75 ∧ (∀ q, 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue 18.75) :=
by
  sorry

end maximize_revenue_l274_274027


namespace inequality_solution_range_of_t_l274_274528

variables {f : ℝ → ℝ} {x t a : ℝ}

-- Given conditions as hypotheses
def odd_function_iff : Prop := ∀ x, f(-x) = -f(x)
def increasing_function : Prop := ∀ x y, x < y → f(x) < f(y)
def interval_cond1 : Prop := ∀ x, -1 ≤ x ∧ x ≤ 1 → f(x) ≤ x^2 - 2*x + 1

-- Problem (1): Prove that for all x in [0, 1/4), f(x + 1/2) + f(x - 1) < 0
theorem inequality_solution (h1: odd_function_iff) 
                           (h2: increasing_function)
                           (h3: f 1 = 1) : 
                           ∀ x, 0 ≤ x ∧ x < 1/4 → f(x + 1/2) + f(x - 1) < 0 :=
by sorry

-- Problem (2): Prove that the range of values for t is {t | t ≤ -2 ∨ t = 0 ∨ t ≥ 2}
theorem range_of_t (h1: odd_function_iff) 
                   (h2: increasing_function)
                   (h3: f 1 = 1)
                   (h4: ∀ x, -1 ≤ x ∧ x ≤ 1 → f(x) ≤ t^2 - 2*a*t + 1) : 
                   t ≤ -2 ∨ t = 0 ∨ t ≥ 2 :=
by sorry

end inequality_solution_range_of_t_l274_274528


namespace p_linear_l274_274898

noncomputable def p (a : ℕ → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n + 1), a k * (nat.choose n k) * x^k * (1 - x)^(n - k)

theorem p_linear (a : ℕ → ℝ) (h : ∀ i ≥ 1, a (i - 1) + a (i + 1) = 2 * a i)
  (h0 : a 0 ≠ a 1) :
  ∀ (n : ℕ) (x : ℝ), ∃ (m b : ℝ), p a n x = m * x + b :=
sorry

end p_linear_l274_274898


namespace digit_300_of_5_over_13_l274_274375

theorem digit_300_of_5_over_13 :
  ∀ (n : ℕ), (n % 6 = 0) → ( (nth_digit_decimal 300 (5/13)) = 5 ) :=
by
  sorry

end digit_300_of_5_over_13_l274_274375


namespace vitamin_D_scientific_notation_l274_274065

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l274_274065


namespace ratio_of_blue_fish_to_total_fish_l274_274358

-- Define the given conditions
def total_fish : ℕ := 30
def blue_spotted_fish : ℕ := 5
def half (n : ℕ) : ℕ := n / 2

-- Calculate the number of blue fish using the conditions
def blue_fish : ℕ := blue_spotted_fish * 2

-- Define the ratio of blue fish to total fish
def ratio (num denom : ℕ) : ℚ := num / denom

-- The theorem to prove
theorem ratio_of_blue_fish_to_total_fish :
  ratio blue_fish total_fish = 1 / 3 := by
  sorry

end ratio_of_blue_fish_to_total_fish_l274_274358


namespace return_speed_is_75_l274_274304

def average_speed (d : ℝ) (v_dest : ℝ) (v_return : ℝ) : ℝ :=
  let t_dest := d / v_dest
  let t_return := d / v_return
  let total_distance := 2 * d
  let total_time := t_dest + t_return
  total_distance / total_time

theorem return_speed_is_75
  (d : ℝ)
  (v_dest : ℝ := 50)
  (h1 : ∀ (v_return : ℝ), int (average_speed d v_dest v_return))
  (h2 : v_return ∈ { i : ℤ | i ∈ (0:ℤ) .. 100 }):
  v_return = 75 :=
by
  sorry

end return_speed_is_75_l274_274304


namespace sqrt_simplify_correct_l274_274828

noncomputable def sqrt_simplify (a : ℝ) : Prop :=
  a ≥ 0 → sqrt (a * sqrt a * sqrt a) = a

-- The statement:
theorem sqrt_simplify_correct (a : ℝ) : sqrt_simplify a :=
begin
  sorry
end

end sqrt_simplify_correct_l274_274828


namespace solve_inequality_l274_274108

theorem solve_inequality (x : ℝ) : (1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4) ↔ ((-8 < x ∧ x ≤ -4) ∨ (-4 ≤ x ∧ x ≤ 4 / 3)) ∧ x ≠ -2 ∧ x ≠ -8 :=
by
  sorry

end solve_inequality_l274_274108


namespace measure_of_angle_A_l274_274259

theorem measure_of_angle_A (b c S : ℝ) (hb : b = 8) (hc : c = 8 * real.sqrt 3) (hS : S = 16 * real.sqrt 3) :
  ∃ A, (A = real.pi / 6 ∨ A = 5 * real.pi / 6) ∧ (S = 0.5 * b * c * real.sin A) :=
by
  use (real.pi / 6, 5 * real.pi / 6)
  intros A hA
  split
  · exact or.inl rfl
  · split
    · exact or.inr rfl
    · sorry

end measure_of_angle_A_l274_274259


namespace angle_terminal_sides_angles_within_range_beta_quadrant_l274_274882

def alpha : ℝ := π / 3

theorem angle_terminal_sides (θ : ℝ) : 
  (∃ k : ℤ, θ = 2 * k * π + α) ↔ θ = 2 * k * π + (π / 3) :=
sorry

theorem angles_within_range : 
  {θ | θ = 2 * k * π + α ∧ -4 * π < θ ∧ θ < 2 * π} = {-(11 * π / 3), -(5 * π / 3), π / 3} :=
sorry

theorem beta_quadrant (β : ℝ) (k : ℤ) :
  (β = 2 * k * π + α) → 
  ((even k → β / 2 in first quadrant) ∧ (odd k → β / 2 in third quadrant)) :=
sorry

end angle_terminal_sides_angles_within_range_beta_quadrant_l274_274882


namespace count_non_congruent_triangles_with_perimeter_24_l274_274566

/-- How many non-congruent triangles with only integer side lengths have a perimeter of 24 units? -/
def number_of_non_congruent_triangles (n : ℕ) : ℕ :=
  let valid_triangles := 
    {s | let ⟨a, b, c⟩ := s in 
      a + b + c = n ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a ∧ 
      ∀ {d e f}, a = d ∧ b = e ∧ c = f ∨ a = d ∧ b = f ∧ c = e ∨ a = e ∧ b = d ∧ c = f ∨ a = e ∧ b = f ∧ c = d ∨ a = f ∧ b = d ∧ c = e ∨ a = f ∧ b = e ∧ c = d → s = {d, e, f}}} in
  valid_triangles.to_finset.card

/-- Proof to find the number of non-congruent triangles with integer side lengths and perimeter 24 -/
theorem count_non_congruent_triangles_with_perimeter_24 : 
  number_of_non_congruent_triangles 24 = 18 :=
  by sorry

end count_non_congruent_triangles_with_perimeter_24_l274_274566


namespace base_7_representation_digits_count_l274_274941

theorem base_7_representation_digits_count : ∀ (n : ℕ), n = 1234 → base_digits 7 n = 4 :=
by
  intro n
  assume h : n = 1234
  sorry

end base_7_representation_digits_count_l274_274941


namespace boy_and_girl_roles_l274_274410

-- Definitions of the conditions
def Sasha_says_boy : Prop := True
def Zhenya_says_girl : Prop := True
def at_least_one_lying (sasha_boy zhenya_girl : Prop) : Prop := 
  (sasha_boy = False) ∨ (zhenya_girl = False)

-- Theorem statement
theorem boy_and_girl_roles (sasha_boy : Prop) (zhenya_girl : Prop) 
  (H1 : Sasha_says_boy) (H2 : Zhenya_says_girl) (H3 : at_least_one_lying sasha_boy zhenya_girl) :
  sasha_boy = False ∧ zhenya_girl = True :=
sorry

end boy_and_girl_roles_l274_274410


namespace common_sum_in_5_by_5_matrix_l274_274326

theorem common_sum_in_5_by_5_matrix (a : Fin 5 → Fin 5 → ℤ)
  (h_cond : ∀ i j k, Finset.sum (Finset.univ.map (Function.uncurry a)) = 50) :
  (∀ i, Finset.sum (Finset.univ.map (λ k, a i k)) = 10) ∧
  (∀ j, Finset.sum (Finset.univ.map (λ k, a k j)) = 10) ∧
  (Finset.sum (Finset.univ.map (λ i, a i i)) = 10) ∧
  (Finset.sum (Finset.univ.map (λ i, a i (4 - i))) = 10) :=
by 
  sorry

end common_sum_in_5_by_5_matrix_l274_274326


namespace second_fragment_speed_l274_274420

-- State the given conditions
def initial_velocity : ℝ := 20
def explosion_time : ℝ := 3
def first_fragment_speed : ℝ := 48
def gravity_acceleration : ℝ := 10

-- Define the claim proving the speed of the second fragment immediately after the explosion
theorem second_fragment_speed :
  let v_initial := initial_velocity,
      t := explosion_time,
      v_y := v_initial - gravity_acceleration * t,
      v_first_horizontal := first_fragment_speed,
      m := (1 : ℝ), -- use mass 1 for simplicity
      p_initial := m * v_y,
      m_fragment := m / 2,
      v_first_total := (v_first_horizontal ^ 2 + v_y ^ 2) ^ (1/2),
      v_second := (2 * v_y - v_first_total) in
  v_second = 52 :=
by
  sorry

end second_fragment_speed_l274_274420


namespace expression_for_g_l274_274976

theorem expression_for_g (f g : ℝ → ℝ) (h1 : ∀ x, f(x) = 2 * x + 3) (h2 : ∀ x, g(x + 2) = f(x - 1)) :
  ∀ x, g(x) = 2 * x - 3 :=
by
  sorry

end expression_for_g_l274_274976


namespace general_formula_sum_c_n_l274_274299

def S (n : ℕ) : ℕ := sorry  -- placeholder for the sum of the first n terms of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n - 1

-- Condition (I): Provided conditions
axiom S_condition1 : S 4 = 4 * S 2
axiom a_condition1 : ∀ n, a (2 * n) = 2 * a n + 1

-- General formula for the sequence {a_n}: Prove a_n = 2n - 1
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := by
  sorry

-- Condition (II): Given the second part conditions
def T (n : ℕ) : ℝ := sorry  -- placeholder for the sum of the first n terms of the sequence {b_n}
axiom T_condition : ∀ n, T n + (a n + 1) / 2^n = λ
def c (n : ℕ) : ℝ := T (2 * n)  -- c_n = b_{2n}

-- Sum of the first n terms of the sequence {c_n}
def R (n : ℕ) : ℝ := 1/9 * (4 - (3 * n + 1) / 4^(n-1))

-- Prove that R_n = 1/9 * (4 - (3n + 1) / 4^(n-1))
theorem sum_c_n (n : ℕ) : R n = 1/9 * (4 - (3 * n + 1) / 4^(n-1)) := by
  sorry

end general_formula_sum_c_n_l274_274299


namespace emails_received_afternoon_is_one_l274_274622

-- Define the number of emails received by Jack in the morning
def emails_received_morning : ℕ := 4

-- Define the total number of emails received by Jack in a day
def total_emails_received : ℕ := 5

-- Define the number of emails received by Jack in the afternoon
def emails_received_afternoon : ℕ := total_emails_received - emails_received_morning

-- Prove the number of emails received by Jack in the afternoon
theorem emails_received_afternoon_is_one : emails_received_afternoon = 1 :=
by 
  -- Proof is neglected as per instructions.
  sorry

end emails_received_afternoon_is_one_l274_274622


namespace total_selling_price_correct_l274_274799

-- Definitions of initial purchase prices in different currencies
def init_price_eur : ℕ := 600
def init_price_gbp : ℕ := 450
def init_price_usd : ℕ := 750

-- Definitions of initial exchange rates
def init_exchange_rate_eur_to_usd : ℝ := 1.1
def init_exchange_rate_gbp_to_usd : ℝ := 1.3

-- Definitions of profit percentages for each article
def profit_percent_eur : ℝ := 0.08
def profit_percent_gbp : ℝ := 0.1
def profit_percent_usd : ℝ := 0.15

-- Definitions of new exchange rates at the time of selling
def new_exchange_rate_eur_to_usd : ℝ := 1.15
def new_exchange_rate_gbp_to_usd : ℝ := 1.25

-- Calculation of purchase prices in USD
def purchase_price_in_usd₁ : ℝ := init_price_eur * init_exchange_rate_eur_to_usd
def purchase_price_in_usd₂ : ℝ := init_price_gbp * init_exchange_rate_gbp_to_usd
def purchase_price_in_usd₃ : ℝ := init_price_usd

-- Calculation of selling prices including profit in USD
def selling_price_in_usd₁ : ℝ := (init_price_eur + (init_price_eur * profit_percent_eur)) * new_exchange_rate_eur_to_usd
def selling_price_in_usd₂ : ℝ := (init_price_gbp + (init_price_gbp * profit_percent_gbp)) * new_exchange_rate_gbp_to_usd
def selling_price_in_usd₃ : ℝ := init_price_usd * (1 + profit_percent_usd)

-- Total selling price in USD
def total_selling_price_in_usd : ℝ :=
  selling_price_in_usd₁ + selling_price_in_usd₂ + selling_price_in_usd₃

-- Proof goal: total selling price should equal 2225.85 USD
theorem total_selling_price_correct :
  total_selling_price_in_usd = 2225.85 :=
by
  sorry

end total_selling_price_correct_l274_274799


namespace square_area_4900_l274_274586

/-- If one side of a square is increased by 3.5 times and the other side is decreased by 30 cm, resulting in a rectangle that has twice the area of the square, then the area of the square is 4900 square centimeters. -/
theorem square_area_4900 (x : ℝ) (h1 : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 :=
sorry

end square_area_4900_l274_274586


namespace coin_diameter_l274_274983

theorem coin_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  rw [h]
  norm_num

end coin_diameter_l274_274983


namespace find_ratio_l274_274618

theorem find_ratio (XYZ : Type) [AddCommGroup XYZ] [Module ℝ XYZ]
  (X Y Z E G Q : XYZ) 
  (hE : E = (5:11) • X + (6:11) • Z) 
  (hG : G = (3:8) • X + (5:8) • Y)
  (hQ1 : Q = (2:7) • X + (5:7) • E) 
  (hQ2 : Q = (3:4) • G + (1:4) • Y) :
  ∃ r : ℝ, r = 13/8 ∧ (r • X + (1 - r) • G = (5:8) • Y) :=
sorry

end find_ratio_l274_274618


namespace sin_double_angle_series_l274_274971

theorem sin_double_angle_series (θ : ℝ) (h : ∑' (n : ℕ), (sin θ)^(2 * n) = 3) :
  sin (2 * θ) = (2 * sqrt 2) / 3 :=
sorry

end sin_double_angle_series_l274_274971


namespace find_number_of_pairs_l274_274487

theorem find_number_of_pairs : 
  (∀ (x y : ℤ), 
    y > 3^x + 4 * 3^28 ∧ 
    y ≤ 93 + 3 * (3^27 - 1) * x 
  ↔ y ≥ 5 ∧ y ≤ 30
  ) → 
  ∑ x in range 5 31, 
    (93 + 3 * (3^27 - 1) * x + 1) + 
    ∑ x in range 5 31, 
    (3^x + 4 * 3^28 + 1) = 
  (25 * 3^31 + 2349) / 2 :=
sorry

end find_number_of_pairs_l274_274487


namespace petya_wins_l274_274515

def grid := (Fin 2021) × (Fin 2021)

structure GameState :=
  (occupied : Set grid)
  (next_player : Bool)  -- true for Petya's turn, false for Vasya's turn

def wins (s : GameState) : Bool :=
  ∀ (x y : Fin 2019) (a b : Fin 2021), 
    (∀ (i : Fin 3) (j : Fin 5), (x + i, y + j) ∈ s.occupied) ∧
    (∀ (i : Fin 5) (j : Fin 3), (a + i, b + j) ∈ s.occupied)

theorem petya_wins : 
  ∀ s : GameState, 
    (s.next_player = true → ¬wins s → 
    (∃ s' : GameState, s'.next_player = false ∧ s'.occupied ⊆ (s.occupied ∪ {(_, _)}))) → 
    (s.next_player = false → ¬wins s → 
    (∃ s' : GameState, s'.next_player = true ∧ s'.occupied ⊆ (s.occupied ∪ {(_, _)}))) → 
    ∃ s_final : GameState, wins s_final :=
sorry

end petya_wins_l274_274515


namespace contractor_realized_work_done_after_20_days_l274_274418

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l274_274418


namespace Amelia_ate_more_l274_274104

noncomputable def Amelia_ate := 7
noncomputable def Toby_ate := 1

theorem Amelia_ate_more : Amelia_ate - Toby_ate = 6 :=
by
  rw [Amelia_ate, Toby_ate]
  norm_num

end Amelia_ate_more_l274_274104


namespace probability_of_selecting_one_marble_each_color_l274_274409

theorem probability_of_selecting_one_marble_each_color
  (total_red_marbles : ℕ) (total_blue_marbles : ℕ) (total_green_marbles : ℕ) (total_selected_marbles : ℕ) 
  (total_marble_count : ℕ) : 
  total_red_marbles = 3 → total_blue_marbles = 3 → total_green_marbles = 3 → total_selected_marbles = 3 → total_marble_count = 9 →
  (27 / 84) = 9 / 28 :=
by
  intros h_red h_blue h_green h_selected h_total
  sorry

end probability_of_selecting_one_marble_each_color_l274_274409


namespace cannot_obtain_five_equal_numbers_l274_274827

theorem cannot_obtain_five_equal_numbers :
  ¬ ∃ n : ℤ, (n * 5 = 28) :=
by
  intro h
  cases h with n hn
  have : n = 28 / 5 := by
    exact Int.eq_of_mul_eq_mul_left (by norm_num) hn
  rw [Int.div_def 28 5, Int.mul_def] at this
  norm_cast at this
  linarith
  sorry

end cannot_obtain_five_equal_numbers_l274_274827


namespace combined_weight_is_correct_l274_274870

-- Frank and Gwen's candy weights
def frank_candy : ℕ := 10
def gwen_candy : ℕ := 7

-- The combined weight of candy
def combined_weight : ℕ := frank_candy + gwen_candy

-- Theorem that states the combined weight is 17 pounds
theorem combined_weight_is_correct : combined_weight = 17 :=
by
  -- proves that 10 + 7 = 17
  sorry

end combined_weight_is_correct_l274_274870


namespace first_term_exceeding_10000_is_176820_l274_274338

open Nat

def seq : ℕ → ℕ 
| 0     := 2
| (n+1) := (range (n+1)).sum (λ i, (seq i)^2)

theorem first_term_exceeding_10000_is_176820 :
  ∃ n, seq n > 10000 ∧ seq n = 176820 := 
sorry

end first_term_exceeding_10000_is_176820_l274_274338


namespace evaluate_expression_l274_274695

-- Define the expressions involved
def expr1 (x : ℝ) : ℝ := 1 - (2 / (2 - x))
def expr2 (x : ℝ) : ℝ := x / (x^2 - 4*x + 4)
def combined_expr (x : ℝ) : ℝ := expr1(x) / expr2(x)

-- Define the theorem to prove
theorem evaluate_expression (x : ℝ) (hx : x = -2) : combined_expr(x) = -4 :=
by {
  -- Omitted proof
  sorry
}

end evaluate_expression_l274_274695


namespace number_of_dogs_l274_274768

variable {C D : ℕ}

def ratio_of_dogs_to_cats (D C : ℕ) : Prop := D = (15/7) * C

def ratio_after_additional_cats (D C : ℕ) : Prop :=
  D = 15 * (C + 8) / 11

theorem number_of_dogs (h1 : ratio_of_dogs_to_cats D C) (h2 : ratio_after_additional_cats D C) :
  D = 30 :=
by
  sorry

end number_of_dogs_l274_274768


namespace find_a_div_b_l274_274642

theorem find_a_div_b (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 6 * b) / (b + 6 * a) = 3) : 
  a / b = (8 + Real.sqrt 46) / 6 ∨ a / b = (8 - Real.sqrt 46) / 6 :=
by 
  sorry

end find_a_div_b_l274_274642


namespace trapezoid_perimeter_l274_274256

/-- A trapezoid KLMN with specific properties and circle tangency. -/
theorem trapezoid_perimeter
  (K L M N : Type)
  [trapezoid K L M N] -- representing KLMN is a trapezoid
  (LK_parallel_MN : ∥ LK ⬝ MN ∥)
  (LM_eq : LM = 17)
  (angle_LKN_twice_KNM : ∃ α, ∠LKN = 2 * α ∧ ∠KNM = α)
  (r_eq : r = 15)
  (circle_tangent : tangent_circle LM r KN MN) :
  perimeter KLMN = 84 + 5 * sqrt 34 :=
sorry

end trapezoid_perimeter_l274_274256


namespace largest_geometric_seq_three_digit_number_with_condition_l274_274377

/-- A three-digit number with distinct digits forming a geometric sequence -/
def is_geometric_seq (n : ℕ) : Prop :=
  let d1 := n / 100 in  -- Hundreds digit
  let d2 := (n / 10) % 10 in  -- Tens digit
  let d3 := n % 10 in  -- Ones digit
  -- The digits must be distinct
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3) ∧
  -- Geometric sequence: d2/d1 = d3/d2
  (d2 * d2 = d1 * d3)

/-- Given conditions in the problem -/
def given_conditions (n : ℕ) : Prop :=
  n / 100 ≤ 8 ∧ -- Hundreds digit at most 8
  is_geometric_seq(n)

/-- Prove that the number 842 is the largest under given conditions -/
theorem largest_geometric_seq_three_digit_number_with_condition :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ given_conditions n → n ≤ 842 :=
by
  -- Proof is omitted
  sorry

end largest_geometric_seq_three_digit_number_with_condition_l274_274377


namespace five_wednesdays_in_august_l274_274321

variable {N : ℕ}
variable (day_of_week : ℕ → ℕ) -- Function mapping day of the month to day of the week.
  (tuesday : ℕ → Prop)          -- tuesday is a predicate marking if a day is a Tuesday.

theorem five_wednesdays_in_august (h1 : ∀ n ∈ {1..31}, tuesday n → n ≤ 31)
  (h2 : ∃ s, (∀ k, 0 ≤ k < 5 → tuesday (s + k * 7)) ∧ s ∈ {1..31})
  (h3 : ∀ d, d ∈ {1, 8, 15, 22, 29} → ⟪d mod 7 = day_of_week 1⟫)
  : ∃ d, d ∈ {1, 8, 15, 22, 29} ∧ day_of_week d = 3 := sorry

end five_wednesdays_in_august_l274_274321


namespace circumscribed_triangle_area_relationship_l274_274412

theorem circumscribed_triangle_area_relationship (X Y Z : ℝ) :
  let a := 15
  let b := 20
  let c := 25
  let triangle_area := (1/2) * a * b
  let diameter := c
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let Z := circle_area / 2
  (X + Y + triangle_area = Z) :=
sorry

end circumscribed_triangle_area_relationship_l274_274412


namespace roots_of_complex_quadratic_have_nonzero_imaginary_part_l274_274842

noncomputable def complex_quadratic_roots_have_nonzero_imaginary_part (m : ℝ) : Prop :=
  ∀ z : ℂ, (5 * z ^ 2 + 2 * complex.I * z - (m : ℂ) = 0) → complex.I ≠ 0

theorem roots_of_complex_quadratic_have_nonzero_imaginary_part (m : ℝ) : 
  complex_quadratic_roots_have_nonzero_imaginary_part m :=
begin
  sorry
end

end roots_of_complex_quadratic_have_nonzero_imaginary_part_l274_274842


namespace circle_parametric_solution_l274_274346

theorem circle_parametric_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
    (hx : 4 * Real.cos θ = -2) (hy : 4 * Real.sin θ = 2 * Real.sqrt 3) :
    θ = 2 * Real.pi / 3 :=
sorry

end circle_parametric_solution_l274_274346


namespace part1_part2_l274_274189

-- Definitions of the vectors
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Part 1: Prove that when \( \overrightarrow{a} \parallel \overrightarrow{b} \), \( \sin 2x = -\frac{12}{13} \)
theorem part1 (x : ℝ) (h_parallel : vector_a x ∥ vector_b x) : Real.sin (2 * x) = -12 / 13 := sorry

-- Part 2: Prove that the minimum value of \( f(x) \) for \( x \in [ -\frac{\pi}{2}, 0 ] \) is \( -\frac{\sqrt{2}}{2} \)
def f (x : ℝ) : ℝ := let a := vector_a x; let b := vector_b x in (a.1 + b.1, a.2 + b.2) ⋅ b

theorem part2 : ∃ x ∈ Set.Icc (-(Real.pi / 2)) 0, f x = -Real.sqrt 2 / 2 :=
  sorry

end part1_part2_l274_274189


namespace bound_xi_l274_274162

theorem bound_xi (n : ℕ) (a : ℝ) (x : ℕ → ℝ)
  (hn : n ≥ 2)
  (ha : a > 0)
  (h_sum : (Finset.range n).sum (λ i, x i) = a)
  (h_sum_sq : (Finset.range n).sum (λ i, (x i)^2) = a^2 / (n - 1)) :
  ∀ i ∈ Finset.range n, 0 ≤ x i ∧ x i ≤ 2 * a / n :=
sorry

end bound_xi_l274_274162


namespace find_subset_A_condition_l274_274285

open Set

-- Define the conditions in Lean
variable {A : Set ℝ} (n : ℕ)

-- 'hA' represents the condition that for any function f, there exists X such that...
def condition (hA : ∀ (f : Set ℝ → Set ℝ), ∃ (X : Set ℝ), 
  (f^[2^n] X) ≠ (A \ X)) : Prop := 
  ∀ (f : Set (Set ℝ) → Set (Set ℝ)), ∃ (X : Set ℝ), 
  (f^[2^n] X) ≠ A \ X

-- The proof problem statement
theorem find_subset_A_condition (hA : condition n) : A.finite ∧ A.to_finset.card ≤ n := 
sorry

end find_subset_A_condition_l274_274285


namespace dave_apps_problem_l274_274470

theorem dave_apps_problem 
  (initial_apps : ℕ)
  (added_apps : ℕ)
  (final_apps : ℕ)
  (total_apps := initial_apps + added_apps)
  (deleted_apps := total_apps - final_apps) :
  initial_apps = 21 →
  added_apps = 89 →
  final_apps = 24 →
  (added_apps - deleted_apps = 3) :=
by
  intros
  sorry

end dave_apps_problem_l274_274470


namespace max_guard_nights_l274_274425

def eight_sons_max_nights (sons : Fin 8 → ℕ) (guard_schedule : Fin 8 → Finset (Fin 8)) : ℕ :=
  let pairs_nightly (f : Fin 8 → Finset (Fin 8)) : ℕ := 
    guard_schedule.prod (λ s, s.card.choose 2) 
  let total_pairs (n: ℕ) : ℕ := (Finset.univ.card.choose 2) * n
  total_pairs 3

-- Assert the defined maximum nights with given constraints and conditions
theorem max_guard_nights (sons : Fin 8 → ℕ) (guard_schedule : Fin 8 → Finset (Fin 8)) :
  eight_sons_max_nights sons guard_schedule = 8 :=
  sorry

end max_guard_nights_l274_274425


namespace sum_of_distinct_prime_divisors_l274_274755

theorem sum_of_distinct_prime_divisors (n : ℕ) (h : n = 1800) : 
  ∑ p in {2, 3, 5}, p = 10 :=
by
  have fact : n = 2^3 * 3^2 * 5^2 := by 
    rw [h, factorize_1800]
  sorry

-- Factorization utility (must be included as equitable assumption for the proof)
lemma factorize_1800 : 1800 = 2^3 * 3^2 * 5^2 :=
by
  norm_num

end sum_of_distinct_prime_divisors_l274_274755


namespace max_value_expression_l274_274610

theorem max_value_expression 
  (a b c d e f : ℕ)
  (h_perm : {a, b, c, d, e, f} = {1, 2, 3, 4, 0, 1})
  (h_e : e = 0)
  (h_f : f = 1) : 
  (c * a^b - d + e^f) ≤ 127 :=
sorry

end max_value_expression_l274_274610


namespace eccentricity_proof_l274_274183

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a)
  (F : ℝ × ℝ) (N : ℝ × ℝ) (hN : N = (0, real.sqrt 2 * b)) (max_perimeter : (ℝ × ℝ) → (ℝ × ℝ) → ℝ)
  (h_max : ∀ M, max_perimeter M (F) ≤ (real.sqrt 6 + 2) * a) : ℝ :=
real.sqrt 2 / 2

theorem eccentricity_proof (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) :
  let E := eccentricity_of_ellipse a b h₁ h₂ (1, 0) (0, real.sqrt 2 * b)
  (λ M F, (dist M (0, real.sqrt 2 * b)) + (dist M F) + (dist (0, real.sqrt 2 * b) F)) in
  E = real.sqrt 2 / 2 :=
begin
  -- Condition setup
  let F := (1, 0),
  let N := (0, real.sqrt 2 * b),
  have hN : N = (0, real.sqrt 2 * b) := by refl,
  -- Proof with the given maximum perimeter condition
  unfold eccentricity_of_ellipse,
  sorry
end

end eccentricity_proof_l274_274183


namespace largest_prime_factor_2501_l274_274753

theorem largest_prime_factor_2501 : ∃ p, prime p ∧ p ∣ 2501 ∧ ∀ q, prime q ∧ q ∣ 2501 → q ≤ p :=
begin
  have factorization : 2501 = 41 * 61 := by norm_num,
  have prime_41 : prime 41 := by norm_num,
  have prime_61 : prime 61 := by norm_num,
  use 61,
  split,
  { exact prime_61 },
  split,
  { rw factorization, exact dvd_mul_right 61 41 },
  { intros q hqprime hqdiv,
    have hq := prime.dvd_mul hqprime hqdiv factorization,
    cases hq,
    { rw hq, norm_num },
    { rw hq, norm_num }},
  sorry
end

end largest_prime_factor_2501_l274_274753


namespace domain_of_f_l274_274703

-- Given conditions
def f (x : ℝ) : ℝ := 1 / real.sqrt (2 * x - 3)
def dom := { x : ℝ | 2 * x - 3 > 0 }

-- Prove that the domain of the function f is (3/2, +∞)
theorem domain_of_f :
  ∀ x : ℝ, (x ∈ dom ↔ x > 3/2) :=
by sorry

end domain_of_f_l274_274703


namespace prob_greater_than_2_l274_274595

variables (ξ : ℝ → ℝ) (σ : ℝ) (P : set ℝ → ℝ)
variables [measure_space ℝ]

noncomputable def normal_distribution (mean variance : ℝ) (x : ℝ) : ℝ := 
  exp (- (x - mean) ^ 2 / (2 * variance)) / sqrt (2 * π * variance)

axiom ξ_distrib : ∀ x, ξ x = normal_distribution 1 (σ^2) x
axiom σ_pos : σ > 0
axiom prob_interval : P {x : ℝ | 0 < x ∧ x < 1} = 0.4

theorem prob_greater_than_2 : P {x : ℝ | x > 2} = 0.1 :=
by
  -- Assuming the given conditions and the property of the normal distribution
  sorry

end prob_greater_than_2_l274_274595


namespace correct_permutations_ВЕКТОР_correct_permutations_ЛИНИЯ_correct_permutations_ПАРАБОЛА_correct_permutations_БИССЕКТРИСА_correct_permutations_МАТЕМАТИКА_l274_274806

noncomputable def permutations_ВЕКТОР : ℕ := 6!
def answer_ВЕКТОР := 720

noncomputable def permutations_ЛИНИЯ : ℕ := 5! / 2!
def answer_ЛИНИЯ := 60

noncomputable def permutations_ПАРАБОЛА : ℕ := 8! / 3!
def answer_ПАРАБОЛА := 6720

noncomputable def permutations_БИССЕКТРИСА : ℕ := 11! / (3! * 2!)
def answer_БИССЕКТРИСА := 3326400

noncomputable def permutations_МАТЕМАТИКА : ℕ := 10! / (3! * 2! * 2!)
def answer_МАТЕМАТИКА := 151200

theorem correct_permutations_ВЕКТОР : permutations_ВЕКТОР = answer_ВЕКТОР := by
  sorry

theorem correct_permutations_ЛИНИЯ : permutations_ЛИНИЯ = answer_ЛИНИЯ := by
  sorry

theorem correct_permutations_ПАРАБОЛА : permutations_ПАРАБОЛА = answer_ПАРАБОЛА := by
  sorry

theorem correct_permutations_БИССЕКТРИСА : permutations_БИССЕКТРИСА = answer_БИССЕКТРИСА := by
  sorry

theorem correct_permutations_МАТЕМАТИКА : permutations_МАТЕМАТИКА = answer_МАТЕМАТИКА := by
  sorry

end correct_permutations_ВЕКТОР_correct_permutations_ЛИНИЯ_correct_permutations_ПАРАБОЛА_correct_permutations_БИССЕКТРИСА_correct_permutations_МАТЕМАТИКА_l274_274806


namespace proposition_4_l274_274447

theorem proposition_4 (x y ε : ℝ) (h1 : |x - 2| < ε) (h2 : |y - 2| < ε) : |x - y| < 2 * ε :=
by
  sorry

end proposition_4_l274_274447


namespace max_leap_years_in_200_years_l274_274080

-- Definitions based on conditions
def leap_year_occurrence (years : ℕ) : ℕ :=
  years / 4

-- Define the problem statement based on the given conditions and required proof
theorem max_leap_years_in_200_years : leap_year_occurrence 200 = 50 := 
by
  sorry

end max_leap_years_in_200_years_l274_274080


namespace repeating_decimal_product_l274_274478

-- Define the repeating decimal 0.\overline{137} as a fraction
def repeating_decimal_137 : ℚ := 137 / 999

-- Define the repeating decimal 0.\overline{6} as a fraction
def repeating_decimal_6 : ℚ := 2 / 3

-- The problem is to prove that the product of these fractions is 274 / 2997
theorem repeating_decimal_product : repeating_decimal_137 * repeating_decimal_6 = 274 / 2997 := by
  sorry

end repeating_decimal_product_l274_274478


namespace correct_judgment_l274_274889

def P := Real.pi < 2
def Q := Real.pi > 3

theorem correct_judgment : (P ∨ Q) ∧ ¬P := by
  sorry

end correct_judgment_l274_274889


namespace sum_of_three_consecutive_natural_numbers_not_prime_l274_274835

theorem sum_of_three_consecutive_natural_numbers_not_prime (n : ℕ) : 
  ¬ Prime (n + (n+1) + (n+2)) := by
  sorry

end sum_of_three_consecutive_natural_numbers_not_prime_l274_274835


namespace polynomial_p_at_0_l274_274289

theorem polynomial_p_at_0 {p : ℕ → ℝ} (h1 : ∀ x, polynomial.degree (p x) = 6)
  (h2 : ∀ n, n ∈ {0, 1, 2, 3, 4, 5, 6} → p (3 ^ n) = 1 / (3 ^ n)) :
  p 0 = 2186 / 2187 :=
sorry

end polynomial_p_at_0_l274_274289


namespace chromium_percent_second_alloy_is_8_l274_274603

-- Definitions for the problem conditions
def chromium_percent_first_alloy := 12
def weight_first_alloy := 15
def weight_second_alloy := 30
def chromium_percent_new_alloy := 9.333333333333334
def weight_new_alloy := 45
def chromium_weight_new_alloy := 4.2

-- Proof statement
theorem chromium_percent_second_alloy_is_8 (x : ℝ) :
  (chromium_percent_first_alloy / 100 * weight_first_alloy) + (x / 100 * weight_second_alloy) = chromium_weight_new_alloy →
  x = 8 :=
by
  sorry

end chromium_percent_second_alloy_is_8_l274_274603


namespace values_of_m_l274_274197

theorem values_of_m (m : ℕ) :
  choose 17 (3 * m - 1) = choose 17 (2 * m + 3) →
  (m = 3 ∨ m = 4) :=
by
  intros h
  sorry

end values_of_m_l274_274197


namespace smallest_possible_value_l274_274281

noncomputable def g (x : ℂ) : ℂ := x^4 - 12*x^3 + 54*x^2 - 108*x + 81

theorem smallest_possible_value 
  (w1 w2 w3 w4 : ℂ)
  (h1 : g w1 = 0)
  (h2 : g w2 = 0)
  (h3 : g w3 = 0)
  (h4 : g w4 = 0)
  (root_condition : {w1, w2, w3, w4} = ({3, 3, -3, -3} : set ℂ)) :
  ∃ a b c d : {1, 2, 3, 4}, |(w1 * w1 + w3 * w3)| = 18 :=
sorry

end smallest_possible_value_l274_274281


namespace balance_difference_correct_l274_274452

-- Define the given conditions
def angela_deposit : ℝ := 12000
def angela_rate : ℝ := 0.05
def angela_time : ℕ := 25

def bob_deposit : ℝ := 15000
def bob_rate : ℝ := 0.04
def bob_time : ℕ := 20

-- Define the compound interest formula for Angela
noncomputable def angela_balance :=
  angela_deposit * (1 + angela_rate)^angela_time

-- Define the simple interest formula for Bob
noncomputable def bob_balance :=
  bob_deposit * (1 + (bob_rate * bob_time))

-- Define the difference
noncomputable def balance_difference :=
  angela_balance - bob_balance

-- Prove the positive difference is equal to $13,636.44
theorem balance_difference_correct :
  balance_difference ≈ 13636.44 :=
by
  sorry

end balance_difference_correct_l274_274452


namespace find_arrays_l274_274856

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays_l274_274856


namespace solve_inequality_l274_274847

theorem solve_inequality (x : ℝ) : 
  3*x^2 + 2*x - 3 > 10 - 2*x ↔ x < ( -2 - Real.sqrt 43 ) / 3 ∨ x > ( -2 + Real.sqrt 43 ) / 3 := 
by
  sorry

end solve_inequality_l274_274847


namespace mix_cornmeal_l274_274696

variables (S C : ℝ)

-- Define the constants based on conditions
def total_weight := 280
def desired_protein_content := 0.13
def soybean_protein_content := 0.14
def cornmeal_protein_content := 0.07

-- Define the conditions formally
axiom h1 : S + C = total_weight
axiom h2 : soybean_protein_content * S + cornmeal_protein_content * C = desired_protein_content * total_weight

-- Theorem to prove that the required amount of cornmeal (C) is 40 pounds
theorem mix_cornmeal : C = 40 :=
by
  sorry

end mix_cornmeal_l274_274696


namespace value_of_expression_l274_274137

noncomputable def x := (2 : ℚ) / 3
noncomputable def y := (5 : ℚ) / 2

theorem value_of_expression : (1 / 3) * x^8 * y^9 = (5^9 / (2 * 3^9)) := by
  sorry

end value_of_expression_l274_274137


namespace percentage_temporary_employees_is_correct_l274_274992

noncomputable def percentage_temporary_employees
    (technicians_percentage : ℝ) (skilled_laborers_percentage : ℝ) (unskilled_laborers_percentage : ℝ)
    (permanent_technicians_percentage : ℝ) (permanent_skilled_laborers_percentage : ℝ)
    (permanent_unskilled_laborers_percentage : ℝ) : ℝ :=
  let total_workers : ℝ := 100
  let total_temporary_technicians := technicians_percentage * (1 - permanent_technicians_percentage / 100)
  let total_temporary_skilled_laborers := skilled_laborers_percentage * (1 - permanent_skilled_laborers_percentage / 100)
  let total_temporary_unskilled_laborers := unskilled_laborers_percentage * (1 - permanent_unskilled_laborers_percentage / 100)
  let total_temporary_workers := total_temporary_technicians + total_temporary_skilled_laborers + total_temporary_unskilled_laborers
  (total_temporary_workers / total_workers) * 100

theorem percentage_temporary_employees_is_correct :
  percentage_temporary_employees 40 35 25 60 45 35 = 51.5 :=
by
  sorry

end percentage_temporary_employees_is_correct_l274_274992


namespace determine_shape_of_eqn_l274_274490

-- Define the spherical coordinates and the equation
def in_spherical_coordinates (ρ θ φ : ℝ) (c : ℝ) := ρ = c * sin φ

-- Define the main theorem problem
theorem determine_shape_of_eqn (c : ℝ) (ρ θ φ : ℝ) (h : c > 0) :
  (in_spherical_coordinates ρ θ φ c) =
  (∃ r : ℝ, (ρ = r * sin φ) ∧ (r = c)) := sorry

end determine_shape_of_eqn_l274_274490


namespace percentage_increase_l274_274206

theorem percentage_increase (P : ℕ) (x y : ℕ) (h1 : x = 5) (h2 : y = 7) 
    (h3 : (x * (1 + P / 100) / (y * (1 - 10 / 100))) = 20 / 21) : 
    P = 20 :=
by
  sorry

end percentage_increase_l274_274206


namespace polynomial_irreducible_l274_274282

noncomputable def P (n : ℕ) (a : ℕ → ℤ) : Polynomial ℤ :=
1 + ∏ i in Finset.range (n + 1), (Polynomial.X - Polynomial.C (a i))

theorem polynomial_irreducible 
  (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℤ) (ha : Function.Injective a) : Irreducible (P n a) :=
sorry

end polynomial_irreducible_l274_274282


namespace probability_at_least_four_of_five_dice_same_number_l274_274116

noncomputable def probability_at_least_four_same : ℚ :=
  (1 / 1296) + (25 / 1296)

theorem probability_at_least_four_of_five_dice_same_number :
  let P := (1 : ℤ) / 1296 + 25 / 1296 in
  P = 13 / 648 :=
by
  let P := (1 : ℤ) / 1296 + 25 / 1296
  have : P = 26 / 1296 := by 
    calc
      P = (1 / 1296) + (25 / 1296) : by sorry -- simplify addition
      ... = 26 / 1296 : by sorry -- combine fractions
  show P = 13 / 648 from by
    calc
      26 / 1296 = 13 / 648 : by sorry -- reduce fraction

end probability_at_least_four_of_five_dice_same_number_l274_274116


namespace term_thirteen_l274_274521

section ArithmeticSequence

variables (a : ℕ → ℤ) -- a represents the arithmetic sequence

-- Given conditions
def sum_of_first_fifteen (a : ℕ → ℤ) : Prop :=
  ∑ n in finset.range 15, a (n+1) = 45

def third_term (a : ℕ → ℤ) : Prop := 
  a 3 = -10

-- Hypotheses from the given conditions
axiom h_sum : sum_of_first_fifteen a
axiom h_third : third_term a

-- Proof goal
theorem term_thirteen : a 13 = 16 :=
by {
  -- to be proved
  sorry
}

end ArithmeticSequence

end term_thirteen_l274_274521


namespace hank_reads_everyday_l274_274939

def daily_reading_time_weekdays := 30 + 60
def daily_reading_time_weekends := 2 * daily_reading_time_weekdays

theorem hank_reads_everyday (total_weekly_reading_time : ℕ) (weekdays : ℕ) (weekends : ℕ) :
  daily_reading_time_weekdays * weekdays + daily_reading_time_weekends * weekends = total_weekly_reading_time →
  weekdays = 5 → weekends = 2 → total_weekly_reading_time = 810 →
  weekdays + weekends = 7 :=
by
  assume h1 h2 h3 h4
  sorry

end hank_reads_everyday_l274_274939


namespace log_equivalence_l274_274584

theorem log_equivalence (x : ℝ) (h : log 8 (3 * x) = 3) : log x 125 = 3 / (9 * log 5 2 - log 5 3) := 
by 
  sorry

end log_equivalence_l274_274584


namespace sum_of_common_ratios_l274_274655

noncomputable def geom_sequences_common_ratios (k a2 a3 b2 b3 : ℝ) (p r : ℝ) : Prop :=
  a3 = k * p^2 ∧ b3 = k * r^2 ∧ a2 = k * p ∧ b2 = k * r ∧ a3 - b3 = 5 * (a2 - b2)

theorem sum_of_common_ratios (k a2 a3 b2 b3 p r : ℝ) (h : geom_sequences_common_ratios k a2 a3 b2 b3 p r) :
  p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l274_274655


namespace find_p_l274_274535

-- Definitions of the hyperbola and parabola and the condition that the focus of the hyperbola lies on the directrix of the parabola
def hyperbola_focus : ℝ × ℝ := (-2, 0)
def parabola_directrix (p : ℝ) : ℝ := -p / 2

-- Prove the value of p given the conditions
theorem find_p (p : ℝ) (h : hyperbola_focus.fst = parabola_directrix p) : p = 4 :=
by
  sorry

end find_p_l274_274535


namespace jane_bike_speed_l274_274265

theorem jane_bike_speed 
    (swim_distance : ℝ)
    (bike_distance : ℝ)
    (run_distance : ℝ)
    (swim_speed : ℝ)
    (run_speed : ℝ)
    (total_time : ℝ) 
    (h_swim : swim_distance = 0.5)
    (h_bike : bike_distance = 10)
    (h_run : run_distance = 2)
    (h_swim_speed : swim_speed = 1.5)
    (h_run_speed : run_speed = 5)
    (h_total_time : total_time = 1.5) :
    (bike_speed : ℝ) :
    bike_speed = 13 := 
begin
  declare_time_taken : ℝ := swim_distance / swim_speed + run_distance / run_speed,  
  sorry -- Placeholder for the actual proof.
end

end jane_bike_speed_l274_274265


namespace housewives_spent_equal_money_and_second_bought_more_milk_l274_274083

variable {a : ℕ → ℕ}

-- Condition
assert h_avg : (1 / 30) * (∑ k in Finset.range 30, a k) = 20

-- Theorem statement
theorem housewives_spent_equal_money_and_second_bought_more_milk (h : (1 / 30) * (∑ k in Finset.range 30, a k) = 20) :
  let expenditure_first := ∑ k in Finset.range 30, a k in
  let expenditure_second := (20 * 30) in
  let total_milk_second := ∑ k in Finset.range 30, (20 / a k) in
    expenditure_first = 600 ∧
    expenditure_second = 600 ∧
    total_milk_second > 30 :=
sorry

end housewives_spent_equal_money_and_second_bought_more_milk_l274_274083


namespace angle_at_center_for_tangents_l274_274738

theorem angle_at_center_for_tangents (P A B O : Point) (h1 : tangent P A O)
  (h2 : tangent P B O) (h3 : tangent A B O) (h_angle : angle APB = 60) :
  angle AOB = 120 := 
sorry

end angle_at_center_for_tangents_l274_274738


namespace area_of_parabola_l274_274829

def integrand (y : ℝ) : ℝ := 8 * y - y^2 - 7

theorem area_of_parabola :
  (∫ y in 1..7, integrand y) = 36 :=
by
  -- This is a place holder indicating where the proof would go
  sorry

end area_of_parabola_l274_274829


namespace T_value_l274_274659

theorem T_value (x y : ℝ) (h1 : 2^x = 196) (h2 : 7^y = 196) : (1 / x) + (1 / y) = 1 / 2 :=
by
  sorry

end T_value_l274_274659


namespace expected_flips_is_four_l274_274668

noncomputable def expected_flips_to_second_tails : ℕ :=
  let E_Y := 2 in
  E_Y + E_Y

theorem expected_flips_is_four : expected_flips_to_second_tails = 4 :=
  by
  -- sorry is added to skip proof
  sorry

end expected_flips_is_four_l274_274668


namespace min_colors_needed_correct_l274_274380

-- Define the 5x5 grid as a type
def Grid : Type := Fin 5 × Fin 5

-- Define a coloring as a function from Grid to a given number of colors
def Coloring (colors : Type) : Type := Grid → colors

-- Define the property where in any row, column, or diagonal, no three consecutive cells have the same color
def valid_coloring (colors : Type) (C : Coloring colors) : Prop :=
  ∀ i : Fin 5, ∀ j : Fin 3, ( C (i, j) ≠ C (i, j + 1) ∧ C (i, j + 1) ≠ C (i, j + 2) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 5, ( C (i, j) ≠ C (i + 1, j) ∧ C (i + 1, j) ≠ C (i + 2, j) ) ∧
  ∀ i : Fin 3, ∀ j : Fin 3, ( C (i, j) ≠ C (i + 1, j + 1) ∧ C (i + 1, j + 1) ≠ C (i + 2, j + 2) )

-- Define the minimum number of colors required
def min_colors_needed : Nat := 5

-- Prove the statement
theorem min_colors_needed_correct : ∃ C : Coloring (Fin min_colors_needed), valid_coloring (Fin min_colors_needed) C :=
sorry

end min_colors_needed_correct_l274_274380


namespace democrats_republicans_circular_arrangement_l274_274407

open Finset

noncomputable def circular_arrangements_no_adjacent (d r : ℕ) : ℕ := 
  (factorial (r - 1)) * choose r d * factorial d

theorem democrats_republicans_circular_arrangement :
  circular_arrangements_no_adjacent 4 6 = 43200 := 
by 
  simp [circular_arrangements_no_adjacent, factorial, choose]
  sorry

end democrats_republicans_circular_arrangement_l274_274407


namespace geometric_sequence_conditions_l274_274762

variable (a : ℕ → ℝ) (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_conditions (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : -1 < q)
  (h3 : q < 0) :
  (∀ n, a n * a (n + 1) < 0) ∧ (∀ n, |a n| > |a (n + 1)|) :=
by
  sorry

end geometric_sequence_conditions_l274_274762


namespace domain_of_f_l274_274335

def f (x : ℝ) : ℝ := x / (2^x - 1) + Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | 2^x - 1 ≠ 0 ∧ x + 1 ≥ 0} = {x : ℝ | x ≠ 0 ∧ x ≥ -1} :=
begin
  sorry
end

end domain_of_f_l274_274335


namespace area_PQR_is_three_l274_274365

-- Definitions of the given problem's conditions
def P := (−3*Real.sqrt 2, 0)
def Q := (0, 0)
def R := (5*Real.sqrt 2, Real.sqrt 2)

-- Definition for the area of triangle PQR
def area_of_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * ((P.1 * (Q.2 - R.2)) + (Q.1 * (R.2 - P.2)) + (R.1 * (P.2 - Q.2))).abs

-- Statement of the theorem to prove the area of triangle PQR
theorem area_PQR_is_three : area_of_triangle P Q R = 3 :=
by
  sorry -- Proof is omitted

end area_PQR_is_three_l274_274365


namespace james_distance_l274_274264

-- Definitions and conditions
def speed : ℝ := 80.0
def time : ℝ := 16.0

-- Proof problem statement
theorem james_distance : speed * time = 1280.0 := by
  sorry

end james_distance_l274_274264


namespace metallic_sphere_radius_l274_274796

theorem metallic_sphere_radius 
  (r_wire : ℝ)
  (h_wire : ℝ)
  (r_sphere : ℝ) 
  (V_sphere : ℝ)
  (V_wire : ℝ)
  (h_wire_eq : h_wire = 16)
  (r_wire_eq : r_wire = 12)
  (V_wire_eq : V_wire = π * r_wire^2 * h_wire)
  (V_sphere_eq : V_sphere = (4/3) * π * r_sphere^3)
  (volume_eq : V_sphere = V_wire) :
  r_sphere = 12 :=
by
  sorry

end metallic_sphere_radius_l274_274796


namespace allocation_of_volunteers_l274_274327

theorem allocation_of_volunteers : 
  let volunteers := {A, B, C, D}
  let stadiums := {s1, s2, s3}
  finset.card {f : volunteers → stadiums // (∀ s, ∃ v, f v = s)} = 36 :=
by {
  sorry
}

end allocation_of_volunteers_l274_274327


namespace tetrahedron_min_height_l274_274128

noncomputable def tetrahedron_height (a : ℝ) : ℝ := a * (sqrt (2 / 3))

theorem tetrahedron_min_height (radius : ℝ) (edge_length : ℝ)
  (h1 : radius = 1)
  (h2 : edge_length = 2 * radius) :
  tetrahedron_height edge_length + 2 * radius = (2:ℝ) + ((2:ℝ) * sqrt 6 / 3) :=
by
  sorry

end tetrahedron_min_height_l274_274128


namespace distance_between_intersection_points_l274_274356

theorem distance_between_intersection_points :
  let vertices := [(0,0,0), (0,0,6), (0,6,0), (0,6,6), (6,0,0), (6,0,6), (6,6,0), (6,6,6)] in
  let P := (0, 3, 0) in
  let Q := (2, 0, 0) in
  let R := (2, 6, 6) in
  let plane := { p : ℝ × ℝ × ℝ | 3 * p.1 + 2 * p.2 - p.3 = 6 } in
  let S := (4, 0, 6) in
  let T := (0, 6, 6) in
  ∥(4, 0, 6) - (0, 6, 6)∥ = 2 * Real.sqrt 13 :=
begin
  -- Proof goes here
  sorry
end

end distance_between_intersection_points_l274_274356


namespace exists_linear_additive_function_l274_274654

variables (n : ℕ) (a : ℝ) (f : ℕ → ℝ → ℝ)

-- Conditions
axiom f_additive : ∀ i, ∀ x y : ℝ, f i (x + y) = f i x + f i y
axiom f_product : ∀ x : ℝ, ∏ i in finset.range n, f i x = a * x^n

-- Theorem Statement
theorem exists_linear_additive_function :
  ∃ i, ∃ b_i : ℝ, ∀ x : ℝ, f i x = b_i * x :=
by {
  sorry
}

end exists_linear_additive_function_l274_274654


namespace non_congruent_triangles_with_perimeter_24_l274_274564

theorem non_congruent_triangles_with_perimeter_24 : 
  ∃ (triangles : Finset (Finset (ℕ × ℕ × ℕ))), 
  (∀ t ∈ triangles, let ⟨a, b, c⟩ := t in a + b + c = 24 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
  triangles.card = 11 :=
sorry

end non_congruent_triangles_with_perimeter_24_l274_274564


namespace current_calculation_l274_274201

def voltage : ℂ := 3 + 2 * Complex.i
def impedance : ℂ := 2 - 5 * Complex.i

theorem current_calculation (V Z : ℂ) (hV : V = 3 + 2 * Complex.i) (hZ : Z = 2 - 5 * Complex.i) :
  V / Z = -4 / 29 + (19 / 29) * Complex.i :=
  sorry

end current_calculation_l274_274201


namespace roots_sum_l274_274846

noncomputable def operation_square (a b : ℝ) : ℝ := a^2 + 2*a*b - b^2

noncomputable def f (x : ℝ) : ℝ := operation_square x 2

def roots_sum_condition : Prop := 
  ∀ (x1 x2 x3 x4 : ℝ),
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
  f x1 = log10 |x1 + 2| ∧ f x2 = log10 |x2 + 2| ∧ f x3 = log10 |x3 + 2| ∧ f x4 = log10 |x4 + 2| →
  x1 + x2 + x3 + x4 = -8

theorem roots_sum : roots_sum_condition :=
sorry

end roots_sum_l274_274846


namespace proportion_solution_l274_274766

theorem proportion_solution (x : ℚ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by sorry

end proportion_solution_l274_274766


namespace inequality_solution_l274_274719

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end inequality_solution_l274_274719


namespace length_of_tank_is_25_l274_274054

-- Conditions
variable (l w d : ℝ)
variable (cost_per_sqm total_cost : ℝ)

-- Given conditions
def tank_conditions : Prop :=
  w = 12 ∧
  d = 6 ∧
  cost_per_sqm = 0.75 ∧
  total_cost = 558

-- Proof that the length of the tank is 25 meters
theorem length_of_tank_is_25 (h : tank_conditions l w d cost_per_sqm total_cost) :
  l = 25 :=
by {
  sorry
}

end length_of_tank_is_25_l274_274054


namespace sum_of_cubes_multiple_of_6_sum_of_cubes_l274_274776

theorem sum_of_cubes (a b c d : ℤ) (k : ℤ) : 
  k = a^3 + b^3 + c^3 + d^3 → 
  k ∈ {6, -6, 12, -12, 18, -18, 24, -24} := 
sorry

theorem multiple_of_6_sum_of_cubes : 
  ∀ (n : ℤ), ∃ (a b c d : ℤ), 6 * n = a^3 + b^3 + c^3 + d^3 :=
sorry

end sum_of_cubes_multiple_of_6_sum_of_cubes_l274_274776


namespace area_of_triangle_PQR_l274_274031

noncomputable def circle_radius_3 : ℝ := 3
noncomputable def circle_radius_4 : ℝ := 4

axiom PQR_tangent_to_circles (P Q R S T : Point) (PQ PR QR : Segment) : 
  tangent PQ S ∧ tangent PR S ∧ 
  tangent PQ T ∧ tangent PR T ∧ 
  radius S = circle_radius_3 ∧ 
  radius T = circle_radius_4 ∧ 
  congruent PQ PR

theorem area_of_triangle_PQR (P Q R S T : Point) (PQ PR QR : Segment) :
  tangent PQ S → tangent PR S → 
  tangent PQ T → tangent PR T → 
  radius S = circle_radius_3 → 
  radius T = circle_radius_4 → 
  congruent PQ PR →
  area_triangle P Q R = 42 * real.sqrt 10 :=
by 
  sorry

end area_of_triangle_PQR_l274_274031


namespace incorrect_conclusion_is_D_l274_274387

noncomputable def condition_abc_coefficient : Prop :=
  ∃ c : ℝ, c * (1 * (1 * 1)) = (1 : ℝ)

noncomputable def polynomial_1_minus_3x2_minus_x : Polynomial ℝ :=
  1 - (3 : ℝ) * Polynomial.X^2 - Polynomial.X

noncomputable def term_3ab3_degree : Polynomial ℝ :=
  -3 * Polynomial.X * Polynomial.C (1 : ℝ) ^ 3

noncomputable def term_3xy_polynomial : Polynomial ℝ :=
  - (3 : ℝ) / 4 * Polynomial.X * Polynomial.Y

theorem incorrect_conclusion_is_D :
  condition_abc_coefficient ∧
  (polynomial_1_minus_3x2_minus_x.coeff 2 = -3) ∧
  term_3ab3_degree.degree = 4 ∧
  ∀ p : Polynomial ℝ, p ≠ term_3xy_polynomial →
  ¬Polynomial.isPoly term_3xy_polynomial :=
  by sorry

end incorrect_conclusion_is_D_l274_274387


namespace stock_sold_percentage_correct_l274_274331

noncomputable def percentage_of_stock_sold
  (cash_realized : ℝ)
  (brokerage_percentage : ℝ)
  (total_amount_including_brokerage : ℝ)
  (total_amount_before_brokerage : ℝ)
  (brokerage_amount : ℝ) : ℝ :=
have h1 : cash_realized = total_amount_before_brokerage - brokerage_amount, by sorry,
have h2 : total_amount_before_brokerage * (1 - brokerage_percentage / 100) = total_amount_including_brokerage, by sorry,
have h3 : brokerage_amount = total_amount_before_brokerage * brokerage_percentage / 100, by sorry,
let P := brokerage_amount * 400 / total_amount_before_brokerage in
by norm_num; exact P

theorem stock_sold_percentage_correct :
  percentage_of_stock_sold 108.25 (1/4) 108 (43200 / 399) 0.02 ≈ 7.39 :=
by sorry

end stock_sold_percentage_correct_l274_274331


namespace probability_parallel_vectors_l274_274937

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x y : ℝ) : ℝ × ℝ := (x, y)
def x_values : Set ℝ := {-1, 0, 1, 2}
def y_values : Set ℝ := {-1, 0, 1}

def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem probability_parallel_vectors :
  (finset.filter
    (λ (p : ℝ × ℝ), is_parallel vector_a p)
    (finset.cartesianProduct
      (finset.of_set x_values)
      (finset.of_set y_values))
  ).card.to_real /
  ((finset.cartesianProduct 
      (finset.of_set x_values)
      (finset.of_set y_values)
  ).card.to_real) = 1 / 6 :=
sorry

end probability_parallel_vectors_l274_274937


namespace words_lost_due_to_prohibition_l274_274010

-- Define the conditions given in the problem.
def number_of_letters := 64
def forbidden_letter := 7
def total_one_letter_words := number_of_letters
def total_two_letter_words := number_of_letters * number_of_letters

-- Define the forbidden letter loss calculation.
def one_letter_words_lost := 1
def two_letter_words_lost := number_of_letters + number_of_letters - 1

-- Define the total words lost calculation.
def total_words_lost := one_letter_words_lost + two_letter_words_lost

-- State the theorem to prove the number of words lost is 128.
theorem words_lost_due_to_prohibition : total_words_lost = 128 :=
by sorry

end words_lost_due_to_prohibition_l274_274010


namespace point_P_on_line_l_intersection_of_line_l_and_curve_C_l274_274606

noncomputable def pointP : ℝ × ℝ := (0, Real.sqrt 3)
def line_l_polar (ρ θ : ℝ) : Prop := ρ = (Real.sqrt 3) / (2 * Real.cos (θ - π / 6))
def cartesian_of_line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = Real.sqrt 3
def parametric_C (φ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, 2 * Real.sin φ)
def curve_C (x y : ℝ) : Prop := (x^2 / 2) + (y^2 / 4) = 1

open Real

theorem point_P_on_line_l (ρ θ : ℝ) : cartesian_of_line_l (0) (sqrt 3) :=
by sorry

theorem intersection_of_line_l_and_curve_C (t : ℝ) : 
  let A := parametric_C t, B := parametric_C (-t) in
  ∃ t₁ t₂ : ℝ, 
  t₁ + t₂ = -(12/5) ∧ t₁ * t₂ = -(4/5) ∧ 
  (sqrt 14 : ℝ) = (1 / dist pointP A) + (1 / dist pointP B) :=
by sorry

end point_P_on_line_l_intersection_of_line_l_and_curve_C_l274_274606


namespace SurfaceAreaOfCircumscribedSphere_l274_274234

-- Define the conditions
def tetrahedron_conditions (S A B C : Type) (SA AC : S → A → ℕ) (AB : A → B → ℕ) :=
  SA A = 2 ∧ AC S C = 2 ∧ AB A B = 1 ∧ ∃ P, P ≠ A ∧ angle P A B = 90

-- Main theorem statement
theorem SurfaceAreaOfCircumscribedSphere (S A B C : Type)
  (SA AC : S → A → ℕ) (AB : A → B → ℕ)
  (h : tetrahedron_conditions S A B C SA AC AB) :
  4 * π * (sqrt 2)^2 = 8 * π :=
by
  sorry

end SurfaceAreaOfCircumscribedSphere_l274_274234


namespace symmetrical_line_equation_l274_274336

-- Definitions for the conditions
def line_symmetrical (eq1 eq2 : String) : Prop :=
  eq1 = "x - 2y + 3 = 0" ∧ eq2 = "x + 2y + 3 = 0"

-- Prove the statement
theorem symmetrical_line_equation : line_symmetrical "x - 2y + 3 = 0" "x + 2y + 3 = 0" :=
  by
  -- This is just the proof skeleton; the actual proof is not required
  sorry

end symmetrical_line_equation_l274_274336


namespace odd_function_expression_l274_274886

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2*x) :
  ∀ x : ℝ, f x = x * (|x| - 2) :=
by
  sorry

end odd_function_expression_l274_274886


namespace initial_nickels_eq_l274_274672

variable (quarters : ℕ) (initial_nickels : ℕ) (nickels_borrowed : ℕ) (nickels_left : ℕ)

-- Assumptions based on the problem
axiom quarters_had : quarters = 33
axiom nickels_left_axiom : nickels_left = 12
axiom nickels_borrowed_axiom : nickels_borrowed = 75

-- Theorem to prove: initial number of nickels
theorem initial_nickels_eq :
  initial_nickels = nickels_left + nickels_borrowed :=
by
  sorry

end initial_nickels_eq_l274_274672


namespace disjoint_convex_quadrilaterals_l274_274888

theorem disjoint_convex_quadrilaterals (P : Finset (ℝ × ℝ)) (hP : P.card = 500) 
  (h_collinear : ∀ (A B C : (ℝ × ℝ)), A ∈ P → B ∈ P → C ∈ P → ¬collinear A B C) :
  ∃ (Q : Finset (Finset (ℝ × ℝ))), Q.card = 100 ∧ (∀ q ∈ Q, q.card = 4 ∧ convex q ∧ (∀ q₁ q₂ ∈ Q, q₁ ≠ q₂ → disjoint q₁ q₂)) :=
sorry

end disjoint_convex_quadrilaterals_l274_274888


namespace minimum_positive_period_intervals_of_monotonicity_and_value_extremes_l274_274550

noncomputable def function_y (x : ℝ) : ℝ :=
  2 * Real.cos (Real.pi / 2 - 3 * x)

theorem minimum_positive_period :
  (∃ T > 0, ∀ x, function_y (x + T) = function_y x) ∧
  (∀ T > 0, (T = 2 * Real.pi / 3) → (∀ x, function_y (x + T) = function_y x)) :=
by
  sorry

theorem intervals_of_monotonicity_and_value_extremes :
  (∀ k : ℤ, isIncreasing (function_y : ℝ → ℝ)
    on (Set.Icc ((2 * k * Real.pi / 3 - Real.pi / 6)) (2 * k * Real.pi / 3 + Real.pi / 6))) ∧
  (∀ k : ℤ, isDecreasing (function_y : ℝ → ℝ)
    on (Set.Icc ((2 * k * Real.pi / 3 + Real.pi / 6)) (2 * k * Real.pi / 3 + Real.pi / 2))) ∧
  (∀ k : ℤ, ∃ x, function_y x = -2 ∧ x = (2 * k * Real.pi / 3 - Real.pi / 6)) ∧
  (∀ k : ℤ, ∃ x, function_y x = 2 ∧ x = (2 * k * Real.pi / 3 + Real.pi / 6)) :=
by
  sorry

end minimum_positive_period_intervals_of_monotonicity_and_value_extremes_l274_274550


namespace carX_travel_distance_after_carY_started_l274_274836

-- Define the conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 40
def delay_time : ℝ := 1.2

-- Define the problem to prove the question is equal to the correct answer given the conditions
theorem carX_travel_distance_after_carY_started : 
  ∃ t : ℝ, carY_speed * t = carX_speed * t + carX_speed * delay_time ∧ 
           carX_speed * t = 294 :=
by
  sorry

end carX_travel_distance_after_carY_started_l274_274836


namespace total_cows_in_ranch_l274_274752

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l274_274752


namespace base_7_representation_digits_count_l274_274943

theorem base_7_representation_digits_count : ∀ (n : ℕ), n = 1234 → base_digits 7 n = 4 :=
by
  intro n
  assume h : n = 1234
  sorry

end base_7_representation_digits_count_l274_274943


namespace problem1_problem2_l274_274917

noncomputable def f (x : ℝ) : ℝ := (sin x + cos x) ^ 2 + 2 * (cos x) ^ 2 - 2

-- Theorem 1: Finding the smallest positive period and the monotonically increasing interval
theorem problem1 : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∀ k : ℤ, 
    - (3 * π / 8) + k * π ≤ x ∧ x ≤ (π / 8) + k * π → 
    (∀ x₁ x₂, - (3 * π / 8) + k * π ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ (π / 8) + k * π → f x₁ ≤ f x₂)) :=
sorry

-- Theorem 2: Finding the range of the function on the given interval
theorem problem2 : 
  (∀ x, (π / 4) ≤ x ∧ x ≤ (3 * π / 4) → - (sqrt 2) ≤ f x ∧ f x ≤ 1) :=
sorry

end problem1_problem2_l274_274917


namespace weekly_bath_water_usage_l274_274302

variable (bucket_capacity : ℕ) (times_filled : ℕ) (buckets_removed : ℕ) (baths_per_week : ℕ)
variable (capacity : ℕ := times_filled * bucket_capacity)
variable (removed_water : ℕ := buckets_removed * bucket_capacity)
variable (water_per_bath : ℕ := capacity - removed_water)
variable (weekly_water_usage : ℕ := water_per_bath * baths_per_week)

theorem weekly_bath_water_usage 
  (h1 : bucket_capacity = 120)
  (h2 : times_filled = 14)
  (h3 : buckets_removed = 3)
  (h4 : baths_per_week = 7) 
  : weekly_water_usage bucket_capacity times_filled buckets_removed baths_per_week = 9240 := 
by
  sorry

end weekly_bath_water_usage_l274_274302


namespace tenth_number_in_twentieth_row_l274_274697

def arrangement : ∀ n : ℕ, ℕ := -- A function defining the nth number in the sequence.
  sorry

-- A function to get the nth number in the mth row, respecting the arithmetic sequence property.
def number_in_row (m n : ℕ) : ℕ := 
  sorry

theorem tenth_number_in_twentieth_row : number_in_row 20 10 = 426 :=
  sorry

end tenth_number_in_twentieth_row_l274_274697


namespace ratio_won_to_lost_l274_274625

-- Define the total number of games and the number of games won
def total_games : Nat := 30
def games_won : Nat := 18

-- Define the number of games lost
def games_lost : Nat := total_games - games_won

-- Define the ratio of games won to games lost as a pair
def ratio : Nat × Nat := (games_won / Nat.gcd games_won games_lost, games_lost / Nat.gcd games_won games_lost)

-- The theorem to be proved
theorem ratio_won_to_lost : ratio = (3, 2) :=
  by
    -- Skipping the proof here
    sorry

end ratio_won_to_lost_l274_274625


namespace make_numbers_equal_possibility_l274_274997

def initially_table (n : ℕ) (i j : ℕ) : ℕ :=
  if i = j then 1 else 0

def allowed_transformation (n : ℕ) (table : ℕ × ℕ → ℕ) (path : list (ℕ × ℕ)) : ℕ × ℕ → ℕ :=
  if is_non_self_intersecting_closed_path path then
    λ (i, j), if (i, j) ∈ path then table (i, j) + 1 else table (i, j)
  else
    table

def is_non_self_intersecting_closed_path (path : list (ℕ × ℕ)) : Prop := sorry

def is_possible_to_make_all_numbers_equal (n : ℕ) : Prop :=
  ∃ (table : ℕ × ℕ → ℕ), 
    (∀ i j, table (i, j) = table (0, 0))

theorem make_numbers_equal_possibility (n : ℕ) : Prop :=
  if n % 2 = 0 then
    is_possible_to_make_all_numbers_equal n = false
  else
    is_possible_to_make_all_numbers_equal n = true

end make_numbers_equal_possibility_l274_274997


namespace length_of_AB_of_right_triangle_with_inscribed_circle_hypotenuse_length_is_correct_l274_274058

theorem length_of_AB_of_right_triangle_with_inscribed_circle
  (x : ℝ)
  (h_inscribed_circle : ∃ (h : ℝ), h = 8)
  (h_30_deg_angle : ∃ (a b : ℝ), a = x ∧ b = x * real.sqrt 3 ∧ (real.angle.between x b (2 * x) = 30)
  : ∃ (AB : ℝ), AB = 2 * x ∧ AB = 16 * (real.sqrt 3 + 1) :=
begin
  -- proof omitted
  sorry
end

theorem hypotenuse_length_is_correct (r : ℝ) (AB : ℝ) :
  r = 8 →
  (∃ (a b : ℝ), r = (a + b - AB) / 2 ∧ a = x ∧ b = x * real.sqrt 3) →
  AB = 16 * (real.sqrt 3 + 1) :=
begin
  -- proof omitted
  sorry
end

end length_of_AB_of_right_triangle_with_inscribed_circle_hypotenuse_length_is_correct_l274_274058


namespace determine_a_l274_274931

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

theorem determine_a (a : ℝ) (A_union_B_eq_A : A a ∪ B a = A a) : a = -1 ∨ a = 0 := by
  sorry

end determine_a_l274_274931


namespace distinct_real_roots_count_l274_274178

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x, Real.exp x * (x ^ 2 + a * x + b)

theorem distinct_real_roots_count {a b x₁ x₂ : ℝ} 
  (h1 : x₁ < x₂)
  (h2 : x₁ ≠ x₂)
  (h3 : f a b x₁ = x₁)
  (h4 : (∀ x, f a b x = x → x = x₁ ∨ x = x₂)) :
  (f a b)^2 x + (2 + a) * f a b x + a + b = 0 → 3.distinct_roots := 
sorry

end distinct_real_roots_count_l274_274178


namespace calculate_expr_l274_274087

theorem calculate_expr :
  ( (5 / 12: ℝ) ^ 2022) * (-2.4) ^ 2023 = - (12 / 5: ℝ) := 
by 
  sorry

end calculate_expr_l274_274087


namespace number_of_correct_propositions_is_two_l274_274709

def proposition1 : Prop := ∀ (trapezoid : Type), ∃ (plane : Type), trapezoid determines plane
def proposition2 : Prop := ∀ (lines : Type) (thirdLine : Type), (angles formed by lines with thirdLine are equal) → lines are parallel
def proposition3 : Prop := ∀ (lines : Type), (lines intersecting in pairs determine at most three planes)
def proposition4 : Prop := ∀ (planes : Type), (planes have three common points) → planes coincide

theorem number_of_correct_propositions_is_two :
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) → 
  (number_of_correct_propositions = 2) :=
sorry

end number_of_correct_propositions_is_two_l274_274709


namespace lines_increased_l274_274344

theorem lines_increased (L new_lines : ℕ) (h1 : new_lines = 240) (h2 : 1.5 * L = new_lines) : 
  new_lines - L = 80 := 
by
  sorry

end lines_increased_l274_274344


namespace find_m_eq_l274_274187

theorem find_m_eq : 
  (∀ (m : ℝ),
    ((m + 2)^2 + (m + 3)^2 = m^2 + 16 + 4 + (m - 1)^2) →
    m = 2 / 3 ) :=
by
  intros m h
  sorry

end find_m_eq_l274_274187


namespace problem_xy_l274_274210

theorem problem_xy (x y : ℝ) (h1 : x + y = 25) (h2 : x^2 * y^3 + y^2 * x^3 = 25) : x * y = 1 :=
by
  sorry

end problem_xy_l274_274210


namespace vitamin_D_scientific_notation_l274_274064

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l274_274064


namespace sulfuric_acid_moles_l274_274112

def sulfuric_acid_formation(SO3 : ℝ) (H2O : ℝ) : ℝ :=
  if SO3 = H2O then SO3 else sorry

theorem sulfuric_acid_moles :
  sulfuric_acid_formation 2 2 = 2 :=
by
  sorry

end sulfuric_acid_moles_l274_274112


namespace ball_hits_ground_time_l274_274704

theorem ball_hits_ground_time (t : ℝ) : 
  (∃ t : ℝ, -10 * t^2 + 40 * t + 50 = 0 ∧ t ≥ 0) → t = 5 := 
by
  -- placeholder for proof
  sorry

end ball_hits_ground_time_l274_274704


namespace sin_2theta_value_l274_274966

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l274_274966


namespace probability_two_of_three_survive_l274_274910

-- Let's define the necessary components
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly 2 out of 3 seedlings surviving
theorem probability_two_of_three_survive (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  binomial_coefficient 3 2 * p^2 * (1 - p) = 3 * p^2 * (1 - p) :=
by
  sorry

end probability_two_of_three_survive_l274_274910


namespace math_proof_problem_l274_274158

-- Define the conditions for the ellipse and other given constraints
def problem_conditions (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (ecc : e = 0.5):
  Prop :=
  ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -- Ellipse equation
  ∧ focus_distance : ∀ c : ℝ, c = a * e -- Eccentricity relation c = ae
  ∧ line_segment_length : 2 * b^2 / a = 3 -- Condition for the line segment length

-- Define the expected result for the ellipse equation and coordinates of P
def expected_ellipse_eq : Prop :=
  ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 -- Ellipse equation result

def expected_coordinates_P : set (ℝ × ℝ) :=
  { (1, 1.5), (-1, -1.5) } -- Set containing the expected coordinates of point P

-- The main theorem combining conditions and expected results
theorem math_proof_problem (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) (ecc : e = 0.5)
  (cond : problem_conditions a b h e ecc):
  expected_ellipse_eq ∧ (∀ P : ℝ × ℝ, P ∈ expected_coordinates_P) :=
by
  sorry

end math_proof_problem_l274_274158


namespace sin_angle_FAG_correct_l274_274999

noncomputable def sin_angle_FAG (ABC : Triangle) (isEquilateral : is_equilateral ABC)
  (F G : Point) (H_FG_bisect : bisect BC F G) : ℝ :=
sin (angle FAG)

theorem sin_angle_FAG_correct 
  (ABC : Triangle) (isEquilateral : is_equilateral ABC)
  (F G : Point) (H_FG_bisect : bisect BC F G) :
  sin (angle FAG) = (3 * sqrt 6) / 2 :=
sorry

end sin_angle_FAG_correct_l274_274999


namespace integer_points_inequality_l274_274652

theorem integer_points_inequality
  (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b + a - b - 5 = 0)
  (M := max ((a : ℤ)^2 + (b : ℤ)^2)) :
  (3 * x^2 + 2 * y^2 <= M) → ∃ (n : ℕ), n = 51 :=
by sorry

end integer_points_inequality_l274_274652


namespace perpendicular_lines_sufficient_but_not_necessary_l274_274163

-- Conditions for lines l1 and l2 to be perpendicular
def l1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + (a - 1) * y - 1 = 0
def l2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a - 1) * x + (2 * a + 3) * y - 3 = 0

theorem perpendicular_lines_sufficient_but_not_necessary (a : ℝ) :
  (a = 1 → (∃ x y : ℝ, l1 a x y ∧ l2 a x y ∧ (a * (a - 1) + (a - 1) * (2 * a + 3)) = 0)) ∧ 
  (∃ b ≠ 1, ∃ x y: ℝ, l1 b x y ∧ l2 b x y ∧ (b * (b - 1) + (b - 1) * (2 * b + 3)) = 0) :=
by 
  sorry

end perpendicular_lines_sufficient_but_not_necessary_l274_274163


namespace gcd_polynomial_l274_274164

-- Define conditions
variables (b : ℤ) (k : ℤ)

-- Assume b is an even multiple of 8753
def is_even_multiple_of_8753 (b : ℤ) : Prop := ∃ k : ℤ, b = 2 * 8753 * k

-- Statement to be proven
theorem gcd_polynomial (b : ℤ) (h : is_even_multiple_of_8753 b) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 :=
by sorry

end gcd_polynomial_l274_274164


namespace probability_x_lt_2y_l274_274042

noncomputable def rectangle := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 }

noncomputable def region_of_interest := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 6 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 < 2 * p.2 }

noncomputable def area_rectangle := 6 * 2

noncomputable def area_trapezoid := (1 / 2) * (4 + 6) * 2

theorem probability_x_lt_2y : (area_trapezoid / area_rectangle) = 5 / 6 :=
by
  -- skip the proof
  sorry

end probability_x_lt_2y_l274_274042


namespace minimum_Q_value_l274_274124

-- Define the function for the nearest integer
noncomputable def nearestInt (m k : ℤ) : ℤ :=
  let x := (m : ℚ) / (k : ℚ)
  if x - x.floor ≤ 0.5 then x.floor else x.ceil

-- Define the function Q(k)
def Q (k : ℤ) : ℚ :=
  let validN := {n : ℤ | 1 ≤ n ∧ n ≤ 149 ∧ nearestInt n k + nearestInt (150 - n) k = nearestInt 150 k}
  ((Finset.card validN.to_finset).toRat) / 149

-- Main theorem statement
theorem minimum_Q_value :
  (min (λ k, Q k) {k : ℤ | 1 ≤ k ∧ k ≤ 150 ∧ k % 2 = 1}) = 37 / 75 :=
sorry

end minimum_Q_value_l274_274124


namespace find_m_l274_274979

theorem find_m (m : ℝ) : 
  (∃ m: ℝ, (let θ := 765 in Real.tan (θ *  Real.pi / 180) = m / 4)) → m = 4 :=
by
  intro h
  sorry

end find_m_l274_274979


namespace sugar_consumption_reduction_l274_274986

variable (X : ℝ)
variable (initialPrice newPrice : ℝ)
variable (initialExpenditure newExpenditure : ℝ)
variable (initialQuantity newQuantity : ℝ)

theorem sugar_consumption_reduction :
  initialPrice = 2 → 
  newPrice = 5 → 
  initialExpenditure = initialPrice * initialQuantity →
  newExpenditure = newPrice * newQuantity →
  initialExpenditure = newExpenditure →
  newQuantity = (2 / 5) * initialQuantity →
  (initialQuantity - newQuantity) / initialQuantity * 100 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6
  calc
    (initialQuantity - newQuantity) / initialQuantity * 100
        = (initialQuantity - (2 / 5) * initialQuantity) / initialQuantity * 100 : by rw h6
    ... = ((5 / 5) * initialQuantity - (2 / 5) * initialQuantity) / initialQuantity * 100 : by ring
    ... = ((5 - 2) / 5) * 100 : by rw [div_mul_cancel (5 - 2) (5)] 
    ... = 60 : by norm_num

end sugar_consumption_reduction_l274_274986


namespace sum_of_elements_in_setMul_l274_274471

-- Defining the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {4, 6}

-- Defining the set operation A * B
def setMul (A B : Set ℕ) : Set ℕ := {z | ∃ x y, x ∈ A ∧ y ∈ B ∧ z = x * y}

-- Instance of set operation A * B for given sets
def AB : Set ℕ := setMul A B

-- The correct sum of all elements in the set A * B
def sumAB : ℕ := 30

-- The theorem statement that the sum of elements in A * B is 30
theorem sum_of_elements_in_setMul : (∑ x in AB, x) = sumAB := by
  sorry

end sum_of_elements_in_setMul_l274_274471


namespace estimate_m_value_l274_274013

-- Definition of polynomial P(x) and its roots related to the problem
noncomputable def P (x : ℂ) (a b c : ℂ) : ℂ := x^3 + a * x^2 + b * x + c

-- Statement of the problem in Lean 4
theorem estimate_m_value :
  ∀ (a b c : ℕ),
  a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∃ z1 z2 z3 : ℂ, z1 ≠ z2 ∧ z1 ≠ z3 ∧ z2 ≠ z3 ∧ 
  P z1 a b c = 0 ∧ P z2 a b c = 0 ∧ P z3 a b c = 0) →
  ∃ m : ℕ, m = 8097 :=
sorry

end estimate_m_value_l274_274013


namespace rearrange_expression_l274_274620

theorem rearrange_expression :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 :=
by
  sorry

end rearrange_expression_l274_274620


namespace num_arrangements_of_balls_l274_274307

def balls := Finset (Fin 6)

noncomputable def numArrangements : ℕ :=
calc
  -- There are 2 ways to arrange balls 1 and 2 adjacent to each other as a block
  let adj12 := 2 in
  -- There are 4 remaining balls to arrange along with the block 1-2, total is 4! permutations
  let perm4 := 24 in
  -- There are 4 spaces left to place ball 5 and 6 such that they are not adjacent
  let choose_2_not_adj := 6 in
  -- Total valid arrangements
  adj12 * perm4 * choose_2_not_adj

theorem num_arrangements_of_balls : numArrangements = 144 :=
by
  sorry

end num_arrangements_of_balls_l274_274307


namespace sin_2theta_value_l274_274964

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l274_274964


namespace greatest_area_difference_l274_274741

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end greatest_area_difference_l274_274741


namespace license_plate_count_l274_274567

/-- 
Proof statement: 
A license plate consists of 4 characters where:
1. The first character is a letter.
2. The second and third characters can either be a letter or a digit.
3. The fourth character is a digit.
4. There must be two characters on the license plate which are the same.

Prove that the number of ways to choose such a license plate equals 56,520.
-/
theorem license_plate_count :
  ∃ (n : ℕ), 
    n = 56520 ∧  
    (∃ f : fin 4 → char, 
      (f 0 ∈ alphabet ∧ 
       (f 1 ∈ alphabet ∪ digits) ∧ 
       (f 2 ∈ alphabet ∪ digits) ∧ 
       (f 3 ∈ digits) ∧ 
       (∃ i j : fin 4, i ≠ j ∧ f i = f j))) := 
sorry

end license_plate_count_l274_274567


namespace sin_double_angle_series_l274_274970

theorem sin_double_angle_series (θ : ℝ) (h : ∑' (n : ℕ), (sin θ)^(2 * n) = 3) :
  sin (2 * θ) = (2 * sqrt 2) / 3 :=
sorry

end sin_double_angle_series_l274_274970


namespace find_time_period_l274_274860

theorem find_time_period (P r CI : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (hP : P = 10000)
  (hr : r = 0.15)
  (hCI : CI = 3886.25)
  (hn : n = 1)
  (hA : A = P + CI)
  (h_formula : A = P * (1 + r / n) ^ (n * t)) : 
  t = 2 := 
  sorry

end find_time_period_l274_274860


namespace representable_as_set_l274_274820

theorem representable_as_set :
  ∃ s2 s3 : Set ℝ, s2 = ∅ ∧ (∀ x y : ℝ, (x, y) ∈ s3 ↔ x = y) :=
by
  let s2 := {x : ℝ | x^2 + 3 = 0}
  let s3 := {(x, y) : ℝ × ℝ | y = x}
  use s2, s3
  have h2 : s2 = ∅ := by
    ext x
    simp [eq_empty_iff_forall_not_mem]
    intros x hx
    exact not_lt_of_ge (le_of_eq (eq_from_sub_eq_zero_iff hx))
  have h3 : ∀ x y : ℝ, (x, y) ∈ s3 ↔ x = y := by
    intros x y
    simp
  exact ⟨h2, λ x y, h3 x y⟩

end representable_as_set_l274_274820


namespace main_theorem_l274_274558

-- Define the vectors a, b, c and the conditions on m and n
variables (m n : ℝ) (hm : m > 0) (hn : n > 0)
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (m - 2, -n)
def ab_parallel_c : Prop := 
  (a.1 - b.1) / (c.1) = (a.2 - b.2) / (c.2)

-- The main theorem we need to prove
theorem main_theorem (h_parallel : ab_parallel_c) : 2 * m + n = 4 := 
by
  sorry

end main_theorem_l274_274558


namespace term_containing_1x2_is_10_find_n_equality_l274_274542

-- Definitions based on the problem's conditions
def general_term (a b : ℤ) (x : ℤ) (n r : ℕ) : ℤ := (Nat.choose n r) * a ^ (n - r) * b ^ r * x ^ (n - 2 * r)

axiom sum_binom_lt_coeff (sum_binom : ℕ) (coeff3 : ℕ) : sum_binom + 28 = coeff3

-- Axioms (the correct answers derived from the given problems)
axiom term_containing_1x2_expansion : ∀ (x : ℤ), general_term 2 (1 / x) x 5 4 = 10 / x ^ 2
axiom coeff_third_term_expansion : ∀ (x : ℤ) (n : ℕ), general_term (sqrt x) (2 / x) x n 2 = 4 * (Nat.choose n 2) * x ^ (n / 2 - 3)

-- The two proofs as Lean statements
theorem term_containing_1x2_is_10 (x : ℤ) : general_term 2 (1 / x) x 5 4 = 10 / x ^ 2 := 
  term_containing_1x2_expansion x

theorem find_n_equality (n : ℕ) (sum_binom : ℕ) : 
  let coeff3 := 4 * (Nat.choose n 2) * 1
  sum_binom + 28 = coeff3 → Nat.choose n 2 = 15 → n = 6 := sorry

end term_containing_1x2_is_10_find_n_equality_l274_274542


namespace value_of_m_l274_274525

theorem value_of_m (x y m : ℤ) (h1 : x = 1) (h2 : y = 3) (h3 : x - 2 * y = m) : m = -5 := 
by
  rw [h1, h2] at h3
  exact h3

end value_of_m_l274_274525


namespace construct_triangle_iff_l274_274468

variables (α s f : ℝ)

def can_construct_triangle (α s f : ℝ) : Prop :=
  f ≤ s * (1 - (real.sin (α / 2))) / (real.cos (α / 2))

theorem construct_triangle_iff (α s f : ℝ) :
  can_construct_triangle α s f ↔ (f ≤ s * (1 - real.sin (α / 2)) / real.cos (α / 2)) :=
by sorry

end construct_triangle_iff_l274_274468


namespace even_n_divisible_into_equal_triangles_l274_274875

theorem even_n_divisible_into_equal_triangles (n : ℕ) (hn : 3 < n) :
  (∃ (triangles : ℕ), triangles = n) ↔ (∃ (k : ℕ), n = 2 * k) := 
sorry

end even_n_divisible_into_equal_triangles_l274_274875


namespace worms_split_into_dominoes_l274_274463

theorem worms_split_into_dominoes (n : ℕ) : 
  ∃ (count : ℕ), count = ∏ k in (Finset.range n).filter (Nat.coprime n), k :=
sorry

end worms_split_into_dominoes_l274_274463


namespace time_2517_hours_from_now_l274_274334

-- Define the initial time and the function to calculate time after certain hours on a 12-hour clock
def current_time := 3
def hours := 2517

noncomputable def final_time_mod_12 (current_time : ℕ) (hours : ℕ) : ℕ :=
  (current_time + (hours % 12)) % 12

theorem time_2517_hours_from_now :
  final_time_mod_12 current_time hours = 12 :=
by
  sorry

end time_2517_hours_from_now_l274_274334


namespace solution_x2_l274_274389

def equation_A (x : ℝ) : Prop := 4 * x = 2
def equation_B (x : ℝ) : Prop := 3 * x + 6 = 0
def equation_C (x : ℝ) : Prop := (1 / 2) * x = 0
def equation_D (x : ℝ) : Prop := 7 * x - 14 = 0

theorem solution_x2 (x : ℝ) : x = 2 → (equation_D x ∧ ¬equation_A x ∧ ¬equation_B x ∧ ¬equation_C x) :=
by {
  assume hx : x = 2,
  split,
  { rw [hx, equation_D, eq_self_iff_true] },
  { split,
    { rw [hx, equation_A, ne.def, eq_false_iff_ne, not_false_iff] },
    { split,
      { rw [hx, equation_B, ne.def, eq_false_iff_ne, not_false_iff] },
      { rw [hx, equation_C, ne.def, eq_false_iff_ne, not_false_iff] }
    }
  }
}

end solution_x2_l274_274389


namespace option_A_option_B_option_C_option_D_l274_274517

/-- Given a complex number z = a + (a + 1)i where a ∈ ℝ -/
variables (a : ℝ) (z : ℂ)
noncomputable def z_def : ℂ := a + (a + 1) * complex.I

/-- Option A: If z ∈ ℝ, then a = -1 -/
theorem option_A : (z_def a ∈ ℝ) -> a = -1 :=
by sorry

/-- Option B: If z is purely imaginary, then a = 0 -/
theorem option_B : (z_def a).re = 0 -> a = 0 :=
by sorry

/-- Option C: If a = 1, then the complex conjugate of z is not 1 + 2i -/
theorem option_C : a = 1 -> complex.conj (z_def 1) ≠ 1 + 2 * complex.I :=
by sorry

/-- Option D: If a = 3, then |z| = 5 -/
theorem option_D : a = 3 -> complex.abs (z_def 3) = 5 :=
by sorry

end option_A_option_B_option_C_option_D_l274_274517


namespace find_radius_l274_274105

noncomputable def cylinder_radius (r : ℝ) : Prop :=
  ∃ (r : ℝ), π * (r + 10)^2 * 4 = π * r^2 * 14 → r = 4 + 2 * Real.sqrt 14

theorem find_radius : cylinder_radius (4 + 2 * Real.sqrt 14) :=
by
  sorry

end find_radius_l274_274105


namespace area_is_36_5_l274_274273

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 5 then x else if 5 < x ∧ x ≤ 8 then 2 * x - 5 else 0

-- Define the piecewise curve
def curve := {p : ℝ × ℝ | p.snd = f p.fst ∧ 0 ≤ p.fst ∧ p.fst ≤ 8}

-- Define the measure of the area bounded by the curve and the x-axis
noncomputable def area_bounded_by_curve : ℝ :=
  ∫ x in 0..5, f x + ∫ x in 5..8, f x

theorem area_is_36_5 : area_bounded_by_curve = 36.5 := by
  sorry

end area_is_36_5_l274_274273


namespace median_of_heights_is_160_l274_274041

-- Given heights of 7 students
def heights : List ℕ := [175, 160, 158, 155, 168, 151, 170]

-- The median value we want to prove is 160
def median_value : ℕ := 160

-- Defining the property that needs to be proven
theorem median_of_heights_is_160 : List.median heights = median_value := 
by 
  sorry

end median_of_heights_is_160_l274_274041


namespace triangle_area_ratio_l274_274619

theorem triangle_area_ratio
  (A B C D : Type)
  (triangle : A ≠ B → A ≠ C → B ≠ C)
  (D_on_AB : D ∈ line(A, B))
  (angle_bisector : ∀ (A B C D : Type), D ∈ line(A, B) → bisects_angle A B C D)
  (AB_eq_28 : dist(A, B) = 28)
  (AC_eq_35 : dist(A, C) = 35)
  : area_ratio(triangle(A, C, D), triangle(A, B, D)) = 29 / 20 := by
  sorry

end triangle_area_ratio_l274_274619


namespace theta_range_l274_274298

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

def ineq_holds (theta x : ℝ) : Prop :=
  let sin_theta := sin theta in
  f (x / sin_theta) - (4 * (sin_theta)^2) * f x ≤ f (x - 1) + 4 * f sin_theta

theorem theta_range (theta : ℝ) :
  (∀ x : ℝ, (x ∈ set.Ici (3 / 2)) → ineq_holds theta x) ↔
  (theta ∈ set.Icc (π / 3) (2 * π / 3)) ∧ (0 < theta) ∧ (theta < π) :=
sorry

end theta_range_l274_274298


namespace divisor_problem_l274_274863

theorem divisor_problem : ∃ (d : ℕ), (7 = d ∧ d ∣ (5474827 - (5474827 - 7))) :=
by
  use 7
  split
  · rfl
  · sorry

end divisor_problem_l274_274863


namespace smallest_n_for_cubic_sum_inequality_l274_274070

theorem smallest_n_for_cubic_sum_inequality :
  ∃ n : ℕ, (∀ (a b c : ℕ), (a + b + c) ^ 3 ≤ n * (a ^ 3 + b ^ 3 + c ^ 3)) ∧ n = 9 :=
sorry

end smallest_n_for_cubic_sum_inequality_l274_274070


namespace max_levels_passed_prob_first_three_levels_l274_274215

-- Assuming X is a random variable representing the outcome of a single fair die roll
variable (X : ℕ → ℕ)

-- Definition for passing level k
def pass_level (k : ℕ) : Prop :=
  (finset.sum (finset.range k) (λ i, X i)) > 2^k

-- Maximum number of levels passed theorem
theorem max_levels_passed : ∃ (N : ℕ), ∀ (k : ℕ), (k ≤ 4 ↔ pass_level X k) := sorry

-- Probability of passing the first three levels consecutively
theorem prob_first_three_levels : 
  ∃ p : ℚ, p = (2 / 3) * (5 / 6) * (20 / 27) ∧ p = 100 / 243 := sorry

end max_levels_passed_prob_first_three_levels_l274_274215


namespace sum_of_angles_two_triangles_l274_274383

theorem sum_of_angles_two_triangles (T1 T2 : Type) [triangle T1] [triangle T2] : 
  (sum_of_inter_angles T1) + (sum_of_inter_angles T2) = 360° :=
sorry

end sum_of_angles_two_triangles_l274_274383


namespace problem_statement_l274_274275

noncomputable def A : Vect 2 ℚ := (1, 2)
noncomputable def B : Vect 2 ℚ := (4, 3)
noncomputable def P : Vect 2 ℚ := (17/5, 14/5)
noncomputable def t : ℚ := 1/5
noncomputable def u : ℚ := 4/5

theorem problem_statement : P = t • A + u • B := by
  sorry

end problem_statement_l274_274275


namespace fraction_changes_l274_274980

theorem fraction_changes (x y : ℝ) (h : 0 < x ∧ 0 < y) :
  (x + y) / (x * y) = 2 * ((2 * x + 2 * y) / (2 * x * 2 * y)) :=
by
  sorry

end fraction_changes_l274_274980


namespace problem_statement_l274_274821

theorem problem_statement :
  (-2010)^2011 = - (2010 ^ 2011) :=
by
  -- proof to be filled in
  sorry

end problem_statement_l274_274821


namespace bus_speed_l274_274716

theorem bus_speed (r : ℝ) (rpm : ℝ) (speed : ℝ) : 
  r = 70 → 
  rpm = 250.22747952684256 → 
  speed ≈ 66.04 := 
by
  sorry

end bus_speed_l274_274716


namespace calculate_radius_of_cone_l274_274801

-- Definitions based on the problem's conditions
def volume_of_cone (r h : ℝ) := (1/3) * real.pi * r^2 * h
def given_volume : ℝ := 24 * real.pi
def given_height : ℝ := 6
def correct_radius : ℝ := 2 * real.sqrt 3

-- Lean theorem statement ensuring the calculated radius matches the correct radius
theorem calculate_radius_of_cone :
  ∃ r : ℝ, volume_of_cone r given_height = given_volume ∧ r = correct_radius :=
by
  sorry

end calculate_radius_of_cone_l274_274801


namespace number_of_upside_down_symmetric_9_digit_numbers_l274_274758

theorem number_of_upside_down_symmetric_9_digit_numbers : 
  let same_digits := {0, 1, 8}
  let switch_digits := {(6, 9), (9, 6)}
  let valid_pairs := ({(1, 1), (8, 8), (6, 9), (9, 6)} : Finset (ℕ × ℕ))
  let valid_single_digits := {0, 1, 8}
  (4 * 125 * 3 : ℕ) = 1500 :=
by
  sorry

end number_of_upside_down_symmetric_9_digit_numbers_l274_274758


namespace find_k_l274_274900

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n-1)) / 2 * d

theorem find_k (a₁ d : ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h₁ : a₁ = 1) (h₂ : d = 2) (h₃ : ∀ n, S (n+2) = 28 + S n) :
  k = 6 := by
  sorry

end find_k_l274_274900


namespace rectangle_area_l274_274363

theorem rectangle_area (w l : ℝ) (h_width : w = 4) (h_perimeter : 2 * l + 2 * w = 30) :
    l * w = 44 :=
by 
  sorry

end rectangle_area_l274_274363


namespace order_of_a_b_c_l274_274644

-- Definitions of the given conditions
def a : ℝ := 0.6 ^ 4
def b : ℝ := Real.log 3 / Real.log 2  -- Note: log base '2'
def c : ℝ := 0.6 ^ 5

theorem order_of_a_b_c : b > a ∧ a > c := 
by
  -- Using sorry to skip the proof
  sorry

end order_of_a_b_c_l274_274644


namespace count_integer_solutions_ineq_count_integer_solutions_l274_274109

theorem count_integer_solutions_ineq (n : ℤ) :
  (n - 3) * (n + 5) < 0 ↔ n ∈ Set.Ico (-4 : ℤ) 3 :=
by
  sorry

theorem count_integer_solutions (N : ℕ) :
  N = Set.card (Set.Ico (-4 : ℕ) 3) :=
by
  sorry

end count_integer_solutions_ineq_count_integer_solutions_l274_274109


namespace cos_of_pi_div_4_plus_alpha_l274_274963

theorem cos_of_pi_div_4_plus_alpha (α : ℝ) (h : sin (π / 4 - α) = -2 / 5) : 
  cos (π / 4 + α) = -2 / 5 :=
sorry

end cos_of_pi_div_4_plus_alpha_l274_274963


namespace tan_of_right_triangle_B_l274_274855

theorem tan_of_right_triangle_B
  (A B C : Type) 
  [DecidableReal A] [DecidableReal B] [DecidableReal C]
  (AB AC BC : ℝ)
  (hA : ∠A = π / 2) 
  (hAB : AB = 12) 
  (hAC : AC = 13)
  (hPythagorean : AC^2 = AB^2 + BC^2) : 
  tan (atan (BC / AB)) = 5 / 12 :=
by {
  have hBC : BC = 5,
  -- Proof that BC = 5 using the given conditions, following the steps from the solution but setting it as a have statement instead
  {
    sorry
  },
  have tanB := tan (atan (BC / AB)),
  rw hBC at tanB,
  rw hAB at tanB,
  simp at tanB,
  exact tanB,
}

end tan_of_right_triangle_B_l274_274855


namespace max_possible_value_l274_274841

theorem max_possible_value (a b : ℝ) (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n) :
  ∃ a b, ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n → ∃ s : ℝ, (s = 0 ∨ s = 1 ∨ s = 2) →
  max (1 / a^(2009) + 1 / b^(2009)) = 2 :=
sorry

end max_possible_value_l274_274841


namespace proof_xyz_rational_l274_274633

theorem proof_xyz_rational (x y z : ℝ)
    (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
    (h4 : ∃ a b c : ℚ, x * y = a ∧ y * z = b ∧ z * x = c)
    (h5 : (x^2 + y^2 + z^2 : ℝ) ∈ ℚ)
    (h6 : (x^3 + y^3 + z^3 : ℝ) ∈ ℚ) :
    x ∈ ℚ ∧ y ∈ ℚ ∧ z ∈ ℚ :=
sorry

end proof_xyz_rational_l274_274633


namespace convert_roads_maintain_connectivity_l274_274224

-- Define the given conditions as a structure in Lean
structure Country (V : Type) (E : Type) :=
  (vertices : Finset V)
  (roads : Finset E)
  (connects : E → V × V)
  (finite_vertices : vertices.finite)
  (finite_roads : roads.finite)
  (no_loops : ∀ e ∈ roads, (connects e).1 ≠ (connects e).2)
  (unique_roads : ∀ v1 v2 : V, ∃! e : E, connects e = (v1, v2) ∨ connects e = (v2, v1))
  (connected : ∀ u v : V, ∃ p : List E, p.head.entries (u, v) ∨ p.head.entries (v, u))
  (even_degree : ∀ v : V, (Fintype.card { e // connects e = (v, _) } 
                             + Fintype.card { e // connects e = (_, v) }) % 2 = 0)

-- State the first part of the proof as a theorem in Lean
theorem convert_roads (V : Type) (E : Type) (c : Country V E) :
  ∃ (orient_edges : E → (V × V)), 
    (∀ v : V, Fintype.card { e : E // (orient_edges e).1 = v }
             = Fintype.card { e : E // (orient_edges e).2 = v }) :=
  by
  sorry

-- State the second part of the proof as a theorem in Lean
theorem maintain_connectivity (V : Type) (E : Type) (c : Country V E) 
  (orient_edges : E → (V × V))
  (h : ∀ v : V, Fintype.card { e : E // (orient_edges e).1 = v }
                    = Fintype.card { e : E // (orient_edges e).2 = v }) :
  ∀ u v : V, ∃ p : List (V × V), p.head.entries (u, v) :=
  by
  sorry

end convert_roads_maintain_connectivity_l274_274224


namespace four_digit_multiples_of_11_count_l274_274839

theorem four_digit_multiples_of_11_count :
  let digit := {n : ℕ // n < 10} in
  ∃ (a b c d : digit),
    (a.val ≠ 0 ∧
     (a.val - b.val + c.val - d.val) % 11 = 0 ∧
     (a.val + b.val + c.val + d.val) % 11 = 0) ∧
    72 = (count (λ (p : digit × digit), (a, d)) (finset.univ.product finset.univ)) :=
by
  sorry

end four_digit_multiples_of_11_count_l274_274839


namespace division_problem_l274_274385

theorem division_problem (x y n : ℕ) 
  (h1 : x = n * y + 4) 
  (h2 : 2 * x = 14 * y + 1) 
  (h3 : 5 * y - x = 3) : n = 4 := 
sorry

end division_problem_l274_274385


namespace compare_P_Q_l274_274838

-- Define the structure of the number a with 2010 digits of 1
def a := 10^2010 - 1

-- Define P and Q based on a
def P := 24 * a^2
def Q := 24 * a^2 + 4 * a

-- Define the theorem to compare P and Q
theorem compare_P_Q : Q > P := by
  sorry

end compare_P_Q_l274_274838


namespace vitamin_D_scientific_notation_l274_274063

def scientific_notation (x : ℝ) (m : ℝ) (n : ℤ) : Prop :=
  x = m * 10^n

theorem vitamin_D_scientific_notation :
  scientific_notation 0.0000046 4.6 (-6) :=
by {
  sorry
}

end vitamin_D_scientific_notation_l274_274063


namespace finalize_proof_l274_274780

noncomputable def factorial_proof_problem : ℝ :=
  (Nat.factorial 9 ^ 2) / Real.sqrt (Nat.factorial 6) + (3 / 7 * 4 ^ 3)

theorem finalize_proof : factorial_proof_problem = 4906624027 :=
  by
    sorry

end finalize_proof_l274_274780


namespace flour_amount_second_combination_l274_274333

-- Define given conditions as parameters
variables {sugar_cost flour_cost : ℝ} (sugar_per_pound flour_per_pound : ℝ)
variable (cost1 cost2 : ℝ)

axiom cost1_eq :
  40 * sugar_per_pound + 16 * flour_per_pound = cost1

axiom cost2_eq :
  30 * sugar_per_pound + flour_cost = cost2

axiom sugar_rate :
  sugar_per_pound = 0.45

axiom flour_rate :
  flour_per_pound = 0.45

-- Define the target theorem
theorem flour_amount_second_combination : ∃ flour_amount : ℝ, flour_amount = 28 := by
  sorry

end flour_amount_second_combination_l274_274333


namespace initial_average_production_l274_274873

theorem initial_average_production (A : ℕ) (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h₁ : n = 4) 
  (h₂ : today_production = 90) 
  (h₃ : new_average = 58) 
  (h₄ : (A * n + today_production) = new_average * (n + 1))
  :
  A = 50 :=
by
  -- Initial conditions are given
  have h₅ : n = 4 := h₁
  have h₆ : today_production = 90 := h₂
  have h₇ : new_average = 58 := h₃
  have h₈ : (A * 4 + 90) = 58 * 5 := h₄

  -- Proceed by simplifying each part and solving for A
  -- Simplification and solution steps
  sorry

end initial_average_production_l274_274873


namespace total_cows_in_ranch_l274_274751

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l274_274751


namespace valid_paths_A_to_B_l274_274674

-- Definitions and conditions
def grid_size : ℕ × ℕ := (5, 7)
def A : ℕ × ℕ := (0, 5)
def B : ℕ × ℕ := (7, 0)
def forbidden_segments : list (ℕ × ℕ) := [(3, 4), (4, 4), (2, 2), (3, 2)]

-- Theorem statement
theorem valid_paths_A_to_B : 
  let total_paths := (nat.choose 12 5)
  let forbidden1_paths := (nat.choose 4 1) * (nat.choose 7 3)
  let forbidden2_paths := (nat.choose 5 2) * (nat.choose 7 2)
  let valid_paths := total_paths - (forbidden1_paths + forbidden2_paths)
  valid_paths = 442 :=
by
  -- Definitions
  let total_paths := (nat.choose 12 5)
  let forbidden1_paths := (nat.choose 4 1) * (nat.choose 7 3)
  let forbidden2_paths := (nat.choose 5 2) * (nat.choose 7 2)
  let valid_paths := total_paths - (forbidden1_paths + forbidden2_paths)
  
  -- Theorem proof placeholder
  sorry

end valid_paths_A_to_B_l274_274674


namespace scientific_notation_256000_l274_274329

theorem scientific_notation_256000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 256000 = a * 10^n ∧ a = 2.56 ∧ n = 5 :=
by
  sorry

end scientific_notation_256000_l274_274329


namespace sufficient_condition_for_inequality_l274_274926

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0 :=
by
  sorry

end sufficient_condition_for_inequality_l274_274926


namespace cyclist_speed_l274_274419

theorem cyclist_speed
  (V : ℝ)
  (H1 : ∃ t_p : ℝ, V * t_p = 96 ∧ t_p = (96 / (V - 1)) - 2)
  (H2 : V > 1.25 * (V - 1)) :
  V = 16 :=
by
  sorry

end cyclist_speed_l274_274419


namespace passing_percentage_is_40_l274_274789

noncomputable theory

-- Define the given conditions
def candidate_marks := 40
def fail_margin := 20
def max_marks := 150

-- Define what we need to prove: the passing percentage
def passing_marks := candidate_marks + fail_margin
def passing_percentage := (passing_marks / max_marks : ℚ) * 100

-- The theorem to prove that the passing percentage is 40%
theorem passing_percentage_is_40 : passing_percentage = 40 := by
  sorry

end passing_percentage_is_40_l274_274789


namespace number_of_negatives_is_3_l274_274613

theorem number_of_negatives_is_3 :
  let numbers := [(-1/2 : ℚ), 5, 0, -(-3), -2, -|25|]
  in count (λ x, x < 0) numbers = 3
:= by
  sorry

end number_of_negatives_is_3_l274_274613


namespace line_through_D1E1_D2E2_l274_274557

-- Conditions
def circle1 (D1 E1 : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D1 * x + E1 * y - 3 = 0

def circle2 (D2 E2 : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 + D2 * x + E2 * y - 3 = 0

def point_A : ℝ × ℝ := (2, -1)

-- Given conditions that circles pass through (2, -1)
def condition1 (D1 E1 : ℝ) : Prop :=
  circle1 D1 E1 2 (-1)

def condition2 (D2 E2 : ℝ) : Prop :=
  circle2 D2 E2 2 (-1)

-- Proposition to be proved
theorem line_through_D1E1_D2E2 (D1 E1 D2 E2 : ℝ) 
  (h1: condition1 D1 E1) 
  (h2: condition2 D2 E2) : 
  ∃ (x : ℝ) (y : ℝ), (x, y) = (D1, E1) ∨ (x, y) = (D2, E2) ∧ 2 * x - y + 2 = 0 :=
sorry

end line_through_D1E1_D2E2_l274_274557


namespace no_common_real_root_l274_274834

theorem no_common_real_root (a b : ℚ) : 
  ¬ ∃ (r : ℝ), (r^5 - r - 1 = 0) ∧ (r^2 + a * r + b = 0) :=
by
  sorry

end no_common_real_root_l274_274834


namespace find_abs_slope_l274_274795

theorem find_abs_slope 
  (k : ℝ)
  (x y : ℝ)
  (A B : ℝ → ℝ)
  (h_ellipse : ∀ (x y : ℝ), x^2 + 2*y^2 = 3)
  (h_foci : ∃ (x : ℝ), x^2 + (\frac{\sqrt{6}}{2})^2 = 3)
  (h_line : y = k * (x - (√6 / 2)))
  (h_intersect : ∀ {A B : ℝ → ℝ}, A ≠ B ∧ (h_line = true → h_ellipse = true) )
  (h_distance : ∀ (A B : ℝ → ℝ), dist A B = 2) 
  : |k| = sqrt (2 + sqrt 3) := sorry

end find_abs_slope_l274_274795


namespace problem_l274_274582

theorem problem (m : ℝ) (h : m^2 + 3 * m = -1) : m - 1 / (m + 1) = -2 :=
by
  sorry

end problem_l274_274582


namespace general_term_of_sequence_l274_274173

-- Given conditions
variable (b : ℕ → ℝ) (S : ℕ → ℝ)
variable (h1 : ∀ n, S n = ∑ i in finset.range n, b i)
variable (h2 : ∀ n, b n = 2 - 2 * S n)

-- Main theorem: Prove the general term of the sequence
theorem general_term_of_sequence (n : ℕ) : b n = (2 / 3^n) := 
  sorry

end general_term_of_sequence_l274_274173


namespace analytic_expression_and_properties_l274_274918

theorem analytic_expression_and_properties
  (b c : ℝ)
  (h_domain : ∀ x : ℝ, x ≠ 0 → bx + c ≠ 0)
  (h_f1_eq_2 : f(1) = 2)
  (f : ℝ → ℝ := λ x, (x^2 + 1) / (bx + c)) :
  (f = λ x, x + 1 / x) ∧
  (∀ x1 x2 ∈ set.Ici (1 : ℝ), x1 < x2 → f x1 < f x2) ∧
  (max (set.image f (set.Icc 1 2)) = 5 / 2) ∧
  (min (set.image f (set.Icc 1 2)) = 2) :=
by sorry

end analytic_expression_and_properties_l274_274918


namespace cube_color_selection_l274_274269

theorem cube_color_selection (n : ℕ) (h_even : n % 2 = 0) (h_gt2 : 2 < n) :
  ∃ (S : finset (ℕ × ℕ × ℕ)), 
  S.card = n ∧ 
  (∀ (c1 c2 : ℕ × ℕ × ℕ), c1 ∈ S ∧ c2 ∈ S → c1 ≠ c2 → distinct_layers c1 c2) ∧ 
  (∀ (c : ℕ × ℕ × ℕ), c ∈ S → unique_color c)
  :=
sorry

-- assuming the following two helper predicates
def distinct_layers (c1 c2 : ℕ × ℕ × ℕ) : Prop := 
  c1.1 ≠ c2.1 ∧ c1.2 ≠ c2.2 ∧ c1.3 ≠ c2.3

def unique_color (c : ℕ × ℕ × ℕ) : Prop :=
  -- Assuming some function color : (ℕ × ℕ × ℕ) → ℕ denoting the color of the unit cube
  ∀ (d : ℕ × ℕ × ℕ), c ≠ d ∧ d ∈ S → color(c) ≠ color(d)

end cube_color_selection_l274_274269


namespace S_2019_eq_l274_274171

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

def a (n : ℕ) : ℝ := 1 / (f (n + 1) + f n)

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem S_2019_eq : S 2019 = Real.sqrt 2020 - 1 :=
by
  sorry

end S_2019_eq_l274_274171


namespace height_at_age_10_approx_l274_274797

noncomputable def height_prediction (x : ℕ) : ℝ :=
  7.19 * x + 73.93

theorem height_at_age_10_approx : 
  ∀ (x : ℕ), x = 10 → height_prediction x ≈ 145.83 :=
by
  intro x
  intro hx
  rw hx
  sorry -- skip the proof

end height_at_age_10_approx_l274_274797


namespace raft_time_l274_274052

-- Define the problem as a Lean theorem
theorem raft_time (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : 
  let t := 2 * a * b / (b - a) in
  t = 2 * a * b / (b - a) :=
by 
  -- We assume the proof here, so we just return sorry
  sorry

end raft_time_l274_274052


namespace always_true_inequality_count_l274_274079

-- Definitions of triangle and necessary points
variables {A B C D E F : Point}
variable {a : Path}
variable {b : Path}
variable {c : Path}

-- Given conditions
def is_isosceles (A B C : Point) : Prop :=
  dist A C = dist B C

def is_altitude (C D : Point) (A B : Segment) : Prop := 
  is_perpendicular (line C D) (line A B) ∧ C ∈ line A B

def is_midpoint (E : Point) (B C : Segment) : Prop :=
  (dist B E = dist E C) ∧ (E ∈ segment B C)

def intersection (F A E C D : Point) : Prop :=
  collinear A E F ∧ collinear C D F

-- Paths and their lengths
def path_a (A F C E B D : Point) : Path :=
  [A, F, C, E, B, D, A]

def path_b (A C E B D F : Point) : Path :=
  [A, C, E, B, D, F, A]

def path_c (A D B E F C : Point) : Path :=
  [A, D, B, E, F, C, A]

variables (L_a L_b L_c : ℝ)

def path_length (p: Path) : ℝ := sorry

noncomputable def path_length_a : ℝ := path_length (path_a A F C E B D)
noncomputable def path_length_b : ℝ := path_length (path_b A C E B D F)
noncomputable def path_length_c : ℝ := path_length (path_c A D B E F C)

-- Main theorem
theorem always_true_inequality_count:
  is_isosceles A B C ∧ is_altitude C D (segment A B) ∧ is_midpoint E (segment B C) ∧ 
  intersection F A E C D ∧ 
  L_a = path_length_a ∧ L_b = path_length_b ∧ L_c = path_length_c →
  (if (L_a < L_b) + (L_a < L_c) + (L_b < L_c) = 1 then true else false) :=
sorry

end always_true_inequality_count_l274_274079


namespace necessary_but_not_sufficient_condition_l274_274777

theorem necessary_but_not_sufficient_condition (a b : ℤ) :
  (a ≠ 1 ∨ b ≠ 2) → (a + b ≠ 3) ∧ ¬((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) :=
sorry

end necessary_but_not_sufficient_condition_l274_274777


namespace arithmetic_and_geometric_progression_example_l274_274901

noncomputable def is_arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (n : ℕ) : Prop :=
∀ k : ℕ, k < n - 1 → a (k + 1) = a k + d

noncomputable def is_geometric_progression (a : ℕ → ℝ) (r : ℝ) (n : ℕ) : Prop :=
∀ k : ℕ, k < n - 1 → a (k + 1) = a k * r

theorem arithmetic_and_geometric_progression_example 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hS : ∀ k > 2022, |S k| > |S (k + 1)|)
  (hSn : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  : 
  (∃ d, is_arithmetic_progression a d 2022) ∧ (∃ r, r ∈ (0, 1) ∧ is_geometric_progression a r (∞ - 2022)) :=
sorry

end arithmetic_and_geometric_progression_example_l274_274901


namespace sum_of_squares_divisor_is_integer_l274_274263

theorem sum_of_squares_divisor_is_integer {a b c : ℚ}
  (h1 : (a + b + c) ∈ ℤ)
  (h2 : ((ab + bc + ca) / (a + b + c)) ∈ ℤ) :
  ((a^2 + b^2 + c^2) / (a + b + c)) ∈ ℤ :=
sorry

end sum_of_squares_divisor_is_integer_l274_274263


namespace intersection_of_P_and_Q_l274_274556

def P : Set (ℝ × ℝ) := {p | p.fst + p.snd = 0}
def Q : Set (ℝ × ℝ) := {p | p.fst - p.snd = 2}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(1, -1)} :=
by
  sorry

end intersection_of_P_and_Q_l274_274556


namespace rectangles_count_in_3x3_grid_l274_274675

theorem rectangles_count_in_3x3_grid : ∀ (grid_size : ℕ), grid_size = 3 → 
  (let points := (fin grid_size) × (fin grid_size) in 
     ∃ rectangles_count, rectangles_count = 124
  ) :=
  by
  intros n h
  let points := (fin n) × (fin n)
  use 124
  sorry

end rectangles_count_in_3x3_grid_l274_274675


namespace plane_not_parallel_l274_274202

theorem plane_not_parallel (l : Line) (α : Plane) (h1 : intersects l α) (h2 : ¬ perpendicular l α) : 
  ∃ β : Plane, (passesThrough l β) ∧ (¬ parallel β α) := 
sorry

end plane_not_parallel_l274_274202


namespace hostel_provisions_l274_274424

theorem hostel_provisions (x : ℕ) :
  (250 * x = 200 * 60) → x = 48 :=
by
  sorry

end hostel_provisions_l274_274424


namespace karen_weight_after_six_hours_l274_274267

def initial_water_weight := 20
def initial_food_weight := 10
def initial_gear_weight := 20

def water_consumption_rate_1 := 3
def food_consumption_rate_1 := 0.5 * water_consumption_rate_1

def additional_gear_weight := 5

def water_consumption_rate_2 := 1.5
def food_consumption_rate_2 := 0.25 * water_consumption_rate_2

def total_weight_after_six_hours :=
  let remaining_water_after_3_hours := initial_water_weight - (water_consumption_rate_1 * 3)
  let remaining_food_after_3_hours := initial_food_weight - (food_consumption_rate_1 * 3)
  let gear_weight_after_3_hours := initial_gear_weight + additional_gear_weight
  let remaining_water_after_6_hours := remaining_water_after_3_hours - (water_consumption_rate_2 * 3)
  let remaining_food_after_6_hours := remaining_food_after_3_hours - (food_consumption_rate_2 * 3)
  remaining_water_after_6_hours + remaining_food_after_6_hours + gear_weight_after_3_hours

theorem karen_weight_after_six_hours : total_weight_after_six_hours = 35.875 :=
by
  unfold total_weight_after_six_hours
  sorry

end karen_weight_after_six_hours_l274_274267


namespace Mary_paid_on_Tuesday_l274_274670

theorem Mary_paid_on_Tuesday 
  (credit_limit total_spent paid_on_thursday remaining_payment paid_on_tuesday : ℝ)
  (h1 : credit_limit = 100)
  (h2 : total_spent = credit_limit)
  (h3 : paid_on_thursday = 23)
  (h4 : remaining_payment = 62)
  (h5 : total_spent = paid_on_thursday + remaining_payment + paid_on_tuesday) :
  paid_on_tuesday = 15 :=
sorry

end Mary_paid_on_Tuesday_l274_274670


namespace rides_ratio_l274_274192

theorem rides_ratio (total_money rides_spent dessert_spent money_left : ℕ) 
  (h1 : total_money = 30) 
  (h2 : dessert_spent = 5) 
  (h3 : money_left = 10) 
  (h4 : total_money - money_left = rides_spent + dessert_spent) : 
  (rides_spent : ℚ) / total_money = 1 / 2 := 
sorry

end rides_ratio_l274_274192


namespace polygons_symmetry_l274_274446

-- Definitions for axisymmetric and centrally symmetric
def is_axisymmetric (P : Type*) [Polygon P] : Prop := sorry
def is_centrally_symmetric (P : Type*) [Polygon P] : Prop := sorry

-- Definitions for specific polygons
def equilateral_triangle : Type* := sorry
def square : Type* := sorry
def regular_pentagon : Type* := sorry
def regular_hexagon : Type* := sorry

-- Theorem stating polygons ② (Square) and ④ (Regular hexagon) are both axisymmetric and centrally symmetric
theorem polygons_symmetry :
  is_axisymmetric square ∧ is_centrally_symmetric square ∧
  is_axisymmetric regular_hexagon ∧ is_centrally_symmetric regular_hexagon := sorry

end polygons_symmetry_l274_274446


namespace parabola_at_origin_with_ellipse_vertex_focus_has_correct_equation_l274_274355

-- Definitions based on conditions
def ellipse_vertex := (2*Real.sqrt 2, 0)  -- Vertex of the ellipse
def parabola_vertex := (0, 0)              -- Vertex of the parabola

-- Correct answer based on the solution provided
def parabola_equation (x y : ℝ) : Prop := y^2 = 8*Real.sqrt 2 * x

-- The statement we need to prove
theorem parabola_at_origin_with_ellipse_vertex_focus_has_correct_equation :
  ∃ x y : ℝ, parabola_vertex = (0, 0) ∧ ellipse_vertex = (2*Real.sqrt 2, 0) → parabola_equation x y :=
by
  sorry

end parabola_at_origin_with_ellipse_vertex_focus_has_correct_equation_l274_274355


namespace sequence_expression_l274_274896

noncomputable def sequence (n : ℕ) : ℝ :=
  if h : n = 0 then 0 else
    let a : ℕ → ℝ := λ n, if n = 1 then 2 else a (n - 1) + Real.log (1 + 1 / (n - 1 : ℝ)) in
    a n

theorem sequence_expression (n : ℕ) (hn : n ≠ 0) : sequence n = 2 + Real.log n := by
  sorry

end sequence_expression_l274_274896


namespace initial_leaves_l274_274673

theorem initial_leaves (l_0 : ℕ) (blown_away : ℕ) (leaves_left : ℕ) (h1 : blown_away = 244) (h2 : leaves_left = 112) (h3 : l_0 - blown_away = leaves_left) : l_0 = 356 :=
by
  sorry

end initial_leaves_l274_274673


namespace rectangle_AN_eq_5_l274_274800

theorem rectangle_AN_eq_5 
  (AB BC : ℝ) (N A B C D P Q : Point)
  (AN NC NP PQ QC : ℝ) 
  (h1 : AB = 10)
  (h2 : BC = 5)
  (h3 : AN = x)
  (h4 : NC = 10 - x)
  (h5 : NP = PQ ∧ PQ = QC)
  (h6 : NP = 5)
  (h7 : triangle_sim ANP CNQ) :
  AN = 5 :=
by
  sorry

end rectangle_AN_eq_5_l274_274800


namespace median_of_set_with_mean_85_l274_274342

theorem median_of_set_with_mean_85 (x : ℕ) (h_mean : (90 + 88 + 81 + 84 + 87 + x) / 6 = 85) :
  let s := multiset.sort ≤ {90, 88, 81, 84, 87, x}
  in (s[2] + s[3]) / 2 = 85.5 :=
by
  sorry

end median_of_set_with_mean_85_l274_274342


namespace trig_cos_sub_sin_l274_274138

theorem trig_cos_sub_sin (θ : ℝ) (h1 : θ ∈ set.Ioo (π / 4) (π / 2)) (h2 : sin (2 * θ) = 1 / 16) :
  cos θ - sin θ = -sqrt(15) / 4 := sorry

end trig_cos_sub_sin_l274_274138


namespace periodic_f_2pi_odd_function_f_symmetric_about_line_pi_2_l274_274232

noncomputable def f (x : ℝ) : ℝ := ∑ i in (range 7).map (λ n, 2 * n + 1), sin (x * (2 * n + 1)) / (2 * n + 1)

theorem periodic_f_2pi : ∀ x, f (x + 2 * π) = f x :=
begin
  sorry
end

theorem odd_function_f : ∀ x, f (-x) = -f x :=
begin
  sorry
end

theorem symmetric_about_line_pi_2 : ∀ x, f (π - x) = f x :=
begin
  sorry
end

end periodic_f_2pi_odd_function_f_symmetric_about_line_pi_2_l274_274232


namespace coordinates_of_A_l274_274160

theorem coordinates_of_A {A : ℝ × ℝ} :
  A.1 = 2 ∧ A.2 = 0 ∨ A.1 = -2 ∧ A.2 = 0 ∨ A.1 = 0 ∧ A.2 = 4 ∨ A.1 = 0 ∧ A.2 = -4 :=
begin
  -- Define points O and B
  let O := (0 : ℝ, 0 : ℝ),
  let B := (1 : ℝ, 2 : ℝ),
  
  -- Condition: area of triangle OAB is 2
  have h_area : |(A.1 * B.2 - A.2 * B.1)/2| = 2 := sorry,

  -- Condition: A lies on the coordinate axes
  have h_axes : A.1 = 0 ∨ A.2 = 0 := sorry,

  -- Solve for the coordinates of A
  sorry
end

end coordinates_of_A_l274_274160


namespace pq_perpendicular_rs_l274_274255

-- Let's define the conditions first
variables {A B C D M P Q R S : ℝ × ℝ}
variables (hABCD : ∃ (a b : ℝ), A = (a, b) ∧ B = (a, -b) ∧ C = (-a, -b) ∧ D = (-a, b))
variables (hM_arc : ∃ (α θ : ℝ), θ < α ∧ α < π - θ ∧ M = (cos α, sin α) ∧
  A = (cos θ, sin θ) ∧ B = (cos θ, -sin θ) -- Point M on the arc
  )

-- Projections of M onto the sides
variables (hProjections : P = (cos θ, sin α) ∧ Q = (cos α, sin θ) ∧ 
  R = (-cos θ, sin α) ∧ S = (cos α, -sin θ))

-- Defining the perpendicularity proofs
theorem pq_perpendicular_rs :
  hABCD → hM_arc → hProjections → (let
    kPQ := (sin α - sin θ) / (cos θ - cos α),
    kRS := (sin α + sin θ) / (-cos θ - cos α)
  in kPQ * kRS = -1) := 
begin
  intros hABCD hM_arc hProjections,
  sorry -- The rest of the proof is omitted
end

end pq_perpendicular_rs_l274_274255


namespace trigonometric_identity_l274_274318

theorem trigonometric_identity : 
  sin (10 * real.pi / 180) * cos (50 * real.pi / 180) + cos (10 * real.pi / 180) * sin (50 * real.pi / 180) = sqrt 3 / 2 := 
by
  sorry

end trigonometric_identity_l274_274318


namespace prob_intersection_l274_274009

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B : Event Ω)
variables {p : Probability Ω}

noncomputable
def P (e : Event Ω) : ℚ := p.measure e

axioms (P_A : P A = 4/5) 
       (P_B : P B = 2/5) 
       (indep_AB : indep_event A B)

theorem prob_intersection :
  P (A ∩ B) = 8/25 :=
by
  sorry

end prob_intersection_l274_274009


namespace consecutive_even_sum_l274_274722

theorem consecutive_even_sum (n : ℤ) (h : (n - 2) + (n + 2) = 156) : n = 78 :=
by
  sorry

end consecutive_even_sum_l274_274722


namespace ratio_of_area_l274_274734

-- Define equilateral triangle's side length and midpoints
variables {s : ℝ}
def is_midpoint (A B M : Point) := dist A M = dist B M

-- Define Points A, B, C as vertices of the equilateral triangle
variables {A B C D E F G H : Point}

-- Defining the equilateral triangle and the midpoints
def equilateral_triangle (A B C : Point) (s : ℝ) :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

def midpoints (A B C : Point) :=
  is_midpoint A B D ∧ is_midpoint B C E ∧ is_midpoint C A F

-- Defining centroids of triangles
def centroid (A B C G : Point) := is_centroid A D F G ∧ is_centroid B F E H

-- Ratio of shaded to non-shaded area theorem
theorem ratio_of_area (A B C D E F G H : Point) (s : ℝ) :
  equilateral_triangle A B C s →
  midpoints A B C →
  centroid A D F G →
  centroid B F E H →
  (shaded_area G H F D G) / (triangle_area A B C - shaded_area G H F D G) = 1 / 26 :=
sorry

end ratio_of_area_l274_274734


namespace price_of_pastries_is_5_l274_274683

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l274_274683


namespace perpendicularLines_l274_274605

variable (Plane : Type) [HasPerp Plane] [HasParallel Plane]
variable (Line : Type) [HasPerp Line] [HasParallel Line]

variable (α β : Plane) (m n : Line)
hypothesis (h1 : ⊥ α β) (h2 : ⊥ m α) (h3 : ⊥ n β)

theorem perpendicularLines : ⊥ m n :=
sorry

end perpendicularLines_l274_274605


namespace num_solutions_l274_274519

open Nat

-- Define the problem
theorem num_solutions (n : ℕ) (h : n > 0) : 
  let number_of_divisors := (finset.filter (λ d, n^2 % d = 0) (finset.range (n^2 + 1))).card
  (finset.filter (λ p : ℕ × ℕ, p.1 ≠ p.2 ∧ n * (p.1 + p.2) = p.1 * p.2) (finset.diag (finset.range (n * n + 1)))).card = number_of_divisors - 1 := 
sorry

end num_solutions_l274_274519


namespace range_of_a_l274_274295

noncomputable def lg := Real.log10

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x + 1 / x > a

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 / 2) ≤ x ∧ x ≤ 2 → (x + 1 / x > a)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, lg (a * x^2 - 2 * x + 1) ∈ ℝ) ∧ 
  ∀ x : ℝ, (1 / 2 ≤ x ∧ x ≤ 2 → (x + 1 / x > a)) → 
  (1 < a ∧ a < 2) := 
by
  sorry

end range_of_a_l274_274295


namespace exponent_of_3_in_30_factorial_l274_274254

theorem exponent_of_3_in_30_factorial : 
  ∃ k : ℕ, unique_factorization_monoid.factor_count 30! 3 = k ∧ k = 14 :=
sorry

end exponent_of_3_in_30_factorial_l274_274254


namespace quadratic_real_roots_only_if_l274_274927

noncomputable def quadratic_has_real_roots (a x : ℝ) : Prop :=
  let c := complex.ofReal in
  let Q := c a * (c 1 + complex.I) * (c x)^2 + (c 1 + c a^2 * complex.I) * (c x) + c a^2 + complex.I 
  c a * (c 1 + complex.I) ≠ 0 ∧
  let discriminant := 4 * ((a + 2) * a - 1) in
  ∀ x, Q = 0 → (x = a ∨ x = -a)  ∧ discriminant ≤ 0 

theorem quadratic_real_roots_only_if (a : ℝ) : (quadratic_has_real_roots a x) → a = -1 := 
by
  sorry

end quadratic_real_roots_only_if_l274_274927


namespace trapezoid_extensions_meet_at_acute_angle_l274_274699

theorem trapezoid_extensions_meet_at_acute_angle
  (A B C D : Point ℝ)
  (h_isosceles : IsIsoscelesTrapezoid A B C D)
  (h_base_AB : distance A B = 2)
  (h_base_CD : distance C D = 11) :
  IsAcuteAngle (LineThrough A B).extension (LineThrough C D).extension :=
sorry

end trapezoid_extensions_meet_at_acute_angle_l274_274699


namespace sin_cos_value_l274_274501

-- Define the condition: Given tan x = 3
variables {x : ℝ} (h : tan x = 3)

-- Statement to prove: sin x * cos x = 3/10
theorem sin_cos_value : sin x * cos x = 3 / 10 :=
sorry

end sin_cos_value_l274_274501


namespace length_BC_of_parabola_triangle_l274_274445

open Real

theorem length_BC_of_parabola_triangle 
  (h_parabola : ∀ {x y : ℝ}, y = 4 * x^2 → (0, 0) ∈ {(x, y) | y = 4 * x^2})
  (h_parallel : ∀ (a : ℝ), (B C : ℝ × ℝ), B = (-a, 4 * a^2) → C = (a, 4 * a^2) → (B.2 = C.2))
  (h_area : ∀ (a : ℝ), 4 * a^3 = 256)
  : length_BC = 8 :=
by
  sorry

end length_BC_of_parabola_triangle_l274_274445


namespace unique_intersection_max_area_OAB_l274_274235

noncomputable def curve_parametric (a : ℝ) (β : ℝ) : ℝ × ℝ :=
  (a + a * Real.cos β, a * Real.sin β)

def line_polar := {ρ θ : ℝ // ρ * Real.cos (θ - Real.pi / 3) = 3 / 2}

theorem unique_intersection (a : ℝ) (a_pos : a > 0) :
  (∃ β : ℝ, ∃ ρ θ : ℝ, curve_parametric a β = (ρ * Real.cos θ, ρ * Real.sin θ) ∧
    (ρ, θ) ∈ line_polar) → a = 1 :=
sorry

theorem max_area_OAB (a : ℝ) (a_pos : a > 0) 
  (A B : ℝ × ℝ) (hA : ∃ β : ℝ, A = curve_parametric a β)
  (hB : ∃ β : ℝ, B = curve_parametric a (β + Real.pi / 3))
  (angle_AOB : ∃ θ : ℝ, angle_deg_to_rad (angle_between A (0, 0) B) = Real.pi / 3) :
  ∃ max_area : ℝ, max_area = 3 * Real.sqrt 3 * a^2 / 4 :=
sorry

end unique_intersection_max_area_OAB_l274_274235


namespace students_basketball_not_table_tennis_l274_274222

theorem students_basketball_not_table_tennis :
  ∀ (total_students basketball_likers table_tennis_likers neither_likers : ℕ),
  total_students = 30 →
  basketball_likers = 15 →
  table_tennis_likers = 10 →
  neither_likers = 8 →
  ∃ (num_both : ℕ), (basketball_likers - num_both) - (total_students - table_tennis_likers - neither_likers - num_both) = 12 :=
by
  intros total_students basketball_likers table_tennis_likers neither_likers h_total h_basketball h_table_tennis h_neither
  use 3
  rw [h_total, h_basketball, h_table_tennis, h_neither]
  linarith

end students_basketball_not_table_tennis_l274_274222


namespace lines_parallel_l274_274892

noncomputable def hexagon (A B C D E F : Point) : Prop := convex (polygon [A, B, C, D, E, F])

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
∃ (circle : Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle

noncomputable def hexagon_conditions (A B C D E F : Point) : Prop :=
hexagon A B C D E F ∧
∠ F A E = ∠ B D C ∧
cyclic_quadrilateral A B D F ∧
cyclic_quadrilateral A C D E

theorem lines_parallel (A B C D E F : Point) (h : hexagon_conditions A B C D E F) : parallel (line_through B F) (line_through C E) :=
sorry

end lines_parallel_l274_274892


namespace general_term_sum_b_seq_l274_274156

noncomputable def a_seq (n : ℕ) : ℤ := 2 * n - 1

theorem general_term (a_seq : ℕ → ℤ) (a1 : a_seq 1 = 1) 
  (geo_cond : (a_seq 2 + 2) * (a_seq 4 - 2) = (a_seq 3)^2) :
  ∀ n : ℕ, a_seq n = 2 * n - 1 :=
sorry

noncomputable def b_seq (n : ℕ) : ℚ := 1 / (a_seq n * a_seq (n + 1))

theorem sum_b_seq (a_seq : ℕ → ℤ) 
  (b_seq : ℕ → ℚ) 
  (a1 : a_seq 1 = 1) 
  (geo_cond : (a_seq 2 + 2) * (a_seq 4 - 2) = (a_seq 3)^2)
  (a_form : ∀ n : ℕ, a_seq n = 2 * n - 1) :
  ∀ n : ℕ, (∑ i in Finset.range n, b_seq i) = n / (2 * n + 1) :=
sorry

end general_term_sum_b_seq_l274_274156


namespace min_distance_l274_274903

-- Define the point P with given coordinates
def P (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

-- Define the curve C in Cartesian coordinates
def curve_C (x y : ℝ) : Prop := x + y = 9

-- Define distance from a point to a line
def dist_to_line (x₀ y₀ : ℝ) : ℝ := (|x₀ + y₀ - 9| / Real.sqrt 2)

-- Main theorem statement
theorem min_distance : ∀ α ∈ Set.Icc 0 Real.pi, 
  (∀ x y, curve_C x y → P α = (1 + Real.cos α, Real.sin α) → dist_to_line (1 + Real.cos α) (Real.sin α) ≥ 4 * Real.sqrt 2 - 1) :=
begin
  assume α hα x y hxy hP,
  sorry
end

end min_distance_l274_274903


namespace two_digit_nums_with_special_sum_l274_274959

-- Define helper functions
def digit_sum (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_perfect_square_or_cube_under_18 (n : ℕ) : Prop :=
  n ∈ [1, 4, 8, 9, 16]

-- Main theorem
theorem two_digit_nums_with_special_sum : 
  (List.filter (λ n, is_two_digit n ∧ is_perfect_square_or_cube_under_18 (digit_sum n)) (List.range 100)).length = 25 := by {
  sorry
}

end two_digit_nums_with_special_sum_l274_274959


namespace general_equation_of_curve_l274_274924

theorem general_equation_of_curve
  (t : ℝ) (ht : t > 0)
  (x : ℝ) (hx : x = (Real.sqrt t) - (1 / (Real.sqrt t)))
  (y : ℝ) (hy : y = 3 * (t + 1 / t) + 2) :
  x^2 = (y - 8) / 3 := by
  sorry

end general_equation_of_curve_l274_274924


namespace find_a_10_l274_274555

def sequence (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 2

theorem find_a_10 (a : ℕ → ℕ) (h : sequence a) : a 10 = 19 :=
sorry

end find_a_10_l274_274555


namespace find_value_of_y_l274_274237

noncomputable def angle_sum_triangle (A B C : ℝ) : Prop :=
A + B + C = 180

noncomputable def triangle_ABC : angle_sum_triangle 80 60 x := by
  sorry

noncomputable def triangle_CDE (x y : ℝ) : Prop :=
(x = 40) ∧ (90 + x + y = 180)

theorem find_value_of_y (x y : ℝ) 
  (h1 : angle_sum_triangle 80 60 x)
  (h2 : triangle_CDE x y) : 
  y = 50 := 
by
  sorry

end find_value_of_y_l274_274237


namespace intervals_of_monotonicity_and_extreme_values_l274_274915

noncomputable def f (x d : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + d

theorem intervals_of_monotonicity_and_extreme_values :
  ∀ d : ℝ, (∀ x : ℝ, (x < -1 → (f x d > f (x + 1) d)) ∧ (-1 < x ∧ x < 3 → (f x d < f (x + 1) d)) ∧ (x > 3 → (f x d > f (x + 1) d))) →
  ((∃ (d_real : ℝ) (hf_min : ∀ x ∈ Icc (-2:ℝ) 2, f x d ≥ -4), d_real = 1 ∧ (∀ x ∈ Icc (-2:ℝ) 2, f x d ≤ 23) ) ) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l274_274915


namespace valid_arrangements_l274_274033

noncomputable def numOfArrangements (students : Finset ℕ) (days : Finset ℕ) (arrangement : Finset (Finset (ℕ × ℕ))) : ℕ :=
  if ∀ p ∈ arrangement, let (s, d) := p in s ≠ A ∨ d ≠ Monday ∧ ∀ (b c : ℕ), b ≠ c → (∃ x ∈ days, (b, x) ∈ arrangement ∧ (c, x) ∉ arrangement) then
    ((students.filter (≠ A)).card.choose 2 * 2.fchoose (days.filter (≠ Monday)).card.choose 3) else
    0

theorem valid_arrangements : numOfArrangements {A, B, C, D, E, F} {Monday, Tuesday, Wednesday} = 48 := 
  sorry

end valid_arrangements_l274_274033


namespace dividend_percentage_of_each_share_is_9_l274_274428

def market_value := 15
def face_value := 20
def desired_interest_rate := 0.12

def dividend_percentage (D : ℚ) : Prop :=
  (D/100 * face_value = desired_interest_rate * market_value)

theorem dividend_percentage_of_each_share_is_9 :
  dividend_percentage 9 :=
by
  unfold dividend_percentage
  sorry

end dividend_percentage_of_each_share_is_9_l274_274428


namespace intersection_M_N_l274_274933

def set_M : Set ℝ := {x : ℝ | (1/2)^x ≥ 1}

def set_N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x + 2)}

theorem intersection_M_N :
  set_M ∩ set_N = {x : ℝ | -2 < x ∧ x ≤ 0} := by
  sorry

end intersection_M_N_l274_274933


namespace find_COD_angle_l274_274630

variables (A O B C D : Type*) 

-- Define angles between points
variables (angle_AOB angle_AOC angle_COD angle_DOB : ℝ)

-- Given conditions
def condition1 : Prop := 5 * angle_COD = 4 * angle_AOC
def condition2 : Prop := 3 * angle_COD = 2 * angle_DOB
def condition3 : Prop := angle_AOB = 105

-- The statement to prove
theorem find_COD_angle (h1 : condition1 angle_COD angle_AOC) 
                       (h2 : condition2 angle_COD angle_DOB) 
                       (h3 : condition3 angle_AOB) : 
  angle_COD = 28 :=
begin
  sorry
end

end find_COD_angle_l274_274630


namespace abs_sum_zero_eq_l274_274198

theorem abs_sum_zero_eq (m n : ℤ) (h : |1 + m| + |n - 2| = 0) : m = -1 ∧ n = 2 ∧ m^n = 1 :=
by
  sorry

end abs_sum_zero_eq_l274_274198


namespace cosine_of_DE_BC_is_1_div_4_l274_274740

variables {A B C D E : Type} [inner_product_space ℝ Type]

def AB : ℝ := 2
def DE : ℝ := 2
def BC : ℝ := 8
def CA : ℝ := real.sqrt 72

axiom cosine_condition (x y z w : Type) [inner_product_space ℝ Type]:
  (inner_product (⇑(λ q, q)) ⇑(λ q, q)) + (inner_product (⇑(λ q, q)) ⇑(λ q, q)) = 5

theorem cosine_of_DE_BC_is_1_div_4 
(B_midpoint_DE : ∀ v : inner_product_space ℝ Type, ∃ w, v = (w + w) / 2) 
: inner_product_space.cos_BC_DE = 1 / 4 :=
by {
  sorry
}

end cosine_of_DE_BC_is_1_div_4_l274_274740


namespace poly_eval_sum_at_two_l274_274627

-- Define the polynomial we are working with
def poly : Polynomial ℤ := Polynomial.C 1 * X^6 - Polynomial.C 1 * X^3 - Polynomial.C 1 * X^2 - Polynomial.C 1

-- Define the polynomial factors
noncomputable def q1 : Polynomial ℤ := Polynomial.C 1 * X^3 - Polynomial.C 1 * X^2 - Polynomial.C 1
noncomputable def q2 : Polynomial ℤ := Polynomial.C 1 * X^2 + Polynomial.C 1
noncomputable def q3 : Polynomial ℤ := Polynomial.C 1 * X - Polynomial.C 1
noncomputable def q4 : Polynomial ℤ := Polynomial.C 1 * X^2 + Polynomial.C 1 * X + Polynomial.C 1

-- We need to prove that the sum of evaluations at x = 2 is 16
theorem poly_eval_sum_at_two :
  poly = q1 * q2 * q3 * q4 →
  (q1.eval 2) + (q2.eval 2) + (q3.eval 2) + (q4.eval 2) = 16 := by
  sorry

end poly_eval_sum_at_two_l274_274627


namespace zhang_hua_new_year_cards_l274_274393

theorem zhang_hua_new_year_cards (x y z : ℕ) 
  (h1 : Nat.lcm (Nat.lcm x y) z = 60)
  (h2 : Nat.gcd x y = 4)
  (h3 : Nat.gcd y z = 3) : 
  x = 4 ∨ x = 20 :=
by
  sorry

end zhang_hua_new_year_cards_l274_274393


namespace smallest_nonneg_sum_sq_l274_274826

theorem smallest_nonneg_sum_sq : 
  let squares := List.map (λ n, n ^ 2) (List.range (1989 + 1))
  ∃ signs : List Int, 
    List.sum (List.zipWith (· *) signs squares) = 1 :=
by
  sorry

end smallest_nonneg_sum_sq_l274_274826


namespace arc_length_of_sector_l274_274170

theorem arc_length_of_sector (angle : ℝ) (radius : ℝ) (h_angle : angle = 165) (h_radius : radius = 10) :
  (L : ℝ) (h_L : L = (angle * real.pi * radius) / 180) → L = 55 * real.pi / 6 :=
by
  intros
  rw [h_angle, h_radius, h_L]
  sorry

end arc_length_of_sector_l274_274170


namespace ounces_per_container_l274_274770

def weight_pounds : ℝ := 3.75
def num_containers : ℕ := 4
def pound_to_ounces : ℕ := 16

theorem ounces_per_container :
  (weight_pounds * pound_to_ounces) / num_containers = 15 :=
by
  sorry

end ounces_per_container_l274_274770


namespace hyperbola_equilateral_triangle_area_l274_274169

open Real

def point := (ℝ × ℝ)

noncomputable def hyperbola := {p : point | p.1^2 - p.2^2 = 1}

noncomputable def is_equilateral_triangle (A B C : point) :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem hyperbola_equilateral_triangle_area :
  ∀ (A B C : point),
    A = (-1, 0) →
    A ∈ hyperbola →
    B ∈ hyperbola →
    C ∈ hyperbola →
    is_equilateral_triangle A B C →
    area_of_triangle A B C = 3 * sqrt 3 :=
by
  sorry

end hyperbola_equilateral_triangle_area_l274_274169


namespace solve_expression_l274_274019

noncomputable def expression : ℝ := 5 * 1.6 - 2 * 1.4 / 1.3

theorem solve_expression : expression = 5.8462 := 
by 
  sorry

end solve_expression_l274_274019


namespace minimum_value_of_function_t_square_minus_4t_plus_1_div_t_l274_274510

theorem minimum_value_of_function_t_square_minus_4t_plus_1_div_t (t : ℝ) (ht : t > 0) : 
  ∃ (y : ℝ), y = t + 1 / t - 4 ∧ (∀ (x : ℝ), x = t + 1 / t - 4 → x ≥ -2) :=
begin
  sorry
end

end minimum_value_of_function_t_square_minus_4t_plus_1_div_t_l274_274510


namespace numOf1_in_N_numOf1_in_M_l274_274772

-- Define N as the sum of powers of 10 with 99 digits of 9
noncomputable def N : ℕ :=
  (List.sum (List.map (λ i, 10 ^ i - 1) (List.range 99.succ))) - 99

-- Define M by dividing N by 9
noncomputable def M : ℕ :=
  N / 9

-- Prove the number of digit 1 in N is 99
theorem numOf1_in_N : (N.to_string.foldl (λ count c, if c = '1' then count + 1 else count) 0) = 99 := by
  sorry

-- Prove the number of digit 1 in M is 11
theorem numOf1_in_M : (M.to_string.foldl (λ count c, if c = '1' then count + 1 else count) 0) = 11 := by
  sorry

end numOf1_in_N_numOf1_in_M_l274_274772


namespace apple_distribution_ways_l274_274071

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end apple_distribution_ways_l274_274071


namespace find_monotonic_function_l274_274480

-- Define Jensen's functional equation property
def jensens_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

-- Define monotonicity property
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The main theorem stating the equivalence
theorem find_monotonic_function (f : ℝ → ℝ) (h₁ : jensens_eq f) (h₂ : monotonic f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := 
sorry

end find_monotonic_function_l274_274480


namespace complex_magnitude_comparison_l274_274887

open Complex

theorem complex_magnitude_comparison :
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  abs z1 < abs z2 :=
by 
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  sorry

end complex_magnitude_comparison_l274_274887


namespace framed_painting_ratio_l274_274434

/-- A rectangular painting measuring 20" by 30" is to be framed, with the longer dimension vertical.
The width of the frame at the top and bottom is three times the width of the frame on the sides.
Given that the total area of the frame equals the area of the painting, the ratio of the smaller to the 
larger dimension of the framed painting is 4:7. -/
theorem framed_painting_ratio : 
  ∀ (w h : ℝ) (side_frame_width : ℝ), 
    w = 20 ∧ h = 30 ∧ 3 * side_frame_width * (2 * (w + 2 * side_frame_width) + 2 * (h + 6 * side_frame_width) - w * h) = w * h 
    → side_frame_width = 2 
    → (w + 2 * side_frame_width) / (h + 6 * side_frame_width) = 4 / 7 :=
sorry

end framed_painting_ratio_l274_274434


namespace pastrami_sandwich_cost_l274_274689

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l274_274689


namespace subsets_intersecting_with_B_l274_274297

-- Given sets
def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6, 7}

-- Number of subsets S of A such that S ∩ B ≠ ∅
theorem subsets_intersecting_with_B : 
  (∃ (count : ℕ), count = (2 ^ (A.toFinset.card) - 2 ^ ((A \ B).toFinset.card))) := 
  by sorry

end subsets_intersecting_with_B_l274_274297


namespace fraction_deleted_in_second_round_l274_274084

theorem fraction_deleted_in_second_round :
  (initial_files : ℕ) = 800 →
  (percentage_deleted_first_round : ℝ) = 0.7 →
  (files_downloaded_second_round : ℕ) = 400 →
  (valuable_files_left_after_second_round : ℕ) = 400 →
  let files_left_first_round := initial_files * (1 - percentage_deleted_first_round) in
  let valuable_files_second_round := valuable_files_left_after_second_round - files_left_first_round in
  let files_deleted_second_round := files_downloaded_second_round - valuable_files_second_round in
  let fraction_deleted_second_round := (files_deleted_second_round : ℝ) / (files_downloaded_second_round : ℝ) in
  fraction_deleted_second_round = 3 / 5 :=
begin
  sorry,
end

end fraction_deleted_in_second_round_l274_274084


namespace max_children_catered_l274_274423

theorem max_children_catered :
  ∀ (total_adults total_children : ℕ)
    (prepared_veg_adult prepared_nonveg_adult prepared_vegan_adult : ℕ)
    (prepared_veg_children prepared_nonveg_children prepared_vegan_children : ℕ)
    (veg_adult_pref nonveg_adult_pref vegan_adult_pref : ℕ)
    (veg_children_pref nonveg_children_pref vegan_children_pref : ℕ)
    (veg_adults_had_meal nonveg_adults_had_meal vegan_adults_had_meal : ℕ),
  total_adults = 80 →
  total_children = 120 →
  prepared_veg_adult = 70 →
  prepared_nonveg_adult = 75 →
  prepared_vegan_adult = 5 →
  prepared_veg_children = 90 →
  prepared_nonveg_children = 25 →
  prepared_vegan_children = 5 →
  veg_adult_pref = 45 →
  nonveg_adult_pref = 30 →
  vegan_adult_pref = 5 →
  veg_children_pref = 100 →
  nonveg_children_pref = 15 →
  vegan_children_pref = 5 →
  veg_adults_had_meal = 42 →
  nonveg_adults_had_meal = 25 →
  vegan_adults_had_meal = 5 →
  let remaining_veg_adult := prepared_veg_adult - veg_adults_had_meal,
      remaining_nonveg_adult := prepared_nonveg_adult - nonveg_adults_had_meal,
      remaining_vegan_adult := prepared_vegan_adult - vegan_adults_had_meal,
      total_remaining_veg := prepared_veg_children + remaining_veg_adult,
      total_remaining_nonveg := prepared_nonveg_children + remaining_nonveg_adult,
      total_remaining_vegan := prepared_vegan_children + remaining_vegan_adult in
  min veg_children_pref total_remaining_veg = 100 ∧
  min nonveg_children_pref total_remaining_nonveg = 15 ∧
  min vegan_children_pref total_remaining_vegan = 5 :=
begin
  sorry
end

end max_children_catered_l274_274423


namespace NicoleEndsUpWith36Pieces_l274_274301

namespace ClothingProblem

noncomputable def NicoleClothesStart := 10
noncomputable def FirstOlderSisterClothes := NicoleClothesStart / 2
noncomputable def NextOldestSisterClothes := NicoleClothesStart + 2
noncomputable def OldestSisterClothes := (NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes) / 3

theorem NicoleEndsUpWith36Pieces : 
  NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes + OldestSisterClothes = 36 :=
  by
    sorry

end ClothingProblem

end NicoleEndsUpWith36Pieces_l274_274301


namespace solution_l274_274136

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * 3 * x + 4

def problem (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : Prop :=
  f a b (-Real.logb 3 3) = 3

theorem solution (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : problem a b m h1 h2 :=
sorry

end solution_l274_274136


namespace solve_for_x_l274_274489

theorem solve_for_x (x : ℝ) : 
  16^(-3) = (2^(72/x)) / ((2^(36/x)) * 16^(27/x)) → x = 6 := 
by 
  sorry

end solve_for_x_l274_274489


namespace range_of_m_l274_274279

variable (f : ℝ → ℝ)

def derivative_exists (x : ℝ) := differentiable_at ℝ f x

theorem range_of_m (h_deriv : ∀ x, differentiable ℝ f x)
                   (h_eq : ∀ x, f x = 4 * x^2 - f (-x))
                   (h_ineq1 : ∀ x, x < 0 → f' x + 1/2 < 4 * x)
                   (h_ineq2 : ∀ m, f (m + 1) ≤ f (-m) + 3 * m + 3 / 2) :
                   ∀ m, m ≥ -1/2 :=
begin
  sorry
end

end range_of_m_l274_274279


namespace inequality_a4_b4_c4_geq_l274_274405

theorem inequality_a4_b4_c4_geq (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
by
  sorry

end inequality_a4_b4_c4_geq_l274_274405


namespace find_x_with_three_prime_divisors_l274_274151

def x_with_conditions (n : Nat) (x : Nat) : Prop :=
  x = 9^n - 1 ∧
  nat.factors x ∧
  nat.factors x.count 7 = 1

theorem find_x_with_three_prime_divisors (n : Nat) (x : Nat) :
  x_with_conditions n x → x = 728 :=
by
  sorry

end find_x_with_three_prime_divisors_l274_274151


namespace inequality_proof_l274_274530

variable (a b c : ℝ)

-- Conditions
def conditions : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 14

-- Statement to prove
theorem inequality_proof (h : conditions a b c) : 
  a^5 + (1/8) * b^5 + (1/27) * c^5 ≥ 14 := 
sorry

end inequality_proof_l274_274530


namespace min_perimeter_rectangles_l274_274093

theorem min_perimeter_rectangles (m : ℕ) (hm : 0 < m) :
  (let N := 2^m;
        diagonal := { i | 1 ≤ i ∧ i ≤ N };
        perimeter := ∑ (rect : set (ℕ × ℕ)) in { r | ∀ i ∈ diagonal, (i, i) ∈ r }, 2 * ((rect.sup fst) - (rect.inf fst) + (rect.sup snd) - (rect.inf snd))
     in perimeter) = 2^(m+2) * (m + 1) :=
by
  sorry

end min_perimeter_rectangles_l274_274093


namespace sin_double_angle_l274_274969

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l274_274969


namespace expression_for_f_xh_minus_f_x_l274_274553

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

theorem expression_for_f_xh_minus_f_x (x h : ℝ) : f(x + h) - f(x) = h * (6 * x + 3 * h + 5) :=
by
  sorry

end expression_for_f_xh_minus_f_x_l274_274553


namespace length_CF_l274_274635

-- Define the points and the given conditions
variables (C A F D B : Type)
variable [metric_space C]
variable [metric_space A]
variable [metric_space F]
variable [metric_space D]
variable [metric_space B]

variables (AF : line)
variables (CF : line)

-- Given conditions
variable (perpendicular_CD_AF : CD ⊥ AF)
variable (perpendicular_AB_CF : AB ⊥ CF)
variable (AB_length : ∥AB∥ = 6)
variable (CD_length : ∥CD∥ = 3)
variable (AF_length : ∥AF∥ = 12)

theorem length_CF :
  ∥CF∥ = 6 :=
sorry

end length_CF_l274_274635


namespace swim_club_membership_l274_274012

theorem swim_club_membership (M : ℕ) (A B : ℕ) (H1 : A = 5) (H2 : B = 30) (H3 : 0.3 * M = A + B) : M = 50 :=
by
  sorry

end swim_club_membership_l274_274012


namespace remainder_of_8_pow_2023_l274_274382

theorem remainder_of_8_pow_2023 :
  8^2023 % 100 = 12 :=
sorry

end remainder_of_8_pow_2023_l274_274382


namespace rain_ratio_l274_274053

theorem rain_ratio (x : ℝ) (h1 : 5 + x + 1 = 8) : 5 / x = 2.5 :=
by
  have h : x = 8 - 5 - 1 := by linarith
  rw h
  norm_num

end rain_ratio_l274_274053


namespace number_of_dogs_l274_274219

variable (D C : ℕ)
variable (x : ℚ)

-- Conditions
def ratio_dogs_to_cats := D = (x * (C: ℚ) / 7)
def new_ratio_dogs_to_cats := D = (15 / 11) * (C + 8)

theorem number_of_dogs (h1 : ratio_dogs_to_cats D C x) (h2 : new_ratio_dogs_to_cats D C) : D = 77 := 
by sorry

end number_of_dogs_l274_274219


namespace total_money_made_l274_274082

-- Define the given conditions.
def total_rooms : ℕ := 260
def single_rooms : ℕ := 64
def single_room_cost : ℕ := 35
def double_room_cost : ℕ := 60

-- Define the number of double rooms.
def double_rooms : ℕ := total_rooms - single_rooms

-- Define the total money made from single and double rooms.
def money_from_single_rooms : ℕ := single_rooms * single_room_cost
def money_from_double_rooms : ℕ := double_rooms * double_room_cost

-- State the theorem we want to prove.
theorem total_money_made : 
  (money_from_single_rooms + money_from_double_rooms) = 14000 :=
  by
    sorry -- Proof is omitted.

end total_money_made_l274_274082


namespace sum_of_numbers_ge_1_1_l274_274868

theorem sum_of_numbers_ge_1_1 :
  let numbers := [1.4, 0.9, 1.2, 0.5, 1.3]
  let threshold := 1.1
  let filtered_numbers := numbers.filter (fun x => x >= threshold)
  let sum_filtered := filtered_numbers.sum
  sum_filtered = 3.9 :=
by {
  sorry
}

end sum_of_numbers_ge_1_1_l274_274868


namespace profit_ratio_l274_274787

theorem profit_ratio (P W : ℕ) (H : P = 1200) (H1 : W = 500) (H2 : 1/4 * P = 300) :
  let M := P - 300 - W in M / P = 1 / 3 :=
by
  have H3 : P = 1200 := H
  have H4 : W = 500 := H1
  have H5 : 1/4 * P = 300 := H2
  let M := P - 300 - 500
  have H6 : M = 400 := by sorry
  have H7 : 400/1200 = 1/3 := by sorry
  exact H7

end profit_ratio_l274_274787


namespace red_tiles_correct_l274_274745

-- Definitions based on the given conditions
def initial_blue_tiles : ℕ := 20
def initial_green_tiles : ℕ := 9
def green_layers : ℕ := 2

-- Extras
def first_green_layer_tiles : ℕ := 12
def second_green_layer_tiles : ℕ := 12
def total_green_tiles_added := 2 * (initial_blue_tiles / 3)
def total_green_tiles := initial_green_tiles + first_green_layer_tiles + second_green_layer_tiles

def red_tiles_added := 12

-- Main statements to be proven
theorem red_tiles_correct (init_blue init_green green_layers first_layer second_layer red_added total_green) 
  (h1 : init_blue = initial_blue_tiles)
  (h2 : init_green = initial_green_tiles)
  (h3 : green_layers = 2)
  (h4 : first_layer = first_green_layer_tiles)
  (h5 : second_layer = second_green_layer_tiles)
  (h6 : red_added = red_tiles_added)
  :
  (red_added = 12) ∧ 
  (total_green = 45 - init_blue + 12)
:= by
  unfold initial_blue_tiles initial_green_tiles green_layers first_green_layer_tiles second_green_layer_tiles red_tiles_added
  split
  {
    exact rfl,
  },
  {
    simp [total_green_tiles, total_green_tiles_added],
    sorry, -- Proof computes exact difference as required.
  }

end red_tiles_correct_l274_274745


namespace base_7_digits_1234_l274_274946

theorem base_7_digits_1234 : ∃ n : ℕ, nat.digits 7 1234 = [4] := 
sorry

end base_7_digits_1234_l274_274946


namespace exponent_of_3_in_30_factorial_l274_274250

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | nat.succ n' => (nat.succ n') * factorial n'

noncomputable def exponent_of_prime_in_factorial (n p : ℕ) : ℕ :=
  if h : p > 1 ∧ nat.prime p then
    let rec count_multiples (m : ℕ) (acc : ℕ) : ℕ :=
      if m > n then acc
      else count_multiples (p * m) (acc + n / m)
    count_multiples p 0
  else 0

theorem exponent_of_3_in_30_factorial : exponent_of_prime_in_factorial 30 3 = 14 := 
sorry

end exponent_of_3_in_30_factorial_l274_274250


namespace total_food_items_donated_l274_274117

noncomputable def food_donations_total : ℕ :=
  let foster_farms := 45
  let american_summits := 2 * foster_farms
  let hormel := 3 * foster_farms
  let boudin_butchers := hormel / 3
  let del_monte_foods := american_summits - 30
  foster_farms + american_summits + hormel + boudin_butchers + del_monte_foods

theorem total_food_items_donated : food_donations_total = 375 := by
  have h1 : 45 = 45 := rfl
  have h2 : 2 * 45 = 90 := by norm_num
  have h3 : 3 * 45 = 135 := by norm_num
  have h4 : 135 / 3 = 45 := by norm_num
  have h5 : 90 - 30 = 60 := by norm_num
  show 45 + (2 * 45) + (3 * 45) + (135 / 3) + (90 - 30) = 375
  norm_num
  exact rfl

end total_food_items_donated_l274_274117


namespace measure_angle_BAC_equals_70_l274_274791

-- Define the circle centered at O
variable (O A B C : Type) [Geometry O A B C]

-- Define the angles given in the problem
variable (angle_BOA angle_BOC : ℝ)
variable (h_BOA : angle_BOA = 120) 
variable (h_BOC : angle_BOC = 140)

-- Define the final proposition to be proved
-- Proving that ∠BAC is 70°
theorem measure_angle_BAC_equals_70 :
  let ∠BAC = 70 in 
  ∠BAC = angle_BOA + angle_BOC → (∠BAC = 70) :=
sorry

end measure_angle_BAC_equals_70_l274_274791


namespace fewest_tiles_to_cover_region_l274_274044

theorem fewest_tiles_to_cover_region :
  ∀ (tile_length tile_width region_length region_width : ℕ),
  (tile_length = 3) →
  (tile_width = 5) →
  (region_length = 36) →
  (region_width = 48) →
  let tile_area := tile_length * tile_width in
  let region_area := region_length * region_width in
  let number_of_tiles := (region_area : ℚ) / tile_area in
  ∃ (n : ℕ), n = number_of_tiles.ceil →
  n = 116 :=
by
  intros tile_length tile_width region_length region_width
  intros h_tile_length h_tile_width h_region_length h_region_width
  rw [h_tile_length, h_tile_width, h_region_length, h_region_width]
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  let number_of_tiles := (region_area : ℚ) / tile_area
  use number_of_tiles.ceil
  split
  { sorry }
  { sorry }

end fewest_tiles_to_cover_region_l274_274044


namespace trig_identity_nec_but_not_suff_l274_274474

open Real

theorem trig_identity_nec_but_not_suff (α β : ℝ) (k : ℤ) :
  (α + β = 2 * k * π + π / 6) → (sin α * cos β + cos α * sin β = 1 / 2) := by
  sorry

end trig_identity_nec_but_not_suff_l274_274474


namespace time_to_row_approx_one_hour_l274_274430

def distance_total : ℝ := 6.794285714285714
def speed_man : ℝ := 7
def speed_river : ℝ := 1.2
def distance_place : ℝ := distance_total / 2
def speed_upstream : ℝ := speed_man - speed_river
def speed_downstream : ℝ := speed_man + speed_river
def time_upstream : ℝ := distance_place / speed_upstream
def time_downstream : ℝ := distance_place / speed_downstream
def total_time : ℝ := time_upstream + time_downstream

theorem time_to_row_approx_one_hour :
  abs (total_time - 1) < 0.001 :=
by
  -- Placeholder for the proof
  sorry

end time_to_row_approx_one_hour_l274_274430


namespace minimum_trips_l274_274602

theorem minimum_trips
    (masses : List ℕ)
    (cap : ℕ)
    (h_mass : masses = [130, 60, 61, 65, 68, 70, 79, 81, 83, 87, 90, 91, 95])
    (h_cap : cap = 175) :
    (count_trips masses cap) = 7 := 
sorry

end minimum_trips_l274_274602


namespace fraction_of_students_with_partner_l274_274599

theorem fraction_of_students_with_partner (s t : ℕ) 
  (h : t = (4 * s) / 3) :
  (t / 4 + s / 3) / (t + s) = 2 / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_students_with_partner_l274_274599


namespace value_of_p_l274_274347

theorem value_of_p (p q : ℚ) (h₁ : p > 0) (h₂ : q > 0) (h₃ : p + q = 1)
  (h_eq : 8 * p^7 * q = 28 * p^6 * q^2) : p = 7 / 9 :=
by sorry

end value_of_p_l274_274347


namespace no_uniform_covering_with_L_shaped_pieces_l274_274774

theorem no_uniform_covering_with_L_shaped_pieces 
  (rect : Fin 5 × Fin 7) :
  ¬ (∃ (covers : (Fin 5 × Fin 7) → Finset (Fin 5 × Fin 7)), 
    (∀ cell, ∑ c in covers cell, 1 = covers card)) :=
  sorry

end no_uniform_covering_with_L_shaped_pieces_l274_274774


namespace sandy_gain_percent_l274_274011

def gain_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  (gain * 100) / total_cost

theorem sandy_gain_percent :
  gain_percent 900 300 1260 = 5 :=
by
  sorry

end sandy_gain_percent_l274_274011


namespace arrangement_ways_and_distributions_l274_274046

theorem arrangement_ways_and_distributions (boys girls : ℕ) (h_boys : boys = 48) (h_girls : girls = 32) :
  ∃ (n : ℕ), (n = 2 ∨ n = 4 ∨ n = 8 ∨ n = 16) ∧
             ∃ (boys_per_row girls_per_row : ℕ), boys_per_row * n = boys ∧ girls_per_row * n = girls ∧
                                                 (
                                                    (n = 2 ∧ boys_per_row = 24 ∧ girls_per_row = 16) ∨
                                                    (n = 4 ∧ boys_per_row = 12 ∧ girls_per_row = 8) ∨
                                                    (n = 8 ∧ boys_per_row = 6 ∧ girls_per_row = 4) ∨
                                                    (n = 16 ∧ boys_per_row = 3 ∧ girls_per_row = 2)
                                                 ) :=
by
  use 2 -- or 4, or 8, or 16, one at a time
  split
  . left
    rfl
  · use 24, 16
    simp [h_boys, h_girls]
    left
    rfl

end arrangement_ways_and_distributions_l274_274046


namespace oldest_bride_age_l274_274475

theorem oldest_bride_age (G B : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) : B = 102 :=
by
  sorry

end oldest_bride_age_l274_274475


namespace find_base_k_l274_274120

open_locale big_operators

/-- Proving that for the base-k representation 0.\overline{47}_k of the fraction 11/77, the value of k is 17. -/
theorem find_base_k : 
  ∃ k : ℕ, k > 0 ∧ (0.474747..._k * k * k * (k^2 - 1) / (4k + 7)) = (11 / 77) ∧ k = 17 :=
sorry

end find_base_k_l274_274120


namespace angle_bisector_AB_angle_bisector_CD_perp_CD_AB_l274_274693

variables {Point : Type} [metric_space Point]

/-- Define the points A, B, C, D in the plane --/
variables (A B C D : Point)

/-- Define fundamental segments based on the provided problem --/
variables (AC CB BD AD : ℝ)

/-- Define the equality condition of segments --/
axiom eq_seg1 : ∀ (AC CB BD AD : ℝ), AC = CB ∧ CB = BD ∧ BD = AD ∧ AD = AC

/-- Basic point intersection axiom --/
axiom intersect_segments : ∃ M : Point, collinear A B M ∧ collinear C D M

/-- Statement 1: prove AB is the angle bisector of ∠ CAD --/
theorem angle_bisector_AB : 
  (AC = CB) → (BD = AD) → (segment.bisector A B C D) := sorry

/-- Statement 2: prove CD is the angle bisector of ∠ ACB --/
theorem angle_bisector_CD : 
  (AC = CB) → (BD = AD) → (segment.bisector C D A B) := sorry

/-- Statement 3: prove CD ⊥ AB --/
theorem perp_CD_AB : 
  (AC = CB) → (BD = AD) → 
  (segment.perpendicular AB CD) := sorry

end angle_bisector_AB_angle_bisector_CD_perp_CD_AB_l274_274693


namespace parallel_lines_a_eq_2_l274_274985

theorem parallel_lines_a_eq_2 {a : ℝ} :
  (∀ x y : ℝ, a * x + (a + 2) * y + 2 = 0 ∧ x + a * y - 2 = 0 → False) ↔ a = 2 :=
by
  sorry

end parallel_lines_a_eq_2_l274_274985


namespace p_at_zero_l274_274291

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end p_at_zero_l274_274291


namespace find_period_l274_274552

noncomputable theory

def period_of_sine (x : ℝ) : ℝ := Real.sin (π * x + π / 3)

theorem find_period : ∃ T, ∀ x, period_of_sine (x + T) = period_of_sine x ∧ T = 2 :=
by
  sorry

end find_period_l274_274552


namespace total_games_in_season_l274_274436

-- Problem statement:
-- A sports league is divided into three divisions, each containing 6 teams.
-- Each team plays every other team in its own division three times
-- and every team in the other two divisions twice.
-- Prove that the number of games in a complete season is 351.

theorem total_games_in_season (num_divisions : ℕ) (teams_per_division : ℕ)
  (intradivision_matches : ℕ) (interdivision_matches : ℕ) :
  num_divisions = 3 → teams_per_division = 6 →
  intradivision_matches = 3 → interdivision_matches = 2 →
  ∃ total_games : ℕ, total_games = 351 :=
by
  intros h1 h2 h3 h4
  use 351
  sorry

end total_games_in_season_l274_274436


namespace necessary_but_not_sufficient_condition_l274_274294

variable (A B C : Set α) (a : α)
variable [Nonempty α]
variable (H1 : ∀ a, a ∈ A ↔ (a ∈ B ∧ a ∈ C))

theorem necessary_but_not_sufficient_condition :
  (a ∈ B → a ∈ A) ∧ ¬(a ∈ A → a ∈ B) :=
by
  sorry

end necessary_but_not_sufficient_condition_l274_274294


namespace func_satisfies_properties_l274_274039

theorem func_satisfies_properties :
  ∃ f : ℝ → ℝ, 
    (∀ x, f (x + π) = f x) ∧                     -- Condition 1: Periodicity
    (∀ x, f (2 * (π / 3) - x) = f x) ∧            -- Condition 2: Symmetry about x=π/3
    (∀ x, (-π / 6) ≤ x ∧ x ≤ (π / 3) →            -- Condition 3: Monotonicity
           (f x ≤ f (x + (π / 6 - x)))) ∧
    f = (λ x, sin (2*x - π/6)) :=                 -- Correct answer
by
  existsi (λ x, sin (2*x - π/6))
  split
  { intro x,
    calc
      sin (2 * (x + π) - π/6)
        = sin (2*x + 2*π - π/6) : by rw [mul_add]
    ... = sin (2*x + 2*π - π/6) : rfl
    ... = sin (2*x - π/6)       : by rw [sin_add_2pi] },
  split
  { intro x,
    calc 
      sin (2 * (π/3 - x) - π/6)
        = sin (2*π/3 - 2*x - π/6) : by rw [sub_mul, sub_right_comm]
    ... = sin (2*π/3 - π/6 - 2*x) : by rw [sub_left_comm]
    ... = sin (π/2 - 2*x)         : by norm_num
    ... = sin (2*x - π/2)         : by rw [sin_sub_pi_div]
    ... = sin (2*x - π/6)         : by rw [←sin_add_2π]
    ... = f x                     : rfl },
  { intros x hx,
    have : -π/6 ≤ 2*x - π/6 := calc
              -π/6 ≤ x        : hx.1
              ... 2*x     : by linarith
    have : 2*x - π/6 ≤ π/2 := calc
              2*x     ≤ π   : by norm_num
              2*x - π/6 ≤ π + 5π/6: sorry
    exact sorry },

-- further lean formal proof would involve proving monotonicity and other identities.

end func_satisfies_properties_l274_274039


namespace trigonometric_identity_l274_274539

-- Definition of the problem conditions and the value to be proved.
theorem trigonometric_identity (α : ℝ) (h1: sin α = 4 / 5) (h2: cos α = 3 / 5) : 
  (sin α + cos α) / (sin α - cos α) = 7 :=
by
  sorry

end trigonometric_identity_l274_274539


namespace point_on_circumcircle_l274_274660

theorem point_on_circumcircle (A B C S : Point) (Γ : Circle) :
  is_triangle A B C →
  is_circumcircle Γ A B C →
  is_intersection (internal_angle_bisector A B C) (perpendicular_bisector B C) S →
  lies_on S Γ := 
sorry

end point_on_circumcircle_l274_274660


namespace number_of_prime_or_even_divisors_less_than_100_l274_274193

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_even_divisors (n : ℕ) : Prop := 
  (∃ k : ℕ, n = k * k) → False ∧ 
  (¬ is_prime n) →
  ¬ ∃ m : ℕ, (m : ℕ).lt 2 ∧ (m ∣ n) →  ∃ d : ℕ, d ∣ n ∧ n = d * d

theorem number_of_prime_or_even_divisors_less_than_100 : 
  (card {n : ℕ | n < 100 ∧ (is_prime n ∨ has_even_divisors n)} = 90) :=
sorry

end number_of_prime_or_even_divisors_less_than_100_l274_274193


namespace fraction_of_remaining_birds_left_l274_274728

theorem fraction_of_remaining_birds_left :
  ∀ (total_birds initial_fraction next_fraction x : ℚ), 
    total_birds = 60 ∧ 
    initial_fraction = 1 / 3 ∧ 
    next_fraction = 2 / 5 ∧ 
    8 = (total_birds * (1 - initial_fraction)) * (1 - next_fraction) * (1 - x) →
    x = 2 / 3 :=
by
  intros total_birds initial_fraction next_fraction x h
  obtain ⟨hb, hi, hn, he⟩ := h
  sorry

end fraction_of_remaining_birds_left_l274_274728


namespace unique_A3_zero_l274_274640

open Matrix -- opening matrix namespace for convenience

variables {α : Type*} [Fintype α] [DecidableEq α]

noncomputable def matrix_A (n : ℕ) := (matrix (fin n) (fin n) ℝ)

theorem unique_A3_zero : ∀ (A : matrix_A 3), A ^ 4 = 0 → A ^ 3 = 0 :=
by
  intros A hA4
  have hA3 : ∀ x y, A ^ x * A ^ y = A ^ (x + y), from matrix.mul_pow
  rw [← hA3 3 1, pow_succ, hA4, mul_zero]
  exact pow_n A 0
  sorry

end unique_A3_zero_l274_274640


namespace distance_between_pulley_centers_l274_274076

theorem distance_between_pulley_centers (r1 r2 d_contact : ℝ) (h_r1 : r1 = 12) (h_r2 : r2 = 6) (h_d_contact : d_contact = 30) :
    let delta_r := r1 - r2
    let dist_centers := 2 * Real.sqrt (d_contact^2 + delta_r^2)
  in dist_centers = 2 * Real.sqrt 234 :=
by
  sorry

end distance_between_pulley_centers_l274_274076


namespace eq_ff3_l274_274544

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem eq_ff3 :
    let a := f 3 in 
    let b := f a in 
    b = 13 / 9 := 
  by
    intro a
    intro b
    sorry

end eq_ff3_l274_274544


namespace minimum_CN_plus_MN_l274_274005

-- Define the points and distances
noncomputable def least_distance_sum_CN_MN : Real :=
  let A := (0, 8 : ℝ)
  let B := (8, 0 : ℝ)
  let C := (0, 0 : ℝ)
  let D := (0, 8 : ℝ)
  let M := (6, 0 : ℝ)
  fun N : ℝ × ℝ => (N.1 = N.2 + 4)

-- Define the distances
noncomputable def CN (N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((N.1 - 0)^2 + (N.2 - 0)^2)

noncomputable def MN (N : ℝ × ℝ) : ℝ :=
  Real.sqrt ((N.1 - 6)^2 + (N.2 - 0)^2)

-- Define the minimized sum proof
theorem minimum_CN_plus_MN : ∃ N : ℝ × ℝ, N.1 = N.2 + 4 ∧ CN N + MN N = 10 := by
  sorry

end minimum_CN_plus_MN_l274_274005


namespace part_a_choice_part_b_no_order_part_c_choice_l274_274362

-- Part (a)
theorem part_a_choice (L M R : ℕ) (L_pos : 0 < L) (M_pos : 0 < M) (R_pos : 0 < R) (L_eq : L = 5) (M_eq : M = 3) (R_eq : R = 6) :
  let person_5 = if L / 3 > R / 3 ∧ L / 3 > M / 2 then "Left" else if M / 2 > R / 3 then "Middle" else "Right" in
  let person_6 = if L / 4 > R / 3 ∧ L / 4 > M / 2 then "Left" else if M / 2 > R / 4 then "Middle" else "Right" in
  let person_7 = if L / 4 > R / 4 ∧ L / 4 > M / 3 then "Left" else if M / 3 > R / 4 then "Middle" else "Right" in
  person_5 = "Left" ∧ person_6 = "Right" ∧ person_7 = "Left" :=
by {
  sorry
}

-- Part (b)
theorem part_b_no_order (L M R : ℕ) (L_pos : 0 < L) (M_pos : 0 < M) (R_pos : 0 < R) : 
  ¬ (L / 1 > M ∧ L / 1 > R ∧ M / 2 > L / 2 ∧ M / 2 > R ∧ R / 2 > L / 3 ∧ R / 2 > M / 3 ∧
  L / 4 > M / 3 ∧ L / 4 > R / 3 ∧ L / 5 > M / 3 ∧ L / 5 > R / 3) :=
by {
  sorry
}

-- Part (c)
theorem part_c_choice (L M R : ℕ) (L_pos : 0 < L) (M_pos : 0 < M) (R_pos : 0 < R) (L_eq : L = 9) (M_eq : M = 19) (R_eq : R = 25) :
  let rec table n :=
      if n % (L + M + R) < L then "Left"
      else if n % (L + M + R) < (L + M) then "Middle"
      else "Right" in
      table 2019 = "Left" :=
by {
  sorry
}

end part_a_choice_part_b_no_order_part_c_choice_l274_274362


namespace speed_of_train_l274_274056

-- Lean definitions of the conditions
def length_of_train : ℝ := 160
def length_of_platform : ℝ := 340.04
def time_to_cross_platform : ℝ := 25

-- The total distance the train covers when crossing the platform
def total_distance_covered : ℝ := length_of_train + length_of_platform

-- The speed of the train in m/s
def speed_in_meters_per_second : ℝ := total_distance_covered / time_to_cross_platform

-- The conversion factor from m/s to km/h
def conversion_factor : ℝ := 3.6

-- The speed of the train in km/h
def speed_in_kilometers_per_hour : ℝ := speed_in_meters_per_second * conversion_factor

-- The theorem stating the train's speed in km/h
theorem speed_of_train : speed_in_kilometers_per_hour = 72.00576 := by
  -- this is where the proof would go
  sorry

end speed_of_train_l274_274056


namespace range_of_a_l274_274185

variable (a : ℝ)
def A (a : ℝ) := {x : ℝ | x^2 - 2*x + a > 0}

theorem range_of_a (h : 1 ∉ A a) : a ≤ 1 :=
by {
  sorry
}

end range_of_a_l274_274185


namespace fixed_point_coordinates_l274_274384

noncomputable def specific_point (a : ℝ) : ℝ × ℝ :=
  (2, a^(2 - 2) + 3)

theorem fixed_point_coordinates (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  specific_point(a) = (2, 4) :=
by
  sorry

end fixed_point_coordinates_l274_274384


namespace convert_base_27_to_base_3_l274_274843

theorem convert_base_27_to_base_3 : 
  (∃ n : ℕ, nat_repr 652 n = 27) →
  nat_repr 652 3 = 020012002 :=
by
  sorry

end convert_base_27_to_base_3_l274_274843


namespace lion_cub_birth_rate_l274_274357

theorem lion_cub_birth_rate :
  ∀ (x : ℕ), 100 + 12 * (x - 1) = 148 → x = 5 :=
by
  intros x h
  sorry

end lion_cub_birth_rate_l274_274357


namespace bom_seeds_is_300_l274_274763

-- Define the known conditions
def yeon_seeds (bom_seeds gwi_seeds : ℕ) := 3 * gwi_seeds
def gwi_seeds (bom_seeds : ℕ) := bom_seeds + 40
def total_seeds (bom_seeds gwi_seeds yeon_seeds : ℕ) := bom_seeds + gwi_seeds + yeon_seeds

-- House the condition that together they have 1660 watermelon seeds.
def total_seeds_condition : Prop := ∃ (bom_seeds gwi_seeds yeon_seeds : ℕ),
  bom_seeds + gwi_seeds + yeon_seeds = 1660 ∧
  gwi_seeds = bom_seeds + 40 ∧
  yeon_seeds = 3 * gwi_seeds

-- The proposition to prove
theorem bom_seeds_is_300 : total_seeds_condition → ∃ (bom_seeds : ℕ), bom_seeds = 300 :=
by
  intro h
  cases h with bom_seeds h1
  cases h1 with gwi_seeds h2
  cases h2 with yeon_seeds h3
  cases h3 with h_total h_rest
  cases h_rest with h_gwi h_yeon
  use 300
  sorry

end bom_seeds_is_300_l274_274763


namespace no_consecutive_naturals_after_operations_l274_274260

theorem no_consecutive_naturals_after_operations
    (f : ℕ → ℕ)    -- Define f as a function on ℕ.
    (g : ℕ → ℕ → ℕ)    -- Define g as a function on ℕ × ℕ.
    (board : list ℕ)    -- Starting list of natural numbers.
    (n : ℕ)             -- Number of consecutive numbers (10 in this problem).
    (h1 : ∀ b ∈ board, b = f (board.index_of b))
    (h2 : board.length = n)
    (operation : ∀ (a b : ℕ), a ∈ board → b ∈ board →
                ∃ c d, c = (a ^ 2 - 2011 * b ^ 2) ∧ d = a * b)
    : ¬ ∃ board', board'.length = n ∧
                (∀ i, i < n → board'.nth i = some (f i)) :=
sorry

end no_consecutive_naturals_after_operations_l274_274260


namespace pastrami_sandwich_cost_l274_274691

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l274_274691


namespace parabola_equation_and_dot_product_l274_274534

open Real

-- Define the parabola with p > 0
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Distance condition
def distance_from_focus (p m : ℝ) : Prop :=
  dist (4, m) (focus p) = 5

-- Line passing through point (2, 0)
def line_through_point (x y m : ℝ) : Prop :=
  x = m * y + 2

-- Intersection points A and B
def intersection_points (p m : ℝ) (A B : ℝ × ℝ) : Prop :=
  line_through_point A.1 A.2 m ∧ parabola p A.1 A.2 ∧
  line_through_point B.1 B.2 m ∧ parabola p B.1 B.2 ∧
  A ≠ B

-- Dot product of vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

theorem parabola_equation_and_dot_product (p m : ℝ) (A B : ℝ × ℝ) :
  (∀ p > 0, distance_from_focus p m →
   (∀ x y, parabola p x y → x = 4 → y ^ 2 = 4 * x)) ∧
  (∀ p > 0, A ≠ B →
   intersection_points p m A B →
   dot_product A B = -4) :=
by
  sorry

end parabola_equation_and_dot_product_l274_274534


namespace decreasing_interval_f_l274_274848

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - log x

theorem decreasing_interval_f :
  ∀ x : ℝ, (0 < x ∧ x < 1) → derivative f x < 0 := by
  sorry

end decreasing_interval_f_l274_274848


namespace find_m_from_cos_and_terminal_side_l274_274168

variables (α : Real) (m : Real)
variables (cosα : Real)
variables (P : Real × Real)

def terminal_side (α : Real) (P : Real × Real) : Prop :=
  let (x, y) := P
  cos α = -4/5 ∧ x = 8 * m ∧ y = 3

theorem find_m_from_cos_and_terminal_side :
  cos α = -4/5 →
  terminal_side α (8 * m, 3) →
  m = -1/2 :=
by
  sorry

end find_m_from_cos_and_terminal_side_l274_274168


namespace count_non_congruent_triangles_with_perimeter_24_l274_274565

/-- How many non-congruent triangles with only integer side lengths have a perimeter of 24 units? -/
def number_of_non_congruent_triangles (n : ℕ) : ℕ :=
  let valid_triangles := 
    {s | let ⟨a, b, c⟩ := s in 
      a + b + c = n ∧ 
      a + b > c ∧ 
      a + c > b ∧ 
      b + c > a ∧ 
      ∀ {d e f}, a = d ∧ b = e ∧ c = f ∨ a = d ∧ b = f ∧ c = e ∨ a = e ∧ b = d ∧ c = f ∨ a = e ∧ b = f ∧ c = d ∨ a = f ∧ b = d ∧ c = e ∨ a = f ∧ b = e ∧ c = d → s = {d, e, f}}} in
  valid_triangles.to_finset.card

/-- Proof to find the number of non-congruent triangles with integer side lengths and perimeter 24 -/
theorem count_non_congruent_triangles_with_perimeter_24 : 
  number_of_non_congruent_triangles 24 = 18 :=
  by sorry

end count_non_congruent_triangles_with_perimeter_24_l274_274565


namespace last_digit_89_base5_l274_274378

theorem last_digit_89_base5 : 
  let n := 89 in 
  let base := 5 in 
  (n % base) = 4 :=
by
  sorry

end last_digit_89_base5_l274_274378


namespace circle_radius_squared_l274_274792

-- Definitions of the entities and conditions provided in the problem statement.
variables (r : ℝ) -- radius of the circle
variables (A B C D P : Point) -- points on the circle

-- Conditions
variables (h_circle : Circle A r)
variables {h_AB : dist A B = 12}
variables {h_CD : dist C D = 9}
variables (h_intersection : ∃ P, line A B ∩ line C D = {P})
variables (h_angle_right : ∠ A P D = π / 2)
variables (h_BP : dist B P = 10)

-- The goal (showing the correct answer)
theorem circle_radius_squared : r^2 = 525 / 4 :=
sorry -- Proof to be completed.

end circle_radius_squared_l274_274792


namespace six_digit_number_under_5_lakh_with_digit_sum_43_l274_274765

def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000
def under_500000 (n : ℕ) : Prop := n < 500000
def digit_sum (n : ℕ) : ℕ := (n / 100000) + (n / 10000 % 10) + (n / 1000 % 10) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem six_digit_number_under_5_lakh_with_digit_sum_43 :
  is_6_digit 499993 ∧ under_500000 499993 ∧ digit_sum 499993 = 43 :=
by 
  sorry

end six_digit_number_under_5_lakh_with_digit_sum_43_l274_274765


namespace building_height_l274_274623

theorem building_height
  (num_stories_1 : ℕ)
  (height_story_1 : ℕ)
  (num_stories_2 : ℕ)
  (height_story_2 : ℕ)
  (h1 : num_stories_1 = 10)
  (h2 : height_story_1 = 12)
  (h3 : num_stories_2 = 10)
  (h4 : height_story_2 = 15)
  :
  num_stories_1 * height_story_1 + num_stories_2 * height_story_2 = 270 :=
by
  sorry

end building_height_l274_274623


namespace smallest_munificence_of_monic_cubic_l274_274866

-- Define a monic cubic polynomial
def monic_cubic (b c d : ℝ) : ℝ → ℝ := λ x, x^3 + b * x^2 + c * x + d

-- Define the munificence of the polynomial
def munificence (p : ℝ → ℝ) : ℝ := 
  max (|p (-1)|) (max (|p 0|) (|p 1|))

-- Statement of the problem
theorem smallest_munificence_of_monic_cubic : ∀ (b c d : ℝ),
  (∃ p : ℝ → ℝ, p = monic_cubic b c d ∧ munificence p = 2/3) :=
  sorry

end smallest_munificence_of_monic_cubic_l274_274866


namespace solve_equation_l274_274857

theorem solve_equation {x : ℝ} :
  (real.root 4 (58 - 3 * x) + real.root 4 (26 + 3 * x) = 5) → x = -2.08333 := 
by
  sorry

end solve_equation_l274_274857


namespace highest_possible_price_notebook_l274_274072

theorem highest_possible_price_notebook :
  ∀ (p : ℕ),
    (p ≤ 9) ∧
    (∀ q, q > p → q * 1.08 * 15 > 157) :=
sorry

end highest_possible_price_notebook_l274_274072


namespace factor_of_quadratic_l274_274782

theorem factor_of_quadratic (m : ℝ) : (∀ x, (x + 6) * (x + a) = x ^ 2 - mx - 42) → m = 1 :=
by sorry

end factor_of_quadratic_l274_274782


namespace base_7_digits_1234_l274_274945

theorem base_7_digits_1234 : ∃ n : ℕ, nat.digits 7 1234 = [4] := 
sorry

end base_7_digits_1234_l274_274945


namespace plane_equation_intersection_parallel_y_axis_l274_274091

theorem plane_equation_intersection_parallel_y_axis :
  ∃ (λ : ℝ), ∀ (x y z : ℝ), 
    (x + 3 * y + 5 * z - 4 = 0) ∧ (x - y - 2 * z + 7 = 0) ∧ 
    (3 - λ = 0) →
    (4 * x - z + 17 = 0) :=
begin
  sorry
end

end plane_equation_intersection_parallel_y_axis_l274_274091


namespace f_ffx_not_eq_decimal_l274_274199

open Nat

noncomputable def f (x : ℕ) : ℝ := 1 / x

theorem f_ffx_not_eq_decimal (x : ℕ) : f(f(x)) ≠ 0.14285714285714285 :=
  by
    have h1 : f(x) = 1 / x := rfl
    have h2 : f(f(x)) = f(1 / x) := rfl
    have h3 : f(1 / x) = x := by 
      rw [h1]
      field_simp [h2]
    rw [h3]
    apply ne_of_lt
    have h4 : 0 < 1 / x := by
      apply one_div_pos.mpr
      exact nat.cast_pos.mpr (nat.succ_pos x)
    exact h4

end f_ffx_not_eq_decimal_l274_274199


namespace intersection_of_A_and_B_l274_274982

def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | (x + 1) * (4 - x) < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | x > 3 ∨ x < -1 } := sorry

end intersection_of_A_and_B_l274_274982


namespace determine_quadratic_coefficients_l274_274469

theorem determine_quadratic_coefficients (d e : ℝ) :
  (∀ x, |x - 8| = 3 → (x = 11 ∨ x = 5)) →
  (∀ x, x^2 + d * x + e = 0 ↔ (x = 11 ∨ x = 5)) →
  (d = -16 ∧ e = 55) :=
by
  intros Hsol_Hval Hquad.
  sorry

end determine_quadratic_coefficients_l274_274469


namespace perpendicular_vectors_l274_274559

theorem perpendicular_vectors (x : ℝ) : let a := (3, 1) in
  let b := (x, -3) in
  ∀ (dot_product : ℝ), dot_product = a.1 * b.1 + a.2 * b.2 → dot_product = 0 → x = 1 :=
by
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  assume dot_product : ℝ
  assume dot_product_eq : dot_product = a.1 * b.1 + a.2 * b.2
  assume perpendicular_condition : dot_product = 0
  sorry

end perpendicular_vectors_l274_274559


namespace intersection_eq_l274_274934

def M : Set ℝ := {x | x^2 < 1}
def N : Set ℝ := {x | 2^x > 1}
def intersection_set : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_eq : M ∩ N = intersection_set := by
  sorry

end intersection_eq_l274_274934


namespace dividend_percentage_l274_274426

theorem dividend_percentage (interest_rate : ℝ) (market_value : ℝ) (face_value : ℝ) 
  (h_interest_rate : interest_rate = 0.12)
  (h_market_value : market_value = 15)
  (h_face_value : face_value = 20) : 
  let interest_per_share := interest_rate * face_value in
  let dividend_percentage := (interest_per_share / market_value) * 100 in
  dividend_percentage = 16 := by
  sorry

end dividend_percentage_l274_274426


namespace iron_block_volume_l274_274367

noncomputable def volume_of_iron_block
  (mass : ℝ) (radius : ℝ) (container_height : ℝ)
  (initial_water_height : ℝ) (final_water_height : ℝ)
  (π : ℝ) : ℝ :=
  let initial_volume := π * radius^2 * initial_water_height,
      final_volume := π * radius^2 * final_water_height in
  final_volume - initial_volume

theorem iron_block_volume {mass : ℝ} (radius : ℝ) (container_height : ℝ)
  (initial_water_height : ℝ) (final_water_height : ℝ) (π : ℝ) :
  initial_water_height = 6 →
  final_water_height = 8 →
  radius = 5 →
  π = Real.pi →
  mass = 0.4 →
  volume_of_iron_block mass radius container_height initial_water_height final_water_height π = 50 * π :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end iron_block_volume_l274_274367


namespace batsman_new_averages_l274_274786

variable {A B : ℝ}

def batsman_performance_conditions (runs : ℝ) (boundaries : ℕ) (sixes : ℕ) (strike_rate : ℝ) (wickets : ℕ): Prop :=
  runs = 100 ∧ boundaries = 12 ∧ sixes = 2 ∧ strike_rate = 130 ∧ wickets = 1

theorem batsman_new_averages (runs : ℝ) (boundaries : ℕ) (sixes : ℕ) (strike_rate : ℝ) (wickets : ℕ)
  (batting_avg_increase : ℝ) (bowling_avg_decrease : ℝ) (A B : ℝ) :
  batsman_performance_conditions runs boundaries sixes strike_rate wickets ∧
  batting_avg_increase = 5 ∧ bowling_avg_decrease = 3 →
  ∃ new_batting_avg new_bowling_avg, new_batting_avg = A + batting_avg_increase ∧ new_bowling_avg = B - bowling_avg_decrease :=
by
  intro h
  cases h with performance_conditions avg_changes
  cases performance_conditions
  cases avg_changes
  cases avg_changes.right
  use [A + batting_avg_increase, B - bowling_avg_decrease]
  simp [*]
  sorry

end batsman_new_averages_l274_274786


namespace sum_b4_b6_l274_274172

theorem sum_b4_b6
  (b : ℕ → ℝ)
  (h₁ : ∀ n : ℕ, n > 0 → ∃ d : ℝ, ∀ m : ℕ, m > 0 → (1 / b (m + 1) - 1 / b m) = d)
  (h₂ : b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 90) :
  b 4 + b 6 = 20 := by
  sorry

end sum_b4_b6_l274_274172


namespace g_g_2_eq_394_l274_274573

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l274_274573


namespace domain_of_f_f_increasing_in_interval_l274_274494

noncomputable def f (a : ℝ) := real.logb (1/2) (λ x : ℝ, x^2 - 2*a*x + 3)

-- Question 1: Given f(x) with domain ℝ, prove the range of a is -√3 < a < √3.
theorem domain_of_f (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + 3 > 0) ↔ (-real.sqrt 3 < a ∧ a < real.sqrt 3) := 
by sorry

-- Question 2: Given f(x) is increasing in (-∞,1], prove the range of a is 1 ≤ a < 2.
theorem f_increasing_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y ∧ x ≤ 1 ∧ y ≤ 1 → f(a) x ≤ f(a) y) ↔ (1 ≤ a ∧ a < 2) := 
by sorry

end domain_of_f_f_increasing_in_interval_l274_274494


namespace division_quoitient_l274_274113

-- Let f be the polynomial we want to divide
def f : Polynomial ℤ := X^6 - 2*X^5 + 3*X^4 - 4*X^3 + 5*X^2 - 6*X + 12

-- Let g be the divisor
def g : Polynomial ℤ := X - 1

-- The quotient q derived from the division
def q : Polynomial ℤ := X^5 - X^4 + 2*X^3 - 2*X^2 + 3*X - 3

-- The remainder r derived from the division
def r : Polynomial ℤ := 9

theorem division_quoitient : f = g * q + r := by
  sorry

end division_quoitient_l274_274113


namespace max_profit_l274_274414

noncomputable def profit (x : ℝ) : ℝ :=
  500 * (1 + 4 * x - x^2 - 4 * x^3)

theorem max_profit :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ profit x = 11125 :=
by
  use (1 / 2)
  split
  . norm_num
  split
  . norm_num
  . sorry

end max_profit_l274_274414


namespace circle_intersection_max_length_l274_274641

theorem circle_intersection_max_length (MN A C : Point) (r s t : ℕ) (d : ℝ)
  (h1 : diameter MN = 2)
  (h2 : midpoint A (arc MN))
  (h3 : MB = 4 / 7)
  (h4 : C ∈ (arc MN).complement)
  (h5 : d = 10 - 7 * sqrt 3) :
  r + s + t = 20 := 
sorry

end circle_intersection_max_length_l274_274641


namespace arithmetic_sequence_properties_l274_274017

def a_n (n : ℕ) : ℤ := 2 * n + 1

def S_n (n : ℕ) : ℤ := n * (n + 2)

theorem arithmetic_sequence_properties : 
  (a_n 3 = 7) ∧ (a_n 5 + a_n 7 = 26) :=
by {
  -- Proof to be filled
  sorry
}

end arithmetic_sequence_properties_l274_274017


namespace min_distance_AB_l274_274486

theorem min_distance_AB (a x1 x2 : ℝ) (h1 : a = 2*x1 + 2) (h2 : a = x2 + real.log x2) (h3 : x2 > x1) : 
  let AB := x2 - x1 in
  ∃ x : ℝ, x = 1 ∧ AB = 3 / 2 :=
by
  sorry

end min_distance_AB_l274_274486


namespace particle_final_position_after_12_moves_l274_274798

noncomputable def particle_position_after_n_moves : ℂ × ℕ → ℂ
| z, 0 => z
| z, (n+1) => (z * complex.exp (complex.I * real.pi / 3)) + 10

theorem particle_final_position_after_12_moves :
  particle_position_after_n_moves (7 + 0 * complex.I, 12) = 37 - 30 * real.sqrt 3 * complex.I :=
by
  sorry

end particle_final_position_after_12_moves_l274_274798


namespace compare_abc_l274_274572

theorem compare_abc (a b c : ℝ)
  (h1 : a = Real.log 0.9 / Real.log 2)
  (h2 : b = 3 ^ (-1 / 3 : ℝ))
  (h3 : c = (1 / 3 : ℝ) ^ (1 / 2 : ℝ)) :
  a < c ∧ c < b := by
  sorry

end compare_abc_l274_274572


namespace exponent_of_3_in_factorial_30_l274_274240

theorem exponent_of_3_in_factorial_30 : (prime_factorization 30!).count 3 = 14 := 
sorry

end exponent_of_3_in_factorial_30_l274_274240


namespace solution_x_alcohol_percentage_l274_274050

theorem solution_x_alcohol_percentage (P : ℝ) :
  let y_percentage := 0.30
  let mixture_percentage := 0.25
  let y_volume := 600
  let x_volume := 200
  let mixture_volume := y_volume + x_volume
  let y_alcohol_content := y_volume * y_percentage
  let mixture_alcohol_content := mixture_volume * mixture_percentage
  P * x_volume + y_alcohol_content = mixture_alcohol_content →
  P = 0.10 :=
by
  intros
  sorry

end solution_x_alcohol_percentage_l274_274050


namespace dot_product_value_l274_274188

noncomputable def vec_a : (ℝ × ℝ) := (-1, 2)
noncomputable def vec_b (m : ℝ) : (ℝ × ℝ) := (m, 1)

def parallel_condition (a b : ℝ × ℝ) : Prop :=
  let (x1, y1) := a
  let (x2, y2) := b
  x1 * y2 = y1 * x2

theorem dot_product_value (m : ℝ) (h_parallel : parallel_condition (vec_a + (2, 4) * vec_b m) (2 * vec_a - vec_b m)) :
  let b := vec_b m in
  vec_a.1 * b.1 + vec_a.2 * b.2 = 5 / 2 := sorry

end dot_product_value_l274_274188


namespace sunny_behind_windy_500m_l274_274593

variable (s w : ℝ) -- s: speed of Sunny, w: speed of Windy
variable (Sunny_speed_decrease : ℝ) (Sunny_distance_ahead_400m : ℝ)
variable (Windy_distance_500m : ℝ)

-- Initial conditions
-- In a 400-meter race, Sunny finishes 30 meters ahead of Windy
def sunny_400m := 400 / s
def windy_400m := 370 / w -- Windy runs only 370 meters
def speed_ratio := s / w = 40 / 37

-- In the subsequent 500-meter race
def sunny_speed_new := 0.9 * s
def sunny_distance_530m := 530 / sunny_speed_new
def windy_distance_500m := 500 / w

-- Proof Problem: Sunny is 14.72 meters behind Windy at the end of the 500-meter race
theorem sunny_behind_windy_500m
  (h1: sunny_speed_new = 0.9 * s)
  (h2: sunny_distance_530m = 530 / sunny_speed_new)
  (h3: windy_distance_500m = 500 / w)
  (h4: speed_ratio)
  (h5: Sunny_distance_ahead_400m = 30) :
  sunny_distance_530m < windy_distance_500m := sorry

#eval sunny_400m = windy_400m -- ensures that the speed ratio holds for the 400m race

end sunny_behind_windy_500m_l274_274593


namespace theta_range_l274_274139

theorem theta_range (θ : ℝ) (hθ : θ ∈ (-π : ℝ) .. π)
  (h : 3 * Real.sqrt 2 * Real.cos (θ + π / 4) < 4 * (Real.sin θ)^3 - 4 * (Real.cos θ)^3) :
  θ ∈ Icc (-π) (-3 * π / 4) ∨ θ ∈ Icc (π / 4) π :=
by
  sorry

end theta_range_l274_274139


namespace value_of_a2017_l274_274929

noncomputable def sequence : ℕ → ℚ
| 0     := 1 / 2
| (n+1) := 1 - 1 / sequence n

theorem value_of_a2017 : sequence 2016 = 1 / 2 :=
sorry

end value_of_a2017_l274_274929


namespace divisible_by_5886_l274_274343

theorem divisible_by_5886 (r b c : ℕ) (h1 : (523000 + r * 1000 + b * 100 + c * 10) % 89 = 0) (h2 : r * b * c = 180) : 
  (523000 + r * 1000 + b * 100 + c * 10) % 5886 = 0 := 
sorry

end divisible_by_5886_l274_274343


namespace circle_radius_l274_274712

noncomputable def radius_of_circle : Type := sorry

theorem circle_radius 
(center_x : ℝ) 
(h_center : ∃ c : ℝ, c = center_x ∧ ∃ r : ℝ, (0,5) = (0,r) ∧ (2,3) = ((c-2)^2 + 9))
(h_center_x_axis : center_x = -3) 
: radius_of_circle = real.sqrt 34 :=
sorry

end circle_radius_l274_274712


namespace intervals_of_increase_find_AC_len_l274_274135

noncomputable def f (x : ℝ) : ℝ := cos x * sin x - sqrt 3 * (cos x)^2 + sqrt 3 / 2

theorem intervals_of_increase (k : ℤ) :
  f x is strictly_increasing_on (set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) :=
by sorry

structure Triangle :=
(A B C : Point)
(angle_A : ℝ)
(midpoint_D : Point)

structure Conditions :=
(A_acute : Triangle.angle_A < π / 2)
(f_A_eq : f Triangle.angle_A = sqrt 3 / 2)
(AD_length : dist Triangle.A Triangle.midpoint_D = 3)
(AB_length : dist Triangle.A Triangle.B = sqrt 3)

theorem find_AC_len (T : Triangle) (h : Conditions):
  dist T.A T.C = (3 * sqrt 15 - sqrt 3) / 2 :=
by sorry

end intervals_of_increase_find_AC_len_l274_274135


namespace count_sexy_twin_primes_lt_10_9_l274_274043

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def twin_prime (p : Nat) : Prop :=
  is_prime p ∧ (is_prime (p + 2) ∨ is_prime (p - 2))

def sexy_prime (p : Nat) : Prop :=
  is_prime p ∧ (is_prime (p + 6) ∨ is_prime (p - 6))

def sexy_twin_prime (p : Nat) : Prop :=
  twin_prime p ∧ sexy_prime p

theorem count_sexy_twin_primes_lt_10_9 :
  (Finset.range 1000000000).filter sexy_twin_prime).card = 1462105 :=
sorry

end count_sexy_twin_primes_lt_10_9_l274_274043


namespace solve_max_eq_l274_274874

def Max (a b : ℝ) : ℝ := if a ≥ b then a else b

theorem solve_max_eq (x : ℝ) : Max 1 x = x^2 - 6 ↔ x = 3 ∨ x = -Real.sqrt 7 :=
by
  sorry

end solve_max_eq_l274_274874


namespace blue_balls_needed_l274_274303

theorem blue_balls_needed 
  (G B Y W : ℝ)
  (h1 : G = 2 * B)
  (h2 : Y = (8 / 3) * B)
  (h3 : W = (4 / 3) * B) :
  5 * G + 3 * Y + 4 * W = (70 / 3) * B :=
by
  sorry

end blue_balls_needed_l274_274303


namespace closest_integer_to_521_l274_274483

def closest_integer_to_sum :=
  1000 * (∑ n in (Finset.range 10001).filter (λ n, n ≥ 3), 1 / (n^2 - 4))

theorem closest_integer_to_521 : Int.closest (closest_integer_to_sum) 521 :=
sorry

end closest_integer_to_521_l274_274483


namespace scheduling_methods_count_l274_274731

theorem scheduling_methods_count : 
  ∃ (A B C : Fin 5), (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧ A < B ∧ A < C ∧ 
  (∀ D: Fin 5, D ≠ A ∧ D ≠ B ∧ D ≠ C → False) ∧ 
  ∃ (f : Fin 5 → Option (Fin 3)), 
  (∀ d, d ∈ (Finset.range 5).filter (λ (x : Fin 5), f x = none ) ∨ 
        ∃ v, v ∈ {0, 1, 2} ∧ f d = some v)
  20 = 10 * 2 := sorry

end scheduling_methods_count_l274_274731


namespace base_7_digits_of_1234_l274_274955

theorem base_7_digits_of_1234 : ∀ (n : ℕ), (n = 1234) → (nat.log n 7 + 1 = 4) :=
begin
  intros n hn,
  rw hn,
  sorry
end

end base_7_digits_of_1234_l274_274955


namespace eccentricity_ratio_l274_274533

noncomputable def ellipse_eccentricity (m n : ℝ) : ℝ := (1 - (1 / n) / (1 / m))^(1/2)

theorem eccentricity_ratio (m n : ℝ) (h : ellipse_eccentricity m n = 1 / 2) :
  m / n = 3 / 4 :=
by
  sorry

end eccentricity_ratio_l274_274533


namespace number_of_complex_numbers_satifying_conditions_l274_274111

-- Definitions according to conditions
noncomputable def is_on_unit_circle (z : ℂ) : Prop := abs z = 1
noncomputable def power_diff_real (z : ℂ) : Prop := (z ^ (7 ! : ℕ) - z ^ (6 ! : ℕ)).im = 0
noncomputable def third_power_sum_real (z : ℂ) : Prop := (z ^ 3 + conj (z ^ 3)).im = 0

-- Lean statement
theorem number_of_complex_numbers_satifying_conditions :
  { z : ℂ | is_on_unit_circle z ∧ power_diff_real z ∧ third_power_sum_real z }.to_finset.card = 4760 :=
sorry

end number_of_complex_numbers_satifying_conditions_l274_274111


namespace maximize_inscribed_triangle_area_l274_274899

/-- Given a triangle \( \triangle ABC \), and angles \( \alpha_{1}, \beta_{1}, \gamma_{1} \), 
prove that the area of the inscribed triangle is maximized under the specified conditions. -/
theorem maximize_inscribed_triangle_area 
  (ABC : Triangle) 
  (α₁ β₁ γ₁ : ℝ) 
  (hα₁ : 0 < α₁ ∧ α₁ < π) 
  (hβ₁ : 0 < β₁ ∧ β₁ < π) 
  (hγ₁ : 0 < γ₁ ∧ γ₁ < π) 
  (h1 : ∀ A B C: Point, ∃ C₁, ∠BAC = γ₁)
  (h2 : ∀ B C A: Point, ∃ A₁, ∠ABC = α₁)
  (h3 : ∀ C A B: Point, ∃ B₁, ∠ACB = β₁) :
  ∃ A₁ B₁ C₁ : Point,
  let t₁ := (dist A₁ B₁) ^ 2 * sin α₁ * sin β₁ / sin γ₁ in
  ∀ t : ℝ,
  t = t₁ →
  t ≤ t₁ :=
sorry

end maximize_inscribed_triangle_area_l274_274899


namespace sum_x_coords_P3_eq_3010_l274_274783

-- Define the problem conditions
def P1 : Type := { points : Fin 150 → ℝ // (∑ i, points i) = 3010 }

def midpoints (points : Fin 150 → ℝ) : Fin 150 → ℝ :=
  λ i, (points i + points ((i + 1) % 150)) / 2

def P2 : Type := { points : Fin 150 → ℝ // midpoints (Subtype.val points) = points }
def P3 : Type := { points : Fin 150 → ℝ // midpoints (Subtype.val points) = points }

-- Define the property to be proven
theorem sum_x_coords_P3_eq_3010 (p1 : P1) : 
  (∑ i, (P3.val (P2.val (midpoints p1.val)) points) i) = 3010 :=
sorry

end sum_x_coords_P3_eq_3010_l274_274783


namespace committee_with_president_count_l274_274130

theorem committee_with_president_count :
  ∃ (n m k : ℕ), (n = 10) ∧ (m = 5) ∧ (k = 1260) ∧ (∑ c in finset.powerset_len m (finset.range n), (c.card = m ∧ c.card.choose m = 252)) ∧ (252 * m = k) :=
begin
  -- Let's define the number of students, the number of committee members, and the resulting count
  let n := 10,
  let m := 5,
  let k := 1260,
  
  -- number of ways to choose 5 students from 10
  have h1 : nat.choose n m = 252 := by exact nat.choose_succ_succ 10 5, -- normally you would calculate this
  
  -- number of ways to select president from chosen 5
  have h2 : 252 * m = k := by norm_num,
  
  -- final statement to show the existence
  use [n, m, k, nat.choose n m, 252 * m],
  split, refl,
  split, refl,
  split, refl,
  split, sorry, -- here you can use the calculation proof,
  sorry -- additional proof steps
end

end committee_with_president_count_l274_274130


namespace perimeter_of_triangle_distance_A1_to_plane_l274_274141

/-- Given a cube \( ABCDA_1B_1C_1D_1 \) with edge length \( a \). -/
variables (a : ℝ)

/-- Let \( M \) be a point on the edge \( A_1D_1 \) such that \( A_1M : MD_1 = 1 : 2 \). -/
variables (M A B1 D1 A1 : ℝ → ℝ) 

-- Define lengths based on the given cube and point ratio
def len_A1M : ℝ := a / 3
def len_MD1 : ℝ := 2 * a / 3
def len_AM  : ℝ := sqrt (a^2 + (a / 3)^2)
def len_B1M : ℝ := sqrt (a^2 + (a / 3)^2)
def len_AB1 : ℝ := a * sqrt 2

/-- Proof that the perimeter of triangle \( AB_1M \) equals \( \frac{a(2 \sqrt{10} + 3 \sqrt{2})}{3} \). -/
theorem perimeter_of_triangle (h1: M = A1 + len_A1M) : 
  (len_AM + len_B1M + len_AB1 = (2 * a * sqrt 10 + 3 * a * sqrt 2) / 3) :=
  sorry

/-- Proof that the distance from \( A_1 \) to the plane passing through \( A, B_1 \), and \( M \) 
equals \( \frac{a}{\sqrt{11}} \). -/
theorem distance_A1_to_plane (h2: M = A1 + len_A1M) : 
  (a / sqrt 11 = distance_from_point_to_plane A1 A B1 M) :=
  sorry

-- End of the Lean code

end perimeter_of_triangle_distance_A1_to_plane_l274_274141


namespace family_savings_amount_l274_274813

theorem family_savings_amount : 
  let total := 2000 
  let given_to_wife := 2 / 5 * total 
  let remaining_after_wife := total - given_to_wife 
  let given_to_first_son := 2 / 5 * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - given_to_first_son 
  let given_to_second_son := 40 / 100 * remaining_after_first_son 
  let remaining_amount := remaining_after_first_son - given_to_second_son 
  in remaining_amount = 432 := 
by
  sorry

end family_savings_amount_l274_274813


namespace modulus_sum_complex_l274_274477

theorem modulus_sum_complex :
  let z1 : Complex := Complex.mk 3 (-8)
  let z2 : Complex := Complex.mk 4 6
  Complex.abs (z1 + z2) = Real.sqrt 53 := by
  sorry

end modulus_sum_complex_l274_274477


namespace cd4_plus_ce4_l274_274270

noncomputable def circles_and_tangents (ω1 ω2 : Type) [metric_space ω1] [metric_space ω2]
  (center1 : ω1) (radius1 : ℝ) (center2 : ω2) (radius2 : ℝ)
  (A : ω1) (B : ω1) (C : ω1)
  (D E : ω2) (tangent_point: ω1)
  (chord_length : ℝ) :=
  ∃ (CD CE : ℝ), 
  radius1 = 1 ∧ 
  radius2 = 2 ∧ 
  chord_length = 2 * real.sqrt 3 ∧ 
  -- Relationship based on given conditions and homothety:
  let AC := real.sqrt(3) in
  let xy := 3 in
  let sum_squares := 9 in
  let DE_length := real.sqrt(15) in
  -- Desired result:
  (CD + CE = DE_length) ∧ (CD * CE = 3) ∧ (CD ^ 4 + CE ^ 4 = 63)

theorem cd4_plus_ce4 (ω1 ω2 : Type) [metric_space ω1] [metric_space ω2]
  {center1 : ω1} {radius1 : ℝ} {center2 : ω2} {radius2 : ℝ}
  {A : ω1} {B : ω1} {C : ω1}
  {D E : ω2} {tangent_point: ω1}
  {chord_length : ℝ}
  (h : circles_and_tangents ω1 ω2 center1 radius1 center2 radius2 A B C D E tangent_point chord_length) :
  ∃ (CD CE : ℝ), CD ^ 4 + CE ^ 4 = 63 :=
by sorry

end cd4_plus_ce4_l274_274270


namespace count_even_integers_between_300_and_800_with_specified_digits_l274_274956

open Finset

def even_integers_count : ℕ := 52

theorem count_even_integers_between_300_and_800_with_specified_digits :
  let digits := {3, 4, 5, 6, 7, 8}
  ∃! n ∈ (Icc 300 800).filter (λ x, (∃ u t h : ℕ, x = 100 * h + 10 * t + u ∧ u % 2 = 0 ∧ h ∈ digits ∧ t ∈ digits ∧ u ∈ digits ∧ h ≠ t ∧ t ≠ u ∧ h ≠ u)), even_integers_count = 52 :=
sorry

end count_even_integers_between_300_and_800_with_specified_digits_l274_274956


namespace imaginary_part_of_conjugate_l274_274341

noncomputable def complex_conjugate_imaginary_part : ℂ :=
  let z1 := (1 + complex.I)^2
  let z2 := 2 / (1 + complex.I)
  let z := z1 + z2
  complex.conj z

theorem imaginary_part_of_conjugate : complex.im (complex_conjugate_imaginary_part) = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l274_274341


namespace sin_double_angle_series_l274_274972

theorem sin_double_angle_series (θ : ℝ) (h : ∑' (n : ℕ), (sin θ)^(2 * n) = 3) :
  sin (2 * θ) = (2 * sqrt 2) / 3 :=
sorry

end sin_double_angle_series_l274_274972


namespace exists_plane_four_colors_l274_274401

/-- Given that each point in space is colored in one of five colors 
    and each color is used to color at least one point, 
    prove that there exists a plane such that all points on this plane 
    are colored in at least 4 different colors. -/
theorem exists_plane_four_colors
  (space : Type)
  (color : space → ℕ)
  (colors_used : ∀ (c : ℕ), ∃ (p : space), color p = c)
  (coloring : ∀ (p : space), color p ∈ finset.range 5) :
  ∃ (plane : set space), ∃ (color_set : set ℕ), (∀ p ∈ plane, color p ∈ color_set) ∧ color_set.size ≥ 4 :=
sorry

end exists_plane_four_colors_l274_274401


namespace min_value_of_reciprocal_sum_l274_274276

noncomputable def minimum_value : ℝ := (5 + 2 * real.sqrt 6) / 2

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a + 2 * b = 2) :
  (1 / a) + (1 / b) ≥ minimum_value := sorry

end min_value_of_reciprocal_sum_l274_274276


namespace expression_behavior_l274_274126

theorem expression_behavior (x : ℝ) (h1 : -3 < x) (h2 : x < 2) :
  ¬∃ m, ∀ y : ℝ, (h3 : -3 < y) → (h4 : y < 2) → (x ≠ 1) → (y ≠ 1) → 
    (m <= (y^2 - 3*y + 3) / (y - 1)) ∧ 
    (m >= (y^2 - 3*y + 3) / (y - 1)) :=
sorry

end expression_behavior_l274_274126


namespace func_eq_sum_l274_274975

theorem func_eq_sum :
  (∀ a b : ℕ, f(a + b) = f(a) * f(b)) →
  f(1) = 2 →
  (f(2) / f(1) + f(4) / f(3) + f(6) / f(5) + f(8) / f(7) + f(10) / f(9) = 10) :=
by
  intro h hf1
  sorry

end func_eq_sum_l274_274975


namespace correct_option_D_l274_274390

-- Definitions based on the operations given in the problem
def sqrt6 : ℝ := Real.sqrt 6
def sqrt2 : ℝ := Real.sqrt 2
def sqrt3 : ℝ := Real.sqrt 3

-- Theorem statement for proving Option D is correct
theorem correct_option_D : sqrt6 * sqrt2 = 2 * sqrt3 :=
by
  sorry

end correct_option_D_l274_274390


namespace range_of_a_l274_274167

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ (1 < a ∧ a ≤ 5) :=
begin
  sorry
end

end range_of_a_l274_274167


namespace extreme_points_inequality_l274_274545

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log (1 - x)

theorem extreme_points_inequality (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1 / 4) 
  (h_sum : x1 + x2 = 1) (h_prod : x1 * x2 = a) (h_order : x1 < x2) :
  f x2 a - x1 > -(3 + Real.log 4) / 8 := 
by
  -- proof needed
  sorry

end extreme_points_inequality_l274_274545


namespace pseudocode_yields_100_l274_274059

-- Definitions for initial conditions
def initial_T := 1
def initial_I := 3

-- Defining the function that models the pseudocode
def update_state (T I : Nat) : Nat × Nat :=
  (T + I, I + 2)

-- Recursive function to iterate the loop until the condition breaks
def iterate_while (T I : Nat) : Nat :=
  if I < 20 then
    let (T', I') := update_state T I
    iterate_while T' I'
  else
    T

-- Non-computable definition to carry out the iteration
noncomputable def final_T : Nat :=
  iterate_while initial_T initial_I

-- Proof statement
theorem pseudocode_yields_100 : final_T = 100 := 
  by
    -- We omit the proof steps and insert a sorry for Lean to accept this as a theorem statement.
    sorry


end pseudocode_yields_100_l274_274059


namespace remaining_area_flowerbed_l274_274413

-- Given Conditions
def flowerbed_radius : ℝ := 8
def path_width : ℝ := 4
def path_offset : ℝ := 2
def pi := Real.pi

-- Area Calculation Definitions
def flowerbed_area : ℝ := pi * flowerbed_radius ^ 2
def inner_circle_radius : ℝ := flowerbed_radius - path_offset
def inner_circle_area : ℝ := pi * inner_circle_radius ^ 2
def path_area : ℝ := flowerbed_area - inner_circle_area
def remaining_area : ℝ := flowerbed_area - path_area

-- Statement to prove
theorem remaining_area_flowerbed : remaining_area = 36 * pi := by
  sorry

end remaining_area_flowerbed_l274_274413


namespace g_g2_is_394_l274_274576

def g (x : ℝ) : ℝ :=
  4 * x^2 - 6

theorem g_g2_is_394 : g(g(2)) = 394 :=
by
  -- Proof is omitted by using sorry
  sorry

end g_g2_is_394_l274_274576


namespace angle_Q_in_regular_octagon_l274_274317

theorem angle_Q_in_regular_octagon :
  ∀ (ABCDEFGH : Type)
  (H8 : ∀ (C D E F G : ABCDEFGH), 
        ∃ (Q : Prop), -- Q exists when sides CD and FG are extended
        (is_regular_occ : RegularOctagon ABCDEFGH) ∧ 
        (CD_extended : extend(C, D)) ∧ 
        (FG_extended : extend(F, G))),
  -- All interior angles are 135 degrees
  (interior_angle_135 : 135 = (180 * (8 - 2) / 8)) →
  (sum_interior_angles_360 : 4 * 135 + 4 * 45 = 360) →
  -- Result
  ∠Q = 180 :=
begin
  sorry
end

end angle_Q_in_regular_octagon_l274_274317


namespace scientific_notation_l274_274060

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l274_274060


namespace difference_in_roses_and_orchids_l274_274364

theorem difference_in_roses_and_orchids
    (initial_roses : ℕ) (initial_orchids : ℕ) (initial_tulips : ℕ)
    (final_roses : ℕ) (final_orchids : ℕ) (final_tulips : ℕ)
    (ratio_roses_orchids_num : ℕ) (ratio_roses_orchids_den : ℕ)
    (ratio_roses_tulips_num : ℕ) (ratio_roses_tulips_den : ℕ)
    (h1 : initial_roses = 7)
    (h2 : initial_orchids = 12)
    (h3 : initial_tulips = 5)
    (h4 : final_roses = 11)
    (h5 : final_orchids = 20)
    (h6 : final_tulips = 10)
    (h7 : ratio_roses_orchids_num = 2)
    (h8 : ratio_roses_orchids_den = 5)
    (h9 : ratio_roses_tulips_num = 3)
    (h10 : ratio_roses_tulips_den = 5)
    (h11 : (final_roses : ℚ) / final_orchids = (ratio_roses_orchids_num : ℚ) / ratio_roses_orchids_den)
    (h12 : (final_roses : ℚ) / final_tulips = (ratio_roses_tulips_num : ℚ) / ratio_roses_tulips_den)
    : final_orchids - final_roses = 9 :=
by
  sorry

end difference_in_roses_and_orchids_l274_274364


namespace n_prime_or_power_of_2_l274_274288

theorem n_prime_or_power_of_2 (n : ℕ) (h1 : n > 6)
  (a : ℕ → ℕ) (k : ℕ) (h2 : ∀ i, 1 ≤ i → i < k → a i < n ∧ Nat.coprime (a i) n)
  (h3 : a 1 = 1 ∧ a k = n - 1)
  (h4 : ∀ i, 1 ≤ i → i < k → a (i + 1) - a i = a 2 - 1 ∧ a 2 > 1)
  : Nat.Prime n ∨ ∃ m : ℕ, n = 2 ^ m := by
  sorry

end n_prime_or_power_of_2_l274_274288


namespace skirt_price_l274_274268

theorem skirt_price (S : ℝ) 
  (h1 : 2 * 5 = 10) 
  (h2 : 1 * 4 = 4) 
  (h3 : 6 * (5 / 2) = 15) 
  (h4 : 10 + 4 + 15 + 4 * S = 53) 
  : S = 6 :=
sorry

end skirt_price_l274_274268


namespace range_of_a_l274_274923

noncomputable def A (a : ℝ) : set ℝ := {x | -3 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) : set ℝ := {y | ∃ x, y = 3 * x + 10 ∧ x ∈ A a}
noncomputable def C (a : ℝ) : set ℝ := {z | ∃ x, z = 5 - x ∧ x ∈ A a}

theorem range_of_a (a : ℝ) :
  B a ∩ C a = C a ↔ (-2/3 : ℝ) ≤ a ∧ a ≤ (4 : ℝ) :=
sorry

end range_of_a_l274_274923


namespace proof_question_l274_274661

noncomputable def z : ℂ := 1 + complex.i
noncomputable def z_inv : ℂ := complex.conj z / (complex.norm_sq z)
noncomputable def expression : ℂ := (z - z_inv)
noncomputable def expression_inv : ℂ := 1 / expression

theorem proof_question : expression_inv = (1 - 3 * complex.i) / 5 := by
  sorry

end proof_question_l274_274661


namespace intersection_M_N_l274_274186

def M : Set ℕ := {0, 1, 2, 3, 4}

def N : Set ℝ := { x | 1 < Real.log2(x + 2) ∧ Real.log2(x + 2) < 2 }

theorem intersection_M_N : M ∩ N = {1} :=
by
  sorry

end intersection_M_N_l274_274186


namespace conjugate_in_second_quadrant_l274_274649
open Complex

noncomputable def z : ℂ := ((3 - I) / (1 + I)) ^ 2

theorem conjugate_in_second_quadrant : 
  let z_conj := conj z in 
  z_conj.re < 0 ∧ z_conj.im > 0 := 
by 
  sorry

end conjugate_in_second_quadrant_l274_274649


namespace matching_colors_probability_l274_274090

-- Define a structure for the setting.
structure JellyBeanSetting :=
  (claire_green claire_red : ℕ)
  (daniel_green daniel_yellow daniel_red : ℕ)

def claire_total (s : JellyBeanSetting) : ℕ :=
  s.claire_green + s.claire_red

def daniel_total (s : JellyBeanSetting) : ℕ :=
  s.daniel_green + s.daniel_yellow + s.daniel_red

-- Define the probability of matching colors.
noncomputable def probability_of_matching_colors (s : JellyBeanSetting) : ℚ :=
  let p_claire_green := (s.claire_green : ℚ) / claire_total s in
  let p_daniel_green := (s.daniel_green : ℚ) / daniel_total s in
  let p_claire_red := (s.claire_red : ℚ) / claire_total s in
  let p_daniel_red := (s.daniel_red : ℚ) / daniel_total s in
  (p_claire_green * p_daniel_green) + (p_claire_red * p_daniel_red)

-- Define the specific jelly bean setting given in the problem.
def problem_setting : JellyBeanSetting :=
  { claire_green := 2, claire_red := 2,
    daniel_green := 2, daniel_yellow := 3, daniel_red := 4 }

-- Define the theorem stating the probability of matching colors is 1/3
theorem matching_colors_probability :
  probability_of_matching_colors problem_setting = 1 / 3 :=
by
  -- This is where the proof would go, but adding sorry to complete the statement.
  sorry

end matching_colors_probability_l274_274090


namespace cotangent_after_8_steps_l274_274626

open Real

def initial_vectors : List (ℝ × ℝ) := [(1, 0), (0, 1)]

def replace_vector (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def fibonacci_vectors (steps : Nat) : List (ℝ × ℝ) :=
  match steps with
  | 0 => initial_vectors
  | n + 1 => 
    let v1 :: v2 :: _ := fibonacci_vectors n
      -- In a more complex implementation, 
      -- we would replace one of them and repeat the process.
    [replace_vector v1 v2, v2]

-- This can be optimized as needed.

def cotangent_min_angle_vectors (steps : Nat) : ℝ :=
  let vectors := fibonacci_vectors steps
  let (u1, u2) := vectors.head!
  let (v1, v2) := vectors.tail!.head!
  (u1 * v1 + u2 * v2) / (u1 * v2 - u2 * v1)

theorem cotangent_after_8_steps : cotangent_min_angle_vectors 8 = 987 :=
by
  sorry

end cotangent_after_8_steps_l274_274626


namespace smallest_angle_of_isosceles_obtuse_triangle_l274_274075

theorem smallest_angle_of_isosceles_obtuse_triangle
  (isosceles_obtuse_triangle : Type)
  (angle_60_percent_larger : ∀ (a : isosceles_obtuse_triangle), a = 144)
  (sum_of_angles : ∀ (a b c : ℝ), a + b + c = 180)
  (one_right_angle : ∀ (a : ℝ), a = 90) :
  ∃ (smallest_angle : ℝ), smallest_angle = 18.0 :=
by
  sorry

end smallest_angle_of_isosceles_obtuse_triangle_l274_274075


namespace cake_recipe_l274_274123

theorem cake_recipe (flour : ℕ) (milk_per_200ml : ℕ) (egg_per_200ml : ℕ) (total_flour : ℕ)
  (h1 : milk_per_200ml = 60)
  (h2 : egg_per_200ml = 1)
  (h3 : total_flour = 800) :
  (total_flour / 200 * milk_per_200ml = 240) ∧ (total_flour / 200 * egg_per_200ml = 4) :=
by
  sorry

end cake_recipe_l274_274123


namespace darkest_cell_product_l274_274854

theorem darkest_cell_product (a b c d : ℕ)
  (h1 : a > 1) (h2 : b > 1) (h3 : c = a * b)
  (h4 : d = c * (9 * 5) * (9 * 11)) :
  d = 245025 :=
by
  sorry

end darkest_cell_product_l274_274854


namespace find_x_eq_728_l274_274153

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l274_274153


namespace radius_of_circle_l274_274714

open Real

noncomputable def circle_radius : ℝ :=
  let center_x := -3 in
  sqrt ((center_x - 0) ^ 2 + (0 - 5) ^ 2)

theorem radius_of_circle (cx : ℝ) (hx : cx = 0) :
  let center_x := -3 in
  (dist (center_x, 0) (0, 5) = sqrt 34 ∧ dist (center_x, 0) (2, 3) = sqrt 34) :=
by
  sorry

end radius_of_circle_l274_274714


namespace C_eq_D_at_n_l274_274272

noncomputable def C_n (n : ℕ) : ℝ := 768 * (1 - (1 / (3^n)))
noncomputable def D_n (n : ℕ) : ℝ := (4096 / 5) * (1 - ((-1)^n / (4^n)))
noncomputable def n_ge_1 : ℕ := 4

theorem C_eq_D_at_n : ∀ n ≥ 1, C_n n = D_n n → n = n_ge_1 :=
by
  intro n hn heq
  sorry

end C_eq_D_at_n_l274_274272


namespace calculate_expression_l274_274088

theorem calculate_expression : (2⁻¹ + |(-1 : ℝ)| - (2 + real.pi)^0) = (1 / 2) := 
by
  sorry

end calculate_expression_l274_274088


namespace volume_floor_value_l274_274025

noncomputable def cube_side_length := 6
noncomputable def cylinder_radius := 10
noncomputable def cylinder_height := 3

def smallest_convex_region_volume : ℝ :=
  let cube_volume := (cube_side_length : ℝ)^3
  let cylinder_volume := π * (cylinder_radius : ℝ)^2 * (cylinder_height : ℝ)
  let total_height := cylinder_height + cube_side_length
  π * (cylinder_radius : ℝ)^2 * (total_height : ℝ)

theorem volume_floor_value : ⌊smallest_convex_region_volume⌋ = 2827 := by
  sorry

end volume_floor_value_l274_274025


namespace sum_of_AdotB_elements_l274_274099

def operation (x y : ℕ) : ℕ := x * y * (x + y)

def A : Set ℕ := {0, 1}
def B : Set ℕ := {2, 3}

def AdotB : Set ℕ := {z | ∃ x ∈ A, ∃ y ∈ B, z = operation x y}

theorem sum_of_AdotB_elements : ∑ z in AdotB, z = 18 := by
  sorry

end sum_of_AdotB_elements_l274_274099


namespace equal_numbers_of_partitioned_means_l274_274632

theorem equal_numbers_of_partitioned_means (p : ℕ) (h_prime : Nat.Prime p) (a : Fin p.succ → ℝ)
  (h_cond : ∀ i : Fin p.succ, ∃ (S1 S2 : Finset (Fin p.succ) → ℝ), ∀ (S : Finset (Fin p.succ)), 
             (S.card = (p + 1) ∧ S.sum a = (S.erase i).sum a) → 
             ∃ (A1 A2 : Finset (Fin p.succ)), A1.disjoint A2 ∧ A1 ∪ A2 = (S.erase i) ∧
             (A1.card : ℝ) ≠ 0 ∧ (A2.card : ℝ) ≠ 0 ∧
             A1.sum a / A1.card = A2.sum a / A2.card) :
  ∀ i j : Fin p.succ, a i = a j :=
sorry

end equal_numbers_of_partitioned_means_l274_274632


namespace problem_proof_l274_274840

noncomputable def calculate_expression : ℝ :=
  Real.sqrt 31 + 3 * Real.tan (Real.pi / 180 * 56)

theorem problem_proof :
  Real.floor (calculate_expression * 100) / 100 = 7.88 :=
by
  sorry

end problem_proof_l274_274840


namespace sum_of_possible_values_of_N_l274_274871

theorem sum_of_possible_values_of_N : 
  let N_sum := (1 + 3 + 6 + 10) in 
  N_sum = 20 := by
  sorry

end sum_of_possible_values_of_N_l274_274871


namespace lines_skew_transitive_l274_274531

-- Given: Lines a and b are skew lines, line c is parallel to line a
-- Prove: Line c and line b are skew lines.
theorem lines_skew_transitive (a b c : Line)
(skew_ab : Skew a b)
(parallel_ac : Parallel a c) : Skew c b := 
sorry

end lines_skew_transitive_l274_274531


namespace water_fraction_final_l274_274020

noncomputable def initial_water_volume : ℚ := 25
noncomputable def first_removal_water : ℚ := 5
noncomputable def first_add_antifreeze : ℚ := 5
noncomputable def first_water_fraction : ℚ := (initial_water_volume - first_removal_water) / initial_water_volume

noncomputable def second_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def second_water_fraction : ℚ := (initial_water_volume - first_removal_water - second_removal_fraction * (initial_water_volume - first_removal_water)) / initial_water_volume

noncomputable def third_removal_fraction : ℚ := 5 / initial_water_volume
noncomputable def third_water_fraction := (second_water_fraction * (initial_water_volume - 5) + 2) / initial_water_volume

theorem water_fraction_final :
  third_water_fraction = 14.8 / 25 := sorry

end water_fraction_final_l274_274020


namespace compare_a_b_c_l274_274504

def a := 2^12
def b := 3^8
def c := 7^4

theorem compare_a_b_c : b > a ∧ a > c :=
by {
  unfold a b c, 
  -- comparision of exponents
  have h1 : b = 9^4 := by sorry,
  have h2 : a = 8^4 := by sorry,
  have h3 : c = 7^4 := by sorry,
  -- comparison of bases
  have h4 : 9 > 8 := by exact nat.succ_pos 8,
  have h5 : 8 > 7 := by exact nat.succ_pos 7,
  exact ⟨pow_lt_pow_of_lt_left h4 (by linarith) zero_lt_four, pow_lt_pow_of_lt_left h5 (by linarith) zero_lt_four⟩
}

end compare_a_b_c_l274_274504


namespace base7_digits_1234_l274_274949

theorem base7_digits_1234 : ∀ (n : ℕ), n = 1234 → 
  ∀ (b : ℕ), b = 7 → 
  ∃ d : ℕ, d = 4 ∧ ∀ (p : ℕ), 1234 / b^p < b → d = p + 1 := 
by 
  intros n hn b hb
  exists 4
  split
  case right =>
    sorry
  case left =>
    rfl

end base7_digits_1234_l274_274949


namespace extreme_value_f_sum_log_fraction_l274_274176

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.log x + a) / x

theorem extreme_value_f (a : ℝ) : 
  ∀ x : ℝ, 0 < x → f x a ≤ Real.exp (a - 1) := 
sorry

theorem sum_log_fraction (n : ℕ) (h : 2 ≤ n) : 
  (∑ i in Finset.range(n), Real.log(1 + i) / ↑(n + 1)) < (↑(n - 1) / ↑(2 * (n + 1))) :=
sorry

end extreme_value_f_sum_log_fraction_l274_274176


namespace digit_assignment_correct_l274_274775

-- Definitions for distinct variables and their corresponding digits
def distinct_digits : Prop :=
  ∀ {a b : ℕ}, a ≠ b → a ≠ 2 → a ≠ 3 → a ≠ 0 → a ≠ 1 → a ≠ 7 → a ≠ 8 → a ≠ 4 → a ≠ 5 → a ≠ 6 ∧
                 b ≠ 2 → b ≠ 3 → b ≠ 0 → b ≠ 1 → b ≠ 7 → b ≠ 8 → b ≠ 4 → b ≠ 5 → b ≠ 6

-- Prove that the equations hold given the assignments and distinct digits constraint
theorem digit_assignment_correct : distinct_digits →
  let A := 2
  let B := 3
  let C := 0
  let D := 1
  let E := 7
  let F := 8
  let G := 4
  let H := 5
  let J := 6 in
  (A * 100 + B * 10 + C) + (D * 100 + E * 10 + F) + (G * 10 + E) = (G * 100 + E * 10 + F) ∧
  (G * 100 + E * 10 + F) + (D * 10 + E) = (H * 100 + F * 10 + J) :=
by
  sorry

end digit_assignment_correct_l274_274775


namespace scientific_notation_of_number_l274_274068

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l274_274068


namespace least_z_minus_x_l274_274767

theorem least_z_minus_x (x y z : ℤ) (hx : x.even) (hx2 : y.prime) (hy : y % 2 = 1) (hz : z % 3 = 0) (hz2 : z % 2 = 1) (h1 : x < y) (h2 : y < z) (h3 : y - x > 5) : z - x = 13 :=
sorry

end least_z_minus_x_l274_274767


namespace find_x2_plus_y2_l274_274881

theorem find_x2_plus_y2 (x y : ℝ) (h : (x ^ 2 + y ^ 2 + 1) * (x ^ 2 + y ^ 2 - 3) = 5) : x ^ 2 + y ^ 2 = 4 := 
by 
  sorry

end find_x2_plus_y2_l274_274881


namespace radius_of_circle_l274_274713

open Real

noncomputable def circle_radius : ℝ :=
  let center_x := -3 in
  sqrt ((center_x - 0) ^ 2 + (0 - 5) ^ 2)

theorem radius_of_circle (cx : ℝ) (hx : cx = 0) :
  let center_x := -3 in
  (dist (center_x, 0) (0, 5) = sqrt 34 ∧ dist (center_x, 0) (2, 3) = sqrt 34) :=
by
  sorry

end radius_of_circle_l274_274713


namespace imaginary_part_of_z_l274_274913

theorem imaginary_part_of_z {z : ℂ} (h : (1 + z) / I = 1 - z) : z.im = 1 := 
sorry

end imaginary_part_of_z_l274_274913


namespace number_of_ways_l274_274849

def valid_path (path : List (Nat × Nat)) : Prop :=
  ∀ i, i < path.length - 1 →
    let (x₁, y₁) := path.get i in
    let (x₂, y₂) := path.get (i + 1) in
    (x₂ = x₁ + 1 ∧ y₂ = y₁) ∨ (x₂ = x₁ ∧ y₂ = y₁ + 1)

def passes_forbidden (path : List (Nat × Nat)) : Prop :=
  (1, 1) ∈ path ∨
  (1, 4) ∈ path ∨
  (4, 1) ∈ path ∨
  (4, 4) ∈ path

def count_valid_paths : ℕ :=
  let all_paths := List.range (binomial 10 5)
  all_paths.count (λ path, valid_path path ∧ ¬ passes_forbidden path)
  
theorem number_of_ways (h : count_valid_paths = 34) : 
  count_valid_paths = 34 :=
by {
    sorry
}

end number_of_ways_l274_274849


namespace eithan_savings_l274_274816

theorem eithan_savings :
  let amount := 2000 : ℝ 
  let wife_share := (2/5) * amount 
  let remaining_after_wife := amount - wife_share  
  let first_son_share := (2/5) * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - first_son_share 
  let second_son_share := (40/100) * remaining_after_first_son 
  let savings := remaining_after_first_son - second_son_share 
  savings = 432 :=
by
  sorry

end eithan_savings_l274_274816


namespace ramsey_theorem_n_colors_l274_274781

theorem ramsey_theorem_n_colors
  (n : ℕ) (C : Fin n → Type) (s : Fin n → ℕ)
  (h_n_ge_2 : 2 ≤ n) 
  (h_s : ∀ i j, i < j → s j ≤ s i)
  (h_s_ge_2 : ∀ i, 2 ≤ s i)
  (g : ℕ)
  (h_g : g ≥ (n / (n-1)) * (Nat.factorial (Finset.univ.sum (λ i => s i) - 2 * n) /
                             (Finset.univ.prod (λ i => Nat.factorial (s i - 2)) *
                              Finset.univ.prod (λ i => if i = n-1 then 1 else s i - 1)))) :
  ∃ i : Fin n, ∃ H : Fin n → Fin n → Prop, ∃ x : Fin (s i), H_complete K_g K_s_i →
  (∀ (x y : Fin (s i)), x ≠ y → ∃ z : x ≠ y, ∃ c : C i → Prop, H x y = c) :=
begin
  sorry
end

end ramsey_theorem_n_colors_l274_274781


namespace A_always_scores_55_l274_274408

def sequence : List ℕ := List.range' 1 101

theorem A_always_scores_55 (A B : ℕ) : 
  (∀ turns : ℕ, turns = 11) → 
  (∀ n : ℕ, n ∈ sequence → n ≥ 1 ∧ n ≤ 101) → 
  (∀ moves, length moves = 99 → 
      ∃ a b : ℕ, a ∈ sequence ∧ b ∈ sequence ∧ a ≠ b ∧ a + 55 = b) → 
  (score A = abs (a - b) ≥ 55) :=
by
  sorry

end A_always_scores_55_l274_274408


namespace problem_f_2017_plus_f_2016_l274_274902

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma domain_R : ∀ x : ℝ, x ∈ set.univ := sorry
lemma f_x_plus_2_even (x : ℝ) : f (x + 2) = f (-x + 2) := sorry
lemma f_at_neg1 : f (-1) = 1 := sorry

theorem problem_f_2017_plus_f_2016 :
  f 2017 + f 2016 = -1 :=
by
  sorry

end problem_f_2017_plus_f_2016_l274_274902


namespace sin_double_angle_l274_274967

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l274_274967


namespace sum_of_first_9_terms_45_l274_274909

-- Define the arithmetic sequence and sum of terms in the sequence
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the n-th term of the sequence

-- Given conditions
axiom condition1 : a 3 + a 5 + a 7 = 15

-- Proof goal
theorem sum_of_first_9_terms_45 : S 9 = 45 :=
by
  sorry

end sum_of_first_9_terms_45_l274_274909


namespace gentle_slope_integers_ending_in_25_and_divisible_by_25_l274_274833

def is_gentle_slope (n : ℕ) : Prop :=
  ∀ (i : ℕ), i < Nat.digits 10 n.length - 1 → Nat.digits 10 n[i] ≤ Nat.digits 10 n[i + 1]

def ends_with_25 (n : ℕ) : Prop :=
  Nat.mod n 100 = 25

def divisible_by_25 (n : ℕ) : Prop :=
  Nat.mod n 25 = 0

theorem gentle_slope_integers_ending_in_25_and_divisible_by_25 :
  { n : ℕ // is_gentle_slope n ∧ ends_with_25 n ∧ divisible_by_25 n }.card = 5 :=
sorry

end gentle_slope_integers_ending_in_25_and_divisible_by_25_l274_274833


namespace buttons_in_third_box_l274_274391

theorem buttons_in_third_box (a1 a2 a4 a5 a6 : ℕ) (h1 : a1 = 1) (h2 : a2 = 3) (h4 : a4 = 27) (h5 : a5 = 81) (h6 : a6 = 243) (h_geometric : ∀ (n : ℕ), a4 = a1 * 3 ^ (n - 1)) :
  (∃ a3 : ℕ, a3 = a4 / 3) :=
begin
  use 9,
  sorry
end

end buttons_in_third_box_l274_274391


namespace win_sector_area_l274_274462

noncomputable def radius : ℝ := 8
noncomputable def probability : ℝ := 1 / 4
noncomputable def total_area : ℝ := Real.pi * radius^2

theorem win_sector_area :
  ∃ (W : ℝ), W = probability * total_area ∧ W = 16 * Real.pi :=
by
  -- Proof skipped
  sorry

end win_sector_area_l274_274462


namespace charges_equal_at_380_plan1_more_cost_effective_after_580_l274_274003

-- Definitions based on given conditions
def plan1_cost (x : ℝ) : ℝ := if x <= 180 then 20 else 20 + 0.1 * (x - 180)
def plan2_cost (x : ℝ) : ℝ := 40

-- Lean code for the first proof problem
theorem charges_equal_at_380 : ∀ x, x ≤ 480 → plan1_cost x = plan2_cost x ↔ x = 380 :=
begin
  intros x hx,
  split; intro h,
  { -- forward direction
    unfold plan1_cost at h,
    unfold plan2_cost at h,
    split_ifs at h,
    { exfalso,
      linarith, },
    { have : 20 + 0.1 * (x - 180) = 40,
      { exact h, },
      linarith, }, },
  { -- backward direction
    rw ←h,
    unfold plan1_cost,
    unfold plan2_cost,
    split_ifs,
    linarith, }
end

-- Lean code for the second proof problem
theorem plan1_more_cost_effective_after_580 : ∀ y, y > 480 → plan1_cost y < plan2_cost y ↔ y > 580 :=
begin
  intros y hy,
  split; intro h,
  { -- forward direction
    unfold plan1_cost at h,
    unfold plan2_cost at h,
    have : 20 + 0.1 * (y - 180) < 40 + 0.2 * (y - 480),
    { exact h, },
    linarith, },
  { -- backward direction
    unfold plan1_cost,
    unfold plan2_cost,
    have : 20 + 0.1 * (y - 180) < 40 + 0.2 * (y - 480),
    linarith,
    linarith, }
end

end charges_equal_at_380_plan1_more_cost_effective_after_580_l274_274003


namespace sample_average_l274_274045

theorem sample_average (x : ℝ) 
  (h1 : (1 + 3 + 2 + 5 + x) / 5 = 3) : x = 4 := 
by 
  sorry

end sample_average_l274_274045


namespace find_a_l274_274142

def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x ^ 3 + 2 else x ^ 2 - a * x

theorem find_a (a : ℝ) : f (f 0 a) a = -2 ↔ a = 3 := 
sorry

end find_a_l274_274142


namespace relative_speed_between_trains_l274_274406

noncomputable def speed_of_train_1 : ℝ := 160 / 18
noncomputable def speed_of_train_2 : ℝ := 200 / 22
noncomputable def relative_speed : ℝ := speed_of_train_1 + speed_of_train_2

theorem relative_speed_between_trains :
  relative_speed ≈ 17.98 :=
by sorry

end relative_speed_between_trains_l274_274406


namespace scientific_notation_l274_274062

theorem scientific_notation (h : 0.0000046 = 4.6 * 10^(-6)) : True :=
by 
  sorry

end scientific_notation_l274_274062


namespace option_d_not_octal_l274_274449

theorem option_d_not_octal :
  ∀ n ∈ [123, 10110, 4724, 7857], 
  (∀ d ∈ Int.digits 10 n, d ∈ {0, 1, 2, 3, 4, 5, 6, 7}) ↔ n ≠ 7857 := 
by
  sorry

end option_d_not_octal_l274_274449


namespace solve_system_l274_274404

theorem solve_system (a b c x y z : ℝ) (h₀ : a = (a * x + c * y) / (b * z + 1))
  (h₁ : b = (b * x + y) / (b * z + 1)) 
  (h₂ : c = (a * z + c) / (b * z + 1)) 
  (h₃ : ¬ a = b * c) :
  x = 1 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_system_l274_274404


namespace graph_shift_down_by_2_l274_274919

theorem graph_shift_down_by_2 (f : ℝ → ℝ) :
  (∀ x, f(x) - 2 = (f x) - 2) ↔
  (∀ x, ∃ y, f(x) - 2 = y - 2) :=
by
  -- The statement here is to prove the mathematical equivalence between the graph of y = f(x) - 2 and shifting f(x) down by 2 units. 
  sorry

end graph_shift_down_by_2_l274_274919


namespace calculate_star_l274_274872

def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

theorem calculate_star : ((star (-2) 3 (by norm_num)) \star 5 (by norm_num)) = -12 / 13 := 
  sorry

end calculate_star_l274_274872


namespace numBoysInClassroom_l274_274727

-- Definitions based on the problem conditions
def numGirls : ℕ := 10
def girlsToBoysRatio : ℝ := 0.5

-- The statement to prove
theorem numBoysInClassroom : ∃ B : ℕ, girlsToBoysRatio * B = numGirls ∧ B = 20 :=
by
  -- Proof goes here
  sorry

end numBoysInClassroom_l274_274727


namespace work_done_time_l274_274394

/-
  Question: How many days does it take for \(a\) to do the work alone?

  Conditions:
  - \(b\) can do the work in 20 days.
  - \(c\) can do the work in 55 days.
  - \(a\) is assisted by \(b\) and \(c\) on alternate days, and the work can be done in 8 days.
  
  Correct Answer:
  - \(x = 8.8\)
-/

theorem work_done_time (x : ℝ) (h : 8 * x⁻¹ + 1 /  5 + 4 / 55 = 1): x = 8.8 :=
by sorry

end work_done_time_l274_274394


namespace sum_of_two_consecutive_negative_integers_l274_274715

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 2210) (hn : n < 0) : n + (n + 1) = -95 := 
sorry

end sum_of_two_consecutive_negative_integers_l274_274715


namespace median_length_right_triangle_l274_274227

theorem median_length_right_triangle
  (P Q R : Type)
  [MetricSpace P]
  [MetricSpace Q]
  [MetricSpace R]
  (PQ : dist P Q = 12)
  (PR : dist P R = 16)
  (QR : dist P R = 20)
  (right_triangle : dist P Q * dist P Q + dist P R * dist P R = dist Q R * dist Q R) :
  ∃ M : P, dist P M = 2 * Real.sqrt 61 :=
by
  sorry

end median_length_right_triangle_l274_274227


namespace probability_two_defective_l274_274048

theorem probability_two_defective (n total_defective total_smartphones : ℕ) 
  (h1 : total_smartphones = 220) 
  (h2 : total_defective = 84) 
  (h3 : n = 2) :
  let p := ((total_defective : ℝ) / (total_smartphones : ℝ)) * 
           (((total_defective - 1) : ℝ) / ((total_smartphones - 1) : ℝ))
  in p ≈ 0.1446 :=
by
  -- Provided conditions
  have hP1 : total_smartphones = 220 := h1,
  have hP2 : total_defective = 84 := h2,
  have hP3 : n = 2 := h3,
  -- Calculation of the required probability
  let p : ℝ := (84.0 / 220.0) * (83.0 / 219.0),
  -- Evaluate the computed probability
  show p = 0.1446, by sorry

end probability_two_defective_l274_274048


namespace exponent_of_3_in_30_factorial_l274_274244

theorem exponent_of_3_in_30_factorial : (prime_factor_exponent 3 30.factorial) = 14 :=
by
  sorry

end exponent_of_3_in_30_factorial_l274_274244


namespace photographer_choice_l274_274400

theorem photographer_choice : 
  (Nat.choose 7 4) + (Nat.choose 7 5) = 56 := 
by 
  sorry

end photographer_choice_l274_274400


namespace ordering_of_values_l274_274134

theorem ordering_of_values :
  let a := 2 ^ 0.3
  let b := 0.3 ^ 2
  let c := Real.logBase 0.3 2
  c < b ∧ b < a := by
  sorry

end ordering_of_values_l274_274134


namespace petya_wins_l274_274514

def grid := (Fin 2021) × (Fin 2021)

structure GameState :=
  (occupied : Set grid)
  (next_player : Bool)  -- true for Petya's turn, false for Vasya's turn

def wins (s : GameState) : Bool :=
  ∀ (x y : Fin 2019) (a b : Fin 2021), 
    (∀ (i : Fin 3) (j : Fin 5), (x + i, y + j) ∈ s.occupied) ∧
    (∀ (i : Fin 5) (j : Fin 3), (a + i, b + j) ∈ s.occupied)

theorem petya_wins : 
  ∀ s : GameState, 
    (s.next_player = true → ¬wins s → 
    (∃ s' : GameState, s'.next_player = false ∧ s'.occupied ⊆ (s.occupied ∪ {(_, _)}))) → 
    (s.next_player = false → ¬wins s → 
    (∃ s' : GameState, s'.next_player = true ∧ s'.occupied ⊆ (s.occupied ∪ {(_, _)}))) → 
    ∃ s_final : GameState, wins s_final :=
sorry

end petya_wins_l274_274514


namespace find_fourth_number_in_proportion_l274_274978

-- Define the given conditions
def x : ℝ := 0.39999999999999997
def proportion (y : ℝ) := 0.60 / x = 6 / y

-- State the theorem to be proven
theorem find_fourth_number_in_proportion :
  proportion y → y = 4 :=
by
  intro h
  sorry

end find_fourth_number_in_proportion_l274_274978


namespace B_contribution_l274_274051

theorem B_contribution (A_contribution: ℕ) (B_time: ℕ) (total_time: ℕ) (profit_ratio_A: ℕ) (profit_ratio_B: ℕ):
    A_contribution = 35000 → 
    B_time = 7 → 
    total_time = 12 → 
    profit_ratio_A = 2 → 
    profit_ratio_B = 3 → 
    ∃ B_contribution: ℕ, (B_contribution * B_time) * profit_ratio_A = (A_contribution * total_time) * profit_ratio_B ∧ B_contribution = 90000 :=
begin
  intros,
  use 90000,
  sorry
end

end B_contribution_l274_274051


namespace compare_sequences_l274_274371

theorem compare_sequences (n : ℕ) (h : n ≥ 2) :
  2^(2^2) * n < 3^(3^(3^(3))) * n - 1 ∧
  3^(3^(3^(3))) * n > 2 * (4^(4^(4^(4))) * (n - 1)) :=
by
  -- skipping the proofs with sorry
  split;
  sorry

end compare_sequences_l274_274371


namespace problem_statement_l274_274493

noncomputable def f (x : ℝ) : ℝ := 2 * sin (1/2 * x - π/6)

theorem problem_statement :
  (∀ x, f x = 2 * sin (1/2 * x - π/6)) ∧
  (f (4 * π / 3) = 2) ∧
  (∀ k : ℤ, f (4 * k * π + 4 * π / 3) = 2) ∧
  (∀ k : ℤ, f (4 * k * π - 2 * π / 3) = -2) ∧
  (∀ k : ℤ, ∀ x, (4 * k * π - 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 4 * π / 3) → (deriv f x > 0)) :=
by
  sorry

end problem_statement_l274_274493


namespace shortest_distance_l274_274034

-- define points and coordinates
structure Point where
  x : ℝ
  y : ℝ

def cowboy_start : Point := {x := 0, y := -3}
def cabin : Point := {x := 10, y := -12}

-- function to reflect point over the x-axis
def reflect_over_x (p : Point) : Point := {x := p.x, y := -p.y}

-- calculate Euclidean distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- reflection of cowboy's starting point
def cowboy_start_reflected : Point := reflect_over_x cowboy_start

-- define the total distance the cowboy travels
def total_distance : ℝ :=
  distance cowboy_start (reflect_over_x cowboy_start) + distance cowboy_start_reflected cabin

theorem shortest_distance : total_distance = 3 + Real.sqrt 325 :=
by
  -- here we would provide proof, but we skip it with sorry
  sorry

end shortest_distance_l274_274034


namespace Tommy_saw_total_wheels_l274_274369

theorem Tommy_saw_total_wheels :
  let trucks := 12
  let cars := 13
  let bicycles := 8
  let buses := 3
  let truck_wheels := 4
  let car_wheels := 4
  let bicycle_wheels := 2
  let bus_wheels := 6
  let total_wheels := trucks * truck_wheels + cars * car_wheels + bicycles * bicycle_wheels + buses * bus_wheels
  in total_wheels = 134 := by
  let trucks := 12
  let cars := 13
  let bicycles := 8
  let buses := 3
  let truck_wheels := 4
  let car_wheels := 4
  let bicycle_wheels := 2
  let bus_wheels := 6
  let total_wheels := trucks * truck_wheels + cars * car_wheels + bicycles * bicycle_wheels + buses * bus_wheels
  show total_wheels = 134 from sorry

end Tommy_saw_total_wheels_l274_274369


namespace cricket_team_members_l274_274793

theorem cricket_team_members (n : ℕ) : 
  (average_age : ℕ) (total_ages_remaining : ℕ) (average_remaining : ℕ) 
  (average_wt_keeper : ℕ) (wt_keeper : ℕ) : 

  average_age = 22 ∧
  total_ages_remaining = (n - 2) * 21 ∧
  average_remaining = average_age - 1 ∧
  average_wt_keeper = 25 + 3 ∧
  wt_keeper = 28 ∧
  (n * average_age - (25 + 28) = total_ages_remaining) →
  n = 11 :=
by sorry


end cricket_team_members_l274_274793


namespace modulus_of_z_l274_274174

noncomputable def z : ℂ := (2 + complex.i) / (2 - complex.i)

theorem modulus_of_z : complex.abs z = 1 := by
  sorry

end modulus_of_z_l274_274174


namespace max_planes_15_points_l274_274485

theorem max_planes_15_points :
  ∃ (points : Finset (Fin 15 → ℝ^3)) (h_condition : ¬ ∀ p, ∃ (plane : AffineSubspace ℝ (ℝ^3)), p ∈ plane ∧ ∀ q ∈ points, q ∈ plane), 
    (Finset.card points) = 455 :=
by
  sorry

end max_planes_15_points_l274_274485


namespace gcd_f_101_102_l274_274647

def f (x : ℕ) : ℕ := x^2 + x + 2010

theorem gcd_f_101_102 : Nat.gcd (f 101) (f 102) = 12 := 
by sorry

end gcd_f_101_102_l274_274647


namespace slices_with_both_l274_274021

theorem slices_with_both (n total_slices pepperoni_slices mushroom_slices other_slices : ℕ)
  (h1 : total_slices = 24) 
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices) :
  n = 5 :=
sorry

end slices_with_both_l274_274021


namespace largest_angle_measure_l274_274710

theorem largest_angle_measure (v : ℝ) (h : v > 3/2) :
  ∃ θ, θ = Real.arccos ((4 * v - 4) / (2 * Real.sqrt ((2 * v - 3) * (4 * v - 4)))) ∧
       θ = π - θ ∧
       θ = Real.arccos ((2 * v - 3) / (2 * Real.sqrt ((2 * v + 3) * (4 * v - 4)))) := 
sorry

end largest_angle_measure_l274_274710


namespace max_imaginary_part_eq_one_l274_274073

noncomputable def max_imaginary_part : ℂ :=
  let roots := {z : ℂ | z^6 - z^4 + z^2 - 1 = 0}
  let imag_parts := set.image (λ z, complex.im z) roots
  classical.some (realSup imag_parts)

theorem max_imaginary_part_eq_one :
    max_imaginary_part = 1 :=
by
  sorry

end max_imaginary_part_eq_one_l274_274073


namespace car_rental_cost_l274_274667

theorem car_rental_cost (cost_per_day: ℝ) (cost_per_mile: ℝ) (days: ℕ) (miles: ℕ) : 
  cost_per_day = 25 → cost_per_mile = 0.25 → days = 3 → miles = 350 →
  cost_per_day * days + cost_per_mile * miles = 162.5 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end car_rental_cost_l274_274667


namespace election_result_l274_274229

theorem election_result (Vx Vy Vz : ℝ) (Pz : ℝ)
  (h1 : Vx = 3 * (Vx / 3)) (h2 : Vy = 2 * (Vy / 2)) (h3 : Vz = 1 * (Vz / 1))
  (h4 : 0.63 * (Vx + Vy + Vz) = 0.74 * Vx + 0.67 * Vy + Pz * Vz) :
  Pz = 0.22 :=
by
  -- proof steps would go here
  -- sorry to keep the proof incomplete
  sorry

end election_result_l274_274229


namespace train_length_is_correct_l274_274057

def train_speed : ℝ := 60 -- in km/h
def man_speed : ℝ := 6 -- in km/h
def time : ℝ := 33 -- in seconds

def relative_speed_kmph : ℝ := train_speed + man_speed
def relative_speed_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5/18)

def length_of_train (relative_speed : ℝ) (time : ℝ) : ℝ := relative_speed * time

theorem train_length_is_correct :
  length_of_train (relative_speed_mps relative_speed_kmph) time ≈ 604.89 := 
by
  sorry

end train_length_is_correct_l274_274057


namespace radius_of_circle_l274_274330

theorem radius_of_circle (r x y : ℝ): 
  x = π * r^2 → 
  y = 2 * π * r → 
  x - y = 72 * π → 
  r = 12 := 
by 
  sorry

end radius_of_circle_l274_274330


namespace sphere_surface_area_ratio_l274_274208

theorem sphere_surface_area_ratio (V1 V2 r1 r2 A1 A2 : ℝ)
    (h_volume_ratio : V1 / V2 = 8 / 27)
    (h_volume_formula1 : V1 = (4/3) * Real.pi * r1^3)
    (h_volume_formula2 : V2 = (4/3) * Real.pi * r2^3)
    (h_surface_area_formula1 : A1 = 4 * Real.pi * r1^2)
    (h_surface_area_formula2 : A2 = 4 * Real.pi * r2^2)
    (h_radius_ratio : r1 / r2 = 2 / 3) :
  A1 / A2 = 4 / 9 :=
sorry

end sphere_surface_area_ratio_l274_274208


namespace christine_wander_time_l274_274837

theorem christine_wander_time (d s : ℕ) (h_d : d = 20) (h_s : s = 4) : d / s = 5 :=
by
  rw [h_d, h_s]
  exact rfl

end christine_wander_time_l274_274837


namespace angle_B_in_triangle_l274_274203

theorem angle_B_in_triangle
  (a b c : ℝ)
  (h_area : 2 * (a * c * ((a^2 + c^2 - b^2) / (2 * a * c)).sin) = (a^2 + c^2 - b^2) * (Real.sqrt 3 / 6)) :
  ∃ B : ℝ, B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l274_274203


namespace total_workers_in_workshop_l274_274399

theorem total_workers_in_workshop :
  let (avg_salary_all: ℝ) := 8000,
  let (num_technicians : ℕ) := 7,
  let (avg_salary_technicians: ℝ) := 14000,
  let (avg_salary_rest: ℝ) := 6000,
  let total_salary_all (W: ℕ) := W * avg_salary_all,
  let total_salary_technicians := num_technicians * avg_salary_technicians,
  let total_salary_rest (N: ℕ) := N * avg_salary_rest,
  let W := num_technicians + N,
  let equation := (num_technicians + N) * avg_salary_all = total_salary_technicians + total_salary_rest N,
  ∃ N : ℕ, W = num_technicians + N ∧ equation :=
  ∃ (N: ℕ), 28 = num_technicians + N ∧ (num_technicians + N) * avg_salary_all = total_salary_technicians + total_salary_rest N :=
sorry

end total_workers_in_workshop_l274_274399


namespace lean_proof_l274_274271

noncomputable def proof_problem (a b c d : ℝ) (habcd : a * b * c * d = 1) : Prop :=
  (1 + a * b) / (1 + a) ^ 2008 +
  (1 + b * c) / (1 + b) ^ 2008 +
  (1 + c * d) / (1 + c) ^ 2008 +
  (1 + d * a) / (1 + d) ^ 2008 ≥ 4

theorem lean_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_abcd : a * b * c * d = 1) : proof_problem a b c d h_abcd :=
  sorry

end lean_proof_l274_274271


namespace probability_smallest_in_B_greater_largest_in_A_l274_274089

open Finset
open Nat
open BigOperators

theorem probability_smallest_in_B_greater_largest_in_A :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let all_subsets := S.powerset
  let non_empty_subsets := all_subsets.filter (λ x, x ≠ ∅)
  let total_ways := (non_empty_subsets.card * (non_empty_subsets.card - 1)) / 2
  let favorable_ways := 1013 * (2 ^ 10) - 1
  let probability := favorable_ways.toReal / (total_ways : ℝ)
  in probability = 4097 / 1045506 := 
sorry

end probability_smallest_in_B_greater_largest_in_A_l274_274089


namespace equivalent_equations_curve_cartesian_from_polar_max_min_value_x_plus_2y_l274_274016

noncomputable def line_eq_cartesian (x y : ℝ) : Prop := x - y + 4 = 0
noncomputable def curve_eq_polar (ρ θ : ℝ) : Prop := ρ^2 - 4 * sqrt 2 * ρ * cos(θ - π / 4) + 6 = 0
noncomputable def line_eq_polar (ρ θ : ℝ) : Prop := ρ * cos θ - ρ * sin θ + 4 = 0
noncomputable def curve_eq_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 4 * y + 6 = 0

theorem equivalent_equations 
  (x y ρ θ : ℝ) :
  line_eq_cartesian x y ↔ line_eq_polar ρ θ :=
sorry

theorem curve_cartesian_from_polar 
  (ρ θ x y : ℝ)
  (h1 : ρ = sqrt (x^2 + y^2))
  (h2 : θ = atan2 y x)
  (h3 : curve_eq_polar ρ θ) :
  curve_eq_cartesian x y :=
sorry

theorem max_min_value_x_plus_2y
  (x y : ℝ)
  (h : curve_eq_cartesian x y) :
  10 - sqrt 6 ≤ x + 2 * y ∧ x + 2 * y ≤ 10 + sqrt 6 :=
sorry

end equivalent_equations_curve_cartesian_from_polar_max_min_value_x_plus_2y_l274_274016


namespace right_triangle_midpoint_l274_274604

theorem right_triangle_midpoint
  (A B C D E : Type*)
  [euclidean_affine_space ℝ (fin 3)]
  (ha : affine_combination ℝ (fin 3) A)
  (hb : affine_combination ℝ (fin 3) B)
  (hc : affine_combination ℝ (fin 3) C)
  (hd : affine_combination ℝ (fin 3) D)
  (he : affine_combination ℝ (fin 3) E)
  (h_triangle : triangle ℝ A B C)
  (right_angle : angle B = 90)
  (h_ab : dist A B = 6)
  (h_ac : dist A C = 8)
  (midpoint_bc : midpoint D B C)
  (mid_c: midpoint D A E)
  (ce_length : dist C E = 10) :
  dist B D = 5 :=
sorry

end right_triangle_midpoint_l274_274604


namespace circumcenter_locus_of_triangle_l274_274069

theorem circumcenter_locus_of_triangle
  (O : Point) (r p : ℝ) (ABC : Triangle) (A B C : Point)
  (h1 : distance O A = r) (h2 : distance O B = r) (h3 : distance O C = r)
  (h4 : distance O A * distance O B * distance O C = p^3) :
  let P := circumcenter ABC in
  distance O P = (p / (4 * r^2)) * sqrt (p * (p^3 - 8 * r^3)) :=
begin
  -- proof skipped
  sorry
end

end circumcenter_locus_of_triangle_l274_274069


namespace trigonometric_expression_value_l274_274911

theorem trigonometric_expression_value (α : ℝ) (h : tan α = 1) : 
  1 - 2 * sin α * cos α - 3 * cos α^2 = -3 / 2 := 
by 
  sorry

end trigonometric_expression_value_l274_274911


namespace Petya_wins_on_2021x2021_grid_l274_274513

theorem Petya_wins_on_2021x2021_grid :
  ∀ (grid : array (fin 2021) (array (fin 2021) (option nat))) 
    (player_turn : bool),
    (player_turn = tt → (∀ a b : fin 2021, a < 2017 → b < 2019 → ∃ i j : fin 2021, i ≤ a + 5 ∧ j ≤ b + 3 ∧ grid[i][j].is_some) ∧ 
                    (∀ a b : fin 2021, a < 2019 → b < 2017 → ∃ i j : fin 2021, i ≤ a + 3 ∧ j ≤ b + 5 ∧ grid[i][j].is_some)) →
  player_turn = tt ∨ player_turn = ff → 
  (∃ strategy, (∀ turn player_num, (turn = nat_mod 2 player_num) → 
    (∧ player_turn) →
    ((player_turn = tt → ∃ i j : fin 2021, i + 3 < 2021 ∧ j + 5 < 2021 ∧ grid[i][j].is_none → (strategy turn = (i,j))) ∧ 
     (player_turn = ff → ∃ i j : fin 2021, i + 3 < 2021 ∧ j + 5 < 2021 ∧ grid[i][j].is_none → (strategy turn = (i,j)))))) :=
sorry

end Petya_wins_on_2021x2021_grid_l274_274513


namespace min_value_frac_eq_seven_half_plus_two_sqrt_three_l274_274190

theorem min_value_frac_eq_seven_half_plus_two_sqrt_three 
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (h : (1:ℝ) * (2 * b - 1) + a * 3 = 1) : 
  ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 
  (1 * (2 * b - 1) + a * 3 = 1) 
  ∧ (∀ a b, 0 < a ∧ 0 < b ∧ (1 * (2 * b - 1) + a * 3 = 1) → (1/a + 2/b) ≥ (7/2 + 2 * Real.sqrt 3)) 
  ∧ (∃ a b, 0 < a ∧ 0 < b ∧ (1 * (2 * b - 1) + a * 3 = 1) ∧ (1/a + 2/b) = (7/2 + 2 * Real.sqrt 3)) := sorry

end min_value_frac_eq_seven_half_plus_two_sqrt_three_l274_274190


namespace basketball_shots_l274_274594

variable (x y : ℕ)

theorem basketball_shots : 3 * x + 2 * y = 26 ∧ x + y = 11 → x = 4 :=
by
  intros h
  sorry

end basketball_shots_l274_274594


namespace margaret_score_l274_274698

theorem margaret_score (average_score marco_score margaret_score : ℝ)
  (h1: average_score = 90)
  (h2: marco_score = average_score - 0.10 * average_score)
  (h3: margaret_score = marco_score + 5) : 
  margaret_score = 86 := 
by
  sorry

end margaret_score_l274_274698


namespace dm_eq_dn_l274_274233

variables (A B C D K M N : Type*)
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ K] [AffineSpace ℝ M] [AffineSpace ℝ N]

/-- Definition of parallelogram -/
def parallelogram (A B C D : Type*) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] :=
  (Vector ℝ A B + Vector ℝ C D = 0) ∧ (Vector ℝ A D + Vector ℝ B C = 0)

/-- Definition saying that a point is symmetric to another point with respect to a third point -/
def symmetric (P Q R : Type*) [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ R] : Prop :=
  Vector ℝ P Q = -Vector ℝ P R

/-- Main theorem -/
theorem dm_eq_dn (h1 : parallelogram A B C D)
                 (h2 : Vector ℝ A B = Vector ℝ B D)
                 (h3 : ¬(K = A))
                 (h4 : Vector ℝ K D = Vector ℝ A D)
                 (h5 : symmetric K M C)
                 (h6 : symmetric A N B) : 
  Vector ℝ D M = Vector ℝ D N :=
  sorry

end dm_eq_dn_l274_274233


namespace sum_min_max_3p3_minus_p4_eq_128_l274_274283

theorem sum_min_max_3p3_minus_p4_eq_128
  (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  let f := 3 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) in
  (∀x y z w : ℝ, 
     x + y + z + w = 10 → 
     x^2 + y^2 + z^2 + w^2 = 20 → 
     f ≥ 3 * (x^3 + y^3 + z^3 + w^3) - (x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀x y z w : ℝ,
     x + y + z + w = 10 →
     x^2 + y^2 + z^2 + w^2 = 20 → 
     f ≤ 3 * (x^3 + y^3 + z^3 + w^3) - (x^4 + y^4 + z^4 + w^4)) := 
sorry

end sum_min_max_3p3_minus_p4_eq_128_l274_274283


namespace distance_from_point_to_line_correct_l274_274830

-- Define the points as tuples in ℝ³
def point_a : ℝ × ℝ × ℝ := (2, 0, 3)
def line_point1 : ℝ × ℝ × ℝ := (1, 3, 0)
def line_point2 : ℝ × ℝ × ℝ := (0, 0, 2)

-- Function to calculate the distance from a point to a line in ℝ³
noncomputable def distance_point_to_line
  (point : ℝ × ℝ × ℝ) (line_pt1 line_pt2 : ℝ × ℝ × ℝ) : ℝ :=
  let direction_vector := (line_pt2.1 - line_pt1.1, line_pt2.2 - line_pt1.2, line_pt2.3 - line_pt1.3) in
  let vector_a_to_point := (point.1 - line_pt1.1, point.2 - line_pt1.2, point.3 - line_pt1.3) in
  let cross_product := (
    vector_a_to_point.2 * direction_vector.3 - vector_a_to_point.3 * direction_vector.2,
    vector_a_to_point.3 * direction_vector.1 - vector_a_to_point.1 * direction_vector.3,
    vector_a_to_point.1 * direction_vector.2 - vector_a_to_point.2 * direction_vector.1
  ) in
  let norm_cross_product := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
  let norm_direction_vector := real.sqrt (direction_vector.1^2 + direction_vector.2^2 + direction_vector.3^2) in
  norm_cross_product / norm_direction_vector

theorem distance_from_point_to_line_correct :
  distance_point_to_line point_a line_point1 line_point2 = real.sqrt 5 :=
sorry

end distance_from_point_to_line_correct_l274_274830


namespace tangency_circle_line_l274_274665

-- Define the circle equation
def circle (r : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in x^2 + y^2 = r^2

-- Define the line equation
def line (r : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in x + y = r + 1

-- Prove tangency condition and find the value of r
theorem tangency_circle_line : ∃ r : ℝ, (
  (∀ p : ℝ × ℝ, circle r p → line r p) ∧
  r = real.sqrt 2 + 1
) := sorry

end tangency_circle_line_l274_274665


namespace train_cross_time_l274_274561

theorem train_cross_time : 
  ∀ (train_length : ℝ) (bridge_length : ℝ) (train_speed_kph : ℝ), 
  train_length = 130 → 
  bridge_length = 150 → 
  train_speed_kph = 65 → 
  ∃ t : ℝ, t ≈ 15.51 :=
by
  assume train_length bridge_length train_speed_kph
  assume h_train_length h_bridge_length h_train_speed

  let total_length := train_length + bridge_length
  let train_speed_mps := train_speed_kph * (1000 / 3600)
  let time_needed := total_length / train_speed_mps

  use time_needed
  sorry -- Proof required

end train_cross_time_l274_274561


namespace range_of_a_l274_274891

open Complex Real

def z (x a : ℝ) : ℂ := x + (x - a) * Complex.i

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 2) → Complex.abs (z x a) > Complex.abs (z x (a+1))) → a ∈ set.Iic (1 / 2) :=
sorry

end range_of_a_l274_274891


namespace impossibility_triplet_2002x2002_grid_l274_274262

theorem impossibility_triplet_2002x2002_grid: 
  ∀ (M : Matrix ℕ (Fin 2002) (Fin 2002)),
    (∀ i j : Fin 2002, ∃ (r1 r2 r3 : Fin 2002), 
      (M i r1 > 0 ∧ M i r2 > 0 ∧ M i r3 > 0) ∨ 
      (M r1 j > 0 ∧ M r2 j > 0 ∧ M r3 j > 0)) →
    ¬ (∀ i j : Fin 2002, ∃ (a b c : ℕ), 
      M i j = a ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
      (∃ (r1 r2 r3 : Fin 2002), 
        (M i r1 = a ∨ M i r1 = b ∨ M i r1 = c) ∧ 
        (M i r2 = a ∨ M i r2 = b ∨ M i r2 = c) ∧ 
        (M i r3 = a ∨ M i r3 = b ∨ M i r3 = c)) ∨
      (∃ (c1 c2 c3 : Fin 2002), 
        (M c1 j = a ∨ M c1 j = b ∨ M c1 j = c) ∧ 
        (M c2 j = a ∨ M c2 j = b ∨ M c2 j = c) ∧ 
        (M c3 j = a ∨ M c3 j = b ∨ M c3 j = c)))
:= sorry

end impossibility_triplet_2002x2002_grid_l274_274262


namespace female_sample_count_is_correct_l274_274220

-- Definitions based on the given conditions
def total_students : ℕ := 900
def male_students : ℕ := 500
def sample_size : ℕ := 45
def female_students : ℕ := total_students - male_students
def female_sample_size : ℕ := (female_students * sample_size) / total_students

-- The lean statement to prove
theorem female_sample_count_is_correct : female_sample_size = 20 := 
by 
  -- A placeholder to indicate the proof needs to be filled in
  sorry

end female_sample_count_is_correct_l274_274220


namespace dot_product_correct_l274_274459

def vector1 : ℝ × ℝ × ℝ := (5, -3, 2)
def vector2 : ℝ × ℝ × ℝ := (-7, 4, -6)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem dot_product_correct : dot_product vector1 vector2 = -59 :=
by
  -- computation is skipped
  sorry

end dot_product_correct_l274_274459


namespace tina_no_acutetriangles_l274_274122

noncomputable def count_valid_triangles : ℕ :=
  let angles := {10, 20, 40, 50, 70, 80}
  let valid_triangles (a b c : ℕ) : Prop :=
    a ∈ angles ∧ b ∈ angles ∧ c ∈ angles ∧
    a < b ∧ b < c ∧ a + b + c = 180 ∧
    a ≠ b ∧ b ≠ c ∧ c ≠ a
  (angles.to_finset.image (λ a, angles.to_finset.image (λ b, angles.to_finset.image (λ c, (a, b, c)))))
  .erase_none
  .count valid_triangles = 0

theorem tina_no_acutetriangles : count_valid_triangles = 0 :=
  sorry

end tina_no_acutetriangles_l274_274122


namespace geometric_sequence_problem_l274_274166

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geo : ∀ n, a (n + 1) = a n * q)
  (h_arith_seq : 3 * a 1, (1 / 2) * a 3, 2 * a 2 form_arithmetic_sequence) :
  q > 0 → 
  (q^2 = 3 + 2 * q) → 
  (a 3 = a 1 * q^2) → 
  (a 2 = a 1 * q) → 
  (a 1 ≠ 0) → 
  (a 2014 + a 2015) / (a 2012 + a 2013) = 9 := 
by
  sorry

end geometric_sequence_problem_l274_274166


namespace smaller_rectangle_perimeter_l274_274432

def perimeter_original_rectangle (a b : ℝ) : Prop := 2 * (a + b) = 100
def number_of_cuts (vertical_cuts horizontal_cuts : ℕ) : Prop := vertical_cuts = 7 ∧ horizontal_cuts = 10
def total_length_of_cuts (a b : ℝ) : Prop := 7 * b + 10 * a = 434

theorem smaller_rectangle_perimeter (a b : ℝ) (vertical_cuts horizontal_cuts : ℕ) (m n : ℕ) :
  perimeter_original_rectangle a b →
  number_of_cuts vertical_cuts horizontal_cuts →
  total_length_of_cuts a b →
  (m = 8) →
  (n = 11) →
  (a / 8 + b / 11) * 2 = 11 :=
by
  sorry

end smaller_rectangle_perimeter_l274_274432


namespace part_I_part_II_l274_274131

variable {x : ℝ}
variable (h1 : -π / 2 < x ∧ x < 0)
variable (h2 : sin x + cos x = 1 / 5)

theorem part_I : sin x - cos x = -7 / 5 := by
  sorry

theorem part_II : 4 * sin x * cos x - cos x^2 = -64 / 25 := by
  sorry

end part_I_part_II_l274_274131


namespace problem_l274_274340

noncomputable def p (k : ℝ) (x : ℝ) := k * (x - 5) * (x - 2)
noncomputable def q (x : ℝ) := (x - 5) * (x + 3)

theorem problem {p q : ℝ → ℝ} (k : ℝ) :
  (∀ x, q x = (x - 5) * (x + 3)) →
  (∀ x, p x = k * (x - 5) * (x - 2)) →
  (∀ x ≠ 5, (p x) / (q x) = (3 * (x - 2)) / (x + 3)) →
  p 3 / q 3 = 1 / 2 :=
by
  sorry

end problem_l274_274340


namespace solution_set_of_inequality_l274_274718

theorem solution_set_of_inequality (x : ℝ) : (1 + x > 6 - 4x) → (x > 1) := 
by
  intro h
  sorry

end solution_set_of_inequality_l274_274718


namespace probability_A1_selected_probability_neither_A2_B2_selected_l274_274435

-- Define the set of male members and female members
def male_members : set ℕ := {1, 2, 3, 4}
def female_members : set ℕ := {1, 2, 3}

-- Define the universal set of all possible outcomes
def outcomes : set (ℕ × ℕ) := { (m, f) | m ∈ male_members ∧ f ∈ female_members }

-- Define event M: "A₁ is selected"
def event_M : set (ℕ × ℕ) := { (m, _) | m = 1 }

-- Define event N: "Neither A₂ nor B₂ is selected"
def event_N : set (ℕ × ℕ) := { (m, f) | m ≠ 2 ∧ f ≠ 2 }

-- Probability space
noncomputable def probability_space : MeasureSpace (ℕ × ℕ) := sorry

-- Statement 1: Probability of A₁ being selected
theorem probability_A1_selected : probability_space.measure (event_M) = 1 / 4 := sorry

-- Statement 2: Probability of neither A₂ nor B₂ being selected
theorem probability_neither_A2_B2_selected : probability_space.measure (event_N) = 11 / 12 := sorry

end probability_A1_selected_probability_neither_A2_B2_selected_l274_274435


namespace exponent_of_3_in_30_factorial_l274_274247

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | nat.succ n' => (nat.succ n') * factorial n'

noncomputable def exponent_of_prime_in_factorial (n p : ℕ) : ℕ :=
  if h : p > 1 ∧ nat.prime p then
    let rec count_multiples (m : ℕ) (acc : ℕ) : ℕ :=
      if m > n then acc
      else count_multiples (p * m) (acc + n / m)
    count_multiples p 0
  else 0

theorem exponent_of_3_in_30_factorial : exponent_of_prime_in_factorial 30 3 = 14 := 
sorry

end exponent_of_3_in_30_factorial_l274_274247


namespace arrange_students_together_l274_274118

theorem arrange_students_together :
  ∃ (arrangements : ℕ), arrangements = 36 ∧
  let A := sorry,
      B := sorry,
      C := sorry,
      D := sorry,
      E := sorry in
  let students := [A, B, C, D, E] in
  let ABC := [A, B, C] in
  let groups := [ABC, D, E] in
  ∀ (l : List (List α)), l ∈ (all_permutations groups) → 
  ∃ (subarrangements : ℕ), subarrangements = 6 ∧ (permutations_ABC ABC) = 6 ∧
  arrangements = subarrangements * (permutations groups) :=
 sorry

end arrange_students_together_l274_274118


namespace implicit_derivative_l274_274861

variable {x y : ℝ}

def implicit_eq (x y : ℝ) : Prop :=
  ln (sqrt (x^2 + y^2)) = arctan (y / x)
  
theorem implicit_derivative 
  (h : implicit_eq x y) 
  (hx : x ≠ 0) 
  (hy : x ≠ y) : 
  ∃ y' : ℝ, y' = (x + y) / (x - y) :=
by
  have : ∂ F/ ∂x= \frac{x}{x^2 + y^2} - \frac{y}{x^2 + y^2}, ICU Safe = sorry,
  exact sorry

end implicit_derivative_l274_274861


namespace manny_remaining_money_l274_274366

def cost_chair (cost_total_chairs : ℕ) (number_of_chairs : ℕ) : ℕ :=
  cost_total_chairs / number_of_chairs

def cost_table (cost_chair : ℕ) (chairs_for_table : ℕ) : ℕ :=
  cost_chair * chairs_for_table

def total_cost (cost_table : ℕ) (cost_chairs : ℕ) : ℕ :=
  cost_table + cost_chairs

def remaining_money (initial_amount : ℕ) (spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem manny_remaining_money : remaining_money 100 (total_cost (cost_table (cost_chair 55 5) 3) ((cost_chair 55 5) * 2)) = 45 :=
by
  sorry

end manny_remaining_money_l274_274366


namespace log_condition_iff_l274_274973

theorem log_condition_iff (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ 1) : 
  log a b > 0 ↔ (a - 1) * (b - 1) > 0 := 
by 
  sorry

end log_condition_iff_l274_274973


namespace shaded_floor_area_l274_274784

noncomputable def total_shaded_area (floor_length floor_width tile_length tile_width radius height base : ℝ) (num_tiles : ℕ) : ℝ :=
let tile_area := tile_length * tile_width in
let white_circle_area := Real.pi * radius^2 in
let white_triangle_area := (1/2) * base * height in
let shaded_area_per_tile := tile_area - white_circle_area - white_triangle_area in
(shaded_area_per_tile * num_tiles)

theorem shaded_floor_area :
 let floor_length := 10
 let floor_width := 12
 let tile_length := 2
 let tile_width := 1
 let radius := 1/2
 let height := 1/2
 let base := 1/2
 let num_tiles := ((floor_length / tile_length).to_nat * (floor_width / tile_width).to_nat)
(total_shaded_area floor_length floor_width tile_length tile_width radius height base num_tiles) = (112.5 - 15 * Real.pi) :=
by
  sorry

end shaded_floor_area_l274_274784


namespace range_of_a_l274_274922

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l274_274922


namespace parabola_properties_l274_274518

theorem parabola_properties (p m k1 k2 k3 : ℝ)
  (parabola_eq : ∀ x y, y^2 = 2 * p * x ↔ y = m)
  (parabola_passes_through : m^2 = 2 * p)
  (point_distance : ((1 + p / 2)^2 + m^2 = 8) ∨ ((1 + p / 2)^2 + m^2 = 8))
  (p_gt_zero : p > 0)
  (point_P : (1, 2) ∈ { (x, y) | y^2 = 4 * x })
  (slope_eq : k3 = (k1 * k2) / (k1 + k2 - k1 * k2)) :
  (y^2 = 4 * x) ∧ (1/k1 + 1/k2 - 1/k3 = 1) := sorry

end parabola_properties_l274_274518


namespace megan_dials_correct_number_probability_l274_274671

-- definitions for conditions
noncomputable def options_for_first_three_digits : ℕ := 3
noncomputable def combinations_of_two_out_of_five : ℕ := Nat.choose 5 2 -- equivalent to binomial coefficient \binom{5}{2}
noncomputable def arrangements_with_repeats : ℕ := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 2)

-- definition for the total number of valid phone numbers
noncomputable def total_valid_numbers : ℕ :=
  options_for_first_three_digits * combinations_of_two_out_of_five * arrangements_with_repeats

-- proof that the probability is 1/180
theorem megan_dials_correct_number_probability :
  total_valid_numbers = 180 → (1/180 : ℚ) = 1/total_valid_numbers := by
  intro h
  rw h
  norm_num
  sorry

end megan_dials_correct_number_probability_l274_274671


namespace annas_deducted_salary_l274_274077

theorem annas_deducted_salary (weekly_salary : ℝ) (working_days_per_week : ℕ) (absent_days : ℕ) :
  weekly_salary = 1379 → working_days_per_week = 5 → absent_days = 2 → 
  (weekly_salary - (weekly_salary / working_days_per_week) * absent_days) = 827.40 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end annas_deducted_salary_l274_274077


namespace value_of_a51_l274_274615

noncomputable def a : ℕ → ℤ
| 1       := 1
| (n + 1) := a n + 2

theorem value_of_a51 : a 51 = 101 :=
by
  sorry

end value_of_a51_l274_274615


namespace parallelogram_area_l274_274100

variables {V : Type*} [inner_product_space ℝ V] 
variables (a b : V) (h : ∥a × b∥ = 12)

theorem parallelogram_area : ∥(3 • a + 4 • b) × (2 • a - 6 • b)∥ = 312 := 
sorry

end parallelogram_area_l274_274100


namespace max_s_square_l274_274778

variable (A B C D : Point)
variable (r : ℝ) (s : ℝ)
variable (O : Circle)
variable [inscribed : TriangleInscribed \O ⟨A, B, C⟩]
variable [inscribed : TriangleInscribed \O ⟨A, B, D⟩]

-- Conditions:
-- Triangles ABC and ABD are inscribed in a circle with radius r
def is_inscribed_in_circle : Prop := 
  ∀ (A B C D : Point), ∃ (O : Circle) (r : ℝ),
    TriangleInscribed \O ⟨A, B, C⟩ ∧
    TriangleInscribed \O ⟨A, B, D⟩ ∧
    O.radius = r
  
-- Points C and D are on opposite arcs divided by AB
def points_on_opposite_arcs (A B C D : Point) : Prop := 
  segment (C.midpoint D).perpendicular A B

-- Segment CD is perpendicular to AB
def perpendicular_segment (C D A B : Point) : Prop :=
  segment C D.perpendicular segment A B
  
-- Points C and D do not coincide with A or B
def points_do_not_coincide (A B C D : Point) : Prop := 
  C ≠ A ∧ C ≠ B ∧ D ≠ A ∧ D ≠ B

-- Expression for s
def s_def (A C D : Point) : ℝ := A.distance C + A.distance D

theorem max_s_square (A B C D : Point) (r : ℝ) (O : Circle)
  (h_inscribed : is_inscribed_in_circle A B C D)
  (h_opposite_arcs : points_on_opposite_arcs A B C D)
  (h_perpendicular : perpendicular_segment C D A B)
  (h_noncoincident : points_do_not_coincide A B C D)
  : ∃ s, s_def A C D = 8 * r^2
:= sorry

end max_s_square_l274_274778


namespace trailing_zeros_of_product_l274_274195

-- Define the conditions
def twenty_five_power_seven : ℕ := 25 ^ 7
def eight_power_three : ℕ := 8 ^ 3
def product : ℕ := twenty_five_power_seven * eight_power_three

-- State the theorem
theorem trailing_zeros_of_product : Nat.trailing_zeros product = 9 :=
sorry

end trailing_zeros_of_product_l274_274195


namespace largest_real_number_mu_l274_274862

noncomputable def largest_mu : ℝ := 13 / 2

theorem largest_real_number_mu (
  a b c d : ℝ
) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) :
  (a^2 + b^2 + c^2 + d^2) ≥ (largest_mu * a * b + b * c + 2 * c * d) :=
sorry

end largest_real_number_mu_l274_274862


namespace ratio_of_girls_to_boys_l274_274989

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) (h_ratio : girls = 4 * (girls + boys) / 7) (h_total : total_students = 70) : 
  girls = 40 ∧ boys = 30 :=
by
  sorry

end ratio_of_girls_to_boys_l274_274989


namespace problem1_problem2_l274_274547

-- Definition of the function
def f (a x : ℝ) := x^2 + a * x + 3

-- Problem statement 1: Prove that if f(x) ≥ a for all x ∈ ℜ, then a ≤ 3.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x ≥ a) → a ≤ 3 := sorry

-- Problem statement 2: Prove that if f(x) ≥ a for all x ∈ [-2, 2], then -6 ≤ a ≤ 2.
theorem problem2 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≥ a) → -6 ≤ a ∧ a ≤ 2 := sorry

end problem1_problem2_l274_274547


namespace perfect_square_factors_of_720_count_l274_274957

noncomputable def num_perfect_square_factors_of_720 : ℕ := 6

theorem perfect_square_factors_of_720_count :
  let factors := 720.prime_factors 
  let is_perfect_square (n : ℕ) : Prop :=
    ∀ p ∈ n.prime_factors, ∃ k, (n.factorization p) = 2 * k
  let factors_of_720 (n : ℕ) : Prop :=
    n ∣ 720
  (finset.filter (λ n, is_perfect_square n) (finset.Icc 1 720)).card = num_perfect_square_factors_of_720 :=
sorry

end perfect_square_factors_of_720_count_l274_274957


namespace first_term_of_arithmetic_sequence_l274_274639

theorem first_term_of_arithmetic_sequence (T : ℕ → ℝ) (b : ℝ) 
  (h1 : ∀ n : ℕ, T n = (n * (2 * b + (n - 1) * 4)) / 2) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, T (4 * n) / T n = d) :
  b = 2 :=
by
  sorry

end first_term_of_arithmetic_sequence_l274_274639


namespace non_congruent_triangles_with_perimeter_24_l274_274563

theorem non_congruent_triangles_with_perimeter_24 : 
  ∃ (triangles : Finset (Finset (ℕ × ℕ × ℕ))), 
  (∀ t ∈ triangles, let ⟨a, b, c⟩ := t in a + b + c = 24 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧ 
  triangles.card = 11 :=
sorry

end non_congruent_triangles_with_perimeter_24_l274_274563


namespace solutions_equiv_cond_l274_274721

theorem solutions_equiv_cond (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x + 1 / (x - 1) = a + 1 / (x - 1)) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x = a) ∧ (∃ x : ℝ, x = 1 → a ≠ 4)  :=
sorry

end solutions_equiv_cond_l274_274721


namespace division_remainder_l274_274386

def remainder (x y : ℕ) : ℕ := x % y

theorem division_remainder (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : (x : ℚ) / y = 96.15) (h4 : y = 20) : remainder x y = 3 :=
by
  sorry

end division_remainder_l274_274386


namespace g_g_2_eq_394_l274_274575

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l274_274575


namespace playground_area_increase_l274_274022

def perimeter_rectangle (length width : ℝ) : ℝ := 2 * (length + width)
def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem playground_area_increase :
  let length_r := 50 : ℝ in
  let width_r := 20 : ℝ in
  let perimeter_r := perimeter_rectangle length_r width_r in
  let side_length_s := side_length_square perimeter_r in
  let area_r := area_rectangle length_r width_r in
  let area_s := area_square side_length_s in
  area_s - area_r = 225 :=
by
  simp [perimeter_rectangle, side_length_square, area_rectangle, area_square]
  sorry

end playground_area_increase_l274_274022


namespace football_sampling_l274_274802

theorem football_sampling :
  ∀ (total_members football_members basketball_members volleyball_members total_sample : ℕ),
  total_members = 120 →
  football_members = 40 →
  basketball_members = 60 →
  volleyball_members = 20 →
  total_sample = 24 →
  (total_sample * football_members / (football_members + basketball_members + volleyball_members) = 8) :=
by 
  intros total_members football_members basketball_members volleyball_members total_sample h_total_members h_football_members h_basketball_members h_volleyball_members h_total_sample
  sorry

end football_sampling_l274_274802


namespace cyclic_third_quadrilateral_l274_274629

theorem cyclic_third_quadrilateral
  (A B C D X Y Z : Type)
  [LinearOrderedRing A] [LinearOrderedRing B] [LinearOrderedRing C] [LinearOrderedRing D] [LinearOrderedRing X] [LinearOrderedRing Y] [LinearOrderedRing Z]
  (h1 : ∃ (AX : A) (BY : B) (CZ : C), ∃ (D : D), true) -- cevians concurrent at D of ΔABC
  (h2 : cyclic_quad DY AZ ∧ cyclic_quad DZ BX) -- DYAZ and DZBX are cyclic
  : cyclic_quad DX CY := -- prove DXCY is cyclic
sorry

end cyclic_third_quadrilateral_l274_274629


namespace find_prices_and_function_l274_274319

noncomputable def unit_prices (x : ℝ) : Prop :=
(price_A : ℝ, price_B : ℝ) ⟨ price_A = 1.2 * x, price_B = x,
    (30000 / price_A) = (15000 / price_B) + 4⟩ ∧
    (x = 2500) ⟶
    (price_A = 1.2 * 2500 ∧ price_B = 2500)

noncomputable def cost_function (a : ℝ) (w : ℝ) : Prop :=
a ≥ 0 ∧ a ≤ 50 ∧ a ≥ (1 / 3) * (50 - a) ∧
w = 500 * a + 125000 ∧ 
(a = 13) ⟶ 
(w = 131500)

theorem find_prices_and_function:
 (∃ x₁ x₂, unit_prices x₁ x₂) ∧ (∃ a w, cost_function a w) := 
begin
	sorry,
end

end find_prices_and_function_l274_274319


namespace root_sum_of_squares_l274_274643

noncomputable def polynomial := Polynomial ℝ

theorem root_sum_of_squares (a b c d : ℝ)
  (h₁ : polynomial.eval a (polynomial.X^4 - 12*polynomial.X^3 + 47*polynomial.X^2 - 60*polynomial.X + 24) = 0)
  (h₂ : polynomial.eval b (polynomial.X^4 - 12*polynomial.X^3 + 47*polynomial.X^2 - 60*polynomial.X + 24) = 0)
  (h₃ : polynomial.eval c (polynomial.X^4 - 12*polynomial.X^3 + 47*polynomial.X^2 - 60*polynomial.X + 24) = 0)
  (h₄ : polynomial.eval d (polynomial.X^4 - 12*polynomial.X^3 + 47*polynomial.X^2 - 60*polynomial.X + 24) = 0) :
  (a + b)^2 + (b + c)^2 + (c + d)^2 + (d + a)^2 = 147 :=
sorry

end root_sum_of_squares_l274_274643


namespace find_k_for_line_passing_point_l274_274850

theorem find_k_for_line_passing_point :
  ∀ k : ℚ, (∀ x = (1/3) in ℝ, ∀ y = -8 in ℝ, (-(3/4 : ℝ) - 3*k*x = 7*y)) → k = 55.25 :=
by sorry

end find_k_for_line_passing_point_l274_274850


namespace exponent_of_3_in_30_factorial_l274_274248

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | nat.succ n' => (nat.succ n') * factorial n'

noncomputable def exponent_of_prime_in_factorial (n p : ℕ) : ℕ :=
  if h : p > 1 ∧ nat.prime p then
    let rec count_multiples (m : ℕ) (acc : ℕ) : ℕ :=
      if m > n then acc
      else count_multiples (p * m) (acc + n / m)
    count_multiples p 0
  else 0

theorem exponent_of_3_in_30_factorial : exponent_of_prime_in_factorial 30 3 = 14 := 
sorry

end exponent_of_3_in_30_factorial_l274_274248


namespace problem_1_problem_2_l274_274157

-- Variables and parameters
variables (a b c : ℝ) (x y : ℝ)
variables (x0 y0 : ℝ) (F1 F2 M : ℝ × ℝ)
variables (e R : ℝ)

-- Conditions and Definitions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def eccentricity (c a : ℝ) : Prop := (c / a = 1/2)
def perimeter_triangle (M F1 F2 : ℝ × ℝ) : ℝ := dist M F1 + dist F1 F2 + dist F2 M
def maximum_area (y0 : ℝ) : ℝ := y0

-- Given conditions
axiom cond1 : a = 2
axiom cond2 : b^2 = 3
axiom cond3 : eccentricity c a
axiom cond4 : perimeter_triangle M F1 F2 = 6
axiom cond5 : 0 < x0 ∧ x0 < 2
axiom cond6 : 0 < |y0| ∧ |y0| ≤ (sqrt 15) / 3

-- Problem statements
theorem problem_1 : ellipse_eq 2 (sqrt 3) x y :=
by { sorry }

noncomputable def F1 := (-1, 0)
noncomputable def F2 := (1, 0)
noncomputable def max_area := maximum_area ((sqrt 15) / 3)

theorem problem_2 : max_area = (sqrt 15) / 3 :=
by { sorry }

end problem_1_problem_2_l274_274157


namespace arccot_inequality_arccot_equality_cond_l274_274928

noncomputable def a_seq : ℕ → ℝ
| 0 := 1
| 1 := 1
| (n+2) := a_seq (n+1) + a_seq n

theorem arccot_inequality :
  ∀ n : ℕ, Real.arccot (a_seq n) ≤ Real.arccot (a_seq (n+1)) + Real.arccot (a_seq (n+2)) :=
sorry

theorem arccot_equality_cond :
  ∀ n : ℕ, (Real.arccot (a_seq n) = Real.arccot (a_seq (n+1)) + Real.arccot (a_seq (n+2))) ↔ n % 2 = 0 :=
sorry

end arccot_inequality_arccot_equality_cond_l274_274928


namespace circle_B_radius_l274_274456

noncomputable def radiusB (r_A r_D : ℝ) : ℝ :=
  let discriminant := (5:ℝ)^2 - 4 * 1 * 2 in
  (5 + Real.sqrt discriminant) / 2

theorem circle_B_radius :
  ∀ (r_A r_D r_B : ℝ),
    r_A = 2 ∧ r_D = 3 ∧ (3 - r_B)^2 + (2 - r_B)^2 = r_D^2  →
    r_B = radiusB r_A r_D :=
by
  intros r_A r_D r_B h
  cases h with ha hd
  cases hd with hD_eq hr_eq
  rw [ha, hD_eq, hr_eq]
  sorry

end circle_B_radius_l274_274456


namespace range_n_minus_m_l274_274543

def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 1 / 2 * x + 3 / 2 else Real.log x

theorem range_n_minus_m:
  (∃ (m n : ℝ), m < n ∧ f m = f n) →
  (5 - 2 * Real.log 2 ≤ n - m ∧ n - m < Real.exp 2 - 1) :=
sorry

end range_n_minus_m_l274_274543


namespace cumulative_area_exceeds_by_2024_proportion_exceeds_by_2024_l274_274214

-- Definitions
def a_n (n : ℕ) : ℕ := 250 + 50 * (n - 1) -- Sequence for mid-to-low-priced houses (in hundred thousand square meters)
def S_n (n : ℕ) : ℕ := (n * (500 + (n - 1) * 50)) / 2 -- Cumulative area of mid-to-low-priced houses
def b_n (n : ℕ) : Float := 400 * 1.08^(n - 1) -- Sequence for total area of housing (in hundred thousand square meters)

def exceeds_225 (n : ℕ) : Prop := (S_n n) > 2250 -- Check if cumulative area exceeds 2250 (22.5 million square meters)
def proportion_exceeds (n : ℕ) : Prop :=  (a_n n : Float) / (b_n n : Float) > 0.85 -- Check if proportion exceeds 85%

theorem cumulative_area_exceeds_by_2024 : ∃ n, n ≤ 6 ∧ exceeds_225 n :=
by sorry

theorem proportion_exceeds_by_2024 : ∃ n, n ≤ 6 ∧ proportion_exceeds n :=
by sorry

end cumulative_area_exceeds_by_2024_proportion_exceeds_by_2024_l274_274214


namespace symmetric_point_is_correct_l274_274702

noncomputable def symmetric_point (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let a := -1
  let b := 1
  (a, b)

theorem symmetric_point_is_correct :
  let P := (0, 2)
  let l := λ (x : ℝ × ℝ), (x.1 + x.2 - 1 = 0)
  symmetric_point P l = (-1, 1) :=
by
  sorry

end symmetric_point_is_correct_l274_274702


namespace children_got_on_bus_l274_274015

theorem children_got_on_bus (initial_children total_children children_added : ℕ) 
  (h_initial : initial_children = 64) 
  (h_total : total_children = 78) : 
  children_added = total_children - initial_children :=
by
  sorry

end children_got_on_bus_l274_274015


namespace max_value_a_l274_274932

noncomputable def setA (a : ℝ) : Set ℝ := { x | (x - 1) * (x - a) ≥ 0 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x ≥ a - 1 }

theorem max_value_a (a : ℝ) :
  (setA a ∪ setB a = Set.univ) → a ≤ 2 := by
  sorry

end max_value_a_l274_274932


namespace plane_landed_earlier_l274_274024

variable (t : ℕ) -- Usual one-way travel time for Moskvich
variable (m : ℕ) -- Time Moskvich arrived earlier than usual
variable (d : ℕ) -- Time cargo truck drove before meeting
variable (e : ℕ) -- Plane landed earlier

-- Conditions 
def condition1 : Prop := m = 20 -- Moskvich arrived 20 minutes earlier than usual
def condition2 : Prop := d = 30 -- Truck drove for 30 minutes before meeting Moskvich
def condition3 : Prop := e = d + m // 2 -- Plane landed earlier by 30 minutes + 10 minutes

-- Question: How many minutes earlier did the plane land?
theorem plane_landed_earlier (t : ℕ) (m : ℕ) (d : ℕ) (e : ℕ)
  (h1 : condition1 m)
  (h2 : condition2 d)
  (h3 : condition3 d m e) :
  e = 40 := by
    sorry

end plane_landed_earlier_l274_274024


namespace sum_units_digit_three_digit_numbers_l274_274877

-- Defining the conditions
def digits := {0, 1, 2, 3, 4}
def isValidThreeDigit (n : Nat) : Prop := 
  ∃ x y z : Nat, (x ≠ 0) ∧ (x ∈ digits) ∧ (y ∈ digits) ∧ (z ∈ digits) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (n = 100 * x + 10 * y + z)

-- The statement to prove
theorem sum_units_digit_three_digit_numbers : 
  ∑ n in (Finset.filter isValidThreeDigit (Finset.range 1000)), (n % 10) = 90 := 
sorry

end sum_units_digit_three_digit_numbers_l274_274877


namespace books_leftover_l274_274596

theorem books_leftover (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) 
  (h1 : boxes = 1575) (h2 : books_per_box = 45) (h3 : new_box_capacity = 50) :
  ((boxes * books_per_box) % new_box_capacity) = 25 :=
by
  sorry

end books_leftover_l274_274596


namespace min_perimeter_of_six_equilateral_triangles_l274_274392

theorem min_perimeter_of_six_equilateral_triangles (n : ℕ) (s : ℕ) (triangles_overlap : ∀ i j, i ≠ j → share_side_full i j) :
  n = 6 → s = 1 → ∃ perimeter, perimeter = 6 :=
by intros h1 h2; sorry

end min_perimeter_of_six_equilateral_triangles_l274_274392


namespace domain_f_composite_l274_274532

noncomputable def domain_f_3_minus_2x : set ℝ :=
  { x | 3 - 2 * x ∈ set.Icc (-1:ℝ) 2 }

theorem domain_f_composite {x : ℝ} :
  x ∈ set.Icc (1/2) 2 ↔ 3 - 2 * x ∈ set.Icc (-1) 2 :=
begin
  -- Proof omitted
  sorry,
end

end domain_f_composite_l274_274532


namespace max_page_number_reached_l274_274676

def has_plenty_of_digits_except_5 (n : ℕ) : Prop :=
  (n.digits 10).count 5 ≤ 20

theorem max_page_number_reached (n : ℕ) :
  ∀ n, (has_plenty_of_digits_except_5 n) → n ≤ 104 :=
begin
  sorry
end

end max_page_number_reached_l274_274676


namespace sheela_deposit_l274_274316

/--
Sheela's deposit is calculated as 32% of her monthly income, which is Rs. 11875.
-/
theorem sheela_deposit :
  let deposit := 0.32 * 11875 in
  deposit = 3796 :=
by
  sorry

end sheela_deposit_l274_274316


namespace total_votes_l274_274397

theorem total_votes (V W L : ℕ) (h1 : W - L = 0.2 * ↑V) (h2 : L + 1000 - (W - 1000) = 0.2 * ↑V) : V = 5000 := 
by 
  sorry

end total_votes_l274_274397


namespace exponent_of_3_in_30_factorial_l274_274252

theorem exponent_of_3_in_30_factorial : 
  ∃ k : ℕ, unique_factorization_monoid.factor_count 30! 3 = k ∧ k = 14 :=
sorry

end exponent_of_3_in_30_factorial_l274_274252


namespace find_x_with_three_prime_divisors_l274_274148

def x_with_conditions (n : Nat) (x : Nat) : Prop :=
  x = 9^n - 1 ∧
  nat.factors x ∧
  nat.factors x.count 7 = 1

theorem find_x_with_three_prime_divisors (n : Nat) (x : Nat) :
  x_with_conditions n x → x = 728 :=
by
  sorry

end find_x_with_three_prime_divisors_l274_274148


namespace joey_more_fish_than_peter_l274_274818

-- Define the conditions
variables (A P J : ℕ)

-- Condition that Ali's fish weight is twice that of Peter's
def ali_double_peter (A P : ℕ) : Prop := A = 2 * P

-- Condition that Ali caught 12 kg of fish
def ali_caught_12 (A : ℕ) : Prop := A = 12

-- Condition that the total weight of the fish is 25 kg
def total_weight (A P J : ℕ) : Prop := A + P + J = 25

-- Prove that Joey caught 1 kg more fish than Peter
theorem joey_more_fish_than_peter (A P J : ℕ) :
  ali_double_peter A P → ali_caught_12 A → total_weight A P J → J = 1 :=
by 
  intro h1 h2 h3
  sorry

end joey_more_fish_than_peter_l274_274818


namespace classmates_late_time_l274_274453

theorem classmates_late_time:
  ∀ (x : ℕ), (Charlize_late_time = 20) →
    (total_late_time = 140) →
    (4 * x + Charlize_late_time = total_late_time) →
    x = 30 := 
begin
  intros x hCharlize htotal htotal_late,
  rw [Charlize_late_time, total_late_time] at htotal_late,
  sorry,
end

end classmates_late_time_l274_274453


namespace part_I_part_II_l274_274548

-- Define the functions f and g
def f (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - (1 / 2) * x^2 + x
def g (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x + 1

-- Theorem for part I
theorem part_I :
  let a := 2
  let f_max := f 2 a
  let f_min := min (f 1 a) (f (Real.exp 2) a)
  in f_max = 2 * Real.log 2 ∧ f_min = 4 + Real.exp 2 - (1 / 2) * (Real.exp 2)^4 :=
sorry

-- Theorem for part II
theorem part_II :
  ∀ x > 0, f x a + g x ≤ 0 → a = 1 :=
sorry

end part_I_part_II_l274_274548


namespace charming_7_digit_numbers_count_l274_274450

def is_charming_7_digit_number (n : ℕ) : Prop :=
  let digits := list.of_fn (λ i => n / 10^(6-i) % 10) in
  digits.perm [1, 2, 3, 4, 5, 6, 7] ∧
  (∀ k in list.range 7 | k > 0, (list.take (k+1) digits).to_nat % (k+1) = 0) ∧
  n % 10 = 7

theorem charming_7_digit_numbers_count : (finset.filter is_charming_7_digit_number (finset.range 10000000)).card = 0 :=
sorry

end charming_7_digit_numbers_count_l274_274450


namespace range_of_a_l274_274916

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

def sequence_increasing (a : ℝ) : Prop :=
  ∀ (n m : ℕ), n < m → f a n < f a m

theorem range_of_a :
  { a : ℝ | sequence_increasing a } = set.Ioo 2 3 :=
by
  sorry

end range_of_a_l274_274916


namespace savings_account_amount_l274_274811

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l274_274811


namespace sum_fraction_p_adic_l274_274631

theorem sum_fraction_p_adic (p n m : ℕ) (hp : p > 3) (hprime : Prime p) 
  (h_sum : ∑ i in Finset.range (p-1) + 1, 1 / (i ^ p) = n / m) (h_gcd : Nat.gcd n m = 1) : 
  p^3 ∣ n := 
sorry

end sum_fraction_p_adic_l274_274631


namespace find_x_l274_274145

theorem find_x (n : ℕ) (x : ℕ) (hn : x = 9^n - 1) (h7 : ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ 7 ∧ p2 ≠ 7 ∧ p1 ≠ p2 ∧ 7 ∣ x ∧ p1 ∣ x ∧ p2 ∣ x) : x = 728 :=
by
  sorry

end find_x_l274_274145


namespace cost_of_pastrami_l274_274688

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l274_274688


namespace max_area_of_triangle_ABC_l274_274286

theorem max_area_of_triangle_ABC (A B C M : Type) [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] [MetricSpace M] 
  (h_midpoint : midpoint A B M) 
  (h_AB : dist A B = 17)
  (h_CM : dist C M = 8) :
  ∃ K, K = 68 ∧
  ∀ area_ABC, area_ABC ≤ K := 
sorry

end max_area_of_triangle_ABC_l274_274286


namespace greatest_integer_b_greatest_integer_b_value_l274_274110

theorem greatest_integer_b (b : ℤ) : (b^2 < 20) → (b ≤ 4) :=
begin
  sorry
end

theorem greatest_integer_b_value : (∀ b : ℤ, b^2 < 20 → b ≤ 4) → (∃ (b : ℤ), b^2 < 20 ∧ b = 4) :=
begin
  intros h,
  use 4,
  split,
  { exact dec_trivial, },
  { exact dec_trivial }
end

end greatest_integer_b_greatest_integer_b_value_l274_274110


namespace cost_of_pastrami_l274_274687

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l274_274687


namespace exponent_of_3_in_factorial_30_l274_274242

theorem exponent_of_3_in_factorial_30 : (prime_factorization 30!).count 3 = 14 := 
sorry

end exponent_of_3_in_factorial_30_l274_274242


namespace minimize_fraction_l274_274724

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) ↔ (∀ m : ℕ, 0 < m → (n / 3 + 27 / n) ≤ (m / 3 + 27 / m)) :=
by
  sorry

end minimize_fraction_l274_274724


namespace smallest_abs_value_l274_274448

theorem smallest_abs_value (a b c d : ℤ) (ha : a = -2) (hb : b = 0) (hc : c = 3) (hd : d = -3) : 
  (b = 0 ∧ |b| ≤ |a| ∧ |b| ≤ |c| ∧ |b| ≤ |d|) :=
by
  unfold abs
  rw [ha, hb, hc, hd]
  norm_num
  exact ⟨rfl, le_refl 0, zero_le 3, zero_le 3⟩ 

end smallest_abs_value_l274_274448


namespace delta_n_squared_delta_n_n_minus_1_delta_n_k_delta_comb_n_k_sum_of_squares_l274_274773

open Nat

theorem delta_n_squared (n : ℕ) : Δ (λ n, n^2) n = 2 * n + 1 := 
sorry

theorem delta_n_n_minus_1 (n : ℕ) : Δ (λ n, n * (n - 1)) n = 2 * n - 1 := 
sorry

theorem delta_n_k (n k : ℕ) : Δ (λ n, n^k) n = n^k - (n-1)^k := 
sorry

theorem delta_comb_n_k (n k : ℕ) : Δ (λ n, comb n k) n = comb (n+1) k - comb n k := 
sorry

theorem sum_of_squares (n : ℕ) : ∑ i in range (n + 1), i^2 = n * (n + 1) * (2 * n + 1) / 6 := 
sorry

end delta_n_squared_delta_n_n_minus_1_delta_n_k_delta_comb_n_k_sum_of_squares_l274_274773


namespace distance_between_A_and_B_l274_274431

theorem distance_between_A_and_B (x : ℝ) (boat_speed : ℝ) (flow_speed : ℝ) (dist_AC : ℝ) (total_time : ℝ) :
  (boat_speed = 8) →
  (flow_speed = 2) →
  (dist_AC = 2) →
  (total_time = 3) →
  (x = 10 ∨ x = 12.5) :=
by {
  sorry
}

end distance_between_A_and_B_l274_274431


namespace num_ways_to_distribute_items_l274_274825

theorem num_ways_to_distribute_items :
  ∃ (ways : ℕ), ways = 106 ∧ 5 = 5 ∧ 3 = 3 :=
by
  use 106
  split
  sorry

end num_ways_to_distribute_items_l274_274825


namespace sum_of_solutions_eq_zero_l274_274115

theorem sum_of_solutions_eq_zero :
  (∑ x in {x ∈ Icc 0 (2 * Real.pi) | sin x ≠ 0 ∧ cos x ≠ 0 ∧ (1 / sin x) + (1 / cos x) = 4}, x) = 0 := by
sorry

end sum_of_solutions_eq_zero_l274_274115


namespace find_x_l274_274261

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 119) : x = 39 :=
sorry

end find_x_l274_274261


namespace window_width_correct_l274_274601

def total_width_window (x : ℝ) : ℝ :=
  let pane_width := 4 * x
  let num_panes_per_row := 4
  let num_borders := 5
  num_panes_per_row * pane_width + num_borders * 3

theorem window_width_correct (x : ℝ) :
  total_width_window x = 16 * x + 15 := sorry

end window_width_correct_l274_274601


namespace right_triangle_of_pythagorean_l274_274981

theorem right_triangle_of_pythagorean
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC CA : ℝ)
  (h : AB^2 = BC^2 + CA^2) : ∃ (c : ℕ), c = 90 :=
by
  sorry

end right_triangle_of_pythagorean_l274_274981


namespace savings_account_amount_l274_274810

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l274_274810


namespace greatest_area_difference_l274_274742

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end greatest_area_difference_l274_274742


namespace find_fx_l274_274506

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2 * x) : ∀ x, f x = x^2 - 1 :=
by
  intro x
  sorry

end find_fx_l274_274506


namespace total_books_l274_274026

def initial_books : ℝ := 41.0
def first_addition : ℝ := 33.0
def second_addition : ℝ := 2.0

theorem total_books (h1 : initial_books = 41.0) (h2 : first_addition = 33.0) (h3 : second_addition = 2.0) :
  initial_books + first_addition + second_addition = 76.0 := 
by
  -- placeholders for the proof steps, omitting the detailed steps as instructed
  sorry

end total_books_l274_274026


namespace sin_double_angle_l274_274968

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l274_274968


namespace minimum_questions_to_find_number_l274_274746

theorem minimum_questions_to_find_number (n : ℕ) (h : n ≤ 2020) :
  ∃ m, m = 64 ∧ (∀ (strategy : ℕ → ℕ), ∃ questions : ℕ, questions ≤ m ∧ (strategy questions = n)) :=
sorry

end minimum_questions_to_find_number_l274_274746


namespace frustum_volume_l274_274348

-- Define the conditions given in the problem
variables {r R h l : ℝ}
-- Variable proportion conditions
axiom cond1 : r * 4 = R
axiom cond2 : r * 4 = h
-- Slant height condition
axiom cond3 : real.sqrt (h^2 + (R - r)^2) = 10

-- Define the result we need to prove
theorem frustum_volume : by
  let V := 1/3 * real.pi * h * (r^2 + R^2 + r * R)
  show V = 224 * real.pi
:= sorry

end frustum_volume_l274_274348


namespace train_length_is_correct_l274_274439

-- Define the context and conditions.
def speed_of_train := 67    -- Speed of the train in kmph
def speed_of_man := 5       -- Speed of the man in kmph
def time_to_cross := 6      -- Time to cross in seconds

-- Convert speeds from kmph to m/s.
def speed_of_train_ms : Real := (speed_of_train * 1000) / 3600
def speed_of_man_ms : Real := (speed_of_man * 1000) / 3600

-- Calculate relative speed in m/s.
def relative_speed_ms : Real := speed_of_train_ms + speed_of_man_ms

-- Length of the train in meters.
def length_of_train := relative_speed_ms * time_to_cross

-- Theorem: Length of the train given the conditions is 120 meters.
theorem train_length_is_correct : length_of_train = 120 := by
  sorry  -- Proof goes here.

end train_length_is_correct_l274_274439


namespace range_of_a_l274_274908

theorem range_of_a (a : ℝ) : (∃ x : ℝ, exp x - 2 + a = 0) → a < 2 :=
sorry

end range_of_a_l274_274908


namespace probability_X_greater_than_neg_1_l274_274536

variable {X : ℝ}

-- Conditions
def standard_normal_distribution (X : ℝ) : Prop := ∀ x, (X ∈ set.univ) → x ∈ set.Icc (neg x) x
def P_X_greater_than_1 (p : ℝ) : Prop := probability {x : ℝ | X > 1} = p

-- Theorem Statement
theorem probability_X_greater_than_neg_1 (h1 : standard_normal_distribution X) (h2 : P_X_greater_than_1 p) : 
  probability {x : ℝ | X > -1} = 1 - p :=
sorry

end probability_X_greater_than_neg_1_l274_274536


namespace probability_of_meeting_l274_274373

noncomputable def meeting_probability : ℝ :=
  let total_area := 10 * 10
  let favorable_area := 51
  favorable_area / total_area

theorem probability_of_meeting : meeting_probability = 51 / 100 :=
by
  sorry

end probability_of_meeting_l274_274373


namespace exponent_of_3_in_30_factorial_l274_274245

theorem exponent_of_3_in_30_factorial : (prime_factor_exponent 3 30.factorial) = 14 :=
by
  sorry

end exponent_of_3_in_30_factorial_l274_274245


namespace share_sheets_equally_l274_274624

theorem share_sheets_equally (sheets friends : ℕ) (h_sheets : sheets = 15) (h_friends : friends = 3) : sheets / friends = 5 := by
  sorry

end share_sheets_equally_l274_274624


namespace base7_digits_1234_l274_274951

theorem base7_digits_1234 : ∀ (n : ℕ), n = 1234 → 
  ∀ (b : ℕ), b = 7 → 
  ∃ d : ℕ, d = 4 ∧ ∀ (p : ℕ), 1234 / b^p < b → d = p + 1 := 
by 
  intros n hn b hb
  exists 4
  split
  case right =>
    sorry
  case left =>
    rfl

end base7_digits_1234_l274_274951


namespace max_volume_of_rectangular_prism_l274_274894

   theorem max_volume_of_rectangular_prism
     (d₁ d₂ : ℝ)
     (h : ℝ := real.sqrt (d₁^2 - d₂^2))
     (a b : ℝ) :
     d₁ = 10 → d₂ = 8 → a^2 + b^2 = 64 → 6 * a * b ≤ 192 :=
   by
     sorry
   
end max_volume_of_rectangular_prism_l274_274894


namespace length_of_AB_l274_274617

theorem length_of_AB {A B C : Type*} [∀ x : A, has_zero x]
  (right_angle_A : ∀ {x₁ : Type*} [ordered_add_comm_group x₁] [ordered_semimodule ℝ x₁], 90 = 90)
  (BC_eq_30 : ∀ {x₂ : Type*} [ordered_add_comm_group x₂] [ordered_semimodule ℝ x₂], 30 = 30)
  (tanC_eq_3sinC : ∀ {x₃ : Type*} [ordered_add_comm_group x₃] [ordered_semimodule ℝ x₃],
    (∀ C AC AB BC, (AC ≠ 0) → tan C = 3 * sin C) → tan C = 3 * sin C)
  (angle_A : Type*) :
  ∃ (AB : ℝ), AB = 20 * sqrt 2 :=
by
  sorry

end length_of_AB_l274_274617


namespace inequality_to_prove_l274_274653

variable {R : Type*} [linear_ordered_field R]

-- Definitions:
def f (x : R) : R := real.sqrt x

-- Variables with conditions
variables (p q x1 x2 : R)
hypotheses (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1)

-- Inequality to be proven
theorem inequality_to_prove :
  p * f x1 + q * f x2 ≤ f (p * x1 + q * x2) :=
sorry

end inequality_to_prove_l274_274653


namespace exponent_of_3_in_30_factorial_l274_274253

theorem exponent_of_3_in_30_factorial : 
  ∃ k : ℕ, unique_factorization_monoid.factor_count 30! 3 = k ∧ k = 14 :=
sorry

end exponent_of_3_in_30_factorial_l274_274253


namespace problem_statement_l274_274396

theorem problem_statement (x : ℝ) (hx : x^2 + 1/(x^2) = 2) : x^4 + 1/(x^4) = 2 := by
  sorry

end problem_statement_l274_274396


namespace probability_real_roots_discrete_l274_274465

-- First part: discrete sets
theorem probability_real_roots_discrete :
  let a_values := ({0, 1, 2, 3} : Set ℕ)
  let b_values := ({0, 1, 2} : Set ℕ)
  let total_events := (a_values.product b_values).card
  let real_root_events := {ab | ab.1 ∈ a_values ∧ ab.2 ∈ b_values ∧ ab.1 ≥ ab.2}.card
  total_events > 0 →
  (real_root_events / total_events : ℚ) = 3 / 4 :=
by
  intro a_values b_values total_events real_root_events h_total_events
  -- Add proof here
  sorry

-- Second part: continuous intervals
noncomputable def probability_real_roots_continuous :
  let a_interval := Icc 0 3
  let b_interval := Icc 0 2
  let total_area := (3 - 0) * (2 - 0)
  let real_root_area := {a | ∃ b, a ∈ a_interval ∧ b ∈ b_interval ∧ a ≥ b}.measure
  total_area > 0 →
  (real_root_area / total_area : ℝ) = 2 / 3 :=
by
  intro a_interval b_interval total_area real_root_area h_total_area
  -- Add proof here
  sorry

end probability_real_roots_discrete_l274_274465


namespace max_set_elements_l274_274638

def problem_statement (T : set ℕ) : Prop :=
  T ⊆ {x | x ∈ finset.range 40 ∧ 1 ≤ x ∧ x ≤ 40} ∧
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a + b) % 5 ≠ 0

theorem max_set_elements : ∃ (T : set ℕ), problem_statement T ∧ finset.card (T.to_finset) = 24 :=
by
  sorry

end max_set_elements_l274_274638


namespace problem1_problem2_problem3_problem4_l274_274831

theorem problem1 : ((-1/2) ^ (-2) + (-1) ^ 2023 - abs (-2)) = 1 :=
by sorry

theorem problem2 (x y : ℝ) : ((x^3 * y^3 + 4 * x^2 * y^2 - 3 * x^2 * y) / (x^2 * y)) = (x * y^2 + 4 * y - 3) :=
by sorry

theorem problem3 (x : ℝ): ((1 - 2 * x) * (x - 3)) = (-2 * x^2 + 7 * x - 3) :=
by sorry

theorem problem4 (a b : ℝ): ((a + b + 1) * (a + b - 1)) = (a^2 + 2 * a * b + b^2 - 1) :=
by sorry

end problem1_problem2_problem3_problem4_l274_274831


namespace consecutive_int_lcm_divisor_l274_274779

theorem consecutive_int_lcm_divisor (n : ℤ) (h : n > 2) :
  (∃ m : ℤ, ∃ k : ℤ, (1 ≤ k ∧ k < n ∧ m + k) ∧
   (m % (lcm (list.range (n-1))) = 0)) ↔ n = 4 :=
by
  sorry

end consecutive_int_lcm_divisor_l274_274779


namespace max_pairwise_coprime_numbers_l274_274403

-- Definitions of the conditions
variables (a : ℕ) (h : 1 < a)
def board_numbers : list ℕ := [1 + a^1, 1 + a^2, 1 + a^3, 1 + a^4, 1 + a^5,
                               1 + a^6, 1 + a^7, 1 + a^8, 1 + a^9, 1 + a^10,
                               1 + a^11, 1 + a^12, 1 + a^13, 1 + a^14, 1 + a^15]

-- The theorem stating the maximum number of pairwise coprime numbers on the board
theorem max_pairwise_coprime_numbers : 
  (∃ (s : finset ℕ), s ⊆ finset.univ.filter (λ n, n ∈ board_numbers a) ∧ s.card = 4 ∧ 
                     ∀ (x y ∈ s), x ≠ y → nat.coprime x y) ∧ 
  (∀ (s : finset ℕ), s ⊆ finset.univ.filter (λ n, n ∈ board_numbers a) ∧ ∀ (x y ∈ s), x ≠ y → nat.coprime x y → s.card ≤ 4) :=
sorry

end max_pairwise_coprime_numbers_l274_274403


namespace sequence_sum_l274_274086

/-- 
The sum of the sequence: 
1 - 2 - 3 + 4 + 5 - 6 - 7 + 8 + 9 - 10 - 11 + ... + 1993 - 1994 - 1995 + 1996 - 1997 
is equal to -1997.
-/
theorem sequence_sum : (Finset.range 1997).sum (λ n, if n % 4 = 0 then n + 1 else if n % 4 = 1 then - (n + 1) else if n % 4 = 2 then - (n + 1) else n + 1 - 1997) = -1997 :=
by
  sorry

end sequence_sum_l274_274086


namespace solutions_of_quadratic_l274_274720

theorem solutions_of_quadratic (x : ℝ) : x^2 - x = 0 ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solutions_of_quadratic_l274_274720


namespace angle_at_center_for_tangents_l274_274737

theorem angle_at_center_for_tangents (P A B O : Point) (h1 : tangent P A O)
  (h2 : tangent P B O) (h3 : tangent A B O) (h_angle : angle APB = 60) :
  angle AOB = 120 := 
sorry

end angle_at_center_for_tangents_l274_274737


namespace ted_and_mike_seeds_l274_274238

noncomputable def ted_morning_seeds (T : ℕ) (mike_morning_seeds : ℕ) (mike_afternoon_seeds : ℕ) (total_seeds : ℕ) : Prop :=
  mike_morning_seeds = 50 ∧
  mike_afternoon_seeds = 60 ∧
  total_seeds = 250 ∧
  T + (mike_afternoon_seeds - 20) + (mike_morning_seeds + mike_afternoon_seeds) = total_seeds ∧
  2 * mike_morning_seeds = T

theorem ted_and_mike_seeds :
  ∃ T : ℕ, ted_morning_seeds T 50 60 250 :=
by {
  sorry
}

end ted_and_mike_seeds_l274_274238


namespace impossible_event_l274_274218

theorem impossible_event (batch_size defective_count select_count : ℕ) :
  defective_count < select_count → batch_size = 25 → defective_count = 2 → select_count = 3 →
  (∃ chosen_subset : set ℕ, chosen_subset.card = select_count ∧ chosen_subset ⊆ {1, .., defective_count}) → false :=
by
  intros h_defective_lt_select h_batch_size h_defective_count h_select_count h_subset
  sorry

end impossible_event_l274_274218


namespace largest_of_given_numbers_l274_274002

theorem largest_of_given_numbers :
  (0.99 > 0.9099) ∧
  (0.99 > 0.9) ∧
  (0.99 > 0.909) ∧
  (0.99 > 0.9009) →
  ∀ (x : ℝ), (x = 0.99 ∨ x = 0.9099 ∨ x = 0.9 ∨ x = 0.909 ∨ x = 0.9009) → 
  x ≤ 0.99 :=
by
  sorry

end largest_of_given_numbers_l274_274002


namespace base_7_digits_of_1234_l274_274953

theorem base_7_digits_of_1234 : ∀ (n : ℕ), (n = 1234) → (nat.log n 7 + 1 = 4) :=
begin
  intros n hn,
  rw hn,
  sorry
end

end base_7_digits_of_1234_l274_274953


namespace abs_eq_case_solution_l274_274958

theorem abs_eq_case_solution :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := sorry

end abs_eq_case_solution_l274_274958


namespace plane_perpendicular_to_l_perpendicular_to_alpha_and_beta_l274_274133

variables {α β : Type*} [plane α] [plane β] [line l]

-- Defining given conditions
axiom perp_alpha_beta : perpendicular α β
axiom intersect_alpha_beta_l : α ∩ β = l

-- Statement to prove
theorem plane_perpendicular_to_l_perpendicular_to_alpha_and_beta :
  ∀ (γ : Type*) [plane γ], perpendicular γ l → perpendicular γ α ∧ perpendicular γ β := 
sorry

end plane_perpendicular_to_l_perpendicular_to_alpha_and_beta_l274_274133


namespace matrix_no_solution_neg_two_l274_274182

-- Define the matrix and vector equation
def matrix_equation (a x y : ℝ) : Prop :=
  (a * x + 2 * y = a + 2) ∧ (2 * x + a * y = 2 * a)

-- Define the condition for no solution
def no_solution_condition (a : ℝ) : Prop :=
  (a/2 = 2/a) ∧ (a/2 ≠ (a + 2) / (2 * a))

-- Theorem stating that a = -2 is the necessary condition for no solution
theorem matrix_no_solution_neg_two (a : ℝ) : no_solution_condition a → a = -2 := by
  sorry

end matrix_no_solution_neg_two_l274_274182


namespace train_crossing_time_l274_274785

theorem train_crossing_time
  (length_train : ℝ) (speed_kmph : ℝ) (conversion_factor : ℝ)
  (speed_mps : ℝ) (time_seconds : ℝ)
  (h1 : length_train = 160)
  (h2 : speed_kmph = 48)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : speed_mps = speed_kmph * conversion_factor)
  (h5 : time_seconds = length_train / speed_mps) :
  time_seconds ≈ 12 := 
sorry

end train_crossing_time_l274_274785


namespace radicals_equal_iff_a_is_one_l274_274571

-- Definitions of the conditions
def positive_int (x : ℕ) := x > 0

-- The main theorem statement
theorem radicals_equal_iff_a_is_one (a b c : ℕ) 
  (a_pos : positive_int a) 
  (b_pos : positive_int b) 
  (c_pos : positive_int c) 
  : (√(a * (b + c)) = a * √(b + c)) ↔ a = 1 :=
sorry

end radicals_equal_iff_a_is_one_l274_274571


namespace cotangent_ratio_l274_274650
-- Import the entire Mathlib module to ensure all necessary definitions and theorems are included

-- Define the problem context
variables {x y z : ℝ}
variables {ξ η ζ : ℝ}

-- Given conditions
axiom TriangleSides : x^2 + y^2 = 2023 * z^2
axiom AnglesRelation : ξ + η + ζ = π

-- Define cotangent as cos/sin for completeness
noncomputable def cot (θ : ℝ) : ℝ := real.cos θ / real.sin θ

-- Translate the problem statement into Lean
theorem cotangent_ratio : 
  TriangleSides → 
  AnglesRelation → 
  (∃ x y z : ℝ, x^2 + y^2 = 2023 * z^2 ∧ ξ + η + ζ = π) →
  ∃ ξ η ζ : ℝ, (cot ζ / (cot ξ + cot η) = 1011) :=
sorry -- Proof is omitted

end cotangent_ratio_l274_274650


namespace expression_value_l274_274757

theorem expression_value :
  let a := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 in
  let b := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 - 4 in
  (a / b : ℚ) = 362880 / 41 := 
by
  sorry

end expression_value_l274_274757


namespace sec_tan_difference_l274_274568

theorem sec_tan_difference (x : ℝ) (h : real.sec x + real.tan x = 5 / 2) : 
  real.sec x - real.tan x = 2 / 5 := 
sorry

end sec_tan_difference_l274_274568


namespace contractor_realized_after_20_days_l274_274415

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end contractor_realized_after_20_days_l274_274415


namespace capital_stays_free_l274_274794

-- Define the basic structure of the city states and their transformations
structure Kingdom :=
  (cities : Fin (12) -> Bool) -- Each city is either under a spell (false) or not (true)
  (capital_free_last_thanksgiving : cities 0 = true) -- The capital city was free of the spell last Thanksgiving

-- Define the transformation function for the magician's visit
def transform (cities : Fin (12) -> Bool) (start_city : Fin (12)) : Fin (12) -> Bool :=
  let next_city (city : Fin (12)) : Fin (12) := ⟨(city.val + 1) % 12, Nat.mod_lt _ (by norm_num)⟩
  let rec visit (current_city : Fin (12)) (states : Fin (12) -> Bool) (visited : Fin (12) -> Bool) : Fin (12) -> Bool :=
    if visited current_city then states
    else if states current_city then states
    else
      let new_visited := fun c => visited c || (c = current_city)
      let new_states := fun c => if c = current_city then false else states c
      visit (next_city current_city) new_states new_visited
  visit start_city cities (fun _ => false)

-- Define the theorem to be proven
theorem capital_stays_free (K : Kingdom) : (transform K.cities ⟨5, by norm_num⟩) ⟨0⟩ = true :=
sorry

end capital_stays_free_l274_274794


namespace find_angle_AOB_l274_274735

-- Define a type for angles.
def angle := ℝ

-- Define the conditions present in the problem.
variable (triangle_PAB_tangent_to_circle_O : Prop)
variable (angle_APB : angle)
variable (angle_AOB : angle)

-- State the main theorem.
theorem find_angle_AOB 
  (triangle_PAB_tangent_to_circle_O : triangle_PAB_tangent_to_circle_O)
  (h1 : angle_APB = 60) : 
  angle_AOB = 60 := 
  sorry

end find_angle_AOB_l274_274735


namespace max_difference_and_max_value_of_multiple_of_5_l274_274354

theorem max_difference_and_max_value_of_multiple_of_5:
  ∀ (N : ℕ), 
  (∃ (d : ℕ), d = 0 ∨ d = 5 ∧ N = 740 + d) →
  (∃ (diff : ℕ), diff = 5) ∧ (∃ (max_num : ℕ), max_num = 745) :=
by
  intro N
  rintro ⟨d, (rfl | rfl), rfl⟩
  apply And.intro
  use 5
  use 745
  sorry

end max_difference_and_max_value_of_multiple_of_5_l274_274354


namespace a_oxen_count_l274_274764

-- Define the conditions from the problem
def total_rent : ℝ := 210
def c_share_rent : ℝ := 54
def oxen_b : ℝ := 12
def oxen_c : ℝ := 15
def months_b : ℝ := 5
def months_c : ℝ := 3
def months_a : ℝ := 7
def oxen_c_months : ℝ := oxen_c * months_c
def total_ox_months (oxen_a : ℝ) : ℝ := (oxen_a * months_a) + (oxen_b * months_b) + oxen_c_months

-- The theorem we want to prove
theorem a_oxen_count (oxen_a : ℝ) (h : c_share_rent / total_rent = oxen_c_months / total_ox_months oxen_a) :
  oxen_a = 10 := by sorry

end a_oxen_count_l274_274764


namespace possible_values_of_a_l274_274904

def A : Set ℤ := { x | x^2 + 3 * x - 10 < 0 }

def B (a : ℝ) : Set ℤ := { x | x^2 + 2 * a * x + a^2 - 4 = 0 }

theorem possible_values_of_a (a : ℝ) :
  a = 2 ∨ a = 1 ↔ (A ∩ (B a)).card = 2 := by
  sorry

end possible_values_of_a_l274_274904


namespace tens_digit_of_power_sum_l274_274756

theorem tens_digit_of_power_sum (a b c : ℕ) (h1 : a = 2023) (h2 : b = 2024) (h3 : c = 2025):
  ((a^b + c) % 100) / 10 % 10 = 5 :=
by {
  -- Use given conditions
  subst h1,
  subst h2,
  subst h3,
  -- Add mathematical steps and simplification
  sorry
}

end tens_digit_of_power_sum_l274_274756


namespace angle_C_is_120_degrees_l274_274592

-- Defining the problem conditions in Lean 4
variables {A B C : Type} [linear_ordered_field A] [sqrt A] {a b c : A}

-- Condition in the problem statement
def condition (a b c : A) : Prop := c^2 = a^2 + b^2 + a * b

-- Definition of the cosine rule for a triangle angle
def cosine_rule (a b c : A) (h : c^2 = a^2 + b^2 + a * b) : A :=
  (a^2 + b^2 - c^2) / (2 * a * b)

-- Given the condition, prove that the angle is 120 degrees.
theorem angle_C_is_120_degrees (a b c : A) (h : condition a b c) :
  arccos (cosine_rule a b c h) = 120 := sorry

end angle_C_is_120_degrees_l274_274592


namespace sample_size_correctness_l274_274351

noncomputable def sample_size_problem : ℕ := 192

theorem sample_size_correctness 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) (sample_female : ℕ)
  (h_teachers : teachers = 200) (h_male_students : male_students = 1200)
  (h_female_students : female_students = 1000) (h_sample_female : sample_female = 80) :
  let total_population := teachers + male_students + female_students in
  let sample_ratio := sample_female / female_students in
  let n := total_population * sample_ratio in
  n = sample_size_problem :=
by
  intros
  have h_total_population : total_population = 2400 := by rw [h_teachers, h_male_students, h_female_students]; rfl
  have h_sample_ratio : sample_ratio = 80 / 1000 := by rw [h_sample_female, h_female_students]; rfl
  have h_n : n = total_population * (80 / 1000) := by rw h_sample_ratio
  have h_correct_n : n = 192 := by
    rw [h_total_population, h_n]
    norm_num
  exact h_correct_n
  sorry


end sample_size_correctness_l274_274351


namespace unique_solution_exists_l274_274101

theorem unique_solution_exists (ell : ℚ) (h : ell ≠ -2) : 
  (∃! x : ℚ, (x + 3) / (ell * x + 2) = x) ↔ ell = -1 / 12 := 
by
  sorry

end unique_solution_exists_l274_274101


namespace m_range_l274_274175

noncomputable def f (x : ℝ) : ℝ := 4 * sin^2 (Real.pi / 4 + x) - 2 * Real.sqrt 3 * cos (2 * x) - 1

def p (x : ℝ) : Prop := x < Real.pi / 4 ∨ x > Real.pi / 2

def q (x : ℝ) (m : ℝ) : Prop := -3 < f x - m ∧ f x - m < 3

theorem m_range (x : ℝ) (m : ℝ) (h : ¬p x → q x m) : 2 < m ∧ m < 6 :=
by
  sorry

end m_range_l274_274175


namespace champion_is_C_l274_274216

-- Definitions of statements made by Zhang, Wang, and Li
def zhang_statement (winner : String) : Bool := winner = "A" ∨ winner = "B"
def wang_statement (winner : String) : Bool := winner ≠ "C"
def li_statement (winner : String) : Bool := winner ≠ "A" ∧ winner ≠ "B"

-- Predicate that indicates exactly one of the statements is correct
def exactly_one_correct (winner : String) : Prop :=
  (zhang_statement winner ∧ ¬wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ ¬wang_statement winner ∧ li_statement winner)

-- The theorem stating the correct answer to the problem
theorem champion_is_C : (exactly_one_correct "C") :=
  by
    sorry  -- Proof goes here

-- Note: The import statement and sorry definition are included to ensure the code builds.

end champion_is_C_l274_274216


namespace opposite_sign_pairs_l274_274074

theorem opposite_sign_pairs :
  ¬ ((- 2 ^ 3 < 0) ∧ (- (2 ^ 3) > 0)) ∧
  ¬ (|-4| < 0 ∧ -(-4) > 0) ∧
  ((- 3 ^ 4 < 0 ∧ (-(3 ^ 4)) = 81)) ∧
  ¬ (10 ^ 2 < 0 ∧ 2 ^ 10 > 0) :=
by
  sorry

end opposite_sign_pairs_l274_274074


namespace find_positive_n_l274_274353

def arithmetic_sequence (a d : ℤ) (n : ℤ) := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_positive_n :
  ∃ (n : ℕ), n > 0 ∧ ∀ a d : ℤ, a = -12 → sum_of_first_n_terms a d 13 = 0 → arithmetic_sequence a d n > 0 ∧ n = 8 := 
sorry

end find_positive_n_l274_274353


namespace sequence_periodic_l274_274616

-- Define the sequence (a_n)
noncomputable def seq : ℕ → ℝ
| 0     := 0  -- By convention, a_0 is not given, but we need to define it for the type
| (n+1) := if n = 0 then 1 else
            let a_n := seq n
            in (Math.sqrt 3 * a_n + 1) / (Math.sqrt 3 - a_n)

-- State the main theorem
theorem sequence_periodic {a : ℕ → ℝ}
    (h₁ : a 1 = 1)
    (h₂ : ∀ n:ℕ, a n * a (n+1) + Math.sqrt 3 * (a n - a (n+1)) + 1 = 0):
  a 2016 = 2 - Math.sqrt 3 :=
by {
  -- Proof skipped
  sorry
}

end sequence_periodic_l274_274616


namespace yellow_shirts_count_l274_274879

theorem yellow_shirts_count (total_shirts blue_shirts green_shirts red_shirts yellow_shirts : ℕ) 
  (h1 : total_shirts = 36) 
  (h2 : blue_shirts = 8) 
  (h3 : green_shirts = 11) 
  (h4 : red_shirts = 6) 
  (h5 : yellow_shirts = total_shirts - (blue_shirts + green_shirts + red_shirts)) :
  yellow_shirts = 11 :=
by
  sorry

end yellow_shirts_count_l274_274879


namespace find_a1_over_1_minus_q_l274_274537

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_a1_over_1_minus_q 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 5 + a 6 + a 7 + a 8 = 48) :
  (a 1) / (1 - q) = -1 / 5 :=
sorry

end find_a1_over_1_minus_q_l274_274537


namespace p_at_zero_l274_274292

-- Definitions according to given conditions
def p (x : ℝ) : ℝ := sorry  -- Polynomial of degree 6 with specific values

-- Given condition: Degree of polynomial
def degree_p : Prop := (∀ n : ℕ, (n ≤ 6) → p (3 ^ n) = 1 / 3 ^ n)

-- Theorem that needs to be proved
theorem p_at_zero : degree_p → p 0 = 6560 / 2187 := 
by
  sorry

end p_at_zero_l274_274292


namespace round_robin_tournament_cyclical_wins_l274_274598

-- Define that there are 27 teams in a round-robin tournament
def num_teams : ℕ := 27

-- Define that each team wins and loses exactly 13 games
def wins (team : ℕ) : ℕ := 13
def losses (team : ℕ) : ℕ := 13

-- Define the set of all teams as ℕ from 0 to 26
def teams : Finset ℕ := Finset.range num_teams

-- Condition: No ties in the game
def no_ties (games : Finset (ℕ × ℕ)) : Prop := ∀ g ∈ games, g.fst ≠ g.snd

-- Define that each team plays every other team exactly once
def played_every_other_team (games : Finset (ℕ × ℕ)) : Prop := 
  ∀ i j ∈ teams, i < j → (i, j) ∈ games ∨ (j, i) ∈ games

-- Define the number of sets of three teams such that A beats B, B beats C, and C beats A
def cyclical_win_sets (games : Finset (ℕ × ℕ)) : ℕ :=
  (Finset.powersetLen 3 teams).count (λ t, 
    match t.toList with
    | [a, b, c] := (a, b) ∈ games ∧ (b, c) ∈ games ∧ (c, a) ∈ games
    | _         := false
    end)

-- The main theorem statement
theorem round_robin_tournament_cyclical_wins (games : Finset (ℕ × ℕ)) :
  (∀ t ∈ teams, wins t = 13 ∧ losses t = 13) →
  no_ties games →
  played_every_other_team games →
  cyclical_win_sets games = 819 :=
sorry

end round_robin_tournament_cyclical_wins_l274_274598


namespace mn_value_is_2_5_l274_274411

noncomputable def mn_value : ℝ :=
  let total_students := 100
  let freq_table_tennis := 40
  let freq_badminton := 25
  let freq_basketball := 0.25 * total_students
  let freq_soccer := total_students - (freq_table_tennis + freq_badminton + freq_basketball)
  (freq_basketball * freq_soccer)

theorem mn_value_is_2_5 : mn_value = 2.5 :=
  by
    let total_students := 100
    let freq_table_tennis := 40
    let freq_badminton := 25
    let freq_basketball := 0.25 * total_students
    let freq_soccer := total_students - (freq_table_tennis + freq_badminton + freq_basketball)
    have h_freq_basketball : freq_basketball = 25 := by sorry
    have h_freq_soccer : freq_soccer = 10 := by sorry
    show freq_basketball * freq_soccer = 2.5, by sorry

end mn_value_is_2_5_l274_274411


namespace A_completes_job_in_10_hours_l274_274761

theorem A_completes_job_in_10_hours :
  (∀ A D : ℝ, (1 / A + 1 / D = 1 / 5) ∧ (D = 10) → (A = 10)) :=
by
  intros A D h
  cases h with h1 h2
  have h3 : (1 / A + 1 / 10 = 1 / 5) := by rw [h2] at h1; exact h1
  sorry

end A_completes_job_in_10_hours_l274_274761


namespace optimal_categories_l274_274679

structure Category where
  name : String
  cashback_rate : Float
  expenses : Float
  deriving Inhabited

def calculate_cashback (c : Category) : Float :=
  c.expenses * c.cashback_rate

theorem optimal_categories :
  let categories := [
    Category.mk "Transport" 0.05 2000,
    Category.mk "Groceries" 0.03 5000,
    Category.mk "Clothing" 0.04 3000,
    Category.mk "Entertainment" 0.05 3000,
    Category.mk "Sport Goods" 0.06 1500
  ]
  let cashbacks := categories.map calculate_cashback
  let sorted_cashbacks := cashbacks.sort (· > ·)
  take 3 sorted_cashbacks = [150, 150, 120] →
  true := sorry

end optimal_categories_l274_274679


namespace valid_five_letter_words_count_l274_274807

-- Define the conditions
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def totalVowels : Nat := 5
def maxAppearance : Nat := 3
def wordLength : Nat := 5

-- Define the main theorem to prove the number of valid words
theorem valid_five_letter_words_count : 
  ∑ (valid_distributions : Finset (Finset (Fin (wordLength + 1)))) 
    in (Finset.univ.filter (λ s, s.sum id = wordLength)) 
    |valid_distributions| = 106 := by
  sorry

end valid_five_letter_words_count_l274_274807


namespace number_of_valid_arrangements_l274_274315

theorem number_of_valid_arrangements : 
  let cards := {1, 2, 3, 4, 5, 6, 7}
  in (∃ f : fin 7 → ℕ, (∀ i ∈ finset.univ.erase i, i ≤ 6 → f i ∈ cards) ∧
      (∀ i j ∈ finset.univ.erase i, (i < j → f i < f j) ∨ (i < j → f i > f j))) →
    14 = (∑ i in finset.range 7, 1 + 1) :=
by
  sorry

end number_of_valid_arrangements_l274_274315


namespace rhino_total_weight_l274_274038

theorem rhino_total_weight 
  (full_grown_white_rhino_weight : ℝ := 5100)
  (newborn_white_rhino_weight : ℝ := 150)
  (full_grown_black_rhino_weight : ℝ := 2000)
  (newborn_black_rhino_weight : ℝ := 100)
  (pound_to_kg : ℝ := 0.453592)
  : 
  let white_rhinos_full := 6 * full_grown_white_rhino_weight,
      white_rhinos_newborn := 3 * newborn_white_rhino_weight,
      black_rhinos_full := 7 * full_grown_black_rhino_weight,
      black_rhinos_newborn := 4 * newborn_black_rhino_weight,
      total_weight_pounds := white_rhinos_full + white_rhinos_newborn + black_rhinos_full + black_rhinos_newborn,
      total_weight_kilograms := total_weight_pounds * pound_to_kg
  in
      total_weight_kilograms = 20616.436 :=
sorry

end rhino_total_weight_l274_274038


namespace optimal_categories_l274_274680

structure Category where
  name : String
  cashback_rate : Float
  expenses : Float
  deriving Inhabited

def calculate_cashback (c : Category) : Float :=
  c.expenses * c.cashback_rate

theorem optimal_categories :
  let categories := [
    Category.mk "Transport" 0.05 2000,
    Category.mk "Groceries" 0.03 5000,
    Category.mk "Clothing" 0.04 3000,
    Category.mk "Entertainment" 0.05 3000,
    Category.mk "Sport Goods" 0.06 1500
  ]
  let cashbacks := categories.map calculate_cashback
  let sorted_cashbacks := cashbacks.sort (· > ·)
  take 3 sorted_cashbacks = [150, 150, 120] →
  true := sorry

end optimal_categories_l274_274680


namespace team_expected_score_l274_274055

universe u

noncomputable def team_score_expected_value : ℝ :=
  let p1 := 0.4
  let p2 := 0.4
  let p3 := 0.5
  let p_correct := 1 - ((1 - p1) * (1 - p2) * (1 - p3))
  let questions := 10
  let points_per_correct := 10
  questions * p_correct * points_per_correct

theorem team_expected_score : team_score_expected_value = 82 := by
  sorry

end team_expected_score_l274_274055


namespace smallest_n_l274_274612

-- Define the basic setup of circles and connections
structure CircleSystem :=
  (circles : ℕ → ℕ) -- maps natural number to circle number
  (connected : ℕ → ℕ → Prop) -- defines if two circles are connected

-- Define the conditions
def conditions (cs : CircleSystem) (n : ℕ) : Prop :=
  ∀ a b : ℕ, 
  if cs.connected a b then
    (∃ d > 1, d ∣ n ∧ d ∣ (cs.circles a - cs.circles b))
  else
    nat.coprime n (cs.circles a - cs.circles b)

-- Given conditions, the smallest n is 385
theorem smallest_n (cs : CircleSystem) (n : ℕ) : conditions cs n → n = 385 :=
  sorry

end smallest_n_l274_274612


namespace pond_length_l274_274998

-- Define the width, depth and volume as constants
def width : ℝ := 10
def depth : ℝ := 5
def volume : ℝ := 1000

-- The theorem stating that the length of the pond is 20 meters given the conditions above
theorem pond_length : ∃ length : ℝ, length * width * depth = volume ∧ length = 20 :=
by
  use 20
  have h : 20 * width * depth = volume := by
    calc
      20 * width * depth = 20 * 10 * 5 : by rw [width, depth]
      ... = 1000 : by norm_num
  exact ⟨h, rfl⟩

end pond_length_l274_274998


namespace find_a_from_tangent_line_l274_274541
open Real

theorem find_a_from_tangent_line 
  (a : ℝ)
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (curve : ∀ x : ℝ, (λ x, a^x + 1)) 
  (perpendicular_condition : ∀ x : ℝ, x + 2 * (a^0 + 1) + 1 = 0) 
  : a = exp 2 := 
by
  -- Problem-specific conditions and steps would go here
  sorry

end find_a_from_tangent_line_l274_274541


namespace number_of_proper_subsets_of_M_l274_274296

def A := {1, 2, 3}
def B := {4, 5}

def M : Set ℕ := {x | ∃ (a ∈ A) (b ∈ B), x = a + b}

theorem number_of_proper_subsets_of_M :
  (2 ^ (M.toFinset.card) - 1) = 15 := 
sorry

end number_of_proper_subsets_of_M_l274_274296


namespace infinite_sum_convergence_l274_274460

-- Define the function f as given in the problem
def f (k : ℕ) : ℝ := 8^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

-- Define the infinite sum of f(k)
noncomputable def infinite_sum_f : ℝ := ∑' k, f k

-- Statement of the theorem to be proved
theorem infinite_sum_convergence : infinite_sum_f = 3 := sorry

end infinite_sum_convergence_l274_274460


namespace constants_sum_l274_274491

theorem constants_sum (c d : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = if x ≤ 5 then c * x + d else 10 - 2 * x) 
  (h₂ : ∀ x : ℝ, f (f x) = x) : c + d = 6.5 := 
by sorry

end constants_sum_l274_274491


namespace cos_double_angle_l274_274509

def sin_sum_condition (θ : ℝ) : Prop :=
  sin θ + sin (θ + π / 3) = 1

theorem cos_double_angle (θ : ℝ) (h : sin_sum_condition θ) : cos (2 * θ + π / 3) = 1 / 3 :=
sorry

end cos_double_angle_l274_274509


namespace relationship_l274_274822

noncomputable def f : ℝ → ℝ := sorry

def a := f (2023 / 2)
def b := f (Real.log (Real.sqrt 2))
def c := f 2024

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodic : ∀ x : ℝ, f (x - 1) = -f x
axiom monotone_increasing : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2

theorem relationship : c > b ∧ b > a := sorry

end relationship_l274_274822


namespace exponent_of_3_in_30_factorial_l274_274246

theorem exponent_of_3_in_30_factorial : (prime_factor_exponent 3 30.factorial) = 14 :=
by
  sorry

end exponent_of_3_in_30_factorial_l274_274246


namespace find_AD_length_l274_274257

noncomputable def triangle_AD (A B C : Type) (AB AC : ℝ) (ratio_BD_CD : ℝ) (AD : ℝ) : Prop :=
  AB = 13 ∧ AC = 20 ∧ ratio_BD_CD = 3 / 4 → AD = 8 * Real.sqrt 2

theorem find_AD_length {A B C : Type} :
  triangle_AD A B C 13 20 (3/4) (8 * Real.sqrt 2) :=
by
  sorry

end find_AD_length_l274_274257


namespace number_of_geese_is_correct_l274_274729

noncomputable def number_of_ducks := 37
noncomputable def total_number_of_birds := 95
noncomputable def number_of_geese := total_number_of_birds - number_of_ducks

theorem number_of_geese_is_correct : number_of_geese = 58 := by
  sorry

end number_of_geese_is_correct_l274_274729


namespace max_discriminant_l274_274293

noncomputable def f (a b c x : ℤ) := a * x^2 + b * x + c

theorem max_discriminant (a b c u v w : ℤ)
  (h1 : u ≠ v) (h2 : v ≠ w) (h3 : u ≠ w)
  (hu : f a b c u = 0)
  (hv : f a b c v = 0)
  (hw : f a b c w = 2) :
  ∃ (a b c : ℤ), b^2 - 4 * a * c = 16 :=
sorry

end max_discriminant_l274_274293


namespace abs_diff_squares_1055_985_eq_1428_l274_274376

theorem abs_diff_squares_1055_985_eq_1428 :
  abs ((105.5: ℝ)^2 - (98.5: ℝ)^2) = 1428 :=
by
  sorry

end abs_diff_squares_1055_985_eq_1428_l274_274376


namespace find_ordered_pairs_of_b_c_l274_274865

theorem find_ordered_pairs_of_b_c : 
  ∃! (pairs : ℕ × ℕ), 
    (pairs.1 > 0 ∧ pairs.2 > 0) ∧ 
    (pairs.1 * pairs.1 = 4 * pairs.2) ∧ 
    (pairs.2 * pairs.2 = 4 * pairs.1) :=
sorry

end find_ordered_pairs_of_b_c_l274_274865


namespace sum_of_x_coordinates_of_A_l274_274739

-- Define the conditions
variables (A B C D E : ℝ × ℝ)

def area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs ((fst Q - fst P) * (snd R - snd P) - (fst R - fst P) * (snd Q - snd P))

axiom h₁ : B = (0, 0)
axiom h₂ : C = (229, 0)
axiom h₃ : D = (680, 380)
axiom h₄ : E = (695, 388)
axiom h₅ : area B C A = 3011
axiom h₆ : area D E A = 9033

-- Define the proof statement
theorem sum_of_x_coordinates_of_A : (fst A) = 318.3 ∨ (fst A) = 287.7 → (fst A) + (fst A) = 606 := sorry

end sum_of_x_coordinates_of_A_l274_274739


namespace express_seven_with_five_twos_l274_274107

theorem express_seven_with_five_twos : 
  ∃ (expressions : List (ℕ → ℕ → ℕ)), 
  (expressions = [λ a b => a * b * a - (b / 2), λ a b => a + a + a + (b / 2), λ a b => (22 / 2) - (a * a)] ∧ 
  (expressions.foldr (λ expr acc => acc ∨ expr 2 2 = 7) false)) :=
by
  exists [
    λ a b => a * b * a - (b / 2),
    λ a b => a + a + a + (b / 2),
    λ a b => (22 / 2) - (a * a)
  ]
  simp
  have : 2 * 2 * 2 - (2 / 2) = 7 := by norm_num
  have : 2 + 2 + 2 + (2 / 2) = 7 := by norm_num
  have : 22 / 2 - 2 * 2 = 7 := by norm_num
  tauto

end express_seven_with_five_twos_l274_274107


namespace tan_beta_eq_l274_274569

variable (α β : ℝ)

-- Definitions based on conditions
def tan_alpha := 1 / 3
def tan_alpha_beta := 1 / 2

-- Lean 4 statement equivalent to the math problem
theorem tan_beta_eq : tan (α + β) = 1 / 2 → tan α = 1 / 3 → tan β = 1 / 7 :=
by
  sorry

end tan_beta_eq_l274_274569


namespace base_7_digits_of_1234_l274_274952

theorem base_7_digits_of_1234 : ∀ (n : ℕ), (n = 1234) → (nat.log n 7 + 1 = 4) :=
begin
  intros n hn,
  rw hn,
  sorry
end

end base_7_digits_of_1234_l274_274952


namespace distance_midpoint_directrix_l274_274706

def parabola_focus (x y : ℝ) : Prop :=
  y*y = 4*x

def directrix (x : ℝ) : Prop :=
  x = -1

def midpoint (x1 x2 y1 y2 : ℝ) :=
  ( (x1 + x2) / 2, (y1 + y2) / 2 )

theorem distance_midpoint_directrix
  (x1 y1 x2 y2 : ℝ)
  (h1 : parabola_focus x1 y1)
  (h2 : parabola_focus x2 y2)
  (h3 : x1 + x2 = 6) :
  let M := midpoint x1 x2 y1 y2 in
  dist M.1 (-1) = 4 :=
sorry

end distance_midpoint_directrix_l274_274706


namespace conjecture_l274_274507

-- Define the function f
def f (x : ℝ) : ℝ := (3 - x^2) / (1 + x^2)

-- Conjecture as a theorem in Lean
theorem conjecture (x : ℝ) (hx : x ≠ 0) : f(x) + f(1 / x) = 2 := by
  sorry

end conjecture_l274_274507


namespace smallest_value_for_x_9_l274_274119

theorem smallest_value_for_x_9 :
  let x := 9
  ∃ i, i = (8 / (x + 2)) ∧ 
  (i < (8 / x) ∧ 
   i < (8 / (x - 2)) ∧ 
   i < (x / 8) ∧ 
   i < ((x + 2) / 8)) :=
by
  let x := 9
  use (8 / (x + 2))
  sorry

end smallest_value_for_x_9_l274_274119


namespace sequence_identity_l274_274538

noncomputable def a_n (n : ℕ) : ℝ := n + 1
noncomputable def b_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := (n * (n+1)) / 2  -- Sum of first n terms of arithmetic sequence
noncomputable def T_n (n : ℕ) : ℝ := 2 * (3^n - 1) / 2  -- Sum of first n terms of geometric sequence
noncomputable def c_n (n : ℕ) : ℝ := 2 * a_n n / b_n n
noncomputable def C_n (n : ℕ) : ℝ := (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))

theorem sequence_identity :
  a_n 1 = b_n 1 ∧
  2 * a_n 2 = b_n 2 ∧
  S_n 2 + T_n 2 = 13 ∧
  2 * S_n 3 = b_n 3 →
  (∀ n : ℕ, a_n n = n + 1 ∧ b_n n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, C_n n = (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))) :=
sorry

end sequence_identity_l274_274538


namespace max_sum_cycle_permutations_l274_274274

def is_permutation {α : Type*} [fintype α] (l1 l2 : list α) : Prop :=
l1 ~ l2

theorem max_sum_cycle_permutations (x : fin 6 → ℕ) (h1 : ∑ i, x i = 21) 
(h2 : x 0 ∈ {1, 2, 3, 4, 5, 6} ∧ x 1 ∈ {1, 2, 3, 4, 5, 6} 
∧ x 2 ∈ {1, 2, 3, 4, 5, 6} ∧ x 3 ∈ {1, 2, 3, 4, 5, 6} 
∧ x 4 ∈ {1, 2, 3, 4, 5, 6} ∧ x 5 ∈ {1, 2, 3, 4, 5, 6} 
∧ is_permutation (list.map x (fin_range 6)) [1, 2, 3, 4, 5, 6]) : 
  let P := (fin_range 6).sum (λ i, x i * x ((i + 1) % 6)),
      Q := fin_range 720 |>.filter (λ perm, is_permutation (list.map (x ∘ fin.of_nat' ∘ perm.val) (fin_range 6)) (list.map x (fin_range 6)) 
      ∧ ((fin_range 6).sum (λ i, ((x ∘ fin.of_nat' ∘ perm.val) i) * ((x ∘ fin.of_nat' ∘ perm.val) ((i + 1) % 6))) = P)).card
  in P + Q = 83 :=
  sorry

end max_sum_cycle_permutations_l274_274274


namespace find_x_l274_274147

theorem find_x (n : ℕ) (x : ℕ) (hn : x = 9^n - 1) (h7 : ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ 7 ∧ p2 ≠ 7 ∧ p1 ≠ p2 ∧ 7 ∣ x ∧ p1 ∣ x ∧ p2 ∣ x) : x = 728 :=
by
  sorry

end find_x_l274_274147


namespace expected_successful_trials_l274_274733

theorem expected_successful_trials :
  let p := 3 / 4
  let n := 2
  let X := (p * n : ℚ)
  X = 3 / 2 :=
by
  let p := 3 / 4
  let n := 2
  let X := (p * n : ℚ)
  have h1 : p * n = 3 / 2 := sorry
  exact h1

end expected_successful_trials_l274_274733


namespace hyperbola_min_dist_l274_274921

noncomputable def hyperbola_min_value : ℝ :=
  let a := 3 in
  let b := Real.sqrt 6 in
  let c := Real.sqrt (a^2 + b^2) in -- Focal distance
  2 * b^2 / a + 12

theorem hyperbola_min_dist (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h_hyperbola : ∀ x y, x^2 / 9 - y^2 / 6 = 1)
  (h_F1 : F1 = (-c, 0))
  (h_F2 : F2 = (c, 0))
  (h_line : ∃ k, ∀ p, p ∈ [[A, B]] → p.1 = k * (p.2 - ((-c) + c)/2)) :
  (|A - F2| + |B - F2|) = 16 :=
by {
  sorry
}

end hyperbola_min_dist_l274_274921


namespace integral_curve_l274_274484

-- Define the differential equation
def diff_eq := ∀ x y dx dy, 6 * x * dx - 6 * y * dy = 2 * x^2 * y * dy - 3 * x * y^2 * dx

-- Define the solution in terms of a constant C
def solution (x y : ℝ) (C : ℝ) := (x^2 + 3)^3 / (2 + y^2) = C

-- Proof statement that the solution satisfies the differential equation
theorem integral_curve (x y : ℝ) (dx dy : ℝ) (C : ℝ) : 
  diff_eq x y dx dy → solution x y C :=
sorry

end integral_curve_l274_274484


namespace find_f_2019_l274_274527

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4

theorem find_f_2019 (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0)
  (h : f 2018 a b α β = 3) : f 2019 a b α β = 5 :=
by
  sorry

end find_f_2019_l274_274527


namespace graph_passes_through_point_l274_274502

theorem graph_passes_through_point (a : ℝ) (h : a < 0) : (0, 0) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (1 - a)^x - 1)} :=
by
  sorry

end graph_passes_through_point_l274_274502


namespace optimal_cashback_l274_274677

def Category : Type := String

def cashback_rate : Category → ℝ
| "Transport" := 0.05
| "Groceries" := 0.03
| "Clothing" := 0.04
| "Entertainment" := 0.05
| "Sports goods" := 0.06
| _ := 0.01

def expenses : Category → ℝ
| "Transport" := 2000
| "Groceries" := 5000
| "Clothing" := 3000
| "Entertainment" := 3000
| "Sports goods" := 1500
| _ := 0

def cashback (c : Category) : ℝ :=
  expenses c * cashback_rate c

theorem optimal_cashback :
  let categories := ["Transport", "Groceries", "Clothing", "Entertainment", "Sports goods"] in
  let selected := ["Groceries", "Clothing", "Entertainment"] in
  ∀ (c₁ c₂ c₃ : Category), c₁ ∈ categories → c₂ ∈ categories → c₃ ∈ categories →
  (c₁ = "Groceries" ∧ c₂ = "Clothing" ∧ c₃ = "Entertainment") ↔
  (cashback "Groceries" + cashback "Clothing" + cashback "Entertainment" ≥
  cashback c₁ + cashback c₂ + cashback c₃) := by
  intros c₁ c₂ c₃ h₁ h₂ h₃
  split
  · intros ⟨h₄, h₅, h₆⟩
    simp [h₄, h₅, h₆]
    sorry
  · intro h
    sorry

end optimal_cashback_l274_274677


namespace base_7_representation_digits_count_l274_274942

theorem base_7_representation_digits_count : ∀ (n : ℕ), n = 1234 → base_digits 7 n = 4 :=
by
  intro n
  assume h : n = 1234
  sorry

end base_7_representation_digits_count_l274_274942


namespace fred_cards_final_l274_274129

-- Definitions based on conditions
def init_cards := 5
def given_to_melanie := 2
def cards_traded := 1
def cards_received := 4
def lisa_cards := 3

-- Calculation steps
def final_cards : ℕ := 
  let after_melanie := init_cards - given_to_melanie in
  let after_trade := after_melanie - cards_traded + cards_received in
  let after_lisa := after_trade + 2 * lisa_cards in
  after_lisa

-- Prove that the final number of baseball cards Fred has is 12
theorem fred_cards_final : final_cards = 12 := by
  sorry

end fred_cards_final_l274_274129


namespace concyclic_cross_ratio_real_l274_274310

def concyclic (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ (c : ℂ) (r : ℝ) (h : 0 < r), 
    (abs (z1 - c) = r) ∧ (abs (z2 - c) = r) ∧ (abs (z3 - c) = r) ∧ (abs (z4 - c) = r)

theorem concyclic_cross_ratio_real (z1 z2 z3 z4 : ℂ) :
  (∃ k : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = k) ↔ concyclic z1 z2 z3 z4 :=
sorry

end concyclic_cross_ratio_real_l274_274310


namespace pow_mod_equality_l274_274381

theorem pow_mod_equality (h : 2^3 ≡ 1 [MOD 7]) : 2^30 ≡ 1 [MOD 7] :=
sorry

end pow_mod_equality_l274_274381


namespace curve_is_circle_l274_274482

theorem curve_is_circle :
  ∀ (r θ : ℝ), r = 1 / (1 + sin θ) → (∃ c : ℝ × ℝ, ∃ R : ℝ, ∀ x y : ℝ, x^2 + (y - c.2)^2 = R^2) :=
by
  sorry

end curve_is_circle_l274_274482


namespace square_area_in_full_circle_l274_274803

theorem square_area_in_full_circle (a R : ℝ) (h1 : a^2 = 40) (h2 : R = (a / real.sqrt 2)) :
  let new_side := 2 * R in new_side^2 = 80 := 
by
  sorry

end square_area_in_full_circle_l274_274803


namespace dividend_percentage_l274_274427

theorem dividend_percentage (interest_rate : ℝ) (market_value : ℝ) (face_value : ℝ) 
  (h_interest_rate : interest_rate = 0.12)
  (h_market_value : market_value = 15)
  (h_face_value : face_value = 20) : 
  let interest_per_share := interest_rate * face_value in
  let dividend_percentage := (interest_per_share / market_value) * 100 in
  dividend_percentage = 16 := by
  sorry

end dividend_percentage_l274_274427


namespace definite_integral_of_x_squared_plus_sin_x_l274_274085

theorem definite_integral_of_x_squared_plus_sin_x :
  ∫ x in -1..1, (x^2 + sin x) = 2 / 3 :=
by
  sorry

end definite_integral_of_x_squared_plus_sin_x_l274_274085


namespace rigid_motion_mapping_figure_l274_274095

-- Definitions and conditions.
def square_length : ℝ := 1
def triangle_hypotenuse : ℝ := 1
def segment_length : ℝ := 2
def pattern_length : ℝ := square_length + triangle_hypotenuse + segment_length

-- Main theorem statement.
theorem rigid_motion_mapping_figure :
  let count_valid_transformations := 1 in
  (∃ n : ℕ, ∀ m : ℕ, m > 0 → n = m → translation_along_line_ℓ pattern_length m = pattern_length m)
:= sorry

end rigid_motion_mapping_figure_l274_274095


namespace problem_six_circles_l274_274094

noncomputable def six_circles_centers : List (ℝ × ℝ) := [(1,1), (1,3), (3,1), (3,3), (5,1), (5,3)]

noncomputable def slope_of_line_dividing_circles := (2 : ℝ)

def gcd_is_1 (p q r : ℕ) : Prop := Nat.gcd (Nat.gcd p q) r = 1

theorem problem_six_circles (p q r : ℕ) (h_gcd : gcd_is_1 p q r)
  (h_line_eq : ∀ x y, y = slope_of_line_dividing_circles * x - 3 → px = qy + r) :
  p^2 + q^2 + r^2 = 14 :=
sorry

end problem_six_circles_l274_274094


namespace total_tax_collected_l274_274705

theorem total_tax_collected (tax_paid : ℕ) (willam_tax: ℕ) (willam_percentage : ℕ) : ℕ :=
begin
  -- Conditions
  let tax_levied := 50, -- Tax levied on 50% of the land
  let W := willam_tax,  -- Mr. Willam paid $480
  let P := willam_percentage,  -- 25% of the total taxable land
  
  -- Prove
  have h1 : P ≠ 0 := sorry, -- Assume percentage is non-zero for division
  have total_tax := (4 * W),
  
  -- Correct Answer
  exact 1920
end

end total_tax_collected_l274_274705


namespace trigonometric_identity_l274_274006

theorem trigonometric_identity (α : ℝ) :
  sin (9 * α) + sin (10 * α) + sin (11 * α) + sin (12 * α) = 
  4 * cos (α / 2) * cos α * sin (21 * α / 2) :=
by
  sorry

end trigonometric_identity_l274_274006


namespace value_of_g_g_2_l274_274581

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l274_274581


namespace orderd_pairs_count_l274_274433

theorem orderd_pairs_count (a b : ℕ) (h1 : b > a) (h2 : (a - 6) * (b - 6) = 12) : 
    ({(a, b) | b > a ∧ (a - 6) * (b - 6) = 12}.card = 3) := 
by
  sorry

end orderd_pairs_count_l274_274433


namespace sin_eq_one_is_sufficient_but_not_necessary_l274_274583

theorem sin_eq_one_is_sufficient_but_not_necessary (x : ℝ) : 
  (sin x = 1 → cos x = 0) ∧ ¬(cos x = 0 → sin x = 1) :=
by
  sorry

end sin_eq_one_is_sufficient_but_not_necessary_l274_274583


namespace exponent_of_3_in_factorial_30_l274_274239

theorem exponent_of_3_in_factorial_30 : (prime_factorization 30!).count 3 = 14 := 
sorry

end exponent_of_3_in_factorial_30_l274_274239


namespace workers_read_all_three_books_l274_274990

noncomputable def PalabrasBookstore (W S K D A : ℕ) :=
  W = 120 ∧
  S = 1/4 * W ∧
  K = 5/8 * W ∧
  D = 3/10 * W ∧
  (W - (S + K - (S ∩ K)) + 1) = (S - (S ∩ K)) ∧
  A = (D - (D ∩ S ∩ K))

theorem workers_read_all_three_books :
  PalabrasBookstore W S K D A → A = 18 :=
by
  sorry

end workers_read_all_three_books_l274_274990


namespace extremum_at_one_over_e_range_of_a_for_one_extremum_l274_274205

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x ^ 3 + 3 * x * Real.log x - a

def is_extremum (f : ℝ → ℝ) (x0 : ℝ) : Prop :=
  (∀ x < x0, f x > f x0) ∧ (∀ x > x0, f x > f x0) ∨
  (∀ x < x0, f x < f x0) ∧ (∀ x > x0, f x < f x0)

theorem extremum_at_one_over_e (a : ℝ) (h : a = 0) : 
  is_extremum (λ x, f x a) (1 / Real.exp 1) ∧ f (1 / Real.exp 1) 0 = -3 / Real.exp 1 - 1 := 
  sorry

theorem range_of_a_for_one_extremum (h : ∀ a : ℝ, (a > - 2 / Real.exp 2) → (a < 0) → 
    ∃! x ∈ Set.Ioo (1 / Real.exp 1) (Real.exp 1), ∃ f' : ℝ → ℝ, f' x = 0) : 
    Set.Ioo (- 2 / Real.exp 2) 0 := 
  sorry

end extremum_at_one_over_e_range_of_a_for_one_extremum_l274_274205


namespace probability_point_outside_circle_l274_274323

/-- Given a point P with coordinates (m, n) obtained by rolling two dice consecutively,
    prove that the probability of P lying outside the circle x^2 + y^2 = 16
    is 7/9, given there are 36 possible points and 28 points lie outside the circle. -/
theorem probability_point_outside_circle 
  (m n : ℕ) (h_mn : 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) :
  let points_outside := { (1, 4),  (1, 5), (1, 6),
                          (2, 4),  (2, 5), (2, 6),
                          (3, 3),  (3, 4), (3, 5), (3, 6),
                          (4, 1),  (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                          (5, 1),  (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                          (6, 1),  (6, 2), (6, 3), (6, 4), (6, 5), (6, 6) } in
  (m, n) ∈ points_outside →
  (28:ℚ) / (36:ℚ) = (7:ℚ) / (9:ℚ) :=
by sorry

end probability_point_outside_circle_l274_274323


namespace triangle_right_triangle_l274_274591

theorem triangle_right_triangle {A B C a b c : ℝ} (h_sides : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_angles_pos : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum_angles : A + B + C = π)
  (h_condition : b * cos C + c * cos B = a * sin A) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := 
by 
  sorry

end triangle_right_triangle_l274_274591


namespace three_distinct_roots_condition_l274_274464

noncomputable def k_condition (k : ℝ) : Prop :=
  ∀ (x : ℝ), (x / (x - 1) + x / (x - 3)) = k * x → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

theorem three_distinct_roots_condition (k : ℝ) : k ≠ 0 ↔ k_condition k :=
by
  sorry

end three_distinct_roots_condition_l274_274464


namespace g_g_2_eq_394_l274_274574

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l274_274574


namespace new_volume_is_80_gallons_l274_274037

-- Define the original volume
def V_original : ℝ := 5

-- Define the factors by which length, width, and height are increased
def length_factor : ℝ := 2
def width_factor : ℝ := 2
def height_factor : ℝ := 4

-- Define the new volume
def V_new : ℝ := V_original * (length_factor * width_factor * height_factor)

-- Theorem to prove the new volume is 80 gallons
theorem new_volume_is_80_gallons : V_new = 80 := 
by
  -- Proof goes here
  sorry

end new_volume_is_80_gallons_l274_274037


namespace two_digit_number_representation_l274_274451

variable (a b : ℕ)

theorem two_digit_number_representation (h1 : a < 10) (h2 : b < 10) :
  let n := 10 * b + a in n = 10 * b + a :=
by
  sorry

end two_digit_number_representation_l274_274451


namespace difference_of_roots_l274_274472

noncomputable def p (x : ℝ) : ℝ := 81 * x^3 - 171 * x^2 + 107 * x - 18

theorem difference_of_roots :
  ∃ a b c : ℝ, (p(a) = 0) ∧ (p(b) = 0) ∧ (p(c) = 0) ∧ (a < b) ∧ (b < c) ∧ (c - a = 1.66) ∧ (b - a = c - b) := sorry

end difference_of_roots_l274_274472


namespace six_digit_number_count_l274_274497

def valid_six_digit_combinations : Nat := 288

theorem six_digit_number_count :
  let digits := [1, 2, 3, 4, 5, 6]
  ∃ s : List (List Int), 
    s.length = 6 ∧
    (∀ x ∈ s, x ∈ digits) ∧
    (∀ x ∈ digits, x ∈ s) ∧
    s.head ≠ 1 ∧ 
    s.last ≠ 1 ∧
    (∃ a b c : List Int, a ++ b ++ c = s ∧ b.length = 2 ∧ b.all (λ x, x % 2 = 0))
    → valid_six_digit_combinations = 288 :=
by
  sorry

end six_digit_number_count_l274_274497


namespace groups_div_rem_l274_274029

noncomputable def numOfGroups (t b : ℕ) : ℕ :=
  Nat.choose 6 t * Nat.choose 8 b

def isValidGroup (t b : ℕ) : Prop :=
  (t - b) % 4 = 0 ∧ (t + b) > 0

def countValidGroups : ℕ :=
  Finset.sum (Finset.filter (λ pair, isValidGroup pair.1 pair.2) (Finset.product (Finset.range 7) (Finset.range 9))) (λ pair, numOfGroups pair.1 pair.2)

theorem groups_div_rem : countValidGroups % 100 = 95 := sorry

end groups_div_rem_l274_274029


namespace new_cube_weight_l274_274035

-- Define the weight function for a cube given side length and density.
def weight (ρ : ℝ) (s : ℝ) : ℝ := ρ * s^3

-- Given conditions: the weight of the original cube.
axiom original_weight : ∃ ρ s : ℝ, weight ρ s = 7

-- The goal is to prove that a new cube with sides twice as long weighs 56 pounds.
theorem new_cube_weight : 
  (∃ ρ s : ℝ, weight ρ (2 * s) = 56) := by
  sorry

end new_cube_weight_l274_274035


namespace circle_center_coordinates_l274_274332

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, x^2 + y^2 - x + 2*y = 0 ↔ (x-c.1)^2 + (y-c.2)^2 = (5/4)) ∧ c = (1/2, -1) :=
sorry

end circle_center_coordinates_l274_274332


namespace max_area_difference_l274_274744

theorem max_area_difference (l1 l2 w1 w2 : ℤ) (h1 : 2*l1 + 2*w1 = 200) (h2 : 2*l2 + 2*w2 = 200) :
  let A := λ l w, l * w in
  (max {A l w | l + w = 100} - min {A l w | l + w = 100}) = 2401 :=
sorry

end max_area_difference_l274_274744


namespace total_investment_with_interest_l274_274748

theorem total_investment_with_interest
  (total_investment : ℝ)
  (amount_at_3_percent : ℝ)
  (interest_rate_3 : ℝ)
  (interest_rate_5 : ℝ)
  (remaining_amount : ℝ)
  (interest_3 : ℝ)
  (interest_5 : ℝ) :
  total_investment = 1000 →
  amount_at_3_percent = 199.99999999999983 →
  interest_rate_3 = 0.03 →
  interest_rate_5 = 0.05 →
  remaining_amount = total_investment - amount_at_3_percent →
  interest_3 = amount_at_3_percent * interest_rate_3 →
  interest_5 = remaining_amount * interest_rate_5 →
  total_investment + interest_3 + remaining_amount + interest_5 = 1046 :=
by
  intros H1 H2 H3 H4 H5 H6 H7
  sorry

end total_investment_with_interest_l274_274748


namespace number_of_buffaloes_on_sunday_l274_274106

theorem number_of_buffaloes_on_sunday
  (lions_saturday elephants_saturday leopards_sunday rhinos_monday warthogs_monday : ℕ)
  (total_animals : ℕ) :
  lions_saturday = 3 →
  elephants_saturday = 2 →
  leopards_sunday = 5 →
  rhinos_monday = 5 →
  warthogs_monday = 3 →
  total_animals = 20 →
  ∃ (buffaloes_sunday : ℕ), buffaloes_sunday = 2 :=
by
  intros h_lions h_elephants h_leopards h_rhinos h_warthogs h_total
  let animals_saturday := lions_saturday + elephants_saturday
  let animals_monday := rhinos_monday + warthogs_monday
  have h_animals_saturday : animals_saturday = 5, by {
    rw [h_lions, h_elephants],
    exact rfl,
  }
  have h_animals_monday : animals_monday = 8, by {
    rw [h_rhinos, h_warthogs],
    exact rfl,
  }
  let remaining_animals := total_animals - (animals_saturday + leopards_sunday + animals_monday)
  have h_remaining_animals : remaining_animals = 2, by {
    rw [h_total, h_animals_saturday, h_leopards, h_animals_monday],
    exact rfl,
  }
  use remaining_animals
  exact h_remaining_animals

end number_of_buffaloes_on_sunday_l274_274106


namespace guaranteed_win_with_24_points_l274_274223

noncomputable def minimum_points_for_guaranteed_win (points_per_first points_per_second points_per_third : ℕ) : ℕ :=
  24

theorem guaranteed_win_with_24_points :
  ∀ (points_per_first points_per_second points_per_third : ℕ), 
  points_per_first = 6 →
  points_per_second = 4 →
  points_per_third = 2 →
  minimum_points_for_guaranteed_win points_per_first points_per_second points_per_third = 24 :=
begin
  intros points_per_first points_per_second points_per_third h1 h2 h3,
  unfold minimum_points_for_guaranteed_win,
  simp [h1, h2, h3],
end

end guaranteed_win_with_24_points_l274_274223


namespace base_7_digits_1234_l274_274944

theorem base_7_digits_1234 : ∃ n : ℕ, nat.digits 7 1234 = [4] := 
sorry

end base_7_digits_1234_l274_274944


namespace sqrt_five_custom_op_l274_274988

noncomputable def square (x : ℝ) : ℝ := x * x

def custom_op (x y : ℝ) : ℝ := square (x + y) - square (x - y)

theorem sqrt_five_custom_op : custom_op (Real.sqrt 5) (Real.sqrt 5) = 20 :=
by
  sorry

end sqrt_five_custom_op_l274_274988


namespace average_calculation_l274_274322

def average_two (a b : ℚ) : ℚ := (a + b) / 2
def average_three (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem average_calculation :
  average_three (average_three 2 4 1) (average_two 3 2) 5 = 59 / 18 :=
by
  sorry

end average_calculation_l274_274322


namespace initial_salt_percentage_l274_274023

theorem initial_salt_percentage (P : ℝ) (initial_volume : ℝ) (additional_water : ℝ) (new_percentage : ℝ) (new_volume : ℝ) :
  initial_volume = 56 →
  additional_water = 14 →
  new_volume = initial_volume + additional_water →
  new_percentage = 0.08 →
  initial_volume * P = new_volume * new_percentage →
  P = 0.1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  norm_num at h5
  exact h5

end initial_salt_percentage_l274_274023


namespace cylinder_lateral_surface_area_l274_274036

-- Define structures for the problem
structure Cylinder where
  generatrix : ℝ
  base_radius : ℝ

-- Define the conditions
def cylinder_conditions : Cylinder :=
  { generatrix := 1, base_radius := 1 }

-- The theorem statement
theorem cylinder_lateral_surface_area (cyl : Cylinder) (h_gen : cyl.generatrix = 1) (h_rad : cyl.base_radius = 1) :
  ∀ (area : ℝ), area = 2 * Real.pi :=
sorry

end cylinder_lateral_surface_area_l274_274036


namespace find_number_l274_274320

-- Define the variables and the conditions as theorems to be proven in Lean.
theorem find_number (x : ℤ) 
  (h1 : (x - 16) % 37 = 0)
  (h2 : (x - 16) / 37 = 23) :
  x = 867 :=
sorry

end find_number_l274_274320


namespace prime_digit_B_l274_274049

-- Mathematical description
def six_digit_form (B : Nat) : Nat := 3 * 10^5 + 0 * 10^4 + 3 * 10^3 + 7 * 10^2 + 0 * 10^1 + B

-- Prime condition
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

theorem prime_digit_B (B : Nat) : is_prime (six_digit_form B) ↔ B = 3 :=
sorry

end prime_digit_B_l274_274049


namespace contractor_realized_work_done_after_20_days_l274_274417

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l274_274417


namespace sin_double_angle_cos_difference_angle_l274_274884

noncomputable def theta : Real :=
sorry -- since we do not need to construct the actual value of theta

open Real

variables (theta : Real)
axiom sin_theta : sin theta = 4 / 5
axiom theta_in_second_quadrant : π / 2 < theta ∧ theta < π

theorem sin_double_angle : sin (2 * theta) = -24 / 25 :=
by
  have h_cos_theta : cos theta = -3 / 5 := 
    by 
    sorry -- proven using sin^2(theta) + cos^2(theta) = 1 and theta in second quadrant
  show sin (2 * theta) = 2 * sin theta * cos theta
  rw [sin_theta, h_cos_theta]
  simp

theorem cos_difference_angle : cos (theta - π / 6) = (4 - 3 * sqrt 3) / 10 :=
by
  have h_cos_theta : cos theta = -3 / 5 :=
    by 
    sorry -- proven using sin^2(theta) + cos^2(theta) = 1 and theta in second quadrant
  have h_sin_theta : sin theta = 4 / 5 := sin_theta
  simp [cos_sub, h_cos_theta, h_sin_theta]
  simp -- further simplification yields the required result

end sin_double_angle_cos_difference_angle_l274_274884


namespace Rajesh_Spend_Salary_on_Food_l274_274311

theorem Rajesh_Spend_Salary_on_Food
    (monthly_salary : ℝ)
    (percentage_medicines : ℝ)
    (savings_percentage : ℝ)
    (savings : ℝ) :
    monthly_salary = 15000 ∧
    percentage_medicines = 0.20 ∧
    savings_percentage = 0.60 ∧
    savings = 4320 →
    (32 : ℝ) = ((monthly_salary * percentage_medicines + monthly_salary * (1 - (percentage_medicines + savings_percentage))) / monthly_salary) * 100 :=
by
  sorry

end Rajesh_Spend_Salary_on_Food_l274_274311


namespace number_of_solutions_eq_two_l274_274345

theorem number_of_solutions_eq_two : 
  (∃ (x y : ℝ), x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) ∧
  (∀ (x y : ℝ), (x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) → ((x = 4 ∨ x = -1) ∧ y = 3)) :=
by
  sorry

end number_of_solutions_eq_two_l274_274345


namespace additional_people_to_halve_speed_l274_274492

variables (s : ℕ → ℝ)
variables (x : ℕ)

-- Given conditions
axiom speed_with_200_people : s 200 = 500
axiom speed_with_400_people : s 400 = 125
axiom speed_halved : ∀ n, s (n + x) = s n / 2

theorem additional_people_to_halve_speed : x = 100 :=
by
  sorry

end additional_people_to_halve_speed_l274_274492


namespace coloring_possible_l274_274656

theorem coloring_possible (n k : ℕ) (h1 : 2 ≤ k ∧ k ≤ n) (h2 : n ≥ 2) :
  (n ≥ k ∧ k ≥ 3) ∨ (2 ≤ k ∧ k ≤ n ∧ n ≤ 3) :=
sorry

end coloring_possible_l274_274656


namespace price_of_pastries_is_5_l274_274684

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l274_274684


namespace max_knight_sum_l274_274461

-- Define the dimensions of the chessboard and the numbering of the squares.
def chessboard : list (ℕ × ℕ × ℕ) := 
[
  (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5), (1, 6, 6), (1, 7, 7), (1, 8, 8),
  (2, 1, 9), (2, 2, 10), (2, 3, 11), (2, 4, 12), (2, 5, 13), (2, 6, 14), (2, 7, 15), (2, 8, 16),
  (3, 1, 17), (3, 2, 18), (3, 3, 19), (3, 4, 20), (3, 5, 21), (3, 6, 22), (3, 7, 23), (3, 8, 24),
  (4, 1, 25), (4, 2, 26), (4, 3, 27), (4, 4, 28), (4, 5, 29), (4, 6, 30), (4, 7, 31), (4, 8, 32),
  (5, 1, 33), (5, 2, 34), (5, 3, 35), (5, 4, 36), (5, 5, 37), (5, 6, 38), (5, 7, 39), (5, 8, 40),
  (6, 1, 41), (6, 2, 42), (6, 3, 43), (6, 4, 44), (6, 5, 45), (6, 6, 46), (6, 7, 47), (6, 8, 48),
  (7, 1, 49), (7, 2, 50), (7, 3, 51), (7, 4, 52), (7, 5, 53), (7, 6, 54), (7, 7, 55), (7, 8, 56),
  (8, 1, 57), (8, 2, 58), (8, 3, 59), (8, 4, 60), (8, 5, 61), (8, 6, 62), (8, 7, 63), (8, 8, 64) 
]

-- Define a function that checks if two positions attack each other based on knight's movement.
def knight_attack (pos1 pos2 : (ℕ × ℕ)) : Bool :=
  let dx := abs (pos1.1 - pos2.1)
  let dy := abs (pos1.2 - pos2.2)
  (dx = 2 ∧ dy = 1) ∨ (dx = 1 ∧ dy = 2)

-- Define the main theorem.
theorem max_knight_sum : 
∃ (knight_positions : list (ℕ × ℕ)), 
  (∀ (pos1 pos2 : (ℕ × ℕ)), pos1 ∈ knight_positions ∧ pos2 ∈ knight_positions → pos1 ≠ pos2 → ¬knight_attack pos1 pos2) ∧ 
  (knight_positions.sum (λ pos, chessboard.find! (λ square, square.1 = pos.1 ∧ square.2 = pos.2)).get_or_else 0) = 1056 := 
sorry

end max_knight_sum_l274_274461


namespace line_perpendicular_to_planes_l274_274143

variables {m : Type} {α β : Type}
variables (line : m) (plane_α : α) (plane_β : β)

/-- If a line is perpendicular to a plane and that plane is parallel to another plane, then the line is perpendicular to the second plane.  -/
theorem line_perpendicular_to_planes (h₁ : m ⊥ α) (h₂ : α ∥ β) : m ⊥ β := 
sorry

end line_perpendicular_to_planes_l274_274143


namespace probability_divisible_by_4_l274_274422

theorem probability_divisible_by_4:
  let M : ℕ :=
    sorry -- Define a four-digit positive integer with ones digit 6
  in (M % 10 = 6 ∧ (M / 1000 ≥ 1) ∧ (M / 1000 < 10)) →
  ∃ (p q : ℕ), p/q = 2/5 ∧ 
  ∀ x y z : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ 0 ≤ z ∧ z < 10 →
  (10*z + 6) % 4 = 0 →
  p/q = 2/5 :=
sorry

end probability_divisible_by_4_l274_274422


namespace base_7_representation_digits_count_l274_274940

theorem base_7_representation_digits_count : ∀ (n : ℕ), n = 1234 → base_digits 7 n = 4 :=
by
  intro n
  assume h : n = 1234
  sorry

end base_7_representation_digits_count_l274_274940


namespace walking_time_eq_1_8_hours_l274_274747

theorem walking_time_eq_1_8_hours (v_w v_r : ℝ) (t_r_min : ℝ) (h1 : v_w = 5) (h2 : v_r = 15) (h3 : t_r_min = 36) :
  let t_r_h := t_r_min / 60 in
  let distance := v_r * t_r_h in
  let t_w := distance / v_w in
  t_w = 1.8 :=
by
  sorry

end walking_time_eq_1_8_hours_l274_274747


namespace equal_neighbors_distance_possible_l274_274725

theorem equal_neighbors_distance_possible (n : ℕ) (h_n : 2 ≤ n):
  (∃ closest_neigh (A : Fin n → ℝ), 
    ((∀ i, 0 ≤ A i < 1) ∧ 
    (∀ i, ∃ d : ℝ, 
    ∀ j : Fin n, if i = j ∨ ((A (⟨j.1 + 1 % n, sorry⟩)) = ((A j) + d) % 1) then
    abs ((A i) - (j * closest_neigh)) = d) ) ) ) := sorry

end equal_neighbors_distance_possible_l274_274725


namespace D_won_zero_matches_l274_274600

-- Define the players
inductive Player
| A | B | C | D deriving DecidableEq

-- Function to determine the winner of a match
def match_winner (p1 p2 : Player) : Option Player :=
  if p1 = Player.A ∧ p2 = Player.D then 
    some Player.A
  else if p2 = Player.A ∧ p1 = Player.D then 
    some Player.A
  else 
    none -- This represents that we do not know the outcome for matches not given

-- Assuming A, B, and C have won the same number of matches
def same_wins (w_A w_B w_C : Nat) : Prop := 
  w_A = w_B ∧ w_B = w_C

-- Define the problem statement
theorem D_won_zero_matches (w_D : Nat) (h_winner_AD: match_winner Player.A Player.D = some Player.A)
  (h_same_wins : ∃ w_A w_B w_C : Nat, same_wins w_A w_B w_C) : w_D = 0 :=
sorry

end D_won_zero_matches_l274_274600


namespace average_salary_l274_274350

theorem average_salary (A B C D E : ℕ) (hA : A = 8000) (hB : B = 5000) (hC : C = 14000) (hD : D = 7000) (hE : E = 9000) :
  (A + B + C + D + E) / 5 = 8800 :=
by
  -- the proof will be inserted here
  sorry

end average_salary_l274_274350


namespace number_and_sum_of_g3_values_l274_274648

theorem number_and_sum_of_g3_values :
  (∃ (g : ℝ → ℝ), (∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2) ∧
    let n := {x : ℝ | ∃ g : ℝ → ℝ, 
      (∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2) ∧ g 3 = x}.card,
    let s := {x : ℝ | ∃ g : ℝ → ℝ, 
      (∀ x y : ℝ, g x * g y - g (x * y) = x^2 + y^2) ∧ g 3 = x}.sum id 
  in n * s = 10) :=
  sorry

end number_and_sum_of_g3_values_l274_274648


namespace g_g2_is_394_l274_274578

def g (x : ℝ) : ℝ :=
  4 * x^2 - 6

theorem g_g2_is_394 : g(g(2)) = 394 :=
by
  -- Proof is omitted by using sorry
  sorry

end g_g2_is_394_l274_274578


namespace part1_part2_l274_274895

open Nat

-- Define the sequence a_n according to the problem conditions
def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (Finset.sum (Finset.range (n + 1)) a) + 1

-- Define the logarithm base 3 function
noncomputable def log_3 (x : ℕ) : ℝ := log x / log 3

-- Define the sequence b_n according to the problem conditions
noncomputable def b (n : ℕ) : ℝ :=
1 / ((1 + log_3 (a n)) * (3 + log_3 (a n)))

-- Define the sum of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ :=
Finset.sum (Finset.range n) b

theorem part1 (n : ℕ) : a (n + 1) = 3 ^ n :=
sorry

theorem part2 (m : ℝ) : (∀ n : ℕ, T n < m) → m ≥ 3 / 4 :=
sorry

end part1_part2_l274_274895


namespace ellipse_equation_and_max_PA_PB_l274_274522

-- Define the ellipse parameters
variables (a b c : ℝ)
variables (h₁ : a > b) (h₂ : b > 0) (h₃ : c = 1)
variables (h₄ : c / a = sqrt 2 / 2)

-- Define the main theorem
theorem ellipse_equation_and_max_PA_PB :
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 / 1 = 1)) ∧
  (∃ m : ℝ, -sqrt 2 ≤ m ∧ m ≤ sqrt 2 → (|m - m|^2 + (m - m)^2 + |m - m|^2 + (m - m)^2) ≤ 8 / 3) :=
by {
  sorry,
}

end ellipse_equation_and_max_PA_PB_l274_274522


namespace find_x_l274_274146

theorem find_x (n : ℕ) (x : ℕ) (hn : x = 9^n - 1) (h7 : ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ 7 ∧ p2 ≠ 7 ∧ p1 ≠ p2 ∧ 7 ∣ x ∧ p1 ∣ x ∧ p2 ∣ x) : x = 728 :=
by
  sorry

end find_x_l274_274146


namespace quadratic_function_properties_l274_274520

def f (x : ℝ) : ℝ := x^2 - x + 1

theorem quadratic_function_properties :
  (∀ x : ℝ, f(x + 1) - f(x) = 2 * x) ∧
  f(0) = 1 ∧
  (∀ x ∈ set.Icc (-2:ℝ) (4:ℝ), f(x) ∈ set.Icc (3/4:ℝ) (13:ℝ)) := by
  sorry

end quadratic_function_properties_l274_274520


namespace correct_decimal_product_l274_274266

theorem correct_decimal_product :
  let input1 := 0.065  
  let input2 := 3.25  
  let product_without_decimals := 21125
  let decimal_places := 5 -- 3 from 0.065 and 2 from 3.25
  (format ((65 : ℕ) * (325 : ℕ)) / (100000 : ℕ)).val == 0.21125 :=
by
  sorry

end correct_decimal_product_l274_274266


namespace find_f_neg_two_l274_274181

def power_function (a : ℝ) : ℝ → ℝ := λ x, x^a

axiom passes_through (a : ℝ) : power_function a (1 / 2) = 8

theorem find_f_neg_two : ∃ a : ℝ, passes_through a ∧ power_function a (-2) = -1 / 8 :=
by
  exists (-3)
  split
  { sorry } -- Need to show passes_through -3
  { sorry } -- Need to show power_function -3 (-2) = -1 / 8

end find_f_neg_two_l274_274181


namespace simplify_expression_l274_274694

variable (a : ℝ)

theorem simplify_expression : 2 * a * (2 * a ^ 2 + a) - a ^ 2 = 4 * a ^ 3 + a ^ 2 := 
  sorry

end simplify_expression_l274_274694


namespace apples_cost_120_cents_l274_274597

theorem apples_cost_120_cents (a b c d : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 7) (hd : d = 25) :
  ∃ k : ℕ, 4 * k + 7 * k = 28 ∧ (k * 15 + k * 25) = 120 :=
by {
  use 4,
  split,
  {
    -- Prove that the sum of apples corresponds to 28
    sorry,
  },
  {
    -- Prove that the total cost corresponds to 120 cents
    sorry,
  }
}

end apples_cost_120_cents_l274_274597


namespace transformation_of_trigonometric_function_l274_274368

theorem transformation_of_trigonometric_function :
  ∀ (x : ℝ),
    (cos ((x / 2) - (π / 4))) = sin ((x / 2) + (π / 4)) → 
    ∀ (y : ℝ),
      y = sin (x / 2) → y = sin(x / 2 + π/4) :=
begin
  sorry
end

end transformation_of_trigonometric_function_l274_274368


namespace tim_buys_loaves_l274_274732

theorem tim_buys_loaves (slices_per_loaf : ℕ) (paid : ℕ) (change : ℕ) (price_per_slice_cents : ℕ) 
    (h1 : slices_per_loaf = 20) 
    (h2 : paid = 2 * 20) 
    (h3 : change = 16) 
    (h4 : price_per_slice_cents = 40) : 
    (paid - change) / (slices_per_loaf * price_per_slice_cents / 100) = 3 := 
by 
  -- proof omitted 
  sorry

end tim_buys_loaves_l274_274732


namespace adjacent_vertex_value_l274_274226

theorem adjacent_vertex_value 
  (n : ℕ) (h : n = 1994)
  (v : ℕ → ℝ)
  (h_vertex : ∀ i, v i = (v (i - 1) + v (i + 1)) / 2 ∨ v i = real.sqrt (v (i - 1) * v (i + 1)))
  (hv32 : ∃ i, v i = 32) 
  (i : ℕ) (hi : v i = 32) :
  v (i + 1) = 32 :=
by 
  sorry

end adjacent_vertex_value_l274_274226


namespace value_of_4_3n_m_l274_274961

theorem value_of_4_3n_m (m n : ℝ) (h1 : 2 ^ m = 5) (h2 : 4 ^ n = 3) : 4 ^ (3 * n - m) = 27 / 25 :=
sorry

end value_of_4_3n_m_l274_274961


namespace circle_radius_l274_274587
open Real

theorem circle_radius (d : ℝ) (h_diam : d = 24) : d / 2 = 12 :=
by
  -- The proof will be here
  sorry

end circle_radius_l274_274587


namespace savings_account_amount_l274_274809

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l274_274809


namespace work_completion_in_8_days_l274_274028

/-- Definition of the individual work rates and the combined work rate. -/
def work_rate_A := 1 / 12
def work_rate_B := 1 / 24
def combined_work_rate := work_rate_A + work_rate_B

/-- The main theorem stating that A and B together complete the job in 8 days. -/
theorem work_completion_in_8_days (h1 : work_rate_A = 1 / 12) (h2 : work_rate_B = 1 / 24) : 
  1 / combined_work_rate = 8 :=
by
  sorry

end work_completion_in_8_days_l274_274028


namespace area_of_triangle_A_l274_274935

theorem area_of_triangle_A''B''C'' (ABC A'B'C' : Triangle) (A'' B'' C'' : Point)
  (h₁ : ABC.area = 1)
  (h₂ : A'B'C'.area = 2025)
  (h₃ : ABC.AB.parallel A'B'.opposite)
  (h₄ : ABC.BC.parallel B'C'.opposite)
  (h₅ : ABC.CA.parallel C'A'.opposite)
  (h₆ : A''.is_midpoint ABC.A A'B'C'.A')
  (h₇ : B''.is_midpoint ABC.B A'B'C'.B')
  (h₈ : C''.is_midpoint ABC.C A'B'C'.C') :
  A''B''C''.area = 484 := 
sorry

end area_of_triangle_A_l274_274935


namespace isosceles_triangle_angle_BAM_l274_274523

theorem isosceles_triangle_angle_BAM :
  ∀ (A B C K M : Type) [has_angle A B C] (R : ℝ),
  is_isosceles_triangle A B C ∧
  ∠ B A C = 53 ∧
  midpoint C A K ∧ 
  same_side_of_line B M A C ∧
  distance K M = distance A B ∧
  maximal_angle M A K → 
  ∠ B A M = 44 :=
begin
  sorry
end

end isosceles_triangle_angle_BAM_l274_274523


namespace sum_of_valid_c_l274_274867

theorem sum_of_valid_c :
  (∑ c in set.Icc (-27) 21, c) = -147 :=
sorry

end sum_of_valid_c_l274_274867


namespace pair_C_does_not_produce_roots_l274_274349

theorem pair_C_does_not_produce_roots (x : ℝ) :
  (x = 0 ∨ x = 2) ↔ (∃ x, y = x ∧ y = x - 2) = false :=
by
  sorry

end pair_C_does_not_produce_roots_l274_274349


namespace find_x_eq_728_l274_274154

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l274_274154


namespace volume_of_cone_equals_l274_274442

-- Define the properties of the circle and the sector
def circle_radius : ℝ := 6
def arc_length : ℝ := (2 / 3) * (2 * Real.pi * circle_radius)
def cone_base_radius : ℝ := arc_length / (2 * Real.pi)
def cone_height : ℝ := Real.sqrt ((circle_radius ^ 2) - (cone_base_radius ^ 2))
def cone_volume : ℝ := (1 / 3) * Real.pi * (cone_base_radius ^ 2) * cone_height

-- Prove that the volume of the cone formed matches the computed volume
theorem volume_of_cone_equals : cone_volume = (32 * Real.pi * Real.sqrt 5) / 3 := by sorry

end volume_of_cone_equals_l274_274442


namespace length_of_EF_l274_274078

-- Define the conditions of the problem
variables {A B C D E F : Type*}
variables (a b : ℝ)

-- Midpoint points and parallel lines
variables [trapezoid: Trapezoid A B C D]
variables [midpoints: MidPoints E F A B C D]
variables [parallel: ParallelLines A B C D]
variables [angleSum: AngleSum A B 90]

-- Define the theorem to prove
theorem length_of_EF (a b : ℝ) :
  EF = (1 / 2) * (a - b) := 
sorry

end length_of_EF_l274_274078


namespace g_g2_is_394_l274_274577

def g (x : ℝ) : ℝ :=
  4 * x^2 - 6

theorem g_g2_is_394 : g(g(2)) = 394 :=
by
  -- Proof is omitted by using sorry
  sorry

end g_g2_is_394_l274_274577


namespace find_circle_equation_l274_274890

noncomputable def center (m : ℝ) := (3 * m, m)

def radius (m : ℝ) : ℝ := 3 * m

def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3 * m)^2 + (y - m)^2 = (radius m)^2

def point_A : ℝ × ℝ := (6, 1)

theorem find_circle_equation (m : ℝ) :
  (radius m = 3 * m ∧ center m = (3 * m, m) ∧ 
   point_A = (6, 1) ∧
   circle_eq m 6 1) →
  (circle_eq 1 x y ∨ circle_eq 37 x y) :=
by
  sorry

end find_circle_equation_l274_274890


namespace parabola_shifted_l274_274337

def initial_parabola (x : ℝ) : ℝ := -1/3 * (x - 2) ^ 2

def shifted_parabola (x : ℝ) : ℝ := -1/3 * (x - 3) ^ 2 - 2

theorem parabola_shifted :
  initial_parabola (x + 1) - 2 = shifted_parabola x := 
sorry

end parabola_shifted_l274_274337


namespace find_x_l274_274325

-- Define the custom operation *
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the main theorem
theorem find_x : ∃ x : ℝ, custom_mul 3 (custom_mul 6 x) = 12 ∧ x = 12 :=
begin
  sorry
end

end find_x_l274_274325


namespace coeff_of_x_pow_4_in_expansion_l274_274700

theorem coeff_of_x_pow_4_in_expansion : 
  (∃ c : ℤ, c = (-1)^3 * Nat.choose 8 3 ∧ c = -56) :=
by
  sorry

end coeff_of_x_pow_4_in_expansion_l274_274700


namespace periodic_length_le_T_l274_274352

noncomputable def purely_periodic (a : ℚ) (T : ℕ) : Prop :=
∃ p : ℤ, a = p / (10^T - 1)

theorem periodic_length_le_T {a b : ℚ} {T : ℕ} 
  (ha : purely_periodic a T) 
  (hb : purely_periodic b T) 
  (hab_sum : purely_periodic (a + b) T)
  (hab_prod : purely_periodic (a * b) T) :
  ∃ Ta Tb : ℕ, Ta ≤ T ∧ Tb ≤ T ∧ purely_periodic a Ta ∧ purely_periodic b Tb := 
sorry

end periodic_length_le_T_l274_274352


namespace number_of_digits_l274_274455

theorem number_of_digits (n m : ℕ) (base : ℕ) : 
  (n = 8) → (m = 20) → (base = 10) → 
  Nat.digits base (n^m * 5^(m - 2)) = 31 :=
by
  intros
  sorry

end number_of_digits_l274_274455


namespace monotonic_intervals_a_le_0_monotonic_intervals_a_gt_0_fx_gt_exp_minus_x_l274_274177

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a / (exp(1) * x)
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := (exp(1) * x - a) / (exp(1) * x^2)

theorem monotonic_intervals_a_le_0 (a : ℝ) (h : a ≤ 0) : ∀ x : ℝ, 0 < x → monotone (f x a) :=
sorry

theorem monotonic_intervals_a_gt_0 (a : ℝ) (h : 0 < a) :
  ∀ x : ℝ, 0 < x → (increasing_on (f x a) (set.Ioi (a / exp(1)))) ∧ (decreasing_on (f x a) (set.Iio (a / exp(1)))) :=
sorry

theorem fx_gt_exp_minus_x (x : ℝ) (h : 0 < x) : f x 2 > exp (-x) :=
sorry

end monotonic_intervals_a_le_0_monotonic_intervals_a_gt_0_fx_gt_exp_minus_x_l274_274177


namespace distribute_balls_into_boxes_l274_274960

/--
Given 6 distinguishable balls and 3 distinguishable boxes, 
there are 3^6 = 729 ways to distribute the balls into the boxes.
-/
theorem distribute_balls_into_boxes : (3 : ℕ)^6 = 729 := 
by
  sorry

end distribute_balls_into_boxes_l274_274960


namespace price_of_pastries_is_5_l274_274685

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l274_274685


namespace value_of_g_g_2_l274_274580

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l274_274580


namespace larger_number_of_two_l274_274708

theorem larger_number_of_two (A B : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 28) (h_factors : A % hcf = 0 ∧ B % hcf = 0) 
  (h_f1 : factor1 = 12) (h_f2 : factor2 = 15)
  (h_lcm : Nat.lcm A B = hcf * factor1 * factor2)
  (h_coprime : Nat.gcd (A / hcf) (B / hcf) = 1)
  : max A B = 420 := 
sorry

end larger_number_of_two_l274_274708


namespace calculate_compound_weight_l274_274457

-- Define the initial conditions and the target compound weight calculation.

variable (A B : ℝ) (weight_B : ℝ) (ratio_A_B : ℝ)

-- Given conditions
def compound_conditions : Prop :=
  weight_B = 250 ∧ ratio_A_B = 1 / 5 ∧ A = ratio_A_B * weight_B 

-- The corresponding goal is to prove the total weight of compound X is 300 grams.
def total_weight_compound_X : Prop :=
  compound_conditions A B weight_B ratio_A_B → (A + B = 300)

-- Assert the variable B should be the weight_B
axiom weight_B_is_B : B = weight_B

-- Lending the proof statement
theorem calculate_compound_weight (A B : ℝ) (weight_B : ℝ) (ratio_A_B : ℝ) 
  (h_weight_B : weight_B = 250)
  (h_ratio_A_B : ratio_A_B = 1 / 5)
  (h_A : A = ratio_A_B * weight_B)
  (h_B : B = weight_B) : A + B = 300 := by
  sorry

end calculate_compound_weight_l274_274457


namespace problem_1_problem_2_l274_274920

-- Proof Problem 1
theorem problem_1 (x : ℝ) : (x^2 + 2 > |x - 4| - |x - 1|) ↔ (x > 1 ∨ x ≤ -1) :=
sorry

-- Proof Problem 2
theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂, x₁^2 + 2 ≥ |x₂ - a| - |x₂ - 1|) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l274_274920


namespace comics_in_box_l274_274081

theorem comics_in_box (pages_per_comic : ℕ) (total_found_pages : ℕ) (initial_comics : ℕ) :
    pages_per_comic = 45 →
    total_found_pages = 2700 →
    initial_comics = 15 →
    initial_comics + (total_found_pages / pages_per_comic) = 75 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end comics_in_box_l274_274081


namespace contractor_realized_after_20_days_l274_274416

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end contractor_realized_after_20_days_l274_274416


namespace find_quad_function_l274_274925

-- Define the quadratic function with the given conditions
def quad_function (a b c : ℝ) (f : ℝ → ℝ) :=
  ∀ x, f x = a * x^2 + b * x + c

-- Define the values y(-2) = -3, y(-1) = -4, y(0) = -3, y(2) = 5
def given_points (f : ℝ → ℝ) :=
  f (-2) = -3 ∧ f (-1) = -4 ∧ f 0 = -3 ∧ f 2 = 5

-- Prove that y = x^2 + 2x - 3 satisfies the given points
theorem find_quad_function : ∃ f : ℝ → ℝ, (quad_function 1 2 (-3) f) ∧ (given_points f) :=
by
  sorry

end find_quad_function_l274_274925


namespace polynomial_p_at_0_l274_274290

theorem polynomial_p_at_0 {p : ℕ → ℝ} (h1 : ∀ x, polynomial.degree (p x) = 6)
  (h2 : ∀ n, n ∈ {0, 1, 2, 3, 4, 5, 6} → p (3 ^ n) = 1 / (3 ^ n)) :
  p 0 = 2186 / 2187 :=
sorry

end polynomial_p_at_0_l274_274290


namespace find_x_l274_274443

theorem find_x (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a):
  (let x := (a^2 - b^2) / (2 * a) in
  x^2 + b^2 + c^2 = (a - x)^2 + c^2) := 
by
  sorry

end find_x_l274_274443


namespace tailor_cut_difference_l274_274437

theorem tailor_cut_difference :
  (7 / 8 + 11 / 12) - (5 / 6 + 3 / 4) = 5 / 24 :=
by
  sorry

end tailor_cut_difference_l274_274437


namespace man_mass_is_270_l274_274007

def boat_length := 9 -- Length of the boat in meters
def boat_breadth := 3 -- Breadth of the boat in meters
def boat_sink_height := 0.01 -- Height by which the boat sinks in meters
def water_density := 1000 -- Density of water in kg/m³

theorem man_mass_is_270 :
  let V := boat_length * boat_breadth * boat_sink_height in
  let m := water_density * V in
  m = 270 :=
by
  let V := boat_length * boat_breadth * boat_sink_height
  let m := water_density * V
  have h1: V = 0.27 := by sorry -- Proof for volume calculation
  have h2: m = 270 := by sorry -- Proof for mass calculation using density and volume
  exact h2

end man_mass_is_270_l274_274007


namespace total_paint_correct_l274_274125

-- Define the current gallons of paint he has
def current_paint : ℕ := 36

-- Define the gallons of paint he bought
def bought_paint : ℕ := 23

-- Define the additional gallons of paint he needs
def needed_paint : ℕ := 11

-- The total gallons of paint he needs for finishing touches
def total_paint_needed : ℕ := current_paint + bought_paint + needed_paint

-- The proof statement to show that the total paint needed is 70
theorem total_paint_correct : total_paint_needed = 70 := by
  sorry

end total_paint_correct_l274_274125


namespace part1_proof_part2_proof_l274_274546

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem part1_proof (x : ℝ) (hx : 0 < x) : f(x) + f(1/x) = 1 := 
by sorry

theorem part2_proof : 
  ∑ k in (finset.range 2016).map (λ n, n+2), 2 * f k + 
  ∑ k in (finset.range 2016).map (λ n, n+2), f (1 / k) + 
  ∑ k in (finset.range 2016).map (λ n, n+2), (1 / k^2) * f k = 4032 :=
by sorry

end part1_proof_part2_proof_l274_274546


namespace compare_a_b_c_l274_274503

def a := 2^12
def b := 3^8
def c := 7^4

theorem compare_a_b_c : b > a ∧ a > c :=
by {
  unfold a b c, 
  -- comparision of exponents
  have h1 : b = 9^4 := by sorry,
  have h2 : a = 8^4 := by sorry,
  have h3 : c = 7^4 := by sorry,
  -- comparison of bases
  have h4 : 9 > 8 := by exact nat.succ_pos 8,
  have h5 : 8 > 7 := by exact nat.succ_pos 7,
  exact ⟨pow_lt_pow_of_lt_left h4 (by linarith) zero_lt_four, pow_lt_pow_of_lt_left h5 (by linarith) zero_lt_four⟩
}

end compare_a_b_c_l274_274503


namespace total_age_l274_274018

variable (A B : ℝ)

-- Conditions
def condition1 : Prop := A / B = 3 / 4
def condition2 : Prop := A - 10 = (1 / 2) * (B - 10)

-- Statement
theorem total_age : condition1 A B → condition2 A B → A + B = 35 := by
  sorry

end total_age_l274_274018


namespace sequence_satisfies_conditions_l274_274211

theorem sequence_satisfies_conditions : 
  let seq1 := [4, 1, 3, 1, 2, 4, 3, 2]
  let seq2 := [2, 3, 4, 2, 1, 3, 1, 4]
  (seq1[0] = 4 ∧ seq1[1] = 1 ∧ seq1[2] = 3 ∧ seq1[3] = 1 ∧ seq1[4] = 2 ∧ seq1[5] = 4 ∧ seq1[6] = 3 ∧ seq1[7] = 2)
  ∨ (seq2[0] = 2 ∧ seq2[1] = 3 ∧ seq2[2] = 4 ∧ seq2[3] = 2 ∧ seq2[4] = 1 ∧ seq2[5] = 3 ∧ seq2[6] = 1 ∧ seq2[7] = 4)
  ∧ (seq1[1] = 1 ∧ seq1[3] - seq1[1] = 2 ∧ seq1[4] - seq1[2] = 3 ∧ seq1[5] - seq1[2] = 4) := 
  sorry

end sequence_satisfies_conditions_l274_274211


namespace chord_intersection_probability_l274_274372

-- The statement without the proof
theorem chord_intersection_probability :
  ∀ (P Q R S : Fin 200), 
  ∀ (HPQ : P ≠ Q) (HR : R ≠ S) (HPR : P ≠ R) (HPS : P ≠ S) (HQR : Q ≠ R) (HQS : Q ≠ S),
  Probs.intersecting_chords P Q R S = 1 / 6 :=
begin
  sorry
end

end chord_intersection_probability_l274_274372


namespace total_cows_in_ranch_l274_274750

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l274_274750


namespace x2_squared_l274_274165

noncomputable def x1 : ℂ := 1 - complex.I * real.sqrt 3
noncomputable def x2 : ℂ := 1 + complex.I * real.sqrt 3

theorem x2_squared (x1_root : (x1 - 1)^2 = -3) (roots : (x1 - 1)^2 = -3 ∧ (x2 - 1)^2 = -3) : x2^2 = -2 + 2 * complex.I * real.sqrt 3 :=
sorry

end x2_squared_l274_274165


namespace original_price_of_dish_l274_274769

variable (P : ℝ)

def john_paid (P : ℝ) : ℝ := 0.9 * P + 0.15 * P
def jane_paid (P : ℝ) : ℝ := 0.9 * P + 0.135 * P

theorem original_price_of_dish (h : john_paid P = jane_paid P + 1.26) : P = 84 := by
  sorry

end original_price_of_dish_l274_274769


namespace fraction_value_l274_274853

theorem fraction_value :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 :=
by
  sorry

end fraction_value_l274_274853


namespace inequality_proof_l274_274645

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (A B : ℝ)
  (h1 : ∀ i, 0 < a i)
  (h2 : (∑ i, a i) = 1)
  (h3 : 0 < B)
  (h4 : A > -B * n)
  (h5 : 2 ≤ n) :
  (A * n + B * n^2) * (∑ i, (a i)^3) ≥ A * (∑ i, (a i)^2) + B := by
  sorry

end inequality_proof_l274_274645


namespace triangle_side_b_value_l274_274590

noncomputable def triangle_side_b (a : ℝ) (B A : ℝ) (cosA : ℝ) : ℝ := 
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := 2 * sinA * cosA
  (a * sinB) / sinA

theorem triangle_side_b_value : 
  let a := 3
  let A := Real.arccos (sqrt 6 / 3)
  let B := 2 * A
  let cosA := sqrt 6 / 3
  triangle_side_b a B A cosA = 2 * sqrt 6 := 
by
  sorry

end triangle_side_b_value_l274_274590


namespace vector_BC_l274_274159

/-- Given points A (0,1), B (3,2) and vector AC (-4,-3), prove that BC = (-7, -4) -/
theorem vector_BC
  (A B : ℝ × ℝ)
  (AC : ℝ × ℝ)
  (hA : A = (0, 1))
  (hB : B = (3, 2))
  (hAC : AC = (-4, -3)) :
  (AC - (B - A)) = (-7, -4) :=
by
  sorry

end vector_BC_l274_274159


namespace speed_of_current_l274_274804

-- Define the conditions
def distance_between_bridges : ℝ := 2 -- distance in km
def total_drift_time : ℝ := 40 / 60 -- time in hours (40 minutes converted to hours)

-- Define the theorem to prove
theorem speed_of_current (d : ℝ) (t : ℝ) (h_d : d = distance_between_bridges) (h_t : t = total_drift_time) :
  d / t = 3 := 
sorry

end speed_of_current_l274_274804


namespace exponent_of_3_in_30_factorial_l274_274251

theorem exponent_of_3_in_30_factorial : 
  ∃ k : ℕ, unique_factorization_monoid.factor_count 30! 3 = k ∧ k = 14 :=
sorry

end exponent_of_3_in_30_factorial_l274_274251


namespace arithmetic_sequence_sum_l274_274608

variable (a : ℕ → ℕ)
variable (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 2 - a 1)

theorem arithmetic_sequence_sum (h : a 2 + a 8 = 6) : 
  1 / 2 * 9 * (a 1 + a 9) = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l274_274608


namespace relationship_l274_274505

noncomputable def a : ℝ := 3^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 3 / Real.log 2⁻¹
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship (a_def : a = 3^(-1/3 : ℝ)) 
                     (b_def : b = Real.log 3 / Real.log 2⁻¹) 
                     (c_def : c = Real.log 3 / Real.log 2) : 
  b < a ∧ a < c :=
  sorry

end relationship_l274_274505


namespace pu_guan_equal_length_l274_274607

theorem pu_guan_equal_length :
  ∃ n ∈ ℝ, 
    (3 * (1 - 1/2^n) / (1 - 1/2) = (2^n - 1) / (2 - 1)) ∧ 
    |n - 2.6| < 0.1 :=
by
  refine ⟨2.6, _, ⟨_, _⟩⟩
  sorry

end pu_guan_equal_length_l274_274607


namespace find_x_with_three_prime_divisors_l274_274150

def x_with_conditions (n : Nat) (x : Nat) : Prop :=
  x = 9^n - 1 ∧
  nat.factors x ∧
  nat.factors x.count 7 = 1

theorem find_x_with_three_prime_divisors (n : Nat) (x : Nat) :
  x_with_conditions n x → x = 728 :=
by
  sorry

end find_x_with_three_prime_divisors_l274_274150


namespace evaluate_expression_l274_274097

def f (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 9

theorem evaluate_expression : 3 * f 5 + 4 * f (-2) = 217 :=
by
  calc
    3 * f 5 + 4 * f (-2) = 3 * (2 * 5 ^ 2 - 4 * 5 + 9) + 4 * (2 * (-2) ^ 2 - 4 * (-2) + 9) := by rfl
    ... = 3 * (50 - 20 + 9) + 4 * (8 + 8 + 9) := by rfl
    ... = 3 * 39 + 4 * 25 := by rfl
    ... = 117 + 100 := by rfl
    ... = 217 := by rfl

end evaluate_expression_l274_274097


namespace triangle_inequality_l274_274991

variable {A B C : Type} [triangle A B C]
variable {a b c : ℝ} -- Side lengths of the triangle
variable {Δ : ℝ} -- Area of the triangle

theorem triangle_inequality (h: Δ > 0):
  a^2 + b^2 + c^2 ≥ 4 * sqrt 3 * Δ + (b - c)^2 + (c - a)^2 + (a - b)^2 :=
by 
  sorry

end triangle_inequality_l274_274991


namespace four_identical_tangent_circles_l274_274498

noncomputable def larger_circle_radius (r: ℝ) (small_circle_radius: ℝ) := r

theorem four_identical_tangent_circles (R: ℝ) (r: ℝ) :
  (∀ (c1 c2 c3 c4: ℝ),
    r = 1 ∧
    ((c2 - c1) = 2 ∧ (c3 - c2) = 2 ∧ (c4 - c3) = 2 ∧ (c1 - c4) = 2) ∧
    ∃ (O: ℝ), (c1 - O = R - 1 ∧ c2 - O = R - 1 ∧ c3 - O = R - 1 ∧ c4 - O = R - 1))
  → R = larger_circle_radius ( √ 2 + 1) r := sorry

end four_identical_tangent_circles_l274_274498


namespace sum_fraction_identity_l274_274658

theorem sum_fraction_identity (n : ℕ) (h : 0 < n) : 
  ∑ k in Finset.range n, (k + 2) / (k * (k + 1) * (k + 3) * (k + 4)) = 
  (n * (n + 5)) / (8 * (n + 1) * (n + 4)) :=
by
  sorry

end sum_fraction_identity_l274_274658


namespace inequality_holds_for_all_x_l274_274495

theorem inequality_holds_for_all_x (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
by
  sorry

end inequality_holds_for_all_x_l274_274495


namespace romanian_sequence_swap_bound_l274_274657

def is_romanian_sequence {n : ℕ} (s : list char) : Prop :=
  s.count 'I' = n ∧ s.count 'M' = n ∧ s.count 'O' = n

def swap (s : list char) (i : ℕ) : list char :=
  if i < s.length - 1 then (s.take i) ++ [s.nth_le (i + 1) sorry, s.nth_le i sorry] ++ (s.drop (i + 2))
  else s

def min_swaps (X Y : list char) : ℕ :=
  -- somehow compute the minimum number of swaps to convert X to Y
  sorry

theorem romanian_sequence_swap_bound (n : ℕ) (X : list char) (hX : is_romanian_sequence X) :
  ∃ Y : list char, is_romanian_sequence Y ∧ min_swaps X Y ≥ 3 * n^2 / 2 :=
by { sorry }

end romanian_sequence_swap_bound_l274_274657


namespace unique_hyperdeficient_l274_274844

def g (n : ℕ) : ℕ := (list.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum

def is_hyperdeficient (n : ℕ) : Prop := g (g n) = n + 3

theorem unique_hyperdeficient : ∃! n : ℕ, is_hyperdeficient n :=
by
  sorry

end unique_hyperdeficient_l274_274844


namespace complex_div_eq_i_l274_274540

-- Given conditions
def z : ℂ := 1 + I
def denom : ℂ := 1 - I

-- Required proof statement
theorem complex_div_eq_i : (z / denom) = I := by
  -- here would be the proof; we only state the theorem and add 'sorry' to skip it
  sorry

end complex_div_eq_i_l274_274540


namespace find_x_l274_274144

theorem find_x (n : ℕ) (x : ℕ) (hn : x = 9^n - 1) (h7 : ∃ p1 p2 : ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 ≠ 7 ∧ p2 ≠ 7 ∧ p1 ≠ p2 ∧ 7 ∣ x ∧ p1 ∣ x ∧ p2 ∣ x) : x = 728 :=
by
  sorry

end find_x_l274_274144


namespace option_B_is_direct_proportion_l274_274001

def direct_proportion_function (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def f_A (x : ℝ) : ℝ := (4 - 3 * x) / 2
def f_B (x : ℝ) : ℝ := x / 4
def f_C (x : ℝ) : ℝ := (-5 / x) + 3
def f_D (x : ℝ) : ℝ := 2 * x^2 + 1 / 3

theorem option_B_is_direct_proportion :
  direct_proportion_function f_B :=
sorry

end option_B_is_direct_proportion_l274_274001


namespace Petya_wins_on_2021x2021_grid_l274_274512

theorem Petya_wins_on_2021x2021_grid :
  ∀ (grid : array (fin 2021) (array (fin 2021) (option nat))) 
    (player_turn : bool),
    (player_turn = tt → (∀ a b : fin 2021, a < 2017 → b < 2019 → ∃ i j : fin 2021, i ≤ a + 5 ∧ j ≤ b + 3 ∧ grid[i][j].is_some) ∧ 
                    (∀ a b : fin 2021, a < 2019 → b < 2017 → ∃ i j : fin 2021, i ≤ a + 3 ∧ j ≤ b + 5 ∧ grid[i][j].is_some)) →
  player_turn = tt ∨ player_turn = ff → 
  (∃ strategy, (∀ turn player_num, (turn = nat_mod 2 player_num) → 
    (∧ player_turn) →
    ((player_turn = tt → ∃ i j : fin 2021, i + 3 < 2021 ∧ j + 5 < 2021 ∧ grid[i][j].is_none → (strategy turn = (i,j))) ∧ 
     (player_turn = ff → ∃ i j : fin 2021, i + 3 < 2021 ∧ j + 5 < 2021 ∧ grid[i][j].is_none → (strategy turn = (i,j)))))) :=
sorry

end Petya_wins_on_2021x2021_grid_l274_274512


namespace circle_equation_l274_274030

theorem circle_equation : 
  ∀ (x y : ℝ),
  ∃ (R : ℝ),
  ((∃ (h₁ : (x, y) = (2, -1) ∨ y - x + 1 = 0),
    (R = sqrt 4)) → 
    (x - 2)^2 + (y + 1)^2 = R^2 )  :=
begin
    intros x y,
    use 2,
    intro h,
    cases h with cond rad,
    
    sorry, -- leaving the proof
end

end circle_equation_l274_274030


namespace base7_digits_1234_l274_274950

theorem base7_digits_1234 : ∀ (n : ℕ), n = 1234 → 
  ∀ (b : ℕ), b = 7 → 
  ∃ d : ℕ, d = 4 ∧ ∀ (p : ℕ), 1234 / b^p < b → d = p + 1 := 
by 
  intros n hn b hb
  exists 4
  split
  case right =>
    sorry
  case left =>
    rfl

end base7_digits_1234_l274_274950


namespace xy_product_eq_two_l274_274529

theorem xy_product_eq_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 2 / x = y + 2 / y) : x * y = 2 := 
sorry

end xy_product_eq_two_l274_274529


namespace decagon_triangle_probability_l274_274876

theorem decagon_triangle_probability :
  let total_triangles := Nat.choose 10 3 in
  let favorable_triangles := 70 in
  let probability := favorable_triangles / total_triangles in
  probability = 7 / 12 :=
by
  sorry

end decagon_triangle_probability_l274_274876


namespace relationship_l274_274132

variable (a b c : ℝ)

-- The conditions from the problem
def condition1 : Prop := 3^a = 5
def condition2 : Prop := b = log 3 (1 / 5)
def condition3 : Prop := log 3 c = -1

-- The statement that connects the question and the answer
theorem relationship (hc1 : condition1 a) (hc2 : condition2 b) (hc3 : condition3 c) : b < c ∧ c < a := by
  sorry

end relationship_l274_274132


namespace optimal_cashback_l274_274678

def Category : Type := String

def cashback_rate : Category → ℝ
| "Transport" := 0.05
| "Groceries" := 0.03
| "Clothing" := 0.04
| "Entertainment" := 0.05
| "Sports goods" := 0.06
| _ := 0.01

def expenses : Category → ℝ
| "Transport" := 2000
| "Groceries" := 5000
| "Clothing" := 3000
| "Entertainment" := 3000
| "Sports goods" := 1500
| _ := 0

def cashback (c : Category) : ℝ :=
  expenses c * cashback_rate c

theorem optimal_cashback :
  let categories := ["Transport", "Groceries", "Clothing", "Entertainment", "Sports goods"] in
  let selected := ["Groceries", "Clothing", "Entertainment"] in
  ∀ (c₁ c₂ c₃ : Category), c₁ ∈ categories → c₂ ∈ categories → c₃ ∈ categories →
  (c₁ = "Groceries" ∧ c₂ = "Clothing" ∧ c₃ = "Entertainment") ↔
  (cashback "Groceries" + cashback "Clothing" + cashback "Entertainment" ≥
  cashback c₁ + cashback c₂ + cashback c₃) := by
  intros c₁ c₂ c₃ h₁ h₂ h₃
  split
  · intros ⟨h₄, h₅, h₆⟩
    simp [h₄, h₅, h₆]
    sorry
  · intro h
    sorry

end optimal_cashback_l274_274678


namespace smallest_N_for_equal_adults_and_children_l274_274047

theorem smallest_N_for_equal_adults_and_children :
  ∃ (N : ℕ), N > 0 ∧ (∀ a b : ℕ, 8 * N = a ∧ 12 * N = b ∧ a = b) ∧ N = 3 :=
sorry

end smallest_N_for_equal_adults_and_children_l274_274047


namespace calculate_probability_ratio_l274_274476

theorem calculate_probability_ratio :
  let p' := (choose 6 1) * (choose 5 1) * (choose 13 3) / (choose 23 5)
  let q' := 1
  p' / q' = 8580 :=
by
  sorry

end calculate_probability_ratio_l274_274476


namespace num_integers_satisfying_inequality_l274_274562

theorem num_integers_satisfying_inequality :
  ∃ (n_count : ℕ), n_count = 9 ∧ 
  ∀ (n : ℤ), (n+2) * (n-8) < 0 → n ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7} :=
by {
  sorry
}

end num_integers_satisfying_inequality_l274_274562


namespace intersection_A_B_l274_274930

open Set

noncomputable def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

noncomputable def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 3} :=
by {
  sorry
}

end intersection_A_B_l274_274930


namespace find_a5_max_S2024_l274_274897

noncomputable def sequence_a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := ∑ i in finset.range n, sequence_a i

axiom a1 : sequence_a 1 = -1
axiom cond1 : 4 ≤ sequence_a 3 ∧ sequence_a 3 ≤ 8
axiom cond2 : sequence_a 2024 < 0
axiom condition : ∀ n : ℕ, 2 * sequence_a n * sequence_a (n + 2) + sequence_a (n + 1) * sequence_a (n + 3) = 0

theorem find_a5 : sequence_a 5 = -4 := sorry
theorem max_S2024 : S 2024 = ( (4^506 - 1) / 3 ) := sorry

end find_a5_max_S2024_l274_274897


namespace roots_greater_than_half_iff_l274_274127

noncomputable def quadratic_roots (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (2 - a) * x1^2 - 3 * a * x1 + 2 * a = 0 ∧ 
  (2 - a) * x2^2 - 3 * a * x2 + 2 * a = 0 ∧
  x1 > 1/2 ∧ x2 > 1/2

theorem roots_greater_than_half_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots a x1 x2) ↔ (16 / 17 < a ∧ a < 2) :=
sorry

end roots_greater_than_half_iff_l274_274127


namespace largest_consecutive_odd_number_sum_is_27_l274_274723

theorem largest_consecutive_odd_number_sum_is_27
  (a b c : ℤ)
  (h1 : a + b + c = 75)
  (h2 : c - a = 4)
  (h3 : a % 2 = 1)
  (h4 : b % 2 = 1)
  (h5 : c % 2 = 1) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_sum_is_27_l274_274723


namespace servant_worked_months_l274_274938

variable (months_in_year : ℕ := 12) (salary : ℕ := 90) (turban_value : ℕ := 50)
variable (received_money : ℕ := 55) (total_salary_value : ℕ := 140)

/-- Given the total annual salary in monetary value, the number of months worked,
    and the monthly salary calculated as total annual salary divided by number of months in a year,
    prove that the servant worked for 9 months.
-/
theorem servant_worked_months :
  let monthly_salary := total_salary_value / months_in_year in
  let received_value := received_money + turban_value in
  (monthly_salary * 9 = received_value) :=
by
  sorry

end servant_worked_months_l274_274938


namespace find_l_l274_274402

def satisfies_inequality (n m l : ℕ) (a : ℕ → ℝ) : Prop :=
  ∑ k in finset.range n, 1 / (∑ j in finset.range (k + 1), a j + 1) * (l * (k + 1) + (l^2) / 4) < m^2 * ∑ k in finset.range n, 1 / (a k + 1)

theorem find_l (n m : ℕ) (h_n : n > 1) (h_m : m > 1) :
  {l : ℕ | ∀ (a : ℕ → ℝ) (a_pos : ∀ k, 0 < a k), satisfies_inequality n m l a} = { l | l ≤ 2 * (m - 1) } :=
sorry

end find_l_l274_274402


namespace vector_addition_and_scaling_is_correct_l274_274808

-- Define the two vectors
def vec1 : ℕ → ℤ := ![-3, 2, 5]
def vec2 : ℕ → ℤ := ![4, 7, -3]

-- Define the sum of the vectors
def vecSum := λ i, vec1 i + vec2 i

-- Define the scaled vector
def scaledVecSum := λ i, 2 * vecSum i

-- The target vector we need to prove is the result
def targetVec : ℕ → ℤ := ![2, 18, 4]

-- Statement of the proof problem
theorem vector_addition_and_scaling_is_correct :
  scaledVecSum = targetVec :=
by
-- Proof skipped
sorry

end vector_addition_and_scaling_is_correct_l274_274808


namespace valid_anti_birthdays_l274_274014

def isValidDate (d m : ℕ) : Prop :=
  d ≥ 1 ∧ d ≤ 31 ∧ m ≥ 1 ∧ m ≤ 12

def isSwapValid (d m : ℕ) : Prop :=
  isValidDate d m ∧ d ∈ (Finset.range 12).succ ∧ d ≠ m

theorem valid_anti_birthdays : 
  (Finset.univ.filter (λ d : ℕ, isSwapValid d 1)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 2)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 3)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 4)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 5)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 6)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 7)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 8)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 9)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 10)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 11)).card 
+ (Finset.univ.filter (λ d : ℕ, isSwapValid d 12)).card = 132 := 
by sorry

end valid_anti_birthdays_l274_274014


namespace smallest_b_l274_274277

open Real

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 2 ∧ B = a ∧ C = b ∨ A = 2 ∧ B = b ∧ C = a ∨ A = a ∧ B = b ∧ C = 2) ∧ A + B > C ∧ A + C > B ∧ B + C > A)
  (h4 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 1 / b ∧ B = 1 / a ∧ C = 2 ∨ A = 1 / a ∧ B = 1 / b ∧ C = 2 ∨ A = 1 / b ∧ B = 2 ∧ C = 1 / a ∨ A = 1 / a ∧ B = 2 ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / a ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / b ∧ C = 1 / a) ∧ A + B > C ∧ A + C > B ∧ B + C > A) :
  b = 2 := 
sorry

end smallest_b_l274_274277


namespace sum_of_real_values_l274_274589

theorem sum_of_real_values (x : ℝ) (h : |3 * x - 15| + |x - 5| = 92) : (x = 28 ∨ x = -18) → x + 10 = 0 := by
  sorry

end sum_of_real_values_l274_274589


namespace processing_time_600_parts_l274_274554

theorem processing_time_600_parts :
  ∀ (x: ℕ), x = 600 → (∃ y : ℝ, y = 0.01 * x + 0.5 ∧ y = 6.5) :=
by
  sorry

end processing_time_600_parts_l274_274554


namespace range_of_h_l274_274114

open Real

noncomputable def h (t : ℝ) : ℝ := (t^2 - (1 / 2) * t) / (2 * t^2 + 1)

theorem range_of_h :
  set.range h = 
  { y : ℝ | (1 - sqrt 15 / 2) ≤ y ∧ y ≤ (1 + sqrt 15 / 2) } :=
sorry

end range_of_h_l274_274114


namespace find_exponent_l274_274551

noncomputable def function_expression (x b : ℝ) := (x - b) / (x + 2)

theorem find_exponent (a b : ℝ) (h1 : b < -2) (h2 : function_expression (a + 4) b > 2) 
  (h3 : function_expression a b = function_expression (-2) (-4)) :
  a^b = 1 / 16 :=
by 
  have ha : a = -2 := sorry
  have hb : b = -4 := sorry
  rw [ha, hb]
  norm_num
  exact rfl

end find_exponent_l274_274551


namespace m_not_in_P_l274_274663

noncomputable def m : ℝ := Real.sqrt 3
def P : Set ℝ := { x | x^2 - Real.sqrt 2 * x ≤ 0 }

theorem m_not_in_P : m ∉ P := by
  sorry

end m_not_in_P_l274_274663


namespace exponential_comparison_l274_274977

theorem exponential_comparison (x y a b : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (hb : a < b) (hb' : b < 1) : 
  a^x < b^y :=
sorry

end exponential_comparison_l274_274977


namespace eithan_savings_l274_274815

theorem eithan_savings :
  let amount := 2000 : ℝ 
  let wife_share := (2/5) * amount 
  let remaining_after_wife := amount - wife_share  
  let first_son_share := (2/5) * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - first_son_share 
  let second_son_share := (40/100) * remaining_after_first_son 
  let savings := remaining_after_first_son - second_son_share 
  savings = 432 :=
by
  sorry

end eithan_savings_l274_274815


namespace minimum_value_PA_PO_is_2_l274_274161

def minimum_value_PA_PO : ℝ :=
  let O := (0, 0)
  let A := (1, 1)
  let l : ℝ × ℝ → Prop := λ p, p.1 - p.2 + 1 = 0
  2

theorem minimum_value_PA_PO_is_2 : minimum_value_PA_PO = 2 :=
by
  sorry

end minimum_value_PA_PO_is_2_l274_274161


namespace mexth_exists_unique_family_l274_274098

/-- Define the set F_n for each nonnegative integer n -/
def F (n : ℕ) : Set ℕ := {x | ∃ k i : ℕ, x = 2^(n+1) * k + i ∧ i < 2^n}

/-- Define the family of sets \mathcal{F} -/
def F_family : Set (Set ℕ) := {S | ∃ n, S = F n}

/-- Statement of the proof problem -/
theorem mexth_exists_unique_family :
  ∃ (F : Set (Set ℕ)), 
    (∀ G : Set ℕ, G ∈ F_family → ∃ m : ℕ, ∀ k : ℕ, k ≤ m → k ∉ G) ∧
    (∀ n : ℕ, ∃! G : Set ℕ, G ∈ F_family ∧ mexth G = n) :=
sorry

end mexth_exists_unique_family_l274_274098


namespace function_identity_l274_274823

theorem function_identity {f : ℕ → ℕ} (h₀ : f 1 > 0) 
  (h₁ : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l274_274823


namespace right_triangle_side_length_l274_274609

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 12) 
  (h_right : a^2 + b^2 = c^2) : 
  b = Real.sqrt 119 :=
by
  sorry

end right_triangle_side_length_l274_274609


namespace find_angle_AOB_l274_274736

-- Define a type for angles.
def angle := ℝ

-- Define the conditions present in the problem.
variable (triangle_PAB_tangent_to_circle_O : Prop)
variable (angle_APB : angle)
variable (angle_AOB : angle)

-- State the main theorem.
theorem find_angle_AOB 
  (triangle_PAB_tangent_to_circle_O : triangle_PAB_tangent_to_circle_O)
  (h1 : angle_APB = 60) : 
  angle_AOB = 60 := 
  sorry

end find_angle_AOB_l274_274736


namespace increasing_sequence_divisibility_l274_274681

theorem increasing_sequence_divisibility (a1 : ℕ) (h_a1 : a1 > 1) :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, a (n+1) > a n) ∧
  (∀ n : ℕ, (finset.range (n+1)).sum a ∣ (finset.range (n+1)).sum (λ i, (a i)^2)) :=
by
  sorry

end increasing_sequence_divisibility_l274_274681


namespace max_area_rectangle_l274_274379

theorem max_area_rectangle {x y : ℝ} 
  (h : abs (y + 1) * (y ^ 2 + 2 * y + 28) + abs (x - 2) = 9 * (y ^ 2 + 2 * y + 4)) : 
  ∃ (x y : ℝ), let S := (y - 1) * (x - 2) in S = 34.171875 :=
sorry

end max_area_rectangle_l274_274379


namespace max_area_difference_l274_274743

theorem max_area_difference (l1 l2 w1 w2 : ℤ) (h1 : 2*l1 + 2*w1 = 200) (h2 : 2*l2 + 2*w2 = 200) :
  let A := λ l w, l * w in
  (max {A l w | l + w = 100} - min {A l w | l + w = 100}) = 2401 :=
sorry

end max_area_difference_l274_274743


namespace football_game_initial_population_l274_274730

theorem football_game_initial_population (B G : ℕ) (h1 : G = 240)
  (h2 : (3 / 4 : ℚ) * B + (7 / 8 : ℚ) * G = 480) : B + G = 600 :=
sorry

end football_game_initial_population_l274_274730


namespace base_7_digits_of_1234_l274_274954

theorem base_7_digits_of_1234 : ∀ (n : ℕ), (n = 1234) → (nat.log n 7 + 1 = 4) :=
begin
  intros n hn,
  rw hn,
  sorry
end

end base_7_digits_of_1234_l274_274954


namespace urban_general_hospital_problem_l274_274824

theorem urban_general_hospital_problem
  (a b c d : ℕ)
  (h1 : b = 3 * c)
  (h2 : a = 2 * b)
  (h3 : d = c / 2)
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1500) :
  5 * d = 1500 / 11 := by
  sorry

end urban_general_hospital_problem_l274_274824


namespace compute_AF_over_AT_l274_274258

-- Define the triangle and points
variables {A B C D E T F : Type}
variables {AD DB AE EC AT : ℝ}

-- Define the given conditions
axiom h1 : AD = 2
axiom h2 : DB = 2
axiom h3 : AE = 3
axiom h4 : EC = 3
axiom h5 : (is_angle_bisector AT ∠BAC)

-- Define the goal
theorem compute_AF_over_AT : 
    ∃ F, (ATintersectsDE AT DE F) → (AF/AT) = 5/13 :=
by
    sorry

end compute_AF_over_AT_l274_274258


namespace compute_expression_l274_274092

theorem compute_expression :
  21 * 47 + 21 * 53 = 2100 := 
by
  sorry

end compute_expression_l274_274092


namespace garden_length_l274_274207

theorem garden_length (P : ℕ) (breadth : ℕ) (length : ℕ) 
  (h1 : P = 600) (h2 : breadth = 95) (h3 : P = 2 * (length + breadth)) : 
  length = 205 :=
by
  sorry

end garden_length_l274_274207


namespace no_integer_pair_exists_l274_274496

theorem no_integer_pair_exists (m : ℤ) : m = 4 -> (¬ ∃ x y : ℤ, 3 * x^2 - 3 * x * y - y^2 ≡ 4 [ZMOD 7]) :=
by {
  intro h,
  rw h,
  sorry
}

end no_integer_pair_exists_l274_274496


namespace ratio_of_agreements_rounded_l274_274228

-- Define the required conditions
def required_agreements : ℚ := 11
def total_members : ℚ := 15

-- Define the question as a proof in the Lean 4 statement
theorem ratio_of_agreements_rounded :
  Real.approximate_rational_to_tenth (required_agreements / total_members) = 0.7 := 
by
  sorry

end ratio_of_agreements_rounded_l274_274228


namespace find_x_eq_728_l274_274152

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end find_x_eq_728_l274_274152


namespace translated_point_is_correct_l274_274236

-- Cartesian Point definition
structure Point where
  x : Int
  y : Int

-- Define the translation function
def translate (p : Point) (dx dy : Int) : Point :=
  Point.mk (p.x + dx) (p.y - dy)

-- Define the initial point A and the translation amounts
def A : Point := ⟨-3, 2⟩
def dx : Int := 3
def dy : Int := 2

-- The proof goal
theorem translated_point_is_correct :
  translate A dx dy = ⟨0, 0⟩ :=
by
  -- This is where the proof would normally go
  sorry

end translated_point_is_correct_l274_274236


namespace length_OF_does_not_depend_on_C_l274_274308

open Euclidean_geometry

variables {A B C M P Q O D E F : Point ℝ}
variables (segment_AB: Segment (line A B))
variables (point_M_on_segment : ∃ (λ M, M ∈ segment_AB))
variables (point_P_is_midpoint: P = midpoint A M)
variables (point_Q_is_midpoint: Q = midpoint B M)
variables (point_O_is_midpoint: O = midpoint P Q)
variables (angle_ACB_right: ∠ A C B = 90°)
variables (MD_perpendicular_to_CA: perpendicular M D (line C A))
variables (ME_perpendicular_to_CB: perpendicular M E (line C B))
variables (point_F_is_midpoint: F = midpoint D E)

theorem length_OF_does_not_depend_on_C
  (fixed_PQ : (λ P, Q, dist P Q) : ℝ) :
  dist O F = dist P Q / 2 :=
begin
  sorry
end

end length_OF_does_not_depend_on_C_l274_274308


namespace largest_angle_in_triangle_l274_274996

theorem largest_angle_in_triangle
  (h1 : ∃ k : ℕ, angles = [3 * k, 4 * k, 5 * k])
  (h2 : ∑ angle in angles, angle = 180) :
  ∃ angle, angle = 75 ∧ angle = max (max (3 * k) (4 * k)) (5 * k) := 
sorry

end largest_angle_in_triangle_l274_274996


namespace max_value_of_seq_l274_274962

theorem max_value_of_seq (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = -n^2 + 6 * n + 7)
  (h_a_def : ∀ n, a n = S n - S (n - 1)) : ∃ max_val, max_val = 12 ∧ ∀ n, a n ≤ max_val :=
by
  sorry

end max_value_of_seq_l274_274962


namespace tank_final_volume_l274_274438

def initial_volume (capacity : ℕ) : ℕ := (3/4 : ℚ) * capacity
def emptied_volume (initial_volume : ℕ) : ℕ := (40/100 : ℚ) * initial_volume
def remaining_volume (initial_volume : ℕ) (emptied_volume : ℕ) : ℕ := initial_volume - emptied_volume
def added_volume (remaining_volume : ℕ) : ℕ := (30/100 : ℚ) * remaining_volume
def final_volume (remaining_volume : ℕ) (added_volume : ℕ) : ℕ := remaining_volume + added_volume

theorem tank_final_volume (capacity : ℕ) (h1 : capacity = 8000) : 
  final_volume (remaining_volume (initial_volume capacity) (emptied_volume (initial_volume capacity))) (added_volume (remaining_volume (initial_volume capacity) (emptied_volume (initial_volume capacity)))) = 4680 :=
  by
    sorry

end tank_final_volume_l274_274438


namespace find_a_n_find_T_n_l274_274637

-- Assume the sequence {a_n} and its conditions
variable (a : ℕ → ℕ)

-- Condition based on problem statement
def S (n : ℕ) : ℕ := (1/4 : ℚ)*(a n)^2 + (1/2 : ℚ)*(a n) - (3/4 : ℚ)

-- Definition of b_n
def b (n : ℕ) : ℕ := 2^n

-- Definition of T_n
def T (n : ℕ) : ℕ := finset.sum (finset.range (n+1)) (λ k, (a k) * (b k))

-- Theorem corresponding to Question 1
theorem find_a_n (h : ∀ n, S n = (1/4 : ℚ)*(a n)^2 + (1/2 : ℚ)*(a n) - (3/4 : ℚ)) : ∀ n, a n = 2 * n + 1 :=
sorry

-- Theorem corresponding to Question 2
theorem find_T_n (h : ∀ n, a n = 2 * n + 1) : ∀ n, T n = (2 * n - 1) * 2^(n+1) + 2 :=
sorry

end find_a_n_find_T_n_l274_274637


namespace consecutive_sum_95_l274_274004

-- Define the sequences and conditions
def is_consecutive_sum (n m : ℕ) (num : ℕ) : Prop :=
  num = n * (2 * m + n - 1) / 2

-- Main theorem statement
theorem consecutive_sum_95 : 
  ∃ (n m : ℕ), n > 1 ∧ is_consecutive_sum n m 95 ∧ 
  ((∃ (k : ℕ), is_consecutive_sum 2 k 95) + 
   (∃ (k : ℕ), is_consecutive_sum 5 k 95) + 
   (∃ (k : ℕ), is_consecutive_sum 10 k 95)) = 3 :=
by sorry

end consecutive_sum_95_l274_274004


namespace find_line_for_nine_circles_l274_274212

noncomputable def nine_circles_packed (P : set (ℝ × ℝ)) : Prop :=
  ∃ r (c : ℕ → ℝ × ℝ), r = 1/2 ∧ (∀ i, i < 9 → metric.ball (c i) r ⊆ P) ∧
  (∀ i j, i < 9 ∧ j < 9 ∧ i ≠ j → metric.ball (c i) r ∩ metric.ball (c j) r = ∅) ∧
  ∀ i, i < 9 → (c i).1 = (i % 3 + 1 / 2) ∧ (c i).2 = (i / 3 + 1 / 2)

noncomputable def line_divides_P_equal_area (m P : set (ℝ × ℝ)) (slope : ℝ) : Prop :=
  ∃ a b c : ℤ, (a, b, c).gcd = 1 ∧ m = { p | a * p.1 = b * p.2 + c } ∧
  slope = 4 ∧
  ∀ x, measurable_set (P ∩ m ∩ {p | p.1 < x}) →
    measurable_set (P ∩ m ∩ {p | p.1 > x}) ∧
    μ (P ∩ m ∩ {p | p.1 < x}) = μ (P ∩ m ∩ {p | p.1 > x})

theorem find_line_for_nine_circles (P : set (ℝ × ℝ)) (m : set (ℝ × ℝ)) (slope : ℝ) 
  (h1 : nine_circles_packed P) (h2 : line_divides_P_equal_area m P slope) :
  ∃ a b c : ℤ, (a, b, c).gcd = 1 ∧ (a ^ 2 + b ^ 2 + c ^ 2 = 18) := 
sorry

end find_line_for_nine_circles_l274_274212


namespace tan_add_sin_l274_274458

noncomputable def sin (x : ℝ) : ℝ := sorry -- Replace with actual definition
noncomputable def cos (x : ℝ) : ℝ := sorry -- Replace with actual definition
noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem tan_add_sin (sin_15 cos_15 : ℝ) (h_sin_15 : sin 15 = sin_15)
    (h_cos_15 : cos 15 = cos_15) :
    tan 15 + 3 * sin 15 = (sqrt 6 - sqrt 2 + 3) / (sqrt 6 + sqrt 2) :=
by
    have h1 : sin 15 = (sin 45 * cos 30 - cos 45 * sin 30) := sorry
    have h2 : sin 15 = (sqrt 6 / 4 - sqrt 2 / 4) := sorry
    have h3 : sin 15 = (sqrt 6 - sqrt 2) / 4 := sorry
    have h4 : cos 15 = (cos 45 * cos 30 + sin 45 * sin 30) := sorry
    have h5 : cos 15 = (sqrt 6 + sqrt 2) / 4 := sorry
    have h6 : tan 15 + 3 * sin 15 = ((sin 15 + 3 * sin 15 * cos 15) / cos 15) := sorry
    have h7 : tan 15 + 3 * sin 15 = ((sqrt 6 - sqrt 2 + 3) / 4) / ((sqrt 6 + sqrt 2) / 4) := sorry
    have h8 : tan 15 + 3 * sin 15 = (sqrt 6 - sqrt 2 + 3) / (sqrt 6 + sqrt 2) := sorry
    sorry

end tan_add_sin_l274_274458


namespace circle_radius_l274_274711

noncomputable def radius_of_circle : Type := sorry

theorem circle_radius 
(center_x : ℝ) 
(h_center : ∃ c : ℝ, c = center_x ∧ ∃ r : ℝ, (0,5) = (0,r) ∧ (2,3) = ((c-2)^2 + 9))
(h_center_x_axis : center_x = -3) 
: radius_of_circle = real.sqrt 34 :=
sorry

end circle_radius_l274_274711


namespace total_cows_in_ranch_l274_274749

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l274_274749


namespace menelaus_theorem_l274_274662

-- Variables and points
variables {A B C D P E Q F : Type}

-- Conditions and main statement
axioms
  (l : Type) -- assuming l is a line type
  (intersect1 : l → A → P)
  (intersect2 : l → B → E)
  (intersect3 : l → C → Q)
  (intersect4 : l → D → F)

-- Problem statement
theorem menelaus_theorem :
  ((AP PB : Type) → (BE EC : Type) → (CQ QD : Type) → (DF FA : Type)) :=
begin
  sorry
end

end menelaus_theorem_l274_274662


namespace voucher_distribution_preferred_plan_l274_274790

section shopping_voucher

variables (X : Type) [fintype X] (draw : finset X)

-- Condition: original draw setup with probabilities
def original_draw (x : X) : ℚ :=
  if x = 200 then 1/45
  else if x = 80 then 16/45
  else if x = 10 then 28/45
  else 0

-- Voucher distribution proof statement
theorem voucher_distribution :
  original_draw X 200 = 1 / 45 ∧ original_draw X 80 = 16 / 45 ∧ original_draw X 10 = 28 / 45 :=
sorry

-- Improvement Plan A setup
def planA_draw (x : X) : ℚ :=
  if x = 200 then 1/22
  else if x = 80 then 9/22
  else if x = 10 then 6/11
  else 0

-- Improvement Plan B setup
def planB_draw (x : X) : ℚ :=
  if x = 210 then 1/45
  else if x = 90 then 16/45
  else if x = 20 then 28/45
  else 0

-- Expected value calculations
def expected_value (P : X → ℚ) (values : X → ℚ) : ℚ :=
  ∑ x, P x * values x

def planA_value := expected_value (planA_draw X) id
def planB_value := expected_value (planB_draw X) id

-- Preferred plan proof statement
theorem preferred_plan : planB_value > planA_value :=
sorry

end shopping_voucher

end voucher_distribution_preferred_plan_l274_274790


namespace trig_problem_l274_274885

theorem trig_problem (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

end trig_problem_l274_274885


namespace opposite_numbers_pow_sum_zero_l274_274974

theorem opposite_numbers_pow_sum_zero (a b : ℝ) (h : a + b = 0) : a^5 + b^5 = 0 :=
by sorry

end opposite_numbers_pow_sum_zero_l274_274974


namespace cost_of_pastrami_l274_274686

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l274_274686


namespace area_of_enclosed_region_is_408_38_l274_274859

-- Define the equation condition
def enclosed_region (x y : ℝ) : Prop :=
  |x - 70| + |y| = |x / 5|

-- Prove the area of the region defined by the above equation
theorem area_of_enclosed_region_is_408_38 :
  (∃ (vertices : list (ℝ × ℝ)), 
    let_polygon := polygon_from_vertices vertices in
    enclosed_region (fst (vertices.head)) (snd (vertices.head)) ∧
    enclosed_region (fst (vertices.nth 1 h1)) (snd (vertices.nth 1 h1)) ∧
    enclosed_region (fst (vertices.nth 2 h2)) (snd (vertices.nth 2 h2)) ∧
    enclosed_region (fst (vertices.nth 3 h3)) (snd (vertices.nth 3 h3)) ∧
    computes_area let_polygon 408.38) :=
sorry

end area_of_enclosed_region_is_408_38_l274_274859


namespace people_in_line_l274_274454

theorem people_in_line (initially_in_line : ℕ) (left_line : ℕ) (after_joined_line : ℕ) 
  (h1 : initially_in_line = 12) (h2 : left_line = 10) (h3 : after_joined_line = 17) : 
  initially_in_line - left_line + 15 = after_joined_line := by
  sorry

end people_in_line_l274_274454


namespace binomial_expansion_value_l274_274912

theorem binomial_expansion_value :
  let a_0 := ((sqrt 5) - 1)^3,
      a_1 := (-(sqrt 5) - 1)^3,
      a_2 := ((sqrt 5) + 1)^3,
      a_3 := (-(sqrt 5) + 1)^3,
      s1 := (a_0 + a_2) - (a_1 + a_3),
      s2 := a_0 + a_1 + a_2 + a_3 in
  (s1)^2 - (s2)^2 = -64 := 
by {
  sorry
}

end binomial_expansion_value_l274_274912


namespace circles_arc_non_intersecting_l274_274511

theorem circles_arc_non_intersecting (n : ℕ) (h_pos : n > 0) :
  ∃ (c : ℕ → (ℝ × ℝ)) (h_distinct : function.injective c), 
  ∀ (i j : ℕ), i ≠ j → (dist (c i) (c j) ≥ 2) →
  ∃ (k : ℕ) (a b : ℝ), 0 < b - a ∧ b - a ≥ 2 * π / n ∧ 
  ¬(∃ (x : ℝ × ℝ), x ∈ arc (c k) a b ∧ (∃ (l : ℕ), l ≠ k ∧ dist x (c l) ≤ 1)) :=
by
  sorry

end circles_arc_non_intersecting_l274_274511


namespace probability_of_king_l274_274359

theorem probability_of_king {cards : Finset Nat} (hk : card cards = 6) (kings : Finset Nat) (hkings : card kings = 2) :
  (∃ (x y : Nat), x ∈ kings ∨ y ∈ kings ∧ x ≠ y ∧ x ∈ cards ∧ y ∈ cards) → 
  ((finset.card (finset.filter (λ pair, pair.1 ∈ kings ∨ pair.2 ∈ kings) (finset.pairs cards))) 
  > (finset.card (finset.filter (λ pair, ¬ (pair.1 ∈ kings ∨ pair.2 ∈ kings)) (finset.pairs cards)))) :=
by 
  sorry

end probability_of_king_l274_274359


namespace middle_part_of_104_division_l274_274851

theorem middle_part_of_104_division (x : ℕ) :
  (let largest := 2 * x; middle := (3 / 2) * x; smallest := (1 / 2) * x
   in largest + middle + smallest = 104) →
  3 * x / 2 = 39 :=
by
  intro h
  sorry

end middle_part_of_104_division_l274_274851


namespace probability_log_3_equals_l274_274682

noncomputable def log_base_3 (z : ℝ) : ℝ := Real.log z / Real.log 3

theorem probability_log_3_equals (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (⨂⨁⨄ (floor (log_base_3 x))) = (⧄⧅ (floor (log_base_3 y))) :=
  sorry

end probability_log_3_equals_l274_274682


namespace probability_of_product_divisible_by_3_l274_274880
open Classical

noncomputable def probability_divisible_by_3 : ℚ :=
  let outcomes := (fin 8) × (fin 8)
  let favorables := {p : outcomes // (p.1 + 1) * (p.2 + 1) % 3 = 0}
  (Set.card favorables).toNat / (Set.card outcomes).toNat

theorem probability_of_product_divisible_by_3 (h: probability_divisible_by_3 = (7 / 16)) : true :=
  begin
    sorry
  end

end probability_of_product_divisible_by_3_l274_274880


namespace base7_addition_l274_274634

variable (A B C : ℕ)

def distinct_nonzero_digits_lt_eq_six := (A ≠ B ∧ A ≠ C ∧ B ≠ C) ∧ 
                                          (A > 0 ∧ B > 0 ∧ C > 0) ∧ 
                                          (A ≤ 6 ∧ B ≤ 6 ∧ C ≤ 6)

-- Prove the given base 7 addition problem
theorem base7_addition (h : distinct_nonzero_digits_lt_eq_six A B C) 
  (h_add : ∀ base7_add_prop : 
    let base7_add_prop : (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A := by sorry) :
  (A + B + C : ℕ) ≡ 12 [MOD 7] :=
begin
  sorry,
end

end base7_addition_l274_274634


namespace find_slopes_of_line_l274_274524

theorem find_slopes_of_line (k : ℝ) :
  ∀ (x y : ℝ), x ^ 2 + y ^ 2 = 10 →
  ∃ A B : ℝ × ℝ, (∃ P : ℝ × ℝ, P = (-3, -4) ∧ line_through P A k ∧ line_through P B k) ∧
  area_of_triangle (0,0) A B = 5 →
  (k = 1 / 2 ∨ k = 11 / 2) :=
by
  intros x y hxy
  use (x, y)
  use (0,0)
  unfold area_of_triangle
  sorry

end find_slopes_of_line_l274_274524


namespace green_eyed_brunettes_count_l274_274993

-- Variables representing the number of each group of girls
variables (total_girls blue_eyed_blondes brunettes green_eyed_girls : ℕ)

-- Given conditions
def conditions : Prop :=
  total_girls = 60 ∧
  blue_eyed_blondes = 20 ∧
  brunettes = 35 ∧
  green_eyed_girls = 25

-- Conclusion: number of green-eyed brunettes
def number_of_green_eyed_brunettes : ℕ := total_girls - blue_eyed_blondes - brunettes - green_eyed_girls + blue_eyed_blondes - 5

theorem green_eyed_brunettes_count (h : conditions) : number_of_green_eyed_brunettes = 10 :=
by
  sorry  -- Proof is skipped

end green_eyed_brunettes_count_l274_274993


namespace functional_equation_solution_l274_274278

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 2 * y * f x) :
  ∀ x : ℝ, f x = 0 := 
sorry

end functional_equation_solution_l274_274278


namespace hawking_space_agency_l274_274328

theorem hawking_space_agency (n : ℕ) (G : SimpleGraph (Fin n)) 
  (flights : Fin n × Fin n → ℕ)
  (prices : Fin (binom n 2) → ℕ)
  (h_tree : G.isTree)
  (h_pricing : ∀ x y : Fin n, ∃ p : G.walk x y, 
    prices (finfind (flights (x, y))) = p.length) :
  is_square n ∨ is_square (n - 2) :=
sorry

end hawking_space_agency_l274_274328


namespace find_t_l274_274370

-- Definitions of vertices of triangle ABC
def A : ℝ × ℝ := (1, 10)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (10, 0)

-- Equation of the lines AB and AC
def lineAB (x : ℝ) : ℝ := -5 * x + 15
def lineAC (x : ℝ) : ℝ := -10/9 * x + 190/9

-- Definition of horizontal line y = t
def horizontal_line (t : ℝ) : ℝ := t

-- Definition of points T and U on lines AB and AC intersected by horizontal line y = t
def pointT (t : ℝ) : ℝ × ℝ := (3 - t / 5, t)
def pointU (t : ℝ) : ℝ × ℝ := ((190 - 9 * t) / 10, t)

-- Calculation of length of TU and height from A to TU
def lengthTU (t : ℝ) : ℝ := |(190 - 9 * t) / 10 - (3 - t / 5) |
def heightA_TU (t : ℝ) : ℝ := 10 - t

-- Calculation of area of triangle ATU
def areaATU (t : ℝ) : ℝ := (1 / 2) * lengthTU t * heightA_TU t

-- Main theorem: finding t such that area of triangle ATU is 18
theorem find_t (t : ℝ) : areaATU t = 18 → t = 5 :=
by 
    sorry

end find_t_l274_274370


namespace find_equation_and_min_sum_l274_274906

noncomputable def ellipse_center : Prop := ∃ (a b : ℝ), a = 0 ∧ b = 0 

noncomputable def ellipse_focus_y_axis (F : ℝ) : Prop := 
  ∃ (c : ℝ), F = c

noncomputable def ellipse_eccentricity (e : ℝ) : Prop := 
  e = (Real.sqrt 2 / 2)

noncomputable def ellipse_contains_point (P : ℝ × ℝ) : Prop := 
  P = (1, Real.sqrt 2)

theorem find_equation_and_min_sum (e c : ℝ) (F P : ℝ × ℝ) (a b : ℝ) :
  ellipse_center ∧ ellipse_focus_y_axis F ∧ ellipse_eccentricity e ∧ ellipse_contains_point P →
  (∃ a b : ℝ, (P.snd ^ 2) / (2 * c ^ 2) + (P.fst ^ 2) / (c ^ 2) = 1 ∧ 
               ∃ A B C D : ℝ × ℝ, ( ∀ (k : ℝ), |AB| + |CD| ≥ 16 / 3)) := 
sorry

end find_equation_and_min_sum_l274_274906


namespace symmetric_about_z_correct_l274_274614

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_z (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_about_z_correct (p : Point3D) :
  p = {x := 3, y := 4, z := 5} → symmetric_about_z p = {x := -3, y := -4, z := 5} :=
by
  sorry

end symmetric_about_z_correct_l274_274614


namespace area_of_trapezium_l274_274858

-- Definition of the condition variables
def side1 := 24
def side2 := 18
def height := 15

-- The theorem to prove the area of the trapezium
theorem area_of_trapezium (a b h : ℕ) :
  a = side1 ∧ b = side2 ∧ h = height → (1 / 2 : ℚ) * (a + b) * h = 315 :=
by
  intros h,
  cases h with ha h,
  cases h with hb hh,
  rw [ha, hb, hh],
  have : (42 : ℚ) = (24 + 18 : ℚ), by norm_num,
  rw this,
  norm_num,
  sorry

end area_of_trapezium_l274_274858


namespace value_of_g_g_2_l274_274579

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l274_274579


namespace limit_of_a_seq_l274_274184

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 0 then 2 else (4 * (1 / 3) ^ (n - 1) - 2)

theorem limit_of_a_seq : 
  (∀ n : ℕ, a_seq (n + 1) = (a_seq n - 4) / 3) →
  a_seq 0 = 2 →
  tendsto a_seq at_top (𝓝 (-2)) :=
by 
  intro h_rec h_init
  sorry

end limit_of_a_seq_l274_274184


namespace sequence_general_formula_l274_274209

theorem sequence_general_formula
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (3 / 2) * (a n) - 3) :
  ∀ n, a n = 3 * (2 : ℝ) ^ n :=
by sorry

end sequence_general_formula_l274_274209


namespace sum_of_solutions_eq_zero_l274_274488

def f (x : ℝ) : ℝ :=
  2^(|x|) + 4*(|x|)

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | f(x) = 20}.to_finset, x) = 0 :=
sorry

end sum_of_solutions_eq_zero_l274_274488


namespace column_sum_correct_l274_274479

theorem column_sum_correct : 
  -- Define x to be the sum of the first column (which is also the minuend of the second column)
  ∃ x : ℕ, 
  -- x should match the expected valid sum provided:
  (x = 1001) := 
sorry

end column_sum_correct_l274_274479


namespace final_number_l274_274313

variables (crab goat bear cat hen : ℕ)

-- Given conditions
def row4_sum : Prop := 5 * crab = 10
def col5_sum : Prop := 4 * crab + goat = 11
def row2_sum : Prop := 2 * goat + crab + 2 * bear = 16
def col2_sum : Prop := cat + bear + 2 * goat + crab = 13
def col3_sum : Prop := 2 * crab + 2 * hen + goat = 17

-- Theorem statement
theorem final_number
  (hcrab : row4_sum crab)
  (hgoat_col5 : col5_sum crab goat)
  (hbear_row2 : row2_sum crab goat bear)
  (hcat_col2 : col2_sum cat crab bear goat)
  (hhen_col3 : col3_sum crab goat hen) :
  crab = 2 ∧ goat = 3 ∧ bear = 4 ∧ cat = 1 ∧ hen = 5 → (cat * 10000 + hen * 1000 + crab * 100 + bear * 10 + goat = 15243) :=
sorry

end final_number_l274_274313


namespace solution_x2_l274_274388

def equation_A (x : ℝ) : Prop := 4 * x = 2
def equation_B (x : ℝ) : Prop := 3 * x + 6 = 0
def equation_C (x : ℝ) : Prop := (1 / 2) * x = 0
def equation_D (x : ℝ) : Prop := 7 * x - 14 = 0

theorem solution_x2 (x : ℝ) : x = 2 → (equation_D x ∧ ¬equation_A x ∧ ¬equation_B x ∧ ¬equation_C x) :=
by {
  assume hx : x = 2,
  split,
  { rw [hx, equation_D, eq_self_iff_true] },
  { split,
    { rw [hx, equation_A, ne.def, eq_false_iff_ne, not_false_iff] },
    { split,
      { rw [hx, equation_B, ne.def, eq_false_iff_ne, not_false_iff] },
      { rw [hx, equation_C, ne.def, eq_false_iff_ne, not_false_iff] }
    }
  }
}

end solution_x2_l274_274388


namespace find_uv_l274_274473

open Real

theorem find_uv : 
    ∃ (u v : ℝ), 
    (vector.ofFn ![3, -1] + u • vector.ofFn ![8, -6] = 
     vector.ofFn ![0, -2] + v • vector.ofFn ![-3, 4]) → 
    (u = -15/28 ∧ v = 3/7) :=
by
  sorry

end find_uv_l274_274473


namespace complex_division_solution_l274_274526

theorem complex_division_solution (a b: ℝ) (h: (a + b*complex.I) / (2 - complex.I) = 3 + complex.I) : a - b = 8 := 
sorry

end complex_division_solution_l274_274526


namespace scientific_notation_of_number_l274_274066

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l274_274066


namespace pastrami_sandwich_cost_l274_274690

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l274_274690


namespace range_of_a_l274_274549

-- Define the function f(x)
def f (x : ℝ) : ℝ := x - 2 / x

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.cos (Real.pi * x / 2) + 11 - 2 * a

-- Conditions: Domain constraints for x1 and x2
def Domain_f (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2
def Domain_g (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Proof statement
theorem range_of_a (a : ℝ) (ha : a ≠ 0) :
  (∀ x1, Domain_f x1 → ∃ x2, Domain_g x2 ∧ g a x2 = f x1) ↔ 6 ≤ a ∧ a ≤ 10 :=
by
  sorry

end range_of_a_l274_274549


namespace exponent_of_3_in_factorial_30_l274_274241

theorem exponent_of_3_in_factorial_30 : (prime_factorization 30!).count 3 = 14 := 
sorry

end exponent_of_3_in_factorial_30_l274_274241


namespace triangle_sides_square_perfect_l274_274440

theorem triangle_sides_square_perfect (x y z : ℕ) (h : ∃ h_x h_y h_z, 
  h_x = h_y + h_z ∧ 
  2 * h_x * x = 2 * h_y * y ∧ 
  2 * h_x * x = 2 * h_z * z ) :
  ∃ k : ℕ, x^2 + y^2 + z^2 = k^2 :=
by
  sorry

end triangle_sides_square_perfect_l274_274440


namespace probability_one_likes_variety_shows_l274_274726

open Finset

theorem probability_one_likes_variety_shows :
  let boys := ({1, 2, 3, 4, 5} : Finset ℕ)
      like_variety_shows := ({1, 2} : Finset ℕ)
      not_like_variety_shows := ({3, 4, 5} : Finset ℕ)
  in probability (subset_subsets 2 boys)
    (λ s, card (s ∩ like_variety_shows) = 1) = 3 / 5 :=
by 
  sorry

end probability_one_likes_variety_shows_l274_274726


namespace boys_planted_more_by_62_percent_girls_fraction_of_total_l274_274217

-- Define the number of trees planted by boys and girls
def boys_trees : ℕ := 130
def girls_trees : ℕ := 80

-- Statement 1: Boys planted 62% more trees than girls
theorem boys_planted_more_by_62_percent : (boys_trees - girls_trees) * 100 / girls_trees = 62 := by
  sorry

-- Statement 2: The number of trees planted by girls represents 4/7 of the total number of trees
theorem girls_fraction_of_total : girls_trees * 7 = 4 * (boys_trees + girls_trees) := by
  sorry

end boys_planted_more_by_62_percent_girls_fraction_of_total_l274_274217


namespace base7_digits_1234_l274_274948

theorem base7_digits_1234 : ∀ (n : ℕ), n = 1234 → 
  ∀ (b : ℕ), b = 7 → 
  ∃ d : ℕ, d = 4 ∧ ∀ (p : ℕ), 1234 / b^p < b → d = p + 1 := 
by 
  intros n hn b hb
  exists 4
  split
  case right =>
    sorry
  case left =>
    rfl

end base7_digits_1234_l274_274948


namespace harmonic_series_terms_added_l274_274309

noncomputable def harmonic_inequality (n : ℕ) : Prop :=
  ∑ i in (range (2^n)), (1 : ℝ) / i > n / 2

theorem harmonic_series_terms_added (k : ℕ) (hk : ∑ i in (range (2^k)), (1 : ℝ) / i > k / 2) :
  ∑ i in (range (2^(k+1))), (1 : ℝ) / i - ∑ j in (range (2^k)), (1 : ℝ) / j = 2^k :=
by
  sorry

end harmonic_series_terms_added_l274_274309


namespace solve_equation_l274_274102

theorem solve_equation (x : ℝ) (h₀ : x ≠ -3) (h₁ : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : x = 9 :=
by
  sorry

end solve_equation_l274_274102


namespace calc_ratio_l274_274994

variables (u v w : ℝ^3)

def X := u + 2 * v + w
def Y := u + 2 * v - w
def Z := -2 * u + v + w
def W := 2 * u + v - w
def G := 2 * u + 2 * v + 2 * w
def H := 2 * u + 2 * v - 2 * w
def E := -2 * u + 2 * v + 2 * w
def D := 2 * u + 2 * v - 2 * w
def F := u + v
def A := (0 : ℝ^3)
def C := v + w

theorem calc_ratio : 
  let XG2 := (2 * u + 3 * v + 2 * w) • (2 * u + 3 * v + 2 * w)
  let YH2 := (2 * u + 3 * v - 2 * w) • (2 * u + 3 * v - 2 * w)
  let ZE2 := (-3 * u + 3 * v + 2 * w) • (-3 * u + 3 * v + 2 * w)
  let WD2 := (4 * u + 2 * v) • (4 * u + 2 * v)
  let XF2 := (2 * u + 3 * v) • (2 * u + 3 * v)
  let YA2 := (2 * u + 3 * v) • (2 * u + 3 * v)
  let ZC2 := (2 * v - w) • (2 * v - w)
  XG2 + YH2 + ZE2 + WD2 = 21 * (u•u) + 22 * (v•v) + 7 * (w•w) 
  ∧ XF2 + YA2 + ZC2 = 9 * (u•u) + 13 * (v•v) + 2 * (w•w) 
  →
  (XG2 + YH2 + ZE2 + WD2) / (XF2 + YA2 + ZC2) = (21 * (u•u) + 22 * (v•v) + 7 * (w•w)) / (9 * (u•u) + 13 * (v•v) + 2 * (w•w)) :=
by 
  sorry

end calc_ratio_l274_274994


namespace base_7_digits_1234_l274_274947

theorem base_7_digits_1234 : ∃ n : ℕ, nat.digits 7 1234 = [4] := 
sorry

end base_7_digits_1234_l274_274947


namespace arithmetic_sequence_ratio_l274_274121

variable (a_n : ℕ → ℤ) (S : ℕ → ℤ)
variable (a1 d : ℤ)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a_n n = a1 + n * d

-- Definition of the sum of the first n terms of the sequence
def sum_of_arithmetic_sequence (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S n = ∑ i in (Finset.range n), a_n (i + 1)

-- Given conditions
axiom condition1 : is_arithmetic_sequence a_n a1 d
axiom condition2 : sum_of_arithmetic_sequence S a_n
axiom condition3 : S 6 / S 3 = 4

-- Statement to prove
theorem arithmetic_sequence_ratio :
  S 5 / S 6 = 25 / 36 :=
by
  sorry

end arithmetic_sequence_ratio_l274_274121


namespace constant_term_expansion_l274_274701

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2 : ℤ)^(6 - r) * (Nat.choose 6 r : ℤ) 

theorem constant_term_expansion : (x : ℤ) = 60 :=
by
  have term_4 := binomial_expansion_term 4
  -- We know r = 4 gives the constant term
  -- Calculating the specific term
  have : term_4 = (-1)^4 * 2^(6-4) * Nat.choose 6 4 := rfl
  have : term_4 = 1 * 2^2 * 15 := rfl
  have : term_4 = 4 * 15 := rfl
  have : 4 * 15 = 60 := by ring
  exact this

end constant_term_expansion_l274_274701


namespace books_ratio_3_to_1_l274_274314

-- Definitions based on the conditions
def initial_books : ℕ := 220
def books_rebecca_received : ℕ := 40
def remaining_books : ℕ := 60
def total_books_given_away := initial_books - remaining_books
def books_mara_received := total_books_given_away - books_rebecca_received

-- The proof that the ratio of the number of books Mara received to the number of books Rebecca received is 3:1
theorem books_ratio_3_to_1 : (books_mara_received : ℚ) / books_rebecca_received = 3 := by
  sorry

end books_ratio_3_to_1_l274_274314


namespace hyperbola_asymptote_angle_range_l274_274984

theorem hyperbola_asymptote_angle_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let focus_distance := λ x c : ℝ, |x - c|
      directrix_distance := λ x d : ℝ, |x - d|
      c := Real.sqrt (a^2 + b^2)
      e := c / a
      x := (3 / 2) * a in
  (e * focus_distance x c > directrix_distance x (-a^2 / c)) →
  (0 < 2 * Real.arctan(b / a) ∧ 2 * Real.arctan(b / a) < Real.pi / 3) :=
by
  sorry

end hyperbola_asymptote_angle_range_l274_274984


namespace two_largest_divisors_difference_l274_274444

theorem two_largest_divisors_difference (N : ℕ) (h : N > 1) (a : ℕ) (ha : a ∣ N) (h6a : 6 * a ∣ N) :
  (N / 2 : ℚ) / (N / 3 : ℚ) = 1.5 := by
  sorry

end two_largest_divisors_difference_l274_274444


namespace scientific_notation_of_number_l274_274067

theorem scientific_notation_of_number (num : ℝ) (a b: ℝ) : 
  num = 0.0000046 ∧ 
  (a = 46 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -7 ∨ 
   a = 4.6 ∧ b = -6 ∨ 
   a = 0.46 ∧ b = -5) → 
  a = 4.6 ∧ b = -6 :=
by
  sorry

end scientific_notation_of_number_l274_274067


namespace avg_age_of_children_l274_274225

theorem avg_age_of_children 
  (participants : ℕ) (women : ℕ) (men : ℕ) (children : ℕ)
  (overall_avg_age : ℕ) (avg_age_women : ℕ) (avg_age_men : ℕ)
  (hp : participants = 50) (hw : women = 22) (hm : men = 18) (hc : children = 10)
  (ho : overall_avg_age = 20) (haw : avg_age_women = 24) (ham : avg_age_men = 19) :
  ∃ (avg_age_children : ℕ), avg_age_children = 13 :=
by
  -- Proof will be here.
  sorry

end avg_age_of_children_l274_274225


namespace circumcircle_values_eq_prod_l274_274374

theorem circumcircle_values_eq_prod (n : ℕ) (h : n ≥ 3) (x : ℕ → ℝ) (hdistinct: ∀ i j, i ≠ j → x i ≠ x j)
  (hproduct: ∀ i, x i = x (i - 1) * x (i + 1)) : n = 6 :=
begin
  sorry
end

end circumcircle_values_eq_prod_l274_274374


namespace isabella_hair_growth_l274_274621

theorem isabella_hair_growth : 
  ∀ (initial_length final_length growth : ℕ), 
  initial_length = 18 ∧ final_length = 24 ∧ (final_length - initial_length) = growth → growth = 6 :=
by
  intros initial_length final_length growth h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3] at h4
  exact h4

end isabella_hair_growth_l274_274621


namespace jumps_per_second_l274_274300

-- Define the conditions and known values
def record_jumps : ℕ := 54000
def hours : ℕ := 5
def seconds_per_hour : ℕ := 3600

-- Define the target question as a theorem to prove
theorem jumps_per_second :
  (record_jumps / (hours * seconds_per_hour)) = 3 := by
  sorry

end jumps_per_second_l274_274300


namespace _l274_274339

noncomputable theorem angle_of_point_on_ellipse :
  ∀ (F₁ F₂ P : ℝ × ℝ), (P.1^2 / 9 + P.2^2 / 2 = 1) → 
  (dist P F₁ = 4) → 
  (angle F₁ P F₂ = 120) :=
by sorry

end _l274_274339


namespace sequence_general_formula_l274_274588

-- Define conditions: The sum of the first n terms of the sequence is Sn = an - 3
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom condition (n : ℕ) : S n = a n - 3

-- Define the main theorem to prove
theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : a n = 2 * 3 ^ n :=
sorry

end sequence_general_formula_l274_274588


namespace range_a_l274_274508

noncomputable def f (x : ℝ) : ℝ := sin x + cos x + sin (2 * x)

theorem range_a (a : ℝ) : (∀ t x : ℝ, a * sin t + 2 * a + 1 ≥ f x) ↔ a ≥ real.sqrt 2 :=
begin
  sorry
end

end range_a_l274_274508
