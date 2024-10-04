import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.CombinatorialProofs
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Nat.Powers
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Algebra.Real
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.MeasureSpace
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Algebra.MonicLimits
import data.real.basic

namespace intersection_M_N_l438_438600

open Set

def M : Set ℝ := {x | x ≤ 0}
def N : Set ℝ := {x | x^2 ≤ 1}

theorem intersection_M_N : M ∩ N = {x | −1 ≤ x ∧ x ≤ 0} := by
  sorry

end intersection_M_N_l438_438600


namespace intervals_of_monotonicity_f_find_a_critical_points_range_of_t_l438_438898

noncomputable def f (x : ℝ) (m : ℝ) := -x^2 + m * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) := x - a / x

theorem intervals_of_monotonicity_f (m : ℝ) :
  (∀ x > 0, m ≤ 0 → deriv (λ x, f x m) x < 0) ∧
  (∀ x > 0, m > 0 → 
    ((0 < x ∧ x < Real.sqrt (m / 2)) → deriv (λ x, f x m) x > 0) ∧
    ((x > Real.sqrt (m / 2)) → deriv (λ x, f x m) x < 0)) :=
sorry

theorem find_a_critical_points (a : ℝ) :
  ∃ x > 0, deriv (λ x, g x a) x = 0 ∧ deriv (λ x, f x 2) x = 0 → a = -1 :=
sorry

theorem range_of_t (x₁ x₂ t : ℝ) (h₁ : x₁ ∈ Icc (1 / Real.exp 1) 5) (h₂ : x₂ ∈ Icc (1 / Real.exp 1) 5) :
  (∀ t, ((t + 1 > 0) → (t > -1) ∧ (t ≥ -4)) ∧ 
        ((t + 1 < 0) → (t < -1) ∧ (t ≤ -156 / 5 + 2 * Real.log 5))) :=
sorry

end intervals_of_monotonicity_f_find_a_critical_points_range_of_t_l438_438898


namespace sector_area_correct_l438_438248

noncomputable def sector_area (r θ : ℝ) : ℝ := 0.5 * θ * r^2

theorem sector_area_correct (r θ : ℝ) (hr : r = 2) (hθ : θ = 2 * Real.pi / 3) :
  sector_area r θ = 4 * Real.pi / 3 :=
by
  subst hr
  subst hθ
  sorry

end sector_area_correct_l438_438248


namespace sum_of_first_10_common_elements_is_correct_l438_438845

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric sequence
def b (k : ℕ) : ℕ := 10 * 2^k

-- Define a function that finds the common elements in both sequences
def common_elements (N : ℕ) : List ℕ :=
List.filter (λ x, ∃ n k, x = a n ∧ x = b k) (List.range (N + 1))

-- Define the sum of the first 10 common elements
def sum_first_10_common_elements : ℕ :=
(List.sum (List.take 10 (common_elements 100)))

theorem sum_of_first_10_common_elements_is_correct :
  sum_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_is_correct_l438_438845


namespace distance_between_tangents_l438_438805

noncomputable def sqrt2 := real.sqrt 2

def parabola1 (x : ℝ) : ℝ := x^2 + 1

def parabola2 (y : ℝ) : ℝ := y^2 + 1

def tangent_line (b : ℝ) (x : ℝ) : ℝ := x + b

theorem distance_between_tangents : 
  ∃ d : ℝ, (d = (3 * sqrt2) / 4) ∧
    (∀ x1 y1 : ℝ, 
      (y1 = parabola1 x1) → 
      (tangent_line (3 / 4) x1 = y1)) ∧
    (∀ x2 y2 : ℝ, 
      (x2 = parabola2 y2) → 
      (tangent_line ((3/sqrt2)/2 - 1) x2 = y2)) :=
sorry

end distance_between_tangents_l438_438805


namespace dot_product_ab_norm_4a_minus_2b_perp_k_a_minus_b_l438_438208

variables (a b : ℝ^3)
variables (k : ℝ)

axioms 
  (norm_a : ∥a∥ = 4)
  (norm_b : ∥b∥ = 8)
  (norm_a_add_b : ∥a + b∥ = 4 * Real.sqrt 3)

theorem dot_product_ab : a • b = -16 := sorry

theorem norm_4a_minus_2b : ∥4 • a - 2 • b∥ = 16 * Real.sqrt 3 := sorry

theorem perp_k_a_minus_b (h : (a + 2 • b) ⬝ (k • a - b) = 0) : k = -7 := sorry

end dot_product_ab_norm_4a_minus_2b_perp_k_a_minus_b_l438_438208


namespace option_d_is_quadratic_equation_l438_438389

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l438_438389


namespace extreme_point_distance_number_of_roots_l438_438225

noncomputable def f (x : ℝ) : ℝ := real.log (x ^ 2 + 1)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 1 / (x ^ 2 - 1) + a
noncomputable def l (x y a : ℝ) : Prop := 2 * real.sqrt 2 * x + y + a + 5 = 0

theorem extreme_point_distance (a : ℝ) :
  let x := 0 in
  |2 * real.sqrt 2 * x + a + 5| = 3 →
  a = -2 ∨ a = -8 := sorry

theorem number_of_roots (a : ℝ) :
  let h (x : ℝ) := f x - g x a in
  if a < 1 then 
    set.count_roots h = 2 
  else if a = 1 then 
    set.count_roots h = 3 
  else 
    set.count_roots h = 4 := sorry

end extreme_point_distance_number_of_roots_l438_438225


namespace pigeons_count_l438_438406

theorem pigeons_count :
  let initial_pigeons := 1
  let additional_pigeons := 1
  (initial_pigeons + additional_pigeons) = 2 :=
by
  sorry

end pigeons_count_l438_438406


namespace not_possible_arrange_cards_l438_438020

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438020


namespace probability_correct_l438_438377

noncomputable def probability_of_at_least_one_black_ball 
  (black white total drawn : ℕ) : ℚ :=
  (Finset.card (Finset.Icc 1 black) * Finset.card (Finset.Icc 1 white) +
  Finset.card (Finset.Icc 1 black) * (Finset.card (Finset.Icc 1 black) - 1) / 2) / 
  (Finset.card (Finset.Icc 1 total) * (Finset.card (Finset.Icc 1 total) - 1) / 2)

theorem probability_correct :
  probability_of_at_least_one_black_ball 5 3 8 2 = (25/28 : ℚ) :=
  sorry

end probability_correct_l438_438377


namespace new_drug_effectiveness_expectation_of_X_company_claim_doubt_l438_438077

section
variables {n a b c d : ℕ}
variables (ta tb tc td : ℕ) (K : ℝ)

-- Part 1: Given the conditions, prove the calculated K^2 value is less than the critical value at 90% confidence
theorem new_drug_effectiveness :
  let K2 := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  in n = 200 → a = 60 → b = 40 → c = 50 → d = 50 →
     K2 < 2.706 :=
sorry

-- Part 2: Given the conditions, find the expectation of X, where X is the number of cured patients out of a sample of 3
theorem expectation_of_X :
  let p_cured := 0.6
  in ∀ X : ℕ → ℝ, (binomial C_3(X) * (p_cured)^X * (1 - p_cured)^(3 - X)) * X in $[0,1,2,3]$
  → ∑ P(X) = 1 → E[X] = 1.8 :=
sorry

-- Part 3: Evaluate the company's claim that the efficacy is 90%
theorem company_claim_doubt :
  let claim_eff := 0.9
  in ∑ P(X ≤ 6 out of 10 patients) ≈ 0.013 
  -- Probability calculated under binomial distribution
  → this probability is very small
  → we should doubt the company's claim :=
sorry
end

end new_drug_effectiveness_expectation_of_X_company_claim_doubt_l438_438077


namespace isosceles_triangle_AC_LT_AB_l438_438876

variables {A B C K N : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space K] [metric_space N]
variables {d : A → A → ℝ}

noncomputable def is_isosceles (T : triangle A) : Prop := 
  d T.a T.b = d T.b T.c

noncomputable def point_on_side (a b c p : A) : Prop := 
  d a b = d a p + d p b ∧ d b c = d b p + d p c

theorem isosceles_triangle_AC_LT_AB 
  {A B C K N : A} 
  (h1 : is_isosceles (triangle.mk A B C))
  (h2 : point_on_side B C K)
  (h3 : point_on_side B C N)
  (h4 : d B K + d K N = d B N)
  (h5 : d K N = d A N) : 
  d A C < d A B := 
sorry

end isosceles_triangle_AC_LT_AB_l438_438876


namespace evaluate_g_l438_438125

noncomputable def g (x : ℝ) : ℝ := x^3 + x^2 + 2 * real.sqrt x

theorem evaluate_g : 2 * g 3 + g 9 = 888 + 4 * real.sqrt 3 :=
by
  sorry

end evaluate_g_l438_438125


namespace jacket_cost_l438_438659

theorem jacket_cost (S J : ℝ) : 
  10 * S + 20 * J = 800 ∧ 5 * S + 15 * J = 550 → J = 30 :=
by
  intro h,
  cases h with h1 h2,
  sorry

end jacket_cost_l438_438659


namespace triangle_similarity_AOI_IOL_l438_438441

open EuclideanGeometry

variables {A B C O I K L : Point}
variables [isCircumcenter O A B C]
variables [isIncenter I A B C]
variables (hK: symmetricPoint I A K)
variables (hL: symmetricPoint (line_through B C) K L)

theorem triangle_similarity_AOI_IOL :
  Triangle AOI ∼ Triangle IOL := sorry

end triangle_similarity_AOI_IOL_l438_438441


namespace no_valid_arrangement_l438_438009

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438009


namespace Antoinette_weight_l438_438778

variable (A R : ℝ)

theorem Antoinette_weight :
  A = 63 → 
  (A = 2 * R - 7) → 
  (A + R = 98) →
  True :=
by
  intros hA hAR hsum
  have h := hAR.symm
  rw [h] at hsum
  have h1 : 2 * R - 7 + R = 98 := hsum
  have h2 : 3 * R - 7 = 98 := h1
  have h3 : 3 * R = 105 := by linarith
  have hR : R = 35 := by linarith
  have hA' : A = 2 * 35 - 7 := by rwa [←h]
  have hA' : A = 63 := by linarith
  sorry

end Antoinette_weight_l438_438778


namespace find_d_minus_c_l438_438665

noncomputable def point_transformed (c d : ℝ) : Prop :=
  let Q := (c, d)
  let R := (2 * 2 - c, 2 * 3 - d)  -- Rotating Q by 180º about (2, 3)
  let S := (d, c)                -- Reflecting Q about the line y = x
  (S.1, S.2) = (2, -1)           -- Result is (2, -1)

theorem find_d_minus_c (c d : ℝ) (h : point_transformed c d) : d - c = -1 :=
by {
  sorry
}

end find_d_minus_c_l438_438665


namespace new_percentage_water_l438_438753

theorem new_percentage_water (original_volume : ℝ) (original_percentage_water : ℝ) (added_water : ℝ) :
  original_volume = 200 ∧
  original_percentage_water = 0.20 ∧
  added_water = 13.333333333333334 →
  ((original_percentage_water * original_volume + added_water) / (original_volume + added_water)) * 100 = 25 :=
by
  intros h
  cases h with vol_and_perc added_eq
  cases vol_and_perc with vol perc
  rw [vol, perc, added_eq]
  sorry

end new_percentage_water_l438_438753


namespace nell_initial_cards_l438_438321

theorem nell_initial_cards (cards_given cards_left total_cards : ℕ)
  (h1 : cards_given = 301)
  (h2 : cards_left = 154)
  (h3 : total_cards = cards_given + cards_left) :
  total_cards = 455 := by
  rw [h1, h2] at h3
  exact h3

end nell_initial_cards_l438_438321


namespace problem_statement_l438_438944

theorem problem_statement (x : ℤ) (h₁ : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := 
sorry

end problem_statement_l438_438944


namespace period_of_cos_3x_plus_4_l438_438697

noncomputable def cos_period : ℝ := 2 * Real.pi

def function_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem period_of_cos_3x_plus_4 : function_period (λ x, Real.cos (3 * x) + 4) (2 * Real.pi / 3) :=
sorry

end period_of_cos_3x_plus_4_l438_438697


namespace max_n_in_grid_l438_438477

/-- Mathematical Problem:
Find the maximum possible value of  n , such that the 
integers  1,2,..., n  can be filled once each in distinct 
cells of a  2015×2015  grid and satisfies the following conditions:
1. For all  1 ≤ i ≤ n-1 , the cells with  i  and  i+1  share an edge. 
   Cells with  1  and  n  also share an edge. 
   In addition, no other pair of numbers share an edge.
2. If two cells with  i < j  in them share a vertex, then  
   min{j-i, n+i-j}=2.

We aim to prove that the maximum possible value of n 
is 2016^2 / 2.
-/
theorem max_n_in_grid :
  ∃ n, n = (2016^2 / 2) ∧ (∀ i, (1 ≤ i ∧ i ≤ n - 1) → (cells_share_edge i (i+1))) ∧
  (cells_share_edge 1 n) ∧
  (∀ i j, i ≠ j → ¬ cells_share_edge i j) ∧
  (∀ i j, i < j → cells_share_vertex i j → (min (j - i) (n + i - j) = 2)) :=
sorry

end max_n_in_grid_l438_438477


namespace brian_spent_on_kiwis_l438_438733

theorem brian_spent_on_kiwis :
  ∀ (cost_per_dozen_apples : ℝ)
    (cost_for_24_apples : ℝ)
    (initial_money : ℝ)
    (subway_fare_one_way : ℝ)
    (total_remaining : ℝ)
    (kiwis_spent : ℝ)
    (bananas_spent : ℝ),
  cost_per_dozen_apples = 14 →
  cost_for_24_apples = 2 * cost_per_dozen_apples →
  initial_money = 50 →
  subway_fare_one_way = 3.5 →
  total_remaining = initial_money - 2 * subway_fare_one_way - cost_for_24_apples →
  total_remaining = 15 →
  bananas_spent = kiwis_spent / 2 →
  kiwis_spent + bananas_spent = total_remaining →
  kiwis_spent = 10 :=
by
  -- Sorry means we are skipping the proof
  sorry

end brian_spent_on_kiwis_l438_438733


namespace desired_percentage_of_alcohol_l438_438731

theorem desired_percentage_of_alcohol 
  (original_volume : ℝ)
  (original_percentage : ℝ)
  (added_volume : ℝ)
  (added_percentage : ℝ)
  (final_percentage : ℝ) :
  original_volume = 6 →
  original_percentage = 0.35 →
  added_volume = 1.8 →
  added_percentage = 1.0 →
  final_percentage = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end desired_percentage_of_alcohol_l438_438731


namespace probability_xi_eq_2_l438_438686

noncomputable def prob_xi_eq_2 : ℚ := 
  let outcomes := [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)] in
  let favorable := outcomes.filter (λ o, (o.1 * o.2) = 2) in
  favorable.length / outcomes.length

theorem probability_xi_eq_2 (cards : Finset ℕ) (hk : cards = {0, 1, 2}) :
  prob_xi_eq_2 = 2 / 9 :=
by
  sorry

end probability_xi_eq_2_l438_438686


namespace h_of_neg_one_l438_438991

def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x) ^ 2 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg_one :
  h (-1) = 298 :=
by
  sorry

end h_of_neg_one_l438_438991


namespace sum_of_first_10_common_elements_l438_438846

-- Define arithmetic sequence
def a_n (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric sequence
def b_k (k : ℕ) : ℕ := 10 * 2^k

-- Define common elements sequence
def common_element (m : ℕ) : ℕ := 10 * 4^m

-- State the theorem
theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum (λ i, common_element i) = 3495250 := sorry

end sum_of_first_10_common_elements_l438_438846


namespace octal_to_binary_135_l438_438126

theorem octal_to_binary_135 : convert_octal_to_binary 135 = 1011101 := by
  sorry

end octal_to_binary_135_l438_438126


namespace number_of_true_propositions_l438_438297

section
variables {l m n : Type*} {α β γ : Type*}

-- Definitions for the conditions
def perp (x y : Type*) : Prop := sorry -- Insert the definition of perpendicularity here
def parallel (x y : Type*) : Prop := sorry -- Insert the definition of parallelism here
def subset_of (x y : Type*) : Prop := sorry -- Insert the definition of subset here
def projection (l α β : Type*) : Type* := sorry -- Insert the definition of projection here

-- Propositions
def p1 (l α m β : Type*) : Prop := perp l α ∧ perp m l ∧ perp m β → perp α β
def p2 (m β n l : Type*) : Prop := subset_of m β ∧ (projection l α β = n) ∧ perp m n → perp m l
def p3 (α β γ : Type*) : Prop := perp α β ∧ perp α γ → parallel α β

-- Main statement
theorem number_of_true_propositions : 
  (p1 l α m β) ∨ (p2 m β n l) ∨ (p3 α β γ) = 2 :=
sorry
end

end number_of_true_propositions_l438_438297


namespace part_a_part_b_l438_438597

-- This definition states that a number p^m is a divisor of a-1
def divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  (p ^ m) ∣ (a - 1)

-- This definition states that (p^(m+1)) is not a divisor of a-1
def not_divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  ¬ (p ^ (m + 1) ∣ (a - 1))

-- Part (a): Prove divisibility
theorem part_a (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  p ^ (m + n) ∣ a ^ (p ^ n) - 1 := 
sorry

-- Part (b): Prove non-divisibility
theorem part_b (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  ¬ p ^ (m + n + 1) ∣ a ^ (p ^ n) - 1 := 
sorry

end part_a_part_b_l438_438597


namespace boat_travel_time_downstream_l438_438410

-- Define the conditions as hypotheses
variables (v t : ℝ)
hypothesis h1 : 120 = (v - 4) * (t + 1)
hypothesis h2 : 120 = (v + 4) * t

-- Define the theorem stating that the time to travel downstream is 1 hour
theorem boat_travel_time_downstream : t = 1 :=
by {
  -- The proof is omitted
  sorry
}

end boat_travel_time_downstream_l438_438410


namespace parabola_intersection_diff_l438_438137

theorem parabola_intersection_diff (a b c d : ℝ) 
  (h₁ : ∀ x y, (3 * x^2 - 2 * x + 1 = y) → (c = x ∨ a = x))
  (h₂ : ∀ x y, (-2 * x^2 + 4 * x + 1 = y) → (c = x ∨ a = x))
  (h₃ : c ≥ a) :
  c - a = 6 / 5 :=
by sorry

end parabola_intersection_diff_l438_438137


namespace parallel_lines_distance_l438_438653

-- Defining the first line as a function L1
def L1 (x y : ℝ) : Prop := x + 2 * y = 5

-- Defining the second line as a function L2
def L2 (x y : ℝ) : Prop := x + 2 * y = 10

-- Function to calculate the distance between two lines of the form ax + by + c = 0
def distance_between_parallel_lines 
(a b c1 c2 : ℝ) : ℝ :=
  |c2 - c1| / Real.sqrt (a^2 + b^2)

-- Proving the given problem
theorem parallel_lines_distance :
  distance_between_parallel_lines 1 2 (-5) (-10) = Real.sqrt 5 :=
by
  sorry

end parallel_lines_distance_l438_438653


namespace CD_length_l438_438500

-- Definitions based on conditions
variables {A B C D L : Type} -- points defining the parallelogram and point L
variables [EuclideanGeometry A B C D L] -- assuming Euclidean geometry

-- Conditions from the problem
variable (angle_D : ∠ D = 100)
variable (BC_len : dist B C = 12)
variable (L_on_AD : L ∈ (segment A D))
variable (angle_ABL : ∠ ABL = 50)
variable (LD_len : dist L D = 4)

-- Theorem to prove
theorem CD_length : dist C D = 8 :=
by
  sorry

end CD_length_l438_438500


namespace non_self_intersecting_polygon_l438_438183

theorem non_self_intersecting_polygon (p : ℕ) (points : Fin p → ℝ × ℝ) 
    (h_no_three_collinear : ∀ (i j k : Fin p), i ≠ j → j ≠ k → i ≠ k → ¬Collinear (points i) (points j) (points k)) :
    ∃ (labeling : Fin p → Fin p), NonSelfIntersectingPolygon (labeling points) :=
by 
    sorry

end non_self_intersecting_polygon_l438_438183


namespace sum_of_first_10_common_elements_l438_438847

-- Define arithmetic sequence
def a_n (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric sequence
def b_k (k : ℕ) : ℕ := 10 * 2^k

-- Define common elements sequence
def common_element (m : ℕ) : ℕ := 10 * 4^m

-- State the theorem
theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum (λ i, common_element i) = 3495250 := sorry

end sum_of_first_10_common_elements_l438_438847


namespace area_increase_by_nine_l438_438673

theorem area_increase_by_nine (a : ℝ) :
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := extended_side_length^2;
  extended_area / original_area = 9 :=
by
  let original_area := a^2;
  let extended_side_length := 3 * a;
  let extended_area := (extended_side_length)^2;
  sorry

end area_increase_by_nine_l438_438673


namespace total_chess_games_l438_438679

theorem total_chess_games (n : ℕ) (h_n : n = 9) : (nat.choose n 2) = 36 :=
by
  rw h_n
  sorry

end total_chess_games_l438_438679


namespace distance_between_Youngjun_house_and_school_l438_438652

theorem distance_between_Youngjun_house_and_school :
  let distance_Seunghun_school := 1.05 -- distance between Seunghun's house and school (in km)
  let closer_distance := 0.46 -- closer distance in km (originally given as 460 meters)
  let distance_Youngjun_school := distance_Seunghun_school - closer_distance -- calculation
  distance_Youngjun_school = 0.59 := by
  unfold distance_Seunghun_school closer_distance distance_Youngjun_school
  sorry

end distance_between_Youngjun_house_and_school_l438_438652


namespace necessary_but_not_sufficient_condition_not_sufficient_condition_l438_438864

theorem necessary_but_not_sufficient_condition (x y : ℝ) (h : x > 0) : 
  (x > |y|) → (x > y) :=
by
  sorry

theorem not_sufficient_condition (x y : ℝ) (h : x > 0) :
  ¬ ((x > y) → (x > |y|)) :=
by
  sorry

end necessary_but_not_sufficient_condition_not_sufficient_condition_l438_438864


namespace bubble_sort_two_rounds_l438_438926

def bubble_sort_pass (l : List ℕ) : List ℕ :=
  l.foldl (λ acc x, acc.init ++ [min acc.head x, max acc.head x]) l

def bubble_sort_rounds (initial_sequence : List ℕ) (target_sequence : List ℕ) : ℕ :=
  (List.range 100).find (λ n, (List.iterate n bubble_sort_pass initial_sequence) = target_sequence) |>.getD 0

theorem bubble_sort_two_rounds {l₁ l₂ : List ℕ} : 
  l₁ = [37, 21, 3, 56, 9, 7] → 
  l₂ = [3, 9, 7, 21, 37, 56] →
  bubble_sort_rounds l₁ l₂ = 2 :=
by
  intros h1 h2
  unfold bubble_sort_rounds
  rw [h1, h2]
  sorry

end bubble_sort_two_rounds_l438_438926


namespace arithmetic_sequence_common_difference_l438_438643

open Function

theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) 
  (h1 : ∀ n, a n = a 0 + n * d)
  (h_sum1 : (Finset.range 50).sum a = 50)
  (h_sum2 : (Finset.range 50).sum (λ n, a (n + 50)) = 150) :
  d = 1 / 25 :=
by
  sorry

end arithmetic_sequence_common_difference_l438_438643


namespace no_possible_arrangement_l438_438001

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l438_438001


namespace probability_of_selecting_one_fork_one_spoon_one_knife_l438_438256

theorem probability_of_selecting_one_fork_one_spoon_one_knife :
  (∃ (drawers : Type) (forks spoons knives : drawers → Prop),
    (∀ (d : drawers), forks d ∨ spoons d ∨ knives d) ∧
    (∀ (d1 d2 : drawers), forks d1 ∧ forks d2 → d1 = d2) ∧
    (∀ (d1 d2 : drawers), spoons d1 ∧ spoons d2 → d1 = d2) ∧
    (∀ (d1 d2 : drawers), knives d1 ∧ knives d2 → d1 = d2) ∧
    (∃ (drawers_list : list drawers), drawers_list.length = 30 ∧
      list.countp forks drawers_list = 10 ∧
      list.countp spoons drawers_list = 10 ∧
      list.countp knives drawers_list = 10) ∧
    ∃ (select : finset drawers), select.card = 3 ∧
      (select.countp forks = 1 ∧ select.countp spoons = 1 ∧ select.countp knives = 1))
  → ( (4060 : ℝ) / ((10:ℕ) * 10 * 10) = (500 : ℝ) / 203)) :=
begin
  sorry
end

end probability_of_selecting_one_fork_one_spoon_one_knife_l438_438256


namespace coefficient_x3y6_in_expansion_l438_438648

theorem coefficient_x3y6_in_expansion (R : Type*) [CommRing R] (x y : R) : 
  coefficient (expand_polynomial ((x - y) ^ 2 * (x + y) ^ 7)) (monomial 3 6) = 0 := 
sorry

end coefficient_x3y6_in_expansion_l438_438648


namespace inaccurate_city_description_l438_438732

theorem inaccurate_city_description :
  let dream_description := true in
  let colleague_reaction := true in
  let nanny_appearance := true in
  let protagonist_avoidance := true in
  ¬ (city_is_indifferent) :=
sorry

end inaccurate_city_description_l438_438732


namespace ratio_YX_XZ_BP_PC_l438_438277

namespace geometry_proof

open EuclideanGeometry

def points_on_sides (A B C P Q R : Point) : Prop :=
  (Line_through B C) ∈ P ∧ (Line_through C A) ∈ Q ∧ (Line_through A B) ∈ R

def circumcircle (A B C P Q R X Y Z : Point) : Prop :=
  ∃ ΓA ΓB ΓC : Circle, 
    is_circumcircle_of_triangle A Q R ΓA ∧ is_circumcircle_of_triangle B R P ΓB ∧ 
    is_circumcircle_of_triangle C P Q ΓC ∧
    (Point_lies_on_circle X ΓA ∧ Point_lies_on_circle Y ΓB ∧ Point_lies_on_circle Z ΓC) ∧
    (Line_through A P).intersects_circles_at AP ΓA X Y Z

theorem ratio_YX_XZ_BP_PC 
  (A B C P Q R X Y Z : Point)
  (h1: points_on_sides A B C P Q R)
  (h2: circumcircle A B C P Q R X Y Z) :
  YX / XZ = BP / PC :=
sorry

end geometry_proof

end ratio_YX_XZ_BP_PC_l438_438277


namespace product_of_solutions_l438_438698

theorem product_of_solutions : 
  (∀ x : ℝ, x + (1 / x) = 3 * x → x = sqrt (1 / 2) ∨ x = -sqrt (1 / 2)) →
  ∏ x in ({sqrt (1 / 2), -sqrt (1 / 2)} : finset ℝ), x = -1 / 2 :=
by
  sorry

end product_of_solutions_l438_438698


namespace arithmetic_sequence_property_l438_438892

variable (a_n : ℕ → ℝ)
variable (S_13 : ℝ)

axiom sum_first_13_terms : S_13 = ∑ i in (finset.range 13).map (finset.nat_embedding (0 : ℕ)).to_list, a_n i
axiom S_13_value : S_13 = 39

theorem arithmetic_sequence_property 
  (a_6, a_7, a_8 : ℝ)
  (h1 : a_n 6 = a_6)
  (h2 : a_n 7 = a_7)
  (h3 : a_n 8 = a_8)
  (h4 : a_n 7 = 3)
  (h5 : a_6 = a_7 - d)
  (h6 : a_8 = a_7 + d) :
  a_6 + a_7 + a_8 = 9 := by
  sorry

end arithmetic_sequence_property_l438_438892


namespace expand_polynomial_l438_438816

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 :=
by sorry

end expand_polynomial_l438_438816


namespace cosine_angle_BM_AC_l438_438958

-- Define a tetrahedron with given properties
variables (A B C D M : Type*)
variables [InnerProductSpace ℝ (A × B × C × D × M)]

-- Given conditions
variable (sqrt_two : ℝ := real.sqrt 2)
variable (sqrt_three : ℝ := real.sqrt 3)
variable (one : ℝ := 1)
variable (half_sqrt_two : ℝ := sqrt_two / 2)

-- Midpoint M of CD
axiom midpoint_def : ∀ (C D : Type*), M = (C + D) / 2

-- Cosine of the angle between BM and AC
def cosine_BM_AC 
    (B C D M A : Type*) 
    [InnerProductSpace ℝ (A × B × C × D × M)] 
    [InnerProductSpace ℝ (B × M)] 
    [InnerProductSpace ℝ (A × C)] 
    : ℝ :=
((sqrt_two * sqrt_two + (sqrt_two / 2)^2 - sqrt_three^2) / 
 (2 * (sqrt_two * sqrt_three)))

theorem cosine_angle_BM_AC : 
    1.
    (cosine_BM_AC B M A C D M) = real.sqrt 2 / 3 := sorry

#check cosine_angle_BM_AC

end cosine_angle_BM_AC_l438_438958


namespace real_part_z_pow_2017_l438_438884

open Complex

noncomputable def z : ℂ := 1 + I

theorem real_part_z_pow_2017 : re (z ^ 2017) = 2 ^ 1008 := sorry

end real_part_z_pow_2017_l438_438884


namespace norbs_age_is_29_l438_438344

-- Define the list of guesses
def guesses := [25, 29, 31, 33, 37, 39, 42, 45, 48, 50]

-- Prime checking function
def is_prime(n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

-- Definition of Norb's age satisfying all conditions
def Norb_age (x : ℕ) : Prop :=
  x ∈ guesses ∧ is_prime(x) ∧
  (finset.card (finset.filter (λ g, g < x) (finset.mk guesses sorry)) < 2 * finset.card (finset.mk guesses sorry) / 3) ∧
  (finset.card (finset.filter (λ g, abs(g - x) = 1) (finset.mk guesses sorry)) = 2)

theorem norbs_age_is_29 : Norb_age 29 :=
  sorry

end norbs_age_is_29_l438_438344


namespace Julie_work_hours_per_week_l438_438988

theorem Julie_work_hours_per_week (
  total_summer_hours : ℝ :=
  48 * 12) 
  (total_summer_earnings : ℝ := 
  5000) 
  (total_school_year_weeks : ℝ :=
  48)
  (total_school_year_earnings : ℝ := 
  8000) :
  19.2 ≈ total_school_year_earnings / (total_summer_earnings / total_summer_hours) / total_school_year_weeks :=
by
  sorry

end Julie_work_hours_per_week_l438_438988


namespace arithmetic_seq_ratio_l438_438547

-- Definitions for the sums of arithmetic sequences
def A_n (n : ℕ) : ℚ := (a_1 + a_n) * n / 2
def B_n (n : ℕ) : ℚ := (b_1 + b_n) * n / 2

-- Given condition 
theorem arithmetic_seq_ratio (a_n b_n : ℕ → ℚ) :
  ∀ n : ℕ, n > 0 → (A_n n) / (B_n n) = (7 * n + 57) / (n + 3) ↔
  let num_integers := {n : ℕ | n > 0 ∧ (a_n n) / (b_n n) ∈ ℤ}.card in
  num_integers = 5 :=
begin
  -- Only the statement is required according to the instructions
  sorry
end

end arithmetic_seq_ratio_l438_438547


namespace sum_values_y_l438_438608

-- Define the function g
def g (x : ℝ) : ℝ := x^3 - x^2 + 2 * x + 3

-- Statement of the problem: Sum of all y such that g(3y) = 9
theorem sum_values_y (S : set ℝ) (h : ∀ y ∈ S, g (3 * y) = 9) : ∑ y in S, y = 1/3 :=
by
  sorry

end sum_values_y_l438_438608


namespace no_valid_arrangement_l438_438031

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438031


namespace find_m_l438_438195

-- Definitions for the lines and the condition of parallelism
def line1 (m : ℝ) (x y : ℝ): Prop := x + m * y + 6 = 0
def line2 (m : ℝ) (x y : ℝ): Prop := 3 * x + (m - 2) * y + 2 * m = 0

-- Condition for lines being parallel
def parallel_lines (m : ℝ) : Prop := 1 * (m - 2) - 3 * m = 0

-- Main formal statement
theorem find_m (m : ℝ) (h1 : ∀ x y, line1 m x y)
                (h2 : ∀ x y, line2 m x y)
                (h_parallel : parallel_lines m) : m = -1 :=
sorry

end find_m_l438_438195


namespace retire_old_cars_each_year_l438_438066

theorem retire_old_cars_each_year :
  ∃ x : ℕ, (∀ f : ℕ, f = 20 ∧ (f - x * 2) < (f / 2) ∧ x > 0 -> x = 6) :=
begin
  sorry
end

end retire_old_cars_each_year_l438_438066


namespace cyclic_inequality_l438_438305

variable (a b c : ℝ)

noncomputable def ab : ℝ := a * b
noncomputable def a5 : ℝ := a ^ 5
noncomputable def b5 : ℝ := b ^ 5
noncomputable def sum_cyc : ℝ := (ab / (ab + a5 + b5)) + (c * a / (c * a + c ^ 5 + a ^ 5)) + (b * c / (b * c + b ^ 5 + c ^ 5))

theorem cyclic_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) : sum_cyc a b c ≤ 1 :=
by
  sorry

end cyclic_inequality_l438_438305


namespace sequence_function_problem_l438_438230

theorem sequence_function_problem
  (a : ℕ → ℤ)
  (f : ℤ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (hrec : ∀ n : ℕ, n > 0 → a (n+2) = a (n+1) - a n)
  (f_def : ∀ x : ℤ, f x = a * x^3 + b * Int.cast (Real.tan x)) 
  (hfa4 : f (a 4) = 9) :
  f (a 1) + f (a 2017) = -18 :=
sorry

end sequence_function_problem_l438_438230


namespace exists_segment_satisfying_condition_l438_438959

theorem exists_segment_satisfying_condition :
  ∃ (x₁ x₂ x₃ : ℚ) (f : ℚ → ℤ), x₃ = (x₁ + x₂) / 2 ∧ f x₁ + f x₂ ≤ 2 * f x₃ :=
sorry

end exists_segment_satisfying_condition_l438_438959


namespace percentage_decrease_l438_438667

theorem percentage_decrease (original_price new_price : ℝ) (h1 : original_price = 1400) (h2 : new_price = 1064) :
  ((original_price - new_price) / original_price * 100) = 24 :=
by
  sorry

end percentage_decrease_l438_438667


namespace fraction_zero_solution_l438_438949

theorem fraction_zero_solution (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x + 3 ≠ 0) : x = 1 :=
by
  sorry

end fraction_zero_solution_l438_438949


namespace total_wet_surface_area_is_correct_l438_438040

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct_l438_438040


namespace chord_length_on_circle_l438_438365

theorem chord_length_on_circle
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y = 0)
  (line_eq : ∀ x y : ℝ, x + 2 * y - 5 + real.sqrt 5 = 0) :
  ∃ l : ℝ, l = 4 :=
by
  sorry -- Proof goes here

end chord_length_on_circle_l438_438365


namespace initial_blocks_l438_438635

-- Definitions of the given conditions
def blocks_eaten : ℕ := 29
def blocks_remaining : ℕ := 26

-- The statement we need to prove
theorem initial_blocks : blocks_eaten + blocks_remaining = 55 :=
by
  -- Proof is not required as per instructions
  sorry

end initial_blocks_l438_438635


namespace no_valid_arrangement_l438_438026

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438026


namespace P_abscissa_range_l438_438212

noncomputable def satisfies_range (x : ℝ) : Prop :=
  x ∈ Icc (1 / 5) 1

theorem P_abscissa_range (a : ℝ) (h1 : ∃ y, y = 2 * a) (h2 : (a - 3) ^ 2 + (2 * a) ^ 2 = 8) :
  satisfies_range a :=
sorry

end P_abscissa_range_l438_438212


namespace find_coterminal_angle_l438_438584

noncomputable def same_terminal_side (θ : ℝ) (θ' : ℝ) (interval : set ℝ) : Prop :=
  ∃ k : ℤ, θ' = θ + k * (2 * Real.pi) ∧ θ' ∈ interval

theorem find_coterminal_angle :
  same_terminal_side (-3 * Real.pi / 4) (5 * Real.pi / 4) (set.Ico 0 (2 * Real.pi)) :=
by
  sorry

end find_coterminal_angle_l438_438584


namespace athlete_heart_beats_l438_438438

theorem athlete_heart_beats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) (total_beats : ℕ) :
  heart_rate = 120 → 
  pace = 6 → 
  distance = 30 →
  total_beats = 21600 := 
by {
  intros,
  sorry
}

end athlete_heart_beats_l438_438438


namespace problem1_problem2_l438_438909

-- Define f and g functions
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - (1 - a) * x - (2 - a) * Real.log x

-- Problem 1: Prove that for g(x) to be increasing in its domain, a must be in [2, +∞)
theorem problem1 (a : ℝ) : (∀ x : ℝ, 0 < x → deriv (g a) x ≥ 0) ↔ (2 ≤ a) := by
  sorry

-- Define F as f - g
def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Problem 2: Prove that F(x) at (x₀, F(x₀)) cannot have a tangent line parallel to the x-axis
-- if the intersection points A and B of F(x) with the x-axis have the midpoint x₀
theorem problem2 (a : ℝ) (x₀ : ℝ) : 
  (∃ A B : ℝ, A < x₀ ∧ x₀ < B ∧ F a A = 0 ∧ F a B = 0 ∧ (A + B) / 2 = x₀) →
  ¬(deriv (F a) x₀ = 0) := by
  sorry

end problem1_problem2_l438_438909


namespace complex_division_l438_438055

def i_units := Complex.I

def numerator := (3 : ℂ) + i_units
def denominator := (1 : ℂ) + i_units
def expected_result := (2 : ℂ) - i_units

theorem complex_division :
  numerator / denominator = expected_result :=
by sorry

end complex_division_l438_438055


namespace seating_arrangements_24_l438_438467

-- Define the students and their grades
inductive Grade | Freshman | Sophomore | Junior | Senior

-- Define the list of students with their grades
structure Student :=
(grade : Grade)

-- Define the eight students
def students : List Student :=
[
    ⟨Grade.Freshman⟩, ⟨Grade.Freshman⟩, -- twin sisters
    ⟨Grade.Sophomore⟩, ⟨Grade.Sophomore⟩,
    ⟨Grade.Junior⟩, ⟨Grade.Junior⟩,
    ⟨Grade.Senior⟩, ⟨Grade.Senior⟩
]

-- Define the cars
structure Car :=
(capacity : Nat)
(max_capacity : capacity ≤ 4)

def carA : Car := {capacity := 4, max_capacity := Nat.le_refl 4}
def carB : Car := {capacity := 4, max_capacity := Nat.le_refl 4}

-- Define the condition: twin sisters must ride in the same car
def twin_sisters_in_same_car (car : Car) (students : List Student) : Prop :=
List.countp (λ s => s.grade = Grade.Freshman) students = 2

-- Define the condition: exactly two students from the same grade in car A
def exactly_two_from_same_grade_in_carA (students : List Student) : Prop :=
∃ g : Grade, List.countp (λ s => s.grade = g) students = 2 ∧
List.countp (λ s => s.grade = g) (students.filter (λ s => s ∉ carA_students)) = 2

-- Translate the mathematical problem to a Lean theorem
theorem seating_arrangements_24 :
  ∃ carA_students carB_students : List Student,
    carA_students.length = 4 ∧
    carB_students.length = 4 ∧
    twin_sisters_in_same_car carA students ∧
    exactly_two_from_same_grade_in_carA carA_students ∧
    24 = List.length (List.permutations carA_students) /\
    24 = List.length (List.permutations carB_students) :=
sorry

end seating_arrangements_24_l438_438467


namespace Jessica_has_3_dozens_l438_438985

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l438_438985


namespace find_s_l438_438130

def F (p q r : ℝ) : ℝ := p * q^r

theorem find_s (s : ℝ) : F s s 4 = 625 ↔ s = 5^0.8 :=
by
  unfold F
  split
  sorry

end find_s_l438_438130


namespace max_value_on_interval_l438_438160

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem max_value_on_interval : ∀ x ∈ set.Icc (1:ℝ) (2:ℝ), f(x) ≤ 0 :=
by
  sorry

end max_value_on_interval_l438_438160


namespace correct_statements_l438_438392

/--
Given:
- A model with a smaller sum of squared residuals has a larger \( R^2 \), indicating a better fit.
- A larger sum of squared residuals results in a smaller \( R^2 \), indicating a worse model fit.

Prove that under these conditions, the correct statements are \( (3) \) and \( (4) \),
i.e., the correct answer choice is B.
-/
theorem correct_statements : 
  ( (∀ {S1 S2 R1 R2 : ℝ}, (S1 > S2) → (R1 < R2) → 
        (R1 = sum_of_squared_residuals_to_coefficient_of_determination S1) ∧ 
        (R2 = sum_of_squared_residuals_to_coefficient_of_determination S2) →
        sum_of_squared_residuals_to_coefficient_of_determination S1 = false)
  ∨
  (∀ {S1 S2 R1 R2 : ℝ}, (S1 < S2) → (R1 > R2) → 
        (R1 = sum_of_squared_residuals_to_coefficient_of_determination S1) ∧ 
        (R2 = sum_of_squared_residuals_to_coefficient_of_determination S2) →
        sum_of_squared_residuals_to_coefficient_of_determination S2 = true))
  ∨
  ( (∀ {S1 S2 R1 R2 : ℝ}, (S1 < S2) → (R1 > R2) → 
        (R1 = sum_of_squared_residuals_to_coefficient_of_determination S1) ∧ 
        (R2 = sum_of_squared_residuals_to_coefficient_of_determination S2) →
        better_model_fit R1 = true)
  ∧
  (∀ {S1 S2 R1 R2 : ℝ}, (S1 > S2) → (R1 < R2) → 
        (R1 = sum_of_squared_residuals_to_coefficient_of_determination S1) ∧ 
        (R2 = sum_of_squared_residuals_to_coefficient_of_determination S2) →
        worse_model_fit R1 = true) )
  :=
sorry

end correct_statements_l438_438392


namespace population_in_terms_of_t_l438_438255

noncomputable def boys_girls_teachers_total (b g t : ℕ) : ℕ :=
  b + g + t

theorem population_in_terms_of_t (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 5 * t) :
  boys_girls_teachers_total b g t = 26 * t :=
by
  sorry

end population_in_terms_of_t_l438_438255


namespace seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l438_438913

-- Problem 1
theorem seq_inv_an_is_arithmetic (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) :
  ∃ d, ∀ n, n ≥ 2 → (1 / a n) = 2 + (n - 1) * d :=
sorry

-- Problem 2
theorem seq_fn_over_an_has_minimum (a f : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) (h3 : ∀ n, f n = (9 / 10) ^ n) :
  ∃ m, ∀ n, n ≠ m → f n / a n ≥ f m / a m :=
sorry

end seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l438_438913


namespace student_ticket_cost_l438_438633

theorem student_ticket_cost 
    (total_tickets : ℕ) 
    (total_income : ℝ) 
    (ron_tickets : ℕ) 
    (adult_ticket_cost : ℝ) 
    (remaining_tickets := total_tickets - ron_tickets) 
    (adult_income := remaining_tickets * adult_ticket_cost)
    (student_income := total_income - adult_income) 
    (ron_ticket_cost := student_income / ron_tickets)
    : total_tickets = 20 
    → total_income = 60 
    → ron_tickets = 12 
    → adult_ticket_cost = 4.5 
    → ron_ticket_cost = 2 := by
  intros h1 h2 h3 h4
  rw [h1] at remaining_tickets
  rw [h3, h4] at remaining_tickets adult_income student_income
  rw [h2] at student_income
  field_simp at student_income
  sorry

end student_ticket_cost_l438_438633


namespace heather_initial_oranges_l438_438923

theorem heather_initial_oranges (given_oranges: ℝ) (total_oranges: ℝ) (initial_oranges: ℝ) 
    (h1: given_oranges = 35.0) 
    (h2: total_oranges = 95) : 
    initial_oranges = 60 :=
by
  sorry

end heather_initial_oranges_l438_438923


namespace permutation_count_l438_438291

open Finset

theorem permutation_count :
  ∃ (b : Fin 10 → Fin 10),
    (injective b) ∧
    (∀ i j : Fin 5, i < j → b i > b j) ∧
    (∀ i j : Fin 5, i < j → b ⟨i+5, sorry⟩ < b ⟨j+5, sorry⟩) ∧
    (b 4 = 0) ∧
    (univ.card = 126) :=
sorry

end permutation_count_l438_438291


namespace inequality_holds_for_n_ge_0_l438_438148

theorem inequality_holds_for_n_ge_0
  (n : ℤ)
  (h : n ≥ 0)
  (a b c x y z : ℝ)
  (Habc : 0 < a ∧ 0 < b ∧ 0 < c)
  (Hxyz : 0 < x ∧ 0 < y ∧ 0 < z)
  (Hmax : max a (max b (max c (max x (max y z)))) = a)
  (Hsum : a + b + c = x + y + z)
  (Hprod : a * b * c = x * y * z) : a^n + b^n + c^n ≥ x^n + y^n + z^n := 
sorry

end inequality_holds_for_n_ge_0_l438_438148


namespace final_cost_correct_l438_438112

-- Definitions based on conditions
def engine_oil_filter_cost : Float := 15
def engine_oil_filter_quantity : Int := 5
def brake_pads_total_cost : Float := 225
def air_filter_cost : Float := 40
def air_filter_quantity : Int := 2
def total_discount : Float → Float
  | cost if cost > 300 => (0.10 * (engine_oil_filter_cost * engine_oil_filter_quantity)) + (0.05 * (air_filter_cost * air_filter_quantity))
  | _ => 0
def sales_tax_rate : Float := 0.08

-- Helper function to calculate final cost
noncomputable def final_cost (before_discount : Float) : Float :=
  let discount := total_discount before_discount
  let after_discount := before_discount - discount
  let tax := sales_tax_rate * after_discount
  after_discount + tax

-- Condition to calculate total cost before discount
def total_cost_before_discount : Float :=
  (engine_oil_filter_cost * Float.ofInt engine_oil_filter_quantity) +
  brake_pads_total_cost +
  (air_filter_cost * Float.ofInt air_filter_quantity)

-- Proof that the final cost is $397.98
theorem final_cost_correct : final_cost total_cost_before_discount = 397.98 :=
by
  have h_before_discount : total_cost_before_discount = 380 := by norm_num
  have h_discount : total_discount total_cost_before_discount = 11.50 := by norm_num
  have h_after_discount : total_cost_before_discount - h_discount = 368.50 := by norm_num
  have h_tax : sales_tax_rate * 368.50 = 29.48 := by norm_num
  have h_total_cost : final_cost total_cost_before_discount = 368.50 + 29.48 := by
    rw [final_cost, h_before_discount, h_discount, h_after_discount, h_tax]
  exact h_total_cost.trans (by norm_num)

end final_cost_correct_l438_438112


namespace find_cos_l438_438209

theorem find_cos (θ : ℝ) 
  (h1 : θ ∈ Ioo (π / 2) π) 
  (h2 : 1 / sin θ + 1 / cos θ = 2 * sqrt 2) : 
  cos (2 * θ + π / 3) = sqrt 3 / 2 :=
sorry

end find_cos_l438_438209


namespace complex_conjugate_example_l438_438895

theorem complex_conjugate_example : 
  let z := Complex.div (Complex.mk (-1) (-2)) (Complex.pow (Complex.mk 1 1) 2) 
  in Complex.conj z = Complex.mk (-1) (-1/2) := by
    let z : ℂ := Complex.div (Complex.mk (-1) (-2)) (Complex.pow (Complex.mk 1 1) 2)
    have h : Complex.conj z = Complex.mk (-1) (-1/2) := by
        sorry
    exact h

end complex_conjugate_example_l438_438895


namespace number_of_band_students_l438_438261

noncomputable def total_students := 320
noncomputable def sports_students := 200
noncomputable def both_activities_students := 60
noncomputable def either_activity_students := 225

theorem number_of_band_students : 
  ∃ B : ℕ, either_activity_students = B + sports_students - both_activities_students ∧ B = 85 :=
by
  sorry

end number_of_band_students_l438_438261


namespace ellipse_equation_dot_product_range_intersect_x_at_fixed_point_l438_438872

noncomputable def ellipse_C : Type :=
  {a b : ℝ // a > b ∧ b > 0 ∧ (∃ c : ℝ, c = a / 2) ∧ 
  (∃ b : ℝ, (b = √3) ∧ (a^2 = b^2 + (a / 2)^2)) ∧ 
  (∀ x y : ℝ, x - y + √6 = 0 → (√3 = √(x^2 + y^2)))}

theorem ellipse_equation (C : ellipse_C) :
 (∃ a b : ℝ, a = 2 ∧ b = √3 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) :=
sorry

def intersects_ellipse : Type := (P : ℝ × ℝ) ∧ 
(∀ k : ℝ, (k ≠ 0) ∧ (P = (4, 0)) ∧ (∃ A B : ℝ × ℝ, 
  (A ≠ B) ∧
  ∀ x y : ℝ, (y = k * (x - 4)) ∧ (x^2 / 4 + y^2 / 3 = 1) →
  (∃ x1 x2 y1 y2 : ℝ, (0 ≤ k^2 < 1/4) ∧
   (x1 + x2 = 32 * k^2 / (3 + 4 * k^2)) ∧
   (x1 * x2 = (64 * k^2 - 12) / (3 + 4 * k^2)) ∧
   (y1 = k * (x1 - 4)) ∧
   (y2 = k * (x2 - 4)) ∧
   (x1 * x2 + y1 * y2 = (25 - 87 / (4 * k^2 + 3))))) → 
   (∃ d : ℝ, d ∈ [-4, 13/4])

theorem dot_product_range (l : intersects_ellipse) :
  ∃ d : ℝ, d ∈ [-4, 13/4] :=
sorry

def symmetric_about_x (B: ℝ × ℝ) : Type :=
  ∃ E : ℝ × ℝ, (E = (B.1, -B.2)) ∧
  (∀ A : ℝ × ℝ, ∃ x : ℝ, 
    x = 1 ∧ (∃ l : intersects_ellipse, l ∧
    (line_through A E → x-axis ∩ (1, 0))))

theorem intersect_x_at_fixed_point (B : ℝ × ℝ) :
  symmetric_about_x B :=
sorry

end ellipse_equation_dot_product_range_intersect_x_at_fixed_point_l438_438872


namespace floor_factorial_expression_l438_438451

-- Mathematical definitions (conditions)
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Mathematical proof problem (statement)
theorem floor_factorial_expression :
  Int.floor ((factorial 2007 + factorial 2004 : ℚ) / (factorial 2006 + factorial 2005)) = 2006 :=
sorry

end floor_factorial_expression_l438_438451


namespace find_a_b_min_modulus_z_l438_438342

/-- Part 1: Determine the values of a and b -/
theorem find_a_b (a b : ℝ)
  (h1 : ∀ x, (x^2 - (6 + complex.I) * x + 9 + a * complex.I = 0) → x = b) :
  a = 3 ∧ b = 3 :=
sorry

/-- Part 2: Find the minimum modulus of |z| -/
theorem min_modulus_z (z : ℂ)
  (z1 : ℂ := 2 / (1 + complex.I))
  (h : |z - (3 + 3 * complex.I)| = |z1|) :
  ∃ w : ℝ, w = 2 * real.sqrt 2 ∧ |z| = w :=
sorry

end find_a_b_min_modulus_z_l438_438342


namespace closest_integer_to_sum_l438_438158

theorem closest_integer_to_sum :
  let S := 500 * ∑ n in Finset.range 4999 \ Finset.singleton 0, 1 / (n + 2)^2 - 1 
  375 = Int.floor S :=
by
  sorry

end closest_integer_to_sum_l438_438158


namespace log_ordering_l438_438558

theorem log_ordering (x y z : ℝ) :
  log 2 (log (1/2) (log 2 x)) = 0 ∧
  log 3 (log (1/3) (log 3 y)) = 0 ∧
  log 5 (log (1/5) (log 5 z)) = 0 →
  x = 2 ∧ y = 3 ∧ z = 5 ∧ x < y ∧ y < z :=
by
  intro h
  sorry

end log_ordering_l438_438558


namespace solve_quadratic_vertex_of_parabola_l438_438641
noncomputable def x_values := 
  {x : ℝ // x^2 + 4 * x - 2 = 0}

theorem solve_quadratic (x : ℝ) (h : x ∈ x_values) : 
  x = -2 + Real.sqrt 6 ∨ x = -2 - Real.sqrt 6 :=
sorry

theorem vertex_of_parabola (x y : ℝ) 
  (h : y = 2 * x^2 - 4 * x + 6) : 
  (x = 1 ∧ y = 4) :=
sorry

end solve_quadratic_vertex_of_parabola_l438_438641


namespace coins_weight_probability_l438_438348

noncomputable def probability_four_genuine_given_equal_weights_of_pairs : ℚ := 
  207 / 581

theorem coins_weight_probability :
  (∃ (coins : Fin 30 → ℝ), 
    (∀ i j, i ≠ j → coins i ≠ coins j) ∧ -- all coins have distinct weights
    (∃ (count_genuine count_counterfeit : Fin 30 → Prop),
      (∑ i, if count_genuine i then 1 else 0 = 20 ∧ ∑ i, if count_counterfeit i then 1 else 0 = 10) ∧
      (∀ i, count_genuine i ↔ i ∈ Set.range 20) ∧
      (∀ i, count_counterfeit i ↔ i ∉ Set.range 20) ∧
      -- Define the event of selecting two pairs and checking conditions
      (∀ (p1 p2 : (Fin 30) × (Fin 30)),
        p1.fst ≠ p1.snd ∧ p2.fst ≠ p2.snd ∧ p1 ≠ p2 →
        (coins p1.fst + coins p1.snd = coins p2.fst + coins p2.snd → 
          (probability_four_genuine_given_equal_weights_of_pairs = 207 / 581)))) := sorry

end coins_weight_probability_l438_438348


namespace probability_of_same_color_picks_l438_438070

def terry_pick_prob : ℚ := (nat.choose 12 2) / (nat.choose 24 2)
def mary_pick_prob_given_terry_red : ℚ := (nat.choose 10 3) / (nat.choose 22 3)
def mary_pick_prob_given_terry_blue : ℚ := (nat.choose 10 3) / (nat.choose 22 3)

theorem probability_of_same_color_picks : 
  let p_same_color := 2 * (terry_pick_prob * mary_pick_prob_given_terry_red)
  in p_same_color = (66 : ℚ) / 1771 := 
by 
  sorry

example : let m := 66
          let n := 1771
          in m + n = 1837 := 
by 
  rfl

end probability_of_same_color_picks_l438_438070


namespace correct_propositions_l438_438896

variables {a b : Type} [LinearSpace a] [LinearSpace b]
variables {alpha beta : SubPlane a}

-- Definitions of perpendicularity and parallelism between lines and planes
def perp {a b : Type} [LinearSpace a] [LinearSpace b] (l m : a) : Prop := sorry
def parallel {a : Type} [LinearSpace a] (l : a) (pi : SubPlane a) : Prop := sorry
def lies_on {a : Type} [LinearSpace a] (l : a) (pi : SubPlane a) : Prop := sorry

-- The four conditions as hypotheses
axiom h1 : ∀ (a b : a) (alpha : SubPlane a), perp a b → perp b alpha → parallel a alpha
axiom h2 : ∀ (a : a) (alpha beta : SubPlane a) (b : a), parallel a alpha → parallel alpha beta → perp b beta → perp a b
axiom h3 : ∀ (a b : a) (alpha : SubPlane a), ¬ lies_on a alpha → parallel a b → lies_on b alpha → parallel a alpha
axiom h4 : ∀ (a : a) (alpha beta : SubPlane a), parallel a alpha → perp alpha beta → perp a beta

-- The statement that verifies the correct propositions
theorem correct_propositions : (h2 ∧ h3) := sorry

end correct_propositions_l438_438896


namespace sphere_intersection_circle_radius_l438_438427

theorem sphere_intersection_circle_radius
  (x1 y1 z1: ℝ) (x2 y2 z2: ℝ) (r1 r2: ℝ)
  (hyp1: x1 = 3) (hyp2: y1 = 5) (hyp3: z1 = 0) 
  (hyp4: r1 = 2) 
  (hyp5: x2 = 0) (hyp6: y2 = 5) (hyp7: z2 = -8) :
  r2 = Real.sqrt 59 := 
by
  sorry

end sphere_intersection_circle_radius_l438_438427


namespace windows_ways_l438_438414

theorem windows_ways (n : ℕ) (h : n = 8) : (n * (n - 1)) = 56 :=
by
  sorry

end windows_ways_l438_438414


namespace number_of_routes_l438_438752

-- Definition of the given problem conditions
def cities : Type := Fin 20  -- A finite set of 20 cities
def roads : Finset (cities × cities) := sorry -- Placeholder for the finite set of 30 roads
def start_city : cities := sorry -- Placeholder for city X
def end_city : cities := sorry -- Placeholder for city Y
def road_travelled (path : List (cities × cities)) : Prop :=
  (List.length path = 17) ∧
  (list.nodup path) ∧
  (path.head = start_city) ∧
  (path.last = end_city)

-- Theorem statement
theorem number_of_routes : 
  (∃ (path : List (cities × cities)), road_travelled path) → 
  (number_of_valid_routes = 5) :=
sorry

end number_of_routes_l438_438752


namespace points_fixed_distance_from_plane_l438_438258

-- Define the condition in the problem
def points_fixed_distance_from_line (line : ℝ × ℝ → Prop) (d : ℝ) : set (ℝ × ℝ) := {
  p | ∃ p1 p2, (line p1 ∧ line p2 ∧ dist p p1 = d ∧ dist p p2 = d ∧ parallel p1 p2)
}

-- Theorem statement
theorem points_fixed_distance_from_plane (plane : ℝ × ℝ × ℝ → Prop) (d : ℝ) : set (ℝ × ℝ × ℝ) := {
  q | ∃ q1 q2, (plane q1 ∧ plane q2 ∧ dist q q1 = d ∧ dist q q2 = d ∧ parallel q1 q2)
}

end points_fixed_distance_from_plane_l438_438258


namespace percentage_decrease_l438_438420

theorem percentage_decrease (x : ℝ) (h : x > 0) : ∃ p : ℝ, p = 0.20 ∧ ((1.25 * x) * (1 - p) = x) :=
by
  sorry

end percentage_decrease_l438_438420


namespace molecular_weight_correct_l438_438786

-- Define the atomic weights
def atomic_weight_nitrogen : Float := 14.01
def atomic_weight_oxygen : Float := 16.00

-- Define the number of atoms in the molecule N₂O₃
def number_of_nitrogen_atoms : Nat := 2
def number_of_oxygen_atoms : Nat := 3

-- Define the molecular weight calculation
def molecular_weight_dinitrogen_trioxide : Float := 
  (number_of_nitrogen_atoms * atomic_weight_nitrogen) +
  (number_of_oxygen_atoms * atomic_weight_oxygen)

-- The theorem to state the molecular weight of N₂O₃ is 76.02 g/mol
theorem molecular_weight_correct : molecular_weight_dinitrogen_trioxide = 76.02 := by
  -- Skipping the proof
  sorry

end molecular_weight_correct_l438_438786


namespace correct_statement_D_l438_438435

namespace MathProofs

/-- Definitions of the statements A, B, C, and D based on the conditions in the problem. --/

def statement_A (k : ℝ) (K2 : ℝ) (X Y : Type) [random_variable K2 X] [random_variable K2 Y] : Prop :=
  k > 0 → credibility_of_relation X Y (k) = "smaller"

def statement_B (x y : ℝ) (x_certain : x ∈ set_of_certain_values) (y_random : y ∈ set_of_random_values) : Prop :=
  nondeterministic_relationship x y = "function relationship"

def statement_C (r2 : ℝ) (X Y : Type) [random_variable r2 X] [random_variable r2 Y] : Prop :=
  0 ≤ r2 ∧ r2 ≤ 1 → correlation_strength r2 X Y "weaker" 

def statement_D (k : ℝ) (K2 : ℝ) (X Y : Type) [random_variable K2 X] [random_variable K2 Y] : Prop :=
  k < 0 → certainty_of_relation X Y (k) = "smaller"

theorem correct_statement_D (k : ℝ) (K2 : ℝ) (X Y : Type) 
  [random_variable K2 X] 
  [random_variable K2 Y]
  (k_nonneg : k ≥ 0):
  ¬(statement_A k K2 X Y) ∧ ¬(statement_B k K2 X Y) ∧ ¬(statement_C k K2 X Y) ∧ (statement_D k K2 X Y) :=
begin
  sorry
end

end MathProofs

end correct_statement_D_l438_438435


namespace finding_f_of_neg_half_l438_438541

def f (x : ℝ) : ℝ := sorry

theorem finding_f_of_neg_half : f (-1/2) = Real.pi / 3 :=
by
  -- Given function definition condition: f (cos x) = x / 2 for 0 ≤ x ≤ π
  -- f should be defined on ℝ -> ℝ such that this condition holds;
  -- Applying this condition should verify our theorem.
  sorry

end finding_f_of_neg_half_l438_438541


namespace determine_a_l438_438102

-- Given conditions
variable {a b : ℝ}
variable (h_neg : a < 0) (h_pos : b > 0) (h_max : ∀ x, -2 ≤ a * sin (b * x) ∧ a * sin (b * x) ≤ 2)

-- Statement to prove
theorem determine_a : a = -2 := by
  sorry

end determine_a_l438_438102


namespace round_52_63847_is_52_64_l438_438336

-- Define the given number
def number : ℝ := 52.63847

-- Define the rounding function to the nearest hundredth
def round_to_nearest_hundredth (n : ℝ) : ℝ :=
  (Real.floor (n * 100 + 0.5)) / 100

-- State the theorem
theorem round_52_63847_is_52_64 : round_to_nearest_hundredth number = 52.64 :=
by
  -- Proof goes here
  sorry

end round_52_63847_is_52_64_l438_438336


namespace min_tosses_one_head_l438_438067

theorem min_tosses_one_head (n : ℕ) (P : ℝ) (h₁ : P = 1 - (1 / 2) ^ n) (h₂ : P ≥ 15 / 16) : n ≥ 4 :=
by
  sorry -- Proof to be filled in.

end min_tosses_one_head_l438_438067


namespace n_multiple_of_40_and_infinite_solutions_l438_438628

theorem n_multiple_of_40_and_infinite_solutions 
  (n : ℤ)
  (h1 : ∃ k₁ : ℤ, 2 * n + 1 = k₁^2)
  (h2 : ∃ k₂ : ℤ, 3 * n + 1 = k₂^2)
  : ∃ (m : ℤ), n = 40 * m ∧ ∃ (seq : ℕ → ℤ), 
    (∀ i : ℕ, ∃ k₁ k₂ : ℤ, (2 * (seq i) + 1 = k₁^2) ∧ (3 * (seq i) + 1 = k₂^2) ∧ 
     (i ≠ 0 → seq i ≠ seq (i - 1))) :=
by sorry

end n_multiple_of_40_and_infinite_solutions_l438_438628


namespace weight_7_moles_AlI3_l438_438383

-- Definitions from the conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_I : ℝ := 126.90
def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ := moles * molecular_weight

-- Theorem stating the weight of 7 moles of AlI3
theorem weight_7_moles_AlI3 : 
  weight_of_compound 7 molecular_weight_AlI3 = 2853.76 :=
by
  -- Proof will be added here
  sorry

end weight_7_moles_AlI3_l438_438383


namespace arithmetic_sequence_general_term_sum_reciprocal_inequality_l438_438870

variables {a : ℕ → ℤ} {d : ℤ} {S : ℕ → ℤ} {T : ℕ → ℚ}

noncomputable def general_term := ∀ n : ℕ, a n = 2 * n + 1

noncomputable def sequence_conditions := 
  (a 5 = 11) ∧ 
  (d ≠ 0) ∧ 
  (S 9 = 99) ∧
  ((a 7) ^ 2 = (a 4) * (a 12)) ∧
  (∀ n, S n = n * (a 1 + a n) / 2)

theorem arithmetic_sequence_general_term 
  (h : sequence_conditions) : general_term :=
sorry

theorem sum_reciprocal_inequality 
  (h : ∀ n, S n = n * (a 1 + a n) / 2) : 
  ∀ n, T n = (1 / S 1 + 1 / S 2 + ... + 1 / S n) → T n < 3 / 4 :=
sorry

end arithmetic_sequence_general_term_sum_reciprocal_inequality_l438_438870


namespace mary_tea_count_l438_438140

/-
Problem statement
During her six-day workweek, Mary buys a beverage either coffee priced at 60 cents or tea priced at 80 cents. Twice a week, she also buys a 40-cent cookie along with her beverage. Her total cost for the week amounts to a whole number of dollars. Prove that Mary bought tea three times.
-/

theorem mary_tea_count (t c : ℕ) (h1 : c + t = 6) (h2 : 80 * t + 60 * c + 2 * 40 = k * 100) : 
  t = 3 :=
by
  assume k : ℕ
  sorry

end mary_tea_count_l438_438140


namespace complete_the_square_l438_438700

theorem complete_the_square (x : ℝ) : ∃ p q : ℝ, (x - 3)^2 = 4 :=
by {
  let a := (x^2 - 6x + 5),          -- Define the quadratic
  have s := (x^2 - 6x + 9 + 5 - 9), -- Adding and subtracting 9
  dsimp at s,
  have h_complete := (x - 3)^2,     -- Completing the square
  use [(-3), 4],
  exact eq.symm h_complete -- Ensure the symmetry for equality
}

end complete_the_square_l438_438700


namespace card_arrangement_impossible_l438_438033

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438033


namespace set_characteristics_l438_438703

-- Define the characteristics of elements in a set
def characteristic_definiteness := true
def characteristic_distinctness := true
def characteristic_unorderedness := true
def characteristic_reality := false -- We aim to prove this

-- The problem statement in Lean
theorem set_characteristics :
  ¬ characteristic_reality :=
by
  -- Here would be the proof, but we add sorry as indicated.
  sorry

end set_characteristics_l438_438703


namespace cook_is_innocent_l438_438100

variable (Person : Type) (stole_pepper : Person → Prop) (tells_truth : Person → Prop)
variable (cook : Person)
variable (knows_thief : Person → Prop)

-- Define the conditions
def always_lies_if_steals (p : Person) : Prop :=
  stole_pepper p → ¬ tells_truth p

def cook_statement (p : Person) : Prop :=
  tells_truth cook ∧ knows_thief cook

theorem cook_is_innocent 
  (H1 : always_lies_if_steals Person stole_pepper tells_truth)
  (H2 : cook_statement cook tells_truth knows_thief) : ¬ stole_pepper cook :=
by
  -- Proof goes here
  sorry

end cook_is_innocent_l438_438100


namespace max_good_cells_l438_438302

theorem max_good_cells (n : ℕ) (grid : Fin n → Fin n → ℝ) (h_n : 0 < n) :
  ∃ m, m = (n - 1) * (n - 1) ∧
  ∀ c : Fin n × Fin n, 
    (grid c.1 c.2 > (∑ i, grid i c.2) / n
    ∧ grid c.1 c.2 < (∑ j, grid c.1 j) / n)
    → m = (n - 1) * (n - 1) := 
sorry

end max_good_cells_l438_438302


namespace largest_is_21_l438_438747

theorem largest_is_21(a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29):
  d = 21 := 
sorry

end largest_is_21_l438_438747


namespace max_students_with_equal_distribution_l438_438661

-- Defining the problem statement using Lean 4
theorem max_students_with_equal_distribution (pens : ℕ) (pencils : ℕ) :
  pens = 2010 → pencils = 1050 → Nat.gcd pens pencils = 30 := by
  intros h1 h2
  rw [h1, h2]
  exact Nat.gcd_eq_right 2010 1050 (by norm_num [Nat.gcd])

end max_students_with_equal_distribution_l438_438661


namespace Marcel_paid_58_30_l438_438316

def price_of_pen := 4
def price_of_briefcase := 5 * price_of_pen
def price_of_notebook := 2 * price_of_pen
def price_of_calculator := 3 * price_of_notebook
def discount_rate := 0.15
def tax_rate := 0.10

def discounted_price_of_briefcase := price_of_briefcase * (1 - discount_rate)
def total_cost_before_tax := price_of_pen + discounted_price_of_briefcase + price_of_notebook + price_of_calculator
def total_tax := total_cost_before_tax * tax_rate
def total_amount_paid := total_cost_before_tax + total_tax

theorem Marcel_paid_58_30 :
  total_amount_paid = 58.30 := by
  sorry

end Marcel_paid_58_30_l438_438316


namespace find_p_q_l438_438583

noncomputable section

open Real

variables (OA OB OC : ℝ → ℝ → ℝ) (p q : ℝ)

-- Given conditions
def norm_OA : ℝ := 2
def norm_OB : ℝ := 3
def norm_OC : ℝ := 2 * sqrt 5
def tan_angle_AOC : ℝ := 2
def angle_BOC : ℝ := π / 3

-- Definitions of vectors (norm conditions only for completeness)
def norm (v : ℝ → ℝ → ℝ) : ℝ := sqrt (v 1 0 * v 1 0 + v 0 1 * v 0 1)

axiom OA_def : norm OA = norm_OA
axiom OB_def : norm OB = norm_OB
axiom OC_def : norm OC = norm_OC

-- Prove the relationship
theorem find_p_q :
  ∃ (p q : ℝ), OC = λ x y, p * OA x y + q * OB x y ∧ p = 5 / 2 ∧ q = 3 * sqrt 5 / 2 :=
sorry

end find_p_q_l438_438583


namespace min_major_axis_length_l438_438366

theorem min_major_axis_length (a b c : ℝ) (h_area : b * c = 1) (h_focal_relation : 2 * a = 2 * Real.sqrt (b^2 + c^2)) :
  2 * a = 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_length_l438_438366


namespace common_ratio_and_sum_l438_438671

noncomputable def sum_arithmetic_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem common_ratio_and_sum 
  (a₁ S₁ S₂ S₃ : ℝ)
  (h1: S₁ = a₁)
  (h2: S₂ = a₁ * (1 + q) / (1 - q))
  (h3: S₃ = a₁ * (1 + q + q^2) / (1 - q))
  (h4: S₁ + S₃ = 2 * S₂)
  (h5: a₁ - a₁ * q^2 = 3)
  (q = -1/2) :
  sum_arithmetic_sequence a₁ q n = (8/3) * (1 - (-1/2)^n) :=
sorry

end common_ratio_and_sum_l438_438671


namespace binomial_constant_term_eq_l438_438269

theorem binomial_constant_term_eq (a : ℝ) : 
  (∃ x : ℝ, x ∈ {1..} ∧ ∀ (k : ℕ), nat.choose 5 k * a^(5 - k) * x^(2*(5 - k) - k/2) = 5*k ↔ k = 4) → a = -2 :=
by
  sorry

end binomial_constant_term_eq_l438_438269


namespace arith_geo_sum_of_reciprocal_terms_l438_438576

theorem arith_geo_sum_of_reciprocal_terms :
  (∃ d : ℤ, d ≠ 0 ∧ ∀ n : ℕ, aₙ n = 2 + (n - 1) * d) ∧
  (∃ n : ℕ, (aₙ 3)^2 = a₁ * a₉) →
  (∀ n : ℕ, aₙ n = 2 * n) ∧
  (∀ n : ℕ, T n = (∑ k in finset.range n, (1 : ℚ) / (aₙ k * aₙ (k + 1))) = n / (4 * (n + 1)))
  := by
  sorry

end arith_geo_sum_of_reciprocal_terms_l438_438576


namespace sum_of_squares_of_roots_l438_438932

/-- If r, s, and t are the roots of the cubic equation x³ - ax² + bx - c = 0, then r² + s² + t² = a² - 2b. -/
theorem sum_of_squares_of_roots (r s t a b c : ℝ) (h1 : r + s + t = a) (h2 : r * s + r * t + s * t = b) (h3 : r * s * t = c) :
    r ^ 2 + s ^ 2 + t ^ 2 = a ^ 2 - 2 * b := 
by 
  sorry

end sum_of_squares_of_roots_l438_438932


namespace monotonicity_f_prime_inequality_a_eq_1_l438_438902

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) * Real.log x

theorem monotonicity_f_prime (a : ℝ) :
  if a ≤ 0 then
    ∀ x > 0, (deriv (f a)) x < (deriv (f a)) (x * 2)
  else
    ∀ x > 0, (x < 1 / a → (deriv (f a)) x < (deriv (f a)) (x + 1)) ∧ (x > 1 / a → (deriv (f a)) x > (deriv (f a)) (x - 1)) :=
sorry

theorem inequality_a_eq_1 (x : ℝ) (hx : x > 0) :
  let f1 := f 1 in x * (Real.exp x + 1) > f1 x + 1 :=
sorry

end monotonicity_f_prime_inequality_a_eq_1_l438_438902


namespace no_possible_arrangement_l438_438003

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l438_438003


namespace expression_value_l438_438359

theorem expression_value : 2 + 3 * 5 + 2 = 19 := by
  sorry

end expression_value_l438_438359


namespace chess_team_selection_l438_438326

theorem chess_team_selection
  (players : Finset ℕ) (twin1 twin2 : ℕ)
  (H1 : players.card = 10)
  (H2 : twin1 ∈ players)
  (H3 : twin2 ∈ players) :
  ∃ n : ℕ, n = 182 ∧ 
  (∃ team : Finset ℕ, team.card = 4 ∧
    (twin1 ∉ team ∨ twin2 ∉ team)) ∧
  n = (players.card.choose 4 - 
      ((players.erase twin1).erase twin2).card.choose 2) := sorry

end chess_team_selection_l438_438326


namespace max_value_of_g_l438_438803

def g : ℕ → ℕ
| n => if n < 7 then n + 7 else g (n - 3)

theorem max_value_of_g : ∀ (n : ℕ), g n ≤ 13 ∧ (∃ n0, g n0 = 13) := by
  sorry

end max_value_of_g_l438_438803


namespace remainder_of_p_div_x_plus_2_l438_438162

def p (x : ℝ) : ℝ := x^4 - x^2 + 3 * x + 4

theorem remainder_of_p_div_x_plus_2 : p (-2) = 10 := by
  sorry

end remainder_of_p_div_x_plus_2_l438_438162


namespace remainder_123456789012_mod_210_l438_438836

theorem remainder_123456789012_mod_210 :
  let N := 123456789012 in 
  N ≡ 0 [MOD 6] ∧ 
  N ≡ 0 [MOD 7] ∧ 
  N ≡ 2 [MOD 5] → 
  N % 210 = 0 :=
by introv; sorry

end remainder_123456789012_mod_210_l438_438836


namespace ivanov_family_net_worth_l438_438718

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l438_438718


namespace no_valid_arrangement_l438_438004

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438004


namespace mary_money_left_l438_438317

def drink_price (p : ℕ) : ℕ := p
def medium_pizza_price (p : ℕ) : ℕ := 2 * p
def large_pizza_price (p : ℕ) : ℕ := 3 * p
def drinks_cost (n : ℕ) (p : ℕ) : ℕ := n * drink_price p
def medium_pizzas_cost (n : ℕ) (p : ℕ) : ℕ := n * medium_pizza_price p
def large_pizza_cost (n : ℕ) (p : ℕ) : ℕ := n * large_pizza_price p
def total_cost (p : ℕ) : ℕ := drinks_cost 5 p + medium_pizzas_cost 2 p + large_pizza_cost 1 p
def money_left (initial_money : ℕ) (p : ℕ) : ℕ := initial_money - total_cost p

theorem mary_money_left (p : ℕ) : money_left 50 p = 50 - 12 * p := sorry

end mary_money_left_l438_438317


namespace f_1789_is_8189_l438_438724

theorem f_1789_is_8189 (f : ℕ → ℕ) (h1 : ∀ n, f(f(n)) = 4 * n + 9) 
  (h2 : ∀ k : ℕ, f(2^k) = 2^(k + 1) + 3) : f(1789) = 8189 := sorry

end f_1789_is_8189_l438_438724


namespace magnitude_of_complex_number_real_part_l438_438202

theorem magnitude_of_complex_number_real_part (m : ℝ) :
  (∃ m : ℝ, ∃ (x : ℝ), (4 + (m)*complex.I) / (1 + 2*complex.I) = x) →
  complex.abs (m + 6*complex.I) = 10 :=
by
  sorry

end magnitude_of_complex_number_real_part_l438_438202


namespace fn_general_formula_l438_438121

   def f (x : ℝ) (h : 0 < x) : ℝ :=
     x / (x + 1)

   noncomputable def f_n (n : ℕ) (x : ℝ) (h : 0 < x) : ℝ :=
     if n = 1 then f x h else f (f_n (n - 1) x h) h

   theorem fn_general_formula (n : ℕ) (x : ℝ) (h : 0 < x) (hn : 2 ≤ n) :
     f_n n x h = x / (n * x + 1) :=
   sorry
   
end fn_general_formula_l438_438121


namespace set_exists_condition_iff_empty_intersection_l438_438603

variables {U : Type*} (A B : set U)

theorem set_exists_condition_iff_empty_intersection :
  (∃ C : set U, A ⊆ C ∧ B ⊆ Cᶜ) ↔ A ∩ B = ∅ :=
sorry

end set_exists_condition_iff_empty_intersection_l438_438603


namespace smallest_even_number_sum_750_l438_438249

theorem smallest_even_number_sum_750 :
  ∃ n : ℤ, (∀ k : ℤ, 0 ≤ k ∧ k < 25 → (n + 2 * k) % 2 = 0) ∧ (∑ k in Finset.range 25, (n + 2 * k) = 750) ∧ n = 6 :=
by
  sorry

end smallest_even_number_sum_750_l438_438249


namespace Leah_money_lost_l438_438594

noncomputable def milkshake_cost (earnings : ℝ) : ℝ := (1/7) * earnings
noncomputable def remaining_after_milkshake (earnings : ℝ) : ℝ := earnings - milkshake_cost earnings
noncomputable def comic_book_cost (remaining_money : ℝ) : ℝ := (1/5) * remaining_money
noncomputable def remaining_after_comic_book (remaining_money : ℝ) : ℝ := remaining_money - comic_book_cost remaining_money
noncomputable def savings_deposit (remaining_money : ℝ) : ℝ := (3/8) * remaining_money
noncomputable def remaining_after_savings (remaining_money : ℝ) : ℝ := remaining_money - savings_deposit remaining_money
noncomputable def amount_not_shredded (remaining_money : ℝ) : ℝ := 0.10 * remaining_money
noncomputable def amount_lost (initial_remaining : ℝ) (amount_not_shredded : ℝ) : ℝ := initial_remaining - amount_not_shredded

theorem Leah_money_lost : 
  let earnings := 28 in
  let step1 := milkshake_cost earnings in
  let remaining1 := remaining_after_milkshake earnings in
  let step2 := comic_book_cost remaining1 in
  let remaining2 := remaining_after_comic_book remaining1 in
  let step3 := savings_deposit remaining2 in
  let remaining3 := remaining_after_savings remaining2 in
  let step4 := amount_not_shredded remaining3 in
  amount_lost remaining3 step4 = 10.8  :=
by sorry

end Leah_money_lost_l438_438594


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l438_438820

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l438_438820


namespace max_true_statements_l438_438294

theorem max_true_statements (c d : ℝ) : 
  (∃ n, 1 ≤ n ∧ n ≤ 5 ∧ 
    (n = (if (1/c > 1/d) then 1 else 0) +
          (if (c^2 < d^2) then 1 else 0) +
          (if (c > d) then 1 else 0) +
          (if (c > 0) then 1 else 0) +
          (if (d > 0) then 1 else 0))) → 
  n ≤ 3 := 
sorry

end max_true_statements_l438_438294


namespace not_periodic_l438_438457

def f_sequence : ℕ → ℕ
| 1       := 1
| n@(k+1) := if n % 2 = 0 then f_sequence (n / 2)
             else if f_sequence (n - 1) % 2 = 1 then f_sequence (n - 1) - 1
             else f_sequence (n - 1) + 1

example : f_sequence (2^2020 - 1) = 0 :=
by sorry

theorem not_periodic (t n0 : ℕ) : ¬ ∀ n ≥ n0, f_sequence (n + t) = f_sequence n :=
by sorry

end not_periodic_l438_438457


namespace Antoinette_weight_l438_438777

variable (A R : ℝ)

theorem Antoinette_weight :
  A = 63 → 
  (A = 2 * R - 7) → 
  (A + R = 98) →
  True :=
by
  intros hA hAR hsum
  have h := hAR.symm
  rw [h] at hsum
  have h1 : 2 * R - 7 + R = 98 := hsum
  have h2 : 3 * R - 7 = 98 := h1
  have h3 : 3 * R = 105 := by linarith
  have hR : R = 35 := by linarith
  have hA' : A = 2 * 35 - 7 := by rwa [←h]
  have hA' : A = 63 := by linarith
  sorry

end Antoinette_weight_l438_438777


namespace fraction_less_than_40_percent_l438_438412

theorem fraction_less_than_40_percent (x : ℝ) (h1 : x * 180 = 48) (h2 : x < 0.4) : x = 4 / 15 :=
by
  sorry

end fraction_less_than_40_percent_l438_438412


namespace Zephyria_license_plates_l438_438089

theorem Zephyria_license_plates :
  let letters := 26
  let digits := 10
  let total_plates := letters^3 * digits^4 in
  total_plates = 175760000 :=
by
  let letters := 26
  let digits := 10
  let total_plates := letters^3 * digits^4
  have h1 : total_plates = 175760000 := sorry
  exact h1

end Zephyria_license_plates_l438_438089


namespace third_sergeant_can_avoid_punishment_initially_l438_438053

structure DutyShift :=
(sergeant : ℕ) -- Identifier for the sergeant on duty (0, 1, or 2).
(soldier_duties : list ℕ) -- List of soldiers assigned an extra duty in this shift.

structure Cycle :=
(shift1 : DutyShift)
(shift2 : DutyShift)
(shift3 : DutyShift)

def condition1 (shift : DutyShift) : Prop :=
shift.soldier_duties.length ≥ 1

def condition2 (shifts : list DutyShift) : Prop :=
∀ s, (shifts.count s ≤ 2 ∧ ∀ t ∈ shifts, t.soldier_duties.count(s) ≤ 1)

def condition3 (shifts : list DutyShift) : Prop :=
∀ (i j : ℕ), i ≠ j → i < shifts.length → j < shifts.length → shifts[i].soldier_duties ≠ shifts[j].soldier_duties

def condition4 (cycle : Cycle) : Prop :=
let shifts := [cycle.shift1, cycle.shift2, cycle.shift3] in 
condition1 cycle.shift1 ∧ condition1 cycle.shift2 ∧ condition1 cycle.shift3 ∧ 
condition2 shifts ∧ condition3 shifts

def can_avoid_punishment (c : Cycle) : Prop :=
∀ i j k, (i ≠ 2 ∧ k ≠ 2) → condition4 c

/-- 
Can the third sergeant assign extra duties initially without being punished,
given the conditions specified?
-/
theorem third_sergeant_can_avoid_punishment_initially : ∃ c : Cycle, can_avoid_punishment c :=
sorry

end third_sergeant_can_avoid_punishment_initially_l438_438053


namespace number_of_divisors_of_3003_l438_438826

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l438_438826


namespace fraction_of_25_exact_value_l438_438382

-- Define the conditions
def eighty_percent_of_sixty : ℝ := 0.80 * 60
def smaller_by_twenty_eight (x : ℝ) : Prop := x * 25 = eighty_percent_of_sixty - 28

-- The proof problem
theorem fraction_of_25_exact_value (x : ℝ) : smaller_by_twenty_eight x → x = 4 / 5 := by
  intro h
  sorry

end fraction_of_25_exact_value_l438_438382


namespace find_x4_minus_x1_l438_438295

variables {f g : ℝ → ℝ}
variables (x_1 x_2 x_3 x_4 : ℝ)
variables (a b c : ℝ) -- Coefficients of the quadratic function

-- Conditions as given in the problem
axiom f_quadratic : ∃ a b c, ∀ x, f(x) = a*x^2 + b*x + c
axiom g_def : ∀ x, g(x) = -f(120 - x)
axiom g_contains_vertex_of_f : ∃ x₀, ∀ x, g(x₀) = -f(120 - x₀)
axiom intercepts_ordered : x_1 < x_2 ∧ x_2 < x_3 ∧ x_3 < x_4
axiom intercepts_spacing : x_3 - x_2 = 160

theorem find_x4_minus_x1 :
  x_4 - x_1 = 640 + 320 * real.sqrt 3 :=
sorry

end find_x4_minus_x1_l438_438295


namespace subset_complU_N_l438_438204

variable {U : Type} {M N : Set U}

-- Given conditions
axiom non_empty_M : ∃ x, x ∈ M
axiom non_empty_N : ∃ y, y ∈ N
axiom subset_complU_M : N ⊆ Mᶜ

-- Prove the statement that M is a subset of the complement of N
theorem subset_complU_N : M ⊆ Nᶜ := by
  sorry

end subset_complU_N_l438_438204


namespace ellipse_equation_line_equation_l438_438875

def ellipse_conditions (a b c : ℝ) (A : ℝ × ℝ) (eccentricity : ℝ) : Prop :=
  eccentricity = c / a ∧
  a > b ∧ b > 0 ∧ A = (2, 0)

def line_passes_through_focus (l : ℝ → ℝ) (F : ℝ × ℝ) (M N : ℝ × ℝ) (ellipse_eq : ℝ × ℝ → Prop) : Prop :=
  F = (1, 0) ∧
  ellipse_eq F ∧
  ellipse_eq M ∧ ellipse_eq N ∧
  l F.1 = F.2 ∧
  l M.1 = M.2 ∧ l N.1 = N.2

noncomputable def area_of_triangle (A M N : ℝ × ℝ) : ℝ :=
  abs (1/2 * (A.1 * (M.2 - N.2) + M.1 * (N.2 - A.2) + N.1 * (A.2 - M.2)))

theorem ellipse_equation (a b c : ℝ) (A : ℝ × ℝ) (ecc : ℝ) :
  ellipse_conditions a b c A ecc →
  ( ∀ (x y : ℝ), (x, y) ∈ (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1) ↔ (x, y) ∈ (λ p : ℝ × ℝ, p.1^2 / 4 + p.2^2 / 3 = 1) ) :=
sorry

theorem line_equation (l : ℝ → ℝ) (A F M N : ℝ × ℝ) :
  ( ∀ (a b c : ℝ), ellipse_conditions a b c A (1/2) ) →
  line_passes_through_focus l F M N (λ p : ℝ × ℝ, p.1^2 / 4 + p.2^2 / 3 = 1) →
  area_of_triangle A M N = 6 * sqrt 2 / 7 →
  l = λ x, x - 1 ∨ l = λ x, -x + 1 :=
sorry

end ellipse_equation_line_equation_l438_438875


namespace problem_solution_l438_438390

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l438_438390


namespace complement_union_l438_438233

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l438_438233


namespace smallest_n_such_that_floor_eq_1989_l438_438838

theorem smallest_n_such_that_floor_eq_1989 :
  ∃ (n : ℕ), (∀ k, k < n -> ¬(∃ x : ℤ, ⌊(10^k : ℚ) / x⌋ = 1989)) ∧ (∃ x : ℤ, ⌊(10^n : ℚ) / x⌋ = 1989) :=
sorry

end smallest_n_such_that_floor_eq_1989_l438_438838


namespace line_l_cartesian_curve_c_general_triangle_PAB_min_perimeter_l438_438265

-- Conditions
def line_l_polar (ρ θ : ℝ) : Prop := ρ * cos (θ + π/4) = sqrt 2 / 2

def curve_c_parametric (θ : ℝ) : ℝ × ℝ := 
  (5 + cos θ, sin θ)

-- Theorem for Cartesian equation of line l
theorem line_l_cartesian (x y : ℝ) : x - y = 1 ↔ ∃ ρ θ, line_l_polar ρ θ ∧ x = ρ * cos θ ∧ y = ρ * sin θ := sorry

-- Theorem for general equation of curve C
theorem curve_c_general (x y : ℝ) : (x-5)^2 + y^2 = 1 ↔ ∃ θ, (x, y) = curve_c_parametric θ := sorry

-- Theorem for minimum perimeter of triangle PAB
theorem triangle_PAB_min_perimeter (A B P : ℝ × ℝ) :
  A = (4, 0) ∧ B = (6, 0) ∧ (∃ x y, y = x - 1 ∧ P = (x, y)) →
  ∀ P : ℝ × ℝ, ∃ A B, Π (p), (A = (4, 0) ∧ B = (6, 0)), min (dist A P + dist P B + dist A B)= 2+ sqrt34 := sorry

end line_l_cartesian_curve_c_general_triangle_PAB_min_perimeter_l438_438265


namespace rectangle_area_l438_438085

theorem rectangle_area (sqr_area : ℕ) (rect_width rect_length : ℕ) (h1 : sqr_area = 25)
    (h2 : rect_width = Int.sqrt sqr_area) (h3 : rect_length = 2 * rect_width) :
    rect_width * rect_length = 50 := by
  sorry

end rectangle_area_l438_438085


namespace fractional_exponent_equality_l438_438725

theorem fractional_exponent_equality :
  (3 / 4 : ℚ) ^ 2017 * (- ((1:ℚ) + 1 / 3)) ^ 2018 = 4 / 3 :=
by
  sorry

end fractional_exponent_equality_l438_438725


namespace solve_for_s_l438_438929

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * 3^s) 
  (h2 : 45 = m * 9^s) : 
  s = 2 :=
sorry

end solve_for_s_l438_438929


namespace FI_squared_l438_438581

-- Definitions for the given conditions
-- Note: Further geometric setup and formalization might be necessary to carry 
-- out the complete proof in Lean, but the setup will follow these basic definitions.

-- Let ABCD be a square
def ABCD_square (A B C D : ℝ × ℝ) : Prop :=
  -- conditions for ABCD being a square (to be properly defined based on coordinates and properties)
  sorry

-- Triangle AEH is an equilateral triangle with side length sqrt(3)
def equilateral_AEH (A E H : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A E = s ∧ dist E H = s ∧ dist H A = s 

-- Points E and H lie on AB and DA respectively
-- Points F and G lie on BC and CD respectively
-- Points I and J lie on EH with FI ⊥ EH and GJ ⊥ EH
-- Areas of triangles and quadrilaterals
def geometric_conditions (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  sorry

-- Final statement to prove
theorem FI_squared (A B C D E F G H I J : ℝ × ℝ) (s : ℝ) 
  (h_square: ABCD_square A B C D) 
  (h_equilateral: equilateral_AEH A E H (Real.sqrt 3))
  (h_geo: geometric_conditions A B C D E F G H I J) :
  dist F I ^ 2 = 4 / 3 :=
sorry

end FI_squared_l438_438581


namespace ratio_a_b_l438_438402

-- Defining the curves
def curve1 (x : ℝ) : ℝ := Real.cos x
def curve2 (y : ℝ) : ℝ := 100 * Real.cos (100 * y)

-- Definition of intersection points with positive coordinates
def intersection_points : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.2 = curve1 p.1 ∧ p.1 = curve2 p.2 }

-- Definitions of a and b
def a : ℝ := ∑ p in intersection_points, p.1
def b : ℝ := ∑ p in intersection_points, p.2

-- Prove that a / b = 100
theorem ratio_a_b : a / b = 100 := by
  sorry

end ratio_a_b_l438_438402


namespace phi_value_for_odd_sin_l438_438940

theorem phi_value_for_odd_sin (ϕ : ℝ) (h : ∀ x : ℝ, sin (−x + ϕ) = -sin (x + ϕ)) : ϕ = π := 
sorry

end phi_value_for_odd_sin_l438_438940


namespace mean_val_d_l438_438851

def S (m n : ℕ) : Set (List ℕ) :=
  { l | (∀ k, (1 ≤ k ∧ k ≤ m) → count l k = n) ∧ l.length = m * n ∧ ∀ x ∈ l, (1 ≤ x ∧ x ≤ m) }

def d (l : List ℕ) : ℕ :=
  l.pairwise (λ x y, abs (x - y)).sum

def mean_d (S : Set (List ℕ)) : ℕ :=
  let N := S.toFinset.toList in N.sum d / N.length

theorem mean_val_d (m n : ℕ) (h : 0 < m ∧ m < 10) :
  mean_d (S m n) = n * (m^2 - 1) / 3 :=
sorry

end mean_val_d_l438_438851


namespace problem_solution_l438_438391

def eq_A (x : ℝ) : Prop := 2 * x = 7
def eq_B (x y : ℝ) : Prop := x^2 + y = 5
def eq_C (x : ℝ) : Prop := x = 1 / x + 1
def eq_D (x : ℝ) : Prop := x^2 + x = 4

def is_quadratic (eq : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ a * x^2 + b * x + c = 0

theorem problem_solution : is_quadratic eq_D := by
  sorry

end problem_solution_l438_438391


namespace max_vehicles_div_10_l438_438622

-- Each vehicle is 5 meters long
def vehicle_length : ℕ := 5

-- The speed rule condition
def speed_rule (m : ℕ) : ℕ := 20 * m

-- Maximum number of vehicles in one hour
def max_vehicles_per_hour (m : ℕ) : ℕ := 4000 * m / (m + 1)

-- N is the maximum whole number of vehicles
def N : ℕ := 4000

-- The target statement to prove: quotient when N is divided by 10
theorem max_vehicles_div_10 : N / 10 = 400 :=
by
  -- Definitions and given conditions go here
  sorry

end max_vehicles_div_10_l438_438622


namespace length_of_platform_l438_438749

theorem length_of_platform (s : ℕ) (t : ℕ) (l_train : ℕ) (v : ℚ) 
  (hs : s = 72) (ht : t = 26) (hl : l_train = 230) (hv : v = (72 * 5 : ℚ) / 18) : 
  ∃ l_platform : ℕ, l_platform = 290 :=
by
  have D : ℚ := v * t
  have hD : D = 520 := by
    calc
      D = ((72 * 5 : ℚ) / 18) * 26 : by rw [hv, ht]
      _ = 20 * 26 : by norm_num
      _ = 520 : by norm_num
  exists 290
  have h_pt_len : 230 + 290 = 520 := by norm_num
  have : 230 + 290 = D := by rw [hD, ←h_pt_len]
  simp only [hl] at this
  exact eq_comm.1 this

end length_of_platform_l438_438749


namespace f_one_eq_zero_f_decreasing_f_min_on_2_to_9_l438_438188

variable (f : ℝ → ℝ)
variable f_prop : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
variable f_neg : ∀ x : ℝ, 1 < x → f x < 0

-- Statement 1: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
sorry

-- Statement 2: Prove that f(x) is a decreasing function on (0, ∞)
theorem f_decreasing : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 :=
sorry

-- Statement 3: Prove that the minimum value of f(x) on [2, 9] is -2, given f(3) = -1.
variable f_three : f 3 = -1
theorem f_min_on_2_to_9 : ∃ x : ℝ, x ∈ (Icc 2 9) ∧ is_min_on f (Icc 2 9) (-2) :=
sorry

end f_one_eq_zero_f_decreasing_f_min_on_2_to_9_l438_438188


namespace remainder_123456789012_mod_210_l438_438837

theorem remainder_123456789012_mod_210 :
  let N := 123456789012 in 
  N ≡ 0 [MOD 6] ∧ 
  N ≡ 0 [MOD 7] ∧ 
  N ≡ 2 [MOD 5] → 
  N % 210 = 0 :=
by introv; sorry

end remainder_123456789012_mod_210_l438_438837


namespace friends_locations_correct_l438_438338

-- Model the locations as natural numbers
inductive Location
| loc1 | loc2 | loc3 | loc4 | loc5 | loc6 | loc7

open Location

-- Define the seven friends
inductive Friend
| Ana | Bento | Celina | Diana | Elisa | Fábio | Guilherme

open Friend

-- Define the conditions from each friend
structure FriendsConditions where
  Ana_no_statement : True
  Bento_inside_single_figure : ∃ l, l ∈ [loc6, loc7]  
  Celina_inside_all_figures : loc3
  Diana_inside_triangle_not_square : loc5
  Elisa_inside_triangle_and_circle : loc4
  Fábio_not_inside_polygon : loc1
  Guilherme_inside_circle : loc2

-- Correct answers were:
structure FriendsLocations where
  Ana : Location
  Bento : Location
  Celina : Location
  Diana : Location
  Elisa : Location
  Fábio : Location
  Guilherme : Location

-- The theorem that establishes the correct assignment of friends to locations
theorem friends_locations_correct : FriendsConditions → FriendsLocations :=
by
  intro h
  exact {
    Ana := loc6
    Bento := loc7
    Celina := loc3
    Diana := loc5
    Elisa := loc4
    Fábio := loc1
    Guilherme := loc2
  }

end friends_locations_correct_l438_438338


namespace integer_solutions_count_l438_438120

theorem integer_solutions_count :
  { x : ℤ | (x^2 - 3*x + 2)^(x^2 - 2*x + 3) = 1 }.finite.to_finset.card = 2 :=
sorry

end integer_solutions_count_l438_438120


namespace maximize_profit_l438_438086

def profit_function (a b : ℕ) : ℕ := a + 2 * b = 400 ∧ 2 * a + b = 350

def max_profit (x y : ℕ) : Prop :=
  (∀ x, 33 ≤ x ∧ x ≤ 70 → y = (100 + 0 - 50) * x + 150 * (100 - x)) ∨
  y = 15000

def max_profit_with_price_adjustment (x y : ℕ) (m : ℕ) : Prop :=
  0 < m ∧ m ≤ 50 →
  ((m < 50 ∧ x = 34 ∧ y = (100 + m - 50) * x + 150 * (100 - x)) ∨
   (m = 50 ∧ 33 ≤ x ∧ x ≤ 70 ∧ y = 15000))

theorem maximize_profit (a b : ℕ) (x y : ℕ ) (m : ℕ) :
  profit_function a b →
    (∃ x, max_profit x y) ∧ 
    (∀ m, max_profit_with_price_adjustment x y m) :=
by
  intros
  sorry

end maximize_profit_l438_438086


namespace points_on_opposite_sides_of_line_l438_438218

theorem points_on_opposite_sides_of_line 
  (a : ℝ) 
  (h : (3 * -3 - 2 * -1 - a) * (3 * 4 - 2 * -6 - a) < 0) : 
  -7 < a ∧ a < 24 :=
sorry

end points_on_opposite_sides_of_line_l438_438218


namespace volume_pyramid_SABC_l438_438507

-- Data definitions based on the conditions
variable (R a b : ℝ)
variable (hR : R > 0)
variable (ha : a > 0)
variable (hb : b > 0)

-- Definition of the volume of the pyramid and the theorem to prove its value
noncomputable def volume_pyramid (R a b : ℝ) : ℝ := 
  R * a^3 * real.sqrt (4 * b^2 - a^2) / (6 * (4 * R^2 + a^2))

theorem volume_pyramid_SABC
  (S A B C : Type)
  [metric_space S] [metric_space A] 
  [metric_space B] [metric_space C]
  (h_SA_Sphere : sphere S R) 
  (h_plane_ABC_Sphere : touches_plane_on_point S R A B C)
  (h_CA_perp : A * C ⟂ sphere_center_point S R A)
  (h_BS_opposite : line_BS_intersect_sphere_opposite S R B C )
  (SABC : pyramid S A B C)
  : volume_pyramid R a b = R * a^3 * real.sqrt (4 * b^2 - a^2) / (6 * (4 * R^2 + a^2)) := 
sorry

end volume_pyramid_SABC_l438_438507


namespace bubble_pass_probability_l438_438345

theorem bubble_pass_probability :
  ∀ (n : ℕ) (r : ℕ → ℕ), 
    n = 50 ∧ 
    (∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → i ≠ j → r i ≠ r j) →
    (p + q = 273) :=
by
  intros n r H,
  have Hn : n = 50 := H.1,
  have distinct : ∀ i j, 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → i ≠ j → r i ≠ r j := H.2,
  -- Further proof steps would go here
  sorry

end bubble_pass_probability_l438_438345


namespace part1_monotonic_part2_two_zeros_part3_stationary_points_l438_438540

-- Part (1)
theorem part1_monotonic (λ : ℝ) (h : ℝ → ℝ := λ x, Real.log x + λ / x) :
  (∀ x > 0, monotone h x) ↔ λ ≤ 0 :=
sorry

-- Part (2)
theorem part2_two_zeros (λ : ℝ) (h : ℝ → ℝ := λ x, Real.log x + λ / x) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0) ↔ (0 < λ ∧ λ < 1 / Real.exp 1) :=
sorry

-- Part (3)
theorem part3_stationary_points (λ : ℝ) (p q : ℝ) (h : ℝ → ℝ := λ x, Real.log x + λ / x) 
  (g : ℝ → ℝ := λ x, h x - λ * x) :
  (p < q ∧ g (p) = g (q)) ∧ (λ ∈ [4 / 17, 2 / 5]) → 
  (∃ c : ℝ → ℝ, ∀ p q, |g (p) - g (q)| = |2 * Real.log p + (4 / (1 + p^2)) - 2|
  ∧ c p ∈ [2 * Real.log 2 - 6 / 5, 4 * Real.log 2 - 30 / 17]) :=
sorry

end part1_monotonic_part2_two_zeros_part3_stationary_points_l438_438540


namespace find_hyperbola_l438_438524

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ∀ x y : ℝ, (x^2 / (a^2)) - (y^2 / (b^2)) = 1

def circle_C : Circle := {
  center := (3, 0),
  radius := 2
}

def is_tangent (C : Circle) (line : ℝ × ℝ) : Prop :=
  let (m, c) := line
  (|C.center.1 * m + C.center.2 - c|) / (sqrt (m^2 + 1)) = C.radius

theorem find_hyperbola (H : Hyperbola) :
  H.a = sqrt 5 ∧ H.b = 2 ∧
  (is_tangent circle_C (b, a)) ∧
  sqrt (H.a^2 + H.b^2) = 3 →
  ∀ x y : ℝ, (x^2 / 5) - (y^2 / 4) = 1 :=
by
  sorry

end find_hyperbola_l438_438524


namespace card_arrangement_impossible_l438_438037

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438037


namespace part_a_1_part_a_2_part_b_l438_438852

noncomputable def f (x : ℝ) : ℝ := (Real.log x)^2 - (1 / 6)

theorem part_a_1 (x : ℝ) (hx : x ≥ 1) : f(x) = (Real.log x)^2 - (1 / 6) :=
by
  sorry

theorem part_a_2 : f(exp(1)) = 5 / 6 :=
by
  let x := exp 1
  have h1 : x = exp 1 := rfl
  have h2 : f x = (Real.log x)^2 - 1 / 6 := part_a_1 x (by linarith)
  rw [h1, Real.log_exp 1] at h2
  rw [h2]
  norm_num

noncomputable def tangent_at_e (x : ℝ) : ℝ := (2 / Real.exp(1)) * x - (7 / 6)

theorem part_b : ∫ t in 1..Real.exp(1), (f t - tangent_at_e t) = 1 - 1 / Real.exp(1) :=
by
  sorry

end part_a_1_part_a_2_part_b_l438_438852


namespace minimum_a_plus_b_l438_438244

theorem minimum_a_plus_b (a b : ℝ) (h : ∀ x : ℝ, x > 0 → log x - a * Real.exp x - b + 1 ≤ 0) : 
  0 ≤ a + b ∧ (∃ a b : ℝ, a = 1 ∧ b = -1 ∧ a + b = 0) :=
by
  sorry

end minimum_a_plus_b_l438_438244


namespace cards_not_divisible_by_7_l438_438016

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438016


namespace no_possible_arrangement_l438_438000

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l438_438000


namespace gary_current_weekly_eggs_l438_438491

noncomputable def egg_laying_rates : List ℕ := [6, 5, 7, 4]

def total_eggs_per_day (rates : List ℕ) : ℕ :=
  rates.foldl (· + ·) 0

def total_eggs_per_week (eggs_per_day : ℕ) : ℕ :=
  eggs_per_day * 7

theorem gary_current_weekly_eggs : 
  total_eggs_per_week (total_eggs_per_day egg_laying_rates) = 154 :=
by
  sorry

end gary_current_weekly_eggs_l438_438491


namespace complex_addition_l438_438340

def a : ℂ := -5 + 3 * Complex.i
def b : ℂ := 2 - 7 * Complex.i

theorem complex_addition : a + b = -3 - 4 * Complex.i := 
by 
  sorry

end complex_addition_l438_438340


namespace ivanov_family_net_worth_l438_438720

theorem ivanov_family_net_worth :
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  in net_worth = 2300000 := 
by
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  show net_worth = 2300000 from sorry

end ivanov_family_net_worth_l438_438720


namespace greatest_common_divisor_is_one_l438_438694

-- Define the expressions for a and b
def a : ℕ := 114^2 + 226^2 + 338^2
def b : ℕ := 113^2 + 225^2 + 339^2

-- Now state that the gcd of a and b is 1
theorem greatest_common_divisor_is_one : Nat.gcd a b = 1 := sorry

end greatest_common_divisor_is_one_l438_438694


namespace count_routes_from_A_to_B_l438_438111

-- Define cities as an inductive type
inductive City
| A
| B
| C
| D
| E

-- Define roads as a list of pairs of cities
def roads : List (City × City) := [
  (City.A, City.B),
  (City.A, City.D),
  (City.B, City.D),
  (City.C, City.D),
  (City.D, City.E),
  (City.B, City.E)
]

-- Define the problem statement
noncomputable def route_count : ℕ :=
  3  -- This should be proven

theorem count_routes_from_A_to_B : route_count = 3 :=
  by
    sorry  -- Proof goes here

end count_routes_from_A_to_B_l438_438111


namespace cups_needed_correct_l438_438422

-- Define the conditions
def servings : ℝ := 18.0
def cups_per_serving : ℝ := 2.0

-- Define the total cups needed calculation
def total_cups (servings : ℝ) (cups_per_serving : ℝ) : ℝ :=
  servings * cups_per_serving

-- State the proof problem
theorem cups_needed_correct :
  total_cups servings cups_per_serving = 36.0 :=
by
  sorry

end cups_needed_correct_l438_438422


namespace part_one_part_two_l438_438715

-- Define the set P and its properties
variables {P : Finset ℕ}
def isPrime (n : ℕ) : Prop := Nat.Prime n
def prime_set : Prop := ∀ n ∈ P, isPrime n

def m (P : Finset ℕ) : ℕ := 
  -- Implementation for m(P) goes here
  sorry

-- The proof problems
theorem part_one (hP : prime_set) : |P| ≤ m(P) ∧ (|P| = m(P) ↔ P.min' ≥ |P|) :=
by sorry

theorem part_two (hP : prime_set) : m(P) < (|P| + 1) * (2 ^ |P| - 1) :=
by sorry

-- Definitions
#align Lean_code_example m

end part_one_part_two_l438_438715


namespace relation_x_y_l438_438525

variable (V : Type) [AddCommGroup V] [Vector V] (OA OB OP PA : V)
variable (x y λ : ℝ)

-- assuming the conditions
axiom non_zero_OA : OA ≠ 0
axiom non_zero_OB : OB ≠ 0
axiom not_collinear : ¬Collinear OA OB
axiom OP_expression : 2 • OP = x • OA + y • OB
axiom PA_expression : PA = λ • (OB - OA)

theorem relation_x_y : x + y - 2 = 0 :=
by sorry

end relation_x_y_l438_438525


namespace f_periodic_if_is_bounded_and_satisfies_fe_l438_438290

variable {f : ℝ → ℝ}

-- Condition 1: f is a bounded real function, i.e., it is bounded above and below
def is_bounded (f : ℝ → ℝ) : Prop := ∃ M, ∀ x, |f x| ≤ M

-- Condition 2: The functional equation given for all x.
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)

-- We need to show that f is periodic with period 1.
theorem f_periodic_if_is_bounded_and_satisfies_fe (h_bounded : is_bounded f) (h_fe : functional_eq f) : 
  ∀ x, f (x + 1) = f x :=
sorry

end f_periodic_if_is_bounded_and_satisfies_fe_l438_438290


namespace vector_combination_correct_l438_438921

-- Define the vectors a, b, and c
def a : ℝ × ℝ × ℝ := (3, 5, -1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)
def c : ℝ × ℝ × ℝ := (4, -1, -3)

-- Define the linear combination of vectors
def vec_combination := (2 * a.1 - 3 * b.1 + 4 * c.1, 2 * a.2 - 3 * b.2 + 4 * c.2, 2 * a.3 - 3 * b.3 + 4 * c.3)

-- Statement to assert the result of the linear combination
theorem vector_combination_correct : vec_combination = (16, 0, -23) := 
by {
    sorry
}

end vector_combination_correct_l438_438921


namespace inequality_div_c_squared_l438_438930

theorem inequality_div_c_squared (a b c : ℝ) (h : a > b) : (a / (c^2 + 1) > b / (c^2 + 1)) :=
by
  sorry

end inequality_div_c_squared_l438_438930


namespace length_of_bridge_is_l438_438764

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 21.998240140788738
noncomputable def speed_kmph : ℝ := 36
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is : bridge_length = 119.98240140788738 :=
by
  have speed_mps_val : speed_mps = 10 := by
    norm_num [speed_kmph, speed_mps]
  have total_distance_val : total_distance = 219.98240140788738 := by
    norm_num [total_distance, speed_mps_val, time_to_cross_bridge]
  have bridge_length_val : bridge_length = 119.98240140788738 := by
    norm_num [bridge_length, total_distance_val, train_length]
  exact bridge_length_val

end length_of_bridge_is_l438_438764


namespace hyperbola_foci_coordinates_l438_438354

theorem hyperbola_foci_coordinates (k : ℝ) (h : 1 + k^2 > 0) :
    let a := real.sqrt (1 + k^2),
        b := real.sqrt (8 - k^2),
        c := real.sqrt (a^2 + b^2)
    in c = 3 ∧ (a^2 = 1 + k^2) ∧ (b^2 = 8 - k^2) →
    ∃ x y : ℝ, (x = 3 ∨ x = -3) ∧ y = 0 :=
by
    sorry

end hyperbola_foci_coordinates_l438_438354


namespace parallel_lines_l438_438943

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, (ax + 2 * y + a = 0 ∧ 3 * a * x + (a - 1) * y + 7 = 0) →
    - (a / 2) = - (3 * a / (a - 1))) ↔ (a = 0 ∨ a = 7) :=
by
  sorry

end parallel_lines_l438_438943


namespace cos_thirteen_pi_over_four_l438_438819

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l438_438819


namespace not_possible_arrange_cards_l438_438021

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438021


namespace current_at_time_l438_438670

noncomputable def I (t : ℝ) : ℝ := 5 * (Real.sin (100 * Real.pi * t + Real.pi / 3))

theorem current_at_time (t : ℝ) (h : t = 1 / 200) : I t = 5 / 2 := by
  sorry

end current_at_time_l438_438670


namespace tradesman_gain_l438_438088

-- Let's define a structure representing the tradesman's buying and selling operation.
structure Trade where
  true_value : ℝ
  defraud_rate : ℝ
  buy_price : ℕ
  sell_price : ℕ

theorem tradesman_gain (T : Trade) (H1 : T.defraud_rate = 0.2) (H2 : T.true_value = 100)
  (H3 : T.buy_price = T.true_value * (1 - T.defraud_rate))
  (H4 : T.sell_price = T.true_value * (1 + T.defraud_rate)) :
  ((T.sell_price - T.buy_price) / T.buy_price) * 100 = 50 := 
by
  sorry

end tradesman_gain_l438_438088


namespace min_digits_to_remove_for_divisibility_by_2016_l438_438695

def digit_sum (n : ℕ) : ℕ := n.digits.sum 

theorem min_digits_to_remove_for_divisibility_by_2016 (n : ℕ) (h : n = 20162016) : 
  ∃ k, k = 3 ∧ ∃ m, m < n ∧ (n.digits.length - m.digits.length = k) ∧ (m % 2016 = 0) :=
begin
  sorry
end

end min_digits_to_remove_for_divisibility_by_2016_l438_438695


namespace range_of_f_l438_438807

noncomputable def f (x : ℝ) : ℝ := log (3^x + 1) / log 2

theorem range_of_f : set.Ioi 0 = set.range f :=
sorry

end range_of_f_l438_438807


namespace new_trailer_homes_added_l438_438380

theorem new_trailer_homes_added :
  ∀ (n : ℕ),
    (∀ (t_old_avg_age : ℕ), t_old_avg_age = 12) →
    (∀ (t_old_count t_new_count total_count : ℕ), 
      t_old_count = 15 → total_count = t_old_count + t_new_count →
      (∀ (current_avg_age : ℕ), current_avg_age = 10 → 
      (∀ (current_total_age : ℕ), current_total_age = 225 + 3 * t_new_count →
      ((current_avg_age = current_total_age / total_count) → t_new_count = 11)))) :=
by
  intros n t_old_avg_age h1 t_old_count t_new_count total_count h2 h3 current_avg_age h4 current_total_age h5 h6
  sorry

end new_trailer_homes_added_l438_438380


namespace find_x_approx_l438_438059

theorem find_x_approx :
  ∀ (x : ℝ), 3639 + 11.95 - x^2 = 3054 → abs (x - 24.43) < 0.01 :=
by
  intro x
  sorry

end find_x_approx_l438_438059


namespace smallest_M_satisfies_modulo_l438_438164

theorem smallest_M_satisfies_modulo :
  ∃ (M : ℕ), M > 0 ∧ (M^2 + 1) % 10000 = M % 10000 ∧ (∀ N : ℕ, N > 0 ∧ (N^2 + 1) % 10000 = N % 10000 → M ≤ N) :=
begin
  use 3125,
  split,
  { exact nat.zero_lt_succ 3124, },
  split,
  { norm_num, },
  { intros N hN_positive hN_mod,
    have hM_mod : (3125^2 + 1) % 10000 = 3125 % 10000 := by norm_num,
    rw ← hM_mod at hN_mod,
    sorry, -- Proof of minimality to be added
  }
end

end smallest_M_satisfies_modulo_l438_438164


namespace optimal_purchase_interval_discount_advantage_l438_438061

/- The functions and assumptions used here. -/
def purchase_feed_days (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) : ℕ :=
-- Implementation omitted
sorry

def should_use_discount (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) : Prop :=
-- Implementation omitted
sorry

/- Conditions -/
def conditions : Prop :=
  let feed_per_day := 200
  let price_per_kg := 1.8
  let storage_cost_per_kg_per_day := 0.03
  let transportation_fee := 300
  let discount_threshold := 5000 -- in kg, since 5 tons = 5000 kg
  let discount_rate := 0.85
  True -- We apply these values in the proofs below.

/- Main statements -/
theorem optimal_purchase_interval : conditions → 
  purchase_feed_days 200 1.8 0.03 300 = 10 :=
by
  intros
  -- Proof is omitted.
  sorry

theorem discount_advantage : conditions →
  should_use_discount 200 1.8 0.03 300 5000 0.85 :=
by
  intros
  -- Proof is omitted.
  sorry

end optimal_purchase_interval_discount_advantage_l438_438061


namespace a_2016_value_l438_438118

noncomputable def seq : ℕ → ℚ
variables (H1 : seq 4 = 1/8)
variables (H2 : ∀ n : ℕ, 0 < n → seq (n+2) - seq n ≤ 3^n)
variables (H3 : ∀ n : ℕ, 0 < n → seq (n+4) - seq n ≥ 10 * 3^n)

theorem a_2016_value : seq 2016 = (81^504 - 80) / 8 :=
by
  sorry

end a_2016_value_l438_438118


namespace closest_integer_to_sum_l438_438157

theorem closest_integer_to_sum : 
  let S := 500 * (∑ n in Finset.range 4999, 1 / (n + 2)^2 - 1);
  abs (S - 375) < 0.1 := 
by 
  sorry

end closest_integer_to_sum_l438_438157


namespace sqrt_diff_approx_l438_438107

noncomputable def x : ℝ := Real.sqrt 50 - Real.sqrt 48

theorem sqrt_diff_approx : abs (x - 0.14) < 0.01 :=
by
  sorry

end sqrt_diff_approx_l438_438107


namespace expression_approx_48_l438_438790

def expression : ℚ :=
  ( ( 5 / 2 : ℚ ) ^ 2 / ( 1 / 2 : ℚ ) ^ 3 * ( 5 / 2 ) ^ 2 ) / ( ( 5 / 3 : ℚ ) ^ 4 * ( 1 / 2 : ℚ ) ^ 2 / ( 2 / 3 ) ^ 3 )

theorem expression_approx_48 :
  abs (expression.toReal - 48) < 0.01 :=
by
  -- Proof will be provided here
  sorry

end expression_approx_48_l438_438790


namespace part1_part2_l438_438227

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (1 - a) * x + (1 - a)

theorem part1 (x : ℝ) : f x 4 ≥ 7 ↔ x ≥ 5 ∨ x ≤ -2 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, -1 < x → f x a ≥ 0) ↔ a ≤ 1 :=
sorry

end part1_part2_l438_438227


namespace circle_through_M_A_B_l438_438370

theorem circle_through_M_A_B :
  let M : Point := (-1, 0)
  let A : Point := (1, 2)
  let B : Point := (1, -2)
  ∃ D E F, x^2 + y^2 + D * x + E * y + F = 0 ∧ (x,y) in {A, B, M} →
  (x - 1)^2 + y^2 = 4 := by
sorry

end circle_through_M_A_B_l438_438370


namespace seq_periodic_4_l438_438272

def seq (a_n : ℕ → ℚ) (n : ℕ) : ℚ :=
  if h : a_n n < 1 / 2 then
    2 * a_n n
  else
    2 * a_n n - 1

theorem seq_periodic_4 (a_1 : ℚ) (h : a_1 = 4 / 5) : 
  let a : ℕ → ℚ := λ n, (nat.rec_on n a_1 (λ n a_n, seq a_n n))
  a 2023 = 1 / 5 :=
sorry

end seq_periodic_4_l438_438272


namespace simplify_expression_l438_438119

theorem simplify_expression (b : ℝ) (h : b ≠ -1) : 
  1 - (1 / (1 - (b / (1 + b)))) = -b :=
by {
  sorry
}

end simplify_expression_l438_438119


namespace sequence_periodicity_l438_438505

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 0 ∧ ∀ n ≥ 1, a (n + 2) = a (n + 1) - a n

theorem sequence_periodicity (a : ℕ → ℤ) (h : sequence a) : a 2017 = 1 :=
by
  sorry

end sequence_periodicity_l438_438505


namespace no_valid_theta_k_l438_438654

noncomputable def P_coordinates (theta k : ℝ) : ℝ × ℝ :=
  (3 * Real.cos theta - k, -5 * Real.sin theta)

theorem no_valid_theta_k (theta k : ℝ) (h1 : k > 0) (h2 : Real.pi / 2 < theta ∧ theta < Real.pi)
  (h3 : |(P_coordinates theta k).2| = 0.5 * |(P_coordinates theta k).1|)
  (h4 : |(P_coordinates theta k).2| + |(P_coordinates theta k).1| = 30) :
  false :=
begin
  -- Add necessary variable definitions
  let x := (P_coordinates theta k).1,
  let y := (P_coordinates theta k).2,
  
  -- Variables based on conditions
  have h_x : x = 3 * Real.cos theta - k := rfl,
  have h_y : y = -5 * Real.sin theta := rfl,
  
  -- Use given conditions to derive contradiction
  have h_dist_y : |y| = |(-5) * Real.sin theta| := by rw h_y,
  have h_dist_x : |x| = |3 * Real.cos theta - k| := by rw h_x,

  have h_eq1 : |(-5) * Real.sin theta| = 0.5 * |3 * Real.cos theta - k|,
  by rw [←h_dist_y, ←h_dist_x] at h3; exact h3,
  
  have h_eq2 : |(-5) * Real.sin theta| + |3 * Real.cos theta - k| = 30,
  by rw [←h_dist_y, ←h_dist_x] at h4; exact h4,
  
  -- Simplify condition 1 using properties of absolute values 
  have h_simplified1 : 5 * Real.sin theta = 0.5 * (3 * Real.cos theta - k),
  by {rw abs_of_pos at h_eq1; rw abs_of_pos; exact h_eq1},

  -- Solve for k and substitute into second condition to show inconsistency
  have h_k : k = 3 * Real.cos theta - 10 * Real.sin theta,
  from
    calc
      k = 3 * Real.cos theta - 10 * Real.sin theta : 
             by {solve_using h_simplified1 ⟨by skip⟩},

  -- Using k in the second condition to show contradiction
  have h_contradiction : 5 * Real.sin theta + |10 * Real.sin theta| = 30,
  rw [h_y, h_k] at h_eq2,
  sorry
end

end no_valid_theta_k_l438_438654


namespace constant_term_x_add_inv_x_pow_six_l438_438154

theorem constant_term_x_add_inv_x_pow_six : 
  let f := λ (r : ℕ), Nat.choose 6 r * x^(6 - 2 * r) in
  (∀ r : ℕ, f r = 1 ↔ r = 3) → 
  Nat.choose 6 3 = 20 :=
by
  intros f h
  sorry

end constant_term_x_add_inv_x_pow_six_l438_438154


namespace kite_problem_proof_l438_438263

variables (A B C D E F G H I J O : Type)
variables [add_comm_group A] [module ℝ A]
variables [add_comm_group B] [add_comm_group C] [add_comm_group D] [add_comm_group E]
variables [add_comm_group F] [add_comm_group G] [add_comm_group H] [add_comm_group I] [add_comm_group J] [add_comm_group O]

noncomputable def kite_properties (AB AD BC CD AC BD : Type) [module ℝ AB] [module ℝ AD] [module ℝ BC] [module ℝ CD] :
  (AB = AD) ∧ (BC = CD) → Type := 
λ h, by sorry

noncomputable def intersect_at_point (AC BD O : Type) [module ℝ AC] [module ℝ BD] [module ℝ O] :
  AC ⊓ BD = O → Type := 
λ h, by sorry

noncomputable def lines_passing_through_O (lines : Π(O : Type) [module ℝ O], list (submodule ℝ O))
  [∀ O, add_comm_group (lines O)] : Type := 
λ h, by sorry

noncomputable def intersections (AD BC AB CD lines F G E H GF EH I J O : Type) [module ℝ AD] [module ℝ BC] [module ℝ AB] [module ℝ CD] [∀ L, module ℝ (lines L)] [module ℝ F] [module ℝ G] [module ℝ E] [module ℝ H] :
  lines (O) ~[AD, BC, AB, CD] F G E H GF EH → intersect F G H E → GF ⊓ AD = I → EH ⊓ BC = J → Type := 
λ h, by sorry

noncomputable def prove_IO_equals_OJ (I J O : Type) [module ℝ O] :
  V = E → Type := 
λ h, by sorry

theorem kite_problem_proof
  (AB AD BC CD AC BD : Type)
  (O F G E H GF EH I J : Type)
  (kite_cond : (AB = AD) ∧ (BC = CD))
  (intersection_cond : AC ⊓ BD = O)
  (lines_cond : Π (O : Type), list (submodule ℝ O))
  (intersect_lines : O ~[AB, AD, BC, CD] F E G H GF EH)
  (intersections_cond : ℝ (F G E H I J)) :
  IO = OJ :=
by sorry

end kite_problem_proof_l438_438263


namespace brendan_cuts_yards_l438_438785

theorem brendan_cuts_yards (x : ℝ) (h : 7 * 1.5 * x = 84) : x = 8 :=
sorry

end brendan_cuts_yards_l438_438785


namespace no_valid_arrangement_l438_438007

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438007


namespace unique_solution_for_system_l438_438519

noncomputable theory

def points_on_line (k : ℝ) (a b1 b2 : ℝ) (P1 P2 : ℝ × ℝ) : Prop :=
  P1 = (a, b1) ∧ P2 = (2, b2) ∧
  b1 = k * a + 1 ∧ b2 = k * 2 + 1

def system_of_equations_solution (a1 a2 b1 b2 : ℝ) : Prop :=
  ∀ x y : ℝ, (a1 * x + b1 * y = 1 ∧ a2 * x + b2 * y = 1) →
  ∃! (x y : ℝ), a1 * x + b1 * y = 1 ∧ a2 * x + b2 * y = 1

theorem unique_solution_for_system (a a1 a2 b1 b2 : ℝ) (k : ℝ) (P1 P2 : ℝ × ℝ)
  (h1 : points_on_line k a b1 b2 P1 P2) (h2 : a ≠ 2) :
  system_of_equations_solution a1 a2 b1 b2 :=
by
  sorry

end unique_solution_for_system_l438_438519


namespace no_possible_arrangement_l438_438002

-- Define the set of cards
def cards := {1, 2, 3, 4, 5, 6, 8, 9}

-- Function to check adjacency criterion
def adjacent_div7 (a b : Nat) : Prop := (a * 10 + b) % 7 = 0

-- Main theorem to state the problem
theorem no_possible_arrangement (s : List Nat) (h : ∀ a ∈ s, a ∈ cards) :
  ¬ (∀ (i j : Nat), i < j ∧ j < s.length → adjacent_div7 (s.nthLe i sorry) (s.nthLe j sorry)) :=
sorry

end no_possible_arrangement_l438_438002


namespace problem_statement_l438_438863

def imaginary_unit (i : ℂ) : Prop :=
  i * i = -1

def complex_sum (n : ℕ) (i : ℂ) : ℂ :=
  (Finset.range n).sum (λ k, i ^ k)

def complex_value (i : ℂ) : ℂ :=
  complex_sum 2023 i / (1 - i)

def conjugate_value (z : ℂ) : ℂ :=
  conj z

theorem problem_statement (i : ℂ) (z : ℂ) (hz : imaginary_unit i) :
  z = complex_value i → 
  let z_conj := conjugate_value z in
  z_conj.re < 0 ∧ z_conj.im > 0 :=
by
  sorry

end problem_statement_l438_438863


namespace power_function_value_l438_438528

-- Given condition: there exists an exponent alpha such that the function f(x) = x^alpha passes through the point (1/2, sqrt(2)/2)
theorem power_function_value :
  ∃ α : ℝ, (1/2)^α = sqrt(2) / 2 ∧ (2:ℝ)^α = sqrt(2) :=
  by
    sorry

end power_function_value_l438_438528


namespace problem_statement_l438_438536

open_locale big_operators

variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b c : α) (θ : ℝ)
variables {AB BC AC : α} 

-- Conditions
def condition_1 : Prop :=
  a ⬝ b ∈ ℝ ∧ (a ⬝ b) • c ∉ ℝ

def condition_2 : Prop :=
  AB + BC - AC = 0

def condition_3 : Prop :=
  let proj_a_on_b := ∥ b ∥ * real.cos θ in true

def condition_4 (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hnc : ¬ collinear a b ∧ ¬ collinear b c ∧ ¬ collinear a c) : Prop :=
  (a ⬝ b) • c = (b ⬝ c) • a → parallel a c

-- Theorem
theorem problem_statement (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hnc : ¬ collinear a b ∧ ¬ collinear b c ∧ ¬ collinear a c) :
  condition_1 a b c ∧ ¬ condition_2 AB BC AC ∧ ¬ condition_3 a b θ ∧ condition_4 a b c ha hb hc hnc :=
sorry

end problem_statement_l438_438536


namespace initial_ratio_milk_water_l438_438956

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 60) 
  (h2 : ∀ k, k = M → M * 2 = W + 60) : (M:ℚ) / (W:ℚ) = 4 / 1 :=
by
  sorry

end initial_ratio_milk_water_l438_438956


namespace validate_triangle_properties_l438_438662

structure Triangle (A B C : Type) :=
(medians : ∀ (E F D : Type), median AE and median BF and median CD)
(segment_eq_parallel : ∀ (FH AE : Type), FH = AE and FH ∥ AE)
(midpoint : ∀ (G BH : Type), midpoint G BH)
(collinear : ∀ (E F G : Type), collinear E F G)

theorem validate_triangle_properties (A B C E F H G D : Type) 
  (t : Triangle A B C) :
  ¬(HE = HG) ∧ ¬(BH = DC) ∧ (FG = 3/4 * AB) :=
sorry

end validate_triangle_properties_l438_438662


namespace sum_13_terms_l438_438582

variables {a : ℕ → ℝ}        -- The arithmetic sequence indexed by natural numbers
variables {d : ℝ}            -- The common difference of the sequence
variables {a1 : ℝ}           -- The first term of the sequence

-- Given an arithmetic sequence where a₆ + a₈ = 8
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

def given_condition (a : ℕ → ℝ) : Prop :=
  a 6 + a 8 = 8

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

-- Sum of the first 13 terms, S₁₃, should be 52
theorem sum_13_terms (a1 d : ℝ) (a : ℕ → ℝ) :
  arithmetic_sequence a a1 d → given_condition a → sum_first_n_terms a 13 = 52 :=
by
  sorry  -- Proof is omitted

end sum_13_terms_l438_438582


namespace quarters_around_nickel_l438_438074

/-- A math problem to prove the number of quarters that can be placed around a nickel,
 each tangent to the central nickel and to two others. -/
theorem quarters_around_nickel : 
  let radius_nickel := 1
  let radius_quarter := 1.2
  let distance_center := radius_nickel + radius_quarter
  (distance_center = 2.2) →
  -- Additional necessary geometric calculations are done here
  -- Implicitly handled by assuming correct geometric placement
  true := -- We know the correct answer should lead us here
  6 := sorry

end quarters_around_nickel_l438_438074


namespace red_blue_beads_ratio_l438_438103

-- Definitions based on the conditions
def has_red_beads (betty : Type) := betty → ℕ
def has_blue_beads (betty : Type) := betty → ℕ

def betty : Type := Unit

-- Given conditions
def num_red_beads : has_red_beads betty := λ _ => 30
def num_blue_beads : has_blue_beads betty := λ _ => 20
def red_to_blue_ratio := 3 / 2

-- Theorem to prove the ratio
theorem red_blue_beads_ratio (R B: ℕ) (h_red : R = 30) (h_blue : B = 20) :
  (R / gcd R B) / (B / gcd R B ) = red_to_blue_ratio :=
by sorry

end red_blue_beads_ratio_l438_438103


namespace sum_first_four_terms_of_arithmetic_sequence_l438_438656

theorem sum_first_four_terms_of_arithmetic_sequence (a₈ a₉ a₁₀ : ℤ) (d : ℤ) (a₁ a₂ a₃ a₄ : ℤ) : 
  (a₈ = 21) →
  (a₉ = 17) →
  (a₁₀ = 13) →
  (d = a₉ - a₈) →
  (a₁ = a₈ - 7 * d) →
  (a₂ = a₁ + d) →
  (a₃ = a₂ + d) →
  (a₄ = a₃ + d) →
  a₁ + a₂ + a₃ + a₄ = 172 :=
by 
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈
  sorry

end sum_first_four_terms_of_arithmetic_sequence_l438_438656


namespace find_m_l438_438948

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → x < 2 → - (1/2)*x^2 + 2*x > -m*x) ↔ m = -1 := 
sorry

end find_m_l438_438948


namespace original_proposition_contrapositive_converse_inverse_negation_false_l438_438346

variable {a b c : ℝ}

-- Original Proposition
theorem original_proposition (h : a < b) : a + c < b + c :=
sorry

-- Contrapositive
theorem contrapositive (h : a + c >= b + c) : a >= b :=
sorry

-- Converse
theorem converse (h : a + c < b + c) : a < b :=
sorry

-- Inverse
theorem inverse (h : a >= b) : a + c >= b + c :=
sorry

-- Negation is false
theorem negation_false (h : a < b) : ¬ (a + c >= b + c) :=
sorry

end original_proposition_contrapositive_converse_inverse_negation_false_l438_438346


namespace complex_modulus_l438_438814

noncomputable theory

open Complex

-- Define the complex number
def z : ℂ := -7 + (11 / 3) * Complex.i + 2

-- Define the desired value
def desired_value : ℝ := Real.sqrt(346) / 3

-- The statement that we want to prove
theorem complex_modulus : abs z = desired_value := sorry

end complex_modulus_l438_438814


namespace measure_of_angle_A_decreasing_intervals_of_f_l438_438869

-- Problem 1: Measure of angle A
theorem measure_of_angle_A (A B C : ℝ) (a b c : ℝ) (h : sin B ^ 2 + sin C ^ 2 - sin A ^ 2 = sin B * sin C) : A = π / 3 :=
  sorry

-- Problem 2: Decreasing intervals of function f(x)
theorem decreasing_intervals_of_f (A : ℝ) (ω : ℝ) (hω : ω > 0) (h_period : ∀ x, f x = sin (ω * x + A)) :
  (∀ k : ℤ, [k * π + π / 12, k * π + 7 * π / 12]) :=
  sorry

-- Definition of f(x) based on given ω and A
def f (x : ℝ) (ω : ℝ) (A : ℝ) : ℝ := sin (ω * x + A)

end measure_of_angle_A_decreasing_intervals_of_f_l438_438869


namespace tan_alpha_sub_pi_over_4_l438_438882

theorem tan_alpha_sub_pi_over_4 :
  ∀ (α : ℝ), (sin α + cos α = √2 / 3) ∧ (0 < α ∧ α < π) → tan (α - π / 4) = 2 * √2 := 
by
  intro α h
  cases h with h1 h2
  sorry

end tan_alpha_sub_pi_over_4_l438_438882


namespace disney_ticket_sales_l438_438409

theorem disney_ticket_sales (total_people residents : ℕ) (price_resident price_non_resident : ℝ)
    (h1 : total_people = 586)
    (h2 : residents = 219) 
    (h3 : price_resident = 12.95)
    (h4 : price_non_resident = 17.95) :
    let non_residents := total_people - residents in
    let total_from_residents := residents * price_resident in
    let total_from_non_residents := non_residents * price_non_resident in
    let total_made := total_from_residents + total_from_non_residents in
    total_made = 9423.70 :=
by
    sorry

end disney_ticket_sales_l438_438409


namespace find_m_l438_438893

theorem find_m (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) 
  (hS : ∀ n, S n = n^2 - 6 * n) :
  (forall m, (5 < a m ∧ a m < 8) → m = 7)
:= 
by
  sorry

end find_m_l438_438893


namespace proof_problem_l438_438213

noncomputable def problem_conditions : Type :=
  Σ α : ℝ, (π / 2 < α) ∧ (α < 3 * π / 2) ∧ (∀ (A B : ℝ × ℝ) (hA : A = (3, 0)) (hB : B = (0, 3)),
    let C := (Real.cos α, Real.sin α),
    let AC := ((Real.cos α) - 3, (Real.sin α)),
    let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
    ‖AC‖ = ‖BC‖ → α = 5 * π / 4)

noncomputable def second_problem_conditions : Type :=
  Σ α : ℝ, (α = 5 * π / 4) ∧ (let C := (Real.cos α, Real.sin α),
    let AC := ((Real.cos α) - 3, (Real.sin α)),
    let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
    ACdotBC : ℝ := (AC.1 * BC.1 + AC.2 * BC.2),
    ACdotBC = -1 →
    (2 * (Real.sin α)^2 + 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.tan α)) = -(5 / 9))

-- The final theorem
theorem proof_problem :
  ∃ α : ℝ, (π / 2 < α) ∧ (α < 3 * π / 2) ∧
  (let C := (Real.cos α, Real.sin α),
   let AC := ((Real.cos α) - 3, (Real.sin α)),
   let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
   ‖AC‖ = ‖BC‖) → (α = 5 * π / 4) ∧
  (let AC := ((Real.cos α) - 3, (Real.sin α)),
   let BC := ((Real.cos α) - 0, (Real.sin α) - 3),
   (AC.1 * BC.1 + AC.2 * BC.2 = -1) →
   ((2 * (Real.sin α)^2 + 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.tan α)) = -(5 / 9))) :=
begin
  sorry
end

end proof_problem_l438_438213


namespace equal_radii_of_S1_S2_l438_438758

variables {A B C D K L M N : Point}
variable {S : Circle}
variables {S1 S2 : Circle}
variable [inscribed : Inscribed (A B C D) S]
variable [touches_S1 : Touches (S1) (S) A]
variable [touches_S2 : Touches (S2) (S) C]
variable [intersects_S1_AB : Intersects (S1) (Line AB) K]
variable [intersects_S1_AD : Intersects (S1) (Line AD) N]
variable [intersects_S2_BC : Intersects (S2) (Line BC) L]
variable [intersects_S2_CD : Intersects (S2) (Line CD) M]
variable [parallel_KL_MN : Parallel (Line KL) (Line MN)]

theorem equal_radii_of_S1_S2 : radius S1 = radius S2 :=
  sorry

end equal_radii_of_S1_S2_l438_438758


namespace total_oranges_for_philip_l438_438489

-- Define the initial conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def combined_oranges : ℕ := betty_oranges + bill_oranges
def frank_oranges : ℕ := 3 * combined_oranges
def seeds_planted : ℕ := 4 * frank_oranges
def successful_trees : ℕ := (3 / 4) * seeds_planted

-- The ratio of trees with different quantities of oranges
def ratio_parts : ℕ := 2 + 3 + 5
def trees_with_8_oranges : ℕ := (2 * successful_trees) / ratio_parts
def trees_with_10_oranges : ℕ := (3 * successful_trees) / ratio_parts
def trees_with_14_oranges : ℕ := (5 * successful_trees) / ratio_parts

-- Calculate the total number of oranges
def total_oranges : ℕ :=
  (trees_with_8_oranges * 8) +
  (trees_with_10_oranges * 10) +
  (trees_with_14_oranges * 14)

-- Statement to prove
theorem total_oranges_for_philip : total_oranges = 2798 :=
by
  sorry

end total_oranges_for_philip_l438_438489


namespace card_arrangement_impossible_l438_438036

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438036


namespace Ashutosh_time_to_complete_job_l438_438347

noncomputable def SureshWorkRate : ℝ := 1 / 15
noncomputable def AshutoshWorkRate (A : ℝ) : ℝ := 1 / A
noncomputable def SureshWorkIn9Hours : ℝ := 9 * SureshWorkRate

theorem Ashutosh_time_to_complete_job (A : ℝ) :
  (1 - SureshWorkIn9Hours) * AshutoshWorkRate A = 14 / 35 →
  A = 35 :=
by
  sorry

end Ashutosh_time_to_complete_job_l438_438347


namespace find_a_l438_438296

open Real

def f (x : ℝ) : ℝ := sorry  -- Placeholder for the unknown odd periodic function
def a : ℝ := sorry -- Placeholder for the value of a

-- Definitions of conditions in the problem
axiom periodicity : ∀ x, f(x + 3) = f(x)
axiom odd_function : ∀ x, f(-x) = -f(x)
axiom f1_gt_one : f(1) > 1
axiom f2015_expr : f(2015) = (2 * a - 3) / (a + 1)

-- The statement we want to prove
theorem find_a : -1 < a ∧ a < (2 / 3) :=
by
  sorry

end find_a_l438_438296


namespace intersection_of_A_and_B_l438_438878

def A : Set ℤ := {1, 2, -3}
def B : Set ℤ := {1, -4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} :=
by sorry

end intersection_of_A_and_B_l438_438878


namespace carmen_total_sales_l438_438791

-- Define the conditions as constants
def green_house_sales := 3 * 4            -- 3 boxes of samoas at $4 each
def yellow_house_sales := 2 * 3.5 + 5     -- 2 boxes of thin mints at $3.50 each and 1 box of fudge delights for $5
def brown_house_sales := 9 * 2            -- 9 boxes of sugar cookies at $2 each

-- The statement that needs to be proved
theorem carmen_total_sales : (green_house_sales + yellow_house_sales + brown_house_sales) = 42 := by
  sorry

end carmen_total_sales_l438_438791


namespace repeated_digit_percentage_l438_438240

theorem repeated_digit_percentage : 
  let total := 90000 in 
  let non_repeated := 9 * 9 * 8 * 7 * 6 in 
  let repeated := total - non_repeated in 
  let x := (repeated.toFloat / total.toFloat) * 100 in 
  x = 69.8 :=
by
  let total := 90000
  let non_repeated := 9 * 9 * 8 * 7 * 6
  let repeated := total - non_repeated
  let x := (repeated.toFloat / total.toFloat) * 100
  have : x = 69.8
  sorry

end repeated_digit_percentage_l438_438240


namespace composite_expr_l438_438174

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem composite_expr (n : ℕ) : n ≥ 2 ↔ is_composite (3^(2*n + 1) - 2^(2*n + 1) - 6^n) :=
sorry

end composite_expr_l438_438174


namespace correct_answer_statement_l438_438704

theorem correct_answer_statement
  (A := "In order to understand the situation of extracurricular reading among middle school students in China, a comprehensive survey should be conducted.")
  (B := "The median and mode of a set of data 1, 2, 5, 5, 5, 3, 3 are both 5.")
  (C := "When flipping a coin 200 times, there will definitely be 100 times when it lands 'heads up.'")
  (D := "If the variance of data set A is 0.03 and the variance of data set B is 0.1, then data set A is more stable than data set B.")
  (correct_answer := "D") : 
  correct_answer = "D" :=
  by sorry

end correct_answer_statement_l438_438704


namespace profitWednesday_l438_438735

-- Define the total profit
def totalProfit : ℝ := 1200

-- Define the profit made on Monday
def profitMonday : ℝ := totalProfit / 3

-- Define the profit made on Tuesday
def profitTuesday : ℝ := totalProfit / 4

-- Theorem to prove the profit made on Wednesday
theorem profitWednesday : 
  let profitWednesday := totalProfit - (profitMonday + profitTuesday)
  profitWednesday = 500 :=
by
  -- proof goes here
  sorry

end profitWednesday_l438_438735


namespace infinite_n_divisible_by_2018_l438_438855

def a_n (n : ℕ) : ℕ := 2 * 10^(n + 2) + 18

theorem infinite_n_divisible_by_2018 : ∃ᶠ n in (Filter.at_top : Filter ℕ), 2018 ∣ a_n n :=
sorry

end infinite_n_divisible_by_2018_l438_438855


namespace fraction_of_milk_in_cup1_l438_438987

def initial_tea_cup1 : ℚ := 6
def initial_milk_cup2 : ℚ := 6

def tea_transferred_step2 : ℚ := initial_tea_cup1 / 3
def tea_cup1_after_step2 : ℚ := initial_tea_cup1 - tea_transferred_step2
def total_cup2_after_step2 : ℚ := initial_milk_cup2 + tea_transferred_step2

def mixture_transfer_step3 : ℚ := total_cup2_after_step2 / 2
def tea_ratio_cup2 : ℚ := tea_transferred_step2 / total_cup2_after_step2
def milk_ratio_cup2 : ℚ := initial_milk_cup2 / total_cup2_after_step2
def tea_transferred_step3 : ℚ := mixture_transfer_step3 * tea_ratio_cup2
def milk_transferred_step3 : ℚ := mixture_transfer_step3 * milk_ratio_cup2

def tea_cup1_after_step3 : ℚ := tea_cup1_after_step2 + tea_transferred_step3
def milk_cup1_after_step3 : ℚ := milk_transferred_step3

def mixture_transfer_step4 : ℚ := (tea_cup1_after_step3 + milk_cup1_after_step3) / 4
def tea_ratio_cup1_step4 : ℚ := tea_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)
def milk_ratio_cup1_step4 : ℚ := milk_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)

def tea_transferred_step4 : ℚ := mixture_transfer_step4 * tea_ratio_cup1_step4
def milk_transferred_step4 : ℚ := mixture_transfer_step4 * milk_ratio_cup1_step4

def final_tea_cup1 : ℚ := tea_cup1_after_step3 - tea_transferred_step4
def final_milk_cup1 : ℚ := milk_cup1_after_step3 - milk_transferred_step4
def final_total_liquid_cup1 : ℚ := final_tea_cup1 + final_milk_cup1

theorem fraction_of_milk_in_cup1 : final_milk_cup1 / final_total_liquid_cup1 = 3/8 := by
  sorry

end fraction_of_milk_in_cup1_l438_438987


namespace last_four_digits_of_m_smallest_l438_438609

theorem last_four_digits_of_m_smallest (m : ℕ) (h1 : m > 0)
  (h2 : m % 6 = 0) (h3 : m % 8 = 0)
  (h4 : ∀ d, d ∈ (m.digits 10) → d = 2 ∨ d = 7)
  (h5 : 2 ∈ (m.digits 10)) (h6 : 7 ∈ (m.digits 10)) :
  (m % 10000) = 2722 :=
sorry

end last_four_digits_of_m_smallest_l438_438609


namespace B_eq_A_pow_2_l438_438132

def A : ℕ → ℚ
| 0 := 1
| n+1 := (A n + 2) / (A n + 1)

def B : ℕ → ℚ
| 0 := 1
| n+1 := (B n ^ 2 + 2) / (2 * B n)

theorem B_eq_A_pow_2 (n : ℕ) : B (n + 1) = A (2 ^ n) :=
sorry

end B_eq_A_pow_2_l438_438132


namespace abs_difference_l438_438307

theorem abs_difference (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 8) : 
  |a - b| = 2 * Real.sqrt 10 :=
by
  sorry

end abs_difference_l438_438307


namespace distance_between_D_and_E_l438_438458

theorem distance_between_D_and_E 
  (A B C D E P : Type)
  (d_AB : ℕ) (d_BC : ℕ) (d_AC : ℕ) (d_PC : ℕ) 
  (AD_parallel_BC : Prop) (AB_parallel_CE : Prop) 
  (distance_DE : ℕ) :
  d_AB = 15 →
  d_BC = 18 → 
  d_AC = 21 → 
  d_PC = 7 → 
  AD_parallel_BC →
  AB_parallel_CE →
  distance_DE = 15 :=
by
  sorry

end distance_between_D_and_E_l438_438458


namespace tangent_line_at_P_l438_438198

open Real

noncomputable def curve (a : ℝ) (x : ℝ) := x / (x + a)

theorem tangent_line_at_P : 
  let P := (-1 : ℝ, -1 : ℝ)
  let a := 2
  in curve a (-1) = -1 → 
     ∃ (m : ℝ) (b : ℝ), (∀ x, (deriv (λ x, curve a x)) x = m) ∧ curve a (-1) = -1 ∧ (1 = 2 ∧ m = 2) :=
by
  let P := (-1 : ℝ, -1 : ℝ)
  let a := 2
  have h1: curve a (-1) = -1,
  { dsimp [curve], ring_nf, norm_num, },
  have h2: deriv (λ x, curve a x) (-1) = 2,
  { dsimp [curve], simp [deriv], ring, },
  
  use 2,
  use 1,
  split,
  { intro x, rw h2 },
  split,
  { exact h1 },
  split,
  { exact h1 },
  { ring_nf }
  done



end tangent_line_at_P_l438_438198


namespace problem_l438_438216

noncomputable def f (x w : ℝ) : ℝ := (Real.sin (w * x))^2 + Real.sqrt 3 * Real.sin (w * x) * Real.sin (w * x + Real.pi / 2)

theorem problem
  (w : ℝ)
  (hx : w > 0)
  (hf_period : ∀ x, f x w = f (x + Real.pi) w) :
  (w = 2)
  ∧ (∀ k : ℤ, (-Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (Real.pi / 3 + k * Real.pi) → ∃ (a b : ℝ), a < b ∧ strict_mono_incr_on (λ x, f x w) a b)
  ∧ (∀ x ∈ Set.Icc 0 (2 * Real.pi / 3), 0 ≤ f x w ∧ f x w ≤ 3 / 2) :=
by sorry

end problem_l438_438216


namespace sum_n_k_eq_8_l438_438358

noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := n.choose k

theorem sum_n_k_eq_8
  (n k : ℕ)
  (h1 : (binomial n k) = 1 * (binomial n (k + 1)) / 3)
  (h2 : (binomial n (k + 1)) = 3 * (binomial n (k + 2)) / 5) :
  n + k = 8 := by
  sorry

end sum_n_k_eq_8_l438_438358


namespace domain_F_l438_438939

variable {α : Type*}

def domain_f_shifted (x : α) : Set α := { y | y + 3 ∈ Set.Icc (-5 : α) (-2) }
def F (x : α) [OrderedAddCommGroup α] : α := sorry

theorem domain_F (α : Type*) [OrderedAddCommGroup α] : 
  (∀ x, domain_f_shifted x → x ∈ Set.Icc (-4 : α) (-3)) :=
sorry

end domain_F_l438_438939


namespace necessary_but_not_sufficient_l438_438551

variables {α β : Type} [plane α] [plane β] [m : line]

-- assuming the basic geometric constructs
axiom plane_perpendicular (α β : Type) [plane α] [plane β] : Prop
def α_perp_β (α β : Type) [plane α] [plane β] := plane_perpendicular α β

axiom line_in_plane (m : line) (α : Type) [plane α] : Prop
def m_in_α (m : line) (α : Type) [plane α] := line_in_plane m α

axiom line_perpendicular_to_plane (m : line) (β : Type) [plane β] : Prop
def m_perp_β (m : line) (β : Type) [plane β] := line_perpendicular_to_plane m β

-- Theorem statement
theorem necessary_but_not_sufficient (α β : Type) [plane α] [plane β] [m : line] 
    (h1 : α.perp_β β) (h2 : m.in_α α) : ¬ ((m.perp_β β) ↔ (α.perp_β β)) := sorry

end necessary_but_not_sufficient_l438_438551


namespace max_unique_triangles_has_15_elements_l438_438080

def triangle (a b c : ℕ) : Prop :=
  a ≥ b ∧ b ≥ c ∧ b + c > a ∧ a < 6 ∧ b < 6 ∧ c < 6

def unique_triangles (T : set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ t1 t2 ∈ T, t1 ≠ t2 → ¬(congruent t1 t2 ∨ similar t1 t2)

noncomputable def max_unique_triangles : set (ℕ × ℕ × ℕ) :=
{ t | ∃ (a b c : ℕ), triangle a b c ∧ (a, b, c) = t }

#check max_unique_triangles

theorem max_unique_triangles_has_15_elements :
  ∃ S : set (ℕ × ℕ × ℕ), unique_triangles S ∧ set.card S = 15 :=
sorry

end max_unique_triangles_has_15_elements_l438_438080


namespace Lin_trip_time_l438_438623

theorem Lin_trip_time
  (v : ℕ) -- speed on the mountain road in miles per minute
  (h1 : 80 = d_highway) -- Lin travels 80 miles on the highway
  (h2 : 20 = d_mountain) -- Lin travels 20 miles on the mountain road
  (h3 : v_highway = 2 * v) -- Lin drives twice as fast on the highway
  (h4 : 40 = 20 / v) -- Lin spent 40 minutes driving on the mountain road
  : 40 + 80 = 120 :=
by
  -- proof steps would go here
  sorry

end Lin_trip_time_l438_438623


namespace sum_of_first_10_common_elements_is_correct_l438_438844

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric sequence
def b (k : ℕ) : ℕ := 10 * 2^k

-- Define a function that finds the common elements in both sequences
def common_elements (N : ℕ) : List ℕ :=
List.filter (λ x, ∃ n k, x = a n ∧ x = b k) (List.range (N + 1))

-- Define the sum of the first 10 common elements
def sum_first_10_common_elements : ℕ :=
(List.sum (List.take 10 (common_elements 100)))

theorem sum_of_first_10_common_elements_is_correct :
  sum_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_is_correct_l438_438844


namespace smallest_x_l438_438839

theorem smallest_x (x : ℕ) : (x + 3457) % 15 = 1537 % 15 → x = 15 :=
by
  sorry

end smallest_x_l438_438839


namespace range_g_l438_438123

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := x^(2*k) - 1

theorem range_g (k : ℝ) (hk : k > 0) : 
  set.range (λ x, g k x) = set.Ici 0 := sorry

end range_g_l438_438123


namespace sum_first_10_common_elements_l438_438840

/-- Definition of arithmetic progression term -/
def arith_term (n : ℕ) : ℕ := 4 + 3 * n

/-- Definition of geometric progression term -/
def geom_term (k : ℕ) : ℕ := 10 * 2 ^ k

/-- Verify if two terms are common elements -/
def is_common_element (n k : ℕ) : Prop := arith_term n = geom_term k

/-- Equivalence proof of sum of first 10 common elements -/
theorem sum_first_10_common_elements : 
  Σ (n k : ℕ) (H : is_common_element n k), (arith_term n) = 3495250 :=
begin
  sorry
end

end sum_first_10_common_elements_l438_438840


namespace substance_volume_new_conditions_l438_438660

/-- Given the mass of a cubic meter of a substance and its density change factor under new conditions,
    prove that the volume in cubic centimeters of 1 gram of the substance is as specified. -/
theorem substance_volume_new_conditions
  (mass_per_cubic_meter : ℝ := 500)
  (density_change_factor : ℝ := 1.25)
  (grams_to_kg : ℝ := 0.001)
  (cubic_meters_to_cubic_cms : ℝ := 1_000_000)
  (initial_density := mass_per_cubic_meter)
  (new_density := initial_density * density_change_factor)
  : 
  (grams_to_kg / new_density * cubic_meters_to_cubic_cms) = 1.6
:= 
  sorry

end substance_volume_new_conditions_l438_438660


namespace MH_never_eq_MK_l438_438312

theorem MH_never_eq_MK 
  (BC HK : Line) 
  (B C H K : Point) 
  (M : Point)
  (x : Real) 
  (θ φ : Real) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (hBM : dist B M = x * dist B C)
  (hMC : dist M C = (1 - x) * dist B C)
  (hθ : ∃ θ, angle B H K = θ)
  (hφ : ∃ φ, angle C K H = φ) :
  ¬(dist M H = dist M K) := 
sorry

end MH_never_eq_MK_l438_438312


namespace sum_of_first_10_common_elements_is_correct_l438_438843

-- Define the arithmetic sequence
def a (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric sequence
def b (k : ℕ) : ℕ := 10 * 2^k

-- Define a function that finds the common elements in both sequences
def common_elements (N : ℕ) : List ℕ :=
List.filter (λ x, ∃ n k, x = a n ∧ x = b k) (List.range (N + 1))

-- Define the sum of the first 10 common elements
def sum_first_10_common_elements : ℕ :=
(List.sum (List.take 10 (common_elements 100)))

theorem sum_of_first_10_common_elements_is_correct :
  sum_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_is_correct_l438_438843


namespace largest_t_l438_438854

noncomputable def score {k n : ℕ} (points : fin n → fin k → ℝ) (i : fin n) : ℝ :=
  ∏ j : fin k, (finset.card {i' : fin n | (points i').val.erase j = (points i).val.erase j})

noncomputable def t_power_mean {t : ℝ} {n : ℕ} (scores : fin n → ℝ) : ℝ :=
  if t ≠ 0 then ((finset.univ.sum (λ i, scores i ^ t)) / n) ^ (1 / t)
  else real.sqrt (finset.univ.prod (λ i, scores i))

theorem largest_t {k n : ℕ} (hk : 1 < k) (points : fin n → fin k → ℝ) :
  ∃ (t : ℝ), (∀ (scores : fin n → ℝ), (t_power_mean scores ≤ n ↔ t ≤ 1 / (k - 1))) :=
begin
  sorry
end

end largest_t_l438_438854


namespace gambler_largest_amount_proof_l438_438746

noncomputable def largest_amount_received_back (initial_amount : ℝ) (value_25 : ℝ) (value_75 : ℝ) (value_250 : ℝ) 
                                               (total_lost_chips : ℝ) (coef_25_75_lost : ℝ) (coef_75_250_lost : ℝ) : ℝ :=
    initial_amount - (
    coef_25_75_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_25 +
    (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_75 +
    coef_75_250_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_250)

theorem gambler_largest_amount_proof :
    let initial_amount := 15000
    let value_25 := 25
    let value_75 := 75
    let value_250 := 250
    let total_lost_chips := 40
    let coef_25_75_lost := 2 -- number of lost $25 chips is twice the number of lost $75 chips
    let coef_75_250_lost := 2 -- number of lost $250 chips is twice the number of lost $75 chips
    largest_amount_received_back initial_amount value_25 value_75 value_250 total_lost_chips coef_25_75_lost coef_75_250_lost = 10000 :=
by {
    sorry
}

end gambler_largest_amount_proof_l438_438746


namespace beadshop_wednesday_profit_l438_438736

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end beadshop_wednesday_profit_l438_438736


namespace angle_DAE_of_isosceles_triangle_and_pentagon_l438_438776

theorem angle_DAE_of_isosceles_triangle_and_pentagon (A B C D E F : Type) [is_regular_pentagon [BCDEF]]
  (isosceles : is_isosceles_triangle A B C (AB = AC))
  (common_side : shares_side BCDEF BC) :
  ∠DAE = 12 :=
sorry

end angle_DAE_of_isosceles_triangle_and_pentagon_l438_438776


namespace penalty_kicks_required_l438_438350

constant num_players : ℕ
constant num_goalkeepers : ℕ

def outfield_players (total_players goalkeepers : ℕ) : ℕ := total_players - goalkeepers

theorem penalty_kicks_required 
  (h1 : num_players = 22)
  (h2 : num_goalkeepers = 4) :
  outfield_players 22 1 * 4 = 84 :=
by {
  -- Define the number of players and goalkeepers
  have h3: outfield_players 22 1 = 21 := rfl,
  -- Calculate the total number of kicks
  show outfield_players 22 1 * 4 = 84,
  rw h3,
  norm_num,
  sorry
}

end penalty_kicks_required_l438_438350


namespace magnitude_of_pure_imaginary_squared_l438_438936

noncomputable def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem magnitude_of_pure_imaginary_squared (a : ℝ) (h : isPureImaginary ((1 + a*complex.I)^2)) : complex.abs ((1 + a*complex.I)^2) = 2 := 
sorry

end magnitude_of_pure_imaginary_squared_l438_438936


namespace train_length_approx_l438_438766

noncomputable def train_length 
  (train_speed : ℝ)    -- speed of train in km/hr
  (trolley_speed : ℝ)  -- speed of trolley in km/hr, opposite direction
  (time : ℝ)           -- time in seconds when train passes trolley
  : ℝ := 
  let relative_speed_m_per_s := (train_speed + trolley_speed) * (5 / 18)
  in relative_speed_m_per_s * time

theorem train_length_approx : 
  train_length 60 12 5.4995600351971845 ≈ 109.99 := by
  sorry

end train_length_approx_l438_438766


namespace partition_students_l438_438678

theorem partition_students (n r : ℕ) (k : ℕ) (numbers : Fin n → Fin r → ℕ) :
  (∀ i j : Fin n, i ≠ j → ∀ a b : Fin r, numbers i a ≠ numbers j b) →
  k ≤ 4 * r →
  (∀ i : Fin n, ∃ c : Fin k, ∀ j : Fin n, i ≠ j → (∀ a b : Fin r, numbers i a = numbers j b) → 
    (numbers i a ≤ (numbers j b)! ∨ numbers i a ≥ (numbers j b)! + 1)) →
  ∃ partition : Fin n → Fin k, ∀ i j : Fin n, (partition i = partition j) →
    ∀ a b : Fin r, ¬((numbers i a - 1)! < numbers j b ∧ numbers j b < (numbers i a + 1)! + 1). 
    sorry

end partition_students_l438_438678


namespace dragon_jewels_end_l438_438745

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l438_438745


namespace root_conjugate_l438_438332

noncomputable def P (z : ℂ) := ∑ i in (finset.range (n + 1)), (a i : ℝ) * (z ^ i)

theorem root_conjugate (a : ℕ → ℝ) (z0 : ℂ) (hPz0 : P a z0 = 0) : P a (conj z0) = 0 :=
by
  -- Proof steps would go here
  sorry

end root_conjugate_l438_438332


namespace kirill_is_62_5_l438_438989

variable (K : ℝ)

def kirill_height := K
def brother_height := K + 14
def sister_height := 2 * K
def total_height := K + (K + 14) + 2 * K

theorem kirill_is_62_5 (h1 : total_height K = 264) : K = 62.5 := by
  sorry

end kirill_is_62_5_l438_438989


namespace bread_last_days_is_3_l438_438680

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end bread_last_days_is_3_l438_438680


namespace number_of_divisors_of_3003_l438_438828

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l438_438828


namespace carol_meets_alice_in_30_minutes_l438_438773

def time_to_meet (alice_speed carol_speed initial_distance : ℕ) : ℕ :=
((initial_distance * 60) / (alice_speed + carol_speed))

theorem carol_meets_alice_in_30_minutes :
  time_to_meet 4 6 5 = 30 := 
by 
  sorry

end carol_meets_alice_in_30_minutes_l438_438773


namespace pump_out_time_l438_438645

theorem pump_out_time
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (rate : ℝ)
  (H_length : length = 50)
  (H_width : width = 30)
  (H_depth : depth = 1.8)
  (H_rate : rate = 2.5) : 
  (length * width * depth) / rate / 60 = 18 :=
by
  sorry

end pump_out_time_l438_438645


namespace ratio_wx_l438_438220

theorem ratio_wx (w x y : ℚ) (h1 : w / y = 3 / 4) (h2 : (x + y) / y = 13 / 4) : w / x = 1 / 3 :=
  sorry

end ratio_wx_l438_438220


namespace abs_value_eq_two_l438_438094

theorem abs_value_eq_two (x : ℝ) (y : ℝ) (z : ℝ) (w : ℝ) :
  (x = |-2^{-1}|) →
  (y = |(±(1/2))^{-2}|) →
  (z = |± 2|) →
  (w = |(-2)^{-1}|) →
  (|z| = 2) :=
by
  sorry

end abs_value_eq_two_l438_438094


namespace seated_students_count_l438_438466

theorem seated_students_count :
  ∀ (S T standing_students total_attendees : ℕ),
    T = 30 →
    standing_students = 25 →
    total_attendees = 355 →
    total_attendees = S + T + standing_students →
    S = 300 :=
by
  intros S T standing_students total_attendees hT hStanding hTotalAttendees hEquation
  sorry

end seated_students_count_l438_438466


namespace angle_bisector_BF_of_triangle_ABC_l438_438275

/-- In triangle ABC, CD is the angle bisector of ACB,
AB = BC,
BD = BK,
BL = CL.
Prove that BF is the angle bisector of CBE. -/
theorem angle_bisector_BF_of_triangle_ABC (A B C D E F K L : Type) [IsTriangle A B C]
  (h1: AngleBisector C D (Angle A C B))
  (h2: EuclideanGeometry.AB = EuclideanGeometry.BC)
  (h3: SameLength B D B K)
  (h4: SameLength B L C L)
:
  AngleBisector B F (Angle C B E) := 
sorry

end angle_bisector_BF_of_triangle_ABC_l438_438275


namespace isabella_hair_length_l438_438979

theorem isabella_hair_length (L₀ growth_rate months : ℕ) (h₀ : L₀ = 18) (r : growth_rate = 2) (t : months = 5) :
  L₀ + growth_rate * months = 28 :=
by
  rw [h₀, r, t]
  -- Proof steps would go here if we were to complete it.
  sorry

end isabella_hair_length_l438_438979


namespace Karl_miles_driven_l438_438286

theorem Karl_miles_driven
  (gas_per_mile : ℝ)
  (tank_capacity : ℝ)
  (initial_gas : ℝ)
  (first_leg_miles : ℝ)
  (refuel_gallons : ℝ)
  (final_gas_fraction : ℝ)
  (total_miles_driven : ℝ) :
  gas_per_mile = 30 →
  tank_capacity = 16 →
  initial_gas = 16 →
  first_leg_miles = 420 →
  refuel_gallons = 10 →
  final_gas_fraction = 3 / 4 →
  total_miles_driven = 420 :=
by
  sorry

end Karl_miles_driven_l438_438286


namespace find_b_l438_438822

noncomputable def b : ℝ := 1 / 19683

theorem find_b (b_prop : log b 729 = -2 / 3) : b = 1 / 19683 :=
sorry

end find_b_l438_438822


namespace quadratic_roots_l438_438833

-- Definitions based on problem conditions
def sum_of_roots (p q : ℝ) : Prop := p + q = 12
def abs_diff_of_roots (p q : ℝ) : Prop := |p - q| = 4

-- The theorem we want to prove
theorem quadratic_roots : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ p q, sum_of_roots p q ∧ abs_diff_of_roots p q → a * (x - p) * (x - q) = x^2 - 12 * x + 32) := sorry

end quadratic_roots_l438_438833


namespace contest_inequality_l438_438253

theorem contest_inequality (m n k : ℕ) (h1 : n ≥ 3) (h2 : odd n) (h3 : ∀ i j : ℕ, i ≠ j → agrees_on_at_most k i j) :
  k / m ≥ (n - 1) / (2 * n) :=
sorry

-- Definitions and assumptions required for the theorem
def agrees_on_at_most (k : ℕ) (i j : ℕ) : Prop := 
  -- Placeholder definition, should express that i and j agree on at most k candidates
sorry

end contest_inequality_l438_438253


namespace arithmetic_sequence_a3_l438_438867

-- Define the sequence
def sequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, a (n + 1) = a n + 2) ∧ (a 1 = 2)

-- Define the proof statement that relates the conditions to the answer
theorem arithmetic_sequence_a3 (a : ℕ → ℤ) (h : sequence a) : a 3 = 6 := 
sorry

end arithmetic_sequence_a3_l438_438867


namespace difference_of_numbers_l438_438374

theorem difference_of_numbers 
  (a b : ℕ) 
  (h1 : a + b = 23976)
  (h2 : b % 8 = 0)
  (h3 : a = 7 * b / 8) : 
  b - a = 1598 :=
sorry

end difference_of_numbers_l438_438374


namespace father_age_in_years_l438_438962

def talias_current_age : ℕ := 20 - 7
def talias_moms_age : ℕ := 3 * talias_current_age
def talias_fathers_current_age : ℕ := 36

theorem father_age_in_years (n : ℕ) (h1 : talias_current_age = 13) (h2 : talias_moms_age = 39) (h3 : talias_fathers_current_age = 36) :
  talias_fathers_current_age + n = talias_moms_age := 
by 
  have h: talias_current_age = 13 := h1,
  have m: talias_moms_age = 39 := h2,
  have f: talias_fathers_current_age = 36 := h3,
  show 36 + n = 39, from sorry

example : ∃ (n : ℕ), talias_fathers_current_age + n = talias_moms_age :=
  ⟨3, father_age_in_years 3 rfl rfl rfl⟩

end father_age_in_years_l438_438962


namespace least_possible_bananas_l438_438857

open_locale classical

noncomputable def condition1 (b b1 b2 b3 b4 : ℕ) : Prop :=
  let monkey1 := 3 / 5 * b1 + 1 / 4 * (b2 + b3 + b4),
      monkey2 := 1 / 2 * b2 + 1 / 4 * (b1 + b3 + b4),
      monkey3 := 1 / 4 * b3 + 1 / 4 * (b1 + b2 + b4),
      monkey4 := 1 / 8 * b4 + 1 / 4 * (b1 + b2 + b3)
  in 4 * monkey4 = monkey3 ∧
     3 * monkey4 = monkey2 ∧
     2 * monkey4 = monkey1

noncomputable def condition2 (b b1 b2 b3 b4 : ℕ) : Prop :=
  b1 + b2 + b3 + b4 = b

noncomputable def monkeys_conditions (b : ℕ) :=
  ∃ b1 b2 b3 b4, condition1 b b1 b2 b3 b4 ∧ condition2 b b1 b2 b3 b4

theorem least_possible_bananas : ∃ b, monkeys_conditions b ∧ b = 600 :=
sorry

end least_possible_bananas_l438_438857


namespace part_a_part_b_l438_438404

variables {f : ℝ → ℝ} {g : ℝ → ℝ} {φ : ℝ → ℝ}

-- Conditions on functions
def non_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def C1 (f : ℝ → ℝ) : Prop := Differentiable ℝ f ∧ Differentiable ℝ (f')
def boundary_condition (f : ℝ → ℝ) : Prop := f 0 = 0
def g_def (f : ℝ → ℝ) (g : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x → x ≤ 1 → g x = f x + (x - 1) * deriv f x

-- Conditions on φ
def phi_conditions (φ : ℝ → ℝ) : Prop := Convex ℝ (set.univ : set ℝ) φ ∧ Differentiable ℝ φ ∧ φ 0 = 0 ∧ φ 1 = 1

-- Proof statements
theorem part_a
  (h_f_non_decreasing : non_decreasing f)
  (h_f_C1 : C1 f)
  (h_f_boundary : boundary_condition f)
  (h_g_def : g_def f g)
  : (∫ x in 0..1, g x) = 0 := by
  sorry

theorem part_b
  (h_f_non_decreasing : non_decreasing f)
  (h_f_C1 : C1 f)
  (h_f_boundary : boundary_condition f)
  (h_g_def : g_def f g)
  (h_phi_conditions : phi_conditions φ)
  : (∫ t in 0..1, g (φ t)) ≤ 0 := by
  sorry

end part_a_part_b_l438_438404


namespace ramsey_tree_complete_l438_438210

theorem ramsey_tree_complete {s t : ℕ} (hs : s > 0) (ht : t > 0) (T : SimpleGraph V) (hT : T.is_tree ∧ T.order = t) :
  ramsey_number (λ G, G.subgraph_isomorphic T) (λ G, G.is_complete_graph s) = (s-1) * (t-1) + 1 :=
sorry

end ramsey_tree_complete_l438_438210


namespace num_two_digit_number_sum_reversal_eq_9_l438_438804

def is_two_digit_number_sum_reversal (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b) + (10 * b + a) = 110

theorem num_two_digit_number_sum_reversal_eq_9 : 
  (finset.univ.filter is_two_digit_number_sum_reversal).card = 9 :=
  sorry

end num_two_digit_number_sum_reversal_eq_9_l438_438804


namespace smallest_positive_period_f_is_pi_maximum_value_and_set_of_x_l438_438549

noncomputable def f (x : ℝ) : ℝ := (1 + real.sin (2 * x)) + (real.sin x - real.cos x) * (real.sin x + real.cos x)

theorem smallest_positive_period_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') := 
begin
  use π,
  split,
  { exact real.pi_pos },
  split,
  { intro x,
    sorry },
  { intros T' T'_pos hyp_period,
    sorry }
end

theorem maximum_value_and_set_of_x :
  ∃ M x_values, (∀ x, f x ≤ M) ∧ (x_values = {x : ℝ | f x = M}) ∧ (M = sqrt 2 + 1) ∧ 
  (x_values = {x | ∃ k : ℤ, x = (3 * real.pi / 8) + k * real.pi}) := 
begin
  use [sqrt 2 + 1, {x | ∃ k : ℤ, x = (3 * real.pi / 8) + k * real.pi}],
  split,
  { intro x,
    sorry },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end smallest_positive_period_f_is_pi_maximum_value_and_set_of_x_l438_438549


namespace derivative_of_y_l438_438563

noncomputable def y (x : ℝ) : ℝ := x^3 + x^(1/3) + cos x
noncomputable def dy_dx (x : ℝ) : ℝ := 3 * x^2 + (1 / 3) * x^(-2/3) - sin x

theorem derivative_of_y : ∀ x : ℝ, deriv y x = dy_dx x :=
by 
  intro x
  sorry

end derivative_of_y_l438_438563


namespace find_real_pairs_l438_438150

theorem find_real_pairs (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_real_pairs_l438_438150


namespace volume_of_body_l438_438187

theorem volume_of_body (p S d : ℝ) : 
  let V := 2 * d * S + π * p * d^2 + (4 / 3) * π * d^3 in
  ∀ convex : ℝ → ℝ → Prop, 
  convex.figure → 
  convex.perimeter = 2 * p → 
  convex.area = S → 
  body.volume (convex, d) = V :=
by
  sorry

end volume_of_body_l438_438187


namespace oranges_in_bowl_initially_l438_438411

open Real

def initialOranges {O : ℕ} : Prop :=
  let total_fruit_after_removal := 14 + O - 19
  0.7 * total_fruit_after_removal = 14

theorem oranges_in_bowl_initially : ∃ O : ℕ, initialOranges O → O = 25 :=
begin
  use 25,
  intro h,
  have h_eq : 0.7 * (25 - 5) = 14 := by sorry,  -- Detailed steps to solve equation skipped
  exact h_eq,
end

end oranges_in_bowl_initially_l438_438411


namespace trains_crossing_time_l438_438046

/-- Given two trains with specific lengths and speeds running in opposite directions,
    the time to cross each other is approximately 9.72 seconds. -/
theorem trains_crossing_time
  (train1_length : ℕ) (train2_length : ℕ)
  (train1_speed_kmh : ℕ) (train2_speed_kmh : ℕ)
  (train1_length_eq : train1_length = 110)
  (train2_length_eq : train2_length = 160)
  (train1_speed_eq : train1_speed_kmh = 60)
  (train2_speed_eq : train2_speed_kmh = 40) :
  let train1_speed_ms := (train1_speed_kmh : ℝ) * (5/18)
  let train2_speed_ms := (train2_speed_kmh : ℝ) * (5/18)
  let relative_speed := train1_speed_ms + train2_speed_ms
  let total_distance := (train1_length : ℝ) + (train2_length : ℝ)
  let crossing_time := total_distance / relative_speed
  crossing_time ≈ 9.72 := by
{
  sorry
}

end trains_crossing_time_l438_438046


namespace points_calculation_l438_438262

noncomputable def points_per_goblin := 3
noncomputable def points_per_troll := 5
noncomputable def points_per_dragon := 10
noncomputable def bonus_per_combination := 7

noncomputable def total_goblins := 14
noncomputable def total_trolls := 15
noncomputable def total_dragons := 4

noncomputable def defeated_goblins := (7 / 10 : ℝ) * total_goblins |>.to_nat
noncomputable def defeated_trolls := (2 / 3 : ℝ) * total_trolls |>.to_nat
noncomputable def defeated_dragons := 1

noncomputable def points_from_goblins := defeated_goblins * points_per_goblin
noncomputable def points_from_trolls := defeated_trolls * points_per_troll
noncomputable def points_from_dragons := defeated_dragons * points_per_dragon
noncomputable def bonus_points := min defeated_goblins (min defeated_trolls defeated_dragons) * bonus_per_combination

noncomputable def total_points := points_from_goblins + points_from_trolls + points_from_dragons + bonus_points

theorem points_calculation : total_points = 94 ∧ bonus_points = 7 :=
by
  sorry

end points_calculation_l438_438262


namespace correct_number_of_propositions_l438_438308

open Function

-- Define the types and conditions
variables {Plane : Type} {Line : Type}
variable {α β : Plane}
variable {l : Line}

-- Definitions assuming the given conditions
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel (l : Line) (α : Plane) : Prop := sorry
def subset (l : Line) (β : Plane) : Prop := sorry

-- Propositions
def proposition_1 : Prop := perpendicular l α ∧ perpendicular α β → subset l β
def proposition_2 : Prop := parallel l α ∧ parallel α β → subset l β
def proposition_3 : Prop := perpendicular l α ∧ parallel α β → perpendicular l β
def proposition_4 : Prop := parallel l α ∧ perpendicular α β → perpendicular l β

-- The statement to be proved
theorem correct_number_of_propositions : (1 : ℕ) := by
  sorry

end correct_number_of_propositions_l438_438308


namespace gcd_product_square_l438_438304

-- Definitions of natural numbers and gcd function available in Mathlib

noncomputable def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem gcd_product_square (x y z : ℕ) (h : (1 / x : ℚ) - 1 / y = 1 / z) : 
  is_square (Nat.gcd x y z * (y - x)) :=
sorry

end gcd_product_square_l438_438304


namespace incorrect_statement_B_l438_438169

-- Define linear function and conditions
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Conditions: k < 0 and b > 0
variables {k b : ℝ} (h_k : k < 0) (h_b : b > 0)

-- Statements
def statement_A : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function k b x₁ > linear_function k b x₂
def statement_B : Prop := linear_function k b (-1) = -2
def statement_C : Prop := linear_function k b 0 = b
def statement_D : Prop := ∀ (x : ℝ), x > -b / k → linear_function k b x < 0

-- We need to prove that statement B is incorrect
theorem incorrect_statement_B : ¬ statement_B :=
by
  -- B is incorrect because it leads to a contradiction with the given conditions
  intro h
  have h1 : k * -1 + b = -2 := h
  have h2 : b = k + -2 := by linarith
  have h3 : k + -2 < 0 := by linarith [h_k]
  have h4 : b ≤ 0 := by linarith [h3]
  contradiction

end incorrect_statement_B_l438_438169


namespace normal_prob_l438_438219

noncomputable theory

def normal_distribution (mean variance : ℝ) : Prop := sorry

variable {ξ : ℝ → ℝ}

def P (X : Set ℝ) : ℝ := sorry

theorem normal_prob :
  (normal_distribution 0 σ^2) ∧ (P {x | -2 ≤ ξ x ∧ ξ x ≤ 2} = 0.4) →
  P {x | ξ x > 2} = 0.3 :=
sorry

end normal_prob_l438_438219


namespace second_machine_equation_l438_438413

-- Let p1_rate and p2_rate be the rates of printing for machine 1 and 2 respectively.
-- Let x be the unknown time for the second machine to print 500 envelopes.

theorem second_machine_equation (x : ℝ) :
    (500 / 8) + (500 / x) = (500 / 2) :=
  sorry

end second_machine_equation_l438_438413


namespace circle_radius_squared_l438_438064

theorem circle_radius_squared {r : ℝ} 
  (AB CD : ℝ) (P A B C D : Type)
  [A = P] [B = P] [C = P] [D = P]
  [chord_AB : AB = 12] 
  [chord_CD : CD = 9]
  [angle_APD : real.angle = 60]
  [BP_length : BP = 10] :
  r ^ 2 = 48 := 
  sorry

end circle_radius_squared_l438_438064


namespace infinitely_many_n_l438_438339

theorem infinitely_many_n (S : Set ℕ) :
  (∀ n ∈ S, n > 0 ∧ (n ∣ 2 ^ (2 ^ n + 1) + 1) ∧ ¬ (n ∣ 2 ^ n + 1)) ∧ S.Infinite :=
sorry

end infinitely_many_n_l438_438339


namespace dice_probability_l438_438144

theorem dice_probability :
  let n := 15 in
  let k := 3 in
  let p := 1 / 6 in
  let q := 5 / 6 in
  let binom_n_k := Nat.choose n k in
  let P := binom_n_k * p^k * q^(n - k) in
  Real.abs (P - 0.237) < 0.001 :=
by
  sorry

end dice_probability_l438_438144


namespace spider_total_distance_l438_438428

-- Define the radius and the third journey length
def radius : ℝ := 50
def third_journey : ℝ := 70

-- Define the diameter (length of the straight walk through the center)
def diameter := 2 * radius

-- State the problem in a Lean 4 theorem
theorem spider_total_distance : 
  let total_distance := diameter + third_journey + real.sqrt (diameter^2 - third_journey^2)
  total_distance = 170 + real.sqrt 5100 := sorry

end spider_total_distance_l438_438428


namespace ellipse_standard_eq_fixed_point_exists_const_dot_product_l438_438510

-- Define the ellipse and initial conditions.
variables {a b : ℝ} (C : Set (ℝ × ℝ)) (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 

-- Condition: Equation of ellipse C
def ellipse_eq : Prop := ∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / a^2) + (y^2 / b^2) = 1

-- Question 1: Standard equation of the ellipse
theorem ellipse_standard_eq (ha : a = sqrt 2) (hb : b = 1) :
  ellipse_eq C a b ↔ ∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / 2) + y^2 = 1 := sorry

-- Define conditions for fixed point on the x-axis and constant dot product
variables {A B : ℝ × ℝ} (E : ℝ × ℝ) (fixed_point : E = (5 / 4, 0))
def EA_dot_EB_constant (EA EB : ℝ × ℝ) : Prop :=
  (∃ k : ℝ, k ≠ 0 ∧ let x_A := A.1, y_A := A.2, x_B := B.1, y_B := B.2, x₀ := E.1 in
    (1 + k^2) * x_A * x_B - (x₀ + k^2) * (x_A + x_B) + x₀^2 + k^2 = -7 / 16)

-- Question 2: Existence of fixed point E with constant value
theorem fixed_point_exists_const_dot_product :
  ∃ E, E = (5 / 4, 0) ∧ ∀ (A B : ℝ × ℝ), EA_dot_EB_constant A B E := sorry

end ellipse_standard_eq_fixed_point_exists_const_dot_product_l438_438510


namespace variance_of_data_set_l438_438533

def data_set := [6, 7, 8, 8, 9, 10]

def mean (xs : List ℕ) : ℚ :=
  xs.foldl (· + ·) 0 / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).foldl (· + ·) 0 / xs.length

theorem variance_of_data_set : variance data_set = 5 / 3 :=
by
  sorry

end variance_of_data_set_l438_438533


namespace howlers_coach_loudvoice_lineups_l438_438646

noncomputable def number_of_permissible_lineups : ℕ :=
let total_lineups := Nat.choose 15 6 in
let restricted_lineups := Nat.choose 12 3 in
total_lineups - restricted_lineups

theorem howlers_coach_loudvoice_lineups : number_of_permissible_lineups = 4785 := by
  sorry

end howlers_coach_loudvoice_lineups_l438_438646


namespace history_only_students_l438_438577

theorem history_only_students 
  (total_students : ℕ)
  (history_students stats_students physics_students chem_students : ℕ) 
  (hist_stats hist_phys hist_chem stats_phys stats_chem phys_chem all_four : ℕ) 
  (h1 : total_students = 500)
  (h2 : history_students = 150)
  (h3 : stats_students = 130)
  (h4 : physics_students = 120)
  (h5 : chem_students = 100)
  (h6 : hist_stats = 60)
  (h7 : hist_phys = 50)
  (h8 : hist_chem = 40)
  (h9 : stats_phys = 35)
  (h10 : stats_chem = 30)
  (h11 : phys_chem = 25)
  (h12 : all_four = 20) : 
  (history_students - hist_stats - hist_phys - hist_chem + all_four) = 20 := 
by 
  sorry

end history_only_students_l438_438577


namespace rearrange_into_square_l438_438459

theorem rearrange_into_square (area : ℕ)
  (h1 : area = 36)
  (side : ℕ)
  (h2 : side = Int.sqrt area)
  (side_length : ℕ)
  (h3 : side_length = 6)
  (cut_line : String)
  (h4 : cut_line = "line AB")
  : ∃ parts : list (Π (p : ℕ), p = 2), 
      ∃ rearranged_square : Π (l : ℕ), l = side,
      rearranged_square (side_length) = 6 ∧ area = 36 := 
sorry

end rearrange_into_square_l438_438459


namespace range_of_a_l438_438914

def setA (a : ℝ) : set ℝ := {x | |x - a| < 2}
def setB : set ℝ := {x | (2 * x - 1) / (x + 2) < 1}

theorem range_of_a (a : ℝ) (h : setA(a) ⊆ setB) : 0 ≤ a ∧ a ≤ 1 := 
by sorry

end range_of_a_l438_438914


namespace maximum_extra_credit_students_l438_438574

theorem maximum_extra_credit_students (n : ℕ) (h : n = 150) (s : ℕ → ℝ)
  (h1 : ∀ i, i ≠ 1 → s i = 100) (h2 : s 1 = 80) :
  ∃ k, k = 149 ∧ ∀ i, i ≠ 1 → s i > (∑ i in finset.range n, s i) / n := by
  intro n h s h1 h2
  have total_score := (149 : ℝ) * 100 + 80
  have mean_score := total_score / 150
  have mean_ineq : 100 > mean_score := by
    calc
      100 > 99.8666 : by norm_num
      ... > mean_score : by linarith
  use 149
  split
  · rfl
  · intro i hi
    rw h1 i hi
    exact mean_ineq
  sorry

end maximum_extra_credit_students_l438_438574


namespace unique_B1_l438_438771

-- Define the conditions of the problem
variables (A : fin 2020 → ℝ × ℝ) (θ : fin 2020 → ℝ)
-- Angles are in the interval (0, π)
variables (θ_pos : ∀ i, 0 < θ i) (θ_lt_pi : ∀ i, θ i < π)
-- The sum of the angles is 1010π
variable (θ_sum : ∑ i, θ i = 1010 * π)
-- Define the exterior isosceles triangles with vertex B_i
variables (B : fin 2020 → ℝ × ℝ)
-- The triangles are isosceles with the specified angles
variable (isosceles_property : ∀ i, dist (B i) (A i) = dist (B i) (A (⟨i.1 + 1, by linarith [i.2]⟩)))

-- The goal is to show that B1 is uniquely determined
theorem unique_B1 (jason_knows : fin 2020 → (ℝ × ℝ)) (jason_knows_angles : fin 2020 → ℝ) :
  ∀ B2_to_B2020 : (fin 2019 → (ℝ × ℝ)),
  ∀ θ_known : fin 2020 → ℝ,
  ∃! B1 : (ℝ × ℝ),
    -- B1 together with known B2 to B2020 and angles satisfies the isosceles property and sum condition
    (∀ i, dist (B1) (A 0) = dist (B1) (A 1) ∧
     ∀ j from 1 to 2019,
     dist (B2_to_B2020 j) (A j) = dist (B2_to_B2020 j) (A (⟨j.1 + 1, by linarith [j.2]⟩))) ∧
    (∑ i, θ i = 1010 * π) :=
sorry

end unique_B1_l438_438771


namespace round_to_nearest_thousandth_l438_438335

-- Definition of the repeating decimal 67.326326...
def repeating_decimal_67point326 : ℝ := 67 + 326 / 999

-- The desired property: rounding to the nearest thousandth
theorem round_to_nearest_thousandth : Real.round_to_nearest_thousandth repeating_decimal_67point326 = 67.326 :=
by
  sorry

end round_to_nearest_thousandth_l438_438335


namespace marksman_probability_l438_438418

-- Definitions for the conditions
def p : ℝ := 0.9
def n : ℕ := 4

-- Statement of the theorem
theorem marksman_probability (p_eq : p = 0.9) (n_eq : n = 4) :
  (1. prob_third_shot : p = 0.9) ∧
  (2. prob_three_hits : ¬(4 * p^3 * (1 - p) = (p^3 * 0.1))) ∧
  (3. prob_at_least_once : (1 - (1 - p)^n = 1 - 0.1^4)) ∧
  (4. average_hits : (n * p = 3.6)) :=
by
  split
  . exact p_eq
  . split
  . intro h
  . have : 4 * p^3 * (1 - p) = 4 * p^3 * (1 - 0.9) := by rw [p_eq]
  have not_h := by show 4 * p^3 * (1 - 0.9) ≠ (p^3 * 0.1)
  contradiction
  . split
  . exact 1 - (1 - p_eq)^n = 1 - 0.1^4
  . exact n * p_eq = 3.6
  done

end marksman_probability_l438_438418


namespace fountain_area_l438_438760

-- Define the points A, B, C, and D and their distances
variable (A B C D : Type)
variable (dist_AB dist_AD dist_DC : ℝ)
variable h1 : dist_AB = 20
variable h2 : dist_AD = dist_AB / 2
variable h3 : dist_DC = 12

-- Define the radius and area of the circular base
noncomputable def radius : ℝ := real.sqrt ((dist_AD ^ 2) + (dist_DC ^ 2))
noncomputable def area : ℝ := real.pi * (radius ^ 2)

-- State the proof problem
theorem fountain_area : 
  dist_AB = 20 → 
  D = (A + B) / 2 → 
  dist_DC = 12 → 
  area = 244 * real.pi := 
by
  intros h1 h2 h3
  sorry

end fountain_area_l438_438760


namespace find_x_tan_sin_cos_l438_438850

theorem find_x_tan_sin_cos (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 360) :
  tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) → x = 160 :=
by
  sorry

end find_x_tan_sin_cos_l438_438850


namespace find_J_l438_438934

variables (J S B : ℕ)

-- Conditions
def condition1 : Prop := J - 20 = 2 * S
def condition2 : Prop := B = J / 2
def condition3 : Prop := J + S + B = 330
def condition4 : Prop := (J - 20) + S + B = 318

-- Theorem to prove
theorem find_J (h1 : condition1 J S) (h2 : condition2 J B) (h3 : condition3 J S B) (h4 : condition4 J S B) :
  J = 170 :=
sorry

end find_J_l438_438934


namespace solve_for_n_l438_438155

theorem solve_for_n :
  ∃ n : ℤ, -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180) ∧ n = 30 :=
by
  sorry

end solve_for_n_l438_438155


namespace sequence_an_general_formula_l438_438311

theorem sequence_an_general_formula :
  ∃ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) - a n = n * real.pi) ∧
  a 1 = 0 ∧ 
  (∀ n : ℕ, f (n) x = | sin (1 / n) (x - a n) | → ∀ b ∈ Ico 0 1, ∃ x1 x2 ∈ Icc (a n) (a (n + 1)), x1 ≠ x2 ∧ f n x1 = b ∧ f n x2 = b) ∧
  ∀ n : ℕ, a n = (n * (n - 1) / 2) * real.pi :=
begin
  sorry
end

end sequence_an_general_formula_l438_438311


namespace angle_pts_l438_438251

theorem angle_pts (P Q R S T : Type)
  [is_triangle P Q R]
  (angle_P : ∠P = 60)
  (angle_R : ∠R = 80)
  (on_side_S : S ∈ line_segment P Q)
  (on_side_T : T ∈ line_segment Q R)
  (eq_sides : dist P S = dist S T) :
  ∠PTS = 70 := by
  sorry

end angle_pts_l438_438251


namespace sum_of_elements_of_set_l438_438060

theorem sum_of_elements_of_set (S : Set ℝ) (hS : S.card = 3) (h_sum : (∑ x in S.powerset, (∑ y in x, y)) = 2012) :
  (∑ x in S, x) = 503 := 
by
   have card_powset := S.card_powerset
   have elems_in_subsets : ∀ x ∈ S, S.powerset.filter (λ t, x ∈ t).card = S.powerset.card / 2 := by sorry
   have contrib_each_elem : ∀ x ∈ S, (∑ t in S.powerset.filter (λ s, x ∈ s), ∑ y in t, y) = 4 * x := by sorry
   have subset_sum_contrib : (∑ x in S.powerset, ∑ y in x, y) = 4 * (∑ x in S, x) := by sorry
   rw [subset_sum_contrib, h_sum]
   linarith

end sum_of_elements_of_set_l438_438060


namespace Suraj_average_after_9th_inning_l438_438955

theorem Suraj_average_after_9th_inning (A : ℝ) (scores : Fin 8 → ℝ) 
  (h1 : ∀ i, 25 ≤ scores i) 
  (h2 : ∀ i, scores i ≤ 80) 
  (h3 : (∑ i, if scores i ≥ 50 then 1 else 0) = 3) 
  (h4 : (∑ i, scores i) = 8 * A) 
  (h5 : A + 6 = (8 * A + 90) / 9) : 
  (8 * A + 90) / 9 = 42 :=
by
  sorry

end Suraj_average_after_9th_inning_l438_438955


namespace CD_length_l438_438501

-- Definitions based on conditions
variables {A B C D L : Type} -- points defining the parallelogram and point L
variables [EuclideanGeometry A B C D L] -- assuming Euclidean geometry

-- Conditions from the problem
variable (angle_D : ∠ D = 100)
variable (BC_len : dist B C = 12)
variable (L_on_AD : L ∈ (segment A D))
variable (angle_ABL : ∠ ABL = 50)
variable (LD_len : dist L D = 4)

-- Theorem to prove
theorem CD_length : dist C D = 8 :=
by
  sorry

end CD_length_l438_438501


namespace not_possible_arrange_cards_l438_438023

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438023


namespace cricket_innings_l438_438351

theorem cricket_innings (n : ℕ) (h1 : (36 * n) / n = 36) (h2 : (36 * n + 80) / (n + 1) = 40) : n = 10 := by
  -- The proof goes here
  sorry

end cricket_innings_l438_438351


namespace proj_a_onto_e_is_3_l438_438919

-- Define vectors a and b
def a : EuclideanSpace ℝ (Fin 2) := ![0, 2 * Real.sqrt 3]
def b : EuclideanSpace ℝ (Fin 2) := ![1, Real.sqrt 3]

-- Define the unit vector e in the direction of b
def e : EuclideanSpace ℝ (Fin 2) := 
  let norm_b := Real.sqrt ((1 : ℝ) ^ 2 + (Real.sqrt 3) ^ 2)
  ![1 / norm_b, Real.sqrt 3 / norm_b]

-- Define the projection of a onto e
def proj_a_onto_e : EuclideanSpace ℝ (Fin 2) := 
  let dot_product := (a.1 * e.1 + a.2 * e.2)
  dot_product • e

-- Prove that the projection of a onto e is 3
theorem proj_a_onto_e_is_3 : proj_a_onto_e = 3 := by
  sorry

end proj_a_onto_e_is_3_l438_438919


namespace cookies_seventh_plate_l438_438677

-- Condition: number of cookies on the plates
def cookies_on_plate : ℕ → ℕ
| 1 := 5
| 2 := 9
| 3 := 14
| 4 := 22
| 5 := 35
| 6 := 55  -- This value is derived from the solution
| 7 := 84  -- This is the answer we want to prove

theorem cookies_seventh_plate : cookies_on_plate 7 = 84 :=
by
  -- We need to prove cookies_on_plate 7 = 84 given the pattern starts
  sorry

end cookies_seventh_plate_l438_438677


namespace full_pound_price_correct_l438_438426

theorem full_pound_price_correct (p : ℝ) (h_discount : 0.11 * p = 2) : p ≈ 18.18 :=
by
  -- Conditions
  have h1: 0.11 * p = 2 := h_discount
  -- Solve for p
  have h2: p = 2 / 0.11 := by sorry
  -- Verify the approximation
  apply h1

end full_pound_price_correct_l438_438426


namespace shelby_gold_stars_today_l438_438636

-- Define the number of gold stars Shelby earned yesterday
def gold_stars_yesterday := 4

-- Define the total number of gold stars Shelby earned
def total_gold_stars := 7

-- Define the number of gold stars Shelby earned today
def gold_stars_today := total_gold_stars - gold_stars_yesterday

-- The theorem to prove
theorem shelby_gold_stars_today : gold_stars_today = 3 :=
by 
  -- The proof will go here.
  sorry

end shelby_gold_stars_today_l438_438636


namespace real_roots_of_quadratic_l438_438190

theorem real_roots_of_quadratic (k : ℝ) : (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) ↔ k ≤ 3 := by
  sorry

end real_roots_of_quadratic_l438_438190


namespace period_of_function_l438_438788

theorem period_of_function : ∀ x, ∀ (y : ℝ → ℝ), (y x = 3 * real.sin x + 4 * real.cos (x - real.pi / 6)) → ∀ t, (y (x + t) = y x) → (t = 2 * real.pi) := 
by
  sorry

end period_of_function_l438_438788


namespace minimum_dihedral_angle_l438_438189

theorem minimum_dihedral_angle 
  (A B C D A₁ B₁ C₁ D₁ P : Point)
  (h₁ : OnEdge P A B)
  (h₂ : IsCube A B C D A₁ B₁ C₁ D₁) :
  ∃ θ, θ = Real.arctan ⟨Real.sqrt 2 / 2, Real.sqrt 2 / 2_ne_0⟩
  ∧ ∀ θ', IsDihedralAngle θ' (PDB₁) (AD D₁ A₁) → θ ≤ θ' := by
  sorry

end minimum_dihedral_angle_l438_438189


namespace no_valid_arrangement_l438_438030

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438030


namespace find_first_term_and_common_difference_l438_438668

-- Define the conditions of the arithmetic progression
def arithmetic_progression (a1 d : ℕ) : Prop :=
  let a := λ n : ℕ, a1 + (n - 1) * d
  (a 3 * a 6 = 406) ∧ (a 9 = 2 * a 4 + 6)

-- Prove that the first term and common difference satisfy the conditions
theorem find_first_term_and_common_difference (a1 d : ℕ) (h : arithmetic_progression a1 d) : a1 = 4 ∧ d = 5 :=
sorry

end find_first_term_and_common_difference_l438_438668


namespace tangerines_count_l438_438376

theorem tangerines_count (apples pears tangerines : ℕ)
  (h1 : apples = 45)
  (h2 : pears = apples - 21)
  (h3 : tangerines = pears + 18) :
  tangerines = 42 :=
by
  sorry

end tangerines_count_l438_438376


namespace inequality_proof_l438_438853

theorem inequality_proof
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 
  64 * a * b * c * d * abs ((a - b) * (b - c) * (c - d) * (d - a)) := 
by
  sorry

end inequality_proof_l438_438853


namespace min_abs_x_minus_abs_y_l438_438559

theorem min_abs_x_minus_abs_y (x y : ℝ) 
(h : log 4 (x + 2 * y) + log 4 (x - 2 * y) = 1) : 
  ∃ (x y : ℝ), abs x - abs y = sqrt 3 := 
sorry

end min_abs_x_minus_abs_y_l438_438559


namespace find_a_l438_438537

theorem find_a (a : ℝ) : 
  (∀ x ∈ set.Icc 0 real.pi, (sin x - 2 * x - a) ≤ -1) → 
  (∃ x ∈ set.Icc 0 real.pi, (sin x - 2 * x - a) = -1) →
  a = 1 :=
begin
  sorry
end

end find_a_l438_438537


namespace fixed_point_l438_438657

noncomputable def function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (x : ℝ) : ℝ :=
  a ^ (x - 1) + 1

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  function a h_pos h_ne_one 1 = 2 :=
by
  sorry

end fixed_point_l438_438657


namespace card_arrangement_impossible_l438_438035

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438035


namespace second_to_last_digit_nine_l438_438754

-- Define what it means for a number to be 'good'
def is_good (n : ℕ) : Prop :=
  let sum_of_digits := λ n : ℕ, n.digits 10 |>.sum in
  (n % sum_of_digits n = 0) ∧ 
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧ 
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧ 
  ((n + 3) % sum_of_digits (n + 3) = 0)

-- Given that n is a natural number ending in 8 and is 'good',
-- we will prove that the second-to-last digit of n must be 9.
theorem second_to_last_digit_nine (n : ℕ) :
  is_good n →
  (n % 10 = 8) →
  ((n / 10) % 10 = 9) :=
begin
  intros h_good h_ends_in_8,
  sorry -- Proof goes here
end

end second_to_last_digit_nine_l438_438754


namespace hyperbola_eccentricity_l438_438599

theorem hyperbola_eccentricity 
  (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : b = 2 * a)
  (hOA : ∀ x y : ℝ, x = 3 ∧ y = 5)
  (hAB : ∀ x y : ℝ, x = sqrt (5^2 - 3^2)) :
  let e : ℝ := sqrt (1 + (b^2 / a^2))
  in e = sqrt 5 := 
by 
  sorry

end hyperbola_eccentricity_l438_438599


namespace radius_of_sphere_is_approximately_correct_l438_438741

noncomputable def radius_of_sphere_in_cylinder_cone : ℝ :=
  let radius_cylinder := 12
  let height_cylinder := 30
  let radius_sphere := 21 - 0.5 * Real.sqrt (30^2 + 12^2)
  radius_sphere

theorem radius_of_sphere_is_approximately_correct : abs (radius_of_sphere_in_cylinder_cone - 4.84) < 0.01 :=
by
  sorry

end radius_of_sphere_is_approximately_correct_l438_438741


namespace solution_condition_l438_438051

noncomputable def has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |x| + |y| = |b|

theorem solution_condition (a b : ℝ) :
  has_solution a b ↔ |a| ≤ |b| ∧ |b| ≤ real.sqrt 2 * |a| :=
begin
  sorry
end

end solution_condition_l438_438051


namespace no_valid_arrangement_l438_438029

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438029


namespace intersection_point_l438_438714

variable {t : ℝ}
variable {x y z : ℝ}

def line_eq : Prop :=
  (x = 1 + 8 * t) ∧ (y = 8 - 5 * t) ∧ (z = -5 + 12 * t)

def plane_eq : Prop :=
  x - 2 * y - 3 * z + 18 = 0

theorem intersection_point (t : ℝ) (x y z : ℝ) (h1 : line_eq) (h2 : plane_eq) : 
  (x, y, z) = (9, 3, 7) :=
by
  sorry

end intersection_point_l438_438714


namespace concyclic_circumcenters_of_midtriangles_l438_438498

open EuclideanGeometry

noncomputable def midpoint (A B : Point) := Midpoint A B
noncomputable def circumcenter (A B C : Point) := Circumcenter A B C

theorem concyclic_circumcenters_of_midtriangles {A B C D P E F G H O₁ O₂ O₃ O₄ : Point}
  (h_convex: ConvexQuadrilateral A B C D)
  (h_diag_inter: ∃ P, Line A C ∩ Line B D = {P})
  (h_midpoints: E = midpoint A B ∧ F = midpoint B C ∧ G = midpoint C D ∧ H = midpoint D A)
  (h_circumcenters: O₁ = circumcenter P H E ∧ O₂ = circumcenter P E F ∧ O₃ = circumcenter P F G ∧ O₄ = circumcenter P G H) :
  (Concyclic O₁ O₂ O₃ O₄ ↔ Concyclic A B C D) :=
sorry

end concyclic_circumcenters_of_midtriangles_l438_438498


namespace line_equation_form_l438_438416

theorem line_equation_form :
  ∀ (x y : ℝ), (⟨1, 3⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨-2, 8⟩) = 0 ↔ y = (-1 / 3) * x + 22 / 3 :=
by
  sorry

end line_equation_form_l438_438416


namespace math_proof_problem_l438_438490

noncomputable def problem_statement : Prop :=
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
  ∃ (f : Finset ℕ), f.card = 4 ∧ ∀ t ∈ f, ∃ (x y : ℕ), t = {x, y, 7} ∧ x ≠ 7 ∧ y ≠ 7 ∧ x + y = 14

theorem math_proof_problem : problem_statement :=
sorry

end math_proof_problem_l438_438490


namespace sin_360_eq_0_l438_438449

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l438_438449


namespace exists_individual_with_few_friend_pairs_among_enemies_l438_438952
-- Given conditions
variables (n q : ℕ) -- Number of people and pairs of friends
-- Assume that every two people are either friendly or hostile
-- Assume that among any three people, at least one pair is hostile

theorem exists_individual_with_few_friend_pairs_among_enemies (h1 : n > 0) (h2 : q ≥ 0) (h3 : ∀ (T : finset ℕ), T.card = 3 → ∃ (a b : ℕ), a ≠ b ∧ a ∈ T ∧ b ∈ T ∧ ¬(friendly a b)) :
  ∃ v : ℕ, number_of_friendly_pairs_among_enemies v ≤ q * (1 - (4 * q) / n^2) :=
sorry

end exists_individual_with_few_friend_pairs_among_enemies_l438_438952


namespace sin_360_eq_0_l438_438450

theorem sin_360_eq_0 : Real.sin (360 * Real.pi / 180) = 0 := by
  sorry

end sin_360_eq_0_l438_438450


namespace becky_winning_strategy_condition_l438_438995

-- Define the game conditions
def has_winning_strategy_for_Becky (n : ℕ) : Prop :=
  -- Assuming the game setup and rules as described
  -- Placeholder function to determine if Becky has a winning strategy, just return true/false based on the problem solution
  sorry

-- Main theorem statement
theorem becky_winning_strategy_condition (n : ℕ) (h : n ≥ 5) :
  has_winning_strategy_for_Becky(n) ↔ n % 4 = 2 :=
sorry

end becky_winning_strategy_condition_l438_438995


namespace solve_for_x_l438_438639

theorem solve_for_x (x : ℝ) (h : 3 * x - 7 = 2 * x + 5) : x = 12 :=
sorry

end solve_for_x_l438_438639


namespace angle_SRQ_l438_438268

-- Define the variables and conditions
variables {l k : Line} {S R Q : Point}
variable (angle : ℝ)

-- Conditions
def parallel_lines : Prop := parallel l k
def angle_RSQ : Prop := angleRSQ = 30
def angle_RQS : Prop := angleRQS = 90


-- Statement to prove
theorem angle_SRQ : parallel_lines ∧ angle_RSQ ∧ angle_RQS → angleSRQ = 60 := by 
  sorry

end angle_SRQ_l438_438268


namespace repeated_digit_percentage_l438_438239

theorem repeated_digit_percentage : 
  let total := 90000 in 
  let non_repeated := 9 * 9 * 8 * 7 * 6 in 
  let repeated := total - non_repeated in 
  let x := (repeated.toFloat / total.toFloat) * 100 in 
  x = 69.8 :=
by
  let total := 90000
  let non_repeated := 9 * 9 * 8 * 7 * 6
  let repeated := total - non_repeated
  let x := (repeated.toFloat / total.toFloat) * 100
  have : x = 69.8
  sorry

end repeated_digit_percentage_l438_438239


namespace minimum_value_of_f_root_of_f_l438_438616

def f (x m : ℝ) : ℝ := exp (x - m) - x

theorem minimum_value_of_f (m : ℝ) : f m m = 1 - m :=
by
  sorry

theorem root_of_f (m : ℝ) (hm : 1 < m) : ∃ x ∈ Ioo m (2 * m), f x m = 0 :=
by
  let a := m
  let b := 2 * m
  have h1 : f a m = 1 - m := by
    sorry
  have h2 : f b m = exp m - 2 * m := by
    sorry
  have h3 : f a m * f b m < 0 := by
    sorry
  exact intermediate_value_theorem (continuous.exp.sub continuous_id) a b h1 h2 h3

end minimum_value_of_f_root_of_f_l438_438616


namespace length_BD_l438_438329

/-- Points A, C, F are collinear, lengths AB, DE, and FC are equal,
    angles ABC, DEC, and FCE are equal,
    angles BAC, EDC, and CFE are equal,
    lengths AF = 21, CE = 13.
    Prove that the length of the segment BD = 5. -/
theorem length_BD (A B C D E F : Point)
  (h1 : collinear A C F)
  (h2 : AB = DE ∧ DE = FC)
  (h3 : ∠ABC = ∠DEC ∧ ∠DEC = ∠FCE)
  (h4 : ∠BAC = ∠EDC ∧ ∠EDC = ∠CFE)
  (h5 : AF = 21)
  (h6 : CE = 13) :
  BD = 5 :=
sorry

end length_BD_l438_438329


namespace relationship_among_abc_l438_438523

noncomputable def a := 5^((Real.log 3.4) / (Real.log 2))
noncomputable def b := 5^((Real.log 3.6) / (2 * Real.log 2))
noncomputable def c := 5^(-(Real.log 0.3) / (Real.log 7))

theorem relationship_among_abc : a > b ∧ b > c :=
by
  sorry

end relationship_among_abc_l438_438523


namespace find_lambda_l438_438606

variables {E : Type*} [AddCommGroup E] [Module ℝ E]
variables (e₁ e₂ : E) (a b : ℝ) (λ : ℝ)

-- Define the non-collinearity of e₁ and e₂
def non_collinear (e₁ e₂ : E) := ¬ ∃ t : ℝ, e₂ = t • e₁

-- Define the vectors a and b
def a : E := 2 • e₁ - 3 • e₂
def b : E := 3 • e₁ + λ • e₂

-- Prove that λ = -9/2 given the conditions
theorem find_lambda (h_non_collinear : non_collinear e₁ e₂)
  (H : ∃ m : ℝ, b = m • a) : λ = -(9/2) :=
sorry

end find_lambda_l438_438606


namespace smallest_constant_c_l438_438453

def satisfies_conditions (f : ℝ → ℝ) :=
  ∀ ⦃x : ℝ⦄, (0 ≤ x ∧ x ≤ 1) → (f x ≥ 0 ∧ (x = 1 → f 1 = 1) ∧
  (∀ y, 0 ≤ y → y ≤ 1 → x + y ≤ 1 → f x + f y ≤ f (x + y)))

theorem smallest_constant_c :
  ∀ {f : ℝ → ℝ},
  satisfies_conditions f →
  ∃ c : ℝ, (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x) ∧
  (∀ c', c' < 2 → ∃ x, 0 ≤ x → x ≤ 1 ∧ f x > c' * x) :=
by sorry

end smallest_constant_c_l438_438453


namespace Mark_owes_total_l438_438618

noncomputable def base_fine : ℕ := 50

def additional_fine (speed_over_limit : ℕ) : ℕ :=
  let first_10 := min speed_over_limit 10 * 2
  let next_5 := min (speed_over_limit - 10) 5 * 3
  let next_10 := min (speed_over_limit - 15) 10 * 5
  let remaining := max (speed_over_limit - 25) 0 * 6
  first_10 + next_5 + next_10 + remaining

noncomputable def total_fine (base : ℕ) (additional : ℕ) (school_zone : Bool) : ℕ :=
  let fine := base + additional
  if school_zone then fine * 2 else fine

def court_costs : ℕ := 350

noncomputable def processing_fee (fine : ℕ) : ℕ := fine / 10

def lawyer_fees (hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours

theorem Mark_owes_total :
  let speed_over_limit := 45
  let base := base_fine
  let additional := additional_fine speed_over_limit
  let school_zone := true
  let fine := total_fine base additional school_zone
  let total_fine_with_costs := fine + court_costs
  let processing := processing_fee total_fine_with_costs
  let lawyer := lawyer_fees 100 4
  let total := total_fine_with_costs + processing + lawyer
  total = 1346 := sorry

end Mark_owes_total_l438_438618


namespace area_triangle_QPO_l438_438960

variables (A B C D P Q N M O: Type)
variables [plane_geometry A B C D P Q N M O]

/-- Given parallelogram ABCD, DP trisects BC at N. -/
variables [parallelogram A B C D]
variables [trisection B C N] (h1 : BN = 1/3 * BC) (h2 : NC = 1/3 * BC)

/-- Line CQ trisects AD at M and extends to P and Q. -/
variables [trisection A D M] (h3 : AM = 1/3 * AD) (h4 : MD = 1/3 * AD)

/-- Intersection of lines DP and CQ is O. -/
variables (h5 : DP ∩ CQ = O)

/-- The area of parallelogram ABCD is k. -/
variable (k : ℝ) [parallelogram_area A B C D k]

theorem area_triangle_QPO : area_triangle Q P O = (10 * k) / 9 :=
sorry

end area_triangle_QPO_l438_438960


namespace cost_of_50_lavenders_l438_438573

noncomputable def cost_of_bouquet (lavenders : ℕ) : ℚ :=
  (25 / 15) * lavenders

theorem cost_of_50_lavenders :
  cost_of_bouquet 50 = 250 / 3 :=
sorry

end cost_of_50_lavenders_l438_438573


namespace diamond_calculation_l438_438802

def diamond (a b : ℚ) : ℚ := (a - b) / (1 + a * b)

theorem diamond_calculation : diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 :=
by
  sorry

end diamond_calculation_l438_438802


namespace equal_shaded_areas_angle_l438_438968

theorem equal_shaded_areas_angle {O P Q : Type} (h : ∀ (r1 r2 : ℝ), (r1 = 1 ∧ r2 = 3) → ∀ (A_small A_large : ℝ), (A_small = π * r1^2 ∧ A_large = π * r2^2) → (A_small = π) → (A_large = 9 * π) → ∀ (x : ℝ), (x = 40) → ∀θ, (θ = 360 * x / 9) → θ = 40) : true :=
begin
  sorry
end

end equal_shaded_areas_angle_l438_438968


namespace space_station_cost_share_l438_438570

theorem space_station_cost_share (total_cost_billion : ℕ) (total_people_million : ℕ) 
  (people_sharing_cost : ℕ) (total_cost_million : ℕ) (cost_per_person : ℕ) :
  total_cost_billion = 50 →
  total_people_million = 400 →
  people_sharing_cost = 200 →
  total_cost_million = 50000 →
  cost_per_person = total_cost_million / people_sharing_cost →
  cost_per_person = 250 :=
begin
  sorry
end

end space_station_cost_share_l438_438570


namespace find_f_neg2014_l438_438906

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem find_f_neg2014 (a b : ℝ) (h : f 2014 a b = 3) : f (-2014) a b = -7 :=
by sorry

end find_f_neg2014_l438_438906


namespace total_workers_construction_l438_438090

def number_of_monkeys : Nat := 239
def number_of_termites : Nat := 622
def total_workers (m : Nat) (t : Nat) : Nat := m + t

theorem total_workers_construction : total_workers number_of_monkeys number_of_termites = 861 := by
  sorry

end total_workers_construction_l438_438090


namespace johns_coin_collection_value_l438_438285

theorem johns_coin_collection_value :
  ∀ (n : ℕ) (value : ℕ), n = 24 → value = 20 → 
  ((n/3) * (value/8)) = 60 :=
by
  intro n value n_eq value_eq
  sorry

end johns_coin_collection_value_l438_438285


namespace tangent_line_ln_curve_l438_438217

theorem tangent_line_ln_curve (a : ℝ) :
  (∃ x₀ y₀ : ℝ, y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a)
  ∧ (Real.derivative (λ x => Real.log (x + a))) x₀ = 1) → a = 2 := by
  sorry

end tangent_line_ln_curve_l438_438217


namespace option_d_is_quadratic_equation_l438_438388

theorem option_d_is_quadratic_equation (x y : ℝ) : 
  (x^2 + x - 4 = 0) ↔ (x^2 + x = 4) := 
by
  sorry

end option_d_is_quadratic_equation_l438_438388


namespace find_ede_l438_438151

-- Define our conditions regarding the three-digit numbers and relatively prime condition
def distinct_digits (a b : ℕ) := ∀ d : ℕ, d ∈ digits 10 a → d ∉ digits 10 b

-- Lean statement begins
theorem find_ede :
  ∃ E D V I, 
    let EDE := E * 101 + D * 10 + E in
    let VIV := V * 101 + I * 10 + V in
    EDE ≠ VIV ∧ gcd EDE VIV = 1 ∧ 
    let q := EDE / VIV in 
    let r := EDE % VIV in 
    0 < r ∧ r < VIV ∧ 
    (∃ G Y O Z : ℕ, VIV * (G * 1000 + Y * 100 + O * 10 + Z) = 9999 * r) ∧
    E * 101 + D * 10 + E = 242 :=
  sorry

end find_ede_l438_438151


namespace range_a_I_range_a_II_l438_438861

variable (a: ℝ)

-- Define the proposition p and q
def p := (Real.sqrt (a^2 + 13) > Real.sqrt 17)
def q := ∀ x, (0 < x ∧ x < 3) → (x^2 - 2 * a * x - 2 = 0)

-- Prove question (I): If proposition p is true, find the range of the real number $a$
theorem range_a_I (h_p : p a) : a < -2 ∨ a > 2 :=
by sorry

-- Prove question (II): If both the proposition "¬q" and "p ∧ q" are false, find the range of the real number $a$
theorem range_a_II (h_neg_q : ¬ q a) (h_p_and_q : ¬ (p a ∧ q a)) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_a_I_range_a_II_l438_438861


namespace smallest_divisible_sum_of_digits_l438_438293

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

theorem smallest_divisible_sum_of_digits :
  let M := Nat.lcm ([1, 2, 3, 4, 5, 6, 7] : List Nat)
  sum_of_digits M = 6 :=
by
  sorry

end smallest_divisible_sum_of_digits_l438_438293


namespace lateral_surface_area_correct_l438_438890

-- Definitions based on the conditions
def base_radius : ℝ := 3
def generatrix : ℝ := 5

-- Definition of the formula for the lateral surface area of a cylinder
def lateral_surface_area (r g : ℝ) : ℝ := 2 * Real.pi * r * g

-- The proof statement
theorem lateral_surface_area_correct :
  lateral_surface_area base_radius generatrix = 30 * Real.pi := by
  sorry

end lateral_surface_area_correct_l438_438890


namespace complex_quadrant_l438_438877

def Z1 : ℂ := 2 + I
def Z2 : ℂ := 1 + I

theorem complex_quadrant :
  let quotient := Z1 / Z2 in
  (quotient.re > 0) ∧ (quotient.im < 0) :=
by {
  let quotient := Z1 / Z2,
  sorry
}

end complex_quadrant_l438_438877


namespace angle_A_and_min_a_angle_A_and_min_a_l438_438951

-- Define the problem conditions and the proof goals

theorem angle_A_and_min_a (a b c: ℝ) (C: ℝ) (h1: b = a * (Real.cos C) + (1/2) * c)
                           (h2: a > 0) (h3: b > 0) (h4: c > 0) (h5: ↔ -a * b + a * c  = 3):
  (A : ℝ) (h6:  A = Real.arccos(1 / 2)):
  A = π / 3 ∧ Mathlib


theorem angle_A_and_min_a (a b c: ℝ) (C: ℝ) (h1: b = a * (Real.cos C) + (1/2) * c)
                           (h2: a > 0) (h3: b > 0) (h4: c > 0) (h5:  (b=c) ) :
  (A : ℝ) (h6: a = b *  c = Mathlib.sqrt(6))  :
    a  = Mathlib.sqrt(6) :=
by
  sorry

end angle_A_and_min_a_angle_A_and_min_a_l438_438951


namespace non_prime_count_l438_438925

def is_prime (n : ℕ) : Prop := ¬∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def expr1 := 1^2 + 2^2
def expr2 := 2^2 + 3^2
def expr3 := 3^2 + 4^2
def expr4 := 4^2 + 5^2
def expr5 := 5^2 + 6^2

def count_non_prime (l : List ℕ) : ℕ :=
  l.countp (λ n => ¬ is_prime n)

theorem non_prime_count : count_non_prime [expr1, expr2, expr3, expr4, expr5] = 1 :=
by
  -- the proof would go here
  sorry

end non_prime_count_l438_438925


namespace parallelogram_to_rectangle_l438_438543

variables {A B C D: Type} [inhabited A]  -- Assume a non-empty type for points

-- ABCD is a parallelogram
def is_parallelogram (A B C D : Type) [inhabited A] : Prop := sorry

-- radii of inscribed circles of triangles ABC and ABD are equal
def inradius_equal (A B C D : Type) [inhabited A] : Prop := sorry

-- radii of circumscribed circles of triangles ABC and ABD are equal
def circumradius_equal (A B C D : Type) [inhabited A] : Prop := sorry

-- ABCD is a rectangle
def is_rectangle (A B C D : Type) [inhabited A] : Prop := sorry

theorem parallelogram_to_rectangle (A B C D : Type) [inhabited A]:
  (is_parallelogram A B C D) ∧ (inradius_equal A B C D ∨ circumradius_equal A B C D)  → is_rectangle A B C D :=
by
  sorry

end parallelogram_to_rectangle_l438_438543


namespace not_possible_arrange_cards_l438_438022

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438022


namespace find_m_for_symmetric_function_l438_438530

theorem find_m_for_symmetric_function (m : ℝ) :
  (m^2 - m - 1 ≠ 1 → f(x) ≠ (m^2 - m - 1) * x^(1-m)) → m = -1 := by
  sorry

end find_m_for_symmetric_function_l438_438530


namespace find_coordinates_of_T_l438_438145

-- Define points O and W
def O := (0, 0)
def W := (4, 4)

-- Define that OSVW is a square with given points O and W
def is_square (O W : ℝ × ℝ) : Prop :=
  O = (0, 0) ∧ W = (4, 4)

-- Define area of a square
def area_square (s : ℝ) : ℝ := s * s

-- Define area of a triangle
def area_triangle (b h : ℝ) : ℝ := 0.5 * b * h

-- The proposition we want to prove
theorem find_coordinates_of_T (S V T : ℝ × ℝ) (b h : ℝ)
  (h1 : is_square O W)
  (h2 : S = (4, 0))
  (h3 : V = (0, 4))
  (h4 : T = (-4, 0))
  (h5 : b = 4)
  (h6 : h = 8) :
  area_triangle b h = area_square 4 :=
by
  sorry

end find_coordinates_of_T_l438_438145


namespace part1_part2_part3_part4_l438_438167

theorem part1 : sin 10 * sin 30 * sin 50 * sin 70 = 1/16 := 
sorry

theorem part2 : sin^2 20 + cos^2 80 + (sqrt 3) * sin 20 * cos 80 = 1/4 := 
sorry

theorem part3 (A : ℝ) : cos^2 A + cos^2 (60 - A) + cos^2 (60 + A) = 3/2 := 
sorry

theorem part4 : 
  cos (π / 15) * cos (2 * π / 15) * 
  cos (3 * π / 15) * cos (4 * π / 15) * 
  cos (5 * π / 15) * cos (6 * π / 15) * 
  cos (7 * π / 15) = 1 / 128 := 
sorry

end part1_part2_part3_part4_l438_438167


namespace find_incorrect_statements_l438_438223

-- Given conditions in the problem
def condition_A := ∀ (b h : ℝ), (3 * b) * h = 3 * (b * h)
def condition_B := ∀ (b h : ℝ), (1 / 2) * (3 * b) * h = 3 * ((1 / 2) * b * h)
def condition_C := ∀ (r : ℝ), π * (3 * r) ^ 2 = 3 * (π * r ^ 2)
def condition_D := ∀ (a b : ℝ), 3 * a / (3 * b) = a / b
def condition_E := ∀ (x : ℝ), x < 0 → 3 * x > x

-- The statement to prove in Lean 4
theorem find_incorrect_statements : ¬ condition_C ∧ ¬ condition_E := by
  sorry

end find_incorrect_statements_l438_438223


namespace samia_walked_distance_l438_438634

theorem samia_walked_distance :
  ∀ (total_distance cycling_speed walking_speed total_time : ℝ), 
  total_distance = 18 → 
  cycling_speed = 20 → 
  walking_speed = 4 → 
  total_time = 1 + 10 / 60 → 
  2 / 3 * total_distance / cycling_speed + 1 / 3 * total_distance / walking_speed = total_time → 
  1 / 3 * total_distance = 6 := 
by
  intros total_distance cycling_speed walking_speed total_time h1 h2 h3 h4 h5
  sorry

end samia_walked_distance_l438_438634


namespace polar_to_rect_eq_point_D_min_dist_l438_438970

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rect_eq {ρ θ : ℝ} (hρ : ρ = 2 * Real.sin θ) : 
  ∃ x y : ℝ, (x, y) = polar_to_rectangular ρ θ ∧ x^2 + y^2 = 2 * y :=
by
  apply Exists.intro (ρ * Real.cos θ)
  apply Exists.intro (ρ * Real.sin θ)
  have h_rect : (ρ * Real.cos θ, ρ * Real.sin θ) = polar_to_rectangular ρ θ := rfl
  split
  . exact h_rect
  . rw [hρ, real_mul_commρ, Real.sin_sq, by sorry -- proof omitted for brevity]

theorem point_D_min_dist : 
  ∃ α : ℝ, let D := (Real.cos α, 1 + Real.sin α) in 
  α = π / 6 ∧ D = ( √3 / 2, 3 / 2) :=
by sorry -- proof omitted for brevity

end polar_to_rect_eq_point_D_min_dist_l438_438970


namespace find_geometric_sequence_values_l438_438175

structure GeometricSequence (a b c d : ℝ) : Prop where
  ratio1 : b / a = c / b
  ratio2 : c / b = d / c

theorem find_geometric_sequence_values (x u v y : ℝ)
    (h1 : x + y = 20)
    (h2 : u + v = 34)
    (h3 : x^2 + u^2 + v^2 + y^2 = 1300) :
    (GeometricSequence x u v y ∧ ((x = 16 ∧ u = 4 ∧ v = 32 ∧ y = 2) ∨ (x = 4 ∧ u = 16 ∧ v = 2 ∧ y = 32))) :=
by
  sorry

end find_geometric_sequence_values_l438_438175


namespace increasing_interval_l438_438663

def f (x : ℝ) := Real.log (Real.cos x)

theorem increasing_interval (k : ℤ) :
  Ioo (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi) ⊆ { x | 0 < Real.cos x ∧ ∀ y1 y2, (2 * k * Real.pi - Real.pi / 2) < y1 → y1 < y2 → y2 < 2 * k * Real.pi → f y1 < f y2 } :=
sorry

end increasing_interval_l438_438663


namespace area_of_triangle_ABC_l438_438691

-- Define the given conditions
structure Triangle :=
(A B C L : Point)
(AB BL BC : ℝ)
(hBL : BL = 9)
(hAB : AB = 15)
(hBC : BC = 17)

-- Define the triangle with its properties
constant Point : Type
constant area_of_triangle : Triangle → ℝ

-- The problem statement
theorem area_of_triangle_ABC (T : Triangle) : area_of_triangle T = 102 :=
by
  sorry

end area_of_triangle_ABC_l438_438691


namespace find_r_l438_438998

theorem find_r (a b m p r : ℝ) (h_roots1 : a * b = 6) 
  (h_eq1 : ∀ x, x^2 - m*x + 6 = 0) 
  (h_eq2 : ∀ x, x^2 - p*x + r = 0) :
  r = 32 / 3 :=
by
  sorry

end find_r_l438_438998


namespace joshua_finishes_after_malcolm_l438_438315

-- Definitions based on conditions.
def malcolm_speed : ℕ := 6 -- Malcolm's speed in minutes per mile
def joshua_speed : ℕ := 8 -- Joshua's speed in minutes per mile
def race_distance : ℕ := 10 -- Race distance in miles

-- Theorem: How many minutes after Malcolm crosses the finish line will Joshua cross the finish line?
theorem joshua_finishes_after_malcolm :
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 20 :=
by
  -- sorry is a placeholder for the proof
  sorry

end joshua_finishes_after_malcolm_l438_438315


namespace part1_part2_l438_438586

open Nat

def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (a n) + 1

def a_n_plus_1_geo (n : ℕ) : Prop := a (n + 1) + 1 = 2 * (a n + 1)

theorem part1 : ∀ (n : ℕ), a_n_plus_1_geo n := 
sorry

def b (n : ℕ) : ℕ → ℚ := 
  λ n, (a (n + 1) + 1) / ((a (n + 1)) * (a n + 1))

def S (n : ℕ) : ℚ :=
  ∑ k in range (n + 1), b k

theorem part2 (n : ℕ) : S n = (2^(n + 1) - 2) / (2^(n + 1) - 1) := 
sorry

end part1_part2_l438_438586


namespace sum_of_first_10_common_elements_l438_438848

-- Define arithmetic sequence
def a_n (n : ℕ) : ℕ := 4 + 3 * n

-- Define geometric sequence
def b_k (k : ℕ) : ℕ := 10 * 2^k

-- Define common elements sequence
def common_element (m : ℕ) : ℕ := 10 * 4^m

-- State the theorem
theorem sum_of_first_10_common_elements : 
  (Finset.range 10).sum (λ i, common_element i) = 3495250 := sorry

end sum_of_first_10_common_elements_l438_438848


namespace Juliska_correct_l438_438981

-- Definitions according to the conditions in a)
def has_three_rum_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "rum" ∈ selected_triplet

def has_three_coffee_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "coffee" ∈ selected_triplet

-- Proof problem statement
theorem Juliska_correct 
  (candies : List String) 
  (h_rum : has_three_rum_candy candies)
  (h_coffee : has_three_coffee_candy candies) : 
  (∀ (selected_triplet : List String), selected_triplet.length = 3 → "walnut" ∈ selected_triplet) :=
sorry

end Juliska_correct_l438_438981


namespace pizzas_served_today_l438_438761

theorem pizzas_served_today (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (h1 : lunch_pizzas = 9) (h2 : dinner_pizzas = 6) : lunch_pizzas + dinner_pizzas = 15 :=
by sorry

end pizzas_served_today_l438_438761


namespace mathematician_age_l438_438419

theorem mathematician_age (n : ℕ) (hn1 : ∃ k : ℕ, k^2 = n) (hn2 : ∃ m : ℕ, number_of_divisors n = m ∧ ∃ j : ℕ, j^2 = m) : 
  n = 36 ∨ n = 100 := 
  sorry

end mathematician_age_l438_438419


namespace general_term_formula_find_positive_integer_n_l438_438508

open Real

noncomputable def a_n (n : ℕ) : ℕ := 3 * n - 1
noncomputable def b_n (n : ℕ) : ℝ := 3 / (a_n n)

theorem general_term_formula :
  ∀ n : ℕ, a_n n = 3 * n - 1 := by
  intro n
  rw [a_n]
  rfl

theorem find_positive_integer_n :
  ∃ (n : ℕ), (∑ k in Finset.range n, b_n k * b_n (k + 1)) = 45 / 32 := by
  use 10
  sorry

end general_term_formula_find_positive_integer_n_l438_438508


namespace min_area_triangle_l438_438247

theorem min_area_triangle (m n : ℝ) (h1 : (1 : ℝ) / m + (2 : ℝ) / n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ A B C : ℝ, 
  ((0 < A) ∧ (0 < B) ∧ ((1 : ℝ) / A + (2 : ℝ) / B = 1) ∧ (A * B = C) ∧ (2 / C = mn)) ∧ (C = 4) :=
by
  sorry

end min_area_triangle_l438_438247


namespace number_of_girls_l438_438669

theorem number_of_girls 
  (B G : ℕ) 
  (h1 : B + G = 480) 
  (h2 : 5 * B = 3 * G) :
  G = 300 := 
sorry

end number_of_girls_l438_438669


namespace solution_set_f_gt_2x_add_4_l438_438655

noncomputable def f : ℝ → ℝ := sorry -- Placeholder definition for f

axiom f_domain : ∀ x : ℝ, true -- Placeholder to indicate that f's domain is ℝ

axiom f_at_neg1 : f (-1) = 2

axiom f'_gt_2 : ∀ x : ℝ, (deriv f x) > 2

theorem solution_set_f_gt_2x_add_4 : {x : ℝ | f x > 2 * x + 4} = Ioi (-1) :=
begin
  sorry
end

end solution_set_f_gt_2x_add_4_l438_438655


namespace largest_square_perimeter_l438_438638

-- Define the conditions
def is_inscribed (small large : ℝ) : Prop :=
  ∀ a, (small.sqrt * a.sqrt) = large

-- Define the side length of the initial smallest square
def side_length_init (a : ℝ) : Prop := a^2 = 1

-- Define the area doubling property
def area_doubling (a b : ℝ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → b = a * 2^i

-- Main statement
theorem largest_square_perimeter :
  ∃ l : ℝ, ∀ a b: ℝ, (side_length_init a) → (is_inscribed a b) → (area_doubling a b 4) → l = 16 :=
begin
  sorry
end

end largest_square_perimeter_l438_438638


namespace incorrect_statement_of_regression_line_l438_438868

-- Given conditions
variables {n : ℕ} {x : Fin n → ℝ} {y : Fin n → ℝ}
def regression_line (b a : ℝ) (x_val : ℝ) : ℝ := b * x_val + a

-- Statement B: The regression line passes through the sample mean point
def passes_through_mean (b a : ℝ) (x y : Fin n → ℝ) : Prop :=
  let x_bar := (∑ i, x i) / n
  let y_bar := (∑ i, y i) / n
  in b * x_bar + a = y_bar

-- Statement A: The regression line does not necessarily pass through a given point
def not_necessary_through_point (b a : ℝ) (x y : Fin n → ℝ) : Prop :=
  ¬ ∃ i : Fin n, regression_line b a (x i) = y i

theorem incorrect_statement_of_regression_line (b a : ℝ) (x y : Fin n → ℝ) :
  passes_through_mean b a x y ∧ not_necessary_through_point b a x y → false := 
sorry

end incorrect_statement_of_regression_line_l438_438868


namespace minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l438_438492

-- Define the conditions once to reuse them for both proof statements.
variables {a b : ℝ} (ha: a > 0) (hb: b > 0) (h: (1/a) + (1/b) = Real.sqrt (a * b))

-- Problem (I)
theorem minimum_a3_b3 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  a^3 + b^3 = 4 * Real.sqrt 2 := 
sorry

-- Problem (II)
theorem no_exist_a_b_2a_3b_eq_6 (h : (1/a) + (1/b) = Real.sqrt (a * b)) (ha: a > 0) (hb: b > 0) :
  ¬ ∃ (a b : ℝ), 2 * a + 3 * b = 6 :=
sorry

end minimum_a3_b3_no_exist_a_b_2a_3b_eq_6_l438_438492


namespace arithmetic_sequence_sum_abs_values_l438_438871

theorem arithmetic_sequence_sum_abs_values (n : ℕ) (a : ℕ → ℤ)
  (h₁ : a 1 = 13)
  (h₂ : ∀ k, a (k + 1) = a k + (-4)) :
  T_n = if n ≤ 4 then 15 * n - 2 * n^2 else 2 * n^2 - 15 * n + 56 :=
by sorry

end arithmetic_sequence_sum_abs_values_l438_438871


namespace monotonic_intervals_and_extremes_range_of_m_l438_438908

noncomputable def f (x : ℝ) : ℝ := 5 + Real.logb (1 / 2) (-x^2 + 2*x + 7)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 9^x - 2*m*3^x

theorem monotonic_intervals_and_extremes :
  (∀ x ∈ Icc (0 : ℝ) 3, f x ∈ Icc (2 : ℝ) 3)
  ∧ (∀ x y ∈ Icc (0 : ℝ) 1, x ≤ y → f x ≥ f y)
  ∧ (∀ x y ∈ Icc (1 : ℝ) 3, x ≤ y → f x ≤ f y)
  ∧ (f 1 = 2)
  ∧ (f 3 = 3) := sorry

theorem range_of_m :
  ∃ m : ℝ, -1/2 ≤ m ∧ m ≤ 1 ∧ (∀ x ∈ Icc 0 1, g x m ∈ Icc 2 3) := sorry

end monotonic_intervals_and_extremes_range_of_m_l438_438908


namespace find_ages_l438_438730

theorem find_ages (J sister cousin : ℝ)
  (h1 : J + 9 = 3 * (J - 11))
  (h2 : sister = 2 * J)
  (h3 : cousin = (J + sister) / 2) :
  J = 21 ∧ sister = 42 ∧ cousin = 31.5 :=
by
  sorry

end find_ages_l438_438730


namespace solve_for_x_l438_438556

theorem solve_for_x (x : Real) (h1 : 16 = 2^4) (h2 : 64 = 2^6) (h3 : 16^x = 64) : x = 3 / 2 :=
by
  sorry

end solve_for_x_l438_438556


namespace length_AB_range_of_k_l438_438910

-- Define the equation of the hyperbola C.
def hyperbola (x y : ℝ) := (x^2 / 2) - y^2 = 1

-- Define point M with coordinates (0, 1).
def M : ℝ × ℝ := (0, 1)

-- Define the line l passing through M with slope 1/2.
def line_through_M (x : ℝ) := (1 / 2) * x + 1

-- Proof that the length of segment AB is 2√15.
theorem length_AB {x1 x2 : ℝ} (hx1 : hyperbola x1 (line_through_M x1))
                          (hx2 : hyperbola x2 (line_through_M x2)) :
  let y1 := line_through_M x1,
      y2 := line_through_M x2 in
  (sqrt ((x2 - x1)^2 + (y2 - y1)^2)) = 2 * sqrt 15 :=
sorry

-- General point P on the hyperbola and its symmetric point Q.
variable (P : ℝ × ℝ) (hP : hyperbola P.1 P.2)
def Q := (-P.1, P.2)

-- Define the dot product k between vector MP and MQ.
def dot_product : ℝ :=
  let MP := (P.1, P.2 - 1),
      MQ := (Q P).1, (Q P).2 - 1 in
  (MP.1 * MQ.1 + MP.2 * MQ.2)

-- Proof that the range of k is (-∞, 0].
theorem range_of_k {n : ℝ} (hp : P.2 = n ∧ \frac {(P.1)^2}{2} - n^2 = 1) :
  ∃ k : ℝ, (k = -(n + 1)^2) ∧ (k ∈ Iic 0) :=
sorry

end length_AB_range_of_k_l438_438910


namespace packages_of_gum_l438_438632

-- Define the conditions
variables (P : Nat) -- Number of packages Robin has

-- State the theorem
theorem packages_of_gum (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end packages_of_gum_l438_438632


namespace a_2018_equals_5_l438_438587

noncomputable def a : ℕ → ℤ
| 0     := 1
| 1     := 5
| (n+2) := a (n+1) - a n

theorem a_2018_equals_5 : a 2018 = 5 :=
by
  sorry

end a_2018_equals_5_l438_438587


namespace locus_is_hyperbola_l438_438866

noncomputable theory

def circle (center : ℝ × ℝ) (radius : ℝ) := 
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def locus_center_external_tangent : Prop :=
  let C1_center := (0, 2)
  let C1_radius := 3
  let C2_center := (0, -2)
  let C2_radius := 5 in
  ∃ f : ℝ → ℝ × ℝ, (∀ r, circle C1_center (C1_radius + r) (f r) ∧ circle C2_center (C2_radius + r) (f r)) ∧
    (∀ x y, f x = f y → x = y) -- unique mapping for given radius.

-- Given Circles C1 and C2, prove locus_center_external_tangent forms one branch of a hyperbola
theorem locus_is_hyperbola : locus_center_external_tangent ∧ 
  (¬(∀ p, circle (0, 2) 3 p → circle (0, -2) 5 p) →
  ∃ q ∈ (λ p, circle (0, 2) 3 p ∧ circle (0, -2) 5 p)), 
  ∃ l, (λ p, circle (0, 2) 3 p ∧ circle (0, -2) 5 p) = λ x, x ∈ l :=
sorry

end locus_is_hyperbola_l438_438866


namespace diane_additional_usd_needed_l438_438139

-- Definitions based on the conditions
def cookies_cost_pence : ℝ := 65 / 100 -- £0.65
def chocolates_cost_gbp : ℝ := 1.25
def discount_rate : ℝ := 0.15
def vat_rate : ℝ := 0.05
def diane_pence_gbp : ℝ := 27 / 100 -- £0.27
def conversion_rate : ℝ := 0.73

-- Main statement
theorem diane_additional_usd_needed : 
  let total_cost_before_discount := cookies_cost_pence + chocolates_cost_gbp,
      discount_amount := discount_rate * total_cost_before_discount,
      total_cost_after_discount := total_cost_before_discount - discount_amount,
      vat_amount := vat_rate * total_cost_after_discount,
      total_cost_with_vat := total_cost_after_discount + vat_amount,
      total_cost_with_pence_removed := total_cost_with_vat - diane_pence_gbp,
      total_cost_in_usd := total_cost_with_pence_removed / conversion_rate
  in 
  total_cost_in_usd.ceil = 1.96 := sorry

end diane_additional_usd_needed_l438_438139


namespace ones_digit_of_8_pow_50_l438_438696

theorem ones_digit_of_8_pow_50 : (8 ^ 50) % 10 = 4 := by
  sorry

end ones_digit_of_8_pow_50_l438_438696


namespace angle_AF1B_ellipse_eccentricity_l438_438873

-- Define the conditions
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) 
variables {F1 F2 A B P : Finset (ℝ × ℝ)}
variables (ellipse_eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
variables (F1_circle_radius : ∀ p ∈ F1, ∀ q ∈ F2, dist p q = 2 * c)
variables (A_on_yaxis : ∀ x ∈ A, x = 0) (B_on_yaxis : ∀ x ∈ B, x = 0)
variables (A_outside_ellipse : ∀ (x y: ℝ), (x, y) ∈ A → x^2 / a^2 + y^2 / b^2 > 1)
variables (B_outside_ellipse : ∀ (x y: ℝ), (x, y) ∈ B → x^2 / a^2 + y^2 / b^2 > 1)
variables (P_on_ellipse : ∀ (x y: ℝ), (x, y) ∈ P → x^2 / a^2 + y^2 / b^2 = 1)
variables (F2P_dot_F2B_zero : ∀ (xp yp xb yb xf2 yf2 : ℝ), 
  xp = xf2 ∧ yp = yf2 → xb = xf2 ∧ yb = -yf2 → xf2 * xf2 + yf2 * yf2 = 0)

-- Define the goals
theorem angle_AF1B : 
  ∃ angle : ℝ, angle = 120 :=
by sorry

theorem ellipse_eccentricity : 
  ∃ e : ℝ, e = sqrt 3 - 1 :=
by sorry

end angle_AF1B_ellipse_eccentricity_l438_438873


namespace no_valid_arrangement_l438_438008

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438008


namespace sara_bought_cards_l438_438337

-- Definition of the given conditions
def initial_cards : ℕ := 39
def torn_cards : ℕ := 9
def remaining_cards_after_sale : ℕ := 15

-- Derived definition: Number of good cards before selling to Sara
def good_cards_before_selling : ℕ := initial_cards - torn_cards

-- The statement we need to prove
theorem sara_bought_cards : good_cards_before_selling - remaining_cards_after_sale = 15 :=
by
  sorry

end sara_bought_cards_l438_438337


namespace consumption_increase_l438_438676

theorem consumption_increase (T C : ℝ) (0.84 * T * (C * (1 + P / 100)) = 0.966 * T * C) : P = 15 :=
by
  sorry

end consumption_increase_l438_438676


namespace integral_squared_geq_l438_438598

variable {n : ℕ} (f : ℝ → ℝ) (h : 0 ≤ f ∧ integrable_on f (0, 1))

theorem integral_squared_geq (h0 : ∫ x in 0..1, 1 = 1)
                             (h1 : ∫ x in 0..1, x * f x = 1)
                             (h2 : ∀ k : ℕ, k ≤ n → ∫ x in 0..1, x^k * f x = 1) :
                              ∫ x in 0..1, (f x)^2 ≥ (n + 1)^2 :=
by
  sorry

end integral_squared_geq_l438_438598


namespace problem1_problem2_l438_438199

-- Given points A and B and the condition on slopes
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := -1 }
def B : Point := { x := 0, y := 1 }

-- Definition of the trajectory C
def C_trajectory (M : Point) : Prop :=
  M.x ≠ 0 ∧ (M.x^2 / 2 + M.y^2 = 1)

-- Definition of line passing through a point with slope k
def line_eq (D : Point) (k : ℝ) (x : ℝ) : ℝ :=
  k * x + D.y

-- Proof outline for problem 1
theorem problem1 (M : Point) (h_slope_product : ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = -1 / 2) :
  C_trajectory M :=
sorry

-- Proof outline for problem 2
theorem problem2 (D : Point) (hD : D = {x := 0, y := 2}) (k : ℝ)
  (h_intersects : ∃ (E F : Point), E ≠ F ∧ C_trajectory E ∧ C_trajectory F ∧ E.y = line_eq D k E.x ∧ F.y = line_eq D k F.x)
  (A_OEF : ℝ) :
  (0, A_OEF] = (0, (Real.sqrt 2) / 2] :=
sorry

end problem1_problem2_l438_438199


namespace solution_l438_438271

noncomputable def line_eqn (t : ℝ) : ℝ × ℝ :=
  (3 - (real.sqrt 2) / 2 * t, real.sqrt 5 + (real.sqrt 2) / 2 * t)

noncomputable def circle_polar (θ : ℝ) : ℝ :=
  2 * real.sqrt 5 * real.sin θ

noncomputable def point_P : ℝ × ℝ := (3, real.sqrt 5)

def line_standard_eqn (x y : ℝ) : Prop := x + y - 3 - real.sqrt 5 = 0

def circle_rect_eqn (x y : ℝ) : Prop := x^2 + (y - real.sqrt 5)^2 = 5

def abs_distance_sum (P : ℝ × ℝ) (t1 t2 : ℝ) (line_eqn : ℝ → ℝ × ℝ) : ℝ :=
  let A := line_eqn t1 in
  let B := line_eqn t2 in
  real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

theorem solution :
  ∃ (x y : ℝ), line_standard_eqn x y ∧ circle_rect_eqn x y ∧ (abs_distance_sum point_P (3*real.sqrt 2 / 2) (3*real.sqrt 2 / 2) line_eqn = 3*real.sqrt 2)
:= sorry

end solution_l438_438271


namespace area_bounded_by_curve_and_line_l438_438966

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 1)

theorem area_bounded_by_curve_and_line :
  (∫ x in 0..1, 1 - f x) = 1 - (Real.pi / 4) - (Real.log 2 / 2) :=
by
  sorry

end area_bounded_by_curve_and_line_l438_438966


namespace max_radius_approx_l438_438750

open Real

def angle_constraint (θ : ℝ) : Prop :=
  π / 4 ≤ θ ∧ θ ≤ 3 * π / 4

def wire_constraint (r θ : ℝ) : Prop :=
  16 = r * (2 + θ)

noncomputable def max_radius (θ : ℝ) : ℝ :=
  16 / (2 + θ)

theorem max_radius_approx :
  ∃ r θ, angle_constraint θ ∧ wire_constraint r θ ∧ abs (r - 3.673) < 0.001 :=
by
  sorry

end max_radius_approx_l438_438750


namespace area_inequality_l438_438170

theorem area_inequality (a b c d S : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (hS : 0 ≤ S) :
  S ≤ (a + c) / 2 * (b + d) / 2 :=
by
  sorry

end area_inequality_l438_438170


namespace evaluate_f_g_f_l438_438607

-- Define f(x)
def f (x : ℝ) : ℝ := 4 * x + 4

-- Define g(x)
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem we're proving
theorem evaluate_f_g_f : f (g (f 3)) = 1360 := by
  sorry

end evaluate_f_g_f_l438_438607


namespace total_hours_worked_l438_438363

-- Define the number of hours worked on Saturday
def hours_saturday : ℕ := 6

-- Define the number of hours worked on Sunday
def hours_sunday : ℕ := 4

-- Define the total number of hours worked on both days
def total_hours : ℕ := hours_saturday + hours_sunday

-- The theorem to prove the total number of hours worked on Saturday and Sunday
theorem total_hours_worked : total_hours = 10 := by
  sorry

end total_hours_worked_l438_438363


namespace determine_k_l438_438138

-- Defining the line equation
def line_eq (x k : ℝ) : ℝ := (-1 / 3) - 3 * k * x

-- Defining the point
def point_x : ℝ := 1 / 3
def point_y : ℝ := -8

-- The proof problem: proving that k = 167 / 3 given that the point (1 / 3, -8) satisfies the line equation
theorem determine_k (x y k : ℝ) (h : y = -8) (h_point : x = 1 / 3) (h_line : line_eq point_x k = 7 * y) : k = 167 / 3 :=
by
  -- Definition of line_eq and points are valid under the conditions given in the problem
  sorry

end determine_k_l438_438138


namespace circumcenter_on_Ω_l438_438353

-- Definitions for points, lines, and circles
variables {A B C O D E F : Point}

-- Given conditions and properties in Lean
def is_isosceles (AB AC : Line) : Prop := AB = AC
def passes_through (ω : Circle) (point : Point) : Prop := point ∈ ω
def center_in_triangle (O A B C : Point) : Prop := O ∈ triangle A B C

-- Triangle ABC is isosceles with AB = AC
axiom isosceles_triangle (A B C : Point) (h_iso : is_isosceles (line A B) (line A C)) : Prop

-- Circle ω passes through vertex C and has center O inside the triangle ABC
axiom circle_ω (ω : Circle) (C O A B : Point) (h_pass : passes_through ω C) (h_center : center_in_triangle O A B C) : Prop

-- Circle ω intersects BC at D ≠ C and AC at E ≠ C
axiom intersects_ω (ω : Circle) (B C D A E : Point) (h_BC : D ≠ C) (h_AC : E ≠ C) : Prop

-- Circumscribed circle Ω of triangle AEO intersects AC at F ≠ E
axiom circle_Ω (Ω : Circle) (A E O F : Point) (h_AEO : A ∈ Ω ∧ E ∈ Ω ∧ O ∈ Ω) (h_AF : F ∈ Ω ∧ F ≠ E)
  : Prop

-- Prove the center of the circumcircle of triangle BDF lies on Ω
theorem circumcenter_on_Ω (A B C O D E F : Point)
  (h_iso : is_isosceles (line A B) (line A C))
  (ω : Circle) (Ω : Circle)
  (hω_pass : passes_through ω C)
  (hω_center : center_in_triangle O A B C)
  (hω_intersects_BC : intersects_ω ω B C D A E)
  (hω_intersects_AC : intersects_ω ω C D A E)
  (hΩ_AEO : circle_Ω Ω A E O F)
  : circumcenter (triangle B D F) ∈ Ω :=
sorry

end circumcenter_on_Ω_l438_438353


namespace water_volume_in_B_when_A_is_0_point_4_l438_438330

noncomputable def pool_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

noncomputable def valve_rate (volume time : ℝ) : ℝ :=
  volume / time

theorem water_volume_in_B_when_A_is_0_point_4 :
  ∀ (length width depth : ℝ)
    (time_A_fill time_A_to_B : ℝ)
    (depth_A_target : ℝ),
    length = 3 → width = 2 → depth = 1.2 →
    time_A_fill = 18 → time_A_to_B = 24 →
    depth_A_target = 0.4 →
    pool_volume length width depth = 7.2 →
    valve_rate 7.2 time_A_fill = 0.4 →
    valve_rate 7.2 time_A_to_B = 0.3 →
    ∃ (time_required : ℝ),
    time_required = 24 →
    (valve_rate 7.2 time_A_to_B * time_required = 7.2) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end water_volume_in_B_when_A_is_0_point_4_l438_438330


namespace number_of_unique_arrangements_l438_438136

theorem number_of_unique_arrangements :
  ∀ (M A D : ℕ), M = 3 → A = 3 → D = 1 → (M + A + D = 7) →
  nat.factorial 7 / (nat.factorial M * nat.factorial A * nat.factorial D) = 140 :=
by
  intros M A D hM hA hD h_sum
  sorry

end number_of_unique_arrangements_l438_438136


namespace sqrt_2_irrational_l438_438394

theorem sqrt_2_irrational : ¬ ∃ (p q : ℤ), gcd p q = 1 ∧ (p:q) = 2 := 
by
  sorry

end sqrt_2_irrational_l438_438394


namespace inequality_solution_l438_438675

-- Define the problem statement formally
theorem inequality_solution (x : ℝ)
  (h1 : 2 * x > x + 1)
  (h2 : 4 * x - 1 > 7) :
  x > 2 :=
sorry

end inequality_solution_l438_438675


namespace train_length_is_150_l438_438765

noncomputable def train_length (v_km_hr : ℝ) (t_sec : ℝ) : ℝ :=
  let v_m_s := v_km_hr * (5 / 18)
  v_m_s * t_sec

theorem train_length_is_150 :
  train_length 122 4.425875438161669 = 150 :=
by
  -- It follows directly from the given conditions and known conversion factor
  -- The actual proof steps would involve arithmetic simplifications.
  sorry

end train_length_is_150_l438_438765


namespace neg_p_iff_neg_q_l438_438513

theorem neg_p_iff_neg_q (a : ℝ) : (¬ (a < 0)) ↔ (¬ (a^2 > a)) :=
by 
    sorry

end neg_p_iff_neg_q_l438_438513


namespace quadrilateral_area_shoelace_l438_438104

def vertex1 : (ℝ × ℝ) := (2, 1)
def vertex2 : (ℝ × ℝ) := (1, 6)
def vertex3 : (ℝ × ℝ) := (5, 5)
def vertex4 : (ℝ × ℝ) := (8, 3)

theorem quadrilateral_area_shoelace :
  let vertices := [vertex1, vertex2, vertex3, vertex4] in
  let area := (1/2) * | (2 * 6 + 1 * 5 + 5 * 3 + 8 * 1) - (1 * 2 + 6 * 5 + 5 * 8 + 3 * 1) | in
  area = 17.5 :=
by
  have h1: (2 * 6 + 1 * 5 + 5 * 3 + 8 * 1) = 40 := by norm_num
  have h2: (1 * 2 + 6 * 5 + 5 * 8 + 3 * 1) = 75 := by norm_num
  have h3: 40 - 75 = -35 := by norm_num
  have h4: | -35 | = 35 := by norm_num
  have h5: (1 / 2) * 35 = 17.5 := by norm_num
  show (1 / 2) * | 40 - 75 | = 17.5 from by
    rw [h1, h2, h3, h4, h5]
    norm_num

end quadrilateral_area_shoelace_l438_438104


namespace triangle_is_obtuse_l438_438942

def is_obtuse_triangle (a b c : ℕ) : Prop := a^2 + b^2 < c^2

theorem triangle_is_obtuse :
    is_obtuse_triangle 4 6 8 :=
by
    sorry

end triangle_is_obtuse_l438_438942


namespace kelly_percentage_less_than_megan_l438_438082

variable (M : ℝ) (P : ℝ)

-- The bridge's weight limit
def bridge_weight_limit := 100

-- Kelly's weight
def kelly_weight := 34

-- Mike's weight, which is 5 kg more than Megan's weight
def mike_weight := M + 5

-- Total weight of the three children being 19 kg too much to cross the bridge
def total_weight := 119

-- Kelly's weight as a percentage reduction from Megan's weight
def kelly_weight_condition := M * (1 - P) = kelly_weight

-- The equation for the total weight of the three children
def total_weight_condition := M + (M * (1 - P)) + mike_weight = total_weight

-- Proving the percentage reduction P is 15%
theorem kelly_percentage_less_than_megan :
  kelly_weight_condition M P ∧ total_weight_condition M P → P = 0.15 := by
  sorry

end kelly_percentage_less_than_megan_l438_438082


namespace percentage_increase_square_area_l438_438711

theorem percentage_increase_square_area :
  ∀ (s : ℝ), let Area_A := s^2,
                 Side_B := 2 * s,
                 Area_B := (Side_B)^2,
                 Side_C := 2.8 * s,
                 Area_C := (Side_C)^2,
                 Sum_Area_A_B := Area_A + Area_B,
                 Difference := Area_C - Sum_Area_A_B,
                 Percentage_Increase := (Difference / Sum_Area_A_B) * 100
  in Percentage_Increase = 56.8 := by
  sorry

end percentage_increase_square_area_l438_438711


namespace five_digit_repeated_digit_percentage_l438_438241

theorem five_digit_repeated_digit_percentage :
  let total_numbers := 90000
  let repeated_digit_count := 90000 - 9 * 9 * 8 * 7 * 6
  ∃ x : ℝ, abs(x - (repeated_digit_count / total_numbers * 100)) < 0.05  :=
by
  let total_numbers := 90000
  let non_repeated_numbers := 9 * 9 * 8 * 7 * 6
  let repeated_digit_count := total_numbers - non_repeated_numbers
  let x := repeated_digit_count / total_numbers * 100
  exists x
  have : abs(x - 69.8) < 0.05 := sorry
  exact this

end five_digit_repeated_digit_percentage_l438_438241


namespace probability_even_zero_points_l438_438538

noncomputable def f (a x : ℝ) : ℝ := cos (a * π / 3 * x)

def zero_points_count (a : ℝ) (s : Set ℝ) : ℕ :=
  (s ∩ {x : ℝ | f a x = 0}).toFinset.card

def even_zero_points_event (a : ℝ) (s : Set ℝ) : Prop :=
  zero_points_count a s % 2 = 0

theorem probability_even_zero_points :
  let s := Set.Icc 0 4
  let prob := Finset.univ.filter (λ a : Finset ℝ, even_zero_points_event a s).card
  let total := Finset.univ.card
  prob.toFloat1 / total.toFloat1 = 1 / 3 :=
by
  sorry

end probability_even_zero_points_l438_438538


namespace distance_midway_new_city_l438_438322

/-- Define the location of New City, Old Town, and Midway -/
def NewCity : ℂ := 0
def OldTown : ℂ := 0 + 3200 * complex.I
def Midway : ℂ := 960 + 1280 * complex.I

/-- Define the distance function on the complex plane -/
def distance (z1 z2 : ℂ) : ℝ := complex.abs (z1 - z2)

/-- Prove the distance from Midway to New City is 3200 -/
theorem distance_midway_new_city : distance Midway NewCity = 3200 := 
  sorry

end distance_midway_new_city_l438_438322


namespace c_share_of_profit_l438_438039

def investment_A := 800
def investment_B := 1000
def investment_C := 1200
def total_profit := 1000

theorem c_share_of_profit :
  let ratio := investment_A + investment_B + investment_C,
      ratio_A := investment_A / 200,
      ratio_B := investment_B / 200,
      ratio_C := investment_C / 200,
      total_parts := ratio_A + ratio_B + ratio_C,
      c_share := (ratio_C / total_parts) * total_profit in
  c_share = 400 :=
by sorry

end c_share_of_profit_l438_438039


namespace angle_equality_pentagon_l438_438442

noncomputable def pentagon (A B C D E : Type) :=
  True -- Dummy definition, because we don't need the actual points

variables {A B C D E : Type} [category A] [category B] [category C] [category D] [category E]

/-- Given that in a convex pentagon ABCDE, ∠ABC = ∠ADE and ∠AEC = ∠ADB, we want to prove that ∠BAC = ∠DAE. -/
theorem angle_equality_pentagon
  (h1 : ∠ABC = ∠ADE)
  (h2 : ∠AEC = ∠ADB) :
  ∠BAC = ∠DAE := 
sorry

end angle_equality_pentagon_l438_438442


namespace find_lambda_l438_438197

-- We define that planes alpha and beta are parallel
def parallel (α β : Type) [Plane α] [Plane β] : Prop :=
  -- condition that the normal vectors of planes α and β are proportional
  ∃ (k : ℝ), ∀ (a b : ℝ × ℝ × ℝ), (a.1, a.2, a.3) = (1, λ, 2) → (b.1, b.2, b.3) = (-3, 6, -6) →
  a = k • b

-- The theorem to be proven: given the condition above, λ must be -2
theorem find_lambda (α β : Type) [Plane α] [Plane β] (h : parallel α β) : λ = -2 := by
  sorry

end find_lambda_l438_438197


namespace monotone_f_on_pos_half_line_min_max_f_on_2_4_l438_438897

section

def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotone_f_on_pos_half_line :
  monotone_on f (set.Ici 1) := sorry

theorem min_max_f_on_2_4 :
  ∃ (a b : ℝ), (∀ x ∈ set.Icc 2 4, a ≤ f x ∧ f x ≤ b) ∧ a = f 2 ∧ b = f 4 := sorry

end

end monotone_f_on_pos_half_line_min_max_f_on_2_4_l438_438897


namespace man_cannot_row_against_stream_l438_438417

theorem man_cannot_row_against_stream (v_s v_m : ℝ) (h1 : v_s = 10) (h2 : v_m = 2) : 
  ¬ ∃ v_against : ℝ, v_against = v_m - (v_s - v_m) ∧ v_against < 0 :=
by
  have v_r := v_s - v_m
  have v_against := v_m - v_r
  have h_v_r : v_r = 8 := by calc
    v_r = v_s - v_m : by sorry
    ... = 8         : by sorry
  have h_v_against : v_against < 0 := by calc
    v_against = v_m - v_r : by sorry
    ... < 0               : by sorry
  exact ⟨v_against, h_v_against⟩

end man_cannot_row_against_stream_l438_438417


namespace processing_time_l438_438772

theorem processing_time 
  (pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ)
  (h1 : pictures = 960) (h2 : minutes_per_picture = 2) (h3 : minutes_per_hour = 60) : 
  (pictures * minutes_per_picture) / minutes_per_hour = 32 :=
by 
  sorry

end processing_time_l438_438772


namespace base_b_eq_five_l438_438933

theorem base_b_eq_five (b : ℕ) (h1 : 1225 = b^3 + 2 * b^2 + 2 * b + 5) (h2 : 35 = 3 * b + 5) :
    (3 * b + 5)^2 = b^3 + 2 * b^2 + 2 * b + 5 ↔ b = 5 :=
by
  sorry

end base_b_eq_five_l438_438933


namespace find_angle_vertex_third_cone_l438_438378

noncomputable def angle_vertex_third_cone (beta : ℝ) : Prop :=
  let alpha := real.pi / 6 in
  let fourth_cone_angle := real.pi / 3 in
  let angle_at_third_cone := 2 * real.arccot(real.sqrt(3) + 4) in
  2 * beta = angle_at_third_cone

theorem find_angle_vertex_third_cone (beta : ℝ) :
  2 * beta = 2 * real.arccot(real.sqrt(3) + 4) :=
by
  sorry

end find_angle_vertex_third_cone_l438_438378


namespace distance_apart_l438_438470

def race_total_distance : ℕ := 1000
def distance_Arianna_ran : ℕ := 184

theorem distance_apart :
  race_total_distance - distance_Arianna_ran = 816 :=
by
  sorry

end distance_apart_l438_438470


namespace percentage_increase_of_cars_l438_438101

theorem percentage_increase_of_cars :
  ∀ (initial final : ℕ), initial = 24 → final = 48 → ((final - initial) * 100 / initial) = 100 :=
by
  intros
  sorry

end percentage_increase_of_cars_l438_438101


namespace unique_natural_number_l438_438146

theorem unique_natural_number (n a b : ℕ) (h1 : a ≠ b) 
(h2 : digits_in_reverse_order (n^a + 1) (n^b + 1)) : n = 3 := sorry

end unique_natural_number_l438_438146


namespace maximize_area_l438_438079

variable (x : ℝ)
def fence_length : ℝ := 240 - 2 * x
def area (x : ℝ) : ℝ := x * fence_length x

theorem maximize_area : fence_length 60 = 120 :=
  sorry

end maximize_area_l438_438079


namespace reciprocal_relationship_l438_438860

theorem reciprocal_relationship (a b : ℝ) (h₁ : a = 2 - Real.sqrt 3) (h₂ : b = Real.sqrt 3 + 2) : 
  a * b = 1 :=
by
  rw [h₁, h₂]
  sorry

end reciprocal_relationship_l438_438860


namespace find_side_b_in_triangle_l438_438572

noncomputable def triangle_side_b (a A : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := Real.sqrt (1 - cosB^2)
  let sinA := Real.sin A
  (a * sinB) / sinA

theorem find_side_b_in_triangle :
  triangle_side_b 5 (Real.pi / 4) (3 / 5) = 4 * Real.sqrt 2 :=
by
  sorry

end find_side_b_in_triangle_l438_438572


namespace slope_of_line_l438_438163

theorem slope_of_line (x y : ℝ) (h : 4 * x - 7 * y = 28) : (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 7) :=
by
  -- Proof omitted
  sorry

end slope_of_line_l438_438163


namespace arithmetic_mean_two_digit_numbers_divisible_by_8_l438_438692

theorem arithmetic_mean_two_digit_numbers_divisible_by_8
  (smallest : ℕ) (largest : ℕ) (common_difference : ℕ) (n : ℕ)
  (h_smallest : smallest = 16) (h_largest : largest = 96) (h_diff : common_difference = 8) 
  (h_n : n = (largest - smallest) / common_difference + 1) :
  (let S := (n * (smallest + largest)) / 2 in S / n = 56) :=
by
  sorry

end arithmetic_mean_two_digit_numbers_divisible_by_8_l438_438692


namespace vector_a_solution_l438_438894

variable (x y : ℝ)

def vector_a := (x, y)
def vector_b := (1, 2)
def dot_product (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
def magnitude (a : ℝ × ℝ) := Real.sqrt (a.1^2 + a.2^2)

theorem vector_a_solution (h1 : dot_product (x, y) (1, 2) = 0) (h2 : magnitude (x, y) = 2 * Real.sqrt 5) :
  (x = 4 ∧ y = -2) ∨ (x = -4 ∧ y = 2) := sorry

end vector_a_solution_l438_438894


namespace angle_bisector_perpendicular_incenter_circumcenter_l438_438401

theorem angle_bisector_perpendicular_incenter_circumcenter {A B C : Type*}
  (a b c : ℝ)
  (h : a = (b + c) / 2)
  (triangle_ABC : triangle A B C) :
  -- Definitions for (I) incenter, (O) circumcenter, and angle bisector omitted for brevity
  let I := incenter A B C in
  let O := circumcenter A B C in
  is_perpendicular (angle_bisector A) (I - O) := sorry

end angle_bisector_perpendicular_incenter_circumcenter_l438_438401


namespace AE_eq_incircle_radius_l438_438974

noncomputable def is_incenter (O : Point) (A B C : Point) : Prop := sorry -- defines O as the incenter of triangle ABC
noncomputable def is_midpoint (A1 : Point) (B C : Point) : Prop := sorry -- defines A1 as the midpoint of BC
noncomputable def altitude_intersection (A1 O E A : Point) : Prop := sorry -- defines the intersection of line A1O with the altitude from A as point E

theorem AE_eq_incircle_radius (A B C A1 O E : Point) (rho : ℝ)
  (h1 : is_triangle A B C)
  (h2 : is_midpoint A1 B C)
  (h3 : is_incenter O A B C)
  (h4 : altitude_intersection A1 O E A)
  (h5 : incircle_radius A B C O = rho) :
  segment_length A E = rho := sorry

end AE_eq_incircle_radius_l438_438974


namespace sum_of_leading_digits_l438_438455

-- Define the number M as provided in the conditions
def M : ℝ := 8.888 * 10^255 -- Simplified representation

-- Define g(r) as the leading digit of the r-th root of M
def leading_digit (x : ℝ) : ℕ :=
  nat.floor (x / (10 ^ (nat.floor (real.log10 x))))

def g (r : ℕ) : ℕ :=
  leading_digit (M ^ (1 / r))

-- Define the main theorem to be proved
theorem sum_of_leading_digits : g 3 + g 4 + g 5 + g 7 + g 8 = 7 := by sorry

end sum_of_leading_digits_l438_438455


namespace B_interval_l438_438117

noncomputable def g : ℕ → ℝ
| 12 := Real.log 12
| (n+1) := Real.log (n + 1 + g n)

theorem B_interval : 
  let B := g 2025 in 
  Real.log 2028 < B ∧ B < Real.log 2029 :=
sorry

end B_interval_l438_438117


namespace problem_l438_438947

theorem problem (n : ℕ) (h : n = 2011) :
  let p := ∏ i in finset.range (n + 1), i
  ∃ k : ℕ, (2010 : ℕ) ^ k ∣ p ∧ ∀ k', (2010 : ℕ) ^ k' ∣ p → k' ≤ 30 :=
begin
  sorry
end

end problem_l438_438947


namespace sequence_a4_value_l438_438425

theorem sequence_a4_value :
  let seq : ℕ → ℚ := λ n, if n = 1 then 2 else if n = 2 then 4 else 
                        seq (n - 2) * seq (n - 1) / (3 * seq (n - 2) - 2 * seq (n - 1))
  in seq 4 = -4 / 5 :=
by
  let seq : ℕ → ℚ := λ n, if n = 1 then 2 else if n = 2 then 4 else 
                          seq (n - 2) * seq (n - 1) / (3 * seq (n - 2) - 2 * seq (n - 1))
  have h1 : seq 1 = 2 := by simp [seq]
  have h2 : seq 2 = 4 := by simp [seq]
  let s3 := seq 3
  have h3 : s3 = seq 1 * seq 2 / (3 * seq 1 - 2 * seq 2) := by simp [seq]
  let s4 := seq 4
  have h4 : s4 = seq 2 * seq 3 / (3 * seq 2 - 2 * seq 3) := by simp [seq]
  sorry

end sequence_a4_value_l438_438425


namespace poly_degree_l438_438461

-- Define the polynomial
def poly := 3 + 7 * x^2 + (1/2) * x^5 - 10 * x + 11

-- Statement to prove the degree of the polynomial
theorem poly_degree : polynomial.degree (3 + 7 * x^2 + (1/2) * x^5 - 10 * x + 11) = 5 :=
by
  sorry

end poly_degree_l438_438461


namespace range_of_m_l438_438224

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt (1 + x) + Real.sqrt (1 - x)) * (2 * Real.sqrt (1 - x^2) - 1)

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x = m) ↔ -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
sorry

end range_of_m_l438_438224


namespace cards_not_divisible_by_7_l438_438017

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438017


namespace no_real_or_imaginary_values_l438_438809

theorem no_real_or_imaginary_values (t : ℂ) : ¬ (sqrt (49 - t^2) + 7 = 0) := 
sorry

end no_real_or_imaginary_values_l438_438809


namespace even_iff_b_eq_zero_l438_438310

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Given that f' is an even function, prove that b = 0.
theorem even_iff_b_eq_zero (h : ∀ x : ℝ, f' x = f' (-x)) : b = 0 :=
  sorry

end even_iff_b_eq_zero_l438_438310


namespace total_students_l438_438464

theorem total_students (x : ℕ) (h1 : 3 * x + 8 = 3 * x + 5) (h2 : 5 * (x - 1) + 3 > 3 * x + 8) : x = 6 :=
sorry

end total_students_l438_438464


namespace find_diameter_C_l438_438110

noncomputable def diameter_of_circle_C (diameter_of_D : ℝ) (ratio_shaded_to_C : ℝ) : ℝ :=
  let radius_D := diameter_of_D / 2
  let radius_C := radius_D / (2 * Real.sqrt ratio_shaded_to_C)
  2 * radius_C

theorem find_diameter_C :
  let diameter_D := 20
  let ratio_shaded_area_to_C := 7
  diameter_of_circle_C diameter_D ratio_shaded_area_to_C = 5 * Real.sqrt 2 :=
by
  -- The proof is omitted.
  sorry

end find_diameter_C_l438_438110


namespace right_triangle_largest_angle_l438_438259

theorem right_triangle_largest_angle (α β : ℝ) (h : 7 * β = 2 * α) 
 (h1 : α + β = 90) : 
  ∃ γ: ℝ, is_right_triangle α β γ ∧ γ = 90 :=
sorry

end right_triangle_largest_angle_l438_438259


namespace polynomial_value_l438_438946

theorem polynomial_value :
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  5 * a + 3 * b + 2 * c + d = 25 :=
by
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  sorry

end polynomial_value_l438_438946


namespace monotonically_decreasing_interval_l438_438941

theorem monotonically_decreasing_interval (a : ℝ) :
  (∀ x ∈ Ioo 1 2, (ln x + a * x^2 - 2) is_monotone_decreasing_in Ioo 1 2) ↔ a < -1 / 8 := by
  sorry

end monotonically_decreasing_interval_l438_438941


namespace series_limit_zero_l438_438485

noncomputable def sum_series : ℝ :=
  ∑' n : ℕ, (3^⟨Real.log n / Real.log 2⟩ + 3^(- ⟨Real.log n / Real.log 2⟩)) / 3^n

theorem series_limit_zero : 
  sum_series = 0 := 
sorry

end series_limit_zero_l438_438485


namespace length_CD_equals_8_l438_438503

variables (A B C D L : Type) [metric_space A] [metric_space B] [metric_space C] 
          [metric_space D] [metric_space L]
          (angleD : angle A D B = 100) (BC_length : dist B C = 12)
          (AD_length : dist A D = 12) (LD_length : dist L D = 4)
          (angleABL : angle A B L = 50)
          (parallelogram_ABCD : parallelogram A B C D)

theorem length_CD_equals_8 :
  dist C D = 8 :=
sorry

end length_CD_equals_8_l438_438503


namespace percentage_of_720_equals_356_point_4_l438_438386

theorem percentage_of_720_equals_356_point_4 : 
  let part := 356.4
  let whole := 720
  (part / whole) * 100 = 49.5 :=
by
  sorry

end percentage_of_720_equals_356_point_4_l438_438386


namespace remainder_when_7x_div_9_l438_438047

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9_l438_438047


namespace line_passes_fixed_point_l438_438468

open Real

theorem line_passes_fixed_point
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)
  (M N : ℝ × ℝ)
  (hM : M.1^2 / a^2 + M.2^2 / b^2 = 1)
  (hN : N.1^2 / a^2 + N.2^2 / b^2 = 1)
  (hMAhNA : (M.1 + a) * (N.1 + a) + M.2 * N.2 = 0):
  ∃ (P : ℝ × ℝ), P = (a * (b^2 - a^2) / (a^2 + b^2), 0) ∧ (N.2 - M.2) * (P.1 - M.1) = (P.2 - M.2) * (N.1 - M.1) :=
sorry

end line_passes_fixed_point_l438_438468


namespace function_ordering_l438_438484

variable {R : Type*} [LinearOrder R] [LinearOrderedField R] 
variable (f : R → R) (a b c : R)

def is_even_function (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def is_monotone_incr_on_negative (f : R → R) : Prop :=
  ∀ (x₁ x₂ : R), x₁ < 0 → x₂ < 0 → x₁ < x₂ → f x₁ < f x₂

noncomputable def a_value : R := (2 : R)^(0.3)
noncomputable def b_value : R := (2 : R)^(0.5)
noncomputable def c_value : R := (3 : R)^(-0.5)

theorem function_ordering (h_even : is_even_function f)
  (h_mono_neg : is_monotone_incr_on_negative f) :
  a_value = a → 
  b_value = b →
  c_value = c →
  f c > f (-a) ∧ f (-a) > f (-b) := by
  sorry

end function_ordering_l438_438484


namespace bread_last_days_l438_438682

theorem bread_last_days (num_members : ℕ) (breakfast_slices : ℕ) (snack_slices : ℕ) (slices_per_loaf : ℕ) (num_loaves : ℕ) :
  num_members = 4 →
  breakfast_slices = 3 →
  snack_slices = 2 →
  slices_per_loaf = 12 →
  num_loaves = 5 →
  (num_loaves * slices_per_loaf) / (num_members * (breakfast_slices + snack_slices)) = 3 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end bread_last_days_l438_438682


namespace point_on_x_axis_l438_438935

theorem point_on_x_axis (m : ℤ) (hx : 2 + m = 0) : (m - 3, 2 + m) = (-5, 0) :=
by sorry

end point_on_x_axis_l438_438935


namespace mean_equality_l438_438367

theorem mean_equality (z : ℝ) :
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 :=
by
  intro h
  sorry

end mean_equality_l438_438367


namespace not_possible_arrange_cards_l438_438018

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438018


namespace placement_of_four_l438_438349

/--
A 4x4 table populated with numbers 1 through 4, where each
row, each column, and each 2x2 subgrid contains all different numbers.
We need to prove the placement of number 4 in each row.
-/
theorem placement_of_four :
  ∀ (table : Fin 4 × Fin 4 → Fin 4),
  (∀ i, ∀ j1 j2, j1 ≠ j2 → table (i, j1) ≠ table (i, j2)) → -- unique in row
  (∀ j, ∀ i1 i2, i1 ≠ i2 → table (i1, j) ≠ table (i2, j)) → -- unique in column
  (∀ (i1 i2 j1 j2 : Fin 2), 
      i1 ≠ i2 ∨ j1 ≠ j2 → 
      table (i1, j1) ≠ table (i2, j2) ∧ 
      table (i1, j1 + 2) ≠ table (i2, j2)) → -- unique in subgrid
  table (⟨0, by decide⟩, ⟨2, by decide⟩) = 3 ∧ -- A-3
  table (⟨1, by decide⟩, ⟨0, by decide⟩) = 1 ∧ -- B-1
  table (⟨2, by decide⟩, ⟨1, by decide⟩) = 2 ∧ -- C-2
  table (⟨3, by decide⟩, ⟨3, by decide⟩) = 4 -- D-4 :=
sorry

end placement_of_four_l438_438349


namespace complex_addition_l438_438341

def a : ℂ := -5 + 3 * Complex.i
def b : ℂ := 2 - 7 * Complex.i

theorem complex_addition : a + b = -3 - 4 * Complex.i := 
by 
  sorry

end complex_addition_l438_438341


namespace correct_substitution_l438_438701

theorem correct_substitution (x y : ℤ) (h1 : x = 3 * y - 1) (h2 : x - 2 * y = 4) :
  3 * y - 1 - 2 * y = 4 :=
by
  sorry

end correct_substitution_l438_438701


namespace total_hamburgers_menu_l438_438552

def meat_patties_choices := 4
def condiment_combinations := 2 ^ 9

theorem total_hamburgers_menu :
  meat_patties_choices * condiment_combinations = 2048 :=
by
  sorry

end total_hamburgers_menu_l438_438552


namespace find_sticker_price_l438_438922

-- Define the conditions and the question
def storeA_price (x : ℝ) : ℝ := 0.80 * x - 80
def storeB_price (x : ℝ) : ℝ := 0.70 * x - 40
def heather_saves_30 (x : ℝ) : Prop := storeA_price x = storeB_price x + 30

-- Define the main theorem
theorem find_sticker_price : ∃ x : ℝ, heather_saves_30 x ∧ x = 700 :=
by
  sorry

end find_sticker_price_l438_438922


namespace circle_line_intersect_l438_438075

theorem circle_line_intersect (a : ℝ) :
  (∀ (b : ℝ), b ∈ set.Icc (-2 : ℝ) 2 → ((|b| / (real.sqrt 2)) ≤ real.sqrt a)) →
  (∃ (b : ℝ), b ∈ set.Icc (-(real.sqrt (2 * a))) (real.sqrt (2 * a))) →
  (2 * (real.sqrt (2 * a)) / 4 = (1 / 2)) → a = (1 / 2) :=
begin
  sorry
end

end circle_line_intersect_l438_438075


namespace triangles_in_plane_l438_438408

-- Define the problem
theorem triangles_in_plane (n : ℕ) (R : ℕ → ℕ)
  (hR : ∀ (n : ℕ), R n = n * (n + 1) / 2 + 1)
  (h_lines_non_parallel : ∀ (i j : ℕ), i ≠ j → ¬ ∥ i ∥ = ∥ j ∥)
  (h_lines_non_concurrent : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ (i ∩ j ∩ k).nonempty)
  (h_lines_count : n = 300) : 
  R 300 ≥ 100 :=
sorry

end triangles_in_plane_l438_438408


namespace trigonometric_identity_l438_438445

theorem trigonometric_identity : 
  sin (44 * Real.pi / 180) * cos (14 * Real.pi / 180) - 
  cos (44 * Real.pi / 180) * cos (76 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof not required for this exercise
  sorry

end trigonometric_identity_l438_438445


namespace necessary_but_not_sufficient_l438_438235

theorem necessary_but_not_sufficient (p q : Prop) : p → (p ∧ q) ∧ ¬(p → (p ∧ q)) → (p ∧ q → p) ∧ ¬(p ∧ q → p) :=
begin
  sorry -- Proof is omitted
end

end necessary_but_not_sufficient_l438_438235


namespace binomial_odd_condition_l438_438630

open Nat

theorem binomial_odd_condition (n k : ℕ) :
  (nat.choose n k % 2 = 1) ↔ (∀ i : ℕ, test_bit k i = tt → test_bit n i = tt) :=
by sorry

end binomial_odd_condition_l438_438630


namespace ratio_instore_sales_l438_438072

-- Define the problem conditions
variables (initial_inventory saturday_instore_sales saturday_online_sales 
           online_sales_increase sunday_shipment_received final_inventory : ℤ)

-- Define the given values
def initial_inventory := 743
def saturday_instore_sales := 37
def saturday_online_sales := 128
def online_sales_increase := 34
def sunday_shipment_received := 160
def final_inventory := 502

-- Helper variables
def sunday_online_sales := saturday_online_sales + online_sales_increase

-- Equation to represent the change in inventory
def net_books_sold (sunday_instore_sales : ℤ) : ℤ := 
  let total_saturday_sales := saturday_instore_sales + saturday_online_sales in
  let total_sunday_sales := sunday_instore_sales + sunday_online_sales in
  (total_saturday_sales + total_sunday_sales) - sunday_shipment_received

-- Prove that the ratio is 2:1
theorem ratio_instore_sales (sunday_instore_sales : ℤ) 
  (h1 : net_books_sold sunday_instore_sales = initial_inventory - final_inventory) :
  sunday_instore_sales / saturday_instore_sales = 2 :=
by 
  have eq1 : 165 = saturday_instore_sales + saturday_online_sales := rfl
  have eq2 : 162 = sunday_online_sales := rfl
  have eq3 : net_books_sold sunday_instore_sales = 327 + sunday_instore_sales - 160 := rfl
  have eq4 : initial_inventory - final_inventory = 241 := rfl
  sorry

end ratio_instore_sales_l438_438072


namespace find_locus_of_Q_l438_438585

-- Define the coordinates of the vertices of the triangle
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -1)
def C : ℝ × ℝ := (1, 3)

-- Equation of the locus of point Q
def equation_of_locus_Q (x y : ℝ) : Prop :=
  2 * x - y - 3 = 0

-- Define the movement of point P along the line BC and the equation of point Q
theorem find_locus_of_Q : ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
  let P : ℝ × ℝ := (1 - t) • B + t • C in
  let PQ : ℝ × ℝ := (A.1 - P.1, A.2 - P.2) + (B.1 - P.1, B.2 - P.2) + (C.1 - P.1, C.2 - P.2) in
  let Q : ℝ × ℝ := (P.1 + PQ.1, P.2 + PQ.2) in
  equation_of_locus_Q Q.1 Q.2 :=
sorry

end find_locus_of_Q_l438_438585


namespace probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l438_438081

-- Definitions of the probabilities
def P_A := 0.24
def P_B := 0.28
def P_C := 0.19
def P_D := 0.16
def P_E := 0.13

-- Prove that the probability of hitting the 10 or 9 rings is 0.52
theorem probability_of_hitting_10_or_9 : P_A + P_B = 0.52 :=
  by sorry

-- Prove that the probability of hitting at least the 7 ring is 0.87
theorem probability_of_hitting_at_least_7 : P_A + P_B + P_C + P_D = 0.87 :=
  by sorry

-- Prove that the probability of hitting less than 8 rings is 0.29
theorem probability_of_hitting_less_than_8 : P_D + P_E = 0.29 :=
  by sorry

end probability_of_hitting_10_or_9_probability_of_hitting_at_least_7_probability_of_hitting_less_than_8_l438_438081


namespace find_slope_intercept_l438_438751

def line_eqn (x y : ℝ) : Prop :=
  -3 * (x - 5) + 2 * (y + 1) = 0

theorem find_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_eqn x y → y = m * x + b) ∧ (m = 3/2) ∧ (b = -17/2) := sorry

end find_slope_intercept_l438_438751


namespace polynomial_identity_roots_l438_438134

-- Define the given polynomial
noncomputable def poly (x : ℝ) (m : ℝ) : ℝ :=
  x^4 + 2*x^3 - 23*x^2 + 12*x + m

-- Define the product of trinomials
noncomputable def trinomial_product (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  (x^2 + a*x + c) * (x^2 + b*x + c)

-- The given conditions
axiom h1 : ∀ a b c : ℝ, a + b = 2
axiom h2 : ∀ (a b c : ℝ), ab + 2*c = -23
axiom h3 : ∀ (a b c : ℝ), c * (a + b) = 12

-- Values of a, b, and c consistent with h1, h2, h3, and unique m
axiom ha : ℝ := 7
axiom hb : ℝ := -5
axiom hc : ℝ := 6
axiom hm : ℝ := 36

-- The main theorem to prove
theorem polynomial_identity : poly x m = trinomial_product x a b c :=
sorry

-- The roots of the polynomial
noncomputable def poly_roots : Set ℝ := {x : ℝ | poly x 36 = 0}

-- Expected roots
axiom root1 : (2 : ℝ)
axiom root2 : (3 : ℝ)
axiom root3 : (-1 : ℝ)
axiom root4 : (-6 : ℝ)

-- Theorem to prove the roots
theorem roots : poly_roots = {root1, root2, root3, root4} :=
sorry

end polynomial_identity_roots_l438_438134


namespace slant_asymptote_and_sum_of_slope_and_intercept_l438_438124

noncomputable def f (x : ℚ) : ℚ := (3 * x^2 + 5 * x + 1) / (x + 2)

theorem slant_asymptote_and_sum_of_slope_and_intercept :
  (∀ x : ℚ, ∃ (m b : ℚ), (∃ r : ℚ, (r = f x ∧ (r + (m * x + b)) = f x)) ∧ m = 3 ∧ b = -1) →
  3 - 1 = 2 :=
by
  sorry

end slant_asymptote_and_sum_of_slope_and_intercept_l438_438124


namespace marika_mother_age_twice_in_2036_l438_438621

theorem marika_mother_age_twice_in_2036 :
  ∀ (marika_age_2006 mother_age_2006 : ℕ), (marika_age_2006 = 10) → (mother_age_2006 = 50) →
  ∃ year : ℕ, (year = 2036 ∧ (2 * (marika_age_2006 + (year - 2006)) = mother_age_2006 + (year - 2006))) := 
by
  intros marika_age_2006 mother_age_2006 h_marika h_mother
  use 2036
  split
  . refl
  . rw [h_marika, h_mother]
    norm_num
    sorry

end marika_mother_age_twice_in_2036_l438_438621


namespace remainder_of_n_plus_4500_l438_438562

theorem remainder_of_n_plus_4500 (n : ℕ) (h : n % 6 = 1) : (n + 4500) % 6 = 1 := 
by
  sorry

end remainder_of_n_plus_4500_l438_438562


namespace find_f_neg_one_l438_438521

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)

theorem find_f_neg_one (f : ℝ → ℝ) (h_even : even_function f)
  (h_ge_zero : ∀ x, 0 ≤ x → f(x) = x + 1) : f (-1) = 2 := by
  sorry

end find_f_neg_one_l438_438521


namespace angle_AMB_150_l438_438975

theorem angle_AMB_150 
  (A B C M : Point)
  (hM_bisector_of_B : M lies_on_angle_bisector_of B)
  (hAM_eq_AC : dist A M = dist A C)
  (hBCM_eq_30 : ∠ B C M = 30°) : 
  ∠ A M B = 150° := 
by 
  sorry

end angle_AMB_150_l438_438975


namespace shortest_chord_length_l438_438532

noncomputable def circle_equation := 
  ∀ (x y : ℝ), (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def line_equation (m : ℝ) := 
  ∀ (x y : ℝ), (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem shortest_chord_length (m : ℝ) : 
  ∃ (length : ℝ), 
  circle_equation ∧ line_equation m → length = 4 * Real.sqrt 5 := 
begin
  sorry
end

end shortest_chord_length_l438_438532


namespace last_colored_cell_is_51_50_l438_438624

def last_spiral_cell (width height : ℕ) : ℕ × ℕ :=
  -- Assuming an external or pre-defined process to calculate the last cell for a spiral pattern
  sorry 

theorem last_colored_cell_is_51_50 :
  last_spiral_cell 200 100 = (51, 50) :=
sorry

end last_colored_cell_is_51_50_l438_438624


namespace triangle_incircle_radius_l438_438431

/-- Given a triangle with side lengths 3, 4, 5, prove that the radius of the inscribed circle is 1. -/
theorem triangle_incircle_radius :
  ∀ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = 5 → 
                (∃ r, r = 1 ∧ 
                       let s := (a + b + c) / 2 in
                       let A := (1 / 2) * 3 * 4 in
                       r = A / s) :=
by
  intros a b c h
  sorry

end triangle_incircle_radius_l438_438431


namespace hexagon_angle_Q_l438_438554

-- Define the given angles
def angle_S : ℝ := 120
def angle_T : ℝ := 130
def angle_U : ℝ := 140
def angle_V : ℝ := 100
def angle_W : ℝ := 85

-- Define the sum of the interior angles of the hexagon
def sum_interior_angles_hexagon (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the proof statement
theorem hexagon_angle_Q : angle_S + angle_T + angle_U + angle_V + angle_W + Q = sum_interior_angles_hexagon 6 → Q = 145 :=
by
  assume h : angle_S + angle_T + angle_U + angle_V + angle_W + Q = sum_interior_angles_hexagon 6
  sorry

end hexagon_angle_Q_l438_438554


namespace tan_third_quadrant_angle_l438_438057

theorem tan_third_quadrant_angle
  (α : ℝ)
  (h : (sin (3 * Real.pi / 2 - α) * cos (Real.pi / 2 + α)) / 
       (cos (Real.pi - α) * sin (3 * Real.pi - α) * sin (-Real.pi - α)) = 3)
  (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.tan α = 1 / (2 * Real.sqrt 2) := 
sorry

end tan_third_quadrant_angle_l438_438057


namespace max_lambda_l438_438496

theorem max_lambda (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) :
  ∃ λ, (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 → a^2 + b^2 + c^2 + λ * (Real.sqrt (a*b*c)) ≤ 1) ∧ λ = 2 * Real.sqrt 3 :=
begin
  sorry
end

end max_lambda_l438_438496


namespace approximation_of_11_28_relative_to_10000_l438_438781

def place_value_to_approximate (x : Float) (reference : Float) : String :=
  if x < reference / 10 then "tens"
  else if x < reference / 100 then "hundreds"
  else if x < reference / 1000 then "thousands"
  else if x < reference / 10000 then "ten thousands"
  else "greater than ten thousands"

theorem approximation_of_11_28_relative_to_10000:
  place_value_to_approximate 11.28 10000 = "hundreds" :=
by
  -- Insert proof here
  sorry

end approximation_of_11_28_relative_to_10000_l438_438781


namespace max_d_value_l438_438605

theorem max_d_value
  (diameter : ℝ) (MN : set ℝ) (A B C : set ℝ)
  (h1 : diameter = 1)
  (h2 : A = midpoint MN)
  (h3 : dist ℝ B M = 4 / 5)
  (h4 : C ∈ semicircular_arc MN)
  (d : ℝ := length(segments(MN, chords(A, C)) ∩ segments(MN, chords(B, C))))
  (h5 : ∃ r s t : ℕ, d = r - s * real.sqrt t ∧ t ∣ t ∧ ∀ p : prime, ¬(p^2 ∣ t)) :
  ∃ r s t : ℕ, r = 13 ∧ s = 3 ∧ t = 8 ∧ r + s + t = 24 := by
    existsi 13, 3, 8
    sorry

end max_d_value_l438_438605


namespace player_A_wins_or_forces_exceed_l438_438327

def domino_points := [1, 2, 3, 4, 5]

def reachable_scores :=
  list.scanl (+) 0 [4, 7, 6, 7, 6, 7]  -- This generates [4, 11, 17, 24, 30, 37]

theorem player_A_wins_or_forces_exceed :
  ∀ (coin_place : ℕ → ℕ), (coin_place 0 = 4) →
  (∃ n, (reachable_scores n = 37) ∨ (reachable_scores n > 37)) :=
by
  intro coin_place h
  have strategy : list ℕ := [4, 11, 17, 24, 30, 37]
  sorry  -- Proof of the theorem goes here

end player_A_wins_or_forces_exceed_l438_438327


namespace angle_D_is_75_l438_438571

theorem angle_D_is_75
    {A B C D E : Type*}
    [inst : linear_ordered_field ℝ]
    (AB BC CD CE : ℝ)
    (h1 : AB = BC)
    (h2 : BC = CD)
    (h3 : CD = CE)
    (a b : ℝ)
    (h4 : a = 4 * b)
    (h5 : A ∈ triangle A B C)
    (h6 : C ∈ triangle C D E)
    : ∠D = 75 :=
by
  sorry

end angle_D_is_75_l438_438571


namespace relationship_among_a_b_c_l438_438205

noncomputable def a : ℝ := 3 ^ Real.log (1 / 2)
noncomputable def b : ℝ := Real.logBase 24 25
noncomputable def c : ℝ := Real.logBase 25 26

theorem relationship_among_a_b_c : b > c ∧ c > a := by
  sorry

end relationship_among_a_b_c_l438_438205


namespace right_triangle_area_l438_438992

theorem right_triangle_area (a b c p S : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a^2 + b^2 = c^2)
  (h4 : p = (a + b + c) / 2) (h5 : S = a * b / 2) :
  p * (p - c) = S ∧ (p - a) * (p - b) = S :=
sorry

end right_triangle_area_l438_438992


namespace ngons_sign_impossible_transformations_K_value_for_n_l438_438957

-- Define the problem for proving the impossible configuration transformation for n-gon
theorem ngons_sign_impossible_transformations 
  (n : ℕ) 
  (h_n_gt_2 : n > 2) :
  ∃ (arrangement : fin n → int), let T := {(i, j) | i < j ∧ j - i ∣ n} in 
    ∀ t ∈ T, (∀ (a b : fin n), a ≠ b → arrangement a ≠ arrangement b) →
    ¬  ∀ (v : fin n), arrangement v = 1 :=
begin
  sorry
end

-- Define the problem of finding K(n) for any n, and specifically for n = 200
theorem K_value_for_n 
  (n : ℕ) :
  ∃ (K : ℕ → ℕ), (K 200 = 2 ^ 80) :=
begin
  sorry
end

end ngons_sign_impossible_transformations_K_value_for_n_l438_438957


namespace dragon_jewels_end_l438_438744

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l438_438744


namespace not_possible_1_and_3_arrangement_l438_438977

theorem not_possible_1_and_3_arrangement :
  ¬ (∃ (matrix: array (fin 5) (array (fin 8) ℕ)),
      (∀ i: fin 5, ∀ j: fin 8, matrix[i][j] = 1 ∨ matrix[i][j] = 3) ∧
      (∀ i: fin 5, (∑ j: fin 8, matrix[i][j]) % 7 = 0) ∧
      (∀ j: fin 8, (∑ i: fin 5, matrix[i][j]) % 7 = 0)) :=
sorry

end not_possible_1_and_3_arrangement_l438_438977


namespace problem1_problem2_l438_438728

theorem problem1 : ∃ (m : ℝ) (b : ℝ), ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 ∧ x - y + 4 = 0 →
  y = m * x + b ∧ (1 / m = -2) ∧ (y = - (2 * x + 2)) :=
sorry

theorem problem2 : ∀ (x y a : ℝ), (x = -1) ∧ (y = 3) → 
  (x + y = a) →
  a = 2 ∧ (x + y - 2 = 0) :=
sorry

end problem1_problem2_l438_438728


namespace questionnaire_visitors_l438_438398

theorem questionnaire_visitors
  (V : ℕ)
  (E U : ℕ)
  (h1 : ∀ v : ℕ, v ∈ { x : ℕ | x ≠ E ∧ x ≠ U } → v = 110)
  (h2 : E = U)
  (h3 : 3 * V = 4 * (E + U - 110))
  : V = 440 :=
by
  sorry

end questionnaire_visitors_l438_438398


namespace problem1_l438_438471

theorem problem1
  (x : ℝ)
  (h1 : sin 15 = x)
  (h2 : cos 15 = y)
  : (sqrt (1 - 2 * sin 15 * cos 15)) / (cos 15 - sqrt (1 - cos 165 ^ 2)) = 1 := by
  sorry

end problem1_l438_438471


namespace derivative_at_pi_div_3_l438_438215

noncomputable def f (f' : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 * f' (Real.pi / 3) + Real.sin x

theorem derivative_at_pi_div_3 (f' : ℝ → ℝ)
  (h : ∀ x, deriv (f f') x = f' x) :
  f' (Real.pi / 3) = 3 / (6 - 4 * Real.pi) :=
begin
  sorry
end

end derivative_at_pi_div_3_l438_438215


namespace distances_between_points_inequality_l438_438182

-- Define the problem
theorem distances_between_points_inequality {P : Fin 10 → ℝ × ℝ} :
  let distances := fun (i j : Fin 10) => Real.sqrt ((P i).1 - (P j).1) ^ 2 + ((P i).2 - (P j).2) ^ 2
  Max (Set.image (fun (pair : Fin 10 × Fin 10) => distances pair.1 pair.2) (Set.univ : Set (Fin 10 × Fin 10))) ≥
  2 * Min (Set.image (fun (pair : Fin 10 × Fin 10) => distances pair.1 pair.2) (Set.univ : Set (Fin 10 × Fin 10))) :=
begin
  sorry
end

end distances_between_points_inequality_l438_438182


namespace acute_triangle_sin_sum_gt_two_l438_438331

theorem acute_triangle_sin_sum_gt_two 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 0 < γ ∧ γ < π / 2) 
  (h4 : α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ > 2) :=
sorry

end acute_triangle_sin_sum_gt_two_l438_438331


namespace sum_pentagon_angles_l438_438423

theorem sum_pentagon_angles 
  (pentagon_in_circle : ∀ (α β : ℕ), α = 108 ∧ β = 144) :
  let int_angle := ∀ α = 108,
      supp_angle := ∀ β = 72,
      ext_inscribed_angle := ∀ γ = 144 
  in 5 * β + 5 * γ = 1080 := 
by
  sorry

end sum_pentagon_angles_l438_438423


namespace closest_point_6_2_4_4_l438_438832

open Real

def line_eq (x : ℝ) : ℝ := (3 * x - 1) / 4

def is_closest_point_on_line (a b x y : ℝ) (x0 y0 : ℝ) : Prop :=
  ((b - y0)^2 + (a - x0)^2) ≤ ((y - y0)^2 + (x - x0)^2)

noncomputable def point_on_line_closest (x₀ y₀ : ℝ) : ℝ × ℝ :=
  (6.2, 4.4)

theorem closest_point_6_2_4_4 :
  ∃ (x y : ℝ), (line_eq x = y) ∧ (is_closest_point_on_line x y 8 2 (6.2) (4.4)) :=
begin
  use [6.2, 4.4],
  split,
  { -- Proof that (6.2, 4.4) lies on the line y = (3x - 1) / 4
    calc line_eq 6.2 = (3 * 6.2 - 1) / 4 : by simp [line_eq]
                ... = 4.4 : by norm_num },
  { -- Proof that (6.2, 4.4) is the closest point to (8, 2)
    sorry
  }
end

end closest_point_6_2_4_4_l438_438832


namespace percent_markdown_l438_438810

theorem percent_markdown (P S : ℝ) (h : S * 1.25 = P) : (P - S) / P * 100 = 20 := by
  sorry

end percent_markdown_l438_438810


namespace perp_HB_HC_AO_l438_438506

variables (A B C O H_B H_C : Type)
[has_center O (circumcircle (triangle A B C))]
[is_feet_altitudes H_B B A C]
[is_feet_altitudes H_C C A B]

theorem perp_HB_HC_AO :
  is_perpendicular (line H_B H_C) (line A O) :=
sorry

end perp_HB_HC_AO_l438_438506


namespace math_problem_l438_438999

variable (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c < d)

theorem math_problem : a - c > b - d :=
by {
  sorry
}

end math_problem_l438_438999


namespace cos_15_degrees_l438_438115

theorem cos_15_degrees :
  cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degrees_l438_438115


namespace range_of_m_l438_438122

noncomputable def f (x m : ℝ) : ℝ := √3 * sin (π * x / m)

theorem range_of_m {x0 m : ℝ} (h : x0^2 + f x0 m ^2 < m^2) : m < -2 ∨ m > 2 :=
by
  sorry

end range_of_m_l438_438122


namespace cubic_polynomial_has_three_real_roots_l438_438371

open Polynomial

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry

axiom P_degree : degree P = 2
axiom Q_degree : degree Q = 3
axiom R_degree : degree R = 3
axiom PQR_relationship : ∀ x : ℝ, P.eval x ^ 2 + Q.eval x ^ 2 = R.eval x ^ 2

theorem cubic_polynomial_has_three_real_roots : 
  (∃ x : ℝ, Q.eval x = 0 ∧ ∃ y : ℝ, Q.eval y = 0 ∧ ∃ z : ℝ, Q.eval z = 0) ∨
  (∃ x : ℝ, R.eval x = 0 ∧ ∃ y : ℝ, R.eval y = 0 ∧ ∃ z : ℝ, R.eval z = 0) :=
sorry

end cubic_polynomial_has_three_real_roots_l438_438371


namespace sum_of_sequence_terms_l438_438211

variable {a : ℕ → ℝ}

noncomputable def arithmetic_seq (a : ℕ → ℝ) := ∃ (d c : ℝ), ∀ n, a n = c + n * d
noncomputable def roots (a : ℕ → ℝ) := a 3 * a 15 = -1 ∧ a 3 + a 15 = 6

theorem sum_of_sequence_terms (h_arith : arithmetic_seq a) (h_roots : roots a) :
    a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
begin
  sorry
end

end sum_of_sequence_terms_l438_438211


namespace ellipse_eccentricity_l438_438192

-- Define the conditions for the ellipse and properties
variables {a b c : ℝ}
variables (cond1 : a > b) (cond2 : b > 0) (cond3 : b = sqrt 3 * c) (cond4 : a^2 = b^2 + c^2)

-- Define a theorem to prove the eccentricity of the ellipse
theorem ellipse_eccentricity (cond1 : a > b) (cond2 : b > 0) (cond3 : b = sqrt 3 * c) (cond4 : a^2 = b^2 + c^2) :
  (c / a) = (1 / 2) :=
by
  sorry

end ellipse_eccentricity_l438_438192


namespace figure_E_has_smallest_surface_area_l438_438141

-- Conditions
def unit_cube_faces : ℕ := 6
def total_cubes : ℕ := 5
def reduction_per_join : ℕ := 2
def initial_surface_area : ℕ := total_cubes * unit_cube_faces
def joins_A_B_C_D : ℕ := 4
def joins_E : ℕ := 5

-- Surface area calculations based on joins
def surface_area_A_B_C_D : ℕ := initial_surface_area - (joins_A_B_C_D * reduction_per_join)
def surface_area_E : ℕ := initial_surface_area - (joins_E * reduction_per_join)

-- Proof statement
theorem figure_E_has_smallest_surface_area :
  surface_area_E < surface_area_A_B_C_D := 
by
  let h := 30 - 10 < 30 - 8
  assumption

end figure_E_has_smallest_surface_area_l438_438141


namespace find_g_neg_sqrt2_l438_438361

-- Define the piecewise function f(x)
def f (x : ℝ) (a b : ℝ) (g : ℝ → ℝ) : ℝ :=
  if x > 0 then
    x^2 + a * x - b
  else if x = 0 then
    0
  else
    g x

-- Given interval and the odd function condition
def is_odd_function_in_interval (a b : ℝ) (g : ℝ → ℝ) : Prop :=
  let I := (a + 4 / a, -b^2 + 4 * b) in
  ∀ x ∈ I, f (-x) a b g + f x a b g = 0

-- Specification of the problem
theorem find_g_neg_sqrt2 (a b : ℝ) (g : ℝ → ℝ)
  (h : is_odd_function_in_interval a b g)
  (h_interval : a + 4 / a = b^2 - 4 * b) (ha : a = -2) (hb : b = 2) :
  g (-real.sqrt 2) = 2 * real.sqrt 2 :=
sorry

end find_g_neg_sqrt2_l438_438361


namespace set_S_is_finite_l438_438722

-- Definitions for phi and tau functions
def phi (n : ℕ) : ℕ := (Finset.range n).filter (Nat.coprime n).card
def tau (n : ℕ) : ℕ := (Finset.range n).filter (λ d, n % d = 0).card

-- The main theorem statement
theorem set_S_is_finite : 
  { n : ℕ | φ n * τ n ≥ int_of_nat (sqrt (n^3 / 3)) }.finite :=
sorry

end set_S_is_finite_l438_438722


namespace extremum_value_l438_438904

noncomputable def f (x a : ℝ) : ℝ := (1 / 2) * x^2 - (a + 1) * x + a * Real.log x

theorem extremum_value (a : ℝ) : 
  (f' : ∀ x, x ≠ 0 → x - (a + 1) + a / x) →
  f' 2 = 0 →
  a = 2 :=
by
  intro f'
  sorry

end extremum_value_l438_438904


namespace problem_equivalent_l438_438626

def number_of_ways (balls : List String) (boxes : List String) (condition1 : ∀ box1 box2, balls.contains "red" → balls.contains "blue" → ¬(box1 = box2))
  (condition2 : ∃ box, ∀ ball ∈ balls, ball ∈ box) : Nat :=
  30

theorem problem_equivalent :
  number_of_ways ["red", "black", "yellow", "blue"] ["box1", "box2", "box3"]
    (λ box1 box2 hred hblue, 
      begin
        -- Condition 1: Red and blue balls cannot be in the same box.
        -- Here, box1 and box2 should not be the same for red and blue
        unfold condition1,
        sorry, -- skip detailed conditions proofs for simplicity
      end)
    (by
      -- Condition 2: Each box must contain at least one ball.
      -- This establishes the presence of each ball in the box.
      unfold condition2,
      sorry -- skip detailed conditions proofs for simplicity
    ) = 30 := 
begin
  sorry -- skip the definitive proof
end

end problem_equivalent_l438_438626


namespace max_arithmetic_progression_1996_max_arithmetic_progression_1997_l438_438300

-- Define the set S
def S := {n : ℕ | ∃ k, n = 1 / (k : ℝ)}

-- Define what it means to be an arithmetic progression in S of given length
def is_arithmetic_progression (a b : ℝ) (len : ℕ) :=
  ∀ n, n < len → a + n * b ∈ S

-- Prove the max length properties
theorem max_arithmetic_progression_1996 :
  ∃ a b, is_arithmetic_progression a b 1996 :=
sorry

theorem max_arithmetic_progression_1997 :
  ∃ a b, is_arithmetic_progression a b 1997 :=
sorry

end max_arithmetic_progression_1996_max_arithmetic_progression_1997_l438_438300


namespace circle_area_ratio_l438_438759

theorem circle_area_ratio (P : ℝ) :
  let l := 2 * (P / 6) in
  let w := P / 6 in
  let C := (5 * P^2 * Real.pi) / 144 in
  let D := (P^2 * Real.pi) / 27 in
  C / D = 15 / 16 :=
by
  sorry

end circle_area_ratio_l438_438759


namespace minimum_distance_l438_438526

theorem minimum_distance :
  ∃ (a : ℝ), (∀ (b : ℝ), b = 2^a → 
  ∀ (PQ : ℝ), PQ = sqrt ((a - b)^2 + (2^a - log 2 b)^2) → 
  ∀ (min_PQ : ℝ), min_PQ = PQ → min_PQ = (sqrt 2 * (1 + log(log 2))) / log 2) :=
by
  sorry

end minimum_distance_l438_438526


namespace vasya_cannot_have_more_shapes_l438_438990

-- Definition of the grid and the conditions involved
def square_size := 120
def area := square_size * square_size

-- Kolya and Vasya details
def number_of_shapes_kolya : ℕ := sorry -- The exact number used by Kolya is unspecified, thus we abstract it
def number_of_shapes_vasya := number_of_shapes_kolya + 5

-- Assuming each shape has cells differing by multiples of 3
def is_valid_figure (cells: ℕ) : Prop := cells % 3 = 0

def is_valid_cut(n: ℕ) : Prop :=
  ∃ shapes : list ℕ, (∀ s ∈ shapes, is_valid_figure s) ∧ list.sum shapes = area

theorem vasya_cannot_have_more_shapes :
  is_valid_cut number_of_shapes_kolya →
  ¬ is_valid_cut number_of_shapes_vasya :=
sorry

end vasya_cannot_have_more_shapes_l438_438990


namespace solve_eq_l438_438564

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq_l438_438564


namespace probability_of_coprime_l438_438058

open Finset
open Rat

noncomputable def set_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def pairs := (set_numbers.val.to_list.product set_numbers.val.to_list).filter (λ x, x.1 < x.2)

def gcd_is_one (x : ℕ × ℕ) : Prop := Nat.gcd x.1 x.2 = 1

def num_coprime_pairs : ℕ := (pairs.filter gcd_is_one).length

def num_total_pairs : ℕ := pairs.length

def probability_coprime_pairs : ℚ := num_coprime_pairs /. num_total_pairs

theorem probability_of_coprime :
  probability_coprime_pairs = 17/21 :=
sorry

end probability_of_coprime_l438_438058


namespace amplification_efficiency_l438_438334

theorem amplification_efficiency 
  (X0 X6 : ℝ) (p : ℝ)
  (h1 : X6 = 100 * X0)
  (log_10_1259 : log10 1.259 ≈ 2.154)
  (log_10_1778 : log10 1.778 ≈ 1.778) :
  p ≈ 1.154 :=
by sorry

end amplification_efficiency_l438_438334


namespace find_a_l438_438246

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) / Real.log a

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) 
  (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f a x ∧ f a x ≤ 1) : a = 2 :=
sorry

end find_a_l438_438246


namespace trains_crossing_time_l438_438690

theorem trains_crossing_time
  (L1 : ℕ) (L2 : ℕ) (T1 : ℕ) (T2 : ℕ)
  (H1 : L1 = 150) (H2 : L2 = 180)
  (H3 : T1 = 10) (H4 : T2 = 15) :
  (L1 + L2) / ((L1 / T1) + (L2 / T2)) = 330 / 27 := sorry

end trains_crossing_time_l438_438690


namespace simple_interest_l438_438429

/-- Given:
    - Principal (P) = Rs. 80325
    - Rate (R) = 1% per annum
    - Time (T) = 5 years
    Prove that the total simple interest earned (SI) is Rs. 4016.25.
-/
theorem simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
  (hP : P = 80325)
  (hR : R = 1)
  (hT : T = 5)
  (hSI : SI = P * R * T / 100) :
  SI = 4016.25 :=
by
  sorry

end simple_interest_l438_438429


namespace parallelogram_of_conditions_l438_438994

variables {A B C D : Point}
variables {AB CD AC BD : ℝ}

def is_parallelogram (A B C D : Point) := 
  (vector.from A B + vector.from C D = vector.from A C * √2) ∧ 
  (vector.from B C + vector.from D A = vector.from B D * √2)

theorem parallelogram_of_conditions
  (h1 : AB + CD = (AC * √2))
  (h2 : BC + DA = (BD * √2)) :
  is_parallelogram A B C D :=
sorry

end parallelogram_of_conditions_l438_438994


namespace required_run_rate_l438_438395

theorem required_run_rate (run_rate_first_10_overs : ℝ) (overs_first_part : ℕ) (target_runs : ℕ) (remaining_overs : ℕ) : 
  run_rate_first_10_overs = 3.2 →
  overs_first_part = 10 →
  target_runs = 292 →
  remaining_overs = 40 →
  let runs_scored_first_10_overs := run_rate_first_10_overs * overs_first_part in
  let runs_needed_remaining_overs := target_runs - runs_scored_first_10_overs in
  let required_run_rate_remaining_overs := runs_needed_remaining_overs / remaining_overs in
  required_run_rate_remaining_overs = 6.5 :=
by
  intros h₀ h₁ h₂ h₃
  rw [h₀, h₁, h₂, h₃]
  unfold let runs_scored_first_10_overs runs_needed_remaining_overs required_run_rate_remaining_overs sorry
  -- Proof is skipped with 'sorry'
  sorry

end required_run_rate_l438_438395


namespace michael_watermelon_weight_l438_438318

theorem michael_watermelon_weight (m c j : ℝ) (h1 : c = 3 * m) (h2 : j = c / 2) (h3 : j = 12) : m = 8 :=
by
  sorry

end michael_watermelon_weight_l438_438318


namespace part_I_part_II_l438_438264

open Real

variables {A B C a b c : ℝ}

-- Condition definitions
def is_obtuse_triangle (A B C : ℝ) := A > π / 2 ∧ B < π / 2 ∧ C < π / 2
def side_relation (a b : ℝ) (B : ℝ) := b = a * tan B

-- Given conditions
axiom obtuse_triangle : is_obtuse_triangle A B C
axiom side_eq : side_relation a b B

-- Part (I) statement
theorem part_I : A - B = π / 2 :=
sorry

-- Part (II) statement
theorem part_II : ∀ (B : ℝ), 0 < B ∧ B < π / 4 → (cos (2 * B) - sin A) ∈ Ioo (-sqrt 2 / 2) 0 :=
sorry

end part_I_part_II_l438_438264


namespace boxes_of_orange_crayons_l438_438685

theorem boxes_of_orange_crayons
  (n_orange_boxes : ℕ)
  (orange_crayons_per_box : ℕ := 8)
  (blue_boxes : ℕ := 7) (blue_crayons_per_box : ℕ := 5)
  (red_boxes : ℕ := 1) (red_crayons_per_box : ℕ := 11)
  (total_crayons : ℕ := 94)
  (h_total_crayons : (n_orange_boxes * orange_crayons_per_box) + (blue_boxes * blue_crayons_per_box) + (red_boxes * red_crayons_per_box) = total_crayons):
  n_orange_boxes = 6 := 
by sorry

end boxes_of_orange_crayons_l438_438685


namespace jessica_has_three_dozens_of_red_marbles_l438_438983

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l438_438983


namespace goods_train_passing_time_l438_438073

-- Definitions based on the conditions
def train_speed : ℝ := 60  -- Man's train speed in km/h
def goods_train_speed : ℝ := 30  -- Goods train speed in km/h
def goods_train_length : ℝ := 300  -- Goods train length in meters

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Relative speed in m/s
def relative_speed : ℝ := kmph_to_mps (train_speed + goods_train_speed)

-- Time it takes for the goods train to pass the man
def passing_time : ℝ := goods_train_length / relative_speed

-- Theorem stating the passing time is 12 seconds
theorem goods_train_passing_time : passing_time = 12 := by
  sorry

end goods_train_passing_time_l438_438073


namespace number_of_integer_solutions_l438_438478

theorem number_of_integer_solutions (h : ∀ n : ℤ, (2020 - n) ^ 2 / (2020 - n ^ 2) ≥ 0) :
  ∃! (m : ℤ), m = 90 := 
sorry

end number_of_integer_solutions_l438_438478


namespace find_x_l438_438403

variable (x : ℤ)

-- Define the conditions based on the problem
def adjacent_sum_condition := 
  (x + 15) + (x + 8) + (x - 7) = x

-- State the goal, which is to prove x = -8
theorem find_x : x = -8 :=
by
  have h : adjacent_sum_condition x := sorry
  sorry

end find_x_l438_438403


namespace sum_binom_eq_n_plus_one_l438_438333

theorem sum_binom_eq_n_plus_one (n : ℕ) :
  ∑ k in Finset.range (n + 1), (-1) ^ k * 2 ^ (2 * n - 2 * k) * Nat.choose (2 * n - k + 1) k = n + 1 :=
sorry

end sum_binom_eq_n_plus_one_l438_438333


namespace range_of_m_for_monotonic_f_l438_438899

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_for_monotonic_f :
  (∀ x ≥ 0, (Real.exp x - m : ℝ) ≥ 0) → m ≤ 1 := by
  sorry

end range_of_m_for_monotonic_f_l438_438899


namespace find_integer_pairs_l438_438149

theorem find_integer_pairs :
  {p : ℤ × ℤ | p.1 * (p.1 + 1) * (p.1 + 7) * (p.1 + 8) = p.2^2} =
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
sorry

end find_integer_pairs_l438_438149


namespace segment_lengths_equal_l438_438993

theorem segment_lengths_equal 
  (circle : Type)
  (A B : circle)
  (AB_diameter : diameter A B)
  (t1 : tangent circle A)
  (t2 : tangent circle B)
  (C : point_on_tangent circle t1 A)
  (D1 D2 E1 E2 : circle)
  (arcs_D1D2_E1E2 : arcs_through_point D1 D2 E1 E2 C) :
  segment_length (intersection t2 (line_through A D1)) (intersection t2 (line_through A D2)) = 
  segment_length (intersection t2 (line_through A E1)) (intersection t2 (line_through A E2)) :=
sorry

end segment_lengths_equal_l438_438993


namespace solution_set_of_inequality_l438_438674

theorem solution_set_of_inequality :
  {x : ℝ | (2 * x - 1) / (x + 1) ≤ 1} = set.Ioo (-1) 2 ∪ {2} :=
sorry

end solution_set_of_inequality_l438_438674


namespace extreme_value_expression_l438_438226

-- Define the function f
def f (x : Real) : Real := x * (Real.sin x)

-- Define the derivative of f
def f' (x : Real) : Real := Real.sin x + x * (Real.cos x)

-- Define the proof statement
theorem extreme_value_expression (x_0 : Real) (h : f' x_0 = 0) :
  (1 + x_0 ^ 2) * (1 + Real.cos (2 * x_0)) = 2 := sorry

end extreme_value_expression_l438_438226


namespace vertex_angle_of_isosceles_with_angle_30_l438_438889

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ a + b + c = 180

theorem vertex_angle_of_isosceles_with_angle_30 (a b c : ℝ) 
  (ha : isosceles_triangle a b c) 
  (h1 : a = 30 ∨ b = 30 ∨ c = 30) :
  (a = 30 ∨ b = 30 ∨ c = 30) ∨ (a = 120 ∨ b = 120 ∨ c = 120) := 
sorry

end vertex_angle_of_isosceles_with_angle_30_l438_438889


namespace correct_options_l438_438891

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function : Prop := ∀ x : ℝ, f x = f (-x)
def function_definition : Prop := ∀ x : ℝ, (0 < x) → f x = x^2 + x

-- Statements to be proved
def option_A : Prop := f (-1) = 2
def option_B_incorrect : Prop := ¬ (∀ x : ℝ, (f x ≥ f 0) ↔ x ≥ 0) -- Reformulated as not a correct statement
def option_C : Prop := ∀ x : ℝ, x < 0 → f x = x^2 - x
def option_D : Prop := ∀ x : ℝ, (0 < x ∧ x < 2) ↔ f (x - 1) < 2

-- Prove that the correct statements are A, C, and D
theorem correct_options (h_even : is_even_function f) (h_def : function_definition f) :
  option_A f ∧ option_C f ∧ option_D f := by
  sorry

end correct_options_l438_438891


namespace angle_RYS_is_105_l438_438967

/-- Proof problem: Given square WXYZ with side length 5, equilateral triangle WXF,
    intersection point R of XF and WY, and point S on YZ such that RS ⊥ YZ and RS = y,
    show that ∠RYS = 105°. --/
theorem angle_RYS_is_105
  (W X Y Z F R S : Type)
  [add_group X]
  [metric_space X]
  (WXYZ_is_square : metric.square W XYZ 5)
  (WXF_is_equilateral : metric.equilateral_triangle W XF 5)
  (intersect_R : metric.intersect_points XF WY R)
  (S_on_YZ : metric.on_segment YZ S)
  (RS_perpendicular_YZ : metric.perpendicular RS YZ)
  (RS_y : metric.length RS = y) :
  metric.angle R Y S = 105 :=
by
  sorry

end angle_RYS_is_105_l438_438967


namespace simplify_expr_l438_438446

-- Define the expression
def expr (a : ℝ) := 4 * a ^ 2 * (3 * a - 1)

-- State the theorem
theorem simplify_expr (a : ℝ) : expr a = 12 * a ^ 3 - 4 * a ^ 2 := 
by 
  sorry

end simplify_expr_l438_438446


namespace abs_value_solutions_l438_438664

theorem abs_value_solutions (x : ℝ) : abs x = 6.5 ↔ x = 6.5 ∨ x = -6.5 :=
by
  sorry

end abs_value_solutions_l438_438664


namespace bacon_suggestion_count_l438_438443

theorem bacon_suggestion_count (B : ℕ) : 
  (∃ B, 457 = B + 63) -> B = 394 :=
by
  intro h
  obtain ⟨B, h₁⟩ := h
  rw h₁
  sorry

end bacon_suggestion_count_l438_438443


namespace range_of_b_l438_438546

theorem range_of_b (M : Set (ℝ × ℝ)) (N : ℝ → ℝ → Set (ℝ × ℝ)) :
  (∀ m : ℝ, (∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ (N m b))) ↔ b ∈ Set.Icc (- Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by
  sorry

end range_of_b_l438_438546


namespace problem_1_problem_2_problem_3_l438_438049

-- Definition for \(a(m, n)\)
def a (m n : ℕ) : ℕ := m * (n + (m - 1) / 2)

-- Proof statements
theorem problem_1 : a 2 1 = 3 ∧ a 2 2 = 5 ∧ a 2 3 = 7 ∧ a 2 4 = 9 :=
by 
  sorry

theorem problem_2 (m n : ℕ) (h : m > 1) : a m n = m * (n + (m - 1) / 2) :=
by 
  sorry

theorem problem_3 :
  ∃ (m1 n1 : ℕ), 2000 = (finset.range (n1 + 32 - m1)).sum (λ i, i + 1 + m1) ∧
  ∃ (m2 n2 : ℕ), 2000 = (finset.range (n2 + 25 - m2)).sum (λ i, i + 1 + m2) ∧
  ∃ (m3 n3 : ℕ), 2000 = (finset.range (n3 + 5 - m3)).sum (λ i, i + 1 + m3) :=
by 
  sorry

end problem_1_problem_2_problem_3_l438_438049


namespace angle_ratio_l438_438267

theorem angle_ratio (BP BQ BM: ℝ) (ABC: ℝ) (quadrisect : BP = ABC/4 ∧ BQ = ABC)
  (bisect : BM = (3/4) * ABC / 2):
  (BM / (ABC / 4 + ABC / 4)) = 1 / 6 := by
    sorry

end angle_ratio_l438_438267


namespace product_coefficient_x4_l438_438787

noncomputable def p (x : ℝ) := 3 * x^5 + 4 * x^3 - 9 * x^2 + 2
noncomputable def q (x : ℝ) := 2 * x^3 - 5 * x + 1
noncomputable def r (x : ℝ) := p x * q x

theorem product_coefficient_x4 : 
  (∃ c : ℝ, (∑ i in (finset.range (11)), coeff ℝ (r) i = c) ∧ 
  c = -29) := 
sorry

end product_coefficient_x4_l438_438787


namespace A_is_infinite_l438_438888

variable {ℝ : Type*} [Real ℝ]

-- Defining the function f
variable (f : ℝ → ℝ)

-- Conditions
axiom f_defined : ∀ x : ℝ, ∃ y : ℝ, f x = y
axiom f_condition : ∀ x : ℝ, f (x) * f (x) ≤ 2 * x * x * f (x / 2)

-- Definition of the set A
def A : set ℝ := { a | f a > a * a }

-- Theorem to be proved
theorem A_is_infinite : (∃ a : ℝ, a ∈ A) → set.infinite A :=
sorry

end A_is_infinite_l438_438888


namespace cuboid_distance_to_plane_l438_438575

theorem cuboid_distance_to_plane (a b c m : ℝ) (h : m ≠ 0) :
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) → 
  (1 / m^2 = 1 / a^2 + 1 / b^2 + 1 / c^2) :=
begin
  sorry
end

end cuboid_distance_to_plane_l438_438575


namespace sequence_converges_and_limit_l438_438797

theorem sequence_converges_and_limit {a : ℝ} (m : ℕ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) :
  (∃ (x : ℕ → ℝ), 
  (x 1 = 1) ∧ 
  (x 2 = a) ∧ 
  (∀ n : ℕ, x (n + 2) = (x (n + 1) ^ m * x n) ^ (↑(1 : ℕ) / (m + 1))) ∧ 
  ∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n > N, |x n - l| < ε) ∧ l = a ^ (↑(m + 1) / ↑(m + 2))) :=
sorry

end sequence_converges_and_limit_l438_438797


namespace complement_union_l438_438234

namespace SetTheory

def U : Set ℕ := {1, 3, 5, 9}
def A : Set ℕ := {1, 3, 9}
def B : Set ℕ := {1, 9}

theorem complement_union (U A B: Set ℕ) (hU : U = {1, 3, 5, 9}) (hA : A = {1, 3, 9}) (hB : B = {1, 9}) :
  U \ (A ∪ B) = {5} :=
by
  sorry

end SetTheory

end complement_union_l438_438234


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l438_438821

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l438_438821


namespace largest_visits_is_four_l438_438399

noncomputable def largest_num_visits (stores people visits : ℕ) (eight_people_two_stores : ℕ) 
  (one_person_min : ℕ) : ℕ := 4 -- This represents the largest number of stores anyone could have visited.

theorem largest_visits_is_four 
  (stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ) 
  (eight_people_two_stores : ℕ) (each_one_store : ℕ) 
  (H1 : stores = 8) 
  (H2 : total_visits = 23) 
  (H3 : people_shopping = 12) 
  (H4 : eight_people_two_stores = 8)
  (H5 : each_one_store = 1) :
  largest_num_visits stores people_shopping total_visits eight_people_two_stores each_one_store = 4 :=
by
  sorry

end largest_visits_is_four_l438_438399


namespace domain_h_h_odd_set_x_f_gt_1_l438_438542

noncomputable def f (a x : ℝ) : ℝ := log a (1 + x)
noncomputable def g (a x : ℝ) : ℝ := log a (1 - x)
noncomputable def h (a x : ℝ) : ℝ := f a x - g a x

-- Given conditions
variables {a : ℝ} (ha1 : a > 0) (ha2 : a ≠ 1)

-- Domain of h(x) is (-1, 1)
theorem domain_h : ∀ x, -1 < x ∧ x < 1 ↔ (-1 < x) ∧ (x < 1) :=
by sorry

-- h(x) is an odd function
theorem h_odd (x : ℝ) : h a (-x) = -h a x :=
by sorry

-- Set of x for which f(x) > 1 when a = log_3 27 + log_1/2 2
noncomputable def a_value := log 3 27 + log (1/2) 2

theorem set_x_f_gt_1 (x : ℝ) (ha : a = a_value) : f a x > 1 ↔ x > 4 :=
by sorry

end domain_h_h_odd_set_x_f_gt_1_l438_438542


namespace ivanov_family_net_worth_l438_438717

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l438_438717


namespace cards_not_divisible_by_7_l438_438015

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438015


namespace selling_price_of_radio_l438_438355

theorem selling_price_of_radio
  (cost_price : ℝ)
  (loss_percentage : ℝ) :
  loss_percentage = 13 → cost_price = 1500 → 
  (cost_price - (loss_percentage / 100) * cost_price) = 1305 :=
by
  intros h1 h2
  sorry

end selling_price_of_radio_l438_438355


namespace no_adjacent_men_women_l438_438407

def men : ℕ := 2
def women : ℕ := 2

theorem no_adjacent_men_women :
  ∃ (arrangements : ℕ), 
  (arrangements = 2 * 2 * 2 * 2 ∧ arrangements = 8) :=
by {
  let patterns := 2,
  let men_arrangements := men.factorial,
  let women_arrangements := women.factorial,
  have total := patterns * men_arrangements * women_arrangements,
  use total,
  split,
  -- number of ways to arrange given patterns
  exact total,
  -- actual arithmetic calculation verification
  exact 8,
  sorry  -- skipping detailed proof parts
}

end no_adjacent_men_women_l438_438407


namespace bread_last_days_is_3_l438_438681

-- Define conditions
def num_members : ℕ := 4
def slices_breakfast : ℕ := 3
def slices_snacks : ℕ := 2
def slices_loaf : ℕ := 12
def num_loaves : ℕ := 5

-- Define the problem statement
def bread_last_days : ℕ :=
  (num_loaves * slices_loaf) / (num_members * (slices_breakfast + slices_snacks))

-- State the theorem to be proved
theorem bread_last_days_is_3 : bread_last_days = 3 :=
  sorry

end bread_last_days_is_3_l438_438681


namespace solve_system_of_equations_l438_438642

theorem solve_system_of_equations (x y z t : ℝ) :
  xy - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18 ↔ (x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∨ t = 0) :=
sorry

end solve_system_of_equations_l438_438642


namespace set_equality_sum_l438_438517

-- Assuming x and y are natural numbers
variables (x y : ℕ)

-- Defining the sets A and B
def A : set ℕ := {2, y}
def B : set ℕ := {x, 3}

-- The theorem to be proved
theorem set_equality_sum (h : A = B) : x + y = 5 :=
by {
  sorry
}

end set_equality_sum_l438_438517


namespace collinear_X_M_H_l438_438276

variables (A B C E F H M X : Type)
variables [hilbert_triangle A B C]
variables [altitude BE : A E = ⊥ A C]
variables [altitude CF : B F = ⊥ B A]
variables [midpoint M : midpoint B C = M]
variables [intersection_tangents X : tangents_intersection (incircle_center B M F) (incircle_center C M E) = X]

theorem collinear_X_M_H : collinear X M H :=
sorry

end collinear_X_M_H_l438_438276


namespace complement_relative_l438_438915

open Set

variable (A B C : Set ℕ)

theorem complement_relative (hA : A = {1, 2, 3, 4}) (hB : B = {1, 3}) : 
  C = A \ B → C = {2, 4} :=
by
  intros hC
  rw [hA, hB] at hC
  exact hC

end complement_relative_l438_438915


namespace no_valid_odd_numbers_cube_l438_438279

theorem no_valid_odd_numbers_cube : 
  ∀ (numbers : Fin 8 → ℕ), 
  (∀ i, numbers i % 2 = 1) ∧ 
  (∀ i, 1 ≤ numbers i ∧ numbers i ≤ 600) → 
  (∀ u v : Fin 8, adjacent vertices u v → common_divisor (numbers u) (numbers v) > 1) → 
  ¬ (∃ u v : Fin 8, ¬adjacent vertices u v ∧ common_divisor (numbers u) (numbers v) > 1) :=
sorry

end no_valid_odd_numbers_cube_l438_438279


namespace power_function_expression_l438_438568

theorem power_function_expression (f : ℝ → ℝ) (α : ℝ) (h : ∀ x : ℝ, f x = x ^ α) (h_point : f 25 = 5) :
  f = λ x, x ^ (1 / 2) :=
by
  sorry

end power_function_expression_l438_438568


namespace number_of_lines_l438_438270

theorem number_of_lines (A B C : ∀ {α : Type} [Field α], Point α)
  (l : ∀ {α : Type} [Field α], Line α)
  (h1 : ∀ {α : Type} [Field α], distance A l = distance B l ∨ distance B l = 2 * distance A l)
  (h2 : ∀ {α : Type} [Field α], distance A l = 2 * distance C l ∨ distance B l = distance C l) :
  count_lines_with_ratios A B C l = 12 := 
sorry

end number_of_lines_l438_438270


namespace area_H1H2H3_l438_438602

-- Define the points Q, D, E, F
variables (Q D E F : Point)
-- Define the centroids H1, H2, H3
variables (H1 : centroid Q E F) (H2 : centroid Q F D) (H3 : centroid Q D E)
-- Define the area of triangle DEF
variable (area_DEF : ℝ)
-- Assume the area of triangle DEF is 24
axiom h_area_DEF : area_DEF = 24

-- The theorem to prove
theorem area_H1H2H3 (Q D E F : Point) 
  (H1 : centroid Q E F) (H2 : centroid Q F D) (H3 : centroid Q D E) 
  (area_DEF : ℝ) (h_area_DEF : area_DEF = 24) : 
  area_H1H2H3 = 8 :=
sorry

end area_H1H2H3_l438_438602


namespace cos_double_angle_l438_438886

variable (α : ℝ)

theorem cos_double_angle (h1 : 0 < α ∧ α < π / 2) 
                         (h2 : Real.cos ( α + π / 4) = 3 / 5) : 
    Real.cos (2 * α) = 24 / 25 :=
by
  sorry

end cos_double_angle_l438_438886


namespace ellipse_standard_eq_max_area_line_eq_l438_438193

-- Definition of ellipse with given conditions
def ellipse_eq (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Given conditions
variables (a b c : ℝ)
variables (h₀ : a > b > 0)
variables (h₁ : |a| = sqrt 2)
variables (h₂ : b * c = 1)
variables (h₃ : a^2 = b^2 + c^2)

-- Equation of ellipse given these conditions
theorem ellipse_standard_eq (x y : ℝ) :
  ellipse_eq x y (sqrt 2) 1 :=
by sorry

-- Proof problem 2 specific conditions
variables (M N : ℝ × ℝ)
variables (h₄ : M.1^2 / 2 + M.2^2 = 1)
variables (h₅ : N.1^2 / 2 + N.2^2 = 1)
variables (h₆ : (|M.1 - 0|^2 + (M.2 - 1)^2) + 
                (|N.1 - 0|^2 + (N.2 - 1)^2) = 
                ((M.1 - N.1)^2 + (M.2 - N.2)^2))

-- Maximum area condition leads to the line equation
theorem max_area_line_eq :
  ∃ k m : ℝ, m = -1/3 ∧ (∀ x y, y = k*x + m) :=
by sorry

end ellipse_standard_eq_max_area_line_eq_l438_438193


namespace swimming_pool_volume_l438_438430

theorem swimming_pool_volume :
  ∀ (width length shallow_depth deep_depth : ℝ),
  (width = 9) →
  (length = 12) →
  (shallow_depth = 1) →
  (deep_depth = 4) →
  (∃ (volume : ℝ), volume = 270) :=
by
  intros width length shallow_depth deep_depth width_eq length_eq shallow_depth_eq deep_depth_eq
  use 270
  -- sorry: the actual proof
  sorry

end swimming_pool_volume_l438_438430


namespace number_of_divisors_of_3003_l438_438830

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

theorem number_of_divisors_of_3003 : count_divisors 3003 = 16 :=
by
  sorry

end number_of_divisors_of_3003_l438_438830


namespace jimmy_matchbooks_l438_438592

def num_matchbooks_jimmy_had (initial_stamps_Tonya : ℕ) (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (remaining_stamps_Tonya: ℕ) : ℕ :=
  (initial_stamps_Tonya - remaining_stamps_Tonya) * stamps_per_match / matches_per_matchbook

theorem jimmy_matchbooks (initial_stamps_Tonya : ℕ) (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (remaining_stamps_Tonya: ℕ)
  (h_initial : initial_stamps_Tonya = 13)
  (h_value : stamps_per_match = 12)
  (h_matchbook : matches_per_matchbook = 24)
  (h_remaining : remaining_stamps_Tonya = 3) :
  num_matchbooks_jimmy_had initial_stamps_Tonya stamps_per_match matches_per_matchbook remaining_stamps_Tonya = 5 :=
by
  rw [h_initial, h_value, h_matchbook, h_remaining]
  simp [num_matchbooks_jimmy_had]
  sorry

end jimmy_matchbooks_l438_438592


namespace prod_eq_25834_l438_438114

theorem prod_eq_25834 : (∏ n in Finset.range 25, (n + 5) / (n + 1)) = 25834 := by
  sorry

end prod_eq_25834_l438_438114


namespace A_inter_B_l438_438232

variable (M : Set ℕ) (A B : Set ℤ)

def set_M : Set ℕ := {0, 1, 2}
def set_A : Set ℤ := {x | ∃ y, y = 2 * x ∧ x ∈ set_M}
def set_B : Set ℤ := {y | ∃ x, y = 2 * x - 2 ∧ x ∈ set_M}

theorem A_inter_B : set_A ∩ set_B = {0, 2} := by
  sorry

end A_inter_B_l438_438232


namespace cube_root_sum_l438_438106

theorem cube_root_sum:
  let x := (Real.cbrt (7 + 2 * Real.sqrt 21) + Real.cbrt (7 - 2 * Real.sqrt 21))
  in x = 1 :=
by
  -- let x definition
  let x := (Real.cbrt (7 + 2 * Real.sqrt 21) + Real.cbrt (7 - 2 * Real.sqrt 21))
  -- state result that needs to be proven
  show x = 1,
  sorry

end cube_root_sum_l438_438106


namespace magnitude_of_b_l438_438550

variable (a b : ℝ)

-- Defining the given conditions as hypotheses
def condition1 : Prop := (a - b) * (a - b) = 9
def condition2 : Prop := (a + 2 * b) * (a + 2 * b) = 36
def condition3 : Prop := a^2 + (a * b) - 2 * b^2 = -9

-- Defining the theorem to prove
theorem magnitude_of_b (ha : condition1 a b) (hb : condition2 a b) (hc : condition3 a b) : b^2 = 3 := 
sorry

end magnitude_of_b_l438_438550


namespace mod_inverse_sum_l438_438444

theorem mod_inverse_sum :
  ∃ a b : ℤ, (5 * a ≡ 1 [MOD 35]) ∧ (15 * b ≡ 1 [MOD 35]) ∧ ((a + b) % 35 = 21) :=
by
  sorry

end mod_inverse_sum_l438_438444


namespace express_in_standard_form_l438_438143

theorem express_in_standard_form (x : ℝ) : x^2 - 6 * x = (x - 3)^2 - 9 :=
by
  sorry

end express_in_standard_form_l438_438143


namespace circle_and_tangent_lines_l438_438185

-- Definitions of points, line, and circle conditions 
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def point_A : Point := {x := 0, y := -6}
def point_B : Point := {x := 1, y := -5}
def line_l : Line := {a := 1, b := -1, c := 1}

def is_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def circle_equation (a b r : ℝ) (p : Point) : Prop :=
  (p.x - a) ^ 2 + (p.y - b) ^ 2 = r

-- Problem statement to prove the circle's equation and tangent lines equations.
theorem circle_and_tangent_lines :
  ∃ a b r, circle_equation a b r point_A ∧
         circle_equation a b r point_B ∧
         is_on_line ⟨a, b⟩ line_l ∧
         ((a, b, r) = (-3, -2, 5) ∧
          ∃ k, (3 * 2 - 4 * 8 + 26 = 0) ∨
          ∃ t, (k ≠ t)) :=
by
  sorry

end circle_and_tangent_lines_l438_438185


namespace abc_is_not_necessarily_a_square_l438_438048

theorem abc_is_not_necessarily_a_square 
  (A B C D : Type)
  [ConvexQuadrilateral A B C D]
  (h1 : Parallel A B C D)
  (h2 : ∠DAC = ∠ABD)
  (h3 : ∠CAB = ∠DBC) : 
  ¬(Square A B C D) := 
sorry

end abc_is_not_necessarily_a_square_l438_438048


namespace card_arrangement_impossible_l438_438038

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438038


namespace cards_not_divisible_by_7_l438_438012

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438012


namespace knights_probability_l438_438688

theorem knights_probability (total_knights : ℕ) (chosen_knights : ℕ) (P_num : ℚ) (P_den : ℚ)
  (h1 : total_knights = 30) 
  (h2 : chosen_knights = 4)
  (h3 : P_num = 541)
  (h4 : P_den = 609) :
  P_num + P_den = 1150 := 
by {
  -- Total ways to choose 4 knights from 30
  have H1 : Nat.choose total_knights chosen_knights = 27405,
  sorry,
  
  -- Ways to choose 4 knights with no neighbors
  have H2 : Nat.choose (total_knights - 3 * chosen_knights) chosen_knights = 3060,
  sorry,

  -- Calculate the probability P as a fraction
  have H3 : P_num = 541,
  sorry,

  have H4 : P_den = 609,
  sorry,

  -- Prove that the sum of P_num and P_den is 1150
  sorry
}

end knights_probability_l438_438688


namespace limit_tan_tan2x_limit_ln_1_x_limit_x_6_1_2lnx_limit_x_m_x2_1_l438_438476

-- Statement for the first limit problem
theorem limit_tan_tan2x : 
  tendsto (λ x : ℝ, (Real.tan x) ^ (Real.tan (2 * x))) (nhds (π / 4)) (nhds (1 / Real.exp 1)) :=
sorry

-- Statement for the second limit problem
theorem limit_ln_1_x : 
  tendsto (λ x : ℝ, (Real.log x) ^ (1 / x)) at_top (nhds 1) :=
sorry

-- Statement for the third limit problem
theorem limit_x_6_1_2lnx : 
  tendsto (λ x : ℝ, x ^ (6 / (1 + 2 * Real.log x))) (nhds_within 0 (Set.Ioi 0)) (nhds (Real.exp 3)) :=
sorry

-- Statement for the fourth limit problem
theorem limit_x_m_x2_1 (m : ℝ) : 
  tendsto (λ x : ℝ, x ^ (m / (x^2 - 1))) (nhds 1) (nhds (Real.exp (m / 2))) :=
sorry

end limit_tan_tan2x_limit_ln_1_x_limit_x_6_1_2lnx_limit_x_m_x2_1_l438_438476


namespace intersection_A_B_l438_438516

def Set := Set (ℝ × ℝ)

def A : Set := {p | ∃ x y : ℝ, p = (x, y) ∧ y = 3 * x - 2}
def B : Set := {p | ∃ x y : ℝ, p = (x, y) ∧ y = x}

theorem intersection_A_B : A ∩ B = {(1, 1)} := 
by
  sorry

end intersection_A_B_l438_438516


namespace profit_percentage_l438_438097

theorem profit_percentage (SP CP : ℝ) (h_SP : SP = 150) (h_CP : CP = 120) : 
  ((SP - CP) / CP) * 100 = 25 :=
by {
  sorry
}

end profit_percentage_l438_438097


namespace eccentricity_of_ellipse_l438_438874

-- Definitions related to the ellipse and the problem conditions
variable (a b : ℝ) (h₀ : a > b > 0)

-- Ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Coordinates of the foci
def F1 : ℝ × ℝ := (-real.sqrt (a^2 - b^2), 0)
def F2 : ℝ × ℝ := (real.sqrt (a^2 - b^2), 0)

-- Assumptions about the point P and the intersection with the y-axis
variable (P Q : ℝ × ℝ)
variable (hP : ellipse a b P.1 P.2)
variable (hQ : Q = (0, P.2 * real.sqrt(a^2 - b^2) / (P.1 + real.sqrt(a^2 - b^2))))

-- Distance conditions: |PQ| = 2|QF1| and isosceles triangle for PF1F2
def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
variable (h_dist : dist Q P = 2 * dist Q F1)

-- Isosceles triangle condition
variable (h_iso : dist P F1 = dist P F2)

-- Conclusion for eccentricity e
theorem eccentricity_of_ellipse : 
  let e := (real.sqrt (a^2 - b^2))/a in
  e = (real.sqrt 3 - 1) / 2 :=
sorry

end eccentricity_of_ellipse_l438_438874


namespace probability_neither_prime_nor_composite_l438_438945

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n
def neither_prime_nor_composite (n : ℕ) : Prop := ¬is_prime n ∧ ¬is_composite n

theorem probability_neither_prime_nor_composite :
  let count := (finset.range 998).filter neither_prime_nor_composite in
  count.card = 1 ∧ 
  (1 : ℚ) / 997 = (1 : ℚ) / (finset.range 997).card :=
by sorry

end probability_neither_prime_nor_composite_l438_438945


namespace mr_william_land_percentage_l438_438473

/--
Given:
1. Farm tax is levied on 90% of the cultivated land.
2. The tax department collected a total of $3840 through the farm tax from the village.
3. Mr. William paid $480 as farm tax.

Prove: The percentage of total land of Mr. William over the total taxable land of the village is 12.5%.
-/
theorem mr_william_land_percentage (T W : ℝ) 
  (h1 : 0.9 * W = 480) 
  (h2 : 0.9 * T = 3840) : 
  (W / T) * 100 = 12.5 :=
by
  sorry

end mr_william_land_percentage_l438_438473


namespace sum_of_10th_row_l438_438782

theorem sum_of_10th_row : ∑ n in finset.range (100 - 82 + 1), (82 + n) = 1729 := 
by sorry

end sum_of_10th_row_l438_438782


namespace simplify_eq_neg_one_l438_438306

variable (a b c : ℝ)

noncomputable def simplify_expression := 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_eq_neg_one 
  (a_ne_zero : a ≠ 0) 
  (b_ne_zero : b ≠ 0) 
  (c_ne_zero : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) 
  : simplify_expression a b c = -1 :=
by sorry

end simplify_eq_neg_one_l438_438306


namespace sum_coordinates_point_C_l438_438328

/-
Let point A = (0,0), point B is on the line y = 6, and the slope of AB is 3/4.
Point C lies on the y-axis with a slope of 1/2 from B to C.
We need to prove that the sum of the coordinates of point C is 2.
-/
theorem sum_coordinates_point_C : 
  ∃ (A B C : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B.2 = 6 ∧ 
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 ∧ 
  C.1 = 0 ∧ 
  (C.2 - B.2) / (C.1 - B.1) = 1 / 2 ∧ 
  C.1 + C.2 = 2 :=
by
  sorry

end sum_coordinates_point_C_l438_438328


namespace arithmetic_series_sum_proof_middle_term_proof_l438_438849

def arithmetic_series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def middle_term (a l : ℤ) : ℤ :=
  (a + l) / 2

theorem arithmetic_series_sum_proof :
  let a := -51
  let d := 2
  let n := 27
  let l := 1
  arithmetic_series_sum a d n = -675 :=
by
  sorry

theorem middle_term_proof :
  let a := -51
  let l := 1
  middle_term a l = -25 :=
by
  sorry

end arithmetic_series_sum_proof_middle_term_proof_l438_438849


namespace present_cost_after_discount_l438_438400

theorem present_cost_after_discount 
  (X : ℝ) (P : ℝ) 
  (h1 : X - 4 = (0.80 * P) / 3) 
  (h2 : P = 3 * X)
  :
  0.80 * P = 48 :=
by
  sorry

end present_cost_after_discount_l438_438400


namespace find_smaller_number_l438_438165

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 45) (h2 : b = 4 * a) : a = 9 :=
by
  sorry

end find_smaller_number_l438_438165


namespace arctan_sum_eq_pi_div_four_l438_438972

variable (a b c k : ℝ)
variable (h1 : a^2 + b^2 = c^2)
variable (h2 : k ≠ 0)

theorem arctan_sum_eq_pi_div_four (hC : ∠C = π/2) :
  arctan (a / (b + c + k)) + arctan (b / (a + c + k)) = π / 4 := sorry

end arctan_sum_eq_pi_div_four_l438_438972


namespace problem_solution_l438_438522

variable (α : Real)
variable h : tan α + cot α = 4

theorem problem_solution : sqrt (sec α ^ 2 + csc α ^ 2 - (1 / 2) * sec α * csc α) = sqrt 14 :=
  sorry

end problem_solution_l438_438522


namespace polynomial_not_obtainable_l438_438278

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem polynomial_not_obtainable (n : ℕ) :
  ¬ ∃ (p : ℝ → ℝ), (p = x^n - 1) ∧
  (∃ (k : ℕ) (ops : list ((ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ))),
    (hd : ℝ → ℝ) (tl : list (ℝ → ℝ)),
    tl = (f :: g :: list.repeat (λx, 0) k) ∧
    hd = list.foldl (λ acc op, op acc hd) f ops ∧
    list.mem' hd list.nil = p) :=
sorry

end polynomial_not_obtainable_l438_438278


namespace quadratic_vertex_symmetry_l438_438465

theorem quadratic_vertex_symmetry (b c : ℝ) 
  (h1 : 1 + b + c = 0) 
  (h2 : b = -4) :
  ¬ (∃ x y : ℝ, (x, y) = (2, -2) ∧ (x, y) = ((-(b / 2)), c - ((b ^ 2) / 4))) := 
  by 
    -- Vertex computation of quadratic function
    let p := -(b / 2)
    let q := c - ((b ^ 2) / 4)
    -- Given conditions
    have h3 : p = 2 := by linarith
    have h4 : q = -1 := by linarith
    -- so, vertex should be (2, -1)
    sorry

end quadratic_vertex_symmetry_l438_438465


namespace mul_mixed_number_eq_l438_438726

theorem mul_mixed_number_eq :
  99 + 24 / 25 * -5 = -499 - 4 / 5 :=
by
  sorry

end mul_mixed_number_eq_l438_438726


namespace schedule_options_l438_438397

-- Define the conditions as variables and assumptions
variable {d1 d2 d3 d4 d5 : ℕ}

-- No consecutive days
axiom non_consecutive : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 3 → (nat.succ i) ≤ (d i) -> (d (nat.add i 1) ≥ d i + 2)

-- Fifth day at least four days after the fourth day
axiom fifth_day_rule : d5 ≥ d4 + 4

-- Total number of days in July
axiom total_days_in_july : d5 ≤ 31

-- Transformations to new variables
def x1 := d1 - 1
def x2 := d2 - d1 - 1
def x3 := d3 - d2 - 1
def x4 := d4 - d3 - 1
def x5 := d5 - d4 - 4

-- Combined equation from the transformations
axiom combined_equation : x1 + x2 + x3 + x4 + x5 = 21

-- The theorem to prove the number of valid schedule options
theorem schedule_options : ∑ i in range 26, (choose i 5) = 12650 := by sorry

end schedule_options_l438_438397


namespace no_even_perfect_squares_infinite_perfect_squares_set_l438_438043

theorem no_even_perfect_squares (a b c : ℕ) :
  ¬ (∃ k m n : ℕ, ab + 1 = 4 * k^2 ∧ bc + 1 = 4 * m^2 ∧ ca + 1 = 4 * n^2) := sorry

theorem infinite_perfect_squares_set : 
  ∃ (a b c d n : ℕ), ∀ n > 2, 
  distinct [a (n-2), b n, c (n+2), d 0] ∧
  (ab + 1 = (n - 1)^2) ∧
  (bc + 1 = (n + 1)^2) ∧ 
  (cd + 1 = 1^2) ∧ (da + 1 = 1^2) := sorry

end no_even_perfect_squares_infinite_perfect_squares_set_l438_438043


namespace find_a_l438_438179

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l438_438179


namespace find_totally_damaged_cartons_l438_438313

def jarsPerCarton : ℕ := 20
def initialCartons : ℕ := 50
def reducedCartons : ℕ := 30
def damagedJarsPerCarton : ℕ := 3
def damagedCartons : ℕ := 5
def totalGoodJars : ℕ := 565

theorem find_totally_damaged_cartons :
  (initialCartons * jarsPerCarton - ((initialCartons - reducedCartons) * jarsPerCarton + damagedJarsPerCarton * damagedCartons - totalGoodJars)) / jarsPerCarton = 1 := by
  sorry

end find_totally_damaged_cartons_l438_438313


namespace inequality_system_correctness_l438_438856

theorem inequality_system_correctness :
  (∀ (x a b : ℝ), 
    (x - a ≥ 1) ∧ (x - b < 2) →
    ((∀ x, -1 ≤ x ∧ x < 3 → (a = -2 ∧ b = 1)) ∧
     (a = b → (a + 1 ≤ x ∧ x < a + 2)) ∧
     (¬(∃ x, a + 1 ≤ x ∧ x < b + 2) → a > b + 1) ∧
     ((∃ n : ℤ, n < 0 ∧ n ≥ -6 - a ∧ n ≥ -5) → -7 < a ∧ a ≤ -6))) :=
sorry

end inequality_system_correctness_l438_438856


namespace number_of_divisors_of_3003_l438_438827

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l438_438827


namespace prove_concyclic_points_l438_438613

variables {A B C D A1 B1 A2 B2 : Type*}

-- Define a trapezoid with AB parallel to CD
def trapezoid (A B C D : Type*) : Prop :=
∃ (line1 line2 : set Type*), (A ∈ line1 ∧ B ∈ line1 ∧ C ∈ line2 ∧ D ∈ line2 ∧ (line1 ∥ line2))

-- Define concyclic points
def concyclic (X Y Z W : Type*) : Prop :=
∃ Ω : set Type*, X ∈ Ω ∧ Y ∈ Ω ∧ Z ∈ Ω ∧ W ∈ Ω ∧ is_circle Ω

-- Define a circle passing through points
def passes_through (ω : set Type*) (X Y : Type*) : Prop :=
X ∈ ω ∧ Y ∈ ω ∧ is_circle ω

-- Define the reflection across a midpoint
def reflection_across_midpoint {P Q R : Type*} (mid : Type*) : Prop :=
∃ M : Type*, (M = midpoint P Q) ∧ R = reflection P M

noncomputable theory

theorem prove_concyclic_points 
  (h_trapezoid : trapezoid A B C D)
  (h_concyclic_ABCD : concyclic A B C D)
  (Ω : set Type*) (h_circumcircle : concyclic Ω A B C D)
  (ω : set Type*) (h_circle_passing : passes_through ω C D)
  (h_A1 : A1 ∈ intersection ω (line_through C A))
  (h_B1 : B1 ∈ intersection ω (line_through C B))
  (mid_AC : Type*) (h_mid_AC : mid_AC = midpoint C A)
  (mid_BC : Type*) (h_mid_BC : mid_BC = midpoint C B)
  (h_A2 : reflection_across_midpoint mid_AC A1 A2)
  (h_B2 : reflection_across_midpoint mid_BC B1 B2) :
  concyclic A B A2 B2 :=
sorry

end prove_concyclic_points_l438_438613


namespace find_modulus_l438_438497

-- Define the complex numbers
variables (z : ℂ)

-- Given condition
def condition := (z - 2*complex.I) * (1 - complex.I) = -2

-- The theorem to prove
theorem find_modulus (h : condition z) : complex.abs z = real.sqrt 2 := 
sorry

end find_modulus_l438_438497


namespace omicron_monograms_count_l438_438320

/-- 
In how many ways can monograms be formed with the initials 'O' (first), and two distinct lowercase letters 
from 'a' to 'm' (middle and last), in alphabetical order? 
The number of ways is 78.
-/
theorem omicron_monograms_count : 
  let alphabet := ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'] in 
  let num_combinations := (alphabet.length.choose 2) in
  num_combinations = 78 := by
{
  -- Define the size of the subset
  let n := 13
  -- Choose 2 out of 13
  let k := 2
  -- Compute the combination
  have h : (n.choose k) = 78 := by sorry
  exact h
}

end omicron_monograms_count_l438_438320


namespace quadratic_minimum_value_proof_l438_438172

-- Define the quadratic function and its properties
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the condition that the coefficient of the squared term is positive
def coefficient_positive : Prop := (2 : ℝ) > 0

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : Prop := h = 3

-- Define the minimum value of the quadratic function
def minimum_value (y_min : ℝ) : Prop := ∀ x : ℝ, y_min ≤ quadratic_function x 

-- Define the correct answer choice
def correct_answer : Prop := minimum_value 2

-- The theorem stating the proof problem
theorem quadratic_minimum_value_proof :
  coefficient_positive ∧ axis_of_symmetry 3 → correct_answer :=
sorry

end quadratic_minimum_value_proof_l438_438172


namespace T_n_plus_1_sum_T_inverse_l438_438052

-- Definitions directly from the problem conditions
def a (n : ℕ) : ℕ :=
  if odd n then n
  else a (n / 2)

def T (n : ℕ) : ℕ :=
  (Finset.range (2 * n + 1)).filter (λ x, x ≠ 0).sum a

-- Statements to be proven
theorem T_n_plus_1 (n : ℕ) : 
  T (n + 1) = 4^n + T n :=
sorry

theorem sum_T_inverse (n : ℕ) :
  ∑ k in Finset.range (n + 1), 1 / (T (k + 1) : ℝ) < 1 :=
sorry

end T_n_plus_1_sum_T_inverse_l438_438052


namespace domain_of_g_l438_438539

noncomputable def is_even_function (f : ℝ → ℝ) := ∀ x, f(x) = f(-x)
noncomputable def g (x : ℝ) (a : ℝ) := if h : log a x - 1 ≥ 0 then sqrt (log a x - 1) else 0

theorem domain_of_g (a : ℝ) (b : ℝ) (ha : a = 1/2) 
  (even_f : is_even_function (λ x, x^2 + (2*a - 1)*x + b)) : 
  {x : ℝ | 0 < x ∧ x ≤ 1/2} ⊆ {x : ℝ | log a x - 1 ≥ 0} :=
sorry

end domain_of_g_l438_438539


namespace C1_standard_eq_C2_cartesian_eq_AB_distance_l438_438911

-- Define the parametric equations of curve C1
def C1_parametric (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, 1 + Real.sin θ)

-- Define the polar equation of curve C2
def C2_polar (θ : ℝ) : ℝ :=
  4 * Real.sin (θ + Real.pi / 3)

-- Define the polar equation of line l
def line_l (θ : ℝ) : Prop :=
  θ = Real.pi / 6

-- Statement 1: Prove the standard equation of curve C1
theorem C1_standard_eq (θ : ℝ) :
  ∃ (x y : ℝ), C1_parametric θ = (x, y) ∧ (x^2 + (y - 1)^2 = 1) := 
begin
  sorry
end

-- Statement 2: Prove the Cartesian coordinate equation of curve C2
theorem C2_cartesian_eq (x y ρ θ : ℝ) :
  ρ = C2_polar θ ∧ ρ = Real.sqrt (x^2 + y^2) ∧ θ = Real.atan2 y x →
  x^2 + y^2 = 2 * y + 2 * Real.sqrt 3 * x := 
begin
  sorry
end

-- Statement 3: Prove the value of |AB| given intersection with line l
theorem AB_distance :
  let ρ1 := C2_polar (Real.pi / 6) in
  let ρ2 := 2 in
  |ρ2 - ρ1| = 2 := 
begin
  sorry
end

end C1_standard_eq_C2_cartesian_eq_AB_distance_l438_438911


namespace reinforcement_approx_l438_438748

noncomputable def reinforcement (initial_men : ℕ) (initial_days : ℕ) (days_passed : ℕ) (remaining_days : ℕ) : ℝ :=
  let provisions_left := initial_men * (initial_days - days_passed)
  let total_days_after_reinforcement := remaining_days
  let new_total_men := provisions_left / total_days_after_reinforcement
  new_total_men - initial_men

theorem reinforcement_approx : reinforcement 2000 120 25 35 ≈ 3429 := by
  -- let provisions_left := 2000 * (120 - 25)
  -- let new_total_men := provisions_left / 35
  -- let reinforcement := new_total_men - 2000
  -- the reinforcement should be approximately 3429
  sorry

end reinforcement_approx_l438_438748


namespace part_I_part_II_l438_438231

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | 5^x > 1 }
def B : Set ℝ := { x | log (1/3) (x + 1) > -1 }

-- Part (I): Prove the intersection of the complement of A with B
theorem part_I : (Aᶜ ∩ B) = { x | -1 < x ∧ x ≤ 0 } :=
sorry

-- Part (II) Define C based on its condition and prove the resulting range for a
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem part_II (a : ℝ) (h : B ∪ (C a) = C a) : a ≥ 2 :=
sorry

end part_I_part_II_l438_438231


namespace certain_number_value_l438_438375

theorem certain_number_value : 
  ∃ x : ℝ, x * 0.0729 * 28.9 / (0.0017 * 0.025 * 8.1) = 382.5 ∧ x ≈ 50.35 :=
by 
  use 50.35
  split
  · sorry
  · sorry

end certain_number_value_l438_438375


namespace area_triangle_PF1F2_eq_one_l438_438221

def ellipse_eq (a x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

def foci (a : ℝ) : (ℝ×ℝ) × (ℝ×ℝ) :=
  let c := real.sqrt (a^2 - 1) in
  ((c, 0), (-c, 0))

def circle_eq (diameter x y : ℝ) : Prop := x^2 + y^2 = (diameter / 2)^2

def tangent_point (a : ℝ) (P : ℝ × ℝ) : Prop :=
  let c := real.sqrt (a^2 - 1) in
  let O := (0 : ℝ, 0 : ℝ) in
  let diameter := 2 * c in
  ellipse_eq a P.1 P.2 ∧ circle_eq diameter P.1 P.2

theorem area_triangle_PF1F2_eq_one (a : ℝ) (P : ℝ × ℝ) :
  a > 1 →
  tangent_point a P →
  let F1 := foci a in
  let F2 := snd F1 in
  let PF2_length := abs (P.1 - fst F1.1) in
  1 = (1 / 2) * (2 * real.sqrt (a^2 - 1)) * PF2_length :=
sorry

end area_triangle_PF1F2_eq_one_l438_438221


namespace not_possible_arrange_cards_l438_438024

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438024


namespace john_boxes_bought_l438_438986

noncomputable def number_of_boxes (B : ℕ) : ℕ :=
  let total_burritos := B * 20
  let two_thirds := (2 * total_burritos) / 3
  let remaining_burritos := two_thirds - 30
  in if remaining_burritos = 10 then B else 0

theorem john_boxes_bought : ∃ B : ℕ, number_of_boxes B = 3 :=
by
  use 3
  sorry

end john_boxes_bought_l438_438986


namespace intersection_A_B_union_CU_A_CU_B_l438_438916

-- Define the universal set U
def U : Set ℝ := set.univ

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Define the complements of A and B in U
def CU_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}
def CU_B : Set ℝ := {x | x < -2 ∨ x > 3}

-- First proof statement
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x ≤ 3} := 
by
  sorry

-- Second proof statement
theorem union_CU_A_CU_B : CU_A ∪ CU_B = {x | x ≤ 1 ∨ x > 3} := 
by
  sorry

end intersection_A_B_union_CU_A_CU_B_l438_438916


namespace jimmy_pizza_cost_l438_438593

/-- Jimmy's pizza cost calculation -/
theorem jimmy_pizza_cost :
  let small_pizza_price := 8.00
  let medium_pizza_price := 12.00
  let large_pizza_price := 15.00
  let toppings_A_cost_first := 2.00
  let toppings_A_cost_additional := 1.50
  let toppings_B_cost_first_2 := 1.00
  let toppings_B_cost_additional := 0.75
  let toppings_C_cost := 0.50
  let discount_small := 1.00
  let discount_medium := 1.50
  let discount_large := 2.00
  let num_slices_medium := 10
  let num_toppings_A := 2
  let num_toppings_B := 3
  let num_toppings_C := 4
in (medium_pizza_price + 
    (toppings_A_cost_first + (num_toppings_A - 1) * toppings_A_cost_additional) + 
    (min 2 num_toppings_B * toppings_B_cost_first_2 + max 0 (num_toppings_B - 2) * toppings_B_cost_additional) + 
    (num_toppings_C * toppings_C_cost) 
    - discount_medium) 
    / num_slices_medium = 1.88 := by
    -- proof goes here
    sorry

end jimmy_pizza_cost_l438_438593


namespace jessica_has_three_dozens_of_red_marbles_l438_438982

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l438_438982


namespace smallest_X_divisible_by_15_l438_438301

theorem smallest_X_divisible_by_15 (T : ℕ) (h_pos : T > 0) (h_digits : ∀ (d : ℕ), d ∈ (Nat.digits 10 T) → d = 0 ∨ d = 1)
  (h_div15 : T % 15 = 0) : ∃ X : ℕ, X = T / 15 ∧ X = 74 :=
sorry

end smallest_X_divisible_by_15_l438_438301


namespace solve_trig_equation_l438_438640

theorem solve_trig_equation :
  (∃ n l : ℤ, x = 2 * n ∧ n ≠ 31 * l) ∨ 
  (∃ n l : ℤ, x = 31 / 33 * (2 * n + 1) ∧ n ≠ 33 * l + 16) 
  ↔
  cos (π * x / 31) * cos (2 * π * x / 31) * cos (4 * π * x / 31) * cos (8 * π * x / 31) * cos (16 * π * x / 31) = 1 / 32 :=
by
  sorry

end solve_trig_equation_l438_438640


namespace sum_three_times_m_and_half_n_square_diff_minus_square_sum_l438_438817

-- Problem (1) Statement
theorem sum_three_times_m_and_half_n (m n : ℝ) : 3 * m + 1 / 2 * n = 3 * m + 1 / 2 * n :=
by
  sorry

-- Problem (2) Statement
theorem square_diff_minus_square_sum (a b : ℝ) : (a - b) ^ 2 - (a + b) ^ 2 = (a - b) ^ 2 - (a + b) ^ 2 :=
by
  sorry

end sum_three_times_m_and_half_n_square_diff_minus_square_sum_l438_438817


namespace number_of_girls_attending_picnic_l438_438553

variables (g b : ℕ)

def hms_conditions : Prop :=
  g + b = 1500 ∧ (3 / 4 : ℝ) * g + (3 / 5 : ℝ) * b = 975

theorem number_of_girls_attending_picnic (h : hms_conditions g b) : (3 / 4 : ℝ) * g = 375 :=
sorry

end number_of_girls_attending_picnic_l438_438553


namespace find_k_sum_1500_l438_438456

theorem find_k_sum_1500 (k : ℕ) (h : let n := 9 * (10^k - 1) / 9 in nat.sum_digits n = 1500) : k = 167 := sorry

end find_k_sum_1500_l438_438456


namespace solve_inequalities_l438_438479

theorem solve_inequalities :
  (let s1 := {x : ℝ | 2 * x^2 + x - 3 < 0} in s1 = set.Ioo (-3 / 2) 1) ∧
  (let s2 := {x : ℝ | x * (9 - x) > 0} in s2 = set.Ioo 0 9) :=
by
  sorry

end solve_inequalities_l438_438479


namespace arith_seq_S13_value_l438_438531

variable {α : Type*} [LinearOrderedField α]

-- Definitions related to an arithmetic sequence
structure ArithSeq (α : Type*) :=
  (a : ℕ → α) -- the sequence itself
  (sum_first_n_terms : ℕ → α) -- sum of the first n terms

def is_arith_seq (seq : ArithSeq α) :=
  ∀ (n : ℕ), seq.a (n + 1) - seq.a n = seq.a 2 - seq.a 1

-- Our conditions
noncomputable def a5 (seq : ArithSeq α) := seq.a 5
noncomputable def a7 (seq : ArithSeq α) := seq.a 7
noncomputable def a9 (seq : ArithSeq α) := seq.a 9
noncomputable def S13 (seq : ArithSeq α) := seq.sum_first_n_terms 13

-- Problem statement
theorem arith_seq_S13_value (seq : ArithSeq α) 
  (h_arith_seq : is_arith_seq seq)
  (h_condition : 2 * (a5 seq) + 3 * (a7 seq) + 2 * (a9 seq) = 14) : 
  S13 seq = 26 := 
  sorry

end arith_seq_S13_value_l438_438531


namespace logan_television_hours_l438_438617

-- Definitions
def minutes_in_an_hour : ℕ := 60
def logan_minutes_watched : ℕ := 300
def logan_hours_watched : ℕ := logan_minutes_watched / minutes_in_an_hour

-- Theorem statement
theorem logan_television_hours : logan_hours_watched = 5 := by
  sorry

end logan_television_hours_l438_438617


namespace expression_bounds_l438_438996

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ≤ 4 * Real.sqrt 2 := 
sorry

end expression_bounds_l438_438996


namespace ellipse_properties_l438_438880

-- Given conditions
def F1 : Point := sorry -- Define F_1
def F2 : Point := sorry -- Define F_2 (right focus)
def D : Point := sorry -- Top vertex
def E : Point := sorry -- Right vertex
def a : ℝ := 2 -- Semi-major axis
def b : ℝ := √3 -- Semi-minor axis
def e : ℝ := 0.5 -- Eccentricity
def area_DEF2 : ℝ := sqrt 3 / 2 -- Area of triangle DEF2

-- Equation of the ellipse
def ellipse_eqn (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Theorem statement
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_eqn x y ↔ (x / 2)^2 + (y / sqrt 3)^2 = 1) ∧
  (∃ t : ℝ, ∀ x1 y1 x2 y2 : ℝ, 
    line_through F2 (x1, y1) ∧ line_through F2 (x2, y2) ∧ ellipse_eqn x1 y1 ∧ ellipse_eqn x2 y2 →
    ∃ val : ℝ,
    val = (|F2A| * |F2B|) / (S OAB) ∧ val = 3/2) :=
by
  sorry

end ellipse_properties_l438_438880


namespace no_valid_arrangement_l438_438028

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438028


namespace no_such_pairs_l438_438486

theorem no_such_pairs :
  ¬ ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4 * c < 0) ∧ (c^2 - 4 * b < 0) := sorry

end no_such_pairs_l438_438486


namespace range_of_a_l438_438245

noncomputable def curve (x a : ℝ) : ℝ := (x - a) * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ m₁ > 0 ∧ m₂ > 0 ∧ 
                   curve m₁ a / m₁ = (Real.log m₁ + (m₁ - a) / m₁) ∧ 
                   curve m₂ a / m₂ = (Real.log m₂ + (m₂ - a) / m₂) ∧
                   (curve m₁ a = 0) ∧ (curve m₂ a = 0)) -> a ∈ Iio (-Real.exp 2) := 
sorry

end range_of_a_l438_438245


namespace sin_beta_value_l438_438881

variable (α β : ℝ)

theorem sin_beta_value (h : sin α * cos (α - β) - cos α * sin (α - β) = 4 / 5) : sin β = 4 / 5 :=
by
  sorry

end sin_beta_value_l438_438881


namespace intersection_eq_union_eq_l438_438292

def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | 1 < x ∧ x ≤ 4 }

theorem intersection_eq : A ∩ B = { x : ℝ | 2 ≤ x ∧ x ≤ 4 } :=
by sorry

theorem union_eq : A ∪ B = { x : ℝ | 1 < x } :=
by sorry

end intersection_eq_union_eq_l438_438292


namespace sum_of_distances_is_correct_l438_438604

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := (x - 5) ^ 2 + 5

-- Points of intersection given
def point1 : ℝ × ℝ := (1, 21)
def point2 : ℝ × ℝ := (10, 30)
def point3 : ℝ × ℝ := (11, 41)

-- The correct solution for the sum of distances
def correct_sum_of_distances : ℝ := 4322.10

-- Calculate the distance from focus to a given point
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Coordinates of the focus of the parabola
def focus : ℝ × ℝ := (5, 5.5)

-- Statement of the property to be proved
theorem sum_of_distances_is_correct (d : ℝ × ℝ) 
    (h_intersects : d ≠ point1 ∧ d ≠ point2 ∧ d ≠ point3
        ∧ parabola d.1 = d.2) :
    distance focus point1 +
    distance focus point2 +
    distance focus point3 +
    distance focus d = correct_sum_of_distances := 
  sorry

end sum_of_distances_is_correct_l438_438604


namespace correct_calculation_l438_438387

-- Definitions of calculations based on conditions
def calc_A (a : ℝ) := a^2 + a^2 = a^4
def calc_B (a : ℝ) := (a^2)^3 = a^5
def calc_C (a : ℝ) := a + 2 = 2 * a
def calc_D (a b : ℝ) := (a * b)^3 = a^3 * b^3

-- Theorem stating that only the fourth calculation is correct
theorem correct_calculation (a b : ℝ) :
  ¬(calc_A a) ∧ ¬(calc_B a) ∧ ¬(calc_C a) ∧ calc_D a b :=
by sorry

end correct_calculation_l438_438387


namespace cos_pi_cos_solution_set_l438_438672

theorem cos_pi_cos_solution_set :
  {x | cos (Real.pi * cos x) = 0 ∧ 0 ≤ x ∧ x ≤ Real.pi} = {Real.pi / 3, 2 * Real.pi / 3} :=
by
  sorry

end cos_pi_cos_solution_set_l438_438672


namespace sum_of_coordinates_point_D_l438_438627

theorem sum_of_coordinates_point_D 
(M : ℝ × ℝ) (C D : ℝ × ℝ) 
(hM : M = (3, 5)) 
(hC : C = (1, 10)) 
(hmid : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))
: D.1 + D.2 = 5 :=
sorry

end sum_of_coordinates_point_D_l438_438627


namespace perimeter_bound_l438_438595

noncomputable def equilateral_triangle (side_length : ℝ) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.2 ≤ (sqrt 3) * p.1 ∧ p.2 ≤ (sqrt 3) * (side_length - p.1) }

variables (K : set (ℝ × ℝ)) (p : ℝ)
variables (n : ℕ)
variables (T : fin n → set (ℝ × ℝ))
variables (ε : ℝ)

def is_homothetic (A B : set (ℝ × ℝ)) (r : ℝ) : Prop :=
∃ (C : ℝ × ℝ), B = { x | ∃ y , y ∈ A ∧ x = C + (r • y) }

def non_overlapping (T : fin n → set (ℝ × ℝ)) : Prop :=
∀ i j, i ≠ j → disjoint (T i) (T j)

def area (s : set (ℝ × ℝ)) : ℝ := sorry -- Define the area of the set

def perimeter (s : set (ℝ × ℝ)) : ℝ := sorry -- Define the perimeter of the set

theorem perimeter_bound (p : ℝ) (hp : p > 0) :
  ∃ ε > 0, ∀ n (T : fin n → set (ℝ × ℝ)),
    non_overlapping T →
    (∀ i, is_homothetic K (T i) (-1)) →
    (sum (λ i, area (T i)) > area K - ε) →
    (sum (λ i, perimeter (T i)) > p) :=
sorry

end perimeter_bound_l438_438595


namespace area_PTR_l438_438254

-- Define points P, Q, R, S, and T
variables (P Q R S T : Type)

-- Assume QR is divided by points S and T in the given ratio
variables (QS ST TR : ℕ)
axiom ratio_condition : QS = 2 ∧ ST = 5 ∧ TR = 3

-- Assume the area of triangle PQS is given as 60 square centimeters
axiom area_PQS : ℕ
axiom area_PQS_value : area_PQS = 60

-- State the problem
theorem area_PTR : ∃ (area_PTR : ℕ), area_PTR = 90 :=
by
  sorry

end area_PTR_l438_438254


namespace Carmen_s_total_money_made_is_42_l438_438794

def money_made_from_green_house := 3 * 4
def money_made_from_yellow_house := 2 * 3.5 + 1 * 5
def money_made_from_brown_house := 9 * 2

def total_money_made := money_made_from_green_house +
                        money_made_from_yellow_house +
                        money_made_from_brown_house

theorem Carmen_s_total_money_made_is_42 : total_money_made = 42 := 
by 
  sorry

end Carmen_s_total_money_made_is_42_l438_438794


namespace difference_of_digits_l438_438650

theorem difference_of_digits (X Y : ℕ) (h1 : 10 * X + Y < 100) 
  (h2 : 72 = (10 * X + Y) - (10 * Y + X)) : (X - Y) = 8 :=
sorry

end difference_of_digits_l438_438650


namespace _l438_438729

noncomputable def AC_CB_Proof : Prop := 
  let A := (1, 2) : ℝ × ℝ
  let B := (17, 14) : ℝ × ℝ
  let C := (13, 11) : ℝ × ℝ
  AC_CB_Proof ⟨(C.1 - A.1)^2 + (C.2 - A.2)^2⟩ = 15^2 ∧ 
  AC_CB_Proof ⟨(B.1 - C.1)^2 + (B.2 - C.2)^2⟩ = 5^2 ∧ 
  3/1 = 3

@[simp] theorem AC_CB_Statement : AC_CB_Proof := 
by 
  sorry

end _l438_438729


namespace bullet_train_time_pass_man_l438_438062

variable (length_of_train : ℝ) (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ)

def relative_speed_kmph : ℝ :=
  speed_of_train_kmph + speed_of_man_kmph

def relative_speed_mps : ℝ :=
  relative_speed_kmph * (1000 / 3600)

def time_to_pass : ℝ :=
  length_of_train / relative_speed_mps

theorem bullet_train_time_pass_man :
  length_of_train = 120 →
  speed_of_train_kmph = 50 →
  speed_of_man_kmph = 4 →
  time_to_pass length_of_train speed_of_train_kmph speed_of_man_kmph = 8 :=
by
  intros
  sorry

end bullet_train_time_pass_man_l438_438062


namespace transformation_correctness_l438_438964

variable (x x' y y' : ℝ)

-- Conditions
def original_curve : Prop := y^2 = 4
def transformed_curve : Prop := (x'^2)/1 + (y'^2)/4 = 1
def transformation_formula : Prop := (x = 2 * x') ∧ (y = y')

-- Proof Statement
theorem transformation_correctness (h1 : original_curve y) (h2 : transformed_curve x' y') :
  transformation_formula x x' y y' :=
  sorry

end transformation_correctness_l438_438964


namespace find_lambda_l438_438920

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 0)
def c : ℝ × ℝ := (1, -2)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

theorem find_lambda (λ : ℝ) 
  (h : collinear (λ * a.1 + b.1, λ * a.2 + b.2) c) : 
  λ = -1 :=
  sorry

end find_lambda_l438_438920


namespace numbers_greater_than_negative_one_l438_438095

theorem numbers_greater_than_negative_one :
  ∀ x : Int, (x ∈ {-3, -2, -1, 0} ∧ x > -1) ↔ x = 0 :=
by
  intro x
  constructor
  · intro h
    -- to state that any number > -1 in the given set must be 0.
    sorry
  · intro h
    -- to state that 0 in the given set is indeed greater than -1.
    sorry

end numbers_greater_than_negative_one_l438_438095


namespace num_valid_arrangements_l438_438168

theorem num_valid_arrangements :
  ∃ (A B C D E : ℕ → Prop), 
    (¬ (A 1 ∨ A 5)) ∧ 
    ((C 2 ∧ D 3) ∨ (C 3 ∧ D 2) ∨ (C 3 ∧ D 4) ∨ (C 4 ∧ D 3) ∨ (C 4 ∧ D 5) ∨ (C 5 ∧ D 4)) →
    (∑ perm in permutations [A, B, C, D, E], 
      if valid_arrangement perm then 1 else 0) = 24 := 
begin
  sorry
end

def valid_arrangement (perm : list (ℕ → Prop)) : bool :=
  -- Definition of an arrangement being valid considering the constraints
  sorry

end num_valid_arrangements_l438_438168


namespace max_log_value_l438_438544

theorem max_log_value (θ a b c: ℝ) (m n : ℝ)
  (h_eq_root: ∀ x, sin θ * x^2 + cos θ * x - 1 = 0 → x = m ∨ x = n)
  (h_distinct : m ≠ n)
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_eq: a * b * c + b^2 + c^2 = 8) :
  log 4 a + log 2 b + log 2 c = 3 / 2 :=
by 
  sorry

end max_log_value_l438_438544


namespace coefficient_x2_in_binomial_expansion_l438_438965

theorem coefficient_x2_in_binomial_expansion : 
  (∃ T : ℕ → ℕ → ℕ → ℕ, (T 7 2 2 = 21)) :=
by
  let T := λ n k r => Nat.choose n k
  have coeff := T 7 2
  exact ⟨coeff, rfl⟩

end coefficient_x2_in_binomial_expansion_l438_438965


namespace ratio_parallel_segments_l438_438288

theorem ratio_parallel_segments {A B C I A1 B1 A2 B2 N M : Type*} [geometry_type I A B C A1 B1 A2 B2 N M]
  (h1 : incircle_center_triangle I A B C) (h2 : intersection A1 (line_through A I) (line_through B C))
  (h3 : intersection B1 (line_through B I) (line_through A C)) (h4 : parallel (line_through A1 (line_through A C))
  (line_through A C)) (h5 : parallel (line_through B1 (line_through B C)) (line_through B C))
  (h6 : intersection A2 (line_through A1 (line_through A C)) (line_through C I))
  (h7 : intersection B2 (line_through B1 (line_through B C)) (line_through C I))
  (h8 : intersection N (line_through A (line_through A2 C)) (line_through B (line_through B2 C)))
  (h9 : midpoint M A B) (h10 : parallel (line_through C N) (line_through I M)) :
  CN / IM = 2 := sorry

end ratio_parallel_segments_l438_438288


namespace cos_pi_minus_2alpha_l438_438238

theorem cos_pi_minus_2alpha (α : ℝ) (h : cos (π / 2 - α) = sqrt 2 / 3) : cos (π - 2 * α) = -5 / 9 :=
by 
  sorry

end cos_pi_minus_2alpha_l438_438238


namespace count_true_propositions_l438_438434

theorem count_true_propositions :
  let prop1 := false  -- Proposition ① is false
  let prop2 := true   -- Proposition ② is true
  let prop3 := true   -- Proposition ③ is true
  let prop4 := false  -- Proposition ④ is false
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
by
  -- The theorem is expected to be proven here
  sorry

end count_true_propositions_l438_438434


namespace sugar_remaining_correct_l438_438795

def remaining_sugar (initial : ℝ) (fraction_lost : ℝ) : ℝ := initial * (1 - fraction_lost)

def total_remaining_sugar (total_sugar : ℝ) (bags : ℕ) (losses : List ℝ) : ℝ :=
  let sugar_per_bag := total_sugar / bags
  let remaining_sugars := losses.map (λ loss → remaining_sugar sugar_per_bag loss)
  remaining_sugars.sum

theorem sugar_remaining_correct : total_remaining_sugar 24 4 [0.10, 0.15, 0.20, 0.25] = 19.8 :=
by
  sorry

end sugar_remaining_correct_l438_438795


namespace max_distance_time_l438_438283

def IvanLap : ℕ := 20
def PeterLap : ℕ := 28
def RunningOppositeDirections : Prop := true
def InitiallyMinimalDistance : Prop := true

theorem max_distance_time : RunningOppositeDirections →
                             InitiallyMinimalDistance →
                             ∃ t : ℚ, t = 35 / 6 :=
by
  intros _ _
  use 35 / 6
  sorry

end max_distance_time_l438_438283


namespace product_of_m_n_l438_438811

noncomputable def inscribed_sphere_radius_cube (a : ℝ) : ℝ := a / 2

noncomputable def inscribed_sphere_radius_octahedron (a : ℝ) : ℝ := a * Real.sqrt 6 / 6

theorem product_of_m_n (a : ℝ) (hx : a > 0):
  let r1 := inscribed_sphere_radius_cube a,
      r2 := inscribed_sphere_radius_octahedron a
  in r1 / r2 = 2 / 3 → (2 * 3 = 6) :=
sorry

end product_of_m_n_l438_438811


namespace real_part_division_number_of_lines_decreasing_interval_max_ratio_MN_AB_l438_438727

-- Proof problem for Question 1
theorem real_part_division (a : ℝ) (ha : (a - complex.i) / (2 + complex.i) ∈ ℝ) : a = - 2 := sorry

-- Proof problem for Question 2
theorem number_of_lines (h : ∀ l : ℝ × ℝ → Prop, l (3, 0)) : ∃! l : ℝ × ℝ → Prop, 
(l ↔ line_through_point (3, 0) ∧ l intersects_one_point_on_hyperbola 4 * x^2 - 9 * y^2 = 36) → 
∃ l₁ l₂ l₃, distinct_lines l₁ l₂ l₃ := sorry

-- Proof problem for Question 3
theorem decreasing_interval (h : ∀ x ∈ Set.Ioo (-1 : ℝ) 4, deriv (λx, ln (4 + 3 * x - x^2)) x < 0) : 
Set.Icc (3 / 2 : ℝ) 4 := sorry

-- Proof problem for Question 4
theorem max_ratio_MN_AB (p : ℝ) (hp : 0 < p ) (f : ℝ × ℝ) (parabola : ∀ y, y^2 = 2 * p * numerator x) : 
(∀ a b : ℝ, to_rat (|AB|) <= √(2) / 2 : by using properties of parabola and inequalities) := sorry

end real_part_division_number_of_lines_decreasing_interval_max_ratio_MN_AB_l438_438727


namespace find_fraction_B_minus_1_over_A_l438_438509

variable (A B : ℝ) (a_n S_n : ℕ → ℝ)
variable (h1 : ∀ n, a_n n + S_n n = A * (n ^ 2) + B * n + 1)
variable (h2 : A ≠ 0)

theorem find_fraction_B_minus_1_over_A : (B - 1) / A = 3 := by
  sorry

end find_fraction_B_minus_1_over_A_l438_438509


namespace sum_of_digits_smallest_N_l438_438997

def is_multiple_of_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def Q (N : ℕ) : ℚ :=
  let numerator := ∑ i in finset.range (N + 2), (max 0 (N + 1 - i - (N / 2)))
  let denominator := (N + 2) * (N + 1)
  numerator / denominator

theorem sum_of_digits_smallest_N (N : ℕ) (h : Q N ≥ (1 / 2)) :
  is_multiple_of_five N → N = 10 → (N.digits 10).sum = 1 :=
by sorry

end sum_of_digits_smallest_N_l438_438997


namespace final_amount_paid_l438_438591

theorem final_amount_paid 
  (price_short : ℕ) (num_short : ℕ)
  (price_shirt : ℕ) (num_shirt : ℕ)
  (senior_discount : ℝ) (tax_rate : ℝ)
  (promotion_factor : ℕ) :
  price_short = 15 →
  num_short = 3 →
  price_shirt = 17 →
  num_shirt = 5 →
  senior_discount = 0.1 →
  tax_rate = 0.05 →
  promotion_factor = 2 →
  (∃ total_paid : ℝ, total_paid = 76.55) :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  use 76.55
  sorry

end final_amount_paid_l438_438591


namespace simplify_expression_l438_438108

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 2)) + (1 / (1 - Real.sqrt 2)) = 1 - 2 * Real.sqrt 2 := by
  sorry

end simplify_expression_l438_438108


namespace study_tour_probability_l438_438739

-- Definition of event N
def event_N (classes destinations : Finset ℕ) : Prop :=
  ∃ d ∈ destinations, ∀ c ∈ classes, c ≠ 1 → ∀ d' ∈ destinations, d' ≠ d → c ≠ d'

-- Definition of event MN
def event_MN (classes destinations : Finset ℕ) : Prop :=
  ∀ c1 c2 ∈ classes, c1 ≠ c2 → ∃ d1 d2 ∈ destinations, d1 ≠ d2 ∧ c1 ≠ d1 ∧ c2 ≠ d2

-- Conditional probability P(M|N)
noncomputable def P_M_given_N (classes destinations : Finset ℕ) : ℚ :=
  if hN : event_N classes destinations then
    let nMN := (4 * 3 * 2 * 1 : ℚ) 
    let nN := (4 * 27 : ℚ)
    nMN / nN
  else 0

-- Theorem to be proved
theorem study_tour_probability : 
  P_M_given_N {1, 2, 3, 4} {a, b, c, d} = 2 / 9 :=
by
  sorry

end study_tour_probability_l438_438739


namespace geometric_sequence_a3_l438_438969

theorem geometric_sequence_a3 (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 5 = 4) (h3 : ∀ n, a n = a 1 * q ^ (n - 1)) : a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l438_438969


namespace solution_l438_438201

def p (x : ℝ) : Prop := x^2 + 2 * x - 3 < 0
def q (x : ℝ) : Prop := x ∈ Set.univ

theorem solution (x : ℝ) (hx : p x ∧ q x) : x = -2 ∨ x = -1 ∨ x = 0 := 
by
  sorry

end solution_l438_438201


namespace ones_digit_sum_l438_438815

theorem ones_digit_sum : (5^45 + 6^45 + 7^45 + 8^45 + 9^45) % 10 = 5 :=
by
  -- Definitions of mod patterns
  have h1 : 5^45 % 10 = 5 := by exact sorry,
  have h2 : 6^45 % 10 = 6 := by exact sorry,
  have h3 : 7^45 % 10 = 7 := by exact sorry,
  have h4 : 8^45 % 10 = 8 := by exact sorry,
  have h5 : 9^45 % 10 = 9 := by exact sorry,
  -- Use the mod patterns to calculate the final result
  calc (5^45 + 6^45 + 7^45 + 8^45 + 9^45) % 10
       = (5 + 6 + 7 + 8 + 9) % 10 : by rw [h1, h2, h3, h4, h5]
   ... = 35 % 10
   ... = 5

end ones_digit_sum_l438_438815


namespace ratio_of_milk_to_water_new_mixture_l438_438713

theorem ratio_of_milk_to_water_new_mixture
  (m1 w1 : ℕ) (m2 w2 : ℕ)
  (h1 : m1 + w1 = m2 + w2)
  (h2 : m1 = 4 * (w1 / 2))
  (h3 : m2 = 5 * (w2 / 1)) :
  (m1 + m2) / (w1 + w2) = 3 :=
by {
  sorry,
}

end ratio_of_milk_to_water_new_mixture_l438_438713


namespace area_sum_eq_l438_438099

-- Define the conditions given in the problem
variables {A B C P Q R M N : Type*}

-- Define the properties of the points
variables (triangle_ABC : Triangle A B C)
          (point_P : OnSegment P A B)
          (point_Q : OnSegment Q B C)
          (point_R : OnSegment R A C)
          (parallelogram_PQCR : Parallelogram P Q C R)
          (intersection_M : Intersection M (LineSegment AQ) (LineSegment PR))
          (intersection_N : Intersection N (LineSegment BR) (LineSegment PQ))

-- Define the areas of the triangles involved
variables (area_AMP area_BNP area_CQR : ℝ)

-- Define the conditions for the areas of the triangles
variables (h_area_AMP : area_AMP = Area (Triangle A M P))
          (h_area_BNP : area_BNP = Area (Triangle B N P))
          (h_area_CQR : area_CQR = Area (Triangle C Q R))

-- The theorem to be proved
theorem area_sum_eq :
  area_AMP + area_BNP = area_CQR :=
sorry

end area_sum_eq_l438_438099


namespace min_value_expression_l438_438859

theorem min_value_expression (α β : ℝ)
  (hα1 : 0 ≤ α) (hα2 : α ≤ π / 2)
  (hβ1 : 0 < β) (hβ2 : β ≤ π / 2) :
  ∃ min_val : ℝ, min_val = 1 ∧ ∀ γ : ℝ, (γ = cos α ^ 2 * sin β + 1 / sin β) → γ ≥ min_val :=
begin
  sorry
end

end min_value_expression_l438_438859


namespace total_votes_cast_l438_438954

theorem total_votes_cast (V: ℕ) (invalid_votes: ℕ) (diff_votes: ℕ) 
  (H1: invalid_votes = 200) 
  (H2: diff_votes = 700) 
  (H3: (0.01 : ℝ) * V = diff_votes) 
  : (V + invalid_votes = 70200) :=
by
  sorry

end total_votes_cast_l438_438954


namespace profitWednesday_l438_438734

-- Define the total profit
def totalProfit : ℝ := 1200

-- Define the profit made on Monday
def profitMonday : ℝ := totalProfit / 3

-- Define the profit made on Tuesday
def profitTuesday : ℝ := totalProfit / 4

-- Theorem to prove the profit made on Wednesday
theorem profitWednesday : 
  let profitWednesday := totalProfit - (profitMonday + profitTuesday)
  profitWednesday = 500 :=
by
  -- proof goes here
  sorry

end profitWednesday_l438_438734


namespace original_area_of_circle_l438_438063

theorem original_area_of_circle
  (A₀ : ℝ) -- original area
  (r₀ r₁ : ℝ) -- original and new radius
  (π : ℝ := 3.14)
  (h_area : A₀ = π * r₀^2)
  (h_area_increase : π * r₁^2 = 9 * A₀)
  (h_circumference_increase : 2 * π * r₁ - 2 * π * r₀ = 50.24) :
  A₀ = 50.24 :=
by
  sorry

end original_area_of_circle_l438_438063


namespace closest_integer_to_sum_l438_438159

theorem closest_integer_to_sum :
  let S := 500 * ∑ n in Finset.range 4999 \ Finset.singleton 0, 1 / (n + 2)^2 - 1 
  375 = Int.floor S :=
by
  sorry

end closest_integer_to_sum_l438_438159


namespace total_money_given_to_children_l438_438421

theorem total_money_given_to_children (B : ℕ) (x : ℕ) (total : ℕ) 
  (h1 : B = 300) 
  (h2 : x = B / 3) 
  (h3 : total = (2 * x) + (3 * x) + (4 * x)) : 
  total = 900 := 
by 
  sorry

end total_money_given_to_children_l438_438421


namespace intersection_points_polar_coords_polar_equation_C1_l438_438963

-- Given parametric equations for C1
def parametric_eqns_C1 (t : ℝ) : ℝ × ℝ :=
(x = t^2, y = t)

-- Given polar equation for C2
def polar_eqn_C2 (ρ θ : ℝ) : Prop :=
ρ^2 + 2*ρ*cos θ - 4 = 0

-- Cartesian equation derived from C2 for intersection
def cartesian_eqn_derived_C2 (x y : ℝ) : Prop :=
(x + 1)^2 + y^2 = 5

-- Cartesian equation derived from C1 for intersection
def cartesian_eqn_derived_C1 (x y : ℝ) : Prop :=
y^2 = x

-- Proving the intersection points
theorem intersection_points_polar_coords :
  ∀ (t ρ θ : ℝ), parametric_eqns_C1(t) ∧ polar_eqn_C2 (ρ θ) →
                (ρ = sqrt 2 ∧ (θ = π / 4 ∨ θ = 7 * π / 4)) :=
begin
  sorry -- proof omitted
end

-- Proving the polar equation of C1
theorem polar_equation_C1 : ∀ (x y ρ θ : ℝ),
   (x = t ^ 2 ∧ y = t → ρ * cos θ = ρ^2 * sin^2 θ → cos θ = ρ * sin^2 θ) :=
begin
  sorry -- proof omitted
end

end intersection_points_polar_coords_polar_equation_C1_l438_438963


namespace acute_triangle_inequalities_l438_438579

theorem acute_triangle_inequalities
  (A B C : ℝ)
  (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sum_pi : A + B + C = π)
  (h_gt : A > B) :
  (sin A > sin B) ∧ (cos A < cos B) ∧ (sin A + sin B > cos A + cos B) :=
by
  sorry

end acute_triangle_inequalities_l438_438579


namespace correct_answer_l438_438222

-- Define the necessary conditions for our problem
variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Let V be a vector space over ℝ
variables {a b c : V}
variables {O A B C : V}
variables {v₁ v₂ v₃ v₄ : V}

-- Define the property conditions (using hypotheses)
def condition1 (ha : a = 0 ∨ b = 0) : Prop := ¬(∀ v, linear_independent ℝ ![a, b, v])

def condition2 (h : ¬ linear_independent ℝ ![\(O - A), \(O - B), \(O - C)]) : Prop := 
  ∃ α β γ ∈ ℝ, α • \(O - A\) + β • \(O - B\) + γ • \(O - C\) = 0

def condition3 : Prop := 
  (linear_independent ℝ ![a, b, c]) → linear_independent ℝ ![a + b, a - b, c]

-- Define the theorem to show the correct answer
theorem correct_answer : 
  (∀ a b, ¬condition1 a b = false) ∧
  (∀ O A B C, condition2 O A B C) ∧
  (condition3)
:= 
  sorry

end correct_answer_l438_438222


namespace multiplication_result_l438_438789

theorem multiplication_result : 
  (500 * 2468 * 0.2468 * 100) = 30485120 :=
by
  sorry

end multiplication_result_l438_438789


namespace intersection_distance_of_ellipse_hyperbola_l438_438069

theorem intersection_distance_of_ellipse_hyperbola :
  let a := 6
  let b := 4
  let c := 2 * Real.sqrt 5
  let ellipse (x y : ℝ) := (x^2 / 36) + (y^2 / 16) = 1
  let hyperbola (x y : ℝ) := (x^2 / (c^2)) - (y^2 / ((c * (2 / 3))^2)) = 1
  let foci : ℝ × ℝ := (c, 0)
  (∀ x y : ℝ, ellipse x y ∧ hyperbola x y →
    let distance := 2 * |2 * Real.sqrt 5 * Real.cos (Real.atan ((2 * Real.sqrt 10) / 3))|
    distance = 12 * Real.sqrt 2 / 5) :=
sorry

end intersection_distance_of_ellipse_hyperbola_l438_438069


namespace series_sum_l438_438610

theorem series_sum (s : ℝ) (h : s^3 + (3 / 7) * s - 1 = 0) :
  s^2 + 3 * s^5 + 5 * s^8 + 7 * s^11 + ∑' n : ℕ, (if n % 3 = 1 then n * s^(3*n-1) else 0) = 7 / 3 :=
begin
  sorry
end

end series_sum_l438_438610


namespace jessies_initial_weight_l438_438770

-- Definitions based on the conditions
def weight_lost : ℕ := 126
def current_weight : ℕ := 66

-- The statement to prove
theorem jessies_initial_weight :
  (weight_lost + current_weight = 192) :=
by 
  sorry

end jessies_initial_weight_l438_438770


namespace real_root_of_quadratic_eq_l438_438494

open Complex

theorem real_root_of_quadratic_eq (k : ℝ) (a : ℝ) :
  (a^2 + (k + 2 * Complex.i) * a + (2 + k * Complex.i) = 0) →
  (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
begin
  sorry
end

end real_root_of_quadratic_eq_l438_438494


namespace find_difference_of_a_b_l438_438615

noncomputable def a_b_are_relative_prime_and_positive (a b : ℕ) (hab_prime : Nat.gcd a b = 1) (ha_pos : a > 0) (hb_pos : b > 0) (h_gt : a > b) : Prop :=
  a ^ 3 - b ^ 3 = (131 / 5) * (a - b) ^ 3

theorem find_difference_of_a_b (a b : ℕ) 
  (hab_prime : Nat.gcd a b = 1) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (h_gt : a > b) 
  (h_eq : (a ^ 3 - b ^ 3 : ℚ) / (a - b) ^ 3 = 131 / 5) : 
  a - b = 7 :=
  sorry

end find_difference_of_a_b_l438_438615


namespace count_non_congruent_triangles_eq_190_l438_438436

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_congruent_triangles_count : ℕ :=
  Nat.card {t : ℕ × ℕ × ℕ // let (a, b, c) := t in a ≤ b ∧ b ≤ c ∧ a + b + c ≤ 100 ∧ c - a ≤ 2 ∧ is_triangle a b c}

theorem count_non_congruent_triangles_eq_190 : non_congruent_triangles_count = 190 := 
  sorry

end count_non_congruent_triangles_eq_190_l438_438436


namespace line_passes_through_circle_center_l438_438184

theorem line_passes_through_circle_center (a : ℝ) : 
  ∀ x y : ℝ, (x, y) = (a, 2*a) → (x - a)^2 + (y - 2*a)^2 = 1 → 2*x - y = 0 :=
by
  sorry

end line_passes_through_circle_center_l438_438184


namespace no_valid_arrangement_l438_438006

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438006


namespace fourth_number_pascal_row_l438_438971

theorem fourth_number_pascal_row : (Nat.choose 12 3) = 220 := sorry

end fourth_number_pascal_row_l438_438971


namespace find_radius_of_original_sphere_l438_438083

-- Define the known quantities and conditions
def original_radius (R : ℝ) (r : ℝ) : Prop :=
  r = 4 * real.cbrt 4 ∧
  R = real.cbrt (1 / 4) * r

-- Aim to prove the radius of the original spherical bubble
theorem find_radius_of_original_sphere (r : ℝ) (R : ℝ) (h : r = 4 * real.cbrt 4):
  original_radius R r → R = 4 :=
by
  sorry

end find_radius_of_original_sphere_l438_438083


namespace geometric_sequence_modulus_one_l438_438520

theorem geometric_sequence_modulus_one (α : ℝ) (i : ℂ) (h : i = Complex.I) :
  ∃ (r : ℂ), (|r| = 1) ∧ (∀ (n : ℕ), n ≥ 1 → (cos (n * α) + i * sin (n * α)) = r^n) :=
by
  sorry

end geometric_sequence_modulus_one_l438_438520


namespace no_positive_integer_solution_exists_l438_438474

theorem no_positive_integer_solution_exists :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * x^2 + 2 * x + 2 = y^2 :=
by
  -- The proof steps will go here.
  sorry

end no_positive_integer_solution_exists_l438_438474


namespace sum_telescoping_fraction_l438_438116

theorem sum_telescoping_fraction :
  ∑ n in Finset.range 999 + 2, (1 / (n * (n - 1))) = 499 / 1000 :=
by
  sorry

end sum_telescoping_fraction_l438_438116


namespace distance_covered_l438_438708

-- Define the given conditions
def time_minutes : ℝ := 42
def speed_km_per_hr : ℝ := 10
def time_hours : ℝ := time_minutes / 60

-- Define the statement to be proved
theorem distance_covered (time : ℝ) (speed : ℝ) (time_in_hours : ℝ) (h_time : time = 42) (h_speed : speed = 10) (h_time_in_hours : time_in_hours = time / 60) : 
  speed * time_in_hours = 7 :=
by
  sorry

end distance_covered_l438_438708


namespace value_of_w_l438_438569

theorem value_of_w (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := 
sorry

end value_of_w_l438_438569


namespace daniel_spent_2290_l438_438128

theorem daniel_spent_2290 (total_games: ℕ) (price_12_games count_price_12: ℕ) 
  (price_7_games frac_price_7: ℕ) (price_3_games: ℕ) 
  (count_price_7: ℕ) (h1: total_games = 346)
  (h2: count_price_12 = 80) (h3: price_12_games = 12)
  (h4: frac_price_7 = 50) (h5: price_7_games = 7)
  (h6: price_3_games = 3) (h7: count_price_7 = (frac_price_7 * (total_games - count_price_12)) / 100):
  (count_price_12 * price_12_games) + (count_price_7 * price_7_games) + ((total_games - count_price_12 - count_price_7) * price_3_games) = 2290 := 
by
  sorry

end daniel_spent_2290_l438_438128


namespace contemporaries_probability_l438_438379

-- Define the conditions for the birth years and the lifespan of the mathematicians.
def birth_year_range := [0, 600]

-- Define a function that checks if two intervals overlap
def intervals_overlap (a b : ℝ × ℝ) : Prop :=
  ∃ x, x ∈ (a.1 .. a.2) ∧ x ∈ (b.1 .. b.2)

-- Given that each mathematician lives for 120 years
def lifespan (birth : ℝ) : ℝ × ℝ :=
  (birth, birth + 120)

-- Define the birth years of Alice, Bob, and Charlie
variable (x y z : ℝ)
-- Define the lifespan intervals
def alice_lifespan := lifespan x
def bob_lifespan := lifespan y
def charlie_lifespan := lifespan z

-- The theorem statement
theorem contemporaries_probability :
  (x ∈ birth_year_range ∧ y ∈ birth_year_range ∧ z ∈ birth_year_range) →
  (intervals_overlap alice_lifespan bob_lifespan ∨ 
  intervals_overlap alice_lifespan charlie_lifespan ∨ 
  intervals_overlap bob_lifespan charlie_lifespan) → 
  ∃ p : ℝ, p ≈ 13.824 / 216 := 
sorry

end contemporaries_probability_l438_438379


namespace bread_last_days_l438_438683

theorem bread_last_days (num_members : ℕ) (breakfast_slices : ℕ) (snack_slices : ℕ) (slices_per_loaf : ℕ) (num_loaves : ℕ) :
  num_members = 4 →
  breakfast_slices = 3 →
  snack_slices = 2 →
  slices_per_loaf = 12 →
  num_loaves = 5 →
  (num_loaves * slices_per_loaf) / (num_members * (breakfast_slices + snack_slices)) = 3 :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end bread_last_days_l438_438683


namespace card_arrangement_impossible_l438_438034

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438034


namespace inscribed_circle_exists_l438_438050

theorem inscribed_circle_exists (P : Type) [convex_polygon P]
  (transformation_property : ∃ (P' : Type) [convex_polygon P'], 
    is_similar P P' ∧ outward_transformation P P' 1) :
  ∃ (O : Point) (r : ℝ), is_inscribed_circle P O r :=
sorry

end inscribed_circle_exists_l438_438050


namespace trash_equilibrium_N_2_trash_equilibrium_N_10_l438_438260

-- Define the condition for the trash transfer for any N untidy people
def trash_transfer (N : ℕ) (initial_trash : Fin N → ℝ) : (Fin N → ℝ) :=
  λ i, initial_trash i + (1/2) * (∑ j in (univ.filter (≠ i)), initial_trash j)

theorem trash_equilibrium_N_2 (initial_trash : Fin 2 → ℝ) (h_nonzero : ∀ i, initial_trash i ≠ 0):
  trash_transfer 2 initial_trash = initial_trash := 
sorry

theorem trash_equilibrium_N_10 (initial_trash : Fin 10 → ℝ) (h_nonzero : ∀ i, initial_trash i ≠ 0):
  trash_transfer 10 initial_trash = initial_trash := 
sorry

end trash_equilibrium_N_2_trash_equilibrium_N_10_l438_438260


namespace numbers_greater_than_negative_one_l438_438096

theorem numbers_greater_than_negative_one :
  ∀ x : Int, (x ∈ {-3, -2, -1, 0} ∧ x > -1) ↔ x = 0 :=
by
  intro x
  constructor
  · intro h
    -- to state that any number > -1 in the given set must be 0.
    sorry
  · intro h
    -- to state that 0 in the given set is indeed greater than -1.
    sorry

end numbers_greater_than_negative_one_l438_438096


namespace card_arrangement_impossible_l438_438032

theorem card_arrangement_impossible (cards : set ℕ) (h1 : cards = {1, 2, 3, 4, 5, 6, 8, 9}) :
  ∀ (sequence : list ℕ), sequence.length = 8 → (∀ i, i < 7 → (sequence.nth_le i sorry) * 10 + (sequence.nth_le (i+1) sorry) % 7 = 0) → false :=
by
  sorry

end card_arrangement_impossible_l438_438032


namespace fred_current_dimes_l438_438858

-- Definitions based on the conditions
def original_dimes : ℕ := 7
def borrowed_dimes : ℕ := 3

-- The theorem to prove
theorem fred_current_dimes : original_dimes - borrowed_dimes = 4 := by
  sorry

end fred_current_dimes_l438_438858


namespace volume_of_sphere_in_cone_l438_438762

-- Defining the conditions for the problem
def is_right_angled_isosceles_triangle (ABC : Triangle) : Prop :=
  (ABC.vertex_angle = 90)

def base_diameter (ABC : Triangle) : ℝ :=
  24

def sphere_inside_cone (conical_base : ℝ) (C_vertex_angle : ℝ) : ℝ :=
  6 * sqrt 2

-- Proof Problem: Volume of the sphere in the specified cone
theorem volume_of_sphere_in_cone : (4 / 3) * π * (6 * sqrt 2) ^ 3 = 576 * sqrt 2 * π :=
by 
  sorry

end volume_of_sphere_in_cone_l438_438762


namespace number_of_divisors_of_3003_l438_438829

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

theorem number_of_divisors_of_3003 : count_divisors 3003 = 16 :=
by
  sorry

end number_of_divisors_of_3003_l438_438829


namespace area_of_triangle_l438_438424

theorem area_of_triangle (m h : ℝ) (hm : m > 0) (hh : h > 0) : 
  let base := h in
  let height := m * h in
  let area := (1 / 2) * base * height in
  area = (1 / 2) * m * h^2 :=
by 
  sorry

end area_of_triangle_l438_438424


namespace length_MN_l438_438299

noncomputable def compute_length_MN 
  (A B C X M N : Type) 
  [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space M] [metric_space N]
  (AB : ℝ) (AC : ℝ) (BC : ℝ) (AX : ℝ) (BX : ℝ) (CX : ℝ) 
  (XM_AC_parallel : Prop) (XN_BC_parallel : Prop) : ℝ :=
  sorry

theorem length_MN
  {A B C X M N : Type} [metric_space A] [metric_space B] 
  [metric_space C] [metric_space X] [metric_space M] [metric_space N]
  (h1 : dist A B = 5)
  (h2 : dist A C = 4)
  (h3 : dist B C = 6)
  (h4 : ∃ X, is_angle_bisector A B C X)
  (M_on_BC : ∃ M, M ∈ line_segment B C ∧ parallel (line_through X M) (line_through A C))
  (N_on_AC : ∃ N, N ∈ line_segment A C ∧ parallel (line_through X N) (line_through B C)) :
  dist M N = 3 * sqrt 14 / 5 := sorry

end length_MN_l438_438299


namespace initially_calculated_average_l438_438647

theorem initially_calculated_average :
  ∀ (S : ℕ), (S / 10 = 18) →
  ((S - 46 + 26) / 10 = 16) :=
by
  sorry

end initially_calculated_average_l438_438647


namespace generate_random_variables_l438_438152

/-- 
Given the joint probability density function f(x, y) = 3/4 * x * y^2 
and the region bounded by the lines x = 0, y = 0, x = 1, and y = 2.
We need to prove that the explicit random generation formulas for X and Y are
x_i = sqrt(r_i) and y_i = 2 * (r_i)^(1/3).
-/
theorem generate_random_variables {r_i : ℝ} (h : 0 ≤ r_i ∧ r_i ≤ 1) :
  let x_i := Real.sqrt r_i,
      y_i := 2 * Real.cbrt r_i in
  (0 ≤ x_i ∧ x_i ≤ 1) ∧ (0 ≤ y_i ∧ y_i ≤ 2) :=
by
  sorry

end generate_random_variables_l438_438152


namespace rowing_distance_l438_438041

theorem rowing_distance :
  let row_speed := 4 -- kmph
  let river_speed := 2 -- kmph
  let total_time := 1.5 -- hours
  ∃ d, 
    let downstream_speed := row_speed + river_speed
    let upstream_speed := row_speed - river_speed
    let downstream_time := d / downstream_speed
    let upstream_time := d / upstream_speed
    downstream_time + upstream_time = total_time ∧ d = 2.25 :=
by
  sorry

end rowing_distance_l438_438041


namespace farthest_vertex_of_dilated_square_l438_438343

noncomputable theory

open Real

-- Definitions for the problem conditions
def center : (ℝ × ℝ) := (5, 5)
def area : ℝ := 16
def dilation_center : (ℝ × ℝ) := (0, 0)
def scale_factor : ℝ := 3

-- The statement to be proved
theorem farthest_vertex_of_dilated_square :
  let side_length := sqrt area,
      vertices := [(center.1 - side_length / 2, center.2 - side_length / 2),
                   (center.1 - side_length / 2, center.2 + side_length / 2),
                   (center.1 + side_length / 2, center.2 + side_length / 2),
                   (center.1 + side_length / 2, center.2 - side_length / 2)],
      dilated_vertices := vertices.map (λ p, (scale_factor * p.1, scale_factor * p.2)),
      distances := dilated_vertices.map (λ p, sqrt (p.1^2 + p.2^2))
  in dilated_vertices distances.index_of_max = (21, 21) :=
by
  sorry

end farthest_vertex_of_dilated_square_l438_438343


namespace problem_proof_l438_438166

noncomputable def problem_statement : Prop :=
  sqrt (45 + 20 * sqrt 5) + sqrt (45 - 20 * sqrt 5) = 10

theorem problem_proof : problem_statement :=
by
  sorry

end problem_proof_l438_438166


namespace no_valid_arrangement_l438_438025

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438025


namespace num_subsets_containing_a_l438_438237

theorem num_subsets_containing_a : 
  let S := {1, 2, 3, 4, 5, 6, 7, 'a'} in
  card {T : set S // 'a' ∈ T} = 128 :=
by sorry

end num_subsets_containing_a_l438_438237


namespace find_2a_2b_2c_2d_l438_438644

open Int

theorem find_2a_2b_2c_2d (a b c d : ℤ) 
  (h1 : a - b + c = 7) 
  (h2 : b - c + d = 8) 
  (h3 : c - d + a = 4) 
  (h4 : d - a + b = 1) : 
  2*a + 2*b + 2*c + 2*d = 20 := 
sorry

end find_2a_2b_2c_2d_l438_438644


namespace expectation_of_transformed_binomial_l438_438229

def binomial_expectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

def linear_property_of_expectation (a b : ℚ) (E_ξ : ℚ) : ℚ :=
  a * E_ξ + b

theorem expectation_of_transformed_binomial (ξ : ℚ) :
  ξ = binomial_expectation 5 (2/5) →
  linear_property_of_expectation 5 2 ξ = 12 :=
by
  intros h
  rw [h]
  unfold linear_property_of_expectation binomial_expectation
  sorry

end expectation_of_transformed_binomial_l438_438229


namespace range_a_l438_438460

noncomputable theory

def R := ℝ

def op (x y: R) : R := x / (2 - y)

theorem range_a (a : R) :
  (∀ x : R, (x - a) / (x - (a+1)) ≥ 0 → (-2 : R) < x ∧ x < 2) →
  -2 < a ∧ a ≤ 1 :=
by
  sorry

end range_a_l438_438460


namespace f_at_3_l438_438938

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition_1 : ∀ x : ℝ, f(x + 2) = f(x + 1) - f(x)
axiom f_condition_2 : f(1) = log 3 - log 2
axiom f_condition_3 : f(2) = log 3 + log 5

theorem f_at_3 : f(3) = 1 :=
by
  sorry

end f_at_3_l438_438938


namespace tom_speed_first_part_l438_438689

-- Definitions of conditions in Lean
def total_distance : ℕ := 20
def distance_first_part : ℕ := 10
def speed_second_part : ℕ := 10
def average_speed : ℚ := 10.909090909090908
def distance_second_part := total_distance - distance_first_part

-- Lean statement to prove the speed during the first part of the trip
theorem tom_speed_first_part (v : ℚ) :
  (distance_first_part / v + distance_second_part / speed_second_part) = total_distance / average_speed → v = 12 :=
by
  intro h
  sorry

end tom_speed_first_part_l438_438689


namespace minimum_value_l438_438298

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 6) :
  37.5 ≤ (9 / x + 25 / y + 49 / z) :=
sorry

end minimum_value_l438_438298


namespace opposite_of_23_l438_438369

theorem opposite_of_23 : ∃ (x : ℤ), 23 + x = 0 ∧ x = -23 :=
by
  use -23
  split
  { linarith }
  { refl }

end opposite_of_23_l438_438369


namespace simplify_fraction_imaginary_l438_438493

theorem simplify_fraction_imaginary (i : ℂ) (hi : i = complex.I) : (5 + 14 * i) / (2 + 3 * i) = 4 + i :=
by
  sorry

end simplify_fraction_imaginary_l438_438493


namespace number_A_is_10_15_times_number_C_l438_438937

variables (A B C : ℝ)

-- Defining the conditions given in the problem
def condition1 := A * 10^(-8) = B * 10^3
def condition2 := B * 10^(-2) = C * 10^2

-- Stating the theorem using the defined conditions.
theorem number_A_is_10_15_times_number_C 
  (A B C : ℝ) 
  (h1 : A * 10^(-8) = B * 10^3) 
  (h2 : B * 10^(-2) = C * 10^2) :
  A = 10^15 * C :=
begin
  sorry
end

end number_A_is_10_15_times_number_C_l438_438937


namespace factorial_div_l438_438405

theorem factorial_div (n : ℕ) (h : n = 2012): (n! / (n-1)!) = n :=
by
  sorry

end factorial_div_l438_438405


namespace magnitude_b_angle_theta_l438_438196

open Real

variables (a b : ℝ → ℝ → ℝ) -- Considering \overrightarrow{a} and \overrightarrow{b} as real-valued vector functions
variables (u v : ℝ) -- Dummy variables for vectors

-- Conditions
axiom a_nonzero : ∀ u, a u ≠ 0
axiom b_nonzero : ∀ v, b v ≠ 0
axiom norm_a : ‖a u‖ = 1
axiom dot_product_condition1 : (a u - b v) • (a u + b v) = 3 / 4

-- Question 1: Prove |b| = 1/2
theorem magnitude_b : ‖b v‖ = 1 / 2 :=
sorry

-- Additional condition for part 2
axiom dot_product_condition2 : (a u) • (b v) = -1 / 4

-- Angle theta between vector a and a + 2b
theorem angle_theta : ∃ θ : ℝ, θ = 60 ∧
  (cos θ = ((a u) • (a u + 2 * (b v))) / (‖a u‖ * ‖a u + 2 * (b v)‖)) :=
sorry

end magnitude_b_angle_theta_l438_438196


namespace focal_length_of_ellipse_l438_438360

theorem focal_length_of_ellipse : 
  ∀ (x y : ℝ), 2 * x^2 + 3 * y^2 = 1 → 
  let a := sqrt (1/2),
      b := sqrt (1/3),
      c := sqrt (a^2 - b^2) in 
  2 * c = sqrt(6) / 3 :=
by
  intros x y h
  have a2 : (sqrt (1/2))^2 = 1 / 2 := by sorry
  have b2 : (sqrt (1/3))^2 = 1 / 3 := by sorry
  have c : sqrt ((sqrt (1/2))^2 - (sqrt (1/3))^2) = sqrt(1/2 - 1/3) := by sorry
  have focal_length : 2 * sqrt(1/2 - 1/3) = sqrt(6) / 3 := by sorry
  exact focal_length

end focal_length_of_ellipse_l438_438360


namespace mike_ride_miles_l438_438620

theorem mike_ride_miles (M : ℝ) (annie_miles : ℝ) :
  let mike_cost := 2.50 + 0.25 * M 
  let annie_cost := 2.50 + 5.00 + 0.25 * annie_miles
  annie_miles = 26 
  → mike_cost = annie_cost 
  → M = 36 := 
by
  intros
  have h1 : mike_cost = 2.50 + 0.25 * M := rfl
  have h2 : annie_cost = 2.50 + 5.00 + 0.25 * 26 := by
    simp only [annie_miles, annie_miles = 26]
  sorry

end mike_ride_miles_l438_438620


namespace hyperbola_eccentricity_is_2_l438_438203

variable {a b : ℝ}

-- Conditions
variables (C : {x // x = (λ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)}) 
variables (a_pos : 0 < a) (b_pos : 0 < b)

-- The point symmetric to F with respect to one asymptote lies on the other asymptote
def is_symmetric_point_lying_on_asymptote (F : ℝ × ℝ) (m : ℝ) : Prop :=
  let F_sym := (m, -b/a * m) in  (∃ m, F_sym = (m, -b/a * m))

-- Eccentricity definition
def eccentricity (a b : ℝ) : ℝ := (Real.sqrt (a^2 + b^2)) / a

-- Lean proof statement
theorem hyperbola_eccentricity_is_2
  (F : ℝ × ℝ) 
  (hF : F = (-Real.sqrt (a^2 + b^2), 0))
  (symm_cond : is_symmetric_point_lying_on_asymptote F (Real.sqrt(a^2 + b^2) / 2)):
  eccentricity a b = 2 := by
  sorry

end hyperbola_eccentricity_is_2_l438_438203


namespace angleAOQ_90_l438_438440

variable {A B C M E F P Q O : Type}
variable [Triangle := Mathlib.Geometry.Triangle]
variable [Midpoint M B C]
variable [Reflection E M AC]
variable [Reflection F M AB]
variable (Line_P_intersection : ∃ P : Type, Line P B F ∧ Line P C E)
variable [Equal QA QM]
variable [Angle90 Q A P]
variable [Circumcenter O P E F]
variable [Angle90Proof : Angle O A Q = 90]

theorem angleAOQ_90 :
  Angle A O Q = 90 := by
  exact Angle90Proof.sorry

end angleAOQ_90_l438_438440


namespace percentage_error_divide_instead_of_multiply_l438_438709

theorem percentage_error_divide_instead_of_multiply (x : ℝ) : 
  let correct_result := 5 * x 
  let incorrect_result := x / 10 
  let error := correct_result - incorrect_result 
  let percentage_error := (error / correct_result) * 100 
  percentage_error = 98 :=
by
  sorry

end percentage_error_divide_instead_of_multiply_l438_438709


namespace exist_three_elements_l438_438289

theorem exist_three_elements {n : ℕ} (hn_pos : n > 0) (hn_squarefree : ∀ p : ℕ, p * p ∣ n → ¬(prime p)) 
  (S : finset ℕ) (hS_sub : ∀ x ∈ S, 1 ≤ x ∧ x ≤ n) (hS_size : 2 * S.card ≥ n) :
  ∃ a b c ∈ S, (a * b) % n = c % n :=
by 
  sorry

end exist_three_elements_l438_438289


namespace even_function_a_value_l438_438561

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 + (a - 2) * x + a^2 - a - 2 = (a + 1) * x^2 - (a - 2) * x + a^2 - a - 2) → a = 2 := 
by sorry

end even_function_a_value_l438_438561


namespace gloria_money_left_l438_438236

theorem gloria_money_left 
  (cost_of_cabin : ℕ) (cash : ℕ)
  (num_cypress_trees num_pine_trees num_maple_trees : ℕ)
  (price_per_cypress_tree price_per_pine_tree price_per_maple_tree : ℕ)
  (money_left : ℕ)
  (h_cost_of_cabin : cost_of_cabin = 129000)
  (h_cash : cash = 150)
  (h_num_cypress_trees : num_cypress_trees = 20)
  (h_num_pine_trees : num_pine_trees = 600)
  (h_num_maple_trees : num_maple_trees = 24)
  (h_price_per_cypress_tree : price_per_cypress_tree = 100)
  (h_price_per_pine_tree : price_per_pine_tree = 200)
  (h_price_per_maple_tree : price_per_maple_tree = 300)
  (h_money_left : money_left = (num_cypress_trees * price_per_cypress_tree + 
                                num_pine_trees * price_per_pine_tree + 
                                num_maple_trees * price_per_maple_tree + 
                                cash) - cost_of_cabin)
  : money_left = 350 :=
by
  sorry

end gloria_money_left_l438_438236


namespace perimeter_of_semicircle_l438_438712

noncomputable def pi_approx : Real := 3.14159

theorem perimeter_of_semicircle (radius : Real) (h : radius = 35) : 
  let diameter := 2 * radius
  let half_circumference := pi_approx * radius
  70 + half_circumference ≈ 179.96 := 
by 
  have : diameter = 2 * 35 := by rw [h]; simp
  have : half_circumference = pi_approx * 35 := by rw [h]; simp 
  sorry

end perimeter_of_semicircle_l438_438712


namespace exists_circle_through_A_B_inside_ω_l438_438614

open EuclideanGeometry

variables {ω : Circle} (O : Point) (A B : Point)

-- Assume ω is a circle with center O 
-- A and B are two points inside ω
def points_inside_circle (ω : Circle) (A B : Point) : Prop :=
  inside A ω ∧ inside B ω

-- Problem statement: 
-- Prove there exists a circle passing through A and B that is entirely contained within ω
theorem exists_circle_through_A_B_inside_ω 
  (O : Point) (ω : Circle) (A B : Point) 
  (h₀ : ω.center = O) 
  (h₁ : inside A ω) 
  (h₂ : inside B ω) :
  ∃ (ω' : Circle), passes_through ω' A ∧ passes_through ω' B ∧ inside ω' ω :=
sorry

end exists_circle_through_A_B_inside_ω_l438_438614


namespace number_of_divisors_of_3003_l438_438831

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

theorem number_of_divisors_of_3003 : count_divisors 3003 = 16 :=
by
  sorry

end number_of_divisors_of_3003_l438_438831


namespace walking_rate_on_escalator_l438_438439

theorem walking_rate_on_escalator 
  (escalator_speed person_time : ℝ) 
  (escalator_length : ℝ) 
  (h1 : escalator_speed = 12) 
  (h2 : person_time = 15) 
  (h3 : escalator_length = 210) 
  : (∃ v : ℝ, escalator_length = (v + escalator_speed) * person_time ∧ v = 2) :=
by
  use 2
  rw [h1, h2, h3]
  sorry

end walking_rate_on_escalator_l438_438439


namespace production_rate_after_decrease_l438_438437

theorem production_rate_after_decrease :
  ∀ x : ℝ,
  (∀ cogs_per_hour_initial cogs_produced_initial cogs_produced_additional total_cogs t_initial t_additional t_total average_output,
    cogs_per_hour_initial = 90 ∧
    cogs_produced_initial = 60 ∧
    cogs_produced_additional = 60 ∧
    total_cogs = 120 ∧
    average_output = 72 ∧
    t_initial = cogs_produced_initial / cogs_per_hour_initial ∧
    t_additional = cogs_produced_additional / x ∧
    t_total = total_cogs / average_output ∧
    t_initial + t_additional = t_total
    → x = 60) :=
begin
  intros x cogs_per_hour_initial cogs_produced_initial cogs_produced_additional total_cogs t_initial t_additional t_total average_output,
  assume h,

  -- Extracting and using assumptions from h
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h_rest,
  cases h_rest with h5 h_rest,
  cases h_rest with h6 h_rest,
  cases h_rest with h7 h_rest,
  cases h_rest with h8 h_rest,

  -- Applying the derived equations from the problem statement
  have h_t_initial : t_initial = 2 / 3,
  { rw [h2, h1], norm_num },

  have h_t_total : t_total = 5 / 3,
  { rw [h5, h4], norm_num },

  -- Setting up the key equation
  have h_key : 2 / 3 + 60 / x = 5 / 3,
  { rw [h8, h_t_initial, h7, h6, h_t_total], norm_num },

  -- Solving the key equation for x
  have h_x : x = 60,
  { field_simp at h_key, linarith },

  -- Concluding the proof
  exact h_x,
end


end production_rate_after_decrease_l438_438437


namespace find_sin_alpha_l438_438823

open Real

theorem find_sin_alpha
  (α β γ : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hγ : 0 < γ ∧ γ < π / 2)
  (h1 : cos α = tan β)
  (h2 : cos β = tan γ)
  (h3 : cos γ = tan α) :
  sin α = sqrt(2) / 2 := 
by
  sorry

end find_sin_alpha_l438_438823


namespace same_function_as_identity_l438_438093

def f1 (x : ℝ) : ℝ := abs x
def f2 (x : ℝ) : ℝ := Real.sqrt (x^2)
def f3 (x : ℝ) : ℝ := (Real.sqrt x)^2
def f4 (x : ℝ) : ℝ := Real.cbrt (x^3)

theorem same_function_as_identity : f4 = id :=
by
  sorry

end same_function_as_identity_l438_438093


namespace volume_original_pyramid_l438_438798

-- Conditions
def side_length_base : ℝ := 6
def edge_length_to_apex : ℝ := 10
def edge_length_larger_tetrahedron : ℝ := 2 * edge_length_to_apex

-- Volume calculation for a larger tetrahedron
def volume_tetrahedron (edge_length : ℝ) : ℝ :=
  (Math.sqrt 2 * edge_length ^ 3) / 12

-- Volume = one fourth of larger tetrahedron
theorem volume_original_pyramid :
  (volume_tetrahedron edge_length_larger_tetrahedron) / 4 = 500 * (Math.sqrt 2) / 3 :=
by
  sorry

end volume_original_pyramid_l438_438798


namespace closest_integer_to_sum_l438_438156

theorem closest_integer_to_sum : 
  let S := 500 * (∑ n in Finset.range 4999, 1 / (n + 2)^2 - 1);
  abs (S - 375) < 0.1 := 
by 
  sorry

end closest_integer_to_sum_l438_438156


namespace magnitude_vec_sum_l438_438918

open Real

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem magnitude_vec_sum
    (a b : ℝ × ℝ)
    (h_angle : ∃ θ, θ = 150 * (π / 180) ∧ cos θ = cos (5 * π / 6))
    (h_norm_a : ‖a‖ = sqrt 3)
    (h_norm_b : ‖b‖ = 2) :
  ‖(2 * a.1 + b.1, 2 * a.2 + b.2)‖ = 2 :=
  by
  sorry

end magnitude_vec_sum_l438_438918


namespace inverse_reciprocal_value_l438_438243

theorem inverse_reciprocal_value :
  ∀ f : ℝ → ℝ, (∀ x, f x = 25 / (4 + 2 * x)) → (∃ x, f x = 5) →
  (∀ g : ℝ → ℝ, (∀ y, f (g y) = y ∧ g (f y) = y) → let x := g 5 in 1 / x = 2) :=
by
  intro f hf hfx g hg
  let x := g 5
  have hx : f x = 5 := (hg (f x)).mpr rfl
  sorry

end inverse_reciprocal_value_l438_438243


namespace slope_tangent_line_at_P_is_10_l438_438373

-- Definitions for the function y = x^3 - 2x + 2
def f (x : ℝ) : ℝ := x^3 - 2*x + 2

-- Definition for the point P(2, 6)
def P : ℝ × ℝ := (2, 6)

-- Lean statement to prove the slope of the tangent line at point (2, 6) is 10
theorem slope_tangent_line_at_P_is_10 :
  let df := deriv f in
  df 2 = 10 :=
by {
  sorry,
}

end slope_tangent_line_at_P_is_10_l438_438373


namespace find_abc_sqrt_f_inequality_range_of_m_l438_438527

variables {x x1 x2 : ℝ} {a b c m : ℝ}

-- (1) Prove values of a, b, c
theorem find_abc (h1 : ∀ x : ℝ, f(-x) = f(x))
                (h2 : ∀ y : ℝ, (x^2 + 2 * b * x + c) = a * (x + 1)^2)
                (h3 : {l} == setOf (f(x) = a * (x + 1)^2)) :
                a = (1 / 2) ∧ b = 0 ∧ c = 1 :=
  sorry

-- (2) Prove the inequality
theorem sqrt_f_inequality (h4 : f(x) = x^2 + 2 * b * x + c)
                          (h5 : b = 0) (h6 : c = 1) :
                          ∀ x ∈ (set.Icc (-2 : ℝ) 2), sqrt (f(x)) ≤ (((sqrt 5) - 1) / 2) * abs x + 1 :=
  sorry

-- (3) Range of m 
theorem range_of_m (h7 : f(x) = x^2 + 2 * b * x + c) 
                   (h8 : b = 0) (h9 : c = 1) 
                   (h10 : g(x) = sqrt (f(x)) + sqrt (f(2 - x)))
                   (h11 : x1 ∈ (set.Icc 0 2) ∧ x2 ∈ (set.Icc 0 2)) 
                   (h12 : | g(x1) - g(x2) | ≥ m) :
                   m ≤ (sqrt 5 + 1 - 2 * sqrt 2) :=
  sorry

end find_abc_sqrt_f_inequality_range_of_m_l438_438527


namespace cards_not_divisible_by_7_l438_438014

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438014


namespace complex_magnitude_problem_l438_438812

theorem complex_magnitude_problem :
  let i := Complex.I in
  let z1 := 1 + 2 * i in
  let z2 := 1 - i in
  let S := z1^12 - z2^12 in
  |S| = 15689 := by
  -- Complex computations go here
  sorry

end complex_magnitude_problem_l438_438812


namespace total_customers_in_line_l438_438127

-- Define the number of people behind the first person
def people_behind := 11

-- Define the total number of people in line
def people_in_line : Nat := people_behind + 1

-- Prove the total number of people in line is 12
theorem total_customers_in_line : people_in_line = 12 :=
by
  sorry

end total_customers_in_line_l438_438127


namespace S5_is_81_l438_438504

noncomputable def a_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * 3^(n-2)

noncomputable def S (n : ℕ) : ℕ :=
  ∑ i in finset.range n, a_sequence (i + 1)

theorem S5_is_81 : S 5 = 81 :=
by {
  sorry
}

end S5_is_81_l438_438504


namespace directrix_of_parabola_l438_438357

-- Define the condition given in the problem
def parabola_eq (x y : ℝ) : Prop := x^2 = 2 * y

-- Define the directrix equation property we want to prove
theorem directrix_of_parabola (x : ℝ) :
  (∃ y : ℝ, parabola_eq x y) → (∃ y : ℝ, y = -1 / 2) :=
by sorry

end directrix_of_parabola_l438_438357


namespace square_area_l438_438084

theorem square_area (a b c : ℝ) (h1 : a = 6.1) (h2 : b = 8.2) (h3 : c = 9.7)
  (h_eq_perimeter : 4 * s = a + b + c) :
  s * s = 36 := 
by {
  have h_perimeter : a + b + c = 24, {
    rw [h1, h2, h3],
    norm_num,
  },
  have h_s : s = 6, {
    linarith [h_eq_perimeter, h_perimeter],
  },
  rw [h_s],
  norm_num,
}

end square_area_l438_438084


namespace annual_interest_rate_equivalent_l438_438319

noncomputable def quarterly_compound_rate : ℝ := 1 + 0.02
noncomputable def annual_compound_amount : ℝ := quarterly_compound_rate ^ 4

theorem annual_interest_rate_equivalent : 
  (annual_compound_amount - 1) * 100 = 8.24 := 
by
  sorry

end annual_interest_rate_equivalent_l438_438319


namespace range_k_fx_greater_than_ln_l438_438900

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem range_k (k : ℝ) : 0 ≤ k ∧ k ≤ Real.exp 1 ↔ ∀ x : ℝ, f x ≥ k * x := 
by 
  sorry

theorem fx_greater_than_ln (t : ℝ) (x : ℝ) : t ≤ 2 ∧ 0 < x → f x > t + Real.log x :=
by
  sorry

end range_k_fx_greater_than_ln_l438_438900


namespace option_B_correct_l438_438774

theorem option_B_correct (x m : ℕ) : (x^3)^m / (x^m)^2 = x^m := sorry

end option_B_correct_l438_438774


namespace range_of_AB_l438_438200

-- Definitions
def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def circle (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem declaration
theorem range_of_AB (x_a y_a x_b y_b : ℝ) 
  (h_a : parabola x_a y_a) 
  (h_b : circle x_b y_b) 
  (h_ab : distance x_a y_a x_b y_b = |ab|) :
  ∃ (l : ℝ), l = Real.sqrt 15 - 1 ∧ ∀ (d : ℝ), d = |ab| → Real.sqrt 15 - 1 ≤ d :=
sorry

end range_of_AB_l438_438200


namespace no_n_gt_1_divisibility_l438_438147

theorem no_n_gt_1_divisibility (n : ℕ) (h : n > 1) : ¬ (3 ^ (n - 1) + 5 ^ (n - 1)) ∣ (3 ^ n + 5 ^ n) :=
by
  sorry

end no_n_gt_1_divisibility_l438_438147


namespace part1_parity_part2_monotonicity_part3_num_of_zeros_l438_438905

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 1 / x

-- Proof of Parity
theorem part1_parity (a x : ℝ) : f a (-x) = - f a x :=
by
  sorry

-- Proof of Monotonicity
theorem part2_monotonicity (a x1 x2 : ℝ) (ha_pos : a > 0) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 < 1 / real.sqrt a) :
  f a x2 < f a x1 :=
by
  sorry

-- Proof of Number of Zeros
def g (x : ℝ) : ℝ := f 1 (real.exp x) - 18

theorem part3_num_of_zeros : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 = 0 ∧ g x2 = 0 :=
by
  sorry

end part1_parity_part2_monotonicity_part3_num_of_zeros_l438_438905


namespace problem_statement_l438_438518

theorem problem_statement (x y : ℝ) (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 18) :
  17 * x ^ 2 + 24 * x * y + 17 * y ^ 2 = 532 :=
by
  sorry

end problem_statement_l438_438518


namespace number_of_integers_satisfying_conditions_l438_438368

theorem number_of_integers_satisfying_conditions : 
  let satisfies_conditions (n : ℕ) :=
    (n % 2 = 1) ∧
    (n % 3 = 2) ∧
    (n % 4 = 3) ∧
    (n % 5 = 4) ∧
    (n % 6 = 5)
  in 
  card ((finset.range 2007).filter satisfies_conditions) = 32 :=
by sorry

end number_of_integers_satisfying_conditions_l438_438368


namespace find_magnitude_condition_l438_438548

variable (m : ℝ)

def vector_a := (m, -1 : ℝ)
def vector_b := (1, 1 : ℝ)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_magnitude_condition :
  magnitude (vector_sub vector_a vector_b) = magnitude vector_a + magnitude vector_b → 
  m = -1 := 
by
  sorry

end find_magnitude_condition_l438_438548


namespace sum_of_reciprocal_powers_l438_438113

theorem sum_of_reciprocal_powers:
  (∑ a in Finset.Icc 1 ∞, ∑ b in Finset.Icc (a+1) ∞, ∑ c in Finset.Icc (b+1) ∞, 
    (1:ℝ) / (2^a * 3^b * 7^c)) = 1/600 := 
by
  sorry

end sum_of_reciprocal_powers_l438_438113


namespace total_cookies_baked_l438_438740

theorem total_cookies_baked (num_members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ) :
  (num_members = 100) → (sheets_per_member = 10) → (cookies_per_sheet = 16) → 
  (num_members * sheets_per_member * cookies_per_sheet = 16000) :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  assumption,
end

end total_cookies_baked_l438_438740


namespace time_to_empty_l438_438705

-- Definitions for the conditions
def rate_fill_no_leak (R : ℝ) := R = 1 / 2 -- Cistern fills in 2 hours without leak
def effective_fill_rate (R L : ℝ) := R - L = 1 / 4 -- Effective fill rate when leaking
def remember_fill_time_leak (R L : ℝ) := (R - L) * 4 = 1 -- 4 hours to fill with leak

-- Main theorem statement
theorem time_to_empty (R L : ℝ) (h1 : rate_fill_no_leak R) (h2 : effective_fill_rate R L)
  (h3 : remember_fill_time_leak R L) : (1 / L = 4) :=
by
  sorry

end time_to_empty_l438_438705


namespace triangle_tan_l438_438274

noncomputable def tan_P (P Q R : Type) [real] [has_ang Q = 90] [real QR = 4] [real PR = 5] : real :=
tan (angle P)

theorem triangle_tan (P Q R : Type) [angle Q = 90] [real QR = 4] [real PR = 5] : tan (angle P) = 4 / 3 :=
sorry

end triangle_tan_l438_438274


namespace time_to_pass_trolley_l438_438707

/--
Conditions:
- Length of the train = 110 m
- Speed of the train = 60 km/hr
- Speed of the trolley = 12 km/hr

Prove that the time it takes for the train to pass the trolley completely is 5.5 seconds.
-/
theorem time_to_pass_trolley :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (trolley_speed_kmh : ℝ),
    train_length = 110 →
    train_speed_kmh = 60 →
    trolley_speed_kmh = 12 →
  train_length / ((train_speed_kmh + trolley_speed_kmh) * (1000 / 3600)) = 5.5 :=
by
  intros
  sorry

end time_to_pass_trolley_l438_438707


namespace Antoinette_weight_l438_438779

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end Antoinette_weight_l438_438779


namespace ali_baba_max_72_coins_l438_438433

-- Definitions and conditions
def piles : ℕ := 10
def coins_per_pile : ℕ := 10
def total_coins : ℕ := piles * coins_per_pile
def ali_baba_takes_three_piles (coins : list ℕ) : ℕ := sorry -- function to define selection of three piles with most coins by Ali Baba

-- Hypotheses (conditions)
variable (steps : ℕ → list ℕ)
variable (bandit_rearrangement : list ℕ → list ℕ)

-- Goal
theorem ali_baba_max_72_coins : 
  ∃ (actions : ℕ → list ℕ), 
  ∀ s: ℕ, s < piles →
  ali_baba_takes_three_piles (actions s) = 72 :=
begin
   sorry
end

end ali_baba_max_72_coins_l438_438433


namespace jellybeans_red_l438_438071

-- Define the individual quantities of each color of jellybean.
def b := 14
def p := 26
def o := 40
def pk := 7
def y := 21
def T := 237

-- Prove that the number of red jellybeans is 129.
theorem jellybeans_red : T - (b + p + o + pk + y) = 129 := by
  -- (optional: you can include intermediate steps if needed, but it's not required here)
  sorry

end jellybeans_red_l438_438071


namespace problem_possible_l438_438173

theorem problem_possible (n : ℕ) (hn : n ≥ 4) :
  (∃ (a : Finₓ (n + 1) → ℕ), (Finₓ (n + 1)).erase n).perm (List.range (n + 1)) ∧
  (∀ i, (1 ≤ i) → (i < n) → abs ((a i) - (a (i + 1))) ≠ abs ((a (i + 1)) - (a (i + 2)))) ∧
  (abs ((a (n)) - (a 1))) ∈ {1, 2, ..., n}
  ↔ n % 4 = 0 ∨ n % 4 = 3 := by
  sorry

end problem_possible_l438_438173


namespace crayons_initially_l438_438625

theorem crayons_initially (crayons_left crayons_lost : ℕ) (h_left : crayons_left = 134) (h_lost : crayons_lost = 345) :
  crayons_left + crayons_lost = 479 :=
by
  sorry

end crayons_initially_l438_438625


namespace no_valid_odd_numbers_cube_l438_438280

theorem no_valid_odd_numbers_cube : 
  ∀ (numbers : Fin 8 → ℕ), 
  (∀ i, numbers i % 2 = 1) ∧ 
  (∀ i, 1 ≤ numbers i ∧ numbers i ≤ 600) → 
  (∀ u v : Fin 8, adjacent vertices u v → common_divisor (numbers u) (numbers v) > 1) → 
  ¬ (∃ u v : Fin 8, ¬adjacent vertices u v ∧ common_divisor (numbers u) (numbers v) > 1) :=
sorry

end no_valid_odd_numbers_cube_l438_438280


namespace mod_210_123456789012_l438_438835

theorem mod_210_123456789012 :
  let M := 123456789012 in
  let N := 210 in
  let a := 2 in
  let b := 3 in
  let c := 5 in
  let d := 7 in
  M % a = 0 ∧ M % b = 0 ∧ M % c = 2 ∧ M % d = 3 → 
  M % N = 17 :=
by
  intros M N a b c d h,
  dsimp at *,
  sorry

end mod_210_123456789012_l438_438835


namespace domain_of_symmetric_function_l438_438214

-- Definitions corresponding to the given conditions
def symmetric_about (f g : ℝ → ℝ) (line : ℝ → ℝ) : Prop :=
∀ x, f (line x) = g x

def inverse_function (f g : ℝ → ℝ) (dom codom : set ℝ) : Prop :=
∀ y ∈ codom, ∃ x ∈ dom, f x = y ∧ g y = x

-- The given functions and their domains
def g (x : ℝ) : ℝ := x^2 + 1
def g_domain : set ℝ := {x | x < 0}

-- The statement to be proven
theorem domain_of_symmetric_function :
  (symmetric_about f g (λ y, y) ∧ inverse_function g f g_domain (set.Ioi 1)) →
  ∀ y, f y = (1 : ℝ) := sorry

end domain_of_symmetric_function_l438_438214


namespace Jessie_weight_l438_438590

theorem Jessie_weight (c l w : ℝ) (hc : c = 27) (hl : l = 101) : c + l = w ↔ w = 128 := by
  sorry

end Jessie_weight_l438_438590


namespace volume_of_reflected_tetrahedron_l438_438631

noncomputable def volume_of_polyhedron_formed (A B C D A' B' C' D' O : Point) (h : regular_tetrahedron A B C D O 1) (h' : reflect_about_center A A' B B' C C' D D' O) : ℝ :=
  by
    -- Prove that the volume is 2^{-3/2}.
    sorry

theorem volume_of_reflected_tetrahedron (A B C D A' B' C' D' O : Point) (h : regular_tetrahedron A B C D O 1) (h' : reflect_about_center A A' B B' C C' D D' O) :
    volume_of_polyhedron_formed A B C D A' B' C' D' O h h' = 2 ^ (-3/2) :=
  by
    -- The proof goes here.
    sorry

end volume_of_reflected_tetrahedron_l438_438631


namespace problem_statement_l438_438883

noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 1
noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem problem_statement :
  (h (-x) = h x) :=
by
  sorry

end problem_statement_l438_438883


namespace solve_for_Theta_l438_438824

-- Define the two-digit number representation condition
def fourTheta (Θ : ℕ) : ℕ := 40 + Θ

-- Main theorem statement
theorem solve_for_Theta (Θ : ℕ) (h1 : 198 / Θ = fourTheta Θ + Θ) (h2 : 0 < Θ ∧ Θ < 10) : Θ = 4 :=
by
  sorry

end solve_for_Theta_l438_438824


namespace area_of_triangle_CDN_l438_438961

noncomputable def right_triangle : Type :=
{x y : ℝ // x^2 + y^2 = (real.sqrt (3^2 + 4^2))^2}

theorem area_of_triangle_CDN :
  let A := (0, 0)
  let B := (3, 0)
  let C := (0, 4)
  let D := (2, 0)
  let BC := real.sqrt (3^2 + 4^2)
  let M := ((3 / 2), 2)
  in 4 = 1 / 2 * abs (2 * (2 - 0) + (3 / 2) * (0 - 4) + 3 * (0 - 2)) :=
by {
  sorry
}

end area_of_triangle_CDN_l438_438961


namespace integral_of_f_l438_438105

-- Define the function f
def f (x : ℝ) : ℝ := x + Real.sqrt (1 - x^2)

-- Define the problem statement to prove
theorem integral_of_f :
  ∫ x in -1..1, f x = Real.pi / 2 :=
by
  sorry

end integral_of_f_l438_438105


namespace max_value_of_expression_l438_438611

theorem max_value_of_expression (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 3) :
  (xy / (x + y + 1) + xz / (x + z + 1) + yz / (y + z + 1)) ≤ 1 :=
sorry

end max_value_of_expression_l438_438611


namespace total_length_of_items_l438_438076

theorem total_length_of_items :
  ∃ (rubber pen pencil marker ruler scissors : ℝ),
    pencil = 12 ∧
    pen = pencil - 2 ∧
    pen = rubber + 3 ∧
    ruler = 3 * rubber ∧
    ruler = pen * 1.2 ∧
    marker = ruler / 2 ∧
    marker = (pen + rubber + pencil) / 3 ∧
    scissors = pencil * 0.75 ∧
    (rubber + pen + pencil + marker + ruler + scissors) = 69.5 :=
begin
  sorry
end

end total_length_of_items_l438_438076


namespace min_presses_to_open_castle_l438_438252

-- Define the number of possible buttons and length of the code
def num_buttons : ℕ := 3
def code_length : ℕ := 3

-- Define the total number of possible codes
def total_codes : ℕ := num_buttons ^ code_length

-- Define a function that calculates the minimum length of the sequence needed to ensure all codes appear
def min_sequence_length (n : ℕ) : ℕ := n - 2

-- Theorem stating the minimum number of button presses required
theorem min_presses_to_open_castle : 
  ∃ (n : ℕ), n >= 29 ∧ min_sequence_length n >= total_codes :=
by
  use 29
  simp [min_sequence_length, total_codes, code_length, num_buttons]
  -- Proof omitted
  sorry

end min_presses_to_open_castle_l438_438252


namespace lecture_room_configuration_l438_438415

theorem lecture_room_configuration (m n : ℕ) (boys_per_row girls_per_column unoccupied_chairs : ℕ) :
    boys_per_row = 6 →
    girls_per_column = 8 →
    unoccupied_chairs = 15 →
    (m * n = boys_per_row * m + girls_per_column * n + unoccupied_chairs) →
    (m = 71 ∧ n = 7) ∨
    (m = 29 ∧ n = 9) ∨
    (m = 17 ∧ n = 13) ∨
    (m = 15 ∧ n = 15) ∨
    (m = 11 ∧ n = 27) ∨
    (m = 9 ∧ n = 69) :=
by
  intros h1 h2 h3 h4
  sorry

end lecture_room_configuration_l438_438415


namespace no_valid_arrangement_l438_438027

theorem no_valid_arrangement : 
    ¬∃ (l : List ℕ), l ~ [1, 2, 3, 4, 5, 6, 8, 9] ∧
    ∀ (i : ℕ), i < l.length - 1 → (10 * l.nthLe i (by sorry) + l.nthLe (i + 1) (by sorry)) % 7 = 0 := 
sorry

end no_valid_arrangement_l438_438027


namespace equal_sides_in_regular_ngon_l438_438578

theorem equal_sides_in_regular_ngon (n : ℕ) (a : Fin n → ℝ) 
  (h_angles : ∀ (i j k: Fin n), ∠ (a i) (a j) (a k) = (n - 2) * 180 / n)
  (h_lengths : ∀ i j : Fin n, i < j → a i ≥ a j) :
  ∀ i j : Fin n, a i = a j :=
by
  sorry

end equal_sides_in_regular_ngon_l438_438578


namespace max_ahn_achieve_max_ahn_achieve_attained_l438_438091

def is_two_digit_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ahn_achieve :
  ∀ (n : ℕ), is_two_digit_integer n → 3 * (300 - n) ≤ 870 := 
by sorry

theorem max_ahn_achieve_attained :
  3 * (300 - 10) = 870 := 
by norm_num

end max_ahn_achieve_max_ahn_achieve_attained_l438_438091


namespace beadshop_wednesday_profit_l438_438737

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end beadshop_wednesday_profit_l438_438737


namespace find_x_y_l438_438865

theorem find_x_y (x y : ℝ)
  (h1 : (x - 1) ^ 2003 + 2002 * (x - 1) = -1)
  (h2 : (y - 2) ^ 2003 + 2002 * (y - 2) = 1) :
  x + y = 3 :=
sorry

end find_x_y_l438_438865


namespace Mary_hourly_rate_l438_438619

theorem Mary_hourly_rate (R : ℝ) (h1 : ∀ t : ℝ, t ≤ 50)
  (h2 : ∀ (t : ℝ) (ht : t ≤ 20), 20 * R)
  (h3 : ∀ (t : ℝ) (ht : t > 20 ∧ t ≤ 50), (t - 20) * 1.25 * R + 20 * R )
  (h4 : 20 * R + 30 * 1.25 * R ≤ 460 ) : R ≤ 8 :=
by {
  sorry
}

end Mary_hourly_rate_l438_438619


namespace nicole_answers_correctly_l438_438323

theorem nicole_answers_correctly :
  ∀ (C K N : ℕ), C = 17 → K = C + 8 → N = K - 3 → N = 22 :=
by
  intros C K N hC hK hN
  sorry

end nicole_answers_correctly_l438_438323


namespace problem_statement_l438_438862

def imaginary_unit (i : ℂ) : Prop :=
  i * i = -1

def complex_sum (n : ℕ) (i : ℂ) : ℂ :=
  (Finset.range n).sum (λ k, i ^ k)

def complex_value (i : ℂ) : ℂ :=
  complex_sum 2023 i / (1 - i)

def conjugate_value (z : ℂ) : ℂ :=
  conj z

theorem problem_statement (i : ℂ) (z : ℂ) (hz : imaginary_unit i) :
  z = complex_value i → 
  let z_conj := conjugate_value z in
  z_conj.re < 0 ∧ z_conj.im > 0 :=
by
  sorry

end problem_statement_l438_438862


namespace number_of_two_digit_integers_l438_438514

def tens_digits : Set ℕ := {2, 3}
def units_digits : Set ℕ := {4, 7}

theorem number_of_two_digit_integers : 
  (tens_digits.card * units_digits.card) = 4 := 
by 
  sorry

end number_of_two_digit_integers_l438_438514


namespace unknown_angles_are_80_l438_438257

theorem unknown_angles_are_80 (y : ℝ) (h1 : y + y + 200 = 360) : y = 80 :=
by
  sorry

end unknown_angles_are_80_l438_438257


namespace max_value_of_expression_l438_438928

theorem max_value_of_expression :
  ∃ x y z : ℝ, (x - y + z - 1 = 0) ∧ (x * y + 2 * z^2 - 6 * z + 1 = 0) ∧
  (∀ a b c : ℝ, (a - b + c - 1 = 0) ∧ (a * b + 2 * c^2 - 6 * c + 1 = 0) → (x - 1)^2 + (y + 1)^2 ≥ (a - 1)^2 + (b + 1)^2) ∧
  ((x - 1)^2 + (y + 1)^2 = 11) :=
begin
  sorry
end

end max_value_of_expression_l438_438928


namespace remainder_pow_700_eq_one_l438_438699

theorem remainder_pow_700_eq_one (number : ℤ) (h : number ^ 700 % 100 = 1) : number ^ 700 % 100 = 1 :=
  by
  exact h

end remainder_pow_700_eq_one_l438_438699


namespace cube_remainder_l438_438385

theorem cube_remainder (n : ℤ) (h : n % 13 = 5) : (n^3) % 17 = 6 :=
by
  sorry

end cube_remainder_l438_438385


namespace part2_l438_438191

noncomputable def seq (n: ℕ) : ℝ := 2 * (n + 1)

noncomputable def Sn (n: ℕ) : ℝ := (n + 1)^2 + (1 / 2) * seq n

noncomputable def A_n (n: ℕ) : ℝ :=
  (List.range (n + 1)).prod (λ i => 1 - (1 / seq i))

noncomputable def f (x : ℝ) (a_n : ℝ) : ℝ := x + (a_n / (2 * x))

noncomputable def inequality_holds (n: ℕ) (a: ℝ) : Prop :=
  A_n n * Real.sqrt ((seq n) + 1) < f a (seq n) - ((seq n) + 3) / (2 * a)

theorem part2 (a: ℝ) : 
  (a > Real.sqrt 3 ∨ a < - (Real.sqrt 3 / 2) ∧ a ≠ 0) → 
  ∀ n: ℕ, n > 0 → inequality_holds n a := 
  sorry

end part2_l438_438191


namespace james_total_time_l438_438284

def time_to_play_main_game : ℕ := 
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let before_tutorial_time := download_time + install_time + update_time + account_time + internet_issues_time
  let tutorial_time := before_tutorial_time * 3
  before_tutorial_time + tutorial_time

theorem james_total_time : time_to_play_main_game = 220 := by
  sorry

end james_total_time_l438_438284


namespace mod_210_123456789012_l438_438834

theorem mod_210_123456789012 :
  let M := 123456789012 in
  let N := 210 in
  let a := 2 in
  let b := 3 in
  let c := 5 in
  let d := 7 in
  M % a = 0 ∧ M % b = 0 ∧ M % c = 2 ∧ M % d = 3 → 
  M % N = 17 :=
by
  intros M N a b c d h,
  dsimp at *,
  sorry

end mod_210_123456789012_l438_438834


namespace ones_digit_of_sum_of_powers_l438_438384

theorem ones_digit_of_sum_of_powers (n : ℕ) :
  let s := (finset.range n).sum (λ x, (x + 1)^2025) in
  n = 2023 → (s % 10) = 6 :=
by
  sorry

end ones_digit_of_sum_of_powers_l438_438384


namespace evaluate_expression_l438_438495

theorem evaluate_expression (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := 
sorry

end evaluate_expression_l438_438495


namespace exercise_books_count_l438_438710

theorem exercise_books_count (pencils pens exercise_books : ℕ) 
  (h1 : pencils = 120) 
  (h2 : pencils : pens : exercise_books = 10 : 2 : 3) : 
  exercise_books = 36 := 
by 
  sorry

end exercise_books_count_l438_438710


namespace dragon_jewels_l438_438742

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l438_438742


namespace nice_sequence_max_length_l438_438612

theorem nice_sequence_max_length (n : ℕ) (hn : 1 ≤ n) :
  ∃ (a : ℕ → ℝ), (a 0 + a 1 = -1 / n) ∧
  (∀ k ≥ 1, (a (k - 1) + a k) * (a k + a (k + 1)) = a (k - 1) + a (k + 1)) ∧
  ∀ b, b = n → ¬∃ (a' : ℕ → ℝ), (a' 0 + a' 1 = -1 / n) ∧
  (∀ k ≥ 1, (a' (k - 1) + a' k) * (a' k + a' (k + 1)) = a' (k - 1) + a' (k + 1)) ∧
  ∃ m, m = b + 1 := sorry

end nice_sequence_max_length_l438_438612


namespace sum_first_10_common_elements_l438_438842

/-- Definition of arithmetic progression term -/
def arith_term (n : ℕ) : ℕ := 4 + 3 * n

/-- Definition of geometric progression term -/
def geom_term (k : ℕ) : ℕ := 10 * 2 ^ k

/-- Verify if two terms are common elements -/
def is_common_element (n k : ℕ) : Prop := arith_term n = geom_term k

/-- Equivalence proof of sum of first 10 common elements -/
theorem sum_first_10_common_elements : 
  Σ (n k : ℕ) (H : is_common_element n k), (arith_term n) = 3495250 :=
begin
  sorry
end

end sum_first_10_common_elements_l438_438842


namespace fraction_eval_l438_438813

theorem fraction_eval :
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25) :=
sorry

end fraction_eval_l438_438813


namespace sales_tax_percentage_l438_438078

noncomputable def original_price : ℝ := 200
noncomputable def discount : ℝ := 0.25 * original_price
noncomputable def sale_price : ℝ := original_price - discount
noncomputable def total_paid : ℝ := 165
noncomputable def sales_tax : ℝ := total_paid - sale_price

theorem sales_tax_percentage : (sales_tax / sale_price) * 100 = 10 := by
  sorry

end sales_tax_percentage_l438_438078


namespace total_number_of_members_l438_438580

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end total_number_of_members_l438_438580


namespace not_possible_arrange_cards_l438_438019

/-- 
  Zhenya had 9 cards numbered from 1 to 9, and lost the card numbered 7. 
  It is not possible to arrange the remaining 8 cards in a sequence such that every pair of adjacent 
  cards forms a number divisible by 7.
-/
theorem not_possible_arrange_cards : 
  ∀ (cards : List ℕ), cards = [1, 2, 3, 4, 5, 6, 8, 9] → 
  ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ (i : ℕ), i < 7 → ((perm.get! i) * 10 + (perm.get! (i + 1))) % 7 = 0) :=
by {
  sorry
}

end not_possible_arrange_cards_l438_438019


namespace multiples_of_7_between_50_and_200_l438_438924

theorem multiples_of_7_between_50_and_200 : 
  ∃ n, n = 21 ∧ ∀ k, (k ≥ 50 ∧ k ≤ 200) ↔ ∃ m, k = 7 * m := sorry

end multiples_of_7_between_50_and_200_l438_438924


namespace angle_acd_is_60_degrees_l438_438266

theorem angle_acd_is_60_degrees
  (A B C D : Type)
  (dist : A → B → C → ℝ)
  (angle : A → B → C → ℝ)
  (h1 : dist A B = dist A C)
  (h2 : dist A B = dist A D)
  (h3 : dist A D = dist B D)
  (h4 : ∠BAC = ∠CBD) : ∠ACD = 60 :=
begin
  sorry
end

end angle_acd_is_60_degrees_l438_438266


namespace sharing_cookies_l438_438783

theorem sharing_cookies (batches : ℕ) (dozens_per_batch : ℕ) (cookies_per_dozen : ℕ) (cookies_per_person : ℕ) : 
  batches = 4 → dozens_per_batch = 2 → cookies_per_dozen = 12 → cookies_per_person = 6 → 
  (batches * dozens_per_batch * cookies_per_dozen) / cookies_per_person = 16 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end sharing_cookies_l438_438783


namespace Carmen_s_total_money_made_is_42_l438_438793

def money_made_from_green_house := 3 * 4
def money_made_from_yellow_house := 2 * 3.5 + 1 * 5
def money_made_from_brown_house := 9 * 2

def total_money_made := money_made_from_green_house +
                        money_made_from_yellow_house +
                        money_made_from_brown_house

theorem Carmen_s_total_money_made_is_42 : total_money_made = 42 := 
by 
  sorry

end Carmen_s_total_money_made_is_42_l438_438793


namespace length_of_train_l438_438042

def train_speed : Real := 60 -- speed in km/hr
def time_crossing_pole : Real := 15 -- time in seconds
def kms_to_meters (km : Real) : Real := km * 1000 -- conversion function from kilometers to meters

theorem length_of_train :
  let time_in_hours := time_crossing_pole / (60 * 60) in
  let distance_in_km := train_speed * time_in_hours in
  let length_of_train_in_meters := kms_to_meters distance_in_km in
  length_of_train_in_meters = 250 :=
by
  sorry

end length_of_train_l438_438042


namespace triple_root_at_zero_l438_438362

open Polynomial

variable {R : Type*} [CommRing R] (a b c d m n : R)

noncomputable def given_polynomial : Polynomial R :=
  X^7 - 9 * X^6 + 27 * X^5 + a * X^4 + b * X^3 + c * X^2 + d * X - m * X - n

theorem triple_root_at_zero 
  (h1 : (given_polynomial a b c d m n) = 
  (X - p)^2 * (X - q)^2 * (X - r)^3)
  (h2 : p ≠ q) (h3 : p ≠ r) (h4 : q ≠ r) (h5 : (r = 0)) :
  r = 0 :=
begin
  sorry
end

end triple_root_at_zero_l438_438362


namespace simplify_complex_expr_l438_438637

theorem simplify_complex_expr : ∀ i : ℂ, i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 - i) = 14 :=
by 
  intro i 
  intro h
  sorry

end simplify_complex_expr_l438_438637


namespace integral_circle_area_solve_function_volunteers_tasks_arrangement_max_min_value_sum_l438_438056

-- Problem 1: Prove that the integral of the function equals 2π
theorem integral_circle_area :
  ∫ x in (-2:ℝ)..2, sqrt (4 - x^2) = 2 * Real.pi :=
sorry

-- Problem 2: Prove the solution to the functional equation
theorem solve_function :
  ∀ f : ℝ → ℝ, (f x + ∫ a in 0..1, f(a)) = 2 * x → f x = (2 * x - 1/2) :=
sorry

-- Problem 3: Prove the number of arrangements of tasks
theorem volunteers_tasks_arrangement :
  (C (4) (2) * A (3) (3)) = 36 :=
sorry

-- Problem 4: Prove max and min values sum for given function
theorem max_min_value_sum :
  ∀ (f : ℝ → ℝ) (t : ℝ), (∀ x ∈ Icc (-t) t, f x = x * ln (exp x + 1) - 1/2 * x^2 + 3) →
  let M := max (λ x, f x) (Icc (-t) t),
      m := min (λ x, f x) (Icc (-t) t)
  in M + m = 6 :=
sorry

end integral_circle_area_solve_function_volunteers_tasks_arrangement_max_min_value_sum_l438_438056


namespace polynomial_factor_11_l438_438462

theorem polynomial_factor_11 (d : ℝ) : 
  (∀ x : ℝ, (x^4 + 3*x^3 + 2*x^2 + d*x + 15) = (λ x, Q(x) := 0)) → Q(-3) = 0 → d = 11 := 
by
  sorry

end polynomial_factor_11_l438_438462


namespace Jessica_has_3_dozens_l438_438984

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l438_438984


namespace no_valid_arrangement_l438_438010

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438010


namespace all_numbers_1984_impossible_all_but_one_1984_impossible_all_but_two_1984_possible_l438_438054

-- Define the conditions
def initial_blackboard : ℕ → ℕ := λ _, 0

def rotated_blackboard (n_rotate : ℕ) (blackboard : ℕ → ℕ) : ℕ → ℕ :=
  λ x, blackboard x + (x - 1) % 12 + 1

def after_n_rotations (n : ℕ) : ℕ → ℕ :=
  fin.foldl (rotated_blackboard) initial_blackboard (fin.finset_range n)

-- Proof problems based on the conditions and conclusions
theorem all_numbers_1984_impossible : 
  ¬ ∃ n, ∀ i : ℕ, i ≤ 11 → after_n_rotations n i = 1984 :=
sorry

theorem all_but_one_1984_impossible : 
  ¬ ∃ n (j : ℕ), j ≤ 11 ∧ ∀ i : ℕ, i ≠ j → i ≤ 11 → after_n_rotations n i = 1984 :=
sorry

theorem all_but_two_1984_possible : 
  ∃ n (j k : ℕ), j ≠ k ∧ j ≤ 11 ∧ k ≤ 11 ∧ ∀ i : ℕ, i ≠ j ∧ i ≠ k → i ≤ 11 → after_n_rotations n i = 1984 :=
sorry

end all_numbers_1984_impossible_all_but_one_1984_impossible_all_but_two_1984_possible_l438_438054


namespace sum_of_reversed_base8_base15_numbers_l438_438480

theorem sum_of_reversed_base8_base15_numbers :
  let numbers := {n : ℕ | n < 64 ∧
                         (∀ a₀ a₁ : ℕ, a₀ < 8 ∧ a₁ < 8 ∧ a₁ = 2 * a₀ →
                         n = 8*a₁ + a₀ ∧ n = 15*a₀ + a₁)} in
  (∑ n in numbers, n) = 102 :=
sorry

end sum_of_reversed_base8_base15_numbers_l438_438480


namespace sequence_count_length_14_l438_438454

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 0
| (n+2) := a n + b n
and b : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := a (n+1) + b n

theorem sequence_count_length_14 : a 14 + b 14 = 172 := by
  sorry

end sequence_count_length_14_l438_438454


namespace integral_eval_l438_438463

noncomputable def definiteIntegral := ∫ x in 0..1, (Real.exp x - 2 * x)

theorem integral_eval : definiteIntegral = Real.exp 1 - 2 :=
by
  sorry

end integral_eval_l438_438463


namespace larger_number_is_1629_l438_438651

theorem larger_number_is_1629 (x y : ℕ) (h1 : y - x = 1360) (h2 : y = 6 * x + 15) : y = 1629 := 
by 
  sorry

end larger_number_is_1629_l438_438651


namespace find_a_l438_438186

variable {z : ℂ} (a : ℝ)

theorem find_a (hz : |z| = 2) (h : (z - a)^2 = a) : a = 2 := 
sorry

end find_a_l438_438186


namespace no_valid_cube_labeling_l438_438282

theorem no_valid_cube_labeling :
  ¬ ∃ (c : fin 8 → ℕ), 
    (∀ i j, i ≠ j → i < j → 1 ≤ c i ∧ c i ≤ 600 ∧ 1 ≤ c j ∧ c j ≤ 600 ∧ c i % 2 = 1 ∧ c j % 2 = 1 ∧ (adjacent i j → gcd (c i) (c j) > 1) ∧ (¬adjacent i j → gcd (c i) (c j) = 1)) :=
sorry

-- Define adjacency for a cube
def adjacent (i j : fin 8) : Prop :=
  <define adjacency relation on vertices of the cube somehow>

end no_valid_cube_labeling_l438_438282


namespace balance_rearrangement_vowels_at_end_l438_438555

theorem balance_rearrangement_vowels_at_end : 
  let vowels := ['A', 'A', 'E'];
  let consonants := ['B', 'L', 'N', 'C'];
  (Nat.factorial 3 / Nat.factorial 2) * Nat.factorial 4 = 72 :=
by
  sorry

end balance_rearrangement_vowels_at_end_l438_438555


namespace combine_sum_l438_438879

def A (n m : Nat) : Nat := n.factorial / (n - m).factorial
def C (n m : Nat) : Nat := n.factorial / (m.factorial * (n - m).factorial)

theorem combine_sum (n m : Nat) (hA : A n m = 272) (hC : C n m = 136) : m + n = 19 := by
  sorry

end combine_sum_l438_438879


namespace sum_of_roots_range_l438_438901

theorem sum_of_roots_range (f : ℝ → ℝ)
  (h₁ : ∀ x ≤ 0, f x = -x^2 - 2x + 1)
  (h₂ : ∀ x > 0, f x = |real.log x / real.log (2 : ℝ)|)
  (h_k : 0 < k ∧ k < 2)
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, f x1 = k ∧ f x2 = k ∧ f x3 = k ∧ f x4 = k ∧ x1 ≠ x2 ∧ x1 ≤ 0 ∧ x2 ≤ 0 ∧ x3 ≠ x4 ∧ x3 > 0 ∧ x4 > 0 ∧
                     (x1 + x2 = -2) ∧ (1 < x4 ∧ x4 ≤ 2) ∧ (2 ≤ x3 + x4 ∧ x3 + x4 ≤ 2.5)) :
  0 ≤ (x1 + x2 + x3 + x4) ∧ (x1 + x2 + x3 + x4) ≤ 0.5 :=
sorry

end sum_of_roots_range_l438_438901


namespace y_paisa_for_each_rupee_x_l438_438087

theorem y_paisa_for_each_rupee_x (p : ℕ) (x : ℕ) (y_share total_amount : ℕ) 
  (h₁ : y_share = 2700) 
  (h₂ : total_amount = 10500) 
  (p_condition : (130 + p) * x = total_amount) 
  (y_condition : p * x = y_share) : 
  p = 45 := 
by
  sorry

end y_paisa_for_each_rupee_x_l438_438087


namespace five_digit_repeated_digit_percentage_l438_438242

theorem five_digit_repeated_digit_percentage :
  let total_numbers := 90000
  let repeated_digit_count := 90000 - 9 * 9 * 8 * 7 * 6
  ∃ x : ℝ, abs(x - (repeated_digit_count / total_numbers * 100)) < 0.05  :=
by
  let total_numbers := 90000
  let non_repeated_numbers := 9 * 9 * 8 * 7 * 6
  let repeated_digit_count := total_numbers - non_repeated_numbers
  let x := repeated_digit_count / total_numbers * 100
  exists x
  have : abs(x - 69.8) < 0.05 := sorry
  exact this

end five_digit_repeated_digit_percentage_l438_438242


namespace cards_not_divisible_by_7_l438_438011

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438011


namespace longest_side_length_quadrilateral_l438_438800

theorem longest_side_length_quadrilateral :
  (∀ (x y : ℝ),
    (x + y ≤ 4) ∧
    (2 * x + y ≥ 3) ∧
    (x ≥ 0) ∧
    (y ≥ 0)) →
  (∃ d : ℝ, d = 4 * Real.sqrt 2) :=
by sorry

end longest_side_length_quadrilateral_l438_438800


namespace length_of_broken_line_and_sum_of_areas_l438_438512

noncomputable theory

def isosceles_right_triangle_area (a : ℝ) : ℝ :=
  (a^2 / 2)

def isosceles_right_triangle_perimeter (a : ℝ) : ℝ :=
  a * (Real.sqrt 2 + 1)

theorem length_of_broken_line_and_sum_of_areas (a : ℝ) :
  let S := isosceles_right_triangle_perimeter a
  let T := a^2 / 2
  S = a * (Real.sqrt 2 + 1) ∧ T = a^2 / 2 :=
by
  sorry

end length_of_broken_line_and_sum_of_areas_l438_438512


namespace ivanov_family_net_worth_l438_438716

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l438_438716


namespace find_analytical_expression_find_min_value_find_max_value_l438_438206

noncomputable def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c

variables (f : ℝ → ℝ)
variables (hf_quad : quadratic_function f)
variables (hf_value : f (-1) = 2)
variables (hf_deriv : deriv f 0 = 0)
variables (hf_integral : ∫ x in 0..1, f x = -2)

theorem find_analytical_expression : f = λ x, 6 * x^2 - 4 :=
by { sorry }

theorem find_min_value : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x ≥ -4 :=
by { sorry }

theorem find_max_value : ∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x ≤ 2 :=
by { sorry }

end find_analytical_expression_find_min_value_find_max_value_l438_438206


namespace find_k_l438_438181

variables {k : ℝ} {a b c d : EuclideanSpace ℝ (Fin 3)}

-- Conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom angle_ab : ∃ θ : ℝ, θ = real.pi / 3 ∧ real.angle a b = θ
axiom vec_c : c = 2 • a + 3 • b
axiom vec_d : d = k • a - b
axiom perp_cd : c ⬝ d = 0

-- Theorem to prove
theorem find_k : k = 14 / 5 := 
sorry

end find_k_l438_438181


namespace sum_of_primes_dividing_expression_l438_438452

theorem sum_of_primes_dividing_expression :
  ∑ p in (Finset.filter (λ p, p ≥ 5 ∧ (p ∣ ((p + 3)^(p - 3) + (p + 5)^(p - 5)))) (Finset.range 2814).attach), p = 2813 :=
sorry

end sum_of_primes_dividing_expression_l438_438452


namespace win_sector_area_l438_438065

theorem win_sector_area (r : ℝ) (P_win : ℝ) (P_bonus_lose : ℝ) :
  r = 8 →
  P_win = 1 / 4 →
  P_bonus_lose = 1 / 8 →
  (P_win * (π * r^2) = 16 * π) :=
by
  intros hr hw hl
  rw [hr, hw, hl]
  sorry

end win_sector_area_l438_438065


namespace triangle_side_inequality_l438_438589

theorem triangle_side_inequality 
  (a b c l_c h_a : ℝ)
  (ha : 0 < a)
  (hc : 0 < c)
  (a_le_b : a ≤ b) 
  (a_le_c : a ≤ c) 
  (c_ge_a_b : c ≥ a + b)
  (S : ℝ) 
  (hS1 : S = (1 / 2) * a * h_a)
  (hS2 : S = (1 / 2) * l_c * (a + b) * Real.sin ((Real.acos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2)) :
  l_c ≤ h_a := by
  sorry

end triangle_side_inequality_l438_438589


namespace train_stops_for_4_minutes_per_hour_l438_438472

/-- Given that the speed of the train excluding stoppages is 45 kmph,
and the speed of the train including stoppages is 42 kmph,
prove that the train stops for 4 minutes per hour. -/
theorem train_stops_for_4_minutes_per_hour 
  (speed_without_stops : ℕ := 45) 
  (speed_with_stops : ℕ := 42) :
  let speed_difference := speed_without_stops - speed_with_stops,
      speed_diff_per_minute := speed_difference / 60,
      speed_without_stops_per_minute := speed_without_stops / 60,
      stoppage_time_per_minute := speed_diff_per_minute / speed_without_stops_per_minute in
  stoppage_time_per_minute * 60 = 4 :=
by
  /- Problem conditions -/
  let speed_without_stops := 45
  let speed_with_stops := 42

  /- Definitions based on the conditions -/
  let speed_difference := speed_without_stops - speed_with_stops
  let speed_diff_per_minute := (speed_difference : ℚ) / 60  -- converting to rational for precision
  let speed_without_stops_per_minute := (speed_without_stops : ℚ) / 60
  let stoppage_time_per_minute := speed_diff_per_minute / speed_without_stops_per_minute

  /- Assertion to prove -/
  have h1 : stoppage_time_per_minute * 60 = 4 := sorry

  exact h1

end train_stops_for_4_minutes_per_hour_l438_438472


namespace cos_gamma_proof_l438_438601

-- Define the given conditions
variables (α β γ : ℝ)
variable (Q : ℝ × ℝ × ℝ)
variable (x y z : ℝ)
variable (cosα : ℝ := 4/5)
variable (cosβ : ℝ := 1/2)
variable (cosγ : ℝ)

-- The hypotheses
axiom cos_alpha_definition : cosα = 4 / 5
axiom cos_beta_definition : cosβ = 1 / 2
axiom directional_cosines : cosα^2 + cosβ^2 + cosγ^2 = 1

-- The theorem to prove
theorem cos_gamma_proof : cosγ = real.sqrt (11) / 10 :=
sorry

end cos_gamma_proof_l438_438601


namespace Isabella_paint_area_l438_438980

def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 11
def bedroom1_height : ℕ := 9

def bedroom2_length : ℕ := 13
def bedroom2_width : ℕ := 12
def bedroom2_height : ℕ := 9

def unpaintable_area_per_bedroom : ℕ := 70

theorem Isabella_paint_area :
  let wall_area (length width height : ℕ) := 2 * (length * height) + 2 * (width * height)
  let paintable_area (length width height : ℕ) := wall_area length width height - unpaintable_area_per_bedroom
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height =
  1520 := 
by
  sorry

end Isabella_paint_area_l438_438980


namespace range_of_b_for_inverse_function_l438_438567

-- Statement: Define the function and prove the range of b
theorem range_of_b_for_inverse_function (b : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 2, let fx := x^2 - b*x + 2 in monotone_on (f' x) (Set.Icc (-1 : ℝ) 2)) : 
  b ≤ -2 ∨ b ≥ 4 :=
by {
  sorry -- Proof steps are omitted as per instructions
}

end range_of_b_for_inverse_function_l438_438567


namespace largest_n_inequality_l438_438825

theorem largest_n_inequality (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
    sqrt (a / (b + c + d + e)) +
    sqrt (b / (a + c + d + e)) +
    sqrt (c / (a + b + d + e)) +
    sqrt (d / (a + b + c + e)) +
    sqrt (e / (a + b + c + d)) > 2 :=
sorry

end largest_n_inequality_l438_438825


namespace geometric_progression_exists_lambda_l438_438545

-- Conditions
variables {a_n : ℕ → ℝ} {b_n : ℕ → ℝ} 

-- Additional property as per problem description for the sequence {a_n}
lemma root_eq_relation (n : ℕ) (hn : n > 0) :
  ∃ a _a, a_n n = a ∧ a_n (n+1) = _a ∧ a * _a = b_n n ∧ a + _a = 2^n := sorry

-- Prove that the sequence {a_n - 1/3 * 2^n} is geometric
theorem geometric_progression (n : ℕ) (hn : n > 0) :
  ∃ r, ∀ k, (a_n k - 1/3 * 2^k) = r * (a_n 1 - 1/3 * 2^1) := sorry

-- Prove existence and range of lambda such that bn > λ Sn for all n
theorem exists_lambda (n : ℕ) (hn : n > 0) (S_n : ℕ → ℝ) 
  (H_Sn : S_n = λ k, ∑ i in finset.range k, a_n i) : 
  ∃ λ, λ < 1 ∧ ∀ n > 0, b_n n > λ * S_n n := sorry

end geometric_progression_exists_lambda_l438_438545


namespace ivanov_family_net_worth_l438_438721

theorem ivanov_family_net_worth :
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  in net_worth = 2300000 := 
by
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  show net_worth = 2300000 from sorry

end ivanov_family_net_worth_l438_438721


namespace intersection_points_on_lines_l438_438250

theorem intersection_points_on_lines {R Q : Type*} [euclidean_space R Q]
  (A1 A2 A3 A4 B1 B2 B3 B4 : Q)
  (h_inscribedR : ∀ A1 A2 A3 A4, is_cyclic A1 A2 A3 A4) 
  (h_inscribedQ : ∀ B1 B2 B3 B4, is_cyclic B1 B2 B3 B4)
  (h_rectangle : is_rectangle A1 A2 A3 A4)
  (h_perpendicular : ∀ (a : ℝ) (b : ℝ), a.perp b)
  (intersections : ∀ (X Y : ℕ), intersection_points X Y) :
    ∃ (L1 L2 L3 L4 : line Q),
      (∀ i ∈ {1, 2, 3, 4}, ∃ (P1 P2 P3 P4 : Q),
        all_points_on_lines {P1, P2, P3, P4} L_i) ∧
      (∀ T : tangents, ∃ (C1 C2 : Q),
        chord_contact_passes_through C1 C2) := sorry

end intersection_points_on_lines_l438_438250


namespace first_payment_amount_l438_438738

-- The number of total payments
def total_payments : Nat := 65

-- The number of the first payments
def first_payments : Nat := 20

-- The number of remaining payments
def remaining_payments : Nat := total_payments - first_payments

-- The extra amount added to the remaining payments
def extra_amount : Int := 65

-- The average payment
def average_payment : Int := 455

-- The total amount paid over the year
def total_amount_paid : Int := average_payment * total_payments

-- The variable we want to solve for: amount of each of the first 20 payments
variable (x : Int)

-- The equation for total amount paid
def total_payments_equation : Prop :=
  20 * x + 45 * (x + 65) = 455 * 65

-- The theorem stating the amount of each of the first 20 payments
theorem first_payment_amount : x = 410 :=
  sorry

end first_payment_amount_l438_438738


namespace iodine_electron_count_l438_438976

noncomputable def iodine_atomic_number : ℕ := 53
noncomputable def iodine_atomic_mass : ℕ := 131
noncomputable def iodine_neutron_count : ℕ := iodine_atomic_mass - iodine_atomic_number

theorem iodine_electron_count : iodine_atomic_number = 53 → iodine_atomic_mass = 131 → iodine_neutron_count = 78 → iodine_atomic_number = 53 :=
by {
  intros h1 h2 h3,
  exact h1,
} sorry

end iodine_electron_count_l438_438976


namespace sum_first_2009_terms_l438_438044

-- Define the sequence U_n
def U (n : ℕ) (a r : ℝ) : ℝ :=
  if n % 4 = 1 ∨ n % 4 = 2 then a * r^(n - 1) else -a * r^(n - 1) 

-- Define the sum S(n) of the first n terms of the sequence U
def S (n : ℕ) (a r : ℝ) : ℝ := ∑ i in Finset.range n + 1, U i a r

theorem sum_first_2009_terms (a r : ℝ) (ha : 0 < a) (hr : 0 < r) :
  S 2009 a r = a * (1 + r - r^2009 + r^2010) / (1 + r^2) := 
sorry

end sum_first_2009_terms_l438_438044


namespace solve_for_x_l438_438560

theorem solve_for_x (x : ℝ) (h : sqrt ((3 / x) + 1) = 5 / 3) : x = 27 / 16 :=
sorry

end solve_for_x_l438_438560


namespace dora_2017th_number_l438_438098

def is_divisible (m n : Nat) : Prop := ∃ k : Nat, m = n * k

def nth_number_dora (k : Nat) : Nat :=
  (List.range (3 * k)).filter (λ n, is_divisible n 2 ∨ is_divisible n 3).nth! (k - 1)

theorem dora_2017th_number : nth_number_dora 2017 = 3026 := 
  sorry

end dora_2017th_number_l438_438098


namespace loss_percentage_second_venture_l438_438768

theorem loss_percentage_second_venture 
  (investment_total : ℝ)
  (investment_each : ℝ)
  (profit_percentage_first_venture : ℝ)
  (total_return_percentage : ℝ)
  (L : ℝ) 
  (H1 : investment_total = 25000) 
  (H2 : investment_each = 16250)
  (H3 : profit_percentage_first_venture = 0.15)
  (H4 : total_return_percentage = 0.08)
  (H5 : (investment_total * total_return_percentage) = ((investment_each * profit_percentage_first_venture) - (investment_each * L))) :
  L = 0.0269 := 
by
  sorry

end loss_percentage_second_venture_l438_438768


namespace cannot_find_k_l438_438723

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

def d (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ k, k > 0 ∧ n % k = 0).card

theorem cannot_find_k (n : ℕ) :
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p^(q-1)) →
  ¬ ∃ k : ℕ, is_perfect_square ((nat.iterate d k) n) :=
sorry

end cannot_find_k_l438_438723


namespace coloring_integers_with_conditions_l438_438629

theorem coloring_integers_with_conditions :
  ∃ c : ℕ → {1, 2, 3},
  (∀ n : ℕ, ∀ x : ℕ, 2^n ≤ x ∧ x < 2^(n + 1) → c x = c (2^n)) ∧
  (∀ x y z : ℕ, (x + y = z^2 ∧ ¬ (x = 2 ∧ y = 2 ∧ z = 2)) → ¬ (c x = c y ∧ c y = c z ∧ c x = c z)) :=
by 
sorry

end coloring_integers_with_conditions_l438_438629


namespace minimum_value_l438_438529

noncomputable def min_value_expression (a b : ℝ) : ℝ := 
  (1 / (1 + a) + 4 / (2 + b))

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 7) :
  min_value_expression a b = (13 + 4 * real.sqrt 3) / 14 :=
sorry

end minimum_value_l438_438529


namespace tangent_line_at_3_tangent_line_through_origin_l438_438907

noncomputable def f : ℝ → ℝ := λ x, x^3 + x - 16

def f_prime (x : ℝ) : ℝ := 3 * x^2 + 1

example : ∀ (x : ℝ), f_prime x = (f' x) := by sorry

theorem tangent_line_at_3 :
  tangent_line_at_3 := (f 3 = 14) ∧ (f_prime 3 = 28) ∧ (∀ x, y = 28 * x - 70) := by sorry

theorem tangent_line_through_origin :
  ∃ (x0 y0 : ℝ), (l : ℝ → ℝ) (l(0) = 0) ∧ (x0 = -2) ∧ (y0 = -26) ∧ (l = (13 * x)) := by sorry

end tangent_line_at_3_tangent_line_through_origin_l438_438907


namespace find_a_l438_438171

-- Define the sum of digits function
def S (n : ℕ) : ℕ := (n.digits 10).sum

-- Define the main theorem
theorem find_a : ∃ a : ℕ, a > 0 ∧ (∀ N : ℕ, ∃ n : ℕ, n > N ∧ S(n) - S(n + a) = 2018) ∧ a = 7 := 
by { use 7, split, linarith, split, intros N, use 10^(225 + N) - 1, sorry, linarith }

end find_a_l438_438171


namespace sum_first_10_common_elements_l438_438841

/-- Definition of arithmetic progression term -/
def arith_term (n : ℕ) : ℕ := 4 + 3 * n

/-- Definition of geometric progression term -/
def geom_term (k : ℕ) : ℕ := 10 * 2 ^ k

/-- Verify if two terms are common elements -/
def is_common_element (n k : ℕ) : Prop := arith_term n = geom_term k

/-- Equivalence proof of sum of first 10 common elements -/
theorem sum_first_10_common_elements : 
  Σ (n k : ℕ) (H : is_common_element n k), (arith_term n) = 3495250 :=
begin
  sorry
end

end sum_first_10_common_elements_l438_438841


namespace minimal_possible_value_of_E_l438_438666

variables {a b c : ℝ} {n : ℕ}

theorem minimal_possible_value_of_E
  (hpos_a : 0 < a)
  (hpos_b : 0 < b)
  (hpos_c : 0 < c)
  (hsum : a + b + c = 1) :
  ∃ E : ℝ, E = (∑ cyc in finset.univ, (λ x, 
                    ( x.1 ^ (-n : ℤ) + x.2 ) / (1 - x.1)))
                  ((a,b,c)) = (3 ^ (n+2) + 3) / 2 := 
sorry

end minimal_possible_value_of_E_l438_438666


namespace max_distance_points_l438_438393

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 2 * x

def A : ℝ × ℝ := (-1, 0)
def D : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1 / Real.sqrt 2, 1 / Real.sqrt 2)
def C : ℝ × ℝ := (1 / Real.sqrt 2, -1 / Real.sqrt 2)

theorem max_distance_points :
  ∀ (x ∈ Icc (-1 : ℝ) 1) (y ∈ Icc (-1 : ℝ) 1),
    y ≠ x → 
    (f x, x) = A ∨ (f x, x) = B ∨ (f x, x) = C ∨ (f x, x) = D ∧ 
    (f y, y) = A ∨ (f y, y) = B ∨ (f y, y) = C ∨ (f y, y) = D → 
    dist (f x, x) (f y, y) ≤ 2 :=
sorry

end max_distance_points_l438_438393


namespace find_a_l438_438177

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l438_438177


namespace complex_number_triangle_l438_438448

noncomputable def eq_triangle_magnitude (p q r : ℂ) (h : p - q ≠ 0 ∧ q - r ≠ 0 ∧ r - p ≠ 0) : Prop :=
(|p - q| = 24 ∧ |q - r| = 24 ∧ |r - p| = 24)

theorem complex_number_triangle (p q r : ℂ)
  (h_eq_triangle : eq_triangle_magnitude p q r (by {
    split;
    norm_num })) 
  (h_sum_magnitude : |p + q + r| = 48) :
  |p * q + p * r + q * r| = 768 :=
sorry

end complex_number_triangle_l438_438448


namespace problem1_problem2_l438_438109

-- Problem 1
theorem problem1 : ((2 / 3 - 1 / 12 - 1 / 15) * -60) = -31 := by
  sorry

-- Problem 2
theorem problem2 : ((-7 / 8) / ((7 / 4) - 7 / 8 - 7 / 12)) = -3 := by
  sorry

end problem1_problem2_l438_438109


namespace cards_not_divisible_by_7_l438_438013

/-- Definition of the initial set of cards -/
def initial_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- Final set after losing the card with number 7 -/
def final_set : Finset ℕ := initial_set \ {7}

/-- Predicate to check if a pair of numbers forms a 2-digit number divisible by 7 -/
def divisible_by_7 (a b : ℕ) : Prop := (10 * a + b) % 7 = 0

/-- Main theorem stating the problem -/
theorem cards_not_divisible_by_7 :
  ¬ ∃ l : List ℕ, l ~ final_set.toList ∧
    (∀ (x y ∈ l), x ≠ y → List.is_adjacent x y → divisible_by_7 x y) :=
sorry

end cards_not_divisible_by_7_l438_438013


namespace hybrid_travel_cost_of_electricity_proof_l438_438775

def cost_of_electricity_per_km : ℝ := 0.3
def distance : ℝ := 100
def max_cost : ℝ := 50
def cost_electric_per_km := 0.3
def cost_oil_per_km := cost_electric_per_km + 0.5
def min_distance_electric := 60

theorem hybrid_travel (y : ℝ) (h_y : 0.3 * y + 0.8 * (100 - y) ≤ 50) : y ≥ 60 := 
begin
    sorry
end

theorem cost_of_electricity_proof :
    ∃ x d, (80 / (x + 0.5) = 30 / x) ∧ x = 0.3 ∧ d = 100 :=
begin
    use [0.3, 100],
    split,
    { 
        field_simp,
        norm_num,
    },
    split; refl,
end

end hybrid_travel_cost_of_electricity_proof_l438_438775


namespace parabola_vertex_form_parabola_intercepts_l438_438499

-- Part 1: Prove the equation of the parabola given the vertex and a point
theorem parabola_vertex_form
  (a b c : ℝ)
  (h : a ≠ 0)
  (vertex_cond : ∀ x y, y = a * x^2 + b * x + c → (x, y) = (1, 10))
  (point_cond : ∀ x y, y = a * x^2 + b * x + c → (x, y) = (-1, -2)) :
  ∃ (a : ℝ), y = -3 * (x - 1)^2 + 10 := sorry

-- Part 2: Prove the equation of the parabola given its intersections with the axes
theorem parabola_intercepts
  (a b c : ℝ)
  (h : a ≠ 0)
  (x_axis_cond : ∀ x, y = a * (x + 1) * (x - 3) → (x = -1 ∨ x = 3) ∧ y = 0)
  (y_axis_cond : ∀ y, y = a * (x + 1) * (x - 3) → x = 0 ∧ y = 3) :
  ∃ (a : ℝ), y = -x^2 + 2x + 3 := sorry

end parabola_vertex_form_parabola_intercepts_l438_438499


namespace lucy_packs_of_cake_l438_438314

theorem lucy_packs_of_cake (total_groceries cookies : ℕ) (h1 : total_groceries = 27) (h2 : cookies = 23) :
  total_groceries - cookies = 4 :=
by
  -- In Lean, we would provide the actual proof here, but we'll use sorry to skip the proof as instructed
  sorry

end lucy_packs_of_cake_l438_438314


namespace find_b_l438_438756

-- The definition of the parabola equation and the conditions.
noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
def point1 := (-1, -8 : ℝ)
def point2 := (2, 10 : ℝ)

-- The theorem we need to prove: given the parabola passes through the points, b must be 5
theorem find_b 
  (b c : ℝ)
  (h1 : parabola (-1) b c = -8)
  (h2 : parabola 2 b c = 10)
  : b = 5 := by
  sorry

end find_b_l438_438756


namespace sin_alpha_is_correct_l438_438887

noncomputable def sin_alpha : ℝ := - (sqrt 5 / 5)

theorem sin_alpha_is_correct (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2)
  (h2 : tan (α + π / 4) = 3) : sin α = sin_alpha :=
begin
  sorry
end

end sin_alpha_is_correct_l438_438887


namespace regression_equation_correct_updated_probability_cost_decrease_l438_438796

-- Conditions
def x_list := [1, 2, 3, 4, 5]
def y_list := [692, 962, 1334, 2091, 3229]
def sum_ln_y := 36.33
def sum_x_ln_y := 112.85
def avg_x := (1 + 2 + 3 + 4 + 5) / 5
def avg_ln_y := 36.33 / 5
def sum_x_sq := (1^2 + 2^2 + 3^2 + 4^2 + 5^2)
def regression_coeff b a := b = 0.386 ∧ a = 6.108
def unchanged_product_cost := 4

-- Main theorem
theorem regression_equation_correct :
  let b := (sum_x_ln_y - 5 * avg_x * avg_ln_y) / (sum_x_sq - 5 * avg_x ^ 2),
      a := avg_ln_y - b * avg_x in
  regression_coeff b a :=
by
  have hb : b = (112.85 - 5 * 3 * 7.266) / (55 - 45),
  have ha : a = 7.266 - 0.386 * 3,
sorry

theorem updated_probability :
  let m := 4 in
  let sigma := sqrt (1 / m),
      probability := 0.9545 in
  P(X - 0 < 1 < 2 * sigma) = probability :=
sorry

theorem cost_decrease :
  let initial_cost := 4,
      final_cost := 1,
      decrease := initial_cost - final_cost in
  final_cost = 1 ∧ decrease = 3 :=
by
  have hdec : initial_cost - final_cost = 3,
  have hfin : final_cost = 1,
sorry

end regression_equation_correct_updated_probability_cost_decrease_l438_438796


namespace inradii_triangle_inequality_l438_438303

theorem inradii_triangle_inequality
  (ABC : Triangle)
  (r : ℝ) (D : Point)
  (r1 r2 : ℝ)
  (h_r : r = inscribedRadius ABC)
  (h_D : Point_on_line D (Triangle.side BC ABC))
  (h_r1 : r1 = inradius (Triangle.subtriangle ABD ABC D))
  (h_r2 : r2 = inradius (Triangle.subtriangle ACD ABC D)) :
  r1 + r2 > r :=
sorry

end inradii_triangle_inequality_l438_438303


namespace cos_thirteen_pi_over_four_l438_438818

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l438_438818


namespace Ana_age_eight_l438_438287

theorem Ana_age_eight (A B n : ℕ) (h1 : A - 1 = 7 * (B - 1)) (h2 : A = 4 * B) (h3 : A - B = n) : A = 8 :=
by
  sorry

end Ana_age_eight_l438_438287


namespace triangle_tangent_cotangent_identity_l438_438950

theorem triangle_tangent_cotangent_identity (A B C : ℝ) (a b c : ℝ) (hA : a = 8) (hB : b = 6) (hC : c = 7) :
  (tan ((A - B) / 2) / cot (C / 2)) - (cot ((A - B) / 2) / tan (C / 2)) = 1 :=
by
  sorry

end triangle_tangent_cotangent_identity_l438_438950


namespace total_matches_played_l438_438352

theorem total_matches_played (avg_score_2 : ℕ) (avg_score_3 : ℕ) (avg_score_all : ℕ) (total_score_2 : ℕ)
  (total_score_3 : ℕ) (total_score_all : ℕ) (num_matches_2 : ℕ) (num_matches_3 : ℕ) (num_matches_all : ℕ) :
  (avg_score_2 = 60) ∧ (num_matches_2 = 2) ∧ (total_score_2 = avg_score_2 * num_matches_2) ∧
  (avg_score_3 = 50) ∧ (num_matches_3 = 3) ∧ (total_score_3 = avg_score_3 * num_matches_3) ∧
  (avg_score_all = 54) ∧ (total_score_all = total_score_2 + total_score_3) ∧
  (num_matches_all = total_score_all / avg_score_all) →
  num_matches_all = 5 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9⟩ := h
  simp [h1, h2, h3, h4, h5, h6, h7, h8, h9]
  sorry

end total_matches_played_l438_438352


namespace seven_perpendicular_lines_possible_l438_438978

-- Define a point in 3-dimensional space
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Define lines through a point in 3-dimensional space
structure LineThrough (p : Point3D) :=
(dir : Vector3D)
(perp_to : Set LineThrough)

-- Vector in 3-dimensional space
structure Vector3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Define the condition that two vectors are perpendicular
def perpendicular (v1 v2 : Vector3D) : Prop :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

-- Define the problem
theorem seven_perpendicular_lines_possible :
  ∃ (p : Point3D) (lines : List (LineThrough p)),
  lines.length = 7 ∧ 
  ∀ (l1 l2 : LineThrough p),
    l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 →
    ∃ (l3 : LineThrough p), 
      l3 ∈ lines ∧ 
      perpendicular l1.dir l3.dir ∧
      perpendicular l2.dir l3.dir :=
sorry

end seven_perpendicular_lines_possible_l438_438978


namespace find_geometric_constant_l438_438135

theorem find_geometric_constant (x : ℝ) : (40 + x)^2 = (10 + x) * (160 + x) → x = 0 :=
by
  intro h
  have h_simplified : 1600 + 80 * x + x^2 = 1600 + 170 * x + x^2 :=
    calc 
      (40 + x)^2 = 1600 + 80 * x + x^2 : by ring
      (10 + x) * (160 + x) = 1600 + 170 * x + x^2 : by ring
  have h_eq : 80 * x = 170 * x := by linarith
  have h_x_zero : x = 0 := by linarith
  exact h_x_zero

end find_geometric_constant_l438_438135


namespace sum_eq_exp23110_l438_438596

def sequence_a : ℕ → ℤ
| 0       := 1
| 1       := 1
| 2       := 1
| 3       := 18
| n := if n < 0 then 0 else sequence_a (n - 1) + 2 * (n - 1) * sequence_a (n - 2) + 9 * (n - 1) * (n - 2) * sequence_a (n - 3) + 8 * (n - 1) * (n - 2) * (n - 3) * sequence_a (n - 4)

noncomputable def target_sum : ℝ :=
  ∑ n in Finset.range (100000), (10^n * (sequence_a n) / (n.factorial : ℤ))

theorem sum_eq_exp23110 :
  target_sum = Real.exp 23110 :=
sorry

end sum_eq_exp23110_l438_438596


namespace parabola_point_value_l438_438228

variable {x₀ y₀ : ℝ}

theorem parabola_point_value
  (h₁ : y₀^2 = 4 * x₀)
  (h₂ : (Real.sqrt ((x₀ - 1)^2 + y₀^2) = 5/4 * x₀)) :
  x₀ = 4 := by
  sorry

end parabola_point_value_l438_438228


namespace no_valid_arrangement_l438_438005

-- Define the set of cards
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 8, 9]

-- Define a function that checks if a pair forms a number divisible by 7
def pair_divisible_by_7 (a b : ℕ) : Prop := (a * 10 + b) % 7 = 0

-- Define the main theorem to be proven
theorem no_valid_arrangement : ¬ ∃ (perm : List ℕ), perm ~ cards ∧ (∀ i, i < perm.length - 1 → pair_divisible_by_7 (perm.nth_le i (by sorry)) (perm.nth_le (i + 1) (by sorry))) :=
sorry

end no_valid_arrangement_l438_438005


namespace find_n_l438_438475

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15827 [MOD 16] ∧ n = 3 :=
by
  use 3
  split
  exact Nat.zero_le 3
  split
  exact Nat.le_refl 3
  split
  exact Nat.ModEq.of_eq (by norm_num)
  rfl

end find_n_l438_438475


namespace frac_diff_zero_l438_438483

theorem frac_diff_zero (a b : ℝ) (h : a + b = a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a) - (1 / b) = 0 := 
sorry

end frac_diff_zero_l438_438483


namespace area_of_quadrilateral_l438_438153

theorem area_of_quadrilateral (d h1 h2 : ℝ) (h1_pos : h1 = 9) (h2_pos : h2 = 6) (d_pos : d = 30) : 
  let area1 := (1/2 : ℝ) * d * h1
  let area2 := (1/2 : ℝ) * d * h2
  (area1 + area2) = 225 :=
by
  sorry

end area_of_quadrilateral_l438_438153


namespace Antoinette_weight_l438_438780

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end Antoinette_weight_l438_438780


namespace find_x_l438_438481

theorem find_x (x : ℝ) (h : sqrt (x - 3) = 5) : x = 28 :=
by
  sorry

end find_x_l438_438481


namespace find_a_g_decreasing_on_interval_find_lambda_l438_438903

/-- Define f(x) = 3^x. -/
def f (x : ℝ) : ℝ := 3^x

/-- Define g(x) = λ ⋅ 2^(ax) - 4^x -/
def g (λ a x : ℝ) : ℝ := λ * 2^(a * x) - 4^x

/-- Define the condition f(a + 2) = 27. -/
def condition_a (a : ℝ) : Prop := f (a + 2) = 27

/-- Define the condition that the domain of g includes [0, 2]. -/
def domain_condition (g : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 2 → g x = g x

/-- Prove that a = 1 given f(a + 2) = 27. -/
theorem find_a : ∃ a : ℝ, condition_a a ∧ a = 1 :=
by {
    sorry
}

/-- Prove that g(x) is decreasing on [0, 2] when λ = 2, a = 1. -/
theorem g_decreasing_on_interval : ∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g 2 1 x2 < g 2 1 x1 :=
by {
    sorry
}

/-- Prove that λ = 4/3 when the maximum value of g(x) is 1/3. -/
theorem find_lambda : ∃ λ : ℝ, ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → g λ 1 x ≤ 1 / 3 :=
by {
    sorry
}

end find_a_g_decreasing_on_interval_find_lambda_l438_438903


namespace scalene_triangle_y_value_l438_438799

noncomputable def y_value (AB BC AC AE BE AD DC EC : ℝ) : ℝ :=
  let y := 7.75
  in y

theorem scalene_triangle_y_value :
  ∀ (AB BC AC AE BE AD DC EC : ℝ), 
    AD = 6 → DC = 4 → BE = 3 → EC = y_value AB BC AC AE BE AD DC EC → 
    y_value AB BC AC AE BE AD DC EC = 31 / 4 :=
by
  intros
  have y := y_value AB BC AC AE BE AD DC EC
  sorry

end scalene_triangle_y_value_l438_438799


namespace find_a_l438_438178

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l438_438178


namespace enrique_shredder_Y_feeds_l438_438469

theorem enrique_shredder_Y_feeds :
  let typeB_contracts := 350
  let pages_per_TypeB := 10
  let shredderY_capacity := 8
  let total_pages_TypeB := typeB_contracts * pages_per_TypeB
  let feeds_ShredderY := (total_pages_TypeB + shredderY_capacity - 1) / shredderY_capacity
  feeds_ShredderY = 438 := sorry

end enrique_shredder_Y_feeds_l438_438469


namespace num_paths_formula_l438_438325

-- Define the rectangular grid
def rectangular_grid (m n : ℕ) : Set (ℤ × ℤ) := 
  {p : ℤ × ℤ | 0 ≤ p.1 ∧ p.1 < n ∧ 0 ≤ p.2 ∧ p.2 < m}

-- Define the allowable movements
def allowable_moves (p q : ℤ × ℤ) : Prop :=
  (q.1 = p.1 + 1 ∧ q.2 = p.2) ∨ (q.1 = p.1 ∧ q.2 = p.2 + 1)

-- Define the number of paths from (0,0) to (n,m) in the grid
noncomputable def paths_from_to (m n : ℕ) : ℕ := 
  (m + n).choose m

-- Theorem statement in Lean 4 to prove the number of paths
theorem num_paths_formula (m n : ℕ) : 
  paths_from_to m n = (m + n)! / (m! * n!) := by
  sorry

end num_paths_formula_l438_438325


namespace area_of_rhombus_perimeter_of_rhombus_l438_438356

-- Definitions and conditions for the area of the rhombus
def d1 : ℕ := 18
def d2 : ℕ := 16

-- Definition for the side length of the rhombus
def side_length : ℕ := 10

-- Statement for the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 144 := by
  sorry

-- Statement for the perimeter of the rhombus
theorem perimeter_of_rhombus : 4 * side_length = 40 := by
  sorry

end area_of_rhombus_perimeter_of_rhombus_l438_438356


namespace correlation_graph_is_scatter_plot_l438_438364

/-- The definition of a scatter plot graph -/
def scatter_plot_graph (x y : ℝ → ℝ) : Prop := 
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)

/-- Prove that the graph representing a set of data for two variables with a correlation is called a "scatter plot" -/
theorem correlation_graph_is_scatter_plot (x y : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)) → 
  (scatter_plot_graph x y) :=
by
  sorry

end correlation_graph_is_scatter_plot_l438_438364


namespace find_p_plus_q_plus_r_plus_s_l438_438973

noncomputable def problem_statement : Prop :=
  let AB : ℝ := 5
  let BC : ℝ := 7
  let CA : ℝ := 8
  let AG : ℝ := 6
  let AH : ℝ := 8
  let GI : ℝ := 3
  let HI : ℝ := 8
  let BH_expr : ℝ := (6 + 47 * Real.sqrt 2) / 9
  let p : ℕ := 6
  let q : ℕ := 47
  let r : ℕ := 2
  let s : ℕ := 9 in
  AB < AG ∧ AG < AH ∧
  (∀ (BH : ℝ), BH = BH_expr → BH = (p + q * Real.sqrt r) / s) ∧
  Nat.coprime p s ∧
  ¬∃ k : ℕ, k * k ∣ r ∧ k > 1 ∧
  (p + q + r + s = 64)

theorem find_p_plus_q_plus_r_plus_s : problem_statement :=
by {
  sorry
}

end find_p_plus_q_plus_r_plus_s_l438_438973


namespace exist_infinite_sets_of_4_consecutive_good_numbers_all_sets_of_5_consecutive_good_numbers_l438_438447

def is_good (n : ℕ) : Prop := ∃ x y : ℕ, n = 2^x + y^2

theorem exist_infinite_sets_of_4_consecutive_good_numbers :
  ∃ S : ℕ → (Fin 4 → ℕ), (∀ n, ∃ t, (fun i => S t (Fin.of_nat i)) 0 = t^2 + 1 ∧ 
    (fun i => S t (Fin.of_nat i)) 1 = t^2 + 2 ∧ 
    (fun i => S t (Fin.of_nat i)) 2 = t^2 + 3 ∧ 
    (fun i => S t (Fin.of_nat i)) 3 = t^2 + 4) := 
sorry

theorem all_sets_of_5_consecutive_good_numbers :
  ∀ (m : ℕ → Prop), ( ∀ i, (1 ≤ i ∧ i < 6) → is_good (m i) ) →
  m = (fun i => if i = 1 then 1 else 
                if i = 2 then 2 else 
                if i = 3 then 3 else 
                if i = 4 then 4 else 5) ∨
  m = (fun i => if i = 1 then 2 else 
                if i = 2 then 3 else 
                if i = 3 then 4 else 
                if i = 4 then 5 else 6) ∨
  m = (fun i => if i = 1 then 8 else 
                if i = 2 then 9 else 
                if i = 3 then 10 else 
                if i = 4 then 11 else 12) ∨
  m = (fun i => if i = 1 then 9 else 
                if i = 2 then 10 else 
                if i = 3 then 11 else 
                if i = 4 then 12 else 13) ∨
  m = (fun i => if i = 1 then 288 else 
                if i = 2 then 289 else 
                if i = 3 then 290 else 
                if i = 4 then 291 else 292) ∨
  m = (fun i => if i = 1 then 289 else 
                if i = 2 then 290 else 
                if i = 3 then 291 else 
                if i = 4 then 292 else 293) :=
sorry

end exist_infinite_sets_of_4_consecutive_good_numbers_all_sets_of_5_consecutive_good_numbers_l438_438447


namespace extra_bananas_l438_438324

theorem extra_bananas (total_children present_children absent_children : ℕ) :
  (total_children = 780) →
  (absent_children = 390) →
  (present_children = total_children - absent_children) →
  let B := total_children * 2 in
  let bananas_per_present_child := B / present_children in
  bananas_per_present_child - 2 = 2 :=
by
  intros h_total_children h_absent_children h_present_children
  let B := total_children * 2
  let bananas_per_present_child := B / present_children
  have h1 : B = 780 * 2 := by rw [h_total_children]
  have h2 : present_children = 780 - 390 := by rw [← h_total_children, ← h_absent_children, h_present_children]
  rw [h2] at bananas_per_present_child
  let num_bananas := 780 * 2
  let denom_children := 780 - 390
  have h3: bananas_per_present_child = num_bananas / denom_children := by sorry
  have h4: 780 * 2 = 1560 := by norm_num
  have h5: 780 - 390 = 390 := by norm_num
  rw [h4, h5] at h3
  have h6: bananas_per_present_child = 1560 / 390 := by sorry
  have h7: 1560 / 390 = 4 := by norm_num
  rw [h7] at bananas_per_present_child
  have h8: 4 - 2 = 2 := by norm_num
  rw [h8]
  sorry

end extra_bananas_l438_438324


namespace penthouse_floors_l438_438755

theorem penthouse_floors (R P : ℕ) (h1 : R + P = 23) (h2 : 12 * R + 2 * P = 256) : P = 2 :=
by
  sorry

end penthouse_floors_l438_438755


namespace volleyball_lineup_count_l438_438767

theorem volleyball_lineup_count :
  let total_players := 10
  let flexible_players := 8
  let specialized_players := 2
  let positions := 5
  let setters_required := 1
  let liberos_required := 1
  let other_positions := 3
  (∃ (libero : ℕ) (setter : ℕ) 
    (outside_hitter : ℕ) (middle_blocker : ℕ) (opposite : ℕ),
    libero ∈ finset.range total_players ∧
    setter ∈ finset.range (total_players - 1) ∧
    outside_hitter ∈ finset.range (total_players - 2) ∧
    middle_blocker ∈ finset.range (total_players - 3) ∧
    opposite ∈ finset.range (total_players - 4) ∧
    libero ≠ setter ∧ setter ≠ outside_hitter ∧ 
    outside_hitter ≠ middle_blocker ∧
    middle_blocker ≠ opposite ∧ libero != opposite ) →
  (10 * 9 * 8 * 7 * 6 = 30240) :=
by
  sorry

end volleyball_lineup_count_l438_438767


namespace alex_growth_rate_l438_438092

noncomputable def growth_rate_per_hour_hanging_upside_down
  (current_height : ℝ)
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) : Prop :=
  current_height + 12 * normal_growth_per_month + 12 * hanging_hours_per_month * answer = required_height

theorem alex_growth_rate 
  (current_height : ℝ) 
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) :
  current_height = 48 → 
  required_height = 54 → 
  normal_growth_per_month = 1/3 → 
  hanging_hours_per_month = 2 → 
  growth_rate_per_hour_hanging_upside_down current_height required_height normal_growth_per_month hanging_hours_per_month answer ↔ answer = 1/12 :=
by sorry

end alex_growth_rate_l438_438092


namespace debby_drink_days_l438_438129

theorem debby_drink_days :
  ∀ (total_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ),
  total_bottles = 301 →
  bottles_per_day = 144 →
  remaining_bottles = 157 →
  (total_bottles - remaining_bottles) / bottles_per_day = 1 :=
by
  intros total_bottles bottles_per_day remaining_bottles ht he hb
  sorry

end debby_drink_days_l438_438129


namespace avg_rest_students_l438_438953

/- Definitions based on conditions -/
def total_students : ℕ := 28
def students_scored_95 : ℕ := 4
def students_scored_0 : ℕ := 3
def avg_whole_class : ℚ := 47.32142857142857
def total_marks_95 : ℚ := students_scored_95 * 95
def total_marks_0 : ℚ := students_scored_0 * 0
def marks_whole_class : ℚ := total_students * avg_whole_class
def rest_students : ℕ := total_students - students_scored_95 - students_scored_0

/- Theorem to prove the average of the rest students given the conditions -/
theorem avg_rest_students : (total_marks_95 + total_marks_0 + rest_students * 45) = marks_whole_class :=
by
  sorry

end avg_rest_students_l438_438953


namespace triangle_area_is_12_l438_438432

noncomputable def triangle_area : Real :=
  let line := λ x y : ℝ, 3 * x + 2 * y = 12
  let base := 4
  let height := 6
  (1 / 2) * base * height

theorem triangle_area_is_12 :
  triangle_area = 12 := by
  sorry

end triangle_area_is_12_l438_438432


namespace no_valid_cube_labeling_l438_438281

theorem no_valid_cube_labeling :
  ¬ ∃ (c : fin 8 → ℕ), 
    (∀ i j, i ≠ j → i < j → 1 ≤ c i ∧ c i ≤ 600 ∧ 1 ≤ c j ∧ c j ≤ 600 ∧ c i % 2 = 1 ∧ c j % 2 = 1 ∧ (adjacent i j → gcd (c i) (c j) > 1) ∧ (¬adjacent i j → gcd (c i) (c j) = 1)) :=
sorry

-- Define adjacency for a cube
def adjacent (i j : fin 8) : Prop :=
  <define adjacency relation on vertices of the cube somehow>

end no_valid_cube_labeling_l438_438281


namespace locus_of_points_l438_438515

-- Definitions of points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Definitions of the conditions in Lean
def distance_squared (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Proposition statement
theorem locus_of_points (A B : Point) (k : ℝ) :
  ∃ M : Point, distance_squared M A - distance_squared M B = k :=
begin
  -- Assume the coordinates of points A and B
  let A := {x := 0, y := 0},
  let B := {x := a, y := 0},

  -- Let M have coordinates (x, y)
  let M := {x := (k + a ^ 2) / (2 * a), y := y},

  -- Show the difference of the squares of the distances is constant
  use M,
  sorry
end

end locus_of_points_l438_438515


namespace infimum_of_g_l438_438488

noncomputable def g (a : ℝ) : ℝ := a^2 - 4*a + 6

theorem infimum_of_g : ∀ a : ℝ, a ≠ 0 → a^2 - 4*a + 6 ≥ 2 :=
by {
    intro a,
    intro ha,
    calc
    a^2 - 4*a + 6 = (a - 2)^2 + 2 : by ring,
    ... ≥ 2 : by {
      have h : (a - 2)^2 ≥ 0, from pow_two_nonneg _,
      linarith,
    }
}

end infimum_of_g_l438_438488


namespace ivanov_family_net_worth_l438_438719

theorem ivanov_family_net_worth :
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  in net_worth = 2300000 := 
by
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  show net_worth = 2300000 from sorry

end ivanov_family_net_worth_l438_438719


namespace slopes_product_eq_neg_quarter_l438_438511

theorem slopes_product_eq_neg_quarter (a b : ℝ) (h : a > b) (h1 : b > 0)
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) 
  (eccentricity_eq : ∀ c : ℝ, c / a = Real.sqrt(1 - (b^2 / a^2)) → c = (Real.sqrt 3) / 2)
  (M : ℝ × ℝ) (M_on_ellipse : (M.1 ^ 2 / a^2) + (M.2 ^ 2 / b^2) = 1)
  (A B : ℝ × ℝ) (symmetric_AB : A.1 = -B.1 ∧ A.2 = -B.2)
  (slopes : ℝ × ℝ):
  slope(ℝ × ℝ) (M.2 - A.2) (M.1 - A.1) * slope(ℝ × ℝ) (M.2 - B.2) (M.1 - B.1) = -1 / 4 :=
sorry

end slopes_product_eq_neg_quarter_l438_438511


namespace chair_price_l438_438784

theorem chair_price (
  (P : ℝ) 
  (H : 5.25 * P = 105)
) : P = 20 :=
sorry

end chair_price_l438_438784


namespace find_a_l438_438180

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l438_438180


namespace product_of_nonreal_roots_of_polynomial_l438_438161

theorem product_of_nonreal_roots_of_polynomial :
  let f := (λ x : ℂ, x^4 - 5*x^3 + 10*x^2 - 10*x - 500) in
  let roots_nonreal := {x : ℂ | f x = 0 ∧ ¬ (x.im = 0)} in
  let product := ∏ x in roots_nonreal, x in
  product = 1 + Complex.sqrt 501 :=
sorry

end product_of_nonreal_roots_of_polynomial_l438_438161


namespace minimum_fruits_l438_438687

open Nat

theorem minimum_fruits (n : ℕ) :
    (n % 3 = 2) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 6 = 5) →
    n = 59 := by
  sorry

end minimum_fruits_l438_438687


namespace limit_problem_l438_438931

theorem limit_problem
  {f : ℝ → ℝ} {x₀ : ℝ}
  (h_deriv : deriv f x₀ = -3) :
  (tendsto (λ h, (f (x₀ + h) - f (x₀ - 3 * h)) / h) (𝓝 0) (𝓝 (-12))) :=
by {
  sorry
}

end limit_problem_l438_438931


namespace length_CD_equals_8_l438_438502

variables (A B C D L : Type) [metric_space A] [metric_space B] [metric_space C] 
          [metric_space D] [metric_space L]
          (angleD : angle A D B = 100) (BC_length : dist B C = 12)
          (AD_length : dist A D = 12) (LD_length : dist L D = 4)
          (angleABL : angle A B L = 50)
          (parallelogram_ABCD : parallelogram A B C D)

theorem length_CD_equals_8 :
  dist C D = 8 :=
sorry

end length_CD_equals_8_l438_438502


namespace range_of_x_when_a_is_1_range_of_a_for_necessity_l438_438207

-- Define the statements p and q based on the conditions
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- (1) Prove the range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_1 {x : ℝ} (h1 : ∀ x, p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- (2) Prove the range of a for p to be necessary but not sufficient for q
theorem range_of_a_for_necessity : ∀ a, (∀ x, p x a → q x) → (1 ≤ a ∧ a ≤ 2) :=
  sorry

end range_of_x_when_a_is_1_range_of_a_for_necessity_l438_438207


namespace find_constants_l438_438487

theorem find_constants (a c : ℝ) :
  (a, c) = (6, 2) ↔
  (let u := ⟨a, -1, c⟩ in
   let v := ⟨7, 3, 5⟩ in
   u × v = ⟨-11, -16, 25⟩) :=
sorry

end find_constants_l438_438487


namespace power_of_complex_expression_l438_438885

noncomputable def i : ℂ := complex.I

theorem power_of_complex_expression : ( (1 + i) / (1 - i) ) ^ 2013 = i := by
  sorry

end power_of_complex_expression_l438_438885


namespace working_together_time_l438_438706

/-- A is 30% more efficient than B,
and A alone can complete the job in 23 days.
Prove that A and B working together take approximately 13 days to complete the job. -/
theorem working_together_time (Ea Eb : ℝ) (T : ℝ) (h1 : Ea = 1.30 * Eb) 
  (h2 : 1 / 23 = Ea) : T = 13 :=
sorry

end working_together_time_l438_438706


namespace travel_time_difference_l438_438658

theorem travel_time_difference :
  let d := 2 -- distance to the park in miles
  let v_j := 10 -- Jill's speed in miles per hour
  let v_k := 5 -- Jack's speed in miles per hour
  let t_j := d / v_j -- Jill's travel time in hours
  let t_k := d / v_k -- Jack's travel time in hours
  let t_j_minutes := t_j * 60 -- Jill's travel time in minutes
  let t_k_minutes := t_k * 60 -- Jack's travel time in minutes
  in t_k_minutes - t_j_minutes = 12 :=
by sorry

end travel_time_difference_l438_438658


namespace fraction_value_l438_438557

theorem fraction_value (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : (x + y + z) / (2 * z) = 9 / 8 :=
by
  sorry

end fraction_value_l438_438557


namespace convert_118_to_base_6_l438_438801

def convert_to_base_6 (n : ℕ) : ℕ :=
  -- Function to convert n to base-6 manually
  let q1 := n / 6 in
  let r1 := n % 6 in
  let q2 := q1 / 6 in
  let r2 := q1 % 6 in
  let q3 := q2 / 6 in
  let r3 := q2 % 6 in
  r3 * 100 + r2 * 10 + r1

theorem convert_118_to_base_6 :
  convert_to_base_6 118 = 314 := 
by 
  sorry

end convert_118_to_base_6_l438_438801


namespace log_equation_solutions_l438_438808

theorem log_equation_solutions : 
  ∃ (a b : ℤ), (log 10 (a^2 - 15 * a) = 3 ∧ log 10 (b^2 - 15 * b) = 3 ∧ a ≠ b) := 
sorry

end log_equation_solutions_l438_438808


namespace smallest_digit_pair_l438_438588

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def valid_seven_digit_number (A B : ℕ) : ℕ :=
  3000000 + A * 100000 + 60792 + B

-- Define the conditions
def palindrome_condition (A B : ℕ) : Prop :=
  B = 3 ∧ A = 6

-- Define the smallest valid pair condition
def smallest_valid_pair (A B : ℕ) :=
  valid_seven_digit_number A B = 3667963

theorem smallest_digit_pair :
  ∃ (A B : ℕ), palindrome_condition A B ∧ is_divisible_by_4 (valid_seven_digit_number A B) ∧ smallest_valid_pair A B :=
begin
  use [6, 3],
  split,
  { -- Proving palindrome condition
    exact ⟨rfl, rfl⟩ },
  split,
  { -- Proving divisibility by 4
    unfold valid_seven_digit_number,
    rw [add_assoc, add_comm 792 3, add_assoc, nat.add_mod, nat.mod_self, zero_add, nat.mod_mod, nat.mod_self],
    norm_num
  },
  { -- Proving it's the smallest valid pair
    unfold valid_seven_digit_number smallest_valid_pair,
    ring }
end

end smallest_digit_pair_l438_438588


namespace max_g_value_l438_438131

def g : Nat → Nat
| n => if n < 15 then n + 15 else g (n - 6)

theorem max_g_value : ∀ n, g n ≤ 29 := by
  sorry

end max_g_value_l438_438131


namespace sin_sum_le_tan_sq_l438_438702

theorem sin_sum_le_tan_sq (x : ℝ) (n : ℕ) (h : sin x ^ 2 ≤ tan x ^ 2) : 
  (∑ k in Finset.range (n + 1), (sin x) ^ (2 * (k + 1))) ≤ tan x ^ 2 := 
by
  sorry

end sin_sum_le_tan_sq_l438_438702


namespace lock4_different_digits_count_l438_438763

def lock4_different_digits_settings (d : Fin 10 → Fin 10 → Fin 10 → Fin 10 → Prop) : ℕ :=
  let choices := 10 * 9 * 8 * 7 in
  choices

theorem lock4_different_digits_count :
  lock4_different_digits_settings (λ d1 d2 d3 d4, d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) = 5040 :=
by
  sorry

end lock4_different_digits_count_l438_438763


namespace collinear_G_E_N_l438_438534

-- Definitions for circles and their intersections
noncomputable def Circle (α : Type*) := set α

variables {α : Type*} [metric_space α]

-- Assume the two circles O1 and O2 intersect
axiom circle_O1 : Circle α
axiom circle_O2 : Circle α
axiom point_A : α
axiom point_B : α
axiom point_A_on_O1 : point_A ∈ circle_O1
axiom point_A_on_O2 : point_A ∈ circle_O2
axiom point_B_on_O1 : point_B ∈ circle_O1
axiom point_B_on_O2 : point_B ∈ circle_O2

-- Chords AD and AC tangent to other circle
axiom point_D : α
axiom point_C : α
axiom chord_AD_O2 : point_D ∉ circle_O1
axiom chord_AC_O1 : point_C ∉ circle_O2

-- Angle bisector conditions and intersections
axiom point_F : α
axiom point_E : α
axiom bisector_CAD_intersects_F_O1 : point_F ∈ circle_O1
axiom bisector_CAD_intersects_E_O2 : point_E ∈ circle_O2

-- Line l parallel to AF and tangent conditions
axiom point_G : α
axiom line_l_parallel_to_AF_tangent_to_circumcircle_of_BEF_at_G : true -- simplified

-- Further intersections of FG and MA with corresponding circles
axiom point_M : α
axiom point_N : α
axiom FG_intersects_O1_at_M : true -- simplified
axiom MA_intersects_O2_at_N : true -- simplified

-- Prove collinearity of G, E, N
theorem collinear_G_E_N : collinear ({G, E, N} : set α) :=
sorry  -- the proof itself is omitted

end collinear_G_E_N_l438_438534


namespace supplement_of_complement_of_30_degrees_l438_438693

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def α : ℝ := 30

theorem supplement_of_complement_of_30_degrees : supplement (complement α) = 120 := 
by
  sorry

end supplement_of_complement_of_30_degrees_l438_438693


namespace total_points_scored_l438_438769

-- Define the conditions
variables (A_shots M_shots A_2s M_3s : ℕ)
variable (x : ℕ) -- number of 2-point shots Adam made, also the number of 3-point shots Mada made
variable h₁ : A_shots = 10 -- Adam makes 10 shots
variable h₂ : M_shots = 11 -- Mada makes 11 shots
variable h₃ : A_2s = M_3s -- Adam's 2-point shots are the same as Mada's 3-point shots
variable h₄ : 2 * x + 3 * (A_shots - x) = 3 * x + 2 * (M_shots - x) -- Both score the same total points

-- Define the goal
theorem total_points_scored :
  2 * x + 3 * (10 - x) + 3 * x + 2 * (11 - x) = 52 :=
by sorry

end total_points_scored_l438_438769


namespace possible_perimeters_of_rectangle_l438_438381

/--
Using 4 squares with a side length of 3 cm, the possible perimeters of the rectangle are 24 cm and 30 cm.
-/
theorem possible_perimeters_of_rectangle (side_length : ℝ) (num_squares : ℕ) 
  (h_sides : side_length = 3) (h_squares : num_squares = 4) : 
  ∃ (P1 P2 : ℝ), P1 = 24 ∧ P2 = 30 :=
by
  have P1 := 2 * (4 * side_length + 3)
  have P2 := 4 * (2 * side_length)
  existsi [P1, P2]
  rw [h_sides, h_squares]
  simp [side_length]
  sorry

end possible_perimeters_of_rectangle_l438_438381


namespace pentagon_perimeter_l438_438757

-- Definition of points forming a pentagon.
def points : List (ℝ × ℝ) := [(0,0), (1,3), (3,3), (4,0), (2,-1)]

def dist (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Calculating the distance between each consecutive points
def perimeter (pts : List (ℝ × ℝ)) : ℝ :=
  let distances := List.map (λ (p : ℝ × ℝ × ℝ × ℝ), dist p.1 p.2) (List.zip pts (pts.tail : List (ℝ × ℝ)) ++ [(List.head pts).iget, pts.getLast!])
  distances.sum

-- Our final proof statement
theorem pentagon_perimeter :
  let p := 2
  let q := 2
  let r := 0
  p + q + r = 4 :=
by
  sorry

end pentagon_perimeter_l438_438757


namespace binomial_sum_eval_l438_438142

theorem binomial_sum_eval :
  (Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 5)) +
  (Nat.factorial 6 / (Nat.factorial 4 * Nat.factorial 2)) = 36 := by
sorry

end binomial_sum_eval_l438_438142


namespace find_xyz_l438_438482

theorem find_xyz (x y z : ℝ) 
  (h1: 3 * x - y + z = 8)
  (h2: x + 3 * y - z = 2) 
  (h3: x - y + 3 * z = 6) :
  x = 1 ∧ y = 3 ∧ z = 8 := by
  sorry

end find_xyz_l438_438482


namespace dragon_jewels_l438_438743

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l438_438743


namespace triangle_sum_correct_l438_438133

def triangle_sum (a : ℕ) : ℕ :=
  a + (a + 1) + (a + 2) + ... + (2 * a - 1)

def triangle_formula (a : ℕ) : ℕ :=
  (a * (3 * a - 1)) / 2

theorem triangle_sum_correct :
  (∑ a in Finset.range 20, triangle_formula (a + 1)) = 4200 :=
  sorry

end triangle_sum_correct_l438_438133


namespace tangent_points_and_lines_l438_438565

theorem tangent_points_and_lines (x y : ℝ) :
  (∃ (x₁ y₁ : ℝ), y = x₁^3 + x₁ - 10 ∧ 3*x₁^2 + 1 = 4 ∧ (x = x₁ ∧ y = y₁) ∧ 
      (y₁ = -8 ∧ y = 4*x - 12) ∨ (y₁ = -12 ∧ y = 4*x - 8)) :=
begin
  sorry,
end

end tangent_points_and_lines_l438_438565


namespace count_valid_score_combinations_l438_438684

def scores := {x : ℕ | x = 89 ∨ x = 90 ∨ x = 91 ∨ x = 92 ∨ x = 93}
def valid_scores (x1 x2 x3 x4 : ℕ) : Prop := 
  scores x1 ∧ scores x2 ∧ scores x3 ∧ scores x4 ∧ (x1 < x2 ∧ x2 ≤ x3 ∧ x3 < x4)

theorem count_valid_score_combinations : 
  { (x1, x2, x3, x4) : ℕ × ℕ × ℕ × ℕ | valid_scores x1 x2 x3 x4 }.to_finset.card = 15 :=
by sorry

end count_valid_score_combinations_l438_438684


namespace intervals_of_monotonicity_and_extreme_values_l438_438806

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (∀ x : ℝ, f 1 = 1 / Real.exp 1) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l438_438806


namespace tangent_line_through_point_A_tangent_lines_perpendicular_to_l_l438_438194

theorem tangent_line_through_point_A (x y : ℝ) : 
  (x - 1)^2 + (y + 2)^2 = 10 ∧ y = (-3 * (x - 4) - 1) → 3 * x + y - 11 = 0 := 
sorry

theorem tangent_lines_perpendicular_to_l (x y : ℝ) (m : ℝ) : 
  (x - 1)^2 + (y + 2)^2 = 10 ∧
  (2 * (1) + 1 * (-2) + m) / (√5) = √10 → 
  (2 * x + y + 5 * √2 = 0 ∨ 2 * x + y - 5 * √2 = 0) := 
sorry

end tangent_line_through_point_A_tangent_lines_perpendicular_to_l_l438_438194


namespace binomial_distribution_parameters_l438_438176

noncomputable def X (n p : ℕ) : Type := sorry

def expectation (X : Type) : ℝ := sorry
def variance (X : Type) : ℝ := sorry

theorem binomial_distribution_parameters (n p : ℕ) (X : Type)
  (h1 : expectation (3 • X + 2) = 9.2)
  (h2 : variance (3 • X + 2) = 12.96) :
  n = 6 ∧ p = 0.4 :=
sorry

end binomial_distribution_parameters_l438_438176


namespace isosceles_trapezoid_ABCD_l438_438912

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Points given in the problem
def A : Point := { x := -6, y := -1 }
def B : Point := { x := 2, y := 3 }
def C : Point := { x := -1, y := 4 }
def D : Point := { x := -5, y := 2 }

-- Function to calculate the slope between two points
def slope (P1 P2 : Point) : ℝ :=
  (P2.y - P1.y) / (P2.x - P1.x)

-- Function to calculate the distance squared between two points
def dist_sq (P1 P2 : Point) : ℝ :=
  (P2.x - P1.x)^2 + (P2.y - P1.y)^2

-- Prove that D completes the isosceles trapezoid ABCD with AB parallel to CD
theorem isosceles_trapezoid_ABCD :
  slope A B = slope C D ∧ dist_sq B C = dist_sq A D :=
by
  sorry

end isosceles_trapezoid_ABCD_l438_438912


namespace subset_iff_l438_438566

open Set

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ 2 ≤ a :=
by sorry

end subset_iff_l438_438566


namespace carmen_total_sales_l438_438792

-- Define the conditions as constants
def green_house_sales := 3 * 4            -- 3 boxes of samoas at $4 each
def yellow_house_sales := 2 * 3.5 + 5     -- 2 boxes of thin mints at $3.50 each and 1 box of fudge delights for $5
def brown_house_sales := 9 * 2            -- 9 boxes of sugar cookies at $2 each

-- The statement that needs to be proved
theorem carmen_total_sales : (green_house_sales + yellow_house_sales + brown_house_sales) = 42 := by
  sorry

end carmen_total_sales_l438_438792


namespace bankers_discount_correct_l438_438045

noncomputable def bankers_discount (BG : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let PV := (BG * 100) / (r * t) in
  let FV := PV + BG in
  (FV * r * t) / 100

theorem bankers_discount_correct :
  bankers_discount 270 12 3 = 367.20 :=
by
  simp [bankers_discount]
  sorry  -- The detailed steps to arrive at the conclusion are omitted.

end bankers_discount_correct_l438_438045


namespace cos_A_in_right_triangle_l438_438273

theorem cos_A_in_right_triangle (A B C : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (α β γ : ℝ) (h_angleB : β = π / 2) (h_sinC : Real.sin γ = 3/5) :
  Real.cos α = (3 * Real.sqrt 34) / 34 := 
sorry

end cos_A_in_right_triangle_l438_438273


namespace two_rotational_homotheties_exist_l438_438917

theorem two_rotational_homotheties_exist (O1 O2 : Point) (r1 r2 : ℝ) (h1 : r1 > 0) (h2 : r2 > 0) (h3 : O1 ≠ O2) :
  ∃! (O : Point), ∃! (f : ℝ × ℝ → ℝ × ℝ), (∃ (θ : ℝ), θ = π / 2) ∧
  (∃ (k : ℝ), k = r1 / r2) ∧
  (∀ (S1 S2 : Circle), S1.center = O1 → S1.radius = r1 → S2.center = O2 → S2.radius = r2 → 
                   is_rotational_homothety f S1 S2 O θ k) := sorry

end two_rotational_homotheties_exist_l438_438917


namespace geometric_seq_comparison_l438_438372

def geometric_seq_positive (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n+1) = a n * q

theorem geometric_seq_comparison (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_seq_positive a q) (h2 : q ≠ 1) (h3 : ∀ n, a n > 0) (h4 : q > 0) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_seq_comparison_l438_438372


namespace ellipse_condition_l438_438535

noncomputable def ellipse_properties (x y m : ℝ) : Prop :=
  ∃ m ≥ 2, ∃ (e : ℝ), 
    e = Real.sqrt((m - 1) / m) ∧ 
    e ∈ Set.Ico (Real.sqrt 2 / 2) 1 /\
  ∀ (x₁ x₂ y₁ y₂ x₀ y₀ : ℝ), 
    x₁ + x₂ = 2 * x₀ → y₁ + y₂ = 2 * y₀ → 
    (1/m * (x₁ ^ 2 + x₂ ^ 2) + y₁ ^ 2 + y₂ ^ 2) = 1 → 
    ∃ K_AB K_OM : ℝ, 
      K_AB = (y₁ - y₂) / (x₁ - x₂) ∧ 
      K_OM = y₀ / x₀ ∧ 
      K_AB * K_OM = -1 / 4 →
  ellipse_eq (K_AB : ℝ) (K_OM : ℝ) {x y : ℝ} := 
    1/m * x^2 + y^2 = 1

theorem ellipse_condition (m e : ℝ) (h₁ : ellipse_properties x y m) : 
  m = 4 → ∃ x y, 1/4 * x^2 + y^2 = 1 :=
begin
  sorry
end

end ellipse_condition_l438_438535


namespace OI_perpendicular_CD_l438_438649

theorem OI_perpendicular_CD
  (A B C D P O I : Point)
  (h1 : convex_quadrilateral A B C D)
  (h2 : same_length A B A C)
  (h3 : same_length A B D B)
  (h4 : diagonals_intersect_at A B C D P)
  (h5 : circumcenter_triangle A B P O)
  (h6 : incenter_triangle A B P I)
  (h7 : O ≠ I) : perpendicular (line_through O I) (line_through C D) := 
sorry

end OI_perpendicular_CD_l438_438649


namespace average_production_l438_438396

theorem average_production (n : ℕ) :
  let total_past_production := 50 * n
  let total_production_including_today := 100 + total_past_production
  let average_production := total_production_including_today / (n + 1)
  average_production = 55
  -> n = 9 :=
by
  sorry

end average_production_l438_438396


namespace problem_statement_l438_438309

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 4)

theorem problem_statement :
  (∀ x, f (x) = 2 * Real.sin (2 * x + π / 4))
  ∧ (f (π / 8) = 2)
  ∧ (∃ x, f (x) = 0 ∧ x = 3 * π / 8)
  ∧ (f (-π / 4) = -Real.sqrt 2)
  ∧ (∀ x ∈ Set.Icc (-π / 4) (π / 4), f (x) ≤ 2 ∧ f (x) ≥ -Real.sqrt 2)
  ∧ (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + 3 * π / 8) (k * π + 7 * π / 8), Real.sin (2 * x - π / 4) ≤ 0)
  ∧ (∀ k : ℤ, ∃ x, 2 * x - π / 4 = k * π ∧ x = (k * π / 2 + π / 8)) :=
begin
  sorry
end

end problem_statement_l438_438309


namespace weight_of_3_moles_of_BaOH2_l438_438927

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_H : ℝ := 1.01

theorem weight_of_3_moles_of_BaOH2 (total_weight : ℝ) (h_total_weight : total_weight = 513) :
  3 * (molar_mass_Ba + 2 * (molar_mass_O + molar_mass_H)) ≈ total_weight :=
by
  sorry

end weight_of_3_moles_of_BaOH2_l438_438927


namespace least_number_of_shots_to_destroy_tank_l438_438068

/-- Given a 41x41 checkerboard field with a tank that moves to a neighboring cell when hit but stays 
in the same cell otherwise, and a pilot needing to hit the tank twice to destroy it, we want to 
prove that the minimum number of shots sufficient to guarantee the tank is destroyed is 2521. -/
theorem least_number_of_shots_to_destroy_tank (n : ℕ) (h1 : n = 41)
  (h2 : ∀ (t : ℕ × ℕ), t.1 ∈ fin n ∧ t.2 ∈ fin n)
  (h3 : ∀ (s : ℕ × ℕ), s.1 ∈ fin n ∧ s.2 ∈ fin n)
  (h4 : ∀ (s t : ℕ × ℕ), (s.1 = t.1 ∧ (s.2 = t.2 + 1 ∨ s.2 = t.2 - 1)) ∨ 
                           (s.2 = t.2 ∧ (s.1 = t.1 + 1 ∨ s.1 = t.1 - 1))) :
  2521 = 41 * 41 + 41 * 41 div 2 + 41 * 41 div 2 := by 
  -- The proof is omitted here.
  sorry

end least_number_of_shots_to_destroy_tank_l438_438068
