import Combinatorics
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeometryArith
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.ModEq
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Vector
import Mathlib.Data.Set.Finite
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Sorry
import Real

namespace track_meet_girls_short_hair_l826_826071

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l826_826071


namespace remainder_of_E_div_88_l826_826941

-- Define the given expression E and the binomial coefficient 
noncomputable def E : ℤ :=
  1 - 90 * Nat.choose 10 1 + 90 ^ 2 * Nat.choose 10 2 - 90 ^ 3 * Nat.choose 10 3 + 
  90 ^ 4 * Nat.choose 10 4 - 90 ^ 5 * Nat.choose 10 5 + 90 ^ 6 * Nat.choose 10 6 - 
  90 ^ 7 * Nat.choose 10 7 + 90 ^ 8 * Nat.choose 10 8 - 90 ^ 9 * Nat.choose 10 9 + 
  90 ^ 10 * Nat.choose 10 10

-- The theorem that we need to prove
theorem remainder_of_E_div_88 : E % 88 = 1 := by
  sorry

end remainder_of_E_div_88_l826_826941


namespace smallest_advantageous_discount_l826_826940

theorem smallest_advantageous_discount :
  ∃ n : ℕ, n = 29 ∧ (
    (1 - n / 100.0) < (1 - 0.12) * (1 - 0.12) ∧
    (1 - n / 100.0) < (1 - 0.08) * (1 - 0.08) * (1 - 0.09) ∧
    (1 - n / 100.0) < (1 - 0.20) * (1 - 0.10)
  ) := sorry

end smallest_advantageous_discount_l826_826940


namespace sin_alpha_plus_pi_over_3_l826_826302

def alpha (α : ℝ) : Prop :=
α > 0 ∧ α < π / 2

theorem sin_alpha_plus_pi_over_3 (α : ℝ) (hα : alpha α) (hcos : cos (α + π / 12) = 3 / 5) :
  sin (α + π / 3) = (7 * real.sqrt 2) / 10 :=
sorry

end sin_alpha_plus_pi_over_3_l826_826302


namespace natural_numbers_not_in_a_n_eq_perfect_squares_l826_826836

def floor (x : ℝ) : ℝ := Real.floor x

noncomputable def a_n (n : ℕ) : ℝ := floor (n + Real.sqrt n + 1/2)

theorem natural_numbers_not_in_a_n_eq_perfect_squares (k : ℕ) : 
  (∀ n : ℕ, a_n n ≠ k) ↔ ∃ m : ℕ, k = m * m := 
sorry

end natural_numbers_not_in_a_n_eq_perfect_squares_l826_826836


namespace problem_l826_826248

theorem problem (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α^2 = 16 / 5 :=
sorry

end problem_l826_826248


namespace train_speed_is_72_km_per_hr_l826_826899

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l826_826899


namespace distinct_ordered_pairs_count_l826_826292

theorem distinct_ordered_pairs_count :
  { (m, n) : ℕ × ℕ | (0 < m) ∧ (0 < n) ∧ (1 / m + 1 / n = 1 / 3) }.to_finset.card = 3 :=
by
  sorry

end distinct_ordered_pairs_count_l826_826292


namespace expression_for_neg_x_l826_826996

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem expression_for_neg_x (f : ℝ → ℝ) (h_odd : odd_function f) (h_nonneg : ∀ (x : ℝ), 0 ≤ x → f x = x^2 - 2 * x) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2 * x :=
by 
  intros x hx 
  have hx_pos : -x > 0 := by linarith 
  have h_fx_neg : f (-x) = -f x := h_odd x
  rw [h_nonneg (-x) (by linarith)] at h_fx_neg
  linarith

end expression_for_neg_x_l826_826996


namespace count_integers_between_5_and_6_l826_826058

theorem count_integers_between_5_and_6 (n : ℕ) :
  (5 < Real.sqrt n ∧ Real.sqrt n < 6) → n ∈ {26, 27, 28, 29, 30, 31, 32, 33, 34, 35} :=
by {
  intros h,
  sorry
}

end count_integers_between_5_and_6_l826_826058


namespace cyclic_quadrilateral_area_l826_826237

theorem cyclic_quadrilateral_area 
  (A B C D : Type) 
  (AB BC CD DA : ℝ)
  (h1 : AB = 2) 
  (h2 : BC = 6) 
  (h3 : CD = 4) 
  (h4 : DA = 4) 
  (is_cyclic_quad : True) : 
  area A B C D = 8 * Real.sqrt 3 := 
sorry

end cyclic_quadrilateral_area_l826_826237


namespace set_B_equals_1_4_l826_826287

open Set

def U : Set ℕ := {1, 2, 3, 4}
def C_U_B : Set ℕ := {2, 3}

theorem set_B_equals_1_4 : 
  ∃ B : Set ℕ, B = {1, 4} ∧ U \ B = C_U_B := by
  sorry

end set_B_equals_1_4_l826_826287


namespace complex_number_z_l826_826750

theorem complex_number_z (i : ℂ) (z : ℂ) (hi : i * i = -1) (h : 2 * i / z = 1 - i) : z = -1 + i :=
by
  sorry

end complex_number_z_l826_826750


namespace smallest_positive_multiple_of_23_mod_89_is_805_l826_826494

theorem smallest_positive_multiple_of_23_mod_89_is_805 : 
  ∃ a : ℕ, 23 * a ≡ 4 [MOD 89] ∧ 23 * a = 805 := 
by
  sorry

end smallest_positive_multiple_of_23_mod_89_is_805_l826_826494


namespace basketball_statistics_l826_826179

/- The given scores of the first 11 games and the score of the 12th game -/
def scores_11 : List ℕ := [42, 47, 53, 53, 58, 58, 58, 61, 64, 65, 73]
def score_12 : ℕ := 80

/- Definitions of range, median, mean, mode, and mid-range -/
def range (l : List ℕ) : ℕ := l.maximum.getOrElse 0 - l.minimum.getOrElse 0

def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 1 then
    nthLe sorted (sorted.length / 2) sorry
  else
    (nthLe sorted (sorted.length / 2 - 1) sorry + nthLe sorted (sorted.length / 2) sorry) / 2

def mean (l : List ℕ) : ℚ := l.sum / l.length

def mode (l : List ℕ) : List ℕ :=
  let counts := l.foldl (λ m e => m.insertWith nat.add e 1) (Std.RBMap.empty ℕ ℕ)
  let max_count := counts.toList.maximumBy (λ e => e.snd).getOrElse (0, 0).snd
  counts.toList.filter (λ e => e.snd = max_count).map (λ e => e.fst)

def mid_range (l : List ℕ) : ℚ := (l.maximum.getOrElse 0 + l.minimum.getOrElse 0) / 2

/- Lean statement that needs to be proven -/
theorem basketball_statistics :
  let old_stats := (range scores_11, mean scores_11, mid_range scores_11)
  let new_scores := scores_11 ++ [score_12]
  let new_stats := (range new_scores, mean new_scores, mid_range new_scores)
  new_stats.1 > old_stats.1 ∧ new_stats.2 > old_stats.2 ∧ new_stats.3 > old_stats.3 := by
      sorry

end basketball_statistics_l826_826179


namespace find_angle_A_l826_826711

theorem find_angle_A (a b B A : ℝ) (h_triangle : a = sqrt 2 ∧ b = 2 ∧ sin B + cos B = sqrt 2) : A = π / 6 :=
  sorry

end find_angle_A_l826_826711


namespace coefficient_of_x5_in_product_l826_826938

noncomputable def polynomial1 : Polynomial ℤ := Polynomial.mk [ -2, 3, -7, 6, -4, 1 ]
noncomputable def polynomial2 : Polynomial ℤ := Polynomial.mk [ 8, 1, 5, -2, 3 ]

theorem coefficient_of_x5_in_product :
  (polynomial1 * polynomial2).coeff 5 = 23 :=
  by
    -- Definitions and calculations required would go here
    sorry

end coefficient_of_x5_in_product_l826_826938


namespace square_side_length_l826_826154

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
by
sorry

end square_side_length_l826_826154


namespace sqrt_div_sqrt_proof_l826_826180

noncomputable def sqrt_div_sqrt_statement : Prop :=
  ∀ (x y : ℝ),
    ( ( (1 / 3)^2 + (1 / 4)^2 ) / ( (1 / 5)^2 + (1 / 6)^2 ) = 21 * x / (65 * y) ) →
    ( √x / √y = (25 * √65) / (2 * √1281) )

theorem sqrt_div_sqrt_proof : sqrt_div_sqrt_statement :=
by
  intro x y h
  sorry

end sqrt_div_sqrt_proof_l826_826180


namespace a_n_formula_S_n_formula_l826_826054

noncomputable section

def a_seq : ℕ → ℝ
| 0       := 0 -- Placeholder for 0-th term, not used in actual problem
| 1       := 1 / 2
| (n + 1) := (n + 1 : ℝ) / (2 * n : ℝ) * a_seq n

def S_n (n : ℕ) : ℝ :=
(nat.range n).sum (λ i, a_seq (i + 1))

theorem a_n_formula (n : ℕ) : a_seq n = n * (1 / 2) ^ n :=
sorry

theorem S_n_formula (n : ℕ) : S_n n = 2 - (1 / 2) ^ (n - 1) - n * (1 / 2) ^ n :=
sorry

end a_n_formula_S_n_formula_l826_826054


namespace geometric_inequalities_l826_826430

variables {R r p a b c : ℝ} -- Define the variables for the proof

-- Define the given conditions
def geometric_identity (R r p abc : ℝ) : Prop :=
  R * r = abc / (4 * p)

def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

def inequality_1 (a b c : ℝ) : Prop :=
  27 * a * b * c ≤ (a + b + c)^3 / 2

-- Problem statement in Lean
theorem geometric_inequalities
  (R r p a b c : ℝ)
  (h1 : geometric_identity R r p (a * b * c))
  (h2 : semiperimeter a b c = p)
  (h3 : inequality_1 a b c) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 :=
sorry -- skipping the proof

end geometric_inequalities_l826_826430


namespace compute_a_plus_b_l826_826364

theorem compute_a_plus_b : ∃ (a b : ℝ), (∀ (x : ℝ), (x^3 - 9 * x^2 + a * x - b = 0) ↔ (x = 1) ∨ (x = 3) ∨ (x = 5)) ∧ (a + b = 38) :=
by {
  existsi (23 : ℝ),
  existsi (15 : ℝ),
  sorry
}

end compute_a_plus_b_l826_826364


namespace mouse_get_farther_from_cheese_l826_826877

def cheese_location : ℝ × ℝ := (16, 8)

def mouse_path_line : ℝ → ℝ := λ x, -3 * x + 10

noncomputable def distance_decreasing_point : ℝ × ℝ :=
let slope_perpendicular := 1 / 3 in
let perpendicular_line := λ x, slope_perpendicular * x + 8 in
let common_point := ((2 : ℝ) / 5, 58 / 5) in
    common_point -- (0.6, 8.2)

theorem mouse_get_farther_from_cheese :
    distance_decreasing_point = (0.6, 8.2) := sorry

end mouse_get_farther_from_cheese_l826_826877


namespace i_power_2016_l826_826521

-- Definition of the imaginary unit i.
def i := Complex.I

-- Given condition: i^2 = -1
lemma i_squared_eq_neg_one : i^2 = -1 := by
  rw [i, Complex.I_mul_I, Complex.of_real_neg, Complex.of_real_one]
  norm_num

-- Theorem to prove: i^2016 = 1
theorem i_power_2016 : i^2016 = 1 := by
  sorry

end i_power_2016_l826_826521


namespace range_of_b_l826_826624

variable (a b c : ℝ)

theorem range_of_b (h1 : a + b + c = 9) (h2 : a * b + b * c + c * a = 24) : 1 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l826_826624


namespace sum_of_even_sequence_is_194_l826_826253

theorem sum_of_even_sequence_is_194
  (a b c d : ℕ) 
  (even_a : a % 2 = 0) 
  (even_b : b % 2 = 0) 
  (even_c : c % 2 = 0) 
  (even_d : d % 2 = 0)
  (a_lt_b : a < b) 
  (b_lt_c : b < c) 
  (c_lt_d : c < d)
  (diff_da : d - a = 90)
  (arith_ab_c : 2 * b = a + c)
  (geo_bc_d : c^2 = b * d)
  : a + b + c + d = 194 := 
sorry

end sum_of_even_sequence_is_194_l826_826253


namespace minimum_positive_period_domain_extrema_l826_826372

noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.cos (Real.pi * x))^2 - 4 * Real.sqrt 3 * (Real.sin (Real.pi * x)) * (Real.cos (Real.pi * x))

theorem minimum_positive_period :
  ∃ π_ : ℝ, π_ > 0 ∧ ∀ x : ℝ, f (x + π_) = f x :=
begin
  use 1,
  split,
  { exact zero_lt_one, },
  { intro x,
    -- Here we would provide the proof that the period is 1
    sorry, }
end

theorem domain_extrema :
  (∀ x ∈ set.Icc (-Real.pi/3) (Real.pi/6), f x ≤ 6) ∧
  (∀ x ∈ set.Icc (-Real.pi/3) (Real.pi/6), f (-Real.pi/6) = 6) ∧
  (∀ x ∈ set.Icc (-Real.pi/3) (Real.pi/6), f x ≥ 0) ∧
  (∀ x ∈ set.Icc (-Real.pi/3) (Real.pi/6), f (Real.pi/6) = 0) :=
begin
  split,
  { intro x,
    intro hx,
    -- Here we would provide the proof for the upper bound of 6
    sorry, },
  split,
  { intro x,
    intro hx,
    -- Here we would provide the proof that the maximum is attained at -π/6
    sorry, },
  split,
  { intro x,
    intro hx,
    -- Here we would provide the proof for the lower bound of 0
    sorry, },
  { intro x,
    intro hx,
    -- Here we would provide the proof that the minimum is attained at π/6
    sorry, }
end

end minimum_positive_period_domain_extrema_l826_826372


namespace gcd_102_238_l826_826453

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l826_826453


namespace num_values_passing_through_vertex_l826_826222

-- Define the parabola and line
def parabola (a : ℝ) : ℝ → ℝ := λ x, x^2 + a^2
def line (a : ℝ) : ℝ → ℝ := λ x, 2 * x + a

-- Define the vertex condition 
def passes_through_vertex (a : ℝ) : Prop :=
  parabola a 0 = line a 0

-- Prove there are exactly 2 values of a that satisfy the condition
theorem num_values_passing_through_vertex : 
  {a : ℝ | passes_through_vertex a}.finite ∧ 
  {a : ℝ | passes_through_vertex a}.toFinset.card = 2 := 
sorry

end num_values_passing_through_vertex_l826_826222


namespace log_stack_total_l826_826545

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l826_826545


namespace sum_bn_first_n_terms_l826_826284

-- Definition of the sequence a_n
def a_n (n : ℕ) : ℝ := n / 2

-- Definition of the sequence b_n based on a_n
def b_n (n : ℕ) : ℝ := 4 * (1 / n - 1 / (n + 1))

-- Theorem statement to prove the sum of the first n terms of b_n
theorem sum_bn_first_n_terms (n : ℕ) : 
  (∑ k in Finset.range n, b_n (k + 1)) = 4 * n / (n + 1) :=
by sorry

end sum_bn_first_n_terms_l826_826284


namespace find_all_valid_pairs_l826_826586

def valid_pairs (a b : ℕ) : Prop :=
  ∃ N : ℕ, ∀ m n : ℕ, m ≥ N → n ≥ N → 
    ∃ (grid_partitioned : (ℕ → ℕ → Prop)), 
       (∀ i j : ℕ, grid_partitioned i j → (∃ r c : ℕ, r ≤ a ∧ c ≤ b)) 
       ∧ (card (set_of (λ p : m × n, ¬ grid_partitioned p.1 p.2)) < a * b)

theorem find_all_valid_pairs :
  { (a, b) | valid_pairs a b } = 
    { (1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) } := sorry

end find_all_valid_pairs_l826_826586


namespace arithmetic_sequence_solution_l826_826469

noncomputable def a_seq (n : ℕ) : ℤ := 13 - 3 * n

def b_seq (n : ℕ) : ℚ := 1 / (a_seq n * a_seq (n + 1))

def T_n (n : ℕ) : ℚ := -(1 / 3) * (1 / 10 + 1 / (3 * n - 10))

theorem arithmetic_sequence_solution (n : ℕ) (a1 : ℤ) (a2 : ℤ) (S_inequality : Sum (List.range n).map a_seq ≤ Sum (List.range 4).map a_seq)
  (h1 : a1 = 10) (h2 : a2 = 7) (h3 : a_seq 1 = a1) (h4 : a_seq 2 = a2) :
  (a_seq n = 13 - 3 * n) ∧ (Finset.sum (Finset.range n) b_seq = T_n n) :=
by
  sorry

end arithmetic_sequence_solution_l826_826469


namespace smallest_base_for_100_l826_826495

theorem smallest_base_for_100 :
  ∃ b : ℕ, b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
sorry

end smallest_base_for_100_l826_826495


namespace people_not_liking_both_l826_826011

theorem people_not_liking_both (n : ℕ) (p_TV p_T_G : ℝ) (h_n : n = 1500) (h_p_TV : p_TV = 0.25) (h_p_T_G : p_T_G = 0.15) : 
  ∃ (k : ℤ), k = 56 :=
by
  -- Calculate number of people who do not like television
  let num_TV := p_TV * n
  -- Calculate number of people who do not like both television and games
  let num_T_G := p_T_G * num_TV
  -- Round to nearest whole number
  let result := (num_T_G).round
  -- State the final result is 56
  use result
  sorry

end people_not_liking_both_l826_826011


namespace front_wheel_revolutions_l826_826462

theorem front_wheel_revolutions (P_front P_back : ℕ) (R_back : ℕ) (H1 : P_front = 30) (H2 : P_back = 20) (H3 : R_back = 360) :
  ∃ F : ℕ, F = 240 := by
  sorry

end front_wheel_revolutions_l826_826462


namespace correct_transformation_l826_826440

theorem correct_transformation (x : ℝ) :
  (6 * ((2 * x + 1) / 3) - 6 * ((10 * x + 1) / 6) = 6) ↔ (4 * x + 2 - 10 * x - 1 = 6) :=
by
  sorry

end correct_transformation_l826_826440


namespace expand_remains_same_l826_826206

variable (m n : ℤ)

-- Define a function that represents expanding m and n by a factor of 3
def expand_by_factor_3 (m n : ℤ) : ℤ := 
  2 * (3 * m) / (3 * m - 3 * n)

-- Define the original fraction
def original_fraction (m n : ℤ) : ℤ :=
  2 * m / (m - n)

-- Theorem to prove that expanding m and n by a factor of 3 does not change the fraction
theorem expand_remains_same (m n : ℤ) : 
  expand_by_factor_3 m n = original_fraction m n := 
by sorry

end expand_remains_same_l826_826206


namespace value_of_y_l826_826368

def k (y x : ℝ) : ℝ := y / (x ^ (1/3))

theorem value_of_y (y : ℝ) (x : ℝ) (k : ℝ) (Hk : k = 8 / (64 ^ (1 / 3))) :
  y = k * (27 ^ (1 / 3)) → y = 6 := by
  intro H
  rw [Hk]
  have : k = 2 := by
    have H1 : 64 ^ (1 / 3) = 4 := by norm_num
    rw [H1] at Hk
    exact (div_eq_iff (ne_of_gt (by norm_num : (4:ℝ) ≠ 0))).mp Hk
  rw [this] at H
  exact H

end value_of_y_l826_826368


namespace each_integer_has_at_least_two_digits_l826_826155

def interchange_problem (nums : List ℤ) (p q : ℤ) : Prop :=
  nums.length = 9 ∧
  list_sum nums / 9 - (list_sum (interchange digits nums p q)) / 9 = 1 ∧
  abs (p - q) = 1 → ∀ n ∈ nums, n ≥ 10

noncomputable def list_sum (lst : List ℤ) : ℤ :=
lst.foldr (λ x acc -> x + acc) 0

noncomputable def interchange_digits (n : ℤ) (p q : ℤ) : ℤ :=
if n % 10 = p ∧ n / 10 % 10 = q then 10 * p + q 
else if n % 10 = q ∧ n / 10 % 10 = p then 10 * q + p 
else n

noncomputable def interchange (lst : List ℤ) (p q : ℤ) : List ℤ :=
lst.map (λ x -> interchange_digits x p q)

theorem each_integer_has_at_least_two_digits (nums : List ℤ) (p q : ℤ) :
  interchange_problem nums p q :=
by
  sorry

end each_integer_has_at_least_two_digits_l826_826155


namespace a_7_eq_190_l826_826339

-- Define the sequence based on the given recurrence relation
def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n + 2

-- The theorem stating that a_7 = 190
theorem a_7_eq_190 : a 7 = 190 :=
sorry

end a_7_eq_190_l826_826339


namespace B_cycling_speed_l826_826902

variable (A_speed B_distance B_time B_speed : ℕ)
variable (t1 : ℕ := 7)
variable (d_total : ℕ := 140)
variable (B_catch_time : ℕ := 7)

theorem B_cycling_speed :
  A_speed = 10 → 
  d_total = 140 →
  B_catch_time = 7 → 
  B_speed = 20 :=
by
  sorry

end B_cycling_speed_l826_826902


namespace tom_trout_count_l826_826756

theorem tom_trout_count (M T : ℕ) (hM : M = 8) (hT : T = 2 * M) : T = 16 :=
by
  -- proof goes here
  sorry

end tom_trout_count_l826_826756


namespace g_function_ratio_l826_826802

theorem g_function_ratio (g : ℝ → ℝ) (h : ∀ c d : ℝ, c^3 * g d = d^3 * g c) (hg3 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := 
by
  sorry

end g_function_ratio_l826_826802


namespace carson_gardening_time_l826_826925

-- Definitions of the problem conditions
def lines_to_mow : ℕ := 40
def minutes_per_line : ℕ := 2
def rows_of_flowers : ℕ := 8
def flowers_per_row : ℕ := 7
def minutes_per_flower : ℚ := 0.5

-- Total time calculation for the proof 
theorem carson_gardening_time : 
  (lines_to_mow * minutes_per_line) + (rows_of_flowers * flowers_per_row * minutes_per_flower) = 108 := 
by 
  sorry

end carson_gardening_time_l826_826925


namespace hyperbola_point_distance_l826_826240

theorem hyperbola_point_distance 
(hyperbola_eq : ∀ (x y : ℝ), x^2 / 16 - y^2 / 36 = 1) 
(distance_to_focus : ∀ (x y : ℝ), (real.sqrt ((x - 2 * real.sqrt 13)^2 + y^2)) = 9)
: ∀ (x y : ℝ), hyperbola_eq x y → distance_to_focus x y → x^2 + y^2 = 133 :=
by
  intro x y h_hyperbola h_distance
  sorry

end hyperbola_point_distance_l826_826240


namespace triangle_area_eq_geometric_mean_l826_826710

-- We define the problem conditions
variables (A B C A1 B1 C1 A2 B2 C2 : Type) 
variable {α : Type*} [field α]

-- Variables for areas
variables (t1 t2 : α)

-- Given conditions
variables (inscribed : A1B1C1 ⊂ ABC)
          (circumscribed : A2B2C2 ⊃ ABC)
          (parallel_b1_b2 : A1B1 ∥ A2B2)
          (parallel_c1_c2 : A1C1 ∥ A2C2)
          (parallel_b1_c1_b2_c2 : B1C1 ∥ B2C2)
          (area_A1B1C1 : area A1B1C1 = t1)
          (area_A2B2C2 : area A2B2C2 = t2)

-- The theorem statement
theorem triangle_area_eq_geometric_mean :
  area ABC = (sqrt (t1 * t2)) :=
begin
  sorry
end

end triangle_area_eq_geometric_mean_l826_826710


namespace oc_length_l826_826135

open Real

theorem oc_length (O A B C : Point)
  (h_circle_centered_O : is_circle O 1)
  (h_A_on_circle : on_circle O 1 A)
  (h_tangent_AB_at_A : is_tangent AB A)
  (h_angle_AOB : angle O A B = π / 4)
  (h_C_on_OA : on_line_segment O A C)
  (h_BC_bisects_angle : bisects_angle C B O A B O)
  (s : ℝ) (hs : s = sin (π / 4))
  (c : ℝ) (hc : c = cos (π / 4)) :
  distance O C = (2 - sqrt 2) / 2 :=
by
  sorry

end oc_length_l826_826135


namespace at_least_one_not_less_than_2_l826_826824

theorem at_least_one_not_less_than_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) :=
sorry

end at_least_one_not_less_than_2_l826_826824


namespace find_y_l826_826975

theorem find_y (a b c x : ℝ) (p q r y: ℝ) (hx : x ≠ 1) 
  (h₁ : (Real.log a) / p = Real.log x) 
  (h₂ : (Real.log b) / q = Real.log x) 
  (h₃ : (Real.log c) / r = Real.log x)
  (h₄ : (b^3) / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
by {
  sorry
}

end find_y_l826_826975


namespace T_on_bisector_of_B_l826_826806

noncomputable def incircle_midline_intersection : Prop :=
  ∀ (A B C P Q R S T : Type) 
    (AB BC AC : ℝ) 
    (hAB_gt_BC : AB > BC)
    (incircle_touches_P_Q : touches_incircle A B C P Q) 
    (RS_midline_parallel_AB : midline_parallel RS A B) 
    (RS_intersects : intersects PQ RS T),
  lies_on_angle_bisector_b A B C T

/-- Assume that the incircle of triangle ABC touches the sides AB and AC at points P and Q respectively. RS is the midline parallel to side AB, and T is the intersection point of lines PQ and RS. Prove that point T lies on the bisector of angle B of triangle ABC. -/
theorem T_on_bisector_of_B (A B C P Q R S T : Type) 
    (AB BC AC : ℝ) 
    (hAB_gt_BC : AB > BC)
    (incircle_touches_P_Q : touches_incircle A B C P Q) 
    (RS_midline_parallel_AB : midline_parallel RS A B) 
    (RS_intersects : intersects PQ RS T) 
  : lies_on_angle_bisector_b A B C T :=
begin
  sorry
end

end T_on_bisector_of_B_l826_826806


namespace total_profit_is_100_l826_826144

-- Define the conditions
variables (A_investment : ℝ) (B_investment : ℝ) (A_duration : ℝ) (B_duration : ℝ)
          (A_share : ℝ) (total_profit : ℝ)

-- Given conditions
def conditions : Prop :=
  A_investment = 100 ∧
  B_investment = 200 ∧
  A_duration = 12 ∧
  B_duration = 6 ∧
  A_share = 50

-- Proof statement 
theorem total_profit_is_100 (h : conditions A_investment B_investment A_duration B_duration A_share total_profit) : total_profit = 100 :=
sorry

end total_profit_is_100_l826_826144


namespace find_k_l826_826633

theorem find_k (k : ℝ) (h : (-2)^2 - k * (-2) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l826_826633


namespace minimum_value_of_x_plus_2y_l826_826614

theorem minimum_value_of_x_plus_2y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 2y + 2 * x * y = 8) : 
  x + 2y ≥ 4 := 
sorry

end minimum_value_of_x_plus_2y_l826_826614


namespace area_equal_l826_826014

variables {A B C D E F K M : Type*}
variables [is_parallelogram : parallelogram A B C D]
variables [midpoint : midpoint E A B] [midpoint : midpoint F A D]
variables [intersect : intersection CE BF K]
variables [on_segment : on_segment M E C]
variables [parallel : parallel BM KD]

theorem area_equal {A B C D E F K M : Type*}
  [parallelogram A B C D] [midpoint E A B] [midpoint F A D]
  [intersection CE BF K] [on_segment M E C] [parallel BM KD] :
  area_of_triangle K F D = area_of_trapezoid K B M D :=
  sorry

end area_equal_l826_826014


namespace num_of_poly_sci_majors_l826_826012

-- Define the total number of applicants
def total_applicants : ℕ := 40

-- Define the number of applicants with GPA > 3.0
def gpa_higher_than_3_point_0 : ℕ := 20

-- Define the number of applicants who did not major in political science and had GPA ≤ 3.0
def non_poly_sci_and_low_gpa : ℕ := 10

-- Define the number of political science majors with GPA > 3.0
def poly_sci_with_high_gpa : ℕ := 5

-- Prove the number of political science majors
theorem num_of_poly_sci_majors : ∀ (P : ℕ),
  P = poly_sci_with_high_gpa + 
      (total_applicants - non_poly_sci_and_low_gpa - 
       (gpa_higher_than_3_point_0 - poly_sci_with_high_gpa)) → 
  P = 20 :=
by
  intros P h
  sorry

end num_of_poly_sci_majors_l826_826012


namespace numbers_left_on_board_l826_826421

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826421


namespace recommended_cups_water_l826_826075

variable (currentIntake : ℕ)
variable (increasePercentage : ℕ)

def recommendedIntake : ℕ := 
  currentIntake + (increasePercentage * currentIntake) / 100

theorem recommended_cups_water (h1 : currentIntake = 15) 
                               (h2 : increasePercentage = 40) : 
  recommendedIntake currentIntake increasePercentage = 21 := 
by 
  rw [h1, h2]
  have h3 : (40 * 15) / 100 = 6 := by norm_num
  rw [h3]
  norm_num
  sorry

end recommended_cups_water_l826_826075


namespace right_angle_triangle_probability_l826_826903

def points_on_circle : Prop :=
  ∀ (A B C D E F G H : Point), OnCircle A B C D E F G H CircleO → Equidistant A B C D E F G H CircleO

noncomputable def probability_right_angle_triangle (A B C D E F G H : Point) (h : points_on_circle) : ℚ :=
  (prob_equilateral_right_triangle A B C D E F G H CircleO) / (prob_all_possible_triangles A B C D E F G H)

theorem right_angle_triangle_probability :
  probability_right_angle_triangle A B C D E F G H h = 3/7 :=
by
  sorry

end right_angle_triangle_probability_l826_826903


namespace canoe_kayak_revenue_l826_826829

noncomputable def total_revenue (K : ℕ) : ℕ :=
  let C := (4 / 3) * K in
  9 * C + 12 * K

theorem canoe_kayak_revenue :
  ∃ (K : ℕ), 4 * K = 3 * K + 18 ∧ total_revenue K = 432 :=
by
  sorry

end canoe_kayak_revenue_l826_826829


namespace abs_difference_extrema_l826_826099

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l826_826099


namespace minimum_sum_of_dimensions_of_box_l826_826037

theorem minimum_sum_of_dimensions_of_box (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 2310) :
  a + b + c ≥ 52 :=
sorry

end minimum_sum_of_dimensions_of_box_l826_826037


namespace star_hexagon_perimeter_l826_826357

theorem star_hexagon_perimeter (ABCDEF : hexagon) (h_equiangular : equiangular ABCDEF) (h_perimeter : perimeter ABCDEF = 1) : 
  let star_perimeter := star_polygon_perimeter ABCDEF
  max_star_perimeter star_perimeter - min_star_perimeter star_perimeter = 0 :=
sorry

end star_hexagon_perimeter_l826_826357


namespace rectangle_area_proof_l826_826514

def rectangle_width : ℕ := 5

def rectangle_length (width : ℕ) : ℕ := 3 * width

def rectangle_area (length width : ℕ) : ℕ := length * width

theorem rectangle_area_proof : rectangle_area (rectangle_length rectangle_width) rectangle_width = 75 := by
  sorry -- Proof can be added later

end rectangle_area_proof_l826_826514


namespace calculate_expression_l826_826366

def f (x : ℝ) := 2 * x^2 - 3 * x + 1
def g (x : ℝ) := x + 2

theorem calculate_expression : f (1 + g 3) = 55 := 
by
  sorry

end calculate_expression_l826_826366


namespace solve_for_x_l826_826218

theorem solve_for_x :
  ∃ x : ℝ, ((17.28 / x) / (3.6 * 0.2)) = 2 ∧ x = 12 :=
by
  sorry

end solve_for_x_l826_826218


namespace solution_correct_l826_826815

-- Define the vertices
def V : Set ℂ := {
  sqrt 3 * Complex.I,
  -sqrt 3 * Complex.I,
  (1 / sqrt 9) * (1 + Complex.I),
  (1 / sqrt 9) * (-1 + Complex.I),
  (1 / sqrt 9) * (1 - Complex.I),
  (1 / sqrt 9) * (-1 - Complex.I),
  (1 / sqrt 9) * (2 + Complex.I),
  (1 / sqrt 9) * (-2 - Complex.I)
}

-- Define the function to choose z_j independently from V
def random_selection (V : Set ℂ) (n : ℕ) : List ℂ := sorry

-- Define the product P
def P (zs : List ℂ) : ℂ := List.prod zs

theorem solution_correct :
  let zs := random_selection V 14,
  let P  := P zs,
  let a := 3003,
  let b := 11,
  let p := 2,
  P = -1 → (∃ a b p : ℕ, p.prime ∧ (a % p ≠ 0) ∧ (P = -1) ∧ (a + b + p = 3016)) :=
sorry

end solution_correct_l826_826815


namespace count_non_congruent_triangles_l826_826293

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ a + b + c = 20

theorem count_non_congruent_triangles : 
  ∃ n : ℕ, n = 11 ∧ 
    (∀ (a b c : ℕ), valid_triangle a b c → 
      (a, b, c) ∈ {(3, 8, 9), (3, 7, 10), (3, 6, 11), (3, 5, 12),
                   (4, 7, 9), (4, 6, 10), (4, 5, 11),
                   (5, 6, 9), (5, 7, 8), (5, 5, 10),
                   (6, 6, 8), (6, 7, 7)} ∧
      (cardinality_of_possible_solutions = 11)) := sorry

end count_non_congruent_triangles_l826_826293


namespace inequality_a_b_c_d_l826_826745

theorem inequality_a_b_c_d
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h₄ : a * b + b * c + c * d + d * a = 1) :
  (a ^ 3 / (b + c + d) + b ^ 3 / (c + d + a) + c ^ 3 / (a + b + d) + d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_a_b_c_d_l826_826745


namespace number_of_even_divisors_of_9_factorial_multiple_of_15_l826_826659

theorem number_of_even_divisors_of_9_factorial_multiple_of_15 : 
  let n : ℕ := 9
  let div_15 := λ (k : ℕ), k % 15 = 0
  let is_even := λ (k : ℕ), k % 2 = 0
  let prime_factors_9_fact := 2 ^ 7 * 3 ^ 4 * 5 * 7
  let count_divisors := 
    (λ (fact : ℕ) (cond : ℕ → Prop), 
    (finset.range (fact + 1)).filter (λ k, fact % k = 0 ∧ cond k)).card
  count_divisors prime_factors_9_fact (λ k, div_15 k ∧ is_even k) = 56 := by sorry

end number_of_even_divisors_of_9_factorial_multiple_of_15_l826_826659


namespace directrix_of_parabola_y_eq_x_sq_l826_826491

def parabola := ∀ x : ℝ, y = x^2

def vertex_at_origin_and_axis_along_y (a : ℝ) := ∀ x : ℝ, y = a * x^2

def directrix_formula (a : ℝ) := y = - (1 / (4 * a))

theorem directrix_of_parabola_y_eq_x_sq : directrix_formula 1 :=
by
  -- conditions: parabola: y = x^2 and the value a=1.
  sorry

end directrix_of_parabola_y_eq_x_sq_l826_826491


namespace cyclic_quadrilateral_area_l826_826234

def area_of_cyclic_quadrilateral (AB BC CD DA : ℝ) : ℝ :=
  let A := 120 * (real.pi / 180) -- Angle in radians
  16 * (real.sin A)

theorem cyclic_quadrilateral_area
  (AB BC CD DA : ℝ)
  (hAB : AB = 2)
  (hBC : BC = 6)
  (hCD : CD = 4)
  (hDA : DA = 4)
  (hCyclic : true) -- Assumption that quadrilateral is cyclic
  : area_of_cyclic_quadrilateral AB BC CD DA = 8 * real.sqrt 3 :=
by
  rw [hAB, hBC, hCD, hDA]
  sorry

end cyclic_quadrilateral_area_l826_826234


namespace expected_value_xi_l826_826476

-- Defining the problem parameters
def n : ℕ := 4
def p : ℝ := 3 / 5

-- Defining the random variable ξ
def ξ : ℕ → PReal := binomial_randvar n p

-- The expected value E(ξ)
theorem expected_value_xi : (expected_value ξ) = 12 / 5 := by
  sorry

end expected_value_xi_l826_826476


namespace geometric_sequence_property_l826_826329

theorem geometric_sequence_property (b : ℕ → ℝ) (h : b 6 = 1) (n : ℕ) (hn : n < 11) :
  (∏ i in finset.range (n+1), b i) = (∏ i in finset.range (11-n), b i) :=
sorry

end geometric_sequence_property_l826_826329


namespace solution_set_of_inequality_l826_826303

noncomputable def greatest_integer_less_equal (x : ℝ) := floor x

theorem solution_set_of_inequality :
  {x : ℝ | |3 * x + 1| - greatest_integer_less_equal x - 3 ≤ 0} =
  {x : ℝ | -1 ≤ x ∧ x ≤ 2/3} ∪ {1} :=
by sorry

end solution_set_of_inequality_l826_826303


namespace range_of_f_l826_826198

def f (x : ℝ) : ℝ := cos (2 * x) + 2 * sin x

theorem range_of_f : set.range f = set.Icc (-3 : ℝ) (3 / 2) :=
by
  sorry

end range_of_f_l826_826198


namespace find_coordinates_of_P_l826_826701

noncomputable def point_on_axes (P : ℝ × ℝ) : Prop :=
  P.1 = 0 ∨ P.2 = 0

def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2))

def A : ℝ × ℝ := (6, 0)
def B : ℝ × ℝ := (0, 4)
def target_area : ℝ := 12

theorem find_coordinates_of_P (P : ℝ × ℝ) (h : point_on_axes P) (h_area : area_triangle A B P = target_area) :
  P = (12, 0) ∨ P = (0, 0) ∨ P = (0, 8) :=
sorry

end find_coordinates_of_P_l826_826701


namespace number_of_integers_in_sequence_l826_826053

theorem number_of_integers_in_sequence 
  (a_0 : ℕ) 
  (h_0 : a_0 = 8820) 
  (seq : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, seq (n + 1) = seq n / 3) :
  ∃ n : ℕ, seq n = 980 ∧ n + 1 = 3 :=
by
  sorry

end number_of_integers_in_sequence_l826_826053


namespace group_size_l826_826033

theorem group_size (n : ℕ) (weight_inc : n * 5.5 = 135.5 - 86) : n = 9 :=
sorry

end group_size_l826_826033


namespace triangle_and_circle_solution_l826_826186

noncomputable def triangle_and_circle_problem : Prop :=
  ∃ (D E F Q : Type) 
    (DF DE EF FQ : ℝ)
    (H_right : ∃ angle : ℝ, angle = 90),
    ∧ (DF = sqrt 85)
    ∧ (DE = 7)
    ∧ (H_tangent_circle: ∃ center : Type, (center ∈ DE) ∧ (tangent to DF ∧ EF))
    ∧ (Q ∈ DF)
    ∧ (FQ = 6)

theorem triangle_and_circle_solution : triangle_and_circle_problem := by
  sorry

end triangle_and_circle_solution_l826_826186


namespace students_neither_lit_nor_hist_l826_826322

theorem students_neither_lit_nor_hist (total_students : ℕ) (lit_students : ℕ) (hist_students : ℕ) (both_students : ℕ) 
  (h_total : total_students = 80)
  (h_lit : lit_students = 50)
  (h_hist : hist_students = 40)
  (h_both : both_students = 25) : 
  total_students - (lit_students - both_students + hist_students - both_students + both_students) = 15 := 
by
  rw [h_total, h_lit, h_hist, h_both]
  norm_num
  sorry

end students_neither_lit_nor_hist_l826_826322


namespace general_formula_sum_first_n_terms_l826_826472

noncomputable theory

open Real

-- Given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ)

-- (Condition 1) All terms are positive
axiom pos_terms : ∀ n, 0 < a n
-- (Condition 2) 2a_1 + 3a_2 = 1
axiom cond1 : 2 * a 1 + 3 * a 2 = 1
-- (Condition 3) a_3^2 = 9a_2a_6
axiom cond2 : a 3 ^ 2 = 9 * a 2 * a 6

-- (Part 1) General formula for the sequence \{a_n\}
theorem general_formula (n : ℕ) : a n = (1 / 3) ^ n :=
sorry

-- (Part 2) Sum of the first n terms of the sequence \{\frac{1}{b_n}\}
theorem sum_first_n_terms (n : ℕ) : 
    (finite_sum (λ (i : ℕ), if i < n then 1 / b (i + 1) else 0)) = -2 * n / (n + 1) :=
sorry

end general_formula_sum_first_n_terms_l826_826472


namespace subcommittee_count_l826_826129

theorem subcommittee_count :
  let totalWays : ℕ :=
    -- 2 Republicans and 3 Democrats
    (@nat.choose 10 2) * (@nat.choose 8 3) +
    -- 3 Republicans and 2 Democrats
    (@nat.choose 10 3) * (@nat.choose 8 2) +
    -- 4 Republicans and 1 Democrat
    (@nat.choose 10 4) * (@nat.choose 8 1) +
    -- 5 Republicans and 0 Democrats
    (@nat.choose 10 5) * (@nat.choose 8 0)
  in totalWays = 7812 := by
  sorry

end subcommittee_count_l826_826129


namespace log_sum_correct_l826_826498

noncomputable def log_sum : Prop :=
  let x := (3/2)
  let y := (5/3)
  (x + y) = (19/6)

theorem log_sum_correct : log_sum :=
by
  sorry

end log_sum_correct_l826_826498


namespace proof_l_shaped_area_l826_826880

-- Define the overall rectangle dimensions
def overall_length : ℕ := 10
def overall_width : ℕ := 7

-- Define the dimensions of the removed rectangle
def removed_length : ℕ := overall_length - 3
def removed_width : ℕ := overall_width - 2

-- Calculate the areas
def overall_area : ℕ := overall_length * overall_width
def removed_area : ℕ := removed_length * removed_width
def l_shaped_area : ℕ := overall_area - removed_area

-- The theorem to be proved
theorem proof_l_shaped_area : l_shaped_area = 35 := by
  sorry

end proof_l_shaped_area_l826_826880


namespace value_of_y_l826_826318

theorem value_of_y :
  let x := (70 - 50 + 1) / 2 * (50 + 70)
  let y := (70 - 50) / 2 + 1 in
  x + y = 1271 → y = 11 :=
by
  let x := (70 - 50 + 1) / 2 * (50 + 70)
  let y := (70 - 50) / 2 + 1
  intro h
  sorry

end value_of_y_l826_826318


namespace longest_side_of_triangle_l826_826621

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

theorem longest_side_of_triangle :
  let A := (1, 1)
  let B := (4, 7)
  let C := (8, 1)
  max (distance A B) (max (distance A C) (distance B C)) = 2 * real.sqrt 13 :=
by
  let A := (1, 1)
  let B := (4, 7)
  let C := (8, 1)
  have h1 : distance A B = real.sqrt 45, by sorry
  have h2 : distance A C = 7, by sorry
  have h3 : distance B C = real.sqrt 52, by sorry
  have h_max : max (real.sqrt 45) (max 7 (real.sqrt 52)) = real.sqrt 52, by sorry
  rw h_max
  exact eq.trans h3 (eq.symm (real.sqrt_eq_rpow (52))))

-- since real.sqrt 52 is equal to 2 * real.sqrt 13

end longest_side_of_triangle_l826_826621


namespace sin2_plus_cos2_eq_one_l826_826747

theorem sin2_plus_cos2_eq_one (θ : ℝ) : let s := Real.sin θ 
                                          let c := Real.cos θ in
                                          s^2 + c^2 = 1 :=
by
  intro s c
  have h1 : s^2 = Real.sin θ ^ 2 := by rfl
  have h2 : c^2 = Real.cos θ ^ 2 := by rfl
  rw [h1, h2]
  exact Real.sin_sq_add_cos_sq θ

end sin2_plus_cos2_eq_one_l826_826747


namespace unit_digit_3_pow_2023_l826_826007

def unit_digit_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0

theorem unit_digit_3_pow_2023 : unit_digit_pattern 2023 = 7 :=
by sorry

end unit_digit_3_pow_2023_l826_826007


namespace proof_x_squared_plus_y_squared_l826_826741

noncomputable def find_x_squared_plus_y_squared (a : ℝ) : ℝ :=
  let x := a^1501 - a^(-1501)
  let y := a^1501 + a^(-1501)
  x^2 + y^2

theorem proof_x_squared_plus_y_squared :
  find_x_squared_plus_y_squared 3002 = 2 * (3002^3002 + 3002^(-3002)) :=
by
  sorry

end proof_x_squared_plus_y_squared_l826_826741


namespace emergency_vehicle_reachable_area_l826_826324

theorem emergency_vehicle_reachable_area :
  let speed_roads := 60 -- velocity on roads in miles per hour
    let speed_sand := 10 -- velocity on sand in miles per hour
    let time_limit := 5 / 60 -- time limit in hours
    let max_distance_on_roads := speed_roads * time_limit -- max distance on roads
    let radius_sand_circle := (10 / 12) -- radius on the sand
    -- calculate area covered
  (5 * 5 + 4 * (1 / 4 * Real.pi * (radius_sand_circle)^2)) = (25 + (25 * Real.pi) / 36) :=
by
  sorry

end emergency_vehicle_reachable_area_l826_826324


namespace find_sum_l826_826509

theorem find_sum (P R : ℝ) (T : ℝ) (hT : T = 3) (h1 : P * (R + 1) * 3 = P * R * 3 + 2500) : 
  P = 2500 := by
  sorry

end find_sum_l826_826509


namespace original_number_is_3_l826_826797

theorem original_number_is_3 
  (A B C D E : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 8) 
  (h2 : (8 + B + C + D + E) / 5 = 9): 
  A = 3 :=
sorry

end original_number_is_3_l826_826797


namespace count_valid_pairs_l826_826664

theorem count_valid_pairs : ∃ (m n : ℕ), (n = 3 * m) ∧ 
                            (m / 100 ∈ {1, 2, 3, 6, 7, 8}) ∧
                            ((m % 100) / 10 ∈ {1, 2, 3, 6, 7, 8}) ∧
                            ((m % 10) ∈ {1, 2, 3, 6, 7, 8}) ∧
                            (m < 1000) ∧ (100 ≤ m) ∧ (n < 1000) ∧ (100 ≤ n) ∧
                            (∀ d ∈ {1, 2, 3, 6, 7, 8}, count d m + count d n = 1) ∧
                            (∃! t : fin 3, (m * 100 + t) = 261 * 100 + t ∨ 
                                            (m * 100 + t) = 126 * 100 + t) :=
by
  sorry

end count_valid_pairs_l826_826664


namespace train_speed_l826_826896

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l826_826896


namespace evaluate_integral_l826_826583
-- Assuming we use Mathlib

open Real

theorem evaluate_integral : ∫ x in 1..2, (2 * x - 1) = 2 := by
  sorry

end evaluate_integral_l826_826583


namespace probability_no_physics_and_chemistry_l826_826048

-- Define the probabilities for the conditions
def P_physics : ℚ := 5 / 8
def P_no_physics : ℚ := 1 - P_physics
def P_chemistry_given_no_physics : ℚ := 2 / 3

-- Define the theorem we want to prove
theorem probability_no_physics_and_chemistry :
  P_no_physics * P_chemistry_given_no_physics = 1 / 4 :=
by sorry

end probability_no_physics_and_chemistry_l826_826048


namespace at_least_two_dice_same_l826_826926

theorem at_least_two_dice_same (num_dice : ℕ) (num_faces : ℕ) (fair_dice : (i : ℕ) → 1 ≤ i ∧ i ≤ num_faces) :
  num_dice = 8 ∧ num_faces = 8 → 
  let prob := 1 - (Nat.factorial num_faces : ℚ) / num_faces^num_dice in
  prob = 415 / 416 :=
by { sorry }

end at_least_two_dice_same_l826_826926


namespace angle_AOC_is_180_degrees_l826_826161

def latitude (p : ℝ × ℝ) : ℝ := p.1
def longitude (p : ℝ × ℝ) : ℝ := p.2

def point_A := (0, 110)
def point_C := (60, -100)
def center_O := (0,0)

theorem angle_AOC_is_180_degrees : 
  ∀ (A C O : (ℝ × ℝ)), 
  (latitude A = 0) → (longitude A = 110) →
  (latitude C = 60) → (longitude C = -100) →
  (latitude O = 0) → (longitude O = 0) →
  angle A O C = 180 :=
by
  intro A C O
  intro hA_lat hA_long hC_lat hC_long hO_lat hO_long
  sorry

end angle_AOC_is_180_degrees_l826_826161


namespace girls_with_short_hair_count_l826_826063

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l826_826063


namespace final_position_after_120_moves_l826_826537

open Complex

noncomputable def cis (θ : ℝ) := Complex.mk (Real.cos θ) (Real.sin θ)

def initial_position : ℂ := Complex.mk 6 0

def move (z : ℂ) : ℂ := cis (Real.pi / 6) * z + Complex.mk 8 0

def final_position : ℕ → ℂ
| 0       => initial_position
| (n + 1) => move (final_position n)

theorem final_position_after_120_moves :
  final_position 120 = Complex.mk 6 0 := 
sorry

end final_position_after_120_moves_l826_826537


namespace pattern_equation_sum_calculation_generalized_sum_l826_826380

theorem pattern_equation (n : ℕ) (hn : 0 < n) : 3^(n+1) - 3^n = 2 * 3^n := 
sorry

theorem sum_calculation : 2 * 3^1 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 = 726 := 
sorry

theorem generalized_sum (n : ℕ) (hn : 0 < n) : 3^1 + 3^2 + ... + 3^n = 1 / 2 * (3^(n+1) - 3) := 
sorry

end pattern_equation_sum_calculation_generalized_sum_l826_826380


namespace petya_numbers_l826_826398

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826398


namespace velocity_at_t1_l826_826527

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2 * t

-- Define the velocity function as the derivative of s
def velocity (t : ℝ) : ℝ := -2 * t + 2

-- Prove that the velocity at t = 1 is 0
theorem velocity_at_t1 : velocity 1 = 0 :=
by
  -- Apply the definition of velocity
    sorry

end velocity_at_t1_l826_826527


namespace sum_distances_less_than_perimeter_l826_826019

theorem sum_distances_less_than_perimeter
  (A B C D : ℝ^3) (O : ℝ^3)
  (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ D) (hD : D ≠ A)
  (hAB : 0 < dist(A,B)) (hBC : 0 < dist(B,C)) 
  (hCD : 0 < dist(C,D)) (hDA : 0 < dist(D,A)) 
  (hAC : 0 < dist(A,C)) (hBD : 0 < dist(B,D))
  (h_in_tetrahedron : is_inside_tetrahedron O A B C D):
  dist(O, A) + dist(O, B) + dist(O, C) + dist(O, D) < 
  dist(A, B) + dist(B, C) + dist(C, D) + dist(D, A) + dist(A, C) + dist(B, D) := 
sorry

end sum_distances_less_than_perimeter_l826_826019


namespace prove_problem1_prove_problem2_l826_826855

open Real

noncomputable def problem1 (n : ℕ) (x : Fin n → ℝ) : Prop :=
  (∑ i, ∑ j, abs (x i - x j))^2 ≤ (2 * (n^2 - 1) / 3) * ∑ i, ∑ j, (x i - x j)^2

noncomputable def problem2 (n : ℕ) (x : Fin n → ℝ) : Prop :=
  ((∑ i, ∑ j, abs (x i - x j))^2 = (2 * (n^2 - 1) / 3) * ∑ i, ∑ j, (x i - x j)^2) ↔
  ∃ d : ℝ, ∃ a : ℝ, ∀ i : Fin n, x i = a + d * (i : ℕ)

theorem prove_problem1 (n : ℕ) (x : Fin n → ℝ) : problem1 n x :=
sorry

theorem prove_problem2 (n : ℕ) (x : Fin n → ℝ) : problem2 n x :=
sorry

end prove_problem1_prove_problem2_l826_826855


namespace polynomial_divisibility_l826_826783

theorem polynomial_divisibility (a : ℤ) (n : ℕ) (h_pos : 0 < n) : 
  (a ^ (2 * n + 1) + (a - 1) ^ (n + 2)) % (a ^ 2 - a + 1) = 0 :=
sorry

end polynomial_divisibility_l826_826783


namespace log_stack_total_l826_826544

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l826_826544


namespace possible_to_cut_and_assemble_l826_826933

def cut_and_assemble_equilateral_triangle (T : Triangle) : Prop :=
  equilateral T ∧ exists (cuts : list Line), (length cuts = 6 ∧ 
  form_identical_equilateral_triangles T (apply_cuts T cuts) 7)

theorem possible_to_cut_and_assemble (T : Triangle) (h : equilateral T) :
  cut_and_assemble_equilateral_triangle T :=
by
  sorry

end possible_to_cut_and_assemble_l826_826933


namespace path_exists_l826_826889

structure Graph (V : Type) :=
  (adj : V → V → ℕ)  -- adjacency matrix with edge multiplicities

def even_deg (G : Graph V) (v : V) : Prop :=
  ∑ w, G.adj v w % 2 = 0

def odd_deg (G : Graph V) (v : V) : Prop :=
  ∑ w, G.adj v w % 2 = 1

theorem path_exists (G : Graph V) (A B : V) (path_exists : ∀ (u v : V), Prop) : 
  (odd_deg G A) ∧ (odd_deg G B) ∧ (∀ x ≠ A ≠ B, even_deg G x) → ∃ (path : list V), (path.head = B ∧ path.last = A) ∧ 
  (∀ (u v : V), (u, v) ∈ path.zip (path.tail) → G.adj u v % 2 = 1) :=
sorry

end path_exists_l826_826889


namespace find_valid_n_values_l826_826466

open List

def is_median_and_mean (S : List ℝ) (m : ℝ) : Prop :=
  let sorted_S := sort (≤) S
  (sorted_S.length % 2 = 1 ∧ sorted_S.get! (sorted_S.length / 2) = m) ∧
  (sorted_S.sum / sorted_S.length = m)

theorem find_valid_n_values :
  let S := [4, 7, 11, 12]
  ∃ n : ℝ, n ∉ S ∧ 
    (is_median_and_mean (n :: S) 7 ∨
     is_median_and_mean (n :: S) 11) →
    n = 1 ∨ n = 21 ∧ (n = 1 ∨ n = 21) →
    1 + 21 = 22 :=
by
  sorry

end find_valid_n_values_l826_826466


namespace surface_area_of_inscribed_sphere_l826_826541

theorem surface_area_of_inscribed_sphere {r : ℝ} (h : r = 1) : 4 * Real.pi * r^2 = 4 * Real.pi :=
by
  rw [h]
  norm_num
  exact mul_one (4 * Real.pi)

example : surface_area_of_inscribed_sphere (by norm_num) := sorry

end surface_area_of_inscribed_sphere_l826_826541


namespace locus_of_midpoints_parallel_l826_826954

-- Definitions for the line l and point A
variables (l : AffineSubspace ℝ ℝ) (A : AffineSpace.Point ℝ ℝ)

-- Conditions ensuring A is not on line l
axiom A_not_on_l : ¬ (A ∈ l)

-- Definition of the locus of midpoints M of segments AB with B on line l
def midpoint (A B : AffineSpace.Point ℝ ℝ) : AffineSpace.Point ℝ ℝ :=
  (A + B) / 2

-- Statement of the geometric property
theorem locus_of_midpoints_parallel (B : AffineSpace.Point ℝ ℝ) (h : B ∈ l) :
  ∃ m : AffineSubspace ℝ ℝ, (∀ B ∈ l, let M := midpoint A B in M ∈ m) ∧ m ∥ l :=
begin
  sorry
end

end locus_of_midpoints_parallel_l826_826954


namespace triangle_to_five_isosceles_l826_826017

theorem triangle_to_five_isosceles (ABC : Triangle) : 
  ∃ (Δ1 Δ2 Δ3 Δ4 Δ5 : IsoscelesTriangle), 
    (Δ1 ∪ Δ2 ∪ Δ3 ∪ Δ4 ∪ Δ5 = ABC) ∧
    (∀ i j, i ≠ j → Interior(Δi) ∩ Interior(Δj) = ∅) := 
sorry

end triangle_to_five_isosceles_l826_826017


namespace difference_in_cents_l826_826432

-- Given definitions and conditions
def number_of_coins : ℕ := 3030
def min_nickels : ℕ := 3
def ratio_pennies_to_nickels : ℕ := 10

-- Problem statement: Prove that the difference in cents between the maximum and minimum monetary amounts is 1088
theorem difference_in_cents (p n : ℕ) (h1 : p + n = number_of_coins)
  (h2 : p ≥ ratio_pennies_to_nickels * n) (h3 : n ≥ min_nickels) :
  4 * 275 = 1100 ∧ (3030 + 1100) - (3030 + 4 * 3) = 1088 :=
by {
  sorry
}

end difference_in_cents_l826_826432


namespace area_difference_is_correct_l826_826573

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def area_circle (diameter : ℝ) : ℝ := (Real.pi * (diameter / 2) ^ 2)

noncomputable def combined_area_difference : ℝ :=
  (area_rectangle 11 11 + area_rectangle 5.5 11) - 
  (area_equilateral_triangle 6 + area_circle 4)
 
theorem area_difference_is_correct :
  |combined_area_difference - 153.35| < 0.001 :=
by
  sorry

end area_difference_is_correct_l826_826573


namespace ellipse_equation_at_origin_l826_826254

theorem ellipse_equation_at_origin (P Q : ℝ → ℝ → Prop) :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1) ∧
    (∀ x y, P x y ↔ y = x + 1) ∧
    (∀ x y, Q x y ↔ y = x + 1) ∧
    (∀ x y, ∃ O P Q, OP ⊥ OQ ∧ |PQ| = sqrt 10 / 2)) →
  (∃ a b : ℝ, 
    (a^2 = 2 ∧ b^2 = 2 / 3) ∨
    (3 * a^2 = 2 ∧ a^2 = 2)) ∧
    ((∃ x y, x^2 + 3 * y^2 = 2) ∨ (3 * x^2 + y^2 = 2)) :=
by
  sorry

end ellipse_equation_at_origin_l826_826254


namespace quadrilateral_area_l826_826031

-- Define the angles A, B, and C and the height H
variables {A B C H : ℝ}

-- Define the theorem statement
theorem quadrilateral_area (h_triangle : A + B + C = π)
  : ∃ S : ℝ, S = (H * H / 2) * (cos (A - C) * sin B) :=
sorry

end quadrilateral_area_l826_826031


namespace dot_product_u_v_l826_826211

def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 1, 2, 3)

theorem dot_product_u_v : (4 * -6 + -3 * 1 + 5 * 2 + -2 * 3) = -23 := by
  sorry

end dot_product_u_v_l826_826211


namespace chord_length_eq_sqrt14_l826_826041

open Real

-- Given conditions
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4
def chord_length (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Proven statement
theorem chord_length_eq_sqrt14 :
  ∃ (x1 y1 x2 y2 : ℝ), line_eqn x1 y1 ∧ line_eqn x2 y2 ∧ circle_eqn x1 y1 ∧ circle_eqn x2 y2 ∧ chord_length x1 y1 x2 y2 = sqrt 14 := 
sorry

end chord_length_eq_sqrt14_l826_826041


namespace hyperbola_standard_equation_l826_826675

theorem hyperbola_standard_equation :
  (∃ h : ℝ, h = 1/4 ∧
           (λ y x : ℝ, y + 2 * y = 0) =
           (λ y x : ℝ, y / (h * h) - x / (4 * h) → 1) :=
begin
  sorry
end

end hyperbola_standard_equation_l826_826675


namespace main_theorem_l826_826724

-- Define the polynomial with integer coefficients
def P : ℤ[X] := sorry

-- Define the degree of the polynomial P
def deg_P : ℤ := P.degree.to_nat

-- Define the number of integer solutions k such that (P(k))^2 = 1
noncomputable def n_P : ℕ := (finset.filter (λ k : ℤ, (P.eval k)^2 = 1) finset.Icc (-P.degree.to_nat) P.degree.to_nat).card

-- Prove the statement
theorem main_theorem : deg_P ≥ 1 → n_P - deg_P ≤ 2 :=
by
  -- Proof omitted
  sorry

end main_theorem_l826_826724


namespace is_exact_time_now_321_l826_826346

noncomputable def current_time_is_321 : Prop :=
  exists t : ℝ, 0 < t ∧ t < 60 ∧ |(6 * t + 48) - (90 + 0.5 * (t - 4))| = 180

theorem is_exact_time_now_321 : current_time_is_321 := 
  sorry

end is_exact_time_now_321_l826_826346


namespace plane_ABC_passes_through_center_l826_826520

/-- Given a point S on a fixed sphere centered at O, consider the tetrahedron SABC which is inscribed in the sphere centered at O, and the edges starting from S are mutually perpendicular.
Prove that the plane ABC passes through the center O of the sphere. -/
theorem plane_ABC_passes_through_center {S A B C O : Point} (hS : S ∈ sphere O radius) 
  (hA : A ∈ sphere O radius) (hB : B ∈ sphere O radius) (hC : C ∈ sphere O radius) 
  (Hmutual : ∀ (X ∈ {A, B, C}), (S - O) ⊥ (X - O)) :
  ∃ (P : Point), P = O → plane ABC passes through P :=
begin
  sorry
end

end plane_ABC_passes_through_center_l826_826520


namespace smallest_whole_number_larger_than_triangle_perimeter_l826_826496

theorem smallest_whole_number_larger_than_triangle_perimeter :
  (∀ s : ℝ, 16 < s ∧ s < 30 → ∃ n : ℕ, n = 60) :=
by
  sorry

end smallest_whole_number_larger_than_triangle_perimeter_l826_826496


namespace Kanul_cash_percentage_l826_826347

theorem Kanul_cash_percentage :
  let raw_materials := 5000
  let machinery := 200
  let total_amount := 7428.57
  let amount_spent := raw_materials + machinery
  let cash := total_amount - amount_spent
  let percentage_cash := (cash / total_amount) * 100
  percentage_cash ≈ 29.99 := by
  sorry

end Kanul_cash_percentage_l826_826347


namespace fraction_calculation_l826_826209

theorem fraction_calculation : ( ( (1/2 : ℚ) + (1/5) ) / ( (3/7) - (1/14) ) * (2/3) ) = 98/75 :=
by
  sorry

end fraction_calculation_l826_826209


namespace necessary_but_not_sufficient_l826_826629

-- Defining the problem in Lean 4 terms.
noncomputable def geom_seq_cond (a : ℕ → ℕ) (m n p q : ℕ) : Prop :=
  m + n = p + q → a m * a n = a p * a q

theorem necessary_but_not_sufficient (a : ℕ → ℕ) (m n p q : ℕ) (h : m + n = p + q) :
  geom_seq_cond a m n p q → ∃ b : ℕ → ℕ, (∀ n, b n = 0 → (m + n = p + q → b m * b n = b p * b q))
    ∧ (∀ n, ¬ (b n = 0 → ∀ q, b (q+1) / b q = b (q+1) / b q)) := sorry

end necessary_but_not_sufficient_l826_826629


namespace find_f_neg2_l826_826281

-- Conditions
def f (x : ℝ) (a : ℝ) : ℝ :=
  a * Real.sin x + x^3 + 1

variable (a : ℝ)
variable (h : f 2 a = 3)

-- Goal
theorem find_f_neg2 : f (-2) a = -1 :=
by
  sorry

end find_f_neg2_l826_826281


namespace find_p_q_r_s_l826_826355

noncomputable def angles_of_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π

noncomputable def obtuse_angle (B : ℝ) : Prop :=
  B > π / 2 ∧ B < π

theorem find_p_q_r_s (A B C : ℝ)
  (angle_triangle : angles_of_triangle A B C)
  (B_obtuse : obtuse_angle B)
  (eq1 : cos A ^ 2 + cos B ^ 2 + 2 * sin A * sin B * cos C = 9 / 5)
  (eq2 : cos B ^ 2 + cos C ^ 2 + 2 * sin B * sin C * cos A = 19 / 12) :
  ∃ (p q r s : ℕ), Nat.gcd (p + q) s = 1 ∧ ¬ ∃ (n : ℕ), n^2 ∣ r ∧ n > 1 ∧
  cos C ^ 2 + cos A ^ 2 + 2 * sin C * sin A * cos B = (p - q * real.sqrt r) / s ∧
  p + q + r + s = 21 :=
sorry

end find_p_q_r_s_l826_826355


namespace sophia_book_length_l826_826513

variables {P : ℕ}

def total_pages (P : ℕ) : Prop :=
  (2 / 3 : ℝ) * P = (1 / 3 : ℝ) * P + 90

theorem sophia_book_length 
  (h1 : total_pages P) :
  P = 270 :=
sorry

end sophia_book_length_l826_826513


namespace short_haired_girls_l826_826068

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l826_826068


namespace inverse_proportion_inequality_l826_826260

-- Define the given conditions for the points on the inverse proportion function
variable (k : ℝ) (h : k > 0)
def y1 := k / -3 
def y2 := k / -1
def y3 := k / 1

-- Formalize the desired inequality as the theorem to be proved
theorem inverse_proportion_inequality (h : k > 0): y2 < y1 ∧ y1 < y3 :=
by
  unfold y1 y2 y3
  sorry

end inverse_proportion_inequality_l826_826260


namespace max_and_min_values_g_l826_826101

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3)
noncomputable def g (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3) - abs (x - 1)

theorem max_and_min_values_g :
  (∀ x, (2 ≤ x ∧ x ≤ 3) → f x = 1) →
  (∃ a b, (∀ x, (2 ≤ x ∧ x ≤ 3) → a ≤ g x ∧ g x ≤ b) ∧ a = -1 ∧ b = 0) :=
by
  intros H
  use [-1, 0]
  split
  sorry
  sorry

end max_and_min_values_g_l826_826101


namespace sum_of_exponents_in_divisors_product_l826_826746

theorem sum_of_exponents_in_divisors_product :
  let n := 360000
  let prime_factors_n := [(2, 6), (3, 2), (5, 4)]
  let d_n := (6 + 1) * (2 + 1) * (4 + 1)
  let m := n ^ (d_n / 2)
  let prime_factors_m := [(2, 315), (3, 105), (5, 210)]
  (prime_factors_m.map Prod.snd).sum = 630 :=
by
  let n := 360000
  let prime_factors_n := [(2, 6), (3, 2), (5, 4)]
  let d_n := (6 + 1) * (2 + 1) * (4 + 1)
  let m := n ^ (d_n / 2)
  let prime_factors_m := [(2, 315), (3, 105), (5, 210)]
  exact (prime_factors_m.map Prod.snd).sum

end sum_of_exponents_in_divisors_product_l826_826746


namespace directrix_of_parabola_l826_826650

theorem directrix_of_parabola (p m : ℝ) (hp : p > 0)
  (hM_on_parabola : (4, m).fst ^ 2 = 2 * p * (4, m).snd)
  (hM_to_focus : dist (4, m) (p / 2, 0) = 6) :
  -p/2 = -2 :=
sorry

end directrix_of_parabola_l826_826650


namespace minimum_pencil_lifts_l826_826832

theorem minimum_pencil_lifts (G : Type) [graph G] :
  ∃ (k : ℕ), k = 10 →
  ∃ (lifts: ℕ), lifts = 6 :=
by
  -- Definitions and conditions from the problem
  assume h1 : ∃ v : finset G, (∑ x in v, odd (degree x)) = 10,
  have h2 : (∑ x in v, odd (degree x)) = 10,
  -- The resulting minimum number of lifts required
  existsi 10,
  existsi 6,
  sorry

end minimum_pencil_lifts_l826_826832


namespace find_PM_l826_826321

variable (P Q R M : Type) [InnerProductSpace ℝ P] [InnerProductSpace ℝ Q] [InnerProductSpace ℝ R] [InnerProductSpace ℝ M]
variable (d_PQ : dist P Q = 24)
variable (d_QR : dist Q R = 26)
variable (d_PR : dist P R = 32)
variable (mid_M : midpoint ℝ Q R = M)

theorem find_PM : dist P M = 31 := by
  sorry  -- Proof needs to be constructed here but omitted as per instructions.

end find_PM_l826_826321


namespace cyclic_quadrilateral_area_l826_826231

theorem cyclic_quadrilateral_area (A B C D : Type)
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (cyclic : is_cyclic_quadrilateral A B C D)
  (hAB : AB = 2) (hBC : BC = 6) (hCD : CD = 4) (hDA : DA = 4) :
  area_of_quadrilateral A B C D = 8 * real.sqrt 3 :=
sorry

end cyclic_quadrilateral_area_l826_826231


namespace final_account_balance_l826_826773

theorem final_account_balance :
  let initial_balance := 400
  let transactions := [90, 60, 50, 120, 200]
  let service_charges := [0.02, 0.02, 0.02, 0.025, 0.03]
  let reversed_transactions := {1, 3}  -- 2nd and 4th transaction indices
  let refund_indices := {0}  -- 1st transaction index
  let get_service_charge (amount : ℕ) (percentage : ℝ) : ℝ := amount * percentage
  let adjust_for_service_charge (amount : ℕ) (percentage : ℝ) (index : ℕ) :=
    if index ∈ refund_indices then -amount
    else amount + (get_service_charge amount percentage).to_nat
  let transaction_effect (index : ℕ) (amount : ℕ) (percentage : ℝ) :=
    if index ∈ reversed_transactions then 0
    else adjust_for_service_charge amount percentage index
  let net_effect := (List.range 5).sum (λ i, transaction_effect i (transactions.nth i).getOrElse 0 (service_charges.nth i).getOrElse 0.0)
  let final_balance := initial_balance - net_effect
  in final_balance = 53 :=
by
  sorry

end final_account_balance_l826_826773


namespace digit_sum_of_sqrt_l826_826964

def construct_large_number : Nat :=
  let part1 := (10^2018 - 1) / 9 -- sequence of 2018 ones
  let part2 := (5 * (10^4035 - 10^2018)) / 9 -- sequence of 2017 fives
  part1 + part2 + 6

theorem digit_sum_of_sqrt (N : Nat) (h : N = construct_large_number) :
  digitSum (Nat.floor (Real.sqrt N)) = 6055 :=
by sorry

end digit_sum_of_sqrt_l826_826964


namespace find_a_l826_826803

-- Definitions
def parabola (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c
def vertex_property (a b c : ℤ) := 
  ∃ x y, x = 2 ∧ y = 5 ∧ y = parabola a b c x
def point_on_parabola (a b c : ℤ) := 
  ∃ x y, x = 1 ∧ y = 2 ∧ y = parabola a b c x

-- The main statement
theorem find_a {a b c : ℤ} (h_vertex : vertex_property a b c) (h_point : point_on_parabola a b c) : a = -3 :=
by {
  sorry
}

end find_a_l826_826803


namespace train_speed_l826_826895

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l826_826895


namespace probability_product_zero_probability_product_negative_l826_826654

def given_set : List ℤ := [-3, -2, -1, 0, 5, 6, 7]

def num_pairs : ℕ := 21

theorem probability_product_zero :
  (6 : ℚ) / num_pairs = 2 / 7 := sorry

theorem probability_product_negative :
  (9 : ℚ) / num_pairs = 3 / 7 := sorry

end probability_product_zero_probability_product_negative_l826_826654


namespace basic_tacit_understanding_probability_l826_826805

/-- 
Proof of the probability of achieving "basic tacit understanding" when 
parents and children randomly select a number from 1, 2, 3, 4, 5 
-/
theorem basic_tacit_understanding_probability :
  (Finset.card (Finset.filter (λ p : ℕ × ℕ, |p.1 - p.2| = 1) ((Finset.range 5).product (Finset.range 5)))) / 25 = 8 / 25 := 
by 
  sorry

end basic_tacit_understanding_probability_l826_826805


namespace shipping_cost_correct_l826_826754

-- Definitions of given conditions
def total_weight_of_fish : ℕ := 540
def weight_of_each_crate : ℕ := 30
def total_shipping_cost : ℚ := 27

-- Calculating the number of crates
def number_of_crates : ℕ := total_weight_of_fish / weight_of_each_crate

-- Definition of the target shipping cost per crate
def shipping_cost_per_crate : ℚ := total_shipping_cost / number_of_crates

-- Lean statement to prove the given problem
theorem shipping_cost_correct :
  shipping_cost_per_crate = 1.50 := by
  sorry

end shipping_cost_correct_l826_826754


namespace stella_annual_income_l826_826793

-- Define the conditions
def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def total_months : ℕ := 12

-- The question: What is Stella's annual income last year?
def annual_income (monthly_income : ℕ) (worked_months : ℕ) : ℕ :=
  monthly_income * worked_months

-- Prove that Stella's annual income last year was $49190
theorem stella_annual_income : annual_income monthly_income (total_months - unpaid_leave_months) = 49190 :=
by
  sorry

end stella_annual_income_l826_826793


namespace committee_formation_l826_826697

theorem committee_formation :
  let club_size := 15
  let num_roles := 2
  let num_members := 3
  let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
  total_ways = 60060 := by
    let club_size := 15
    let num_roles := 2
    let num_members := 3
    let total_ways := (15 * 14) * Nat.choose (15 - num_roles) num_members
    show total_ways = 60060
    sorry

end committee_formation_l826_826697


namespace a_2_pow_100_eq_l826_826736

def a_sequence : ℕ → ℕ
| 1 := 1
| (2 * n) := n * a_sequence n
| n := 0  -- For undefined cases in the given problem

theorem a_2_pow_100_eq : a_sequence (2^100) = 2^4950 := 
by sorry

end a_2_pow_100_eq_l826_826736


namespace girls_with_short_hair_count_l826_826065

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l826_826065


namespace bin_ball_problem_l826_826861

theorem bin_ball_problem (k : ℕ) (x : ℕ) (h1 : x = 3) 
  (h2 : 7 * x - k * x = 7 + k * 1)
  (h3 : x ≤ 4) : k = 2 := by
  have h4 : k = 2 * (3 / 2).toNat
  sorry

end bin_ball_problem_l826_826861


namespace lines_perpendicular_in_plane_l826_826674

variable {α : Type} [Plane α] (a : Line α)

theorem lines_perpendicular_in_plane (h : ¬Perpendicular a α) : ∃ infinite_set (l ∈ lines_in_plane α), Perpendicular l a :=
sorry

end lines_perpendicular_in_plane_l826_826674


namespace count_final_numbers_l826_826410

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826410


namespace sum_of_opposite_numbers_is_zero_l826_826304

theorem sum_of_opposite_numbers_is_zero {a b : ℝ} (h : a + b = 0) : a + b = 0 := 
h

end sum_of_opposite_numbers_is_zero_l826_826304


namespace circle_radius_l826_826136

theorem circle_radius (P Q : ℝ) (r : ℝ)
  (h1 : P = real.pi * r^2)
  (h2 : Q = 2 * real.pi * r)
  (h3 : P / Q = 10) : r = 20 :=
by
  sorry

end circle_radius_l826_826136


namespace T_10_eq_144_l826_826599

def T : ℕ → ℕ 
| 0       := 0
| 1       := 2 -- "A", "B"
| 2       := 4 -- "AA", "AB", "BA", "BB"
| (n + 1) := let a_n1 := T n
                 a_n2 := T n
                 b_n1 := T n
                 b_n2 := T n
             in a_n1 + a_n2 + b_n1 + b_n2

theorem T_10_eq_144 : T 10 = 144 :=
sorry

end T_10_eq_144_l826_826599


namespace total_cost_l826_826022

/-- Sam initially has s yellow balloons.
He gives away a of these balloons to Fred.
Mary has m yellow balloons.
Each balloon costs c dollars.
Determine the total cost for the remaining balloons that Sam and Mary jointly have.
Given: s = 6.0, a = 5.0, m = 7.0, c = 9.0 dollars.
Expected result: the total cost is 72.0 dollars.
-/
theorem total_cost (s a m c : ℝ) (h_s : s = 6.0) (h_a : a = 5.0) (h_m : m = 7.0) (h_c : c = 9.0) :
  (s - a + m) * c = 72.0 := 
by
  rw [h_s, h_a, h_m, h_c]
  -- At this stage, the proof would involve showing the expression is 72.0, but since no proof is required:
  sorry

end total_cost_l826_826022


namespace elise_savings_l826_826582

theorem elise_savings (e c p l x : ℕ) (h1 : e = 8) (h2 : c = 2) (h3 : p = 18) (h4 : l = 1) (h5 : e + x - (c + p) = l) :
  x = 13 :=
by
  rw [h1, h2, h3, h4] at h5
  linarith [h5]

end elise_savings_l826_826582


namespace polar_coordinates_l826_826256

/-- Define the complex number representing the point P. -/
def complex_number : ℂ := -(2 + 2 * complex.I)

/-- Define the polar coordinates of the point P. -/
theorem polar_coordinates (k : ℤ) : 
  complex.abs complex_number = 2 * sqrt 2 ∧
  complex.arg complex_number = (5 * real.pi / 4 + 2 * k * real.pi) := 
begin
  sorry
end

end polar_coordinates_l826_826256


namespace sin_minus_cos_l826_826607

theorem sin_minus_cos {θ : ℝ} (h1 : sin θ + cos θ = 3 / 4) (ht : 0 < θ ∧ θ < π) : sin θ - cos θ = (sqrt 23) / 4 := 
sorry

end sin_minus_cos_l826_826607


namespace probability_C_l826_826862

-- Definitions of probabilities
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Main proof statement
theorem probability_C :
  ∀ P_C : ℚ, P_A + P_B + P_C + P_D = 1 → P_C = 1 / 4 :=
by
  intro P_C h
  sorry

end probability_C_l826_826862


namespace parabola_height_at_5_inch_l826_826876

noncomputable def parabola_height 
  (a x : ℝ) (vertex_height span : ℝ) : ℝ :=
  a * x^2 + vertex_height

theorem parabola_height_at_5_inch 
  (vertex_height span : ℝ)
  (h1 : vertex_height = 16)
  (h2 : span = 40) :
  parabola_height (-1 / 25) 5 vertex_height 40 = 15 := 
by
  rw [parabola_height, h1]
  norm_num
  have : (-1 / 25) * 25 = -1 := by norm_num
  rw this
  norm_num

end parabola_height_at_5_inch_l826_826876


namespace reena_loan_l826_826115

/-- 
  Problem setup:
  Reena took a loan of $1200 at simple interest for a period equal to the rate of interest years. 
  She paid $192 as interest at the end of the loan period.
  We aim to prove that the rate of interest is 4%. 
-/
theorem reena_loan (P : ℝ) (SI : ℝ) (R : ℝ) (N : ℝ) 
  (hP : P = 1200) 
  (hSI : SI = 192) 
  (hN : N = R) 
  (hSI_formula : SI = P * R * N / 100) : 
  R = 4 := 
by 
  sorry

end reena_loan_l826_826115


namespace not_mysterious_diff_consecutive_odd_l826_826572

/-- A mysterious number is defined as the difference of squares of two consecutive even numbers. --/
def is_mysterious (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2 * k + 2)^2 - (2 * k)^2

/-- The difference of the squares of two consecutive odd numbers. --/
def diff_squares_consecutive_odd (k : ℤ) : ℤ :=
  (2 * k + 1)^2 - (2 * k - 1)^2

/-- Prove that the difference of squares of two consecutive odd numbers is not a mysterious number. --/
theorem not_mysterious_diff_consecutive_odd (k : ℤ) : ¬ is_mysterious (Int.natAbs (diff_squares_consecutive_odd k)) :=
by
  sorry

end not_mysterious_diff_consecutive_odd_l826_826572


namespace distinct_non_neg_numbers_condition_l826_826762

theorem distinct_non_neg_numbers_condition (N : ℕ) (h1 : N ≥ 9)
    (h2 : ∀ S : Finset ℝ, (S.card = 8) → (∃ t ∈ Finset.range N, t ∉ S ∧ (∃ sum : ℝ, (sum = S.sum id + t) ∧ (sum ∈ ℤ)))) 
    : N = 9 :=
sorry

end distinct_non_neg_numbers_condition_l826_826762


namespace sum_of_center_coordinates_l826_826217

theorem sum_of_center_coordinates (x y : ℝ) :
    (x^2 + y^2 - 6*x + 8*y = 18) → (x = 3) → (y = -4) → x + y = -1 := 
by
    intro h1 hx hy
    rw [hx, hy]
    norm_num

end sum_of_center_coordinates_l826_826217


namespace frog_max_jump_path_length_l826_826766

theorem frog_max_jump_path_length : 
  ∀ (mark_points : Finset ℕ),
  (∀ n, n ∈ mark_points ↔ 1 ≤ n ∧ n ≤ 2006) →
  (∀ frog_jumps : list ℕ, 
   frog_jumps.head = 1 →
   frog_jumps.last = 1 →
   frog_jumps.length = 2006 + 1 →
   (frog_jumps.nodup ∧ frog_jumps.all (λ x, x ∈ mark_points)) →
  ∑ i in (finRange frog_jumps.length).tail, abs (frog_jumps.nth_le i sorry - frog_jumps.nth_le (i - 1) sorry) <= 2012018) :=
sorry

end frog_max_jump_path_length_l826_826766


namespace matrix_rank_equal_cardinality_l826_826354

variable (n : ℕ) (a : Fin n → ℝ)

noncomputable def A (n : ℕ) (a : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j, max (a i) (a j)

theorem matrix_rank_equal_cardinality (h1 : 2 ≤ n) (h2 : ∀ i, a i ≠ 0) :
  Matrix.rank (A n a) = (Finset.univ.image a).card :=
sorry

end matrix_rank_equal_cardinality_l826_826354


namespace line_equation_l826_826632

-- Define the focus of the hyperbola
def focus : { x : ℝ × ℝ // x = (sqrt 5, 0) ∨ x = (-sqrt 5, 0) } :=
  ⟨(sqrt 5, 0), or.inl rfl⟩

-- Define the asymptotes of the hyperbola
def is_asymptote (m : ℝ) : Prop :=
  m = 1 / 2 ∨ m = -1 / 2

-- Define the line l passing through a given point with a given slope
structure Line (m b : ℝ) :=
  (passes_through_focus : ∃ x, (x, m * x + b) = focus.val)
  (slope_is_asymptote_parallel : is_asymptote m)

theorem line_equation :
  ∃ b, Line (-1/2) b ∧ b = sqrt 5 / 2 :=
by
  -- We can state the desired line equation with slope -1/2 and intercept sqrt 5 / 2
  use sqrt 5 / 2
  split
  -- Show that the line passes through the focus
  { existsi sqrt 5
    simp [focus, Line, is_asymptote]
    use or.inl rfl },
  -- Show that the intercept is sqrt 5 / 2
  { rfl }

end line_equation_l826_826632


namespace travis_discount_percentage_l826_826483

theorem travis_discount_percentage (P D : ℕ) (hP : P = 2000) (hD : D = 1400) :
  ((P - D) / P * 100) = 30 := by
  -- sorry to skip the proof
  sorry

end travis_discount_percentage_l826_826483


namespace people_on_last_boat_l826_826551

variable (x : ℕ)

def total_people := 8 * x + 6

theorem people_on_last_boat :
  total_people x - 12 * (x - 2) = -4 * x + 30 :=
by sorry

end people_on_last_boat_l826_826551


namespace intersection_A_B_l826_826990

-- Define the conditions of set A and B using the given inequalities and constraints
def set_A : Set ℤ := {x | -2 < x ∧ x < 3}
def set_B : Set ℤ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the proof problem translating conditions and question to Lean
theorem intersection_A_B : (set_A ∩ set_B) = {0, 1, 2} := by
  sorry

end intersection_A_B_l826_826990


namespace sin_230_plus_sin_260_l826_826121

theorem sin_230_plus_sin_260 : sin (230 : ℝ) + sin (260 : ℝ) = 1 := by
  -- Identifying the conditions in the problem
  have h1 : ∀ x : ℝ, sin (180 + x) = -sin x := sorry
  have h2 : ∀ x : ℝ, sin (360 - x) = -sin x := sorry
  
  -- Prove the main statement using the above conditions
  sorry

end sin_230_plus_sin_260_l826_826121


namespace prove_a_minus_b_plus_c_eq_3_l826_826365

variable {a b c m n : ℝ}

theorem prove_a_minus_b_plus_c_eq_3 
    (h : ∀ x : ℝ, m * x^2 - n * x + 3 = a * (x - 1)^2 + b * (x - 1) + c) :
    a - b + c = 3 :=
sorry

end prove_a_minus_b_plus_c_eq_3_l826_826365


namespace expected_value_eq_35_9_l826_826435

noncomputable def probability (i : ℕ) : ℚ :=
  if i = 8 then 1/3
  else if i = 2 ∨ i = 4 ∨ i = 6 then 1/6
  else if i = 1 ∨ i = 3 ∨ i = 5 ∨ i = 7 then 1/12
  else 0

noncomputable def expected_value : ℚ :=
  ∑ i in [1, 2, 3, 4, 5, 6, 7, 8], i * probability i

theorem expected_value_eq_35_9 : expected_value = 35 / 9 :=
by
  sorry

end expected_value_eq_35_9_l826_826435


namespace proof_of_inequality_proof_of_coprime_l826_826870

-- Define the probability that ab + 2c >= abc
def probabilityOfInequality : ℚ :=
  -- Total outcomes: 6^3 = 216
  -- Favorable outcomes: 58
  29 / 108

-- Define the probability that ab + 2c and 2abc are coprime
def probabilityOfCoprime : ℚ :=
  -- Total outcomes: 216
  -- Favorable outcomes: 39
  13 / 72

-- Prove the probability of the inequality
theorem proof_of_inequality :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6) →
  P(ab + 2c ≥ abc) = probabilityOfInequality :=
begin
  sorry
end

-- Prove the probability of being coprime
theorem proof_of_coprime :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6) →
  P(gcd(ab + 2c, 2abc) = 1) = probabilityOfCoprime :=
begin
  sorry
end

end proof_of_inequality_proof_of_coprime_l826_826870


namespace oil_less_than_15_after_12_days_l826_826142

def fraction_remaining : ℕ → ℚ
| 1     := 4/5
| (n + 1) := (n + 4 + 1) / (n + 4 + 5)

noncomputable def remaining_oil : ℕ → ℚ
| 0     := 100
| (n + 1) := remaining_oil n * fraction_remaining (n + 1)

theorem oil_less_than_15_after_12_days : 
  remaining_oil 12 < 15 := 
sorry

end oil_less_than_15_after_12_days_l826_826142


namespace find_a_9_l826_826338

variable {a : ℕ → ℤ}

-- Given conditions
def a_2_a_5_eq_neg32 := (a 2) * (a 5) = -32
def a_3_a_4_sum := (a 3) + (a 4) = 4
def common_ratio_is_integer := ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

-- To prove that a₉ = -256
theorem find_a_9
  (h1 : a_2_a_5_eq_neg32)
  (h2 : a_3_a_4_sum)
  (h3 : common_ratio_is_integer) : 
  a 9 = -256 :=
  sorry

end find_a_9_l826_826338


namespace sum_first_n_terms_l826_826968

noncomputable def a_seq : ℕ → ℝ
| 0     := 1
| 1     := 1/2
| (n+2) := real.sqrt (a_seq n * a_seq (n+1))

def S_n (n : ℕ) : ℝ := 2 * (1 - (1/2) ^ n)

theorem sum_first_n_terms (n : ℕ) :
    ∑ i in finset.range n, a_seq i = S_n n :=
sorry

end sum_first_n_terms_l826_826968


namespace max_mn_l826_826452

theorem max_mn (m n : ℝ) (h : m + n = 1) : mn ≤ 1 / 4 :=
by
  sorry

end max_mn_l826_826452


namespace other_odd_number_value_l826_826072

theorem other_odd_number_value :
  ∀ (a e x : ℕ), 
  e = 79 →
  a = e - 8 →
  a + x = 146 →
  x = 75 :=
by
  intros a e x he ha hex
  rw [ha, he] at hex
  exact Eq.trans (by norm_num : 71 + x = 146) hex.symm
  sorry -- proof omitted

end other_odd_number_value_l826_826072


namespace determine_value_of_e_l826_826353

theorem determine_value_of_e {a b c d e : ℝ} (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) 
    (h5 : a + b = 32) (h6 : a + c = 36) (h7 : b + c = 37 ∨ a + d = 37) 
    (h8 : c + e = 48) (h9 : d + e = 51) : e = 27.5 :=
sorry

end determine_value_of_e_l826_826353


namespace distinct_tile_arrangements_l826_826689

open Finset

/-- The number of distinct arrangements of six tiles, where four are painted blue, one is painted red, 
and one is painted green, considering arrangements identical if they can be reflected along the midpoint, is 15. -/
theorem distinct_tile_arrangements : 
  let all_distinct_arrangements :=
    univ.powerset.filter (λ s : Finset ℕ, s.card = 6 ∧ 
      s.filter (λ x, x = 1) ∈ univ.powerset.filter (λ t, t.card = 1) ∧
      s.filter (λ x, x = 2) ∈ univ.powerset.filter (λ t, t.card = 1) ∧
      s.filter (λ x, x = 4) ∈ univ.powerset.filter (λ t, t.card = 4)) in
  let distinct_arrangements := univ.powerset.filter (λ s : Finset ℕ, s.card = 3 ∧ 
      s.filter (λ x, x = 1) ∈ univ.powerset.filter (λ t, t.card = 1) ∧
      s.filter (λ x, x = 2) ∈ univ.powerset.filter (λ t, t.card = 1)) in
  distinct_arrangements.card = 15 := by
  sorry

end distinct_tile_arrangements_l826_826689


namespace find_m_n_sum_l826_826087

noncomputable def point (x : ℝ) (y : ℝ) := (x, y)

def center_line (P : ℝ × ℝ) : Prop := P.1 - P.2 - 2 = 0

def on_circle (C : ℝ × ℝ) (P : ℝ × ℝ) (r : ℝ) : Prop := 
  (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2

def circles_intersect (A B C D : ℝ × ℝ) (r₁ r₂ : ℝ) : Prop :=
  on_circle A C r₁ ∧ on_circle A D r₂ ∧ on_circle B C r₁ ∧ on_circle B D r₂

theorem find_m_n_sum 
  (A : ℝ × ℝ) (m n : ℝ)
  (C D : ℝ × ℝ)
  (r₁ r₂ : ℝ)
  (H1 : A = point 1 3)
  (H2 : circles_intersect A (point m n) C D r₁ r₂)
  (H3 : center_line C ∧ center_line D) :
  m + n = 4 :=
sorry

end find_m_n_sum_l826_826087


namespace cordelia_bleaching_l826_826932

noncomputable def bleaching_time (B : ℝ) : Prop :=
  B + 4 * B + B / 3 = 10

theorem cordelia_bleaching : ∃ B : ℝ, bleaching_time B ∧ B = 1.875 :=
by {
  sorry
}

end cordelia_bleaching_l826_826932


namespace remaining_numbers_after_erasure_l826_826392

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826392


namespace farmer_initial_days_l826_826141

theorem farmer_initial_days 
  (x : ℕ) 
  (plan_daily : ℕ) 
  (actual_daily : ℕ) 
  (extra_days : ℕ) 
  (left_area : ℕ) 
  (total_area : ℕ)
  (h1 : plan_daily = 120) 
  (h2 : actual_daily = 85) 
  (h3 : extra_days = 2) 
  (h4 : left_area = 40) 
  (h5 : total_area = 720): 
  85 * (x + extra_days) + left_area = total_area → x = 6 :=
by
  intros h
  sorry

end farmer_initial_days_l826_826141


namespace gcd_of_products_l826_826310

def gcd (a b : Nat) : Nat := Nat.gcd a b

theorem gcd_of_products :
   gcd (gcd 20 16) (gcd 18 24) = 2 := 
by
    sorry

end gcd_of_products_l826_826310


namespace unique_solution_conditions_l826_826588

-- Definitions based on the conditions
variables {x y a : ℝ}

def inequality_condition (x y a : ℝ) : Prop := 
  x^2 + y^2 + 2 * x ≤ 1

def equation_condition (x y a : ℝ) : Prop := 
  x - y = -a

-- Main Theorem Statement
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, inequality_condition x y a ∧ equation_condition x y a) ↔ (a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2) :=
sorry

end unique_solution_conditions_l826_826588


namespace mailman_total_pieces_l826_826873

def piecesOfMailFirstHouse := 6 + 5 + 3 + 4 + 2
def piecesOfMailSecondHouse := 4 + 7 + 2 + 5 + 3
def piecesOfMailThirdHouse := 8 + 3 + 4 + 6 + 1

def totalPiecesOfMail := piecesOfMailFirstHouse + piecesOfMailSecondHouse + piecesOfMailThirdHouse

theorem mailman_total_pieces : totalPiecesOfMail = 63 := by
  sorry

end mailman_total_pieces_l826_826873


namespace horse_tile_system_l826_826702

theorem horse_tile_system (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + (1 / 3 : ℚ) * y = 100) : 
  ∃ (x y : ℕ), (x + y = 100) ∧ (3 * x + (1 / 3 : ℚ) * y = 100) :=
by sorry

end horse_tile_system_l826_826702


namespace rectangle_EF_length_l826_826852

theorem rectangle_EF_length
  (ABCD : Type)
  (AB AD p : ℕ)
  (AB_eq : AB = 3 * p + 4)
  (AD_eq : AD = 2 * p + 6)
  (p_eq : p = 12)
  : (let AB := 3 * p + 4 in
     let AD := 2 * p + 6 in
     let BD := Real.sqrt (AB^2 + AD^2) in
     let cos_theta := AD / BD in
     let DE := AD * cos_theta in
     let BF := AD * cos_theta in
     let EF := BD - DE - BF
     in EF = 14) :=
by
  sorry

end rectangle_EF_length_l826_826852


namespace track_meet_girls_short_hair_l826_826070

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l826_826070


namespace sum_of_integer_solutions_l826_826212

theorem sum_of_integer_solutions :
  ∀ x : ℤ, (x^4 - 13 * x^2 + 36 = 0) → (∃ a b c d, x = a + b + c + d ∧ a + b + c + d = 0) :=
begin
  sorry
end

end sum_of_integer_solutions_l826_826212


namespace line_angle_l826_826812

theorem line_angle (x y : ℝ) (h : sqrt 3 * x + 3 * y - 1 = 0) : angle_with_x_axis (Line (sqrt 3) 3 (-1)) = 150 := 
sorry

end line_angle_l826_826812


namespace common_ratio_is_2_l826_826262

noncomputable def common_ratio_of_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n+1) = a n * q) ∧ (∀ m n, m < n → a m < a n)

theorem common_ratio_is_2
  (a : ℕ → ℝ) (q : ℝ)
  (hgeo : common_ratio_of_increasing_geometric_sequence a q)
  (h1 : a 1 + a 5 = 17)
  (h2 : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end common_ratio_is_2_l826_826262


namespace ratio_of_avg_speeds_l826_826845

-- Defining the conditions
def travel_time_eddy : ℝ := 3 -- Eddy's travel time in hours
def travel_time_freddy : ℝ := 3 -- Freddy's travel time in hours
def distance_a_b : ℝ := 600 -- Distance from city A to city B in km
def distance_a_c : ℝ := 300 -- Distance from city A to city C in km

-- Defining average speeds
def avg_speed_eddy : ℝ := distance_a_b / travel_time_eddy
def avg_speed_freddy : ℝ := distance_a_c / travel_time_freddy

-- Proving the ratio of their average speeds
theorem ratio_of_avg_speeds : avg_speed_eddy / avg_speed_freddy = 2 :=
by {
  have h1 : avg_speed_eddy = 200 := by { simp [avg_speed_eddy, distance_a_b, travel_time_eddy] },
  have h2 : avg_speed_freddy = 100 := by { simp [avg_speed_freddy, distance_a_c, travel_time_freddy] },
  simp [h1, h2],
  exact div_eq_of_eq_mul_right (by norm_num) (by norm_num)
}

end ratio_of_avg_speeds_l826_826845


namespace max_and_min_values_g_l826_826103

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3)
noncomputable def g (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3) - abs (x - 1)

theorem max_and_min_values_g :
  (∀ x, (2 ≤ x ∧ x ≤ 3) → f x = 1) →
  (∃ a b, (∀ x, (2 ≤ x ∧ x ≤ 3) → a ≤ g x ∧ g x ≤ b) ∧ a = -1 ∧ b = 0) :=
by
  intros H
  use [-1, 0]
  split
  sorry
  sorry

end max_and_min_values_g_l826_826103


namespace tangent_line_sum_l826_826264

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end tangent_line_sum_l826_826264


namespace GCF_LCM_proof_l826_826743

-- Define GCF (greatest common factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (least common multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_proof :
  GCF (LCM 9 21) (LCM 14 15) = 21 :=
by
  sorry

end GCF_LCM_proof_l826_826743


namespace darrel_rate_l826_826952

noncomputable def steven_rate : ℕ := 75
noncomputable def total_compost : ℕ := 2550
noncomputable def total_time : ℕ := 30 

theorem darrel_rate :
  ∃ D : ℕ, (steven_rate + D) * total_time = total_compost ∧ D = 10 :=
by {
  use 10,
  split,
  { rw [add_comm, add_mul, mul_comm _ 30, mul_comm _ 30, mul_comm, nat.cast_add, nat.cast_mul],
    norm_num,
  },
  { refl }
}

end darrel_rate_l826_826952


namespace num_combinations_of_531_l826_826721

theorem num_combinations_of_531 : ∃ (n : ℕ), n = 3! ∧ n = 6 :=
by
  use 3!
  split
  case _1 => rfl
  case _2 => sorry

end num_combinations_of_531_l826_826721


namespace total_cost_is_correct_l826_826082

noncomputable def total_cost_bricks := 
  let cost_brick_30 := 0.3 * 1000 * (0.50 - 0.50 * 0.50)
  let cost_brick_40 := 0.4 * 1000 * (0.50 - 0.50 * 0.20)
  let cost_brick_30_full := 0.3 * 1000 * 0.50
  let total_bricks_cost := cost_brick_30 + cost_brick_40 + cost_brick_30_full
  total_bricks_cost * 1.05 -- tax included

noncomputable def total_cost_materials := 200 * 1.07 -- tax included

def labor_fees := 20 * 10

def total_cost_shed := total_cost_bricks + total_cost_materials + labor_fees

theorem total_cost_is_correct : total_cost_shed = 818.25 := by
  calc 
    total_cost_bricks = 404.25 := sorry
    total_cost_materials = 214 := sorry
    labor_fees = 200 := rfl
    total_cost_shed = 404.25 + 214 + 200 := rfl
    _ = 818.25 := sorry

end total_cost_is_correct_l826_826082


namespace cyclic_quadrilateral_area_l826_826233

theorem cyclic_quadrilateral_area (A B C D : Type)
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (cyclic : is_cyclic_quadrilateral A B C D)
  (hAB : AB = 2) (hBC : BC = 6) (hCD : CD = 4) (hDA : DA = 4) :
  area_of_quadrilateral A B C D = 8 * real.sqrt 3 :=
sorry

end cyclic_quadrilateral_area_l826_826233


namespace probability_P_closer_to_A_l826_826341

def is_triangle (A B C : Type) := 
  (dist A B ≠ 0) ∧ (dist B C ≠ 0) ∧ (dist A C ≠ 0)

def is_isosceles_triangle (A B C : Type) :=
  is_triangle A B C ∧ (dist A B = dist A C)

noncomputable def is_locus_P_in_circle (A B C P : Type) :=
  dist P B = 2 * dist P C ∧ P ∈ circle (B, 10 / 3)

theorem probability_P_closer_to_A 
  (A B C P : Type) 
  (h1 : is_isosceles_triangle A B C) 
  (h2 : dist A B = 7 ∧ dist A C = 7)
  (h3 : dist B C = 10) 
  (h4 : is_locus_P_in_circle A B C P) 
  : probability (dist P A < dist P B ∧ dist P A < dist P C) = 1 := sorry

end probability_P_closer_to_A_l826_826341


namespace smallest_b_for_factorization_l826_826592

theorem smallest_b_for_factorization :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 1764 → r + s = b) → b = 84 :=
begin
  sorry
end

end smallest_b_for_factorization_l826_826592


namespace greatest_perfect_square_composed_of_distinct_prime_factors_l826_826759

theorem greatest_perfect_square_composed_of_distinct_prime_factors :
  ∃ n : ℕ, n < 200 ∧ (∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ n = (p^2) * (q^2)) ∧ 
            (∃ k : ℕ, n = k^2) ∧ (card (divisors n) % 2 = 1) ∧ 
            (∀ m : ℕ, m < 200 → (∃ p q : ℕ, prime p ∧ prime q ∧ p ≠ q ∧ m = (p^2) * (q^2)) → 
                       (∃ k : ℕ, m = k^2) → (card (divisors m) % 2 = 1) → m ≤ n) :=
sorry

end greatest_perfect_square_composed_of_distinct_prime_factors_l826_826759


namespace simplify_expression_l826_826437

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) : 
  (x^2 - x) / (x^2 - 2 * x + 1) = 2 + Real.sqrt 2 :=
by
  sorry

end simplify_expression_l826_826437


namespace sqrt_expression_value_l826_826194

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem sqrt_expression_value :
  sqrt ((3 - cos_squared (π / 9)) * (3 - cos_squared (2 * π / 9)) * (3 - cos_squared (4 * π / 9))) = 3 * sqrt 2 :=
by
  sorry

end sqrt_expression_value_l826_826194


namespace width_of_wall_l826_826529

def volume_of_brick (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_wall (length width height : ℝ) : ℝ :=
  length * width * height

theorem width_of_wall
  (l_b w_b h_b : ℝ) (n : ℝ) (L H : ℝ)
  (volume_brick := volume_of_brick l_b w_b h_b)
  (total_volume_bricks := n * volume_brick) :
  volume_of_wall L (total_volume_bricks / (L * H)) H = total_volume_bricks :=
by
  sorry

end width_of_wall_l826_826529


namespace remaining_numbers_count_l826_826419

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826419


namespace sum_primes_50_to_70_l826_826497

open Nat

def isPrime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem sum_primes_50_to_70 : (∑ n in (Finset.filter isPrime (Finset.range (70 + 1))), n) = 240 := by
  sorry

end sum_primes_50_to_70_l826_826497


namespace face_opposite_to_A_l826_826571

-- Define the faces of the cube
inductive Face
| A | B | C | D | E | F
deriving DecidableEq

open Face

-- Define conditions as Lean definitions
def onTop (f : Face) : Prop := f = F
def adjacent (d a b c e : Face) : Prop :=
  d = D ∧ (a = A ∧ b = B ∧ c = C) ∧ ¬ (d = e)

-- Define the question as a theorem
theorem face_opposite_to_A (f d a b c e : Face) :
  onTop f ∧ adjacent d a b c e → e = E :=
by
  intro h
  cases h with hf hadj
  simp [onTop, adjacent] at hf hadj
  subst hf hadj.left
  cases hadj.right with hadj_abc hadj_e
  rw [hadj_abc.left, hadj_abc.right.left, hadj_abc.right.right]
  simp only [eq_self_iff_true, and_true]
  exact hadj_e
  sorry

end face_opposite_to_A_l826_826571


namespace exists_fraction_expression_l826_826299

theorem exists_fraction_expression (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) :
  ∃ (m : ℕ) (h₀ : 3 ≤ m) (h₁ : m ≤ p - 2) (x y : ℕ), (m : ℚ) / (p^2 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ) :=
sorry

end exists_fraction_expression_l826_826299


namespace coefficient_x2_is_negative_40_l826_826255

noncomputable def x2_coefficient_in_expansion (a : ℕ) : ℤ :=
  (-1)^3 * a^2 * Nat.choose 5 2

theorem coefficient_x2_is_negative_40 :
  x2_coefficient_in_expansion 2 = -40 :=
by
  sorry

end coefficient_x2_is_negative_40_l826_826255


namespace find_constants_l826_826450

theorem find_constants (a b c d : ℕ) (h_eq : 
  (∀ (x : ℝ), 
    1/x + 1/(x+2) - 1/(x+4) - 1/(x+6) - 1/(x+8) - 1/(x+10) + 1/(x+12) + 1/(x+14) = 0) 
  ∧ ∃ (x : ℝ),
    x = -a + real.sqrt (b + c * real.sqrt d) 
    ∨ x = -a - real.sqrt (b + c * real.sqrt d)
    ∨ x = -a + real.sqrt (b - c * real.sqrt d)
    ∨ x = -a - real.sqrt (b - c * real.sqrt d))
  (h_pos_int: (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d)) 
  (h_square_prime : ∀ p : ℕ, nat.prime p -> ¬ (p * p) ∣ d) : 
  a + b + c + d = 37  :=
sorry

end find_constants_l826_826450


namespace largest_median_of_set_l826_826090

theorem largest_median_of_set (x : ℤ) : 
  ∃ m, m = 7 ∧ ∀ S, S = {4, 6, 7, x, 2 * x} →
  (∃ l1 l2 l3 r1 r2, l1 ≤ l2 ≤ l3 ≤ r1 ≤ r2 ∧ l3 = 7 ∧ S = {l1, l2, l3, r1, r2}) :=
sorry

end largest_median_of_set_l826_826090


namespace triangle_angles_correct_l826_826343

open Real

noncomputable def angle_triple (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 2 * b * cos C ∧ 
    sin A * sin (B / 2 + C) = sin C * (sin (B / 2) + sin A)

theorem triangle_angles_correct (A B C : ℝ) (h : angle_triple A B C) :
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := 
sorry

end triangle_angles_correct_l826_826343


namespace equal_distances_l826_826727

def Point := ℝ × ℝ × ℝ

def dist (p1 p2 : Point) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 + (z1 - z2) ^ 2

def A : Point := (-8, 0, 0)
def B : Point := (0, 4, 0)
def C : Point := (0, 0, -6)
def D : Point := (0, 0, 0)
def P : Point := (-4, 2, -3)

theorem equal_distances : dist P A = dist P B ∧ dist P B = dist P C ∧ dist P C = dist P D :=
by
  sorry

end equal_distances_l826_826727


namespace max_accepted_applicants_l826_826032

theorem max_accepted_applicants :
  let avg1 := 10
  let sd1 := 8
  let range1 := set.Icc (avg1 - sd1) (avg1 + sd1)
  let avg2 := 20
  let sd2 := 5
  let range2 := set.Icc (avg2 - sd2) (avg2 + sd2)
  let overlap := range1 ∩ range2
  let acceptable_ages := {a : ℤ | a ∈ overlap}
  set.size acceptable_ages = 4 :=
by {
  let avg1 := 10
  let sd1 := 8
  let range1 := set.Icc (avg1 - sd1) (avg1 + sd1)
  let avg2 := 20
  let sd2 := 5
  let range2 := set.Icc (avg2 - sd2) (avg2 + sd2)
  let overlap := range1 ∩ range2
  let acceptable_ages := {a : ℤ | a ∈ overlap}
  sorry
}

end max_accepted_applicants_l826_826032


namespace problem1_problem2_problem3_problem4_l826_826564

theorem problem1 : (-1)^2023 - Real.sqrt (3^2) + (1/2)^(-2) - (3 - Real.pi)^0 = -1 := 
by 
  -- Proof steps would go here
  sorry

theorem problem2 (a : ℝ) : 8 * a^6 / (2 * a^2) + (3 * a^2)^2 - a * a^3 = 12 * a^4 := 
by 
  -- Proof steps would go here
  sorry

theorem problem3 (a b : ℝ) : (3 * a + b) * (a - b) + b * (4 * a - b) = 3 * a^2 + 2 * a * b - 2 * b^2 := 
by 
  -- Proof steps would go here
  sorry

theorem problem4 (x y : ℝ) : (x + y - 2) * (x - y - 2) = x^2 - 4 * x + 4 - y^2 := 
by 
  -- Proof steps would go here
  sorry

end problem1_problem2_problem3_problem4_l826_826564


namespace verify_probabilities_l826_826130

/-- A bag contains 2 red balls, 3 black balls, and 4 white balls, all of the same size.
    A ball is drawn from the bag at a time, and once drawn, it is not replaced. -/
def total_balls := 9
def red_balls := 2
def black_balls := 3
def white_balls := 4

/-- Calculate the probability that the first ball is black and the second ball is white. -/
def prob_first_black_second_white :=
  (black_balls / total_balls) * (white_balls / (total_balls - 1))

/-- Calculate the probability that the number of draws does not exceed 3, 
    given that drawing a red ball means stopping. -/
def prob_draws_not_exceed_3 :=
  (red_balls / total_balls) +
  ((total_balls - red_balls) / total_balls) * (red_balls / (total_balls - 1)) +
  ((total_balls - red_balls - 1) / total_balls) *
  ((total_balls - red_balls) / (total_balls - 1)) *
  (red_balls / (total_balls - 2))

/-- Theorem that verifies the probabilities based on the given conditions. -/
theorem verify_probabilities :
  prob_first_black_second_white = 1 / 6 ∧
  prob_draws_not_exceed_3 = 7 / 12 :=
by
  sorry

end verify_probabilities_l826_826130


namespace log_equation_solution_l826_826109

theorem log_equation_solution (x : ℝ) (hx_pos : 0 < x) : 
  (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ↔ x ≠ 1 :=
by
  sorry

end log_equation_solution_l826_826109


namespace line_equation_through_point_and_area_l826_826146

theorem line_equation_through_point_and_area (b S x y : ℝ) 
  (h1 : ∀ y, (x, y) = (-2*b, 0) → True) 
  (h2 : ∀ p1 p2 p3 : ℝ × ℝ, p1 = (-2*b, 0) → p2 = (0, 0) → 
        ∃ k, p3 = (0, k) ∧ S = 1/2 * (2*b) * k) : 2*S*x - b^2*y + 4*b*S = 0 :=
sorry

end line_equation_through_point_and_area_l826_826146


namespace total_worth_all_crayons_l826_826378

def cost_of_crayons (packs: ℕ) (cost_per_pack: ℝ) : ℝ := packs * cost_per_pack

def discounted_cost (cost: ℝ) (discount_rate: ℝ) : ℝ := cost * (1 - discount_rate)

def tax_amount (cost: ℝ) (tax_rate: ℝ) : ℝ := cost * tax_rate

theorem total_worth_all_crayons : 
  let cost_per_pack := 2.5
  let discount_rate := 0.15
  let tax_rate := 0.07
  let packs_already_have := 4
  let packs_to_buy := 2
  let cost_two_packs := cost_of_crayons packs_to_buy cost_per_pack
  let discounted_two_packs := discounted_cost cost_two_packs discount_rate
  let tax_two_packs := tax_amount cost_two_packs tax_rate
  let total_cost_two_packs := discounted_two_packs + tax_two_packs
  let cost_four_packs := cost_of_crayons packs_already_have cost_per_pack
  cost_four_packs + total_cost_two_packs = 14.60 := 
by 
  sorry

end total_worth_all_crayons_l826_826378


namespace average_abc_l826_826475

theorem average_abc (A B C : ℚ) 
  (h1 : 2002 * C - 3003 * A = 6006) 
  (h2 : 2002 * B + 4004 * A = 8008) 
  (h3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := 
sorry

end average_abc_l826_826475


namespace calculate_expression_l826_826182

noncomputable def fraction_exponentiation : ℝ := (1 / 3) ^ (-2)
noncomputable def sine_term : ℝ := 2 * Real.sin (Real.pi / 3)
noncomputable def absolute_term : ℝ := |2 - Real.sqrt 3|

theorem calculate_expression :
  fraction_exponentiation + sine_term - absolute_term = 7 + 2 * Real.sqrt 3 := by
  sorry

end calculate_expression_l826_826182


namespace minimum_dot_product_PA_PB_l826_826622

theorem minimum_dot_product_PA_PB {P A B : Type*} 
(hC : ∀ (x y : ℝ), x^2 + y^2 = 2) 
(hP : ∃ (x y : ℝ), x - y + 2 * real.sqrt 2 = 0) 
(hIntersection : ∃ (A B : ℝ × ℝ), ¬(A = P) ∧ ¬(B = P) ∧ ∃ (a b : ℝ), a * A.1 + b * A.2 = 2 ∧ a * B.1 + b * B.2 = 2) : 
∃ (PA PB : ℝ), PA * PB = 2 := 
sorry

end minimum_dot_product_PA_PB_l826_826622


namespace friend_owns_10_bicycles_l826_826682

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l826_826682


namespace quadratic_discriminant_real_roots_k_range_for_positive_root_l826_826273

theorem quadratic_discriminant_real_roots (k : ℝ) :
  let Δ := (k + 3)^2 - 4 * (2 * k + 2) in Δ ≥ 0 :=
by
  let Δ := (k + 3)^2 - 4 * (2 * k + 2)
  have : Δ = (k - 1)^2, sorry
  rw this,
  exact pow_two_nonneg _

theorem k_range_for_positive_root (k : ℝ) (x : ℝ) :
  x^2 - (k+3)*x + 2*k + 2 = 0 ∧ 0 < x ∧ x < 1 → -1 < k ∧ k < 0 :=
by
  intro h
  sorry

end quadratic_discriminant_real_roots_k_range_for_positive_root_l826_826273


namespace remaining_numbers_count_l826_826420

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826420


namespace part_I_part_II_part_III_l826_826285

-- Define the set R_n and distance function
def R_n (n : ℕ) : set (fin n → bool) :=
  { X | ∀ i : fin n, X i = 0 ∨ X i = 1 }

def d {n : ℕ} (A B : fin n → bool) : ℕ :=
  finset.univ.sum (λ i, int.nat_abs (A i - B i))

-- (I) Prove that the maximum distance \(d(A,B)\) between any two elements in \(R_2\) is 2
theorem part_I : ∀ A B ∈ R_n 2, d A B ≤ 2 := 
sorry

-- (II) If \( M \subseteq R_3 \) and the distance between any two elements in \( M \) is 2,
-- prove that the maximum number of elements in \( M \) is 4
theorem part_II : ∀ M : set (fin 3 → bool), (∀ A B ∈ M, d A B = 2) → M.finite ∧ M.card ≤ 4 :=
  sorry

-- (III) For \( P \subseteq R_n \) containing \( m \) elements,
-- prove that \( \overset{.}{d}(P) \leq \frac{mn}{2(m-1)} \)
theorem part_III {n m : ℕ} (P : finset (fin n → bool)) (hP : P.card = m) (hm : 2 ≤ m) :
  let avg_d := (finset.card (P.pair_combinations) : ℕ)⁻¹ * 
    (finset.sum (P.pair_combinations) (λ (AB : (fin n → bool) × (fin n → bool)), d AB.1 AB.2))
  in avg_d ≤ (mn / (2 * (m - 1))) :=
sorry

end part_I_part_II_part_III_l826_826285


namespace c_symmetry_l826_826598

def c : ℕ → ℕ → ℕ
| n, 0 => 1
| n, k => if k = n then 1 else 2^k * c n k + c n (k - 1)

theorem c_symmetry (n k : ℕ) (h : n ≥ k) : c n k = c n (n - k) :=
sorry

end c_symmetry_l826_826598


namespace tangent_segment_annulus_area_l826_826615

theorem tangent_segment_annulus_area :
  let radius := 3
  let segment_length := 6
  let inner_radius := 3
  let outer_radius := 3 * Real.sqrt 2
  (π * outer_radius^2 - π * inner_radius^2 = 9 * π) :=

by
  let radius := 3
  let segment_length := 6
  let inner_radius := 3
  let outer_radius := 3 * Real.sqrt 2
  show π * outer_radius^2 - π * inner_radius^2 = 9 * π, from sorry

end tangent_segment_annulus_area_l826_826615


namespace circumcenter_necessity_and_sufficiency_l826_826243

variables {A B C P O : Type} {R PA PB PC : ℝ}
variables {sin 2A sin 2B sin 2C : ℝ}
variables {sin A sin B sin C : ℝ}

noncomputable def circumcenter (A B C P O : Type) : Prop :=
PA = PB ∧ PB = PC ∧ PC = R

noncomputable def equality_condition (PA PB PC R sin2A sin2B sin2C sinA sinB sinC : ℝ) : Prop :=
PA^2 * sin2A + PB^2 * sin2B + PC^2 * sin2C = 4 * R^2 * sinA * sinB * sinC

theorem circumcenter_necessity_and_sufficiency
  (PA PB PC R sin2A sin2B sin2C sinA sinB sinC : ℝ) :
  circumcenter A B C P O ↔ equality_condition PA PB PC R sin2A sin2B sin2C sinA sinB sinC :=
by
  sorry

end circumcenter_necessity_and_sufficiency_l826_826243


namespace part1_part2_l826_826647

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f(x) + f(2 * x + 1) ≥ 6 ↔ x ∈ (-∞, -1] ∪ [3, ∞) :=
by sorry

theorem part2 (a b m : ℝ) (h_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1) :
  (∀ x : ℝ, f(x - m) - f(-x) ≤ (4 / a) + (1 / b)) ↔ m ∈ [-13, 5] :=
by sorry

end part1_part2_l826_826647


namespace abs_diff_max_min_l826_826097

noncomputable def min_and_max_abs_diff (x : ℝ) : ℝ :=
|x - 2| + |x - 3| - |x - 1|

theorem abs_diff_max_min (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) :
  ∃ (M m : ℝ), M = 0 ∧ m = -1 ∧
    M = max (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) ∧ 
    m = min (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) :=
by
  use [0, -1]
  split
  case inl => sorry
  case inr => sorry

end abs_diff_max_min_l826_826097


namespace remaining_numbers_count_l826_826414

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826414


namespace problem_statement_l826_826261

noncomputable def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * sqrt (2 * π))) * exp (-(x - μ)^2 / (2 * σ^2))

noncomputable def P_leq (X : ℝ → ℝ) (a : ℝ) : ℝ :=
  ∫ x in (-∞..a), X x

theorem problem_statement {σ : ℝ} (h_pos : σ > 0) :
  ∀ (X : ℝ → ℝ), 
    (∀ x, X x = normal_distribution 2 σ x) → 
    P_leq X 4 = 0.84 → 
    P_leq X 0 = 0.16 :=
by
  intro X hX hP
  sorry

end problem_statement_l826_826261


namespace exponent_property_l826_826300

theorem exponent_property (a b : ℝ) (h1 : 10^a = 8) (h2 : 10^b = 2) : 10^(a - 2 * b) = 2 :=
by
  -- skip the proof with sorry
  sorry

end exponent_property_l826_826300


namespace remaining_numbers_count_l826_826415

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826415


namespace part1_part2_l826_826972

variable (a : ℝ)

theorem part1 (h : 0 < a ∧ a < 1 / 2) : 1 - a > a^2 :=
sorry

theorem part2 (h : 0 < a ∧ a < 1 / 2) :
  let A := 1 - a^2
  let B := 1 + a^2
  let C := 1 / (1 - a)
  let D := 1 / (1 + a)
  in D < A ∧ A < B ∧ B < C :=
sorry

end part1_part2_l826_826972


namespace exists_fibonacci_with_three_trailing_zeros_l826_826055

-- Define the Fibonacci sequence as a function in Lean
def fibonacci (n : ℕ) : ℕ :=
  let fib_aux : ℕ → ℕ → ℕ → ℕ
    | 0, a, _ => a
    | (n + 1), a, b => fib_aux n b (a + b)
  in fib_aux n 1 1

-- The statement to prove: there exists an n such that fibonacci(n) ends in three trailing zeros
theorem exists_fibonacci_with_three_trailing_zeros : ∃ n : ℕ, fibonacci n % 1000 = 0 :=
by
  sorry

end exists_fibonacci_with_three_trailing_zeros_l826_826055


namespace abs_lt_two_nec_but_not_suff_l826_826519

theorem abs_lt_two_nec_but_not_suff (x : ℝ) :
  (|x - 1| < 2) → (0 < x ∧ x < 3) ∧ ¬((0 < x ∧ x < 3) → (|x - 1| < 2)) := sorry

end abs_lt_two_nec_but_not_suff_l826_826519


namespace arrangement_count_l826_826601

-- Definitions of the conditions
def volunteers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : ℕ := 1
def B : ℕ := 2
def monday_to_friday : Finset ℕ := {1, 2, 3, 4, 5}

-- The statement to prove
theorem arrangement_count :
  -- At least one of A and B must participate
  let with_a := volunteers \ {B}
  let with_b := volunteers \ {A}
  let with_a_and_b := volunteers \ {A, B}
  let only_one_participates := (with_a.card.choose 4 * 5.factorial) + (with_b.card.choose 4 * 5.factorial)
  
  -- If both A and B participate, they must not be adjacent
  let non_adjacent_arrangement := 1440 -- precomputed for simplicity
  only_one_participates + non_adjacent_arrangement = 5040 := 
sorry

end arrangement_count_l826_826601


namespace midpoint_C_collinear_circumcenter_l826_826765

variable {A B C P Q K L : Point}
variable {ACPQ BKLC: Rectangle}
variable {circum_circle: Circle}

def midpoint (X Y : Point) : Point := sorry
def collinear (X Y Z : Point) : Prop := sorry
def is_center_of_circumcircle (O : Point) (ABC : Triangle) : Prop := sorry
def area (r : Rectangle) : ℝ := sorry
def equal_area (r1 r2 : Rectangle) : Prop := (area r1) = (area r2)
def is_acute (t: Triangle) : Prop := sorry

theorem midpoint_C_collinear_circumcenter {A B C P Q K L C O: Point}
  (h_triangle_ABC : is_acute (Triangle.mk A B C))
  (h_rect1 : ACPQ = Rectangle.mk A C P Q)
  (h_rect2 : BKLC = Rectangle.mk B K L C)
  (h_equal_area : equal_area ACPQ BKLC)
  (h_circumcenter : is_center_of_circumcircle O (Triangle.mk A B C)) :
  collinear (midpoint P L) C O := 
sorry

end midpoint_C_collinear_circumcenter_l826_826765


namespace sum_abs_roots_poly_eq_sqrt_17_l826_826594

theorem sum_abs_roots_poly_eq_sqrt_17 :
  let f := λ x : ℝ, x^4 - 4 * x^3 + 6 * x^2 + 14 * x - 12 in
  ∑ (r : ℝ) in (multiset.roots f).filter (λ x, x ≠ 0), |r| = Real.sqrt 17 :=
by
  -- Proof goes here
  sorry

end sum_abs_roots_poly_eq_sqrt_17_l826_826594


namespace distinct_pos_integers_inequality_l826_826977

theorem distinct_pos_integers_inequality {n : ℕ} (a : ℕ → ℕ) (h_pos : ∀ (k : ℕ), 1 ≤ k → k ≤ n → 0 < a k) 
  (h_distinct : ∀ (i j : ℕ), 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → i ≠ j → a i ≠ a j) :
  ∑ k in Finset.range n + 1, (a k : ℝ) / (k + 1)^2 ≥ ∑ k in Finset.range n + 1, 1 / (k + 1) :=
sorry

end distinct_pos_integers_inequality_l826_826977


namespace correct_equation_l826_826837

theorem correct_equation :
  (4 * 4 * 4 ≠ 3 * 4) ∧
  (5 ^ 3 ≠ 3 ^ 5) ∧
  ((-3) * (-3) * (-3) * (-3) = 3 ^ 4) ∧
  (¬ ((-2 / 3) ^ 3 ≠ (-2 / 3) * (-2 / 3) * (-2 / 3))) → 
  ((-3) * (-3) * (-3) * (-3) = 3 ^ 4) :=
by
  intros hA hB hC hD
  exact hC

end correct_equation_l826_826837


namespace tangent_line_sum_l826_826265

theorem tangent_line_sum {f : ℝ → ℝ} (h_tangent : ∀ x, f x = (1/2 * x) + 2) :
  (f 1) + (deriv f 1) = 3 :=
by
  -- derive the value at x=1 and the derivative manually based on h_tangent
  sorry

end tangent_line_sum_l826_826265


namespace train_speed_l826_826897

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l826_826897


namespace sum_of_elements_in_T_l826_826734

def T := {x : ℕ | ∀ d ∈ digits 3 x, d < 3 ∧ 6561 ≤ x ∧ x ≤ 24263}

theorem sum_of_elements_in_T :
  (∑ x in T, x) = 21102002₃ := sorry

end sum_of_elements_in_T_l826_826734


namespace find_a_l826_826671

theorem find_a (a : ℝ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := sorry

end find_a_l826_826671


namespace line_passes_through_parabola_vertex_l826_826220

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_parabola_vertex_l826_826220


namespace bisect_perimeter_bisect_area_l826_826532

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def midpoint_arc (B C : Point) (circle : set Point) : Point := sorry
noncomputable def bisecting_line_perimeter (A B C : Point) (circle : set Point) : Line :=
  let M := midpoint_arc B C circle in
  let N := midpoint B (A + C) in
  ⟨N, M⟩

noncomputable def bisecting_line_area (A B C : Point) (circle : set Point) : Line :=
  let D := midpoint_arc B C circle in
  let F := midpoint B C in
  let l := parallel_line_through F (A - D) in
  let E := intersect l (line_through A C) in
  ⟨D, E⟩

theorem bisect_perimeter (A B C : Point) (circle : set Point) :
  bisects_perimeter (bisecting_line_perimeter A B C circle) (convex_boundary A B C circle) :=
sorry

theorem bisect_area (A B C : Point) (circle : set Point) :
  bisects_area (bisecting_line_area A B C circle) (convex_area A B C circle) :=
sorry

end bisect_perimeter_bisect_area_l826_826532


namespace find_ab_l826_826252

theorem find_ab (a b c : ℤ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) (h3 : a + b + c = 21) : a * b = 10 := 
sorry

end find_ab_l826_826252


namespace sum_of_elements_in_T_l826_826733

def T := {x : ℕ | ∀ d ∈ digits 3 x, d < 3 ∧ 6561 ≤ x ∧ x ≤ 24263}

theorem sum_of_elements_in_T :
  (∑ x in T, x) = 21102002₃ := sorry

end sum_of_elements_in_T_l826_826733


namespace number_of_logs_in_stack_l826_826549

theorem number_of_logs_in_stack : 
    let first_term := 15 in
    let last_term := 5 in
    let num_terms := first_term - last_term + 1 in
    let average := (first_term + last_term) / 2 in
    let sum := average * num_terms in
    sum = 110 :=
by
  sorry

end number_of_logs_in_stack_l826_826549


namespace infinite_primes_with_integer_solutions_l826_826782

theorem infinite_primes_with_integer_solutions :
  ∀ᶠ p in Filter.atTop, ∃ x y : ℤ, prime p ∧ (x^2 + x + 1 = p * y) :=
sorry

end infinite_primes_with_integer_solutions_l826_826782


namespace rice_purchase_after_reductions_l826_826770

noncomputable def rice_quantity (P : ℝ) : ℝ :=
  let M := 20 * P
  let P_new1 := 0.80 * P
  let P_new2 := 0.72 * P
  let P_pound := P_new2 / 2.2
  let M_pounds := M * 2.2 / P_pound
  let M_kg := M_pounds / 2.2
  M_kg

theorem rice_purchase_after_reductions
  (P : ℝ)
  (hP_pos : P > 0) :
  rice_quantity P ≈ 27.78 :=
by
  sorry

end rice_purchase_after_reductions_l826_826770


namespace value_of_expression_l826_826308

theorem value_of_expression 
  (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 :=
by
  sorry

end value_of_expression_l826_826308


namespace rectangular_to_polar_coordinates_l826_826931

theorem rectangular_to_polar_coordinates :
  ∀ (x y : ℝ) (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) →
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) →
  (r, θ) = (4, Real.pi / 4) :=
by
  intros x y r θ h1 h2 h3
  sorry

end rectangular_to_polar_coordinates_l826_826931


namespace cyclic_quadrilateral_area_l826_826239

theorem cyclic_quadrilateral_area 
  (A B C D : Type) 
  (AB BC CD DA : ℝ)
  (h1 : AB = 2) 
  (h2 : BC = 6) 
  (h3 : CD = 4) 
  (h4 : DA = 4) 
  (is_cyclic_quad : True) : 
  area A B C D = 8 * Real.sqrt 3 := 
sorry

end cyclic_quadrilateral_area_l826_826239


namespace geometric_sequence_a_l826_826471

theorem geometric_sequence_a (a : ℝ) (h1 : a > 0) (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 180 / 49) :
  a = 32.07 :=
by sorry

end geometric_sequence_a_l826_826471


namespace feasible_14_not_feasible_12_l826_826508

-- Definitions of variables and conditions
variables (n ℓ a b c : ℕ)

-- Condition: Formula for n and inequalities
def valid_hexagon (n ℓ a b c : ℕ) : Prop :=
  n = ℓ^2 - a^2 - b^2 - c^2 ∧
  ℓ > a + b ∧
  ℓ > a + c ∧
  ℓ > b + c

-- Prove that 14 is feasible
theorem feasible_14 : ∃ ℓ a b c, valid_hexagon 14 ℓ a b c :=
by
  exists 5 3 1 1
  unfold valid_hexagon
  exact ⟨by simp [Nat.pow_two], by norm_num, by norm_num, by norm_num⟩

-- Prove that 12 is not feasible
theorem not_feasible_12 : ¬ ∃ ℓ a b c, valid_hexagon 12 ℓ a b c :=
by
  intro h
  cases h with ℓ h
  cases h with a h
  cases h with b h
  cases h with c h
  have h₁ := h.1
  have h₂ := h.2.1
  have h₃ := h.2.2.1
  have h₄ := h.2.2.2
  sorry

end feasible_14_not_feasible_12_l826_826508


namespace sqrt_cos_cubic_poly_identity_l826_826196

theorem sqrt_cos_cubic_poly_identity :
  sqrt ((3 - cos^2 (π / 9)) * (3 - cos^2 (2 * π / 9)) * (3 - cos^2 (4 * π / 9))) = 27 * sqrt 2 / 16 := 
by
  sorry

end sqrt_cos_cubic_poly_identity_l826_826196


namespace subsets_of_B_are_valid_l826_826604

def B : Set ℕ := {0, 1, 2}

def valid_subsets : Set (Set ℕ) := {{}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}

theorem subsets_of_B_are_valid (A : Set ℕ) (hA : A ⊆ B) : 
  A = ∅ ∨ A = {0} ∨ A = {1} ∨ A = {2} ∨ A = {0, 1} ∨ A = {0, 2} ∨ A = {1, 2} :=
begin
  sorry
end

end subsets_of_B_are_valid_l826_826604


namespace rick_kept_cards_l826_826021

def initial_cards : ℕ := 130
def cards_given_to_miguel : ℕ := 13
def friends : ℕ := 8
def cards_per_friend : ℕ := 12
def sisters : ℕ := 2
def cards_per_sister : ℕ := 3

theorem rick_kept_cards :
  ∃ (kept : ℕ), kept = initial_cards - cards_given_to_miguel - (friends * cards_per_friend) - (sisters * cards_per_sister) ∧ kept = 15 :=
begin
  sorry
end

end rick_kept_cards_l826_826021


namespace remainder_mod_9_l826_826835

noncomputable def sqrt3 : ℝ := Real.sqrt 3

def a (n : ℕ) : ℝ := (sqrt3 + 5)^n + (sqrt3 - 5)^n

def b (n : ℕ) : ℝ := (sqrt3 + 5)^n - (sqrt3 - 5)^n

theorem remainder_mod_9 :
  (b 103) % 9 = 1 :=
sorry

end remainder_mod_9_l826_826835


namespace part_1_part_2_l826_826608

noncomputable def f (a m x : ℝ) := a ^ m / x

theorem part_1 (a : ℝ) (m : ℝ) (H1 : a > 1) (H2 : ∀ x, x ∈ Set.Icc a (2*a) → f a m x ∈ Set.Icc (a^2) (a^3)) :
  a = 2 :=
sorry

theorem part_2 (t : ℝ) (s : ℝ) (H1 : ∀ x, x ∈ Set.Icc 0 s → (x + t) ^ 2 + 2 * (x + t) ≤ 3 * x) :
  s ∈ Set.Ioc 0 5 :=
sorry

end part_1_part_2_l826_826608


namespace segment_count_at_least_1998_l826_826570

theorem segment_count_at_least_1998 (P : Finset (ℝ × ℝ)) (h1 : P.card = 111) (h2 : ∀ p ∈ P, (p.1)^2 + (p.2)^2 ≤ 1) :
  ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ s ∈ S, s.1.1 = s.2.1 ∧ s.1.2 = s.2.2) → (S.card ≥ 1998) ∧ (∀ s ∈ S, (s.1.1 - s.2.1)^2 + (s.1.2 - s.2.2)^2 ≤ 3) := 
sorry

end segment_count_at_least_1998_l826_826570


namespace distance_between_trees_l826_826126

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (num_gaps : ℕ) 
  (h1 : yard_length = 250) (h2 : num_trees = 51) (h3 : num_gaps = num_trees - 1) :
  yard_length / num_gaps = 5 :=
by
  rw [h1, h2, h3]
  simp only [Nat.sub_tsub_assoc_pred, Nat.pred_sub_pred, Nat.sub_zero, Nat.pred_succ, Nat.div_self]
  exact Nat.div_self (Nat.pos_of_num_ne_zero $ lt_trans zero_lt_one $ Nat.lt_base 50 0)
  exact zero_le _

end distance_between_trees_l826_826126


namespace sunset_time_on_that_day_l826_826383

-- Define the relevant times and delay
def delay_per_day : ℝ := 1.2
def number_of_days : ℕ := 40
def current_time : ℝ := 6 * 60 + 10 -- 6:10 PM in minutes
def sunset_delay_today : ℕ := 38

-- Define the sunset time today in minutes
def sunset_time_today : ℕ := current_time + sunset_delay_today

-- Define the total delay over the given period
def total_delay : ℝ := delay_per_day * number_of_days

-- Prove the sunset time on the certain day
theorem sunset_time_on_that_day :
  let sunset_time_on_that_day := sunset_time_today - total_delay in
  sunset_time_on_that_day = 6 * 60 :=
by
  sorry

end sunset_time_on_that_day_l826_826383


namespace composite_has_at_least_three_factors_l826_826869

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n :=
sorry

end composite_has_at_least_three_factors_l826_826869


namespace distance_between_equally_spaced_parallel_lines_l826_826480

noncomputable def distance_between_parallel_lines (r : ℝ) : ℝ :=
  let d := sqrt (152 / 81) in
  d

theorem distance_between_equally_spaced_parallel_lines (r : ℝ) : 
  distance_between_parallel_lines r = 41 / 30 :=
by
  sorry

end distance_between_equally_spaced_parallel_lines_l826_826480


namespace first_scenario_machines_l826_826313

theorem first_scenario_machines (M : ℕ) (h1 : 20 = 10 * 2 * M) (h2 : 140 = 20 * 17.5 * 2) : M = 5 :=
by sorry

end first_scenario_machines_l826_826313


namespace race_possibilities_l826_826289

theorem race_possibilities :
  let participants := ["Harry", "Ron", "Hermione", "Neville"] in 
  ∃ (orders : List (List String)), (orders.length = 18) ∧
    ∀ order ∈ orders, "Hermione" ∈ order.init ∧ (order.nodup = true) :=
by
  sorry

end race_possibilities_l826_826289


namespace CatFoodConsumption_l826_826434

/-- 
  Roy's cat now eats 1/4 of a can of cat food each morning, 1/6 of a can every midday, and 1/3 of a can each evening.
  If Roy starts with 8 cans of cat food on Monday morning, the cat finishes all the cat food on Sunday.
-/

theorem CatFoodConsumption :
  let daily_consumption := (1/4 : ℚ) + (1/6 : ℚ) + (1/3 : ℚ),
      total_cans := 8,
      consumption_on_Day (n : ℕ) := n * daily_consumption in
  (∃ d : ℕ, d * daily_consumption = total_cans) → d = 7 := 
sorry

end CatFoodConsumption_l826_826434


namespace speed_of_each_train_proof_l826_826117

noncomputable def speed_of_each_train (v : ℝ) : ℝ :=
  (v * 3.6 : ℝ)

theorem speed_of_each_train_proof :
  ∀ (v : ℝ),
    (∀ (d t : ℝ), d = 240 ∧ t = 24 → v = d / t / 2) →
    speed_of_each_train (5 : ℝ) = 18 :=
by
  intros v h
  have h1 : v = 5 := h 240 24 ⟨rfl, rfl⟩
  rw [h1]
  unfold speed_of_each_train
  norm_num
  done

end speed_of_each_train_proof_l826_826117


namespace kickball_students_total_l826_826761

theorem kickball_students_total :
  let wednesday_early_morning := 37
  let wednesday_late_morning := 12 + Int.ofNat (Float.toInt (0.75 * wednesday_early_morning))
  let wednesday_afternoon := 15 + Int.ofNat (Float.toInt (0.50 * (wednesday_early_morning + wednesday_late_morning)))
  let thursday_morning := Int.ofNat (Float.toInt (0.90 * wednesday_early_morning))
  let thursday_afternoon := 14 + Int.ofNat (Float.toInt (0.80 * thursday_morning))
  
  let wednesday_total := wednesday_early_morning + wednesday_late_morning + wednesday_afternoon
  let thursday_total := thursday_morning + thursday_afternoon
  
  wednesday_total + thursday_total = 185 := 
by
  sorry

end kickball_students_total_l826_826761


namespace surface_area_unchanged_if_one_small_cube_is_removed_l826_826872

-- Definitions for the problem
def cube (n : ℕ) := { x : ℝ × ℝ × ℝ | abs x.1 < n/2 ∧ abs x.2 < n/2 ∧ abs x.3 < n/2 }
def large_cube := cube 2  -- The large cube is of side length 2
def small_cube := cube 1  -- Each small cube is of side length 1

-- Condition: A large cube is formed by assembling 8 smaller cubes of the same size
def large_cube_from_smaller_cubes := large_cube = ⋃ (i, j, k : ℤ) (h : (i, j, k) ∈ ({-1, 1} : set ℤ)^3), translate (small_cube) (i, j, k)

-- Statement to prove
theorem surface_area_unchanged_if_one_small_cube_is_removed :
  large_cube_from_smaller_cubes →
  ∀ (i j k : ℤ) (hij : (i, j, k) ∈ ({-1, 1} : set ℤ)^3),
    surface_area (large_cube \ translate (small_cube) (i, j, k)) = surface_area large_cube := sorry

end surface_area_unchanged_if_one_small_cube_is_removed_l826_826872


namespace ratio_tough_to_good_sales_l826_826010

-- Define the conditions
def tough_week_sales : ℤ := 800
def total_sales : ℤ := 10400
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the problem in Lean 4:
theorem ratio_tough_to_good_sales : ∃ G : ℤ, (good_weeks * G) + (tough_weeks * tough_week_sales) = total_sales ∧ 
  (tough_week_sales : ℚ) / (G : ℚ) = 1 / 2 :=
sorry

end ratio_tough_to_good_sales_l826_826010


namespace ellipse_eqn_proof_area_proof_line_eqn_proof_l826_826651

-- Condition Definitions
def ellipse_eqn (t : ℝ) (t_pos : t > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / (2 * t^2) + y^2 / t^2 = 1) ↔ ((x, y) ∈ { p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1 })

def minimum_distance (t : ℝ) (t_pos : t > 0) (c : ℝ) : Prop :=
  ∀ (d : ℝ), (d = c) ↔ (c = 2 * √2 - 2 ∧ c = (√2 * t - t))

def vectors_parallel (M N F1 F2 : ℝ × ℝ) : Prop :=
  ∀ (α : ℝ), (α * (F1.1 + M.1, F1.2 + M.2) = (F2.1 + N.1, F2.2 + N.2))

def dot_product_zero (N F1 F2 : ℝ × ℝ) (theta : ℝ) : Prop :=
  ∀ (cos_theta sin_theta : ℝ), (N = (2 * √2 * cos_theta, 2 * sin_theta)) →
                               ((2 * √2 * cos_theta + 2) * (2 * √2 * cos_theta - 2) + 4 * sin_theta^2 = 0)

def distance_difference_sqrt6 (M N F1 F2 : ℝ × ℝ) : Prop :=
  ∀ (|F1M| |F2N| : ℝ), (|F2N| - |F1M| = sqrt 6)

-- Proof Statements
theorem ellipse_eqn_proof : 
  ∃ (t : ℝ) (t_pos : t > 0), ellipse_eqn t t_pos → ellipse_eqn 2 sorry :=
sorry

theorem area_proof : 
  ∃ (F1 N F2 : ℝ × ℝ), dot_product_zero N F1 F2 0 → area_F1MN N F1 F2 = 4/3 :=
sorry

theorem line_eqn_proof : 
  ∃ (M N F1 F2 : ℝ × ℝ), distance_difference_sqrt6 M N F1 F2 →
  line_eqn F2 N = x + √2 y - 2 :=
sorry

end ellipse_eqn_proof_area_proof_line_eqn_proof_l826_826651


namespace tom_new_stamp_count_l826_826482

theorem tom_new_stamp_count (original_stamps : ℕ) (mike_gift : ℕ)
  (harry_gift_formula : mike_gift → ℕ) :
  original_stamps = 3000 →
  mike_gift = 17 →
  harry_gift_formula mike_gift = 2 * mike_gift + 10 →
  original_stamps + mike_gift + harry_gift_formula mike_gift = 3061 :=
begin
  intros h_orig h_mike h_harry,
  sorry
end

end tom_new_stamp_count_l826_826482


namespace find_insect_stickers_l826_826089

noncomputable def flower_stickers : ℝ := 15
noncomputable def animal_stickers : ℝ := 2 * flower_stickers - 3.5
noncomputable def space_stickers : ℝ := 1.5 * flower_stickers + 5.5
noncomputable def total_stickers : ℝ := 70
noncomputable def insect_stickers : ℝ := total_stickers - (animal_stickers + space_stickers)

theorem find_insect_stickers : insect_stickers = 15.5 := by
  sorry

end find_insect_stickers_l826_826089


namespace max_value_of_b_in_rectangular_prism_l826_826881

theorem max_value_of_b_in_rectangular_prism
  (V : ℕ)
  (a b c : ℕ)
  (prime_a : nat.prime a)
  (prime_b : nat.prime b)
  (prime_c : nat.prime c)
  (distinct_factors : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (volume_eq : a * b * c = V)
  (c_eq_three : c = 3)
  (abc_order : 1 < c ∧ c < b ∧ b < a) :
  b = 5 :=
sorry

end max_value_of_b_in_rectangular_prism_l826_826881


namespace remaining_length_after_cut_l826_826878

/- Definitions -/
def original_length (a b : ℕ) : ℕ := 5 * a + 4 * b
def rectangle_perimeter (a b : ℕ) : ℕ := 2 * (a + b)
def remaining_length (a b : ℕ) : ℕ := original_length a b - rectangle_perimeter a b

/- Theorem statement -/
theorem remaining_length_after_cut (a b : ℕ) : remaining_length a b = 3 * a + 2 * b := 
by 
  sorry

end remaining_length_after_cut_l826_826878


namespace count_final_numbers_l826_826409

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826409


namespace problem_statement_l826_826574

def curve_C1 (θ : ℝ) : ℝ × ℝ := (12 * Real.cos θ, 4 * Real.sin θ)

def curve_C2_polar (θ : ℝ) : ℝ := 3 / Real.cos (θ + π / 3)

def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

def point_Q_polar : ℝ × ℝ := (4 * Real.sqrt 2, π / 4)

def rectangular_Q : ℝ × ℝ := polar_to_rectangular (4 * Real.sqrt 2) (π / 4)

def line_C2_rectangular (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 6 = 0

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.1 + b * M.2 + c) / Real.sqrt (a^2 + b^2)

theorem problem_statement :
  ∃ (x y : ℝ), 
    line_C2_rectangular x y ∧ 
    rectangular_Q = (4, 4) ∧
    (∀ θ : ℝ, 2 - Real.sqrt 3 ≤ distance_to_line (midpoint (curve_C1 θ) rectangular_Q) 1 (-Real.sqrt 3) (-6)) :=
sorry

end problem_statement_l826_826574


namespace candy_sampling_l826_826512

theorem candy_sampling (total_customers caught_sampling not_caught_sampling : ℝ) :
  caught_sampling = 0.22 * total_customers →
  not_caught_sampling = 0.12 * (total_customers * sampling_percent) →
  (sampling_percent * total_customers = caught_sampling / 0.78) :=
by
  intros h1 h2
  sorry

end candy_sampling_l826_826512


namespace total_bill_is_correct_l826_826013

-- Define conditions as constant values
def cost_per_scoop : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

-- Define the total bill calculation
def total_bill := (pierre_scoops * cost_per_scoop) + (mom_scoops * cost_per_scoop)

-- State the theorem that the total bill equals 14
theorem total_bill_is_correct : total_bill = 14 := by
  sorry

end total_bill_is_correct_l826_826013


namespace parallelogram_UVCW_l826_826767

variables (A B C U V W : Type)
  [add_group A] [add_group B] [add_group C] [add_group U] [add_group V] [add_group W]

def is_isosceles_right (a b c : Type) [add_group a] [add_group b] [add_group c] : Prop :=
  let ab := (a + b) / 2,
      ac := (a + c) / 2,
      bc := (b + c) / 2 in
  ab * ab + ab * ab = ac * ac + ac * ac = bc * bc + bc * bc

theorem parallelogram_UVCW 
  (A B C U V W : Type) 
  [add_group A] [add_group B] [add_group C] [add_group U] [add_group V] [add_group W]
  (is_isosceles_right_AB : is_isosceles_right A U B)
  (is_isosceles_right_BC : is_isosceles_right C V B)
  (is_isosceles_right_CA : is_isosceles_right A W C) 
  : (U + V) - (W + C) = 0 :=
by 
  sorry

end parallelogram_UVCW_l826_826767


namespace like_terms_sum_l826_826851

theorem like_terms_sum (m n : ℤ) (h_x : 1 = m - 2) (h_y : 2 = n + 3) : m + n = 2 :=
by
  sorry

end like_terms_sum_l826_826851


namespace robin_earns_30_percent_more_than_erica_l826_826061

variable (E R C : ℝ)

theorem robin_earns_30_percent_more_than_erica
  (h1 : C = 1.60 * E)
  (h2 : C = 1.23076923076923077 * R) :
  R = 1.30 * E :=
by
  sorry

end robin_earns_30_percent_more_than_erica_l826_826061


namespace binomial_constant_term_l826_826703

theorem binomial_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, (∑ i in (finset.range (n + 1)), (finset.choose n i) * (3*x)^(n-i) * (-2/x)^i) = 256) →
  (∃ r : ℕ, (8 - 2 * r = 0) ∧ (-2)^r * (finset.choose 8 r) = 112) := sorry

end binomial_constant_term_l826_826703


namespace people_counted_on_second_day_l826_826177

theorem people_counted_on_second_day (x : ℕ) (H1 : 2 * x + x = 1500) : x = 500 :=
by {
  sorry -- Proof goes here
}

end people_counted_on_second_day_l826_826177


namespace track_meet_girls_short_hair_l826_826069

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end track_meet_girls_short_hair_l826_826069


namespace min_inequality_l826_826742

theorem min_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz_sum : x + y + z = 9) :
    ( \frac{x^2 + y^2}{3*(x+y)} + \frac{x^2 + z^2}{3*(x+z)} + \frac{y^2 + z^2}{3*(y+z)} ) ≥ 3 :=
by
  sorry

end min_inequality_l826_826742


namespace clear_time_is_approximately_7_point_1_seconds_l826_826487

-- Constants for the lengths of the trains in meters
def length_train1 : ℕ := 121
def length_train2 : ℕ := 165

-- Constants for the speeds of the trains in km/h
def speed_train1 : ℕ := 80
def speed_train2 : ℕ := 65

-- Kilometer to meter conversion
def km_to_meter (km : ℕ) : ℕ := km * 1000

-- Hour to second conversion
def hour_to_second (h : ℕ) : ℕ := h * 3600

-- Relative speed of the trains in meters per second
noncomputable def relative_speed_m_per_s : ℕ := 
  (km_to_meter (speed_train1 + speed_train2)) / hour_to_second 1

-- Total distance to be covered in meters
def total_distance : ℕ := length_train1 + length_train2

-- Time to be completely clear of each other in seconds
noncomputable def clear_time : ℝ := total_distance / (relative_speed_m_per_s : ℝ)

theorem clear_time_is_approximately_7_point_1_seconds :
  abs (clear_time - 7.1) < 0.01 :=
by
  sorry

end clear_time_is_approximately_7_point_1_seconds_l826_826487


namespace congruent_triangles_l826_826578

variables {A B C A1 B1 C1 A2 B2 C2 : Type*}
variables [metric_space A] [euclidean_geometry A]

-- Define the triangle ABC and points A1, B1, C1 based on the given conditions
def projections (triangle : triangle A B C) : Prop :=
  let CA := line_segment C A in
  let AB := line_segment A B in
  let BC := line_segment B C in
  ∃ A1 B1 C1 : A,
  A1 ∈ BC ∧
  is_projection A1 CA B1 ∧
  is_projection B1 AB C1 ∧
  is_projection C1 BC A1

-- Define the perpendicularity conditions for points A2, B2, C2
def perpendiculars (triangle : triangle A B C) : Prop :=
  ∃ A2 B2 C2 : A,
  A2 ∈ (line_segment B C) ∧
  B2 ∈ (line_segment C A) ∧
  C2 ∈ (line_segment A B) ∧
  is_perpendicular (line_segment A2 C2) (line_segment B A) ∧
  is_perpendicular (line_segment C2 B2) (line_segment A C) ∧
  is_perpendicular (line_segment B2 A2) (line_segment C B)

-- Proof statement
theorem congruent_triangles (triangle : triangle A B C) :
  projections triangle ∧ perpendiculars triangle →
  triangle_congruent (triangle_of_points A1 B1 C1) (triangle_of_points A2 B2 C2) :=
sorry

end congruent_triangles_l826_826578


namespace distinct_values_min_l826_826533

theorem distinct_values_min (lst : List ℕ) (h_length : lst.length = 3042) (h_mode : ∃! n, n ∈ lst ∧ (lst.count n = 15)) :
  ∃ n, ∀ m, m ∉ lst → list.distinct_values_min lst = 218 :=
by
  -- proof to be given here
  sorry

end distinct_values_min_l826_826533


namespace recommended_water_intake_l826_826078

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end recommended_water_intake_l826_826078


namespace general_term_formulas_smallest_n_satisfies_l826_826631

-- Define the sequences and conditions
def a_n (n : ℕ) : ℝ := (1 / 2) ^ (n - 2)
def b_n (n : ℕ) : ℝ := 2 * n

-- Conditions
axiom b2_eq_4a2 : b_n 2 = 4 * a_n 2
axiom a2b3_eq_6 : a_n 2 * b_n 3 = 6

-- Define the smallest positive integer n such that a_{b_n} < 0.001
def smallest_n : ℕ := 6

-- Prove that the sequence definitions are derived correctly
theorem general_term_formulas : 
  ∀ n : ℕ, n ≥ 1 → a_n n = (1 / 2) ^ (n - 2) ∧ b_n n = 2 * n := 
by
  sorry

-- Prove that the smallest n for which a_{b_n} < 0.001 is 6
theorem smallest_n_satisfies : 
  0 < smallest_n ∧ a_n (b_n smallest_n) < 0.001 :=
by
  sorry

end general_term_formulas_smallest_n_satisfies_l826_826631


namespace mean_of_remaining_six_numbers_l826_826458

theorem mean_of_remaining_six_numbers (mean_initial : ℕ) (size_initial : ℕ) (mean_removed : ℕ) (size_removed : ℕ) :
  mean_initial = 12 ∧ size_initial = 8 ∧ mean_removed = 18 ∧ size_removed = 2 →
  (96 - 36) / 6 = 10 :=
by
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry

end mean_of_remaining_six_numbers_l826_826458


namespace cost_per_sqft_is_3_l826_826377

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def extra_cost_per_sqft : ℝ := 1
def total_cost : ℝ := 4800

theorem cost_per_sqft_is_3
    (area : ℝ := deck_length * deck_width)
    (sealant_cost : ℝ := area * extra_cost_per_sqft)
    (deck_construction_cost : ℝ := total_cost - sealant_cost) :
    deck_construction_cost / area = 3 :=
by
  sorry

end cost_per_sqft_is_3_l826_826377


namespace amounts_saved_arent_more_when_purchases_together_l826_826131

theorem amounts_saved_arent_more_when_purchases_together :
  let book_price := 120
  let free_books_for_five := 2
  let alice_books := 10
  let bob_books := 15
  ∀ (alice_savings_individual bob_savings_individual combined_savings : ℤ),

  -- Calculate individual total cost without offer
  let alice_cost_no_offer := alice_books * book_price in
  let bob_cost_no_offer := bob_books * book_price in

  -- Calculate individual total cost with the offer
  let alice_paid_books := alice_books - (alice_books / 5 * free_books_for_five) in
  let bob_paid_books := bob_books - (bob_books / 5 * free_books_for_five) in
  let alice_cost_with_offer := alice_paid_books * book_price in
  let bob_cost_with_offer := bob_paid_books * book_price in

  -- Calculate individual savings
  let alice_savings := alice_cost_no_offer - alice_cost_with_offer in
  let bob_savings := bob_cost_no_offer - bob_cost_with_offer in

  -- Calculate combined total cost without offer
  let combined_books := alice_books + bob_books in
  let combined_cost_no_offer := combined_books * book_price in

  -- Calculate combined total cost with the offer
  let combined_paid_books := combined_books - (combined_books / 5 * free_books_for_five) in
  let combined_cost_with_offer := combined_paid_books * book_price in

  -- Calculate combined savings
  let combined_savings := combined_cost_no_offer - combined_cost_with_offer in

  -- Calculate total savings when purchased separately
  let separate_savings := alice_savings + bob_savings in

  separate_savings = combined_savings :=
sorry

end amounts_saved_arent_more_when_purchases_together_l826_826131


namespace solve_pizza_problem_l826_826536

noncomputable def pizza_problem : Prop :=
  ∃ (h p j a b c x : ℕ),
  h + p + j + a + b + c + x = 24 ∧
  h + a + b + x = 15 ∧
  p + a + c + x = 10 ∧
  j + b + c + x = 14 ∧
  j = x ∧
  x = 5

theorem solve_pizza_problem : pizza_problem :=
begin
  -- Proof will go here
  sorry
end

end solve_pizza_problem_l826_826536


namespace angle_BAC_in_isosceles_triangle_l826_826342

theorem angle_BAC_in_isosceles_triangle
  (A B C D : Type)
  (AB AC : ℝ)
  (BD DC : ℝ)
  (angle_BDA : ℝ)
  (isosceles_triangle : AB = AC)
  (midpoint_D : BD = DC)
  (external_angle_D : angle_BDA = 120) :
  ∃ (angle_BAC : ℝ), angle_BAC = 60 :=
by
  sorry

end angle_BAC_in_isosceles_triangle_l826_826342


namespace yellow_gumdrops_count_l826_826871

theorem yellow_gumdrops_count 
  (total_gumdrops : ℕ) 
  (green_gumdrops : ℕ)
  (blue_percentage : ℚ) 
  (brown_percentage : ℚ) 
  (red_percentage : ℚ) 
  (yellow_percentage : ℚ) 
  (green_percentage : ℚ) :
  blue_percentage = 0.4 → 
  brown_percentage = 0.15 → 
  red_percentage = 0.1 → 
  yellow_percentage = 0.2 → 
  green_gumdrops = 50 → 
  green_percentage = 1 - (blue_percentage + brown_percentage + red_percentage + yellow_percentage) →
  green_percentage * total_gumdrops = green_gumdrops →
  let total_red := red_percentage * total_gumdrops,
      total_yellow := yellow_percentage * total_gumdrops in
  total_red / 3 + total_yellow = 78 :=
sorry -- Proof is omitted for now

end yellow_gumdrops_count_l826_826871


namespace sequence_sum_difference_l826_826831

def sum_odd (n : ℕ) : ℕ := n * n
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sequence_sum_difference :
  sum_even 1500 - sum_odd 1500 + sum_triangular 1500 = 563628000 :=
by
  sorry

end sequence_sum_difference_l826_826831


namespace isosceles_triangle_perimeter_l826_826695

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  ∃ (c : ℝ), (a = b ∧ 7 = c ∨ a = c ∧ 7 = b) ∧ a + b + c = 17 :=
by
  use 17
  sorry

end isosceles_triangle_perimeter_l826_826695


namespace base_k_to_decimal_l826_826314

theorem base_k_to_decimal (k : ℕ) (h : 0 < k ∧ k < 10) : 
  1 * k^2 + 7 * k + 5 = 125 → k = 8 := 
by
  sorry

end base_k_to_decimal_l826_826314


namespace cyclic_quadrilateral_area_l826_826236

def area_of_cyclic_quadrilateral (AB BC CD DA : ℝ) : ℝ :=
  let A := 120 * (real.pi / 180) -- Angle in radians
  16 * (real.sin A)

theorem cyclic_quadrilateral_area
  (AB BC CD DA : ℝ)
  (hAB : AB = 2)
  (hBC : BC = 6)
  (hCD : CD = 4)
  (hDA : DA = 4)
  (hCyclic : true) -- Assumption that quadrilateral is cyclic
  : area_of_cyclic_quadrilateral AB BC CD DA = 8 * real.sqrt 3 :=
by
  rw [hAB, hBC, hCD, hDA]
  sorry

end cyclic_quadrilateral_area_l826_826236


namespace pythagorean_triple_B_l826_826105

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l826_826105


namespace cannot_be_factored_l826_826982

-- Define the polynomial with integer coefficients
def polynomial (b c d : ℤ) : ℤ[X] := X^3 + b * X^2 + c * X + d

-- Define the conditions
def condition (b c d : ℤ) := (b * d + c * d) % 2 = 1

-- The theorem we want to prove
theorem cannot_be_factored (b c d : ℤ) : condition b c d → ¬ ∃ p q r : ℤ, polynomial b c d = (X + polynomial.C p) * (X^2 + q * X + r) :=
  by
  intros h
  sorry

end cannot_be_factored_l826_826982


namespace perimeter_reduction_percentage_l826_826516

-- Given initial dimensions x and y
-- Initial Perimeter
def initial_perimeter (x y : ℝ) : ℝ := 2 * (x + y)

-- First reduction
def first_reduction_length (x : ℝ) : ℝ := 0.9 * x
def first_reduction_width (y : ℝ) : ℝ := 0.8 * y

-- New perimeter after first reduction
def new_perimeter_first (x y : ℝ) : ℝ := 2 * (first_reduction_length x + first_reduction_width y)

-- Condition: new perimeter is 88% of the initial perimeter
def perimeter_condition (x y : ℝ) : Prop := new_perimeter_first x y = 0.88 * initial_perimeter x y

-- Solve for x in terms of y
def solve_for_x (y : ℝ) : ℝ := 4 * y

-- Second reduction
def second_reduction_length (x : ℝ) : ℝ := 0.8 * x
def second_reduction_width (y : ℝ) : ℝ := 0.9 * y

-- New perimeter after second reduction
def new_perimeter_second (x y : ℝ) : ℝ := 2 * (second_reduction_length x + second_reduction_width y)

-- Proof statement
theorem perimeter_reduction_percentage (x y : ℝ) (h : perimeter_condition x y) : 
  new_perimeter_second x y = 0.82 * initial_perimeter x y :=
by
  sorry

end perimeter_reduction_percentage_l826_826516


namespace parallelogram_side_length_l826_826535

theorem parallelogram_side_length 
  (s : ℝ) 
  (A : ℝ)
  (angle : ℝ)
  (adj1 adj2 : ℝ) 
  (h : adj1 = s) 
  (h1 : adj2 = 2 * s) 
  (h2 : angle = 30)
  (h3 : A = 8 * Real.sqrt 3): 
  s = 2 * Real.sqrt 2 :=
by
  -- sorry to skip proofs
  sorry

end parallelogram_side_length_l826_826535


namespace complex_number_quadrant_l826_826740

def quadrant (z: ℂ): string :=
  if z.re > 0 ∧ z.im > 0 then "first"
  else if z.re < 0 ∧ z.im > 0 then "second"
  else if z.re < 0 ∧ z.im < 0 then "third"
  else if z.re > 0 ∧ z.im < 0 then "fourth"
  else "on axis"

theorem complex_number_quadrant
  (i: ℂ)
  (hi: i = complex.I) :
  quadrant ((i - 3) / (1 + i)) = "second" :=
by
  sorry

end complex_number_quadrant_l826_826740


namespace four_digit_numbers_count_l826_826460

theorem four_digit_numbers_count : (3:ℕ) ^ 4 = 81 := by
  sorry

end four_digit_numbers_count_l826_826460


namespace max_bishops_1000x1000_l826_826844

def bishop_max_non_attacking (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem max_bishops_1000x1000 : bishop_max_non_attacking 1000 = 1998 :=
by sorry

end max_bishops_1000x1000_l826_826844


namespace explicit_form_of_h_range_of_a_for_positive_f_range_of_a_for_increasing_F_l826_826275

def f (a x : ℝ) : ℝ := a * x^2 - 2 * x + 2

def h (a x : ℝ) : ℝ := 
  let t : ℝ := 10^x
  f a x + x + 1

theorem explicit_form_of_h (a : ℝ) :
  h a x = a * (log x)^2 - log x + 3 := by
  sorry

theorem range_of_a_for_positive_f (a : ℝ) :
  (∀ x ∈ Icc (1:ℝ) 2, f a x > 0) → a > 1 / 2 := by
  sorry

def F (a x : ℝ) : ℝ := 
  abs (f a x)

theorem range_of_a_for_increasing_F (a : ℝ) :
  (∀ x1 x2 ∈ Icc (1:ℝ) 2, x1 ≠ x2 → (F a x1 - F a x2) / (x1 - x2) > 0) → 
  a ∈ Icc (-∞: ℝ) 0 ∪ Icc 1 (∞: ℝ) := by
  sorry

end explicit_form_of_h_range_of_a_for_positive_f_range_of_a_for_increasing_F_l826_826275


namespace intersection_eq_l826_826000

noncomputable def A := {x : ℝ | x^2 - 4*x + 3 < 0 }
noncomputable def B := {x : ℝ | 2*x - 3 > 0 }

theorem intersection_eq : (A ∩ B) = {x : ℝ | (3 / 2) < x ∧ x < 3} := by
  sorry

end intersection_eq_l826_826000


namespace at_least_one_not_less_than_two_l826_826794

variable (x y z : ℝ)

noncomputable def a := x + 1 / y
noncomputable def b := y + 1 / z
noncomputable def c := z + 1 / x

theorem at_least_one_not_less_than_two (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  a x y z ≥ 2 ∨ b x y z ≥ 2 ∨ c x y z ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l826_826794


namespace remaining_numbers_after_erasure_l826_826386

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826386


namespace minimum_value_of_f_inequality_proof_l826_826280

-- Step 1: Definitions of the problem
def f (x m : ℝ) : ℝ := |x - m| + |x + 1|

theorem minimum_value_of_f (m : ℝ) : m = -5 ∨ m = 3 → (f 0 m = 4) := by
  sorry

-- Step 2: Definitions for the inequality proof
variables {a b c : ℝ}
hypothesis h : a + 2 * b + 3 * c = 3

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) ≥ 3 := by
  sorry

end minimum_value_of_f_inequality_proof_l826_826280


namespace g_is_odd_l826_826344

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g x = -g (-x) :=
by
  intros x
  have : g (-x) = (-1 / (3^x - 1)) + (-1 / 3),
    sorry
  rw this
  simp [g]

end g_is_odd_l826_826344


namespace simplify_expression_l826_826027

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l826_826027


namespace first_discount_percentage_l826_826052

theorem first_discount_percentage (D : ℝ) :
  (345 * (1 - D / 100) * 0.75 = 227.70) → (D = 12) :=
by
  intro cond
  sorry

end first_discount_percentage_l826_826052


namespace min_value_of_a_l826_826046

-- Conditions
variables {a b c : ℕ}
variables {p q : ℝ}
variables (P : ℝ → ℝ)

-- The definition of the polynomial
def polynomial (x : ℝ) : ℝ := a * x^2 - b * x + c

-- Root conditions
variable h_roots : polynomial a b c p = 0 ∧ polynomial a b c q = 0

-- Given conditions on roots and coefficients
variable h_range : 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1
variable h_pos : 0 < a ∧ 0 < b ∧ 0 < c

-- Vieta's formulas
variable h_vieta_sum : p + q = b / a
variable h_vieta_product : p * q = c / a

-- Discriminant condition
variable h_discriminant : b^2 - 4 * a * c > 0

theorem min_value_of_a (h_roots : polynomial a b c p = 0 ∧ polynomial a b c q = 0)
                       (h_range : 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1)
                       (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
                       (h_vieta_sum : p + q = b / a)
                       (h_vieta_product : p * q = c / a)
                       (h_discriminant : b^2 - 4 * a * c > 0)
                       : a = 5 :=
sorry

end min_value_of_a_l826_826046


namespace total_num_problems_eq_30_l826_826874

-- Define the conditions
def test_points : ℕ := 100
def points_per_3_point_problem : ℕ := 3
def points_per_4_point_problem : ℕ := 4
def num_4_point_problems : ℕ := 10

-- Define the number of 3-point problems
def num_3_point_problems : ℕ :=
  (test_points - num_4_point_problems * points_per_4_point_problem) / points_per_3_point_problem

-- Prove the total number of problems is 30
theorem total_num_problems_eq_30 :
  num_3_point_problems + num_4_point_problems = 30 := 
sorry

end total_num_problems_eq_30_l826_826874


namespace remainder_of_1999_pow_81_mod_7_eq_1_l826_826500

/-- 
  Prove the remainder R when 1999^81 is divided by 7 is equal to 1.
  Conditions:
  - number: 1999
  - divisor: 7
-/
theorem remainder_of_1999_pow_81_mod_7_eq_1 : (1999 ^ 81) % 7 = 1 := 
by 
  sorry

end remainder_of_1999_pow_81_mod_7_eq_1_l826_826500


namespace sum_of_elements_in_T_in_base3_l826_826731

def T : Set ℕ := {n | ∃ (d1 d2 d3 d4 d5 : ℕ), (1 ≤ d1 ∧ d1 ≤ 2) ∧ (0 ≤ d2 ∧ d2 ≤ 2) ∧ (0 ≤ d3 ∧ d3 ≤ 2) ∧ (0 ≤ d4 ∧ d4 ≤ 2) ∧ (0 ≤ d5 ∧ d5 ≤ 2) ∧ n = d1 * 3^4 + d2 * 3^3 + d3 * 3^2 + d4 * 3^1 + d5}

theorem sum_of_elements_in_T_in_base3 : (∑ x in T, x) = 2420200₃ := 
sorry

end sum_of_elements_in_T_in_base3_l826_826731


namespace best_fit_model_l826_826691

noncomputable def model1_R2 : ℝ := 0.05
noncomputable def model2_R2 : ℝ := 0.49
noncomputable def model3_R2 : ℝ := 0.89
noncomputable def model4_R2 : ℝ := 0.98

theorem best_fit_model :
  ∀ i, i ∈ {model1_R2, model2_R2, model3_R2} → model4_R2 > i :=
by
sorries

end best_fit_model_l826_826691


namespace cosine_of_angle_is_one_over_5sqrt13_lines_intersect_at_correct_values_l826_826930

noncomputable def cosine_of_angle_between_lines : ℝ := 
  let direction_vector1 := (4, -3)
  let direction_vector2 := (2, 3)
  let dot_product := direction_vector1.1 * direction_vector2.1 + direction_vector1.2 * direction_vector2.2
  let magnitude1 := Real.sqrt ((direction_vector1.1 ^ 2) + (direction_vector1.2 ^ 2))
  let magnitude2 := Real.sqrt ((direction_vector2.1 ^ 2) + (direction_vector2.2 ^ 2))
  Real.abs (dot_product / (magnitude1 * magnitude2))

theorem cosine_of_angle_is_one_over_5sqrt13 :
  cosine_of_angle_between_lines = (1 / (5 * Real.sqrt 13)) :=
  by
  sorry

structure Line (α : Type) [Field α] :=
  (point : Prod α α)
  (direction : Prod α α)

def line1 := Line.mk (1, 3) (4, -3)
def line2 := Line.mk (2, -1) (2, 3)

def intersecting_point (line1 line2 : Line ℝ) : Option (ℝ × ℝ) := 
  let t := 11 / 18
  let u := 13 / 18
  if (1 + 4 * t = 2 + 2 * u) ∧ (3 - 3 * t = -1 + 3 * u) then
    some ((1 + 4 * t), (3 - 3 * t))
  else
    none
  
theorem lines_intersect_at_correct_values :
  intersecting_point line1 line2 = some (11 / 18 * 4 + 1, 3 - 11 / 18 * 3) :=
  by
  sorry

end cosine_of_angle_is_one_over_5sqrt13_lines_intersect_at_correct_values_l826_826930


namespace tangent_segment_length_l826_826485

noncomputable def length_of_segment (x y : ℝ) : ℝ :=
  2 * sqrt(5^2 - x^2)

theorem tangent_segment_length
  (r1 r2 r3 : ℝ)
  (h1 : r1 = 3)
  (h2 : r2 = 4)
  (h3 : r3 = 5)
  (h4 : (14 * (r3 - x) = 10) → (x = 5/7)):
  length_of_segment (5/7) (2 * sqrt(25 - (5/7)^2)) = (20 * sqrt(3)) / 7 :=
sorry

end tangent_segment_length_l826_826485


namespace correct_option_l826_826504

theorem correct_option : 
  (-(2:ℤ))^3 ≠ -6 ∧ 
  (-(1:ℤ))^10 ≠ -10 ∧ 
  (-(1:ℚ)/3)^3 ≠ -1/9 ∧ 
  -(2:ℤ)^2 = -4 :=
by 
  sorry

end correct_option_l826_826504


namespace sum_of_first_five_smallest_arguments_l826_826045

noncomputable def P (x : ℂ) : ℂ := (finset.sum (finset.range 20) (λ n, x^n))^2 - x^19

theorem sum_of_first_five_smallest_arguments :
  let α := λ k : ℕ, ((k : ℕ) / 21 : ℚ) in
  α 1 + α 1 + α 2 + α 2 + α 3 = (79 : ℚ) / 399 :=
sorry

end sum_of_first_five_smallest_arguments_l826_826045


namespace employee_decrease_percent_l826_826381

variable (E : ℝ) (S : ℝ) (E_new : ℝ)

/-- Given that the total number of employees decreased by some percent 
    and the average salary per employee increased by 10%, while the 
    total combined salaries remained the same,
    prove that the percent decrease in the total number of employees is approximately 9.09%. -/
theorem employee_decrease_percent (h1 : E_new * (1.1 * S) = E * S) : 
  ((E - E_new) / E) * 100 ≈ 9.09 :=
by
  rw [mul_comm E_new (1.1 * S), ←h1] at h1
  have : E_new = E / 1.1 := by rw div_eq_mul_inv; linarith
  rw this
  field_simp
  linarith
  sorry

end employee_decrease_percent_l826_826381


namespace coeff_x29_in_expansion_l826_826192

theorem coeff_x29_in_expansion : 
  polynomial.coeff (expand_series (1 + x^5 + x^7 + x^9) 16) 29 = 65520 :=
sorry

end coeff_x29_in_expansion_l826_826192


namespace correct_statement_l826_826506

/-- Given the following statements:
 1. Seeing a rainbow after rain is a random event.
 2. To check the various equipment before a plane takes off, a random sampling survey should be conducted.
 3. When flipping a coin 20 times, it will definitely land heads up 10 times.
 4. The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B.

 Prove that the correct statement is: Seeing a rainbow after rain is a random event.
-/
theorem correct_statement : 
  let statement_A := "Seeing a rainbow after rain is a random event"
  let statement_B := "To check the various equipment before a plane takes off, a random sampling survey should be conducted"
  let statement_C := "When flipping a coin 20 times, it will definitely land heads up 10 times"
  let statement_D := "The average precipitation in two cities, A and B, over the past 5 years is 800 millimeters. The variances are 3.4 for city A and 4.3 for city B. The city with the most stable annual precipitation is city B"
  statement_A = "Seeing a rainbow after rain is a random event" := by
sorry

end correct_statement_l826_826506


namespace six_universal_appropriate_l826_826382

variable (Candidates : Type)

variable [DecidableEq Candidates]

-- Definitions
def appropriate_set (S : Finset Candidates) : Prop :=
  sorry -- Assume this definition to be provided

def perspective_set (S : Finset Candidates) : Prop :=
  ∃ c : Candidates, appropriate_set (S ∪ {c})

def universal (c : Candidates) : Prop :=
  ∀ S : Finset Candidates, S.card = 5 → perspective_set S → appropriate_set (S ∪ {c})

-- Statement to prove
theorem six_universal_appropriate {lineup : Finset Candidates} (h_card : lineup.card = 6)
  (h_universal : ∀ c ∈ lineup, universal c) : appropriate_set lineup :=
  sorry

end six_universal_appropriate_l826_826382


namespace negation_of_universal_l826_826988

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, x > 0 → x^3 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^3 ≤ 0) :=
by sorry

end negation_of_universal_l826_826988


namespace find_b_amount_l826_826842

theorem find_b_amount (A B : ℝ) (h1 : A + B = 100) (h2 : (3 / 10) * A = (1 / 5) * B) : B = 60 := 
by 
  sorry

end find_b_amount_l826_826842


namespace find_values_of_a_and_b_find_sqrt_of_a_plus_2b_l826_826999

   section
   -- Definitions of conditions
   variable (a b : ℝ)
   variable (h1 : sqrt a = 3 ∨ sqrt a = -3)
   variable (h2 : sqrt (a * b) = 2)

   theorem find_values_of_a_and_b :
     a = 9 ∧ b = 4 / 9 :=
   by 
     sorry

   theorem find_sqrt_of_a_plus_2b (ha : a = 9) (hb : b = 4 / 9) :
     sqrt (a + 2 * b) = sqrt 89 / 3 ∨ sqrt (a + 2 * b) = -sqrt 89 / 3 :=
   by 
     sorry
   end
   
end find_values_of_a_and_b_find_sqrt_of_a_plus_2b_l826_826999


namespace range_correct_variance_correct_l826_826550

variable (scores : List ℕ) := [9, 5, 8, 4, 6, 10]

def range_of_scores (scores : List ℕ) : ℕ :=
  List.maximum scores - List.minimum scores

def mean_of_scores (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance_of_scores (scores : List ℕ) : ℚ :=
  let mean := mean_of_scores scores
  (scores.map (λ score => (score - mean)^2)).sum / scores.length

theorem range_correct : range_of_scores scores = 6 := by
  sorry

theorem variance_correct : variance_of_scores scores = 14 / 3 := by
  sorry

end range_correct_variance_correct_l826_826550


namespace number_of_ways_to_select_values_l826_826826

theorem number_of_ways_to_select_values :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  ∃ A B C D ∈ S, A ≠ C ∧ B > D ∧ 
  ∀ (A B C D : ℕ), 
    A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧ A ≠ C ∧ B > D ∧ 
    (S.erase A).erase B =  S.erase C.erase D =
    1512 :=
sorry

end number_of_ways_to_select_values_l826_826826


namespace number_of_remaining_numbers_problem_solution_l826_826401

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826401


namespace alpha_value_l826_826606

theorem alpha_value (alpha : ℝ) (h1 : cos alpha = -1 / 6) (h2 : 0 < alpha ∧ alpha < pi) :
  alpha = pi - real.arccos (1 / 6) :=
sorry

end alpha_value_l826_826606


namespace problem1_problem2_problem3_problem4_l826_826183

theorem problem1 : (-20 + (-14) - (-18) - 13) = -29 := by
  sorry

theorem problem2 : (-6 * (-2) / (1 / 8)) = 96 := by
  sorry

theorem problem3 : (-24 * (-3 / 4 - 5 / 6 + 7 / 8)) = 17 := by
  sorry

theorem problem4 : (-1^4 - (1 - 0.5) * (1 / 3) * (-3)^2) = -5 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_l826_826183


namespace last_letter_of_312th_permutation_l826_826443

-- Define the alphabet used in the word.
def alphabet : List Char := ['A', 'H', 'S', 'M', 'E', 'B']

-- Define the total number of permutations.
def num_permutations : Nat := (List.permutations alphabet).length

-- The main statement to prove
theorem last_letter_of_312th_permutation :
  (List.getNth (List.permutations alphabet) (312 - 1)).getLast == 'S' :=
by
  sorry

end last_letter_of_312th_permutation_l826_826443


namespace max_mean_BC_l826_826120

theorem max_mean_BC (A_n B_n C_n A_total_weight B_total_weight C_total_weight : ℕ)
    (hA_mean : A_total_weight = 45 * A_n)
    (hB_mean : B_total_weight = 55 * B_n)
    (hAB_mean : (A_total_weight + B_total_weight) / (A_n + B_n) = 48)
    (hAC_mean : (A_total_weight + C_total_weight) / (A_n + C_n) = 50) :
    ∃ m : ℤ, m = 66 := by
  sorry

end max_mean_BC_l826_826120


namespace remaining_numbers_count_l826_826418

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826418


namespace two_digit_square_difference_l826_826928

theorem two_digit_square_difference :
  let is_square (n : ℕ) := ∃ k : ℕ, k * k = n in
  ∃ n : ℕ, (10 ≤ 9 * n ∧ 9 * n ≤ 99) ∧ (is_square (9 * n)) ∧
  finset.card ((finset.Ico 0 10).filter (λ b, ∃ a : ℕ, (finset.Ico 1 10).mem a ∧ 9 * (a - b) = 9 * n)) = 6 :=
by sorry

end two_digit_square_difference_l826_826928


namespace sum_f_l826_826751

-- Define the function f
def f (x : ℝ) : ℝ := 2^x / (2^x + real.sqrt 2)

-- Define the property of the function f
lemma f_property (x : ℝ) : f(x) + f(1 - x) = 1 :=
by
  have h1 : 2^x / (2^x + real.sqrt 2) + 2^(1 - x) / (2^(1 - x) + real.sqrt 2) = 
    (2^x * (2^(1 - x) + real.sqrt 2) + 2^(1 - x) * (2^x + real.sqrt 2)) / ((2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2)),
  calc
  2^x / (2^x + real.sqrt 2) + 2^(1 - x) / (2^(1 - x) + real.sqrt 2)
    ... = 2^x / (2^x + real.sqrt 2) + 2^(1 - x) / (2^(1 - x) + real.sqrt 2) : by simp
  ... = (2^x / (2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2) + 
         2^(1 - x) / (2^(1 - x) + real.sqrt 2) * (2^x + real.sqrt 2)) /
        ((2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2)) : by simp
  ... = ((2^x * (2^(1 - x) + real.sqrt 2)) + (2^(1 - x) * (2^x + real.sqrt 2))) /
        ((2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2)) : by simp
  ... = (2 + 2^x * real.sqrt 2 + 2^x * 2^(1 - x) + 2^(1 - x) * real.sqrt 2) / ((2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2)),
  have h2 : 2^x * 2^(1 - x) = 2^1 := by sorry,
  have h3 : 2 + real.sqrt 2 * (2^x + 2^(1 - x)) + 2^1 = 2 + real.sqrt 2 * (2^x + 2^(1 - x)) + 2 := by sorry,
  calc
  (2 + real.sqrt 2 * (2^x + 2^(1 - x)) + 2) / ((2^x + real.sqrt 2) * (2^(1 - x) + real.sqrt 2))
    ... = 1 : by sorry

theorem sum_f : (∑ k in finset.range 2017, f (k + 1) / 2018) = 2017 / 2 :=
by
  sorry

end sum_f_l826_826751


namespace a_2009_value_l826_826611

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
∀ n, a n + a (n + 1) + a (n + 2) = 7

theorem a_2009_value (a : ℕ → ℤ) (h_seq : sequence a) (h_a3 : a 3 = 5) (h_a5 : a 5 = 8) : 
  a 2009 = 8 :=
by
  sorry

end a_2009_value_l826_826611


namespace num_values_passing_through_vertex_l826_826223

-- Define the parabola and line
def parabola (a : ℝ) : ℝ → ℝ := λ x, x^2 + a^2
def line (a : ℝ) : ℝ → ℝ := λ x, 2 * x + a

-- Define the vertex condition 
def passes_through_vertex (a : ℝ) : Prop :=
  parabola a 0 = line a 0

-- Prove there are exactly 2 values of a that satisfy the condition
theorem num_values_passing_through_vertex : 
  {a : ℝ | passes_through_vertex a}.finite ∧ 
  {a : ℝ | passes_through_vertex a}.toFinset.card = 2 := 
sorry

end num_values_passing_through_vertex_l826_826223


namespace discount_on_shoes_l826_826202

theorem discount_on_shoes (x : ℝ) :
  let shoe_price := 200
  let shirt_price := 80
  let total_spent := 285
  let total_shirt_price := 2 * shirt_price
  let initial_total := shoe_price + total_shirt_price
  let disc_shoe_price := shoe_price - (shoe_price * x / 100)
  let pre_final_total := disc_shoe_price + total_shirt_price
  let final_total := pre_final_total * (1 - 0.05)
  final_total = total_spent → x = 30 :=
by
  intros shoe_price shirt_price total_spent total_shirt_price initial_total disc_shoe_price pre_final_total final_total h
  dsimp [shoe_price, shirt_price, total_spent, total_shirt_price, initial_total, disc_shoe_price, pre_final_total, final_total] at h
  -- Here, we would normally continue the proof, but we'll insert 'sorry' for now as instructed.
  sorry

end discount_on_shoes_l826_826202


namespace hexagon_angle_U_l826_826696

theorem hexagon_angle_U 
  (F I U G E R : ℝ)
  (h1 : F = I) 
  (h2 : I = U)
  (h3 : G + E = 180)
  (h4 : R + U = 180)
  (h5 : F + I + G + U + R + E = 720) :
  U = 120 := by
  sorry

end hexagon_angle_U_l826_826696


namespace geometric_representation_of_inequalities_l826_826867

theorem geometric_representation_of_inequalities :
  (∀ x y : ℝ, |x| + |y| ≤ sqrt(2 * (x^2 + y^2)) ∧ sqrt(2 * (x^2 + y^2)) ≤ 2 * max (|x|) (|y|)) →
  "II" =
  "Represents the inequalities |x| + |y| ≤ sqrt(2 * (x^2 + y^2)) and sqrt(2 * (x^2 + y^2)) ≤ 2 * max (|x|) (|y|) geometrically" :=
by
  sorry

end geometric_representation_of_inequalities_l826_826867


namespace correct_statements_count_l826_826038

open Nat

theorem correct_statements_count (a b : ℕ) (h_gcd : gcd a b = 20) (h_lcm : lcm a b = 100) :
  let statements := [a * b = 2000, gcd (10 * a) (10 * b) = 2000, lcm (10 * a) (10 * b) = 1000, (10 * a) * (10 * b) = 200000] in
  (statements.filter id).length = 3 :=
by {
  sorry
}

end correct_statements_count_l826_826038


namespace part1_distance_pq_parallel_x_part2_max_distance_unbounded_l826_826176

open Real

noncomputable def distance_from_origin_to_pq (a : ℝ) : ℝ :=
  a^2

theorem part1_distance_pq_parallel_x (a : ℝ) : 
  distance_from_origin_to_pq a = a^2 :=
by
  unfold distance_from_origin_to_pq
  sorry

theorem part2_max_distance_unbounded : ¬(∃ M : ℝ, ∀ a : ℝ, distance_from_origin_to_pq a ≤ M) :=
by
  intro h
  obtain ⟨M, hM⟩ := h
  set a := M + 1 with haa
  specialize hM a
  have : distance_from_origin_to_pq a = a^2 := by
    unfold distance_from_origin_to_pq
  rw this at hM
  linarith
  sorry

end part1_distance_pq_parallel_x_part2_max_distance_unbounded_l826_826176


namespace orthogonal_pairs_zero_l826_826959

open Matrix

theorem orthogonal_pairs_zero : 
  ¬ ∃ (a d : ℝ), (fun M : Matrix (Fin 2) (Fin 2) ℝ => 
    (Mᵀ ⬝ M = (1 : Matrix (Fin 2) (Fin 2) ℝ)) ∧ 
    M = ![![a, 4], ![-9, d]]) :=
by 
  intro h 
  rcases h with ⟨a, d, orthogonal, matrix_def⟩
  rw matrix_def at orthogonal
  have eq1 : a * a + 16 = 1 := by sorry
  have eq2 : 81 + d * d = 1 := by sorry
  have eq3 : -9 * a + 4 * d = 0 := by sorry
  have h1 : ¬ ∃ a : ℝ, a * a = -15 := by
    intro h
    rcases h with ⟨a, eq⟩
    linarith
  contradiction

end orthogonal_pairs_zero_l826_826959


namespace recommended_cups_water_l826_826076

variable (currentIntake : ℕ)
variable (increasePercentage : ℕ)

def recommendedIntake : ℕ := 
  currentIntake + (increasePercentage * currentIntake) / 100

theorem recommended_cups_water (h1 : currentIntake = 15) 
                               (h2 : increasePercentage = 40) : 
  recommendedIntake currentIntake increasePercentage = 21 := 
by 
  rw [h1, h2]
  have h3 : (40 * 15) / 100 = 6 := by norm_num
  rw [h3]
  norm_num
  sorry

end recommended_cups_water_l826_826076


namespace perimeter_ABFCDE_l826_826934

theorem perimeter_ABFCDE 
  (ABCD_perimeter : ℝ)
  (ABCD : ℝ)
  (triangle_BFC : ℝ -> ℝ)
  (translate_BFC : ℝ -> ℝ)
  (ABFCDE : ℝ -> ℝ -> ℝ)
  (h1 : ABCD_perimeter = 40)
  (h2 : ABCD = ABCD_perimeter / 4)
  (h3 : triangle_BFC ABCD = 10 * Real.sqrt 2)
  (h4 : translate_BFC (10 * Real.sqrt 2) = 10 * Real.sqrt 2)
  (h5 : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2)
  : ABFCDE ABCD (10 * Real.sqrt 2) = 40 + 20 * Real.sqrt 2 := 
by 
  sorry

end perimeter_ABFCDE_l826_826934


namespace friend_owns_10_bikes_l826_826681

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l826_826681


namespace incorrect_statement_l826_826643

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 - 4 * x - 2 * x^2

-- Statement that needs to be proven incorrect
theorem incorrect_statement : ¬ (∀ x ∈ Ico 1 2, f x ≤ -3 ∧ ∃ y ∈ Ico 1 2, f y = -13) := 
sorry

end incorrect_statement_l826_826643


namespace product_of_positive_integer_solutions_l826_826961

theorem product_of_positive_integer_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (n^2 - 47 * n + 660 = p) → False :=
by
  -- Placeholder for proof, based on the problem conditions.
  sorry

end product_of_positive_integer_solutions_l826_826961


namespace point_on_transformed_plane_l826_826848

noncomputable def transformation (x y z : ℝ) : ℝ :=
  4 * x - 3 * y + 5 * z - 5

theorem point_on_transformed_plane : 
  let A := (1 / 4, 1 / 3, 1 : ℝ) in
  transformation A.1 A.2 A.2 = 0 :=
by
  let A : ℝ × ℝ × ℝ := (1 / 4, 1 / 3, 1)
  have h1 : transformation A.1 A.2 A.2 = 4 * (1 / 4) - 3 * (1 / 3) + 5 * 1 - 5 := rfl
  calc 
    4 * (1 / 4) - 3 * (1 / 3) + 5 * 1 - 5 = 1 - 1 + 5 - 5 : by ring
    ... = 0 : by rfl

end point_on_transformed_plane_l826_826848


namespace find_expression_find_max_value_l826_826983

-- Conditions
variables {a b : ℝ} (h₀ : a ≠ 0)
def f (x : ℝ) : ℝ := a * x^2 + b * x

-- First proof: Finding the expression of f(x)
theorem find_expression (h₁ : ∀ x : ℝ, f(x - 1) = f(3 - x))
                        (h₂ : ∀ x : ℝ, f(x) = 2 * x → (∃ x₀, f(x₀) = 2 * x₀ ∧ f(x₀) = 2 * x₀)) : 
  f = λ x, -x^2 + 2x :=
begin
  sorry, -- Proof not required
end

-- Second proof: Finding the maximum value of f(x) on the interval [0,t]
theorem find_max_value {t : ℝ} (ht₀ : 0 ≤ t) :
  (λ x, -x^2 + 2 * x).max_on_interval (0, t) = 
    if t > 1 
    then 1 
    else -t^2 + 2 * t :=
begin
  sorry, -- Proof not required
end

end find_expression_find_max_value_l826_826983


namespace order_tan_values_l826_826610

-- Define the values of a, b, and c according to the given problem conditions
def a : ℝ := Real.tan 1
def b : ℝ := Real.tan 2
def c : ℝ := Real.tan 3

-- The goal is to prove the inequality b < c < a
theorem order_tan_values : b < c ∧ c < a := 
by
  -- The detailed proof steps are omitted
  sorry

end order_tan_values_l826_826610


namespace solution_inequality_l826_826818

open Real

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1)

-- State the theorem for the given proof problem
theorem solution_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | x > 3 ∨ x < -1} :=
by
  sorry

end solution_inequality_l826_826818


namespace solve_complex_problem_l826_826974

noncomputable def complex_problem : Prop :=
  ∃ (a b : ℝ), (a + b * complex.i) ≠ 0 ∧ 
               (1 + 2 * complex.i) / (a + b * complex.i) = (1 - complex.i) ∧
               complex.abs (a + b * complex.i) = (real.sqrt 10) / 2

theorem solve_complex_problem : complex_problem :=
sorry

end solve_complex_problem_l826_826974


namespace average_difference_l826_826445

-- Definitions for the conditions
def set1 : List ℕ := [20, 40, 60]
def set2 : List ℕ := [10, 60, 35]

-- Function to compute the average of a list of numbers
def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

-- The main theorem to prove the difference between the averages is 5
theorem average_difference : average set1 - average set2 = 5 := by
  sorry

end average_difference_l826_826445


namespace point_outside_circle_range_l826_826998

theorem point_outside_circle_range (m : ℝ) : 
  (0 < m ∧ m < 1/4) ∨ (1 < m) ↔ (1 + 1 + 4*m - 2 + 5*m > 0) :=
begin
  sorry
end

end point_outside_circle_range_l826_826998


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_l826_826362

theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ) (h_d_nonzero : d ≠ 0)
  (h_arith : ∀ n, a_n = a_n 0 + n * d)
  (h_S9 : S 9 = 90)
  (h_geom : ∃ (a1 a2 a4 : ℕ), a2^2 = a1 * a4)
  (h_common_diff : d = a_n 1 - a_n 0)
  : ∀ n, a_n = 2 * n  := 
sorry

theorem sum_of_first_n_terms
  (b_n : ℕ → ℕ)
  (T : ℕ → ℕ)
  (a_n : ℕ → ℕ) 
  (h_b_def : ∀ n, b_n = 1 / (a_n n * a_n (n+1)))
  (h_a_form : ∀ n, a_n = 2 * n)
  : ∀ n, T n = n / (4 * n + 4) :=
sorry

end arithmetic_sequence_general_term_sum_of_first_n_terms_l826_826362


namespace tan_periodic_example_l826_826181

theorem tan_periodic_example : Real.tan (13 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_periodic_example_l826_826181


namespace arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l826_826617

open Nat

variable (a : ℕ → ℝ)
variable (c : ℕ → ℝ)
variable (k b : ℝ)

-- Condition 1: sequence definition
def sequence_condition := ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + n + 1

-- Condition 2: initial value
def initial_value := a 1 = -1

-- Condition 3: c_n definition
def geometric_sequence_condition := ∀ n : ℕ, 0 < n → c (n + 1) / c n = 2

-- Problem 1: Arithmetic sequence parameters
theorem arith_sequence_parameters (h1 : sequence_condition a) (h2 : initial_value a) : a 1 = -3 ∧ 2 * (a 1 + 2) - a 1 - 7 = -1 :=
by sorry

-- Problem 2: Cannot be a geometric sequence
theorem not_geo_sequence (h1 : sequence_condition a) (h2 : initial_value a) : ¬ (∃ q, ∀ n : ℕ, 0 < n → a n * q = a (n + 1)) :=
by sorry

-- Problem 3: c_n is a geometric sequence and general term for a_n
theorem geo_sequence_and_gen_term (h1 : sequence_condition a) (h2 : initial_value a) 
    (h3 : ∀ n : ℕ, 0 < n → c n = a n + k * n + b)
    (hk : k = 1) (hb : b = 2) : sequence_condition a ∧ initial_value a :=
by sorry

end arith_sequence_parameters_not_geo_sequence_geo_sequence_and_gen_term_l826_826617


namespace mark_squares_no_adjacent_36_l826_826507

/--
Given a \(10 \times 2\) grid of unit squares, there are 36 ways to mark exactly nine squares such that no two marked squares are adjacent.
-/
theorem mark_squares_no_adjacent_36 (grid : fin 10 → fin 2 → bool) (H_adj : ∀ (i j : fin 10) (k l : fin 2), |i - j| + |k - l| = 1 → ¬(grid i k = tt ∧ grid j l = tt)) (H_marked : (∑ i j, if grid i j then 1 else 0) = 9) :
  ∃ (count : nat), count = 36 := 
sorry

end mark_squares_no_adjacent_36_l826_826507


namespace subset_gcd_property_exists_l826_826024

open Finset Nat

theorem subset_gcd_property_exists (S : Finset ℕ) (h : S ⊆ range 2008) (h_card : S.card = 27) :
  ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ gcd a b ∣ c := 
sorry

end subset_gcd_property_exists_l826_826024


namespace A_can_complete_work_in_12_days_l826_826528

-- Given: B can complete the work in 15 days, i.e., B's work rate is 1/15 work per day
-- A and B worked together for 6 days
-- After 6 days, B was replaced by C
-- C can complete the work in 50 days, i.e., C's work rate is 1/50 work per day
-- The work was completed in next 5 days

theorem A_can_complete_work_in_12_days :
  let W : ℝ := 1 in -- Let W be the total work
  let B_work_rate : ℝ := W / 15 in
  let C_work_rate : ℝ := W / 50 in
  let A_work_rate (A_days : ℝ) : ℝ := W / A_days in
  let total_work_done := 6 * (A_work_rate 12 + B_work_rate) + 5 * C_work_rate in
  total_work_done = W → A_work_rate 12 = W / 12 :=
by
  intros
  sorry

end A_can_complete_work_in_12_days_l826_826528


namespace collinearity_of_DE_F_l826_826569

noncomputable def points_collinear (A B C P : Point) (D E F : Point) : Prop :=
  let P_to_D := perpendicular_from P to (B - C);
  let P_to_E := perpendicular_from P to (C - A);
  let P_to_F := perpendicular_from P to (A - B);
  is_perpendicular P P_to_D D ∧
  is_perpendicular P P_to_E E ∧
  is_perpendicular P P_to_F F ∧
  (collinear D E F)

theorem collinearity_of_DE_F
  (A B C : Point)
  (P : Point)
  (D E F : Point)
  (h₁ : ∃ (X Y Z : Line), X = perpendicular_from P to (B - C) ∧ Y = perpendicular_from P to (C - A) ∧ Z = perpendicular_from P to (A - B))
  (h₂ : ∃ (d e f : Point), d = intersection X (B - C) ∧ e = intersection Y (C - A) ∧ f = intersection Z (A - B))
  (h₃ : ∧ (is_perpendicular P X D) ∧ (is_perpendicular P Y E) ∧ (is_perpendicular P Z F)) : collinear D E F :=
sorry

end collinearity_of_DE_F_l826_826569


namespace find_k_l826_826086

theorem find_k 
    (P : ℝ × ℝ) (S : ℝ × ℝ) (Q : ℝ)
    (P_on_larger_circle : P = (5, 12))
    (S_on_smaller_circle : ∃ k : ℝ, S = (0, k))
    (QR : Q = 4) :
    S = (0, 9) := by
  -- Since Q is used to refer to the radius difference, we know QR means Q is the length 4
  let OP := real.sqrt (5^2 + 12^2)
  have origin_to_P : OP = 13 := by norm_num
  let OR := OP
  -- Radius of the smaller circle
  let OQ := OR - Q
  have OQ_value : OQ = 9 := by norm_num
  -- Since "S" is on the smaller circle, its radius should be 9 which implies S = (0, 9)
  cases S_on_smaller_circle with k hk
  rw hk
  exact congr_arg (prod.mk 0) (eq.symm OQ_value)

end find_k_l826_826086


namespace find_n_mod_31_l826_826830

theorem find_n_mod_31 : ∃ (n : ℕ), 0 ≤ n ∧ n < 31 ∧ 49325 % 31 = n := by
  use 2
  apply And.intro
  { exact Nat.zero_le 2 }
  apply And.intro
  { exact Nat.lt_of_lt_of_le (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ Nat.zero_lt_three))) (by decide) }
  { exact Nat.mod_eq_of_lt (Nat.div_lt_self (show 49325 > 0 from by decide) (show 31 > 1 from by decide)) }
  sorry

end find_n_mod_31_l826_826830


namespace speed_of_train_in_km_per_hr_l826_826890

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l826_826890


namespace range_g_l826_826948

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + 2 * Real.arcsin x

theorem range_g : 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -Real.pi / 2 ≤ g x ∧ g x ≤ 3 * Real.pi / 2) := 
by {
  sorry
}

end range_g_l826_826948


namespace all_points_lie_on_circle_l826_826224

theorem all_points_lie_on_circle {s : ℝ} :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := (2 * s) / (s^2 + 1)
  x^2 + y^2 = 1 :=
by
  sorry

end all_points_lie_on_circle_l826_826224


namespace general_formula_b_l826_826737

noncomputable def a (n : ℕ) : ℝ :=
  match n with
  | 0 => 2
  | k + 1 => 2 / (a k + 1)

def b (n : ℕ) : ℝ := abs ((a n + 2) / (a n - 1))

theorem general_formula_b (n : ℕ) : b (n + 1) = 2 ^ (n + 1) :=
by
  sorry

end general_formula_b_l826_826737


namespace weekly_spending_l826_826819

-- Definitions based on the conditions outlined in the original problem
def weekly_allowance : ℝ := 50
def hours_per_week : ℕ := 30
def hourly_wage : ℝ := 9
def weeks_per_year : ℕ := 52
def first_year_allowance : ℝ := weekly_allowance * weeks_per_year
def second_year_earnings : ℝ := (hourly_wage * hours_per_week) * weeks_per_year
def total_car_cost : ℝ := 15000
def additional_needed : ℝ := 2000
def total_savings : ℝ := first_year_allowance + second_year_earnings

-- The amount Thomas needs over what he has saved
def total_needed : ℝ := total_savings + additional_needed
def amount_spent_on_self : ℝ := total_needed - total_car_cost
def total_weeks : ℕ := 2 * weeks_per_year

theorem weekly_spending :
  amount_spent_on_self / total_weeks = 35 := by
  sorry

end weekly_spending_l826_826819


namespace log_inequality_solution_set_l826_826649

theorem log_inequality_solution_set (x : ℝ) : log 2 (x - 3) < 0 ↔ 3 < x ∧ x < 4 :=
by sorry

end log_inequality_solution_set_l826_826649


namespace remaining_numbers_after_erasure_l826_826387

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826387


namespace weighted_average_AC_l826_826062

theorem weighted_average_AC (avgA avgB avgC wA wB wC total_weight: ℝ)
  (h_avgA : avgA = 7.3)
  (h_avgB : avgB = 7.6) 
  (h_avgC : avgC = 7.2)
  (h_wA : wA = 3)
  (h_wB : wB = 4)
  (h_wC : wC = 1)
  (h_total_weight : total_weight = 5) :
  ((avgA * wA + avgC * wC) / total_weight) = 5.82 :=
by
  sorry

end weighted_average_AC_l826_826062


namespace sqrt_fourth_power_eq_l826_826204

theorem sqrt_fourth_power_eq:
  (sqrt (2 + sqrt (2 + sqrt 2))) ^ 4 = 6 + 4 * sqrt (2 + sqrt 2) + sqrt 2 :=
by
  -- the proof would go here
  sorry

end sqrt_fourth_power_eq_l826_826204


namespace imag_part_z_is_2_l826_826039

def z : ℂ := (2 / (1 - (I : ℂ))) + 2 + (I : ℂ)

theorem imag_part_z_is_2 : z.im = 2 :=
sorry

end imag_part_z_is_2_l826_826039


namespace crushing_load_value_l826_826201

-- Define the given formula and values
def T : ℕ := 3
def H : ℕ := 9
def K : ℕ := 2

-- Given formula for L
def L (T H K : ℕ) : ℚ := 50 * T^5 / (K * H^3)

-- Prove that L = 8 + 1/3
theorem crushing_load_value :
  L T H K = 8 + 1 / 3 :=
by
  sorry

end crushing_load_value_l826_826201


namespace number_of_remaining_numbers_problem_solution_l826_826400

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826400


namespace hypotenuse_length_similarity_CH_functions_f_varphi_l826_826856

/-- 
Given a right triangle \( \triangle ABC \) with legs \( AB = a \) and \( BC = 2a \), 
prove that the hypotenuse \( AC = a\sqrt{5} \).
-/
theorem hypotenuse_length (a : ℝ) :  
  let AB := a, BC := 2 * a 
  in AC = real.sqrt (a^2 + (2*a)^2) := 
by
  let AB := a
  let BC := 2 * a 
  have AC := real.sqrt (a^2 + (2*a)^2)
  exact sorry

/--
On the hypotenuse \( AC \), mark a segment \( CD \) such that \( CD = \sqrt{5} \). 
From point \( D \), draw a perpendicular \( DH \). Using the similarity of triangles \( DHC \) 
and \( ABC \), prove that \( CH = 2 \).
-/
theorem similarity_CH (a : ℝ) : 
  let AC := a * real.sqrt 5, CD := real.sqrt 5, BC := 2 * a
  in CH := 2 :=
by
  let AC := a * real.sqrt 5 
  let CD := real.sqrt 5
  let BC := 2 * a 
  have CH := 2 * (1 : ℝ)
  exact sorry

/-- 
Given the equations:
\[
\begin{cases}
f(t - 1) + 2 \varphi(2 t + 1) = \frac{t - 5}{2} \\
f(x - 1) + \varphi(2 x + 1) = 2x
\end{cases}
\]
prove: 
\[ 
f(x) = \frac{7x + 12}{2} \quad \text{and} \quad \varphi(x) = \frac{-3x + 7}{4}
\]
-/
theorem functions_f_varphi (t x : ℝ) :
  let x := (t - 3) / 2, 
  f t := (7 * t + 5) / 2, 
  φ t := (-3 * t - 5) / 2,
  f x := (7 * x + 12) / 2, 
  φ x := (-3 * x + 7) / 4  := 
by
  let x := (t - 3) / 2 
  let f := (7 * t + 5) / 2 
  let φ := (-3 * t -5) / 2  
  have f := (7 * x + 12) / 2 
  have φ := (-3 * x + 7) / 4
  exact sorry

end hypotenuse_length_similarity_CH_functions_f_varphi_l826_826856


namespace cube_volume_from_surface_area_l826_826834

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 150) : (S / 6) ^ (3 / 2) = 125 := by
  sorry

end cube_volume_from_surface_area_l826_826834


namespace closest_integer_perimeter_is_21_l826_826559

-- Define the necessary entities in the problem
def number_of_triangles := 16
def segment_length : ℝ := 1
def hypotenuse_length : ℝ := Real.sqrt 17

-- Define the total_perimeter calculation
def total_perimeter : ℝ :=
  number_of_triangles * segment_length + hypotenuse_length

-- Define the closest integer to the perimeter
def closest_integer (x : ℝ) : ℤ := Int.round x

-- The Lean theorem statement
theorem closest_integer_perimeter_is_21 : closest_integer total_perimeter = 21 :=
by
  sorry

end closest_integer_perimeter_is_21_l826_826559


namespace cosine_angle_and_point_distance_l826_826918

noncomputable def pyramid_conditions :=
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let B := (sqrt 3, 0, 0)
  let C := (sqrt 3, 1, 0)
  let D := (0, 1, 0)
  let P := (0, 0, 2)
  let E := (0, 1/2, 1)
  (A, B, C, D, P, E)

theorem cosine_angle_and_point_distance :
  let (A, B, C, D, P, E) := pyramid_conditions
  has_coordinates A = (0, 0, 0) ∧
  has_coordinates B = (sqrt 3, 0, 0) ∧
  has_coordinates C = (sqrt 3, 1, 0) ∧
  has_coordinates D = (0, 1, 0) ∧
  has_coordinates P = (0, 0, 2) ∧
  has_coordinates E = (0, 1/2, 1) →
  (vector_angle (sqrt 3, 1, 0) (sqrt 3, 0, -2) = 3*sqrt(7)/14) ∧
  let N := (sqrt 3 / 6, 0, 1)
  (distance_to N ≠ 0) ∧
  (distance_to_AB N = 1) ∧
  (distance_to_AP N = sqrt 3 / 6) :=
by
  sorry

end cosine_angle_and_point_distance_l826_826918


namespace moles_of_naoh_combined_number_of_moles_of_naoh_combined_l826_826958

-- Define the reaction equation and given conditions
def reaction_equation := "2 NaOH + Cl₂ → NaClO + NaCl + H₂O"

-- Given conditions
def moles_chlorine : ℕ := 2
def moles_water_produced : ℕ := 2
def moles_naoh_needed_for_one_mole_water : ℕ := 2

-- Stoichiometric relationship from the reaction equation
def moles_naoh_per_mole_water : ℕ := 2

-- Theorem to prove the number of moles of NaOH combined
theorem moles_of_naoh_combined (moles_water_produced : ℕ)
  (moles_naoh_per_mole_water : ℕ) : ℕ :=
  moles_water_produced * moles_naoh_per_mole_water

-- Statement of the theorem
theorem number_of_moles_of_naoh_combined : moles_of_naoh_combined 2 2 = 4 :=
by sorry

end moles_of_naoh_combined_number_of_moles_of_naoh_combined_l826_826958


namespace no_primes_in_sequence_l826_826927

noncomputable def Q : ℕ := Nat.primes.filter (λ p, p ≤ 67).prod

def isComposite (n : ℕ) : Prop := ∃ a b, a > 1 ∧ b > 1 ∧ a * b = n

theorem no_primes_in_sequence : ∀ m, 2 ≤ m ∧ m ≤ 65 → ∃ a b, a > 1 ∧ b > 1 ∧ a * b = Q + m :=
by {
  intro m,
  intro h,
  sorry
}

end no_primes_in_sequence_l826_826927


namespace sqrt_expression_value_l826_826195

noncomputable def cos_squared (θ : ℝ) : ℝ := (Real.cos θ) ^ 2

theorem sqrt_expression_value :
  sqrt ((3 - cos_squared (π / 9)) * (3 - cos_squared (2 * π / 9)) * (3 - cos_squared (4 * π / 9))) = 3 * sqrt 2 :=
by
  sorry

end sqrt_expression_value_l826_826195


namespace short_haired_girls_l826_826067

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l826_826067


namespace ant_routes_on_cube_l826_826168

theorem ant_routes_on_cube :
  let distance (p1 p2 : ℕ × ℕ × ℕ) : ℝ := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2
  let adjacent (p1 p2 : ℕ × ℕ × ℕ) : Bool := ((p1.1 = p2.1 ∧ p1.2 = p2.2 ∧ (p1.3 - p2.3).natAbs = 1) ∨
                                                (p1.1 = p2.1 ∧ (p1.2 - p2.2).natAbs = 1 ∧ p1.3 = p2.3) ∨
                                                ((p1.1 - p2.1).natAbs = 1 ∧ p1.2 = p2.2 ∧ p1.3 = p2.3))
  let valid_moves (route : List (ℕ × ℕ × ℕ)) : Bool :=
    route.length = 8 ∧ distance (route.head! route.getLast!) = 3 ∧ -- sqrt(3)^2 = 3
    ∀ i < route.length - 1, adjacent (route.nth! i) (route.nth! (i + 1))

  (Set.toFinset {route : List (ℕ × ℕ × ℕ) | valid_moves route}).card = 546 := sorry

end ant_routes_on_cube_l826_826168


namespace eddie_weekly_earnings_l826_826944

theorem eddie_weekly_earnings :
  let mon_hours := 2.5
  let tue_hours := 7 / 6
  let wed_hours := 7 / 4
  let sat_hours := 3 / 4
  let weekday_rate := 4
  let saturday_rate := 6
  let mon_earnings := mon_hours * weekday_rate
  let tue_earnings := tue_hours * weekday_rate
  let wed_earnings := wed_hours * weekday_rate
  let sat_earnings := sat_hours * saturday_rate
  let total_earnings := mon_earnings + tue_earnings + wed_earnings + sat_earnings
  total_earnings = 26.17 := by
  simp only
  norm_num
  sorry

end eddie_weekly_earnings_l826_826944


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l826_826108

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l826_826108


namespace prob_select_exactly_one_geo_one_igso_prob_select_at_least_one_igso_l826_826699

-- Given definitions
def num_geo : ℕ := 3
def num_igso : ℕ := 3
def total_pairs : ℕ := (num_geo + num_igso) * (num_geo + num_igso - 1) / 2

def event_A_pairs : set (ℕ × ℕ) :=
{(a, b) | (a <= num_geo ∧ b > num_geo) ∨ (b <= num_geo ∧ a > num_geo)}

def prob_event_A : rational :=
(rat.of_int (set.size event_A_pairs)) / rat.of_int total_pairs

def event_B_pairs : set (ℕ × ℕ) :=
{(a, b) | a > num_geo ∨ b > num_geo}

def prob_event_B : rational :=
(rat.of_int (set.size event_B_pairs)) / rat.of_int total_pairs

-- Proof statements
theorem prob_select_exactly_one_geo_one_igso :
  prob_event_A = 3 / 5 := sorry

theorem prob_select_at_least_one_igso :
  prob_event_B = 4 / 5 := sorry

end prob_select_exactly_one_geo_one_igso_prob_select_at_least_one_igso_l826_826699


namespace new_rectangle_dimensions_l826_826539

theorem new_rectangle_dimensions (l w : ℕ) (h_l : l = 12) (h_w : w = 10) :
  ∃ l' w' : ℕ, l' = l ∧ w' = w / 2 ∧ l' = 12 ∧ w' = 5 :=
by
  sorry

end new_rectangle_dimensions_l826_826539


namespace platform_length_proof_l826_826858

noncomputable def train_length : ℝ := 480

noncomputable def speed_kmph : ℝ := 55

noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

noncomputable def crossing_time : ℝ := 71.99424046076314

noncomputable def total_distance_covered : ℝ := speed_mps * crossing_time

noncomputable def platform_length : ℝ := total_distance_covered - train_length

theorem platform_length_proof : platform_length = 620 := by
  sorry

end platform_length_proof_l826_826858


namespace pn_plus_1_is_cube_l826_826726

theorem pn_plus_1_is_cube (n p : ℕ) (h1 : n ∣ (p - 3)) (h2 : p ∣ (n + 1)^3 - 1) (hp : nat.prime p) (hp_gt_3 : p > 3) : 
  ∃ k : ℕ, pn + 1 = k^3 :=
sorry

end pn_plus_1_is_cube_l826_826726


namespace sum_eq_neg_20_div_3_l826_826363
-- Import the necessary libraries

-- The main theoretical statement
theorem sum_eq_neg_20_div_3
    (a b c d : ℝ)
    (h : a + 2 = b + 4 ∧ b + 4 = c + 6 ∧ c + 6 = d + 8 ∧ d + 8 = a + b + c + d + 10) :
    a + b + c + d = -20 / 3 :=
by
  sorry

end sum_eq_neg_20_div_3_l826_826363


namespace exists_language_spoken_by_at_least_three_l826_826816

noncomputable def smallestValue_n (k : ℕ) : ℕ :=
  2 * k + 3

theorem exists_language_spoken_by_at_least_three (k n : ℕ) (P : Fin n → Set ℕ) (K : ℕ → ℕ) :
  (n = smallestValue_n k) →
  (∀ i, (K i) ≤ k) →
  (∀ (x y z : Fin n), ∃ l, l ∈ P x ∧ l ∈ P y ∧ l ∈ P z ∨ l ∈ P y ∧ l ∈ P z ∨ l ∈ P z ∧ l ∈ P x ∨ l ∈ P x ∧ l ∈ P y) →
  ∃ l, ∃ (a b c : Fin n), l ∈ P a ∧ l ∈ P b ∧ l ∈ P c :=
by
  intros h1 h2 h3
  sorry

end exists_language_spoken_by_at_least_three_l826_826816


namespace maria_change_l826_826755

def cost_per_apple : ℝ := 0.75
def number_of_apples : ℕ := 5
def amount_paid : ℝ := 10.0
def total_cost := number_of_apples * cost_per_apple
def change_received := amount_paid - total_cost

theorem maria_change :
  change_received = 6.25 :=
sorry

end maria_change_l826_826755


namespace find_m_l826_826656

-- Define the vectors a and b
def veca (m : ℝ) : ℝ × ℝ := (m, 4)
def vecb (m : ℝ) : ℝ × ℝ := (m + 4, 1)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition that the dot product of a and b is zero
def are_perpendicular (m : ℝ) : Prop :=
  dot_product (veca m) (vecb m) = 0

-- The goal is to prove that if a and b are perpendicular, then m = -2
theorem find_m (m : ℝ) (h : are_perpendicular m) : m = -2 :=
by {
  -- Proof will be filled here
  sorry
}

end find_m_l826_826656


namespace intersection_of_line_and_circle_tangent_angle_constant_slope_MN_l826_826636

-- Statement of the Lean proof problem:
theorem intersection_of_line_and_circle (x0 y0 : ℝ) (hA_outside : x0 ^ 2 + y0 ^ 2 > 13) :
  (13:ℝ) / real.sqrt (x0 ^ 2 + y0 ^ 2) < real.sqrt 13 :=
sorry

theorem tangent_angle {x0 y0 : ℝ} (h_circle_eq : x0 ^ 2 + y0 ^ 2 = 13) (h_x0 : x0 = 2)
  (h_y0_pos : y0 > 0) (kAM : ℝ) (h_slope : kAM = 3 / 2) :
  real.tan (real.arctan (3 / 2) - real.arctan (-3 / 2)) = 12 / 5 :=
sorry

theorem constant_slope_MN {x0 y0 : ℝ} (h_circle_eq : x0 ^ 2 + y0 ^ 2 = 13) (h_x0 : x0 = 2)
  (h_y0_pos : y0 > 0) (k : ℝ) :
  let xM := (2 * k ^ 2 - 6 * k - 2) / (k ^ 2 + 1),
      xN := (2 * k ^ 2 + 6 * k - 2) / (k ^ 2 + 1)
  in (y0 - (k * x0 + 3 - 2 * k)) / (xM - xN) = 2 / 3 :=
sorry

end intersection_of_line_and_circle_tangent_angle_constant_slope_MN_l826_826636


namespace prove_f_2013_l826_826980

-- Defining the function f that satisfies the given conditions
variable (f : ℕ → ℕ)

-- Conditions provided in the problem
axiom cond1 : ∀ n, f (f n) + f n = 2 * n + 3
axiom cond2 : f 0 = 1
axiom cond3 : f 2014 = 2015

-- The statement to be proven
theorem prove_f_2013 : f 2013 = 2014 := sorry

end prove_f_2013_l826_826980


namespace solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l826_826791

-- Problem 1: Prove the solutions to x^2 = 2
theorem solve_quad_eq1 : ∃ x : ℝ, x^2 = 2 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by
  sorry

-- Problem 2: Prove the solutions to 4x^2 - 1 = 0
theorem solve_quad_eq2 : ∃ x : ℝ, 4 * x^2 - 1 = 0 ∧ (x = 1/2 ∨ x = -1/2) :=
by
  sorry

-- Problem 3: Prove the solutions to (x-1)^2 - 4 = 0
theorem solve_quad_eq3 : ∃ x : ℝ, (x - 1)^2 - 4 = 0 ∧ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 4: Prove the solutions to 12 * (3 - x)^2 - 48 = 0
theorem solve_quad_eq4 : ∃ x : ℝ, 12 * (3 - x)^2 - 48 = 0 ∧ (x = 1 ∨ x = 5) :=
by
  sorry

end solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l826_826791


namespace annual_depletion_rate_l826_826534

theorem annual_depletion_rate
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (depletion_rate : ℝ)
  (h_initial_value : initial_value = 40000)
  (h_final_value : final_value = 36100)
  (h_time : time = 2)
  (decay_eq : final_value = initial_value * (1 - depletion_rate)^time) :
  depletion_rate = 0.05 :=
by 
  sorry

end annual_depletion_rate_l826_826534


namespace function_relationship_and_comparison_l826_826249

theorem function_relationship_and_comparison :
  (∀ (x y : ℝ), y + 2 = 6 * x ↔ y + 2 = 3 * 2 * x) →
  ((6 * 1 - 2 = 4) ∧ (∀ (a b : ℝ), (-1, a) ∈ set_of (λ p : ℝ × ℝ, p.snd = 6 * p.fst - 2) →
    ((2, b) ∈ set_of (λ p : ℝ × ℝ, p.snd = 6 * p.fst - 2) → a < b))) :=
begin
  sorry
end

end function_relationship_and_comparison_l826_826249


namespace units_digit_of_square_ne_2_l826_826104

theorem units_digit_of_square_ne_2 (n : ℕ) : (n * n) % 10 ≠ 2 :=
sorry

end units_digit_of_square_ne_2_l826_826104


namespace partition_positive_integers_l826_826587

    noncomputable def pow2_factorization (n : ℕ) : ℕ := 
      if n = 0 then 0 else 
        nat.find_greatest (λ k, 2^k ∣ n) n

    theorem partition_positive_integers (a b : ℕ) (H_1 H_2 : set ℕ) 
        (h_partition : ∀ n, n > 0 → (n ∈ H_1 ∨ n ∈ H_2) ∧ (n ∈ H_1 ↔ n ∉ H_2))
        (a_not_diff_H1 : ∀ x y ∈ H_1, x - y ≠ a)
        (a_not_diff_H2 : ∀ x y ∈ H_2, x - y ≠ a)
        (b_not_diff_H1 : ∀ x y ∈ H_1, x - y ≠ b)
        (b_not_diff_H2 : ∀ x y ∈ H_2, x - y ≠ b) :
        pow2_factorization a = pow2_factorization b := by 
    sorry
    
end partition_positive_integers_l826_826587


namespace prod_sum_of_squares_l826_826242

namespace my_namespace

theorem prod_sum_of_squares (n : ℕ) (x : Fin n → ℕ) : 
  ∃ a b : ℤ, (∏ i in Finset.univ, (1 + (x i)^2)) = a^2 + b^2 :=
by
  sorry

end my_namespace

end prod_sum_of_squares_l826_826242


namespace problem_solution_l826_826044

-- Define the points of intersection
def num_distinct_points (C1 C2 : ℝ → ℝ → Prop) : ℕ := 
  let points := {(x : ℝ, y : ℝ) | C1 x y ∧ C2 x y}
  points.count

-- Define the two given curves
def curve1 (x y : ℝ) : Prop := 3 * x^2 + y^2 = 3
def curve2 (x y : ℝ) : Prop := x^2 + 3 * y^2 = 1

theorem problem_solution : num_distinct_points curve1 curve2 = 2 := 
  by
  sorry

end problem_solution_l826_826044


namespace determine_a_b_l826_826226

def points_collinear (A B C : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, B = (λ x y z, (x + t, y + t, z + t)) A ∧ C = (λ x y z, (x + 2*t, y + 2*t, z + 2*t)) A

theorem determine_a_b (a b : ℝ) (A B C : ℝ × ℝ × ℝ) 
  (hA : A = (2, a+1, b+1)) 
  (hB : B = (a+1, 3, b+1)) 
  (hC : C = (a+1, b+1, 4)) 
  (hcol : points_collinear A B C) :
  a = 1 ∧ b = 3 ∧ a + b = 4 :=
sorry

end determine_a_b_l826_826226


namespace quadratic_equation_with_rational_coeffs_l826_826584

/-
  Given the conditions:
  1. A root \(\sqrt{5} - 3\)
  2. Rational coefficients
  3. Quadratic term \(x^2\)

  Prove that the quadratic equation is \(x^2 + 6x - 4\).
-/
theorem quadratic_equation_with_rational_coeffs (a b c : ℚ) (root1 root2 : ℝ) 
  (h1 : root1 = sqrt 5 - 3)
  (h2 : root2 = - sqrt 5 - 3)
  (h3 : a = 1) 
  (h4 : b = - (root1 + root2))
  (h5 : c = root1 * root2) :
  ∃ (p : polynomial ℚ), p = polynomial.X ^ 2 + 6 * polynomial.X - 4 :=
begin
  sorry
end

end quadratic_equation_with_rational_coeffs_l826_826584


namespace total_students_in_college_l826_826686

-- Definitions for the conditions in the problem.
def classA_boys_girls_ratio : ℕ × ℕ := (5, 7)
def classA_girls : ℕ := 140

def classB_boys_girls_ratio : ℕ × ℕ := (3, 5)
def classB_total_students : ℕ := 280

-- Statement of the proof problem.
theorem total_students_in_college : 
  let classA_boys := (classA_boys_girls_ratio.1 * classA_girls) / classA_boys_girls_ratio.2 in
  let classA_total := classA_boys + classA_girls in
  let classB_parts := classB_boys_girls_ratio.1 + classB_boys_girls_ratio.2 in
  let classB_boys := (classB_boys_girls_ratio.1 * classB_total_students) / classB_parts in
  let classB_girls := (classB_boys_girls_ratio.2 * classB_total_students) / classB_parts in
  let college_total_students := classA_total + classB_total_students in
  college_total_students = 520 :=
by
  let classA_boys := (classA_boys_girls_ratio.1 * classA_girls) / classA_boys_girls_ratio.2
  let classA_total := classA_boys + classA_girls
  let classB_parts := classB_boys_girls_ratio.1 + classB_boys_girls_ratio.2
  let classB_boys := (classB_boys_girls_ratio.1 * classB_total_students) / classB_parts
  let classB_girls := (classB_boys_girls_ratio.2 * classB_total_students) / classB_parts
  let college_total_students := classA_total + classB_total_students
  exact calc
    college_total_students = 240 + 280 : sorry
    ... = 520 : rfl

end total_students_in_college_l826_826686


namespace leak_drain_time_l826_826538

theorem leak_drain_time :
  ∀ (P L : ℝ),
  P = 1/6 →
  P - L = 1/12 →
  (1/L) = 12 :=
by
  intros P L hP hPL
  sorry

end leak_drain_time_l826_826538


namespace hyperbola_eccentricity_hyperbola_focus_intersection_l826_826282

noncomputable def hyperbola_equation (a b x y : ℝ) : Prop :=
  (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def line_equation (m x y : ℝ) : Prop :=
  (y = x + m)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2) / a)

theorem hyperbola_eccentricity (m : ℝ) (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 2) :
  eccentricity a b = Real.sqrt 2 :=
by
  sorry

theorem hyperbola_focus_intersection (c b : ℝ) (hb : b^2 = 7) :
  ∃ a : ℝ, hyperbola_equation a b = (λ x y, (x^2 / 2 - y^2 / 7 = 1)) :=
by
  sorry

end hyperbola_eccentricity_hyperbola_focus_intersection_l826_826282


namespace function_characterization_l826_826351

open Nat

-- Define the domain as positive integers
def PositiveInteger := {n : ℕ // n > 0}

-- Declare the function f, taking two positive integers and returning a positive integer
def f (x y : PositiveInteger) : PositiveInteger := ⟨x.val + y.val - 1, by apply Nat.add_sub_succ_lt; exact x.property; exact y.property⟩

-- Define the constants C_t
def C (t : ℕ) (T : ℕ) : ℕ := if t ≤ T then T else 0

-- The main proof statement
theorem function_characterization (T : ℕ) (hT : T > 0)
  (f : ∀ (x y : PositiveInteger), PositiveInteger)
  (hf1 : ∀ n : ℕ, n > 0 →
    (Finset.card { p : PositiveInteger × PositiveInteger | p.1 ≠ 0 ∧ p.2 ≠ 0 ∧ f p.1 p.2 = ⟨n, _⟩ } = n))
  (hf2 : ∀ t : ℕ, t ≤ T →
    ∀ (k l : PositiveInteger), f ⟨k.val + t, sorry⟩ ⟨l.val + (T - t), sorry⟩.val - (f k l).val = C t T) :
  ∀ x y : PositiveInteger, f x y = ⟨x.val + y.val - 1, sorry⟩ :=
sorry

end function_characterization_l826_826351


namespace simplify_expression_with_negative_exponents_l826_826503

section 
variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

theorem simplify_expression_with_negative_exponents :
  (a + b)⁻² * (a⁻² + b⁻²) = (a^2 * b^2)⁻¹ * (a^2 + b^2) * (a + b)⁻² := 
by 
  sorry
end

end simplify_expression_with_negative_exponents_l826_826503


namespace equilateral_triangle_AM_eq_BM_CM_l826_826356
open Real

variables {A B C M : Point}
variables {circumcircle : Circle}
variables {s : ℝ}

-- Define an equilateral triangle
def is_equilateral (A B C : Point) : Prop := 
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Define the circumcircle of the triangle ABC
def is_circumcircle (circumcircle : Circle) (A B C : Point) : Prop := 
  circumcircle.contains A ∧ circumcircle.contains B ∧ circumcircle.contains C

-- Define a point M on the arc BC not passing through A
def on_arc_BC_not_through_A (circumcircle : Circle) (A B C M : Point) : Prop := 
  circumcircle.contains M ∧ ¬(M = A) ∧ arc_not_through (circumcircle.arc B C) A M

-- State the final theorem to be proved
theorem equilateral_triangle_AM_eq_BM_CM : 
  is_equilateral A B C →
  is_circumcircle circumcircle A B C →
  on_arc_BC_not_through_A circumcircle A B C M →
  dist A M = dist B M + dist C M :=
by sorry

end equilateral_triangle_AM_eq_BM_CM_l826_826356


namespace probability_of_9_out_of_10_l826_826997

open Real

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_9_out_of_10 (n k : ℕ) (p : ℝ) (h_n : n = 10) (h_k : k = 9) (h_p : p = 0.9) :
  binomial_probability n k p ≠ 0 :=
by
  sorry

end probability_of_9_out_of_10_l826_826997


namespace factorization_theorem_l826_826969

-- Define the polynomial p(x, y)
def p (x y k : ℝ) : ℝ := x^2 - 2*x*y + k*y^2 + 3*x - 5*y + 2

-- Define the condition for factorization into two linear factors
def can_be_factored (x y m n : ℝ) : Prop :=
  (p x y (m * n)) = ((x + m * y + 1) * (x + n * y + 2))

-- The main theorem proving that k = -3 is the value for factorizability
theorem factorization_theorem (k : ℝ) : (∃ m n : ℝ, can_be_factored x y m n) ↔ k = -3 := by sorry

end factorization_theorem_l826_826969


namespace Q_correct_l826_826956

def vector_projection_matrix (v : ℝ × ℝ × ℝ) : matrix (fin 3) (fin 3) ℝ :=
  let a := (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)
  λ i j, (vec v i * vec v j) / a

def vec (v : ℝ × ℝ × ℝ) : fin 3 → ℝ
| ⟨0, _⟩ => v.1
| ⟨1, _⟩ => v.2
| ⟨2, _⟩ => v.3

noncomputable def Q : matrix (fin 3) (fin 3) ℝ :=
  vector_projection_matrix (3, -1, 2)

theorem Q_correct : Q = 
  ![
    ![9/14, -3/14, 6/14],
    ![-3/14, 1/14, -2/14],
    ![6/14, -2/14, 4/14]
  ] :=
by sorry

end Q_correct_l826_826956


namespace distinct_solutions_of_abs_eq_l826_826441

theorem distinct_solutions_of_abs_eq (x : ℝ) : (|x - |2x + 3|| = 5) → x = 2 ∨ x = -8/3 :=
sorry

end distinct_solutions_of_abs_eq_l826_826441


namespace isosceles_triangle_BQC_l826_826748

open EuclideanGeometry

variables (A B C D P Q : Point)

-- Definitions of midpoint and perpendicularity
def isMidpoint (P M : Point) := P = midpoint P M
def isPerpendicular (L1 L2 : Line) := Perpendicular L1 L2

-- Given conditions
axiom rectangle_ABCD : Rectangle A B C D
axiom P_midpoint_AB : isMidpoint P AB
axiom Q_on_PD : OnSegment Q P D
axiom CQ_PD_perpendicular : isPerpendicular (line C Q) (line P D)

-- The theorem to be proven
theorem isosceles_triangle_BQC : Isosceles B Q C := by
  sorry

end isosceles_triangle_BQC_l826_826748


namespace hash_nesting_example_l826_826575

def hash (N : ℝ) : ℝ :=
  0.5 * N + 2

theorem hash_nesting_example : hash (hash (hash (hash 20))) = 5 :=
by
  sorry

end hash_nesting_example_l826_826575


namespace problem_solution_l826_826247

open Set Real

def setA := {x : ℝ | 2^(4 * x + 6) ≥ 64^x}
def setB := {x : ℝ | 2 * x^2 + x - 15 ≤ 0}
def setC (k : ℝ) := {x : ℝ | -2 ≤ x - k ∧ x - k ≤ 1/2}

theorem problem_solution : 
  (setA = {x : ℝ | x ≤ 3}) ∧ 
  (compl setA ∪ setB = { x : ℝ | -3 ≤ x ∧ x ≤ 5/2 ∨ x > 3 }) ∧
  (∀ k : ℝ, setC k ⊆ setB → -1 ≤ k ∧ k ≤ 2) := 
  by
  sorry

end problem_solution_l826_826247


namespace simplify_fraction_l826_826790

-- Define the fraction and the GCD condition
def fraction_numerator : ℕ := 66
def fraction_denominator : ℕ := 4356
def gcd_condition : ℕ := Nat.gcd fraction_numerator fraction_denominator

-- State the theorem that the fraction simplifies to 1/66 given the GCD condition
theorem simplify_fraction (h : gcd_condition = 66) : (fraction_numerator / fraction_denominator = 1 / 66) :=
  sorry

end simplify_fraction_l826_826790


namespace constant_term_of_quadratic_eq_l826_826448

theorem constant_term_of_quadratic_eq (b : ℝ) :
  ∃ c : ℝ, (2:ℝ) * x^2 - b * x - c = 0 := 
begin
  use(-1),
  sorry,
end

end constant_term_of_quadratic_eq_l826_826448


namespace time_to_fill_cistern_l826_826486

noncomputable def pipe_A_rate := 1 / 10
noncomputable def pipe_B_rate := 1 / 12
noncomputable def pipe_C_rate := -1 / 15

def combined_rate := pipe_A_rate + pipe_B_rate + pipe_C_rate

theorem time_to_fill_cistern :
  1 / combined_rate = 60 / 7 :=
by
  sorry

end time_to_fill_cistern_l826_826486


namespace exists_diff_four_of_subset_of_1001_l826_826016

theorem exists_diff_four_of_subset_of_1001 (f : Fin 1997 → Fin 1997) (hf1 : ∀ x, f x ≤ 1997) (hf2 : ∀ x1 x2, x1 ≠ x2 → f x1 ≠ f x2) (S : Finset (Fin 1997)) (hS : S.card = 1001) 
: ∃ (a b ∈ S), a ≠ b ∧ |a.val - b.val| = 4 :=
by
  sorry

end exists_diff_four_of_subset_of_1001_l826_826016


namespace naomi_stickers_l826_826093

theorem naomi_stickers :
  ∃ S : ℕ, S > 1 ∧
    (S % 5 = 2) ∧
    (S % 9 = 2) ∧
    (S % 11 = 2) ∧
    S = 497 :=
by
  sorry

end naomi_stickers_l826_826093


namespace PropositionA_PropositionD_l826_826505

-- Proposition A: a > 1 is a sufficient but not necessary condition for 1/a < 1.
theorem PropositionA (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by sorry

-- PropositionD: a ≠ 0 is a necessary but not sufficient condition for ab ≠ 0.
theorem PropositionD (a b : ℝ) (h : a ≠ 0) : a * b ≠ 0 :=
by sorry
 
end PropositionA_PropositionD_l826_826505


namespace power_sum_mod_7_l826_826567

theorem power_sum_mod_7 : 
  (∑ i in Finset.range 2012, 2^(i+1) + 5^2012) % 7 = 6 := 
by 
sorry

end power_sum_mod_7_l826_826567


namespace sum_of_numerator_denominator_l826_826970

open Set BigOperators

noncomputable def probability_brick_fits_box {a1 a2 a3 b1 b2 b3 : ℕ} :=
  (a1 < b1) ∧ (a2 < b2) ∧ (a3 < b3)

def number_of_ways_choose_six : ℕ :=
  nat.choose 1000 6

theorem sum_of_numerator_denominator : ℕ :=
  let p := 1 / 4 in
  let numerator := 1 in
  let denominator := 4 in
  numerator + denominator

end sum_of_numerator_denominator_l826_826970


namespace same_sign_m_minus_n_opposite_sign_m_plus_n_l826_826612

-- Definitions and Conditions
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom abs_m_eq_4 : |m| = 4
axiom abs_n_eq_3 : |n| = 3

-- Part 1: Prove m - n when m and n have the same sign
theorem same_sign_m_minus_n :
  (m > 0 ∧ n > 0) ∨ (m < 0 ∧ n < 0) → (m - n = 1 ∨ m - n = -1) :=
by
  sorry

-- Part 2: Prove m + n when m and n have opposite signs
theorem opposite_sign_m_plus_n :
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = 1 ∨ m + n = -1) :=
by
  sorry

end same_sign_m_minus_n_opposite_sign_m_plus_n_l826_826612


namespace number_of_subsets_of_A_l826_826653

-- Define the set A
def A := {1, 2, 3}

-- Prove that the number of subsets of set A is 8
theorem number_of_subsets_of_A : set.subset A = 8 := by
  sorry

end number_of_subsets_of_A_l826_826653


namespace number_of_remaining_numbers_problem_solution_l826_826403

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826403


namespace speed_of_train_in_km_per_hr_l826_826893

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l826_826893


namespace evaluate_expression_l826_826205

theorem evaluate_expression : 
  (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = 1372 * 10^1003 := 
by sorry

end evaluate_expression_l826_826205


namespace number_of_paths_K_to_L_l826_826706

-- Definition of the problem structure
def K : Type := Unit
def A : Type := Unit
def R : Type := Unit
def L : Type := Unit

-- Defining the number of paths between each stage
def paths_from_K_to_A := 2
def paths_from_A_to_R := 4
def paths_from_R_to_L := 8

-- The main theorem stating the number of paths from K to L
theorem number_of_paths_K_to_L : paths_from_K_to_A * 2 * 2 = 8 := by 
  sorry

end number_of_paths_K_to_L_l826_826706


namespace sin_alpha_value_l826_826634

def x : ℝ := 3
def y : ℝ := 4
def r : ℝ := Real.sqrt (x^2 + y^2)

theorem sin_alpha_value : Real.sin (Real.arctan (y / x)) = 4 / 5 := sorry

end sin_alpha_value_l826_826634


namespace part_a_l826_826843

variables {Point Line : Type*}
variable tangent : Point → Line → Prop
variable (A B : Point) (l : Line)

theorem part_a :
  ∃ (O : Point) (r : ℝ), 
    tangent A l ∧ tangent B l ∧ tangent O l := sorry

end part_a_l826_826843


namespace part1_local_extrema_part2_monotonicity_l826_826276

section Part1
  variable (x : ℝ)

  noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 4 * Real.log x

  theorem part1_local_extrema :
    let f1 := f 1
    let f2 := f 2
    f1 = -5 ∧ f2 = 4 * Real.log 2 - 8 :=
  sorry
end Part1

section Part2
  variable (x a : ℝ)

  noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*(a+1)*x + 2*a*(Real.log x)

  theorem part2_monotonicity (a : ℝ) (ha : a > 0) :
    (0 < a ∧ a < 1 → 
      (∀ x, (0 < x < a → deriv (f x a) x > 0) ∧
      (a < x < 1 → deriv (f x a) x < 0) ∧
      (1 < x → deriv (f x a) x > 0))) ∧
    (a = 1 →
      (∀ x, (0 < x → deriv (f x a) x > 0))) ∧
    (a > 1 →
      (∀ x, (0 < x < 1 → deriv (f x a) x > 0) ∧
      (1 < x < a → deriv (f x a) x < 0) ∧
      (a < x → deriv (f x a) x > 0))) :=
  sorry
end Part2

end part1_local_extrema_part2_monotonicity_l826_826276


namespace original_price_l826_826905

theorem original_price (P : ℝ) (h : P * 0.5 = 1200) : P = 2400 := 
by
  sorry

end original_price_l826_826905


namespace max_matches_for_winner_l826_826524

-- Define the necessary conditions
def participants : ℕ := 55
def sequential_matches : Prop :=
  ∀ (match_number : ℕ), match_number ≤ participants
def balanced_victories (match_number : ℕ) : Prop :=
  ∀ (participant_1_wins participant_2_wins : ℕ),
    |participant_1_wins - participant_2_wins| ≤ 1

-- Define the maximum number of matches
def max_matches : ℕ := 8

-- Formalize the proof problem in Lean
theorem max_matches_for_winner :
  ∃ (matches : ℕ), matches = max_matches
  ∧ participants = 55
  ∧ sequential_matches
  ∧ balanced_victories matches :=
  sorry

end max_matches_for_winner_l826_826524


namespace abs_diff_max_min_l826_826095

noncomputable def min_and_max_abs_diff (x : ℝ) : ℝ :=
|x - 2| + |x - 3| - |x - 1|

theorem abs_diff_max_min (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) :
  ∃ (M m : ℝ), M = 0 ∧ m = -1 ∧
    M = max (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) ∧ 
    m = min (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) :=
by
  use [0, -1]
  split
  case inl => sorry
  case inr => sorry

end abs_diff_max_min_l826_826095


namespace avg_of_combined_data_l826_826267

variables (x1 x2 x3 y1 y2 y3 a b : ℝ)

-- condition: average of x1, x2, x3 is a
axiom h1 : (x1 + x2 + x3) / 3 = a

-- condition: average of y1, y2, y3 is b
axiom h2 : (y1 + y2 + y3) / 3 = b

-- Prove that the average of 3x1 + y1, 3x2 + y2, 3x3 + y3 is 3a + b
theorem avg_of_combined_data : 
  ((3 * x1 + y1) + (3 * x2 + y2) + (3 * x3 + y3)) / 3 = 3 * a + b :=
by
  sorry

end avg_of_combined_data_l826_826267


namespace friend_owns_10_bikes_l826_826680

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end friend_owns_10_bikes_l826_826680


namespace P_2017_eq_14_l826_826792

def sumOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def numberOfDigits (n : Nat) : Nat :=
  n.digits 10 |>.length

def P (n : Nat) : Nat :=
  sumOfDigits n + numberOfDigits n

theorem P_2017_eq_14 : P 2017 = 14 :=
by
  sorry

end P_2017_eq_14_l826_826792


namespace find_ellipse_equation_find_std_ellipse_equation_l826_826854

-- Define conditions and statement for Question 1
def ellipse_conditions (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ b < a ∧
  (a^2 = b^2 + (a * √(2/2))^2) ∧
  (0, -1) ∈ {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1} ∧
  b = 1

theorem find_ellipse_equation (a b : ℝ) (h : ellipse_conditions a b) :
  (a^2 = 2 ∧ b = 1 → (∀ x y : ℝ, (y^2) = 1 - (x^2)/2)) :=
begin
  sorry
end

-- Define conditions and statement for Question 2
def std_ellipse_conditions (x y m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n ∧
  (2 / m + √2 / n = 1) ∧
  (√6 / m + 1 / n = 1)

theorem find_std_ellipse_equation (m n : ℝ) (h : std_ellipse_conditions m n) :
  (m = 8 ∧ n = 4 → (∀ x y : ℝ, y^2 / 4 + x^2 / 8 = 1)) :=
begin
  sorry
end

end find_ellipse_equation_find_std_ellipse_equation_l826_826854


namespace problem_part1_problem_part2_l826_826808

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

-- Condition 1: Maximum value is 3
axiom max_value_of_f : ∀ x : ℝ, f x ≤ 3

-- Condition 2: The distance between symmetry axes
axiom symmetry_distance : ∀ x : ℝ, f (x + Real.pi / 2) = f x

-- Define the bounds for x
def x_interval := Icc 0 Real.pi

-- Problem part 1: Prove f(x) and its monotonically decreasing interval
theorem problem_part1 : 
  (∀ x : ℝ, f x ≤ 3) ∧
  (∀ x : ℝ, f (x + Real.pi / 2) = f x) →
  (f (2 * Real.sin (2 * x - Real.pi / 6) + 1) ) ∧ 
  ( ∀ x ∈ x_interval, f x = 2 * Real.sin (2 * x - Real.pi / 6) + 1 ) :=
sorry

-- Problem part 2: Given f(a/2) = 2 with a in (0, Real.pi / 2), find a
theorem problem_part2 (a : ℝ) (h1 : f (a / 2) = 2) (h2 : 0 < a) (h3 : a < Real.pi / 2) : 
  a = Real.pi / 3 :=
sorry

end problem_part1_problem_part2_l826_826808


namespace solve_general_term_l826_826246

-- Defining the sequences and conditions
def sequences (a b : ℕ → ℝ) : Prop :=
  (∀ n, b (2*n + 1) = a (n + 1)) ∧
  (∀ n, b (2 * n) = real.sqrt (a (2 * n + 1))) ∧
  (∃ r : ℝ, ∀ m n, b m * b n = b (m + n - 1) * r) ∧
  (a 2 + b 2 = 108)

-- The general term we want to prove
def general_term (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 9 ^ n

theorem solve_general_term :
  ∃ a b, sequences a b ∧ general_term a := sorry

end solve_general_term_l826_826246


namespace count_final_numbers_l826_826408

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826408


namespace train_speed_is_72_km_per_hr_l826_826900

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l826_826900


namespace simplify_fraction_l826_826789

variable (k : ℤ)

theorem simplify_fraction (a b : ℤ)
  (hk : a = 2)
  (hb : b = 4) :
  (6 * k + 12) / 3 = 2 * k + 4 ∧ (a : ℚ) / (b : ℚ) = 1 / 2 := 
by
  sorry

end simplify_fraction_l826_826789


namespace power_multiplication_eq_neg4_l826_826922

theorem power_multiplication_eq_neg4 :
  (-0.25) ^ 11 * (-4) ^ 12 = -4 := 
  sorry

end power_multiplication_eq_neg4_l826_826922


namespace evaluate_expression_l826_826946

theorem evaluate_expression : 
  (3 * real.sqrt 10) / (real.sqrt 3 + real.sqrt 5 + real.sqrt 7) = 
  (-2 * real.sqrt 7 + real.sqrt 3 + real.sqrt 5) / 59 :=
by
  sorry

end evaluate_expression_l826_826946


namespace binomial_expansion_correctness_l826_826335

-- Definitions based on the conditions
def binomial_expansion (x : ℝ) : ℝ :=
  (2 * real.sqrt x - 1 / real.sqrt x) ^ 8

-- Proof statement which will conclude with the correct options being B, C, and D
theorem binomial_expansion_correctness (x : ℝ) : 
  -- Sum of the coefficients is 1 
  (binomial_expansion 1 = 1) ∧
  -- Largest coefficient at the 5th term
  (true) ∧ -- Placeholder for the proof of right terms being the max in their places value.
  -- Coefficient of the 4th term being the smallest
  (true) := -- Placeholder for the detailed analysis.
by 
  -- Placeholder for the proof to ensure the statement builds successfully.
  sorry

end binomial_expansion_correctness_l826_826335


namespace number_of_remaining_numbers_problem_solution_l826_826404

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826404


namespace sin_300_l826_826122

-- Define the key trigonometric properties used in the conditions
lemma reduction_formula_sin (x : ℝ) : sin (360 - x) = - sin x :=
by sorry

lemma sin_60 : sin 60 = (√3 / 2) :=
by sorry

-- Main theorem to prove
theorem sin_300 : sin 300 = - (√3 / 2) :=
by 
  rw [show 300 = 360 - 60, by norm_num],
  rw [reduction_formula_sin, sin_60]
  done

end sin_300_l826_826122


namespace union_A_B_l826_826989

noncomputable def set_A : set ℝ := {x | log 2 (x - 2) > 0}
noncomputable def set_B : set ℝ := {y | ∃ x ∈ (set_A ∩ {x | x > 3}), y = x^2 - 4 * x + 5}
noncomputable def union_set : set ℝ := {z | z > 2}

theorem union_A_B :
  (set_A ∪ set_B) = union_set :=
    sorry

end union_A_B_l826_826989


namespace line_circle_intersect_l826_826051

theorem line_circle_intersect (a : ℝ) : 
  let line := {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ ax + y - a = 0} in
  let circle := {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 = 4} in
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ line ∧ p1 ∈ circle ∧ p2 ∈ line ∧ p2 ∈ circle ∧ p1 ≠ p2 :=
sorry

end line_circle_intersect_l826_826051


namespace triangle_area_eq_18_l826_826210

theorem triangle_area_eq_18 :
  let A := (4, -3) in
  let B := (-3, 1) in
  let C := (2, -7) in
  let v := (A.1 - C.1, A.2 - C.2) in
  let w := (B.1 - C.1, B.2 - C.2) in
  let parallelogram_area := abs (v.1 * w.2 - v.2 * w.1) in
  parallelogram_area / 2 = 18 :=
by
  let A := (4, -3)
  let B := (-3, 1)
  let C := (2, -7)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs (v.1 * w.2 - v.2 * w.1)
  have h : parallelogram_area / 2 = 18
  sorry

end triangle_area_eq_18_l826_826210


namespace negation_of_proposition_l826_826043

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≤ 0 ∧ x^2 ≥ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 < 0 :=
by
  sorry

end negation_of_proposition_l826_826043


namespace xy_product_eq_one_l826_826307

noncomputable def xy_value (x y : ℝ) := x * y

theorem xy_product_eq_one (x y : ℝ) 
  (h : (1 + complex.i : ℂ) * x + (1 - complex.i : ℂ) * y = (2 : ℂ)) :
  xy_value x y = 1 :=
sorry

end xy_product_eq_one_l826_826307


namespace total_amount_distributed_l826_826502

theorem total_amount_distributed (A : ℝ) :
  (∀ A, (A / 14 = A / 18 + 80) → A = 5040) :=
by
  sorry

end total_amount_distributed_l826_826502


namespace bill_caroline_ratio_l826_826178

theorem bill_caroline_ratio (B C : ℕ) (h1 : B = 17) (h2 : B + C = 26) : B / C = 17 / 9 := by
  sorry

end bill_caroline_ratio_l826_826178


namespace new_ratio_of_pets_l826_826658

def initial_dog_to_cat_to_bird_ratio : ℕ × ℕ × ℕ := (10, 17, 8)
def total_pets : ℕ := 315

def dogs_given_away : ℕ := 15
def birds_adopted : ℕ := 7
def cats_adopted : ℕ := 4
def cats_found_home : ℕ := 12
def birds_found_home : ℕ := 5

theorem new_ratio_of_pets :
  let initial_total_ratio := (initial_dog_to_cat_to_bird_ratio.1 + initial_dog_to_cat_to_bird_ratio.2 + initial_dog_to_cat_to_bird_ratio.3)
  let pets_per_part := total_pets / initial_total_ratio
  let initial_dogs := initial_dog_to_cat_to_bird_ratio.1 * pets_per_part
  let initial_cats := initial_dog_to_cat_to_bird_ratio.2 * pets_per_part
  let initial_birds := initial_dog_to_cat_to_bird_ratio.3 * pets_per_part

  let new_dogs := initial_dogs - dogs_given_away
  let new_cats := initial_cats + cats_adopted - cats_found_home
  let new_birds := initial_birds + birds_adopted - birds_found_home

  gcd new_dogs (gcd new_cats new_birds) = 1 →
  new_dogs = 75 ∧
  new_cats = 145 ∧
  new_birds = 74 :=
by
  sorry

end new_ratio_of_pets_l826_826658


namespace cannot_reach_0_0_from_0_1_l826_826005

def reachable_moves : (ℤ × ℤ) → set (ℤ × ℤ)
| (x, y) := {(y, x), (3 * x, -2 * y), (-2 * x, 3 * y), (x + 1, y + 4), (x - 1, y - 4)}

def reachable_from (start : ℤ × ℤ) : set (ℤ × ℤ) :=
  {p | ∃ n, (reachable_moves ^^ n) {start} p}

def v (pt : ℤ × ℤ) := (pt.1 + pt.2) % 5

theorem cannot_reach_0_0_from_0_1 :
  (0, 0) ∉ reachable_from (0, 1) :=
sorry

end cannot_reach_0_0_from_0_1_l826_826005


namespace bran_has_enough_money_l826_826921

def tuition_fee := 3000
def additional_expenses := 800
def savings_goal := 500
def part_time_hourly_wage := 20
def part_time_hours_per_week := 15
def tutoring_bi_weekly_payment := 100
def scholarship_coverage := 0.40
def tax_rate := 0.10
def weeks_per_month := 4
def months_per_semester := 4

-- Define the total tuition cost after scholarships
def net_tuition_fee := tuition_fee * (1 - scholarship_coverage)

-- Define the total cost Bran needs to cover
def total_needed := net_tuition_fee + additional_expenses + savings_goal

-- Define the total income from part-time job after tax
def part_time_income :=
  part_time_hours_per_week * weeks_per_month * months_per_semester * part_time_hourly_wage

def net_part_time_income := part_time_income * (1 - tax_rate)

-- Define the total income from tutoring job
def total_tutoring_income := tutoring_bi_weekly_payment * (months_per_semester * (weeks_per_month / 2))

-- Total income from both jobs
def total_income := net_part_time_income + total_tutoring_income

-- Amount Bran still needs to pay
def surplus_income := total_income - total_needed

theorem bran_has_enough_money : surplus_income = 2020 := by
  -- This is where the proof would go
  sorry

end bran_has_enough_money_l826_826921


namespace rope_length_after_knots_l826_826828

def num_ropes : ℕ := 64
def length_per_rope : ℕ := 25
def length_reduction_per_knot : ℕ := 3
def num_knots : ℕ := num_ropes - 1
def initial_total_length : ℕ := num_ropes * length_per_rope
def total_reduction : ℕ := num_knots * length_reduction_per_knot
def final_rope_length : ℕ := initial_total_length - total_reduction

theorem rope_length_after_knots :
  final_rope_length = 1411 := by
  sorry

end rope_length_after_knots_l826_826828


namespace find_tan_alpha_l826_826971

theorem find_tan_alpha (α : ℝ) (h : (sin α - 2 * cos α) / (2 * sin α + 3 * cos α) = 2) : tan α = -8 / 3 :=
by
  sorry

end find_tan_alpha_l826_826971


namespace find_x_coordinate_of_point_M_l826_826800

noncomputable def point_on_parabola (M : ℝ × ℝ) : Prop :=
  let (x_M, y_M) := M
  y_M^2 = 4 * x_M

noncomputable def distance_to_focus (M : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  let (x_M, y_M) := M
  let (x_F, y_F) := F
  real.sqrt ((x_F - x_M)^2 + (y_F - y_M)^2)

theorem find_x_coordinate_of_point_M (x_M y_M : ℝ) (h1 : point_on_parabola (x_M, y_M))
(h2 : distance_to_focus (x_M, y_M) (1, 0) = 4) : x_M = 3 :=
  sorry

end find_x_coordinate_of_point_M_l826_826800


namespace geometric_N_digit_not_20_l826_826638

-- Variables and definitions
variables (a b c : ℕ)

-- Given conditions
def geometric_progression (a b c : ℕ) : Prop :=
  ∃ q : ℚ, (b = q * a) ∧ (c = q * b)

def ends_with_20 (N : ℕ) : Prop := N % 100 = 20

-- Prove the main theorem
theorem geometric_N_digit_not_20 (h1 : geometric_progression a b c) (h2 : ends_with_20 (a^3 + b^3 + c^3 - 3 * a * b * c)) :
  False :=
sorry

end geometric_N_digit_not_20_l826_826638


namespace find_positive_x_l826_826595

theorem find_positive_x (x : ℝ) (h1 : x * ⌊x⌋ = 72) (h2 : x > 0) : x = 9 :=
by 
  sorry

end find_positive_x_l826_826595


namespace product_of_elements_is_zero_l826_826056

theorem product_of_elements_is_zero (n : ℕ) (M : Fin n → ℝ) (hn1 : Odd n) (hn2 : 1 < n)
  (h : ∀ (i : Fin n), (∑ j, if i = j then ∑ k, if i = k then (0 : ℝ) else M k else M j) = ∑ j, M j) :
  ∏ i, M i = 0 :=
by
  sorry

end product_of_elements_is_zero_l826_826056


namespace least_lcm_420_l826_826454

open Nat

/-- 
Prove that given lcm(a, b) = 20 and lcm(b, c) = 21, 
the least possible value of lcm(a, c) is 420 
-/
theorem least_lcm_420 (a b c : ℕ) (h1 : lcm a b = 20) (h2 : lcm b c = 21) : lcm a c = 420 := by
  sorry

end least_lcm_420_l826_826454


namespace numbers_left_on_board_l826_826425

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826425


namespace sin_phi_psi_ratio_l826_826518

theorem sin_phi_psi_ratio (A B C M : Type) [EuclideanGeometry A B C M]
  (h1 : is_right_triangle A B C (∠C = 90)) 
  (h2 : ∠ M A B = ∠ M B C = ∠ M C A = φ) 
  (h3 : is_median_angle A C B C Ψ) : 
  sin(φ + Ψ) / sin(φ - Ψ) = 5 :=
sorry

end sin_phi_psi_ratio_l826_826518


namespace units_digit_G_1009_l826_826189

/-- Problem Statement: --/
def G (n : ℕ) : ℕ := 3^(2^n) + 1

theorem units_digit_G_1009 :
  Nat.unitsDigit (G 1009) = 4 :=
by
  -- We will fill this part with the actual proof later
  sorry

end units_digit_G_1009_l826_826189


namespace number_of_second_grade_students_selected_l826_826883

/-- Given a school with 3300 students partitioned into first, second, and third grades
    in the ratio 12:10:11, when 66 students are selected via stratified sampling,
    the number of second-grade students selected is 20. -/
theorem number_of_second_grade_students_selected
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (selected_students : ℕ)
    (h_total_students : total_students = 3300)
    (h_ratio : ratio_first = 12 ∧ ratio_second = 10 ∧ ratio_third = 11)
    (h_selected_students : selected_students = 66) :
    let total_ratio := ratio_first + ratio_second + ratio_third,
        proportion_second := (ratio_second : ℚ) / total_ratio,
        second_grade_selected := (proportion_second * (selected_students : ℚ)).to_nat
    in second_grade_selected = 20 :=
by
  sorry

end number_of_second_grade_students_selected_l826_826883


namespace quadratic_real_roots_range_of_k_l826_826271

theorem quadratic_real_roots (k : ℝ) :
    let a := 1
    let b := -(k + 3)
    let c := 2 * k + 2
    let delta := b^2 - 4 * a * c
    delta = (k - 1)^2 ∧ ((k - 1)^2 ≥ 0) := 
by {
    let a := 1
    let b := -(k + 3)
    let c := 2 * k + 2
    let delta := b^2 - 4 * a * c
    have h1 : delta = (k - 1)^2 :=
      calc delta = (k + 3)^2 - 4 * 1 * (2 * k + 2) : by simp [a, b, c]
          ... = k^2 + 6 * k + 9 - 8 * k - 8 : by ring
          ... = k^2 - 2 * k + 1 : by ring,
    have h2 : (k - 1)^2 ≥ 0 := by apply pow_two_nonneg,
    exact ⟨h1, h2⟩
}
-- Question 2
theorem range_of_k (k : ℝ) (h: 0 < k + 1 ∧ k + 1 < 1) : -1 < k ∧ k < 0 := 
by {
    cases h with h1 h2,
    have h3 : k + 1 < 1 := h2,
    have h4 : -1 < k := by linarith,
    have h5 : k < 0 := by linarith,
    exact ⟨h4, h5⟩,
}

end quadratic_real_roots_range_of_k_l826_826271


namespace no_integer_solution_l826_826781

theorem no_integer_solution (x y z : ℤ) (n : ℕ) (h1 : Prime (x + y)) (h2 : Odd n) : ¬ (x^n + y^n = z^n) :=
sorry

end no_integer_solution_l826_826781


namespace train_speed_l826_826157

theorem train_speed 
  (t1 : ℝ) (t2 : ℝ) (L : ℝ) (v : ℝ) 
  (h1 : t1 = 12) 
  (h2 : t2 = 44) 
  (h3 : L = v * 12)
  (h4 : L + 320 = v * 44) : 
  (v * 3.6 = 36) :=
by
  sorry

end train_speed_l826_826157


namespace speed_of_train_in_km_per_hr_l826_826892

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l826_826892


namespace find_n_l826_826134

-- Defining the conditions given in the problem
def condition_eq (n : ℝ) : Prop :=
  10 * 1.8 - (n * 1.5 / 0.3) = 50

-- Stating the goal: Prove that the number n is -6.4
theorem find_n : condition_eq (-6.4) :=
by
  -- Proof is omitted
  sorry

end find_n_l826_826134


namespace math_problem_equivalence_l826_826269

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def foci_def (x1 y1 x2 y2 : ℝ) : Prop := 
  -- Define foci of the ellipse
  true 

noncomputable def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  ellipse_equation a b x y

noncomputable def distance_relation (a : ℝ) : Prop :=
  2 * a = 4

noncomputable def line_through (x1 y1 x2 y2 : ℝ) : Prop :=
  -- Define a line passing through the given points
  true

noncomputable def perpendicular_line_to_xaxis (x y : ℝ) : Prop := 
  -- Define a perpendicular line
  true

theorem math_problem_equivalence :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ point_on_ellipse 2 (sqrt 3) (2 * sqrt 6 / 3) (-1) ∧ distance_relation 2
  ∧ ∃ (k : ℝ), line_through 4 0 (k * 4 - 4) (k * (x - 4)) ∧ perpendicular_line_to_xaxis (x - 4) x
  ∧ (ellipse_equation 2 (sqrt 3) x y → ∃ λ : ℝ, true) :=
begin
  sorry,
end

end math_problem_equivalence_l826_826269


namespace odd_function_sufficient_not_necessary_l826_826801

variables {R : Type*} [Real : real field]

def is_odd_function (f : R → R) : Prop := ∀ x : R, f(x) = -f(-x)

theorem odd_function_sufficient_not_necessary (f : R → R)
  (hf : ∀ x : R, f(x) = -f(-x)) :
  ∃ x : R, f(x) + f(-x) = 0 ∧ ¬ (∀ x : R, (f(x) + f(-x) = 0) → (∀ y, f(y) = -f(-y))) :=
by sorry

end odd_function_sufficient_not_necessary_l826_826801


namespace xyz_squared_eq_one_l826_826250

theorem xyz_squared_eq_one (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h_eq : ∃ k, x + (1 / y) = k ∧ y + (1 / z) = k ∧ z + (1 / x) = k) : 
    x^2 * y^2 * z^2 = 1 := 
  sorry

end xyz_squared_eq_one_l826_826250


namespace number_of_logs_in_stack_l826_826548

theorem number_of_logs_in_stack : 
    let first_term := 15 in
    let last_term := 5 in
    let num_terms := first_term - last_term + 1 in
    let average := (first_term + last_term) / 2 in
    let sum := average * num_terms in
    sum = 110 :=
by
  sorry

end number_of_logs_in_stack_l826_826548


namespace not_necessarily_heavier_l826_826340

noncomputable def elephants := Fin 10 → ℝ

def condition (e : elephants) : Prop :=
  ∀ L R : Finset (Fin 10), L.card = 4 ∧ R.card = 3 ∧ L ∩ R = ∅ → (∑ i in L, e i) > (∑ i in R, e i)

def five_left_four_right (e : elephants) : Prop :=
  ∀ L R : Finset (Fin 10), L.card = 5 ∧ R.card = 4 ∧ L ∩ R = ∅ → 
  (∑ i in L, e i) > (∑ i in R, e i)

theorem not_necessarily_heavier (e : elephants) (h : condition e) :
  ¬ five_left_four_right e := 
sorry

end not_necessarily_heavier_l826_826340


namespace cost_price_calculation_l826_826169

theorem cost_price_calculation (C M : ℝ) (hM : M = 131.58) (hSP_cost : ∃ SP : ℝ, SP = 1.25 * C)
  (hSP_marked : ∃ SP : ℝ, SP = 0.95 * M) : C ≈ 100.00 :=
by sorry

end cost_price_calculation_l826_826169


namespace coeff_x3_in_qx_cube_zero_l826_826306

theorem coeff_x3_in_qx_cube_zero : 
  let q := λ x : ℝ, x^4 - 4 * x^2 + 3 
  in coeff (q x ^ 3) 3 = 0 :=
by
  sorry

end coeff_x3_in_qx_cube_zero_l826_826306


namespace length_E_to_E_l826_826084

structure Point where
  x : ℤ
  y : ℤ

def reflect_x (p : Point) : Point :=
  {x := p.x, y := -p.y}

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

theorem length_E_to_E' : 
  let E := Point.mk 2 3
  let E' := reflect_x E
  distance E E' = 6 := by
  sorry

end length_E_to_E_l826_826084


namespace repeating_decimal_to_fraction_l826_826207

theorem repeating_decimal_to_fraction (x : ℝ) (hx : x = 0.464646464646...) : x = 46 / 99 :=
sorry

end repeating_decimal_to_fraction_l826_826207


namespace equilateral_triangle_minimizes_OA2_OB2_OC2_l826_826979

theorem equilateral_triangle_minimizes_OA2_OB2_OC2 {O A B C r : ℝ} (h1 : O ≠ A) (h2 : O ≠ B) (h3 : O ≠ C)
  (tangent : ∀ {X Y Z : ℝ}, ∃ k : set (ℝ × ℝ), is_incircle k X Y Z O) :
  (∀ {k : set (ℝ × ℝ)}, is_incircle k A B C O → OA^2 + OB^2 + OC^2 ≥ 12 * r^2) ∧
  (OA^2 + OB^2 + OC^2 = 12 * r^2 ↔ is_equilateral_triangle A B C) :=
begin
  sorry
end

end equilateral_triangle_minimizes_OA2_OB2_OC2_l826_826979


namespace proof_problem_1_proof_problem_2_proof_problem_3_l826_826646

-- Problem 1: Monotonic Intervals for a Specific Function
def monotonic_intervals (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : Prop :=
  f(x) = real.log x + x^2 - 3 * x ∧
  ((0 < x ∧ x < 1 / 2 → f'(x) > 0) ∧ (1 / 2 < x ∧ x < 1 → f'(x) < 0) ∧ (1 < x → f'(x) > 0))

-- Problem 2: Range of a
def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x > 0, f(x) ≤ 2 * x^2) → a ≥ -1

-- Problem 3: Inequality Involving Logarithm of n
def log_inequality (n : ℕ) : Prop :=
  n ≥ 2 → real.log n > ∑ i in finset.filter (λ x, odd x) (finset.range (2 * n)), (1 : ℝ) / i

-- Combining all statements
theorem proof_problem_1 (f : ℝ → ℝ) (x : ℝ) : monotonic_intervals f 3 x :=
by sorry

theorem proof_problem_2 (f : ℝ → ℝ) (a : ℝ) : range_of_a f a :=
by sorry

theorem proof_problem_3 (n : ℕ) : log_inequality n :=
by sorry

end proof_problem_1_proof_problem_2_proof_problem_3_l826_826646


namespace problem1_problem2_problem3_l826_826263

noncomputable def a_n (n : ℕ) : ℕ := 3 * (2 ^ n) - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * a_n n - 3 * n

-- 1. Prove a_1 = 3 and a_2 = 9 given S_n = 2a_n - 3n
theorem problem1 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    a_n 1 = 3 ∧ a_n 2 = 9 :=
  sorry

-- 2. Prove that the sequence {a_n + 3} is a geometric sequence and find the general term formula for the sequence {a_n}.
theorem problem2 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    ∀ n, (a_n (n + 1) + 3) / (a_n n + 3) = 2 ∧ a_n n = 3 * (2 ^ n) - 3 :=
  sorry

-- 3. Prove {S_{n_k}} is not an arithmetic sequence given S_n = 2a_n - 3n and {n_k} is an arithmetic sequence
theorem problem3 (n_k : ℕ → ℕ) (h_arithmetic : ∃ d, ∀ k, n_k (k + 1) - n_k k = d) :
    ¬ ∃ d, ∀ k, S_n (n_k (k + 1)) - S_n (n_k k) = d :=
  sorry

end problem1_problem2_problem3_l826_826263


namespace rectangle_area_is_constant_l826_826879

noncomputable def prove_area_constancy (x y : ℚ) : Prop :=
  let A := x * y in
  A = ((x - (7/2)) * (y + (3/2))) ∧ A = ((x + (7/2)) * (y - (5/2)))

theorem rectangle_area_is_constant (x y A : ℚ) (h : prove_area_constancy x y) :
  A = (20/7) :=
begin
  sorry
end

end rectangle_area_is_constant_l826_826879


namespace sum_of_distinct_products_of_geometric_sequence_l826_826849

theorem sum_of_distinct_products_of_geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) (hq : q ≠ 1) :
  (∑ i in finset.range n, ∑ j in finset.Ico i.succ n, a * (q^i) * a * (q^j)) =
  (a^2 * q * (1 - q^n) * (1 - q^(n - 1))) / ((1 - q)^2 * (1 + q)) :=
by
  sorry

end sum_of_distinct_products_of_geometric_sequence_l826_826849


namespace sa_times_sd_eq_se_times_sf_l826_826749

variables {A B C S D E F : Type*} [inst : LinearOrderedField A] {Γ : set (Point A)}

-- Definitions and conditions
def is_triangle (A B C : Point A) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ A

def circumcircle (Γ : set (Point A)) (A B C : Point A) : Prop :=
  ∀ (P : Point A), P ∈ Γ ↔ P ≠ A ∧ P ≠ B ∧ P ≠ C ∧
  collinear {P, A, B} ∧ collinear {P, A, C} ∧ collinear {P, B, C}

def south_pole (S : Point A) (A : Point A) (Γ : set (Point A)) : Prop :=
  ∀ (P : Point A), P ≠ S ∧ collinear {P, S, A}

def intersects_at (D : Point A) (AS BC : Line A) : Prop :=
  D ∈ AS ∧ D ∈ BC

def passes_through (S : Point A) (E : Point A) (BC_F : set (Point A)) : Prop :=
  E ∈ BC ∧ ((F ∈ BC_F ∧ F ∈ Γ) ∧ collinear {S, F})

-- Theorem Statement
theorem sa_times_sd_eq_se_times_sf 
  (A B C S D E F : Point A)
  (Γ : set (Point A))
  (h_tri : is_triangle A B C)
  (h_circ : circumcircle Γ A B C)
  (h_south : south_pole S A Γ)
  (h_inter_D : intersects_at D (line_through A S) (line_through B C))
  (h_pass_E : passes_through S E (line_through B C ∩ Γ)) :
  (dist S A) * (dist S D) = (dist S E) * (dist S F) :=
sorry

end sa_times_sd_eq_se_times_sf_l826_826749


namespace ellipse_eccentricity_l826_826268

noncomputable def solve_ellipse : Prop :=
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 5) + (y^2 / m) = 1 ∧ ( ∃ (e : ℝ), e = (√10) / 5) ) →
    (m = 3 ∨ m = 25 / 3)

-- The theorem to be proven
theorem ellipse_eccentricity (m : ℝ) (h : solve_ellipse) : m = 3 ∨ m = 25 / 3 :=
sorry

end ellipse_eccentricity_l826_826268


namespace range_of_m_ln_extreme_points_l826_826279

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x - (1 / 2) * m * x^2 - x

theorem range_of_m (m : ℝ) : (∀ x > 0, (f' x = ∘(Real.log x - m * x)) ≤ 0) → m ≥ 1 / Real.exp := sorry

theorem ln_extreme_points (m x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2)
  (h4 : f x1 m = 0) (h5 : f x2 m = 0) : Real.log x1 + Real.log x2 > 2 := sorry

end range_of_m_ln_extreme_points_l826_826279


namespace max_product_sum_1979_l826_826962

theorem max_product_sum_1979 :
  ∃ (S : List ℕ), (∀ n ∈ S, n > 0) ∧ S.sum = 1979 ∧ (S.product = 2 * 3 ^ 659) := 
sorry

end max_product_sum_1979_l826_826962


namespace car_storm_times_half_sum_l826_826133

-- Definitions and conditions
def car_position (t : ℝ) := (2/3 * t, 0 : ℝ)
def storm_center (t : ℝ) := (3/4 * t, 130 - 3/4 * t : ℝ)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
def within_storm (t : ℝ) : Prop := distance (car_position t) (storm_center t) ≤ 60

-- The main theorem we aim to prove
theorem car_storm_times_half_sum :
  ∃ t1 t2 : ℝ, within_storm t1 ∧ within_storm t2 ∧
  (∀ t : ℝ, (t1 < t ∧ t < t2) → ¬ within_storm t) ∧
  1/2 * (t1 + t2) = 343 :=
sorry -- proof placeholder

end car_storm_times_half_sum_l826_826133


namespace satisfy_inequality_l826_826953

theorem satisfy_inequality (x : ℝ) :
  x ≠ -1 ∧ x ≠ 1 → 
  (x^2 / (x + 1) ≥ 2 / (x - 1) + 7 / 4 ↔ x ∈ set.Ioo (-1 : ℝ) 1 ∪ set.Ici 4) :=
by
  intro h
  sorry

end satisfy_inequality_l826_826953


namespace distance_of_Q_l826_826320

-- Define the triangle with given side lengths
structure Triangle :=
  (a b c : ℝ)
  (is_right : a^2 + b^2 = c^2)

-- Define the circle with center Q and a radius
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the primary function to compute the distance traveled by Q
def distanceTraveled (T : Triangle) (C : Circle) : ℝ :=
  if (T.a = 5 ∧ T.b = 12 ∧ T.c = 13 ∧ T.is_right ∧ C.radius = 2) then 18 else 0

-- The theorem to prove the distance traveled
theorem distance_of_Q :
  ∃ (T : Triangle) (C : Circle),
    T.a = 5 ∧ T.b = 12 ∧ T.c = 13 ∧ T.is_right ∧ C.radius = 2 ∧ distanceTraveled T C = 18 :=
by
  existsi Triangle.mk 5 12 13 (by linarith),
  existsi Circle.mk (0, 0) 2,
  simp [distanceTraveled],
  sorry

end distance_of_Q_l826_826320


namespace total_time_for_two_round_trips_l826_826057

open Float

def boat_speed := 15 -- kmph
def stream_speed := 8 -- kmph
def distance := 350 -- km

def downstream_speed : Float := boat_speed + stream_speed
def upstream_speed : Float := boat_speed - stream_speed

def t_downstream : Float := distance / downstream_speed
def t_upstream : Float := distance / upstream_speed

def t_total_one_trip : Float := t_downstream + t_upstream
def t_total_two_trips : Float := 2 * t_total_one_trip

theorem total_time_for_two_round_trips (approx_equal : Float → Float → Prop) : 
    approx_equal t_total_two_trips 130.434 :=
sorry

end total_time_for_two_round_trips_l826_826057


namespace geometric_sum_ratio_eq_l826_826728

variable {a : ℕ → ℝ} {r : ℝ}
variable (a1 : a 5 + 2 * a 10 = 0) (s : ℕ → ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  S n = a 1 * (1 - q ^ n) / (1 - q)

theorem geometric_sum_ratio_eq (q : ℝ) (hq : q ^ 5 = -1 / 2) 
  (geometric_seq : geometric_sequence a q)
  (sum_geo_seq : sum_geometric_sequence a q s) : 
  s 20 / s 10 = 5 / 4 :=
by
  sorry

end geometric_sum_ratio_eq_l826_826728


namespace locus_of_orthocenter_is_circle_without_two_points_l826_826429

noncomputable def circle_locus_exclude_pts
    (A B : Point)
    (circ : Circle)
    (on_circle : ∀ (C : Point), C ∈ circ → is_triangle_orthocenter A B C)
    (perpendiculars : Line ⊥ LineAB)
    (original_circle_sym: Circle → Circle) : Set Point :=
{ H | ∃ (C : Point) (H : Point), 
    C ∈ circ ∧ H = orthocenter A B C ∧ 
    H ∈ original_circle_sym circ ∧
    H ∉ {P | P = Line(⊥ A) ∩ circ} ∧ 
    H ∉ {P | P = Line(⊥ B) ∩ circ} }

-- Predicate for orthocenter
axiom is_triangle_orthocenter 
  (A B C : Point) : Prop

-- To state that the locus forms another circle excluding two points
theorem locus_of_orthocenter_is_circle_without_two_points 
    (A B : Point)
    (circ : Circle)
    (on_circle : ∀ (C : Point), C ∈ circ → is_triangle_orthocenter A B C)
    (perpendiculars : Line ⊥ LineAB)
    (original_circle_sym: Circle → Circle) :
  ∃ (locus : Set Point),
  locus = circle_locus_exclude_pts A B circ on_circle perpendiculars original_circle_sym := 
sorry

end locus_of_orthocenter_is_circle_without_two_points_l826_826429


namespace angle_TRS_45_degrees_l826_826565

-- Define datatypes for points and circles
structure Point where
  x : ℝ
  y : ℝ

-- Circle structure
structure Circle where
  center : Point
  radius : ℝ

-- Definitions of points and circles based on the problem
def P : Point := {x := 0, y := 0}
def Q : Point := {x := r, y := 0}
def X : Circle := {center := P, radius := r}
def Y : Circle := {center := Q, radius := 2*r}

-- Definitions to state that Q lies on circle X and circles X and Y intersect at R and S
def Q_on_X : Q.x ^ 2 + Q.y ^ 2 = X.radius ^ 2 := by
  -- calculation to show this
  sorry

-- Line PQ intersects X at T and Y at U
def int_T_Y : Point := sorry
def int_U_Y : Point := sorry

-- Definitions for angle TRS
def angle_TRS := 45

-- Final theorem statement in Lean 4
theorem angle_TRS_45_degrees (P Q R S T U : Point) (X Y : Circle) (r : ℝ) :
  X.center = P ∧ X.radius = r ∧
  Y.center = Q ∧ Y.radius = 2 * r ∧
  Q_on_X ∧
  (T = int_T_Y) ∧ (U = int_U_Y) ∧
  (∃ R S : Point, (R = sorry) ∧ (S = sorry)) →
  angle_TRS = 45 := sorry

end angle_TRS_45_degrees_l826_826565


namespace find_length_of_room_l826_826040

noncomputable def cost_of_paving : ℝ := 21375
noncomputable def rate_per_sq_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem find_length_of_room :
  ∃ l : ℝ, l = (cost_of_paving / rate_per_sq_meter) / width_of_room ∧ l = 5 := by
  sorry

end find_length_of_room_l826_826040


namespace range_of_a_l826_826973

theorem range_of_a (a : ℝ) :
  (∀ x, (x^2 - x ≤ 0 → 2^(1 - x) + a ≤ 0)) ↔ (a ≤ -2) := by
  sorry

end range_of_a_l826_826973


namespace mountain_number_count_l826_826489

def isMountainNumber (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 ≠ 0 ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧
  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧
  d2 > d1 ∧ d2 > d4 ∧ d3 > d1 ∧ d3 > d4

theorem mountain_number_count : {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ isMountainNumber n}.card = 3024 := 
  sorry

end mountain_number_count_l826_826489


namespace meal_cost_l826_826447

theorem meal_cost:
  ∀ (s c p k : ℝ), 
  (2 * s + 5 * c + 2 * p + 3 * k = 6.30) →
  (3 * s + 8 * c + 2 * p + 4 * k = 8.40) →
  (s + c + p + k = 3.15) :=
by
  intros s c p k h1 h2
  sorry

end meal_cost_l826_826447


namespace minimum_value_of_fraction_l826_826590

theorem minimum_value_of_fraction (x : ℝ) (hx : x > 10) : ∃ m, m = 30 ∧ ∀ y > 10, (y * y) / (y - 10) ≥ m :=
by 
  sorry

end minimum_value_of_fraction_l826_826590


namespace tan_x_eq_2_solution_l826_826467

noncomputable def solution_set_tan_2 : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2}

theorem tan_x_eq_2_solution :
  {x : ℝ | Real.tan x = 2} = solution_set_tan_2 :=
by
  sorry

end tan_x_eq_2_solution_l826_826467


namespace find_second_number_l826_826796

theorem find_second_number 
  (h1 : (20 + 40 + 60) / 3 = (10 + x + 45) / 3 + 5) :
  x = 50 :=
sorry

end find_second_number_l826_826796


namespace Anil_profit_in_rupees_l826_826173

def cost_scooter (C : ℝ) : Prop := 0.10 * C = 500
def profit (C P : ℝ) : Prop := P = 0.20 * C

theorem Anil_profit_in_rupees (C P : ℝ) (h1 : cost_scooter C) (h2 : profit C P) : P = 1000 :=
by
  sorry

end Anil_profit_in_rupees_l826_826173


namespace number_of_remaining_numbers_problem_solution_l826_826406

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826406


namespace susan_bought_36_items_l826_826774

noncomputable def cost_per_pencil : ℝ := 0.25
noncomputable def cost_per_pen : ℝ := 0.80
noncomputable def pencils_bought : ℕ := 16
noncomputable def total_spent : ℝ := 20.0

theorem susan_bought_36_items :
  ∃ (pens_bought : ℕ), pens_bought * cost_per_pen + pencils_bought * cost_per_pencil = total_spent ∧ pencils_bought + pens_bought = 36 := 
sorry

end susan_bought_36_items_l826_826774


namespace cube_has_8_vertices_l826_826296

-- Define a cube and its properties
def isCube (object : Type) : Prop :=
  ∃ (face_count : ℕ) (edge_count : ℕ) (vertex_count : ℕ), face_count = 6 ∧ edge_count = 12 ∧ vertex_count = 8

-- Theorem stating that a cube has 8 vertices
theorem cube_has_8_vertices (C : Type) (h : isCube C) : ∃ (vertex_count : ℕ), vertex_count = 8 :=
by
  obtain ⟨face_count, edge_count, vertex_count, h1, h2, h3⟩ := h
  use vertex_count
  exact h3

end cube_has_8_vertices_l826_826296


namespace range_of_a_if_f_increasing_l826_826676

noncomputable def f (x a : ℝ) : ℝ := x - (5 / x) - a * Real.log x

theorem range_of_a_if_f_increasing :
  (∀ x : ℝ, 1 ≤ x → Deriv (λ x, f x a) x ≥ 0) → a ≤ 2 * Real.sqrt 5 :=
by 
  intro h
  -- Proof will be here
  sorry

end range_of_a_if_f_increasing_l826_826676


namespace crayons_left_l826_826772

def initial_crayons : ℕ := 253
def lost_or_given_away_crayons : ℕ := 70
def remaining_crayons : ℕ := 183

theorem crayons_left (initial_crayons : ℕ) (lost_or_given_away_crayons : ℕ) (remaining_crayons : ℕ) :
  initial_crayons - lost_or_given_away_crayons = remaining_crayons :=
by {
  sorry
}

end crayons_left_l826_826772


namespace total_chapters_read_l826_826006

def books_read : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : books_read * chapters_per_book = 384 :=
by
  sorry

end total_chapters_read_l826_826006


namespace compute_nested_operation_l826_826461

def my_op (a b : ℚ) : ℚ := (a^2 - b^2) / (1 - a * b)

theorem compute_nested_operation : my_op 1 (my_op 2 (my_op 3 4)) = -18 := by
  sorry

end compute_nested_operation_l826_826461


namespace ratio_of_AB_to_focal_distance_l826_826170

theorem ratio_of_AB_to_focal_distance (p q : ℝ) (h_ellipse : ∀ x y, (x^2 / p^2) + (y^2 / q^2) = 1)
  (B_point : (0, q))
  (line_AC_parallel_to_xaxis : ∀ A C : ℝ × ℝ, A.2 = C.2)
  (foci : ∀ F1 F2 : ℝ × ℝ, F1.1 ∈ [B_point, C.1] ∧ F2.1 ∈ [A.1, B_point.1])
  (focal_distance : ℝ) (h_focal_distance : focal_distance = 2) :
  AB / focal_distance = 8 / 5 :=
by
  sorry

end ratio_of_AB_to_focal_distance_l826_826170


namespace Steven_has_more_peaches_l826_826715

variable (Steven_peaches : Nat) (Jill_peaches : Nat)
variable (h1 : Steven_peaches = 19) (h2 : Jill_peaches = 6)

theorem Steven_has_more_peaches : Steven_peaches - Jill_peaches = 13 :=
by
  sorry

end Steven_has_more_peaches_l826_826715


namespace find_years_in_future_l826_826811

theorem find_years_in_future 
  (S F : ℕ)
  (h1 : F = 4 * S + 4)
  (h2 : F = 44) :
  ∃ x : ℕ, F + x = 2 * (S + x) + 20 ∧ x = 4 :=
by 
  sorry

end find_years_in_future_l826_826811


namespace point_on_circle_l826_826259

theorem point_on_circle (a b : ℝ) 
  (h1 : (b + 2) * x + a * y + 4 = 0) 
  (h2 : a * x + (2 - b) * y - 3 = 0) 
  (parallel_lines : ∀ x y : ℝ, ∀ C1 C2 : ℝ, 
    (b + 2) * x + a * y + C1 = 0 ∧ a * x + (2 - b) * y + C2 = 0 → 
    - (b + 2) / a = - a / (2 - b)
  ) : a^2 + b^2 = 4 :=
sorry

end point_on_circle_l826_826259


namespace number_of_remaining_numbers_problem_solution_l826_826405

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826405


namespace calculate_flat_rate_shipping_l826_826558

noncomputable def flat_rate_shipping : ℝ :=
  17.00

theorem calculate_flat_rate_shipping
  (price_per_shirt : ℝ)
  (num_shirts : ℤ)
  (price_pack_socks : ℝ)
  (num_packs_socks : ℤ)
  (price_per_short : ℝ)
  (num_shorts : ℤ)
  (price_swim_trunks : ℝ)
  (num_swim_trunks : ℤ)
  (total_bill : ℝ)
  (total_items_cost : ℝ)
  (shipping_cost : ℝ) :
  price_per_shirt * num_shirts + 
  price_pack_socks * num_packs_socks + 
  price_per_short * num_shorts +
  price_swim_trunks * num_swim_trunks = total_items_cost →
  total_bill - total_items_cost = shipping_cost →
  total_items_cost > 50 → 
  0.20 * total_items_cost ≠ shipping_cost →
  flat_rate_shipping = 17.00 := 
sorry

end calculate_flat_rate_shipping_l826_826558


namespace rectangle_square_area_ratio_l826_826807

theorem rectangle_square_area_ratio (t : ℝ) :
  (let long_side := 1.2 * t in
   let short_side := 0.8 * t in
   let area_Q := long_side * short_side in
   let area_T := t * t in
   area_Q / area_T = 24 / 25) :=
by
  sorry

end rectangle_square_area_ratio_l826_826807


namespace horse_revolutions_l826_826143

theorem horse_revolutions (d1 d2 : ℝ) (N1 : ℕ) (h1 : d1 = 30) (h2 : d2 = 10) (h3 : N1 = 40) : 
  ∃ N2 : ℕ, N2 = 120 := 
by 
  -- Definitions of d1, d2, N1, and h'1, h'2, h'3 ensure the conditions are used
  use 120
  sorry

end horse_revolutions_l826_826143


namespace verify_recurrence_half_open_interval_interval_length_tends_to_zero_irrational_infinite_cont_frac_l826_826361

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

def continued_fraction_a (ω : ℝ) (n : ℕ) : ℤ :=
  match n with
  | 0 => integer_part ω
  | _ => integer_part (ℝ.inv $ fractional_part $ continued_fraction_a ω (n - 1))

-- recurrence relations
def recurrence_p (a : ℕ → ℤ) : ℕ → ℤ := sorry
def recurrence_q (a : ℕ → ℤ) : ℕ → ℤ := sorry

-- Verify conditions
theorem verify_recurrence 
  (ω : ℝ) 
  (hω : 0 ≤ ω ∧ ω < 1)
  (a : ℕ → ℤ := continued_fraction_a ω) 
  (p q : ℕ → ℤ := recurrence_p a, recurrence_q a)
  (n : ℕ)
  :
  p (n - 1) * q n - p n * q (n - 1) = (-1 : ℤ)^n := 
  sorry

-- Prove the half-open interval
theorem half_open_interval (k : ℕ → ℤ)
  :
  ∃ l r : ℝ, 
    (0 ≤ l ∧ l < 1) ∧ 
    (0 ≤ r ∧ r < 1) ∧ 
    ∀ ω, 
       (a1 0 ω = k0 ∧ a1 1 ω = k1 ∧ ... ∧ a1 n ω = kn)
       ↔
       (l ≤ ω ∧ ω < r) 
    := sorry

-- Verify interval length -> 0 as n -> infinity
theorem interval_length_tends_to_zero (k : ℕ → ℤ)
  :
  ∀ ε > 0, 
  ∃ N, 
    ∀ n > N, 
      length (
          {ω : ℝ | 
            ∀ ω, 
               (a1 0 ω = k0 ∧ a1 1 ω = k1 ∧ ... ∧ a1 n ω = kn)
             }
        ) < ε 
  := sorry

-- Irrational numbers infinite continued fraction, rational numbers finite continued fraction
theorem irrational_infinite_cont_frac 
  (ω : ℝ)
  :
  (irrational ω → fraction_length ω = ∞) ∧
  (rational ω → ∃ N, fraction_length ω = N) 
  := sorry

end verify_recurrence_half_open_interval_interval_length_tends_to_zero_irrational_infinite_cont_frac_l826_826361


namespace max_N_l826_826985

theorem max_N (N : ℕ) (A1 A2 A3 A4 : finset ℕ) (cond1 : ∀ x ∈ (finset.range (N+1)), ∀ y ∈ (finset.range (N+1)), ∃ i ∈ ({1, 2, 3, 4} : finset ℕ), x ∈ [A1, A2, A3, A4][i] ∧ y ∈ [A1, A2, A3, A4][i]) (hA1 : A1.card = 500) (hA2 : A2.card = 500) (hA3 : A3.card = 500) (hA4 : A4.card = 500) : N ≤ 833 :=
sorry

end max_N_l826_826985


namespace max_BF_minus_CF_l826_826709
open Real

theorem max_BF_minus_CF (A B C E D F : Point) 
  (h_right_triangle : right_triangle A B C)
  (h_AC : dist A C = 6)
  (h_E_on_AC : between A E C)
  (h_CE_2AE : 2 * dist A E = dist E C)
  (h_D_mid_AB : midpoint D A B)
  (h_F_on_BC : between B F C)
  (h_EDF_90 : ∠ D E F = 90°) :
  ∃ F : Point, (dist B F - dist F C = 2 * sqrt 3) :=
sorry

end max_BF_minus_CF_l826_826709


namespace water_height_in_cylinder_l826_826139

-- Definitions of conditions
def radius_cone : ℝ := 10
def height_cone : ℝ := 15
def radius_cylinder : ℝ := 20

-- The problem to solve: Prove that the height of water in the cylinder is 1.25 cm.
theorem water_height_in_cylinder :
  let volume_cone := (1/3) * Real.pi * radius_cone^2 * height_cone in
  let volume_cylinder := (volume_cone / (Real.pi * radius_cylinder^2)) in
  volume_cylinder = 1.25 :=
by
  sorry

end water_height_in_cylinder_l826_826139


namespace intersection_A_C_l826_826635

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := real.sqrt ((1 + x) * (2 - x))

noncomputable def g (x : ℝ) : ℝ := real.log (x - a)

def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def C : set ℝ := {x | 2^(x^2 - 2*x - 3) < 1}

theorem intersection_A_C :
  A ∩ C = {x | -1 < x ∧ x ≤ 2} :=
sorry

end intersection_A_C_l826_826635


namespace number_of_true_propositions_l826_826655

variables {a b c d : ℝ}

theorem number_of_true_propositions :
  (ab > 0) → (-c / a < -d / b) → (bc > ad) ∧
  (ab > 0) → (bc > ad) → (-c / a < -d / b) ∧
  (-c / a < -d / b) → (bc > ad) → (ab > 0) → 
  true :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  split
  { split; try { sorry } }; 
  exact true.intro

end number_of_true_propositions_l826_826655


namespace distance_P_to_outer_circle_l826_826137

theorem distance_P_to_outer_circle
  (r_large r_small : ℝ) 
  (h_tangent_inner : true) 
  (h_tangent_diameter : true) 
  (P : ℝ) 
  (O1P : ℝ)
  (O2P : ℝ := r_small)
  (O1O2 : ℝ := r_large - r_small)
  (h_O1O2_eq_680 : O1O2 = 680)
  (h_O2P_eq_320 : O2P = 320) :
  r_large - O1P = 400 :=
by
  sorry

end distance_P_to_outer_circle_l826_826137


namespace hyunji_candies_l826_826665

theorem hyunji_candies : 
  (let bags := 25 in let candies_per_bag := 16 in let total_candies := bags * candies_per_bag in total_candies / 2) = 200 := by
  sorry

end hyunji_candies_l826_826665


namespace final_price_is_correct_l826_826769

def cost_cucumber : ℝ := 5
def cost_tomato : ℝ := cost_cucumber - 0.2 * cost_cucumber
def cost_bell_pepper : ℝ := cost_cucumber + 0.5 * cost_cucumber
def total_cost_before_discount : ℝ := 2 * cost_tomato + 3 * cost_cucumber + 4 * cost_bell_pepper
def final_price : ℝ := total_cost_before_discount - 0.1 * total_cost_before_discount

theorem final_price_is_correct : final_price = 47.7 := sorry

end final_price_is_correct_l826_826769


namespace amanda_average_speed_l826_826553

def amanda_distance1 : ℝ := 450
def amanda_time1 : ℝ := 7.5
def amanda_distance2 : ℝ := 420
def amanda_time2 : ℝ := 7

def total_distance : ℝ := amanda_distance1 + amanda_distance2
def total_time : ℝ := amanda_time1 + amanda_time2
def expected_average_speed : ℝ := 60

theorem amanda_average_speed :
  (total_distance / total_time) = expected_average_speed := by
  sorry

end amanda_average_speed_l826_826553


namespace elevation_equals_depression_l826_826449

variables (a b : Type) (c d : ℝ)

def elevation_angle (a b : Type) : ℝ := c
def depression_angle (a b : Type) : ℝ := d

theorem elevation_equals_depression :
  elevation_angle a b = depression_angle a b := 
  sorry

end elevation_equals_depression_l826_826449


namespace distinct_ordered_pairs_count_l826_826291

theorem distinct_ordered_pairs_count :
  { (m, n) : ℕ × ℕ | (0 < m) ∧ (0 < n) ∧ (1 / m + 1 / n = 1 / 3) }.to_finset.card = 3 :=
by
  sorry

end distinct_ordered_pairs_count_l826_826291


namespace expand_product_l826_826950

theorem expand_product (x : ℝ): (x + 4) * (x - 5 + 2) = x^2 + x - 12 :=
by 
  sorry

end expand_product_l826_826950


namespace calendar_matrix_sum_l826_826525

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![5, 6, 7], 
    ![8, 9, 10], 
    ![11, 12, 13]]

def modified_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![m 0 2, m 0 1, m 0 0], 
    ![m 1 0, m 1 1, m 1 2], 
    ![m 2 2, m 2 1, m 2 0]]

def diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def edge_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 1 + m 0 2 + m 2 0 + m 2 1

def total_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  diagonal_sum m + edge_sum m

theorem calendar_matrix_sum :
  total_sum (modified_matrix initial_matrix) = 63 :=
by
  sorry

end calendar_matrix_sum_l826_826525


namespace find_angles_l826_826625

theorem find_angles (A B : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (h : sin A * cos B + sqrt (2 * sin A) * sin B = (3 * sin A + 1) / sqrt 5) :
  A = π / 6 ∧ B = π / 2 - Real.arcsin (sqrt 5 / 5) :=
  sorry

end find_angles_l826_826625


namespace binomial_coefficient_x_pow_2_in_expansion_l826_826332

theorem binomial_coefficient_x_pow_2_in_expansion :
  let C := Nat.choose in
  let T := λ r : ℕ, (-2)^r * C 5 r * (x : ℝ)^(5 - 3/2 * r) in
  T 2 = 40 * (x : ℝ)^2 := 
by 
  sorry

end binomial_coefficient_x_pow_2_in_expansion_l826_826332


namespace measure_segment_with_ruler_l826_826764

theorem measure_segment_with_ruler :
  ∃ (a b c d e : ℕ), a = 0 ∧ b = 2 ∧ c = 5 ∧ d = 3 ∧ e = 6 ∧ e = a + d + d :=
by
  use 0, 2, 5, 3, 6
  split; reflexivity; sorry

end measure_segment_with_ruler_l826_826764


namespace S6_eq_24_l826_826468

-- Definitions based on the conditions provided
def is_arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def S : ℕ → ℝ := sorry  -- Sum of the first n terms of some arithmetic sequence

-- Given conditions
axiom S2_eq_2 : S 2 = 2
axiom S4_eq_10 : S 4 = 10

-- The main theorem to prove
theorem S6_eq_24 : S 6 = 24 :=
by 
  sorry  -- Proof is omitted

end S6_eq_24_l826_826468


namespace speed_of_train_in_km_per_hr_l826_826891

-- Definitions for the condition
def length_of_train : ℝ := 180 -- in meters
def time_to_cross_pole : ℝ := 9 -- in seconds

-- Conversion factor
def meters_per_second_to_kilometers_per_hour (speed : ℝ) := speed * 3.6

-- Proof statement
theorem speed_of_train_in_km_per_hr : 
  meters_per_second_to_kilometers_per_hour (length_of_train / time_to_cross_pole) = 72 := 
by
  sorry

end speed_of_train_in_km_per_hr_l826_826891


namespace jump_difference_l826_826804

def frog_jump := 39
def grasshopper_jump := 17

theorem jump_difference :
  frog_jump - grasshopper_jump = 22 := by
  sorry

end jump_difference_l826_826804


namespace fourth_number_ninth_row_eq_206_l826_826465

-- Define the first number in a given row
def first_number_in_row (i : Nat) : Nat :=
  2 + 4 * 6 * (i - 1)

-- Define the number in the j-th position in the i-th row
def number_in_row (i j : Nat) : Nat :=
  first_number_in_row i + 4 * (j - 1)

-- Define the 9th row and fourth number in it
def fourth_number_ninth_row : Nat :=
  number_in_row 9 4

-- The theorem to prove the fourth number in the 9th row is 206
theorem fourth_number_ninth_row_eq_206 : fourth_number_ninth_row = 206 := by
  sorry

end fourth_number_ninth_row_eq_206_l826_826465


namespace max_value_f_l826_826251

noncomputable def f (x y : ℝ) := real.sqrt (8 * y - 6 * x + 50) + real.sqrt (8 * y + 6 * x + 50)

theorem max_value_f :
  ∃ x y : ℝ, x ^ 2 + y ^ 2 = 25 ∧ f x y = 6 * real.sqrt 10 :=
by
  sorry

end max_value_f_l826_826251


namespace polygon_intersection_max_l826_826820

theorem polygon_intersection_max (m : ℕ) (A1 A2 : Type) [convex_polygon A1] [convex_polygon A2] 
  [has_sides A1 (fin m)] [has_sides A2 (fin (m + 2))]:
  (∀ s₁ ∈ sides A1, ∀ s₂ ∈ sides A2, intersects s₁ s₂ ∧ ¬(overlaps s₁ s₂)) → 
  max_intersections A1 A2 = m^2 + 2 * m :=
sorry

end polygon_intersection_max_l826_826820


namespace find_y_positive_monotone_l826_826966

noncomputable def y (y : ℝ) : Prop :=
  0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 ∧ y = 12

theorem find_y_positive_monotone : ∃ y : ℝ, 0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 := by
  sorry

end find_y_positive_monotone_l826_826966


namespace prob_exactly_four_twos_l826_826945

-- Define the probability of success (rolling a 2)
def p (k : ℕ) (n : ℕ) : ℚ := (choose n k) * ((1/6)^k) * ((5/6)^(n-k))

-- Define the specific instance for the problem
def probability_ex4_dice_show_2 : ℚ := p 4 8

-- The main assertion
theorem prob_exactly_four_twos : probability_ex4_dice_show_2 ≈ 0.026 :=
by
  sorry -- Proof not provided, just the statement.

end prob_exactly_four_twos_l826_826945


namespace probability_of_fork_spoon_knife_different_colors_l826_826688

noncomputable def total_ways_to_choose_3_items (n : ℕ) : ℕ :=
  n.choose 3

def ways_to_choose_one_of_each : ℕ :=
  8 * 8 * 8 * 2

theorem probability_of_fork_spoon_knife_different_colors :
  let n := 48
  let total_ways := total_ways_to_choose_3_items n
  let favorable_ways := ways_to_choose_one_of_each
  total_ways ≠ 0 →
  (favorable_ways / total_ways : ℚ) = (32 : ℚ) / 541 :=
by
  intros
  let total_ways := total_ways_to_choose_3_items 48
  let favorable_ways := ways_to_choose_one_of_each
  have h_total_ways : total_ways = 17296 := sorry
  have h_favorable_ways : favorable_ways = 1024 := sorry
  rw [h_total_ways, h_favorable_ways]
  norm_num
  rw [div_eq_div_iff]
  norm_num
  sorry

end probability_of_fork_spoon_knife_different_colors_l826_826688


namespace inequality_proof_l826_826776

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b) * (a + c) ≥ 2 * Real.sqrt (a * b * c * (a + b + c)) := 
sorry

end inequality_proof_l826_826776


namespace sum_log2_sequence_l826_826596

theorem sum_log2_sequence (a : ℕ → ℝ) (n : ℕ) 
(h : ∀ n, a n = (n + 1) * 2 ^ n) : 
  (∑ i in finset.range 10, real.log2 ((a i) / (i + 1))) = 55 :=
begin
  sorry
end

end sum_log2_sequence_l826_826596


namespace range_of_a_if_f1_lt_2_number_of_zeros_of_g_l826_826229

-- Given condition: a ∈ ℝ and the function f
variable (a : ℝ)
def f (x : ℝ) : ℝ := log 2 ((1 / x) + a)

-- Question (1)
theorem range_of_a_if_f1_lt_2 (h : f a 1 < 2) : a ∈ set.Ioo (-1 : ℝ) 3 :=
sorry

-- Question (2)
def g (x : ℝ) : ℝ := f a x - log 2 ((a-4) * x + 2 * a - 5)

-- State the cases for the number of zeros of g(x)
theorem number_of_zeros_of_g :
  (a ≤ 1 → ∀ x, g a x ≠ 0) ∧
  (1 < a ∧ a ≤ 2 → ∃ x, g a x = 0 ∧ ∀ y, y ≠ x → g a y ≠ 0) ∧
  (a = 3 → ∃ x, g a x = 0 ∧ ∀ y, y ≠ x → g a y ≠ 0) ∧
  (a = 4 → ∃ x, g a x = 0 ∧ ∀ y, y ≠ x → g a y ≠ 0) ∧
  (a > 2 ∧ a ≠ 3 ∧ a ≠ 4 → ∃ x y, x ≠ y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z, z ≠ x ∧ z ≠ y → g a z ≠ 0) :=
sorry

end range_of_a_if_f1_lt_2_number_of_zeros_of_g_l826_826229


namespace remaining_numbers_after_erasure_l826_826391

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826391


namespace sin_double_alpha_l826_826603

theorem sin_double_alpha (α β : ℝ) (h1 : 0 < β) (h2 : β < α) (h3 : α < π / 4)
(h4 : cos (α - β) = 12 / 13) (h5 : sin (α + β) = 4 / 5) :
sin (2 * α) = 63 / 65 := by
  sorry

end sin_double_alpha_l826_826603


namespace problem_statement_l826_826667

variable {a b c d : ℝ}

theorem problem_statement
  (pos_a : a > 0) 
  (pos_b : b > 0) 
  (pos_c : c > 0) 
  (pos_d : d > 0) 
  (abcd_eq : a * b * c * d = 1) :
  (ab_plus_one_div_a_plus_one : (a * b + 1) / (a + 1)) + 
  (bc_plus_one_div_b_plus_one : (b * c + 1) / (b + 1)) + 
  (cd_plus_one_div_c_plus_one : (c * d + 1) / (c + 1)) + 
  (da_plus_one_div_d_plus_one : (d * a + 1) / (d + 1)) >= 4 := 
sorry

end problem_statement_l826_826667


namespace alyssas_weekly_allowance_l826_826165

-- Define the constants and parameters
def spent_on_movies (A : ℝ) := 0.5 * A
def spent_on_snacks (A : ℝ) := 0.2 * A
def saved_for_future (A : ℝ) := 0.25 * A

-- Define the remaining allowance after expenses
def remaining_allowance_after_expenses (A : ℝ) := A - spent_on_movies A - spent_on_snacks A - saved_for_future A

-- Define Alyssa's allowance given the conditions
theorem alyssas_weekly_allowance : ∀ (A : ℝ), 
  remaining_allowance_after_expenses A = 12 → 
  A = 240 :=
by
  -- Proof omitted
  sorry

end alyssas_weekly_allowance_l826_826165


namespace extreme_points_of_f_range_of_m_l826_826645

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x + m * (x - 1)^2

theorem extreme_points_of_f (m : ℝ) :
  (m = 0 → ∀ x > 0, f' x m > 0) ∧
  (0 < m ∧ m ≤ 2 → ∀ x > 0, f' x m ≥ 0) ∧
  (m > 2 → ∃ x1 x2 > 0, x1 < x2 ∧ f' x1 m = 0 ∧ f' x2 m = 0) ∧
  (m < 0 → ∃ x > 0, f' x m = 0) :=
sorry

theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, f x m ≥ 0) ↔ (0 ≤ m) :=
sorry

end extreme_points_of_f_range_of_m_l826_826645


namespace pythagorean_triple_B_l826_826106

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l826_826106


namespace T9_equals_257_l826_826360

-- Define the initial conditions
def A1 : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 4
| (n+4) := A1 n.succ + A2 n.succ + A3 n.succ - A1 n

def A2 : ℕ → ℕ
| 1 := 1
| 2 := 1
| 3 := 1
| (n+4) := A1 n.succ

def A3 : ℕ → ℕ
| 1 := 0
| 2 := 1
| 3 := 1
| (n+4) := A2 n.succ

def T : ℕ → ℕ
| n := A1 n + A2 n + A3 n

theorem T9_equals_257 : T 9 = 257 := by
  sorry

end T9_equals_257_l826_826360


namespace numbers_left_on_board_l826_826424

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826424


namespace log2_bounds_sum_l826_826935

theorem log2_bounds_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : (a : ℝ) < Real.log 50 / Real.log 2) (h4 : Real.log 50 / Real.log 2 < (b : ℝ)) :
  a + b = 11 :=
sorry

end log2_bounds_sum_l826_826935


namespace bulbs_per_pack_l826_826436

theorem bulbs_per_pack : 
  let bedroom_bulbs := 2
  let bathroom_bulbs := 1
  let kitchen_bulbs := 1
  let basement_bulbs := 4
  let garage_bulbs := (bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs) / 2
  let total_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs + garage_bulbs
  let packs := 6
in 
  total_bulbs / packs = 2 := 
by 
  sorry

end bulbs_per_pack_l826_826436


namespace convex_17_gon_3_color_l826_826018

theorem convex_17_gon_3_color {
  V : Type
  (P : set V)
  (h_card : P.card = 17)
  (color : V → V → fin 3)
  (h_convex : convex_hull P = P)
  (h_total : ∀ v1 v2 v3 ∈ P, ∃ c, color v1 v2 = c ∧ color v2 v3 = c ∧ color v3 v1 = c) :
  ∃ (v1 v2 v3 : V), v1 ∈ P ∧ v2 ∈ P ∧ v3 ∈ P ∧ 
                      color v1 v2 = color v2 v3 ∧ color v2 v3 = color v3 v1 := 
sorry

end convex_17_gon_3_color_l826_826018


namespace find_roots_correct_l826_826591

noncomputable def find_roots (p : Polynomial ℝ) : Set ℝ :=
  {x : ℝ | p.eval x = 0}

def p : Polynomial ℝ := Polynomial.C 5 * Polynomial.X ^ 4 - Polynomial.C 28 * Polynomial.X ^ 3
  + Polynomial.C 49 * Polynomial.X ^ 2 - Polynomial.C 28 * Polynomial.X + Polynomial.C 5

theorem find_roots_correct : find_roots p = {2, 1 / 2, (5 + Real.sqrt 21) / 5, (5 - Real.sqrt 21) / 5} :=
by
  sorry

end find_roots_correct_l826_826591


namespace roots_square_sum_l826_826113

theorem roots_square_sum (a b : ℝ) 
  (h1 : a^2 - 4 * a + 4 = 0) 
  (h2 : b^2 - 4 * b + 4 = 0) 
  (h3 : a = b) :
  a^2 + b^2 = 8 := 
sorry

end roots_square_sum_l826_826113


namespace doughnut_cost_l826_826008

theorem doughnut_cost:
  ∃ (D C : ℝ), 
    3 * D + 4 * C = 4.91 ∧ 
    5 * D + 6 * C = 7.59 ∧ 
    D = 0.45 :=
by
  sorry

end doughnut_cost_l826_826008


namespace mirror_reflection_SHAVE_correct_l826_826859

theorem mirror_reflection_SHAVE_correct :
  ∃ x, x = "EVAHS" ↔ reverse "SHAVE" = x :=
by
  sorry

end mirror_reflection_SHAVE_correct_l826_826859


namespace is_not_necessarily_parallelogram_but_isosceles_trapezoid_l826_826345

-- Define the conditions for the quadrilateral ABCD
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable {AB : A → B → ℝ}
variable {CD : C → D → ℝ}
variable (AD_eq_BC : A → D → B → C → Prop)

-- Define the properties of the quadrilateral
def quadrilateral_properties (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] (AB CD : A → B → ℝ) (AD_eq_BC : A → D → B → C → Prop) :=
  AB = CD ∧ AD_eq_BC

-- Define the isosceles trapezoid property
def is_isosceles_trapezoid (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] (AB CD : A → B → ℝ) (AD_eq_BC : A → D → B → C → Prop) :=
  quadrilateral_properties A B C D AB CD AD_eq_BC

-- State the theorem (proof not included)
theorem is_not_necessarily_parallelogram_but_isosceles_trapezoid 
  (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
  (AB CD : A → B → ℝ) (AD_eq_BC : A → D → B → C → Prop) :
  is_isosceles_trapezoid A B C D AB CD (AD_eq_BC) := sorry

end is_not_necessarily_parallelogram_but_isosceles_trapezoid_l826_826345


namespace remaining_numbers_count_l826_826416

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826416


namespace number_of_students_in_line_l826_826030

theorem number_of_students_in_line (n_in_front : ℕ) (n_behind : ℕ) (taehyung : ℕ) : 
  n_in_front = 9 → n_behind = 16 → taehyung = 1 → n_in_front + taehyung + n_behind = 26 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end number_of_students_in_line_l826_826030


namespace mandy_used_nutmeg_l826_826375

theorem mandy_used_nutmeg (x : ℝ) (h1 : 0.67 = x + 0.17) : x = 0.50 :=
  by
  sorry

end mandy_used_nutmeg_l826_826375


namespace total_cost_after_discounts_and_taxes_l826_826200

theorem total_cost_after_discounts_and_taxes :
  let original_price_iphone := 820
      discount_iphone := 0.15
      original_price_iwatch := 320
      discount_iwatch := 0.10
      original_price_ipad := 520
      discount_ipad := 0.05
      promotion_discount := 0.03
      tax_rate_iphone := 0.07
      tax_rate_iwatch := 0.05
      tax_rate_ipad := 0.06
      cashback_discount := 0.02
      discounted_price_iphone := original_price_iphone * (1 - discount_iphone)
      discounted_price_iwatch := original_price_iwatch * (1 - discount_iwatch)
      discounted_price_ipad := original_price_ipad * (1 - discount_ipad)
      total_price_before_promotion := discounted_price_iphone + discounted_price_iwatch + discounted_price_ipad
      price_after_promotion := total_price_before_promotion * (1 - promotion_discount)
      tax_iphone := discounted_price_iphone * tax_rate_iphone
      tax_iwatch := discounted_price_iwatch * tax_rate_iwatch
      tax_ipad := discounted_price_ipad * tax_rate_ipad
      total_tax := tax_iphone + tax_iwatch + tax_ipad
      price_after_tax := price_after_promotion + total_tax
      final_price := price_after_tax * (1 - cashback_discount)
  in final_price = 1496.91 :=
by
  -- The proof steps are omitted.
  sorry

end total_cost_after_discounts_and_taxes_l826_826200


namespace prove_solution_l826_826589

theorem prove_solution : ∃ x y : ℝ, 
  let v₁ := (3: ℝ, 1: ℝ)
      v₂ := (9: ℝ, -7: ℝ)
      v₃ := (2: ℝ, -2: ℝ)
      v₄ := (-3: ℝ, 4: ℝ) in
  (v₁.1 + x * v₂.1 = v₃.1 + y * v₄.1) ∧ 
  (v₁.2 + x * v₂.2 = v₃.2 + y * v₄.2) → 
  (x = -13/15 ∧ y = 34/15) :=
begin
  sorry
end

end prove_solution_l826_826589


namespace finding_angle_tetrahedral_pyramid_l826_826327

def regular_tetrahedral_pyramid 
  (A B C D : Type) 
  (edge_length : ℝ)
  (angle : ℝ) 
  (base_plane : Type) 
  (lateral_plane : Type) 
  (ABC_face : Type) : Prop :=
    ∀ (lateral_edge : ℝ),
    regular_pyramid A B C D ∧
    angle_between_lateral_edge_and_base_plane lateral_edge base_plane = 
    angle_between_lateral_edge_and_other_lateral_face lateral_edge ABC_face →
    angle = arctan (sqrt (3 / 2))

theorem finding_angle_tetrahedral_pyramid 
  (A B C D : Type) 
  (edge_length : ℝ) 
  (angle : ℝ)
  (base_plane : Type) 
  (lateral_plane : Type) 
  (ABC_face : Type) 
  (lateral_edge : ℝ)
  (h : Prop)
  [h : regular_tetrahedral_pyramid A B C D edge_length angle base_plane lateral_plane ABC_face] :
  angle = arctan (sqrt (3 / 2)) := 
sorry

end finding_angle_tetrahedral_pyramid_l826_826327


namespace probability_correct_l826_826821

noncomputable def probability_P_plus_S_is_two_less_than_multiple_of_7 : ℚ :=
  let choices : ℕ := 60.choose 2
  let valid_choices : ℕ := 444
  valid_choices / choices

theorem probability_correct :
  probability_P_plus_S_is_two_less_than_multiple_of_7 = 148 / 590 := by
  sorry

end probability_correct_l826_826821


namespace minimum_value_of_f_l826_826648

def f (x a : ℝ) : ℝ := abs (x + 1) + abs (a * x + 1)

theorem minimum_value_of_f (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 / 2) →
  (∃ x : ℝ, f x a = 3 / 2) →
  (a = -1 / 2 ∨ a = -2) :=
by
  intros h1 h2
  sorry

end minimum_value_of_f_l826_826648


namespace linear_independence_exp_sin_cos_l826_826788

theorem linear_independence_exp_sin_cos (α β : ℝ) (hβ : β ≠ 0) :
  ∀ (α1 α2 : ℝ), (∀ x : ℝ, α1 * (Real.exp (α * x) * Real.sin (β * x)) +
                              α2 * (Real.exp (α * x) * Real.cos (β * x)) = 0) →
                 α1 = 0 ∧ α2 = 0 :=
by
  intro α1 α2 h,
  have h1 := h 0,
  simp at h1,
  cases hβ
  sorry

end linear_independence_exp_sin_cos_l826_826788


namespace alpha_value_l826_826315

theorem alpha_value (k : ℤ) (α : ℝ) :
  (∀ x : ℝ, f(x) = f(-x)) → α = k * π - π / 6 :=
by
  let f := λ (x : ℝ), sqrt 3 * cos (2 * x + α) - sin (2 * x + α)
  sorry

end alpha_value_l826_826315


namespace range_of_a_l826_826981

noncomputable def A (a : ℝ) : Set ℝ := { x | 3 + a ≤ x ∧ x ≤ 4 + 3 * a }
noncomputable def B : Set ℝ := { x | -4 ≤ x ∧ x < 5 }

theorem range_of_a (a : ℝ) : A a ⊆ B ↔ -1/2 ≤ a ∧ a < 1/3 :=
  sorry

end range_of_a_l826_826981


namespace max_quotient_l826_826301

theorem max_quotient (a b : ℕ) 
  (h1 : 400 ≤ a) (h2 : a ≤ 800) 
  (h3 : 400 ≤ b) (h4 : b ≤ 1600) 
  (h5 : a + b ≤ 2000) 
  : b / a ≤ 4 := 
sorry

end max_quotient_l826_826301


namespace range_of_x_for_f_lt_0_l826_826670

noncomputable def f (x : ℝ) : ℝ := x^2 - x^(1/2)

theorem range_of_x_for_f_lt_0 :
  {x : ℝ | f x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end range_of_x_for_f_lt_0_l826_826670


namespace recommended_water_intake_l826_826077

theorem recommended_water_intake (current_intake : ℕ) (increase_percentage : ℚ) (recommended_intake : ℕ) : 
  current_intake = 15 → increase_percentage = 0.40 → recommended_intake = 21 :=
by
  intros h1 h2
  sorry

end recommended_water_intake_l826_826077


namespace spring_work_done_l826_826311

theorem spring_work_done (F : ℝ) (l : ℝ) (stretched_length : ℝ) (k : ℝ) (W : ℝ) 
  (hF : F = 10) (hl : l = 0.1) (hk : k = F / l) (h_stretched_length : stretched_length = 0.06) : 
  W = 0.18 :=
by
  sorry

end spring_work_done_l826_826311


namespace ladder_wood_length_l826_826716

theorem ladder_wood_length (rung_length spacing_per_rung : ℝ)
  (climb_height_ft: ℝ) (climb_height_in: ℝ) 
  (ft_to_inch: 𝔽) :
  rung_length = 18 →
  spacing_per_rung = 6 →
  climb_height_ft = 50 →
  climb_height_in = climb_height_ft * ft_to_inch →
  climb_height_in = 600 →
  ft_to_inch = 12 →
  37.5 = (climb_height_in / (rung_length + spacing_per_rung)) * (rung_length / ft_to_inch) :=
by
  intros
  sorry

end ladder_wood_length_l826_826716


namespace area_of_semicircle_is_correct_l826_826127

-- Given: A 1x3 rectangle inscribed in a semicircle with the longer side on the diameter.
def radius_of_semicircle (diameter : ℝ) : ℝ := diameter / 2
def area_of_semicircle (r : ℝ) : ℝ := (1 / 2) * (π * r^2)

-- The diameter is the longer side of the rectangle, which is 3 units.
def diameter : ℝ := 3

-- Therefore, the radius of the semicircle is:
def r : ℝ := radius_of_semicircle diameter

-- Prove that the area of the semicircle is 9π/8
theorem area_of_semicircle_is_correct : area_of_semicircle r = (9 * π) / 8 :=
by
  -- Calculation here
  sorry

end area_of_semicircle_is_correct_l826_826127


namespace numbers_left_on_board_l826_826427

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826427


namespace women_at_each_table_l826_826159

theorem women_at_each_table (W : ℕ) (h1 : ∃ W, ∀ i : ℕ, (i < 7) → W + 2 = 7 * W + 14) (h2 : 7 * W + 14 = 63) : W = 7 :=
by
  sorry

end women_at_each_table_l826_826159


namespace darius_age_is_8_l826_826717

def age_of_darius (jenna_age darius_age : ℕ) : Prop :=
  jenna_age = darius_age + 5

theorem darius_age_is_8 (jenna_age darius_age : ℕ) (h1 : jenna_age = darius_age + 5) (h2: jenna_age = 13) : 
  darius_age = 8 :=
by
  sorry

end darius_age_is_8_l826_826717


namespace sufficient_but_not_necessary_l826_826976

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) :
  (0 ≤ a * b ∧ a * b ≤ 1) ∧ ¬(∀ a b : ℝ, (0 ≤ a * b ∧ a * b ≤ 1) → (0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1)) :=
by
  split
  sorry
  sorry

end sufficient_but_not_necessary_l826_826976


namespace probability_AEMC9_is_1_over_84000_l826_826708

-- Define possible symbols for each category.
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def nonVowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

-- Define the total number of possible license plates.
def totalLicensePlates : Nat := 
  (vowels.length) * (vowels.length - 1) * 
  (nonVowels.length) * (nonVowels.length - 1) * 
  (digits.length)

-- Define the number of favorable outcomes.
def favorableOutcomes : Nat := 1

-- Define the probability calculation.
noncomputable def probabilityAEMC9 : ℚ := favorableOutcomes / totalLicensePlates

-- The theorem to prove.
theorem probability_AEMC9_is_1_over_84000 :
  probabilityAEMC9 = 1 / 84000 := by
  sorry

end probability_AEMC9_is_1_over_84000_l826_826708


namespace response_rate_percentage_l826_826531

theorem response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ)
  (h1 : responses_needed = 240) (h2 : questionnaires_mailed = 400) : 
  (responses_needed : ℝ) / (questionnaires_mailed : ℝ) * 100 = 60 := 
by 
  sorry

end response_rate_percentage_l826_826531


namespace perpendicular_midpoint_conditions_l826_826904

-- Assuming A, B, C, D are points represented by vectors in a 3D space.

structure Point :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

def vector_sub (P Q : Point) : Point :=
  ⟨P.x - Q.x, P.y - Q.y, P.z - Q.z⟩

def dot_product (P Q : Point) : ℝ :=
  (P.x * Q.x) + (P.y * Q.y) + (P.z * Q.z)

theorem perpendicular_midpoint_conditions
    (A B C D : Point)
    (M : Point := midpoint A B)
    (N : Point := midpoint C D)
    (MN : Point := vector_sub N M) :
    (dot_product MN (vector_sub A B) = 0 ∧ dot_product MN (vector_sub C D) = 0) ↔
    (dot_product (vector_sub A C) (vector_sub A C) = dot_product (vector_sub B D) (vector_sub B D) ∧
     dot_product (vector_sub A D) (vector_sub A D) = dot_product (vector_sub B C) (vector_sub B C)) := sorry

end perpendicular_midpoint_conditions_l826_826904


namespace fixed_point_for_line_l826_826939

theorem fixed_point_for_line (x y a : ℝ) : 
    (∀ a : ℝ, (a - 1) * x - y + 2 * a + 1 = 0) → 
    x = -2 ∧ y = 3 :=
by 
  intro h
  have h1 : (x + 2) = 0 := by 
    specialize (h 0)
    linarith
  have h2 : (-x - y + 1) = 0 := by 
    specialize (h 1)
    linarith
  simp at h1 h2
  exact ⟨h1, h2⟩

end fixed_point_for_line_l826_826939


namespace good_carrots_l826_826088

-- Definitions
def vanessa_carrots : ℕ := 17
def mother_carrots : ℕ := 14
def bad_carrots : ℕ := 7

-- Proof statement
theorem good_carrots : (vanessa_carrots + mother_carrots) - bad_carrots = 24 := by
  sorry

end good_carrots_l826_826088


namespace abs_diff_max_min_l826_826096

noncomputable def min_and_max_abs_diff (x : ℝ) : ℝ :=
|x - 2| + |x - 3| - |x - 1|

theorem abs_diff_max_min (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) :
  ∃ (M m : ℝ), M = 0 ∧ m = -1 ∧
    M = max (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) ∧ 
    m = min (min_and_max_abs_diff 2) (min_and_max_abs_diff 3) :=
by
  use [0, -1]
  split
  case inl => sorry
  case inr => sorry

end abs_diff_max_min_l826_826096


namespace melanie_more_turnips_l826_826004

theorem melanie_more_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) :
  melanie_turnips - benny_turnips = 26 := by
  sorry

end melanie_more_turnips_l826_826004


namespace product_of_two_numbers_is_320_l826_826060

theorem product_of_two_numbers_is_320 (x y : ℕ) (h1 : x + y = 36) (h2 : x - y = 4) (h3 : x = 5 * (y / 4)) : x * y = 320 :=
by {
  sorry
}

end product_of_two_numbers_is_320_l826_826060


namespace T1_T2_T3_all_theorems_hold_l826_826619

-- Define the pibs and maas as sets
constant Pib : Type
constant Maa : Type

-- Assume we have 4 pibs
@[instance] axiom FourPibs : Fintype Pib
axiom pibs_are_4 : ∀ s : Finset Pib, s.card = 4

-- Assume every pib is a set of maas
axiom P1 : Pib → Finset Maa

-- Assume any two different pibs have exactly one common maa
axiom P2 : ∀ (p1 p2 : Pib), p1 ≠ p2 → ∃! (m : Maa), m ∈ P1 p1 ∧ m ∈ P1 p2

-- Assume every maa belongs to exactly two pibs
axiom P3 : ∀ m : Maa, ∃ (p1 p2 : Pib), p1 ≠ p2 ∧ m ∈ P1 p1 ∧ m ∈ P1 p2

-- Theorems to be proved
theorem T1 : ∃! (S : Finset Maa), S.card = 6 := sorry
theorem T2 : ∀ (p : Pib), (P1 p).card = 3 := sorry
theorem T3 : ∀ (m : Maa), ∃ (m' : Maa), m ≠ m' ∧ ∀ (p : Pib), (m ∈ P1 p → m' ∉ P1 p) := sorry

-- Prove that the three theorems hold given the axioms
theorem all_theorems_hold : T1 ∧ T2 ∧ T3 := sorry

end T1_T2_T3_all_theorems_hold_l826_826619


namespace positive_difference_45_y_is_16_l826_826798

-- Define the condition
def average_condition (y : ℝ) : Prop :=
  (45 + y) / 2 = 53

-- Define the positive difference function
def positive_difference (a b : ℝ) : ℝ :=
  if a > b then a - b else b - a

-- The main theorem to prove the positive difference is 16 given the condition
theorem positive_difference_45_y_is_16 (y : ℝ) (h : average_condition y) : positive_difference 45 y = 16 := 
by
  sorry

end positive_difference_45_y_is_16_l826_826798


namespace triangle_is_right_triangle_l826_826319

theorem triangle_is_right_triangle 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : b - a * Real.cos B = a * Real.cos C - c) 
  (h₂ : ∀ (angle : ℝ), 0 < angle ∧ angle < π) : A = π / 2 := 
sorry

end triangle_is_right_triangle_l826_826319


namespace percent_voters_for_candidate_A_l826_826114

theorem percent_voters_for_candidate_A :
  ∀ (total_voters : ℝ) (dem_percent rep_percent dem_voted_percent rep_voted_percent : ℝ)
  (n1 : 60) (n2 : 40)
  (condition1 : dem_percent = 0.60) 
  (condition2 : rep_percent = 0.40)
  (condition3 : dem_voted_percent = 0.75)
  (condition4 : rep_voted_percent = 0.20),
  100 * ((dem_voted_percent * (total_voters * dem_percent) + rep_voted_percent * (total_voters * rep_percent)) / total_voters) = 53 := by
  intros
  simp only
  sorry

end percent_voters_for_candidate_A_l826_826114


namespace inequality_solution_1_inequality_solution_2_l826_826118

-- Definition for part 1
theorem inequality_solution_1 (x : ℝ) : x^2 + 3*x - 4 > 0 ↔ x > 1 ∨ x < -4 :=
sorry

-- Definition for part 2
theorem inequality_solution_2 (x : ℝ) : (1 - x) / (x - 5) ≥ 1 ↔ 3 ≤ x ∧ x < 5 :=
sorry

end inequality_solution_1_inequality_solution_2_l826_826118


namespace num_unique_sum_4_l826_826290

-- Definition of the set
def my_set : Set ℕ := {2, 6, 10, 14, 18, 22, 26, 30}

-- Function to count the number of different integers expressible as the sum of four distinct elements from the given set
noncomputable def count_unique_sums (s : Set ℕ) (k : ℕ) : ℕ :=
  {n | ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = n}.toFinset.card

-- Statement of the problem in Lean
theorem num_unique_sum_4 (h : my_set.card = 8) : count_unique_sums my_set 4 = 17 := by
  sorry

end num_unique_sum_4_l826_826290


namespace sin_double_angle_plus_pi_over_2_l826_826626

theorem sin_double_angle_plus_pi_over_2 (θ : ℝ) (h : Real.cos θ = -1/3) :
  Real.sin (2 * θ + Real.pi / 2) = -7/9 :=
sorry

end sin_double_angle_plus_pi_over_2_l826_826626


namespace remainder_poly_div_l826_826501

theorem remainder_poly_div (p : ℚ[X]) (h1 : p.eval (-1) = 3) (h2 : p.eval (-3) = -4) :
  ∃ (a b : ℚ), (p % ((X + 1) * (X + 3)) = a * X + b) ∧ a = 7 / 2 ∧ b = 13 / 2 :=
by
  sorry

end remainder_poly_div_l826_826501


namespace average_items_per_month_l826_826868

theorem average_items_per_month:
  (∀ n : ℕ, (n = 3 → ∃ x, x = 4000 * n) ∧ (n = 9 → ∃ y, y = 4500 * n)) →
  ∃ z : ℝ, z = (12000 + 40500) / 12 ∧ z = 4375 :=
by
  intro h
  have h1 : 12000 = 3 * 4000 := by norm_num
  have h2 : 40500 = 9 * 4500 := by norm_num
  have h3 : 52500 = 12000 + 40500 := by norm_num
  have h4 : 4375 = 52500 / 12 := by norm_num
  use 4375
  constructor
  . exact h3 ▸ h4
  . exact h4
  sorry

end average_items_per_month_l826_826868


namespace evaluate_expression_l826_826028

variable (a b : ℤ)

-- Define the main expression
def main_expression (a b : ℤ) : ℤ :=
  (a - b)^2 + (a + 3 * b) * (a - 3 * b) - a * (a - 2 * b)

theorem evaluate_expression : main_expression (-1) 2 = -31 := by
  -- substituting the value and solving it in the proof block
  sorry

end evaluate_expression_l826_826028


namespace solve_fraction_inequality_l826_826439

theorem solve_fraction_inequality (x : ℝ) : 
  (x ≠ -2 ∧ ((x - 1) / (x + 2) ≥ 0)) ↔ (x ∈ set.Ioc (-∞ : ℝ) (-2) ∨ x ∈ set.Icc 1 (∞ : ℝ)) :=
by
  sorry

end solve_fraction_inequality_l826_826439


namespace P_equals_neg12_l826_826605

def P (a b : ℝ) : ℝ :=
  (2 * a + 3 * b)^2 - (2 * a + b) * (2 * a - b) - 2 * b * (3 * a + 5 * b)

lemma simplified_P (a b : ℝ) : P a b = 6 * a * b :=
  by sorry

theorem P_equals_neg12 (a b : ℝ) (h : b = -2 / a) : P a b = -12 :=
  by sorry

end P_equals_neg12_l826_826605


namespace second_largest_between_28_and_31_l826_826091

theorem second_largest_between_28_and_31 : 
  ∃ (n : ℕ), n > 28 ∧ n ≤ 31 ∧ (∀ m, (m > 28 ∧ m ≤ 31 ∧ m < 31) ->  m ≤ 30) :=
sorry

end second_largest_between_28_and_31_l826_826091


namespace carved_statue_l826_826886

-- Define the initial conditions and final weight
variables (x : ℝ) (initial_weight : ℝ) (final_weight : ℝ)
def initial_weight := 250
def final_weight := 105

-- Hypotheses based on the conditions
variables (h1 : 0 ≤ x) (h2 : x ≤ 100)

-- Setting up the equations derived from the conditions
theorem carved_statue (hx : (0.75 * 0.80 * (1 - x / 100) * initial_weight) = final_weight) : x = 30 :=
by
  sorry

end carved_statue_l826_826886


namespace find_general_students_l826_826073

-- Define the conditions and the question
structure Halls :=
  (general : ℕ)
  (biology : ℕ)
  (math : ℕ)
  (total : ℕ)

def conditions_met (h : Halls) : Prop :=
  h.biology = 2 * h.general ∧
  h.math = (3 / 5 : ℚ) * (h.general + h.biology) ∧
  h.total = h.general + h.biology + h.math ∧
  h.total = 144

-- The proof problem statement
theorem find_general_students (h : Halls) (h_cond : conditions_met h) : h.general = 30 :=
sorry

end find_general_students_l826_826073


namespace smaller_number_l826_826001

-- Define the positive numbers x and y with the given ratio
variables {k d : ℝ} (h₁ : k > 0) (h₂ : d > 0)
def x := 2 * k
def y := 3 * k

-- Define the condition 2x + 3y = d
def condition : Prop := 2 * x + 3 * y = d

-- Prove that the smaller number among x and y is (2 * d / 13)
theorem smaller_number (h₃ : condition) : x = 2 * d / 13 :=
by {
  -- Proof steps would be written here.
  sorry
}

end smaller_number_l826_826001


namespace average_percentage_of_10_students_l826_826124

theorem average_percentage_of_10_students
  (average_15 : ℝ)
  (total_average_25 : ℝ)
  (average_15_students : average_15 = 73)
  (total_average : total_average_25 = 79) :
  let total_percentage_25 := 25 * total_average_25
  let total_percentage_15 := 15 * average_15 in
  (total_percentage_25 - total_percentage_15) / 10 = 88 :=
by
  sorry

end average_percentage_of_10_students_l826_826124


namespace min_mn_value_l826_826230

theorem min_mn_value (m n : ℕ) (hmn : m > n) (hn : n ≥ 1) 
  (hdiv : 1000 ∣ 1978 ^ m - 1978 ^ n) : m + n = 106 :=
sorry

end min_mn_value_l826_826230


namespace adam_simon_100_miles_apart_l826_826160

theorem adam_simon_100_miles_apart :
  ∃ x : ℝ, (0 < x) ∧ (real.sqrt ((10 * x) ^ 2 + (8 * x) ^ 2) = 100) :=
sorry

end adam_simon_100_miles_apart_l826_826160


namespace alicia_tax_deduction_l826_826552

theorem alicia_tax_deduction :
  (let wage_dollars := 25
   let wage_cents := wage_dollars * 100
   let tax_rate := 2 / 100
   let tax_deduction := tax_rate * wage_cents
   tax_deduction = 50
  ) sorry

end alicia_tax_deduction_l826_826552


namespace find_X_l826_826358

def r (X Y : ℕ) : ℕ := X^2 + Y^2

theorem find_X (X : ℕ) (h : r X 7 = 338) : X = 17 := by
  sorry

end find_X_l826_826358


namespace weight_of_mixture_l826_826474

theorem weight_of_mixture (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b : ℝ)
  (total_volume : ℝ)
  (h_weight_a : weight_a_per_liter = 950)
  (h_weight_b : weight_b_per_liter = 850)
  (h_ratio : ratio_a / (ratio_a + ratio_b) = 3 / 5 ∧ ratio_b / (ratio_a + ratio_b) = 2 / 5)
  (h_total_volume : total_volume = 4) :
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume in
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume in
  let weight_a := volume_a * weight_a_per_liter in
  let weight_b := volume_b * weight_b_per_liter in
  ((weight_a + weight_b) / 1000) = 3.64 :=
by
  sorry

end weight_of_mixture_l826_826474


namespace median_of_gasoline_prices_l826_826081

-- Define the gasoline prices for 7 cities as a list of real numbers
def gasoline_prices : List ℝ := sorry

-- Define the median function for a list of real numbers
def median (prices : List ℝ) : ℝ :=
  let sorted_prices := prices.qsort (λ a b => a < b)
  sorted_prices.get! (sorted_prices.length / 2)

-- The theorem to prove that the median of the gasoline prices is correct
theorem median_of_gasoline_prices :
  gasoline_prices.length = 7 → median gasoline_prices = (sorted_prices.get! (sorted_prices.length / 2)) := sorry

end median_of_gasoline_prices_l826_826081


namespace find_k_l826_826085

theorem find_k 
    (P : ℝ × ℝ) (S : ℝ × ℝ) (Q : ℝ)
    (P_on_larger_circle : P = (5, 12))
    (S_on_smaller_circle : ∃ k : ℝ, S = (0, k))
    (QR : Q = 4) :
    S = (0, 9) := by
  -- Since Q is used to refer to the radius difference, we know QR means Q is the length 4
  let OP := real.sqrt (5^2 + 12^2)
  have origin_to_P : OP = 13 := by norm_num
  let OR := OP
  -- Radius of the smaller circle
  let OQ := OR - Q
  have OQ_value : OQ = 9 := by norm_num
  -- Since "S" is on the smaller circle, its radius should be 9 which implies S = (0, 9)
  cases S_on_smaller_circle with k hk
  rw hk
  exact congr_arg (prod.mk 0) (eq.symm OQ_value)

end find_k_l826_826085


namespace trigonometric_periods_l826_826163

theorem trigonometric_periods :
  (∀ x, tan (x + Real.pi) = tan x) ∧ (∀ x, cot (x + Real.pi) = cot x) →
  ∃ p < 2 * Real.pi, (∀ x, tan (x + p) = tan x) ∧ (∀ x, cot (x + p) = cot x) :=
by
  intro h
  use Real.pi
  split
  · norm_num
  · exact h

end trigonometric_periods_l826_826163


namespace count_positive_integer_solutions_l826_826270

theorem count_positive_integer_solutions :
  (∃ x1 x2 x3 : ℕ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ x1 + x2 + x3 = 30) ->
  (nat.factorial 29 / (nat.factorial 2 * nat.factorial (29 - 2)) = 406) :=
by
  sorry

end count_positive_integer_solutions_l826_826270


namespace train_speed_l826_826894

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end train_speed_l826_826894


namespace allay_internet_days_l826_826906

theorem allay_internet_days (cost_per_day : ℝ) (max_debt : ℝ) (initial_balance : ℝ) (payment : ℝ) : 
    cost_per_day = 0.5 → 
    max_debt = 5 → 
    initial_balance = 0 → 
    payment = 7 → 
    14 = (max_debt / cost_per_day) + ((payment - max_debt) / cost_per_day) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end allay_internet_days_l826_826906


namespace coins_problem_l826_826132

theorem coins_problem : 
  ∃ x : ℕ, 
  (x % 8 = 6) ∧ 
  (x % 7 = 5) ∧ 
  (x % 9 = 1) ∧ 
  (x % 11 = 0) := 
by
  -- Proof to be provided here
  sorry

end coins_problem_l826_826132


namespace min_distance_l826_826744

/-
We will define our points and the parabola.
-/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-
We define points A and B as given.
-/
def A := Point.mk 1 0
def B := Point.mk 4 3

/-
We assert that P is a point on the parabola y^2 = 4x.
-/
def on_parabola (P : Point) := P.y^2 = 4 * P.x

/-
Define the distance function between two points.
-/
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

/-
Formalize the main theorem: the minimum value of AP + BP.
-/
theorem min_distance : ∃ P : Point, on_parabola P ∧ P.x ≥ 0 ∧ P.y ≥ 0 ∧ 
  ∀ Q : Point, on_parabola Q ∧ Q.x ≥ 0 ∧ Q.y ≥ 0 → dist A P + dist B P ≤ dist A Q + dist B Q := 
begin
  sorry
end

end min_distance_l826_826744


namespace justin_blocks_from_home_l826_826720

-- Definitions based on the conditions
def minutes_to_run_n_blocks (blocks : ℕ) (minutes : ℝ) : ℝ := minutes / blocks
def blocks_run_in_minutes (blocks_per_minute : ℝ) (minutes : ℝ) : ℝ := blocks_per_minute * minutes

-- Given Justin runs 2 blocks in 1.5 minutes
def justin_blocks_per_1_5_minute : ℝ := 2
def justin_minutes_for_2_blocks : ℝ := 1.5 

-- Define the blocks per minute he can run
def justin_blocks_per_minute := justin_blocks_per_1_5_minute / justin_minutes_for_2_blocks

-- Define the total minutes Justin takes to get home
def justin_time_to_home : ℝ := 6 

-- Define the theorem to prove how many blocks Justin is from home
theorem justin_blocks_from_home : blocks_run_in_minutes justin_blocks_per_minute justin_time_to_home = 8 := 
sorry

end justin_blocks_from_home_l826_826720


namespace complex_exp_neg_sum_l826_826309

theorem complex_exp_neg_sum (α β : ℝ) 
  (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = (3/5) + (2/5) * complex.I) :
  complex.exp (-complex.I * α) + complex.exp (-complex.I * β) = (3/5) - (2/5) * complex.I :=
sorry

end complex_exp_neg_sum_l826_826309


namespace shaded_area_l826_826333

def radius_larger_circle := 8
def radius_smaller_circle := radius_larger_circle / 2

def area_circle (r : ℝ) : ℝ := π * r^2

theorem shaded_area :
  let area_larger := area_circle radius_larger_circle
  let area_smaller := area_circle radius_smaller_circle
  let total_smaller_areas := 2 * area_smaller
  area_larger - total_smaller_areas = 32 * π :=
by
  sorry

end shaded_area_l826_826333


namespace acute_angle_condition_l826_826752

def acute_angled_triangle (α β γ : ℝ) : Prop :=
  (π / 4 < α ∧ α < π / 2) ∧ (π / 4 < β ∧ β < π / 2) ∧ (π / 4 < γ ∧ γ < π / 2) ∨
  (α < π / 4 ∧ β < π / 4 ∧ γ < 3 * π / 4) ∨
  (β < π / 4 ∧ γ < π / 4 ∧ α < 3 * π / 4) ∨
  (γ < π / 4 ∧ α < π / 4 ∧ β < 3 * π / 4)

theorem acute_angle_condition (α β γ α_1 β_1 γ_1 : ℝ) (h₁ : α_1 = π - 2 * α) (h₂ : β_1 = π - 2 * β) (h₃ : γ_1 = π - 2 * γ)
  (hα : π / 4 < α ∧ α < π / 2) (hβ : π / 4 < β ∧ β < π / 2) (hγ : π / 4 < γ ∧ γ < π / 2) :
  acute_angled_triangle α β γ :=
  by sorry

end acute_angle_condition_l826_826752


namespace function_arrangement_l826_826451

noncomputable def f : ℝ → ℝ := sorry

theorem function_arrangement (h1 : ∀ x : ℝ, differentiable ℝ f)
    (h2 : ∀ x : ℝ, f x = f (4 - x))
    (h3 : ∀ x : ℝ, x < 2 → (x - 2) * (deriv f x) < 0) :
    f(-1) < f(4) ∧ f(4) < f(1) := 
sorry

end function_arrangement_l826_826451


namespace perpendicular_lines_eq_l826_826995

def dot_product (v1 v2 : Vector ℝ 3) : ℝ :=
  v1.head * v2.head + v1.tail.head * v2.tail.head + v1.tail.tail.head * v2.tail.tail.head

theorem perpendicular_lines_eq (m : ℝ) 
  (a : Vector ℝ 3 := ⟨[1, m, -1]⟩) 
  (b : Vector ℝ 3 := ⟨[-2, 1, 1]⟩) 
  (h : dot_product a b = 0) : 
  m = 3 :=
by {
  sorry
}

end perpendicular_lines_eq_l826_826995


namespace setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l826_826107

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : Int) : Prop :=
  a^2 + b^2 = c^2

-- Define the given sets
def setA : (Int × Int × Int) := (12, 15, 18)
def setB : (Int × Int × Int) := (3, 4, 5)
def setC : (Rat × Rat × Rat) := (1.5, 2, 2.5)
def setD : (Int × Int × Int) := (6, 9, 15)

-- Proven statements about each set
theorem setB_is_PythagoreanTriple : isPythagoreanTriple 3 4 5 :=
  by
  sorry

theorem setA_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 12 15 18 :=
  by
  sorry

-- Pythagorean triples must consist of positive integers
theorem setC_is_not_PythagoreanTriple : ¬ ∃ (a b c : Int), a^2 + b^2 = c^2 ∧ 
  a = 3/2 ∧ b = 2 ∧ c = 5/2 :=
  by
  sorry

theorem setD_is_not_PythagoreanTriple : ¬ isPythagoreanTriple 6 9 15 :=
  by
  sorry

end setB_is_PythagoreanTriple_setA_is_not_PythagoreanTriple_setC_is_not_PythagoreanTriple_setD_is_not_PythagoreanTriple_l826_826107


namespace polyhedron_ball_coverage_l826_826350

noncomputable def c_P := (A^3 * (4 * π / 3)^2) / π^3

theorem polyhedron_ball_coverage
  (P : Type) [Polyhedron P] (A : ℝ) (n : ℕ) (V : ℝ) :
  ∃ C > 0, n ≥ C / V^2 :=
begin
  let C := c_P,
  use C,
  sorry  -- Proof is omitted
end

end polyhedron_ball_coverage_l826_826350


namespace nine_otimes_three_equals_fourteen_l826_826810

def otimes (a b : ℝ) : ℝ := a + (5 * a) / (3 * b)

theorem nine_otimes_three_equals_fourteen : otimes 9 3 = 14 :=
by
  sorry

end nine_otimes_three_equals_fourteen_l826_826810


namespace difference_in_savings_correct_l826_826723

def S_last_year : ℝ := 45000
def saved_last_year_pct : ℝ := 0.083
def raise_pct : ℝ := 0.115
def saved_this_year_pct : ℝ := 0.056

noncomputable def saved_last_year_amount : ℝ := saved_last_year_pct * S_last_year
noncomputable def S_this_year : ℝ := S_last_year * (1 + raise_pct)
noncomputable def saved_this_year_amount : ℝ := saved_this_year_pct * S_this_year
noncomputable def difference_in_savings : ℝ := saved_last_year_amount - saved_this_year_amount

theorem difference_in_savings_correct :
  difference_in_savings = 925.20 := by
  sorry

end difference_in_savings_correct_l826_826723


namespace icosahedron_edges_l826_826919

theorem icosahedron_edges (faces : ℕ) (vertices : ℕ) (vertex_edges : ℕ) : faces = 20 → vertices = 12 → vertex_edges = 5 → ∃ edges : ℕ, edges = 30 :=
by {
  intros h_faces h_vertices h_vertex_edges,
  have h : ∃ (E : ℕ), E = (vertex_edges * vertices) / 2,
  { use (5 * 12) / 2, 
    norm_num,
    exact ⟨30⟩ },
  exact h
}

end icosahedron_edges_l826_826919


namespace sum_of_coefficients_l826_826216

theorem sum_of_coefficients :
  (Nat.choose 50 3 + Nat.choose 50 5) = 2138360 := 
by 
  sorry

end sum_of_coefficients_l826_826216


namespace exists_xy_nat_divisible_l826_826780

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end exists_xy_nat_divisible_l826_826780


namespace cloth_sales_worth_l826_826910

/--
An agent gets a commission of 2.5% on the sales of cloth. If on a certain day, he gets Rs. 15 as commission, 
proves that the worth of the cloth sold through him on that day is Rs. 600.
-/
theorem cloth_sales_worth (commission : ℝ) (rate : ℝ) (total_sales : ℝ) 
  (h_commission : commission = 15) (h_rate : rate = 2.5) (h_commission_formula : commission = (rate / 100) * total_sales) : 
  total_sales = 600 := 
by
  sorry

end cloth_sales_worth_l826_826910


namespace total_goldfish_preferring_students_l826_826915

def goldfish_preferring_students (num_students: ℕ) (fraction: ℚ) : ℕ :=
  (fraction * num_students).toNat

theorem total_goldfish_preferring_students :
  let mj_pref := goldfish_preferring_students 30 (1 / 6)
  let mf_pref := goldfish_preferring_students 30 (2 / 3)
  let mh_pref := goldfish_preferring_students 30 (1 / 5)
  mj_pref + mf_pref + mh_pref = 31 :=
by
  -- We use 'by' to indicate the need for a proof, but we leave it unproven with 'sorry'.
  sorry

end total_goldfish_preferring_students_l826_826915


namespace jogging_track_circumference_l826_826191

theorem jogging_track_circumference
  (Deepak_speed : ℝ)
  (Wife_speed : ℝ)
  (meet_time_minutes : ℝ)
  (H_deepak_speed : Deepak_speed = 4.5)
  (H_wife_speed : Wife_speed = 3.75)
  (H_meet_time_minutes : meet_time_minutes = 3.84) :
  let meet_time_hours := meet_time_minutes / 60
  let distance_deepak := Deepak_speed * meet_time_hours
  let distance_wife := Wife_speed * meet_time_hours
  let total_distance := distance_deepak + distance_wife
  let circumference := 2 * total_distance
  circumference = 1.056 :=
by
  sorry

end jogging_track_circumference_l826_826191


namespace double_sum_cos_nonneg_l826_826015

theorem double_sum_cos_nonneg (n : ℕ) (x : Fin n → ℝ) (α : Fin n → ℝ) :
  0 ≤ ∑ i, ∑ j, x i * x j * Real.cos (α i - α j) :=
sorry

end double_sum_cos_nonneg_l826_826015


namespace domain_of_f_l826_826193

noncomputable def f (x : ℝ) : ℝ := (x^3 - 64) / (x - 8)

theorem domain_of_f :
  {x : ℝ | f x ≠ ∞} = {x : ℝ | x ≠ 8} :=
by
  sorry

end domain_of_f_l826_826193


namespace sum_of_valid_Cs_l826_826963

theorem sum_of_valid_Cs : 
  (∑ c in Finset.Icc (-27) 21, c) = -147 :=
by
  sorry

end sum_of_valid_Cs_l826_826963


namespace incorrect_statement_B_l826_826167

-- Define the conditions as hypotheses
variables {α : Type*} [linear_ordered_field α]

def range (s : set α) : α := if h : s.nonempty then s.sup' h - s.inf' h else 0
def variance (s : list α) : α := (s.sum (λ x, (x - s.sum / s.length)^2)) / s.length
def std_dev (s : list α) : α := real.sqrt (variance s)

-- Hypotheses
axiom h1 : ∀ s : set α, (range s) ≥ 0  -- Range reflects max - min values
axiom h2 : ∀ s : list α, (variance s) ≥ 0  -- Variance and std_dev measure magnitude of fluctuations
axiom h3 : ∀ s : set α, (range s) = 0 ↔ s.sup' h = s.inf' h  -- Smaller range -> more concentrated
axiom h4 : ∀ s : list α, (std_dev s) = 0 ↔ ∀ x ∈ s, x = s.head  -- Smaller std_dev -> more stable
axiom h5 : ∀ s : list α, s.sum / s.length = s.sum / s.length  -- Smaller mean does not reflect stability

-- Theorem: Statement B is incorrect given the conditions
theorem incorrect_statement_B : 
  ∀ (s : list α), 
    ¬(s.sum / s.length = s.sum / s.length → 
      (∀ x y, x - y = 0) ∨ 
      std_dev s < std_dev s ∨ 
      variance s < variance s) :=
by
  intros,
  -- The proof would go here, but we leave it as a stub since it's not required.
  sorry

end incorrect_statement_B_l826_826167


namespace remaining_numbers_count_l826_826417

-- Defining the initial set from 1 to 20
def initial_set : finset ℕ := finset.range 21 \ {0}

-- Condition: Erase all even numbers
def without_even : finset ℕ := initial_set.filter (λ n, n % 2 ≠ 0)

-- Condition: Erase numbers that give a remainder of 4 when divided by 5 from the remaining set
def final_set : finset ℕ := without_even.filter (λ n, n % 5 ≠ 4)

-- Statement to prove
theorem remaining_numbers_count : final_set.card = 8 :=
by
  -- admitting the proof
  sorry

end remaining_numbers_count_l826_826417


namespace common_points_equilateral_triangle_l826_826568

theorem common_points_equilateral_triangle :
  (∀ ⟨x, y⟩, 
    x ^ 2 + (y - 1) ^ 2 = 1 ∧ 9 * x ^ 2 + (y + 1) ^ 2 = 9 → 
    (x = 0 ∧ y = 2) ∨ 
    (x = sqrt 3 / 2 ∧ y = 1 / 2) ∨ 
    (x = - sqrt 3 / 2 ∧ y = 1 / 2)) →
  distance (0, 2) (sqrt 3 / 2, 1 / 2) = distance (sqrt 3 / 2, 1 / 2) (- sqrt 3 / 2, 1 / 2) ∧
  distance (sqrt 3 / 2, 1 / 2) (- sqrt 3 / 2, 1 / 2) = distance (- sqrt 3 / 2, 1 / 2) (0, 2) :=
sorry

end common_points_equilateral_triangle_l826_826568


namespace beetle_minimum_distance_positions_l826_826164
-- Import necessary library

-- Define constants and parameters given in the problem.
def strider_speed_ratio : ℝ := 2
def beetle_initial_pos : ℝ × ℝ := (5, 5 * Real.sqrt 7)
def strider_initial_pos : ℝ × ℝ := (-2, -2 * Real.sqrt 7)
def strider_radius : ℝ := 4 * Real.sqrt 2
def beetle_radius : ℝ := 10 * Real.sqrt 2

-- Define the correct answer coordinates as given.
def beetle_positions : list (ℝ × ℝ) := [
  (5 / Real.sqrt 2 * (1 - Real.sqrt 7), 5 / Real.sqrt 2 * (1 + Real.sqrt 7)),
  (-5 / Real.sqrt 2 * (1 + Real.sqrt 7), 5 / Real.sqrt 2 * (1 - Real.sqrt 7)),
  (5 / Real.sqrt 2 * (Real.sqrt 7 - 1), -5 / Real.sqrt 2 * (1 + Real.sqrt 7)),
  (5 / Real.sqrt 2 * (1 + Real.sqrt 7), 5 / Real.sqrt 2 * (Real.sqrt 7 - 1))
]

-- Main theorem statement asserting the proof.
theorem beetle_minimum_distance_positions
    (strider_speed_ratio : ℝ)
    (beetle_initial_pos : ℝ × ℝ)
    (strider_initial_pos : ℝ × ℝ)
    (strider_radius : ℝ)
    (beetle_radius : ℝ) :
  beetle_positions = [
    (5 / (Real.sqrt 2) * (1 - Real.sqrt 7), 5 / (Real.sqrt 2) * (1 + Real.sqrt 7)),
    (-5 / (Real.sqrt 2) * (1 + Real.sqrt 7), 5 / (Real.sqrt 2) * (1 - Real.sqrt 7)),
    (5 / (Real.sqrt 2) * (Real.sqrt 7 - 1), -5 / (Real.sqrt 2) * (1 + Real.sqrt 7)),
    (5 / (Real.sqrt 2) * (1 + Real.sqrt 7), 5 / (Real.sqrt 2) * (Real.sqrt 7 - 1))
  ] :=
sorry

end beetle_minimum_distance_positions_l826_826164


namespace greg_needs_additional_amount_l826_826657

def total_cost : ℤ := 90
def saved_amount : ℤ := 57
def additional_amount_needed : ℤ := total_cost - saved_amount

theorem greg_needs_additional_amount :
  additional_amount_needed = 33 :=
by
  sorry

end greg_needs_additional_amount_l826_826657


namespace percentage_passed_all_three_l826_826330

variable (F_H F_E F_M F_HE F_EM F_HM F_HEM : ℝ)

theorem percentage_passed_all_three :
  F_H = 0.46 →
  F_E = 0.54 →
  F_M = 0.32 →
  F_HE = 0.18 →
  F_EM = 0.12 →
  F_HM = 0.1 →
  F_HEM = 0.06 →
  (100 - (F_H + F_E + F_M - F_HE - F_EM - F_HM + F_HEM)) = 2 :=
by sorry

end percentage_passed_all_three_l826_826330


namespace smallest_solution_floor_eq_l826_826593

theorem smallest_solution_floor_eq (x : ℝ) (h : ∀ y : ℝ, (floor (x^2) - (floor x)^2 = 25) → (floor y^2 - (floor y)^2 ≠ 25 ∨ x ≤ y)) :
  x = real.sqrt 194 :=
sorry

end smallest_solution_floor_eq_l826_826593


namespace angle_BDC_60_degrees_l826_826123

theorem angle_BDC_60_degrees 
  (A B C D : Type)
  [inner_product_geometry A B C]
  (h1 : ∃ (C : A), ∠ABC ∧ ∠BAC = 30)
  (h2 : right_triangle (triangle.mk A B C))
  (h3 : BD_bisects_ABC (BD : B B) ∧ D ∈ AC)
  : angle BDC = 60 :=
begin
  sorry,
end

end angle_BDC_60_degrees_l826_826123


namespace collinear_BCE_concyclic_ABDE_l826_826916

variables {A B C D E : Type} [EuclideanGeometry A B C D E]

-- Given conditions
axiom (excenter :  ∃ triangle : EuclideanTriangle A B C, Excenter triangle D)
axiom (reflection : ReflectAcrossLine D C A E)

-- Statement to be proved, Question 1
theorem collinear_BCE : Collinear B C E := by
  sorry

-- Statement to be proved, Question 2
theorem concyclic_ABDE : Concyclic A B D E := by
  sorry

end collinear_BCE_concyclic_ABDE_l826_826916


namespace number_of_remaining_numbers_problem_solution_l826_826402

def initial_set : List Nat := (List.range 20).map (fun n => n + 1)
def is_even (n : Nat) : Bool := n % 2 = 0
def remainder_4_mod_5 (n : Nat) : Bool := n % 5 = 4

def remaining_numbers (numbers : List Nat) : List Nat :=
  numbers.filter (fun n => ¬ is_even n).filter (fun n => ¬ remainder_4_mod_5 n)

theorem number_of_remaining_numbers : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] :=
  by {
    sorry
  }

theorem problem_solution : remaining_numbers initial_set = [1, 3, 5, 7, 11, 13, 15, 17] -> length (remaining_numbers initial_set) = 8 :=
  by {
    intro h,
    rw h,
    simp,
  }

end number_of_remaining_numbers_problem_solution_l826_826402


namespace quadratic_real_roots_range_of_k_l826_826272

theorem quadratic_real_roots (k : ℝ) :
    let a := 1
    let b := -(k + 3)
    let c := 2 * k + 2
    let delta := b^2 - 4 * a * c
    delta = (k - 1)^2 ∧ ((k - 1)^2 ≥ 0) := 
by {
    let a := 1
    let b := -(k + 3)
    let c := 2 * k + 2
    let delta := b^2 - 4 * a * c
    have h1 : delta = (k - 1)^2 :=
      calc delta = (k + 3)^2 - 4 * 1 * (2 * k + 2) : by simp [a, b, c]
          ... = k^2 + 6 * k + 9 - 8 * k - 8 : by ring
          ... = k^2 - 2 * k + 1 : by ring,
    have h2 : (k - 1)^2 ≥ 0 := by apply pow_two_nonneg,
    exact ⟨h1, h2⟩
}
-- Question 2
theorem range_of_k (k : ℝ) (h: 0 < k + 1 ∧ k + 1 < 1) : -1 < k ∧ k < 0 := 
by {
    cases h with h1 h2,
    have h3 : k + 1 < 1 := h2,
    have h4 : -1 < k := by linarith,
    have h5 : k < 0 := by linarith,
    exact ⟨h4, h5⟩,
}

end quadratic_real_roots_range_of_k_l826_826272


namespace num_of_satisfying_n_l826_826978

theorem num_of_satisfying_n : 
  {n : ℤ | |n| ≤ 20 ∧ ∃ (a : ℚ), (x^2 + x + n = (x - a)(x + a + 1))}.finite.to_finset.card = 5 :=
by
  sorry

end num_of_satisfying_n_l826_826978


namespace initial_hours_per_day_l826_826125

/-- 
Given:
1. 18 men working a certain number of hours per day dig 30 meters deep.
2. To dig to a depth of 50 meters, working 6 hours per day, 22 extra men should be put to work (total of 40 men).

Prove:
The initial 18 men were working \(\frac{200}{9}\) hours per day.
-/
theorem initial_hours_per_day 
  (h : ℚ)
  (work_done_18_men : 18 * h * 30 = 40 * 6 * 50) :
  h = 200 / 9 :=
by
  sorry

end initial_hours_per_day_l826_826125


namespace sum_of_solutions_l826_826214

theorem sum_of_solutions (x : ℤ) (h : x^4 - 13 * x^2 + 36 = 0) : 
  (finset.sum (finset.filter (λ (x : ℤ), x^4 - 13 * x^2 + 36 = 0) (finset.range 4))) = 0 :=
by
  sorry

end sum_of_solutions_l826_826214


namespace max_sum_of_ratios_l826_826725

theorem max_sum_of_ratios
  (a b c d : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h_sum : a + c = 20)
  (h_ineq : (a / b.toRat) + (c / d.toRat) < 1.toRat) : 
  (a / b.toRat) + (c / d.toRat) ≤ 20 / 21 := sorry

end max_sum_of_ratios_l826_826725


namespace find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l826_826623

variables {m : ℝ} 
def point_on_y_axis (P : (ℝ × ℝ)) := P = (0, -3)
def point_distance_to_y_axis (P : (ℝ × ℝ)) := P = (6, 0) ∨ P = (-6, -6)
def point_in_third_quadrant_and_equidistant (P : (ℝ × ℝ)) := P = (-6, -6)

theorem find_coords_of_P_cond1 (P : ℝ × ℝ) (h : 2 * m + 4 = 0) : point_on_y_axis P ↔ P = (0, -3) :=
by {
  sorry
}

theorem find_coords_of_P_cond2 (P : ℝ × ℝ) (h : abs (2 * m + 4) = 6) : point_distance_to_y_axis P ↔ (P = (6, 0) ∨ P = (-6, -6)) :=
by {
  sorry
}

theorem find_coords_of_P_cond3 (P : ℝ × ℝ) (h1 : 2 * m + 4 < 0) (h2 : m - 1 < 0) (h3 : abs (2 * m + 4) = abs (m - 1)) : point_in_third_quadrant_and_equidistant P ↔ P = (-6, -6) :=
by {
  sorry
}

end find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l826_826623


namespace admin_staff_in_sample_l826_826152

theorem admin_staff_in_sample (total_staff : ℕ) (admin_staff : ℕ) (total_samples : ℕ)
  (probability : ℚ) (h1 : total_staff = 200) (h2 : admin_staff = 24)
  (h3 : total_samples = 50) (h4 : probability = 50 / 200) :
  admin_staff * probability = 6 :=
by
  -- Proof goes here
  sorry

end admin_staff_in_sample_l826_826152


namespace still_need_to_travel_l826_826166

def amoli_speed := 42
def amoli_time := 3
def anayet_speed := 61
def anayet_time := 2.5
def bimal_speed := 55
def bimal_time := 4
def total_distance := 1045

def amoli_distance := amoli_speed * amoli_time
def anayet_distance := anayet_speed * anayet_time
def bimal_distance := bimal_speed * bimal_time

def total_distance_covered := amoli_distance + anayet_distance + bimal_distance
def distance_remaining := total_distance - total_distance_covered

theorem still_need_to_travel : distance_remaining = 546.5 :=
by
  sorry

end still_need_to_travel_l826_826166


namespace power_function_inequality_l826_826639

theorem power_function_inequality (m n : ℝ) (f : ℝ → ℝ) (a b c : ℝ)
    (h₁ : m - 2 = 1)
    (h₂ : f = (λ x, (m - 2) * x ^ n))
    (h₃ : m = 3)
    (h₄ : 3 ^ n = 9)
    (h₅ : a = f (m ^ (-1 / 3)))
    (h₆ : b = f (Real.log (1 / 3)))
    (h₇ : c = f (Real.sqrt 2 / 2)) :
    a < c < b := 
  sorry

end power_function_inequality_l826_826639


namespace polynomial_transformation_l826_826679

theorem polynomial_transformation :
  ∀ (a h k : ℝ), (8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → a + h + k = -23.5 :=
by
  intros a h k h_eq
  sorry

end polynomial_transformation_l826_826679


namespace quadrilateral_area_l826_826642

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y = 0
def point_on_circle (P : ℝ × ℝ) : Prop := circle_eq P.1 P.2

theorem quadrilateral_area (x y : ℝ) (Hcircle : circle_eq x y) (Hchords : (3, 5) = (x, y)) :
    ∃ (A B C D : ℝ × ℝ), 
        (point_on_circle A) ∧ (point_on_circle B) ∧ (point_on_circle C) ∧ (point_on_circle D) ∧
        A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
        let AC := ((A.1 - C.1)^2 + (A.2 - C.2)^2).sqrt in
        let BD := ((B.1 - D.1)^2 + (B.2 - D.2)^2).sqrt in
        (AC = 10) ∧ (BD = 2 * Real.sqrt 24) ∧
        (1/2 * AC * BD = 20 * Real.sqrt 6) :=
by
  sorry

end quadrilateral_area_l826_826642


namespace num_elements_in_arith_seq_l826_826662

noncomputable def arith_seq_first_term : ℝ := 2.5
noncomputable def arith_seq_common_diff : ℝ := 5
noncomputable def arith_seq_last_term : ℝ := 62.5

theorem num_elements_in_arith_seq :
  ∃ n : ℕ, n = ((arith_seq_last_term - arith_seq_first_term) / arith_seq_common_diff).to_nat + 1 :=
begin
  sorry
end

end num_elements_in_arith_seq_l826_826662


namespace terminal_side_in_second_or_fourth_quadrant_l826_826669

theorem terminal_side_in_second_or_fourth_quadrant (x : ℝ) (h : sin x * cos x < 0) : 
  (sin x > 0 ∧ cos x < 0) ∨ (sin x < 0 ∧ cos x > 0) :=
by
  sorry

end terminal_side_in_second_or_fourth_quadrant_l826_826669


namespace batsman_average_after_17th_inning_l826_826860

-- Define the conditions
variable (A : ℝ) -- Average score before 17th inning
variable (H1 : 85 = 17 * (A + 3) - 16 * A) -- Condition derived from the given problem

-- Theorem statement
theorem batsman_average_after_17th_inning (H1 : 85 = 17 * (A + 3) - 16 * A) : A = 34 → A + 3 = 37 := 
by {
  intros h,
  rw h,
  norm_num,
}

# You can remove sorry to actually include the solving part if needed
-- sorry

end batsman_average_after_17th_inning_l826_826860


namespace max_trailing_zeros_l826_826517

theorem max_trailing_zeros :
  ∃ (a b c d e f g : ℕ),
  (a ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (b ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (c ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (d ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (e ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (f ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (g ∈ {1, 2, 3, 5, 8, 10, 11}) ∧
  (∀ x y, (x = y) → ({a, b, c, d, e, f, g} \ {x}) = {y, x}) ∧
  min (3 * a + c + 2 * e + g) (2 * b + g) = 32 :=
sorry

end max_trailing_zeros_l826_826517


namespace marks_lost_per_wrong_answer_l826_826694

theorem marks_lost_per_wrong_answer
    (total_questions : ℕ)
    (correct_questions : ℕ)
    (total_marks : ℕ)
    (marks_per_correct : ℕ)
    (marks_lost : ℕ)
    (x : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_questions = 44)
    (h3 : total_marks = 160)
    (h4 : marks_per_correct = 4)
    (h5 : marks_lost = 176 - total_marks)
    (h6 : marks_lost = x * (total_questions - correct_questions)) :
    x = 1 := by
  sorry

end marks_lost_per_wrong_answer_l826_826694


namespace no_adjacent_numbers_differ_by_10_or_multiple_10_l826_826907

theorem no_adjacent_numbers_differ_by_10_or_multiple_10 :
  ¬ ∃ (f : Fin 25 → Fin 25),
    (∀ n : Fin 25, f (n + 1) - f n = 10 ∨ (f (n + 1) - f n) % 10 = 0) :=
by
  sorry

end no_adjacent_numbers_differ_by_10_or_multiple_10_l826_826907


namespace false_propositions_l826_826359

section

variable (α : Type) [Plane α]
variable (a b : Line)

-- Definitions of conditions
def Prop1 : Prop := a ∥ α ∧ a ⟂ b → b ⟂ α
def Prop2 : Prop := a ∥ b ∧ a ⟂ α → b ⟂ α
def Prop3 : Prop := a ⟂ α ∧ a ⟂ b → b ∥ α
def Prop4 : Prop := a ⟂ α ∧ b ⟂ α → a ∥ b

-- Mathematical proof problem statement
theorem false_propositions : (¬ Prop1) ∧ (¬ Prop3) := by
  sorry

end

end false_propositions_l826_826359


namespace range_of_a_for_two_points_intersection_l826_826042

-- Definitions based on given conditions
def is_intersection (a : ℝ) : Prop :=
  ∃ x : ℝ, y : ℝ, (y = -x + a) ∧ (y = sqrt (1 - x^2))

def line_circle_intersect_two_points (a : ℝ) : Prop :=
  count { x // ∃ y : ℝ, (y = -x + a) ∧ (y = sqrt (1 - x^2)) } = 2

theorem range_of_a_for_two_points_intersection :
  { a : ℝ | line_circle_intersect_two_points a } = { a : ℝ | 1 ≤ a ∧ a < sqrt 2 } :=
by 
  sorry

end range_of_a_for_two_points_intersection_l826_826042


namespace flag_blue_squares_percentage_l826_826543

theorem flag_blue_squares_percentage (s : ℝ) (H : 0 < s) :
  let flag_area := s^2 in
  let cross_area := 0.4 * flag_area in
  ∃ (two_blue_squares_area : ℝ),
    2 * two_blue_squares_area = 0.2 * flag_area :=
by 
  let flag_area := s^2
  let cross_area := 0.4 * flag_area
  use 0.1 * flag_area
  have : 2 * (0.1 * flag_area) = 0.2 * flag_area, by ring
  exact this

end flag_blue_squares_percentage_l826_826543


namespace corrected_log_values_l826_826822

variable (a b c : ℝ)

def log_values (x : ℝ) : Option ℝ :=
  if x = 0.021 then some (2*a + b + c - 3) else
  if x = 0.27 then some (6*a - 3*b - 2) else
  if x = 1.5 then some (3*a - b + c) else
  if x = 2.8 then some (1 - 2*a + 2*b - c) else
  if x = 3 then some (2*a - b) else
  if x = 5 then some (a + c) else
  if x = 6 then some (1 + a - b - c) else
  if x = 7 then some (2*(b + c)) else
  if x = 8 then some (3 - 3*a - 3*c) else
  if x = 9 then some (4*a - 2*b) else
  if x = 14 then some (1 - c + 2*b) else
  none

theorem corrected_log_values :
  (log_values a b c 1.5 = some (3*a - b + c - 1)) ∧
  (log_values a b c 7 = some (2*b + c)) :=
by
  sorry

end corrected_log_values_l826_826822


namespace percentage_loss_is_correct_l826_826020

-- Definitions from the conditions
def selling_price := 990
def total_cost := 1980
def profit_percentage := 0.10

-- Calculate cost price of first bicycle
def cost_price_first_bicycle := selling_price / (1 + profit_percentage)

-- Calculate cost price of second bicycle
def cost_price_second_bicycle := total_cost - cost_price_first_bicycle

-- Calculate loss on second bicycle
def loss_on_second_bicycle := cost_price_second_bicycle - selling_price

-- Calculate percentage loss on second bicycle
def percentage_loss_on_second_bicycle := (loss_on_second_bicycle / cost_price_second_bicycle) * 100

-- Statement of the problem to prove
theorem percentage_loss_is_correct : percentage_loss_on_second_bicycle = 8.33 :=
by
  sorry

end percentage_loss_is_correct_l826_826020


namespace perpendicular_chords_l826_826463

-- Definition for the circle and the points on it
variables {A B C D M N K L : Type*}

-- Assume all points are on a circle
def points_on_circle (A B C D : Type*) : Prop := ∀ (P : Type*), P ∈ {A, B, C, D} → ∃ (O : Type*), Metric.sphere O (distance O P)

-- Definition for midpoints of arcs
def is_midpoint_of_arc (M A B : Type*) : Prop := ∃ (O : Type*), M ∈ Metric.sphere O (distance O A) ∧ M ∈ Metric.sphere O (distance O B)

-- The main proof theorem statement
theorem perpendicular_chords (A B C D M N K L : Type*)
  (on_circle : points_on_circle A B C D)
  (midpoints : (is_midpoint_of_arc M A B) ∧ (is_midpoint_of_arc N B C) ∧ (is_midpoint_of_arc K C D) ∧ (is_midpoint_of_arc L D A)) :
  ⊥ MK NL :=
sorry

end perpendicular_chords_l826_826463


namespace conjugate_of_z_is_correct_l826_826640

def z := (2 - complex.I)^2
def z_conjugate : complex := complex.conj z

theorem conjugate_of_z_is_correct : z_conjugate = 3 + 4 * complex.I := by
  -- proof is omitted
  sorry

end conjugate_of_z_is_correct_l826_826640


namespace time_per_item_l826_826147

theorem time_per_item (total_items : ℕ) (hours : ℕ) (minutes_in_an_hour : ℕ) (h : total_items = 360) (h2 : hours = 2) (h3 : minutes_in_an_hour = 60): 
  (2 * 60) / 360 = 1 / 3 := 
by
  rw [mul_comm]
  norm_num
  sorry

end time_per_item_l826_826147


namespace minsu_age_l826_826074

theorem minsu_age :
  ∃ (M M_m : ℕ), (M_m - M = 28) ∧ (M_m + 13 = 2 * (M + 13)) ∧ (M = 15) :=
by
  use 15, 43 
  split
  · show 43 - 15 = 28 by sorry
  split
  · show 43 + 13 = 2 * (15 +13) by sorry
  · show 15 = 15 by rfl

end minsu_age_l826_826074


namespace number_of_different_sums_l826_826613

-- Define the coin types and their quantities
def coins : List (ℕ × ℕ) :=
  [(1, 1), (2, 1), (5, 1), (10, 4), (50, 2)]

-- Define the set of possible sums that can be formed from these coins
def possible_sums (coins : List (ℕ × ℕ)) : Finset ℕ :=
  coins.foldl (λ acc coin, let (v, n) := coin in 
                    Finset.bind acc (λ sum, Finset.image (λ k, sum + k * v) (Finset.range (n + 1))))
            (Finset.singleton 0)

-- Statement to prove the number of different sums that can be formed using the given coins
theorem number_of_different_sums : (possible_sums coins).card = 120 :=
by
  sorry

end number_of_different_sums_l826_826613


namespace num_perfect_square_factors_gt_one_l826_826294

theorem num_perfect_square_factors_gt_one {S : Set ℕ} (hS : S = { n | n ∈ finset.range 101 }) :
  (∃ n ∈ S, ∃ m > 1, ∃ k : ℕ, m * m = k ∧ k ∣ n) → finset.card { n ∈ S | ∃ m > 1, m * m ∣ n } = 40 :=
by
  sorry

end num_perfect_square_factors_gt_one_l826_826294


namespace log_max_value_l826_826560

noncomputable theory
open Real

theorem log_max_value (a b : ℝ) (ha : a > b) (hb : b ≥ 2) : 
  log a (a^2 / b) + log b (b^2 / a) ≤ 2 :=
begin
  sorry -- proof here
end

example : ∃ a b : ℝ, a > b ∧ b ≥ 2 ∧ log a (a^2 / b) + log b (b^2 / a) = 2 :=
begin
  use [3, 2], -- Let's use specific values to demonstrate
  split,
  exact by norm_num, -- 3 > 2
  split,
  exact by norm_num, -- 2 ≥ 2
  calc log 3 (3^2 / 2) + log 2 (2^2 / 3)
      = log 3 4.5 + log 2 (4 / 3) : by norm_num
  ... = 2 : by {some_concrete_proof_for_log}
end

end log_max_value_l826_826560


namespace hyperbola_equation_l826_826455

noncomputable def focus_distance (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem hyperbola_equation (a b : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (c := focus_distance a b)
  (h_pf : ∀ x y, 4 * b * (b^2 + x * a) = 2 * (real.sqrt (2) * c * (c + x * a)))
  (h_slope : ∀ x a b c, (b / c) / ((a^2 / c) + c) = (real.sqrt 2 / 4)) :
  (a = real.sqrt 2 ∧ b = 2) →
  (∃ h : \frac{x^2}{2} - \frac{y^2}{4} = 1, h := sorry) :=
by
  sorry

end hyperbola_equation_l826_826455


namespace sequence_an_sum_bn_l826_826283

theorem sequence_an (a : ℕ → ℝ)
  (h : ∀ n : ℕ, 0 < n → (∑ k in finset.range n.succ, k.succ * a k.succ) = 4 - (n + 2) / 2 ^ (n - 1)) :
  ∀ n : ℕ, 0 < n → a n = 1 / 2 ^ (n - 1) :=
begin
  sorry
end

theorem sum_bn (a b : ℕ → ℝ) (S : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < n → (∑ k in finset.range n.succ, k.succ * a k.succ) = 4 - (n + 2) / 2 ^ (n - 1))
  (hb : ∀ n : ℕ, 0 < n → b n = (3 * n - 2) * a n)
  (hS : ∀ n : ℕ, S n = ∑ k in finset.range n.succ, b k.succ) :
  ∀ n : ℕ, 0 < n → S n = 8 - (3 * n + 4) / 2 ^ (n - 1) :=
begin
  sorry
end

end sequence_an_sum_bn_l826_826283


namespace similar_cyclic_quadrilateral_l826_826349

variables {A B C D K L M N S T O Ha Hb Hc Hd : Type*}
variable [metric_space (Type*)]

noncomputable def cyclic_quadrilateral (A B C D : Type*) := sorry

noncomputable def midpoint (x y : Type) := sorry

noncomputable def circumcenter (x y z : Type) := sorry

-- Define that A, B, C, D is a cyclic quadrilateral
axiom abcd_is_cyclic : cyclic_quadrilateral A B C D

-- Define midpoints
axiom K_midpoint : K = midpoint A B
axiom L_midpoint : L = midpoint B C
axiom M_midpoint : M = midpoint C D
axiom N_midpoint : N = midpoint D A
axiom S_midpoint : S = midpoint A C
axiom T_midpoint : T = midpoint B D

-- Define circumcenters of the circumcircle of triangles formed by midpoints
axiom circumcenter_KLS : circumcenter K L S
axiom circumcenter_LMT : circumcenter L M T
axiom circumcenter_MNS : circumcenter M N S
axiom circumcenter_NKT : circumcenter N K T

-- The theorem to prove similarity of quadrilaterals
theorem similar_cyclic_quadrilateral :
  cyclic_quadrilateral (circumcenter K L S) (circumcenter L M T) (circumcenter M N S) (circumcenter N K T) :=
sorry

end similar_cyclic_quadrilateral_l826_826349


namespace fewest_tiles_needed_l826_826882

open scoped Classical

/-- A rectangular tile measures 3 inches by 4 inches. 
A rectangular region is 3 feet by 6 feet, with a square area of 1 foot by 1 foot already covered.
This Lean statement states that the fewest number of these tiles needed to completely cover 
the remaining area is 204. -/
theorem fewest_tiles_needed : 
  let tile_area := 3 * 4
  let total_area := (3 * 12) * (6 * 12)
  let covered_area := (1 * 12) * (1 * 12)
  let remaining_area := total_area - covered_area
  let tiles_needed := remaining_area / tile_area
  in tiles_needed = 204 :=
by
  simp only [*, mul_comm, mul_assoc, nat.div_eq_of_eq_mul_right, nat.div_self (by norm_num : 12 ≠ 0)]
  sorry

end fewest_tiles_needed_l826_826882


namespace minimal_AC_value_l826_826187

theorem minimal_AC_value (AC CD : ℕ) (h : ∃ (CD : ℕ), BD = sqrt 77 ∧ AC = (CD^2 + 77) / (2 * CD)) : AC = 12 :=
begin
  sorry
end

end minimal_AC_value_l826_826187


namespace average_score_of_nine_l826_826326

def total_score (average : ℕ) (n : ℕ) : ℕ := average * n

def other_score (total : ℕ) (xiaoming_score : ℕ) : ℕ := total - xiaoming_score

def average_of_nine (total_of_nine : ℕ) (n : ℕ) : ℕ := total_of_nine / n

theorem average_score_of_nine :
  let average := 84
  let n := 10
  let xiaoming_score := 93
  let total := total_score average n
  let total_of_nine := other_score total xiaoming_score
  let n_nine := 9
  average_of_nine total_of_nine n_nine = 83 :=
by
  let average := 84
  let n := 10
  let xiaoming_score := 93
  let total := total_score average n
  let total_of_nine := other_score total xiaoming_score
  let n_nine := 9
  have h_total : total = 840 := rfl
  have h_total_of_nine : total_of_nine = 747 := rfl
  have h_average : average_of_nine total_of_nine n_nine = 83 := rfl
  exact h_average

end average_score_of_nine_l826_826326


namespace cost_of_first_20_kgs_l826_826175

theorem cost_of_first_20_kgs (l q : ℕ)
  (h1 : 30 * l + 3 * q = 168)
  (h2 : 30 * l + 6 * q = 186) :
  20 * l = 100 :=
by
  sorry

end cost_of_first_20_kgs_l826_826175


namespace abs_less_than_2_sufficient_but_not_necessary_l826_826799

theorem abs_less_than_2_sufficient_but_not_necessary (x : ℝ) : 
  (|x| < 2 → (x^2 - x - 6 < 0)) ∧ ¬(x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_less_than_2_sufficient_but_not_necessary_l826_826799


namespace number_of_throwers_l826_826763

-- Definitions of the entities in the problem
variable {players throwers non_throwers right_handed left_handed : ℕ}

-- Given conditions
def total_players := players = 70
def all_throwers_right_handed := ∀ t, t < throwers → right_handed > t
def one_third_left_handed := left_handed = (1 / 3) * (players - throwers)
def total_right_handed := right_handed = 56
def non_thrower_right_handed := right_handed - throwers = (2 / 3) * (players - throwers)

-- Theorem to prove the number of throwers
theorem number_of_throwers (h1 : total_players)
                            (h2 : all_throwers_right_handed)
                            (h3 : one_third_left_handed)
                            (h4 : total_right_handed)
                            (h5 : non_thrower_right_handed) :
  throwers = 28 := 
sorry

end number_of_throwers_l826_826763


namespace lauren_telephone_count_l826_826162

def laurensTelephoneNumbers : ℕ :=
  let totalDigits := 9 -- Digits from 1 to 9
  let chooseDigits := 6 -- Choose 6 unique digits
  let numWaysChooseDigits := Nat.choose totalDigits chooseDigits
  let numWaysRepeatDigit := chooseDigits -- One of the 6 digits to repeat
  let repeatPositions := Nat.choose 3 1 * Nat.choose 5 1 -- positions for repeated digit
  numWaysChooseDigits * numWaysRepeatDigit * repeatPositions

theorem lauren_telephone_count : laurensTelephoneNumbers = 7560 := by
  sorry

end lauren_telephone_count_l826_826162


namespace simplify_poly_l826_826438

open Polynomial

variable {R : Type*} [CommRing R]

-- Definition of the polynomials
def p1 : Polynomial R := 2 * X ^ 6 + 3 * X ^ 5 + X ^ 4 - X ^ 2 + 15
def p2 : Polynomial R := X ^ 6 + X ^ 5 - 2 * X ^ 4 + X ^ 3 + 5

-- Simplified polynomial
def expected_result : Polynomial R := X ^ 6 + 2 * X ^ 5 + 3 * X ^ 4 - X ^ 3 + X ^ 2 + 10

-- The theorem to state the equivalence
theorem simplify_poly : p1 - p2 = expected_result :=
by sorry

end simplify_poly_l826_826438


namespace range_of_a_l826_826678

variable (f : ℝ → ℝ)
variable (a : ℝ)

theorem range_of_a (h : set.range f = set.Icc (-1/2) 1) : set.range a = set.Icc (-0.3398) 0.3398 :=
sorry

end range_of_a_l826_826678


namespace native_character_l826_826847

universe u

def QuestionNature :=
  | human
  | zombie
  | polu_zombie

structure Native (A : Type u) where
  population : A -> QuestionNature
  mutable lies : A -> Bool

noncomputable def determine_native (response : String) (lang : String) : QuestionNature :=
  if response = "нет" then QuestionNature.polu_zombie
  else Sorry  -- remainder of logic based on given conditions

axiom conditions {A : Type u} (native : Native A)
  (population_types : A -> QuestionNature)
  (polu_lie : A -> Bool)
  (word_meaning : String -> String)
  (answers_english_or_native : A -> Bool)
  (inspector_question : String)
  (unknown_response_lang : String) : Prop

theorem native_character {A : Type u}
  (native : Native A)
  (response : String)
  (lang : String)
  (correct_answer : response = "нет")
  (determined_character : determine_native response lang = QuestionNature.polu_zombie) : Prop := 
  sorry

end native_character_l826_826847


namespace polygon_sides_l826_826768

-- Definition of the problem conditions
def interiorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)
def givenAngleSum (n : ℕ) : ℕ := 140 + 145 * (n - 1)

-- Problem statement: proving the number of sides
theorem polygon_sides (n : ℕ) (h : interiorAngleSum n = givenAngleSum n) : n = 10 :=
sorry

end polygon_sides_l826_826768


namespace solve_fraction_equation_l826_826442

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7) + 1 / (2 * x + 36)) →
  (x ≈ -6.614) ∨ (x ≈ -21.386) :=
by
  sorry

end solve_fraction_equation_l826_826442


namespace sum_of_remainders_l826_826094

theorem sum_of_remainders (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 25) (h3 : c % 53 = 6) (h4 : d % 53 = 12) : 
  (a + b + c + d) % 53 = 23 :=
by {
  sorry
}

end sum_of_remainders_l826_826094


namespace scheduling_courses_in_non_consecutive_periods_l826_826297

theorem scheduling_courses_in_non_consecutive_periods :
  (∃ (n m : ℕ), n = 4 ∧ m = 7 ∧ 
   (∑ v in (finset.filter (λ s, s.card = n ∧ ∀ x y ∈ s, x ≠ y ∧ abs (x - y) ≠ 1) (finset.powerset (finset.range m))), 
   finset.prod v (λ _, 1)) * nat.factorial n = 96) :=
sorry

end scheduling_courses_in_non_consecutive_periods_l826_826297


namespace product_expression_evaluation_l826_826923

theorem product_expression_evaluation :
  (1 + 2 / 1) * (1 + 2 / 2) * (1 + 2 / 3) * (1 + 2 / 4) * (1 + 2 / 5) * (1 + 2 / 6) - 1 = 25 / 3 :=
by
  sorry

end product_expression_evaluation_l826_826923


namespace common_internal_tangent_length_l826_826035

noncomputable def length_common_internal_tangent
  (d r1 r2 : ℝ) : ℝ :=
  real.sqrt (d^2 - (r1 + r2)^2)

theorem common_internal_tangent_length (d : ℝ) (r1 : ℝ) (r2 : ℝ) :
  d = 50 → r1 = 7 → r2 = 10 → length_common_internal_tangent d r1 r2 = real.sqrt 2211 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold length_common_internal_tangent
  simp [real.sqrt]


end common_internal_tangent_length_l826_826035


namespace find_general_term_and_sum_l826_826266

variable {N : Type} [Nat N]

-- Definitions based on conditions from part a)
def arithmetic_sequence (a : Nat → Int) : Prop :=
  ∀ n, a (n + 1) = a n + d

def initial_conditions (a : Nat → Int) : Prop :=
  a 1 = 1 ∧ a 4 = 7

-- Lean statement for the problem
theorem find_general_term_and_sum (a : Nat → Int) (d : Int) (n : Nat) 
    (arith_seq : arithmetic_sequence a)
    (init_cond : initial_conditions a) :
  (∀ n, a n = 2 * n - 1) ∧ (a 2 + a 6 + a 10 + ... + a (4 * n + 10) = (n + 3) * (4 * n + 11)) :=
by
  sorry  -- Proof goes here

end find_general_term_and_sum_l826_826266


namespace complex_moduli_sum_l826_826203

theorem complex_moduli_sum :
  abs (Complex.mk 3 (-5)) + abs (Complex.mk 3 5) = 2 * Real.sqrt 34 :=
by
  sorry

end complex_moduli_sum_l826_826203


namespace probability_minimal_S_l826_826760

def ball_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def equidistant_points (n: ℕ) : List ℕ := List.range n 

def sum_absolute_differences (l : List ℕ) : ℕ :=
  (List.zipWith (λ a b => abs (a - b)) l (List.rotate 1 l)).sum

def distinct_circular_permutations (l : List ℕ) : Multiset (List ℕ) :=
  (l.permutations.map (λ perm => perm.rotateRight 1)).toMultiset

def minimal_S (l : List ℕ) : Prop :=
  sum_absolute_differences l = 16

def count_distinct_minimal_arrangements (l : List ℕ) : ℕ :=
  (distinct_circular_permutations l).countp minimal_S

theorem probability_minimal_S (unique_arrangements: ℕ) (distinct_minimal_arrangements : ℕ) : 
  (unique_arrangements = 20160) → 
  (distinct_minimal_arrangements = 64) →
  (unique_arrangements / distinct_minimal_arrangements = 315) :=
sorry

end probability_minimal_S_l826_826760


namespace tan_addition_max_value_f_l826_826627

theorem tan_addition (alpha beta : ℝ) (h1 : Real.tan alpha = -(1/3)) (h2 : Real.cos beta = (Real.sqrt 5) / 5) (h3 : 0 < alpha ∧ alpha < Real.pi) (h4 : 0 < beta ∧ beta < Real.pi) :
  Real.tan (alpha + beta) = 1 := 
sorry

theorem max_value_f (alpha beta : ℝ) (h1 : Real.tan alpha = -(1/3)) (h2 : Real.cos beta = (Real.sqrt 5) / 5) (h3 : 0 < alpha ∧ alpha < Real.pi) (h4 : 0 < beta ∧ beta < Real.pi) :
  ∃ x : ℝ, ∀ y : ℝ, (sqrt 2 * Real.sin (y - alpha) + Real.cos (y + beta)) ≤ sqrt 5 ∧ (sqrt 2 * Real.sin (x - alpha) + Real.cos (x + beta)) = sqrt 5 := 
sorry

end tan_addition_max_value_f_l826_826627


namespace product_inequality_l826_826431

theorem product_inequality (n : ℕ) (h : n > 1) : 
  ∏ (k : ℕ) in finset.range (n + 1), k^k < n^(n * (n + 1) / 2) := 
begin
  sorry
end

end product_inequality_l826_826431


namespace find_n_tangent_l826_826955

theorem find_n_tangent (n : ℤ) (h1 : -180 < n) (h2 : n < 180) (h3 : ∃ k : ℤ, 210 = n + 180 * k) : n = 30 :=
by
  -- Proof steps would go here
  sorry

end find_n_tangent_l826_826955


namespace corner_cells_difference_divisible_by_6_l826_826827

theorem corner_cells_difference_divisible_by_6 :
  ∃ (a b : ℕ), 
  a ≠ b ∧ 
  ∀ i j, (i = a ∨ i = b) → 1 ≤ i ∧ i ≤ 81 ∧ 
  (∀ m n : ℕ, ∀ x y : ℕ, (m ≠ n ∧ m % 3 = n % 3) → 
    ((x = m ∧ y = n) →
     (∀ r s : ℕ, (r = x - 1 ∧ s = y) ∨ (r = x + 1 ∧ s = y) ∨
                  (r = x ∧ s = y - 1) ∨ (r = x ∧ s = y + 1) → 
                  ((abs (r - s) = 3))) ∨ 
     (x = m ∨ y = n))),
(a - b) % 6 = 0 :=
sorry

end corner_cells_difference_divisible_by_6_l826_826827


namespace petya_numbers_l826_826399

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826399


namespace father_present_age_l826_826515

theorem father_present_age (S F : ℕ) 
  (h1 : F = 3 * S + 3) 
  (h2 : F + 3 = 2 * (S + 3) + 10) : 
  F = 33 :=
by
  sorry

end father_present_age_l826_826515


namespace sum_of_elements_in_T_in_base3_l826_826730

def T : Set ℕ := {n | ∃ (d1 d2 d3 d4 d5 : ℕ), (1 ≤ d1 ∧ d1 ≤ 2) ∧ (0 ≤ d2 ∧ d2 ≤ 2) ∧ (0 ≤ d3 ∧ d3 ≤ 2) ∧ (0 ≤ d4 ∧ d4 ≤ 2) ∧ (0 ≤ d5 ∧ d5 ≤ 2) ∧ n = d1 * 3^4 + d2 * 3^3 + d3 * 3^2 + d4 * 3^1 + d5}

theorem sum_of_elements_in_T_in_base3 : (∑ x in T, x) = 2420200₃ := 
sorry

end sum_of_elements_in_T_in_base3_l826_826730


namespace PQ_parallel_to_axes_l826_826986

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

def point_P : ℝ × ℝ := (√2, √2 / 2)

def line_l (x m : ℝ) : ℝ := (1 / 2) * x + m

def intersects_ellipse (x y : ℝ) : Prop := ellipse x y ∧ ∃ m, y = line_l x m

def angle_bisector_intersects (A P B Q : ℝ × ℝ) : Prop := 
  ∃ (k₁ k₂ : ℝ × ℝ → ℝ), k₁ P A = k₂ P B ∧ k₂ P Q = k₂ A Q

theorem PQ_parallel_to_axes (A B Q : ℝ × ℝ) :
  ellipse (fst point_P) (snd point_P) →
  intersects_ellipse (fst A) (snd A) →
  intersects_ellipse (fst B) (snd B) →
  angle_bisector_intersects A point_P B Q →
  (fst point_P = fst Q ∨ snd point_P = snd Q) :=
by sorry

end PQ_parallel_to_axes_l826_826986


namespace position_XENBT_l826_826825

theorem position_XENBT : 
  let letters := ['B', 'E', 'N', 'T', 'X'],
      word := "XENBT",
      position := 115
  in (position = (∑ l in letters, if l < 'X' then 24 else 0) + ∑ (s : string) in (list.permutations (letters.filter (λ c => c ≠ 'X'))), if s < "ENBT" then 1 else 0 + 1) :=
rfl

end position_XENBT_l826_826825


namespace find_y_l826_826705

-- Definitions and conditions
def is_square (A B C D : Point) : Prop := 
  -- (Define what it means for ABCD to be a square)
  sorry

def equal_side_length (ABCD DEFG : Square) : Prop := 
  -- (Define what it means for two squares to have equal side length)
  sorry

def isosceles_triangle (D C E : Point) : Prop :=
  DC = DE ∧ ⦟DCE = 70

-- Lean theorem statement
theorem find_y
  (A B C D E F G : Point) 
  (ABCD : is_square A B C D)
  (DEFG : is_square D E F G)
  (eq_sides : equal_side_length ABCD DEFG)
  (angle_DCE : ⦟DCE = 70) :
  ∃ y : ℝ, y = 140 :=
begin
  sorry
end

end find_y_l826_826705


namespace cube_expansion_l826_826850

variable {a b : ℝ}

theorem cube_expansion (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 :=
  sorry

end cube_expansion_l826_826850


namespace petya_numbers_l826_826397

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826397


namespace xy_minus_x_plus_4y_eq_15_l826_826459

theorem xy_minus_x_plus_4y_eq_15 (x y : ℕ) :
  xy - x + 4y = 15 ↔ (x = 11 ∧ y = 2) ∨ (x = 1 ∧ y = 4) :=
sorry

end xy_minus_x_plus_4y_eq_15_l826_826459


namespace exists_x_y_divisible_l826_826778

theorem exists_x_y_divisible (n : ℕ) : ∃ (x y : ℤ), n ∣ (x^2 + y^2 - 2018) :=
by {
  use [43, 13],
  simp,
  sorry
}

end exists_x_y_divisible_l826_826778


namespace inequality_one_inequality_two_l826_826245

variable {a b c : ℝ}

-- Prove the first inequality
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
    sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c := 
sorry

-- Prove the second inequality, given the additional condition
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * c * a) / (c + a) ≤ 1 := 
sorry

end inequality_one_inequality_two_l826_826245


namespace sequence_increasing_l826_826585

open Real

theorem sequence_increasing (a_0 : ℝ) (hn : ∀ n : ℕ, n > 0 → 2^n - 3*a_0 > 3 ^ n * ((-1 : ℝ) ^ n * a_0 - (-1 : ℝ) ^ (n - 1) * a_0)) :
  a_0 = 1/5 → (∀ n : ℕ, n > 0 → a_{n+1} > a_n) :=
begin
  intro h,
  sorry
end

end sequence_increasing_l826_826585


namespace no_real_roots_l826_826787

open Real

def factorial (n : ℕ) : ℝ :=
  if n = 0 then 1 else nat.factorial n

noncomputable def series (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range (2 * n + 1), (x ^ i) / (factorial i)

theorem no_real_roots (x : ℝ) (n : ℕ) : series x n ≠ 0 := 
  sorry

end no_real_roots_l826_826787


namespace larger_angle_decrease_percentage_l826_826050

-- Definitions for the conditions
def is_complementary (a b : ℝ) : Prop := a + b = 90
def ratio (a b : ℝ) (r s : ℝ) : Prop := a / b = r / s
def percent_increase (x p : ℝ) : ℝ := x * (1 + p / 100)
def percent_decrease (y q : ℝ) : ℝ := y * (1 - q / 100)

theorem larger_angle_decrease_percentage 
  (a b : ℝ) (r s : ℝ) (p : ℝ) (q : ℝ)
  (h1 : is_complementary a b)
  (h2 : ratio a b r s)
  (h3 : r = 3)
  (h4 : s = 6)
  (h5 : p = 20)
  (h6 : percent_increase a p + percent_decrease b q = 90) :
  q = 10 :=
by
  sorry

end larger_angle_decrease_percentage_l826_826050


namespace percentage_with_no_conditions_l826_826888

-- Define the sets and cardinalities
variables (A B C : Type) [Fintype A] [Fintype B] [Fintype C]

-- Survey data as specific cardinalities
def high_bp : ℕ := 110
def heart_trouble : ℕ := 80
def cholesterol_issues : ℕ := 50
def high_bp_and_heart_trouble : ℕ := 30
def heart_trouble_and_cholesterol_issues : ℕ := 20
def high_bp_and_cholesterol_issues : ℕ := 10
def all_three_conditions : ℕ := 5
def total_surveyed : ℕ := 200

theorem percentage_with_no_conditions :
  (total_surveyed - (high_bp + heart_trouble + cholesterol_issues 
    - high_bp_and_heart_trouble - heart_trouble_and_cholesterol_issues 
    - high_bp_and_cholesterol_issues + all_three_conditions)) * 100 / total_surveyed = 7.5 := 
sorry

end percentage_with_no_conditions_l826_826888


namespace num_elements_in_arith_seq_l826_826661

noncomputable def arith_seq_first_term : ℝ := 2.5
noncomputable def arith_seq_common_diff : ℝ := 5
noncomputable def arith_seq_last_term : ℝ := 62.5

theorem num_elements_in_arith_seq :
  ∃ n : ℕ, n = ((arith_seq_last_term - arith_seq_first_term) / arith_seq_common_diff).to_nat + 1 :=
begin
  sorry
end

end num_elements_in_arith_seq_l826_826661


namespace plane_hit_probability_l826_826526

theorem plane_hit_probability :
  let P_A : ℝ := 0.3
  let P_B : ℝ := 0.5
  let P_not_A : ℝ := 1 - P_A
  let P_not_B : ℝ := 1 - P_B
  let P_both_miss : ℝ := P_not_A * P_not_B
  let P_plane_hit : ℝ := 1 - P_both_miss
  P_plane_hit = 0.65 :=
by
  sorry

end plane_hit_probability_l826_826526


namespace determine_k_l826_826049

-- Define the roots of the quartic equation
variables (a b c d : ℝ)

-- Define the conditions
def quartic_equation_roots := (a + b + c + d = 18) ∧ 
                             (a * b * c * d = -1984) ∧ 
                             (a * b * c + a * b * d + a * c * d + b * c * d = -200) ∧ 
                             (a * b + a * c + a * d + b * c + b * d + c * d = k) ∧ 
                             (a * b = -32)

-- The proposition to prove
theorem determine_k (h : quartic_equation_roots a b c d) : k = 86 :=
sorry

end determine_k_l826_826049


namespace rationalize_denominator_l826_826785

noncomputable def sqrt3 (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := sqrt3 8 in
  let b := sqrt3 7 in
  (a = 2) →
  ∀ x y : ℝ, x = 2 - b → y = (2^2 + 2 * b + (b^2)) →
  (x * y = 1) →
  ∃ A B C D : ℕ, A + B + C + D = 14 :=
by sorry

end rationalize_denominator_l826_826785


namespace time_spent_at_destination_l826_826003

-- Given conditions
def destination_distance : ℕ := 55
def additional_return_distance : ℕ := 10
def speed_miles_per_2_minutes : ℕ := 1
def total_tour_time_hours : ℕ := 6

-- Proof goal
theorem time_spent_at_destination : 
  let return_trip_distance := destination_distance + additional_return_distance in
  let total_distance := destination_distance + return_trip_distance in
  let driving_time_minutes := total_distance * 2 in
  let driving_time_hours := driving_time_minutes / 60 in
  total_tour_time_hours - driving_time_hours = 2 :=
by
  sorry

end time_spent_at_destination_l826_826003


namespace eagles_per_section_l826_826920

theorem eagles_per_section (total_eagles sections : ℕ) (h1 : total_eagles = 18) (h2 : sections = 3) :
  total_eagles / sections = 6 := by
  sorry

end eagles_per_section_l826_826920


namespace solve_for_x_l826_826833

-- Define the problem with the given conditions
def sum_of_triangle_angles (x : ℝ) : Prop := x + 2 * x + 30 = 180

-- State the theorem
theorem solve_for_x : ∀ (x : ℝ), sum_of_triangle_angles x → x = 50 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l826_826833


namespace exists_sequence_with_gcd_property_l826_826580

theorem exists_sequence_with_gcd_property :
  ∃ (a : ℕ → ℕ), (strict_mono a) ∧ (∀ i (hi : i < 100), (Nat.gcd (a i) (a (i+1)) > Nat.gcd (a (i+1)) (a (i+2)))) :=
by 
  sorry

end exists_sequence_with_gcd_property_l826_826580


namespace maximum_real_part_sum_l826_826370

def complex_roots := {z : ℂ // ∃ k : ℕ, k < 12 ∧ z = 8 * complex.exp(2 * k * complex.pi * complex.I / 12)}

def w_j (z : ℂ) : ℂ :=
if z.im >= 0 then z else complex.I * z

theorem maximum_real_part_sum :
  ∃ w : (fin 12) → ℂ,
    (∀ j, w j = w_j (root_of_unity 12 j)) ∧
    real.re (∑ j in finrange 12, w j) = 16 * (1 + real.sqrt 3) :=
sorry

end maximum_real_part_sum_l826_826370


namespace petya_numbers_l826_826393

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826393


namespace problem_solution_l826_826576

open Finset

def minimally_intersecting (A B C : Finset ℕ) : Prop :=
  |A ∩ B| = 1 ∧ |B ∩ C| = 1 ∧ |C ∩ A| = 1 ∧ (A ∩ B ∩ C) = ∅

def N : ℕ :=
  ((Finset.powerset (range 8)).filter (λ A B C, minimally_intersecting A B C)).card

theorem problem_solution : N % 1000 = 344 :=
sorry

end problem_solution_l826_826576


namespace hexagon_toothpicks_l826_826083

theorem hexagon_toothpicks (n : ℕ) (h : n = 6) : 6 * n = 36 :=
by
  rw [h]
  exact rfl

end hexagon_toothpicks_l826_826083


namespace abs_difference_extrema_l826_826098

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l826_826098


namespace total_matches_played_l826_826151

def home_team_wins := 3
def home_team_draws := 4
def home_team_losses := 0
def rival_team_wins := 2 * home_team_wins
def rival_team_draws := home_team_draws
def rival_team_losses := 0

theorem total_matches_played :
  home_team_wins + home_team_draws + home_team_losses + rival_team_wins + rival_team_draws + rival_team_losses = 17 :=
by
  sorry

end total_matches_played_l826_826151


namespace train_length_l826_826510

def speed_km_per_hr := 40 -- Speed in km/hr
def time_to_cross_post := 19.8 -- Time in seconds
def length_of_train := 220.178 -- The expected length of the train in meters

theorem train_length :
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  let length := speed_m_per_s * time_to_cross_post in
  length = length_of_train :=
by
  sorry

end train_length_l826_826510


namespace all_terms_perfect_squares_l826_826652

noncomputable theory

-- Define the sequence of integers aₙ
def sequence (a : ℕ → ℤ) : Prop :=
  (∀ n ≥ 2, a (n + 1) = 3 * a n - 3 * a (n - 1) + a (n - 2)) ∧
  (2 * a 1 = a 0 + a 2 - 2) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, ∀ i ∈ finset.range m, nat.perfect_square (a (k + i)))

-- The theorem we need to prove
theorem all_terms_perfect_squares (a : ℕ → ℤ) (h : sequence a) :
  ∀ n : ℕ, nat.perfect_square (a n) :=
begin
  sorry
end

end all_terms_perfect_squares_l826_826652


namespace length_P2013_P2014_l826_826738

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.2 - p.1, p.2 + p.1)

-- Define the sequence of points P_n
def P : ℕ+ → ℝ × ℝ
| 1 := (0, 2)
| (n+1) := f (P n)

-- Calculate the distance between two points
def distance (a b : ℝ × ℝ) : ℝ :=
(real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2))

-- Theorem statement for the length of the line segment between P_2013 and P_2014
theorem length_P2013_P2014 : distance (P 2013) (P 2014) = 2^1007 :=
by
  sorry

end length_P2013_P2014_l826_826738


namespace unpainted_unit_cubes_l826_826128

/-- 
A $6 \times 6 \times 6$ cube is formed by assembling 216 unit cubes. Each of the six faces of the 
cube has a cross pattern painted on it, where the middle four unit squares and one additional unit 
square extending along each direction from the middle form the pattern, totaling 10 painted unit 
squares on each face. Prove that the number of unit cubes that have no paint on them is 180.
-/
theorem unpainted_unit_cubes (n : ℕ) (cubes faces middle additional : ℕ) 
  (h1 : n = 6 * 6 * 6) 
  (h2 : cubes = 216) 
  (h3 : faces = 6) 
  (h4 : middle = 4) 
  (h5 : additional = 6) 
  (h6 : faces * (middle + additional) - 24 = 36) 
  (h7 : cubes - 36 = 180) : 
  n = cubes → 
  faces = 6 → 
  faces * (middle + additional) = 60 → 
  faces * (middle + additional) - 24 - 12 = 36 → 
  cubes - 36 = 180 := 
by 
  intro h8 h9 h10 h11 h12 
  rw [h8, h9, h10, h11, h12]
  sorry

end unpainted_unit_cubes_l826_826128


namespace triangle_count_quadrilateral_count_l826_826384

theorem triangle_count (points_line1 points_line2 : ℕ) (h1 : points_line1 = 10) (h2 : points_line2 = 11) : 
  (points_line1 * Nat.choose points_line2 2 + points_line2 * Nat.choose points_line1 2) = 1045 :=
by
  rw [h1, h2]
  norm_num

theorem quadrilateral_count (points_line1 points_line2 : ℕ) (h1 : points_line1 = 10) (h2 : points_line2 = 11) : 
  (Nat.choose points_line1 2 * Nat.choose points_line2 2) = 2475 :=
by
  rw [h1, h2]
  norm_num

end triangle_count_quadrilateral_count_l826_826384


namespace circumcenter_DEX_on_perpendicular_bisector_of_OA_l826_826140

-- Define cyclic quadrilateral with circumcenter O
variable (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X]

-- Define D on line BX such that |AD| = |BD|
variable { D : Type } [MetricSpace D]
variable (AD_eq_BD : dist (A, D) = dist (B, D))

-- Define E on line CX such that |AE| = |CE|
variable { E : Type } [MetricSpace E]
variable (AE_eq_CE : dist (A, E) = dist (C, E))

-- Given that circumcenter O of quadrilateral ABXC
variable { O : Type } [MetricSpace O]
variable (is_circumcenter_O : is_circumcenter O A B X C)

-- Prove that the circumcenter of triangle DEX lies on the perpendicular bisector of OA
theorem circumcenter_DEX_on_perpendicular_bisector_of_OA : 
  (is_circumcenter O A B X C) → (dist A D = dist B D) → (dist A E = dist C E) →
  lies_on_perpendicular_bisector (circumcenter (triangle D E X)) (segment O A) :=
by
  -- Proof will be provided here
  sorry

end circumcenter_DEX_on_perpendicular_bisector_of_OA_l826_826140


namespace checkerboard_squares_with_black_cells_l826_826009

theorem checkerboard_squares_with_black_cells :
  ∃ squares : ℕ, squares = 97 ∧
    ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 10) → 
    ∑ (i : ℕ) in finset.range (11 - k), 
    ∑ (j : ℕ) in finset.range (11 - k), 
    ((k^2 / 2 ≤ 8) → 0) + 
    ((k^2 / 2 > 8) → (if even (i + j) then k else 0)) = squares := 
  by
    sorry

end checkerboard_squares_with_black_cells_l826_826009


namespace weeds_never_cover_entire_field_l826_826884

-- Define the initial conditions
def field_side_length : ℝ := 13.87
def num_plots : ℕ := 100
def initial_weedy_plots : ℕ := 9

-- Define the condition for weed spreading
def weed_spread_condition (adjacent_weedy_plots : ℕ) : Prop :=
  adjacent_weedy_plots >= 2

-- Define the maximum possible boundary
def max_boundary_length (num_plots : ℕ) : ℕ :=
  4 * (num_plots.to_nat.sqrt)

-- Theorem: Weeds will never cover the entire field
theorem weeds_never_cover_entire_field
  (f_side_len : ℝ) (n_plots init_weeds : ℕ)
  (spread_cond : ∀ p, p >= 2 → Prop)
  (max_boundary : ℕ) :
  f_side_len = 13.87 →
  n_plots = 100 →
  init_weeds = 9 →
  spread_cond 2 →
  max_boundary = max_boundary_length n_plots →
  ∀ (current_weeds : ℕ), (current_weeds ≤ init_weeds) →
  ¬(∀ (spread: ℕ), (spread >= 2 → current_weeds = n_plots)) :=
by
  intros f_side_len n_plots init_weeds spread_cond max_boundary H1 H2 H3 H4 H5 H6
  sorry

end weeds_never_cover_entire_field_l826_826884


namespace quadratic_root_difference_squared_l826_826735

theorem quadratic_root_difference_squared :
  let a := 1 / 2 in
  let b := 3 in
  let h1 : 2 * a ^ 2 - 7 * a + 3 = 0 := by norm_num,
  let h2 : 2 * b ^ 2 - 7 * b + 3 = 0 := by norm_num,
  (a - b)^2 = 25 / 4 :=
sorry

end quadratic_root_difference_squared_l826_826735


namespace correct_statements_l826_826478

-- Definitions based on conditions
def is_proposition (stmt : Prop) : Prop :=
  ∃ b : Bool, stmt = to_bool b

def satisfiable_ineq (x : ℝ) : Prop := x^3 + 1 ≤ 0

def negation (p : Prop) : Prop := ¬p

def conjunction (p q : Prop) : Prop := p ∧ q

def equivalence_condition (m : ℝ) : Prop := 
  ∀ x : ℝ, (m * x - 2 > 0) ↔ (x - 2 > 0)

-- Lean 4 statement for the proof problem
theorem correct_statements :
  (¬ is_proposition (is_proposition true)) ∧
  (∃ x : ℝ, satisfiable_ineq x) ∧
  (negation (¬ (∀ x y z : ℝ, ¬ (x > 0 ∧ y > 0 ∧ z > 0)))) ∧
  ¬ (∃ (p q : Prop), p ∧ ¬q ∧ conjunction p ¬q) ∧
  equivalence_condition 1 :=
by
  sorry

end correct_statements_l826_826478


namespace cube_surface_area_l826_826672

theorem cube_surface_area (P : ℝ) (h : P = 52) : 
  let s := P / 4 in
  let A := s^2 in
  let SA := 6 * A in
  SA = 1014 := 
by
  sorry

end cube_surface_area_l826_826672


namespace proof_problem_l826_826668

-- Define the variables and conditions
variables {a b : ℝ}
axiom (h1 : 0 < a ∧ a < 1)
axiom (h2 : log b a < 1)

-- Define the theorem statement
theorem proof_problem : (0 < b ∧ b < a) ∨ (b > 1) :=
by
  sorry

end proof_problem_l826_826668


namespace inequality_a_b_c_l826_826278

def f (x : ℝ) : ℝ := (x^2) / 2 + Real.cos x
noncomputable def a : ℝ := f (2 ^ 0.2)
noncomputable def b : ℝ := f (0.2 ^ 0.2)
noncomputable def c : ℝ := f (Real.log 2 / Real.log 0.2)

theorem inequality_a_b_c : a > b ∧ b > c :=
by
  sorry

end inequality_a_b_c_l826_826278


namespace correctly_calculated_value_l826_826110

theorem correctly_calculated_value :
  ∀ (x : ℕ), (x * 15 = 45) → ((x * 5) * 10 = 150) := 
by
  intro x
  intro h
  sorry

end correctly_calculated_value_l826_826110


namespace find_cos_C_find_cos_2C_plus_pi_over_3_l826_826684

-- Conditions given in the problem.
def conditions (a b c : ℝ) (B C : ℝ) :=
  b = 3 ∧ cos B = 1 / 3 ∧ a * c = 6

-- First question
theorem find_cos_C (a b c : ℝ) (B C : ℝ) (h : conditions a b c B C) :
  cos C = 7 / 9 :=
by
  sorry

-- Second question
theorem find_cos_2C_plus_pi_over_3 (a b c : ℝ) (B C : ℝ) (h : conditions a b c B C) (h_cos_C : cos C = 7 / 9) :
  cos (2 * C + Real.pi / 3) = (17 - 56 * Real.sqrt 6) / 162 :=
by
  sorry

end find_cos_C_find_cos_2C_plus_pi_over_3_l826_826684


namespace xiaohong_money_l826_826840

def cost_kg_pears (x : ℝ) := x

def cost_kg_apples (x : ℝ) := x + 1.1

theorem xiaohong_money (x : ℝ) (hx : 6 * x - 3 = 5 * (x + 1.1) - 4) : 6 * x - 3 = 24 :=
by sorry

end xiaohong_money_l826_826840


namespace minimum_value_expression_l826_826957

-- Define the function
def f (x y : ℝ) : ℝ := 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x - 10 * y

-- State the proof problem
theorem minimum_value_expression : 
  ∃ (x y : ℝ), ∀ (u v : ℝ), f u v ≥ f x y ∧ f x y = -7208 := 
begin 
  let x := 77,
  let y := -120,
  use [x, y],
  split,
  { sorry }, -- Proof that for all u, v in ℝ, f u v ≥ f x y
  { sorry }  -- Proof that f x y = -7208
end

end minimum_value_expression_l826_826957


namespace points_on_circle_l826_826600

theorem points_on_circle (t : ℝ) : (∃ (x y : ℝ), x = Real.cos (2 * t) ∧ y = Real.sin (2 * t) ∧ (x^2 + y^2 = 1)) := by
  sorry

end points_on_circle_l826_826600


namespace deal_or_no_deal_l826_826337

def box_values : List ℚ := [0.01, 1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]

def high_value_boxes : List ℚ := [200000, 300000, 400000, 500000, 750000, 1000000]

theorem deal_or_no_deal (total_boxes high_value_count remaining_boxes : ℕ) (half_chance: ℚ) : 
  total_boxes = 26 →
  high_value_count = 6 →
  remaining_boxes = 12 →
  half_chance = 1/2 →
  total_boxes - remaining_boxes = 14 := 
by
  intros h_total h_high h_remaining h_chance
  calc
    total_boxes - remaining_boxes = 26 - 12 := by rw [h_total, h_remaining]
    ... = 14 := by norm_num

end deal_or_no_deal_l826_826337


namespace four_digit_numbers_count_l826_826562

noncomputable def count_four_digit_numbers_with_cond (lower upper : ℕ) : ℕ :=
  let candidates := (list.range' 1000 (9999 - 1000 + 1)).filter (λ n, (digits n).nodup) in
  candidates.count (λ n, abs ((n / 1000) - (n % 10)) = 2)

theorem four_digit_numbers_count :
  @count_four_digit_numbers_with_cond 1000 9999 = 840 :=
sorry

end four_digit_numbers_count_l826_826562


namespace area_of_trapezoid_l826_826620

theorem area_of_trapezoid (A B C D H : PPoint)
  (AD BC : Line)
  (BC_eq_5 AC_eq_5 AD_eq_6 : ℝ) 
  (angle_ACB_eq_2_angle_ADB : ℝ) :
  area (Trapezoid A B C D AD BC AD_eq_6 BC_eq_5 AC_eq_5 angle_ACB_eq_2_angle_ADB) = 22 :=
sorry

end area_of_trapezoid_l826_826620


namespace petya_numbers_l826_826395

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826395


namespace most_appropriate_chart_for_temperature_changes_l826_826080

theorem most_appropriate_chart_for_temperature_changes (reflect_changes_temperature: Prop) :
  reflect_changes_temperature ↔ (∃ (chart : Type) (line_chart : chart), line_chart = "line chart") :=
by
  sorry

end most_appropriate_chart_for_temperature_changes_l826_826080


namespace ellipse_properties_l826_826991

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_properties {
  a b : ℝ}
  (ha : a > b) (hb : b > 0) (hmajor : 2 * a = 2 * Real.sqrt 2) :
  (∀ (x0 y0 : ℝ), 
    (let P : ℝ × ℝ := (x0, y0) in
    let A2 : ℝ × ℝ := (Real.sqrt 2, 0) in
    let O : ℝ × ℝ := (0, 0) in
    let M : ℝ × ℝ := ((x0 + Real.sqrt 2) / 2, y0 / 2) in
    ((y0 / (x0 - Real.sqrt 2)) * (y0 / ((x0 + Real.sqrt 2) / 2)) = -1/2) →
    (x0^2 / 2 + y0^2 = 1) ∧ b = 1)) →
  ∀ (F1 : ℝ × ℝ)
    (l : ℝ → ℝ) (k : ℝ)
    (hF1k : k ≠ 0)
    (hline : ∀ (x : ℝ), l x = k * (x + 1))
    (N : ℝ × ℝ)
    (hN : ∀ Q : ℝ × ℝ, Q = (-(2 * k^2) / (2 * k^2 + 1), k / (2 * k^2 + 1))
           → N = (-(k^2) / (2 * k^2 + 1), 0) )
    (hN_range : -1/4 < -(k^2) / (2 * k^2 + 1) ∧ -(k^2) / (2 * k^2 + 1) < 0),
  (ellipse_equation (Real.sqrt 2) 1 (sqrt_pos.2) one_pos) ∧
  (∀ k : ℝ, 1 < 2 * k^2 + 1 → 2 * k^2 + 1 < 2 → 
    let AB_length := Real.sqrt 2 * (1 + 1 / (2 * k^2 + 1)) in
    (3 * Real.sqrt 2 / 2 < AB_length) ∧ (AB_length < 2 * Real.sqrt 2)) :=
by
  sorry

end ellipse_properties_l826_826991


namespace truck_distance_l826_826158

theorem truck_distance (V_t : ℝ) (D : ℝ) (h1 : D = V_t * 8) (h2 : D = (V_t + 18) * 5) : D = 240 :=
by
  sorry

end truck_distance_l826_826158


namespace count_final_numbers_l826_826407

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826407


namespace johns_savings_percentage_l826_826719

-- Define the conditions
variables (E : ℝ) (interest_rate : ℝ)
def rent : ℝ := 0.40 * E
def dishwasher : ℝ := 0.70 * rent
def groceries : ℝ := 1.15 * rent
def total_spent : ℝ := rent + dishwasher + groceries
def amount_saved : ℝ := E - total_spent
def amount_after_one_year : ℝ := amount_saved * (1 + interest_rate)

-- Prove that the amount after interest accrual is 90.3% of last month's earnings
theorem johns_savings_percentage (E : ℝ) (interest_rate : ℝ) (h1 : interest_rate = 0.05) :
  let rent := 0.40 * E,
      dishwasher := 0.70 * rent,
      groceries := 1.15 * rent,
      total_spent := rent + dishwasher + groceries,
      amount_saved := E - total_spent,
      amount_after_one_year := amount_saved * (1 + interest_rate) in
  (amount_after_one_year / E) * 100 = 90.3 :=
by
  sorry

end johns_savings_percentage_l826_826719


namespace average_of_first_6_numbers_l826_826444

-- Definitions extracted from conditions
def average_of_11_numbers := 60
def average_of_last_6_numbers := 65
def sixth_number := 258
def total_sum := 11 * average_of_11_numbers
def sum_of_last_6_numbers := 6 * average_of_last_6_numbers

-- Lean 4 statement for the proof problem
theorem average_of_first_6_numbers :
  (∃ A, 6 * A = (total_sum - (sum_of_last_6_numbers - sixth_number))) →
  (∃ A, 6 * A = 528) :=
by
  intro h
  exact h

end average_of_first_6_numbers_l826_826444


namespace power_of_complex_expression_l826_826185

theorem power_of_complex_expression :
  (Real.cos (215 * Real.pi / 180) + Complex.i * Real.sin (215 * Real.pi / 180))^72 = 1 := 
by
  sorry

end power_of_complex_expression_l826_826185


namespace total_students_prefer_goldfish_l826_826912

theorem total_students_prefer_goldfish :
  let students_per_class := 30
  let Miss_Johnson_fraction := 1 / 6
  let Mr_Feldstein_fraction := 2 / 3
  let Ms_Henderson_fraction := 1 / 5
  (Miss_Johnson_fraction * students_per_class) + 
  (Mr_Feldstein_fraction * students_per_class) +
  (Ms_Henderson_fraction * students_per_class) = 31 := 
by
  skip_proof

end total_students_prefer_goldfish_l826_826912


namespace all_real_roots_in_interval_l826_826618

noncomputable def q : ℕ → ℝ := sorry -- The sequence of positive numbers (q_n)

/-- Definition of the sequence of polynomials. -/
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := λ x, (1 + q n) * x * f n x - q n * f (n - 1) x

/-- Theorem stating that all roots of f_n(x) are within the interval [-1, 1]. -/
theorem all_real_roots_in_interval {n : ℕ} : ∀ x : ℝ, f n x = 0 → -1 ≤ x ∧ x ≤ 1 :=
sorry

end all_real_roots_in_interval_l826_826618


namespace raghu_investment_is_2200_l826_826488

noncomputable def RaghuInvestment : ℝ := 
  let R := 2200
  let T := 0.9 * R
  let V := 1.1 * T
  if R + T + V = 6358 then R else 0

theorem raghu_investment_is_2200 :
  RaghuInvestment = 2200 := by
  sorry

end raghu_investment_is_2200_l826_826488


namespace sum_of_first_n_integers_l826_826025

theorem sum_of_first_n_integers (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i) = n * (n + 1) / 2 :=
sorry

end sum_of_first_n_integers_l826_826025


namespace union_cardinality_l826_826371

theorem union_cardinality :
  let A := {4, 5, 7, 9}
  let B := {3, 4, 7, 8, 9}
  (A ∪ B).card = 6 := 
by
  let A := {4, 5, 7, 9}
  let B := {3, 4, 7, 8, 9}
  sorry

end union_cardinality_l826_826371


namespace alternating_intersections_l826_826323

theorem alternating_intersections (n : ℕ)
  (roads : Fin n → ℝ → ℝ) -- Roads are functions from reals to reals
  (h_straight : ∀ (i : Fin n), ∃ (a b : ℝ), ∀ x, roads i x = a * x + b) 
  (h_intersect : ∀ (i j : Fin n), i ≠ j → ∃ x, roads i x = roads j x)
  (h_two_roads : ∀ (x y : ℝ), ∃! (i j : Fin n), i ≠ j ∧ roads i x = roads j y) :
  ∃ (design : ∀ (i : Fin n), ℝ → Prop), 
  -- ensuring alternation, road 'i' alternates crossings with other roads 
  (∀ (i : Fin n) (x y : ℝ), 
    roads i x = roads i y → (design i x ↔ ¬design i y)) := sorry

end alternating_intersections_l826_826323


namespace gather_candies_into_one_plate_l826_826477

theorem gather_candies_into_one_plate (n : ℕ) (candies : ℕ) (plates : Fin n → ℕ)
  (hn : 4 ≤ n) (hc : 4 ≤ candies) (hc_valid : ∑ i, plates i = candies)
  (h_op : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → plates k = plates k + 2 → plates i = plates i - 1 → plates j = plates j - 1 → True) :
  ∃ k : Fin n, ∀ i : Fin n, i ≠ k → plates i = 0 := sorry

end gather_candies_into_one_plate_l826_826477


namespace square_area_on_parabola_l826_826542

theorem square_area_on_parabola (s : ℝ) (h : 0 < s) (hG : (3 + s)^2 - 6 * (3 + s) + 5 = -2 * s) : 
  (2 * s) * (2 * s) = 24 - 8 * Real.sqrt 5 := 
by 
  sorry

end square_area_on_parabola_l826_826542


namespace determine_m_with_opposite_roots_l826_826199

theorem determine_m_with_opposite_roots (c d k : ℝ) (h : c + d ≠ 0):
  (∃ m : ℝ, ∀ x : ℝ, (x^2 - d * x) / (c * x - k) = (m - 2) / (m + 2) ∧ 
            (x = -y ∧ y = -x)) ↔ m = 2 * (c - d) / (c + d) :=
sorry

end determine_m_with_opposite_roots_l826_826199


namespace extra_men_needed_l826_826911

theorem extra_men_needed (
  total_length : ℝ,
  total_time : ℝ,
  initial_men : ℝ,
  completed_length : ℝ,
  completed_time : ℝ,
  remaining_time : ℝ) :
  total_length = 10 ∧ 
  total_time = 300 ∧ 
  initial_men = 30 ∧ 
  completed_length = 2 ∧ 
  completed_time = 100 ∧
  remaining_time = 200 →
  let work_per_day := total_length / total_time in
  let work_per_man := work_per_day / initial_men in
  let remaining_work := total_length - completed_length in
  let required_work_rate := remaining_work / remaining_time in
  let required_men := required_work_rate / work_per_man in
  let extra_men := required_men - initial_men in
  extra_men = 6 :=
begin
  sorry
end

end extra_men_needed_l826_826911


namespace fermat_has_large_prime_factor_l826_826367

open Nat

-- Definition of Fermat number
def fermat (n : ℕ) : ℕ := 2 ^ 2 ^ n + 1

-- Theorem statement that for n ≥ 3, Fermat number has a prime factor greater than 2^(n+2)*(n+1)
theorem fermat_has_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ fermat n ∧ p > 2 ^ (n + 2) * (n + 1) :=
sorry

end fermat_has_large_prime_factor_l826_826367


namespace polynomial_difference_of_squares_l826_826555

theorem polynomial_difference_of_squares:
  (∀ a b : ℝ, ¬ ∃ x1 x2 : ℝ, a^2 + (-b)^2 = (x1 - x2) * (x1 + x2)) ∧
  (∀ m n : ℝ, ¬ ∃ x1 x2 : ℝ, 5 * m^2 - 20 * m * n = (x1 - x2) * (x1 + x2)) ∧
  (∀ x y : ℝ, ¬ ∃ x1 x2 : ℝ, -x^2 - y^2 = (x1 - x2) * (x1 + x2)) →
  ∃ x1 x2 : ℝ, -x^2 + 9 = (x1 - x2) * (x1 + x2) :=
by 
  sorry

end polynomial_difference_of_squares_l826_826555


namespace harvard_acceptance_rate_l826_826336

theorem harvard_acceptance_rate
  (total_applicants : ℕ)
  (attending_students : ℕ)
  (acceptance_rate : ℕ → ℕ)
  (accepted_students : ℕ)
  (H1 : total_applicants = 20000)
  (H2 : attending_students = 900)
  (H3 : ∀ n, acceptance_rate n = n / 0.90)
  (H4 : accepted_students = acceptance_rate attending_students)
  (percentage_accepted : ℕ → ℕ)
  (H5 : ∀ m, percentage_accepted m = (m * 100 / total_applicants)) :
  percentage_accepted accepted_students = 5 :=
by
  sorry

end harvard_acceptance_rate_l826_826336


namespace paving_stone_width_l826_826530

theorem paving_stone_width :
  ∀ (L₁ L₂ : ℝ) (n : ℕ) (length width : ℝ), 
    L₁ = 30 → L₂ = 16 → length = 2 → n = 240 →
    (L₁ * L₂ = n * (length * width)) → width = 1 :=
by
  sorry

end paving_stone_width_l826_826530


namespace correct_propositions_l826_826929

-- Define the conditions as propositions in Lean
def p1 (z : ℂ) : Prop := (1 / z ∈ ℝ) → (z ∈ ℝ)
def p2 (z : ℂ) : Prop := (z^2 ∈ ℝ) → (z ∈ ℝ)
def p3 (z1 z2 : ℂ) : Prop := (z1 * z2 ∈ ℝ) → (z1 = conj z2)
def p4 (z : ℂ) : Prop := (z ∈ ℝ) → (conj z ∈ ℝ)

-- Define the theorem to identify the true propositions
theorem correct_propositions :
  (∀ z : ℂ, p1 z) ∧ (∀ z : ℂ, p4 z) ∧ ¬ (∀ z : ℂ, p2 z) ∧ ¬ (∀ z1 z2 : ℂ, p3 z1 z2)
:= by
  sorry

end correct_propositions_l826_826929


namespace three_digit_integers_congruent_mod_5_l826_826295

theorem three_digit_integers_congruent_mod_5 :
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 5 = 3}.to_finset.card = 180 :=
by
  sorry

end three_digit_integers_congruent_mod_5_l826_826295


namespace collinearity_of_O1_O2_O3_l826_826566

noncomputable def circle (center : Point) (radius : ℝ) : Set Point := sorry

axiom intersect_circles (γ1 γ2 : Set Point) (p q : Point) : p ∈ (γ1 ∩ γ2) ∧ q ∈ (γ1 ∩ γ2)

structure Configuration :=
(P Q P1 Q1 R R1 : Point)
(Γ Γ1 Γ2 : Set Point)
(hΓ1_tangent : (circle P radius₂) ⊆ Γ1)
(hΓ2_tangent : (circle Q radius₃) ⊆ Γ2)
(hΓ_internal_tangent_Γ1 : P ∈ circle center₁ radius₁ ∧ P ∈ circle center radius₂)
(hΓ_internal_tangent_Γ2 : Q ∈ circle center₂ radius₁ ∧ Q ∈ circle center radius₃)
(hP1_on_Γ1 : P1 ∈ Γ1)
(hQ1_on_Γ2 : Q1 ∈ Γ2)
(hP1Q1_tangent : ∀ x, x ∈ P1Q1 → x ∈ tangent_line site_point)
(hΓ1_intersect_Γ2 : intersect_circles Γ1 Γ2 R R1)

def O1 (cfg : Configuration) : Point := sorry
def O2 (cfg : Configuration) : Point := sorry
def O3 (cfg : Configuration) : Point := sorry

theorem collinearity_of_O1_O2_O3 (cfg : Configuration) : collinear (O1 cfg) (O2 cfg) (O3 cfg) :=
sorry

end collinearity_of_O1_O2_O3_l826_826566


namespace cyclic_quadrilateral_area_l826_826232

theorem cyclic_quadrilateral_area (A B C D : Type)
  [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (cyclic : is_cyclic_quadrilateral A B C D)
  (hAB : AB = 2) (hBC : BC = 6) (hCD : CD = 4) (hDA : DA = 4) :
  area_of_quadrilateral A B C D = 8 * real.sqrt 3 :=
sorry

end cyclic_quadrilateral_area_l826_826232


namespace eval_expression_l826_826814

theorem eval_expression : 5 + 4 - 3 + 2 - 1 = 7 :=
by
  -- Mathematically, this statement holds by basic arithmetic operations.
  sorry

end eval_expression_l826_826814


namespace greatest_distance_between_vertices_l826_826885

-- Definitions for conditions
def inner_square_perimeter : ℝ := 24
def outer_square_perimeter : ℝ := 32
def inner_square_side : ℝ := inner_square_perimeter / 4
def outer_square_side : ℝ := outer_square_perimeter / 4

-- Problem statement
theorem greatest_distance_between_vertices : 
  let max_distance := real.sqrt ((outer_square_side/2 + (outer_square_side - inner_square_side) / 2) ^ 2 + (outer_square_side/2 + (outer_square_side - inner_square_side) / 2) ^ 2)
  in max_distance = 5 * real.sqrt 2 := sorry

end greatest_distance_between_vertices_l826_826885


namespace square_area_of_complex_points_l826_826687

theorem square_area_of_complex_points (z : ℂ) (h1 : z ≠ 0) (h2 : z ≠ 1) 
  (h3 : z^4 - z = complex.I * (z^2 - z)) : 
  ∃ n : ℝ, n = 2 ∧ (∀ x y : ℂ, x = z → y = z^2 → dist x y ≠ 0 ∧ dist x y = dist y x ∧ (y ≠ z^4)) :=
by
  sorry

end square_area_of_complex_points_l826_826687


namespace sequence_sum_l826_826581

theorem sequence_sum (r z w : ℝ) (h1 : 4 * r = 1) (h2 : 256 * r = z) (h3 : z * r = w) : z + w = 80 :=
by
  -- Proceed with your proof here.
  -- sorry for skipping the proof part.
  sorry

end sequence_sum_l826_826581


namespace distance_between_houses_l826_826714

-- Definitions
def speed : ℝ := 2          -- Amanda's speed in miles per hour
def time : ℝ := 3           -- Time taken by Amanda in hours

-- The theorem to prove distance is 6 miles
theorem distance_between_houses : speed * time = 6 := by
  sorry

end distance_between_houses_l826_826714


namespace question_I_trajectory_question_II_exists_point_l826_826875

noncomputable def trajectory_of_P (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

theorem question_I_trajectory :
  ∀ (x y : ℝ), 
  (dist (x, y) (1, 0)) / (abs (x - 2)) = (real.sqrt 2) / 2 →
  trajectory_of_P x y :=
sorry

theorem question_II_exists_point (x1 y1 x2 y2 t : ℝ) :
  ∀ (k : ℝ), k ≠ 0 → 
  (y1 = k * (x1 - 1)) →
  (y2 = k * (x2 - 1)) →
  (x1 + x2) = (4 * k^2) / (1 + 2 * k^2) →
  (x1 * x2) = (2 * k^2 - 2) / (1 + 2 * k^2) →
  (∃ (E : ℝ × ℝ), E = (2, 0)) :=
sorry

end question_I_trajectory_question_II_exists_point_l826_826875


namespace range_of_positive_integers_in_list_H_l826_826002

noncomputable def list_H_lower_bound : Int := -15
noncomputable def list_H_length : Nat := 30

theorem range_of_positive_integers_in_list_H :
  ∃(r : Nat), list_H_lower_bound + list_H_length - 1 = 14 ∧ r = 14 - 1 := 
by
  let upper_bound := list_H_lower_bound + Int.ofNat list_H_length - 1
  use (upper_bound - 1).toNat
  sorry

end range_of_positive_integers_in_list_H_l826_826002


namespace girls_with_short_hair_count_l826_826064

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l826_826064


namespace acute_angle_at_9_35_l826_826492

def minute_hand_degree (minute : ℕ) : ℝ :=
  (minute / 60) * 360

def hour_hand_degree (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour / 12) * 360 + (minute / 60) * 30

def acute_angle (h_deg : ℝ) (m_deg : ℝ) : ℝ :=
  let angle_diff := abs (h_deg - m_deg)
  if angle_diff > 180 then 360 - angle_diff else angle_diff

theorem acute_angle_at_9_35 : acute_angle (hour_hand_degree 9 35) (minute_hand_degree 35) = 77.5 := by
  sorry

end acute_angle_at_9_35_l826_826492


namespace students_with_uncool_parents_l826_826457

theorem students_with_uncool_parents (class_size : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ) : 
  class_size = 40 → cool_dads = 18 → cool_moms = 20 → both_cool_parents = 10 → 
  (class_size - (cool_dads - both_cool_parents + cool_moms - both_cool_parents + both_cool_parents) = 12) :=
by
  sorry

end students_with_uncool_parents_l826_826457


namespace paige_finished_problems_l826_826771

-- Define the conditions
def initial_problems : ℕ := 110
def problems_per_page : ℕ := 9
def remaining_pages : ℕ := 7

-- Define the statement we want to prove
theorem paige_finished_problems :
  initial_problems - (remaining_pages * problems_per_page) = 47 :=
by sorry

end paige_finished_problems_l826_826771


namespace stock_yield_percentage_l826_826857

theorem stock_yield_percentage
    (annual_dividend : ℝ)
    (market_value : ℝ)
    (yield_percentage : ℝ)
    (h1 : market_value = 162.5)
    (h2 : yield_percentage = (annual_dividend / market_value) * 100)
    (h3 : annual_dividend = 0.13 * 100) :
    yield_percentage = 8 :=
by 
    rw [h1, h3] at h2
    simp at h2
    exact h2

end stock_yield_percentage_l826_826857


namespace tan_theta_value_l826_826628

noncomputable def tan_theta (θ : ℝ) : ℝ :=
  if (0 < θ) ∧ (θ < 2 * Real.pi) ∧ (Real.cos (θ / 2) = 1 / 3) then
    (2 * (2 * Real.sqrt 2) / (1 - (2 * Real.sqrt 2) ^ 2))
  else
    0 -- added default value for well-definedness

theorem tan_theta_value (θ : ℝ) (h₀: 0 < θ) (h₁ : θ < 2 * Real.pi) (h₂ : Real.cos (θ / 2) = 1 / 3) : 
  tan_theta θ = -4 * Real.sqrt 2 / 7 :=
by
  sorry

end tan_theta_value_l826_826628


namespace intersection_M_N_l826_826286

def M := { x : ℝ | -1 < x ∧ x < 2 }
def N := { x : ℝ | x ≤ 1 }
def expectedIntersection := { x : ℝ | -1 < x ∧ x ≤ 1 }

theorem intersection_M_N :
  M ∩ N = expectedIntersection :=
by
  sorry

end intersection_M_N_l826_826286


namespace line_passes_through_parabola_vertex_l826_826221

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_parabola_vertex_l826_826221


namespace area_of_quadrilateral_DEFG_l826_826433

-- Define the main geometric objects and their properties.
def is_rectangle (A B C D : Point) (AB BC CD DA : ℝ) : Prop :=
  AB * BC = 48 ∧ AB = CD ∧ BC = DA

def point_ratio (A E D : Point) (AE ED : ℝ) : Prop :=
  AE / ED = 2 / 1

def midpoint (C G D : Point) : Prop :=
  dist C G = dist G D

def midpoint (B F C : Point) : Prop :=
  dist B F = dist F C

def quadrilateral_area (D E F G : Point) : ℝ := sorry  -- Placeholder for quadrilateral area calculation

-- Main theorem
theorem area_of_quadrilateral_DEFG
  (A B C D E F G : Point)
  (AB BC CD DA AE ED : ℝ)
  (h_rect : is_rectangle A B C D AB BC CD DA)
  (h_point_ratio : point_ratio A E D AE ED)
  (h_midpoint_CD : midpoint C G D)
  (h_midpoint_BC : midpoint B F C) :
  quadrilateral_area D E F G = 8 := sorry

end area_of_quadrilateral_DEFG_l826_826433


namespace solution_to_problem_l826_826119

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∃ (A A1 B C H O : Point) (r : ℝ), 
  is_altitude A A1 B C ∧ 
  is_orthocenter H A B C ∧ 
  is_circumcenter O A B C ∧ 
  distance A H = 3 ∧ 
  distance A1 H = 2 ∧ 
  radii_equal O A B C 4 ∧ 
  distance O H = 2

theorem solution_to_problem : problem_statement :=
by sorry

end solution_to_problem_l826_826119


namespace count_final_numbers_l826_826412

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826412


namespace koala_fiber_absorption_l826_826348

theorem koala_fiber_absorption (x : ℝ) (hx : 0.30 * x = 12) : x = 40 :=
by
  sorry

end koala_fiber_absorption_l826_826348


namespace sum_g_inverse_up_to_5000_correct_l826_826739

noncomputable def g (n : ℕ) : ℤ :=
  if h : (n : ℝ) < 0 then -⌊-(n : ℝ)^(1/3) + 0.5⌉ else ⌊(n : ℝ)^(1/3) + 0.5⌉

noncomputable def sum_g_inverse_up_to_5000 : ℝ :=
  (∑ k in Finset.range 5000, (1 / (g (k + 1) : ℝ)))

theorem sum_g_inverse_up_to_5000_correct :
  sum_g_inverse_up_to_5000 = 765.5 :=
by
  sorry

end sum_g_inverse_up_to_5000_correct_l826_826739


namespace number_of_correct_propositions_l826_826993

variable (m n : Type) 
variable (α β : Type)

-- Definitions of basic relations between lines and planes
-- Note: These would typically be expanded upon in a full formalization 
-- We use "Prop" simply so they type check in Lean; you would define them rigorously.
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Propositions given in the problem
def prop1 : Prop := (perpendicular α β) ∧ (parallel m α) → (perpendicular m β)
def prop2 : Prop := (perpendicular m α) ∧ (perpendicular n β) ∧ (perpendicular m n) → (perpendicular α β)
def prop3 : Prop := (perpendicular m β) ∧ (parallel m α) → (perpendicular α β)
def prop4 : Prop := (parallel m α) ∧ (parallel n β) ∧ (parallel m n) → (parallel α β)

-- Lean statement for the problem to prove the number of correct propositions
theorem number_of_correct_propositions : 
  (¬prop1) ∧ prop2 ∧ prop3 ∧ (¬prop4) → (number_of_correct_propositions = 2) :=
by
  sorry

end number_of_correct_propositions_l826_826993


namespace number_of_2007_digit_numbers_with_odd_9s_l826_826809

theorem number_of_2007_digit_numbers_with_odd_9s:
  let n := 2006,
      A := finset.sum (finset.Ico 1 (n + 1)) (λ k, if odd k then (nat.choose n k) * 9^(n-k) else 0) in
  A = 10^n - 8^n :=
by
  sorry

end number_of_2007_digit_numbers_with_odd_9s_l826_826809


namespace yearly_savings_l826_826111

-- Define the weekly and monthly rent prices
def weekly_rent (weeks : Nat) : Nat := weeks * 10
def monthly_rent (months : Nat) : Nat := months * 35

-- Define the number of weeks and months in a year
def weeks_per_year := 52
def months_per_year := 12

-- Theorem stating the amount saved per year
theorem yearly_savings : (weekly_rent weeks_per_year) - (monthly_rent months_per_year) = 100 :=
by
  -- Calculate the total cost for renting by week and by month
  let total_weekly_cost := weekly_rent weeks_per_year
  let total_monthly_cost := monthly_rent months_per_year

  -- Define the expected savings
  let expected_savings := total_weekly_cost - total_monthly_cost
  
  -- Assert the savings are $100
  have : total_weekly_cost = 520 := by sorry
  have : total_monthly_cost = 420 := by sorry

  -- Conclude that the savings are $100
  show expected_savings = 100 from by sorry

end yearly_savings_l826_826111


namespace remaining_numbers_after_erasure_l826_826389

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826389


namespace cost_per_kg_l826_826174

noncomputable def apple_cost_l : ℝ := 3.62 / 10
noncomputable def apple_cost_m : ℝ := 0.27

theorem cost_per_kg (l m : ℝ) (h1 : 10 * l = 3.62) (h2 : 30 * l + 3 * m = 11.67) (h3 : 30 * l + 6 * m = 12.48) : 
  m = apple_cost_m :=
by {
  have l_def : l = apple_cost_l, from (by linarith [h1]),
  rw [l_def] at h2 h3,
  have m_def : 3 * m = 11.67 - 30 * apple_cost_l, from (by linarith [h2]),
  linarith [m_def]
}

end cost_per_kg_l826_826174


namespace volume_ratio_l826_826473

-- Conditions
variables (A B C A1 B1 C1 : Type) -- Points
variables (cone prism : Type)      -- The geometrical shapes
variables [fintype cone] [fintype prism]

-- Ratios and geometric relationships
variable (AB : ℝ) -- Initial length AB
variable (ratio : ℝ) -- Ratio AB_1 : AB
variable (AB1 : ℝ := ratio * AB) -- AB_1 calculated from ratio
variable (a : ℝ := 1) -- Normalized length AB = a
variable (r : ℝ := (5 * a * real.sqrt 2) / 2) -- Radius of the cone's base calculated from the problem
variable (h : ℝ := (5 * a * real.sqrt 2) / 2) -- Height of the cone calculated from the problem
variable (V1 : ℝ := (1 / 3) * real.pi * r^2 * h) -- Volume of the cone calculated using basic formulas
variable (V : ℝ := (real.sqrt 3 / 4) * a^2 * 2 * a * real.sqrt 6) -- Volume of the prism calculated using basic formulas

-- Theorem: Ratio of volumes
theorem volume_ratio (h_ratio : ratio = 5) : V1 / V = 125 * real.pi / 18 :=
by 
  -- The proof will typically follow from the provided evidence and the use of real arithmetic.
  sorry

end volume_ratio_l826_826473


namespace ryan_chinese_learning_hours_l826_826949

theorem ryan_chinese_learning_hours
    (hours_per_day : ℕ) 
    (days : ℕ) 
    (h1 : hours_per_day = 4) 
    (h2 : days = 6) : 
    hours_per_day * days = 24 := 
by 
    sorry

end ryan_chinese_learning_hours_l826_826949


namespace problem1_problem2_problem3_problem4_l826_826184

-- Problem 1
theorem problem1 : (-3 : ℝ) ^ 2 + (1 / 2) ^ (-1 : ℝ) + (Real.pi - 3) ^ 0 = 12 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (8 * x ^ 4 + 4 * x ^ 3 - x ^ 2) / (-2 * x) ^ 2 = 2 * x ^ 2 + x - 1 / 4 :=
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (2 * x + 1) ^ 2 - (4 * x + 1) * (x + 1) = -x :=
by
  sorry

-- Problem 4
theorem problem4 (x y : ℝ) : (x + 2 * y - 3) * (x - 2 * y + 3) = x ^ 2 - 4 * y ^ 2 + 12 * y - 9 :=
by
  sorry

end problem1_problem2_problem3_problem4_l826_826184


namespace f_increasing_on_interval_a_less_than_minus_one_l826_826277

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x - 1) / (x + 1)

-- Problem 1: Prove that if a = 2, then f(x) is increasing on (-∞, -1)
theorem f_increasing_on_interval (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x2 < -1) :
  f 2 x1 < f 2 x2 := 
by sorry

-- Problem 2: Prove that if f(x) is decreasing on (-∞, -1), then a < -1
theorem a_less_than_minus_one (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 < -1 → f a x1 > f a x2) → a < -1 := 
by sorry

end f_increasing_on_interval_a_less_than_minus_one_l826_826277


namespace problem1_problem2_l826_826965

variable (n : ℕ)

theorem problem1 (hn : n ≥ 3) (hn_pos : 0 < n) (h : (fact (2*n))^3 = 2 * (fact (n + 1))^4) : n = 5 := sorry

theorem problem2 (hn : n ≥ 3) (hn_pos : 0 < n) (h : choose (n + 2) (n - 2) + choose (n + 2) (n - 3) = (1 / 10) * (fact (n + 3))^3) : n = 4 := sorry

end problem1_problem2_l826_826965


namespace range_of_t_for_two_extreme_points_l826_826644

noncomputable def f (x t : ℝ) : ℝ := (Real.exp x) / x + t * (3 * Real.log x + 2 / x - x)

theorem range_of_t_for_two_extreme_points :
  ∀ t : ℝ, (∃ p1 p2 : ℝ, p1 ≠ p2 ∧ (∂ (f p1 t) / ∂ x = 0) ∧ (∂ (f p2 t) / ∂ x = 0)) ↔ t ∈ Set.Ioo (-∞) (-e) ∪ Set.Ioo (-e) (-1 / 2) := 
sorry

end range_of_t_for_two_extreme_points_l826_826644


namespace ratio_of_shaded_area_l826_826917

-- Define the problem in terms of ratios of areas
open Real

theorem ratio_of_shaded_area (ABCD E F G H : ℝ) (h : ∀ (E F G H ∈ ABCD), 
  (E = 1/3 * BA) ∧ (F = 2/3 * CB) ∧ (G = 2/3 * DC) ∧ (H = 1/3 * AD)) :
  ∃ (shaded_area ABCD_area : ℝ), shaded_area / ABCD_area = 5 / 9 := sorry

end ratio_of_shaded_area_l826_826917


namespace short_haired_girls_l826_826066

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l826_826066


namespace find_angle_on_positive_y_axis_l826_826554

noncomputable def angle_pos_y_axis (α : ℝ) : Prop :=
  ∃ k : ℤ, α = (π / 2) + 2 * k * π

theorem find_angle_on_positive_y_axis :
  let angles := [π / 4, π / 2, π, 3 * π / 2]
  ∃ α ∈ angles, angle_pos_y_axis α ∧ α = π / 2 :=
by
  sorry

end find_angle_on_positive_y_axis_l826_826554


namespace count_final_numbers_l826_826413

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826413


namespace log_exp_power_order_l826_826838

theorem log_exp_power_order :
  log 4 0.3 < 0.4^3 ∧ 0.4^3 < 3^0.4 :=
sorry

end log_exp_power_order_l826_826838


namespace second_discount_percentage_is_20_l826_826556

theorem second_discount_percentage_is_20 
    (normal_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (first_discount_percentage : ℝ)
    (h1 : normal_price = 149.99999999999997)
    (h2 : final_price = 108)
    (h3 : first_discount_percentage = 10)
    (h4 : first_discount = normal_price * (first_discount_percentage / 100)) :
    (((normal_price - first_discount) - final_price) / (normal_price - first_discount)) * 100 = 20 := by
  sorry

end second_discount_percentage_is_20_l826_826556


namespace initial_oranges_l826_826887

theorem initial_oranges (X : ℕ) : 
  (X - 9 + 38 = 60) → X = 31 :=
sorry

end initial_oranges_l826_826887


namespace probability_exists_bc_l826_826352

open Probability

noncomputable def f : ℕ → ℕ := sorry
noncomputable def G (f : ℕ → ℕ) : ℕ := sorry

theorem probability_exists_bc {n : ℕ} (h_n : n > 0) :
  let f := λ (x : Fin n), Fin n in
  let a := (uniform (Fin n)) in
  let event := {a : Fin n | ∃ b c : ℕ, b ≥ 1 ∧ c ≥ 1 ∧ f^[b] 1 = a ∧ f^[c] a = 1 } in
  Pr[event] = 1 / n :=
by
  sorry

end probability_exists_bc_l826_826352


namespace sum_of_center_coords_l826_826942

theorem sum_of_center_coords (x y : ℝ) :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2) →
  let c := 2, d := -6 in
  x + y = c + d →
  let center := (c, d) in
  c + d = -4 :=
by
  sorry

end sum_of_center_coords_l826_826942


namespace compound_interest_calculation_l826_826116

theorem compound_interest_calculation
  (CI : ℝ)
  (r : ℝ)
  (t : ℕ)
  (n : ℕ)
  (principal : ℝ)
  (final_amount : ℝ) :
  CI = 326.40 → r = 0.04 → t = 2 → n = 1 → principal = 4000 →
  final_amount = principal + CI →
  final_amount = 4326.40 :=
by
  intros hCI hr ht hn hp hf
  rw [hf, hp, hCI]
  exact rfl
  
-- Proof is omitted
sorry

end compound_interest_calculation_l826_826116


namespace prove_correct_options_l826_826839

open EuclideanGeometry

-- Definitions of conditions from the problem
variables {a b : EuclideanSpace ℝ ℕ} (h1 : 2 • a = -3 • b) -- Condition for question 1
variables {e1 e2 : EuclideanSpace ℝ ℕ} (h2 : linear_independent ℝ ![e1, e2]) -- Condition for question 2 (basis)
variables (a b1 : EuclideanSpace ℝ ℕ) (h3a : a = (-2, 3)) (h3b : b1 = (1, -2)) -- Condition for question 3
variables {A B C D E F : EuclideanSpace ℝ ℕ} (h4 : regular_hexagon A B C D E F) -- Condition for question 4

-- Theorem statement
theorem prove_correct_options : (2 • a = -3 • b → (∃ k: ℝ, a = k • b)) ∧
  ((linear_independent ℝ ![e1, e2]) → linear_independent ℝ![e1 + 2 • e2, e1 - 2 • e2]) ∧
  ((a = (-2, 3) ∧ b1 = (1, -2)) → norm (a + b1) ≠ 1) ∧
  (regular_hexagon A B C D E F → angle A B C = (2 * π / 3)) :=
by
  sorry

end prove_correct_options_l826_826839


namespace rationalize_denominator_l826_826784

theorem rationalize_denominator :
  let a := (5 : ℝ)^(1/3)
  let b := (3 : ℝ)^(1/3)
  let A := 25
  let B := 15
  let C := 9
  let D := 2
  a ≠ b →
  (A + B + C + D = 51) →
  (1 / (a - b) = (a^2 + a * b + b^2) / (D)) :=
begin
  intros h₁ h₂,
  sorry
end

end rationalize_denominator_l826_826784


namespace new_rectangle_area_l826_826034

theorem new_rectangle_area (x y : ℝ) (h : x ≤ y) :
    let d := Real.sqrt (x^2 + y^2) in
    ((d + y) * (d - y) = x^2) :=
by
  let d := Real.sqrt (x^2 + y^2)
  sorry

end new_rectangle_area_l826_826034


namespace functional_relationship_value_at_neg1_l826_826713

variable (x y y₁ y₂ k b : ℝ)

-- Conditions
def cond1 : Prop := y = y₁ + y₂
def cond2 : Prop := y₁ = k / x
def cond3 : Prop := y₂ = b * (x - 2)
def cond4 : Prop := (x = 1 → y = -1)
def cond5 : Prop := (x = 3 → y = 5)

-- Theorem to prove the functional relationship between y and x
theorem functional_relationship (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) :
  y = (3 / x) + 4 * (x - 2) := by
  sorry

-- Theorem to prove the value of y when x = -1
theorem value_at_neg1 (h : y = (3 / x) + 4 * (x - 2)) :
  (x = -1 → y = -15) := by
  sorry

end functional_relationship_value_at_neg1_l826_826713


namespace time_to_cover_escalator_l826_826172

-- Definitions of the rates and length
def escalator_speed : ℝ := 12
def person_speed : ℝ := 2
def escalator_length : ℝ := 210

-- Theorem statement that we need to prove
theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed) = 15) :=
by
  sorry

end time_to_cover_escalator_l826_826172


namespace tomato_plants_count_l826_826298

theorem tomato_plants_count :
  ∀ (sunflowers corn tomatoes total_rows plants_per_row : ℕ),
  sunflowers = 45 →
  corn = 81 →
  plants_per_row = 9 →
  total_rows = (sunflowers / plants_per_row) + (corn / plants_per_row) →
  tomatoes = total_rows * plants_per_row →
  tomatoes = 126 :=
by
  intros sunflowers corn tomatoes total_rows plants_per_row Hs Hc Hp Ht Hm
  rw [Hs, Hc, Hp] at *
  -- Additional calculation steps could go here to prove the theorem if needed
  sorry

end tomato_plants_count_l826_826298


namespace smallest_n_for_terminating_decimal_l826_826092

-- Theorem follows the tuple of (question, conditions, correct answer)
theorem smallest_n_for_terminating_decimal (n : ℕ) (h : ∃ k : ℕ, n + 75 = 2^k ∨ n + 75 = 5^k ∨ n + 75 = (2^k * 5^k)) :
  n = 50 :=
by
  sorry -- Proof is omitted

end smallest_n_for_terminating_decimal_l826_826092


namespace negative_integer_solution_l826_826813

theorem negative_integer_solution (N : ℤ) (hN : N^2 + N = -12) : N = -3 ∨ N = -4 :=
sorry

end negative_integer_solution_l826_826813


namespace real_part_divisibility_l826_826369

theorem real_part_divisibility (a b : ℤ) (p : ℕ) [hp : Fact (Nat.Prime p)] (hodd : p % 2 = 1) :
  p ∣ ((a + b * complex.I) ^ p - (a + b * complex.I)).re :=
sorry

end real_part_divisibility_l826_826369


namespace quadratic_discriminant_real_roots_k_range_for_positive_root_l826_826274

theorem quadratic_discriminant_real_roots (k : ℝ) :
  let Δ := (k + 3)^2 - 4 * (2 * k + 2) in Δ ≥ 0 :=
by
  let Δ := (k + 3)^2 - 4 * (2 * k + 2)
  have : Δ = (k - 1)^2, sorry
  rw this,
  exact pow_two_nonneg _

theorem k_range_for_positive_root (k : ℝ) (x : ℝ) :
  x^2 - (k+3)*x + 2*k + 2 = 0 ∧ 0 < x ∧ x < 1 → -1 < k ∧ k < 0 :=
by
  intro h
  sorry

end quadratic_discriminant_real_roots_k_range_for_positive_root_l826_826274


namespace numbers_left_on_board_l826_826423

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826423


namespace third_root_of_polynomial_l826_826464

theorem third_root_of_polynomial (a b c : ℚ) (h_poly : ∀ x : ℂ, x^3 + a * x^2 + b * x + c = 0)
  (h_root1 : (2 - sqrt 5) ^ 3 + a * (2 - sqrt 5) ^ 2 + b * (2 - sqrt 5) + c = 0)
  (h_root2 : 1 ^ 3 + a * 1 ^ 2 + b * 1 + c = 0) :
  ∃ x : ℂ, x ^ 3 + a * x ^ 2 + b * x + c = 0 ∧ x ≠ (2 - sqrt 5) ∧ x ≠ (2 + sqrt 5) :=
begin
  -- Let the third root be x.
  -- Utilize Vieta's formulas to equate the sum of the roots to -a.
  -- Since sum of roots = (2 - sqrt(5)) + (2 + sqrt(5)) + 1 = 5, and -a = 5, we have a = -5.
  -- Use these conditions to solve for the third root.
  sorry
end

end third_root_of_polynomial_l826_826464


namespace radius_of_garden_outer_boundary_l826_826138

-- Definitions based on the conditions from the problem statement
def fountain_diameter : ℝ := 12
def garden_width : ℝ := 10

-- Question translated to a proof statement
theorem radius_of_garden_outer_boundary :
  (fountain_diameter / 2 + garden_width) = 16 := 
by 
  sorry

end radius_of_garden_outer_boundary_l826_826138


namespace remarkable_number_sum_2001_l826_826753

/-- A natural number is "remarkable" if it is the smallest among natural numbers with the same sum of digits. -/
def remarkable (n : ℕ) : Prop :=
  ∀ m, sum_of_digits m = sum_of_digits n → m ≥ n

/-- We aim to prove that the sum of the digits of the two-thousand-and-first remarkable number is 2001. -/
theorem remarkable_number_sum_2001 : 
  ∃ n, remarkable n ∧ nth_2001st_remarkable (sum_of_digits n) = 2001 :=
sorry

end remarkable_number_sum_2001_l826_826753


namespace profit_eqn_65_to_75_maximize_profit_with_discount_l826_826866

-- Definitions for the conditions
def total_pieces (x y : ℕ) : Prop := x + y = 100

def total_cost (x y : ℕ) : Prop := 80 * x + 60 * y ≤ 7500

def min_pieces_A (x : ℕ) : Prop := x ≥ 65

def profit_without_discount (x : ℕ) : ℕ := 10 * x + 3000

def profit_with_discount (x a : ℕ) (h1 : 0 < a) (h2 : a < 20): ℕ := (10 - a) * x + 3000

-- Proof statement
theorem profit_eqn_65_to_75 (x: ℕ) (h1: total_pieces x (100 - x)) (h2: total_cost x (100 - x)) (h3: min_pieces_A x) :
  65 ≤ x ∧ x ≤ 75 → profit_without_discount x = 10 * x + 3000 :=
by
  sorry

theorem maximize_profit_with_discount (x a : ℕ) (h1 : total_pieces x (100 - x)) (h2 : total_cost x (100 - x)) (h3 : min_pieces_A x) (h4 : 0 < a) (h5 : a < 20) :
  if a < 10 then x = 75 ∧ profit_with_discount 75 a h4 h5 = (10 - a) * 75 + 3000
  else if a = 10 then 65 ≤ x ∧ x ≤ 75 ∧ profit_with_discount x a h4 h5 = 3000
  else x = 65 ∧ profit_with_discount 65 a h4 h5 = (10 - a) * 65 + 3000 :=
by
  sorry

end profit_eqn_65_to_75_maximize_profit_with_discount_l826_826866


namespace diophantine_equation_solvable_l826_826523

theorem diophantine_equation_solvable (a : ℕ) (ha : 0 < a) : 
  ∃ (x y : ℤ), x^2 - y^2 = a^3 :=
by
  let x := (a * (a + 1)) / 2
  let y := (a * (a - 1)) / 2
  have hx : x^2 = (a * (a + 1) / 2 : ℤ)^2 := sorry
  have hy : y^2 = (a * (a - 1) / 2 : ℤ)^2 := sorry
  use x
  use y
  sorry

end diophantine_equation_solvable_l826_826523


namespace sum_of_four_primes_l826_826047

def is_prime (n : ℕ) : Prop := nat.prime n

theorem sum_of_four_primes (A B : ℕ) 
  (h1 : is_prime A) (h2 : is_prime B) (h3 : is_prime (A - B)) (h4 : is_prime (A - 2*B)) :
  3 * A - 2 * B = 17 :=
by 
  sorry

end sum_of_four_primes_l826_826047


namespace Ella_food_each_day_l826_826698

variable {E : ℕ} -- Define E as the number of pounds of food Ella eats each day

def food_dog_eats (E : ℕ) : ℕ := 4 * E -- Definition of food the dog eats each day

def total_food_eaten_in_10_days (E : ℕ) : ℕ := 10 * E + 10 * (food_dog_eats E) -- Total food (Ella and dog) in 10 days

theorem Ella_food_each_day : total_food_eaten_in_10_days E = 1000 → E = 20 :=
by
  intros h -- Assume the given condition
  sorry -- Skip the actual proof

end Ella_food_each_day_l826_826698


namespace color_lines_general_position_l826_826312

-- Definitions
def in_general_position (lines : set (AffinePlane ℝ)) : Prop :=
  (∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ l1.is_parallel l2) ∧ (∀ l1 l2 l3 ∈ lines, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬ collinear {l1, l2, l3})

def finite_regions (lines : set (AffinePlane ℝ)) : set (set (AffinePlane ℝ)) :=
  {region | is_bounded region}

theorem color_lines_general_position (n : ℕ) (lines : set (AffinePlane ℝ)) :
  n ≥ sufficiently_large → in_general_position lines → ∃ blue_lines : set (AffinePlane ℝ), 
  finite_regions lines → (|blue_lines| ≥ ⌈√ n⌉) 
  ∧ ∀ (finite_region ∈ finite_regions lines), 
  ∃ l ∈ finite_region, l ∉ blue_lines :=
sorry

end color_lines_general_position_l826_826312


namespace string_cheese_per_package_l826_826722

-- Definitions based on conditions
def days_per_week := 5
def weeks := 4
def oldest_child_cheese_per_day := 2
def youngest_child_cheese_per_day := 1
def packages_needed := 2

-- Definition for the total string cheeses per day per child
def total_cheese_per_week (oldest_per_day : ℕ) (youngest_per_day : ℕ) : ℕ :=
  (oldest_per_day * days_per_week) + (youngest_per_day * days_per_week)

-- Definition for the total string cheeses required for 4 weeks
def total_cheese (cheese_per_week : ℕ) (number_of_weeks : ℕ) : ℕ :=
  cheese_per_week * number_of_weeks

-- Theorem statement to prove the number of string cheeses per package
theorem string_cheese_per_package
  (oldest_per_day : ℕ) (youngest_per_day : ℕ) (days : ℕ) (weeks : ℕ) (packages : ℕ)
  (total_per_week : ℕ := total_cheese_per_week oldest_per_day youngest_per_day)
  (total_for_weeks : ℕ := total_cheese total_per_week weeks) :
  (total_for_weeks / packages) = 30 :=
by
  unfold total_cheese_per_week total_cheese
  sorry

end string_cheese_per_package_l826_826722


namespace column_sum_problem_l826_826428

theorem column_sum_problem :
  ∃ (f : Fin 9 → ℕ),
    (∀ i, f i ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) ∧
    (∑ i : Fin 9, f i = 45) ∧
    (∑ i in Finset.range 1, f (Fin.ofNat i) = 7) ∧
    (∑ i in Finset.range 1, f (Fin.ofNat i) + 1 = ∑ i in Finset.range 2, f (Fin.ofNat i)) ∧
    (∑ i in Finset.range 2, f (Fin.ofNat i) + 1 = ∑ i in Finset.range 3, f (Fin.ofNat i)) ∧
    (∑ i in Finset.range 3, f (Fin.ofNat i) + 1 = ∑ i in Finset.range 4, f (Fin.ofNat i)) ∧
    (∑ i in Finset.range 4, f (Fin.ofNat i) + 1 = ∑ i in Finset.range 5, f (Fin.ofNat i)) :=
sorry

end column_sum_problem_l826_826428


namespace polygon_sides_from_diagonals_l826_826673

theorem polygon_sides_from_diagonals (n D : ℕ) (h1 : D = 15) (h2 : D = n * (n - 3) / 2) : n = 8 :=
by
  -- skipping proof
  sorry

end polygon_sides_from_diagonals_l826_826673


namespace remainder_of_1450_div_45_l826_826325

theorem remainder_of_1450_div_45 : nat.mod 1450 45 = 10 := 
by
  sorry

end remainder_of_1450_div_45_l826_826325


namespace remaining_numbers_after_erasure_l826_826390

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826390


namespace num_factors_20160_l826_826577

theorem num_factors_20160 : Nat.divisors 20160 = 72 := by
  sorry

end num_factors_20160_l826_826577


namespace determine_tax_rate_l826_826864

-- Definition of the conditions
def annual_sales_volume (r : ℝ) : ℝ := 80 - (20 / 3) * r
def minimum_tax_revenue : ℝ := 25600 * 1000 -- in yuan

def satisfies_tax_condition (r : ℝ) : Prop :=
  120 * (annual_sales_volume r) * r * 1000 ≥ minimum_tax_revenue

-- Statement of the theorem
theorem determine_tax_rate (r : ℝ) (h : 4 ≤ r ∧ r ≤ 8) : satisfies_tax_condition r :=
  by sorry

end determine_tax_rate_l826_826864


namespace remainder_when_divided_by_15_l826_826376

theorem remainder_when_divided_by_15 (c d : ℤ) (h1 : c % 60 = 47) (h2 : d % 45 = 14) : (c + d) % 15 = 1 :=
  sorry

end remainder_when_divided_by_15_l826_826376


namespace field_trip_l826_826219

theorem field_trip (students : Fin 20) (trips : Type) 
  (participates : students → trips → Prop)
  (H_trip_participation : ∀ t : trips, ∃ s1 s2 s3 s4 : students, participates s1 t ∧ participates s2 t ∧ participates s3 t ∧ participates s4 t) :
  ∃ t : trips, ∀ s : students, participates s t → ∃ n : ℕ, n ≥ 1 ∧ participates_at_least s n trips := sorry

end field_trip_l826_826219


namespace larger_number_l826_826666

theorem larger_number (x y : ℕ) (h1 : x * y = 40) (h2 : x + y = 14) (h3 : |x - y| ≤ 6) : max x y = 10 :=
sorry

end larger_number_l826_826666


namespace eccentricity_range_l826_826992

open Set

-- Definitions directly from condition a)
variable (a b : ℝ) (h : a > b ∧ b > 0) (c : ℝ) (h1 : c^2 = a^2 - b^2)

def on_ellipse (P : ℝ × ℝ) : Prop := (P.1^2 / a^2) + (P.2^2 / b^2) = 1

def perpendicular (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
(P.1 - F1.1) * (P.1 - F2.1) + P.2 * P.2 = 0

-- Lean theorem statement
theorem eccentricity_range (e : ℝ) :
  (∃ P : ℝ × ℝ, on_ellipse a b P ∧ perpendicular P (-c, 0) (c, 0)) →
  ∃ e, (sqrt (1 - (b^2 / a^2))) = e ∧ (sqrt 2 / 2) ≤ e ∧ e < 1 :=
sorry

end eccentricity_range_l826_826992


namespace product_of_possible_b_values_l826_826456

theorem product_of_possible_b_values (b : ℝ) :
  (∀ (y1 y2 x1 x2 : ℝ), y1 = -1 ∧ y2 = 3 ∧ x1 = 2 ∧ (x2 = b) ∧ (y2 - y1 = 4) → 
   (b = 2 + 4 ∨ b = 2 - 4)) → 
  (b = 6 ∨ b = -2) → (b = 6) ∧ (b = -2) → 6 * -2 = -12 :=
sorry

end product_of_possible_b_values_l826_826456


namespace max_and_min_values_g_l826_826102

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3)
noncomputable def g (x : ℝ) : ℝ := abs (x - 2) + abs (x - 3) - abs (x - 1)

theorem max_and_min_values_g :
  (∀ x, (2 ≤ x ∧ x ≤ 3) → f x = 1) →
  (∃ a b, (∀ x, (2 ≤ x ∧ x ≤ 3) → a ≤ g x ∧ g x ≤ b) ∧ a = -1 ∧ b = 0) :=
by
  intros H
  use [-1, 0]
  split
  sorry
  sorry

end max_and_min_values_g_l826_826102


namespace range_of_a_l826_826677

def f (x : ℝ) : ℝ := x^3 - 3 * x

noncomputable def derivative_f (x : ℝ) : ℝ := deriv f x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Ioo a (6 - a^2), f x ≥ f 1) →
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l826_826677


namespace range_of_m_l826_826331

theorem range_of_m (m : ℝ) : 0 < m ∧ m < 2 ↔ (2 - m > 0 ∧ - (1 / 2) * m < 0) := by
  sorry

end range_of_m_l826_826331


namespace ellipse_dot_product_constant_l826_826241

noncomputable def ellipse_C := {a b : ℝ // a = 2 ∧ b = sqrt 2 ∧ a > b > 0 ∧ (∃ x y, (x^2 / a^2) + (y^2 / b^2) = 1)}

theorem ellipse_dot_product_constant (M R : ℝ × ℝ) (P : ℝ × ℝ) (h_P : P = (sqrt 2, 1)) 
  {a b : ℝ} (h_a : a = 2) (h_b : b = sqrt 2) 
  (h_eccentricity : sqrt (a^2 - b^2) / a = sqrt 2 / 2) 
  (h_MA2_perpendicular : ∀ M A1 A2 : ℝ × ℝ, M =(2,y0) ∧ A1 = (-2,0) ∧ A2 = (2,0) → M.1 - A2.1 = M.2 - A2.2) 
  (h_MA1_intersection : ∀ M R A1 : ℝ × ℝ, M = (2, y0) ∧ A1 = (-2,0) ∧ R ≠ A1 → ∃ R, (R.1^2 / a^2) + (R.2^2 / b^2) = 1 ∧ (y = y0 / 4 * x + y0 / 2))
  (h_point_R : ∃ x1 y1, ∀ M : ℝ × ℝ, let R := (x1, y1) in x1 = -2 * (y0^2 - 8) / (y0^2 + 8) ∧ y1 = 8 * y0 / (y0^2 + 8))
  (h_dot_product : ∀ (OR OM : ℝ × ℝ), OR = (x1, y1) ∧ OM = (2, y0) → OR.1 * OM.1 + OR.2 * OM.2 = 4) :
  ∀ M R, ∃ c, ∀ y0 : ℝ, c = 4 := 
sorry

end ellipse_dot_product_constant_l826_826241


namespace log_stack_total_l826_826546

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l826_826546


namespace find_n_gizmos_in_two_hours_l826_826690

-- Define the conditions
def workers_produce_gadgets_gizmos_condition1 (num_workers : ℕ) (g_prod : ℕ) (zg_prod : ℕ) : Prop :=
  num_workers = 150 ∧ g_prod = 450 ∧ zg_prod = 300

def workers_produce_gadgets_gizmos_condition2 (num_workers : ℕ) (g_prod : ℕ) (zg_prod : ℕ) : Prop :=
  num_workers = 100 ∧ g_prod = 600 ∧ zg_prod = 900

def workers_produce_gadgets_gizmos_condition3 (num_workers : ℕ) (g_prod : ℕ) (zg_prod : ℕ) : Prop :=
  num_workers = 75 ∧ g_prod = 300

-- The theorem statement encapsulating the problem
theorem find_n_gizmos_in_two_hours :
  (∀ n : ℕ, workers_produce_gadgets_gizmos_condition1 150 450 300 →
            workers_produce_gadgets_gizmos_condition2 100 600 900 →
            workers_produce_gadgets_gizmos_condition3 75 300 n → 
            n = 450) :=
by 
  intro n
  assume (h1 : workers_produce_gadgets_gizmos_condition1 150 450 300)
         (h2 : workers_produce_gadgets_gizmos_condition2 100 600 900)
         (h3 : workers_produce_gadgets_gizmos_condition3 75 300 n)
  sorry

end find_n_gizmos_in_two_hours_l826_826690


namespace train_speed_is_72_km_per_hr_l826_826898

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l826_826898


namespace sin_BPD_l826_826775

noncomputable def points : Type := sorry

variables (A B C D E : points)
variables (P : points)

axiom equally_spaced : (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D E)
axiom cos_BPC : real.cos (angle B P C) = 3 / 5
axiom cos_CPD : real.cos (angle C P D) = 4 / 5

theorem sin_BPD : real.sin (angle B P D) = 1 := 
  sorry

end sin_BPD_l826_826775


namespace hexagon_inequality_l826_826704

variables {A B C D E F G H : Type*}
variables [metric_space ℝ] [ordered_field ℝ] [add_comm_group ℝ] [module ℝ ℝ] [has_complex_field ℝ] 
variables [has_inner ℝ ℝ] [inner_product_space ℝ ℝ] [ring ℝ] (AG GB GH DH HE CF : ℝ)

def hexagon_property (AB BC CD DE EF FA : ℝ) (angle_BCD angle_EFA : ℝ) : Prop :=
  AB = BC ∧ BC = CD ∧ DE = EF ∧ EF = FA ∧ angle_BCD = 60 ∧ angle_EFA = 60

theorem hexagon_inequality 
  (h : hexagon_property AB BC CD DE EF FA angle_BCD angle_EFA)
  (G H : ℝ) :
  AG + GB + GH + DH + HE ≥ CF := 
sorry

end hexagon_inequality_l826_826704


namespace find_time_when_velocity_is_one_l826_826863

-- Define the equation of motion
def equation_of_motion (t : ℝ) : ℝ := 7 * t^2 + 8

-- Define the velocity function as the derivative of the equation of motion
def velocity (t : ℝ) : ℝ := by
  let s := equation_of_motion t
  exact 14 * t  -- Since we calculated the derivative above

-- Statement of the problem to be proved
theorem find_time_when_velocity_is_one : 
  (velocity (1 / 14)) = 1 :=
by
  -- Placeholder for the proof
  sorry

end find_time_when_velocity_is_one_l826_826863


namespace logarithm_identity_l826_826947

variable (m n p q x z : ℝ)
variable (h₁ : m > 0) (h₂ : n > 0) (h₃ : p > 0) (h₄ : q > 0) (h₅ : x > 0) (h₆ : z > 0)

theorem logarithm_identity :
  log (m / n) + log (n / p) + log (p / q) - log (mx / qz) = log (z / x) :=
by
  sorry

end logarithm_identity_l826_826947


namespace exists_xy_nat_divisible_l826_826779

theorem exists_xy_nat_divisible (n : ℕ) : ∃ x y : ℤ, (x^2 + y^2 - 2018) % n = 0 :=
by
  use 43, 13
  sorry

end exists_xy_nat_divisible_l826_826779


namespace sum_of_elements_in_T_in_base3_l826_826729

def T : Set ℕ := {n | ∃ (d1 d2 d3 d4 d5 : ℕ), (1 ≤ d1 ∧ d1 ≤ 2) ∧ (0 ≤ d2 ∧ d2 ≤ 2) ∧ (0 ≤ d3 ∧ d3 ≤ 2) ∧ (0 ≤ d4 ∧ d4 ≤ 2) ∧ (0 ≤ d5 ∧ d5 ≤ 2) ∧ n = d1 * 3^4 + d2 * 3^3 + d3 * 3^2 + d4 * 3^1 + d5}

theorem sum_of_elements_in_T_in_base3 : (∑ x in T, x) = 2420200₃ := 
sorry

end sum_of_elements_in_T_in_base3_l826_826729


namespace friend_owns_10_bicycles_l826_826683

variable (ignatius_bicycles : ℕ)
variable (tires_per_bicycle : ℕ)
variable (friend_tires_ratio : ℕ)
variable (unicycle_tires : ℕ)
variable (tricycle_tires : ℕ)

def friend_bicycles (friend_bicycle_tires : ℕ) : ℕ :=
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_10_bicycles :
  ignatius_bicycles = 4 →
  tires_per_bicycle = 2 →
  friend_tires_ratio = 3 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_bicycles (friend_tires_ratio * (ignatius_bicycles * tires_per_bicycle) - unicycle_tires - tricycle_tires) = 10 :=
by
  intros
  -- Proof goes here
  sorry

end friend_owns_10_bicycles_l826_826683


namespace average_sequence_x_l826_826446

theorem average_sequence_x (x : ℚ) (h : (5050 + x) / 101 = 50 * x) : x = 5050 / 5049 :=
by
  sorry

end average_sequence_x_l826_826446


namespace exists_x_y_divisible_l826_826777

theorem exists_x_y_divisible (n : ℕ) : ∃ (x y : ℤ), n ∣ (x^2 + y^2 - 2018) :=
by {
  use [43, 13],
  simp,
  sorry
}

end exists_x_y_divisible_l826_826777


namespace profit_percentage_correct_l826_826148

-- Define the conversion rates, purchase price and selling price
def conversion_rate_A_C : ℝ := 0.75
def conversion_rate_B_C : ℝ := 1.25
def purchase_price_A : ℝ := 50
def selling_price_B : ℝ := 100

-- Calculate the purchase price and selling price in currency C
def purchase_price_C : ℝ := purchase_price_A * conversion_rate_A_C
def selling_price_C : ℝ := selling_price_B * conversion_rate_B_C

-- Calculate the profit in currency C
def profit_C : ℝ := selling_price_C - purchase_price_C

-- Calculate the profit percentage in terms of currency C
def profit_percentage_C : ℝ := (profit_C / purchase_price_C) * 100

-- Statement to prove that the profit percentage in terms of currency C is 233.33%
theorem profit_percentage_correct : profit_percentage_C = 233.33 := by
  sorry

end profit_percentage_correct_l826_826148


namespace walnut_tree_initial_count_l826_826817

theorem walnut_tree_initial_count 
  (trees_planted : ℕ) 
  (total_trees : ℕ) 
  (original_trees : ℕ) 
  (h1 : trees_planted = 104) 
  (h2 : total_trees = 211) 
  (h3 : total_trees = original_trees + trees_planted) :
  original_trees = 107 := 
by
  have h4 : original_trees + trees_planted = 211 := by rw [h2, h3]
  have h5 : original_trees + 104 = 211 := by rw [h1, h4]
  have h6 : original_trees = 211 - 104 := by linarith
  exact eq.trans h6 (by norm_num)

end walnut_tree_initial_count_l826_826817


namespace odd_function_monotonic_behavior_l826_826258

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Definition of a monotonically decreasing function on an interval
def is_monotonically_decreasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f (x1) > f (x2)

-- The proof problem statement
theorem odd_function_monotonic_behavior (f : ℝ → ℝ) :
  is_odd f → 
  is_monotonically_decreasing_on f 1 2 →
  is_monotonically_decreasing_on f (-2) (-1) ∧ 
  ∃ x ∈ Icc (-2 : ℝ) (-1), f x = -f 2 :=
by
  sorry

end odd_function_monotonic_behavior_l826_826258


namespace no_solutions_l826_826960

theorem no_solutions :
  ∀ x : ℝ, ¬ (sqrt (9 - x^2) = x * sqrt (9 - x^2) + x) :=
by
  sorry

end no_solutions_l826_826960


namespace least_subtracted_divisible_l826_826112

theorem least_subtracted_divisible :
  ∃ k, (5264 - 11) = 17 * k :=
by
  sorry

end least_subtracted_divisible_l826_826112


namespace projections_on_same_circle_l826_826522

variable (A B C D I B1 B2 D1 D2 : Point)
variable (IA IC : Line)

-- Given conditions
def quadrilateral_circumscribed (A B C D I : Point) : Prop := -- Define the circumscribed quadrilateral
  ∃ k : Circle, k.center = I ∧ quadrilateral A B C D ∧ circle_tangent_to_all_sides k A B C D

def orthogonal_projection (P : Point) (l : Line) : Point := sorry

-- The specific projections
axiom B1_proj : orthogonal_projection B IA = B1
axiom B2_proj : orthogonal_projection B IC = B2
axiom D1_proj : orthogonal_projection D IA = D1
axiom D2_proj : orthogonal_projection D IC = D2

-- The goal to prove
theorem projections_on_same_circle 
  (circumscribed : quadrilateral_circumscribed A B C D I)
  (B1_proj : orthogonal_projection B IA = B1)
  (B2_proj : orthogonal_projection B IC = B2)
  (D1_proj : orthogonal_projection D IA = D1)
  (D2_proj : orthogonal_projection D IC = D2)
  : ∃ T : Point, is_midpoint T B D ∧ on_circle T {B1, B2, D1, D2} :=
  sorry

end projections_on_same_circle_l826_826522


namespace calculate_sum_calculate_product_l826_826924

theorem calculate_sum : 13 + (-7) + (-6) = 0 :=
by sorry

theorem calculate_product : (-8) * (-4 / 3) * (-0.125) * (5 / 4) = -5 / 3 :=
by sorry

end calculate_sum_calculate_product_l826_826924


namespace cost_price_per_meter_of_cloth_l826_826156

theorem cost_price_per_meter_of_cloth 
  (total_meters : ℕ)
  (selling_price : ℝ)
  (profit_per_meter : ℝ) 
  (total_profit : ℝ)
  (cp_45 : ℝ)
  (cp_per_meter: ℝ) :
  total_meters = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * total_meters →
  cp_45 = selling_price - total_profit →
  cp_per_meter = cp_45 / total_meters →
  cp_per_meter = 86 :=
by
  -- your proof here
  sorry

end cost_price_per_meter_of_cloth_l826_826156


namespace non_intersecting_teams_l826_826540

-- Define the conditions of the problem
def school_senior_classes (M : ℕ) (class1 class2 class3 : finset ℕ) : Prop :=
  class1.card = M ∧ class2.card = M ∧ class3.card = M

def knows_each_other (students1 students2 students3 : finset ℕ) (a b c : ℕ) : Prop :=
  a ∈ students1 ∧ b ∈ students2 ∧ c ∈ students3 ∧
  ∀ x ∈ students1, ∀ y ∈ students2, ∀ z ∈ students3, x ≠ a → y ≠ b → z ≠ c →
  x ≠ y ∧ y ≠ z ∧ z ≠ x

def conditions (M : ℕ) (class1 class2 class3 : finset ℕ) : Prop :=
  ∀ a ∈ class1, (finset.filter (λ x, x ∈ class2) (finset.of_list [1..M])).card ≥ (3 * M / 4) ∧
  ∀ b ∈ class2, (finset.filter (λ x, x ∈ class3) (finset.of_list [1..M])).card ≥ (3 * M / 4) ∧
  ∀ c ∈ class3, (finset.filter (λ x, x ∈ class1) (finset.of_list [1..M])).card ≥ (3 * M / 4)

-- The statement of the problem
theorem non_intersecting_teams
  (M : ℕ)
  (class1 class2 class3 : finset ℕ)
  (h_classes : school_senior_classes M class1 class2 class3)
  (h_conditions : conditions M class1 class2 class3) :
  ∃ teams : finset (fin (3 * M)), teams.card = M ∧
  ∀ team ∈ teams, ∃ a b c : ℕ, a ∈ class1 ∧ b ∈ class2 ∧ c ∈ class3 ∧ knows_each_other class1 class2 class3 a b c :=
  sorry

end non_intersecting_teams_l826_826540


namespace find_theta_l826_826257

noncomputable def z1 (θ : ℝ) : ℂ := (Real.sin θ)^2 + complex.I
noncomputable def z2 (θ : ℝ) : ℂ := - (Real.cos θ)^2 + (Real.cos (2 * θ)) * complex.I

theorem find_theta (θ : ℝ) (hθ : θ > 0 ∧ θ < Real.pi) :
  let z := -1 + (Real.cos (2 * θ) - 1) * complex.I in
  (z.im = (1 / 2) * z.re) → (θ = Real.pi / 6 ∨ θ = 5 * Real.pi / 6) :=
by
  intros
  sorry

end find_theta_l826_826257


namespace color_arrangement_l826_826479

-- Let us define the problem conditions and the proof goal 
theorem color_arrangement (people : Finset ℕ) (color : ℕ → char) (h1 : card people = 4)
  (h2 : ∀ x y ∈ people, x ≠ y → (color x = 'R' ∨ color x = 'Y') ∧ (color x ≠ color y) → 
    no_adj_same_color (people : list ℕ) (color : ℕ → char)) : ∃ n, n = 8 := by
  sorry

end color_arrangement_l826_826479


namespace remaining_numbers_after_erasure_l826_826388

theorem remaining_numbers_after_erasure :
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  list.length final_list = 8 :=
by
  let initial_list := list.range 20
  let odd_numbers := list.filter (λ x, ¬ (x % 2 = 0)) initial_list
  let final_list := list.filter (λ x, ¬ (x % 5 = 4)) odd_numbers
  show list.length final_list = 8 from sorry

end remaining_numbers_after_erasure_l826_826388


namespace projection_correct_l826_826150

open Real

-- Define the initial vectors and their projection
def vec_a : ℝ × ℝ := (6, 2)
def proj_vec_a : ℝ × ℝ := (36/5, 12/5)

-- Define the vector to project
def vec_b : ℝ × ℝ := (2, -4)

-- Define the function for vector projection onto another vector
def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  let factor := dot_uv / dot_vv
  (factor * v.1, factor * v.2)

-- Define the simplified projection vector we project onto
def u : ℝ × ℝ := (3, 1)

-- The expected result of the projection
def expected_proj_vec_b : ℝ × ℝ := (3/5, 1/5)

theorem projection_correct :
  proj vec_b u = expected_proj_vec_b :=
sorry

end projection_correct_l826_826150


namespace find_b_l826_826385

theorem find_b (a b : ℤ) (h_a : a = 2) (h : |a - b| = 7) : b = -5 ∨ b = 9 :=
by
  rw h_a at h
  norm_num at h
  sorry

end find_b_l826_826385


namespace perfect_cube_factors_of_108_l826_826663

theorem perfect_cube_factors_of_108 : 
  let n := 108 in
  let prime_factorization := (2, 2) :: (3, 3) :: [] in
  let is_perfect_cube (factor : ℕ) := 
    ∀ (p e : ℕ), p ∣ n → factor = p^e → e % 3 = 0 in
  (∃ count, count = 2 ∧ 
   count = (∑ f in ((2 :: 3 :: []), if is_perfect_cube f then 1 else 0))) := 
sorry

end perfect_cube_factors_of_108_l826_826663


namespace average_speed_last_segment_l826_826718

-- Definitions based on the conditions
def total_distance : ℝ := 150
def total_time : ℝ := 135 / 60
def average_speed_first_segment : ℝ := 75
def average_speed_second_segment : ℝ := 80

-- Statement to be proved
theorem average_speed_last_segment :
  let x := (3 * (total_distance / total_time)) - average_speed_first_segment - average_speed_second_segment in
  x = 45 := by
  sorry

end average_speed_last_segment_l826_826718


namespace abs_difference_extrema_l826_826100

theorem abs_difference_extrema (x : ℝ) (h : 2 ≤ x ∧ x < 3) :
  max (|x-2| + |x-3| - |x-1|) = 0 ∧ min (|x-2| + |x-3| - |x-1|) = -1 :=
by
  sorry

end abs_difference_extrema_l826_826100


namespace wheelC_rotation_l826_826334

def radiusA := 35 -- cm
def radiusB := 20 -- cm
def radiusC := 8 -- cm
def angleA := 72 -- degrees

theorem wheelC_rotation : 
  let arc_length := (angleA / 360) * (2 * (Real.pi) * radiusA),
      angleC := 360 * (arc_length / (2 * (Real.pi) * radiusC))
  in angleC = 315 :=
by
  sorry

end wheelC_rotation_l826_826334


namespace find_b_l826_826059

theorem find_b (a b c : ℝ) (h1 : a + b + c = 120) (h2 : a + 5 = b - 5) (h3 : b - 5 = c^2) : b = 61.25 :=
by {
  sorry
}

end find_b_l826_826059


namespace train_cross_time_l826_826712

def length_of_train : ℝ := 240 -- Length of the train in meters
def speed_in_kmh : ℝ := 126 -- Speed of the train in km/hr
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * 1000 / 3600 -- Conversion factor from km/hr to m/s

def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := kmh_to_ms speed_kmh
  length / speed_ms

theorem train_cross_time : time_to_cross_pole length_of_train speed_in_kmh = 240 / (126 * 1000 / 3600) :=
by
  sorry

end train_cross_time_l826_826712


namespace marshmallow_ratio_l826_826288

theorem marshmallow_ratio:
  (∀ h m b, 
    h = 8 ∧ 
    m = 3 * h ∧ 
    h + m + b = 44
  ) → (1 / 2 = b / m) :=
by
sorry

end marshmallow_ratio_l826_826288


namespace lucy_found_additional_shells_l826_826374

theorem lucy_found_additional_shells (initial final : ℝ) (h1 : initial = 68.3) (h2 : final = 89.5) :
  final - initial = 21.2 :=
by {
  rw [h1, h2],
  norm_num,
  exact rfl,
}

end lucy_found_additional_shells_l826_826374


namespace area_of_square_PQRS_l826_826700

-- Define the properties as per the conditions
variables {P Q R S M N : Type} -- Points in the type, assumed, can be further specified

-- Assume P, Q, R, S form a square
axiom square_PQRS : @Geometry.isSquare P Q R S 

-- Assume M lies on side PQ and N lies on side RS with given lengths
axioms
  (PM MQ : ℝ) -- lengths of PM and MQ
  (RN NS : ℝ) -- lengths of RN and NS
  (PN NM MS : ℝ) -- lengths are all 20
  (PM_eq_MQ : PM = MQ)
  (RN_eq_NS : RN = NS)
  (PN_eq_NM_eq_MS : PN = 20 ∧ NM = 20 ∧ MS = 20)

-- The side length of the square
noncomputable def side_length (x : ℝ) : ℝ := x

-- Prove the area of the square PQRS is 800
theorem area_of_square_PQRS (x : ℝ) 
  (h1 : PM = x / 2) 
  (h2 : RN = x / 2)
  (h3 : x * real.sqrt 2 = 40) : 
  @Geometry.area P Q R S = 800 := 
sorry

end area_of_square_PQRS_l826_826700


namespace remainder_geometric_series_mod2000_l826_826563

theorem remainder_geometric_series_mod2000 :
  let S := (3^1501 - 1) / 2
  in S % 2000 = N :=
by
  sorry

end remainder_geometric_series_mod2000_l826_826563


namespace shaded_area_in_pattern_l826_826023

theorem shaded_area_in_pattern (d : ℝ) (length_ft : ℝ) (r : ℝ) 
  (area : ℝ) (num_semicircles : ℝ) :
  d = 3 → length_ft = 3 → r = d / 2 →
  num_semicircles = (length_ft * 12) →
  area = (num_semicircles * (1 / 2) * π * r^2) →
  area = 27 * π := 
by 
  intros h_d h_length h_r h_num h_area
  have : num_semicircles = 24 := 
  by { rw h_num, norm_num }
  have : r = 3 / 2 := by { rw h_r, norm_num }
  have h_semicircle_area : (1 / 2) * π * r^2 = (9 / 8) * π := 
  by { rw this, ring }
  have key : area = 24 * (9 / 8) * π := 
  by { rw [h_area, ←this, h_semicircle_area], ring }
  show area = 27 * π, 
  by { rw key, norm_num }

-- This theorem can be used for other diameters and lengths by manipulating the conditions accordingly.

end shaded_area_in_pattern_l826_826023


namespace other_coin_value_l826_826029

-- Condition definitions
def total_coins : ℕ := 36
def dime_count : ℕ := 26
def total_value_dollars : ℝ := 3.10
def dime_value : ℝ := 0.10

-- Derived definitions
def total_dimes_value : ℝ := dime_count * dime_value
def remaining_value : ℝ := total_value_dollars - total_dimes_value
def other_coin_count : ℕ := total_coins - dime_count

-- Proof statement
theorem other_coin_value : (remaining_value / other_coin_count) = 0.05 := by
  sorry

end other_coin_value_l826_826029


namespace median_number_of_children_l826_826757

theorem median_number_of_children 
    (counts : List ℕ)
    (h : counts = [1, 3, 2, 5, 3, 1, 4, 2, 3, 4]) : 
    median counts = 3 :=
by
  sorry

end median_number_of_children_l826_826757


namespace petya_numbers_l826_826394

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826394


namespace total_goldfish_preferring_students_l826_826914

def goldfish_preferring_students (num_students: ℕ) (fraction: ℚ) : ℕ :=
  (fraction * num_students).toNat

theorem total_goldfish_preferring_students :
  let mj_pref := goldfish_preferring_students 30 (1 / 6)
  let mf_pref := goldfish_preferring_students 30 (2 / 3)
  let mh_pref := goldfish_preferring_students 30 (1 / 5)
  mj_pref + mf_pref + mh_pref = 31 :=
by
  -- We use 'by' to indicate the need for a proof, but we leave it unproven with 'sorry'.
  sorry

end total_goldfish_preferring_students_l826_826914


namespace number_of_logs_in_stack_l826_826547

theorem number_of_logs_in_stack : 
    let first_term := 15 in
    let last_term := 5 in
    let num_terms := first_term - last_term + 1 in
    let average := (first_term + last_term) / 2 in
    let sum := average * num_terms in
    sum = 110 :=
by
  sorry

end number_of_logs_in_stack_l826_826547


namespace eight_sided_die_expected_value_l826_826228
noncomputable def eight_sided_die_probabilities : List (ℕ × ℚ) :=
[(1, 5 / 56), (2, 5 / 56), (3, 5 / 56), (4, 5 / 56), (5, 5 / 56), (6, 5 / 56), (7, 5 / 56), (8, 3 / 8)]

def expected_value (probs : List (ℕ × ℚ)) : ℚ :=
  probs.foldl (λ acc p, acc + p.1 * p.2) 0

theorem eight_sided_die_expected_value :
  expected_value eight_sided_die_probabilities = 5.5 :=
by
  -- Begin proof (to be completed)
  sorry

end eight_sided_die_expected_value_l826_826228


namespace pet_purchase_ways_l826_826149

-- Define the conditions
def number_of_puppies : Nat := 20
def number_of_kittens : Nat := 6
def number_of_hamsters : Nat := 8

def alice_choices : Nat := number_of_puppies

-- Define the problem statement in Lean
theorem pet_purchase_ways : 
  (number_of_puppies = 20) ∧ 
  (number_of_kittens = 6) ∧ 
  (number_of_hamsters = 8) → 
  (alice_choices * 2 * number_of_kittens * number_of_hamsters) = 1920 := 
by
  intros h
  sorry

end pet_purchase_ways_l826_826149


namespace right_triangle_ratio_l826_826692

theorem right_triangle_ratio (a b c r s : ℝ) (h_right_angle : (a:ℝ)^2 + (b:ℝ)^2 = c^2)
  (h_perpendicular : ∀ h : ℝ, c = r + s)
  (h_ratio_ab : a / b = 2 / 5)
  (h_geometry_r : r = a^2 / c)
  (h_geometry_s : s = b^2 / c) :
  r / s = 4 / 25 :=
sorry

end right_triangle_ratio_l826_826692


namespace total_students_prefer_goldfish_l826_826913

theorem total_students_prefer_goldfish :
  let students_per_class := 30
  let Miss_Johnson_fraction := 1 / 6
  let Mr_Feldstein_fraction := 2 / 3
  let Ms_Henderson_fraction := 1 / 5
  (Miss_Johnson_fraction * students_per_class) + 
  (Mr_Feldstein_fraction * students_per_class) +
  (Ms_Henderson_fraction * students_per_class) = 31 := 
by
  skip_proof

end total_students_prefer_goldfish_l826_826913


namespace point_of_tangency_l826_826641

def f (x : ℝ) : ℝ := (x^2) / 4 - Real.log x

theorem point_of_tangency : 
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧ 0 < x₀ ∧ (1/2)*x₀ - 1/x₀ = - (1/2) ∧ x₀ = 1 ∧ y₀ = 1/4 :=
by
  sorry

end point_of_tangency_l826_826641


namespace new_employees_hired_l826_826557

theorem new_employees_hired (initial_workers : ℕ) (men_fraction : ℚ) (initial_women : ℕ) (new_women : ℕ) (final_percentage : ℚ) : ℕ :=
  let total_initial_workers := initial_workers
  let num_men := (men_fraction * initial_workers).to_nat
  let num_women := total_initial_workers - num_men
  let x := new_women
  have h_initial_women : num_women = initial_women, from sorry
  have h_final_percent : ((initial_women + x : ℚ) / (initial_workers + x : ℚ)) = final_percentage, from sorry
  have h_equation : initial_women + x = final_percentage * (initial_workers + x), from sorry
  let solution := sorry -- The solution to the equation
  solution

noncomputable def number_of_employees_hired : ℕ :=
  new_employees_hired 90 (2/3) 30 10 (40 / 100)

end new_employees_hired_l826_826557


namespace find_slope_k_l826_826987

/- Problem Definitions -/
def c := Real.sqrt 5
def P := (0, 2 * Real.sqrt 3)

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  (x^2) / 9 + (y^2) / 4 = 1

def line_through_P (k : ℝ) : ℝ × ℝ → Prop := 
  λ (M : ℝ × ℝ), M.2 = k * M.1 + 2 * Real.sqrt 3

def segment_ratio (M N : ℝ × ℝ) : Prop := 
  (N.1 - P.1) = 3 * (M.1 - P.1) ∧ (N.2 - P.2) = 3 * (M.2 - P.2)

/- Proof Problem -/
theorem find_slope_k :
  (∃ a b,
    ellipse_eq a b ∧
    b = 2 * Real.sqrt 3 / 2 ∧
    a = Real.sqrt 9 ∧
    segment_ratio (a / 2, Real.sqrt 3) P) →
  (∃ k : ℝ,
    ∀ M N : ℝ × ℝ,
    line_through_P k M →
    line_through_P k N →
    segment_ratio M N →
    k = 4 * Real.sqrt 2 / 3 ∨ k = -4 * Real.sqrt 2 / 3) :=
sorry

end find_slope_k_l826_826987


namespace mutually_exclusive_not_necessarily_complementary_l826_826758

-- Define what it means for events to be mutually exclusive
def mutually_exclusive (E1 E2 : Prop) : Prop :=
  ¬ (E1 ∧ E2)

-- Define what it means for events to be complementary
def complementary (E1 E2 : Prop) : Prop :=
  (E1 ∨ E2) ∧ ¬ (E1 ∧ E2) ∧ (¬ E1 ∨ ¬ E2)

theorem mutually_exclusive_not_necessarily_complementary :
  ∀ E1 E2 : Prop, mutually_exclusive E1 E2 → ¬ complementary E1 E2 :=
sorry

end mutually_exclusive_not_necessarily_complementary_l826_826758


namespace repeating_decimal_sum_l826_826951

theorem repeating_decimal_sum :
  let x := (1 : ℚ) / 3
  let y := (7 : ℚ) / 33
  x + y = 6 / 11 :=
  by
  sorry

end repeating_decimal_sum_l826_826951


namespace find_value_of_n_l826_826967

noncomputable def is_correct_n (n : ℕ) : Prop :=
  (n ≥ 100 ∧ n < 1000) ∧ -- n is a 3-digit number
  (∃ k : ℤ, log 9 (n : ℝ) = k) ∧ -- log_9(n) is a whole number
  (∃ m : ℤ, log 3 (n : ℝ) + log 9 (n : ℝ) = m) -- log_3(n) + log_9(n) is a whole number

theorem find_value_of_n : ∀ n : ℕ, is_correct_n n → n = 9 :=
by
  intros n hn
  sorry

end find_value_of_n_l826_826967


namespace gauss_floor_function_root_is_2_l826_826602

noncomputable def f (x : ℝ) : ℝ := Real.log x - (2 / x)

theorem gauss_floor_function_root_is_2 (x₀ : ℝ) (h₀ : f x₀ = 0) : 
  ∀ x ∈ Icc 2 3, ⌊x₀⌋ = 2 := by
  sorry

end gauss_floor_function_root_is_2_l826_826602


namespace temperature_on_thursday_l826_826470

theorem temperature_on_thursday (temp_sun temp_mon temp_tue temp_wed temp_fri temp_sat temp_thu : ℕ) 
    (avg_temp : ℕ) 
    (h1 : temp_sun = 40) 
    (h2 : temp_mon = 50) 
    (h3 : temp_tue = 65) 
    (h4 : temp_wed = 36) 
    (h5 : temp_fri = 72) 
    (h6 : temp_sat = 26) 
    (h_avg : avg_temp = 53) :
    temp_thu = 82 :=
by
  let total_days := 7
  let sum_known_temperatures := temp_sun + temp_mon + temp_tue + temp_wed + temp_fri + temp_sat
  let total_sum := avg_temp * total_days
  have : total_sum = 371 := by sorry
  have : sum_known_temperatures = 289 := by sorry
  let temp_thu_calc := total_sum - sum_known_temperatures
  have : temp_thu_calc = 82 := by sorry
  rw h1 at this
  rw h2 at this
  rw h3 at this
  rw h4 at this
  rw h5 at this
  rw h6 at this
  exact this

end temperature_on_thursday_l826_826470


namespace train_length_l826_826511

theorem train_length (speed : ℝ) (time_seconds : ℝ) (time_hours : ℝ) (distance_km : ℝ) (distance_m : ℝ) 
  (h1 : speed = 60) 
  (h2 : time_seconds = 42) 
  (h3 : time_hours = time_seconds / 3600)
  (h4 : distance_km = speed * time_hours) 
  (h5 : distance_m = distance_km * 1000) :
  distance_m = 700 :=
by 
  sorry

end train_length_l826_826511


namespace derivative_of_y_l826_826846

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  log (sin (1 / 2)) - (1 / 24) * (cos (12 * x))^2 / sin (24 * x)

-- State the theorem to prove the derivative of y is y'
theorem derivative_of_y (x : ℝ) : deriv y x = 1 / (4 * (sin (12 * x))^2) :=
sorry

end derivative_of_y_l826_826846


namespace shooting_competition_hits_l826_826693

noncomputable def a1 : ℝ := 1
noncomputable def d : ℝ := 0.5
noncomputable def S_n (n : ℝ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem shooting_competition_hits (n : ℝ) (h : S_n n = 7) : 25 - n = 21 :=
by
  -- sequence of proof steps
  sorry

end shooting_competition_hits_l826_826693


namespace prob_min_ge_half_l826_826795

open_locale big_operators

-- The card picking problem setup
def num_cards : ℕ := 52
def total_ways_Blair_Corey : ℕ := nat.choose (num_cards - 2) 2

-- Function to calculate the probability p(a)
def p (a : ℕ) : ℚ :=
  (nat.choose (43 - a) 2 + nat.choose (a - 1) 2) / total_ways_Blair_Corey

-- The proof goal is to find m + n
theorem prob_min_ge_half : ∃ m n : ℕ,
  (m + n = 263) ∧ (∀ a : ℕ, p(a) ≥ (1 / 2) → m / n = p(a) ∧ nat.gcd m n = 1) :=
sorry

end prob_min_ge_half_l826_826795


namespace count_natural_numbers_perfect_square_l826_826660

theorem count_natural_numbers_perfect_square :
  ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (n1^2 - 19 * n1 + 91) = m^2 ∧ (n2^2 - 19 * n2 + 91) = k^2 ∧
  ∀ n : ℕ, (n^2 - 19 * n + 91) = p^2 → n = n1 ∨ n = n2 := sorry

end count_natural_numbers_perfect_square_l826_826660


namespace find_m_l826_826616

-- Definitions for the conditions
def point_on_terminal_side (α : ℝ) (m : ℝ) : Prop :=
  let P := (-8 * m, -3)
  ∧ (cos α = -4 / 5)

-- Problem statement to prove m = 3/4 given the conditions
theorem find_m (α : ℝ) (m : ℝ) :
  point_on_terminal_side α m →
  cos α = -4 / 5 →
  m = 3 / 4 :=
by
  intros h h_cos
  sorry

end find_m_l826_826616


namespace quadratic_always_positive_l826_826937

theorem quadratic_always_positive (k : ℝ) :
  ∀ x : ℝ, x^2 - (k - 4) * x + k - 7 > 0 :=
sorry

end quadratic_always_positive_l826_826937


namespace dillon_vs_luca_sum_diff_l826_826579

theorem dillon_vs_luca_sum_diff :
  let dillon_sum : ℕ := (List.range' 1 40).sum
  let luca_numbers := (List.range' 1 40).map (λ n, 
    let str_n := n.toString
    str_n.toList.map (λ c => if c = '3' then '2' else c)).map (λ l, l.toString.toNat)
  let luca_sum : ℕ := luca_numbers.sum
  dillon_sum = luca_sum + 104 :=
by
  sorry

end dillon_vs_luca_sum_diff_l826_826579


namespace sqrt_cos_cubic_poly_identity_l826_826197

theorem sqrt_cos_cubic_poly_identity :
  sqrt ((3 - cos^2 (π / 9)) * (3 - cos^2 (2 * π / 9)) * (3 - cos^2 (4 * π / 9))) = 27 * sqrt 2 / 16 := 
by
  sorry

end sqrt_cos_cubic_poly_identity_l826_826197


namespace perfect_squares_from_equation_l826_826225

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ a b c : ℕ, x - y = a^2 ∧ 2 * x + 2 * y + 1 = b^2 ∧ 3 * x + 3 * y + 1 = c^2 :=
by
  sorry

end perfect_squares_from_equation_l826_826225


namespace probability_of_inequality_is_61_over_81_l826_826379

noncomputable def probability_ball_inequality : ℚ :=
  let outcomes := {(a, b) | a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}};
  let favorable := {(a, b) ∈ outcomes | a - 2 * b + 10 > 0};
  favorable.card / outcomes.card

theorem probability_of_inequality_is_61_over_81 :
  probability_ball_inequality = 61 / 81 :=
sorry

end probability_of_inequality_is_61_over_81_l826_826379


namespace functional_eq_solution_l826_826936

noncomputable def functional_eq (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_eq f) :
  f = (λ x, 0) ∨ f = (λ x, x^2) :=
sorry

end functional_eq_solution_l826_826936


namespace cyclic_quadrilateral_area_l826_826238

theorem cyclic_quadrilateral_area 
  (A B C D : Type) 
  (AB BC CD DA : ℝ)
  (h1 : AB = 2) 
  (h2 : BC = 6) 
  (h3 : CD = 4) 
  (h4 : DA = 4) 
  (is_cyclic_quad : True) : 
  area A B C D = 8 * Real.sqrt 3 := 
sorry

end cyclic_quadrilateral_area_l826_826238


namespace Trent_traveled_distance_l826_826484

variable (blocks_length : ℕ := 50)
variables (walking_blocks : ℕ := 4) (bus_blocks : ℕ := 7) (bicycle_blocks : ℕ := 5)
variables (walking_round_trip : ℕ := 2 * walking_blocks * blocks_length)
variables (bus_round_trip : ℕ := 2 * bus_blocks * blocks_length)
variables (bicycle_round_trip : ℕ := 2 * bicycle_blocks * blocks_length)

def total_distance_traveleed : ℕ :=
  walking_round_trip + bus_round_trip + bicycle_round_trip

theorem Trent_traveled_distance :
  total_distance_traveleed = 1600 := by
    sorry

end Trent_traveled_distance_l826_826484


namespace numbers_left_on_board_l826_826422

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826422


namespace max_slips_not_divisible_by_5_l826_826685

-- Definitions based on conditions
def is_not_divisible_by_5 (n : ℕ) : Prop := n % 5 ≠ 0

def valid_triplet (a b c : ℕ) : Prop := 
  ∃ x y, x ≠ y ∧ {x, y} ⊆ {a, b, c} ∧ (x + y) % 5 = 0

def valid_set (s : Finset ℕ) : Prop := 
  ∀ (a b c : ℕ), {a, b, c} ⊆ s → valid_triplet a b c

-- The theorem to prove
theorem max_slips_not_divisible_by_5 (s : Finset ℕ) (h : valid_set s) :
  (∃ t : Finset ℕ, t ⊆ s ∧ (∀ n ∈ t, is_not_divisible_by_5 n) ∧ t.card ≤ 4) :=
sorry

end max_slips_not_divisible_by_5_l826_826685


namespace quadrilateral_angles_l826_826190

variable {α : Type} [LinearOrder α] [AddGroup α] [HasMul α]

theorem quadrilateral_angles :
  ∀ (A B C D : α)
  (AB AD BC : α)
  (angleA angleB angleC angleD : α)
  (h1 : angleA + angleB + angleC + angleD = 360)
  (h2 : angleB = 3 * angleA)
  (h3 : angleC = 3 * angleB)
  (h4 : angleD = 3 * angleC)
  (h5 : AD = BC),
  angleA = 9 ∧ angleB = 27 ∧ angleC = 81 ∧ angleD = 243 :=
by
  sorry

end quadrilateral_angles_l826_826190


namespace tangent_line_to_curve_l826_826317

section TangentLine

variables {x m : ℝ}

theorem tangent_line_to_curve (x0 : ℝ) :
  (∀ x : ℝ, x > 0 → y = x * Real.log x) →
  (∀ x : ℝ, y = 2 * x + m) →
  (x0 > 0) →
  (x0 * Real.log x0 = 2 * x0 + m) →
  m = -Real.exp 1 :=
by
  sorry

end TangentLine

end tangent_line_to_curve_l826_826317


namespace identify_irrational_number_l826_826909

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem identify_irrational_number :
  let options := [22/7, 3.14, -6, Real.sqrt 2] in
  (∀ x ∈ options, ¬(Real.sqrt 2 = x) → is_rational x) ∧ ¬is_rational (Real.sqrt 2) → 
  (∃ y ∈ options, y = Real.sqrt 2 ∧ ¬is_rational y) :=
by
  sorry

end identify_irrational_number_l826_826909


namespace margo_and_irma_pairing_probability_l826_826481

/-- Let there be 32 students in a class, with one student named Bob who cannot participate in the pairing.
The pairing is done randomly among the remaining 31 students. Prove that the probability that a student 
named Margo is paired with her best friend, Irma, is 1/30. -/
theorem margo_and_irma_pairing_probability : 
  (∃ n : ℕ, n = 32 ∧ ∀ bob : student, ∃ remaining : set student, remaining.card = 31 ∧ 
  ∀ pairs : list (student × student), pairs.perm (pair_students remaining) → 
  ∃ margo irma : student, margo ≠ irma → margo.paired_with ⟨irma, 1/30⟩) :=
sorry

end margo_and_irma_pairing_probability_l826_826481


namespace sum_of_solutions_l826_826215

theorem sum_of_solutions (x : ℤ) (h : x^4 - 13 * x^2 + 36 = 0) : 
  (finset.sum (finset.filter (λ (x : ℤ), x^4 - 13 * x^2 + 36 = 0) (finset.range 4))) = 0 :=
by
  sorry

end sum_of_solutions_l826_826215


namespace sum_of_elements_in_T_l826_826732

def T := {x : ℕ | ∀ d ∈ digits 3 x, d < 3 ∧ 6561 ≤ x ∧ x ≤ 24263}

theorem sum_of_elements_in_T :
  (∑ x in T, x) = 21102002₃ := sorry

end sum_of_elements_in_T_l826_826732


namespace main_theorem_l826_826244

def p : Prop :=
  ¬∃ x : ℝ, x > 1 ∧ log 2 (x + 1) - 1 = 0

def q : Prop :=
  ∀ a : ℝ, ((∃ x y : ℝ, (a-1) * x + 2 * y = 0 ∧ x - a * y + 1 = 0) ↔ a = -1)

theorem main_theorem : (¬p ∧ q) :=
by
  sorry

end main_theorem_l826_826244


namespace tangent_of_alpha_l826_826994

noncomputable def alpha : Real := sorry

theorem tangent_of_alpha 
  (P : Real × Real)
  (cos_alpha : Real)
  (hy_neg : P.snd < 0)
  (hcos : cos_alpha = 3 / 5)
  (hP : P = (3, P.snd)) :
  Real.tan alpha = -4 / 3 :=
by
  -- Given data
  have h1 : Real.cos alpha = cos_alpha := sorry
  have h2 : Real.sin alpha = -sqrt (1 - (cos_alpha ^ 2)) := sorry
  show Real.tan alpha = (Real.sin alpha) / (Real.cos alpha) := sorry
  rw [h1, h2]
  -- Simplify to get the result
  sorry

end tangent_of_alpha_l826_826994


namespace arithmetic_mean_neg3_to_6_l826_826490

theorem arithmetic_mean_neg3_to_6 : 
  let nums := [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6] in
  (sum_list nums) / (length nums) = 1.5 := 
sorry

end arithmetic_mean_neg3_to_6_l826_826490


namespace area_bounded_region_l826_826188

theorem area_bounded_region : 
  let eq := λ (x y : ℝ), y^2 + 2 * x * y + 30 * abs x = 500
  ∃ (A : ℝ), (∀ x y : ℝ, eq x y → True) ∧ A = 5000 / 3 :=
by
  let eq := λ (x y : ℝ), y^2 + 2 * x * y + 30 * abs x = 500
  existsi 5000 / 3
  split
  intros x y h
  trivial
  sorry

end area_bounded_region_l826_826188


namespace ryegrass_percent_of_mixture_l826_826786

noncomputable def mixture_percent_ryegrass (X_rye Y_rye portion_X : ℝ) : ℝ :=
  let portion_Y := 1 - portion_X
  let total_rye := (X_rye * portion_X) + (Y_rye * portion_Y)
  total_rye * 100

theorem ryegrass_percent_of_mixture :
  let X_rye := 40 / 100 
  let Y_rye := 25 / 100
  let portion_X := 1 / 3
  mixture_percent_ryegrass X_rye Y_rye portion_X = 30 :=
by
  sorry

end ryegrass_percent_of_mixture_l826_826786


namespace valid_sequences_count_is_2_pow_22_l826_826153

-- Definitions for transformations L, R, H, and V
def L := sorry -- 90 degree counterclockwise rotation
def R := sorry -- 90 degree clockwise rotation
def H := sorry -- reflection across the x-axis
def V := sorry -- reflection across the y-axis

-- We can skip the actual transformation mappings since we are interested in the sequence count
noncomputable def sequence_count_22_transformations : ℕ := 2^22

-- Statement to be proved
theorem valid_sequences_count_is_2_pow_22 :
  sequence_count_22_transformations = 2^22 :=
by
  -- proof will be provided here
  sorry

end valid_sequences_count_is_2_pow_22_l826_826153


namespace simplify_expression_l826_826026

variable (x : ℝ)

theorem simplify_expression : 1 - (2 - (3 - (4 - (5 - x)))) = 3 - x :=
by
  sorry

end simplify_expression_l826_826026


namespace sum_of_possible_d_values_l826_826865

def count_binary_digits (n : ℕ) : ℕ :=
  Nat.size n

theorem sum_of_possible_d_values : 
  let lower_bound := 36
  let upper_bound := 215
  let possible_d_values := {d | ∃ n (hn : 36 ≤ n ∧ n ≤ 215), count_binary_digits n = d}
  ∑ d in possible_d_values, d = 21 :=
by 
  sorry

end sum_of_possible_d_values_l826_826865


namespace trapezoid_shorter_base_length_l826_826328

theorem trapezoid_shorter_base_length
  (L B : ℕ)
  (hL : L = 125)
  (hB : B = 5)
  (h : ∀ x, (L - x) / 2 = B → x = 115) :
  ∃ x, x = 115 := by
    sorry

end trapezoid_shorter_base_length_l826_826328


namespace ratio_of_triangle_to_trapezoid_l826_826171

theorem ratio_of_triangle_to_trapezoid :
  -- Define the area function for an equilateral triangle
  ∀ (s₁ s₂ : ℕ) (h : s₂ = 2 * s₁),
  let area := λ s, (sqrt 3 / 4) * s^2 in
  let ratio := λ s₁ s₂, area s₁ / (3 * area s₁) in
  ratio 6 12 = 1 / 3 :=
begin
  intros s₁ s₂ h,
  unfold area ratio,
  rw h,
  simp,
  sorry,  -- the proof would go here
end

end ratio_of_triangle_to_trapezoid_l826_826171


namespace petya_numbers_l826_826396

theorem petya_numbers (S : Finset ℕ) (T : Finset ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} →
  T = S.filter (λ n, ¬ (Even n ∨ n % 5 = 4)) →
  T.card = 8 :=
by
  intros hS hT
  have : S = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19} := by sorry
  have : T = {1, 3, 5, 7, 11, 13, 15, 17} := by sorry
  sorry

end petya_numbers_l826_826396


namespace acute_angle_at_9_35_l826_826493

def minute_hand_degree (minute : ℕ) : ℝ :=
  (minute / 60) * 360

def hour_hand_degree (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour / 12) * 360 + (minute / 60) * 30

def acute_angle (h_deg : ℝ) (m_deg : ℝ) : ℝ :=
  let angle_diff := abs (h_deg - m_deg)
  if angle_diff > 180 then 360 - angle_diff else angle_diff

theorem acute_angle_at_9_35 : acute_angle (hour_hand_degree 9 35) (minute_hand_degree 35) = 77.5 := by
  sorry

end acute_angle_at_9_35_l826_826493


namespace find_g_l826_826305

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := f(x - 2)

theorem find_g : ∀ x : ℝ, g(x) = 2 * x - 1 := by
  intro x
  dsimp [g, f]
  sorry

end find_g_l826_826305


namespace sum_of_integer_solutions_l826_826213

theorem sum_of_integer_solutions :
  ∀ x : ℤ, (x^4 - 13 * x^2 + 36 = 0) → (∃ a b c d, x = a + b + c + d ∧ a + b + c + d = 0) :=
begin
  sorry
end

end sum_of_integer_solutions_l826_826213


namespace exists_a_triangle_with_three_distinct_colors_l826_826597

variable {A B C : Point}
variable (n : ℕ)

-- Defining our problem setup and constraints
def equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def divides_into_equal_segments (A B : Point) (n : ℕ) : Prop := 
  ∃ (points : Fin (n + 1) → Point), 
    points 0 = A ∧ points n = B ∧ 
    (∀ i, dist (points i) (points (i + 1)) = dist A B / n)

def colored_vertex (color : Color) (P : Point) : Prop := 
  ∃ c : ℕ, c ∈ {0, 1, 2} ∧ P.color = color

def condition_BC (P : Point) : Prop := 
  ∀ P, P ∈ BC → ¬ (P.color = red)

def condition_CA (P : Point) : Prop := 
  ∀ P, P ∈ CA → ¬ (P.color = blue)

def condition_AB (P : Point) : Prop := 
  ∀ P, P ∈ AB → ¬ (P.color = yellow)

theorem exists_a_triangle_with_three_distinct_colors
  (h1 : equilateral_triangle A B C)
  (h2 : divides_into_equal_segments A B n)
  (h3 : divides_into_equal_segments B C n)
  (h4 : divides_into_equal_segments C A n)
  (h5 : ∀ p, p ∈ {p | colored_vertex p})
  (h6 : condition_BC P)
  (h7 : condition_CA P)
  (h8 : condition_AB P) :
  ∃ (t : triangle), t ∈ all_small_triangles ∧ 
    (distinct_colors t.vertex1.color t.vertex2.color t.vertex3.color) :=
sorry

end exists_a_triangle_with_three_distinct_colors_l826_826597


namespace find_a100_find_a1983_l826_826841

open Nat

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k, a (a k) = 3 * k

theorem find_a100 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 100 = 181 := 
sorry

theorem find_a1983 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 1983 = 3762 := 
sorry

end find_a100_find_a1983_l826_826841


namespace train_speed_is_72_km_per_hr_l826_826901

-- Define the conditions
def length_of_train : ℕ := 180   -- Length in meters
def time_to_cross_pole : ℕ := 9  -- Time in seconds

-- Conversion factor
def conversion_factor : ℝ := 3.6

-- Prove that the speed of the train is 72 km/hr
theorem train_speed_is_72_km_per_hr :
  (length_of_train / time_to_cross_pole) * conversion_factor = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l826_826901


namespace max_xy_value_l826_826823

theorem max_xy_value (x y : ℝ) (h : x^2 + y^2 + 3 * x * y = 2015) : xy <= 403 :=
sorry

end max_xy_value_l826_826823


namespace closest_to_200_closest_to_200_rounded_l826_826853

noncomputable def original_expr := (2.73 * 7.91) * (4.25 + 5.75 - 0.5)
noncomputable def approx_expr  := (2.7 * 8) * 9.5

theorem closest_to_200 : abs(original_expr - 205.2) < 5 :=
by sorry

theorem closest_to_200_rounded : abs(original_expr - 200) < 100 :=
by
  have h : abs(original_expr - 205.2) < 5 := by apply closest_to_200
  -- given the expression was approximated to 205.2,
  -- and rounding that to the nearest hundred gives 200.
  sorry

end closest_to_200_closest_to_200_rounded_l826_826853


namespace find_a_plus_k_l826_826373

variable (a k : ℝ)

noncomputable def f (x : ℝ) : ℝ := (a - 1) * x^k

theorem find_a_plus_k
  (h1 : f a k (Real.sqrt 2) = 2)
  (h2 : (Real.sqrt 2)^2 = 2) : a + k = 4 := 
sorry

end find_a_plus_k_l826_826373


namespace inequality_relationship_l826_826609

noncomputable def a : ℝ := (Real.log 3 / Real.log 2)^3
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := 1 / Real.sqrt 5

theorem inequality_relationship : c < b < a :=
by sorry

end inequality_relationship_l826_826609


namespace planted_fraction_proof_l826_826208

-- Definitions based on given conditions
def right_triangle_field : Prop :=
  ∃ (AB AC BC : ℝ), AB = 5 ∧ AC = 12 ∧ BC = 13 ∧ ∠ AB = 90

def unplanted_square (x : ℝ) : Prop :=
  x = 20/13

def planted_fraction (planted_area total_area : ℚ) : Prop :=
  planted_area / total_area = 470/507

-- The main theorem statement
theorem planted_fraction_proof :
  right_triangle_field ∧ (unplanted_square (20 / 13)) → planted_fraction (4700/169) 30 :=
by
  sorry -- Proof omitted

end planted_fraction_proof_l826_826208


namespace probability_of_nickel_l826_826145

-- Define the conditions
def value_dimes : ℝ := 10.0
def value_nickels : ℝ := 5.0
def value_pennies : ℝ := 2.0

-- Define the individual coin values
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05
def value_per_penny : ℝ := 0.01

-- Calculate the number of each type of coin
def num_dimes : ℝ := value_dimes / value_per_dime
def num_nickels : ℝ := value_nickels / value_per_nickel
def num_pennies : ℝ := value_pennies / value_per_penny

-- Total number of coins
def total_coins : ℝ := num_dimes + num_nickels + num_pennies

-- The probability that a randomly chosen coin is a nickel
def probability_nickel : ℝ := num_nickels / total_coins

theorem probability_of_nickel :
  probability_nickel = 1/4 := 
sorry

end probability_of_nickel_l826_826145


namespace part1_part2_l826_826637

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x 
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1 (a : ℝ) (h : (derivative (f a) e) = 3) : a = 1 := 
sorry

theorem part2 (k : ℤ) (hk : ∀ (x : ℝ), 1 < x → k < g x) : k = 3 := 
sorry

end part1_part2_l826_826637


namespace a_n_formula_b_n_formula_G_n_formula_G_n_bounds_l826_826984

-- Definition of the sequences and the necessary conditions.

def a : ℕ → ℕ
| 1 := 1
| 2 := 3
| (n+1) := 2 * (a n) + 1

theorem a_n_formula (n : ℕ) (h_n : 2 ≤ n) : a n = 2^n - 1 :=
sorry

def b : ℕ → ℕ
| 1 := 1
| (n+1) := log2 (a n + 1) + b n

theorem b_n_formula (n : ℕ) (h_n : 1 ≤ n) : b n = (n * (n - 1)) / 2 + 1 :=
sorry

def c (n : ℕ) : ℝ := 4 ^ ((b (n+1) - 1) / (n+1 :ℝ)) / ((a n) * (a (n+1)))

def G (n : ℕ) : ℝ := ∑ k in Finset.range n, c (k + 1)

theorem G_n_formula (n : ℕ) : G n = 1 - 1 / (2 ^ (n+1) - 1) :=
sorry

theorem G_n_bounds (n : ℕ) : (2:ℝ) / 3 ≤ G n ∧ G n < 1 :=
sorry

end a_n_formula_b_n_formula_G_n_formula_G_n_bounds_l826_826984


namespace numbers_left_on_board_l826_826426

theorem numbers_left_on_board : 
  let initial_numbers := { n ∈ (finset.range 21) | n > 0 },
      without_evens := initial_numbers.filter (λ n, n % 2 ≠ 0),
      final_numbers := without_evens.filter (λ n, n % 5 ≠ 4)
  in final_numbers.card = 8 := by sorry

end numbers_left_on_board_l826_826426


namespace part_a_possible_part_b_impossible_l826_826227

-- Part (a): Prove possibility for amounts $10, $20, $30, $40, $50, and $60
theorem part_a_possible :
  (∀ (a b c d : ℕ) (amounts : Finset ℕ), 
  -- Conditions
  amounts = {10, 20, 30, 40, 50, 60} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  -- Financially neutral condition
  ∀ x ∈ {a, b, c, d}, 
  (∑ y in {a, b, c, d}, if x = y then -10 else 0) + 
  (∑ y in {a, b, c, d}, if x ≠ y then 10 else 0) = 0)
  → True :=
by
  sorry

-- Part (b): Prove impossibility for amounts $20, $30, $40, $50, $60, and $70
theorem part_b_impossible :
  ¬ (∀ (a b c d : ℕ) (amounts : Finset ℕ), 
  -- Conditions
  amounts = {20, 30, 40, 50, 60, 70} ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  -- Financially neutral condition
  ∀ x ∈ {a, b, c, d}, 
  (∑ y in {a, b, c, d}, if x = y then -20 else 0) + 
  (∑ y in {a, b, c, d}, if x ≠ y then 20 else 0) = 0) :=
by
  sorry

end part_a_possible_part_b_impossible_l826_826227


namespace count_final_numbers_l826_826411

def initial_numbers : List ℕ := List.range' 1 20

def erased_even_numbers : List ℕ :=
  initial_numbers.filter (λ n => n % 2 ≠ 0)

def final_numbers : List ℕ :=
  erased_even_numbers.filter (λ n => n % 5 ≠ 4)

theorem count_final_numbers : final_numbers.length = 8 :=
by
  -- Here, we would proceed to prove the statement.
  sorry

end count_final_numbers_l826_826411


namespace find_k_inverse_proportion_l826_826316

theorem find_k_inverse_proportion :
  ∃ k : ℝ, (∀ x y : ℝ, y = (k + 1) / x → (x = 1 ∧ y = -2) → k = -3) :=
by
  sorry

end find_k_inverse_proportion_l826_826316


namespace line_slope_tangent_to_circle_l826_826630

-- Define the given conditions: parabola and circle.
def parabola : ℝ → ℝ := λ y, y ^ 2 / 4
def circle : ℝ × ℝ → ℝ := λ p, (p.1 - 4) ^ 2 + p.2 ^ 2 - 4

-- Define the line passing through the focus of the parabola.
def line (k : ℝ) : ℝ → ℝ := λ x, k * (x - 1)

-- Define the function to calculate the distance from a point (x0, y0) to a line y = k(x-1).
def distance_to_line (x0 y0 A B C : ℝ) : ℝ := (A * x0 + B * y0 + C).abs / Real.sqrt (A ^ 2 + B ^ 2)

theorem line_slope_tangent_to_circle
  (k : ℝ)
  (hl : ∃ k, ∀ x, line k x)
  (hc : ∀ p, circle p = 0) :
  distance_to_line 4 0 k (-1) (-k) = 2 → k = 2 * Real.sqrt 5 / 5 ∨ k = -2 * Real.sqrt 5 / 5 := by
  sorry

end line_slope_tangent_to_circle_l826_826630


namespace bingyi_games_won_l826_826908

theorem bingyi_games_won (total_games : ℕ) (equal_games : ℕ) (alvin_win_rate : ℚ) (bingyi_win_rate : ℚ) (cheska_win_rate : ℚ) :
  total_games = 60 →
  equal_games = total_games / 3 →
  alvin_win_rate = 0.2 →
  bingyi_win_rate = 0.6 →
  cheska_win_rate = 0.4 →
  let games_ab := equal_games in
  let games_bc := equal_games in
  let games_ca := equal_games in
  let alvin_wins_ab := alvin_win_rate * games_ab in
  let bingyi_wins_ab := games_ab - alvin_wins_ab in
  let bingyi_wins_bc := bingyi_win_rate * games_bc in
  let bingyi_total_wins := bingyi_wins_ab + bingyi_wins_bc in
  bingyi_total_wins = 28 :=
by
  intro h_total_games h_equal_games h_alvin_win_rate h_bingyi_win_rate h_cheska_win_rate
  let games_ab := equal_games
  let games_bc := equal_games
  let games_ca := equal_games
  let alvin_wins_ab := alvin_win_rate * games_ab
  let bingyi_wins_ab := games_ab - alvin_wins_ab
  let bingyi_wins_bc := bingyi_win_rate * games_bc
  let bingyi_total_wins := bingyi_wins_ab + bingyi_wins_bc
  show bingyi_total_wins = 28
  sorry

end bingyi_games_won_l826_826908


namespace cyclic_quadrilateral_area_l826_826235

def area_of_cyclic_quadrilateral (AB BC CD DA : ℝ) : ℝ :=
  let A := 120 * (real.pi / 180) -- Angle in radians
  16 * (real.sin A)

theorem cyclic_quadrilateral_area
  (AB BC CD DA : ℝ)
  (hAB : AB = 2)
  (hBC : BC = 6)
  (hCD : CD = 4)
  (hDA : DA = 4)
  (hCyclic : true) -- Assumption that quadrilateral is cyclic
  : area_of_cyclic_quadrilateral AB BC CD DA = 8 * real.sqrt 3 :=
by
  rw [hAB, hBC, hCD, hDA]
  sorry

end cyclic_quadrilateral_area_l826_826235


namespace cost_of_four_dozen_bananas_l826_826561

/-- Given that five dozen bananas cost $24.00,
    prove that the cost for four dozen bananas is $19.20. -/
theorem cost_of_four_dozen_bananas 
  (cost_five_dozen: ℝ)
  (rate: cost_five_dozen = 24) : 
  ∃ (cost_four_dozen: ℝ), cost_four_dozen = 19.2 := by
  sorry

end cost_of_four_dozen_bananas_l826_826561


namespace area_inner_square_eq_l826_826707

theorem area_inner_square_eq :
  ∀ (A B C D : Point) (ABCD : quadrilateral),
    is_square ABCD →
    length (ABCD.side AB) = 10 →
    ∀ (E F G H : Point) (EFGH : quadrilateral),
      is_square EFGH →
      lies_on_segment E (B, D) →
      length (B, E) = 3 →
      radius (circle E 2) = 2 →
      touches_side (circle E 2) (ADCD) →
      area EFGH = 100 - 6 * real.sqrt 91 :=
by
  sorry

end area_inner_square_eq_l826_826707


namespace number_square_root_divide_6_l826_826499

theorem number_square_root_divide_6 (x : ℝ) (h : sqrt x / 6 = 1) : x = 36 :=
by
  sorry

end number_square_root_divide_6_l826_826499


namespace collinear_E_UK_VH_l826_826036

theorem collinear_E_UK_VH (ABCD : Type) [convex_quadrilateral ABCD] 
  (A B C D E U V H K : Point) 
  (h_diag : diagonals_intersect ABCD E) 
  (hU : circumcenter (triangle A B E) U) 
  (hH : orthocenter (triangle A B E) H) 
  (hV : circumcenter (triangle C D E) V) 
  (hK : orthocenter (triangle C D E) K) :
  collinear {U, E, K} ↔ collinear {V, E, H} :=
sorry

end collinear_E_UK_VH_l826_826036


namespace thermometer_distribution_l826_826079

theorem thermometer_distribution : 
  (∑ i in finset.range 10, if i < 10 then 1 else 0) ≥ 0 → (∑ i in finset.range 10, if i < 23 then 1 else 0) ≥ 0 → nat.choose 12 9 = 220 :=
sorry

end thermometer_distribution_l826_826079


namespace determine_coefficient_l826_826943

noncomputable def Q (x d : ℝ) : ℝ := x^3 - 3 * x^2 + d * x - 8

theorem determine_coefficient 
(d : ℝ) :
  (Q 3 d = 0) ↔ (d = 8 / 3) :=
by
  -- Define Q(x) based on the problem conditions
  let Q := λ (x d : ℝ), x^3 - 3 * x^2 + d * x - 8
  -- Evaluate Q(3)
  have hQ3 : Q 3 d = 3^3 - 3 * 3^2 + d * 3 - 8 := by rfl
  -- Prove the statement
  rw hQ3
  rw [pow_succ, pow_succ, pow_one, pow_one]
  norm_num
  split
  intro h
  linarith
  intro h
  linarith

end determine_coefficient_l826_826943
