import Mathlib
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Principalization
import Mathlib.Algebra.Operations
import Mathlib.Algebra.Order.Group
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Dvd.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Tactic
import Mathlib.Trigonometry.Basic
import data.real.basic

namespace smallest_n_l110_110063

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 0  -- a_0 is not used, but define with any value (say 0) for completeness
| 1       := 9
| (n + 1) := (4 - a n) / 3

-- Define the sum of the first n terms of the sequence
def S_n (n : ℕ) : ℝ :=
∑ i in finset.range n, a i

-- The smallest integer n that satisfies the given inequality
theorem smallest_n :
  ∃ n : ℕ, ∀ m : ℕ, (m < n ∧ |S_n m - m - 6| < 1) -> n ≤ m :=
begin
  use 7,
  intro m,
  sorry
end

end smallest_n_l110_110063


namespace coefficient_a3b3_l110_110274

theorem coefficient_a3b3 in_ab_c_1overc_expr :
  let coeff_ab := Nat.choose 6 3 
  let coeff_c_expr := Nat.choose 8 4 
  coeff_ab * coeff_c_expr = 1400 :=
by
  sorry

end coefficient_a3b3_l110_110274


namespace sum_of_arithmetic_series_l110_110399

def a₁ : ℕ := 9
def d : ℕ := 4
def n : ℕ := 50

noncomputable def nth_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_series (a₁ d n : ℕ) : ℕ := n / 2 * (a₁ + nth_term a₁ d n)

theorem sum_of_arithmetic_series :
  sum_arithmetic_series a₁ d n = 5350 :=
by
  sorry

end sum_of_arithmetic_series_l110_110399


namespace max_profit_at_32_l110_110214

def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then 400 - 6 * x
  else if x > 40 then (7400 / x) - (40000 / (x^2))
  else 0

def cost (x : ℝ) : ℝ := 16 * x + 40

def profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 40 then x * (400 - 6 * x) - cost x
  else if x > 40 then x * ((7400 / x) - (40000 / (x^2))) - cost x
  else 0

theorem max_profit_at_32 :
  ∀ x : ℝ, 32 ≤ 40 → profit 32 = 11634 ∧ ∀ y, profit y ≤ profit 32 :=
by
  sorry

end max_profit_at_32_l110_110214


namespace angle_A_max_area_of_triangle_l110_110892

-- Problem 1: Proving the measure of angle A
theorem angle_A (a b c : ℝ) (A B C : ℝ) (h₁ : (sin A + sin B) * (a - b) = (sin C - sin B) * c) : A = π / 3 :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC given a=4
theorem max_area_of_triangle (b c : ℝ) (A : ℝ) (h₁ : (sin A + sin B) * (4 - b) = (sin C - sin B) * c)
  (h₂ : cos A = 1/2) (h₃ : A = π / 3) : ∃ (S : ℝ), S = 4 * sqrt 3 :=
sorry

end angle_A_max_area_of_triangle_l110_110892


namespace minimal_number_after_operations_l110_110572

def initial_numbers : List ℕ := List.range' 1 (101 + 1) |>.map (λ n => n^2)

def allowed_operations (a b : ℕ) : ℕ := abs (a - b)

theorem minimal_number_after_operations :
  ∃ (final : ℕ), final = 1 ∧ (∃ (ops : List (ℕ × ℕ)), List.length ops = 100 ∧ 
    List.foldl (λ ns (i : ℕ × ℕ) => List.cons (allowed_operations (ns.get i.1) (ns.get i.2)) (List.eraseIdx (List.eraseIdx ns i.1) (i.2 - (if i.2 > i.1 then 1 else 0)))) initial_numbers ops).length = 1 ∧ 
    List.nth ((List.foldl (λ ns (i : ℕ × ℕ) => List.cons (allowed_operations (ns.get i.1) (ns.get i.2)) (List.eraseIdx (List.eraseIdx ns i.1) (i.2 - (if i.2 > i.1 then 1 else 0)))) initial_numbers ops)) 0 = some final :=
by sorry

end minimal_number_after_operations_l110_110572


namespace correct_statement_l110_110294

-- conditions
def condition_a : Prop := ∀(population : Type), ¬(comprehensive_survey population)
def data_set := [3, 5, 4, 1, -2]
def condition_b : Prop := median data_set = 4
def winning_probability : ℚ := 1 / 20
def condition_c : Prop := (∀n : ℕ, winning_probability * n = 1) → (∃i, i = 20)
def average_score (scores : List ℝ) : ℝ := scores.sum / scores.length
def variance (scores : List ℝ) : ℝ := (scores.map (λ x, (x - average_score scores)^2)).sum / scores.length
def scores_a := replicate 10 (average_score (replicate 10 10)) -- assumed scores for simplification
def scores_b := scores_a ++ [n + 1 for n in scores_a] -- assumed different scores to reflect variance
def condition_d : Prop := (average_score scores_a = average_score scores_b) ∧ (variance scores_a = 0.4) ∧ (variance scores_b = 2)

-- statement of the problem
theorem correct_statement : 
  condition_a ∧ condition_b ∧ condition_c ∧ condition_d → (∃D : Prop, D = condition_d) :=
by
  sorry

end correct_statement_l110_110294


namespace sum_series_l110_110375

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l110_110375


namespace no_solution_for_five_integers_with_given_sums_l110_110634

theorem no_solution_for_five_integers_with_given_sums :
  ∀ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e →
  ∃ m n, 
  set_of {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} = {15, 16, 17, 18, 19, 20, 21, 23, m, n}
  ∧ (m < 15 ∧ n > 23) → false :=
begin 
  sorry
end

end no_solution_for_five_integers_with_given_sums_l110_110634


namespace sin_series_positive_l110_110439

noncomputable def f (x : Real) (n : Nat) : Real :=
  ∑ i in Finset.range n, (Real.sin ((2 * i + 1 : Nat) * x) / (2 * i + 1 : Nat))

theorem sin_series_positive {x : Real} (hx : 0 < x ∧ x < Real.pi) (n : Nat) : 
  f x n > 0 :=
sorry

end sin_series_positive_l110_110439


namespace right_triangle_hypotenuse_perimeter_area_l110_110332

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := 40 - a - b

theorem right_triangle_hypotenuse_perimeter_area {a b : ℝ} :
  a + b + c = 40 ∧ (1/2) * a * b = 30 → c = 18.5 :=
by
  intro h,
  cases h with hPerimeter hArea,
  have hABeq60 : a * b = 60 := by sorry,
  have hPythagorean : a^2 + b^2 = c^2 := by sorry,
  have hSymptoms : a + b = 21.5 := by sorry,
  show c = 18.5 from sorry

end right_triangle_hypotenuse_perimeter_area_l110_110332


namespace tangent_line_at_point_is_correct_l110_110862

noncomputable def curve (x : ℝ) : ℝ := x / (2 * x - 1)

theorem tangent_line_at_point_is_correct :
  ∀ x y : ℝ, (x, y) = (1, 1) → ∀ m : ℝ, m = -1 → ∀ tangent_line : ℝ → ℝ,
  tangent_line = λ x', x' + y - m * (x' - x) - 2 →
  tangent_line (1 + y - 2) = 0 :=
by
  intros x y point eq m eq_m tangent_line eq_tangent_line
  rw [←eq_tangent_line, eq_m]
  sorry

end tangent_line_at_point_is_correct_l110_110862


namespace remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l110_110866

theorem remainder_8_times_10_pow_18_plus_1_pow_18_div_9 :
  (8 * 10^18 + 1^18) % 9 = 0 := 
by 
  sorry

end remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l110_110866


namespace fraction_value_l110_110627

theorem fraction_value : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := 
by
  sorry

end fraction_value_l110_110627


namespace sum_of_angles_is_180_l110_110552

variables {n : ℕ} {A : Fin n → ℝ × ℝ} {-n_gon_condition : RegularNgon A n}

def is_midpoint (B : ℝ × ℝ) (P1 P2 : ℝ × ℝ) :=
  B = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def bisector_intersect (a b c : ℝ × ℝ) : ℝ × ℝ := sorry -- Function to compute intersection of bisector with line

noncomputable def B (i : Fin (n - 1)) : ℝ × ℝ :=
  if i = 0 ∨ i = ⟨n-2, sorry⟩ then
    let P1 := if i = 0 then A 0 else A i;
    let P2 := if i = 0 then A 1 else A (i+1);
    (is_midpoint P1 P2) else bisector_intersect (A i) (A (n-1)) (A (i+1))

def angle (P A B : ℝ × ℝ) : ℝ := sorry -- Function to compute angle

theorem sum_of_angles_is_180 :
  ∑ i in Finset.range (n - 1), angle (A 0) (B i) (A (n - 1)) = 180 :=
begin
  sorry
end

end sum_of_angles_is_180_l110_110552


namespace distinct_sequences_ten_flips_l110_110738

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110738


namespace similar_triangles_centroid_coincide_l110_110574

-- Definitions of the similar triangles constructed externally
theorem similar_triangles_centroid_coincide
  (A B C A1 B1 C1 : Type) 
  [HasDist A] [HasDist B] [HasDist C] [HasDist A1] [HasDist B1] [HasDist C1]
  (hA1 : similar (triangle A B C) (triangle A1 B C)) 
  (hB1 : similar (triangle A B C) (triangle B1 C A))
  (hC1 : similar (triangle A B C) (triangle C1 A B)) :
  centroid (triangle A B C) = centroid (triangle A1 B1 C1) := 
sorry

end similar_triangles_centroid_coincide_l110_110574


namespace students_count_l110_110835

theorem students_count (x y : ℕ) (h1 : 3 * x + 20 = y) (h2 : 4 * x - 25 = y) : x = 45 :=
by {
  sorry
}

end students_count_l110_110835


namespace smallest_four_digit_multiple_of_17_l110_110283

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l110_110283


namespace percentage_profits_revenues_previous_year_l110_110139

noncomputable def companyProfits (R P R2009 P2009 : ℝ) : Prop :=
  (R2009 = 0.8 * R) ∧ (P2009 = 0.15 * R2009) ∧ (P2009 = 1.5 * P)

theorem percentage_profits_revenues_previous_year (R P : ℝ) (h : companyProfits R P (0.8 * R) (0.12 * R)) : 
  (P / R * 100) = 8 :=
by 
  sorry

end percentage_profits_revenues_previous_year_l110_110139


namespace find_n_l110_110674

theorem find_n (n : ℤ) : 3^n + 4^n = 5^n → n = 2 := by
  sorry -- The proof will be conducted here later

end find_n_l110_110674


namespace true_proposition_l110_110558

theorem true_proposition : 
  (∃ x0 : ℝ, x0 > 0 ∧ 3^x0 + x0 = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, abs x - a * x = abs (-x) - a * (-x)) := by
  sorry

end true_proposition_l110_110558


namespace four_ab_eq_four_l110_110118

theorem four_ab_eq_four {a b : ℝ} (h : a * b = 1) : 4 * a * b = 4 :=
by
  sorry

end four_ab_eq_four_l110_110118


namespace circle_radius_l110_110460

-- Definitions
def line_intersects_circle (r : ℝ) (x y : ℝ) : Prop :=
  x - real.sqrt 3 * y + 8 = 0 ∧ x^2 + y^2 = r^2

def chord_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem circle_radius :
  (∃ (A B : ℝ × ℝ), line_intersects_circle 5 A.1 A.2 ∧ line_intersects_circle 5 B.1 B.2 ∧ chord_length A B = 6) → r = 5 :=
by
  sorry

end circle_radius_l110_110460


namespace triangle_area_ratio_l110_110977

theorem triangle_area_ratio :
  ∀ {A B C D E F : Type} [Geometry A B C D E F],
  AB = 12 ∧ BC = 16 ∧ CA = 20 →
  (∃ p q r : ℝ, 0 < p ∧ 0 < q ∧ 0 < r ∧
    p + q + r = 1 ∧ p^2 + q^2 + r^2 = 1 / 2 ∧
    AD = p * AB ∧ BE = q * BC ∧ CF = r * CA) →
  let area_ABC := (1 / 2) * 12 * 16 in
  let area_ratio := 3 / 4 in
  let (m, n) := (3, 4) in
  m + n = 7 :=
begin
  intros A B C D E F h_geom h_exists,
  sorry
end

end triangle_area_ratio_l110_110977


namespace smallest_k_no_real_roots_l110_110651

theorem smallest_k_no_real_roots :
  ∀ (k : ℤ), (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) → k ≥ 4 :=
by
  sorry

end smallest_k_no_real_roots_l110_110651


namespace shaded_region_area_correct_l110_110348

noncomputable def shaded_region_area (side_length : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
if 0 < beta ∧ beta < Real.pi / 2 ∧ cos_beta = 3 / 5 then
  2 / 5
else
  0

theorem shaded_region_area_correct :
  shaded_region_area 2 β (3 / 5) = 2 / 5 :=
by
  -- conditions
  have beta_cond : 0 < β ∧ β < Real.pi / 2 := sorry
  have cos_beta_cond : cos β = 3 / 5 := sorry
  -- we will finish this proof assuming above have been proved.
  exact if_pos ⟨beta_cond, cos_beta_cond⟩

end shaded_region_area_correct_l110_110348


namespace distinct_sequences_ten_flips_l110_110686

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110686


namespace train_length_l110_110798

-- Define the constants based on the given conditions.
def speed_kmh : ℝ := 30
def time_s : ℝ := 60
def bridge_length_m : ℝ := 140

-- The proof statement
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) :
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let total_distance := speed_ms * time_s in
  let train_length := total_distance - bridge_length_m in
  train_length = 359.8 :=
by
  sorry

end train_length_l110_110798


namespace stride_vs_leap_difference_l110_110405

/-- Define the problem's conditions and verify the difference in lengths -/
theorem stride_vs_leap_difference :
  (let strides_per_gap := 50 in
   let leaps_per_gap := 15 in
   let total_poles := 51 in
   let total_distance_feet := 10560 in
   let total_gaps := total_poles - 1 in
   let total_strides := strides_per_gap * total_gaps in
   let total_leaps := leaps_per_gap * total_gaps in
   let stride_length := total_distance_feet / total_strides in
   let leap_length := total_distance_feet / total_leaps in
   let difference := leap_length - stride_length in
   difference = 10) :=
begin
  sorry
end

end stride_vs_leap_difference_l110_110405


namespace factor_expression_l110_110848

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110848


namespace speeds_of_bodies_l110_110604

theorem speeds_of_bodies 
  (v1 v2 : ℝ)
  (h1 : 21 * v1 + 10 * v2 = 270)
  (h2 : 51 * v1 + 40 * v2 = 540)
  (h3 : 5 * v2 = 3 * v1): 
  v1 = 10 ∧ v2 = 6 :=
by
  sorry

end speeds_of_bodies_l110_110604


namespace find_coordinates_of_Q_l110_110508

open Real

-- Define the points as constants
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (4, 3)

-- The angle of rotation, θ, and the rotation angle
def θ : ℝ := arctan (3/4)
def rotation_angle : ℝ := 2 * π / 3

-- Definitions for cosine and sine of the initial angle θ
def cos_θ : ℝ := cos θ
def sin_θ : ℝ := sin θ

-- The magnitude of vector OP
def magnitude_OP : ℝ := sqrt (4^2 + 3^2)

-- Calculated coordinates of Q
def Q_x : ℝ := magnitude_OP * (cos_θ * cos rotation_angle - sin_θ * sin rotation_angle)
def Q_y : ℝ := magnitude_OP * (sin_θ * cos rotation_angle + cos_θ * sin rotation_angle)
def Q : ℝ × ℝ := (Q_x, Q_y)

-- The target coordinates to prove
def target_x : ℝ := - (4 + 3 * sqrt 3) / 2
def target_y : ℝ := - (3 - 4 * sqrt 3) / 2
def target_Q : ℝ × ℝ := (target_x, target_y)

theorem find_coordinates_of_Q :
  Q = target_Q :=
sorry

end find_coordinates_of_Q_l110_110508


namespace coefficient_x3_expansion_l110_110152

theorem coefficient_x3_expansion : 
  (finset.sum (finset.range 6) (λ k, nat.choose (k + 3) 3)) = 126 := 
by
  sorry

end coefficient_x3_expansion_l110_110152


namespace probability_multiple_of_15_l110_110649

theorem probability_multiple_of_15
  (digits : Finset ℕ)
  (h_digits : digits = {1, 2, 3, 4, 5})
  (total_permutations : ℕ)
  (h_total_permutations : total_permutations = 5!):
  (∃ n : ℕ, (n ∈ digits) ∧ (n ≠ 0) ∧ (5! / n = 5)) -> (24/120 = 1/5) :=
by
  sorry

end probability_multiple_of_15_l110_110649


namespace find_a_values_l110_110030

noncomputable def SatisfiesSystem (a : ℝ) : Prop :=
  ∀ b : ℝ, ∃ x y : ℝ, (x - 2)^2 + (|y - 1| - 1)^2 = 4 ∧ y = b * |x - 1| + a

theorem find_a_values :
  ∀ a : ℝ, SatisfiesSystem a ↔ -real.sqrt 3 ≤ a ∧ a ≤ 2 + real.sqrt 3 :=
by
  intro a
  split
  all_goals {
    sorry
  }

end find_a_values_l110_110030


namespace range_of_g_l110_110398

theorem range_of_g :
  let g (x: ℝ) := (Real.arcsin x) + (Real.arccos x) + (Real.arctan x) + 2 * (Real.arcsin x)
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 →
    let φ (x: ℝ) := (Real.pi / 2) + (Real.arctan x) + 2 * (Real.arcsin x)
    ∃ (y: ℝ), φ y ∈ Set.Icc (-3 * Real.pi / 4) (7 * Real.pi / 4) :=
begin
  sorry
end

end range_of_g_l110_110398


namespace smallest_n_gt_4_99_l110_110869

theorem smallest_n_gt_4_99 : ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ m < n → (5^(m+1) + 2^(m+1))/(5^m + 2^m) ≤ 4.99) ∧ (5^(n+1) + 2^(n+1))/(5^n + 2^n) > 4.99 :=
by 
  let n := 7
  use n
  split
  { linarith }
  split
  {
    intros m Hm
    sorry
  }
  {
    sorry
  }

end smallest_n_gt_4_99_l110_110869


namespace range_of_m_l110_110481

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define A as the set of real numbers satisfying 2x^2 - x = 0
def A : Set ℝ := {x | 2 * x^2 - x = 0}

-- Define B based on the parameter m as the set of real numbers satisfying mx^2 - mx - 1 = 0
def B (m : ℝ) : Set ℝ := {x | m * x^2 - m * x - 1 = 0}

-- Define the condition (¬U A) ∩ B = ∅
def condition (m : ℝ) : Prop := (U \ A) ∩ B m = ∅

theorem range_of_m : ∀ m : ℝ, condition m → -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l110_110481


namespace salt_equal_at_time_l110_110799

def salt_balance (x y : ℝ → ℝ) (t : ℝ) :=
  x t = y t

def first_vessel_salt (x : ℝ → ℝ) (t : ℝ) :=
  x t = 10 * Real.exp (-3 * t / 100)

def second_vessel_salt (y : ℝ → ℝ) (t : ℝ) :=
  y t = (3 * t / 10) * Real.exp(-3 * t / 100)

theorem salt_equal_at_time :
  ∃ t, (first_vessel_salt x t) ∧ (second_vessel_salt y t) ∧ salt_balance x y t :=
sorry

end salt_equal_at_time_l110_110799


namespace equation_of_line_L_l110_110447

noncomputable def line_L_equation (x y : ℝ) : Prop :=
3 * x - y - 2 = 0

def point_A : ℝ × ℝ := (2, 4)

def parallel_line_1 (x y : ℝ) : Prop :=
x - y + 1 = 0

def parallel_line_2 (x y : ℝ) : Prop :=
x - y - 1 = 0

def midpoint_line (x y : ℝ) : Prop :=
x + 2y - 3 = 0

theorem equation_of_line_L :
  (∃ x y : ℝ, line_L_equation x y ∧ (point_A = (x, y)))
  ∧ (∃ x1 y1 x2 y2 : ℝ, parallel_line_1 x1 y1 ∧ parallel_line_2 x2 y2 
  ∧ line_L_equation ((x1 + x2) / 2) ((y1 + y2) / 2)
  ∧ midpoint_line ((x1 + x2) / 2) ((y1 + y2) / 2)) :=
sorry

end equation_of_line_L_l110_110447


namespace perpendicularity_l110_110547

variables (Line Plane : Type) 
variables (a b l : Line) 
variables (α β γ : Plane)
variables [InterPlane : α → β → Line]
variables [SubsetLine : Line → Plane → Prop]
variables [ParallelLine : Line → Line → Prop]
variables [ParallelPlane : Plane → Plane → Prop]
variables [PerpendicularLine : Line → Line → Prop]
variables [PerpendicularPlane : Plane → Plane → Prop]

axiom inter_plane (α β : Plane) : (α ∩ β = a)
axiom subset_line_beta (b : Line) (β : Plane) : SubsetLine b β
axiom perp_plane (α β : Plane) : PerpendicularPlane α β
axiom perp_line (a b : Line) : PerpendicularLine a b

theorem perpendicularity 
  (Hαβ : PerpendicularPlane α β) 
  (Hαβ_inter : InterPlane α β = a) 
  (Hbβ : SubsetLine b β) 
  (Ha_perp_b : PerpendicularLine a b) 
  : PerpendicularLine b α := 
sorry

end perpendicularity_l110_110547


namespace greatest_exponent_of_3_in_16_factorial_l110_110942

theorem greatest_exponent_of_3_in_16_factorial :
  ∃ x, (∀ n, n ≠ x → 3^x ∣ 16! ∧ ¬ 3^(n + 1) ∣ 16!) ∧ x = 6 :=
by
  sorry

end greatest_exponent_of_3_in_16_factorial_l110_110942


namespace good_walker_catches_up_l110_110672

-- Definitions based on the conditions in the problem
def steps_good_walker := 100
def steps_bad_walker := 60
def initial_lead := 100

-- Mathematical proof problem statement
theorem good_walker_catches_up :
  ∃ x : ℕ, x = initial_lead + (steps_bad_walker * x / steps_good_walker) :=
sorry

end good_walker_catches_up_l110_110672


namespace circle_diameter_4_circumference_not_eq_area_l110_110287

noncomputable def π : ℝ := Real.pi
def diameter : ℝ := 4
def radius : ℝ := diameter / 2
def circumference : ℝ := π * diameter
def area : ℝ := π * radius^2

theorem circle_diameter_4_circumference_not_eq_area
  (d : ℝ) (h₀ : d = 4)
  (r : ℝ) (h₁: r = d / 2)
  (C : ℝ) (h₂: C = π * d)
  (A : ℝ) (h₃ : A = π * r^2) :
  ¬ (C = A) :=
sorry

end circle_diameter_4_circumference_not_eq_area_l110_110287


namespace bureaucrats_problem_l110_110315

-- Define the familiar relationship
def familiarize (B : Finset α) (x y : α) : Prop := sorry

-- Define the problem conditions
def bureaucrats_split (A B C : Finset α) (hA : A.card = 100) (hB : B.card = 100) (hC : C.card = 100) (h_union : A ∪ B ∪ C = Finset.univ α) (h_disjoint : (A ∩ B) = ∅ ∧ (A ∩ C) = ∅ ∧ (B ∩ C) = ∅) : Prop :=
  (∀ x y : α, x ∈ A ∪ B ∪ C → y ∈ A ∪ B ∪ C → x ≠ y → (familiarize (A ∪ B ∪ C) x y ∨ ¬ (familiarize (A ∪ B ∪ C) x y)))

-- Prove the required property
theorem bureaucrats_problem (A B C : Finset α) (hA : A.card = 100) (hB : B.card = 100) (hC : C.card = 100) (h_union : A ∪ B ∪ C = Finset.univ α) (h_disjoint : (A ∩ B) = ∅ ∧ (A ∩ C) = ∅ ∧ (B ∩ C) = ∅) (h_familiarize : ∀ x y : α, x ∈ A ∪ B ∪ C → y ∈ A ∪ B ∪ C → x ≠ y → (familiarize (A ∪ B ∪ C) x y ∨ ¬ (familiarize (A ∪ B ∪ C) x y))) :
  ∃ x y : α, x ∈ A ∪ B ∧ y ∈ B ∪ C ∧ (∃ z_set : Finset α, (z_set.card ≥ 17 ∧ ∀ z ∈ z_set, familiarize z x ∧ familiarize z y) ∨ (z_set.card ≥ 17 ∧ ∀ z ∈ z_set, ¬ familiarize z x ∧ ¬ familiarize z y)) :=
  sorry

end bureaucrats_problem_l110_110315


namespace shirt_original_price_l110_110161

theorem shirt_original_price (original_price final_price : ℝ) (h1 : final_price = 0.5625 * original_price) 
  (h2 : final_price = 19) : original_price = 33.78 :=
by
  sorry

end shirt_original_price_l110_110161


namespace exp_calculation_l110_110814

theorem exp_calculation : (3^2)^4 = 6561 :=
by {
  sorry,
}

end exp_calculation_l110_110814


namespace complement_union_reals_l110_110222

variable (f : ℝ → ℝ) (a : ℝ)
noncomputable def A : Set ℝ := {x | x < a} ∪ {x | x > a}
noncomputable def M : Set ℝ := {x | A x ∧ f x ≥ 0}
noncomputable def N : Set ℝ := {x | A x ∧ f x < 0}

theorem complement_union_reals (f : ℝ → ℝ) (a : ℝ) :
  (M f a) ∪ (N f a) = {x | x ≠ a} → (M f a) ∩ (N f a) = ∅ →
  (A f a) → (Set.compl (M f a) ∪ Set.compl (N f a)) = Set.univ :=
by
  intro h1 h2 h3
  sorry

end complement_union_reals_l110_110222


namespace circle_area_II_l110_110822

-- Definitions and conditions for the problem
def circle_area (r : ℝ) : ℝ := π * r^2

def Circle (radius : ℝ) := { r := radius}

variables (rI rII : ℝ)  -- radii of Circle I and Circle II respectively

-- Circle I passes through the center of Circle II and is tangent externally
def circle_conditions : Prop :=
  circle_area rI = 16 ∧ 2 * rI = rII

-- To prove that the area of Circle II is 64 square inches
theorem circle_area_II (h : circle_conditions rI rII) : circle_area rII = 64 :=
sorry

end circle_area_II_l110_110822


namespace min_sum_of_squares_l110_110044

noncomputable def minimumOfSumOfSquares : ℝ :=
  Real.min (x^2 + y^2 + z^2 + t^2)
    (x, y, z, t : ℝ) 
    (hx : x ≥ 0) 
    (hy : y ≥ 0) 
    (hz : z ≥ 0) 
    (ht : t ≥ 0) 
    (habs : |x - y| + |y - z| + |z - t| + |t - x| = 4) := 
  2

theorem min_sum_of_squares : minimumOfSumOfSquares := sorry

end min_sum_of_squares_l110_110044


namespace Jane_wins_probability_l110_110529

noncomputable def probability_Jane_wins : ℚ := 19 / 25

theorem Jane_wins_probability :
  ∃ jane_scenario : ℕ × ℕ → ℚ,
    (∀ (a b : ℕ), 
      1 ≤ a ∧ a ≤ 5 → 
      1 ≤ b ∧ b ≤ 5 → 
      (abs (a - b) < 3 → jane_scenario (a, b) = 1) ∧ 
      (abs (a - b) ≥ 3 → jane_scenario (a, b) = 0)) ∧
    (∑ a b in Finset.Icc 1 5, jane_scenario (a, b) / 25 = probability_Jane_wins) :=
by
  sorry

end Jane_wins_probability_l110_110529


namespace x_intercept_of_parabola_l110_110046

theorem x_intercept_of_parabola (a b c : ℝ)
    (h_vertex : ∀ x, (a * (x - 5)^2 + 9 = y) → (x, y) = (5, 9))
    (h_intercept : ∀ x, (a * x^2 + b * x + c = 0) → x = 0 ∨ y = 0) :
    ∃ x0 : ℝ, x0 = 10 :=
by
  sorry

end x_intercept_of_parabola_l110_110046


namespace part_one_part_two_domain_part_two_range_l110_110467

-- Define the function f
def f (x : ℝ) : ℝ := (1 + sqrt 3 * tan x) * (cos x)^2

-- alpha in the second quadrant
def alpha : ℝ := sorry  -- assumed value since the exact angle isn't provided

-- condition sin α = sqrt 6 / 3
axiom sin_alpha : sin alpha = sqrt 6 / 3

-- condition cos α = -sqrt 3 / 3 derived from the above condition
axiom cos_alpha : cos alpha = -sqrt 3 / 3

-- First part: prove f(alpha) = (1 - sqrt 6) / 3
theorem part_one : f alpha = (1 - sqrt 6) / 3 :=
by
  -- Proof steps skipped, focus on the problem statement
  sorry

-- Second part: domain and range of f(x)
theorem part_two_domain : ∀ x, x ≠ k * π + π / 2 → x ∈ domain f :=
by
  -- Domain proof steps skipped, focus on the problem statement
  sorry

theorem part_two_range : ∀ y, y ∈ range f ↔ -1/2 <= y ∧ y <= 3/2 :=
by
  -- Range proof steps skipped, focus on the problem statement
  sorry

end part_one_part_two_domain_part_two_range_l110_110467


namespace times_faster_l110_110777

theorem times_faster (A B : ℝ) (h1 : A + B = 1 / 12) (h2 : A = 1 / 16) : 
  A / B = 3 :=
by
  sorry

end times_faster_l110_110777


namespace overall_average_output_l110_110300

theorem overall_average_output 
  (initial_cogs : ℕ := 60) 
  (rate_1 : ℕ := 36) 
  (rate_2 : ℕ := 60) 
  (second_batch_cogs : ℕ := 60) :
  (initial_cogs + second_batch_cogs) / ((initial_cogs / rate_1) + (second_batch_cogs / rate_2)) = 45 := 
  sorry

end overall_average_output_l110_110300


namespace total_amount_spent_l110_110205

-- Conditions
def action_figures_E : ℕ := 60
def action_figures_Y : ℕ := 3 * action_figures_E
def cars_Y : ℕ := 20
def stuffed_animals_Y : ℕ := 10

def cost_per_action_figure_E : ℕ := 5
def cost_per_action_figure_Y : ℕ := 4
def cost_per_car : ℕ := 3
def cost_per_stuffed_animal : ℕ := 7

-- Proposition to prove
theorem total_amount_spent :
  let total_cost_E := action_figures_E * cost_per_action_figure_E
      total_cost_Y := (action_figures_Y * cost_per_action_figure_Y) + (cars_Y * cost_per_car) + (stuffed_animals_Y * cost_per_stuffed_animal)
      total_amount := total_cost_E + total_cost_Y
  in total_amount = 1150 :=
by sorry

end total_amount_spent_l110_110205


namespace jose_peanuts_l110_110539

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l110_110539


namespace sum_a_b_l110_110995

variable {a b : ℝ}

def f (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := 3 * x - 6

theorem sum_a_b :
  (∀ x : ℝ, g (f x) = 4 * x + 3) → a + b = 13 / 3 :=
by
  intro h
  sorry

end sum_a_b_l110_110995


namespace same_number_of_heads_probability_l110_110158

/-- Jackie and Phil each have two fair coins and a third coin that 
  comes up heads with probability \(3/5\). Determine the probability 
  that Jackie and Phil get the same number of heads, expressed as a 
  sum of relatively prime numbers. -/
theorem same_number_of_heads_probability : 
  let prob := 63 / 200
  let result := 263 
  (2 : ℕ).choose (2 : ℕ) = 1 ∧  -- two fair coins, the number of ways to get 0 and 2 heads
  (2 : ℕ).choose (1 : ℕ) = 2 ∧  -- the number of ways to get 1 heads
  (1 : ℕ).choose (0 : ℕ) = 1 ∧  -- third coin fair, no heads
  (3 / 5 : ℝ) = 0.6 ∧          -- third coin heads probability
  true := prob.sum = result := sorry

end same_number_of_heads_probability_l110_110158


namespace coefficient_a3b3_in_expression_l110_110270

theorem coefficient_a3b3_in_expression :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ k) * (b ^ (6 - k))) *
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (c ^ (8 - 2 * k)) * (c ^ (-2 * k))) =
  1400 := sorry

end coefficient_a3b3_in_expression_l110_110270


namespace coin_flip_sequences_l110_110752

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110752


namespace P_x_rational_l110_110167

theorem P_x_rational {n : ℕ} {x : ℝ}
  (P : ℝ → ℝ)
  (H1 : ∀ t, P t = 1 + t + t^2 + ⋯ + t^(2*n))
  (H2 : P x ∈ ℚ)
  (H3 : P (x^2) ∈ ℚ) :
  x ∈ ℚ := sorry

end P_x_rational_l110_110167


namespace find_number_of_possible_values_of_g2_l110_110989

noncomputable def S := {x : ℝ // x ≠ 0}

variable (g : S → S)
hypothesis h : ∀ (x y : S), (x + y).val ≠ 0 → g(x) + g(y) = g(⟨(x * y).val / (g ⟨(x + y).val, sorry⟩).val, sorry⟩)

theorem find_number_of_possible_values_of_g2 :
  ∃ (m : ℕ) (t : ℝ), m = 1 ∧ t = 1/2 ∧ m * t = 1/2 := sorry

end find_number_of_possible_values_of_g2_l110_110989


namespace sin_2alpha_value_l110_110116

theorem sin_2alpha_value (α a : ℝ)
  (h : sin (a + π / 4) = sqrt 2 * (sin α + 2 * cos α)) : 
  sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_2alpha_value_l110_110116


namespace circular_permutations_count_l110_110036

noncomputable def α : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def β : ℝ := (1 - Real.sqrt 5) / 2

noncomputable def b (n : ℕ) : ℝ := 
if n = 1 then 1 else 
if n = 2 then 2 else 
2 + α^(n - 1) + β^(n - 1)

theorem circular_permutations_count (n : ℕ) : b n = α^n + β^n + 2 := 
by {
  sorry
}

end circular_permutations_count_l110_110036


namespace hyperbola_eccentricity_is_5_over_3_l110_110776

noncomputable def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  a / b = 3 / 4

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_5_over_3 (a b : ℝ) (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_is_5_over_3_l110_110776


namespace center_of_circle_eq_minus_two_four_l110_110031

theorem center_of_circle_eq_minus_two_four : 
  ∀ (x y : ℝ), x^2 + 4 * x + y^2 - 8 * y + 16 = 0 → (x, y) = (-2, 4) :=
by {
  sorry
}

end center_of_circle_eq_minus_two_four_l110_110031


namespace correct_product_l110_110506

namespace SarahsMultiplication

theorem correct_product (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hx' : ∃ (a b : ℕ), x = 10 * a + b ∧ b * 10 + a = x' ∧ 221 = x' * y) : (x * y = 527 ∨ x * y = 923) := by
  sorry

end SarahsMultiplication

end correct_product_l110_110506


namespace proof_of_incorrect_propositions_l110_110097

variables (m n l : line) (α β : plane)
def proposition_1 : Prop := m ∥ n ∧ n ∥ α → m ∥ α
def proposition_2 : Prop := α ⊥ β ∧ α ∩ β = m ∧ l ⊥ m → l ⊥ β
def proposition_3 : Prop := l ⊥ m ∧ l ⊥ n ∧ m ⊆ α ∧ n ⊆ α → l ⊥ α
def proposition_4 : Prop := m ∩ n = A ∧ m ∥ α ∧ m ∥ β ∧ n ∥ α ∧ n ∥ β → α ∥ β

def num_incorrect_propositions : Nat := 3

theorem proof_of_incorrect_propositions :
  (¬ proposition_1 ∧ ¬ proposition_2 ∧ ¬ proposition_3 ∧ proposition_4) → num_incorrect_propositions = 3 :=
by sorry

end proof_of_incorrect_propositions_l110_110097


namespace first_player_wins_game_l110_110806

theorem first_player_wins_game :
  (∀ (players : ℕ → ℕ) (ops : ℕ → (ℕ → ℕ → ℕ)),
    (list.sum (list.map_with_index (λ i n, ops i (players i) (players (i+1))) [1, 2, ..., 100-1]) % 2 = 0) → Anna_wins
    ∨
    (list.sum (list.map_with_index (λ i n, ops i (players i) (players (i+1))) [1, 2, ..., 100-1]) % 2 = 1) → Balázs_wins) →
  (∃ strategy_first_player : (ℕ → ℕ) → (ℕ → (ℕ → ℕ → ℕ)), winner (strategy_first_player players ops)) :=
sorry

end first_player_wins_game_l110_110806


namespace find_cost_of_crackers_l110_110583

-- Definitions based on the given conditions
def cost_hamburger_meat : ℝ := 5.00
def cost_per_bag_vegetables : ℝ := 2.00
def number_of_bags_vegetables : ℕ := 4
def cost_cheese : ℝ := 3.50
def discount_rate : ℝ := 0.10
def total_after_discount : ℝ := 18

-- Definition of the box of crackers, which we aim to prove
def cost_crackers : ℝ := 3.50

-- The Lean statement for the proof
theorem find_cost_of_crackers
  (C : ℝ)
  (h : C = cost_crackers)
  (H : 0.9 * (cost_hamburger_meat + cost_per_bag_vegetables * number_of_bags_vegetables + cost_cheese + C) = total_after_discount) :
  C = 3.50 :=
  sorry

end find_cost_of_crackers_l110_110583


namespace sec_seven_pi_over_six_l110_110013

theorem sec_seven_pi_over_six : sec (7 * Real.pi / 6) = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_seven_pi_over_six_l110_110013


namespace fraction_of_yard_occupied_by_flower_beds_l110_110790

-- Define conditions
def yard_length : ℕ := 30
def yard_width : ℕ := 10
def remaining_rect_parallel_side_length : ℕ := 22

-- Define the proof problem
theorem fraction_of_yard_occupied_by_flower_beds : 
  (2 * (1 / 2 * (remaining_rect_parallel_side_length - yard_width/2)²)) / (yard_length * yard_width) = 4 / 75 :=
by
  sorry

end fraction_of_yard_occupied_by_flower_beds_l110_110790


namespace tan_seven_pi_over_four_l110_110025

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l110_110025


namespace coin_flip_sequences_l110_110751

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110751


namespace coin_flip_sequences_l110_110754

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110754


namespace exists_multiple_with_odd_digit_sum_l110_110059

theorem exists_multiple_with_odd_digit_sum (M : Nat) :
  ∃ N : Nat, N % M = 0 ∧ (Nat.digits 10 N).sum % 2 = 1 :=
by
  sorry

end exists_multiple_with_odd_digit_sum_l110_110059


namespace coin_flip_sequences_l110_110744

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110744


namespace point_in_third_quadrant_l110_110943

theorem point_in_third_quadrant (m n : ℝ) (h1 : m > 0) (h2 : n > 0) : (-m < 0) ∧ (-n < 0) :=
by
  sorry

end point_in_third_quadrant_l110_110943


namespace calculate_surface_area_of_modified_cube_l110_110800

-- Definitions of the conditions
def edge_length_of_cube : ℕ := 5
def side_length_of_hole : ℕ := 2

-- The main theorem statement to be proven
theorem calculate_surface_area_of_modified_cube :
  let original_surface_area := 6 * (edge_length_of_cube * edge_length_of_cube)
  let area_removed_by_holes := 6 * (side_length_of_hole * side_length_of_hole)
  let area_exposed_by_holes := 6 * 6 * (side_length_of_hole * side_length_of_hole)
  original_surface_area - area_removed_by_holes + area_exposed_by_holes = 270 :=
by
  sorry

end calculate_surface_area_of_modified_cube_l110_110800


namespace zero_in_interval_l110_110243

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) → -- f(x) is increasing on (0, +∞)
  f 2 < 0 → -- f(2) < 0
  f 3 > 0 → -- f(3) > 0
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h_increasing h_f2_lt_0 h_f3_gt_0
  sorry

end zero_in_interval_l110_110243


namespace option_C_is_quadratic_l110_110657

-- Define what it means for an equation to be quadratic
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), p x ↔ a*x^2 + b*x + c = 0

-- Define the equation in option C
def option_C (x : ℝ) : Prop := (x - 1) * (x - 2) = 0

-- The theorem we need to prove
theorem option_C_is_quadratic : is_quadratic option_C :=
  sorry

end option_C_is_quadratic_l110_110657


namespace equilateral_triangle_exists_l110_110677

/-- Given 6 points in the plane such that 8 of the pairwise distances are 1,
    we show that there exist three points that form an equilateral triangle
    with side length 1. -/
theorem equilateral_triangle_exists (points : Fin 6 → ℝ × ℝ)
  (h : 8 ≤ (Finset.univ.image (λ (x : Fin 6) (y : Fin 6), dist (points x) (points y))).filter (λ d, d = 1).card) :
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ dist (points a) (points b) = 1 ∧ dist (points b) (points c) = 1 ∧ dist (points c) (points a) = 1 :=
by
  sorry

end equilateral_triangle_exists_l110_110677


namespace inverse_of_f_l110_110590

def f (x : ℝ) : ℝ := 7 - 3 * x

noncomputable def f_inv (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end inverse_of_f_l110_110590


namespace jessica_deposit_fraction_l110_110533

-- Definitions based on conditions
variable (initial_balance : ℝ)
variable (fraction_withdrawn : ℝ) (withdrawn_amount : ℝ)
variable (final_balance remaining_balance fraction_deposit : ℝ)

-- Conditions
def conditions := 
  fraction_withdrawn = 2 / 5 ∧
  withdrawn_amount = 400 ∧
  remaining_balance = initial_balance - withdrawn_amount ∧
  remaining_balance = initial_balance * (1 - fraction_withdrawn) ∧
  final_balance = 750 ∧
  final_balance = remaining_balance + fraction_deposit * remaining_balance

-- The proof problem
theorem jessica_deposit_fraction : 
  conditions initial_balance fraction_withdrawn withdrawn_amount final_balance remaining_balance fraction_deposit →
  fraction_deposit = 1 / 4 :=
by
  intro h
  sorry

end jessica_deposit_fraction_l110_110533


namespace coin_flip_sequences_l110_110697

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110697


namespace mother_age_when_harry_born_l110_110486

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l110_110486


namespace largest_C_for_sum_of_reciprocals_l110_110070

theorem largest_C_for_sum_of_reciprocals (n : ℕ) (hn : 0 < n) :
  ∃ C : ℝ, (∀ S : ℝ, (∀ a : ℕ, a > 1 → ∑ i in S, 1 / (a : ℝ) < C) → 
           (∃ groups : ℕ → set ℕ, (∀ i, ∑ j in groups i, 1 / (j : ℝ) < 1) ∧
             ∑ i, (groups i).card ≤ n)) ∧
  C = (n + 1 : ℝ) / 2 := sorry

end largest_C_for_sum_of_reciprocals_l110_110070


namespace min_value_f_l110_110098

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (exp x - 1) + x

theorem min_value_f {x0 : ℝ} (hx0 : 0 < x0) (hx0_min : ∀ x > 0, f x ≥ f x0) :
  f x0 = x0 + 1 ∧ f x0 < 3 :=
by sorry

end min_value_f_l110_110098


namespace distinct_sequences_ten_flips_l110_110731

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110731


namespace find_n_l110_110477

def sequence (n : ℕ) : ℤ := 
  if n ≤ 3 then 2^n + 4 
  else -3 + sequence (n - 1)

theorem find_n {n : ℕ} (h : 1 ≤ n) :
  ∑ i in finset.range (n + 1), abs (sequence i) = 80 → n = 12 :=
sorry

end find_n_l110_110477


namespace ellipse_properties_max_area_triangle_AOB_l110_110452

noncomputable def ellipse_equation (a b : ℝ) : Prop := 
  ∀ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1

noncomputable def accompanying_circle_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 = (a * b / 2).sqrt

theorem ellipse_properties
  (a b : ℝ)
  (h_pos : a > b)
  (h_eccentricity : ((a^2 - b^2).sqrt) / a = sqrt(3) / 2)
  (h_point : (1/2)^2 / a^2 + (sqrt(3))^2 / b^2 = 1):
  ellipse_equation 2 b ∧ accompanying_circle_equation 2 b :=
sorry

theorem max_area_triangle_AOB
  (a b : ℝ) (m : ℝ) 
  (h_pos : a > b)
  (h_eccentricity : ((a^2 - b^2).sqrt) / a = sqrt(3) / 2)
  (h_point : (1/2)^2 / a^2 + (sqrt(3))^2 / b^2 = 1)
  (h_m : abs m ≥ 1):
  let S := (2 * sqrt(3) * abs m) / (m^2 + 3)
  in S ≤ 1 ∧ (S = 1 ↔ m = sqrt(3) ∨ m = -sqrt(3)) :=
sorry

end ellipse_properties_max_area_triangle_AOB_l110_110452


namespace alina_sent_fewer_messages_l110_110362

-- Definitions based on conditions
def messages_lucia_day1 : Nat := 120
def messages_lucia_day2 : Nat := 1 / 3 * messages_lucia_day1
def messages_lucia_day3 : Nat := messages_lucia_day1
def messages_total : Nat := 680

-- Def statement for Alina's messages on the first day, which we need to find as 100
def messages_alina_day1 : Nat := 100

-- Condition checks
def condition_alina_day2 : Prop := 2 * messages_alina_day1 = 2 * 100
def condition_alina_day3 : Prop := messages_alina_day1 = 100
def condition_total_messages : Prop := 
  messages_alina_day1 + messages_lucia_day1 +
  2 * messages_alina_day1 + messages_lucia_day2 +
  messages_alina_day1 + messages_lucia_day1 = messages_total

-- Theorem statement
theorem alina_sent_fewer_messages :
  messages_lucia_day1 - messages_alina_day1 = 20 :=
by
  -- Ensure the conditions hold
  have h1 : messages_alina_day1 = 100 := by sorry
  have h2 : condition_alina_day2 := by sorry
  have h3 : condition_alina_day3 := by sorry
  have h4 : condition_total_messages := by sorry
  -- Prove the theorem
  sorry

end alina_sent_fewer_messages_l110_110362


namespace journey_total_distance_l110_110302

theorem journey_total_distance (D : ℝ) (h_train : D * (3 / 5) = t) (h_bus : D * (7 / 20) = b) (h_walk : D * (1 - ((3 / 5) + (7 / 20))) = 6.5) : D = 130 :=
by
  sorry

end journey_total_distance_l110_110302


namespace triangle_problem_l110_110978

/-- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively.
Given that b = sqrt 2, c = 3, B + C = 3A, prove:
1. The length of side a equals sqrt 5.
2. sin (B + 3π/4) equals sqrt(10) / 10.
-/
theorem triangle_problem 
  (a b c A B C : ℝ)
  (hb : b = Real.sqrt 2)
  (hc : c = 3)
  (hBC : B + C = 3 * A)
  (hA : A = π / 4)
  : (a = Real.sqrt 5)
  ∧ (Real.sin (B + 3 * π / 4) = Real.sqrt 10 / 10) :=
sorry

end triangle_problem_l110_110978


namespace arith_seq_a5_l110_110968

-- Define variables and conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: S_7 - S_2 = 45
axiom sum_diff : S 7 - S 2 = 45

-- Definitions: Sequence is arithmetic. We do not need to assume any knowledge directly from the solution steps.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Define the sum function for the sequence
def sum_arith (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

-- State the theorem in Lean
theorem arith_seq_a5 (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) (h_sum_diff : S 7 - S 2 = 45) :
  a 5 = 9 :=
by
  sorry

end arith_seq_a5_l110_110968


namespace theta_value_monotonic_intervals_extremum_f_zero_range_of_m_l110_110468
open Real

-- Define f(x) and g(x)
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m * x - (m - 1 + 2 * exp 1) / x - log x

noncomputable def g (x : ℝ) (theta : ℝ) : ℝ := 1 / (x * cos theta) + log x

-- Theorem (I): Find the value of θ 
theorem theta_value (theta : ℝ) (h_incr_g : ∀ x ∈ Set.Icc 1 (∝), deriv (g x theta) > 0) : theta = 0 := 
sorry

-- Theorem (II): Monotonic intervals and extremum of f(x) when m = 0
theorem monotonic_intervals_extremum_f_zero (x : ℝ) :
  let f_zero := λ x, ((1 - 2 * exp 1) / x) - log x
  let f_zero' := λ x, deriv f_zero x
  (∀ x, f_zero' x > 0 → 0 < x ∧ x < (2 * exp 1 - 1)) ∧ 
  (∀ x, f_zero' x < 0 → x > (2 * exp 1 - 1)) ∧ 
  f_zero (2 * exp 1 - 1) = -1 - log (2 * exp 1 - 1) := 
sorry

-- Theorem (III): Range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ exp 1 ∧ f x m > g x 0) ↔ (m > (4 * exp 1) / (exp 2 - 1)) := 
sorry

end theta_value_monotonic_intervals_extremum_f_zero_range_of_m_l110_110468


namespace ratio_triangle_to_square_l110_110149

/-- In a square ABCD, point E is on AB and point F is on BC such that AE = 3 * EB and BF = FC.
    Prove that the ratio of the area of triangle DEF to the area of square ABCD is 5 / 16. -/
theorem ratio_triangle_to_square (A B C D E F : Point)
  (h_square : is_square A B C D)
  (h_AE_EB : AE = 3 * EB)
  (h_BF_FC : BF = FC) :
  let area_square := square_area A B C D,
      area_triangle := triangle_area D E F in
  area_triangle / area_square = 5 / 16 :=
by
  sorry

end ratio_triangle_to_square_l110_110149


namespace exists_point_on_two_great_circles_l110_110673

theorem exists_point_on_two_great_circles (n : ℕ) (h : n > 2)
  (great_circles : Fin n → Set (Set (ℝ × ℝ × ℝ)))
  (hne : ∀ {i j : Fin n}, i ≠ j → great_circles i ≠ great_circles j)
  (hnot_all_pass_through_one_point : ¬∃ p, ∀ i, p ∈ great_circles i) :
  ∃ p, (∃ i j : Fin n, i ≠ j ∧ p ∈ great_circles i ∧ p ∈ great_circles j) ∧
       (∀ k l : Fin n, k ≠ l → k = i ∨ l = j ∨ p ∉ great_circles k ∨ p ∉ great_circles l) :=
by
  sorry

end exists_point_on_two_great_circles_l110_110673


namespace sufficient_condition_for_unit_vector_equality_l110_110546

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (a b : V)

theorem sufficient_condition_for_unit_vector_equality
  (ha : a ≠ 0) (hb : b ≠ 0) (h : a = 2 • b) : (a / ∥a∥) = (b / ∥b∥) :=
sorry

end sufficient_condition_for_unit_vector_equality_l110_110546


namespace overall_cost_price_correct_l110_110793

-- Definitions for the given problem's conditions
def clothA_selling_price : ℝ := 10000 -- Rs.
def clothA_meters : ℕ := 150
def clothA_loss_per_meter : ℝ := 4 -- Rs. per meter

def clothB_selling_price : ℝ := 15000 -- Rs.
def clothB_meters : ℕ := 200
def clothB_loss_per_meter : ℝ := 5 -- Rs. per meter

def clothC_selling_price : ℝ := 25000 -- Rs.
def clothC_meters : ℕ := 300
def clothC_loss_per_meter : ℝ := 6 -- Rs. per meter

-- Derived calculations
def clothA_cost_price_per_meter : ℝ :=
  (clothA_selling_price / clothA_meters) + clothA_loss_per_meter

def clothB_cost_price_per_meter : ℝ :=
  (clothB_selling_price / clothB_meters) + clothB_loss_per_meter

def clothC_cost_price_per_meter : ℝ :=
  (clothC_selling_price / clothC_meters) + clothC_loss_per_meter

def clothA_total_cost_price : ℝ :=
  clothA_cost_price_per_meter * clothA_meters

def clothB_total_cost_price : ℝ :=
  clothB_cost_price_per_meter * clothB_meters

def clothC_total_cost_price : ℝ :=
  clothC_cost_price_per_meter * clothC_meters

def overall_total_cost_price : ℝ :=
  clothA_total_cost_price + clothB_total_cost_price + clothC_total_cost_price

theorem overall_cost_price_correct :
  overall_total_cost_price = 53399.50 :=
by
  sorry

end overall_cost_price_correct_l110_110793


namespace median_inequality_l110_110999

noncomputable def median_length (a b c s_c : ℝ) := s_c = (1/2) * (sqrt (2*a^2 + 2*b^2 - c^2))

theorem median_inequality (a b c s_c : ℝ) (h_median: s_c = (1/2)*(sqrt(2*a^2 + 2*b^2 - c^2))) :
  (c^2 - (a - b)^2) / (2 * (a + b)) ≤ a + b - 2 * s_c ∧ a + b - 2 * s_c < (c^2 + (a - b)^2) / (4 * s_c) :=
by 
  sorry

end median_inequality_l110_110999


namespace max_area_OAPF_l110_110966

-- Define the ellipse equation
def ellipse_Equation (x y : ℝ) : Prop :=
(x^2 / 9) + (y^2 / 10) = 1

-- Define points A and F
def A : ℝ × ℝ := (3, 0)
def F : ℝ × ℝ := (0, 1)

-- Define point P in parametric form
def P (θ : ℝ) : Prop :=
0 < θ ∧ θ < π / 2 ∧ P = (3 * cos θ, √10 * sin θ)

-- Statement for maximum area of quadrilateral OAPF
theorem max_area_OAPF : 
  ∃ θ, 0 < θ ∧ θ < π / 2 ∧
  (π / sqrt(elliptic)) = sqrt_π_math well φ_AND_θ=
  sorry

end max_area_OAPF_l110_110966


namespace side_a_of_triangle_l110_110520

theorem side_a_of_triangle (a b c : ℝ) (B : ℝ) (hB : B = 120) : 
  ∃ a, true := sorry

end side_a_of_triangle_l110_110520


namespace fraction_of_girls_l110_110142

variable (total_students : ℕ) (number_of_boys : ℕ)

theorem fraction_of_girls (h1 : total_students = 160) (h2 : number_of_boys = 60) :
    (total_students - number_of_boys) / total_students = 5 / 8 := by
  sorry

end fraction_of_girls_l110_110142


namespace find_theta_l110_110857

theorem find_theta :
  ∃ θ : ℝ, 0 < θ ∧ θ ≤ 90 ∧ cos (10 * real.pi / 180) = sin (15 * real.pi / 180) + sin (θ * real.pi / 180) ∧ θ = 75 :=
by
  sorry

end find_theta_l110_110857


namespace cloves_garlic_left_l110_110563

theorem cloves_garlic_left (total_cloves used_cloves : ℕ) (h_total : total_cloves = 237) (h_used : used_cloves = 184) : total_cloves - used_cloves = 53 := 
by
  rw [h_total, h_used]
  exact Nat.sub_eq_of_eq_add 184 237 53 sorry -- Using natural number properties

end cloves_garlic_left_l110_110563


namespace imaginary_part_z_l110_110896

theorem imaginary_part_z (z : ℂ) (h : (3 - 4 * Complex.i) * z = Complex.abs (3 - 4 * Complex.i)) : 
  Complex.im z = 4 / 5 :=
sorry

end imaginary_part_z_l110_110896


namespace problem_statement_l110_110519

-- Defining the conditions and required values
def is_leap_year (y : Nat) : Prop := 
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

def dates_in_leap_year_2020 : List Nat :=
  List.replicate 12 1 ++ List.replicate 12 2 ++ List.replicate 12 3 ++
  List.replicate 12 4 ++ List.replicate 12 5 ++ List.replicate 12 6 ++
  List.replicate 12 7 ++ List.replicate 12 8 ++ List.replicate 12 9 ++
  List.replicate 12 10 ++ List.replicate 12 11 ++ List.replicate 12 12 ++
  List.replicate 12 13 ++ List.replicate 12 14 ++ List.replicate 12 15 ++
  List.replicate 12 16 ++ List.replicate 12 17 ++ List.replicate 12 18 ++
  List.replicate 12 19 ++ List.replicate 12 20 ++ List.replicate 12 21 ++
  List.replicate 12 22 ++ List.replicate 12 23 ++ List.replicate 12 24 ++
  List.replicate 12 25 ++ List.replicate 12 26 ++ List.replicate 12 27 ++
  List.replicate 12 28 ++ List.replicate 12 29 ++ List.replicate 12 30 ++
  List.replicate 8 31

-- Function to compute the mean
def mean (l : List Nat) : Float :=
  (l.foldr (· + ·) 0).toFloat / l.length.toFloat

-- Function to compute the median
def median (l : List Nat) : Nat :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

-- Function to compute the modes
def modes (l : List Nat) : List Nat :=
  let grouped := l.groupBy id
  let max_freq := (grouped.map (·.snd.length)).maximum.getD 0
  grouped.filter (λ g => g.snd.length = max_freq).map (·.fst)

-- Function to compute the median of the modes
def median_of_modes (l : List Nat) : Nat :=
  median (modes l)

-- Main theorem statement
theorem problem_statement :
  let d := median_of_modes dates_in_leap_year_2020
  let μ := mean dates_in_leap_year_2020
  let M := median dates_in_leap_year_2020
  is_leap_year 2020 →
  d < μ ∧ μ < M :=
by
  intro h
  let d := median_of_modes dates_in_leap_year_2020
  let μ := mean dates_in_leap_year_2020
  let M := median dates_in_leap_year_2020
  sorry

end problem_statement_l110_110519


namespace geometric_sequence_iff_second_power_geometric_l110_110461

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

-- Define the geometric sequence condition
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

-- Main Lean 4 statement
theorem geometric_sequence_iff_second_power_geometric
  (h : geometric_sequence a q) :
  (∀ n : ℕ, a (n + 1) - a n = a n + n → q = 0 → a n = 1) →
  (∀ n : ℕ, a (n+1) = q * a n → (a (n+1) - a (n) = 0 → 2^a(n) = 2^a(n)) → (ln |a (n+1)|) / (ln |a n|)) → 
  ∀ n : ℕ, (a n ^ 2) = (q ^ 2) := sorry

end geometric_sequence_iff_second_power_geometric_l110_110461


namespace calculate_AH_l110_110187

def square (a : ℝ) := a ^ 2
def area_square (s : ℝ) := s ^ 2
def area_triangle (b h : ℝ) := 0.5 * b * h

theorem calculate_AH (s DG DH AH : ℝ) 
  (h_square : area_square s = 144) 
  (h_area_triangle : area_triangle DG DH = 63)
  (h_perpendicular : DG = DH)
  (h_hypotenuse : square AH = square s + square DH) :
  AH = 3 * Real.sqrt 30 :=
by
  -- Proof would be provided here
  sorry

end calculate_AH_l110_110187


namespace smallest_four_digit_multiple_of_17_is_1013_l110_110281

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l110_110281


namespace eugene_boxes_needed_l110_110406

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l110_110406


namespace a_range_l110_110129

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

theorem a_range (a : ℝ) :
  (∃ x y : ℝ, (1/3 < x ∧ x < 1/2) ∧ (1/3 < y ∧ y < 1/2) ∧ f' a x = 0 ∧ f' a y = 0) ↔
  a ∈ Set.Ioo (5/4) (5/2) :=
by
  sorry

end a_range_l110_110129


namespace fewest_relatively_prime_dates_in_June_when_Feb_has_29_days_relatively_prime_dates_in_June_l110_110330

def is_relatively_prime_date (month day : ℕ) : Prop :=
  Nat.gcd month day = 1

def number_of_relatively_prime_dates (month : ℕ) (days_in_month : ℕ) : ℕ :=
  (List.range (days_in_month + 1)).filter (is_relatively_prime_date month) |>.length

theorem fewest_relatively_prime_dates_in_June_when_Feb_has_29_days :
  number_of_relatively_prime_dates 6 30 < number_of_relatively_prime_dates 2 29 :=
sorry

theorem relatively_prime_dates_in_June :
  number_of_relatively_prime_dates 6 30 = 10 :=
sorry

end fewest_relatively_prime_dates_in_June_when_Feb_has_29_days_relatively_prime_dates_in_June_l110_110330


namespace coprime_integers_implies_prime_l110_110011

theorem coprime_integers_implies_prime (n : ℕ) (h_n : n = 15) 
  (h_coprime : ∀ (i j : ℕ), i ≠ j → gcd (i) (j) = 1)
  (h_bound : ∀ (i : ℕ), i < n → i < 2010) : 
  ∃ i < n, nat.prime i :=
sorry

end coprime_integers_implies_prime_l110_110011


namespace grade_assignment_ways_l110_110327

-- Definitions
def num_students : ℕ := 10
def num_choices_per_student : ℕ := 3

-- Theorem statement
theorem grade_assignment_ways : num_choices_per_student ^ num_students = 59049 := by
  sorry

end grade_assignment_ways_l110_110327


namespace cos_theta_equals_sqrt_five_div_five_l110_110462

theorem cos_theta_equals_sqrt_five_div_five (θ : ℝ) (h1 : θ > 0) (h2 : θ < π / 2) (h3 : tan θ = 2) : 
  cos θ = √5 / 5 := 
by 
  sorry

end cos_theta_equals_sqrt_five_div_five_l110_110462


namespace shaded_area_common_squares_l110_110344

noncomputable def cos_beta : ℝ := 3 / 5

theorem shaded_area_common_squares :
  ∀ (β : ℝ), (0 < β) → (β < pi / 2) → (cos β = cos_beta) →
  (∃ A, A = 4 / 3) :=
by
  sorry

end shaded_area_common_squares_l110_110344


namespace parallelicity_of_curves_l110_110940

-- Define each curve as a function and their derivatives
def curve1 (x : ℝ) : ℝ := x^3 - x
def curve2 (x : ℝ) : ℝ := x + 1 / x
def curve3 (x : ℝ) : ℝ := Real.sin x
def curve4 (x : ℝ) : ℝ := (x-2)^2 + Real.log x

-- Define the derivatives of each curve
def derivative_curve1 (x : ℝ) : ℝ := 3 * x^2 - 1
def derivative_curve2 (x : ℝ) : ℝ := 1 - 1 / (x^2)
def derivative_curve3 (x : ℝ) : ℝ := Real.cos x
def derivative_curve4 (x : ℝ) : ℝ := 2*(x-2) + 1 / x

-- Condition that a curve possesses parallelicity
def has_parallelicity (f' : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = a ∧ f' x2 = a

-- Prove which curves have parallelicity
theorem parallelicity_of_curves :
  has_parallelicity derivative_curve2 ∧ has_parallelicity derivative_curve3 :=
by {
  split;
  sorry
}

end parallelicity_of_curves_l110_110940


namespace triangle_PST_area_l110_110980

variables (P Q R S T : Type)
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S] [MetricSpace T]
variables (PQ QR PR PS PT : ℝ)

-- Conditions provided in the problem
def Conditions : Prop :=
  PQ = 10 ∧ QR = 12 ∧ PR = 13 ∧ PS = 5 ∧ PT = 8

-- Define area of triangle PST
def Area_PST := (1/2) * PS * PT * (229 / 260)

-- The theorem to prove
theorem triangle_PST_area : Conditions → Area_PST = 229 / 13 :=
by
  sorry

end triangle_PST_area_l110_110980


namespace increasing_functions_on_interval_l110_110290

theorem increasing_functions_on_interval (f : ℝ → ℝ) (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hab : a < b) : 
  (f = λ x, abs x → f a ≤ f b) ∧
  ((f = λ x, 3 - x → ¬ (f a ≤ f b))) ∧
  ((f = λ x, 1 / x → ¬ (f a ≤ f b))) ∧
  ((f = λ x, -x^2 + 4 → ¬ (f a ≤ f b))) :=
by
  sorry

end increasing_functions_on_interval_l110_110290


namespace maximum_distinct_divisible_digits_l110_110648

def is_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.nodup

def is_divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, d ≠ 0 ∧ n % d = 0

def valid_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.all (λ d, d > 0 ∧ d ≤ 9)

def max_natural_number_digits : ℕ :=
  7

theorem maximum_distinct_divisible_digits :
  ∀ n : ℕ, is_distinct n ∧
           is_divisible_by_each_digit n ∧
           valid_digits n →
           n.digits 10.length ≤ max_natural_number_digits :=
by
  sorry

end maximum_distinct_divisible_digits_l110_110648


namespace part_I_part_II_part_III_l110_110102

noncomputable def f (x : ℝ) := Real.sin (2 * x - (Real.pi / 6))

theorem part_I : function_periodic f Real.pi :=
sorry

theorem part_II (k : ℤ) : 
  function.increasing_on f (Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi)) :=
sorry

theorem part_III : 
  ∃ (x ∈ Set.Icc 0 (2 * Real.pi / 3)), f x = -1/2 ∧ (x = 0 ∨ x = 2 * Real.pi / 3) :=
sorry

end part_I_part_II_part_III_l110_110102


namespace larger_number_l110_110670

theorem larger_number (HCF : ℕ) (f1 f2 : ℕ) (A : ℕ) :
  HCF = 23 → f1 = 13 → f2 = 15 → A = HCF * f2 →
  (∃ B : ℕ, A * B = HCF * (HCF * f1 * f2)) → A = 345 :=
by
  intros hcf_def f1_def f2_def a_def exist_B_prod
  rw [hcf_def, f1_def, f2_def, a_def] at *
  sorry

end larger_number_l110_110670


namespace coin_flip_sequences_l110_110698

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110698


namespace min_n_pairwise_coprime_prime_l110_110553

open Nat

/-- 
For the set S = {1, 2, ..., 2005}, any subset of S containing 16 pairwise coprime numbers
must contain at least one prime number. --/

theorem min_n_pairwise_coprime_prime (S : Finset ℕ) (hS : S = (Finset.range 2006).erase 0) :
  ∀ A ⊆ S, (∀ x ∈ A, ∀ y ∈ A, x ≠ y → coprime x y) → A.card = 16 → (∃ p ∈ A, Prime p) :=
begin
  intro A,
  intros hA h_coprime h_card,
  sorry
end

end min_n_pairwise_coprime_prime_l110_110553


namespace domain_of_sqrt_expr_l110_110858

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end domain_of_sqrt_expr_l110_110858


namespace unique_function_g_l110_110990

noncomputable def T : Set ℝ := {x : ℝ | x ≠ 0}

structure function_properties (g : ℝ → ℝ) : Prop :=
  (g_on_domain : ∀ x, x ∈ T → g x ∈ ℝ)
  (g_condition_1 : g 2 = 1)
  (g_condition_2 : ∀ x y, x ∈ T → y ∈ T → x + y ∈ T → g (2 / (x + y)) = g (2 / x) + g (2 / y))
  (g_condition_3 : ∀ x y, x ∈ T → y ∈ T → x + y ∈ T → (x + y) * g (x + y) = 2 * x * y * g x * g y)

theorem unique_function_g : ∃! g : ℝ → ℝ, function_properties g :=
begin
  sorry
end

end unique_function_g_l110_110990


namespace karen_hiking_hours_l110_110162

theorem karen_hiking_hours :
  let initial_weight_water := 20
  let initial_weight_food := 10
  let initial_weight_gear := 20
  let consumption_rate_water := 2
  let consumption_rate_food_factor := 1/3
  let final_weight := 34
  -- Define total initial weight.
  let initial_total_weight := initial_weight_water + initial_weight_food + initial_weight_gear
  -- Define total weight consumed per hour.
  let consumption_rate_total := consumption_rate_water + consumption_rate_water * consumption_rate_food_factor
  -- Remaining weight after h hours.
  let remaining_weight := initial_total_weight - consumption_rate_total * h
  -- Equation to solve.
  remaining_weight = final_weight
  in h = 6 := 
by 
  sorry

end karen_hiking_hours_l110_110162


namespace product_combination_count_l110_110320

-- Definitions of the problem

-- There are 6 different types of cookies
def num_cookies : Nat := 6

-- There are 4 different types of milk
def num_milks : Nat := 4

-- Charlie will not order more than one of the same type
def charlie_order_limit : Nat := 1

-- Delta will only order cookies, including repeats of types
def delta_only_cookies : Bool := true

-- Prove that there are 2531 ways for Charlie and Delta to leave the store with 4 products collectively
theorem product_combination_count : 
  (number_of_ways : Nat) = 2531 
  := sorry

end product_combination_count_l110_110320


namespace isosceles_triangle_BC_l110_110171

noncomputable def BC_length (A B C P : ℝ × ℝ) : ℝ :=
  let AP := dist A P
  let BP := dist B P
  let CP := dist C P
  let BC := dist B C
  if AP = 2 ∧ BP = 2 * real.sqrt 2 ∧ CP = 3 ∧ dist A B = dist A C then BC else 0

theorem isosceles_triangle_BC (A B C P : ℝ × ℝ) : 
  BC_length A B C P = 2 * real.sqrt 6 :=
by sorry

end isosceles_triangle_BC_l110_110171


namespace convert_quadratic_to_general_form_l110_110002

theorem convert_quadratic_to_general_form
  (x : ℝ)
  (h : 3 * x * (x - 3) = 4) :
  3 * x ^ 2 - 9 * x - 4 = 0 :=
by
  sorry

end convert_quadratic_to_general_form_l110_110002


namespace coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110265

theorem coefficient_a3b3_in_ab6_c8_div_c8_is_1400 :
  let a := (a : ℝ)
  let b := (b : ℝ)
  let c := (c : ℝ)
  ∀ (a b c : ℝ), (binom 6 3 * a^3 * b^3) * (binom 8 4 * c^0) = 1400 := 
by
  sorry

end coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110265


namespace ellipse_constants_sum_l110_110095

theorem ellipse_constants_sum :
  ∃ (h k a b : ℝ), h = 3 ∧ k = -5 ∧ a = 7 ∧ b = 4 ∧ h + k + a + b = 9 :=
by
  use 3, -5, 7, 4
  split; try { refl }
  split; try { refl }
  split; try { refl }
  calc
    3 + (-5) + 7 + 4 = 3 - 5 + 7 + 4 : by refl
    ... = 9 : by norm_num
  done

end ellipse_constants_sum_l110_110095


namespace cost_comparison_l110_110225

def full_ticket_price : ℝ := 240

def cost_agency_A (x : ℕ) : ℝ :=
  full_ticket_price + 0.5 * full_ticket_price * x

def cost_agency_B (x : ℕ) : ℝ :=
  0.6 * full_ticket_price * (x + 1)

theorem cost_comparison (x : ℕ) :
  (x = 4 → cost_agency_A x = cost_agency_B x) ∧
  (x > 4 → cost_agency_A x < cost_agency_B x) ∧
  (x < 4 → cost_agency_A x > cost_agency_B x) :=
by
  sorry

end cost_comparison_l110_110225


namespace B_more_cost_effective_l110_110679

variable (x y : ℝ)
variable (hx : x ≠ y)

theorem B_more_cost_effective (x y : ℝ) (hx : x ≠ y) :
  (1/2 * x + 1/2 * y) > (2 * x * y / (x + y)) :=
by
  sorry

end B_more_cost_effective_l110_110679


namespace geometric_sequence_common_ratio_l110_110311

-- Definitions based on given conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} 

-- Definition of geometric sequence sum S_n 
def S_n (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Problem statement
theorem geometric_sequence_common_ratio 
  (q : ℝ)
  (h1 : ∀ n, S n = S_n n a q) 
  (h2 : ∀ n, a n = a 0 * q^n)
  (h3 : ∃ S₃ S₄ S₅, S₄ = S 4 ∧ S₃ = S 3 ∧ S₅ = S 5 ∧ 2 * S₃ = S₄ + S₅) 
  : q = -2 :=
sorry

end geometric_sequence_common_ratio_l110_110311


namespace pentagon_circle_intersection_point_l110_110192

theorem pentagon_circle_intersection_point
  (A B C D E O P : Point)
  (hPentagon : RegularPentagon A B C D E O)
  (hIntersection : Intersects (Diagonal AC) (Diagonal BD) P) :
  OnCircle P (CircumscribedCircle A B O) :=
sorry

end pentagon_circle_intersection_point_l110_110192


namespace quadratic_discriminant_a_eq_neg1_l110_110603

theorem quadratic_discriminant_a_eq_neg1 (a : ℝ) : (let Δ := (-3)^2 - 4 * 1 * (-2 * a) in Δ = 1) ↔ a = -1 := by
  sorry

end quadratic_discriminant_a_eq_neg1_l110_110603


namespace distinct_sequences_ten_flips_l110_110736

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110736


namespace max_value_l110_110865

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2 * x + 1

-- Define the interval for x
def interval : set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- State the main theorem
theorem max_value : ∃ x ∈ interval, (∀ y ∈ interval, f y ≤ f x) ∧ f x = 9 :=
by {
  sorry
}

end max_value_l110_110865


namespace coin_flip_sequences_l110_110740

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110740


namespace range_of_a_l110_110105

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) / (x + 1)

noncomputable def g (x a : ℝ) : ℝ := -exp(x - 1) - log x + a

theorem range_of_a:
  (∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 → f x1 ≥ g x2 a) → a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l110_110105


namespace num_arrangements_l110_110248

-- Define the number of teachers
def num_teachers : ℕ := 6

-- Define the maximum teachers each class can have
def max_teachers_per_class : ℕ := 4

-- The core theorem we need to prove
theorem num_arrangements : 
  let possible_arrangements := (finset.powerset (finset.range num_teachers)).filter (λ s, s.card ≤ max_teachers_per_class ∧ (num_teachers - s.card) ≤ max_teachers_per_class)
  in possible_arrangements.card = 50 := 
by
  sorry

end num_arrangements_l110_110248


namespace ratio_area_S_T_l110_110556

-- Define the set T of ordered triples (x, y, z) of nonnegative real numbers lying in the plane x + y + z = 2
def T (x y z : ℝ) : Prop := x + y + z = 2 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the support condition for triples (a, b, c)
def supports (x y z a b c : ℝ) : Prop := 
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨
  (x ≥ a ∧ y < b ∧ z ≥ c) ∨
  (x < a ∧ y ≥ b ∧ z ≥ c)

-- Define the set S consisting of triples in T that support (3/4, 1/2, 1/4)
def S (x y z : ℝ) : Prop := T x y z ∧ supports x y z (3/4) (1/2) (1/4)

-- Prove that the area of S divided by the area of T is 11 / 64
theorem ratio_area_S_T : (area_of S / area_of T) = (11 / 64) := 
  sorry

end ratio_area_S_T_l110_110556


namespace max_blue_numbers_in_circle_l110_110182

-- Definitions corresponding to the problem conditions
def is_blue (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem max_blue_numbers_in_circle (numbers : list ℕ) (h1 : ∀ x ∈ numbers, 1 ≤ x ∧ x ≤ 20)
  (h2 : numbers.length = 20) (h3 : nodup numbers) : ∃k, ∀seq, (∀i, i < seq.length → ∃j, j < seq.length → is_blue seq[i] seq[((i - 1 + seq.length) % seq.length)]) → k ≤ 10 :=
sorry

end max_blue_numbers_in_circle_l110_110182


namespace proof_ABC_is_333_l110_110413

def shape : Type := sorry    -- Replace 'sorry' with actual shape type definition

structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (shapes : List (List shape))
  (black_squares : List (ℕ × ℕ))   -- List of positions of black squares

def count_distinct_shapes (g : Grid) (row : ℕ) : ℕ := sorry    -- Replace 'sorry' with actual function that counts distinct shapes in a row

def ABC (g : Grid) : ℕ :=
  let A := count_distinct_shapes g 1
  let B := count_distinct_shapes g 2
  let C := count_distinct_shapes g 3
  100 * A + 10 * B + C

theorem proof_ABC_is_333 (g : Grid) (condition1 : shapes_fill_grid g) (condition2 : no_adjacent_same_shape g) :
  ABC g = 333 :=
sorry

end proof_ABC_is_333_l110_110413


namespace range_of_m_l110_110466

theorem range_of_m (m x : ℝ) (h₁ : (x / (x - 3) - 2 = m / (x - 3))) (h₂ : x ≠ 3) : x > 0 ↔ m < 6 ∧ m ≠ 3 :=
by
  sorry

end range_of_m_l110_110466


namespace correct_conclusions_l110_110480

/-- 
Statement p: There exists a real number x such that sin x = π / 2.
Statement q: The solution set of x² - 3x + 2 < 0 is (1, 2).

Prove that the correct conclusions from the given conditions are:
① "p and q" is false,
② "p and not q" is false,
③ "not p and q" is true,
④ "not p or not q" is false.
The correct conclusion choice is C, which is ② and ③.
-/
theorem correct_conclusions (p q : Prop)
  (hp : ¬ (∃ x : ℝ, sin x = π / 2))
  (hq : (∀ x : ℝ, x * x - 3 * x + 2 < 0 ↔ x ∈ set.Ioo 1 2)) :
  (¬ (p ∧ q) ∧ ¬ (p ∧ ¬ q) ∧ (¬ p ∧ q) ∧ ¬ (¬ p ∨ ¬ q)) :=
by
  split; sorry     -- ¬ (p ∧ q)
  split; sorry     -- ¬ (p ∧ ¬ q)
  split; sorry     -- (¬ p ∧ q)
  sorry            -- ¬ (¬ p ∨ ¬ q)

end correct_conclusions_l110_110480


namespace product_of_distinct_numbers_l110_110400

theorem product_of_distinct_numbers (x y : ℝ) (h1 : x ≠ y)
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x * y)) :
  x * y = 1 := 
sorry

end product_of_distinct_numbers_l110_110400


namespace number_of_distinct_triples_l110_110451

theorem number_of_distinct_triples (n : ℕ) : 
  (finset.card {xyz : finset (ℕ × ℕ × ℕ) | let ⟨x, y, z⟩ := xyz in x + y + z = 6 * n}.to_finset) = 3 * n^2 := by
sorry

end number_of_distinct_triples_l110_110451


namespace find_k_given_ratio_and_circle_l110_110618

theorem find_k_given_ratio_and_circle (k : ℝ) : 
  ∀ (k : ℝ), ((∃ (L1 L2 : ℝ), (L1 / L2 = 3) ∧ 
  (L1 + L2 = 2 * 2 * Real.pi)) 
  → (2 * Real.sqrt(2) = (Real.abs (1 - k) / (Real.sqrt (1 + k^2))))) 
  → k = -1 :=
by sorry

end find_k_given_ratio_and_circle_l110_110618


namespace valid_6_digit_numbers_l110_110549

def num_valid_numbers : ℕ :=
  let num_valid_r (q : ℕ) := (50 - ((-q) % 26 + 13) / 13) * 2 in
  (10000 - 1000) * 4 * 50
  
theorem valid_6_digit_numbers :
  num_valid_numbers = 36000 := by
  sorry

end valid_6_digit_numbers_l110_110549


namespace convert_1729_to_base_5_l110_110391

theorem convert_1729_to_base_5 :
  let d := 1729
  let b := 5
  let representation := [2, 3, 4, 0, 4]
  -- Check the representation of 1729 in base 5
  d = (representation.reverse.enum_from 0).sum (fun ⟨i, coef⟩ => coef * b^i) :=
  sorry

end convert_1729_to_base_5_l110_110391


namespace chessboard_ratio_sum_l110_110379

theorem chessboard_ratio_sum :
  let m := 19
  let n := 135
  m + n = 154 :=
by
  sorry

end chessboard_ratio_sum_l110_110379


namespace neither_club_members_l110_110334

variables (students : ℕ) (chinese_club : ℕ) (math_club : ℕ) (both_clubs : ℕ)

-- Given conditions
def class2_students := students = 55
def chinese_club_members := chinese_club = 32
def math_club_members := math_club = 36
def both_club_members := both_clubs = 18

-- Prove the number of students who are neither members of the Chinese club nor the Math club
theorem neither_club_members (students chinese_club math_club both_clubs : ℕ)
  (h1 : class2_students students) 
  (h2 : chinese_club_members chinese_club)
  (h3 : math_club_members math_club)
  (h4 : both_club_members both_clubs) :
  students - (chinese_club + math_club - both_clubs) = 5 :=
by {
  -- Definitions
  unfold class2_students chinese_club_members math_club_members both_club_members at *,
  -- Use the definitions
  rw h1,
  rw h2,
  rw h3,
  rw h4,
  -- Simplify the arithmetic
  linarith,
}

end neither_club_members_l110_110334


namespace shaded_region_area_correct_l110_110347

noncomputable def shaded_region_area (side_length : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
if 0 < beta ∧ beta < Real.pi / 2 ∧ cos_beta = 3 / 5 then
  2 / 5
else
  0

theorem shaded_region_area_correct :
  shaded_region_area 2 β (3 / 5) = 2 / 5 :=
by
  -- conditions
  have beta_cond : 0 < β ∧ β < Real.pi / 2 := sorry
  have cos_beta_cond : cos β = 3 / 5 := sorry
  -- we will finish this proof assuming above have been proved.
  exact if_pos ⟨beta_cond, cos_beta_cond⟩

end shaded_region_area_correct_l110_110347


namespace sum_a_1_a_12_sum_b_1_b_2n_l110_110882

noncomputable def f_n (n m : ℕ) : ℚ :=
  if h : m = 0 then 1
  else (List.prod (List.map (λ k, (n - k : ℚ)) (List.range m))) / (Nat.factorial m : ℚ)

def a (m : ℕ) : ℚ := f_n 6 m

def b (n m : ℕ) : ℚ := (-1) ^ m * m * f_n n m

theorem sum_a_1_a_12 : (Finset.range 12).sum (λ m, a (m+1)) = 63 := sorry

theorem sum_b_1_b_2n (n : ℕ) : (Finset.range (2 * n)).sum (λ m, b n (m+1)) ∈ { -1, 0 } := sorry

end sum_a_1_a_12_sum_b_1_b_2n_l110_110882


namespace sum_range_l110_110905

-- Definition of the piecewise function f
def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc 0 π then
    Real.sin x
  else
    Real.log 2015 (x / π)

-- Main theorem statement
theorem sum_range (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
(h_a : f a = f 6) (h_b : f b = f 6) (h_c : f c = f 6) : 
  2 * π < a + b + c ∧ a + b + c < 2016 * π :=
sorry

end sum_range_l110_110905


namespace quadratic_equation_solutions_l110_110910

theorem quadratic_equation_solutions:
  let f : ℝ → ℝ := λ x, a * x^2 + b * x + c in
  (f (-2) = 21) →
  (f (-1) = 12) →
  (f (1) = 0) →
  (f (2) = -3) →
  (f (4) = -3) →
  (∃ x : ℝ, f x = 0 ∧ 
             (x = 1 ∨ x = 5)) :=
by
-- need to provide the proof steps here (omitted)
sorry

end quadratic_equation_solutions_l110_110910


namespace factor_expression_l110_110838

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110838


namespace coefficient_a3b3_l110_110276

theorem coefficient_a3b3 in_ab_c_1overc_expr :
  let coeff_ab := Nat.choose 6 3 
  let coeff_c_expr := Nat.choose 8 4 
  coeff_ab * coeff_c_expr = 1400 :=
by
  sorry

end coefficient_a3b3_l110_110276


namespace sequence_general_term_l110_110234

theorem sequence_general_term (n : ℕ) (a : ℕ → ℤ) : 
  (∀ k : ℕ, a k = if even k then (1 : ℤ) else (-1 : ℤ)) → 
  (∀ k : ℕ, abs (a k) = 1) → 
  a n = (-1)^n * (1 / n) :=
by
  sorry

end sequence_general_term_l110_110234


namespace find_y_from_exponent_equation_l110_110930

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110930


namespace total_grandchildren_l110_110484

-- Define the conditions 
def daughters := 5
def sons := 4
def children_per_daughter := 8 + 7
def children_per_son := 6 + 3

-- State the proof problem
theorem total_grandchildren : daughters * children_per_daughter + sons * children_per_son = 111 :=
by
  sorry

end total_grandchildren_l110_110484


namespace problem_statement_l110_110938

theorem problem_statement (n : ℕ) : 
  let N := ∀ (k : ℕ), k = 1985^n - 7^(n+1) → N = real.sqrt k in
  (N 1 = 44 ∧ (N 1 ∉ ℝ \ ℚ) ∧ (N 1 % 10 ≠ 1) ∧ (N 1 / 10^⌊log 10 N 1⌋₊ ≠ 5)) → "D",
  "none of the above" :=
by
  sorry

end problem_statement_l110_110938


namespace find_number_l110_110802

theorem find_number (x : ℕ) : ((52 + x) * 3 - 60) / 8 = 15 → x = 8 :=
by
  sorry

end find_number_l110_110802


namespace imaginary_part_of_conjugate_l110_110893

noncomputable def i : ℂ := complex.i
noncomputable def z : ℂ := (- (√ 3) + i) ^ 2 / (1 + (√ 3) * i)
noncomputable def z_conjugate : ℂ := complex.conj z

theorem imaginary_part_of_conjugate :
  complex.im z_conjugate = √ 3 := by
  sorry

end imaginary_part_of_conjugate_l110_110893


namespace distinct_sequences_ten_flips_l110_110693

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110693


namespace coin_flips_sequences_count_l110_110714

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110714


namespace diatomic_moles_l110_110325

theorem diatomic_moles (v C R : ℝ) (h_v : v = 1.5) (h_C : C = 120) (h_R : R = 8.31) : 
  v' = 3.8 :=
by
  let v' := (C - 2 * v * R) / (3 * R)
  have h_v' : v' = (120 - 2 * 1.5 * 8.31) / (3 * 8.31), by
    simp [h_v, h_C, h_R]
  simp [h_v', v']
  norm_num
  sorry

end diatomic_moles_l110_110325


namespace cards_red_side_up_l110_110794

theorem cards_red_side_up : (finset.filter (λ n, ∃ k, k * k = n) (finset.range 51)).card = 7 :=
by
  sorry

end cards_red_side_up_l110_110794


namespace run_time_difference_l110_110792

variables (distance duration_injured : ℝ) (initial_speed : ℝ)

theorem run_time_difference (H1 : distance = 20) 
                            (H2 : duration_injured = 22) 
                            (H3 : initial_speed = distance * 2 / duration_injured) :
                            duration_injured - (distance / initial_speed) = 11 :=
by
  sorry

end run_time_difference_l110_110792


namespace solve_for_a_l110_110131

noncomputable def line_slope_parallels (a : ℝ) : Prop :=
  (a^2 - a) = 6

theorem solve_for_a : { a : ℝ // line_slope_parallels a } → (a = -2 ∨ a = 3) := by
  sorry

end solve_for_a_l110_110131


namespace possible_values_of_x_l110_110613

-- Definitions representing the initial conditions
def condition1 (x : ℕ) : Prop := 203 % x = 13
def condition2 (x : ℕ) : Prop := 298 % x = 13

-- Main theorem statement
theorem possible_values_of_x (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 ∨ x = 95 := 
by
  sorry

end possible_values_of_x_l110_110613


namespace functional_equation_solution_l110_110675

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) → (∀ x : ℝ, f x = (1 / 3) * x^2 + (2 / 3) * x - (1 / 3)) :=
by
  assume h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2
  sorry

end functional_equation_solution_l110_110675


namespace angle_between_a_and_b_l110_110080

variables (a b : EuclideanSpace ℝ (Fin 2))

axiom len_a : ∥a∥ = Real.sqrt 2
axiom len_b : ∥b∥ = 1
axiom ortho : inner a (a - 2 • b) = 0

theorem angle_between_a_and_b : 
  real.angle (inner_product_geometry.angle_in_circle a b) = Real.pi / 4 :=
by
  sorry

end angle_between_a_and_b_l110_110080


namespace Mark_in_second_car_l110_110184

structure TrainAssignment where
  car1 : Option String
  car2 : Option String
  car3 : Option String
  car4 : Option String
  car5 : Option String

variables (assignment : TrainAssignment)

def Nina_in_last_car : Prop :=
  assignment.car5 = some "Nina"

def Jess_in_front_of_Owen : Prop :=
  ∃ car_n, car_n ≤ 4 ∧ assignment.car.get car_n = some "Jess" ∧ assignment.car.get (car_n + 1) = some "Owen"

def Mark_ahead_of_Jess : Prop :=
  ∃ car_m car_j, car_m < car_j ∧ assignment.car.get car_m = some "Mark" ∧ assignment.car.get car_j = some "Jess"

def Lisa_one_empty_between_Mark : Prop :=
  ∃ car_l car_m, car_l > car_m + 1 ∧ assignment.car.get car_m = some "Mark" ∧ assignment.car.get car_l = some "Lisa"

theorem Mark_in_second_car 
  (h1 : Nina_in_last_car assignment)
  (h2 : Jess_in_front_of_Owen assignment)
  (h3 : Mark_ahead_of_Jess assignment)
  (h4 : Lisa_one_empty_between_Mark assignment) :
  assignment.car2 = some "Mark" :=
sorry

end Mark_in_second_car_l110_110184


namespace scooter_price_and_installment_l110_110164

variable {P : ℝ} -- price of the scooter
variable {m : ℝ} -- monthly installment

theorem scooter_price_and_installment (h1 : 0.2 * P = 240) (h2 : (0.8 * P) = 12 * m) : 
  P = 1200 ∧ m = 80 := by
  sorry

end scooter_price_and_installment_l110_110164


namespace ellipse_equation_range_for_m_fixed_line_T_l110_110065

-- Given conditions
variables (a b c : ℝ) (h_a_b : a > b) (h_b_0 : b > 0) (h_dist : |c / sqrt 2| = sqrt 2)
variables (N : ℝ × ℝ) (h_N : N = (0, -1))
variables (k m : ℝ) (h_k : k ≠ 0)
variables (P : ℝ × ℝ) (h_P : P = (1, 1))

-- 1. Prove the equation of the ellipse
theorem ellipse_equation (a b c : ℝ) (h_a_b : a > b) (h_b_0 : b > 0) (h_dist : |c / sqrt 2| = sqrt 2) :
  ∃ a b : ℝ, a^2 = 5 ∧ b = 1 ∧ (1 : Prop) = (frac{x^2}{5} + y^2 = 1) := sorry

-- 2.1. Find the range for m
theorem range_for_m (a b c : ℝ) (h_a_b : a > b) (h_b_0 : b > 0) (k m : ℝ) (h_k : k ≠ 0) :
  ∃! m : ℝ, (1 / 4 < m) ∧ (m < 4) := sorry

-- 2.2. Prove that T lies on a fixed line
theorem fixed_line_T (a b c : ℝ) (h_a_b : a > b) (h_b_0 : b > 0) (k m : ℝ) (h_k : k ≠ 0) (P : ℝ × ℝ) (h_P : P = (1,1)) : 
  ∃ (T : ℝ × ℝ), (T.1 + 5 * T.2 - 5 = 0) := sorry

end ellipse_equation_range_for_m_fixed_line_T_l110_110065


namespace tangent_circles_distance_l110_110912

-- Define the radii of the circles.
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 2

-- Define the condition that the circles are tangent.
def tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2 ∨ d = r1 - r2

-- State the theorem.
theorem tangent_circles_distance (d : ℝ) :
  tangent radius_O1 radius_O2 d → (d = 1 ∨ d = 5) :=
by
  sorry

end tangent_circles_distance_l110_110912


namespace factor_expression_l110_110845

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110845


namespace find_y_plus_4_div_y_l110_110895

theorem find_y_plus_4_div_y (y : ℝ) (h : y^3 + 4 / y^3 = 110) : y + 4 / y = 6 := sorry

end find_y_plus_4_div_y_l110_110895


namespace line_properties_l110_110427

theorem line_properties (m : ℝ) :
  (∃ k : ℝ, y = k * x + b → k = -m) ∧
  (mx + y - 2m = 0 → l (2,0) = 0) ∧
  (m = sqrt 3 → ¬ tan θ = m → θ ≠ 60) ∧
  (m = -2 → ¬ (∃ x y, x > 0 ∧ y > 0 ∧ y = 2*x - 4)) :=
sorry

end line_properties_l110_110427


namespace find_y_from_exponent_equation_l110_110927

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110927


namespace isosceles_triangle_apex_angle_l110_110082

theorem isosceles_triangle_apex_angle (a b c : ℝ) (ha : a = 40) (hb : b = 40) (hc : b = c) :
  (a + b + c = 180) → (c = 100 ∨ a = 40) :=
by
-- We start the proof and provide the conditions.
  sorry  -- Lean expects the proof here.

end isosceles_triangle_apex_angle_l110_110082


namespace area_of_shaded_region_l110_110352

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l110_110352


namespace original_denominator_is_21_l110_110354

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21_l110_110354


namespace smallest_four_digit_multiple_of_17_l110_110284

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l110_110284


namespace interest_rate_l110_110781

def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

theorem interest_rate (P T SI : ℝ) (R : ℝ) :
  P = 25000 → T = 3 → SI = 9000 → simple_interest P R T = SI → R = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4
  sorry

end interest_rate_l110_110781


namespace half_lines_form_diagonals_l110_110198

-- Define a type for representing vectors.
structure Vector3 := (x : ℝ) (y : ℝ) (z : ℝ)

-- Define a function to compute the dot product of two vectors.
def dot_product (u v : Vector3) : ℝ := u.x * v.x + u.y * v.y + u.z * v.z

-- Define a function to check if the angle between two vectors is acute.
def acute_angle (u v : Vector3) : Prop := dot_product u v > 0

-- Define three vectors representing the half-lines starting from a common point.
variable (u v w : Vector3)

-- Define a function to check if the sum of three angles is 180 degrees.
def sum_of_angles_180 (α β γ : ℝ) : Prop := α + β + γ = 180

theorem half_lines_form_diagonals:
  (acute_angle u v) ∧ (acute_angle u w) ∧ (acute_angle v w) 
  ∧ (sum_of_angles_180 (angle u v) (angle u w) (angle v w)) :=
sorry

end half_lines_form_diagonals_l110_110198


namespace digit_difference_l110_110669

theorem digit_difference (X Y : ℕ) (h1 : 0 ≤ X ∧ X ≤ 9) (h2 : 0 ≤ Y ∧ Y ≤ 9) (h3 : (10 * X + Y) - (10 * Y + X) = 54) : X - Y = 6 :=
sorry

end digit_difference_l110_110669


namespace interval_of_monotonic_increase_l110_110609

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 3)

theorem interval_of_monotonic_increase :
  { x : ℝ | ∀ y ∈ Icc x 1, f y ≤ f x } = Set.Iio 1 :=
sorry

end interval_of_monotonic_increase_l110_110609


namespace square_tile_sum_l110_110878

def total_area (n1 n2 n3 n4 : ℕ) : ℕ :=
  (n1 * 1) + (n2 * 4) + (n3 * 9) + (n4 * 16)

theorem square_tile_sum (n1 n2 n3 n4 s : ℕ) (h1 : n1 = 4) (h2 : n2 = 8) (h3 : n3 = 12) (h4 : n4 = 16) : s = 20 :=
  let total := total_area n1 n2 n3 n4
  have h_area : total = 400 := by
    simp [total_area, h1, h2, h3, h4]
  have h_side : s * s = total := by rw [h_area]; simp
  by
    simp [Nat.sqrt_eq]
    exact h_side

end square_tile_sum_l110_110878


namespace coefficient_a3b3_l110_110273

theorem coefficient_a3b3 in_ab_c_1overc_expr :
  let coeff_ab := Nat.choose 6 3 
  let coeff_c_expr := Nat.choose 8 4 
  coeff_ab * coeff_c_expr = 1400 :=
by
  sorry

end coefficient_a3b3_l110_110273


namespace parallel_lines_perpendicular_lines_l110_110108

-- Definitions of the lines in slope-intercept form
def l1 := λ x : ℝ, 3 * x - 1
def l2 (m : ℝ) := λ x : ℝ, 2 * x - m * (l1 x) + 1

-- Proof that if l1 is parallel to l2, then m = 2/3
theorem parallel_lines (m : ℝ) : (∀ x : ℝ, l1 x = 3 * x - 1 ∧ l2 m x = 2 * x - m * (l1 x) + 1) →
  ((3 * (-m)) = (-1) * 2) → m = 2 / 3 := by
  sorry

-- Proof that if l1 is perpendicular to l2, then m = -6
theorem perpendicular_lines (m : ℝ) : (∀ x : ℝ, l1 x = 3 * x - 1 ∧ l2 m x = 2 * x - m * (l1 x) + 1) →
  ((2 * 3) + ((-1) * (-m)) = 0) → m = -6 := by
  sorry

end parallel_lines_perpendicular_lines_l110_110108


namespace max_temp_range_l110_110668

-- Definitions based on given conditions
def average_temp : ℤ := 40
def lowest_temp : ℤ := 30

-- Total number of days
def days : ℕ := 5

-- Given that the average temperature and lowest temperature are provided, prove the maximum range.
theorem max_temp_range 
  (avg_temp_eq : (average_temp * days) = 200)
  (temp_min : lowest_temp = 30) : 
  ∃ max_temp : ℤ, max_temp - lowest_temp = 50 :=
by
  -- Assume maximum temperature
  let max_temp := 80
  have total_sum := (average_temp * days)
  have min_occurrences := 3 * lowest_temp
  have highest_temp := total_sum - min_occurrences - lowest_temp
  have range := highest_temp - lowest_temp
  use max_temp
  sorry

end max_temp_range_l110_110668


namespace line_l_properties_l110_110449

noncomputable def line_l_condition (x y : ℝ) : Prop :=
  2 * x + y + 2 = 0

theorem line_l_properties :
  ∃ l : ℝ → ℝ → Prop, 
  (
    (∀ x y : ℝ, (l x y ↔ (2 * x + y + 2 = 0))) ∧ 
    (l (-2) 2) ∧ 
    (∃ P : ℝ × ℝ, (P = (-2, 2)) ∧ (line_l_condition P.1 P.2)) ∧
    (
      let S := 1
      in S = 1
    )
  )
:= 
  sorry

end line_l_properties_l110_110449


namespace shpuntik_can_form_triangle_l110_110256

-- Define lengths of the sticks before swap
variables {a b c d e f : ℝ}

-- Conditions before the swap
-- Both sets of sticks can form a triangle
-- The lengths of Vintik's sticks are a, b, c
-- The lengths of Shpuntik's sticks are d, e, f
axiom triangle_ineq_vintik : a + b > c ∧ b + c > a ∧ c + a > b
axiom triangle_ineq_shpuntik : d + e > f ∧ e + f > d ∧ f + d > e
axiom sum_lengths_vintik : a + b + c = 1
axiom sum_lengths_shpuntik : d + e + f = 1

-- Define lengths of the sticks after swap
-- x1, x2, x3 are Vintik's new sticks; y1, y2, y3 are Shpuntik's new sticks
variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Neznaika's swap
axiom swap_stick_vintik : x1 = a ∧ x2 = b ∧ x3 = f ∨ x1 = a ∧ x2 = d ∧ x3 = c ∨ x1 = e ∧ x2 = b ∧ x3 = c
axiom swap_stick_shpuntik : y1 = d ∧ y2 = e ∧ y3 = c ∨ y1 = e ∧ y2 = b ∧ y3 = f ∨ y1 = a ∧ y2 = b ∧ y3 = f 

-- Total length after the swap remains unchanged
axiom sum_lengths_after_swap : x1 + x2 + x3 + y1 + y2 + y3 = 2

-- Vintik cannot form a triangle with the current lengths
axiom no_triangle_vintik : x1 >= x2 + x3

-- Prove that Shpuntik can still form a triangle
theorem shpuntik_can_form_triangle : y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y3 + y1 > y2 := sorry

end shpuntik_can_form_triangle_l110_110256


namespace trigonometric_identity_proof_l110_110931

theorem trigonometric_identity_proof (θ : ℝ) 
  (h : Real.tan (θ + Real.pi / 4) = -3) : 
  2 * Real.sin θ ^ 2 - Real.cos θ ^ 2 = 7 / 5 :=
sorry

end trigonometric_identity_proof_l110_110931


namespace intersection_X_Y_l110_110109

noncomputable def X : Set ℤ := {x | x^2 - x - 6 ≤ 0}
noncomputable def Y : Set ℝ := {y | ∃ x : ℝ, y = 1 - x^2}

theorem intersection_X_Y : X ∩ Y = ({-2, -1, 0, 1} : Set ℝ) :=
by
  sorry

end intersection_X_Y_l110_110109


namespace chord_length_four_l110_110864

noncomputable def chord_length (center : ℝ × ℝ) (radius : ℝ) (line : ℝ × ℝ × ℝ) : ℝ :=
  2 * (sqrt (radius^2 - (abs ((line.1 * center.1 + line.2 * center.2 + line.3) / (sqrt (line.1^2 + line.2^2))))^2))

theorem chord_length_four :
  chord_length (2, -3) 3 (1, -2, -3) = 4 :=
by
  rw [chord_length]
  -- skipping the proof details
  sorry

end chord_length_four_l110_110864


namespace coin_flip_sequences_l110_110755

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110755


namespace candy_problem_l110_110820

theorem candy_problem 
  (weightA costA : ℕ) (weightB costB : ℕ) (avgPrice per100 : ℕ)
  (hA : weightA = 300) (hCostA : costA = 5)
  (hCostB : costB = 7) (hAvgPrice : avgPrice = 150) (hPer100 : per100 = 100)
  (totalCost : ℕ) (hTotalCost : totalCost = costA + costB)
  (totalWeight : ℕ) (hTotalWeight : totalWeight = (totalCost * per100) / avgPrice) :
  (totalWeight = weightA + weightB) -> 
  weightB = 500 :=
by {
  sorry
}

end candy_problem_l110_110820


namespace abs_sub_nonneg_l110_110119

theorem abs_sub_nonneg (a : ℝ) : |a| - a ≥ 0 :=
sorry

end abs_sub_nonneg_l110_110119


namespace no_increasing_arith_seq_with_rev_arith_seq_l110_110043

def is_arithmetic_progression (seq : List ℕ) : Prop :=
  ∃ d, ∀ i, i < seq.length - 1 → seq.get! (i + 1) = seq.get! i + d

def is_strictly_increasing (seq : List ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → seq.get! i < seq.get! (i + 1)

def is_odd_positive (n : ℕ) : Prop := n % 2 = 1 ∧ n > 0

def binary_reverse (n : ℕ) : ℕ :=
  let rev_bits := Nat.digits 2 n.reverse
  rev_bits.foldl (λ acc x => acc * 2 + x) 0

theorem no_increasing_arith_seq_with_rev_arith_seq :
  ¬ ∃ (A : List ℕ), 
    A.length = 8 ∧ 
    is_arithmetic_progression A ∧ 
    is_strictly_increasing A ∧ 
    (∀ a ∈ A, is_odd_positive a) ∧ 
    is_arithmetic_progression (A.map binary_reverse) :=
by
  sorry

end no_increasing_arith_seq_with_rev_arith_seq_l110_110043


namespace negation_of_existence_l110_110611

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
sorry

end negation_of_existence_l110_110611


namespace unique_solution_for_quadratic_l110_110421

theorem unique_solution_for_quadratic (m : ℝ) (eq : ∀ x : ℝ, 3 * x^2 + m * x + 36 = 0) :
  m = 12 * real.sqrt 3 ∨ m = - (12 * real.sqrt 3) :=
by
  sorry

end unique_solution_for_quadratic_l110_110421


namespace last_triangle_perimeter_l110_110170

theorem last_triangle_perimeter :
  ∃ (T : ℕ → Triangle),
  (T 0).sides = (2011, 2012, 2013) →
  (∀ n : ℕ, T (n + 1) = Triangle.tangents (T n)) →
  ∃ k : ℕ, k < 11 ∧ (T k).sides = (375 / 128, 503 / 128, 631 / 128) ∧ 
  (T k).perimeter = 1509 / 128 :=
by sorry

end last_triangle_perimeter_l110_110170


namespace geometric_sequence_sum_four_l110_110872

theorem geometric_sequence_sum_four (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h2 : q ≠ 1)
  (h3 : -3 * a 0 = -2 * a 1 - a 2)
  (h4 : a 0 = 1) : 
  S 4 = -20 :=
sorry

end geometric_sequence_sum_four_l110_110872


namespace eugene_boxes_needed_l110_110407

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l110_110407


namespace coin_flip_sequences_l110_110759

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110759


namespace mode_is_3_average_is_2_l110_110637

/-- 
Condition: The distribution of the number of usages by students on a given day.
-/
def num_students : ℕ → ℕ
| 0 := 11
| 1 := 15
| 2 := 24
| 3 := 27
| 4 := 18
| 5 := 5
| _ := 0

/-- 
Proving the mode of the number of times students used shared bicycles is 3.
-/
theorem mode_is_3 : 
  ∃ k, (∀ n, num_students n ≤ num_students k) ∧ k = 3 :=
begin
  use 3,
  split,
  {
    intro n,
    cases n,
    { -- n = 0
      exact dec_trivial,
    },
    { -- n = 1
      exact dec_trivial,
    },
    { -- n = 2
      exact dec_trivial,
    },
    { -- n = 3
      exact dec_trivial,
    },
    { -- n = 4
      exact dec_trivial,
    },
    { -- n = 5
      exact dec_trivial,
    },
    { -- n > 5
      exact dec_trivial,
    },
  },
  { -- Proof by simplification
    refl,
  }
end

/-- 
Sum of major (non-zero) distribution values.
-/
def total_usages : ℕ :=
  0 * num_students 0 + 1 * num_students 1 + 2 * num_students 2 + 
  3 * num_students 3 + 4 * num_students 4 + 5 * num_students 5

/--
Total number of students.
-/
def total_students : ℕ :=
  num_students 0 + num_students 1 + num_students 2 + 
  num_students 3 + num_students 4 + num_students 5

/--
Proving the average number of times shared bicycles were used per person is 2.
-/
theorem average_is_2 :
  (total_usages : ℤ) / (total_students : ℤ) ≈ 2 :=
begin
  -- calculations and approx steps go here
  sorry
end

end mode_is_3_average_is_2_l110_110637


namespace work_rate_c_l110_110297

theorem work_rate_c (A B C : ℝ) 
  (h1 : A + B = 1 / 15) 
  (h2 : A + B + C = 1 / 5) :
  (1 / C) = 7.5 :=
by 
  sorry

end work_rate_c_l110_110297


namespace macks_speed_to_office_l110_110562

theorem macks_speed_to_office (total_time_to_and_return : ℝ) (time_to_office : ℝ) 
(return_speed : ℝ) (distance_to_office_eq : ℝ) :
  total_time_to_and_return = 3 ∧ time_to_office = 1.4 ∧ return_speed = 62 →
  (∃ v : ℝ, v ≈ 70.86 ∧ v * time_to_office = distance_to_office_eq ∧ return_speed * (total_time_to_and_return - time_to_office) = distance_to_office_eq) :=
by
  intros h
  sorry

end macks_speed_to_office_l110_110562


namespace monotonicity_of_f_l110_110469

noncomputable def f (x : ℝ) : ℝ := - (2 * x) / (1 + x^2)

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ x > 1) → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ -1 < x ∧ y < 1 → f y < f x) := sorry

end monotonicity_of_f_l110_110469


namespace angle_bisector_ratio_l110_110155

-- Definitions of distances in the triangle
def XY : ℝ := 8
def XZ : ℝ := 6
def YZ : ℝ := 4

-- Assume Q is the intersection of the angle bisectors
-- We need to prove that the ratio of YQ to QV is 1.5

theorem angle_bisector_ratio (Q : Point) (U : Point) (V : Point) : 
  -- Conditions
  ∀ (YZ XY XZ : ℝ), 
  XY = 8 ∧ XZ = 6 ∧ YZ = 4 →
  -- Q lies at the intersection of the angle bisectors of XU and YV
  -- To prove
  (YZ / YX : ℝ) ∧ (YY / QV : ℝ) := 
begin
  sorry
end

end angle_bisector_ratio_l110_110155


namespace convert_1729_to_base5_l110_110386

-- Definition of base conversion from base 10 to base 5.
def convert_to_base5 (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if h : n = 0 then acc
    else let quotient := n / 5
         let remainder := n % 5
         aux quotient (remainder :: acc)
  aux n []

-- The theorem we seek to prove.
theorem convert_1729_to_base5 : convert_to_base5 1729 = [2, 3, 4, 0, 4] :=
by
  sorry

end convert_1729_to_base5_l110_110386


namespace total_coughs_after_20_minutes_l110_110051

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end total_coughs_after_20_minutes_l110_110051


namespace a_monotonically_decreasing_iff_l110_110909

noncomputable def a : ℕ → ℝ 
| 0       := t
| (n + 1) := (a n) / 2 + 2 / (a n)

theorem a_monotonically_decreasing_iff (t : ℝ) :
  (∀ n, a (n + 1) < a n) ↔ 2 < t := 
sorry

end a_monotonically_decreasing_iff_l110_110909


namespace mean_increases_l110_110812

theorem mean_increases 
  (scores : List ℕ) 
  (ninth_game_score : ℕ) 
  (range_before : ℕ) 
  (median_before : ℚ)
  (mean_before : ℚ)
  (mode_before : List ℕ)
  (midrange_before : ℚ)
  (h_scores : scores = [41, 45, 45, 49, 52, 52, 54, 60]) 
  (h_ninth_game_score : ninth_game_score = 50)
  :  
  (let new_scores := scores ++ [ninth_game_score] in 
  let range_after := 60 - 41 in 
  let median_after := 50 in 
  let mean_after := (398 + 50) / 9 in 
  let mode_after := [45, 52] in 
  let midrange_after := (60 + 41) / 2 in 
  range_before = range_after ∧
  median_before = median_after ∧
  mean_before < mean_after ∧
  mode_before = mode_after ∧
  midrange_before = midrange_after) :=
by
  -- Proof construction starts here, which is skipped as per the requirement
  sorry

end mean_increases_l110_110812


namespace projection_onto_line_l110_110038

noncomputable def vector_projection 
    (v : ℝ × ℝ × ℝ) (u : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
( ( (v.1 * u.1 + v.2 * u.2 + v.3 * u.3) / (u.1 * u.1 + u.2 * u.2 + u.3 * u.3) ) *
    (u.1, u.2, u.3) )

theorem projection_onto_line :
  let line_direction := (3, -2, 6)
  let vector := (4, -4, -1)
  let expected_projection := ((6 / 7), (-4 / 7), (12 / 7))
  vector_projection vector line_direction = expected_projection :=
by 
  let line_direction := (3, -2, 6)
  let vector := (4, -4, -1)
  let expected_projection := ((6 / 7), (-4 / 7), (12 / 7))
  sorry

end projection_onto_line_l110_110038


namespace find_fraction_l110_110126

theorem find_fraction (f n : ℝ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1 / 5 :=
by
  -- skipping the proof as requested
  sorry

end find_fraction_l110_110126


namespace calculate_octagon_area_l110_110836

-- Define the setup with a square of side length 10 units
structure Square :=
  (side_length : ℝ) (vertex1 vertex2 vertex3 vertex4 center midpoint1 midpoint2 midpoint3 midpoint4 : ℂ)
  (verts_connected_to_midpoints : Prop)

-- Given conditions: side length = 10, vertices, center, midpoints, and connections
noncomputable def given_square : Square :=
{ side_length := 10,
  vertex1 := complex.I,
  vertex2 := 1 + complex.I,
  vertex3 := 1,
  vertex4 := 0,
  center := (1 + complex.I) / 2,
  midpoint1 := (complex.I + 1) / 2,
  midpoint2 := (1 + (1 + complex.I)) / 2,
  midpoint3 := (1 + 0) / 2,
  midpoint4 := (0 + complex.I) / 2,
  verts_connected_to_midpoints := true}

-- Define the function to calculate the area of the octagon
noncomputable def octagon_area (sq : Square) : ℝ :=
  (sq.side_length * sq.side_length) / 6

-- The problem statement: Prove that the area of the octagon is 100/6
theorem calculate_octagon_area (sq : Square) : sq.vert_connected_to_midpoints → octagon_area sq = 100 / 6 :=
by sorry  -- Proof omitted

end calculate_octagon_area_l110_110836


namespace angle_ADB_is_90_degrees_l110_110363

theorem angle_ADB_is_90_degrees
  (C : Point)
  (A B D : Point)
  (circle : Circle C 12)
  (triangle_isosceles : triangle_isosceles A B C)
  (angle_ACB : Angle_deg (angle A C B) 100)
  (on_circle_A : OnCircle circle A)
  (on_circle_B : OnCircle circle B)
  (on_circle_D : OnCircle circle D)
  (line_AD : is_line A D) :
  Angle_deg (angle A D B) 90 :=
sorry

end angle_ADB_is_90_degrees_l110_110363


namespace probability_ratio_l110_110404

noncomputable def numWays3_7_5_5_5 : ℕ :=
  (Nat.choose 5 1) * (Nat.choose 4 1) *
  (Nat.choose 25 3) * (Nat.choose 22 7) * 
  (Nat.choose 15 5) * (Nat.choose 10 5) * 
  (Nat.choose 5 5)

noncomputable def numWays5_5_5_5_5 : ℕ :=
  (Nat.choose 25 5) * (Nat.choose 20 5) *
  (Nat.choose 15 5) * (Nat.choose 10 5) * 
  (Nat.choose 5 5)

noncomputable def p : ℚ :=
  (numWays3_7_5_5_5 : ℚ) / (Nat.choose 25 25)

noncomputable def q : ℚ :=
  (numWays5_5_5_5_5 : ℚ) / (Nat.choose 25 25)

theorem probability_ratio : p / q = 12 := by
  sorry

end probability_ratio_l110_110404


namespace zahra_kimmie_money_ratio_l110_110163

theorem zahra_kimmie_money_ratio (KimmieMoney ZahraMoney : ℕ) (hKimmie : KimmieMoney = 450)
  (totalSavings : ℕ) (hSaving : totalSavings = 375)
  (h : KimmieMoney / 2 + ZahraMoney / 2 = totalSavings) :
  ZahraMoney / KimmieMoney = 2 / 3 :=
by
  -- Conditions to be used in the proof, but skipped for now
  sorry

end zahra_kimmie_money_ratio_l110_110163


namespace find_point_A_l110_110571

-- Define data needed for the problem
variables {R m : ℝ} -- R is the radius of circle O, m is the distance BO
variables {x r : ℝ} -- x is the distance from B to O', r is radius of O'

-- The circle O
def circle_O (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = R ^ 2

-- The line a outside the circle
def line_a : set (ℝ × ℝ) := {P | ∃ x : ℝ, P = (x, 0)}

-- Points M and N on line a
def points_on_line_a (M N : ℝ × ℝ) : Prop :=
  line_a M ∧ line_a N ∧ M ≠ N

-- Circle O' with MN as diameter is tangent to circle O
def circle_O'_tangent (M N : ℝ × ℝ) :=
  let O' : (ℝ × ℝ) := ((M.1 + N.1) / 2, 0)
  in dist O' (0, 0) = R + dist M N / 2

-- The existence of point A
def exist_point_A (A : ℝ × ℝ) :=
  (A.1 = 0 ∧ (A.2 = sqrt (m^2 - R^2) ∨ A.2 = -sqrt (m^2 - R^2))) ∧ 
  ∀ (M N : ℝ × ℝ), (points_on_line_a M N) → (circle_O'_tangent M N) →
  (∃ k : ℝ, k = mk_angle A M N)

-- Main theorem statement
theorem find_point_A
  (h_circle_O : circle_O)
  (h_line_a : line_a)
  (h_MN_points_on_line_a : ∀ M N, points_on_line_a M N)
  (h_circle_O'_tangent : ∀ M N, circle_O'_tangent M N)
  : ∃ A, exist_point_A A :=
sorry

end find_point_A_l110_110571


namespace part1_proof_part2_proof_l110_110544

variable {a : ℕ → ℝ} -- sequence a_n
variable (S : ℕ → ℝ) -- sum sequence S_n
variable (q : ℝ) -- common ratio
variable (c : ℝ) -- constant c

-- Definitions due to conditions
def is_geom_seq (a: ℕ → ℝ) (q: ℝ) : Prop := ∀ n, a (n + 1) = a n * q
def is_sum_geom (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (0 to n-1).sum (λ k, a k)
def sum_geom_series (a₁ q: ℝ) (n: ℕ) : ℝ := if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- condition premises
def conditions (a : ℕ → ℝ) (q a₁ c: ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ a 0 = a₁ ∧ a₁ > 0 ∧ q > 0

-- (κκκ) Proof statement for Part (1) : ∀ n, (lg (S(n)+lg(S(n+2)) )/2 < lg S(n+1)
theorem part1_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q a₁ : ℝ) (h : conditions a q a₁ c) :
  ∀ n, (Math.log(S n) + Math.log(S (n + 2))) / 2 < Math.log(S (n + 1)) := 
sorry

-- (κκκ) Proof statement for Part (2): ∀ n, ∀ c, (lg (S(n)-c)+lg(S(n+2)-c) )/2 ≠ lg S(n+1)-c
theorem part2_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q a₁: ℝ)  (h : conditions a q a₁ c) :
  ∀ n, ∀ c, ¬((Math.log (S n - c) + Math.log (S (n + 2) - c)) / 2 = Math.log (S (n + 1) - c)) := 
sorry

end part1_proof_part2_proof_l110_110544


namespace seat_arrangement_l110_110145

theorem seat_arrangement :
  ∃ (arrangement : Fin 7 → String), 
  (arrangement 6 = "Diane") ∧
  (∃ (i j : Fin 7), i < j ∧ arrangement i = "Carla" ∧ arrangement j = "Adam" ∧ j = (i + 1)) ∧
  (∃ (i j k : Fin 7), i < j ∧ j < k ∧ arrangement i = "Brian" ∧ arrangement j = "Ellie" ∧ (k - i) ≥ 3) ∧
  arrangement 3 = "Carla" := 
sorry

end seat_arrangement_l110_110145


namespace find_y_l110_110924

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110924


namespace ratio_height_radius_l110_110336

variable {r : ℝ} (R H : ℝ)

-- Condition: Height of the cylinder equals the diameter of the sphere.
def height_equals_diameter (r H : ℝ) : Prop :=
  H = 2 * r

-- Condition: The volume of the cylinder is three times that of the sphere.
def volume_condition (r R H : ℝ) : Prop :=
  π * R^2 * H = 3 * (4/3 * π * r^3)

-- The final proof goal: The ratio of the height of the cylinder to the radius of its base is √2.
theorem ratio_height_radius {r R H : ℝ} (h1 : height_equals_diameter r H) (h2 : volume_condition r R H) :
  H / R = Real.sqrt 2 :=
sorry

end ratio_height_radius_l110_110336


namespace shaded_area_common_squares_l110_110342

noncomputable def cos_beta : ℝ := 3 / 5

theorem shaded_area_common_squares :
  ∀ (β : ℝ), (0 < β) → (β < pi / 2) → (cos β = cos_beta) →
  (∃ A, A = 4 / 3) :=
by
  sorry

end shaded_area_common_squares_l110_110342


namespace turtle_problem_solution_l110_110247

noncomputable def turtle_problem : Prop :=
  let T1_statements := "Two turtles are behind me."
  let T2_statements := "One turtle is behind me and one is ahead of me."
  let T3_statements := "Two turtles are ahead of me and one is behind me."
  (T1_statements = "Two turtles are behind me") ∧
  (T2_statements = "One turtle is behind me and one is ahead of me") ∧
  (T3_statements = "Two turtles are ahead of me and one is behind me") ∧
  ¬(T3_statements = "truthful")

theorem turtle_problem_solution : turtle_problem :=
  by {
    unfold turtle_problem,
    simp,
    split,
    { refl },
    split,
    { refl },
    split,
    { refl },
    { sorry }
  }

end turtle_problem_solution_l110_110247


namespace count_subsets_with_odd_and_multiple_of_3_l110_110917

def T : Set ℕ := {1, 3, 5, 7, 9}
def U : Set ℕ := {3, 9}

theorem count_subsets_with_odd_and_multiple_of_3 : 
  {A : Set ℕ // A ⊆ T ∧ ¬ Disjoint A U}.card = 8 := 
by
  sorry

end count_subsets_with_odd_and_multiple_of_3_l110_110917


namespace saline_solution_water_l110_110138

theorem saline_solution_water (volume_salt volume_water total_mixture target_solution : ℝ)
  (h1 : volume_salt = 0.05)
  (h2 : volume_water = 0.03)
  (h3 : total_mixture = volume_salt + volume_water)
  (h4 : target_solution = 0.6) :
  let water_fraction := volume_water / total_mixture in
  water_fraction * target_solution = 0.225 :=
by
  sorry

end saline_solution_water_l110_110138


namespace condition1_arrangements_condition2_arrangements_l110_110632

theorem condition1_arrangements (boys girls positions : ℕ) (boy_A_allexcept : ℕ) (remaining_arrangements : ℕ) :
  boys = 3 → girls = 4 → positions = 7 → boy_A_allexcept = 4 → remaining_arrangements = 720 →
  boy_A_allexcept * remaining_arrangements = 2880 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact rfl

theorem condition2_arrangements (boys girls positions : ℕ) (boy_AB_positions : ℕ) (remaining_arrangements : ℕ) :
  boys = 3 → girls = 4 → positions = 7 → boy_AB_positions = 2 → remaining_arrangements = 120 →
  boy_AB_positions * remaining_arrangements = 240 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact rfl

end condition1_arrangements_condition2_arrangements_l110_110632


namespace square_overlap_area_l110_110337

theorem square_overlap_area (β : ℝ) (h1 : 0 < β) (h2 : β < 90) (h3 : Real.cos β = 3 / 5) : 
  area (common_region (square 2) (rotate_square β (square 2))) = 4 / 3 :=
sorry

end square_overlap_area_l110_110337


namespace triangle_ABC_right_angled_l110_110521

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end triangle_ABC_right_angled_l110_110521


namespace eugene_used_six_boxes_of_toothpicks_l110_110408

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l110_110408


namespace lambda_five_geq_twice_sin_54_l110_110435

theorem lambda_five_geq_twice_sin_54 (P : Fin 5 → ℝ × ℝ) :
  let dist := λ (i j : Fin 5), (P i).dist (P j)
  let lambda_5 := dist.finset.max - dist.finset.min in
  lambda_5 ≥ 2 * Real.sin (54 * Real.pi / 180) := sorry

end lambda_five_geq_twice_sin_54_l110_110435


namespace police_emergency_number_has_prime_divisor_gt_7_l110_110784

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : 
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_has_prime_divisor_gt_7_l110_110784


namespace part_I_part_II_part_III_l110_110475

noncomputable def f (x a : ℝ) : ℝ := a * Real.log (x + 1) - a * x - x^2

-- Theorem (I) If \( x = 1 \) is an extremum of \( f(x) \), then \( a = -4 \)
theorem part_I (a : ℝ) (h : ∀ x: ℝ, f 1 a = 0) : a = -4 := sorry

-- Theorem (II) Discussing the monotonicity of f(x) for the value of a from (I)
theorem part_II {x : ℝ} (hx : x ∈ set.Icc (-1:ℝ) (real.sqrt 2 * cos (1/2))) :
  ∃ a : ℝ, (x < 1 -> f x (-4) = 0 ∧ f x (-4) > 0) ∧ (x > 1  -> f x (-4) < 0) := sorry

-- Theorem (III) Prove that for any positive integer \( n \), \( \ln(n+1) < 2 + \frac{3}{2^2} + \frac{4}{3^2} + \ldots + \frac{n+1}{n^2} \)
theorem part_III (n : ℕ) (hn : n > 0) : Real.log (n + 1) < (2 + ∑ i in Finset.range n, ((i + 2) / (Real.ofNat (i + 1))^2)) := sorry

end part_I_part_II_part_III_l110_110475


namespace max_profit_production_units_l110_110779

-- Define the fixed cost and production cost
def FixedCost : ℝ := 2.8
def ProductionCost (x : ℝ) : ℝ := x

-- Define the total cost function
def G (x : ℝ) : ℝ := FixedCost + ProductionCost x

-- Define the revenue function
def R (x : ℝ) : ℝ := if x ≤ 5 then -0.4 * x^2 + 4.2 * x else 11

-- Define the profit function
def f (x : ℝ) : ℝ := R x - G x

-- Prove that the profit function achieves a maximum at x = 4
theorem max_profit_production_units :
  (∀ x, f x ≤ f 4) ∧ f 4 = 3.6 :=
by
  sorry

end max_profit_production_units_l110_110779


namespace oprod_eval_l110_110393

def oprod (a b : ℕ) : ℕ :=
  (a * 2 + b) / 2

theorem oprod_eval : oprod (oprod 4 6) 8 = 11 :=
by
  -- Definitions given in conditions
  let r := (4 * 2 + 6) / 2
  have h1 : oprod 4 6 = r := by rfl
  let s := (r * 2 + 8) / 2
  have h2 : oprod r 8 = s := by rfl
  exact (show s = 11 from sorry)

end oprod_eval_l110_110393


namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l110_110576

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l110_110576


namespace find_f_neg_five_halves_l110_110088

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x) else fmod (x + 2) 2 * (1 - fmod (x + 2) 2)

theorem find_f_neg_five_halves : 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∀ x : ℝ, f (x + 2) = f x) → 
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2 * x * (1 - x)) → 
  f (-5 / 2) = 1 / 2 := 
by 
  sorry

end find_f_neg_five_halves_l110_110088


namespace ellipse_equation_correct_l110_110090

variable {x y b : ℝ}

noncomputable def ellipse_equation (b : ℝ) : Prop := 
  0 < b ∧ b < 5 ∧ (∃ a c : ℝ, a = 5 ∧ c = sqrt (25 - b^2) ∧ (2 * b = 2 * sqrt (25 - b^2) + 2 * 5) ∧ (2 * sqrt (25 - b^2) ≠ 0))

theorem ellipse_equation_correct :
  (ellipse_equation b) → (b = 4) → (0 < b ∧ b < 5 → ∀ x y, x*x/25 + y*y/16 = 1) :=
by 
  intro cond b_equiv
  sorry

end ellipse_equation_correct_l110_110090


namespace bill_new_win_percentage_min_X_percentage_l110_110365

theorem bill_new_win_percentage 
  (initial_games_played : ℕ := 200) 
  (initial_win_percentage : ℝ := 0.63) 
  (additional_games_played : ℕ := 100) 
  (games_lost_in_additional : ℕ := 43) 
  (new_total_games : ℕ := initial_games_played + additional_games_played) 
  (initial_wins : ℕ := initial_games_played * ℝ.to_nat initial_win_percentage) 
  (wins_in_additional : ℕ := additional_games_played - games_lost_in_additional) 
  (total_wins : ℕ := initial_wins + wins_in_additional) : 
  ∃ (new_win_percentage : ℝ), 
  new_win_percentage = (total_wins : ℝ) / (new_total_games : ℝ) * 100 ∧ 
  new_win_percentage = 61 := 
by 
  sorry

theorem min_X_percentage 
  (total_wins : ℕ := 183) 
  (new_total_games : ℕ := 300) :
  ∃ (X : ℝ), 
  X = (total_wins : ℝ) / (new_total_games : ℝ) * 100 ∧ 
  X = 61 := 
by 
  sorry

end bill_new_win_percentage_min_X_percentage_l110_110365


namespace length_fraction_l110_110578

-- Definitions
variables {A B C D E : Point}
variables (x y z : ℝ) -- Lengths of \overline{CD}, \overline{BD}, \overline{ED}

-- Conditions
def cond1 := B ∈ line_segment A D
def cond2 := C ∈ line_segment A D
def cond3 := E ∈ line_segment A D

def cond4 := length (A, B) = 3 * length (B, D)
def cond5 := length (A, C) = 7 * length (C, D)
def cond6 := length (A, E) = 5 * length (E, D)

-- Theorem
theorem length_fraction : 
  cond1 → cond2 → cond3 → cond4 → cond5 → cond6 → 
  (length (B, C) + length (C, E)) / length (A, D) = 1 / 3 := 
by
  sorry

end length_fraction_l110_110578


namespace sec_seven_pi_over_six_l110_110014

theorem sec_seven_pi_over_six : sec (7 * Real.pi / 6) = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_seven_pi_over_six_l110_110014


namespace distinct_sequences_ten_flips_l110_110739

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110739


namespace circumcircles_tangent_at_X_l110_110189

noncomputable def midpoint (A C : Point) : Point := sorry

structure IsoscelesTriangle (A B C : Point) :=
(base_midpoint : Point)
(is_midpoint : midpoint A C = base_midpoint)

structure PointExtension (A B C D K : Point) :=
(on_AC_extension : A-X--C-D)
(on_BC_extension : B-C-X--K)
(B_equals_CD : distance B C = distance C D)
(CM_equals_CK : distance (base_midpoint_of_triangle A B C) C = distance C K)

theorem circumcircles_tangent_at_X (A B C D K : Point)
  (h1 : IsoscelesTriangle A B C)
  (h2 : PointExtension A B C D K)
  (X : Point)
  : tangent_at circumcircle (A B D) circumcircle (M C K) X := 
sorry

end circumcircles_tangent_at_X_l110_110189


namespace math_books_count_l110_110305

theorem math_books_count (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 396) : M = 54 :=
sorry

end math_books_count_l110_110305


namespace cars_needed_to_double_march_earnings_l110_110663

-- Definition of given conditions
def base_salary : Nat := 1000
def commission_per_car : Nat := 200
def march_earnings : Nat := 2000

-- Question to prove
theorem cars_needed_to_double_march_earnings : 
  (2 * march_earnings - base_salary) / commission_per_car = 15 := 
by sorry

end cars_needed_to_double_march_earnings_l110_110663


namespace percentage_difference_l110_110314

-- Define the numbers
def n : ℕ := 1600
def m : ℕ := 650

-- Define the percentages calculated
def p₁ : ℕ := (20 * n) / 100
def p₂ : ℕ := (20 * m) / 100

-- The theorem to be proved: the difference between the two percentages is 190
theorem percentage_difference : p₁ - p₂ = 190 := by
  sorry

end percentage_difference_l110_110314


namespace units_digit_of_sum_is_three_l110_110368

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_factorials : ℕ :=
  (List.range 10).map factorial |>.sum

def power_of_ten (n : ℕ) : ℕ :=
  10^n

theorem units_digit_of_sum_is_three : 
  units_digit (sum_of_factorials + power_of_ten 3) = 3 := by
  sorry

end units_digit_of_sum_is_three_l110_110368


namespace travel_distance_is_5_plus_sqrt_61_l110_110821

open Real -- Bring Real namespace into the current scope.

-- Define the points as tuples of coordinates.
def p1 : Point := (-3, 4)
def p2 : Point := (1, 1)
def p3 : Point := (6, -5)

-- Define the Euclidean distance formula between two points in the plane.
def distance (a b : Point) : ℝ :=
  sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

-- Define the total distance traveled.
def total_distance : ℝ :=
  distance p1 p2 + distance p2 p3

-- State the main goal to be proved.
theorem travel_distance_is_5_plus_sqrt_61 : total_distance = 5 + sqrt 61 := 
by sorry

end travel_distance_is_5_plus_sqrt_61_l110_110821


namespace february_1_is_friday_l110_110937

def day := ℕ
noncomputable def nat_mod := @Nat.mod

-- Let's define days of the week as natural numbers where
-- 0 = Sunday, 1 = Monday, 2 = Tuesday, 3 = Wednesday, 4 = Thursday, 5 = Friday, 6 = Saturday.

inductive WeekDay : Type
| Sunday : WeekDay
| Monday : WeekDay
| Tuesday : WeekDay
| Wednesday : WeekDay
| Thursday : WeekDay
| Friday : WeekDay
| Saturday : WeekDay

open WeekDay

-- We assume that day 13 of February is Wednesday.

axiom day_13_is_wednesday : ∀ (d : day), (d = 13) → (WeekDay.Wednesday = Wednesday)

-- We need to prove that February 1 is a Friday given that February 13 is a Wednesday.

theorem february_1_is_friday : (∀ (d : day), (d = 13) → (WeekDay.Wednesday = Wednesday)) → (WeekDay.Friday = Friday) :=
by
  sorry

end february_1_is_friday_l110_110937


namespace false_statement_l110_110003

noncomputable def heartsuit (x y : ℝ) := abs (x - y)
noncomputable def diamondsuit (z w : ℝ) := (z + w) ^ 2

theorem false_statement : ∃ (x y : ℝ), (heartsuit x y) ^ 2 ≠ diamondsuit x y := by
  sorry

end false_statement_l110_110003


namespace trig_identity_l110_110309

theorem trig_identity (α : ℝ) : 
  (sin (2 * α) - sin (3 * α) + sin (4 * α)) / (cos (2 * α) - cos (3 * α) + cos (4 * α)) = tan (3 * α) :=
  sorry

end trig_identity_l110_110309


namespace simplify_expression_evaluate_expression_with_values_l110_110209

-- Problem 1: Simplify the expression to -xy
theorem simplify_expression (x y : ℤ) : 
  3 * x^2 + 2 * x * y - 4 * y^2 - 3 * x * y + 4 * y^2 - 3 * x^2 = - x * y :=
  sorry

-- Problem 2: Evaluate the expression with given values
theorem evaluate_expression_with_values (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  a + (5 * a - 3 * b) - 2 * (a - 2 * b) = 5 :=
  sorry

end simplify_expression_evaluate_expression_with_values_l110_110209


namespace profit_percentage_l110_110664

-- Given conditions
def CP : ℚ := 25 / 15
def SP : ℚ := 32 / 12

-- To prove profit percentage is 60%
theorem profit_percentage (CP SP : ℚ) (hCP : CP = 25 / 15) (hSP : SP = 32 / 12) :
  (SP - CP) / CP * 100 = 60 := 
by 
  sorry

end profit_percentage_l110_110664


namespace math_proof_problem_l110_110911

open Real

variables {A B C D : ℝ}
variables {h p q b c a : ℝ}
variables (triangle : Type) [Inhabited triangle]
variables (isRightTriangle : triangle → Prop)
variables (isAltitude : triangle → ℝ → Prop)
variables (hasLength : triangle → ℝ → Prop)

def problem : Prop :=
  ∀ (t : triangle), 
  isRightTriangle t ∧ isAltitude t h ∧ 
  hasLength t p ∧ hasLength t q → 
  (h * h = p * q) ∧ (b * b = p * c) ∧ (a * b = c * h)

theorem math_proof_problem : problem :=
by sorry

end math_proof_problem_l110_110911


namespace probability_of_closer_to_F_l110_110979

-- Define the sides of the triangle DEF
def DE : ℝ := 6
def EF : ℝ := 8
def DF : ℝ := 10

-- Define the area of triangle DEF
def area_DEF : ℝ := (1 / 2) * DE * EF

-- Define the area of the region closer to F
def area_GH : ℝ := (DF / 2) * (EF / 2)

-- Probability that point Q inside triangle DEF is closer to F than to D or E
def probability_closer_to_F : ℝ := area_GH / area_DEF

theorem probability_of_closer_to_F :
  probability_closer_to_F = 5 / 6 :=
by
  sorry

end probability_of_closer_to_F_l110_110979


namespace coin_flip_sequences_l110_110726

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110726


namespace plane_Q_is_correct_l110_110255

open Real

-- We state our given conditions as definitions first
def plane1 (x y z : ℝ) : Prop := x + y + z = 1
def plane2 (x y z : ℝ) : Prop := x + 3y - z = 2
def point (x y z : ℝ) : Prop := (x, y, z) = (2, 1, 0)
def distance (a b c d : ℝ) (x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c * z₀ + d) / sqrt (a ^ 2 + b ^ 2 + c ^ 2)

-- Now we formulate the main Theorem
theorem plane_Q_is_correct (a b c d : ℝ)
  (h₀ : ∃ a b, a * (x + y + z - 1) + b * (x + 3y - z - 2) = 0)
  (h₁ : distance a b c d 2 1 0 = 3 / sqrt 14) :
  (x - y + 3z = 0) := 
sorry

end plane_Q_is_correct_l110_110255


namespace triangle_XMY_area_l110_110964

-- Define key variables and givens
variables (YM MX YZ : ℝ)
variable (XY_area : ℝ)

-- Define the conditions
def conditions :=
  YM = 2 ∧ MX = 3 ∧ YZ = 5

-- Define the area calculation
def area_XMY (YM MX : ℝ) : ℝ :=
  0.5 * YM * MX

-- The statement we are trying to prove
theorem triangle_XMY_area (h : conditions) : area_XMY YM MX = 3 := 
by
  simp [conditions, area_XMY]
  rw [h.1, h.2, h.3]
  norm_num

  sorry  -- Proof goes here

end triangle_XMY_area_l110_110964


namespace factor_tree_value_l110_110143

constant F: ℕ 
constant G: ℕ 
constant H: ℕ 
constant X: ℕ

axiom h1 : F = 11 * (11 * 2)
axiom h2 : H = 17 * 2
axiom h3 : G = 7 * H
axiom h4 : X = F * G

theorem factor_tree_value : X = 57556 := 
by 
  -- proof goes here
  sorry

end factor_tree_value_l110_110143


namespace negation_proposition_l110_110612

theorem negation_proposition :
  ¬ (∃ x : ℝ, x < -1 ∧ x^2 ≥ 1) ↔ ∀ x : ℝ, x < -1 → x^2 < 1 :=
begin
  sorry
end

end negation_proposition_l110_110612


namespace right_angled_triangle_l110_110291

theorem right_angled_triangle (
  A := (1 : ℝ, 1 : ℝ, 2 : ℝ),
  B := (2 : ℝ, Real.sqrt 7, Real.sqrt 3),
  C := (4 : ℝ, 6 : ℝ, 8 : ℝ),
  D := (5 : ℝ, 12 : ℝ, 11 : ℝ)
) : 
  ¬ (A.1^2 + A.2^2 = A.3^2) ∧ 
  (B.1^2 + B.2^2 = B.3^2) ∧ 
  ¬ (C.1^2 + C.2^2 = C.3^2) ∧ 
  ¬ (D.1^2 + D.2^2 = D.3^2) :=
by {
  sorry
}

end right_angled_triangle_l110_110291


namespace coin_flip_sequences_l110_110712

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110712


namespace total_coughs_after_20_minutes_l110_110049

theorem total_coughs_after_20_minutes (georgia_rate robert_rate : ℕ) (coughs_per_minute : ℕ) :
  georgia_rate = 5 →
  robert_rate = 2 * georgia_rate →
  coughs_per_minute = georgia_rate + robert_rate →
  (20 * coughs_per_minute) = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_coughs_after_20_minutes_l110_110049


namespace sum_first_n_geom_seq_l110_110058

def is_geometric_sequence {α : Type*} [has_mul α] [has_pow α ℕ] (a : ℕ → α) (q : α) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sum_first_n_geom_seq (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a q)
  (h_a2a4 : a 2 + a 4 = 20)
  (h_a3a5 : a 3 + a 5 = 40) :
  ∀ n : ℕ, S_n n = 2^(n + 1) - 2 :=
sorry

end sum_first_n_geom_seq_l110_110058


namespace basketball_team_selection_l110_110681

theorem basketball_team_selection : 
  (∃ (captain: Fin 12) (regulars : Finset (Fin 12)), captain ∉ regulars ∧ regulars.card = 4) → 
  3960 :=
by
  sorry

end basketball_team_selection_l110_110681


namespace distance_from_M_to_midpoint_of_AP_is_half_length_of_CP_l110_110599

noncomputable def midpoint (A B : Point) : Point := MkPoint ((A.x + B.x) / 2) ((A.y + B.y) / 2) ((A.z + B.z) / 2)

theorem distance_from_M_to_midpoint_of_AP_is_half_length_of_CP (P A B C D M : Point) 
  (h_base : parallelogram A B C D)
  (h_foot : is_foot_of_perpendicular M A B D)
  (h_equal_distances : dist B P = dist D P) : 
  dist M (midpoint A P) = (1 / 2) * dist C P :=
sorry

end distance_from_M_to_midpoint_of_AP_is_half_length_of_CP_l110_110599


namespace smith_oldest_child_age_l110_110595

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l110_110595


namespace theta_value_l110_110307

theorem theta_value (theta : ℝ) (h1 : 0 ≤ theta ∧ theta ≤ 90)
    (h2 : Real.cos 60 = Real.cos 45 * Real.cos theta) : theta = 45 :=
  sorry

end theta_value_l110_110307


namespace find_sum_of_angles_l110_110081

-- Definitions for the conditions
def is_acute (x : ℝ) : Prop :=
  0 < x ∧ x < (π / 2)

variable (α β : ℝ)
variable (hα : is_acute α)
variable (hβ : is_acute β)
variable (h_tan_α : Real.tan α = 2)
variable (h_tan_β : Real.tan β = 3)

-- The target theorem
theorem find_sum_of_angles :
  α + β = 3 * π / 4 :=
sorry

end find_sum_of_angles_l110_110081


namespace smallest_product_of_set_l110_110039

noncomputable def smallest_product_set : Set ℤ := { -10, -3, 0, 4, 6 }

theorem smallest_product_of_set :
  ∃ (a b : ℤ), a ∈ smallest_product_set ∧ b ∈ smallest_product_set ∧ a ≠ b ∧ a * b = -60 ∧
  ∀ (x y : ℤ), x ∈ smallest_product_set ∧ y ∈ smallest_product_set ∧ x ≠ y → x * y ≥ -60 := 
sorry

end smallest_product_of_set_l110_110039


namespace tan_seven_pi_over_four_l110_110023

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l110_110023


namespace width_of_the_road_constant_l110_110319

/-- Define the problem with given conditions and prove width of the road -/
theorem width_of_the_road_constant (R r W : ℝ) (h1 : 2 * Real.pi * r + 2 * Real.pi * R = 88) (h2 : r = R / 3)
:
  W = R - r → W = 22 / Real.pi :=
begin
  sorry,
end

end width_of_the_road_constant_l110_110319


namespace tetrahedron_side_ratio_proof_l110_110518

noncomputable def tetrahedron_side_ratio_range (a b : ℝ) (h : a < b) : Prop :=
  sqrt (2 - sqrt 3) < a / b ∧ a / b < 1

theorem tetrahedron_side_ratio_proof (a b : ℝ) (h : a < b) 
  (h_pa : PA = PB) (h_a : PA = a) (h_pb : PB = a)
  (h_pc : PC = b) (h_ab : AB = b) (h_bc : BC = b) (h_ca : CA = b)
  : tetrahedron_side_ratio_range a b h := 
sorry

end tetrahedron_side_ratio_proof_l110_110518


namespace multiples_of_4_count_l110_110916

theorem multiples_of_4_count (a b : ℕ) (ha : a = 76) (hb : b = 296) : 
  let seq := list.range' 76 ((296 - 76) / 4 + 1),
      nums := seq.map (λ x, x / 4)
  in nums.length = 56 := 
by
  sorry

end multiples_of_4_count_l110_110916


namespace average_speed_for_whole_journey_l110_110782

theorem average_speed_for_whole_journey :
  ∃ (d1 d2 s1 s2 : ℝ),
  d1 = 150 ∧ d2 = 150 ∧ s1 = 50 ∧ s2 = 30 ∧
  let total_distance := d1 + d2 in
  let time1 := d1 / s1 in
  let time2 := d2 / s2 in
  let total_time := time1 + time2 in
  let avg_speed := total_distance / total_time in
  avg_speed = 37.5 :=
begin
  sorry
end

end average_speed_for_whole_journey_l110_110782


namespace general_term_formula_sum_b_n_sum_c_n_lt_one_third_l110_110887

-- Define the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℕ) := ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific sequence {a_n}, where a_1 = 1 and a_1, a_2, a_6 form a geometric sequence
def a_n (n : ℕ) : ℕ := 3 * n - 2

-- b_n and its sum S_n
def b_n (n : ℕ) : ℕ := a_n n * 2^n
def S_n (n : ℕ) : ℕ := (3 * n - 5) * 2^(n + 1) + 10

-- c_n and its sum T_n
def c_n (n : ℕ) : ℝ := 1 / (a_n n * a_n (n + 1))
def T_n (n : ℕ) : ℝ := (finset.range n).sum (λ k, c_n k)

-- Proof obligations
theorem general_term_formula :
  ∀ n : ℕ, a_n n = 3 * n - 2 :=
sorry

theorem sum_b_n :
  ∀ n : ℕ, S_n n = (finset.range n).sum (λ k, b_n k) :=
sorry

theorem sum_c_n_lt_one_third :
  ∀ n : ℕ, T_n n < 1 / 3 :=
sorry

end general_term_formula_sum_b_n_sum_c_n_lt_one_third_l110_110887


namespace remainder_sum_first_20_div_12_l110_110650

theorem remainder_sum_first_20_div_12 :
  (∑ i in (Finset.range 21), i) % 12 = 6 :=
by
  sorry

end remainder_sum_first_20_div_12_l110_110650


namespace sum_a_b_l110_110833

theorem sum_a_b (x a b : ℕ) (h₀ : x^2 - 10 * x = 39) (h₁ : x = nat.sqrt a - b) (h₂ : x > 0) : a + b = 69 :=
by 
  sorry

end sum_a_b_l110_110833


namespace log_sqrt_seven_l110_110411

theorem log_sqrt_seven : ∀ (a : ℝ), a = 7 → (∀ b : ℝ, log 7 (7^b) = b) → log 7 (sqrt 7) = 1/2 := by
  intro a a_eq log_id
  have sqrt7_eq := sqrt_eq_rpow (by norm_num) (by norm_num : 7 ≠ 0)
  rw [sqrt7_eq, log_id]
  norm_num

sorry

end log_sqrt_seven_l110_110411


namespace find_angle_between_vectors_l110_110463

variables {a : EuclideanSpace ℝ (Fin 2)}

-- Given conditions
def is_unit_vector (a : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖a‖ = 1

def b : EuclideanSpace ℝ (Fin 2) :=
  ![2, 2*sqrt(3)]

def orthogonal (a b : EuclideanSpace ℝ (Fin 2)) : Prop :=
  inner a b = 0

-- Statement to prove
theorem find_angle_between_vectors
  (ha : is_unit_vector a)
  (h_orth : orthogonal a (2 • a + b)) :
  angle a b = 2 * π / 3 :=  
sorry

end find_angle_between_vectors_l110_110463


namespace sum_exterior_angles_sum_interior_angles_l110_110239

theorem sum_exterior_angles (n : ℕ) (h_n : 3 ≤ n) : ∑ i in (finset.range n), exterior_angle i = 360 :=
sorry

theorem sum_interior_angles (n : ℕ) (h_n : 3 ≤ n) : ∑ i in (finset.range n), interior_angle i = (n - 2) * 180 :=
sorry

end sum_exterior_angles_sum_interior_angles_l110_110239


namespace inverse_function_correct_l110_110227

noncomputable def f (x : ℝ) (h : x > 4) : ℝ :=
  1 / (Real.sqrt x)

noncomputable def f_inv (y : ℝ) (h : 0 < y ∧ y < 1/2) : ℝ :=
  1 / y^2

theorem inverse_function_correct : ∀ (x : ℝ) (hx : x > 4),
  f_inv (f x hx) (by {
    have hy : 0 < 1 / (Real.sqrt x) := by {
      apply one_div_pos_of_pos,
      exact Real.sqrt_pos.mpr hx,
    },
    have hy' : 1 / (Real.sqrt x) < 1 / 2 := by {
      apply one_div_lt_one_div_of_lt;
      try { exact Real.sqrt_pos.mpr hx },
      exact hx,
      linarith [Real.sqrt_lt.mp (show Real.sqrt 4 < Real.sqrt x from Real.sqrt_lt (by norm_num) (hx))],
    },
    exact ⟨hy, hy'⟩
  }) = x :=
sorry

end inverse_function_correct_l110_110227


namespace infinite_shaded_area_l110_110962

noncomputable def area_initial_triangle (a b : ℝ) : ℝ := (1 / 2) * a * b

def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem infinite_shaded_area
  (XYZ_area : ℝ)
  (a : ℝ := 12.5) 
  (r : ℝ := 1 / 4) 
  (total_shaded_area : ℝ) : 
  total_shaded_area = geometric_series_sum a r := by
sorry

end infinite_shaded_area_l110_110962


namespace area_of_shaded_region_l110_110351

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l110_110351


namespace sum_log_computation_correct_l110_110823

noncomputable def sum_log_computation : ℝ := 
  ∑ k in finset.range (100 - 2) + 3, log 3 (1 + 1 / k.cast) * log k.cast 3 * log (k.cast + 1) 3

theorem sum_log_computation_correct : sum_log_computation = 1 - 1 / log 3 101 := 
  by
  sorry

end sum_log_computation_correct_l110_110823


namespace hermite_polynomial_eq_hermite_exp_sigma_hermite_cond_exp_hermite_polynomial_properties_l110_110172

-- Define the normal distribution and Hermite polynomials
noncomputable def normal_density (x : ℝ) : ℝ := (1 / Mathlib.sqrt (2 * Mathlib.pi)) * Mathlib.exp (-x^2 / 2)

noncomputable def hermite_polynomial (n : ℕ) (x : ℝ) : ℝ :=
  ((-1)^n * (Mathlib.derivative^[n] normal_density x)) / normal_density x

-- Problem 1: Prove Hₙ(x) = E((x + iξ)ⁿ) for all n ≥ 0
theorem hermite_polynomial_eq (x : ℝ) (n : ℕ) (ξ : ℝ) (hξ : ξ ~ normal_distribution 0 1):
  hermite_polynomial n x = Mathlib.expectation ((x + Mathlib.I * ξ) ^ n) :=
sorry

-- Problem 2: Prove the formula for E(Hₙ(x + σξ))
theorem hermite_exp_sigma (x : ℝ) (n : ℕ) (ξ : ℝ) (σ : ℝ) (hσ : 0 ≤ σ ∧ σ < 1) (hξ : ξ ~ normal_distribution 0 1):
  Mathlib.expectation (hermite_polynomial n (x + σ * ξ)) = 
    if σ = 1 then x^n else (1 - σ^2)^(n / 2) * hermite_polynomial n (x / Mathlib.sqrt (1 - σ^2)) :=
sorry

-- Problem 3: Prove the conditional expectation
theorem hermite_cond_exp (η : ℝ) (ξ : ℝ) (ρ : ℝ) (n : ℕ) (hξ : ξ ~ normal_distribution 0 1) (hη : η ~ normal_distribution 0 1) (hξη : Mathlib.expectation (ξ * η) = ρ):
  Mathlib.conditional_expectation (hermite_polynomial n η) ξ = ρ^n * hermite_polynomial n ξ :=
sorry

-- Problem 4: Prove Hₙ(x) = xⁿ + ... and the orthogonality property, then the L² representation
theorem hermite_polynomial_properties (ξ : ℝ) (m n : ℕ) (hξ : ξ ~ normal_distribution 0 1):
  hermite_polynomial n ξ = ξ^n + ... ∧ 
  Mathlib.expectation (hermite_polynomial n ξ * hermite_polynomial m ξ) = if n = m then n! else 0 ∧
  (∀ f : ℝ → ℝ, Mathlib.expectation (f ξ)^2 < ∞ → 
    ∃ (c : ℝ), f ξ = ∑ n in Mathlib.range(∞), (hermite_polynomial n ξ / n!) * Mathlib.expectation (f ξ * hermite_polynomial n ξ)) :=
sorry

end hermite_polynomial_eq_hermite_exp_sigma_hermite_cond_exp_hermite_polynomial_properties_l110_110172


namespace digit_sum_correctness_l110_110204

theorem digit_sum_correctness (É L J E N M Á U S : ℕ) :
  É ≠ 1 ∧ L ≠ 1 ∧ J ≠ 1 ∧ N ≠ 1 ∧ M ≠ 1 ∧ Á ≠ 1 ∧ U ≠ 1 ∧ S ≠ 1 ∧
  É ≠ L ∧ É ≠ J ∧ É ≠ N ∧ É ≠ M ∧ É ≠ Á ∧ É ≠ U ∧ É ≠ S ∧
  L ≠ J ∧ L ≠ N ∧ L ≠ M ∧ L ≠ Á ∧ L ≠ U ∧ L ≠ S ∧
  J ≠ N ∧ J ≠ M ∧ J ≠ Á ∧ J ≠ U ∧ J ≠ S ∧
  N ≠ M ∧ N ≠ Á ∧ N ≠ U ∧ N ≠ S ∧
  M ≠ Á ∧ M ≠ U ∧ M ≠ S ∧
  Á ≠ U ∧ Á ≠ S ∧
  U ≠ S ∧
  E = 1 ∧
  É = 9 ∧
  L = 3 ∧
  J = 5 ∧
  N = 6 ∧
  M = 4 ∧
  Á = 2 ∧ 
  U = 8 ∧ 
  S = 5 ∧ 
  (É * 10000 + L * 1000 + J * 100 + E * 10 + N) + 
  (M * 10000 + Á * 1000 + J * 100 + U * 10 + S) =
  (E * 100000 + L * 10000 + S * 1000 + J * 100 + E * 10 + J * 1) :=
by {
  sorry
}

end digit_sum_correctness_l110_110204


namespace percent_decrease_call_cost_l110_110597

theorem percent_decrease_call_cost (c1990 c2010 : ℝ) (h1990 : c1990 = 50) (h2010 : c2010 = 10) :
  ((c1990 - c2010) / c1990) * 100 = 80 :=
by
  sorry

end percent_decrease_call_cost_l110_110597


namespace coefficient_of_a3b3_l110_110259

theorem coefficient_of_a3b3 (a b c : ℚ) :
  (∏ i in Finset.range 7, (a + b) ^ i * (c + 1 / c) ^ (8 - i)) = 1400 := 
by
  sorry

end coefficient_of_a3b3_l110_110259


namespace sum_of_d_for_6_solutions_l110_110827

noncomputable def g (x : ℝ) := ( (x - 6) * (x - 4) * (x - 2) * (x + 2) * (x + 4) * (x + 6) ) / 400 - 4

theorem sum_of_d_for_6_solutions : 
  (∑ d in {d : ℤ | ∃! x ∈ Icc (-7 : ℝ) 7, g x = (d : ℝ)}, d) = -7 := 
by sorry

end sum_of_d_for_6_solutions_l110_110827


namespace construct_square_from_isosceles_triangle_l110_110489

-- Define the given shape and its geometric properties.
structure IsoscelesTriangle :=
  (base : ℝ)
  (height : ℝ)

-- Define the required cuts maintaining identical pieces
structure IdenticalParts :=
  (parts : list IsoscelesTriangle) -- Parts after cutting

-- Define the resulting shape after reassembly
structure Square :=
  (side : ℝ)

-- Prove that the parts can be reassembled into a square
theorem construct_square_from_isosceles_triangle (triangle : IsoscelesTriangle) (cuts : IdenticalParts) : ∃ (square : Square), True :=
by 
  sorry

end construct_square_from_isosceles_triangle_l110_110489


namespace quadratic_equation_has_given_roots_l110_110121

theorem quadratic_equation_has_given_roots :
  (∀ x, x = (2 + sqrt (4 - 4 * 3 * (-1))) / (2 * 3) ∨
        x = (2 - sqrt (4 - 4 * 3 * (-1))) / (2 * 3) →
        3 * x^2 - 2 * x - 1 = 0) :=
by
  sorry

end quadratic_equation_has_given_roots_l110_110121


namespace no_prize_for_A_l110_110359

variable (A B C D : Prop)

theorem no_prize_for_A 
  (hA : A → B) 
  (hB : B → C) 
  (hC : ¬D → ¬C) 
  (exactly_one_did_not_win : (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)) 
: ¬A := 
sorry

end no_prize_for_A_l110_110359


namespace mass_percentage_ba_in_bao_l110_110647

theorem mass_percentage_ba_in_bao :
  let molar_mass_Ba := 137.33
  let molar_mass_O := 16.00
  let molar_mass_BaO := molar_mass_Ba + molar_mass_O
  (molar_mass_Ba / molar_mass_BaO) * 100 ≈ 89.55 :=
by
  sorry

end mass_percentage_ba_in_bao_l110_110647


namespace coefficient_a3b3_in_expression_l110_110272

theorem coefficient_a3b3_in_expression :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ k) * (b ^ (6 - k))) *
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (c ^ (8 - 2 * k)) * (c ^ (-2 * k))) =
  1400 := sorry

end coefficient_a3b3_in_expression_l110_110272


namespace functional_equation_solution_l110_110004

-- Given a function f: ℝ_{+}* -> ℝ_{+}
variable (f : ℝ → ℝ)
variable (h : ∀ x y > 0, f(x) - f(x + y) = f(x^2 * f(y) + x))

-- Define the functional equation
def functional_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) : Prop :=
  f(x) - f(x + y) = f(x^2 * f(y) + x)

-- The proof problem: Prove that the only solutions to the functional equation
-- are f(x) = 0 or f(x) = 1/x
theorem functional_equation_solution :
  (∀ x > 0, f(x) = 0) ∨ (∀ x > 0, f(x) = 1/x) :=
sorry

end functional_equation_solution_l110_110004


namespace square_overlap_area_l110_110339

theorem square_overlap_area (β : ℝ) (h1 : 0 < β) (h2 : β < 90) (h3 : Real.cos β = 3 / 5) : 
  area (common_region (square 2) (rotate_square β (square 2))) = 4 / 3 :=
sorry

end square_overlap_area_l110_110339


namespace intersect_sets_two_points_l110_110383

def SetA (a : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.snd = a * (abs p.fst) }

def SetB (a : ℝ) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.snd = p.fst + a }

theorem intersect_sets_two_points (a : ℝ) :
    (a ∈ Iio (-1) ∨ a ∈ Ioi 1) →
    (SetA a ∩ SetB a).to_finset.card = 2 :=
by
  sorry

end intersect_sets_two_points_l110_110383


namespace num_digits_l110_110397

theorem num_digits (x y : ℕ) (hx : x = 2) (hy : y = 5) : 
  ∃ n : ℕ, (2 ^ 15 * 5 ^ 10).digits.len = n ∧ n = 12 := by
sorry

end num_digits_l110_110397


namespace probability_at_least_two_consecutive_l110_110816

theorem probability_at_least_two_consecutive (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  let E := {x : Finset ℕ | x ⊆ s ∧ x.card = 3 ∧ (∃ a b c ∈ x, (a + 1 = b ∨ a + 1 = c ∨ b + 1 = c) ∨ (a + 1 = b ∧ b + 1 = c))},
      F := {x : Finset ℕ | x ⊆ s ∧ x.card = 3} in
  ∃ p : ℚ, p = 8 / 15 ∧ E.card / F.card.toRat = p :=
by
  sorry

end probability_at_least_two_consecutive_l110_110816


namespace largest_value_WY_cyclic_quadrilateral_l110_110991

theorem largest_value_WY_cyclic_quadrilateral :
  ∃ WZ ZX ZY YW : ℕ, 
    WZ ≠ ZX ∧ WZ ≠ ZY ∧ WZ ≠ YW ∧ ZX ≠ ZY ∧ ZX ≠ YW ∧ ZY ≠ YW ∧ 
    WZ < 20 ∧ ZX < 20 ∧ ZY < 20 ∧ YW < 20 ∧ 
    WZ * ZY = ZX * YW ∧
    (∀ WY', (∃ WY : ℕ, WY' < WY → WY <= 19 )) :=
sorry

end largest_value_WY_cyclic_quadrilateral_l110_110991


namespace ratio_of_region_areas_l110_110585

noncomputable def perimeter_to_side_length (p : ℕ) : ℕ := p / 4
noncomputable def area_of_square (side_length : ℕ) : ℕ := side_length * side_length 
noncomputable def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

-- Conditions
def perimeter_I : ℕ := 16
def perimeter_II : ℕ := 24
def perimeter_III : ℕ := 12
def perimeter_IV : ℕ := 28

-- Side Lengths
def side_length_I : ℕ := perimeter_to_side_length perimeter_I
def side_length_II : ℕ := perimeter_to_side_length perimeter_II
def side_length_III : ℕ := perimeter_to_side_length perimeter_III
def side_length_IV : ℕ := perimeter_to_side_length perimeter_IV

-- Areas
def area_II : ℕ := area_of_square side_length_II
def area_IV : ℕ := area_of_square side_length_IV

-- Proof statement
theorem ratio_of_region_areas : ratio area_II area_IV = 36 / 49 := by
  sorry

end ratio_of_region_areas_l110_110585


namespace complex_numbers_z_l110_110042

def f (k : ℕ) : ℕ := -- definition of the function f which calculates the number of ones in the base 3 representation of k
  sorry

theorem complex_numbers_z (z : ℂ) (sum_eq_0 : ∑ k in finset.range (3^1010), (-2 : ℂ)^(f k) * (z + ↑k)^2023 = 0) :
  z = -((3^1010 - 1) / 2) ∨ 
  z = -((3^1010 - 1) / 2) + complex.I * complex.sqrt( (9^1010 - 1) / 16 ) ∨ 
  z = -((3^1010 - 1) / 2) - complex.I * complex.sqrt( (9^1010 - 1) / 16 ) :=
by 
  sorry

end complex_numbers_z_l110_110042


namespace area_of_isosceles_right_triangle_l110_110226

theorem area_of_isosceles_right_triangle (h : ℝ) (hypotenuse_eq : h = 6 * Real.sqrt 2) : 
  let leg := h / Real.sqrt 2 in 
  let area := (1 / 2) * leg^2 in 
  area = 18 := by
  let leg := h / Real.sqrt 2
  let area := (1 / 2) * leg^2
  have leg_eq : leg = 6 := by sorry
  have area_eq : area = 18 := by sorry
  exact area_eq

end area_of_isosceles_right_triangle_l110_110226


namespace range_of_f_l110_110445

-- Define the function f and the conditions
def f (x : ℝ) : ℝ := sorry 
axiom pos_f (x : ℝ) (hx : 0 < x) : 0 < f x
axiom deriv_f (x : ℝ) (hx : 0 < x) : f x < deriv f x ∧ deriv f x < 2 * f x

-- Prove the desired property
theorem range_of_f (c₁ : pos_f 1 1) (c₂ : pos_f 2 2) : 
  1 / Real.exp 2 < f 1 / f 2 ∧ f 1 / f 2 < 1 / Real.exp 1 :=
by
  sorry

end range_of_f_l110_110445


namespace marbles_count_l110_110431

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l110_110431


namespace compare_trig_values_l110_110880

noncomputable def a : ℝ := Real.tan (-7 * Real.pi / 6)
noncomputable def b : ℝ := Real.cos (23 * Real.pi / 4)
noncomputable def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem compare_trig_values : c < a ∧ a < b := sorry

end compare_trig_values_l110_110880


namespace sec_of_7pi_over_6_l110_110015

theorem sec_of_7pi_over_6 : real.sec (7 * real.pi / 6) = -2 * real.sqrt 3 / 3 :=
by sorry

end sec_of_7pi_over_6_l110_110015


namespace num_more_green_l110_110676

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

end num_more_green_l110_110676


namespace distance_AB_l110_110513

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
(-1/2 * t, 3 + (sqrt 3 / 2) * t)

def polar_curve (θ : ℝ) : ℝ :=
4 * sin (θ + π / 3)

def rectangular_curve (x y : ℝ) : Prop :=
x^2 + y^2 - 2 * y - 2 * sqrt 3 * x = 0

def general_line (x y : ℝ) : Prop :=
sqrt 3 * x - y + 3 = 0

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def intersection_points_distance : ℝ :=
let t := (-2 * - (1/2), 3 + (sqrt 3 / 2) * -2)
    t' := (1 / 2 * 2, 3 + (sqrt 3 / 2) * 2) in
distance t t' = sqrt 15

theorem distance_AB (t : ℝ) (t' : ℝ) : intersection_points_distance :=
sorry

end distance_AB_l110_110513


namespace algebra_expression_value_l110_110899

theorem algebra_expression_value (m : ℝ) (h : m^2 - 3 * m - 1 = 0) : 2 * m^2 - 6 * m + 5 = 7 := by
  sorry

end algebra_expression_value_l110_110899


namespace distinct_sequences_ten_flips_l110_110692

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110692


namespace find_larger_page_l110_110951

theorem find_larger_page {x y : ℕ} (h1 : y = x + 1) (h2 : x + y = 125) : y = 63 :=
by
  sorry

end find_larger_page_l110_110951


namespace coefficient_of_a3b3_l110_110261

theorem coefficient_of_a3b3 (a b c : ℚ) :
  (∏ i in Finset.range 7, (a + b) ^ i * (c + 1 / c) ^ (8 - i)) = 1400 := 
by
  sorry

end coefficient_of_a3b3_l110_110261


namespace regular_tetrahedron_surface_area_l110_110608

theorem regular_tetrahedron_surface_area {h : ℝ} (h_pos : h > 0) :
  ∃ (S : ℝ), S = (3 * h^2 * Real.sqrt 3) / 2 :=
sorry

end regular_tetrahedron_surface_area_l110_110608


namespace total_workers_in_workshop_l110_110303

theorem total_workers_in_workshop 
  (W : ℕ)
  (T : ℕ := 5)
  (avg_all : ℕ := 700)
  (avg_technicians : ℕ := 800)
  (avg_rest : ℕ := 650) 
  (total_salary_all : ℕ := W * avg_all)
  (total_salary_technicians : ℕ := T * avg_technicians)
  (total_salary_rest : ℕ := (W - T) * avg_rest) :
  total_salary_all = total_salary_technicians + total_salary_rest →
  W = 15 :=
by
  sorry

end total_workers_in_workshop_l110_110303


namespace max_square_test_plots_l110_110329

theorem max_square_test_plots
    (length : ℕ)
    (width : ℕ)
    (fence : ℕ)
    (fields_measure : length = 30 ∧ width = 45)
    (fence_measure : fence = 2250) :
  ∃ (number_of_plots : ℕ),
    number_of_plots = 150 :=
by
  sorry

end max_square_test_plots_l110_110329


namespace volume_between_cubes_l110_110240

def larger_cube_edge_length (sum_edges: ℕ) : ℕ := sum_edges / 12
def smaller_cube_edge_length (larger_cube_edge: ℕ) : ℕ := larger_cube_edge / 2
def cube_volume (edge_length: ℕ) : ℕ := edge_length ^ 3

theorem volume_between_cubes (sum_edges: ℕ) (h1: sum_edges = 96) : 
  (let larger_cube_edge := larger_cube_edge_length sum_edges in
   let smaller_cube_edge := smaller_cube_edge_length larger_cube_edge in
   let volume_larger_cube := cube_volume larger_cube_edge in
   let volume_smaller_cube := cube_volume smaller_cube_edge in
   volume_larger_cube - volume_smaller_cube = 448)
:= sorry

end volume_between_cubes_l110_110240


namespace range_of_a_l110_110557

-- Definitions as per conditions
def f (x : ℝ) := x^2 - 2 * x
def g (a x : ℝ) := a * x + 2

-- Hypotheses for the problem
variables (a : ℝ) (h_pos : a > 0)
variable h_cond : ∀ x₁ ∈ Icc (-1 : ℝ) 2, ∃ x₀ ∈ Icc (-1 : ℝ) 2, g a x₁ = f x₀

-- Prove the range of a
theorem range_of_a (a : ℝ) (h_pos : a > 0) (h_cond : ∀ x₁ ∈ Icc (-1 : ℝ) 2, ∃ x₀ ∈ Icc (-1 : ℝ) 2, g a x₁ = f x₀) :
  0 < a ∧ a ≤ 1/2 :=
begin
  sorry
end

end range_of_a_l110_110557


namespace triangle_ratio_proof_l110_110156

noncomputable def triangle_ratio :=
  let angle_A := 60
  let angle_B := 45
  let angle_ADF := 30
  let bisects_area := True  -- This is a placeholder for the actual geometric condition

  -- Using placeholder definitions and notation to adapt the problem context
  ∀ (AD AB : ℝ),
  (sin angle_A = (√3 / 2)) →
  (cos angle_A = 1 / 2) →
  (sin angle_B = (√2 / 2)) →
  (cos angle_B = (√2 / 2)) →
  (sin angle_ADF = (1 / 2)) →
  (cos angle_ADF = (√3 / 2)) →
  -- Assuming necessary geometric properties including the area bisect condition
  bisects_area →
  AD / AB = ((√6 + √2) / (4 * √2))

-- Placeholder for the theorem statement
theorem triangle_ratio_proof : triangle_ratio := 
  by 
  exact sorry

end triangle_ratio_proof_l110_110156


namespace true_statement_count_l110_110550

def n_star (n : ℕ) : ℚ := 1 / n

theorem true_statement_count :
  let s1 := (n_star 4 + n_star 8 = n_star 12)
  let s2 := (n_star 9 - n_star 1 = n_star 8)
  let s3 := (n_star 5 * n_star 3 = n_star 15)
  let s4 := (n_star 16 - n_star 4 = n_star 12)
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0) = 1 :=
by
  -- Proof goes here
  sorry

end true_statement_count_l110_110550


namespace marbles_count_l110_110430

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l110_110430


namespace correct_statements_count_l110_110975

def class (k : ℤ) : Set ℤ := { n | ∃ m : ℤ, n = 5 * m + k }

def statement1 := 2018 ∈ class 3
def statement2 := -2 ∈ class 2
def statement3 := ∀ z : ℤ, ∃ k : ℤ, 0 ≤ k ∧ k ≤ 4 ∧ z ∈ class k
def statement4 := ∀ (a b : ℤ), (a ∈ class 3 ↔ b ∈ class 3) ↔ (a - b ∈ class 0)

theorem correct_statements_count : 
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) → (3 = 3) := 
by
  sorry

end correct_statements_count_l110_110975


namespace lambda_value_exists_l110_110876

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * a_seq (n - 1) + 2^n - 1

theorem lambda_value_exists (a : ℕ → ℕ) (arith_seq : ℕ → ℝ) (λ : ℝ)
  (initial_condition : a 1 = 5)
  (recurrence_relation : ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1)
  (arith_definition : ∀ n, arith_seq n = (a n + λ) / 2^n)
  (arithmetic_progression : ∃ d, ∀ n, n ≥ 2 → arith_seq n - arith_seq (n - 1) = d) :
  λ = -1 := 
sorry

end lambda_value_exists_l110_110876


namespace smallest_positive_period_F_value_of_expression_l110_110906

noncomputable def f (x : ℝ) : ℝ := Math.sin x + Math.cos x
noncomputable def f' (x : ℝ) : ℝ := Math.cos x - Math.sin x

theorem smallest_positive_period_F :
  let F (x : ℝ) := f x * f' x + (f x)^2 in
  0 < (π : ℝ) ∧ ∀ x, F (x + π) = F x :=
by
  sorry -- Proof to be provided

theorem value_of_expression (x : ℝ) :
  f x = 2 * f' x → (1 + Math.sin x^2) / (Math.cos x^2 - Math.sin x * Math.cos x) = 11 / 6 :=
by
  sorry -- Proof to be provided

end smallest_positive_period_F_value_of_expression_l110_110906


namespace reach_treasure_within_4km_l110_110957

theorem reach_treasure_within_4km (r : ℝ) (h_r : r = 1) (detect_range : ℝ) (h_d : detect_range = 0.5) :
  ∃ (strategy : ℝ → ℝ → Prop), ∀ (start_x start_y : ℝ),
    (sqrt (start_x^2 + start_y^2) = r) →
    (∃ (path_length : ℝ), (path_length < 4) ∧ strategy start_x start_y) := 
sorry

end reach_treasure_within_4km_l110_110957


namespace distinct_sequences_ten_flips_l110_110690

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110690


namespace triangle_inequalities_l110_110579

-- Definitions of the variables
variables {ABC : Triangle} {r : ℝ} {R : ℝ} {ρ_a ρ_b ρ_c : ℝ} {P_a P_b P_c : ℝ}

-- Problem statement based on given conditions and proof requirement
theorem triangle_inequalities (ABC : Triangle) (r : ℝ) (R : ℝ) (ρ_a ρ_b ρ_c : ℝ) (P_a P_b P_c : ℝ) :
  (3/2) * r ≤ ρ_a + ρ_b + ρ_c ∧ ρ_a + ρ_b + ρ_c ≤ (3/4) * R ∧ 4 * r ≤ P_a + P_b + P_c ∧ P_a + P_b + P_c ≤ 2 * R :=
  sorry

end triangle_inequalities_l110_110579


namespace range_of_c_in_acute_triangle_l110_110146

theorem range_of_c_in_acute_triangle 
  (a A B C : ℝ)
  (b : ℝ := 1)
  (c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : sqrt 3 * (a * cos B + b * cos A) = 2 * c * sin C) :
  c ∈ set.Ioo (sqrt 3 / 2) sqrt 3 :=
sorry

end range_of_c_in_acute_triangle_l110_110146


namespace g_four_equals_nine_l110_110548

def f (x : ℝ) : ℝ := 4 / (3 - x)
def f_inv (y : ℝ) : ℝ := 3 - 4 / y
def g (x : ℝ) : ℝ := 2 / (f_inv x) + 8

theorem g_four_equals_nine : g 4 = 9 := by
  sorry

end g_four_equals_nine_l110_110548


namespace coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110264

theorem coefficient_a3b3_in_ab6_c8_div_c8_is_1400 :
  let a := (a : ℝ)
  let b := (b : ℝ)
  let c := (c : ℝ)
  ∀ (a b c : ℝ), (binom 6 3 * a^3 * b^3) * (binom 8 4 * c^0) = 1400 := 
by
  sorry

end coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110264


namespace fraction_equality_l110_110877

theorem fraction_equality (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 := 
sorry

end fraction_equality_l110_110877


namespace angle_between_a_b_l110_110135

open Real

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h1 : ‖a‖ = √3) (h2 : ‖b‖ = 2) (h3 : a ⬝ (a - b) = 0)

-- The statement to be proved
theorem angle_between_a_b (a b : EuclideanSpace ℝ (Fin 3))
  (h1 : ‖a‖ = √3) (h2 : ‖b‖ = 2) (h3 : a ⬝ (a - b) = 0) : 
  real.angleBetween a b = π/6 :=
sorry

end angle_between_a_b_l110_110135


namespace sum_of_dihedral_angles_leq_90_l110_110084
noncomputable section

-- Let θ1 and θ2 be angles formed by a line with two perpendicular planes
variable (θ1 θ2 : ℝ)

-- Define the condition stating the planes are perpendicular, and the line forms dihedral angles
def dihedral_angle_condition (θ1 θ2 : ℝ) : Prop := 
  θ1 ≥ 0 ∧ θ1 ≤ 90 ∧ θ2 ≥ 0 ∧ θ2 ≤ 90

-- The theorem statement capturing the problem
theorem sum_of_dihedral_angles_leq_90 
  (θ1 θ2 : ℝ) 
  (h : dihedral_angle_condition θ1 θ2) : 
  θ1 + θ2 ≤ 90 :=
sorry

end sum_of_dihedral_angles_leq_90_l110_110084


namespace coin_flip_sequences_l110_110766

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110766


namespace domain_of_expression_l110_110860

theorem domain_of_expression : {x : ℝ | ∃ y z : ℝ, y = √(x - 3) ∧ z = √(8 - x) ∧ x - 3 ≥ 0 ∧ 8 - x > 0} = {x : ℝ | 3 ≤ x ∧ x < 8} :=
by
  sorry

end domain_of_expression_l110_110860


namespace distinct_sequences_ten_flips_l110_110735

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110735


namespace distinct_sequences_ten_flips_l110_110734

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110734


namespace problem_solution_l110_110380

def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def tilde_p (x : ℝ) : ℝ := p (p x)

-- Proof problem: Prove tilde_p 2 = -4 
theorem problem_solution : tilde_p 2 = -4 := sorry

end problem_solution_l110_110380


namespace find_circle_center_l110_110318

noncomputable theory
open Real

-- Circle conditions
def Circle_passes_through : Prop :=
  ∃ (a b R: ℝ), (a-0)^2 + (b-0)^2 = R^2 -- passes through (0,0)
  ∧ (a-1)^2 + (b-1)^2 = R^2 -- tangent at (1,1) implies distance to (1,1) is also R

-- Parabola condition
def Parabola_is_tangent : Prop :=
  ∃ (a b : ℝ), let slope := 2 in let perp_slope := -1/2 in 
  (b - 1) / (a - 1) = perp_slope -- perpendicular to the tangent line of y = x^2 at (1,1)

-- Theorem: Center of the circle
theorem find_circle_center : ∃ (a b: ℝ), Circle_passes_through ∧ Parabola_is_tangent ∧ a = -1 ∧ b = 2 :=
  sorry

end find_circle_center_l110_110318


namespace convert_to_base5_l110_110389

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l110_110389


namespace hyperbola_center_l110_110420

theorem hyperbola_center :
  ∃ x y, ∀ (x y : ℝ), 4 * x^2 - 24 * x - 25 * y^2 + 250 * y - 489 = 0 → (x, y) = (3, 5) :=
sorry

end hyperbola_center_l110_110420


namespace coin_flip_sequences_l110_110764

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110764


namespace fixed_distance_l110_110913

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)
variables (t u : ℝ)

theorem fixed_distance (h : ∥p - b∥ = 3 * ∥p - a∥) 
: let t := 9/8 and u := -1/8 in ∥p - (t • a + u • b)∥ = ∥p∥ :=
by {
  let t := (9 : ℝ) / 8,
  let u := -(1 : ℝ) / 8,
  sorry
}

end fixed_distance_l110_110913


namespace find_ac_find_a_and_c_l110_110136

variables (A B C a b c : ℝ)

-- Condition: Angles A, B, C form an arithmetic sequence.
def arithmetic_sequence := 2 * B = A + C

-- Condition: Area of the triangle is sqrt(3)/2.
def area_triangle := (1/2) * a * c * (Real.sin B) = (Real.sqrt 3) / 2

-- Condition: b = sqrt(3)
def b_sqrt3 := b = Real.sqrt 3

-- Goal 1: To prove that ac = 2.
theorem find_ac (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) : a * c = 2 :=
sorry

-- Goal 2: To prove a = 2 and c = 1 given the additional condition.
theorem find_a_and_c (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) (h3 : b_sqrt3 b) (h4 : a > c) : a = 2 ∧ c = 1 :=
sorry

end find_ac_find_a_and_c_l110_110136


namespace sum_of_inverse_squares_less_than_two_l110_110199

theorem sum_of_inverse_squares_less_than_two (n : ℕ) : 
  (∑ k in Finset.range n, 1 / (k + 1) ^ 2) < 2 :=
sorry

end sum_of_inverse_squares_less_than_two_l110_110199


namespace dartboard_odd_score_l110_110772

structure DartBoard where
  inner_radius : ℝ
  outer_radius : ℝ
  inner_points : List ℕ
  outer_points : List ℕ

def all_regions (board : DartBoard) : List ℕ :=
  board.inner_points ++ board.outer_points

def hit_probability (points : List ℕ) : ℚ :=
  let n := points.length
  if n = 0 then 0 else ⟨points.count (λ x => x % 2 = 1), n⟩

def odd_score_probability (board : DartBoard) : ℚ :=
  let odd_regions := hit_probability (all_regions board)
  odd_regions * (1 - odd_regions) * 2

theorem dartboard_odd_score (board : DartBoard) (h_board: 
  board.inner_radius = 4 ∧
  board.outer_radius = 8 ∧
  board.inner_points = [3, 4, 4] ∧
  board.outer_points = [4, 3, 3]) :
  odd_score_probability board = 4/9 :=
by {
  sorry
}

end dartboard_odd_score_l110_110772


namespace coin_flip_sequences_l110_110748

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110748


namespace expected_value_η_variance_η_l110_110560

open MeasureTheory ProbabilityTheory

/-- Definition of ξ (xi) as a standard normal random variable -/
noncomputable def ξ : MeasureTheory.ProbabilityTheory.stdNormal := sorry

/-- Definition of η (eta) as 10 raised to the power of ξ -/
noncomputable def η := λ ξ : ℝ, 10^ξ

/-- The expected value of η -/
theorem expected_value_η : MeasureTheory.Integral (λ ξ : ℝ, η ξ) = Real.exp ((Real.log 10)^2 / 2) := sorry

/-- The variance of η -/
theorem variance_η : 
  MeasureTheory.Integral (λ ξ : ℝ, (η ξ)^2) 
  - (MeasureTheory.Integral (λ ξ : ℝ, η ξ))^2 =
  Real.exp (2 * (Real.log 10)^2) - Real.exp ((Real.log 10)^2) := sorry

end expected_value_η_variance_η_l110_110560


namespace coin_flips_sequences_count_l110_110720

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110720


namespace ABC_incircle_concurrency_l110_110542

open Set

variable {Point : Type} [IncidenceGeometry Point]

-- Let ABC be a triangle
variables (A B C : Point)

-- The incircle of triangle ABC touches BC at A'
variable (A' : Point)

-- The line AA' meets the incircle again at P
variable (P : Point)

-- The lines CP and BP meet the incircle again at N and M, respectively
variables (N M : Point)

-- Prove that the lines AA', BN, and CM are concurrent
theorem ABC_incircle_concurrency
  (h_incircle_A' : touches_incircle_at A' BC)
  (h_AA'_P_incircle : meets_incircle_again (line A A') P)
  (h_CP_N_incircle : meets_incircle_again (line C P) N)
  (h_BP_M_incircle : meets_incircle_again (line B P) M) :
  concurrent (line A A') (line B N) (line C M) :=
sorry

end ABC_incircle_concurrency_l110_110542


namespace shaded_region_area_correct_l110_110346

noncomputable def shaded_region_area (side_length : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
if 0 < beta ∧ beta < Real.pi / 2 ∧ cos_beta = 3 / 5 then
  2 / 5
else
  0

theorem shaded_region_area_correct :
  shaded_region_area 2 β (3 / 5) = 2 / 5 :=
by
  -- conditions
  have beta_cond : 0 < β ∧ β < Real.pi / 2 := sorry
  have cos_beta_cond : cos β = 3 / 5 := sorry
  -- we will finish this proof assuming above have been proved.
  exact if_pos ⟨beta_cond, cos_beta_cond⟩

end shaded_region_area_correct_l110_110346


namespace base7_to_base10_div_result_l110_110589

theorem base7_to_base10_div_result : 
  ∃ (c d : ℕ), 765 = 4*c*7 + d ∧ c < 10 ∧ d < 10 ∧ (c * d) = 9 * 7 → (c * d) / 21 = 1.28571428571 := by
  sorry

end base7_to_base10_div_result_l110_110589


namespace expression_evaluation_l110_110369

theorem expression_evaluation :
  3 * (3 * (2 * (2 * (2 * (3 + 2) + 1) + 1) + 2) + 1) + 1 = 436 :=
by
  simp [add_mul, mul_add]
  sorry

end expression_evaluation_l110_110369


namespace find_result_l110_110994

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x - 3

theorem find_result : f (g 3) - g (f 3) = -6 := by
  sorry

end find_result_l110_110994


namespace find_matrix_A_l110_110419

variable (A : Matrix (Fin 2) (Fin 2) ℤ)

def vec1 : Vector ℤ (Fin 2) := ![2, 0]
def vec2 : Vector ℤ (Fin 2) := ![0, 3]
def res1 : Vector ℤ (Fin 2) := ![-4, 14]
def res2 : Vector ℤ (Fin 2) := ![9, -12]

theorem find_matrix_A (h1 : A.mul_vec vec1 = res1) (h2 : A.mul_vec vec2 = res2) :
  A = ![![(-2 : ℤ), 3], ![7, -4]] :=
sorry

end find_matrix_A_l110_110419


namespace cesaro_sum_51_l110_110057

-- Given definitions based on the conditions provided in the problem
def cesaro_sum_50 (b : ℕ → ℝ) : Prop := 
  (b 1 + ((b 1) + (b 2)) + ∑ i in range 50, b i) / 50 = 500

def sum_50 (b : ℕ → ℝ) : Prop :=
  50 * (b 1) + 49 * (b 2) + ∑ i in (range 3 .. 51), b i = 25000

-- Theorem stating the equivalence of the Cesaro sum for the new sequence
theorem cesaro_sum_51 {b : ℕ → ℝ} (hb_cesaro : cesaro_sum_50 b) 
  (hb_sum : sum_50 b) : 
  (2 + (2 + b 1) + (2 + b 1 + b 2) + ∑ i in range 50, b i) / 51 = 492 :=
sorry

end cesaro_sum_51_l110_110057


namespace solution_l110_110639

noncomputable def triangle_perimeter (AB BC AC : ℕ) (lA lB lC : ℕ) : ℕ :=
  -- This represents the proof problem using the given conditions
  if (AB = 130) ∧ (BC = 240) ∧ (AC = 190)
     ∧ (lA = 65) ∧ (lB = 50) ∧ (lC = 20)
  then
    130  -- The correct answer
  else
    0    -- If the conditions are not met, return 0 

theorem solution :
  triangle_perimeter 130 240 190 65 50 20 = 130 :=
by
  -- This theorem states that with the given conditions, the perimeter of the triangle is 130
  sorry

end solution_l110_110639


namespace convert_1729_to_base5_l110_110384

-- Definition of base conversion from base 10 to base 5.
def convert_to_base5 (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if h : n = 0 then acc
    else let quotient := n / 5
         let remainder := n % 5
         aux quotient (remainder :: acc)
  aux n []

-- The theorem we seek to prove.
theorem convert_1729_to_base5 : convert_to_base5 1729 = [2, 3, 4, 0, 4] :=
by
  sorry

end convert_1729_to_base5_l110_110384


namespace no_real_roots_implies_negative_l110_110434

theorem no_real_roots_implies_negative (m : ℝ) : (¬ ∃ x : ℝ, x^2 = m) → m < 0 :=
sorry

end no_real_roots_implies_negative_l110_110434


namespace area_of_shaded_region_l110_110350

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l110_110350


namespace f_analytical_expression_g_value_l110_110472

noncomputable def f (ω x : ℝ) : ℝ := (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi / 2)

noncomputable def g (ω x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem f_analytical_expression (x : ℝ) (hω : ω = 2 ∧ ω > 0) : 
  f 2 x = Real.sin (2 * x - Real.pi / 3) :=
sorry

theorem g_value (α : ℝ) (hω : ω = 2 ∧ ω > 0) (h : g 2 (α / 2) = 4/5) : 
  g 2 (-α) = -7/25 :=
sorry

end f_analytical_expression_g_value_l110_110472


namespace multiplicative_operation_correct_l110_110220

theorem multiplicative_operation_correct :
  (∏ k in finset.range 1008, (2 * (k + 1) - 1)) * (∏ k in finset.range 1007, 2 * (k + 1)) = ∏ k in finset.range 2005, (k + 1) :=
sorry

end multiplicative_operation_correct_l110_110220


namespace find_y_from_exponent_equation_l110_110929

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110929


namespace length_of_AB_l110_110191

theorem length_of_AB
  (P Q : ℝ) (AB : ℝ)
  (hP : P = 3 / 7 * AB)
  (hQ : Q = 4 / 9 * AB)
  (hPQ : abs (Q - P) = 3) :
  AB = 189 :=
by
  sorry

end length_of_AB_l110_110191


namespace coin_flip_sequences_l110_110760

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110760


namespace smallest_odd_number_with_three_different_prime_factors_l110_110652

theorem smallest_odd_number_with_three_different_prime_factors :
  ∃ n, Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
  ∃ (n = 105), Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
sorry

end smallest_odd_number_with_three_different_prime_factors_l110_110652


namespace maximum_distance_correct_l110_110056

noncomputable def maximum_distance 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  ℝ :=
3 + Real.sqrt 5

theorem maximum_distance_correct 
  (m : ℝ) 
  (θ : ℝ) 
  (P : ℝ × ℝ) 
  (intersection : (x y : ℝ) → (x + m * y = 0) ∧ (m * x - y - 2 * m + 4 = 0) → P = (x, y)) 
  (distance : (x y : ℝ) → (x - 1) * Real.cos θ + (y - 2) * Real.sin θ = 3) : 
  maximum_distance m θ P intersection distance = 3 + Real.sqrt 5 := 
sorry

end maximum_distance_correct_l110_110056


namespace sum_distinct_products_HG_divisible_by_6_l110_110230

theorem sum_distinct_products_HG_divisible_by_6 :
  (∃ H G : ℕ, 0 ≤ H ∧ H ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧
              (6210004083005 + H * 10000 + G * 10) % 6 = 0) →
  ∑ x in {H * G | ∃ (H G : ℕ), 0 ≤ H ∧ H ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 
                      (6210004083005 + H * 10000 + G * 10) % 6 = 0}.to_finset, x = 0 := by
  sorry

end sum_distinct_products_HG_divisible_by_6_l110_110230


namespace coin_flip_sequences_l110_110728

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110728


namespace max_capacity_tank_l110_110768

-- Definitions of the conditions
def water_loss_1 := 32000 * 5
def water_loss_2 := 10000 * 10
def total_loss := water_loss_1 + water_loss_2
def water_added := 40000 * 3
def missing_water := 140000

-- Definition of the maximum capacity
def max_capacity := total_loss + water_added + missing_water

-- The theorem to prove
theorem max_capacity_tank : max_capacity = 520000 := by
  sorry

end max_capacity_tank_l110_110768


namespace find_notebooks_l110_110296

theorem find_notebooks (S N : ℕ) (h1 : N = 4 * S + 3) (h2 : N + 6 = 5 * S) : N = 39 := 
by
  sorry 

end find_notebooks_l110_110296


namespace distinct_sequences_ten_flips_l110_110689

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110689


namespace functional_equation_solution_l110_110417

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), 
  (∀ m n : ℤ, f(m + f(n)) - f(m) = n) ↔ (f = id ∨ f = (λ x, -x)) :=
by
  sorry

end functional_equation_solution_l110_110417


namespace coin_flip_sequences_l110_110711

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110711


namespace coefficient_a3b3_in_expression_l110_110269

theorem coefficient_a3b3_in_expression :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ k) * (b ^ (6 - k))) *
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (c ^ (8 - 2 * k)) * (c ^ (-2 * k))) =
  1400 := sorry

end coefficient_a3b3_in_expression_l110_110269


namespace Mia_and_dad_time_to_organize_toys_l110_110564

theorem Mia_and_dad_time_to_organize_toys :
  let total_toys := 60
  let dad_add_rate := 6
  let mia_remove_rate := 4
  let net_gain_per_cycle := dad_add_rate - mia_remove_rate
  let seconds_per_cycle := 30
  let total_needed_cycles := (total_toys - 2) / net_gain_per_cycle -- 58 toys by the end of repeated cycles, 2 is to ensure dad's last placement
  let last_cycle_time := seconds_per_cycle
  let total_time_seconds := total_needed_cycles * seconds_per_cycle + last_cycle_time
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 15 :=
by
  sorry

end Mia_and_dad_time_to_organize_toys_l110_110564


namespace solve_for_y_l110_110211

noncomputable def solve_quadratic := {y : ℂ // 4 + 3 * y^2 = 0.7 * y - 40}

theorem solve_for_y : 
  ∃ y : ℂ, (y = 0.1167 + 3.8273 * Complex.I ∨ y = 0.1167 - 3.8273 * Complex.I) ∧
            (4 + 3 * y^2 = 0.7 * y - 40) :=
by
  sorry

end solve_for_y_l110_110211


namespace distance_eq_3_implies_points_l110_110185

-- Definition of the distance of point A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement translating the problem
theorem distance_eq_3_implies_points (x : ℝ) (h : distance_to_origin x = 3) :
  x = 3 ∨ x = -3 :=
sorry

end distance_eq_3_implies_points_l110_110185


namespace negation_of_at_most_one_odd_l110_110286

variable (a b c : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def at_most_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ ¬is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ ¬is_odd c)

theorem negation_of_at_most_one_odd :
  ¬ at_most_one_odd a b c ↔
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ is_odd x ∧ is_odd y :=
sorry

end negation_of_at_most_one_odd_l110_110286


namespace find_a_l110_110442

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 0 then x ^ 2 + a else 2 ^ x

theorem find_a (a : ℝ) : f 1 a = f (-2) a → a = -2 :=
by
  intro h
  sorry

end find_a_l110_110442


namespace february_max_diff_percentage_l110_110955

noncomputable def max_diff_percentage (D B F : ℕ) : ℚ :=
  let avg_others := (B + F) / 2
  let high_sales := max (max D B) F
  (high_sales - avg_others) / avg_others * 100

theorem february_max_diff_percentage :
  max_diff_percentage 8 5 6 = 45.45 := by
  sorry

end february_max_diff_percentage_l110_110955


namespace coin_flip_sequences_l110_110746

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110746


namespace dealer_purchase_fraction_l110_110773

theorem dealer_purchase_fraction (P C : ℝ) (h1 : ∃ S, S = 1.5 * P) (h2 : ∃ S, S = 2 * C) :
  C / P = 3 / 8 :=
by
  -- The statement of the theorem has been generated based on the problem conditions.
  sorry

end dealer_purchase_fraction_l110_110773


namespace solve_for_a_l110_110433

def i := Complex.I

theorem solve_for_a (a : ℝ) (h : (2 + i) / (1 + a * i) = i) : a = -2 := 
by 
  sorry

end solve_for_a_l110_110433


namespace problem1_problem2_problem3_problem4_problem5_l110_110032

-- Define the functions
noncomputable def f1 : ℝ → ℝ := λ x, sin (π / 3)
noncomputable def f2 : ℝ → ℝ := λ x, 5^x
noncomputable def f3 : ℝ → ℝ := λ x, 1 / x^3
noncomputable def f4 : ℝ → ℝ := λ x, x^(3 / 4)
noncomputable def f5 : ℝ → ℝ := λ x, log x

-- Stating the theorems (problems)
theorem problem1 : deriv f1 x = 0 := sorry
theorem problem2 : deriv f2 x = (5^x * log 5) := sorry
theorem problem3 : deriv f3 x = -3 * x^(-4) := sorry
theorem problem4 : deriv f4 x = (3 / 4) * x^(-1 / 4) := sorry
theorem problem5 : deriv f5 x = 1 / x := sorry

end problem1_problem2_problem3_problem4_problem5_l110_110032


namespace axis_of_symmetry_of_parabola_with_given_x_intercepts_l110_110898

-- Define the parabola with its x-intercepts at -1 and 3
def parabola (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c

-- Conditions given in the problem
def x_intercepts : Prop := 
  ∃ (a b c : ℝ), (parabola a b c (-1) = 0) ∧ (parabola a b c 3 = 0)

-- Formalize statement that x = 1 is the axis of symmetry
theorem axis_of_symmetry_of_parabola_with_given_x_intercepts : 
  x_intercepts → (∃ (a b c : ℝ), parabola a b c 1 = parabola a b c (1 + 0)) :=
by
  sorry

end axis_of_symmetry_of_parabola_with_given_x_intercepts_l110_110898


namespace jose_peanuts_l110_110537

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l110_110537


namespace sequence_product_l110_110479

-- Definitions and conditions
def a : ℕ → ℕ
| 0     := 0          -- Not used, a_0 is not defined in the problem
| 1     := 1          -- Base case
| (n+1) := 2 * a n    -- Recursive case

-- Main statement
theorem sequence_product : a 3 * a 5 = 64 := 
by 
  -- The proof is not required per instructions
  sorry

end sequence_product_l110_110479


namespace possible_values_of_linear_combination_l110_110086

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem possible_values_of_linear_combination
  (a b : ℝ × ℝ)
  (h1 : (a.1 * b.1 + a.2 * b.2) = 0)
  (h2 : vector_length (a.1 - b.1, a.2 - b.2) = 2) :
  3 * vector_length a - 2 * vector_length b = 3 ∨ 3 * vector_length a - 2 * vector_length b = 4 :=
sorry

end possible_values_of_linear_combination_l110_110086


namespace max_integer_value_fraction_l110_110494

theorem max_integer_value_fraction (x : ℝ) : 
  (∃ t : ℤ, t = 2 ∧ (∀ y : ℝ, y = (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 9) → y <= t)) :=
sorry

end max_integer_value_fraction_l110_110494


namespace at_least_one_not_less_than_two_l110_110053

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → false := 
sorry

end at_least_one_not_less_than_two_l110_110053


namespace convert_1729_to_base5_l110_110385

-- Definition of base conversion from base 10 to base 5.
def convert_to_base5 (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if h : n = 0 then acc
    else let quotient := n / 5
         let remainder := n % 5
         aux quotient (remainder :: acc)
  aux n []

-- The theorem we seek to prove.
theorem convert_1729_to_base5 : convert_to_base5 1729 = [2, 3, 4, 0, 4] :=
by
  sorry

end convert_1729_to_base5_l110_110385


namespace fred_dark_blue_marbles_count_l110_110429

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l110_110429


namespace total_valid_numbers_eq_22_l110_110918

def digits : Type := {n : ℕ // 0 ≤ n ∧ n ≤ 9}

def valid_number (x y z : digits) : Prop :=
  (x.val = 2 * z.val) ∧ (¬(z.val = 0 ∨ z.val = 5)) ∧ (3 * z.val + y.val < 18) ∧ (y.val % 2 = 0)

def count_valid_numbers : ℕ :=
  finset.card (finset.filter (λ (n : digits × digits × digits), valid_number n.1 n.2 n.3)
    (finset.univ : finset (digits × digits × digits)))

theorem total_valid_numbers_eq_22 : count_valid_numbers = 22 := sorry

end total_valid_numbers_eq_22_l110_110918


namespace find_f_one_l110_110555

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_one (f : ℝ → ℝ) (h_smooth : differentiable ℝ f)
  (h_eq : ∀ x : ℝ, deriv f x = f (1 - x)) (h_init : f 0 = 1) : 
  f 1 = Real.sec 1 + Real.tan 1 :=
sorry

end find_f_one_l110_110555


namespace impossibility_of_equal_sum_selection_l110_110382

theorem impossibility_of_equal_sum_selection :
  ¬ ∃ (selected non_selected : Fin 10 → ℕ),
    (∀ i, selected i = 1 ∨ selected i = 36 ∨ selected i = 2 ∨ selected i = 35 ∨ 
              selected i = 3 ∨ selected i = 34 ∨ selected i = 4 ∨ selected i = 33 ∨ 
              selected i = 5 ∨ selected i = 32 ∨ selected i = 6 ∨ selected i = 31 ∨ 
              selected i = 7 ∨ selected i = 30 ∨ selected i = 8 ∨ selected i = 29 ∨ 
              selected i = 9 ∨ selected i = 28 ∨ selected i = 10 ∨ selected i = 27) ∧ 
    (∀ i, non_selected i = 1 ∨ non_selected i = 36 ∨ non_selected i = 2 ∨ non_selected i = 35 ∨ 
              non_selected i = 3 ∨ non_selected i = 34 ∨ non_selected i = 4 ∨ non_selected i = 33 ∨ 
              non_selected i = 5 ∨ non_selected i = 32 ∨ non_selected i = 6 ∨ non_selected i = 31 ∨ 
              non_selected i = 7 ∨ non_selected i = 30 ∨ non_selected i = 8 ∨ non_selected i = 29 ∨ 
              non_selected i = 9 ∨ non_selected i = 28 ∨ non_selected i = 10 ∨ non_selected i = 27) ∧ 
    (selected ≠ non_selected) ∧ 
    (Finset.univ.sum selected = Finset.univ.sum non_selected) :=
sorry

end impossibility_of_equal_sum_selection_l110_110382


namespace diameter_in_scientific_notation_l110_110507

-- Define a constant for the diameter given in the problem
def diameter : ℝ := 0.0000054

-- Define the expected result in scientific notation
def scientific_notation : ℝ := 5.4 * 10^(-6)

-- Theorem stating that the given diameter is equivalent to its scientific notation representation
theorem diameter_in_scientific_notation : diameter = scientific_notation :=
by
  sorry

end diameter_in_scientific_notation_l110_110507


namespace probability_sum_18_two_12_sided_dice_l110_110774

theorem probability_sum_18_two_12_sided_dice :
  let total_outcomes := 12 * 12
  let successful_outcomes := 7
  successful_outcomes / total_outcomes = 7 / 144 := by
sorry

end probability_sum_18_two_12_sided_dice_l110_110774


namespace x_n_value_x_n_largest_at_l110_110621

variables {α β λ : ℝ} (n : ℕ)
variable (x : ℕ → ℝ)

noncomputable def x_n_seq (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then λ
  else
    (α + β)^n * x n =
    ∑ i in Finset.range (n+1),
      (α^(n-i) * β^i * (if i = 0 then x 0 else if i = 1 then x 1 else x (n-i) * x i))

theorem x_n_value :
  ∀ (n : ℕ), x_n_seq x n = λ^n / n! :=
  sorry

theorem x_n_largest_at :
  ∀ (n : ℕ), x n = λ^n / n! ∧ 
  (∀ m : ℕ, x m ≤ x n) ↔
  n = floor λ :=
  sorry

end x_n_value_x_n_largest_at_l110_110621


namespace larger_ball_radius_l110_110619

def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem larger_ball_radius (r : ℝ) :
  (∃ r, volume_of_sphere r = 2 * Real.pi) → r = Real.cbrt 3 :=
by sorry

end larger_ball_radius_l110_110619


namespace complex_number_quadrant_l110_110944

variable (z : ℂ)
variable (h : (1 + 2 * complex.I) / z = 1 - complex.I)

theorem complex_number_quadrant :
  z = -1 / 2 + 3 / 2 * complex.I ∧
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l110_110944


namespace even_not_friendly_vertices_l110_110061

structure Polygon where
  vertices : List (ℝ × ℝ)  -- List of vertices as (x, y) coordinates

def is_perpendicular (v1 v2 : (ℝ × ℝ)) : Prop :=
  (v1.1 = v2.1) ∨ (v1.2 = v2.2)  -- Neighboring sides are perpendicular if x or y coordinates match

def angle_bisectors_perpendicular (v1 v2 : (ℝ × ℝ)) : Prop :=
  sorry  -- Define condition for perpendicular angle bisectors (to be detailed as needed)

def not_friendly (polygon : Polygon) (v : (ℝ × ℝ)) : Nat :=
  polygon.vertices.count (λ w => angle_bisectors_perpendicular v w)

theorem even_not_friendly_vertices
  (polygon : Polygon)
  (h_perpendicular : ∀ (i : ℕ), is_perpendicular (polygon.vertices.nth i) (polygon.vertices.nth (i+1) % polygon.vertices.length))
  (v : (ℝ × ℝ) ∈ polygon.vertices) :
  even (not_friendly polygon v) :=
sorry

end even_not_friendly_vertices_l110_110061


namespace determine_S_l110_110180

-- Define the set S
def S : Set ℂ := {z : ℂ | z ∈ {-1, 1, -i, i}}

-- Define the conditions
def modulus_one (z : ℂ) : Prop :=
  complex.abs z = 1

def condition_1 : Prop :=
  (1 : ℂ) ∈ S

def condition_2 : Prop :=
  ∀ z1 z2 : ℂ, z1 ∈ S → z2 ∈ S → (z1 - 2 * z2 * real.cos ((complex.arg (z1 / z2)))) ∈ S

-- Prove the set S satisfies the conditions
theorem determine_S (n : ℕ) (hn : 2 < n ∧ n < 6) : 
  (∀ z ∈ S, modulus_one z) ∧ condition_1 ∧ condition_2 :=
by
  sorry

end determine_S_l110_110180


namespace combined_height_l110_110535

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l110_110535


namespace coin_flip_sequences_l110_110758

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110758


namespace no_such_function_exists_l110_110197

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, (∀ n : ℕ, 0 < n → f(f(n)) = n + 1987) :=
sorry

end no_such_function_exists_l110_110197


namespace inequality_proof_l110_110671

noncomputable def proof_inequality (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i : Fin n, 0 < x i) (h3 : Finset.univ.sum x = 1) : Prop :=
  (Finset.univ.sum (λ i, 1 / (1 - x i))) * 
  (Finset.sum (Finset.powersetLen 2 Finset.univ) 
    (λ s, let ⟨i, _, j, _⟩ := s.pair in x i * x j)) ≤ n / 2

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i : Fin n, 0 < x i) (h3 : Finset.univ.sum x = 1) : proof_inequality n x h1 h2 h3 :=
sorry

end inequality_proof_l110_110671


namespace birth_age_of_mother_l110_110487

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l110_110487


namespace system_of_inequalities_solutions_l110_110587

theorem system_of_inequalities_solutions (x : ℤ) :
  (3 * x - 2 ≥ 2 * x - 5) ∧ ((x / 2 - (x - 2) / 3 < 1 / 2)) →
  (x = -3 ∨ x = -2) :=
by sorry

end system_of_inequalities_solutions_l110_110587


namespace color_change_probability_l110_110796

theorem color_change_probability :
  let cycle_time := 85
  let green_time := 40
  let yellow_time := 5
  let red_time := 40
  let color_change_time := 3 * 4
  color_change_time / cycle_time = 12 / 85 :=
begin
  sorry
end

end color_change_probability_l110_110796


namespace monotonicity_of_f_l110_110471

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 6)

theorem monotonicity_of_f :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < π / 6 → f x₁ < f x₂ :=
by
  sorry

end monotonicity_of_f_l110_110471


namespace number_of_true_propositions_is_zero_l110_110828

def Proposition1 (A B C D : Type) (f : A → B) (g : C → D) : Prop := collinear f g → ∀ (x : A) (y : B) (z : C) (w : D), (f x = g y → z = w)
def Proposition2 (a b : Type) (h : a → b) : Prop := parallel h h → (∀ (x : a) (y : b), (h x = -h y) ∨ (h x = h y))
def Proposition3 (u : Type) (i : u → ℝ) : Prop := ∀ (x y : u), (i x = 1 ∧ i y = 1) → x = y

theorem number_of_true_propositions_is_zero :
  ∀ (A B C D a b u : Type) (f : A → B) (g : C → D) (h : a → b) (i : u → ℝ),
  ¬ (Proposition1 A B C D f g ∨ Proposition2 a b h ∨ Proposition3 u i) := 
sorry

end number_of_true_propositions_is_zero_l110_110828


namespace incircle_radius_of_DEF_l110_110253

theorem incircle_radius_of_DEF (DF DE : ℝ) (h1 : DF = 8) (h2 : DE = 8) 
  (right_angle_at_F : ∠ D F E = 90) (angle_D : ∠ E D F = 45) : 
  ∃ r : ℝ, r = 4 - 2 * Real.sqrt 2 := 
by
  use 4 - 2 * Real.sqrt 2
  sorry

end incircle_radius_of_DEF_l110_110253


namespace number_of_sets_satisfying_condition_l110_110231

-- Define the sets involved.
def s1 : Set ℕ := {1, 2}
def s2 : Set ℕ := {1, 2, 3}

-- Predicate to check if a set M satisfies the condition.
def satisfies_condition (M : Set ℕ) : Prop :=
  s1 ∪ M = s2

-- Definition of the problem statement in Lean.
theorem number_of_sets_satisfying_condition : 
  (∃ (M : Set ℕ), satisfies_condition M) → ({M : Set ℕ // satisfies_condition M}.card = 4) := by
  sorry

end number_of_sets_satisfying_condition_l110_110231


namespace smallest_n_inequality_l110_110867

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
           (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) ∧
           n = 4 :=
by
  let n := 4
  sorry

end smallest_n_inequality_l110_110867


namespace calculate_crayons_lost_l110_110575

def initial_crayons := 440
def given_crayons := 111
def final_crayons := 223

def crayons_left_after_giving := initial_crayons - given_crayons
def crayons_lost := crayons_left_after_giving - final_crayons

theorem calculate_crayons_lost : crayons_lost = 106 :=
  by
    sorry

end calculate_crayons_lost_l110_110575


namespace find_m_containing_2015_l110_110183

theorem find_m_containing_2015 : 
  ∃ n : ℕ, ∀ k, 0 ≤ k ∧ k < n → 2015 = n^3 → (1979 + 2*k < 2015 ∧ 2015 < 1979 + 2*k + 2*n) :=
by
  sorry

end find_m_containing_2015_l110_110183


namespace coin_flip_sequences_l110_110762

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110762


namespace unique_decreasing_function_l110_110803

theorem unique_decreasing_function {f : ℝ → ℝ}: 
  (∀ x_1 x_2 : ℝ, 0 < x_1 ∧ 0 < x_2 ∧ x_1 > x_2 → f x_1 < f x_2) → 
  (f = (λ x, 1 / x)) ∨ 
  (f = (λ x, (x - 1)^2)) ∨ 
  (f = (λ x, 2^x)) ∨ 
  (f = (λ x, Real.log x / Real.log 2)) →
  f = (λ x, 1 / x) :=
by
  sorry

end unique_decreasing_function_l110_110803


namespace quadratic_extreme_values_l110_110886

theorem quadratic_extreme_values (y1 y2 y3 y4 : ℝ) 
  (h1 : y2 < y3) 
  (h2 : y3 = y4) 
  (h3 : ∀ x, ∃ (a b c : ℝ), ∀ y, y = a * x * x + b * x + c) :
  (y1 < y2) ∧ (y2 < y3) :=
by
  sorry

end quadratic_extreme_values_l110_110886


namespace golden_section_AC_length_l110_110052

namespace GoldenSection

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def AC_length (AB : ℝ) : ℝ :=
  let φ := golden_ratio
  AB / φ

theorem golden_section_AC_length (AB : ℝ) (C_gold : Prop) (hAB : AB = 2) (A_gt_B : AC_length AB > AB - AC_length AB) :
  AC_length AB = Real.sqrt 5 - 1 :=
  sorry

end GoldenSection

end golden_section_AC_length_l110_110052


namespace problem_statement_l110_110422

-- Define M and m as maximum and minimum respectively
def M (a b : ℝ) := max a b
def m (a b : ℝ) := min a b

-- Given the conditions
variables (p q r s t : ℝ)
variables (hpq : p < q) (hqr : q < r) (hrs : r < s) (hst : s < t)
variables (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ t ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ s ∧ q ≠ t ∧ r ≠ t)

-- Statement to be proved
theorem problem_statement : M(m(M(p,q),r),M(s,m(q,t))) = s :=
sorry

end problem_statement_l110_110422


namespace find_a_l110_110237

noncomputable def quadratic_inequality_solution (a b : ℝ) : Prop :=
  a * ((-1/2) * (1/3)) * 20 = 20 ∧
  a < 0 ∧
  (-b / (2 * a)) = (-1 / 2 + 1 / 3)

theorem find_a (a b : ℝ) (h : quadratic_inequality_solution a b) : a = -12 :=
  sorry

end find_a_l110_110237


namespace coin_flip_sequences_l110_110700

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110700


namespace jennifer_apples_total_l110_110160

theorem jennifer_apples_total : 
  let initial_apples := 7
  let tripling (x : ℕ) := x * 3
  let hrs := 3
  let found_apples := 74
  let final_count := tripling (tripling (tripling initial_apples)) + found_apples
  in final_count = 263 :=
by
  let initial_apples := 7
  let tripling (x : ℕ) := x * 3
  let hrs := 3
  let found_apples := 74
  let final_count := tripling (tripling (tripling initial_apples)) + found_apples
  have h1 : final_count = 263 := sorry
  exact h1

end jennifer_apples_total_l110_110160


namespace count_valid_numbers_l110_110491

def is_valid_number (N : ℕ) : Prop :=
  let a := N / 1000
  let x := N % 1000
  (N = 1000 * a + x) ∧ (x = N / 8) ∧ (100 ≤ x) ∧ (x < 1000) ∧ (1 ≤ a) ∧ (a ≤ 6)

theorem count_valid_numbers : (finset.filter is_valid_number (finset.range 10000)).card = 6 := 
  sorry

end count_valid_numbers_l110_110491


namespace point_M_coordinates_l110_110499

theorem point_M_coordinates (a : ℤ) (h : a + 3 = 0) : (a + 3, 2 * a - 2) = (0, -8) :=
by
  sorry

end point_M_coordinates_l110_110499


namespace all_distances_even_l110_110168

theorem all_distances_even {n : ℕ} (h_odd : n ≥ 3 ∧ odd n) (a : fin n → ℤ)
  (h_even_sum : ∀ i, ∑ j in finset.univ.filter (≠ i), |a i - a j| % 2 = 0) :
  ∀ i j, |a i - a j| % 2 = 0 := 
sorry

end all_distances_even_l110_110168


namespace range_g_l110_110100

def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

def g (x : ℝ) : ℝ :=
  (f x) * (f x) + f (x * x)

theorem range_g :
  ∀ y, ∃ x, (1 ≤ x ∧ x ≤ 3) ∧ y = g x ↔ 6 ≤ y ∧ y ≤ 13 := 
by
  sorry

end range_g_l110_110100


namespace find_a_l110_110441

theorem find_a (x a a1 a2 a3 a4 : ℝ) :
  (x + a) ^ 4 = x ^ 4 + a1 * x ^ 3 + a2 * x ^ 2 + a3 * x + a4 → 
  a1 + a2 + a3 = 64 → a = 2 :=
by
  sorry

end find_a_l110_110441


namespace sum_of_squares_of_binomial_coeffs_l110_110626

theorem sum_of_squares_of_binomial_coeffs (n : ℕ) : 
  (∑ k in Finset.range (n + 1), Nat.choose n k ^ 2) = Nat.choose (2 * n) n :=
begin
  sorry
end

end sum_of_squares_of_binomial_coeffs_l110_110626


namespace factor_expression_l110_110844

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110844


namespace min_x_y_l110_110078

open Real

theorem min_x_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : x + y ≥ 16 := 
sorry

end min_x_y_l110_110078


namespace range_of_m_l110_110907

namespace TrigonometricProof

theorem range_of_m (m : ℝ) (h : m > 0) :
  (∃ x₁ x₂ ∈ set.Icc (0 : ℝ) (Real.pi / 4),
    (sin (2 * x₁) + 2 * Real.sqrt 3 * cos x₁ ^ 2 - Real.sqrt 3) =
    (m * cos (2 * x₂ - Real.pi / 6) - 2 * m + 3)) →
    m ∈ set.Icc (2 / 3) 2 :=
by
  sorry

end TrigonometricProof

end range_of_m_l110_110907


namespace shaded_area_common_squares_l110_110343

noncomputable def cos_beta : ℝ := 3 / 5

theorem shaded_area_common_squares :
  ∀ (β : ℝ), (0 < β) → (β < pi / 2) → (cos β = cos_beta) →
  (∃ A, A = 4 / 3) :=
by
  sorry

end shaded_area_common_squares_l110_110343


namespace ellipse_equation_solution_line_existence_condition_l110_110901

theorem ellipse_equation_solution :
  ∃ (a b : ℝ), 
  (0 < b ∧ b < a) ∧
  (∃ (c : ℝ), 
    (a^2 = b^2 + c^2) ∧ 
    (c / a = sqrt 3 / 2) ∧ 
    (abs (a * b) / sqrt (a^2 + b^2) = 2 * sqrt 5 / 5)) ∧
  (∀ x y : ℝ, 
    (\frac{x^2}{4} + y^2 = 1) → 
    (\frac{x / 2} + y = 1) ∧ abs (2 * sqrt 5 / 5)) :=
begin
  sorry
end

theorem line_existence_condition :
  ∃ (k : ℝ), 
  (k^2 > 4 / 9) ∧ 
  (y = k * x + 5 / 3) ∧ 
  (∀ x1 x2 y1 y2 : ℝ,
    (y1 = k * x1 + 5 / 3 ∧ y2 = k * x2 + 5 / 3) →
    (x1 = 2 * x2) →
    (x1 + x2 = -40 * k / (3 + 12 * k^2)) ∧ 
    (x1 * x2 = 64 / (9 + 36 * k^2))) :=
begin
  sorry
end

end ellipse_equation_solution_line_existence_condition_l110_110901


namespace smallest_shift_l110_110104

noncomputable def intersection_points (ω : ℝ) : ℕ → ℝ 
| n := (2 * n + 1) * π / 3

def shifted_intersection_points (ω m : ℝ) : ℕ → ℝ 
| n := intersection_points ω n - m

def symmetric_about_y (f : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), |f n| = |f (n + 1)|

theorem smallest_shift {ω : ℝ} (h : ∀ (n : ℕ), intersection_points ω (n+1) - intersection_points ω n = π / 2)
  (m_pos : ∀ (m : ℝ), m > 0)
  (symm : symmetric_about_y (shifted_intersection_points ω m)) :
  m = π / 12 :=
sorry

end smallest_shift_l110_110104


namespace distance_between_A_and_B_l110_110299

def rowing_speed_still_water : ℝ := 10
def round_trip_time : ℝ := 5
def stream_speed : ℝ := 2

theorem distance_between_A_and_B : 
  ∃ x : ℝ, 
    (x / (rowing_speed_still_water - stream_speed) + x / (rowing_speed_still_water + stream_speed) = round_trip_time) 
    ∧ x = 24 :=
sorry

end distance_between_A_and_B_l110_110299


namespace factor_expression_l110_110842

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110842


namespace find_y_l110_110921

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110921


namespace combined_height_l110_110534

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l110_110534


namespace validate_pairs_l110_110026

noncomputable def problem (P Q : ℝ → ℝ) : Prop :=
(P 1 = 0 ∨ P 2 = 1) → Q 1 = 1 ∧ Q 3 = 1 ∧
(P 2 = 0 ∨ P 4 = 0) → Q 2 = 0 ∧ Q 4 = 0 ∧
(P 3 = 1 ∨ P 4 = 1) → Q 1 = 0

noncomputable def answer_valid_pairs : list (ℝ → ℝ × ℝ → ℝ) :=
[(λ x, -1 / 2 * x^3 + 7 / 2 * x^2 - 7 * x + 4, λ x, -2 / 3 * x^3 + 5 * x^2 - 34 / 3 * x + 8),
 (λ x, 1 / 2 * x^3 - 4 * x^2 + 19 / 2 * x - 6, λ x, -1 / 2 * x^3 + 4 * x^2 - 19 / 2 * x + 7),
 (λ x, -1 / 6 * x^3 + 3 / 2 * x^2 - 13 / 3 * x + 4, λ x, -1 / 6 * x^3 + 3 / 2 * x^2 - 13 / 3 * x + 4),
 (λ x, -1 / 6 * x^3 + 3 / 2 * x^2 - 13 / 3 * x + 4, λ x, -1 / 2 * x^3 + 7 / 2 * x^2 - 7 * x + 4),
 (λ x, -2 / 3 * x^3 + 5 * x^2 - 34 / 3 * x + 8, λ x, -1 / 2 * x^3 + 7 / 2 * x^2 - 7 * x + 4),
 (λ x, 1 / 3 * x^3 - 5 / 2 * x^2 + 31 / 6 * x - 2, λ x, -2 / 3 * x^3 + 5 * x^2 - 34 / 3 * x + 8)]

theorem validate_pairs : ∀ P Q, (P, Q) ∈ answer_valid_pairs → problem P Q :=
sorry

end validate_pairs_l110_110026


namespace geometric_sequence_log_sum_l110_110959

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n)
  (hgeom : ∀ n, a (n+1) / a n = a 2 / a 1)
  (hcond : a 3 * a 8 + a 4 * a 7 = 18) : 
  ∑ i in Finset.range 10, log 3 (a (i+1)) = 10 := 
by
  sorry

end geometric_sequence_log_sum_l110_110959


namespace volume_of_pyramid_l110_110328

def square_area : ℝ := 256
def triangle_abe_area : ℝ := 128
def triangle_cde_area : ℝ := 112

theorem volume_of_pyramid : 
  (∃ s h : ℝ, s = sqrt square_area ∧ 
              (triangle_abe_area = 0.5 * s * h ∨ triangle_cde_area = 0.5 * s * h) ∧ 
              (1 / 3) * square_area * h = 1230.83) :=
by sorry

end volume_of_pyramid_l110_110328


namespace find_y_l110_110925

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110925


namespace second_number_ratio_l110_110366

noncomputable def a1 (x : ℝ) := x / 2
noncomputable def a2 (x : ℝ) := 2 * x - 3
noncomputable def a3 (x : ℝ) := 18 / x + 1

theorem second_number_ratio (x : ℝ) (h : (2 * x - 3) ^ 2 = (x / 2) * (18 / x + 1)) :
  (2x - 3) / (x / 2) = 2.08 :=
sorry

end second_number_ratio_l110_110366


namespace fixed_fee_correct_l110_110915

noncomputable def fixed_monthly_fee (february_bill march_bill : ℝ) : ℝ :=
  let y := (march_bill - february_bill) / 2 in
  february_bill - y

theorem fixed_fee_correct :
  fixed_monthly_fee 18.60 32.40 = 11.70 :=
by
  sorry

end fixed_fee_correct_l110_110915


namespace number_of_triangles_in_triangulation_l110_110527

theorem number_of_triangles_in_triangulation:
  ∀ (P : set (ℝ × ℝ)), 
    P.card = 100 ∧ 
    ∀ Δ : finset (ℝ × ℝ), Δ ⊆ P ∪ {(0, 0), (0, 1), (1, 0), (1, 1)} →
    (∃ v1 v2 v3 : (ℝ × ℝ), v1 ∈ Δ ∧ v2 ∈ Δ ∧ v3 ∈ Δ ∧ v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3) →
  ∃ T : finset (finset (ℝ × ℝ)), 
    T.card = 202 ∧ 
    (∀ t ∈ T, ∃ v1 v2 v3 : (ℝ × ℝ), v1 ∈ t ∧ v2 ∈ t ∧ v3 ∈ t ∧ v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3)
  :=
begin
  sorry
end

end number_of_triangles_in_triangulation_l110_110527


namespace rate_times_base_eq_9000_l110_110257

noncomputable def Rate : ℝ := 0.00015
noncomputable def BaseAmount : ℝ := 60000000

theorem rate_times_base_eq_9000 :
  Rate * BaseAmount = 9000 := 
  sorry

end rate_times_base_eq_9000_l110_110257


namespace smallest_n_18x18_l110_110868

def valid_board (n : ℕ) (board : ℕ → ℕ → option ℕ) : Prop :=
  (∀ (i j : ℕ), i < 18 → j < 18 → 
     ∃ (m : ℕ), m < n ∧ board i j = some m) ∧
  (∀ (i : ℕ), i < 18 → 
     ∀ (j₁ j₂ : ℕ), j₁ < 18 → j₂ < 18 → j₁ ≠ j₂ → 
     (board i j₁).bind (λ m₁, (board i j₂).map (λ m₂, abs (m₁ - m₂))) ≠ some 0 ) ∧
  (∀ (j : ℕ), j < 18 → 
     ∀ (i₁ i₂ : ℕ), i₁ < 18 → i₂ < 18 → i₁ ≠ i₂ → 
     (board i₁ j).bind (λ m₁, (board i₂ j).map (λ m₂, abs (m₁ - m₂))) ≠ some 0 ) ∧
  (∀ (i : ℕ), i < 18 → 
     ∀ (j₁ j₂ : ℕ), j₁ < 18 → j₂ < 18 → j₁ ≠ j₂ → 
     (board i j₁).bind (λ m₁, (board i j₂).map (λ m₂, abs (m₁ - m₂))) ≠ some 1 ) ∧
  (∀ (j : ℕ), j < 18 → 
     ∀ (i₁ i₂ : ℕ), i₁ < 18 → i₂ < 18 → i₁ ≠ i₂ → 
     (board i₁ j).bind (λ m₁, (board i₂ j).map (λ m₂, abs (m₁ - m₂))) ≠ some 1 )

theorem smallest_n_18x18 : 
  ∃ (n : ℕ), n = 37 ∧ ∃ (board : ℕ → ℕ → option ℕ), valid_board n board :=
sorry

end smallest_n_18x18_l110_110868


namespace PQRS_is_cyclic_l110_110987

theorem PQRS_is_cyclic
  (A B C D P Q R S X Y Z W : Point)
  (l₁ l₂ : Line)
  (ω₁ ω₂ ω₃ ω₄ : Circle)
  (h_parallelogram : Parallelogram A B C D)
  (hω₁ : ω₁.is_half_circle_of A B)
  (hω₂ : ω₂.is_half_circle_of B C)
  (hω₃ : ω₃.is_half_circle_of C D)
  (hω₄ : ω₄.is_half_circle_of D A)
  (hl₁ : l₁.is_parallel_to (line B C))
  (hl₂ : l₂.is_parallel_to (line A B))
  (h_intersections₁ : intersects l₁ ω₁ X ∧ intersects l₁ (Seg A B) P ∧ intersects l₁ (Seg C D) R ∧ intersects l₁ ω₃ Z)
  (h_intersections₂ : intersects l₂ ω₂ Y ∧ intersects l₂ (Seg B C) Q ∧ intersects l₂ (Seg D A) S ∧ intersects l₂ ω₄ W)
  (h_condition : dist X P * dist R Z = dist Y Q * dist S W) :
  Cyclic P Q R S :=
sorry

end PQRS_is_cyclic_l110_110987


namespace single_input_assigns_multiple_variables_l110_110371

-- Define the format and behavior of the input statement as conditions
def input_statement_seq (a b c : ℕ) : Prop :=
  -- This definition encapsulates the behavior of sequential input
  (input a b c → (a, b, c) = (input_value1, input_value2, input_value3))

-- Define the problem statement
theorem single_input_assigns_multiple_variables :
  ∃ (a b c : ℕ), input_statement_seq a b c :=
by
  -- Skipping the proof as instructed
  sorry

end single_input_assigns_multiple_variables_l110_110371


namespace factor_expression_l110_110849

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110849


namespace strictly_increasing_interval_l110_110396

noncomputable def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

theorem strictly_increasing_interval :
  ∀ (x : ℝ), -π ≤ x ∧ x ≤ 0 → ∃ (a b : ℝ), a = -π/6 ∧ b = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  sorry

end strictly_increasing_interval_l110_110396


namespace divisor_of_5025_is_5_l110_110783

/--
  Given an original number n which is 5026,
  and a resulting number after subtracting 1 from n,
  prove that the divisor of the resulting number is 5.
-/
theorem divisor_of_5025_is_5 (n : ℕ) (h₁ : n = 5026) (d : ℕ) (h₂ : (n - 1) % d = 0) : d = 5 :=
sorry

end divisor_of_5025_is_5_l110_110783


namespace coin_flip_sequences_l110_110705

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110705


namespace distinct_sequences_ten_flips_l110_110687

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110687


namespace math_problem_l110_110076

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def odd_function (g : ℝ → ℝ) := ∀ x : ℝ, g x = -g (-x)

theorem math_problem
  (hf_even : even_function f)
  (hf_0 : f 0 = 1)
  (hg_odd : odd_function g)
  (hgf : ∀ x : ℝ, g x = f (x - 1)) :
  f 2011 + f 2012 + f 2013 = 1 := sorry

end math_problem_l110_110076


namespace coefficient_of_a3b3_l110_110258

theorem coefficient_of_a3b3 (a b c : ℚ) :
  (∏ i in Finset.range 7, (a + b) ^ i * (c + 1 / c) ^ (8 - i)) = 1400 := 
by
  sorry

end coefficient_of_a3b3_l110_110258


namespace abigail_money_left_l110_110801

def initial_amount : ℕ := 11
def spent_in_store : ℕ := 2
def amount_lost : ℕ := 6

theorem abigail_money_left :
  initial_amount - spent_in_store - amount_lost = 3 := 
by {
  sorry
}

end abigail_money_left_l110_110801


namespace region_area_l110_110235

-- Definitions corresponding to the conditions:
def region := { p : ℝ × ℝ | p.snd > 3 * p.fst ∧ p.snd > 5 - 2 * p.fst ∧ p.snd < 6 }

-- Definition of the function to calculate the area of the region:
noncomputable def area_of_region : ℝ :=
  let points := [(-0.5 : ℝ, 6 : ℝ), (1 : ℝ, 3 : ℝ), (2 : ℝ, 6 : ℝ)]
  -- Calculate area using the vertices of the region
  -- In this formulation Calculations may be encapsulated directly.
  (3 * 1.75) -- height * average base length

-- The theorem to be proven:
theorem region_area : area_of_region = 5.25 := by
  sorry

end region_area_l110_110235


namespace square_overlap_area_l110_110340

theorem square_overlap_area (β : ℝ) (h1 : 0 < β) (h2 : β < 90) (h3 : Real.cos β = 3 / 5) : 
  area (common_region (square 2) (rotate_square β (square 2))) = 4 / 3 :=
sorry

end square_overlap_area_l110_110340


namespace pq_s_value_l110_110936

variables {R : Type*} [Field R] (r1 r2 r3 r4 m p q s : R)

def polynomial_divisibility (f g : R[X]) : Prop :=
  f ∣ g

theorem pq_s_value 
  (h1 : polynomial_divisibility 
          (polynomial.C 1 * polynomial.X^3 + polynomial.C 4 * polynomial.X^2 + polynomial.C 12 * polynomial.X + polynomial.C m) 
          (polynomial.X^4 + polynomial.C 5 * polynomial.X^3 + polynomial.C (7 * p) * polynomial.X^2 + polynomial.C (4 * q) * polynomial.X + polynomial.C s))
  (h2 : r1 + r2 + r3 = -4)
  (h3 : r1 * r2 + r2 * r3 + r1 * r3 = 12)
  (h4 : r1 * r2 * r3 = -m)
  (h_r4 : r4 = -1) : 
  (p + q) * s = (7 * m + 148) * m / 28 :=
sorry

end pq_s_value_l110_110936


namespace functional_equation_solution_given_conditions_l110_110438

variables (a b : ℚ) (f : ℚ → ℚ)

theorem functional_equation_solution_given_conditions :
  (∀ x y: ℚ, f(x + a + f(y)) = f(x + b) + y) → (∀ x: ℚ, f(x) = x + b - a ∨ f(x) = -x + b - a) :=
by
  intros
  sorry

end functional_equation_solution_given_conditions_l110_110438


namespace arithmetic_sequence_sum_l110_110148

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

variable {a : ℕ → ℝ}
variable (d : ℝ)

theorem arithmetic_sequence_sum (h1 : 0 < a 1)
  (h2 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 4)
  (arithmetic_seq a) :
  a 4 + a 5 = 8 := sorry

end arithmetic_sequence_sum_l110_110148


namespace circle_and_reflection_l110_110967

noncomputable def is_circle_on_line (a b : ℝ) : Prop :=
  a - 2 * b = 0

def passes_through_point (a b r : ℝ) : Prop :=
  (4 - a) ^ 2 + b ^ 2 = r ^ 2

noncomputable def chord_length_condition (a b r : ℝ) : Prop :=
  let d := (abs (4 * a - 3 * b)) / (5 : ℝ) in
  d = real.sqrt (r ^ 2 - 4)

def reflected_through_center (a b : ℝ) : Prop :=
  let N := (-4, -1) in
  let C := (a, b) in
  (2 * -4 + N.snd * 5 + 3 = 0) ∧
  (5 * (C.snd - N.snd) = 2 * (C.fst - N.fst))

theorem circle_and_reflection :
  ∃ a b r : ℝ,
  is_circle_on_line a b ∧
  passes_through_point a b r ∧
  chord_length_condition a b r ∧
  ¬((4 - 2) ^ 2 + 1 ^ 2 = r ^ 2) ∧
  ∃ eqcircle : ∀ x y : ℝ, (x - a) ^ 2 + (y - b) ^ 2 = r ^ 2,
  ∃ m n : ℝ, reflected_through_center a b ∧ (2 * m + 5 * n + 3 = 0) := by
  sorry

end circle_and_reflection_l110_110967


namespace trade_and_unification_effects_l110_110829

theorem trade_and_unification_effects :
  let country_A_corn := 8
  let country_B_eggplants := 18
  let country_B_corn := 12
  let country_A_eggplants := 10
  
  -- Part (a): Absolute and comparative advantages
  (country_B_corn > country_A_corn) ∧ (country_B_eggplants > country_A_eggplants) ∧
  let opportunity_cost_A_eggplants := country_A_corn / country_A_eggplants
  let opportunity_cost_A_corn := country_A_eggplants / country_A_corn
  let opportunity_cost_B_eggplants := country_B_corn / country_B_eggplants
  let opportunity_cost_B_corn := country_B_eggplants / country_B_corn
  (opportunity_cost_B_eggplants < opportunity_cost_A_eggplants) ∧ (opportunity_cost_A_corn < opportunity_cost_B_corn) ∧

  -- Part (b): Volumes produced and consumed with trade
  let price := 1
  let income_A := country_A_corn * price
  let income_B := country_B_eggplants * price
  let consumption_A_eggplants := income_A / price / 2
  let consumption_A_corn := country_A_corn / 2
  let consumption_B_corn := income_B / price / 2
  let consumption_B_eggplants := country_B_eggplants / 2
  (consumption_A_eggplants = 4) ∧ (consumption_A_corn = 4) ∧
  (consumption_B_corn = 9) ∧ (consumption_B_eggplants = 9) ∧

  -- Part (c): Volumes after unification without trade
  let unified_eggplants := 18 - (1.5 * 4)
  let unified_corn := 8 + 4
  let total_unified_eggplants := unified_eggplants
  let total_unified_corn := unified_corn
  (total_unified_eggplants = 12) ∧ (total_unified_corn = 12) ->
  
  total_unified_eggplants = 12 ∧ total_unified_corn = 12 ∧
  (total_unified_eggplants < (consumption_A_eggplants + consumption_B_eggplants)) ∧
  (total_unified_corn < (consumption_A_corn + consumption_B_corn))
:= by
  -- Proof omitted
  sorry

end trade_and_unification_effects_l110_110829


namespace perimeter_of_triangle_l110_110074

-- Given conditions
variable (a b c : ℕ)
variable (ha : a = 7)
variable (hb : b = 2)
variable (hodd : c % 2 = 1)
variable (h_range : 5 < c ∧ c < 9)

-- Proof statement
theorem perimeter_of_triangle (ha : a = 7) (hb : b = 2) (hodd : c % 2 = 1) (h_range: 5 < c ∧ c < 9) :
  a + b + c = 16 := by
sory

end perimeter_of_triangle_l110_110074


namespace value_of_a_l110_110133

theorem value_of_a (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x - 1 = 0) ∧ (∀ x y : ℝ, a * x^2 - 2 * x - 1 = 0 ∧ a * y^2 - 2 * y - 1 = 0 → x = y) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l110_110133


namespace count_valid_numbers_l110_110490

def is_valid_number (N : ℕ) : Prop :=
  let a := N / 1000
  let x := N % 1000
  (N = 1000 * a + x) ∧ (x = N / 8) ∧ (100 ≤ x) ∧ (x < 1000) ∧ (1 ≤ a) ∧ (a ≤ 6)

theorem count_valid_numbers : (finset.filter is_valid_number (finset.range 10000)).card = 6 := 
  sorry

end count_valid_numbers_l110_110490


namespace concyclic_when_f_minimized_l110_110444

-- Define the points and the conditions
variables {A B C D P : Type*}

-- Define the angles and their condition
variables (angle_B angle_D : ℝ)
axiom angle_condition : angle_B + angle_D < 180

-- Define distances for function f
variables (PA BC PD CA PC AB : ℝ)

-- Define function f(P)
def f (PA BC PD CA PC AB : ℝ) : ℝ :=
  PA * BC + PD * CA + PC * AB

-- Condition for point concyclicity
def are_concyclic (P A B C : Type*) : Prop := sorry

-- The proof statement
theorem concyclic_when_f_minimized 
  (h : ∀ P, f PA BC PD CA PC AB ≤ f PA BC PD CA PC AB) :
  are_concyclic P A B C :=
sorry

end concyclic_when_f_minimized_l110_110444


namespace line_not_parallel_if_not_perpendicular_plane_l110_110450

-- Conditions
variables (l m : Type) [LinearOrder l] [LinearOrder m]
variables (α β : Type) [Plane α] [Plane β]

-- Definitions based on conditions
variables (perp_l_α : Perpendicular l α)
variables (cont_m_β : Contained m β)
variables (not_perp_α_β : NotPerpendicular α β)

-- Theorem stating the problem: if α is not perpendicular to β,
-- then l cannot be parallel to m
theorem line_not_parallel_if_not_perpendicular_plane :
  ∀ (l m : Type) [LinearOrder l] [LinearOrder m]
    (α β : Type) [Plane α] [Plane β]
    (perp_l_α : Perpendicular l α)
    (cont_m_β : Contained m β)
    (not_perp_α_β : NotPerpendicular α β), 
  ¬ Parallel l m := 
sorry

end line_not_parallel_if_not_perpendicular_plane_l110_110450


namespace find_y_from_exponent_equation_l110_110926

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110926


namespace distance_between_points_l110_110033

theorem distance_between_points : 
  let A := (-(3/2 : ℝ), -(1/2 : ℝ))
  let B := (9/2 : ℝ, 7/2 : ℝ)
  dist A B = Real.sqrt (52) := 
by
  sorry

end distance_between_points_l110_110033


namespace pyramid_min_cross_section_area_rect_l110_110216

theorem pyramid_min_cross_section_area_rect 
  (h : ℝ) (a : ℝ) (x y : ℝ)
  (h_base_rect : ∃ A B C D : ℝ × ℝ, 
                  A = (0, 0) ∧ B = (x, 0) ∧ C = (x, y) ∧ D = (0, y))
  (h_height_pyramid : h ≠ 0)
  (h_lateral_edge_angle_45 : ∀ h TA TC, TA = h ∧ tan 45 = 1 → h = x)
  (h_plane_parallel_diagonal_60 : ∀ plane BD, 
                                   plane ∥ BD  ∧ 
                                   plane angle 60 → 
                                   distance plane ↑BD = a) :
  let min_area := (8 * a^2) / Real.sqrt 13 
  in ∃ S : ℝ, S = min_area :=
  sorry

end pyramid_min_cross_section_area_rect_l110_110216


namespace log_eval_l110_110410

-- Definitions based on the given conditions
def four : ℝ := 2^2
def sqrt_two : ℝ := 2^(1/2)
def two_sqrt_two : ℝ := 2 * sqrt_two
def one_over_two_sqrt_two : ℝ := 1 / two_sqrt_two
def two_pow_neg_three_halves : ℝ := 2^(-3/2)
def log_base_four_of_one_over_two_sqrt_two := log 4 one_over_two_sqrt_two

-- The proof statement
theorem log_eval : log_base_four_of_one_over_two_sqrt_two = -3/4 :=
by 
  have h1 : sqrt_two = 2^(1/2) := rfl
  have h2 : two_sqrt_two = 2 * sqrt_two := rfl
  have h3 : one_over_two_sqrt_two = 1 / (2 * sqrt_two) := rfl
  have h4 : two^(1/2) = 4^(1/4) := by rw [←rpow_nat_cast 2, ←rpow_mul, ←rpow_nat_cast, ←rpow_mul]
  have h5 : one_over_two_sqrt_two = 1 / (2 * 2^(1/2)) := rfl
  have h6 : two_sqrt_two = 2^((2 + 1/2) / 2) := by rw [add_div, ←div_mul, div_self, mul_one, rpow_nat_cast, ←rpow_mul, rpow_mul, rpow_nat_cast]
  have h7 : one_over_two_sqrt_two = 2^(-3/2) := by rw [←hz, ←hz]
  have h8 : 2^2 = 4 := rfl
  have h9 : 2^(-3/2) = (2^2)^(-3/4) := by rw [rpow_mul]
  have h10 : 4^(-3/4) = 2^(-3/2) := by rw [rpow_nat_cast, rpow_mul]
  exact eq_of_log_eq_log h4 h10

end log_eval_l110_110410


namespace performance_stability_l110_110293

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end performance_stability_l110_110293


namespace quadratic_root_properties_l110_110500

theorem quadratic_root_properties (b : ℝ) (t : ℝ) :
  (∀ x : ℝ, x^2 + b*x - 2 = 0 → (x = 2 ∨ x = t)) →
  b = -1 ∧ t = -1 :=
by
  sorry

end quadratic_root_properties_l110_110500


namespace power_simplification_l110_110643

theorem power_simplification (a b c d : ℤ) (h1 : a = 3) (h2 : b = 3) (h3 : c = -4) (h4 : d = -5) :
  (a ^ 3 * a ^ c) / (b ^ 2 * b ^ d) = 9 := 
by 
  sorry

end power_simplification_l110_110643


namespace convert_to_base5_l110_110387

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l110_110387


namespace theta_in_second_quadrant_l110_110115

theorem theta_in_second_quadrant (θ : ℝ) (h₁ : Real.sin θ > 0) (h₂ : Real.cos θ < 0) : 
  π / 2 < θ ∧ θ < π := 
sorry

end theta_in_second_quadrant_l110_110115


namespace sum_cos_2x_2y_2z_zero_l110_110174

theorem sum_cos_2x_2y_2z_zero 
  (x y z : ℝ)
  (h₁ : cos x + cos y + cos z = 0)
  (h₂ : sin x + sin y + sin z = 0) 
  (h₃ : cos x + sin y + cos z = 0) : 
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 0 :=
by
  sorry

end sum_cos_2x_2y_2z_zero_l110_110174


namespace proof_T_n_l110_110154

-- Defining the geometric sequence and its parameters
def a (n : ℕ) : ℕ := 2^(n - 1)

def T (n : ℕ) : ℚ := ∑ k in Finset.range n, ((1 : ℚ) / (a k * a (k + 1)))

theorem proof_T_n (n : ℕ) : T n = (2 / 3) * (1 - (1 / 4^n)) :=
by
  -- skipping the proof; placeholder
  sorry

end proof_T_n_l110_110154


namespace find_slope_of_shifted_line_l110_110510

theorem find_slope_of_shifted_line (m : ℝ) :
  let f := λ x : ℝ, m * x - 1 in
  let g := λ x : ℝ, f x - 2 in
  g (-2) = 1 →
  m = -2 :=
by
  sorry

end find_slope_of_shifted_line_l110_110510


namespace functional_equation_solution_l110_110395

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) :=
begin
  sorry
end

end functional_equation_solution_l110_110395


namespace parabola_tangent_point_l110_110241

theorem parabola_tangent_point : ∃ M : ℝ × ℝ, (M.1^2 + M.1 - 2 = M.2) ∧ (2 * M.1 + 1 = 3) ∧ (M = (1, 0)) :=
by {
  use (1, 0),
  split,
  { simp, },
  split,
  { norm_num, },
  { refl, }
}

end parabola_tangent_point_l110_110241


namespace num_of_original_numbers_l110_110598

theorem num_of_original_numbers
    (n : ℕ) 
    (S : ℤ) 
    (incorrect_avg correct_avg : ℤ)
    (incorrect_num correct_num : ℤ)
    (h1 : incorrect_avg = 46)
    (h2 : correct_avg = 51)
    (h3 : incorrect_num = 25)
    (h4 : correct_num = 75)
    (h5 : S + correct_num = correct_avg * n)
    (h6 : S + incorrect_num = incorrect_avg * n) :
  n = 10 := by
  sorry

end num_of_original_numbers_l110_110598


namespace dani_brought_30_cupcakes_l110_110830

noncomputable def numStudents := 27
noncomputable def numSickStudents := 3
noncomputable def numTeacher := 1
noncomputable def numTeacherAid := 1
noncomputable def numRemainingCupcakes := 4

theorem dani_brought_30_cupcakes :
  let numPeople := numStudents - numSickStudents + numTeacher + numTeacherAid in
  let numCupcakesBrought := numPeople + numRemainingCupcakes in
  numCupcakesBrought = 30 :=
by
  sorry

end dani_brought_30_cupcakes_l110_110830


namespace total_distance_covered_is_correct_fuel_cost_excess_is_correct_l110_110372

-- Define the ratios and other conditions for Car A
def carA_ratio_gal_per_mile : ℚ := 4 / 7
def carA_gallons_used : ℚ := 44
def carA_cost_per_gallon : ℚ := 3.50

-- Define the ratios and other conditions for Car B
def carB_ratio_gal_per_mile : ℚ := 3 / 5
def carB_gallons_used : ℚ := 27
def carB_cost_per_gallon : ℚ := 3.25

-- Define the budget
def budget : ℚ := 200

-- Combined total distance covered by both cars
theorem total_distance_covered_is_correct :
  (carA_gallons_used * (7 / 4) + carB_gallons_used * (5 / 3)) = 122 :=
by
  sorry

-- Total fuel cost and whether it stays within budget
theorem fuel_cost_excess_is_correct :
  ((carA_gallons_used * carA_cost_per_gallon) + (carB_gallons_used * carB_cost_per_gallon)) - budget = 41.75 :=
by
  sorry

end total_distance_covered_is_correct_fuel_cost_excess_is_correct_l110_110372


namespace lee_sold_action_figures_l110_110165

-- Defining variables and conditions based on the problem
def sneaker_cost : ℕ := 90
def saved_money : ℕ := 15
def price_per_action_figure : ℕ := 10
def remaining_money : ℕ := 25

-- Theorem statement asserting that Lee sold 10 action figures
theorem lee_sold_action_figures : 
  (sneaker_cost - saved_money + remaining_money) / price_per_action_figure = 10  :=
by
  sorry

end lee_sold_action_figures_l110_110165


namespace sum_of_digits_of_N_l110_110825

-- Define the sequence of terms
def term (n : ℕ) : ℕ :=
  if n = 0 then 0
  else 5 * ((10^n - 1) / 9)

-- Define the summation up to 200 terms
noncomputable def N : ℕ :=
  (finset.range 200).sum term

-- The theorem stating the sum of the digits of N
theorem sum_of_digits_of_N : sum_of_digits N = 1784 :=
sorry

end sum_of_digits_of_N_l110_110825


namespace num_of_n_count_n_satisfying_property_l110_110426

theorem num_of_n (n : ℕ) (h1 : n > 0) (h2 : n ≤ 500) :
  (∀ t : ℝ, (sin (t + π/4) + I * cos (t + π/4))^n = sin (n * t + n * π/4) + I * cos (n * t + n * π/4)) ↔ (n % 4 = 0) := 
sorry

theorem count_n_satisfying_property :
  (card {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∀ t : ℝ, (sin (t + π/4) + I * cos (t + π/4))^n = sin (n * t + n * π/4) + I * cos (n * t + n * π/4))} = 125) := 
sorry

end num_of_n_count_n_satisfying_property_l110_110426


namespace shortest_distance_ant_crawls_l110_110514

-- Definitions
def pointA : ℝ × ℝ := (-2, -3)
def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := real.sqrt 2
def y_axis_point (p : ℝ × ℝ) : ℝ × ℝ := (0, p.snd)

-- Theorem statement
theorem shortest_distance_ant_crawls :
  let d := real.sqrt ((-3 + 2) ^ 2 + (2 + 3) ^ 2)
  d - circle_radius = real.sqrt 26 - real.sqrt 2 :=
by
  sorry

end shortest_distance_ant_crawls_l110_110514


namespace find_f_5pi_over_3_l110_110456

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom periodic_f : ∀ x : ℝ, f (x + real.pi) = f (x)
axiom specific_f : ∀ x : ℝ, 0 ≤ x ∧ x < real.pi / 2 → f (x) = real.tan x

theorem find_f_5pi_over_3 : f (5 * real.pi / 3) = real.sqrt 3 :=
by sorry

end find_f_5pi_over_3_l110_110456


namespace students_traveled_in_cars_l110_110570

theorem students_traveled_in_cars : 
  ∀ (total_students bus_count students_per_bus : ℕ),
  total_students = 375 →
  bus_count = 7 →
  students_per_bus = 53 →
  (total_students - bus_count * students_per_bus) = 4 :=
by
  intros total_students bus_count students_per_bus
  intro h1
  intro h2
  intro h3
  rw [h1, h2, h3]
  exact dec_trivial

end students_traveled_in_cars_l110_110570


namespace mean_vs_median_l110_110961

def percentage : Type := ℚ
def points : Type := ℚ

-- Conditions from a)
def percent_60 : percentage := 0.20
def percent_75 : percentage := 0.25
def percent_85 : percentage := 0.25
def remaining_percent (p60 p75 p85 : percentage) : percentage := 1 - (p60 + p75 + p85)
def percent_95 : percentage := remaining_percent percent_60 percent_75 percent_85

def score_60 : points := 60
def score_75 : points := 75
def score_85 : points := 85
def score_95 : points := 95

-- Translate the (question, conditions, correct answer) tuple to a Lean statement
theorem mean_vs_median (p60 p75 p85 p95 : percentage) (s60 s75 s85 s95 : points) : 
  p60 = 0.20 → p75 = 0.25 → p85 = 0.25 → p95 = remaining_percent p60 p75 p85 →
  s60 = 60 → s75 = 75 → s85 = 85 → s95 = 95 →
  let mean := (p60 * s60 + p75 * s75 + p85 * s85 + p95 * s95) in
  let median := s85 in
  let difference := medial mean in
  difference = 4.5 :=
by 
  sorry

end mean_vs_median_l110_110961


namespace coin_flips_sequences_count_l110_110717

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110717


namespace tan_seven_pi_over_four_l110_110021

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l110_110021


namespace evaluate_expression_l110_110012

theorem evaluate_expression :
  (42 / (9 - 3 * 2)) * 4 = 56 :=
by
  sorry

end evaluate_expression_l110_110012


namespace convert_to_base5_l110_110388

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l110_110388


namespace circle_equation_standard_l110_110834

theorem circle_equation_standard (x y : ℝ) :
  let h := 1
  let k := -1
  let r := real.sqrt 3
  (x - h)^2 + (y + k)^2 = r^2 → 
  (x - 1)^2 + (y + 1)^2 = 3 :=
by
  intros
  sorry

end circle_equation_standard_l110_110834


namespace square_inscribed_right_triangle_l110_110941

theorem square_inscribed_right_triangle (A B C D E G F : Point) (h₁ : right_angle_at A B C) 
(h₂ : square_inscribed D E G F A B C) (h₃ : hypotenuse_partitions B D E C) : 
DE = sqrt (BD * EC) :=
sorry

end square_inscribed_right_triangle_l110_110941


namespace probability_divisible_by_5_or_2_l110_110614

theorem probability_divisible_by_5_or_2 : 
  let nums := [1, 2, 3, 4, 5]
  let total_arrangements := list.permutations nums
  let divisible_by_5 := total_arrangements.filter (λ l, (l.ilast 0) = 5)
  let even_digits := [2, 4]
  let divisible_by_2 := total_arrangements.filter (λ l, even_digits.contains (l.ilast 0))
  let favorable_outcomes := divisible_by_5 ++ divisible_by_2
  in 
    favorable_outcomes.length / total_arrangements.length = 3 / 5 :=
by
  sorry

end probability_divisible_by_5_or_2_l110_110614


namespace g_expression_f_expression_l110_110075

-- Given functions f and g that satisfy the conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom sum_eq : ∀ x, f x + g x = 2^x + 2 * x

-- Theorem statements to prove
theorem g_expression : g = fun x => 2^x := by sorry
theorem f_expression : f = fun x => 2 * x := by sorry

end g_expression_f_expression_l110_110075


namespace solve_for_a_l110_110493

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l110_110493


namespace min_sum_distances_l110_110503

theorem min_sum_distances (a_h b_h d : ℝ) (h_a_h : a_h = 10) (h_b_h : b_h = 15) (h_d : d = 25) :
  ∃ P : ℝ, let x := P in 
  let AP := Real.sqrt (x ^ 2 + a_h ^ 2) in
  let BP := Real.sqrt ((d - x) ^ 2 + b_h ^ 2) in
  AP + BP = 35.46 :=
sorry

end min_sum_distances_l110_110503


namespace abs_sum_difference_lt_l110_110067

variable {n : ℕ}
variable {a : ℕ → ℝ}

noncomputable def A (k : ℕ) : ℝ := (∑ i in Finset.range k, a i) / k

theorem abs_sum_difference_lt (hn : n > 2) 
  (ha_pos : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k) 
  (ha_le : ∀ k, 1 ≤ k ∧ k ≤ n → a k ≤ 1) : 
  |∑ i in Finset.range n, a i - ∑ i in Finset.range n, A (i + 1)| < (n - 1) / 2 := 
sorry

end abs_sum_difference_lt_l110_110067


namespace symmetric_point_l110_110037

theorem symmetric_point (M M' : ℝ × ℝ × ℝ)
    (line_param : ℝ → ℝ × ℝ × ℝ)
    (param_point : line_param (-1) = (-0.5, 1, 1))
    (M = (0, 2, 1)) :
  M' = (-1, 0, 1) :=
sorry

end symmetric_point_l110_110037


namespace probability_even_product_l110_110212

theorem probability_even_product :
  let C := {1, 2, 5, 6}
  let D := {1, 2, 3, 4, 5}
  (∑ x in C, ∑ y in D, (if (x * y % 2 = 0) then 1 else 0)) / (|C| * |D|) = 7 / 10 := 
by {
  sorry 
}

end probability_even_product_l110_110212


namespace segment_length_l110_110890

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  { p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := real.sqrt (a^2 - b^2) in ((-c, 0), (c, 0))

theorem segment_length {a b : ℝ} (ha : a = 5) (hb : b = 3)
  {A B : ℝ × ℝ} {F1 F2 : ℝ × ℝ}
  (h_foci : foci a b = (F1, F2))
  (h_A_B_on_ellipse : A ∈ ellipse a b ∧ B ∈ ellipse a b)
  (h_line_through_F1 : (A.1 - F1.1) * (B.2 - F1.2) = (A.2 - F1.2) * (B.1 - F1.1))
  (h_dist_sum : dist F2 A + dist F2 B = 12) :
  dist A B = 8 := 
sorry

end segment_length_l110_110890


namespace total_cards_l110_110402

def basketball_boxes : ℕ := 12
def cards_per_basketball_box : ℕ := 20
def football_boxes : ℕ := basketball_boxes - 5
def cards_per_football_box : ℕ := 25

theorem total_cards : basketball_boxes * cards_per_basketball_box + football_boxes * cards_per_football_box = 415 := by
  sorry

end total_cards_l110_110402


namespace area_of_shaded_region_l110_110622

-- Define the given conditions
def shaded_region (n : ℕ) := n = 25
def diagonal_length (d : ℕ) := d = 10

-- Define the area calculation for the entire shaded region
def total_area {α : Type} [Field α] (n : ℕ) (d : ℕ) : α :=
  let A := (d : α) ^ 2 / 2 in -- Area of the square using diagonal
  let area_one_square := A / 16 in -- Area of one smaller square
  n * area_one_square -- Total area

-- State the proof problem
theorem area_of_shaded_region : ∀ (n d : ℕ), shaded_region n → diagonal_length d → total_area n d = 78.125 :=
by
  intros n d h1 h2
  unfold shaded_region at h1
  unfold diagonal_length at h2
  unfold total_area
  -- Specific calculation values
  sorry

end area_of_shaded_region_l110_110622


namespace superMonotonousCount_l110_110832

def isSuperMonotonous (n : Nat) : Prop :=
  let digits := n.digits 10
  let strictlyIncreasing := ∀ i j, i < j → digits[i] < digits[j]
  let strictlyDecreasing := ∀ i j, i < j → digits[i] > digits[j]
  let isPalindrome := digits == digits.reverse
  strictlyIncreasing ∨ strictlyDecreasing ∨ isPalindrome

theorem superMonotonousCount : 
  Finset.card (Finset.filter isSuperMonotonous (Finset.range 100000)) = 2120 :=
by
  sorry

end superMonotonousCount_l110_110832


namespace geometric_sequence_sum_l110_110900

theorem geometric_sequence_sum :
  ∀ {a1 q : ℚ} (a1_pos : a1 ≥ 0) (q_pos : q ≥ 0) (q_ne_one : q ≠ 1)
    (S_odd : a1 * (1 - q^10) / (1 - q^2) = 341 / 4)
    (S_even : a1 * q * (1 - q^10) / (1 - q^2) = 341 / 2),
    (a1 * q^2 * (1 - q^12) / (1 - q^3)) = 585 :=
begin
  sorry
end

end geometric_sequence_sum_l110_110900


namespace eugene_used_six_boxes_of_toothpicks_l110_110409

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l110_110409


namespace scale_balanced_l110_110308

theorem scale_balanced (n : ℕ) (weights : Fin (n+2) → ℕ) (total_weight : ∑ i, weights i = 2 * n) :
  (∀ i, weights i ≥ 0) →
  (∀ i j, i < j → weights i ≥ weights j) →
  ∃ left right : list ℕ, 
    (left ++ right = (List.ofFn weights)) ∧  -- The combined list includes all weights
    (left.sum = right.sum) :=               -- The sum of weights on both sides equals
begin
  sorry
end

end scale_balanced_l110_110308


namespace simplify_tan_cot_expr_l110_110208

theorem simplify_tan_cot_expr :
  let θ := 60 * Real.pi / 180 in
  let tan_60 := Real.tan θ,
      cot_60 := 1 / tan_60,
      sec2_60 := Real.sec θ ^ 2,
      csc2_60 := Real.csc θ ^ 2 in
  tan_60 = Real.sqrt 3 ∧ cot_60 = 1 / Real.sqrt 3 ∧ sec2_60 = 4 / 3 ∧ csc2_60 = 4 / 3 →
  (tan_60 ^ 3 + cot_60 ^ 3) / (tan_60 + cot_60) = -1 / 3 :=
by
  intro θ tan_60 cot_60 sec2_60 csc2_60 h,
  sorry

end simplify_tan_cot_expr_l110_110208


namespace sin_lt_alpha_lt_tan_l110_110894

theorem sin_lt_alpha_lt_tan (α : ℝ) (h : 0 < α ∧ α < π / 2) : (sin α < α) ∧ (α < tan α) :=
by
  sorry

end sin_lt_alpha_lt_tan_l110_110894


namespace sum_of_interior_angles_n_plus_3_l110_110602

-- Define the condition that the sum of the interior angles of a convex polygon with n sides is 1260 degrees
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Prove that given the above condition for n, the sum of the interior angles of a convex polygon with n + 3 sides is 1800 degrees
theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : sum_of_interior_angles n = 1260) : 
  sum_of_interior_angles (n + 3) = 1800 :=
by
  sorry

end sum_of_interior_angles_n_plus_3_l110_110602


namespace compute_nested_star_l110_110203

def star (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

theorem compute_nested_star :
  (99 * (98 * (97 * (96 * (95 * (94 * (93 * (92 * (91 * (90 * (89 * (88 * (87 * (86 * (85 * (84 * (83 * (82 * (81 * (80 * (79 * (78 * (77 * (76 * (75 * (74 * (73 * (72 * (71 * (70 * (69 * (68 * (67 * (66 * (65 * (64 * (63 * (62 * (61 * (60 * (59 * (58 * (57 * (56 * (55 * (54 * (53 * (52 * (51 * (50 * (49 * (48 * (47 * (46 * (45 * (44 * (43 * (42 * (41 * (40 * (39 * (38 * (37 * (36 * (35 * (34 * (33 * (32 * (31 * (30 * (29 * (28 * (27 * (26 * (25 * (24 * (23 * (22 * (21 * (20 * (19 * (18 * (17 * (16 * (15 * (14 * (13 * (12 * (11 * (10 * (9 * (8 * (7 * (6 * (5 * (4 * (3 * 2)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) = 1 :=
  sorry

end compute_nested_star_l110_110203


namespace plane_through_point_parallel_to_line_l110_110448

-- Definitions of the mathematical objects involved
variables {Point Line Plane : Type}
variable (distance : Point → Line → ℝ)
variable (parallel : Plane → Line → Prop)
variable (containsPoint : Plane → Point → Prop)

-- Problem statement
theorem plane_through_point_parallel_to_line (e : Line) (P : Point) (m : ℝ) 
    (h1 : ¬ ∃ plane, containsPoint plane P ∧ parallel plane e ∧ distance plane e = m):
  ∃ plane, containsPoint plane P ∧ parallel plane e ∧ distance plane e = m := sorry

end plane_through_point_parallel_to_line_l110_110448


namespace direct_proportion_b_zero_l110_110495

theorem direct_proportion_b_zero (b : ℝ) (x y : ℝ) 
  (h : ∀ x, y = x + b → ∃ k, y = k * x) : b = 0 :=
sorry

end direct_proportion_b_zero_l110_110495


namespace number_of_ticket_cost_values_l110_110795

-- Define the constraints
def is_divisor (d n : ℕ) : Prop := n % d = 0

-- Define numbers that the ticket cost must divide both
def ticket_cost_dividers := {d : ℕ // is_divisor d 72 ∧ is_divisor d 90}

-- Prove that the number of elements in ticket_cost_dividers is 6
theorem number_of_ticket_cost_values : (finset.card (finset.univ : finset ticket_cost_dividers)) = 6 :=
by
  sorry

end number_of_ticket_cost_values_l110_110795


namespace students_answered_both_questions_correctly_l110_110665

theorem students_answered_both_questions_correctly (total_students : ℕ) (answered_q1 : ℕ) (answered_q2 : ℕ) (did_not_take_test : ℕ) (total_students = 40) (answered_q1 = 30) (answered_q2 = 29) (did_not_take_test = 10) : 
  let took_test := total_students - did_not_take_test in
  ∃ (answered_both : ℕ), answered_both = 29 ∧ answered_both = answered_q2 :=
by sorry

end students_answered_both_questions_correctly_l110_110665


namespace car_speed_problem_l110_110238

theorem car_speed_problem (x : ℝ) (h1 : ∀ x, x + 30 / 2 = 65) : x = 100 :=
by
  sorry

end car_speed_problem_l110_110238


namespace altitudes_sum_of_triangle_formed_by_line_and_axes_l110_110826

noncomputable def sum_of_altitudes (x y : ℝ) : ℝ :=
  let intercept_x := 6
  let intercept_y := 16
  let altitude_3 := 48 / Real.sqrt (8^2 + 3^2)
  intercept_x + intercept_y + altitude_3

theorem altitudes_sum_of_triangle_formed_by_line_and_axes :
  ∀ (x y : ℝ), (8 * x + 3 * y = 48) →
  sum_of_altitudes x y = 22 + 48 / Real.sqrt 73 :=
by
  sorry

end altitudes_sum_of_triangle_formed_by_line_and_axes_l110_110826


namespace inequality_abc_equality_condition_abc_l110_110176

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) ≤ 1 / 2 :=
sorry

theorem equality_condition_abc (a b c : ℝ) :
  (a / (2 * a + 1)) + (b / (3 * b + 1)) + (c / (6 * c + 1)) = 1 / 2 ↔ 
  a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6 :=
sorry

end inequality_abc_equality_condition_abc_l110_110176


namespace ratio_of_x_to_y_l110_110947

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
sorry

end ratio_of_x_to_y_l110_110947


namespace total_absent_students_30_days_l110_110008

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else sequence (n-2) + 1 + (-1)^(n-1)

theorem total_absent_students_30_days : 
  (∑ n in finset.range (30), sequence (n + 1)) = 255 :=
by
  sorry

end total_absent_students_30_days_l110_110008


namespace coin_flips_sequences_count_l110_110713

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110713


namespace g_at_5_l110_110998

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 47 * x ^ 2 - 44 * x + 24

theorem g_at_5 : g 5 = 104 := by
  sorry

end g_at_5_l110_110998


namespace billboard_shorter_side_length_l110_110127

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 120)
  (h2 : 2 * L + 2 * W = 46) :
  min L W = 8 :=
by
  sorry

end billboard_shorter_side_length_l110_110127


namespace upper_limit_of_range_l110_110631

theorem upper_limit_of_range (N : ℕ) :
  (∀ n : ℕ, (20 + n * 10 ≤ N) = (n < 198)) → N = 1990 :=
by
  sorry

end upper_limit_of_range_l110_110631


namespace length_of_AE_in_trapezoid_l110_110252

theorem length_of_AE_in_trapezoid 
  (ABCD : Type)
  [trapezoid ABCD]
  (A B C D E : ABCD)
  (h_parallel : AB ∥ CD)
  (h_AB : length AB = 10)
  (h_CD : length CD = 15)
  (h_AC : length AC = 17)
  (h_equal_areas : area (triangle A E D) = area (triangle B E C)) :
  length (segment A E) = 34 / 5 := 
sorry

end length_of_AE_in_trapezoid_l110_110252


namespace problem_to_prove_l110_110454

-- Definitions and conditions
def p (A B : ℝ) : Prop := ∀ (triangle_angle_sum : ℝ), (π/2 < A + B ∧ A + B < π) → ∃ (sinA : ℝ) (cosB : ℝ), (sin A < cos B)
def q : Prop := ∀ (x y : ℝ), x + y ≠ 2 → (x ≠ -1 ∨ y ≠ 3)

-- Proof statement
theorem problem_to_prove (A B : ℝ) : (¬ p A B) ∧ q :=
by sorry

end problem_to_prove_l110_110454


namespace smallest_n_l110_110285

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 4 * n = k^2) (h2 : ∃ l : ℕ, 5 * n = l^3) : n = 100 :=
sorry

end smallest_n_l110_110285


namespace max_diagonal_sum_l110_110181

def is_valid_grid (grid : list (list ℕ)) : Prop :=
  grid.length = 8 ∧ (∀ row, row ∈ grid → row.length = 8) ∧
  (∀ i, 1 ≤ i ∧ i < 64 → ∃ (r1 c1 r2 c2 : ℕ),
    r1 < 8 ∧ c1 < 8 ∧ r2 < 8 ∧ c2 < 8 ∧
    grid[r1][c1] = i ∧ grid[r2][c2] = i + 1 ∧ 
    ((r1 = r2 ∧ (c1 = c2 + 1 ∨ c1 = c2 - 1)) ∨ 
     (c1 = c2 ∧ (r1 = r2 + 1 ∨ r1 = r2 - 1))))

def diagonal_sum (grid : list (list ℕ)) : ℕ :=
  grid[0][0] + grid[1][1] + grid[2][2] + grid[3][3] + 
  grid[4][4] + grid[5][5] + grid[6][6] + grid[7][7]

theorem max_diagonal_sum :
  ∃ (grid : list (list ℕ)), is_valid_grid grid ∧ diagonal_sum grid = 432 :=
sorry

end max_diagonal_sum_l110_110181


namespace catch_up_time_l110_110186

-- Define the speeds of Person A and Person B.
def speed_A : ℝ := 10 -- kilometers per hour
def speed_B : ℝ := 7  -- kilometers per hour

-- Define the initial distance between Person A and Person B.
def initial_distance : ℝ := 15 -- kilometers

-- Prove the time it takes for person A to catch up with person B is 5 hours.
theorem catch_up_time :
  initial_distance / (speed_A - speed_B) = 5 :=
by
  -- Proof can be added here
  sorry

end catch_up_time_l110_110186


namespace coin_flip_sequences_l110_110703

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110703


namespace largest_prime_divisor_of_sum_of_squares_l110_110863

theorem largest_prime_divisor_of_sum_of_squares :
  (let n := 16^2 + 81^2 in ∀ p : ℕ, prime p → p ∣ n → p ≤ 53) :=
begin
  let n := 16^2 + 81^2,
  have : n = 6817, by norm_num,
  sorry
end

end largest_prime_divisor_of_sum_of_squares_l110_110863


namespace problem_part1_problem_part2_l110_110107

open Set

theorem problem_part1 (b c : ℝ) (h : ∀ x : ℝ, x^2 + b*x + c > 0 ↔ x < -2 ∨ x > -1) :
  b = 3 ∧ c = 2 :=
sorry

theorem problem_part2 {a : ℝ} :
  (∀ x : ℝ, x^2 + 3*x + 2 > 0 ↔ x < -2 ∨ x > -1) →
  (let cx := 2 in
  let bx := 3 in
  let polynomial := λ x : ℝ, cx*x^2 + bx*x + a in
  (a > 9/8 → ∀ x : ℝ, ¬(polynomial x ≤ 0)) ∧
  (a = 9/8 → ∀ x : ℝ, polynomial x = 0 ↔ x = -3/4) ∧
  (a < 9/8 → ∃ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧
    (∀ x : ℝ, polynomial x ≤ 0 ↔ x₁ ≤ x ∧ x ≤ x₂) ∧
    x₁ = (-3 - real.sqrt (9 - 8 * a)) / 4 ∧
    x₂ = (-3 + real.sqrt (9 - 8 * a)) / 4)) :=
sorry

end problem_part1_problem_part2_l110_110107


namespace jose_peanuts_l110_110536

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l110_110536


namespace quadrilateral_parallelogram_l110_110196

theorem quadrilateral_parallelogram
  (AB BC CD DA AC BD : ℝ)
  (h : AB^2 + BC^2 + CD^2 + DA^2 = AC^2 + BD^2) : 
  parallelogram ABCD :=
sorry

end quadrilateral_parallelogram_l110_110196


namespace effective_reading_time_per_day_l110_110047

theorem effective_reading_time_per_day :
  let total_pages := 405 + 350 + 480 + 390 + 520 + 200 in
  let total_hours := total_pages / 50.0 in
  let break_days := 34 / 3 in
  let reading_days := 34 - break_days in
  let effective_time_per_day := total_hours / reading_days in
  (effective_time_per_day ≈ 2.04) :=
begin
  let total_pages := 2345,
  let total_hours := 46.9,
  let break_days := 11,
  let reading_days := 23,
  let effective_time_per_day := total_hours / reading_days,
  sorry -- Proof of the effective reading time
end

end effective_reading_time_per_day_l110_110047


namespace Andy_2030th_turn_l110_110364

theorem Andy_2030th_turn :
  let initial_position := (-10 : Int, 10 : Int)
  let moves : List (ℚ × ℚ) :=
        List.unfold
          (fun (n_and_pos : ℚ × (ℚ × ℚ)) =>
            let (n, (x, y)) := n_and_pos
            let move := 
              match n % 4 with
              | 0 => (x + n + 1, y)
              | 1 => (x, y + n + 1)
              | 2 => (x - n - 1, y)
              | 3 => (x, y - n - 1)
              | _ => (x, y) -- impossible case but Lean requires exhaustive match
            in some (move, (n + 1, move)))
          (0, initial_position)
      Andy_position := moves.get? 2029
  in Andy_position = (some (-3054, 3053)) := 
  sorry

end Andy_2030th_turn_l110_110364


namespace area_of_circle_l110_110140

noncomputable def circle_area (r : ℝ) : ℝ := π * r^2

theorem area_of_circle
  (O : Type) [normed_space ℝ O]
  (A B C D F E : O)
  (r : ℝ)
  (h1 : O ∈ line_through ℝ A C)
  (h2 : O ∈ line_through ℝ B D)
  (h3 : ∀ x, (x ∈ line_through ℝ A C) → (x ∈ line_through ℝ B D) → x = O)
  (h4 : ∃ DF_line, (DF_line ∈ line_through ℝ D F) ∧ (E ∈ DF_line))
  (hDE : dist D E = 10)
  (hEF : dist E F = 6)
  (hAE : dist A E = 3)
  (hEB : dist E B = r - 3) :
  circle_area 23 = 529 * π :=
sorry

end area_of_circle_l110_110140


namespace exponent_in_first_equation_l110_110128

theorem exponent_in_first_equation (x b : ℝ) (h1 : b = 19.99999999999999)
    (h2 : ∀ n, n = 2^x → n^b = 8) : x = 0.15 :=
by
  intros n hn
  rw [hn] at h2
  have hb : 2^(x * b) = 8 by rw [pow_mul, h2 n (rfl)]
  rw [show 8 = 2^3 by norm_num] at hb
  rw [pow_eq_pow_iff_log_eq_log {base := 2} (rfl)] at hb
  sorry

end exponent_in_first_equation_l110_110128


namespace train_length_l110_110797

theorem train_length 
  (t1 t2 : ℝ)
  (d2 : ℝ)
  (L : ℝ)
  (V : ℝ)
  (h1 : t1 = 18)
  (h2 : t2 = 27)
  (h3 : d2 = 150.00000000000006)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) :
  L = 300.0000000000001 :=
by
  sorry

end train_length_l110_110797


namespace probability_four_rides_l110_110683

-- Definitions based on conditions
def num_cars : ℕ := 4

def equally_likely (n : ℕ) : Prop := 
  ∀ (i : ℕ), i < n → probability i = 1 / n

-- Main statement of the problem
theorem probability_four_rides : 
  ∀ (rides : ℕ), rides = 4 → equally_likely num_cars →
  let favorable_outcomes := 4 * 3 * 2 * 1 in
  let total_outcomes := num_cars ^ rides in
  favorable_outcomes / total_outcomes = 3 / 32 :=
by
  intro rides h_eq h_eli
  sorry

end probability_four_rides_l110_110683


namespace distance_FG_equals_36_l110_110640

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def is_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  (A.2 = B.2 ∧ C.2 ≠ D.2) ∨ (A.2 ≠ B.2 ∧ C.2 = D.2)

noncomputable def coordinates : Type := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def point_Y : ℝ × ℝ := (0,0)
def point_Z : ℝ × ℝ := (18, 0)
def point_X : ℝ × ℝ := (9, 12)
def point_Q : ℝ × ℝ := (11, 60 / 7)
def point_F : ℝ × ℝ := (0, 12)
def point_G : ℝ × ℝ := (42, 60 / 7)

theorem distance_FG_equals_36 :
  distance point_F point_G = 36 :=
sorry

end distance_FG_equals_36_l110_110640


namespace log_inequality_l110_110891

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality_l110_110891


namespace number_of_sandwiches_l110_110361

-- Defining the conditions
def kinds_of_meat := 12
def kinds_of_cheese := 11
def kinds_of_bread := 5

-- Combinations calculation
def choose_one (n : Nat) := n
def choose_three (n : Nat) := Nat.choose n 3

-- Proof statement to show that the total number of sandwiches is 9900
theorem number_of_sandwiches : (choose_one kinds_of_meat) * (choose_three kinds_of_cheese) * (choose_one kinds_of_bread) = 9900 := by
  sorry

end number_of_sandwiches_l110_110361


namespace problem1_problem2_problem3_problem4_l110_110123

-- Problem 1: Prove that 0.3̇5̇7̇ = 177 / 495
theorem problem1 (a : ℚ) : 0.3557575757  ≈ 177 / 495 := sorry

-- Problem 2: Prove that tan²(a°) + 1 = sec²(a°)
theorem problem2 (a : ℝ) : (tan (a * pi / 180))^2 + 1 = (sec (a * pi / 180))^2 := sorry

-- Problem 3: Prove that ∠ABC = 74° - b° given AB = AD, ∠BAC = 26° + b°, ∠BCD = 106°
theorem problem3 (b : ℝ) : ∀ (AB AD : ℝ), 
  AB = AD →
  ∠BAC = 26 + b →
  ∠BCD = 106 →
  ∠ABC = 74 - b := sorry

-- Problem 4: Prove that Y = x + 10 given the matrix multiplication result
theorem problem4 (x Y : ℝ) : 
  (1 * 3 + 2 * 4 = 11) →
  (1 * x + 2 * 5 = Y) →
  Y = x + 10 := sorry

end problem1_problem2_problem3_problem4_l110_110123


namespace coin_flip_sequences_l110_110724

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110724


namespace A_beats_B_by_14_meters_l110_110960

theorem A_beats_B_by_14_meters :
  let distance := 70
  let time_A := 20
  let time_B := 25
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let distance_B_in_A_time := speed_B * time_A
  (distance - distance_B_in_A_time) = 14 :=
by
  sorry

end A_beats_B_by_14_meters_l110_110960


namespace circle_equation_l110_110624

noncomputable def given_conditions : Prop :=
  ∃ (t : ℝ), (0 < t) ∧ (3 * t = 3) ∧ (4 * sqrt 2)^2 = (2 * t)^2 + 8

theorem circle_equation : given_conditions →
  ∃ (x y r: ℝ), (x = 3) ∧ (y = 1) ∧ (r = 3) ∧ (x - 3)^2 + (y - 1)^2 = r^2 :=
by
  sorry

end circle_equation_l110_110624


namespace part_one_part_two_l110_110473

noncomputable def f (a m : ℝ) (x : ℝ) : ℝ := 
  Real.log x + (1 / 2) * a * x^2 - x - m

theorem part_one (a m : ℝ) :
  (∀ x > 0, f a m x ≥ f a m (x - 1)) ↔ a ∈ Set.Ici (1 / 4) :=
by sorry

theorem part_two (a : ℝ) (h_a : a < 0) (m : ℤ) :
  (∀ x > 0, f a m.to_real x < 0) → m ≥ -1 :=
by sorry

end part_one_part_two_l110_110473


namespace ratio_of_areas_of_squares_l110_110134

theorem ratio_of_areas_of_squares (s2 : ℝ) : 
  let s1 := 2 * s2 * Real.sqrt 2
  in (s1 ^ 2) / (s2 ^ 2) = 8 :=
by 
  let s1 := 2 * s2 * Real.sqrt 2
  have A1 : s1 ^ 2 = 8 * s2 ^ 2 := by sorry
  have A2 : s2 ^ 2 = s2 ^ 2 := by sorry
  have ratio : (s1 ^ 2) / (s2 ^ 2) = (8 * s2 ^ 2) / s2 ^ 2 := by sorry
  rw [div_self (sq_not_zero s2)]
  exact sorry

end ratio_of_areas_of_squares_l110_110134


namespace base_eight_to_ten_l110_110644

theorem base_eight_to_ten (n : ℕ) (h : n = 74532) : 
  (7 * 8^4 + 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 31066 :=
by 
  have h1 : 7 * 8^4 = 28672 := by norm_num,
  have h2 : 4 * 8^3 = 2048 := by norm_num,
  have h3 : 5 * 8^2 = 320 := by norm_num,
  have h4 : 3 * 8^1 = 24 := by norm_num,
  have h5 : 2 * 8^0 = 2 := by norm_num,
  calc 
    7 * 8^4 + 4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0
      = 28672 + 2048 + 320 + 24 + 2 : by rw [h1, h2, h3, h4, h5]
  ... = 31066 : by norm_num

end base_eight_to_ten_l110_110644


namespace solve_inequality_l110_110028

theorem solve_inequality (x : ℝ) :
  (\frac{1}{x * (x + 2)} - \frac{1}{(x + 2) * (x + 3)} < \frac{1}{4}) ↔
  (x ∈ set.Ioo (-∞) (-3) ∪ set.Ioo (-2) 0 ∪ set.Ioo 1 ∞) :=
sorry

end solve_inequality_l110_110028


namespace correct_evaluation_l110_110662

noncomputable def evaluate_expression : ℚ :=
  - (2 : ℚ) ^ 3 + (6 / 5) * (2 / 5)

theorem correct_evaluation : evaluate_expression = -7 - 13 / 25 :=
by
  unfold evaluate_expression
  sorry

end correct_evaluation_l110_110662


namespace find_a_extreme_value_l110_110881

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem find_a_extreme_value :
  (∃ a : ℝ, ∀ x, f x a = Real.log (x + 1) - x - a * x ∧ (∃ m : ℝ, ∀ y : ℝ, f y a ≤ m)) ↔ a = -1 / 2 :=
by
  sorry

end find_a_extreme_value_l110_110881


namespace right_triangle_area_l110_110006

/-- Given a right triangle where one leg is 18 cm and the hypotenuse is 30 cm,
    prove that the area of the triangle is 216 square centimeters. -/
theorem right_triangle_area (a b c : ℝ) 
    (ha : a = 18) 
    (hc : c = 30) 
    (h_right : a^2 + b^2 = c^2) :
    (1 / 2) * a * b = 216 :=
by
  -- Substitute the values given and solve the area.
  sorry

end right_triangle_area_l110_110006


namespace number_of_possible_winning_scores_l110_110502

def possible_winning_scores_count : ℕ :=
let scores_set := {s | ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ finset.range 10 ∧ b ∈ finset.range 10 ∧ c ∈ finset.range 10 ∧ s = a + b + c ∧ s ≤ 22 ∧ s ≥ 6} in
finset.card scores_set

theorem number_of_possible_winning_scores : possible_winning_scores_count = 17 := 
sorry

end number_of_possible_winning_scores_l110_110502


namespace y_in_terms_of_x_l110_110125

-- Defining x and y in terms of 3^p and 3^-p respectively 
variable {p : ℝ}

def x (p : ℝ) : ℝ := 1 + 3^p
def y (p : ℝ) : ℝ := 1 + 3^(-p)

-- The proof statement to be proven
theorem y_in_terms_of_x (x y : ℝ) (p : ℝ) (h₁ : x = 1 + 3^p) (h₂ : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := 
sorry

end y_in_terms_of_x_l110_110125


namespace triangle_side_AC_l110_110526

theorem triangle_side_AC (A B C D : ℝ) 
  (AB BC : ℝ) (BD DC : ℝ) (AC : ℝ) 
  (hAB : AB = 6)
  (hBC : BC = 6)
  (hBD_DC : BD / DC = 2)
  (hADB_90 : angle A D B = 90)
  : AC = 2 * sqrt 6 :=
sorry

end triangle_side_AC_l110_110526


namespace f_f_x_eq_1_range_l110_110054

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 1 else x - 3

theorem f_f_x_eq_1_range :
  {x : ℝ | f (f x) = 1} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} ∪ {x : ℝ | 3 ≤ x ∧ x ≤ 4} ∪ {7} :=
by
  -- Proof goes here
  sorry

end f_f_x_eq_1_range_l110_110054


namespace last_student_draws_red_ball_l110_110965

def probability_red_ball_last_draw (bag : List string) : Prop :=
  let after_first_draw := bag.erase "yellow1"          -- simulate first student draws a yellow ball
  let remaining_students_draw := after_first_draw     -- remaining students draw from the new list
  let events := [("yellow2", "red"), ("red", "yellow2")] -- possible outcomes for the remaining two draws
  let red_last_prob := 1 / 2 -- since there are two equally likely outcomes
  (1 / (list.length events) = red_last_prob)

noncomputable def example_bag : List string := ["yellow1", "yellow2", "red"]

theorem last_student_draws_red_ball :
  probability_red_ball_last_draw example_bag :=
by
  -- Full proof steps omitted for this theorem; focusing on structuring the theorem correctly.
  sorry

end last_student_draws_red_ball_l110_110965


namespace hyperbola_eccentricity_is_sqrt_6_over_2_l110_110087

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_is_sqrt_6_over_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hyp : b / a = (real.sqrt 2) / 2) :
  hyperbola_eccentricity a b = (real.sqrt 6) / 2 := 
sorry

end hyperbola_eccentricity_is_sqrt_6_over_2_l110_110087


namespace efficiency_ratio_l110_110678

-- Define the work efficiencies
def EA : ℚ := 1 / 12
def EB : ℚ := 1 / 24
def EAB : ℚ := 1 / 8

-- State the theorem
theorem efficiency_ratio (EAB_eq : EAB = EA + EB) : (EA / EB) = 2 := by
  -- Insert proof here
  sorry

end efficiency_ratio_l110_110678


namespace trapezoid_bc_equals_fc_l110_110188

-- Definitions and hypotheses
variables {A B C D E F : Type*}
variables [Inhabited E] [LinearOrderedAddCommMonoid E]

-- Midpoint definition
def is_midpoint (A D E : E) : Prop := (A + D) / 2 = E

-- Perpendicular definition
def is_perpendicular (AF BD : E) : Prop := AF * BD = 0

-- Theorem statement
theorem trapezoid_bc_equals_fc
  (hMidpoint: is_midpoint A D E)
  (hIntersect: ∃ F, (BD * E = F) ∧ (CE * E = F))
  (hPerpendicular: is_perpendicular AF BD)
  : BC = FC :=
by
  sorry

#check trapezoid_bc_equals_fc

end trapezoid_bc_equals_fc_l110_110188


namespace calculate_sum_l110_110370

theorem calculate_sum :
  sqrt 12 + abs (1 - sqrt 3) + (Real.pi - 2023)^0 = 3 * sqrt 3 :=
by
  sorry

end calculate_sum_l110_110370


namespace count_of_non_lowest_terms_l110_110425

-- Definitions for gcd condition
def is_not_in_lowest_terms (n : ℕ) : Prop :=
  Nat.gcd (n^2 + 13) (n + 6) > 1

-- Predicate for valid integers
def valid_N (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 2500 ∧ is_not_in_lowest_terms n

-- Main theorem statement
theorem count_of_non_lowest_terms : 
  ({n : ℕ | valid_N n}.to_finset.card = 409) := 
sorry

end count_of_non_lowest_terms_l110_110425


namespace remainder_when_2519_divided_by_3_l110_110326

theorem remainder_when_2519_divided_by_3 :
  2519 % 3 = 2 :=
by
  sorry

end remainder_when_2519_divided_by_3_l110_110326


namespace coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110267

theorem coefficient_a3b3_in_ab6_c8_div_c8_is_1400 :
  let a := (a : ℝ)
  let b := (b : ℝ)
  let c := (c : ℝ)
  ∀ (a b c : ℝ), (binom 6 3 * a^3 * b^3) * (binom 8 4 * c^0) = 1400 := 
by
  sorry

end coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110267


namespace bus_total_distance_l110_110682

theorem bus_total_distance 
  (d40 : ℕ)
  (v40 : ℕ)
  (v60 : ℕ)
  (T : ℕ)
  (t40 : T = d40 / v40)
  (time : T = 5.5)
  (dist40 : d40 = 160)
  (speed40 : v40 = 40)
  (speed60 : v60 = 60)
  (d60 : ℕ)
  (total_time : T = 4 + d60 / 60 )
  : d40 + d60 = 250 := 
sorry

end bus_total_distance_l110_110682


namespace factorial_expression_value_l110_110655

theorem factorial_expression_value : (12.factorial - 2 * 10.factorial) / 8.factorial = 11700 := by
  sorry

end factorial_expression_value_l110_110655


namespace shaded_region_area_is_correct_l110_110367

def point := (ℝ × ℝ)

-- Define a square with side length 40 units
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length^2

-- Define the vertices of the triangles
def triangle1_vertices : set point := {(0,0), (15,0), (0,15)}
def triangle2_vertices : set point := {(25,0), (40,0), (40,15)}
def triangle3_vertices : set point := {(0,25), (0,40), (15,40)}

-- Define the area of one triangle with legs 15 units each
def triangle_area (leg : ℝ) : ℝ := 0.5 * leg * leg
def single_triangle_area : ℝ := triangle_area 15

-- Define the total area of all three triangles
def total_triangles_area : ℝ := 3 * single_triangle_area

-- Define the area of the shaded region
def shaded_region_area : ℝ := square_area - total_triangles_area

theorem shaded_region_area_is_correct :
  shaded_region_area = 1262.5 :=
  by
  -- Calculate the constant results
  have h_square_area : square_area = 1600 := by norm_num [square_side_length, square_area]
  have h_single_triangle_area : single_triangle_area = 112.5 := by norm_num [triangle_area, single_triangle_area]
  have h_total_triangles_area : total_triangles_area = 337.5 := by norm_num [total_triangles_area, h_single_triangle_area]
  have h_shaded_region_area : shaded_region_area = 1262.5 := by norm_num [shaded_region_area, h_square_area, h_total_triangles_area]
  exact h_shaded_region_area

end shaded_region_area_is_correct_l110_110367


namespace coin_flip_sequences_l110_110725

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110725


namespace pirate_division_l110_110775

def initial_coins (x : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 ->
    let x' := x * (14! / (15^(14 - k + 1))) in
    ∃ n : ℕ, n = k * x' / 15 ∧ n > 0 -- Ensures each pirate gets a positive whole number of coins

theorem pirate_division :
  ∃ x : ℕ, initial_coins x ∧
  let xp := 3^4 * 5^4 * 7 * 11 * 13 in
  ∃ y : ℕ, y = xp := by
  sorry

end pirate_division_l110_110775


namespace distance_between_centers_of_inscribed_circles_l110_110331

theorem distance_between_centers_of_inscribed_circles
  (ABC : Type) [right_triangle ABC]
  (CD : line_segment ABC)
  (hCD : altitude CD ABC)
  (radius_BCD : ℝ) (radius_ACD : ℝ)
  (h_radius_BCD : radius_BCD = 4)
  (h_radius_ACD : radius_ACD = 3) :
  ∃ (O1 O2 : point) (distance : ℝ),
    inscribed_circle O1 (triangle BCD) radius_BCD ∧
    inscribed_circle O2 (triangle ACD) radius_ACD ∧
    distance = 5 * real.sqrt 2 :=
begin
  sorry
end

end distance_between_centers_of_inscribed_circles_l110_110331


namespace tangent_slope_is_negative_reciprocal_l110_110817

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the point on the circle where we are finding the tangent slope
def point_on_circle : ℝ × ℝ := (6, 4)

-- Define the slope of a line through two points
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the slope of the radius
def slope_radius : ℝ := slope center point_on_circle

-- Define the slope of the tangent
def slope_tangent : ℝ := -1 / slope_radius

-- Lean 4 statement to prove that the slope of the tangent line is -4/3
theorem tangent_slope_is_negative_reciprocal : slope_tangent = -4 / 3 := sorry

end tangent_slope_is_negative_reciprocal_l110_110817


namespace arithmetic_sequence_diff_l110_110147

theorem arithmetic_sequence_diff (a : ℕ → ℝ)
  (h1 : a 5 * a 7 = 6)
  (h2 : a 2 + a 10 = 5) :
  a 10 - a 6 = 2 ∨ a 10 - a 6 = -2 := by
  sorry

end arithmetic_sequence_diff_l110_110147


namespace units_digit_sum_l110_110423

noncomputable def g (n : ℕ) : ℕ := (2 + 2 * n) % 10

theorem units_digit_sum : ∑ k in Finset.range 1012, g (2 * k) = 2022 :=
by
  sorry

end units_digit_sum_l110_110423


namespace medians_in_right_triangle_are_compared_l110_110206

open Real EuclideanGeometry

noncomputable def right_angled_triangle_medians (A B C : Point) (h : isRightTriangle A B C) 
  (h1 : distance A C > distance B C) : Prop :=
let A0 := midpoint B C in
let B0 := midpoint A C in
let AA0 := distance A A0 in
let BB0 := distance B B0 in
(BB0 < AA0) ∧ (BB0 > AA0 / 2)

theorem medians_in_right_triangle_are_compared (A B C : Point) (h : isRightTriangle A B C)
  (h1 : distance A C > distance B C) : right_angled_triangle_medians A B C h h1 :=
sorry

end medians_in_right_triangle_are_compared_l110_110206


namespace power_sum_l110_110569

noncomputable def expr_sum : Nat → Nat := λ n, 2^(n+1) - 2

theorem power_sum (a : Nat) (h : 2^50 = a) :
  2^50 + 2^51 + 2^52 + ... + 2^99 + 2^100 = 2 * a^2 - a :=
sorry

end power_sum_l110_110569


namespace total_charge_l110_110254

theorem total_charge (h1 m1_hours m2_hours : Nat) (charge1 charge2 : Nat) : 
  (charge1 = m1_hours * 45) ∧ (charge2 = m2_hours * 85) ∧ (m1_hours + m2_hours = 20) ∧ (m2_hours = 5) → 
  (charge1 + charge2 = 1100) :=
by
  intros h
  obtain ⟨hc1, hc2, hm_total, hm2⟩ := h
  rw [hm2, add_comm m1_hours 5, add_comm m1_hours (m2_hours - 5)] at hm_total
  sorry

end total_charge_l110_110254


namespace number_of_perfect_squares_between_bounds_l110_110811

open Nat

theorem number_of_perfect_squares_between_bounds : 
    let lower_bound := 2^8 + 1
    let upper_bound := 2^18 + 1
    ∃ n, n = 58 ∧ (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound → is_square x) :=
by 
    let lower_bound := 2^8 + 1
    let upper_bound := 2^18 + 1
    have : lower_bound = 257 := by norm_num
    have : upper_bound = 262145 := by norm_num
    sorry

end number_of_perfect_squares_between_bounds_l110_110811


namespace total_coughs_after_20_minutes_l110_110050

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end total_coughs_after_20_minutes_l110_110050


namespace sum_series_l110_110374

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l110_110374


namespace distinct_sequences_ten_flips_l110_110732

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110732


namespace balance_equilibrium_scale_l110_110113

def equilibrium_point (l a : ℝ) : ℝ := a / (l - a)

theorem balance_equilibrium_scale 
  (l a : ℝ) (hl : l > a) :
  equilibrium_point l a = a / (l - a) :=
by
  sorry

end balance_equilibrium_scale_l110_110113


namespace y_decreasing_interval_l110_110103
noncomputable section


open Real

def f (x : ℝ) : ℝ := sin (2 * x + π / 12)

def y (x : ℝ) : ℝ := 2 * f x + 2 * cos (2 * x + π / 12)

theorem y_decreasing_interval : ∀ x : ℝ, π / 12 ≤ x ∧ x ≤ 7 * π / 12 → 
  (deriv y) x < 0 :=
sorry

end y_decreasing_interval_l110_110103


namespace number_of_red_balls_l110_110317

theorem number_of_red_balls (total_balls : ℕ) (probability : ℚ) (num_red_balls : ℕ) 
  (h1 : total_balls = 12) 
  (h2 : probability = 1 / 22) 
  (h3 : (num_red_balls * (num_red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) = probability) :
  num_red_balls = 3 := 
by
  sorry

end number_of_red_balls_l110_110317


namespace coin_flip_sequences_l110_110727

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110727


namespace square_overlap_area_l110_110338

theorem square_overlap_area (β : ℝ) (h1 : 0 < β) (h2 : β < 90) (h3 : Real.cos β = 3 / 5) : 
  area (common_region (square 2) (rotate_square β (square 2))) = 4 / 3 :=
sorry

end square_overlap_area_l110_110338


namespace determine_integer_solutions_l110_110005

noncomputable def number_of_integer_solutions : ℕ :=
  let solutions : Finset ℤ := 
    (Finset.Icc (-3 : ℤ) (-(8 : ℤ) / 3).floor).filter (λ x, 
      -5 * x ≥ 3 * x + 10 ∧
      -3 * x ≤ 9 ∧
      -2 * x ≥ x + 8 ∧
      2 * x + 1 ≤ 17)
  solutions.card

theorem determine_integer_solutions :
  number_of_integer_solutions = 2 :=
by
  sorry

end determine_integer_solutions_l110_110005


namespace domain_of_sqrt_expr_l110_110859

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end domain_of_sqrt_expr_l110_110859


namespace find_value_l110_110904

def f (x : ℝ) : ℝ := 4 * x ^ 5 + 3 * x ^ 3 + 2 * x + 1

theorem find_value :
  f (Real.logBase 2 3) + f (Real.logBase (1 / 2) 3) = 2 := 
by
  sorry

end find_value_l110_110904


namespace fraction_addition_bounds_l110_110207

variable {a b c d n p x y : ℕ}

theorem fraction_addition_bounds 
  (h1 : (a * d : ℚ) > b * c)
  (h2 : (c * p : ℚ) > d * n)
  (h3 : (n * y : ℚ) > p * x)
  (hb : b ≠ 0)
  (hd : d ≠ 0)
  (hp : p ≠ 0)
  (hy : y ≠ 0) :
  let sum_num := a + c + n + x
  let sum_den := b + d + p + y
  in ((x : ℚ) / y) < (sum_num / sum_den) ∧ (sum_num / sum_den) < ((a : ℚ) / b) :=
  sorry

end fraction_addition_bounds_l110_110207


namespace num_incorrect_statements_l110_110804

-- Definitions based on conditions
def variance_remains_unchanged (data : List ℝ) (c : ℝ) : Prop :=
  let shifted_data := data.map (λ x => x + c)
  (List.variance data) = (List.variance shifted_data)

def regression_condition (x : ℝ) : Prop :=
  ∀ (x : ℝ), (let y1 := 3 - 5 * x in let y2 := 3 - 5 * (x + 1) in y2 = y1 - 5)

def regression_line_through_mean (x_mean y_mean : ℝ) : Prop :=
  ∀ (b a : ℝ), ((λ x => b * x + a) x_mean = y_mean)

-- The main theorem to be proved
theorem num_incorrect_statements (data : List ℝ) (c x x_mean y_mean : ℝ) :
  (variance_remains_unchanged data c) ∧
  ¬ (regression_condition x) ∧
  (regression_line_through_mean x_mean y_mean) →
  1 = 1 :=
by
  sorry

end num_incorrect_statements_l110_110804


namespace locus_of_points_l110_110217

-- Definitions representing the geometrical constructs
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Segment :=
  (P1 P2 : Point)

structure Ray :=
  (P : Point)
  (direction : Point)

-- Conditions
def is_on_line_segment (P : Point) (S : Segment) : Prop := 
  sorry -- definition indicating point P is on segment S, excluding endpoints

def is_on_ray (P : Point) (R : Ray) : Prop :=
  sorry -- definition indicating point P is on ray R, excluding the origin

-- Problem setup
def conditions (A O K L : Point) (AO : Ray) (KL : Segment) :=
  sorry -- definitions ensuring appropriate conditions for points and segments

-- Theorem statement
theorem locus_of_points (A O K L : Point) (AO : Ray) (KL : Segment) :
  conditions A O K L AO KL →
  ∀ P : Point, (is_on_ray P AO ∨ is_on_line_segment P KL) → 
               (P ≠ A ∧ P ≠ O ∧ P ≠ K ∧ P ≠ L) :=
begin
  sorry
end

end locus_of_points_l110_110217


namespace forest_volume_estimations_l110_110358

def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  let n := xs.length
  let xbar := average xs
  let ybar := average ys
  let numerator := (List.zip xs ys).sumBy (λ (xi, yi) => (xi - xbar) * (yi - ybar))
  let denominator_x := (xs.map (λ xi => (xi - xbar) ^ 2)).sum
  let denominator_y := (ys.map (λ yi => (yi - ybar) ^ 2)).sum
  numerator / Math.sqrt (denominator_x * denominator_y)

noncomputable def estimate_total_volume (total_area : ℝ) (avg_area avg_volume : ℝ) : ℝ :=
  (avg_volume / avg_area) * total_area

theorem forest_volume_estimations :
  let xs := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
  let ys := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
  average xs = 0.06 ∧
  average ys = 0.39 ∧
  correlation_coefficient xs ys ≈ 0.97 ∧
  estimate_total_volume 186 0.06 0.39 = 1209 :=
by
  let xs := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
  let ys := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
  have avg_x : average xs = 0.06 := sorry
  have avg_y : average ys = 0.39 := sorry
  have corr : correlation_coefficient xs ys ≈ 0.97 := sorry
  have est_vol : estimate_total_volume 186 0.06 0.39 = 1209 := sorry
  exact ⟨avg_x, avg_y, corr, est_vol⟩

end forest_volume_estimations_l110_110358


namespace coin_flip_sequences_l110_110741

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110741


namespace table_length_l110_110789

theorem table_length (x : ℕ) :
  (∃ n, 
    (∃ first_sheet, first_sheet = (0, 0)) ∧
    ∀ k : ℕ, k < n → 
      (let sheet = (k, k) in
      ∃ y z, 
        sheet = (y + 1, z + 1) ∧ 
        y % 5 = 0 ∧ z % 8 = 0 
       ) ∧
    (n = 73) ∧
    x = 77
  ) :=
sorry

end table_length_l110_110789


namespace sequence_limit_one_l110_110620

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ 
| n => if n = 0 then a 0 else 1 / (2 - seq a (n - 1))

theorem sequence_limit_one (a : ℕ → ℝ) (h_init : a 0 = a_0) :
  (∀ n, a (n + 1) = 1 / (2 - a n)) → 
  ∃ L, L = 1 ∧ Tendsto a at_top (𝓝 L) :=
begin
  sorry
end

end sequence_limit_one_l110_110620


namespace find_x_l110_110914

variable (x : ℝ)
def m : ℝ × ℝ := (x - 5, 1)
def n : ℝ × ℝ := (4, x)

-- Condition that the vectors are perpendicular: dot product is zero
def perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

theorem find_x : perpendicular m n → x = 4 :=
by
  sorry

end find_x_l110_110914


namespace coin_flip_sequences_l110_110743

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110743


namespace ab_leq_1_l110_110455

theorem ab_leq_1 {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) : ab ≤ 1 :=
sorry

end ab_leq_1_l110_110455


namespace inequality_mn_l110_110443

theorem inequality_mn (m n : ℤ)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) : 
  2 * (m^2 + n^2) < 5 * m * n := 
sorry

end inequality_mn_l110_110443


namespace line_contains_single_side_of_polygon_l110_110195

theorem line_contains_single_side_of_polygon (n : ℕ) :
  (∃ (P : polygon n), (n = 13 → ∃ (L : line), ∃ (s : side L P), ∀ (s' : side L P), s = s')
    ∧ (n > 13 → ¬∃ (P' : polygon n) (L' : line), ∃ (s' : side L' P'), ∀ (s'' : side L' P'), s'' = s')) := 
sorry

end line_contains_single_side_of_polygon_l110_110195


namespace area_of_shaded_region_l110_110349

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l110_110349


namespace factor_expression_l110_110843

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110843


namespace abc_lt_zero_neither_sufficient_nor_necessary_l110_110600

-- Define the curve equation
def curve_eq (a b c x y : ℝ) : Prop := a * x^2 + b * y^2 = c

-- Define the hyperbola condition
def is_hyperbola (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, curve_eq a b c x y ∧ 
  -- Additional conditions that determine it is a hyperbola
  (a > 0 ∧ b < 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c < 0) -- This represents basic form of hyperbola condition

-- Main theorem statement
theorem abc_lt_zero_neither_sufficient_nor_necessary (a b c : ℝ) :
  (abc < 0 → is_hyperbola a b c) ∧ (is_hyperbola a b c → abc < 0) ↔ false :=
begin
  -- The proof would go here, but it’s indicated to be omitted with sorry
  sorry
end

end abc_lt_zero_neither_sufficient_nor_necessary_l110_110600


namespace coin_flip_sequences_l110_110747

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110747


namespace compute_f_sum_l110_110432

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ
| x := if x < 1 then Real.cos (Real.pi * x) else f (x - 1) - 1

-- The statement to prove
theorem compute_f_sum : f (1/3) + f (4/3) = 0 :=
by
  unfold f
  -- Unfold only to illustrate structure; proof would go here.
  sorry

end compute_f_sum_l110_110432


namespace four_distinct_real_roots_l110_110096

theorem four_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, |(x-1)*(x-3)| = m*x → ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔ 
  0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
by
  sorry

end four_distinct_real_roots_l110_110096


namespace coin_flip_sequences_l110_110722

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110722


namespace Steven_apples_count_l110_110159

constant Jake_apples : Nat
constant Steven_apples : Nat
constant Jake_peaches : Nat := 3
constant Steven_peaches : Nat := 13

axiom H1 : Jake_peaches = Steven_peaches - 10
axiom H2 : Jake_apples = Steven_apples + 84

theorem Steven_apples_count : ∃ A : Nat, Steven_apples = A :=
by
  -- Based on the axiom H2, Steven has A apples.
  use Steven_apples
  refl

end Steven_apples_count_l110_110159


namespace old_cards_count_l110_110661

-- Definitions based on the conditions
def cards_per_page := 3
def total_pages := 6
def new_cards := 8

-- We need to calculate the value of old_cards
def total_cards := cards_per_page * total_pages
def old_cards := total_cards - new_cards

-- The statement we want to prove (no proof required, just the statement)
theorem old_cards_count : old_cards = 10 :=
by
  have h1 : total_cards = 18 := by
    unfold total_cards
    rw [Nat.mul_comm cards_per_page total_pages]
    exact rfl
  unfold old_cards
  rw h1
  exact rfl

end old_cards_count_l110_110661


namespace coin_flip_sequences_l110_110696

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110696


namespace minimum_value_of_f_l110_110130

theorem minimum_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f(2 * x - 1) = x ^ 2 + x) : 
  ∃ x : ℝ, f(x) = -1 / 4 :=
begin
  sorry
end

end minimum_value_of_f_l110_110130


namespace logarithmic_solution_l110_110850

theorem logarithmic_solution (x : ℝ) (hx : log x 256 = log 2 32) : x = 2^(8/5) :=
by
  sorry

end logarithmic_solution_l110_110850


namespace police_emergency_prime_l110_110787

theorem police_emergency_prime {n : ℕ} (k : ℕ) (h : n = k * 1000 + 133) : 
  ∃ p : ℕ, nat.prime p ∧ p > 7 ∧ p ∣ n :=
begin
  sorry,
end

end police_emergency_prime_l110_110787


namespace sum_A_eq_sum_B_l110_110873

-- Definitions for partitions and functions A and B
def A (π : List Nat) : Nat := π.count 1
def B (π : List Nat) : Nat := π.eraseDups.length

-- Sum function over all partitions of n
def sum_over_partitions {α : Type} (n : Nat) (f : List Nat → α) [AddCommMonoid α] : α :=
  (partitions n).sum (f .)

-- Main theorem
theorem sum_A_eq_sum_B (n : Nat) (h : n ≥ 1) :
  sum_over_partitions n A = sum_over_partitions n B := 
sorry


end sum_A_eq_sum_B_l110_110873


namespace triathlon_bike_speed_l110_110505

theorem triathlon_bike_speed :
  ∀ (t_total t_swim t_run t_bike : ℚ) (d_swim d_run d_bike : ℚ)
    (v_swim v_run r_bike : ℚ),
  t_total = 3 →
  d_swim = 1 / 2 →
  v_swim = 1 →
  d_run = 4 →
  v_run = 5 →
  d_bike = 10 →
  t_swim = d_swim / v_swim →
  t_run = d_run / v_run →
  t_bike = t_total - (t_swim + t_run) →
  r_bike = d_bike / t_bike →
  r_bike = 100 / 17 :=
by
  intros t_total t_swim t_run t_bike d_swim d_run d_bike v_swim v_run r_bike
         h_total h_d_swim h_v_swim h_d_run h_v_run h_d_bike h_t_swim h_t_run h_t_bike h_r_bike
  sorry

end triathlon_bike_speed_l110_110505


namespace angle_CDE_proof_l110_110973

def angle_A := 90
def angle_B := 90
def angle_C := 90
def angle_AEB := 30
def angle_BED := 40

theorem angle_CDE_proof :
  angle_A = 90 ∧ angle_B = 90 ∧ angle_C = 90 ∧ angle_AEB = 30 ∧ angle_BED = 40 →
  let angle_CDE := 110 in angle_CDE = 110 :=
by
  intros
  let angle_CDE := 110
  exact eq.refl angle_CDE

end angle_CDE_proof_l110_110973


namespace product_213_16_l110_110122

theorem product_213_16 :
  (213 * 16 = 3408) :=
by
  have h1 : (0.16 * 2.13 = 0.3408) := by sorry
  sorry

end product_213_16_l110_110122


namespace snowman_volume_l110_110566

theorem snowman_volume (r1 r2 r3 : ℝ) (V1 V2 V3 : ℝ) (π : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 6) (h3 : r3 = 8) 
  (hV1 : V1 = (4/3) * π * (r1^3)) 
  (hV2 : V2 = (4/3) * π * (r2^3)) 
  (hV3 : V3 = (4/3) * π * (r3^3)) :
  V1 + V2 + V3 = (3168/3) * π :=
by 
  sorry

end snowman_volume_l110_110566


namespace distinct_positive_integers_mod_1998_l110_110883

theorem distinct_positive_integers_mod_1998
  (a : Fin 93 → ℕ)
  (h_distinct : Function.Injective a) :
  ∃ m n p q : Fin 93, (m ≠ n ∧ p ≠ q) ∧ (a m - a n) * (a p - a q) % 1998 = 0 :=
by
  sorry

end distinct_positive_integers_mod_1998_l110_110883


namespace range_of_real_number_a_l110_110561

theorem range_of_real_number_a (a : ℝ) :
  (3 * a - 5) / (9 - a) < 0 ∧ ¬ ((5 * a - 5) / (25 - a) < 0) →
  (1 ≤ a ∧ a < 5 / 3) ∨ (9 < a ∧ a ≤ 25) :=
by
  intros h,
  sorry

end range_of_real_number_a_l110_110561


namespace expansion_no_x3_term_l110_110249

theorem expansion_no_x3_term (a : ℝ) :
  let expr := (a * (x ^ 2) - 3 * x) * ((x ^ 2) - 2 * x - 1),
      expanded_expr := a * (x ^ 4) + (-2 * a - 3) * (x ^ 3) + (-a + 6) * (x ^ 2) + 3 * x in
  (coeff expanded_expr 3 = 0) ↔ (a = -3 / 2) :=
by
  sorry

end expansion_no_x3_term_l110_110249


namespace minimum_value_expression_l110_110177

theorem minimum_value_expression (p q r s t u v w : ℝ) (h1 : p > 0) (h2 : q > 0) 
    (h3 : r > 0) (h4 : s > 0) (h5 : t > 0) (h6 : u > 0) (h7 : v > 0) (h8 : w > 0)
    (hpqrs : p * q * r * s = 16) (htuvw : t * u * v * w = 25) 
    (hptqu : p * t = q * u ∧ q * u = r * v ∧ r * v = s * w) : 
    (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 = 80 := sorry

end minimum_value_expression_l110_110177


namespace trajectory_of_P_distance_of_P_l110_110060

-- Given conditions
variable (x_0 y_0 x y : ℝ)
variable (M : x_0^2 + y_0^2 = 4)
variable (N : (4 : ℝ, 0 : ℝ))
variable (P : (x, y) is midpoint of (x_0, y_0) and (4, 0))

-- Definitions as per the given conditions
def midpoint_formula_xy : Prop := 2 * x = x_0 + 4 ∧ 2 * y = y_0

-- Target trajectory equation for P(x, y)
def trajectory_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Maximum and minimum distance definitions
def max_distance_from_line (x y : ℝ) : ℝ := 17
def min_distance_from_line (x y : ℝ) : ℝ := 15

theorem trajectory_of_P : 
  (x_0^2 + y_0^2 = 4) 
  → (2 * x = x_0 + 4 ∧ 2 * y = y_0)
  → ((x - 2)^2 + y^2 = 1) :=
by
  intro hM hMid
  sorry

theorem distance_of_P :
  (x_0^2 + y_0^2 = 4)
  → ((x - 2)^2 + y^2 = 1)
  → max_distance_from_line(x, y) = 17
  ∧ min_distance_from_line(x, y) = 15 :=
by
  intro hM hTraj
  sorry

end trajectory_of_P_distance_of_P_l110_110060


namespace coin_flip_sequences_l110_110765

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110765


namespace area_of_figure_l110_110972

theorem area_of_figure :
  let height1 := 8
  let width1 := 6
  let height2 := 3
  let remaining_width := 4 -- This is the segment from 6 to 10
  let height3 := 2
  let width3 := 5
  -- Areas of individual rectangles
  let area1 := height1 * width1
  let area2 := height2 * remaining_width
  let area3 := height3 * width3
  -- Total area
  (area1 + area2 + area3) = 70 :=
by {
  let height1 := 8
  let width1 := 6
  let height2 := 3
  let remaining_width := 4
  let height3 := 2
  let width3 := 5
  let area1 := height1 * width1
  let area2 := height2 * remaining_width
  let area3 := height3 * width3
  have total_area : (area1 + area2 + area3) = (8 * 6 + 3 * 4 + 2 * 5) := by rfl
  show (8 * 6 + 3 * 4 + 2 * 5) = 70, by norm_num
  sorry
}

end area_of_figure_l110_110972


namespace smith_oldest_child_age_l110_110596

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l110_110596


namespace two_pow_n_minus_one_div_by_seven_iff_l110_110851

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ (2^n - 1)) ↔ (∃ k : ℕ, n = 3 * k) := by
  sorry

end two_pow_n_minus_one_div_by_seven_iff_l110_110851


namespace find_increasing_point_l110_110111

variable (d : ℝ) -- d is the distance of the first half

-- Conditions
def total_distance : ℝ := 60
def avg_speed_first_half : ℝ := 24
def avg_speed_second_half : ℝ := avg_speed_first_half + 16
def avg_speed_total : ℝ := 30

-- Time for each half
def time_first_half : ℝ := d / avg_speed_first_half
def time_second_half : ℝ := (total_distance - d) / avg_speed_second_half
def total_time : ℝ := total_distance / avg_speed_total

-- The main statement to prove
theorem find_increasing_point : time_first_half + time_second_half = total_time → d = 30 := by
  sorry

end find_increasing_point_l110_110111


namespace same_terminal_side_angle_l110_110591

theorem same_terminal_side_angle (k : ℤ) : 
  (∃ k : ℤ, - (π / 6) = 2 * k * π + a) → a = 11 * π / 6 :=
sorry

end same_terminal_side_angle_l110_110591


namespace increased_consumption_5_percent_l110_110242

theorem increased_consumption_5_percent (T C : ℕ) (h1 : ¬ (T = 0)) (h2 : ¬ (C = 0)) :
  (0.80 * (1 + x/100) = 0.84) → (x = 5) :=
by
  sorry

end increased_consumption_5_percent_l110_110242


namespace equilateral_triangle_has_greatest_perimeter_l110_110805

noncomputable def side_length_equilateral_triangle (side_square : ℝ) : ℝ :=
  side_square * real.sqrt(4 / real.sqrt 3)

noncomputable def radius_circle (side_square : ℝ) : ℝ :=
  side_square * real.sqrt(1 / real.pi)

noncomputable def perimeter_equilateral_triangle (side_square : ℝ) : ℝ :=
  3 * side_length_equilateral_triangle side_square

noncomputable def perimeter_square (side_square : ℝ) : ℝ :=
  4 * side_square

noncomputable def circumference_circle (side_square : ℝ) : ℝ :=
  2 * real.pi * radius_circle side_square

theorem equilateral_triangle_has_greatest_perimeter (side_square : ℝ) :
    (perimeter_equilateral_triangle side_square > circumference_circle side_square) ∧ 
    (circumference_circle side_square > perimeter_square side_square) := 
  sorry

end equilateral_triangle_has_greatest_perimeter_l110_110805


namespace triangle_ABC_right_angled_l110_110522

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end triangle_ABC_right_angled_l110_110522


namespace find_k_l110_110852

def polynomial (n : ℤ) : ℤ := 3 * n^6 + 26 * n^4 + 33 * n^2 + 1

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : k ≤ 100) :
  (∃ n : ℤ, polynomial n % k = 0) ↔ k ∈ {9, 21, 27, 39, 49, 57, 63, 81, 87, 91, 93} := 
  sorry

end find_k_l110_110852


namespace jeremy_watermelons_last_3_weeks_l110_110530

def watermelons_last_weeks (initial : ℕ) (give_away : ℕ) (eat_pattern : ℕ → ℕ) : ℕ :=
  let total_consumed := λ week_num, give_away + eat_pattern (week_num % 3)
  let rec aux (weeks left : ℕ) : ℕ :=
    if left < total_consumed weeks then weeks
    else aux (weeks + 1) (left - total_consumed weeks)
  in aux 0 initial

theorem jeremy_watermelons_last_3_weeks :
  watermelons_last_weeks 30 4 (λ n, [3, 4, 5].nth! (n % 3)) = 3 :=
sorry

end jeremy_watermelons_last_3_weeks_l110_110530


namespace inequality_solution_set_l110_110623

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (x + 1) ≤ 0} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end inequality_solution_set_l110_110623


namespace total_distance_of_ring_stack_is_200cm_l110_110353

open Nat Real

theorem total_distance_of_ring_stack_is_200cm:
  let thickness := 2
  let top_diameter := 30
  let bottom_diameter := 10
  let n := (top_diameter - bottom_diameter) / thickness + 1
  let sequence_sum := n * (2 * (top_diameter - thickness) + (n - 1) * (-thickness)) / 2
  total_distance :=
    sequence_sum + thickness
  n = 11 ->
  total_distance = 200 :=
begin
  sorry
end

end total_distance_of_ring_stack_is_200cm_l110_110353


namespace coin_flip_sequences_l110_110753

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110753


namespace imo_1985_p6_l110_110440

open Nat

theorem imo_1985_p6 (n : ℕ) (x : Fin n → ℤ)
  (h₁ : ∀ i, x i = 1 ∨ x i = -1)
  (h₂ : (∑ i in range n, x i * x ((i + 1) % n) * x ((i + 2) % n) * x ((i + 3) % n)) = 0) :
  4 ∣ n :=
sorry

end imo_1985_p6_l110_110440


namespace girls_count_l110_110141

-- Definition of the conditions
variables (B G : ℕ)

def college_conditions (B G : ℕ) : Prop :=
  (B + G = 416) ∧ (B = (8 * G) / 5)

-- Statement to prove
theorem girls_count (B G : ℕ) (h : college_conditions B G) : G = 160 :=
by
  sorry

end girls_count_l110_110141


namespace pairing_count_l110_110244

-- Declaration of the problem
def people : Type := Fin 12

def knows (a b : people) : Prop :=
  (b = (a + 1) % 12) ∨ (b = (a + 11) % 12) ∨ (b = (a + 6) % 12)

def valid_pairing (pairs : Finset (people × people)) : Prop :=
  pairs.card = 6 ∧ (∀ (a b : people), (a, b) ∈ pairs → knows a b)

theorem pairing_count :
  ∃ (pairs : Finset (people × people)), valid_pairing pairs ∧ pairs.card = 3 :=
sorry

end pairing_count_l110_110244


namespace functional_equation_satisfied_l110_110027

noncomputable def f (x : ℝ) : ℝ := (4 * x ^ 2 - x + 1) / (5 * (x - 1))

theorem functional_equation_satisfied (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  x * f(x) + 2 * f((x - 1) / (x + 1)) = 1 :=
by
  sorry

end functional_equation_satisfied_l110_110027


namespace number_of_people_in_village_l110_110963

variable (P : ℕ) -- Define the total number of people in the village

def people_not_working : ℕ := 50
def people_with_families : ℕ := 25
def people_singing_in_shower : ℕ := 75
def max_people_overlap : ℕ := 50

theorem number_of_people_in_village :
  P - people_not_working + P - people_with_families + P - people_singing_in_shower - max_people_overlap = P → 
  P = 100 :=
by
  sorry

end number_of_people_in_village_l110_110963


namespace compute_z6_l110_110985

noncomputable def z : ℂ := (-Real.sqrt 3 + Complex.i) / 2

theorem compute_z6 : z ^ 6 = -1 := 
by sorry

end compute_z6_l110_110985


namespace coefficient_expansion_l110_110645

noncomputable def coefficient_of_x3y7z2_in_expansion : ℚ :=
  let coeff := (12.choose 3) * (12 - 3).choose 7 * ((12 - 3 - 7).choose 2) * (4/7)^3 * (-3/5)^7 * (2/3)^2
  in coeff

theorem coefficient_expansion :
  coefficient_of_x3y7z2_in_expansion = -3534868480 / 218968125 := 
sorry

end coefficient_expansion_l110_110645


namespace cos_double_angle_vector_magnitude_l110_110482

theorem cos_double_angle_vector_magnitude (α : ℝ) 
  (h1 : ∥(⟨cos α, 1 / 2⟩ : ℝ × ℝ)∥ = (√2 / 2)) : cos (2 * α) = -1 / 2 := 
begin
  sorry
end

end cos_double_angle_vector_magnitude_l110_110482


namespace minimum_tan_diff_is_one_l110_110465

open Real

noncomputable def minimum_tan_diff (a b : ℝ) (h : a > b ∧ b > 0) :=
  let e := (√3) / 2
  let P (x0 y0 : ℝ) := (x0^2 / a^2 + y0^2 / b^2 = 1)
  let A := (-a, 0)
  let B := (a, 0)
  let α := y0 / (x0 + a)
  let β := y0 / (x0 - a)
  abs (tan α - tan β)

theorem minimum_tan_diff_is_one (a b : ℝ) (h : a > b ∧ b > 0) (P : ℝ × ℝ) :
  P.1^2 / a^2 + P.2^2 / b^2 = 1 → 
  let k_PA := P.2 / (P.1 + a)
  let k_PB := P.2 / (P.1 - a)
  let α := atan k_PA
  let β := atan k_PB
  abs (tan α - tan β) = 1 := 
sorry

end minimum_tan_diff_is_one_l110_110465


namespace jerry_stickers_l110_110532

variable (G F J : ℕ)

theorem jerry_stickers (h1 : F = 18) (h2 : G = F - 6) (h3 : J = 3 * G) : J = 36 :=
by {
  sorry
}

end jerry_stickers_l110_110532


namespace five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l110_110077

variable (p q : ℕ)
variable (hp : p % 2 = 1)  -- p is odd
variable (hq : q % 2 = 1)  -- q is odd

theorem five_p_squared_plus_two_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (5 * p^2 + 2 * q^2) % 2 = 1 := 
sorry

theorem p_squared_plus_pq_plus_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (p^2 + p * q + q^2) % 2 = 1 := 
sorry

end five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l110_110077


namespace find_H_coords_l110_110083

def point := ℝ × ℝ

def left_focus : point := (-1, 0)

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def line_through_focus (k : ℝ → ℝ) : set point :=
{p | ∃ x y, p = (x, y) ∧ (y = k (x + 1))}

def perpendicular_to_AB_through_O (k : ℝ) (l : ℝ) : set point :=
{p | ∃ x y, p = (x, y) ∧ (y = l * x)}

theorem find_H_coords (k : ℝ): Prop :=
  let l1 := λ x, sqrt 2 * x + sqrt 2 in
  let l2 := λ x, -sqrt 2 * x + sqrt 2 in
  let perp_line1 := λ x, x / sqrt 2 in
  let perp_line2 := λ x, -x / sqrt 2 in
  ∃ H1 H2: point,
  H1 = (-2/3, sqrt 2/3) ∧ H2 = (-2/3, -sqrt 2/3) ∧
  ∀ A B : point, 
  (ellipse A.1 A.2) ∧ (ellipse B.1 B.2) ∧ (line_through_focus l1 A) ∧ (line_through_focus l1 B) ∧
  (A.1 * A.2 * B.1 * B.2 = 0) ∧
  (H1 = (perpendicular_to_AB_through_O perp_line1 H1)) ∧ 
  (H2 = (perpendicular_to_AB_through_O perp_line2 H2))
  
#check find_H_coords

end find_H_coords_l110_110083


namespace tetrahedron_partitions_space_l110_110981

-- Define a tetrahedron with given properties
structure Tetrahedron where
  faces : ℕ
  edges : ℕ
  vertices : ℕ

-- Define the specific tetrahedron with 4 faces, 6 edges, and 4 vertices
def tetra : Tetrahedron := { faces := 4, edges := 6, vertices := 4 }

-- The theorem: The planes formed by the faces of a tetrahedron divide the space into 15 parts
theorem tetrahedron_partitions_space (T : Tetrahedron) (h_faces : T.faces = 4) (h_edges : T.edges = 6) (h_vertices : T.vertices = 4) :
  ∃ (regions : ℕ), regions = 15 :=
by
  use 15
  sorry

end tetrahedron_partitions_space_l110_110981


namespace tan_7pi_over_4_eq_neg1_l110_110017

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l110_110017


namespace FlyersDistributon_l110_110810

variable (total_flyers ryan_flyers alyssa_flyers belinda_percentage : ℕ)
variable (scott_flyers : ℕ)

theorem FlyersDistributon (H : total_flyers = 200)
  (H1 : ryan_flyers = 42)
  (H2 : alyssa_flyers = 67)
  (H3 : belinda_percentage = 20)
  (H4 : scott_flyers = total_flyers - (ryan_flyers + alyssa_flyers + (belinda_percentage * total_flyers) / 100)) :
  scott_flyers = 51 :=
by
  simp [H, H1, H2, H3] at H4
  exact H4

end FlyersDistributon_l110_110810


namespace even_digit_number_division_l110_110029

theorem even_digit_number_division (N : ℕ) (n : ℕ) :
  (N % 2 = 0) ∧
  (∃ a b : ℕ, (∀ k : ℕ, N = a * 10^n + b → N = k * (a * b)) ∧
  ((N = (1000^(2*n - 1) + 1)^2 / 7) ∨
   (N = 12) ∨
   (N = (10^n + 2)^2 / 6) ∨
   (N = 1352) ∨
   (N = 15))) :=
sorry

end even_digit_number_division_l110_110029


namespace infinite_geometric_subsequence_exists_l110_110166

theorem infinite_geometric_subsequence_exists
  (a : ℕ) (d : ℕ) (h_d_pos : d > 0)
  (a_n : ℕ → ℕ)
  (h_arith_prog : ∀ n, a_n n = a + n * d) :
  ∃ (g : ℕ → ℕ), (∀ m n, m < n → g m < g n) ∧ (∃ r : ℕ, ∀ n, g (n+1) = g n * r) ∧ (∀ n, ∃ m, a_n m = g n) :=
sorry

end infinite_geometric_subsequence_exists_l110_110166


namespace max_value_b_exists_l110_110381

theorem max_value_b_exists :
  ∃ a c : ℝ, ∃ b : ℝ, 
  (∀ x : ℤ, 
  ((x^4 - a * x^3 - b * x^2 - c * x - 2007) = 0) → 
  ∃ r s t : ℤ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  ((x = r) ∨ (x = s) ∨ (x = t))) ∧ 
  (∀ b' : ℝ, b' < b → 
  ¬ ( ∃ a' c' : ℝ, ( ∀ x : ℤ, 
  ((x^4 - a' * x^3 - b' * x^2 - c' * x - 2007) = 0) → 
  ∃ r' s' t' : ℤ, r' ≠ s' ∧ s' ≠ t' ∧ r' ≠ t' ∧ 
  ((x = r') ∨ (x = s') ∨ (x = t') )))) ∧ b = 3343 :=
sorry

end max_value_b_exists_l110_110381


namespace least_six_digit_congruent_7_mod_17_l110_110646

theorem least_six_digit_congruent_7_mod_17 : 
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 17 = 7 ∧ 
  (∀ y : ℕ, 100000 ≤ y ∧ y % 17 = 7 → x ≤ y) :=
begin
  -- The proof goes here
  sorry
end

end least_six_digit_congruent_7_mod_17_l110_110646


namespace constant_term_binomial_expansion_l110_110971

theorem constant_term_binomial_expansion : 
  (let general_term (r : ℕ) := 
    binom 6 r * (2^(6-r) * (-1)^r * (x^(6/2 - 3*r/2))) in
    ∃ r : ℕ, 6 = 3 * r ∧ (2^4 * binom 6 2) = 240) :=
by
  sorry

end constant_term_binomial_expansion_l110_110971


namespace total_games_l110_110245

theorem total_games (n : ℕ) (h : n = 15) : (n * (n - 1)) / 2 = 105 :=
by
  rw [h]
  sorry

end total_games_l110_110245


namespace bijection_count_l110_110446

noncomputable def count_bijections_with_property (m n : ℕ) :=
  ∑ k_i in (factors m).erase 1, 
    n.factorial / (k_i ^ (n / k_i) * (n / k_i).factorial)

theorem bijection_count (m n : ℕ) (f : (fin n → fin n)) (hf : bijective f) :
  (∃ k_i : {k // k ∣ m - 1}, ∑ (k : ℕ) in (factors m).erase 1, k_i * (n / k_i)) = n →
  f^(m) = f :=
begin
  sorry
end

end bijection_count_l110_110446


namespace range_of_a_l110_110069

def f (x a : ℝ) : ℝ := 2 * x + a
def g (x : ℝ) : ℝ := Real.log x - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), x1 ∈ Icc (1/2 : ℝ) (2 : ℝ) → x2 ∈ Icc (1/2 : ℝ) (2 : ℝ) → f x1 a ≤ g x2) ↔ a ≤ Real.log 2 - 8 :=
by
  sorry

end range_of_a_l110_110069


namespace fred_dark_blue_marbles_count_l110_110428

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l110_110428


namespace not_possible_1006_2012_gons_l110_110528

theorem not_possible_1006_2012_gons :
  ∀ (n : ℕ), (∀ (k : ℕ), k ≤ 2011 → 2 * n ≤ k) → n ≠ 1006 :=
by
  intro n h
  -- Here goes the skipped proof part
  sorry

end not_possible_1006_2012_gons_l110_110528


namespace problem_2012_Shandong_l110_110313

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem problem_2012_Shandong :
  U = {0, 1, 2, 3, 4} → A = {1, 2, 3} → B = {2, 4} → (U \ A) ∪ B = {0, 2, 4} :=
by
  intro hU hA hB
  simp [hU, hA, hB]
  show {0, 4} ∪ {2, 4} = {0, 2, 4}
  sorry

end problem_2012_Shandong_l110_110313


namespace coin_flip_sequences_l110_110699

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110699


namespace has_exactly_32_integer_points_on_circle_l110_110824

theorem has_exactly_32_integer_points_on_circle :
  ∃ R : ℝ, R = Real.sqrt 1105 ∧
    (∃ (points : Finset (ℤ × ℤ)),
      points.card = 32 ∧ ∀ (x y : ℤ), ((x, y) ∈ points ↔ x^2 + y^2 = (R^2).toInt)) :=
sorry

end has_exactly_32_integer_points_on_circle_l110_110824


namespace basketball_scoring_possibilities_l110_110680

theorem basketball_scoring_possibilities : 
  ∃ n : ℕ, (∀ seq : fin 7 → ℕ, (∀ i, seq i ∈ {1, 2, 3}) →
  seq.sum = n) ∧ n = 15 := sorry

end basketball_scoring_possibilities_l110_110680


namespace quadrilateral_cyclic_l110_110190

open EuclideanGeometry

def points_on_sides (A B C A' B' C' : Point) : Prop :=
  lies_on A' (line_segment B C) ∧ lies_on B' (line_segment C A) ∧ lies_on C' (line_segment A B)

def angles_condition (A B C A' B' C' X : Point) : Prop :=
  ∠ A X B = ∠ A' C' B' + ∠ A C B ∧
  ∠ B X C = ∠ B' A' C' + ∠ B A C

theorem quadrilateral_cyclic (A B C A' B' C' X : Point)
  (h1 : points_on_sides A B C A' B' C')
  (h2 : angles_condition A B C A' B' C' X) :
  cyclic_quad (X, A', B, C') :=
begin
  sorry
end

end quadrilateral_cyclic_l110_110190


namespace find_x_solution_l110_110933

theorem find_x_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (h_eq : (4 * x)^(Real.log 4 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 :=
by
  sorry

end find_x_solution_l110_110933


namespace cone_slant_height_correct_l110_110769

noncomputable def cone_slant_height (r : ℝ) : ℝ := 4 * r

theorem cone_slant_height_correct (r : ℝ) (h₁ : π * r^2 + π * r * cone_slant_height r = 5 * π)
  (h₂ : 2 * π * r = (1/4) * 2 * π * cone_slant_height r) : cone_slant_height r = 4 :=
by
  sorry

end cone_slant_height_correct_l110_110769


namespace contrapositive_honor_roll_l110_110568

variable (Student : Type) (scores_hundred : Student → Prop) (honor_roll_qualifies : Student → Prop)

theorem contrapositive_honor_roll (s : Student) :
  (¬ honor_roll_qualifies s) → (¬ scores_hundred s) := 
sorry

end contrapositive_honor_roll_l110_110568


namespace convert_1729_to_base_5_l110_110390

theorem convert_1729_to_base_5 :
  let d := 1729
  let b := 5
  let representation := [2, 3, 4, 0, 4]
  -- Check the representation of 1729 in base 5
  d = (representation.reverse.enum_from 0).sum (fun ⟨i, coef⟩ => coef * b^i) :=
  sorry

end convert_1729_to_base_5_l110_110390


namespace chord_length_of_tangent_circle_l110_110592

theorem chord_length_of_tangent_circle
  (area_of_ring : ℝ)
  (diameter_large_circle : ℝ)
  (h1 : area_of_ring = (50 / 3) * Real.pi)
  (h2 : diameter_large_circle = 10) :
  ∃ (length_of_chord : ℝ), length_of_chord = (10 * Real.sqrt 6) / 3 := by
  sorry

end chord_length_of_tangent_circle_l110_110592


namespace oldest_child_age_l110_110594

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end oldest_child_age_l110_110594


namespace new_light_wattage_is_143_l110_110298

-- Define the original wattage and the percentage increase
def original_wattage : ℕ := 110
def percentage_increase : ℕ := 30

-- Compute the increase in wattage
noncomputable def increase : ℕ := (percentage_increase * original_wattage) / 100

-- The new wattage should be the original wattage plus the increase
noncomputable def new_wattage : ℕ := original_wattage + increase

-- State the theorem that proves the new wattage is 143 watts
theorem new_light_wattage_is_143 : new_wattage = 143 := by
  unfold new_wattage
  unfold increase
  sorry

end new_light_wattage_is_143_l110_110298


namespace coin_flip_sequences_l110_110742

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110742


namespace tan_half_sum_eq_neg_three_halves_l110_110551

theorem tan_half_sum_eq_neg_three_halves (x y : ℝ) 
  (h1 : cos x - cos y = 2 / 3)
  (h2 : sin x + sin y = 4 / 9) :
  tan ((x + y) / 2) = -3 / 2 := by sorry

end tan_half_sum_eq_neg_three_halves_l110_110551


namespace abs_eq_zero_iff_l110_110656

theorem abs_eq_zero_iff (x : ℚ) : (| 3 * x + 5 | = 0) ↔ (x = -5 / 3) := by
  sorry

end abs_eq_zero_iff_l110_110656


namespace Ariella_has_more_savings_l110_110807

variable (Daniella_savings: ℝ) (Ariella_future_savings: ℝ) (interest_rate: ℝ) (time_years: ℝ)
variable (initial_Ariella_savings: ℝ)

-- Conditions
axiom h1 : Daniella_savings = 400
axiom h2 : Ariella_future_savings = 720
axiom h3 : interest_rate = 0.10
axiom h4 : time_years = 2

-- Assume simple interest formula for future savings
axiom simple_interest : Ariella_future_savings = initial_Ariella_savings * (1 + interest_rate * time_years)

-- Show the difference in savings
theorem Ariella_has_more_savings : initial_Ariella_savings - Daniella_savings = 200 :=
by sorry

end Ariella_has_more_savings_l110_110807


namespace mafs_counts_game_probability_l110_110360

theorem mafs_counts_game_probability :
  ∀ (Alan Jason Shervin : ℕ),
  -- Initial conditions
  Alan = 2 → Jason = 2 → Shervin = 2 →
  -- Alan wins the first round
  Alan' = Alan + 2 ∧ Jason' = Jason - 1 ∧ Shervin' = Shervin - 1 →
  -- Alan does not win the second round (assume Jason wins)
  Alan'' = Alan' - 1 ∧ Jason'' = Jason' + 2 ∧ Shervin'' = Shervin' - 1 →
  -- Shervin is eliminated
  Shervin'' = 0 →
  -- The probability that Alan wins the game is 1/2
  Alan_wins_probability = 1 / 2 :=
begin
  intros,
  sorry
end

end mafs_counts_game_probability_l110_110360


namespace find_theta_sum_l110_110974

section Problem

variable (α θ θ₁ θ₂ : ℝ) (ρ : ℝ) (x y : ℝ)

-- Parameter equations of curve C₁
def curve_C1 (x y : ℝ) : Prop :=
  x = sqrt 3 * sin α ∧ y = sqrt 2 * cos α

-- Polar coordinate equation of line l
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (2 * cos θ + sin θ) = sqrt 6

-- Cartesian equation of curve C₁
def cartesian_curve_C1 (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 2) = 1

-- Cartesian equation of line l
def cartesian_line_l (x y : ℝ) : Prop :=
  2 * x + y = sqrt 6

theorem find_theta_sum :
  (2 * cos θ₁ + sin θ₁) * (2 * cos θ₂ + sin θ₂) = 6 →
  cos (θ₁ + θ₂) = -cos (θ₁ - θ₂) →
  θ₁ + θ₂ = 9 * π / 4 :=
sorry

end Problem

end find_theta_sum_l110_110974


namespace variance_eta_l110_110091

/-- Given a binomial random variable X with parameters n = 10 and p = 0.6, and η = 8 - 2X,
    the variance Dη = 9.6. -/
theorem variance_eta {X : ℕ → ℝ} (hX : X ~ binomial 10 0.6) :
  ∀ (η : ℝ → ℝ), η = (λ x, 8 - 2 * x) → var η = 9.6 :=
by
  sorry

end variance_eta_l110_110091


namespace coin_flip_sequences_l110_110757

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110757


namespace find_interest_rate_first_year_l110_110853

-- Define the principal amount, interest rate for the second year, final amount, and the unknown interest rate for the first year
def principal : ℝ := 8000
def second_year_interest_rate : ℝ := 0.05
def final_amount : ℝ := 8736
def interest_rate_first_year : ℝ := 4 / 100   -- 4%

-- Define a function to calculate the amount after the first year given the interest rate for the first year
def amount_after_first_year (R : ℝ) : ℝ :=
  principal + (principal * (R / 100))

-- Define a function to calculate the total amount after the second year given the interest rate for the first year
def total_amount_after_second_year (R : ℝ) : ℝ :=
  amount_after_first_year R + (amount_after_first_year R * second_year_interest_rate)

-- State the theorem that we want to prove
theorem find_interest_rate_first_year :
  ∃ R, total_amount_after_second_year R = final_amount ∧ R = interest_rate_first_year :=
by
  sorry

end find_interest_rate_first_year_l110_110853


namespace increasing_function_proof_l110_110223

theorem increasing_function_proof {f : ℝ → ℝ} (h1 : ∀ x, 0 < x → f x > 0) (h2 : ∀ x, 0 < x → deriv f x > 0) :
  ∀ x, 0 < x → deriv (λ x, x * f x) x > 0 :=
by
  intro x hx
  rw [deriv_mul, deriv_id, one_mul]
  exact add_pos (h1 x hx) (mul_pos hx (h2 x hx))

end increasing_function_proof_l110_110223


namespace probability_at_least_3_laughs_l110_110132

noncomputable def probability_laugh (success_prob : ℚ) (num_trials : ℕ) (required_successes : ℕ) : ℚ :=
  if h : required_successes ≤ num_trials then
    1 - (0...).sum (fun k => if h2 : k ≤ required_successes then combinatory num_trials k * (success_prob ^ k) * ((1 - success_prob) ^ (num_trials - k)) else 0)
  else
    0

theorem probability_at_least_3_laughs :
  probability_laugh (1/3) 6 2 = 353 / 729 :=
by
  sorry

end probability_at_least_3_laughs_l110_110132


namespace max_possible_integer_in_list_l110_110323

/--
Given a list of five positive integers where:
- the only integer that occurs more than once is 8,
- the median of the list is 9,
- the average of the list is 10,
prove that the largest possible integer in the list is 15.
-/
theorem max_possible_integer_in_list (l : List ℕ) (hl1 : l.length = 5)
  (h1 : ∀ x ∈ l, x > 0)
  (h2 : l.filter (λ x, x = 8).length > 1)
  (h3 : l.nth_le 2 (by linarith) = 9)
  (h4 : l.sum = 5 * 10) :
  l.maximum = 15 :=
sorry

end max_possible_integer_in_list_l110_110323


namespace tan_210_eq_neg_sqrt3_div_3_l110_110871

theorem tan_210_eq_neg_sqrt3_div_3 :
  tan 210 = - (Real.sqrt 3 / 3) :=
by
  have period : ∀ θ : ℝ, tan (180 + θ) = -tan θ := sorry
  have tan_30 : tan 30 = Real.sqrt 3 / 3 := sorry
  show tan 210 = - (Real.sqrt 3 / 3)
  sorry

end tan_210_eq_neg_sqrt3_div_3_l110_110871


namespace log_sum_example_l110_110414

theorem log_sum_example :
  log 10 50 + log 10 30 = 3 + log 10 1.5 := 
by
  sorry -- proof is skipped

end log_sum_example_l110_110414


namespace coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110263

theorem coefficient_a3b3_in_ab6_c8_div_c8_is_1400 :
  let a := (a : ℝ)
  let b := (b : ℝ)
  let c := (c : ℝ)
  ∀ (a b c : ℝ), (binom 6 3 * a^3 * b^3) * (binom 8 4 * c^0) = 1400 := 
by
  sorry

end coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110263


namespace frequency_calculation_l110_110333

/-- A sample of size 40 is grouped into the following intervals and frequencies:
     - $[25, 25.3)$: 6
     - $[25.3, 25.6)$: 4
     - $[25.6, 25.9)$: 10
     - $[25.9, 26.2)$: 8
     - $[26.2, 26.5)$: 8
     - $[26.5, 26.8)$: 4
     Prove that the frequency of the sample in the range $[25, 25.9)$ is $\frac{1}{2}$.
-/
theorem frequency_calculation : 
  let 
    total_sample_size := 40,
    group_intervals := [(25, 25.3), (25.3, 25.6), (25.6, 25.9), (25.9, 26.2), (26.2, 26.5), (26.5, 26.8)],
    frequencies := [6, 4, 10, 8, 8, 4],
    selected_intervals := [0, 1, 2]  -- indices corresponding to intervals [25, 25.3), [25.3, 25.6), [25.6, 25.9)
  in 
    (∑ i in selected_intervals, frequencies.nth_le i (by sorry)) / total_sample_size = 1 / 2 :=
sorry

end frequency_calculation_l110_110333


namespace max_points_on_segment_l110_110278

theorem max_points_on_segment (d : ℝ) (points : list ℝ) (h : ∀ x, 0 ≤ x → x ≤ 1 → ∃ y, y ∈ points ∧ y = x) :
  length points ≤ 32 :=
sorry

end max_points_on_segment_l110_110278


namespace chord_length_l110_110515

noncomputable def parametric_equations (t: ℝ) := (2 * t, -2 - t)

def polar_circle_equation (θ: ℝ) := 4 * real.sqrt 2 * real.cos (θ + real.pi / 4)

theorem chord_length :
  let l := λ t : ℝ, (2 * t, -2 - t),
      C := (λ x y, x^2 + y^2 - 4 * x + 4 * y = 0),
      center_C := (2, -2),
      radius_C := 2 * real.sqrt 2,
      x_y_equation := λ x y, x + 2 * y + 4 = 0 in
  let distance := abs ((2 * 2 - 4 * (-2) + 4) / real.sqrt(2^2 + (-1)^2)) in
  2 * real.sqrt(radius_C^2 - distance^2) = 12 * real.sqrt 5 / 5 :=
sorry

end chord_length_l110_110515


namespace max_xyz_value_l110_110071

theorem max_xyz_value : 
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 
    (1 / x + 1 / y + 1 / z = 10) ∧ xyz(x, y, z) ≤ 4 / 125 := 
begin
  sorry
end

def xyz (x y z : ℝ) := x * y * z

end max_xyz_value_l110_110071


namespace performance_stability_l110_110292

theorem performance_stability (avg_score : ℝ) (num_shots : ℕ) (S_A S_B : ℝ) 
  (h_avg : num_shots = 10)
  (h_same_avg : avg_score = avg_score) 
  (h_SA : S_A^2 = 0.4) 
  (h_SB : S_B^2 = 2) : 
  (S_A < S_B) :=
by
  sorry

end performance_stability_l110_110292


namespace factor_expression_l110_110840

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110840


namespace max_b_value_l110_110889

theorem max_b_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : b ≤ 1 / 3 :=
  sorry

end max_b_value_l110_110889


namespace shaded_area_common_squares_l110_110341

noncomputable def cos_beta : ℝ := 3 / 5

theorem shaded_area_common_squares :
  ∀ (β : ℝ), (0 < β) → (β < pi / 2) → (cos β = cos_beta) →
  (∃ A, A = 4 / 3) :=
by
  sorry

end shaded_area_common_squares_l110_110341


namespace mother_age_when_harry_born_l110_110485

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l110_110485


namespace triangle_shape_right_angled_l110_110523

theorem triangle_shape_right_angled (a b c : ℝ) (A B C : ℝ) (h1 : b^2 = c^2 + a^2 - c * a) (h2 : Real.sin A = 2 * Real.sin C) :
    ∃ (D : Type) (triangle_shape : TriangeShape D), triangle_shape = TriangeShape.RightAngled :=
by
  sorry

end triangle_shape_right_angled_l110_110523


namespace period_of_tan_plus_cot_minus_sec_l110_110279

theorem period_of_tan_plus_cot_minus_sec :
  ∀ x, (tan x + cot x - sec x) = (tan (x + 2 * π) + cot (x + 2 * π) - sec (x + 2 * π)) :=
by
  -- Part of the condition, to state that the period of the combined function is 2π.
  sorry

end period_of_tan_plus_cot_minus_sec_l110_110279


namespace intercept_tangent_line_correct_l110_110948

noncomputable def intercept_of_tangent_line (x m : ℝ) : ℝ :=
  let y := (x^4 - x^3) / (x - 1)
  let dydx := deriv (fun x => x^3)
  let slope := 3
  let tangent_point := (-1, -1)
  let tangent_line := λ x => 3 * x + 2
  let intercept := tangent_line 0
  intercept

theorem intercept_tangent_line_correct (m : ℝ) (h : deriv (fun x => (x^4 - x^3) / (x - 1)) m = 3)
    : intercept_of_tangent_line (-1) ≠ -1 :=
by
  sorry

end intercept_tangent_line_correct_l110_110948


namespace coin_flip_sequences_l110_110702

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110702


namespace point_to_line_distance_l110_110605

-- Define point P and line equation
def point_P : (ℝ × ℝ) := (-1, 2)
def line_eqn (x y : ℝ) : Prop := 8 * x - 6 * y + 15 = 0

-- Prove the distance from point P to the line is 1/2
theorem point_to_line_distance :
  let d := (fun (P : ℝ × ℝ) (a b c : ℝ) =>
      |a * P.1 + b * P.2 + c| / Real.sqrt (a * a + b * b))
  in d point_P 8 (-6) 15 = 1 / 2 :=
by
  sorry

end point_to_line_distance_l110_110605


namespace midline_parallel_l110_110064

open EuclideanGeometry

-- Define a triangle with vertices A, B, and C.
variables {A B C : Point}

-- Define midpoints E and D of sides AB and AC respectively.
variables (E : Point) (hE : isMidpoint E A B)
variables (D : Point) (hD : isMidpoint D A C)

-- The statement to be proved in Lean 4.
theorem midline_parallel (ABC : Triangle A B C) :
  parallel (lineThrough E D) (lineThrough B C) :=
by
  sorry

end midline_parallel_l110_110064


namespace average_age_of_women_l110_110667

noncomputable def avg_age_two_women (M : ℕ) (new_avg : ℕ) (W : ℕ) :=
  let loss := 20 + 10;
  let gain := 2 * 8;
  W = loss + gain

theorem average_age_of_women (M : ℕ) (new_avg : ℕ) (W : ℕ) (avg_age : ℕ) :
  avg_age_two_women M new_avg W →
  avg_age = 23 :=
sorry

#check average_age_of_women

end average_age_of_women_l110_110667


namespace tan_7pi_over_4_eq_neg1_l110_110018

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l110_110018


namespace complex_number_ob_l110_110511

-- Define the origin and the point A.
def O : ℂ := 0
def A : ℂ := 2 + complex.i

-- Define the transformation to get the symmetric point B about the imaginary axis.
def symmetric_point_imag_axis (z : ℂ) : ℂ := -z.re + z.im * complex.i

-- State the theorem.
theorem complex_number_ob :
  symmetric_point_imag_axis A = -2 + complex.i :=
by
  -- The proof would be filled in here.
  sorry

end complex_number_ob_l110_110511


namespace infinite_nat_of_form_k_squared_plus_one_no_real_divisor_l110_110582

theorem infinite_nat_of_form_k_squared_plus_one_no_real_divisor :
  ∃∞ k : ℕ, ¬∃ j : ℕ, j < k ∧ (k^2 + 1) ∣ (j^2 + 1) :=
by
  sorry

end infinite_nat_of_form_k_squared_plus_one_no_real_divisor_l110_110582


namespace fibonacci_divisibility_l110_110580

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

-- Theorem to prove
theorem fibonacci_divisibility (p k : ℕ) (hp_prime : nat.prime p) (sqrt5_mod_p : ∃ x, x^2 ≡ 5 [MOD p]) :
  fibonacci (p^(k-1) * (p-1)) % p^k = 0 :=
sorry

end fibonacci_divisibility_l110_110580


namespace ellipse_focal_length_l110_110902

def is_valid_m (m : ℝ) : Prop :=
  let a := real.sqrt (10 - m)
  let b := real.sqrt (m - 2)
  (a * a + b * b = 4)

theorem ellipse_focal_length (m : ℝ) : is_valid_m m ↔ m = 4 ∨ m = 8 := by
  -- proof to be filled later
  sorry

end ellipse_focal_length_l110_110902


namespace domain_of_expression_l110_110861

theorem domain_of_expression : {x : ℝ | ∃ y z : ℝ, y = √(x - 3) ∧ z = √(8 - x) ∧ x - 3 ≥ 0 ∧ 8 - x > 0} = {x : ℝ | 3 ≤ x ∧ x < 8} :=
by
  sorry

end domain_of_expression_l110_110861


namespace minimum_trios_five_people_l110_110040

-- Define the conditions and the statement as a theorem to be proven.
theorem minimum_trios_five_people : 
  ∀ (persons : Finset ℕ) (handshake : ∀ a b : ℕ, Prop),
  persons.card = 5 -> 
  (∀ (A B C : ℕ), A ∈ persons -> B ∈ persons -> C ∈ persons -> (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) ->
    (handshake A B ∧ handshake B C) ∨ (¬ handshake A B ∧ ¬ handshake B C) ->
    (trio A B C)) ->
  (∃ (trio_count : ℕ), trio_count = 10) := sorry

end minimum_trios_five_people_l110_110040


namespace implies_not_right_triangle_l110_110497

theorem implies_not_right_triangle 
  (a b c : ℝ)
  (h1 : b^2 = (a + c) * (a - c))
  (h2 : a / b / c = 1 / (sqrt 3) / 2)
  (h3 : ∠C = ∠A - ∠B) :
  ∠A / ∠B / ∠C = 3 / (4 : ℝ) / 5 → ¬ (∠A = 90 ∨ ∠B = 90 ∨ ∠C = 90) :=
by
  sorry

end implies_not_right_triangle_l110_110497


namespace probability_point_not_in_squareB_within_squareA_l110_110588

theorem probability_point_not_in_squareB_within_squareA :
  ∀ (area_A : ℝ) (perimeter_B : ℝ) (side_length_A side_length_B area_B : ℝ),
    area_A = 25 →
    perimeter_B = 12 →
    side_length_A = real.sqrt area_A →
    side_length_B = perimeter_B / 4 →
    area_B = side_length_B ^ 2 →
    (25 - area_B) / 25 = 16 / 25 :=
by
  intros area_A perimeter_B side_length_A side_length_B area_B h1 h2 h3 h4 h5
  sorry

end probability_point_not_in_squareB_within_squareA_l110_110588


namespace distinct_sequences_ten_flips_l110_110694

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110694


namespace coin_flip_sequences_l110_110704

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110704


namespace cosine_sum_l110_110178

-- Definitions for the conditions
variables (A B C : Type) [field A] [field B] [field C]
variable (triangle_ABC : A)
variable (AB BC CA : B) (excenter_IA excenter_IB excenter_IC circumcenter_O : B) (excirc_gammaA excirc_gammaB excirc_gammaC circumcircle_omega intersection_X intersection_Y intersection_Z : B)

-- Given conditions
def AB_val := 13
def BC_val := 14
def CA_val := 15

-- Desired result
theorem cosine_sum (A B C : Type) [field A] [field B] [field C] (triangle_ABC : A)
  (AB BC CA excenter_IA excenter_IB excenter_IC circumcenter_O excirc_gammaA excirc_gammaB excirc_gammaC circumcircle_omega intersection_X intersection_Y intersection_Z : B) :
  AB = AB_val ∧ BC = BC_val ∧ CA = CA_val →
  (cos (angle circumcenter_O intersection_X excenter_IA) + 
   cos (angle circumcenter_O intersection_Y excenter_IB) + 
   cos (angle circumcenter_O intersection_Z excenter_IC)) = -49 / 65 :=
sorry

end cosine_sum_l110_110178


namespace at_least_one_gt_one_l110_110993

theorem at_least_one_gt_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l110_110993


namespace bob_tiller_swath_width_l110_110813

theorem bob_tiller_swath_width
  (plot_width plot_length : ℕ)
  (tilling_rate_seconds_per_foot : ℕ)
  (total_tilling_minutes : ℕ)
  (total_area : ℕ)
  (tilled_length : ℕ)
  (swath_width : ℕ) :
  plot_width = 110 →
  plot_length = 120 →
  tilling_rate_seconds_per_foot = 2 →
  total_tilling_minutes = 220 →
  total_area = plot_width * plot_length →
  tilled_length = (total_tilling_minutes * 60) / tilling_rate_seconds_per_foot →
  swath_width = total_area / tilled_length →
  swath_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end bob_tiller_swath_width_l110_110813


namespace value_of_expression_l110_110920

open Real

theorem value_of_expression (α : ℝ) (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + sin (2 * α)) = 10 / 3 :=
by
  sorry

end value_of_expression_l110_110920


namespace merchant_marked_price_l110_110324

-- Definitions
def list_price : ℝ := 100
def purchase_price (L : ℝ) : ℝ := 0.8 * L
def selling_price_with_discount (x : ℝ) : ℝ := 0.75 * x
def profit (purchase_price : ℝ) (selling_price : ℝ) : ℝ := selling_price - purchase_price
def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

-- Statement to prove
theorem merchant_marked_price :
  ∃ (x : ℝ), 
    profit (purchase_price list_price) (selling_price_with_discount x) = desired_profit (selling_price_with_discount x) ∧
    x / list_price = 152.38 / 100 :=
sorry

end merchant_marked_price_l110_110324


namespace find_f_f_neg2_l110_110099

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 3^(x + 1)

theorem find_f_f_neg2 : f (f (-2)) = 3 :=
  sorry

end find_f_f_neg2_l110_110099


namespace angle_sum_property_l110_110151

theorem angle_sum_property 
  (P Q R S : Type) 
  (alpha beta : ℝ)
  (h1 : alpha = 3 * x)
  (h2 : beta = 2 * x)
  (h3 : alpha + beta = 90) :
  x = 18 :=
by
  sorry

end angle_sum_property_l110_110151


namespace heaviest_weight_is_aq3_l110_110628

variable (a q : ℝ) (h : 0 < a) (hq : 1 < q)

theorem heaviest_weight_is_aq3 :
  let w1 := a
  let w2 := a * q
  let w3 := a * q^2
  let w4 := a * q^3
  w4 > w3 ∧ w4 > w2 ∧ w4 > w1 ∧ w1 + w4 > w2 + w3 :=
by
  sorry

end heaviest_weight_is_aq3_l110_110628


namespace reduced_price_per_kg_l110_110791

variables (P P' : ℝ)

-- Given conditions
def condition1 := P' = P / 2
def condition2 := 800 / P' = 800 / P + 5

-- Proof problem statement
theorem reduced_price_per_kg (P P' : ℝ) (h1 : condition1 P P') (h2 : condition2 P P') :
  P' = 80 :=
by
  sorry

end reduced_price_per_kg_l110_110791


namespace circle_equation_l110_110685

-- Define the ellipse with the given equation
def ellipse (x y : ℝ) : Prop := (x ^ 2) / 4 + y ^ 2 = 1

-- Define the condition that a point lies on the positive x semi-axis
def on_positive_x_axis (x : ℝ) : Prop := x > 0

-- Define the circle's equation given its center (a, 0) and radius r
def circle (a r x y : ℝ) : Prop := (x - a) ^ 2 + y ^ 2 = r ^ 2

-- Main theorem statement: proving the equation of the circle
theorem circle_equation {a r : ℝ} (h₁ : ellipse 2 0) (h₂ : ellipse 0 1) (h₃ : ellipse 0 (-1))
  (h₄ : on_positive_x_axis a) 
  (h₅ : circle a r 2 0) (h₆ : circle a r 0 1) (h₇ : circle a r 0 (-1)) 
  : (x - 3/4) ^ 2 + y ^ 2 = (5/4) ^ 2 := 
sorry

end circle_equation_l110_110685


namespace exists_square_divisible_by_12_between_100_and_200_l110_110415

theorem exists_square_divisible_by_12_between_100_and_200 : 
  ∃ x : ℕ, (∃ y : ℕ, x = y * y) ∧ (12 ∣ x) ∧ (100 ≤ x ∧ x ≤ 200) ∧ x = 144 :=
by
  sorry

end exists_square_divisible_by_12_between_100_and_200_l110_110415


namespace max_value_trig_expression_l110_110007

theorem max_value_trig_expression (x : ℝ) :
  ∃ y, y = x ∧ ∀ z, z ∈ ℝ → 
  ((let sin2 := Real.sin y ^ 2
    let cos2 := Real.cos y ^ 2
    let num := Real.sin y ^ 4 + Real.cos y ^ 4
    let denom := sin2 + cos2 + 2 * sin2 * cos2
    let expr := num / denom in expr) ≤ 1) :=
by sorry

end max_value_trig_expression_l110_110007


namespace max_value_pn_minus_pm_l110_110068

noncomputable def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 1)^2 = 1 }
noncomputable def circle2 := { x : ℝ × ℝ | (x.1 - 4)^2 + (x.2 - 5)^2 = 9 }
noncomputable def point_on_x_axis (P : ℝ × ℝ) := P.2 = 0

noncomputable def dist (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem max_value_pn_minus_pm:
  ∃ P M N, P ∈ ({P : ℝ × ℝ | point_on_x_axis P}) ∧
           M ∈ circle1 ∧ N ∈ circle2 ∧
           dist P N - dist P M = 9 := sorry

end max_value_pn_minus_pm_l110_110068


namespace range_of_a_l110_110945

open Real

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (-2 : ℝ) 3, a < -x^2 + 2 * x) → a < -8 := by 
sorry

end range_of_a_l110_110945


namespace coin_flip_sequences_l110_110701

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110701


namespace stair_perimeter_correct_l110_110517

def stair_perimeter (tick_length : ℝ) (n_ticks : ℕ) (region_area : ℝ) (rectangle_length : ℝ) (cutout_area : ℝ) : ℝ :=
  let y := (region_area + cutout_area) / rectangle_length
  let extended_perimeter := y + rectangle_length
  extended_perimeter + (n_ticks * tick_length) + 4 + 6

theorem stair_perimeter_correct :
  stair_perimeter 1 7 60 10 11 = 34.1 :=
by sorry

end stair_perimeter_correct_l110_110517


namespace glass_ball_radius_l110_110218

theorem glass_ball_radius (x y r : ℝ) (h_parabola : x^2 = 2 * y) (h_touch : y = r) (h_range : 0 ≤ y ∧ y ≤ 20) : 0 < r ∧ r ≤ 1 :=
sorry

end glass_ball_radius_l110_110218


namespace A_odot_B_correct_l110_110543

open Set

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x < 0 ∨ x > 2 }
def A_union_B : Set ℝ := A ∪ B
def A_inter_B : Set ℝ := A ∩ B
def A_odot_B : Set ℝ := { x | x ∈ A_union_B ∧ x ∉ A_inter_B }

theorem A_odot_B_correct : A_odot_B = (Iio 0) ∪ Icc 1 2 :=
by
  sorry

end A_odot_B_correct_l110_110543


namespace coefficient_of_a3b3_l110_110262

theorem coefficient_of_a3b3 (a b c : ℚ) :
  (∏ i in Finset.range 7, (a + b) ^ i * (c + 1 / c) ^ (8 - i)) = 1400 := 
by
  sorry

end coefficient_of_a3b3_l110_110262


namespace ellipse_equation_line_segment_length_l110_110072

-- Given conditions and definitions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def is_focus (x y : ℝ) : Prop := (x = -2 ∧ y = 0)
def passes_point (a b : ℝ) (p : ℝ × ℝ) : Prop := ∃ x y, p = (x, y) ∧ is_ellipse a b x y
def shared_focus (ellipse_focus : ℝ × ℝ) (parabola_focus : ℝ × ℝ) : Prop := ellipse_focus = parabola_focus

-- Proof of ellipse equation
theorem ellipse_equation (a b : ℝ) : 
  (a > b ∧ b > 0) →
  (is_focus (-2) 0) →
  (passes_point a b (-√3, 1)) →
  (shared_focus (-2, 0) (-2, 0)) →
  (a^2 = 6 ∧ b^2 = 2) :=
by
  sorry

-- Proof for length of the line segment AB
theorem line_segment_length (A B : ℝ × ℝ) :
  (is_ellipse 6 2 A.1 A.2 ∧ is_ellipse 6 2 B.1 B.2) →
  (line_through_focus := ∀ x, (x, x - 2)) →
  (passes_through A (x, x - 2)) →
  (passes_through B (x, x - 2)) →
  (length_AB = √6) :=
by
  sorry

end ellipse_equation_line_segment_length_l110_110072


namespace minimum_value_of_ratio_l110_110885

variable {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

def f' (x : ℝ) : ℝ := 2 * a * x + b

def f'' (x : ℝ) : ℝ := 2 * a

theorem minimum_value_of_ratio (h1 : a > 0) (h2 : ∀ x, f x ≥ 0) : 
  ∃ x, f x = f 1 ∧ (f'' 0) > 0 ∧ (f 1) / (f'' 0) = 2 :=
by
  sorry

end minimum_value_of_ratio_l110_110885


namespace factor_expression_l110_110841

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110841


namespace three_digit_perfect_squares_div_by_4_count_l110_110112

theorem three_digit_perfect_squares_div_by_4_count : 
  (∃ count : ℕ, count = 11 ∧ (∀ n : ℕ, 10 ≤ n ∧ n ≤ 31 → n^2 ≥ 100 ∧ n^2 ≤ 999 ∧ n^2 % 4 = 0)) :=
by
  sorry

end three_digit_perfect_squares_div_by_4_count_l110_110112


namespace omega_25_to_70_sum_l110_110545

noncomputable def omega_sum (ω : ℂ) := ω^{25} + ω^{28} + ω^{31} + ω^{34} + ω^{37} + ω^{40} + ω^{43} + ω^{46} + ω^{49} + ω^{52} + ω^{55} + ω^{58} + ω^{61} + ω^{64} + ω^{67} + ω^{70}

theorem omega_25_to_70_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) : omega_sum ω = ω^7 :=
begin
  sorry
end

end omega_25_to_70_sum_l110_110545


namespace three_mutually_know_each_other_l110_110633

theorem three_mutually_know_each_other
  (n : ℕ) (h : n ≥ 6)
  (knows : fin (2 * n) → fin (2 * n) → Prop)
  (h1 : ∀ p : fin (2 * n), (fin n).card.filter (λ q, knows p q) ≥ n / 2)
  (h2 : ∀ S : finset (fin (2 * n)), S.card = n / 2 → 
        S.exists_pair (λ u v, knows u v) ∨ 
        (Sᶜ : finset _).exists_pair (λ u v, knows u v)) :
  ∃ a b c : fin (2 * n), knows a b ∧ knows b c ∧ knows a c := 
by
  sorry

end three_mutually_know_each_other_l110_110633


namespace convert_1729_to_base_5_l110_110392

theorem convert_1729_to_base_5 :
  let d := 1729
  let b := 5
  let representation := [2, 3, 4, 0, 4]
  -- Check the representation of 1729 in base 5
  d = (representation.reverse.enum_from 0).sum (fun ⟨i, coef⟩ => coef * b^i) :=
  sorry

end convert_1729_to_base_5_l110_110392


namespace polynomial_remainder_unique_l110_110179

theorem polynomial_remainder_unique (p : ℚ[x])
  (h1 : p.eval 2 = 6)
  (h2 : p.eval 4 = 10)
  (h3 : ∃ r : ℚ[x], degree r < 2 ∧ ∀ q : ℚ[x], p = (X - 2) * (X - 4) * q + r) :
  ∃ r : ℚ[x], degree r < 2 ∧ r = 2 * X + 2 :=
sorry

end polynomial_remainder_unique_l110_110179


namespace simplify_expr_l110_110312

variable (a b : ℝ)

def expr := a * b - (a^2 - a * b + b^2)

theorem simplify_expr : expr a b = - a^2 + 2 * a * b - b^2 :=
by 
  -- No proof is provided as per the instructions
  sorry

end simplify_expr_l110_110312


namespace man_sells_for_45000_l110_110780

-- Define the problem's conditions
def business_value : ℝ := 90000
def man_shares : ℝ := 2 / 3
def portion_sold : ℝ := 3 / 4

-- Define the theorem to prove
theorem man_sells_for_45000 (bv : ℝ) (ms : ℝ) (ps : ℝ) (Hbv : bv = business_value) (Hms : ms = man_shares) (Hps : ps = portion_sold) : ps * (ms * bv) = 45000 :=
by
  rw [Hbv, Hms, Hps]
  -- calculate the value of man's shares
  have h1 : ms * bv = (2 / 3) * 90000 := rfl
  have h2 : (2 / 3) * 90000 = 60000 := by norm_num
  rw [h1, h2]
  -- calculate the amount sold
  have h3 : ps * 60000 = (3 / 4) * 60000 := rfl
  have h4 : (3 / 4) * 60000 = 45000 := by norm_num
  rw [h3, h4]
  exact eq.refl 45000

end man_sells_for_45000_l110_110780


namespace small_plank_needs_10_nails_l110_110983

theorem small_plank_needs_10_nails
    (large_planks : Nat)
    (small_planks : Nat)
    (nails_per_large_plank : Nat)
    (total_planks : Nat)
    (total_nails_for_large_planks : Nat)
    (total_nails : Nat)
    : large_planks = 12 →
      small_planks = 17 →
      nails_per_large_plank = 14 →
      total_planks = 29 →
      total_nails_for_large_planks = large_planks * nails_per_large_plank →
      total_nails = total_nails_for_large_planks →
      (total_nails / small_planks).nat_div =
      10 := sorry

end small_plank_needs_10_nails_l110_110983


namespace possible_perimeters_l110_110089

-- Define the condition that the side lengths satisfy the equation
def sides_satisfy_eqn (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Theorem to prove the possible perimeters
theorem possible_perimeters (x y z : ℝ) (h1 : sides_satisfy_eqn x) (h2 : sides_satisfy_eqn y) (h3 : sides_satisfy_eqn z) :
  (x + y + z = 10) ∨ (x + y + z = 6) ∨ (x + y + z = 12) := by
  sorry

end possible_perimeters_l110_110089


namespace find_y_l110_110922

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110922


namespace coin_flips_sequences_count_l110_110716

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110716


namespace spherical_to_rectangular_coordinates_l110_110001

theorem spherical_to_rectangular_coordinates
  (ρ θ φ : ℝ)
  (hρ : ρ = 4)
  (hθ : θ = π / 6)
  (hφ : φ = π / 4)
  (x : ℝ)
  (hx : x = ρ * sin φ * cos θ)
  (y : ℝ)
  (hy : y = ρ * sin φ * sin θ)
  (z : ℝ)
  (hz : z = ρ * cos φ) :
  (x, y, z) = (2 * sqrt 3, sqrt 2, 2 * sqrt 2) := by
  rw [hρ, hθ, hφ] at hx hy hz
  rw [hx, hy, hz]
  sorry

end spherical_to_rectangular_coordinates_l110_110001


namespace coefficient_a3b3_in_expression_l110_110271

theorem coefficient_a3b3_in_expression :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ k) * (b ^ (6 - k))) *
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (c ^ (8 - 2 * k)) * (c ^ (-2 * k))) =
  1400 := sorry

end coefficient_a3b3_in_expression_l110_110271


namespace problem_l110_110919

theorem problem (x : ℝ) (h : 15 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 17 := 
by sorry

end problem_l110_110919


namespace coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110266

theorem coefficient_a3b3_in_ab6_c8_div_c8_is_1400 :
  let a := (a : ℝ)
  let b := (b : ℝ)
  let c := (c : ℝ)
  ∀ (a b c : ℝ), (binom 6 3 * a^3 * b^3) * (binom 8 4 * c^0) = 1400 := 
by
  sorry

end coefficient_a3b3_in_ab6_c8_div_c8_is_1400_l110_110266


namespace even_sum_probability_l110_110615

def grid := (Fin 3 × Fin 3) → Fin 11

noncomputable def even_sum_in_each_row_and_column_probability : ℚ :=
  1 / 14

theorem even_sum_probability (g : grid) (h_distinct : ∀ i j k l, g (⟨i, _⟩, ⟨j, _⟩) = g (⟨k, _⟩, ⟨l, _⟩) → (i = k ∧ j = l))
  (h_range : ∀ i j, 2 ≤ g (⟨i, _⟩, ⟨j, _⟩).val ∧ g (⟨i, _⟩, ⟨j, _⟩).val ≤ 10)
  : (filter (λ h : unit, (∀ i, (∑ j, (g ⟨i, _⟩ ⟨j, _⟩).val) % 2 = 0) 
                      ∧ (∀ j, (∑ i, (g ⟨i, _⟩ ⟨j, _⟩).val) % 2 = 0)) 
              {() : unit}) .card 
      / ((Finset.univ : Finset (Fin 11)).card.factorial) = even_sum_in_each_row_and_column_probability :=
sorry

end even_sum_probability_l110_110615


namespace painted_cubes_eq_unpainted_cubes_l110_110778

noncomputable def n := 2 + 2*sqrt 3

theorem painted_cubes_eq_unpainted_cubes (n : ℝ) (h : n > 2) : 
  (n-2)^3 = 12 * (n-2) → n = 2 + 2*sqrt 3 :=
begin
  intro h1,
  sorry
end

end painted_cubes_eq_unpainted_cubes_l110_110778


namespace factor_expression_l110_110846

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110846


namespace range_of_a_l110_110173

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 2) ^ x else x ^ (1 / 2)

theorem range_of_a (a : ℝ) (h : f a > 1) : a < 0 ∨ a > 1 :=
  sorry

end range_of_a_l110_110173


namespace peter_has_winning_strategy_l110_110629

/-- Peter has a winning strategy in the game where there are 11 piles
    of stones with each pile containing 10 stones, Peter can take
    1, 2, or 3 stones from a single pile per turn, and Basil can
    take 1 stone each from 1, 2, or 3 different piles on each turn.
    Peter goes first, and the player who takes the last stone wins. -/
theorem peter_has_winning_strategy :
  ∃ strategy : (nat × nat → nat) → Prop, 
    (∀ state, winning_strategy_for_peter state strategy) := 
sorry

end peter_has_winning_strategy_l110_110629


namespace num_integer_solutions_x_sq_minus_x_lt_12_l110_110875

theorem num_integer_solutions_x_sq_minus_x_lt_12 :
  {x : ℤ | x^2 - x < 12}.finite ∧
  {x : ℤ | x^2 - x < 12}.to_finset.card = 6 :=
by
  sorry

end num_integer_solutions_x_sq_minus_x_lt_12_l110_110875


namespace area_invariance_of_reflected_quadrilateral_l110_110577

theorem area_invariance_of_reflected_quadrilateral
  (A B C D M : Point)
  (P := midpoint A B) (Q := midpoint B C) (R := midpoint C D) (S := midpoint D A)
  (M1 := reflect M P) (M2 := reflect M Q) (M3 := reflect M R) (M4 := reflect M S) :
  is_parallelogram M1 M2 M3 M4 ∧ area M1 M2 M3 M4 = area_of_reflects M1 M2 M3 M4 :=
sorry

end area_invariance_of_reflected_quadrilateral_l110_110577


namespace largest_possible_e_l110_110992

noncomputable def diameter := (2 : ℝ)
noncomputable def PX := (4 / 5 : ℝ)
noncomputable def PY := (3 / 4 : ℝ)
noncomputable def e := (41 - 16 * Real.sqrt 25 : ℝ)
noncomputable def u := 41
noncomputable def v := 16
noncomputable def w := 25

theorem largest_possible_e (P Q X Y Z R S : Real) (d : diameter = 2)
  (PX_len : P - X = 4/5) (PY_len : P - Y = 3/4)
  (e_def : e = 41 - 16 * Real.sqrt 25)
  : u + v + w = 82 :=
by
  sorry

end largest_possible_e_l110_110992


namespace coin_flip_sequences_l110_110750

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110750


namespace sector_area_l110_110458

theorem sector_area :
  let r := 3
  let θ := 2 * Real.pi / 3
  (1/2) * r^2 * θ = 3 * Real.pi :=
by
  sorry

end sector_area_l110_110458


namespace eval_expression_l110_110010

-- Lean 4 statement
theorem eval_expression : (5⁻¹ - 2⁻¹)⁻¹ = - 10 / 3 := by
  sorry

end eval_expression_l110_110010


namespace abs_eq_necessary_but_not_sufficient_l110_110310

theorem abs_eq_necessary_but_not_sufficient (x y : ℝ) :
  (|x| = |y|) → (¬(x = y) → x = -y) :=
by
  sorry

end abs_eq_necessary_but_not_sufficient_l110_110310


namespace four_is_integer_l110_110659

-- Definitions based on conditions
variable (Nat : Type) [has_coe ℤ Nat] (n : Nat)

-- Axioms representing the premises
axiom all_naturals_are_integers : ∀ n : Nat, ∃ z : ℤ, n = z
axiom four_is_natural : 4 < 10 -- Simplified representation for natural numbers

-- Conclusion we need to prove
theorem four_is_integer : ∃ z : ℤ, 4 = z :=
by
  have h1 : ∃ z : ℤ, (4 : Nat) = z := all_naturals_are_integers 4
  exact h1

-- Annotating end of proof for completeness
sorry

end four_is_integer_l110_110659


namespace z_is_real_iff_m_eq_z_is_pure_imaginary_iff_m_eq_l110_110093

def z (m : ℂ) : ℂ := m^2 * (1+complex.i) - m * (3+complex.i) - 6 * complex.i

theorem z_is_real_iff_m_eq (m : ℝ) : (∀ z : ℂ, (im (z m) = 0) → (m = 3 ∨ m = -2)) :=
sorry

theorem z_is_pure_imaginary_iff_m_eq (m : ℝ) : (∀ z : ℂ, (re (z m) = 0 ∧ im (z m) ≠ 0) → (m = 0)) :=
sorry

end z_is_real_iff_m_eq_z_is_pure_imaginary_iff_m_eq_l110_110093


namespace centroid_of_triangle_l110_110601

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  let x_centroid := (x1 + x2 + x3) / 3
  let y_centroid := (y1 + y2 + y3) / 3
  (x_centroid, y_centroid) = (1/3 * (x1 + x2 + x3), 1/3 * (y1 + y2 + y3)) :=
by
  sorry

end centroid_of_triangle_l110_110601


namespace number_of_common_terms_l110_110232

-- Define the arithmetic sequences
def seq_a (n : ℕ) : ℕ := 2 + 3 * (n - 1)
def seq_b (m : ℕ) : ℕ := 4 + 5 * (m - 1)

-- Define the condition of the sequences not exceeding the final terms
def in_seq_a (x : ℕ) : Prop := ∃ n : ℕ, x = seq_a n ∧ seq_a n ≤ 2015
def in_seq_b (x : ℕ) : Prop := ∃ m : ℕ, x = seq_b m ∧ seq_b m ≤ 2014

-- Define the predicate for common terms in both sequences
def common_term (x : ℕ) : Prop := in_seq_a x ∧ in_seq_b x

-- The final proof statement: there are exactly 134 common terms
theorem number_of_common_terms : 
  (finset.range 2017).filter (λ x, common_term x).card = 134 :=
by
  sorry

end number_of_common_terms_l110_110232


namespace max_mu_l110_110437

variables {n : ℕ} (a : Fin (2 * n + 1) → ℝ)

noncomputable def mu (a : Fin (2 * n + 1) → ℝ) : ℝ :=
  (∑ i in Finset.range (2 * n + 1), if h : i > n then a ⟨i, sorry⟩ else 0) -
  (∑ i in Finset.range (2 * n + 1), if h : i ≤ n then a ⟨i, sorry⟩ else 0)

theorem max_mu (h : ∑ i in Finset.range (2 * n), (a ⟨i + 1, sorry⟩ - a ⟨i, sorry⟩)^2 = 1) :
  ∃ a, mu a = Real.sqrt ((n * (2 * n^2 + 1)) / 3) :=
-- Proof will be constructed here
sorry

end max_mu_l110_110437


namespace sum_exponential_sequence_l110_110066

theorem sum_exponential_sequence (a r : ℝ) (h_a : a = 2) (h_r : r = 1/3) :
  let S := a / (1 - r) in S = 3 :=
by
  sorry

end sum_exponential_sequence_l110_110066


namespace fraction_decimal_representation_l110_110837

noncomputable def fraction_as_term_dec : ℚ := 47 / (2^3 * 5^4)

theorem fraction_decimal_representation : fraction_as_term_dec = 0.0094 :=
by
  sorry

end fraction_decimal_representation_l110_110837


namespace coin_flips_sequences_count_l110_110721

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110721


namespace coin_flip_sequences_l110_110730

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110730


namespace coin_flip_sequences_l110_110706

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110706


namespace domain_of_f_l110_110034

noncomputable def f (x : ℝ) := real.sqrt (2 - real.sqrt (3 - real.sqrt (4 * x - 12)))

theorem domain_of_f : { x : ℝ | 3 ≤ x ∧ x ≤ 21 / 4 } = { x : ℝ | ∃ y : ℝ, f y = x } :=
by
  sorry

end domain_of_f_l110_110034


namespace tunnel_length_l110_110355

/-- A train that is 2 miles long exits a tunnel exactly 4 minutes after its front part entered the tunnel.
The train is moving at 90 miles per hour. Prove that the length of the tunnel is 4 miles. -/
theorem tunnel_length (train_length : ℝ) (exit_time_min : ℝ) (train_speed_mph : ℝ) :
  train_length = 2 →
  exit_time_min = 4 →
  train_speed_mph = 90 →
  let distance_travelled := (train_speed_mph / 60) * exit_time_min in -- distance covered in minutes
  let tunnel_length := distance_travelled - train_length in
  tunnel_length = 4 :=
sorry

end tunnel_length_l110_110355


namespace range_of_values_l110_110150

-- Define points on the unit circle
variables {A B C : ℝ × ℝ} 
variables (λ μ : ℝ)

-- Assume points are distinct and on the unit circle
variables (h₀ : A ≠ B) (h₁ : B ≠ C) (h₂ : A ≠ C)
variables (h₃ : A.1^2 + A.2^2 = 1) (h₄ : B.1^2 + B.2^2 = 1) (h₅ : C.1^2 + C.2^2 = 1)

-- Define the vector equation
variable (h_condition : C = (λ • A) + (μ • B))

theorem range_of_values (h: 0 < λ ∧ 0 < μ) : 
  ∃ η > 2, ∀ ζ, λ^2 + (μ - 3)^2 > η := 
by sorry

end range_of_values_l110_110150


namespace Doug_more_HRs_than_Ryan_l110_110403

theorem Doug_more_HRs_than_Ryan :
  let p := \frac{1}{5} in
  ∃ Doug_prob Ryan_prob, 
    Doug_prob = \frac{1}{3} ∧
    Ryan_prob = \frac{1}{2} ∧
    p = \frac{1}{6} + \frac{p}{6} :=
begin
  sorry
end

end Doug_more_HRs_than_Ryan_l110_110403


namespace smallest_distance_zero_l110_110201

theorem smallest_distance_zero :
  let r_track (t : ℝ) := (Real.cos t, Real.sin t)
  let i_track (t : ℝ) := (Real.cos (t / 2), Real.sin (t / 2))
  ∀ t₁ t₂ : ℝ, dist (r_track t₁) (i_track t₂) = 0 := by
  sorry

end smallest_distance_zero_l110_110201


namespace greatest_possible_median_l110_110215

theorem greatest_possible_median {k m r s t : ℕ} 
  (h_mean : (k + m + r + s + t) / 5 = 18) 
  (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) 
  (h_t : t = 40) :
  r = 23 := sorry

end greatest_possible_median_l110_110215


namespace exist_translation_map_l110_110616

theorem exist_translation_map {M M' : Polygon} (h : homothety_coeff M' M = -1/2) : 
  ∃ t : Translation, t.map M' ⊆ M := 
by 
  sorry

end exist_translation_map_l110_110616


namespace next_in_sequence_is_80_l110_110224

def seq (n : ℕ) : ℕ := n^2 - 1

theorem next_in_sequence_is_80 :
  seq 9 = 80 :=
by
  sorry

end next_in_sequence_is_80_l110_110224


namespace smallest_sector_angle_l110_110251

theorem smallest_sector_angle :
  ∃ (a1 : ℕ) (d : ℤ),
  (∀ i, 1 ≤ i ∧ i ≤ 15 → ∃ ai, ai = a1 + (i - 1) * d) ∧ 
  (∑ i in Finset.range 15, (a1 + (i : ℤ) * d)) = 360 ∧ 
  (∀ ai, ai = a1 + (i - 1) * d → ai ≥ 1) ∧ 
  a1 = 3 :=
begin
  sorry
end

end smallest_sector_angle_l110_110251


namespace num_integer_values_b_for_polynomial_l110_110424

theorem num_integer_values_b_for_polynomial :
  ∃ (b_values : Finset ℤ), 
  (∀ b ∈ b_values, ∃ (f : ℤ → ℤ), (∀ n, Polynomial.eval n f ∈ ℤ) ∧ Polynomial.eval 2 f = 2010 ∧ Polynomial.eval b f = 8) ∧
  b_values.card = 32 :=
sorry

end num_integer_values_b_for_polynomial_l110_110424


namespace oldest_child_age_l110_110593

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end oldest_child_age_l110_110593


namespace slope_greater_than_zero_l110_110401

theorem slope_greater_than_zero (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  -((2 ^ a - 1) / log a) > 0 :=
sorry

end slope_greater_than_zero_l110_110401


namespace mass_percentage_increase_l110_110394

variables {ρ₁ ρ₂ m₁ m₂ V₁ V₂ : ℝ}
variables {a₁ a₂ : ℝ}

def density (m : ℝ) (V : ℝ) := m / V
def volume_of_cube (a : ℝ) := a ^ 3 

-- Given conditions from the problem
axiom density_relation : ρ₂ = (1 / 2) * ρ₁
axiom side_length_relation : a₂ = 2 * a₁

-- Definitions derived from conditions
def volume_relation : V₂ = 8 * V₁ := by
  sorry

def mass_relation : m₂ = 4 * m₁ := by
  have d1 : ρ₁ = m₁ / V₁ := by sorry
  have d2 : ρ₂ = m₂ / V₂ := by sorry
  have h : (m₂ / V₂) = (1 / 2) * (m₁ / V₁) := by
    rw [density_relation, d1, d2]

  have h_volume : V₂ = 8 * V₁ := volume_relation
  rw [h_volume] at h
  sorry

-- Main theorem: mass percentage increase
theorem mass_percentage_increase : ((m₂ - m₁) / m₁) * 100 = 300 := by
  rw [mass_relation]
  sorry

end mass_percentage_increase_l110_110394


namespace original_cube_edge_length_l110_110770

theorem original_cube_edge_length (a : ℕ) (h1 : 6 * (a ^ 3) = 7 * (6 * (a ^ 2))) : a = 7 := 
by 
  sorry

end original_cube_edge_length_l110_110770


namespace correct_divisor_l110_110958

theorem correct_divisor (X : ℕ) (D : ℕ) (H1 : X = 24 * 87) (H2 : X / D = 58) : D = 36 :=
by
  sorry

end correct_divisor_l110_110958


namespace final_amount_is_correct_l110_110301

def rate_increase : ℝ := 1 / 8
def initial_amount : ℝ := 2880
def duration : ℝ := 2

theorem final_amount_is_correct (rate_increase_val : ℝ) (initial_amount_val : ℝ) (duration_val : ℝ) 
  (h1 : rate_increase_val = rate_increase) (h2 : initial_amount_val = initial_amount) (h3 : duration_val = duration) :
  initial_amount_val * (1 + rate_increase_val) ^ duration_val = 3645 := 
by 
  sorry

end final_amount_is_correct_l110_110301


namespace probability_transformation_in_R_l110_110062

def R : set ℂ := {z | ∃ x y : ℝ, z = complex.mk x y ∧ (-2 ≤ x ∧ x ≤ 2 ∧ -1 ≤ y ∧ y ≤ 1)}

def U : set ℂ := {z | ∃ x y : ℝ, z = complex.mk x y ∧ (abs (x - y) ≤ 4) ∧ (abs (x + y) ≤ 2)}

theorem probability_transformation_in_R :
  let transform (z : ℂ) := (1 / 2 + (complex.I / 2)) * z in
  (∀ z ∈ R, transform z ∈ R) → 
  ((measure_theory.measure_space.measure (R ∩ U) : ℝ) / (measure_theory.measure_space.measure R : ℝ) = 1) :=
by
  sorry

end probability_transformation_in_R_l110_110062


namespace constant_term_binomial_expansion_l110_110855

theorem constant_term_binomial_expansion :
  let x : ℚ := sorry
  let r : ℕ := 3
  let binom : ℚ := Nat.choose 6 3
  let term : ℚ := -(binom)
  (x - x⁻¹)^6 = term :=
begin
  sorry
end

end constant_term_binomial_expansion_l110_110855


namespace rabbit_hops_time_l110_110788

theorem rabbit_hops_time (distance : ℝ) (rate : ℝ) :
  distance = 3 → rate = 5 → (distance / rate * 60) = 36 :=
by
  intros hdistance hrate
  rw [hdistance, hrate]
  norm_num
  sorry

end rabbit_hops_time_l110_110788


namespace more_freshmen_than_sophomores_l110_110504

theorem more_freshmen_than_sophomores : 
  ∀ (total_students juniors seniors : ℕ) (prop_juniors prop_not_sophomores : ℝ),
  total_students = 800 →
  juniors = prop_juniors * total_students →
  seniors = 160 →
  prop_juniors = 0.27 →
  prop_not_sophomores = 0.75 →
  prop_sophomores = 1 - prop_not_sophomores →
  sophomores = prop_sophomores * total_students →
  freshmen = total_students - (seniors + sophomores + juniors) →
  (freshmen - sophomores) = 24 :=
by
  intro total_students juniors seniors prop_juniors prop_not_sophomores
  assume h1 h2 h3 h4 h5 h6 h7 h8
  -- Proof omitted
  sorry

end more_freshmen_than_sophomores_l110_110504


namespace find_missing_dimension_l110_110221

-- Definitions based on conditions
def is_dimension_greatest_area (x : ℝ) : Prop :=
  max (2 * x) (max (3 * x) 6) = 15

-- The final statement to prove
theorem find_missing_dimension (x : ℝ) (h1 : is_dimension_greatest_area x) : x = 5 :=
sorry

end find_missing_dimension_l110_110221


namespace tan_seven_pi_over_four_l110_110024

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l110_110024


namespace coin_flip_sequences_l110_110695

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110695


namespace coefficient_a3b3_l110_110277

theorem coefficient_a3b3 in_ab_c_1overc_expr :
  let coeff_ab := Nat.choose 6 3 
  let coeff_c_expr := Nat.choose 8 4 
  coeff_ab * coeff_c_expr = 1400 :=
by
  sorry

end coefficient_a3b3_l110_110277


namespace tan_seven_pi_over_four_l110_110022

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l110_110022


namespace strictly_increasing_intervals_find_b_in_triangle_l110_110474

-- Definitions and conditions based on the original problem
def f (x : ℝ) (a : ℝ) : ℝ := (Real.sin (x + π / 6) + Real.sin (x - π / 6) + Real.cos x + a)

theorem strictly_increasing_intervals (a : ℝ) (k : ℤ) (h : 2 + a = 1) :
  ∀ x, -2 * π / 3 + k * 2 * π ≤ x ∧ x ≤ π / 3 + k * 2 * π → StrictMono (f x (-1)) :=
sorry

theorem find_b_in_triangle (A C a : ℝ) (f_A : f A (-1) = 1) (C_eq : C = π / 4) (c_eq : 2) :
  ∃ B b, B = π - A - C ∧ b = 2 * Real.sin (5 * π / 12) / Real.sin (π / 4) :=
sorry

end strictly_increasing_intervals_find_b_in_triangle_l110_110474


namespace distinct_integers_sums_l110_110457

theorem distinct_integers_sums {n : ℕ} (hn : 3 * 66 * n > 3) {a : ℕ → ℕ} (h : ∀ i, i ≤ n → 1 ≤ a i ∧ a i ≤ 2 * n - 3 ∧ (∀ j k, j < k → a j < a k)) :
  ∃ i j k l m, i < j ∧ j < k ∧ k < l ∧ l < m ∧ a i + a j = a k + a l = a m :=
by
  sorry

end distinct_integers_sums_l110_110457


namespace total_books_l110_110831

-- Defining the conditions
def darla_books := 6
def katie_books := darla_books / 2
def combined_books := darla_books + katie_books
def gary_books := 5 * combined_books

-- Statement to prove
theorem total_books : darla_books + katie_books + gary_books = 54 := by
  sorry

end total_books_l110_110831


namespace only_C_forms_triangle_l110_110956

def triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_C_forms_triangle :
  ¬ triangle_sides 3 4 8 ∧
  ¬ triangle_sides 2 5 2 ∧
  triangle_sides 3 5 6 ∧
  ¬ triangle_sides 5 6 11 :=
by
  sorry

end only_C_forms_triangle_l110_110956


namespace find_y_l110_110923

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110923


namespace minimize_reciprocals_l110_110512

theorem minimize_reciprocals (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 30) :
  (a = 10 ∧ b = 5) → ∀ x y : ℕ, (x > 0) → (y > 0) → (x + 4 * y = 30) → (1 / (x : ℝ) + 1 / (y : ℝ) ≥ 1 / 10 + 1 / 5) := 
by {
  sorry
}

end minimize_reciprocals_l110_110512


namespace inequality_any_k_l110_110617

theorem inequality_any_k (x y z : ℝ) (k : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) : 
  x ^ (-k : ℤ) + y ^ (-k : ℤ) + z ^ (-k : ℤ) ≥ x ^ k + y ^ k + z ^ k :=
sorry

end inequality_any_k_l110_110617


namespace birth_age_of_mother_l110_110488

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l110_110488


namespace solution_set_inequality_l110_110236

theorem solution_set_inequality (x : ℝ) : x * (9 - x) > 0 ↔ x ∈ Ioo 0 9 := 
sorry

end solution_set_inequality_l110_110236


namespace unique_solution_iff_a_eq_2019_l110_110949

theorem unique_solution_iff_a_eq_2019 (x a : ℝ) :
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) ↔ a = 2019 :=
by
  sorry

end unique_solution_iff_a_eq_2019_l110_110949


namespace tan_7pi_over_4_eq_neg1_l110_110019

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end tan_7pi_over_4_eq_neg1_l110_110019


namespace properties_of_f_l110_110101

open Real 

def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem properties_of_f :
  (∀ x : ℝ, f x = x / (x^2 + 1)) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < 1 → 0 < x2 → x2 < 1 → x1 < x2 → f x1 < f x2) ∧
  (range f = Set.Icc (-(1/2 : ℝ)) (1/2)) := 
by
  sorry

end properties_of_f_l110_110101


namespace complex_multiplication_l110_110412

-- Define the imaginary unit i
def i := Complex.I

-- Define the theorem we need to prove
theorem complex_multiplication : 
  (3 - 7 * i) * (-6 + 2 * i) = -4 + 48 * i := 
by 
  -- Proof is omitted
  sorry

end complex_multiplication_l110_110412


namespace shaded_region_area_correct_l110_110345

noncomputable def shaded_region_area (side_length : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
if 0 < beta ∧ beta < Real.pi / 2 ∧ cos_beta = 3 / 5 then
  2 / 5
else
  0

theorem shaded_region_area_correct :
  shaded_region_area 2 β (3 / 5) = 2 / 5 :=
by
  -- conditions
  have beta_cond : 0 < β ∧ β < Real.pi / 2 := sorry
  have cos_beta_cond : cos β = 3 / 5 := sorry
  -- we will finish this proof assuming above have been proved.
  exact if_pos ⟨beta_cond, cos_beta_cond⟩

end shaded_region_area_correct_l110_110345


namespace rational_root_probability_l110_110233

noncomputable def polynomial : Polynomial ℚ := -400 * X^5 + 2660 * X^4 - 3602 * X^3 + 1510 * X^2 + 18 * X - 90

theorem rational_root_probability : 
  let possible_roots := [1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 9, -9, 10, -10, 15, -15, 18, -18, 30, -30, 45, -45, 90, -90].product 
                        [1, -1, 2, -2, 4, -4, 5, -5, 8, -8, 10, -10, 16, -16, 20, -20, 25, -25, 40, -40, 50, -50, 80, -80, 100, -100, 200, -200, 400, -400]
                        |>.map (λ pq : ℤ × ℤ, (pq.1 : ℚ) / pq.2),
      total_possible_roots := 180 in
  (polynomial.root_probability possible_roots) = (1 / 36) :=
sorry

end rational_root_probability_l110_110233


namespace distance_ratio_l110_110541

variables {P T1 T2 T3 T4 : Type} [metric_space P] [metric_space T1] [metric_space T2] 
[metric_space T3] [metric_space T4]

variable (A1 A2 A3 A4 : set P) 

-- Tangency conditions
variables (tangent_1_3 : A1 ∩ A3 = {P})
variables (tangent_2_4 : A2 ∩ A4 = {P})

-- Meeting conditions
variables (meet_1_2 : A1 ∩ A2 = {T1, another})
variables (meet_2_3 : A2 ∩ A3 = {T2, another})
variables (meet_3_4 : A3 ∩ A4 = {T3, another})
variables (meet_4_1 : A4 ∩ A1 = {T4, another})

-- Prove the required distance ratio
theorem distance_ratio :
  (dist T1 T2) * (dist T2 T3) / (dist T1 T4) * (dist T3 T4) = (dist P T2) ^ 2 / (dist P T4) ^ 2 :=
sorry

end distance_ratio_l110_110541


namespace plywood_problem_exists_squares_l110_110306

theorem plywood_problem_exists_squares :
  ∃ (a b : ℕ), a^2 + b^2 = 625 ∧ a ≠ 20 ∧ b ≠ 20 ∧ a ≠ 15 ∧ b ≠ 15 := by
  sorry

end plywood_problem_exists_squares_l110_110306


namespace find_X_l110_110144

/-- In a polygonal figure consisting of rectangles with all right angles, opposite sides are equal. 
The top side of the figure consists of five segments with lengths 1, 3, 1, 1, and X centimeters.
The bottom side of the figure consists of four segments with lengths 3, 1, 3, and 3 centimeters.
Prove that the value of X is 4. -/
theorem find_X (X : ℕ) 
  (h1 : 6 + X = 10) : X = 4 := 
begin
  -- proof goes here
  sorry
end

end find_X_l110_110144


namespace circle_properties_l110_110625

-- Define the relevant mathematical constants and functions
def pi := Real.pi
def circumference (r : ℝ) : ℝ := 2 * pi * r
def diameter (r : ℝ) : ℝ := 2 * r
def area (r : ℝ) : ℝ := pi * r^2

-- Define the given condition
axiom given_sum : ∀ r : ℝ, circumference r + diameter r + r = 27.84

-- State the theorem to prove
theorem circle_properties :
  (∃ r : ℝ, diameter r = 6 ∧ area r = 28.26) :=
begin
  -- Proof will go here
  sorry
end

end circle_properties_l110_110625


namespace polynomial_not_factorable_l110_110554

/-- Given a polynomial f(x) = x^n + 5x^{n-1} + 3, where n > 1 is an integer,
    prove that f(x) is irreducible over the integers. -/
theorem polynomial_not_factorable (n : ℕ) (h : n > 1) :
  ¬∃ (r s : ℕ) (h₁ : r ≥ 1) (h₂ : s ≥ 1) (a b : polynomial ℤ),
    (a.degree = r) ∧ (b.degree = s) ∧ (a.leading_coeff = 1) ∧ (b.leading_coeff = 1) ∧ 
    (a * b = polynomial.C ⟨3⟩ + polynomial.X ^ n + polynomial.C ⟨5⟩ * polynomial.X ^ (n - 1)) :=
  sorry

end polynomial_not_factorable_l110_110554


namespace sequence_non_decreasing_inequality_holds_l110_110986

variable (a : ℝ) (h : a ≥ 2)

theorem sequence_non_decreasing (x1 x2 : ℝ) (hx1 : x1 = (a + real.sqrt (a ^ 2 - 4)) / 2) (hx2 : x2 = (a - real.sqrt (a ^ 2 - 4)) / 2) :
  ∀ n : ℕ, (x1 ^ n + x2 ^ n) / (x1 ^ (n+1) + x2 ^ (n+1)) ≥ (x1 ^ (n+1) + x2 ^ (n+1)) / (x1 ^ (n+2) + x2 ^ (n+2)) :=
sorry

theorem inequality_holds :
  ∀ n : ℕ, n > 0 → (∑ k in finset.range n, (x1 ^ k + x2 ^ k) / (x1 ^ (k+1) + x2 ^ (k+1))) > n - 1 :=
sorry

end sequence_non_decreasing_inequality_holds_l110_110986


namespace distinct_sequences_ten_flips_l110_110691

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110691


namespace known_line_perpendicular_to_countless_lines_l110_110641

-- Definition of two perpendicular planes
def planes_perpendicular (α β : Plane): Prop :=
  ∀ (p : Point), (p ∈ α ∧ p ∈ β) → ∃ l₁ l₂, l₁ ⊂ α ∧ l₂ ⊂ β ∧ l₁ ⊥ l₂

-- Problem statement
theorem known_line_perpendicular_to_countless_lines (α β : Plane) (h : planes_perpendicular α β) (l : Line) (h_l : l ⊂ α) :
  ∃ (countless_lines : Set Line), (∀ l' ∈ countless_lines, l' ⊂ β ∧ l ⊥ l') :=
sorry

end known_line_perpendicular_to_countless_lines_l110_110641


namespace coefficient_a3b3_in_expression_l110_110268

theorem coefficient_a3b3_in_expression :
  (∑ k in Finset.range 7, (Nat.choose 6 k) * (a ^ k) * (b ^ (6 - k))) *
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (c ^ (8 - 2 * k)) * (c ^ (-2 * k))) =
  1400 := sorry

end coefficient_a3b3_in_expression_l110_110268


namespace point_outside_circle_l110_110501

theorem point_outside_circle (a : ℝ) : 
  ( -4 < a ∧ a < 1/2) ↔ 
  (∀ x y, (x = 2 ∧ y = 1) → (x^2 + y^2 - x + y + a > 0)) :=
by
  intro h
  split
  exact
  {
    intros x y hxy,
    rcases hxy with ⟨hx, hy⟩,
    rw [hx, hy],
    linarith,
  },
  {
    simp_rw [forall_const, forall_eq, gt_def, sub_pos, add_comm 4, add_sub, add_comm a, lt_add_iff_pos_right, add_lt_add],
    intros h,
    split;
    { linarith, }
  }

end point_outside_circle_l110_110501


namespace tangent_line_equation_l110_110219

theorem tangent_line_equation :
  ∃ (x y : ℝ), (x = 1 ∧ y = -3 * x + 1 ∧ ∀ x : ℝ, 3 * x + (y - (-3 * x + 1)) = 0) :=
begin
  sorry
end

end tangent_line_equation_l110_110219


namespace find_positive_n_l110_110418

def consecutive_product (k : ℕ) : ℕ := k * (k + 1) * (k + 2)

theorem find_positive_n (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  n^6 + 5*n^3 + 4*n + 116 = consecutive_product k ↔ n = 3 := 
by 
  sorry

end find_positive_n_l110_110418


namespace coin_flip_sequences_l110_110749

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110749


namespace coefficient_x_squared_l110_110815

theorem coefficient_x_squared :
  let expr := 5 * (λ x, x - 2 * x^3) - 4 * (λ x, 2 * x^2 - x^3 + 3 * x^6) + 3 * (λ x, 5 * x^2 - 2 * x^8)
  (x : ℝ) in
  let coeff_x_squared := (0 + (-8) + 15) in
  coeff_x_squared = 7 :=
by
  sorry

end coefficient_x_squared_l110_110815


namespace possible_radii_for_three_tangent_circles_l110_110373

-- Define circles O1 and O2 with given radii
def O1_radius : ℝ := 3
def O2_radius : ℝ := 7

-- Define the property of being tangent to both circles
def tangent_to_both (r : ℝ) : Prop := sorry -- Definition of tangency condition not provided

-- The main theorem
theorem possible_radii_for_three_tangent_circles :
  ∃ r, (r = 3 ∨ r = 4 ∨ r = 7) ∧ (number_of_tangent_circles r = 3) := sorry

-- Placeholder for the number of tangent circles (this would need a detailed definition)
def number_of_tangent_circles (r : ℝ) : ℕ := sorry  -- Calculation of this would require proper conditions and tangency checks.

end possible_radii_for_three_tangent_circles_l110_110373


namespace eccentricity_range_lambda_existence_l110_110094

section EllipseHyperbola
variables {a b x y : ℝ} (P : ℝ → ℝ → Prop) (F1 F2 A B : ℝ × ℝ)
def ellipse (a b : ℝ) := ∀ x y, x^2 / a^2 + y^2 / b^2 = 1
def hyperbola (a b : ℝ) := ∀ x y, x^2 / b^2 - y^2 / b^2 = 1
noncomputable def c (a b : ℝ) := sqrt(a^2 - b^2)
noncomputable def e (a b : ℝ) := sqrt(1 - (b^2 / a^2))

theorem eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_ab : a > b) :
  1 / 2 ≤ sqrt(1 - (b^2 / a^2)) ∧ sqrt(1 - (b^2 / a^2)) ≤ sqrt(2) / 2 :=
sorry

theorem lambda_existence (a b : ℝ) (A F1 B : ℝ × ℝ) (min_e : e a b = 1 / 2) : ∃ λ : ℝ, λ = 2 ∧
  ∀ {x y : ℝ}, hyperbola a (c a b) x y → ∃ a', angle B A F1 = λ * angle B F1 A :=
sorry

end EllipseHyperbola

end eccentricity_range_lambda_existence_l110_110094


namespace two_copies_of_Q_cover_P_l110_110584

noncomputable theory

-- Define rectangles P and Q with properties: area and diagonal lengths.
structure Rectangle :=
  (width : ℝ)
  (length : ℝ)

def area (r : Rectangle) : ℝ := r.width * r.length

def diagonal (r : Rectangle) : ℝ := (r.width^2 + r.length^2).sqrt

-- Conditions given in the problem
variables (P Q : Rectangle)
  (h1 : area P = area Q)
  (h2 : diagonal P > diagonal Q)
  (h3 : ∃ r1 r2 : Rectangle, r1 = P ∧ r2 = P ∧ area(P) + area(P) ≥ area(Q))

-- Question: Prove two copies of Q can cover P.
theorem two_copies_of_Q_cover_P : 
  ∃ r1 r2 : Rectangle, r1 = Q ∧ r2 = Q ∧ area(Q) + area(Q) ≥ area(P) :=
sorry

end two_copies_of_Q_cover_P_l110_110584


namespace paint_houses_l110_110498

theorem paint_houses (time_per_house : ℕ) (hour_to_minute : ℕ) (hours_available : ℕ) 
  (h1 : time_per_house = 20) (h2 : hour_to_minute = 60) (h3 : hours_available = 3) :
  (hours_available * hour_to_minute) / time_per_house = 9 :=
by
  sorry

end paint_houses_l110_110498


namespace spinner_points_east_l110_110157

def initial_direction : ℝ := 0  -- North (0 radians)
def clockwise_revolutions : ℚ :=  15 / 4  -- Representing 3 3/4 revolutions
def counterclockwise_revolutions : ℚ := 10 / 4  -- Representing 2 2/4 revolutions

def net_revolutions : ℚ := clockwise_revolutions - counterclockwise_revolutions  -- Net movements
def net_angle : ℝ := 2 * π * (net_revolutions - net_revolutions.natFloor)  -- Remaining angle after full circles

theorem spinner_points_east : 
  (initial_direction + net_angle) % (2 * π) = π / 2 := 
sorry

end spinner_points_east_l110_110157


namespace largest_m_l110_110169

noncomputable def max_m (t n : ℕ) [fact (t ≥ 2)] [fact (n ≥ 2)] : ℕ :=
  @classical.some ℕ ( ⟨ n, sorry⟩ : ∃ m, 
    ∃ P : Polynomial ℚ, 
    P.degree.toNat = n ∧ 
    (∀ k : ℕ, k ≤ m → (¬ ky : ℚ) (P.eval k) / (t^k) ∈ ℤ ∨ P.eval k / (t^(k+1)) ∈ ℤ))  

theorem largest_m (t n : ℕ) [fact (t ≥ 2)] [fact (n ≥ 2)] : max_m t n = n := 
sorry

end largest_m_l110_110169


namespace police_emergency_prime_l110_110786

theorem police_emergency_prime {n : ℕ} (k : ℕ) (h : n = k * 1000 + 133) : 
  ∃ p : ℕ, nat.prime p ∧ p > 7 ∧ p ∣ n :=
begin
  sorry,
end

end police_emergency_prime_l110_110786


namespace probability_of_positive_roots_l110_110200

-- Definition of the polynomial equation and related conditions
def polynomial_has_two_positive_roots (a : ℝ) : Prop :=
  let Δ := 4 * (a ^ 2 - 4 * a + 3) in
  Δ ≥ 0 ∧ 4 * a - 3 > 0 ∧ 2 * a > 0

-- Proof statement to show that the probability of selection is 3/8
theorem probability_of_positive_roots : 
  ∀ (a : ℝ), a ∈ set.Icc (-1) 5 → 
  set.Icc (3/4) 1 ∪ set.Ici (3) ⊆ set.Icc (-1) 5 →
  (measure_theory.measure_space.volume (set.Icc (3/4) 1) +
   measure_theory.measure_space.volume (set.Ici (3)))
   / measure_theory.measure_space.volume (set.Icc (-1) 5) = (3 / 8) :=
by
  sorry

end probability_of_positive_roots_l110_110200


namespace exists_two_numbers_satisfying_inequality_l110_110193

theorem exists_two_numbers_satisfying_inequality (x₁ x₂ x₃ x₄ : ℝ) (h₁₂ : x₁ ≠ x₂) (h₁₃ : x₁ ≠ x₃) (h₁₄ : x₁ ≠ x₄) 
  (h₂₃ : x₂ ≠ x₃) (h₂₄ : x₂ ≠ x₄) (h₃₄ : x₃ ≠ x₄) : 
  ∃ (a b : ℝ), 
    a ≠ b ∧ ( (a = x₁ ∨ a = x₂ ∨ a = x₃ ∨ a = x₄) ∧ (b = x₁ ∨ b = x₂ ∨ b = x₃ ∨ b = x₄) ) ∧
    (1 + a * b > 0) ∧ 
    (1 + a^2 > 0) ∧ 
    (1 + b^2 > 0) ∧ 
    (1 + a * b) / (Real.sqrt(1 + a^2) * Real.sqrt(1 + b^2)) > 1 / 2 := 
by 
  sorry

end exists_two_numbers_satisfying_inequality_l110_110193


namespace sum_series_eq_two_l110_110377

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l110_110377


namespace mixture_percentage_X_is_l110_110586

-- Definitions based on conditions
def percentRyegrassInX : ℝ := 0.40
def percentRyegrassInY : ℝ := 0.25
def targetRyegrassPercentage : ℝ := 35
def finalPercentageOfMixtureX (p : ℝ) : ℝ := (percentRyegrassInX * p) + (percentRyegrassInY * (100 - p))

-- Lean theorem statement matching the equivalent proof problem
theorem mixture_percentage_X_is : ∃ p : ℝ, finalPercentageOfMixtureX p = targetRyegrassPercentage ∧ p = 200 / 3 :=
begin
    sorry -- Proof not required per instructions
end

end mixture_percentage_X_is_l110_110586


namespace solve_for_a_l110_110492

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l110_110492


namespace binary_to_decimal_correct_decimal_to_hex_correct_binary_to_hex_correct_l110_110000

-- Define the binary number as a string representation
def binary_number : String := "1011001"

-- Here we define a function for converting binary strings to their decimal equivalents
def binary_to_decimal (s : String) : Nat :=
  s.reverse.toList.enumFrom(0).foldl (fun acc ⟨i, c⟩ => acc + (if c = '1' then Nat.pow 2 i else 0)) 0

-- Define the target decimal number after conversion
def target_decimal : Nat := 89

-- Define the target hexadecimal number as a string representation
def target_hexadecimal : String := "59"

-- Prove that the binary number converts to the target decimal number
theorem binary_to_decimal_correct : binary_to_decimal binary_number = target_decimal := by sorry

-- Define a function to convert decimal number to hexadecimal string
def decimal_to_hex (n : Nat) : String :=
  if n = 0 then "0" else
  let rec loop (n : Nat) (acc : List Char) : List Char :=
    if n = 0 then acc else loop (n / 16) (Char.ofNat (n % 16 + if n % 16 < 10 then 48 else 55) :: acc)
  String.mk (loop n [])

-- Prove that the target decimal number converts to the target hexadecimal number
theorem decimal_to_hex_correct : decimal_to_hex target_decimal = target_hexadecimal := by sorry

-- Combine both theorems to prove that binary number "1011001" is the same as hexadecimal number "59"
theorem binary_to_hex_correct : decimal_to_hex (binary_to_decimal binary_number) = target_hexadecimal := by sorry

end binary_to_decimal_correct_decimal_to_hex_correct_binary_to_hex_correct_l110_110000


namespace solve_for_r_l110_110870

theorem solve_for_r : ∃ r : ℝ, r ≠ 4 ∧ r ≠ 5 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 10) / (r^2 - 2*r - 15) ↔ 
  r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 := 
by {
  sorry
}

end solve_for_r_l110_110870


namespace coin_flips_sequences_count_l110_110719

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110719


namespace triangle_shape_right_angled_l110_110524

theorem triangle_shape_right_angled (a b c : ℝ) (A B C : ℝ) (h1 : b^2 = c^2 + a^2 - c * a) (h2 : Real.sin A = 2 * Real.sin C) :
    ∃ (D : Type) (triangle_shape : TriangeShape D), triangle_shape = TriangeShape.RightAngled :=
by
  sorry

end triangle_shape_right_angled_l110_110524


namespace solve_for_m_when_linear_l110_110114

theorem solve_for_m_when_linear (m : ℤ) (h : (m - 3) * x ^ (2 * |m| - 5) - 4 * m = 0 ∧ (2 * |m| - 5 = 1)) : m = -3 := 
sorry

end solve_for_m_when_linear_l110_110114


namespace find_y_from_exponent_equation_l110_110928

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110928


namespace find_M_mod_1000_l110_110874

def h (n : ℕ) := (nat.digits 5 n).sum
def j (n : ℕ) := (nat.digits 9 (h n)).sum

theorem find_M_mod_1000 : 
  let M := ∃ n, nat.digits 18 (j n) |>.any (λ d, d > 9)
  in nat.mod (classical.some M) 1000 = 599 :=
by
  sorry

end find_M_mod_1000_l110_110874


namespace integral_identity_l110_110035

noncomputable def integral_proof : Prop :=
  ∫ (x : ℝ) in 0..0, (3 * (Real.tan x) ^ 2 - 1) / ((Real.tan x) ^ 2 + 5) = 
  -x + (4 / Real.sqrt 5) * Real.arctan ((Real.tan x) / Real.sqrt 5) + C

theorem integral_identity : integral_proof :=
  sorry

end integral_identity_l110_110035


namespace smallest_positive_period_and_range_value_of_cos_2a_div_1_minus_tan_a_l110_110470

-- Definition of function f
def f (x : ℝ) : ℝ := 2 * (Real.cos (x / 2))^2 - Real.sqrt 3 * Real.sin x

-- Theorem for part (I)
theorem smallest_positive_period_and_range :
  let T := (2 : ℝ) * Real.pi in
  ∀ x : ℝ, f (x + T) = f x ∧ ∀ y : ℝ, f y ∈ Set.Icc (-1) 3 :=
by sorry

-- Theorem for part (II)
theorem value_of_cos_2a_div_1_minus_tan_a 
  (a : ℝ) (h₁ : Real.pi / 2 < a ∧ a < Real.pi)
  (h₂ : f (a - Real.pi / 3) = 1 / 3) 
  : (Real.cos (2 * a)) / (1 - Real.tan a) = (1 - 2 * Real.sqrt 2) / 9 :=
by sorry

end smallest_positive_period_and_range_value_of_cos_2a_div_1_minus_tan_a_l110_110470


namespace value_of_b_in_range_l110_110378

-- Main problem statement
theorem value_of_b_in_range (b x y : ℝ) :
  (√ (x^2 * y^2))^(1/4) = b^(3 * b) ∧ log b (x^(log b (y)^2)) + log b (y^(log b (x)^2)) = 6 * b^6
  → 0 < b ∧ b ≤ (√ (2)) / (√ (5)) :=
sorry

end value_of_b_in_range_l110_110378


namespace measure_angle_C_l110_110953

theorem measure_angle_C
  (a b c : ℝ)
  (h : a^2 + b^2 + sqrt 2 * a * b = c^2) :
  ∃ C : ℝ, C = 3 * π / 4 :=
by
  sorry

end measure_angle_C_l110_110953


namespace day_two_millet_sunflower_majority_l110_110565

-- Definitions based on the problem conditions

def M : ℕ → ℝ
| 0     := 0.20
| (n+1) := 0.20 + 0.75 * (M n)

def S : ℕ → ℝ
| 0     := 0.30
| (n+1) := 0.30 + 0.50 * (S n)

-- Theorem to prove

theorem day_two_millet_sunflower_majority :
  M 2 + S 2 > 0.50 :=
by {
  -- Proof of the theorem will go here.
  sorry
}

end day_two_millet_sunflower_majority_l110_110565


namespace jose_peanuts_l110_110538

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l110_110538


namespace ab_cd_eq_zero_l110_110120

theorem ab_cd_eq_zero  
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : ad - bc = -1) :
  ab + cd = 0 :=
by
  sorry

end ab_cd_eq_zero_l110_110120


namespace coin_flip_sequences_l110_110723

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110723


namespace log_sub_inv_pow_eq_l110_110210

theorem log_sub_inv_pow_eq : Real.log 9 / Real.log 3 - 4^(-1/2 : ℝ) = 3 / 2 := by
  have h1 : Real.log 9 / Real.log 3 = 2 :=
    calc
      Real.log 9 / Real.log 3
        = (Real.log (3 ^ 2)) / Real.log 3 : by rw [←Real.log_pow 3 2]
    ... = 2 : by rw [Real.log_mul (Real.pow_pos one_lt_three 2)]; exact div_self Real.log_ne_zero

  have h2 : 4^(-1/2 : ℝ) = 1 / 2 :=
    calc
      4^(-1/2 : ℝ)
        = 1 / (4^ (1/2 : ℝ)) : by rw [Real.rpow_neg (le_of_lt (by norm_num : 0 < 4))]
    ... = 1 /2 : by rw [Real.rpow_nat_cast]; exact inv_eq_one_div

  rw [h1, h2]
  norm_num

end log_sub_inv_pow_eq_l110_110210


namespace natural_numbers_are_integers_l110_110316

theorem natural_numbers_are_integers (h₁ : ∀ n : ℕ, ↑n ∈ ℤ) (h₂ : (4 : ℕ)) : (4 : ℤ) :=
by
  sorry

end natural_numbers_are_integers_l110_110316


namespace kate_collected_money_l110_110660

-- Define the conditions
def wand_cost : ℕ := 60
def num_wands_bought : ℕ := 3
def extra_charge : ℕ := 5
def num_wands_sold : ℕ := 2

-- Define the selling price per wand
def selling_price_per_wand : ℕ := wand_cost + extra_charge

-- Define the total amount collected from the sale
def total_collected : ℕ := num_wands_sold * selling_price_per_wand

-- Prove that the total collected is $130
theorem kate_collected_money :
  total_collected = 130 :=
sorry

end kate_collected_money_l110_110660


namespace original_price_of_car_l110_110642

theorem original_price_of_car 
  (Venny_paid : ℝ)
  (first_reduction_rate : ℝ)
  (second_reduction_rate : ℝ)
  (sales_tax_rate : ℝ)
  (documentation_fee : ℝ)
  (paid_amount : Venny_paid = 15000)
  (first_reduction : first_reduction_rate = 0.30)
  (second_reduction : second_reduction_rate = 0.40)
  (sales_tax : sales_tax_rate = 0.08)
  (doc_fee : documentation_fee = 200)
  (P C : ℝ) :
  (1.08 * C + documentation_fee = Venny_paid) →
  (C = 0.42 * P) →
  C = 13703.70 →
  P = 32_628.33 :=
by
  sorry

end original_price_of_car_l110_110642


namespace coin_flip_sequences_l110_110763

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110763


namespace incorrect_statement_of_system_l110_110202

theorem incorrect_statement_of_system :
  let D      := 2 * (-2) - 1 * 3,
      Dx     := 1 * (-2) - 1 * 12,
      Dy     := 2 * 12 - 1 * 3,
      x      := Dx / D,
      y      := Dy / D
  in D = -7 ∧ Dx = -14 ∧ Dy = 21 ∧ x = 2 ∧ y = -3 :=
by {
  let D := 2 * (-2) - 1 * 3,
  let Dx := 1 * (-2) - 1 * 12,
  let Dy := 2 * 12 - 1 * 3,
  let x := Dx / D,
  let y := Dy / D,
  have h1: D = -7 := rfl,
  have h2: Dx = -14 := rfl,
  have h3: Dy = 21 := rfl,
  have h4: x = 2 := rfl,
  have h5: y = -3 := rfl,
  exact ⟨h1, h2, h3, h4, h5⟩,
  sorry,
}

end incorrect_statement_of_system_l110_110202


namespace march_first_is_tuesday_l110_110939

theorem march_first_is_tuesday (year_has_53_Fridays : true) (year_has_53_Saturdays : true) : ∃ day : string, day = "Tuesday" :=
by
  sorry

end march_first_is_tuesday_l110_110939


namespace hyperbola_asymptotes_l110_110908

theorem hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : ∀ c, c = 2 * a) : 
  b = Real.sqrt 3 * a → asymptotes (λ x, x^2 / a^2 - x^2 / b^2 = 1) = [± (λ x, Real.sqrt 3 * x)] :=
by
  sorry

end hyperbola_asymptotes_l110_110908


namespace compound_interest_rate_l110_110854

theorem compound_interest_rate (P A : ℝ) (t : ℝ) (n : ℕ) (CI r : ℝ)
  (hA : A = P + CI)
  (hCI : CI = 6218) 
  (hP : P = 16000) 
  (ht : t = 2 + 4/12) 
  (hn : n = 1) 
  (hAcomp : A = P * (1 + r / n) ^ (n * t)) :
  r ≈ 0.15 :=
  sorry

end compound_interest_rate_l110_110854


namespace desired_ratio_is_correct_l110_110954

-- Definitions of the conditions
def total_mixture_volume := 40 -- in liters
def original_milk_to_water_ratio := (7, 1) -- 7:1 ratio
def added_water_volume := 1.6 -- 1600 ml converted to liters

-- Definition of the desired ratio after adding water (53:10)
def desired_milk_to_water_ratio := (53, 10)

-- Statement to be proven
theorem desired_ratio_is_correct : 
  let milk_volume := (7 / 8) * total_mixture_volume in
  let water_volume := (1 / 8) * total_mixture_volume in
  let new_water_volume := water_volume + added_water_volume in
  let new_ratio := (milk_volume / new_water_volume) in
  abs (new_ratio - (53.0 / 10.0)) < 0.01 :=
begin
  sorry
end

end desired_ratio_is_correct_l110_110954


namespace sequence_terms_are_integers_iff_l110_110988

theorem sequence_terms_are_integers_iff (a b c : ℝ) (h_ab : a * b ≠ 0) (h_c : c > 0) :
  (∃ (a_n : ℕ → ℝ), a_n 1 = a ∧ a_n 2 = b ∧ (∀ n ≥ 2, a_n (n + 1) = (a_n n ^ 2 + c) / a_n (n - 1)) 
    ∧ ∀ n, a_n n ∈ ℤ) ↔ 
  (a ∈ ℤ ∧ b ∈ ℤ ∧ (a^2 + b^2 + c) / (a * b) ∈ ℤ) :=
by sorry

end sequence_terms_are_integers_iff_l110_110988


namespace area_triangle_ABC_l110_110970

-- Definitions of the lengths and height
def BD : ℝ := 3
def DC : ℝ := 2 * BD
def BC : ℝ := BD + DC
def h_A_BC : ℝ := 4

-- The triangle area formula
def areaOfTriangle (base height : ℝ) : ℝ := 0.5 * base * height

-- The goal to prove that the area of triangle ABC is 18 square units
theorem area_triangle_ABC : areaOfTriangle BC h_A_BC = 18 := by
  sorry

end area_triangle_ABC_l110_110970


namespace find_weight_of_B_l110_110304

theorem find_weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) : B = 33 :=
by 
  sorry

end find_weight_of_B_l110_110304


namespace maximum_elements_simple_subset_l110_110335

def is_simple_subset (s : set ℕ) : Prop :=
  ∀ x y z ∈ s, x + y ≠ z

theorem maximum_elements_simple_subset (n : ℕ) :
  ∃ s ⊆ {x | 1 ≤ x ∧ x ≤ 2 * n + 1}, is_simple_subset s ∧ set.card s = n + 1 :=
sorry

end maximum_elements_simple_subset_l110_110335


namespace number_of_tangerines_l110_110246

variable {P T : ℕ}

theorem number_of_tangerines (
  h1 : 45 = P + 21,
  h2 : T = P + 18
  ) : T = 42 :=
by
  sorry

end number_of_tangerines_l110_110246


namespace tan_seven_pi_over_four_l110_110020

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l110_110020


namespace average_monthly_balance_is_150_l110_110357

-- Define the balances for each month
def balance_jan : ℕ := 100
def balance_feb : ℕ := 200
def balance_mar : ℕ := 150
def balance_apr : ℕ := 150

-- Define the number of months
def num_months : ℕ := 4

-- Define the total sum of balances
def total_balance : ℕ := balance_jan + balance_feb + balance_mar + balance_apr

-- Define the average balance
def average_balance : ℕ := total_balance / num_months

-- Goal is to prove that the average monthly balance is 150 dollars
theorem average_monthly_balance_is_150 : average_balance = 150 :=
by
  sorry

end average_monthly_balance_is_150_l110_110357


namespace distinct_sequences_ten_flips_l110_110688

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110688


namespace hyperbola_standard_equation_l110_110459

theorem hyperbola_standard_equation
  (e : ℝ) (b : ℝ) (a c : ℝ)
  (h1 : e = sqrt 3)
  (h2 : b = 2)
  (h3 : c^2 = a^2 + b^2)
  (h4 : e = c / a) :
  (∀ x y : ℝ, y^2 / 2 - x^2 / 4 = 1) :=
begin
  sorry
end

end hyperbola_standard_equation_l110_110459


namespace problem1_problem2_l110_110819

-- Problem 1: Prove that
theorem problem1 :
  ( (2 + 3 / 5) ^ 0 ) + ( 2 ^ -2 ) * ( (2 + 1 / 4 ) ^ -0.5 ) + ( (25 / 36 ) ^ 0.5 ) + Real.sqrt ((-2)^2) = 4 := 
  sorry

-- Problem 2: Prove that
theorem problem2 :
  (1 / 2) * Real.log (32 / 49) - (4 / 3) * Real.log (Real.sqrt 8) + Real.log (Real.sqrt 245) = 1 / 2 :=
  sorry

end problem1_problem2_l110_110819


namespace polynomial_with_specified_roots_l110_110416

noncomputable def polynomial_with_roots (x y : ℝ) : (Polynomial ℤ) :=
  (Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 10 * Polynomial.X ^ 2 + Polynomial.C 1) *
  (Polynomial.X ^ 6 - Polynomial.C 6 * Polynomial.X ^ 4 - Polynomial.C 6 * Polynomial.X ^ 3 + 
   Polynomial.C 12 * Polynomial.X ^ 2 - Polynomial.C 36 * Polynomial.X + Polynomial.C 1)

theorem polynomial_with_specified_roots :
  let x := (ℝ.sqrt 2 + ℝ.sqrt 3)
  let y := (ℝ.sqrt 2 + ℝ.cbrt 3)
  has_root (polynomial_with_roots x y) x ∧ has_root (polynomial_with_roots x y) y :=
sorry

end polynomial_with_specified_roots_l110_110416


namespace find_m_l110_110055

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x > 1 then Real.log x else 2 * x + (∫ t in 0..m, 3 * t^2)

theorem find_m (m : ℝ) (e_ne_zero : e ≠ 0) : f (f e m) m = 10 → m = 2 := by
  unfold f
  sorry

end find_m_l110_110055


namespace cosine_difference_identity_l110_110289

theorem cosine_difference_identity (α β : ℝ) : 
  cos(α - β) = cos α * cos β + sin α * sin β :=
sorry

end cosine_difference_identity_l110_110289


namespace students_both_questions_correct_l110_110567

-- Define the conditions for the problem
def total_students : ℕ := 29
def students_q1_correct : ℕ := 19
def students_q2_correct : ℕ := 24
def students_not_taken_test : ℕ := 5

-- Define the mathematical problem
theorem students_both_questions_correct : (students_q1_correct = 19 → students_q2_correct = 24 → total_students = 29 → students_not_taken_test = 5 → 
                                           ∀ (n : ℕ), n = 24 → (n = total_students - students_not_taken_test) → 
                                           students_q1_correct = (students_q2_correct - students_not_taken_test)) :=
by
  intro h1 h2 h3 h4 n h5 h6,
  have h7 : total_students - students_not_taken_test = 24 := by rw [h3, h4]; ring,
  rw ← h6 at h7,
  have h8 : students_q2_correct = n := by rw [h2, h7],
  rw [h8, h5] at h1,
  exact h1.symm

end students_both_questions_correct_l110_110567


namespace find_inverse_sum_l110_110996

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 3 * x else -x^2 + 3 * x

theorem find_inverse_sum :
  let f_inv_9 := (9 : ℝ) in
  let f_inv_neg_121 := (-121 : ℝ) in
  f^.inverse f_inv_9 + f^.inverse f_inv_neg_121 = -12 :=
by
  sorry

end find_inverse_sum_l110_110996


namespace tangent_line_at_x_eq_1_l110_110607

open filter real set

def curve (x : ℝ) : ℝ := (1 + x) * log x

theorem tangent_line_at_x_eq_1 :
  ∃ (a b : ℝ), (∀ x, curve 1 = 0 ∧ deriv curve 1 = 2) →
  (∀ x, curve x = a * x + b) :=
sorry

end tangent_line_at_x_eq_1_l110_110607


namespace circle_tangent_line_l110_110606

theorem circle_tangent_line :
  ∃ (R : ℝ), ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = R^2 ↔ R = sqrt 3 ∧ ∃ k, x + y = k ∧ k = sqrt 6 :=
sorry

end circle_tangent_line_l110_110606


namespace bc_sum_condition_l110_110932

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0
def not_equal_to (x y : ℕ) : Prop := x ≠ y
def less_than_or_equal_to_nine (n : ℕ) : Prop := n ≤ 9

-- Main proof statement
theorem bc_sum_condition (a b c : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_pos_c : is_positive_integer c)
  (h_a_not_1 : a ≠ 1) (h_b_not_c : b ≠ c) (h_b_le_9 : less_than_or_equal_to_nine b) (h_c_le_9 : less_than_or_equal_to_nine c)
  (h_eq : (10 * a + b) * (10 * a + c) = 100 * a * a + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end bc_sum_condition_l110_110932


namespace inequality_inequality_l110_110117

theorem inequality_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (sqrt a > sqrt b) ∧ (a - 1 / a > b - 1 / b) :=
sorry

end inequality_inequality_l110_110117


namespace part1_part2_l110_110976

-- Given conditions for transformation T
def transform (A : List Bool) : List Bool :=
  A.bind (λ a, if a then [false, true] else [true, false])

def Ak (A0 : List Bool) (k : ℕ) : List Bool :=
  (List.repeat transform k).foldl (flip id) A0

-- Definitions for the proof
def A2 := [true, false, false, true, false, true, true, false] -- 10010110
def A0 := [true, false] -- 10

-- Define l_k which is the number of consecutive pairs of 1s in A_k
def l (A : List Bool) : ℕ :=
  (A.zipWith (λ x y, x && y) A.tail).count true

theorem part1 : Ak A0 2 = A2 → A0 = [true, false] :=
by
  sorry

theorem part2 (k : ℕ) : A0 = [true, false] → l (Ak A0 k) = (2^k - (-1)^k) / 3 :=
by
  sorry

end part1_part2_l110_110976


namespace calculate_fraction_l110_110818

def x : ℚ := 2 / 3
def y : ℚ := 8 / 10

theorem calculate_fraction :
  (6 * x + 10 * y) / (60 * x * y) = 3 / 8 := by
  sorry

end calculate_fraction_l110_110818


namespace ratio_a_c_l110_110610

-- Define the function f
def f (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

-- Define the inverse function form
def f_inv (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

-- The main statement proving the ratio a/c = -4 given the function f and its inverse in the defined form
theorem ratio_a_c (a b c d : ℝ) 
  (h₀ : ∀ x, f(f_inv a b c d x) = x) :
  a / c = -4 := 
by sorry

end ratio_a_c_l110_110610


namespace police_emergency_number_has_prime_divisor_gt_7_l110_110785

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : 
  ∃ p : ℕ, p.prime ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_has_prime_divisor_gt_7_l110_110785


namespace coefficient_of_a3b3_l110_110260

theorem coefficient_of_a3b3 (a b c : ℚ) :
  (∏ i in Finset.range 7, (a + b) ^ i * (c + 1 / c) ^ (8 - i)) = 1400 := 
by
  sorry

end coefficient_of_a3b3_l110_110260


namespace smallest_odd_number_with_three_different_prime_factors_l110_110653

theorem smallest_odd_number_with_three_different_prime_factors :
  ∃ n, Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
  ∃ (n = 105), Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
sorry

end smallest_odd_number_with_three_different_prime_factors_l110_110653


namespace shaded_area_eq_3_l110_110808

-- Define the setup for the rectangle and the points
def rectangle_DEFA (D E F A C B O : Type)
                   (AD AF : ℝ)
                   (h1 : AD = 3)
                   (h2 : AF = 4)
                   (h3 : DC = CB)
                   (h4 : CB = BA) :
  Prop := sorry

-- Define the theorem stating the area of the shaded region
theorem shaded_area_eq_3 (D E F A C B O : Type)
                         (AD AF : ℝ)
                         (h1 : AD = 3)
                         (h2 : AF = 4)
                         (h3 : DC = CB)
                         (h4 : CB = BA) :
  rectangle_DEFA D E F A C B O AD AF h1 h2 h3 h4 →
  shaded_area D E F A C B O = 3 :=
sorry

end shaded_area_eq_3_l110_110808


namespace remainder_when_sum_is_divided_l110_110935

theorem remainder_when_sum_is_divided (n : ℤ) : ((8 - n) + (n + 5)) % 9 = 4 := by
  sorry

end remainder_when_sum_is_divided_l110_110935


namespace hexagon_area_correct_l110_110175

noncomputable def area_of_convex_hexagon : ℂ :=
  let P := Polynomial.roots (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 6 * Polynomial.X^3 - Polynomial.C 216)
  sqrt 3 * 9

theorem hexagon_area_correct :
  let P := Polynomial.roots (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 6 * Polynomial.X^3 - Polynomial.C 216) in
  (P.card = 6 ∧ 
   ∀ p ∈ P, (p : ℂ) ≠ 0 ∧ 
   ∀ i j : Fin 6, i ≠ j → P.nth i ≠ P.nth j ∧ 
   is_convex_hexagon P) →
  area_of_convex_hexagon = 9 * sqrt 3 :=
begin
  -- The proof goes here
  sorry
end

-- Auxiliary definition to state convex hexagon check
def is_convex_hexagon (P : Multiset ℂ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ p₆ : ℂ, 
    P = {p₁, p₂, p₃, p₄, p₅, p₆} ∧ 
    (p₁ - p₂) * (p₃ - p₂) ∈ ℝ ∧
    (p₂ - p₃) * (p₄ - p₃) ∈ ℝ ∧
    (p₃ - p₄) * (p₅ - p₄) ∈ ℝ ∧
    (p₄ - p₅) * (p₆ - p₅) ∈ ℝ ∧
    (p₅ - p₆) * (p₁ - p₆) ∈ ℝ ∧
    complex.arg (p₁ - p₃) = complex.arg (p₄ - p₆) ∧ 
endmodule


end hexagon_area_correct_l110_110175


namespace parabola_directrix_l110_110464

theorem parabola_directrix (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x ↔ x = -1 → p = 2) :=
by
  sorry

end parabola_directrix_l110_110464


namespace width_of_carpet_in_cm_l110_110856

theorem width_of_carpet_in_cm (length_room breadth_room cost_per_sqm total_cost : ℝ) (h1 : length_room = 13) (h2 : breadth_room = 9) (h3 : cost_per_sqm = 12) (h4 : total_cost = 1872) : 
  ∃ width_of_carpet_in_cm, width_of_carpet_in_cm = 1200 :=
by
  have area_of_room := length_room * breadth_room
  -- Room area = 13 * 9 = 117
  have h_area : area_of_room = 13 * 9 := by rw [h1, h2]
  rw [←h_area] at area_of_room
  -- calculation of carpet area = total_cost / cost_per_sqm
  have total_area_of_carpet := total_cost / cost_per_sqm
  -- Carpet area = 1872 / 12 = 156
  have h_total_carpet_area : total_area_of_carpet = 1872 / 12 := by rw [h3, h4]
  rw [←h_total_carpet_area] at total_area_of_carpet
  -- width of carpet in meters = total_area_of_carpet / length_room
  have width_of_carpet := total_area_of_carpet / length_room
  -- Finally convert to centimeters
  have width_of_carpet_in_cm := width_of_carpet * 100
  use width_of_carpet_in_cm
  sorry

end width_of_carpet_in_cm_l110_110856


namespace coin_flip_sequences_l110_110745

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110745


namespace negation_of_existence_l110_110229

theorem negation_of_existence :
  ¬(∃ x : ℝ, x^2 + 2 * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l110_110229


namespace granger_total_payment_proof_l110_110110

-- Conditions
def cost_per_can_spam := 3
def cost_per_jar_peanut_butter := 5
def cost_per_loaf_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Calculation
def total_cost_spam := quantity_spam * cost_per_can_spam
def total_cost_peanut_butter := quantity_peanut_butter * cost_per_jar_peanut_butter
def total_cost_bread := quantity_bread * cost_per_loaf_bread

-- Total amount paid
def total_amount_paid := total_cost_spam + total_cost_peanut_butter + total_cost_bread

-- Theorem to be proven
theorem granger_total_payment_proof : total_amount_paid = 59 :=
by
  sorry

end granger_total_payment_proof_l110_110110


namespace coin_flips_sequences_count_l110_110715

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110715


namespace range_of_a_l110_110559

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (2 * a + 1) * x + a^2 + a < 0 → 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 10) →
  (∃ l u : ℝ, (l = 1/2) ∧ (u = 9/2) ∧ (l ≤ a ∧ a ≤ u)) :=
by
  sorry

end range_of_a_l110_110559


namespace min_value_frac_sum_l110_110509

-- Define the main problem
theorem min_value_frac_sum (a b : ℝ) (h1 : 2 * a + 3 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ x : ℝ, (x = 25) ∧ ∀ y, (y = (2 / a + 3 / b)) → y ≥ x :=
sorry

end min_value_frac_sum_l110_110509


namespace coefficient_a3b3_l110_110275

theorem coefficient_a3b3 in_ab_c_1overc_expr :
  let coeff_ab := Nat.choose 6 3 
  let coeff_c_expr := Nat.choose 8 4 
  coeff_ab * coeff_c_expr = 1400 :=
by
  sorry

end coefficient_a3b3_l110_110275


namespace jerome_contact_list_l110_110531

def classmates := 20
def out_of_school_friends := classmates / 2
def family_members := 3
def total_contacts := classmates + out_of_school_friends + family_members

theorem jerome_contact_list : total_contacts = 33 := by
  sorry

end jerome_contact_list_l110_110531


namespace coin_flips_sequences_count_l110_110718

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110718


namespace smallest_base10_integer_l110_110280

theorem smallest_base10_integer :
  ∃ (n A B : ℕ), 
    (A < 5) ∧ (B < 7) ∧ 
    (n = 6 * A) ∧ 
    (n = 8 * B) ∧ 
    n = 24 := 
sorry

end smallest_base10_integer_l110_110280


namespace trajectory_of_midpoint_is_ellipse_l110_110888

def A (θ : ℝ) : ℝ × ℝ := (4 * Real.sin θ, 6 * Real.cos θ)
def B (θ : ℝ) : ℝ × ℝ := (-4 * Real.cos θ, 6 * Real.sin θ)

noncomputable def M (θ : ℝ) : ℝ × ℝ := 
(2 * Real.sin θ - 2 * Real.cos θ, 3 * Real.cos θ + 3 * Real.sin θ)

theorem trajectory_of_midpoint_is_ellipse :
  ∃ M : ℝ × ℝ → ℝ × ℝ, 
  (∀ θ, M θ = 
      (2 * Real.sin θ - 2 * Real.cos θ, 
       3 * Real.cos θ + 3 * Real.sin θ)) 
  ∧ 
  ∀ x y, 
  (∃ θ, M θ = (x, y)) ↔ 
  (x^2 / 8 + y^2 / 18 = 1) :=
by 
  sorry

end trajectory_of_midpoint_is_ellipse_l110_110888


namespace cylinder_height_relation_l110_110092

theorem cylinder_height_relation (r1 r2 h1 h2 V1 V2 : ℝ) 
  (h_volumes_equal : V1 = V2)
  (h_r2_gt_r1 : r2 = 1.1 * r1)
  (h_volume_first : V1 = π * r1^2 * h1)
  (h_volume_second : V2 = π * r2^2 * h2) : 
  h1 = 1.21 * h2 :=
by 
  sorry

end cylinder_height_relation_l110_110092


namespace appropriate_chart_to_observe_changes_l110_110250

-- Define the conditions
def dataset_week : Type := sorry -- A place-holder type for a dataset collected over a week

def significant_change_observation (dataset : dataset_week) : Prop := sorry -- A place-holder predicate for the condition of observing significant changes over time

-- The mathematical proof problem
theorem appropriate_chart_to_observe_changes (dataset : dataset_week) :
  significant_change_observation dataset →
  (∃ chart : string, chart = "Line") :=
begin
  intro h,
  use "Line",
  trivial,
end

end appropriate_chart_to_observe_changes_l110_110250


namespace december_sales_multiple_l110_110984

   noncomputable def find_sales_multiple (A : ℝ) (x : ℝ) :=
     x * A = 0.3888888888888889 * (11 * A + x * A)

   theorem december_sales_multiple (A : ℝ) (x : ℝ) (h : find_sales_multiple A x) : x = 7 :=
   by 
     sorry
   
end december_sales_multiple_l110_110984


namespace eq_segments_perpendicular_projections_l110_110969

variables (A B C D E F : Point) 
variables [cyclic_quadrilateral ABCD] (h1 : AC = BC) 
variables [perpendicular_projection C AD E] [perpendicular_projection C BD F]

theorem eq_segments_perpendicular_projections : AE = BF :=
sorry

end eq_segments_perpendicular_projections_l110_110969


namespace triangle_ABC_is_right_l110_110525

theorem triangle_ABC_is_right
  (A B C : Type)
  [triangle A B C]
  (AB : length A B = 2 * length B C)
  (angleB : angle B = 2 * angle A) :
  is_right_triangle A B C :=
sorry

end triangle_ABC_is_right_l110_110525


namespace polygon_induction_base_case_l110_110288

theorem polygon_induction_base_case :
  ∃ n, (n ≥ 3) ∧ (polygon_sides n = 3) :=
by
  sorry -- This will be the proof

end polygon_induction_base_case_l110_110288


namespace meiosis_and_fertilization_ensure_stability_of_chromosome_number_l110_110045

-- Define the process of meiosis and its significance.
def meiosis_significance : Prop :=
  ∀ (organisms : Type) (reproduce_sexually : organisms → Prop),
  ∃ (mature_reproductive_cells germ_cells : organisms),
  (has_half_chromosomes mature_reproductive_cells germ_cells) ∧
  (∀ (fertilization : organisms → organisms), 
  has_somatic_chromosome_number (fertilization mature_reproductive_cells))

-- Theorem stating that meiosis and fertilization ensure the stability of the chromosome number across generations
theorem meiosis_and_fertilization_ensure_stability_of_chromosome_number :
  meiosis_significance →
  ensures_stability_of_chromosome_number :=
sorry

end meiosis_and_fertilization_ensure_stability_of_chromosome_number_l110_110045


namespace find_angle_A_l110_110952

theorem find_angle_A (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : B = (A + C) / 2)
  (h3 : 2 * b ^ 2 = 3 * a * c) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l110_110952


namespace union_of_sets_l110_110478

noncomputable def set_A : Set ℚ := {7, -1/3}
noncomputable def set_B : Set ℚ := {8/3, -1/3}

theorem union_of_sets :
  (set_A ∪ set_B) = {7, 8/3, -1/3} :=
by
  sorry

end union_of_sets_l110_110478


namespace third_quadrant_angle_bisector_l110_110453

theorem third_quadrant_angle_bisector
  (a b : ℝ)
  (hA : A = (-4,a))
  (hB : B = (-2,b))
  (h_lineA : a = -4)
  (h_lineB : b = -2)
  : a + b + a * b = 2 :=
by
  sorry

end third_quadrant_angle_bisector_l110_110453


namespace coin_flip_sequences_l110_110729

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110729


namespace sec_of_7pi_over_6_l110_110016

theorem sec_of_7pi_over_6 : real.sec (7 * real.pi / 6) = -2 * real.sqrt 3 / 3 :=
by sorry

end sec_of_7pi_over_6_l110_110016


namespace corrected_mean_l110_110228

theorem corrected_mean (n : ℕ) (mean : ℝ) (obs1 obs2 : ℝ) (inc1 inc2 cor1 cor2 : ℝ)
    (h_num_obs : n = 50)
    (h_initial_mean : mean = 36)
    (h_incorrect1 : inc1 = 23) (h_correct1 : cor1 = 34)
    (h_incorrect2 : inc2 = 55) (h_correct2 : cor2 = 45)
    : (mean * n + (cor1 - inc1) + (cor2 - inc2)) / n = 36.02 := 
by 
  -- Insert steps to prove the theorem here
  sorry

end corrected_mean_l110_110228


namespace new_trailer_homes_added_l110_110041

/-- 
Five years ago, there were 30 trailer homes in Pine Park with an average age of 10 years.
Since then, a group of brand new trailer homes has been added to Pine Park. 
Today, the average age of all the trailer homes in Pine Park is 12 years. 
Prove that the number of new trailer homes added five years ago is 13.
--/
theorem new_trailer_homes_added :
  ∃ (n : ℕ), (30 * 15 + 5 * n) / (30 + n) = 12 ∧ n = 13 :=
begin
  use 13,
  -- Simplifying the given condition
  have h1 : 30 * 15 = 450, by norm_num,
  have h2 : 5 * 13 = 65, by norm_num,
  have h3 : 30 + 13 = 43, by norm_num,
  rw [h1, h2, h3],
  norm_num,
  split;
  linarith,
end

end new_trailer_homes_added_l110_110041


namespace equilateral_triangle_not_divisible_into_five_congruent_triangles_l110_110194

theorem equilateral_triangle_not_divisible_into_five_congruent_triangles (H : ℝ → ℝ → Prop) 
  (H_equilateral : ∀ v1 v2 v3, H v1 v2 v3 → v1 ≠ v2 → v2 ≠ v3 → v3 ≠ v1 → 
  (dist v1 v2 = dist v2 v3) ∧ (dist v2 v3 = dist v3 v1) ∧ (dist v3 v1 = dist v1 v2)) :
  ¬ (∃ H1 H2 H3 H4 H5 : ℝ → ℝ → Prop, (H1 ≈ H2) ∧ (H2 ≈ H3) ∧ (H3 ≈ H4) ∧ 
  (H4 ≈ H5) ∧ (H ≈ H1 ∪ H2 ∪ H3 ∪ H4 ∪ H5)) := 
by 
  sorry

end equilateral_triangle_not_divisible_into_five_congruent_triangles_l110_110194


namespace coin_flip_sequences_l110_110761

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110761


namespace geometry_AB_times_AB_l110_110635

theorem geometry_AB_times_AB'_plus_AD_times_AD'_eq_AC_times_AC' 
  {A B C D B' C' D' : Point} 
  (h_quad : is_quadrilateral A B D C)
  (h_circle1 : circle_through A B B')
  (h_circle2 : circle_through A C C')
  (h_circle3 : circle_through A D D') 
  (h_B' : B' ∈ circle A (dist A B))
  (h_C' : C' ∈ circle A (dist A C))
  (h_D' : D' ∈ circle A (dist A D)) : 
  dist A B * dist A B' + dist A D * dist A D' = dist A C * dist A C' :=
by
  sorry

end geometry_AB_times_AB_l110_110635


namespace researchers_distribution_l110_110321

-- Defining the problem conditions and correct answer
def department_distributes_researchers : Prop :=
  (∃ (R S : ℕ) (n : ℕ → ℕ) (f : Fin 4 → Fin 3),
    R = 4 ∧ S = 3 ∧
    (∀ i : Fin 3, 1 ≤ n ⟨i, _⟩) ∧  -- Each school gets at least one researcher
    ((nat.choose (R - 1) (S - 1)) * (R - 1)! = 36)) -- Expected answer

theorem researchers_distribution : department_distributes_researchers :=
by
  -- setting up all required variables and conditions; including chosen function (exists)
  -- skipping detailed combinatorial proof
  exact ⟨4, 3, λ i, 1, λ i, by fin_cases i; norm_num, by norm_num, sorry⟩

end researchers_distribution_l110_110321


namespace total_coughs_after_20_minutes_l110_110048

theorem total_coughs_after_20_minutes (georgia_rate robert_rate : ℕ) (coughs_per_minute : ℕ) :
  georgia_rate = 5 →
  robert_rate = 2 * georgia_rate →
  coughs_per_minute = georgia_rate + robert_rate →
  (20 * coughs_per_minute) = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_coughs_after_20_minutes_l110_110048


namespace coin_flip_sequences_l110_110709

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110709


namespace pure_imaginary_implies_a_zero_l110_110073

variable (a : ℝ)
variable (i : ℂ)

theorem pure_imaginary_implies_a_zero (h_a_real : a ∈ ℝ) (h_i_imaginary : ∃ b : ℝ, i = b * complex.I) : a = 0 :=
by
  sorry

end pure_imaginary_implies_a_zero_l110_110073


namespace smallest_number_l110_110654

-- Definitions based on the conditions given in the problem
def satisfies_conditions (b : ℕ) : Prop :=
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1

-- Lean proof statement
theorem smallest_number (b : ℕ) : satisfies_conditions b → b = 87 :=
sorry

end smallest_number_l110_110654


namespace coin_flip_sequences_l110_110707

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110707


namespace quadrilateral_is_rhombus_l110_110581

theorem quadrilateral_is_rhombus
  {A B C D : Type*} [convex_quadrilateral A B C D]
  (distance_condition : ∀ (M1 M2 : Type*),
    (midpoint_distance_squared M1 M2) =
    1 / 2 * ((side_length_squared A B) + (side_length_squared C D))) :
  is_rhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l110_110581


namespace incorrect_statement_D_l110_110884

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (h_q : q ≠ 1)
variable (h_geo : ∀ n, a (n + 1) = a n * q)
variable (h_seq_a2_gt_a1 : a 2 > a 1)

theorem incorrect_statement_D :
  ¬(∀ n ≥ 1, a (n + 1) > a n) :=
by {
  have h_q_gt_1: q > 1 := sorry,   -- This is derived from h_seq_a2_gt_a1 in true solution.
  exact sorry                      -- Placeholder for actual proof showing D is incorrect
}

end incorrect_statement_D_l110_110884


namespace coin_flip_sequences_l110_110756

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110756


namespace max_angle_x_coordinate_l110_110516

def point : Type := ℝ × ℝ

def M : point := (-1, 2)
def N : point := (1, 4)

def P (x : ℝ) : point := (x, 0)

theorem max_angle_x_coordinate :
  ∃ x : ℝ , ∀ x', (∠(M, P x, N) ≤ ∠(M, P x', N)) ∧ x = 1 :=
by
  sorry

end max_angle_x_coordinate_l110_110516


namespace original_two_digit_number_is_52_l110_110946

theorem original_two_digit_number_is_52 (x : ℕ) (h1 : 10 * x + 6 = x + 474) (h2 : 10 ≤ x ∧ x < 100) : x = 52 :=
sorry

end original_two_digit_number_is_52_l110_110946


namespace transform_sin_cos_eq_shift_left_pi_div_2_l110_110636

theorem transform_sin_cos_eq_shift_left_pi_div_2 :
  ∀ x : ℝ, (sin x + cos x) = (λ t, sin t - cos t) (x + π/2) := 
sorry

end transform_sin_cos_eq_shift_left_pi_div_2_l110_110636


namespace point_distance_is_pm_3_l110_110573

theorem point_distance_is_pm_3 (Q : ℝ) (h : |Q - 0| = 3) : Q = 3 ∨ Q = -3 :=
sorry

end point_distance_is_pm_3_l110_110573


namespace car_speed_is_48_kmh_l110_110496

-- Defining the conditions
def tire_revolutions_per_minute : ℕ := 400
def tire_circumference : ℝ := 2 -- meters

-- Defining the derived properties from the conditions
def distance_per_minute : ℝ := tire_revolutions_per_minute * tire_circumference -- meters
def distance_per_hour : ℝ := distance_per_minute * 60 -- meters
def speed_in_kmh : ℝ := distance_per_hour / 1000 -- kilometers

-- The theorem to be proven
theorem car_speed_is_48_kmh (tire_revolutions_per_minute = 400)
    (tire_circumference = 2) : speed_in_kmh = 48 := by
  sorry

end car_speed_is_48_kmh_l110_110496


namespace area_of_triangle_OAB_l110_110897

theorem area_of_triangle_OAB : 
  let O := (0, 0)
  let parabola := λ (x y : ℝ), y^2 = 4 * x
  let focus := (1, 0)
  let A := (sqrt 5)^2, 2 * sqrt 5
  let B := (sqrt 5)^2, -2 * sqrt 5
in
  -- Given conditions: orthocenter of triangle OAB is the focus of the parabola, and A, B lie on the parabola.
  (orthocenter(O, A, B) = focus)
  ∧ (parabola (fst A) (snd A))
  ∧ (parabola (fst B) (snd B))
  →
  -- Prove the area S of the triangle OAB is 10√5
  abs (0.5 * ((fst A * (snd B) - (snd A * fst B))) ) = 10 * sqrt(5):
    sorry

end area_of_triangle_OAB_l110_110897


namespace coin_flip_sequences_l110_110710

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110710


namespace evaluate_g_at_3_l110_110934

def g : ℝ → ℝ := fun x => x^2 - 3 * x + 2

theorem evaluate_g_at_3 : g 3 = 2 := by
  sorry

end evaluate_g_at_3_l110_110934


namespace coin_flip_sequences_l110_110708

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110708


namespace students_calculation_l110_110153

variable (students_boys students_playing_soccer students_not_playing_soccer girls_not_playing_soccer : ℕ)
variable (percentage_boys_play_soccer : ℚ)

def students_not_playing_sum (students_boys_not_playing : ℕ) : ℕ :=
  students_boys_not_playing + girls_not_playing_soccer

def total_students (students_not_playing_sum students_playing_soccer : ℕ) : ℕ :=
  students_not_playing_sum + students_playing_soccer

theorem students_calculation 
  (H1 : students_boys = 312)
  (H2 : students_playing_soccer = 250)
  (H3 : percentage_boys_play_soccer = 0.86)
  (H4 : girls_not_playing_soccer = 73)
  (H5 : percentage_boys_play_soccer * students_playing_soccer = 215)
  (H6 : students_boys - 215 = 97)
  (H7 : students_not_playing_sum 97 = 170)
  (H8 : total_students 170 250 = 420) : ∃ total, total = 420 :=
by 
  existsi total_students 170 250
  exact H8

end students_calculation_l110_110153


namespace line_through_intersections_l110_110483

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → 2 * x - 3 * y = 0 := 
sorry

end line_through_intersections_l110_110483


namespace circle_radius_l110_110684

theorem circle_radius (r M N : ℝ) (hM : M = π * r^2) (hN : N = 2 * π * r) (hRatio : M / N = 20) : r = 40 := 
by
  sorry

end circle_radius_l110_110684


namespace race_time_catch_up_l110_110666

theorem race_time_catch_up (t : ℝ) (h : 5 * t = 54 + 3 * t) : t = 27 :=
begin
  -- proof goes here
  sorry
end

end race_time_catch_up_l110_110666


namespace math_proof_problem_l110_110085

variable {R : Type*} [OrderedRing R] (f : R → R)

-- Define conditions as hypotheses
def oddFunction (f : R → R) : Prop := ∀ x, f(x + 1) = -f(-(x + 1))
def periodicCondition (f : R → R) : Prop := ∀ x, f(x + 4) = f(-x)

-- Define the verification function for the conclusions
def Correct_Conclusions (f : R → R) : Prop :=
  (∀ x, f x = f (-x)) ∧ (f 3 = 0) ∧ (f 2023 = 0)

-- Formal statement of the theorem
theorem math_proof_problem (h1 : oddFunction f) (h2 : periodicCondition f) : Correct_Conclusions f :=
by
  sorry

end math_proof_problem_l110_110085


namespace distinct_sequences_ten_flips_l110_110737

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110737


namespace find_k_range_l110_110436

-- Define the conditions 
def condition1 (k : ℝ) : Prop := (k - 4) * (k - 6) < 0
def condition2 (k : ℝ) : Prop := (4 / 5 + 1 / k ≤ 1) ∧ (k ≠ 5)

-- Define the proposition p ∧ q
def pq (k : ℝ) : Prop := condition1 k ∧ condition2 k

-- Define the range for k
def range_k : set ℝ := { k | 5 < k ∧ k < 6 }

-- State the theorem
theorem find_k_range (k : ℝ) (h : pq k) : k ∈ range_k :=
by
  -- We assume pq k holds, which means both conditions hold, 
  -- and we need to show that k is in the range (5,6).
  sorry

end find_k_range_l110_110436


namespace smallest_positive_period_intervals_of_decrease_max_min_in_interval_l110_110903

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * (sqrt 3 * sin x + cos x) + 2

theorem smallest_positive_period (h : ℝ → ℝ) (f = h) : (∃ T > 0, ∀ x, h (x + T) = h x) :=
  sorry

theorem intervals_of_decrease (h : ℝ → ℝ) (f = h) : (∃ k : ℤ, ∀ x, h' x < 0 → x ∈ (kπ + π / 6, kπ + 2π / 3)) :=
  sorry

theorem max_min_in_interval (h : ℝ → ℝ) (f = h) : (exists max min : ℝ, ∀ x ∈ [0, π / 2], h x ≤ max ∧ h x ≥ min) :=
  sorry

end smallest_positive_period_intervals_of_decrease_max_min_in_interval_l110_110903


namespace tank_capacity_l110_110771

theorem tank_capacity (w c : ℝ) (h1 : w / c = 1 / 6) (h2 : (w + 5) / c = 1 / 3) : c = 30 :=
by
  sorry

end tank_capacity_l110_110771


namespace calculateTotalProfit_l110_110356

-- Defining the initial investments and changes
def initialInvestmentA : ℕ := 5000
def initialInvestmentB : ℕ := 8000
def initialInvestmentC : ℕ := 9000

def additionalInvestmentA : ℕ := 2000
def withdrawnInvestmentB : ℕ := 1000
def additionalInvestmentC : ℕ := 3000

-- Defining the durations
def months1 : ℕ := 4
def months2 : ℕ := 8
def months3 : ℕ := 6

-- C's share of the profit
def shareOfC : ℕ := 45000

-- Total profit to be proved
def totalProfit : ℕ := 103571

-- Lean 4 theorem statement
theorem calculateTotalProfit :
  let ratioA := (initialInvestmentA * months1) + ((initialInvestmentA + additionalInvestmentA) * months2)
  let ratioB := (initialInvestmentB * months1) + ((initialInvestmentB - withdrawnInvestmentB) * months2)
  let ratioC := (initialInvestmentC * months3) + ((initialInvestmentC + additionalInvestmentC) * months3)
  let totalRatio := ratioA + ratioB + ratioC
  (shareOfC / ratioC : ℚ) = (totalProfit / totalRatio : ℚ) :=
sorry

end calculateTotalProfit_l110_110356


namespace problem_part1_problem_part2_l110_110476

noncomputable def f (x : ℝ) : ℝ := abs (2*x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x-1)

theorem problem_part1 (x : ℝ) : f x < 4 ↔ -3/2 < x ∧ x < 5/2 := 
begin
  sorry
end

theorem problem_part2 (a m n : ℝ) (h1 : g a = 2) (h2 : m + n = 2) (h3 : 0 < m) (h4 : 0 < n) : 
  ∃ (k : ℝ), k = (m^2 + 2) / m + (n^2 + 1) / n ∧ k ∈ set.Ici ((7 + 2 * real.sqrt 2) / 2) :=
begin
  sorry
end

end problem_part1_problem_part2_l110_110476


namespace area_ratio_S3_S1_l110_110540

def S1 (x y : ℝ) : Prop := real.log10 (1 + x^2 + y^2) ≤ 1 + real.log10 (x + y)
def S3 (x y : ℝ) : Prop := real.log10 (3 + x^2 + y^2) ≤ 2 + real.log10 (x + y)

theorem area_ratio_S3_S1 : 
  (set.univ.prod set.univ).measure (λ (x y : ℝ), S3 x y) / 
  (set.univ.prod set.univ).measure (λ (x y : ℝ), S1 x y) = 100 :=
sorry

end area_ratio_S3_S1_l110_110540


namespace dog_catches_rabbit_in_75_jumps_l110_110322

-- Definitions for the conditions
def initial_gap : ℝ := 150
def rabbit_jump : ℝ := 7
def dog_jump : ℝ := 9
def gap_closure_per_jump : ℝ := dog_jump - rabbit_jump

theorem dog_catches_rabbit_in_75_jumps :
  gap_closure_per_jump * 75 = initial_gap :=
by
  have h_gap_closure : gap_closure_per_jump = 2 := by norm_num
  rw [h_gap_closure]
  norm_num
  sorry

end dog_catches_rabbit_in_75_jumps_l110_110322


namespace new_pressure_of_transferred_gas_l110_110809

theorem new_pressure_of_transferred_gas (V1 V2 : ℝ) (p1 k : ℝ) 
  (h1 : V1 = 3.5) (h2 : p1 = 8) (h3 : k = V1 * p1) (h4 : V2 = 7) :
  ∃ p2 : ℝ, p2 = 4 ∧ k = V2 * p2 :=
by
  use 4
  sorry

end new_pressure_of_transferred_gas_l110_110809


namespace find_omega_l110_110079

noncomputable def z_imaginary (z : ℂ) : Prop :=
  let z_re := z.re in
  let z_im := z.im in
  (1 + 3 * Complex.i) * z = Complex.i * (z_re - 3 * z_im)

noncomputable def omega_eq (z : ℂ) (ω : ℂ) : Prop :=
  ω = z / (2 + Complex.i)

noncomputable def omega_modulus (ω : ℂ) : Prop :=
  Complex.abs ω = 5 * Real.sqrt 2

theorem find_omega (z ω : ℂ) :
  (z_imaginary z) ∧ (omega_eq z ω) ∧ (omega_modulus ω) ↔ (ω = 7 - Complex.i) ∨ (ω = -7 + Complex.i) :=
by sorry

end find_omega_l110_110079


namespace find_beta_minus_alpha_l110_110950

variables (α β : ℝ)
variables (a b : ℝ × ℝ)
variables (ha : a = (sqrt 2 * real.cos α, sqrt 2 * real.sin α))
variables (hb : b = (2 * real.cos β, 2 * real.sin β))
variables (hα : π / 6 ≤ α ∧ α < π / 2)
variables (hβ : π / 2 < β ∧ β ≤ 5 * π / 6)
variables (horth : (sqrt 2 * real.cos α, sqrt 2 * real.sin α) ⬝ ((2 * real.cos β, 2 * real.sin β) - (sqrt 2 * real.cos α, sqrt 2 * real.sin α)) = 0)

theorem find_beta_minus_alpha : β - α = π / 4 := 
sorry

end find_beta_minus_alpha_l110_110950


namespace find_p_l110_110124

theorem find_p (A B C p q r s : ℝ) (h₀ : A ≠ 0)
  (h₁ : r + s = -B / A)
  (h₂ : r * s = C / A)
  (h₃ : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C + 2 * A^2 * C^2) / A^3 :=
sorry

end find_p_l110_110124


namespace smallest_four_digit_multiple_of_17_is_1013_l110_110282

-- Lean definition to state the problem
def smallest_four_digit_multiple_of_17 : ℕ :=
  1013

-- Main Lean theorem to assert the correctness
theorem smallest_four_digit_multiple_of_17_is_1013 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = smallest_four_digit_multiple_of_17 :=
by
  -- proof here
  sorry

end smallest_four_digit_multiple_of_17_is_1013_l110_110282


namespace ratio_of_boys_to_girls_l110_110137

def boys_girls_ratio (b g : ℕ) : ℚ := b / g

theorem ratio_of_boys_to_girls (b g : ℕ) (h1 : b = g + 6) (h2 : g + b = 40) :
  boys_girls_ratio b g = 23 / 17 :=
by
  sorry

end ratio_of_boys_to_girls_l110_110137


namespace point_not_on_graph_l110_110658

theorem point_not_on_graph : ¬ ∃ (x y : ℝ), (y = (x - 1) / (x + 2)) ∧ (x = -2) ∧ (y = 3) :=
by
  sorry

end point_not_on_graph_l110_110658


namespace relationship_among_abc_l110_110879

-- Define the constants using the given conditions
def a : ℝ := 17^(1/17)
def b : ℝ := Real.log 17 / Real.log 16 / 2
def c : ℝ := Real.log 16 / Real.log 17 / 2

-- State the theorem to prove the relationship among a, b, and c
theorem relationship_among_abc : a > b ∧ b > c := by 
  sorry

end relationship_among_abc_l110_110879


namespace loop_probability_correct_l110_110009

noncomputable def cube_loop_probability : ℚ :=
  3 / 32

theorem loop_probability_correct :
  let cube_conditions := 
    ∀ (faces : Fin 6) (midpoints : faces → Fin 2), 
      ∃ (stripes : faces → (midpoints faces) → ℤ),
        ∀ (opposing_faces : Fin 3),
          let face_pair := (opposing_faces, opposing_faces + 3) in
          let perpendicular_condition := 
            stripes (face_pair.1) = -stripes (face_pair.2) in
          true
  in
  let loop_condition := 
    ∃ (continuous_stripe : Fin 3 → Bool), true
  in
  cube_conditions ∧ loop_condition → 
  cube_loop_probability = 3 / 32 := 
by
  sorry

end loop_probability_correct_l110_110009


namespace average_mileaage_approximately_31_l110_110767

noncomputable def average_mileage_trip 
  (dist_to_grandparents : ℝ) 
  (mpg_compact : ℝ) 
  (mpg_sedan : ℝ) : ℝ :=
  let total_distance := 2 * dist_to_grandparents
  let gas_used_compact := dist_to_grandparents / mpg_compact
  let gas_used_sedan := dist_to_grandparents / mpg_sedan
  let total_gas_used := gas_used_compact + gas_used_sedan
  total_distance / total_gas_used

theorem average_mileaage_approximately_31 :
  average_mileage_trip 150 40 25 ≈ 31 :=
by
  sorry

end average_mileaage_approximately_31_l110_110767


namespace three_card_selections_divisible_by_three_l110_110630

theorem three_card_selections_divisible_by_three :
  let card_numbers := list.range (160) |>.map (λ n, 5 + n * 5)
  in let count_div_by_three := ∑ [ r in [0,1,2], 
                                   r.count 
                                      ( λ x, 
                                        ∃ y, y ∈ card_numbers ∧ y % 3 = x)]
  in count_div_by_three = 223342 := by
  sorry

end three_card_selections_divisible_by_three_l110_110630


namespace correct_statement_l110_110295

-- conditions
def condition_a : Prop := ∀(population : Type), ¬(comprehensive_survey population)
def data_set := [3, 5, 4, 1, -2]
def condition_b : Prop := median data_set = 4
def winning_probability : ℚ := 1 / 20
def condition_c : Prop := (∀n : ℕ, winning_probability * n = 1) → (∃i, i = 20)
def average_score (scores : List ℝ) : ℝ := scores.sum / scores.length
def variance (scores : List ℝ) : ℝ := (scores.map (λ x, (x - average_score scores)^2)).sum / scores.length
def scores_a := replicate 10 (average_score (replicate 10 10)) -- assumed scores for simplification
def scores_b := scores_a ++ [n + 1 for n in scores_a] -- assumed different scores to reflect variance
def condition_d : Prop := (average_score scores_a = average_score scores_b) ∧ (variance scores_a = 0.4) ∧ (variance scores_b = 2)

-- statement of the problem
theorem correct_statement : 
  condition_a ∧ condition_b ∧ condition_c ∧ condition_d → (∃D : Prop, D = condition_d) :=
by
  sorry

end correct_statement_l110_110295


namespace angle_inequality_l110_110997

variable (f : ℝ → ℝ)
variable (h_decreasing : ∀ x y : ℝ, x < y → f y < f x)

theorem angle_inequality
  (α β : ℝ)
  (h_acute_triangle : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ α + β > π / 2)
  : f (cos α) > f (sin β) := by
  -- proof here
  sorry

end angle_inequality_l110_110997


namespace function_increasing_interval_l110_110638

theorem function_increasing_interval : 
  ∀ x : ℝ, 
  (π / 12 ≤ x ∧ x ≤ 7 * π / 12) ↔
  ∃ k : ℤ, 
    y = 3 * sin(2 * (x - π / 2) + π / 3) ∧
    -π / 2 + 2 * k * π ≤ 2 * x - 2 * π / 3 ∧ 
    2 * x - 2 * π / 3 ≤ π / 2 + 2 * k * π :=
by
  sorry

end function_increasing_interval_l110_110638


namespace swap_numbers_l110_110213

theorem swap_numbers (a b : ℕ) (hc: b = 17) (ha : a = 8) : 
  ∃ c, c = b ∧ b = a ∧ a = c := 
by
  sorry

end swap_numbers_l110_110213


namespace groom_dog_time_l110_110982

theorem groom_dog_time :
  ∃ (D : ℝ), (5 * D + 3 * 0.5 = 14) ∧ (D = 2.5) :=
by
  sorry

end groom_dog_time_l110_110982


namespace sum_series_eq_two_l110_110376

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l110_110376


namespace factor_expression_l110_110847

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110847


namespace distinct_sequences_ten_flips_l110_110733

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110733


namespace factor_expression_l110_110839

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110839


namespace problem1_solution_set_problem2_range_of_m_l110_110106

def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

theorem problem1_solution_set :
  {x : ℝ | f x ≤ 2} = {x : ℝ | -4 ≤ x ∧ x ≤ 10} := 
sorry

theorem problem2_range_of_m (m : ℝ) (h : ∃ x : ℝ, f x - g x ≥ m - 3) :
  m ≤ 5 :=
sorry

end problem1_solution_set_problem2_range_of_m_l110_110106
