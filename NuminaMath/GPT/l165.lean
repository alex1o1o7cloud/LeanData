import Mathlib

namespace trajectory_equation_no_such_point_l165_165481

-- Conditions for (I): The ratio of the distances is given
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let M := (1, 0)
  let N := (4, 0)
  2 * Real.sqrt ((P.1 - M.1)^2 + P.2^2) = Real.sqrt ((P.1 - N.1)^2 + P.2^2)

-- Proof of (I): Find the trajectory equation of point P
theorem trajectory_equation : 
  ∀ P : ℝ × ℝ, ratio_condition P → P.1^2 + P.2^2 = 4 :=
by
  sorry

-- Conditions for (II): Given points A, B, C
def points_condition (P : ℝ × ℝ) : Prop :=
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (-4, 2)
  (P.1 + 2)^2 + (P.2 + 2)^2 + 
  (P.1 + 2)^2 + (P.2 - 6)^2 + 
  (P.1 + 4)^2 + (P.2 - 2)^2 = 36

-- Proof of (II): Determine the non-existence of point P
theorem no_such_point (P : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 4 → ¬ points_condition P :=
by
  sorry

end trajectory_equation_no_such_point_l165_165481


namespace sum_of_divisors_of_12_l165_165420

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165420


namespace tangent_line_at_x1_g_leq_zero_l165_165855

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a * x^2 - 2 * a * x

theorem tangent_line_at_x1 (a : ℝ) (h_a : a = 1) :
  let f_a := f 1 a in
  f_a = -1 ∧ (∃ m, m = 1 ∧ ∃ b, b = -2 ∧ (∀ x, f_a = m * (x - 1) + b)) :=
by sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a - (1 / 2) * x^2

theorem g_leq_zero (a : ℝ) : (∀ x > 1, g x a ≤ 0) ↔ -1/2 ≤ a ∧ a ≤ 1/2 :=
by sorry

end tangent_line_at_x1_g_leq_zero_l165_165855


namespace maximum_value_of_function_l165_165358

noncomputable def f (x y : ℝ) : ℝ :=
  (2 * x + 3 * y + 4) / Real.sqrt (x ^ 2 + 2 * y ^ 2 + 1)

theorem maximum_value_of_function :
  ∃ x y : ℝ, f(x, y) = Real.sqrt 29 := 
by
  sorry

end maximum_value_of_function_l165_165358


namespace angle_CPQ_l165_165044

variables {Point : Type} [EuclideanGeometry Point]
variables {A P B Q C : Point}
variables {l m t : Line Point}

-- Given conditions
def condition_1 : Parallel l m := sorry
def condition_2 : intersects t l P := sorry
def condition_3 : intersects t m Q := sorry
def condition_4 : angle APQ = 130 := sorry
def condition_5 : angle BQP = 140 := sorry

-- Proof statement
theorem angle_CPQ : angle CPQ = 50 :=
by
  sorry

end angle_CPQ_l165_165044


namespace coeff_x3_binomial_expansion_l165_165633

theorem coeff_x3_binomial_expansion :
  (∃ (C : ℕ → ℕ → ℕ),
    C 6 3 * (-2)^3 = -160) :=
begin
  use binomial,
  sorry,
end

end coeff_x3_binomial_expansion_l165_165633


namespace sum_of_perfect_square_squares_less_than_1000_l165_165172

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165172


namespace intersection_complement_l165_165499

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { y | y ≥ 0 }
noncomputable def B : Set ℝ := { y | y ≥ 1 }

theorem intersection_complement :
  A ∩ (U \ B) = Ico 0 1 :=
by
  sorry

end intersection_complement_l165_165499


namespace cosine_pi_over_2_plus_alpha_l165_165842

theorem cosine_pi_over_2_plus_alpha
  (P : ℝ × ℝ)
  (hP : P = (-4/5, 3/5))
  (h_unit_circle : P.1^2 + P.2^2 = 1) :
  cos (π/2 + arc_sin P.2) = -P.2 :=
by
  have h_cos_alpha : cos (arc_sin P.2) = -4/5,
  { -- This uses the condition that P is on the unit circle, hence we get cos α implicitly 
    -- knowing that sin α = P.2 and cos² α + sin² α = 1.
    sorry },
  have h_sin_alpha : sin (arc_sin P.2) = P.2,
  { -- By definition of arc_sin
    sorry },
  have trigonometric_identity : cos (π/2 + arc_sin P.2) = -sin (arc_sin P.2),
  { -- Proven by known trigonometric identity
    sorry },
  rw [h_sin_alpha] at trigonometric_identity,
  exact trigonometric_identity

end cosine_pi_over_2_plus_alpha_l165_165842


namespace cos_value_l165_165451

variable (x : ℝ)
variable (h : sin (x + π / 12) = 1 / 3)

theorem cos_value : cos (x + 7 * π / 12) = -1 / 3 :=
by
  sorry

end cos_value_l165_165451


namespace equation_of_tangent_circle_l165_165828

theorem equation_of_tangent_circle :
  ∃ (h k r : ℝ), 
    (k = -h) ∧
    (r = sqrt 2) ∧
    (abs (2 * h) / sqrt 2 = sqrt 2) ∧
    ((h = 1 ∧ k = -1) ∨ (h = -1 ∧ k = 1)) ∧
    ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → 
      (x - 1)^2 + (y + 1)^2 = 2 :=
begin
  sorry
end

end equation_of_tangent_circle_l165_165828


namespace part1_part2_l165_165450

variable (x m : ℝ)

def A : ℝ := 2 * x^2 + 3 * m * x - 2 * x - 1
def B : ℝ := - x^2 + m * x - 1

theorem part1 : 3 * A + 6 * B = (15 * m - 6) * x - 9 := sorry

theorem part2 (h : (15 * m - 6) * x - 9 = 0) : m = 2 / 5 := by
  have h' : 15 * m - 6 = 0 := by
    sorry
  linarith

end part1_part2_l165_165450


namespace min_treasures_buried_l165_165327

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165327


namespace plan_A_is_cost_effective_l165_165345

-- Definitions of the costs considering the problem's conditions
def cost_plan_A (days_A : ℕ) (rate_A : ℕ) : ℕ := days_A * rate_A
def cost_plan_C (days_AB : ℕ) (rate_A : ℕ) (rate_B : ℕ) (remaining_B : ℕ) : ℕ :=
  (days_AB * (rate_A + rate_B)) + (remaining_B * rate_B)

-- Specification of the days and rates from the conditions
def days_A := 12
def rate_A := 10000
def rate_B := 6000
def days_AB := 3
def remaining_B := 13

-- Costs for each plan
def A_cost := cost_plan_A days_A rate_A
def C_cost := cost_plan_C days_AB rate_A rate_B remaining_B

-- Theorem stating that Plan A is more cost-effective
theorem plan_A_is_cost_effective : A_cost < C_cost := by
  unfold A_cost
  unfold C_cost
  sorry

end plan_A_is_cost_effective_l165_165345


namespace decreasing_sequence_inequality_l165_165587

-- Define decreasing sequence with appropriate condition
def decreasing_seq (x : ℕ → ℝ) : Prop :=
  ∀ m n : ℕ, (m < n) → (x n ≤ x m)

-- Define the condition for the given sequence
def sequence_condition (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, x 1 + ∑ i in (finset.range n).filter (λ i, (nat.sqrt i.succ)^2 = i.succ), (x i.succ) / (nat.sqrt i.succ) ≤ 1

-- Define the target inequality to prove
def target_inequality (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x 1 + ∑ i in (finset.range n), (x (i + 1)) / (i + 1) < 3

-- Main theorem statement
theorem decreasing_sequence_inequality
  (x : ℕ → ℝ) (pos_x : ∀ n, x n > 0)
  (dec_seq : decreasing_seq x)
  (cond_seq : sequence_condition x) :
  ∀ n : ℕ, target_inequality x n :=
sorry

end decreasing_sequence_inequality_l165_165587


namespace solve_cubic_eq_l165_165973

theorem solve_cubic_eq : 
  ∀ x : ℝ, (x^3 - 3 * x^2 * real.sqrt 3 + 9 * x - 3 * real.sqrt 3) + (x - real.sqrt 3)^2 = 0 → 
  x = real.sqrt 3 ∨ x = -1 + real.sqrt 3 :=
by
  intros x h
  sorry

end solve_cubic_eq_l165_165973


namespace ab_bc_ca_abc_inequality_l165_165937

open Real

theorem ab_bc_ca_abc_inequality :
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 + a * b * c = 4 →
    0 ≤ a * b + b * c + c * a - a * b * c ∧ a * b + b * c + c * a - a * b * c ≤ 2 :=
by
  intro a b c
  intro h
  sorry

end ab_bc_ca_abc_inequality_l165_165937


namespace subset_complement_M_U_l165_165865

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set M in terms of parameter a
def M (a : ℝ) : Set ℝ := {x | (3 * a - 1) < x ∧ x < 2 * a}

-- Define the set N
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Define the complement of M in U
def complement_M_U (a : ℝ) : Set ℝ := U \ (M a)

-- Final theorem to prove
theorem subset_complement_M_U (a : ℝ) :
    N ⊆ complement_M_U a ↔ (a ≤ -1 / 2) ∨ (a ≥ 1) :=
begin
    sorry
end

end subset_complement_M_U_l165_165865


namespace jay_paid_amount_l165_165016

theorem jay_paid_amount
  (cost_book : ℕ) (cost_pen : ℕ) (cost_ruler : ℕ) (change : ℕ)
  (h_book : cost_book = 25) (h_pen : cost_pen = 4) (h_ruler : cost_ruler = 1) (h_change : change = 20) :
  let total_cost := cost_book + cost_pen + cost_ruler in
  let amount_paid := total_cost + change in
  amount_paid = 50 :=
by
  sorry

end jay_paid_amount_l165_165016


namespace sum_of_integer_solutions_l165_165789

theorem sum_of_integer_solutions (h : ∀ x : ℤ, x^4 - 33 * x^2 + 224 = 0) : ∑ (x : ℤ) in {x : ℤ | x^4 - 33 * x^2 + 224 = 0}.toFinset = 0 := 
by
  sorry

end sum_of_integer_solutions_l165_165789


namespace correct_sampling_methods_l165_165108

-- Define the conditions as assumptions
def community_with_income_families : Prop :=
  ∃ (high_income middle_income low_income : ℕ),
    high_income = 125 ∧
    middle_income = 280 ∧
    low_income = 95

def interest_in_social_purchasing_power : Prop := true
def small_number_of_athletes : Prop := true

-- Define the conclusions as the sampling methods
def use_stratified_sampling (survey: Prop) : Prop := survey = interest_in_social_purchasing_power
def use_simple_random_sampling (survey: Prop) : Prop := survey = small_number_of_athletes

-- The proof statement combining conditions and conclusions
theorem correct_sampling_methods :
  (community_with_income_families → interest_in_social_purchasing_power → use_stratified_sampling interest_in_social_purchasing_power) ∧
  (small_number_of_athletes → use_simple_random_sampling small_number_of_athletes) :=
begin
  split,
  { intros _ _,
    exact eq.refl _ },
  { intros _,
    exact eq.refl _ }
end

end correct_sampling_methods_l165_165108


namespace exists_two_lines_with_angle_le_26_degrees_l165_165805

theorem exists_two_lines_with_angle_le_26_degrees (lines : List ( ℝ × ℝ × ℝ )) 
  (h_len : lines.length = 7) 
  (h_pairwise : ∀ (i j : ℕ), i ≠ j → ¬( ∃ k : ℝ, ∀ (x y : ℝ), lines.nth i = some (x, k * x + y, 0) ∧ lines.nth j = some (x, k * x + y, 0))) :
  ∃ (i j : ℕ), i ≠ j ∧ (angle_between (lines.nth i) (lines.nth j) ≤ 26) :=
sorry

end exists_two_lines_with_angle_le_26_degrees_l165_165805


namespace min_distance_value_l165_165784

noncomputable def distance_a (x : ℝ) : ℝ := real.sqrt (x ^ 2 + (2 - x) ^ 2)
noncomputable def distance_b (x : ℝ) : ℝ := real.sqrt ((1 - x) ^ 2 + (-1 + x) ^ 2)

theorem min_distance_value : 
  ∀ x : ℝ, distance_a x + distance_b x ≥ real.sqrt 10 :=
by
  intro x
  -- Creating the definition for given distances
  let AP := distance_a x
  let BP := distance_b x
  have : AP + BP ≥ real.sqrt 10 := sorry
  exact this

end min_distance_value_l165_165784


namespace probability_smallest_diff_ge_three_l165_165661

-- Define the set of numbers
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the smallest difference constraint
def smallest_diff_ge_three (a b c : ℕ) : Prop :=
  (|a - b| ≥ 3) ∧ (|a - c| ≥ 3) ∧ (|b - c| ≥ 3)

-- Define a function that computes the number of valid combinations
noncomputable def num_valid_combinations : ℕ := 
  Finset.card {(x, y, z) ∈ Finset.product (Finset.product S S) S | smallest_diff_ge_three x y z}

-- Define the total number of combinations
def total_combinations : ℕ :=
  Nat.choose 8 3 -- This is binom(8, 3)

-- Define the probability calculation
def probability_valid_combinations : ℚ :=
  num_valid_combinations / total_combinations

-- The theorem to prove
theorem probability_smallest_diff_ge_three : probability_valid_combinations = 1 / 28 :=
by
  sorry

end probability_smallest_diff_ge_three_l165_165661


namespace treasure_under_minimum_signs_l165_165274

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165274


namespace sum_of_squares_of_perfect_squares_l165_165133

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165133


namespace magnitude_of_sum_l165_165478

open Real

-- Define the angle between vectors
def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  acos ((a.1 * b.1 + a.2 * b.2) / (sqrt (a.1^2 + a.2^2) * sqrt (b.1^2 + b.2^2)))

-- Given conditions
def vec_a : ℝ × ℝ := (2, 0) -- Magnitude |a| = 2
def vec_b (α : ℝ) : ℝ × ℝ := (cos α, sin α) -- Vector b

-- Prove the statement
theorem magnitude_of_sum (α : ℝ) (h : angle_between_vectors vec_a (vec_b α) = π / 3) :
  let ab := (vec_a.1 + 2 * vec_b α.1, vec_a.2 + 2 * vec_b α.2) -- a + 2b
  sqrt (ab.1^2 + ab.2^2) = 2 * sqrt 3 :=
by
  sorry

end magnitude_of_sum_l165_165478


namespace smallest_possible_l_l165_165715

theorem smallest_possible_l (a b c L : ℕ) (h1 : a * b = 7) (h2 : a * c = 27) (h3 : b * c = L) (h4 : ∃ k, a * b * c = k * k) : L = 21 := sorry

end smallest_possible_l_l165_165715


namespace problem_solution_l165_165938

theorem problem_solution :
  ∀ (a b c d : ℝ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
    (a^2 = 7 ∨ a^2 = 8) →
    (b^2 = 7 ∨ b^2 = 8) →
    (c^2 = 7 ∨ c^2 = 8) →
    (d^2 = 7 ∨ d^2 = 8) →
    a^2 + b^2 + c^2 + d^2 = 30 :=
by sorry

end problem_solution_l165_165938


namespace hours_to_drive_l165_165729

theorem hours_to_drive
  (initial_bac : ℝ)
  (decay_rate : ℝ)
  (safe_bac : ℝ)
  (h_initial : initial_bac = 0.3)
  (h_decay : decay_rate = 0.75)
  (h_safe : safe_bac = 0.09) :
  ∃ t : ℕ, t ≥ ⌈ log (safe_bac / initial_bac) / log decay_rate ⌉ :=
by
  sorry

end hours_to_drive_l165_165729


namespace min_treasures_buried_l165_165270

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165270


namespace al_sew_dresses_time_l165_165197

-- Define the conditions
def Allison_time : ℝ := 9
def together_time : ℝ := 3
def Allison_after_Al_leaves_time : ℝ := 3.75

-- Define the rates of work
def Allison_rate (Allison_time : ℝ) : ℝ := 1 / Allison_time
def Al_rate (Al_time : ℝ) : ℝ := 1 / Al_time

-- Define the total work done
def total_work_done (Allison_time : ℝ) (Al_time : ℝ) (together_time : ℝ) (Allison_after_Al_leaves_time : ℝ) : ℝ :=
  together_time * (Allison_rate Allison_time + Al_rate Al_time) + Allison_after_Al_leaves_time * Allison_rate Allison_time

-- Main theorem statement
theorem al_sew_dresses_time (Al_time : ℝ) 
  (h : total_work_done Allison_time Al_time together_time Allison_after_Al_leaves_time = 1) : 
  Al_time = 12 := 
begin
  sorry
end

end al_sew_dresses_time_l165_165197


namespace num_valid_sets_l165_165834

open Set

def valid_M (M : Set ℕ) : Prop :=
  M ⊆ {0, 1, 2, 3, 4, 5} ∧ M ≠ ∅ ∧ (∀ x ∈ M, x * x ∉ M ∧ (sqrt x).nat_abs ∉ M)

theorem num_valid_sets : (Finset.filter valid_M (Finset.powerset {0, 1, 2, 3, 4, 5})).card = 11 := sorry

end num_valid_sets_l165_165834


namespace probability_f_gt_16_l165_165847

-- Definitions
variables (a : ℝ) (x : ℝ)
def f (x : ℝ) : ℝ := a ^ x

-- Given conditions
axiom a_gt_zero (ha0 : 0 < a)
axiom a_ne_one (ha1 : a ≠ 1)
axiom passes_through_P (hP : f a 2 = 4)

-- The Problem to be proved
theorem probability_f_gt_16 (h_probs : 0 < x ∧ x ≤ 10) 
    (f_at_P : ∃ (a : ℝ), (f a 2 = 4)) : 
  (∀ (x : ℝ), (0 < x ∧ x ≤ 10) → f a x > 16): 
  (∃ (p : ℝ), p = 3/5) :=
by
  sorry

end probability_f_gt_16_l165_165847


namespace intersect_at_circumcenter_l165_165931

variable {A B C H O : Type}
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] 
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]

def is_orthocenter (H : A) (A : A) (B : B) (C : C) : Prop :=
is_orthocenter_def

def is_circumcenter (O : A) (A : A) (B : B) (C : C) : Prop :=
is_circumcenter_def

def symmetric_line (A : A) (N : B) (bisector : C) : Prop :=
symmetric_line_def

theorem intersect_at_circumcenter 
  (H : A) (A B C : A) (O : A)
  (hH : is_orthocenter H A B C)
  (hO : is_circumcenter O A B C)
  (symmetric_A : symmetric_line A H (angle_bisector A))
  (symmetric_B : symmetric_line B H (angle_bisector B))
  (symmetric_C : symmetric_line C H (angle_bisector C)) :
  intersect_lines_A_B_C_at_O (symmetric_A) (symmetric_B) (symmetric_C) = O := 
sorry

end intersect_at_circumcenter_l165_165931


namespace circular_patch_radius_l165_165218

theorem circular_patch_radius : 
  let r_cylinder := 3  -- radius of the container in cm
  let h_cylinder := 6  -- height of the container in cm
  let t_patch := 0.2   -- thickness of each patch in cm
  let V := π * r_cylinder^2 * h_cylinder -- Volume of the liquid

  let V_patch := V / 2                  -- Volume of each patch
  let r := 3 * Real.sqrt 15              -- the radius we want to prove

  r^2 * π * t_patch = V_patch           -- the volume equation for one patch
  →

  r = 3 * Real.sqrt 15 := 
by
  sorry

end circular_patch_radius_l165_165218


namespace sum_of_divisors_of_12_l165_165417

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165417


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165153

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165153


namespace triangle_angle_D_l165_165912

theorem triangle_angle_D (F E D : ℝ) (hF : F = 15) (hE : E = 3 * F) (h_triangle : D + E + F = 180) : D = 120 := by
  sorry

end triangle_angle_D_l165_165912


namespace sum_of_perfect_square_squares_less_than_1000_l165_165167

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165167


namespace unique_intersection_point_l165_165928

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point : ∃ a : ℝ, f a = a ∧ f a = -1 ∧ f a = f⁻¹ a :=
by 
  sorry

end unique_intersection_point_l165_165928


namespace area_union_after_rotation_l165_165008

-- Define the sides of the triangle
def PQ : ℝ := 11
def QR : ℝ := 13
def PR : ℝ := 12

-- Define the condition that H is the centroid of the triangle PQR
def centroid (P Q R H : ℝ × ℝ) : Prop := sorry -- This definition would require geometric relationships.

-- Statement to prove the area of the union of PQR and P'Q'R' after 180° rotation about H.
theorem area_union_after_rotation (P Q R H : ℝ × ℝ) (hPQ : dist P Q = PQ) (hQR : dist Q R = QR) (hPR : dist P R = PR) (hH : centroid P Q R H) : 
  let s := (PQ + QR + PR) / 2
  let area_PQR := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  2 * area_PQR = 12 * Real.sqrt 105 :=
sorry

end area_union_after_rotation_l165_165008


namespace fifth_number_in_eighth_row_l165_165050

theorem fifth_number_in_eighth_row : 
  (∀ n : ℕ, ∃ k : ℕ, k = n * n ∧ 
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
      k - (n - m) = 54 → m = 5 ∧ n = 8) := by sorry

end fifth_number_in_eighth_row_l165_165050


namespace minimum_treasure_buried_l165_165290

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165290


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165138

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165138


namespace johns_total_packs_l165_165560

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l165_165560


namespace length_of_DC_l165_165542

noncomputable def AB : ℝ := 30
noncomputable def sine_A : ℝ := 4 / 5
noncomputable def sine_C : ℝ := 1 / 4
noncomputable def angle_ADB : ℝ := Real.pi / 2

theorem length_of_DC (h_AB : AB = 30) (h_sine_A : sine_A = 4 / 5) (h_sine_C : sine_C = 1 / 4) (h_angle_ADB : angle_ADB = Real.pi / 2) :
  ∃ DC : ℝ, DC = 24 * Real.sqrt 15 :=
by sorry

end length_of_DC_l165_165542


namespace limit_of_sequence_z_l165_165061

open Nat Real

noncomputable def sequence_z (n : ℕ) : ℝ :=
  -3 + (-1)^n / (n^2 : ℝ)

theorem limit_of_sequence_z :
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, abs (sequence_z n + 3) < ε :=
by
  sorry

end limit_of_sequence_z_l165_165061


namespace min_workers_needed_to_make_profit_l165_165221

def wage_per_worker_per_hour := 20
def fixed_cost := 800
def units_per_worker_per_hour := 6
def price_per_unit := 4.5
def hours_per_workday := 9

theorem min_workers_needed_to_make_profit : ∃ (n : ℕ), 243 * n > 800 + 180 * n ∧ n ≥ 13 :=
by
  sorry

end min_workers_needed_to_make_profit_l165_165221


namespace toy_factory_days_per_week_l165_165222

theorem toy_factory_days_per_week (toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : toys_per_week = 4560) (h₂ : toys_per_day = 1140) : toys_per_week / toys_per_day = 4 := 
by {
  -- Proof to be provided
  sorry
}

end toy_factory_days_per_week_l165_165222


namespace length_of_az_l165_165335

noncomputable def circle_circumference := 18 * Real.pi
def angle_qaz := 45

theorem length_of_az
  (circumference_q : Real := circle_circumference)
  (diameter_ab : Bool := true)
  (angle_qaz_deg : Real := angle_qaz) :
  sqrt(162 - 81 * (sqrt 2)) = 9 * sqrt(2 - sqrt 2) :=
by
  sorry

end length_of_az_l165_165335


namespace remainder_when_squared_mod_seven_l165_165043

theorem remainder_when_squared_mod_seven
  (x y : ℤ) (k m : ℤ)
  (hx : x = 52 * k + 19)
  (hy : 3 * y = 7 * m + 5) :
  ((x + 2 * y)^2 % 7) = 1 := by
  sorry

end remainder_when_squared_mod_seven_l165_165043


namespace problem_l165_165474

noncomputable def excircle_center (A B C : Point) : Point := sorry  --defining excircle center function
noncomputable def midpoint (A C : Point) : Point := sorry  --defining midpoint function
noncomputable def intersection (l1 l2 : Line) : Point := sorry  --defining intersection function
noncomputable def intersection_point (A B C : Point) (D E : Point) : Point := sorry --defining intersection point function
def angle_eq (α β : Angle) : Prop := α = β --defining angle equality
def length_eq (a b : length) : Prop := a = b --defining length equality

variables (A B C I_A D E : Point)
variables (α : Angle)

axiom excircle_center_definition : I_A = excircle_center A B C --axitimization of excircle center definition
axiom midpoint_definition : D = midpoint A C --axitimizng midpoint definition
axiom intersection_definition : E = intersection (line_thro I_A D) (line_thro B C) -- axiomatization of point of intersection
axiom angle_condition : angle_eq (angle_ABC A B) (2 * angle_ABC A C) --axiomatization of angle equality

theorem problem :
  ∀ (A B C I_A D E : Point),
  ∀ (α : Angle),
  I_A = excircle_center A B C →
  D = midpoint A C →
  E = intersection (line_thro I_A D) (line_thro B C) →
  angle_eq (angle_ABC A B) (2 * angle_ABC A C) →
  length_eq (length_ABC AB) (length_ABC BE) :=
by
  intros A B C I_A D E α
  intros excircle_center_definition midpoint_definition intersection_definition angle_condition 
  sorry

end problem_l165_165474


namespace find_arithmetic_progression_terms_l165_165782

noncomputable def arithmetic_progression_terms (a1 a2 a3 : ℕ) (d : ℕ) 
  (condition1 : a1 + (a1 + d) = 3 * 2^2) 
  (condition2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) : Prop := 
  a1 = 3 ∧ a2 = 9 ∧ a3 = 15

theorem find_arithmetic_progression_terms
  (a1 a2 a3 : ℕ) (d : ℕ)
  (cond1 : a1 + (a1 + d) = 3 * 2^2)
  (cond2 : a1 + (a1 + d) + (a1 + 2 * d) = 3 * 3^2) :
  arithmetic_progression_terms a1 a2 a3 d cond1 cond2 :=
sorry

end find_arithmetic_progression_terms_l165_165782


namespace probability_one_number_is_twice_another_l165_165797

def numbers : Finset ℕ := {1, 2, 3, 4}

def possible_pairs : Finset (ℕ × ℕ) := 
  numbers.product numbers.filter (λ p, p.1 < p.2)

def is_twice (p : ℕ × ℕ) : Prop := p.1 * 2 = p.2 ∨ p.2 * 2 = p.1

def favorable_pairs : Finset (ℕ × ℕ) :=
  possible_pairs.filter is_twice

theorem probability_one_number_is_twice_another :
  (favorable_pairs.card : ℚ) / possible_pairs.card = 1 / 3 :=
sorry

end probability_one_number_is_twice_another_l165_165797


namespace polynomial_mod_p_integer_average_l165_165569

theorem polynomial_mod_p_integer_average (n : ℕ) (p : ℕ) (h1 : n ≥ 2) (hp : Nat.Prime p) : 
  (1 : ℚ) / (p - 1) * ∑ a in Finset.range (p-1), (Finset.card { b | ∃ c, (c ^ n + a * c) % p = b}) ∈ ℤ := 
sorry

end polynomial_mod_p_integer_average_l165_165569


namespace thabo_total_books_l165_165630

noncomputable def total_books (H PNF PF : ℕ) : ℕ := H + PNF + PF

theorem thabo_total_books :
  ∀ (H PNF PF : ℕ),
    H = 30 →
    PNF = H + 20 →
    PF = 2 * PNF →
    total_books H PNF PF = 180 :=
by
  intros H PNF PF hH hPNF hPF
  sorry

end thabo_total_books_l165_165630


namespace susan_spent_75_percent_l165_165073

variables (B b s : ℝ)

-- Conditions
def condition1 : Prop := b = 0.25 * (B - 3 * s)
def condition2 : Prop := s = 0.10 * (B - 2 * b)

-- Theorem
theorem susan_spent_75_percent (h1 : condition1 B b s) (h2 : condition2 B b s) : b + s = 0.75 * B := 
sorry

end susan_spent_75_percent_l165_165073


namespace sum_of_fourth_powers_less_than_1000_l165_165190

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165190


namespace boatworks_total_canoes_l165_165743

theorem boatworks_total_canoes : 
  let c1 := 5 in
  let c2 := 3 * c1 in
  let c3 := 3 * c2 in
  let c4 := 3 * c3 in
  c1 + c2 + c3 + c4 = 200 :=
by 
  let c1 := 5
  let c2 := 3 * c1
  let c3 := 3 * c2
  let c4 := 3 * c3
  sorry

end boatworks_total_canoes_l165_165743


namespace increased_colored_area_l165_165952

theorem increased_colored_area
  (P : ℝ) -- Perimeter of the original convex pentagon
  (s : ℝ) -- Distance from the points colored originally
  : 
  s * P + π * s^2 = 23.14 :=
by
  sorry

end increased_colored_area_l165_165952


namespace weight_of_new_person_l165_165694

theorem weight_of_new_person (avg_increase : ℝ) (weight_replaced : ℝ) :
  avg_increase = 2.5 → weight_replaced = 70 → 
  let total_increase := 8 * avg_increase in
  let weight_new := weight_replaced + total_increase in
  weight_new = 90 := 
by 
  intros h_avg h_weight;
  unfold total_increase weight_new;
  rw [h_avg, h_weight];
  norm_num;
  sorry

end weight_of_new_person_l165_165694


namespace sum_divisors_of_12_l165_165396

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165396


namespace magnitude_of_b_is_sqrt5_l165_165501

def vec_mag {α : Type*} [LinearOrderedField α] (v : α × α) : α :=
  (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem magnitude_of_b_is_sqrt5 
  (k : ℝ) 
  (a := ((2 - k), 4)) 
  (b := (2, k - 3)) 
  (perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  vec_mag b = Real.sqrt 5 :=
sorry

end magnitude_of_b_is_sqrt5_l165_165501


namespace sum_of_divisors_of_12_l165_165422

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165422


namespace part1_part2_l165_165845

-- Definition of the problem based on the conditions provided
structure Ellipse where
  a : ℝ
  b : ℝ
  (h : a > b ∧ b > 0)
  e : ℝ
  (h_e : e = 1 / 2)
  minor_axis_endpoints : set ℝ

-- Given conditions
def given_ellipse := Ellipse.mk 2 (sqrt 3)
  (by linarith) -- a > b > 0
  (1 / 2)
  (by norm_num [one_div, sqrte_eq_rational] : sqrt (3 / 4) = 1 / 2)
  { A | ∃ x, (A = (x, sqrt 3) ∨ A = (x, -sqrt 3))}

-- Define the problem statement: prove the almost true equation of the ellipse
def ellipse_equation (e : Ellipse) : Prop :=
  ∃ k, ∀ x y, x^2 / (2 * k)^2 + y^2 / (sqrt 3 * k)^2 = 1

theorem part1 (e : Ellipse) (h : e = given_ellipse) : 
  ellipse_equation e := by sorry

-- Part 2: Line MN' and fixed point
structure Line where
  m : ℝ
  n : ℝ

def fixed_point (l : Line) : Prop :=
  ∀ M N N', (N' = (fst N, - snd N)) →
  (l.m * fst M + l.n * snd M = 4) ∧
  (l.m * fst N' + l.n * snd N' = 0)

theorem part2 (l : Line) (h : l.m ≠ 0): 
  fixed_point l := by sorry


end part1_part2_l165_165845


namespace necessary_and_sufficient_condition_l165_165028

variables (x y : ℝ)

theorem necessary_and_sufficient_condition (h1 : x > y) (h2 : 1/x > 1/y) : x * y < 0 :=
sorry

end necessary_and_sufficient_condition_l165_165028


namespace normal_distribution_probability_l165_165480

noncomputable def P_geq_4 : ℝ :=
  0.0228

theorem normal_distribution_probability {xi : ℝ} (h : xi ~ ℕ(1, 9/4)) :
  (P xi (≥ 4)) = P_geq_4 := 
sorry

end normal_distribution_probability_l165_165480


namespace conjugate_of_z_l165_165088

-- Define the given complex number
def z : ℂ := 2 / (1 - I)

-- The main statement to be proven
theorem conjugate_of_z : conj z = 1 - I := by 
  sorry

end conjugate_of_z_l165_165088


namespace largest_distance_12_l165_165582

noncomputable def largest_distance (z : ℂ) (h : |z| = 3) : ℝ :=
  |(1 + 2 * complex.i) * z - z^2|

theorem largest_distance_12 (z : ℂ) (h : |z| = 3) : largest_distance z h = 12 :=
by
  sorry

end largest_distance_12_l165_165582


namespace symmetrical_shapes_congruent_l165_165119

theorem symmetrical_shapes_congruent
  (shapes : Type)
  (is_symmetrical : shapes → shapes → Prop)
  (congruent : shapes → shapes → Prop)
  (symmetrical_implies_equal_segments : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (segment : ℝ), segment_s1 = segment_s2)
  (symmetrical_implies_equal_angles : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (angle : ℝ), angle_s1 = angle_s2) :
  ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → congruent s1 s2 :=
by
  sorry

end symmetrical_shapes_congruent_l165_165119


namespace minimum_treasure_buried_l165_165291

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165291


namespace ball_hits_ground_time_l165_165987

-- The initial equation for the height of the ball
def height (t : ℝ) : ℝ := -16*t^2 - 30*t + 60

-- The statement we want to prove: the time when the ball hits the ground
theorem ball_hits_ground_time : 
  (∃ t : ℝ, height t = 0 ∧ t = (-15 + 5 * real.sqrt 237) / 16) :=
begin
  sorry
end

end ball_hits_ground_time_l165_165987


namespace probability_no_adjacent_seats_l165_165046

theorem probability_no_adjacent_seats :
  let total_ways := Nat.choose 10 3
  let adjacent_2_1 := 9 * 8
  let adjacent_3 := 8
  let ways_adjacent := adjacent_2_1 + adjacent_3
  let p_none_adjacent := 1 - (ways_adjacent / total_ways : ℚ)
  p_none_adjacent = 1 / 3 :=
by
  let total_ways := Nat.choose 10 3
  let adjacent_2_1 := 9 * 8
  let adjacent_3 := 8
  let ways_adjacent := adjacent_2_1 + adjacent_3
  let p_none_adjacent := 1 - (ways_adjacent / total_ways : ℚ)
  have h1 : total_ways = 120 := by simp
  have h2 : adjacent_2_1 = 72 := by simp
  have h3 : adjacent_3 = 8 := by simp
  have h4 : ways_adjacent = 80 := by simp [h2, h3]
  have h5 : p_none_adjacent = (120 - 80) / 120 := by simp [h4, h1]
  simp [h5]
  norm_num [h1]
  done

end probability_no_adjacent_seats_l165_165046


namespace sum_of_arithmetic_progressions_l165_165793

theorem sum_of_arithmetic_progressions :
  let S_p (p : ℕ) := 20 * (80 * p + 78) in
  (∑ p in Finset.range 10, S_p (p + 1)) = 103600 := 
by
  sorry

end sum_of_arithmetic_progressions_l165_165793


namespace sum_fourth_powers_lt_1000_l165_165177

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165177


namespace find_y_in_terms_of_x_l165_165442

theorem find_y_in_terms_of_x (x y : ℝ) (h : sqrt (4 - 5 * x + y) = 9) : y = 77 + 5 * x :=
  sorry

end find_y_in_terms_of_x_l165_165442


namespace part1_part2_l165_165862

def setA : set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def setB (m : ℝ) : set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

theorem part1 (m : ℝ) : (A ∩ B m) = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 :=
sorry

theorem part2 (m : ℝ) : (A ⊆ ᶜ (setB m)) → m < -3 ∨ m > 5 :=
sorry

end part1_part2_l165_165862


namespace auto_group_inequality_l165_165565

/-- Let  (R,+,·)  be a ring with center  Z={a∈ℝ:ar=ra,∀ r∈ℝ} 
with the property that the group  U=U(R)  of its invertible elements is finite.
Given that  G  is the group of automorphisms of the additive group  (R,+), prove that 
|G| ≥ |U|^2 / |Z ∩ U|. -/
theorem auto_group_inequality {R : Type*} [ring R]
  [fintype (units R)] -- U is finite as a set of units
  {Z : set R} (hZ : Z = { a : R | ∀ r : R, a * r = r * a}) 
  (U := {u : units R // true}) -- U is the set of units
  (G := add_aut R) : 
  ∃ G : add_aut R, fintype G ∧ 
  ∀ (ZcapU : set R) (hZcapU : ZcapU = Z ∩ (coe '' U)), 
  ∃ H : subgroup G, 
  ∃ K : subgroup G, 
  ∃ iso_h : fintype.card H = fintype.card U / fintype.card ZcapU, 
  ∃ pseudo_iso : fintype.card K = fintype.card U, 
  ∃ HcapK : ∀ p ∈ H ∩ K, p = 1, 
  fintype.card G ≥ fintype.card H * fintype.card K :=
begin
  -- proof placeholder to be filled
  sorry
end

end auto_group_inequality_l165_165565


namespace f_five_eq_three_f_three_x_inv_f_243_l165_165879

-- Define the function f satisfying the given conditions.
def f (x : ℕ) : ℕ :=
  if x = 5 then 3
  else if x = 15 then 9
  else if x = 45 then 27
  else if x = 135 then 81
  else if x = 405 then 243
  else 0

-- Define the condition f(5) = 3
theorem f_five_eq_three : f 5 = 3 := rfl

-- Define the condition f(3x) = 3f(x) for all x
theorem f_three_x (x : ℕ) : f (3 * x) = 3 * f x :=
sorry

-- Prove that f⁻¹(243) = 405.
theorem inv_f_243 : f (405) = 243 :=
by sorry

-- Concluding the proof statement using the concluded theorems.
example : f (405) = 243 :=
by apply inv_f_243

end f_five_eq_three_f_three_x_inv_f_243_l165_165879


namespace train_passes_man_in_approx_12_seconds_l165_165245

noncomputable def time_to_pass_man (train_length : ℕ) (train_speed_kmph : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := (relative_speed_kmph : ℝ) * (1000 / 3600)
  train_length / relative_speed_mps

theorem train_passes_man_in_approx_12_seconds :
  time_to_pass_man 220 60 6 ≈ 12 := sorry

end train_passes_man_in_approx_12_seconds_l165_165245


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165140

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165140


namespace length_BF_l165_165198

-- Define the geometrical configuration
structure Point :=
  (x : ℝ) (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 6 4.8
def C := Point.mk 12 0
def D := Point.mk 3 (-6)
def E := Point.mk 3 0
def F := Point.mk 6 0

-- Define given conditions
def AE := (3 : ℝ)
def CE := (9 : ℝ)
def DE := (6 : ℝ)
def AC := AE + CE

theorem length_BF : (BF = (72 / 7 : ℝ)) :=
by
  sorry

end length_BF_l165_165198


namespace probability_real_part_greater_l165_165211

/-- Representation of the problem using Lean 4 -/
theorem probability_real_part_greater (x y : Fin 6) (hx : x.val + 1 ∈ {1, 2, 3, 4, 5, 6}) (hy : y.val + 1 ∈ {1, 2, 3, 4, 5, 6}) :
  (set.univ.filter (λ (p : Fin 6 × Fin 6), p.1.val + 1 > p.2.val + 1)).card = 15 ∧
  (set.univ.filter (λ (p : Fin 6 × Fin 6), p.1.val + 1 > p.2.val + 1)).card / (set.univ.filter (λ (p : Fin 6 × Fin 6), true)).card = 5 / 12 := 
by
  sorry

end probability_real_part_greater_l165_165211


namespace ending_number_of_range_divisible_by_five_l165_165106

theorem ending_number_of_range_divisible_by_five
  (first_number : ℕ)
  (number_of_terms : ℕ)
  (h_first : first_number = 15)
  (h_terms : number_of_terms = 10)
  : ∃ ending_number : ℕ, ending_number = first_number + 5 * (number_of_terms - 1) := 
by
  sorry

end ending_number_of_range_divisible_by_five_l165_165106


namespace john_buys_packs_l165_165552

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l165_165552


namespace min_treasures_buried_l165_165269

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165269


namespace min_treasure_signs_buried_l165_165285

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165285


namespace muffin_cost_ratio_l165_165074

theorem muffin_cost_ratio (m b : ℝ) (h1 : 3 * (5 * m + 4 * b) = 3 * m + 20 * b) :
  m = (2 / 3) * b :=
by {
  sorry
}

end muffin_cost_ratio_l165_165074


namespace total_area_correct_l165_165741

-- Define the conversion factors
def feet_to_meters : ℝ := 0.3048
def inches_to_meters : ℝ := 0.0254

-- Define the dimensions of the rooms in feet and inches
structure RoomDimensions where
  feet_length : ℝ
  inches_length : ℝ
  feet_width : ℝ
  inches_width : ℝ

def RoomA : RoomDimensions := { feet_length := 14, inches_length := 8, feet_width := 10, inches_width := 5 }
def RoomB : RoomDimensions := { feet_length := 12, inches_length := 3, feet_width := 11, inches_width := 2 }
def RoomC : RoomDimensions := { feet_length := 9, inches_length := 7, feet_width := 7, inches_width := 10 }

-- Function to convert room dimensions from feet and inches to meters
def convert_to_meters (dim : RoomDimensions) : (ℝ × ℝ) :=
  let length_m := (dim.feet_length * feet_to_meters) + (dim.inches_length * inches_to_meters)
  let width_m := (dim.feet_width * feet_to_meters) + (dim.inches_width * inches_to_meters)
  (length_m, width_m)

-- Function to calculate the area of a room given its dimensions in meters
def area_in_square_meters (length_m : ℝ) (width_m : ℝ) : ℝ :=
  length_m * width_m

-- Calculate total area of all rooms in square meters
noncomputable def total_area : ℝ :=
  let (lengthA, widthA) := convert_to_meters RoomA
  let (lengthB, widthB) := convert_to_meters RoomB
  let (lengthC, widthC) := convert_to_meters RoomC
  area_in_square_meters lengthA widthA + area_in_square_meters lengthB widthB + area_in_square_meters lengthC widthC

-- Theorem statement to prove the total area
theorem total_area_correct : total_area ≈ 33.87 := by
  -- sorry indicates we are skipping the proof.
  sorry

end total_area_correct_l165_165741


namespace solve_log_equation_l165_165069

open Real

-- Define the logarithmic equation condition
def log_equation_condition (x : ℝ) : Prop := log 2 (x^2 - 12 * x) = 5

-- Final theorem statement: prove that solutions to the equation are integers
theorem solve_log_equation : ∃ x1 x2 : ℤ, log_equation_condition x1 ∧ log_equation_condition x2 :=
by
  -- Placeholder for the actual proof
  sorry

end solve_log_equation_l165_165069


namespace m_range_intersection_length_perpendicular_intersection_l165_165846

-- Define a circle
def circle (x y m : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0

-- Define a line
def line (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- The radius condition rewritten as a proof
theorem m_range (c : ℝ) : 
  (∃ x y m, circle x y m ∧ (x - 1)^2 + (y - 2)^2 = c) → 5 - c > 0 :=
sorry

-- Given circle intersects line with length condition of the chord
theorem intersection_length (m : ℝ) (h : ∃ x y, circle x y m ∧ line x y) :
  (∃ x y m', circle x y m' ∧ |√((x - 1)^2 + (y - 2)^2)| = 4 * √(5) / 5) → m = 4 :=
sorry

-- Given circle intersects line, perpendicularity condition of ON and OM
theorem perpendicular_intersection (m : ℝ) (h : ∃ x y, circle x y m ∧ line x y) :
  (∃ x1 x2 y1 y2, x1 = 1 + 2 * y1 - 4 ∧ y1 * y2 + (4 - 2 * y1) * (4 - 2 * y2) = 0 ∧ circle x1 y1 m ∧ circle x2 y2 m) → m = 5 / 8 :=
sorry

end m_range_intersection_length_perpendicular_intersection_l165_165846


namespace original_number_l165_165237

theorem original_number (x y : ℕ) (h1 : odd (3 * x) ∧ 9 ∣ (3 * x)) (h2 : x * y = 108) : x = 27 :=
  sorry

end original_number_l165_165237


namespace sqrt_expression_simplification_l165_165702

theorem sqrt_expression_simplification :
  (real.sqrt ((9^8 + 3^14) / (9^6 + 3^15)) = real.sqrt (15 / 14)) :=
by sorry

end sqrt_expression_simplification_l165_165702


namespace average_age_without_teacher_l165_165081

theorem average_age_without_teacher 
  (A : ℕ) 
  (h : 15 * A + 26 = 16 * (A + 1)) : 
  A = 10 :=
sorry

end average_age_without_teacher_l165_165081


namespace sum_divisors_12_eq_28_l165_165364

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165364


namespace lights_glow_count_l165_165944

theorem lights_glow_count :
  let duration := 4969 in
  let interval_A := 18 in
  let interval_B := 24 in
  let interval_C := 30 in
  let lcm_ABC := Nat.lcm 18 (Nat.lcm 24 30) in

  (duration / interval_A = 276) ∧ 
  (duration / interval_B = 207) ∧ 
  (duration / interval_C = 165) ∧ 
  (duration / lcm_ABC = 13) :=
by
  let duration := 4969
  let interval_A := 18
  let interval_B := 24
  let interval_C := 30
  let lcm_ABC := Nat.lcm 18 (Nat.lcm 24 30)
  have hA : duration / interval_A = 276 := sorry
  have hB : duration / interval_B = 207 := sorry
  have hC : duration / interval_C = 165 := sorry
  have hABC : duration / lcm_ABC = 13 := sorry
  exact ⟨hA, hB, hC, hABC⟩

end lights_glow_count_l165_165944


namespace find_a11_l165_165545

noncomputable def geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ n m k : ℕ, n + m = 2 * k → a n * a m = (a k) ^ 2

theorem find_a11
  (a : ℕ → ℤ)
  (h_geom : geometric_sequence a)
  (h_a5 : a 5 = -9)
  (h_a8 : a 8 = 6) :
  a 11 = -4 :=
by
  have h := h_geom 5 11 8 (by norm_num)
  rw [h_a5, h_a8, sq] at h
  norm_num at h
  exact h.symm

end find_a11_l165_165545


namespace problem1_problem2_problem3_l165_165802

-- Definitions of the function and problem conditions
variables {f : ℝ → ℝ}

-- Condition: Cauchy functional equation
axiom A1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)

-- Given Data
axiom A2 : f 1 = 2
axiom A3 : ∀ x : ℝ, x > 0 → f x < 0

-- The First Problem: Proving f(x) is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := 
by sorry

-- The Second Problem: Proving f(x) is decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := 
by sorry

-- The Third Problem: Solving the inequality
theorem problem3 {x : ℝ} : f (x - 1) - f (1 - 2 * x - x^2) < 4 ↔ x < -4 ∨ x > 1 := 
by sorry

end problem1_problem2_problem3_l165_165802


namespace calculate_LN_l165_165622

theorem calculate_LN (sinN : ℝ) (LM LN : ℝ) (h1 : sinN = 4 / 5) (h2 : LM = 20) : LN = 25 :=
by
  sorry

end calculate_LN_l165_165622


namespace min_stamps_to_50_l165_165744

-- Defining the problem conditions
def is_valid_combination (c f g : ℕ) : Bool :=
  3 * c + 4 * f + 6 * g = 50

def min_num_stamps (c f g : ℕ) : ℕ :=
  c + f + g

-- Stating the theorem for the problem
theorem min_stamps_to_50 : 
  ∃ (c f g : ℕ), is_valid_combination c f g ∧ min_num_stamps c f g = 10 :=
by
  exists 2 2 6
  split
  { sorry }
  { sorry }

end min_stamps_to_50_l165_165744


namespace BoatCrafters_l165_165742

/-
  Let J, F, M, A represent the number of boats built in January, February,
  March, and April respectively.

  Conditions:
  1. J = 4
  2. F = J / 2
  3. M = F * 3
  4. A = M * 3

  Goal:
  Prove that J + F + M + A = 30.
-/

def BoatCrafters.total_boats_built : Nat := 4 + (4 / 2) + ((4 / 2) * 3) + (((4 / 2) * 3) * 3)

theorem BoatCrafters.boats_built_by_end_of_April : 
  BoatCrafters.total_boats_built = 30 :=   
by 
  sorry

end BoatCrafters_l165_165742


namespace expected_stand_ups_theorem_expected_no_stand_up_theorem_l165_165047

-- Define the expected number of times a maiden stands up for other maidens
def expected_stand_ups (n : ℕ) : ℚ :=
  n * (n - 1) / 4

-- Define the harmonic number
def harmonic_number (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (k + 1)

-- Theorem for part (a)
theorem expected_stand_ups_theorem (n : ℕ) : 
  ∑ k in Finset.range n, (n - 1 - k) / 2 = expected_stand_ups n :=
by sorry

-- Theorem for part (b)
theorem expected_no_stand_up_theorem (n : ℕ) : 
  harmonic_number n = 1 + ∑ k in Finset.range n, 1 / (k + 1) :=
by sorry

end expected_stand_ups_theorem_expected_no_stand_up_theorem_l165_165047


namespace largest_fraction_l165_165686

theorem largest_fraction :
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  frac3 > frac1 ∧ frac3 > frac2 ∧ frac3 > frac4 ∧ frac3 > frac5 :=
by
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  sorry

end largest_fraction_l165_165686


namespace jennifer_book_spending_l165_165920

variable (initial_total : ℕ)
variable (spent_sandwich : ℚ)
variable (spent_museum : ℚ)
variable (money_left : ℕ)

theorem jennifer_book_spending :
  initial_total = 90 → 
  spent_sandwich = 1/5 * 90 → 
  spent_museum = 1/6 * 90 → 
  money_left = 12 →
  (initial_total - money_left - (spent_sandwich + spent_museum)) / initial_total = 1/2 :=
by
  intros h_initial_total h_spent_sandwich h_spent_museum h_money_left
  sorry

end jennifer_book_spending_l165_165920


namespace min_treasures_buried_l165_165268

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165268


namespace not_in_second_column_l165_165948

theorem not_in_second_column : ¬∃ (n : ℕ), (1 ≤ n ∧ n ≤ 400) ∧ 3 * n + 1 = 131 :=
by sorry

end not_in_second_column_l165_165948


namespace min_treasure_signs_buried_l165_165286

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165286


namespace rhombus_area_l165_165726

def d1 : ℝ := 36
def d2 : ℝ := 2400
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area : area_of_rhombus d1 d2 = 43200 := by
  rw [area_of_rhombus]
  sorry

end rhombus_area_l165_165726


namespace carolyn_removal_sum_correct_l165_165072

-- Define the initial conditions
def n : Nat := 10
def initialList : List Nat := List.range (n + 1)  -- equals [0, 1, 2, ..., 10]

-- Given that Carolyn removes specific numbers based on the game rules
def carolynRemovals : List Nat := [6, 10, 8]

-- Sum of numbers removed by Carolyn
def carolynRemovalSum : Nat := carolynRemovals.sum

-- Theorem stating the sum of numbers removed by Carolyn
theorem carolyn_removal_sum_correct : carolynRemovalSum = 24 := by
  sorry

end carolyn_removal_sum_correct_l165_165072


namespace collinear_points_condition_l165_165827

-- Given conditions:
variables {V : Type*} [add_comm_group V] [module ℝ V]
variable {a b : V}
variables {λ μ : ℝ}
variable non_collinear : ¬(∃ k : ℝ, a = k • b)

axiom AB_def : ∀ (A B : V), ∃ (λ : ℝ), ∃ b : V, (B - A = λ • a + b)
axiom AC_def : ∀ (A C : V), ∃ a : V, ∃ (μ : ℝ), (C - A = a + μ • b)

-- The goal to prove:
theorem collinear_points_condition (A B C : V) :
  (∃ k : ℝ, B - A = k • (C - A)) ↔ (λ * μ = 1) := sorry

end collinear_points_condition_l165_165827


namespace min_treasure_count_l165_165312

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165312


namespace minimum_treasures_count_l165_165310

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165310


namespace option_b_does_not_round_to_52_l165_165691

/-- Define a predicate function to round a number to the nearest hundredth --/
def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (Real.floor (x * 100 + 0.5)) / 100

/-- Proof Problem Statement --/
theorem option_b_does_not_round_to_52.37 :
  ¬ (round_to_nearest_hundredth 52.375 = 52.37) :=
  sorry

end option_b_does_not_round_to_52_l165_165691


namespace price_of_each_shirt_is_15_30_l165_165550

theorem price_of_each_shirt_is_15_30:
  ∀ (shorts_price : ℝ) (num_shorts : ℕ) (shirt_num : ℕ) (total_paid : ℝ) (discount : ℝ),
  shorts_price = 15 →
  num_shorts = 3 →
  shirt_num = 5 →
  total_paid = 117 →
  discount = 0.10 →
  (total_paid - (num_shorts * shorts_price - discount * (num_shorts * shorts_price))) / shirt_num = 15.30 :=
by 
  sorry

end price_of_each_shirt_is_15_30_l165_165550


namespace mixed_alcohol_solution_l165_165620

theorem mixed_alcohol_solution 
    (vol_x : ℝ) (vol_y : ℝ) (conc_x : ℝ) (conc_y : ℝ) (target_conc : ℝ) (vol_y_given : vol_y = 750) 
    (conc_x_given : conc_x = 0.10) (conc_y_given : conc_y = 0.30) (target_conc_given : target_conc = 0.25) : 
    vol_x = 250 → 
    (conc_x * vol_x + conc_y * vol_y) / (vol_x + vol_y) = target_conc :=
by
  intros h_x
  rw [vol_y_given, conc_x_given, conc_y_given, target_conc_given, h_x]
  sorry

end mixed_alcohol_solution_l165_165620


namespace probability_of_at_least_two_threes_l165_165448

def balls : finset ℕ := {1, 2, 3, 4}

def draws : fin 4 → ℕ := sorry /- Function from the index of draws to the ball number drawn, needs defining -/

def sum_condition (r : vector ℕ 4) : Prop := r.sum = 10

def at_least_two_threes (r : vector ℕ 4) : Prop := (r.filter (λ x => x = 3)).length ≥ 2

theorem probability_of_at_least_two_threes :
  (probability (at_least_two_threes) (sum_condition)) = 1 / 7 :=
sorry

end probability_of_at_least_two_threes_l165_165448


namespace total_cans_collected_l165_165051

theorem total_cans_collected :
  let cans_in_first_bag := 5
  let cans_in_second_bag := 7
  let cans_in_third_bag := 12
  let cans_in_fourth_bag := 4
  let cans_in_fifth_bag := 8
  let cans_in_sixth_bag := 10
  let cans_in_seventh_bag := 15
  let cans_in_eighth_bag := 6
  let cans_in_ninth_bag := 5
  let cans_in_tenth_bag := 13
  let total_cans := cans_in_first_bag + cans_in_second_bag + cans_in_third_bag + cans_in_fourth_bag + cans_in_fifth_bag + cans_in_sixth_bag + cans_in_seventh_bag + cans_in_eighth_bag + cans_in_ninth_bag + cans_in_tenth_bag
  total_cans = 85 :=
by
  sorry

end total_cans_collected_l165_165051


namespace circle_equation_on_x_axis_passing_through_origin_and_tangent_to_y_eq_4_l165_165992

/-- The general equation of a circle whose center is on the x-axis, passes through the origin,
and is tangent to the line y = 4 is x^2 + y^2 ± 8x = 0. -/
theorem circle_equation_on_x_axis_passing_through_origin_and_tangent_to_y_eq_4
  (a R : ℝ) (eq_circle : ∀ x y, (x - a) ^ 2 + y ^ 2 = R ^ 2)
  (passes_through_origin : eq_circle 0 0)
  (tangent_to_y_eq_4 : ∃ (x₀ : ℝ), eq_circle x₀ 4 ∧ (∀ y, eq_circle x₀ y → y = 4)) :
  ∃ a : ℝ, ∀ x y : ℝ, (x^2 + y^2 ± 8 * x = 0) :=
sorry

end circle_equation_on_x_axis_passing_through_origin_and_tangent_to_y_eq_4_l165_165992


namespace question1_question2_l165_165225

open Finset 

noncomputable def choose_internal_specific_surj_exclude (total_internal : ℕ) (total_surgeon : ℕ) 
  (doctors_to_choose : ℕ) (specific_internal_must : ℕ) (specific_surgeon_exclude : ℕ) : ℕ :=
choose (total_internal + total_surgeon - (specific_internal_must + specific_surgeon_exclude)) 
       (doctors_to_choose - specific_internal_must)

theorem question1:
  let total_internal := 12
  let total_surgeon := 8
  let doctors_to_choose := 5
  let specific_internal_must := 1
  let specific_surgeon_exclude := 1 in
  choose_internal_specific_surj_exclude total_internal total_surgeon doctors_to_choose 
    specific_internal_must specific_surgeon_exclude = 3060 := by sorry

noncomputable def choose_both_included (total_internal : ℕ) (total_surgeon : ℕ) 
  (doctors_to_choose : ℕ) : ℕ :=
choose (total_internal + total_surgeon) doctors_to_choose - 
choose total_internal doctors_to_choose - 
choose total_surgeon doctors_to_choose

theorem question2:
  let total_internal := 12
  let total_surgeon := 8
  let doctors_to_choose := 5 in
  choose_both_included total_internal total_surgeon doctors_to_choose = 14656 := by sorry

end question1_question2_l165_165225


namespace find_number_l165_165785

theorem find_number (x : ℝ) (h : x + (2/3) * x + 1 = 10) : x = 27/5 := 
by
  sorry

end find_number_l165_165785


namespace sum_of_solutions_congruence_l165_165126

theorem sum_of_solutions_congruence :
  let S := {x : ℕ | 0 < x ∧ x ≤ 30 ∧ 15 * (3 * x - 4) % 10 = 30 % 10} in
  ∑ x in S, x = 240 :=
by
  sorry

end sum_of_solutions_congruence_l165_165126


namespace max_xy_l165_165581

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 8 * y = 48) : x * y ≤ 18 :=
sorry

end max_xy_l165_165581


namespace trapezoid_CD_length_l165_165006

/-- In trapezoid ABCD with AD parallel to BC and diagonals intersecting:
  - BD = 2
  - ∠DBC = 36°
  - ∠BDA = 72°
  - The ratio BC : AD = 5 : 3

We are to show that the length of CD is 4/3. --/
theorem trapezoid_CD_length
  {A B C D : Type}
  (BD : ℝ) (DBC : ℝ) (BDA : ℝ) (BC_over_AD : ℝ)
  (AD_parallel_BC : Prop) (diagonals_intersect : Prop)
  (hBD : BD = 2) 
  (hDBC : DBC = 36) 
  (hBDA : BDA = 72)
  (hBC_over_AD : BC_over_AD = 5 / 3) 
  :  CD = 4 / 3 :=
by
  sorry

end trapezoid_CD_length_l165_165006


namespace vector_difference_magnitude_l165_165631

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions as hypotheses
def magnitude_a : ℝ := ∥a∥ = 1
def magnitude_b : ℝ := ∥b∥ = 2
def angle_ab : ℝ := real.angle a b = π / 3

-- Desired proof
theorem vector_difference_magnitude :
  magnitude_a a → magnitude_b b → angle_ab a b →
  2 * ∥a - b∥ = 4 := begin
sorry
end

end vector_difference_magnitude_l165_165631


namespace min_value_expression_l165_165009

variables {A B C : ℝ} {O D : ℝ}
variable (R : ℝ)
variable (a b c : ℝ)

-- Assume triangle ABC with given conditions
noncomputable def triangle_ABC_conditions (A B C O D R : ℝ) :=
  (O = circumcenter A B C) ∧
  (D = midpoint A C) ∧
  (dot_product (OB) (OD) = dot_product (OA) (OC))

-- Prove the minimum value of the expression
theorem min_value_expression (h: triangle_ABC_conditions A B C O D R) :
  ∃ x, x = 1 ∧ (∀ y, (y = (sin B / tan A + sin B / tan C)) → x ≤ y) :=
begin
  sorry
end

end min_value_expression_l165_165009


namespace transformed_passes_through_l165_165645

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end transformed_passes_through_l165_165645


namespace min_treasure_count_l165_165314

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165314


namespace quadratic_inequality_exists_l165_165699

variable {α β : ℝ}
variable {f g : ℝ → ℝ}
variable {x x₁ x₂ x₃ x₄ : ℝ}

theorem quadratic_inequality_exists
  (h1 : ∀ x, x₁ < x ∧ x < x₂ → f x < 0)
  (h2 : ∀ x, x₃ < x ∧ x < x₄ → g x < 0)
  (h3 : x₂ < x₃) :
  ∃ α β > 0, ∀ x, α * f x + β * g x > 0 :=
sorry

end quadratic_inequality_exists_l165_165699


namespace compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l165_165026

def a_n (p n : ℕ) : ℕ := (2 * n + 1) ^ p
def b_n (p n : ℕ) : ℕ := (2 * n) ^ p + (2 * n - 1) ^ p

theorem compare_magnitude_p2_for_n1 :
  b_n 2 1 < a_n 2 1 := sorry

theorem compare_magnitude_p2_for_n2 :
  b_n 2 2 = a_n 2 2 := sorry

theorem compare_magnitude_p2_for_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  b_n 2 n > a_n 2 n := sorry

theorem compare_magnitude_p_eq_n_for_all_n (n : ℕ) :
  a_n n n ≥ b_n n n := sorry

end compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l165_165026


namespace not_all_prime_product_plus_one_l165_165011

-- Define the sequence of prime numbers
def prime_seq (n : ℕ) : ℕ := (Nat.primePi n).val

-- The main statement we want to prove
theorem not_all_prime_product_plus_one :
  ∃ n : ℕ, ∃ (ps : Fin n → ℕ), (∀ i : Fin n, Nat.prime (ps i)) ∧ ¬ Nat.prime ((Finset.prod Finset.univ ps) + 1) :=
by
  -- The proof of this theorem is not given, it should show the example
  -- provided in the original solution.
  sorry

end not_all_prime_product_plus_one_l165_165011


namespace H_is_incenter_of_DEF_l165_165465

variable {A B C H D E F : Type} {triangle : Type} [inhabited triangle]

-- Given conditions
def is_obtuse_triangle (ABC : triangle) : Prop := sorry
def orthocenter (ABC triangle) : Type := sorry
def feet_of_altitudes (ABC : triangle) (D E F : Type) : Prop := sorry
def incenter (DEF : triangle) (H : Type) : Prop := sorry

-- Assume triangle ∆ABC is obtuse
axiom ABC_is_obtuse : is_obtuse_triangle triangle

-- Assume points D, E, F are the feet of the altitudes from A, B, C respectively
axiom DEF_feet_of_altitudes : feet_of_altitudes triangle D E F

-- Assume H is the orthocenter of triangle ABC
axiom H_orthocenter : orthocenter triangle = H

-- Proof Statement
theorem H_is_incenter_of_DEF : incenter (triangle D E F) H := sorry

end H_is_incenter_of_DEF_l165_165465


namespace treasure_15_signs_l165_165298

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165298


namespace min_distance_curve_to_line_l165_165546

theorem min_distance_curve_to_line :
  let l := λ x y : ℝ, x - y + 4 = 0
  let C_x := λ α : ℝ, sqrt 3 * cos α
  let C_y := λ α : ℝ, sin α
  let Q := λ α : ℝ, (C_x α, C_y α)
  let distance := λ (x y : ℝ), |x - y + 4| / sqrt 2
  ∃ α : ℝ, ∀ q : ℝ × ℝ, q = Q α → distance q.fst q.snd ≥ sqrt 2 :=
sorry

end min_distance_curve_to_line_l165_165546


namespace budget_remaining_l165_165230

noncomputable def solve_problem : Nat :=
  let total_budget := 325
  let cost_flasks := 150
  let cost_test_tubes := (2 / 3 : ℚ) * cost_flasks
  let cost_safety_gear := (1 / 2 : ℚ) * cost_test_tubes
  let total_expenses := cost_flasks + cost_test_tubes + cost_safety_gear
  total_budget - total_expenses

theorem budget_remaining : solve_problem = 25 := by
  sorry

end budget_remaining_l165_165230


namespace axis_of_symmetry_of_parabola_l165_165084

theorem axis_of_symmetry_of_parabola :
  ∀ (x : ℝ), let eq : ℝ → ℝ := λ x, -(1/2:ℝ) * x^2 + x - (5/2:ℝ) in
    (∃ x0 : ℝ, x0 = 1 ∧ (∀ x, eq x = eq (2 * x0 - x))) :=
by
  sorry

end axis_of_symmetry_of_parabola_l165_165084


namespace div_by_six_l165_165609

theorem div_by_six (n : ℕ) : 6 ∣ (17^n - 11^n) :=
by
  sorry

end div_by_six_l165_165609


namespace abs_sum_a_to_7_l165_165810

-- Sequence definition with domain
def a (n : ℕ) : ℤ := 2 * (n + 1) - 7  -- Lean's ℕ includes 0, so use (n + 1) instead of n here.

-- Prove absolute value sum of first seven terms
theorem abs_sum_a_to_7 : (|a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| = 25) :=
by
  -- Placeholder for actual proof
  sorry

end abs_sum_a_to_7_l165_165810


namespace brownies_count_l165_165964

variable (total_people : Nat) (pieces_per_person : Nat) (cookies : Nat) (candy : Nat) (brownies : Nat)

def total_dessert_needed : Nat := total_people * pieces_per_person

def total_pieces_have : Nat := cookies + candy

def total_brownies_needed : Nat := total_dessert_needed total_people pieces_per_person - total_pieces_have cookies candy

theorem brownies_count (h1 : total_people = 7)
                       (h2 : pieces_per_person = 18)
                       (h3 : cookies = 42)
                       (h4 : candy = 63) :
                       total_brownies_needed total_people pieces_per_person cookies candy = 21 :=
by
  rw [h1, h2, h3, h4]
  sorry

end brownies_count_l165_165964


namespace cally_white_shirts_l165_165748

-- Define the total number of clothes Cally has with W being the number of white shirts
def cally_clothes (W : ℕ) : ℕ := W + 5 + 7 + 6

-- Define the total number of clothes Danny has
def danny_clothes : ℕ := 6 + 8 + 10 + 6

-- Given the total number of clothes they washed together
def total_washed_clothes : ℕ := 58

-- Proof that the number of white shirts Cally washed is 10
theorem cally_white_shirts : ∃ W : ℕ, cally_clothes W + danny_clothes = total_washed_clothes ∧ W = 10 := 
by 
  existsi 10
  simp [cally_clothes, danny_clothes, total_washed_clothes]
  sorry

end cally_white_shirts_l165_165748


namespace finite_non_overlapping_crosses_in_circle_l165_165747

-- Definition of a cross
def cross (side_length : ℝ) := 
  {center : ℝ × ℝ // ∀ x y, (x, y) ∈ center → center = (side_length / 2 √ 2)=}

-- Condition: cross is defined with side length 1
def my_cross := cross 1

-- Given a circle of radius 100
def my_circle_radius : ℝ := 100

-- The theorem statement
theorem finite_non_overlapping_crosses_in_circle :
  ∃ (n : ℕ), ∀ (crosses : fin n → (ℝ × ℝ)),
  (∀ i j, i ≠ j → (crosses i) ∈ my_cross → (crosses j) ∈ my_cross → 
    set.disjoint (crosses i) (crosses j)) →
  set.univ ⊆ ball (0, 0) my_circle_radius →
  set.finite {position : ℝ × ℝ // cross 1 = my_cross} :=
by
  sorry

end finite_non_overlapping_crosses_in_circle_l165_165747


namespace expenditure_representation_correct_l165_165883

-- Define the representation of income
def income_representation (income : ℝ) : ℝ :=
  income

-- Define the representation of expenditure
def expenditure_representation (expenditure : ℝ) : ℝ :=
  -expenditure

-- Condition: an income of 10.5 yuan is represented as +10.5 yuan.
-- We need to prove: an expenditure of 6 yuan is represented as -6 yuan.
theorem expenditure_representation_correct (h : income_representation 10.5 = 10.5) : 
  expenditure_representation 6 = -6 :=
by
  sorry

end expenditure_representation_correct_l165_165883


namespace john_buys_360_packs_l165_165555

def John_buys_packs (classes students_per_class packs_per_student total_packs : ℕ) : Prop :=
  classes = 6 →
  students_per_class = 30 →
  packs_per_student = 2 →
  total_packs = (classes * students_per_class) * packs_per_student
  → total_packs = 360

theorem john_buys_360_packs : John_buys_packs 6 30 2 360 :=
by { intros, sorry }

end john_buys_360_packs_l165_165555


namespace max_possible_x_plus_y_l165_165866

theorem max_possible_x_plus_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : x * y - (x + y) = Nat.gcd x y + Nat.lcm x y) :
  x + y ≤ 10 := sorry

end max_possible_x_plus_y_l165_165866


namespace triangle_ratio_l165_165007

noncomputable def triangle_problem (BC AC : ℝ) (angleC : ℝ) : ℝ :=
  let CD := AC / 2
  let BD := BC - CD
  let HD := BD / 2
  let AD := (3^(1/2)) * CD
  let AH := AD - HD
  (AH / HD)

theorem triangle_ratio (BC AC : ℝ) (angleC : ℝ) (h1 : BC = 6) (h2 : AC = 3 * Real.sqrt 3) (h3 : angleC = Real.pi / 6) :
  triangle_problem BC AC angleC = -2 - Real.sqrt 3 :=
by
  sorry  

end triangle_ratio_l165_165007


namespace factorial_ratio_l165_165076

theorem factorial_ratio :
  ∀ (n : ℕ), n > 0 → (nat.factorial (n - 1)) / (nat.factorial n) = 1 / n :=
by {
  intros n hn,
  rw nat.factorial_succ,
  rw div_eq_mul_inv,
  simp,
}

example : (nat.factorial 23) / (nat.factorial 24) = 1 / 24 :=
begin
  have : 24 > 0 := nat.succ_pos 23,
  convert factorial_ratio 24 this,
  exact nat.factorial_succ 23,
end

end factorial_ratio_l165_165076


namespace sum_of_divisors_of_12_l165_165430

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165430


namespace sum_of_squares_of_perfect_squares_l165_165130

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165130


namespace sum_of_divisors_of_12_l165_165426

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165426


namespace all_iterated_quadratic_eq_have_integer_roots_l165_165982

noncomputable def initial_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, x1 + x2 = -p ∧ x1 * x2 = q

noncomputable def iterated_quadratic_eq_has_integer_roots (p q : ℤ) : Prop :=
  ∀ i : ℕ, i ≤ 9 → ∃ x1 x2 : ℤ, x1 + x2 = -(p + i) ∧ x1 * x2 = (q + i)

theorem all_iterated_quadratic_eq_have_integer_roots :
  ∃ p q : ℤ, initial_quadratic_eq_has_integer_roots p q ∧ iterated_quadratic_eq_has_integer_roots p q :=
sorry

end all_iterated_quadratic_eq_have_integer_roots_l165_165982


namespace f_range_greater_than_2_l165_165040

-- Defining the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x * (x - 1)
  else 2 - f (-x)

-- Defining the sets for which f(x) > 2
def range_of_f_greater_than_2 := {x : ℝ | f x > 2}

-- Stating the theorem as required
theorem f_range_greater_than_2 :
  range_of_f_greater_than_2 = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
by
  -- The proof is omitted
  sorry

end f_range_greater_than_2_l165_165040


namespace sum_of_squares_of_perfect_squares_l165_165132

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165132


namespace sum_divisors_12_eq_28_l165_165363

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165363


namespace angle_between_lines_AB_CD_l165_165792

-- Define the conditions of the problem: Square ABCD and fold along diagonal AC
variable (A B C D : Type) [Square A B C D] (AC : Diagonal A C)

-- The theorem statement representing the problem question and conditions
theorem angle_between_lines_AB_CD (A B C D : Type) [Square A B C D] (AC : Diagonal A C) :
  angle_between_lines_containing A B C D AC = 60 :=
by
  sorry

end angle_between_lines_AB_CD_l165_165792


namespace smallest_b_for_34b_perfect_square_is_4_l165_165681

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l165_165681


namespace exists_k_composite_l165_165611

theorem exists_k_composite (h : Nat) : ∃ k : ℕ, ∀ n : ℕ, 0 < n → ∃ p : ℕ, Prime p ∧ p ∣ (k * 2 ^ n + 1) :=
by
  sorry

end exists_k_composite_l165_165611


namespace swap_increases_div_result_by_100_l165_165000

def swapped_number (n : Nat) (i j : Nat) : Nat :=
  let digits := n.digits 10
  let digits' := digits.swap i j
  digits'.to_nat

theorem swap_increases_div_result_by_100 :
  ∃ (i j : Nat), 952473.div 18 + 100 = (swapped_number 952473 i j).div 18 :=
by
  use 2, 3
  sorry

end swap_increases_div_result_by_100_l165_165000


namespace final_price_correct_percent_decrease_correct_l165_165706

variable (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (sales_tax : ℝ)

def price_after_discounts_and_tax (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let first_discount_amount := original_price * first_discount
  let price_after_first_discount := original_price - first_discount_amount
  let second_discount_amount := price_after_first_discount * second_discount
  let price_after_second_discount := price_after_first_discount - second_discount_amount
  let tax_amount := price_after_second_discount * sales_tax
  let final_price := price_after_second_discount + tax_amount
  Float.round final_price 2

def percent_decrease (original_price : ℝ) (final_price : ℝ) : ℝ :=
  let decrease_amount := original_price - final_price
  (decrease_amount / original_price) * 100

theorem final_price_correct : 
  price_after_discounts_and_tax 72.95 0.10 0.15 0.07 = 59.71 := sorry

theorem percent_decrease_correct : 
  percent_decrease 72.95 59.71 = 23.5 := sorry

end final_price_correct_percent_decrease_correct_l165_165706


namespace exists_distinct_natural_numbers_m_n_p_q_l165_165342

theorem exists_distinct_natural_numbers_m_n_p_q :
  ∃ (m n p q : ℕ), 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q ∧
  (m + n = p + q) ∧ 
  (Real.sqrt m + Real.cbrt n = Real.sqrt p + Real.cbrt q) ∧
  (Real.sqrt m + Real.cbrt n > 2004) := 
sorry

end exists_distinct_natural_numbers_m_n_p_q_l165_165342


namespace main_theorem_l165_165340

noncomputable def harmonic_mean (n : ℕ) (P : ℕ → ℕ) : ℚ := 
  n / (P 1 + P 2 + ... + P n)

def an (n : ℕ) : ℕ := 4 * n - 1
def bn (n : ℕ) : ℚ := (an n + 1) / 4

theorem main_theorem : 
  (∑ n in Finset.range 10, 1 / (bn (n + 1) * bn (n + 2))) = 10/11 := by
  sorry

end main_theorem_l165_165340


namespace cn_equidistant_am_bk_l165_165985

variables {A B C D M N K : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M] [MetricSpace N] [MetricSpace K]

-- Defining conditions
def perpendicular (CD : A) (ABC : B) : Prop := 
  classical.some sorry

def midpoint (M : C) (DB : D) : Prop := 
  classical.some sorry

def midpoint (N : C) (AB : D) : Prop := 
  classical.some sorry

def divides (K : A) (CD : D) (ratio : ℕ) : Prop := 
  classical.some sorry

-- The proposition to prove
theorem cn_equidistant_am_bk 
  (h1 : perpendicular CD ABC) 
  (h2 : midpoint M DB) 
  (h3 : midpoint N AB) 
  (h4 : divides K CD 1 2) : 
  equidistant CN AM BK :=
sorry

end cn_equidistant_am_bk_l165_165985


namespace sum_of_divisors_of_12_l165_165427

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165427


namespace sum_of_divisors_of_12_l165_165390

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165390


namespace solution_of_equation_of_plane_l165_165356

def solveEquationOfPlane : Prop :=
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.gcd (Int.gcd (|A|) (|B|)) (|C|)) (|D|) = 1 ∧
  A = 1 ∧ B = -7 ∧ C = 8 ∧ D = 147 ∧
  ∀ x y z : ℝ, (A * x + B * y + C * z + D = 0) ↔ 
    (x = 1 ∧ y = 4 ∧ z = -3) ∨
    ∃ t : ℝ, (x = 4 * t + 2 ∧ y = -2 * t - 1 ∧ z = 3 * t + 3)

theorem solution_of_equation_of_plane : solveEquationOfPlane := 
by 
s_ry

end solution_of_equation_of_plane_l165_165356


namespace triangle_sine_law_l165_165523

theorem triangle_sine_law (A B a b : ℝ) 
  (hA : A = real.pi / 3)
  (ha : a = real.sqrt 3)
  (h_sine_law : a / real.sin A = b / real.sin B) :
  (a + b) / (real.sin A + real.sin B) = 2 :=
by
  sorry

end triangle_sine_law_l165_165523


namespace exists_integer_solutions_l165_165608

theorem exists_integer_solutions (n : ℕ) (h : n > 1) :
  ∃ (x : Fin (n+1) → ℤ), 
    (∑ i in Finset.univ.erase (Fin.last n), 1/(x i)^2) = 1/(x (Fin.last n))^2 := 
sorry

end exists_integer_solutions_l165_165608


namespace triangle_ABC_is_right_angled_l165_165815

noncomputable def A : ℝ × ℝ := (1, 2)
noncomputable def point_on_line : ℝ × ℝ := (5, -2)
noncomputable def parabola (x : ℝ) : ℝ := 2 * real.sqrt x

theorem triangle_ABC_is_right_angled
  (exists_BC : ∃ t s : ℝ, parabola (t^2) = 2*t ∧ parabola (s^2) = 2*s ∧
                           (5 - t^2)/(2 - 2*t) = (5 - s^2)/(2 - 2*s) ∧
                           A.1 = (t^2 + s^2) / 2 ∧ A.2 = (2*t + 2*s) / 2) :
  ∃ B C : ℝ × ℝ, (B = (t^2, 2*t) ∧ C = (s^2, 2*s) ∧  
                   ((B.2 - point_on_line.2) / (B.1 - point_on_line.1)) *
                   ((C.2 - point_on_line.2) / (C.1 - point_on_line.1)) = -1) :=
sorry

end triangle_ABC_is_right_angled_l165_165815


namespace exponentiation_problem_l165_165654

theorem exponentiation_problem :
  (-0.125 ^ 2003) * (-8 ^ 2004) = -8 := 
sorry

end exponentiation_problem_l165_165654


namespace correct_positioning_l165_165003

namespace PuzzleSolution

def Square := {A, B, C, D, E, F, G}
def Position := Array 9 (Option Nat)

noncomputable def arrows : Position := 
  #[some 1, some 2, some 4, some 5, some 3, some 8, some 7, none, some 9]

theorem correct_positioning : 
  ∀ (p : Position), 
    (p.getD 1 none = some 1 ∧
    p.getD 2 none = some 2 ∧
    p.getD 3 none = some 4 ∧
    p.getD 4 none = some 5 ∧
    p.getD 5 none = some 3 ∧
    p.getD 6 none = some 8 ∧
    p.getD 7 none = some 7 ∧
    p.getD 8 none = some 9) →
  ∃ (A B C D E F G : Square),
    (p.getD A none = some 6) ∧
    (p.getD B none = some 2) ∧
    (p.getD C none = some 4) ∧
    (p.getD D none = some 5) ∧
    (p.getD E none = some 3) ∧
    (p.getD F none = some 8) ∧
    (p.getD G none = some 7) :=
  sorry

end PuzzleSolution

end correct_positioning_l165_165003


namespace max_f_val_solve_a_maximum_value_proof_l165_165491

noncomputable def f (x a : ℝ) : ℝ := (Real.log x) / (x + a)

theorem max_f_val (a : ℝ) (h_tangent : ∀ x, deriv (λ x, f x a) 1 = 1)
  (x : ℝ) : x ∈ Set.Ioo (0 : ℝ) (Real.exp 1) ∨ x ∈ Set.Ioi (Real.exp 1) :=
sorry

theorem solve_a (a : ℝ) (h_tangent : ∀ x, deriv (λ x, f x a) 1 = 1) : a = 0 :=
sorry

theorem maximum_value (h_tangent : ∀ x, deriv (λ x, f x 0) 1 = 1) : ∃ x, f x 0 = 1 / (Real.exp 1) :=
sorry

theorem proof (a : ℝ) :
  (∀ x, deriv (λ x, f x a) 1 = 1) →
  ∃ x, f x 0 = 1 / (Real.exp 1) :=
begin
  intro h_tangent,
  have h_a : a = 0 := solve_a a h_tangent,
  rw [h_a],
  exact maximum_value h_tangent,
end

end max_f_val_solve_a_maximum_value_proof_l165_165491


namespace comparison_l165_165991

-- Definitions and conditions
def f (x : ℝ) : ℝ := sorry
axiom differentiable_f : ∀ x : ℝ, differentiable ℝ f
axiom symmetric_f : ∀ x : ℝ, f(x) = f(2 - x)
axiom decreasing_f : ∀ x : ℝ, x < 1 → deriv f x < 0

-- Definitions of a, b, c
def a := f(0)
def b := f(1 / 2)
def c := f(3)

-- Theorem to prove
theorem comparison (a b c : ℝ) : c < a ∧ a < b :=
by
  -- Placeholder for the actual proof
  sorry

end comparison_l165_165991


namespace min_treasure_signs_buried_l165_165284

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165284


namespace sum_positive_integral_values_l165_165384

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165384


namespace sum_positive_integral_values_l165_165380

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165380


namespace system_of_equations_solution_l165_165070

theorem system_of_equations_solution :
  ∃ x y z : ℚ, 3 * x - 4 * y = 12 ∧ -5 * x + 6 * y - z = 9 ∧ x + 2 * y + 3 * z = 0 ∧
             x = -262 / 75 ∧ y = -2075 / 200 ∧ z = -105 / 100 :=
by
  have h1 : 3 * (-262 / 75 : ℚ) - 4 * (-2075 / 200 : ℚ) = 12 := by sorry,
  have h2 : -5 * (-262 / 75 : ℚ) + 6 * (-2075 / 200 : ℚ) - (-105 / 100 : ℚ) = 9 := by sorry,
  have h3 : (-262 / 75 : ℚ) + 2 * (-2075 / 200 : ℚ) + 3 * (-105 / 100 : ℚ) = 0 := by sorry,
  use [-262 / 75, -2075 / 200, -105 / 100],
  exact ⟨h1, h2, h3, rfl, rfl, rfl⟩

end system_of_equations_solution_l165_165070


namespace time_to_hit_plane_l165_165727

-- Given conditions
def plane_speed := 1000 -- km/h
def missile_speed := 1000 -- km/h
def circle_radius := 10 -- km

-- Proof problem statement
theorem time_to_hit_plane (plane_speed missile_speed circle_radius : ℝ) (h_plane_speed : plane_speed = 1000) (h_missile_speed : missile_speed = 1000) (h_circle_radius : circle_radius = 10) : 
  let time := (1/4 * 2 * real.pi * circle_radius) / plane_speed * 3600 in 
  time = 18 * real.pi :=
by {
  sorry
}

end time_to_hit_plane_l165_165727


namespace vincent_back_to_A_after_5_min_p_plus_q_computation_l165_165705

def probability (n : ℕ) : ℚ :=
  if n = 0 then 1
  else 1 / 4 * (1 - probability (n - 1))

theorem vincent_back_to_A_after_5_min : 
  probability 5 = 51 / 256 :=
by sorry

theorem p_plus_q_computation :
  51 + 256 = 307 :=
by linarith

end vincent_back_to_A_after_5_min_p_plus_q_computation_l165_165705


namespace ratio_of_frank_to_joystick_l165_165120

-- Define the costs involved
def cost_table : ℕ := 140
def cost_chair : ℕ := 100
def cost_joystick : ℕ := 20
def diff_spent : ℕ := 30

-- Define the payments
def F_j := 5
def E_j := 15

-- The ratio we need to prove
def ratio_frank_to_total_joystick (F_j : ℕ) (total_joystick : ℕ) : (ℕ × ℕ) :=
  (F_j / Nat.gcd F_j total_joystick, total_joystick / Nat.gcd F_j total_joystick)

theorem ratio_of_frank_to_joystick :
  let F_j := 5
  let total_joystick := 20
  ratio_frank_to_total_joystick F_j total_joystick = (1, 4) := by
  sorry

end ratio_of_frank_to_joystick_l165_165120


namespace angle_between_lines_is_25_degrees_l165_165543

theorem angle_between_lines_is_25_degrees
  (A B C D E F : Point)
  (hACD : ∠ A C D = 90)
  (hECB : ∠ E C B = 65)
  (hDF : F ∈ AB) :
  ∠ D C F = 25 :=
by
  sorry

end angle_between_lines_is_25_degrees_l165_165543


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165157

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165157


namespace min_treasures_buried_l165_165265

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165265


namespace percent_of_volume_filled_by_cubes_l165_165725

theorem percent_of_volume_filled_by_cubes :
  let box_width := 8
  let box_height := 6
  let box_length := 12
  let cube_size := 2
  let box_volume := box_width * box_height * box_length
  let cube_volume := cube_size ^ 3
  let num_cubes := (box_width / cube_size) * (box_height / cube_size) * (box_length / cube_size)
  let cubes_volume := num_cubes * cube_volume
  (cubes_volume / box_volume : ℝ) * 100 = 100 := by
  sorry

end percent_of_volume_filled_by_cubes_l165_165725


namespace sum_of_divisors_of_12_l165_165440

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165440


namespace min_value_at_constraints_l165_165825

open Classical

noncomputable def min_value (x y : ℝ) : ℝ := (x^2 + y^2 + x) / (x * y)

def constraints (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x + 2 * y = 1

theorem min_value_at_constraints : 
∃ (x y : ℝ), constraints x y ∧ min_value x y = 2 * Real.sqrt 2 + 2 :=
by
  sorry

end min_value_at_constraints_l165_165825


namespace least_common_multiple_of_a_and_c_l165_165996

open Nat

theorem least_common_multiple_of_a_and_c (a b c : ℕ) 
  (h₁ : lcm a b = 40) 
  (h₂ : lcm b c = 21) : 
  lcm a c = 24 :=
sorry

end least_common_multiple_of_a_and_c_l165_165996


namespace sum_expression_correct_l165_165329

theorem sum_expression_correct : 
  (∑ k in Finset.range 2014, k.succ * (k + 2)) / ((∑ k in Finset.range 2014.succ, k + 1) * (1 / 5)) = 6710 :=
by
  sorry

end sum_expression_correct_l165_165329


namespace cos_expr_l165_165513

theorem cos_expr (α : ℝ) (h : sin (π / 6 - α) = 1 / 3) : cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_expr_l165_165513


namespace sum_of_two_longest_altitudes_of_triangle_l165_165876

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  if a^2 + b^2 = c^2 then (1 / 2) * a * b else 0

noncomputable def altitude (area hypotenuse : ℝ) : ℝ :=
  (2 * area) / hypotenuse

theorem sum_of_two_longest_altitudes_of_triangle (a b c : ℝ) (h : a = 9) (i : b = 40) (j : c = 41):
  let area := triangle_area a b c in
  let hypotenuse := c in
  let alt_to_hypotenuse := altitude area hypotenuse in
  a + b = 49 :=
by
  sorry

end sum_of_two_longest_altitudes_of_triangle_l165_165876


namespace parabola_eq_for_focus_directrix_distance_minimum_value_of_t_l165_165860

noncomputable def parabola_focus_directrix_distance : ℝ :=
1 / 2

def parabola_equation (x : ℝ) : ℝ :=
x^2

theorem parabola_eq_for_focus_directrix_distance :
  (∃ p > 0, ∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y) :=
begin
  use [1/2],
  split,
  { linarith },
  { intros x y,
    split; intro h,
    rw [mul_comm] at h,
    linarith,
    rw [mul_comm, ←h],
    linarith },
end

theorem minimum_value_of_t (t : ℝ) (h_t : t > 0) :
  (t = 2 / 3) :=
sorry


end parabola_eq_for_focus_directrix_distance_minimum_value_of_t_l165_165860


namespace winning_candidate_percentage_l165_165535

theorem winning_candidate_percentage :
  let votes := [12136.5, 17636.8, 23840.1, 19568.4, 17126.6, 20640.2, 26228.9, 19874.3, 21568.7]
  let total_votes := votes.sum
  let winning_votes := (List.maximum? votes).get_or_else 0
  (winning_votes / total_votes) * 100 ≈ 14.68 := 
by
  sorry

end winning_candidate_percentage_l165_165535


namespace min_treasures_buried_l165_165326

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165326


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165141

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165141


namespace problem_N_lowest_terms_l165_165444

theorem problem_N_lowest_terms :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 2500 ∧ ∃ k : ℕ, k ∣ 128 ∧ (n + 11) % k = 0 ∧ (Nat.gcd (n^2 + 7) (n + 11)) > 1) →
  ∃ cnt : ℕ, cnt = 168 :=
by
  sorry

end problem_N_lowest_terms_l165_165444


namespace point_q_location_l165_165703

variable (m : ℝ)

theorem point_q_location (h : m < 0) : -m > 0 ∧ (1 : ℝ) = 1 :=
by
  split
  { sorry }
  { refl }

end point_q_location_l165_165703


namespace bivariate_polynomial_form_l165_165350

variable {R : Type*} [Field R]

noncomputable def bivariate_polynomial (n : ℕ) (f : R → R → R) :=
  ∀ (x y : R), f (x + 1) (y + 1) = f x y

theorem bivariate_polynomial_form (n : ℕ) (f : R → R → R) (h : bivariate_polynomial n f) :
  ∃ (a : fin (n + 1) → R), a n ≠ 0 ∧ f = (λ (x y : R), ∑ k in finset.range (n + 1), a k * (x - y)^k) :=
  sorry

end bivariate_polynomial_form_l165_165350


namespace percentage_of_male_students_l165_165894

theorem percentage_of_male_students :
  ∃ M F : ℝ, M + F = 1 ∧ 0.60 * M + 0.80 * F = 0.72 ∧ M = 0.40 :=
by
  -- Definitions of M and F
  let M := 0.40
  let F := 0.60  -- Derived from 1 - M
  use M, F
  -- Conditions
  split
  -- First condition
  · exact add_self_div_two 1
  -- Second condition
  split
  -- Simplification
  · calc
    0.60 * 0.40 + 0.80 * 0.60 = 0.24 + 0.48 := by ring
    ... = 0.72 := by linarith
  -- Proven percentage of male students
  exact eq.refl 0.40

end percentage_of_male_students_l165_165894


namespace odd_function_f_expression_solve_inequality_l165_165831

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 + 2*x else -x^2 + 2*x

theorem odd_function_f_expression :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x > 0 → f x = x^2 + 2*x) →
  (∀ x : ℝ, f x = if x >= 0 then x^2 + 2*x else -x^2 + 2*x) :=
by
  intros h_odd h_pos
  ext x
  sorry

theorem solve_inequality : 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∀ x : ℝ, x > 0 → f x = x^2 + 2*x) → 
  (∀ x : ℝ, f (2*x - 1) + f (x + 1) ≤ 0 ↔ x ≤ 0) :=
by
  intros h_odd h_pos
  sorry

end odd_function_f_expression_solve_inequality_l165_165831


namespace epidemic_control_condition_met_l165_165769

theorem epidemic_control_condition_met :
  (∀ (x : ℕ → ℕ), (∀ i, x i ≤ 5) → 
    average x 7 ≤ 3 ∧ range x 7 ≤ 2 ∨ (mode x 7 = 1 ∧ range x 7 ≤ 1)) := 
begin
  sorry
end

end epidemic_control_condition_met_l165_165769


namespace grasshopper_jump_distance_l165_165994

theorem grasshopper_jump_distance (g f m : ℕ)
    (h1 : f = g + 32)
    (h2 : m = f - 26)
    (h3 : m = 31) : g = 25 :=
by
  sorry

end grasshopper_jump_distance_l165_165994


namespace albania_inequality_l165_165568

variable (a b c r R s : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)
variable (h4 : r > 0)
variable (h5 : R > 0)
variable (h6 : s = (a + b + c) / 2)

theorem albania_inequality :
    1 / (a + b) + 1 / (a + c) + 1 / (b + c) ≤ r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s) :=
sorry

end albania_inequality_l165_165568


namespace exists_zero_in_interval_l165_165094

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem exists_zero_in_interval :
  ∃ c ∈ set.Ioo 0 1, f c = 0 :=
begin
  -- Insert proof here
  sorry
end

end exists_zero_in_interval_l165_165094


namespace smallest_b_for_34b_perfect_square_is_4_l165_165682

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l165_165682


namespace distinction_percentage_l165_165897

noncomputable def totalCandidates := 2500
noncomputable def girls := 1100
noncomputable def boys := totalCandidates - girls

noncomputable def boysPassed := (35 / 100) * boys
noncomputable def girlsPassed := (40 / 100) * girls

noncomputable def boysDistinction := (15 / 100) * boysPassed
noncomputable def girlsDistinction := (25 / 100) * girlsPassed

noncomputable def totalDistinction := boysDistinction + girlsDistinction

noncomputable def totalPercentageDistinction := (totalDistinction / totalCandidates) * 100

theorem distinction_percentage :
  totalPercentageDistinction = 7.32 := 
  sorry

end distinction_percentage_l165_165897


namespace sum_of_divisors_of_12_l165_165373

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165373


namespace extreme_value_at_one_monotonicity_of_f_l165_165851

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 1 + a) / x - a * real.log x

-- (Ⅰ) Prove that when a = 1, the function f(x) has a local minimum at x = sqrt(2)
-- with the value f(sqrt(2)) = sqrt(2) + 3/2 - 1/2 * log 2.
theorem extreme_value_at_one {x : ℝ} (hxpos : 0 < x) :
    let a := 1 in f (real.sqrt 2) a = real.sqrt 2 + 3 / 2 - 1 / 2 * real.log 2 :=
  sorry

-- (Ⅱ) Prove the monotonicity of the function f(x) for general a
theorem monotonicity_of_f (a: ℝ) :
  ((a ≤ -1) → (∀ x > 0, ∀ y > x, f y a ≥ f x a)) ∧
  ((a > -1) →
    (∀ x > 0, (x < 1 + a → ∀ y > x, f y a ≤ f x a) ∧ (x > 1 + a → ∀ y > x, f y a ≥ f x a))) :=
  sorry

end extreme_value_at_one_monotonicity_of_f_l165_165851


namespace monotone_f_range_a_l165_165990

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

theorem monotone_f_range_a (a : ℝ) :
  (∀ (x y : ℝ), x <= y → f a x >= f a y) →
  1 / 2 <= a ∧ a <= 5 / 8 :=
sorry

end monotone_f_range_a_l165_165990


namespace determine_a_l165_165519

noncomputable def z (a : ℝ) : ℂ := (1 + a * Complex.I) / (2 - Complex.I)

def is_pure_imaginary (z : ℂ) : Prop := 
  z.re = 0 ∧ z.im ≠ 0

theorem determine_a (a : ℝ) : is_pure_imaginary (z a) → a = 2 :=
begin
  sorry,
end

end determine_a_l165_165519


namespace sum_divisors_12_eq_28_l165_165361

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165361


namespace find_250th_term_l165_165124

-- Define a predicate for perfect squares
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a predicate for multiples of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Define a sequence of positive integers omitting perfect squares and multiples of 3
def valid_sequence (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x, ¬ is_perfect_square x ∧ ¬ is_multiple_of_3 x)

-- Prove that the 250th term of this sequence is 362
theorem find_250th_term : Finset.nth (valid_sequence 362) 249 = some 362 := 
sorry

end find_250th_term_l165_165124


namespace last_digit_largest_prime_l165_165983

-- Definition and conditions
def largest_known_prime : ℕ := 2^216091 - 1

-- The statement of the problem we want to prove
theorem last_digit_largest_prime : (largest_known_prime % 10) = 7 := by
  sorry

end last_digit_largest_prime_l165_165983


namespace office_light_ratio_l165_165597

theorem office_light_ratio (bedroom_light: ℕ) (living_room_factor: ℕ) (total_energy: ℕ) 
  (time: ℕ) (ratio: ℕ) (office_light: ℕ) :
  bedroom_light = 6 →
  living_room_factor = 4 →
  total_energy = 96 →
  time = 2 →
  ratio = 3 →
  total_energy = (bedroom_light * time) + (office_light * time) + ((bedroom_light * living_room_factor) * time) →
  (office_light / bedroom_light) = ratio :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  -- The actual solution steps would go here
  sorry

end office_light_ratio_l165_165597


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165155

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165155


namespace total_ranking_sequences_l165_165740

-- Define teams
inductive Team
| A | B | C | D

-- Define the conditions
def qualifies (t : Team) : Prop := 
  -- Each team must win its qualifying match to participate
  true

def plays_saturday (t1 t2 t3 t4 : Team) : Prop :=
  (t1 = Team.A ∧ t2 = Team.B) ∨ (t3 = Team.C ∧ t4 = Team.D)

def plays_sunday (t1 t2 t3 t4 : Team) : Prop := 
  -- Winners of Saturday's matches play for 1st and 2nd, losers play for 3rd and 4th
  true

-- Lean statement for the proof problem
theorem total_ranking_sequences : 
  (∀ t : Team, qualifies t) → 
  (∀ t1 t2 t3 t4 : Team, plays_saturday t1 t2 t3 t4) → 
  (∀ t1 t2 t3 t4 : Team, plays_sunday t1 t2 t3 t4) → 
  ∃ n : ℕ, n = 16 :=
by 
  sorry

end total_ranking_sequences_l165_165740


namespace min_treasure_signs_buried_l165_165281

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165281


namespace part1_part2_l165_165817

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 1 < x ∧ x < 5 }
def setC (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 3 }

-- part (1)
theorem part1 (x : ℝ) : (x ∈ setA ∨ x ∈ setB) ↔ (-2 ≤ x ∧ x < 5) :=
sorry

-- part (2)
theorem part2 (a : ℝ) : ((setA ∩ setC a) = setC a) ↔ (a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2)) :=
sorry

end part1_part2_l165_165817


namespace max_points_of_intersection_of_fifth_degree_polynomials_l165_165125

theorem max_points_of_intersection_of_fifth_degree_polynomials (p q : ℝ[X]) 
  (hp : p.degree = 5) (hq : q.degree = 5) 
  (hpc : p.leading_coeff = 1) (hqc : q.leading_coeff = 1) 
  (hne : p ≠ q) : 
  (p - q).natDegree ≤ 4 := 
sorry

end max_points_of_intersection_of_fifth_degree_polynomials_l165_165125


namespace integral_unit_circle_half_area_l165_165772

theorem integral_unit_circle_half_area :
  ∫ x in -1..1, sqrt (1 - x^2) = Real.pi / 2 :=
  sorry

end integral_unit_circle_half_area_l165_165772


namespace johns_total_packs_l165_165559

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l165_165559


namespace john_buys_packs_l165_165553

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l165_165553


namespace sum_divisors_of_12_l165_165402

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165402


namespace original_price_l165_165100

theorem original_price (P : ℝ) (h1 : P + 0.10 * P = 330) : P = 300 := 
by
  sorry

end original_price_l165_165100


namespace cookie_radius_l165_165075

theorem cookie_radius (x y : ℝ) :
    (x^2 + y^2 + 26 = 6 * x + 12 * y) → ∃ r, r = sqrt 19 :=
by
  sorry

end cookie_radius_l165_165075


namespace cube_faces_divide_space_l165_165341

theorem cube_faces_divide_space (planes : Fin 6 → Plane) (cube : IsCube (Set.Range planes)) :
  number_of_regions planes = 27 := 
sorry

end cube_faces_divide_space_l165_165341


namespace sum_of_fourth_powers_less_than_1000_l165_165185

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165185


namespace least_number_four_digits_divisible_by_15_25_40_75_l165_165696

noncomputable def least_four_digit_multiple : ℕ :=
  1200

theorem least_number_four_digits_divisible_by_15_25_40_75 :
  (∀ n, (n ∣ 15) ∧ (n ∣ 25) ∧ (n ∣ 40) ∧ (n ∣ 75)) → least_four_digit_multiple = 1200 :=
sorry

end least_number_four_digits_divisible_by_15_25_40_75_l165_165696


namespace circumcenter_property_l165_165049

theorem circumcenter_property (A B C O D E F : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space O] [metric_space D] [metric_space E] [metric_space F]
  [circumcenter A B C O]
  (circumdistance_eq : ∀ {P Q : Type} [metric_space P] [metric_space Q], distance O P = distance O Q)
  (lines_intersect : ∀ {P : Type} [metric_space P], intersects AO P D) : 
  (1 / distance A D) + (1 / distance B E) + (1 / distance C F) = 2 / distance A O :=
by 
  sorry

end circumcenter_property_l165_165049


namespace correct_idiom_l165_165109

-- Define the conditions given in the problem
def context := "The vast majority of office clerks read a significant amount of materials"
def idiom_usage := "to say _ of additional materials"

-- Define the proof problem
theorem correct_idiom (context: String) (idiom_usage: String) : idiom_usage.replace "_ of additional materials" "nothing of newspapers and magazines" = "to say nothing of newspapers and magazines" :=
sorry

end correct_idiom_l165_165109


namespace tangent_segment_length_l165_165728

theorem tangent_segment_length {base height tangent_length : ℝ} 
  (h1 : base = 12) 
  (h2 : height = 8) 
  (h3 : tangent_length = 3)
  (isosceles : ∃ (A B C : Type) (Triangle_ABC : isosceles ⟨A, B, C⟩), ∀ M : Type, M = midpoint(B, C) → height(A, M) ∧ base(B, C) = base) :
  tangent_length = 3 :=
by
  sorry

end tangent_segment_length_l165_165728


namespace sum_of_faces_vertices_edges_l165_165603

theorem sum_of_faces_vertices_edges (faces_prism edges_prism vertices_prism faces_pyramid edges_pyramid vertices_pyramid : ℕ) :
  faces_prism = 8 →
  edges_prism = 12 →
  vertices_prism = 12 →
  faces_pyramid = 6 →
  edges_pyramid = 6 →
  vertices_pyramid = 1 →
  (faces_prism - 1 + faces_pyramid) + (edges_prism + edges_pyramid) + (vertices_prism + vertices_pyramid) = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end sum_of_faces_vertices_edges_l165_165603


namespace treasure_under_minimum_signs_l165_165276

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165276


namespace find_line_eqns_l165_165833

-- Definitions
def passes_through (P : ℝ × ℝ) (l : ℝ → ℝ) : Prop := ∃ x y : ℝ, l x = y ∧ P = (x, y)
def circle (C : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop := (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2
def chord_length (C : ℝ × ℝ) (r : ℝ) (l : ℝ → ℝ) (d : ℝ) : Prop := ∃ c1 c2 : ℝ × ℝ,
  passes_through c1 l ∧ passes_through c2 l ∧ circle C r c1 ∧ circle C r c2 ∧ dist c1 c2 = d

-- Given conditions
def P := (2, 4) : ℝ × ℝ
def C := (1, 2) : ℝ × ℝ
def radius := sqrt 10
def chord_len := 6

-- The equations to be proved
theorem find_line_eqns : 
  ∃ (l1 l2 : ℝ → ℝ), 
  (passes_through P l1 ∧ chord_length C radius l1 chord_len ∧
    ∀ x : ℝ, l1 x = (3 * x - 10) / 4) ∧
  (passes_through P l2 ∧ chord_length C radius l2 chord_len ∧
    ∀ x : ℝ, l2 x = 0 ∧ x = 2) := by
  sorry

end find_line_eqns_l165_165833


namespace miaomiao_primary_school_scores_l165_165770

theorem miaomiao_primary_school_scores :
  let scores := [233, 132, 127, 91, 112, 115, 181, 124, 91]
  (list.sum scores / list.length scores = 134) ∧ (list.median scores = 124) :=
by
  let scores := [233, 132, 127, 91, 112, 115, 181, 124, 91]
  have h_mean : list.sum scores / list.length scores = 134 := sorry
  have h_median : list.median scores = 124 := sorry
  exact ⟨h_mean, h_median⟩

end miaomiao_primary_school_scores_l165_165770


namespace part1_part2_part3_l165_165572

-- Part (1)
theorem part1 (A : Set ℕ) (hA : A = {1, 3, 6}) : 
  (Set (abs (u - v)) | u, v ∈ A, u ≠ v) = {2, 3, 5} :=
by sorry

-- Part (2)
theorem part2 (A : Set ℕ) (hA_size : A.card = 5) :
  ∃ (B : Set ℕ) (hB : B = (Set (abs (u - v)) | u, v ∈ A, u ≠ v)), B.card = 4 :=
by sorry

-- Part (3)
theorem part3 : ¬∃ (A : Set ℕ) (hA : A.card = 4), 
  (Set (abs (u - v)) | u, v ∈ A, u ≠ v) = {2, 3, 5, 6, 10, 16} :=
by sorry

end part1_part2_part3_l165_165572


namespace range_of_a_l165_165989

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ a then sin x else 1 / x

theorem range_of_a (a : ℝ) : (∀ x, -1 ≤ f a x ∧ f a x ≤ 1) ↔ 1 ≤ a :=
by
  intro h
  sorry

end range_of_a_l165_165989


namespace circle_intersection_l165_165902

theorem circle_intersection (x : ℝ) :
  let center := (6, 0)
  let r : ℝ := 12
  let eq_circle := λ x y, (x - center.1)^2 + y^2 = r^2
  eq_circle x 10 ↔ x = 6 + 2*Real.sqrt 11 ∨ x = 6 - 2*Real.sqrt 11 :=
by
  let center := (6, 0)
  let r : ℝ := 12
  let eq_circle := λ x y, (x - center.1)^2 + y^2 = r^2
  sorry

end circle_intersection_l165_165902


namespace transformed_passes_through_l165_165644

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end transformed_passes_through_l165_165644


namespace sum_of_divisors_of_12_l165_165395

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165395


namespace number_of_geese_l165_165953

theorem number_of_geese (A x n k : ℝ) 
  (h1 : A = k * x * n)
  (h2 : A = (k + 20) * x * (n - 75))
  (h3 : A = (k - 15) * x * (n + 100)) 
  : n = 300 :=
sorry

end number_of_geese_l165_165953


namespace oleg_cannot_obtain_39699_l165_165598

def points_on_Ox := set.univ.Icc (0 : ℕ) 100

structure quadratic_function (a b : ℝ) :=
(point_a : a ∈ points_on_Ox)
(point_b : b ∈ points_on_Ox)
(f : ℚ → ℚ)
(touches : ∀ x : ℚ, f x = -1 → ∃ c : ℚ, c ∈ set.range (λ x, f x))
(passes_through : ∀ x : ℚ, x ∈ points_on_Ox → f x = x * x + b * x + a)

def sum_intersections_is_39699 : Prop :=
∃ (fs : finset (quadratic_function 0 1)), (fs.card = 200) ∧
  sum (λ (pair : quadratic_function 0 1 × quadratic_function 0 1), 
    (pair.1.f = pair.2.f : ℚ → Prop).card) fs = 39699

theorem oleg_cannot_obtain_39699 : sum_intersections_is_39699 ↔ false :=
by
  sorry

end oleg_cannot_obtain_39699_l165_165598


namespace cipher_solution_l165_165005

-- Definitions for symbols and their corresponding meanings
def symbol_equiv : char → string 
| '-'   := "=" 
| '~'   := "+"
| '<'   := "-" 
| '='    := "1"
| 'V'   := "9"
| '\u039B' := "2"   -- Greek capital letter Lambda
| '\u03B8' := "3"   -- Greek small letter theta
| '\u2299' := "4"   -- Circled Dot Operator
| 'O'   := "5"
| '\u0394' := "6"   -- Greek capital letter Delta
| '\u2295' := "7"   -- Circled Plus
| '>'    := "8"
| '\u2207' := "0"   -- Nabla

-- The proof problem
theorem cipher_solution (cipher : string): (∀ s ∈ cipher, (symbol_equiv s ≠ "")) → 
(ciphers : string): list string := 
  sorry

end cipher_solution_l165_165005


namespace collinear_M_N_X_Y_l165_165929

-- Defining points and collinearity
variables {A B C D Y X N M : Type*}

-- Assuming given conditions
axiom Trapezoid (A B C D : Type*) : (parallel A B D C)
axiom Intersection_Y (A D B C : Type*) : intersect A D B C Y
axiom Intersection_X (A C B D : Type*) : intersect A C B D X
axiom Midpoint_N (A B N : Type*) : midpoint A B N
axiom Midpoint_M (D C M : Type*) : midpoint D C M

-- The theorem stating the collinearity
theorem collinear_M_N_X_Y (h_trapezoid : Trapezoid A B C D)
  (h_Y : Intersection_Y A D B C Y)
  (h_X : Intersection_X A C B D X)
  (h_N : Midpoint_N A B N)
  (h_M : Midpoint_M D C M) : collinear M N X Y :=
sorry

end collinear_M_N_X_Y_l165_165929


namespace sum_of_first_five_terms_sequence_l165_165102

-- Definitions derived from conditions
def seventh_term : ℤ := 4
def eighth_term : ℤ := 10
def ninth_term : ℤ := 16

-- The main theorem statement
theorem sum_of_first_five_terms_sequence : 
  ∃ (a d : ℤ), 
    a + 6 * d = seventh_term ∧
    a + 7 * d = eighth_term ∧
    a + 8 * d = ninth_term ∧
    (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = -100) :=
by
  sorry

end sum_of_first_five_terms_sequence_l165_165102


namespace min_number_of_skilled_players_l165_165207

def tournament (n : ℕ) := 
Graph (fin n) := sorry
-- The graph of the tournament. Treating vertices as players and directed edges as wins.

def is_skilled_player (G : tournament 2023) (A : fin 2023) : Prop :=
  ∀ (B : fin 2023), G.edge B A → ∃ (C : fin 2023), G.edge C B ∧ G.edge A C
-- Player A is skilled if for each player B who won against A, there exists a player C who won against B and lost to A.

noncomputable def min_skilled_players : ℕ :=
  3
-- The minimum number of skilled players as proven is 3.

theorem min_number_of_skilled_players :
  ∀ (G : tournament 2023), 
    ∃ (S : finset (fin 2023)), S.card = min_skilled_players ∧ ∀ x ∈ S, is_skilled_player G x :=
sorry
-- The minimum value of the number of skilled players in any tournament of 2023 players is 3.

end min_number_of_skilled_players_l165_165207


namespace min_treasure_count_l165_165319

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165319


namespace range_of_m_l165_165804

noncomputable def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (m < 0)

noncomputable def q (m : ℝ) : Prop :=
  (16*(m-2)^2 - 16 < 0)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  intro h
  sorry

end range_of_m_l165_165804


namespace longest_route_from_A_to_B_l165_165217

structure CityMap where
  intersections : Finset (Fin 36)
  start        : Fin 36
  end          : Fin 36

def longestRoute (M : CityMap) (noRepeatIntersections : Prop) : Nat :=
  if noRepeatIntersections then 34 else 0

theorem longest_route_from_A_to_B (M : CityMap) (hA : M.start != M.end) 
  (noRepeatIntersections : M.intersections.card = 36 ∧ M.intersections ∀ i ∈ M.intersections, ¬(i = M.start ∨ i = M.end) → true) : 
  longestRoute M noRepeatIntersections = 34 := 
sorry


end longest_route_from_A_to_B_l165_165217


namespace product_less_than_40_l165_165963

def probability_less_than_40 : ℚ :=
  let prob := (6 * 8).toRat                 -- Total equally likely outcomes
  let successful_outcomes := (5 * 8) + 6    -- 5 full sets of 8 successful outcomes, plus 6 successful outcomes for Paco's spin of 6
  let probability := successful_outcomes / prob
  probability

theorem product_less_than_40 :
  probability_less_than_40 = 23 / 24 := 
by
  -- Proof logic here
  sorry

end product_less_than_40_l165_165963


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165152

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165152


namespace area_intuitive_diagram_l165_165840

theorem area_intuitive_diagram (a : ℝ) (h : 0 < a) :
  let s_original := (Real.sqrt 3 / 4) * a^2 in
  let relationship := (Real.sqrt 2 / 4) in
  let s_intuitive := relationship * s_original in
  s_intuitive = (Real.sqrt 6 / 16) * a^2 :=
by
  sorry

end area_intuitive_diagram_l165_165840


namespace intersection_eq_l165_165818

def set1 : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def set2 : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_eq : (set1 ∩ set2) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_eq_l165_165818


namespace problem_solution_l165_165776

theorem problem_solution :
  (2 * (5⁻¹ : ℤ) + 8 * (11⁻¹ : ℤ)) % 56 = 50 % 56 :=
by
  sorry

end problem_solution_l165_165776


namespace students_present_each_day_l165_165892
open BigOperators

namespace Absenteeism

def absenteeism_rate : ℕ → ℝ 
| 0 => 14
| n+1 => absenteeism_rate n + 2

def present_rate (n : ℕ) : ℝ := 100 - absenteeism_rate n

theorem students_present_each_day :
  present_rate 0 = 86 ∧
  present_rate 1 = 84 ∧
  present_rate 2 = 82 ∧
  present_rate 3 = 80 ∧
  present_rate 4 = 78 := 
by
  -- Placeholder for the proof steps
  sorry

end Absenteeism

end students_present_each_day_l165_165892


namespace function_has_two_zeros_l165_165856

/-- 
Given the function y = x + 1/(2x) + t has two zeros under the condition t > 0,
prove that the range of the real number t is (-∞, -√2).
-/
theorem function_has_two_zeros (t : ℝ) (ht : t > 0) : t < -Real.sqrt 2 :=
sorry

end function_has_two_zeros_l165_165856


namespace sum_divisors_12_eq_28_l165_165365

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165365


namespace shapes_both_axisymmetric_centrosymmetric_l165_165731

-- Define the type for shapes
inductive Shape
| EquilateralTriangle
| Parallelogram
| Rectangle
| IsoscelesTrapezoid
| Circle

open Shape

-- Define predicates for axisymmetric and centrosymmetric
def is_axisymmetric : Shape → Prop
| EquilateralTriangle   := true
| Parallelogram         := false
| Rectangle             := true
| IsoscelesTrapezoid    := true
| Circle                := true

def is_centrosymmetric : Shape → Prop
| EquilateralTriangle   := false
| Parallelogram         := true
| Rectangle             := true
| IsoscelesTrapezoid    := false
| Circle                := true

-- Prove that rectangles and circles are the only shapes that are both axisymmetric and centrosymmetric
theorem shapes_both_axisymmetric_centrosymmetric : 
  ∀ s : Shape, (is_axisymmetric s ∧ is_centrosymmetric s) ↔ (s = Rectangle ∨ s = Circle) :=
by
  intro s
  cases s;
  -- Use cases to cover all shapes
  simp [is_axisymmetric, is_centrosymmetric];
  -- Prove each case
  -- Rectangle: (true ∧ true) ↔ true
  -- Circle: (true ∧ true) ↔ true
  -- Others: (true ∧ false | false ∧ true) ↔ false
  try { constructor; intro h; cases h; contradiction }; -- Handle contradictory cases
  constructor; -- Rectangle and Circle cases
  intro h; exact Or.inl sorry; -- Replace sorry with actual proof for Rectangle
  intro h; exact Or.inr sorry -- Replace sorry with actual proof for Circle

end shapes_both_axisymmetric_centrosymmetric_l165_165731


namespace sum_positive_integral_values_l165_165381

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165381


namespace solution_l165_165205

def problem_statement : Prop :=
  (∏ k in Finset.range 63, Real.log (k + 3) / Real.log (k + 2)) = 6

theorem solution : problem_statement :=
  sorry

end solution_l165_165205


namespace pool_perimeter_l165_165528

variables (garden_length garden_width pool_area walkways_width : ℝ)

-- Given conditions
def garden_dimensions : Prop := (garden_length = 8) ∧ (garden_width = 6)
def pool_specifications : Prop := (8 - 2 * walkways_width) * (6 - 2 * walkways_width) = 24

-- Desired statement to prove
theorem pool_perimeter (h1 : garden_dimensions) (h2 : pool_specifications) : 
  2 * ((8 - 2 * walkways_width) + (6 - 2 * walkways_width)) = 20 :=
sorry

end pool_perimeter_l165_165528


namespace correct_option_given_inequality_l165_165878

theorem correct_option_given_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
sorry

end correct_option_given_inequality_l165_165878


namespace sum_of_divisors_of_12_l165_165389

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165389


namespace jake_earns_720_in_5_days_l165_165549
noncomputable def jake_earnings_5_days : ℕ :=
  let jacob_hourly := 6 in
  let jake_hourly := 3 * jacob_hourly in
  let hours_per_day := 8 in
  let days := 5 in
  jake_hourly * hours_per_day * days

theorem jake_earns_720_in_5_days :
  jake_earnings_5_days = 720 :=
by
  sorry

end jake_earns_720_in_5_days_l165_165549


namespace eccentricity_hyperbola_l165_165858

noncomputable def c (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

def ecc (a b : ℝ) : ℝ := c a b / a

theorem eccentricity_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (hyp : b = 2 * a) :
  ecc a b = sqrt 5 := by
  sorry

end eccentricity_hyperbola_l165_165858


namespace pq_implies_a_range_l165_165816

def proposition_p (a : ℝ) := ∀ x ∈ set.Icc 1 2, 2 * x^2 - a ≥ 0

def proposition_q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem pq_implies_a_range (a : ℝ) :
  (proposition_p a ∧ proposition_q a) → (a ≤ -2 ∨ 1 ≤ a ∧ a ≤ 2) := 
by
  sorry

end pq_implies_a_range_l165_165816


namespace b_is_zero_if_f_is_odd_f_is_decreasing_on_1_inf_range_of_m_l165_165850

-- Definition of f
def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Condition that f is odd
def is_odd_function (a b : ℝ) : Prop := 
  ∀ x : ℝ, f(a, b, -x) = -f(a, b, x)

-- Problem 1
theorem b_is_zero_if_f_is_odd (a : ℝ) (h : is_odd_function a 0) : 
  0 = 0 := by
  sorry

-- Problem 2
theorem f_is_decreasing_on_1_inf (a : ℝ) (h : 0 < a) :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → 1 < x₂ → x₁ < x₂ → f(a, 0, x₁) > f(a, 0, x₂) := by
  sorry

-- Definition of g
def g (m x : ℝ) : ℝ := m * x^2 - 2 * x + 2 - m

-- Condition for Problem 3
def exists_x2 (a : ℝ) (g : ℝ → ℝ) : Prop := 
  ∀ x1 ∈ [1, 3], ∃ x2 ∈ [0, 1], f(a, 0, x1) + 1/2 = g(x2)

-- Problem 3
theorem range_of_m (h : exists_x2 1 (g m)) :
  -∞ < m ∧ m ≤ 1 := by
  sorry

end b_is_zero_if_f_is_odd_f_is_decreasing_on_1_inf_range_of_m_l165_165850


namespace wheels_on_front_axle_l165_165105

variable (x : ℕ) (f : ℕ)
variable (t : ℕ)

theorem wheels_on_front_axle (ht : t = 5) (total_wheels : 18 - f) (front_axle_wheels : (t = 3.50 + 0.50 * (x - 2))) : f = 2 := by
sorry

end wheels_on_front_axle_l165_165105


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165135

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165135


namespace min_treasure_count_l165_165315

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165315


namespace sum_of_squares_of_perfect_squares_l165_165128

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165128


namespace sum_of_perfect_square_squares_less_than_1000_l165_165173

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165173


namespace quadratic_inequality_solution_set_l165_165482

theorem quadratic_inequality_solution_set (a b c : ℝ) (h : set.Ioo (-(1 : ℝ) / 2) 2 = { x | ax^2 - bx + c > 0 }) :
  b < 0 ∧ c > 0 ∧ a - b + c > 0 :=
by
  sorry

end quadratic_inequality_solution_set_l165_165482


namespace sum_of_fourth_powers_below_1000_l165_165160

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165160


namespace cos_inequality_y_zero_l165_165351

theorem cos_inequality_y_zero (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hy : 0 ≤ y ∧ y ≤ π / 2) :
  (∀ x ∈ Icc (0 : ℝ) (π / 2), cos (x + y) ≥ cos x * cos y) ↔ y = 0 := by
  sorry

end cos_inequality_y_zero_l165_165351


namespace principal_amount_equals_1000_l165_165193

-- Definitions for conditions
def rate : ℝ := 10 -- rate of interest per annum (in %)
def time : ℝ := 3 -- time in years
def difference : ℝ := 31.000000000000455 -- difference between CI and SI

-- Proof statement to check the principal amount P
theorem principal_amount_equals_1000 :
  ∃ (P : ℝ), P * (0.331 - 0.3) = difference :=
by {
  use 1000, -- Potential value for the principal amount based on given solution
  -- Here we show the necessary calculation to equate the difference
  simp [rate, time, difference],
  norm_num,
  -- The calculation here is precisely.
  sorry -- Proof steps to be completed.
}

end principal_amount_equals_1000_l165_165193


namespace triangle_inequality_l165_165610

variable {a b c : ℝ}

def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

def inscribed_circle_radius (a b c : ℝ) (A : ℝ) : ℝ :=
  A / semiperimeter a b c

theorem triangle_inequality 
  (a b c : ℝ)
  (s := semiperimeter a b c)
  (A := sqrt (s * (s - a) * (s - b) * (s - c)))
  (r := inscribed_circle_radius a b c A)
  (P := 2 * s)
  (α : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A = 1/2 * a * b * sin α ∧ 
  A = sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  sin α <= 1 →
  a * b ≥ P * r :=
by
  sorry

end triangle_inequality_l165_165610


namespace john_buys_360_packs_l165_165556

def John_buys_packs (classes students_per_class packs_per_student total_packs : ℕ) : Prop :=
  classes = 6 →
  students_per_class = 30 →
  packs_per_student = 2 →
  total_packs = (classes * students_per_class) * packs_per_student
  → total_packs = 360

theorem john_buys_360_packs : John_buys_packs 6 30 2 360 :=
by { intros, sorry }

end john_buys_360_packs_l165_165556


namespace Roselyn_initial_books_l165_165613

theorem Roselyn_initial_books :
  ∀ (books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books : ℕ),
    books_given_to_Rebecca = 40 →
    books_remaining = 60 →
    books_given_to_Mara = 3 * books_given_to_Rebecca →
    total_books_given = books_given_to_Mara + books_given_to_Rebecca →
    initial_books = books_remaining + total_books_given →
    initial_books = 220 :=
by
  intros books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Roselyn_initial_books_l165_165613


namespace tangent_triangle_area_l165_165979

noncomputable def exp_fn (x : ℝ) := Real.exp (1 / 2 * x)
noncomputable def exp_fn_deriv (x : ℝ) := 1 / 2 * Real.exp (1 / 2 * x)

theorem tangent_triangle_area : 
  let tangent_at  (x := 4) := exp_fn 4 
  let tangent_deriv_at_4 := exp_fn_deriv 4 
  let tangent_line (x : ℝ) := tangent_deriv_at_4 * (x - 4) + tangent_at
  let x_intercept := 2
  let y_intercept := - tangent_deriv_at_4 * 4 + tangent_at
  ∃ (tangent_at : ℝ), 
  tangent_at = exp_fn 4 ∧ 
  ∃ (tangent_deriv_at_4 : ℝ),
  tangent_deriv_at_4 = exp_fn_deriv 4 ∧
  ∃ (area : ℝ),
  area = let base := x_intercept in
         let height := y_intercept in
         1 / 2 * base * height →
  area = Real.exp 2 :=
by
  sorry

end tangent_triangle_area_l165_165979


namespace solution_set_of_inequality_l165_165595

/-- Given an odd function f that is increasing on the interval (0, +∞) and f(1) = 0, 
    prove that the solution set of the inequality f(x) < 0 is (-1, 0) ∪ (0, 1). -/
theorem solution_set_of_inequality {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ ⦃x y⦄, 0 < x → x < y → x < +∞ → y < +∞ → f x < f y)
  (h_f1 : f 1 = 0) : 
  { x | f x < 0 } = set.union (set.Ioo (-1) 0) (set.Ioo 0 1) := 
sorry

end solution_set_of_inequality_l165_165595


namespace lin_climb_stairs_l165_165774

def g : ℕ → ℕ
| 0 := 1
| 1 := 1
| 2 := 2
| 3 := 4
| (n+4) := g n + g (n+1) + g (n+2) + g (n+3)

theorem lin_climb_stairs : g 8 = 108 :=
by {
  sorry,
}

end lin_climb_stairs_l165_165774


namespace triangle_PQR_equilateral_l165_165019

noncomputable theory
open_locale classical

variables {A B C P Q R : Type*}
variables {AC_line : is_line_segment A C}
variables {B_on_AC : is_between B A C}
variables {isosceles_PAB : is_isosceles_triangle P A B}
variables {isosceles_QAC : is_isosceles_triangle Q A C}
variables {isosceles_RAC : is_isosceles_triangle R A C}
variables {angle_APB : ∠P A B = 120 ∧ ∠B Q C = 120}
variables {angle_ARC : ∠R A C = 120}

theorem triangle_PQR_equilateral :
  is_equilateral_triangle P Q R :=
sorry

end triangle_PQR_equilateral_l165_165019


namespace sum_of_divisors_of_12_l165_165393

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165393


namespace number_of_odd_perfect_cubes_or_squares_l165_165874

theorem number_of_odd_perfect_cubes_or_squares (N : ℕ) (hN : N = 1000) :
  let odd_squares := { n | n % 2 = 1 ∧ n^2 < N },
      odd_cubes := { n | n % 2 = 1 ∧ n^3 < N },
      odd_sixths := { n | n % 2 = 1 ∧ n^6 < N } in
  odd_squares.card + odd_cubes.card - odd_sixths.card = 20 :=
sorry

end number_of_odd_perfect_cubes_or_squares_l165_165874


namespace polynomial_irreducible_l165_165932

def polynomial (p : ℕ) : Polynomial ℤ :=
  Polynomial.sum (List.range p) (fun n => (Polynomial.C 1) * Polynomial.X ^ n)

theorem polynomial_irreducible (p : ℕ) [hp : Nat.Prime p] : Irreducible (polynomial p) := by 
  sorry

end polynomial_irreducible_l165_165932


namespace sum_of_divisors_of_12_l165_165392

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165392


namespace intersection_eq_neg1_l165_165500

open Set

noncomputable def setA : Set Int := {x : Int | x^2 - 1 ≤ 0}
def setB : Set Int := {x : Int | x^2 - x - 2 = 0}

theorem intersection_eq_neg1 : setA ∩ setB = {-1} := by
  sorry

end intersection_eq_neg1_l165_165500


namespace work_completion_l165_165206

theorem work_completion (W : ℝ) (W_x W_y : ℝ) (hx : 8 * W_x + 20 * W_y = W) (hy : W_y = W / 25) : 
  W / W_x = 40 :=
by
  have h1 : 8 * W_x + 20 * (W / 25) = W, from hx
  have h2 : 8 * W_x + (4/5) * W = W, by rw [←mul_div_assoc, hy, mul_div_cancel' _ (by norm_num)]
  have h3 : 8 * W_x = W - (4/5) * W, by linarith
  have h4 : 8 * W_x = (1/5) * W, by linarith
  have h5 : W_x = W / 40, by linarith
  exact (W / W_x).trans h5.symm

end work_completion_l165_165206


namespace calculate_expr_equals_l165_165334

noncomputable def calculate_complex_expr : ℝ :=
  (3.14 - Real.pi)^0 - (1 / 2)^(-1) + |Real.sqrt 3 - 2| + Real.sqrt 27

theorem calculate_expr_equals :
  calculate_complex_expr = 1 + 2 * Real.sqrt 3 := by
  sorry

end calculate_expr_equals_l165_165334


namespace min_value_x_squared_y_squared_z_squared_l165_165454

theorem min_value_x_squared_y_squared_z_squared
  (x y z : ℝ)
  (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ (18 / 7) :=
sorry

end min_value_x_squared_y_squared_z_squared_l165_165454


namespace slope_tangent_at_pi_div_6_l165_165649

def f (x : ℝ) := Real.sin x

theorem slope_tangent_at_pi_div_6 : 
  HasDerivAt f (Real.cos (Real.pi / 6)) (Real.pi / 6) ∧ (Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :=
sorry

end slope_tangent_at_pi_div_6_l165_165649


namespace last_two_digits_sum_factorials_l165_165674

-- Define a function that computes the factorial of a number
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the sum of factorials up to 14
def sum_of_factorials_to_14 : ℕ := ∑ i in finset.range 15, factorial i -- range 0 to 14

-- Prove the last two digits of this sum is 13
theorem last_two_digits_sum_factorials : sum_of_factorials_to_14 % 100 = 13 := by
  sorry

end last_two_digits_sum_factorials_l165_165674


namespace cindys_correct_result_l165_165915

-- Explicitly stating the conditions as definitions
def incorrect_operation_result := 260
def x := (incorrect_operation_result / 5) - 7

theorem cindys_correct_result : 5 * x + 7 = 232 :=
by
  -- Placeholder for the proof
  sorry

end cindys_correct_result_l165_165915


namespace count_three_digit_multiples_13_and_5_l165_165875

theorem count_three_digit_multiples_13_and_5 : 
  ∃ count : ℕ, count = 14 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 65 = 0) → 
  (∃ k : ℕ, n = k * 65 ∧ 2 ≤ k ∧ k ≤ 15) → count = 14 :=
by
  sorry

end count_three_digit_multiples_13_and_5_l165_165875


namespace sum_of_divisors_of_12_l165_165428

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165428


namespace sum_of_divisors_of_12_l165_165423

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165423


namespace sum_of_divisors_of_12_l165_165407

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165407


namespace cutoff_score_admission_l165_165241

theorem cutoff_score_admission (x : ℝ) 
  (h1 : (2 / 5) * (x + 15) + (3 / 5) * (x - 20) = 90) : x = 96 :=
sorry

end cutoff_score_admission_l165_165241


namespace squares_in_F10_l165_165091

noncomputable def seq_squares (n : ℕ) : ℕ :=
  if n = 1 then 
    2 -- S_1 = 2 
  else if n = 2 then 
    14 -- S_2 = 14 
  else 
    seq_squares (n - 1) + 3 * n^2 -- S_n = S_{n-1} + 3n^2 for n >= 3

theorem squares_in_F10 : seq_squares 10 = 1239 :=
by
  sorry

end squares_in_F10_l165_165091


namespace solve_linear_system_l165_165786

theorem solve_linear_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ 
  x = -219 / 88 ∧ y = 97 / 22 :=
by
  sorry

end solve_linear_system_l165_165786


namespace sum_fourth_powers_lt_1000_l165_165181

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165181


namespace solve_system_of_equations_l165_165621

theorem solve_system_of_equations :
  ∃ (x y : ℕ), (x + 2 * y = 5) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by {
  sorry
}

end solve_system_of_equations_l165_165621


namespace cotton_fiber_length_problem_l165_165663

noncomputable def count_fibers_less_than_20 (fiber_lengths : List ℝ) : ℕ :=
  (fiber_lengths.filter (λ x, x < 20)).length

theorem cotton_fiber_length_problem
  (fiber_lengths : List ℝ)
  (h_sample_size : fiber_lengths.length = 100)
  (h_interval : ∀ x ∈ fiber_lengths, 5 ≤ x ∧ x ≤ 40)
  (h_fibers_count : count_fibers_less_than_20 fiber_lengths = 60) :
  ∃ n, n = 60 ∧ count_fibers_less_than_20 fiber_lengths = n :=
by {
  use 60,
  split,
  { refl },
  { exact h_fibers_count }
}

end cotton_fiber_length_problem_l165_165663


namespace find_pairs_l165_165779

theorem find_pairs (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : (mn - 1) ∣ (n^3 - 1)) :
  ∃ k : ℕ, 1 < k ∧ ((m = k ∧ n = k^2) ∨ (m = k^2 ∧ n = k)) :=
sorry

end find_pairs_l165_165779


namespace sum_of_divisors_of_12_l165_165409

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165409


namespace nat_sqrt_condition_l165_165352

theorem nat_sqrt_condition (n : ℕ) (h1: ∃ k : ℕ, n = k ^ 8) (h2: n ^ (7 / 8) < 2219) :
  n = 1 ∨ n = 256 :=
by {
  sorry
}

end nat_sqrt_condition_l165_165352


namespace quadratic_trinomial_is_valid_l165_165606

noncomputable def quadratic_trinomial_example : Polynomial ℤ := 
  Polynomial.Coeff (0 : ℤ) 1 * Polynomial.X ^ 2 + Polynomial.Coeff (0 : ℤ) 1 * Polynomial.X + Polynomial.Coeff (0 : ℤ) 1

theorem quadratic_trinomial_is_valid (p : Polynomial ℤ) :
  p.degree = 2 ∧ Polynomial.Coeff (0 : ℤ) (p.degree_zero_coefficient) = 1 ∧ 
  Polynomial.Coeff (0 : ℤ) (p.degree_one_coefficient) = 1 ∧ 
  Polynomial.Coeff (0 : ℤ) (p.degree_two_coefficient) = 1 :=
by 
  let example : Polynomial ℤ := quadratic_trinomial_example
  have h : example = Polynomial.Coeff (0 : ℤ) 1 * Polynomial.X ^ 2 + Polynomial.Coeff (0 : ℤ) 1 * Polynomial.X + Polynomial.Coeff (0 : ℤ) 1, from rfl,
  exact ⟨Polynomial.degree_example, Polynomial.degree_zero_coefficient_example, Polynomial.degree_one_coefficient_example, Polynomial.degree_two_coefficient_example⟩

end quadratic_trinomial_is_valid_l165_165606


namespace time_to_remove_remaining_water_l165_165717

/-- 
Given:
1. With one pump running, the water level rises by 2 cm/min (20 cm in 10 minutes).
2. With two pumps running, the water level drops by 2 cm/min (10 cm in 5 minutes).
3. The leak was sealed.

Prove:
The time to remove the remaining water is 5/4 minutes.
-/
theorem time_to_remove_remaining_water :
  let P := 4 in  -- Pumping capacity of one pump (cm/min)
  let L := 6 in  -- Leakage rate (cm/min)
  let remaining_water := 10 in  -- Assume 10 cm of remaining water
  (2 * P - L = 2) ∧ (L - P = 2) →
  let two_pumps_rate := 2 * P in
  two_pumps_rate = 8 →
  (remaining_water : ℝ) / (two_pumps_rate : ℝ) = 5 / 4 :=
by
  intros P L remaining_water h_eqns two_pumps_rate h_rate
  sorry

end time_to_remove_remaining_water_l165_165717


namespace ellipse_equation_exists_and_line_condition_l165_165462

-- Definition of the ellipse parameters
def ellipse_eccentricity := (C : Type) (a b : ℝ) (h1 : a > b > 0) (e : ℝ) (h2 : e = b / a) : Prop :=
  ∃ c : ℝ, e = c / a ∧ c^2 = a^2 - b^2

-- Definition of the ellipse equation given a point
def point_on_ellipse (a b : ℝ) (M : ℝ × ℝ) (h : M = (1, 3/2)) : Prop :=
  (1^2 / a^2) + ((3/2)^2 / b^2) = 1

-- Definition of the line intersecting at two distinct points and satisfying given conditions
def line_intersection_condition (C : Type) (l : Type) (P A B M : ℝ × ℝ ) : Prop :=
  P = (2, 1) → M = (1, 3/2) → ∃ k : ℝ, l = (λ x : ℝ, k * (x - 2) + 1) ∧
  ∀ x1 y1 x2 y2 : ℝ, (x1, y1) ≠ (x2, y2) ∧ (8 * k * (2*k-1))^2 > 4 * (3 + 4*k^2) * (16*k^2 - 16*k - 8) ∧
  ((x1 - 2) * (x2 - 2) + (y1 - 1) * (y2 - 1)) = (4 + 4 * k^2) / (3 + 4 * k^2) = 5 / 4

theorem ellipse_equation_exists_and_line_condition :
  ∃ (a b : ℝ) (e : ℝ), (ellipse_eccentricity C a b (by linarith) e (by norm_num)) ∧ 
  (point_on_ellipse a b (1, 3/2)) ∧
  (∃ l, line_intersection_condition C l (2, 1) (some A : ℝ × ℝ) (some B : ℝ × ℝ) (1, 3/2)) :=
sorry

end ellipse_equation_exists_and_line_condition_l165_165462


namespace complex_number_addition_l165_165484

theorem complex_number_addition : 
  (∃ a b : ℝ, (1 + 2*complex.I) / (1 + complex.I) = a + b * complex.I) → 
  (∃ a b : ℝ, a + b = 2) :=
by
  sorry

end complex_number_addition_l165_165484


namespace fill_time_is_60_minutes_l165_165199

def tank_capacity : ℝ := 10000
def initial_water : ℝ := tank_capacity / 2
def fill_rate : ℝ := 1 / 2 -- kiloliters per minute
def drain_rate1 : ℝ := 1 / 4 -- kiloliters per minute
def drain_rate2 : ℝ := 1 / 6 -- kiloliters per minute
def remaining_volume : ℝ := initial_water
def net_flow_rate : ℝ := fill_rate - (drain_rate1 + drain_rate2)
def time_to_fill (remaining_volume : ℝ) (net_flow_rate : ℝ) : ℝ := remaining_volume / net_flow_rate

theorem fill_time_is_60_minutes :
  time_to_fill remaining_volume net_flow_rate = 60 := 
by
  -- Here one would provide the proof.
  sorry

end fill_time_is_60_minutes_l165_165199


namespace fraction_identity_l165_165446

noncomputable def calc_fractions (x y : ℝ) : ℝ :=
  (x + y) / (x - y)

theorem fraction_identity (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) : calc_fractions x y = -1001 :=
by
  sorry

end fraction_identity_l165_165446


namespace license_plate_count_l165_165719

theorem license_plate_count :
  let num_digits := 10
  let num_letters := 26
  let digit_places := 5
  let letter_unit := num_letters ^ 2
  let positions := 6
  6 * (num_digits ^ digit_places) * letter_unit = 406560000 :=
by
  let num_digits := 10
  let num_letters := 26
  let digit_places := 5
  let letter_unit := num_letters ^ 2
  let positions := 6
  have h1 : 6 * (num_digits ^ digit_places) * letter_unit = 6 * 100000 * 676,
    by calc
    6 * (num_digits ^ digit_places) * letter_unit 
      = 6 * (10 ^ 5) * (26 ^ 2) : by sorry,
    _ = 6 * 100000 * 676 : by sorry,
  have h2 : 6 * 100000 * 676 = 406560000,
    by norm_num,
  exact (Eq.trans h1 h2),
sorry

end license_plate_count_l165_165719


namespace bottom_layer_lamps_l165_165251

noncomputable def lamps_on_layers (a1 n : ℕ) : ℕ :=
  ∑ k in finset.range 9, (a1 + k * n)

theorem bottom_layer_lamps (a1 n : ℕ) (h1 : lamps_on_layers a1 n = 126) (h2 : a1 + 8 * n = 13 * a1) :
  a1 + 8 * n = 26 :=
by
  sorry

end bottom_layer_lamps_l165_165251


namespace billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l165_165114

theorem billy_avoids_swimming_n_eq_2022 :
  ∀ n : ℕ, n = 2022 → (∃ (strategy : ℕ → ℕ), ∀ k, strategy (2022 + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_odd_n (n : ℕ) (h : n > 10 ∧ n % 2 = 1) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_even_n (n : ℕ) (h : n > 10 ∧ n % 2 = 0) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

end billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l165_165114


namespace sum_of_perfect_square_squares_less_than_1000_l165_165174

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165174


namespace find_value_of_a_l165_165039

theorem find_value_of_a (a : ℝ) (h1 : a > 0) (h2 : ∫ x in 0..a, real.sqrt x = a ^ 2) :
  a = 4 / 9 :=
sorry

end find_value_of_a_l165_165039


namespace clyde_sequence_result_l165_165071

theorem clyde_sequence_result :
  let initial_number := 10^8
  let operation (n : ℕ) (x : ℕ) :=
    if n % 2 = 0 then x / 3 else x * 7
  let final_number := (fin 16).foldl (λ x i, operation i x) initial_number
  final_number = (490 / 9)^8 :=
by
  sorry

end clyde_sequence_result_l165_165071


namespace corrections_needed_l165_165510

-- Define the corrected statements
def corrected_statements : List String :=
  ["A = 50", "B = A", "x = 1", "y = 2", "z = 3", "INPUT“How old are you?”;x",
   "INPUT x", "PRINT“A+B=”;C", "PRINT“Good-bye!”"]

-- Define the function to check if the statement is correctly formatted
def is_corrected (statement : String) : Prop :=
  statement ∈ corrected_statements

-- Lean theorem statement to prove each original incorrect statement should be correctly formatted
theorem corrections_needed (s : String) (incorrect : s ∈ ["A = B = 50", "x = 1, y = 2, z = 3", 
  "INPUT“How old are you”x", "INPUT, x", "PRINT A+B=;C", "PRINT Good-bye!"]) :
  ∃ t : String, is_corrected t :=
by 
  sorry

end corrections_needed_l165_165510


namespace number_of_men_business_class_is_32_l165_165065

-- Definitions of the conditions
def total_passengers : ℕ := 300
def percentage_men : ℝ := 0.7
def percentage_men_business_class : ℝ := 0.15

-- Calculate number of men in train
def number_of_men : ℕ := (total_passengers : ℝ) * percentage_men |>.toInt

-- Calculate number of men in business class
def number_of_men_business_class : ℕ := (number_of_men : ℝ * percentage_men_business_class).round

-- Statement to prove
theorem number_of_men_business_class_is_32 : number_of_men_business_class = 32 := by
  sorry

end number_of_men_business_class_is_32_l165_165065


namespace opposite_of_2023_is_neg_2023_l165_165099

def opposite_of (x : Int) : Int := -x

theorem opposite_of_2023_is_neg_2023 : opposite_of 2023 = -2023 :=
by
  sorry

end opposite_of_2023_is_neg_2023_l165_165099


namespace chinese_remainder_solution_l165_165122

theorem chinese_remainder_solution 
  (r : ℕ)
  (a : fin r → ℤ) 
  (m : fin r → ℤ) 
  (h_coprime : ∀ i j : fin r, i ≠ j → m i ≠ 0 ∧ m j ≠ 0 ∧ gcd (m i) (m j) = 1)
  (M := ∏ i, m i)
  (M_k := λ k : fin r, M / (m k))
  (i_k := λ k : fin r, (1 : ℤ) * (1 : ℤ) → ℤ) -- Placeholder for modular inverse, will need explicit definition in practice
  :
  ∃ x : ℤ, x ≡ ∑ i, a i * (i_k i) * (M_k i) [MOD M] :=
  sorry

end chinese_remainder_solution_l165_165122


namespace cubic_roots_sum_cube_l165_165940

theorem cubic_roots_sum_cube (a b c : ℂ) (h : ∀x : ℂ, (x=a ∨ x=b ∨ x=c) → (x^3 - 2*x^2 + 3*x - 4 = 0)) : a^3 + b^3 + c^3 = 2 :=
sorry

end cubic_roots_sum_cube_l165_165940


namespace cos_angle_ACB_l165_165900

variables (A B C D : Type) [inner_product_space ℝ Type] 
variables {x y θ : ℝ}
variables {α β : ℝ}

-- Define conditions
def condition_1 (h1 : ∠ADB = 90 ∧ ∠BDC = 90) : Prop := h1
def condition_2 (h2 : ∠ADC = θ ∧ θ ≠ 90) : Prop := h2
def condition_3 (h3 : x = sin (∠CAD)) : Prop := h3
def condition_4 (h4 : y = sin (∠CBD)) : Prop := h4

-- Prove that cos(∠ACB) = x * y * cos θ
theorem cos_angle_ACB (h1 : ∠ADB = 90 ∧ ∠BDC = 90)
                      (h2 : ∠ADC = θ ∧ θ ≠ 90)
                      (h3 : x = sin (∠CAD))
                      (h4 : y = sin (∠CBD)) :
  cos (∠ACB) = x * y * cos θ :=
sorry

end cos_angle_ACB_l165_165900


namespace birth_rate_l165_165634

variable (D : ℝ) (NGR : ℝ) (P: ℝ) 

theorem birth_rate (hD : D = 11) (hNGR : NGR = 2.1) (hP: P = 1000) : 
  B = 32 := 
by 
  let B := (NGR * P) / 100 + D
  have : B = ((2.1 * 1000) / 100 + 11) := by 
    rw [hNGR, hP, hD]
    norm_num
  exact this

end birth_rate_l165_165634


namespace sequence_last_digit_count_l165_165243

theorem sequence_last_digit_count :
  let seq := λ n, (7^n : ℕ) % 10
  ∃ k, k = 502 ∧ (∀ n ∈ finset.range 2008, seq n = 3 ↔ (n + 1) % 4 = 3) :=
begin
  sorry
end

end sequence_last_digit_count_l165_165243


namespace highest_probability_event_l165_165261

variable (Ω : Type) [ProbabilitySpace Ω]

def Event (Ω : Type) := Set Ω

variables (A B C : Event Ω)

-- Conditions: Subset relationships of the events
axiom hCAB : C ⊆ B
axiom hBAA : B ⊆ A

theorem highest_probability_event :
  ∀ (P : Event Ω → ℝ), P A ≥ P B ∧ P B ≥ P C :=
by
  intro P
  have h1 : P C ≤ P B := sorry
  have h2 : P B ≤ P A := sorry
  exact ⟨h2, h1⟩

end highest_probability_event_l165_165261


namespace original_price_l165_165697

theorem original_price
  (final_price : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (H1 : final_price = 126)
  (H2 : discount1 = 0.20)
  (H3 : discount2 = 0.10) :
  let P := final_price / ((1 - discount1) * (1 - discount2)) in
  P = 175 :=
by
  let price_after_first_discount := final_price / (1 - discount1)
  let original_price := price_after_first_discount / (1 - discount2)
  have h1 : price_after_first_discount = 126 / 0.80 := by simp [H1, H2]
  have h2 : original_price = 157.50 / 0.90 := by simp [h1]
  have h3 : original_price = 175 := by simp [h2]
  exact h3


end original_price_l165_165697


namespace parallel_vectors_x_value_l165_165867

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_vectors_x_value (x : ℝ) :
  ∀ k : ℝ,
  k ≠ 0 ∧ k * 1 = -2 ∧ k * -2 = x →
  x = 4 :=
by
  intros k hk
  have hk1 : k * 1 = -2 := hk.2.1
  have hk2 : k * -2 = x := hk.2.2
  -- Proceed from here to the calculations according to the steps in b):
  sorry

end parallel_vectors_x_value_l165_165867


namespace sum_of_divisors_of_12_l165_165388

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165388


namespace sum_positive_integral_values_l165_165379

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165379


namespace budget_remaining_l165_165231

noncomputable def solve_problem : Nat :=
  let total_budget := 325
  let cost_flasks := 150
  let cost_test_tubes := (2 / 3 : ℚ) * cost_flasks
  let cost_safety_gear := (1 / 2 : ℚ) * cost_test_tubes
  let total_expenses := cost_flasks + cost_test_tubes + cost_safety_gear
  total_budget - total_expenses

theorem budget_remaining : solve_problem = 25 := by
  sorry

end budget_remaining_l165_165231


namespace gift_shop_combinations_l165_165224

def num_varieties_wrapping_paper : ℕ := 10
def num_colors_ribbon : ℕ := 5
def num_types_gift_card : ℕ := 6
def num_specific_ribbons (colors : Fin num_colors_ribbon -> Prop) : ℕ :=
  (if colors ⟨0, by simp⟩ then 1 else 0) + (if colors ⟨1, by simp⟩ then 1 else 0)

theorem gift_shop_combinations (colors : Fin num_colors_ribbon -> Prop) (h : colors ⟨0, by simp⟩ ∧ colors ⟨1, by simp⟩) : 
  (num_varieties_wrapping_paper * num_specific_ribbons colors * num_types_gift_card) = 120 :=
by
  -- leaving the proof as a sorry, as per instruction
  sorry

end gift_shop_combinations_l165_165224


namespace evaluate_P35_l165_165032

noncomputable def P (x : ℝ) : ℝ := sorry

theorem evaluate_P35 (P : ℝ → ℝ) 
  (h_deg : ∃n, ∀ (x y : ℝ), P(x + y) = (P x) + (P y))
  (h_values : ∀ k : ℝ, 0 ≤ k ∧ k ≤ 34 → P k = k * (k + 1)) : 
  42840 * P 35 = 40460 := 
by sorry

end evaluate_P35_l165_165032


namespace minimum_treasures_count_l165_165311

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165311


namespace max_value_on_interval_l165_165997

noncomputable def test_function (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem max_value_on_interval : ∀ (x : ℝ), x ∈ set.Icc 0 4 → test_function x ≤ 10 := by
  sorry

end max_value_on_interval_l165_165997


namespace complete_the_square_l165_165670

theorem complete_the_square (x : ℝ) : 
  (x^2 - 2 * x - 3 = 0) ↔ ((x - 1)^2 = 4) :=
by sorry

end complete_the_square_l165_165670


namespace team_card_sending_l165_165196

theorem team_card_sending (x : ℕ) (h : x * (x - 1) = 56) : x * (x - 1) = 56 := 
by 
  sorry

end team_card_sending_l165_165196


namespace sum_of_fourth_powers_less_than_1000_l165_165186

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165186


namespace total_people_at_gathering_l165_165737

theorem total_people_at_gathering :
  ∀ (W S J WS WJ SJ WSJ : ℕ),
  W = 26 →
  S = 22 →
  J = 18 →
  WS = 17 →
  WJ = 12 →
  SJ = 10 →
  WSJ = 8 →
  (W + S + J - WS - WJ - SJ + WSJ) = 35 :=
by
  intros W S J WS WJ SJ WSJ hW hS hJ hWS hWJ hSJ hWSJ
  simp [hW, hS, hJ, hWS, hWJ, hSJ, hWSJ]
  sorry

end total_people_at_gathering_l165_165737


namespace sixth_number_is_12_l165_165532

def sequence : List ℕ := [2, 16, 4, 14, 6, 12, 8]

theorem sixth_number_is_12 : sequence.get! 5 = 12 :=
by
  sorry

end sixth_number_is_12_l165_165532


namespace sqrt_meaningful_iff_x_geq_7_l165_165514

theorem sqrt_meaningful_iff_x_geq_7 (x : ℝ) : (∃ r : ℝ, r = sqrt (x - 7)) ↔ x ≥ 7 := 
by
  sorry

end sqrt_meaningful_iff_x_geq_7_l165_165514


namespace sum_fourth_powers_lt_1000_l165_165182

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165182


namespace hyperbola_equation_l165_165476

theorem hyperbola_equation
  (a b : ℝ)
  (h_ellipse : ∀ x y, x^2 / 5 + y^2 = 1)
  (h_asymptote : ∀ x y, (√3) * x - y = 0)
  (h_foci : ∀ c, c^2 = 1 + 1) :
  (∃ a b, a^2 = 1 ∧ b^2 = 3 ∧ a^2 + b^2 = 4 ∧ (b / a = √3)) → 
  (∀ x y, x^2 - y^2 / 3 = 1) := 
sorry

end hyperbola_equation_l165_165476


namespace smallest_c_l165_165036

theorem smallest_c (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) (hx_nonneg : ∀ i, 0 ≤ x i) :
  (∑ i in Finset.range n, ∑ j in Finset.range i, x (Fin.succ i) * x (Fin.succ j) * (x (Fin.succ i) ^ 2 + x (Fin.succ j) ^ 2)) ≤ 
  (1 / 8) * (∑ i in Finset.range n, x (Fin.succ i)) ^ 4 :=
by
  sorry

end smallest_c_l165_165036


namespace find_k_l165_165794

noncomputable def f (x : ℝ) : ℝ := Real.cot (x / 3) - Real.cot (2 * x)

theorem find_k :
  (∀ x : ℝ, f x = (sin (5 * x / 3)) / (sin (x / 3) * sin (2 * x))) :=
sorry

end find_k_l165_165794


namespace value_of_f_l165_165849

def f : ℝ → ℝ :=
λ x, if 4 ≤ x then 2 ^ x else f (x + 1)

theorem value_of_f : f (2 + Real.log 3 / Real.log 2) = 24 :=
sorry

end value_of_f_l165_165849


namespace sum_interesting_buildings_ge_27_specific_configuration_l165_165757

-- Given: 20 buildings with distinct floor numbers between 1 and 20, arranged in a circle.
-- Definitions and conditions
def interesting_building (B : ℕ → ℕ) (i : ℕ) : Prop :=
  B (i % 20) < B ((i + 1) % 20) ∧ B ((i + 1) % 20) > B ((i + 2) % 20) ∨
  B (i % 20) > B ((i + 1) % 20) ∧ B ((i + 1) % 20) < B ((i + 2) % 20)
  
def distinct_floors (B : ℕ → ℕ) : Prop :=
  ∀ i j, i ≠ j → B i ≠ B j

-- Theorem 1: Sum of floors of interesting buildings should be ≥ 27
theorem sum_interesting_buildings_ge_27 (B : ℕ → ℕ) :
  distinct_floors B ∧ (∀ i, 1 ≤ B i ∧ B i ≤ 20) →
  (∑ i in finset.range 20, if interesting_building B i then B i else 0) >= 27 :=
sorry

-- Theorem 2: Existence of a specific configuration with sum exactly 27
theorem specific_configuration:
  ∃ B : ℕ → ℕ, distinct_floors B ∧ (∀ i, 1 ≤ B i ∧ B i ≤ 20) ∧ 
  (∑ i in finset.range 20, if interesting_building B i then B i else 0 = 27) :=
sorry

end sum_interesting_buildings_ge_27_specific_configuration_l165_165757


namespace current_inventory_l165_165234

noncomputable def initial_books : ℕ := 743
noncomputable def fiction_books : ℕ := 520
noncomputable def nonfiction_books : ℕ := 123
noncomputable def children_books : ℕ := 100

noncomputable def saturday_instore_sales : ℕ := 37
noncomputable def saturday_fiction_sales : ℕ := 15
noncomputable def saturday_nonfiction_sales : ℕ := 12
noncomputable def saturday_children_sales : ℕ := 10
noncomputable def saturday_online_sales : ℕ := 128

noncomputable def sunday_instore_multiplier : ℕ := 2
noncomputable def sunday_online_addition : ℕ := 34

noncomputable def new_shipment : ℕ := 160

noncomputable def current_books := 
  initial_books 
  - (saturday_instore_sales + saturday_online_sales)
  - (sunday_instore_multiplier * saturday_instore_sales + saturday_online_sales + sunday_online_addition)
  + new_shipment

theorem current_inventory : current_books = 502 := by
  sorry

end current_inventory_l165_165234


namespace value_2_std_dev_less_than_mean_l165_165079

def mean : ℝ := 16.5
def std_dev : ℝ := 1.5

theorem value_2_std_dev_less_than_mean :
  mean - 2 * std_dev = 13.5 := by
  sorry

end value_2_std_dev_less_than_mean_l165_165079


namespace treasure_under_minimum_signs_l165_165273

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165273


namespace angle_AOB_l165_165913

/-- In triangle PAB, three tangents to circle O form the sides. Given ∠APB = 50° and ∠BAP = 30°,
    prove that ∠AOB = 65° --/
theorem angle_AOB {P A B O : Type} [Angle A P B] [Angle B A P] (h₁ : ∠ APB = 50) (h₂ : ∠ BAP = 30) : ∠ AOB = 65 := 
sorry

end angle_AOB_l165_165913


namespace grasshopper_cannot_return_to_origin_l165_165716

open Real

/-!
  A grasshopper starts jumping from the origin on a coordinate plane. The first jump is 1 cm long along the X-axis. 
  Each subsequent jump is 1 cm longer than the previous one, and is made perpendicular to the previous jump 
  in one of the two directions of the grasshopper’s choice. Prove that the grasshopper cannot return to the origin 
  after the 100th jump.
-/

theorem grasshopper_cannot_return_to_origin :
  ∀ (jumps : ℕ → (ℝ × ℝ)), 
    jumps 0 = (0, 0) ∧ -- Starts from the origin
    (∀ n, (jumps (n + 1)).fst = (jumps n).fst + (if n % 2 = 0 then n + 1 else 0) * (if n % 4 = 0 then 1 else -1)) ∧ -- Horizontal jumps
    (∀ n, (jumps (n + 1)).snd = (jumps n).snd + (if n % 2 = 1 then n + 1 else 0) * (if (n + 1) % 4 = 0 then 1 else -1)) ∧ -- Vertical jumps
    (∀ n, n < 100)
  → jumps 100 ≠ (0, 0) :=
begin
  sorry
end

end grasshopper_cannot_return_to_origin_l165_165716


namespace sqrt_of_sum_of_floors_l165_165806

theorem sqrt_of_sum_of_floors :
  let S := ∑ i in (Finset.range 1989), (Nat.floor (Real.sqrt i))
  ∈ {S // S = ∑ i in Finset.range 1989, (Nat.floor (Real.sqrt i))} →
  Nat.floor (Real.sqrt S) = 241 :=
by
  let S := ∑ i in (Finset.range 1989), (Nat.floor (Real.sqrt i))
  have hS: S = 58146 := sorry
  have hS_sqrt: Real.sqrt S ≈ 241 := sorry
  exact Nat.floor_of_real_floor hS_sqrt

end sqrt_of_sum_of_floors_l165_165806


namespace min_treasure_signs_buried_l165_165282

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165282


namespace contains_sufficient_triangles_l165_165583

-- Define the graph G with vertices and degrees
def is_graph (G : Type*) [fintype G] (V : G → finset G) : Prop :=
  ∀ u v : G, u ≠ v → (u ∈ V v ∧ v ∈ V u)

-- Define average degree condition
def avg_degree_condition (G : Type*) [fintype G] (V : G → finset G) (c : ℝ) (n : ℕ) :=
  (2 * ∑ v : G, (V v).card) / fintype.card G ≥ (1/2 + c) * n

-- Define existence of sufficient number of triangles
def sufficient_triangles (G : Type*) [fintype G] (V : G → finset G) (c : ℝ) (n : ℕ) :=
  ∃ T : ℕ, T ≥ c * nat.choose n 3

theorem contains_sufficient_triangles
  (G : Type*) [fintype G] (V : G → finset G) (c : ℝ) (n : ℕ)
  (h1 : is_graph G V)
  (h2 : c > 0)
  (h3 : avg_degree_condition G V c n) :
  sufficient_triangles G V c n :=
sorry

end contains_sufficient_triangles_l165_165583


namespace melting_point_of_ice_is_zero_degree_Celsius_l165_165671

theorem melting_point_of_ice_is_zero_degree_Celsius
    (boiling_point_water_F : ℝ)
    (boiling_point_water_C : ℝ)
    (melting_point_ice_F : ℝ)
    (pot_water_temp_C : ℝ)
    (pot_water_temp_F : ℝ)
    (conversion_f : ℝ → ℝ):
    boiling_point_water_F = 212 →
    boiling_point_water_C = 100 →
    melting_point_ice_F = 32 →
    pot_water_temp_C = 55 →
    pot_water_temp_F = 131 →
    conversion_f = (λ C, 9/5 * C + 32) →
    (∃ (melting_point_ice_C : ℝ), 
        conversion_f melting_point_ice_C = melting_point_ice_F 
        ∧ melting_point_ice_C = 0) :=
by
  intros
  sorry

end melting_point_of_ice_is_zero_degree_Celsius_l165_165671


namespace Karen_sold_boxes_l165_165925

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end Karen_sold_boxes_l165_165925


namespace prime_p_satisfies_conditions_l165_165353

theorem prime_p_satisfies_conditions (p : ℕ) (hp1 : Nat.Prime p) (hp2 : p ≠ 2) (hp3 : p ≠ 7) :
  ∃ n : ℕ, n = 29 ∧ ∀ x y : ℕ, (1 ≤ x ∧ x ≤ 29) ∧ (1 ≤ y ∧ y ≤ 29) → (29 ∣ (y^2 - x^p - 26)) :=
sorry

end prime_p_satisfies_conditions_l165_165353


namespace last_two_digits_sum_factorials_l165_165673

-- Define a function that computes the factorial of a number
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define the sum of factorials up to 14
def sum_of_factorials_to_14 : ℕ := ∑ i in finset.range 15, factorial i -- range 0 to 14

-- Prove the last two digits of this sum is 13
theorem last_two_digits_sum_factorials : sum_of_factorials_to_14 % 100 = 13 := by
  sorry

end last_two_digits_sum_factorials_l165_165673


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165139

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165139


namespace num_trucks_washed_l165_165563

theorem num_trucks_washed (total_revenue cars_revenue suvs_revenue truck_charge : ℕ) 
  (h_total : total_revenue = 100)
  (h_cars : cars_revenue = 7 * 5)
  (h_suvs : suvs_revenue = 5 * 7)
  (h_truck_charge : truck_charge = 6) : 
  ∃ T : ℕ, (total_revenue - suvs_revenue - cars_revenue) / truck_charge = T := 
by {
  use 5,
  sorry
}

end num_trucks_washed_l165_165563


namespace jack_and_jill_meet_distance_l165_165013

theorem jack_and_jill_meet_distance :
  ∃ t : ℝ, t = 15 / 60 ∧ 14 * t ≤ 4 ∧ 15 * (t - 15 / 60) ≤ 4 ∧
  ( 14 * t - 4 + 18 * (t - 2 / 7) = 15 * (t - 15 / 60) ∨ 15 * (t - 15 / 60) = 4 - 18 * (t - 2 / 7) ) ∧
  4 - 15 * (t - 15 / 60) = 851 / 154 :=
sorry

end jack_and_jill_meet_distance_l165_165013


namespace new_person_weight_is_correct_l165_165083

noncomputable def weight_of_person_replaced_kg : Int := 67
noncomputable def avg_weight_increase_kg : Int := 2.5
noncomputable def num_people : Int := 8
noncomputable def height_replaced_person_cm : Int := 171
noncomputable def height_new_person_cm : Int := 185
noncomputable def avg_age_of_group_years : Int := 32

theorem new_person_weight_is_correct :
  let total_weight_increase_kg := avg_weight_increase_kg * num_people,
      new_person_weight_kg := weight_of_person_replaced_kg + total_weight_increase_kg
  in new_person_weight_kg = 87 :=
by
  sorry

end new_person_weight_is_correct_l165_165083


namespace sum_of_divisors_of_12_l165_165394

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165394


namespace coefficient_x_8_expansion_l165_165544

theorem coefficient_x_8_expansion :
  (coeff (expand ((1+x)*(1+x^2)*(1+x^3)*(1+x^4)*(1+x^5)*(1+x^6)*(1+x^7)*(1+x^8)*(1+x^9)*(1+x^10))) 8) =
    number_of_ways_to_select_weights_sum_8 :=
sorry

end coefficient_x_8_expansion_l165_165544


namespace sum_of_fractions_eq_half_l165_165456

def coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

noncomputable def sum_fractions (n : ℕ) : ℚ :=
  ∑ p in Finset.range n, 
  ∑ q in Finset.range (n + 1), 
  if coprime p q ∧ 0 < p ∧ p < q ∧ q ≤ n ∧ p + q > n then (1 : ℚ) / (p * q) else 0

theorem sum_of_fractions_eq_half (n : ℕ) : sum_fractions n = 1 / 2 := 
  sorry

end sum_of_fractions_eq_half_l165_165456


namespace constant_term_f_f_x_l165_165594

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - (1 / x))^6
  else -real.sqrt x

theorem constant_term_f_f_x (x : ℝ) (hx : x > 0) : 
  let gx := (1 / real.sqrt x) - real.sqrt x in
  ∃ c : ℝ, (gx)^6 = c * x^0 ∧ c = -20 :=
  sorry

end constant_term_f_f_x_l165_165594


namespace total_basketballs_correct_l165_165947

-- Define the number of balls Lucca has
def lucca_balls : Nat := 100 

-- Define the percentage of Lucca's balls that are basketballs
def lucca_basketball_percentage : ℚ := 0.10 

-- Define the number of balls Lucien has
def lucien_balls : Nat := 200 

-- Define the percentage of Lucien's balls that are basketballs
def lucien_basketball_percentage : ℚ := 0.20 

-- Define the number of basketballs Lucca has
def lucca_basketballs : Nat := (lucca_basketball_percentage * lucca_balls).toNat 

-- Define the number of basketballs Lucien has
def lucien_basketballs : Nat := (lucien_basketball_percentage * lucien_balls).toNat 

-- Define the total number of basketballs Lucca and Lucien have:
def total_basketballs : Nat := lucca_basketballs + lucien_basketballs

-- The theorem to prove
theorem total_basketballs_correct : total_basketballs = 50 := by
  -- The proof will go here
  sorry

end total_basketballs_correct_l165_165947


namespace sum_divisors_of_12_l165_165399

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165399


namespace diff_cubes_square_of_squares_l165_165048

theorem diff_cubes_square_of_squares {x y : ℤ} (h1 : (x + 1) ^ 3 - x ^ 3 = y ^ 2) :
  ∃ (a b : ℤ), y = a ^ 2 + b ^ 2 ∧ a = b + 1 :=
sorry

end diff_cubes_square_of_squares_l165_165048


namespace triangle_area_l165_165483

theorem triangle_area (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * (a * b) = 24 := by
  sorry

end triangle_area_l165_165483


namespace rate_of_increase_twice_l165_165708

theorem rate_of_increase_twice {x : ℝ} (h : (1 + x)^2 = 2) : x = (Real.sqrt 2) - 1 :=
sorry

end rate_of_increase_twice_l165_165708


namespace planting_methods_l165_165971

theorem planting_methods :
  ∀ (plots : Finset (Fin 3)) (vegetables : Finset (String)),
    "cucumber" ∈ vegetables ∧ vegetables = finset.insert "cucumber" (finset.ofNat 3)
    → plot.card = 3 → vegetables.card = 4
    → ∃ methods : ℕ, methods = 18 := 
begin
  sorry
end

end planting_methods_l165_165971


namespace car_travel_time_difference_l165_165749

theorem car_travel_time_difference :
  ∀ (t m : ℝ), (35 * (t + m) = 98) ∧ (50 * t = 98) → m = 50.4 / 60 :=
by 
  intros t m h,
  cases h with hx hy,
  have ht : t = 1.96, 
  { rw [hy], exact div_eq_iff_eq_mul.mpr rfl },
  have hm : 35 * (1.96 + m) = 98 := hx,
  have : 68.6 + 35 * m = 98 := by exactly rfl,
  simp [hm] at *,
  sorry

end car_travel_time_difference_l165_165749


namespace invertible_interval_includes_2_l165_165972

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 8

-- Define the condition that g should be invertible in the interval that includes x = 2
theorem invertible_interval_includes_2 :
  ∃ I : set ℝ, I = set.Ici 1 ∧ (∀ x y ∈ I, g x = g y → x = y) ∧ (2 ∈ I) :=
by
  sorry

end invertible_interval_includes_2_l165_165972


namespace number_of_students_l165_165738

theorem number_of_students (total_students : ℕ) :
  (total_students = 19 * 6 + 4) ∧ 
  (∃ (x y : ℕ), x + y = 22 ∧ x > 7 ∧ total_students = x * 6 + y * 5) →
  total_students = 118 :=
by
  sorry

end number_of_students_l165_165738


namespace intersection_A_B_l165_165042

open Set

variable (x : ℝ)

def setA : Set ℝ := {x | x^2 - 3 * x ≤ 0}
def setB : Set ℝ := {1, 2}

theorem intersection_A_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_A_B_l165_165042


namespace _l165_165811

open EuclideanGeometry

noncomputable theorem midpoint_arc_bc (A B C M N W : Point) (O : Circle) 
  (h_tri: Triangle A B C) (h_circumcircle: Circle A B O) (h_AB_gt_AC : A ∉ Segment B C) 
  (h_AM_eq_AC: AM = AC) (h_m: M ∈ Line A B) (h_n: N ∈ Circle A B O) 
  (h_inter: ∀ (MN_line: Line) (circumcircle: Circle), 
    intersects_at N (MN_line) (circumcircle)) 
  : midpoint W (arc BC) := sorry

end _l165_165811


namespace percent_less_z_than_y_l165_165522

variable {w u y z : ℝ}

-- Conditions
def w_is_60_percent_u := w = 0.60 * u
def u_is_60_percent_y := u = 0.60 * y
def z_is_150_percent_w := z = 1.50 * w

-- Goal
theorem percent_less_z_than_y 
  (hw : w_is_60_percent_u)
  (hu : u_is_60_percent_y)
  (hz : z_is_150_percent_w) :
  (y - z) / y = 0.46 :=
by
  rw [w_is_60_percent_u] at hz,
  rw [u_is_60_percent_y] at hw,
  rw [hw, hu] at hz,
  rw [hz],
  sorry

end percent_less_z_than_y_l165_165522


namespace sum_positive_integral_values_l165_165383

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165383


namespace cos_sin_125_deg_pow_40_l165_165752

theorem cos_sin_125_deg_pow_40 :
  (complex.mk (real.cos 125) (real.sin 125))^40 = complex.mk (real.cos 40) (-real.sin 40) :=
sorry

end cos_sin_125_deg_pow_40_l165_165752


namespace range_of_a_perpendicular_tangents_l165_165853

theorem range_of_a_perpendicular_tangents :
  (∃ A B : ℝ, A ≠ B ∧ ∃ a : ℝ, ∀ a ∈ Set.Interval (-1) 1, 
   ∀ f (x : ℝ), f = a * x + Real.sin x + Real.cos x ∧
   ∃ f' : ℝ → ℝ, f' = λ x => a + Real.cos x - Real.sin x ∧
   (∀ x1 x2 : ℝ, x1 ≠ x2 → 
   (f' x1) * (f' x2) = -1)) → 
   a ∈ Set.Interval (-1) 1 :=
sorry

end range_of_a_perpendicular_tangents_l165_165853


namespace uniquely_determines_plane_l165_165687

theorem uniquely_determines_plane (A B C D : Prop) : 
  (A = "Three points") →
  (B = "Center and two points on a circle") →
  (C = "Two sides of a trapezoid") →
  (D = "A point and a line") →
  (∀ a, a = A → ¬(three_points_uniquely_determines_plane a)) →
  (∀ b, b = B → ¬(circle_center_and_points_uniquely_determines_plane b)) →
  (trapezoid_sides_uniquely_determines_plane C) →
  (∀ d, d = D → ¬(point_and_line_uniquely_determines_plane d)) →
  answer = C := by sorry

end uniquely_determines_plane_l165_165687


namespace count_valid_10_digit_integers_l165_165869

theorem count_valid_10_digit_integers : 
  ∃ n : ℕ, (∀ d₁ d₁' : ℕ, d₁ ∈ {1, 2, 3} ∧ d₁' ∈ {1, 2, 3} → d₁ ≠ d₁' → 
  ∃ (digits : Vector ℕ 10), digits.head = d₁ ∧ digits.last = d₁ ∧ 
  (∀ i, i < 9 → digits.get i ≠ digits.get (i + 1))) ∧ n = 768 := 
by 
  sorry

end count_valid_10_digit_integers_l165_165869


namespace triangle_sum_of_angles_120_l165_165247

theorem triangle_sum_of_angles_120 (a b c : ℝ) (h₀ : a = 15) (h₁ : b = 21) (h₂ : c = 24) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  (A + C = 120 * Real.pi / 180) :=
by
  have h3 : Real.cos B = (15^2 + 24^2 - 21^2) / (2 * 15 * 24) := by sorry
  have h4 : B = Real.acos ((15^2 + 24^2 - 21^2) / (2 * 15 * 24)) := by sorry
  have h5 : B = Real.pi / 3 := by sorry -- Given acos(1/2) = π/3 (i.e. 60 degrees)
  have h6 : A + C = 180 * Real.pi / 180 - B := by
    -- Sum of angles in a triangle
    sorry
  exact h6

end triangle_sum_of_angles_120_l165_165247


namespace infinite_coprime_binom_l165_165035

theorem infinite_coprime_binom (k l : ℕ) (hk : k > 0) (hl : l > 0) : 
  ∃ᶠ m in atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
sorry

end infinite_coprime_binom_l165_165035


namespace tobias_shoveled_driveways_l165_165112

open Nat

theorem tobias_shoveled_driveways 
  (shoe_cost : ℕ) (shoe_cost = 95)
  (change : ℕ) (change = 15)
  (allowance_per_month : ℕ) (allowance_per_month = 5)
  (months_saving : ℕ) (months_saving = 3)
  (charge_mow_lawn : ℕ) (charge_mow_lawn = 15)
  (charge_shovel_driveway : ℕ) (charge_shovel_driveway = 7)
  (num_lawns_mowed : ℕ) (num_lawns_mowed = 4) : 
  ∃ num_driveways_shoveled : ℕ, num_driveways_shoveled = 5 :=
by
  sorry

end tobias_shoveled_driveways_l165_165112


namespace vector_dot_product_and_period_l165_165801

open Real

/-- Given vectors a and b defined as (sqrt(3) * sin x, cos x) and (cos x, cos x) respectively.
  1. If a ⋅ b = 1 and x ∈ [-π/4, π/4], then x = 0.
  2. The function f(x) = a ⋅ b has period π and the interval of monotonic decrease is
     [kπ + π / 6, kπ + 2π / 3] for integers k. -/
theorem vector_dot_product_and_period (x : ℝ) :
  let a := (sqrt 3 * sin x, cos x)
  let b := (cos x, cos x)
  let f := λ x, (sqrt 3 * sin x * cos x + cos x * cos x)
  (a ⋅ b = 1) ∧ (x ∈ set.Icc (-π / 4) (π / 4)) ->
  (x = 0) ∧ 
  (∃ T ∈ ℝ, T = π) ∧ 
  (∀ k ∈ ℤ, ∃ I, I = set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) := 
by
    intros
    sorry

end vector_dot_product_and_period_l165_165801


namespace max_point_h_l165_165090

-- Definitions of the linear functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := -x - 3

-- The product of f(x) and g(x)
def h (x : ℝ) : ℝ := f x * g x

-- Statement: Prove that x = -2 is the maximum point of h(x)
theorem max_point_h : ∃ x_max : ℝ, h x_max = (-2) :=
by
  -- skipping the proof
  sorry

end max_point_h_l165_165090


namespace sum_of_fourth_powers_le_1000_l165_165148

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165148


namespace problem_statement_l165_165813

noncomputable def arithmetic_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

def a_sequence (n : ℕ) : ℕ := n + 1

def g_sequence (n : ℕ) : ℝ := 32 * (1/2) ^ (n - 1)

def T_n (n : ℕ) : ℝ := (n * (n + 3)) / 2 + 2 ^ (6 - n) - 64

theorem problem_statement (n : ℕ) (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 2) 
  (h2 : a2 = a1 + 1) 
  (h3 : a3 = a2 + 1) 
  (h4 : a4 = a3 + 1)
  (h5 : a5 = a4 + 1)
  (h_arith_sum : arithmetic_sum 5 2 1 = 20)
  (h_geo_common_ratio : ∀ k, g_sequence (k + 1) = 32 * (1 / 2) ^ k)
  (h_arith_b_geo : a4 + g_sequence 4 = 9) :
  ((∀ n : ℕ, a_sequence n = n + 1) ∧ (∀ n : ℕ, T_n n = (n * (n + 3)) / 2 + 2 ^ (6 - n) - 64)) :=
by {
    sorry
}

end problem_statement_l165_165813


namespace collinear_intersection_points_l165_165263

variables {A B C D Q1 Q2 Q3 Q4 P O : Type*}
variables {circle : Set (A)}
variables {perpendicular : ∀ X Y : A, Prop}

-- Definitions for conditions
def is_convex_quadrilateral (ABCD : Set (A)) : Prop := sorry
def is_inscribed_in_circle (A B C D : A) (circle : Set (A)) : Prop := sorry
def are_intersections_of_perpendiculars (Q1 Q2 Q3 Q4 : A) (A B C D : A) : Prop := sorry
def intersection_of_diagonals (P : A) (A B C D : A) : Prop := sorry
def center_of_circle (O : A) (circle : Set (A)) : Prop := sorry

-- Theorem statement
theorem collinear_intersection_points
  (ABCD : Set (A))
  (H1 : is_convex_quadrilateral ABCD)
  (H2 : is_inscribed_in_circle A B C D circle)
  (H3 : are_intersections_of_perpendiculars Q1 Q2 Q3 Q4 A B C D)
  (H4 : intersection_of_diagonals P A B C D)
  (H5 : center_of_circle O circle) :
  collinear Q1 Q2 Q3 Q4 P O := sorry

end collinear_intersection_points_l165_165263


namespace relationship_of_x_l165_165761

def f (x : ℝ) : ℝ := Real.log (x^2 + 1) + abs x

theorem relationship_of_x (x : ℝ) (h : f (2*x - 1) > f (x + 1)) : x ∈ (Set.Ioi 2) ∪ (Set.Iio 0) := 
sorry

end relationship_of_x_l165_165761


namespace jonas_shoes_l165_165018

theorem jonas_shoes (socks pairs_of_pants t_shirts shoes : ℕ) (new_socks : ℕ) (h1 : socks = 20) (h2 : pairs_of_pants = 10) (h3 : t_shirts = 10) (h4 : new_socks = 35 ∧ (socks + new_socks = 35)) :
  shoes = 35 :=
by
  sorry

end jonas_shoes_l165_165018


namespace probability_product_less_than_40_l165_165960

def pacoSpins (paco : ℕ) : Prop := paco ∈ {1, 2, 3, 4, 5, 6}
def manuSpins (manu : ℕ) : Prop := manu ∈ {1, 2, 3, 4, 5, 6, 7, 8}

def validProduct (paco manu : ℕ) : Prop := paco * manu < 40

theorem probability_product_less_than_40 : 
  (∑ p in {1, 2, 3, 4, 5, 6}, ∑ m in {1, 2, 3, 4, 5, 6, 7, 8}, 
    if validProduct p m then (1/6) * (1/8) else 0) = 15/16 := 
sorry

end probability_product_less_than_40_l165_165960


namespace min_a_b_l165_165443

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end min_a_b_l165_165443


namespace is_isosceles_right_triangle_l165_165062

def Point3D := (ℝ × ℝ × ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def A : Point3D := (4, 1, 9)
def B : Point3D := (10, -1, 6)
def C : Point3D := (2, 4, 3)

def AB := distance A B
def AC := distance A C
def BC := distance B C

theorem is_isosceles_right_triangle :
  AB = AC ∧ AB^2 + AC^2 = BC^2 :=
by
  sorry

end is_isosceles_right_triangle_l165_165062


namespace find_c_l165_165029

-- Define that \( r \) and \( s \) are roots of \( 2x^2 - 4x - 5 \)
variables (r s : ℚ)
-- Condition: sum of roots \( r + s = 2 \)
axiom sum_of_roots : r + s = 2
-- Condition: product of roots \( rs = -5/2 \)
axiom product_of_roots : r * s = -5 / 2

-- Definition of \( c \) based on the roots \( r-3 \) and \( s-3 \)
def c : ℚ := (r - 3) * (s - 3)

-- The theorem to be proved
theorem find_c : c = 1 / 2 :=
by
  sorry

end find_c_l165_165029


namespace probability_at_least_one_of_each_color_l165_165226

theorem probability_at_least_one_of_each_color
  (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h_total : total_balls = 16)
  (h_black : black_balls = 8)
  (h_white : white_balls = 5)
  (h_red : red_balls = 3) :
  ((black_balls.choose 1) * (white_balls.choose 1) * (red_balls.choose 1) : ℚ) / total_balls.choose 3 = 3 / 14 :=
by
  sorry

end probability_at_least_one_of_each_color_l165_165226


namespace evaluate_expression_l165_165349

theorem evaluate_expression : (1 / (5^2)^4) * 5^15 = 5^7 :=
by
  sorry

end evaluate_expression_l165_165349


namespace ratio_proof_l165_165921

-- Define the given conditions
def first_month_daily_earning : ℕ := 10
def days_per_month : ℕ := 30
def total_earning : ℕ := 1200

-- Define earning calculations
def first_month_total_earning : ℕ := first_month_daily_earning * days_per_month
noncomputable def second_month_daily_earning (x : ℝ) : ℝ := first_month_daily_earning * x
def second_month_total_earning (x : ℝ) : ℝ := second_month_daily_earning x * days_per_month
def third_month_total_earning (x : ℝ) : ℝ := second_month_daily_earning x * (days_per_month / 2)

-- Main statement proving the ratio
theorem ratio_proof : ∃ x : ℝ, first_month_total_earning + second_month_total_earning x + third_month_total_earning x = total_earning ∧ x = 2 :=
by
  sorry

end ratio_proof_l165_165921


namespace sum_of_fourth_powers_le_1000_l165_165143

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165143


namespace sum_of_first_41_terms_is_94_l165_165531

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end sum_of_first_41_terms_is_94_l165_165531


namespace external_angle_bisectors_quadrilateral_l165_165536

noncomputable theory

variables (A B C D E F G H : Type*)
variables [has_dist A] [has_dist B] [has_dist C] [has_dist D] [has_dist E] [has_dist F] [has_dist G] [has_dist H]

-- Given quadrilateral ABCD with external angle bisectors forming quadrilateral EFGH
variables (quadrilateral_ABCD_conditions : Prop)
  (angle_bisectors_form_quadrilateral_EFGH : quadrilateral_ABCD_conditions → Prop)

-- Prove that EG = AB + BC
theorem external_angle_bisectors_quadrilateral :
  quadrilateral_ABCD_conditions →
  angle_bisectors_form_quadrilateral_EFGH →
  dist (point E) (point G) = dist (point A) (point B) + dist (point B) (point C) :=
by sorry

end external_angle_bisectors_quadrilateral_l165_165536


namespace sum_of_divisors_of_12_l165_165374

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165374


namespace line_perp_to_plane_implies_perp_to_line_in_plane_l165_165467

-- Assume a definition for planes and perpendicular relationship for lines and planes.
variables {Point : Type} [EuclideanGeometry Point]
variables (a b : Line Point) (α : Plane Point)

-- Assumptions
axiom perp_line_plane (a : Line Point) (α : Plane Point) : Prop -- a ⊥ α
axiom line_in_plane (b : Line Point) (α : Plane Point) : Prop -- b ⊂ α
axiom perp_line_in_plane (a : Line Point) (b : Line Point) (α : Plane Point) 
  (haα : perp_line_plane a α) (hbα : line_in_plane b α) : perp a b -- ⊥ property in the plane

-- The statement to be proven
theorem line_perp_to_plane_implies_perp_to_line_in_plane 
  (haα : perp_line_plane a α) (hbα : line_in_plane b α) : perp a b :=
perp_line_in_plane a b α haα hbα

end line_perp_to_plane_implies_perp_to_line_in_plane_l165_165467


namespace probability_of_selection_of_X_l165_165668

theorem probability_of_selection_of_X 
  (P_Y : ℝ)
  (P_X_and_Y : ℝ) :
  P_Y = 2 / 7 →
  P_X_and_Y = 0.05714285714285714 →
  ∃ P_X : ℝ, P_X = 0.2 :=
by
  intro hY hXY
  sorry

end probability_of_selection_of_X_l165_165668


namespace midpoint_on_y_axis_l165_165490

noncomputable def f : ℝ → ℝ → ℝ := λ a x, a^x

theorem midpoint_on_y_axis (a : ℝ) (x₁ x₂ : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x₁ + x₂ = 0) :
  f a x₁ * f a x₂ = 1 :=
sorry

end midpoint_on_y_axis_l165_165490


namespace find_x_range_l165_165517

-- Define the condition for the expression to be meaningful
def meaningful_expr (x : ℝ) : Prop := x - 3 ≥ 0

-- The range of values for x is equivalent to x being at least 3
theorem find_x_range (x : ℝ) : meaningful_expr x ↔ x ≥ 3 := by
  sorry

end find_x_range_l165_165517


namespace find_unknown_towel_rate_l165_165200

noncomputable def unknown_towel_rate (x : ℤ) : Prop :=
  let cost_known_towels := 3 * 100 + 5 * 150  -- Total cost of known-priced towels
  let total_cost := 165 * 10  -- Total cost given by average
  cost_known_towels + 2 * x = total_cost

theorem find_unknown_towel_rate : ∃ x : ℤ, unknown_towel_rate x ∧ x = 300 :=
by
  -- assuming the equation holds
  existsi 300
  unfold unknown_towel_rate
  simp
  norm_num

end find_unknown_towel_rate_l165_165200


namespace probability_double_domino_l165_165220

def is_domino (a b : ℕ) : Prop := a ∈ finset.range 13 ∧ b ∈ finset.range 13

def is_double (a b : ℕ) : Prop := a = b

theorem probability_double_domino : 
  (finset.filter (λ (d : ℕ × ℕ), is_double d.1 d.2) (finset.product (finset.range 13) (finset.range 13))).card = 13 →
  (finset.product (finset.range 13) (finset.range 14)).card = 91 →
  (13 : ℚ) / 91 = 1 / 7 :=
by
  intros h1 h2
  rw h1 at *
  rw h2 at *
  norm_num
  sorry

end probability_double_domino_l165_165220


namespace smallest_positive_integer_prod_is_perfect_square_l165_165248

def x : ℕ := 2^2 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 9^9

theorem smallest_positive_integer_prod_is_perfect_square : 
  ∃ y : ℕ, (y * x).is_square ∧ (∀ z : ℕ, (z * x).is_square → y ≤ z) :=
by
  -- Proof goes here
  sorry

end smallest_positive_integer_prod_is_perfect_square_l165_165248


namespace quadrilateral_perimeter_l165_165624

theorem quadrilateral_perimeter (P A B C D : Point)
  (h1 : angle P A B = angle P D A)
  (h2 : angle P A D = angle P D C)
  (h3 : angle P B A = angle P C B)
  (h4 : angle P B C = angle P C D)
  (PA PB PC : ℝ)
  (hPA : PA = 4)
  (hPB : PB = 5)
  (hPC : PC = 10) :
  perimeter (A B C D) = 9 * real.sqrt 410 / 5 :=
sorry

end quadrilateral_perimeter_l165_165624


namespace largest_vertex_sum_l165_165445

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0) 
  (hA : (0 : ℤ, 0 : ℤ) ∈ ({(0, 0), (2*T, 0), (2*T + 1, 35)} : set (ℤ × ℤ))) 
  (hB : (2*T, 0) ∈ ({(0, 0), (2*T, 0), (2*T + 1, 35)} : set (ℤ × ℤ)))
  (hC : (2*T + 1, 35) ∈ ({(0, 0), (2*T, 0), (2*T + 1, 35)} : set (ℤ × ℤ))) :
  ∃ (N : ℤ), N = 34 :=
by
  sorry

end largest_vertex_sum_l165_165445


namespace cover_pentagon_with_circles_l165_165807

open Set

-- Definitions based on the conditions given:
structure Pentagon (α : Type) :=
(vertices : Fin 5 → α)
(all_angles_obtuse : ∀ i : Fin 5, obtuse (angle (vertices i) (vertices ((i + 1) % 5)) (vertices ((i + 2) % 5))))
(convex : Convex α (Range vertices))

noncomputable def obtuse (θ : ℝ) := θ > π/2 ∧ θ < π

noncomputable def angle (a b c : Point) : ℝ := -- some definition of the angle here (placeholder)
sorry

-- Math problem statement
theorem cover_pentagon_with_circles (P : Pentagon ℝ) :
  ∃ u v w x : Point, u ∈ Range P.vertices ∧ v ∈ Range P.vertices ∧ w ∈ Range P.vertices ∧ x ∈ Range P.vertices ∧
  let C₁ := Circle.mk (Segment.mk u v).midpoint ((Segment.mk u v).length / 2) in
  let C₂ := Circle.mk (Segment.mk w x).midpoint ((Segment.mk w x).length / 2) in
  ∀ q ∈ ConvexHull (Range P.vertices), q ∈ C₁ ∨ q ∈ C₂ :=
sorry

end cover_pentagon_with_circles_l165_165807


namespace subtract_scaled_vector_l165_165800

def vec_a : ℝ × ℝ × ℝ := (-7, 0, 1)
def vec_b : ℝ × ℝ × ℝ := (4, 2, -1)

theorem subtract_scaled_vector :
  (vec_a.1 - 3 * vec_b.1, vec_a.2 - 3 * vec_b.2, vec_a.3 - 3 * vec_b.3) = (-19, -6, 4) :=
by 
  sorry

end subtract_scaled_vector_l165_165800


namespace problem_1_problem_2_l165_165332

def simplify_calc : Prop :=
  125 * 3.2 * 25 = 10000

def solve_equation : Prop :=
  ∀ x: ℝ, 24 * (x - 12) = 16 * (x - 4) → x = 28

theorem problem_1 : simplify_calc :=
by
  sorry

theorem problem_2 : solve_equation :=
by
  sorry

end problem_1_problem_2_l165_165332


namespace train_length_is_correct_l165_165709

noncomputable def length_of_train
  (train_speed_kmh : ℝ)
  (man_speed_kmh : ℝ)
  (time_to_cross_s : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 5) / 18
  in relative_speed_ms * time_to_cross_s

theorem train_length_is_correct :
  length_of_train 63 3 35.99712023038157 = 600 :=
by
  sorry

end train_length_is_correct_l165_165709


namespace return_time_is_2_hours_l165_165927

noncomputable def distance_home_city_hall := 6
noncomputable def speed_to_city_hall := 3 -- km/h
noncomputable def additional_distance_return := 2 -- km
noncomputable def speed_return := 4 -- km/h
noncomputable def total_trip_time := 4 -- hours

theorem return_time_is_2_hours :
  (distance_home_city_hall + additional_distance_return) / speed_return = 2 :=
by
  sorry

end return_time_is_2_hours_l165_165927


namespace number_exceeds_its_part_by_20_l165_165236

theorem number_exceeds_its_part_by_20 (x : ℝ) (h : x = (3/8) * x + 20) : x = 32 :=
sorry

end number_exceeds_its_part_by_20_l165_165236


namespace sum_of_divisors_of_12_l165_165377

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165377


namespace contrapositive_inverse_converse_true_propositions_count_l165_165496

theorem contrapositive_inverse_converse_true_propositions_count {a b : ℝ} :
  (a^2 + b^2 = 0) → (a^2 - b^2 = 0) →
  (1 = fintype.card {q : Prop // (q = ((a^2 + b^2 = 0) → (a^2 - b^2 = 0)) ∨
                                      (q = ((a^2 - b^2 ≠ 0) → (a^2 + b^2 ≠ 0))) ∨
                                      (q = ((a^2 - b^2 = 0) → (a^2 + b^2 = 0))) ∨
                                      (q = ((a^2 + b^2 ≠ 0) → (a^2 - b^2 ≠ 0)))
                                     ) ∧ q = true {/* Proof of truth value */})
  ) :=
sorry

end contrapositive_inverse_converse_true_propositions_count_l165_165496


namespace sum_of_fourth_powers_le_1000_l165_165147

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165147


namespace locus_of_centers_proof_l165_165459

noncomputable def locus_of_centers (r : ℝ) (P O : EuclideanSpace ℝ 3) (OP : ℝ) (h_unit_sphere : ∥O∥ = 1): Set (EuclideanSpace ℝ 3) :=
  if h1 : OP < sqrt (1 - r^2) then ∅
  else if h2 : OP = sqrt (1 - r^2) then {P}
  else
    let O' := O + (1 - r^2) / OP • (P - O) in
    let r' := sqrt ((1 - r^2) * (OP^2 + r^2 - 1)) / OP in
    sphere O' r'

theorem locus_of_centers_proof (r : ℝ) (P O : EuclideanSpace ℝ 3) (OP : ℝ) (h_unit_sphere : ∥O∥ = 1) (h_pos : 0 < r) (h_le_one : r ≤ 1) :
  locus_of_centers r P O OP h_unit_sphere =
    if h1 : OP < sqrt (1 - r^2) then ∅
    else if h2 : OP = sqrt (1 - r^2) then {P}
    else 
      let O' := O + (1 - r^2) / OP • (P - O) in
      let r' := sqrt ((1 - r^2) * (OP^2 + r^2 - 1)) / OP in
      sphere O' r' :=
sorry

end locus_of_centers_proof_l165_165459


namespace walk_to_lake_park_restaurant_is_zero_l165_165917

noncomputable def time_to_hidden_lake : ℕ := 15
noncomputable def time_to_return_from_hidden_lake : ℕ := 7
noncomputable def total_walk_time_dante : ℕ := 22

theorem walk_to_lake_park_restaurant_is_zero :
  ∃ (x : ℕ), (2 * x + time_to_hidden_lake + time_to_return_from_hidden_lake = total_walk_time_dante) → x = 0 :=
by
  use 0
  intros
  sorry

end walk_to_lake_park_restaurant_is_zero_l165_165917


namespace problem_statement_l165_165470

theorem problem_statement (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1 / 2) :
  a^{\sqrt{a}} > a^{a^{a}} ∧ a^{a^{a}} > a :=
sorry

end problem_statement_l165_165470


namespace find_x_l165_165980

theorem find_x :
  ∃ x : ℕ, (x + 35 + 58) / 3 = ((19 + 51 + 29) / 3) + 6 :=
begin
  use 24,
  sorry
end

end find_x_l165_165980


namespace varpi_value_l165_165887

theorem varpi_value (varpi : ℝ) (h0 : varpi > 0)
  (h1 : ∀ x1 x2 : ℝ, (sin (varpi * x1 + π / 8) = 0) ∧ (sin (varpi * x2 + π / 8) = 0) ∧ (x2 - x1 = π / 6)) :
  varpi = 6 := by
  sorry

end varpi_value_l165_165887


namespace sum_of_areas_l165_165754

theorem sum_of_areas (r s t : ℝ)
  (h1 : r + s = 13)
  (h2 : s + t = 5)
  (h3 : r + t = 12)
  (h4 : t = r / 2) : 
  π * (r ^ 2 + s ^ 2 + t ^ 2) = 105 * π := 
by
  sorry

end sum_of_areas_l165_165754


namespace probability_sum_15_pair_octahedral_dice_l165_165089

theorem probability_sum_15_pair_octahedral_dice (n : ℕ) (h1 : n = 8) :
  (∃ (A B : fin n → fin n), (A + B = 15) ∧ ((A, B) = (7, 8) ∨ (A, B) = (8, 7))) → 
  (2 / (n * n) = 1 / 32) :=
by
  sorry

end probability_sum_15_pair_octahedral_dice_l165_165089


namespace tangent_parallel_to_AD_l165_165214

variable (A B C D E F: Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variable (ω1 ω2: Type) [Circle ω1] [Circle ω2]

-- Let quadrilateral ABCD be inside circle ω1 (circumscribed circle around quadrilateral ABCD).
variable (CircumscribedQuadrilateral: circumscribed_quadrilateral ω1 A B C D)

-- Let circle ω2 pass through points A and B, intersecting ray DB at point E ≠ B.
variable (PassesThroughAB: passes_through ω2 A B)
variable (EneqB: E ≠ B)
variable (IntersectsDB: ray_intersection ω2 (ray D B) E)

-- Ray CA intersects circle ω2 at point F ≠ A.
variable (IntersectsCA: ray_intersection ω2 (ray C A) F)
variable (FneqA: F ≠ A)

-- The tangent to circle ω1 at point C is parallel to line AE.
variable (TangentParallelToAE: tangent_parallel ω1 C (line A E))

-- Prove that the tangent at F to circle ω2 is parallel to line AD.
theorem tangent_parallel_to_AD :
  tangent_parallel ω2 F (line A D) :=
sorry

end tangent_parallel_to_AD_l165_165214


namespace find_a_plus_b_l165_165577

theorem find_a_plus_b :
  ∃ (a b : ℝ), (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 5) ∧ a + b = 5 :=
by 
  sorry

end find_a_plus_b_l165_165577


namespace increase_in_votes_l165_165529

noncomputable def initial_vote_for (y : ℝ) : ℝ := 500 - y
noncomputable def revote_for (y : ℝ) : ℝ := (10 / 9) * y

theorem increase_in_votes {x x' y m : ℝ}
  (H1 : x + y = 500)
  (H2 : y - x = m)
  (H3 : x' - y = 2 * m)
  (H4 : x' + y = 500)
  (H5 : x' = (10 / 9) * y)
  (H6 : y = 282) :
  revote_for y - initial_vote_for y = 95 :=
by sorry

end increase_in_votes_l165_165529


namespace log_ratio_inequality_l165_165607

theorem log_ratio_inequality {a b c : ℝ} (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) : 
    (ln a / ((a - b) * (a - c))) + (ln b / ((b - c) * (b - a))) + (ln c / ((c - a) * (c - b))) < 0 := 
by {
    intros,
    sorry
}

end log_ratio_inequality_l165_165607


namespace Karen_sold_boxes_l165_165926

theorem Karen_sold_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by
  sorry

end Karen_sold_boxes_l165_165926


namespace minimum_treasure_buried_l165_165288

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165288


namespace num_non_congruent_triangles_with_perimeter_9_l165_165871

theorem num_non_congruent_triangles_with_perimeter_9 :
  ∃ n : ℕ, n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a + b > c → a + c > b → b + c > a → n = 2 :=
begin
  sorry
end

end num_non_congruent_triangles_with_perimeter_9_l165_165871


namespace cosine_of_angle_between_planes_l165_165835

open Real

noncomputable def n1 : ℝ × ℝ × ℝ := (3, 2, 1)
noncomputable def n2 : ℝ × ℝ × ℝ := (2, 0, -1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
def norm (v : ℝ × ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cosine_between_planes (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (norm v1 * norm v2)

theorem cosine_of_angle_between_planes :
  cosine_between_planes n1 n2 = sqrt 70 / 14 := 
  sorry

end cosine_of_angle_between_planes_l165_165835


namespace range_of_x_l165_165485

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_x (x : ℝ) (h₀ : -1 < x ∧ x < 1) (h₁ : f 0 = 0) (h₂ : f (1 - x) + f (1 - x^2) < 0) :
  1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end range_of_x_l165_165485


namespace ratio_of_P_Q_l165_165995

theorem ratio_of_P_Q (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 4 →
    P / (x + 5) + Q / (x^2 - 4 * x) = (x^2 + x + 15) / (x^3 + x^2 - 20 * x)) :
  Q / P = -45 / 2 :=
by
  sorry

end ratio_of_P_Q_l165_165995


namespace min_treasure_count_l165_165318

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165318


namespace correlation_snoring_heart_disease_l165_165896

theorem correlation_snoring_heart_disease {patients : ℕ} (correlation_confidence : ℝ) :
  (correlation_confidence > 99 / 100) ∧ 
  (patients = 100) →
  ∃ (snoring_patients : ℕ → Prop), 
    (∀ p, snoring_patients p → p < patients) ∧ 
    (∃ p, ¬ snoring_patients p) :=
by
  sorry

end correlation_snoring_heart_disease_l165_165896


namespace sam_puppies_count_l165_165617

variable (initial_puppies : ℝ) (given_away_puppies : ℝ)

theorem sam_puppies_count (h1 : initial_puppies = 6.0) 
                          (h2 : given_away_puppies = 2.0) : 
                          initial_puppies - given_away_puppies = 4.0 :=
by simp [h1, h2]; sorry

end sam_puppies_count_l165_165617


namespace problem_statement_l165_165977

noncomputable def f : ℝ → ℝ := sorry
variable {k : ℝ} (hk : k > 0) (hf : ∀ x, f x = Real.exp (-k * x))

theorem problem_statement (f_pos : ∀ x, f x > 0)
  (f_mult : ∀ a b, f a * f b = f (a + b)) 
  (f_zero : f 0 = 1) 
  (f_inv : ∀ a, f (-a) = 1 / f a) 
  (f_cubic : ∀ a, f a = Real.cbrt (f (3 * a))) 
  (f_order : ∀ a b, a < b → f a > f b) 
  (hf_f0 : f 0 = 1) 
  (hf_fneg : ∀ a, f (-a) = 1 / f a) 
  (hf_fcubic : ∀ a, f a = Real.cbrt (f (3 * a))) 
  (hf_forder : ∀ a b, a < b → f a > f b) 
  : true :=
sorry

end problem_statement_l165_165977


namespace max_value_on_interval_max_value_at_one_l165_165854

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - 3

theorem max_value_on_interval (a : ℝ) (h_a : a = 2) : 
  ∃ x ∈ set.Icc 3 5, f x a = 2 :=
by {
  use 5,
  split,
  linarith,
  linarith,
  rw [h_a],
  unfold f,
  norm_num,
  ring,
  linarith,
  sorry
}

theorem max_value_at_one (a : ℝ) (h_a : 2 ≤ a) : 
  f 1 a ≤ -6 :=
by {
  unfold f,
  rw[←sub_nonneg],
  norm_num,
  nlinarith,
  linarith,
  sorry
}

end max_value_on_interval_max_value_at_one_l165_165854


namespace polynomial_solution_characterization_l165_165780

theorem polynomial_solution_characterization :
  ∀ (p : ℝ[X]), (∀ x : ℝ, (p.eval (x^2 + x + 1)) = (p.eval x) * (p.eval (x + 1)))
  ↔ p = 0 ∨ p = 1 ∨ ∃ n : ℕ, p = (X^2 + 1)^n := by
sorry

end polynomial_solution_characterization_l165_165780


namespace sum_of_divisors_of_12_l165_165375

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165375


namespace casper_initial_candies_l165_165750

theorem casper_initial_candies : ∃ x : ℕ,
  (let after_first_day := x / 2 - 3 in
   let after_second_day := after_first_day / 4 - 5 in
   let after_third_day := after_second_day / 6 - 2 in
   after_third_day - 10 = 0) ∧
  x = 622 :=
by
  use 622
  let after_first_day := 622 / 2 - 3
  let after_second_day := after_first_day / 4 - 5
  let after_third_day := after_second_day / 6 - 2
  have : after_third_day = 10 := sorry
  split
  case _ => sorry -- Proof involving calculations and checks for after_third_day result
  case _ => rfl

end casper_initial_candies_l165_165750


namespace max_single_player_salary_l165_165242

variable (n : ℕ) (m : ℕ) (p : ℕ) (s : ℕ)

theorem max_single_player_salary
  (h1 : n = 18)
  (h2 : ∀ i : ℕ, i < n → p ≥ 20000)
  (h3 : s = 800000)
  (h4 : n * 20000 ≤ s) :
  ∃ x : ℕ, x = 460000 :=
by
  sorry

end max_single_player_salary_l165_165242


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165136

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165136


namespace f_l165_165475

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x * f' 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * f' 1

theorem f'_0_eq_neg_2 : f' 0 = -2 := by
  sorry

end f_l165_165475


namespace option_A_correct_l165_165041

variable (f g : ℝ → ℝ)

-- Given conditions
axiom cond1 : ∀ x : ℝ, f x - g (4 - x) = 2
axiom cond2 : ∀ x : ℝ, deriv g x = deriv f (x - 2)
axiom cond3 : ∀ x : ℝ, f (x + 2) = - f (- x - 2)

theorem option_A_correct : ∀ x : ℝ, f (4 + x) + f (- x) = 0 :=
by
  -- Proving the theorem
  sorry

end option_A_correct_l165_165041


namespace forty_percent_of_number_l165_165959

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 20) : 0.40 * N = 240 :=
by
  sorry

end forty_percent_of_number_l165_165959


namespace correct_option_among_given_conditions_l165_165732

-- Definitions based on conditions
def Condition_A : Prop := ∀ (A B C : Point), (¬Collinear A B C) → ∃ plane : Plane, (A ∈ plane ∧ B ∈ plane ∧ C ∈ plane)
def Condition_B : Prop := ∀ (l : Line) (P : Point), (P ∉ l) → ∃ plane : Plane, (P ∈ plane ∧ ∀ q ∈ l, q ∈ plane)
def Condition_C : Prop := ∀ (l1 l2 l3 : Line), (Parallel l1 l2 ∧ ∃ P, (P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3)) → (∃ plane, (l1 ∈ plane ∧ l3 ∈ plane))
def Condition_D : Prop := ∀ (l1 l2 l3 : Line), (Intersects l1 l2 ∧ Intersects l2 l3 ∧ Intersects l1 l3) → (SamePlane l1 l2 l3)

-- Statement to prove
theorem correct_option_among_given_conditions : Condition_C := 
by 
  sorry

end correct_option_among_given_conditions_l165_165732


namespace max_robot_weight_calc_l165_165056

theorem max_robot_weight_calc :
  let standard_weight := 100
  let battery_weight := 20
  let min_weight_without_battery := standard_weight + 5
  let min_weight_with_battery := min_weight_without_battery + battery_weight
  let max_weight_with_battery := 2 * min_weight_with_battery
  max_weight_with_battery = 250 := 
by {
  -- Define all the constants and assumptions
  let standard_weight := 100
  let battery_weight := 20
  let min_weight_without_battery := standard_weight + 5
  let min_weight_with_battery := min_weight_without_battery + battery_weight
  let max_weight_with_battery := 2 * min_weight_with_battery

  -- Proceed with the necessary calculations
  calc
    max_weight_with_battery
        = 2 * (min_weight_without_battery + battery_weight) : by rw min_weight_without_battery
    ... = 2 * (105 + battery_weight) : by unfold min_weight_without_battery
    ... = 2 * (105 + 20) : by rw battery_weight
    ... = 2 * 125 : by norm_num
    ... = 250 : by norm_num
}

end max_robot_weight_calc_l165_165056


namespace slices_per_large_pizza_l165_165244

structure PizzaData where
  total_pizzas : Nat
  small_pizzas : Nat
  medium_pizzas : Nat
  slices_per_small : Nat
  slices_per_medium : Nat
  total_slices : Nat

def large_slices (data : PizzaData) : Nat := (data.total_slices - (data.small_pizzas * data.slices_per_small + data.medium_pizzas * data.slices_per_medium)) / (data.total_pizzas - data.small_pizzas - data.medium_pizzas)

def PizzaSlicingConditions := {data : PizzaData // 
  data.total_pizzas = 15 ∧
  data.small_pizzas = 4 ∧
  data.medium_pizzas = 5 ∧
  data.slices_per_small = 6 ∧
  data.slices_per_medium = 8 ∧
  data.total_slices = 136}

theorem slices_per_large_pizza (data : PizzaSlicingConditions) : large_slices data.val = 12 :=
by
  sorry

end slices_per_large_pizza_l165_165244


namespace triangle_inequality_l165_165592

variable {X Y Z : ℝ}
  
theorem triangle_inequality 
  (h : X + Y + Z = π/2) : 
  (sin X / cos (Y - Z)) + (sin Y / cos (Z - X)) + (sin Z / cos (X - Y)) ≥ 3 / 2 := by 
  sorry

end triangle_inequality_l165_165592


namespace arithmetic_sequence_nth_term_l165_165637

theorem arithmetic_sequence_nth_term (x n : ℝ) 
  (h1 : 3*x - 4 = a1)
  (h2 : 7*x - 14 = a2)
  (h3 : 4*x + 6 = a3)
  (h4 : a_n = 3012) :
n = 392 :=
  sorry

end arithmetic_sequence_nth_term_l165_165637


namespace point_on_graph_inv_prop_point_b_on_graph_l165_165688

theorem point_on_graph_inv_prop (k : ℝ) (h : k = -4) (x y : ℝ) (hx : x = -2) (hy : y = 2) : x * y = k :=
by
  rw [hx, hy]
  -- Now we need to prove that (-2) * 2 = -4
  -- Since k = -4, we can proceed to simplify
  simp
  exact h

/-- Given point (-2, 2) lies on the graph of the function y = -4/x -/
theorem point_b_on_graph : (-2:ℝ) * 2 = -4 := by
  exact point_on_graph_inv_prop (-4) rfl (-2) 2 rfl rfl

end point_on_graph_inv_prop_point_b_on_graph_l165_165688


namespace quarry_sales_correct_l165_165240

noncomputable def total_sales (p : ℕ → ℝ) :=
  ∑' n, 0.6 * 0.4^(n-1) * 0.9^(n-1) * 3200

theorem quarry_sales_correct :
  total_sales (λ n, 3200 * 0.9^(n-1)) = 3000 :=
by sorry

end quarry_sales_correct_l165_165240


namespace graph_edge_labeling_l165_165934

-- Define the connected graph G and the number of edges k
variables {G : Type} [Graph G] (v : G) (E : set (v → ℕ))
variable (k : ℕ)

-- Condition: G is a connected graph
def is_connected (G : Type) [Graph G] : Prop := sorry

-- Condition: G has k edges
def has_k_edges (G : Type) [Graph G] (k : ℕ) : Prop := sorry

-- Definition of the problem in Lean 4
theorem graph_edge_labeling (hconn : is_connected G) (hkedges : has_k_edges G k) :
  ∃ (label : E → ℕ), (∀ v : G, ∀ e₁ e₂ ∈ E v, v.degree ≥ 2 → GCD (label e₁) (label e₂) = 1) := sorry

end graph_edge_labeling_l165_165934


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165142

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165142


namespace max_height_arithmetic_sequence_max_height_geometric_sequence_l165_165723

/-- 
  Problem (1): Prove that the maximum height the model airplane will ascend to 
  in the first scenario is 64 meters given the conditions of the arithmetic sequence.
 -/

theorem max_height_arithmetic_sequence :
  let a_1 := 15
  let d := -2
  ∃ n, ∑ i in finset.range n, (a_1 + i * d) = 64 :=
by {
  let n := 8,
  use n,
  sorry
}


/-- 
  Problem (2): Prove that the model airplane's maximum ascending height in the second scenario 
  cannot exceed 75 meters given the conditions of the geometric sequence.
 -/

theorem max_height_geometric_sequence :
  let b_1 :=  15
  let q := 0.8
  ∀ (n : ℕ), ∑ i in finset.range (n+1), (b_1 * q^i) ≤ 75 :=
by {
  sorry
}

end max_height_arithmetic_sequence_max_height_geometric_sequence_l165_165723


namespace solve_for_x_l165_165974

theorem solve_for_x : (1 / 3 - 1 / 4) * 2 = 1 / 6 :=
by
  -- Sorry is used to skip the proof; the proof steps are not included.
  sorry

end solve_for_x_l165_165974


namespace find_b_value_l165_165004

noncomputable def tangent_line_slope_eq_curve_slope (b : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀ > 0 ∧ (1 / x₀ = 1 ∧ (ln x₀) = 0 ∧ (1 + b = 0))

theorem find_b_value (b : ℝ) : tangent_line_slope_eq_curve_slope b ↔ b = -1 := by
  sorry

end find_b_value_l165_165004


namespace calculate_f_neg10_l165_165824

def f (x : ℝ) (a : ℝ) : ℝ := |x| * (Real.exp (a * x) - Real.exp (-a * x)) + 2

theorem calculate_f_neg10 (a : ℝ) (h₁ : f 10 a = 1) : f (-10) a = 3 := by
  sorry

end calculate_f_neg10_l165_165824


namespace sum_of_divisors_of_12_l165_165405

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165405


namespace chord_bisected_by_point_P_line_equation_l165_165464

theorem chord_bisected_by_point_P_line_equation :
  let ellipse := λ (x y : ℝ), x^2 / 16 + y^2 / 4 = 1
  let point_P := (3 : ℝ, 1 : ℝ)
  ∀ (P1 P2 : ℝ × ℝ),
  let (x₁, y₁) := P1
  let (x₂, y₂) := P2
  (x₁ + x₂ = 6) →
  (y₁ + y₂ = 2) →
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  (3 * 3 + 4 * 1 - 13 = 0) :=
by
  intros ellipse point_P P1 P2 x₁ y₁ x₂ y₂ h1 h2 h3 h4
  sorry

end chord_bisected_by_point_P_line_equation_l165_165464


namespace treasure_under_minimum_signs_l165_165275

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165275


namespace Roselyn_initial_books_correct_l165_165616

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end Roselyn_initial_books_correct_l165_165616


namespace remainder_of_8x_minus_5_l165_165195

theorem remainder_of_8x_minus_5 (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 :=
by
  sorry

end remainder_of_8x_minus_5_l165_165195


namespace sum_positive_integral_values_l165_165378

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165378


namespace triangle_acute_l165_165521

theorem triangle_acute (a b c : ℝ) (h : {a, b, c} = {3, 4, 4.5}) : 
  (a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :=
by
  sorry

end triangle_acute_l165_165521


namespace sum_of_values_l165_165191

theorem sum_of_values (x : ℝ) (h : sqrt ((x - 2)^2) = 8) : (x = 10 ∨ x = -6) ∧ (10 + (-6) = 4) :=
by
  sorry

end sum_of_values_l165_165191


namespace sum_of_perfect_square_squares_less_than_1000_l165_165169

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165169


namespace sum_of_perfect_square_squares_less_than_1000_l165_165170

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165170


namespace tony_drive_time_l165_165666

noncomputable def time_to_first_friend (d₁ d₂ t₂ : ℝ) : ℝ :=
  let v := d₂ / t₂
  d₁ / v

theorem tony_drive_time (d₁ d₂ t₂ : ℝ) (h_d₁ : d₁ = 120) (h_d₂ : d₂ = 200) (h_t₂ : t₂ = 5) : 
    time_to_first_friend d₁ d₂ t₂ = 3 := by
  rw [h_d₁, h_d₂, h_t₂]
  -- Further simplification would follow here based on the proof steps, which we are omitting
  sorry

end tony_drive_time_l165_165666


namespace probability_of_winning_l165_165710

def total_products_in_box : ℕ := 6
def winning_products_in_box : ℕ := 2

theorem probability_of_winning : (winning_products_in_box : ℚ) / (total_products_in_box : ℚ) = 1 / 3 :=
by sorry

end probability_of_winning_l165_165710


namespace parabola_properties_l165_165355

theorem parabola_properties :
  let a := -2
  let b := 4
  let c := 8
  ∃ h k : ℝ, 
    (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
    (h = 1) ∧ 
    (k = 10) ∧ 
    (a < 0) ∧ 
    (axisOfSymmetry = h) ∧ 
    (vertex = (h, k)) :=
by
  sorry

end parabola_properties_l165_165355


namespace jason_cuts_lawns_l165_165919

theorem jason_cuts_lawns 
  (time_per_lawn: ℕ)
  (total_cutting_time_hours: ℕ)
  (total_cutting_time_minutes: ℕ)
  (total_yards_cut: ℕ) : 
  time_per_lawn = 30 → 
  total_cutting_time_hours = 8 → 
  total_cutting_time_minutes = total_cutting_time_hours * 60 → 
  total_yards_cut = total_cutting_time_minutes / time_per_lawn → 
  total_yards_cut = 16 :=
by
  intros
  sorry

end jason_cuts_lawns_l165_165919


namespace sum_of_divisors_of_12_l165_165435

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165435


namespace integral_of_quarter_circle_l165_165347

noncomputable def integral_value : ℝ :=
  ∫ x in 0..1, real.sqrt (1 - (x - 1) ^ 2)

theorem integral_of_quarter_circle :
  integral_value = real.pi / 4 :=
sorry

end integral_of_quarter_circle_l165_165347


namespace greatest_x_l165_165695

theorem greatest_x (x : ℤ) (h1 : 2.134 * 10^x < 21000) : x ≤ 3 :=
begin
  sorry
end

end greatest_x_l165_165695


namespace part1_part2_part3_l165_165808

section problem

def A_plus (A : set ℤ) : set ℤ := {x | ∃ a b ∈ A, x = a + b}
def A_minus (A : set ℤ) : set ℤ := {x | ∃ a b ∈ A, x = abs (a - b)}

-- Part (1)
theorem part1 (A : set ℤ) (h : A = {-1, 1}) : 
  A_plus A = {-2, 0, 2} ∧ A_minus A = {0, 2} := 
sorry

-- Part (2)
theorem part2 {x1 x2 x3 x4 : ℤ} (h : x1 < x2 ∧ x2 < x3 ∧ x3 < x4) (A : set ℤ) (hA : A = {x1, x2, x3, x4}) 
  (hA_minus : A_minus A = A) : x1 + x4 = x2 + x3 := 
sorry

-- Part (3)
theorem part3 (A : set ℕ) (h1 : A ⊆ {x | 0 ≤ x ∧ x ≤ 2020})
  (h2 : A_plus A ∩ A_minus A = ∅) : ∃ k : ℕ, k = 1347 ∧ k = A.card := 
sorry

end problem

end part1_part2_part3_l165_165808


namespace max_non_disjoint_3_element_subsets_l165_165020

open Finset

variable {α : Type} 

-- Definition of a set with n elements 
def X (n : ℕ) : Finset (Fin n) := univ

-- The statement of the problem
theorem max_non_disjoint_3_element_subsets (n : ℕ) (hn : n ≥ 6) :
  ∃ (F : Finset (Finset (Fin n))), 
    (∀ A ∈ F, A.card = 3) ∧ 
    (∀ A B ∈ F, A ≠ B → A ∩ B ≠ ∅) ∧ 
    F.card = choose (n - 1) 2 :=
sorry

end max_non_disjoint_3_element_subsets_l165_165020


namespace ways_to_schedule_courses_l165_165508

theorem ways_to_schedule_courses : 
  ∃ (n : ℕ), 
    (n = 60) ∧ 
    (∀ (P : Finset ℕ), 
      P.card = 3 → 
      P ⊆ Finset.range 7 → 
      ∀ (p1 p2 : ℕ), 
        p1 ∈ P → p2 ∈ P → 
        abs (p1 - p2) ≠ 1 
    ) :=
sorry

end ways_to_schedule_courses_l165_165508


namespace graph_intersects_self_9_times_between_2_and_60_l165_165235

noncomputable def intersections_between_2_and_60 : ℕ :=
  let x := λ t : ℝ, Real.cos t + t / 3
  let y := λ t : ℝ, Real.sin t
  -- Here, we want to count the number of times the graph intersects itself
  -- between x = 2 and x = 60.
  if points := {t: ℝ | 2 ≤ x t ∧ x t ≤ 60} then -- using intersections conditions
    points.count sorry -- define count function and conditions correctly based on given answer
  else
    0

theorem graph_intersects_self_9_times_between_2_and_60 :
  intersections_between_2_and_60 = 9 := sorry

end graph_intersects_self_9_times_between_2_and_60_l165_165235


namespace log_trig_identity_l165_165701

theorem log_trig_identity :
  log 2 (Real.sin (Real.pi / 12)) - log (1 / 2) (Real.cos (23 * Real.pi / 12))
  = -2 :=
by
  sorry

end log_trig_identity_l165_165701


namespace sum_positive_integral_values_l165_165385

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165385


namespace find_average_speed_l165_165707

noncomputable def average_speed : ℝ :=
  let distances := [120, 60, 90, 80, 100, 70]
  let speeds := [30, 40, 35, 50, 25, 45]
  let total_distance := distances.sum -- total distance
  let times := list.zip_with (λ d s => d / s) distances speeds
  let total_time := times.sum  -- total time
  total_distance / total_time  -- average speed

theorem find_average_speed : average_speed = 34.16 := by
  sorry

end find_average_speed_l165_165707


namespace four_digit_numbers_count_l165_165486

theorem four_digit_numbers_count : ∃ n, n = 192 ∧ 
  ∀ (d1 d2 d3 d4 : ℕ),
  (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧
  (d2 = 0 ∨ d2 = 1 ∨ d2 = 2 ∨ d2 = 3) ∧
  (d3 = 0 ∨ d3 = 1 ∨ d3 = 2 ∨ d3 = 3) ∧
  (d4 = 0 ∨ d4 = 1 ∨ d4 = 2 ∨ d4 = 3) →
  {x : ℕ | x = d1 * 1000 + d2 * 100 + d3 * 10 + d4}.card = n :=
sorry

end four_digit_numbers_count_l165_165486


namespace solution_set_of_inequality_l165_165104

theorem solution_set_of_inequality :
  ∀ (x : ℝ), abs (2 * x + 1) < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_of_inequality_l165_165104


namespace cube_less_than_three_times_square_l165_165672

theorem cube_less_than_three_times_square (x : ℤ) : x^3 < 3 * x^2 → x = 1 ∨ x = 2 :=
by
  sorry

end cube_less_than_three_times_square_l165_165672


namespace tobias_shoveled_driveways_l165_165110

theorem tobias_shoveled_driveways :
  ∀ (shoe_cost change months allowance_per_month charge_mow charge_shovel mowed_lawns : ℕ),
  shoe_cost = 95 ∧
  change = 15 ∧
  months = 3 ∧
  allowance_per_month = 5 ∧
  charge_mow = 15 ∧
  charge_shovel = 7 ∧
  mowed_lawns = 4 →
  let total_savings := shoe_cost + change in
  let total_allowance := months * allowance_per_month in
  let earnings_from_mowing := mowed_lawns * charge_mow in
  let earnings_from_shoveling := total_savings - total_allowance - earnings_from_mowing in
  earnings_from_shoveling / charge_shovel = 5 :=
by
  intros shoe_cost change months allowance_per_month charge_mow charge_shovel mowed_lawns h,
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h,
  let total_savings := shoe_cost + change,
  let total_allowance := months * allowance_per_month,
  let earnings_from_mowing := mowed_lawns * charge_mow,
  let earnings_from_shoveling := total_savings - total_allowance - earnings_from_mowing,
  exact (earnings_from_shoveling / charge_shovel = 5)

end tobias_shoveled_driveways_l165_165110


namespace num_of_valid_matrices_l165_165986

-- Definition of a 3x3 matrix with entries from 1 to 9
def valid_matrix (m : Matrix (Fin 3) (Fin 3) (Fin 10)) : Prop :=
  (∀ i j, i < j → m i 0 < m j 0) ∧
  (∀ i j, i < j → m i 1 < m j 1) ∧
  (∀ i j, i < j → m i 2 < m j 2) ∧
  (∀ i j, i < j → m 0 i < m 0 j) ∧
  (∀ i j, i < j → m 1 i < m 1 j) ∧
  (∀ i j, i < j → m 2 i < m 2 j) ∧
  (∀ k, ∃ n, m k n = k + 1) -- Ensure all numbers from 1 to 9 are used exactly once

-- The theorem to prove
theorem num_of_valid_matrices : Finset.card 
  { m : Matrix (Fin 3) (Fin 3) (Fin 10) | valid_matrix m } = 42 := sorry

end num_of_valid_matrices_l165_165986


namespace tobias_shoveled_driveways_l165_165111

theorem tobias_shoveled_driveways :
  ∀ (shoe_cost change months allowance_per_month charge_mow charge_shovel mowed_lawns : ℕ),
  shoe_cost = 95 ∧
  change = 15 ∧
  months = 3 ∧
  allowance_per_month = 5 ∧
  charge_mow = 15 ∧
  charge_shovel = 7 ∧
  mowed_lawns = 4 →
  let total_savings := shoe_cost + change in
  let total_allowance := months * allowance_per_month in
  let earnings_from_mowing := mowed_lawns * charge_mow in
  let earnings_from_shoveling := total_savings - total_allowance - earnings_from_mowing in
  earnings_from_shoveling / charge_shovel = 5 :=
by
  intros shoe_cost change months allowance_per_month charge_mow charge_shovel mowed_lawns h,
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h,
  let total_savings := shoe_cost + change,
  let total_allowance := months * allowance_per_month,
  let earnings_from_mowing := mowed_lawns * charge_mow,
  let earnings_from_shoveling := total_savings - total_allowance - earnings_from_mowing,
  exact (earnings_from_shoveling / charge_shovel = 5)

end tobias_shoveled_driveways_l165_165111


namespace probability_not_related_to_layers_and_stratification_l165_165537

-- Definition: Stratified sampling implies each individual has an equal chance of being sampled.
def stratified_sampling (n_layers : ℕ) (n_strata : ℕ) : Prop :=
∀ individual, (∀ l s, probability_of_sampling individual l s n_layers n_strata) = (1 / (n_layers * n_strata))

-- Theorem: The statement "the probability of each individual being sampled is related to the number of layers and the stratification" is false.
theorem probability_not_related_to_layers_and_stratification :
  ¬ (∀ (n_layers n_strata : ℕ), ∃ (individual : ℕ), (probability_of_sampling individual n_layers n_strata ≠ 1 / (n_layers * n_strata))) :=
by {
  sorry
}

end probability_not_related_to_layers_and_stratification_l165_165537


namespace median_runner_is_Tom_l165_165993

theorem median_runner_is_Tom :
  let distances := [(2, "Pete"), (4, "Phil"), (6, "Tom"), (7, "Sanjay"), (8, "Amal")]
  let sorted_distances := distances.map Prod.fst |>.sort
  let median_distance := sorted_distances.nth ((sorted_distances.length + 1) / 2 - 1) = some 6
  in median_distance →
    let runner_at_median := distances.filter (λ d, d.1 = 6) |>.head?.map Prod.snd = some "Tom"
    in runner_at_median :=
sorry

end median_runner_is_Tom_l165_165993


namespace john_buys_packs_l165_165554

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l165_165554


namespace undefined_value_l165_165768

theorem undefined_value (x : ℝ) : (x^2 - 16 * x + 64 = 0) → (x = 8) := by
  sorry

end undefined_value_l165_165768


namespace sum_of_divisors_of_12_l165_165425

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165425


namespace sum_of_divisors_of_12_l165_165371

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165371


namespace sequence_general_term_l165_165458

noncomputable def a (n : ℕ) : ℕ :=
  if h : n > 0 then Classical.choose (nat.rec_on n
    (⟨1, by simp⟩)
    (λ n ih, ⟨(n+1) * ih.val / n, by {
      have h1 : n + 1 > 0 := nat.succ_pos n,
      cases ih with a ha,
      exact nat.div_lt_self h1 (nat.succ_pos n),
    }⟩)) else 0

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : a n = n :=
  sorry

end sequence_general_term_l165_165458


namespace sum_of_fourth_powers_less_than_1000_l165_165184

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165184


namespace maximum_value_l165_165942

theorem maximum_value (x y : ℝ) (a b : ℝ) 
  (hx : a ^ x = 3) (hy : b ^ y = 3) 
  (hab : a + b = 2 * sqrt 3) (ha : a > 1) (hb : b > 1) :
  (1 / x + 1 / y) ≤ 1 :=
by sorry

end maximum_value_l165_165942


namespace sum_of_fourth_powers_below_1000_l165_165165

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165165


namespace min_treasures_buried_l165_165271

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165271


namespace sum_of_divisors_of_12_l165_165432

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165432


namespace tan_sum_simplification_l165_165066

theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 4)) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  sorry

end tan_sum_simplification_l165_165066


namespace sum_of_divisors_of_12_l165_165369

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165369


namespace equal_split_trout_l165_165596

theorem equal_split_trout (total_trout : ℕ) (num_people : ℕ) (h_total : total_trout = 18) (h_people : num_people = 2) :
  total_trout / num_people = 9 :=
by {
  rw [h_total, h_people],
  norm_num,
  sorry -- Here we skip the detailed arithmetic steps
}

end equal_split_trout_l165_165596


namespace karen_boxes_l165_165924

theorem karen_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by {
  rw [h_cases, h_boxes_per_case],
  norm_num,
  sorry
}

end karen_boxes_l165_165924


namespace sin_half_angle_product_in_triangle_l165_165914

-- Define the problem in Lean using the assumptions and the target to prove.
theorem sin_half_angle_product_in_triangle
  (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_C_eq_60 : C = 60)
  (h_tan_sum_eq_one : tan (A / 2) + tan (B / 2) = 1) :
  sin (A / 2) * sin (B / 2) = (√3 - 1) / 2 :=
by
  sorry

end sin_half_angle_product_in_triangle_l165_165914


namespace arithmetic_geometric_sequence_l165_165814

variable {α : Type*} [OrderedSemiring α]

noncomputable def S(n : ℕ) : ℕ → α
| 0 => 0
| (n+1) => S n + a (n+1)

noncomputable def a : ℕ → α
| 0 => 1 -- assuming a_1 = 1, the index should be 0 for initial setup
| n+1 => a n * (real.cbrt 3) -- the term at position n+1

theorem arithmetic_geometric_sequence (h : S 6 = 4 * S 3) : 
  a 7 = 9 := 
sorry

end arithmetic_geometric_sequence_l165_165814


namespace ratio_AM_BM_l165_165899

variables {A B C D M : Type}
variables [HasSymmetry A, HasSymmetry B, HasSymmetry C, HasSymmetry D, HasSymmetry M]
variables (AB AC AD BC BD AM BM : ℝ)
variable k : ℝ
variables (rayAM_sym : AM = reflect AC AD)
variables (rayBM_sym : BM = reflect BD BC)
variables (pointM_intersect : ∃ P, P = intersection AM BM)

theorem ratio_AM_BM (h1 : AB ≠ BC)
                    (h2 : AC / BD = k)
                    (h3 : is_symmetric_with_respect AC AM AD)
                    (h4 : is_symmetric_with_respect BD BM BC)
                    (h5 : is_intersection_point_ M AM BM) :
                    (AM / BM = k^2) :=
begin
  sorry
end

end ratio_AM_BM_l165_165899


namespace sum_divisors_of_12_l165_165403

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165403


namespace problem_M_minus_m_l165_165027

theorem problem_M_minus_m (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  let m := 0 in
  let M := 1 in
  M - m = 1 :=
by
  sorry

end problem_M_minus_m_l165_165027


namespace treasure_15_signs_l165_165301

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165301


namespace num_non_congruent_triangles_with_perimeter_9_l165_165870

theorem num_non_congruent_triangles_with_perimeter_9 :
  ∃ n : ℕ, n = 2 ∧ ∀ (a b c : ℕ), a + b + c = 9 → a + b > c → a + c > b → b + c > a → n = 2 :=
begin
  sorry
end

end num_non_congruent_triangles_with_perimeter_9_l165_165870


namespace sum_of_squares_of_perfect_squares_lt_1000_l165_165137

theorem sum_of_squares_of_perfect_squares_lt_1000 : 
  (∑ n in { n | ∃ k : ℕ, n = k^4 ∧ n < 1000 ∧ n > 0 }, n) = 979 := 
by
  sorry

end sum_of_squares_of_perfect_squares_lt_1000_l165_165137


namespace total_time_watching_videos_l165_165328

theorem total_time_watching_videos 
  (cat_video_length : ℕ)
  (dog_video_length : ℕ)
  (gorilla_video_length : ℕ)
  (h1 : cat_video_length = 4)
  (h2 : dog_video_length = 2 * cat_video_length)
  (h3 : gorilla_video_length = 2 * (cat_video_length + dog_video_length)) :
  cat_video_length + dog_video_length + gorilla_video_length = 36 :=
  by
  sorry

end total_time_watching_videos_l165_165328


namespace conjugate_of_z_l165_165086

-- Definition of the imaginary unit i and the complex number z
def i := Complex.I
def z := 2 / (1 - i)

-- Theorem statement: The conjugate of z is 1 - i
theorem conjugate_of_z : Complex.conj z = 1 - i := sorry

end conjugate_of_z_l165_165086


namespace max_min_sum_of_f_l165_165223

theorem max_min_sum_of_f (f : ℝ → ℝ) (h1 : ∀ a b, a ∈ Icc (-2016) 2016 → b ∈ Icc (-2016) 2016 → f (a + b) = f a + f b - 2012)
  (h2 : ∀ x, x > 0 → x ∈ Icc (-2016) 2016 → f x > 2012) :
  ∃ (M N : ℝ), 
    M = (Icc (-2016) 2016).sup f ∧ 
    N = (Icc (-2016) 2016).inf f ∧ 
    M + N = 4024 :=
sorry

end max_min_sum_of_f_l165_165223


namespace students_above_120_l165_165904

noncomputable def normal_distribution (mean variance : ℝ) : Prop := sorry

theorem students_above_120 (sigma : ℝ) (h_sigma : sigma > 0) : 
  normal_distribution 90 (sigma ^ 2) ∧
  (∀ ξ, P(60 ≤ ξ ∧ ξ ≤ 120) = 0.8) ∧
  (∀ students, students = 780)
  → 
  (∃ n ≥ 0, n = 78) :=
begin
  sorry
end

end students_above_120_l165_165904


namespace minimum_treasure_buried_l165_165295

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165295


namespace sum_of_fourth_powers_le_1000_l165_165149

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165149


namespace find_m_value_l165_165888

theorem find_m_value (x m : ℝ) (H : (x + m) * (x + 8) = x^2 + 8m) : m = -8 :=
by
  have h1: (x + m) * (x + 8) = x^2 + (m + 8) * x + 8 * m,
  {
    sorry
  }
  have h2: m + 8 = 0,
  {
    sorry
  }
  have h3: m = -8,
  {
    sorry
  }
  exact h3

end find_m_value_l165_165888


namespace solution_l165_165830

axiom f : ℝ → ℝ

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → y ≤ 0 → f x > f y

def main_problem : Prop :=
  even_function f ∧ decreasing_function f ∧ f (-2) = 0 → ∀ x, f x < 0 ↔ x > -2 ∧ x < 2

theorem solution : main_problem :=
by
  sorry

end solution_l165_165830


namespace B_squared_ge_AC_l165_165967

variable {a b c A B C : ℝ}

theorem B_squared_ge_AC
  (h1 : b^2 < a * c)
  (h2 : a * C - 2 * b * B + c * A = 0) :
  B^2 ≥ A * C := 
sorry

end B_squared_ge_AC_l165_165967


namespace shifted_parabola_passes_through_point_l165_165642

theorem shifted_parabola_passes_through_point :
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  transformed_eq (-1) = 1 :=
by
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  sorry

end shifted_parabola_passes_through_point_l165_165642


namespace average_age_decrease_l165_165082

theorem average_age_decrease (N T : ℕ) (h₁ : (T : ℝ) / N - 3 = (T - 30 : ℝ) / N) : N = 10 :=
sorry

end average_age_decrease_l165_165082


namespace minimum_treasures_count_l165_165308

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165308


namespace octal_mult_l165_165330

theorem octal_mult : octal_mult 325 07 = 2723 := 
by 
  sorry

end octal_mult_l165_165330


namespace cowboy_shortest_distance_l165_165216

noncomputable def distance : ℝ :=
  let C := (0, 5)
  let B := (-10, 11)
  let C' := (0, -5)
  5 + Real.sqrt ((C'.1 - B.1)^2 + (C'.2 - B.2)^2)

theorem cowboy_shortest_distance :
  distance = 5 + Real.sqrt 356 :=
by
  sorry

end cowboy_shortest_distance_l165_165216


namespace minimum_treasure_buried_l165_165292

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165292


namespace sum_of_divisors_of_12_l165_165434

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165434


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165158

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165158


namespace alcohol_concentration_never_higher_l165_165660

-- Define the initial state of the glasses
structure GlassState where
  water : ℝ
  alcohol : ℝ

-- Assume initial conditions
def initial_glass1 : GlassState := { water := 100, alcohol := 0 }
def initial_glass2 : GlassState := { water := 0, alcohol := 100 }

-- Define the operation of transferring and mixing liquids between glasses
def transfer (x y : ℝ) (g1 g2 : GlassState) : GlassState × GlassState :=
  let new_g2 := { water := g2.water + x, alcohol := g2.alcohol + y }
  let y_mixed := y / (g2.water + x + g2.alcohol)
  let new_g1 := { water := g1.water - x + y_mixed * (g2.water + x), alcohol := g1.alcohol + y_mixed * g2.alcohol }
  (new_g1, new_g2)

-- Prove that at no point can the concentration of alcohol in the first glass exceed that in the second glass
theorem alcohol_concentration_never_higher (g1 g2 : GlassState) :
  (∀ x y : ℝ, let (new_g1, new_g2) := transfer x y g1 g2 in
              (new_g1.alcohol / (new_g1.water + new_g1.alcohol)) ≤ (new_g2.alcohol / (new_g2.water + new_g2.alcohol))) :=
sorry

end alcohol_concentration_never_higher_l165_165660


namespace extremum_a_range_of_f_on_interval_range_of_m_l165_165848

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3 * a * x - 1

theorem extremum_a (a : ℝ) :
  (∃ (x : ℝ), x = -1 ∧ deriv (f a) x = 0) → a = 1 :=
by
  sorry

theorem range_of_f_on_interval (a : ℝ) (ha : a = 1) :
  ∃ (l u : ℝ), l = -3 ∧ u = 17 ∧ ∀ (x : ℝ), x ∈ set.Icc (-2) 3 → f a x ∈ set.Icc l u :=
by
  sorry

theorem range_of_m (a : ℝ) (ha : a = 1) :
  ∃ (l u : ℝ), l = -17 ∧ u = 15 ∧ ∀ (m : ℝ), (∃ (x1 x2 x3 : ℝ), 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f a x1 = 9 * x1 + m ∧ f a x2 = 9 * x2 + m ∧ f a x3 = 9 * x3 + m) → m ∈ set.Ioo l u :=
by
  sorry

end extremum_a_range_of_f_on_interval_range_of_m_l165_165848


namespace distance_midpoint_after_movement_eq_sqrt_2_l165_165599

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem distance_midpoint_after_movement_eq_sqrt_2
  (a b : ℝ × ℝ) :
  let A' := (a.1 + 3, a.2 + 5),
      B' := (b.1 - 5, b.2 - 3),
      M := midpoint a b,
      M' := midpoint A' B' in
  real.sqrt ((M.1 - M'.1)^2 + (M.2 - M'.2)^2) = real.sqrt 2 :=
by 
  sorry

end distance_midpoint_after_movement_eq_sqrt_2_l165_165599


namespace square_side_length_l165_165635

theorem square_side_length (d : ℝ) (h : d = 2) : ∃ s : ℝ, s = sqrt 2 ∧ d = s * sqrt 2 :=
begin
  use sqrt 2,
  split,
  { refl, },
  { rw h,
    norm_num, }
end

end square_side_length_l165_165635


namespace acute_angles_identity_l165_165819

theorem acute_angles_identity
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)  -- α is acute
  (hβ : 0 < β ∧ β < π / 2)  -- β is acute
  (h : α - β = π / 3) :      -- α - β = π / 3
  sin α ^ 2 + cos β ^ 2 - sqrt 3 * sin α * cos β = 1 / 4 := 
sorry

end acute_angles_identity_l165_165819


namespace isosceles_triangle_length_PQ_l165_165965

noncomputable def length_PQ (p : ℝ) : ℝ := 2 * p

theorem isosceles_triangle_length_PQ (p : ℝ) : 
  ∀ (P Q : ℝ × ℝ), 
    P = (p, -p^2) ∧ Q = (-p, -p^2) ∧ (∀ O : ℝ × ℝ, O = (0, 0) ∧ dist P O = dist Q O) 
    → dist P Q = length_PQ p :=
by
  intros P Q h
  cases' h with P_cond Q_cond
  cases' P_cond with H_P H_Q
  rw [H_P, H_Q]
  simp only [dist, subtype.mk_eq_mk, dist_eq_norm, norm_smul, norm_mul, sub_eq_zero, norm_eq_abs, complex.abs_cast, abs_of_nonneg, abs_neg, one_mul]
  sorry

end isosceles_triangle_length_PQ_l165_165965


namespace interest_problem_l165_165233

theorem interest_problem 
  (n : ℕ)
  (A_to_B : ℕ → ℕ)
  (A_to_C : ℕ)
  (rate : ℕ)
  (total_interest : ℕ) : 
  (∃ n, A_to_B n + A_to_C = total_interest) :=
begin
  let A_to_B := λ n, (5000 * rate * n) / 100,
  let A_to_C := (3000 * rate * 4) / 100,
  let rate := 10,
  let total_interest := 2200,
  use 2,
  sorry
end

end interest_problem_l165_165233


namespace circumcenter_PQH_median_ABC_l165_165935

variable (Ω : Type*) [EuclideanSpace Ω]

--Let Θ be the space containing the points A, B, C, O, H, P, Q
structure Geometry where
  (A B C O H P Q : Ω)
  -- O is the circumcenter of an acute scalene triangle ABC
  is_circumcenter_O : ∀ (T : Triangle Ω), (T.isAcute ∧ T.isScalene) →
                    is_circumcenter O (make_circum_circle T) →
                    (T.hasVertices A B C)
  -- Line OA intersects the altitudes of ABC through B and C at P and Q, respectively
  exists_PQ : intersects (line_through O A) (altitude_through B) = P ∧
              intersects (line_through O A) (altitude_through C) = Q
  -- The altitudes of ABC meet at H
  altitudes_meet_at_H : ∀ (T : Triangle Ω), T.hasVertices A B C →
                     𝔮 (Altitudes T).meet_at = H

-- Prove the circumcenter of triangle PQH lies on a median of triangle ABC
theorem circumcenter_PQH_median_ABC (geom : Geometry Ω) :
  ∃ M : Ω,
    is_circumcenter (circumcircle_of_triangle (geom.P) (geom.Q) (geom.H)) ∧
    lies_on_median M (triangle_of_points (geom.A) (geom.B) (geom.C)) :=
sorry

end circumcenter_PQH_median_ABC_l165_165935


namespace solve_for_x_l165_165511

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - y = 7)
  (h2 : x + 3 * y = 7) :
  x = 2.8 :=
by
  sorry

end solve_for_x_l165_165511


namespace distance_between_parallel_lines_is_4_over_15_l165_165984

noncomputable def distance_parallel_lines : ℝ :=
  let l1 : ℝ → ℝ → ℝ := λ x y => 3 * x + 4 * y - 2
  let l2 : ℝ → ℝ → ℝ := λ x y => 3 * x + 6 * y - 5
  let point_on_l2 : ℝ × ℝ := (0, 5/6)
  let distance_to_l1 (P : ℝ × ℝ) : ℝ :=
    (abs (3 * P.1 + 4 * P.2 - 2)) / (real.sqrt ((3:ℝ)^2 + (4:ℝ)^2))
  distance_to_l1 point_on_l2

theorem distance_between_parallel_lines_is_4_over_15 :
  distance_parallel_lines = 4 / 15 :=
sorry

end distance_between_parallel_lines_is_4_over_15_l165_165984


namespace oranges_and_apples_costs_l165_165210

theorem oranges_and_apples_costs :
  ∃ (x y : ℚ), 7 * x + 5 * y = 13 ∧ 3 * x + 4 * y = 8 ∧ 37 * x + 45 * y = 93 :=
by 
  sorry

end oranges_and_apples_costs_l165_165210


namespace geometry_problem_l165_165638

theorem geometry_problem
  (A B C I A_1 B_1 K L : Type)
  [InHab : Inhabited (A × B × C × I × A_1 × B_1 × K × L)] 
  (ABC_triangle : Triangle A B C)
  (I_center : Center I (inscribed_circle ABC_triangle))
  (A1_tangent : Tangent (inscribed_circle ABC_triangle) (segment BC) A_1)
  (B1_tangent : Tangent (inscribed_circle ABC_triangle) (segment AC) B_1)
  (CI_perp_bisector : PerpendicularBisector (segment CI) (segment BK) K)
  (I_line_perp_KB1 : Perpendicular (Line I L) (Line K B_1))
  : Perpendicular (Line AC) (Line A_1 L) :=
sorry

end geometry_problem_l165_165638


namespace solution_set_ineq_min_value_l165_165452
noncomputable theory

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2 * x - 5

-- Prove that the solution set of |f(x)| + |g(x)| ≤ 2 is [5/3, 3].
theorem solution_set_ineq : {x : ℝ | |f(x)| + |g(x)| ≤ 2} = set.Icc (5/3 : ℝ) 3 := sorry

-- Prove that the minimum value of |f(2x)| + |g(x)| is 1.
theorem min_value : ∀x : ℝ, |f (2 * x)| + |g(x)| ≥ 1 ∧
  (∃x : ℝ, |f (2 * x)| + |g(x)| = 1) := sorry

end solution_set_ineq_min_value_l165_165452


namespace extreme_values_of_f_intervals_of_monotonic_decrease_value_of_g_at_pi_over_6_l165_165852

section
variable (x : ℝ)

def f (x : ℝ) : ℝ := sin x + sqrt 3 * cos x

theorem extreme_values_of_f :
  (∀ k ∈ ℤ, f(2 * k * π - π / 2) = -2 ∧ f(2 * k * π + π / 2) = 2) := sorry

theorem intervals_of_monotonic_decrease :
  (∀ k ∈ ℤ, ∀ x, 2 * k * π + π / 6 ≤ x ∧ x ≤ 2 * k * π + 7 * π / 6 → 
    deriv f x < 0) := sorry

def g (x : ℝ) : ℝ := 2 * sin (x / 2 + 2 * π / 3)

theorem value_of_g_at_pi_over_6 : g(π / 6) = sqrt 2 := sorry
end

end extreme_values_of_f_intervals_of_monotonic_decrease_value_of_g_at_pi_over_6_l165_165852


namespace false_conjunction_l165_165690

theorem false_conjunction (p q : Prop) (h : ¬(p ∧ q)) : ¬ (¬p ∧ ¬q) := sorry

end false_conjunction_l165_165690


namespace hexagon_radius_l165_165479

theorem hexagon_radius (a : ℝ) (h : a > 0) (h_p : ∀ (hex : Type) (perimeter : hex → ℝ), perimeter hex = 12 * a) :
  ∀ (radius : Type) (hex : radius → ℝ), hex radius = 2 * a :=
by
  sorry

end hexagon_radius_l165_165479


namespace undefined_expression_l165_165447

theorem undefined_expression (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_expression_l165_165447


namespace treasure_15_signs_l165_165296

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165296


namespace sum_fourth_powers_lt_1000_l165_165179

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165179


namespace proper_subsets_M_intersect_N_l165_165863

noncomputable def M : Set ℤ := {-1, 0, 1}

noncomputable def N : Set ℝ := {y | ∃ x ∈ M, y = 1 + Real.sin (Real.pi * x / 2)}

def M_intersect_N : Set ℝ := {m | m ∈ M ∧ m ∈ N}

def num_proper_subsets (S : Set ℝ) : ℕ := 2^(Set.size S) - 1

theorem proper_subsets_M_intersect_N : num_proper_subsets M_intersect_N = 3 := by
  sorry

end proper_subsets_M_intersect_N_l165_165863


namespace sum_factorials_last_two_digits_l165_165675

theorem sum_factorials_last_two_digits :
  (∑ n in Finset.range 10, n.factorial) % 100 = 13 :=
by
  sorry

end sum_factorials_last_two_digits_l165_165675


namespace max_point_in_unit_disc_l165_165787

-- Define the closed unit disc
def unit_disc (x y : ℝ) : Prop := x^2 + y^2 ≤ 1

-- Define the function f
def f (x y : ℝ) : ℝ := x + y

-- Prove that the point (1/√2, 1/√2) attains the maximum value of the function f in the unit disc
theorem max_point_in_unit_disc : 
  ∃ x y : ℝ, unit_disc x y ∧ f x y = √2 ∧ (∀ x' y' : ℝ, unit_disc x' y' → f x' y' ≤ f x y) :=
begin
  use (1 / Real.sqrt 2, 1 / Real.sqrt 2),
  split,
  { sorry }, -- Prove that the point is in the unit disc
  split,
  { sorry }, -- Prove that f(1/√2, 1/√2) = √2
  { sorry }  -- Prove that this point gives the maximum value in the unit disc
end

end max_point_in_unit_disc_l165_165787


namespace function_has_one_root_l165_165098

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem function_has_one_root : ∃! x : ℝ, f x = 0 :=
by
  -- Indicate that we haven't included the proof
  sorry

end function_has_one_root_l165_165098


namespace sum_of_fourth_powers_le_1000_l165_165146

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165146


namespace treasure_under_minimum_signs_l165_165272

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165272


namespace num_non_congruent_triangles_with_perimeter_9_l165_165872

theorem num_non_congruent_triangles_with_perimeter_9 :
  { (a, b, c) : ℕ × ℕ × ℕ // a ≤ b ∧ b ≤ c ∧ a + b + c = 9 ∧ a + b > c ∧ a + c > b ∧ b + c > a }.to_finset.card = 2 :=
begin
  sorry
end

end num_non_congruent_triangles_with_perimeter_9_l165_165872


namespace sum_of_fourth_powers_below_1000_l165_165163

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165163


namespace trigonometric_identity_l165_165880

theorem trigonometric_identity
  (α : ℝ)
  (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : sin (2 * α) = sin (α - π / 2) * cos (π + α)) :
  sqrt 2 * cos (2 * α - π / 4) = 7 / 5 := by
  sorry

end trigonometric_identity_l165_165880


namespace locus_of_points_equidistant_from_rays_l165_165118

variable {Point Line : Type} [MetricSpace Point] [AffineSpace.LinePoint]

def equidistant_locus 
  (L1 L2 : Line)
  (ray1 ray2 : Set Point)
  (a b : Point)
  (condition1 : a ∈ ray1) 
  (condition2 : b ∈ ray2) 
  (condition3 : rays_eq L1 L2 ray1 ray2) :
  Set Point := sorry

theorem locus_of_points_equidistant_from_rays
  (L1 L2 : Line)
  (ray1 ray2 : Set Point)
  (a b : Point)
  (condition1 : a ∈ ray1) 
  (condition2 : b ∈ ray2) 
  (condition3 : rays_eq L1 L2 ray1 ray2) :
  equidistant_locus L1 L2 ray1 ray2 a b condition1 condition2 condition3 :=
sorry

end locus_of_points_equidistant_from_rays_l165_165118


namespace shifted_parabola_passes_through_point_l165_165643

theorem shifted_parabola_passes_through_point :
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  transformed_eq (-1) = 1 :=
by
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  sorry

end shifted_parabola_passes_through_point_l165_165643


namespace sum_of_perpendiculars_l165_165460

-- Define the given conditions as Lean hypotheses
variables (α b : ℝ)

-- Hypothesis: Given angle α and AC = b
-- We need to prove that the sum of the lengths of the perpendiculars is b * cot(α/2)

theorem sum_of_perpendiculars (h0 : 0 < α ∧ α < π) (h1 : 0 < b)
  : ∀ (sum_perpendiculars : ℝ), sum_perpendiculars = b * Real.cot(α / 2) :=
by
  -- Statement placeholder, actual proof needs to be provided
  sorry

end sum_of_perpendiculars_l165_165460


namespace sum_of_divisors_of_12_l165_165421

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165421


namespace ellipse_eccentricity_l165_165877

variables {F1 F2 P : ℝ × ℝ} {a b : ℝ}

def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {P | (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1}

def orthogonal (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2) = 0

def tan_angle (P F1 F2 : ℝ × ℝ) : Prop :=
  (P.2 - F1.2) / (P.1 - F1.1) = 1 / 2

theorem ellipse_eccentricity (hF1F2 : dist F1 F2 = sqrt 5) (hPF1_orthogonal : orthogonal P F1 F2)
  (h_tan : tan_angle P F1 F2) (hP_on_ellipse : P ∈ ellipse a b) (ha_pos : a > 0) (hb_pos : b > 0)
  (hab : a > b) : abs((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2).sqrt / (2 * a) = sqrt 5 / 3 :=
by
  sorry

end ellipse_eccentricity_l165_165877


namespace brownies_pieces_count_l165_165943

-- Definitions of the conditions
def pan_length : ℕ := 24
def pan_width : ℕ := 15
def pan_area : ℕ := pan_length * pan_width -- pan_area = 360

def piece_length : ℕ := 3
def piece_width : ℕ := 2
def piece_area : ℕ := piece_length * piece_width -- piece_area = 6

-- Definition of the question and proving the expected answer
theorem brownies_pieces_count : (pan_area / piece_area) = 60 := by
  sorry

end brownies_pieces_count_l165_165943


namespace bus_seats_needed_l165_165647

def members_playing_instruments : Prop :=
  let flute := 5
  let trumpet := 3 * flute
  let trombone := trumpet - 8
  let drum := trombone + 11
  let clarinet := 2 * flute
  let french_horn := trombone + 3
  let saxophone := (trumpet + trombone) / 2
  let piano := drum + 2
  let violin := french_horn - clarinet
  let guitar := 3 * flute
  let total_members := flute + trumpet + trombone + drum + clarinet + french_horn + saxophone + piano + violin + guitar
  total_members = 111

theorem bus_seats_needed : members_playing_instruments :=
by
  sorry

end bus_seats_needed_l165_165647


namespace sum_of_fourth_powers_less_than_1000_l165_165188

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165188


namespace find_n_l165_165881

-- Given conditions in the problem
variables (n : ℝ) (x : ℝ)

-- Condition: x = 12
axiom h1 : x = 12

-- Original equation condition: 5 + n / x = 6 - 5 / x
axiom h2 : 5 + n / x = 6 - 5 / x

-- Goal: Prove that n = 7
theorem find_n : n = 7 :=
by
  -- Use conditions to construct the proof
  have h3 : x = 12 := h1,
  have h4 : 5 + n / 12 = 6 - 5 / 12 := by rw [h3] at h2; exact h2,
  sorry

end find_n_l165_165881


namespace minimum_treasures_count_l165_165307

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165307


namespace treasure_under_minimum_signs_l165_165277

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165277


namespace max_k_l165_165857

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x - 1
noncomputable def g (x k : ℝ) : ℝ := Real.log x + k * x

theorem max_k : ∀ x ∈ Ioi (0 : ℝ), ∀ k, (f x) ≥ (g x k) ↔ k ≤ 1 :=
  sorry

end max_k_l165_165857


namespace synchronized_motion_l165_165957

noncomputable def circular_track (perimeter : ℝ) := ∃ (A B : ℝ → ℝ), 
  (∀ t, A t % perimeter = A t) ∧ 
  (∀ t, B t % perimeter = B t) ∧
  (A 0 = 20) ∧
  (B 0 = 0) ∧
  (∀ t, B (t + 1) = B t + 1) ∧
  (∀ t, if (A t - B t + perimeter) % perimeter < 10 
        then A (t + 1) = A t - 3 
        else A (t + 1) = A t + 3) ∧
  (∃ T, T = 120 ∧
        ((λ t, (A (t + T) = A t ∧ B (t + T) = B t)))) 

theorem synchronized_motion : circular_track 40 :=
sorry

end synchronized_motion_l165_165957


namespace difference_between_jo_and_kate_l165_165551

def joSum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def nearestMultipleOf10 (x : ℕ) : ℕ :=
  if x % 10 < 5 then (x / 10) * 10 else (x / 10 + 1) * 10

def kateSum (n : ℕ) : ℕ :=
  let rounded_vals := List.map nearestMultipleOf10 (List.range' 1 n) in
  rounded_vals.sum

theorem difference_between_jo_and_kate (n : ℕ) (h : n = 60) :
  |joSum n - kateSum n| = 1470 :=
by
  rw [h]
  sorry

end difference_between_jo_and_kate_l165_165551


namespace problem_solution_l165_165453

variables {m n : ℝ}

theorem problem_solution (h1 : m^2 - n^2 = m * n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  (n / m) - (m / n) = -1 :=
sorry

end problem_solution_l165_165453


namespace functional_eq_solutions_l165_165778

theorem functional_eq_solutions
  (f : ℚ → ℚ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  ∀ x : ℚ, f x = x ∨ f x = -x := 
sorry

end functional_eq_solutions_l165_165778


namespace find_m_of_parallelepiped_volume_l165_165656

theorem find_m_of_parallelepiped_volume 
  {m : ℝ} 
  (h_pos : m > 0) 
  (h_vol : abs (3 * (m^2 - 9) - 2 * (4 * m - 15) + 2 * (12 - 5 * m)) = 20) : 
  m = (9 + Real.sqrt 249) / 6 :=
sorry

end find_m_of_parallelepiped_volume_l165_165656


namespace complement_intersection_l165_165864

noncomputable def U : set ℝ := set.univ
noncomputable def M : set ℝ := { x | x < 0 ∨ x > 2 }
noncomputable def N : set ℝ := { x | x^2 - 4 * x + 3 < 0 }

theorem complement_intersection :
  (N \ (M ∩ N)) = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
by {
  sorry
}

end complement_intersection_l165_165864


namespace find_AD_l165_165239

-- Definitions
def radius := 100 * Real.sqrt 3
def AB := 100
def BC := 200
def CD := 300
def AC := 200 * Real.sqrt 3

-- Theorem Statement
theorem find_AD (r : ℝ) (AB BC CD AC : ℝ) : 
  r = 100 * Real.sqrt 3 → 
  AB = 100 → 
  BC = 200 → 
  CD = 300 → 
  AC = 200 * Real.sqrt 3 → 
  ∃ (AD : ℝ), AD = 450 :=
by
  intros hr hab hbc hcd hac
  use 450
  sorry

end find_AD_l165_165239


namespace heptagon_area_l165_165756

noncomputable def convex_heptagon_area : ℝ :=
  2 + (real.sqrt 2 / 2) * (real.sqrt 3 - real.sqrt 2)

theorem heptagon_area :
  let unit_circle (x y : ℝ) := x ^ 2 + y ^ 2 = 1 
  ∃ T1 T2 T3 S1 S2 S3 S4 : ℝ × ℝ,
    (unit_circle T1.1 T1.2) ∧ 
    (unit_circle T2.1 T2.2) ∧ 
    (unit_circle T3.1 T3.2) ∧ 
    (unit_circle S1.1 S1.2) ∧ 
    (unit_circle S2.1 S2.2) ∧ 
    (unit_circle S3.1 S3.2) ∧ 
    (unit_circle S4.1 S4.2) ∧ 
    (∃ m b, ∀ x y, 
      (y = m * x + b → y = T1.2) ∨ 
      (y = m * x + b → y = S1.2)) ∧ 
    convex_heptagon_area = 2 + (real.sqrt 2 / 2) * (real.sqrt 3 - real.sqrt 2) := 
sorry

end heptagon_area_l165_165756


namespace sequence_no_perfect_square_iff_prime_l165_165795

def positive_divisors_count (n : ℕ) : ℕ := n.factorization.card + 1

def sequence (k : ℕ) : ℕ → ℕ
| 0     := k
| (n+1) := positive_divisors_count (sequence n)

theorem sequence_no_perfect_square_iff_prime (k : ℕ) (h : k ≥ 2) :
  (∀ n, ¬ ∃ m, sequence k n = m * m) ↔ k.prime :=
by {
  sorry,
}

end sequence_no_perfect_square_iff_prime_l165_165795


namespace sum_of_values_satisfying_equation_l165_165651

-- Define the conditions of the problem
def equation_condition (x : ℝ) : Prop := |x - 1| = 3 * |x + 3|

-- Define the set of all real values satisfying the condition
def satisfying_values : set ℝ := { x | equation_condition x }

-- Define the sum of the values satisfying the condition
def sum_satisfying_values : ℝ := (∑ x in satisfying_values.to_finset, x)

-- Statement to be proven
theorem sum_of_values_satisfying_equation : sum_satisfying_values = -7 :=
by sorry

end sum_of_values_satisfying_equation_l165_165651


namespace domino_trick_l165_165721

theorem domino_trick (x y : ℕ) (h1 : x ≤ 6) (h2 : y ≤ 6)
  (h3 : 10 * x + y + 30 = 62) : x = 3 ∧ y = 2 :=
by
  sorry

end domino_trick_l165_165721


namespace sum_of_divisors_of_12_l165_165372

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165372


namespace ratio_of_areas_is_pi_over_4_l165_165713

noncomputable def ratio_of_areas (a b : ℤ) (hb : b ≠ 0) : ℝ :=
  (π * (a^2 : ℝ) / (b^2 : ℝ)) / (4 * (a^2 : ℝ) / (b^2 : ℝ))

theorem ratio_of_areas_is_pi_over_4 (a b : ℤ) (hb : b ≠ 0) : 
  ratio_of_areas a b hb = π / 4 :=
by
  sorry

end ratio_of_areas_is_pi_over_4_l165_165713


namespace remaining_subtasks_l165_165012

def total_problems : ℝ := 72.0
def finished_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks :
    (total_problems * subtasks_per_problem - finished_problems * subtasks_per_problem) = 200 := 
by
  sorry

end remaining_subtasks_l165_165012


namespace find_c_l165_165338

def g (x c : ℝ) : ℝ := 2 / (3 * x + c)
def g_inv (x : ℝ) : ℝ := (3 - 4 * x) / (6 * x)

theorem find_c (c : ℝ) :
  (∀ x : ℝ, g_inv (g x c) = x) ↔ c = 0 :=
by
  sorry

end find_c_l165_165338


namespace GP_length_l165_165910

theorem GP_length 
  (A B C D E F G P Q : Type)
  [triangle A B C]
  (hAB : segment A B = 8)
  (hAC : segment A C = 15)
  (hBC : segment B C = 17)
  (hAD : median A B C)
  (hBE : median B A C)
  (hCF : median C A B)
  (hG : centroid A B C = G)
  (hP : foot G B C = P) :
  segment G P = 40 / 17 :=
sorry

end GP_length_l165_165910


namespace higher_amount_is_sixty_l165_165052

theorem higher_amount_is_sixty (R : ℕ) (n : ℕ) (H : ℝ) 
  (h1 : 2000 = 40 * n + H * R)
  (h2 : 1800 = 40 * (n + 10) + H * (R - 10)) :
  H = 60 :=
by
  sorry

end higher_amount_is_sixty_l165_165052


namespace light_glow_interval_l165_165639

theorem light_glow_interval :
  let total_time := 3600 + 1369 -- in seconds
  let number_of_glows := 165.63333333333333
  total_time / number_of_glows ≈ 30 := sorry

end light_glow_interval_l165_165639


namespace music_tool_cost_l165_165015

noncomputable def flute_cost : ℝ := 142.46
noncomputable def song_book_cost : ℝ := 7
noncomputable def total_spent : ℝ := 158.35

theorem music_tool_cost :
    total_spent - (flute_cost + song_book_cost) = 8.89 :=
by
  sorry

end music_tool_cost_l165_165015


namespace area_of_triangle_BNF_l165_165968

-- Definitions and conditions
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (AB_length : dist A B = 10)
  (BC_length : dist B C = 12)
  (is_rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ C.1 = B.1 ∧ C.2 = D.2)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.1 - p1.1) * (p3.1 - p2.1) + (p2.2 - p1.2) * (p3.2 - p2.2) = 0

-- Theorem to prove the area of triangle BNF
theorem area_of_triangle_BNF (R : Rectangle) :
  let N := midpoint R.B R.D
  ∧ ∃ F : ℝ × ℝ, F ∈ segment R.A R.B ∧ perpendicular R.B F N
  in area (triangle R.B N F) = 15 :=
sorry

end area_of_triangle_BNF_l165_165968


namespace joe_investment_rate_l165_165203

noncomputable def rate_of_interest (P r : ℝ) (t1 t2 : ℕ) : Prop :=
  let I := P * r * t2 in
  let I_3 := P * r * t1 in
  ((260 : ℝ) = P + I_3) ∧ ((360 : ℝ) = 260 + (I - I_3)) ∧ (100 = 260 * r * (t2 - t1))

theorem joe_investment_rate :
  ∃ r : ℝ, rate_of_interest 260 r 3 8 ∧ r = 1 / 13 :=
by
  sorry

end joe_investment_rate_l165_165203


namespace minimum_perimeter_of_section_AEFG_l165_165809

/-- Given a right tetrahedron S-ABCD with side length 4 and an angle ∠ASB of 30 degrees,
a plane passes through point A and intersects the side edges SB, SC, and SD at points E, F, and G
respectively. Prove the minimum perimeter of the section AEFG is 4√3. -/
theorem minimum_perimeter_of_section_AEFG :
  ∀ (S A B C D E F G : Point)
    (side_length : ℝ) (angle_ASB : ℝ)
    (h1: tetrahedron S A B C D)
    (h2: ∀ (P : Point), distance S P = side_length)
    (h3: angle S A B = angle_ASB)
    (h4: plane_through A intersects S B at E)
    (h5: plane_through A intersects S C at F)
    (h6: plane_through A intersects S D at G),
  side_length = 4 → angle_ASB = π / 6 → 
  minimum_perimeter A E F G = 4 * real.sqrt 3 :=
by
  intros
  sorry

end minimum_perimeter_of_section_AEFG_l165_165809


namespace minimum_treasures_count_l165_165304

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165304


namespace problem_part1_problem_part2_l165_165488

noncomputable def f (x : ℝ) : ℝ := (4 * x) / (x^2 + 1)

theorem problem_part1 : (∀ x, f'(x) = -4 * x^2 + 4 / (x^2 + 1)^2) ∧ (f 1 = 2) := by
  sorry

theorem problem_part2 (x₀ : ℝ) : 
  let t := 1 / (x₀^2 + 1) in 
  t ∈ (0, 1] →
  let k := -4 * t + 8 * t^2 in 
  -1/2 ≤ k ∧ k ≤ 4 := by
  sorry

end problem_part1_problem_part2_l165_165488


namespace increasing_interval_of_sine_l165_165640

noncomputable def minimum_positive_period_of_sine (ω : ℝ) (hω : ω > 0) : ℝ := π

theorem increasing_interval_of_sine (k : ℤ) :
  let f := λ x : ℝ, sin (2 * x - π / 3)
  ∃ ω > 0, (∀ x, f (x + π / ω) = f x) →
  (ω = 2) →
  (∀ x, f (1 / 12 * π - π / k) ≤ sin (2 * x - π / 3) ∧ sin (2 * x - π / 3) ≤ f (5 / 12 * π + π / k) ) := sorry

end increasing_interval_of_sine_l165_165640


namespace percentage_lower_grades_have_cars_l165_165658

-- Definitions for the conditions
def n_seniors : ℕ := 300
def p_car : ℚ := 0.50
def n_lower : ℕ := 900
def p_total : ℚ := 0.20

-- Definition for the number of students who have cars in the lower grades
def n_cars_lower : ℚ := 
  let total_students := n_seniors + n_lower
  let total_cars := p_total * total_students
  total_cars - (p_car * n_seniors)

-- Prove the percentage of freshmen, sophomores, and juniors who have cars
theorem percentage_lower_grades_have_cars : 
  (n_cars_lower / n_lower) * 100 = 10 := 
by sorry

end percentage_lower_grades_have_cars_l165_165658


namespace rational_number_is_sqrt_4_l165_165255

theorem rational_number_is_sqrt_4 : (∃ q : ℚ, q = 2) :=
by
  use 2
  sorry

end rational_number_is_sqrt_4_l165_165255


namespace min_value_fractions_l165_165025

open Real

theorem min_value_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) :
  3 ≤ (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a)) :=
sorry

end min_value_fractions_l165_165025


namespace sum_divisors_12_eq_28_l165_165367

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165367


namespace area_of_trapezoid_l165_165262

-- Define the geometry and conditions
variable (O : Point) (r : ℝ) [hr : r = 5]
variable (trapezoid : Trapezoid) (circumscribes : Circumscribes trapezoid O)
variable (distance_tangency_points : ℝ) [hDistance : distance_tangency_points = 8]

-- The statement we want to prove
theorem area_of_trapezoid {O : Point} {r : ℝ} (h_r : r = 5) (trapezoid : Trapezoid)
  (circumscribes : Circumscribes trapezoid O) {distance_tangency_points : ℝ}
  (hDistance : distance_tangency_points = 8) :
  area trapezoid = 125 := 
sorry

end area_of_trapezoid_l165_165262


namespace yoongi_initial_money_l165_165692

-- Definitions of conditions
variables (initial money: ℕ) (candy: ℕ) (pocket money: ℕ) (leftover: ℕ) (spent_half: ℕ)

-- Assigning values to the conditions
def candy := 250
def pocket_money := 500
def leftover := 420
def spent_half := leftover * 2

-- The main theorem to prove
theorem yoongi_initial_money (initial_money: ℕ) (candy pocket_money leftover: ℕ): 
  candy = 250 → 
  pocket_money = 500 → 
  leftover = 420 → 
  spent_half = leftover * 2 →
  initial_money = spent_half - pocket_money + candy → 
  initial_money = 590 :=
begin
  intros,
  sorry -- Proof goes here
end

end yoongi_initial_money_l165_165692


namespace concyclic_points_l165_165826

theorem concyclic_points 
  (A B C M P Q L K P₁ Q₁: Type)
  [geometry.point_type A]
  [geometry.point_type B]
  [geometry.point_type C]
  [geometry.point_type M]
  [geometry.point_type P]
  [geometry.point_type Q]
  [geometry.point_type L]
  [geometry.point_type K]
  (h1 : M ∈ seg B C)
  (h2 : P = circumcenter ⟨A, B, M⟩)
  (h3 : Q = circumcenter ⟨A, C, M⟩)
  (h4 : L ∈ (line_through B P) ∩ (line_through C Q))
  (h5 : reflection L P Q = K) :
  concyclic M P Q K :=
sorry

end concyclic_points_l165_165826


namespace johns_total_packs_l165_165558

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l165_165558


namespace math_problem_equiv_l165_165333

theorem math_problem_equiv :
  let a := (- (1 : ℚ) / 2) ^ -2
  let b := (Real.pi - 3.14) ^ 0
  let c := 4 * Real.cos (45 * Real.pi / 180)
  let d := Real.abs (1 - Real.sqrt 2)
  a + b + c - d = Real.sqrt 2 + 6 :=
by
  sorry

end math_problem_equiv_l165_165333


namespace number_of_lines_l165_165895

theorem number_of_lines (n : ℕ) (h : n ≥ 3) (no_three_collinear : ∀ (P : Finset (Fin n)), P.card = 3 → ¬ ∃ l : Set (Fin n), ∀ p ∈ P, p ∈ l ∧ l.card = 2) : 
  (Finset.univ : Finset (Fin n)).card.choose 2 = n * (n - 1) / 2 :=
by 
  sorry

end number_of_lines_l165_165895


namespace sum_of_perfect_square_squares_less_than_1000_l165_165168

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165168


namespace john_buys_360_packs_l165_165557

def John_buys_packs (classes students_per_class packs_per_student total_packs : ℕ) : Prop :=
  classes = 6 →
  students_per_class = 30 →
  packs_per_student = 2 →
  total_packs = (classes * students_per_class) * packs_per_student
  → total_packs = 360

theorem john_buys_360_packs : John_buys_packs 6 30 2 360 :=
by { intros, sorry }

end john_buys_360_packs_l165_165557


namespace find_units_digit_l165_165790

def units_digit (n : ℕ) : ℕ := n % 10

theorem find_units_digit :
  units_digit (3 * 19 * 1933 - 3^4) = 0 :=
by
  sorry

end find_units_digit_l165_165790


namespace area_of_quadrilateral_ADEC_l165_165820

variables {A B C D E : Type} [ordered_euclidean_space ℝ A]
variables {M N P Q R : EPoint A B C}
variables {AB MEAS DE : ℝ}

noncomputable def quadrilateral_area_problem 
  (angle_ABC : ∠ B A C = 90)
  (AD_eq_DB : dist A D = dist D B)
  (DE_perpendicular_AB : ∠ D E A = 90 ∧ ∠ D E B = 90)
  (AB_length : dist A B = 20)
  (AC_length : dist A C = 12) : ℝ :=
  58.5

-- Prove the area of quadrilateral AD EC is 58.5
theorem area_of_quadrilateral_ADEC 
  (angle_ABC : ∠ B C A = 90)
  (AD_eq_DB : dist A D = dist D B)
  (DE_perpendicular_AB : ∠ D E A = 90 ∧ ∠ D E B = 90)
  (AB_length : dist A B = 20)
  (AC_length : dist A C = 12) :
  quadrilateral_area_problem angle_ABC AD_eq_DB DE_perpendicular_AB AB_length AC_length = 58.5 :=
sorry

end area_of_quadrilateral_ADEC_l165_165820


namespace triangle_right_triangle_l165_165525

variable {A B C : Real}  -- Define the angles A, B, and C

theorem triangle_right_triangle (sin_A sin_B sin_C : Real)
  (h : sin_A^2 + sin_B^2 = sin_C^2) 
  (triangle_cond : A + B + C = 180) : 
  (A = 90) ∨ (B = 90) ∨ (C = 90) := 
  sorry

end triangle_right_triangle_l165_165525


namespace integral_relationship_l165_165571

theorem integral_relationship :
  let m := ∫ x in 0..1, Real.exp x
  let n := ∫ x in 1..Real.exp 1, 1/x
  m > n :=
by {
  -- Proof omitted
  sorry
}

end integral_relationship_l165_165571


namespace sum_of_divisors_of_12_l165_165413

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165413


namespace number_of_factors_of_10_in_2010_factorial_l165_165882

noncomputable def count_factors_of_5 (n : ℕ) : ℕ :=
  let rec aux (n : ℕ) (k : ℕ) : ℕ :=
    if n = 0 then k
    else aux (n / 5) (k + n / 5)
  aux n 0

theorem number_of_factors_of_10_in_2010_factorial :
  ∃ M k : ℕ, (2010.factorial = M * 10^k) ∧ (¬ (10 ∣ M)) ∧ (k = 501) := 
by 
  use 2010.factorial / 10^501
  use 501
  split
  { -- Proof that 2010.factorial = (2010.factorial / 10^501) * 10^501
    sorry },
  split
  { -- Proof that 10 does not divide (2010.factorial / 10^501)
    sorry },
  { -- Proof that k = 501
    suffices count_factors_of_5 2010 = 501, from this,
    -- You might need to prove here that count_factors_of_5 returns 501 given n=2010
    sorry }

end number_of_factors_of_10_in_2010_factorial_l165_165882


namespace compute_g_neg1001_l165_165625

noncomputable def g : ℝ → ℝ := sorry

-- Stating the conditions
axiom functional_eqn : ∀ x y : ℝ, g(x * y) + x^2 = x * g(y) + g(x)
axiom g_neg1_eqn : g (-1) = 7

-- Stating the goal
theorem compute_g_neg1001 : g (-1001) = 6006013 :=
sorry

end compute_g_neg1001_l165_165625


namespace chord_length_problem_l165_165884

-- The points of intersection between the line and the conic curve
variables {x1 y1 x2 y2 : ℝ}

-- Define the line equation and the condition on the distance between the points
def line_eq : Prop := 2 * x1 - y1 - 1 = 0 ∧ 2 * x2 - y2 - 1 = 0

def distance_eq : Prop := real.dist (x1, y1) (x2, y2) = real.sqrt 10

-- The theorem we need to prove
theorem chord_length_problem (h₁ : line_eq) (h₂ : distance_eq) : |x1 - x2| = real.sqrt 2 :=
sorry

end chord_length_problem_l165_165884


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165154

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165154


namespace problem_am_hm_l165_165590

open Real

theorem problem_am_hm (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 2) :
  ∃ S : Set ℝ, (∀ s ∈ S, (2 ≤ s)) ∧ (∀ z, (2 ≤ z) → (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ z = 1/x + 1/y))
  ∧ (S = {z | 2 ≤ z}) := sorry

end problem_am_hm_l165_165590


namespace minimum_treasure_buried_l165_165289

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165289


namespace A_in_terms_of_B_l165_165576

-- Definitions based on conditions
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B x : ℝ) : ℝ := B * x^2

-- Theorem statement
theorem A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) : A = 3 * B / 16 :=
by
  -- Proof omitted
  sorry

end A_in_terms_of_B_l165_165576


namespace complex_number_solution_l165_165518

theorem complex_number_solution (z : ℂ) (h : (conj z) / (1 - complex.i) = complex.i) : z = 1 - complex.i :=
sorry

end complex_number_solution_l165_165518


namespace find_p_q_l165_165037

theorem find_p_q 
  (p q: ℚ)
  (a : ℚ × ℚ × ℚ × ℚ := (4, p, -2, 1))
  (b : ℚ × ℚ × ℚ × ℚ := (3, 2, q, -1))
  (orthogonal : (4 * 3 + p * 2 + (-2) * q + 1 * (-1) = 0))
  (equal_magnitudes : (4^2 + p^2 + (-2)^2 + 1^2 = 3^2 + 2^2 + q^2 + (-1)^2))
  : p = -93/44 ∧ q = 149/44 := 
  by 
    sorry

end find_p_q_l165_165037


namespace count_ordered_pairs_imaginary_diff_real_l165_165753

theorem count_ordered_pairs_imaginary_diff_real :
  let n := 150
  let count_multiples_of_4 := 37
  let count_mod_2 := 37
  let count_mod_4 := 38
  (count_multiples_of_4 * (count_multiples_of_4 - 1)) / 2 +
  (count_mod_2 * count_mod_4) = 2072 :=
by
  let n := 150
  let count_multiples_of_4 := 37
  let count_mod_2 := 37
  let count_mod_4 := 38
  calc
    (count_multiples_of_4 * (count_multiples_of_4 - 1)) / 2 + 
    (count_mod_2 * count_mod_4)
    = 666 + 1406 : by norm_num
    ... = 2072 : by norm_num

end count_ordered_pairs_imaginary_diff_real_l165_165753


namespace sum_of_squares_of_distances_to_line_l165_165441

noncomputable def sum_of_squares_of_distances (n : ℕ) (R : ℝ) (e : Fin n → ℝ × ℝ) (x : ℝ × ℝ) : ℝ :=
  ∑ i, (e i).fst * x.fst + (e i).snd * x.snd

theorem sum_of_squares_of_distances_to_line (n : ℕ) (R : ℝ)
  (h_reg_poly : ∀ i, ∃ θ : ℝ, θ = 2 * π * i / n ∧ e i = (R * Real.cos θ, R * Real.sin θ))
  (x : ℝ × ℝ)
  (h_x_unit : x.fst ^ 2 + x.snd ^ 2 = 1) :
  sum_of_squares_of_distances n R e x = 1 / 2 * n * R ^ 2 :=
by
  sorry

end sum_of_squares_of_distances_to_line_l165_165441


namespace min_treasures_buried_l165_165320

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165320


namespace sum_divisors_12_eq_28_l165_165360

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165360


namespace MrC_net_outcome_l165_165949

noncomputable def initial_value := 20000
noncomputable def profit_percent := 0.15
noncomputable def loss_percent := 0.15
noncomputable def transaction_fee_percent := 0.05

theorem MrC_net_outcome :
  let selling_price := initial_value * (1 + profit_percent),
      price_before_fee := selling_price * (1 - loss_percent),
      transaction_fee := price_before_fee * transaction_fee_percent,
      final_buying_price := price_before_fee + transaction_fee,
      net_outcome := selling_price - final_buying_price in
  net_outcome = 2472.5 :=
by sorry

end MrC_net_outcome_l165_165949


namespace treasure_15_signs_l165_165303

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165303


namespace range_of_m_l165_165889

theorem range_of_m (m : ℝ) : (∃ x : ℝ, x^2 + 2 * x - m - 1 = 0) → m ≥ -2 := 
by
  sorry

end range_of_m_l165_165889


namespace max_non_similar_matrices_2020_l165_165336

open Matrix

noncomputable def max_non_similar_matrices (n : ℕ) : ℕ :=
  if n = 2020 then 673 else 0

variables {A : Matrix (Fin 2020) (Fin 2020) ℂ}

def is_adjugate (A A_adj : Matrix (Fin 2020) (Fin 2020) ℂ) : Prop := 
  A + A_adj = 1 ∧ A ⬝ A_adj = 1

theorem max_non_similar_matrices_2020 :
  ∀ (A : Matrix (Fin 2020) (Fin 2020) ℂ),
  (∃ A_adj, is_adjugate A A_adj) →
  max_non_similar_matrices 2020 = 673 :=
by
  sorry

end max_non_similar_matrices_2020_l165_165336


namespace min_treasures_buried_l165_165267

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165267


namespace arithmetic_mean_of_4_and_16_l165_165472

-- Define the arithmetic mean condition
def is_arithmetic_mean (a b x : ℝ) : Prop :=
  x = (a + b) / 2

-- Theorem to prove that x = 10 if it is the mean of 4 and 16
theorem arithmetic_mean_of_4_and_16 (x : ℝ) (h : is_arithmetic_mean 4 16 x) : x = 10 :=
by
  sorry

end arithmetic_mean_of_4_and_16_l165_165472


namespace max_loss_on_subinterval_min_avg_cost_volume_l165_165664

noncomputable def processing_cost (x : ℝ) : ℝ :=
if x ∈ set.Ico 120 144 then (1/3) * x^3 - 80 * x^2 + 5040 * x
else if x ∈ set.Ico 144 500 then (1/2) * x^2 - 200 * x + 80000
else 0  -- Default case to handle values outside the given range.

noncomputable def product_value_per_ton : ℝ := 200

noncomputable def profit (x : ℝ) : ℝ :=
product_value_per_ton * x - processing_cost x

theorem max_loss_on_subinterval : 
∀ x ∈ set.Icc 200 300, profit x ≤ -5000 :=
sorry

theorem min_avg_cost_volume : 
∀ x, (x ∈ set.Icc 120 500 → (processing_cost x / x) ≥ (processing_cost 400 / 400)) ∧ 
      ((processing_cost 400 / 400) ≤ processing_cost x / x) :=
sorry

end max_loss_on_subinterval_min_avg_cost_volume_l165_165664


namespace maximum_planes_l165_165357

-- Defining the set of 6 points and the properties of the planes
noncomputable def points : Set (Fin 6) := {0, 1, 2, 3, 4, 5}

-- Condition: Each plane contains at least 4 points
def plane_contains_at_least_four (plane : Set (Fin 6)) : Prop :=
  4 ≤ plane.card

-- Condition: No four points are collinear
def no_four_collinear (points : Set (Fin 6)) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin 6), {p1, p2, p3, p4}.card = 4 → ¬ collinear {p1, p2, p3, p4}

-- Auxiliary definition for the collinearity of points (this typically depends on the definition in geom package)
axiom collinear : Set (Fin 6) → Prop

-- Defining the theorem with given conditions to prove the maximum number of planes
theorem maximum_planes (h1 : ∀ plane : Set (Fin 6), plane_contains_at_least_four plane)
                       (h2 : no_four_collinear points) : 
                       ∃ n_plane_given_points, n_plane_given_points = 6 := 
begin
  sorry
end

end maximum_planes_l165_165357


namespace roger_total_miles_l165_165969

def morning_miles : ℕ := 2
def evening_multiplicative_factor : ℕ := 5
def evening_miles := evening_multiplicative_factor * morning_miles
def third_session_subtract : ℕ := 1
def third_session_miles := (2 * morning_miles) - third_session_subtract
def total_miles := morning_miles + evening_miles + third_session_miles

theorem roger_total_miles : total_miles = 15 := by
  sorry

end roger_total_miles_l165_165969


namespace find_f_at_1_l165_165493

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem find_f_at_1 : f 1 = 2 := by
  sorry

end find_f_at_1_l165_165493


namespace sum_divisors_of_12_l165_165397

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165397


namespace sally_initial_poems_l165_165064

theorem sally_initial_poems (recited: ℕ) (forgotten: ℕ) (h1 : recited = 3) (h2 : forgotten = 5) : 
  recited + forgotten = 8 := 
by
  sorry

end sally_initial_poems_l165_165064


namespace equilateral_triangle_circumcircle_area_l165_165258

-- Define the equilateral triangle DEF with side length 6√2
def triangle_DEF := {D E F : Type} [equilateral_triangle DEF]
def side_length := 6 * real.sqrt 2

-- Define the circle with radius 4√3 and tangent to sides DE and DF at points D and F
def circle_tangent := {C : Type} (radius : 4 * real.sqrt 3) (tangent_to_DE_at_D : Prop) (tangent_to_DF_at_F : Prop)

-- Define the circumcircle of triangle DEF
noncomputable def circumradius (s : ℝ) : ℝ := s / real.sqrt 3

-- Define the area of the circumcircle
noncomputable def circumcircle_area (R : ℝ) : ℝ := real.pi * (R ^ 2)

-- The final theorem to prove
theorem equilateral_triangle_circumcircle_area :
  circumcircle_area (circumradius side_length) = 24 * real.pi :=
by
  -- Insert proof here
  sorry

end equilateral_triangle_circumcircle_area_l165_165258


namespace last_two_digits_x_pow_y_add_y_pow_x_l165_165586

noncomputable def proof_problem (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : ℕ :=
  (x^y + y^x) % 100

theorem last_two_digits_x_pow_y_add_y_pow_x {x y : ℕ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : 
  proof_problem x y h1 h2 h3 h4 = 74 :=
sorry

end last_two_digits_x_pow_y_add_y_pow_x_l165_165586


namespace sum_fourth_powers_lt_1000_l165_165178

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165178


namespace treasure_15_signs_l165_165300

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165300


namespace locus_of_incenter_l165_165252

-- Define a basic structure for a circle with diameters.
structure Circle := 
  (center : Point)
  (radius : ℝ)
  (diam1 diam2 : Line) -- Perpendicular diameters

-- Define a basic structure for points and lines.
structure Point := 
  (x : ℝ) 
  (y : ℝ)

structure Line :=
  (slope : ℝ) 
  (y_intercept : ℝ)

noncomputable def tangent_at_Point (c : Circle) (p : Point) : Line := sorry

noncomputable def intersection (l1 l2 : Line) : Point := sorry

noncomputable def incenter (a b c : Point) : Point := sorry

def is_on_chord (p : Point) (l : Line) : Prop := sorry

-- Given conditions
def circle := Circle.mk (Point.mk 0 0) 1 (Line.mk 0 0) (Line.mk (1/0) 0) -- Unit circle with diameters along x and y axes
def point_P := Point.mk 1 0 -- Arbitrary point on the unit circle at (1, 0), can vary as needed
def tangent_T := tangent_at_Point circle point_P
def point_T := intersection (circle.diam1) tangent_T

-- Theorem to be proved
theorem locus_of_incenter (circle : Circle) (point_P point_T : Point) : 
  ∃ C, incenter circle.center point_P point_T = C ∧ is_on_chord C (Line.mk 0 0) :=
sorry

end locus_of_incenter_l165_165252


namespace common_difference_arith_seq_l165_165812

theorem common_difference_arith_seq (s : Fin 12 → ℕ) 
  (h1 : (∑ i in Finset.filter (λ i, i % 2 = 1) (Finset.univ : Finset (Fin 12)), s i) = 10)
  (h2 : (∑ i in Finset.filter (λ i, i % 2 = 0) (Finset.univ : Finset (Fin 12)), s i) = 22) : 
  (∃ d, d = 2) :=
by
  sorry

end common_difference_arith_seq_l165_165812


namespace parallel_lines_conditions_l165_165901

variables (α β γ : Plane) (m n : Line)
variables (hαβ : α ∩ β = m) (hnγ : n ⊆ γ)

theorem parallel_lines_conditions :
  (α ‖ γ ∧ n ⊆ β) ∨ (n ‖ β ∧ m ⊆ γ) → m ‖ n :=
by
  sorry

end parallel_lines_conditions_l165_165901


namespace betta_fish_guppies_eaten_l165_165014

theorem betta_fish_guppies_eaten (moray_eel_guppies : ℕ) (betta_fish_count : ℕ) (total_guppies : ℕ)
  (h_moray_eel_guppies : moray_eel_guppies = 20)
  (h_betta_fish_count : betta_fish_count = 5)
  (h_total_guppies : total_guppies = 55) :
  let guppies_per_betta_fish := (total_guppies - moray_eel_guppies) / betta_fish_count
  in guppies_per_betta_fish = 7 := 
by 
  sorry

end betta_fish_guppies_eaten_l165_165014


namespace total_jogging_distance_l165_165956

theorem total_jogging_distance :
  let monday_jog := 2
  let tuesday_jog := 5
  let wednesday_jog := 9
  monday_jog + tuesday_jog + wednesday_jog = 16 :=
by {
  let monday_jog := 2
  let tuesday_jog := 5
  let wednesday_jog := 9
  have : monday_jog + tuesday_jog + wednesday_jog = 16,
  { sorry },
  exact this
}

end total_jogging_distance_l165_165956


namespace number_of_permutations_is_24_l165_165045

def competitors : List String := ["Luna", "Ginny", "Hermione", "Cho"]

theorem number_of_permutations_is_24 
  (h : competitors.length = 4) :
  List.permutations competitors |>.length = Nat.factorial 4 := 
by
  sorry

end number_of_permutations_is_24_l165_165045


namespace unique_solution_a_x2_sin2_x_l165_165781

theorem unique_solution_a_x2_sin2_x {a : ℝ} (h : ∀ x : ℝ, a * x^2 + sin x ^ 2 = a^2 - a → x = 0):
  a = 1 :=
begin
  sorry
end

end unique_solution_a_x2_sin2_x_l165_165781


namespace sum_divisors_of_12_l165_165400

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165400


namespace expected_value_eta_l165_165837

noncomputable def xi : ℕ → ℝ → ℕ := sorry -- Definition of binomial random variable
noncomputable def eta (xi : ℕ → ℝ → ℕ) (n : ℕ) (p : ℝ) := 2 * xi n p - 1

noncomputable def E_xi : ℕ → ℝ → ℝ
| 5, (1 / 3) => (5 : ℝ) * (1 / 3)

noncomputable def E_eta : ℕ → ℝ → ℝ
| 5, (1 / 3) => 2 * E_xi 5 (1 / 3) - 1

theorem expected_value_eta :
  E_eta 5 (1 / 3) = (7 / 3 : ℝ) :=
by
  sorry

end expected_value_eta_l165_165837


namespace sum_of_fourth_powers_le_1000_l165_165150

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165150


namespace proof_A_proof_B_proof_C_proof_D_l165_165868

variables {a b : ℝ → ℝ} -- This line is under the assumption vector a and b are functions from real to real.

-- Assuming the Euclidean norm and dot product in ℝ^n
def norm (v : ℝ → ℝ) : ℝ := sqrt (∑ i, v i * v i)
def dot_product (u v : ℝ → ℝ) : ℝ := ∑ i, u i * v i

-- Defining conditions
variables (h1 : norm a = 1) (h2 : norm b = 2) (h3 : dot_product a b = norm a * norm b * (1 / 2))

theorem proof_A : dot_product a b = 1 :=
by
  rw [←h3, h1, h2]
  norm_num
  sorry

theorem proof_B : norm (λ i, 2 * a i + b i) = 2 * sqrt 3 :=
by
  rw [norm, sqrt_eq_rpow, pow_add, pow_mul, h1, h2, h3]
  sorry

theorem proof_C : norm (λ i, 2 * a i - b i) = 2 :=
by
  rw [norm, sqrt_eq_rpow, pow_add, pow_sub, pow_mul, h1, h2, h3]
  sorry

theorem proof_D : dot_product a (λ i, a i - b i) = 0 :=
by
  rw [dot_product, h1]
  norm_num
  rw [h3]
  norm_num
  sorry

end proof_A_proof_B_proof_C_proof_D_l165_165868


namespace area_of_triangle_AEF_l165_165911

theorem area_of_triangle_AEF 
  (A B C E F : Type) (AE AC AB : ℝ) 
  (hEA : AE = 1 / 3 * AC) 
  (hF_midpoint : F = (A + B) / 2) 
  (h_area_ABC : (1 / 2) * (AB * AC) = 36) :
  let area_ABC := 36 in
  let ratio_AE_AC := 1 / 3 in
  let area_ABF := 18 in
  let area_AEF := 12 in
  (area_AEF = 1 / 3 * area_ABF) :=
  sorry

end area_of_triangle_AEF_l165_165911


namespace sum_of_divisors_of_12_l165_165414

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165414


namespace tan_E_tan_F_eq_3point4_l165_165641

noncomputable def DEF_triangle (D E F : Type) (d e f : ℝ) :=
∃ H M : D, 
seg_len HM = 10 ∧
seg_len HD = 24 ∧
altitude DM H M EF ∧
orthocenter H DEF d e f

theorem tan_E_tan_F_eq_3point4
  {D E F : Type} {d e f : ℝ} :
  DEF_triangle D E F d e f →
  (∃ E F : ℝ, tan E * tan F = 3.4) :=
begin
  intro DEF_triangle,
  sorry,
end

end tan_E_tan_F_eq_3point4_l165_165641


namespace number_of_groups_of_oranges_l165_165107

-- Defining the conditions
def total_oranges : ℕ := 356
def oranges_per_group : ℕ := 2

-- The proof statement
theorem number_of_groups_of_oranges : total_oranges / oranges_per_group = 178 := 
by 
  sorry

end number_of_groups_of_oranges_l165_165107


namespace min_treasure_count_l165_165317

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165317


namespace remainder_when_dividing_g_x_10_by_g_x_is_6_l165_165585

def g (x : ℤ) : ℤ := x^5 - x^4 + x^3 - x^2 + x - 1

theorem remainder_when_dividing_g_x_10_by_g_x_is_6 (x : ℤ) :
  let g10 := g (x^10)
  in ∃ r : ℤ, r = 6 ∧ ∃ q : ℤ, g10 = g x * q + r :=
by
  sorry

end remainder_when_dividing_g_x_10_by_g_x_is_6_l165_165585


namespace ratio_preference_l165_165955

-- Definitions based on conditions
def total_respondents : ℕ := 180
def preferred_brand_x : ℕ := 150
def preferred_brand_y : ℕ := total_respondents - preferred_brand_x

-- Theorem statement to prove the ratio of preferences
theorem ratio_preference : preferred_brand_x / preferred_brand_y = 5 := by
  sorry

end ratio_preference_l165_165955


namespace quotient_of_poly_div_l165_165678

theorem quotient_of_poly_div :
  (10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6) / (5 * X^2 + 7) =
  2 * X^2 - X - (11 / 5) :=
sorry

end quotient_of_poly_div_l165_165678


namespace sum_of_perfect_square_squares_less_than_1000_l165_165171

theorem sum_of_perfect_square_squares_less_than_1000 : 
  ∑ i in finset.filter (λ n, ∃ k, n = k^4) (finset.range 1000), i = 979 := 
by
  sorry

end sum_of_perfect_square_squares_less_than_1000_l165_165171


namespace seeds_per_can_l165_165600

theorem seeds_per_can (total_seeds : ℕ) (num_cans : ℕ) (h1 : total_seeds = 54) (h2 : num_cans = 9) : total_seeds / num_cans = 6 :=
by {
  sorry
}

end seeds_per_can_l165_165600


namespace sum_positive_integral_values_l165_165382

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165382


namespace monotonically_decreasing_interval_and_center_of_symmetry_max_area_of_triangle_l165_165503

noncomputable def m (x : ℝ) : ℝ × ℝ := (√3 * sin (x / 4), -1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4)^2)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem monotonically_decreasing_interval_and_center_of_symmetry (k : ℤ) :
  ∃ (I : set ℝ) (c : ℝ × ℝ), 
    I = set.Icc (4 * k * π + 4 * π / 3) (4 * k * π + 10 * π / 3) ∧ 
    c = (2 * k * π + π / 3, -1 / 2) := by
  sorry

theorem max_area_of_triangle (A : ℝ) (a b c : ℝ) (hA : f A = -1 / 2) (ha : a = 2) :
  b^2 + c^2 - b * c = 4 ∧ 
  1 / 2 * b * c * (√3 / 2) = √3 := by
  sorry

end monotonically_decreasing_interval_and_center_of_symmetry_max_area_of_triangle_l165_165503


namespace sequence_third_term_l165_165648

theorem sequence_third_term :
  (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) + 3) → 
  (a 1 = 1) → 
  a 3 = 31 :=
by
  intros hrec hinit
  let a1 := hinit
  have a2 := hrec 2 (by norm_num)
  have a3 := hrec 3 (by norm_num)
  sorry

end sequence_third_term_l165_165648


namespace fixed_line_circumcenter_l165_165939

open EuclideanGeometry

theorem fixed_line_circumcenter 
  (ABCDE : RegularPentagon) 
  (P : Point) (PA_ne_PB : PA ≠ PB) 
  (Q : Point) 
  (R : Point)
  (circ_PAE : Circle)
  (circ_PBC : Circle)
  (circ_DPQ : Circle) 
  (is_interior_P : P ∈ interior_AB)
  (circumcenter_R : R = circumcenter (triangle DPQ))
  (circumcircle_intersection : Q ∈ (circumcircle (triangle PAE) ∩ circumcircle (triangle PBC))) :
  fixed_line (circumcenter (triangle DPQ)) :=
by 
  -- Proof omitted
  sorry

end fixed_line_circumcenter_l165_165939


namespace sugar_percentage_l165_165055

variable (original_sol_weight replacement_weight original_sugar_percentage replacement_sugar_percentage resulting_sugar_percentage : ℝ)

-- Conditions
def original_solution_sugar := (original_sugar_percentage / 100) * original_sol_weight
def replace_weight := replacement_weight
def replace_sugar := (original_sugar_percentage / 100) * replace_weight
def add_sugar := (replacement_sugar_percentage / 100) * replace_weight
def resulting_sol_weight := original_sol_weight
def resulting_sugar := original_solution_sugar - replace_sugar + add_sugar

-- Theorem Statement
theorem sugar_percentage (h1 : original_sol_weight = 100) 
                         (h2 : replacement_weight = original_sol_weight / 4)
                         (h3 : original_sugar_percentage = 8) 
                         (h4 : replacement_sugar_percentage = 40) 
                         (h5 : resulting_sol_weight = original_sol_weight) :
  resulting_sugar / resulting_sol_weight * 100 = 16 :=
by
  sorry

end sugar_percentage_l165_165055


namespace external_tangent_length_l165_165116

theorem external_tangent_length {r1 r2 d : ℝ} (hr1 : r1 = 7) (hr2 : r2 = 9) (hd : d = 50) :
  ∃ l : ℝ, l = real.sqrt (d^2 - (r2 - r1)^2) ∧ l ≈ 49.96 :=
by {
  use real.sqrt (d^2 - (r2 - r1)^2),
  split,
  { refl },
  { norm_num,
    exact real.sqrt_eq_r_approx (2496: ℝ) 49.96 sorry }
}

end external_tangent_length_l165_165116


namespace remaining_budget_l165_165229

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l165_165229


namespace number_exceeds_25_percent_by_150_l165_165204

theorem number_exceeds_25_percent_by_150 (x : ℝ) : (0.25 * x + 150 = x) → x = 200 :=
by
  sorry

end number_exceeds_25_percent_by_150_l165_165204


namespace find_four_real_numbers_l165_165354

theorem find_four_real_numbers (x1 x2 x3 x4 : ℝ) :
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end find_four_real_numbers_l165_165354


namespace minimum_treasure_buried_l165_165293

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165293


namespace maxPeopleCanRideFerrisWheel_l165_165739

def ferrisWheelMaxCapacity : ℕ := 65

noncomputable def totalSeats : ℕ := 14
noncomputable def wheelchairSeats : ℕ := 2
noncomputable def wheelchairCapacity : ℕ := 4
noncomputable def otherSeatsCapacities : List ℕ := [5, 5, 6, 6, 6, 6, 7, 7, 8, 9]
noncomputable def totalCapacity : ℕ :=
  (wheelchairSeats * wheelchairCapacity) + (List.sum otherSeatsCapacities)
noncomputable def reservedForQueue : ℕ := ((totalCapacity * 10) / 100).ceil.toNat

theorem maxPeopleCanRideFerrisWheel :
  ferrisWheelMaxCapacity = totalCapacity - reservedForQueue := by
  sorry

end maxPeopleCanRideFerrisWheel_l165_165739


namespace right_triangle_of_angle_condition_l165_165078

-- Defining the angles of the triangle
variables (α β γ : ℝ)

-- Defining the condition where the sum of angles in a triangle is 180 degrees
def sum_of_angles_in_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Defining the given condition 
def angle_condition (γ α β : ℝ) : Prop :=
  γ = α + β

-- Stating the theorem to be proved
theorem right_triangle_of_angle_condition (α β γ : ℝ) :
  sum_of_angles_in_triangle α β γ → angle_condition γ α β → γ = 90 :=
by
  intro hsum hcondition
  sorry

end right_triangle_of_angle_condition_l165_165078


namespace f_2021_eq_2_l165_165578

-- Define the function f with the given properties.
axiom f : ℝ → ℝ
axiom f_pos : ∀ x, 0 < x → 0 < f x
axiom f_eqn : ∀ x y, 0 < y → y < x → f (x - y) = sqrt (f (x * y) + 2)

-- The goal: Prove that f(2021) = 2.
theorem f_2021_eq_2 : f 2021 = 2 :=
sorry

end f_2021_eq_2_l165_165578


namespace minimum_value_of_f_l165_165096

noncomputable def f (x : ℝ) : ℝ := 3 * x + 1 + 12 / (x ^ 2)

theorem minimum_value_of_f : ∃ x > 0, f x = 10 ∧ ∀ y > 0, f y ≥ f x :=
begin
  sorry
end

end minimum_value_of_f_l165_165096


namespace domain_of_func_l165_165636

noncomputable def dom := set ℝ

def func (x : ℝ) : ℝ := x^(-3/4)

theorem domain_of_func : ∀ x, func x ∈ ℝ → x ∈ set.Ioi 0 :=
by sorry

end domain_of_func_l165_165636


namespace sum_of_divisors_of_12_l165_165411

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165411


namespace wire_pieces_difference_l165_165736

theorem wire_pieces_difference (L1 L2 : ℝ) (H1 : L1 = 14) (H2 : L2 = 16) : L2 - L1 = 2 :=
by
  rw [H1, H2]
  norm_num

end wire_pieces_difference_l165_165736


namespace find_larger_number_l165_165783

theorem find_larger_number :
  ∃ (L S : ℕ), L - S = 1365 ∧ L = 6 * S + 15 ∧ L = 1635 :=
sorry

end find_larger_number_l165_165783


namespace sum_of_divisors_of_12_l165_165387

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165387


namespace shaded_region_area_correct_l165_165103

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length ^ 2

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def side_lengths : list ℕ := [2, 4, 6, 8, 10]

def large_square_side : ℕ := 10

def small_square_side : ℕ := 2

def triangles_bases_and_heights : list (ℕ × ℕ) := [(2, 4), (2, 6), (2, 8), (2, 10)]

noncomputable def total_area_of_triangles (lst : list (ℕ × ℕ)) : ℕ :=
  lst.foldr (λ p acc, acc + 2 * area_of_triangle p.1 p.2) 0

noncomputable def shaded_region_area (large_square_side small_square_side : ℕ) 
  (triangles_bases_and_heights : list (ℕ × ℕ)) : ℕ :=
  area_of_square large_square_side - area_of_square small_square_side - total_area_of_triangles triangles_bases_and_heights

theorem shaded_region_area_correct :
  shaded_region_area large_square_side small_square_side triangles_bases_and_heights = 40 :=
sorry

end shaded_region_area_correct_l165_165103


namespace cats_added_l165_165238

theorem cats_added (siamese_cats house_cats total_cats : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : total_cats = 28) : 
  total_cats - (siamese_cats + house_cats) = 10 := 
by 
  sorry

end cats_added_l165_165238


namespace conjugate_of_z_l165_165085

-- Definition of the imaginary unit i and the complex number z
def i := Complex.I
def z := 2 / (1 - i)

-- Theorem statement: The conjugate of z is 1 - i
theorem conjugate_of_z : Complex.conj z = 1 - i := sorry

end conjugate_of_z_l165_165085


namespace angle_passing_through_point_l165_165655

-- Definition of the problem conditions
def is_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  let r := Real.sqrt (x^2 + y^2);
  (x = Real.cos α * r) ∧ (y = Real.sin α * r)

-- Lean 4 statement of the problem
theorem angle_passing_through_point (α : ℝ) :
  is_terminal_side_of_angle 1 (-1) α → α = - (Real.pi / 4) :=
by sorry

end angle_passing_through_point_l165_165655


namespace central_projection_line_l165_165060

theorem central_projection_line (O : ℝ^3) (l : Set ℝ^3) (α_2 : Set ℝ^3) :
  (¬ exceptional_line l O α_2) → (∀ P ∈ l, ∃ P' ∈ α_2, central_projection O P P') → is_line (central_projection O l α_2) :=
by
  sorry

end central_projection_line_l165_165060


namespace trajectory_and_radius_range_l165_165539

structure Point :=
  (x : ℝ)
  (y : ℝ)

def circle (r : ℝ) := { P : Point // P.x^2 + P.y^2 = r^2 }

def segment_A_B (λ : ℝ) (A B : Point) (P : Point) : Prop :=
  0 ≤ λ ∧ λ ≤ 1 ∧ (P.x - A.x = λ * (B.x - A.x)) ∧ (P.y - A.y = λ * (B.y - A.y))

theorem trajectory_and_radius_range :
  ∀ (r : ℝ) (A B P : Point),
  circle r A ∧ A = {x := 4, y := 0} ∧
  B = {x := 0, y := 4} ∧
  (∃ λ, segment_A_B λ A B P) →
  (∀ P, P ∈ {P : Point | P.x + P.y - 4 = 0 ∧ 0 ≤ P.x ∧ P.x ≤ 4}) →
  (∃ l M N : Point,
    l = { P : Point // ∃ P, ∃ N, ∃ M,
      circle r N ∧ M.x = (N.x + P.x) / 2 ∧ M.y = (N.y + P.y) / 2 ∧
      (∀ P, (abs ((P.x) + (4 - P.y) - 3 * r^2) / sqrt (4 * P.x^2 + 4 * (4 - P.y)^2)) ≤ r)
      (2*P.x^2 - 8*P.y + 16 ≤ 9*r^2)
    }) →
    ∃ (r_range : set ℝ), r_range = set.Icc (4 / 3) (2 * real.sqrt 2) 
    ∧ ∀ (r : ℝ), r ∈ r_range → r ≠ 2 * real.sqrt 2 :=
sorry

end trajectory_and_radius_range_l165_165539


namespace sum_of_divisors_of_12_l165_165429

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165429


namespace geometric_sequence_sum_l165_165841

theorem geometric_sequence_sum (a : ℝ) (S : ℕ+ → ℝ) :
  (∀ n : ℕ+, S n = 2^n + a) → (a = -1) :=
by
  sorry

end geometric_sequence_sum_l165_165841


namespace part_a_part_b_l165_165796

noncomputable def f (z : ℂ) := ∑' k : ℤ, 1 / (Complex.log z + 2 * k * π * Complex.I) ^ 4

theorem part_a (z : ℂ) (hz : z ≠ 0 ∧ z ≠ 1) :
  ∃ P Q : ℂ[X], f z = P.eval z / Q.eval z :=
sorry

theorem part_b (z : ℂ) (hz : z ≠ 0 ∧ z ≠ 1) :
  f z = (z ^ 3 + 4 * z ^ 2 + z) / (6 * (z - 1) ^ 4) :=
sorry

end part_a_part_b_l165_165796


namespace parabola_y_intercepts_l165_165765

theorem parabola_y_intercepts : 
  (∀ y, (3 * y ^ 2 - 4 * y + 5 ≠ 0)) → (∀ x y, x = 3 * y ^ 2 - 4 * y + 5 → x ≠ 0) :=
by intros h y hy
   exact h y

end parabola_y_intercepts_l165_165765


namespace volume_relation_l165_165457

-- Definition of the volumes
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * real.pi * r^2 * h
def volume_cylinder (r h : ℝ) : ℝ := real.pi * r^2 * h
def volume_sphere (h : ℝ) : ℝ := (4 / 3) * real.pi * h^3

-- Mathematical proof problem: proving the relation A + M = C given r = h
theorem volume_relation (r h : ℝ) (hrh : r = h) : volume_cone r h + volume_cylinder r h = volume_sphere h :=
by
  sorry

end volume_relation_l165_165457


namespace sum_of_divisors_of_12_l165_165437

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165437


namespace sum_of_divisors_of_12_l165_165391

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165391


namespace second_divisor_l165_165359

theorem second_divisor (x : ℕ) (k q : ℤ) : 
  (197 % 13 = 2) → 
  (x > 13) → 
  (197 % x = 5) → 
  x = 16 :=
by sorry

end second_divisor_l165_165359


namespace min_n_for_sets_l165_165976

-- Define the sets S and T
def S (a b c : Fin n → ℝ) : Fin n → Fin n → Fin n → Prop :=
λ i j k => (a i + b j + c k < 1)

def T (a b c : Fin n → ℝ) : Fin n → Fin n → Fin n → Prop :=
λ i j k => (a i + b j + c k > 2)

-- The diameter of S and T sets
def S_size (a b c : Fin n → ℝ) : ℕ :=
Finset.card {ijk : Fin n × Fin n × Fin n | S a b c ijk.1 ijk.2 ijk.3}

def T_size (a b c : Fin n → ℝ) : ℕ :=
Finset.card {ijk : Fin n × Fin n × Fin n | T a b c ijk.1 ijk.2 ijk.3}

-- The problem statement
theorem min_n_for_sets :
  ∃ (n : ℕ), (∀ (a b c : Fin n → ℝ), (∀ i, a i ∈ Set.Icc (0 : ℝ) 1) ∧ 
  (∀ j, b j ∈ Set.Icc (0 : ℝ) 1) ∧ (∀ k, c k ∈ Set.Icc (0 : ℝ) 1) →
  S_size a b c ≥ 2018 ∧ T_size a b c ≥ 2018) ∧ n = 20 := 
sorry

end min_n_for_sets_l165_165976


namespace triangle_incircle_midpoint_l165_165093

theorem triangle_incircle_midpoint (A B C X Y K : Point) 
  (h1 : incircle_touching_sides A B C X Y)
  (h2 : is_midpoint_of_arc K A B (circumcircle A B C) C)
  (h3 : bisects_segment XY AK) :
  measure_angle A B C = 120 :=
begin
  sorry
end

end triangle_incircle_midpoint_l165_165093


namespace smallest_n_lil_wayne_rain_l165_165945

noncomputable def probability_rain (n : ℕ) : ℝ := 
  1 / 2 - 1 / 2^(n + 1)

theorem smallest_n_lil_wayne_rain :
  ∃ n : ℕ, probability_rain n > 0.499 ∧ (∀ m : ℕ, m < n → probability_rain m ≤ 0.499) ∧ n = 9 := 
by
  sorry

end smallest_n_lil_wayne_rain_l165_165945


namespace treasure_15_signs_l165_165299

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165299


namespace inequality_proof_equality_condition_l165_165033

variables (a b c : ℝ) (h : a ≠ 0)
axiom real_roots_in_interval : ∃ x ∈ Icc (-1 : ℝ) 1, 2 * a * x^2 + b * x + c = 0

theorem inequality_proof :
  ∃ x ∈ Icc (-1 : ℝ) 1, 2 * a * x^2 + b * x + c = 0 →
  min c (a + c + 1) ≤ max (abs (b - a + 1)) (abs (b + a - 1)) :=
by sorry

theorem equality_condition :
  ∃ x ∈ Icc (-1 : ℝ) 1, 2 * a * x^2 + b * x + c = 0 →
  (min c (a + c + 1) = max (abs (b - a + 1)) (abs (b + a - 1))
  ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a ≤ -1 ∧ 2 * a - abs b + c = 0)) :=
by sorry

end inequality_proof_equality_condition_l165_165033


namespace estimate_2_sqrt_5_l165_165346

theorem estimate_2_sqrt_5: 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_2_sqrt_5_l165_165346


namespace sum_positive_integral_values_l165_165386

theorem sum_positive_integral_values {n : ℕ} (hn : 0 < n) (h : (n + 12) % n = 0) : 
  (∑ n in Finset.filter (λ n, (n + 12) % n = 0) (Finset.range 13)) = 28 :=
by
  sorry

end sum_positive_integral_values_l165_165386


namespace min_treasure_signs_buried_l165_165287

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165287


namespace sum_of_divisors_of_12_l165_165412

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165412


namespace math_problem_l165_165123

def calc_expr : ℝ :=
  (10^3) - (1/3 * 270) + (Real.sqrt 144)

theorem math_problem : calc_expr = 922 := by
  -- sorry allows us to skip the proof
  sorry

end math_problem_l165_165123


namespace sum_fourth_powers_lt_1000_l165_165176

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165176


namespace dynamically_balanced_count_l165_165259

def is_dynamically_balanced (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  |(a + b) - (c + d)| = 1

def dynamically_balanced_integers : Finset ℕ :=
  {n ∈ Finset.range 10000 | 1000 ≤ n ∧ is_dynamically_balanced n}

theorem dynamically_balanced_count : dynamically_balanced_integers.card = 722 :=
  sorry

end dynamically_balanced_count_l165_165259


namespace sufficient_condition_for_beta_l165_165024

theorem sufficient_condition_for_beta (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (x ≤ m)) → (3 ≤ m) :=
by
  sorry

end sufficient_condition_for_beta_l165_165024


namespace ratio_of_areas_is_correct_l165_165030

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def ratio_of_triangle_areas 
(A1 A2 A3 : ℝ × ℝ)
(A4 : ℝ × ℝ := A1) (A5 : ℝ × ℝ := A2) 
(B1 B2 B3 C1 C2 C3 D1 D2 D3 E1 E2 E3 : ℝ × ℝ) : ℝ :=
let B1 := midpoint A1 A2,
    B2 := midpoint A2 A3,
    B3 := midpoint A3 A4,
    C1 := midpoint A1 B1,
    C2 := midpoint A2 B2,
    C3 := midpoint A3 B3,
    D1 := -- derive D1 as defined,
    D2 := -- derive D2 as defined,
    D3 := -- derive D3 as defined,
    E1 := -- derive E1 as defined,
    E2 := -- derive E2 as defined,
    E3 := -- derive E3 as defined
in 
-- function to calculate the area of a triangle
let area (P Q R : ℝ × ℝ) : ℝ := 
    abs (((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1)) / 2) 
in
(area D1 D2 D3) / (area E1 E2 E3)

theorem ratio_of_areas_is_correct (A1 A2 A3 : ℝ × ℝ) 
(B1 B2 B3 C1 C2 C3 D1 D2 D3 E1 E2 E3 : ℝ × ℝ) :
  ratio_of_triangle_areas A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 E1 E2 E3 = 25 / 49 := sorry

end ratio_of_areas_is_correct_l165_165030


namespace problem_statement_l165_165038

open Real

theorem problem_statement (x y : ℝ) (hx_gt_one : 1 < x) (hy_gt_one : 1 < y)
    (h : (log x / log 2)^3 + (log y / log 3)^3 + 9 = 9 * (log x / log 2) * (log y / log 3)) :
    x^2 + y^3 = 2^(2 * real.cbrt 3) + 27 :=
by
  sorry

end problem_statement_l165_165038


namespace cindy_play_area_l165_165751

noncomputable def barn_length : ℝ := 4
noncomputable def barn_width : ℝ := 5
noncomputable def leash_length : ℝ := 6
noncomputable def arc_degrees : ℝ := 270
noncomputable def full_circle_degrees : ℝ := 360
noncomputable def additional_sector_degrees : ℝ := 90
noncomputable def additional_sector_radius : ℝ := 1

theorem cindy_play_area :
  let area_arc := (arc_degrees / full_circle_degrees) * real.pi * (leash_length ^ 2),
      area_additional_sectors := 2 * (additional_sector_degrees / full_circle_degrees) * real.pi * (additional_sector_radius ^ 2),
      total_area := area_arc + area_additional_sectors
  in total_area = (55 / 2) * real.pi :=
by
  sorry

end cindy_play_area_l165_165751


namespace sum_of_divisors_of_12_l165_165438

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165438


namespace regular_tetrahedron_height_eq_4r_l165_165477

noncomputable def equilateral_triangle_inscribed_circle_height (r : ℝ) : ℝ :=
3 * r

noncomputable def regular_tetrahedron_inscribed_sphere_height (r : ℝ) : ℝ :=
4 * r

theorem regular_tetrahedron_height_eq_4r (r : ℝ) :
  regular_tetrahedron_inscribed_sphere_height r = 4 * r :=
by
  unfold regular_tetrahedron_inscribed_sphere_height
  sorry

end regular_tetrahedron_height_eq_4r_l165_165477


namespace sum_of_fourth_powers_le_1000_l165_165144

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165144


namespace diagonals_of_square_equal_proof_l165_165843

-- Let us define the conditions
def square (s : Type) : Prop := True -- Placeholder for the actual definition of square
def parallelogram (p : Type) : Prop := True -- Placeholder for the actual definition of parallelogram
def diagonals_equal (q : Type) : Prop := True -- Placeholder for the property that diagonals are equal

-- Given conditions
axiom square_is_parallelogram {s : Type} (h1 : square s) : parallelogram s
axiom diagonals_of_parallelogram_equal {p : Type} (h2 : parallelogram p) : diagonals_equal p
axiom diagonals_of_square_equal {s : Type} (h3 : square s) : diagonals_equal s

-- Proof statement
theorem diagonals_of_square_equal_proof (s : Type) (h1 : square s) : diagonals_equal s :=
by
  apply diagonals_of_square_equal h1

end diagonals_of_square_equal_proof_l165_165843


namespace sum_of_divisors_of_12_l165_165406

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165406


namespace Janabel_widgets_total_is_correct_l165_165053

theorem Janabel_widgets_total_is_correct :
  let a : ℕ → ℕ := λ n, 2 + (n - 1) * 3 in
  let S : ℕ → ℕ := λ n, n * (a 1 + a n) / 2 in
  S 15 = 345 :=
by
  sorry

end Janabel_widgets_total_is_correct_l165_165053


namespace solve_log_equation_l165_165777

theorem solve_log_equation :
  ∃ x : ℝ, logb 125 (3 * x - 2) = -1 / 3 ∧ x = 11 / 15 := 
by
  use 11 / 15
  split
  · sorry
  · rfl

end solve_log_equation_l165_165777


namespace log_quadratic_solutions_l165_165068

noncomputable def quadratic_solutions (a b c : ℝ) : ℝ × ℝ :=
  let disc := b^2 - 4 * a * c in
  ((-b + Real.sqrt disc) / (2 * a), (-b - Real.sqrt disc) / (2 * a))

theorem log_quadratic_solutions :
  ∀ (a : ℝ), log 10 (3 * a^2 - 18 * a) = 1 → (a = 3 + Real.sqrt 37 ∨ a = 3 - Real.sqrt 37) :=
begin
  intros a h,
  have h1 : 3 * a^2 - 18 * a = 10,
  { rw [log, Real.log_eq_iff_exp_eq] at h,
    exact h.symm },
  have h2 : 3*a^2 - 18*a - 10 = 0,
  { linarith },
  sorry
end

end log_quadratic_solutions_l165_165068


namespace calculate_final_price_l165_165219

def original_price : ℝ := 120
def fixture_discount : ℝ := 0.20
def decor_discount : ℝ := 0.15

def discounted_price_after_first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * (1 - d)

def final_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let price_after_first_discount := discounted_price_after_first_discount p d1
  price_after_first_discount * (1 - d2)

theorem calculate_final_price :
  final_price original_price fixture_discount decor_discount = 81.60 :=
by sorry

end calculate_final_price_l165_165219


namespace sum_of_divisors_of_12_l165_165370

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165370


namespace rationalize_fraction_l165_165667

theorem rationalize_fraction :
  (5 * real.cbrt 6 - 2 * real.cbrt 12) / (4 * real.cbrt 12 + 2 * real.cbrt 6) = 
  (20 * real.cbrt 4 - 12 * real.cbrt 2 - 11) / 34 :=
by
  sorry

end rationalize_fraction_l165_165667


namespace henry_books_l165_165504

def initial_books := 99
def boxes := 3
def books_per_box := 15
def room_books := 21
def coffee_table_books := 4
def kitchen_books := 18
def picked_books := 12

theorem henry_books :
  (initial_books - (boxes * books_per_box + room_books + coffee_table_books + kitchen_books) + picked_books) = 23 :=
by
  sorry

end henry_books_l165_165504


namespace eccentricity_of_ellipse_fixed_point_of_line_MN_l165_165844

-- Definitions for the given problem
def ellipse (a : ℝ) : Prop := a > 1 ∧ ∃ (x y : ℝ), (x^2 / a^2) + y^2 = 1
def circle : Prop := ∃ (x y : ℝ), x^2 + (y - 3/2)^2 = 4

-- Intersections and properties
def intersects_at_points_A_and_B : Prop :=
  ∃ (A B : ℝ × ℝ), ellipse a ∧ circle ∧ dist A B = 2 * sqrt 3

def intersects_negative_y_axis_at_D : Prop :=
  ∃ D : ℝ × ℝ, circle ∧ D = (0, -1/2)

-- Theorems
theorem eccentricity_of_ellipse (a : ℝ) (h_a : ellipse a)
  (h_inter : intersects_at_points_A_and_B) : 
  ∃ e : ℝ, e = sqrt 3 / 2 :=
sorry

theorem fixed_point_of_line_MN' (a : ℝ)
  (h_ellipse : ellipse a)
  (h_inter : intersects_at_points_A_and_B)
  (h_circle : circle)
  (h_D : intersects_negative_y_axis_at_D) :
  ∃ F : ℝ × ℝ, F = (0, -2) :=
sorry

end eccentricity_of_ellipse_fixed_point_of_line_MN_l165_165844


namespace sum_of_squares_of_perfect_squares_l165_165134

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165134


namespace two_values_of_z_l165_165579

theorem two_values_of_z :
  ∃ z : ℂ, (|z| = 7 ∧ -complex.I * complex.conj z = z) → ∃ z₁ z₂ : ℂ, 
    z₁ ≠ z₂ ∧ |z₁| = 7 ∧ -complex.I * complex.conj z₁ = z₁ ∧ 
    |z₂| = 7 ∧ -complex.I * complex.conj z₂ = z₂ :=
begin
  sorry
end

end two_values_of_z_l165_165579


namespace problem_remainder_A_mod_126_l165_165933

-- Define A as the concatenation of numbers from 100 to 799
noncomputable def A : ℕ := (List.range' 100 700).join.!

-- The main theorem to prove A ≡ 91 (mod 126)
theorem problem_remainder_A_mod_126 : A % 126 = 91 := by
  sorry

end problem_remainder_A_mod_126_l165_165933


namespace sum_of_babies_ages_in_five_years_l165_165978

-- Given Definitions
def lioness_age := 12
def hyena_age := lioness_age / 2
def lioness_baby_age := lioness_age / 2
def hyena_baby_age := hyena_age / 2

-- The declaration of the statement to be proven
theorem sum_of_babies_ages_in_five_years : (lioness_baby_age + 5) + (hyena_baby_age + 5) = 19 :=
by 
  sorry 

end sum_of_babies_ages_in_five_years_l165_165978


namespace total_fare_for_100_miles_l165_165730

theorem total_fare_for_100_miles (b c : ℝ) (h₁ : 200 = b + 80 * c) : 240 = b + 100 * c :=
sorry

end total_fare_for_100_miles_l165_165730


namespace range_of_abs_diff_roots_l165_165803

noncomputable def f (a b c x : ℝ) : ℝ := a * x ^ 2 + (b - a) * x + c - b

theorem range_of_abs_diff_roots (a b c x1 x2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0)
  (h4 : x1 + x2 = (2 + c / a)) (h5 : x1 * x2 = (1 + 2 * c / a)) :
  ∃ (d1 d2 : ℝ), (abs (x1 - x2) ∈ set.Ioo d1 d2) ∧ d1 = 3 / 2 ∧ d2 = 2 * sqrt 3 := 
sorry

end range_of_abs_diff_roots_l165_165803


namespace area_of_rectangle_l165_165057

-- Definitions based on provided conditions
structure Point where
  x : ℝ
  y : ℝ

structure Rectangle where
  A B C D : Point
  h1: A.y = B.y
  h2: D.y = C.y
  h3: D.x = A.x
  h4: C.x = B.x
  h5: B.x - A.x = 2 * (B.y - D.y)

def E (D C : Point) : Point :=
  { x := (C.x + 2 * D.x) / 3, y := D.y }

def lineEquation (P Q : Point) (x : ℝ) : ℝ :=
  ((Q.y - P.y) / (Q.x - P.x)) * (x - P.x) + P.y

def intersects (P Q P1 Q1 : Point) : Point :=
  let m1 := (Q.y - P.y) / (Q.x - P.x)
  let m2 := (Q1.y - P1.y) / (Q1.x - P1.x)
  let x := (m1 * P.x - m2 * P1.x + P1.y - P.y) / (m1 - m2)
  let y := (m1 * (x - P.x) + P.y)
  { x := x, y := y }

def areaTriangle (B E F : Point) : ℝ :=
  0.5 * abs (B.x * (E.y - F.y) + E.x * (F.y - B.y) + F.x * (B.y - E.y))

def areaRectangle (B D : Point) : ℝ :=
  abs ((B.x - D.x) * (B.y - D.y))

-- Given conditions for proof
theorem area_of_rectangle (A B C D : Point) (h : areaTriangle B (E D C) (intersects B (E D C) A C) = 18)
  (h1: A.x = 0) (h2: A.y = 0) (h3: B.y = C.y) (h4: D.x = C.x) (h5: C.x = 2 * D.y) :
  areaRectangle B D = 108 :=
by
  sorry

end area_of_rectangle_l165_165057


namespace repair_cost_total_l165_165561

def hourly_labor_cost : ℝ := 75
def labor_hours : ℝ := 16
def part_cost : ℝ := 1200
def labor_cost : ℝ := hourly_labor_cost * labor_hours
def total_cost : ℝ := labor_cost + part_cost

theorem repair_cost_total : total_cost = 2400 := 
by
  -- Proof omitted
  sorry

end repair_cost_total_l165_165561


namespace count_solutions_l165_165208

def g (x : ℝ) : ℝ := 3 * Real.cos (π * x)

theorem count_solutions :
  ∃ (S : Finset ℝ), S.card = 36 ∧ ∀ x ∈ S, -1.5 ≤ x ∧ x ≤ 1.5 ∧ g (g (g x)) = g x :=
  sorry

end count_solutions_l165_165208


namespace karen_boxes_l165_165923

theorem karen_boxes (cases : ℕ) (boxes_per_case : ℕ) (h_cases : cases = 3) (h_boxes_per_case : boxes_per_case = 12) :
  cases * boxes_per_case = 36 :=
by {
  rw [h_cases, h_boxes_per_case],
  norm_num,
  sorry
}

end karen_boxes_l165_165923


namespace sum_of_divisors_of_12_l165_165408

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165408


namespace sum_of_divisors_of_12_l165_165424

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165424


namespace find_A_capital_l165_165249

variable (P Ca : ℝ)
variable (income_increase profit_increase : ℝ)
axiom partners (a_b_share : ℝ) (b_c_share : ℝ)
axiom income_increase_eq : income_increase = 300
axiom original_profit_rate new_profit_rate : ℝ
axiom profit_rate_increase : new_profit_rate - original_profit_rate = 0.02
axiom a_profit_share : a_b_share = 2 / 3
axiom others_share : b_c_share = 1 / 6

noncomputable def A_capital : Prop :=
  ∃ (P Ca : ℝ), income_increase = (a_b_share * (P * profit_rate_increase)) ∧
                Ca = (a_b_share * P) / original_profit_rate

/-- The capital of partner A --/
theorem find_A_capital : A_capital :=
  exists.intro P (exists.intro Ca (and.intro sorry sorry))

end find_A_capital_l165_165249


namespace find_k_l165_165839

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (hSn : ∀ n, S n = ((1 - 2 ^ (n+1)) / (1 - 2))) 
  (hLog : log 4 (S k + 1) = 4) : 
  k = 8 := 
sorry

end find_k_l165_165839


namespace balls_left_correct_l165_165506

def initial_balls : ℕ := 10
def balls_removed : ℕ := 3
def balls_left : ℕ := initial_balls - balls_removed

theorem balls_left_correct : balls_left = 7 := 
by
  -- Proof omitted
  sorry

end balls_left_correct_l165_165506


namespace sum_of_a5_a6_l165_165893

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
a 1 + a 2 = 1 ∧ a 3 + a 4 = 4 ∧ q^2 = 4

theorem sum_of_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) (h_cond : geometric_conditions a q) :
  a 5 + a 6 = 16 :=
sorry

end sum_of_a5_a6_l165_165893


namespace treasure_under_minimum_signs_l165_165278

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165278


namespace min_value_of_expression_l165_165885

-- Conditions
variables {A B C : ℝ}
-- A, B, C are acute angles
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π/2
-- Condition on sines of the angles
def sin_sum_eq_two : Prop := sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 2

-- Main theorem
theorem min_value_of_expression (hA : is_acute A) (hB : is_acute B) (hC : is_acute C) (hsin_sum : sin_sum_eq_two) :
  (1 / (sin A ^ 2 * (cos B ^ 2) ^ 2) + 1 / (sin B ^ 2 * (cos C ^ 2) ^ 2) + 1 / (sin C ^ 2 * (cos A ^ 2) ^ 2)) = 81 / 2 :=
sorry

end min_value_of_expression_l165_165885


namespace sum_real_imaginary_part_l165_165580

noncomputable def imaginary_unit : ℂ := Complex.I

theorem sum_real_imaginary_part {z : ℂ} (h : z * imaginary_unit = 1 + imaginary_unit) :
  z.re + z.im = 2 := 
sorry

end sum_real_imaginary_part_l165_165580


namespace sum_of_fourth_powers_below_1000_l165_165161

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165161


namespace projection_d_l165_165574

open Real

variables (c d v : ℝ × ℝ)

def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let k := (u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)
  in (k * v.1, k * v.2)

theorem projection_d (hc : orthogonal c d)
  (hp : proj (4, -1) c = (1, 2)) : proj (4, -1) d = (3, -3) :=
by 
  sorry

end projection_d_l165_165574


namespace sixth_expression_nth_expression_product_calculation_l165_165954

theorem sixth_expression : 6 * 8 + 1 = 7^2 := by sorry

theorem nth_expression (n : ℕ) : n * (n + 2) + 1 = (n + 1)^2 := by sorry

theorem product_calculation : 
  (∏ k in Finset.range 98, (1 + 1 / ((k + 1) * (k + 3)))) = (99 / 50) := by sorry

end sixth_expression_nth_expression_product_calculation_l165_165954


namespace simplify_expression_l165_165348

theorem simplify_expression (a b c : ℝ) : [a - (b - c)] - [(a - b) - c] = 2c :=
by sorry

end simplify_expression_l165_165348


namespace parallel_trans_new_basis_l165_165821

-- Definitions for vectors and conditions
variables (V : Type) [InnerProductSpace ℝ V]
variables (a b c : V)

-- Conditions: a, b, c are unit vectors
def is_unit_vector (v : V) : Prop := ∥v∥ = 1
axiom ha : is_unit_vector a
axiom hb : is_unit_vector b
axiom hc : is_unit_vector c

-- Option A: Parallel relation
def parallel (u v : V) : Prop := ∃ (k : ℝ), u = k • v

-- Theorem for option A: Proving transitivity of parallel vectors
theorem parallel_trans : parallel a b → parallel b c → parallel a c :=
sorry

-- Option C: Definitions for basis and linear independence
variable {V}
def is_basis (s : Set V) : Prop := ∀ v ∈ Set.Univ, ∃ l : (s →₀ ℝ), v = Finsupp.total _ _ _ id l

-- Theorem for option C: Proving new set is a basis given the old set is a basis
theorem new_basis (basis_a_b_c : is_basis ({a, b, c} : Set V)) :
  is_basis ({a + b, b + c, c + a} : Set V) :=
sorry

end parallel_trans_new_basis_l165_165821


namespace crayon_cost_is_37_l165_165950

noncomputable def crayon_cost (class_size : ℕ) (more_than_half : ℕ → Prop) (total_cost : ℕ) : ℕ :=
  let s := { s : ℕ // more_than_half s ∧ s > class_size / 2 }
  let c := { c : ℕ // c * some s.1 > some s.1 }
  let n := { n : ℕ // n < some c.1 }
  let total_cost_eq : total_cost = some s.1 * some c.1 * some n.1 := sorry
  some c.1

theorem crayon_cost_is_37 : crayon_cost 50 (λ s, s > 25) 1998 = 37 := by
  sorry

end crayon_cost_is_37_l165_165950


namespace value_of_expression_l165_165455

theorem value_of_expression (x y : ℝ) (hx : x = sqrt 3 + 1) (hy : y = sqrt 3 - 1) :
  x^2 + x * y + y^2 = 10 :=
by
  sorry

end value_of_expression_l165_165455


namespace inequality_condition_l165_165589

theorem inequality_condition 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c > Real.sqrt (a^2 + b^2)) := 
sorry

end inequality_condition_l165_165589


namespace set_equality_power_evaluation_l165_165498

theorem set_equality_power_evaluation (a b : ℝ) (h : {a, b / a, 1} = {a^2, a + b, 0}) :
  a ^ 2015 + b ^ 2016 = -1 :=
by
  sorry

end set_equality_power_evaluation_l165_165498


namespace concentration_after_removing_water_l165_165684

theorem concentration_after_removing_water (initial_volume : ℝ) 
    (desired_concentration : ℝ) (removed_water : ℝ) : 
    initial_volume = 21 → 
    desired_concentration = 0.60 → 
    removed_water = 7 → 
    let final_volume := initial_volume - removed_water in 
    (desired_concentration * final_volume) = 0.4 * initial_volume :=
by 
  intros h_initial_volume h_desired_concentration h_removed_water 
  rw [h_initial_volume, h_desired_concentration, h_removed_water]
  have acid_initial := (0.4 * 21 : ℝ)
  have final_volume := (21 - 7 : ℝ)
  have acid_final := (0.60 * final_volume : ℝ)
  exact acid_initial = acid_final

end concentration_after_removing_water_l165_165684


namespace doctor_lindsay_revenue_l165_165194

theorem doctor_lindsay_revenue :
  let adult_patients_per_hour := 4 in
  let child_patients_per_hour := 3 in
  let adult_cost := 50 in
  let child_cost := 25 in
  let hours_worked := 8 in
  let total_revenue := (adult_patients_per_hour * adult_cost + child_patients_per_hour * child_cost) * hours_worked in
  total_revenue = 2200 :=
by
  sorry

end doctor_lindsay_revenue_l165_165194


namespace smallest_number_property_l165_165733

theorem smallest_number_property 
  (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h₁ : a = -3.14) 
  (h₂ : b = 0) 
  (h₃ : c = -π) 
  (h₄ : d = -sqrt 3) 
: (c < a) ∧ (a < d) ∧ (d < b) :=
by
  sorry

end smallest_number_property_l165_165733


namespace sum_of_divisors_of_12_l165_165410

theorem sum_of_divisors_of_12 :
  ∑ (n : ℕ) in (Finset.filter (λ d, d ∣ 12) (Finset.range 13)), n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165410


namespace solve_xyz_l165_165516

variable (x y z : ℝ)

theorem solve_xyz :
  2 * x + y = 4 →
  x + 2 * y = 5 →
  3 * x - 1.5 * y + z = 7 →
  (x + y + z) / 3 = 10 / 3 :=
by
  assume h1 : 2 * x + y = 4
  assume h2 : x + 2 * y = 5
  assume h3 : 3 * x - 1.5 * y + z = 7
  sorry

end solve_xyz_l165_165516


namespace find_a_l165_165487

def f (a: ℝ) (x: ℝ) : ℝ :=
  if x < 1 then 3 * x + 2 else x^2 + a * x

theorem find_a (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := 
by {
  intro h,
  sorry
}

end find_a_l165_165487


namespace count_valid_numbers_l165_165650

-- Define the conditions under which the problem operates
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Define the function that operates on the two-digit number
def operation (a b : ℕ) : ℕ := (10 * a + b) - (a + b) / 2

-- Specification of the problem condition
def units_digit_is_four (n : ℕ) : Prop := n % 10 = 4

-- Define the main proof problem
theorem count_valid_numbers : 
  { (a b : ℕ) // 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
                   units_digit_is_four (operation a b) } = 2 := by
  sorry

end count_valid_numbers_l165_165650


namespace sin_alpha_is_neg_5_over_13_l165_165512

-- Definition of the problem conditions
variables (α : Real) (h1 : 0 < α) (h2 : α < 2 * Real.pi)
variable (quad4 : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)
variable (h3 : Real.tan α = -5 / 12)

-- Proof statement
theorem sin_alpha_is_neg_5_over_13:
  Real.sin α = -5 / 13 :=
sorry

end sin_alpha_is_neg_5_over_13_l165_165512


namespace Zhang_Laoshi_pens_l165_165628

theorem Zhang_Laoshi_pens (x : ℕ) (original_price new_price : ℝ)
  (discount : new_price = 0.75 * original_price)
  (more_pens : x * original_price = (x + 25) * new_price) :
  x = 75 :=
by
  sorry

end Zhang_Laoshi_pens_l165_165628


namespace only_f2_is_linear_l165_165254

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

def f1 (x : ℝ) : ℝ := x ^ 3
def f2 (x : ℝ) : ℝ := -2 * x + 1
def f3 (x : ℝ) : ℝ := 2 / x
def f4 (x : ℝ) : ℝ := 2 * x ^ 2 + 1

theorem only_f2_is_linear :
  is_linear f2 ∧ ¬ is_linear f1 ∧ ¬ is_linear f3 ∧ ¬ is_linear f4 := by
  sorry

end only_f2_is_linear_l165_165254


namespace trigonometric_identity_simplification_l165_165067

theorem trigonometric_identity_simplification
  {α : Real}
  (h_quadrant: ∃ k : ℤ, α = 2 * k * π + atan 1 + 3 * π / 2)
  : cos α * sqrt ( (1 - sin α) / (1 + sin α) ) + sin α * sqrt ( (1 - cos α) / (1 + cos α) ) = cos α - sin α := 
sorry

end trigonometric_identity_simplification_l165_165067


namespace coeff_sum_square_difference_l165_165449

theorem coeff_sum_square_difference {a0 a1 a2 a3 a4 : ℝ} :
  (a0 + a2 + a4) * (a0 + a2 + a4) - (a1 + a3) * (a1 + a3) = 1 :=
begin
  have h1 : (1 + real.sqrt 2) ^ 4 = a0 + a1 + a2 + a3 + a4 := sorry,
  have h2 : (1 - real.sqrt 2) ^ 4 = a0 - a1 + a2 - a3 + a4 := sorry,
  sorry
end

end coeff_sum_square_difference_l165_165449


namespace tobias_shoveled_driveways_l165_165113

open Nat

theorem tobias_shoveled_driveways 
  (shoe_cost : ℕ) (shoe_cost = 95)
  (change : ℕ) (change = 15)
  (allowance_per_month : ℕ) (allowance_per_month = 5)
  (months_saving : ℕ) (months_saving = 3)
  (charge_mow_lawn : ℕ) (charge_mow_lawn = 15)
  (charge_shovel_driveway : ℕ) (charge_shovel_driveway = 7)
  (num_lawns_mowed : ℕ) (num_lawns_mowed = 4) : 
  ∃ num_driveways_shoveled : ℕ, num_driveways_shoveled = 5 :=
by
  sorry

end tobias_shoveled_driveways_l165_165113


namespace solution_complex_problem_l165_165209

noncomputable def complex_problem : Prop :=
  ∃ z : ℂ, 
    z = ((sqrt 3) + complex.I) / (1 - sqrt 3 * complex.I ^ 2) ∧
    z * conj z = 4 - 2 * sqrt 3

theorem solution_complex_problem : complex_problem :=
by
  have z : ℂ := ((sqrt 3) + complex.I) / (1 - sqrt 3 * complex.I ^ 2)
  use z
  split
  { 
    exact rfl 
  }
  {
    sorry -- The proof for the equality z * conj z = 4 - 2 * sqrt 3
  }

end solution_complex_problem_l165_165209


namespace pooja_cross_time_l165_165121

noncomputable def speed_in_m_s (speed_in_kmh : ℝ) : ℝ := speed_in_kmh * (1000 / 3600)

noncomputable def time_to_cross_train_B (length_A length_B speed_A_kmh speed_B_kmh speed_Pooja_kmh : ℝ) : ℝ :=
  let speed_A := speed_in_m_s speed_A_kmh
  let speed_B := speed_in_m_s speed_B_kmh
  let speed_Pooja := speed_in_m_s speed_Pooja_kmh
  let relative_speed_Pooja_B := speed_A + speed_Pooja
  length_B / relative_speed_Pooja_B

theorem pooja_cross_time 
  (length_A : ℝ := 225)
  (length_B : ℝ := 150)
  (speed_A_kmh : ℝ := 54)
  (speed_B_kmh : ℝ := 36)
  (speed_Pooja_kmh : ℝ := 1.2) :
  time_to_cross_train_B length_A length_B speed_A_kmh speed_B_kmh speed_Pooja_kmh ≈ 9.78 :=
by simp [time_to_cross_train_B]; sorry

end pooja_cross_time_l165_165121


namespace max_area_triangle_ABC_l165_165906

noncomputable def max_area_triangle (PA PB PC BC: ℝ) : ℝ :=
  if h: PA = 2 ∧ PB = 3 ∧ PC = 4 ∧ BC = 5 then 11 else 0

theorem max_area_triangle_ABC : max_area_triangle 2 3 4 5 = 11 :=
by
  simp [max_area_triangle]
  split
  { intro h; cases h with hPA h; cases h with hPB h; cases h with hPC hBC
    exact hBC } -- Replace this part with a proper proof
  { contradiction }

end max_area_triangle_ABC_l165_165906


namespace harmonic_mean_is_54_div_11_l165_165859

-- Define lengths of sides
def a : ℕ := 3
def b : ℕ := 6
def c : ℕ := 9

-- Define the harmonic mean calculation function
def harmonic_mean (x y z : ℕ) : ℚ :=
  let reciprocals_sum : ℚ := (1 / x + 1 / y + 1 / z)
  let average_reciprocal : ℚ := reciprocals_sum / 3
  1 / average_reciprocal

-- Prove that the harmonic mean of the given lengths is 54/11
theorem harmonic_mean_is_54_div_11 : harmonic_mean a b c = 54 / 11 := by
  sorry

end harmonic_mean_is_54_div_11_l165_165859


namespace find_y_in_triangle_area_l165_165530

theorem find_y_in_triangle_area :
  ∃ y : ℝ, let x1 := -1, y1 := 0, x2 := 7, y3 := -4, area := 32 in
  abs (x1 * (y - y3) + x2 * (y3 - y1) + x2 * (y1 - y)) = 64 ∧  y > y3 := by
  sorry

end find_y_in_triangle_area_l165_165530


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165156

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165156


namespace arc_length_120_degrees_l165_165836

theorem arc_length_120_degrees (π : ℝ) : 
  let R := π
  let n := 120
  (n * π * R) / 180 = (2 * π^2) / 3 := 
by
  let R := π
  let n := 120
  sorry

end arc_length_120_degrees_l165_165836


namespace sum_of_fourth_powers_below_1000_l165_165162

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165162


namespace sum_of_divisors_of_12_l165_165376

theorem sum_of_divisors_of_12 : 
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165376


namespace condition_necessary_but_not_sufficient_l165_165700

variables {α : Type*} (A B : set α) (x : α)

def necessary_but_not_sufficient : Prop :=
  (x ∈ (A ∪ B)) → (x ∈ A ∨ x ∈ B) ∧ ¬((x ∈ A ∧ x ∈ B) → (x ∈ A ∪ B))

theorem condition_necessary_but_not_sufficient :
  necessary_but_not_sufficient A B x :=
begin
  sorry
end

end condition_necessary_but_not_sufficient_l165_165700


namespace license_plates_count_l165_165507

theorem license_plates_count:
  let num_consonants := 20 in
  let num_vowels := 6 in
  let num_digits_and_symbols := 12 in
  num_consonants * num_vowels^2 * num_digits_and_symbols = 103680 :=
by
  sorry

end license_plates_count_l165_165507


namespace profit_sharing_l165_165250

theorem profit_sharing
  (A_investment B_investment C_investment total_profit : ℕ)
  (A_share : ℕ)
  (ratio_A ratio_B ratio_C : ℕ)
  (hA : A_investment = 6300)
  (hB : B_investment = 4200)
  (hC : C_investment = 10500)
  (hShare : A_share = 3810)
  (hRatio : ratio_A = 3 ∧ ratio_B = 2 ∧ ratio_C = 5)
  (hTotRatio : ratio_A + ratio_B + ratio_C = 10)
  (hShareCalc : A_share = (3/10) * total_profit) :
  total_profit = 12700 :=
sorry

end profit_sharing_l165_165250


namespace variance_const_addition_l165_165077

theorem variance_const_addition (ages : list ℝ) (variance : ℝ) (c : ℝ) (h_ages : ages = [15, 13, 15, 14, 13]) (h_variance : variance = 0.8) (h_c : c = 3) :
  list.variance (ages.map (λ x, x + c)) = variance := 
sorry

end variance_const_addition_l165_165077


namespace rosa_called_last_week_l165_165505

noncomputable def total_pages_called : ℝ := 18.8
noncomputable def pages_called_this_week : ℝ := 8.6
noncomputable def pages_called_last_week : ℝ := total_pages_called - pages_called_this_week

theorem rosa_called_last_week :
  pages_called_last_week = 10.2 :=
by
  sorry

end rosa_called_last_week_l165_165505


namespace eccentricity_of_ellipse_l165_165023

-- Define the conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def foci (a b : ℝ) (F1 F2 : ℝ × ℝ) (ellipse : Prop) : Prop :=
  ∃ c : ℝ, c = real.sqrt (a^2 - b^2) ∧
    F1 = (-c, 0) ∧
    F2 = (c, 0)

def point_on_line (a b x : ℝ) (P : ℝ × ℝ) : Prop :=
  P = (3 * a / 2, x)

def isosceles_triangle (angle : ℝ) (F1 F2 P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, θ = angle

-- Define the problem statement
theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b > 0)
  (F1 F2 P : ℝ × ℝ)
  (h2 : ellipse a b h1.1 h1.2 h1.2)
  (h3 : foci a b F1 F2 h2)
  (h4 : point_on_line a (P.snd) P)
  (h5 : isosceles_triangle (real.pi / 6) F1 F2 P) :
  let c := real.sqrt (a^2 - b^2)
  let e := c / a
  e = 3 / 4 :=
by sorry

end eccentricity_of_ellipse_l165_165023


namespace number_of_subsets_CU_A_union_B_l165_165653

def U := {1, 2, 3, 4, 5, 6}
def A := {x | x^2 - 3 * x + 2 = 0}
def B := {x | ∃ a ∈ A, x = 2 * a}
def A_union_B := {x | x ∈ A ∨ x ∈ B}
def CU (X : Set ℕ) := U \ X

theorem number_of_subsets_CU_A_union_B : 
  ∃ S, S = CU A_union_B ∧ (2 ^ (Set.card S) = 8) :=
by
  sorry

end number_of_subsets_CU_A_union_B_l165_165653


namespace lucas_additional_pet_beds_l165_165946

-- Define the conditions
def beds_per_pet : ℕ := 2
def number_of_pets : ℕ := 10
def existing_beds : ℕ := 12

-- Define the claim using these conditions
theorem lucas_additional_pet_beds :
  let total_beds := beds_per_pet * number_of_pets in
  let additional_beds := total_beds - existing_beds in
  additional_beds = 8 :=
by
  let total_beds := beds_per_pet * number_of_pets
  let additional_beds := total_beds - existing_beds
  show additional_beds = 8
  sorry

end lucas_additional_pet_beds_l165_165946


namespace equal_angles_count_l165_165662

-- Definitions corresponding to the problem conditions
def fast_clock_angle (t : ℝ) : ℝ := |30 * t - 5.5 * (t * 60)|
def slow_clock_angle (t : ℝ) : ℝ := |15 * t - 2.75 * (t * 60)|

theorem equal_angles_count :
  ∃ n : ℕ, n = 18 ∧ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 12 →
  fast_clock_angle t = slow_clock_angle t ↔ n = 18 :=
sorry

end equal_angles_count_l165_165662


namespace range_of_2a_minus_b_l165_165823

theorem range_of_2a_minus_b (a b : ℝ) (h1 : a > b) (h2 : 2 * a^2 - a * b - b^2 - 4 = 0) :
  (2 * a - b) ∈ (Set.Ici (8 / 3)) :=
sorry

end range_of_2a_minus_b_l165_165823


namespace find_c_l165_165473

theorem find_c (c : ℕ) (h : 111111222222 = c * (c + 1)) : c = 333333 :=
by
  -- proof goes here
  sorry

end find_c_l165_165473


namespace sum_divisors_12_eq_28_l165_165366

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165366


namespace inequality_sqrt_l165_165059

theorem inequality_sqrt (m n : ℕ) (h : m < n) : 
  (m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n)) :=
by
  sorry

end inequality_sqrt_l165_165059


namespace sum_of_fourth_powers_less_than_1000_l165_165189

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165189


namespace area_hexagon_STUVWX_l165_165907

noncomputable def area_of_hexagon (area_PQR : ℕ) (small_area : ℕ) : ℕ := 
  area_PQR - (3 * small_area)

theorem area_hexagon_STUVWX : 
  let area_PQR := 45
  let small_area := 1 
  ∃ area_hexagon, area_hexagon = 42 := 
by
  let area_PQR := 45
  let small_area := 1
  let area_hexagon := area_of_hexagon area_PQR small_area
  use area_hexagon
  sorry

end area_hexagon_STUVWX_l165_165907


namespace remaining_budget_l165_165228

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l165_165228


namespace min_indecent_syllables_l165_165534

theorem min_indecent_syllables (n : ℕ) : 
  ∃ k, (∀ (w : List (Fin n)), (∀ i j, (i ≤ j) → (w.mem (i, j) → False)) → w.length < n.succ) ∧ k = (n * (n + 1)) / 2 :=
sorry

end min_indecent_syllables_l165_165534


namespace total_stickers_at_end_of_week_l165_165601

-- Defining the initial and earned stickers as constants
def initial_stickers : ℕ := 39
def earned_stickers : ℕ := 22

-- Defining the goal as a proof statement
theorem total_stickers_at_end_of_week : initial_stickers + earned_stickers = 61 := 
by {
  sorry
}

end total_stickers_at_end_of_week_l165_165601


namespace slope_of_line_through_midpoints_l165_165767

open Real

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
(x1, y1) = p1
(x2, y2) = p2
((x1 + x2) / 2, (y1 + y2) / 2)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
(x1, y1) = p1
(x2, y2) = p2
(y2 - y1) / (x2 - x1)

theorem slope_of_line_through_midpoints :
  let mp1 := midpoint (1, 2) (3, 5) in
  let mp2 := midpoint (4, 2) (7, 6) in
  slope mp1 mp2 = 1 / 7 :=
by
  let mp1 := midpoint (1, 2) (3, 5)
  let mp2 := midpoint (4, 2) (7, 6)
  have h1 : mp1 = (2, 3.5) := by
    simp [midpoint]
  have h2 : mp2 = (5.5, 4) := by
    simp [midpoint]
  have h3 : slope mp1 mp2 = 1 / 7 := by
    simp [slope, h1, h2]
  exact h3

end slope_of_line_through_midpoints_l165_165767


namespace sum_of_divisors_of_12_l165_165415

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165415


namespace sum_of_divisors_of_12_l165_165419

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165419


namespace people_visited_on_Sunday_l165_165735

theorem people_visited_on_Sunday (ticket_price : ℕ) 
                                 (people_per_day_week : ℕ) 
                                 (people_on_Saturday : ℕ) 
                                 (total_revenue : ℕ) 
                                 (days_week : ℕ)
                                 (total_days : ℕ) 
                                 (people_per_day_mf : ℕ) 
                                 (people_on_other_days : ℕ) 
                                 (revenue_other_days : ℕ)
                                 (revenue_Sunday : ℕ)
                                 (people_Sunday : ℕ) :
    ticket_price = 3 →
    people_per_day_week = 100 →
    people_on_Saturday = 200 →
    total_revenue = 3000 →
    days_week = 5 →
    total_days = 7 →
    people_per_day_mf = people_per_day_week * days_week →
    people_on_other_days = people_per_day_mf + people_on_Saturday →
    revenue_other_days = people_on_other_days * ticket_price →
    revenue_Sunday = total_revenue - revenue_other_days →
    people_Sunday = revenue_Sunday / ticket_price →
    people_Sunday = 300 := 
by 
  sorry

end people_visited_on_Sunday_l165_165735


namespace sum_divisors_of_12_l165_165398

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165398


namespace sum_factorials_last_two_digits_l165_165676

theorem sum_factorials_last_two_digits :
  (∑ n in Finset.range 10, n.factorial) % 100 = 13 :=
by
  sorry

end sum_factorials_last_two_digits_l165_165676


namespace conjugate_of_z_l165_165087

-- Define the given complex number
def z : ℂ := 2 / (1 - I)

-- The main statement to be proven
theorem conjugate_of_z : conj z = 1 - I := by 
  sorry

end conjugate_of_z_l165_165087


namespace swimming_upstream_distance_l165_165722

noncomputable def speed_in_still_water : ℝ := 7
noncomputable def distance_downstream : ℝ := 40
noncomputable def time_downstream : ℝ := 5
noncomputable def time_upstream : ℝ := 5

theorem swimming_upstream_distance :
  ∃ (v : ℝ), let effective_speed_upstream := speed_in_still_water - v in
  (speed_in_still_water + v) * time_downstream = distance_downstream ∧
  effective_speed_upstream * time_upstream = 30 :=
by
  sorry

end swimming_upstream_distance_l165_165722


namespace sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165151

def is_square_of_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (k^2)^2

def sum_of_squares_of_perfect_squares : ℕ :=
  (Finset.range 1000).filter is_square_of_perfect_square |>.sum id

theorem sum_of_all_squares_of_perfect_squares_below_1000_eq_979 :
  sum_of_squares_of_perfect_squares = 979 :=
by
  sorry

end sum_of_all_squares_of_perfect_squares_below_1000_eq_979_l165_165151


namespace decimal_multiplication_l165_165886

noncomputable def repeating_decimal : ℝ := 3.131 / 9999

theorem decimal_multiplication :
  (10^5 - 10^3) * (0.000313131...) = 309.969 :=
by
  have h1 : \((10^5 - 10^3) = 99000\), by sorry
  have h2 : (0.000313131...) = (repeating_decimal / 10^4), by sorry
  have h3 : (99000 * (repeating_decimal / 10^4)) = 309.969, by sorry
  exact h3

#print axioms decimal_multiplication

end decimal_multiplication_l165_165886


namespace min_treasure_signs_buried_l165_165280

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165280


namespace product_less_than_40_l165_165962

def probability_less_than_40 : ℚ :=
  let prob := (6 * 8).toRat                 -- Total equally likely outcomes
  let successful_outcomes := (5 * 8) + 6    -- 5 full sets of 8 successful outcomes, plus 6 successful outcomes for Paco's spin of 6
  let probability := successful_outcomes / prob
  probability

theorem product_less_than_40 :
  probability_less_than_40 = 23 / 24 := 
by
  -- Proof logic here
  sorry

end product_less_than_40_l165_165962


namespace exists_small_area_triangle_l165_165540

structure LatticePoint where
  x : Int
  y : Int

def isValidPoint (p : LatticePoint) : Prop := 
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

def noThreeCollinear (points : List LatticePoint) : Prop := 
  ∀ (p1 p2 p3 : LatticePoint), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ((p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y))

def triangleArea (p1 p2 p3 : LatticePoint) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) : ℝ)|

theorem exists_small_area_triangle
  (points : List LatticePoint)
  (h1 : ∀ p ∈ points, isValidPoint p)
  (h2 : noThreeCollinear points) :
  ∃ (p1 p2 p3 : LatticePoint), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l165_165540


namespace pn_value_2_pn_value_3_pn_minus_qn_geometric_pn_value_l165_165527

noncomputable def p (n : ℕ) : ℝ := sorry
noncomputable def q (n : ℕ) : ℝ := sorry

axiom pn_qn_recursion (n : ℕ) (hn : n ≥ 2) :
  p n = p (n - 1) * (1/6) + q (n - 1) * (5/6)
  ∧ q n = q (n - 1) * (1/6) + p (n - 1) * (5/6)

axiom initial_conditions : p 1 = 1 ∧ q 1 = 0

theorem pn_value_2 :
  p 2 = 1/6 := sorry

theorem pn_value_3 :
  p 3 = 26/36 := sorry

theorem pn_minus_qn_geometric (n : ℕ) (hn : n ≥ 2) :
  (p n - q n) = (-2/3)^(n-1) := sorry

theorem pn_value (n : ℕ) (hn : n ≥ 1) :
  p n = (1/2) * ((-2/3)^(n-1) + 1) := sorry

end pn_value_2_pn_value_3_pn_minus_qn_geometric_pn_value_l165_165527


namespace roots_of_equation_l165_165101

theorem roots_of_equation :
  (∃ (x_1 x_2 : ℝ), x_1 > x_2 ∧ (∀ x, x^2 - |x-1| - 1 = 0 ↔ x = x_1 ∨ x = x_2)) :=
sorry

end roots_of_equation_l165_165101


namespace required_run_rate_l165_165202

theorem required_run_rate
  (run_rate_first_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_first : ℕ)
  (overs_remaining : ℕ)
  (H_run_rate_10_overs : run_rate_first_10_overs = 3.2)
  (H_target_runs : target_runs = 222)
  (H_overs_first : overs_first = 10)
  (H_overs_remaining : overs_remaining = 40) :
  ((target_runs - run_rate_first_10_overs * overs_first) / overs_remaining) = 4.75 := 
by
  sorry

end required_run_rate_l165_165202


namespace max_diff_a2018_a2017_l165_165021

theorem max_diff_a2018_a2017 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 0 ∧
    a 1 = 1 ∧
    (∀ n ≥ 2, ∃ k, 1 ≤ k ∧ k ≤ n ∧ a n = (∑ i in (finset.range k).map (λ i, a (n - 1 - i))) / k) ∧
    ∃ M, M = (2016 / 2017^2) ∧ (a 2018 - a 2017) ≤ M :=
begin
  sorry
end

end max_diff_a2018_a2017_l165_165021


namespace john_will_always_win_l165_165922

-- Given conditions in Lean definitions
def interval (m : ℤ) (h : m > 5) : set ℝ := { x | 1 ≤ x ∧ x ≤ m }

def valid_choice (chosen : set ℝ) (x : ℝ) : Prop :=
  ∀ y ∈ chosen, abs (x - y) > 1

def player_wins (player : string) : Prop := 
  ∀ (m : ℤ) (h : m > 5), 
    let interval := interval m h in
    -- Assuming optimal strategies
    ∃ (j_chosen : set ℝ), 
    (∀ x ∈ interval, x ∈ j_chosen → valid_choice j_chosen x) ∧
    (∀ x ∈ interval, valid_choice j_chosen x → (∃ y ∈ interval, valid_choice (j_chosen ∪ { x }) y)) ∧
    player = "John"

theorem john_will_always_win : player_wins "John" := by
  sorry

end john_will_always_win_l165_165922


namespace construct_n_gon_from_points_and_angles_l165_165759

-- Define the problem with given conditions and show your existence proof.
theorem construct_n_gon_from_points_and_angles 
{n : ℕ} (M : Fin n → Point) (α : Fin n → ℝ) : 
  ∀ (A : Fin n → Point), 
  (∀ i : Fin n, 
    isosceles_triangle (A i) (M i) (A (i+1) % n) ∧
    angle (A i) (M i) (A (i+1) % n) = α i ) →
  (∑ i in Finset.range n, α i ≠ (k : ℤ) * 360) →
  ∃! (A : Fin n → Point), 
    ∀ i : Fin n, 
    isosceles_triangle (A i) (M i) (A (i+1) % n) ∧
    angle (A i) (M i) (A (i+1) % n) = α i := 
sorry

end construct_n_gon_from_points_and_angles_l165_165759


namespace cos_B_eq_side_c_eq_side_b_eq_l165_165547

noncomputable def given_triangle : Type := {
  a : ℝ,
  sinB2 : ℝ,
  area : ℝ
}

def triangle_conditions (t : given_triangle) : Prop :=
  t.a = 2 ∧ t.sinB2 = sqrt(5) / 5 ∧ t.area = 4

theorem cos_B_eq (t : given_triangle) (hc : triangle_conditions t) : cos(B) = 3 / 5 :=
  sorry

theorem side_c_eq (t : given_triangle) (hc : triangle_conditions t) : c = 5 :=
  sorry

theorem side_b_eq (t : given_triangle) (hc : triangle_conditions t) : b = sqrt(17) :=
  sorry

end cos_B_eq_side_c_eq_side_b_eq_l165_165547


namespace minimum_treasures_count_l165_165309

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165309


namespace total_quartet_songs_l165_165791

/-- 
Five girls — Mary, Alina, Tina, Hanna, and Elsa — sang songs in a concert as quartets,
with one girl sitting out each time. Hanna sang 9 songs, which was more than any other girl,
and Mary sang 3 songs, which was fewer than any other girl. If the total number of songs
sung by Alina and Tina together was 16, then the total number of songs sung by these quartets is 8. -/
theorem total_quartet_songs
  (hanna_songs : ℕ) (mary_songs : ℕ) (alina_tina_songs : ℕ) (total_songs : ℕ)
  (h_hanna : hanna_songs = 9)
  (h_mary : mary_songs = 3)
  (h_alina_tina : alina_tina_songs = 16) :
  total_songs = 8 :=
sorry

end total_quartet_songs_l165_165791


namespace tulip_ratio_l165_165260

theorem tulip_ratio (total_tulips eyes smile : ℕ) (tulips_per_eye smile_tulips : ℕ) (total_red_tulips : total_red_tulips = 2 * tulips_per_eye + smile_tulips) (total_yellow_tulips : total_yellow_tulips = total_tulips - total_red_tulips) : 
  total_tulips = 196 ->
  tulips_per_eye = 8 ->
  smile_tulips = 18 ->
  total_red_tulips = 2 * tulips_per_eye + smile_tulips ->
  total_yellow_tulips = total_tulips - total_red_tulips ->
  ratio (total_yellow_tulips / smile_tulips) 9 := 
by
  intros
  sorry

end tulip_ratio_l165_165260


namespace woman_weaves_amount_on_20th_day_l165_165905

theorem woman_weaves_amount_on_20th_day
  (a d : ℚ)
  (a2 : a + d = 17) -- second-day weaving in inches
  (S15 : 15 * a + 105 * d = 720) -- total for the first 15 days in inches
  : a + 19 * d = 108 := -- weaving on the twentieth day in inches (9 feet)
by
  sorry

end woman_weaves_amount_on_20th_day_l165_165905


namespace range_of_g_l165_165766

-- Definition of the function
def g (x : ℝ) : ℝ := (Real.sin x)^4 + 2 * (Real.sin x)^2 * (Real.cos x)^2 + 4 * (Real.cos x)^4

-- Theorem stating the range of g(x)
theorem range_of_g : (∀ x : ℝ, 4 ≤ g x ∧ g x ≤ 4.5) :=
by
  sorry

end range_of_g_l165_165766


namespace circumradius_of_triangle_l165_165010

-- Given variables and conditions
variables (a b c R : ℝ)
variables (A B C : ℝ)
variables (BC CA AB : ℝ)

-- Condition definitions
def sides := (a = BC ∧ b = CA ∧ c = AB)
def sum_of_squares := (a^2 + b^2 = 6)
def cosine_condition := (cos (A - B) * cos C = 2 / 3)

-- The proof statement
theorem circumradius_of_triangle :
    sides a b c BC CA AB →
    sum_of_squares a b →
    cosine_condition A B C →
    R = 3 * sqrt 10 / 10 :=
    by
    intros h_sides h_sum_squares h_cosine_condition
    sorry

end circumradius_of_triangle_l165_165010


namespace treasure_under_minimum_signs_l165_165279

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l165_165279


namespace martian_amoeba_function_l165_165758

-- Define a datatype for Martian Amoebas
inductive MartianAmoeba
| A
| B
| C

open MartianAmoeba

-- Define the function f such that the conditions are represented in Lean
def f : MartianAmoeba → Nat
| A => 3
| B => 5
| C => 6

-- Prove the function satisfies the conditions
theorem martian_amoeba_function : 
  f A ⊕ f B = f C ∧ f A ⊕ f C = f B ∧ f B ⊕ f C = f A := by
  -- Prove the first condition
  have h1 : f A ⊕ f B = f C := by
    simp [f, Nat.bitwiseXor]
    sorry
  -- Prove the second condition
  have h2 : f A ⊕ f C = f B := by
    simp [f, Nat.bitwiseXor]
    sorry
  -- Prove the third condition
  have h3 : f B ⊕ f C = f A := by
    simp [f, Nat.bitwiseXor]
    sorry
  -- Conclude the theorem
  exact ⟨h1, h2, h3⟩

end martian_amoeba_function_l165_165758


namespace minimum_treasures_count_l165_165306

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165306


namespace proof_max_k_for_no_roots_l165_165256
noncomputable def max_k_for_no_roots (a : ℕ → ℕ) : ℕ :=
  let k := 88
  if (∀ i : ℕ, i < k - 2 → (a (i+1))^2 < a i * a (i+2)) ∧ 
     (∀ i j : ℕ, i < j → a i ≠ a j) ∧ 
     (a (k-1) - a 0 < 1000) then k else
  sorry

theorem proof_max_k_for_no_roots :
  ∃ k, k = max_k_for_no_roots (λ n, if n < 44 then 10000 + (44-n) * (45-n) / 2 else 10000 + (n-44) * (n-43) / 2) := 
begin
  use 88,
  sorry
end

end proof_max_k_for_no_roots_l165_165256


namespace sum_of_fourth_powers_below_1000_l165_165159

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165159


namespace friends_ride_roller_coaster_l165_165646

theorem friends_ride_roller_coaster :
    ∀ (tickets_per_ride total_tickets : ℕ), tickets_per_ride = 6 → total_tickets = 48 → total_tickets / tickets_per_ride = 8 :=
by
  intro tickets_per_ride total_tickets h1 h2
  rw [h1, h2]
  norm_num
  exact rfl

end friends_ride_roller_coaster_l165_165646


namespace num_non_congruent_triangles_with_perimeter_9_l165_165873

theorem num_non_congruent_triangles_with_perimeter_9 :
  { (a, b, c) : ℕ × ℕ × ℕ // a ≤ b ∧ b ≤ c ∧ a + b + c = 9 ∧ a + b > c ∧ a + c > b ∧ b + c > a }.to_finset.card = 2 :=
begin
  sorry
end

end num_non_congruent_triangles_with_perimeter_9_l165_165873


namespace plane_volume_division_ratio_l165_165981

-- Definitions from the conditions
structure Parallelogram (A B C D : Type) := 
(V : ℝ) -- Volume of the pyramid

def midpoint (A P : Type) : Type := 
(K : Type)-- Midpoint of A and P

def cn_np_ratio (C P : Type) := 
(N : Type) -- such that CN : NP = 1 : 3

def bm_bc_ratio (B C : Type) := 
(M : Type) -- such that BM = 2 BC

-- The proof problem in Lean 4 statement
theorem plane_volume_division_ratio (P A B C D K N M: Type) 
  [Parallelogram A B C D] 
  [midpoint A P] 
  [cn_np_ratio C P] 
  [bm_bc_ratio B C] : 
  -- Statement:
  ∀ (V : ℝ), divides_volume P A B C D K M N (405 / 1672) :=
by
  sorry

end plane_volume_division_ratio_l165_165981


namespace cyclic_quadrilateral_l165_165566

-- Definitions 
noncomputable def is_cyclic (A B C D : Type) : Prop :=
sorry  -- This should define cyclic quadrilateral property

variables {A B C D P O B' D' : Type} 
variables (AB CD OP : Type) 
-- Conditions
variable (h1 : P = AB ∩ CD)
variable (h2 : O = intersection_of_perpendicular_bisectors AB CD)
variable (h3 : ¬ (O ∈ AB))
variable (h4 : ¬ (O ∈ CD))
variable (h5 : B' = reflection B OP)
variable (h6 : D' = reflection D OP)
variable (h7 : (AB' ∩ CD') ∈ OP)

-- Proof statement
theorem cyclic_quadrilateral : is_cyclic A B C D :=
sorry

end cyclic_quadrilateral_l165_165566


namespace smallest_positive_int_linear_combination_l165_165683

theorem smallest_positive_int_linear_combination (m n : ℤ) :
  ∃ k : ℤ, 4509 * m + 27981 * n = k ∧ k > 0 ∧ k ≤ 4509 * m + 27981 * n → k = 3 :=
by
  sorry

end smallest_positive_int_linear_combination_l165_165683


namespace common_ratio_geometric_sequence_l165_165677

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

open Real

-- Statement
theorem common_ratio_geometric_sequence (a : ℝ) :
  let t1 := a + log_base 2 3
      t2 := a + log_base 4 3
      t3 := a + log_base 8 3
  in (t2 / t1 = t3 / t2) → (t2 / t1 = 1 / 3) :=
by {
  intros h,
  sorry
}

end common_ratio_geometric_sequence_l165_165677


namespace xy_fraction_equivalence_l165_165623

theorem xy_fraction_equivalence
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (x^2 + 4 * x * y) / (y^2 - 4 * x * y) = 3) :
  (x^2 - 4 * x * y) / (y^2 + 4 * x * y) = -1 :=
sorry

end xy_fraction_equivalence_l165_165623


namespace treasure_15_signs_l165_165297

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165297


namespace solution_set_of_inequality_l165_165570

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end solution_set_of_inequality_l165_165570


namespace circle_areas_l165_165215

-- Define the side length of the equilateral triangle
def side_length : ℝ := 12

-- Define the radius of the circumscribed circle
def circumscribed_radius (s : ℝ) : ℝ := s / Real.sqrt 3

-- Define the area of the circumscribed circle
def circumscribed_area (s : ℝ) : ℝ := Real.pi * (circumscribed_radius s)^2

-- Define the radius of the inscribed circle
def inscribed_radius (s : ℝ) : ℝ := (Real.sqrt 3 / 6) * s

-- Define the area of the inscribed circle
def inscribed_area (s : ℝ) : ℝ := Real.pi * (inscribed_radius s)^2

-- Prove the areas are as expected
theorem circle_areas (s : ℝ) (h : s = 12) : circumscribed_area s = 48 * Real.pi ∧ inscribed_area s = 12 * Real.pi :=
by
  rw [h]
  dsimp [circumscribed_area, inscribed_area, circumscribed_radius, inscribed_radius]
  norm_num
  split; norm_num
  sorry

end circle_areas_l165_165215


namespace real_number_c_l165_165652

theorem real_number_c (x1 x2 c : ℝ) (h_eqn : x1 + x2 = -1) (h_prod : x1 * x2 = c) (h_cond : x1^2 * x2 + x2^2 * x1 = 3) : c = -3 :=
by sorry

end real_number_c_l165_165652


namespace jonessa_total_pay_l165_165629

theorem jonessa_total_pay (total_pay : ℝ) (take_home_pay : ℝ) (h1 : take_home_pay = 450) (h2 : 0.90 * total_pay = take_home_pay) : total_pay = 500 :=
by
  sorry

end jonessa_total_pay_l165_165629


namespace minimum_treasures_count_l165_165305

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l165_165305


namespace pow_mul_eq_add_l165_165746

theorem pow_mul_eq_add (a : ℝ) : a^3 * a^4 = a^7 :=
by
  -- This is where the proof would go.
  sorry

end pow_mul_eq_add_l165_165746


namespace monotone_interval_a_l165_165520

noncomputable def f (a x : ℝ) : ℝ := (a - sin x) / cos x

theorem monotone_interval_a (a : ℝ) :
  (∀ x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 3), 0 ≤ f' a x) ↔ 2 ≤ a := by
  sorry

end monotone_interval_a_l165_165520


namespace min_treasure_count_l165_165316

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165316


namespace minimum_RS_distance_l165_165930

variables {A B C D O M : Type} [HilbertPlane A B C D]
variable [rhombus : IsRhombus A B C D]
variables (AC BD : ℝ) (R S : A)

-- Given conditions
axioms  
  (h1 : AC = 20)
  (h2 : BD = 24)
  (h3 : ∃ (M : A), ∃ (R S: A), R ⊥ AC ∧ S ⊥ BD ∧ line_through M R ∧ line_through M S)

-- Define the function to be minimized
def RS_distance (M R S : A) : ℝ := distance R S

-- Define the minimum distance problem
theorem minimum_RS_distance : 
  (∃ (M : A) (R S: A), R ⊥ AC ∧ S ⊥ BD ∧ RS_distance M R S = 9.944) :=
sorry

end minimum_RS_distance_l165_165930


namespace negation_of_existence_implication_l165_165097

theorem negation_of_existence_implication :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ ln x₀ > 3 - x₀) ↔ ∀ x : ℝ, x > 0 → ln x ≤ 3 - x :=
by sorry

end negation_of_existence_implication_l165_165097


namespace min_treasures_buried_l165_165322

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165322


namespace num_bad_oranges_l165_165526

theorem num_bad_oranges (G B : ℕ) (hG : G = 24) (ratio : G / B = 3) : B = 8 :=
by
  sorry

end num_bad_oranges_l165_165526


namespace sum_of_divisors_of_12_l165_165439

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165439


namespace treasure_15_signs_l165_165302

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l165_165302


namespace expand_and_simplify_l165_165775

theorem expand_and_simplify :
  ∀ x : ℝ, (x^3 - 3*x + 3)*(x^2 + 3*x + 3) = x^5 + 3*x^4 - 6*x^2 + 9 := by sorry

end expand_and_simplify_l165_165775


namespace cannot_be_combined_with_sqrt2_l165_165689

def can_be_combined (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y

theorem cannot_be_combined_with_sqrt2 :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 8
  let c := Real.sqrt 12
  let d := -Real.sqrt 18
  ¬ can_be_combined c (Real.sqrt 2) := 
by
  sorry

end cannot_be_combined_with_sqrt2_l165_165689


namespace no_positive_numbers_satisfy_conditions_l165_165343

theorem no_positive_numbers_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = ab + ac + bc) ∧ (ab + ac + bc = abc) :=
by
  sorry

end no_positive_numbers_satisfy_conditions_l165_165343


namespace rationalize_denominator_l165_165612

-- Define conditions
def fraction : ℚ := 5 / 12
def sqrt12_simplified : ℝ := 2 * Real.sqrt 3
def rationalized (x : ℝ) : Prop := x = Real.sqrt 15 / 6

-- Define the problem statement
theorem rationalize_denominator : 
  rationalized (Real.sqrt (5 / 12)) := 
by 
  have h1 : Real.sqrt (5 / 12) = Real.sqrt 5 / sqrt12_simplified,
  sorry

end rationalize_denominator_l165_165612


namespace new_year_starts_more_often_on_sunday_l165_165201

theorem new_year_starts_more_often_on_sunday :
  (count_new_year_start_day_over_400_years sunday) > (count_new_year_start_day_over_400_years saturday) := sorry

end new_year_starts_more_often_on_sunday_l165_165201


namespace coefficient_and_constant_term_l165_165001

theorem coefficient_and_constant_term (x : ℝ) :
  let binom (n k : ℕ) := Nat.choose n k in
  (coeff_of_x3 : ℝ) = binom 50 3 * 2 ^ 47 ∧
  (constant_term : ℝ) = 2 ^ 50 :=
by
  -- Coefficient of x^3 term
  let coeff_of_x3 := binom 50 3 * 2 ^ 47
  -- Constant term
  let constant_term := 2 ^ 50
  sorry

end coefficient_and_constant_term_l165_165001


namespace sum_of_fourth_powers_le_1000_l165_165145

-- Define the fourth powers less than 1000
def fourth_powers_le_1000 := {n : ℕ | ∃ k : ℕ, k^4 = n ∧ n < 1000}

-- Define the sum of these fourth powers
def sum_fourth_powers : ℕ := ∑ n in fourth_powers_le_1000, n

theorem sum_of_fourth_powers_le_1000 :
  sum_fourth_powers = 979 :=
by
  sorry

end sum_of_fourth_powers_le_1000_l165_165145


namespace ellipse_eq_line_eq_l165_165337

-- Conditions for part (I)
def cond1 (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a > b
def pt_p_cond (PF1 PF2 : ℝ) : Prop := PF1 = 4 / 3 ∧ PF2 = 14 / 3 ∧ PF1^2 + PF2^2 = 1

-- Theorem for part (I)
theorem ellipse_eq (a b : ℝ) (PF1 PF2 : ℝ) (h₁ : cond1 a b) (h₂ : pt_p_cond PF1 PF2) : 
  (a = 3 ∧ b = 2 ∧ PF1 = 4 / 3 ∧ PF2 = 14 / 3) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Conditions for part (II)
def center_circle (M : ℝ × ℝ) : Prop := M = (-2, 1)
def pts_symmetric (A B M : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

-- Theorem for part (II)
theorem line_eq (A B M : ℝ × ℝ) (k : ℝ) (h₁ : center_circle M) (h₂ : pts_symmetric A B M) :
  k = 8 / 9 → (∀ x y : ℝ, 8 * x - 9 * y + 25 = 0) :=
sorry

end ellipse_eq_line_eq_l165_165337


namespace overall_profit_percentage_l165_165627

-- Definitions for cost prices (conditions from a)
def cost_price_A (A_s : ℝ) : ℝ := 0.9 * A_s
def cost_price_B (B_s : ℝ) : ℝ := 0.9 * B_s
def cost_price_C (C_s : ℝ) : ℝ := 0.9 * C_s

-- Statement of the theorem to prove
theorem overall_profit_percentage (A_s B_s C_s : ℝ) :
  let A_c := cost_price_A A_s in
  let B_c := cost_price_B B_s in
  let C_c := cost_price_C C_s in
  ((0.1 * (A_s + B_s + C_s)) / (0.9 * (A_s + B_s + C_s)) * 100) = 11.11 :=
by
  sorry

end overall_profit_percentage_l165_165627


namespace find_a_l165_165468

theorem find_a (a : ℝ) (A : set ℝ) (B : set ℝ) (hA : A = {1, 2}) (hB : B = {a, a^2 - 1}) (hInter : A ∩ B = {1}) :
  a = 1 ∨ a = real.sqrt 2 ∨ a = -real.sqrt 2 := 
sorry

end find_a_l165_165468


namespace graph_single_point_l165_165626

theorem graph_single_point (c : ℝ) : 
  (∃ x y : ℝ, ∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + 16 * x' - 6 * y' + c = 0 → (x' = x ∧ y' = y)) → c = 7 := 
by
  sorry

end graph_single_point_l165_165626


namespace total_travel_time_proof_l165_165246

def total_travel_time (d1 d2 r1 r2 : ℝ) : ℝ := (d1 / r1) + (d2 / r2)

theorem total_travel_time_proof :
  total_travel_time 120 120 40 49.99999999999999 = 5.4 :=
by 
  rw [total_travel_time, div_eq_mul_inv, div_eq_mul_inv, mul_inv, mul_inv, add]
  norm_num



end total_travel_time_proof_l165_165246


namespace months_to_save_l165_165092

/-- The grandfather saves 530 yuan from his pension every month. -/
def savings_per_month : ℕ := 530

/-- The price of the smartphone is 2000 yuan. -/
def smartphone_price : ℕ := 2000

/-- The number of months needed to save enough money to buy the smartphone. -/
def months_needed : ℕ := smartphone_price / savings_per_month

/-- Proof that the number of months needed is 4. -/
theorem months_to_save : months_needed = 4 :=
by
  sorry

end months_to_save_l165_165092


namespace eval_polynomial_at_neg2_l165_165771

-- Define the polynomial function
def polynomial (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

-- Statement of the problem, proving that the polynomial equals 11 when x = -2
theorem eval_polynomial_at_neg2 : polynomial (-2) = 11 := by
  sorry

end eval_polynomial_at_neg2_l165_165771


namespace sum_divisors_of_12_l165_165404

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165404


namespace Roselyn_initial_books_l165_165614

theorem Roselyn_initial_books :
  ∀ (books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books : ℕ),
    books_given_to_Rebecca = 40 →
    books_remaining = 60 →
    books_given_to_Mara = 3 * books_given_to_Rebecca →
    total_books_given = books_given_to_Mara + books_given_to_Rebecca →
    initial_books = books_remaining + total_books_given →
    initial_books = 220 :=
by
  intros books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Roselyn_initial_books_l165_165614


namespace common_tangents_count_l165_165541

open Real

def point (x y : ℝ) := ℝ × ℝ

-- Define points A and B
def A : point := (1, 2)
def B : point := (3, 1)

-- Define the distance function
def distance (p q : point) : ℝ := sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the circles around A and B
def circle (center : point) (radius : ℝ) := { p : point | distance p center = radius }

-- Define the lines at distances of 1 and 2 from points A and B respectively
def lines_distance_from_point (p : point) (d : ℝ) := { l : set point | ∀ q ∈ l, distance q p = d }

-- Define the common tangents condition
def common_tangents (l1 l2 : set point → Prop) := ∃ l : set point, l1 l ∧ l2 l

-- State the proof problem
theorem common_tangents_count :
  let l1 := lines_distance_from_point A 1 in
  let l2 := lines_distance_from_point B 2 in
  ∃ l : set point, common_tangents l1 l2 ∧ (l : set point).card = 2 :=
sorry

end common_tangents_count_l165_165541


namespace shirt_cost_l165_165619

variables (S : ℝ)

theorem shirt_cost (h : 2 * S + (S + 3) + (1/2) * (2 * S + S + 3) = 36) : S = 7.88 :=
sorry

end shirt_cost_l165_165619


namespace f_triple_application_l165_165941

-- Define the function f according to the conditions given
def f (x : ℝ) : ℝ :=
  if x < 10 then x^2 - 1
  else x - 20

-- State the main theorem to prove
theorem f_triple_application : f (f (f 25)) = 4 := 
  sorry

end f_triple_application_l165_165941


namespace lawrence_walked_distance_l165_165564

theorem lawrence_walked_distance :
  ∀ (speed time distance : ℝ), 
    speed = 3 ∧ time = 1.33 ∧ distance = speed * time → 
    distance = 3.99 := 
by
  intros speed time distance h
  cases' h with hs ht
  cases' ht with ht hd
  rw [hs, ht] at hd
  assumption

end lawrence_walked_distance_l165_165564


namespace part1_l165_165593

def f (x : ℝ) := x^2 - 2*x

theorem part1 (x : ℝ) :
  (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ 3 ≤ x ∨ x = 0) :=
sorry

end part1_l165_165593


namespace Nancy_shelved_biographies_l165_165951

def NancyBooks.shelved_books_from_top : Nat := 12 + 8 + 4 -- history + romance + poetry
def NancyBooks.total_books_on_cart : Nat := 46
def NancyBooks.bottom_books_after_top_shelved : Nat := 46 - 24
def NancyBooks.mystery_books_on_bottom : Nat := NancyBooks.bottom_books_after_top_shelved / 2
def NancyBooks.western_novels_on_bottom : Nat := 5
def NancyBooks.biographies : Nat := NancyBooks.bottom_books_after_top_shelved - NancyBooks.mystery_books_on_bottom - NancyBooks.western_novels_on_bottom

theorem Nancy_shelved_biographies : NancyBooks.biographies = 6 := by
  sorry

end Nancy_shelved_biographies_l165_165951


namespace smallest_angle_of_quadrilateral_l165_165998

theorem smallest_angle_of_quadrilateral 
  (x : ℝ) 
  (h1 : x + 2 * x + 3 * x + 4 * x = 360) : 
  x = 36 :=
by
  sorry

end smallest_angle_of_quadrilateral_l165_165998


namespace solve_sqrt_equation_l165_165975

theorem solve_sqrt_equation (m : ℝ) :
  ∃ x : ℝ, (sqrt (x + 1) - sqrt (2 * x + 1) = m) ↔ (m ∈ Set.Iio (Real.sqrt 2 / 2) ∨ (m = Real.sqrt 2 / 2 ∧ x = -1 / 2)) :=
by
  sorry

end solve_sqrt_equation_l165_165975


namespace hyperbola_equations_l165_165829

def eq1 (x y : ℝ) : Prop := x^2 - 4 * y^2 = (5 + Real.sqrt 6)^2
def eq2 (x y : ℝ) : Prop := 4 * y^2 - x^2 = 4

theorem hyperbola_equations 
  (x y : ℝ)
  (hx1 : x - 2 * y = 0)
  (hx2 : x + 2 * y = 0)
  (dist : Real.sqrt ((x - 5)^2 + y^2) = Real.sqrt 6) :
  eq1 x y ∧ eq2 x y := 
sorry

end hyperbola_equations_l165_165829


namespace borel_cantelli_variant_l165_165063

variables {Ω : Type*} {A : ℕ → Set Ω} {P : MeasureTheory.Measure Ω}

-- Defining the given conditions
def condition1 : Prop := ∀ ε > 0, ∃ N, ∀ n ≥ N, P (A n) < ε
def condition2 : Prop := ∑' n, P (A n \ A (n + 1)) < ∞

-- The main theorem statement
theorem borel_cantelli_variant (h1 : condition1) (h2 : condition2) :
  P (Set.Union (λ n, Set.Inter (Set.Icc n (∞)) A)) = 0 :=
sorry

end borel_cantelli_variant_l165_165063


namespace complex_magnitude_l165_165799

theorem complex_magnitude (z : ℂ) (h : (1 + complex.I) * z = 3 + complex.I) : complex.abs z = real.sqrt 5 :=
sorry

end complex_magnitude_l165_165799


namespace mean_of_random_error_is_zero_l165_165117

-- Definition of the problem conditions
variables {x y e : ℝ} {b a : ℝ}

-- The linear regression model condition
axiom linear_regression_model (hx : x) (hy : y) (he : e) : y = b * x + a + e

-- The random error expectation condition
axiom random_error_is_mean_zero : ∀ (e : ℝ), e = y - b * x - a → E e = 0

-- The theorem to be proven
theorem mean_of_random_error_is_zero : E e = 0 :=
by sorry

end mean_of_random_error_is_zero_l165_165117


namespace length_DE_l165_165588

-- Define the quadrilateral inscribed in a circle with diameter AC
def is_cyclic_quadrilateral (A B C D : Type) : Prop :=
  ∃ (O : Type), (circle_circumference O = 2 * π * radius O) ∧ (radius O = (distance A C) / 2)
  ∧ (on_the_circle A O) ∧ (on_the_circle B O) ∧ (on_the_circle C O) ∧ (on_the_circle D O)

-- Define the properties of the quadrilateral ABCD
variables {A B C D E : Type} 

-- Given the conditions
axiom eq_distance : distance A D = distance D C 
axiom area_ABCD : area_quadrilateral A B C D = 24
axiom is_perpendicular : is_perpendicular (line D E) (line A B)

-- Prove that DE = 2√6
theorem length_DE : 
  is_cyclic_quadrilateral A B C D →
  eq_distance →
  area_ABCD →
  is_perpendicular →
  distance D E = 2 * sqrt 6 :=
by
  sorry

end length_DE_l165_165588


namespace sum_of_squares_of_perfect_squares_l165_165127

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165127


namespace class_total_students_l165_165212

-- Definitions based on the conditions
def number_students_group : ℕ := 12
def frequency_group : ℚ := 0.25

-- Statement of the problem in Lean
theorem class_total_students (n : ℕ) (h : frequency_group = number_students_group / n) : n = 48 :=
by
  sorry

end class_total_students_l165_165212


namespace max_value_f_on_domain_l165_165495

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x - 1)

theorem max_value_f_on_domain : 
  (∀ x ∈ set.Ico (-8 : ℝ) (-4 : ℝ), f x ≤ (5 / 3)) ∧ (∃ x ∈ set.Ico (-8 : ℝ) (-4 : ℝ), f x = (5 / 3)) ∧ (¬ ∃ m, ∀ x ∈ set.Ico (-8 : ℝ) (-4 : ℝ), f x >= m) :=
by
  sorry

end max_value_f_on_domain_l165_165495


namespace sum_fourth_powers_lt_1000_l165_165180

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165180


namespace find_angle_CME_l165_165548

-- Given conditions as definitions
variables {A B C M D E F : Point} {ω : Circle}

-- Median from A to BC
axiom is_median_AM : is_median A B C M

-- Circle ω passing through A, touching BC at M, and intersecting AB and AC at D and E respectively
axiom circle_passing_through_A : passes_through ω A
axiom circle_touches_BC_at_M : touches_at ω M BC
axiom circle_intersects_AB_at_D : intersects_at ω AB D
axiom circle_intersects_AC_at_E : intersects_at ω AC E

-- Point F on arc AD not containing E
axiom point_F_on_arc_AD_not_containing_E : on_arc_not_containing F A D E ω

-- Given angles
axiom angle_BFE : ∠ B F E = 72
axiom angle_DEF_eq_angle_ABC : ∠ D E F = ∠ A B C

-- Prove that angle CME is 36 degrees
theorem find_angle_CME : ∠ C M E = 36 :=
by sorry

end find_angle_CME_l165_165548


namespace cos_value_range_f_2B_l165_165489

-- First question: proving the given condition leads to the correct cosine value
theorem cos_value (x : ℝ) 
  (h : (sqrt 3) * sin (x / 4) * cos (x / 4) + cos (x / 4)^2 = 1) : 
  cos (2 * π / 3 - x) = -1 / 2 :=
by
  sorry

-- Second question: proving the range of values for f(2B) in the specified triangle
theorem range_f_2B (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : a * cos C + c / 2 = b)
  (h5 : A + B + C = π) :
  (1 + sqrt 3) / 2 < sin (B + π / 6) + 1 / 2 ∧ sin (B + π / 6) + 1 / 2 < 3 / 2 :=
by
  sorry

end cos_value_range_f_2B_l165_165489


namespace min_treasures_buried_l165_165321

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165321


namespace element_of_M_l165_165497

def M : Set (ℕ × ℕ) := { (2, 3) }

theorem element_of_M : (2, 3) ∈ M :=
by
  sorry

end element_of_M_l165_165497


namespace correct_statements_l165_165798

variables {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
def S_n (n : ℕ) : ℝ := (n * a_n 1) + (n * (n - 1) / 2 * d)
def S6_gt_S7_gt_S5 : Prop := S 6 > S 7 ∧ S 7 > S 5

-- Prove the required statements
theorem correct_statements (S6_gt_S7_gt_S5 : S6_gt_S7_gt_S5) : 
  (d < 0 ∧ |a_n 6| > |a_n 7|) ∧ 
  ¬(S 12 < 0) ∧ ¬(∀ n, S n > 0) ∧
  ¬ (∀ n, S n ≤ S 11) :=
by 
  -- Proof omitted
  sorry


end correct_statements_l165_165798


namespace sum_of_divisors_of_12_l165_165431

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem sum_of_divisors_of_12 :
  (∑ n in (Finset.filter (λ n, is_divisible 12 n) (Finset.range 13)), n) = 28 :=
by
  sorry

end sum_of_divisors_of_12_l165_165431


namespace magnitude_difference_l165_165502

noncomputable
def vector_a : ℝ × ℝ := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
noncomputable
def vector_b : ℝ × ℝ := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))

noncomputable
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_difference :
  magnitude (vector_a - (2 : ℝ) • vector_b) = Real.sqrt 3 :=
by
  sorry

end magnitude_difference_l165_165502


namespace age_of_fourth_child_l165_165562

theorem age_of_fourth_child (c1 c2 c3 c4 : ℕ) (h1 : c1 = 15)
  (h2 : c2 = c1 - 1) (h3 : c3 = c2 - 4)
  (h4 : c4 = c3 - 2) : c4 = 8 :=
by
  sorry

end age_of_fourth_child_l165_165562


namespace find_percentage_of_number_l165_165602

theorem find_percentage_of_number (N : ℕ) (h : (1/4 : ℚ) * (1/3 : ℚ) * (2/5 : ℚ) * (N : ℚ) = 25) :
  0.40 * N = 300 := by
  sorry

end find_percentage_of_number_l165_165602


namespace sum_kml_l165_165755

theorem sum_kml (k m l : ℤ) (b : ℤ → ℤ)
  (h_seq : ∀ n, ∃ k, b n = k * (Int.floor (Real.sqrt (n + m : ℝ))) + l)
  (h_b1 : b 1 = 2) :
  k + m + l = 3 := by
  sorry

end sum_kml_l165_165755


namespace min_treasures_buried_l165_165324

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165324


namespace line_through_point_hyperbola_l165_165720

theorem line_through_point_hyperbola {x y k : ℝ} : 
  (∃ k : ℝ, ∃ x y : ℝ, y = k * (x - 3) ∧ x^2 / 4 - y^2 = 1 ∧ (1 - 4 * k^2) = 0) → 
  (∃! k : ℝ, (k = 1 / 2) ∨ (k = -1 / 2)) := 
sorry

end line_through_point_hyperbola_l165_165720


namespace find_ab_l165_165936

theorem find_ab (a b q r : ℕ) (h : a > 0) (h2 : b > 0) (h3 : (a^2 + b^2) / (a + b) = q) (h4 : (a^2 + b^2) % (a + b) = r) (h5 : q^2 + r = 2010) : a * b = 1643 :=
sorry

end find_ab_l165_165936


namespace min_treasures_buried_l165_165266

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165266


namespace sum_of_fourth_powers_less_than_1000_l165_165183

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165183


namespace prime_count_le_23_l165_165058

open Nat

theorem prime_count_le_23 (k : ℕ) (hk : k ≥ 2) : 
  let interval := (10 * k, 10 * k + 100)
  let primes_in_interval := {p ∈ Finset.range (interval.2 + 1) | p ≥ interval.1 ∧ prime p}
  primes_in_interval.card ≤ 23 := sorry

end prime_count_le_23_l165_165058


namespace smallest_sum_of_integers_on_square_vertices_l165_165724

theorem smallest_sum_of_integers_on_square_vertices :
  ∃ (a b c d : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  (a % b = 0 ∨ b % a = 0) ∧ (c % a = 0 ∨ a % c = 0) ∧ 
  (d % b = 0 ∨ b % d = 0) ∧ (d % c = 0 ∨ c % d = 0) ∧ 
  a % c ≠ 0 ∧ a % d ≠ 0 ∧ b % c ≠ 0 ∧ b % d ≠ 0 ∧ 
  (a + b + c + d = 35) := sorry

end smallest_sum_of_integers_on_square_vertices_l165_165724


namespace tax_rate_as_percent_l165_165711

def TaxAmount (amount : ℝ) : Prop := amount = 82
def BaseAmount (amount : ℝ) : Prop := amount = 100

theorem tax_rate_as_percent {tax_amt base_amt : ℝ} 
  (h_tax : TaxAmount tax_amt) (h_base : BaseAmount base_amt) : 
  (tax_amt / base_amt) * 100 = 82 := 
by 
  sorry

end tax_rate_as_percent_l165_165711


namespace sum_of_fourth_powers_less_than_1000_l165_165187

theorem sum_of_fourth_powers_less_than_1000 :
  ∑ n in Finset.filter (fun n => n ^ 4 < 1000) (Finset.range 100), n ^ 4 = 979 := by
  sorry

end sum_of_fourth_powers_less_than_1000_l165_165187


namespace sum_fourth_powers_lt_1000_l165_165175

theorem sum_fourth_powers_lt_1000 : 
  let S := {x : ℕ | x < 1000 ∧ ∃ k : ℕ, x = k ^ 4} in
  ∑ x in S, x = 979 :=
by 
  -- proof goes here
  sorry

end sum_fourth_powers_lt_1000_l165_165175


namespace S_eq_2D_l165_165031

open Finset

def H : Finset ℕ := (Finset.range 2006).map ⟨Nat.succ, Nat.succ_injective⟩

def subsets_with_remainder (n : ℕ) (r : ℕ) : ℕ :=
  (H.powerset.filter (λ s, s.sum % n = r)).card

def D : ℕ := subsets_with_remainder 32 7
def S : ℕ := subsets_with_remainder 16 14

theorem S_eq_2D : S = 2 * D := sorry

end S_eq_2D_l165_165031


namespace prove_concyclic_l165_165002

noncomputable def cyclic_points
  (A B C I D K G E F : Type) 
  [Incenter A B C I]
  (h1 : Intersect AI BC D)
  (h2 : Circumcircle_intersect BID ABC K)
  (h3 : Circumcircle_intersect CID ABC G)
  (h4 : OnLine E BC ∧ CE = CA)
  (h5 : OnLine F BC ∧ BF = BA) : Prop :=
  Concyclic K E G F

theorem prove_concyclic 
  (A B C I D K G E F : Type) 
  [Incenter A B C I]
  (h1 : Intersect AI BC D)
  (h2 : Circumcircle_intersect BID ABC K)
  (h3 : Circumcircle_intersect CID ABC G)
  (h4 : OnLine E BC ∧ CE = CA)
  (h5 : OnLine F BC ∧ BF = BA) :
  cyclic_points A B C I D K G E F :=
  sorry

end prove_concyclic_l165_165002


namespace sphere_cylinder_surface_area_diff_l165_165533

theorem sphere_cylinder_surface_area_diff :
  let r_sphere := 4
  let surface_area_sphere := 4 * Real.pi * r_sphere^2
  ∃ (h r_cylinder : ℝ), 
  (r_cylinder^2 + h^2 = r_sphere^2) ∧
  (let lateral_surface_area_cylinder := 2 * Real.pi * r_cylinder * h,
     lateral_surface_area_cylinder = 32 * Real.pi ) ∧
  let surface_area_diff := surface_area_sphere - 32 * Real.pi
  in surface_area_diff = 32 * Real.pi :=
by
  sorry

end sphere_cylinder_surface_area_diff_l165_165533


namespace parabola_transformation_l165_165909

-- Defining the original parabola
def original_parabola (x : ℝ) : ℝ :=
  3 * x^2

-- Condition: Transformation 1 -> Translation 4 units to the right
def translated_right_parabola (x : ℝ) : ℝ :=
  original_parabola (x - 4)

-- Condition: Transformation 2 -> Translation 1 unit upwards
def translated_up_parabola (x : ℝ) : ℝ :=
  translated_right_parabola x + 1

-- Statement that needs to be proved
theorem parabola_transformation :
  ∀ x : ℝ, translated_up_parabola x = 3 * (x - 4)^2 + 1 :=
by
  intros x
  sorry

end parabola_transformation_l165_165909


namespace largest_possible_product_l165_165605

theorem largest_possible_product : 
  ∃ S1 S2 : Finset ℕ, 
  (S1 ∪ S2 = {1, 3, 4, 6, 7, 8, 9} ∧ S1 ∩ S2 = ∅ ∧ S1.prod id = S2.prod id) ∧ 
  (S1.prod id = 504 ∧ S2.prod id = 504) :=
by
  sorry

end largest_possible_product_l165_165605


namespace sum_of_first_10_terms_l165_165034

noncomputable def a (n : ℕ) : ℕ := 2^n
noncomputable def b (n : ℕ) : ℕ := n

noncomputable def A (n : ℕ) : ℕ := (Finset.range n).sum (λ k, 2^(k+1))
noncomputable def B (n : ℕ) : ℕ := (n * (n + 1)) / 2

noncomputable def c (n : ℕ) : ℕ := 
  a n * B n + b n * A n - a n * b n

noncomputable def sum_c (n : ℕ) : ℕ := 
  (Finset.range n).sum c

theorem sum_of_first_10_terms : 
  sum_c 10 = 110 * (2^10 - 1) :=
sorry

end sum_of_first_10_terms_l165_165034


namespace sum_of_fourth_powers_below_1000_l165_165166

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165166


namespace coefficient_of_x3_in_expansion_correct_l165_165763

noncomputable def coefficient_of_x3_in_expansion : ℕ :=
  6.choose 4 * 2

theorem coefficient_of_x3_in_expansion_correct :
  coefficient_of_x3_in_expansion = 30 :=
by
  sorry

end coefficient_of_x3_in_expansion_correct_l165_165763


namespace probability_discrete_case_probability_continuous_case_l165_165591

def line_intersects_circle (a b : ℝ) : Prop :=
  (a * (2 * real.sqrt 2)) / real.sqrt (a^2 + b^2) < real.sqrt 6

theorem probability_discrete_case :
  let event_A := λ (a b : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ line_intersects_circle a b in
  ((finset.filter event_A {p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6}.to_finset).card : ℚ) / 36 = 7 / 12 := 
  sorry

theorem probability_continuous_case :
  let event_A := λ (a b : ℝ), ((a - real.sqrt 3)^2 + (b - 1)^2 ≤ 4) ∧ line_intersects_circle a b in
  (∫∫ (λ (a b : ℝ), if event_A a b then 1 else 0) (-2) real.sqrt 7 (b, a) ) / 
  (π * 4) = 1 / 2 := 
  sorry

end probability_discrete_case_probability_continuous_case_l165_165591


namespace trig_identity_l165_165773

theorem trig_identity :
  sin (24 * (Real.pi / 180)) * cos (54 * (Real.pi / 180)) - cos (24 * (Real.pi / 180)) * sin (54 * (Real.pi / 180)) = - 1 / 2 :=
begin
  sorry
end

end trig_identity_l165_165773


namespace equivalent_problem_l165_165745

theorem equivalent_problem :
  let a : ℤ := (-6)
  let b : ℤ := 6
  let c : ℤ := 2
  let d : ℤ := 4
  (a^4 / b^2 - c^5 + d^2 = 20) :=
by
  sorry

end equivalent_problem_l165_165745


namespace arithmetic_mean_q_r_l165_165080

theorem arithmetic_mean_q_r (p q r : ℝ) (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) (h3 : r - p = 34) : (q + r) / 2 = 27 :=
sorry

end arithmetic_mean_q_r_l165_165080


namespace remaining_budget_l165_165227

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l165_165227


namespace sqrt_of_factorial_5_l165_165192

theorem sqrt_of_factorial_5 : ∃ k, ∃ m, k = 240 ∧ m = 30 ∧ (5! * (5!)^2) ^ (1 / 2 : ℤ) = k * real.sqrt m := by
  sorry

end sqrt_of_factorial_5_l165_165192


namespace exists_number_divisible_by_exactly_50_l165_165344

def is_divisible_by_exactly_50 (n : ℕ) : Prop :=
  let s := { i | 1 ≤ i ∧ i ≤ 100 }
  (∃ (P : ℕ), (∀ k ∈ s, k ∣ P ↔ k % 2 = 1 ∧ k ≤ 99) ∧ 
  (∃ n : ℕ, (count (∣ P) s) = 50))

theorem exists_number_divisible_by_exactly_50 :
  ∃ P, is_divisible_by_exactly_50 P :=
sorry

end exists_number_divisible_by_exactly_50_l165_165344


namespace smallest_base_b_l165_165679

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l165_165679


namespace circumcenter_BCD_on_circumcircle_ABC_l165_165567

-- Definitions and conditions
variables {A B C D : Point}
variables {k1 k2 : Circle}
variables (O₁ : is_circumcenter k1 B)
variables (O₂ : is_circumcenter k2 C D B)

-- Condition 1: B is on circle k1
axiom B_on_k1 : on_circle B k1

-- Condition 2: A is not equal to B and lies on the tangent to k1 at B
axiom A_not_eq_B : A ≠ B
axiom A_on_tangent : on_tangent A k1 B

-- Condition 3: C is not on k1 and AC meets k1 at two distinct points
axiom C_not_on_k1 : ¬on_circle C k1
axiom AC_meets_k1_twice : ∃ P Q, P ≠ Q ∧ on_line P A C ∧ on_circle P k1 ∧ on_circle Q k1

-- Condition 4: Circle k2 is tangent to AC at C and to k1 at D, k2 does not lie in the same half-plane as B with respect to AC
axiom k2_tangent_AC_at_C : tangent_to_line k2 AC C
axiom k2_tangent_k1_at_D : tangent_to_circle k2 k1 D
axiom k2_not_same_half_plane_as_B : ¬same_half_plane k2 B A C

-- Theorem to prove: The circumcenter of triangle BCD lies on the circumcircle of triangle ABC
theorem circumcenter_BCD_on_circumcircle_ABC :
  let O₃ := circumcenter B C D in
  on_circle O₃ (circumcircle A B C) :=
sorry

end circumcenter_BCD_on_circumcircle_ABC_l165_165567


namespace min_treasure_count_l165_165313

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l165_165313


namespace problem_1_problem_2_l165_165494

-- Problem 1: Prove that if the solution set for the given function is as described, then a = -1.
theorem problem_1 {a : ℝ} (h1 : ∀ x, f(a, x) ≤ 2 ↔ -3 ≤ x ∧ x ≤ 1) : a = -1 :=
sorry

def f (a x : ℝ) := abs (a * x - 1)

-- Problem 2: Prove that for a = 1 and given inequality, m ≤ 5/2.
theorem problem_2 {m : ℝ} (h2 : ∃ x : ℝ, f(1, 2*x + 1) - f(1, x - 1) ≤ 3 - 2*m) : m ≤ 5 / 2 :=
sorry

end problem_1_problem_2_l165_165494


namespace sum_of_distances_ellipse_l165_165966

theorem sum_of_distances_ellipse {c a : ℝ} (h_c_gt_zero : c > 0) (h_a_gt_c : a > c) :
  ∀ (x y : ℝ), 
  sqrt ((x - c) ^ 2 + y ^ 2) + sqrt ((x + c) ^ 2 + y ^ 2) = 2 * a →
  (x^2 / a^2 + y^2 / (a^2 - c^2) = 1) :=
begin
  sorry
end

end sum_of_distances_ellipse_l165_165966


namespace sector_area_l165_165095

theorem sector_area (L r : ℝ) (hL : L = 1) (hr : r = 4) : 
    let θ := L / r
    let A := 1/2 * r^2 * θ
    A = 2 := by
      have θ_def : θ = 0.25 by calc
        θ = L / r          := by rfl
        ... = 1 / 4        := by rw [hL, hr]
        ... = 0.25         := by rfl
      have A_def : A = 1/2 * r ^ 2 * θ := by rfl
      calc
        A = 1/2 * r ^ 2 * θ := A_def
        ... = 1/2 * 16 * 0.25 := by rw [hr, θ_def]
        ... = 2 := by norm_num

end sector_area_l165_165095


namespace trig_identity_example_l165_165331

theorem trig_identity_example :
  sin (real.pi * 18 / 180) * sin (real.pi * 78 / 180) - 
  cos (real.pi * 162 / 180) * cos (real.pi * 78 / 180) = 
  1 / 2 := 
sorry

end trig_identity_example_l165_165331


namespace proof_problem_l165_165903

-- Definitions of the parametric equations for curve C1
def parametric_eq_C1 (α : ℝ) := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- The general equation of curve C1
def general_eq_C1 (x y : ℝ) := x^2 + (y - 2)^2 = 4

-- Definition of the polar coordinate equation for curve C2
def polar_eq_C2 (ρ θ : ℝ) := ρ^2 = 2 / (1 + (Real.sin θ)^2)

-- The Cartesian equation of curve C2
def cartesian_eq_C2 (x y : ℝ) := (x^2 / 2) + y^2 = 1

-- Definition of the Euclidean distance |MN|
def dist_MN (x_m y_m : ℝ) := Real.sqrt ((x_m)^2 + (y_m - 2)^2)

-- Statement that encapsulates the problem to be proven
theorem proof_problem :
  (∀ α, let (x, y) := parametric_eq_C1 α in general_eq_C1 x y) ∧
  (∀ ρ θ, polar_eq_C2 ρ θ → let x := ρ * Real.cos θ, y := ρ * Real.sin θ in cartesian_eq_C2 x y) ∧
  (∀ x_m y_m, cartesian_eq_C2 x_m y_m → dist_MN x_m y_m ≤ 3 ∧ ∃ x_m y_m, dist_MN x_m y_m = 3) :=
by
  sorry

end proof_problem_l165_165903


namespace circumcircle_tangent_to_omega_l165_165632

open EuclideanGeometry

variables {P K I Q E F : Point}
variable {circle Ω : Circle}
variables {triangle EIF : Triangle}

/-- The statement of the proof problem in Lean 4 -/
theorem circumcircle_tangent_to_omega
  (hEIF_circum : Ω = C_circumcircle triangle EIF)
  (hIKP_90 : ∠ I K P = 90)
  (hPK_intersect_omega : Q ∈ Line P K ∧ Q ∈ Ω) :
  is_tangent (C_circumcircle (Triangle.mk E Q F)) Ω :=
sorry

end circumcircle_tangent_to_omega_l165_165632


namespace find_f_of_1_l165_165832

def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem find_f_of_1 (f : ℝ → ℝ) (h : ∀ y : ℝ, f_inv (f y) = y) : f 1 = 1 :=
by
sorry

end find_f_of_1_l165_165832


namespace Roselyn_initial_books_correct_l165_165615

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end Roselyn_initial_books_correct_l165_165615


namespace TomsTotalScoreIs2250_l165_165665

def toms_total_score (kill_points : ℕ) (bonus_percentage : ℝ) (enemies_killed : ℕ) : ℕ :=
  let base_score := kill_points * enemies_killed
  let bonus := if enemies_killed ≥ 100 then base_score * (bonus_percentage / 100) else 0
  base_score + bonus

theorem TomsTotalScoreIs2250 :
  toms_total_score 10 50 150 = 2250 := 
by 
  sorry

end TomsTotalScoreIs2250_l165_165665


namespace chickens_bought_l165_165253

theorem chickens_bought (total_spent : ℤ) (egg_count : ℤ) (egg_price : ℤ) (chicken_price : ℤ) (egg_cost : ℤ := egg_count * egg_price) (chicken_spent : ℤ := total_spent - egg_cost) : total_spent = 88 → egg_count = 20 → egg_price = 2 → chicken_price = 8 → chicken_spent / chicken_price = 6 :=
by
  intros
  sorry

end chickens_bought_l165_165253


namespace sum_of_recorded_products_l165_165115

theorem sum_of_recorded_products : 
  let n := 25 
  in (∑ i in (finset.range (n-1)), let x := (n - 1 - i) in let y := (i + 1) in x * y) = 300 := 
by 
  sorry

end sum_of_recorded_products_l165_165115


namespace exists_adjacent_even_sum_l165_165657

-- Given conditions
def circle_of_nat_nums : Prop := ∃ (nums : list ℕ), nums.length = 2019 ∧ ∀ i < 2019, (nums.nth i ≠ none)

theorem exists_adjacent_even_sum (nums : list ℕ) (h1 : nums.length = 2019) (h2 : ∀ i < 2019, (nums.nth i ≠ none)) :
  ∃ i < 2019, (nums.nth_le i (by linarith)).get_or_else 0 + (nums.nth_le ((i + 1) % 2019) (by linarith)).get_or_else 0 % 2 = 0 :=
begin
  sorry
end

end exists_adjacent_even_sum_l165_165657


namespace probability_PORTLAND_l165_165017

noncomputable def probability_dock : ℚ :=
  1 / Nat.choose 4 2

noncomputable def probability_plants : ℚ :=
  2 / Nat.choose 6 4

noncomputable def probability_hero : ℚ :=
  3 / Nat.choose 4 3

noncomputable def total_probability : ℚ :=
  probability_dock * probability_plants * probability_hero

theorem probability_PORTLAND : total_probability = 1 / 40 :=
by
  sorry

end probability_PORTLAND_l165_165017


namespace sum_divisors_of_12_l165_165401

theorem sum_divisors_of_12 :
  ∑ n in {1, 2, 3, 4, 6, 12}, n = 28 :=
by
  -- Proof will be provided here
  sorry

end sum_divisors_of_12_l165_165401


namespace max_value_of_expression_l165_165471

theorem max_value_of_expression (x : ℝ) (h : 0 < x ∧ x < 6) : 
  (6 - x) * x ≤ 9 :=
begin
  sorry
end

end max_value_of_expression_l165_165471


namespace sequence_product_l165_165762

theorem sequence_product : 
  let a : ℕ → ℝ := λ n, (nat.rec (1 / 3) (λ n a_n, 1 + (a_n - 1)^3) n)
  in (∏ n, a n) = 3 / 5 :=
sorry

end sequence_product_l165_165762


namespace aaron_guesses_correctly_l165_165918

noncomputable def P_H : ℝ := 2 / 3
noncomputable def P_T : ℝ := 1 / 3
noncomputable def P_G_H : ℝ := 2 / 3
noncomputable def P_G_T : ℝ := 1 / 3

noncomputable def p : ℝ := P_H * P_G_H + P_T * P_G_T

theorem aaron_guesses_correctly :
  9000 * p = 5000 :=
by
  sorry

end aaron_guesses_correctly_l165_165918


namespace count_b_divisible_7_l165_165575

noncomputable def d3 (b : ℤ) : ℤ := b^3 + 3^b + b * 3^((b + 2) / 3)
noncomputable def d4 (b : ℤ) : ℤ := b^3 + 3^b - b * 3^((b + 2) / 3)

theorem count_b_divisible_7 (n : ℕ) (range : 1 <= n ∧ n <= 300) :
  (∃ (b_set : finset ℤ), b_set.card = 171 ∧
    ∀ b ∈ b_set, let d_3 := d3 b, d_4 := d4 b in (d_3 * d_4) % 7 = 0) :=
sorry

end count_b_divisible_7_l165_165575


namespace part1_part2_l165_165469

-- Define the sets A, B, and C
def setA (m : ℤ) : Set ℤ := { |m|, 0 }
def setB : Set ℤ := { -2, 0, 2 }
def setC : Set ℤ := { -2, -1, 0, 1, 2, 3 }

-- First proof: if A ⊆ B, prove m = ±2
theorem part1 (m : ℤ) (h_subset : setA m ⊆ setB) : m = 2 ∨ m = -2 :=
  sorry

-- Second proof: number of possible sets P that satisfy B ⊆ P ⊆ C is 8
theorem part2 : (finset.powerset setC.to_finset).filter (λ P, setB.to_finset ⊆ P ∧ P ⊆ setC.to_finset).card = 8 :=
  sorry

end part1_part2_l165_165469


namespace symmetric_circle_equation_l165_165704

open EuclideanGeometry

def circleEquation (h k r : ℝ) := ∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem symmetric_circle_equation :
  let C1 := circleEquation (-1) 1 1 in
  let symmetric_line : ℝ×ℝ → ℝ := λ p, p.1 - p.2 - 1 in
  let sym_point := λ p:ℝ×ℝ, (2 * 1 - p.1, 2 * 1 - p.2) in -- Calculation of symmetrical point (2, -2) 
  let C2 := circleEquation 2 (-2) 1 in
  C2 = (λ x y, (x - 2) ^ 2 + (y + 2) ^ 2 = 1) :=
by
  sorry

end symmetric_circle_equation_l165_165704


namespace sum_divisors_12_eq_28_l165_165368

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165368


namespace arithmetic_sequence_problem_l165_165461

variable {a : ℕ → ℕ} -- Assuming a_n is a function from natural numbers to natural numbers

theorem arithmetic_sequence_problem (h1 : a 1 + a 2 = 10) (h2 : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_sequence_problem_l165_165461


namespace AF_squared_l165_165890

-- The statement encapsulating the equivalent proof problem.
theorem AF_squared (ω : Circle) (A B C D E F : Point) (hω : inscribed_in_triangle A B C ω)
  (hAB : distance A B = 6) (hBC : distance B C = 8) (hAC : distance A C = 4)
  (h_D_angle_bisector : is_angle_bisector D A B C)
  (D_on_BC : lies_on D (line_segment B C))
  (E_on_ω : lies_on E ω ∧ E ≠ D)
  (γ : Circle)
  (h_γ_diameter : diameter γ = line_segment D E)
  (h_ω_γ_intersect : intersects_at_two_points ω γ E F) :
  (distance A F) ^ 2 = 96 / 25 := sorry

end AF_squared_l165_165890


namespace sum_of_fourth_powers_below_1000_l165_165164

theorem sum_of_fourth_powers_below_1000 : 
  (∑ n in finset.filter (fun n => ∃ (k:ℕ), n = k^4) (finset.range 1000), n) = 979 := 
by
  sorry

end sum_of_fourth_powers_below_1000_l165_165164


namespace burn_rate_walking_l165_165257

def burn_rate_running : ℕ := 10
def total_calories : ℕ := 450
def total_time : ℕ := 60
def running_time : ℕ := 35

theorem burn_rate_walking :
  ∃ (W : ℕ), ((running_time * burn_rate_running) + ((total_time - running_time) * W) = total_calories) ∧ (W = 4) :=
by
  sorry

end burn_rate_walking_l165_165257


namespace probability_odd_broken_mul_probability_even_fixed_mul_is_greater_l165_165054

-- Part (a) - The multiplication button is broken
theorem probability_odd_broken_mul : 
  (calc_prob_odd_broken_mul : ℚ) = 1 / 2 := 
sorry

-- Part (b) - The multiplication button is fixed
theorem probability_even_fixed_mul_is_greater : 
  ∀ n : ℕ, n > 0 → (calc_prob_even_fixed_mul : ℚ) > (calc_prob_odd_fixed_mul n) := 
sorry

end probability_odd_broken_mul_probability_even_fixed_mul_is_greater_l165_165054


namespace problem1_problem2_l165_165524

-- Definition for Problem 1
def seagull_oxygen_consumption (x : ℝ) (x₀ : ℝ) (v : ℝ) : ℝ :=
  0 = (1/2) * Real.logBase 3 (x / 100) - Real.log x₀

-- Definition for Problem 2
def seagull_oxygens_ratio (x₁ x₂ : ℝ) (x₀ v₁ v₂ : ℝ) : ℝ :=
  ((1/2) * Real.logBase 3 (x₂ / 100) - Real.log x₀) - ((1/2) * Real.logBase 3 (x₁ / 100) - Real.log x₀) = 0.3

-- Statement for Problem 1
theorem problem1 (x₀ : ℝ) (h₁ : x₀ = 10) : ∃ x : ℝ, seagull_oxygen_consumption x x₀ 0 ∧ x = 900 := 
sorry

-- Statement for Problem 2
theorem problem2 (x₀ : ℝ) (h₁ : x₀ = 10) (v₁ v₂ : ℝ) (h₂ : v₁ = 0.5) (h₃ : v₂ = 0.8) :
  ∃ x₁ x₂ : ℝ, seagull_oxygen_consumption x₁ x₀ v₁ ∧ seagull_oxygen_consumption x₂ x₀ v₂ ∧ (x₂ / x₁ = 1.93) := 
sorry

end problem1_problem2_l165_165524


namespace min_treasures_buried_l165_165323

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165323


namespace min_treasure_signs_buried_l165_165283

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l165_165283


namespace total_votes_is_120_l165_165891

-- Define the conditions
def Fiona_votes : ℕ := 48
def fraction_of_votes : ℚ := 2 / 5

-- The proof goal
theorem total_votes_is_120 (V : ℕ) (h : Fiona_votes = fraction_of_votes * V) : V = 120 :=
by
  sorry

end total_votes_is_120_l165_165891


namespace sin_A_of_isosceles_trapezoid_l165_165022

theorem sin_A_of_isosceles_trapezoid (
  A B C D : Type) [IsoscelesTrapezoid A B C D]
  (AB_parallel_CD : parallel AB CD)
  (angle_A : angle_measure A D = 120)
  (AB_eq : AB = 160)
  (CD_eq : CD = 240)
  (AD_eq_BC : AD = BC)
  (perimeter_ABCD : AB + CD + 2 * AD = 800) :
  sin A = sqrt 3 / 2 :=
by {
  sorry
}

end sin_A_of_isosceles_trapezoid_l165_165022


namespace min_treasures_buried_l165_165325

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l165_165325


namespace total_revenue_calculation_l165_165669

variables (a b : ℕ) -- Assuming a and b are natural numbers representing the number of newspapers

-- Define the prices
def purchase_price_per_copy : ℝ := 0.4
def selling_price_per_copy : ℝ := 0.5
def return_price_per_copy : ℝ := 0.2

-- Define the revenue and cost calculations
def revenue_from_selling (b : ℕ) : ℝ := selling_price_per_copy * b
def revenue_from_returning (a b : ℕ) : ℝ := return_price_per_copy * (a - b)
def cost_of_purchasing (a : ℕ) : ℝ := purchase_price_per_copy * a

-- Define the total revenue
def total_revenue (a b : ℕ) : ℝ :=
  revenue_from_selling b + revenue_from_returning a b - cost_of_purchasing a

-- The theorem we need to prove
theorem total_revenue_calculation (a b : ℕ) :
  total_revenue a b = 0.3 * b - 0.2 * a :=
by
  sorry

end total_revenue_calculation_l165_165669


namespace variance_Y_eq_15_l165_165838

-- Definitions for the problem

-- Condition 1: X follows a binomial distribution B(5, 1/4)
def binom_dist : ProbabilityTheory.Discrete.val (prob = ProbabilityTheory.Hypergeometric(0, 1, 2, 3)) := sorry

-- Condition 2: Y is defined as 4 * X - 3
def Y (X : ℝ) := 4 * X - 3

-- The proof goal: V(Y) = 15
theorem variance_Y_eq_15 (X : ℝ) [hx : ProbabilityTheory.Binomial (5, 1/4)] : 
  let Y := 4 * X - 3 in
  ProbabilityTheory.Variance Y = 15 := by
sorry

end variance_Y_eq_15_l165_165838


namespace negative_integers_count_in_list_l165_165908

def is_negative_integer (n : ℚ) : Prop :=
  n < 0 ∧ n.den = 1

def list_of_numbers : List ℚ :=
  [+8.3, -4, -0.8, -1/5, 0, 90, - (4 + 1/2), - |24|]

theorem negative_integers_count_in_list :
  (list_of_numbers.filter is_negative_integer).length = 1 :=
sorry

end negative_integers_count_in_list_l165_165908


namespace sum_of_divisors_of_12_l165_165418

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165418


namespace sum_of_divisors_of_12_l165_165416

theorem sum_of_divisors_of_12 : 
  (∑ n in {n : ℕ | n > 0 ∧ 12 % n = 0}, n) = 28 :=
sorry

end sum_of_divisors_of_12_l165_165416


namespace sum_of_squares_of_perfect_squares_l165_165131

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165131


namespace minimum_treasure_buried_l165_165294

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l165_165294


namespace probability_product_less_than_40_l165_165961

def pacoSpins (paco : ℕ) : Prop := paco ∈ {1, 2, 3, 4, 5, 6}
def manuSpins (manu : ℕ) : Prop := manu ∈ {1, 2, 3, 4, 5, 6, 7, 8}

def validProduct (paco manu : ℕ) : Prop := paco * manu < 40

theorem probability_product_less_than_40 : 
  (∑ p in {1, 2, 3, 4, 5, 6}, ∑ m in {1, 2, 3, 4, 5, 6, 7, 8}, 
    if validProduct p m then (1/6) * (1/8) else 0) = 15/16 := 
sorry

end probability_product_less_than_40_l165_165961


namespace smallest_multiple_of_5_and_21_l165_165788

theorem smallest_multiple_of_5_and_21 : ∃ b : ℕ, b > 0 ∧ b % 5 = 0 ∧ b % 21 = 0 ∧ ∀ c : ℕ, c > 0 ∧ c % 5 = 0 ∧ c % 21 = 0 → b ≤ c := 
by {
  use 105,
  split,
  { -- Proof that 105 is greater than 0
    exact Nat.zero_lt_of_pos,
  },
  split,
  { -- Proof that 105 is a multiple of 5
    exact Nat.mod_eq_zero_of_dvd,
    exact dvd_refl 105,
  },
  split,
  { -- Proof that 105 is a multiple of 21
    exact Nat.mod_eq_zero_of_dvd,
    exact dvd_refl 105,
  },
  { -- Proof that 105 is the smallest number satisfying these conditions
    intros c h1 h2 h3,
    obtain ⟨k1, hk1⟩ := Nat.exists_eq_mul_right_of_dvd h2,
    obtain ⟨k2, hk2⟩ := Nat.exists_eq_mul_right_of_dvd h3,
    rw [hk1, hk2] at h,
    exact le_trans le_rfl h,
  },
  sorry,
}

end smallest_multiple_of_5_and_21_l165_165788


namespace total_handshakes_11_boys_l165_165515

theorem total_handshakes_11_boys : 
  let n := 11 in 
  n * (n - 1) / 2 = 55 :=
by sorry

end total_handshakes_11_boys_l165_165515


namespace xiaohua_final_score_l165_165714

-- Definitions for conditions
def education_score : ℝ := 9
def experience_score : ℝ := 7
def work_attitude_score : ℝ := 8
def weight_education : ℝ := 1
def weight_experience : ℝ := 2
def weight_attitude : ℝ := 2

-- Computation of the final score
noncomputable def final_score : ℝ :=
  education_score * (weight_education / (weight_education + weight_experience + weight_attitude)) +
  experience_score * (weight_experience / (weight_education + weight_experience + weight_attitude)) +
  work_attitude_score * (weight_attitude / (weight_education + weight_experience + weight_attitude))

-- The statement we want to prove
theorem xiaohua_final_score :
  final_score = 7.8 :=
sorry

end xiaohua_final_score_l165_165714


namespace avg_visitors_other_days_l165_165718

-- Definitions for average visitors on Sundays and average visitors over the month
def avg_visitors_on_sundays : ℕ := 600
def avg_visitors_over_month : ℕ := 300
def days_in_month : ℕ := 30

-- Given conditions
def num_sundays_in_month : ℕ := 5
def total_days : ℕ := days_in_month
def total_visitors_over_month : ℕ := avg_visitors_over_month * days_in_month

-- Goal: Calculate the average number of visitors on other days (Monday to Saturday)
theorem avg_visitors_other_days :
  (avg_visitors_on_sundays * num_sundays_in_month + (total_days - num_sundays_in_month) * 240) = total_visitors_over_month :=
by
  -- Proof expected here, but skipped according to the instructions
  sorry

end avg_visitors_other_days_l165_165718


namespace systematic_sampling_correct_l165_165659

def systematic_sample (total_products : ℕ) (num_selected : ℕ) (k : ℕ) : list ℕ :=
  list.map (λ i, k + (i * (total_products / num_selected))) (list.range num_selected)

theorem systematic_sampling_correct :
  ∀ (total_products num_selected : ℕ),
  total_products = 50 →
  num_selected = 5 →
  (∃ k : ℕ, systematic_sample total_products num_selected k = [10, 20, 30, 40, 50]) := by
  sorry

end systematic_sampling_correct_l165_165659


namespace cube_puzzle_impossible_trapezoid_MN_length_l165_165916

-- Part 1: Cube and Bricks Problem
theorem cube_puzzle_impossible : 
  ¬ ∃ (bricks : Finset (ℕ × ℕ × ℕ)) (cube : Finset (ℕ × ℕ × ℕ)), 
    (cube.card = 26) ∧ (bricks.card = 13) ∧ 
    (∀ b ∈ bricks, ∃ p : ℕ × ℕ × ℕ, p ∈ cube ∧ ((p.1 + 1, p.2, p.3) ∈ cube ∨ (p.1, p.2 + 1, p.3) ∈ cube ∨ (p.1, p.2, p.3 + 1) ∈ cube)) ∧ 
    ((1, 1, 1) ∉ cube) :=
sorry

-- Part 2: Length of MN in Trapezoid
theorem trapezoid_MN_length (a b : ℕ) : 
    ∀ (AB A M B C N D : ℕ), 
        (AB > 0) ∧ (AM = 3 * MA) ∧ (CN = 3 * ND) ∧ 
        (MN ∥ AD) →
        ∃ (MN : ℕ), MN = (\frac{1}{8} * (5a + 3b)) :=
sorry

end cube_puzzle_impossible_trapezoid_MN_length_l165_165916


namespace sum_divisors_12_eq_28_l165_165362

theorem sum_divisors_12_eq_28 : (Finset.sum (Finset.filter (λ n, 12 % n = 0) (Finset.range 13))) = 28 :=
by
  sorry

end sum_divisors_12_eq_28_l165_165362


namespace find_value_of_a_find_equation_of_line_l165_165463

noncomputable def value_of_a (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x, y) = (sqrt 3, 0) → (x^2 - 1 = 3 * a^2))

noncomputable def equation_of_line {a : ℝ} (l : ℝ × ℝ → Prop) : Prop :=
  (∀ A B P : ℝ × ℝ, (A.1 + B.1 = 1) ∧ (A.2 + B.2 = 1) → 
  (A.1^2 + 4 * A.2^2 = 4) ∧ (B.1^2 + 4 * B.2^2 = 4) →
  P = (1/2, 1/2) → (l = (λ (x : ℝ × ℝ), 2 * x.1 + 8 * x.2 - 5 = 0)))

theorem find_value_of_a : ∃ a : ℝ, value_of_a a :=
begin
  use 2,
  sorry
end

theorem find_equation_of_line : ∃ l : ℝ × ℝ → Prop, equation_of_line l :=
begin
  use (λ (x : ℝ × ℝ), 2 * x.1 + 8 * x.2 - 5 = 0),
  sorry
end

end find_value_of_a_find_equation_of_line_l165_165463


namespace profit_equations_l165_165712

-- Define the conditions
def total_workers : ℕ := 150
def fabric_per_worker_per_day : ℕ := 30
def clothing_per_worker_per_day : ℕ := 4
def fabric_needed_per_clothing : ℝ := 1.5
def profit_per_meter : ℝ := 2
def profit_per_clothing : ℝ := 25

-- Define the profit functions
def profit_clothing (x : ℕ) : ℝ := profit_per_clothing * clothing_per_worker_per_day * x
def profit_fabric (x : ℕ) : ℝ := profit_per_meter * (fabric_per_worker_per_day * (total_workers - x) - fabric_needed_per_clothing * clothing_per_worker_per_day * x)

-- Define the total profit function
def total_profit (x : ℕ) : ℝ := profit_clothing x + profit_fabric x

-- Prove the given statements
theorem profit_equations (x : ℕ) :
  profit_clothing x = 100 * x ∧
  profit_fabric x = 9000 - 72 * x ∧
  total_profit 100 = 11800 :=
by
  -- Proof omitted
  sorry

end profit_equations_l165_165712


namespace value_of_m_l165_165861

theorem value_of_m (m : ℝ) :
  let A := {2, 3}
  let B := {x : ℝ | m * x - 6 = 0}
  (B ⊆ A) → (m = 0 ∨ m = 2 ∨ m = 3) :=
by
  intros A B h
  sorry

end value_of_m_l165_165861


namespace log_base_2_derivative_correct_l165_165685

theorem log_base_2_derivative_correct :
  (∂ (λ x : ℝ, log 2 x) ∂x) = (λ x, 1 / (x * log 2)) :=
sorry

end log_base_2_derivative_correct_l165_165685


namespace coeff_a4_b3_c2_in_expansion_l165_165764

def term_coefficient (a b c : ℕ) (n : ℕ): ℕ := 
  Nat.choose n a * Nat.choose (n - a) b

theorem coeff_a4_b3_c2_in_expansion : 
  term_coefficient 4 5 3 9 = 1260 :=
by 
  sorry

end coeff_a4_b3_c2_in_expansion_l165_165764


namespace total_difference_is_correct_l165_165958

-- Define the harvest rates
def valencia_weekday_ripe := 90
def valencia_weekday_unripe := 38
def navel_weekday_ripe := 125
def navel_weekday_unripe := 65
def blood_weekday_ripe := 60
def blood_weekday_unripe := 42

def valencia_weekend_ripe := 75
def valencia_weekend_unripe := 33
def navel_weekend_ripe := 100
def navel_weekend_unripe := 57
def blood_weekend_ripe := 45
def blood_weekend_unripe := 36

-- Define the number of weekdays and weekend days
def weekdays := 5
def weekend_days := 2

-- Calculate the total harvests
def total_valencia_ripe := valencia_weekday_ripe * weekdays + valencia_weekend_ripe * weekend_days
def total_valencia_unripe := valencia_weekday_unripe * weekdays + valencia_weekend_unripe * weekend_days
def total_navel_ripe := navel_weekday_ripe * weekdays + navel_weekend_ripe * weekend_days
def total_navel_unripe := navel_weekday_unripe * weekdays + navel_weekend_unripe * weekend_days
def total_blood_ripe := blood_weekday_ripe * weekdays + blood_weekend_ripe * weekend_days
def total_blood_unripe := blood_weekday_unripe * weekdays + blood_weekend_unripe * weekend_days

-- Calculate the total differences
def valencia_difference := total_valencia_ripe - total_valencia_unripe
def navel_difference := total_navel_ripe - total_navel_unripe
def blood_difference := total_blood_ripe - total_blood_unripe

-- Define the total difference
def total_difference := valencia_difference + navel_difference + blood_difference

-- Theorem statement
theorem total_difference_is_correct :
  total_difference = 838 := by
  sorry

end total_difference_is_correct_l165_165958


namespace new_assistant_draw_time_l165_165693

-- Definitions based on conditions
def capacity : ℕ := 36
def halfway : ℕ := capacity / 2
def rate_top : ℕ := 1 / 6
def rate_bottom : ℕ := 1 / 4
def extra_time : ℕ := 24

-- The proof statement
theorem new_assistant_draw_time : 
  ∃ t : ℕ, ((capacity - (extra_time * rate_bottom * 1)) - halfway) = (t * rate_bottom * 1) ∧ t = 48 := by
sorry

end new_assistant_draw_time_l165_165693


namespace ryan_lamps_probability_l165_165970

theorem ryan_lamps_probability :
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_ways_to_arrange := Nat.choose total_lamps red_lamps
  let total_ways_to_turn_on := Nat.choose total_lamps 4
  let remaining_blue := blue_lamps - 1 -- Due to leftmost lamp being blue and off
  let remaining_red := red_lamps - 1 -- Due to rightmost lamp being red and on
  let remaining_red_after_middle := remaining_red - 1 -- Due to middle lamp being red and off
  let remaining_lamps := remaining_blue + remaining_red_after_middle
  let ways_to_assign_remaining_red := Nat.choose remaining_lamps remaining_red_after_middle
  let ways_to_turn_on_remaining_lamps := Nat.choose remaining_lamps 2
  let favorable_ways := ways_to_assign_remaining_red * ways_to_turn_on_remaining_lamps
  let total_possibilities := total_ways_to_arrange * total_ways_to_turn_on
  favorable_ways / total_possibilities = (10 / 490) := by
  sorry

end ryan_lamps_probability_l165_165970


namespace part1_part2_l165_165213

open Real

noncomputable def annual_cost (x : ℝ) : ℝ := (1 / 10) * x ^ 2 - 30 * x + 4000

noncomputable def avg_cost (x : ℝ) : ℝ := (1 / 10) * x - 30 + 4000 / x

theorem part1 (x : ℝ) (h₁ : annual_cost x ≤ 2000) (h₂ : 150 ≤ x ∧ x ≤ 250) : 150 ≤ x ∧ x ≤ 200 :=
begin
  sorry
end

theorem part2 (x : ℝ) (h₁ : 150 ≤ x ∧ x ≤ 250) : 
  avg_cost x = min_avg_cost ↔ x = 200 ∧ min_avg_cost = 10 :=
begin
  let min_avg_cost := 10,
  sorry
end

end part1_part2_l165_165213


namespace number_of_distinct_products_of_two_elements_in_S_l165_165573

-- Definitions based on the conditions from the problem
def S : Set ℕ := {d ∣ 72000 | d > 0}

-- Define the function to check the product of two distinct divisors
def product_of_two_distinct_elements (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b

-- The statement to prove
theorem number_of_distinct_products_of_two_elements_in_S : 
  {p | ∃ a b : ℕ, product_of_two_distinct_elements a b ∧ p = a * b}.card = 308 :=
sorry

end number_of_distinct_products_of_two_elements_in_S_l165_165573


namespace timmy_money_left_after_oranges_l165_165604

-- Conditions
def orange_calories : ℕ := 80
def pack_size : ℕ := 3
def price_per_orange : ℝ := 1.20
def timmy_money : ℝ := 10.00
def required_calories : ℕ := 400
def sales_tax_rate : ℝ := 0.05

-- Proof goal
theorem timmy_money_left_after_oranges : 
  let orange_packs_needed := ⌈(required_calories : ℝ) / orange_calories / pack_size⌉ in
  let total_cost := orange_packs_needed * (price_per_orange * pack_size : ℝ) in
  let total_cost_with_tax := total_cost * (1 + sales_tax_rate) in
  timmy_money - total_cost_with_tax = 2.44 := 
  sorry

end timmy_money_left_after_oranges_l165_165604


namespace sum_of_divisors_of_12_l165_165433

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165433


namespace num_valid_pairs_l165_165466

noncomputable def f (x : ℝ) : ℝ := sorry

theorem num_valid_pairs :
  (∀ x y : ℝ, f(x) + f(y) = f(x + y) + x * y) →
  (∃ m : ℤ, f(1) = m) →
  let count_pairs : ℕ := (convex_combinations (λ m n : ℤ, f(n) = 2019)).count_pairs in
  count_pairs = 8 :=
by
  sorry

end num_valid_pairs_l165_165466


namespace ratio_of_divisors_l165_165584

def M : Nat := 75 * 75 * 140 * 343

noncomputable def sumOfOddDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all odd divisors of n. (placeholder)
  sorry

noncomputable def sumOfEvenDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all even divisors of n. (placeholder)
  sorry

theorem ratio_of_divisors :
  let sumOdd := sumOfOddDivisors M
  let sumEven := sumOfEvenDivisors M
  sumOdd / sumEven = 1 / 6 := 
by
  sorry

end ratio_of_divisors_l165_165584


namespace three_digit_multiples_of_three_count_l165_165734

open Finset

noncomputable def num_three_digit_multiples_of_three : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} in
  let is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000 in
  let is_multiple_of_3 (n : ℕ) := n % 3 = 0 in
  let choices := digits.to_list.perms.erase_dup.filter 
    (λ l, l.length = 3 ∧ is_three_digit (10 * (l.nth 0).get_or_else 0 + 10 * (l.nth 1).get_or_else 0 + (l.nth 2).get_or_else 0) ∧ 
          is_multiple_of_3 (10 * (l.nth 0).get_or_else 0 + 10 * (l.nth 1).get_or_else 0 + (l.nth 2).get_or_else 0)) in
  choices.length

theorem three_digit_multiples_of_three_count : num_three_digit_multiples_of_three = 228 :=
by sorry

end three_digit_multiples_of_three_count_l165_165734


namespace minimum_value_of_a_b_l165_165822

theorem minimum_value_of_a_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / (2 + a) + 1 / (a + 2 * b)) = 1) : 
  a + b = sqrt 2 + 1/2 ∧ a = sqrt 2 := 
sorry

end minimum_value_of_a_b_l165_165822


namespace polynomial_properties_of_P_l165_165698

variable {R : Type*} [CommRing R]

noncomputable def P : Polynomial R := sorry

theorem polynomial_properties_of_P (n : ℕ) (P_deg_n : P.degree = n) :
  (P.comp (P.comp (Polynomial.X)) - Polynomial.X).degree = 2 * n ∧
  (P.comp (P.comp (Polynomial.X)) - Polynomial.X) ∣ (P.comp (P.comp (P.comp (Polynomial.X))) - P.comp (Polynomial.X)) :=
by
  sorry

end polynomial_properties_of_P_l165_165698


namespace largest_sum_of_two_largest_angles_in_PQRS_l165_165898

theorem largest_sum_of_two_largest_angles_in_PQRS
  (smallest_angle : ℝ)
  (h1 : smallest_angle = 30)
  (h2 : ∀ (a b c d : ℝ), a + b + c + d = 360)
  (h3 : ∃ (x : ℝ) (d : ℝ), x = 30 ∧ angles = [x, x + d, x + 2 * d, x + 3 * d])
  : 260 ∈ @List.sum [smallest_angle + 40, smallest_angle + 80, smallest_angle + 110, smallest_angle + 150] := 
sorry

end largest_sum_of_two_largest_angles_in_PQRS_l165_165898


namespace smallest_base_b_l165_165680

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l165_165680


namespace pq_equiv_continued_fraction_l165_165618

-- Definitions and conditions
def continued_fraction (n : ℕ) : ℚ :=
  let rec frac i := 
    if i = 0 then (1 : ℚ) 
    else 1 / (2 + frac (i - 1))
  in 1 + frac n

def p_q_cond (p q : ℕ) : Prop :=
  p ^ 2 - 2 * q ^ 2 = 1 ∨ p ^ 2 - 2 * q ^ 2 = -1

def coprime (p q : ℕ) : Prop :=
  Nat.gcd p q = 1

-- Desired theorem statement
theorem pq_equiv_continued_fraction (p q : ℕ) (n : ℕ) :
  p_q_cond p q ∧ coprime p q ↔ p = (continued_fraction n).num ∧ q = (continued_fraction n).den :=
sorry

end pq_equiv_continued_fraction_l165_165618


namespace justin_and_tim_play_together_210_times_l165_165760

theorem justin_and_tim_play_together_210_times
  (players : Finset ℕ) (P : ℕ) (condition1 : players.card = 12)
  (Justin Tim : ℕ) (condition2 : Justin ∈ players) (condition3 : Tim ∈ players)
  (games : Finset (Finset ℕ)) 
  (condition4 : ∀ g ∈ games, g.card = 6) 
  (condition5 : ∀ g1 g2 ∈ games, g1 ≠ g2 → g1 ≠ g2) 
  (condition6 : ∀ s : Finset ℕ, s.card = 6 → s ∈ games) :
  (∃ c, c = 210 ∧ (λ g, Justin ∈ g ∧ Tim ∈ g) '' games).card = c := sorry

end justin_and_tim_play_together_210_times_l165_165760


namespace fraction_sum_of_simplest_form_l165_165999

theorem fraction_sum_of_simplest_form (c d : ℕ) (h1 : 0.375 = c / d) (h2 : ∀ k : ℕ, k > 1 → ¬ (k ∣ c ∧ k ∣ d)) : c + d = 11 := 
sorry

end fraction_sum_of_simplest_form_l165_165999


namespace sum_of_divisors_of_12_l165_165436

theorem sum_of_divisors_of_12 : 
  (∑ d in (Finset.filter (λ d, d > 0) (Finset.divisors 12)), d) = 28 := 
by
  sorry

end sum_of_divisors_of_12_l165_165436


namespace min_treasures_buried_l165_165264

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l165_165264


namespace intersection_and_distance_main_proof_l165_165538

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def curveA_cartesian : Prop :=
  ∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ 
    ∃ (ρ θ : ℝ), (ρ^2 = 12 / (3 + sin θ^2)) ∧ (x, y) = polar_to_cartesian ρ θ

def curveB_parametric (t : ℝ) : ℝ × ℝ :=
  (-1 + (sqrt 2 / 2) * t, 1 + (sqrt 2 / 2) * t)

theorem intersection_and_distance : Prop :=
  ∀ t₁ t₂ : ℝ, 
  (3 * (-1 + (sqrt 2 / 2) * t₁)^2 + 4 * (1 + (sqrt 2 / 2) * t₁)^2 = 12) ∧
  (3 * (-1 + (sqrt 2 / 2) * t₂)^2 + 4 * (1 + (sqrt 2 / 2) * t₂)^2 = 12) →
  (abs t₁ + abs t₂ = 12 * sqrt 2 / 7)

theorem main_proof : Prop :=
  curveA_cartesian ∧
  (∃ t : ℝ, curveB_parametric t) ∧
  intersection_and_distance

end intersection_and_distance_main_proof_l165_165538


namespace money_spent_on_video_games_l165_165509

def allowance := 50
def fraction_books := 1 / 4
def fraction_snacks := 1 / 5
def fraction_stationery := 1 / 10
def fraction_shoes := 3 / 10

theorem money_spent_on_video_games :
  let books := fraction_books * allowance,
      snacks := fraction_snacks * allowance,
      stationery := fraction_stationery * allowance,
      shoes := fraction_shoes * allowance,
      total_spent := books + snacks + stationery + shoes,
      video_games := allowance - total_spent
  in video_games = 7.5 :=
by
  sorry

end money_spent_on_video_games_l165_165509


namespace number_of_true_props_l165_165988

def prop1 (a x : ℝ) : Prop := (0 < a ∧ a < 1) → x < 0 → a^x > 1

def prop2 (a m n : ℝ) : Prop := (∀ x, y = log a (x - 1) + 1 → (2, 1)) → log m n = 0

def prop3 (y : ℝ → ℝ) : Prop := (y = λ x, x⁻¹) → ∀ x, x < 0 → decreasing_in y (set.Ioo (- ∞) 0) ∧ 0 < x → decreasing_in y (set.Ioo 0 (∞))

def prop4 (f g : ℝ → ℝ) : Prop := (f = λ x, 2^x ∧ g = λ x, log 2 x) → ∀ x, f (g x) = x ∧ g (f x) = x

theorem number_of_true_props (a m n : ℝ) (f g y: ℝ → ℝ) (x : ℝ) :
  prop1 a x ∧ prop2 a m n ∧ prop3 y ∧ prop4 f g → 
  (∃ correct_props : ℕ, correct_props = 3) :=
begin
  sorry
end

end number_of_true_props_l165_165988


namespace Mickey_mounts_per_week_l165_165339

theorem Mickey_mounts_per_week (days_in_week : ℕ) (Mickey_per_day : ℕ) :
  let M := days_in_week + 3 in
  let Mickey_per_day := 2 * M - 6 in
  days_in_week = 7 →
  Mickey_per_day * days_in_week = 98 :=
by
  intros days_in_week value_Mickey_per_day days_in_week_eq
  sorry

end Mickey_mounts_per_week_l165_165339


namespace sum_of_squares_of_perfect_squares_l165_165129

theorem sum_of_squares_of_perfect_squares (n : ℕ) (h : n < 1000) (hsq : ∃ k : ℕ, n = k^4) : 
  finset.sum (finset.filter (λ x, x < 1000 ∧ (∃ k : ℕ, x = k^4)) (finset.range 1000)) = 979 :=
by
  sorry

end sum_of_squares_of_perfect_squares_l165_165129


namespace budget_remaining_l165_165232

noncomputable def solve_problem : Nat :=
  let total_budget := 325
  let cost_flasks := 150
  let cost_test_tubes := (2 / 3 : ℚ) * cost_flasks
  let cost_safety_gear := (1 / 2 : ℚ) * cost_test_tubes
  let total_expenses := cost_flasks + cost_test_tubes + cost_safety_gear
  total_budget - total_expenses

theorem budget_remaining : solve_problem = 25 := by
  sorry

end budget_remaining_l165_165232


namespace f_monotonically_increasing_intervals_f_min_max_values_l165_165492

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2

theorem f_monotonically_increasing_intervals (k : ℤ) :
  ∀ x, k * real.pi - (5 * real.pi) / 12 ≤ x ∧ x ≤ k * real.pi + real.pi / 12 → 
  (f x = 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2) → 
  ∃ k : ℤ, (k * real.pi - (5 * real.pi) / 12 ≤ x ∧ x ≤ k * real.pi + real.pi / 12) := 
sorry

theorem f_min_max_values :
  ∀ x, x ∈ set.Icc (-real.pi / 3) (real.pi / 3) →
  (f x = 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2) →
  2 - sqrt 3 ≤ f x ∧ f x ≤ 4 :=
sorry

end f_monotonically_increasing_intervals_f_min_max_values_l165_165492
