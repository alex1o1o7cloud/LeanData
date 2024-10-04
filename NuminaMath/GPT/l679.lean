import Mathlib
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Modulo
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle
import Mathlib.LinearAlgebra.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Simpa
import Mathlib.Topology
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real

namespace sum_greater_than_zero_l679_679057

noncomputable def initial_values : List ℤ := [2, -2, 0]

def operation (x : ℤ) : ℤ :=
if x = 2 then (real.sqrt 2).to_int
else if x = -2 then (2^2)
else if x = 0 then 0
else x

def final_value (x : ℤ) : ℤ :=
match x with
| 2 := (real.sqrt 2).to_int
| -2 := 2^2
| 0 := 0
| _ := x

noncomputable def final_sum : ℤ :=
(final_value 2) + (final_value -2) + (final_value 0)

theorem sum_greater_than_zero : final_sum > 0 := by
sorry

end sum_greater_than_zero_l679_679057


namespace composite_proposition_l679_679345

-- Definitions based on the conditions
def prop_p : Prop := ∃ x : ℝ, x^2 - x + 2 < 0
def f (x : ℝ) : ℝ := 4 / x - log 3 x
def prop_q : Prop := ∀ x ∈ set.Ioo 3 4, f x ≠ 0

-- The main theorem
theorem composite_proposition : (¬ prop_p) ∧ (¬ prop_q) :=
  by sorry

end composite_proposition_l679_679345


namespace minimal_primes_ensuring_first_player_win_l679_679181

-- Define primes less than or equal to 100
def primes_le_100 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Define function to get the last digit of a number
def last_digit (n : Nat) : Nat := n % 10

-- Define function to get the first digit of a number
def first_digit (n : Nat) : Nat :=
  let rec first_digit_aux (m : Nat) :=
    if m < 10 then m else first_digit_aux (m / 10)
  first_digit_aux n

-- Define a condition that checks if a prime number follows the game rule
def follows_rule (a b : Nat) : Bool :=
  last_digit a = first_digit b

theorem minimal_primes_ensuring_first_player_win :
  ∃ (p1 p2 p3 : Nat),
  p1 ∈ primes_le_100 ∧
  p2 ∈ primes_le_100 ∧
  p3 ∈ primes_le_100 ∧
  follows_rule p1 p2 ∧
  follows_rule p2 p3 ∧
  p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
sorry

end minimal_primes_ensuring_first_player_win_l679_679181


namespace max_value_of_N_l679_679046

def N (a b c : ℕ) : ℕ := a * b * c + a * b + b * c + a - b - c

theorem max_value_of_N :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ {a, b, c} ⊆ {2, 3, 4, 5, 6} ∧
    ∀ (a' b' c' : ℕ), a' ≠ b' ∧ a' ≠ c' ∧ b' ≠ c' ∧ {a', b', c'} ⊆ {2, 3, 4, 5, 6} →
      N a b c ≥ N a' b' c' :=
begin
  use [6, 5, 4],
  split, norm_num,
  split, norm_num,
  split, norm_num,
  split,
  { simp [set.subset_def], intros, norm_num, },
  intros a' b' c' h₁ h₂ h₃ h₄,
  sorry
end

end max_value_of_N_l679_679046


namespace find_f_of_negative_one_l679_679331

def f : ℝ → ℝ
| x => if x > 0 then x^2 - 1 else f (x + 1) - 1

theorem find_f_of_negative_one : f (-1) = -2 :=
sorry

end find_f_of_negative_one_l679_679331


namespace similarity_triangle_statement_l679_679109

variables (A B C C1 A1 B1 A2 B2 C2 : Type)
variables (a b c k : ℝ) (dist_AC1 dist_C1B : ℝ) (dist_BA1 dist_A1C : ℝ) (dist_CB1 dist_B1A : ℝ)
variables (dist_A1C2 dist_C2B1 : ℝ) (dist_B1A2 dist_A2C1 : ℝ) (dist_C1B2 dist_B2A1 : ℝ)

-- Conditions for the first set of points
def condition1 := dist_AC1 / dist_C1B = k
def condition2 := dist_BA1 / dist_A1C = k
def condition3 := dist_CB1 / dist_B1A = k

-- Conditions for the second set of points
def condition4 := dist_A1C2 / dist_C2B1 = 1 / k
def condition5 := dist_B1A2 / dist_A2C1 = 1 / k
def condition6 := dist_C1B2 / dist_B2A1 = 1 / k

theorem similarity_triangle_statement :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 →
  similar (triangle A2 B2 C2) (triangle A1 B1 C1) ∧
  (similarity_coefficient (triangle A2 B2 C2) (triangle A1 B1 C1) = (k^2 - k + 1) / (k + 1)^2) :=
begin
  sorry
end

end similarity_triangle_statement_l679_679109


namespace four_p_plus_one_composite_l679_679499

theorem four_p_plus_one_composite (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_five : p ≥ 5) (h2p_plus1_prime : Nat.Prime (2 * p + 1)) : ¬ Nat.Prime (4 * p + 1) :=
sorry

end four_p_plus_one_composite_l679_679499


namespace simplify_expression_l679_679866

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 5 + 2) + 2 / (Real.sqrt 7 - 2))) = 
  (6 * Real.sqrt 7 + 9 * Real.sqrt 5 + 6) / (118 + 12 * Real.sqrt 35) :=
  sorry

end simplify_expression_l679_679866


namespace cattle_train_left_6_hours_before_l679_679977

theorem cattle_train_left_6_hours_before 
  (Vc : ℕ) (Vd : ℕ) (T : ℕ) 
  (h1 : Vc = 56)
  (h2 : Vd = Vc - 33)
  (h3 : 12 * Vd + 12 * Vc + T * Vc = 1284) : 
  T = 6 := 
by
  sorry

end cattle_train_left_6_hours_before_l679_679977


namespace train_speed_l679_679549

theorem train_speed
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (initial_distance : ℝ)
  (speed_train2_kmh : ℝ)
  (time_to_meet : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 100 ∧
  length_train2 = 200 ∧
  initial_distance = 840 ∧
  speed_train2_kmh = 72 ∧
  time_to_meet = 23.99808015358771 →
  (speed_train2 : ℝ) := speed_train2_kmh * (1 / 3.6) →
  (total_distance := initial_distance + length_train1 + length_train2) →
  (total_distance = (speed_train1 + speed_train2) * time_to_meet) →
  speed_train1 = 27.5 :=
by
  sorry

end train_speed_l679_679549


namespace phantom_additional_money_needed_l679_679112

theorem phantom_additional_money_needed
  (given_money : ℕ := 50)
  (cost_black : ℕ := 11)
  (cost_red : ℕ := 15)
  (cost_yellow : ℕ := 13)
  (num_black : ℕ := 2)
  (num_red : ℕ := 3)
  (num_yellow : ℕ := 2) :
  let total_cost := num_black * cost_black + num_red * cost_red + num_yellow * cost_yellow
  in total_cost - given_money = 43 := 
by 
  let total_cost : ℕ := num_black * cost_black + num_red * cost_red + num_yellow * cost_yellow;
  have h1 : total_cost = 93 := by sorry;
  have h2 : total_cost - given_money = 43 := by
    rw [h1];
    norm_num;
  exact h2

end phantom_additional_money_needed_l679_679112


namespace domain_of_function_l679_679140

theorem domain_of_function :
  {x : ℝ | (2 - x ≥ 0) ∧ (x + 3 ≠ 0)} = {x : ℝ | x ∈ (Iic 2 \ { -3 })} :=
by
  sorry

end domain_of_function_l679_679140


namespace sum_of_values_l679_679152

theorem sum_of_values (N : ℝ) (h : N * (N + 4) = 8) : N + (4 - N - 8 / N) = -4 := 
sorry

end sum_of_values_l679_679152


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679477

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679477


namespace necessary_condition_a_eq_b_l679_679106

theorem necessary_condition_a_eq_b (a b : ℕ)
  (h : ∃ f : ℕ → Prop, ∀ n, (f n ↔ (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) ∧ ∀ m, ∃ n > m, f n) :
  a = b :=
begin
  sorry
end

end necessary_condition_a_eq_b_l679_679106


namespace trapezoid_median_length_l679_679131

theorem trapezoid_median_length
  (h: ℝ) (A_triangle: ℝ) (A_trapezoid: ℝ) (median: ℝ)
  (height_trapezoid_eq_height_triangle : h = 8 * real.sqrt 3)
  (triangle_area : A_triangle = 64 * real.sqrt 3)
  (trapezoid_area_eq_3_times_triangle : A_trapezoid = 3 * A_triangle)
  (trapezoid_area : A_trapezoid = h * median)
  : median = 24 := sorry

end trapezoid_median_length_l679_679131


namespace area_of_intersection_l679_679175

def rectangle_vertices : List (ℝ × ℝ) := [(5, 11), (16, 11), (16, -2), (5, -2)]
def ellipse_eq (x y : ℝ) : Prop := (x - 5)^2 / 16 + (y + 2)^2 / 4 = 1

theorem area_of_intersection :
  let vertices := rectangle_vertices in
  let ellipse := ellipse_eq in
  ∃ v1 v2 v3 v4 ∈ vertices,
    v1 = (5, 11) ∧ v2 = (16, 11) ∧ v3 = (16, -2) ∧ v4 = (5, -2) ∧
    ∀ x y, ellipse x y → 
    y ∈ Icc (-2) 11 ∧ x ∈ Icc 5 16 →
    (1 / 4) * (π * 4 * 2) = 2 * π := 
sorry

end area_of_intersection_l679_679175


namespace second_machine_rate_is_55_l679_679588

-- Define constants for the rates of the machines and total copies made
constants (first_machine_rate second_machine_rate total_copies half_hour_minutes : ℕ)

-- Set the known values for first machine and time period
def first_machine_rate := 40
def half_hour_minutes  := 30
def total_copies := 2850

-- Introduce the theorem to prove the second machine rate
theorem second_machine_rate_is_55 (x : ℕ) :
  x = (total_copies - first_machine_rate * half_hour_minutes) / half_hour_minutes :=
by
  -- The rate of the second machine can be computed by subtracting the 
  -- copies made by the first machine in 30 minutes from the total copies
  -- and then dividing by 30 minutes.
  sorry

end second_machine_rate_is_55_l679_679588


namespace part1_correct_part2_correct_l679_679512

noncomputable theory -- Enabling noncomputable theory for calculations

-- Part 1: Prove that C's calculation is correct
def linear_regression_correct (t : List ℕ) (y : List ℝ) : Prop :=
  let linear_equa_C : ℕ → ℝ := λ t, 0.5 * t + 2.3 in
  let calculated_y := t.map linear_equa_C in
  ∀ k : ℕ, k ∈ [1, 2, 3, 4, 5, 6, 7] → y.nth_le k sorry = calculated_y.nth_le k sorry

-- Part 2: Prove expected number of perfect data points is 2/3
def expected_perfect_data (t : List ℕ) (y : List ℝ) : Prop :=
  let linear_equa_C : ℕ → ℝ := λ t, 0.5 * t + 2.3 in
  let estimated_y := t.map linear_equa_C in
  let errors := (List.zip y estimated_y).map (λ p, (p.1 - p.2).abs) in
  let perfect_data := errors.filter (λ e, e = 0) in
  let remaining_data := errors.filter (λ e, e ≤ 0.1 ∧ e ≠ 0) in
  let prob_X_eq_0 := (remaining_data.length.choose 2).toFloat / ((remaining_data.length + perfect_data.length).choose 2).toFloat in
  let prob_X_eq_1 := (perfect_data.length * remaining_data.length).toFloat / ((remaining_data.length + perfect_data.length).choose 2).toFloat in
  let prob_X_eq_2 := (perfect_data.length.choose 2).toFloat / ((remaining_data.length + perfect_data.length).choose 2).toFloat in
  let expec_X := prob_X_eq_1 + 2 * prob_X_eq_2 in
  expec_X = 2 / 3

theorem part1_correct : ∀ (t y : List ℕ), t = [1, 2, 3, 4, 5, 6, 7] ∧ y = [2.9, 3.3, 3.6, 4.4, 4.8, 5.2, 5.9] → linear_regression_correct t y :=
sorry

theorem part2_correct : ∀ (t y : List ℕ), t = [1, 2, 3, 4, 5, 6, 7] ∧ y = [2.9, 3.3, 3.6, 4.4, 4.8, 5.2, 5.9] → expected_perfect_data t y :=
sorry

end part1_correct_part2_correct_l679_679512


namespace square_length_CE1_l679_679666

noncomputable def side_length : ℝ := real.sqrt 144
noncomputable def side_length_abc := 12
noncomputable def triangle_ABC := 12
noncomputable def bd1 := real.sqrt 16

theorem square_length_CE1 :
  ∀ (ABC AD1E1 : Triangle) (BD1 : ℝ),
    side_length = 12 →
    congruent AD1E1 ABC →
    BD1 = 4 →
    ∃ CE1 : ℝ, CE1^2 = 144 :=
by
  sorry

end square_length_CE1_l679_679666


namespace range_of_x0_l679_679442

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 2) ^ x - 1 else real.sqrt x

theorem range_of_x0 (x_0 : ℝ) : f x_0 > 1 ↔ x_0 > 1 ∨ x_0 < -1 :=
by
  sorry

end range_of_x0_l679_679442


namespace multiples_of_15_between_17_152_l679_679751

theorem multiples_of_15_between_17_152 : 
  let first_multiple := 30 in
  let last_multiple := 150 in
  let common_difference := 15 in
  let num_terms := (last_multiple - first_multiple) / common_difference + 1 in
  num_terms = 9 := 
by
  let first_multiple := 30
  let last_multiple := 150
  let common_difference := 15
  let num_terms := (last_multiple - first_multiple) / common_difference + 1
  show num_terms = 9, from
  sorry

end multiples_of_15_between_17_152_l679_679751


namespace jenny_run_distance_l679_679421

theorem jenny_run_distance (walk_distance : ℝ) (ran_walk_diff : ℝ) (h_walk : walk_distance = 0.4) (h_diff : ran_walk_diff = 0.2) :
  (walk_distance + ran_walk_diff) = 0.6 :=
sorry

end jenny_run_distance_l679_679421


namespace printer_diff_l679_679950

theorem printer_diff (A B : ℚ) (hA : A * 60 = 35) (hAB : (A + B) * 24 = 35) : B - A = 7 / 24 := by
  sorry

end printer_diff_l679_679950


namespace proof_problem_l679_679740

-- Defining lines l1, l2, l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l2 (x y : ℝ) : Prop := 2 * x + y = -2
def l3 (x y : ℝ) : Prop := x - 2 * y = 1

-- Point P being the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Definition of the first required line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Definition of the second required line passing through P and perpendicular to l3
def required_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- The theorem to prove
theorem proof_problem :
  (∃ x y, l1 x y ∧ l2 x y ∧ (x, y) = P) →
  (∀ x y, (x, y) = P → line_through_P_and_origin x y) ∧
  (∀ x y, (x, y) = P → required_line x y) :=
by
  sorry

end proof_problem_l679_679740


namespace probability_neither_red_nor_green_l679_679810

-- Define the contents of each bag
def bag1 := (5, 6, 7)  -- (green, black, red)
def bag2 := (3, 4, 8)  -- (green, black, red)
def bag3 := (2, 7, 5)  -- (green, black, red)

-- Total pens in each bag
def total_pens_bag1 : ℕ := bag1.1 + bag1.2 + bag1.3
def total_pens_bag2 : ℕ := bag2.1 + bag2.2 + bag2.3
def total_pens_bag3 : ℕ := bag3.1 + bag3.2 + bag3.3

-- Probability of picking a black pen from each bag
def prob_black_bag1 : ℚ := bag1.2 / total_pens_bag1
def prob_black_bag2 : ℚ := bag2.2 / total_pens_bag2
def prob_black_bag3 : ℚ := bag3.2 / total_pens_bag3

-- Total number of pens across all bags
def total_pens : ℕ := total_pens_bag1 + total_pens_bag2 + total_pens_bag3

-- Weighted probability of picking a black pen
def weighted_prob_black : ℚ :=
  (prob_black_bag1 * total_pens_bag1 / total_pens) +
  (prob_black_bag2 * total_pens_bag2 / total_pens) +
  (prob_black_bag3 * total_pens_bag3 / total_pens)

-- Theorem statement
theorem probability_neither_red_nor_green :
  weighted_prob_black = 17/47 :=
sorry

end probability_neither_red_nor_green_l679_679810


namespace alien_heads_l679_679628

theorem alien_heads (l o : ℕ) 
  (h1 : l + o = 60) 
  (h2 : 4 * l + o = 129) : 
  l + 2 * o = 97 := 
by 
  sorry

end alien_heads_l679_679628


namespace min_reciprocal_sum_l679_679709

theorem min_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : 
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_reciprocal_sum_l679_679709


namespace basketball_transportation_feasibility_l679_679397

-- Definitions
variable {Person : Type} [Inhabited Person]
variable (height : Person → ℝ)
variable (neighborhood : Person → set Person)
variable (play_basketball : Person → Prop)
variable (free_transportation : Person → Prop)

-- Proof Problem Statement
theorem basketball_transportation_feasibility (persons : fin 10) :
  (∀ p, play_basketball p ↔ (∃ r₁, card {n ∈ neighborhood p | height p > height n} > card {n ∈ neighborhood p | height p <= height n})) ∧
  (∀ p, free_transportation p ↔ (∃ r₂, card {n ∈ neighborhood p | height p < height n} > card {n ∈ neighborhood p | height p >= height n})) →
  (∃ S1 S2 : finset Person, S1.card ≥ 9 ∧ S2.card ≥ 9 ∧ 
  (∀ p ∈ S1, play_basketball p) ∧ 
  (∀ p ∈ S2, free_transportation p)) :=
sorry

end basketball_transportation_feasibility_l679_679397


namespace constant_term_in_expansion_l679_679189

theorem constant_term_in_expansion :
  let f := λ x : ℝ, (x^(1/2) + (3 / x))^12 in
  binomial (12, 8) * 81 = 40095 :=
by
  sorry

end constant_term_in_expansion_l679_679189


namespace right_triangle_legs_l679_679737

theorem right_triangle_legs (R r : ℝ) : 
  ∃ a b : ℝ, a = Real.sqrt (2 * (R^2 + r^2)) ∧ b = Real.sqrt (2 * (R^2 - r^2)) :=
by
  sorry

end right_triangle_legs_l679_679737


namespace smallest_n_common_factor_l679_679193

theorem smallest_n_common_factor :
  ∃ n : ℤ, n > 0 ∧ (gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 10 :=
by
  sorry

end smallest_n_common_factor_l679_679193


namespace prove_shape_of_set_X_l679_679858

variables {n : ℕ} (k : fin n → ℝ) (a b : fin n → ℝ) (c x y : ℝ)

def A_i_X_sq (i : fin n) : ℝ := (x - a i) ^ 2 + (y - b i) ^ 2

def sum_A_i_X_sq : ℝ := ∑ i, k i * A_i_X_sq k a b x y i

def alpha : ℝ := ∑ i, k i
def beta : ℝ := ∑ i, k i * a i
def gamma : ℝ := ∑ i, k i * b i
def delta : ℝ := ∑ i, k i * ((a i) ^ 2 + (b i) ^ 2)

theorem prove_shape_of_set_X :
  (sum_A_i_X_sq k a b x y = c) →
  ((alpha k = 0 ∧ (∃ β γ : ℝ, β * x + γ * y = (delta k a b - c) / 2)) ∨
   (alpha k ≠ 0 ∧
      (∃ r : ℝ,
         r = (beta k a b)^2 + (gamma k a b)^2 - (alpha k) * (delta k a b - c) / (alpha k * alpha k) ∧
           (r > 0 ∧ ∃ cx cy : ℝ, (x - cx)^2 + (y - cy)^2 = r) ∨ r ≤ 0))) :=
by sorry

end prove_shape_of_set_X_l679_679858


namespace calculate_r_over_s_at_2_l679_679650

theorem calculate_r_over_s_at_2 (c : ℝ) :
  let r := λ x : ℝ, c * x
  let s := λ x : ℝ, (x + 2) * (x - 3)
  (h_asymptotes : ∀ x, ¬ ∃ (y : ℝ), y = (r x) / (s x) ∧ (x = -2 ∨ x = 3))
  (h_passing_points_0_0 : r 0 / s 0 = 0)
  (h_passing_points_1_neg2 : r 1 / s 1 = -2) :=
  r 2 / s 2 = -6 :=
begin
  sorry
end

end calculate_r_over_s_at_2_l679_679650


namespace triangle_ABC_angles_l679_679407

theorem triangle_ABC_angles (A B C K L : Type)
    [inhabited A] [inhabited B] [inhabited C] [inhabited K] [inhabited L]
    (ABC_right : ∠ BAC = 90)
    (perpendicular_bisector_BC_intersects_AC_at_K : ∀ P: Type, is_perpendicular bisector P B C → (is_line_segment A C) P K)
    (perpendicular_bisector_BK_intersects_AB_at_L : ∀ Q: Type, is_perpendicular bisector Q B K → (is_line_segment A B) Q L)
    (CL_bisects_∠ACB : bisector C L (∠ ACB)):
    (∃ (angle_ABC angle_ACB : ℝ), (angle_ABC = 54 ∧ angle_ACB = 36) ∨ (angle_ABC = 30 ∧ angle_ACB = 60)) :=
sorry

end triangle_ABC_angles_l679_679407


namespace prime_sum_of_composites_l679_679564

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def can_be_expressed_as_sum_of_two_composites (p : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ p = a + b

theorem prime_sum_of_composites :
  can_be_expressed_as_sum_of_two_composites 13 ∧ 
  ∀ p : ℕ, is_prime p ∧ p > 13 → can_be_expressed_as_sum_of_two_composites p :=
by 
  sorry

end prime_sum_of_composites_l679_679564


namespace monkey_height_37_minutes_l679_679042

noncomputable def monkey_climb (minutes : ℕ) : ℕ :=
if minutes = 37 then 60 else 0

theorem monkey_height_37_minutes : (monkey_climb 37) = 60 := 
by
  sorry

end monkey_height_37_minutes_l679_679042


namespace decimal_to_binary_125_l679_679925

theorem decimal_to_binary_125 : 
  (∃ (digits : list ℕ), (125 = digits.length - 1).sum (λ i, (2 ^ i) * digits.nth_le i sorry) ∧ digits = [1,1,1,1,1,0,1]) := 
sorry

end decimal_to_binary_125_l679_679925


namespace first_year_after_2020_with_digit_sum_5_l679_679050

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + ((n / 1000) % 10)

theorem first_year_after_2020_with_digit_sum_5 : ∃ y : ℕ, y > 2020 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, (z > 2020 ∧ z < y) → sum_of_digits z ≠ 5 :=
by
  use 2021
  split
  · exact lt_add_one 2020
  split
  · dsimp [sum_of_digits]
    norm_num
  · intros z hz
    dsimp [sum_of_digits] at hz
    sorry

end first_year_after_2020_with_digit_sum_5_l679_679050


namespace limit_solution_l679_679273

noncomputable def limit_problem (x : ℝ) : ℝ := (exp x + exp (-x) - 2) / (sin x)^2

theorem limit_solution :
  (real.tendsto limit_problem (nhds 0) (nhds 1)) :=
by
  sorry

end limit_solution_l679_679273


namespace find_derivative_l679_679352

theorem find_derivative (f : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = 2 * x + 2 * deriv f 1)
  (h_def : f = λ x, x^2 + 2 * x * deriv f 1) : deriv f 1 = -2 :=
sorry

end find_derivative_l679_679352


namespace number_of_alternating_subsets_l679_679205

-- Definition of the set S.
def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 100}

-- Definition of an alternating subset.
def is_alternating_subset (subset : Finset ℕ) : Prop :=
  subset.card = 10 ∧ ∃ (l : List ℕ), l.to_finset = subset ∧
  (∀ i < 9, (l.nth_le i _).odd ≠ (l.nth_le (i + 1) _).odd)

-- Main statement to prove
theorem number_of_alternating_subsets :
  (Finset.filter is_alternating_subset (S.powerset 10)).card = Nat.choose 55 10 + Nat.choose 54 10 := 
sorry

end number_of_alternating_subsets_l679_679205


namespace fruit_salad_cherries_l679_679229

theorem fruit_salad_cherries (b r g c : ℕ) 
  (h1 : b + r + g + c = 390)
  (h2 : r = 3 * b)
  (h3 : g = 2 * c)
  (h4 : c = 5 * r) :
  c = 119 :=
by
  sorry

end fruit_salad_cherries_l679_679229


namespace cos_sum_lt_sum_cos_l679_679762

theorem cos_sum_lt_sum_cos (α β : ℝ) (h1 : 0 < α) (h2 : α < real.pi / 2) (h3 : 0 < β) (h4 : β < real.pi / 2) : 
  real.cos (α + β) < real.cos α + real.cos β := sorry

end cos_sum_lt_sum_cos_l679_679762


namespace shorter_train_length_proof_l679_679912

variables {speed_train1 speed_train2 : ℝ} 
variables {length_longer_train time_to_clear : ℝ}

-- Given conditions
def shorter_train_length (speed_train1 speed_train2 length_longer_train time_to_clear : ℝ) : ℝ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let total_clear_distance := relative_speed * time_to_clear
  let length_shorter_train := total_clear_distance - length_longer_train
  length_shorter_train

-- The target statement to be proven
theorem shorter_train_length_proof : 
  shorter_train_length 80 55 165 7.626056582140095 ≈ 120.98 :=
by 
  sorry

end shorter_train_length_proof_l679_679912


namespace distance_between_Sasha_and_Kolya_l679_679471

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679471


namespace general_term_formula_exists_sum_first_n_bn_l679_679701

-- Define the sequence and initial conditions
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

axiom a_10 : ∀ (a : ℕ → ℝ) (d : ℝ), a 10 = 21
axiom S_10 : ∀ (a : ℕ → ℝ) (d : ℝ), Sn a 10 = 120

-- The general term formula
theorem general_term_formula_exists (a : ℕ → ℝ) (a1: ℝ) (d: ℝ) : 
  (∃ a1 d, a = λ n, a1 + (n - 1) * d) → a 10 = 21 → Sn a 10 = 120 → 
  ∀ n, a n = 2 * n + 1 := 
sorry

-- Define bn and Tn
def bn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / (a n * a (n + 1)) + 1

noncomputable def Tn (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b i

-- The sum of the first n terms for the sequence bn
theorem sum_first_n_bn (a : ℕ → ℝ) (b: ℕ → ℝ) (n: ℕ) : 
  (∀ n, b n = 1 / (a n * a (n + 1)) + 1) → 
  (∀ n, a n = 2 * n + 1) →
  Tn b n = (n / (6 * n + 9)) + n :=
sorry

end general_term_formula_exists_sum_first_n_bn_l679_679701


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679005

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679005


namespace find_lambda_l679_679343

variables {m n : Vector ℝ 2} {λ : ℝ} (h1 : ‖m‖ ≠ 0) (h2 : ‖n‖ ≠ 0) (h3 : angle m n = π / 3)
          (h4 : ‖n‖ = λ * ‖m‖) (hλ : λ > 0)

theorem find_lambda (h_min : ∀ x1 y1 x2 y2 x3 y3, 
  x1 ∈ {m, n, n} → x2 ∈ {m, n, n} → x3 ∈ {m, n, n} →
  y1 ∈ {m, m, n} → y2 ∈ {m, m, n} → y3 ∈ {m, m, n} →
  (x1 • y1 + x2 • y2 + x3 • y3) ≥ 4 * ‖m‖^2) : λ = 8 / 3 :=
by
  sorry

end find_lambda_l679_679343


namespace f_x_plus_3_odd_l679_679886

noncomputable def f : ℝ → ℝ := sorry

-- Defining the conditions
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given conditions
axiom f_dom : ∀ x : ℝ, f x ∈ ℝ
axiom f_x_plus_1_odd : is_odd (λ x, f (x + 1))
axiom f_x_minus_1_odd : is_odd (λ x, f (x - 1))

-- The goal is to prove that f(x + 3) is an odd function
theorem f_x_plus_3_odd : is_odd (λ x, f (x + 3)) :=
sorry

end f_x_plus_3_odd_l679_679886


namespace collinear_M_R_Q_l679_679811

variables {α : Type*} [euclidean_geometry α]

open euclidean_geometry

-- Given definitions and conditions
variables {A B C D E H M P Q N O1 O2 F G R : α} 

def triangle_ABC (A B C : α) : Prop := 
    ∃ (triangle : triangle α), 
    triangle.A = A ∧ triangle.B = B ∧ triangle.C = C

def internal_angles_C_angle (A B C : α) : Prop := 
    angle C = 3 * (angle A - angle B)

def feet_of_altitudes (A C A B : α) (D E : α) : Prop := 
    height D (from A to segment AC) ∧ height E (from A to segment AB)

def orthocenter (H : α) (A B C : α) : Prop := 
    ∃ (H : orthocenter α), H = orthocenter of triangle ABC

def midpoint_M (M : α) (A B : α) : Prop :=
    mid_point M of segment AB

def point_on_circumcircle_P (P : α) (A B C : α) : Prop :=
    ∃ (P : point α), P lies on circumcircle of triangle ABC on arc BC not containing A

def midpoint_Q (Q : α) (H P : α) : Prop :=
    mid_point Q of segment HP

def line_intersection_N (N : α) (DM EQ : line α) : Prop :=
    N is intersection of line DM and line EQ

def circumcenter_O1 (O1 : α) (D E N : α) : Prop :=
    circumcenter O1 of triangle DEN

def circumcenter_O2 (O2 : α) (D N Q : α) : Prop :=
    circumcenter O2 of triangle DNQ

def intersection_at_F (F : α) (O1 O2 DE : line α) : Prop :=
    F is intersection of line O1O2 and DE

def perpendicular_at_F (G F : α) (O1 O2 N : line α) : Prop :=
    G lies on extension of O2N and perpendicular from F to O1O2

def circles_intersect_at_R (R F G O2 : α) : Prop :=
    ∃ circle(1), circle(1) centered at G with radius GF
    ∃ circle(2), circle(2) centered at O2 with radius O2F
    R is intersection of circle(1) and circle(2) other than F.

-- Theorem to be proven
theorem collinear_M_R_Q (A B C D E H M P Q N O1 O2 F G R : α) :
  triangle_ABC A B C ∧ 
  internal_angles_C_angle A B C ∧ 
  feet_of_altitudes A C A B D E ∧ 
  orthocenter H A B C ∧ 
  midpoint_M M A B ∧ 
  point_on_circumcircle_P P A B C ∧ 
  midpoint_Q Q H P ∧ 
  line_intersection_N N (line_through D M) (line_through E Q) ∧ 
  circumcenter_O1 O1 D E N ∧ 
  circumcenter_O2 O2 D N Q ∧ 
  intersection_at_F F (line_through O1 O2) (line_through D E) ∧ 
  perpendicular_at_F G F (line_through O1 O2) (line_through O2 N) ∧ 
  circles_intersect_at_R R F G O2
  →
  collinear M R Q := sorry

end collinear_M_R_Q_l679_679811


namespace find_A_plus_B_l679_679445

theorem find_A_plus_B (A B : ℚ) (h : ∀ x : ℚ, x ≠ 4 → x ≠ 5 → (Bx - 15) / (x^2 - 9x + 20) = A / (x - 4) + 4 / (x - 5)) : A + B = -5/2 :=
sorry

end find_A_plus_B_l679_679445


namespace three_powers_in_two_digit_range_l679_679000

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l679_679000


namespace find_a_l679_679147

theorem find_a (a : ℝ) (h : ∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) 0 → f x = x^3 - 3 * x + a) 
  (h_min : ∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) 0 → f x ≥ 1) : a = 3 := 
sorry

end find_a_l679_679147


namespace completed_shape_perimeter_602_l679_679850

noncomputable def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

noncomputable def total_perimeter_no_overlap (n : ℕ) (length width : ℝ) : ℝ :=
  n * rectangle_perimeter length width

noncomputable def total_reduction (n : ℕ) (overlap : ℝ) : ℝ :=
  (n - 1) * overlap

noncomputable def overall_perimeter (n : ℕ) (length width overlap : ℝ) : ℝ :=
  total_perimeter_no_overlap n length width - total_reduction n overlap

theorem completed_shape_perimeter_602 :
  overall_perimeter 100 3 1 2 = 602 :=
by
  sorry

end completed_shape_perimeter_602_l679_679850


namespace number_of_female_students_l679_679959

-- Given conditions
variables (F : ℕ)

-- The average score of all students (90)
def avg_all_students := 90
-- The total number of male students (8)
def num_male_students := 8
-- The average score of male students (87)
def avg_male_students := 87
-- The average score of female students (92)
def avg_female_students := 92

-- We want to prove the following statement
theorem number_of_female_students :
  num_male_students * avg_male_students + F * avg_female_students = (num_male_students + F) * avg_all_students →
  F = 12 :=
sorry

end number_of_female_students_l679_679959


namespace guess_number_three_questions_l679_679036

theorem guess_number_three_questions (n : ℕ) (h : 1 ≤ n ∧ n ≤ 8) : ∃ (questions : list (ℕ → bool)), questions.length = 3 ∧ (∀ possible_number, (1 ≤ possible_number ∧ possible_number ≤ 8) → ∃ unique_number, ∀ q in questions, q unique_number = q possible_number) :=
begin
  sorry
end

end guess_number_three_questions_l679_679036


namespace find_length_BI_l679_679906

noncomputable def triangle (A B C I : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space I] :=
  ∃ (AB AC BC BI: ℝ), 
    AB = 31 ∧ 
    AC = 29 ∧ 
    BC = 30 ∧ 
    is_incenter I A B C ∧ 
    metric.dist B I = 17.22

theorem find_length_BI (A B C I : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space I]
  (h : triangle AB AC BC BI) : 
  metric.dist B I = 17.22 :=
sorry

end find_length_BI_l679_679906


namespace prove_odd_function_definition_l679_679384

theorem prove_odd_function_definition (f : ℝ → ℝ) 
  (odd : ∀ x : ℝ, f (-x) = -f x)
  (pos_def : ∀ x : ℝ, 0 < x → f x = 2 * x ^ 2 - x + 1) :
  ∀ x : ℝ, x < 0 → f x = -2 * x ^ 2 - x - 1 :=
by
  intro x hx
  sorry

end prove_odd_function_definition_l679_679384


namespace correct_statements_about_population_and_sample_l679_679396

def students_population : Nat := 70000
def students_sample : Nat := 1000

theorem correct_statements_about_population_and_sample (A B C D : Prop) :
  (students_sample < students_population) →
  (∀ x : Nat, x ∈ fin students_sample → x < students_population) →
  (students_population ∈ fin students_population) →
  (students_sample ∈ fin students_sample) →
  A → B → C → D → 
  (A ∧ B ∧ C ∧ D) :=
by
  intro h1 h2 h3 h4 ha hb hc hd
  exact ⟨ha, hb, hc, hd⟩

end correct_statements_about_population_and_sample_l679_679396


namespace triangle_perimeter_l679_679209

theorem triangle_perimeter (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 20) (h₃ : c = 30) : a + b + c = 55 := 
by
  rw [h₁, h₂, h₃]
  rfl

end triangle_perimeter_l679_679209


namespace lucky_card_probability_l679_679622

-- Define the elements
def total_combinations := 10^4
def non_lucky_combinations := 8^4
def probability_non_lucky := non_lucky_combinations / total_combinations
def probability_lucky := 1 - probability_non_lucky

-- Theorem statement
theorem lucky_card_probability : probability_lucky = 0.5904 := by
  sorry

end lucky_card_probability_l679_679622


namespace hiker_speed_calculation_l679_679231

theorem hiker_speed_calculation :
  ∃ (h_speed : ℝ),
    let c_speed := 10
    let c_time := 5.0 / 60.0
    let c_wait := 7.5 / 60.0
    let c_distance := c_speed * c_time
    let h_distance := c_distance
    h_distance = h_speed * c_wait ∧ h_speed = 10 * (5 / 7.5) := by
  sorry

end hiker_speed_calculation_l679_679231


namespace area_of_wall_l679_679238

-- Definitions based on conditions
-- Length to width ratios
def ratio_small: ℕ := 2
def ratio_regular_jumbo: ℕ := 3

-- Percentages of tiles
def percent_small: ℚ := 2/5
def percent_regular: ℚ := 3/10
def percent_jumbo: ℚ := 1/4

-- Coverage areas
def area_regular_tiles: ℚ := 90 -- sq ft

-- Lengths
def length_jumbo_to_regular: ℕ := 3

-- Theorem to prove total wall area
theorem area_of_wall 
    (ratio_small : ℕ = 2)
    (ratio_regular_jumbo : ℕ = 3)
    (percent_small : ℚ = 2/5)
    (percent_regular : ℚ = 3/10)
    (percent_jumbo : ℚ = 1/4)
    (area_regular_tiles : ℚ = 90)
    (length_jumbo_to_regular : ℕ = 3)
    (A : ℚ) :
    A = 300 :=
sorry

end area_of_wall_l679_679238


namespace time_to_pass_faster_train_l679_679913

noncomputable def speed_slower_train_kmph : ℝ := 36
noncomputable def speed_faster_train_kmph : ℝ := 45
noncomputable def length_faster_train_m : ℝ := 225.018
noncomputable def kmph_to_mps_factor : ℝ := 1000 / 3600

noncomputable def relative_speed_mps : ℝ := (speed_slower_train_kmph + speed_faster_train_kmph) * kmph_to_mps_factor

theorem time_to_pass_faster_train : 
  (length_faster_train_m / relative_speed_mps) = 10.001 := 
sorry

end time_to_pass_faster_train_l679_679913


namespace number_of_valid_n_l679_679326

theorem number_of_valid_n : 
  {n : ℤ | ∃ k, 3200 = k * (5:ℤ) ^ n ∧ k ≠ 0 ∧ (3200 * (4:ℝ) ^ n * (5:ℝ) ^ -n) ∈ (Set.Icc (0 : ℝ) ∞)}.size = 6 := 
by
  sorry

end number_of_valid_n_l679_679326


namespace percent_increase_in_pizza_area_l679_679772

theorem percent_increase_in_pizza_area (r : ℝ) (h : 0 < r) :
  let r_large := 1.10 * r
  let A_medium := π * r^2
  let A_large := π * r_large^2
  let percent_increase := ((A_large - A_medium) / A_medium) * 100 
  percent_increase = 21 := 
by sorry

end percent_increase_in_pizza_area_l679_679772


namespace find_a_l679_679324

noncomputable def star (a b : ℝ) := a * (a + b) + b

theorem find_a (a : ℝ) (h : star a 2.5 = 28.5) : a = 4 ∨ a = -13/2 := 
sorry

end find_a_l679_679324


namespace no_roots_inside_circle_l679_679823

noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

theorem no_roots_inside_circle {k : ℕ} (n : Fin k → ℕ) (h : ∀ i j : Fin k, i < j → n i < n j) :
  ∀ z : ℂ, abs z < φ → (1 + ∑ i, z^(n i)) ≠ 0 := 
sorry

end no_roots_inside_circle_l679_679823


namespace chessboard_problem_chessboard_problem_n7_l679_679060
noncomputable def r (n : ℕ) : ℕ := sorry

theorem chessboard_problem (n : ℕ) : r(n) ≤ (n + n * Real.sqrt (4 * n - 3)) / 2 := sorry

theorem chessboard_problem_n7 : r 7 ≤ 21 := sorry

end chessboard_problem_chessboard_problem_n7_l679_679060


namespace minimize_elements_in_A_l679_679045

def SetA (k : ℝ) : Set ℤ := 
  {x : ℤ | (k * x - k ^ 2 - 6) * (x - 4) > 0}

def minimized_set_elements (k : ℝ) (A : Set ℤ) : Prop :=
  ∃ m n : ℤ, SetA k = A ∧ m = -3 ∧ n = -2
  
theorem minimize_elements_in_A : ∀ k : ℝ, 
  (SetA k).finite → minimized_set_elements k (SetA k) :=
by 
  intro k
  sorry

end minimize_elements_in_A_l679_679045


namespace equation_of_parabola_and_point_l679_679333

-- Given conditions

-- Condition 1: parabola C defined by y^2 = 2px (p > 0)
def parabola (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) :=
  {xy | xy.snd ^ 2 = 2 * p * xy.fst}

-- Condition 2: line with slope 2√2 intersects parabola at A(x1, y1) and B(x2, y2)
def intersect_points (p : ℝ) (p_pos : p > 0) : Set (ℝ × ℝ) :=
  {xy | xy ∈ parabola p p_pos ∧ xy.snd = 2 * Real.sqrt 2 * (xy.fst - p / 2)}

-- Condition 3: distance |AB| = 9/2
def distance_AB (x1 x2 : ℝ) : Prop :=
  |x2 - x1| = 9 / 2

-- Main theorem: equation of parabola and point D
theorem equation_of_parabola_and_point D_on_parabola (x y p : ℝ) (p_pos : p > 0)
  (h_parabola : parabola p p_pos (x, y))
  (h_distance_AB : ∀ (x1 x2 : ℝ), (x1, x1 * 2 * Real.sqrt 2 - p / 2) ∈ intersect_points p p_pos ∧ (x2, x2 * 2 * Real.sqrt 2 - p / 2) ∈ intersect_points p p_pos → distance_AB x1 x2)
  (D_on_line_shortest : ∀ (D : ℝ × ℝ), D ∈ parabola p p_pos → (|D.fst - D.snd + 3| / Real.sqrt 2 ≤ (|x - y + 3| / Real.sqrt 2))) :
  (∀ (p : ℝ), p > 0 → ∀ (x y : ℝ), y^2 = 2 * p * x = (y^2 = 4 * x)) ∧ (x, y) = (1, 2) :=
sorry

end equation_of_parabola_and_point_l679_679333


namespace arithmetic_sequence_value_l679_679796

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end arithmetic_sequence_value_l679_679796


namespace max_Q_value_l679_679325

noncomputable def Q (b : ℝ) : ℝ :=
  let uniform_x := measure_theory.measure_space.quasi_measure_preserving.measureable_set [0, b]
  let uniform_y := measure_theory.measure_space.quasi_measure_preserving.measureable_set [0, 2*b]
  \mathbb{P} (λ p : ℝ × ℝ, sin(π * p.1) * sin(π * p.2) > 0.5)

theorem max_Q_value : ∀ b ∈ set.Icc (0 : ℝ) (0.5 : ℝ), Q b ≤ 1/16 :=
sorry

end max_Q_value_l679_679325


namespace final_tank_volume_l679_679617

-- Definitions based on the conditions
def initially_liters : ℕ := 6000
def evaporated_liters : ℕ := 2000
def drained_by_bob : ℕ := 3500
def rain_minutes : ℕ := 30
def rain_interval : ℕ := 10
def rain_rate : ℕ := 350

-- Theorem statement
theorem final_tank_volume:
  let remaining_water := initially_liters - evaporated_liters - drained_by_bob,
      rain_cycles := rain_minutes / rain_interval,
      added_by_rain := rain_cycles * rain_rate,
      final_volume := remaining_water + added_by_rain
  in final_volume = 1550 := by {
  sorry
}

end final_tank_volume_l679_679617


namespace percentage_distance_l679_679985

theorem percentage_distance (start : ℝ) (end_point : ℝ) (point : ℝ) (total_distance : ℝ)
  (distance_from_start : ℝ) :
  start = -55 → end_point = 55 → point = 5.5 → total_distance = end_point - start →
  distance_from_start = point - start →
  (distance_from_start / total_distance) * 100 = 55 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_distance_l679_679985


namespace BD_in_quadrilateral_l679_679063

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) : ℕ :=
  if 6 < 11 ∧ 11 < 18 then 11 else 0

theorem BD_in_quadrilateral :
  AB = 7 → BC = 11 → CD = 7 → DA = 13 → (BD = 11) :=
by
  intros h1 h2 h3 h4
  exact congr_arg quadrilateral_BD h1 h2 h3 h4
  sorry

end BD_in_quadrilateral_l679_679063


namespace solution_set_quadratic_l679_679387

variable {x : ℝ}
variable {a b : ℝ}

theorem solution_set_quadratic (h1 : ∀ x, ax > b ↔ x ∈ Set.Ioo ⊤ (1 / 5))
  (h2 : a < 0)
  (h3 : b = a * (1 / 5)) :
  ∀ x, ax^2 + bx - (4 / 5) * a > 0 ↔ x ∈ Set.Ioo (-1:ℝ) (4 / 5 : ℝ) :=
sorry

end solution_set_quadratic_l679_679387


namespace cos_negative_23_pi_over_4_l679_679668

theorem cos_negative_23_pi_over_4 : cos (-23 / 4 * real.pi) = sqrt 2 / 2 := by
  sorry

end cos_negative_23_pi_over_4_l679_679668


namespace sequence_according_to_syllogism_pattern_l679_679948

theorem sequence_according_to_syllogism_pattern
  (S1 : ∀ x : ℝ, y = cos x → is_trigonometric_function y)
  (S2 : ∀ (f : ℝ → ℝ), is_trigonometric_function f → is_periodic_function f)
  (S3 : ∀ x : ℝ, y = cos x → is_periodic_function y) :
  (S2 (λ x => cos x) (S1 (λ x => cos x))) →
  (S1 (λ x => cos x)) →
  (S3 (λ x => cos x)) :=
by
  sorry

end sequence_according_to_syllogism_pattern_l679_679948


namespace contractor_earnings_l679_679227

def total_days : ℕ := 30
def work_rate : ℝ := 25
def fine_rate : ℝ := 7.5
def absent_days : ℕ := 8
def worked_days : ℕ := total_days - absent_days
def total_earned : ℝ := worked_days * work_rate
def total_fine : ℝ := absent_days * fine_rate
def total_received : ℝ := total_earned - total_fine

theorem contractor_earnings : total_received = 490 :=
by
  sorry

end contractor_earnings_l679_679227


namespace size_of_bottle_l679_679816

theorem size_of_bottle (costco_price : ℕ) (store_price : ℕ) (savings : ℕ) (gallon_size : ℕ) :
  costco_price = 8 → store_price = 3 → savings = 16 → gallon_size = 128 → (gallon_size / ((savings + costco_price) / store_price)) = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end size_of_bottle_l679_679816


namespace sum_x_coordinates_Q3_l679_679582

theorem sum_x_coordinates_Q3 (y: Fin 50 → ℝ) (h: (∑ i, y i) = 1050) :
  (∑ i, y i) = 1050 :=
by {
  sorry
}

end sum_x_coordinates_Q3_l679_679582


namespace minimum_value_of_quadratic_l679_679314

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 2 * x * y + y^2 - 6 * x + 2 * y + 8

theorem minimum_value_of_quadratic :
  ∃ x y : ℝ, quadratic_expression x y = -1 ∧ ∀ (a b : ℝ), quadratic_expression a b ≥ -1 :=
begin
  -- Start by finding appropriate values of x and y that achieve the minimum,
  -- then show that the expression is always ≥ -1 for all x and y.
  sorry
end

end minimum_value_of_quadratic_l679_679314


namespace evaluate_expression_l679_679295

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end evaluate_expression_l679_679295


namespace total_turtles_l679_679265

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l679_679265


namespace election_votes_l679_679957

theorem election_votes
  (V : ℕ)  -- total number of votes
  (candidate1_votes_percent : ℕ := 80)  -- first candidate percentage
  (second_candidate_votes : ℕ := 480)  -- votes for second candidate
  (second_candidate_percent : ℕ := 20)  -- second candidate percentage
  (h : second_candidate_votes = (second_candidate_percent * V) / 100) :
  V = 2400 :=
sorry

end election_votes_l679_679957


namespace particular_solutions_of_diff_eq_l679_679679

variable {x y : ℝ}

theorem particular_solutions_of_diff_eq
  (h₁ : ∀ C : ℝ, x^2 = C * (y - C))
  (h₂ : x > 0) :
  (y = 2 * x ∨ y = -2 * x) ↔ (x * (y')^2 - 2 * y * y' + 4 * x = 0) := 
sorry

end particular_solutions_of_diff_eq_l679_679679


namespace cats_with_both_characteristics_l679_679903

theorem cats_with_both_characteristics :
  ∃ (X : ℕ), let total_cats := 66 in
             let undesirable_cats := 21 in
             let stripes := 32 in
             let black_ear := 27 in
             let desirable_cats := total_cats - undesirable_cats in
             let X := 14 in
             (stripes - X) + (black_ear - X) + X + (desirable_cats - ((stripes - X) + (black_ear - X) + X)) = desirable_cats ∧ 
             X = 14 :=
begin
  sorry
end

end cats_with_both_characteristics_l679_679903


namespace find_unknown_value_l679_679511

theorem find_unknown_value (x : ℝ) (h : (3 + 5 + 6 + 8 + x) / 5 = 7) : x = 13 :=
by
  sorry

end find_unknown_value_l679_679511


namespace avg_height_correct_l679_679146

theorem avg_height_correct (h1 h2 h3 h4 : ℝ) (h_distinct: h1 ≠ h2 ∧ h2 ≠ h3 ∧ h3 ≠ h4 ∧ h1 ≠ h3 ∧ h1 ≠ h4 ∧ h2 ≠ h4)
  (h_tallest: h4 = 152) (h_shortest: h1 = 137) 
  (h4_largest: h4 > h3 ∧ h4 > h2 ∧ h4 > h1) (h1_smallest: h1 < h2 ∧ h1 < h3 ∧ h1 < h4) :
  ∃ (avg : ℝ), avg = 145 ∧ (h1 + h2 + h3 + h4) / 4 = avg := 
sorry

end avg_height_correct_l679_679146


namespace total_turtles_l679_679266

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l679_679266


namespace two_digit_numbers_of_form_3_pow_n_l679_679018

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679018


namespace part_a_part_b_part_c_l679_679184

noncomputable def ctg_sum_square (n : ℕ) : ℝ :=
  ∑ i in range (n+1), (cot (i * π / (2 * n + 1))) ^ 2

noncomputable def sin_reciprocal_sum_square (n : ℕ) : ℝ :=
  ∑ i in range (n+1), 1 / (sin (i * π / (2 * n + 1))) ^ 2 

noncomputable def sin_product (n : ℕ) : ℝ :=
  ∏ i in range (n+1), sin (i * π / (2 * n + 1))

theorem part_a (n : ℕ) : ctg_sum_square n = n * (2 * n - 1) / 3 := 
  sorry

theorem part_b (n : ℕ) : sin_reciprocal_sum_square n = 2 * n * (n + 1) / 3 := 
  sorry

theorem part_c (n : ℕ) : sin_product n = (√ (2 * n + 1)) / (2^n) := 
  sorry

end part_a_part_b_part_c_l679_679184


namespace complement_set_l679_679734

-- Define the universal set U and the set A
def U := Set.univ
def A := { x : ℝ | 1 ≤ x ∧ x ≤ 4 }

-- Define the complement of A in U
def C_U_A := { x : ℝ | x < 1 ∨ x > 4 }

-- Theorem statement
theorem complement_set :
  (U \ A) = C_U_A :=
sorry

end complement_set_l679_679734


namespace range_of_x_l679_679730

noncomputable def problem_statement (x : ℝ) : Prop :=
  ∀ m : ℝ, abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 

theorem range_of_x (x : ℝ) :
  problem_statement x → ( ( -1 + Real.sqrt 7) / 2 < x ∧ x < ( 1 + Real.sqrt 3) / 2) :=
by
  intros h
  sorry

end range_of_x_l679_679730


namespace largest_2_digit_number_l679_679676

theorem largest_2_digit_number:
  ∃ (N: ℕ), N >= 10 ∧ N < 100 ∧ N % 4 = 0 ∧ (∀ k: ℕ, k ≥ 1 → (N^k) % 100 = N % 100) ∧ 
  (∀ M: ℕ, M >= 10 → M < 100 → M % 4 = 0 → (∀ k: ℕ, k ≥ 1 → (M^k) % 100 = M % 100) → N ≥ M) :=
sorry

end largest_2_digit_number_l679_679676


namespace complex_symmetric_division_l679_679440

noncomputable def z1 : ℂ := 1 + complex.i
noncomputable def z2 : ℂ := 1 - complex.i

theorem complex_symmetric_division : 
  (z1 / z2) = complex.i := 
  by 
  -- Here would be the actual proof, 
  -- but for now, we use sorry 
  sorry 

end complex_symmetric_division_l679_679440


namespace average_interest_rate_correct_l679_679594

-- Constants representing the conditions
def totalInvestment : ℝ := 5000
def rateA : ℝ := 0.035
def rateB : ℝ := 0.07

-- The condition that return from investment at 7% is twice that at 3.5%
def return_condition (x : ℝ) : Prop := 0.07 * x = 2 * 0.035 * (5000 - x)

-- The average rate of interest formula
noncomputable def average_rate_of_interest (x : ℝ) : ℝ := 
  (0.07 * x + 0.035 * (5000 - x)) / 5000

-- The theorem to prove the average rate is 5.25%
theorem average_interest_rate_correct : ∃ (x : ℝ), return_condition x ∧ average_rate_of_interest x = 0.0525 := 
by
  sorry

end average_interest_rate_correct_l679_679594


namespace minimum_value_minimum_achieved_l679_679377

theorem minimum_value (x y z : ℝ) (h : x + 2*y + real.sqrt 3*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/8 :=
sorry

theorem minimum_achieved (x y z : ℝ) (hx : x = y / 2) (hy : y = z * real.sqrt 3) (h: x + 2*y + real.sqrt 3*z = 1) :
  x^2 + y^2 + z^2 = 1/8 :=
sorry

end minimum_value_minimum_achieved_l679_679377


namespace base5_addition_l679_679253

theorem base5_addition : 
  (14 : ℕ) + (132 : ℕ) = (101 : ℕ) :=
by {
  sorry
}

end base5_addition_l679_679253


namespace two_digit_numbers_of_form_3_pow_n_l679_679020

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679020


namespace fg_3_eq_7_l679_679768

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 2) ^ 2

theorem fg_3_eq_7 : f (g 3) = 7 :=
by
  sorry

end fg_3_eq_7_l679_679768


namespace min_third_score_at_least_95_l679_679514

open_locale big_operators

theorem min_third_score_at_least_95 (s : Fin 6 → ℤ)
  (distinct : function.injective s)
  (avg_score : ∑ i, s i = 555)
  (max_score : ∃ i, s i = 99)
  (min_score : ∃ i, s i = 76) :
  ∃ i, ∀ j k m, i ≠ j → i ≠ k → i ≠ m → 
    i < j ∨ i < k ∨ i < m → 95 ≤ s i :=
by sorry

end min_third_score_at_least_95_l679_679514


namespace tangent_line_condition_l679_679966

theorem tangent_line_condition (a b : ℝ):
  ((a = 1 ∧ b = 1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) ∧
  ( (a = -1 ∧ b = -1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) →
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end tangent_line_condition_l679_679966


namespace find_d_f_h_l679_679648

noncomputable theory
open Complex

def complex_sum (a b c d e f g h : ℝ) : ℂ :=
  (2 * a + b * I) + (c + 2 * d * I) + (e + f * I) + (g + 2 * h * I)

theorem find_d_f_h 
(a b c d e f g h : ℝ) 
(h_b : b = 6)
(h_g : g = -2 * a - c - e)
(h_sum : complex_sum a b c d e f g h = 8 * I) 
: d + f + h = 3 / 2 :=
sorry

end find_d_f_h_l679_679648


namespace distinct_digits_satisfy_eq_l679_679099

-- Define the distinct digits A, B, C
variables {A B C : ℕ}

-- Define the condition that A, B, C are distinct and range from 1 to 9
def valid_digits := 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 1 ≤ C ∧ C ≤ 9 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the main theorem to be proved
theorem distinct_digits_satisfy_eq (h : valid_digits) :
  let ABC := 100 * A + 10 * B + C    in
  let AB := 10 * A + B               in
  let BC := 10 * B + C               in
  let CA := 10 * C + A               in
  ABC = AB * C + BC * A + CA * B 
  ↔ (A = 7 ∧ B = 8 ∧ C = 1) ∨ (A = 5 ∧ B = 1 ∧ C = 7) :=
sorry

end distinct_digits_satisfy_eq_l679_679099


namespace sum_of_fourth_powers_less_than_50_l679_679944

theorem sum_of_fourth_powers_less_than_50 :
  (∑ n in Finset.filter (λ n, n < 50) (Finset.image (λ x, x^4) (Finset.range 50))) = 17 :=
by
  sorry

end sum_of_fourth_powers_less_than_50_l679_679944


namespace infinite_chain_resistance_l679_679658

variables (R_0 R_X : ℝ)
def infinite_chain_resistance_condition (R_0 : ℝ) (R_X : ℝ) : Prop :=
  R_X = R_0 + (R_0 * R_X) / (R_0 + R_X)

theorem infinite_chain_resistance (R_0 : ℝ) (h : R_0 = 50) :
  ∃ R_X, infinite_chain_resistance_condition R_0 R_X ∧ R_X = (R_0 * (1 + Real.sqrt 5)) / 2 :=
  sorry

end infinite_chain_resistance_l679_679658


namespace trigonometric_identity_cos24_cos36_sub_sin24_cos54_l679_679538

theorem trigonometric_identity_cos24_cos36_sub_sin24_cos54  :
  (Real.cos (24 * Real.pi / 180) * Real.cos (36 * Real.pi / 180) - Real.sin (24 * Real.pi / 180) * Real.cos (54 * Real.pi / 180) = 1 / 2) := by
  sorry

end trigonometric_identity_cos24_cos36_sub_sin24_cos54_l679_679538


namespace find_x_of_perpendicular_l679_679366

-- Definitions based on the conditions in a)
def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

-- The mathematical proof problem in Lean 4 statement: prove that the dot product is zero implies x = -2/3
theorem find_x_of_perpendicular (x : ℝ) (h : (a x).fst * b.fst + (a x).snd * b.snd = 0) : x = -2 / 3 := 
by
  sorry

end find_x_of_perpendicular_l679_679366


namespace rectangle_area_eq_240_l679_679960

theorem rectangle_area_eq_240
  (area_square : ℝ)
  (side_square : ℝ)
  (radius_circle : ℝ)
  (length_rectangle : ℝ)
  (breadth_rectangle : ℝ)
  (h1 : area_square = 3600)
  (h2 : side_square = real.sqrt area_square)
  (h3 : radius_circle = side_square)
  (h4 : length_rectangle = (2/5) * radius_circle)
  (h5 : breadth_rectangle = 10) :
  length_rectangle * breadth_rectangle = 240 := 
by 
  sorry

end rectangle_area_eq_240_l679_679960


namespace factor_5x²_6xy_8y²_factor_x²_2x_15_ax_5a_l679_679670

theorem factor_5x²_6xy_8y² (x y : ℝ) : 5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) :=
by sorry

theorem factor_x²_2x_15_ax_5a (x a : ℝ) : x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - 3 - a) :=
by sorry

end factor_5x²_6xy_8y²_factor_x²_2x_15_ax_5a_l679_679670


namespace cost_of_fencing_per_meter_in_paise_l679_679532

theorem cost_of_fencing_per_meter_in_paise
  (area : ℝ)
  (ratio_length_width : ℝ)
  (total_cost : ℝ)
  (length_proportion : ℝ)
  (width_proportion : ℝ)
  (area_eq : area = 5766)
  (ratio_eq : ratio_length_width = 3 / 2)
  (total_cost_eq : total_cost = 155)
  (length_eq : length_proportion = 3)
  (width_eq : width_proportion = 2) :
  (total_cost / (2 * (length_proportion * (sqrt ((area * ratio_length_width) / length_proportion / width_proportion)) + width_proportion * (sqrt ((area * ratio_length_width) / length_proportion / width_proportion))))) * 100 = 50 :=
by
  sorry

end cost_of_fencing_per_meter_in_paise_l679_679532


namespace triangle_is_isosceles_l679_679849

-- Define the points and triangle
structure Triangle (P : Type) :=
  (A B C : P)

structure Within (P : Type) :=
  (triangle : Triangle P)
  (M N : P)

-- Define what it means for two points to be on the sides of the triangle
def PointsOnSides {P : Type} [AddGroup P] [HasZero P] (W : Within P) :=
  ∃ a b c d e f : ℝ, a + b = 1 ∧ c + d = 1 ∧ e + f = 1 ∧ 
  W.M = (a • W.triangle.A + b • W.triangle.B) ∧
  W.M = (c • W.triangle.B + d • W.triangle.C) ∧
  W.N = (e • W.triangle.B + f • W.triangle.C)

-- Define the perimeter function
def perimeter {P : Type} [MetricSpace P] (t : Triangle P) : ℝ :=
  dist t.A t.B + dist t.B t.C + dist t.C t.A

-- State the given conditions
variables {P : Type} [MetricSpace P] 
  (W : Within P)
  (PointsOnSidesCond : PointsOnSides W)

-- Conditions: Perimeter equalities 
axiom AMC_eq_CNA : perimeter {A := W.triangle.A, B := W.M, C := W.triangle.C} = 
                   perimeter {A := W.triangle.C, B := W.N, C := W.triangle.A}

axiom ANB_eq_CMB : perimeter {A := W.triangle.A, B := W.N, C := W.triangle.B} = 
                   perimeter {A := W.triangle.C, B := W.M, C := W.triangle.B}

-- Question: Prove that triangle ABC is isosceles
theorem triangle_is_isosceles : 
  is_isosceles (W.triangle : Triangle P) :=
sorry

end triangle_is_isosceles_l679_679849


namespace quadrilateral_ABCD_theorem_l679_679404

-- noncomputable because we involve square roots which are generally considered noncomputational elements.
noncomputable def quadrilateral_ABCD_proof : Prop :=
  ∃ (r s : ℕ), r > 0 ∧ s > 0 ∧ 
    let AB := r + Real.sqrt s in
    AB = 20 ∧ r + s = 356

-- Without the proof for the moment
theorem quadrilateral_ABCD_theorem : quadrilateral_ABCD_proof := sorry

end quadrilateral_ABCD_theorem_l679_679404


namespace race_distance_between_Sasha_and_Kolya_l679_679486

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679486


namespace domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l679_679951
-- Import the necessary library.

-- Define the domains for the given functions.
def domain_func_1 (x : ℝ) : Prop := true

def domain_func_2 (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2

def domain_func_3 (x : ℝ) : Prop := x ≥ -3 ∧ x ≠ 1

def domain_func_4 (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3

-- Prove the domains of each function.
theorem domain_of_func_1 : ∀ x : ℝ, domain_func_1 x :=
by sorry

theorem domain_of_func_2 : ∀ x : ℝ, domain_func_2 x ↔ (1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem domain_of_func_3 : ∀ x : ℝ, domain_func_3 x ↔ (x ≥ -3 ∧ x ≠ 1) :=
by sorry

theorem domain_of_func_4 : ∀ x : ℝ, domain_func_4 x ↔ (2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3) :=
by sorry

end domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l679_679951


namespace determine_equation_line_m_l679_679663

-- Define the initial conditions and point coordinates
def line_l (x y : ℝ) : Prop := 2 * x + y = 0
def point_Q : ℝ × ℝ := (3, -2)
def point_Qd : ℝ × ℝ := (-1, 5)
def origin : ℝ × ℝ := (0, 0)

-- Define the predicate for intersection at origin
def intersects_at_origin (l m : ℝ → ℝ → Prop) : Prop := l 0 0 ∧ m 0 0

-- Define the reflection properties
def reflection (P : ℝ × ℝ) (line : ℝ → ℝ → Prop) : ℝ × ℝ → Prop := sorry

theorem determine_equation_line_m :
  (∃ m : ℝ → ℝ → Prop,
    intersects_at_origin line_l m ∧
    reflection point_Q line_l (λ Q' => reflection Q' m point_Qd)) →
  ∃ m : ℝ → ℝ → Prop, ∀ x y : ℝ, m x y ↔ 3 * x + y = 0 :=
by
  intros h
  sorry

end determine_equation_line_m_l679_679663


namespace option_a_water_amount_option_b_water_amount_option_b_less_water_minimize_total_water_usage_impact_of_a_on_usage_l679_679997

-- Definitions based on conditions
def initial_cleanliness : ℝ := 0.8
def required_cleanliness : ℝ := 0.99

-- Option A
def option_a_water_usage (x : ℝ) : Prop := (x + initial_cleanliness) / (x + 1) = required_cleanliness

-- Option B
def option_b_water_usage (a y : ℝ) (c : ℝ := 0.95): Prop :=
  (c > initial_cleanliness) ∧ (c < required_cleanliness) ∧ (y + a * c) / (y + a) = required_cleanliness

-- Prove that for option A, x = 19
theorem option_a_water_amount : ∃ x : ℝ, option_a_water_usage x ∧ x = 19 := 
sorry

-- Prove that for option B with c = 0.95, y = 4a
theorem option_b_water_amount (a : ℝ) : 
  1 ≤ a ∧ a ≤ 3 → ∃ y : ℝ, option_b_water_usage a y 0.95 ∧ y = 4 * a :=
sorry

-- Prove that Option B requires less water
theorem option_b_less_water (a : ℝ) (h₁ : 1 ≤ a) (h₂ : a ≤ 3) : 
  (16 - 4 * a > 0) :=
sorry

-- Define the total water usage for Option B
def total_water_usage_option_b (a : ℝ) (c : ℝ := 0.95) : ℝ :=
  let x := (5 * c - 4) / (5 * (1 - c))
  let y := a * (99 - 100 * c)
  x + y  

-- Prove the minimum total water usage for Option B
theorem minimize_total_water_usage (a : ℝ) : 
  1 ≤ a ∧ a ≤ 3 → total_water_usage_option_b(a) = -a + 4 * sqrt(5 * a) - 1 :=
sorry

-- Prove the impact of a on minimum total water usage
theorem impact_of_a_on_usage (a : ℝ) :
  1 ≤ a ∧ a ≤ 3 → (0 < a → T'(a) > 0) :=
sorry

end option_a_water_amount_option_b_water_amount_option_b_less_water_minimize_total_water_usage_impact_of_a_on_usage_l679_679997


namespace total_expenditure_l679_679819

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l679_679819


namespace find_f_2007_plus_f_2008_l679_679348

variable (f : ℝ → ℝ) 
variable (g : ℝ → ℝ)

-- Definitions based on given conditions
def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

def isOddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g(-x) = -g(x)

def passesThrough (g : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  g p.1 = p.2

def relationBetweenFg (f g : ℝ → ℝ) : Prop :=
  ∀ x, g(x) = f(x - 1)

-- The theorem statement
theorem find_f_2007_plus_f_2008
  (even_f : isEvenFunction f)
  (odd_g : isOddFunction g)
  (g_at_neg1 : passesThrough g (-1, 3))
  (rel_fg : relationBetweenFg f g) :
  f 2007 + f 2008 = -3 := 
sorry

end find_f_2007_plus_f_2008_l679_679348


namespace temperature_at_midnight_l679_679852

theorem temperature_at_midnight :
  ∀ (morning_temp noon_rise midnight_drop midnight_temp : ℤ),
    morning_temp = -3 →
    noon_rise = 6 →
    midnight_drop = -7 →
    midnight_temp = morning_temp + noon_rise + midnight_drop →
    midnight_temp = -4 :=
by
  intros
  sorry

end temperature_at_midnight_l679_679852


namespace race_distance_between_Sasha_and_Kolya_l679_679490

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679490


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679492

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679492


namespace find_x_l679_679376

theorem find_x (x : ℝ) : -2 ∈ ({3, 5, x, x^2 + 3*x} : set ℝ) → x = -1 :=
by
  intro h
  sorry

end find_x_l679_679376


namespace delta_k_four_zero_l679_679328

def u (n : ℕ) : ℕ := n^3 + n

def Δ' (u : ℕ → ℕ) (n : ℕ) : ℕ := u (n + 1) - u n

def Δ (k : ℕ) (u : ℕ → ℕ) (n : ℕ) : ℕ :=
  Nat.rec (u n) (fun k' Δk => Δ' (λ n', Δ k' u n') n) k

theorem delta_k_four_zero (n : ℕ) : Δ 4 u n = 0 ∧ (∀ k < 4, Δ k u n ≠ 0) :=
by
  sorry

end delta_k_four_zero_l679_679328


namespace lim_an_times_n_to_zero_l679_679443

-- Definitions of conditions
variable {a_n : ℕ → ℝ}

-- Assume a_n is positive
variable (h_pos : ∀ n, 0 < a_n n)

-- Assume a_n is monotonically decreasing
variable (h_decreasing : ∀ n, a_n (n + 1) ≤ a_n n)

-- Assume the sum of any finite number of terms is not greater than 1
variable (h_sum_le_one : ∀ n, ∑ i in Finset.range (n + 1), a_n i ≤ 1)

-- Prove that \(\lim_{n \rightarrow \infty} n a_{n}=0\)
theorem lim_an_times_n_to_zero :
  Tendsto (λ n : ℕ, n * a_n n) atTop (nhds 0) :=
begin
  sorry,
end

end lim_an_times_n_to_zero_l679_679443


namespace height_difference_proof_l679_679416

noncomputable def height_difference_in_feet (a b c d: ℕ): ℕ :=
  ((b - a) * c * d) / 12

noncomputable def height_difference_in_meters (a b c d: ℕ): ℝ :=
  ((b - a) * c * d : ℝ) * 0.0254

theorem height_difference_proof :
  (height_difference_in_feet 3 6 12 8 = 24) ∧
  (height_difference_in_meters 3 6 12 8 ≈ 7.32) :=
by
  sorry

end height_difference_proof_l679_679416


namespace exists_consecutive_numbers_with_prime_divisors_l679_679745

theorem exists_consecutive_numbers_with_prime_divisors (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p < q ∧ q < 2 * p) :
  ∃ n m : ℕ, (m = n + 1) ∧ 
             (Nat.gcd n p = p) ∧ (Nat.gcd m p = 1) ∧ 
             (Nat.gcd m q = q) ∧ (Nat.gcd n q = 1) :=
by
  sorry

end exists_consecutive_numbers_with_prime_divisors_l679_679745


namespace number_of_ways_to_satisfy_condition_l679_679689

theorem number_of_ways_to_satisfy_condition : 
  (∃ (s : finset ℕ), s ⊂ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ s.card = 3 ∧ ((∑ x in s, x) % 3 = 0)) :=
  sorry

end number_of_ways_to_satisfy_condition_l679_679689


namespace sufficientButNotNecessary_l679_679137

theorem sufficientButNotNecessary (x : ℝ) : ((x + 1) * (x - 3) < 0) → x < 3 ∧ ¬(x < 3 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficientButNotNecessary_l679_679137


namespace polynomial_coefficient_sum_l679_679761

theorem polynomial_coefficient_sum :
  let p := (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6)
  let q := 4 * x^4 + 10 * x^3 + x^2 + 15 * x - 18
  p = q →
  (4 + 10 + 1 + 15 - 18 = 12) :=
by
  intro p_eq_q
  sorry

end polynomial_coefficient_sum_l679_679761


namespace third_square_is_G_l679_679665

def square_placement (squares : Fin 8 → Fin 8) : Prop :=
  ∃ pos, squares[pos] = 7

def visible_squares_third (squares : Fin 8 → Fin 8) : Prop :=
  square_placement squares ∧ squares 2 = 6

theorem third_square_is_G {squares : Fin 8 → Fin 8} (h : square_placement squares) : visible_squares_third squares :=
sorry

end third_square_is_G_l679_679665


namespace red_light_cherries_cost_price_min_value_m_profit_l679_679783

-- Define the constants and cost conditions
def cost_price_red_light_cherries (x : ℝ) (y : ℝ) : Prop :=
  (6000 / (2 * x) - 100 = 1000 / x)

-- Define sales conditions and profit requirement
def min_value_m (m : ℝ) (profit : ℝ) : Prop :=
  (20 * 3 * m + 20 * (20 - 0.5 * m) + (28 - 20) * (50 - 3 * m - 20) >= profit)

-- Define the main proof goal statements
theorem red_light_cherries_cost_price :
  ∃ x, cost_price_red_light_cherries x 6000 ∧ 20 = x :=
sorry

theorem min_value_m_profit :
  ∃ m, min_value_m m 770 ∧ m >= 5 :=
sorry

end red_light_cherries_cost_price_min_value_m_profit_l679_679783


namespace rectangle_diagonals_equal_l679_679965

theorem rectangle_diagonals_equal (ABCD : Quartilateral) (h : is_rectangle ABCD) : 
  diagonals_equal ABCD := 
  sorry

end rectangle_diagonals_equal_l679_679965


namespace find_number_l679_679581

theorem find_number (x : ℝ) (h : 0.30 * x - 70 = 20) : x = 300 :=
sorry

end find_number_l679_679581


namespace ratio_FQ_HQ_l679_679907

variables (E F G H Q : Type)

noncomputable def EQ : ℝ := 5
noncomputable def GQ : ℝ := 7
variable (FQ HQ : ℝ)

axiom intersect_at_Q :
  ∃ Q : Point, (E ∈ Circle) ∧ (F ∈ Circle) ∧ (G ∈ Circle) ∧ (H ∈ Circle)

theorem ratio_FQ_HQ :
  EQ * FQ = GQ * HQ → FQ / HQ = 7 / 5 :=
sorry

end ratio_FQ_HQ_l679_679907


namespace john_percentage_claims_more_than_jan_l679_679646

def N : ℕ := 20
def M : ℕ := 41
def J : ℕ := M - 15

theorem john_percentage_claims_more_than_jan :
  ((J - N):ℚ / N * 100) = 30 := by
  sorry

end john_percentage_claims_more_than_jan_l679_679646


namespace total_surveyed_people_l679_679401

-- Definitions based on conditions
def percentageOfBelievers := 92.3 / 100
def percentageOfMisconception := 38.2 / 100
def numberOfMisconceptionHolders := 29

-- Definition of the question and ultimately the answer
theorem total_surveyed_people : 
  ∃ (y : ℕ), ∃ (x : ℕ),
  percentageOfMisconception * x = numberOfMisconceptionHolders ∧
  percentageOfBelievers * y = x ∧
  y = 83 :=
by
  sorry

end total_surveyed_people_l679_679401


namespace annual_yield_calculation_correct_l679_679272

def security_data :=
  [{quantity := 1000, price_purchase := 95.3, price_180_days := 98.6},
   {quantity := 1000, price_purchase := 89.5, price_180_days := 93.4},
   {quantity := 1000, price_purchase := 92.1, price_180_days := 96.2},
   {quantity := 1,    price_purchase := 100000, price_180_days := 104300},
   {quantity := 1,    price_purchase := 200000, price_180_days := 209420},
   {quantity := 40,   price_purchase := 3700, price_180_days := 3900},
   {quantity := 500,  price_purchase := 137, price_180_days := 142}]

def return_180_day (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

def annual_return (return_180_day : ℝ) : ℝ :=
  ((1 + (return_180_day / 100)) ^ 2 - 1) * 100

def weighted_average (yields : List (ℝ × ℝ)) : ℝ :=
  let total_investment := yields.foldr (λ (yi : ℝ × ℝ) acc => yi.1 + acc) 0
  (yields.foldr (λ (yi : ℝ × ℝ) acc => (yi.1 * yi.2) + acc) 0) / total_investment

def calculate_annual_yield (data : List (ℝ × ℝ × ℝ)) : ℝ :=
  let yields := data.map (λ (d : ℝ × ℝ × ℝ) =>
                   (d.1 * d.2,
                    annual_return (return_180_day d.2 d.3)))
  weighted_average yields

theorem annual_yield_calculation_correct :
  calculate_annual_yield (security_data.map
                           (λ d => (d.quantity, d.price_purchase, d.price_180_days))) ≈ 9.21 :=
by sorry

end annual_yield_calculation_correct_l679_679272


namespace Alice_fills_needed_l679_679999

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l679_679999


namespace base_2_representation_of_125_l679_679922

theorem base_2_representation_of_125 : 
  ∃ (L : List ℕ), L = [1, 1, 1, 1, 1, 0, 1] ∧ 
    125 = (L.reverse.zipWith (λ b p, b * (2^p)) (List.range L.length)).sum := 
by
  sorry

end base_2_representation_of_125_l679_679922


namespace evaluate_expression_l679_679558

-- Define x as given in the condition
def x : ℤ := 5

-- State the theorem we need to prove
theorem evaluate_expression : x^3 - 3 * x = 110 :=
by
  -- Proof will be provided here
  sorry

end evaluate_expression_l679_679558


namespace find_KY_length_l679_679217

-- Define the initial conditions and given values
variables {XY YZ XZ : ℝ}
variable {K L M : Point}
variable {circumcenter_1 circumcenter_2 circumcenter_3 : Point}

-- Define the theorem to prove
theorem find_KY_length :
  let a := 15
  let b := 1
  a + b = 16 :=
by
  intro a b,
  rw [a, b],
  norm_num

end find_KY_length_l679_679217


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679011

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679011


namespace area_of_quadrilateral_PQRT_l679_679405

open Real

theorem area_of_quadrilateral_PQRT
  (P Q R S T : Real × Real)
  (h_triangle : (Q.1 - P.1) * (R.2 - P.2) = (R.1 - P.1) * (Q.2 - P.2) ∧ (Q.1 - P.1) * (Q.2 - P.2) = 0)
  (h_PQ : P.1 = 0 ∧ P.2 = 0 ∧ Q.1 = 60 ∧ Q.2 = 0)
  (h_PR : R.1 = 0 ∧ R.2 = 45)
  (h_S : S.1 = 20 ∧ S.2 = 0)
  (h_T : T.1 = 0 ∧ T.2 = 15) :
  let A_PQR := (1 / 2 : ℝ) * (Q.1 - P.1) * (R.2 - P.2)
  let A_PSQ := (1 / 2 : ℝ) * (S.1 - P.1) * (R.2 - P.2)
  let A_PTR := (1 / 2 : ℝ) * (T.1 - P.1) * (Q.2 - P.2)
  let A_PQRT := A_PQR - A_PSQ - A_PTR
  in A_PQRT = 450 :=
by sorry

end area_of_quadrilateral_PQRT_l679_679405


namespace farthest_point_on_circle_farthest_from_line_l679_679138

-- Define the equation of the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the equation of the line
def line (x y : ℝ) : Prop := 4 * x + 3 * y - 12 = 0

-- Define the point coordinates
def point (x y : ℝ) : Prop := (x = -8/5) ∧ (y = -6/5)

-- Statement: Prove that the point on the circle that is farthest from the line is ( -8/5, -6/5 )
theorem farthest_point_on_circle_farthest_from_line 
  : ∃ (x y : ℝ), circle x y ∧ point x y → ∀ (a b : ℝ), circle a b → ¬ line a b → (x = a) ∧ (y = b) :=
sorry

end farthest_point_on_circle_farthest_from_line_l679_679138


namespace diagonal_angle_of_adjacent_sides_of_a_cube_l679_679555

theorem diagonal_angle_of_adjacent_sides_of_a_cube {e : ℝ}
  (h1 : ∀ (s : Type) [linear_ordered_field s], is_cube s)
  (h2 : ∀ (s : Type) [linear_ordered_field s], is_square_face s)
  (h3 : ∀ (a b c d : ℝ), is_perpendicular_diagonal a b c d) :
  e = 90 :=
  sorry

end diagonal_angle_of_adjacent_sides_of_a_cube_l679_679555


namespace f_of_odd_and_shift_l679_679349

noncomputable def f : ℝ → ℝ :=
sorry -- To be defined based on the conditions

theorem f_of_odd_and_shift (x : ℝ) (h₁ : f x = -f (-x))
    (h₂ : ∀ x ∈ (Icc 2 3 : set ℝ), f (2 - x) = f x)
    (h₃ : ∀ x ∈ (Icc 2 3 : set ℝ), f x = Real.log (x - 1) / Real.log 2) :
    ∀ x ∈ (Icc 1 2 : set ℝ), f x = -Real.log (3 - x) / Real.log 2 :=
begin
    sorry
end

end f_of_odd_and_shift_l679_679349


namespace range_of_a_l679_679090

def f (x : ℝ) : ℝ := x^2 - 2 * x - 4 * real.log x

def P := {x : ℝ | 2 < x}

def Q (a : ℝ) := {x : ℝ | x^2 + (a - 1) * x - a > 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ P → x ∈ Q a) → (∃ x, x ∈ Q a ∧ x ∉ P) → a ∈ set.Ici (-2) :=
begin
  sorry
end

end range_of_a_l679_679090


namespace total_pencils_l679_679158

   variables (n p t : ℕ)

   -- Condition 1: number of students
   def students := 12

   -- Condition 2: pencils per student
   def pencils_per_student := 3

   -- Theorem statement: Given the conditions, the total number of pencils given by the teacher is 36
   theorem total_pencils : t = students * pencils_per_student :=
   by
   sorry
   
end total_pencils_l679_679158


namespace dishonest_dealer_profit_l679_679952

-- Define the dishonest dealer's weights
def cost_weight_grams : Real := 1000
def actual_weight_grams : Real := 700

-- Calculate the shortfall in weight
def shortfall_grams : Real := cost_weight_grams - actual_weight_grams

-- Calculate the shortfall percentage
def shortfall_percentage : Real := shortfall_grams / cost_weight_grams * 100

-- State the theorem
theorem dishonest_dealer_profit : shortfall_percentage = 30 := by
  sorry

end dishonest_dealer_profit_l679_679952


namespace conservation_center_total_turtles_l679_679258

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l679_679258


namespace angle_C_eq_pi_over_3_l679_679052

variable {A B C a b c : ℝ}
variable {ΔABC : Triangle}
variable {angle_opposite : ΔABC → ℝ → ℝ → ℝ → Prop}
variable {sin cos : ℝ → ℝ}

-- Assuming angle_opposite(ΔABC, A, a) and so on, to denote sides and angles
axiom angle_opposite_A : angle_opposite ΔABC A a
axiom angle_opposite_B : angle_opposite ΔABC B b
axiom angle_opposite_C : angle_opposite ΔABC C c

axiom law_of_sines : sin A / a = sin C / c
axiom given_condition : sin A / a = sqrt 3 * cos C / c

theorem angle_C_eq_pi_over_3 (h1 : angle_opposite_A)
                            (h2 : angle_opposite_B)
                            (h3 : angle_opposite_C)
                            (h4 : law_of_sines)
                            (h5 : given_condition) :
  C = π / 3 :=
sorry

end angle_C_eq_pi_over_3_l679_679052


namespace count_valid_triples_l679_679760

-- Define the necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_positive (n : ℕ) : Prop := n > 0
def valid_triple (p q n : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_positive n ∧ (1/p + 2013/q = n/5)

-- Lean statement for the proof problem
theorem count_valid_triples : 
  ∃ c : ℕ, c = 5 ∧ 
  (∀ p q n : ℕ, valid_triple p q n → true) :=
sorry

end count_valid_triples_l679_679760


namespace stability_requires_variance_l679_679566

theorem stability_requires_variance (performances : Fin 10 → ℝ) :
  (∃ measure, measure = "variance" ∧ reflects_fluctuation_size measure) → 
  (∃ answer, answer = "variance" ∧ ∀ m, reflects_fluctuation_size m → m = answer) :=
by
  sorry

end stability_requires_variance_l679_679566


namespace base_2_representation_of_125_l679_679923

theorem base_2_representation_of_125 : 
  ∃ (L : List ℕ), L = [1, 1, 1, 1, 1, 0, 1] ∧ 
    125 = (L.reverse.zipWith (λ b p, b * (2^p)) (List.range L.length)).sum := 
by
  sorry

end base_2_representation_of_125_l679_679923


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679013

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679013


namespace quadrilateral_angle_A_l679_679064

-- Define the quadrilateral and its properties
variables {A B C D : Type} [quadrilateral A B C D]

-- Define the angles in the quadrilateral
def angle_B : ℝ := 70
def sum_of_angles := ∀ (A B C D : Type) [quadrilateral A B C D], ∠A + ∠B + ∠C + ∠D = 360

-- Given angle B, we want to prove angle A
theorem quadrilateral_angle_A (hB : ∠B = 70) (hABCD : quadrilateral A B C D) : ∠A = 110 :=
begin
  sorry
end

end quadrilateral_angle_A_l679_679064


namespace no_guard_schedule_l679_679621

theorem no_guard_schedule (S : Finset ℕ) (h_card : S.card = 100) :
  ¬ ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, t.card = 3) ∧
    (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ∃! t ∈ T, {x, y} ⊆ t) :=
by
  sorry

end no_guard_schedule_l679_679621


namespace partition_displacement_l679_679606

variables (l : ℝ) (R T : ℝ) (initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)

-- Conditions
def initial_conditions (initial_V1 initial_V2 : ℝ) : Prop :=
  initial_V1 + initial_V2 = l ∧
  initial_V2 = 2 * initial_V1 ∧
  initial_P1 * initial_V1 = R * T ∧
  initial_P2 * initial_V2 = 2 * R * T ∧
  initial_P1 = initial_P2

-- Final volumes
def final_volumes (final_Vleft final_Vright : ℝ) : Prop :=
  final_Vleft = l / 2 ∧ final_Vright = l / 2 

-- Displacement of the partition
def displacement (initial_position final_position : ℝ) : ℝ :=
  initial_position - final_position

-- Theorem statement: the displacement of the partition is l / 6
theorem partition_displacement (l R T initial_V1 initial_V2 final_Vleft final_Vright initial_P1 initial_P2 : ℝ)
  (h_initial_cond : initial_conditions l R T initial_V1 initial_V2 initial_P1 initial_P2)
  (h_final_vol : final_volumes l final_Vleft final_Vright) 
  (initial_position final_position : ℝ)
  (initial_position_def : initial_position = 2 * l / 3)
  (final_position_def : final_position = l / 2) :
  displacement initial_position final_position = l / 6 := 
by sorry

end partition_displacement_l679_679606


namespace ratio_water_to_orangejuice_l679_679451

variable (O W : ℝ)

-- Conditions
def first_day_orangeade := 2 * O
def second_day_orangeade := O + W
def revenue_same : Prop := first_day_orangeade = second_day_orangeade

-- Theorem statement
theorem ratio_water_to_orangejuice (O W : ℝ) (h1 : first_day_orangeade O = 2 * O)
    (h2 : second_day_orangeade O W = O + W)
    (h3 : revenue_same O W) : W = O := 
sorry

end ratio_water_to_orangejuice_l679_679451


namespace tank_water_after_rain_final_l679_679618

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end tank_water_after_rain_final_l679_679618


namespace percentage_of_men_l679_679785

variables {M W : ℝ}
variables (h1 : M + W = 100)
variables (h2 : 0.20 * M + 0.40 * W = 34)

theorem percentage_of_men :
  M = 30 :=
by
  sorry

end percentage_of_men_l679_679785


namespace bridge_length_is_correct_l679_679995

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_covered := speed_mps * time
  distance_covered - train_length

theorem bridge_length_is_correct :
  length_of_bridge 100 16.665333439991468 54 = 149.97999909987152 :=
by sorry

end bridge_length_is_correct_l679_679995


namespace p_q_sum_l679_679836

theorem p_q_sum (p q : ℝ) (hp : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 1)
  (hq : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 3)
  (hr : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 5) :
  p + q = 38 :=
sorry

end p_q_sum_l679_679836


namespace solve_for_p_l679_679841

theorem solve_for_p : 
  ∀ (p : ℂ) (f : ℤ) (w : ℂ), 
  f = 7 → 
  w = (7 : ℂ) + 175 * complex.I → 
  f * p - w = 15000 → 
  p = (2143 : ℂ) + 25 * complex.I := 
by
  intros p f w h1 h2 h3
  sorry

end solve_for_p_l679_679841


namespace perimeter_triangle_ABF2_l679_679979

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 9) = 1

def length_AB (A B : ℝ × ℝ) := dist A B = 6

noncomputable def focus_left := (-4, 0)  -- Assuming F₁ at (-a, 0)
noncomputable def focus_right := (4, 0)  -- Assuming F₂ at (a, 0)

theorem perimeter_triangle_ABF2 :
  ∃ (A B : ℝ × ℝ), 
  hyperbola_equation A.fst A.snd 
  ∧ hyperbola_equation B.fst B.snd 
  ∧ length_AB A B 
  ∧ (dist A focus_left = dist B focus_left) 
  ∧ (dist A focus_right + dist B focus_right = 22) → 
  (dist A focus_left + dist B focus_left + dist A focus_right + dist B focus_right = 28) := sorry

end perimeter_triangle_ABF2_l679_679979


namespace remainder_7531_mod_11_is_5_l679_679556

theorem remainder_7531_mod_11_is_5 :
  let n := 7531
  let m := 7 + 5 + 3 + 1
  n % 11 = 5 ∧ m % 11 = 5 :=
by
  let n := 7531
  let m := 7 + 5 + 3 + 1
  have h : n % 11 = m % 11 := sorry  -- by property of digits sum mod
  have hm : m % 11 = 5 := sorry      -- calculation
  exact ⟨h, hm⟩

end remainder_7531_mod_11_is_5_l679_679556


namespace sum_of_sides_of_similar_triangle_l679_679779

theorem sum_of_sides_of_similar_triangle (a b c : ℕ) (scale_factor : ℕ) (longest_side_sim : ℕ) (sum_of_other_sides_sim : ℕ) : 
  a * scale_factor = 21 → c = 7 → b = 5 → a = 3 → 
  sum_of_other_sides = a * scale_factor + b * scale_factor → 
sum_of_other_sides = 24 :=
by
  sorry

end sum_of_sides_of_similar_triangle_l679_679779


namespace intersecting_points_are_same_l679_679908

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def radius1 : ℝ := 5

def center2 : ℝ × ℝ := (3, 6)
def radius2 : ℝ := 3

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := (x - center1.1)^2 + (y + center1.2)^2 = radius1^2
def circle2 (x y : ℝ) : Prop := (x - center2.1)^2 + (y - center2.2)^2 = radius2^2

-- Prove that points C and D coincide
theorem intersecting_points_are_same : ∃ x y, circle1 x y ∧ circle2 x y → (0 = 0) :=
by
  sorry

end intersecting_points_are_same_l679_679908


namespace multiplicative_inverse_l679_679426

def A : ℕ := 123456
def B : ℕ := 171428
def mod_val : ℕ := 1000000
def sum_A_B : ℕ := A + B
def N : ℕ := 863347

theorem multiplicative_inverse : (sum_A_B * N) % mod_val = 1 :=
by
  -- diverting proof with sorry since proof steps aren't the focus
  sorry

end multiplicative_inverse_l679_679426


namespace solve_for_a_l679_679831

def f (x : ℝ) : ℝ :=
if x < 0 then x^2 else x + 1

theorem solve_for_a (a : ℝ) (h : f (f a) = 4) : a = 2 ∨ a = - Real.sqrt 3 := by
  sorry

end solve_for_a_l679_679831


namespace f_8_plus_f_9_l679_679339

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_even_transformed : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 1

theorem f_8_plus_f_9 : f 8 + f 9 = 1 :=
sorry

end f_8_plus_f_9_l679_679339


namespace luis_bought_6_pairs_of_blue_socks_l679_679447

open Nat

-- Conditions
def total_pairs_red := 4
def total_cost_red := 3
def total_cost := 42
def blue_socks_cost := 5

-- Deduce the spent amount on red socks, and from there calculate the number of blue socks bought.
theorem luis_bought_6_pairs_of_blue_socks :
  (yes : ℕ) -> yes * blue_socks_cost = total_cost - total_pairs_red * total_cost_red → yes = 6 :=
sorry

end luis_bought_6_pairs_of_blue_socks_l679_679447


namespace arithmetic_sequence_S13_zero_l679_679795

noncomputable def a (n : ℕ) : ℤ
noncomputable def S (n : ℕ) : ℤ

theorem arithmetic_sequence_S13_zero (h1 : ∃ d a₁ : ℤ, ∀ n : ℕ, a n = a₁ + n * d)
    (h2 : a 3 + a 9 = a 5) 
    (hS : ∀ (n : ℕ) (d a₁ : ℤ), S n = n * (2 * a₁ + (n - 1) * d) / 2) :
  S 13 = 0 := 
sorry

end arithmetic_sequence_S13_zero_l679_679795


namespace number_of_pieces_sold_on_third_day_l679_679584

variable (m : ℕ)

def first_day_sales : ℕ := m
def second_day_sales : ℕ := (m / 2) - 3
def third_day_sales : ℕ := second_day_sales m + 5

theorem number_of_pieces_sold_on_third_day :
  third_day_sales m = (m / 2) + 2 := by sorry

end number_of_pieces_sold_on_third_day_l679_679584


namespace quadratic_completion_l679_679506

theorem quadratic_completion (x : ℝ) :
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3 / 4)^2 - 1 / 8 = 0 :=
by
  sorry

end quadratic_completion_l679_679506


namespace polar_eq_is_circle_l679_679883

-- Define the polar equation as a condition
def polar_eq (ρ : ℝ) := ρ = 5

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Prove that the curve represented by the polar equation is a circle
theorem polar_eq_is_circle (P : ℝ × ℝ) : (∃ ρ θ, P = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_eq ρ) ↔ dist P origin = 5 := 
by 
  sorry

end polar_eq_is_circle_l679_679883


namespace sin_A_sub_B_eq_l679_679706

variables (A B : ℝ)

-- Conditions
#check tan A = 2 * tan B
#check sin (A + B) = 1 / 4

-- Theorem: Prove sin(A - B) = 1 / 12
theorem sin_A_sub_B_eq :
  tan A = 2 * tan B → sin (A + B) = 1 / 4 → sin (A - B) = 1 / 12 :=
by
  intros h1 h2
  sorry

end sin_A_sub_B_eq_l679_679706


namespace total_expenditure_l679_679820

variable (num_coffees_per_day : ℕ) (cost_per_coffee : ℕ) (days_in_april : ℕ)

theorem total_expenditure (h1 : num_coffees_per_day = 2) (h2 : cost_per_coffee = 2) (h3 : days_in_april = 30) :
  num_coffees_per_day * cost_per_coffee * days_in_april = 120 := by
  sorry

end total_expenditure_l679_679820


namespace equalize_expenses_l679_679081

/-- Problem Statement:
Given the amount paid by LeRoy (A), Bernardo (B), and Carlos (C),
prove that the amount LeRoy must adjust to share the costs equally is (B + C - 2A) / 3.
-/
theorem equalize_expenses (A B C : ℝ) : 
  (B+C-2*A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equalize_expenses_l679_679081


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679007

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679007


namespace two_digit_numbers_of_form_3_pow_n_l679_679019

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679019


namespace interior_diagonal_length_eq_sqrt_15_l679_679537

theorem interior_diagonal_length_eq_sqrt_15
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + a * c) = 34)
  (h2 : 4 * (a + b + c) = 28) :
  real.sqrt (a^2 + b^2 + c^2) = real.sqrt 15 :=
by
  sorry

end interior_diagonal_length_eq_sqrt_15_l679_679537


namespace distinct_positive_integer_roots_pq_l679_679835

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l679_679835


namespace percentage_weight_loss_l679_679575

/-- Definitions of the initial conditions. -/
section
variables (W : ℝ) -- initial body weight

/-- Lose 14 percentage of body weight. -/
def weight_loss_percentage := 14 / 100

/-- The new weight after losing 14%. -/
def new_weight := (1 - weight_loss_percentage) * W

/-- Clothes add 2% to the new weight. -/
def clothes_percentage := 2 / 100
def final_weight_with_clothes := new_weight W * (1 + clothes_percentage)

theorem percentage_weight_loss (W : ℝ) (hW : W > 0) :
  ((W - final_weight_with_clothes W) / W * 100) = 12.28 := by
  sorry
end

end percentage_weight_loss_l679_679575


namespace darnel_sprint_laps_l679_679286

theorem darnel_sprint_laps 
  (jog_laps : ℝ) 
  (sprint_additional_laps : ℝ) 
  (h_jog : jog_laps = 0.75) 
  (h_sprint_additional : sprint_additional_laps = 0.13) : 
  sprint_laps = 0.88 :=
by
  let sprint_laps := jog_laps + sprint_additional_laps
  rw [h_jog, h_sprint_additional] at sprint_laps
  exact sprint_laps

end darnel_sprint_laps_l679_679286


namespace line_mn_bisects_pq_l679_679739

-- Variables and geometric objects
variables {C : Type*} [metric_space C] [normed_add_comm_group C] [normed_space ℝ C]
variables {A B M N P Q : C}
  {circle1 circle2 : set C} 
  (h_circle1 : metric.subset_eq circle1 (metric.ball A B))
  (h_circle2 : metric.subset_eq circle2 (metric.ball A B))
  (h_tangent1 : metric.is_tangent_point circle1 C M)
  (h_tangent2 : metric.is_tangent_point circle1 C N)
  (h_pq_a : metric.line_intersect (metric.line_through P M) circle2)
  (h_pq_b : metric.line_intersect (metric.line_through Q N) circle2)
  (h_ab : metric.line_through A B)

-- Theorem: Line MN bisects segment PQ
theorem line_mn_bisects_pq :
  -- line segment PQ
  let K := metric.midpoint P Q in  -- Introduce point K which is the midpoint
  metric.line_through M N = metric.line_through P Q → metric.dist Q K = metric.dist K P :=
by sorry

end line_mn_bisects_pq_l679_679739


namespace find_a1_l679_679338
-- Importing the necessary library

-- Define the arithmetic sequence sum and term conditions
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

-- Define the conditions
axiom S₁₀_eq_five : sum_arithmetic_sequence a₁ d 10 = 5
axiom a₇_eq_one : arithmetic_sequence a₁ d 7 = 1

-- Prove the value of a₁
theorem find_a1 (a₁ d : ℚ) (S₁₀_eq_five : sum_arithmetic_sequence a₁ d 10 = 5)
               (a₇_eq_one : arithmetic_sequence a₁ d 7 = 1) : 
  a₁ = -1 :=
by
  sorry

end find_a1_l679_679338


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679479

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679479


namespace total_birdseed_amount_l679_679822

-- Define the birdseed amounts in the boxes
def box1_amount : ℕ := 250
def box2_amount : ℕ := 275
def box3_amount : ℕ := 225
def box4_amount : ℕ := 300
def box5_amount : ℕ := 275
def box6_amount : ℕ := 200
def box7_amount : ℕ := 150
def box8_amount : ℕ := 180

-- Define the weekly consumption of each bird
def parrot_consumption : ℕ := 100
def cockatiel_consumption : ℕ := 50
def canary_consumption : ℕ := 25

-- Define a theorem to calculate the total birdseed that Leah has
theorem total_birdseed_amount : box1_amount + box2_amount + box3_amount + box4_amount + box5_amount + box6_amount + box7_amount + box8_amount = 1855 :=
by
  sorry

end total_birdseed_amount_l679_679822


namespace counting_formula_C_n_n_l679_679094

noncomputable def C (n m : ℕ) : ℕ :=
∑ j in Finset.range (n + 1), (-1)^(n - j) * Nat.choose n j * Nat.choose (2^j - 1) m

theorem counting_formula_C_n_n (n : ℕ) :
  C n n = ∑ j in Finset.range (n + 1), (-1)^(n - j) * Nat.choose n j * Nat.choose (2^j - 1) n :=
sorry

end counting_formula_C_n_n_l679_679094


namespace continuity_at_2_l679_679459

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end continuity_at_2_l679_679459


namespace prob_transform_in_S_l679_679239

def S : Set ℂ := { z : ℂ | -1 ≤ z.re ∧ z.re ≤ 1 ∧ -1 ≤ z.im ∧ z.im ≤ 1 }

def C_transform (z : ℂ) : ℂ := (1 / 2 + 1 / 2 * Complex.I) * z

theorem prob_transform_in_S (z : ℂ) (hz : z ∈ S) : 
  probability (C_transform(z) ∈ S | z ∈ S) = 1 :=
sorry

end prob_transform_in_S_l679_679239


namespace ratio_PM_MQ_eq_one_l679_679074

theorem ratio_PM_MQ_eq_one :
  ∀ (A B C D E M P Q : ℝ × ℝ),
  let AE := ((0, 10) : ℝ × ℝ) to ((3, 0) : ℝ × ℝ)
  let midpoint := (λ (p1 p2 : ℝ × ℝ), ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2))
  M = midpoint (0, 10) (3, 0) →
  let bisector := (λ (M : ℝ × ℝ), (λ (x : ℝ), 5 + (3 / 10) * (x - 1.5)))
  P.2 = 10 ∧ P.1 = 53 / 3 ∧ Q.2 = 0 ∧ Q.1 = -47 / 3 →
  let PM := M.2 → let MQ := M.2 - Q.2 →
  PM / MQ = 1 :=
by
  intros A B C D E M P Q AE midpoint bisector
  rw [AE, midpoint, bisector]
  simp
  sorry

end ratio_PM_MQ_eq_one_l679_679074


namespace cricket_target_run_l679_679408

theorem cricket_target_run (r1 r2 : ℝ) (H1 : r1 = 3.2) (H2 : r2 = 6.5) : 
  let T := 10 * r1 + 40 * r2 
  in T = 292 :=
by {
  sorry
}

end cricket_target_run_l679_679408


namespace hyperbola_equation_l679_679729

-- Define the conditions
variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) 
          (h_focal_len : a^2 + b^2 = 5)
          (h_asymptote : a = 2 * b)

-- State the problem
theorem hyperbola_equation : (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x^2 / 4 - y^2 = 1)) :=
by
  intro hab_eq
  cases hab_eq with ha hb
  sorry

end hyperbola_equation_l679_679729


namespace collinear_vectors_x_value_l679_679368

variables (x : ℝ)

def is_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem collinear_vectors_x_value :
  is_collinear (1, 2) (x, -4) → x = -2 :=
by
  intro h
  cases h with k hk
  simp at hk
  sorry

end collinear_vectors_x_value_l679_679368


namespace xy_value_in_range_l679_679088

theorem xy_value_in_range (x y : ℝ) (hx₀ : ∀ z, [z] = z.floor) (hx₁ : y = 3 * ⌊x⌋ + 4)
  (hx₂ : y = 4 * ⌊x - 3⌋ + 7) (hx₃ : ¬ ⌊x⌋.fractional_part = 0) :
  40 < x + y ∧ x + y < 41 :=
sorry

end xy_value_in_range_l679_679088


namespace sum_between_common_multiples_l679_679221

-- Definitions based on conditions in a)
def smallest_common_multiple (n : ℕ) (a b : ℕ) : Prop :=
  n = nat.lcm a b

-- Proof statement for problem in c)
theorem sum_between_common_multiples {n a b : ℕ}
  (h : smallest_common_multiple n a b) (h1: n = 21) (h2: a = 3) (h3: b = 7) : 
  (189 + 210 + 231) = 630 :=
sorry

end sum_between_common_multiples_l679_679221


namespace equilateral_triangle_AM_lt_BM_CM_l679_679087

theorem equilateral_triangle_AM_lt_BM_CM (A B C M : Point)
  (h_equilateral : equilateral_triangle A B C)
  (h_M_on_ext_BC : collinear (C :: M :: nil) ∧ between B C M) :
  distance A M < distance B M + distance C M :=
sorry

end equilateral_triangle_AM_lt_BM_CM_l679_679087


namespace continuous_function_property_l679_679671

-- Define the conditions: d in (0,1], and f is a continuous function on [0,1] with f(0) = f(1)
noncomputable def is_valid_d (d : ℝ) : Prop :=
  d ∈ Set.Ioc (0 : ℝ) (1 : ℝ) ∧ ∃ k : ℕ, d = 1 / (k : ℝ)

-- Given the conditions, prove that there exists x₀ in [0,1-d] such that f(x₀) = f(x₀ + d)
theorem continuous_function_property (d : ℝ) (f : ℝ → ℝ) (h0 : 0 < d) (h1 : d ≤ 1) (hf_cont : ContinuousOn f (Set.Icc (0 : ℝ) (1 : ℝ))) (hf_eq : f 0 = f 1) :
  is_valid_d d :=
begin
  sorry
end

end continuous_function_property_l679_679671


namespace truckToCarRatio_l679_679812

-- Conditions
def liftsCar (C : ℕ) : Prop := C = 5
def peopleNeeded (C T : ℕ) : Prop := 6 * C + 3 * T = 60

-- Theorem statement
theorem truckToCarRatio (C T : ℕ) (hc : liftsCar C) (hp : peopleNeeded C T) : T / C = 2 :=
by
  sorry

end truckToCarRatio_l679_679812


namespace find_gear_p_rpm_l679_679277

def gear_p_rpm (r : ℕ) (gear_p_revs : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) : Prop :=
  r = gear_p_revs * 2

theorem find_gear_p_rpm (r : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) :
  gear_q_rpm = 40 ∧ time_seconds = 30 ∧ extra_revs_q_over_p = 15 ∧ gear_p_revs = 10 / 2 →
  r = 10 :=
by
  sorry

end find_gear_p_rpm_l679_679277


namespace polynomial_degree_l679_679656

-- Definitions of the constants and conditions
variables (a b c d e f : ℝ)
hypothesis (non_zero_a : a ≠ 0)
hypothesis (non_zero_b : b ≠ 0)
hypothesis (non_zero_c : c ≠ 0)
hypothesis (non_zero_d : d ≠ 0)
hypothesis (non_zero_e : e ≠ 0)
hypothesis (non_zero_f : f ≠ 0)
hypothesis (condition1 : a + b = c)
hypothesis (condition2 : d + e = f)

-- The target theorem stating that the degree of the polynomial product is 13
theorem polynomial_degree :
  degree ((X^5 + C a * X^8 + C b * X^2 + C c) * (C 2 * X^4 + C d * X^3 + C e) * (C 3 * X)) = 13 :=
by
  sorry

end polynomial_degree_l679_679656


namespace german_students_count_l679_679787

def total_students : ℕ := 45
def both_english_german : ℕ := 12
def only_english : ℕ := 23

theorem german_students_count :
  ∃ G : ℕ, G = 45 - (23 + 12) + 12 :=
sorry

end german_students_count_l679_679787


namespace value_of_x_l679_679391

theorem value_of_x (x : ℝ) (h : x = 88 + 0.3 * 88) : x = 114.4 :=
by
  sorry

end value_of_x_l679_679391


namespace count_multiples_of_15_l679_679752

theorem count_multiples_of_15 : 
  ∃ n : ℕ, n = ∑ k in (finset.filter (λ x, ∃ m, x = 15 * m) (finset.Icc 5 135)), 1 ∧ n = 9 :=
by
  sorry

end count_multiples_of_15_l679_679752


namespace triangle_sides_arithmetic_sequence_l679_679058

-- Define a structure for a triangle with vertices and its incenter and centroid
structure Triangle :=
  (A B C : ℝ × ℝ) -- vertices
  (a b c : ℝ)     -- side lengths

-- Define the condition that the line between the incenter and centroid is parallel to one of the sides
def parallel_incentre_centroid (T : Triangle) : Prop :=
  let A := T.A
  let B := T.B
  let C := T.C
  let G := (1 / 3 * (A.1 + B.1 + C.1), 1 / 3 * (A.2 + B.2 + C.2)) -- Centroid
  let I := (-- compute x-coordinate of incenter
             (C.1 * real.sqrt (A.2 ^ 2 + B.1 ^ 2) - B.1 * real.sqrt (A.2 ^ 2 + C.1 ^ 2)) / 
             (B.1 + C.1 + real.sqrt (A.2 ^ 2 + B.1 ^ 2) + real.sqrt (A.2 ^ 2 + C.1 ^ 2)),
           -- compute y-coordinate of incenter
             A.2 * (B.1 + C.1) / 
             (B.1 + C.1 + real.sqrt (A.2 ^ 2 + B.1 ^ 2) + real.sqrt (A.2 ^ 2 + C.1 ^ 2))) -- Incenter
  I.2 = G.2 -- The y-coordinates must be proportional

-- The main theorem statement
theorem triangle_sides_arithmetic_sequence (T : Triangle) 
  (h : parallel_incentre_centroid T) : 
  T.c + T.b = 2 * T.a := -- side lengths form an arithmetic sequence
sorry

end triangle_sides_arithmetic_sequence_l679_679058


namespace equivalent_problem_l679_679198

theorem equivalent_problem : 2 ^ (1 + 2 + 3) - (2 ^ 1 + 2 ^ 2 + 2 ^ 3) = 50 := by
  sorry

end equivalent_problem_l679_679198


namespace maximize_probability_l679_679973

noncomputable def probability_of_win (n : ℕ) : ℚ := 
  10 * n / ((n + 5) * (n + 4))

noncomputable def probability_of_one_win_out_of_three (n : ℕ) : ℚ :=
  3 * (probability_of_win n) * (1 - probability_of_win n) ^ 2
    
theorem maximize_probability (n > 1) : (argmax (λ n : ℕ, probability_of_one_win_out_of_three n) (n > 1)) = 20 :=
sorry

end maximize_probability_l679_679973


namespace traffic_signal_red_light_is_random_event_l679_679201

theorem traffic_signal_red_light_is_random_event : 
  ∀ (events : ℕ → Prop), 
  (∀ n, events n ↔ (n % 3 = 0)) →  -- A simple model of traffic light cycle.
  (∃ n, events n) :=               -- There exists some times when the light is red.
by
  intros events h
  have hmod : ∃ n, n % 3 = 0 := sorry
  cases hmod with n hn
  use n
  rw h
  exact hn

end traffic_signal_red_light_is_random_event_l679_679201


namespace product_of_integers_abs_less_than_six_l679_679528

theorem product_of_integers_abs_less_than_six : (∏ i in (-5 : ℤ).val.to_finset ∪ (-4 : ℤ).val.to_finset ∪ (-3 : ℤ).val.to_finset ∪ (-2 : ℤ).val.to_finset ∪ (-1 : ℤ).val.to_finset ∪ (0 : ℤ).val.to_finset ∪ (1 : ℤ).val.to_finset ∪ (2 : ℤ).val.to_finset ∪ (3 : ℤ).val.to_finset ∪ (4 : ℤ).val.to_finset ∪ (5 : ℤ).val.to_finset, i) = 0 := by
  sorry

end product_of_integers_abs_less_than_six_l679_679528


namespace eccentricity_range_l679_679346

theorem eccentricity_range (a b : ℝ) (e : ℝ) (h₁ : a > b) (h₂ : b > 0)
  (h₃ : no_point_P_with_right_angle : ∃ P: ℝ × ℝ, ¬ (∠ (0, 0) P (a, 0) = π / 2)) : 
  0 < e ∧ e ≤ (Real.sqrt 2) / 2 :=
by sorry

end eccentricity_range_l679_679346


namespace unique_digit_numbers_count_l679_679753

theorem unique_digit_numbers_count : 
  (∃ nums : finset ℕ, (∀ n ∈ nums, ∀ d ∈ nat.digits 10 n, d = 0 ∨ d = 1 ∨ d = 2) ∧ (∀ n ∈ nums, ∃ l, n < 10^l) ∧ (∀ n ∈ nums, nat.nodup (nat.digits 10 n)) ∧ nums.card = 11) :=
sorry

end unique_digit_numbers_count_l679_679753


namespace faster_train_length_l679_679212

-- Conditions
def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def crossing_time_seconds : ℝ := 12
def conversion_factor_kmph_to_mps : ℝ := 5 / 18

-- We want to prove the length of the faster train
theorem faster_train_length :
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor_kmph_to_mps
  (relative_speed_mps * crossing_time_seconds) = 120 := by
  -- Proof will go here
  sorry

end faster_train_length_l679_679212


namespace number_of_candy_packages_l679_679465

theorem number_of_candy_packages (total_candies pieces_per_package : ℕ) 
  (h_total_candies : total_candies = 405)
  (h_pieces_per_package : pieces_per_package = 9) :
  total_candies / pieces_per_package = 45 := by
  sorry

end number_of_candy_packages_l679_679465


namespace distance_between_Sasha_and_Kolya_l679_679472

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679472


namespace sum_of_all_possible_m_whole_values_l679_679246

def forms_triangle (a b c : ℕ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

theorem sum_of_all_possible_m_whole_values : (∑ x in Finset.Icc 6 18, x) = 153 :=
by
  sorry

end sum_of_all_possible_m_whole_values_l679_679246


namespace order_of_three_numbers_l679_679150

-- Given conditions
def cond1 : Prop := sqrt 6 > 1
def cond2 : Prop := 0.5 ^ 6 < 1
def cond3 : Prop := log 0.5 6 < 0

-- Conclusion based on conditions
theorem order_of_three_numbers (h1 : cond1) (h2 : cond2) (h3 : cond3) : log 0.5 6 < 0.5 ^ 6 ∧ 0.5 ^ 6 < sqrt 6 :=
by
  sorry

end order_of_three_numbers_l679_679150


namespace repeating_decimal_to_fraction_l679_679299

theorem repeating_decimal_to_fraction : (x : ℝ) (h : x = 0.353535...) → x = 35 / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l679_679299


namespace manager_salary_l679_679513

theorem manager_salary (avg_25 : ℝ) (avg_26 : ℝ) (num_employees : ℕ) (num_with_manager : ℕ) (total_25 : ℝ) 
(total_26 : ℝ) (total_with_manager : ℝ) : 
    avg_25 = 2500 ∧ avg_25 * num_employees = total_25 ∧ avg_26 = avg_25 + 400 ∧ num_employees = 25 ∧ 
    num_with_manager = 26 ∧ avg_26 * num_with_manager = total_with_manager ∧ 
    total_with_manager = 75400 ∧ total_25 = 62500 → total_with_manager - total_25 = 12900 :=
by
    intros h
    have avg_25_eq_2500 : avg_25 = 2500 := h.1
    have total_25_eq : avg_25 * num_employees = total_25 := h.2.1
    have avg_26_eq : avg_26 = avg_25 + 400 := h.2.2.1
    have num_employees_eq : num_employees = 25 := h.2.2.2.1
    have num_with_manager_eq : num_with_manager = 26 := h.2.2.2.2.1
    have total_with_manager_eq : avg_26 * num_with_manager = total_with_manager := h.2.2.2.2.2.1
    have total_with_manager_val : total_with_manager = 75400 := h.2.2.2.2.2.2.1
    have total_25_val : total_25 = 62500 := h.2.2.2.2.2.2.2.1
    sorry

end manager_salary_l679_679513


namespace post_height_l679_679993

/-- A squirrel runs up a cylindrical post, in a perfect spiral path making one circuit for each rise of 5 feet. The post has a circumference of 3 feet and the squirrel travels a total of 15 feet. 
Prove that the height of the post is 15 feet. -/
theorem post_height 
  (circumference rise_per_circuit total_distance : ℝ)
  (circuits := total_distance / rise_per_circuit)
  (height := circuits * rise_per_circuit)
  (circumference = 3)
  (rise_per_circuit = 5)
  (total_distance = 15)
  : height = 15 :=
by
  sorry

end post_height_l679_679993


namespace factor_of_sum_of_four_consecutive_integers_l679_679172

theorem factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
    ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k := 
begin
  sorry
end

end factor_of_sum_of_four_consecutive_integers_l679_679172


namespace y_payment_l679_679904

variable (x y z : ℝ)

def payment_condition_1 : Prop := x + y + z = 1000
def payment_condition_2 : Prop := x = 1.2 * y
def payment_condition_3 : Prop := z = 0.8 * y
def payment_condition_4 : Prop := x + z = 600

theorem y_payment (h1 : payment_condition_1 x y z)
                  (h2 : payment_condition_2 x y z)
                  (h3 : payment_condition_3 x y z)
                  (h4 : payment_condition_4 x y z) :
                  y = 300 :=
  sorry

end y_payment_l679_679904


namespace det_product_l679_679037

variable {A B C : Matrix ℕ ℕ ℝ}

def detA : ℝ := 3
def detB : ℝ := 8
def detC : ℝ := 5

theorem det_product (hA : det A = detA) (hB : det B = detB) (hC : det C = detC) : 
  det (A ⬝ B ⬝ C) = 120 := by
sorry

end det_product_l679_679037


namespace nancy_rose_bracelets_l679_679105

-- Definitions based on conditions
def metal_beads_nancy : ℕ := 40
def pearl_beads_nancy : ℕ := metal_beads_nancy + 20
def total_beads_nancy : ℕ := metal_beads_nancy + pearl_beads_nancy

def crystal_beads_rose : ℕ := 20
def stone_beads_rose : ℕ := 2 * crystal_beads_rose
def total_beads_rose : ℕ := crystal_beads_rose + stone_beads_rose

def number_of_bracelets (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

-- Theorem to be proved
theorem nancy_rose_bracelets : number_of_bracelets (total_beads_nancy + total_beads_rose) 8 = 20 := 
by
  -- Definitions will be expanded here
  sorry

end nancy_rose_bracelets_l679_679105


namespace number_of_long_furred_dogs_l679_679395

/-- In a certain kennel, each of the 45 dogs is a single color. Each of the dogs either has long fur or does not. 
    30 dogs are brown, 8 dogs are neither long-furred nor brown. 19 dogs are both long-furred and brown.
    Prove that the number of dogs with long fur is 26. --/
theorem number_of_long_furred_dogs :
  ∀ (total_dogs brown_dogs neither_long_furred_nor_brown dogs_long_furred_and_brown dogs_long_furred),
    total_dogs = 45 →
    brown_dogs = 30 →
    neither_long_furred_nor_brown = 8 →
    dogs_long_furred_and_brown = 19 →
    dogs_long_furred = 26 :=
begin
  intros total_dogs brown_dogs neither_long_furred_nor_brown dogs_long_furred_and_brown dogs_long_furred,
  sorry
end

end number_of_long_furred_dogs_l679_679395


namespace h_h_3_eq_3568_l679_679766

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h_3_eq_3568_l679_679766


namespace exp_convex_on_ℝ_ln_concave_on_0_inf_l679_679500

namespace Proofs

-- Proof Problem 1: Prove that exp(x) is convex on ℝ.
theorem exp_convex_on_ℝ : ∀ (x : ℝ), (deriv^[2] (λ x, Real.exp x)) x ≥ 0 :=
by
  intro x
  -- The second derivative of exp(x) is exp(x) which is always positive on ℝ
  have h : (deriv^[2] (λ x, Real.exp x)) x = Real.exp x := by simp [Real.exp]
  linarith

-- Proof Problem 2: Prove that ln(x) is concave on (0, +∞).
theorem ln_concave_on_0_inf : ∀ (x : ℝ), 0 < x → (deriv^[2] (λ x, Real.log x)) x ≤ 0 :=
by
  intro x hx
  -- The second derivative of ln(x) is -1/(x^2) which is always negative on (0, +∞)
  have h : (deriv^[2] (λ x, Real.log x)) x = -(1 / (x^2)) := by simp [Real.log]
  linarith
end Proofs

end exp_convex_on_ℝ_ln_concave_on_0_inf_l679_679500


namespace distinct_positive_integer_roots_pq_l679_679834

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l679_679834


namespace problem_l679_679291

theorem problem (x y : ℝ) : 
  2 * x + y = 11 → x + 2 * y = 13 → 10 * x^2 - 6 * x * y + y^2 = 530 :=
by
  sorry

end problem_l679_679291


namespace problem1_problem2_l679_679727

noncomputable def f (x : ℝ) : ℝ := -x + real.log (1 - x) / real.log 2 - real.log (1 + x) / real.log 2

theorem problem1 : f (1 / 2016) + f (-1 / 2016) = 0 :=
by sorry

theorem problem2 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  ∀ x ∈ set.Icc (-a) a, f x ≥ f a :=
by sorry

end problem1_problem2_l679_679727


namespace area_of_K_lt_4_l679_679083

-- Assume K is a convex polygon
variable {K : set (ℝ × ℝ)}

-- Conditions given in the problem
def is_convex_polygon (K : set (ℝ × ℝ)) : Prop := 
  convex ℝ K -- "convex" requires the convex lemma for appropriate set interpretations

def area_K (K : set (ℝ × ℝ)) := measure_theory.measure_space.measure (K : set (ℝ × ℝ))

def Q1 := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0}
def Q2 := {p : ℝ × ℝ | p.1 ≤ 0 ∧ p.2 ≥ 0}
def Q3 := {p : ℝ × ℝ | p.1 ≤ 0 ∧ p.2 ≤ 0}
def Q4 := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≤ 0}

def quadrant_condition (K : set (ℝ × ℝ)) : Prop :=
  area_K (K ∩ Q1) = (1/4) * area_K K ∧ 
  area_K (K ∩ Q2) = (1/4) * area_K K ∧ 
  area_K (K ∩ Q3) = (1/4) * area_K K ∧ 
  area_K (K ∩ Q4) = (1/4) * area_K K

def no_nonzero_lattice_point (K : set (ℝ × ℝ)) : Prop := 
  ∀ p ∈ K, p ≠ (0, 0) → ∀ m n : ℤ, (p.1, p.2) ≠ (m, n)

-- The main theorem statement
theorem area_of_K_lt_4 {K : set (ℝ × ℝ)} (h1 : is_convex_polygon K) 
  (h2 : quadrant_condition K) 
  (h3 : no_nonzero_lattice_point K) :
  area_K K < 4 := 
sorry

end area_of_K_lt_4_l679_679083


namespace apples_left_correct_l679_679508

noncomputable def apples_left (initial_apples : ℝ) (additional_apples : ℝ) (apples_for_pie : ℝ) : ℝ :=
  initial_apples + additional_apples - apples_for_pie

theorem apples_left_correct :
  apples_left 10.0 5.5 4.25 = 11.25 :=
by
  sorry

end apples_left_correct_l679_679508


namespace chord_length_tangent_l679_679881

noncomputable def chord_length (a b : ℝ) : ℝ := 2 * real.sqrt 50

theorem chord_length_tangent (a b : ℝ) (h : a ^ 2 - b ^ 2 = 50) :
  chord_length a b = 10 * real.sqrt 2 := 
by
  have h1 : (chord_length a b / 2) ^ 2 = 50 := 
    by rw [chord_length, div_mul_div_comm, mul_div_cancel_left, sq_sqrt, real.sqrt_eq_rpow, @two_ne_zero ℝ]; norm_num
  rw [chord_length, mul_div_cancel_left];
  sorry

end chord_length_tangent_l679_679881


namespace sum_of_x_coordinates_mod13_intersection_l679_679108

theorem sum_of_x_coordinates_mod13_intersection :
  (∀ x y : ℕ, y ≡ 3 * x + 5 [MOD 13] → y ≡ 7 * x + 4 [MOD 13]) → (x ≡ 10 [MOD 13]) :=
by
  sorry

end sum_of_x_coordinates_mod13_intersection_l679_679108


namespace vets_in_state_l679_679220

variable {V : ℝ}

-- Define the conditions from step a)
def percentage_Puppy_Kibble : ℝ := 0.20 * V
def percentage_Yummy_Dog_Kibble : ℝ := 0.30 * V
def diff_100_vets : Prop := percentage_Yummy_Dog_Kibble - percentage_Puppy_Kibble = 100

-- Prove the question equals the correct answer
theorem vets_in_state (h : diff_100_vets) : V = 1000 :=
by
  sorry

end vets_in_state_l679_679220


namespace tenth_term_expansion_constant_term_expansion_max_abs_coeff_term_l679_679356

noncomputable def n : ℕ :=
  Nat.recOn 5 (2 ^ (2 * _)) - (2 ^ _) = 992

theorem tenth_term_expansion :
  let T := (2 * x - 1 / x) ^ (2 * n)
  (T.coeff 9) = -20 * x ^ -8 :=
sorry

theorem constant_term_expansion :
  let T := (2 * x - 1 / x) ^ (2 * n)
  T.coeff 10 = -8064 :=
sorry

theorem max_abs_coeff_term :
  let T := (2 * x - 1 / x) ^ (2 * n)
  (∃ r : ℕ, T.coeff r = -15360 * x ^ 4) :=
sorry

end tenth_term_expansion_constant_term_expansion_max_abs_coeff_term_l679_679356


namespace factor_theorem_l679_679292

noncomputable def Q (b x : ℝ) : ℝ := x^4 - 3 * x^3 + b * x^2 - 12 * x + 24

theorem factor_theorem (b : ℝ) : (∃ x : ℝ, x = -2) ∧ (Q b x = 0) → b = -22 :=
by
  sorry

end factor_theorem_l679_679292


namespace f_periodic_property_l679_679696

theorem f_periodic_property (f : ℝ → ℝ)
  (h₁ : ∀ x, f(x + 3) = -f(x))
  (h₂ : ∀ x, x ∈ set.Ico (-3 : ℝ) 0 → f(x) = 2^x + real.sin (real.pi * x / 3)) :
  f(2023) = -1/4 + real.sqrt 3 / 2 :=
sorry

end f_periodic_property_l679_679696


namespace num_nonempty_subsets_of_odds_l679_679758

open Set

theorem num_nonempty_subsets_of_odds :
  let odds := {1, 3, 5, 7, 9}
  (2 ^ odds.size - 1) = 31 :=
by
  let odds := {1, 3, 5, 7, 9}
  have h_size : odds.size = 5 := by simp
  have h_subsets : 2 ^ odds.size = 32 := by norm_num
  show (2 ^ odds.size - 1) = 31 from
    by norm_num


end num_nonempty_subsets_of_odds_l679_679758


namespace friend_initial_money_l679_679204

theorem friend_initial_money (F : ℕ) : 
    (160 + 25 * 7 = F + 25 * 5) → 
    (F = 210) :=
by
  sorry

end friend_initial_money_l679_679204


namespace total_feet_in_garden_l679_679166

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l679_679166


namespace total_votes_l679_679976

variable (V : ℝ)

theorem total_votes (h1 : 0.34 * V + 640 = 0.66 * V) : V = 2000 :=
by 
  sorry

end total_votes_l679_679976


namespace smallest_positive_integer_l679_679767

theorem smallest_positive_integer (k : ℕ) :
  (∃ k : ℕ, ((2^4 ∣ 1452 * k) ∧ (3^3 ∣ 1452 * k) ∧ (13^3 ∣ 1452 * k))) → 
  k = 676 := 
sorry

end smallest_positive_integer_l679_679767


namespace num_integers_condition_l679_679757

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem num_integers_condition : 
  {n : ℕ // n > 0 ∧ n < 2000 ∧ n = 9 * sum_of_digits n}.card = 4 := 
by sorry

end num_integers_condition_l679_679757


namespace powers_of_three_two_digit_count_l679_679030

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679030


namespace sum_of_first_n_nat_sum_of_products_nat_sum_of_squares_nat_sum_of_cubes_nat_l679_679462

-- 1. Sum of first n natural numbers.
theorem sum_of_first_n_nat (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i) = n * (n + 1) / 2 :=
by
  sorry

-- 2. Sum of products of consecutive natural numbers.
theorem sum_of_products_nat (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i * (i + 1)) = n * (n + 1) * (n + 2) / 3 :=
by
  sorry

-- 3. Sum of squares of first n natural numbers.
theorem sum_of_squares_nat (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i^2) = n * (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

-- 4. Sum of cubes of first n natural numbers.
theorem sum_of_cubes_nat (n : ℕ) : 
  (∑ i in Finset.range (n + 1), i^3) = (n * (n + 1) / 2)^2 :=
by
  sorry

end sum_of_first_n_nat_sum_of_products_nat_sum_of_squares_nat_sum_of_cubes_nat_l679_679462


namespace square_area_from_isosceles_triangle_l679_679612

theorem square_area_from_isosceles_triangle:
  ∀ (b h : ℝ) (Side_of_Square : ℝ), b = 2 ∧ h = 3 ∧ Side_of_Square = (6 / 5) 
  → (Side_of_Square ^ 2) = (36 / 25) := 
by
  intro b h Side_of_Square
  rintro ⟨hb, hh, h_side⟩
  sorry

end square_area_from_isosceles_triangle_l679_679612


namespace boy_reaches_early_l679_679551

-- Given conditions
def usual_time : ℚ := 42
def rate_multiplier : ℚ := 7 / 6

-- Derived variables
def new_time : ℚ := (6 / 7) * usual_time
def early_time : ℚ := usual_time - new_time

-- The statement to prove
theorem boy_reaches_early : early_time = 6 := by
  sorry

end boy_reaches_early_l679_679551


namespace area_BEDC_is_120_l679_679132

-- Definitions of the conditions
def BC : ℝ := 15
def ED : ℝ := 9
def area_ABCD : ℝ := 150
def BE : ℝ := 10 -- assumed from context

-- Definition of the area calculation for the shaded region BEDC
def area_triangle_ABE : ℝ := 1 / 2 * BE * (BC - ED)

-- Lean problem statement to prove the equality
theorem area_BEDC_is_120 (h_BC : BC = 15) (h_ED : ED = 9) (h_area_ABCD : area_ABCD = 150) (h_BE : BE = 10) :
  area_ABCD - area_triangle_ABE = 120 :=
by 
  sorry

end area_BEDC_is_120_l679_679132


namespace man_total_earnings_l679_679233

-- Define the conditions
def total_days := 30
def wage_per_day := 10
def fine_per_absence := 2
def days_absent := 7
def days_worked := total_days - days_absent
def earned := days_worked * wage_per_day
def fine := days_absent * fine_per_absence
def total_earnings := earned - fine

-- State the theorem
theorem man_total_earnings : total_earnings = 216 := by
  -- Using the definitions provided, the proof should show that the calculations result in 216
  sorry

end man_total_earnings_l679_679233


namespace smallest_integer_cost_equal_l679_679319

def cost_decimal (n : ℕ) : ℕ :=
  n.digits 10 |>.map (λ d => d.to_nat) |>.sum

def cost_binary (n : ℕ) : ℕ :=
  n.digits 2 |>.map (λ d => d.to_nat) |>.sum

theorem smallest_integer_cost_equal (n : ℕ) : 1000 < n ∧ n < 2000 ∧ cost_decimal n = cost_binary n → n = 1101 :=
  by 
  sorry

end smallest_integer_cost_equal_l679_679319


namespace product_of_two_numbers_l679_679157

theorem product_of_two_numbers (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 6) : x * y = 616 :=
sorry

end product_of_two_numbers_l679_679157


namespace ω_range_l679_679724

-- Define the function f(x) and its range
def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

-- The interval for x
def x_interval : Set ℝ := { x | -Real.pi / 3 <= x ∧ x <= Real.pi / 4 }

-- The condition for the minimum value
def min_value_condition (ω : ℝ) : Prop := ∀ x ∈ x_interval, f ω x >= -2

-- The solution statement
theorem ω_range :
  {ω | ∀ x ∈ x_interval, min_value_condition ω} = 
  {ω | ω ≤ -2 ∨ ω ≥ 3 / 2} :=
sorry

end ω_range_l679_679724


namespace units_digit_of_expression_l679_679274

theorem units_digit_of_expression :
  ∃ d : ℕ, (d < 10) ∧ 
  (∀ (x y : ℕ), (x % 10 = 4) → (y % 10 = 2) → 
  ((x^4 + y^4 + x^2 + y^2) % 10 = d)) :=
begin
  use 2,
  intros x y hx hy,
  have hx4 : (x % 10)^4 % 10 = 6, { sorry },
  have hy4 : (y % 10)^4 % 10 = 6, { sorry },
  have hx2 : (x % 10)^2 % 10 = 6, { sorry },
  have hy2 : (y % 10)^2 % 10 = 4, { sorry },
  calc
    (x^4 + y^4 + x^2 + y^2) % 10
        = ((x % 10)^4 + (y % 10)^4 + (x % 10)^2 + (y % 10)^2) % 10 : sorry
    ... = (6 + 6 + 6 + 4) % 10 : by rw [hx4, hy4, hx2, hy2]
    ... = 22 % 10 : by norm_num
    ... = 2 : by norm_num,
end

end units_digit_of_expression_l679_679274


namespace number_of_circles_passing_through_F_and_M_l679_679359

theorem number_of_circles_passing_through_F_and_M (
  p : ℝ := 2
) : 
  let F : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (4, 4)
  ∀ (g h : ℝ),
  (4 - g)^2 + (4 - h)^2 = (1 + g)^2 ∧
  (1 - g)^2 + h^2 = (1 + g)^2 → 
    -3 * h^2 + 8 * h - 15 = 0 →
  2 :=
sorry

end number_of_circles_passing_through_F_and_M_l679_679359


namespace team_A_first_game_win_probability_l679_679877

-- Definitions based on conditions
def equally_likely : ℕ → ℕ → bool := λ a b, true
def no_ties : bool := true
def independent_outcomes : bool := true

-- Probability function
noncomputable def probability_first_game_win 
  (team_A_wins_third : Prop)
  (team_B_wins_series : Prop) : ℚ :=
  if team_A_wins_third ∧ team_B_wins_series then 1/2 else 0

-- Theorem statement
theorem team_A_first_game_win_probability
  (team_A_wins_third : Prop)
  (team_B_wins_series : Prop)
  (eq_likely: equally_likely 1 1)
  (no_tie: no_ties)
  (indep_outcomes: independent_outcomes)
  (team_A_cond: team_A_wins_third = true)
  (team_B_cond: team_B_wins_series = true):
  probability_first_game_win team_A_wins_third team_B_wins_series = 1/2 := 
sorry

end team_A_first_game_win_probability_l679_679877


namespace subsets_union_card_eq_eight_l679_679705

theorem subsets_union_card_eq_eight :
  let A := {1, 2}
  let B := {2, 3}
  # finset.powerset (A ∪ B) = 8 := 
by
  let A := {1, 2}
  let B := {2, 3}
  let union_set := A ∪ B
  have : union_set = {1, 2, 3} := by simp [A, B]
  have subset_count := finset.card (finset.powerset union_set)
  have : subset_count = 8 := by simp [union_set, finset.card_powerset]
  exact this

end subsets_union_card_eq_eight_l679_679705


namespace train_speed_is_72_km_per_hr_l679_679248

-- Given conditions
def train_length : ℝ := 140
def bridge_length : ℝ := 132
def time_to_cross_bridge : ℝ := 13.598912087033037

-- Calculated values
def total_distance : ℝ := train_length + bridge_length
def speed_m_per_s : ℝ := total_distance / time_to_cross_bridge
def km_per_hour_conversion : ℝ := 3.6
def speed_km_per_hr : ℝ := speed_m_per_s * km_per_hour_conversion

-- Theorem to prove
theorem train_speed_is_72_km_per_hr : speed_km_per_hr = 72 := by
  sorry

end train_speed_is_72_km_per_hr_l679_679248


namespace count_valid_subsets_eq_371_l679_679444

open Finset

def S : Finset ℕ := {1, 2, ..., 15}

def is_valid_subset (A : Finset ℕ) : Prop :=
  ∃ a1 a2 a3, A = {a1, a2, a3} ∧ 1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 15 ∧ a3 - a2 ≤ 6

def count_valid_subsets : ℕ :=
  (S.powerset.filter is_valid_subset).card

theorem count_valid_subsets_eq_371 : count_valid_subsets = 371 :=
  sorry

end count_valid_subsets_eq_371_l679_679444


namespace sum_k_expression_l679_679185

theorem sum_k_expression (a b c : ℕ) (hb : b = 50) (ha : a = 2702) (hc : c = 1) :
  (\(\sum_{k = 1}^{50} (-1)^k \cdot \frac{k^3 + k^2 + k + 1}{k!}) = (\frac{a}{b!} - c)) → (a + b + c = 2753) :=
  by
  sorry

end sum_k_expression_l679_679185


namespace inequalities_l679_679827

-- Definitions of sigma-algebras and mixing coefficients
variables {Ω : Type*} {P : Measure Ω}
variables (A B : Set Ω) (𝒜 𝒝 : MeasurableSet Ω)

/-- Definitions of mixing coefficients -/
def alpha (𝒜 𝒝 : MeasurableSet Ω) := Sup { |P[A ∩ B] - P[A] * P[B]| | A ∈ 𝒜, B ∈ 𝒝 }

def varphi (𝒜 𝒝 : MeasurableSet Ω) := Sup { |P[B | A] - P[B]| | A ∈ 𝒜, B ∈ 𝒝, P[A] > 0 }

def psi (𝒜 𝒝 : MeasurableSet Ω) := Sup { | (P[A ∩ B]) / (P[A] * P[B]) - 1 | | A ∈ 𝒜, B ∈ 𝒝, P[A] * P[B] > 0 }

def beta (𝒜 𝒝 : MeasurableSet Ω) := Sup { 1/2 * ∑ i in Finset.univ, ∑ j in Finset.univ, | P[A_i ∩ B_j] - P[A_i] * P[B_j]| | A_i ∈ 𝒜, B_j ∈ 𝒝 }

def I (𝒜 𝒝 : MeasurableSet Ω) := Sup { ∑ i in Finset.univ, ∑ j in Finset.univ, P[A_i ∩ B_j] * log (P[A_i ∩ B_j] / (P[A_i] * P[B_j])) | A_i ∈ 𝒜, B_j ∈ 𝒝 }

def rho_star (𝒜 𝒝 : MeasurableSet Ω) := -- Definition not provided in the problem, taking as a given

theorem inequalities (𝒜 𝒝 : MeasurableSet Ω):
  2 * alpha 𝒜 𝒝 ≤ beta 𝒜 𝒝 ∧ beta 𝒜 𝒝 ≤ varphi 𝒜 𝒝 ∧ varphi 𝒜 𝒝 ≤ psi 𝒜 𝒝 / 2 ∧
  4 * alpha 𝒜 𝒝 ≤ rho_star 𝒜 𝒝 ∧ rho_star 𝒜 𝒝 ≤ psi 𝒜 𝒝 ∧
  rho_star 𝒜 𝒝 ≤ 2 * sqrt (varphi 𝒜 𝒝) * sqrt (varphi 𝒝 𝒜) ∧
  sqrt(2) * beta 𝒜 𝒝 ≤ sqrt(I 𝒜 𝒝) :=
sorry

end inequalities_l679_679827


namespace books_distribution_l679_679793

theorem books_distribution :
  ∃ (n : ℕ), (n = 5) →
  ∃ (k : ℕ), (k = 4) →
  ∃ (ways : ℕ), (ways = 240) →
  ways = (∑ s in ((Finset.range k).powerset.filter (λ s, s.card = n%k)), 
               (∏ j in Finset.range (Finset.card s), nat.choose (n - j * k, k - j))) :=
  sorry

end books_distribution_l679_679793


namespace parallelogram_reflection_l679_679854

def H'' (E F G H : (ℝ × ℝ)) : Prop :=
  reflect_across_line (reflect_across_x_axis H) (λ p, (p.1, p.2 - 1)) = (2, 5)

theorem parallelogram_reflection (E F G H : (ℝ × ℝ))
  (hE : E = (3, 3)) (hF : F = (6, 7)) (hG : G = (9, 3)) (hH : H = (6, -1)) :
  H'' E F G H :=
by
  sorry

end parallelogram_reflection_l679_679854


namespace sum_squares_mod_13_is_zero_l679_679939

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l679_679939


namespace two_digit_numbers_of_form_3_pow_n_l679_679023

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679023


namespace product_inequality_l679_679438

-- Define the proof problem in Lean 4
theorem product_inequality (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) :
  (∏ i, (1 + x i)) ≤ ∑ k in Finset.range (n+1), (Finset.univ.sum x)^k / (Nat.factorial k) :=
by
  sorry

end product_inequality_l679_679438


namespace number_of_regions_divided_by_simple_hyperplanes_l679_679222

-- Define the condition for simple hyperplanes
def is_simple_hyperplane (k1 k2 k3 k4 : ℤ) : Prop :=
  k1 ∈ {-1, 0, 1} ∧ k2 ∈ {-1, 0, 1} ∧ k3 ∈ {-1, 0, 1} ∧ k4 ∈ {-1, 0, 1} ∧ (k1 ≠ 0 ∨ k2 ≠ 0 ∨ k3 ≠ 0 ∨ k4 ≠ 0)

-- Define the function R that calculates the number of regions in which k hyperplanes divide ℝ^n
noncomputable def R (n k : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), Nat.choose k i

-- Given the above definitions, we state the theorem
theorem number_of_regions_divided_by_simple_hyperplanes :
  R 4 80 = 1661981 :=
by
  -- Proof will be provided here
  sorry

end number_of_regions_divided_by_simple_hyperplanes_l679_679222


namespace repeating_decimal_to_fraction_l679_679300

theorem repeating_decimal_to_fraction : (x : ℝ) (h : x = 0.353535...) → x = 35 / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l679_679300


namespace fraction_ratio_equivalence_l679_679956

theorem fraction_ratio_equivalence :
  ∃ (d : ℚ), d = 240 / 1547 ∧ ((2 / 13) / d) = ((5 / 34) / (7 / 48)) := 
by
  sorry

end fraction_ratio_equivalence_l679_679956


namespace find_x_l679_679681

theorem find_x (x : ℕ) (hx1 : 1 ≤ x) (hx2 : x ≤ 100) (hx3 : (31 + 58 + 98 + 3 * x) / 6 = 2 * x) : x = 21 :=
by
  sorry

end find_x_l679_679681


namespace urn_problem_l679_679631

theorem urn_problem (N : ℕ) (h : (0.5 * (10 / (10 + N)) + 0.5 * (N / (10 + N))) = 0.52) : N = 15 :=
sorry

end urn_problem_l679_679631


namespace measure_angle_QIU_l679_679413

theorem measure_angle_QIU (P Q R U V W I : Type)
  (PQ QR RP : ℕ)
  (h_ratio : PQ : QR : RP = 3 : 4 : 5)
  (h_bisectors : is_angle_bisector PU P Q R ∧ 
                 is_angle_bisector QV Q P R ∧ 
                 is_angle_bisector RW R P Q)
  (h_incenter : incenter I P Q R PU QV RW)
  (h_angle_PRQ : ∠PRQ = 48) :
  ∠QIU = 66 :=
begin
  sorry
end

end measure_angle_QIU_l679_679413


namespace cos_angle_BAC_is_expected_l679_679344

variable (A B C : ℝ × ℝ)

noncomputable def vec := (p1 : ℝ × ℝ) → (p2 : ℝ × ℝ) → ℝ × ℝ
  | (x1, y1), (x2, y2) => (x2 - x1, y2 - y1)

noncomputable def dot_product := (v1 : ℝ × ℝ) → (v2 : ℝ × ℝ) → ℝ
  | (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

noncomputable def magnitude := (v : ℝ × ℝ) → ℝ
  | (x, y) => Real.sqrt (x^2 + y^2)

noncomputable def cos_angle (A B C : ℝ × ℝ) : ℝ :=
  let AB := vec A B
  let AC := vec A C
  dot_product AB AC / (magnitude AB * magnitude AC)

theorem cos_angle_BAC_is_expected : cos_angle (1, -2) (-3, 1) (5, 2) = -Real.sqrt 2 / 10 :=
  by
    sorry

end cos_angle_BAC_is_expected_l679_679344


namespace ball_hits_9_walls_l679_679990

theorem ball_hits_9_walls :
  let lcm_12_10 := 60 in
  let x_intervals := 12 in
  let y_intervals := 10 in
  (lcm_12_10 / x_intervals - 1) + (lcm_12_10 / y_intervals - 1) = 9 :=
by
  let lcm_12_10 := 60
  have h_x : lcm_12_10 / 12 = 5 := by norm_num
  have h_y : lcm_12_10 / 10 = 6 := by norm_num
  rw [h_x, h_y]
  norm_num

end ball_hits_9_walls_l679_679990


namespace find_apex_angle_of_identical_cones_l679_679174

noncomputable def apex_angle_of_identical_cones : ℝ :=
  2 * Real.arctan (4 / 5)

theorem find_apex_angle_of_identical_cones
  (A : Point)
  (cone1 cone2 : Cone)
  (cone3 : Cone)
  (plane : Plane): 
  (vertex cone1 = A) →
  (vertex cone2 = A) →
  (vertex cone3 = A) →
  (apex_angle cone3 = Real.pi / 2) →
  (touches plane cone1) →
  (touches plane cone2) →
  (touches plane cone3) →
  (touches cone1 cone2) →
  (touches cone1 cone3) →
  (touches cone2 cone3) →
  apex_angle cone1 = apex_angle_of_identical_cones :=
by
  sorry

end find_apex_angle_of_identical_cones_l679_679174


namespace part1_part2_l679_679329

variables (α : ℝ)

-- Condition: tan α = 3
def tan_alpha : α → ℝ := λ α, 3

-- First theorem:
theorem part1 (h : tan α = 3) :
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 :=
by sorry

-- Second theorem:
theorem part2 (h : tan α = 3) :
  1 - 4 * sin α * cos α + 2 * cos α ^ 2 = 0 :=
by sorry

end part1_part2_l679_679329


namespace find_x_l679_679305

theorem find_x (x : ℝ) (h : 2 * arctan (1 / 5) + 2 * arctan (1 / 10) + arctan (1 / x) = π / 2) : x = 120 / 119 := 
sorry

end find_x_l679_679305


namespace distance_between_Sasha_and_Kolya_l679_679469

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679469


namespace min_n_partition_l679_679091

theorem min_n_partition (n : ℕ) (h : n ≥ 2) 
    (T : Finset ℕ := Finset.range (nat.succ n) \ Finset.range 1 |>.map (λ k => 2^k)) :
    (∀ (X Y : Finset ℕ), X ∪ Y = T → (∃ a b c ∈ X, a * b = c) ∨ (∃ a b c ∈ Y, a * b = c)) ↔ n ≥ 5 := 
by
  sorry

end min_n_partition_l679_679091


namespace sum_of_triangle_sides_l679_679403

-- Define the problem as a Lean statement
theorem sum_of_triangle_sides (A B C : Type) 
  (angle_A : ℝ) (angle_C : ℝ) (side_BC : ℝ) :
  angle_A = 30 ∧ angle_C = 60 ∧ side_BC = 8 → 
  ∃ sum_of_sides : ℝ, sum_of_sides = 18.9 :=
begin
  sorry
end

end sum_of_triangle_sides_l679_679403


namespace find_a_l679_679098

theorem find_a (a x : ℝ) (h1 : a < -1) (h2 : ∀ x, x^2 + a * x ≤ -x) (h3 : is_min (λ x, x^2 + a * x) (-1/2)) : 
  a = -3/2 :=
sorry

noncomputable def is_min (f : ℝ → ℝ) (m : ℝ) : Prop :=
∀ x, f x ≥ m ∧ (∃ the X, m = f the X)


end find_a_l679_679098


namespace lambda_range_l679_679654

noncomputable def f : ℝ → ℝ :=
  λ x, if x < 0 then - (2^x) / (4^x + 1)
       else if x = 0 then 0
       else (2^x) / (4^x + 1)

theorem lambda_range (λ : ℝ) :
  (∀ x > 1, (2^x) / (f x) - λ * 2^x - 2 ≥ 0) → λ ≤ 3 / 2 :=
begin
  intro h,
  sorry
end

end lambda_range_l679_679654


namespace remainder_of_largest_divided_by_second_smallest_l679_679170

theorem remainder_of_largest_divided_by_second_smallest 
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  c % b = 1 :=
by
  -- We assume and/or prove the necessary statements here.
  -- The core of the proof uses existing facts or assumptions.
  -- We insert the proof strategy or intermediate steps here.
  
  sorry

end remainder_of_largest_divided_by_second_smallest_l679_679170


namespace angle_EDF_140_degrees_l679_679412

variable {A B C D E F : Type}
variable [IsoscelesTriangle A B C]
variable (AB_eq_AC : AB = AC)
variable (angle_A : ∠A = 100°)
variable (D_midpoint_BC : IsMidpoint D B C)
variable (CE_eq_ED : CE = ED)
variable (BF_eq_FD : BF = FD)

theorem angle_EDF_140_degrees (A B C D E F : Type) [IsoscelesTriangle A B C]
  (AB_eq_AC : AB = AC) (angle_A : ∠A = 100°) 
  (D_midpoint_BC : IsMidpoint D B C) 
  (CE_eq_ED : CE = ED) 
  (BF_eq_FD : BF = FD) 
  : ∠EDF = 140° := sorry

end angle_EDF_140_degrees_l679_679412


namespace repeating_decimal_as_fraction_l679_679301

theorem repeating_decimal_as_fraction : ∀ x : ℝ, (x = 0.353535...) → (99 * x = 35) → x = 35 / 99 := by
  intros x h1 h2
  sorry

end repeating_decimal_as_fraction_l679_679301


namespace triangle_side_c_l679_679393

-- Proving the value of side c given the conditions in triangle ABC
theorem triangle_side_c (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos B + b * Real.cos A = 2)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) : c = 2 :=
sorry

end triangle_side_c_l679_679393


namespace exp_continuous_at_l679_679864

theorem exp_continuous_at (a α : ℝ) (h : 0 < a) : 
  filter.tendsto (λ x, a^x) (nhds α) (nhds (a^α)) :=
sorry

end exp_continuous_at_l679_679864


namespace Ram_independent_days_l679_679859

-- Definitions based on the given conditions
def Ram_work_rate (R : ℝ) := 1 / R
def Gohul_work_rate := 1 / 15
def combined_work_rate (R : ℝ) := Ram_work_rate R + Gohul_work_rate

-- Theorem statement
theorem Ram_independent_days (R : ℝ) (h : combined_work_rate R = 1 / 6) : R = 10 := by
  sorry

end Ram_independent_days_l679_679859


namespace limit_f_at_zero_l679_679214

open Real Filter

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt (1 + tan x) - sqrt (1 + sin x)) / (x^3)

theorem limit_f_at_zero :
  tendsto f (nhds 0) (nhds (1/4)) :=
begin
  sorry
end

end limit_f_at_zero_l679_679214


namespace number_of_cans_l679_679815

theorem number_of_cans (carry_cans : ℕ) (drain_time : ℕ) (walk_time_per_way : ℕ) (total_time : ℕ) :
  carry_cans = 4 → drain_time = 30 → walk_time_per_way = 10 → total_time = 350 →
  let round_trip_time := drain_time + 2 * walk_time_per_way in
  total_time / round_trip_time * carry_cans = 28 :=
by
  intros h_carry_cans h_drain_time h_walk_time h_total_time
  simp [h_carry_cans, h_drain_time, h_walk_time, h_total_time]
  let round_trip_time := 30 + 2 * 10
  have ht : total_time / round_trip_time * carry_cans = 28 := sorry
  exact ht

end number_of_cans_l679_679815


namespace find_min_value_of_a_l679_679720

noncomputable def f (a x : ℝ) := real.exp(x) * (x^3 - 3*x + 3) - a * real.exp(x) - x

theorem find_min_value_of_a (a : ℝ) :
  (∃ x : ℝ, x ≥ -2 ∧ f a x ≤ 0) → a ≥ 1 - (1 / real.exp(1)) :=
by
  sorry

end find_min_value_of_a_l679_679720


namespace find_number_l679_679876

theorem find_number (x : ℕ) (h : x + 15 = 96) : x = 81 := 
sorry

end find_number_l679_679876


namespace f_neg_3_value_l679_679579

-- Define the condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the function f for x ≥ 0
def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -(x^2 + x)

-- The main statement to be proved: f(-3) = -12
theorem f_neg_3_value :
  is_odd_function f →
  (∀ x : ℝ, x ≥ 0 → f x = x^2 + x) →
  f (-3) = -12 :=
  by
    intros hf hfx
    sorry

end f_neg_3_value_l679_679579


namespace count_special_numbers_l679_679755

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem count_special_numbers : 
  (Finset.filter (λ n, n = 9 * (sum_of_digits n)) (Finset.range 2000)).card = 4 :=
sorry

end count_special_numbers_l679_679755


namespace minimize_cylinder_surface_area_l679_679717

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def cylinder_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem minimize_cylinder_surface_area :
  ∃ r h : ℝ, cylinder_volume r h = 16 * Real.pi ∧
  (∀ r' h', cylinder_volume r' h' = 16 * Real.pi → cylinder_surface_area r h ≤ cylinder_surface_area r' h') ∧ r = 2 := by
  sorry

end minimize_cylinder_surface_area_l679_679717


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679497

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679497


namespace texts_on_friday_l679_679847

-- Define the constants and parameters
constant cost_per_text : ℝ := 0.1
constant total_budget : ℝ := 20.0

-- Define the number of texts sent each day
def texts_sent_on_monday : ℕ := 5 * 4
def texts_sent_on_tuesday : ℕ := 15 + 10 + 12 + 8
def texts_sent_on_wednesday : ℕ := 20 + 18 + 7 + 14
def texts_sent_on_thursday : ℕ := 0 + 25 + 10 + 5

-- Define the total number of texts sent from Monday to Thursday
def total_texts_sent : ℕ := texts_sent_on_monday + texts_sent_on_tuesday + texts_sent_on_wednesday + texts_sent_on_thursday

-- Define the cost calculations
def total_cost : ℝ := total_texts_sent * cost_per_text
def remaining_budget : ℝ := total_budget - total_cost

-- Calculate the maximum number of texts that can be sent on Friday
def max_texts_on_friday : ℕ := (remaining_budget / cost_per_text).floor.to_nat

-- Assert the expected result
theorem texts_on_friday : max_texts_on_friday = 36 :=
by
  sorry

end texts_on_friday_l679_679847


namespace p_plus_q_expression_l679_679728

-- Define the polynomial forms
def p (x : ℝ) := a₀ * x^2 + a₁ * x + a₂
def q (x : ℝ) := b₀ * x^3 + b₁ * x^2 + b₂ * x + b₃

-- Given Conditions
axiom p_quadratic : ∃ a₀ a₁ a₂ : ℝ, p = fun x => a₀ * x^2 + a₁ * x + a₂
axiom q_cubic : ∃ b₀ b₁ b₂ b₃ : ℝ, q = fun x => b₀ * x^3 + b₁ * x^2 + b₂ * x + b₃

axiom p_at_4 : p 4 = 4
axiom q_at_1 : q 1 = 0
axiom q_at_3 : q 3 = 3
axiom q_asymptote : ∃ c : ℝ, q = fun x => c * (x - 1) * (x - 2) * x

noncomputable def p_plus_q (x : ℝ) := p x + q x

theorem p_plus_q_expression : p_plus_q x = (1/2) * x^3 - (5/4) * x^2 + (17/4) * x := by
  sorry

end p_plus_q_expression_l679_679728


namespace find_angle_A_l679_679700

theorem find_angle_A (a b c : ℝ) (h : a^2 = b^2 + c^2 - b * c) : 
  ∠A = π / 3 := 
begin
  sorry
end

end find_angle_A_l679_679700


namespace floor_abs_sum_seq_l679_679507

theorem floor_abs_sum_seq (x : ℕ → ℝ) 
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1005 → x 1 + 1 = x 2 + 3 = x 3 + 6 = ... = x n + n * (n + 1) / 2 = ∑ i in (finset.range 1005), x i + 1006) : 
  ∃ (S : ℝ), S = ∑ i in (finset.range 1005), x i ∧ ⌊|S|⌋ = 1517552 := 
by 
  sorry

end floor_abs_sum_seq_l679_679507


namespace choose_non_consecutive_l679_679110

open Finset

/-- There are 8 ways to choose 3 numbers from {1, 2, 3, 4, 5, 6, 7, 8} such that no two are consecutive. -/
theorem choose_non_consecutive : 
  {s : Finset ℕ // s ⊆ (range 8).map (λ n, n + 1) ∧ s.card = 3 ∧ ∀ (x ∈ s) (y ∈ s), abs (x - y) ≠ 1 }.card = 8 :=
sorry

end choose_non_consecutive_l679_679110


namespace QR_length_l679_679780

noncomputable theory

open_locale big_operators

structure Triangle :=
(A B C : ℝ)
(sides : {AB AC BC : ℝ})

def triangle_ABC : Triangle := 
{ A := 10, 
  B := 8, 
  C := 6, 
  sides := {AB := 10, AC := 8, BC := 6} }

def circle_P_tangent_to_AB_and_pass_through_C (triangle : Triangle) : Prop := 
tangent_to_AB_circle_P ∧ passes_through_C_circle_P

theorem QR_length {triangle : Triangle} 
  (h₁ : triangle = triangle_ABC)
  (h₂ : circle_P_tangent_to_AB_and_pass_through_C triangle) : 
triangle.QR = 4.8 :=
sorry

end QR_length_l679_679780


namespace part1_part2_l679_679441

section problem

variable (a : ℝ)

def setA (a : ℝ) := {x : ℝ | a - 3 < x ∧ x < a + 3}
def setB := {x : ℝ | x < -1 ∨ x > 3}

theorem part1 : setA 3 ∪ setB = {x : ℝ | x < -1 ∨ x > 0} :=
by sorry

theorem part2 (h : (setA a ∪ setB) = set.univ) : 0 < a ∧ a < 4 :=
by sorry

end problem

end part1_part2_l679_679441


namespace simplify_sqrt_product_l679_679124

theorem simplify_sqrt_product (y : ℝ) (hy : y > 0) : 
  (Real.sqrt (45 * y) * Real.sqrt (20 * y) * Real.sqrt (30 * y) = 30 * y * Real.sqrt (30 * y)) :=
by
  sorry

end simplify_sqrt_product_l679_679124


namespace probability_of_johns_8th_roll_l679_679380

noncomputable def probability_johns_8th_roll_is_last : ℚ :=
  (7/8)^6 * (1/8)

theorem probability_of_johns_8th_roll :
  probability_johns_8th_roll_is_last = 117649 / 2097152 := by
  sorry

end probability_of_johns_8th_roll_l679_679380


namespace largest_difference_l679_679086

noncomputable def A : ℕ := 3 * 2010 ^ 2011
noncomputable def B : ℕ := 2010 ^ 2011
noncomputable def C : ℕ := 2009 * 2010 ^ 2010
noncomputable def D : ℕ := 3 * 2010 ^ 2010
noncomputable def E : ℕ := 2010 ^ 2010
noncomputable def F : ℕ := 2010 ^ 2009

theorem largest_difference :
  (A - B = 2 * 2010 ^ 2011) ∧ 
  (B - C = 2010 ^ 2010) ∧ 
  (C - D = 2006 * 2010 ^ 2010) ∧ 
  (D - E = 2 * 2010 ^ 2010) ∧ 
  (E - F = 2009 * 2010 ^ 2009) ∧ 
  (2 * 2010 ^ 2011 > 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2006 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2009 * 2010 ^ 2009) :=
by
  sorry

end largest_difference_l679_679086


namespace find_required_school_year_hours_l679_679080

-- Define constants for the problem
def summer_hours_per_week : ℕ := 40
def summer_weeks : ℕ := 12
def summer_earnings : ℕ := 6000
def school_year_weeks : ℕ := 36
def school_year_earnings : ℕ := 9000

-- Calculate total summer hours, hourly rate, total school year hours, and required school year weekly hours
def total_summer_hours := summer_hours_per_week * summer_weeks
def hourly_rate := summer_earnings / total_summer_hours
def total_school_year_hours := school_year_earnings / hourly_rate
def required_school_year_hours_per_week := total_school_year_hours / school_year_weeks

-- Prove the required hours per week is 20
theorem find_required_school_year_hours : required_school_year_hours_per_week = 20 := by
  sorry

end find_required_school_year_hours_l679_679080


namespace average_payment_is_442_50_l679_679585

-- The conditions given in the problem
def first_20_payment (n : ℕ) : ℝ := if n < 20 then 410 else 0
def remaining_20_payment (n : ℕ) : ℝ := if 20 ≤ n ∧ n < 40 then 410 + 65 else 0
def total_payment (n : ℕ) : ℝ := first_20_payment n + remaining_20_payment n

-- The proposition to be proven
theorem average_payment_is_442_50 : 
    (∑ k in finset.range 40, total_payment k) / 40 = 442.50 :=
by
  -- proof omitted for brevity.
  sorry

end average_payment_is_442_50_l679_679585


namespace solve_for_s_and_t_l679_679323

theorem solve_for_s_and_t : ∃ s t : ℤ, 15 * s + 10 * t = 270 ∧ s = 3 * t - 4 ∧ s = 14 ∧ t = 6 :=
by
  use 14, 6
  split
  · exact calc
    15 * 14 + 10 * 6 = 210 + 60 := by norm_num
                   ... = 270 := by norm_num
  · split
    · rfl
    · split
      · rfl
      · rfl

end solve_for_s_and_t_l679_679323


namespace maximum_triangle_area_le_8_l679_679565

def lengths : List ℝ := [2, 3, 4, 5, 6]

-- Function to determine if three lengths can form a valid triangle
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a 

-- Heron's formula to compute the area of a triangle given its sides
noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Statement to prove that the maximum possible area with given stick lengths is less than or equal to 8 cm²
theorem maximum_triangle_area_le_8 :
  ∃ (a b c : ℝ), a ∈ lengths ∧ b ∈ lengths ∧ c ∈ lengths ∧ 
  is_valid_triangle a b c ∧ heron_area a b c ≤ 8 :=
sorry

end maximum_triangle_area_le_8_l679_679565


namespace value_of_a_l679_679945

noncomputable def equation_has_three_solutions (a : ℝ) : Prop :=
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3
  ∧ (| |x1 + 3| - 1 | = a) ∧ (| |x2 + 3| - 1 | = a) ∧ (| |x3 + 3| - 1 | = a)
  
theorem value_of_a : ∀ (a : ℝ), equation_has_three_solutions a ↔ a = 1 :=
by
  sorry

end value_of_a_l679_679945


namespace ratio_of_circumscribed_areas_l679_679604

noncomputable def rect_pentagon_circ_ratio (P : ℝ) : ℝ :=
  let s : ℝ := P / 8
  let r_circle : ℝ := (P * Real.sqrt 10) / 16
  let A : ℝ := Real.pi * (r_circle ^ 2)
  let pentagon_side : ℝ := P / 5
  let R_pentagon : ℝ := P / (10 * Real.sin (Real.pi / 5))
  let B : ℝ := Real.pi * (R_pentagon ^ 2)
  A / B

theorem ratio_of_circumscribed_areas (P : ℝ) : rect_pentagon_circ_ratio P = (5 * (5 - Real.sqrt 5)) / 64 :=
by sorry

end ratio_of_circumscribed_areas_l679_679604


namespace average_age_new_students_l679_679882

theorem average_age_new_students (N n : ℕ) (A : ℕ) : 
  (40 - 6 = A) ∧ ((40 * N + A * n) / (N + n) = 36) → A = 34 :=
by 
  intros h,
  cases h with h1 h2,
  rw h1,
  sorry

end average_age_new_students_l679_679882


namespace solution_of_system_of_equations_l679_679898

-- Define the conditions of the problem.
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 6) ∧ (x = 2 * y)

-- Define the correct answer as a set.
def solution_set : Set (ℝ × ℝ) :=
  { (4, 2) }

-- State the proof problem.
theorem solution_of_system_of_equations : 
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ system_of_equations x y} = solution_set :=
  sorry

end solution_of_system_of_equations_l679_679898


namespace angle_BFE_given_conditions_l679_679803

theorem angle_BFE_given_conditions
  {A B C D E : Point}
  (hCol1 : Collinear A B C)
  (hCol2 : Collinear D B E)
  (hRadii1 : dist A B = dist A E)
  (hRadii2 : dist B D = dist B C)
  (hAngleDBC : angle B D C = 57) :
  angle B F E = 24 :=
sorry

end angle_BFE_given_conditions_l679_679803


namespace width_of_room_correct_l679_679890

noncomputable def length_of_room : ℝ := 5.5
noncomputable def total_cost : ℝ := 24750
noncomputable def rate_per_sq_meter : ℝ := 1200
noncomputable def area_of_floor : ℝ := total_cost / rate_per_sq_meter
noncomputable def width_of_room : ℝ := area_of_floor / length_of_room

theorem width_of_room_correct : width_of_room = 3.75 :=
by
    -- Skipping the proof steps with sorry
    sorry

end width_of_room_correct_l679_679890


namespace log_three_div_square_l679_679116

theorem log_three_div_square (x y : ℝ) (h₁ : x ≠ 1) (h₂ : y ≠ 1) (h₃ : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h₄ : x * y = 243) :
  (Real.log (x / y) / Real.log 3) ^ 2 = 9 := 
sorry

end log_three_div_square_l679_679116


namespace number_of_liars_is_eleven_l679_679547

-- Formalize candidates and their statements
def candidate (n : Nat) : Prop := 
  match n with
  | 1 => "We've lied once so far."
  | 2 => "Now, we've lied twice."
  | 3 => "Three times now."
  | 4 => "Four times now."
  | 5 => "Five times now."
  | 6 => "Six times now."
  | 7 => "Seven times now."
  | 8 => "Eight times now."
  | 9 => "Nine times now."
  | 10 => "Ten times now."
  | 11 => "Eleven times now."
  | 12 => "There have been twelve lies so far."
  | _ => "Error: Invalid candidate."

-- Total number of candidates
def total_candidates : Nat := 12

-- Assertion that at least one candidate has told the truth
axiom truthful_candidate_exists (k : Nat) : (k ≥ 1) ∧ (k ≤ 12) ∧ ⟦candidate k⟧ = k

-- Proof problem statement
theorem number_of_liars_is_eleven : ∃ k : Nat, (k = 1) ∧ (∀ m : Nat, (1 ≤ m ∧ m ≤ total_candidates ∧ m ≠ k → ¬⟦candidate m⟧ = m) ∧ (total_candidates - 1 = 11)) :=
by
  sorry

end number_of_liars_is_eleven_l679_679547


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679476

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679476


namespace second_company_managers_percent_l679_679226

/-- A company's workforce consists of 10 percent managers and 90 percent software engineers.
    Another company's workforce consists of some percent managers, 10 percent software engineers, 
    and 60 percent support staff. The two companies merge, and the resulting company's 
    workforce consists of 25 percent managers. If 25 percent of the workforce originated from the 
    first company, what percent of the second company's workforce were managers? -/
theorem second_company_managers_percent
  (F S : ℝ)
  (h1 : 0.10 * F + m * S = 0.25 * (F + S))
  (h2 : F = 0.25 * (F + S)) :
  m = 0.225 :=
sorry

end second_company_managers_percent_l679_679226


namespace sum_S9_l679_679067

variable (a : ℕ → ℤ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given condition for the sum of specific terms
def condition_given (a : ℕ → ℤ) : Prop :=
  a 2 + a 5 + a 8 = 12

-- Sum of the first 9 terms
def sum_of_first_nine_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8

-- Problem statement: Prove that given the arithmetic sequence and the condition,
-- the sum of the first 9 terms is 36
theorem sum_S9 :
  arithmetic_sequence a → condition_given a → sum_of_first_nine_terms a = 36 :=
by
  intros
  sorry

end sum_S9_l679_679067


namespace continuity_at_2_l679_679460

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end continuity_at_2_l679_679460


namespace countIncorrectRelations_l679_679719

-- Defining the conditions from the problem
def cond1 := 1 ⊆ ({1, 2, 3} : Set ℕ)
def cond2 := {1} ∈ ({1, 2, 3} : Set (Set ℕ))
def cond3 := {1, 2, 3} ⊆ ({1, 2, 3} : Set ℕ)
def cond4 := ∅ ⊆ ({1} : Set (Set ℕ))

-- Statement to prove the number of incorrect relations is 2
theorem countIncorrectRelations : (cond1 → False) ∧ (cond2 → False) → 2 = 2 := sorry

end countIncorrectRelations_l679_679719


namespace prob1_prob2_prob3_prob4_prob5_l679_679219

/-- Define the expressions we want to prove. -/
noncomputable def expr1 := (1 + 2 - 3 + 4 + 5 - 6 + ⋯ + 220 + 221 - 222 : ℝ)
noncomputable def expr2 := ((22 + 23 + 24 + 25 + 26) * (25 + 26 + 27) - (22 + 23 + 24 + 25 + 26 + 27) * (25 + 26) : ℝ)
noncomputable def expr3 := (∑ i in Finset.filter (λ x, x % 2 = 1) (Finset.range 20), (↑i+1)^2 : ℝ)
noncomputable def expr4 := (∑ i in Finset.range 2021, (Finset.range (i + 1)).sum (λ j, ↑j / ↑(i + 1)) : ℝ)
noncomputable def expr5 := ((∑ i in Finset.range 1000, if i % 2 = 0 then (-1) ^ i * (1 / (↑i + 1)) else (1 / (↑i + 1))) / 
                             (∑ i in Finset.range 500, (1 / (↑(1002 + 2 * i) + 1))) : ℝ)

/-- Prove the expressions are equal to their calculated values. -/
theorem prob1 : expr1 = 8103 := by
  sorry

theorem prob2 : expr2 = 1863 := by
  sorry

theorem prob3 : expr3 = 1330 := by
  sorry

theorem prob4 : expr4 = 1020605 := by
  sorry

theorem prob5 : expr5 = 2 := by
  sorry

end prob1_prob2_prob3_prob4_prob5_l679_679219


namespace sequence_sum_18_l679_679155

noncomputable def sequence (n : ℕ) : ℤ :=
  if n = 1 then -7
  else if n = 2 then 5
  else sequence (n - 2) + 2

def sum_first_n_terms (f : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).sum.map f

theorem sequence_sum_18 :
  let a_n := sequence in
  let s18 := sum_first_n_terms a_n 18 in
  s18 = 126 := by
    sorry

end sequence_sum_18_l679_679155


namespace units_digit_5689_pow_439_l679_679320

-- Our goal is to prove that the units digit of 5689^439 is 9.
theorem units_digit_5689_pow_439 : (5689 ^ 439) % 10 = 9 := 
by {
  -- We state our conditions
  have base_units_digit : 5689 % 10 = 9 := by norm_num,
  have exponent_mod : 439 % 2 = 1 := by norm_num,

  -- Combine the results to find the units digit
  rw [←pow_mod, base_units_digit, nat.pow_two_cycles_eq_pow, exponent_mod],
  exact nat.pow_two_cycles_mod_self_eq 9 2 1,
}
open_locale nat

end units_digit_5689_pow_439_l679_679320


namespace largest_k_for_sum_of_integers_l679_679290

theorem largest_k_for_sum_of_integers (k : ℕ) (n : ℕ) (h1 : 3^12 = k * n + k * (k + 1) / 2) 
  (h2 : k ∣ 2 * 3^12) (h3 : k < 1031) : k ≤ 486 :=
by 
  sorry -- The proof is skipped here, only the statement is required 

end largest_k_for_sum_of_integers_l679_679290


namespace powers_of_three_two_digit_count_l679_679026

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679026


namespace correct_calculation_l679_679560

theorem correct_calculation (a : ℝ) : (-a)^6 / a^3 = a^3 :=
by {
  -- Using the properties of exponents and division,
  -- we can simplify the left-hand side as follows:
  have h1 : (-a)^6 = a^6 := by sorry,
  have h2 : a^6 / a^3 = a^(6-3) := by sorry,
  rw [h1, h2],
  rw [sub_eq_add_neg, add_comm, neg_add_self],
  exact rfl,
}

end correct_calculation_l679_679560


namespace area_ABDF_is_560_l679_679894

-- Defining the points and their properties
variable (A C B E D F : Point)
variable (length_AC width_AE : ℝ)
variable (midpoint_B: B = midpoint A C)
variable (third_way_F: ∃ (k : ℝ), k = 1/3 ∧ F = A + k • (E - A))

-- Defining the area of the quadrilateral ABDF
def area_quadrilateral_ABDF (A C B E D F : Point) : ℝ := sorry

-- Proposition that the area of quadrilateral ABDF is 560
theorem area_ABDF_is_560 
  (length_AC := 40)
  (width_AE := 24)
  (midpoint_B : B = midpoint A C)
  (third_way_F : ∃ (k : ℝ), k = 1/3 ∧ F = A + k • (E - A))
  : area_quadrilateral_ABDF A C B E D F = 560 := 
sorry

end area_ABDF_is_560_l679_679894


namespace degree_sum_polynomials_l679_679128

noncomputable def f : Polynomial ℂ := Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 2 * Polynomial.X ^ 2 + Polynomial.C 3 * Polynomial.X + Polynomial.C 4
noncomputable def g : Polynomial ℂ := Polynomial.C 5 * Polynomial.X ^ 2 + Polynomial.C 6 * Polynomial.X + Polynomial.C 7

theorem degree_sum_polynomials (hf : degree f = 3) (hg : degree g = 2) : degree (f + g) = 3 :=
by
  sorry

end degree_sum_polynomials_l679_679128


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679478

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679478


namespace total_turtles_taken_l679_679263

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l679_679263


namespace cost_of_socks_l679_679690

/-- Given initial amount of $100 and cost of shirt is $24,
    find out the cost of socks if the remaining amount is $65. --/
theorem cost_of_socks
  (initial_amount : ℕ)
  (cost_of_shirt : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : cost_of_shirt = 24)
  (h3 : remaining_amount = 65) : 
  (initial_amount - cost_of_shirt - remaining_amount) = 11 :=
by
  sorry

end cost_of_socks_l679_679690


namespace hoseok_value_l679_679875

theorem hoseok_value (x : ℕ) (h : x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end hoseok_value_l679_679875


namespace minimum_value_l679_679315

def min_value_func : ℝ :=
  let y := λ x : ℝ, 2 * Real.sin x
  Inf {y x | (Real.pi / 3) ≤ x ∧ x ≤ (5 * Real.pi / 6)}

theorem minimum_value : min_value_func = 1 := 
by
  sorry

end minimum_value_l679_679315


namespace min_value_expression_l679_679313

noncomputable def min_expression : ℝ := 3 * Real.cos θ + (2 / Real.sin θ) + 2 * Real.sqrt 2 * Real.tan θ

theorem min_value_expression : (∃ θ, 0 < θ ∧ θ < (π / 2) ∧ min_expression θ = (11 * Real.sqrt 2) / 2) :=
by
  sorry

end min_value_expression_l679_679313


namespace base_2_representation_of_125_l679_679924

theorem base_2_representation_of_125 : 
  ∃ (L : List ℕ), L = [1, 1, 1, 1, 1, 0, 1] ∧ 
    125 = (L.reverse.zipWith (λ b p, b * (2^p)) (List.range L.length)).sum := 
by
  sorry

end base_2_representation_of_125_l679_679924


namespace sequence_product_l679_679934

theorem sequence_product : (2 : ℝ) * (∏ n in Finset.range (2009 - 2 + 1), (n + 3) / (n + 2)) = 1005 := by sorry

end sequence_product_l679_679934


namespace blue_balls_prob_l679_679283

def prob_same_color (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

theorem blue_balls_prob {n : ℕ} (h : prob_same_color n = 1 / 2) : n = 1 ∨ n = 9 :=
by
  sorry

end blue_balls_prob_l679_679283


namespace projection_correct_l679_679644

def vec := ℝ × ℝ × ℝ
def proj_matrix : vec → vec := λ v, 
  let (vx, vy, vz) := v in 
  (1 / 11 * vx + 3 / 11 * vy - 1 / 11 * vz,
   3 / 11 * vx + 9 / 11 * vy - 3 / 11 * vz,
   -1 / 11 * vx - 3 / 11 * vy + 1 / 11 * vz)

def projection_condition (v w : vec) : Prop :=
  let (vx, vy, vz) := v in 
  let (wx, wy, wz) := w in 
  wx = (1 / 11 * vx + 3 / 11 * vy - 1 / 11 * vz) ∧
  wy = (3 / 11 * vx + 9 / 11 * vy - 3 / 11 * vz) ∧
  wz = (-1 / 11 * vx - 3 / 11 * vy + 1 / 11 * vz)

theorem projection_correct (v : vec) : 
  let q := proj_matrix v in projection_condition v q :=
by sorry

end projection_correct_l679_679644


namespace time_to_cross_first_platform_l679_679249

variable (length_first_platform : ℝ)
variable (length_second_platform : ℝ)
variable (time_to_cross_second_platform : ℝ)
variable (length_of_train : ℝ)

theorem time_to_cross_first_platform :
  length_first_platform = 160 →
  length_second_platform = 250 →
  time_to_cross_second_platform = 20 →
  length_of_train = 110 →
  (270 / (360 / 20) = 15) := 
by
  intro h1 h2 h3 h4
  sorry

end time_to_cross_first_platform_l679_679249


namespace Lisa_types_correctly_l679_679515

-- Given conditions
def Rudy_wpm : ℕ := 64
def Joyce_wpm : ℕ := 76
def Gladys_wpm : ℕ := 91
def Mike_wpm : ℕ := 89
def avg_wpm : ℕ := 80
def num_employees : ℕ := 5

-- Define the hypothesis about Lisa's typing speaking
def Lisa_wpm : ℕ := (num_employees * avg_wpm) - Rudy_wpm - Joyce_wpm - Gladys_wpm - Mike_wpm

-- The statement to prove
theorem Lisa_types_correctly :
  Lisa_wpm = 140 := by
  sorry

end Lisa_types_correctly_l679_679515


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679493

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679493


namespace find_side_length_of_base_of_pyramid_l679_679453

-- Define the problem's parameters and conditions
def angle_of_inclination : ℝ := Real.arctan (3 / 4)

structure PyramidBase :=
(side_length : ℝ)
(surface_area : ℝ)

def base_side_length (A : PyramidBase) : Prop :=
  A.surface_area = 53 * Real.sqrt 3 ∧ angle_of_inclination = Real.arctan (3 / 4)

-- The main theorem to be proven
theorem find_side_length_of_base_of_pyramid (A : PyramidBase) :
  base_side_length A → A.side_length = 6 :=
by
  intros h
  sorry

end find_side_length_of_base_of_pyramid_l679_679453


namespace trader_profit_percentage_l679_679207

-- Define the conditions.
variables (indicated_weight actual_weight_given claimed_weight : ℝ)
variable (profit_percentage : ℝ)

-- Given conditions
def conditions :=
  indicated_weight = 1000 ∧
  actual_weight_given = claimed_weight / 1.5 ∧
  claimed_weight = indicated_weight ∧
  profit_percentage = (claimed_weight - actual_weight_given) / actual_weight_given * 100

-- Prove that the profit percentage is 50%
theorem trader_profit_percentage : conditions indicated_weight actual_weight_given claimed_weight profit_percentage → profit_percentage = 50 :=
by
  sorry

end trader_profit_percentage_l679_679207


namespace sqrt_operations_correctness_l679_679562

open Real

theorem sqrt_operations_correctness :
  (sqrt 2 + sqrt 3 ≠ sqrt 5) ∧
  (sqrt (2/3) * sqrt 6 = 2) ∧
  (sqrt 9 = 3) ∧
  (sqrt ((-6) ^ 2) = 6) :=
by
  sorry

end sqrt_operations_correctness_l679_679562


namespace powers_of_three_two_digit_count_l679_679028

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679028


namespace trapezoid_bases_difference_l679_679521

structure Trapezoid :=
  (A B C D : Point)
  (AB CD : Line)
  (BC AD : Line)
  (angleA angleB angleC angleD : ℝ)
  (AB_len: ℝ)
  (BC_len: ℝ)
  (CD_len: ℝ)
  (AD_len: ℝ)
  (is_right_trapezoid: (angleA = 90 ∧ angleB = 90) ∧ (angleD = 135))

def shorter_leg (T : Trapezoid) := T.BC_len

def bases_difference (T: Trapezoid) : ℝ :=
  abs (T.AD_len - T.BC_len)

theorem trapezoid_bases_difference (T : Trapezoid)
  (h1 : T.is_right_trapezoid)
  (h2 : shorter_leg T = 18) :
  bases_difference T = 18 :=
by
  sorry

end trapezoid_bases_difference_l679_679521


namespace trigonometric_identity_l679_679161

open Real

theorem trigonometric_identity : 4 * sin (π / 12) * cos (π / 12) = 1 :=
by
  -- Using the double angle identity: sin(2x) = 2 * sin(x) * cos(x)
  have h1 : sin (2 * (π / 12)) = 2 * sin (π / 12) * cos (π / 12),
    from sin_double_angle (π / 12)
  -- Simplify 2 * (π / 12) into π / 6
  have h2 : 2 * (π / 12) = π / 6,
    by norm_num
  -- Use the double angle formula in the given expression
  rw [←h2, h1]
  -- Simplify sin (π / 6)
  have h3 : sin (π / 6) = 1 / 2
    from sin_pi_div_six
  linarith

end trigonometric_identity_l679_679161


namespace find_segment_B1A1_l679_679857

-- Definitions of points and properties
def Point := ℝ × ℝ
def Circle (r : ℝ) := {p : Point | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}

variables (A₁ A₂ A₃ A₄ A₅ A₆ B₁ B₂ B₃ B₄ B₅ B₆ : Point)
variable (O : Point) -- center of the circle
variable (r : ℝ) -- radius of the circle

-- Conditions
axiom point_circle (p : Point) : Circle r p
axiom equal_division : (dist O A₁ = r) ∧ (dist A₁ A₂ = dist A₂ A₃) ∧ (dist A₂ A₃ = dist A₃ A₄) ∧ 
                        (dist A₃ A₄ = dist A₄ A₅) ∧ (dist A₄ A₅ = dist A₅ A₆) ∧ (dist A₅ A₆ = dist A₆ A₁)
axiom perpendicular_drops : 
  ∃ (l₁ l₂ l₃ l₄ l₅ l₆ : Point × Point),
    line_through A₁ A₂ = l₁ ∧ line_through A₂ A₃ = l₂ ∧ line_through A₃ A₄ = l₃ ∧
    line_through A₄ A₅ = l₄ ∧ line_through A₅ A₆ = l₅ ∧ line_through A₆ A₁ = l₆ ∧
    perpendicular B₁ l₆ B₂ ∧ perpendicular B₂ l₅ B₃ ∧ perpendicular B₃ l₄ B₄ ∧
    perpendicular B₄ l₃ B₅ ∧ perpendicular B₅ l₂ B₆ ∧ perpendicular B₆ l₁ B₁

-- Proof goal
theorem find_segment_B1A1 : dist B₁ A₁ = 2 :=
sorry

end find_segment_B1A1_l679_679857


namespace quadratic_equation_condition_l679_679142

theorem quadratic_equation_condition (m : ℝ) : 
  (m ≠ 1) ↔ is_quadratic (λ x, m * x^2 - 3 * x - (x^2 - m * x + 2)) :=
by
  sorry

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c)
  -- Function is quadratic if it can be written in the form ax^2 + bx + c with a ≠ 0,
  -- which represents the generic form of a quadratic equation.

end quadratic_equation_condition_l679_679142


namespace sum_of_squares_mod_13_l679_679940

theorem sum_of_squares_mod_13 : 
  (∑ i in Finset.range 13, i^2) % 13 = 0 :=
by
  sorry

end sum_of_squares_mod_13_l679_679940


namespace no_integer_solutions_l679_679435

theorem no_integer_solutions (a : ℕ) (h : a % 4 = 3) : ¬∃ (x y : ℤ), x^2 + y^2 = a := by
  sorry

end no_integer_solutions_l679_679435


namespace determine_x_l679_679808

noncomputable def solve_x (x : ℝ) : Prop := 
  let area_triangle_abe := (1 / 2) * x * 1
  let area_rectangle_efgh := x * (2 - x)
  area_triangle_abe = area_rectangle_efgh

theorem determine_x : ∃x : ℝ, solve_x x ∧ x = 3/2 := by
  have h : solve_x (3 / 2) := by
    sorry -- the proof steps go here
  use (3 / 2)
  exact ⟨h, rfl⟩

end determine_x_l679_679808


namespace prove_angle_C_prove_max_area_l679_679049

open Real

variables {A B C : ℝ} {a b c : ℝ} (abc_is_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (R : ℝ) (circumradius_is_sqrt2 : R = sqrt 2)
variables (H : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
variables (law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C)

-- Part 1: Prove that angle C = π / 3
theorem prove_angle_C : C = π / 3 :=
sorry

-- Part 2: Prove that the maximum value of the area S of triangle ABC is (3 * sqrt 3) / 2
theorem prove_max_area : (1 / 2) * a * b * sin C ≤ (3 * sqrt 3) / 2 :=
sorry

end prove_angle_C_prove_max_area_l679_679049


namespace inequality_sqrt_l679_679747

open Real

theorem inequality_sqrt (x y : ℝ) :
  (sqrt (x^2 - 2*x*y) > sqrt (1 - y^2)) ↔ 
    ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by
  sorry

end inequality_sqrt_l679_679747


namespace probability_between_lines_l679_679047

theorem probability_between_lines (x y : ℝ) :
  let l := -3 * x + 9
      m := -6 * x + 9
  in (0 < x ∧ 0 < y ∧ y < l) →
     let area_l := (1/2) * 3 * 9
         area_m := (1/2) * 1.5 * 9
         area_between := area_l - area_m
     in (area_between / area_l = 1/2) :=
by
  sorry

end probability_between_lines_l679_679047


namespace T_in_form_a_b_sqrt_c_l679_679826

noncomputable def sum_T : ℝ :=
  (∑ n in finset.range 10000, 1 / (real.sqrt (n + 2 * real.sqrt (n^2 - 4))))

theorem T_in_form_a_b_sqrt_c
  (a b c : ℕ) 
  (hT_form : sum_T = a + b * real.sqrt c)
  (h_a_pos : 0 < a) 
  (h_b_pos : 0 < b)
  (h_c_pos : 0 < c)
  (h_c_square_free : ∀ p : ℕ, nat.prime p → ¬ p ^ 2 ∣ c) :
  a + b + c = 142 :=
begin
  sorry
end

end T_in_form_a_b_sqrt_c_l679_679826


namespace bread_cost_l679_679176

theorem bread_cost (H C B : ℕ) (h₁ : H = 150) (h₂ : C = 200) (h₃ : H + B = C) : B = 50 :=
by
  sorry

end bread_cost_l679_679176


namespace ABD_collinear_l679_679741

noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1) * k = p3.1 - p1.1 ∧ (p2.2 - p1.2) * k = p3.2 - p1.2

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables {a b : ℝ × ℝ}
variables {A B C D : ℝ × ℝ}

axiom a_ne_zero : a ≠ (0, 0)
axiom b_ne_zero : b ≠ (0, 0)
axiom a_b_not_collinear : ∀ k : ℝ, a ≠ k • b
axiom AB_def : B = (A.1 + a.1 + b.1, A.2 + a.2 + b.2)
axiom BC_def : C = (B.1 + a.1 + 10 * b.1, B.2 + a.2 + 10 * b.2)
axiom CD_def : D = (C.1 + 3 * (a.1 - 2 * b.1), C.2 + 3 * (a.2 - 2 * b.2))

theorem ABD_collinear : collinear A B D :=
by
  sorry

end ABD_collinear_l679_679741


namespace problem_statement_l679_679243

-- Define the sequence using the given conditions
def seq_b : ℕ → ℚ
| 1       := 2
| 2       := 5/11
| (n + 1) := if n ≥ 2 then (seq_b (n - 1) * seq_b n) / (3 * seq_b (n - 1) - 2 * seq_b n) else 0

-- The problem to prove
theorem problem_statement : ∃ p q : ℕ, p + q = 4 ∧ (q ≠ 0) ∧ (nat.gcd p q = 1) ∧ (seq_b 10 = p / q) :=
sorry

end problem_statement_l679_679243


namespace hh_value_l679_679764

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem hh_value :
  h(h(3)) = 3568 :=
by
  sorry

end hh_value_l679_679764


namespace tank_water_after_rain_final_l679_679619

theorem tank_water_after_rain_final (initial_water evaporated drained rain_rate rain_time : ℕ)
  (initial_water_eq : initial_water = 6000)
  (evaporated_eq : evaporated = 2000)
  (drained_eq : drained = 3500)
  (rain_rate_eq : rain_rate = 350)
  (rain_time_eq : rain_time = 30) :
  let water_after_evaporation := initial_water - evaporated
  let water_after_drainage := water_after_evaporation - drained 
  let rain_addition := (rain_time / 10) * rain_rate
  let final_water := water_after_drainage + rain_addition
  final_water = 1550 :=
by
  sorry

end tank_water_after_rain_final_l679_679619


namespace velocity_zero_at_t_eq_4_or_8_l679_679236

open Real

-- Define the distance function s
def s (t : ℝ) : ℝ := (1/3) * t^3 - 6 * t^2 + 32 * t

-- State the theorem for the instants when the velocity is zero
theorem velocity_zero_at_t_eq_4_or_8 : ∀ t : ℝ, deriv s t = 0 ↔ t = 4 ∨ t = 8 := by
  sorry

end velocity_zero_at_t_eq_4_or_8_l679_679236


namespace number_of_elements_l679_679824

open Set

-- Definitions based on the problem statement
def S : ℕ → Set (Set (Set ℕ))
| 0     := ∅
| (n+1) := { S 0, S 1, S 2, ..., S n }

theorem number_of_elements :
  (S 10 ∩ S 20 ∪ S 30 ∩ S 40).card = 30 :=
sorry

end number_of_elements_l679_679824


namespace trapezoid_vertices_parallelogram_condition_collinear_condition_l679_679069

variables {Point : Type} [MetricSpace Point]

-- Assumptions about the parallelogram ABCD and midpoints
variables (A B C D A1 B1 C1 D1 E F M M1 M2 M3 G : Point)
variables (h_parallel_1 : Parallelogram A B C D)
variables (h_mid_A1 : Midpoint A B A1)
variables (h_mid_B1 : Midpoint B C B1)
variables (h_mid_C1 : Midpoint C D C1)
variables (h_mid_D1 : Midpoint D A D1)

-- Assumptions about reflections of M
variables (h_refl_M1 : Reflect M A1 M1)
variables (h_refl_M2 : Reflect M B1 M2)
variables (h_refl_M3 : Reflect M C1 M3)

-- Assumptions about the symmetric points E and F on line BC
variables (h_symmetric_EF : SymmetricPoints B1 E F)
variables (h_EF_line : Collinear B C E)
variables (h_EF_line2 : Collinear B C F)

-- Define G to be the intersection of the diagonals of the trapezoid formed by E, F, M1, and M3
def G_is_intersection := Intersection ((diagonal_1 E F M1 M3)) (diagonal_2 E F M1 M3) G

-- Prove that E, F, M1, and M3 form a trapezoid
theorem trapezoid_vertices :
  Trapezoid E F M1 M3 :=
sorry

-- Condition for E, F, M1, and M3 to form a parallelogram
theorem parallelogram_condition :
  E F = M1 M3 ↔ Parallelogram E F M1 M3 :=
sorry

-- Condition for D1, M2, and G to be collinear
theorem collinear_condition :
  Collinear D1 M2 G ↔ (Collinear M B1 D1 ∨ (E F = M1 M3 ∧ Intersection(diagonal_1 E F M1 M3 ) (diagonal_2 E F M1 M3) G)) :=
sorry

end trapezoid_vertices_parallelogram_condition_collinear_condition_l679_679069


namespace general_term_l679_679353

noncomputable def a : ℕ → ℤ
| 1     := -1
| (n+1) := 2 * a n + 2

theorem general_term (n : ℕ) : a n = 2^(n-1) - 2 := by
  sorry

end general_term_l679_679353


namespace prime_divisors_consecutive_l679_679743

theorem prime_divisors_consecutive (p q : ℕ) [hp : Prime p] [hq : Prime q] (h1 : p < q) (h2 : q < 2 * p) :
  ∃ (n m : ℕ), Nat.gcd n m = 1 ∧ abs (n - m) = 1 ∧ (∀ p' : ℕ, Prime p' → p' ∣ n → p' = p) ∧ (∀ q' : ℕ, Prime q' → q' ∣ m → q' = q) := 
  sorry

end prime_divisors_consecutive_l679_679743


namespace circle_radius_l679_679071

theorem circle_radius (θ : ℝ) (ρ : ℝ) : (ρ = 2 * Real.sin θ) → ∃ r, r = 1 := by
  intro h
  use 1
  sorry

end circle_radius_l679_679071


namespace exponential_function_condition_l679_679144

theorem exponential_function_condition (a : ℝ) (x : ℝ) 
  (h1 : a^2 - 5 * a + 5 = 1) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) : 
  a = 4 := 
sorry

end exponential_function_condition_l679_679144


namespace cyclic_quadrilateral_APBR_l679_679786

-- Introduce the necessary points and segments.
variables (A B L M K Q R : Type)
variables (circle1 circle2 : Type)
variables (h1 : Segment A B ⊥ circle1)
variables (h2 : L ∈ circle2 ∧ M ∈ circle2)
variables (h3 : ∃ (intR : circle2 ∩ Triangle K M Q), intR = R)

-- State the proposition to prove that APBR is cyclic.
theorem cyclic_quadrilateral_APBR (A B L M K Q R : Type) 
    (circle1 circle2 : Type)
    (h1 : Segment A B ⊥ circle1)
    (h2 : L ∈ circle2 ∧ M ∈ circle2)
    (h3 : ∃ (intR : circle2 ∩ Triangle K M Q), intR = R) 
    : cyclic_quadrilateral A P B R := 
sorry  -- The detailed proof is omitted.

end cyclic_quadrilateral_APBR_l679_679786


namespace arrangement_count_l679_679165

-- Graduate A and Graduate B are represented as type members
inductive Graduate | A | B | C | D

-- Definition of schools
inductive School | School1 | School2

-- Condition: Each graduate interns at one school
def intern_at (g : Graduate) : School

-- Condition: Each school must have at least one intern
def school_has_at_least_one_intern (s : School) : Prop :=
  ∃ g : Graduate, intern_at g = s

-- Condition: Graduates A and B cannot intern at the same school
def A_and_B_not_same_school : Prop :=
  intern_at Graduate.A ≠ intern_at Graduate.B 

-- The number of possible arrangements of graduates
def number_of_arrangements := 8

-- The main theorem that we need to prove
theorem arrangement_count : 
  (∀ g : Graduate, intern_at g ∈ {School.School1, School.School2}) ∧ 
  (school_has_at_least_one_intern School.School1) ∧
  (school_has_at_least_one_intern School.School2) ∧ 
  A_and_B_not_same_school → 
  ∃ n : ℕ, n = number_of_arrangements ∧ n = 8 :=
begin
  sorry
end

end arrangement_count_l679_679165


namespace circumcenter_EFG_on_Γ1_l679_679284

-- Defining the problem and conditions 
variable {A B C E F G : Type}
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E] [Inhabited F] [Inhabited G]
variable (AB AC BC : ℝ)
variable (Γ1 Γ2_BAC Γ3_CAB : Type)
variable [Inhabited Γ1] [Inhabited Γ2_BAC] [Inhabited Γ3_CAB]
variable [HAB: AB < AC] [HAC: AC < BC]

-- Definitions of circles and intersection points
variable (Γ2_BAC : B × AC) -- Circle 2 with center B and radius AC
variable (Γ3_CAB : C × AB) -- Circle 3 with center C and radius AB
variable (common_point_E : Γ2_BAC ∩ Γ3_CAB = {E}) -- Point E is the unique intersection
variable (common_point_F : Γ1 ∩ Γ3_CAB = {F}) -- Point F is the unique intersection
variable (common_point_G : Γ1 ∩ Γ2_BAC = {G}) -- Point G is the unique intersection
variable (same_semiplane : E ∈ ℝ ∧ F ∈ ℝ ∧ G ∈ ℝ ∧ ∃ t : A → ℝ, t(E) ≠ t(A) ∧ t(F) ≠ t(A) ∧ t(G) ≠ t(A))

-- Proof statement
theorem circumcenter_EFG_on_Γ1 :
  (∃ O : Type, IsCircumcenter O E F G ∧ O ∈ Γ1) :=
sorry

end circumcenter_EFG_on_Γ1_l679_679284


namespace david_scored_32_points_l679_679287

theorem david_scored_32_points :
  ∀ (initial_lead brenda_play final_lead : ℕ), 
    initial_lead = 22 → 
    brenda_play = 15 → 
    final_lead = 5 → 
    let brenda_lead_after_play := initial_lead + brenda_play in
    let david_score := brenda_lead_after_play - final_lead in
    david_score = 32 :=
begin
  intros initial_lead brenda_play final_lead h1 h2 h3,
  rw [h1, h2, h3],
  let brenda_lead_after_play := 22 + 15,
  let david_score := brenda_lead_after_play - 5,
  have h4 : brenda_lead_after_play = 37 := by norm_num,
  rw h4 at *,
  have h5 : david_score = 32 := by norm_num,
  exact h5,
end

end david_scored_32_points_l679_679287


namespace original_price_l679_679970

variable (x : ℝ)

-- Condition 1: Selling at 60% of the original price results in a 20 yuan loss
def condition1 : Prop := 0.6 * x + 20 = x * 0.8 - 15

-- The goal is to prove that the original price is 175 yuan under the given conditions
theorem original_price (h : condition1 x) : x = 175 :=
sorry

end original_price_l679_679970


namespace two_digit_numbers_of_form_3_pow_n_l679_679021

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679021


namespace magnitude_difference_l679_679694

variable {α : Type*} [InnerProductSpace ℝ α]

theorem magnitude_difference (a b : α) (h₁ : ‖a‖ = 3) (h₂ : ‖b‖ = 2) (h₃ : real.angle (inner_product_space.angle a b) = real.angle.one_div 3) :
  ‖a - b‖ = real.sqrt 7 :=
sorry

end magnitude_difference_l679_679694


namespace nonnegative_integer_solution_unique_l679_679673

theorem nonnegative_integer_solution_unique :
  ∃ (x y z : ℕ), 5 * x + 7 * y + 5 * z = 37 ∧ 6 * x - y - 10 * z = 3 ∧
    (x = 4 ∧ y = 1 ∧ z = 2) :=
by
  use 4
  use 1
  use 2
  split
  rfl
  split
  rfl
  split
  rfl
  sorry

end nonnegative_integer_solution_unique_l679_679673


namespace parallelogram_area_l679_679188

theorem parallelogram_area (b h : ℕ) (hb : b = 20) (hh : h = 4) : b * h = 80 := by
  sorry

end parallelogram_area_l679_679188


namespace pow_sum_mod_eight_l679_679190

theorem pow_sum_mod_eight : 
  let S := (Finset.range 2010).sum (fun n => 3^n)
  S % 8 = 4 := by
{
  let S := (Finset.range 2010).sum (fun n => 3^n)
  show S % 8 = 4,
  sorry
}

end pow_sum_mod_eight_l679_679190


namespace marcia_average_cost_l679_679448

theorem marcia_average_cost :
  let price_apples := 2
  let price_bananas := 1
  let price_oranges := 3
  let count_apples := 12
  let count_bananas := 4
  let count_oranges := 4
  let offer_apples_free := count_apples / 10 * 2
  let offer_oranges_free := count_oranges / 3
  let total_apples := count_apples + offer_apples_free
  let total_oranges := count_oranges + offer_oranges_free
  let total_fruits := total_apples + count_bananas + count_oranges
  let cost_apples := price_apples * (count_apples - offer_apples_free)
  let cost_bananas := price_bananas * count_bananas
  let cost_oranges := price_oranges * (count_oranges - offer_oranges_free)
  let total_cost := cost_apples + cost_bananas + cost_oranges
  let average_cost := total_cost / total_fruits
  average_cost = 1.85 :=
  sorry

end marcia_average_cost_l679_679448


namespace jenna_height_cm_l679_679078

noncomputable def inchToCm (inches : ℝ) : ℝ := inches * 2.54

theorem jenna_height_cm : 
  let heightInInches := 72
  let heightInCm := inchToCm heightInInches
  let roundedHeightInCm := Real.round heightInCm
  roundedHeightInCm = 183 :=
by
  sorry

end jenna_height_cm_l679_679078


namespace probability_S1_greater_2S2_l679_679048

def point_on_AB (A B P : Point) : Prop := lies_on (line_through A B) P

def area (h AP BP : ℝ) : ℝ :=
  let S1 := 1/2 * AP * h
  let S2 := 1/2 * BP * h
  S1 > 2 * S2

theorem probability_S1_greater_2S2 (h AP BP : ℝ) (A B P : Point)
  (h_Geometry : point_on_AB A B P) :
  ∃ p : ℝ, p = 1/3 :=
by
  sorry

end probability_S1_greater_2S2_l679_679048


namespace harvest_rate_l679_679369

def days := 3
def total_sacks := 24
def sacks_per_day := total_sacks / days

theorem harvest_rate :
  sacks_per_day = 8 :=
by
  sorry

end harvest_rate_l679_679369


namespace returns_to_start_furthest_distance_from_origin_total_sesame_seeds_l679_679608

namespace InsectCrawl

def distances : List Int := [5, -3, 10, -8, -6, 12, -10]

theorem returns_to_start : List.sum distances = 0 :=
by
  sorry

theorem furthest_distance_from_origin : List.foldl (λ (max, current) d, max (max, current + d, Native.irrelevant) + max) (0, 0) distances = 14 :=
by
  sorry

theorem total_sesame_seeds : List.foldl (λ acc d, acc + abs d) 0 distances = 54 :=
by
  sorry

end InsectCrawl

end returns_to_start_furthest_distance_from_origin_total_sesame_seeds_l679_679608


namespace part_a_part_b_l679_679135
open Set

noncomputable def bus_trip : Prop :=
  let stops := Finset.range 14
  ∃ (A B : Finset (Finset ℕ)), 
    (A.card = 4 ∧ B.card = 4 ∧ A ∩ B = ∅ ∧ 
    ∀ (a ∈ A) (b ∈ B), 
      ∀ (s ∈ stops), 
        if a < s ∧ s < b then False)

theorem part_a (h : ∀ (A B : Finset (Finset ℕ)), 
  ∃ (M : Finset (Finset (Finset ℕ))),
    M.card ≤ 25 → 
    bus_trip) : 
    ∃ (A B : Finset (Finset ℕ)),
      A.card = 4 ∧ B.card = 4 ∧ A ∩ B = ∅ :=
by sorry

theorem part_b : 
  ∀ (A B : Finset (Finset ℕ)), 
    ¬ ∃ (C D : Finset (Finset ℕ)), 
      (C.card = 5 ∧ D.card = 5 ∧ C ∩ D = ∅) :=
by sorry

end part_a_part_b_l679_679135


namespace sequence_condition_l679_679918

def u_sequence (u1 u2 : ℕ) : ℕ → ℕ
| 0     := u1
| 1     := u2
| (n+2) := min (abs (u_sequence n - u_sequence (n+1)))
/-- The condition that u1 and u2 are within the given range -/
def initial_condition (u1 u2 : ℕ) : Prop :=
  1 ≤ u1 ∧ u1 ≤ 10000 ∧ 1 ≤ u2 ∧ u2 ≤ 10000

theorem sequence_condition (u1 u2 : ℕ) (h : initial_condition u1 u2) : u_sequence u1 u2 20 = 0 :=
by sorry

end sequence_condition_l679_679918


namespace creative_sum_l679_679522

def letterValue (ch : Char) : Int :=
  let n := (ch.toNat - 'a'.toNat + 1) % 12
  if n = 0 then 2
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 2
  else if n = 5 then 1
  else if n = 6 then 0
  else if n = 7 then -1
  else if n = 8 then -2
  else if n = 9 then -3
  else if n = 10 then -2
  else if n = 11 then -1
  else 0 -- this should never happen

def wordValue (word : String) : Int :=
  word.foldl (λ acc ch => acc + letterValue ch) 0

theorem creative_sum : wordValue "creative" = -2 :=
  by
    sorry

end creative_sum_l679_679522


namespace largest_positive_root_between_4_and_5_l679_679651

theorem largest_positive_root_between_4_and_5 :
  ∃ (b_2 b_1 b_0 : ℝ), |b_2| ≤ 3 ∧ |b_1| ≤ 5 ∧ |b_0| ≤ 3 ∧ ∀ s : ℝ, 
  (x^3 + b_2 * x^2 + b_1 * x + b_0 = 0) → 4 < s < 5 := 
sorry

end largest_positive_root_between_4_and_5_l679_679651


namespace minimal_primes_ensuring_first_player_win_l679_679180

-- Define primes less than or equal to 100
def primes_le_100 : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

-- Define function to get the last digit of a number
def last_digit (n : Nat) : Nat := n % 10

-- Define function to get the first digit of a number
def first_digit (n : Nat) : Nat :=
  let rec first_digit_aux (m : Nat) :=
    if m < 10 then m else first_digit_aux (m / 10)
  first_digit_aux n

-- Define a condition that checks if a prime number follows the game rule
def follows_rule (a b : Nat) : Bool :=
  last_digit a = first_digit b

theorem minimal_primes_ensuring_first_player_win :
  ∃ (p1 p2 p3 : Nat),
  p1 ∈ primes_le_100 ∧
  p2 ∈ primes_le_100 ∧
  p3 ∈ primes_le_100 ∧
  follows_rule p1 p2 ∧
  follows_rule p2 p3 ∧
  p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
sorry

end minimal_primes_ensuring_first_player_win_l679_679180


namespace integral_result_l679_679643

theorem integral_result :
  ∫ x in 0..(Real.arccos (1 / Real.sqrt 6)), 
    (3 * (Real.tan x)^2 - 1) / 
    ((Real.tan x)^2 + 5) = Real.pi / Real.sqrt 5 - Real.arctan (Real.sqrt 5) :=
by
  sorry

end integral_result_l679_679643


namespace calculate_initial_books_l679_679424

-- Define the initial number of books
def initial_books := 800

-- Define the number of books sold each day
def books_sold_monday := 62
def books_sold_tuesday := 62
def books_sold_wednesday := 60
def books_sold_thursday := 48
def books_sold_friday := 40

-- Calculate total books sold
def total_books_sold := books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday + books_sold_friday

-- Define the percentage of books sold
def percentage_sold := 34 / 100

-- Lean statement to prove
theorem calculate_initial_books (h1 : total_books_sold = 272) (h2 : percentage_sold = 0.34) :
  (percentage_sold * initial_books) = total_books_sold → initial_books = 800 :=
by
  -- assume the given conditions and prove the theorem (proof omitted)
  sorry

end calculate_initial_books_l679_679424


namespace powers_of_three_two_digit_count_l679_679032

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679032


namespace continuous_at_x0_l679_679458

variable (x : ℝ) (ε : ℝ)

def f (x : ℝ) : ℝ := -3 * x^2 - 5

def x0 : ℝ := 2

theorem continuous_at_x0 : 
  (0 < ε) → 
  ∃ δ > 0, ∀ x, abs (x - x0) < δ → abs (f x - f x0) < ε := by
sory

end continuous_at_x0_l679_679458


namespace num_integers_condition_l679_679756

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem num_integers_condition : 
  {n : ℕ // n > 0 ∧ n < 2000 ∧ n = 9 * sum_of_digits n}.card = 4 := 
by sorry

end num_integers_condition_l679_679756


namespace number_of_Ds_l679_679536

theorem number_of_Ds 
  (normal_recess : ℕ)
  (extra_per_A : ℕ)
  (extra_per_B : ℕ)
  (extra_per_C : ℕ)
  (penalty_per_D : ℕ)
  (num_A : ℕ)
  (num_B : ℕ)
  (num_C : ℕ)
  (total_recess : ℕ)
  (D : ℕ) :
  normal_recess = 20 →
  extra_per_A = 2 →
  extra_per_B = 1 →
  extra_per_C = 0 →
  penalty_per_D = 1 →
  num_A = 10 →
  num_B = 12 →
  num_C = 14 →
  total_recess = 47 →
  D = 5 := 
begin
  intros,
  sorry
end

end number_of_Ds_l679_679536


namespace problem_l679_679840

noncomputable def a : ℚ := 1 / 15

def xi_distribution (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 5) : ℚ :=
  match hk with
  | ⟨_, _⟩ => a * k

theorem problem (ξ : ℚ) (hξ_dist : ∃ k : ℕ, 1 ≤ k ∧ k ≤ 5 ∧ ξ = k / 5) :
  (1/10 < ξ ∧ ξ < 1/2) → (∃ k : ℕ, k ∈ {1, 2} ∧ ξ = k / 5) :=
sorry

end problem_l679_679840


namespace angle_OP_AM_90_degrees_l679_679068

noncomputable def cube_vertex : Type := ℝ × ℝ × ℝ

structure Cube :=
(vertices : fin 8 → cube_vertex)
(midpoint : cube_vertex)
(center : cube_vertex)
(point_on_edge : cube_vertex)

def is_midpoint (M D D1 : cube_vertex) : Prop :=
  M = ((D.1 + D1.1) / 2, (D.2 + D1.2) / 2, (D.3 + D1.3) / 2)

def is_center (O A B C D : cube_vertex) : Prop :=
  O = ((A.1 + B.1 + C.1 + D.1) / 4, (A.2 + B.2 + C.2 + D.2) / 4, (A.3 + B.3 + C.3 + D.3) / 4)

def is_point_on_edge (P A1 B1 : cube_vertex) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A1.1 + t * (B1.1 - A1.1), A1.2 + t * (B1.2 - A1.2), A1.3 + t * (B1.3 - A1.3))

def perpendicular (A B C : cube_vertex) : Prop :=
  let v1 := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let v2 := (C.1 - B.1, C.2 - B.2, C.3 - B.3) in
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3) = 0

theorem angle_OP_AM_90_degrees
  (A B C D A1 B1 C1 D1 M O P : cube_vertex)
  (hM : is_midpoint M D D1)
  (hO : is_center O A B C D)
  (hP : is_point_on_edge P A1 B1)
  (hCube : Cube.mk (λ i, match i.1 with
                           | 0 => A | 1 => B | 2 => C | 3 => D
                           | 4 => A1 | 5 => B1 | 6 => C1 | 7 => D1 
                         end)
                   M O P) :
  perpendicular O P M :=
sorry

end angle_OP_AM_90_degrees_l679_679068


namespace shortest_distance_PQ_l679_679428

variables (t s : ℝ)

def P : ℝ × ℝ × ℝ := (1 + 3 * t, 2 - 3 * t, -1 + 2 * t)
def Q : ℝ × ℝ × ℝ := (-1 + 2 * s, -1 + 4 * s, 5 - 3 * s)
def distance_sq (P Q : ℝ × ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2

theorem shortest_distance_PQ :
  ∃ t s : ℝ, -1 + 2 * t = 0 ∧
  let P := (1 + 3 * t, 2 - 3 * t, 0),
      Q := (-1 + 2 * s, -1 + 4 * s, 5 - 3 * s),
      PQ_sq := distance_sq P Q,
      deriv := (dPQ_sq : derivative PQ_sq) in
      deriv = 0 ∧
  PQ_sq = 7.41 := 
sorry

end shortest_distance_PQ_l679_679428


namespace number_of_proper_subsets_B_is_31_l679_679361

-- Define the given set A
def A : Set ℤ := {-1, 0, 1}

-- Define the set B based on the conditions
def B : Set ℤ := {z | ∃ x ∈ A, ∃ y ∈ A, z = x + y}

-- Prove the number of proper subsets of B is 31
theorem number_of_proper_subsets_B_is_31 : (2 ^ B.to_finset.card - 1) = 31 := by
  sorry

end number_of_proper_subsets_B_is_31_l679_679361


namespace repeating_decimal_to_fraction_l679_679298

theorem repeating_decimal_to_fraction : (x : ℝ) (h : x = 0.353535...) → x = 35 / 99 :=
by
  sorry

end repeating_decimal_to_fraction_l679_679298


namespace sum_of_digits_of_cube_l679_679156

def sum_of_digits (n : ℕ) : ℕ :=
(n.to_digits 10).sum

theorem sum_of_digits_of_cube :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 :=
sorry

end sum_of_digits_of_cube_l679_679156


namespace overall_average_marks_l679_679954

theorem overall_average_marks 
  (n1 : ℕ) (m1 : ℕ) 
  (n2 : ℕ) (m2 : ℕ) 
  (n3 : ℕ) (m3 : ℕ) 
  (n4 : ℕ) (m4 : ℕ) 
  (h1 : n1 = 70) (h2 : m1 = 50) 
  (h3 : n2 = 35) (h4 : m2 = 60)
  (h5 : n3 = 45) (h6 : m3 = 55)
  (h7 : n4 = 42) (h8 : m4 = 45) :
  (n1 * m1 + n2 * m2 + n3 * m3 + n4 * m4) / (n1 + n2 + n3 + n4) = 9965 / 192 :=
by
  sorry

end overall_average_marks_l679_679954


namespace powers_of_three_two_digit_count_l679_679027

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679027


namespace arithmetic_sequence_goal_l679_679798

open Nat

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
axiom h1 : a 2 + a 7 = 10

-- Goal
theorem arithmetic_sequence_goal (h : is_arithmetic_sequence a d) : 3 * a 4 + a 6 = 20 :=
sorry

end arithmetic_sequence_goal_l679_679798


namespace third_side_of_triangle_l679_679073

theorem third_side_of_triangle (AB AC l : ℝ) (h1 : AB = c) (h2 : AC = b) : 
  BC = (b + c) * sqrt((b * c - l ^ 2) / (b * c)) :=
by 
  sorry

end third_side_of_triangle_l679_679073


namespace equivalence_of_functions_l679_679630

def f1 (x : ℝ) : ℝ := |x|
def f2 (t : ℝ) : ℝ := Real.sqrt (t^2)

theorem equivalence_of_functions : ∀ x : ℝ, f1 x = f2 x :=
by intros x; 
   sorry

end equivalence_of_functions_l679_679630


namespace domain_of_function_l679_679141

theorem domain_of_function : 
  {x : ℝ | 2 - x ≥ 0 ∧ x - 1 > 0} = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by {
  sorry
}

end domain_of_function_l679_679141


namespace complex_numbers_satisfying_condition_l679_679308

open Complex

theorem complex_numbers_satisfying_condition (z : ℂ) (n : ℕ) (h₀ : z ≠ 0) (h₁ : z^(n-1) = conj z) :
  ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ z = exp (2 * k * π * I / n) :=
  sorry

end complex_numbers_satisfying_condition_l679_679308


namespace necessary_and_sufficient_condition_for_f_to_be_odd_l679_679148

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (a b x : ℝ) : ℝ :=
  x * abs (x + a) + b

theorem necessary_and_sufficient_condition_for_f_to_be_odd (a b : ℝ) :
  is_odd_function (f a b) ↔ sorry :=
by
  -- This is where the proof would go.
  sorry

end necessary_and_sufficient_condition_for_f_to_be_odd_l679_679148


namespace money_received_by_A_l679_679570

theorem money_received_by_A (a_invest : ℝ) (b_invest : ℝ) (total_profit : ℝ) (managing_fee_ratio : ℝ):
  a_invest = 2000 →
  b_invest = 3000 →
  total_profit = 9600 →
  managing_fee_ratio = 0.10 →
  let managing_fee := managing_fee_ratio * total_profit in
    let remaining_profit := total_profit - managing_fee in
      let total_invest := a_invest + b_invest in
        let a_share := (a_invest / total_invest) * remaining_profit in
          let total_amount_received_by_A := managing_fee + a_share in
            total_amount_received_by_A = 4416 :=
by
  intros h1 h2 h3 h4
  simp only at h1 h2 h3 h4
  sorry

end money_received_by_A_l679_679570


namespace triangle_ratios_correct_l679_679627

noncomputable def triangle_conditions (c p1 p2 q1 q2 m_a m_b: Real) : Prop := 
  ∃ (ABC : Triangle), 
    ABC.side_AB = c ∧
    (∃ (M A1 : Point), 
      ALT_A1 = m_a ∧ 
      divides [A, M, A1] (p1, p2) ∧
      ORTHO_M = true) ∧
    (∃ (B1 : Point),
      ALT_B1 = m_b ∧ 
      divides [B, M, B1] (q1, q2)) ∧
    angle_ABC < 90 ∧
    angle_BCA < 90 ∧
    angle_CAB < 90

def triangle_ratios (p1 p2 q1 q2 : Real) : Prop := 
  let μ_a := p1 / (p1 + p2) 
  let μ_b := q1 / (q1 + q2)
  let μ_c := 2 - (μ_a + μ_b)
  μ_a + μ_b + μ_c = 2

theorem triangle_ratios_correct (c p1 p2 q1 q2 m_a m_b: Real) 
  (hc : c > 0) (hp1 : p1 > 0) (hp2 : p2 > 0) (hq1 : q1 > 0) (hq2 : q2 > 0) 
  (h_ma : m_a > 0) (h_mb : m_b > 0) :
  triangle_conditions c p1 p2 q1 q2 m_a m_b → triangle_ratios p1 p2 q1 q2 := 
by {
  intro h_conditions,
  let μ_a := p1 / (p1 + p2),
  let μ_b := q1 / (q1 + q2),
  let μ_c := 2 - (μ_a + μ_b),
  have h_sum : μ_a + μ_b + μ_c = 2,
  { sorry },
  exact h_sum,
}

end triangle_ratios_correct_l679_679627


namespace find_lambda_l679_679367

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_lambda (λ : ℝ) :
  let a := (-1, 2)
  let b := (2, -3)
  let c := (-4, 7)
  collinear (λ * a.1 + b.1, λ * a.2 + b.2) c ↔ λ = -2 :=
by
  assume λ
  let a := (-1, 2)
  let b := (2, -3)
  let c := (-4, 7)
  sorry

end find_lambda_l679_679367


namespace exists_large_amplitude_with_inscribed_squares_l679_679120

theorem exists_large_amplitude_with_inscribed_squares :
  ∃ A : ℝ, A > 1978 * 2 * Real.pi ∧
  (∃ S : set (ℝ × ℝ) → Prop, 
    (∀ s ∈ S, 
      s ⊆ { p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = A * Real.sin x} ∧
      function.has_four_distinct_vertices_on_graph s)) ∧
  S.card >= 1978 :=
sorry

end exists_large_amplitude_with_inscribed_squares_l679_679120


namespace original_square_area_l679_679613

theorem original_square_area (s : ℕ) (h1 : s + 5 = s + 5) (h2 : (s + 5)^2 = s^2 + 225) : s^2 = 400 :=
by
  sorry

end original_square_area_l679_679613


namespace find_annual_interest_rate_l679_679986

-- Definitions based on given conditions
def initial_investment : ℝ := 20000
def first_year_withdrawal : ℝ := 10000
def second_year_total : ℝ := 13200

-- The annual interest rate, which we need to prove is 0.1
def interest_rate : ℝ := 0.1

-- The proof statement
theorem find_annual_interest_rate :
  ∃ (x : ℝ),
    (let amount_after_first_year := initial_investment * (1 + x),
         reinvested_amount := amount_after_first_year - first_year_withdrawal,
         amount_after_second_year := reinvested_amount * (1 + x)
     in amount_after_second_year = second_year_total) →
    x = interest_rate :=
sorry

end find_annual_interest_rate_l679_679986


namespace sin_theta_line_plane_l679_679430

theorem sin_theta_line_plane :
  let d := (3, 4, 5)
  let n := (8, 3, -9)
  let dot_product := d.1 * n.1 + d.2 * n.2 + d.3 * n.3
  let norm_d := Real.sqrt (d.1^2 + d.2^2 + d.3^2)
  let norm_n := Real.sqrt (n.1^2 + n.2^2 + n.3^2)
  let cos_90_minus_theta := dot_product / (norm_d * norm_n)
  let sin_theta := Real.sqrt (1 - cos_90_minus_theta^2)
  in sin_theta=Real.sqrt(7619/7700) := by sorry

end sin_theta_line_plane_l679_679430


namespace geometric_sequence_a_eq_neg4_l679_679736

theorem geometric_sequence_a_eq_neg4 
    (a : ℝ)
    (h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : 
    a = -4 :=
sorry

end geometric_sequence_a_eq_neg4_l679_679736


namespace arithmetic_sequence_value_l679_679797

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end arithmetic_sequence_value_l679_679797


namespace tip_percentage_l679_679159

theorem tip_percentage
  (original_bill : ℝ)
  (shared_per_person : ℝ)
  (num_people : ℕ)
  (total_shared : ℝ)
  (tip_percent : ℝ)
  (h1 : original_bill = 139.0)
  (h2 : shared_per_person = 50.97)
  (h3 : num_people = 3)
  (h4 : total_shared = shared_per_person * num_people)
  (h5 : total_shared - original_bill = 13.91) :
  tip_percent = 13.91 / 139.0 * 100 := 
sorry

end tip_percentage_l679_679159


namespace lines_are_perpendicular_l679_679527

-- Define the direction vectors
def v1 : ℝ × ℝ × ℝ := (1, -1, 2)
def v2 : ℝ × ℝ × ℝ := (0, 2, 1)

-- Define the dot product function for 3-dimensional vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Definition saying u and v are perpendicular if their dot product is zero
def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The statement to be proved
theorem lines_are_perpendicular : is_perpendicular v1 v2 :=
sorry

end lines_are_perpendicular_l679_679527


namespace ages_of_Linda_and_Jane_l679_679842

theorem ages_of_Linda_and_Jane : 
  ∃ (J L : ℕ), 
    (L = 2 * J + 3) ∧ 
    (∃ (p : ℕ), Nat.Prime p ∧ p = L - J) ∧ 
    (L + J = 4 * J - 5) ∧ 
    (L = 19 ∧ J = 8) :=
by
  sorry

end ages_of_Linda_and_Jane_l679_679842


namespace central_not_axial_sym_l679_679411

-- Definitions of properties
structure Quad (α : Type) :=
(a b c d : α)
-- Conditions on diagonals
variable {V : Type} [InnerProductSpace ℝ V]

def is_perpendicular {V : Type} [InnerProductSpace ℝ V] (u v : V) : Prop :=
  inner u v = 0

def is_equal {V : Type} [InnerProductSpace ℝ V] (u v : V) : Prop :=
  ∥u∥ = ∥v∥

def is_quadrilateral_with_diagonals (Q : Quad V) : Prop :=
  ∃ p1 p2 p3 p4 : V, (Q.a = p1 ∧ Q.b = p2 ∧ Q.c = p3 ∧ Q.d = p4)

def diagonals_A {V : Type} [InnerProductSpace ℝ V] (Q : Quad V) : Prop :=
  is_quadrilateral_with_diagonals Q ∧ is_perpendicular (Q.a - Q.c) (Q.b - Q.d)

def diagonals_B {V : Type} [InnerProductSpace ℝ V] (Q : Quad V) : Prop :=
  is_quadrilateral_with_diagonals Q ∧ is_equal (Q.a - Q.c) (Q.b - Q.d)

def diagonals_C {V : Type} [InnerProductSpace ℝ V] (Q : Quad V) : Prop :=
  is_quadrilateral_with_diagonals Q ∧ is_perpendicular (Q.a - Q.c) (Q.b - Q.d) ∧ is_equal (Q.a - Q.c) (Q.b - Q.d)

def diagonals_D {V : Type} [InnerProductSpace ℝ V] (Q : Quad V) : Prop :=
  is_quadrilateral_with_diagonals Q ∧ ¬is_perpendicular (Q.a - Q.c) (Q.b - Q.d) ∧ ¬is_equal (Q.a - Q.c) (Q.b - Q.d)

-- Centrally symmetric but not axially symmetric property
theorem central_not_axial_sym {V : Type} [InnerProductSpace ℝ V] (Q : Quad V) : 
  diagonals_D Q → 
  (is_central_symmetrical (midpoint_quad Q) ∧ ¬is_axial_symmetrical (midpoint_quad Q)) := sorry

end central_not_axial_sym_l679_679411


namespace minimum_value_of_x_squared_l679_679523

theorem minimum_value_of_x_squared : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, y = x^2 → y ≥ 0 :=
by
  sorry

end minimum_value_of_x_squared_l679_679523


namespace inequality_solution_l679_679715

theorem inequality_solution (b c x : ℝ) (x1 x2 : ℝ)
  (hb_pos : b > 0) (hc_pos : c > 0) 
  (h_eq1 : x1 * x2 = 1) 
  (h_eq2 : -1 + x2 = 2 * x1) 
  (h_b : b = 5 / 2) 
  (h_c : c = 1) 
  : (1 < x ∧ x ≤ 5 / 2) ↔ (1 < x ∧ x ≤ 5 / 2) :=
sorry

end inequality_solution_l679_679715


namespace incorrect_real_root_statement_l679_679686

theorem incorrect_real_root_statement 
  (n : ℕ)
  (h1 : ∀ k, 0 ≤ k ∧ k < n → ∃ z : ℂ, z = complex.exp (2 * complex.I * real.pi * k / n) )
  (h2 : (∏ i in finset.range n, (λ x : ℂ, x - complex.exp (2 * complex.I * real.pi * i / n))) = (polynomial.C (1:ℂ) * (∏ i in finset.range n, (λ x : polynomial ℂ, polynomial.X - polynomial.C (complex.exp (2 * complex.I * real.pi * i / n))))) )
  (h3 : n.even → ∀ k, 0 < k ∧ k ≤ (n / 2) → complex.conj (complex.exp (2 * complex.I * real.pi * k / n)) = complex.exp (-2 * complex.I * real.pi * k / n))
  : ¬ (∀ x : ℂ, x^n = 1 → x = 1) :=
begin
  -- Proof would go here
  sorry
end

end incorrect_real_root_statement_l679_679686


namespace min_primes_to_guarantee_win_l679_679179

theorem min_primes_to_guarantee_win : 
  ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧
    p1 < 100 ∧ p2 < 100 ∧ p3 < 100 ∧
    (p1 % 10 = p2 / 10 % 10) ∧ 
    (p2 % 10 = p3 / 10 % 10) ∧ 
    (p3 % 10 = p1 / 10 % 10) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
  by sorry

end min_primes_to_guarantee_win_l679_679179


namespace three_powers_in_two_digit_range_l679_679002

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l679_679002


namespace cost_percentage_l679_679208

theorem cost_percentage (t b : ℝ) : 
  let C := t * b^4,
      C_new := t * (2 * b)^4
  in (C_new / C) * 100 = 1600 :=
by
  sorry

end cost_percentage_l679_679208


namespace exists_infinite_n_for_rectangle_l679_679501

theorem exists_infinite_n_for_rectangle :
  ∃ᶠ n in Filter.atTop, ∃ (m : ℕ), n = 9 + 12 * m ∧
  (∃ (arrangement : (Fin n) → Fin 3 → ℕ), 
      (∀ i j, arrangement i j ∈ Finset.range (3 * n) ∧ 
      (∀ j, ∑ i in Finset.univ, arrangement i j ≡ 0 [MOD 6]) ∧
      (∀ i, ∑ j in Finset.univ, arrangement i j ≡ 0 [MOD 6]))) :=
sorry

end exists_infinite_n_for_rectangle_l679_679501


namespace triangle_A1B1C1_angles_l679_679891

theorem triangle_A1B1C1_angles
  (A B C : Point)
  (A0 B0 C0 : Point)
  (A1 B1 C1 : Point)
  -- Conditions
  (h_angles : ∠ A B C = 120∠ ∧ ∠ B C A = 30∠ ∧ ∠ C A B = 30∠)
  (h_medians : Midpoint A B C0 ∧ Midpoint B C A0 ∧ Midpoint C A B0)
  (h_perpendicular_A : perpendicular (Line A0 A1) (Line B C))
  (h_perpendicular_B : perpendicular (Line B0 B1) (Line C A))
  (h_perpendicular_C : perpendicular (Line C0 C1) (Line A B)) :
    -- Conclusion
    ∠ A1 B1 C1 = 60∠ :=
sorry

end triangle_A1B1C1_angles_l679_679891


namespace number_of_dogs_l679_679056

theorem number_of_dogs (D : ℕ) (h1 : 7 + 4 + 18 + D = 29 + D)
                        (h2 : (7 * 2) + (4 * 2) + (18 * 4) + (4 * D) = (29 + D) + 74 - 94) :
  D = 3 :=
by
  have h3 : 94 + 4 * D = 103 + D := by rw [h2]
  have h4 : 4 * D - D = 9 := by linarith
  have h5 : 3 * D = 9 := by linarith
  have h6 : D = 3 := by linarith
  exact h6

end number_of_dogs_l679_679056


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679483

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679483


namespace tunnel_digging_duration_l679_679949

theorem tunnel_digging_duration (daily_progress : ℕ) (total_length_km : ℕ) 
    (meters_per_km : ℕ) (days_per_year : ℕ) : 
    daily_progress = 5 → total_length_km = 2 → meters_per_km = 1000 → days_per_year = 365 → 
    total_length_km * meters_per_km / daily_progress > 365 :=
by
  intros hprog htunnel hmeters hdays
  /- ... proof steps will go here -/
  sorry

end tunnel_digging_duration_l679_679949


namespace sin_cos_and_diff_values_l679_679711

noncomputable theory

-- Define the necessary inputs and properties
variables {α : ℝ}
axiom alpha_range : α ∈ (5 / 4 * Real.pi, 3 / 2 * Real.pi)
axiom tan_property : Real.tan α + 1 / Real.tan α = 8

-- State the theorem
theorem sin_cos_and_diff_values :
  (Real.sin α * Real.cos α = 1 / 8) ∧ (Real.sin α - Real.cos α = -Real.sqrt 3 / 2) :=
sorry

end sin_cos_and_diff_values_l679_679711


namespace train_speed_l679_679623

theorem train_speed (distance_meters : ℕ) (time_seconds : ℕ) 
  (h_distance : distance_meters = 150) (h_time : time_seconds = 20) : 
  distance_meters / 1000 / (time_seconds / 3600) = 27 :=
by 
  have h1 : distance_meters = 150 := h_distance
  have h2 : time_seconds = 20 := h_time
  -- other intermediate steps would go here, but are omitted
  -- for now, we assume the final calculation is:
  sorry

end train_speed_l679_679623


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679475

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679475


namespace decimal_to_binary_125_l679_679932

theorem decimal_to_binary_125 : ∃ (b : ℕ), 125 = b ∧ b = (1 * 2^6) + (1 * 2^5) + (1 * 2^4) + (1 * 2^3) + (1 * 2^2) + (0 * 2^1) + (1 * 2^0) :=
by {
  use 0b1111101,
  exact ⟨rfl, rfl⟩,
}

end decimal_to_binary_125_l679_679932


namespace wholesale_cost_proof_l679_679991

-- Definitions based on conditions
def wholesale_cost (W : ℝ) := W
def retail_price (W : ℝ) := 1.20 * W
def employee_paid (R : ℝ) := 0.90 * R

-- Theorem statement: given the conditions, prove that the wholesale cost is $200.
theorem wholesale_cost_proof : 
  ∃ W : ℝ, (retail_price W = 1.20 * W) ∧ (employee_paid (retail_price W) = 216) ∧ W = 200 :=
by 
  let W := 200
  have hp : retail_price W = 1.20 * W := by sorry
  have ep : employee_paid (retail_price W) = 216 := by sorry
  exact ⟨W, hp, ep, rfl⟩

end wholesale_cost_proof_l679_679991


namespace proof_problem_1_proof_problem_2_l679_679645

/-
  Problem statement and conditions:
  (1) $(2023-\sqrt{3})^0 + \left| \left( \frac{1}{5} \right)^{-1} - \sqrt{75} \right| - \frac{\sqrt{45}}{\sqrt{5}}$
  (2) $(\sqrt{3}-2)^2 - (\sqrt{2}+\sqrt{3})(\sqrt{3}-\sqrt{2})$
-/

noncomputable def problem_1 := 
  (2023 - Real.sqrt 3)^0 + abs ((1/5: ℝ)⁻¹ - Real.sqrt 75) - Real.sqrt 45 / Real.sqrt 5

noncomputable def problem_2 := 
  (Real.sqrt 3 - 2) ^ 2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2)

theorem proof_problem_1 : problem_1 = 5 * Real.sqrt 3 - 7 :=
  by
    sorry

theorem proof_problem_2 : problem_2 = 6 - 4 * Real.sqrt 3 :=
  by
    sorry


end proof_problem_1_proof_problem_2_l679_679645


namespace calculate_total_feet_in_garden_l679_679169

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l679_679169


namespace area_of_region_l679_679920

noncomputable def region_area : ℝ := 
  let u (x : ℝ) := 4 * x - 16 in
  let v (y : ℝ) := 3 * y + 9 in
  let Inequality := (|u x| + |v y| ≤ 6) in
  if Inequality
  then 6
  else 0

theorem area_of_region : region_area = 6 := 
by
  sorry

end area_of_region_l679_679920


namespace solve_for_x_l679_679867

theorem solve_for_x : ∀ x : ℤ, 5 - x = 8 → x = -3 :=
by
  intros x h
  sorry

end solve_for_x_l679_679867


namespace powers_of_three_two_digit_count_l679_679029

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679029


namespace projection_ratio_l679_679657

theorem projection_ratio (x y : ℚ)
  (h : (Matrix.of ![![9/50, -40/50], ![-40/50, 41/50]] ⬝ (λ i, if i = 0 then x else y) = (λ i, if i = 0 then x else y))) :
  y / x = 41 / 40 :=
by
  sorry

end projection_ratio_l679_679657


namespace total_travel_time_l679_679591

theorem total_travel_time (v1 v2 d : ℝ) (h1 : v1 = 6) (h2 : v2 = 4) (h3 : d = 24) : 
  let T := d / v1 + d / v2 in T = 10 := 
by
  synthesize_sorry

end total_travel_time_l679_679591


namespace larger_segment_length_l679_679251

theorem larger_segment_length 
  (x y : ℝ)
  (h1 : 40^2 = x^2 + y^2)
  (h2 : 90^2 = (110 - x)^2 + y^2) :
  110 - x = 84.55 :=
by
  sorry

end larger_segment_length_l679_679251


namespace probability_obtuse_triangle_is_one_fourth_l679_679809

-- Define the set of possible integers
def S : Set ℤ := {1, 2, 3, 4, 5, 6}

-- Condition for forming an obtuse triangle
def is_obtuse_triangle (a b c : ℤ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b ∧ 
  (a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2)

-- List of valid triples that can form an obtuse triangle
def valid_obtuse_triples : List (ℤ × ℤ × ℤ) :=
  [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 6), (3, 5, 6)]

-- Total number of combinations
def total_combinations : Nat := 20

-- Number of valid combinations for obtuse triangles
def valid_combinations : Nat := 5

-- Calculate the probability
def probability_obtuse_triangle : ℚ := valid_combinations / total_combinations

theorem probability_obtuse_triangle_is_one_fourth :
  probability_obtuse_triangle = 1 / 4 :=
by
  sorry

end probability_obtuse_triangle_is_one_fourth_l679_679809


namespace women_fraction_half_l679_679400

theorem women_fraction_half
  (total_people : ℕ)
  (married_fraction : ℝ)
  (max_unmarried_women : ℕ)
  (total_people_eq : total_people = 80)
  (married_fraction_eq : married_fraction = 1 / 2)
  (max_unmarried_women_eq : max_unmarried_women = 32) :
  (∃ (women_fraction : ℝ), women_fraction = 1 / 2) :=
by
  sorry

end women_fraction_half_l679_679400


namespace cos_15_degree_l679_679279

theorem cos_15_degree : 
  let d15 := 15 * Real.pi / 180
  let d45 := 45 * Real.pi / 180
  let d30 := 30 * Real.pi / 180
  Real.cos d15 = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degree_l679_679279


namespace lines_parallel_l679_679364

-- Define line l1 and line l2
def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

-- Prove that l1 is parallel to l2
theorem lines_parallel : ∀ x : ℝ, (l1 x - l2 x) = -4 := by
  sorry

end lines_parallel_l679_679364


namespace train_takes_longer_l679_679250

-- Definitions for the conditions
def train_speed : ℝ := 48
def ship_speed : ℝ := 60
def distance : ℝ := 480

-- Theorem statement for the proof
theorem train_takes_longer : (distance / train_speed) - (distance / ship_speed) = 2 := by
  sorry

end train_takes_longer_l679_679250


namespace find_n_l679_679725

def f (x : ℝ) : ℝ := 2^x + x - 5

theorem find_n (h : ∃ x, f x = 0) : ∃ n : ℤ, (∀ x, (n : ℝ) < x ∧ x < (n + 1 : ℝ) → f x = 0) ∧ n = 1 :=
by
  use 1
  split
  · sorry -- This is the part where we would provide the proof steps, which are skipped.
  · sorry -- This is the part where we would provide the proof steps, which are skipped.

end find_n_l679_679725


namespace conservation_center_total_turtles_l679_679260

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l679_679260


namespace principal_amount_compound_interest_l679_679151

theorem principal_amount_compound_interest :
  let A : ℝ := 10210.25
  let r : ℝ := 0.05
  let n : ℕ := 1
  let t : ℕ := 5
  let factor : ℝ := (1 + r / n)^t in
  (A / factor) ≈ 8000 :=
by
  let A : ℝ := 10210.25
  let r : ℝ := 0.05
  let n : ℕ := 1
  let t : ℕ := 5
  let factor : ℝ := (1 + r / n)^t
  sorry

end principal_amount_compound_interest_l679_679151


namespace sqrt_function_defined_range_l679_679776

theorem sqrt_function_defined_range (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x + 2)) ↔ x ≥ -2 := 
begin
  sorry -- proof omitted
end

end sqrt_function_defined_range_l679_679776


namespace sum_a_b_l679_679707

theorem sum_a_b (a b : ℕ) (h1 : 2 + 2 / 3 = 2^2 * (2 / 3))
(h2: 3 + 3 / 8 = 3^2 * (3 / 8)) 
(h3: 4 + 4 / 15 = 4^2 * (4 / 15)) 
(h_n : ∀ n, n + n / (n^2 - 1) = n^2 * (n / (n^2 - 1)) → 
(a = 9^2 - 1) ∧ (b = 9)) : 
a + b = 89 := 
sorry

end sum_a_b_l679_679707


namespace keiko_speed_l679_679821

-- Define the conditions
variable (width : ℝ) (time_diff : ℝ) (semi_circumference_diff : ℝ)
variable (C_o : ℝ) (C_i : ℝ)

-- Given values
def track_width : Real := 8
def time_difference : Real := 48
def radius_difference : Real := 16 * Real.pi -- corresponds to the circumference difference of the semicircle ends

-- Prove that Keiko's speed is π/3 meters per second
theorem keiko_speed (h1 : time_diff = 48) (h2 : semi_circumference_diff = 16 * Real.pi) :
  (semi_circumference_diff / time_diff) = (Real.pi / 3) := 
by
  sorry

end keiko_speed_l679_679821


namespace two_digit_numbers_of_form_3_pow_n_l679_679015

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679015


namespace fermat_large_prime_solution_l679_679833

theorem fermat_large_prime_solution (n : ℕ) (hn : n > 0) :
  ∃ (p : ℕ) (hp : Nat.Prime p) (x y z : ℤ), 
    (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^n + y^n ≡ z^n [ZMOD p]) :=
sorry

end fermat_large_prime_solution_l679_679833


namespace determine_T_1501_from_S_1000_l679_679683

variable (a d : ℤ)
def S (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

def T (n : ℕ) : ℤ := ∑ k in Finset.range (n + 1), S a d k

theorem determine_T_1501_from_S_1000 (S1000 : ℤ) (h : S 1000 = S1000) : T a d 1501 = (1501 * (1502) * (3 * a + 1500 * d)) / 6 :=
  by {
    sorry
  }

end determine_T_1501_from_S_1000_l679_679683


namespace p_q_sum_l679_679837

theorem p_q_sum (p q : ℝ) (hp : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 1)
  (hq : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 3)
  (hr : is_root (λ x, x^3 - 9 * x^2 + p * x - q) 5) :
  p + q = 38 :=
sorry

end p_q_sum_l679_679837


namespace books_loaned_out_l679_679573

theorem books_loaned_out (total_books_initial : ℕ) (books_end_month : ℕ) (perc_returned : ℝ) :
  total_books_initial = 75 →
  books_end_month = 68 →
  perc_returned = 0.65 →
  let books_not_returned := total_books_initial - books_end_month in
  let perc_not_returned := 1 - perc_returned in
  let x := books_not_returned / perc_not_returned in 
  x = 20 := by
  intros h1 h2 h3
  let books_not_returned := total_books_initial - books_end_month
  let perc_not_returned := 1 - perc_returned
  let x := books_not_returned / perc_not_returned
  sorry

end books_loaned_out_l679_679573


namespace diagonals_in_convex_polygon_l679_679371

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l679_679371


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679496

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679496


namespace vector_expression_evaluation_l679_679446

theorem vector_expression_evaluation (θ : ℝ) :
  let a := (2 * Real.cos θ, Real.sin θ)
  let b := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7 / 6 :=
by
  intros a b h
  sorry

end vector_expression_evaluation_l679_679446


namespace jeff_probability_multiple_of_4_l679_679077

theorem jeff_probability_multiple_of_4 :
  let card_prob := (1 : ℚ) / 12
      move_2_left_prob := (1 : ℚ) / 4
      move_2_right_prob := (1 : ℚ) / 2
      move_1_right_prob := (1 : ℚ) / 4
      start_multiple_4_prob := (3 : ℚ) / 12
      end_multiple_4_if_start_multiple_4_prob := 
        (move_2_left_prob * move_2_right_prob + move_2_right_prob * move_2_left_prob) in
  (start_multiple_4_prob * end_multiple_4_if_start_multiple_4_prob) = (1 : ℚ) / 32 :=
by {
  sorry
}

end jeff_probability_multiple_of_4_l679_679077


namespace number_of_women_in_preston_after_one_year_l679_679117

def preston_is_25_times_leesburg (preston leesburg : ℕ) : Prop := 
  preston = 25 * leesburg

def leesburg_population : ℕ := 58940

def women_percentage_leesburg : ℕ := 40

def women_percentage_preston : ℕ := 55

def growth_rate_leesburg : ℝ := 0.025

def growth_rate_preston : ℝ := 0.035

theorem number_of_women_in_preston_after_one_year : 
  ∀ (preston leesburg : ℕ), 
  preston_is_25_times_leesburg preston leesburg → 
  leesburg = 58940 → 
  (women_percentage_preston : ℝ) / 100 * (preston * (1 + growth_rate_preston) : ℝ) = 838788 :=
by 
  sorry

end number_of_women_in_preston_after_one_year_l679_679117


namespace smallest_vertical_segment_length_l679_679145

-- Define the absolute value function |x|
def abs (x : ℝ) : ℝ :=
  if x >= 0 then x else -x

-- Define the two functions y1 and y2
def y1 (x : ℝ) : ℝ := abs x
def y2 (x : ℝ) : ℝ := -x^2 - 3 * x - 2

-- Smallest possible length of vertical segment connecting y1 and y2
theorem smallest_vertical_segment_length : ∃ l : ℝ, l = 1 ∧ 
  (∀ x : ℝ, (y1 x - y2 x) ≥ l) ∧
  ∃ x : ℝ, (y1 x - y2 x) = l :=
by
  sorry

end smallest_vertical_segment_length_l679_679145


namespace glasses_needed_l679_679625

theorem glasses_needed (total_juice : ℕ) (juice_per_glass : ℕ) : Prop :=
  total_juice = 153 ∧ juice_per_glass = 30 → (total_juice + juice_per_glass - 1) / juice_per_glass = 6

-- This will state our theorem but we include sorry to omit the proof.

end glasses_needed_l679_679625


namespace diagonals_in_15_sided_polygon_l679_679374

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l679_679374


namespace avg_speed_including_stoppages_l679_679297

theorem avg_speed_including_stoppages (speed_without_stoppages : ℝ) (stoppage_time_per_hour : ℝ) 
  (h₁ : speed_without_stoppages = 60) (h₂ : stoppage_time_per_hour = 0.5) : 
  (speed_without_stoppages * (1 - stoppage_time_per_hour)) / 1 = 30 := 
  by 
  sorry

end avg_speed_including_stoppages_l679_679297


namespace OI_perp_MN_l679_679434

-- Definitions relating to the geometric points and their properties
variables {A B C D E F I O P Q M N : Type}
variables [circumcenter O A B C] [incenter I A B C]
variables [tangent_point D (incircle A B C) BC]
variables [tangent_point E (incircle A B C) CA]
variables [tangent_point F (incircle A B C) AB]
variables [midpoint P E PE] [midpoint Q F QF]
variables [midpoint M P PE] [midpoint N Q QF]

-- The theorem to prove
theorem OI_perp_MN 
  (h_circumcenter : Circumcenter O A B C)
  (h_incenter : Incenter I A B C)
  (h_tangent_points : (TangentPoint D (Incircle A B C) BC) ∧ 
                      (TangentPoint E (Incircle A B C) CA) ∧ 
                      (TangentPoint F (Incircle A B C) AB))
  (h_P_intersect : LineThrough FD CA P)
  (h_Q_intersect : LineThrough DE AB Q)
  (h_M_midpoint : Midpoint M P (Segment PE))
  (h_N_midpoint : Midpoint N Q (Segment QF)) :
  Perpendicular (Line O I) (Line M N) :=
sorry

end OI_perp_MN_l679_679434


namespace base_2_representation_of_125_l679_679921

theorem base_2_representation_of_125 : 
  ∃ (L : List ℕ), L = [1, 1, 1, 1, 1, 0, 1] ∧ 
    125 = (L.reverse.zipWith (λ b p, b * (2^p)) (List.range L.length)).sum := 
by
  sorry

end base_2_representation_of_125_l679_679921


namespace total_amount_l679_679958

theorem total_amount (T_pq r : ℝ) (h1 : r = 2/3 * T_pq) (h2 : r = 1600) : T_pq + r = 4000 :=
by
  -- proof skipped
  sorry

end total_amount_l679_679958


namespace functional_equation_solution_l679_679672

noncomputable def f (x : ℝ) : ℝ := x 

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ (x y : ℝ), x > 0 ∧ y > 0 → 
  f(x + f(y + x * y)) = (y + 1) * f(x + 1) - 1 ) : 
  ∀ x : ℝ, x > 0 → f(x) = x :=
sorry

end functional_equation_solution_l679_679672


namespace euler_lines_coincide_l679_679089

noncomputable def pedal_triangle (Δ : Triangle) : Triangle := sorry

noncomputable def points_of_tangency (Δ : Triangle) : Triangle := sorry

noncomputable def euler_line (Δ : Triangle) : Line := sorry

variables {ABC A1B1C1 A2B2C2 : Triangle}

def given_triangl_ABC_pedal_triangle (h1: A1B1C1 = pedal_triangle ABC) : Prop := 
  euler_line A2B2C2 = euler_line ABC

theorem euler_lines_coincide
  (h1: A1B1C1 = pedal_triangle ABC)
  (h2: A2B2C2 = points_of_tangency A1B1C1) :
  euler_line A2B2C2 = euler_line ABC :=
sorry

end euler_lines_coincide_l679_679089


namespace systematic_sampling_removal_count_l679_679247

-- Define the conditions
def total_population : Nat := 1252
def sample_size : Nat := 50

-- Define the remainder after division
def remainder := total_population % sample_size

-- Proof statement
theorem systematic_sampling_removal_count :
  remainder = 2 := by
    sorry

end systematic_sampling_removal_count_l679_679247


namespace path_length_of_B_l679_679268

noncomputable def lengthPathB (BC : ℝ) : ℝ :=
  let radius := BC
  let circumference := 2 * Real.pi * radius
  circumference

theorem path_length_of_B (BC : ℝ) (h : BC = 4 / Real.pi) : lengthPathB BC = 8 := by
  rw [lengthPathB, h]
  simp [Real.pi_ne_zero, div_mul_cancel]
  sorry

end path_length_of_B_l679_679268


namespace prime_divisors_consecutive_l679_679742

theorem prime_divisors_consecutive (p q : ℕ) [hp : Prime p] [hq : Prime q] (h1 : p < q) (h2 : q < 2 * p) :
  ∃ (n m : ℕ), Nat.gcd n m = 1 ∧ abs (n - m) = 1 ∧ (∀ p' : ℕ, Prime p' → p' ∣ n → p' = p) ∧ (∀ q' : ℕ, Prime q' → q' ∣ m → q' = q) := 
  sorry

end prime_divisors_consecutive_l679_679742


namespace min_x0_plus_p_constant_length_DE_l679_679731

-- (1) Minimum value of x₀ + p for given conditions
theorem min_x0_plus_p (p : ℝ) (x0 : ℝ) :
  (∃ x0 y0 m : ℝ, p > 1 ∧ y0^2 = 2*p*x0 ∧ (1/2)*(abs (sqrt(2*p + y0^2))) / (abs y0 + m) = p) →
  x0 + p = 1 + (sqrt 2)/2 := sorry

-- (2) Existence of real number n such that length of |DE| is constant
theorem constant_length_DE (p : ℝ) (n : ℝ) :
  (∃ y x m t : ℝ, p > 1 ∧ y^2 = 2*p*1 ∧ (1/2)*(abs (sqrt(2*p + y^2))) / (abs y + m) = p
    ∧ OA ⊥ OB ∧ (x - n)^2 + y^2 = 1 ∧ x = t*y + m) →
  n = 4 := sorry

end min_x0_plus_p_constant_length_DE_l679_679731


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679494

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679494


namespace fermat_point_sum_distance_l679_679917

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
    Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem fermat_point_sum_distance :
  let A := (0, 0)
  let B := (10, 0)
  let C := (3, 5)
  let P := (4, 2)
  distance A P + distance B P + distance C P = 2 * Real.sqrt 5 + 3 * Real.sqrt 10 :=
by {
    let A := (0, 0)
    let B := (10, 0)
    let C := (3, 5)
    let P := (4, 2)
    calc
      distance A P = 2 * Real.sqrt 5 : sorry,
      distance B P = 2 * Real.sqrt 10 : sorry,
      distance C P = Real.sqrt 10 : sorry,
      ...
}

end fermat_point_sum_distance_l679_679917


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679484

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679484


namespace angle_BCD_180_deg_l679_679053

theorem angle_BCD_180_deg (A B C D E : Type)
  [circle : circle A B E]
  (diamEB : diameter E B)
  (parallel_DC_EB : parallel D C E B)
  (parallel_AC_ED : parallel A C E D)
  (angle_ratio : ∃ x, ∠AEB = 3 * x ∧ ∠ABE = 7 * x) :
  ∠BCD = 180 :=
by
  sorry

end angle_BCD_180_deg_l679_679053


namespace HK_perpendicular_AB_l679_679082

-- Define the points, lines, and circle involved
structure IsoscelesTriangle (A B C : Point) : Prop :=
  (AB_eq_AC : distance A B = distance A C)

structure Circle (K : Point) (radius : ℝ) :=
  (tangent_point : Point)
  (tangency_condition : ∀ (A C : Point), distance K tangent_point = radius → ∠ A C K = 90)

-- Define the conditions provided in the problem
variables (A B C K H : Point)
variable (ω : Circle K (distance K C))

-- Given: ABC is an isosceles triangle with AB = AC
axiom isosceles_triangle_ABC : IsoscelesTriangle A B C

-- Given: Circle has center K and is tangent to AC at C
axiom circle_tangent_AC_at_C : ω.tangency_condition C A

-- Circle ω intersects BC at H
axiom intersect_BC_at_H : ∀ (B C H : Point), ∃ (ω : Circle K (distance K C)), ∈ H

-- To Prove: HK ⊥ AB
theorem HK_perpendicular_AB : ∠ (line_through K H) (line_through A B) = 90 := by 
sorry

end HK_perpendicular_AB_l679_679082


namespace largest_fraction_l679_679561

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (6 : ℚ) / 13
  let C := (18 : ℚ) / 37
  let D := (101 : ℚ) / 202
  let E := (200 : ℚ) / 399
  E > A ∧ E > B ∧ E > C ∧ E > D := by
  sorry

end largest_fraction_l679_679561


namespace optimal_roof_angle_no_friction_l679_679595

theorem optimal_roof_angle_no_friction {g x : ℝ} (hg : 0 < g) (hx : 0 < x) :
  ∃ α : ℝ, α = 45 :=
by
  sorry

end optimal_roof_angle_no_friction_l679_679595


namespace find_n_value_l679_679316

open Nat

def euler_totient (n : ℕ) : ℕ :=
  (univ_divisors n).filter (fun k => coprime k n).card

theorem find_n_value : 
  ∃ (N : ℕ) (α β γ : ℕ), 
  N = 3^α * 5^β * 7^γ ∧ euler_totient N = 3600 ∧ N = 7875 :=
sorry

end find_n_value_l679_679316


namespace batsman_average_after_20th_innings_l679_679974

theorem batsman_average_after_20th_innings 
    (score_20th_innings : ℕ)
    (previous_avg_increase : ℕ)
    (total_innings : ℕ)
    (never_not_out : Prop)
    (previous_avg : ℕ)
    : score_20th_innings = 90 →
      previous_avg_increase = 2 →
      total_innings = 20 →
      previous_avg = (19 * previous_avg + score_20th_innings) / total_innings →
      ((19 * previous_avg + score_20th_innings) / total_innings) + previous_avg_increase = 52 :=
by 
  sorry

end batsman_average_after_20th_innings_l679_679974


namespace Bret_catches_12_frogs_l679_679121

-- Conditions from the problem
def frogs_caught_by_Alster : Nat := 2
def frogs_caught_by_Quinn : Nat := 2 * frogs_caught_by_Alster
def frogs_caught_by_Bret : Nat := 3 * frogs_caught_by_Quinn

-- Statement of the theorem to be proved
theorem Bret_catches_12_frogs : frogs_caught_by_Bret = 12 :=
by
  sorry

end Bret_catches_12_frogs_l679_679121


namespace correct_quotient_l679_679597

def original_number : ℕ :=
  8 * 156 + 2

theorem correct_quotient :
  (8 * 156 + 2) / 5 = 250 :=
sorry

end correct_quotient_l679_679597


namespace light_travel_50_years_l679_679885

theorem light_travel_50_years :
  let one_year_distance := 9460800000000 -- distance light travels in one year
  let fifty_years_distance := 50 * one_year_distance
  let scientific_notation_distance := 473.04 * 10^12
  fifty_years_distance = scientific_notation_distance :=
by
  sorry

end light_travel_50_years_l679_679885


namespace math_club_team_selection_l679_679234

theorem math_club_team_selection : 
  let boys := 7
  let girls := 9
  let team_boys := 4
  let team_girls := 2
  nat.choose boys team_boys * nat.choose girls team_girls = 1260 :=
by
  let boys := 7
  let girls := 9
  let team_boys := 4
  let team_girls := 2
  sorry

end math_club_team_selection_l679_679234


namespace continuous_at_one_l679_679660

-- Conditions
def numerator (x : ℝ) := x^3 - 1
def denominator (x : ℝ) := x^2 - 5 * x + 6

-- Problem statement translated to Lean
theorem continuous_at_one : (numerator 1) / (denominator 1) = 0 :=
by
  have h_num: numerator 1 = 0 := by norm_num [numerator]
  have h_den: denominator 1 ≠ 0 := by norm_num [denominator]
  rw [h_num, h_den]
  norm_num
  sorry

end continuous_at_one_l679_679660


namespace solve_equation_l679_679869

theorem solve_equation 
  (x : ℝ)
  (hx : 3 * real.sqrt((x+1)^2) + real.sqrt(x^2 - 4*x + 4) = |1 + 6*x| - 4 * |x - 1|) :
  1 ≤ x ∧ x ≤ 2 :=
sorry

end solve_equation_l679_679869


namespace geometric_sum_eqn_l679_679350

theorem geometric_sum_eqn 
  (a1 q : ℝ) 
  (hne1 : q ≠ 1) 
  (hS2 : a1 * (1 - q^2) / (1 - q) = 1) 
  (hS4 : a1 * (1 - q^4) / (1 - q) = 3) :
  a1 * (1 - q^8) / (1 - q) = 15 :=
by
  sorry

end geometric_sum_eqn_l679_679350


namespace lesha_can_determine_numbers_l679_679748

-- Define the chessboard and the placement of numbers
def Chessboard : Type := Matrix (Fin 8) (Fin 8) ℕ

-- Numbered chessboard with the condition that the sum of each rectangle of two cells is provided
def isNumberedChessboard (board : Chessboard) (sums : Fin 8 × Fin 8 → Fin 8 × Fin 8 → ℕ) : Prop :=
  ∀ i j k l : Fin 8, sums (i, j) (k, l) = board i j + board k l

-- Subset containing numbers from 1 to 64 on the board
def validNumbers (board : Chessboard) : Prop := (∀ i j : Fin 8, 1 ≤ board i j ∧ board i j ≤ 64) ∧ 
  (Finset.univ.image (λ (i : Fin 8 × Fin 8), board i.1 i.2)).card = 64

-- Define diagonal property for 1 and 64
def sameDiagonal (board : Chessboard) : Prop :=
  ∃ (i j : Fin 8), board i j = 1 ∧ board i.1 i.2 = 64 ∧ (i + j = board i.1 i.2)

-- Main theorem stating that Lesha can uniquely determine the numbers on the board
theorem lesha_can_determine_numbers (board : Chessboard) (sums : Fin 8 × Fin 8 → Fin 8 × Fin 8 → ℕ)
  (hBoard : isNumberedChessboard board sums)
  (hValid : validNumbers board)
  (hDiagonal : sameDiagonal board) :
  ∀ (board' : Chessboard), isNumberedChessboard board' sums → validNumbers board' → sameDiagonal board' → board = board' :=
sorry

end lesha_can_determine_numbers_l679_679748


namespace compute_runs_l679_679788

theorem compute_runs
  (run_rate_first_10_overs : ℝ)
  (overs_first_10 : ℕ)
  (run_rate_remaining_40_overs : ℝ)
  (overs_remaining_40 : ℕ)
  (h1 : run_rate_first_10_overs = 3.2)
  (h2 : overs_first_10 = 10)
  (h3 : run_rate_remaining_40_overs = 4.75)
  (h4 : overs_remaining_40 = 40) :
  target (run_rate_first_10_overs * overs_first_10 + run_rate_remaining_40_overs * overs_remaining_40) = 222 :=
by {
  sorry
}

end compute_runs_l679_679788


namespace Christine_savings_l679_679276

theorem Christine_savings : 
  let e_sales := 12000 
  let c_sales := 8000
  let f_sales := 4000
  let e_commission_rate := 0.15
  let c_commission_rate := 0.10
  let f_commission_rate := 0.20
  let personal_needs_ratio := 0.55
  let investments_ratio := 0.30
  let e_commission := e_sales * e_commission_rate
  let c_commission := c_sales * c_commission_rate
  let f_commission := f_sales * f_commission_rate
  let total_earnings := e_commission + c_commission + f_commission
  let personal_needs := total_earnings * personal_needs_ratio
  let investments := total_earnings * investments_ratio
  in total_earnings - (personal_needs + investments) = 510 := by
  sorry

end Christine_savings_l679_679276


namespace num_pairs_abs_val_10_l679_679509

/-- The number of pairs (a, b) of integers such that |a| + |b| ≤ 10 is 221. -/
theorem num_pairs_abs_val_10 : 
  {p : ℤ × ℤ | Int.abs p.1 + Int.abs p.2 ≤ 10}.to_finset.card = 221 := 
sorry

end num_pairs_abs_val_10_l679_679509


namespace paul_is_19_years_old_l679_679111

theorem paul_is_19_years_old
  (mark_age : ℕ)
  (alice_age : ℕ)
  (paul_age : ℕ)
  (h1 : mark_age = 20)
  (h2 : alice_age = mark_age + 4)
  (h3 : paul_age = alice_age - 5) : 
  paul_age = 19 := by 
  sorry

end paul_is_19_years_old_l679_679111


namespace area_of_right_isosceles_triangle_l679_679406

theorem area_of_right_isosceles_triangle 
  (X Y Z : Type) [Field X] [Field Y] [Field Z] (angle_X angle_Z : ℝ) (XZ : ℝ)
  (h1 : angle_X = angle_Z)
  (h2 : XZ = 8 * Real.sqrt 2): 
  let XY := XZ / (Real.sqrt 2),
      YZ := XZ / (Real.sqrt 2) in
  (1/2) * XY * YZ = 32 := 
by
  sorry

end area_of_right_isosceles_triangle_l679_679406


namespace beans_termination_and_uniqueness_l679_679455

-- Define what a State is
def State := ℤ → ℕ -- A function from integers (representing squares) to natural numbers (number of beans)

-- Define the move operation
def move (s : State) (i : ℤ) : State :=
  if s i >= 2 then
    fun x => if x = i then s x - 2
             else if x = i-1 ∨ x = i+1 then s x + 1
             else s x
  else
    s

-- Define termination condition
def is_terminated (s : State) : Prop :=
  ∀ i, s i ≤ 1

-- Define the main theorem statement
theorem beans_termination_and_uniqueness (initial_state : State) :
  ∃ k : ℕ, ∃ final_state : State,
    (∀ i, (move^[k] initial_state) i = final_state i) ∧ is_terminated final_state
    ∧ ∀ (other_final_state : State),
        (∀ j, move other_final_state j = other_final_state) → final_state = other_final_state :=
sorry

end beans_termination_and_uniqueness_l679_679455


namespace find_rate_of_interest_l679_679863

theorem find_rate_of_interest (P I : ℝ) (R : ℝ) :
  P = 1200 → I = 432 → T = R → R = 6 :=
begin
  sorry
end

end find_rate_of_interest_l679_679863


namespace collinear_solution_angle_solution_l679_679746

variable {k : ℝ}
def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (0, -2)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, u = (c * v.1, c * v.2)

def angle_120 (u v : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, θ = 120 * (π / 180) ∧ (u.1 * v.1 + u.2 * v.2) = ∥u∥ * ∥v∥ * Real.cos θ

theorem collinear_solution : collinear (k • vector_a - vector_b) (vector_a + vector_b) ↔ k = -1 := 
by sorry

theorem angle_solution : 
  angle_120 (k • vector_a - vector_b) (vector_a + vector_b) ↔ k = -1 + Real.sqrt 3 ∨ k = -1 - Real.sqrt 3 :=
by sorry

end collinear_solution_angle_solution_l679_679746


namespace fifth_month_sale_correct_l679_679982

noncomputable def fifth_month_sale
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sales 0 + sales 1 + sales 2 + sales 3 + sixth_month_sale
  total_sales - known_sales

theorem fifth_month_sale_correct :
  ∀ (sales : Fin 4 → ℕ) (sixth_month_sale : ℕ) (average_sale : ℕ),
    sales 0 = 6435 →
    sales 1 = 6927 →
    sales 2 = 6855 →
    sales 3 = 7230 →
    sixth_month_sale = 5591 →
    average_sale = 6600 →
    fifth_month_sale sales sixth_month_sale average_sale = 13562 :=
by
  intros sales sixth_month_sale average_sale h0 h1 h2 h3 h4 h5
  unfold fifth_month_sale
  sorry

end fifth_month_sale_correct_l679_679982


namespace total_turtles_taken_l679_679262

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l679_679262


namespace average_children_in_families_with_children_l679_679171

theorem average_children_in_families_with_children
  (total_families : ℕ)
  (avg_children_per_family : ℝ)
  (childless_families : ℕ)
  (total_families = 12)
  (avg_children_per_family = 2.5)
  (childless_families = 2) :
  let total_children := total_families * avg_children_per_family,
      families_with_children := total_families - childless_families,
      avg_children_per_family_with_children := total_children / families_with_children 
  in avg_children_per_family_with_children = 3.0 :=
by sorry

end average_children_in_families_with_children_l679_679171


namespace distinct_ordered_pairs_count_l679_679317

def satisfies_condition (x y : ℕ) : Prop :=
  (x^4 * y^4 - 16 * x^2 * y^2 + 15 = 0)

theorem distinct_ordered_pairs_count :
  {xy_pairs : (ℕ × ℕ) | satisfies_condition xy_pairs.1 xy_pairs.2}.to_finset.card = 1 :=
by
  -- Proof goes here
  sorry

end distinct_ordered_pairs_count_l679_679317


namespace horner_at_3_l679_679693

def f (x : ℝ) : ℝ := 2 * x^5 - x + 3 * x^2 + x + 1

theorem horner_at_3 :
  let v0 := 2 in
  let v1 := v0 * 3 + 0 in
  let v2 := v1 * 3 - 1 in
  let v3 := v2 * 3 + 3 in
  f 3 = v3 :=
by
  show f 3 = 54
  sorry

end horner_at_3_l679_679693


namespace material_for_one_pillowcase_l679_679543

def material_in_first_bale (x : ℝ) : Prop :=
  4 * x + 1100 = 5000

def material_in_third_bale : ℝ := 0.22 * 5000

def total_material_used_for_producing_items (x y : ℝ) : Prop :=
  150 * (y + 3.25) + 240 * y = x

theorem material_for_one_pillowcase :
  ∀ (x y : ℝ), 
    material_in_first_bale x → 
    material_in_third_bale = 1100 → 
    (x = 975) → 
    total_material_used_for_producing_items x y →
    y = 1.25 :=
by
  intro x y h1 h2 h3 h4
  rw [h3] at h4
  have : 150 * (y + 3.25) + 240 * y = 975 := h4
  sorry

end material_for_one_pillowcase_l679_679543


namespace trig_identity_l679_679347

theorem trig_identity (α : ℝ) (h1 : α ∈ Ioo (3 / 2 * Real.pi) (2 * Real.pi)) (h2 : Real.cos (α + 2017 / 2 * Real.pi) = 3 / 5) :
  Real.sin α + Real.cos α = 1 / 5 :=
sorry

end trig_identity_l679_679347


namespace sin_bound_l679_679733

theorem sin_bound (a : ℝ) (h : ¬ ∃ x : ℝ, Real.sin x > a) : a ≥ 1 := 
sorry

end sin_bound_l679_679733


namespace rational_roots_of_quadratic_l679_679674

theorem rational_roots_of_quadratic (r : ℚ) :
  (∃ a b : ℤ, a ≠ b ∧ (r * a^2 + (r + 1) * a + r = 1 ∧ r * b^2 + (r + 1) * b + r = 1)) ↔ (r = 1 ∨ r = -1 / 7) :=
by
  sorry

end rational_roots_of_quadratic_l679_679674


namespace find_P_l679_679394

variable (A B C D M N : ℝ × ℝ)
variable (θ : ℝ)

def is_square (A B C D : ℝ × ℝ) : Prop := 
  A = (0,0) ∧ B = (4,0) ∧ C = (4,4) ∧ D = (0,4)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := 
  (P.1 + Q.1) / 2, (P.2 + Q.2) / 2

def angle_ratio (θ : ℝ) : Prop :=
  tan 2 * θ = 4/3

theorem find_P
  (h1 : is_square A B C D)
  (h2 : M = midpoint A D)
  (h3 : N = midpoint M D)
  (h4 : angle_ratio θ) :
  ∃ P, P = 2 := 
  sorry

end find_P_l679_679394


namespace number_of_frogs_is_3_l679_679398

structure Amphibian :=
(toad : Prop) -- True if the amphibian is a toad, False if a frog

def amphibian_toads : Amphibian → Prop := λ a => a.toad
def amphibian_frogs : Amphibian → Prop := λ a => ¬ a.toad

variables Brian Chris LeRoy Mike Julia : Amphibian

axiom Brian_statement : amphibian_toads Chris
axiom Chris_statement : amphibian_toads Chris ↔ amphibian_frogs LeRoy
axiom LeRoy_statement : amphibian_frogs Mike
axiom Mike_statement : amphibian_toads Brian + amphibian_toads Chris + amphibian_toads LeRoy + amphibian_toads Mike + amphibian_toads Julia ≥ 3
axiom Julia_statement : amphibian_toads Brian = amphibian_toads Julia

theorem number_of_frogs_is_3 : amphibian_frogs Brian + amphibian_frogs Chris + amphibian_frogs LeRoy + amphibian_frogs Mike + amphibian_frogs Julia = 3 := sorry

end number_of_frogs_is_3_l679_679398


namespace ways_to_write_2020_as_sum_of_twos_and_threes_l679_679035

def write_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2 + 1) else 0

theorem ways_to_write_2020_as_sum_of_twos_and_threes :
  write_as_sum_of_twos_and_threes 2020 = 337 :=
sorry

end ways_to_write_2020_as_sum_of_twos_and_threes_l679_679035


namespace area_of_billboard_l679_679237

variable (L W : ℕ) (P : ℕ)
variable (hW : W = 8) (hP : P = 46)

theorem area_of_billboard (h1 : P = 2 * L + 2 * W) : L * W = 120 :=
by
  sorry

end area_of_billboard_l679_679237


namespace triangle_segment_length_l679_679133

theorem triangle_segment_length (base : ℝ) (h_base : base = 36) 
  (A : ℝ) (h_A : A > 0) 
  (h_half_area : (A / 2) = (1/2) * A ) : 
  ∃ segment_length : ℝ, segment_length = 18 * Real.sqrt(2) :=
by
  use 18 * Real.sqrt(2)
  sorry

end triangle_segment_length_l679_679133


namespace winning_candidate_percentage_l679_679792

theorem winning_candidate_percentage (P : ℕ) (majority : ℕ) (total_votes : ℕ) (h1 : majority = 188) (h2 : total_votes = 470) (h3 : 2 * majority = (2 * P - 100) * total_votes) : 
  P = 70 := 
sorry

end winning_candidate_percentage_l679_679792


namespace find_x_perpendicular_l679_679365

-- Definitions used in the conditions
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Condition: vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- The theorem we want to prove
theorem find_x_perpendicular : ∀ x : ℝ, perpendicular a (b x) → x = -8 / 3 :=
by
  intros x h
  sorry

end find_x_perpendicular_l679_679365


namespace volume_surface_area_ratio_l679_679984

-- Defining the shape structure and conditions
def cube := ℕ
def unit_volume (c : cube) : ℕ := 1
def total_cubes : ℕ := 9
def volume_of_shape : ℕ := total_cubes * unit_volume (1)
def exposed_faces_per_corner_cube : ℕ := 3
def total_corner_cubes : ℕ := 8
def surface_area_of_shape : ℕ := total_corner_cubes * exposed_faces_per_corner_cube

-- Proving the ratio of volume to surface area
theorem volume_surface_area_ratio : volume_of_shape / surface_area_of_shape = 3 / 8 :=
by
  have V := volume_of_shape
  have S := surface_area_of_shape
  have ratio : ℚ := V / S
  have desired_ratio : ℚ := 3 / 8
  exact Eq.refl desired_ratio

end volume_surface_area_ratio_l679_679984


namespace decimal_to_binary_125_l679_679928

theorem decimal_to_binary_125 : 
  (∃ (digits : list ℕ), (125 = digits.length - 1).sum (λ i, (2 ^ i) * digits.nth_le i sorry) ∧ digits = [1,1,1,1,1,0,1]) := 
sorry

end decimal_to_binary_125_l679_679928


namespace quadrilateral_area_24_l679_679602

open Classical

noncomputable def quad_area (a b : ℤ) (h : a > b ∧ b > 0) : ℤ :=
let P := (a, b)
let Q := (2*b, a)
let R := (-a, -b)
let S := (-2*b, -a)
-- The proved area
24

theorem quadrilateral_area_24 (a b : ℤ) (h : a > b ∧ b > 0) :
  quad_area a b h = 24 :=
sorry

end quadrilateral_area_24_l679_679602


namespace adults_wearing_sunglasses_l679_679107

def total_adults : ℕ := 2400
def one_third_of_adults (total : ℕ) : ℕ := total / 3
def women_wearing_sunglasses (women : ℕ) : ℕ := (15 * women) / 100
def men_wearing_sunglasses (men : ℕ) : ℕ := (12 * men) / 100

theorem adults_wearing_sunglasses : 
  let women := one_third_of_adults total_adults
  let men := total_adults - women
  let women_in_sunglasses := women_wearing_sunglasses women
  let men_in_sunglasses := men_wearing_sunglasses men
  women_in_sunglasses + men_in_sunglasses = 312 :=
by
  sorry

end adults_wearing_sunglasses_l679_679107


namespace combined_mpg_l679_679370

-- Define the conditions as given in the problem
def Henry_mpg := 30
def Laura_mpg := 50
def Henry_distance := 120
def Laura_distance := 60

-- Problem statement: Prove the combined rate of miles per gallon
theorem combined_mpg : 
  (Henry_distance + Laura_distance) / ((Henry_distance / Henry_mpg) + (Laura_distance / Laura_mpg)) = 34.615 :=
by
  sorry

end combined_mpg_l679_679370


namespace probability_point_in_rectangle_l679_679113

theorem probability_point_in_rectangle (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 2500) (hy : 0 ≤ y ∧ y ≤ 2505) :
  (prob (x > 8 * y | 0 ≤ x ∧ x ≤ 2500 ∧ 0 ≤ y ∧ y ≤ 2505) = 125 / 2002) :=
sorry

end probability_point_in_rectangle_l679_679113


namespace find_hundreds_digit_l679_679901

theorem find_hundreds_digit :
  ∃ n : ℕ, (n % 37 = 0) ∧ (n % 173 = 0) ∧ (10000 ≤ n) ∧ (n < 100000) ∧ ((n / 1000) % 10 = 3) ∧ (((n / 100) % 10) = 2) :=
sorry

end find_hundreds_digit_l679_679901


namespace rabbit_distribution_l679_679399

theorem rabbit_distribution:
  let rabbits := ["Robert", "Rebecca", "Benjamin", "Daisy", "Edward", "Lily"]
  let parents := ["Robert", "Rebecca"]
  let stores := {1, 2, 3, 4, 5}
  ∃ (distribution : Finset (Set (Finset ℕ))),
    (distribution.card = 380) ∧
    (∀ (d : Set (Finset ℕ)), d ∈ distribution →
      (∀ s ∈ d, ∃ r ∈ s, r ∉ parents)  -- No store gets both a parent and any of their offspring
    ) 

end rabbit_distribution_l679_679399


namespace tan_symmetry_l679_679202

theorem tan_symmetry : 
  ∀ x : ℝ, y = tan (x + π / 3) → ∃ k : ℤ, (x = -π / 3 + (k * π / 2)) :=
by sorry

end tan_symmetry_l679_679202


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679485

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679485


namespace remainder_when_dividing_by_2x_minus_4_l679_679191

def f (x : ℝ) := 4 * x^3 - 9 * x^2 + 12 * x - 14
def g (x : ℝ) := 2 * x - 4

theorem remainder_when_dividing_by_2x_minus_4 : f 2 = 6 := by
  sorry

end remainder_when_dividing_by_2x_minus_4_l679_679191


namespace length_CD_l679_679281

-- Definition of the circle and its properties
def center_O := (0, 0)
def radius := 1
def diameter := 2 * radius

-- Chord AB and its length
def A := (-sqrt 3 / 2, -1 / 2)
def B := (sqrt 3 / 2, -1 / 2)
def AB := sqrt (3)
def angle_AOB := 120

-- Point C on circle
def C := (0, 1)

-- Point D on circle, such that CD is perpendicular to AB
axiom D : Type
axiom on_circle : D → Prop
axiom perpendicular : D → Prop

-- Assuming D lies on the circle and CD is perpendicular to AB
axiom D_on_circle : on_circle D
axiom CD_perpendicular_AB : perpendicular D

-- Find the length of segment CD
theorem length_CD : ∀ (D : D), (on_circle D) → (perpendicular D) → (CD = radius) :=
by {
    intro D,
    assume D_on_circle,
    assume CD_perpendicular_AB,
    sorry
}

end length_CD_l679_679281


namespace hyperbola_asymptote_l679_679311

theorem hyperbola_asymptote :
  ∀ x y : ℝ, x^2 - y^2 / 2 = 1 → (sqrt 2) * x = y ∨ (sqrt 2) * x = -y :=
by
  intros x y h
  sorry

end hyperbola_asymptote_l679_679311


namespace find_angle_x_l679_679801

theorem find_angle_x (P Q R T : Type) 
  (PQRS_line : collinear P Q R S) 
  (not_on_line : ¬ collinear Q R T) 
  (angle_PQT : ∠ PQT = 170)
  (angle_QRT : ∠ QRT = 50)
  (angle_QTR : ∠ QTR = 60) : 
  ∠ QTR = 60 :=
sorry

end find_angle_x_l679_679801


namespace annual_production_2010_l679_679129

-- Defining the parameters
variables (a x : ℝ)

-- Define the growth formula
def annual_growth (initial : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate)^years

-- The statement we need to prove
theorem annual_production_2010 :
  annual_growth a x 5 = a * (1 + x) ^ 5 :=
by
  sorry

end annual_production_2010_l679_679129


namespace bridge_max_weight_l679_679245

variables (M K Mi B : ℝ)

-- Given conditions
def kelly_weight : K = 34 := sorry
def kelly_megan_relation : K = 0.85 * M := sorry
def mike_megan_relation : Mi = M + 5 := sorry
def total_excess : K + M + Mi = B + 19 := sorry

-- Proof goal: The maximum weight the bridge can hold is 100 kg.
theorem bridge_max_weight : B = 100 :=
by
  sorry

end bridge_max_weight_l679_679245


namespace center_of_circle_l679_679718

-- Defining the equation of the circle as a hypothesis
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y = 0

-- Stating the theorem about the center of the circle
theorem center_of_circle : ∀ x y : ℝ, circle_eq x y → (x = 2 ∧ y = -1) :=
by
  sorry

end center_of_circle_l679_679718


namespace problem_statement_l679_679554

theorem problem_statement (h : 36 = 6^2) : 6^15 / 36^5 = 7776 := by
  sorry

end problem_statement_l679_679554


namespace geometric_series_sum_range_l679_679708

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def is_geometric_series (a : ℕ → ℝ) (a2 a5 : ℝ) :=
  geometric_sequence a ∧ a 2 = a2 ∧ a 5 = a5

theorem geometric_series_sum_range : 
  ∀ (a : ℕ → ℝ), is_geometric_series a 2 16 → 
  ∀ (n : ℕ), 0 < n → 
  8 < (∑ i in finset.range n, a i * a (i + 1)) :=
sorry

end geometric_series_sum_range_l679_679708


namespace dentist_ratio_l679_679553

-- Conditions
def cost_cleaning : ℕ := 70
def cost_filling : ℕ := 120
def cost_extraction : ℕ := 290

-- Theorem statement
theorem dentist_ratio : (cost_cleaning + 2 * cost_filling + cost_extraction) / cost_filling = 5 := 
by
  -- To be proven
  sorry

end dentist_ratio_l679_679553


namespace max_friends_upper_bound_l679_679402

theorem max_friends_upper_bound (m : ℕ) (hm : m ≥ 3) (friends : Type) (is_friend : friends → friends → Prop)
  (mutual : ∀ a b, is_friend a b ↔ is_friend b a)
  (no_self_friend : ∀ a, ¬is_friend a a)
  (common_friend : ∀ (people : Finset friends) (hm_size : people.card = m), ∃ c, ∀ x ∈ people, is_friend c x) :
  ∀ p : friends, ∃ (k : ℕ), ∀ k' : ℕ, k' ≤ k → (∀ person, person ≠ p → size (set_of (λ q, is_friend p q)) ≤ m) :=
begin
  sorry
end

end max_friends_upper_bound_l679_679402


namespace birthday_guests_l679_679845

theorem birthday_guests (total_guests : ℕ) (women men children guests_left men_left children_left : ℕ)
  (h_total : total_guests = 60)
  (h_women : women = total_guests / 2)
  (h_men : men = 15)
  (h_children : children = total_guests - (women + men))
  (h_men_left : men_left = men / 3)
  (h_children_left : children_left = 5)
  (h_guests_left : guests_left = men_left + children_left) :
  (total_guests - guests_left) = 50 :=
by sorry

end birthday_guests_l679_679845


namespace barrels_needed_l679_679989

-- Define the dimensions of the barrel in meters
def Length : ℝ := 6.40
def Width  : ℝ := 9
def Height : ℝ := 5.20

-- The total volume of sand
def TotalVolume : ℝ := Length * Width * Height

-- The volume of each barrel
def VolumeOfOneBarrel : ℝ := 1

-- The number of barrels needed
def NumberOfBarrelsNeeded : ℕ := Int.ceil (TotalVolume / VolumeOfOneBarrel)

-- Prove that the number of barrels needed is 300
theorem barrels_needed : NumberOfBarrelsNeeded = 300 := by
  have h1 : TotalVolume = 299.52 := by norm_num [TotalVolume, Length, Width, Height]
  have h2 : TotalVolume / VolumeOfOneBarrel = 299.52 := by norm_num [TotalVolume, VolumeOfOneBarrel]
  have h3 : Int.ceil 299.52 = 300 := by sorry
  exact h3

end barrels_needed_l679_679989


namespace conservation_center_total_turtles_l679_679259

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l679_679259


namespace nested_operation_result_l679_679381

def operation (a b : ℝ) : ℝ := (Real.sqrt (a^2 + 3 * a * b + b^2 - 2 * a - 2 * b + 4)) / (a * b + 4)

theorem nested_operation_result : 
  operation (operation (operation (operation (operation (operation (operation (operation (operation (operation 2010 2009) 2008) 2007) 2006) 2005) 2004) 2003) 2002) 2001) 2000) 1999) = (Real.sqrt 15) / 9 := 
by
  sorry

end nested_operation_result_l679_679381


namespace cat_finishes_food_on_saturday_cat_finishes_food_on_saturday_l679_679122

theorem cat_finishes_food_on_saturday :
  (A B C D : ℕ → ℚ) -- declare Roy's daily morning, evening consumption and check if sum is greater
  (T : ℕ → ℕ) -- checks how many days it will take 
  (day_of_week : ℕ) -- to get day of week from rest

: A(1) + B(1) = fromRational 2/3
: C(8) / (A(1) + B(1)) = 12 -- daily check
: T(12..15) = day_of_week
: day_of_week = 6 := sorry

-- check daily food consmption and try to find total days to finish food 
-- importing rational numbers to make sure we keep compute rational numbers
noncomputable
theorem cat_finishes_food_on_saturday (morning_consumption evening_consumption total_cans : ℚ)
  (initial_day : ℕ) : 
  (morning_consumption = 1/2) →
  (evening_consumption = 1/6) →
  (total_cans = 8) →
  (initial_day = 1) → -- Monday is represented by 1
    let daily_consumption := morning_consumption + evening_consumption in
    let total_days := total_cans / daily_consumption in
    let final_day := (initial_day + total_days) % 7 in
    final_day = 6 := -- totals to Saturday after 12 days  
begin
  sorry
end

end cat_finishes_food_on_saturday_cat_finishes_food_on_saturday_l679_679122


namespace limit_length_of_line_l679_679232

noncomputable def line_growth_series : ℝ := 
  2 + ∑' n : ℕ, (1/(3^n)) * (√3) + ∑' n : ℕ, 1/(3^n)

theorem limit_length_of_line :
  (2 + ∑' n : ℕ, (1/(3^n)) * (√3) + 1/(3^n)) = (1/2) * (6 + √3) :=
sorry

end limit_length_of_line_l679_679232


namespace solve_for_x_l679_679868

theorem solve_for_x : 
  ∃ x₁ x₂ : ℝ, abs (x₁ - 0.175) < 1e-3 ∧ abs (x₂ - 18.325) < 1e-3 ∧
    (∀ x : ℝ, (8 * x ^ 2 + 120 * x + 7) / (3 * x + 10) = 4 * x + 2 → x = x₁ ∨ x = x₂) := 
by 
  sorry

end solve_for_x_l679_679868


namespace brenda_has_eight_l679_679971

-- Define the amounts each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (emma_money / 4)
def jeff_money : ℕ := (2 * daya_money) / 5
def brenda_money : ℕ := jeff_money + 4

-- Define the theorem to prove Brenda's money is 8
theorem brenda_has_eight : brenda_money = 8 := by
  sorry

end brenda_has_eight_l679_679971


namespace problem_solution_l679_679838

noncomputable def verify_solution (x y z : ℝ) : Prop :=
  x = 12 ∧ y = 10 ∧ z = 8 →
  (x > 4) ∧ (y > 4) ∧ (z > 4) →
  ( ( (x + 3)^2 / (y + z - 3) ) + 
    ( (y + 5)^2 / (z + x - 5) ) + 
    ( (z + 7)^2 / (x + y - 7) ) = 45)

theorem problem_solution :
  verify_solution 12 10 8 := by
  sorry

end problem_solution_l679_679838


namespace ramu_selling_price_l679_679860

-- Define the problem conditions
def boughtPrice : ℝ := 42000
def repairCost : ℝ := 12000
def profitPercent : ℝ := 20.185185185185187

-- The total cost is the sum of the bought price and the repair cost.
def totalCost := boughtPrice + repairCost

-- The profit is the total cost multiplied by the profit percentage as a decimal.
def profit := totalCost * (profitPercent / 100)

-- The selling price is the total cost plus the profit.
def sellingPrice := totalCost + profit

-- Prove that the calculated selling price is 64900.
theorem ramu_selling_price : sellingPrice = 64900 := by
  -- We are skipping the proof here and just stating the theorem.
  sorry

end ramu_selling_price_l679_679860


namespace range_m_distinct_roots_l679_679383

theorem range_m_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4^x₁ - m * 2^(x₁+1) + 2 - m = 0) ∧ (4^x₂ - m * 2^(x₂+1) + 2 - m = 0)) ↔ 1 < m ∧ m < 2 :=
by
  sorry

end range_m_distinct_roots_l679_679383


namespace diagonals_in_15_sided_polygon_l679_679373

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l679_679373


namespace equal_black_white_areas_l679_679600

-- Define what it means for a point to be inside a regular 2n-gon
def inside_regular_polygon (P : ℝ × ℝ) (n : ℕ) (vertices : Fin (2 * n) → ℝ × ℝ) : Prop :=
  ∀ i : Fin (2 * n), 
    let (x, y) := P in 
    -- some condition to ensure P is inside the polygon can be added here
    true

-- Define the area function of a triangle formed by vertices A, B and a point P
def area_triangle (A B P : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x, y) := P in
  abs ((x1 * y2 + x2 * y + x * y1) - (y1 * x2 + y2 * x + y * x1)) / 2

-- Define the color alternation (black or white) for the triangles
def is_black_triangle (i : Fin (2 * n)) : Bool :=
  i.val % 2 = 0

-- Define the function to compute total black and white areas
def total_colored_areas (P : ℝ × ℝ) (n : ℕ) (vertices : Fin (2 * n) → ℝ × ℝ) : ℝ × ℝ :=
  let areas := List.map (λ i : Fin (2 * n), area_triangle (vertices i) (vertices ((i + 1) % (2 * n))) P) (Finset.range (2 * n)).val
  let (black_areas, white_areas) := List.partitionWith (λ ⟨i, area⟩, if is_black_triangle i then Sum.inl area else Sum.inr area) (List.zip (Finset.range (2 * n)).val areas)
  (black_areas.sum, white_areas.sum)

-- The main theorem to prove the equality of the areas
theorem equal_black_white_areas (n : ℕ) (vertices : Fin (2 * n) → ℝ × ℝ) (P : ℝ × ℝ) 
  (h : inside_regular_polygon P n vertices) :
  ∃ (a b : ℝ), total_colored_areas P n vertices = (a, b) ∧ a = b :=
by
  sorry

end equal_black_white_areas_l679_679600


namespace toilet_paper_duration_l679_679270

theorem toilet_paper_duration :
  let bill_weekday := 3 * 5
  let wife_weekday := 4 * 8
  let kid_weekday := 5 * 6
  let total_weekday := bill_weekday + wife_weekday + 2 * kid_weekday
  let bill_weekend := 4 * 6
  let wife_weekend := 5 * 10
  let kid_weekend := 6 * 5
  let total_weekend := bill_weekend + wife_weekend + 2 * kid_weekend
  let total_week := 5 * total_weekday + 2 * total_weekend
  let total_squares := 1000 * 300
  let weeks_last := total_squares / total_week
  let days_last := weeks_last * 7
  days_last = 2615 :=
sorry

end toilet_paper_duration_l679_679270


namespace gcd_solution_l679_679039

theorem gcd_solution {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 10) : Nat.gcd (12 * m) (18 * n) = 60 := 
sorry

end gcd_solution_l679_679039


namespace find_x_l679_679972

theorem find_x (x : ℝ) : 0.6 * x = (x / 3) + 110 → x = 412.5 := 
by
  intro h
  sorry

end find_x_l679_679972


namespace cars_selected_l679_679607

theorem cars_selected (num_cars num_clients selections_made total_selections : ℕ)
  (h1 : num_cars = 16)
  (h2 : num_clients = 24)
  (h3 : selections_made = 2)
  (h4 : total_selections = num_clients * selections_made) :
  num_cars * (total_selections / num_cars) = 48 :=
by
  sorry

end cars_selected_l679_679607


namespace arithmetic_seq_a2_a8_a5_l679_679282

-- Define the sequence and sum conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Define the given conditions
axiom seq_condition (n : ℕ) : (1 - q) * S n + q * a n = 1
axiom q_nonzero : q * (q - 1) ≠ 0
axiom geom_seq : ∀ n, a n = q^(n - 1)

-- Main theorem (consistent with both parts (Ⅰ) and (Ⅱ) results)
theorem arithmetic_seq_a2_a8_a5 (S_arith : S 3 + S 6 = 2 * S 9) : a 2 + a 5 = 2 * a 8 :=
by
    sorry

end arithmetic_seq_a2_a8_a5_l679_679282


namespace trapezoid_areas_sum_l679_679996

noncomputable def sum_of_trapezoid_areas : ℝ :=
  let sides := {4, 6, 8, 10}
  let semi_perimeter a b c := (a + b + c) / 2
  let herons_area a b c :=
    let s := semi_perimeter a b c
    real.sqrt (s * (s - a) * (s - b) * (s - c))
  let valid_areas := 
    [herons_area 4 8 10, herons_area 8 4 10].filter (λ A, A ≠ 0)
  valid_areas.sum

theorem trapezoid_areas_sum : sum_of_trapezoid_areas = 2 * real.sqrt 33 := sorry

end trapezoid_areas_sum_l679_679996


namespace M_trajectory_is_parabola_l679_679800

variable (M : ℝ³) (plane_ABB_1A_1 : Subspace ℝ ℝ³) (plane_ADD_1A_1 : Subspace ℝ ℝ³) 
variable (line_BC : Line ℝ³)

-- Assumptions / Conditions
-- M is on the plane ABB_1A_1
axiom M_on_plane_ABB_1A_1 : M ∈ plane_ABB_1A_1
-- Distance from M to plane ADD_1A_1 is equal to the distance from M to line BC
axiom dist_M_to_add1a1_eq_dist_M_to_BC : dist M plane_ADD_1A_1 = dist M line_BC

-- The proposition to prove
theorem M_trajectory_is_parabola : is_parabola (trajectory M plane_ABB_1A_1) :=
by
  sorry

end M_trajectory_is_parabola_l679_679800


namespace triangle_angle_bisector_theorem_l679_679915

variable {α : Type*} [LinearOrderedField α]

theorem triangle_angle_bisector_theorem (A B C D : α)
  (h1 : A^2 = (C + D) * (B - (B * D / C)))
  (h2 : B / C = (B * D / C) / D) :
  A^2 = C * B - D * (B * D / C) := 
  by
  sorry

end triangle_angle_bisector_theorem_l679_679915


namespace families_seating_arrangements_l679_679580

theorem families_seating_arrangements : 
  let factorial := Nat.factorial
  let family_ways := factorial 3
  let bundles := family_ways * family_ways * family_ways
  let bundle_ways := factorial 3
  bundles * bundle_ways = (factorial 3) ^ 4 := by
  sorry

end families_seating_arrangements_l679_679580


namespace rate_of_current_l679_679899

variable (c : ℝ)

def speed_boat_still_water := 15 -- km/hr
def distance_downstream := 3.6 -- km
def time_downstream := 12 / 60 -- hours (12 minutes)

theorem rate_of_current (h : distance_downstream = (speed_boat_still_water + c) * time_downstream) : c = 3 :=
by
  sorry

end rate_of_current_l679_679899


namespace edge_length_of_cube_l679_679802

universe u

-- Definitions based on conditions
def length : ℝ := 2
def width : ℝ := 4
def height : ℝ := 8
def volume_rectangular_solid : ℝ := length * width * height
def volume_equal (v1 v2 : ℝ) : Prop := v1 = v2

-- Theorem proof problem
theorem edge_length_of_cube (s : ℝ) : 
  volume_equal (s^3) volume_rectangular_solid → s = 4 :=
by
  sorry

end edge_length_of_cube_l679_679802


namespace ratio_sheep_to_horses_l679_679634

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l679_679634


namespace two_digit_numbers_of_form_3_pow_n_l679_679016

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679016


namespace powers_of_three_two_digit_count_l679_679033

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679033


namespace min_value_of_a_l679_679357

noncomputable def x (t a : ℝ) : ℝ :=
  5 * (t + 1)^2 + a / (t + 1)^5

theorem min_value_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ 2 * Real.sqrt ((24 / 7)^7) :=
sorry

end min_value_of_a_l679_679357


namespace time_spent_making_coffee_l679_679076

-- Conditions
def batch_size_gallons : ℝ := 1.5
def ounces_per_gallon : ℝ := 128
def daily_consumption_ounces : ℝ := 96 / 2
def time_to_make_batch_hours : ℝ := 20
def total_days : ℝ := 24

-- Question translated to proof problem
theorem time_spent_making_coffee : 
  (total_days / (batch_size_gallons * ounces_per_gallon / daily_consumption_ounces)) * time_to_make_batch_hours = 120 :=
by
  sorry

end time_spent_making_coffee_l679_679076


namespace h_h_3_eq_3568_l679_679765

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h_3_eq_3568_l679_679765


namespace unique_tangent_line_value_l679_679322

theorem unique_tangent_line_value (b : ℝ) :
  (∃ m n : ℝ, (n = - (1 / 2) * m + real.log m) ∧ (n = (1 / 2) * m + b) ∧ 
  (- (1 / 2) + 1 / m = 1 / 2)) → b = -1 := by
  sorry

end unique_tangent_line_value_l679_679322


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679495

theorem distance_between_sasha_and_kolya_when_sasha_finishes : 
  ∀ {v_S v_L v_K : ℝ}, 
    (∀ t_S t_L t_K : ℝ, 
      0 < v_S ∧ 0 < v_L ∧ 0 < v_K ∧
      t_S = 100 / v_S ∧ t_L = 90 / v_L ∧ t_K = 100 / v_K ∧
      v_L = 0.9 * v_S ∧ v_K = 0.9 * v_L)
    → (100 - (v_K * (100 / v_S)) = 19) :=
begin
  sorry
end

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679495


namespace michelle_initial_crayons_l679_679102

variable (m j : Nat)

axiom janet_crayons : j = 2
axiom michelle_has_after_gift : m + j = 4

theorem michelle_initial_crayons : m = 2 :=
by
  sorry

end michelle_initial_crayons_l679_679102


namespace diff_4_5_9_exists_l679_679340

theorem diff_4_5_9_exists (S : Finset ℕ) (h_card : S.card = 70) (h_max : ∀ a ∈ S, a ≤ 200) :
  ∃ x y ∈ S, x ≠ y ∧ (|x - y| = 4 ∨ |x - y| = 5 ∨ |x - y| = 9) :=
by
  sorry

end diff_4_5_9_exists_l679_679340


namespace stock_profit_percentage_l679_679244

theorem stock_profit_percentage 
  (total_stock : ℝ) (total_loss : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ)
  (percentage_sold_at_profit : ℝ) :
  total_stock = 12499.99 →
  total_loss = 500 →
  profit_percentage = 0.20 →
  loss_percentage = 0.10 →
  (0.10 * ((100 - percentage_sold_at_profit) / 100) * 12499.99) - (0.20 * (percentage_sold_at_profit / 100) * 12499.99) = 500 →
  percentage_sold_at_profit = 20 :=
sorry

end stock_profit_percentage_l679_679244


namespace sum_real_roots_eq_4_l679_679653

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f x ≤ f y

noncomputable def sum_of_real_roots (f : ℝ → ℝ) : ℝ :=
  if is_even f ∧ is_increasing_on_nonneg f then 4 else 0

theorem sum_real_roots_eq_4 (f : ℝ → ℝ) (h₁ : is_even f) (h₂ : is_increasing_on_nonneg f) :
  sum_of_real_roots f = 4 :=
by
  sorry

end sum_real_roots_eq_4_l679_679653


namespace fractional_equation_no_solution_l679_679504

theorem fractional_equation_no_solution (x : ℝ) (h1 : x ≠ 3) : (2 - x) / (x - 3) ≠ 1 + 1 / (3 - x) :=
by
  sorry

end fractional_equation_no_solution_l679_679504


namespace race_distance_between_Sasha_and_Kolya_l679_679489

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679489


namespace two_digit_numbers_of_form_3_pow_n_l679_679022

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679022


namespace sum_of_edges_112_l679_679540

-- Define the problem parameters
def volume (a b c : ℝ) : ℝ := a * b * c
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)
def sum_of_edges (a b c : ℝ) : ℝ := 4 * (a + b + c)

-- The main theorem 
theorem sum_of_edges_112
  (b s : ℝ) (h1 : volume (b / s) b (b * s) = 512)
  (h2 : surface_area (b / s) b (b * s) = 448)
  (h3 : 0 < b ∧ 0 < s) : 
  sum_of_edges (b / s) b (b * s) = 112 :=
sorry

end sum_of_edges_112_l679_679540


namespace powers_of_three_two_digit_count_l679_679031

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679031


namespace smallest_alpha_for_quadratic_l679_679680

theorem smallest_alpha_for_quadratic (a b c : ℝ) 
  (h : ∀ x ∈ Icc (0 : ℝ) 1, |a*x^2 + b*x + c| ≤ 1) 
  : |b| ≤ 8 := 
sorry

end smallest_alpha_for_quadratic_l679_679680


namespace log_problem_l679_679218

theorem log_problem : (2 / 3 * log 8 + (log 5)^2 + log 2 * log 50 + log 25) = 3 := sorry

end log_problem_l679_679218


namespace sum_slopes_const_zero_l679_679143

-- Define variables and constants
variable (p : ℝ) (h : 0 < p)

-- Define parabola and circle equations
def parabola_C1 (x y : ℝ) : Prop := y^2 = 2 * p * x
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 = p^2

-- Condition: The line segment length from circle cut by directrix
def segment_length_condition : Prop := ∃ d : ℝ, d^2 + 3 = p^2

-- The main theorem to prove
theorem sum_slopes_const_zero
  (A : ℝ × ℝ)
  (F : ℝ × ℝ := (p / 2, 0))
  (M N : ℝ × ℝ)
  (line_n_through_A : ∀ x : ℝ, x = 1 / p - 1 + 1 / p → (1 / p - 1 + x) = 0)
  (intersection_prop: parabola_C1 p M.1 M.2 ∧ parabola_C1 p N.1 N.2) 
  (slope_MF : ℝ := (M.2 / (p / 2 - M.1)) ) 
  (slope_NF : ℝ := (N.2 / (p / 2 - N.1))) :
  slope_MF + slope_NF = 0 := 
sorry

end sum_slopes_const_zero_l679_679143


namespace hh_value_l679_679763

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem hh_value :
  h(h(3)) = 3568 :=
by
  sorry

end hh_value_l679_679763


namespace total_eggs_l679_679136

theorem total_eggs (eggs_today eggs_yesterday : ℕ) (h_today : eggs_today = 30) (h_yesterday : eggs_yesterday = 19) : eggs_today + eggs_yesterday = 49 :=
by
  sorry

end total_eggs_l679_679136


namespace num_lines_through_point_l679_679794

-- Definitions based on conditions in a)
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def line_through_point (a b : ℕ) (p : ℕ × ℕ) : Prop := 
  let (x, y) := p in
  a > 0 ∧ is_prime a ∧ b > 0 ∧ ∃ c d : ℕ, c = a - 6 ∧ d = b - 5 ∧ 
  c * d = 30 ∧ 
  x * b + y * a = a * b

-- Statement of the theorem to be proved
theorem num_lines_through_point : 
  (∃ a b : ℕ, line_through_point a b (6, 5)) → 
  (∃ a1 b1 a2 b2 : ℕ, 
    a1 ≠ a2 ∧ 
    line_through_point a1 b1 (6, 5) ∧ 
    line_through_point a2 b2 (6, 5)) :=
sorry

end num_lines_through_point_l679_679794


namespace imaginary_part_of_i_l679_679832

theorem imaginary_part_of_i : (complex.imag (complex.i) = 1) :=
by
  sorry

end imaginary_part_of_i_l679_679832


namespace hyperbola_equation_l679_679385

theorem hyperbola_equation (P : ℝ × ℝ) (asymptote_eq : ℝ → ℝ) (λ : ℝ) (h : P = (2, real.sqrt 2)) (hasymptote : ∀ x, asymptote_eq x = x) :
    (∃ λ, ∀ (x y : ℝ), (x^2 - y^2 = λ ↔ λ ≠ 0) ∧ (2^2 - (real.sqrt 2)^2 = λ)) →
    (∀ (x y : ℝ), ((x^2 / 2) - (y^2 / 2) = 1)) :=
by
  sorry

end hyperbola_equation_l679_679385


namespace fifty_gon_parallel_sides_exists_l679_679539

theorem fifty_gon_parallel_sides_exists (L : Fin 50 → ℤ) (h1 : (∀ i : Fin 50, 1 ≤ L i ∧ L i ≤ 50)) (h2 : ∀ i : Fin 25, L i - L (i + 25) = 25 ∨ L i - L (i + 25) = -25) : 
  ∃ k : Fin 50, (L (k + 1) + L (k + 2) + ... + L (k + 24)) = (L (k + 26) + L (k + 27) + ... + L (k + 49)) :=
sorry

end fifty_gon_parallel_sides_exists_l679_679539


namespace minimum_sugar_correct_l679_679639

noncomputable def minimum_sugar (f : ℕ) (s : ℕ) : ℕ := 
  if (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) then s else sorry

theorem minimum_sugar_correct (f s : ℕ) : 
  (f ≥ 8 + s / 2 ∧ f ≤ 3 * s) → s ≥ 4 :=
by sorry

end minimum_sugar_correct_l679_679639


namespace sum_of_squares_l679_679530

theorem sum_of_squares (x : ℚ) (h : x + 2 * x + 3 * x = 14) : 
  (x^2 + (2 * x)^2 + (3 * x)^2) = 686 / 9 :=
by
  sorry

end sum_of_squares_l679_679530


namespace fair_game_l679_679062

open_locale big_operators

inductive Ball
| zero : Ball
| one : Ball
| two : Ball
| four : Ball
| six : Ball
| seven : Ball
| eight : Ball
| nine : Ball

namespace Ball

def is_odd : Ball → Prop
| zero := false
| one := true
| two := false
| four := false
| six := false
| seven := true
| eight := false
| nine := true

def is_even (b : Ball) : Prop := ¬ is_odd b

def fair_game_ming : Ball → Prop
| zero := true
| one := true
| two := true
| four := true
| six := false
| seven := false
| eight := false
| nine := false

def fair_game_gang (b : Ball) : Prop := ¬ fair_game_ming b

lemma prob_odd_ball :
  (∀ (b : Ball), b = Ball.one ∨ b = Ball.seven ∨ b = Ball.nine) ∨
  ¬ (∀ (b : Ball), b = Ball.one ∨ b = Ball.seven ∨ b = Ball.nine) :=
sorry

lemma game_initial_not_fair :
  ¬ ((∀ (b : Ball), is_odd b → 3/8 = 1/2) ∧ (∀ (b : Ball), is_even b → 5/8 = 1/2)) :=
sorry

theorem fair_game :
  (∀ (b : Ball), fair_game_ming b → 4/8 = 1/2) ∧
  (∀ (b : Ball), fair_game_gang b → 4/8 = 1/2) :=
sorry

end Ball

end fair_game_l679_679062


namespace remainder_when_divided_l679_679318

noncomputable def poly := (8 * x^4) - (10 * x^3) + (7 * x^2) - (5 * x) - 30
noncomputable def div := 2 * x - 4
noncomputable def x_value := 2

theorem remainder_when_divided :
  (poly.eval x_value) % (div.eval x_value) = 36 :=
sorry

end remainder_when_divided_l679_679318


namespace find_CD_maximum_area_BCD_l679_679410

/-- Quadrilateral ABCD properties -/
structure Quadrilateral :=
  (A B C D : Type)
  (AB : ℝ)
  (AD : ℝ)
  (BD : ℝ)
  (angle_BAD : ℝ)
  (angle_BCD : ℝ)
  (angle_BDC : ℝ)
  (AB_eq_2 : AB = 2)
  (AD_eq_sqrt2 : AD = Real.sqrt 2)
  (BD_eq_sqrt10 : BD = Real.sqrt 10)
  (angle_BAD_eq_3_angle_BCD : angle_BAD = 3 * angle_BCD)
  (angle_BDC_eq_5pi_over_12 : angle_BDC = (5 * Real.pi / 12))

/-- Prove that CD = sqrt(15) given the conditions -/
theorem find_CD (quad : Quadrilateral) : 
  ∃ CD : ℝ, CD = Real.sqrt 15 :=
by {
  cases quad,
  sorry
}

/-- Prove that the maximum area of triangle BCD is (5√2)/2 + 5/2 -/
theorem maximum_area_BCD (quad : Quadrilateral) : 
  ∃ area : ℝ, area = 5 * Real.sqrt 2 / 2 + 5 / 2 :=
by {
  cases quad,
  sorry
}

end find_CD_maximum_area_BCD_l679_679410


namespace mario_oranges_l679_679905

theorem mario_oranges (M L N T x : ℕ) 
  (H_L : L = 24) 
  (H_N : N = 96) 
  (H_T : T = 128) 
  (H_total : x + L + N = T) : 
  x = 8 :=
by
  rw [H_L, H_N, H_T] at H_total
  linarith

end mario_oranges_l679_679905


namespace decimal_to_binary_125_l679_679931

theorem decimal_to_binary_125 : ∃ (b : ℕ), 125 = b ∧ b = (1 * 2^6) + (1 * 2^5) + (1 * 2^4) + (1 * 2^3) + (1 * 2^2) + (0 * 2^1) + (1 * 2^0) :=
by {
  use 0b1111101,
  exact ⟨rfl, rfl⟩,
}

end decimal_to_binary_125_l679_679931


namespace conjugate_of_z_l679_679726

theorem conjugate_of_z (z : ℂ) (hz : z = (2 * complex.I) / (1 + complex.I)) : (conj z) = 1 - complex.I :=
by sorry

end conjugate_of_z_l679_679726


namespace total_feet_in_garden_l679_679167

theorem total_feet_in_garden (num_dogs num_ducks feet_per_dog feet_per_duck : ℕ)
  (h1 : num_dogs = 6) (h2 : num_ducks = 2)
  (h3 : feet_per_dog = 4) (h4 : feet_per_duck = 2) :
  num_dogs * feet_per_dog + num_ducks * feet_per_duck = 28 :=
by
  sorry

end total_feet_in_garden_l679_679167


namespace three_powers_in_two_digit_range_l679_679003

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l679_679003


namespace bottles_per_pack_l679_679464

-- Define the conditions
def daily_soda_intake : ℝ := 0.5  -- daily amount of soda consumed by Rebecca
def bottles_left_after_4_weeks : ℕ := 4  -- bottles remaining after 4 weeks
def num_packs : ℕ := 3  -- number of packs bought by Rebecca

-- Define the number of days in 4 weeks
def days_in_4_weeks : ℕ := 28

-- Define the function or statement we need to prove
theorem bottles_per_pack 
  (h_bottles_left_after_4_weeks : ℕ = bottles_left_after_4_weeks)
  (h_num_packs : ℕ = num_packs)
  (h_daily_soda_intake : ℝ = daily_soda_intake)
  (h_days_in_4_weeks : ℕ = days_in_4_weeks) :
  let total_bottles := 3 * (∀ (x : ℕ), x)
  in
  let bottles_consumed := daily_soda_intake * days_in_4_weeks
  in
  (total_bottles - bottles_left_after_4_weeks) = bottles_consumed →
  (∀ (x : ℕ), x = 6) :=
by
  sorry

end bottles_per_pack_l679_679464


namespace phoenix_hike_length_l679_679454

theorem phoenix_hike_length (a b c d : ℕ)
  (h1 : a + b = 22)
  (h2 : b + c = 26)
  (h3 : c + d = 30)
  (h4 : a + c = 26) :
  a + b + c + d = 52 :=
sorry

end phoenix_hike_length_l679_679454


namespace value_of_a_l679_679388

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  sorry

end value_of_a_l679_679388


namespace percentage_copper_second_alloy_l679_679267

variables (c₁ c₂ : ℝ) (x y z : ℝ)

-- Conditions
def condition1 := c₁ = 0.20 * 30
def condition2 := y = 70
def condition3 := z = 24.9

-- Question
def final_alloy_copper := c₁ + (x/100.0) * y

-- Lean 4 statement
theorem percentage_copper_second_alloy :
  c₁ = 0.20 * 30 →
  y = 100 - 30 →
  final_alloy_copper c₁ x y = 24.9 →
  x = 27 :=
begin
  intros,
  unfold final_alloy_copper,
  sorry
end

end percentage_copper_second_alloy_l679_679267


namespace girls_to_boys_ratio_l679_679051

theorem girls_to_boys_ratio (g b : ℕ) (h1 : g = b + 5) (h2 : g + b = 35) : g / b = 4 / 3 :=
by
  sorry

end girls_to_boys_ratio_l679_679051


namespace flowchart_output_value_l679_679778

theorem flowchart_output_value :
  ∃ n : ℕ, S = n * (n + 1) / 2 ∧ n = 10 → S = 55 :=
by
  sorry

end flowchart_output_value_l679_679778


namespace repeating_decimal_as_fraction_l679_679303

theorem repeating_decimal_as_fraction : ∀ x : ℝ, (x = 0.353535...) → (99 * x = 35) → x = 35 / 99 := by
  intros x h1 h2
  sorry

end repeating_decimal_as_fraction_l679_679303


namespace num_smaller_cubes_of_length_1_l679_679228

-- Define the properties of the original cube
def original_edge_length : ℕ := 6
def original_surface_area := 6 * (original_edge_length * original_edge_length)

-- Define the condition for the total surface area of smaller cubes
def new_total_surface_area := (10 / 3) * original_surface_area

-- Define the proof problem
theorem num_smaller_cubes_of_length_1 :
  ∃ (n : ℕ), n = 56 ∧
    ∃ (smaller_cubes : list ℕ),
      (∀ x ∈ smaller_cubes, x ∈ [1, 2, 3, 4, 5]) ∧
      sum (map (λ x, x^3) smaller_cubes) = original_edge_length^3 ∧
      (∑ x in smaller_cubes, (6 * x^2)) = new_total_surface_area :=
sorry

end num_smaller_cubes_of_length_1_l679_679228


namespace sufficient_but_not_necessary_perpendicular_condition_l679_679363

noncomputable def perpendicular_condition (l m n : Line) (α : Plane) : Prop :=
  (l ∩ m = ∅) ∧ (l ∩ n = ∅) ∧ (m ⊆ α) ∧ (n ⊆ α)

theorem sufficient_but_not_necessary_perpendicular_condition
  (l m n : Line) (α : Plane)
  (hlmn : perpendicular_condition l m n α) :
  (perpendicular_to_plane l α) -> (perpendicular_to_lines l m n)
  ∧ (¬ (perpendicular_to_lines l m n)->(perpendicular_to_plane l α)) :=
sorry

end sufficient_but_not_necessary_perpendicular_condition_l679_679363


namespace proof_problem_l679_679770

noncomputable def log2 : ℝ := Real.log 3 / Real.log 2
noncomputable def log5 : ℝ := Real.log 3 / Real.log 5

variables {x y : ℝ}

theorem proof_problem
  (h1 : log2 > 1)
  (h2 : 0 < log5 ∧ log5 < 1)
  (h3 : (log2^x - log5^x) ≥ (log2^(-y) - log5^(-y))) :
  x + y ≥ 0 :=
sorry

end proof_problem_l679_679770


namespace base8_subtraction_correct_l679_679186

-- Define what it means to subtract in base 8
def base8_sub (a b : ℕ) : ℕ :=
  let a_base10 := 8 * (a / 10) + (a % 10)
  let b_base10 := 8 * (b / 10) + (b % 10)
  let result_base10 := a_base10 - b_base10
  8 * (result_base10 / 8) + (result_base10 % 8)

-- The given numbers in base 8
def num1 : ℕ := 52
def num2 : ℕ := 31
def expected_result : ℕ := 21

-- The proof problem statement
theorem base8_subtraction_correct : base8_sub num1 num2 = expected_result := by
  sorry

end base8_subtraction_correct_l679_679186


namespace triangle_angle_area_l679_679342

theorem triangle_angle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : a * cos B + b * cos A = 2 * c * sin C)
  (h2 : 0 < C ∧ C < π) :
  (C = π / 6 ∨ C = 5 * π / 6) ∧ 
  (b = 2 * sqrt 3) ∧ (c = sqrt 19) → 
    (∃ S, (S = 7 * sqrt 3 / 2) ∨ (S = sqrt 3 / 2)) :=
by
  sorry  -- proof to be completed

end triangle_angle_area_l679_679342


namespace find_genuine_coins_l679_679914

-- Define the conditions and the question
variables {α : Type} [linear_ordered_add_comm_group α]

structure Coin (α : Type) :=
(weight : α)

def is_genuine (coins : list (Coin α)) (k : Coin α) : Prop :=
  k ∈ coins ∧ ∀ c ∈ coins, c.weight = k.weight

def is_counterfeit (coins : list (Coin α)) (k : Coin α) : Prop :=
  k ∈ coins ∧ ∀ c ∈ coins, c.weight < k.weight

theorem find_genuine_coins 
  (coins : list (Coin α))
  (h₁ : coins.length = 7)
  (h₂ : ∃ gcs: list (Coin α), gcs.length = 5 ∧ ∀ c ∈ gcs, is_genuine coins c)
  (h₃ : ∃ ccs: list (Coin α), ccs.length = 2 ∧ ∀ c ∈ ccs, is_counterfeit coins c)
  (h₄ : ∀ c ∈ coins, is_genuine coins c ∨ is_counterfeit coins c):
  ∃ g1 g2 g3, g1 ∈ coins ∧ g2 ∈ coins ∧ g3 ∈ coins ∧ g1 ≠ g2 ∧ g1 ≠ g3 ∧ g2 ≠ g3
    ∧ is_genuine coins g1 ∧ is_genuine coins g2 ∧ is_genuine coins g3 :=
sorry -- Proof goes here, but is not required

end find_genuine_coins_l679_679914


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679008

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679008


namespace trigonometric_identity_l679_679321

theorem trigonometric_identity : 
  sin (20 * Real.pi / 180) * sin (80 * Real.pi / 180) - cos (160 * Real.pi / 180) * sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trigonometric_identity_l679_679321


namespace polar_center_coordinates_l679_679070

-- Define polar coordinate system equation
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sin θ

-- Define the theorem: Given the equation of a circle in polar coordinates, its center in polar coordinates.
theorem polar_center_coordinates :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ ρ, polar_circle ρ θ) →
  (∀ ρ θ, polar_circle ρ θ → 0 ≤ θ ∧ θ < 2 * Real.pi → (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = -1 ∧ θ = 3 * Real.pi / 2)) :=
by {
  sorry 
}

end polar_center_coordinates_l679_679070


namespace roots_of_equation_l679_679531

theorem roots_of_equation :
  {x : ℝ // x ≠ 2 ∧ x ≠ -2} → (15 / (x^2 - 4) - 2 / (x - 2) = 1) ↔ (x = -3 ∨ x = 5) :=
by
  intro h
  sorry

end roots_of_equation_l679_679531


namespace mrs_wilsborough_change_l679_679844

/-- Mrs. Wilsborough bought 4 VIP tickets at $120 each, 
    5 regular tickets at $60 each, and 3 special discount tickets at $30 each.
    She paid for all tickets with a $1000 bill.
    This theorem proves that the change she received is $130 if 
    the cost calculations are performed correctly. -/

theorem mrs_wilsborough_change :
  ∀ (VIP_count regular_count discount_count : ℕ), 
    ∀ (VIP_price regular_price discount_price: ℕ), 
    VIP_count = 4 → 
    regular_count = 5 → 
    discount_count = 3 → 
    VIP_price = 120 → 
    regular_price = 60 → 
    discount_price = 30 → 
    let VIP_total := VIP_count * VIP_price in
    let regular_total := regular_count * regular_price in
    let discount_total := discount_count * discount_price in
    let total_cost := VIP_total + regular_total + discount_total in
    1000 - total_cost = 130 :=
by
  intros VIP_count regular_count discount_count VIP_price regular_price discount_price 
         hVIP_count hregular_count hdiscount_count hVIP_price hregular_price hdiscount_price
  let VIP_total := VIP_count * VIP_price
  let regular_total := regular_count * regular_price
  let discount_total := discount_count * discount_price
  let total_cost := VIP_total + regular_total + discount_total
  rw [hVIP_count, hregular_count, hdiscount_count, hVIP_price, hregular_price, hdiscount_price]
  have : total_cost = 480 + 300 + 90 := by sorry
  rw this
  norm_num
  done

end mrs_wilsborough_change_l679_679844


namespace find_a_l679_679804

theorem find_a (a : ℝ) : 
  (∃ t : ℕ → ℝ, t 2 = 30 ∧ t 3 = a) → a = 2 :=
by
  assume h
  sorry

end find_a_l679_679804


namespace hypotenuse_min_length_l679_679335

theorem hypotenuse_min_length
  (a b l : ℝ)
  (h_area : (1/2) * a * b = 8)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = l)
  (h_min_l : l = 8 + 4 * Real.sqrt 2) :
  Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_min_length_l679_679335


namespace pears_picked_l679_679419

def Jason_pears : ℕ := 46
def Keith_pears : ℕ := 47
def Mike_pears : ℕ := 12
def total_pears : ℕ := 105

theorem pears_picked :
  Jason_pears + Keith_pears + Mike_pears = total_pears :=
by
  exact rfl

end pears_picked_l679_679419


namespace total_turtles_l679_679264

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l679_679264


namespace max_significantly_dissatisfied_participants_l679_679853

def number_participants : ℕ := 500
def problem_conditions_dissatisfied : ℕ := 30
def event_organization_dissatisfied : ℕ := 40
def winners_method_dissatisfied : ℕ := 50

theorem max_significantly_dissatisfied_participants : ∃ S : ℕ, S = 60 ∧ 
    2 * S ≤ (problem_conditions_dissatisfied + event_organization_dissatisfied + winners_method_dissatisfied) :=
by 
  use 60
  split
  · exact rfl
  · simp [problem_conditions_dissatisfied, event_organization_dissatisfied, winners_method_dissatisfied]
  sorry

end max_significantly_dissatisfied_participants_l679_679853


namespace carol_spending_l679_679647

noncomputable def savings (S : ℝ) : Prop :=
∃ (X : ℝ) (stereo_spending television_spending : ℝ), 
  stereo_spending = (1 / 4) * S ∧
  television_spending = X * S ∧
  stereo_spending + television_spending = 0.25 * S ∧
  (stereo_spending - television_spending) / S = 0.25

theorem carol_spending (S : ℝ) : savings S :=
sorry

end carol_spending_l679_679647


namespace false_statement_about_isosceles_right_triangles_l679_679203

-- Definitions of the geometric properties
def isosceles_right_triangle (α β γ : ℝ) : Prop :=
  α = 45 ∧ β = 45 ∧ γ = 90

def regular_polygon (sides : ℕ) (angles : ℕ) (equil : ℕ) : Prop :=
  sides > 2 ∧ angles = equil

def similar_triangles (α1 β1 γ1 α2 β2 γ2 : ℝ) : Prop :=
  (α1 = α2 ∧ β1 = β2 ∧ γ1 = γ2)

-- Given conditions
theorem false_statement_about_isosceles_right_triangles :
  ∀ α β γ : ℝ,
  isosceles_right_triangle α β γ →
  regular_polygon 3 α 90 →
  ¬ regular_polygon 3 45 45 :=
by
  intros α β γ h h₁
  cases h with hα h₂
  cases h₂ with hβ hγ
  sorry

end false_statement_about_isosceles_right_triangles_l679_679203


namespace olive_fraction_fleas_l679_679691

def gertrude_fleas : Nat := 10
def total_fleas : Nat := 40

variable (o m : Nat) (f : ℚ)

-- Define Maud's fleas based on Olive's fleas
def maud_fleas (o : Nat) : Nat := 5 * o
-- Define Olive's fleas based on Gertrude's fleas and the fraction
def olive_fleas (f : ℚ) (g : Nat) : ℚ := f * g

theorem olive_fraction_fleas (f : ℚ) : 
  let o := olive_fleas f gertrude_fleas in
  let m := maud_fleas o in
  gertrude_fleas + o + m = total_fleas -> 
  f = 1 / 2 := 
by
  intro h
  sorry

end olive_fraction_fleas_l679_679691


namespace vic_max_marks_l679_679550

theorem vic_max_marks (M : ℝ) (h : 0.92 * M = 368) : M = 400 := 
sorry

end vic_max_marks_l679_679550


namespace weight_of_2019_is_correct_l679_679375

-- The number of sticks for each digit
def sticks_per_digit (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- For simplicity, we assume other digits are not part of the problem.

-- The total number of sticks needed for a number
def total_sticks (num : List Nat) : Nat :=
  num.foldl (λ acc d => acc + sticks_per_digit d) 0

-- Given a digit 1 weighs 1 kg (comprising 2 sticks)
def stick_weight_kg := 0.5

-- Calculate the weight of the number 2019
def weight_of_number_2019_kg := total_sticks [2, 0, 1, 9] * stick_weight_kg

-- The theorem statement in Lean
theorem weight_of_2019_is_correct :
  weight_of_number_2019_kg = 9.5 :=
by
  sorry

end weight_of_2019_is_correct_l679_679375


namespace amount_decreased_is_5_l679_679040

noncomputable def x : ℕ := 50
noncomputable def equation (x y : ℕ) : Prop := (1 / 5) * x - y = 5

theorem amount_decreased_is_5 : ∃ y : ℕ, equation x y ∧ y = 5 :=
by
  sorry

end amount_decreased_is_5_l679_679040


namespace tank_plastering_cost_proof_l679_679620

/-- 
Given a tank with the following dimensions:
length = 35 meters,
width = 18 meters,
depth = 10 meters.
The cost of plastering per square meter is ₹135.
Prove that the total cost of plastering the walls and bottom of the tank is ₹228,150.
-/
theorem tank_plastering_cost_proof (length width depth cost_per_sq_meter : ℕ)
  (h_length : length = 35)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost_per_sq_meter : cost_per_sq_meter = 135) : 
  (2 * (length * depth) + 2 * (width * depth) + length * width) * cost_per_sq_meter = 228150 := 
by 
  -- The proof is not required as per the problem statement
  sorry

end tank_plastering_cost_proof_l679_679620


namespace unique_x_value_l679_679572

theorem unique_x_value (x : ℝ) (h : x ≠ 0) (h_sqrt : Real.sqrt (5 * x / 7) = x) : x = 5 / 7 :=
by
  sorry

end unique_x_value_l679_679572


namespace mat_weavers_equiv_l679_679871

theorem mat_weavers_equiv {x : ℕ} 
  (h1 : 4 * 1 = 4) 
  (h2 : 16 * (64 / 16) = 64) 
  (h3 : 1 = 64 / (16 * x)) : x = 4 :=
by
  sorry

end mat_weavers_equiv_l679_679871


namespace perpendicular_lines_a_value_l679_679044

theorem perpendicular_lines_a_value :
  ∀ a : ℝ, 
    (∀ x y : ℝ, 2*x + a*y - 7 = 0) → 
    (∀ x y : ℝ, (a-3)*x + y + 4 = 0) → a = 2 :=
by
  sorry

end perpendicular_lines_a_value_l679_679044


namespace find_length_y_l679_679677

def length_y (AO OC DO BO BD y : ℝ) : Prop := 
  AO = 3 ∧ OC = 11 ∧ DO = 3 ∧ BO = 6 ∧ BD = 7 ∧ y = 3 * Real.sqrt 91

theorem find_length_y : length_y 3 11 3 6 7 (3 * Real.sqrt 91) :=
by
  sorry

end find_length_y_l679_679677


namespace arctan_sum_eq_l679_679461

noncomputable def problem (x y : ℝ) (ε : ℤ) : Prop :=
  arctan x + arctan y = arctan ((x + y) / (1 - x * y)) + ε * Real.pi

theorem arctan_sum_eq (x y : ℝ) (h1 : x * y < 1 → ε = 0) 
(h2 : x * y > 1 → x < 0 → ε = -1) 
(h3 : x * y > 1 → x > 0 → ε = 1) : 
problem x y ε := 
  sorry

end arctan_sum_eq_l679_679461


namespace imaginary_part_z_coeff_x2_in_expansion_num_5digit_numbers_no_repeats_eccentricity_hyperbola_l679_679578

-- Condition: Define z
def z : ℂ := ((i - 1)^2 + 4) / (i + 1)

-- Proof: The imaginary part of z
theorem imaginary_part_z : z.im = -3 := by sorry

-- Condition: Define polynomial expansion
def polynomial := (2 * x - 1 / 2) ^ 6

-- Proof: Coefficient of x^2 in the expansion
theorem coeff_x2_in_expansion : polynomial.coeff ⟨2⟩ = 60 := by sorry

-- Condition: Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Proof: Number of 5-digit numbers formed without repeating digits
theorem num_5digit_numbers_no_repeats : count_5digit_numbers digits = 72 := by sorry

-- Condition: Define the hyperbola
structure Hyperbola :=
  (a : ℝ)
  (b : ℝ)
  (angle : ℝ)
  (hp : a > 0)

def hyperbola : Hyperbola := { a := sqrt(3), b := 1, angle := π / 6, hp := by sorry }

-- Proof: Eccentricity of the hyperbola
theorem eccentricity_hyperbola : eccentricity hyperbola = 2 * sqrt(3) / 3 := by sorry

end imaginary_part_z_coeff_x2_in_expansion_num_5digit_numbers_no_repeats_eccentricity_hyperbola_l679_679578


namespace appropriate_sampling_method_l679_679968

noncomputable def school_survey_sampling_method : Prop :=
  let male_students := 500
  let female_students := 500
  let total_students := male_students + female_students
  let sample_size := 100
  let ratio_preserved := male_students / total_students = female_students / total_students
  -- The goal is to show that the appropriate sampling method is the "stratified sampling method."
  stratified_sampling_method_correct total_students sample_size := 
    ratio_preserved → sample_method = "stratified sampling method"

theorem appropriate_sampling_method (male_students female_students sample_size : ℕ) (ratio_preserved : male_students / (male_students + female_students) = female_students / (male_students + female_students)) :
    stratified_sampling_method_correct (male_students + female_students) sample_size :=
by
  sorry

end appropriate_sampling_method_l679_679968


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679482

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679482


namespace find_greatest_number_l679_679947

def numbers := [0.07, -0.41, 0.8, 0.35, -0.9]

theorem find_greatest_number :
  ∃ x ∈ numbers, x > 0.7 ∧ ∀ y ∈ numbers, y > 0.7 → y = 0.8 :=
by
  sorry

end find_greatest_number_l679_679947


namespace sum_binom_even_odd_eq_l679_679092

theorem sum_binom_even_odd_eq (n : ℕ) (h : n > 0) :
  (∑ k in (finset.filter (λ k => 2 * k ≤ n) (finset.range (n+1))), nat.choose n (2 * k)) =
  (∑ k in (finset.filter (λ k => 2 * k + 1 ≤ n) (finset.range (n+1))), nat.choose n (2 * k + 1)) := 
sorry

end sum_binom_even_odd_eq_l679_679092


namespace ratio_of_foci_and_vertices_l679_679429

theorem ratio_of_foci_and_vertices :
  let P := { p : ℝ × ℝ | p.snd = -(p.fst)^2 }
  let V1 : (ℝ × ℝ) := (0, 0)
  let F1 : (ℝ × ℝ) := (0, -1/4)
  let locus_M a b := (a + b) / 2, -((a + b)^2) / 2 - 1
  let Q := { q : ℝ × ℝ | q.snd = -2 * (q.fst)^2 - 1 }
  let V2 : (ℝ × ℝ) := (0, -1)
  let F2 : (ℝ × ℝ) := (0, -9 / 8)
  in
  ∃ A B ∈ P, ∠ A V1 B = 90 ∧ (F1F2.distance / V1V2.distance = 7 / 8)
:=
sorry

end ratio_of_foci_and_vertices_l679_679429


namespace polynomial_divisibility_l679_679379

theorem polynomial_divisibility (f g : Polynomial ℝ)
  (h_f := f = Polynomial.C 1 + Polynomial.X + Polynomial.X^2 * 6 + Polynomial.X^3 * 4 + Polynomial.X^4)
  (h_g := g = Polynomial.C s + Polynomial.X * 5r + Polynomial.X^2 * 10q + Polynomial.X^3 * 10p + Polynomial.X^4 * 5 + Polynomial.X^5)
  (divisible : ∃ q : ℝ, g = f * q):
  (p + q + r) * s = -1.7 :=
sorry

end polynomial_divisibility_l679_679379


namespace continuous_at_x0_l679_679457

variable (x : ℝ) (ε : ℝ)

def f (x : ℝ) : ℝ := -3 * x^2 - 5

def x0 : ℝ := 2

theorem continuous_at_x0 : 
  (0 < ε) → 
  ∃ δ > 0, ∀ x, abs (x - x0) < δ → abs (f x - f x0) < ε := by
sory

end continuous_at_x0_l679_679457


namespace sphere_volume_increase_l679_679534

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let S := 4 * Real.pi * r^2
  let V := (4/3) * Real.pi * r^3
  let S_new := 4 * S
  let R := 2 * r
  let V_new := (4/3) * Real.pi * (R^3)
  in V_new = 8 * V :=
by
  sorry

end sphere_volume_increase_l679_679534


namespace triangle_is_acute_l679_679386

theorem triangle_is_acute (x : ℝ) (h : 2 * x + 3 * x + 4 * x = 180) : 
  4 * (180 / 9 : ℝ) < 90 :=
by
  have h1 : 9 * x = 180, from (by ring) ▸ h,
  have h2 : x = (180 / 9 : ℝ), from (by rw ←h1; rw div_eq_mul_inv; rw mul_comm; exact rfl),
  have h3 : 4 * x = 4 * (180 / 9 : ℝ), from (by rw h2),
  rw h3,
  exact dec_trivial

end triangle_is_acute_l679_679386


namespace least_perimeter_isosceles_triangle_l679_679629

-- Lean statement to formalize the conditions and the proof result
theorem least_perimeter_isosceles_triangle 
  (α : ℝ) (r : ℝ) (A B C : Type) [triangle A B C] 
  (fixed_r : ∀ (triangle : A B C), inradius triangle = r)
  (fixed_α : ∀ (triangle : A B C), angle_BAC triangle = α) :
  ∃ (triangle : A B C), is_isosceles_at_A triangle ∧ ∀ (triangle' : A B C), perimeter triangle ≤ perimeter triangle' := 
sorry

end least_perimeter_isosceles_triangle_l679_679629


namespace sum_of_undefined_values_l679_679194

theorem sum_of_undefined_values :
  (let roots := {x : ℝ | x^2 - 7 * x + 10 = 0} in ∑ x in roots, x) = 7 :=
by
  -- We introduce a variable to hold the set of roots where the quadratic equation is zero.
  let roots := {x : ℝ | x^2 - 7 * x + 10 = 0}
  -- Here we skip the actual proof steps
  sorry

end sum_of_undefined_values_l679_679194


namespace trucks_filled_up_l679_679789

theorem trucks_filled_up (service_cost : ℝ) (cost_per_liter : ℝ) (num_minivans : ℕ) 
  (total_cost : ℝ) (minivan_tank : ℝ) (truck_tank_factor : ℝ) 
  (all_empty : Bool) : ℕ :=
by
  -- Conditions
  have h_service_cost : service_cost = 2.10 := sorry
  have h_cost_per_liter : cost_per_liter = 0.70 := sorry
  have h_num_minivans : num_minivans = 3 := sorry
  have h_total_cost : total_cost = 347.2 := sorry
  have h_minivan_tank : minivan_tank = 65 := sorry
  have h_truck_tank_factor : truck_tank_factor = 1.2 := sorry
  have h_all_empty : all_empty = true := sorry
  
  -- Correct answer to prove
  have answer : ℝ := 2

  -- Prove the number of trucks is 2
  exact answer

end trucks_filled_up_l679_679789


namespace solve_equation_l679_679125

theorem solve_equation (x : ℝ) :
  (4 * x + 1) * (3 * x + 1) * (2 * x + 1) * (x + 1) = 3 * x ^ 4  →
  x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 :=
by
  sorry

end solve_equation_l679_679125


namespace hexagon_equal_angles_construct_hexagon_sides_l679_679516

-- Part (a)
theorem hexagon_equal_angles
  (A B C D E F : Point)
  (hexagon : ConvexHexagon A B C D E F)
  (equal_angles : ∀ i j k l, angle i j k = angle k l i) :
  (dist AB - dist DE = dist EF - dist BC ∧ dist EF - dist BC = dist CD - dist FA) := 
sorry

-- Part (b)
theorem construct_hexagon_sides
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (condition : a1 - a4 = a5 - a2 ∧ a5 - a2 = a3 - a6) :
  ∃ (A B C D E F : Point), 
    hexagon_sides A B C D E F a1 a2 a3 a4 a5 a6 ∧
    equal_angles A B C D E F :=
sorry

end hexagon_equal_angles_construct_hexagon_sides_l679_679516


namespace legacy_earnings_per_hour_l679_679975

-- Define the conditions
def totalFloors : ℕ := 4
def roomsPerFloor : ℕ := 10
def hoursPerRoom : ℕ := 6
def totalEarnings : ℝ := 3600

-- The statement to prove
theorem legacy_earnings_per_hour :
  (totalFloors * roomsPerFloor * hoursPerRoom) = 240 → 
  (totalEarnings / (totalFloors * roomsPerFloor * hoursPerRoom)) = 15 := by
  intros h
  sorry

end legacy_earnings_per_hour_l679_679975


namespace table_relation_l679_679675

theorem table_relation (x y : ℕ) (hx : x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6) :
  (y = 3 ∧ x = 2) ∨ (y = 8 ∧ x = 3) ∨ (y = 15 ∧ x = 4) ∨ (y = 24 ∧ x = 5) ∨ (y = 35 ∧ x = 6) ↔ 
  y = x^2 - x + 2 :=
sorry

end table_relation_l679_679675


namespace area_of_isosceles_right_triangle_l679_679127

-- Definitions based on given conditions.
def is_isosceles_right_triangle (ABC : Triangle) : Prop :=
  ABC.is_right_triangle ∧ ABC.AB = ABC.BC

def point_on_hypotenuse (P : Point) (ABC : Triangle) : Prop :=
  P ∈ segment AC ∧ P ∠ ABC = 45

def segment_AP_CP (P : Point) (A C : Point) : Prop :=
  dist(P, A) = 2 ∧ dist(P, C) = 1

-- The theorem to prove.
theorem area_of_isosceles_right_triangle (ABC : Triangle) (P : Point):
  is_isosceles_right_triangle ABC →
  point_on_hypotenuse P ABC →
  segment_AP_CP P ABC →
  area ABC = 9 / 4 :=
by
  sorry

end area_of_isosceles_right_triangle_l679_679127


namespace solution_set_of_inequality_l679_679139

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true
axiom f_zero_eq : f 0 = 2
axiom f_derivative_ineq : ∀ x : ℝ, f x + (deriv f x) > 1

theorem solution_set_of_inequality : { x : ℝ | e^x * f x > e^x + 1 } = { x | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l679_679139


namespace exists_triangle_with_prime_angles_l679_679075

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

-- Definition of being an angle of a triangle
def is_valid_angle (α : ℕ) : Prop := α > 0 ∧ α < 180

-- Main statement
theorem exists_triangle_with_prime_angles :
  ∃ (α β γ : ℕ), is_prime α ∧ is_prime β ∧ is_prime γ ∧ is_valid_angle α ∧ is_valid_angle β ∧ is_valid_angle γ ∧ α + β + γ = 180 :=
by
  sorry

end exists_triangle_with_prime_angles_l679_679075


namespace a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l679_679526

-- Definitions for the point P and movements
def move (P : ℤ) (flip : Bool) : ℤ :=
  if flip then P + 1 else -P

-- Definitions for probabilities
def probability_of_event (events : ℕ) (successful : ℕ) : ℚ :=
  successful / events

def probability_a3_zero : ℚ :=
  probability_of_event 8 2  -- 2 out of 8 sequences lead to a3 = 0

def probability_a4_one : ℚ :=
  probability_of_event 16 2  -- 2 out of 16 sequences lead to a4 = 1

noncomputable def probability_an_n_minus_3 (n : ℕ) : ℚ :=
  if n < 3 then 0 else (n - 1) / (2 ^ n)

-- Statements to prove
theorem a3_probability_is_one_fourth : probability_a3_zero = 1/4 := by
  sorry

theorem a4_probability_is_one_eighth : probability_a4_one = 1/8 := by
  sorry

theorem an_n_minus_3_probability (n : ℕ) (hn : n ≥ 3) : probability_an_n_minus_3 n = (n - 1) / (2^n) := by
  sorry

end a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l679_679526


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679006

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679006


namespace time_for_trains_to_pass_completely_l679_679183

def length_train := 300 -- length of each train in meters
def speed_train1 := 95 -- speed of train 1 in km/h
def speed_train2 := 80 -- speed of train 2 in km/h

def relative_speed_kmh := speed_train1 + speed_train2 -- relative speed in km/h
def relative_speed_ms := (relative_speed_kmh * 1000) / 3600 -- convert km/h to m/s

def combined_length := length_train * 2 -- combined length of both trains, since they have the same length

noncomputable def time_to_pass := combined_length / relative_speed_ms -- time in seconds

theorem time_for_trains_to_pass_completely : 
  abs (time_to_pass - 12.34) < 0.01 := 
sorry

end time_for_trains_to_pass_completely_l679_679183


namespace sixth_selected_tv_l679_679294

def random_number_table : list (list ℕ) := [
  [7816, 6572, 0802, 6314, 0702, 4369, 9728, 0198],
  [3204, 9234, 4935, 8200, 3623, 4869, 6938, 7481]
]

def selected_numbers (table: list (list ℕ)) : list ℕ :=
  table.head.drop(3).filter (λ n, n < 30)

theorem sixth_selected_tv : selected_numbers random_number_table !! 5 = some 03 := by
  -- Starting from the first row, fourth column, reading left to right
  -- the numbers selected are 20, 26, 24, 19, 23, 03
  -- So the 6th selected number should be 03
  sorry

end sixth_selected_tv_l679_679294


namespace proof_goals_l679_679427

-- Definitions and assumptions
variables {A B C E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited E]

-- Assumptions about the geometry
variables (AB AC BC AE : ℝ) -- Assume side lengths and segments as real numbers

-- Let ABC be an isosceles triangle with AB = AC
def is_isosceles_triangle := AB = AC

-- Assume circle ω is circumscribed around triangle ABC
-- and tangent at A intersects BC at point E
def is_circumscribed_and_tangent (ω : Type) [Inhabited ω]
  (circum_circle : A → B → C → ω) (tangent : ω → A → E) := sorry

-- Proof goals
theorem proof_goals :
  (is_isosceles_triangle AB AC) →
  (∃ ω, is_circumscribed_and_tangent ω (λ A B C, ω) (λ ω A, E)) →
  (BE = EC) ∧ 
  (AE bisects ∠BAC) ∧
  (∠EAB = ∠ECB) :=
sorry

end proof_goals_l679_679427


namespace scalene_triangle_count_l679_679525

-- Define the condition for being a scalene triangle with perimeter less than 20
def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b + c < 20 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- The main theorem: there are exactly 9 valid scalene triangles
theorem scalene_triangle_count : {t : ℕ × ℕ × ℕ // is_valid_scalene_triangle t.1 t.2.1 t.2.2}.to_finset.card = 9 :=
by sorry

end scalene_triangle_count_l679_679525


namespace fish_per_person_l679_679813

theorem fish_per_person (eyes_per_fish : ℕ) (fish_caught : ℕ) (total_eyes : ℕ) (dog_eyes : ℕ) (oomyapeck_eyes : ℕ) (n_people : ℕ) :
  total_eyes = oomyapeck_eyes + dog_eyes →
  total_eyes = fish_caught * eyes_per_fish →
  n_people = 3 →
  oomyapeck_eyes = 22 →
  dog_eyes = 2 →
  eyes_per_fish = 2 →
  fish_caught / n_people = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end fish_per_person_l679_679813


namespace minimize_distance_sum_l679_679697

open Classical
open Real
noncomputable theory

/-- Let e be a line in space, and let A and B be points not lying on e. Then, the point P on e
which minimizes the sum of the distances PA and PB is the intersection of e with 
the line segment joining A and the reflection of B about e. -/
theorem minimize_distance_sum (e : ℝ → ℝ × ℝ × ℝ) (A B : ℝ × ℝ × ℝ) :
  (¬ ∃ t, e t = A) ∧ (¬ ∃ t, e t = B) →
  ∃ P, P ∈ e ∧ (∀ Q ∈ e, dist P A + dist P B ≤ dist Q A + dist Q B) :=
sorry

end minimize_distance_sum_l679_679697


namespace polynomial_division_remainder_l679_679192

-- Define the dividend polynomial
def dividend : Polynomial ℝ := (2:ℝ) * X^5 + (11:ℝ) * X^4 - (48:ℝ) * X^3 - (60:ℝ) * X^2 + (20:ℝ) * X + (50:ℝ)

-- Define the divisor polynomial
def divisor : Polynomial ℝ := X^3 + (7:ℝ) * X^2 + (4:ℝ)

-- Define the expected remainder polynomial
def expected_remainder : Polynomial ℝ := -(27:ℝ) * X^3 - (68:ℝ) * X^2 + (32:ℝ) * X + (50:ℝ)

-- Statement of the proof
theorem polynomial_division_remainder :
  (dividend % divisor) = expected_remainder :=
by sorry

end polynomial_division_remainder_l679_679192


namespace correct_standardized_statement_l679_679563

-- Define and state the conditions as Lean 4 definitions and propositions
structure GeometricStatement :=
  (description : String)
  (is_standardized : Prop)

def optionA : GeometricStatement := {
  description := "Line a and b intersect at point m",
  is_standardized := False -- due to use of lowercase 'm'
}

def optionB : GeometricStatement := {
  description := "Extend line AB",
  is_standardized := False -- since a line cannot be further extended
}

def optionC : GeometricStatement := {
  description := "Extend ray AO (where O is the endpoint) in the opposite direction",
  is_standardized := False -- incorrect definition of ray extension
}

def optionD : GeometricStatement := {
  description := "Extend line segment AB to C such that BC=AB",
  is_standardized := True -- correct by geometric principles
}

-- The theorem stating that option D is the correct and standardized statement
theorem correct_standardized_statement : optionD.is_standardized = True ∧
                                         optionA.is_standardized = False ∧
                                         optionB.is_standardized = False ∧
                                         optionC.is_standardized = False :=
  by sorry

end correct_standardized_statement_l679_679563


namespace find_f_l679_679662

namespace DiannaProblem

theorem find_f :
  ∀ (a b c d e f : ℤ), 
    a = 1 → b = 2 → c = 3 → d = 4 → e = 5 →
    (a - (b - (c - (d + (e - f)))) = a - b - c - d + e - f) →
    f = 2 :=
by
  intros a b c d e f ha hb hc hd he h
  rw [ha, hb, hc, hd, he] at h
  calc
    1 - (2 - (3 - (4 + (5 - f)))) = 1 - 2 + 3 - 4 + 5 - f : by linarith
    _ = -3 - f : by linarith
    sorry

end find_f_l679_679662


namespace coefficient_x3_l679_679807

theorem coefficient_x3 (a : ℝ) : 
  (let f := (ax + 1) * (x + (1/x)) ^ 6 in 
  term_coefficient(f, 3)) = 30 → a = 2 := 
begin
  sorry
end

end coefficient_x3_l679_679807


namespace complement_U_B_l679_679362

variable (x : ℝ)

def A : Set ℝ := {1, 3, x}
def B : Set ℝ := {1, x^2}
def U : Set ℝ := A ∪ B
def complementU (B : Set ℝ) : Set ℝ := (U \ B)

theorem complement_U_B (B ∪ complementU B = A) : 
  x = 0 ∧ complementU B = {3} ∨ 
  x = sqrt 3 ∧ complementU B = {sqrt 3} ∨ 
  x = -sqrt 3 ∧ complementU B = {-sqrt 3} :=
sorry

end complement_U_B_l679_679362


namespace proof_equivalent_l679_679574

variables {α : Type*} [Field α]

theorem proof_equivalent (a b c d e f : α)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1 :=
by sorry

end proof_equivalent_l679_679574


namespace exists_k_single_real_root_l679_679095

noncomputable def cubic_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d: ℝ, a ≠ 0 ∧ ∀ x, f(x) = a*x^3 + b*x^2 + c*x + d

theorem exists_k_single_real_root (f : ℝ → ℝ) (h : cubic_polynomial f) :
  ∃ k : ℕ, k > 0 ∧ (∑ i in Finset.range k, (λ x, f (x + i)) = λ F x : ℝ, F(x)) has exactly one real root :=
sorry

end exists_k_single_real_root_l679_679095


namespace find_a_l679_679805

theorem find_a (a : ℝ) : 
  (∃ t : ℕ → ℝ, t 2 = 30 ∧ t 3 = a) → a = 2 :=
by
  assume h
  sorry

end find_a_l679_679805


namespace fruit_prices_and_max_m_l679_679586

-- Definitions of conditions:
theorem fruit_prices_and_max_m :
  -- Prices from the batch purchases
  (a b : ℝ) (ha : 60 * a + 40 * b = 1520) (hb : 30 * a + 50 * b = 1360) 
  -- Further conditions for the third purchase
  (x : ℝ) (hx : (80 ≤ x ∧ x ≤ 200))
  -- Proof about prices of type A and B fruits and max value of m
  (hpricea : a = 12) (hpriceb : b = 20)
  -- Resulting max m for conditions to hold
  (hmax_m : ∀ (m : ℝ), (0 < m ∧ m ≤ 22) → (-5 * x - 35 * m + 2000 ≥ 800)) :
  a = 12 ∧ b = 20 ∧ ∃ (m : ℕ), m = 22 := sorry

end fruit_prices_and_max_m_l679_679586


namespace mobile_phone_numbers_count_l679_679256

/-- 
  Prove that the number of 4-digit sequences formed using the digits 1, 3, and 5,
  where each digit appears at least once, is equal to 54.
-/
theorem mobile_phone_numbers_count :
  let digits := {1, 3, 5}
  let seqs := {seqs : (Fin 4 → {1, 3, 5}) | (Set.card (Set.image seqs Set.univ) = 3).toFin }
  (seqs.card = 54) :=
sorry

end mobile_phone_numbers_count_l679_679256


namespace calculate_total_feet_in_garden_l679_679168

-- Define the entities in the problem
def dogs := 6
def feet_per_dog := 4

def ducks := 2
def feet_per_duck := 2

-- Define the total number of feet in the garden
def total_feet_in_garden : Nat :=
  (dogs * feet_per_dog) + (ducks * feet_per_duck)

-- Theorem to state the total number of feet in the garden
theorem calculate_total_feet_in_garden :
  total_feet_in_garden = 28 :=
by
  sorry

end calculate_total_feet_in_garden_l679_679168


namespace Donovan_Mitchell_goal_average_l679_679293

theorem Donovan_Mitchell_goal_average 
  (current_avg_pg : ℕ)     -- Donovan's current average points per game.
  (played_games : ℕ)       -- Number of games played so far.
  (required_avg_pg : ℕ)    -- Required average points per game in remaining games.
  (total_games : ℕ)        -- Total number of games in the season.
  (goal_avg_pg : ℕ)        -- Goal average points per game for the entire season.
  (H1 : current_avg_pg = 26)
  (H2 : played_games = 15)
  (H3 : required_avg_pg = 42)
  (H4 : total_games = 20) :
  goal_avg_pg = 30 :=
by
  sorry

end Donovan_Mitchell_goal_average_l679_679293


namespace point_probability_in_cone_l679_679351

noncomputable def volume_of_cone (S : ℝ) (h : ℝ) : ℝ :=
  (1/3) * S * h

theorem point_probability_in_cone (P M : ℝ) (S_ABC : ℝ) (h_P h_M : ℝ)
  (h_volume_condition : volume_of_cone S_ABC h_P ≤ volume_of_cone S_ABC h_M / 3) :
  (1 - (2 / 3) ^ 3) = 19 / 27 :=
by
  sorry

end point_probability_in_cone_l679_679351


namespace parallelepiped_volume_l679_679897

open Real

noncomputable def volume_parallelepiped
  (a b : ℝ) (angle : ℝ) (S : ℝ) (sin_30 : angle = π / 6) : ℝ :=
  let h := S / (2 * (a + b))
  let base_area := (a * b * sin (π / 6)) / 2
  base_area * h

theorem parallelepiped_volume 
  (a b : ℝ) (S : ℝ) (h : S ≠ 0 ∧ a > 0 ∧ b > 0) :
  volume_parallelepiped a b (π / 6) S (rfl) = (a * b * S) / (4 * (a + b)) :=
by
  sorry

end parallelepiped_volume_l679_679897


namespace pq_area_l679_679839

theorem pq_area (A B C : ℝ × ℝ) (P Q R S T : ℝ × ℝ) 
    (h_area_ABC : 1/2 * (C.1 - B.1) * (A.2 - B.2) = 60)
    (h_P : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (h_Q : Q = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
    (h_R : R = ((2 * B.1 + C.1) / 3, (2 * B.2 + C.2) / 3))
    (h_S : S = ((B.1 + 2 * C.1) / 3, (B.2 + 2 * C.2) / 3))
    (h_T : T = ((3 * A.1 + 8 * C.1) / 21, 7 * (A.2) / 7)) :
    let PQ := ((Q.1 - P.1) * (Q.1 - P.1) + (Q.2 - P.2) * (Q.2 - P.2)).sqrt in
    let height := ((P.2 - T.2).abs) in
    1/2 * PQ * height = 75 / 7
  :=
sorry

end pq_area_l679_679839


namespace selling_price_l679_679603

/-- The cost price of the radio -/
def CP : ℝ := 2400

/-- The loss percentage -/
def loss_percentage : ℝ := 12.5

/-- The selling price (SP) of the radio after a loss of 12.5% -/
def SP : ℝ := CP - (loss_percentage / 100) * CP

theorem selling_price (h : SP = 2100) : 
  SP = 2100 := 
sorry

end selling_price_l679_679603


namespace total_time_of_flight_l679_679223

variables {V_0 g t t_1 H : ℝ}  -- Define variables

-- Define conditions
def initial_condition (V_0 g t_1 H : ℝ) : Prop :=
H = (1/2) * g * t_1^2

def return_condition (V_0 g t : ℝ) : Prop :=
t = 2 * (V_0 / g)

theorem total_time_of_flight
  (V_0 g : ℝ)
  (h1 : initial_condition V_0 g (V_0 / g) (1/2 * g * (V_0 / g)^2))
  : return_condition V_0 g (2 * V_0 / g) :=
by
  sorry

end total_time_of_flight_l679_679223


namespace correct_quadratic_equation_none_of_these_l679_679910

theorem correct_quadratic_equation_none_of_these :
  ¬(∃ (a b c : ℝ), 
     (x = 5 ∨ x = 1) ∧ (x = -6 ∨ x = -4) ∧
     (∀ x, a * x^2 + b * x + c = 0) ∧
     (a = 1) ∧ 
     (b = -6) ∧ 
     (c = 24) ∧ 
     (x^2 - 9x + 5 = 0 ∨
      x^2 + 9x + 6 = 0 ∨
      x^2 - 10x + 5 = 0 ∨
      x^2 + 10x + 24 = 0 ∨
      x^2 - 10x + 6 = 0)) :=
by
  sorry

end correct_quadratic_equation_none_of_these_l679_679910


namespace polynomial_evaluation_eq_zero_l679_679559

theorem polynomial_evaluation_eq_zero (x : ℝ) (h : x = 2) : x^2 - 3 * x + 2 = 0 :=
by 
  rw [h]
  norm_num
  sorry

end polynomial_evaluation_eq_zero_l679_679559


namespace number_of_divisible_factorials_l679_679327

theorem number_of_divisible_factorials:
  ∃ (count : ℕ), count = 36 ∧ ∀ n, 1 ≤ n ∧ n ≤ 50 → (∃ k : ℕ, n! = k * (n * (n + 1)) / 2) ↔ n ≤ n - 14 :=
sorry

end number_of_divisible_factorials_l679_679327


namespace f_sum_zero_l679_679288

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero_l679_679288


namespace trajectory_line_segment_l679_679703

noncomputable theory
open_locale classical

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def is_trajectory (M F1 F2 : Point) :=
  distance F1 F2 = 8 ∧ distance M F1 + distance M F2 = 8

theorem trajectory_line_segment (M F1 F2 : Point) :
  is_trajectory M F1 F2 → M = F1 ∨ M = F2 :=
by
  intros h,
  sorry

end trajectory_line_segment_l679_679703


namespace find_original_number_l679_679571

theorem find_original_number
  (x : ℤ)
  (h : 3 * (2 * x + 5) = 123) :
  x = 18 := 
sorry

end find_original_number_l679_679571


namespace distance_between_Sasha_and_Kolya_l679_679468

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679468


namespace total_turtles_taken_l679_679261

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l679_679261


namespace intervals_of_monotonicity_range_of_a_l679_679097

noncomputable def f (x : ℝ) : ℝ := (x + 2) * Real.exp x

variable (a : ℝ)

theorem intervals_of_monotonicity : 
  (∀ x < -3, (f x)' < 0) ∧ (∀ x > -3, (f x)' > 0) := 
  sorry

theorem range_of_a (h : ∀ x ≥ 0, (f x - Real.exp x) / (a * x + 1) ≥ 1) : 
  0 ≤ a ∧ a ≤ 2 :=
  sorry

end intervals_of_monotonicity_range_of_a_l679_679097


namespace parabola_point_distance_l679_679164

theorem parabola_point_distance (x y : ℝ) (h : y^2 = 2 * x) (d : ℝ) (focus_x : ℝ) (focus_y : ℝ) :
    focus_x = 1/2 → focus_y = 0 → d = 3 →
    (x + 1/2 = d) → x = 5/2 :=
by
  intros h_focus_x h_focus_y h_d h_dist
  sorry

end parabola_point_distance_l679_679164


namespace log_inequality_l679_679119

theorem log_inequality (a b c : ℝ) (h : 2 ≤ a ∧ 2 ≤ b ∧ 2 ≤ c) :
  log (a^2) / log (b + c) + log (b^2) / log (c + a) + log (c^2) / log (a + b) ≥ 3 := sorry

end log_inequality_l679_679119


namespace symmetric_line_equation_l679_679773

theorem symmetric_line_equation {l : ℝ} (h1 : ∀ x y : ℝ, x + y - 1 = 0 → (-x) - y + 1 = l) : l = 0 :=
by
  sorry

end symmetric_line_equation_l679_679773


namespace count_three_digit_integers_with_tens_7_divisible_by_25_l679_679759

theorem count_three_digit_integers_with_tens_7_divisible_by_25 :
  ∃ n, n = 33 ∧ ∃ k1 k2 : ℕ, 175 = 25 * k1 ∧ 975 = 25 * k2 ∧ (k2 - k1 + 1 = n) :=
by
  sorry

end count_three_digit_integers_with_tens_7_divisible_by_25_l679_679759


namespace option_b_correct_option_a_incorrect_option_c_incorrect_option_d_incorrect_l679_679200

theorem option_b_correct : (sqrt 27 / sqrt 3 = 3) :=
by sorry

theorem option_a_incorrect : (2 * sqrt 3 + 4 * sqrt 2 ≠ 6 * sqrt 5) :=
by sorry

theorem option_c_incorrect : (3 * sqrt 3 + 3 * sqrt 2 ≠ 3 * sqrt 6) :=
by sorry

theorem option_d_incorrect : (sqrt ((-5: ℝ)^2) ≠ -5) :=
by sorry

end option_b_correct_option_a_incorrect_option_c_incorrect_option_d_incorrect_l679_679200


namespace expenditure_to_savings_ratio_l679_679529

-- Definitions for expenditures, savings, and income
variables (E S : ℝ)  -- original expenditure and savings
variable (I : ℝ)  -- original income such that I = E + S

-- Conditions given in the problem
axiom (income_relation : I = E + S)
axiom (new_income : ℝ) (new_income_val : new_income = 1.15 * I)
axiom (new_savings : ℝ) (new_savings_val : new_savings = 1.06 * S)
axiom (new_expenditure : ℝ) (new_expenditure_val : new_expenditure = 1.21 * E)

-- New income is also the sum of the new expenditure and new savings
axiom (new_income_relation : new_income = new_expenditure + new_savings)

-- The goal is to prove the ratio of expenditure to savings
theorem expenditure_to_savings_ratio : E = 1.5 * S :=
by 
  -- Proof will be filled in here
  sorry

end expenditure_to_savings_ratio_l679_679529


namespace sum_of_squares_mod_13_l679_679942

theorem sum_of_squares_mod_13 : 
  (∑ i in Finset.range 13, i^2) % 13 = 0 :=
by
  sorry

end sum_of_squares_mod_13_l679_679942


namespace select_n_integers_divisible_by_n_l679_679695

theorem select_n_integers_divisible_by_n (n : ℕ) (a : ℕ → ℤ) (h : ∀ i : ℕ, i < 2 * n - 1 → a i < n) :
  ∃ S : Finset ℕ, S.card = n ∧ (∑ i in S, a i) % n = 0 :=
sorry

end select_n_integers_divisible_by_n_l679_679695


namespace area_CMN_l679_679066

-- Definitions for the geometrical entities.
structure Square (A B C D : Type) (side : ℝ) :=
(area : ℝ)
(angle_eq : ∀ (x y z : Type), ∠ x y z = 90)

structure Triangle (C M N : Type) (side : ℝ) :=
(is_equilateral : (side = dist C M ∧ side = dist M N ∧ side = dist N C))

-- Define given conditions.
axiom ABCD_area : ∀ (A B C D : Type) (ABCD: Square A B C D 1), ABCD.area = 1
axiom CMN_eq_triangle : ∀ (C M N : Type) (CMN: Triangle C M N (dist C M)), CMN.is_equilateral

-- Mathematical proof statements.
theorem area_CMN: 
  ∀ (A B C D M N : Type) 
  (ABCD : Square A B C D 1)
  (CMN : Triangle C M N (dist C M)), 
  ABCD_area A B C D ABCD → 
  CMN_eq_triangle C M N CMN → 
  (Triangle.area CMN) = 2 * sqrt 3 - 3 :=
by sorry

end area_CMN_l679_679066


namespace num_ordered_pairs_l679_679967

theorem num_ordered_pairs :
  {n : ℕ // ∃ s : finset (ℤ × ℤ), (∀ p ∈ s, (p.1 ∈ { -1, 0, 1, 2 }) ∧ (p.2 ∈ { -1, 0, 1, 2 }) ∧ 
  (∃ x : ℝ, x ^ 2 * (p.1:ℝ) + 2 * x + (p.2:ℝ) = 0)) ∧ s.card = n} = ⟨13, _⟩ := sorry

end num_ordered_pairs_l679_679967


namespace min_edge_disjoint_cycles_l679_679210

noncomputable def minEdgesForDisjointCycles (n : ℕ) (h : n ≥ 6) : ℕ := 3 * (n - 2)

theorem min_edge_disjoint_cycles (n : ℕ) (h : n ≥ 6) : minEdgesForDisjointCycles n h = 3 * (n - 2) := 
by
  sorry

end min_edge_disjoint_cycles_l679_679210


namespace total_amount_is_175_l679_679994

noncomputable def calc_total_amount (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
x + y + z

theorem total_amount_is_175 (x y z : ℝ) 
  (h1 : y = 0.45 * x)
  (h2 : z = 0.30 * x)
  (h3 : y = 45) :
  calc_total_amount x y z = 175 :=
by
  -- sorry to skip the proof
  sorry

end total_amount_is_175_l679_679994


namespace scientific_notation_l679_679596

theorem scientific_notation {a : ℝ} (h : a = 3 * 10^(-7)) : a = 0.0000003 := 
sorry

end scientific_notation_l679_679596


namespace problem1_problem2_problem3_problem4_l679_679870

theorem problem1 (x : ℝ) : x^2 - 2 * x + 1 = 0 ↔ x = 1 := 
by sorry

theorem problem2 (x : ℝ) : x^2 + 2 * x - 3 = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem problem3 (x : ℝ) : 2 * x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 33) / 4 ∨ x = (-5 - Real.sqrt 33) / 4 :=
by sorry

theorem problem4 (x : ℝ) : 2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 :=
by sorry

end problem1_problem2_problem3_problem4_l679_679870


namespace bottles_per_day_production_l679_679225

constant CaseHolds : Nat := 5
constant DailyCases : Nat := 12000

theorem bottles_per_day_production : CaseHolds * DailyCases = 60000 := by
  sorry

end bottles_per_day_production_l679_679225


namespace height_of_flagpole_l679_679590

/-- Define the known conditions and question for the problem. -/
variables 
  (shadow_length_flagpole : ℕ)
  (shadow_length_building : ℕ)
  (height_building : ℕ)
  (height_flagpole : ℕ)
  (ratio_eq: (height_flagpole : ℝ) / (shadow_length_flagpole : ℝ) = (height_building : ℝ) / (shadow_length_building : ℝ))

/-- Given conditions from the problem. -/
def conditions : Prop :=
  shadow_length_flagpole = 45 ∧
  shadow_length_building = 70 ∧
  height_building = 28

/-- The statement we need to prove, given the conditions, the height of the flagpole is 18 meters. -/
theorem height_of_flagpole (h_conditions: conditions) (h_ratio: ratio_eq) : height_flagpole = 18 :=
sorry

end height_of_flagpole_l679_679590


namespace generalized_convex_inequality_l679_679830

variables (I : set ℝ) (f : ℝ → ℝ) (m r : ℝ) (n : ℕ) (x : ℕ → ℝ) (p : ℕ → ℝ)

noncomputable def M_nm (f : ℝ → ℝ) (x : ℕ → ℝ) (p : ℕ → ℝ) (m : ℝ) : ℝ :=
  (∑ i in set.range n, p i * f (x i) ^ m) ^ (1 / m)

noncomputable def M_nr (x : ℕ → ℝ) (p : ℕ → ℝ) (r : ℝ) : ℝ :=
  (∑ i in set.range n, p i * x i ^ r) ^ (1 / r)

theorem generalized_convex_inequality 
  (h_f_convex : ∀ (x₁ x₂ ∈ I), ∀ (t ∈ I), f (((1 - t) * x₁ + t * x₂) ^ m) ≤ ((1 - t) * f (x₁ ^ m) + t * f (x₂ ^ m)) ^ (1 / m))
  (h_p_pos : ∀ i, p i > 0) 
  (h_x_in_I : ∀ i, x i ∈ I) :
  (M_nm f x p m) ≥ f (M_nr x p r) :=
sorry

end generalized_convex_inequality_l679_679830


namespace students_scoring_at_least_80_approx_l679_679781

noncomputable def number_of_students_scoring_at_least_80 (total_students : ℕ) :=
  if total_students = 1600 then 1400 else 0

theorem students_scoring_at_least_80_approx :
  ∀ (total_students : ℕ) (mean : ℝ) (std_dev : ℝ),
    total_students = 1600 →
    mean = 100 →
    std_dev > 0 →
    (∀ (X : ℝ), X > 80 ∧ X < 120 → X ∼ normal_distribution(mean, std_dev)) →
    number_of_students_scoring_at_least_80 total_students = 1400 :=
by
  intros total_students mean std_dev h1 h2 h3 h4
  sorry

end students_scoring_at_least_80_approx_l679_679781


namespace arithmetic_seq_log_l679_679149

theorem arithmetic_seq_log (a b : ℝ) (n : ℕ)
  (h1 : log (a^4 * b^5) = log a^4 + log b^5)
  (h2 : log (a^7 * b^11) = log a^7 + log b^11)
  (h3 : log (a^10 * b^15) = log a^10 + log b^15) :
  (log a^4 + log b^5) + (15-1) * (((log a^7 + log b^11) - (log a^4 + log b^5)) / 2) = log (b^181) :=
begin
  sorry
end

end arithmetic_seq_log_l679_679149


namespace euler_formula_quadrant_l679_679667

theorem euler_formula_quadrant (x : ℝ) (hx : x = -2) : 
  let z := complex.exp (complex.I * x) in
  z.re < 0 ∧ z.im < 0 :=
by
  -- Given Euler’s formula
  have h1 : complex.exp (complex.I * x) = complex.cos x + complex.I * complex.sin x,
  { rw complex.exp_mul_I },
  -- Use the information about the range of cosine and sine for angle -2 radians
  have h2 : complex.cos x < 0 ∧ complex.sin x < 0,
  { rw hx,
    have h_cos : complex.cos (-2) < 0, 
    { norm_num,
      -- This assumes knowledge of the cosine function properties, 
      -- but in a formal proof you'd refer to the range interval -pi <= x <= pi
      exact real.cos_neg (real.pi_div_two_le_neg_of_eq_neg_pi 1) },
    have h_sin : complex.sin (-2) < 0, 
    { norm_num,
      -- Using properties of the sine function, similar assumptions apply
      exact real.sin_neg (by linarith [real.pi_neg_le_neg x 1]) },
    exact ⟨h_cos, h_sin⟩ },
  -- Combine to show the complex number falls in the third quadrant
  rw h1,
  exact h2

end euler_formula_quadrant_l679_679667


namespace find_bicycle_speed_l679_679911

def distanceAB := 40 -- Distance from A to B in km
def speed_walk := 6 -- Speed of the walking tourist in km/h
def distance_ahead := 5 -- Distance by which the second tourist is ahead initially in km
def speed_car := 24 -- Speed of the car in km/h
def meeting_time := 2 -- Time after departure when they meet in hours

theorem find_bicycle_speed (v : ℝ) : 
  (distanceAB = 40 ∧ speed_walk = 6 ∧ distance_ahead = 5 ∧ speed_car = 24 ∧ meeting_time = 2) →
  (v = 9) :=
by 
sorry

end find_bicycle_speed_l679_679911


namespace fifth_term_series_sum_l679_679846

/-- Define the sequence aₙ -/
def a (n : ℕ) : ℝ :=
  1 / (2 * n - 1) / (2 * n + 1)

theorem fifth_term :
  a 5 = 1 / (9 * 11) := 
by
  sorry

theorem series_sum :
  ∑ i in Finset.range 2023, a (i + 1) = 2023 / 4047 :=
by
  sorry

end fifth_term_series_sum_l679_679846


namespace range_of_m_min_value_of_4a_plus_7b_l679_679722

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x = sqrt (abs (x + 2) + abs (x - 4) - m)) : 
  ∀ x : ℝ, 0 ≤ abs (x + 2) + abs (x - 4) - m → m ≤ 6 :=
begin
  sorry
end

theorem min_value_of_4a_plus_7b {a b : ℝ} 
  (n : ℝ) (h₁ : n = 6) 
  (h₂ : 4 / (a + 5 * b) + 1 / (3 * a + 2 * b) = n) 
  (h₃ : 0 < a) (h₄ : 0 < b) : 
  4 * a + 7 * b ≥ 3 / 2 :=
begin
  sorry
end

end range_of_m_min_value_of_4a_plus_7b_l679_679722


namespace sum_blue_equals_sum_red_of_intersecting_squares_l679_679177

theorem sum_blue_equals_sum_red_of_intersecting_squares
  (identical_squares : Square)
  (octagon : Octagon)
  (blue_sides_of_squares : ∀ {s : Square}, Side s → Color)
  (red_sides_of_squares : ∀ {s : Square}, Side s → Color)
  (intersection : intersect identical_squares identical_squares = octagon)
  (color_blue : ∀ b_side: Side identical_squares, blue_sides_of_squares b_side = blue)
  (color_red : ∀ r_side: Side identical_squares, red_sides_of_squares r_side = red) :
  sum_of_sides (blue_sides_of_squares) octagon = sum_of_sides (red_sides_of_squares) octagon :=
sorry

end sum_blue_equals_sum_red_of_intersecting_squares_l679_679177


namespace no_four_chips_form_square_after_removal_l679_679632

-- Define the initial setup of 20 chips arranged in the shape of a cross
-- Define the concept of forming a square and the positions of 6 specific chips to be removed
def chips_arrangement : Type := sorry  -- the exact definition would be a representation of the 20 chips
def forms_square (chips : chips_arrangement) : Prop := sorry  -- checks if 4 chips form a square
def removed_positions (chips : chips_arrangement) : Prop := sorry  -- checks if the 6 chips labeled e are removed

-- The theorem stating the desired property
theorem no_four_chips_form_square_after_removal (chips : chips_arrangement) (removed_chips : removed_positions chips) :
  ¬ ∃ (c1 c2 c3 c4 : chips_arrangement), forms_square (c1, c2, c3, c4) :=
sorry

end no_four_chips_form_square_after_removal_l679_679632


namespace divides_prime_factors_l679_679828

theorem divides_prime_factors (a b : ℕ) (p : ℕ → ℕ → Prop) (k l : ℕ → ℕ) (n : ℕ) : 
  (a ∣ b) ↔ (∀ i : ℕ, i < n → k i ≤ l i) :=
by
  sorry

end divides_prime_factors_l679_679828


namespace train_scheduled_speed_l679_679624

theorem train_scheduled_speed (a v : ℝ) (hv : 0 < v)
  (h1 : a / v - a / (v + 5) = 1 / 3)
  (h2 : a / (v - 5) - a / v = 5 / 12) : v = 45 :=
by
  sorry

end train_scheduled_speed_l679_679624


namespace running_time_l679_679079

variable (t : ℝ)
variable (v_j v_p d : ℝ)

-- Given conditions
variable (v_j : ℝ := 0.133333333333)  -- Joe's speed
variable (v_p : ℝ := 0.0666666666665) -- Pete's speed
variable (d : ℝ := 16)                -- Distance between them after t minutes

theorem running_time (h : v_j + v_p = 0.2 * t) : t = 80 :=
by
  -- Distance covered by Joe and Pete running in opposite directions
  have h1 : v_j * t + v_p * t = d := by sorry
  -- Given combined speeds
  have h2 : v_j + v_p = 0.2 := by sorry
  -- Using the equation to solve for time t
  exact sorry

end running_time_l679_679079


namespace sector_area_is_80pi_l679_679774

noncomputable def sectorArea (θ r : ℝ) : ℝ := 
  1 / 2 * θ * r^2

theorem sector_area_is_80pi :
  sectorArea (2 * Real.pi / 5) 20 = 80 * Real.pi :=
by
  sorry

end sector_area_is_80pi_l679_679774


namespace quartic_polynomial_roots_l679_679306

noncomputable def quartic_polynomial : polynomial ℚ :=
  polynomial.X^4 - 10 * polynomial.X^3 + 25 * polynomial.X^2 - 18 * polynomial.X - 12

theorem quartic_polynomial_roots :
  (polynomial.aeval (3 + real.sqrt 5 : ℚ) quartic_polynomial = 0) ∧
  (polynomial.aeval (3 - real.sqrt 5 : ℚ) quartic_polynomial = 0) ∧
  (polynomial.aeval (2 + real.sqrt 7 : ℚ) quartic_polynomial = 0) ∧
  (polynomial.aeval (2 - real.sqrt 7 : ℚ) quartic_polynomial = 0) :=
by
  sorry

end quartic_polynomial_roots_l679_679306


namespace trail_mix_total_weight_l679_679271

def peanuts : ℝ := 0.17
def chocolate_chips : ℝ := 0.17
def raisins : ℝ := 0.08

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.42 :=
by
  -- The proof would go here
  sorry

end trail_mix_total_weight_l679_679271


namespace correct_function_l679_679255

-- Definitions for each function
def funcA (x : ℝ) := 2 * x
def funcB (x : ℝ) := Real.sin x
def funcC (x : ℝ) := Real.log x / Real.log 2
def funcD (x : ℝ) := x * |x|

-- Properties of odd and increasing function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f x < f y

-- Statement to identify the correct function
theorem correct_function : 
  (is_odd funcD ∧ is_increasing funcD) ∧
  ¬ (is_odd funcA ∧ is_increasing funcA) ∧
  ¬ (is_odd funcB ∧ is_increasing funcB) ∧
  ¬ (is_odd funcC ∧ is_increasing funcC) := 
by
  sorry

end correct_function_l679_679255


namespace find_chord_points_l679_679289

/-
Define a parabola and check if the points given form a chord that intersects 
the point (8,4) in the ratio 1:4.
-/

def parabola (P : ℝ × ℝ) : Prop :=
  P.snd^2 = 4 * P.fst

def divides_in_ratio (C A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  (A.fst * n + B.fst * m = C.fst * (m + n)) ∧ 
  (A.snd * n + B.snd * m = C.snd * (m + n))

theorem find_chord_points :
  ∃ (P1 P2 : ℝ × ℝ),
  parabola P1 ∧
  parabola P2 ∧
  divides_in_ratio (8, 4) P1 P2 1 4 ∧ 
  ((P1 = (1, 2) ∧ P2 = (36, 12)) ∨ (P1 = (9, 6) ∧ P2 = (4, -4))) :=
sorry

end find_chord_points_l679_679289


namespace prod_ai_minus_i_even_l679_679334

theorem prod_ai_minus_i_even {n : ℕ} (h_perm : Permutation (a : Fin n → Fin (n + 1))) (h_odd : n % 2 = 1) : 
  (∏ i in Finset.range n, (a i : ℤ) - (i : ℤ)) % 2 = 0 :=
sorry

end prod_ai_minus_i_even_l679_679334


namespace ratio_cylinder_height_to_sphere_diameter_l679_679609

/--
A sphere is inscribed in a right circular cylinder such that the diameter of the sphere is equal to the diameter of the cylinder.
Given that the volume of the cylinder is twice that of the sphere,
find the ratio of the height of the cylinder to the diameter of the sphere.
-/

theorem ratio_cylinder_height_to_sphere_diameter
  (r h : ℝ)
  (condition1 : ∀ (s r c : ℝ), s = 2 * r →  c = 2 * s → c = r )
  (condition2 : ∀ V_c V_s : ℝ, V_s = (4 / 3) * π * r^3 →
    V_c = π * r^2 * h →
    V_c = 2 * V_s):
  (h / (2 * r) = 4 / 3) :=
by
  -- Using condition1 for sphere diameter equals cylinder diameter
  let d := 2 * r
  have diameter_equiv: d = 2 * r, from condition1 2 * r 2 * r
  -- Using condition2 for volume equivalence
  have volume_equiv: π * r^2 * h = 8 * π * r^3 / 3, from condition2 ((4 / 3)π * r^3) (π * r^2 * h)
  sorry

end ratio_cylinder_height_to_sphere_diameter_l679_679609


namespace length_PR_l679_679115

theorem length_PR {O P Q R S : Point} {r : ℝ}
  (hO : O.is_center_of_circle r)
  (h1 : P.on_circle O r)
  (h2 : Q.on_circle O r)
  (h3 : d(P, Q) = 8)
  (h4 : S = midpoint P Q)
  (h5 : R = midpoint_minor_arc P Q)
  : d(P, R) = real.sqrt (98 - 14 * real.sqrt 33) :=
by 
  -- Proof skipped.
  sorry

end length_PR_l679_679115


namespace total_valid_arrangements_l679_679541

-- Declare the problem as a theorem in Lean
theorem total_valid_arrangements : 
  let n := 5 in 
  let blue := 2 in 
  let yellow := 2 in 
  let red := 1 in 
  let non_adjacent := ∀ (arr : list (Σ (c : string), ℕ)), 
    (∀ i < arr.length - 1, arr[i].1 ≠ arr[i+1].1) → 
    (arr.length = n ∧ 
    ↑(arr.countp (λ x, x.1 = "blue")) = blue ∧ 
    ↑(arr.countp (λ x, x.1 = "yellow")) = yellow ∧ 
    ↑(arr.countp (λ x, x.1 = "red")) = red) → 
    48 
by
  sorry

end total_valid_arrangements_l679_679541


namespace number_of_boxes_l679_679254

theorem number_of_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 35) (h2 : oranges_per_box = 5) : total_oranges / oranges_per_box = 7 :=
by 
  rw [h1, h2]
  norm_num

end number_of_boxes_l679_679254


namespace falseStatementIsD_l679_679738

-- Definitions for planes and lines, assuming necessary primitives are available
variable (α β : Plane) (m n : Line)

-- Statement definitions
def statementA : Prop := (m ∥ n ∧ m ⟂ α) → n ⟂ α
def statementB : Prop := (m ⟂ α ∧ m ⟂ β) → α ∥ β
def statementC : Prop := (m ⟂ α ∧ m ∥ n ∧ n ⊆ β) → α ⟂ β
def statementD : Prop := (m ∥ α ∧ α ∩ β = n) → m ∥ n

-- Main theorem stating that the false statement is option D
theorem falseStatementIsD (hA : statementA α β m n) (hB : statementB α β m n) (hC : statementC α β m n) : ¬ statementD α β m n :=
sorry

end falseStatementIsD_l679_679738


namespace race_distance_between_Sasha_and_Kolya_l679_679488

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679488


namespace trip_time_l679_679583

theorem trip_time (x : ℝ) (T : ℝ) :
  (70 * 4 + 60 * 5 + 50 * x) / (4 + 5 + x) = 58 → 
  T = 4 + 5 + x → 
  T = 16.25 :=
by
  intro h1 h2
  sorry

end trip_time_l679_679583


namespace problem1_problem2_problem3_l679_679721

noncomputable def f (x : ℝ) (a : ℝ) := (1 - x) / (a * x) + Real.log x

theorem problem1 : 
  f 1 1 = 0 ∧ ∀ x ∈ Set.Icc (1/2) 2, f x 1 ≥ f 1 1 := 
sorry

theorem problem2 
  (h : ∀ x ∈ Set.Ici (1/2 : ℝ), DifferentiableAt ℝ (λ x, f x a) x ∧ deriv (λ x, f x a) x > 0) :
  a ≥ 2 := 
sorry

theorem problem3 
  (h : ∀ g, g = (λ x, (1 - x) / (2 * x) + Real.log x) → 
    ∃! x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1), g x = m) :
  (1 / 2 - Real.log 2 < m ∧ m ≤ (Real.exp 1 - 3) / 2) := 
sorry

end problem1_problem2_problem3_l679_679721


namespace total_toothpicks_l679_679544

theorem total_toothpicks (height width : ℕ) (h_height : height = 15) (h_width : width = 12) :
  let horizontal_toothpicks := (height + 1) * width,
      vertical_toothpicks := (width + 1) * height,
      diagonal_toothpicks := height * width,
      total_toothpicks := horizontal_toothpicks + vertical_toothpicks + diagonal_toothpicks
  in total_toothpicks = 567 :=
by
  sorry

end total_toothpicks_l679_679544


namespace smallest_n_distinct_rectangles_l679_679611

theorem smallest_n_distinct_rectangles : 
  ∃ (n : ℕ), (∀ (a b : ℕ → ℕ), (∃ (i : ℕ) (h : i ≤ n), a = b + i) → 
  (∀ (j : ℕ), j ≤ n → nat.prime j) ∧ set.size (set.univ ∩ (set.range a ∪ set.range b)) = 2 * n) ∧ 
  (∀ (m : ℕ), m < n → (∃ (x y : ℕ), set.size (set.univ ∩ (set.range x ∪ set.range y)) < 2 * m)) :=
begin
  sorry,
end

end smallest_n_distinct_rectangles_l679_679611


namespace sum_of_areas_equal_l679_679576

-- Define a regular hexagon with the point O inside it and the coloring conditions
structure RegularHexagon :=
  (side_length : ℝ)
  (point_O : ℝ × ℝ)
  (vertices : Fin 6 → ℝ × ℝ)
  (is_regular : ∀ i, Euclidean.distance (vertices i) (vertices ((i + 1) % 6)) = side_length)
  (point_interior : is_in_interior point_O vertices)

def alternate_coloring (i : Fin 6) : Prop :=
  if i % 2 = 0 then red else blue

-- Define the property of area sum equivalence
theorem sum_of_areas_equal (hex : RegularHexagon) :
  let red_triangles := {i : Fin 6 // alternate_coloring i = red}
      blue_triangles := {i : Fin 6 // alternate_coloring i = blue}
  in ∑ i in red_triangles, area (hex.vertices i) hex.point_O (hex.vertices (i + 1) % 6) =
     ∑ i in blue_triangles, area (hex.vertices i) hex.point_O (hex.vertices (i + 1) % 6) :=
sorry

end sum_of_areas_equal_l679_679576


namespace GM_HM_Inequality_l679_679969

theorem GM_HM_Inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (sqrt (a * b) = (2 / ((1 / a) + (1 / b)))) ↔ (a = b) :=
by
  sorry

end GM_HM_Inequality_l679_679969


namespace probability_female_students_l679_679688

-- Definitions of the conditions
def total_students := 6
def male_students := 4
def female_students := 2
def selected_students := 3
def ξ (selection : Finset ℕ) := selection.filter (λ i, i < female_students).card

-- The formal statement of the problem
theorem probability_female_students 
  (P : (Finset (Finset ℕ)) → ℚ)
  (H : ∀ selection ∈ Finset.powerset (Finset.range total_students), 
       P selection = (selection.card = selected_students →
                      ξ selection ≤ 1 → 
                      4 / 5)) :
  P (Finset.filter (λ s, s.card = selected_students ∧ ξ s ≤ 1) 
  (Finset.powerset (Finset.range total_students))) = 4 / 5 :=
sorry

end probability_female_students_l679_679688


namespace find_x_l679_679041

def hash_op (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) (h : hash_op x 6 = 48) : x = 3 :=
by
  sorry

end find_x_l679_679041


namespace trains_clear_in_approx_6_85_seconds_l679_679182

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train : ℝ := 80 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def speed_second_train : ℝ := 65 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_in_approx_6_85_seconds : abs (time_to_clear - 6.85) < 0.01 := sorry

end trains_clear_in_approx_6_85_seconds_l679_679182


namespace find_q_l679_679710

noncomputable theory

variables (a b p q : ℝ)

def is_root (r : ℂ) (p q : ℝ) : Prop :=
  r ^ 2 + p * r + q = 0

theorem find_q (h1 : is_root (↑b + complex.i) p q)
               (h2 : is_root (2 - a * complex.i) p q)
               (a_real : a ∈ ℝ)
               (b_real : b ∈ ℝ) :
               q = 5 :=
sorry

end find_q_l679_679710


namespace average_age_of_team_l679_679211

-- Define the captain's age
def captain_age : ℕ := 25

-- Define the wicket keeper's age (5 years older than the captain)
def wicket_keeper_age : ℕ := captain_age + 5

-- Define the number of players on the team
def number_of_players : ℕ := 11

-- Excluding the captain and wicket keeper
def remaining_players : ℕ := number_of_players - 2

-- Define the conditions for the average age of the team (A) in Lean
theorem average_age_of_team :
  ∃ A : ℝ,
  let total_age_of_team := number_of_players * A in
  let total_age_excluding_captain_and_keeper := total_age_of_team - (captain_age + wicket_keeper_age) in
  let average_age_of_remaining_players := A - 1 in
  let total_age_of_remaining_players := remaining_players * average_age_of_remaining_players in
  total_age_excluding_captain_and_keeper = total_age_of_remaining_players →
  A = 23 :=
begin
  -- Proof is not required, so we use sorry
  sorry
end

end average_age_of_team_l679_679211


namespace diagonals_in_convex_polygon_l679_679372

-- Define the number of sides for the polygon
def polygon_sides : ℕ := 15

-- The main theorem stating the number of diagonals in a convex polygon with 15 sides
theorem diagonals_in_convex_polygon : polygon_sides = 15 → ∃ d : ℕ, d = 90 :=
by
  intro h
  -- sorry is a placeholder for the proof
  sorry

end diagonals_in_convex_polygon_l679_679372


namespace exists_consecutive_numbers_with_prime_divisors_l679_679744

theorem exists_consecutive_numbers_with_prime_divisors (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p < q ∧ q < 2 * p) :
  ∃ n m : ℕ, (m = n + 1) ∧ 
             (Nat.gcd n p = p) ∧ (Nat.gcd m p = 1) ∧ 
             (Nat.gcd m q = q) ∧ (Nat.gcd n q = 1) :=
by
  sorry

end exists_consecutive_numbers_with_prime_divisors_l679_679744


namespace count_special_numbers_l679_679754

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem count_special_numbers : 
  (Finset.filter (λ n, n = 9 * (sum_of_digits n)) (Finset.range 2000)).card = 4 :=
sorry

end count_special_numbers_l679_679754


namespace man_age_year_l679_679593

theorem man_age_year (x : ℕ) (h1 : x^2 = 1892) (h2 : 1850 ≤ x ∧ x ≤ 1900) :
  (x = 44) → (1892 = 1936) := by
sorry

end man_age_year_l679_679593


namespace find_sum_p_q_l679_679545

-- Define the geometrical setup and variables.
variables {EF FG GH HE : ℕ} (h1 : EF = 75) (h2 : FG = 50) (h3 : GH = 24) (h4 : HE = 70)
           (h_parallel : EF ∥ GH)

noncomputable def EQ : ℚ :=
  525 / 11

-- State the problem to find p + q given that p/q is the length of EQ, with p and q relatively prime.
theorem find_sum_p_q (p q : ℕ) (h_rel_prime : Nat.coprime p q) (h_EQ : EQ = p / q) :
  p + q = 536 :=
sorry

end find_sum_p_q_l679_679545


namespace part1_part2_case1_part2_case2_part2_case3_l679_679699

open Nat

-- Define the sequence {a_n}
def seq_a : ℕ → ℝ
| 0     := 0                    -- not used
| n + 1 := if n = 0 then 1 / 2 else seq_a n / (2 * seq_a n + 1)

-- Prove that 1/a_n forms an arithmetic sequence with a_n = 1/(2n)
theorem part1 (n : ℕ) (hn : n > 0) : 1 / (seq_a n) = 2 * n :=
by sorry

-- Case 1: b_n = a_n * a_{n+1}
def seq_b1 : ℕ → ℝ
| n := seq_a n * seq_a (n + 1)

theorem part2_case1 (n : ℕ) (hn : n > 0) : 
  ∑ i in (range n).map succ, seq_b1 i = n / (4 * n + 4) :=
by sorry

-- Case 2: b_n = (-1)^n / a_n
def seq_b2 : ℕ → ℝ
| n := (-1) ^ n / seq_a n

theorem part2_case2 (n : ℕ) (hn : n > 0) : 
  (∑ i in (range n).map succ, seq_b2 i) = 
  if n % 2 = 1 then -n - 1 else n :=
by sorry

-- Case 3: b_n = 1 / a_n + (1/3)^(1 / a_n)
def seq_b3 : ℕ → ℝ
| n := 1 / seq_a n + (1 / 3) ^ (1 / seq_a n)

theorem part2_case3 (n : ℕ) (hn : n > 0) : 
  (∑ i in (range n).map succ, seq_b3 i) = 
  n^2 + n + (1 / 8) * (1 - 1 / 9^n) :=
by sorry

end part1_part2_case1_part2_case2_part2_case3_l679_679699


namespace difference_ne_1998_l679_679963

-- Define the function f(n) = n^2 + 4n
def f (n : ℕ) : ℕ := n^2 + 4 * n

-- Statement: For all natural numbers n and m, the difference f(n) - f(m) is not 1998
theorem difference_ne_1998 (n m : ℕ) : f n - f m ≠ 1998 := 
by {
  sorry
}

end difference_ne_1998_l679_679963


namespace smith_gave_randy_l679_679861

theorem smith_gave_randy :
  ∀ (s amount_given amount_left : ℕ), amount_given = 1200 → amount_left = 2000 → s = amount_given + amount_left → s = 3200 :=
by
  intros s amount_given amount_left h_given h_left h_total
  rw [h_given, h_left] at h_total
  exact h_total

end smith_gave_randy_l679_679861


namespace zack_initial_marbles_l679_679567

noncomputable def total_initial_marbles (x : ℕ) : ℕ :=
  81 * x + 27

theorem zack_initial_marbles :
  ∃ x : ℕ, total_initial_marbles x = 270 :=
by
  use 3
  sorry

end zack_initial_marbles_l679_679567


namespace range_of_f_l679_679893

def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^x else -x^2 + 2*x + 1

theorem range_of_f : set.image f set.univ = set.Iic (2 : ℝ) :=
by
  sorry

end range_of_f_l679_679893


namespace common_ratio_is_neg_one_l679_679337

theorem common_ratio_is_neg_one (a : ℕ → ℝ) (d : ℝ) (h₀ : 0 < d ∧ d < 2 * Real.pi) (h₁ : ∀ n : ℕ, a n = a 1 + (n-1) * d)
  (h₂ : ∃ r : ℝ, ∀ n : ℕ, cosine_sequence_is_geometric_sequence a r) : r = -1 :=
by
  sorry

def cosine_sequence_is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, (cos(a n) / cos(a (n-1))) = r

end common_ratio_is_neg_one_l679_679337


namespace sum_of_x_l679_679702

-- define the function f as an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- define the function f as strictly monotonic on the interval (0, +∞)
def is_strictly_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- define the main problem statement
theorem sum_of_x (f : ℝ → ℝ) (x : ℝ) (h1 : is_even_function f) (h2 : is_strictly_monotonic_on_positive f) (h3 : x ≠ 0)
  (hx : f (x^2 - 2*x - 1) = f (x + 1)) : 
  ∃ (x1 x2 x3 x4 : ℝ), (x1 + x2 + x3 + x4 = 4) ∧
                        (x1^2 - 3*x1 - 2 = 0) ∧
                        (x2^2 - 3*x2 - 2 = 0) ∧
                        (x3^2 - x3 = 0) ∧
                        (x4^2 - x4 = 0) :=
sorry

end sum_of_x_l679_679702


namespace calculate_expression_l679_679196

theorem calculate_expression : (3072 - 2993) ^ 2 / 121 = 49 :=
by
  sorry

end calculate_expression_l679_679196


namespace p_at_5_l679_679436

def h (x : ℝ) : ℝ := 5 / (3 - x)
def h_inv (x : ℝ) : ℝ := 3 - 5 / x
def p (x : ℝ) : ℝ := 1 / (h_inv x) + 7

theorem p_at_5 : p 5 = 7.5 := by
  sorry

end p_at_5_l679_679436


namespace digits_unique_l679_679887

noncomputable def base_representation (a : ℕ) : ℕ :=
(A+1) * (A+1) * (A+1) * (A+1) + (A+1) * (A+1) * (A+1) + (A+1) * (A+1) + (A+1)

def left_hand_side (A : ℕ) : ℕ :=
(base_representation A) ^ 2

def right_hand_side (A B C : ℕ) : ℕ :=
A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + B * (A+1)^4 + C * (A+1)^3 + C * (A+1)^2 + C * (A+1) + B

theorem digits_unique (A B C : ℕ) : A = 2 ∧ B = 1 ∧ C = 0 :=
by
  -- Given the constraints and elements to prove.
  sorry

end digits_unique_l679_679887


namespace Kamal_biology_marks_l679_679425

theorem Kamal_biology_marks 
  (E : ℕ) (M : ℕ) (P : ℕ) (C : ℕ) (A : ℕ) (N : ℕ) (B : ℕ) 
  (hE : E = 66)
  (hM : M = 65)
  (hP : P = 77)
  (hC : C = 62)
  (hA : A = 69)
  (hN : N = 5)
  (h_total : N * A = E + M + P + C + B) 
  : B = 75 :=
by
  sorry

end Kamal_biology_marks_l679_679425


namespace cozy_dash_sum_digits_l679_679285

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem cozy_dash_sum_digits :
  (∑ n in {n | (Nat.ceil (n / 3) - Nat.ceil (n / 7) = 25)}, sum_of_digits n) = 3 :=
sorry

end cozy_dash_sum_digits_l679_679285


namespace odd_function_interval_symmetric_l679_679888

theorem odd_function_interval_symmetric {f : ℝ → ℝ} {b : ℝ} 
  (h_odd : ∀ x ∈ set.Icc (b-1) 2, f (-x) = - (f x)) : 
  b = -1 :=
by {
  sorry
}

end odd_function_interval_symmetric_l679_679888


namespace evaluate_g_at_3_l679_679769

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 - 7 * x + 3

theorem evaluate_g_at_3 : g 3 = 126 := 
by 
  sorry

end evaluate_g_at_3_l679_679769


namespace maxInsuredEmployees_is_121_l679_679964

noncomputable def maxInsuredEmployees : ℕ :=
  let premium := 5000000
  let outpatientCost := 18000
  let hospitalizationCost := 60000
  let hospitalizationProportion := 0.25
  let overheadCost := 2800
  let profitMargin := 0.15
  let outpatientTotal := outpatientCost
  let hospitalizationTotal := hospitalizationCost * hospitalizationProportion
  let totalCostWithoutProfit := outpatientTotal + hospitalizationTotal + overheadCost
  let profit := totalCostWithoutProfit * profitMargin
  let totalCostWithProfit := totalCostWithoutProfit + profit
  Nat.floor (premium / totalCostWithProfit)

/- The following theorem states that the maximum number of insured employees is 121 given 
   the conditions described. -/
theorem maxInsuredEmployees_is_121 : maxInsuredEmployees = 121 :=
  sorry

end maxInsuredEmployees_is_121_l679_679964


namespace least_clock_equiv_square_l679_679450

def is_clock_equiv (a b : ℕ) (m : ℕ) : Prop :=
  (a - b) % m = 0

def find_least_clock_equiv_greater_than_five : ℕ :=
  @WellFounded.fix _ _ ⟨5, sorry⟩ ⟨_, sorry⟩

theorem least_clock_equiv_square (n : ℕ) (h : n > 5) :
  n = find_least_clock_equiv_greater_than_five
→ is_clock_equiv ((find_least_clock_equiv_greater_than_five)^2) (find_least_clock_equiv_greater_than_five) 12 :=
by
  intro hn
  intro hclock
  sorry

example : find_least_clock_equiv_greater_than_five = 9 :=
by 
  sorry

end least_clock_equiv_square_l679_679450


namespace find_complex_solutions_l679_679307

noncomputable def z_solutions : Set ℂ :=
  {z : ℂ | (z^2 = -45 - 28 * Complex.I) ∧ (z^3 = 8 + 26 * Complex.I)}

theorem find_complex_solutions :
  z_solutions = {√10 - Complex.I * √140, -√10 + Complex.I * √140} := sorry

end find_complex_solutions_l679_679307


namespace claire_meets_dan_distance_l679_679548

theorem claire_meets_dan_distance :
  ∀ {C D : EuclideanSpace ℝ 2} (CD : dist C D = 120)
  (angle_C_line : ∀ {E : EuclideanSpace ℝ 2}, ∃ θ : ℝ, θ = real.pi / 4 → angle ⟨C, D, E⟩ θ)
  (claire_speed : ∀ {t : ℝ}, t ≥ 0 → dist C (C + t • (1, 1)) = 9 * t)
  (dan_speed : ∀ {t : ℝ}, t ≥ 0 → dist D (D + t • (1, -1)) = 6 * t),
  (∃ t : ℝ, t = 24 ∧ dist C (C + t • (1, 1)) = 216) :=
begin
  intros C D CD angle_C_line claire_speed dan_speed,
  -- Proof omitted
  sorry
end

end claire_meets_dan_distance_l679_679548


namespace lengths_of_legs_l679_679535

noncomputable def point_on_parabola (x₀ : ℝ) : ℝ × ℝ :=
  (x₀, x₀^2)

noncomputable def tangent_line (x₀ : ℝ) : ℝ → ℝ :=
  λ x, 2 * x₀ * (x - x₀) + x₀^2

noncomputable def x_intercept (x₀ : ℝ) : ℝ :=
  x₀ / 2

noncomputable def y_intercept (x₀ : ℝ) : ℝ :=
  -x₀^2

noncomputable def area_of_triangle (x₀ : ℝ) : ℝ :=
  0.5 * x_intercept x₀ * (-y_intercept x₀)

theorem lengths_of_legs (x₀ : ℝ) (h : x₀ > 0) (h_area : area_of_triangle x₀ = 16) :
  x_intercept x₀ = 2 ∧ y_intercept x₀ = -16 :=
by
  sorry

end lengths_of_legs_l679_679535


namespace truncated_cone_contact_radius_l679_679610

theorem truncated_cone_contact_radius (R r r' ζ : ℝ)
  (h volume_condition : ℝ)
  (R_pos : 0 < R)
  (r_pos : 0 < r)
  (r'_pos : 0 < r')
  (ζ_pos : 0 < ζ)
  (h_eq : h = 2 * R)
  (volume_condition_eq :
    (2 : ℝ) * ((4 / 3) * Real.pi * R^3) = 
    (2 / 3) * Real.pi * h * (r^2 + r * r' + r'^2)) :
  ζ = (2 * R * Real.sqrt 5) / 5 :=
by
  sorry

end truncated_cone_contact_radius_l679_679610


namespace max_colors_in_cube_l679_679336

-- Defining the problem setting
def max_colors (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Main theorem statement
theorem max_colors_in_cube (n : ℕ) (hn : n ≥ 2) :
  ∀ (C : ℕ),
  ∃ (colors : set ℕ), 
  (colors.card = C) ∧  
  (∀ (x y z : ℕ), x < n → y < n → z < n → 
  (colors x ∧ colors y ∧ colors y = colors z)) → C ≤ max_colors n :=
begin
  sorry -- Proof not required
end

end max_colors_in_cube_l679_679336


namespace solve_real_eq_l679_679310

noncomputable def real_solutions (x y : ℝ) (k : ℤ) : Prop :=
  (x = -1 ∧ y = -π/2 - 2 * k * π) ∨ (x = 1 ∧ y = 3 * π / 2 + 2 * k * π)

theorem solve_real_eq (x y : ℝ) :
  ∃ k : ℤ, x^2 + 2 * x * real.sin (x * y) + 1 = 0 ↔ real_solutions x y k :=
by
  sorry

end solve_real_eq_l679_679310


namespace proj_w_2v_3u_eq_l679_679096

open Matrix

-- Definitions of vectors v and u with their projections
def v : Vector (Fin 2) ℝ := ![4, -1]
def u : Vector (Fin 2) ℝ := ![1, 3]

-- Proving that the projection of (2v + 3u) equals the given vector (11, 7)
theorem proj_w_2v_3u_eq : 
  @proj _ _ _ ![2, v] + @proj _ _ _ ![3, u] = ![11, 7] :=
by
  sorry

end proj_w_2v_3u_eq_l679_679096


namespace seven_thousand_twenty_two_is_7022_l679_679498

-- Define the translations of words to numbers
def seven_thousand : ℕ := 7000
def twenty_two : ℕ := 22

-- Define the full number by summing its parts
def seven_thousand_twenty_two : ℕ := seven_thousand + twenty_two

theorem seven_thousand_twenty_two_is_7022 : seven_thousand_twenty_two = 7022 := by
  sorry

end seven_thousand_twenty_two_is_7022_l679_679498


namespace incircle_centers_equidistant_midpoint_arc_ABC_l679_679452

-- Define all the necessary properties and points
variables {A B C K L: Point} (circumcircle: Circle)
(def A B C : on circumcircle)
(def K : on arc A B circumcircle)
(def L : on arc B C circumcircle)
(def KL_parallel_AC : Line KL || Line AC)

/-- Proof Statement -/
theorem incircle_centers_equidistant_midpoint_arc_ABC :
  let I1 := incenter (triangle A B K)
  let I2 := incenter (triangle C B L)
  let R  := midpoint_arc ABC circumcircle
  distance I1 R = distance I2 R :=
sorry

end incircle_centers_equidistant_midpoint_arc_ABC_l679_679452


namespace area_of_QTUR_l679_679216

open Real

-- Define the problem
noncomputable def equilateral_triangle (a : ℝ) : Type :=
{P Q R : Type // distance P Q = a ∧ distance Q R = a ∧ distance P R = a}

noncomputable def extend_segment (P Q : Type) (k : ℝ) : Type :=
{Q S : Type // distance Q S = k * distance Q R }

noncomputable def midpoint (P Q : Type) : Type :=
{T : Type // distance P T = distance T Q}

noncomputable def intersection (line1 line2 : Type) : Type :=
{U : Type} -- Not elaborating further as specifics are not given

-- Problem in Lean 4
theorem area_of_QTUR :
  ∀ (P Q R S T U : Type) (a : ℝ) (k : ℝ),
  (⟦equilateral_triangle a⟧ P Q R) →
  (⟦extend_segment Q R k⟧ Q S) →
  (⟦midpoint P Q⟧ T) →
  (⟦intersection (line P R) (line T S) ⟧ U) →
  k = 1 / 2 →
  let area := (2 : ℝ) * (sqrt 3) in
  3 * area / 2 = 3 * sqrt 3 := 
sorry

end area_of_QTUR_l679_679216


namespace q1_q2_q3_l679_679862

-- Question (1)
theorem q1:
  let a := Real.sqrt 5 - 2 in
  let b := 3 in
  a + b + 5 = Real.sqrt 5 + 6 :=
by
  sorry

-- Question (2)
theorem q2:
  ∃ (x : ℤ) (y : ℝ), 10 + Real.sqrt 3 = x + y ∧ 0 < y ∧ y < 1 ∧ - (x - y + Real.sqrt 3) = -12 :=
by
  sorry

-- Question (3)
theorem q3:
  let a := Real.sqrt 11 - 3 in
  let b := 4 - Real.sqrt 11 in
  a + b = 1 :=
by
  sorry

end q1_q2_q3_l679_679862


namespace proposition_check_l679_679355

variable {α : Type*}
variables {M P : Set α}

theorem proposition_check (h : ¬ (M ⊆ P)) :
  (¬ ∀ x ∈ M, x ∈ P) ∧ (∃ x ∈ M, x ∉ P) :=
by
  split
  { intro H
    apply h
    exact fun x hx => H x hx }
  { sorry }

end proposition_check_l679_679355


namespace square_probability_l679_679825

theorem square_probability :
  let S := square_with_diagonal ⟨1/8, 3/8⟩ ⟨-1/8, -3/8⟩
  let v := (x, y)
  let T_v := translate S v
  (0 ≤ x ∧ x ≤ 1000) ∧ (0 ≤ y ∧ y ≤ 1000) →
  probability (λ p, contains_exactly_three_integers (T_v p)) = 1 / 50 :=
sorry

end square_probability_l679_679825


namespace find_natural_numbers_with_integer_roots_l679_679309

theorem find_natural_numbers_with_integer_roots :
  ∃ (p q : ℕ), 
    (∀ x : ℤ, x * x - (p * q) * x + (p + q) = 0 → ∃ (x1 x2 : ℤ), x = x1 ∧ x = x2 ∧ x1 + x2 = p * q ∧ x1 * x2 = p + q) ↔
    ((p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
-- proof skipped
sorry

end find_natural_numbers_with_integer_roots_l679_679309


namespace evaluate_range_y_l679_679296

theorem evaluate_range_y (b : Fin 15 → ℕ) (h : ∀ i, b i = 0 ∨ b i = 1) :
  3 ≤ (∑ i : Fin 15, (b i) / 2^(i.succ)) + 3 ∧ (∑ i : Fin 15, (b i) / 2^(i.succ)) + 3 < 4 := 
sorry

end evaluate_range_y_l679_679296


namespace number_of_planes_determined_l679_679524

-- Define a rectangular parallelepiped structure
structure RectangularParallelepiped :=
  (faces : Fin 6 → Fin 4 → ℝ × ℝ)  -- assuming faces are indexed by 6 faces, each face has 4 vertices

-- The theorem to prove the number of planes determined by the diagonals on each face of a rectangular parallelepiped
theorem number_of_planes_determined (R : RectangularParallelepiped) : 
  let face_diagonals_planes := 3 * 2,
      vertex_planes := 8 in
  face_diagonals_planes + vertex_planes = 14 :=
by
  sorry

end number_of_planes_determined_l679_679524


namespace solution_set_of_inequality_l679_679533

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 4 * x - 3 > 0 } = { x : ℝ | 1 < x ∧ x < 3 } := sorry

end solution_set_of_inequality_l679_679533


namespace sum_integers_minus15_to_6_l679_679943

def sum_range (a b : ℤ) : ℤ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus15_to_6 : sum_range (-15) (6) = -99 :=
  by
  -- Skipping the proof details
  sorry

end sum_integers_minus15_to_6_l679_679943


namespace minimum_envelopes_for_50_percent_repeated_flag_probability_l679_679878

theorem minimum_envelopes_for_50_percent_repeated_flag_probability
  (flags : ℕ) (envelope_flags : ℕ) (opened_envelopes : ℕ → ℕ) :
  flags = 12 →
  envelope_flags = 2 →
  ∃ k, (opened_envelopes k) = 3 ∧
  let total_flags := flags in
  let total_chosen := (opened_envelopes k) * envelope_flags in
  let probability_all_different := 
    (∏ i in finset.range total_chosen, (total_flags - i)) / (total_flags ^ total_chosen : ℚ) in
  probability_all_different < 0.5 :=
begin
  sorry
end

end minimum_envelopes_for_50_percent_repeated_flag_probability_l679_679878


namespace two_digit_numbers_of_form_3_pow_n_l679_679017

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679017


namespace alpha_square_greater_beta_square_l679_679771

theorem alpha_square_greater_beta_square
  (alpha beta : ℝ)
  (h_alpha : alpha ∈ Icc (-real.pi / 2) (real.pi / 2))
  (h_beta : beta ∈ Icc (-real.pi / 2) (real.pi / 2))
  (h_condition : alpha * real.sin alpha - beta * real.sin beta > 0) :
  alpha^2 > beta^2 := 
by
  sorry

end alpha_square_greater_beta_square_l679_679771


namespace difference_operator_l679_679732

variables {R : Type*} [Field R]

noncomputable def polynomial (x : R) (k : ℕ) (coeffs : Fin (k + 1) → R) : R :=
  (coeffs 0) * x^k + (coeffs 1) * x^(k - 1) + ... + (coeffs (k - 1)) * x + (coeffs k)

theorem difference_operator {R : Type*} [Field R] {a₀ : R} (a₀_ne_zero : a₀ ≠ 0)
  (a₁ a₂ ... aₖ : R) {k : ℕ} :
  ∀ (x : R), Δ^(k) (λ x, a₀ * x^k + a₁ * x^(k-1) + ... + aₖ) x = k! * a₀ ∧
  Δ^(k+1) (λ x, a₀ * x^k + a₁ * x^(k-1) + ... + aₖ) x = 0 :=
sorry

end difference_operator_l679_679732


namespace find_principal_l679_679206

-- Defining the conditions
def A : ℝ := 5292
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The theorem statement
theorem find_principal :
  ∃ (P : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ P = 4800 :=
by
  sorry

end find_principal_l679_679206


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679481

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679481


namespace erased_number_eq_7_l679_679242

theorem erased_number_eq_7 (n x : ℕ) (h₁ : x ≤ n) 
  (h₂ : (2 * (∑ i in (finset.range (n+1)).erase x, i : ℕ) / (n - 1)) = 70 + 14 / 17) : 
  x = 7 := 
sorry

end erased_number_eq_7_l679_679242


namespace q1_q2_l679_679235

-- Define the contingency table data
def CaseNotGood := 40
def CaseGood := 60
def ControlNotGood := 10
def ControlGood := 90

-- Define total samples sizes
def TotalCases := CaseNotGood + CaseGood
def TotalControl := ControlNotGood + ControlGood
def TotalNotGood := CaseNotGood + ControlNotGood
def TotalGood := CaseGood + ControlGood
def TotalSamples := TotalCases + TotalControl

-- Define the critical value for 99% confidence from Chi-Squared distribution
def CriticalValue99 := 6.635

-- Define the K^2 calculation formula
def calcK2 (a b c d n : ℕ) : ℚ :=
  let ad_minus_bc := a * d - b * c
  n * (ad_minus_bc ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculate K^2 with provided survey data
def k2_value := calcK2 CaseNotGood CaseGood ControlNotGood ControlGood TotalSamples

-- Define the risk ratio calculation
def risk_ratio (PAB PA_B PA_B PB_A : ℚ) : ℚ :=
  (PAB / PA_B) * (PA_B / PB_A)

-- Calculate estimated probabilities using survey data
def P_A_given_B := (CaseNotGood : ℚ) / (TotalCases : ℚ)
def P_A_given_B_compl := (ControlNotGood : ℚ) / (TotalControl : ℚ)
def P_A_compl_given_B := 1 - P_A_given_B
def P_A_compl_given_B_compl := 1 - P_A_given_B_compl

-- Calculate the risk ratio R using the defined probabilities
def risk_ratio_value := risk_ratio P_A_given_B P_A_given_B_compl P_A_compl_given_B P_A_compl_given_B_compl

-- Theorem statements
theorem q1 : k2_value > CriticalValue99 := by sorry

theorem q2 : risk_ratio_value = 6 := by sorry

end q1_q2_l679_679235


namespace base6_arithmetic_l679_679919

theorem base6_arithmetic :
  let a := 4512
  let b := 2324
  let c := 1432
  let base := 6
  let a_b10 := 4 * base^3 + 5 * base^2 + 1 * base + 2
  let b_b10 := 2 * base^3 + 3 * base^2 + 2 * base + 4
  let c_b10 := 1 * base^3 + 4 * base^2 + 3 * base + 2
  let result_b10 := a_b10 - b_b10 + c_b10
  let result_base6 := 4020
  (result_b10 / base^3) % base = 4 ∧
  (result_b10 / base^2) % base = 0 ∧
  (result_b10 / base) % base = 2 ∧
  result_b10 % base = 0 →
  result_base6 = 4020 := by
  sorry

end base6_arithmetic_l679_679919


namespace problem_statement_l679_679520

noncomputable theory
open Real

def intervals_of_monotonic_increase (y : ℝ → ℝ) : Set (Set ℝ) :=
  { I : Set ℝ | ∀ x ∈ I, ∀ y ∈ I, x ≤ y → y' x ≤ y' y }
  where y' := y

theorem problem_statement : 
  let y := λ x : ℝ, abs (log (abs (x-2)) / log 2)
  intervals_of_monotonic_increase y = { Set.Ioo 1 2, Set.Ioi 3 } :=
sorry

end problem_statement_l679_679520


namespace distance_between_sasha_and_kolya_when_sasha_finishes_l679_679474

theorem distance_between_sasha_and_kolya_when_sasha_finishes
  (vs vl vk : ℝ) -- speeds of Sasha, Lyosha, Kolya
  (h1 : vl = 0.9 * vs) -- Lyosha's speed is 90% of Sasha's speed
  (h2 : vk = 0.9 * vl) -- Kolya's speed 90% of Lyosha's speed
  (h3 : vs > 0) (h4 : vl > 0) (h5 : vk > 0) -- speeds are positive
  : let t := 100 / vs in
    100 - (vk * t) = 19 :=
by 
  sorry

end distance_between_sasha_and_kolya_when_sasha_finishes_l679_679474


namespace limit_f_at_zero_l679_679215

open Real Filter

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt (1 + tan x) - sqrt (1 + sin x)) / (x^3)

theorem limit_f_at_zero :
  tendsto f (nhds 0) (nhds (1/4)) :=
begin
  sorry
end

end limit_f_at_zero_l679_679215


namespace powers_of_three_two_digit_count_l679_679024

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679024


namespace central_symmetry_preserves_distance_l679_679118

variables {Point : Type} [MetricSpace Point]

def central_symmetry (O A A' B B' : Point) : Prop :=
  dist O A = dist O A' ∧ dist O B = dist O B'

theorem central_symmetry_preserves_distance {O A A' B B' : Point}
  (h : central_symmetry O A A' B B') : dist A B = dist A' B' :=
sorry

end central_symmetry_preserves_distance_l679_679118


namespace divisibility_theorem_l679_679463

theorem divisibility_theorem (a b n : ℕ) (h : a^n ∣ b) : a^(n + 1) ∣ (a + 1)^b - 1 :=
by 
sorry

end divisibility_theorem_l679_679463


namespace distance_between_Sasha_and_Kolya_l679_679473

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679473


namespace problem_statement_l679_679431

open Real

-- Define the distinct real numbers a, b, c
variables (a b c : ℝ)
-- Assume a, b, c are distinct 
variable [h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a]

-- Define the condition of the problem
axiom h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0

-- Define the proof problem
theorem problem_statement : 
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 :=
sorry

end problem_statement_l679_679431


namespace amy_money_left_l679_679199

-- Definitions for item prices
def stuffed_toy_price : ℝ := 2
def hot_dog_price : ℝ := 3.5
def candy_apple_price : ℝ := 1.5
def soda_price : ℝ := 1.75
def ferris_wheel_ticket_price : ℝ := 2.5

-- Tax rate
def tax_rate : ℝ := 0.1 

-- Initial amount Amy had
def initial_amount : ℝ := 15

-- Function to calculate price including tax
def price_with_tax (price : ℝ) (tax_rate : ℝ) : ℝ := price * (1 + tax_rate)

-- Prices including tax
def stuffed_toy_price_with_tax := price_with_tax stuffed_toy_price tax_rate
def hot_dog_price_with_tax := price_with_tax hot_dog_price tax_rate
def candy_apple_price_with_tax := price_with_tax candy_apple_price tax_rate
def soda_price_with_tax := price_with_tax soda_price tax_rate
def ferris_wheel_ticket_price_with_tax := price_with_tax ferris_wheel_ticket_price tax_rate

-- Discount rates
def discount_most_expensive : ℝ := 0.5
def discount_second_most_expensive : ℝ := 0.25

-- Applying discounts
def discounted_hot_dog_price := hot_dog_price_with_tax * (1 - discount_most_expensive)
def discounted_ferris_wheel_ticket_price := ferris_wheel_ticket_price_with_tax * (1 - discount_second_most_expensive)

-- Total cost with discounts
def total_cost_with_discounts : ℝ := 
  stuffed_toy_price_with_tax + discounted_hot_dog_price + candy_apple_price_with_tax +
  soda_price_with_tax + discounted_ferris_wheel_ticket_price

-- Amount left after purchases
def amount_left : ℝ := initial_amount - total_cost_with_discounts

theorem amy_money_left : amount_left = 5.23 := by
  -- Here the proof will be provided.
  sorry

end amy_money_left_l679_679199


namespace symmetric_difference_cardinality_l679_679390

-- Definitions based on given conditions
variables (x y : Set ℤ)
#check Finset
variables (hx : x.card = 8) (hy : y.card = 10) (hxy : (x ∩ y).card = 6)

-- Statement to be proved
theorem symmetric_difference_cardinality : (x \triangle y).card = 6 :=
by 
  sorry

end symmetric_difference_cardinality_l679_679390


namespace max_range_of_five_distinct_numbers_avg_13_median_15_l679_679712

noncomputable def max_range (s : Set ℕ) : ℕ :=
  if s.Nonempty then s.max' (Set.finite_of_finite_toSet (finite_mem_finset.mpr (Set.to_finset_finite s)))
  - s.min' (Set.finite_of_finite_toSet (finite_mem_finset.mpr (Set.to_finset_finite s)))
  else 0

theorem max_range_of_five_distinct_numbers_avg_13_median_15 (s : Finset ℕ) 
  (h_len : s.card = 5) (h_distinct : s = s.to_finset) 
  (h_avg : s.Sum / 5 = 13) (h_median : s.to_list.sorted.nth 2 = some 15) : 
  max_range s.to_set = 33 :=
sorry

end max_range_of_five_distinct_numbers_avg_13_median_15_l679_679712


namespace finley_tickets_l679_679916

theorem finley_tickets (Wally_tickets : ℕ)
                       (fraction_given : ℚ)
                       (ratio_jensen parts_jensen : ℕ)
                       (ratio_finley parts_finley : ℕ)
                       (total_parts : ratio_jensen + ratio_finley = parts_jensen + parts_finley):
                       fraction_given * Wally_tickets * (ratio_finley / total_parts) = 220 :=
by
  sorry

end finley_tickets_l679_679916


namespace problem1_problem2_problem3_problem4_l679_679641

-- Problem 1
theorem problem1 : (2 / 19) * (8 / 25) + (17 / 25) / (19 / 2) = 2 / 19 := 
by sorry

-- Problem 2
theorem problem2 : (1 / 4) * 125 * (1 / 25) * 8 = 10 := 
by sorry

-- Problem 3
theorem problem3 : ((1 / 3) + (1 / 4)) / ((1 / 2) - (1 / 3)) = 7 / 2 := 
by sorry

-- Problem 4
theorem problem4 : ((1 / 6) + (1 / 8)) * 24 * (1 / 9) = 7 / 9 := 
by sorry

end problem1_problem2_problem3_problem4_l679_679641


namespace solution_set_f_inequality_l679_679230

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1
  else abs (x - 3) - 1

theorem solution_set_f_inequality :
  { x : ℝ | f x < -1 / 2 } = { x : ℝ | x < -1 } ∪ { x : ℝ | 5 / 2 < x ∧ x < 7 / 2 } :=
by
  -- proof omitted
  sorry

end solution_set_f_inequality_l679_679230


namespace sum_of_perimeters_of_infinite_triangles_l679_679257

theorem sum_of_perimeters_of_infinite_triangles (s1 : ℝ) (h : s1 = 45) :
  let S := ∑' (n : ℕ), 3 * s1 / 2^n in
  S = 270 :=
by
  -- Given constraints
  rw h
  -- Key formula
  have geometric_series_sum : ∑' (n : ℕ), 3 * 45 / 2^n = 6 * 45 :=
    sorry  -- This step would require proof of summing the geometric series
  -- Apply the sum result
  rw geometric_series_sum
  -- Simplify
  ring
  -- Conclusion
  refl

end sum_of_perimeters_of_infinite_triangles_l679_679257


namespace quadratic_solution_l679_679503

theorem quadratic_solution :
  (∀ x : ℝ, 3 * x^2 - 13 * x + 5 = 0 → 
           x = (13 + Real.sqrt 109) / 6 ∨ x = (13 - Real.sqrt 109) / 6) 
  := by
  sorry

end quadratic_solution_l679_679503


namespace race_distance_between_Sasha_and_Kolya_l679_679491

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679491


namespace shortest_distance_point_l679_679704

-- Definition of points A, B, and coordinates of C to be determined.
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 2 }
def B : Point := { x := 2, y := 2 }

def isShortestDistance (C : Point) : Prop :=
  C.y = 0 ∧ (∀ P : Point, P.y = 0 → dist A C + dist B C ≤ dist A P + dist B P)

theorem shortest_distance_point : ∃ C : Point, C = { x := 1/2, y := 0 } ∧ isShortestDistance C :=
sorry

end shortest_distance_point_l679_679704


namespace train_length_120_l679_679983

structure JoggerAndTrain :=
  (speed_jogger_kmph : ℝ)
  (lead_jogger_m : ℝ)
  (speed_train_kmph : ℝ)
  (time_pass_s : ℝ)

def length_of_train (data : JoggerAndTrain) : ℝ :=
  let speed_jogger_mps := data.speed_jogger_kmph * 1000 / 3600
  let speed_train_mps := data.speed_train_kmph * 1000 / 3600
  let relative_speed_mps := speed_train_mps - speed_jogger_mps
  let distance_covered_m := relative_speed_mps * data.time_pass_s
  distance_covered_m - data.lead_jogger_m

theorem train_length_120 (data : JoggerAndTrain) (h1 : data.speed_jogger_kmph = 9)
    (h2 : data.lead_jogger_m = 280) 
    (h3 : data.speed_train_kmph = 45)
    (h4 : data.time_pass_s = 40.00000000000001) : 
    length_of_train data = 120 := 
by
  sorry

end train_length_120_l679_679983


namespace third_number_in_first_set_l679_679510

theorem third_number_in_first_set (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end third_number_in_first_set_l679_679510


namespace box_volume_l679_679652

-- Definitions for the dimensions of the box: Length (L), Width (W), and Height (H)
variables (L W H : ℝ)

-- Condition 1: Area of the front face is half the area of the top face
def condition1 := L * W = 0.5 * (L * H)

-- Condition 2: Area of the top face is 1.5 times the area of the side face
def condition2 := L * H = 1.5 * (W * H)

-- Condition 3: Area of the side face is 200
def condition3 := W * H = 200

-- Theorem stating the volume of the box is 3000 given the above conditions
theorem box_volume : condition1 L W H ∧ condition2 L W H ∧ condition3 W H → L * W * H = 3000 :=
by sorry

end box_volume_l679_679652


namespace desired_expression_equals_3_l679_679784

variable {ABC : Type}
variables (A B C D E : ABC)
variable [InTriangle ABC A B C]
variable [is_perpendicular E B (segment AD)]
variable [bisector_angle A D]

-- Given lengths
variable (AB : ℝ := 5)
variable (BE : ℝ := 4)
variable (AE : ℝ := 3)
variable (AC : ℝ)

-- Given relationships
variable (h1 : AC > AB)
variable (h2 : InTriangle_internal_angle_bisector A D B C)

open Real EuclideanGeometry

-- Prove the desired expression equals 3
theorem desired_expression_equals_3
  (h1 : AC > 5)
  (h2 : InTriangle_internal_angle_bisector A D B C)
  (h3 : segment_length AB = 5)
  (h4 : segment_length BE = 4)
  (h5 : segment_length AE = 3) :
  ( (AC + AB) / (AC - AB) ) * (segment_length (ED)) = 3 :=
sorry

end desired_expression_equals_3_l679_679784


namespace jennifer_min_study_hours_l679_679420

noncomputable def study_hours_for_average_score_of_90 (study_hours_first_test : ℝ) (score_first_test : ℝ) (desired_average : ℝ) : ℝ :=
let total_score_needed := 2 * desired_average in
let score_second_test := total_score_needed - score_first_test in
(score_first_test * study_hours_first_test) / score_second_test

theorem jennifer_min_study_hours :
  study_hours_for_average_score_of_90 5 85 90 = 4.5 :=
by
  sorry

end jennifer_min_study_hours_l679_679420


namespace collinearity_F_A_I_iff_AB_eq_AC_l679_679519

variables (A B C D E F I : Type)
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
variables [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F] [AffineSpace ℝ I]
variables (AB AC : ℝ)

-- Conditions
axiom inner_bisectors_intersect : collinear ℝ {B, I, C}
axiom A_parallel_BI_intersects_CI_at_D : ∀ P, P = A → parallel ℝ (line_through ℝ A B) (line_through ℝ P I) → collinear ℝ {C, I, D}
axiom A_parallel_CI_intersects_BI_at_E : ∀ Q, Q = A → parallel ℝ (line_through ℝ A C) (line_through ℝ Q I) → collinear ℝ {B, I, E}
axiom BD_CE_intersect_at_F : ∀ R S, (R = B ∧ S = D) ∨ (R = C ∧ S = E) → collinear ℝ {R, F, S}

-- Proof problem statement
theorem collinearity_F_A_I_iff_AB_eq_AC :
  collinear ℝ {F, A, I} ↔ AB = AC :=
sorry

end collinearity_F_A_I_iff_AB_eq_AC_l679_679519


namespace maximize_hat_percentage_sum_l679_679055

theorem maximize_hat_percentage_sum (hats caps : ℕ) (items_shelf1 items_shelf2 : ℕ) (hats_shelf1 hats_shelf2 : ℕ) :
  hats = 21 → caps = 18 → items_shelf1 = 20 → items_shelf2 = 19 → hats_shelf1 = 2 → hats_shelf2 = 19 →
  hats_shelf1 + items_shelf1 - hats_shelf1 = items_shelf1 → hats_shelf2 + items_shelf2 - hats_shelf2 = items_shelf2 → 
  (hats_shelf1 / items_shelf1 * 100 + hats_shelf2 / items_shelf2 * 100) = (2 / 20 * 100 + 19 / 19 * 100) :=
  by
    intros h_hat h_cap s1_cap s2_cap h1 h2 ps1 ps2 eq_a eq_b
    have S : (hats_shelf1 : ℝ) / items_shelf1 * 100 + (hats_shelf2 : ℝ) / items_shelf2 * 100 =
              (2 : ℝ) / 20 * 100 + (19 : ℝ) / 19 * 100 := sorry
    exact S

end maximize_hat_percentage_sum_l679_679055


namespace range_of_independent_variable_l679_679154

theorem range_of_independent_variable (x : ℝ) : x ≠ -3 ↔ ∃ y : ℝ, y = 1 / (x + 3) :=
by 
  -- Proof is omitted
  sorry

end range_of_independent_variable_l679_679154


namespace siding_cost_l679_679467

noncomputable def wall_area (width height : ℕ) : ℕ := width * height

noncomputable def triangular_area (base height : ℕ) : ℕ := (base * height) / 2

noncomputable def total_area (wall_area triangular_area : ℕ) : ℕ := 
  2 * wall_area + 2 * triangular_area

noncomputable def section_area (width height : ℕ) : ℕ := width * height

noncomputable def sections_required (total_area section_area : ℕ) : ℕ := 
  (total_area + section_area - 1) / section_area

noncomputable def total_cost (sections cost_per_section : ℕ) : ℕ :=
  sections * cost_per_section

theorem siding_cost 
     (wall_width wall_height tri_base tri_height sec_width sec_height sec_cost : ℕ)
     : total_cost 
         (sections_required
           (total_area
             (wall_area wall_width wall_height)
             (triangular_area tri_base tri_height))
           (section_area sec_width sec_height))
         sec_cost = 70 := 
by
  have wall_area_calc : wall_area wall_width wall_height = 60 := by
    exact calc
      wall_width * wall_height = 10 * 6 : by simp [wall_width, wall_height]
      ... = 60 : by norm_num

  have triangular_area_calc : triangular_area tri_base tri_height = 35 := by
    exact calc
      (tri_base * tri_height) / 2 = (10 * 7) / 2 : by simp [tri_base, tri_height]
      ... = 35 : by norm_num

  have total_area_calc : total_area 60 35 = 190 := by
    exact calc
      2 * 60 + 2 * 35 = 2 * 60 + 2 * 35 : by simp
      ... = 120 + 70 : by norm_num
      ... = 190 : by norm_num

  have section_area_calc : section_area sec_width sec_height = 150 := by
    exact calc
      sec_width * sec_height = 10 * 15 : by simp [sec_width, sec_height]
      ... = 150 : by norm_num

  have sections_required_calc : sections_required 190 150 = 2 := by
    exact calc
      (190 + 150 - 1) / 150 = 339 / 150 : by simp [total_area_calc, section_area_calc]
      ... = 2 : by norm_num

  have total_cost_calc : total_cost 2 35 = 70 := by
    exact calc
      2 * 35 = 70 : by norm_num

  exact total_cost_calc

end siding_cost_l679_679467


namespace polyhedron_odd_faces_l679_679587

/-- A convex polyhedron with 2003 vertices divided by a closed broken line passing each vertex exactly once has an odd number of faces with an odd number of sides in each part formed by this line. --/
theorem polyhedron_odd_faces (P : Polyhedron) (vertices : Fintype P.vertices) (h_vertices : Fintype.card P.vertices = 2003)
  (closed_line : P.Line) (h_closed : is_closed_broken_line closed_line)
  (h_passes : ∀v ∈ P.vertices, v ∈ closed_line.vertices) :
  ∀ part ∈ parts formed by closed_line (number_of_faces_with_odd_sides part).odd := sorry

end polyhedron_odd_faces_l679_679587


namespace number_of_students_at_perimeter_l679_679874

theorem number_of_students_at_perimeter (rows cols : ℕ) (h_rows : rows = 8) (h_cols : cols = 4) : 
  let top_row := cols,
      bottom_row := cols,
      middle_rows := (rows - 2) * 2
  in top_row + bottom_row + middle_rows = 20 :=
by
  intros
  sorry

end number_of_students_at_perimeter_l679_679874


namespace problem_divisible_by_1946_l679_679197

def F (n : ℕ) : ℤ := 1492 ^ n - 1770 ^ n - 1863 ^ n + 2141 ^ n

theorem problem_divisible_by_1946 
  (n : ℕ) 
  (hn : n ≤ 1945) : 
  1946 ∣ F n :=
sorry

end problem_divisible_by_1946_l679_679197


namespace number_of_valid_sequences_l679_679873

def transformation (square : Type) := 
| L : square -> square
| R : square -> square
| H : square -> square
| V : square -> square

def vertex := (ℝ × ℝ)

def A : vertex := (1, 1)
def B : vertex := (-1, 1)
def C : vertex := (-1, -1)
def D : vertex := (1, -1)

def transformations : Finset (transformation vertex) :=
  {[ 
    transformation.L,
    transformation.R,
    transformation.H,
    transformation.V 
  ]}.to_finset

def valid_transformations (seq : List (transformation vertex)) :=
  List.length seq = 20 ∧
  -- condition to check if the final position equals to the initial position after applying sequence of transformations
  (square application logic here)

noncomputable def count_valid_sequences : ℕ :=
  (Finset.filter valid_transformations (List.replicate 20 transformations)).card

theorem number_of_valid_sequences :
  count_valid_sequences = 286 :=
sorry

end number_of_valid_sequences_l679_679873


namespace simplify_expr_1_simplify_expr_2_l679_679502

theorem simplify_expr_1 (x y : ℝ) :
  12 * x - 6 * y + 3 * y - 24 * x = -12 * x - 3 * y :=
by
  sorry

theorem simplify_expr_2 (a b : ℝ) :
  (3 / 2) * (a^2 * b - 2 * (a * b^2)) - (1 / 2) * (a * b^2 - 4 * (a^2 * b)) + (a * b^2) / 2 = (7 / 2) * (a^2 * b) - 3 * (a * b^2) :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l679_679502


namespace probability_center_hexagon_l679_679589

-- Definitions and conditions

def isRegularHexagon (h : Type) : Prop := sorry

def dividedIntoRegionsByMidpoints (h : Type) : Prop := sorry

def isEquallyLikelyToLandAnywhere (h : Type) : Prop := sorry

-- Define the main structure
noncomputable def hexagon : Type := sorry

-- Main theorem
theorem probability_center_hexagon :
  isRegularHexagon hexagon →
  dividedIntoRegionsByMidpoints hexagon →
  isEquallyLikelyToLandAnywhere hexagon →
  let A_inner := (3 * real.sqrt 3) / 8
  let A_full := (3 * real.sqrt 3) / 2
  (A_inner / A_full) = (1 / 4) :=
by
  intros
  sorry

end probability_center_hexagon_l679_679589


namespace runner_time_to_cover_parade_l679_679043

variable (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ)

def relative_speed (runner_speed parade_speed : ℝ) := runner_speed - parade_speed

def time_to_run (distance speed : ℝ) := distance / speed

theorem runner_time_to_cover_parade
  (h1 : parade_length = 2)
  (h2 : parade_speed = 3)
  (h3 : runner_speed = 6) :
  time_to_run parade_length (relative_speed runner_speed parade_speed) * 60 = 40 :=
by
  sorry

end runner_time_to_cover_parade_l679_679043


namespace tan_half_sum_l679_679433

theorem tan_half_sum (p q : ℝ)
  (h1 : Real.cos p + Real.cos q = (1:ℝ)/3)
  (h2 : Real.sin p + Real.sin q = (8:ℝ)/17) :
  Real.tan ((p + q) / 2) = (24:ℝ)/17 := 
sorry

end tan_half_sum_l679_679433


namespace percentage_of_alcohol_in_first_vessel_l679_679998

variable (x : ℝ) -- percentage of alcohol in the first vessel in decimal form, i.e., x% is represented as x/100

-- conditions
variable (v1_capacity : ℝ := 2)
variable (v2_capacity : ℝ := 6)
variable (v2_alcohol_concentration : ℝ := 0.5)
variable (total_capacity : ℝ := 10)
variable (new_concentration : ℝ := 0.37)

theorem percentage_of_alcohol_in_first_vessel :
  (x / 100) * v1_capacity + v2_alcohol_concentration * v2_capacity = new_concentration * total_capacity -> x = 35 := 
by
  sorry

end percentage_of_alcohol_in_first_vessel_l679_679998


namespace num_of_arithmetic_and_harmonic_sets_l679_679992

def is_arithmetic (a b c : ℤ) : Prop :=
  a = (b + c) / 2 ∨ b = (a + c) / 2 ∨ c = (a + b) / 2

def is_harmonic (a b c : ℤ) : Prop :=
  a = (2 * b * c) / (b + c) ∨ b = (2 * a * c) / (a + c) ∨ c = (2 * a * b) / (a + b)

def is_valid_set (a b c : ℤ) : Prop :=
  -2011 < a ∧ a < 2011 ∧ -2011 < b ∧ b < 2011 ∧ -2011 < c ∧ c < 2011 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  is_arithmetic a b c ∧ is_harmonic a b c

theorem num_of_arithmetic_and_harmonic_sets :
  ∃! n, n = 1004 ∧
    (∃ (s : set (ℤ × ℤ × ℤ)), (∀ (a b c : ℤ), (a, b, c) ∈ s ↔ is_valid_set a b c) ∧ finset.card s = n) :=
sorry

end num_of_arithmetic_and_harmonic_sets_l679_679992


namespace graph_transform_lg_l679_679517

theorem graph_transform_lg :
  ∀ (x : ℝ), (∃ f : ℝ → ℝ, f(x) = 10^x) → 
  ∃ g : ℝ → ℝ, g(x + 3) = 10^x - 3 ∧ (λ y, g x = y) = λ y, ⟨y, x⟩ := 
by 
  sorry

end graph_transform_lg_l679_679517


namespace arithmetic_sequence_goal_l679_679799

open Nat

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
axiom h1 : a 2 + a 7 = 10

-- Goal
theorem arithmetic_sequence_goal (h : is_arithmetic_sequence a d) : 3 * a 4 + a 6 = 20 :=
sorry

end arithmetic_sequence_goal_l679_679799


namespace consecutive_nat_sum_l679_679687

theorem consecutive_nat_sum (n : ℕ) (h : n > 1) :
  (∃ a : ℕ, (finset.range n).sum (λ i, a + i) = 2016) ↔ n = 3 ∨ n = 7 ∨ n = 9 ∨ n = 21 ∨ n = 63 :=
by sorry

end consecutive_nat_sum_l679_679687


namespace truth_count_is_two_l679_679855

def Person := List String

def PierreTruth : Prop :=
  (Pierre, Qadr, Ratna, Sven, Tanya : Person) → 
  Pierre = "There is exactly one person telling the truth" →
  Qadr = "Pierre is not telling the truth" →
  Ratna = "Qadr is not telling the truth" →
  Sven = "Ratna is not telling the truth" →
  Tanya = "Sven is not telling the truth" →
  ∃ Q, ⟨Qadr, Sven⟩ → (Q ∈ [Pierre, Qadr, Ratna, Sven, Tanya]) ∧ 
  (Pierre ≠ ⟨2⟩) ∧ (Qadr = ⟨2⟩) ∧ (Ratna ≠ ⟨2⟩) ∧ (Sven = ⟨2⟩) ∧ (Tanya ≠ ⟨2⟩)

theorem truth_count_is_two : PierreTruth :=
begin
  -- placeholder to set up and analyze the problem
  sorry
end

end truth_count_is_two_l679_679855


namespace decimal_to_binary_125_l679_679926

theorem decimal_to_binary_125 : 
  (∃ (digits : list ℕ), (125 = digits.length - 1).sum (λ i, (2 ^ i) * digits.nth_le i sorry) ∧ digits = [1,1,1,1,1,0,1]) := 
sorry

end decimal_to_binary_125_l679_679926


namespace range_shifted_function_equal_l679_679332

-- Define the function domain and range
variable (f : ℝ → ℝ)
variable (a b : ℝ)
variable (h_range : ∀ y, y ∈ set.range f ↔ a ≤ y ∧ y ≤ b)

-- Theorem stating 
theorem range_shifted_function_equal :
  (∀ x, (f (x + a)) ∈ set.range f) → (set.range (λ x, f (x + a)) = set.Icc a b) :=
by
  intro h
  ext y
  simp
  split
  · intro hy
    obtain ⟨x, hx⟩ := hy
    specialize h x
    rw ← hx at h
    exact ⟨_, h⟩
  · intro hy
    obtain ⟨x, hx⟩ := hy
    use (x - a)
    rw ← hx
    specialize h (x - a)
    exact h

end range_shifted_function_equal_l679_679332


namespace distance_between_Sasha_and_Koyla_is_19m_l679_679480

-- Defining variables for speeds
variables (v_S v_L v_K : ℝ)
-- Additional conditions
variables (h1 : ∃ (t : ℝ), t > 0 ∧ 100 = v_S * t) -- Sasha finishes the race in time t
variables (h2 : 90 = v_L * (100 / v_S)) -- Lyosha is 10 meters behind when Sasha finishes
variables (h3 : v_K = 0.9 * v_L) -- Kolya's speed is 0.9 times Lyosha's speed

theorem distance_between_Sasha_and_Koyla_is_19m :
  ∀ (v_S v_L v_K : ℝ), (h1 : ∃ t > 0, 100 = v_S * t) → (h2 : 90 = v_L * (100 / v_S)) → (h3 : v_K = 0.9 * v_L)  →
  (100 - (0.81 * 100)) = 19 :=
by
  intros v_S v_L v_K h1 h2 h3
  sorry

end distance_between_Sasha_and_Koyla_is_19m_l679_679480


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679009

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679009


namespace expenditure_recording_l679_679382

def income : ℕ := 200
def recorded_income : ℤ := 200
def expenditure (e : ℕ) : ℤ := -(e : ℤ)

theorem expenditure_recording (e : ℕ) :
  expenditure 150 = -150 := by
  sorry

end expenditure_recording_l679_679382


namespace new_ratio_of_gold_to_silver_l679_679790

theorem new_ratio_of_gold_to_silver (G S : ℕ) (h1 : G = S / 3) (h2 : G + S + 15 = 135) :
  (G + 15) : S = 1 : 2 := by
sorry

end new_ratio_of_gold_to_silver_l679_679790


namespace telepathically_linked_probability_correct_l679_679856

def number_set := {1, 2, 3, 4, 5, 6}

def telepathically_linked (a b : ℕ) : Prop := |a - b| ≤ 1

def telepathically_linked_pairs : set (ℕ × ℕ) := 
  { (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,3), (3,4),
    (4,3), (4,4), (4,5), (5,4), (5,5), (5,6), (6,5), (6,6) }

def total_events : ℕ := 36

def telepathically_linked_probability : ℚ := 16 / 36

theorem telepathically_linked_probability_correct : 
  telepathically_linked_probability = 4 / 9 := 
by 
  sorry

end telepathically_linked_probability_correct_l679_679856


namespace distinct_sphere_configurations_l679_679065

theorem distinct_sphere_configurations (p1 p2 p3 : Plane) (s1 : Sphere) : 
  ∃ (n : ℕ), n = 16 ∧ unique_sphere_placements p1 p2 p3 s1 n :=
by sorry

end distinct_sphere_configurations_l679_679065


namespace power_modulo_l679_679936

theorem power_modulo (h : 3 ^ 4 ≡ 1 [MOD 10]) : 3 ^ 2023 ≡ 7 [MOD 10] :=
by
  sorry

end power_modulo_l679_679936


namespace race_distance_between_Sasha_and_Kolya_l679_679487

theorem race_distance_between_Sasha_and_Kolya
  (vS vL vK : ℝ)
  (h1 : vK = 0.9 * vL)
  (h2 : ∀ t_S, 100 = vS * t_S → vL * t_S = 90)
  (h3 : ∀ t_L, 100 = vL * t_L → vK * t_L = 90)
  : ∀ t_S, 100 = vS * t_S → (100 - vK * t_S) = 19 :=
by
  sorry


end race_distance_between_Sasha_and_Kolya_l679_679487


namespace jon_coffee_spending_in_april_l679_679818

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l679_679818


namespace complex_numbers_xyz_l679_679093

theorem complex_numbers_xyz (x y z : ℂ) (h1 : x * y + 5 * y = -20) (h2 : y * z + 5 * z = -20) (h3 : z * x + 5 * x = -20) :
  x * y * z = 100 :=
sorry

end complex_numbers_xyz_l679_679093


namespace ratio_sheep_to_horses_l679_679633

theorem ratio_sheep_to_horses 
  (horse_food_per_day : ℕ) 
  (total_horse_food : ℕ) 
  (num_sheep : ℕ) 
  (H1 : horse_food_per_day = 230) 
  (H2 : total_horse_food = 12880) 
  (H3 : num_sheep = 48) 
  : (num_sheep : ℚ) / (total_horse_food / horse_food_per_day : ℚ) = 6 / 7
  :=
by
  sorry

end ratio_sheep_to_horses_l679_679633


namespace real_roots_determinant_l679_679432

variable (a b c k : ℝ)
variable (k_pos : k > 0)
variable (a_nonzero : a ≠ 0) 
variable (b_nonzero : b ≠ 0)
variable (c_nonzero : c ≠ 0)
variable (k_nonzero : k ≠ 0)

theorem real_roots_determinant : 
  ∃! x : ℝ, (Matrix.det ![![x, k * c, -k * b], ![-k * c, x, k * a], ![k * b, -k * a, x]] = 0) :=
sorry

end real_roots_determinant_l679_679432


namespace angle_ABD_30_degrees_l679_679414

theorem angle_ABD_30_degrees (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB BD : ℝ) (angle_DBC : ℝ)
  (h1 : BD = AB * (Real.sqrt 3 / 2))
  (h2 : angle_DBC = 90) : 
  ∃ angle_ABD, angle_ABD = 30 :=
by
  sorry

end angle_ABD_30_degrees_l679_679414


namespace anna_score_below_90_no_A_l679_679103

def score_implies_grade (score : ℝ) : Prop :=
  score > 90 → true

theorem anna_score_below_90_no_A (score : ℝ) (A_grade : Prop) (h : score_implies_grade score) :
  score < 90 → ¬ A_grade :=
by sorry

end anna_score_below_90_no_A_l679_679103


namespace age_difference_is_six_l679_679879

theorem age_difference_is_six 
  (k l m : ℝ) 
  (P M Mo N O : ℝ)
  (h1 : P = 3 * k)
  (h2 : M = 5 * k) 
  (h3 : M = 3 * l) 
  (h4 : Mo = 5 * l) 
  (h5 : N = 7 * l) 
  (h6 : Mo = 4 * m) 
  (h7 : N = 3 * m) 
  (h8 : O = 2 * m) 
  (h_sum : P + M + Mo + N + O = 146) :
  |P - O| ≈ 6 :=
by
  sorry

end age_difference_is_six_l679_679879


namespace binomial_abs_sum_eq_l679_679692

theorem binomial_abs_sum_eq :
  let p := (1 - 2*x)^7
  ∃ (a : Fin 8 → ℤ), p = ∑ i, a i * x^i ∧ (∑ i, |a i|) = 2187 :=
by
  sorry

end binomial_abs_sum_eq_l679_679692


namespace cost_price_per_meter_is_correct_l679_679955

-- Defining the conditions
def N : ℕ := 85
def SP : ℝ := 8925
def P : ℝ := 20

-- Statement of the problem
theorem cost_price_per_meter_is_correct : 
  let total_profit := P * N in
  let total_cost_price := SP - total_profit in
  let CP := total_cost_price / N in
  CP = 85 :=
by
  sorry

end cost_price_per_meter_is_correct_l679_679955


namespace jake_total_distance_l679_679417

noncomputable def jake_rate : ℝ := 4 -- Jake's walking rate in miles per hour
noncomputable def total_time : ℝ := 2 -- Jake's total walking time in hours
noncomputable def break_time : ℝ := 0.5 -- Jake's break time in hours

theorem jake_total_distance :
  jake_rate * (total_time - break_time) = 6 :=
by
  sorry

end jake_total_distance_l679_679417


namespace cube_root_floor_product_l679_679278

theorem cube_root_floor_product :
  (∏ i in finset.range 1006, ⌊(3 + 2 * i)^(1/3)⌋) /
  (∏ i in finset.range 1006, ⌊(4 + 2 * i)^(1/3)⌋) = 1 / 13 :=
sorry

end cube_root_floor_product_l679_679278


namespace find_f_alpha_find_max_min_f_l679_679358

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * (sqrt 3 * cos x + sin x) - 2

theorem find_f_alpha (α : ℝ) (h : tan α = -sqrt 3 / 3) : f α = -3 :=
sorry

theorem find_max_min_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : 
  -2 ≤ f x ∧ f x ≤ 1 :=
sorry

end find_f_alpha_find_max_min_f_l679_679358


namespace minimum_i_l679_679685

def divides_plane (n : ℕ) (x : ℕ) : Prop :=
  ∃ (lines : list (set (ℝ × ℝ))), 
    (lines.length = n) ∧ (¬ ∃ (p : ℝ × ℝ), ∀ l ∈ lines, p ∈ l) ∧ 
    (number_of_regions lines = x)

def S (n : ℕ) : set ℕ :=
  { x | divides_plane n x }

theorem minimum_i (i : ℕ) (h : ∀ j < i, |S j| < 4) : |S i| ≥ 4 →
  i = 4 :=
sorry

end minimum_i_l679_679685


namespace supremum_alpha_n_l679_679961

noncomputable def alpha_n (n : ℕ) : ℝ :=
  Inf { abs (a - b * real.sqrt 3) | a b : ℕ, a + b = n }

theorem supremum_alpha_n :
  is_lub {alpha_n n | n : ℕ} ((real.sqrt 3 + 1) / 2) :=
sorry

end supremum_alpha_n_l679_679961


namespace dave_tickets_left_l679_679637

theorem dave_tickets_left (initial_tickets spent_tickets : ℕ) (h_initial : initial_tickets = 98) (h_spent : spent_tickets = 43) : initial_tickets - spent_tickets = 55 :=
by
  rw [h_initial, h_spent]
  norm_num

end dave_tickets_left_l679_679637


namespace decimal_to_binary_125_l679_679930

theorem decimal_to_binary_125 : ∃ (b : ℕ), 125 = b ∧ b = (1 * 2^6) + (1 * 2^5) + (1 * 2^4) + (1 * 2^3) + (1 * 2^2) + (0 * 2^1) + (1 * 2^0) :=
by {
  use 0b1111101,
  exact ⟨rfl, rfl⟩,
}

end decimal_to_binary_125_l679_679930


namespace min_even_integers_six_l679_679909

theorem min_even_integers_six (x y a b m n : ℤ) 
  (h1 : x + y = 30) 
  (h2 : x + y + a + b = 50) 
  (h3 : x + y + a + b + m + n = 70) 
  (hm_even : Even m) 
  (hn_even: Even n) : 
  ∃ k, (0 ≤ k ∧ k ≤ 6) ∧ (∀ e, (e = m ∨ e = n) → ∃ j, (j = 2)) :=
by
  sorry

end min_even_integers_six_l679_679909


namespace equitable_polygon_not_centrally_symmetric_l679_679988

noncomputable def equitable_polygon : Prop :=
  let vertices := [(1, 0), (0, 1), (0, 5), (5, 0), (7, 0), (0, -7), (-7, 0), (-√84, 0), (0, √84), (0, 6), (-6, 0), (-5, 0), (0, -5), (0, -1)] in
  let is_equitable := ∀ line : ℝ × ℝ, ∃ region1 region2 : ℕ, (area_through_origin line vertices = (region1, region2)) ∧ (region1 = region2) in
  let is_not_centrally_symmetric := ¬(∀ v ∈ vertices, ∃ w ∈ vertices, (v.1 = -w.1) ∧ (v.2 = -w.2)) in
  is_equitable ∧ is_not_centrally_symmetric

theorem equitable_polygon_not_centrally_symmetric : equitable_polygon :=
  sorry

end equitable_polygon_not_centrally_symmetric_l679_679988


namespace zero_count_f_l679_679892

def f (x : ℝ) := Real.sin (Real.pi * Real.cos x)

theorem zero_count_f :
  ∃ count : ℕ, count = 5 ∧
  ∀ y ∈ Icc 0 (2 * Real.pi), f y = 0 → count = 5 := sorry

end zero_count_f_l679_679892


namespace max_combinations_at_5_l679_679101

theorem max_combinations_at_5 :
  (∃ f : ℕ → ℕ, ∀ n (hn : 2 ≤ n ∧ n ≤ 6), 
    f 1 = 1 ∧
    f n = n^6 - ∑ i in finset.range (n - 1), finset.choose n i * f (i + 1)) →
  ∀ x ∈ {1, 2, 3, 4, 5, 6}, 
    f 5 ≥ f x :=
sorry

end max_combinations_at_5_l679_679101


namespace max_length_MQ_l679_679354

-- Define the conditions
def A (a : ℝ) : ℝ × ℝ := (a, 0)
def circle : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def is_on_circle (P : ℝ × ℝ) : Prop := P ∈ circle
def symmetrical_counterpart (P A M : ℝ × ℝ) : Prop := M = (2 * A.1 - P.1, 2 * A.2 - P.2)
def perpendicular (P Q O : ℝ × ℝ) : Prop := (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 0

-- Define the distance formula
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Main theorem statement
theorem max_length_MQ (a : ℝ) (ha : a > 0) (P Q M : ℝ × ℝ) (hP : is_on_circle P) (hQ : is_on_circle Q) 
  (hPerpendicular : perpendicular P Q (0, 0)) (hSym : symmetrical_counterpart P (A a) M) :
  ∃ (θ : ℝ), distance M Q = real.sqrt (4 * a^2 + 4 * real.sqrt 2 * a + 2) :=
sorry

end max_length_MQ_l679_679354


namespace min_distance_l679_679312

noncomputable def min_distance_to_x_axis (x y : ℝ) (hx : x > 2) (hcurve : x^2 - x * y + 2 * y + 1 = 0) : ℝ :=
  4 + 2 * Real.sqrt 5

theorem min_distance (x y : ℝ) (hx : x > 2) (hcurve : x^2 - x * y + 2 * y + 1 = 0) :
  ∃ mx : ℝ, mx = min_distance_to_x_axis x y hx hcurve :=
begin
  use 4 + 2 * Real.sqrt 5,
  exact rfl,
end

end min_distance_l679_679312


namespace interval_as_set_l679_679889

theorem interval_as_set :
  (set_of (λ x : ℝ, -3 < x ∧ x ≤ 2)) = {x : ℝ | -3 < x ∧ x ≤ 2} :=
sorry

end interval_as_set_l679_679889


namespace hyperbola_eccentricity_l679_679598

-- Define the hyperbola and its properties
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the focus of the hyperbola
def focus (a b : ℝ) : (ℝ × ℝ) := (real.sqrt (a^2 + b^2), 0)

-- Define the equation of the asymptote
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = (b / a) * x

-- Define the perpendicular bisector condition
def perpendicular_bisector_condition (a b : ℝ) : Prop :=
  let F := focus a b in
  let D := (real.sqrt (a^2 + b^2) / 2, (b * real.sqrt (a^2 + b^2)) / (2 * a)) in
  let OD_slope := D.2 / D.1 in
  let DF_slope := ((b * real.sqrt (a^2 + b^2)) / (2 * a)) / (real.sqrt (a^2 + b^2) / 2 - real.sqrt (a^2 + b^2)) in
  DF_slope * OD_slope = -1

-- Specify the main theorem statement for proving the required eccentricity
theorem hyperbola_eccentricity (a b : ℝ) (h_perp_bisector : perpendicular_bisector_condition a b) : real.sqrt (a^2 + b^2) / a = real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l679_679598


namespace boy_reaches_early_l679_679552

theorem boy_reaches_early (usual_rate new_rate : ℝ) (Usual_Time New_Time : ℕ) 
  (Hrate : new_rate = 9/8 * usual_rate) (Htime : Usual_Time = 36) :
  New_Time = 32 → Usual_Time - New_Time = 4 :=
by
  intros
  subst_vars
  sorry

end boy_reaches_early_l679_679552


namespace tickets_for_one_ride_l679_679418

-- Definitions
def tickets_for_roller_coaster := Nat
def tickets_for_giant_slide : Nat := 3
def total_tickets_needed (x : tickets_for_roller_coaster) : Prop := 
  7 * x + 4 * tickets_for_giant_slide = 47

-- Theorem statement
theorem tickets_for_one_ride (x : tickets_for_roller_coaster) (h : total_tickets_needed x) : x = 5 :=
  sorry

end tickets_for_one_ride_l679_679418


namespace no_valid_coloring_l679_679415

theorem no_valid_coloring (colors : Fin 4 → Prop) (board : Fin 5 → Fin 5 → Fin 4) :
  (∀ i j : Fin 5, ∃ c1 c2 c3 : Fin 4, 
    (c1 ≠ c2) ∧ (c2 ≠ c3) ∧ (c1 ≠ c3) ∧ 
    (board i j = c1 ∨ board i j = c2 ∨ board i j = c3)) → False :=
by
  sorry

end no_valid_coloring_l679_679415


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679004

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679004


namespace repeating_decimal_as_fraction_l679_679302

theorem repeating_decimal_as_fraction : ∀ x : ℝ, (x = 0.353535...) → (99 * x = 35) → x = 35 / 99 := by
  intros x h1 h2
  sorry

end repeating_decimal_as_fraction_l679_679302


namespace unique_positive_b_one_solution_l679_679678

theorem unique_positive_b_one_solution (c : ℝ) : 
  (∃! b > 0, ∃ x : ℝ, x^2 + 2 * (b + 1 / b) * x + c = 0 ∧ ∃ y : ℝ, y^2 + 2 * (b + 1 / b) * y + c = 0 ∧ x = y) ↔ c = 4 := 
by
  sorry

end unique_positive_b_one_solution_l679_679678


namespace fraction_simplify_l679_679456

theorem fraction_simplify (x : ℝ) (hx : x ≠ 1) (hx_ne_1 : x ≠ -1) :
  (x^2 - 1) / (x^2 - 2 * x + 1) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_simplify_l679_679456


namespace fraction_of_120_l679_679187

theorem fraction_of_120 : (1/3 * 1/4 * 1/6) * 120 = 5/3 := 
by 
  have h : (1/3 * 1/4 * 1/6) = 1/72,
  { 
    calc (1/3 * 1/4 * 1/6) = (1/3) * (1/4) * (1/6) : by rw mul_assoc (1/3) (1/4) (1/6)
                          ... = 1/(3*4*6)          : by ring
                          ... = 1/72               : by norm_num }
  have h2 : (1/72) * 120 = 120/72,
  {
    calc (1/72) * 120 = 120 / 72 : by field_simp
  }
  calc (1/3 * 1/4 * 1/6) * 120 = (1/72) * 120 : by rw h
                           ... = 120/72        : by rw h2
                           ... = 5/3           : by norm_num

end fraction_of_120_l679_679187


namespace equal_angles_in_isosceles_right_triangle_l679_679061

variable {A B C M P : Point}
variable {triangle : IsoscelesRightTriangle ABC}
variable {A90 : angle A = 90}
variable {midpointM : Midpoint M A B}
variable {perpendicularLine : PerpendicularLine A (LineThroughPoints C M) P BC}

theorem equal_angles_in_isosceles_right_triangle 
  (h₁ : IsoscelesRightTriangle ABC)
  (h₂ : angle A = 90)
  (h₃ : Midpoint M A B)
  (h₄ : PerpendicularLine A (LineThroughPoints C M) P BC) : 
  angle (AngleFromPoints A M C) = angle (AngleFromPoints B M P) :=
sorry

end equal_angles_in_isosceles_right_triangle_l679_679061


namespace problem_statement_l679_679084
-- Import the entire Mathlib library to include all necessary definitions

-- Definition for the problem
variable (G : Type) [AddCommGroup G] [Fintype G]
variable (A : Finset G) 
variable (c : ℝ) (k : ℕ) (h1 : 1 ≤ c) (h2 : 2 ≤ k)

-- Main theorem statement
theorem problem_statement (h : (A.card : ℝ) ≤ c * A.card) :
  ((A + ... + A).card : ℝ) ≤ (c ^ k) * A.card :=
sorry

end problem_statement_l679_679084


namespace coin_flip_probability_l679_679449

theorem coin_flip_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)
  (h_win : ∑' n, (1 - p) ^ n * p ^ (n + 1) = 1 / 2) :
  p = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end coin_flip_probability_l679_679449


namespace units_digit_of_17_pow_3_mul_24_l679_679659

def unit_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_3_mul_24 :
  unit_digit (17^3 * 24) = 2 :=
by
  sorry

end units_digit_of_17_pow_3_mul_24_l679_679659


namespace hikers_trip_duration_l679_679592

theorem hikers_trip_duration :
  ∃ d : ℕ, (∀ up_rate : ℕ, up_rate = 6 → (∀ down_rate : ℕ, down_rate = up_rate * 3 / 2 → 
  (∀ time_equal : ℕ, time_equal = d → (∀ down_distance : ℕ, down_distance = 18 → 
  down_rate * d = down_distance))) → d = 2 :=
begin
  sorry
end

end hikers_trip_duration_l679_679592


namespace games_left_after_sale_l679_679782

def initial_games : ℕ := 150
def sold_first_week (n : ℕ) : ℕ := n * 60 / 100
def after_first_week (n : ℕ) : ℕ := n - sold_first_week n
def after_delivery (n : ℕ) : ℕ := 2 * after_first_week n
def sold_weekend (n : ℕ) : ℕ := after_delivery n * 45 / 100
def left_after_weekend (n : ℕ) : ℕ := after_delivery n - sold_weekend n

theorem games_left_after_sale (initial_games = 150) : 
    left_after_weekend initial_games = 66 :=
sorry

end games_left_after_sale_l679_679782


namespace value_of_c_l679_679518

theorem value_of_c (c : ℝ) : 
  let midpoint := (1 + 5 : ℝ) / 2, (6 + 12 : ℝ) / 2 in
  2 * midpoint.1 + midpoint.2 = c →
  (1 + 5) / 2 = 3 ∧ (6 + 12) / 2 = 9 ∧ 2 * 3 + 9 = c :=
by
  sorry

end value_of_c_l679_679518


namespace find_f4_l679_679735

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def equilibrium_condition (f4 : ℝ × ℝ) : Prop :=
  f1 + f2 + f3 + f4 = (0, 0)

-- Statement that needs to be proven
theorem find_f4 : ∃ (f4 : ℝ × ℝ), equilibrium_condition f4 :=
  by
  use (1, 2)
  sorry

end find_f4_l679_679735


namespace three_powers_in_two_digit_range_l679_679001

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l679_679001


namespace max_vertex_value_l679_679577

-- Define the initial state of the cube
def initial_cube_state : ℕ → ℕ := 
  λ v, v  -- Assume each vertex v has a distinct label v (1 to 8)

-- Define the transformations for a black card and a white card
def black_card_transform (cube : ℕ → ℕ) : ℕ → ℕ :=
  λ v, cube (v % 8 + 1) + cube ((v + 1) % 8 + 1) + cube ((v + 2) % 8 + 1)

def white_card_transform (cube : ℕ → ℕ) : ℕ → ℕ :=
  λ v, cube (v % 8 + 2) + cube ((v + 2) % 8 + 2) + cube ((v + 4) % 8 + 2)

-- Applying the sequence of 4 black and 4 white card draws
def final_cube_state (cube : ℕ → ℕ) : ℕ → ℕ :=
  (white_card_transform ∘ black_card_transform ∘ white_card_transform ∘ black_card_transform ∘
  white_card_transform ∘ black_card_transform ∘ white_card_transform ∘ black_card_transform) cube

-- Prove the maximum possible value on any vertex after all transformations is 42648
theorem max_vertex_value : 
  ∀ (vertex : ℕ) (initial_labels : ℕ → ℕ), (final_cube_state initial_labels vertex) ≤ 42648 := 
sorry

end max_vertex_value_l679_679577


namespace one_fourth_more_equals_thirty_percent_less_l679_679173

theorem one_fourth_more_equals_thirty_percent_less :
  ∃ n : ℝ, 80 - 0.30 * 80 = (5 / 4) * n ∧ n = 44.8 :=
by
  sorry

end one_fourth_more_equals_thirty_percent_less_l679_679173


namespace number_is_correct_l679_679946

noncomputable def number_satisfies_condition : ℚ :=
  7200 / 31

theorem number_is_correct (x : ℚ) (h : x / 5 = 40 + x / 6) : 
  x = 232.2581 :=
begin
  sorry
end

end number_is_correct_l679_679946


namespace extreme_value_point_of_f_l679_679713

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume the definition of f that derives this f'

def f' (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem extreme_value_point_of_f : (∃ x : ℝ, x = -2 ∧ ∀ y : ℝ, y ≠ -2 → f' y < 0) := sorry

end extreme_value_point_of_f_l679_679713


namespace hexagon_perpendicularity_l679_679880

noncomputable def common_perpendicular (a b c : ℝ^3) : ℝ^3 := cross a (cross b c)

theorem hexagon_perpendicularity {a b c a1 b1 c1 : ℝ^3}
  (h1 : a1 = cross b c)
  (h2 : b1 = cross c a)
  (h3 : c1 = cross a b) :
  common_perpendicular a b c + common_perpendicular b c a + common_perpendicular c a b = 0 :=
by
  sorry

end hexagon_perpendicularity_l679_679880


namespace no_blue_frogs_l679_679872

def truth_teller (x : Type) : Prop := sorry
def liar (x : Type) : Prop := sorry
def blue_frog (x : Type) : Prop := sorry
def red_frog (x : Type) : Prop := sorry
def frog (x : Type) := ∃ (b : Prop), truth_teller x ∨ liar x ∧ (blue_frog x ∨ red_frog x)

axiom Bre_statement : ∀ (x : Type), frog x -> ¬ (blue_frog x)
axiom Ke_statement : liar Bre ∧ blue_frog Bre
axiom Keks_statement : liar Bre ∧ red_frog Bre

theorem no_blue_frogs (x : Type) [frog x] : ¬ (blue_frog x) :=
by
  sorry

end no_blue_frogs_l679_679872


namespace merchant_loss_l679_679851

theorem merchant_loss
  (sp : ℝ)
  (profit_percent: ℝ)
  (loss_percent:  ℝ)
  (sp1 : ℝ)
  (sp2 : ℝ)
  (cp1 cp2 : ℝ)
  (net_loss : ℝ) :
  
  sp = 990 → 
  profit_percent = 0.1 → 
  loss_percent = 0.1 →
  sp1 = sp → 
  sp2 = sp → 
  cp1 = sp1 / (1 + profit_percent) →
  cp2 = sp2 / (1 - loss_percent) →
  net_loss = (cp2 - sp2) - (sp1 - cp1) →
  net_loss = 20 :=
by 
  intros _ _ _ _ _ _ _ _ 
  -- placeholders for intros to bind variables
  sorry

end merchant_loss_l679_679851


namespace matrix_product_l679_679280

noncomputable def R : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]]

noncomputable def S (d c b : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[d^2, dc, db],
    [dc , c^2, cb],
    [db , cb, b^2]]

theorem matrix_product (d c b : ℝ) :
  R ⬝ (S d c b) = ![[-dc, -c^2, -cb],
                      [d^2, dc, db],
                      [db , cb, b^2]] :=
by
  sorry

end matrix_product_l679_679280


namespace eval_expression_result_l679_679669

def eval_expression (a b c d : ℝ) : ℝ := -((a / b) * c - 50 + d^2)

theorem eval_expression_result : eval_expression 16 4 6 5 = 1 := by
  sorry

end eval_expression_result_l679_679669


namespace minimum_value_of_f_l679_679439

noncomputable def f (a b : ℝ) : ℝ :=
  3 * a^2 + 3 * b^2 + 1 / (a + b)^2 + 4 / (a^2 * b^2)

theorem minimum_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ m, m ∈ (set.range (λ ab : ℝ × ℝ, f ab.1 ab.2)) ∧ m = 6 :=
sorry

end minimum_value_of_f_l679_679439


namespace ticket_sales_ratio_l679_679664

theorem ticket_sales_ratio (tickets_first_week_reduced : ℕ) (total_tickets : ℕ) (tickets_full_price_total : ℕ)
  (tickets_remaining_reduced : ℕ) (tickets_remaining_full : ℕ)
  (h1 : tickets_first_week_reduced = 5400)
  (h2 : total_tickets = 25200)
  (h3 : tickets_full_price_total = 16500)
  (h4 : tickets_remaining_reduced = total_tickets - tickets_first_week_reduced - tickets_remaining_full)
  (h5 : tickets_remaining_full = tickets_full_price_total) :
  tickets_remaining_full = tickets_remaining_reduced :=
by {
  intros,
  sorry
}

end ticket_sales_ratio_l679_679664


namespace unique_prime_root_k_l679_679640

theorem unique_prime_root_k :
  ∃! k : ℕ, (∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ p + q = 63 ∧ p * q = k) :=
sorry

end unique_prime_root_k_l679_679640


namespace final_tank_volume_l679_679616

-- Definitions based on the conditions
def initially_liters : ℕ := 6000
def evaporated_liters : ℕ := 2000
def drained_by_bob : ℕ := 3500
def rain_minutes : ℕ := 30
def rain_interval : ℕ := 10
def rain_rate : ℕ := 350

-- Theorem statement
theorem final_tank_volume:
  let remaining_water := initially_liters - evaporated_liters - drained_by_bob,
      rain_cycles := rain_minutes / rain_interval,
      added_by_rain := rain_cycles * rain_rate,
      final_volume := remaining_water + added_by_rain
  in final_volume = 1550 := by {
  sorry
}

end final_tank_volume_l679_679616


namespace total_cakes_served_l679_679605

theorem total_cakes_served (l : ℝ) (p : ℝ) (s : ℝ) (total_cakes_served_today : ℝ) :
  l = 48.5 → p = 0.6225 → s = 95 → total_cakes_served_today = 108 :=
by
  intros hl hp hs
  sorry

end total_cakes_served_l679_679605


namespace min_value_u_l679_679437

theorem min_value_u (x y : ℝ) 
  (h : x^2 - 2*x*y + y^2 - real.sqrt 2 * x - real.sqrt 2 * y + 6 = 0) : 
  ∃ (u : ℝ), u = x + y ∧ u = 3 * real.sqrt 2 :=
sorry

end min_value_u_l679_679437


namespace radius_of_inscribed_circle_in_rhombus_l679_679935

noncomputable def radius_of_inscribed_circle (d₁ d₂ : ℕ) : ℝ :=
  (d₁ * d₂) / (2 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

theorem radius_of_inscribed_circle_in_rhombus :
  radius_of_inscribed_circle 8 18 = 36 / Real.sqrt 97 :=
by
  -- Skip the detailed proof steps
  sorry

end radius_of_inscribed_circle_in_rhombus_l679_679935


namespace decimal_to_binary_125_l679_679927

theorem decimal_to_binary_125 : 
  (∃ (digits : list ℕ), (125 = digits.length - 1).sum (λ i, (2 ^ i) * digits.nth_le i sorry) ∧ digits = [1,1,1,1,1,0,1]) := 
sorry

end decimal_to_binary_125_l679_679927


namespace product_price_interval_l679_679054

def is_too_high (price guess : ℕ) : Prop := guess > price
def is_too_low  (price guess : ℕ) : Prop := guess < price

theorem product_price_interval 
    (price : ℕ)
    (h1 : is_too_high price 2000)
    (h2 : is_too_low price 1000)
    (h3 : is_too_high price 1500)
    (h4 : is_too_low price 1250)
    (h5 : is_too_low price 1375) :
    1375 < price ∧ price < 1500 :=
    sorry

end product_price_interval_l679_679054


namespace driving_time_first_part_is_one_hour_l679_679160

noncomputable def time_to_drive_first_part 
  (total_distance : ℕ)
  (fraction_first_part : ℚ)
  (lunch_duration : ℕ)
  (total_time : ℕ) 
  (same_speed : Prop) : ℕ :=
  let first_part_distance := total_distance * fraction_first_part in
  let remaining_distance := total_distance - first_part_distance in
  let second_part_time := total_time - lunch_duration - first_part_distance / (first_part_distance / total_time) in
  first_part_distance / (first_part_distance / total_time)

theorem driving_time_first_part_is_one_hour 
  (total_distance : ℕ)
  (fraction_first_part : ℚ)
  (lunch_duration : ℕ)
  (total_time : ℕ) 
  (same_speed : Prop) (h1: total_distance = 200) (h2: fraction_first_part = 1/4)
  (h3: lunch_duration = 1) (h4: total_time = 5) (h5: same_speed) :
  time_to_drive_first_part total_distance fraction_first_part lunch_duration total_time same_speed = 1 := 
by 
  sorry

end driving_time_first_part_is_one_hour_l679_679160


namespace cube_volume_l679_679542

-- Definition of the cube's volume given the side area.
noncomputable def volume_of_cube_with_side_area (A : ℝ) : ℝ :=
  let side_length := Real.sqrt A
  in side_length * side_length * side_length

-- Theorem statement
theorem cube_volume (A : ℝ) (volume : ℝ) (h : A = 64) : volume = 512 :=
by
  let side_length := Real.sqrt A
  have volume_calc : volume = side_length * side_length * side_length := sorry
  have side_length_val : side_length = 8 := sorry
  rw [side_length_val, volume_calc]
  norm_num
  assumption

end cube_volume_l679_679542


namespace incircle_radius_correct_l679_679546

noncomputable def radius_of_incircle_of_triangle_def
  (D E F : ℝ × ℝ)
  (h_right : ∃ A B, (B - A).angle (D - A) = π / 2)
  (h_angle_D : ∃ α : ℝ, α = π / 3)
  (h_DF : (DF : ℝ) = 12) : ℝ :=
6 * (Real.sqrt 3 - 1)

theorem incircle_radius_correct
  (D E F : ℝ × ℝ)
  (h_right : ∃ A B, (B - A).angle (D - A) = π / 2)
  (h_angle_D : ∃ α : ℝ, α = π / 3)
  (h_DF : (DF : ℝ) = 12) :
  radius_of_incircle_of_triangle_def D E F h_right h_angle_D h_DF = 6 * (Real.sqrt 3 - 1) :=
sorry

end incircle_radius_correct_l679_679546


namespace count_ways_to_sum_2016_with_2s_and_3s_l679_679034

theorem count_ways_to_sum_2016_with_2s_and_3s :
  (∑ i in finset.range 337, (∑ j in finset.range (337 - i), 2 + 3)) = 2016 :=
by {
  sorry
}

end count_ways_to_sum_2016_with_2s_and_3s_l679_679034


namespace length_of_PW_l679_679126

noncomputable def PW_length (PQ RS QR : ℝ) (UR PR PU R: ℝ) : ℝ :=
  let side_length := 4
  let diagonal := real.sqrt (side_length^2 + side_length^2)
  let r := 1
  let PX := real.sqrt (PU^2 - r^2)
  let angle_UPX := real.arcsin (r / PU) + (real.pi / 4)
  let XW := r /  real.tan angle_UPX
  PX + XW

theorem length_of_PW :
  let PQ := 4
  let RS := 4
  let QR := 4
  let PR := 4 * (real.sqrt 2)
  let UR := real.sqrt 2
  let PU := 3 * (real.sqrt 2)
  PW_length PQ RS QR UR PR PU = 4.685 :=
by
  sorry

end length_of_PW_l679_679126


namespace find_numbers_l679_679341

theorem find_numbers 
  (a b c d : ℝ)
  (h1 : b / c = c / a)
  (h2 : a + b + c = 19)
  (h3 : b - c = c - d)
  (h4 : b + c + d = 12) :
  (a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2) :=
sorry

end find_numbers_l679_679341


namespace positive_difference_between_sums_l679_679422

def sum_integers_1_to_60 : Nat :=
  (60 * 61) / 2

def round_to_nearest_10 (n : Nat) : Nat :=
  if n % 10 < 5 then (n / 10) * 10 else ((n / 10) + 1) * 10

def sum_rounded_integers_1_to_60 : Nat :=
  List.sum (List.map round_to_nearest_10 (List.range (60 + 1)))

theorem positive_difference_between_sums :
  abs (sum_integers_1_to_60 - sum_rounded_integers_1_to_60) = 1530 :=
by
  sorry

end positive_difference_between_sums_l679_679422


namespace shaded_area_of_sector_l679_679791

theorem shaded_area_of_sector 
  (EF FH : ℝ) 
  (h1 : EF = 5) 
  (h2 : FH = 12) 
  (h3 : ∀ E F G H : ℝ, (EFGH is a rectangle ∧ H is the center of a circle ∧ F is on the circle)) 
  : 
    (169 / 2 * π - 60).toReal = 205.33 := 
begin 
  sorry 
end 

end shaded_area_of_sector_l679_679791


namespace seashells_second_day_l679_679104

theorem seashells_second_day (x : ℕ) (h1 : 5 + x + 2 * (5 + x) = 36) : x = 7 :=
by
  sorry

end seashells_second_day_l679_679104


namespace Y_complete_days_l679_679213

-- Define the problem
variables (X_done_in : ℕ) (X_days_worked : ℕ) (Y_days_remaining_work : ℕ) : ℕ
variable (total_days) : ℕ

-- Conditions from the problem
def X_complete_days : X_done_in = 40 := sorry
def X_partial_work_days : X_days_worked = 8 := sorry
def Y_finish_remaining_days : Y_days_remaining_work = 20 := sorry

-- Proof statement to determine how long Y will take to complete the work alone
theorem Y_complete_days (X_done_in : ℕ) (X_days_worked : ℕ) (Y_days_remaining_work : ℕ) :
  X_done_in = 40 → X_days_worked = 8 → Y_days_remaining_work = 20 → total_days = 25 := sorry

end Y_complete_days_l679_679213


namespace problem_1_and_2_l679_679723

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (x + a) * Real.exp (b * x)

theorem problem_1_and_2 :
  (∀ a b, ((∀ x, f x a b = (x + a) * Real.exp (b * x)) ∧
    (∀ (x : ℝ), f 1 a b = 0 ∧ (f.derivative 1) = Real.exp 1) →
    (a = -1 ∧ b = 1 ∧ ∃ m, m = -1))) ∧
  (∀ k x1 x2, k > 2 ∧ f x1 (-1) 1 = k * x1^2 - 2 ∧ f x2 (-1) 1 = k * x2^2 - 2 ∧ x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 →
    abs (x1 - x2) > Real.log (4 / Real.exp 1)) := sorry

end problem_1_and_2_l679_679723


namespace calculate_sum_l679_679642

namespace SequenceSum

def S (n : ℕ) : ℤ :=
  if even n then -↑n / 2
  else (↑n + 1) / 2

theorem calculate_sum :
  S 15 + S 20 + S 35 = 16 := by
-- we state the conditions derived from the given problem
-- specific proof steps are omitted
sorry

end SequenceSum

end calculate_sum_l679_679642


namespace repeating_decimal_sum_l679_679304

theorem repeating_decimal_sum : 
  (0.\overline{1} + 0.\overline{02} + 0.\overline{0002}) = (1315 / 9999) :=
by 
  have h1 : 0.\overline{1} = 1 / 9 := sorry,
  have h2 : 0.\overline{02} = 2 / 99 := sorry,
  have h3 : 0.\overline{0002} = 2 / 9999 := sorry,
  -- combining the converted fractions
  have h4 : (1 / 9) + (2 / 99) + (2 / 9999) = (1315 / 9999) := sorry,
  exact h4

end repeating_decimal_sum_l679_679304


namespace max_parts_divided_by_6_planes_l679_679987

theorem max_parts_divided_by_6_planes : 
  let n := 6 in (n^3 + 5 * n + 6) / 6 = 42 :=
by
  let n := 6
  have h := n^3 + 5 * n + 6
  have h1 : h = 252 := by
    sorry
  have h2 : h / 6 = 42 := by
    sorry
  exact h2

end max_parts_divided_by_6_planes_l679_679987


namespace speed_ratio_injury_initial_l679_679241

theorem speed_ratio_injury_initial (
  time_first_half : ℕ,
  v1 v2 : ℝ,
  h1 : 40 = 2 * 20,
  h2 : 4 + time_first_half = 8,
  h3 : 20 = v1 * ↑time_first_half,
  h4 : 20 = v2 * 8
) : v2 / v1 = 1 / 2 :=
by
  sorry

end speed_ratio_injury_initial_l679_679241


namespace decimal_to_binary_125_l679_679929

theorem decimal_to_binary_125 : ∃ (b : ℕ), 125 = b ∧ b = (1 * 2^6) + (1 * 2^5) + (1 * 2^4) + (1 * 2^3) + (1 * 2^2) + (0 * 2^1) + (1 * 2^0) :=
by {
  use 0b1111101,
  exact ⟨rfl, rfl⟩,
}

end decimal_to_binary_125_l679_679929


namespace minimum_black_edges_l679_679649

-- Define a cube type
structure Cube :=
(edges : Fin 12 → Bool)  -- Each edge can be red (false) or black (true)

-- Define a predicate that checks if each face has at least one black edge
def face_has_black_edge (c : Cube) (faces : List (Finset (Fin 12))) : Prop :=
  ∀ f in faces, ∃ e in f, c.edges e = true

-- The six faces of a cube represented as sets of edges
def cube_faces : List (Finset (Fin 12)) := [
  {0, 1, 2, 3},
  {4, 5, 6, 7},
  {8, 9, 10, 11},
  {0, 4, 8, 5},
  {1, 6, 9, 2},
  {3, 7, 11, 10}
]

-- The theorem that minimum number of black edges is 3
theorem minimum_black_edges (c : Cube)
  (h : face_has_black_edge c cube_faces) :
  ∃ (k : ℕ), k = 3 ∧ ∑ i, if c.edges i then 1 else 0 = k :=  
sorry

end minimum_black_edges_l679_679649


namespace polynomial_expression_value_l679_679360

/-
 We are given the polynomial f(x) = 2x^3 - 3x^2 + 5x - 7, and we need to calculate
 the value of the expression 16a - 9b + 3c - 2d given the coefficients a, b, c, and d.
-/
def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 5 * x - 7

theorem polynomial_expression_value : 
    let a := 2
        b := -3
        c := 5
        d := -7 in
    (16 * a - 9 * b + 3 * c - 2 * d) = 88 := 
by
    -- The below 'sorry' allows us to skip the proof.
    -- The statement is equivalent to the mathematical problem and can be built successfully.
    sorry

end polynomial_expression_value_l679_679360


namespace enclosed_area_l679_679130

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then x + 1
else if 0 ≤ x ∧ x ≤ (Real.pi / 2) then Real.cos x
else 0

theorem enclosed_area :
  (∫ x in -1..0, f x) + (∫ x in 0..(Real.pi / 2), f x) = 3 / 2 := 
by
  sorry

end enclosed_area_l679_679130


namespace total_days_proof_l679_679895

-- Conditions given in the problem
variables (total_first20_days total_remaining_days daily_earnings : ℕ)
variables (earnings_first20_days earnings_remaining_days : ℕ)

-- Define the values as given in the conditions
def total_first20_days := 120
def total_remaining_days := 66
def daily_earnings := 6

-- Prove the total combined days is 31 given the conditions
theorem total_days_proof : total_first20_days / daily_earnings + total_remaining_days / daily_earnings = 31 := by
  sorry

end total_days_proof_l679_679895


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679012

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679012


namespace squares_difference_l679_679378

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference_l679_679378


namespace angle_ACD_is_45_l679_679884

-- Define the basic setup of a parallelogram
variables {A B C D : Type} [Point A] [Point B] [Point C] [Point D]
variables [Segment AB] [Segment BC] [Segment CD] [Segment DA] [Segment BD]
variables [Angle BDA 45] [Angle BCD 45]
variables [Parallelogram ABCD]

-- Statement: Prove that angle ACD is 45 degrees given the conditions.
theorem angle_ACD_is_45 (h1 : ∡BDA = 45) (h2 : ∡BCD = 45) : ∡ACD = 45 :=
sorry

end angle_ACD_is_45_l679_679884


namespace arithmetic_sequence_number_of_terms_l679_679389

theorem arithmetic_sequence_number_of_terms 
  (a d : ℝ) (n : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 34) 
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146) 
  (h3 : (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 390) : 
  n = 13 :=
by 
  sorry

end arithmetic_sequence_number_of_terms_l679_679389


namespace zoey_finishes_20th_book_on_wednesday_l679_679568

theorem zoey_finishes_20th_book_on_wednesday :
  let days_spent := (20 * 21) / 2
  (days_spent % 7) = 0 → 
  (start_day : ℕ) → start_day = 3 → ((start_day + days_spent) % 7) = 3 :=
by
  sorry

end zoey_finishes_20th_book_on_wednesday_l679_679568


namespace Chloe_initial_picked_carrots_l679_679275

variable (x : ℕ)

theorem Chloe_initial_picked_carrots :
  (x - 45 + 42 = 45) → (x = 48) :=
by
  intro h
  sorry

end Chloe_initial_picked_carrots_l679_679275


namespace sum_of_x_coordinates_eq_l679_679848

noncomputable def intersection_points (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : set ℝ :=
{ x | ∃ (p q : fin 4), p ≠ q ∧ x = (b (p.1) - b (q.1)) / (a (q.1) - a (p.1)) }

theorem sum_of_x_coordinates_eq (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) (h : intersection_points a1 a2 a3 a4 b1 b2 b3 b4  = {x1, x2, x3, x4}) (x1 x2 x3 x4 : ℝ) :
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 →
  x1 + x4 = x2 + x3 :=
begin 
  sorry 
end

end sum_of_x_coordinates_eq_l679_679848


namespace log_equation_solution_l679_679569

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (log x 3 + log 3 x = log (sqrt x) 3 + log 3 (sqrt x) + 0.5) → x = 9 ∨ x = 1/3 :=
by
  sorry

end log_equation_solution_l679_679569


namespace jake_has_peaches_l679_679814

variable (Jake Steven Jill : Nat)

def given_conditions : Prop :=
  (Steven = 15) ∧ (Steven = Jill + 14) ∧ (Jake = Steven - 7)

theorem jake_has_peaches (h : given_conditions Jake Steven Jill) : Jake = 8 :=
by
  cases h with
  | intro hs1 hrest =>
      cases hrest with
      | intro hs2 hs3 =>
          sorry

end jake_has_peaches_l679_679814


namespace Joan_bought_72_eggs_l679_679423

def dozen := 12
def dozens_Joan_bought := 6
def eggs_Joan_bought := dozens_Joan_bought * dozen

theorem Joan_bought_72_eggs : eggs_Joan_bought = 72 := by
  sorry

end Joan_bought_72_eggs_l679_679423


namespace total_seashells_l679_679466

theorem total_seashells 
  (sally_seashells : ℕ)
  (tom_seashells : ℕ)
  (jessica_seashells : ℕ)
  (h1 : sally_seashells = 9)
  (h2 : tom_seashells = 7)
  (h3 : jessica_seashells = 5) : 
  sally_seashells + tom_seashells + jessica_seashells = 21 :=
by
  sorry

end total_seashells_l679_679466


namespace ratio_adidas_skechers_l679_679843

-- Conditions
def total_expenditure : ℤ := 8000
def expenditure_adidas : ℤ := 600
def expenditure_clothes : ℤ := 2600
def expenditure_nike := 3 * expenditure_adidas

-- Calculation for sneakers
def total_sneakers := total_expenditure - expenditure_clothes
def expenditure_nike_adidas := expenditure_nike + expenditure_adidas
def expenditure_skechers := total_sneakers - expenditure_nike_adidas

-- Prove the ratio
theorem ratio_adidas_skechers (H1 : total_expenditure = 8000)
                              (H2 : expenditure_adidas = 600)
                              (H3 : expenditure_nike = 3 * expenditure_adidas)
                              (H4 : expenditure_clothes = 2600) :
  expenditure_adidas / expenditure_skechers = 1 / 5 :=
by
  sorry

end ratio_adidas_skechers_l679_679843


namespace sum_squares_mod_13_is_zero_l679_679937

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l679_679937


namespace lake_depth_calculation_l679_679163

theorem lake_depth_calculation
    (water_surface_elevation : ℝ)
    (lake_floor_elevation : ℝ)
    (water_surface_elevation_value : water_surface_elevation = 180)
    (lake_floor_elevation_value : lake_floor_elevation = -220) :
    water_surface_elevation - lake_floor_elevation = 400 :=
by
  rw [water_surface_elevation_value, lake_floor_elevation_value]
  norm_num
  exact sorry

end lake_depth_calculation_l679_679163


namespace sum_of_integer_solutions_l679_679557

open Int

theorem sum_of_integer_solutions : 
  let S := {x : ℤ | 1 < (x - 3)^2 ∧ (x - 3)^2 < 36}
  ∑ x in S, x = 30 :=
by
  sorry

end sum_of_integer_solutions_l679_679557


namespace first_divisibility_second_divisibility_l679_679865

variable {n : ℕ}
variable (h : n > 0)

theorem first_divisibility :
  17 ∣ (5 * 3^(4*n+1) + 2^(6*n+1)) :=
sorry

theorem second_divisibility :
  32 ∣ (25 * 7^(2*n+1) + 3^(4*n)) :=
sorry

end first_divisibility_second_divisibility_l679_679865


namespace closest_to_17_l679_679981

noncomputable def closestWholeNumberAreaDifference
  (rect_length : ℝ) (rect_width : ℝ)
  (circle_diameter : ℝ) : ℕ :=
  let rect_area := rect_length * rect_width
  let circle_radius := circle_diameter / 2
  let circle_area := Real.pi * (circle_radius ^ 2)
  let shaded_area := rect_area - circle_area
  Int.toNat (Real.floor (shaded_area + 0.5))

theorem closest_to_17 : closestWholeNumberAreaDifference 4 5 2 = 17 :=
by
  sorry

end closest_to_17_l679_679981


namespace num_ordered_pairs_solves_eqn_l679_679655

theorem num_ordered_pairs_solves_eqn : 
  { pair : ℤ × ℤ // pair.1^4 + pair.2^2 = 2 * pair.2 }.to_finset.card = 4 := 
sorry

end num_ordered_pairs_solves_eqn_l679_679655


namespace jon_coffee_spending_in_april_l679_679817

def cost_per_coffee : ℕ := 2
def coffees_per_day : ℕ := 2
def days_in_april : ℕ := 30

theorem jon_coffee_spending_in_april :
  (coffees_per_day * cost_per_coffee) * days_in_april = 120 :=
by
  sorry

end jon_coffee_spending_in_april_l679_679817


namespace range_of_a_l679_679716

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^(x^2 + a * x) ≤ 1 / 2) → (a < -2 ∨ a > 2) :=
by {
  intro h,
  have : ∃ x : ℝ, 2^(x^2 + a * x) > 1 / 2,
  from sorry
}

end range_of_a_l679_679716


namespace cover_square_with_rect_l679_679962

theorem cover_square_with_rect (w h : ℝ) (split_point_left_x split_point_left_y split_point_bottom_x split_point_bottom_y : ℝ) :
  w = 9 ∧ h = 16 ∧ 
  split_point_left_x = 0 ∧ split_point_left_y = 3 ∧ 
  split_point_bottom_x = 4 ∧ split_point_bottom_y = 0 ∧
  (
    ∃ a b c d (x y : ℝ), 
      -- Coordinates of the cut points (a, b) and (c, d)
      a = split_point_left_x ∧ b = split_point_left_y ∧
      c = split_point_bottom_x ∧ d = split_point_bottom_y ∧
      x = 12 ∧ y = 12
  ) →
  ∃ (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ),
    -- Prove the existence of points after rearrangement that form a 12x12 square
    p1 = (split_point_left_x, split_point_left_y) ∧
    p2 = (split_point_bottom_x, split_point_bottom_y) ∧
    q1 = (0, y) ∧ q2 = (x, 0) ∧
    (
      (p1 = (0, 3) ∧ p2 = (4, 0)) ∧ 
      (q1 = (0, 12) ∧ q2 = (12, 0))
    ) →
    x = 12 ∧ y = 12 := sorry

end cover_square_with_rect_l679_679962


namespace num_cages_l679_679599

-- Define the conditions as given
def parrots_per_cage : ℕ := 8
def parakeets_per_cage : ℕ := 2
def total_birds_in_store : ℕ := 40

-- Prove that the number of bird cages is 4
theorem num_cages (x : ℕ) (h : 10 * x = total_birds_in_store) : x = 4 :=
sorry

end num_cages_l679_679599


namespace sum_of_squares_mod_13_l679_679941

theorem sum_of_squares_mod_13 : 
  (∑ i in Finset.range 13, i^2) % 13 = 0 :=
by
  sorry

end sum_of_squares_mod_13_l679_679941


namespace sum_of_digits_Y_squared_l679_679195

def sum_of_digits (n : ℕ) : ℕ :=
  nat.digits 10 n |>.sum

def Y : ℕ := 999999999

theorem sum_of_digits_Y_squared : sum_of_digits (Y * Y) = 81 := 
  sorry

end sum_of_digits_Y_squared_l679_679195


namespace max_circles_tangent_l679_679749

/-- Given a unit circle with radius 1, the maximum number of circles of radius 1
     that can be arranged tangentially around it, such that no two circles intersect
     each other and none of these circles contain the center of another circle inside 
     them, is 12. -/

theorem max_circles_tangent (R : ℝ) (n : ℕ) : 
  (∀ (S C : Type) [metric_space S] [metric_space C] (s : set S) (c : set C),
    ∃ r R > 0,
    metric.ball s.center R ∩ metric.ball c.center r = ∅ ∧
    ∀ o ∈ s.center, ∀ p ∈ c.center, 
    dist o p ≤ 2 * R) → 
  (n : ℕ) = 12 := 
sorry

end max_circles_tangent_l679_679749


namespace smaller_cuboid_length_l679_679750

theorem smaller_cuboid_length
  (L : ℝ)
  (h1 : 32 * (L * 4 * 3) = 16 * 10 * 12) :
  L = 5 :=
by
  sorry

end smaller_cuboid_length_l679_679750


namespace correct_product_of_a_b_l679_679100

theorem correct_product_of_a_b (a b : ℕ) (h1 : (a - (10 * (a / 10 % 10) + 1)) * b = 255)
                              (h2 : (a - (10 * (a / 100 % 10 * 10 + a % 10 - (a / 100 % 10 * 10 + 5 * 10)))) * b = 335) :
  a * b = 285 := sorry

end correct_product_of_a_b_l679_679100


namespace area_DFG_l679_679114

-- Let a, b >= 0 represent the sides AD and DC of the rectangle ABCD respectively.
variable {a b : ℝ}
-- Let HD = x, DK = y be the perpendicular segments dropped from G and F respectively.
variable {x y : ℝ}
-- The area of rectangle ABCD is given to be 20.
axiom ABCD_area : a * b = 20
-- The area of rectangle HEK (or HEKD) is given to be 8.
axiom HEK_area : x * y = 8

-- The theorem to prove that the area of triangle DFG is 6.
theorem area_DFG : Δ_DFG := 1 / 2 * (a * b - x * y) → area_DFG = 6 :=
by
  sorry

end area_DFG_l679_679114


namespace symmetry_of_transformed_graphs_l679_679684

theorem symmetry_of_transformed_graphs (f : ℝ → ℝ) : 
  symmetric_with_respect_to_line (f ∘ (λ x, x - 1)) (f ∘ (λ x, -x + 1)) 1 := 
sorry

end symmetry_of_transformed_graphs_l679_679684


namespace two_digit_numbers_of_form_3_pow_n_l679_679014

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l679_679014


namespace even_function_odd_function_f_on_interval_axis_of_symmetry_x1_equation_solutions_math_problem_equivalent_proof_l679_679714

noncomputable def f (x : ℝ) : ℝ := sorry

def domain_R (f : ℝ → ℝ) : Prop := ∀ x, x ∈ ℝ

-- Define the conditions
theorem even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x + 1) = f(-(x + 1))

theorem odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(x + 2) = -f(-(x + 2))

theorem f_on_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) = 2^x - 1

-- Define the statements to verify
theorem axis_of_symmetry_x1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(1 + x) = f(1 - x)

theorem equation_solutions (f : ℝ → ℝ) : Prop :=
  ∃ a b c, (f(a) - a = 0) ∧ (f(b) - b = 0) ∧ (f(c) - c = 0)

-- The proof goal that needs to be established
theorem math_problem_equivalent_proof (f : ℝ → ℝ)
  (h1 : domain_R f)
  (h2 : even_function f)
  (h3 : odd_function f)
  (h4 : f_on_interval f) :
  axis_of_symmetry_x1 f ∧ equation_solutions f :=
by
  sorry

end even_function_odd_function_f_on_interval_axis_of_symmetry_x1_equation_solutions_math_problem_equivalent_proof_l679_679714


namespace min_primes_to_guarantee_win_l679_679178

theorem min_primes_to_guarantee_win : 
  ∃ (p1 p2 p3 : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧
    p1 < 100 ∧ p2 < 100 ∧ p3 < 100 ∧
    (p1 % 10 = p2 / 10 % 10) ∧ 
    (p2 % 10 = p3 / 10 % 10) ∧ 
    (p3 % 10 = p1 / 10 % 10) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    p1 = 19 ∧ p2 = 97 ∧ p3 = 79 :=
  by sorry

end min_primes_to_guarantee_win_l679_679178


namespace coefficient_x3_l679_679806

theorem coefficient_x3 (a : ℝ) : 
  (let f := (ax + 1) * (x + (1/x)) ^ 6 in 
  term_coefficient(f, 3)) = 30 → a = 2 := 
begin
  sorry
end

end coefficient_x3_l679_679806


namespace f_log3_27_eq_neg1_l679_679777

-- Define the inverse function condition
def inv_f (y : ℝ) := y^2 + 2

-- Hypothesis: Inverse of f is given and considered for y < 0
axiom inv_f_condition : ∀ y, y < 0 → inv_f (f y) = y

-- Theorem to prove
theorem f_log3_27_eq_neg1 (f : ℝ → ℝ) (hf : ∀ y, y < 0 → inv_f (f y) = y) : f (log 3 27) = -1 :=
by {
    -- The proof goes here, we'll leave it as sorry since no proof is required
    sorry
}

end f_log3_27_eq_neg1_l679_679777


namespace tuples_and_triples_counts_are_equal_l679_679085

theorem tuples_and_triples_counts_are_equal (n : ℕ) (h : n > 0) :
  let countTuples := 8^n - 2 * 7^n + 6^n
  let countTriples := 8^n - 2 * 7^n + 6^n
  countTuples = countTriples :=
by
  sorry

end tuples_and_triples_counts_are_equal_l679_679085


namespace integral_evaluation_l679_679775

noncomputable def f (a : ℝ) : ℝ := ∫ x in 0..a, (2 + Real.sin x)

theorem integral_evaluation : f (Real.pi / 2) = Real.pi + 1 := by
  sorry

end integral_evaluation_l679_679775


namespace goods_train_speed_l679_679953

noncomputable def speed_of_goods_train (girl_train_speed_kmph : ℕ) (goods_train_length_m : ℕ) (pass_time_sec : ℕ) : ℚ :=
let 
  girl_train_speed_mps := (girl_train_speed_kmph : ℚ) * 1000 / 3600, -- converting km/h to m/s
  relative_speed_mps := goods_train_length_m / pass_time_sec,
  goods_train_speed_mps := relative_speed_mps - girl_train_speed_mps
in 
  goods_train_speed_mps * 3600 / 1000 -- converting m/s back to km/h

theorem goods_train_speed : 
  ∀ (girl_train_speed_kmph : ℕ) (goods_train_length_m : ℕ) (pass_time_sec : ℕ),
  girl_train_speed_kmph = 100 → 
  goods_train_length_m = 560 →
  pass_time_sec = 6 →
  speed_of_goods_train girl_train_speed_kmph goods_train_length_m pass_time_sec = 235.988 := 
by 
  intros girl_train_speed_kmph goods_train_length_m pass_time_sec 
         h_girl_train_speed h_goods_train_length h_pass_time
  unfold speed_of_goods_train
  simp [h_girl_train_speed, h_goods_train_length, h_pass_time]
  norm_num
  sorry -- skip the proof

end goods_train_speed_l679_679953


namespace percent_less_than_m_plus_d_l679_679978

variable (P : Type) [Probability P]
variables (m d : ℝ) (p : P)

-- Condition: the distribution is symmetric about mean m
def symmetric_about_mean (m : ℝ) (p : P) : Prop :=
  ∀ x : ℝ, P.measure {y | y = x + m} = P.measure {y | y = m - x}

-- Condition: 36 percent of the distribution within one standard deviation d of the mean
axiom within_one_std_dev (m d : ℝ) (p : P) : P.measure {x | abs (x - m) ≤ d} = 0.36

-- Question: What percent of the distribution is less than m + d
def percent_less_than (m d : ℝ) (p : P) : ℝ :=
  P.measure {x | x < m + d}

-- Proof: percent of the distribution that is less than m + d is 68 percent
theorem percent_less_than_m_plus_d (m d : ℝ) (p : P) 
  [symmetry : symmetric_about_mean m p] 
  [within_std_dev : within_one_std_dev m d p] :
  percent_less_than m d p = 0.68 := 
sorry

end percent_less_than_m_plus_d_l679_679978


namespace joe_max_money_after_bets_l679_679224

/-
  Given:
  - initial money = 100
  - number of bets = 5
  - max bet amount = 17
  - consolation rule: if lose 4 times in a row, win the fifth bet
  Prove: Joe can guarantee at least $98 after making exactly 5 bets.
-/
noncomputable def maxGuaranteed (initial_money : ℕ) (num_bets : ℕ) (max_bet : ℕ) : ℕ :=
  98

theorem joe_max_money_after_bets 
  (initial_money : ℕ)
  (num_bets : ℕ)
  (max_bet : ℕ)
  : initial_money = 100 ∧ num_bets = 5 ∧ max_bet <= 17
  → maxGuaranteed initial_money num_bets max_bet = 98 :=
by
  intros _ _ _ h
  sorry  -- proof steps will go here

end joe_max_money_after_bets_l679_679224


namespace log_lt_implies_lt_log_sufficient_not_necessary_l679_679072

-- Definitions from conditions
def log_strict_mono {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : (Real.log x < Real.log y) ↔ (x < y) :=
  Real.log_strict_mono hx hy

-- Theorem statement
theorem log_lt_implies_lt {x y : ℝ} (hx : 0 < x) (hy : 0 < y) : (Real.log x < Real.log y) → (x < y) :=
  by apply log_strict_mono hx hy

-- Proof that the condition is sufficient but not necessary
theorem log_sufficient_not_necessary {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
  (Real.log x < Real.log y) ↔ (x < y) ∧ (x < y) ↔ (Real.log x < Real.log y) → false :=
  by {
    -- log_strict_mono establishes sufficiency part
    have suff : (Real.log x < Real.log y) → (x < y),
    from log_lt_implies_lt hx hy,
    -- Necessary part fails because reverse does not always hold due to domain issues
    have not_nec : ¬((x < y) ↔ (Real.log x < Real.log y)),
    from λ h, by {
      have contra : x = y,
      from Iff.mp h (Real.log_strict_anti hx hy),
      exact Real.lt_irrefl x contra,
    },
    exact And.intro suff not_nec,
  }

end log_lt_implies_lt_log_sufficient_not_necessary_l679_679072


namespace sum_squares_mod_13_is_zero_l679_679938

def sum_squares_mod_13 : ℕ :=
  (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 + 10^2 + 11^2 + 12^2) % 13

theorem sum_squares_mod_13_is_zero : sum_squares_mod_13 = 0 := by
  sorry

end sum_squares_mod_13_is_zero_l679_679938


namespace sequence_s_100_l679_679698

-- Definitions for the sequence {a_n} with initial conditions and recurrence relation
def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ 
  (a 2 = 2) ∧ 
  (∀ n : ℕ, (a n) * (a (n + 1)) ≠ 1) ∧ 
  (∀ n : ℕ, (a n) * (a (n + 1)) * (a (n + 2)) = (a n) + (a (n + 1)) + (a (n + 2)))

-- Definition for the sum S_n
def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in range (n + 1), a i

-- The theorem we want to prove
theorem sequence_s_100 (a : ℕ → ℕ) (h : sequence a) : S a 100 = 199 := 
sorry

end sequence_s_100_l679_679698


namespace max_acute_angles_convex_octagon_l679_679933

/-- A convex octagon has at most 6 acute angles. -/
theorem max_acute_angles_convex_octagon : 
  ∀ (angles : (Fin 8) → ℝ), 
  (∀ i, angles i < 180 ∧ angles i > 0) → 
  (∑ i, angles i = 1080) → 
  (6 ≤ (Finset.filter (λ i, angles i < 90) Finset.univ).card) :=
by
  sorry

end max_acute_angles_convex_octagon_l679_679933


namespace find_x_l679_679162

theorem find_x (x y z : ℕ) (h1 : x = y / 2) (h2 : y = z / 3) (h3 : z = 90) : x = 15 :=
by
  sorry

end find_x_l679_679162


namespace sufficient_but_not_necessary_not_necessary_example_l679_679330

theorem sufficient_but_not_necessary (a b : ℝ) : (a > 2 ∧ b > 2) → a + b > 4 :=
by sorry

theorem not_necessary_example : ∃ (a b : ℝ), a + b > 4 ∧ ¬(a > 2 ∧ b > 2) :=
by
  use 3, 1
  have hab := 3 + 1
  have hgt4 : ¬(3 + 1 > 4) := by norm_num
  have ha2 : ¬(3 > 2 ∧ 1 > 2) := by
    norm_num,
  exact ⟨hab, hgt4, ha2⟩

end sufficient_but_not_necessary_not_necessary_example_l679_679330


namespace right_triangle_area_l679_679123

open Real

theorem right_triangle_area
  (a : ℝ) 
  (h₁ : a > 0) 
  (h₂ : a < 24)
  (h₃ : 24^2 + a^2 = (48 - a)^2) : 
  1/2 * 24 * a = 216 :=
by
  -- This is just a statement, the proof is omitted
  sorry

end right_triangle_area_l679_679123


namespace triangle_is_equilateral_l679_679153

noncomputable definition triangle_equilateral (a b c : ℝ) (h_a h_b h_c : ℝ) (r : ℝ) =
  let s := (a + b + c) / 2 in
  let area := s in
  r = 1 ∧ h_a ≥ 3 ∧ h_b ≥ 3 ∧ h_c ≥ 3 ∧
  a * h_a / 2 = area ∧ b * h_b / 2 = area ∧ c * h_c / 2 = area

theorem triangle_is_equilateral {a b c h_a h_b h_c : ℝ} (h_src : r = 1) 
                                  (h_altitudes : h_a ≥ 3 ∧ h_b ≥ 3 ∧ h_c ≥ 3)
                                  (h_area_a : a * h_a / 2 = (a + b + c) / 2)
                                  (h_area_b : b * h_b / 2 = (a + b + c) / 2)
                                  (h_area_c : c * h_c / 2 = (a + b + c) / 2) :
  a = b ∧ b = c :=
by {
  sorry
}

end triangle_is_equilateral_l679_679153


namespace all_sets_B_l679_679896

open Set

theorem all_sets_B (B : Set ℕ) :
  { B | {1, 2} ∪ B = {1, 2, 3} } =
  ({ {3}, {1, 3}, {2, 3}, {1, 2, 3} } : Set (Set ℕ)) :=
sorry

end all_sets_B_l679_679896


namespace min_tangent_length_l679_679615

noncomputable def center := (3 : ℝ, -1 : ℝ)
def radius := real.sqrt 2
def line (x : ℝ) : ℝ := x + 2
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 2

theorem min_tangent_length :
  ∃ P : ℝ × ℝ, line P.1 = P.2 ∧
  ∃ M : ℝ × ℝ, circle_eq M.1 M.2 ∧
  let PM_len := real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) in
  PM_len = 4 :=
sorry

end min_tangent_length_l679_679615


namespace ratio_sheep_horses_eq_six_seven_l679_679636

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l679_679636


namespace cos_alpha_sqrt_l679_679038

theorem cos_alpha_sqrt {α : ℝ} (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  Real.cos α = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end cos_alpha_sqrt_l679_679038


namespace total_handshakes_l679_679269

-- There are 5 members on each of the two basketball teams.
def teamMembers : Nat := 5

-- There are 2 referees.
def referees : Nat := 2

-- Each player from one team shakes hands with each player from the other team.
def handshakesBetweenTeams : Nat := teamMembers * teamMembers

-- Each player shakes hands with each referee.
def totalPlayers : Nat := 2 * teamMembers
def handshakesWithReferees : Nat := totalPlayers * referees

-- Prove that the total number of handshakes is 45.
theorem total_handshakes : handshakesBetweenTeams + handshakesWithReferees = 45 := by
  -- Total handshakes is the sum of handshakes between teams and handshakes with referees.
  sorry

end total_handshakes_l679_679269


namespace vector_dot_product_l679_679392

noncomputable theory

open_locale classical

variables {V : Type*} [inner_product_space ℝ V]

variables (A B C P : V)
variables (a b c : ℝ)
variables h1 : ∥B - A∥ = 5
variables h2 : 20 * a • (C - B) + 15 * b • (A - C) + 12 * c • (B - A) = 0
variables h3 : P - B = ⅔ • (A - P)

theorem vector_dot_product :
  (P - C) ⬝ (B - A) = -23/3 :=
sorry

end vector_dot_product_l679_679392


namespace ratio_sheep_horses_eq_six_seven_l679_679635

noncomputable def total_food_per_day : ℕ := 12880
noncomputable def food_per_horse_per_day : ℕ := 230
noncomputable def num_sheep : ℕ := 48
noncomputable def num_horses : ℕ := total_food_per_day / food_per_horse_per_day
noncomputable def ratio_sheep_to_horses := num_sheep / num_horses

theorem ratio_sheep_horses_eq_six_seven :
  ratio_sheep_to_horses = 6 / 7 :=
by
  sorry

end ratio_sheep_horses_eq_six_seven_l679_679635


namespace powers_of_three_two_digit_count_l679_679025

theorem powers_of_three_two_digit_count : 
  (finset.filter (λ n, 10 ≤ 3^n ∧ 3^n ≤ 99) (finset.range 10)).card = 2 := by
sorry

end powers_of_three_two_digit_count_l679_679025


namespace minimize_expression_l679_679829

noncomputable def minimum_value (a b : ℝ) : ℝ :=
  a^2 + b^2 + (1 / a^2) + (1 / b^2) + (b / a)

theorem minimize_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (x : ℝ), minimum_value a b = √15 :=
sorry

end minimize_expression_l679_679829


namespace combined_weight_cats_l679_679252

-- Define the weights of the cats
def weight_cat1 := 2
def weight_cat2 := 7
def weight_cat3 := 4

-- Prove the combined weight of the three cats is 13 pounds
theorem combined_weight_cats :
  weight_cat1 + weight_cat2 + weight_cat3 = 13 := by
  sorry

end combined_weight_cats_l679_679252


namespace minimum_distance_tangent_to_circle_l679_679900

theorem minimum_distance_tangent_to_circle :
  let l := (x + y - 2 = 0)
  let C := ((x + 2)^2 + y^2 = 1)
  let center := (-2, 0)
  (∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ C ∧ (dist p q = 2 * real.sqrt 2 - 1)) :=
begin
  let l : ℝ → ℝ := λ x, -x + 2,
  let C : set (ℝ × ℝ) := {p | (p.1 + 2)^2 + p.2^2 = 1},
  let center := (-2, 0),
  existsi l,
  existsi C,
  sorry
end

end minimum_distance_tangent_to_circle_l679_679900


namespace even_grid_possible_l679_679682

-- Define the grid and operations
def grid (n : ℕ) := list (list ℤ)

-- Define the condition for the initial state of the grid
def init_grid (n : ℕ) : grid n := list.repeat (list.repeat 1 n) n

-- Define the function to perform the operation on the grid
noncomputable def flip_adjacent (g : grid n) (i j : ℕ) : grid n := sorry

-- Define the primary statement to be proved
theorem even_grid_possible (n : ℕ) (h : 2 ≤ n) (even_n : n % 2 = 0) :
  ∃ fin_ops : list (ℕ × ℕ), 
    let final_grid := list.foldl (λ g op, flip_adjacent g op.fst op.snd) (init_grid n) fin_ops 
    in final_grid = list.repeat (list.repeat (-1) n) n := sorry

end even_grid_possible_l679_679682


namespace base_conversion_is_248_l679_679134

theorem base_conversion_is_248 (a b c n : ℕ) 
  (h1 : n = 49 * a + 7 * b + c) 
  (h2 : n = 81 * c + 9 * b + a) 
  (h3 : 0 ≤ a ∧ a ≤ 6) 
  (h4 : 0 ≤ b ∧ b ≤ 6) 
  (h5 : 0 ≤ c ∧ c ≤ 6)
  (h6 : 0 ≤ a ∧ a ≤ 8) 
  (h7 : 0 ≤ b ∧ b ≤ 8) 
  (h8 : 0 ≤ c ∧ c ≤ 8) 
  : n = 248 :=
by 
  sorry

end base_conversion_is_248_l679_679134


namespace find_abc_l679_679240

theorem find_abc (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h_eq : 10 * a + 11 * b + c = 25) : a = 0 ∧ b = 2 ∧ c = 3 := 
sorry

end find_abc_l679_679240


namespace cricket_target_run_l679_679409

theorem cricket_target_run (run_rate1 run_rate2 : ℝ) (overs1 overs2 : ℕ) (T : ℝ) 
  (h1 : run_rate1 = 3.2) (h2 : overs1 = 10) (h3 : run_rate2 = 25) (h4 : overs2 = 10) :
  T = (run_rate1 * overs1) + (run_rate2 * overs2) → T = 282 :=
by
  sorry

end cricket_target_run_l679_679409


namespace triangle_acute_angle_condition_l679_679626

theorem triangle_acute_angle_condition 
  (A B C X : out_param (ℝ × ℝ))
  (h_triangle : (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C))
  (h_obtuse : ∃ (X : ℝ × ℝ), (dist X A) > (dist X B) ∧ (dist X A) > (dist X C))
  (h_XA_longest : (dist X A) ≥ max (dist X B) (dist X C)) :
  angle A B C < π / 2 := 
begin
  sorry
end

end triangle_acute_angle_condition_l679_679626


namespace abs_inequality_solution_l679_679505

theorem abs_inequality_solution (x : ℝ) : (|2 * x - 1| - |x - 2| < 0) ↔ (-1 < x ∧ x < 1) := 
sorry

end abs_inequality_solution_l679_679505


namespace sin_10pi_over_3_l679_679661

theorem sin_10pi_over_3 : Real.sin (10 * Real.pi / 3) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_10pi_over_3_l679_679661


namespace chord_and_radius_length_l679_679980

open EuclideanGeometry

variables (A B C D E : Point)
variables (h_circle : CircleThrough AB)
variables (h_intersectBC : IntersectsSides h_circle BC D)
variables (h_intersectAC : IntersectsSides h_circle AC E)
variables (h_area : Area (Triangle C D E) /= 7 * Area (Quadrilateral A B D E))
variables (h_AB : dist A B = 4)
variables (h_angle : ∠ A C B = 45)

-- Proof obligations
theorem chord_and_radius_length
  (h_conditions : 
    (exists h_circle : Circle Through A B, 
        IntersectsSides h_circle BC D 
    /\ IntersectsSides h_circle AC E))
  :
  let DE := dist D E
  let R := radius h_circle
  in DE = sqrt 2 
    && R = sqrt 5
:= sorry

end chord_and_radius_length_l679_679980


namespace two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679010

theorem two_digit_numbers_of_3_pow {n : ℤ} : 
  (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99) → n ∈ {3, 4} :=
by {
  sorry
}

theorem number_of_two_digit_numbers_of_3_pow : 
  ∃ (s : Finset ℤ), (∀ n, n ∈ s ↔ (10 ≤ 3 ^ n ∧ 3 ^ n ≤ 99)) ∧ s.card = 2 :=
by {
  use {3, 4},
  split,
  { intro n,
    split,
    { intro h,
      rw Finset.mem_insert,
      rw Finset.mem_singleton,
      rw ← two_digit_numbers_of_3_pow h,
      tauto,
    },
    { intro h,
      cases h,
      { simp only [h, pow_succ, pow_one, mul_three] },
      { simp only [h, pow_succ, pow_one, mul_three] }
    }
  },
  refl
}

end two_digit_numbers_of_3_pow_number_of_two_digit_numbers_of_3_pow_l679_679010


namespace proof_problem_l679_679059

-- Given conditions
def is_acute_triangle (A B C : ℝ) : Prop := A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2

variables (a b c A B C : ℝ)

-- Condition 1: Triangle is acute
def condition1 : Prop := is_acute_triangle A B C

-- Condition 2: Sides opposite to angles A, B, and C are a, b, and c respectively
def condition2 : Prop := True  -- This is implicit in the problem statement

-- Condition 3: 2a sin B = sqrt(3) b
def condition3 : Prop := 2 * a * real.sin B = real.sqrt 3 * b

-- Given this problem setup, we need to prove:
-- 1. Angle A = 60 degrees (π / 3 radians)
noncomputable def question1 := A = π / 3

-- 2. Given b = 1, a = sqrt(3), find the area of the triangle
def area (a b C : ℝ) := 0.5 * a * b * real.sin C
def given_conditions : Prop := b = 1 ∧ a = real.sqrt 3
noncomputable def question2 : Prop := given_conditions → area a b C = real.sqrt 3 / 2

-- Combining all in one proof statement
theorem proof_problem :
  (condition1 ∧ condition2 ∧ condition3) → (question1 ∧ question2) :=
by
  sorry

end proof_problem_l679_679059


namespace quadrilateral_concurrent_lines_l679_679601

variables {A B C D K L M N K' L' M' N' : Type*}
variables [plane_geom A B C D K L M N] -- Assuming an abstract plane geometry
variables (cyclic : cyclic A B C D)
variables (circumscribed : circumscribed A B C D K L M N)
variables (ext_bisectors : ext_bisectors_intersect A B C D K' L' M' N')

theorem quadrilateral_concurrent_lines 
  (H_cyclic : cyclic)
  (H_circumscribed : circumscribed)
  (H_ext_bisectors : ext_bisectors_intersect) : 
  concurrent K K' L L' M M' N N' :=
sorry

end quadrilateral_concurrent_lines_l679_679601


namespace rulers_added_l679_679902

theorem rulers_added (original_rulers new_rulers added_rulers : ℕ) : 
  original_rulers = 11 →
  new_rulers = 25 →
  added_rulers = new_rulers - original_rulers →
  added_rulers = 14 :=
begin
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  exact h3,
end

end rulers_added_l679_679902


namespace find_b_l679_679638

-- Define the constants and assumptions
variables {a b c d : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d)

-- The function completes 5 periods between 0 and 2π
def completes_5_periods (b : ℝ) : Prop :=
  (2 * Real.pi) / b = (2 * Real.pi) / 5

theorem find_b (h : completes_5_periods b) : b = 5 :=
sorry

end find_b_l679_679638


namespace distance_between_Sasha_and_Kolya_l679_679470

theorem distance_between_Sasha_and_Kolya :
  ∀ (vS vL vK : ℝ) (tS tL : ℝ),
  (vK = 0.9 * vL) →
  (tS = 100 / vS) →
  (vL * tS = 90) →
  (vL = 0.9 * vS) →
  (vK * tS = 81) →
  (100 - vK * tS = 19) :=
begin
  intros,
  sorry
  end

end distance_between_Sasha_and_Kolya_l679_679470


namespace max_marks_l679_679614

theorem max_marks (M p : ℝ) (h1 : p = 0.60 * M) (h2 : p = 160 + 20) : M = 300 := by
  sorry

end max_marks_l679_679614
