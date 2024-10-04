import Integration
import Mathlib
import Mathlib.Algebra.ConicSection
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Order.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.FiniteDimensional
import Mathlib.NumberTheory.GCD
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityTheory
import Mathlib.ProbabilityTheory
import Mathlib.Topology.Instances.Real
import Real

namespace smallest_repeating_block_digits_l821_821895

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end smallest_repeating_block_digits_l821_821895


namespace prove_a2_l821_821022

def arithmetic_seq (a d : ℕ → ℝ) : Prop :=
  ∀ n m, a n + d (n - m) = a m

theorem prove_a2 (a : ℕ → ℝ) (d : ℕ → ℝ) :
  (∀ n, a n = a 0 + (n - 1) * 2) → 
  (a 1 + 4) / a 1 = (a 1 + 6) / (a 1 + 4) →
  (d 1 = 2) →
  a 2 = -6 :=
by
  intros h_seq h_geo h_common_diff
  sorry

end prove_a2_l821_821022


namespace correct_propositions_l821_821043

structure Conditions :=
  (prop1 : ∀ x, x ∈ Ioo 2 ⊤ → monotone_decr_on (λ (x : ℝ), log (1/2) (x ^ 2 - 4)) (Ioo 2 ⊤))
  (prop2 : ∀ f, (∀ x, f x = f (2 - x)) → (symmetric_about_line f 1))
  (prop3 : even_function (λ (x : ℝ), log (x + 1) + log (x - 1)))
  (prop4 : ∀ (f : ℝ → ℝ) (x0 : ℝ), (deriv f x0 = 0) → (is_extreme_point f x0))

theorem correct_propositions (c : Conditions) : 
  ({1, 2} : set ℕ) := 
by sorry

end correct_propositions_l821_821043


namespace seating_arrangements_l821_821957

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821957


namespace symmetric_polar_curve_l821_821110

/-- Symmetry of polar curve about specific line -/
theorem symmetric_polar_curve (theta rho : ℝ) 
  (h1 : rho = cos theta + 1)
  (h2 : theta = π / 6) :
  rho = sin (theta + π / 6) + 1 := 
by 
  sorry

end symmetric_polar_curve_l821_821110


namespace number_of_possible_M_values_l821_821061

-- Definition of the problem setup
def steel_bars : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def three_bags_with_equal_mass : List ℕ → Prop
| [a, b, c, d, e, f] := 
  let bags := [(a + b), (c + d), (e + f)] in 
  bags.all (fun m => m = bags.head!)

-- Definition to enumerate possible values of M (mass in each bag)
def possible_values_of_M (steel_bars : List ℕ) : List ℕ :=
  let combinations := steel_bars.combinations 6 in
  let valid_combinations := combinations.filter (λ l, three_bags_with_equal_mass l) in
  valid_combinations.map (λ l, l.sum / 3)

-- The main theorem to prove
theorem number_of_possible_M_values (steel_bars : List ℕ) :
  steel_bars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] →
  (possible_values_of_M steel_bars).to_finset.card = 19 :=
by
  intros h,
  sorry

end number_of_possible_M_values_l821_821061


namespace arrangement_count_l821_821932

/-- Number of ways to arrange Xiao Kai, 2 elderly people, and 3 volunteers in a line,
    satisfying the conditions that Xiao Kai must be adjacent to both elderly people
    and the elderly people must not be at either end of the line. -/
theorem arrangement_count : 
  let people := ["Xiao Kai", "elderly1", "elderly2", "volunteer1", "volunteer2", "volunteer3"] in
  ∃ (arrangements : Finset (List String)),
    (∀ (arrangement : List String) (h : arrangement ∈ arrangements), 
      "Xiao Kai" :: "elderly1" :: "elderly2" :: [] ∈ arrangements ∨
      "Xiao Kai" :: "elderly2" :: "elderly1" :: [] ∈ arrangements ∨
      "elderly1" :: "Xiao Kai" :: "elderly2" :: [] ∈ arrangements ∨
      "elderly2" :: "Xiao Kai" :: "elderly1" :: [] ∈ arrangements) ∧
    (∀ arrangement ∈ arrangements, 
      arrangement.head ≠ "elderly1" ∧ 
      arrangement.head ≠ "elderly2" ∧ 
      arrangement.last ≠ "elderly1" ∧ 
      arrangement.last ≠ "elderly2") ∧
    arrangements.card = 48 :=
by sorry

end arrangement_count_l821_821932


namespace digit_to_make_52B6_divisible_by_3_l821_821670

theorem digit_to_make_52B6_divisible_by_3 (B : ℕ) (hB : 0 ≤ B ∧ B ≤ 9) : 
  (5 + 2 + B + 6) % 3 = 0 ↔ (B = 2 ∨ B = 5 ∨ B = 8) := 
by
  sorry

end digit_to_make_52B6_divisible_by_3_l821_821670


namespace shaded_area_converges_l821_821173

theorem shaded_area_converges (AC CG : ℝ) (hAC : AC = 6) (hCG : CG = 6) : 
  ∑' n : ℕ, (4.5 * (1/4)^n) = 6 :=
by
  sorry

end shaded_area_converges_l821_821173


namespace sqrt_2021_construction_l821_821161

noncomputable def construct_sqrt_2021 (P Q : ℝ × ℝ) (hPQ : dist P Q = 1) : Prop :=
∃ (L : list (ℝ × ℝ → ℝ) ∪ list (ℝ × ℝ × ℝ)), 
  length L ≤ 10 ∧ 
  ∃ A B : ℝ × ℝ, 
    A ≠ B ∧ (A, B) ∈ L ∧ dist A B = Real.sqrt 2021

theorem sqrt_2021_construction :
  ∀ (P Q : ℝ × ℝ), dist P Q = 1 →
  construct_sqrt_2021 P Q :=
begin
  sorry
end

end sqrt_2021_construction_l821_821161


namespace solution_set_for_inequality_l821_821815

theorem solution_set_for_inequality : {x : ℝ | x ≠ 0 ∧ (x-1)/x ≤ 0} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end solution_set_for_inequality_l821_821815


namespace jury_stabilize_after_15_sessions_l821_821274

theorem jury_stabilize_after_15_sessions (J : Type) [DecidableEq J] (M : J → J → Prop) 
  (hj_init : Fintype.card J = 30) 
  (M_irreflexive : ∀ j, ¬ M j j) 
  (exclusion_rule : ∀ (S : Finset J), ∀ j ∈ S, 
    (2 * (S.filter (λ m, M m j)).card < S.card → j ∉ S)) : 
  ∃ N ≤ 15, ∀ (S : Finset J) (hn : Fintype.card S = N), ∀ j ∈ S, ¬ (2 * (S.filter (λ m, M m j)).card < S.card) := sorry

end jury_stabilize_after_15_sessions_l821_821274


namespace acceptable_arrangements_correct_l821_821943

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821943


namespace ylona_initial_bands_l821_821970

variable (B J Y : ℕ)  -- Represents the initial number of rubber bands for Bailey, Justine, and Ylona respectively

-- Define the conditions
axiom h1 : J = B + 10
axiom h2 : J = Y - 2
axiom h3 : B - 4 = 8

-- Formulate the statement
theorem ylona_initial_bands : Y = 24 :=
by
  sorry

end ylona_initial_bands_l821_821970


namespace intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l821_821046

noncomputable def f (x : ℝ) := x * Real.log (-x)
noncomputable def g (x a : ℝ) := x * f (a * x) - Real.exp (x - 2)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < -1 / Real.exp 1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 / Real.exp 1 < x ∧ x < 0 → deriv f x < 0) ∧
  f (-1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

theorem number_of_zeros_of_g (a : ℝ) :
  (a > 0 ∨ a = -1 / Real.exp 1 → ∃! x : ℝ, g x a = 0) ∧
  (a < 0 ∧ a ≠ -1 / Real.exp 1 → ∀ x : ℝ, g x a ≠ 0) :=
sorry

end intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l821_821046


namespace area_of_VXYZ_is_100_l821_821098

noncomputable def calculate_area_of_VXYZ
  (base_WZ : ℝ) (height_Y_to_WZ : ℤ)
  (WV_plus_VZ : ℝ) (VZ : ℝ)
  (area_VXYZ : ℝ) : Prop :=
base_WZ = 12 ∧ height_Y_to_WZ = 10 ∧ WV_plus_VZ = 12 ∧ VZ = 8 ∧ area_VXYZ = 100

theorem area_of_VXYZ_is_100 :
  calculate_area_of_VXYZ 12 10 12 8 100 :=
by
  simp [calculate_area_of_VXYZ]
  sorry

end area_of_VXYZ_is_100_l821_821098


namespace train_length_proof_l821_821312

-- Define the conditions as parameters
variables (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ)

-- Assume specific values for the given problem conditions
def speed_value : ℕ := 72
def time_value : ℕ := 25
def bridge_value : ℕ := 140

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_kmh * 1000 / 3600

-- Calculate the total distance covered while passing the bridge
def total_distance : ℕ := speed_m_per_s * time_s

-- Prove the length of the train
theorem train_length_proof : 
  speed_m_per_s = 20 →  -- Proof step: Convert 72 km/h = 20 m/s
  total_distance = 500 →  -- Proof step: 20 m/s * 25 s = 500 m
  total_distance - bridge_length_m = 360 := 
by 
  intros h₁ h₂
  rw [h₁, h₂]
  -- Calculation for the length of the train
  sorry

end train_length_proof_l821_821312


namespace first_player_wins_l821_821160

theorem first_player_wins :
  ∀ (grid : ℕ × ℕ) (points : ℕ)
    (total_pairs : ℕ) 
    (pairs_strategy : ℕ) 
    (v : list (ℕ × ℕ)),
    grid = (49, 69) →
    points = 50 * 70 →
    total_pairs = 840 + 910 →
    pairs_strategy ≥ 200 →
    (∀ i, v[i] = horizontal ∨ v[i] = vertical) →
    (∃ (v_sum : ℕ × ℕ), v_sum = (0, 0)) → 
  True :=
begin
  intros,
  sorry
end

end first_player_wins_l821_821160


namespace evaluate_9_x_minus_1_l821_821071

theorem evaluate_9_x_minus_1 (x : ℝ) (h : (3 : ℝ)^(2 * x) = 16) : (9 : ℝ)^(x - 1) = 16 / 9 := by
  sorry

end evaluate_9_x_minus_1_l821_821071


namespace divisors_units_digit_three_l821_821004

def units_digit (n : ℕ) : ℕ := n % 10

theorem divisors_units_digit_three (n : ℕ) (h_pos : 0 < n) : 
  let S_n : Finset ℕ := n.divisors in
  (Finset.filter (λ d => units_digit d = 3) S_n).card ≤ S_n.card / 2 :=
by sorry

end divisors_units_digit_three_l821_821004


namespace constant_function_value_l821_821506

theorem constant_function_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x) = 5) (x : ℝ) : g(3 * x - 7) = 5 :=
sorry

end constant_function_value_l821_821506


namespace lambda_parallel_l821_821456

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821456


namespace decreasing_linear_function_iff_l821_821637

theorem decreasing_linear_function_iff {m : ℝ} : 
  (∀ x1 x2 : ℝ, x1 < x2 → (m-2) * x1 + 1 > (m-2) * x2 + 1) ↔ m < 2 :=
by
  split
  sorry

end decreasing_linear_function_iff_l821_821637


namespace third_trial_point_l821_821667

theorem third_trial_point (a b : ℝ) (ϕ : ℝ := 0.618) (x1 : ℝ := a + ϕ * (b - a)) (x2 : ℝ := a + (b - x1)) :
  a = 2 → b = 4 → x1 > x2 → (4 - ϕ * (4 - x1) = 3.528) :=
by
  intros ha hb hgt
  rw [ha, hb, x1, x2]
  -- Further steps would follow with simplifications and calculations (omitted by the "sorry")
  sorry

end third_trial_point_l821_821667


namespace seating_arrangements_l821_821961

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821961


namespace find_prime_p_l821_821352

open Int

theorem find_prime_p (p k m n : ℕ) (hp : Nat.Prime p) 
  (hk : 0 < k) (hm : 0 < m)
  (h_eq : (mk^2 + 2 : ℤ) * p - (m^2 + 2 * k^2 : ℤ) = n^2 * (mp + 2 : ℤ)) :
  p = 3 ∨ p = 1 := sorry

end find_prime_p_l821_821352


namespace rotate_point_6_4_pi_over_3_l821_821417

def rotate_point (p: ℂ) (θ: ℝ) : ℂ :=
  p * complex.exp(θ * complex.I)

theorem rotate_point_6_4_pi_over_3 :
  rotate_point (6 + 4 * complex.I) (real.pi / 3) = (3 - 2 * real.sqrt 3) + (2 + 3 * real.sqrt 3) * complex.I :=
by
  sorry

end rotate_point_6_4_pi_over_3_l821_821417


namespace number_of_integers_n_l821_821358

theorem number_of_integers_n (n : ℤ) : n ∈ {n : ℤ | n ≥ 1 ∧ n ≤ 2000 ∧ ∃ a b : ℤ, n = a * (a + 1) ∧ a + b = -1 ∧ a ≠ b ∧ a ∈ Set.Icc 1 2000 ∧ b ∈ Set.Icc 1 2000} ∧ is_square n :=
  sorry

end number_of_integers_n_l821_821358


namespace isosceles_triangle_length_l821_821662

theorem isosceles_triangle_length (a : ℝ) (h : 3 * (1 / 2 * (1 / 3) * h) = (sqrt 3 / 4)) : a = sqrt 3 / 3 :=
sorry

end isosceles_triangle_length_l821_821662


namespace lambda_parallel_l821_821457

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821457


namespace brendan_grass_cutting_l821_821759

theorem brendan_grass_cutting (daily_yards : ℕ) (percentage_increase : ℕ) (original_days_per_week : ℕ) (expected_result : ℕ) :
  daily_yards = 8 →
  percentage_increase = 50 →
  let additional_yards_per_day := (percentage_increase * daily_yards) / 100 in
  let total_yards_per_day := daily_yards + additional_yards_per_day in
  let total_yards_per_week := total_yards_per_day * original_days_per_week in
  original_days_per_week = 7 →
  expected_result = 84 →
  total_yards_per_week = expected_result :=
by
  intros h_daily_yards h_percentage_increase additional_yards_per_day total_yards_per_day total_yards_per_week h_days_per_week h_expected_result
  rw [h_daily_yards, h_percentage_increase, h_days_per_week, h_expected_result]
  simp
  sorry

end brendan_grass_cutting_l821_821759


namespace slope_of_line_angle_l821_821228

theorem slope_of_line_angle :
  let x := λ y: ℝ, (sqrt 3) * y - 1 in
  ∀ θ : ℝ, (tan θ = (1 / (sqrt 3))) → θ = 30 := sorry

end slope_of_line_angle_l821_821228


namespace acceptable_arrangements_correct_l821_821940

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821940


namespace decreasing_interval_of_even_function_l821_821914

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k - 1)*x + 2

theorem decreasing_interval_of_even_function (k : ℝ) (h : ∀ x : ℝ, f k x = f k (-x)) :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x < 0 → f k x > f k (-x)) := 
sorry

end decreasing_interval_of_even_function_l821_821914


namespace triangle_obtuse_at_most_one_l821_821258

open Real -- Work within the Real number system

-- Definitions and main proposition
def is_obtuse (angle : ℝ) : Prop := angle > 90

def triangle (a b c : ℝ) : Prop := a + b + c = 180

theorem triangle_obtuse_at_most_one (a b c : ℝ) (h : triangle a b c) :
  is_obtuse a ∧ is_obtuse b → false :=
by
  sorry

end triangle_obtuse_at_most_one_l821_821258


namespace area_sum_l821_821516

-- Definitions of given conditions
def midpoint (A B M : Point) := dist A M = dist M B

def is_right_angle (A B C : Point) := angle A B C = 90

def isosceles_right_triangle (A B C : Point) :=
  is_right_angle A B C ∧ angle B A C = 45 ∧ angle B C A = 45

def triangle_area (A B C : Point) :=
  (dist A B * dist B C * sin (angle A B C)) / 2

-- Specific conditions
variables {P Q R S T : Point}
variables (h_midpoint : midpoint Q R S)
variables (h_t_on_pr : on_segment P R T)
variables (h_pr_length : dist P R = 12)
variables (h_angles : ∠QPR = 45 ∧ ∠PQR = 90 ∧ ∠PRQ = 45)
variables (h_angle_rts : ∠RTS = 45)

theorem area_sum :
  let ΔPQR := triangle_area P Q R in
  let ΔRST := triangle_area R S T in
  ΔPQR + 2 * ΔRST = 54 :=
begin
  sorry
end

end area_sum_l821_821516


namespace loss_calculation_l821_821539

def principal : ℝ := 6500
def annual_rate : ℝ := 0.04
def time_years : ℝ := 2
def compounding_frequency : ℝ := 1

def compound_amount : ℝ :=
  principal * (1 + annual_rate / compounding_frequency) ^ (compounding_frequency * time_years)

def compound_interest : ℝ := compound_amount - principal

def simple_interest : ℝ := principal * annual_rate * time_years

def loss : ℝ := compound_interest - simple_interest

theorem loss_calculation :
  loss = 9.40 := by
  sorry

end loss_calculation_l821_821539


namespace solution_set_f_ge_0_l821_821865

variables {f : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, f (-x) = -f x  -- f is odd function
axiom h2 : ∀ x y : ℝ, 0 < x → x < y → f x < f y  -- f is monotonically increasing on (0, +∞)
axiom h3 : f 3 = 0  -- f(3) = 0

theorem solution_set_f_ge_0 : { x : ℝ | f x ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } ∪ { x : ℝ | 3 ≤ x } :=
by
  sorry

end solution_set_f_ge_0_l821_821865


namespace distinct_roots_difference_l821_821138

theorem distinct_roots_difference
  (h : ∀ x, (x - 5) * (x + 5) = 17 * x - 85) :
  ∃ (p q : ℝ), p ≠ q ∧ p > q ∧ (p > 5 ∧ q = 5 ∨ p = 5 ∧ q < 5) ∧ p - q = 7 :=
by {
  let f := λ x, (x - 5) * (x + 5),
  let g := λ x, 17 * x - 85,
  have f_eq_g : ∀ x, f x = g x := h,
  -- Derive the distinct roots and other properties
  sorry
}

end distinct_roots_difference_l821_821138


namespace gcd_180_270_eq_90_l821_821672

-- Problem Statement
theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := 
by 
  sorry

end gcd_180_270_eq_90_l821_821672


namespace unique_solution_l821_821339

noncomputable def pair_satisfying_equation (m n : ℕ) : Prop :=
  2^m - 1 = 3^n

theorem unique_solution : ∀ (m n : ℕ), m > 0 → n > 0 → pair_satisfying_equation m n → (m, n) = (2, 1) :=
by
  intros m n m_pos n_pos h
  sorry

end unique_solution_l821_821339


namespace Jaime_spending_on_lunch_l821_821784

-- Definitions corresponding to the conditions
def savings_per_week := 50
def total_savings_after_5_weeks := 135
def weeks := 5

-- The main statement corresponding to the problem
theorem Jaime_spending_on_lunch (savings_per_week * weeks = 250) 
  (total_savings_after_5_weeks = 135) 
  (savings_diff = 250 - 135) 
  (lunch_count := 2): 
  savings_diff / lunch_count = 57.50 := by
sorry

end Jaime_spending_on_lunch_l821_821784


namespace factor_x_squared_minus_64_l821_821795

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821795


namespace probability_correct_l821_821284

-- Define the bag of balls with labels from 1 to 8
def balls := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability calculation considering the scenario of drawing with replacement
noncomputable def probability_not_less_than_15 : ℚ :=
  let total_outcomes := (balls.length : ℚ) * (balls.length : ℚ) in
  let favorable_outcomes := 3 in
  favorable_outcomes / total_outcomes

-- State the theorem to be proved
theorem probability_correct :
  probability_not_less_than_15 = 3 / 64 :=
sorry

end probability_correct_l821_821284


namespace min_value_x2_plus_y2_l821_821601

variable {x y : ℝ}

theorem min_value_x2_plus_y2 (hx : x ^ 2 + 2 * x * y - 3 * y ^ 2 = 1) :
  ∃ t : ℝ, t ≠ 0 ∧ x = (t + (1 / t)) / 4 ∧ y = (t - (1 / t)) / 4 ∧ x ^ 2 + y ^ 2 = (sqrt 5 + 1) / 4 :=
by
  sorry

end min_value_x2_plus_y2_l821_821601


namespace find_lambda_l821_821439

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821439


namespace salt_weight_l821_821286

theorem salt_weight {S : ℝ} (h1 : 16 + S = 46) : S = 30 :=
by
  sorry

end salt_weight_l821_821286


namespace initial_set_3_3_3_is_valid_initial_set_2_2_2_is_not_valid_l821_821235

def transformation (a b c : ℤ) : ℤ × ℤ × ℤ :=
  if a ≤ b ∧ a ≤ c then (b, c, b + c - 1)
  else if b ≤ a ∧ b ≤ c then (a, c, a + c - 1)
  else (a, b, a + b - 1)

def transformation_sequence (a b c : ℤ) (n : ℕ) : ℤ × ℤ × ℤ :=
  Nat.iterate n (λ abc, transformation abc.1 abc.2 abc.3) (a, b, c)

def valid_initial_set (a b c : ℤ) : Prop :=
  ∃ n, transformation_sequence a b c n = (17, 1967, 1983)

theorem initial_set_3_3_3_is_valid : valid_initial_set 3 3 3 := by
  sorry

theorem initial_set_2_2_2_is_not_valid : ¬ valid_initial_set 2 2 2 := by
  sorry

end initial_set_3_3_3_is_valid_initial_set_2_2_2_is_not_valid_l821_821235


namespace expected_steps_to_opposite_face_l821_821746

section ant_on_cube

-- Define the faces
inductive Face
| A | B1 | B2 | B3 | B4 | C

open Face

-- Define the probabilities and expectations
def E : Face → ℝ
| C   := 0
| B1  := 2 + (1/2) * E A
| B2  := 2 + (1/2) * E A
| B3  := 2 + (1/2) * E A
| B4  := 2 + (1/2) * E A
| A   := 1 + (2 + (1/2) * E A)

theorem expected_steps_to_opposite_face : E A = 6 :=
by sorry

end ant_on_cube

end expected_steps_to_opposite_face_l821_821746


namespace cost_of_largest_pot_l821_821155

theorem cost_of_largest_pot (x : ℝ) (h1 : 6 * (x + 1/20) = 33/4) : (x + 1/2) = 13/8 :=
by
  have h2 : x = 9/8 :=
  sorry
  rw [h2]
  norm_num
  sorry

end cost_of_largest_pot_l821_821155


namespace sequence_general_formula_l821_821111

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  let a : ℕ → ℝ := λ n, if n = 1 then 1 else (2 * a (n - 1)) / (2 + a (n - 1))
  in a n = 2 / (n + 1) :=
sorry

end sequence_general_formula_l821_821111


namespace work_ratio_l821_821643

-- Definitions based on the given conditions
variables (α n R : Real)
variables (p10 p20 V1 V2 : Real)
variables (k1 k2 : Real := (p10 / V1))
variables (p : Real -> Real := λ V, k1 * V)

-- Work done in the first and second processes
noncomputable def W1 : Real :=
  k1 * ((V2^2 - V1^2) / 2)

noncomputable def W2 : Real :=
  2 * k1 * ((V2^2 - V1^2) / 2)

-- Proof statement
theorem work_ratio : W2 / W1 = 2 :=
by 
  sorry

end work_ratio_l821_821643


namespace sum_of_center_and_radius_l821_821342

-- Define the given equation as an axiom
axiom circle_eqn : ∀ x y : ℝ, x^2 + 2 * x - 4 * y - 7 = -y^2 + 8 * x

-- Define the center and radius
def center_x := 3
def center_y := 2
def radius := 2 * Real.sqrt 5

-- Define the sum c + d + s
def c_d_s_sum := center_x + center_y + radius

-- Theorem statement to be proved
theorem sum_of_center_and_radius : 
  c_d_s_sum = 5 + 2 * Real.sqrt 5 :=
sorry

end sum_of_center_and_radius_l821_821342


namespace width_of_room_l821_821122

theorem width_of_room (area : ℝ) (wall_length : ℝ) (h_area : area = 12.0) (h_wall_length : wall_length = 1.5) : 
  ∃ width : ℝ, width = 8.0 :=
by {
  have h1 : area / wall_length = 8.0,
  { rw [h_area, h_wall_length, ←div_eq_mul_one_div, div_self, mul_one]; linarith, },
  use area / wall_length,
  exact h1,
}

end width_of_room_l821_821122


namespace sqrt27_times_cbrt125_eq_15_sqrt3_l821_821187

theorem sqrt27_times_cbrt125_eq_15_sqrt3 : (Real.sqrt 27) * (Real.cbrt 125) = 15 * (Real.sqrt 3) := 
by 
  sorry

end sqrt27_times_cbrt125_eq_15_sqrt3_l821_821187


namespace complex_in_fourth_quadrant_l821_821105

def complex_number := complex (frac 1) ((1 + complex.i)^2 + 1)
def simplified_complex_number := complex (frac 1 5) (- (frac 2 5))

theorem complex_in_fourth_quadrant :
  simplified_complex_number.im < 0 ∧ simplified_complex_number.re > 0 :=
by sorry

end complex_in_fourth_quadrant_l821_821105


namespace equation_of_line_l821_821241

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := x^2 + 4 * x + 4

-- Define the line equation with parameters m and b
def line (m b x : ℝ) : ℝ := m * x + b

-- Define the point of intersection with the parabola on the line x = k
def intersection_point_parabola (k : ℝ) : ℝ := parabola k

-- Define the point of intersection with the line on the line x = k
def intersection_point_line (m b k : ℝ) : ℝ := line m b k

-- Define the vertical distance between the points on x = k
def vertical_distance (k m b : ℝ) : ℝ :=
  abs ((parabola k) - (line m b k))

-- Define the condition that vertical distance is exactly 4 units
def intersection_distance_condition (k m b : ℝ) : Prop :=
  vertical_distance k m b = 4

-- The line passes through point (2, 8)
def passes_through_point (m b : ℝ) : Prop :=
  line m b 2 = 8

-- Non-zero y-intercept condition
def non_zero_intercept (b : ℝ) : Prop := 
  b ≠ 0

-- The final theorem stating the required equation of the line
theorem equation_of_line (m b : ℝ) (h1 : ∃ k, intersection_distance_condition k m b)
  (h2 : passes_through_point m b) (h3 : non_zero_intercept b) : 
  (m = 12 ∧ b = -16) :=
by
  sorry

end equation_of_line_l821_821241


namespace incorrect_proposition_b_l821_821685

theorem incorrect_proposition_b :
  ∀ (P Q: Plane) (l : Line),
  (∀ P' : Plane, P ≠ P' → l ∈ P → Parallel l P' → Parallel P P') → False :=
sorry

end incorrect_proposition_b_l821_821685


namespace sum_T_l821_821997

def T (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, (-1) ^ (Int.toNat ∘ Int.floor)((k : ℤ - 1) / 2) * (k + 1)

theorem sum_T : T 19 + T 34 + T 51 = 0 :=
by
  sorry

end sum_T_l821_821997


namespace simplify_trig_identity_l821_821186

-- Define the main theorem with given conditions.
theorem simplify_trig_identity (α : Real) : 
  cos^2 (π / 4 - α) - sin^2 (π / 4 - α) = sin (2 * α) := 
by 
  sorry

end simplify_trig_identity_l821_821186


namespace mara_needs_to_jog_210_minutes_on_ninth_day_l821_821153

-- Define the conditions first
def minutes_jogged_in_six_days : ℕ := 6 * 80
def minutes_jogged_in_two_days : ℕ := 2 * 105
def total_minutes_required : ℕ := 9 * 100

-- Define the proof problem
theorem mara_needs_to_jog_210_minutes_on_ninth_day :
  minutes_jogged_in_six_days + minutes_jogged_in_two_days + x = total_minutes_required → 
  x = 210 :=
by
  -- Given the sum of jogged minutes for the first 8 days
  let total_minutes_so_far := minutes_jogged_in_six_days + minutes_jogged_in_two_days
  
  -- The total required minutes over 9 days
  have total_required : total_minutes_required = 900 := rfl
  
  -- The total minutes required for the ninth day to maintain the average
  have jog_minutes_ninth_day : x = total_minutes_required - total_minutes_so_far := by
    simp [total_minutes_required, total_minutes_so_far]
  
  -- Conclusion
  show x = 210 from jog_minutes_ninth_day

end mara_needs_to_jog_210_minutes_on_ninth_day_l821_821153


namespace find_sum_l821_821204

def f (x : ℝ) : ℝ := sorry

axiom f_non_decreasing : ∀ {x1 x2 : ℝ}, 0 ≤ x1 → x1 ≤ 1 → 0 ≤ x2 → x2 ≤ 1 → x1 < x2 → f x1 ≤ f x2
axiom f_at_0 : f 0 = 0
axiom f_scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (x / 3) = (1 / 2) * f x
axiom f_symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → f (1 - x) = 1 - f x

theorem find_sum :
  f (1 / 3) + f (2 / 3) + f (1 / 9) + f (1 / 6) + f (1 / 8) = 7 / 4 :=
by
  sorry

end find_sum_l821_821204


namespace min_terminals_three_connector_windmill_l821_821294

-- Definitions of the 3-connector and windmill
structure ThreeConnector (G : Type) :=
  (vertices : Set G)
  (directly_communicates : G → G → Prop)
  (three_connector_condition : ∀ x y z : G, x ∈ vertices → y ∈ vertices → z ∈ vertices → 
    (directly_communicates x y ∨ directly_communicates y z ∨ directly_communicates z x))

structure Windmill (G : Type) :=
  (terminals : Finset G)
  (hub : G)
  (directly_communicates : ∀ t : G, t ∈ terminals → G → Prop)
  (contains_windmill_condition : ∀ i : ℕ, i < terminals.card / 2 →
    let x := terminals.to_list.nth_le (2 * i) sorry in
    let y := terminals.to_list.nth_le (2 * i + 1) sorry in
    directly_communicates x y ∧ directly_communicates hub x ∧ directly_communicates hub y)

-- Main theorem statement
theorem min_terminals_three_connector_windmill (n : ℕ) : 
  ∃ f, (∀ (G : Type) [ThreeConnector G], ThreeConnector.vertices G.card = f → 
  (∃ W : Windmill G, W.terminals.card = 2 * n + 1)) ∧ 
  (∀ f', f' < f → ∀ (G : Type) [ThreeConnector G], 
  ¬(exists W : Windmill G, W.terminals.card = 2 * n + 1)) :=
begin
  sorry
end

end min_terminals_three_connector_windmill_l821_821294


namespace find_x_values_l821_821006

theorem find_x_values (x : ℝ) :
  x^3 - 9 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (6 < x) :=
by
  sorry

end find_x_values_l821_821006


namespace angle_APB_eq_60_line_CD_equations_circle_through_A_P_M_fixed_point_l821_821037

-- Definitions for the first question
def circle (x y : ℝ) := x^2 + (y - 2)^2 = 1
def line_l (x y : ℝ) := x - 2 * y = 0
def point_P := (0, 0 : ℝ)

-- Proof of the first question, just the statement
theorem angle_APB_eq_60:
  ∀ A B : ℝ × ℝ,
  tangent A (circle) point_P ∧ tangent B (circle) point_P →
  angle A point_P B = 60 := by sorry

-- Definitions for the second question
def point_P2 := (2, 1 : ℝ)
def line_CD (k : ℝ) (x y : ℝ) := y - 1 = k * (x - 2)

-- Proof of the second question, just the statement
theorem line_CD_equations:
  ∃ k : ℝ, (k = -1 ∨ k = -1/7) →
  line_eq (k) (point_P2) -- line equation for CD with given P2
  := by sorry

-- Definitions for the third question
def midpoint_MP (m : ℝ) := (m, m / 2 + 1 : ℝ)

-- Proof of the third question, just the statement
theorem circle_through_A_P_M_fixed_point :
  ∃ m : ℝ, ∃ (x y : ℝ), x^2 + y^2 - 2 * y - m * (2 * x + y - 2) = 0 →
  (x = 4 / 5 ∧ y = 2 / 5) := by sorry

end angle_APB_eq_60_line_CD_equations_circle_through_A_P_M_fixed_point_l821_821037


namespace second_discount_percentage_l821_821649

-- Defining the conditions
def initial_price : ℝ := 550
def first_discount : ℝ := 18 / 100
def sale_price : ℝ := 396.88

-- Defining the Lean statement to prove the second discount percentage
theorem second_discount_percentage :
  ∃ (x : ℝ), (initial_price * (1 - first_discount)) * (1 - x/100) = sale_price ∧ x ≈ 12 :=
by
  sorry

end second_discount_percentage_l821_821649


namespace prob_two_sunny_days_l821_821199

-- Define the probability of rain and sunny
def probRain : ℚ := 3 / 4
def probSunny : ℚ := 1 / 4

-- Define the problem statement
theorem prob_two_sunny_days : (10 * (probSunny^2) * (probRain^3)) = 135 / 512 := 
by
  sorry

end prob_two_sunny_days_l821_821199


namespace parallel_vectors_lambda_l821_821488

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821488


namespace length_of_platform_l821_821739

theorem length_of_platform 
  (length_of_train : ℕ)
  (speed_kmph : ℕ)
  (time_seconds : ℕ)
  (converted_speed : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 300)
  (total_cycle : time_seconds = 15)
  (length_of_train_correct : length_of_train = 250)
  (length_of_platform : 300 - length_of_train_correct = 50) :
  length_of_platform = 50 :=
by sorry

end length_of_platform_l821_821739


namespace volume_of_rectangular_prism_l821_821817

    theorem volume_of_rectangular_prism (height base_perimeter: ℝ) (h: height = 5) (b: base_perimeter = 16) :
      ∃ volume, volume = 80 := 
    by
      -- Mathematically equivalent proof goes here
      sorry
    
end volume_of_rectangular_prism_l821_821817


namespace sum_is_1275_last_number_is_755_l821_821731

def initial_sequence : List ℕ := List.range' 1 51

def step_operation (seq : List ℕ) : List ℕ :=
if seq.length < 4 then seq else (seq.drop 4).append [(seq.take 4).sum]

def final_sequence : List ℕ := 
  let rec iter (s : List ℕ) : List ℕ :=
    if s.length < 4 then s else iter (step_operation s)
  iter initial_sequence

-- Define the sum of the final sequence
def sum_final_sequence : ℕ := final_sequence.sum

-- Define the last single number written
def last_number_written : ℕ := 
  let seq := initial_sequence
  let rec iter (s : List ℕ) : List ℕ :=
    if s.length < 4 then s else iter (step_operation s)
  (iter seq).reverse.headD 0

theorem sum_is_1275 : sum_final_sequence = 1275 := sorry

theorem last_number_is_755 : last_number_written = 755 := sorry

end sum_is_1275_last_number_is_755_l821_821731


namespace tan_theta_value_l821_821508

theorem tan_theta_value (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : Real.cos θ = -4/5) : 
  Real.tan θ = -3/4 :=
  sorry

end tan_theta_value_l821_821508


namespace find_lambda_l821_821432

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821432


namespace ordered_pairs_count_l821_821499

theorem ordered_pairs_count :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 + Nat.gcd p.1 p.2 = 33}.card = 21 := 
sorry

end ordered_pairs_count_l821_821499


namespace length_of_bridge_l821_821695

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_per_hr : ℕ)
  (crossing_time_sec : ℕ)
  (h_train_length : length_of_train = 100)
  (h_speed : speed_km_per_hr = 45)
  (h_time : crossing_time_sec = 30) :
  ∃ (length_of_bridge : ℕ), length_of_bridge = 275 :=
by
  -- Convert speed from km/hr to m/s
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  -- Total distance the train travels in crossing_time_sec
  let total_distance := speed_m_per_s * crossing_time_sec
  -- Length of the bridge
  let bridge_length := total_distance - length_of_train
  use bridge_length
  -- Skip the detailed proof steps
  sorry

end length_of_bridge_l821_821695


namespace remainder_of_N_mod_103_l821_821142

noncomputable def N : ℕ :=
  sorry -- This will capture the mathematical calculation of N using the conditions stated.

theorem remainder_of_N_mod_103 : (N % 103) = 43 :=
  sorry

end remainder_of_N_mod_103_l821_821142


namespace factorization_of_x_squared_minus_64_l821_821801

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821801


namespace harmonic_series_inequality_l821_821144

theorem harmonic_series_inequality (n : ℕ) (h : n ≥ 2) : 
  let a : ℕ → ℚ := λ n, ∑ i in finset.range (n + 1), (1 / i.succ : ℚ)
  in (a n) ^ 2 > 2 * (finset.sum (finset.range n) (λ k, a (k + 1) / (k + 1))) := by
  sorry

end harmonic_series_inequality_l821_821144


namespace parallel_vectors_lambda_l821_821471

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821471


namespace Katie_marble_count_l821_821988

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end Katie_marble_count_l821_821988


namespace smallest_positive_period_l821_821049

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem smallest_positive_period (ω : ℝ) (hω : ω > 0)
  (H : ∀ x1 x2 : ℝ, abs (f ω x1 - f ω x2) = 2 → abs (x1 - x2) = Real.pi / 2) :
  ∃ T > 0, T = Real.pi ∧ (∀ x : ℝ, f ω (x + T) = f ω x) := 
sorry

end smallest_positive_period_l821_821049


namespace exists_unique_alpha_and_f_l821_821351

open Function

noncomputable def main_theorem : Prop :=
  ∃ (α : ℤ) (f : ℕ → ℕ), 
    (∀ (m n : ℕ), f (m * n ^ 2) = f (m * n) + α * f n) ∧
    (∀ (n : ℕ) (p : ℕ), Prime p → p ∣ n → f p ≠ 0 ∧ f p ∣ f n) → 
    (α = 1 ∧ ∀ n : ℕ, ∃ (c : ℕ → ℕ), (∀ (p : ℕ), Prime p → f p = c p) ∧ 
                       (∀ (p : ℕ) (e : ℕ), f (p ^ e) = e * f p) ∧ 
                       (∀ (n : ℕ), f n = ∑ (p ∣ n), Prime p → c p * n.factors.count p))

theorem exists_unique_alpha_and_f : main_theorem := 
  sorry

end exists_unique_alpha_and_f_l821_821351


namespace plane_regions_divided_by_lines_4_l821_821336

theorem plane_regions_divided_by_lines_4 :
  ∀ (x y : ℝ), ((y = 3 * x) ∨ (y = x / 3)) → (plane_divided_into_regions 4) :=
sorry

end plane_regions_divided_by_lines_4_l821_821336


namespace parallel_vectors_lambda_l821_821449

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821449


namespace B_work_rate_l821_821707

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end B_work_rate_l821_821707


namespace smallest_n_square_partition_l821_821678

theorem smallest_n_square_partition (n : ℕ) (h : ∃ a b : ℕ, a ≥ 1 ∧ b ≥ 1 ∧ n = 40 * a + 49 * b) : n ≥ 2000 :=
by sorry

end smallest_n_square_partition_l821_821678


namespace vacation_months_away_l821_821979

theorem vacation_months_away (total_savings : ℕ) (pay_per_check : ℕ) (checks_per_month : ℕ) :
  total_savings = 3000 → pay_per_check = 100 → checks_per_month = 2 → 
  total_savings / pay_per_check / checks_per_month = 15 :=
by 
  intros h1 h2 h3
  sorry

end vacation_months_away_l821_821979


namespace parallel_vectors_l821_821479

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821479


namespace number_of_votes_in_favor_of_candidate_A_l821_821094

-- Definition for total votes and percentages
def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 0.15
def valid_percentage : ℚ := 1 - invalid_percentage
def candidate_A_percentage : ℚ := 0.70

-- Total valid votes
def total_valid_votes : ℕ := (total_votes * valid_percentage).toInt

-- Prove the number of valid votes polled in favor of candidate A
theorem number_of_votes_in_favor_of_candidate_A : 
  (total_valid_votes * candidate_A_percentage).toInt = 333200 :=
by 
  sorry

end number_of_votes_in_favor_of_candidate_A_l821_821094


namespace OM_angle_bisector_of_QOT_l821_821699

variable {Point : Type} [MetricSpace Point]
variables {S Q T M O : Point}

-- Conditions definitions
def is_angle_bisector (A B C D : Point) : Prop := 
  ∃ (line : Set Point), line ∈ LinesThrough A B ∩ LinesThrough A C ∩ LinesThrough A D

def angle_eq_sum (A B C D E : Point) : Prop := 
  ∠ B C D = ∠ C D E + ∠ D E A

-- Lean statement to prove
theorem OM_angle_bisector_of_QOT (h1 : is_angle_bisector S M Q T)
                                 (h2 : angle_eq_sum O Q T Q T S) :
  is_angle_bisector O M Q T := 
sorry

end OM_angle_bisector_of_QOT_l821_821699


namespace quart_paint_coverage_l821_821911

theorem quart_paint_coverage :
  (paint_cost_per_quart : ℝ) (total_paint_cost : ℝ) (cube_edge : ℝ) (coverage : ℝ) 
  (paint_cost_per_quart = 3.20)
  (total_paint_cost = 16)
  (cube_edge = 10)
  (coverage = (6 * cube_edge^2) / (total_paint_cost / paint_cost_per_quart)) 
  : coverage = 120 :=
sorry

end quart_paint_coverage_l821_821911


namespace number_of_arrangements_l821_821945

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821945


namespace factor_difference_of_squares_l821_821786

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821786


namespace geometric_seq_condition_l821_821056

variable (n : ℕ) (a : ℕ → ℝ)

-- The definition of a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) * a (n + 1) = a n * a (n + 2)

-- The main theorem statement
theorem geometric_seq_condition :
  (is_geometric_seq a n → ∀ n, |a n| ≥ 0) →
  ∃ (a : ℕ → ℝ), (∀ n, a n * a (n + 2) = a (n + 1) * a (n + 1)) →
  (∀ m, a m = 0 → ¬(is_geometric_seq a n)) :=
sorry

end geometric_seq_condition_l821_821056


namespace minimum_sewage_pipe_length_l821_821244

theorem minimum_sewage_pipe_length {A B C P : Type} [metric_space A] [metric_space B] [metric_space C]
  (distance_AB : dist A B = 5) (distance_BC : dist B C = 5) (distance_AC : dist A C = 6)
  (on_side_P : P ∈ line_segment ℝ A B ∨ P ∈ line_segment ℝ B C ∨ P ∈ line_segment ℝ A C) :
  AP + BP + CP = 10 :=
sorry

end minimum_sewage_pipe_length_l821_821244


namespace parallel_vectors_l821_821484

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821484


namespace length_of_rectangle_fraction_l821_821636

def radius_of_circle_eq_side_of_square (area_square : ℕ) : ℕ :=
  Int.natAbs (Real.sqrt area_square).toInt

def area_rectangle (length breadth : ℕ) : ℕ :=
  length * breadth

theorem length_of_rectangle_fraction
  (area_square : ℕ) (area_square_value : area_square = 1225)
  (area_rect : ℕ) (area_rect_value : area_rect = 140)
  (breadth : ℕ) (breadth_value : breadth = 10)
  : (area_rectangle (area_rect / breadth) breadth : ℚ) / (radius_of_circle_eq_side_of_square area_square : ℚ)
    = 2 / 5 := by
  sorry

end length_of_rectangle_fraction_l821_821636


namespace coconut_grove_x_value_l821_821931

theorem coconut_grove_x_value :
  ∀ (x : ℕ), 
    let trees_40 := 40 * (x + 2),
        trees_120 := 120 * x,
        trees_180 := 180 * (x - 2),
        total_nuts := trees_40 + trees_120 + trees_180,
        total_trees := 3 * x,
        average_yield := 100 in
    total_nuts / total_trees = average_yield → x = 7 :=
by
  sorry

end coconut_grove_x_value_l821_821931


namespace validate_tax_authority_notification_l821_821583

/-
Federal Law 173-FZ states:
1. Currency transactions between residents of the Russian Federation are prohibited, unless exceptions apply.
2. Settlements within Russia must be made in Russian rubles.
-/

def law_173_fz (a b : Type) [resident a] [resident b] (amount : ℝ) 
  (currency : Type) [euro_currency currency] : Prop :=
  ∀ (transaction_in_rubles : Type → Type → ℝ → Prop),
  (transaction_in_rubles a b amount) → (amount ∉ euro_currency)

def transaction_agreement_violation : Prop :=
  let Mikhail := resident.mk JSC_New_Years_Joy
  let Valentin := resident.mk LLC_Holiday_Comes_to_You
  let transaction_amount := 2700
  let currency := euro_currency.mk 
  law_173_fz Mikhail Valentin transaction_amount currency

theorem validate_tax_authority_notification :
  transaction_agreement_violation → tax_authority_correct :=
sorry

end validate_tax_authority_notification_l821_821583


namespace find_ellipse_equation_prove_area_constant_l821_821392

-- Define the conditions in Lean 4
variables {a b c k m x1 x2 y1 y2 : ℝ}

-- Given the equation of an ellipse and the conditions
def ellipse (a b : ℝ) (a_gt_b: a > b) :=
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

def foci_distance (c a : ℝ) (eccentricity : ℝ) :=
  c / a = eccentricity

def distance_sum (a : ℝ) :=
  2 * a = 4 * sqrt 2

def slope_condition (k1 k2 : ℝ) :=
  k1 * k2 = -1/2

-- The coordinates of points A and B
def line_intersects (x1 x2 y1 y2 k m : ℝ) :=
  y1 = k * x1 + m ∧ y2 = k * x2 + m

-- Part (1) Proving the equation of the ellipse
theorem find_ellipse_equation :
  (ellipse a b (by linarith)) →
  (foci_distance c a (sqrt 2 / 2)) →
  (distance_sum a) →
  (a = 2 * sqrt 2) →
  (c = 2) →
  (b^2 = a^2 - c^2) →
  a = 2 * sqrt 2 ∧ b = 2 :=
sorry

-- Part (2) Proving the area of the triangle is constant
theorem prove_area_constant
  (k m x1 x2 y1 y2 : ℝ)
  (hk : line_intersects x1 x2 y1 y2 k m)
  (heq : ellipse a b (by linarith))
  (hsum : distance_sum a)
  (hslope : slope_condition y1 y2)
  (hfocus : foci_distance c a (sqrt 2 / 2)) :
  (2 * sqrt 2 = 2 * sqrt 2) :=
sorry

end find_ellipse_equation_prove_area_constant_l821_821392


namespace parallel_vectors_lambda_l821_821493

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821493


namespace math_proof_problem_l821_821684

noncomputable def prop_A := ∀ x : ℝ, (x ∈ [0, 3]) ↔ (x ≠ 1 ∧ (x + 1) / ((x - 1) ^ 2) > 1)

noncomputable def prop_B (a : ℝ) := 
  ∀ x : ℝ, (x^2 - ax - 4 = 0) → (x > 2 ∨ x < -1) ↔ 0 < a ∧ a < 3

noncomputable def prop_C : ℝ → ℝ := 
  fun x => (x^4 - 1) / (x^2 + 1)

noncomputable def prop_D (f : ℝ → ℝ) (x : ℝ) := 
  ∀ y : ℝ, (-2 ≤ y ∧ y ≤ 2) ↔ (-1/2 ≤ (2 * x - 1) ∧ (2 * x - 1) ≤ 3/2)

theorem math_proof_problem :
  (∃ a : ℝ, prop_B a) ∧ (∀ x : ℝ, prop_C x = x^2 - 1) ∧ prop_D f x :=
by
  sorry

end math_proof_problem_l821_821684


namespace trigonometric_identity_l821_821362

theorem trigonometric_identity : 
  sin (20 * pi / 180) * sin (10 * pi / 180) - cos (10 * pi / 180) * sin (70 * pi / 180) = -√3 / 2 :=
by
  sorry -- Proof to be filled in


end trigonometric_identity_l821_821362


namespace total_eggs_needed_l821_821890

-- Define the conditions
def eggsFromAndrew : ℕ := 155
def eggsToBuy : ℕ := 67

-- Define the total number of eggs
def totalEggs : ℕ := eggsFromAndrew + eggsToBuy

-- The theorem to be proven
theorem total_eggs_needed : totalEggs = 222 := by
  sorry

end total_eggs_needed_l821_821890


namespace line_inclination_angle_l821_821227

theorem line_inclination_angle (x y : ℝ) (θ : ℝ) :
  (x - sqrt 3 * y + 1 = 0) →
  θ = 30 :=
by
  intro h
  sorry

end line_inclination_angle_l821_821227


namespace tank_dimension_l821_821306

theorem tank_dimension (cost_per_sf : ℝ) (total_cost : ℝ) (length1 length3 : ℝ) (surface_area : ℝ) (dimension : ℝ) :
  cost_per_sf = 20 ∧ total_cost = 1520 ∧ 
  length1 = 4 ∧ length3 = 2 ∧ 
  surface_area = total_cost / cost_per_sf ∧
  12 * dimension + 16 = surface_area → dimension = 5 :=
by
  intro h
  obtain ⟨hcps, htac, hl1, hl3, hsa, heq⟩ := h
  sorry

end tank_dimension_l821_821306


namespace lambda_parallel_l821_821458

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821458


namespace exists_subset_sum_l821_821148

noncomputable theory

-- Define the conditions as Lean definitions
def sequence_sum (x : Fin 100 → ℝ) := ∑ i, x i = 1

def sequence_difference (x : Fin 100 → ℝ) := ∀ (k : Fin 99), |x k.succ - x k| < 1 / 50

-- Define the Lean theorem statement
theorem exists_subset_sum 
  (x : Fin 100 → ℝ) 
  (h_sum : sequence_sum x) 
  (h_diff : sequence_difference x) : 
  ∃ (i : Fin 50 → Fin 100) (h : StrictMono i), 
    (49 / 100 : ℝ) ≤ ∑ j, x (i j) ∧ ∑ j, x (i j) ≤ 51 / 100 :=
sorry

end exists_subset_sum_l821_821148


namespace next_shared_meeting_day_l821_821631

-- Definitions based on the conditions:
def dramaClubMeetingInterval : ℕ := 3
def choirMeetingInterval : ℕ := 5
def debateTeamMeetingInterval : ℕ := 7

-- Statement to prove:
theorem next_shared_meeting_day : Nat.lcm (Nat.lcm dramaClubMeetingInterval choirMeetingInterval) debateTeamMeetingInterval = 105 := by
  sorry

end next_shared_meeting_day_l821_821631


namespace total_food_amount_l821_821370

-- Define constants for the given problem
def chicken : ℕ := 16
def hamburgers : ℕ := chicken / 2
def hot_dogs : ℕ := hamburgers + 2
def sides : ℕ := hot_dogs / 2

-- Prove the total amount of food Peter will buy is 39 pounds
theorem total_food_amount : chicken + hamburgers + hot_dogs + sides = 39 := by
  sorry

end total_food_amount_l821_821370


namespace two_a_minus_b_eq_two_plus_sqrt_thirteen_l821_821399

theorem two_a_minus_b_eq_two_plus_sqrt_thirteen :
  ∃ a b : ℝ, a = (7 - Real.sqrt 13).floor ∧ b = (7 - Real.sqrt 13) - a ∧ (2 * a - b) = (2 + Real.sqrt 13) :=
by sorry

end two_a_minus_b_eq_two_plus_sqrt_thirteen_l821_821399


namespace seating_arrangements_l821_821952

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821952


namespace count_valid_n_l821_821360

theorem count_valid_n : 
  let num_valid_n := 
    (finset.range 215).card   +  -- For 7m with 1 <= m <= 214
    (finset.range 214).card   +  -- For 7m + 1 with 0 <= m <= 213
    (finset.range 214).card   +  -- For 7m + 3 with 0 <= m <= 213
    (finset.range 214).card     -- For 7m + 4 with 0 <= m <= 213
  in num_valid_n = 856 := 
begin
  -- Here we would put the proof based on the translated steps,
  -- but for now, we use sorry to mark it as unproven.
  sorry
end

end count_valid_n_l821_821360


namespace necessary_but_not_sufficient_condition_l821_821867

-- Define a structure to represent pairs and conditions
structure SamplePoint (n : ℕ) :=
  (x : Fin n → ℝ)
  (y : Fin n → ℝ)
  (b a : ℝ)

-- Define the linear regression equation
def satisfies_linear_regression (sp : SamplePoint 10) (x0 y0 : ℝ) : Prop :=
  y0 = sp.b * x0 + sp.a

-- Define the centroid condition
def is_centroid (sp : SamplePoint 10) (x0 y0 : ℝ) : Prop :=
  x0 = (finset.univ : finset (fin 10)).sum (λ i, sp.x i) / 10 ∧
  y0 = (finset.univ : finset (fin 10)).sum (λ i, sp.y i) / 10

-- The statement that needs to be proved
theorem necessary_but_not_sufficient_condition (sp : SamplePoint 10) :
  ∀ (x0 y0 : ℝ), satisfies_linear_regression sp x0 y0 →
  is_centroid sp x0 y0 → 
    "The pair (x0, y0) satisfies the linear regression equation is a necessary but not sufficient condition for (x0, y0) being the centroid of the sample points." :=
  sorry

end necessary_but_not_sufficient_condition_l821_821867


namespace solve_equation_solve_proportion_l821_821609

theorem solve_equation (x : ℚ) :
  (3 + x) * (30 / 100) = 4.8 → x = 13 :=
by sorry

theorem solve_proportion (x : ℚ) :
  (5 / x) = (9 / 2) / (8 / 5) → x = (16 / 9) :=
by sorry

end solve_equation_solve_proportion_l821_821609


namespace jerry_trips_l821_821546

-- Define the conditions
def trays_per_trip : Nat := 8
def trays_table1 : Nat := 9
def trays_table2 : Nat := 7

-- Define the proof problem
theorem jerry_trips :
  trays_table1 + trays_table2 = 16 →
  (16 / trays_per_trip) = 2 :=
by
  sorry

end jerry_trips_l821_821546


namespace find_speed_of_faster_train_l821_821664

noncomputable def speed_of_faster_train
  (length_each_train_m : ℝ)
  (speed_slower_kmph : ℝ)
  (time_pass_s : ℝ) : ℝ :=
  let distance_km := (2 * length_each_train_m / 1000)
  let time_pass_hr := (time_pass_s / 3600)
  let relative_speed_kmph := (distance_km / time_pass_hr)
  let speed_faster_kmph := (relative_speed_kmph - speed_slower_kmph)
  speed_faster_kmph

theorem find_speed_of_faster_train :
  speed_of_faster_train
    250   -- length_each_train_m
    30    -- speed_slower_kmph
    23.998080153587715 -- time_pass_s
  = 45 := sorry

end find_speed_of_faster_train_l821_821664


namespace paco_gives_14_cookies_l821_821164

theorem paco_gives_14_cookies
    (initial_cookies : ℕ)
    (eaten_cookies : ℕ)
    (left_cookies : ℕ)
    (initial_condition : initial_cookies = 36)
    (eaten_condition : eaten_cookies = 10)
    (left_condition : left_cookies = 12) :
    initial_cookies - eaten_cookies - left_cookies = 14 :=
by
  have h1 : initial_cookies = 36 := initial_condition
  have h2 : eaten_cookies = 10 := eaten_condition
  have h3 : left_cookies = 12 := left_condition
  calc
    initial_cookies - eaten_cookies - left_cookies
        = 36 - 10 - 12 : by rw [h1, h2, h3]
    ... = 26 - 12      : by rfl
    ... = 14           : by rfl

end paco_gives_14_cookies_l821_821164


namespace find_pairs_l821_821239

def regions_divided (h s : ℕ) : ℕ :=
  1 + s * (s + 1) / 2 + h * (s + 1)

theorem find_pairs (h s : ℕ) :
  regions_divided h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  sorry

end find_pairs_l821_821239


namespace parallel_vectors_l821_821477

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821477


namespace round_to_nearest_hundredth_l821_821605

theorem round_to_nearest_hundredth (x : ℝ) :
  (x = 1.423 → round (x * 100) / 100 = 1.42) ∧
  (x = 3.2387 → round (x * 100) / 100 = 3.24) ∧
  (x = 1.996 → round (x * 100) / 100 = 2.00) :=
by {
  split,
  { intro h, rw h, norm_num1, },
  split,
  { intro h, rw h, norm_num1, },
  { intro h, rw h, norm_num1, },
}

end round_to_nearest_hundredth_l821_821605


namespace vector_dot_product_parallel_l821_821887

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, -1)
noncomputable def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem vector_dot_product_parallel (m : ℝ) (h_parallel : is_parallel a (a.1 + m, a.2 + (-1))) :
  (a.1 * m + a.2 * (-1) = -5 / 2) :=
sorry

end vector_dot_product_parallel_l821_821887


namespace quadratic_roots_5_sqrt_15_l821_821514

theorem quadratic_roots_5_sqrt_15 (k : ℝ) (h : ∀ x, 2 * x^2 - 10 * x + k = 0 ↔ x = 5 + sqrt 15 ∨ x = 5 - sqrt 15) : 
  k = 85 / 8 :=
by
  sorry

end quadratic_roots_5_sqrt_15_l821_821514


namespace hyperbola_eccentricity_l821_821053

theorem hyperbola_eccentricity 
  (a b : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b)
  (h_parabola : ∀ x y : ℝ, y^2 = -4 * (sqrt 2) * x → (x, y) = (-sqrt 2, 0))
  (h_asymptote : ∀ x y : ℝ, (b * x + a * y = 0) → (sqrt 2 * b) / (sqrt (a^2 + b^2)) = sqrt 10 / 5) : 
  sqrt(a^2 + b^2) / a = sqrt 5 / 2 := 
by 
  sorry

end hyperbola_eccentricity_l821_821053


namespace range_of_m_l821_821384

theorem range_of_m (f : ℝ → ℝ) (h1 : ∀ x, differentiable_at ℝ f x) (h2 : ∀ x, deriv f x < 1) (h3 : ∀ m, f (1 - m) - f m > 1 - 2 * m) : ∀ m, m > (1 / 2) :=
begin
  intro m,
  specialize h3 m,
  set g := λ x, f x - x with hgdef,
  have hg1 : ∀ x, differentiable_at ℝ g x,
    from λ x, differentiable_at.sub (h1 x) differentiable_at_id,
  have hg2 : ∀ x, deriv g x < 0,
    from λ x, by { unfold g, simp only [deriv_sub, deriv_id', sub_zero], exact sub_lt_sub_right (h2 x) 1 },
  have hg3 : ∀ x y, x < y → g y < g x,
    from λ x y hxy, by { simpa [← sub_lt_zero, sub_sub_sub_cancel_right] using hg2 (y - x) },
  specialize hg3 (1 - m) m,
  simp [sub_lt_sub_iff_right, sub_right_comm, sub_add_eq_sub_sub, h3] at hg3,
  exact lt_of_lt_of_le zero_lt_one (not_le.1 hg3)
end

end range_of_m_l821_821384


namespace parallel_vectors_lambda_l821_821489

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821489


namespace expand_and_simplify_l821_821347

-- Define the two polynomials P and Q.
def P (x : ℝ) := 5 * x + 3
def Q (x : ℝ) := 2 * x^2 - x + 4

-- State the theorem we want to prove.
theorem expand_and_simplify (x : ℝ) : (P x * Q x) = 10 * x^3 + x^2 + 17 * x + 12 := 
by
  sorry

end expand_and_simplify_l821_821347


namespace bromine_atoms_calculation_l821_821295

-- Given conditions
def Nitrogen_atoms := 1
def Hydrogen_atoms := 4
def molecular_weight := 98
def atomic_weight_Nitrogen := 14.01
def atomic_weight_Hydrogen := 1.01
def atomic_weight_Bromine := 79.90

-- Question to prove:
theorem bromine_atoms_calculation:
  ∃ x : ℕ, molecular_weight = (Nitrogen_atoms * atomic_weight_Nitrogen) + 
                           (Hydrogen_atoms * atomic_weight_Hydrogen) + 
                           (x * atomic_weight_Bromine) ∧ 
             x = 1 := 
by {
  sorry
}

end bromine_atoms_calculation_l821_821295


namespace sum_of_coefficients_l821_821209

theorem sum_of_coefficients (A B C : ℤ)
  (h : ∀ x, x^3 + A * x^2 + B * x + C = (x + 3) * x * (x - 3))
  : A + B + C = -9 :=
sorry

end sum_of_coefficients_l821_821209


namespace profit_share_B_l821_821315

theorem profit_share_B (capital_A capital_B capital_C : ℝ) (profit_A profit_C diff : ℝ) : 
  capital_A = 8000 ∧ capital_B = 10000 ∧ capital_C = 12000 ∧ 
  profit_A = 2000 ∧ diff = 999.9999999999998 → 
  let total_profit := 2000 * ((2 + 2.5 + 3) / 2) in
  let profit_B := (2.5 / (2 + 2.5 + 3)) * total_profit in
  profit_B = 2500 := by
  intros h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hpa := h.2.2.2.1
  have hd := h.2.2.2.2
  sorry

end profit_share_B_l821_821315


namespace weight_of_b_l821_821693

theorem weight_of_b (a b c : ℝ) (h1 : (a + b + c) / 3 = 45) (h2 : (a + b) / 2 = 40) (h3 : (b + c) / 2 = 43) : b = 31 :=
by
  sorry

end weight_of_b_l821_821693


namespace prove_trigonometric_identity1_prove_trigonometric_identity2_l821_821182

noncomputable def trigonometric_identity1 (n : ℕ) (h : 2 ≤ n) : Prop :=
  (∏ k in finset.range (n-1), Real.sin ((k+1 : ℝ) * π / (2 * n))) = Real.sqrt n / (2 ^ (n - 1))

noncomputable def trigonometric_identity2 (n : ℕ) : Prop :=
  (∏ k in finset.range (2 * n) | k % 2 = 1, Real.sin ((k : ℝ) * π / (4 * n))) = Real.sqrt 2 / (2 ^ n)

theorem prove_trigonometric_identity1 (n : ℕ) (h : 2 ≤ n) : trigonometric_identity1 n h :=
  sorry

theorem prove_trigonometric_identity2 (n : ℕ) : trigonometric_identity2 n :=
  sorry

end prove_trigonometric_identity1_prove_trigonometric_identity2_l821_821182


namespace rate_of_interest_l821_821919

theorem rate_of_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h : P > 0 ∧ T = 7 ∧ SI = P / 5 ∧ SI = (P * R * T) / 100) : 
  R = 20 / 7 := 
by
  sorry

end rate_of_interest_l821_821919


namespace prime_factor_sequence_l821_821418

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 2
| 1       => 1
| (n + 1) => a n + a (n - 1)

-- Ensure a is correctly defined via the given recurrence relation
lemma a_recurrence (n : ℕ) : a (n + 2) = a (n + 1) + a n :=
begin
  induction n with n ih,
  { refl },
  { simp [a, ih] },
end

-- Define the core proof statement
theorem prime_factor_sequence (k p : ℕ) (prime_p : Nat.Prime p) 
  (h : p ∣ a (2 * k) - 2) : p ∣ a (2 * k + 1) - 1 :=
begin
  -- using sorry to indicate the proof is omitted
  sorry
end

end prime_factor_sequence_l821_821418


namespace max_real_part_sum_correct_l821_821568

noncomputable def max_real_part_sum : ℝ :=
  let z := λ j : ℕ, 8 * Complex.exp (2 * Complex.pi * Complex.I * j / 10)
  let w := λ j : ℕ, if (Complex.re (z j)) > 0 then z j else if (Complex.im (z j)) > 0 then z j * Complex.I else z j * -Complex.I
  ∑ j in Finset.range 10, Complex.re (w j)

theorem max_real_part_sum_correct : max_real_part_sum = 8 + 8 * Real.sqrt 5 :=
sorry

end max_real_part_sum_correct_l821_821568


namespace sum_first_13_terms_l821_821001

variable {a_n : ℕ → ℝ} (S : ℕ → ℝ)
variable (a_1 d : ℝ)

-- Arithmetic sequence properties
axiom arithmetic_sequence (n : ℕ) : a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms
axiom sum_of_terms (n : ℕ) : S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_specific_terms : a_n 2 + a_n 7 + a_n 12 = 30

-- Theorem to prove
theorem sum_first_13_terms : S 13 = 130 := sorry

end sum_first_13_terms_l821_821001


namespace range_of_a_l821_821852

/-- Problem: Given the propositions p and q as described, determine the correct range for a --/

def p (x1 x2 m : ℝ) (a : ℝ) : Prop :=
  (x1 + x2 = m) ∧ (x1 * x2 = -2) ∧ 
  ∀ m ∈ Icc (-1 : ℝ) 1, a^2 - 5 * a - 3 ≥ abs (x1 - x2)

def q (a : ℝ) : Prop :=
  ∃ (x : ℝ), ax^2 + 2 * x - 1 > 0

def P_and_Q_false (x1 x2 m a : ℝ) : Prop :=
  ¬ (p x1 x2 m a ∧ q a)

def not_p_false (x1 x2 m : ℝ) (a : ℝ) : Prop :=
  ¬ ¬ (p x1 x2 m a)

theorem range_of_a (x1 x2 m a : ℝ): 
  P_and_Q_false x1 x2 m a → not_p_false x1 x2 m a → a ≤ -1 :=
sorry

end range_of_a_l821_821852


namespace period_started_at_7_am_l821_821376

-- Define the end time of the period
def end_time : ℕ := 16 -- 4 pm in 24-hour format

-- Define the total duration in hours
def duration : ℕ := 9

-- Define the start time of the period
def start_time : ℕ := end_time - duration

-- Prove that the start time is 7 am
theorem period_started_at_7_am : start_time = 7 := by
  sorry

end period_started_at_7_am_l821_821376


namespace factor_difference_of_squares_l821_821791

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821791


namespace second_train_length_l821_821666

noncomputable def length_of_second_train (v1 v2 : ℕ) (t : ℕ) (l1 : ℕ) : ℕ :=
  let relative_speed := (v1 + v2) * 1000 / 3600
  let covered_distance := relative_speed * t
  covered_distance - l1

theorem second_train_length :
  length_of_second_train 42 48 12 137 = 163 :=
by
  unfold length_of_second_train
  have : (42 + 48) * 1000 / 3600 = 25, from sorry
  rw this
  have : 25 * 12 = 300, from sorry
  rw this
  have : 300 - 137 = 163, from sorry
  rw this
  exact rfl

end second_train_length_l821_821666


namespace number_of_arrangements_l821_821948

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821948


namespace no_solution_49_minus_t_squared_plus_7_eq_0_l821_821773

theorem no_solution_49_minus_t_squared_plus_7_eq_0 (t : ℂ) : ¬ (sqrt(49 - t^2) + 7 = 0) :=
sorry

end no_solution_49_minus_t_squared_plus_7_eq_0_l821_821773


namespace negation_proposition_l821_821218

theorem negation_proposition :
  (¬ (x ≠ 3 ∧ x ≠ 2) → ¬ (x ^ 2 - 5 * x + 6 ≠ 0)) =
  ((x = 3 ∨ x = 2) → (x ^ 2 - 5 * x + 6 = 0)) :=
by
  sorry

end negation_proposition_l821_821218


namespace T_seq_bound_l821_821020

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2 else 1 / (n * (n + 1))

noncomputable def S_seq (n : ℕ) : ℝ :=
  n ^ 2 * a_seq n

noncomputable def b_seq (n : ℕ) : ℝ :=
  if n = 1 then 0 else S_seq (n - 1) / S_seq n

noncomputable def T_seq (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), b_seq (i + 1)

theorem T_seq_bound (n : ℕ) : T_seq n < n ^ 2 / (n + 1) :=
sorry

end T_seq_bound_l821_821020


namespace constant_function_value_l821_821504

variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g x = 5)

theorem constant_function_value (x : ℝ) : g (3 * x - 7) = 5 :=
by
  apply h
  sorry

end constant_function_value_l821_821504


namespace parallel_vectors_implies_value_of_λ_l821_821427

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821427


namespace expression_equals_12_l821_821681

-- Define the values of a, b, c, and k
def a : ℤ := 10
def b : ℤ := 15
def c : ℤ := 3
def k : ℤ := 2

-- Define the expression to be evaluated
def expr : ℤ := (a - (b - k * c)) - ((a - b) - k * c)

-- Prove that the expression equals 12
theorem expression_equals_12 : expr = 12 :=
by
  -- The proof will go here, leaving a placeholder for now
  sorry

end expression_equals_12_l821_821681


namespace sum_of_13th_diagonal_in_Pascal_triangle_l821_821324

-- Define the Fibonacci function
def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fib (n + 1) + fib n

-- State the problem as a theorem to prove
theorem sum_of_13th_diagonal_in_Pascal_triangle : fib 14 = 610 :=
by
  sorry

end sum_of_13th_diagonal_in_Pascal_triangle_l821_821324


namespace cards_probability_ratio_l821_821349

theorem cards_probability_ratio :
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  q / p = 44 :=
by
  let num_cards := 50
  let num_each := 4
  let num_unique := 12
  let num_drawn := 5
  let total_ways := Nat.choose (num_cards - 2) num_drawn
  let p := num_unique / total_ways
  let q := (num_unique * (num_unique - 1) * num_each) / total_ways
  have : q / p = 44 := sorry
  exact this

end cards_probability_ratio_l821_821349


namespace parallel_vectors_lambda_l821_821445

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821445


namespace value_of_g_at_7_l821_821510

def g (x : ℝ) : ℝ := (x + 2) / (4 * x - 5)

theorem value_of_g_at_7 : g 7 = 9 / 23 := by
  sorry

end value_of_g_at_7_l821_821510


namespace lines_concurrence_or_parallel_l821_821559

variables (O A B C : Point)
variables (triangle_ABC : Triangle A B C)
variables (M N P Q R T : Point)
variables (hM : foot_of_perpendicular O (angle_bisector_interior A triangle_ABC) M)
variables (hN : foot_of_perpendicular O (angle_bisector_exterior A triangle_ABC) N)
variables (hP : foot_of_perpendicular O (angle_bisector_interior B triangle_ABC) P)
variables (hQ : foot_of_perpendicular O (angle_bisector_exterior B triangle_ABC) Q)
variables (hR : foot_of_perpendicular O (angle_bisector_interior C triangle_ABC) R)
variables (hT : foot_of_perpendicular O (angle_bisector_exterior C triangle_ABC) T)

theorem lines_concurrence_or_parallel :
  are_concurrent_or_parallel (line MN) (line PQ) (line RT) :=
sorry

end lines_concurrence_or_parallel_l821_821559


namespace inequality_proof_l821_821845

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821845


namespace xiao_hua_spent_7_yuan_l821_821711

theorem xiao_hua_spent_7_yuan :
  ∃ (a b c d: ℕ), a + b + c + d = 30 ∧
                   ((a = 5 ∧ b = 5 ∧ c = 10 ∧ d = 10) ∨
                    (a = 5 ∧ b = 10 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 5 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 10 ∧ c = 5 ∧ d = 5) ∨
                    (a = 5 ∧ b = 10 ∧ c = 10 ∧ d = 5) ∨
                    (a = 10 ∧ b = 5 ∧ c = 10 ∧ d = 5)) ∧
                   10 * c + 15 * a + 25 * b + 40 * d = 700 :=
by {
  sorry
}

end xiao_hua_spent_7_yuan_l821_821711


namespace surface_area_ratio_l821_821408

-- Given conditions
variables (R1 R2 R3 : ℝ)
-- Assume radii are in the ratio 1:2:3
axiom radii_ratio1 (h1 : R1 ≠ 0) : R2 = 2 * R1
axiom radii_ratio2 (h2 : R1 ≠ 0) : R3 = 3 * R1

-- The Lean statement to prove the desired ratio
theorem surface_area_ratio (h1 : R1 ≠ 0) (h2 : R1 ≠ 0) : 
  (4 * real.pi * R3^2) / (4 * real.pi * R1^2 + 4 * real.pi * R2^2) = 9 / 5 :=
by
  -- Utilize the axioms to substitute R2 and R3 with respective ratios
  rw [radii_ratio1,h1, radii_ratio2,h2]
  -- Substitute surface area formula and simplify
  sorry

end surface_area_ratio_l821_821408


namespace math_problem_l821_821383

noncomputable def circle_center : ℝ × ℝ := (1, 2)
noncomputable def radius_squared : ℝ := 5
noncomputable def circle_equation : Prop :=
  ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = r ↔ r = radius_squared

noncomputable def tangents_equation : Prop :=
  ∀ (k : ℝ),
    k = sqrt 5 / 2 ∨ k = -sqrt 5 / 2 →
    ((y - 2 = k * (x + 2)) ( x y : ℝ))

theorem math_problem :
  circle_equation ∧ tangents_equation :=
by
  sorry

end math_problem_l821_821383


namespace parallel_vectors_lambda_value_l821_821467

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821467


namespace max_solutions_l821_821827

theorem max_solutions (a : ℝ) : 
  (1006^2 - 2012 < a ∧ a ≤ 1006^2 - 1936) → 
  ∃ n : ℕ, 
  (∀ x : ℝ, x ∈ set.Icc (⌊x⌋ : ℝ) (⌊x⌋ + 1) → (⌊x⌋^2 + 2012 * x + a = 0 → x ∈ ℤ)) ∧ n = 89 := 
sorry

end max_solutions_l821_821827


namespace pyramid_volume_l821_821604

-- Define the given conditions
def regular_octagon (A B C D E F G H : Point) : Prop := sorry
def right_pyramid (P A B C D E F G H : Point) : Prop := sorry
def equilateral_triangle (P A D : Point) (side_length : ℝ) : Prop := sorry

-- Define the specific pyramid problem with all the given conditions
noncomputable def volume_pyramid (P A B C D E F G H : Point) (height : ℝ) (base_area : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D E F G H P : Point) 
(h1 : regular_octagon A B C D E F G H)
(h2 : right_pyramid P A B C D E F G H)
(h3 : equilateral_triangle P A D 10) :
  volume_pyramid P A B C D E F G H (5 * Real.sqrt 3) (50 * Real.sqrt 3) = 250 := 
sorry

end pyramid_volume_l821_821604


namespace last_score_is_87_l821_821587

-- Definitions based on conditions:
def scores : List ℕ := [73, 78, 82, 84, 87, 95]
def total_sum := 499
def final_median := 83

-- Prove that the last score is 87 under given conditions.
theorem last_score_is_87 (h1 : total_sum = 499)
                        (h2 : ∀ n ∈ scores, (499 - n) % 6 ≠ 0)
                        (h3 : final_median = 83) :
  87 ∈ scores := sorry

end last_score_is_87_l821_821587


namespace first_shadow_length_twice_table_height_l821_821322

theorem first_shadow_length_twice_table_height
  (h l1 l2 : ℝ) (α β : ℝ)
  (h1 : l1 = h * tan α) 
  (h2 : l2 = h * tan β) 
  (h3 : tan (α - β) = 1/3) 
  (h4 : tan β = 1) : 
  l1 = 2 * h :=
by
  sorry

end first_shadow_length_twice_table_height_l821_821322


namespace new_volume_of_pyramid_l821_821305

-- Definitions based on conditions
def initial_volume : ℝ := 72
def base_scale_factor : ℝ := 6  -- Tripling the base and doubling the height: 3 * 2 = 6
def height_scale_factor : ℝ := 1.4  -- Height increased by 40%

-- Problem statement
theorem new_volume_of_pyramid : 
  let new_volume := initial_volume * base_scale_factor * height_scale_factor in 
  new_volume = 604.8 :=
by 
  let new_volume := initial_volume * base_scale_factor * height_scale_factor
  have h := new_volume
  exact eq.refl h

end new_volume_of_pyramid_l821_821305


namespace factor_difference_of_squares_l821_821789

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821789


namespace area_le_0_34_l821_821596

theorem area_le_0_34 (S : set (ℝ × ℝ)) (h_condition : ∀ (p q ∈ S), p ≠ q → dist p q ≠ 0.001) :
  (measure_theory.measure_space.measure (measure_theory.outer_measure.caratheodory S) S ≤ 0.34) :=
sorry

end area_le_0_34_l821_821596


namespace number_of_rectangles_l821_821067

-- Defining the grid size
def columns : ℕ := 5
def rows : ℕ := 4

-- Stating the theorem to prove the number of rectangles
theorem number_of_rectangles : 
  ∑ x in finset.range (columns + 1), ∑ y in finset.range (rows + 1), (columns - x) * (rows - y) = 24 := 
by
  sorry

end number_of_rectangles_l821_821067


namespace Phillip_correct_total_l821_821166

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l821_821166


namespace validate_tax_authority_notification_l821_821584

/-
Federal Law 173-FZ states:
1. Currency transactions between residents of the Russian Federation are prohibited, unless exceptions apply.
2. Settlements within Russia must be made in Russian rubles.
-/

def law_173_fz (a b : Type) [resident a] [resident b] (amount : ℝ) 
  (currency : Type) [euro_currency currency] : Prop :=
  ∀ (transaction_in_rubles : Type → Type → ℝ → Prop),
  (transaction_in_rubles a b amount) → (amount ∉ euro_currency)

def transaction_agreement_violation : Prop :=
  let Mikhail := resident.mk JSC_New_Years_Joy
  let Valentin := resident.mk LLC_Holiday_Comes_to_You
  let transaction_amount := 2700
  let currency := euro_currency.mk 
  law_173_fz Mikhail Valentin transaction_amount currency

theorem validate_tax_authority_notification :
  transaction_agreement_violation → tax_authority_correct :=
sorry

end validate_tax_authority_notification_l821_821584


namespace smallest_repeating_block_of_fraction_l821_821894

theorem smallest_repeating_block_of_fraction (a b : ℕ) (h : a = 8 ∧ b = 11) :
  ∃ n : ℕ, n = 2 ∧ decimal_expansion_repeating_block_length (a / b) = n := by
  sorry

end smallest_repeating_block_of_fraction_l821_821894


namespace kolya_or_leva_wins_l821_821552

variable (k l : ℝ) -- Defining k and l as real numbers

-- Define a function to check the winner based on the lengths k and l
def determine_winner (k l : ℝ) : String :=
  if k > l then "Kolya" else "Leva"

-- The theorem to prove our solution statement
theorem kolya_or_leva_wins (k l : ℝ) (hk : 0 < k) (hl : 0 < l) :
  (determine_winner k l = "Kolya" ↔ k > l) ∧ (determine_winner k l = "Leva" ↔ k ≤ l) :=
by
  split
  · split
    · intro h
      simp [determine_winner] at h
      exact h
    · intro h
      simp [determine_winner]
      exact h
  · split
    · intro h
      simp [determine_winner] at h
      exact h
    · intro h
      simp [determine_winner]
      exact h

end kolya_or_leva_wins_l821_821552


namespace min_value_of_z_l821_821777

theorem min_value_of_z : 
  let z := λ x: ℝ, 5 * x^2 + 10 * x + 20
  in ∃ x: ℝ, ∀ y: ℝ, z x ≤ z y :=
begin
  let z := λ x: ℝ, 5 * x^2 + 10 * x + 20,
  use (-1),
  intro y,
  calc
    z (-1) = 5 * (-1)^2 + 10 * (-1) + 20 : by simp [z]
        ... = 15                       : by linarith
    ... ≤ 5 * y^2 + 10 * y + 20         : by {
      have h1: (y + 1)^2 ≥ 0, from pow_two_nonneg _,
      linarith,
    }
end

end min_value_of_z_l821_821777


namespace neighbors_have_even_total_bells_not_always_divisible_by_3_l821_821234

def num_bushes : ℕ := 19

def is_neighbor (circ : ℕ → ℕ) (i j : ℕ) : Prop := 
  if i = num_bushes - 1 then j = 0
  else j = i + 1

-- Part (a)
theorem neighbors_have_even_total_bells (bells : Fin num_bushes → ℕ) :
  ∃ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 2 = 0 := sorry

-- Part (b)
theorem not_always_divisible_by_3 (bells : Fin num_bushes → ℕ) :
  ¬ (∀ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 3 = 0) := sorry

end neighbors_have_even_total_bells_not_always_divisible_by_3_l821_821234


namespace integral_f_l821_821870

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ Set.Icc (-3 : ℝ) 0 then - (1 / 3) * x ^ 2 + 3
else if x ∈ Set.Ioc 0 3 then Real.sqrt (9 - x ^ 2)
else 0

theorem integral_f : ∫ x in -3..3, f x = 6 + 9 * Real.pi / 4 :=
by
  sorry

end integral_f_l821_821870


namespace original_number_of_girls_l821_821307

theorem original_number_of_girls (b g : ℕ) (h1 : b = g)
                                (h2 : 3 * (g - 25) = b)
                                (h3 : 6 * (b - 60) = g - 25) :
  g = 67 :=
by sorry

end original_number_of_girls_l821_821307


namespace range_of_k_l821_821654

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → y^2 = 2 * x → (∃! (x₀ y₀ : ℝ), y₀ = k * x₀ + 1 ∧ y₀^2 = 2 * x₀)) ↔ 
  (k = 0 ∨ k ≥ 1/2) :=
sorry

end range_of_k_l821_821654


namespace max_section_area_l821_821270

theorem max_section_area (a b c : ℝ) (hab : a ≤ b) (hbc : b ≤ c) :
  ∃ (area : ℝ), area = c * real.sqrt(a^2 + b^2) ∧ ∀ (S : ℝ),
  (S = a * real.sqrt(b^2 + c^2) ∨ S = b * real.sqrt(a^2 + c^2) ∨ S = c * real.sqrt(a^2 + b^2)) → S ≤ area :=
begin
  sorry
end

end max_section_area_l821_821270


namespace available_seats_now_l821_821524

def total_seats : ℕ := 800
def initially_taken_seats : ℕ := (2 * total_seats) / 5
def broken_seats : ℕ := total_seats / 10
def initial_occupied_percentage : ℚ := 0.4
def increased_percentage : ℚ := initial_occupied_percentage + 0.12
def newly_occupied_seats : ℕ := (increased_percentage * total_seats).toNat

theorem available_seats_now :
  total_seats - newly_occupied_seats - broken_seats = 304 :=
by
  sorry

end available_seats_now_l821_821524


namespace right_triangle_circumcircle_l821_821387

open EuclideanGeometry

theorem right_triangle_circumcircle
  (A B C K L : Point)
  (hABC : right_triangle A B C)
  (hK : angle_bisector B K A)
  (hL : circumcircle (triangle A K B) ∩ segment B C):

  (distance C B + distance C L = distance A B) :=
sorry

end right_triangle_circumcircle_l821_821387


namespace intersection_points_count_length_AB_is_2_l821_821769

-- Define the parametric equations of line l
def line_parametric_eq (t : ℝ) : ℝ × ℝ := (1/2 * t, 1 - (√3 / 2) * t)

-- Define the polar coordinate equation of circle C
def circle_polar_eq (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Define the standard form of line l obtained by eliminating parameter t
def line_standard_eq (x y : ℝ) : Prop := √3 * x + y = 1

-- Define the rectangular coordinate equation of circle C by converting from polar
def circle_rect_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0

-- Prove that the number of intersection points between line l and circle C is 2
theorem intersection_points_count : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_standard_eq p1.1 p1.2 ∧ circle_rect_eq p1.1 p1.2) ∧
  (line_standard_eq p2.1 p2.2 ∧ circle_rect_eq p2.1 p2.2) ∧
  p1 ≠ p2 := 
sorry

-- Prove that if circle C intersects line l at points A and B, the length of line segment AB is 2
theorem length_AB_is_2 (A B : ℝ × ℝ) :
  line_standard_eq A.1 A.2 → circle_rect_eq A.1 A.2 → 
  line_standard_eq B.1 B.2 → circle_rect_eq B.1 B.2 →
  A ≠ B → 
  Real.dist A B = 2 := 
sorry

end intersection_points_count_length_AB_is_2_l821_821769


namespace systematic_sampling_C_count_l821_821668

theorem systematic_sampling_C_count :
  (∃ (total_people selected_people group_size number_drawn groups_a groups_b : ℕ),
   total_people = 960 ∧ 
   selected_people = 32 ∧ 
   group_size * selected_people = total_people ∧ 
   number_drawn = 9 ∧ 
   selected_people = (groups_a + groups_b + 32 - (groups_a + groups_b + 1)) / group_size ∧ 
   32 - groups_a - groups_b = 7) ↔ true := 
begin
  sorry
end

end systematic_sampling_C_count_l821_821668


namespace right_triangle_MB_CB_CM_solution_l821_821973

theorem right_triangle_MB_CB_CM_solution (x h k : ℝ) 
  (h1 : x + real.sqrt ((x + h)^2 + k^2) = h + k) : 
  x = h * k / (2 * h + k) :=
by
  sorry

end right_triangle_MB_CB_CM_solution_l821_821973


namespace complex_number_equality_l821_821564

theorem complex_number_equality (a b : ℝ) (h : (1 + 2 * complex.I) / (a + b * complex.I) = 1 + complex.I) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
by
  sorry

end complex_number_equality_l821_821564


namespace find_lambda_l821_821434

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821434


namespace complement_of_A_l821_821885

theorem complement_of_A (U A : Set ℕ) (hU : U = {1, 2, 3}) (hA : A = {1, 3}) :
  compl A U = {2} :=
by
  sorry

end complement_of_A_l821_821885


namespace part1_solution_set_a_eq_1_part2_range_of_values_a_l821_821873

def f (x a : ℝ) : ℝ := |(2 * x - a)| + |(x - 3 * a)|

theorem part1_solution_set_a_eq_1 :
  ∀ x : ℝ, f x 1 ≤ 4 ↔ 0 ≤ x ∧ x ≤ 2 :=
by sorry

theorem part2_range_of_values_a :
  ∀ a : ℝ, (∀ x : ℝ, f x a ≥ |(x - a / 2)| + a^2 + 1) ↔
    ((-2 : ℝ) ≤ a ∧ a ≤ -1 / 2) ∨ (1 / 2 ≤ a ∧ a ≤ 2) :=
by sorry

end part1_solution_set_a_eq_1_part2_range_of_values_a_l821_821873


namespace pyramid_volume_l821_821021

variables {α : Type*} [field α] {A B C : α → α → α} {S R : α}

-- Define the necessary conditions for the given problem
def triangle_area (A B C : α → α → α) : α := S
def circumradius (A B C : α → α → α) : α := R
def altitudes_length (A1 B1 C1 A B C : α → α → α) : α := sorry  -- Placeholder for altitudes computation

-- The expected volume of the pyramid
def expected_volume (S R : α) : α := (4 / 3) * S * R

-- The theorem to prove
theorem pyramid_volume {A A1 B B1 C C1 : α → α → α} 
  (h_triangle : triangle_area A B C = S) 
  (h_circumradius : circumradius A B C = R) 
  (h_altitudes : altitudes_length A1 B1 C1 A B C = sorry) : 
  volume_of_pyramid A1 B1 C1 A B C = expected_volume S R := sorry

end pyramid_volume_l821_821021


namespace arithmetic_sequence_sum_17_l821_821092

-- Definition of arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n m : ℕ, m > n → a m - a n = d * (m - n)

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- arbitrary definition, not needed for statement

theorem arithmetic_sequence_sum_17 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a9 : a 9 = 3) : 
  let S_17 := (17 * (a 1 + a 17)) / 2 in S_17 = 51 :=
by
  sorry

end arithmetic_sequence_sum_17_l821_821092


namespace solve_trigonometric_eqn_l821_821689

theorem solve_trigonometric_eqn (x : ℝ) : 
  (∃ k : ℤ, x = 3 * (π / 4 * (4 * k + 1))) ∨ (∃ n : ℤ, x = π * (3 * n + 1) ∨ x = π * (3 * n - 1)) :=
by 
  sorry

end solve_trigonometric_eqn_l821_821689


namespace abs_sqrt_identity_l821_821073

noncomputable def abs_val (x : ℝ) : ℝ :=
  if x < 0 then -x else x

noncomputable def sqrt_val (x : ℝ) : ℝ :=
  real.sqrt x

theorem abs_sqrt_identity (a : ℝ) (h : a < 0) : abs_val (a - 3) - sqrt_val (a^2) = 3 :=
by
  sorry

end abs_sqrt_identity_l821_821073


namespace dividend_is_10_l821_821518

theorem dividend_is_10
  (q d r : ℕ)
  (hq : q = 3)
  (hd : d = 3)
  (hr : d = 3 * r) :
  (q * d + r = 10) :=
by
  sorry

end dividend_is_10_l821_821518


namespace parallel_vectors_lambda_l821_821475

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821475


namespace lengths_of_sides_l821_821083

theorem lengths_of_sides (A B C : ℝ) (a b c : ℝ) (h : c = 10) (h1 : (cos A / cos B) = (b / a)) (h2 : (b / a) = (4 / 3)) :
  a = 6 ∧ b = 8 :=
by
  sorry

end lengths_of_sides_l821_821083


namespace sum_squares_and_products_of_nonneg_reals_l821_821921

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l821_821921


namespace parallel_vectors_lambda_l821_821442

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821442


namespace max_distance_MN_l821_821054

noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

def point_M : ℝ × ℝ := (2, 0)

theorem max_distance_MN :
  ∃ N : ℝ × ℝ, circle_eq N.1 N.2 ∧ 
  ∀ P : ℝ × ℝ, circle_eq P.1 P.2 → dist P point_M ≤ dist N point_M :=
∃ (N : ℝ × ℝ), circle_eq N.1 N.2 ∧ 
  ∀ P, circle_eq P.1 P.2 → dist P point_M ≤ dist N point_M :=
sorry

end max_distance_MN_l821_821054


namespace factorization_correct_l821_821804

noncomputable def P (x : ℤ) : ℤ := 5 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 4 * x * x

theorem factorization_correct (x : ℤ) : 
    P(x) = (5 * x * x + 94 * x + 385) * (x + 3) * (x + 14) := 
by 
    sorry

end factorization_correct_l821_821804


namespace plane_equation_and_gcd_l821_821207

variable (x y z : ℝ)

theorem plane_equation_and_gcd (A B C D : ℤ) (h1 : A = 8) (h2 : B = -6) (h3 : C = 5) (h4 : D = -125) :
    (A * x + B * y + C * z + D = 0 ↔ x = 8 ∧ y = -6 ∧ z = 5) ∧
    Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by sorry

end plane_equation_and_gcd_l821_821207


namespace increased_cost_is_4_percent_l821_821522

-- Initial declarations
variables (initial_cost : ℕ) (price_change_eggs price_change_apples percentage_increase : ℕ)

-- Cost definitions based on initial conditions
def initial_cost_eggs := 100
def initial_cost_apples := 100

-- Price adjustments
def new_cost_eggs := initial_cost_eggs - (initial_cost_eggs * 2 / 100)
def new_cost_apples := initial_cost_apples + (initial_cost_apples * 10 / 100)

-- New combined cost
def new_combined_cost := new_cost_eggs + new_cost_apples

-- Old combined cost
def old_combined_cost := initial_cost_eggs + initial_cost_apples

-- Increase in cost
def increase_in_cost := new_combined_cost - old_combined_cost

-- Percentage increase
def calculated_percentage_increase := (increase_in_cost * 100) / old_combined_cost

-- The proof statement
theorem increased_cost_is_4_percent :
  initial_cost = 100 →
  price_change_eggs = 2 →
  price_change_apples = 10 →
  percentage_increase = 4 →
  calculated_percentage_increase = percentage_increase :=
sorry

end increased_cost_is_4_percent_l821_821522


namespace max_profit_300_l821_821717

noncomputable def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_revenue (x : ℝ) : ℝ :=
if x ≤ 400 then (400 * x - (1 / 2) * x^2)
else 80000

noncomputable def total_profit (x : ℝ) : ℝ :=
total_revenue x - total_cost x

theorem max_profit_300 :
    ∃ x : ℝ, (total_profit x = (total_revenue 300 - total_cost 300)) := sorry

end max_profit_300_l821_821717


namespace parallel_vectors_implies_value_of_λ_l821_821425

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821425


namespace positive_difference_after_10_years_l821_821610

def compound_interest (P r: ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r: ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem positive_difference_after_10_years : 
  let P := 14000
  let r_e := 0.06
  let r_o := 0.08
  let n := 10 
  let A_e := compound_interest P r_e n
  let A_o := simple_interest P r_o n 
  in Em $| A_o - A_e ≈ 128
:=
by sorry

end positive_difference_after_10_years_l821_821610


namespace range_of_a_l821_821368

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Ioc 0 (π / 2), cos x ^ 2 + sin x + a = 0) ↔ a ∈ Icc (-5 / 4) (-1) := 
by
  sorry

end range_of_a_l821_821368


namespace probability_three_digit_multiple_of_3_l821_821285

theorem probability_three_digit_multiple_of_3 :
  let digits := {1, 3, 5, 7, 9}
  let total_ways := 5 * 4 * 3
  let valid_ways := 6
  (valid_ways : ℕ) / (total_ways : ℕ) = 1 / 10 :=
by
  sorry

end probability_three_digit_multiple_of_3_l821_821285


namespace determine_one_l821_821639

-- Conditions: The operations the MK-97 can perform
def add (a b : ℝ) : ℝ := a + b
def eq (a b : ℝ) : Prop := a = b
def roots (a b c : ℝ) : Option (ℝ × ℝ) :=
  let Δ := b * b - 4 * a * c
  if h : Δ < 0 then none
  else some ((-b + real.sqrt Δ) / (2 * a), (-b - real.sqrt Δ) / (2 * a))

-- Definition: combining the operations to check if x = 1
theorem determine_one (x : ℝ) (h1 : eq x (add x x)) : x = 0 ∨ x = 1 :=
by
  -- If x = 2x, then x must be 0
  by_cases h2 : x = 0
  { left, exact h2 }
  { 
    -- If x ≠ 0, then form the quadratic equation y^2 + 2xy + x = 0
    have h3 : roots 1 (2 * x) x = some (x, x) ∨ x = 1,
    { 
      -- Calculate discriminant Δ = 4 * x * (x - 1)
      let Δ := 4 * (x * x - x)
      have hΔ : Δ = 0 := sorry, -- The discriminant calculation Δ = 4x(x - 1)
      cases hΔ,
        right, -- Since Δ = 0 implies x(x - 1) = 0, we already know x ≠ 0, so x must be 1
        show x = 1, from eq.trans (eq.refl 1) rfl 
    }
    exact h3.elim id id
  }

end determine_one_l821_821639


namespace cosine_of_vertex_angle_l821_821036

theorem cosine_of_vertex_angle (a : ℝ) (h : real.cos a = 1 / 3) : real.cos (π - 2 * a) = 7 / 9 :=
sorry

end cosine_of_vertex_angle_l821_821036


namespace math_problem_proof_l821_821232

-- Define M as the product of (2^i + 1) for i in powers of 2
def M : ℕ := (2^1 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) * (2^64 + 1)

-- Define N as M + 1
def N : ℕ := M + 1

-- Prove that the given expression evaluates to 128
theorem math_problem_proof : 10 * 72 * (Real.log2 N) = 128 := 
by 
  -- Proof steps to be filled
  sorry

end math_problem_proof_l821_821232


namespace part_i_part_ii_l821_821048

-- Define function f based on the given conditions
def f (x t : ℝ) : ℝ := (x + 1) * (x - t) / (x^2)

-- Given 1: Prove that t = 1 if f is an even function
theorem part_i (t : ℝ) (h_even : ∀ x : ℝ, f (-x) t = f x t) : t = 1 :=
sorry

-- Given 2: Prove that there do not exist real numbers b > a > 0 such that
-- the range of f(x) for x ∈ [a, b] is [2 - 2/a, 2 - 2/b]
theorem part_ii : ¬ ∃ a b : ℝ, a > 0 ∧ b > a ∧
  (∀ x ∈ set.Icc a b, f x 1 = 2 - 2 / a ∧ f x 1 = 2 - 2 / b) :=
sorry

end part_i_part_ii_l821_821048


namespace smallest_repeating_block_digits_l821_821896

theorem smallest_repeating_block_digits (n : ℕ) (d : ℕ) (hd_pos : d > 0) (hd_coprime : Nat.gcd n d = 1)
  (h_fraction : (n : ℚ) / d = 8 / 11) : n = 2 :=
by
  -- proof will go here
  sorry

end smallest_repeating_block_digits_l821_821896


namespace parallel_AB_FG_set_of_midpoints_S_l821_821162

variables {k : Type*} [metric_space k] [normed_group k] [inner_product_space ℝ k]
variables (A B C D E F G : k)
variables (M O S : k)

-- Conditions
def points_on_circle (A B : k) (k : set k) : Prop := 
  circle_contains A B k ∧ ¬ (diameter A B k)

def triangle_acute (A B C : k) : Prop :=
  angle_acute (angle A B C) ∧ angle_acute (angle B C A) ∧ angle_acute (angle C A B)

def is_feet_of_altitude (D E : k) (A B C : k) : Prop :=
  altitude_foot A D B C ∧ altitude_foot B E A C

def projections (D E F G : k) (A B C : k) : Prop :=
  projection D F A C ∧ projection E G B C

-- The problem statement
theorem parallel_AB_FG 
  (h1 : points_on_circle A B (set_of_points k))
  (h2 : ∃ C, move_along_arc C (arc AB) ∧ triangle_acute A B C)
  (h3 : is_feet_of_altitude D E A B C)
  (h4 : projections D E F G A B C):
  parallel AB FG := 
sorry

theorem set_of_midpoints_S 
  (h1 : points_on_circle A B (set_of_points k))
  (h2 : is_feet_of_altitude D E A B C)
  (h3 : projections D E F G A B C):
  ∃ O, ∃ r, (∀ S, midpoint_of FG S → circle_center_radius O r S) :=
sorry

end parallel_AB_FG_set_of_midpoints_S_l821_821162


namespace factor_x_squared_minus_64_l821_821793

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821793


namespace jo_blair_sequence_30th_term_l821_821983

-- Definition of the arithmetic sequence
def arithmetic_sequence (a d : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := arithmetic_sequence a d n + d

-- The main theorem statement
theorem jo_blair_sequence_30th_term : 
  arithmetic_sequence 3 2 29 = 61 := 
sorry

end jo_blair_sequence_30th_term_l821_821983


namespace perfect_squares_in_range_l821_821900

theorem perfect_squares_in_range :
  let perfect_squares := { n : ℕ | (10 < n) ∧ (n < 20) ∧ ∃ m, n = m^2 }
  let n := perfect_squares.card
  let s := perfect_squares.sum id
  n = 9 ∧ s = 2185 :=
by
  sorry

end perfect_squares_in_range_l821_821900


namespace parallel_vectors_lambda_l821_821486

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821486


namespace integral_sqrt_is_two_thirds_l821_821626

-- Define the condition about the binomial expansion coefficient
def binomial_condition (a : ℝ) := 
  (Nat.choose 6 1) * (a^5) * (sqrt 3 / 6) = sqrt 3

-- Define the integral to be evaluated
def integral_sqrt (a : ℝ) := ∫ x in 0..a, sqrt x

-- The theorem statement
theorem integral_sqrt_is_two_thirds (a : ℝ) (h : binomial_condition a) : integral_sqrt a = 2 / 3 := 
sorry

end integral_sqrt_is_two_thirds_l821_821626


namespace roots_sum_of_products_l821_821143

theorem roots_sum_of_products (a b c : ℝ) (h : Polynomial.root_set (Polynomial.C 5 * Polynomial.X^3 +
    Polynomial.C (-4) * Polynomial.X^2 + Polynomial.C 15 * Polynomial.X + Polynomial.C (-12)) {a, b, c}) :
    a * b + a * c + b * c = -3 :=
by
  sorry

end roots_sum_of_products_l821_821143


namespace value_of_a7_l821_821112

theorem value_of_a7 (a : ℕ → ℤ) (h1 : a 1 = 0) (h2 : ∀ n, a (n + 2) - a n = 2) : a 7 = 6 :=
by {
  sorry -- Proof goes here
}

end value_of_a7_l821_821112


namespace inverse_of_f_l821_821572

-- Define the invertibility of the functions
variables {X Y Z W : Type} [invertible a : X → Y] [invertible b : Y → Z] [invertible c : Z → W]

-- Define the function f
def f (x : X) : W := b (c (a x))

-- Proof statement that the inverse of f is a∘−1 ∘ c∘−1 ∘ b∘−1
theorem inverse_of_f : function.inverse f = (a⁻¹ ∘ c⁻¹ ∘ b⁻¹) :=
sorry

end inverse_of_f_l821_821572


namespace julian_initial_owing_l821_821549

theorem julian_initial_owing (jenny_owing_initial: ℕ) (borrow: ℕ) (total_owing: ℕ):
    borrow = 8 → total_owing = 28 → jenny_owing_initial + borrow = total_owing → jenny_owing_initial = 20 :=
by intros;
   exact sorry

end julian_initial_owing_l821_821549


namespace max_value_frac_expr_l821_821674

theorem max_value_frac_expr (t : ℝ) :
  ∃ t : ℝ, ∀ t', (t' : ℝ), (t' ≠ t) →
    (frac_expr t' : ℝ) ≤ max_value t :=
sorry

def frac_expr (t : ℝ) : ℝ := (3^t - 4*t) * t / 9^t

noncomputable def max_value : ℝ := 1/16

end max_value_frac_expr_l821_821674


namespace lines_parallel_to_plane_ABP_l821_821030

noncomputable def cube : Type := sorry -- Placeholder for the actual cube definition
structure Point := {x : ℝ, y : ℝ, z : ℝ}

def edge_DD1 : set Point := {P : Point | sorry} -- Define the edge DD_1 (placeholder)

def line_DC : set Point := sorry -- Define the line DC (placeholder)
def line_D1C1 : set Point := sorry -- Define the line D1C1 (placeholder)
def line_A1B1 : set Point := sorry -- Define the line A1B1 (placeholder)

def plane_ABP (A B P : Point) : set Point := sorry -- Define the plane ABP given points A, B, and P (placeholder)

theorem lines_parallel_to_plane_ABP (A B D D1 P : Point) (hP : P ∈ edge_DD1) :
  (∀ x ∈ line_DC, x ∈ plane_ABP A B P) ∧
  (∀ y ∈ line_D1C1, y ∈ plane_ABP A B P) ∧
  (∀ z ∈ line_A1B1, z ∈ plane_ABP A B P) :=
begin
  sorry
end

end lines_parallel_to_plane_ABP_l821_821030


namespace dist_lt_half_in_equilateral_triangle_l821_821541

noncomputable def equilateral_triangle (a : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (0 ≤ x ∧ x ≤ a) ∧ (0 ≤ y ∧ y ≤ (sqrt 3 / 2) * a) ∧ (y ≤ sqrt 3 * (a - x))}

theorem dist_lt_half_in_equilateral_triangle (a : ℝ) (a_pos : 0 < a) :
  ∀ (points : fin 5 → ℝ × ℝ),
  (∀ i, points i ∈ equilateral_triangle a) →
  ∃ (i j : fin 5), i ≠ j ∧ dist (points i) (points j) < a / 2 :=
begin
  sorry
end

end dist_lt_half_in_equilateral_triangle_l821_821541


namespace point_tangent_l821_821409

theorem point_tangent (y : ℝ) (α : ℝ) (h1 : tan α = 1 / 2) (h2 : ∃ α : ℝ, ∃ y : ℝ, (-1, y) ∈ {P : ℝ × ℝ | tan α = 1 / 2}) : y = -1 / 2 :=
by 
  -- Some intermediate steps are skipped for this example
  sorry

end point_tangent_l821_821409


namespace min_value_PF_PA_l821_821878

noncomputable def hyperbola_eq (x y : ℝ) := (x^2 / 4) - (y^2 / 12) = 1

noncomputable def focus_left : ℝ × ℝ := (-4, 0)
noncomputable def focus_right : ℝ × ℝ := (4, 0)
noncomputable def point_A : ℝ × ℝ := (1, 4)

theorem min_value_PF_PA (P : ℝ × ℝ)
  (hP : hyperbola_eq P.1 P.2)
  (hP_right_branch : P.1 > 0) :
  ∃ P : ℝ × ℝ, ∀ X : ℝ × ℝ, hyperbola_eq X.1 X.2 → X.1 > 0 → 
               (dist X focus_left + dist X point_A) ≥ 9 ∧
               (dist P focus_left + dist P point_A) = 9 := 
sorry

end min_value_PF_PA_l821_821878


namespace Mika_stickers_l821_821582

theorem Mika_stickers
  (initial_stickers : ℕ)
  (bought_stickers : ℕ)
  (received_stickers : ℕ)
  (given_stickers : ℕ)
  (used_stickers : ℕ)
  (final_stickers : ℕ) :
  initial_stickers = 45 →
  bought_stickers = 53 →
  received_stickers = 35 →
  given_stickers = 19 →
  used_stickers = 86 →
  final_stickers = initial_stickers + bought_stickers + received_stickers - given_stickers - used_stickers →
  final_stickers = 28 :=
by
  intros
  sorry

end Mika_stickers_l821_821582


namespace approximate_width_of_ditch_l821_821781

/-- Given the configuration in the problem, the width of the ditch is approximately 2.70 meters --/
theorem approximate_width_of_ditch : ∃ (x : ℝ), x ≈ 2.70 ∧ 
(∃ (h : ℝ), h = 1 ∧ 
    (1 / ∥(sqrt (25 - x^2))∥ + 1 / ∥(sqrt (9 - x^2))∥ = 1/h)) :=
sorry

end approximate_width_of_ditch_l821_821781


namespace parallel_vectors_lambda_l821_821470

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821470


namespace magnitude_combination_l821_821407

variables (m n : ℝ^2)
variables (m_unit : ‖m‖ = 1) (n_unit : ‖n‖ = 1)
variables (orthogonal : inner m n = 0)

theorem magnitude_combination : ‖3 • m + 4 • n‖ = 5 :=
by
  sorry

end magnitude_combination_l821_821407


namespace transformed_center_coordinates_l821_821766

-- Define the original center of the circle
def center_initial : ℝ × ℝ := (3, -4)

-- Define the function for reflection across the x-axis
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the function for translation by a certain number of units up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Define the problem statement
theorem transformed_center_coordinates :
  translate_up (reflect_x_axis center_initial) 5 = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l821_821766


namespace largest_inner_circle_radius_l821_821309

theorem largest_inner_circle_radius 
  (side_length : ℝ) 
  (quarter_circle_radius : ℝ)
  (h1 : side_length = 4)
  (h2 : quarter_circle_radius = side_length / 2):
  ∃ r : ℝ, r = sqrt 2 ∧ 
            ∀ (x y : ℝ), 
              (0 ≤ x ∧ x ≤ side_length) ∧ 
              (0 ≤ y ∧ y ≤ side_length) → 
              (x = quarter_circle_radius ∨ x = side_length - quarter_circle_radius) →
              (y = quarter_circle_radius ∨ y = side_length - quarter_circle_radius) →
              dist (x, y) (side_length / 2, side_length / 2) = r * sqrt 2 := 
begin
  sorry
end

end largest_inner_circle_radius_l821_821309


namespace find_angle_ACB_l821_821136

variable (A B C D E F : Type) [HilbertSpace A] [InnerProductSpace ℝ B]
variable (AD BE CF : B)
variable (α : ℝ)

noncomputable def angle_ACB : ℝ := 
let α := inner_product_space.angle (AD - A) (CF - C) in
α

theorem find_angle_ACB
  (h₁ : 9 • AD + 4 • BE + 7 • CF = (0 : B))
  (h₂ : InnerProductSpace.angle AD CF = α)
  : α = 60 :=
sorry

end find_angle_ACB_l821_821136


namespace university_minimum_spend_l821_821257

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def box_cost : ℝ := 1.20
def total_volume : ℝ := 3.06 * (10^6)

def box_volume : ℕ := box_length * box_width * box_height

noncomputable def number_of_boxes : ℕ := Nat.ceil (total_volume / box_volume)
noncomputable def total_cost : ℝ := number_of_boxes * box_cost

theorem university_minimum_spend : total_cost = 612 := by
  sorry

end university_minimum_spend_l821_821257


namespace inequality_proof_l821_821849

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l821_821849


namespace thabo_total_books_l821_821612

/-- Thabo owns 55 hardcover nonfiction books. -/
constant H : ℕ
axiom H_eq : H = 55

/-- Thabo owns 20 more paperback nonfiction books than hardcover nonfiction books. -/
constant PNF : ℕ
axiom PNF_eq : PNF = H + 20

/-- Thabo owns twice as many paperback fiction books as paperback nonfiction books. -/
constant PF : ℕ
axiom PF_eq : PF = 2 * PNF

/-- Prove that the total number of books Thabo owns is 280. -/
theorem thabo_total_books : H + PNF + PF = 280 :=
by 
  sorry

end thabo_total_books_l821_821612


namespace min_sum_of_factors_240_l821_821677

theorem min_sum_of_factors_240 :
  ∃ a b : ℕ, a * b = 240 ∧ (∀ a' b' : ℕ, a' * b' = 240 → a + b ≤ a' + b') ∧ a + b = 31 :=
sorry

end min_sum_of_factors_240_l821_821677


namespace complement_in_M_l821_821149

open Set

variable {α : Type*}

theorem complement_in_M (M N : Set α) [Fintype α] :
  M = {0, 1, 2, 3, 4} → N = {0, 1, 3} → M \ N = {2, 4} :=
by
  intro hM hN
  rw [hM, hN]
  rw [diff_eq, singleton_eq_singleton_iff, mem_compl, mem_singleton_iff, not_or_distrib, not_and_distrib,
      Set.eq_univ_iff_forall, nonempty_iff_ne_empty, Set.sep_eq_sep_iff_eq, singleton_eq_singleton_iff_two, Set.ext_iff]
  sorry

end complement_in_M_l821_821149


namespace vasya_can_win_l821_821669

noncomputable def initial_first : ℝ := 1 / 2009
noncomputable def initial_second : ℝ := 1 / 2008
noncomputable def increment : ℝ := 1 / (2008 * 2009)

theorem vasya_can_win :
  ∃ n : ℕ, ((2009 * n) * increment = 1) ∨ ((2008 * n) * increment = 1) :=
sorry

end vasya_can_win_l821_821669


namespace abs_neg_ten_l821_821333

theorem abs_neg_ten : abs (-10) = 10 := 
by {
  sorry
}

end abs_neg_ten_l821_821333


namespace xy_product_l821_821861

variable {x y : ℝ}

theorem xy_product (h1 : x ≠ y) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x + 3/x = y + 3/y) : x * y = 3 :=
sorry

end xy_product_l821_821861


namespace seating_arrangements_l821_821956

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821956


namespace tan_x_eq_sqrt3_f_monotonic_increasing_l821_821830

-- Definition of vectors m and n
def m (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi / 6), 1)
def n (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)

-- Problem 1: Prove that tan(x) = sqrt(3) given m || n
theorem tan_x_eq_sqrt3 (x : ℝ) (h : m x = n x) : Real.tan x = Real.sqrt 3 :=
by 
  sorry

-- Problem 2: Prove that f(x) = m · n is monotonically increasing on the interval 
-- [ -π/6 + kπ, π/3 + kπ ] for k ∈ ℤ.
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_monotonic_increasing (k : ℤ) (x : ℝ) (h : -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi) : 
  ∃ (c : ℝ), f x = c ∧ c > 0 :=
by 
  sorry

end tan_x_eq_sqrt3_f_monotonic_increasing_l821_821830


namespace gain_percentage_l821_821185

theorem gain_percentage (MP CP : ℝ) (h1 : 0.90 * MP = 1.17 * CP) :
  (((MP - CP) / CP) * 100) = 30 := 
by
  sorry

end gain_percentage_l821_821185


namespace length_relation_l821_821277

namespace Geometry

open Real EuclideanGeometry

variables {A B C P Q : Point} -- Variables representing points

-- Definitions of conditions
def midpoint (A B C : Point) : Prop := dist A C = dist B C
def on_circle_diameter (P A B : Point) : Prop := dist P (midpoint A B) = dist A B / 2
def perpendicular (PQ AB : Line) : Prop := angle PQ AB = π / 2

-- Given conditions as assumptions
axiom midpoint_condition : midpoint A B C
axiom P_on_circle_AB : on_circle_diameter P A B
axiom Q_on_circle_AC : on_circle_diameter Q A C
axiom PQ_perpendicular_AB : perpendicular (line_through P Q) (line_through A B)

-- The statement we want to prove
theorem length_relation (A B C P Q : Point)
  (h1 : midpoint A B C)
  (h2 : on_circle_diameter P A B)
  (h3 : on_circle_diameter Q A C)
  (h4 : perpendicular (line_through P Q) (line_through A B)) :
  dist A P = √2 * dist A Q := sorry

end Geometry

end length_relation_l821_821277


namespace smallest_expression_value_l821_821814

theorem smallest_expression_value (a b c : ℝ) (h₁ : b > c) (h₂ : c > 0) (h₃ : a ≠ 0) :
  (2 * a + b) ^ 2 + (b - c) ^ 2 + (c - 2 * a) ^ 2 ≥ (4 / 3) * b ^ 2 :=
by
  sorry

end smallest_expression_value_l821_821814


namespace simplified_f_f_value_l821_821860

noncomputable def f (x : ℝ) (n : ℤ) : ℝ :=
  (cos (n * π + x))^2 * (sin (n * π - x))^2 / (cos ((2 * n + 1) * π - x))^2

theorem simplified_f (x : ℝ) (n : ℤ) : f x n = sin x ^ 2 := by
  sorry

theorem f_value : 
  f (π / 2016) (0 : ℤ) + f (1007 * π / 2016) (0 : ℤ) = 1 := by
  sorry

end simplified_f_f_value_l821_821860


namespace slope_of_line_angle_l821_821229

theorem slope_of_line_angle :
  let x := λ y: ℝ, (sqrt 3) * y - 1 in
  ∀ θ : ℝ, (tan θ = (1 / (sqrt 3))) → θ = 30 := sorry

end slope_of_line_angle_l821_821229


namespace range_of_m_value_of_expression_l821_821369

noncomputable def problem1 (α : ℝ) (m : ℝ) : Prop :=
  0 < α ∧ α < π ∧ ∃ x : ℝ, (x^2 + 4 * x * sin(α/2) + m * tan(α/2) = 0)

theorem range_of_m (α : ℝ) (m : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : ∃ x : ℝ, (x^2 + 4 * x * sin(α/2) + m * tan(α/2) = 0) ∧ 
    (x^2 + 4 * x * sin(α/2) + m * tan(α/2) = 0)) :
  m ∈ set.Ioc 0 2 :=
sorry

noncomputable def problem2 (α : ℝ) (m : ℝ) : Prop :=
  0 < α ∧ α < π ∧ m + 2 * cos α = 4 / 3

theorem value_of_expression (α : ℝ) (m : ℝ) (h1 : 0 < α) (h2 : α < π)
  (h3 : m + 2 * cos α = 4 / 3) :
  (1 + sin(2 * α) - cos(2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end range_of_m_value_of_expression_l821_821369


namespace hats_problem_l821_821303

theorem hats_problem 
  (H : ℕ)
  (brown_fraction : ℚ)
  (sold_brown_fraction : ℚ)
  (unsold_brown_fraction : ℚ) :
  brown_fraction = 1 / 4 -> 
  sold_brown_fraction = 4 / 5 -> 
  unsold_brown_fraction = 0.15 -> 
  ∃ F : ℚ, F = 2 / 3 :=
by
  assume h1 : brown_fraction = 1 / 4
  assume h2 : sold_brown_fraction = 4 / 5
  assume h3 : unsold_brown_fraction = 0.15
  use 2 / 3
  sorry

end hats_problem_l821_821303


namespace probability_f_has_zero_point_l821_821040

-- Defining the binomial distribution of X with parameters n=5 and p=1/2
noncomputable def X : ℕ → ℝ := λ k, ℙ (binomial 5 (1/2)) k

-- Function f(x) = x² + 4x + X
def f (x : ℝ) (X : ℝ) : ℝ := x^2 + 4*x + X

-- Event that f(x) has a zero point
def has_zero_point (X : ℕ) : Prop :=
  ∃ x : ℝ, f x (X : ℝ) = 0

-- Theorem statement: the probability that f(x) has a zero point given X ~ Binomial(5, 1/2) is 31/32
theorem probability_f_has_zero_point : ℙ (has_zero_point (X)) = 31/32 :=
sorry

end probability_f_has_zero_point_l821_821040


namespace fixed_point_of_invariant_line_l821_821588

theorem fixed_point_of_invariant_line :
  ∀ (m : ℝ) (x y : ℝ), (3 * m + 4) * x + (5 - 2 * m) * y + 7 * m - 6 = 0 →
  (x = -1 ∧ y = 2) :=
by
  intro m x y h
  sorry

end fixed_point_of_invariant_line_l821_821588


namespace find_z_l821_821627

open Complex

theorem find_z 
  (z : ℂ) 
  (h : conj z * (1 + 2 * I) = 4 + 3 * I) : z = 2 + I := 
by
  sorry

end find_z_l821_821627


namespace ratio_of_areas_l821_821266

noncomputable def radius_AB := 5
noncomputable def radius_AC := 3
noncomputable def radius_CB := 2
noncomputable def shaded_area := (1/2 : ℝ) * (radius_AB ^ 2) * π - (1/2 : ℝ) * (radius_AC ^ 2) * π - (1/2 : ℝ) * (radius_CB ^ 2) * π
noncomputable def radius_CD := 2 * Real.sqrt 6
noncomputable def area_circle_CD := (radius_CD ^ 2) * π

theorem ratio_of_areas : (shaded_area / area_circle_CD) = (5 / 24) := sorry

end ratio_of_areas_l821_821266


namespace prob_multiple_of_4_or_6_l821_821607

theorem prob_multiple_of_4_or_6 :
  let n := 60
  let unselectable := {58, 60}
  let remaining_balls := {1..n} \ unselectable
  let multiples_of_4 := { k in remaining_balls | k % 4 = 0 }
  let multiples_of_6 := { k in remaining_balls | k % 6 = 0 }
  let multiples_of_4_or_6 := multiples_of_4 ∪ multiples_of_6
  let favorable_count := multiples_of_4_or_6.card
  let total_count := remaining_balls.card
  (favorable_count : ℚ) / total_count = 19 / 58 :=
by
  sorry

end prob_multiple_of_4_or_6_l821_821607


namespace cuboid_height_l821_821512

-- Define the conditions

def small_cube_side : ℕ := 1

def base_area : ℕ := 4

def total_cubes : ℕ := 12 -- Derived from the problem context

-- Define the proof statement
theorem cuboid_height (h : ℕ) (v : ℕ) 
    (h₁ : v = total_cubes)
    (h₂ : v / base_area = h) :
    h = 3 := 
by {
    -- Include the conditions
    have h₃ : base_area = small_cube_side * small_cube_side * 4 := by {
        norm_num [small_cube_side, base_area]
    },
    sorry
}

end cuboid_height_l821_821512


namespace sum_of_nonnegative_reals_l821_821923

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l821_821923


namespace inequalities_hold_l821_821013

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l821_821013


namespace max_value_of_quadratic_l821_821879

theorem max_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y ≥ -x^2 + 5 * x - 4) ∧ y = 9 / 4 :=
sorry

end max_value_of_quadratic_l821_821879


namespace hyperbola_sum_l821_821090

noncomputable def hyperbola_constants : ℝ × ℝ × ℝ × ℝ := 
  let h : ℝ := -3
  let k : ℝ := 0
  let a : ℝ := abs(h - (-7))
  let c : ℝ := abs(h - (-3 + real.sqrt 41))
  let b : ℝ := real.sqrt (c^2 - a^2)
  (h, k, a, b)

theorem hyperbola_sum : 
  let (h, k, a, b) := hyperbola_constants 
  h + k + a + b = 6 :=
by 
  -- Introduce variables h, k, a, b from the structure hyperbola_constants
  let (h, k, a, b) := hyperbola_constants 
  sorry

end hyperbola_sum_l821_821090


namespace angle_is_two_pi_over_three_l821_821864

noncomputable def angle_between_vectors (a b : ℝ → ℝ) : ℝ := 
  let cos_theta := (2 * (a 1) + (b 1)) * (a 1) = 0 in
  if cos_theta = 0 then real.arccos (-1 / 2) else 0

theorem angle_is_two_pi_over_three {a b : ℝ → ℝ} 
  (a_mag : |a 1| = 1) 
  (b_mag : |b 1| = 4) 
  (h : (2 * a 1 + b 1) * a 1 = 0) 
  : angle_between_vectors a b = 2 * real.pi / 3 := 
sorry

end angle_is_two_pi_over_three_l821_821864


namespace inequalities_hold_l821_821011

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l821_821011


namespace total_fence_poles_l821_821299

def num_poles_per_side : ℕ := 27
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

theorem total_fence_poles : 
  (num_poles_per_side * sides_of_square) - corners_of_square = 104 :=
  sorry

end total_fence_poles_l821_821299


namespace range_of_a_l821_821050

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + 4

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = 0 → (f a x = 0 → x > 0)) ↔ a > 3 := sorry

end range_of_a_l821_821050


namespace blue_lights_count_l821_821236

def num_colored_lights := 350
def num_red_lights := 85
def num_yellow_lights := 112
def num_green_lights := 65
def num_blue_lights := num_colored_lights - (num_red_lights + num_yellow_lights + num_green_lights)

theorem blue_lights_count : num_blue_lights = 88 := by
  sorry

end blue_lights_count_l821_821236


namespace question1_first_yields_higher_reward_l821_821086

variables (a1 a2 p1 p2 ξ1 ξ2 : ℝ)

-- Given Conditions
def condition1 : Prop := a1 = 2 * a2
def condition2 : Prop := p1 + p2 = 1

-- Expected Rewards Definitions
def E_ξ1 : ℝ := a1 * p1 * p1 + (a1 + a2) * p1 * (1 - p1)
def E_ξ2 : ℝ := a2 * (1 - p1) * (1 - p1) + (a1 + a2) * p1 * (1 - p1)

theorem question1_first_yields_higher_reward (h1: condition1) (h2: condition2) : (E_ξ1 > E_ξ2) :=
sorry

end question1_first_yields_higher_reward_l821_821086


namespace equivalent_angles_l821_821993

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Define points A, B, C as vertices of the triangle
variables (A B C D E F : α)

-- Midpoints of the sides
variables (midpoint_BC midpoint_CA midpoint_AB : α)

-- Given that D, E, F are midpoints of the sides BC, CA, and AB respectively
def is_midpoint (P Q R midpoint : α) : Prop :=
  (dist midpoint P = dist midpoint Q) ∧
  (dist midpoint P = dist R / 2) ∧
  (dist midpoint Q = dist R / 2)

variable (h_mid_BC : is_midpoint B C D midpoint_BC)
variable (h_mid_CA : is_midpoint C A E midpoint_CA)
variable (h_mid_AB : is_midpoint A B F midpoint_AB)

-- Angles
variables (angle_DAC angle_ABE angle_AFC angle_BDA : ℝ)

-- Define the angles
variable (h_angle_DAC : angle_DAC = ∡ D A C)
variable (h_angle_ABE : angle_ABE = ∡ A B E)
variable (h_angle_AFC : angle_AFC = ∡ A F C)
variable (h_angle_BDA : angle_BDA = ∡ B D A)

-- The main theorem to prove
theorem equivalent_angles :
  (angle_DAC = angle_ABE) ↔ (angle_AFC = angle_BDA) :=
sorry

end equivalent_angles_l821_821993


namespace part_I_part_II_l821_821415

-- Definitions of functions
def f (x b : ℝ) := abs (x + b^2) - abs (-x + 1)
def g (x a b c : ℝ) := abs (x + a^2 + c^2) + abs (x - 2 * b^2)

-- Assumption about positive real numbers
variables (a b c : ℝ)
-- Condition on the sum
axiom abc_condition : a * b + b * c + a * c = 1

-- Part (Ⅰ)
theorem part_I (b : ℝ) (hb : b = 1) : { x : ℝ | f x b ≥ 1 } = 
  Ici (1 / 2) := 
by
  sorry

-- Part (Ⅱ)
theorem part_II (x : ℝ) : f x b ≤ g x a b c := 
by
  sorry

end part_I_part_II_l821_821415


namespace age_ratio_l821_821727

theorem age_ratio (S M : ℕ) (h₁ : M = S + 35) (h₂ : S = 33) : 
  (M + 2) / (S + 2) = 2 :=
by
  -- proof goes here
  sorry

end age_ratio_l821_821727


namespace flag_arrangement_modulo_l821_821661

theorem flag_arrangement_modulo : 
  let red_flags := 13
  let yellow_flags := 12
  let total_flags := 25
  let gap_options := red_flags + 1
  let total_arrangements := (Nat.choose gap_options yellow_flags) * (red_flags + 1) - 2 * (Nat.choose (red_flags + 1) yellow_flags)
  let M := total_arrangements
  M % 1000 = 188 :=
by {
  let red_flags := 13
  let yellow_flags := 12
  let gap_options := red_flags + 1
  let total_arrangements := (Nat.choose gap_options yellow_flags) * (red_flags + 1) - 2 * (Nat.choose (red_flags + 1) yellow_flags)
  let M := total_arrangements
  show M % 1000 = 188,
  sorry
}

end flag_arrangement_modulo_l821_821661


namespace no_real_solution_l821_821826

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Lean statement: prove that the equation x^2 - 4x + 6 = 0 has no real solution
theorem no_real_solution : ¬ ∃ x : ℝ, f x = 0 :=
sorry

end no_real_solution_l821_821826


namespace appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l821_821350

-- Definitions for binomial coefficient and Pascal's triangle

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Check occurrences in Pascal's triangle more than three times
theorem appears_more_than_three_times_in_Pascal (n : ℕ) :
  n = 10 ∨ n = 15 ∨ n = 21 → ∃ a b c : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ 
    (binomial_coeff a 2 = n ∨ binomial_coeff a 3 = n) ∧
    (binomial_coeff b 2 = n ∨ binomial_coeff b 3 = n) ∧
    (binomial_coeff c 2 = n ∨ binomial_coeff c 3 = n) := 
by
  sorry

-- Check occurrences in Pascal's triangle more than four times
theorem appears_more_than_four_times_in_Pascal (n : ℕ) :
  n = 120 ∨ n = 210 ∨ n = 3003 → ∃ a b c d : ℕ, 
    (1 < a) ∧ (1 < b) ∧ (1 < c) ∧ (1 < d) ∧ 
    (binomial_coeff a 3 = n ∨ binomial_coeff a 4 = n) ∧
    (binomial_coeff b 3 = n ∨ binomial_coeff b 4 = n) ∧
    (binomial_coeff c 3 = n ∨ binomial_coeff c 4 = n) ∧
    (binomial_coeff d 3 = n ∨ binomial_coeff d 4 = n) := 
by
  sorry

end appears_more_than_three_times_in_Pascal_appears_more_than_four_times_in_Pascal_l821_821350


namespace find_xyz_ratio_l821_821378

theorem find_xyz_ratio (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 2) 
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 :=
by sorry

end find_xyz_ratio_l821_821378


namespace power_sum_inequality_l821_821267

theorem power_sum_inequality (k l m : ℕ) : 
  2 ^ (k + l) + 2 ^ (k + m) + 2 ^ (l + m) ≤ 2 ^ (k + l + m + 1) + 1 := 
by 
  sorry

end power_sum_inequality_l821_821267


namespace arithmetic_mean_relation_l821_821372

noncomputable def m : ℝ := (a + b + c + d + e) / 5
noncomputable def k : ℝ := (a + b) / 2
noncomputable def l : ℝ := (c + d + e) / 3
noncomputable def p : ℝ := (k + l) / 2

theorem arithmetic_mean_relation (a b c d e : ℝ) :
  ¬((m = p) ∨ (m > p) ∨ (m < p)) := sorry

end arithmetic_mean_relation_l821_821372


namespace polynomial_equality_l821_821570

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 7
noncomputable def g (x : ℝ) : ℝ := 12 * x^2 - 19 * x + 25

theorem polynomial_equality :
  f 3 = g 3 ∧ f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3) ∧ f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3) :=
by
  sorry

end polynomial_equality_l821_821570


namespace no_prime_between_factorial_plus_2_and_factorial_plus_n_l821_821002

theorem no_prime_between_factorial_plus_2_and_factorial_plus_n (n : ℕ) (h : 2 < n) :
  ∀ k, n! + 2 < k ∧ k < n! + n → ¬ prime k :=
by sorry

end no_prime_between_factorial_plus_2_and_factorial_plus_n_l821_821002


namespace parallel_vectors_lambda_l821_821492

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821492


namespace solve_for_x_l821_821196

theorem solve_for_x (x y : ℚ) :
  (x + 1) / (x - 2) = (y^2 + 4*y + 1) / (y^2 + 4*y - 3) →
  x = -(3*y^2 + 12*y - 1) / 2 :=
by
  intro h
  sorry

end solve_for_x_l821_821196


namespace parallel_vectors_l821_821485

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821485


namespace acceptable_arrangements_correct_l821_821939

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821939


namespace investigate_local_extrema_l821_821542

noncomputable def f (x1 x2 : ℝ) : ℝ :=
  3 * x1^2 * x2 - x1^3 - (4 / 3) * x2^3

def is_local_maximum (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∀ (x y : ℝ × ℝ), dist x c < ε → f x.1 x.2 ≤ f c.1 c.2

def is_saddle_point (f : ℝ → ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ ε > 0, ∃ (x1 y1 x2 y2 : ℝ × ℝ),
    dist x1 c < ε ∧ dist y1 c < ε ∧ dist x2 c < ε ∧ dist y2 c < ε ∧
    (f x1.1 x1.2 > f c.1 c.2 ∧ f y1.1 y1.2 < f c.1 c.2) ∧
    (f x2.1 x2.2 < f c.1 c.2 ∧ f y2.1 y2.2 > f c.1 c.2)

theorem investigate_local_extrema :
  is_local_maximum f (6, 3) ∧ is_saddle_point f (0, 0) :=
sorry

end investigate_local_extrema_l821_821542


namespace pentagon_area_l821_821521

/-- Define the vertices of the pentagon as a list of points. --/
def pentagon_vertices : List (ℤ × ℤ) := [(-3, 1), (1, 1), (1, -2), (-3, -2), (-2, 0)]

/-- Predicate that the area of the pentagon is 13 square units. --/
theorem pentagon_area (h : pentagon_vertices.length = 5) :
    area_of_irregular_polygon pentagon_vertices = 13 :=
sorry -- proof is to be filled

end pentagon_area_l821_821521


namespace magnet_cost_times_sticker_l821_821754

theorem magnet_cost_times_sticker
  (M S A : ℝ)
  (hM : M = 3)
  (hA : A = 6)
  (hMagnetCost : M = (1/4) * 2 * A) :
  M = 4 * S :=
by
  -- Placeholder, the actual proof would go here
  sorry

end magnet_cost_times_sticker_l821_821754


namespace problem_multiple_41_l821_821178

theorem problem_multiple_41 (x y : ℤ) : 
  (25 * x + 31 * y) % 41 = 0 → (3 * x + 7 * y) % 41 = 0 ∧
  (3 * x + 7 * y) % 41 = 0 → (25 * x + 31 * y) % 41 = 0 :=
by
  intros h1 h2
  split
  all_goals 
    sorry

end problem_multiple_41_l821_821178


namespace x_axis_line_l821_821080

variable (A B C : ℝ)

theorem x_axis_line (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : B ≠ 0 ∧ A = 0 ∧ C = 0 := by
  sorry

end x_axis_line_l821_821080


namespace quadratic_real_roots_iff_l821_821701

theorem quadratic_real_roots_iff (α : ℝ) : (∃ x : ℝ, x^2 - 2 * x + α = 0) ↔ α ≤ 1 :=
by
  sorry

end quadratic_real_roots_iff_l821_821701


namespace units_digit_combination_l821_821231

theorem units_digit_combination (a b c : ℕ) (h1 : 2^a % 10 = 2) (h2 : 7^b % 10 = 9) (h3 : 3^c % 10 = 1) :
  2^101 * 7^1002 * 3^1004 % 10 = 8 :=
by
  have h2 : 2^101 % 10 = 2, from h1
  have h7 : 7^1002 % 10 = 9, from h2
  have h3 : 3^1004 % 10 = 1, from h3
  sorry

end units_digit_combination_l821_821231


namespace cyclic_quadrilateral_BXMY_l821_821748

variables {A B C M P Q X Y : Type*}

-- Conditions
def midpoint (M A C : Type*) : Prop := M = (A + C) / 2
def on_segment (P M A : Type*) : Prop := ∃ t : Type*, 0 ≤ t ∧ t ≤ 1 ∧ P = t * M + (1 - t) * A -- P is on segment AM
def length_relation (PQ AC : Type*) : Prop := PQ = AC / 2
def intersection_X (circumcircle_ABQ BC : Type*) (B X : Type*) : Prop := X ∈ circumcircle_ABQ ∧ X ≠ B
def intersection_Y (circumcircle_BCP AB : Type*) (B Y : Type*) : Prop := Y ∈ circumcircle_BCP ∧ Y ≠ B

-- Proof that BXMY is cyclic
theorem cyclic_quadrilateral_BXMY
  (ABC : Type*)
  (M : Type*) (hM : midpoint M A C)
  (P Q : Type*) (hP : on_segment P M A) (hQ : on_segment Q M C)
  (hPQ : length_relation PQ AC)
  (circumcircle_ABQ : set Type*) (circumcircle_BCP : set Type*)
  (hX : intersection_X circumcircle_ABQ BC B X)
  (hY : intersection_Y circumcircle_BCP AB B Y) :
  cyclic (set.of [B, X, M, Y]) :=
sorry

end cyclic_quadrilateral_BXMY_l821_821748


namespace parallel_vectors_lambda_value_l821_821459

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821459


namespace find_expression_max_value_min_value_l821_821414

namespace MathProblem

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := a * x^2 + a^2 * x + 2 * b - a^3

-- Hypotheses based on problem conditions
lemma a_neg (a b : ℝ) : a < 0 := sorry
lemma root_neg2 (a b : ℝ) : f a b (-2) = 0 := sorry
lemma root_6 (a b : ℝ) : f a b 6 = 0 := sorry

-- Proving the explicit expression for f(x)
theorem find_expression (a b : ℝ) (x : ℝ) : 
  a = -4 → 
  b = -8 → 
  f a b x = -4 * x^2 + 16 * x + 48 :=
sorry

-- Maximum value of f(x) on the interval [1, 10]
theorem max_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 2 = 64 :=
sorry

-- Minimum value of f(x) on the interval [1, 10]
theorem min_value (x : ℝ) (h : 1 ≤ x ∧ x ≤ 10) : 
  f (-4) (-8) 10 = -192 :=
sorry

end MathProblem

end find_expression_max_value_min_value_l821_821414


namespace find_b_vector_l821_821561

def vector3 := (ℝ × ℝ × ℝ)

def a : vector3 := (5, -2, -6)
def c : vector3 := (-4, 2, 4)
def b : vector3 := (6.8, -2.8, -8)

def collinear (v1 v2 v3 : vector3) : Prop :=
  ∃ k : ℝ, v2 = (v1.1 + k * (v3.1 - v1.1), v1.2 + k * (v3.2 - v1.2), v1.3 + k * (v3.3 - v1.3))

def dot_product (v1 v2 : vector3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : vector3) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def bisects_angle (v1 v2 v3 : vector3) : Prop :=
  dot_product v1 v2 / (norm v1 * norm v2) = dot_product v2 v3 / (norm v2 * norm v3)

theorem find_b_vector :
  collinear a b c ∧ bisects_angle a b c :=
by
  sorry

end find_b_vector_l821_821561


namespace no_subseq_010101_l821_821533

def sequence (x : ℕ → ℕ) : Prop :=
  ∀ n, n >= 6 → x n = (x (n-6) + x (n-5) + x (n-4) + x (n-3) + x (n-2) + x (n-1)) % 10

theorem no_subseq_010101 (x : ℕ → ℕ) (h : x 0 = 1 ∧ x 1 = 0 ∧ x 2 = 1 ∧ x 3 = 0 ∧ x 4 = 1 ∧ x 5 = 0 ∧ sequence x) :
  ∀ i, ¬(x i = 0 ∧ x (i+1) = 1 ∧ x (i+2) = 0 ∧ x (i+3) = 1 ∧ x (i+4) = 0 ∧ x (i+5) = 1) :=
sorry

end no_subseq_010101_l821_821533


namespace base_number_equals_2_l821_821079

theorem base_number_equals_2 (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^26) (h2 : n = 25) : x = 2 :=
by
  sorry

end base_number_equals_2_l821_821079


namespace problem1_problem2_problem3_l821_821388

-- Define the sequence {a_n}

noncomputable def a : ℕ → ℚ
| 0     => 1 / 2
| (n+1) => 4 * a n + 1

-- Problem 1: Prove a_1 + a_2 + a_3 = 33 / 2
theorem problem1 : a 0 + a 1 + a 2 = 33 / 2 := 
sorry

-- Define the sequence {b_n} where b_n = a_n + 1 / 3
noncomputable def b (n : ℕ) := a n + 1 / 3

-- Problem 2: Prove that {b_n} is a geometric sequence with first term 5/6 and ratio 4
theorem problem2 : ∀ n, b (n + 1) = 4 * b n :=
-- We need to prove the first term b_1 = 5/6
∧ b 0 = 5 / 6
sorry

-- Problem 3: Prove the sum of the first n terms of {b_n} is (5/18)(4^n - 1)
noncomputable def T (n : ℕ) := (∑ i in finset.range n, b i)

theorem problem3 (n : ℕ) : T n = (5 / 18) * (4^n - 1) := 
sorry

end problem1_problem2_problem3_l821_821388


namespace container_mass_and_water_l821_821118

-- Definitions of the given conditions
def mass_of_vessel (A B : ℝ) (x y : ℝ) : Prop :=
  A = (4 / 5) * y ∧
  A + (y - x) = 8 * x ∧
  (y - x) - ((4 / 5) * y - x) = 50

-- The main result to prove given the conditions
theorem container_mass_and_water :
  ∃ (x y : ℝ), mass_of_vessel (4 / 5 * y) y x ∧ x = 50 ∧ (4 / 5 * y - x) = 150 ∧ (y - x) = 200 :=
by {
  sorry
}

end container_mass_and_water_l821_821118


namespace range_of_x_plus_y_l821_821134

theorem range_of_x_plus_y (x y : ℝ) (hx1 : y = 3 * ⌊x⌋ + 4) (hx2 : y = 4 * ⌊x - 3⌋ + 7) (hxnint : ¬ ∃ z : ℤ, x = z): 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end range_of_x_plus_y_l821_821134


namespace solve_quadratic_eqns_l821_821194

theorem solve_quadratic_eqns :
  (∃ x1 x2 : ℝ, (x1 = -6 ∧ x2 = 2) ∧ (x+1) * (x+3) = 15 → (x = x1 ∨ x = x2)) ∧
  (∃ y1 y2 : ℝ, (y1 = 1 ∧ y2 = 2) ∧ (y-3)^2 + 3*(y-3) + 2 = 0 → (y = y1 ∨ y = y2)) := 
begin
  sorry
end

end solve_quadratic_eqns_l821_821194


namespace ellipse_equation_constant_distance_l821_821024

theorem ellipse_equation {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : ∃ x y : ℝ, (x, y) = (0, 1)) (h4 : (a^2 - b^2 = 1)) :
  (∃ c d : ℝ, (c /a)^2 + (d /b)^2 = 1) :=
sorry

theorem constant_distance {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : ∃ x y : ℝ, (x, y) = (0, 1)) (h4 : (a^2 - b^2 = 1)) 
  (h5 : - sqrt 2 < m < sqrt 2) :
  ∃ B N : ℝ, (sqrt 10 / 2) :=
sorry

end ellipse_equation_constant_distance_l821_821024


namespace total_ladders_climbed_in_inches_l821_821124

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l821_821124


namespace x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l821_821567

variable (x : ℝ)

theorem x_positive_implies_abs_positive (hx : x > 0) : |x| > 0 := sorry

theorem abs_positive_not_necessiarily_x_positive : (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := sorry

theorem x_positive_is_sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ 
  (∃ x : ℝ, |x| > 0 ∧ ¬(x > 0)) := 
  ⟨x_positive_implies_abs_positive, abs_positive_not_necessiarily_x_positive⟩

end x_positive_implies_abs_positive_abs_positive_not_necessiarily_x_positive_x_positive_is_sufficient_but_not_necessary_l821_821567


namespace factor_x_squared_minus_64_l821_821797

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821797


namespace angle_equality_l821_821135

-- Definitions for the given problem
variable (ω1 ω2 : Circle) (A B X Y : Point)

-- Conditions we need: tangency, containment, distinct points, and tangents
variable (tangent_at_A : tangent_at ω1 ω2 A)
variable (ω2_inside_ω1 : inside ω2 ω1)
variable (B_on_ω2 : on_circle B ω2)
variable (B_ne_A : B ≠ A)
variable (tangent_at_B : Tangent ω2 B X Y ω1)

-- Statement to prove
theorem angle_equality :
  ∠ B A X = ∠ B A Y := by
  sorry

end angle_equality_l821_821135


namespace rate_of_current_l821_821302

theorem rate_of_current : 
  ∀ (v c : ℝ), v = 3.3 → (∀ d: ℝ, d > 0 → (d / (v - c) = 2 * (d / (v + c))) → c = 1.1) :=
by
  intros v c hv h
  sorry

end rate_of_current_l821_821302


namespace square_octagon_can_cover_ground_l821_821259

def square_interior_angle := 90
def octagon_interior_angle := 135

theorem square_octagon_can_cover_ground :
  square_interior_angle + 2 * octagon_interior_angle = 360 :=
by
  -- Proof skipped with sorry
  sorry

end square_octagon_can_cover_ground_l821_821259


namespace distance_from_point_to_plane_l821_821597

variables {α : Type} [metric_space α] -- assuming α is a metric space
variables {A B : α} -- Points A and B in space
variable (B1 : α) -- B1 is the orthogonal projection of B onto plane α
variable (plane : set α) -- plane α 

-- Conditions
variable (hA : A ∈ plane) -- Point A lies on plane α
variable (h_proj_AB : dist A B1 = 1) -- The orthogonal projection of segment AB onto the plane is equal to 1
variable (h_AB : dist A B = 2) -- The length of segment AB is 2
variable (h_perp : B1 ∈ plane) -- B1 lies on the plane (orthogonal projection implies this)

-- Question: Find the distance from point B to plane α (which is the same as finding dist B B1)
theorem distance_from_point_to_plane : dist B B1 = sqrt 3 :=
sorry

end distance_from_point_to_plane_l821_821597


namespace smallest_angle_l821_821975

noncomputable def smallest_angle_in_triangle (a b c : ℝ) : ℝ :=
  if h : 0 <= a ∧ 0 <= b ∧ 0 <= c ∧ a + b > c ∧ a + c > b ∧ b + c > a then
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  else
    0

theorem smallest_angle (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : c = 2) :
  smallest_angle_in_triangle a b c = Real.arccos (7 / 8) :=
sorry

end smallest_angle_l821_821975


namespace ellipse_eqn_max_AB_collinear_k_l821_821025

-- Definitions and conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (e : ℝ) (c a : ℝ) : Prop := e = c / a
def focal_distance (d c : ℝ) : Prop := d = 2 * c
def line_slope (k : ℝ) : Prop := k = 1
def point_P := (-2 : ℝ, 0 : ℝ)
def point_Q := (- (7 : ℝ) / 4, 1 : ℝ / 4)

-- Statements to be proven
theorem ellipse_eqn (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (e : ℝ) (h3 : eccentricity e (sqrt 2) a)
    (d : ℝ) (h4 : focal_distance d (sqrt 2)) : 
  ellipse = fun (x y : ℝ) => (x^2 / 3) + y^2 = 1 :=
sorry

theorem max_AB (k m x1 x2 y1 y2 : ℝ) (h1 : line_slope k) 
    (h2 : ∀ m, -2 < m ∧ m < 2) : 
  |AB| = sqrt 6 :=
sorry

theorem collinear_k (x1 x2 y1 y2 : ℝ) (h1 : line_slope k) 
    (h2 : point_P = (-2, 0)) (h3 : point_Q = (- (7 : ℝ) / 4, 1 : ℝ / 4)) : 
  k = 1 :=
sorry

end ellipse_eqn_max_AB_collinear_k_l821_821025


namespace gain_percentage_l821_821500

theorem gain_percentage (C1 C2 SP1 SP2 : ℝ) (h1 : C1 + C2 = 540) (h2 : C1 = 315)
    (h3 : SP1 = C1 - (0.15 * C1)) (h4 : SP1 = SP2) :
    ((SP2 - C2) / C2) * 100 = 19 :=
by
  sorry

end gain_percentage_l821_821500


namespace max_diff_in_grid_l821_821355

theorem max_diff_in_grid (m : ℕ) (h : 2 ≤ m) :
  ∃ (n : ℕ), (∀ (grid : array m (array m ℕ)) (cell_values : ∀ i j, grid i j ∈ finset.range (m^2 + 1)),
  ∃ (i j k l : ℕ), (i ≠ k ∨ j ≠ l)) ∧ (grid i j, grid k l ∈ finset.range (m^2 + 1)) ∧ (abs (grid i j - grid k l) ≥ n) :=
  n = (m^2 + m - 2) / 2 :=
sorry

end max_diff_in_grid_l821_821355


namespace find_phi_l821_821875

variables {φ : ℝ} {k : ℤ}

def function_symmetric (φ : ℝ) (x : ℝ) : Prop :=
  y = 2 * sin(2 * x + φ) → y = 2 * sin(2 * (π/6 - x) + φ)

theorem find_phi:
  (∀ x, function_symmetric φ x) 
  ∧ (-π/2 < φ) 
  ∧ (φ < π/2) 
  → φ = π/6 :=
sorry

end find_phi_l821_821875


namespace consecutive_num_odd_n_l821_821007

theorem consecutive_num_odd_n (n : ℕ) (a : ℕ) :
  (a * (a + 1) * (a + 2) * ... * (a + n - 1) = n * a + (n * (n - 1)) / 2) →
  (odd n) :=
by
  sorry

end consecutive_num_odd_n_l821_821007


namespace num_pairs_product_neg_six_l821_821917

theorem num_pairs_product_neg_six : 
  (∃ (s : set (ℤ × ℤ)), 
     (∀ ⦃a b : ℤ⦄, (a, b) ∈ s ↔ a * b = -6) ∧ 
     s.card = 4) := 
sorry

end num_pairs_product_neg_six_l821_821917


namespace slope_sum_of_line_cutting_parallelogram_into_congruent_polygons_l821_821335

theorem slope_sum_of_line_cutting_parallelogram_into_congruent_polygons :
  let A := (4, 20)
  let B := (4, 56)
  let C := (13, 81)
  let D := (13, 45)
  ∃ (m n : ℤ), (∀ k, k ∣ m → k ∣ n → k = 1) ∧ (m + n = 62) ∧ (line_through_origin (m, n) (set_of (λ p, parallelogram_polygon p A B C D))) := sorry

end slope_sum_of_line_cutting_parallelogram_into_congruent_polygons_l821_821335


namespace find_f_inv_six_l821_821553

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f(f(x)) = 1 / (3 * x)
axiom specific_value : f(2) = 1 / 9

theorem find_f_inv_six : f(1 / 6) = 3 := sorry

end find_f_inv_six_l821_821553


namespace prime_exp_sum_not_perfect_power_l821_821554

theorem prime_exp_sum_not_perfect_power (p : ℕ) (a n : ℕ) (hp : p.prime) (ha : 0 < a) (hn : 0 < n) : 
  2^p + 3^p = a^n → n = 1 :=
by sorry

end prime_exp_sum_not_perfect_power_l821_821554


namespace max_common_elements_l821_821810

-- Define the main problem statement
theorem max_common_elements :
  ∀ (S : set (set (fin 1000))) (H : (S.card = 5)) (Hsub : ∀ A ∈ S, A.card = 500),
  ∃ A B ∈ S, A ≠ B ∧ (A ∩ B).card ≥ 200 :=
by sorry

end max_common_elements_l821_821810


namespace parallel_vectors_implies_value_of_λ_l821_821428

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821428


namespace cost_of_candy_l821_821770

theorem cost_of_candy (initial_amount remaining_amount : ℕ) (h_init : initial_amount = 4) (h_remaining : remaining_amount = 3) : initial_amount - remaining_amount = 1 :=
by
  sorry

end cost_of_candy_l821_821770


namespace maximize_area_APF_l821_821042

-- Introducing the necessary constants and variables
def a : ℝ := 3
def b : ℝ := sqrt 5
def c : ℝ := sqrt (a^2 - b^2)  -- focal distance
def F : ℝ × ℝ := (c, 0)
def F' : ℝ × ℝ := (-c, 0)
def A : ℝ × ℝ := (0, 2 * sqrt 3)

-- Ellipse equation
def ellipse (p: ℝ × ℝ) : Prop := (p.1 ^ 2) / 9 + (p.2 ^ 2) / 5 = 1

-- Define the condition that P is on the ellipse
def P_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P

-- Define condition that P is on the extension line of AF'
def collinear (P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, P = (k * (A.1 - F'.1) + F'.1, k * (A.2 - F'.2) + F'.2) ∧ k ≠ 0

-- Define the area of triangle APF when perimeter is maximized
def area_APF_maximized (P : ℝ × ℝ) : ℝ :=
  abs ((2 * F.1) * (A.2 - P.2)) / 2

-- The theorem to be proved
theorem maximize_area_APF :
  ∀ (P : ℝ × ℝ), P_on_ellipse P → collinear P → area_APF_maximized P = 21 * sqrt 3 / 4 :=
by
  intros P hP_on_ellipse hCollinear
  sorry

end maximize_area_APF_l821_821042


namespace kendra_weekday_shirts_l821_821551

variable (W : ℕ) (shifts_club per_sunday per_saturday : ℕ)

axiom shirts_after_school_club : shifts_club = 3 * 2
axiom shirts_saturdays : per_saturday = 1 * 2
axiom shirts_sundays : per_sunday = 2 * 2
axiom total_shirts_needed : W + (shifts_club + per_saturday + per_sunday) = 22

theorem kendra_weekday_shirts : W = 5 :=
by
  add_calc
  sorry

end kendra_weekday_shirts_l821_821551


namespace cannot_reach_eighth_vertex_l821_821101

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end cannot_reach_eighth_vertex_l821_821101


namespace sequence_a_2017_equals_4033_l821_821389

open Nat

def arithmetic_sequence (A B : ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, (∑ i in finRange n.succ, a i) = A * n * n + B * n

theorem sequence_a_2017_equals_4033
  (A B : ℤ)
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence A B a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3) :
  a 2017 = 4033 := 
sorry

end sequence_a_2017_equals_4033_l821_821389


namespace ratio_prime_to_composite_l821_821377

-- Definition of prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Given list of numbers
def numbers_list := [3, 4, 5, 6, 7, 8, 9]

-- Definition to get all prime numbers in the list
def prime_numbers := numbers_list.filter is_prime

-- Definition to get all composite numbers in the list
def is_composite (n : ℕ) : Prop := ¬is_prime n ∧ n > 1
def composite_numbers := numbers_list.filter is_composite

-- The goal statement
theorem ratio_prime_to_composite : prime_numbers.length / composite_numbers.length = 3 / 4 := 
sorry

end ratio_prime_to_composite_l821_821377


namespace max_a_and_min_squares_sum_condition_l821_821876

/-- Given functions f and g and inequality condition. Prove the maximum value of a and minimum value of squares sum under given constraints. -/
theorem max_a_and_min_squares_sum_condition
  (f g : ℝ → ℝ)
  (θ : ℝ)
  (hgθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi)
  (hf : ∀ x : ℝ, f x = |x + sin θ ^ 2|)
  (hg : ∀ x : ℝ, g x = 2 * |x - cos θ ^ 2|)
  (ineq : ∀ x : ℝ, 2 * f x ≥ a - g x) :
  (a ≤ 2) ∧ (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (a + 2 * b + 3 * c = 4) →
  (a^2 + b^2 + c^2) ≥ 8 / 7) :=
by
  sorry

end max_a_and_min_squares_sum_condition_l821_821876


namespace f_2007_2007_l821_821566

def f (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d * d) |>.sum

def f_k : ℕ → ℕ → ℕ
| 0, n => n
| (k+1), n => f (f_k k n)

theorem f_2007_2007 : f_k 2007 2007 = 145 :=
by
  sorry -- Proof omitted

end f_2007_2007_l821_821566


namespace perpendicular_DPEF_l821_821994

variables {A B C Q D E F I P : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space Q] [metric_space D]
variables [metric_space E] [metric_space F] [metric_space I] [metric_space P]

-- Define the triangle ABC
variable (ABC : triangle A B C)

-- Define point Q with the given perpendicular conditions
variable (Q_cond1 : AB ⟂ QB)
variable (Q_cond2 : AC ⟂ QC)

-- Define the inscribed circle with center I
variable (circle_inscribed : incircle I ABC)

-- Define the tangency points D, E, F
variable (tangent_points : tangents_at circle_inscribed BC D CA E AB F)

-- Define the intersection of line QI with EF at point P
variable (ray_intersects : intersect_at (line_through Q I) EF P)

-- The theorem statement that needs to be proven
theorem perpendicular_DPEF : DP ⟂ EF :=
sorry

end perpendicular_DPEF_l821_821994


namespace pentagon_concyclic_l821_821569

noncomputable def TBA_angle_equal : Prop :=
  ∃ (A B C D E T P Q R S : Type)
    (h1 : BC = DE)
    (h2 : TB = TD)
    (h3 : TC = TE)
    (h4 : ∠TBA = ∠AET)
    (h5 : collinear P B A Q)
    (h6 : collinear R E A S)
    (h7 : inside_polygon T ABCDE),
  P, S, Q, R are_concyclic

theorem pentagon_concyclic
  (A B C D E T P Q R S : Type)
  (BC_eq_DE : BC = DE)
  (TB_eq_TD : TB = TD)
  (TC_eq_TE : TC = TE)
  (TBA_eq_AET : ∠TBA = ∠AET)
  (collinear_PBAQ : collinear P B A Q)
  (collinear_REAS : collinear R E A S)
  (T_inside_ABCDE : inside_polygon T ABCDE) :
  are_concyclic P S Q R :=
by
  -- Proof omitted
  sorry

end pentagon_concyclic_l821_821569


namespace lambs_goats_solution_l821_821686

theorem lambs_goats_solution : ∃ l g : ℕ, l > 0 ∧ g > 0 ∧ 30 * l + 32 * g = 1200 ∧ l = 24 ∧ g = 15 :=
by
  existsi 24
  existsi 15
  repeat { split }
  sorry

end lambs_goats_solution_l821_821686


namespace largest_sum_l821_821765

theorem largest_sum :
  max (max (max (max (1/4 + 1/9) (1/4 + 1/10)) (1/4 + 1/11)) (1/4 + 1/12)) (1/4 + 1/13) = 13/36 := 
sorry

end largest_sum_l821_821765


namespace max_value_of_f_l821_821365

def f (x : ℝ) : ℝ := 10 * x - 2 * x ^ 2

theorem max_value_of_f : ∃ M : ℝ, (∀ x : ℝ, f x ≤ M) ∧ (∃ x : ℝ, f x = M) :=
  ⟨12.5, sorry⟩

end max_value_of_f_l821_821365


namespace gcd_180_270_eq_90_l821_821671

-- Problem Statement
theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := 
by 
  sorry

end gcd_180_270_eq_90_l821_821671


namespace isosceles_triangle_vertex_cosine_l821_821938

theorem isosceles_triangle_vertex_cosine (B : ℝ) (h₁ : B ≠ 0) (h₂ : B ≠ π)
  (h_isosceles : ∃ C, C = B) (h_cos_base : cos B = 2 / 3) :
  cos (π - 2 * B) = 1 / 9 :=
by
  sorry

end isosceles_triangle_vertex_cosine_l821_821938


namespace gym_cardio_replacement_cost_l821_821722

theorem gym_cardio_replacement_cost :
  (let 
    num_gyms := 20,
    num_bikes_per_gym := 10,
    num_treadmills_per_gym := 5,
    num_ellipticals_per_gym := 5,
    cost_per_bike := 700,
    cost_per_treadmill := cost_per_bike + cost_per_bike / 2,
    cost_per_elliptical := 2 * cost_per_treadmill
  in
    let
      total_cost_one_gym := num_bikes_per_gym * cost_per_bike +
                            num_treadmills_per_gym * cost_per_treadmill +
                            num_ellipticals_per_gym * cost_per_elliptical
    in
      num_gyms * total_cost_one_gym = 455000) :=
  sorry

end gym_cardio_replacement_cost_l821_821722


namespace myopia_analysis_l821_821246

-- Given conditions and parameters
def a := 250 -- Number of Myopic Students in School A
def b := 250 -- Number of Non-Myopic Students in School A
def c := 300 -- Number of Myopic Students in School B
def d := 200 -- Number of Non-Myopic Students in School B
def n := a + b + c + d

def k2 := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

def freq_school_A := a / (a + b : ℝ)
def freq_school_B := c / (c + d : ℝ)

theorem myopia_analysis :
  freq_school_A = 0.5 ∧ 
  freq_school_B = 0.6 ∧
  k2 > 6.635 :=
by
  sorry

end myopia_analysis_l821_821246


namespace cosine_of_angle_between_vectors_a_b_l821_821058

open Real EuclideanSpace

noncomputable def cos_theta_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (inner a b) / (norm a * norm b)

theorem cosine_of_angle_between_vectors_a_b :
  let a : EuclideanSpace ℝ (Fin 2) := ![1, 1]
  let b : EuclideanSpace ℝ (Fin 2) := ![2, 0]
  cos_theta_between_vectors a b = (sqrt 2) / 2 :=
by
  let a : EuclideanSpace ℝ (Fin 2) := ![1, 1]
  let b : EuclideanSpace ℝ (Fin 2) := ![2, 0]
  sorry

end cosine_of_angle_between_vectors_a_b_l821_821058


namespace non_adjacent_arrangements_l821_821968

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821968


namespace ellipse_standard_equation_maximum_area_of_triangle_l821_821838

section Ellipse
variables {a b : ℝ}
variables (e : ℝ) (h_e : e = sqrt 6 / 3) (h_perimeter : ∀ A B F1 F2 : ℝ, F1 lies on AB ∧ (BF1 + BF2 + AF1 + AF2 = 4sqrt 3))

noncomputable def standard_equation_of_ellipse : Prop :=
  ∃ (a b c : ℝ), (a = sqrt 3 ∧ c = sqrt 2 ∧ b = 1 ∧ a > b > 0 ∧ e = c / a ∧ 
  a > 0 ∧ b > 0) ∧ (∀ x y, (x ∈ ℝ ∧ y ∈ ℝ) → (x^2 / 3 + y^2 = 1))

theorem ellipse_standard_equation : standard_equation_of_ellipse e :=
by
  use a,
  use b,
  use c,
  have a = sqrt 3 := by sorry,
  have b = 1 := by sorry,
  have c = sqrt 2 := by sorry,
  have a > b > 0 := by sorry,
  have e = c / a := by sorry,
  use a, b, c,
  intro x y hx hy,
  split; intro h,
  { simp only [eq_self_iff_true],
    sorry }
end

section Circle
variables {x0 y0 k1 k2 : ℝ}
variables (h_circle : x0^2 + y0^2 = 4)
variables (h_tangent : ∀P, ∃(M N : ℝ), is_tangent P M ∧ is_tangent P N)

noncomputable def max_area_triangle_PMN : ℝ :=
  ∃ (area : ℝ), area = 4

theorem maximum_area_of_triangle (P : ℝ) : max_area_triangle_PMN :=
by
  use 4,
  have k1 * k2 = -1 := by sorry,
  have MN = diameter := by sorry,
  have position_of_P := by sorry,
  have area_of_triangle := 1 / 2 * 4 * 2 := by sorry,
  simp only [eq_self_iff_true],
  intro h,
  { simp only [eq_self_iff_true],
    sorry }
end

end ellipse_standard_equation_maximum_area_of_triangle_l821_821838


namespace standard_form_of_parabola_l821_821863

theorem standard_form_of_parabola
  (k : ℝ) (h : k = 2) (p : ℝ) (hp : p > 0) (ha : -p / 2 = -k) :
  (y : ℝ) (x : ℝ) : y^2 = 2 * 4 * x :=
by {
  let p := 4,
  have : p = 4 := by { exact sorry },
  sorry
}

end standard_form_of_parabola_l821_821863


namespace preference_change_difference_l821_821328

theorem preference_change_difference :
  let initial_online := 0.40
  let initial_traditional := 0.60
  let final_online := 0.80
  let final_traditional := 0.20
  let min_change := 0.40 -- minimum % change scenario
  let max_change := 0.80 -- maximum % change scenario
  in max_change - min_change = 0.40 :=
by
  -- The proof part is left as an exercise
  sorry

end preference_change_difference_l821_821328


namespace parallel_vectors_lambda_l821_821446

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821446


namespace total_ladders_climbed_in_inches_l821_821125

-- Define the conditions as hypotheses
def keaton_ladder_length := 30
def keaton_climbs := 20
def reece_ladder_difference := 4
def reece_climbs := 15
def feet_to_inches := 12

-- Define the lengths climbed by Keaton and Reece
def keaton_total_feet := keaton_ladder_length * keaton_climbs
def reece_ladder_length := keaton_ladder_length - reece_ladder_difference
def reece_total_feet := reece_ladder_length * reece_climbs

-- Calculate the total feet climbed and convert to inches
def total_feet := keaton_total_feet + reece_total_feet
def total_inches := total_feet * feet_to_inches

-- Prove the final result
theorem total_ladders_climbed_in_inches : total_inches = 11880 := by
  sorry

end total_ladders_climbed_in_inches_l821_821125


namespace symmetry_center_of_tangent_function_l821_821655

theorem symmetry_center_of_tangent_function (k : ℤ) : 
  is_symmetry_center (λ x, (1 / 2) * Real.tan (2 * x + (Real.pi / 3)) + 1) 
                     (↑(1 / 4 * k * Real.pi - Real.pi / 6), 1) :=
sorry

end symmetry_center_of_tangent_function_l821_821655


namespace solution_set_l821_821038

-- Define the function f based on the given conditions
def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

-- State the main theorem to prove
theorem solution_set :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end solution_set_l821_821038


namespace probability_one_boy_one_girl_l821_821828

/-- 
From a group of 3 boys and 2 girls, 
the probability that 2 randomly selected participants consist of exactly 1 boy and 1 girl is 3/5.
-/
theorem probability_one_boy_one_girl (total_boys total_girls total_participants select_participants : Nat)
    (h_boys : total_boys = 3)
    (h_girls : total_girls = 2)
    (total_counts : total_participants = total_boys + total_girls)
    (h_select : select_participants = 2) :
    let total_ways := choose total_participants select_participants
    let ways_one_boy := choose total_boys 1
    let ways_one_girl := choose total_girls 1
    let desired_ways := ways_one_boy * ways_one_girl
    let probability := desired_ways / total_ways
    probability = 3 / 5 :=
by
  sorry

end probability_one_boy_one_girl_l821_821828


namespace log_implication_l821_821683

theorem log_implication (x : ℝ) : ¬((x < 0 → log (x + 1) ≤ 0) ∧ (log (x + 1) ≤ 0 → x < 0)) :=
by
  sorry

end log_implication_l821_821683


namespace find_remainder_for_x_150_div_x2_4x_3_l821_821813

def polynomial_remainder (x : ℕ) : ℚ :=
  (3^x - 1) * (x : ℚ) / 2 + (4 - 3^x) / 2

theorem find_remainder_for_x_150_div_x2_4x_3 :
  ∃ Q(x) R, x^150 = (x - 3) * (x - 1) * Q(x) + R ∧
  degree R < 2 ∧
  R = polynomial_remainder 150 :=
sorry

end find_remainder_for_x_150_div_x2_4x_3_l821_821813


namespace graph_of_direct_proportion_is_line_l821_821635

-- Define the direct proportion function
def direct_proportion (k : ℝ) (x : ℝ) : ℝ :=
  k * x

-- State the theorem to prove the graph of this function is a straight line
theorem graph_of_direct_proportion_is_line (k : ℝ) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, direct_proportion k x = a * x + b ∧ b = 0 := 
by 
  sorry

end graph_of_direct_proportion_is_line_l821_821635


namespace possible_range_of_c_l821_821116

theorem possible_range_of_c (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (hC_obtuse : c^2 > a^2 + b^2) (triangle_inequality : c < a + b) :
  sqrt 13 < c ∧ c < 5 :=
by
  have h_a : a = 2 := h1
  have h_b : b = 3 := h2
  have h_c_obtuse : c^2 > a^2 + b^2 := hC_obtuse
  have h_triangle_inequality : c < a + b := triangle_inequality
  sorry

end possible_range_of_c_l821_821116


namespace perpendicular_lines_parallel_lines_distance_l821_821841

-- Definitions of the lines l1 and l2
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a ^ 2 - 1 = 0

-- Definition for lines being perpendicular
def lines_perpendicular (a : ℝ) : Prop :=
  let slope_l1 := -a / 2
  let slope_l2 := -1 / (a - 1)
  slope_l1 * slope_l2 = -1

-- Definition for lines being parallel
def lines_parallel (a : ℝ) : Prop :=
  let slope_l1 := -a / 2
  let slope_l2 := -1 / (a - 1)
  slope_l1 = slope_l2

-- Proving that if l1 ⟂ l2, then a = 2/3
theorem perpendicular_lines (a : ℝ) : lines_perpendicular a → a = 2 / 3 :=
by
  intro h
  sorry

-- Distance between two parallel lines
def distance_between_parallel_lines (a : ℝ) : ℝ :=
  let A := 1
  let B := -2
  let C₁ := -6
  let C₂ := 0
  (|C₁ - C₂|) / (Real.sqrt (A^2 + B^2))

-- Proving that if l1 || l2 and a = -1, the distance between l1 and l2 is 6 sqrt(5) / 5
theorem parallel_lines_distance (a : ℝ) : lines_parallel a ∧ a = -1 → distance_between_parallel_lines a = 6 * Real.sqrt 5 / 5 :=
by
  intro h
  sorry

end perpendicular_lines_parallel_lines_distance_l821_821841


namespace count_expressible_integers_l821_821065

def g (x : ℝ) : ℕ :=
  ⌊2 * x⌋₊ + ⌊4 * x⌋₊ + ⌊6 * x⌋₊ + ⌊8 * x⌋₊ + ⌊10 * x⌋₊

theorem count_expressible_integers :
  ∃ (n : ℕ), n = 2000 ∧ (finset.range 2000).filter (λ m, ∃ x : ℝ, g x = m).card = 1666 :=
by
  sorry

end count_expressible_integers_l821_821065


namespace count_divisors_of_25320_is_7_l821_821898

theorem count_divisors_of_25320_is_7 : 
  ∃ (s : Set ℕ), s = {n | n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 25320 % n = 0} ∧ s.card = 7 :=
by
  sorry

end count_divisors_of_25320_is_7_l821_821898


namespace achieve_daily_profit_of_200_maximize_daily_profit_l821_821740

def initial_price_per_kg : ℝ := 3
def purchase_price_per_kg : ℝ := 2
def initial_daily_sales : ℝ := 200
def additional_sales_per_0_1_reduction : ℝ := 40
def daily_fixed_costs : ℝ := 24

def profit (price_reduction : ℝ) : ℝ :=
  let selling_price := initial_price_per_kg - price_reduction
  let daily_sales := initial_daily_sales + additional_sales_per_0_1_reduction * price_reduction / 0.1
  (selling_price - purchase_price_per_kg) * daily_sales - daily_fixed_costs

theorem achieve_daily_profit_of_200 (price_reduction : ℝ) :
  (profit price_reduction = 200) ↔ (price_reduction = 0.3) :=
by
  sorry

theorem maximize_daily_profit (price_reduction : ℝ) :
  ∀ price_reduction, (∀ other_reduction, profit other_reduction ≤ profit price_reduction) ↔ (price_reduction = 0.25) :=
by
  sorry

end achieve_daily_profit_of_200_maximize_daily_profit_l821_821740


namespace find_ABC_sum_l821_821877

/- Conditions -/
variables {A B C : ℤ}
def polynomial := λ x : ℝ, x^3 + A * x^2 + B * x + C

/- Main Statement -/
theorem find_ABC_sum (h_asymptotes : (polynomial (-1) = 0) ∧ (polynomial 0 = 0) ∧ (polynomial 3 = 0)) : 
  A + B + C = -5 := 
sorry

end find_ABC_sum_l821_821877


namespace parallel_vectors_l821_821483

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821483


namespace number_of_positive_integers_l821_821367

theorem number_of_positive_integers (x : ℕ) :
  (40 < x^2 + 4 * x + 4 ∧ x^2 + 4 * x + 4 < 100) ↔ (x = 5 ∨ x = 6 ∨ x = 7) := 
by sorry

example : {x : ℕ | 40 < x^2 + 4 * x + 4 ∧ x^2 + 4 * x + 4 < 100}.to_finset.card = 3 :=
by sorry

end number_of_positive_integers_l821_821367


namespace woodworker_extra_parts_l821_821741

theorem woodworker_extra_parts :
  ∃ (P P_new Total_new Extra : ℕ), 
      (P * 24 = 360) ∧ 
      (P_new = P + 5) ∧ 
      (Total_new = P_new * 22) ∧ 
      (Extra = Total_new - 360) ∧ 
      (Extra = 80) :=
by 
  -- Definitions based on conditions
  let P := 360 / 24
  have : P * 24 = 360, by sorry
  
  let P_new := P + 5
  have : P_new = P + 5, by sorry
  
  let Total_new := P_new * 22
  have : Total_new = P_new * 22, by sorry
  
  let Extra := Total_new - 360
  have : Extra = Total_new - 360, by sorry
  
  have : Extra = 80, by sorry
  use P, P_new, Total_new, Extra
  repeat { assumption }
  sorry

end woodworker_extra_parts_l821_821741


namespace percentage_reduction_is_15_l821_821724

-- Definitions of conditions stated in the problem
def reduced_price_per_kg : ℝ := 24
def extra_kg_obtained : ℝ := 5
def total_cost : ℝ := 800

-- Definition to encapsulate the original price using the conditions
def original_price_per_kg : ℝ := total_cost / ((total_cost / reduced_price_per_kg) - extra_kg_obtained)

-- The percentage reduction in the price formula
def percentage_reduction(price_before : ℝ) (price_after : ℝ) : ℝ :=
  ((price_before - price_after) / price_before) * 100

-- Main statement to prove
theorem percentage_reduction_is_15 :
  percentage_reduction original_price_per_kg reduced_price_per_kg = 15 := by
  sorry

end percentage_reduction_is_15_l821_821724


namespace radius_scientific_notation_l821_821646

theorem radius_scientific_notation :
  696000 = 6.96 * 10^5 :=
sorry

end radius_scientific_notation_l821_821646


namespace total_shaded_area_after_100_iterations_l821_821734

-- Defining the conditions
def initialArea : ℝ := 64
def shadedAreaFirstIteration : ℝ := initialArea / 4
def commonRatio : ℝ := 1 / 4

-- The geometric series sum formula
def totalShadedArea : ℝ := shadedAreaFirstIteration / (1 - commonRatio)

-- Lean 4 statement to prove
theorem total_shaded_area_after_100_iterations : totalShadedArea = 64 / 3 :=
by
-- Proof goes here
sorry

end total_shaded_area_after_100_iterations_l821_821734


namespace chip_exits_from_A2_l821_821929

noncomputable def chip_exit_cell (grid_size : ℕ) (initial_cell : ℕ × ℕ) (move_direction : ℕ × ℕ → ℕ × ℕ) : ℕ × ℕ :=
(1, 2) -- A2; we assume the implementation of function movement follows the solution as described

theorem chip_exits_from_A2 :
  chip_exit_cell 4 (3, 2) move_direction = (1, 2) :=
sorry  -- Proof omitted

end chip_exits_from_A2_l821_821929


namespace plane_PEF_perpendicular_l821_821019

-- Define basic premises
variables {P E F G H : Type*}
variables [geom : geometry P E F G H]

open geom

-- Definitions based on given conditions
def is_perpendicular_to_plane (P E : Type*) : Prop :=
  ∀ (a b : Type*), a ∈ plane E ∧ b ∈ plane E ∧ a ≠ b → ⊥ P (plane E)

def plane_contains_square (E F G H : Type*) : Prop := 
  ∀ (E F G H : Type*), (square E F G H) → (plane (square E F G H))

-- Define the plane PEF
def plane_PEF := plane (point P) (line E F)

-- Define the plane PFG
def plane_PFG := plane (point P) (line F G)

-- Define the plane PEH
def plane_PEAnd := plane (point P) (line E H)

-- Mathematical proof problem reformulated in Lean 4 statement
theorem plane_PEF_perpendicular (P : Type*) (E F G H : Type*) [square E F G H] (h₁ : is_perpendicular_to_plane P (plane E F G H)) : 
  is_perpendicular_to_plane (plane P E F) (plane P H) ∧ 
  is_perpendicular_to_plane (plane P E F) (plane P G) :=
sorry

end plane_PEF_perpendicular_l821_821019


namespace fangfang_coins_l821_821805

def coins_count (total_coins total_amount : ℕ) (num_five_jiao num_one_jiao : ℕ) : Prop :=
  total_coins = num_five_jiao + num_one_jiao ∧
  total_amount = 5 * num_five_jiao + num_one_jiao

theorem fangfang_coins :
  ∃ (num_five_jiao num_one_jiao : ℕ), coins_count 30 86 num_five_jiao num_one_jiao ∧
  num_five_jiao = 14 ∧ num_one_jiao = 16 :=
by
  use 14, 16
  split
  · sorry
  · sorry

end fangfang_coins_l821_821805


namespace convex_polygon_diagonals_l821_821063

theorem convex_polygon_diagonals (n : ℕ) (h1 : n = 23) : 
  let sides := n,
      vertices := sides,
      diags_per_vertex := vertices - 3,
      total_connections := vertices * diags_per_vertex,
      diagonals := total_connections / 2
  in diagonals = 230 :=
by
  have h2 := congr_arg (λ x, x) rfl
  have h3 := congr_arg (λ x, x - 3) h2
  have h4 := congr_arg (λ x, x * 20) h2
  have h5 := congr_arg (λ x, x / 2) h4
  sorry

end convex_polygon_diagonals_l821_821063


namespace seating_arrangements_l821_821955

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821955


namespace mady_balls_sum_of_digits_2010_l821_821152

def senary_sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 6 n).sum

theorem mady_balls_sum_of_digits_2010 :
  senary_sum_of_digits 2010 = 11 :=
by
  -- The proof is omitted as requested.
  sorry

end mady_balls_sum_of_digits_2010_l821_821152


namespace product_of_four_consecutive_integers_not_necessarily_divisible_by_24_l821_821198

theorem product_of_four_consecutive_integers_not_necessarily_divisible_by_24 (j : ℤ) :
  let m := j * (j + 1) * (j + 2) * (j + 3) in 11 ∣ m → ¬(24 ∣ m) :=
by
  intro m
  intro h_div_11
  have h_not_div_24 : ¬(24 ∣ m) := sorry
  exact h_not_div_24

end product_of_four_consecutive_integers_not_necessarily_divisible_by_24_l821_821198


namespace bounds_for_k_l821_821832

theorem bounds_for_k (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxz : x * y * z ≤ 2) (h_sum : 1 / x^2 + 1 / y^2 + 1 / z^2 < k) (hk : 2 ≤ k) : 
  k ∈ set.Icc (2 : ℝ) (9 / 4 : ℝ) := 
sorry

end bounds_for_k_l821_821832


namespace cost_of_rice_l821_821520

-- Define the cost variables
variables (E R K : ℝ)

-- State the conditions as assumptions
def conditions (E R K : ℝ) : Prop :=
  (E = R) ∧
  (K = (2 / 3) * E) ∧
  (2 * K = 48)

-- State the theorem to be proven
theorem cost_of_rice (E R K : ℝ) (h : conditions E R K) : R = 36 :=
by
  sorry

end cost_of_rice_l821_821520


namespace log2_f3_eq_sixteen_l821_821075

def f (x : ℝ) : ℝ := 1 + (8.choose 1) * x + (8.choose 2) * x^2 + (8.choose 3) * x^3 + 
                (8.choose 4) * x^4 + (8.choose 5) * x^5 + (8.choose 6) * x^6 + 
                (8.choose 7) * x^7 + (8.choose 8) * x^8

theorem log2_f3_eq_sixteen : ∀ x : ℝ, log2 (f 3) = 16 := by
  sorry

end log2_f3_eq_sixteen_l821_821075


namespace number_of_votes_in_favor_of_candidate_A_l821_821093

-- Definition for total votes and percentages
def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 0.15
def valid_percentage : ℚ := 1 - invalid_percentage
def candidate_A_percentage : ℚ := 0.70

-- Total valid votes
def total_valid_votes : ℕ := (total_votes * valid_percentage).toInt

-- Prove the number of valid votes polled in favor of candidate A
theorem number_of_votes_in_favor_of_candidate_A : 
  (total_valid_votes * candidate_A_percentage).toInt = 333200 :=
by 
  sorry

end number_of_votes_in_favor_of_candidate_A_l821_821093


namespace max_value_of_expression_l821_821356

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 := 
sorry

end max_value_of_expression_l821_821356


namespace mix_liquids_l821_821927

theorem mix_liquids (n : ℕ) (h : n > 0) :
  ∃ steps : ℕ, ∀ (cyl : Fin n → Fin (n + 1) → ℝ), 
  (∀ i : Fin n, cyl i 0 = 1) → 
  (∀ j : Fin (n + 1), cyl 0 j = 0) →
  (∀ i : Fin n, ∑ k, cyl i k = 1) → 
  (∃ F : Fin n → Fin n → Fin (n + 1) → ℝ, 
    (∀ i j : Fin n, ∀ k : Fin (n + 1), cylinder i j = 1 / n)) :=
sorry

end mix_liquids_l821_821927


namespace probability_of_green_tile_l821_821287

theorem probability_of_green_tile :
  let total_tiles := 100
  let green_tiles := 14
  let probability := green_tiles / total_tiles
  probability = 7 / 50 :=
by
  sorry

end probability_of_green_tile_l821_821287


namespace no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l821_821783

theorem no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5 :
  ¬ ∃ n : ℕ, (∀ d ∈ (Nat.digits 10 n), 5 < d) ∧ (∀ d ∈ (Nat.digits 10 (n^2)), d < 5) :=
by
  sorry

end no_positive_integer_with_digits_greater_than_5_and_square_digits_less_than_5_l821_821783


namespace parallel_vectors_l821_821480

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821480


namespace find_ln_a_l821_821129

noncomputable def integral_ln_ax : ℝ → ℝ := sorry
noncomputable def integral_x : ℝ := sorry
noncomputable def integral_ln_ax_over_x : ℝ → ℝ := sorry

theorem find_ln_a (a : ℝ) (h : a > 0) : 
    (integral_ln_ax a) / integral_x = integral_ln_ax_over_x a → 
    ln a = - (exp 2 – 5) / (2 * (exp 2 - 2 * exp 1 + 2)) := 
  sorry

end find_ln_a_l821_821129


namespace seating_arrangements_l821_821960

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821960


namespace cubic_box_edge_length_l821_821062

theorem cubic_box_edge_length (N : ℕ) (V_cube : ℝ) (V_box : ℝ) (L_cm L_m : ℝ)
  (hN : N = 1000)
  (hV_cube : V_cube = 1000)
  (hV_box : V_box = V_cube * N)
  (hL_cm : L_cm ^ 3 = V_box)
  (hL_m_convert : L_m = L_cm / 100) :
  L_m = 1 :=
by
  have h1 : V_box = 1_000_000 := by sorry
  have h2 : L_cm = 100 := by sorry
  have h3 : L_m = 1 := by sorry
  exact h3

end cubic_box_edge_length_l821_821062


namespace parallel_vectors_implies_value_of_λ_l821_821429

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821429


namespace cannot_reach_eighth_vertex_l821_821102

def Point := ℕ × ℕ × ℕ

def symmetry (p1 p2 : Point) : Point :=
  let (a, b, c) := p1
  let (a', b', c') := p2
  (2 * a' - a, 2 * b' - b, 2 * c' - c)

def vertices : List Point :=
  [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]

theorem cannot_reach_eighth_vertex : ∀ (p : Point), p ∈ vertices → ∀ (q : Point), q ∈ vertices → 
  ¬(symmetry p q = (1, 1, 1)) :=
by
  sorry

end cannot_reach_eighth_vertex_l821_821102


namespace correct_sampling_methods_l821_821044

/-- We have a community with specified varieties of income levels -
 - 100 high-income families, 380 middle-income families,
 - and 120 low-income families out of a total of 600 families. 
 - We need to select a sample of 100 households to 
 - understand an indicator related to their purchasing 
 - power for family cars.
 - We also need to draw 3 students from a group of 15 to participate in a seminar.
 - We aim to identify the correct sampling methods for these problems from:
 - I. Simple Random Sampling
 - II. Systematic Sampling
 - III. Stratified Sampling
 - The proof goal is to establish that the correct methods are:
 - Problem (1) should use III (Stratified),
 - Problem (2) should use I (Simple Random).
-/
theorem correct_sampling_methods :
  let families := 600
  let high_income := 100
  let middle_income := 380
  let low_income := 120
  let sample_size_houses := 100
  let total_students := 15
  let sample_size_students := 3
  problem_one_sampling := 3
  problem_two_sampling := 1
  in problem_one_sampling = 3 ∧ problem_two_sampling = 1 :=
by
  sorry

end correct_sampling_methods_l821_821044


namespace max_value_of_sum_l821_821998

-- Define the arithmetic sequence and its sum
def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def S13 := 26
def S14 := -14

theorem max_value_of_sum (a1 d : ℝ) :
  sum_arithmetic_sequence a1 d 13 = S13 →
  sum_arithmetic_sequence a1 d 14 = S14 →
  ∃ n, n = 7 ∧ ∀ m > n, sum_arithmetic_sequence a1 d m < sum_arithmetic_sequence a1 d n :=
by
  sorry

end max_value_of_sum_l821_821998


namespace probability_of_drawing_2_red_1_white_l821_821085

def total_balls : ℕ := 7
def red_balls : ℕ := 4
def white_balls : ℕ := 3
def draws : ℕ := 3

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_drawing_2_red_1_white :
  (combinations red_balls 2) * (combinations white_balls 1) / (combinations total_balls draws) = 18 / 35 := by
  sorry

end probability_of_drawing_2_red_1_white_l821_821085


namespace min_value_of_inequality_l821_821032

theorem min_value_of_inequality (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h1 : ∀ x y : ℝ, (x - y - 1 ≤ 0) → (2x - y - 3 ≥ 0) → a * x + b * y = 3)
: 2 / a + 1 / b = 3 :=
sorry

end min_value_of_inequality_l821_821032


namespace kelsey_speed_l821_821991

theorem kelsey_speed 
  (v : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ = total_distance / 2)
  (second_half_distance : ℝ = total_distance / 2)
  (time1 : ℝ)
  (time2 : ℝ) :
  total_time = 10 ∧ second_half_speed = 40 ∧ total_distance = 400 ∧
  second_half_distance = 200 ∧ 
  time2 = second_half_distance / second_half_speed ∧ 
  time2 = 5 ∧ 
  time1 = total_time - time2 → 
  time1 = 5 → 
  v = first_half_distance / time1 →
  v = 40 := 
  sorry

end kelsey_speed_l821_821991


namespace center_of_circle_l821_821293

noncomputable def center_is_correct (x y : ℚ) : Prop :=
  (5 * x - 2 * y = -10) ∧ (3 * x + y = 0)

theorem center_of_circle : center_is_correct (-10 / 11) (30 / 11) :=
by
  sorry

end center_of_circle_l821_821293


namespace line_equation_l821_821912

/-
Given points M(2, 3) and N(4, -5), and a line l passes through the 
point P(1, 2). Prove that the line l has equal distances from points 
M and N if and only if its equation is either 4x + y - 6 = 0 or 
3x + 2y - 7 = 0.
-/

theorem line_equation (M N P : ℝ × ℝ)
(hM : M = (2, 3))
(hN : N = (4, -5))
(hP : P = (1, 2))
(l : ℝ → ℝ → Prop)
(h_l : ∀ x y, l x y ↔ (4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0))
: ∀ (dM dN : ℝ), 
(∀ x y , l x y → (x = 1) → (y = 2) ∧ (|M.1 - x| + |M.2 - y| = |N.1 - x| + |N.2 - y|)) :=
sorry

end line_equation_l821_821912


namespace smallest_n_in_T_and_largest_N_not_in_T_l821_821133

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3 * x + 4) / (x + 3)}

theorem smallest_n_in_T_and_largest_N_not_in_T :
  (∀ n, n = 4 / 3 → n ∈ T) ∧ (∀ N, N = 3 → N ∉ T) :=
by
  sorry

end smallest_n_in_T_and_largest_N_not_in_T_l821_821133


namespace Phillip_correct_total_l821_821167

def number_questions_math : ℕ := 40
def number_questions_english : ℕ := 50
def percentage_correct_math : ℚ := 0.75
def percentage_correct_english : ℚ := 0.98

noncomputable def total_correct_answers : ℚ :=
  (number_questions_math * percentage_correct_math) + (number_questions_english * percentage_correct_english)

theorem Phillip_correct_total : total_correct_answers = 79 := by
  sorry

end Phillip_correct_total_l821_821167


namespace lambda_parallel_l821_821451

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821451


namespace vacation_savings_l821_821619

-- Definitions
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500
def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Prove the amount set aside for vacation
theorem vacation_savings :
  let 
    total_income := parents_salary + grandmothers_pension + sons_scholarship,
    total_expenses := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses,
    surplus := total_income - total_expenses,
    deposit := (10 * surplus) / 100, 
    vacation_money := surplus - deposit
  in
    vacation_money = 16740 := by
      -- Calculation steps skipped; proof not required
      sorry

end vacation_savings_l821_821619


namespace time_spent_adding_milk_l821_821242

noncomputable def initial_milk : ℕ := 30000
noncomputable def pump_rate : ℕ := 2880
noncomputable def pump_time : ℕ := 4
noncomputable def added_rate : ℕ := 1500
noncomputable def final_milk : ℕ := 28980

theorem time_spent_adding_milk (t : ℕ) :
  t = 7 :=
  let pumped_out := pump_rate * pump_time in
  let remaining_after_pump := initial_milk - pumped_out in
  let total_added := final_milk - remaining_after_pump in
  t = total_added / added_rate :=
sorry

end time_spent_adding_milk_l821_821242


namespace totalCorrectQuestions_l821_821168

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l821_821168


namespace Force_Inversely_Proportional_l821_821208

theorem Force_Inversely_Proportional
  (L₁ F₁ L₂ F₂ : ℝ)
  (h₁ : L₁ = 12)
  (h₂ : F₁ = 480)
  (h₃ : L₂ = 18)
  (h_inv : F₁ * L₁ = F₂ * L₂) :
  F₂ = 320 :=
by
  sorry

end Force_Inversely_Proportional_l821_821208


namespace jason_total_earnings_l821_821981

variables (after_school_rate saturday_rate : ℝ)
variables (total_hours saturday_hours : ℝ)

theorem jason_total_earnings :
  after_school_rate = 4 ∧ saturday_rate = 6 ∧ total_hours = 18 ∧ saturday_hours = 8 → 
  let after_school_hours := total_hours - saturday_hours in
  let after_school_earnings := after_school_hours * after_school_rate in
  let saturday_earnings := saturday_hours * saturday_rate in
  let total_earnings := after_school_earnings + saturday_earnings in
  total_earnings = 88 :=
by
  intros
  sorry

end jason_total_earnings_l821_821981


namespace variance_is_correct_l821_821081

-- Define the sum of squares (S) and the mean (mean_x)
variables (S : ℝ) (mean_x : ℝ)

-- State the conditions given: sum of squares is 56 and mean is sqrt(2)/2
axiom sum_of_squares_condition : S = 56
axiom mean_condition : mean_x = (Real.sqrt 2) / 2

-- State to prove that the variance is 9/10
theorem variance_is_correct (n : ℕ) (S mean_x : ℝ) : 
  n = 40 → S = 56 → mean_x = (Real.sqrt 2) / 2 → 
  (S / n - mean_x^2) = 9 / 10 :=
by
  intros hn hs hm sorry

end variance_is_correct_l821_821081


namespace litter_patrol_total_l821_821200

theorem litter_patrol_total (glass_bottles : Nat) (aluminum_cans : Nat) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 :=
by
  sorry

end litter_patrol_total_l821_821200


namespace Ana_wins_l821_821747

def sequence_valid (s : List ℤ) : Prop :=
  s.length = 2016 ∧ s.count 1 = 1008 ∧ s.count (-1) = 1008

def can_win (s : List ℤ) (N : ℕ) : Prop :=
  ∃ (blocks : List (List ℤ)), (blocks.bind id = s) ∧
                              (blocks.map (λ block, (block.sum) ^ 2)).sum = N

theorem Ana_wins (N : ℕ) (s : List ℤ) (hseq: sequence_valid s) : 
  (N % 2 = 0 ∧ N ≤ 2016) → can_win s N :=
by
  sorry

end Ana_wins_l821_821747


namespace trig_inequalities_l821_821015

theorem trig_inequalities :
  let a := Real.cos 1
  let b := Real.sin 1
  let c := Real.tan 1
  in a < b ∧ b < c :=
by
  let a := Real.cos 1
  let b := Real.sin 1
  let c := Real.tan 1
  have h1 : a = Real.cos 1 := rfl
  have h2 : b = Real.sin 1 := rfl
  have h3 : c = Real.tan 1 := rfl
  -- Proof goes here
  sorry

end trig_inequalities_l821_821015


namespace solve_inequality_l821_821779

theorem solve_inequality :
  {x : ℝ // - x^2 - 2 * x + 3 > 0} = {x : ℝ // x ∈ set.Ioo (-3 : ℝ) (1 : ℝ)} :=
by sorry

end solve_inequality_l821_821779


namespace number_of_integral_points_l821_821899

-- Define the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def on_curve (x y : ℕ) : Prop := y = x^2

def in_range (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 6

-- State the theorem.
theorem number_of_integral_points :
  {p : ℕ × ℕ | in_range p.1 ∧ is_odd p.1 ∧ on_curve p.1 p.2 ∧ is_multiple_of_three p.2}.to_finset.card = 1 :=
by sorry

end number_of_integral_points_l821_821899


namespace non_adjacent_arrangements_l821_821967

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821967


namespace tamara_pans_brownies_l821_821326

theorem tamara_pans_brownies (total_income : ℕ) (price_per_brownie : ℕ) (pieces_per_pan : ℕ) 
  (total_brownies : ℕ) (total_pans : ℕ) 
  (h1 : total_income = 32) (h2 : price_per_brownie = 2) 
  (h3 : pieces_per_pan = 8) (h4 : total_brownies = total_income / price_per_brownie) 
  (h5 : total_pans = total_brownies / pieces_per_pan): 
  total_pans = 2 :=
by {
  rw [h1, h2, h3] at *,
  rw [nat.div_eq 32 2] at h4,
  rw [nat.div_eq 16 8] at h5,
  assumption
}

end tamara_pans_brownies_l821_821326


namespace parallel_vectors_lambda_l821_821474

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821474


namespace sum_polynomial_coefficients_l821_821070

theorem sum_polynomial_coefficients :
  let a := 1
  let a_sum := -2
  (2009 * a + a_sum) = 2007 :=
by
  sorry

end sum_polynomial_coefficients_l821_821070


namespace tan_pi_minus_a_l821_821502

theorem tan_pi_minus_a (a : ℝ) (h1 : sin (a - π/2) = 4/5) (h2 : π/2 < a ∧ a < π) : tan (π - a) = 3/4 :=
sorry

end tan_pi_minus_a_l821_821502


namespace factorization_of_x_squared_minus_64_l821_821802

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821802


namespace existence_of_large_meeting_l821_821658

noncomputable def boys := 2007
noncomputable def girls := 2007
noncomputable def max_meetings_each := 100

theorem existence_of_large_meeting 
  (B : Finset ℕ) (G : Finset ℕ) (M : Finset (Finset ℕ))
  (hB : B.card = boys) (hG : G.card = girls)
  (hS : ∀ s ∈ B ∪ G, (s ∈ B → s.attendings.card ≤ max_meetings_each) ∧ (s ∈ G → s.attendings.card ≤ max_meetings_each))
  (hAtt : ∀ b ∈ B, ∀ g ∈ G, ∃ m ∈ M, b ∈ m ∧ g ∈ m) :
  ∃ m ∈ M, (∃ bset ⊆ B, bset.card ≥ 11 ∧ ∀ b ∈ bset, b ∈ m) ∧
            (∃ gset ⊆ G, gset.card ≥ 11 ∧ ∀ g ∈ gset, g ∈ m) :=
  sorry

end existence_of_large_meeting_l821_821658


namespace find_angle_l821_821811

theorem find_angle (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by 
  sorry

end find_angle_l821_821811


namespace simplest_square_root_l821_821262

theorem simplest_square_root (a b c d : ℝ) (h1 : a = 3) (h2 : b = 2 * Real.sqrt 3) (h3 : c = (Real.sqrt 2) / 2) (h4 : d = Real.sqrt 10) :
  d = Real.sqrt 10 ∧ (a ≠ Real.sqrt 10) ∧ (b ≠ Real.sqrt 10) ∧ (c ≠ Real.sqrt 10) := 
by 
  sorry

end simplest_square_root_l821_821262


namespace sum_of_first_24_terms_l821_821854

noncomputable def a_sequence : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| (n+3) := a_sequence n + 2

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a_sequence i

theorem sum_of_first_24_terms :
  S 24 = 216 :=
by sorry

end sum_of_first_24_terms_l821_821854


namespace company_bought_oil_l821_821203

-- Define the conditions
def tank_capacity : ℕ := 32
def oil_in_tank : ℕ := 24

-- Formulate the proof problem
theorem company_bought_oil : oil_in_tank = 24 := by
  sorry

end company_bought_oil_l821_821203


namespace parallel_vectors_lambda_value_l821_821461

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821461


namespace sum_f_eq_10_l821_821003

def f (n : ℕ) : ℝ := if ∃ k : ℤ, n = 5^k then real.log n / real.log 5 else 0

theorem sum_f_eq_10 : (∑ n in finset.range (1997 + 1), f n) = 10 := 
sorry

end sum_f_eq_10_l821_821003


namespace six_digit_number_count_correct_l821_821250

-- Defining the 6-digit number formation problem
def count_six_digit_numbers_with_conditions : Nat := 1560

-- Problem statement
theorem six_digit_number_count_correct :
  count_six_digit_numbers_with_conditions = 1560 :=
sorry

end six_digit_number_count_correct_l821_821250


namespace sum_of_squares_of_roots_l821_821780

theorem sum_of_squares_of_roots :
  let a := 1
  let b := 8
  let c := -12
  let r1_r2_sum := -(b:ℝ) / a
  let r1_r2_product := (c:ℝ) / a
  (r1_r2_sum) ^ 2 - 2 * r1_r2_product = 88 :=
by
  sorry

end sum_of_squares_of_roots_l821_821780


namespace parallel_vectors_l821_821482

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821482


namespace pioneer_prep_school_pass_l821_821752

theorem pioneer_prep_school_pass (total_problems : ℕ) (pass_percentage : ℕ) (min_score : ℕ) :
  total_problems = 50 → pass_percentage = 75 → min_score = 75 → 
  let max_missed := total_problems - (pass_percentage * total_problems / 100)
  in max_missed = 12 :=
by intros htotal hpass hscore
   have h1 : max_missed = total_problems - (pass_percentage * total_problems / 100), sorry
   have h2 : total_problems = 50, from htotal, sorry
   have h3 : pass_percentage * total_problems / 100 = 37.5, sorry
   have h4 : 50 - 37.5 = 12.5, sorry
   have h5 : 12.5.round = 12, sorry
   show max_missed = 12

end pioneer_prep_school_pass_l821_821752


namespace derivative_of_reciprocal_at_one_l821_821628

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal_at_one : (deriv f 1) = -1 :=
by {
    sorry
}

end derivative_of_reciprocal_at_one_l821_821628


namespace problem_statement_l821_821999

open Real

theorem problem_statement
  (n : ℕ) (hn : n ≥ 2)
  (a : ℕ → ℝ) (h₀ : ∀ i, 0 ≤ a i ∧ a i ≤ π / 2) :
  (1 / n * ∑ i in Finset.range n, 1 / (1 + sin (a i))) *
  (1 + ∏ i in Finset.range n, (sin (a i)) ^ (1 / n) ) ≤ 1 :=
sorry

end problem_statement_l821_821999


namespace average_salary_of_all_workers_l821_821615

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l821_821615


namespace ratio_of_volumes_l821_821742

variable (π : Real)

def volume_cylinder (r h : ℝ) : ℝ :=
  π * (r^2) * h

def alex_cans_total_volume : ℝ :=
  3 * volume_cylinder 4 10

def felicia_cans_total_volume : ℝ :=
  2 * volume_cylinder 5 8

theorem ratio_of_volumes : 
  alex_cans_total_volume π / felicia_cans_total_volume π = 6 / 5 :=
  sorry

end ratio_of_volumes_l821_821742


namespace prob_exact_heads_l821_821121

-- Definitions based on conditions
def num_flips : ℕ := 8
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def num_heads : ℕ := 3

theorem prob_exact_heads :
  -- Probability that Jenny flips exactly 3 heads in 8 flips is 1792/6561
  (nat.choose num_flips num_heads * (prob_heads ^ num_heads) * (prob_tails ^ (num_flips - num_heads))) = 1792 / 6561 :=
sorry

end prob_exact_heads_l821_821121


namespace volume_of_sphere_above_l821_821035

noncomputable def volume_of_sphere {R : ℝ} (R_pos : 0 < R) : ℝ :=
  4 / 3 * Real.pi * R^3

theorem volume_of_sphere_above {a R : ℝ} 
  (h_vertex_on_sphere : ∀ v ∈ set.univ, ∥v∥ = R)
  (h_surface_area : 6 * a^2 = 18)
  (h_cube_in_sphere : (Real.sqrt 3) * a = 2 * R) :
  volume_of_sphere R  (by linarith) =  9 * Real.pi / 2 := by 
  sorry

end volume_of_sphere_above_l821_821035


namespace seating_arrangements_l821_821959

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821959


namespace sally_shots_l821_821316

theorem sally_shots (made20 : ℕ) (percent20 : ℝ) (total25 : ℕ) (percent25 : ℝ) :
  made20 = 11 →
  percent20 = 0.55 →
  total25 = 14 →
  percent25 = 0.56 →
  ∃ (made_last5 : ℕ), made_last5 = 3 :=
by
  intros h1 h2 h3 h4
  use 3
  sorry

end sally_shots_l821_821316


namespace problem_PQR_centroid_and_perimeter_l821_821247

def Point := (ℝ × ℝ)

def triangle (P Q R: Point) : Prop :=
  sorry -- definition of a triangle goes here

def centroid (P Q R: Point) : Point :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
    ((x1 + x2 + x3)/3, (y1 + y2 + y3)/3)

def distance (A B : Point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  real.sqrt ((x2-x1)^2 + (y2-y1)^2)

def perimeter (A B C : Point) : ℝ :=
  distance A B + distance B C + distance C A

theorem problem_PQR_centroid_and_perimeter :
  let P : Point := (8, 10)
  let Q : Point := (4, 0)
  let R : Point := (9, 4)
  let S := centroid P Q R
  (10 * S.1 + S.2 = 224 / 3) ∧ (perimeter P Q R = real.sqrt 116 + real.sqrt 37 + real.sqrt 41) :=
by
  sorry

end problem_PQR_centroid_and_perimeter_l821_821247


namespace line_through_origin_tangent_lines_line_through_tangents_l821_821393

section GeomProblem

variables {A : ℝ × ℝ} {C : ℝ × ℝ → Prop}

def is_circle (C : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∀ (P : ℝ × ℝ), C P ↔ (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

theorem line_through_origin (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ m : ℝ, ∀ P : ℝ × ℝ, C P → abs ((m * P.1 - P.2) / Real.sqrt (m ^ 2 + 1)) = 1)
    ↔ m = 0 :=
sorry

theorem tangent_lines (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P : ℝ × ℝ, C P → (P.2 - 2 * Real.sqrt 3) = k * (P.1 - 1))
    ↔ (∀ P : ℝ × ℝ, C P → (Real.sqrt 3 * P.1 - 3 * P.2 + 5 * Real.sqrt 3 = 0 ∨ P.1 = 1)) :=
sorry

theorem line_through_tangents (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P D E : ℝ × ℝ, C P → (Real.sqrt 3 * D.1 - 3 * D.2 + 5 * Real.sqrt 3 = 0 ∧
                                      (E.1 - 1 = 0 ∨ Real.sqrt 3 * E.1 - 3 * E.2 + 5 * Real.sqrt 3 = 0)) →
    (D.1 + Real.sqrt 3 * D.2 - 1 = 0 ∧ E.1 + Real.sqrt 3 * E.2 - 1 = 0)) :=
sorry

end GeomProblem

end line_through_origin_tangent_lines_line_through_tangents_l821_821393


namespace count_pos_even_mult_3_perfect_squares_l821_821902

theorem count_pos_even_mult_3_perfect_squares :
  {k : ℕ | 36 * k^2 < 3000}.card = 9 := 
sorry

end count_pos_even_mult_3_perfect_squares_l821_821902


namespace johns_change_l821_821984

/-- Define the cost of Slurpees and amount given -/
def cost_per_slurpee : ℕ := 2
def amount_given : ℕ := 20
def slurpees_bought : ℕ := 6

/-- Define the total cost of the Slurpees -/
def total_cost : ℕ := cost_per_slurpee * slurpees_bought

/-- Define the change John gets -/
def change (amount_given total_cost : ℕ) : ℕ := amount_given - total_cost

/-- The statement for Lean 4 that proves the change John gets is $8 given the conditions -/
theorem johns_change : change amount_given total_cost = 8 :=
by 
  -- Rest of the proof omitted
  sorry

end johns_change_l821_821984


namespace problem_1_problem_2_l821_821406

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem problem_1 (x : ℝ) : (g x ≥ abs (x - 1)) ↔ (x ≥ 2/3) :=
by
  sorry

theorem problem_2 (c : ℝ) : (∀ x, abs (g x) - c ≥ abs (x - 1)) → (c ≤ -1/2) :=
by
  sorry

end problem_1_problem_2_l821_821406


namespace isosceles_triangle_perimeter_l821_821515

-- Definitions of the side lengths
def side1 : ℝ := 8
def side2 : ℝ := 4

-- Theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter (side1 side2 : ℝ) (h1 : side1 = 8 ∨ side2 = 8) (h2 : side1 = 4 ∨ side2 = 4) : ∃ p : ℝ, p = 20 :=
by
  -- We omit the proof using sorry
  sorry

end isosceles_triangle_perimeter_l821_821515


namespace number_of_arrangements_l821_821950

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821950


namespace max_value_64_l821_821087

-- Define the types and values of gemstones
structure Gemstone where
  weight : ℕ
  value : ℕ

-- Introduction of the three types of gemstones
def gem1 : Gemstone := ⟨3, 9⟩
def gem2 : Gemstone := ⟨5, 16⟩
def gem3 : Gemstone := ⟨2, 5⟩

-- Maximum weight Janet can carry
def max_weight := 20

-- Problem statement: Proving the maximum value Janet can carry is $64
theorem max_value_64 (n1 n2 n3 : ℕ) (h1 : n1 ≥ 15) (h2 : n2 ≥ 15) (h3 : n3 ≥ 15) 
  (weight_limit : n1 * gem1.weight + n2 * gem2.weight + n3 * gem3.weight ≤ max_weight) : 
  n1 * gem1.value + n2 * gem2.value + n3 * gem3.value ≤ 64 :=
sorry

end max_value_64_l821_821087


namespace no_real_b_for_line_to_vertex_of_parabola_l821_821823

theorem no_real_b_for_line_to_vertex_of_parabola : 
  ¬ ∃ b : ℝ, ∃ x : ℝ, y = x + b ∧ y = x^2 + b^2 + 1 :=
by
  sorry

end no_real_b_for_line_to_vertex_of_parabola_l821_821823


namespace find_A_l821_821883

def U : Set ℕ := {1, 2, 3, 4, 5}

def compl_U (A : Set ℕ) : Set ℕ := U \ A

theorem find_A (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (h_compl_U : compl_U A = {2, 3}) : A = {1, 4, 5} :=
by
  sorry

end find_A_l821_821883


namespace parallel_vectors_lambda_l821_821469

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821469


namespace factorization_of_x_squared_minus_64_l821_821803

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821803


namespace JuanitaDessertCost_l821_821301

-- Define costs as constants
def brownieCost : ℝ := 2.50
def regularScoopCost : ℝ := 1.00
def premiumScoopCost : ℝ := 1.25
def deluxeScoopCost : ℝ := 1.50
def syrupCost : ℝ := 0.50
def nutsCost : ℝ := 1.50
def whippedCreamCost : ℝ := 0.75
def cherryCost : ℝ := 0.25

-- Define the total cost calculation
def totalCost : ℝ := brownieCost + regularScoopCost + premiumScoopCost +
                     deluxeScoopCost + syrupCost + syrupCost + nutsCost + whippedCreamCost + cherryCost

-- The proof problem: Prove that total cost equals $9.75
theorem JuanitaDessertCost : totalCost = 9.75 :=
by
  -- Proof is omitted
  sorry

end JuanitaDessertCost_l821_821301


namespace geometric_problem_l821_821837

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
variables (eccentricity : ℝ) (e_pos : eccentricity = 1 / 2)
variables (line_l : ℝ → ℝ → Prop) (line_l_def : ∀ x y, line_l x y ↔ x + 2 * y = 4)
variables (ellipse_C : ℝ → ℝ → Prop) (ellipse_C_def : ∀ x y, ellipse_C x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1)
variables (O : ℝ × ℝ) (O_def : O = (0, 0))
variables (T : ℝ × ℝ) (T_def : ∃ x y, line_l x y ∧ ellipse_C x y)
variables (l' : ℝ → ℝ → Prop) (l'_def : ∃ (l' : ℝ → ℝ → Prop), ∀ x y, l' x y ↔ y = (3 / 2) * x + sqrt 6 ∨ y = (3 / 2) * x - sqrt 6)

theorem geometric_problem :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ (1 / 2 : ℝ) = sqrt (1 - b^2 / a^2) ∧
    (∀ x y : ℝ, ellipse_C_def x y ↔ (x^2 / 4) + (y^2 / 3) = 1) ∧
    (∃ x y, line_l_def x y ∧ ellipse_C_def x y ∧ x = 1 ∧ y = 3 / 2) ∧
    T = (1, 1.5) ∧
    (∃ x y, l'_def x y ∧ (l' x y ↔ ∀ x y, y = (3 / 2) * x + sqrt 6 ∨ y = (3 / 2) * x - sqrt 6))) :=
begin
  sorry
end

end geometric_problem_l821_821837


namespace angle_sum_problem_l821_821108

theorem angle_sum_problem 
  (BAC : ℕ) (ADC : ℕ) (CDE : ℕ)
  (h1 : BAC = 40)
  (h2 : ADC = 180)
  (h3 : CDE = 42) :
  ∃ x, x = 178 :=
by
  use 178
  sorry

end angle_sum_problem_l821_821108


namespace tan_alpha_is_five_cos_product_value_l821_821379

variables (α : Real) (h : (sin α + cos α) / (sin α - 2 * cos α) = 2)

theorem tan_alpha_is_five : tan α = 5 := 
by
  sorry

theorem cos_product_value :
  cos (π / 2 - α) * cos (-π + α) = -5 / 26 := 
by
  sorry

end tan_alpha_is_five_cos_product_value_l821_821379


namespace symmetric_points_on_parabola_l821_821026

theorem symmetric_points_on_parabola {a b m n : ℝ}
  (hA : m = a^2 - 2*a - 2)
  (hB : m = b^2 - 2*b - 2)
  (hP : n = (a + b)^2 - 2*(a + b) - 2)
  (h_symmetry : (a + b) / 2 = 1) :
  n = -2 :=
by {
  -- Proof omitted
  sorry
}

end symmetric_points_on_parabola_l821_821026


namespace triangle_obtuse_inequality_l821_821537

variables {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def is_obtuse_angle_C (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
∃ a b c : ℝ, (a^2 + b^2) < c^2

theorem triangle_obtuse_inequality
  {A B C D : ℝ} 
  (h: is_obtuse_angle_C A B C) : 
  CD^2 ≤ AD^2 + BD^2 :=
sorry

end triangle_obtuse_inequality_l821_821537


namespace line_through_point_parallel_l821_821633

theorem line_through_point_parallel (p : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0)
  (hp : a * p.1 + b * p.2 + c = 0) :
  ∃ k : ℝ, a * p.1 + b * p.2 + k = 0 :=
by
  use - (a * p.1 + b * p.2)
  sorry

end line_through_point_parallel_l821_821633


namespace discount_percentage_is_10_l821_821730

variables (wholesalePrice retailPrice profit sellingPrice discount discountPercentage : ℝ)

noncomputable def wholesalePrice := 126
noncomputable def retailPrice := 168
noncomputable def profit := 0.20 * wholesalePrice
noncomputable def sellingPrice := wholesalePrice + profit
noncomputable def discount := retailPrice - sellingPrice
noncomputable def discountPercentage := (discount / retailPrice) * 100

theorem discount_percentage_is_10 :
  discountPercentage = 10 := 
sorry

end discount_percentage_is_10_l821_821730


namespace number_pairs_sum_diff_prod_quotient_l821_821254

theorem number_pairs_sum_diff_prod_quotient (x y : ℤ) (h : x ≥ y) :
  (x + y) + (x - y) + x * y + x / y = 800 ∨ (x + y) + (x - y) + x * y + x / y = 400 :=
sorry

-- Correct answers for A = 800
example : (38 + 19) + (38 - 19) + 38 * 19 + 38 / 19 = 800 := by norm_num
example : (-42 + -21) + (-42 - -21) + (-42 * -21) + (-42 / -21) = 800 := by norm_num
example : (72 + 9) + (72 - 9) + 72 * 9 + 72 / 9 = 800 := by norm_num
example : (-88 + -11) + (-88 - -11) + -(88 * -11) + (-88 / -11) = 800 := by norm_num
example : (128 + 4) + (128 - 4) + 128 * 4 + 128 / 4 = 800 := by norm_num
example : (-192 + -6) + (-192 - -6) + -192 * -6 + ( -192 / -6 ) = 800 := by norm_num
example : (150 + 3) + (150 - 3) + 150 * 3 + 150 / 3 = 800 := by norm_num
example : (-250 + -5) + (-250 - -5) + (-250 * -5) + (-250 / -5) = 800 := by norm_num
example : (200 + 1) + (200 - 1) + 200 * 1 + 200 / 1 = 800 := by norm_num
example : (-600 + -3) + (-600 - -3) + -600 * -3 + -600 / -3 = 800 := by norm_num

-- Correct answers for A = 400
example : (19 + 19) + (19 - 19) + 19 * 19 + 19 / 19 = 400 := by norm_num
example : (-21 + -21) + (-21 - -21) + (-21 * -21) + (-21 / -21) = 400 := by norm_num
example : (36 + 9) + (36 - 9) + 36 * 9 + 36 / 9 = 400 := by norm_num
example : (-44 + -11) + (-44 - -11) + (-44 * -11) + (-44 / -11) = 400 := by norm_num
example : (64 + 4) + (64 - 4) + 64 * 4 + 64 / 4 = 400 := by norm_num
example : (-96 + -6) + (-96 - -6) + (-96 * -6) + (-96 / -6) = 400 := by norm_num
example : (75 + 3) + (75 - 3) + 75 * 3 + 75 / 3 = 400 := by norm_num
example : (-125 + -5) + (-125 - -5) + (-125 * -5) + (-125 / -5) = 400 := by norm_num
example : (100 + 1) + (100 - 1) + 100 * 1 + 100 / 1 = 400 := by norm_num
example : (-300 + -3) + (-300 - -3) + (-300 * -3) + (-300 / -3) = 400 := by norm_num

end number_pairs_sum_diff_prod_quotient_l821_821254


namespace average_salary_of_all_workers_l821_821614

theorem average_salary_of_all_workers :
  let technicians := 7
  let technicians_avg_salary := 20000
  let rest := 49 - technicians
  let rest_avg_salary := 6000
  let total_workers := 49
  let total_tech_salary := technicians * technicians_avg_salary
  let total_rest_salary := rest * rest_avg_salary
  let total_salary := total_tech_salary + total_rest_salary
  (total_salary / total_workers) = 8000 := by
  sorry

end average_salary_of_all_workers_l821_821614


namespace bacteria_population_at_10_20_l821_821123

noncomputable def population_at_10_20 (
  initial_population : ℕ := 50
  initial_time : ℕ := 0
  target_time : ℕ := 20
  doubling_period : ℕ := 4
  net_growth_factor : ℝ := 1.8
  doubling_periods : ℕ := target_time / doubling_period
) : ℕ := 
  let net_growth := net_growth_factor ^ doubling_periods
  in Nat.round (initial_population * net_growth)

theorem bacteria_population_at_10_20 : 
  population_at_10_20 50 0 20 4 1.8 5 = 945 :=
begin
  sorry
end

end bacteria_population_at_10_20_l821_821123


namespace find_f_expression_l821_821074

theorem find_f_expression : 
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f (2 * x - 1) = 3 * x^2 + 1) → 
  f = λ x, (3/4) * x^2 + (3/2) * x + (7/4) :=
by
  sorry

end find_f_expression_l821_821074


namespace symmetric_lines_intersect_at_one_point_l821_821599

-- Definitions and conditions based on the problem
variables {α : Type*} [euclidean_space α]

structure Triangle (α : Type*) [euclidean_space α] :=
(A B C : α)

def orthocenter {α : Type*} [euclidean_space α] (t : Triangle α) : α := sorry

def symmetric_line {α : Type*} [euclidean_space α] (l : line α) (side : line α) : line α := sorry

-- The actual theorem statement
theorem symmetric_lines_intersect_at_one_point
  (t : Triangle α)
  (H : α = orthocenter t)
  (l : line α)
  (L₁ := symmetric_line l (line.of_points t.A t.B))
  (L₂ := symmetric_line l (line.of_points t.B t.C))
  (L₃ := symmetric_line l (line.of_points t.C t.A)) :
  ∃ P : α, P ∈ L₁ ∧ P ∈ L₂ ∧ P ∈ L₃ := sorry

end symmetric_lines_intersect_at_one_point_l821_821599


namespace intersection_points_l821_821665

noncomputable def y1 := 2*((7 + Real.sqrt 61)/2)^2 - 3*((7 + Real.sqrt 61)/2) + 1
noncomputable def y2 := 2*((7 - Real.sqrt 61)/2)^2 - 3*((7 - Real.sqrt 61)/2) + 1

theorem intersection_points :
  ∃ (x y : ℝ), (y = 2*x^2 - 3*x + 1) ∧ (y = x^2 + 4*x + 4) ∧
                ((x = (7 + Real.sqrt 61)/2 ∧ y = y1) ∨
                 (x = (7 - Real.sqrt 61)/2 ∧ y = y2)) :=
by
  sorry

end intersection_points_l821_821665


namespace cost_per_ounce_l821_821282

theorem cost_per_ounce (total_cost : ℕ) (number_of_ounces : ℕ) (h1 : total_cost = 84) (h2 : number_of_ounces = 12) : total_cost / number_of_ounces = 7 :=
by
  rw [h1, h2]
  simp
  sorry

end cost_per_ounce_l821_821282


namespace number_of_arrangements_l821_821946

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821946


namespace mean_score_juniors_is_103_l821_821230

noncomputable def mean_score_juniors : Prop :=
  ∃ (students juniors non_juniors m_j m_nj : ℝ),
  students = 160 ∧
  (students * 82) = (juniors * m_j + non_juniors * m_nj) ∧
  juniors = 0.4 * non_juniors ∧
  m_j = 1.4 * m_nj ∧
  m_j = 103

theorem mean_score_juniors_is_103 : mean_score_juniors :=
by
  sorry

end mean_score_juniors_is_103_l821_821230


namespace max_perimeter_triangle_l821_821930

theorem max_perimeter_triangle 
(O H A B C : Point)
(circle_center : Center O)
(radius : ℝ)
(triangle_ABC : Triangle A B C)
(orthocenter : Orthocenter H triangle_ABC)
(triangle_A'B'C' : Triangle (side_length AB) (side_length CH) (2 * radius))
(∠C : Angle (vertex C))
(hypothesis : ∠C = π / 4)
: perimeter triangle_A'B'C' = (2 * √2 + 2) * radius := 
sorry

end max_perimeter_triangle_l821_821930


namespace parallel_vectors_lambda_value_l821_821466

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821466


namespace tan_alpha_value_l821_821401

theorem tan_alpha_value (α : ℝ) (h1 : sin (π - α) = sqrt 5 / 5) (h2 : π / 2 < α ∧ α < π) : 
  tan α = -1 / 2 :=
sorry

end tan_alpha_value_l821_821401


namespace beyonce_album_songs_l821_821757

theorem beyonce_album_songs
  (singles : ℕ)
  (album1_songs album2_songs album3_songs total_songs : ℕ)
  (h1 : singles = 5)
  (h2 : album1_songs = 15)
  (h3 : album2_songs = 15)
  (h4 : total_songs = 55) :
  album3_songs = 20 :=
by
  sorry

end beyonce_album_songs_l821_821757


namespace average_sleep_per_day_l821_821889

-- Define a structure for time duration
structure TimeDuration where
  hours : ℕ
  minutes : ℕ

-- Define instances for each day
def mondayNight : TimeDuration := ⟨8, 15⟩
def mondayNap : TimeDuration := ⟨0, 30⟩
def tuesdayNight : TimeDuration := ⟨7, 45⟩
def tuesdayNap : TimeDuration := ⟨0, 45⟩
def wednesdayNight : TimeDuration := ⟨8, 10⟩
def wednesdayNap : TimeDuration := ⟨0, 50⟩
def thursdayNight : TimeDuration := ⟨10, 25⟩
def thursdayNap : TimeDuration := ⟨0, 20⟩
def fridayNight : TimeDuration := ⟨7, 50⟩
def fridayNap : TimeDuration := ⟨0, 40⟩

-- Function to convert TimeDuration to total minutes
def totalMinutes (td : TimeDuration) : ℕ :=
  td.hours * 60 + td.minutes

-- Define the total sleep time for each day
def mondayTotal := totalMinutes mondayNight + totalMinutes mondayNap
def tuesdayTotal := totalMinutes tuesdayNight + totalMinutes tuesdayNap
def wednesdayTotal := totalMinutes wednesdayNight + totalMinutes wednesdayNap
def thursdayTotal := totalMinutes thursdayNight + totalMinutes thursdayNap
def fridayTotal := totalMinutes fridayNight + totalMinutes fridayNap

-- Sum of all sleep times
def totalSleep := mondayTotal + tuesdayTotal + wednesdayTotal + thursdayTotal + fridayTotal
-- Average sleep in minutes per day
def averageSleep := totalSleep / 5
-- Convert average sleep in total minutes back to hours and minutes
def averageHours := averageSleep / 60
def averageMinutes := averageSleep % 60

theorem average_sleep_per_day :
  averageHours = 9 ∧ averageMinutes = 6 := by
  sorry

end average_sleep_per_day_l821_821889


namespace gov_addresses_l821_821096

theorem gov_addresses (S H K : ℕ) 
  (H1 : S = 2 * H) 
  (H2 : K = S + 10) 
  (H3 : S + H + K = 40) : 
  S = 12 := 
sorry 

end gov_addresses_l821_821096


namespace relation_among_a_b_c_l821_821397

open Real

theorem relation_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = log 3 / log 2)
  (h2 : b = log 7 / (2 * log 2))
  (h3 : c = 0.7 ^ 4) :
  a > b ∧ b > c :=
by
  -- we leave the proof as an exercise
  sorry

end relation_among_a_b_c_l821_821397


namespace kite_cost_l821_821343

variable (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ)

theorem kite_cost (initial_amount : ℕ) (cost_frisbee : ℕ) (amount_left : ℕ) (h_initial_amount : initial_amount = 78) (h_cost_frisbee : cost_frisbee = 9) (h_amount_left : amount_left = 61) : 
  initial_amount - amount_left - cost_frisbee = 8 :=
by
  -- Proof can be completed here
  sorry

end kite_cost_l821_821343


namespace h_strictly_increasing_exists_x0_l821_821130

noncomputable def f (x : ℝ) := x + (1 / x)

noncomputable def h (x : ℝ) := (x^4) / ((1 - x)^6)

noncomputable def g (x : ℝ) := f (h(x))

theorem h_strictly_increasing :
  ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 → x < y → h(x) < h(y) :=
sorry

theorem exists_x0 :
  ∃ x0 : ℝ, (0 < x0 ∧ x0 < 1) ∧
    (∀ x : ℝ, 0 < x ∧ x ≤ x0 → g x = f (h x) ∧ g (x) < g (x0)) ∧
    (∀ x : ℝ, x0 ≤ x ∧ x < 1 → g x = f (h x) ∧ g (x0) < g (x)) :=
sorry

end h_strictly_increasing_exists_x0_l821_821130


namespace train_speed_is_36_l821_821311

def train_speed (train_length bridge_length time_to_cross : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_meters_per_second := total_distance / time_to_cross
  speed_meters_per_second * 3.6

theorem train_speed_is_36 (train_length bridge_length time_to_cross : ℝ) (h1 : train_length = 100) 
  (h2 : bridge_length = 170) (h3 : time_to_cross = 26.997840172786177) : 
  train_speed train_length bridge_length time_to_cross = 36 :=
by
  -- Proof goes here
  sorry

end train_speed_is_36_l821_821311


namespace find_y_l821_821059

def vector_a := (2, -1, 3)
def vector_b (y : ℝ) := (-4, y, 2)
def vector_sum (y : ℝ) := (2 + vector_b y.1, -1 + vector_b y.2, 3 + vector_b y.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_y (y : ℝ) : dot_product vector_a (vector_sum y) = 0 → y = 12 := 
  by
  sorry

end find_y_l821_821059


namespace math_problem_l821_821139

theorem math_problem (p q : ℕ) (hp : p % 13 = 7) (hq : q % 13 = 7) (hp_lower : 1000 ≤ p) (hp_upper : p < 10000) (hq_lower : 10000 ≤ q) (min_p : ∀ n, n % 13 = 7 → 1000 ≤ n → n < 10000 → p ≤ n) (min_q : ∀ n, n % 13 = 7 → 10000 ≤ n → q ≤ n) : 
  q - p = 8996 := 
sorry

end math_problem_l821_821139


namespace ways_to_select_books_l821_821238

theorem ways_to_select_books (nChinese nMath nEnglish : ℕ) (h1 : nChinese = 9) (h2 : nMath = 7) (h3 : nEnglish = 5) :
  (nChinese * nMath + nChinese * nEnglish + nMath * nEnglish) = 143 :=
by
  sorry

end ways_to_select_books_l821_821238


namespace number_of_permutations_l821_821743

theorem number_of_permutations : 
  ∃ n : ℕ, (n = 4! ∧ ∀ (d1 d2 d3 d4 : ℕ), {d1, d2, d3, d4} = {5, 3, 8, 2} → n = 24) :=
sorry

end number_of_permutations_l821_821743


namespace ab_square_value_l821_821146

noncomputable def cyclic_quadrilateral (AX AY BX BY CX CY AB2 : ℝ) : Prop :=
  AX * AY = 6 ∧
  BX * BY = 5 ∧
  CX * CY = 4 ∧
  AB2 = 122 / 15

theorem ab_square_value :
  ∃ (AX AY BX BY CX CY : ℝ), cyclic_quadrilateral AX AY BX BY CX CY (122 / 15) :=
by
  sorry

end ab_square_value_l821_821146


namespace cos_diff_pentagon_sin_reciprocal_heptagon_sin_sum_angles_l821_821181

-- Part (a)
theorem cos_diff_pentagon : cos (π / 5) - cos (2 * π / 5) = 1 / 2 :=
by
  sorry

-- Part (b)
theorem sin_reciprocal_heptagon : 
  1 / sin (π / 7) = 1 / sin (2 * π / 7) + 1 / sin (3 * π / 7) :=
by
  sorry

-- Part (c)
theorem sin_sum_angles : 
  sin (9 * π / 180) + sin (49 * π / 180) + sin (89 * π / 180) +
  sin (129 * π / 180) + sin (169 * π / 180) + sin (209 * π / 180) + 
  sin (249 * π / 180) + sin (289 * π / 180) + sin (329 * π / 180) = 0 :=
by
  sorry

end cos_diff_pentagon_sin_reciprocal_heptagon_sin_sum_angles_l821_821181


namespace jae_woong_dong_hun_meet_time_l821_821543

theorem jae_woong_dong_hun_meet_time
  (start_at_same_place : True)
  (playground_length_km : ℝ := 3)
  (speed_jaewoong_m_per_min : ℝ := 100)
  (speed_donghun_m_per_min : ℝ := 150) :
  let playground_length_m := playground_length_km * 1000 in
  let combined_speed_m_per_min := speed_jaewoong_m_per_min + speed_donghun_m_per_min in
  let meeting_time_min := playground_length_m / combined_speed_m_per_min in
  meeting_time_min = 12 :=
by
  sorry

end jae_woong_dong_hun_meet_time_l821_821543


namespace car_travel_distance_l821_821290

-- Definitions based on the conditions
def car_speed : ℕ := 60  -- The actual speed of the car
def faster_speed : ℕ := car_speed + 30  -- Speed if the car traveled 30 km/h faster
def time_difference : ℚ := 0.5  -- 30 minutes less in hours

-- The distance D we need to prove
def distance_traveled : ℚ := 90

-- Main statement to be proven
theorem car_travel_distance : ∀ (D : ℚ),
  (D / car_speed) = (D / faster_speed) + time_difference →
  D = distance_traveled :=
by
  intros D h
  sorry

end car_travel_distance_l821_821290


namespace problem_lean_l821_821776

theorem problem_lean (k b : ℤ) : 
  ∃ n : ℤ, n = 25 ∧ n^2 = (k + 1)^4 - k^4 ∧ 3 * n + 100 = b^2 :=
sorry

end problem_lean_l821_821776


namespace solve_for_y_l821_821189

theorem solve_for_y :
  ∃ y : ℚ, 2 * y + 3 * y = 200 - (4 * y + (10 * y / 2)) ∧ y = 100 / 7 :=
by {
  -- Assertion only, proof is not required as per instructions.
  sorry
}

end solve_for_y_l821_821189


namespace factor_difference_of_squares_l821_821787

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821787


namespace monthly_savings_correct_l821_821622

-- Define each component of the income and expenses
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Define the main theorem
theorem monthly_savings_correct :
  let I := parents_salary + grandmothers_pension + sons_scholarship in
  let E := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses in
  let Surplus := I - E in
  let Deposit := (Surplus * 10) / 100 in
  let AmountSetAside := Surplus - Deposit in
  AmountSetAside = 16740 :=
by sorry

end monthly_savings_correct_l821_821622


namespace rosy_fish_count_l821_821574

theorem rosy_fish_count (L R T : ℕ) (hL : L = 10) (hT : T = 19) : R = T - L := by
  sorry

end rosy_fish_count_l821_821574


namespace jerry_can_escape_l821_821982

theorem jerry_can_escape (d : ℝ) (V_J V_T : ℝ) (h1 : (1 / 5) < d) (h2 : d < (1 / 4)) (h3 : V_T = 4 * V_J) :
  (4 * d) / V_J < 1 / (2 * V_J) :=
by
  sorry

end jerry_can_escape_l821_821982


namespace brendan_grass_cutting_l821_821758

theorem brendan_grass_cutting (daily_yards : ℕ) (percentage_increase : ℕ) (original_days_per_week : ℕ) (expected_result : ℕ) :
  daily_yards = 8 →
  percentage_increase = 50 →
  let additional_yards_per_day := (percentage_increase * daily_yards) / 100 in
  let total_yards_per_day := daily_yards + additional_yards_per_day in
  let total_yards_per_week := total_yards_per_day * original_days_per_week in
  original_days_per_week = 7 →
  expected_result = 84 →
  total_yards_per_week = expected_result :=
by
  intros h_daily_yards h_percentage_increase additional_yards_per_day total_yards_per_day total_yards_per_week h_days_per_week h_expected_result
  rw [h_daily_yards, h_percentage_increase, h_days_per_week, h_expected_result]
  simp
  sorry

end brendan_grass_cutting_l821_821758


namespace complex_transformation_l821_821323

open Complex

theorem complex_transformation :
  let z := -3 - (8 * I) in
  let z_rotated := z * I in
  let z_dilated := z_rotated * Real.sqrt 2 in
  z_dilated = 8 * Real.sqrt 2 - 3 * Real.sqrt 2 * I :=
by
  -- The proof goes here
  sorry

end complex_transformation_l821_821323


namespace sqrt_11_12_13_14_plus_1_eq_155_l821_821159

theorem sqrt_11_12_13_14_plus_1_eq_155 : (sqrt (11 * 12 * 13 * 14 + 1) = 155) :=
sorry

end sqrt_11_12_13_14_plus_1_eq_155_l821_821159


namespace correct_propositions_l821_821412

theorem correct_propositions (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * a + b = 1) :
  ¬ (a > b > 0 → 1 / a > 1 / b) ∧
  (a > b > 0 → a - 1 / a > b - 1 / b) ∧
  ¬ (a > b > 0 → ((2 * a + b) / (a + 2 * b)) > (a / b)) ∧
  (a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → ∃ x, x ≥ 9 ∧ x = 2 / a + 1 / b) 
:= 
sorry

end correct_propositions_l821_821412


namespace num_integers_satisfy_condition_l821_821359

theorem num_integers_satisfy_condition :
  {n : ℤ | 2 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉}.finite ∧
  {n : ℤ | 2 + ⌊(200 * n) / 201⌋ = ⌈(198 * n) / 200⌉}.to_finset.card = 40200 :=
by
  sorry

end num_integers_satisfy_condition_l821_821359


namespace PW_length_is_10_l821_821114

noncomputable def length_PW (XY YZ XZ PZ PW : ℝ) (cyclic_hexagon : Prop) : Prop :=
  ∃ (X Y Z P W : Point), 
    dist X Y = XY ∧ dist Y Z = YZ ∧ dist X Z = XZ ∧
    dist P Z = PZ ∧ dist W P = PW ∧
    P ∈ segment X Z ∧ W ∈ line_through P Y ∧
    is_parallel (line_through X W) (line_through Z Y) ∧
    cyclic_hexagon

-- Given the problem conditions
def problem_conditions : Prop :=
  length_PW 10 11 12 6 10
  (cyclic_hexagon (YXYZWX := {X, Y, Z, W, YX}))

-- The theorem to prove
theorem PW_length_is_10 : problem_conditions :=
  sorry

end PW_length_is_10_l821_821114


namespace slope_tangent_line_at_neg1_l821_821652

theorem slope_tangent_line_at_neg1 : 
  let y := λ x : ℝ, (1 / 3) * x^3 - 2 in
  let dydx := λ x : ℝ, x^2 in
  dydx (-1) = 1 :=
by
  let y := λ x : ℝ, (1 / 3) * x^3 - 2
  let dydx := λ x : ℝ, x^2
  show dydx (-1) = 1
  sorry

end slope_tangent_line_at_neg1_l821_821652


namespace locus_of_point_M_l821_821382

theorem locus_of_point_M (r a : ℝ) :
  ∀ (M : ℝ × ℝ), (∃ (β : ℝ), M = (r^2 - β^2) / a, β) → 
  (M.1)^2 + (M.2)^2 = r^2 → 
  (M.2)^2 = r^2 - a * (M.1) :=
sorry

end locus_of_point_M_l821_821382


namespace sum_of_all_possible_values_of_N_l821_821644

theorem sum_of_all_possible_values_of_N :
  ∃ (a b c N : ℕ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a < b) ∧ (b < c) ∧
  (c = a + b) ∧ (N = a * b * c) ∧ (N = 8 * (a + b + c)) ∧ (N = 160) :=
begin
  sorry
end

end sum_of_all_possible_values_of_N_l821_821644


namespace lambda_parallel_l821_821450

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821450


namespace sum_of_remainders_mod_30_l821_821245

theorem sum_of_remainders_mod_30 (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 11) (h3 : c % 30 = 19) :
  (a + b + c) % 30 = 14 :=
by
  sorry

end sum_of_remainders_mod_30_l821_821245


namespace non_adjacent_arrangements_l821_821966

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821966


namespace factor_difference_of_squares_l821_821790

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821790


namespace geometric_feasibility_l821_821590

-- Define the sets representing squares and points within them
variable (Square1 Square2 : Set ℝ ℝ) -- Two squares on the plane
variable (P : Fin 5 → ℝ × ℝ)         -- Five points on the plane
variable (inside_square1 : Fin 5 → Prop) -- Predicate for points inside Square1
variable (inside_square2 : Fin 5 → Prop) -- Predicate for points inside Square2

-- Define conditions
def points_in_square1 : Prop := (inside_square1 0) ∧ (inside_square1 1) ∧ (inside_square1 2)
def points_in_square2 : Prop := (inside_square2 0) ∧ (inside_square2 1) ∧ (inside_square2 3) ∧ (inside_square2 4)

def mutual_points : Prop := ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ (inside_square1 i ∧ inside_square2 i) ∧ (inside_square1 j ∧ inside_square2 j) ∧ (inside_square1 k ∧ inside_square2 k)

-- Define the equivalent proof problem
theorem geometric_feasibility :
  (∃ Square1 Square2 (P : Fin 5 → ℝ × ℝ) (inside_square1 inside_square2 : Fin 5 → Prop),
      points_in_square1 ∧ points_in_square2 ∧ mutual_points) :=
sorry

end geometric_feasibility_l821_821590


namespace OM_angle_bisector_of_QOT_l821_821700

variable {Point : Type} [MetricSpace Point]
variables {S Q T M O : Point}

-- Conditions definitions
def is_angle_bisector (A B C D : Point) : Prop := 
  ∃ (line : Set Point), line ∈ LinesThrough A B ∩ LinesThrough A C ∩ LinesThrough A D

def angle_eq_sum (A B C D E : Point) : Prop := 
  ∠ B C D = ∠ C D E + ∠ D E A

-- Lean statement to prove
theorem OM_angle_bisector_of_QOT (h1 : is_angle_bisector S M Q T)
                                 (h2 : angle_eq_sum O Q T Q T S) :
  is_angle_bisector O M Q T := 
sorry

end OM_angle_bisector_of_QOT_l821_821700


namespace darla_pays_per_watt_l821_821337

-- Define the given conditions
variables (x : ℝ) -- amount Darla pays per watt
constant watts : ℝ := 300
constant late_fee : ℝ := 150
constant total_paid : ℝ := 1350

-- Problem statement
theorem darla_pays_per_watt :
  300 * x + 150 = 1350 → x = 4 :=
by {
  sorry
}

end darla_pays_per_watt_l821_821337


namespace inequalities_hold_l821_821012

theorem inequalities_hold (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧ (b - a) / c > 0 ∧ (a - c) / (a * c) < 0 :=
by 
  sorry

end inequalities_hold_l821_821012


namespace find_velocity_l821_821642

-- Definitions based on given conditions
def pressure_on_sail (k : ℝ) (A : ℝ) (V : ℝ) : ℝ :=
  k * A * V^3

-- Given constants and conditions
def k : ℝ := 1 / 512
def P1 : ℝ := 1
def A1 : ℝ := 1
def V1 : ℝ := 8

-- Additional conditions
def P2 : ℝ := 27
def A2 : ℝ := 9

-- Statement to be proved
theorem find_velocity :
  ∃ V : ℝ, pressure_on_sail k A2 V = P2 ∧ V = 12 :=
begin
  sorry
end

end find_velocity_l821_821642


namespace find_n_probability_event_A_l821_821345

noncomputable theory

-- Definitions
def lanterns := λ (n : ℕ), [1, 2, 2] ++ list.repeat 3 n
def P_pick_3 (n : ℕ) : ℝ := n / (3 + n)
def basic_events (n : ℕ) := list.sigma (lanterns n) (lanterns n) -- basic events for two picks without replacement
def event_A (p : ℕ × ℕ) := p.fst + p.snd ≥ 4
def P_event_A (n : ℕ) : ℝ :=
  let total_events := basic_events n in
  let event_A_occurrences := list.filter event_A total_events in
  event_A_occurrences.length / total_events.length

-- The two theorems
theorem find_n : (n : ℕ) (h1 : P_pick_3 n = 1/4) : n = 1 := sorry

theorem probability_event_A : (n : ℕ) (h2 : n = 1) : P_event_A n = 2/3 := sorry

end find_n_probability_event_A_l821_821345


namespace katie_total_marbles_l821_821990

theorem katie_total_marbles :
  ∀ (pink marbles orange marbles purple marbles total : ℕ),
    pink = 13 →
    orange = pink - 9 →
    purple = 4 * orange →
    total = pink + orange + purple →
    total = 33 :=
by
  intros pink marbles orange marbles purple marbles total
  assume h_pink h_orange h_purple h_total
  sorry

end katie_total_marbles_l821_821990


namespace first_term_of_arithmetic_series_l821_821819

theorem first_term_of_arithmetic_series 
  (a d : ℝ)
  (h1 : 20 * (2 * a + 39 * d) = 600)
  (h2 : 20 * (2 * a + 119 * d) = 1800) :
  a = 0.375 :=
by
  sorry

end first_term_of_arithmetic_series_l821_821819


namespace sqrt_sequence_convergence_l821_821909

theorem sqrt_sequence_convergence :
  ∃ x : ℝ, (x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2) :=
sorry

end sqrt_sequence_convergence_l821_821909


namespace min_xy_value_l821_821501

theorem min_xy_value (x y : ℝ) 
  (h : 2 - sin (x + 2 * y - 1) ^ 2 = (x ^ 2 + y ^ 2 - 2 * (x + 1) * (y - 1)) / (x - y + 1)) :
  xy = 1 / 9 :=
by
-- Proof here.
sorry

end min_xy_value_l821_821501


namespace cos_arithmetic_sequence_sum_eq_neg_half_l821_821041

-- Define the arithmetic sequence
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d

-- The problem statement in Lean 4
theorem cos_arithmetic_sequence_sum_eq_neg_half
  (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_sequence a d)
  (h_sum : a 0 + a 4 + a 8 = π) :
  cos (a 1 + a 7) = -1 / 2 :=
by
  sorry

end cos_arithmetic_sequence_sum_eq_neg_half_l821_821041


namespace probability_leq_one_interval_neg2_3_l821_821600

theorem probability_leq_one_interval_neg2_3 : 
  let x := Uniform (Set.Icc (-2 : ℝ) 3) in 
  Prob (x ≤ 1) = 3 / 5 :=
sorry

end probability_leq_one_interval_neg2_3_l821_821600


namespace total_strawberries_l821_821594

-- Define the number of original strawberries and the number of picked strawberries
def original_strawberries : ℕ := 42
def picked_strawberries : ℕ := 78

-- Prove the total number of strawberries
theorem total_strawberries : original_strawberries + picked_strawberries = 120 := by
  -- Proof goes here
  sorry

end total_strawberries_l821_821594


namespace smallest_repeating_block_of_fraction_l821_821893

theorem smallest_repeating_block_of_fraction (a b : ℕ) (h : a = 8 ∧ b = 11) :
  ∃ n : ℕ, n = 2 ∧ decimal_expansion_repeating_block_length (a / b) = n := by
  sorry

end smallest_repeating_block_of_fraction_l821_821893


namespace length_EI_eq_six_l821_821107

-- Definitions of the conditions
def EFGH : Type := sorry -- Rectangle with given dimensions
def IJKL : Type := sorry -- Rectangle with given dimensions
def EF : ℝ := 8
def EH : ℝ := 4
def IL : ℝ := 12
def IK : ℝ := 4
def GH_perpendicular_IJ : Prop := sorry -- GH and IJ are perpendicular
def shaded_area_within_IJKL : ℝ := (3 / 4) * (EF * EH)

-- The theorem we want to prove
theorem length_EI_eq_six :
  EF = 8 → EH = 4 → IL = 12 → IK = 4 → GH_perpendicular_IJ → shaded_area_within_IJKL = 24 → 
  EI = 6 := 
sorry

end length_EI_eq_six_l821_821107


namespace log10_gt_fraction_l821_821907

theorem log10_gt_fraction {x : ℝ} (h : x > 1) : log 10 (1 + x) > x / (2 * (1 + x)) :=
sorry

end log10_gt_fraction_l821_821907


namespace unique_solution_coefficient_l821_821918

theorem unique_solution_coefficient (a : ℝ) (A : set ℝ) :
  (A = {x | ax^2 - 3x + 2 = 0} ∧ ∃! x, x ∈ A) → (a = 0 ∨ a = 9/8) :=
by
  sorry

end unique_solution_coefficient_l821_821918


namespace marie_gave_joyce_eggs_l821_821985

theorem marie_gave_joyce_eggs (original_eggs : ℕ) (final_eggs : ℕ) (eggs_given_by_marie : ℕ) :
  original_eggs = 8 → final_eggs = 14 → eggs_given_by_marie = final_eggs - original_eggs → eggs_given_by_marie = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end marie_gave_joyce_eggs_l821_821985


namespace probability_odd_l821_821298

-- Define the sample space of a fair six-sided die
def sample_space : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event that the number on the uppermost face is odd
def event_odd : Set ℕ := {1, 3, 5}

-- Prove that the probability of the event 'event_odd' is 1/2
theorem probability_odd : (event_odd.card / sample_space.card : ℚ) = 1 / 2 := by
  sorry

end probability_odd_l821_821298


namespace det_matrix_A_l821_821366

noncomputable def matrix_A (n : ℕ) : Matrix (Fin n) (Fin n) ℕ :=
λ i j, Nat.choose (i + j - 2) (j - 1)

theorem det_matrix_A {n : ℕ} (hn : 0 < n) : Matrix.det (matrix_A n) = 1 :=
sorry

end det_matrix_A_l821_821366


namespace equation_of_line_intersecting_ellipse_l821_821808

theorem equation_of_line_intersecting_ellipse
  (x y : ℝ)
  (hM : (1, 1) ∈ {p | let (x, y) := p in x^2 / 4 + y^2 / 3 = 1})
  (exists_AB : ∃ (A B : ℝ × ℝ), 
      A ∈ ({p : ℝ × ℝ | let (x, y) := p in x^2 / 4 + y^2 / 3 = 1}) ∧ 
      B ∈ ({p : ℝ × ℝ | let (x, y) := p in x^2 / 4 + y^2 / 3 = 1}) ∧
      (1, 1) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ m : ℝ, let l := fun y => m * (y - 1) + 1,
             line_eq := ∃ (m : ℝ), 3 * ((m * (y - 1) + 1)) + 4 * y - 7 = 0 :=
  sorry

end equation_of_line_intersecting_ellipse_l821_821808


namespace triangle_side_length_l821_821223

-- Definitions based on conditions
def ratio_of_altitudes (a1 a2 a3 : ℕ) : Prop := a1 : a2 : a3 = 3 : 4 : 5
def sides_are_integers (X Y Z : ℕ) : Prop := X % 1 = 0 ∧ Y % 1 = 0 ∧ Z % 1 = 0

-- Theorem statement proving X = 12 based on the given conditions
theorem triangle_side_length 
  (X Y Z a1 a2 a3 : ℕ) 
  (h1 : ratio_of_altitudes a1 a2 a3)
  (h2 : sides_are_integers X Y Z)
  -- Derived from area relationship and given ratio a1 : a2 : a3 = 3 : 4 : 5
  (h3 : (X, Y, Z) = (20 * k, 15 * k, 12 * k) for some k : ℕ) :
  X = 12 :=
sorry

end triangle_side_length_l821_821223


namespace octagon_area_difference_is_512_l821_821325

noncomputable def octagon_area_difference (side_length : ℝ) : ℝ :=
  let initial_octagon_area := 2 * (1 + Real.sqrt 2) * side_length^2
  let triangle_area := (1 / 2) * side_length^2
  let total_triangle_area := 8 * triangle_area
  let inner_octagon_area := initial_octagon_area - total_triangle_area
  initial_octagon_area - inner_octagon_area

theorem octagon_area_difference_is_512 :
  octagon_area_difference 16 = 512 :=
by
  -- This is where the proof would be filled in.
  sorry

end octagon_area_difference_is_512_l821_821325


namespace sum_interior_numbers_eighth_row_of_pascals_triangle_l821_821679

theorem sum_interior_numbers_eighth_row_of_pascals_triangle :
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  sum_interior_numbers = 126 :=
by
  let n := 8
  let sum_all_elements := 2 ^ (n - 1)
  let sum_interior_numbers := sum_all_elements - 2
  show sum_interior_numbers = 126
  sorry

end sum_interior_numbers_eighth_row_of_pascals_triangle_l821_821679


namespace sum_of_arithmetic_sequence_l821_821563

theorem sum_of_arithmetic_sequence (n : ℕ) (d : ℝ) (a : ℕ → ℝ)
  (h1 : d ≠ 0) 
  (h2 : a 1 = 2) 
  (h3 : (a 3 - a 1 : ℝ) = 2 * d)
  (h4 : (a 6 : ℝ) = a 1 + 5 * d)
  (h5 : (a 3 ^ 2 : ℝ) = a 1 * a 6) 
  :
  (finset.sum (finset.range n) (λ k, a (k + 1)) = n^2 / 4 + 7 * n / 4) :=
by
  sorry

end sum_of_arithmetic_sequence_l821_821563


namespace smallest_n_exists_exists_example_for_26_l821_821141

theorem smallest_n_exists
  (n : ℕ)
  (x : ℕ → ℝ)
  (h_abs_lt_one : ∀ i, i < n → |x i| < 1)
  (h_eq : (Σ' i, if i < n then |x i| else 0) = 25 + |(Σ' i, if i < n then x i else 0)| ) :
  26 ≤ n :=
by
  sorry

theorem exists_example_for_26 :
  ∃ (x : ℕ → ℝ), (∀ i, i < 26 → |x i| < 1) ∧ ((Σ' i, if i < 26 then |x i| else 0) = 25 + |(Σ' i, if i < 26 then x i else 0)|) :=
by
  sorry

end smallest_n_exists_exists_example_for_26_l821_821141


namespace find_circumradius_l821_821095

-- Definitions of the conditions
variables (α m : ℝ) (A B C : Type*) 

-- Given conditions for the triangle
def is_isosceles_triangle : Prop := (A = C)
def angle_at_base := α
def altitude_greater_than_inradius := exists (r : ℝ), BD = r + m

-- The theorem to prove R given the conditions
theorem find_circumradius (h1 : is_isosceles_triangle A C)
                          (h2 : angle_at_base α)
                          (h3 : altitude_greater_than_inradius m) :
  R = m / (4 * (sin (α / 2))^2) :=
sorry

end find_circumradius_l821_821095


namespace triangle_area_is_50_l821_821812

open BigOperators

-- Define the points of the triangle
def A : ℝ × ℝ := (-10, 0)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (0, 0)

-- Function to compute the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Define the actual theorem that the area is 50 square units
theorem triangle_area_is_50 :
  triangle_area A B C = 50 := by
  sorry

end triangle_area_is_50_l821_821812


namespace sum_recursion_identity_l821_821558

noncomputable def G : ℕ → ℝ
| 0     := 0
| 1     := 5 / 2
| (n + 2) := (7 / 2) * G (n + 1) - G n

theorem sum_recursion_identity :
  (∑ n, 1 / G (2^n)) = 1 :=
by
  sorry

end sum_recursion_identity_l821_821558


namespace who_is_A_l821_821641

variables (A : Type) (is_monkey : A → Prop) (is_knight : A → Prop) (is_human : A → Prop)

-- A's statement: "It is not true that I am a monkey and a knight."
def statement (a : A) := ¬ (is_monkey a ∧ is_knight a)

-- Given conditions for this problem:
variables (a : A) (h_statement : statement a)

-- Proof to show: a is a human and a knight
theorem who_is_A : is_human a ∧ is_knight a :=
sorry

end who_is_A_l821_821641


namespace factor_x_squared_minus_64_l821_821792

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821792


namespace solve_for_x_l821_821373

theorem solve_for_x (x : ℝ) : (3 : ℝ)^(4 * x^2 - 3 * x + 5) = (3 : ℝ)^(4 * x^2 + 9 * x - 6) ↔ x = 11 / 12 :=
by sorry

end solve_for_x_l821_821373


namespace min_detectors_for_7x7_board_with_2x2_ship_l821_821705

noncomputable def min_detectors_needed : ℕ :=
  16

theorem min_detectors_for_7x7_board_with_2x2_ship :
  ∀ (board : array (7 * 7) (option bool)),
    (∃ ship_coords : list (ℕ × ℕ), 
      ship_coords.length = 4 ∧
      ∀ (coord : ℕ × ℕ), coord ∈ ship_coords → 
      coord.fst < 7 ∧ coord.snd < 7 ∧ 
      (∀ (i j : ℕ), (i, j) ∈ ship_coords → (i, j + 1) ∈ ship_coords ∨ 
         (i + 1, j) ∈ ship_coords)
      ) →           
    ∃ (detectors : list (ℕ × ℕ)),
      detectors.length = 16 ∧
      ∀ (ship : list (ℕ × ℕ)), ship.length = 4 → 
      (∀ detector ∈ detectors, ∃ cell ∈ ship, detector = cell) :=
by
  sorry

end min_detectors_for_7x7_board_with_2x2_ship_l821_821705


namespace length_of_MN_l821_821591

-- Definitions
variables (ABCD : Type) [Trapezoid ABCD] (M N : Point) (a b : ℝ)

-- Conditions
variables (BC_length : BC_length ABCD = a)
variables (AD_length : AD_length ABCD = b)
variables (M_on_AB : M ∈ Side_AB ABCD)
variables (N_on_CD : N ∈ Side_CD ABCD)
variables (MN_parallel : parallel MN (Line BC))
variables (area_half : divides_area_half MN ABCD)

-- Theorem statement
theorem length_of_MN :
  length MN = sqrt ((a^2 + b^2) / 2) :=
sorry

end length_of_MN_l821_821591


namespace average_words_and_difference_l821_821756

-- Definitions based on the conditions
variable (days_in_two_weeks : ℕ := 14)
variable (words_per_pencil : ℕ := 1050)

-- Define puzzles and their respective sizes
variable (puzzles_completed_in_two_weeks : ℕ := days_in_two_weeks)
variable (average_words_per_puzzle : ℕ := words_per_pencil / puzzles_completed_in_two_weeks)

-- Puzzle grid sizes
variable (size_15x15 : ℕ := 15 * 15)
variable (size_21x21 : ℕ := 21 * 21)
variable (ratio : ℚ := size_21x21 / size_15x15)
variable (estimated_words_per_21x21 : ℕ := (average_words_per_puzzle * ratio).toNat)
variable (difference_in_words : ℕ := estimated_words_per_21x21 - average_words_per_puzzle)

-- Lean theorem stating the equivalence to the problem's answers
theorem average_words_and_difference :
  average_words_per_puzzle = 75 ∧ difference_in_words = 72 :=
by
  -- Proof is skipped for now
  sorry

end average_words_and_difference_l821_821756


namespace number_of_arrangements_l821_821947

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821947


namespace triangle_inequality_calculate_perimeter_l821_821314

def triangle_sides := {a : ℕ // a = 10} ∪ {b : ℕ // b = 7} ∪ {c : ℕ // c = 5}

theorem triangle_inequality (a b c : ℕ) (h1: a + b > c) (h2: a + c > b) (h3: b + c > a) : true :=
begin
  sorry
end

theorem calculate_perimeter (a b c : ℕ) 
  (h1 : a = 10) (h2 : b = 7) (h3 : c = 5)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : (a + b + c = 22) :=
by {
  calc (10 + 7 + 5) = 22 : by norm_num
}

end triangle_inequality_calculate_perimeter_l821_821314


namespace second_blend_price_l821_821163

-- Define the constant prices and quantities given in the conditions
def price_first_blend : ℝ := 9.0
def total_weight_blend : ℝ := 20.0
def target_price_blend : ℝ := 8.4
def weight_second_blend : ℝ := 12.0
def weight_first_blend : ℝ := total_weight_blend - weight_second_blend

-- Define the price of the second blend to be solved for
def price_second_blend := 8.0

theorem second_blend_price:
  let cost_first_blend := weight_first_blend * price_first_blend in
  let cost_second_blend := weight_second_blend * price_second_blend in
  let total_cost := total_weight_blend * target_price_blend in
  cost_first_blend + cost_second_blend = total_cost :=
sorry

end second_blend_price_l821_821163


namespace cevian_triangle_construction_exists_l821_821177

-- Defining a Triangle structure
structure Triangle :=
  (A B C : Point) -- Points representing vertices of the triangle
  (side1 : LineSegment A B) -- Side AB
  (side2 : LineSegment B C) -- Side BC
  (side3 : LineSegment C A) -- Side CA

-- Definition of Cevian
def isCevian (P Q R : Point) (cev : LineSegment P R) : Prop :=
  cev.start = P ∧ R ≠ Q ∧ cev.contains R

-- Main theorem statement
theorem cevian_triangle_construction_exists (T : Triangle) :
  ∃ (V : T.A = A ∨ T.B = B ∨ T.C = C),
    ∀ (cev : LineSegment V _),
    (isCevian V _ cev ) →
    (∃ U : Point, U ≠ _ ∧ LineSegment V U ∧ LineSegment V _) :=
sorry

end cevian_triangle_construction_exists_l821_821177


namespace k_range_l821_821364

theorem k_range (a b k : ℝ) : 
  (∀ a b : ℝ, (a - b)^2 ≥ k * a * b) ↔ k ∈ set.Icc (-4 : ℝ) 0 :=
sorry

end k_range_l821_821364


namespace largest_digit_change_to_correct_sum_l821_821613

theorem largest_digit_change_to_correct_sum :
  let a := 731,
      b := 962,
      c := 843 in
  -- the original sum
  let original_sum := a + b + c,
      provided_sum := 2436,
      discrepancy := original_sum - provided_sum in
  -- verify the sum after changing the appropriate digit in b
  let changed_b := b - 100 in
  changed_b = (b - 100) ∧ original_sum - 100 = provided_sum ∧ 9 = 9 :=
begin
  -- declare the sum calculation
  let a := 731,
      b := 962,
      c := 843,
      provided_sum := 2436,
      correct_sum := 2536,
      discrepancy := correct_sum - provided_sum,
      changed_b := b - 100,
      new_sum := a + changed_b + c in
  -- verify the conditions
  have h_sum: correct_sum = 731 + 962 + 843 := rfl,
  have h_discrepancy: discrepancy = 100 := rfl,
  have h_changed_digit: changed_b = 862 := rfl,
  have h_new_sum: new_sum = provided_sum := rfl,
  have h_largest_digit: 9 = 9 := rfl,
  exact ⟨h_changed_digit, h_new_sum, h_largest_digit⟩,
end

end largest_digit_change_to_correct_sum_l821_821613


namespace math_problem_l821_821398

theorem math_problem (a b c d x : ℝ)
  (h1 : a = -(-b))
  (h2 : c = -1 / d)
  (h3 : |x| = 3) :
  x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36 :=
by sorry

end math_problem_l821_821398


namespace no_term_sum_of_three_7th_powers_l821_821179

def recurrence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) % 29 = (a (n + 1) % 29)^2 + 12 * (a (n + 1) % 29) * (a n % 29) + (a (n + 1) % 29) + 11 * (a n % 29)

def initial_terms (a : ℕ → ℤ) : Prop :=
  a 1 % 29 = 8 ∧ a 2 % 29 = 20

def periodic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n % 4 + 1) ≡ (if n % 4 = 0 then 8 else if n % 4 = 1 then 20 else if n % 4 = 2 then 21 else 9) [MOD 29]

theorem no_term_sum_of_three_7th_powers (a : ℕ → ℤ) :
  recurrence a → initial_terms a → periodic_sequence a →
  ∀ n, ∀ x y z : ℤ, (a n % 29 ≠ (x^7 % 29 + y^7 % 29 + z^7 % 29) % 29) :=
by
  intros h_recurrence h_initial h_periodicity n x y z
  sorry

end no_term_sum_of_three_7th_powers_l821_821179


namespace ott_fraction_of_total_money_l821_821829

theorem ott_fraction_of_total_money (x : ℝ) (m l n g : ℝ)
    (hm : m = 6*x) (hl : l = 3*x) (hn : n = 2*x) (hg : g = 4*x) :
    (4 * x) / (m + l + n + g) = 4 / 15 :=
by {
  -- Definitions based on what friends gave Ott
  have hmoe := hm,
  have hloki := hl,
  have hnick := hn,
  have hgil := hg,
  -- Total money calculation
  let total_money := m + l + n + g,
  calc
    (4 * x) / total_money = (4 * x) / (6*x + 3*x + 2*x + 4*x) : by rw [hmoe, hloki, hnick, hgil]
    ... = 4 / 15 : by ring
}

end ott_fraction_of_total_money_l821_821829


namespace min_points_irrational_distance_n1_min_points_irrational_distance_n_gt_1_l821_821675

-- Define the problems for n = 1 and n > 1

theorem min_points_irrational_distance_n1 (x : ℝ) :
  ∃ (points : List ℝ), points.length = 2 ∧ 
  (∀ (y : ℝ), ∃ (p : ℝ), p ∈ points ∧ ¬ is_rat (abs (y - p))) :=
sorry

theorem min_points_irrational_distance_n_gt_1 (n : ℕ) (hn : n > 1) (x : EuclideanSpace ℝ (Fin n)) :
  ∃ (points : List (EuclideanSpace ℝ (Fin n))), points.length = 3 ∧ 
  (∀ (y : EuclideanSpace ℝ (Fin n)), ∃ (p : EuclideanSpace ℝ (Fin n)), p ∈ points ∧ ¬ is_rat (dist y p)) :=
sorry

end min_points_irrational_distance_n1_min_points_irrational_distance_n_gt_1_l821_821675


namespace probability_same_person_l821_821595

theorem probability_same_person (A B C D : Type) (send_to : A → D) 
  (ha : ∀ (x : A), send_to x = C ∨ send_to x = D) :
  let p := 1 / 2 
  in ∑ a₁ ∈ {C, D}, ∑ b₁ ∈ {C, D}, (send_to a₁ = send_to b₁ : Prop) 
    / (∑ a₂ ∈ {C, D}, 1 * ∑ b₂ ∈ {C, D}, 1) = p :=
by sorry

end probability_same_person_l821_821595


namespace dan_bought_one_candy_bar_l821_821771

-- Define the conditions
def initial_money : ℕ := 4
def cost_per_candy_bar : ℕ := 3
def money_left : ℕ := 1

-- Define the number of candy bars Dan bought
def number_of_candy_bars_bought : ℕ := (initial_money - money_left) / cost_per_candy_bar

-- Prove the number of candy bars bought is equal to 1
theorem dan_bought_one_candy_bar : number_of_candy_bars_bought = 1 := by
  sorry

end dan_bought_one_candy_bar_l821_821771


namespace angle_CAB_60_l821_821996

variables {A B C K B₁ C₁ B₂ C₂ : Type} [EuclideanGeometry A B C K B₁ C₁ B₂ C₂]

def is_incenter (K : Type) (Δ : Triangle Type) : Prop := -- define what an incenter is
sorry

def is_midpoint (M P Q : Type) : Prop := -- define what a midpoint is
sorry

theorem angle_CAB_60 
  (K_is_incenter : is_incenter K (Triangle A B C))
  (C1_midpoint : is_midpoint C₁ A B)
  (B1_midpoint : is_midpoint B₁ A C)
  (B2_on_AC : ∃ (D : Type), D = (Line C₁ K) ∧ D ∩ (Line A C) = B₂)
  (C2_on_AB : ∃ (E : Type), E = (Line B₁ K) ∧ E ∩ (Line A B) = C₂)
  (area_equal : Area (Triangle A B₂ C₂) = Area (Triangle A B C)) :
  ∠A B C = 60 :=
begin
  sorry
end

end angle_CAB_60_l821_821996


namespace find_line_l_l821_821532

def line_equation (x y: ℤ) : Prop := x - 2 * y = 2

def scaling_transform_x (x: ℤ) : ℤ := x
def scaling_transform_y (y: ℤ) : ℤ := 2 * y

theorem find_line_l :
  ∀ (x y x' y': ℤ),
  x' = scaling_transform_x x →
  y' = scaling_transform_y y →
  line_equation x y →
  x' - y' = 2 := by
  sorry

end find_line_l_l821_821532


namespace trigonometric_comparison_l821_821857

open Real

theorem trigonometric_comparison :
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  a < b ∧ b < c := 
by
  let a := 2 * sin (1 / 2)
  let b := 3 * sin (1 / 3)
  let c := 3 * cos (1 / 3)
  sorry

end trigonometric_comparison_l821_821857


namespace smallest_number_is_correct_l821_821361

noncomputable def smallest_number_divisible_by_225 : ℕ :=
  11111111100

theorem smallest_number_is_correct :
  (smallest_number_divisible_by_225 % 225 = 0) ∧
  (∀ (m : ℕ), (m % 225 = 0) → (m.toDigits 2).All (fun d => d = 0 ∨ d = 1) → (m ≥ smallest_number_divisible_by_225)) :=
by
  sorry

end smallest_number_is_correct_l821_821361


namespace parallel_vectors_lambda_l821_821447

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821447


namespace sum_of_x_coords_Q3_l821_821281

theorem sum_of_x_coords_Q3 (x_coords : Fin 50 → ℝ)
  (sum_x_coords_Q1 : ∑ i, x_coords i = 4050) : 
  let Q1_midpoints : Fin 50 → ℝ := λ i, (x_coords i + x_coords (i + 1)) / 2
  let Q2_midpoints : Fin 50 → ℝ := λ i, (Q1_midpoints i + Q1_midpoints (i + 1)) / 2
  in (∑ i, Q2_midpoints i) = 4050 :=
by sorry

end sum_of_x_coords_Q3_l821_821281


namespace num_students_class1_eq_30_l821_821240

-- Define the known conditions
def average_mark_class1 := 40
def average_mark_class2 := 90
def num_students_class2 := 50
def combined_average_mark := 71.25

-- Define the variables and calculations
def total_marks_class1 (x : ℕ) := average_mark_class1 * x
def total_marks_class2 := average_mark_class2 * num_students_class2
def total_students (x : ℕ) := x + num_students_class2
def total_marks (x : ℕ) := total_marks_class1 x + total_marks_class2

-- The main statement to be proved
theorem num_students_class1_eq_30 : ∃ (x : ℕ), (total_marks x) / (total_students x) = combined_average_mark ∧ x = 30 :=
by
  sorry

end num_students_class1_eq_30_l821_821240


namespace rounding_problem_l821_821184

def given_number : ℝ := 3967149.487234

theorem rounding_problem : (3967149.487234).round = 3967149 := sorry

end rounding_problem_l821_821184


namespace inequalities_hold_l821_821014

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l821_821014


namespace not_possible_to_fill_grid_l821_821119

theorem not_possible_to_fill_grid :
  ¬ ∃ (f : Fin 7 → Fin 7 → ℝ), ∀ i j : Fin 7,
    ((if j > 0 then f i (j - 1) else 0) +
     (if j < 6 then f i (j + 1) else 0) +
     (if i > 0 then f (i - 1) j else 0) +
     (if i < 6 then f (i + 1) j else 0)) = 1 :=
by
  sorry

end not_possible_to_fill_grid_l821_821119


namespace parallel_lines_drawing_l821_821592

theorem parallel_lines_drawing
    (lines_parallel : ∀ (l1 l2 l3 : Line), (corresponding_angles l1 l3) = (corresponding_angles l2 l3) → (parallel l1 l2))
    (triangular_ruler : ∀ (l1 l2 : Line) (ruler : Ruler) (stationary : Line), (consistent_angle ruler l1 stationary) → (consistent_angle ruler l2 stationary) → (parallel l1 l2))
    (other_instrument : ∀ (instrument : Instrument) (l1 l2 : Line) (stationary : Line), (fixed_points instrument) ∧ (consistent_angle_with_fixed_points instrument l1 stationary) → (consistent_angle_with_fixed_points instrument l2 stationary) → (parallel l1 l2)) :
    True :=
by
  sorry

end parallel_lines_drawing_l821_821592


namespace range_of_f_neg2_l821_821051

def quadratic_fn (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_neg2 (a b : ℝ) (h1 : 1 ≤ quadratic_fn a b (-1) ∧ quadratic_fn a b (-1) ≤ 2)
    (h2 : 2 ≤ quadratic_fn a b 1 ∧ quadratic_fn a b 1 ≤ 4) :
    3 ≤ quadratic_fn a b (-2) ∧ quadratic_fn a b (-2) ≤ 12 :=
sorry

end range_of_f_neg2_l821_821051


namespace Katie_marble_count_l821_821987

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end Katie_marble_count_l821_821987


namespace extra_food_needed_l821_821682

theorem extra_food_needed (f1 f2 : ℝ) (h1 : f1 = 0.5) (h2 : f2 = 0.9) :
  f2 - f1 = 0.4 :=
by sorry

end extra_food_needed_l821_821682


namespace bad_iff_prime_l821_821252

def a_n (n : ℕ) : ℕ := (2 * n)^2 + 1

def is_bad (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a_n n = a^2 + b^2

theorem bad_iff_prime (n : ℕ) : is_bad n ↔ Nat.Prime (a_n n) :=
by
  sorry

end bad_iff_prime_l821_821252


namespace exists_sum_coprime_seventeen_not_sum_coprime_l821_821176

/-- 
 For any integer \( n \) where \( n > 17 \), there exist integers \( a \) and \( b \) 
 such that \( n = a + b \), \( a > 1 \), \( b > 1 \), and \( \gcd(a, b) = 1 \).
 Additionally, the integer 17 does not have this property.
-/
theorem exists_sum_coprime (n : ℤ) (h : n > 17) : 
  ∃ (a b : ℤ), n = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

/-- 
 The integer 17 cannot be expressed as the sum of two integers greater than 1 
 that are coprime.
-/
theorem seventeen_not_sum_coprime : 
  ¬ ∃ (a b : ℤ), 17 = a + b ∧ a > 1 ∧ b > 1 ∧ Int.gcd a b = 1 :=
sorry

end exists_sum_coprime_seventeen_not_sum_coprime_l821_821176


namespace vacation_savings_l821_821620

-- Definitions
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500
def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Prove the amount set aside for vacation
theorem vacation_savings :
  let 
    total_income := parents_salary + grandmothers_pension + sons_scholarship,
    total_expenses := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses,
    surplus := total_income - total_expenses,
    deposit := (10 * surplus) / 100, 
    vacation_money := surplus - deposit
  in
    vacation_money = 16740 := by
      -- Calculation steps skipped; proof not required
      sorry

end vacation_savings_l821_821620


namespace y_intercept_of_line_m_l821_821575

open Real

-- Definitions for the conditions
def line_m_in_xy_plane : Prop := true

def slope_of_line_m : ℝ := 1

def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)

def passes_through_midpoint : Prop :=
  let p := midpoint 2 8 14 4
  p = (8, 6)

-- The proof problem
theorem y_intercept_of_line_m : 
  line_m_in_xy_plane → 
  slope_of_line_m = 1 → 
  passes_through_midpoint → 
  let m := slope_of_line_m in
  let b := 6 - m * 8 in
  b = -2 :=
by
  intros _ h₁ h₂
  sorry

end y_intercept_of_line_m_l821_821575


namespace area_of_quadrilateral_is_correct_l821_821052

noncomputable def area_of_quadrilateral (a b c : ℝ) : ℝ :=
  let P1 := (2 * sqrt (30) / 5, sqrt 5 / 5)
  let P2 := (-2 * sqrt (30) / 5, sqrt 5 / 5)
  let P3 := (-2 * sqrt (30) / 5, -sqrt 5 / 5)
  let P4 := (2 * sqrt (30) / 5, -sqrt 5 / 5)
  let length := (4 * sqrt (30) / 5)
  let width := (2 * sqrt 5 / 5)
  (length * width)

theorem area_of_quadrilateral_is_correct :
  let C := λ (P : ℝ × ℝ), (P.1 ^ 2) / 4 - P.2 ^ 2 = 1 in
  let F1 := (-sqrt 5, 0) in
  let F2 := (sqrt 5, 0) in
  let satisfies_condition := λ (P : ℝ × ℝ), (P.1, P.2) ∈ C ∧ 
                              (((P.1 - -sqrt 5) * (P.1 - sqrt 5)) + (P.2 * P.2)) = 0 in
  let P1 := (2 * sqrt (30) / 5, sqrt 5 / 5) in
  let P2 := (-2 * sqrt (30) / 5, sqrt 5 / 5) in
  let P3 := (-2 * sqrt (30) / 5, -sqrt 5 / 5) in
  let P4 := (2 * sqrt (30) / 5, -sqrt 5 / 5) in
  let area := area_of_quadrilateral C F1 F2 in
  area = 8 * sqrt 6 / 5 := sorry

end area_of_quadrilateral_is_correct_l821_821052


namespace volume_prism_l821_821273

theorem volume_prism (AC1 PQ : ℝ) (ϕ : ℝ) 
  (sin_ϕ cos_ϕ : ℝ)
  (h_AC1 : AC1 = 3) (h_PQ : PQ = sqrt 3) 
  (h_ϕ : ϕ = real.to_radians 30) 
  (h_sin : sin_ϕ = 1 / 2) 
  (h_cos : cos_ϕ = sqrt 3 / 2) :
  ∃ V : ℝ, V = sqrt 6 / 2 :=
by
  sorry

end volume_prism_l821_821273


namespace constant_function_value_l821_821505

variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g x = 5)

theorem constant_function_value (x : ℝ) : g (3 * x - 7) = 5 :=
by
  apply h
  sorry

end constant_function_value_l821_821505


namespace gcd_seven_digit_repeated_l821_821732

theorem gcd_seven_digit_repeated (n : ℕ) (h1 : 100 ≤ n) (h2 : n < 1000) : 
  ∃ d, (∀ m : ℕ, 100 ≤ m ∧ m < 1000 → d ∣ (1001 * m)) ∧ (∀ c, (∀ m : ℕ, 100 ≤ m ∧ m < 1000 → c ∣ (1001 * m)) → c ≤ d) :=
begin
  use 1001,
  split,
  {
    intros m hm,
    exact dvd_mul_left 1001 m,
  },
  {
    intros c hc,
    apply le_of_dvd,
    {
      use 1001,
    },
    {
      exact hc 100 (by norm_num : 100 ≤ 100 ∧ 100 < 1000),
    },
  },
end

end gcd_seven_digit_repeated_l821_821732


namespace totalCorrectQuestions_l821_821169

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end totalCorrectQuestions_l821_821169


namespace find_m_value_l821_821755

theorem find_m_value :
  let x_values := [8, 9.5, m, 10.5, 12]
  let y_values := [16, 10, 8, 6, 5]
  let regression_eq (x : ℝ) := -3.5 * x + 44
  let avg (l : List ℝ) := l.sum / l.length
  avg y_values = 9 →
  avg x_values = (40 + m) / 5 →
  9 = regression_eq (avg x_values) →
  m = 10 :=
by
  sorry

end find_m_value_l821_821755


namespace polynomial_inequality_solution_l821_821858

theorem polynomial_inequality_solution
  (f : ℝ → ℝ)
  (Hf : ∃ (g : ℝ → ℝ), ∀ x, f(x) = g(x))
  (Hineq : ∀ x, f(x+1) + f(x-1) ≤ 2 * x^2 - 4 * x) :
  (∃ c, ∀ x, f(x) = x^2 - 2 * x + c ∧ c ≤ -1) ∨
  (∃ a b c, ∀ x, f(x) = a * x^2 + b * x + c ∧ a < 1 ∧ c ≤ ((b + 2)^2) / (4 * (1 - a)) - a) :=
sorry

end polynomial_inequality_solution_l821_821858


namespace shoes_remaining_l821_821151

theorem shoes_remaining (monthly_goal : ℕ) (sold_last_week : ℕ) (sold_this_week : ℕ) (remaining_shoes : ℕ) :
  monthly_goal = 80 →
  sold_last_week = 27 →
  sold_this_week = 12 →
  remaining_shoes = monthly_goal - sold_last_week - sold_this_week →
  remaining_shoes = 41 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end shoes_remaining_l821_821151


namespace triangle_similarity_equivalence_l821_821573

variables {A B C D E F : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]

-- Define the triangle ABC and the points of tangency D, E, and F
structure Triangle (A B C : Type*) :=
(perimeter : ℝ)
(incircle_tangent_BC : D)
(incircle_tangent_CA : E)
(incircle_tangent_AB : F)

def similar (T1 T2 : Triangle A B C) : Prop :=
∃ (f : T1 → T2), 
  -- similarity transformation exists
  sorry

def equilateral (T : Triangle A B C) : Prop :=
T.perimeter / 3 = sorry -- each side of the triangle is equal

-- Formal statement: Triangle ABC is similar to Triangle DEF if and only if ABC is an equilateral triangle
theorem triangle_similarity_equivalence (ABC DEF : Triangle A B C) :
  (similar ABC DEF) ↔ (equilateral ABC) :=
sorry

end triangle_similarity_equivalence_l821_821573


namespace sin_A_of_tan_A_l821_821082

theorem sin_A_of_tan_A {A B C : ℝ} (hC : ∠A B C = 90) (tanA : Real.tan A = 1 / 3) : Real.sin A = Real.sqrt 10 / 10 :=
by
  sorry

end sin_A_of_tan_A_l821_821082


namespace parallel_vectors_lambda_l821_821476

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821476


namespace angle_bisector_property_l821_821697

-- Definitions and hypothesis in Lean 4
variables {S Q T O M : Type} [Nonempty S] [Nonempty Q] [Nonempty T] [Nonempty O] [Nonempty M]
variables (angle SQT : ℝ) (angle QTS : ℝ) (angle QST : ℝ) (angle OQT : ℝ)

-- Given conditions
def is_angle_bisector (S M T : Type) : Prop := sorry
def meets_condition (O S T Q : Type) (angle OQT : ℝ) (angle QTS : ℝ) (angle QST : ℝ) : Prop :=
  angle OQT = angle QTS + angle QST

-- The proof problem to be shown
theorem angle_bisector_property
  (h1 : is_angle_bisector S M T)
  (h2 : meets_condition O S T Q angle OQT angle QTS angle QST) :
  is_angle_bisector O M (QOT : angle) :=
sorry

end angle_bisector_property_l821_821697


namespace seating_arrangements_l821_821962

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821962


namespace distance_O_to_farthest_vertex_l821_821201

open Real EuclideanGeometry

-- Definitions from conditions
variable (A B C D O : Point)
variable (areaABCD : area (rectangle A B C D) = 48)
variable (diagABCD : dist A C = 10)
variable (distOB : dist O B = 13)
variable (distOD : dist O D = 13)

-- The theorem to state the problem equivalently
theorem distance_O_to_farthest_vertex :
  ∃ (X : Point), X ∈ {A, B, C, D} ∧ (∀ Y ∈ {A, B, C, D}, dist O Y ≤ dist O X) ∧ dist O X = 7 * Real.sqrt (29/5) :=
by
  sorry

end distance_O_to_farthest_vertex_l821_821201


namespace no_closed_25gon_exists_l821_821390

theorem no_closed_25gon_exists
    (A : Point)
    (l : Line)
    (l_not_through_A : ¬ (A ∈ l))
    (segments : Fin 25 → LineSegment)
    (distinct_segments : Function.Injective segments)
    (start_at_A : ∀ i, segments i).start = A)
    (ends_on_l : ∀ i, (segments i).end ∈ l) :
    ¬ ∃ (polygon : Fin 25 → LineSegment),
        (∀ i, ∃ j, (polygon i).length = (segments j).length ∧ (polygon i) ∥ (segments j)) ∧
        (loop_closed : (Σ i, polygon i).end = polygon 0).start :=
by
  sorry

end no_closed_25gon_exists_l821_821390


namespace find_orthocenter_l821_821088

-- Define the points A, B, C
def A := (13 : ℝ, 43 : ℝ)
def B := (-36 : ℝ, 19 * Real.sqrt 2)
def C := (18 : ℝ, -11 * Real.sqrt 14)

-- Define the orthocenter H to be proved
def H := (-5 : ℝ, 43 + 19 * Real.sqrt 2 - 11 * Real.sqrt 14)

-- Statement: H is the orthocenter of triangle ABC
theorem find_orthocenter :
  ∃ H : ℝ × ℝ, H = (-5, 43 + 19 * Real.sqrt 2 - 11 * Real.sqrt 14) ∧ 
  is_orthocenter A B C H := sorry

end find_orthocenter_l821_821088


namespace flashes_in_fraction_of_hour_l821_821268

-- Definitions for the conditions
def flash_interval : ℕ := 6       -- The light flashes every 6 seconds
def hour_in_seconds : ℕ := 3600 -- There are 3600 seconds in an hour
def fraction_of_hour : ℚ := 3/4 -- ¾ of an hour

-- The translated proof problem statement in Lean
theorem flashes_in_fraction_of_hour (interval : ℕ) (sec_in_hour : ℕ) (fraction : ℚ) :
  interval = flash_interval →
  sec_in_hour = hour_in_seconds →
  fraction = fraction_of_hour →
  (fraction * sec_in_hour) / interval = 450 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end flashes_in_fraction_of_hour_l821_821268


namespace margie_change_l821_821579

theorem margie_change :
  let num_apples := 5
  let cost_per_apple := 0.30
  let discount := 0.10
  let amount_paid := 10.00
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount)
  let change_received := amount_paid - discounted_cost
  change_received = 8.65 := sorry

end margie_change_l821_821579


namespace divisors_remainder_5_l821_821260

theorem divisors_remainder_5 (d : ℕ) : d ∣ 2002 ∧ d > 5 ↔ d = 7 ∨ d = 11 ∨ d = 13 ∨ d = 14 ∨ 
                                      d = 22 ∨ d = 26 ∨ d = 77 ∨ d = 91 ∨ 
                                      d = 143 ∨ d = 154 ∨ d = 182 ∨ d = 286 ∨ 
                                      d = 1001 ∨ d = 2002 :=
by sorry

end divisors_remainder_5_l821_821260


namespace projection_vector_l821_821728

theorem projection_vector :
  let u := (⟨2, -1⟩ : ℝ × ℝ)
  let v := (⟨-3, 2⟩ : ℝ × ℝ)
  let projection := (⟨1, -1/2⟩ : ℝ × ℝ)
  (u.1 * u.1 + u.2 * u.2 ≠ 0) →
  (u.1 * projection.1 + u.2 * projection.2 = u.1 * u.1 + u.2 * u.2) →
  ∃ w: ℝ × ℝ, w = ⟨-16 / 5, 8 / 5⟩ ∧
  w.1 = (u.1 * (u.1 * v.1 + u.2 * v.2)) / (u.1 * u.1 + u.2 * u.2) ∧
  w.2 = (u.2 * (u.1 * v.1 + u.2 * v.2)) / (u.1 * u.1 + u.2 * u.2) :=
by {
  intros,
  sorry
}

end projection_vector_l821_821728


namespace GP_GQ_GR_sum_eq_47_15_l821_821974

noncomputable def triangle_GP_GQ_GR_sum (XY XZ YZ : ℝ) (G P Q R : Point) (G_on_YZ G_on_XZ G_on_XY : Line) :=
  let XY_on := XY = 4
  let XZ_on := XZ = 3
  let YZ_on := YZ = 5
  let GP := (2 * 2) / YZ
  let GQ := (2 * 2) / XZ
  let GR := (2 * 2) / XY
  in GP + GQ + GR = 47 / 15

open Set

-- Assuming point type and line type definitions
variables {Point : Type} {Line : Type}
variables [MetricSpace Point]

-- Define the theorem
theorem GP_GQ_GR_sum_eq_47_15
  (XY XZ YZ : ℝ) (X Y Z G P Q R : Point)
  (XM YN ZO G_on : ∀(p1 p2: Point), Line)
  (hcond1 : XY = 4) (hcond2 : XZ = 3) (hcond3 : YZ = 5)
  (h_projection_GP : ∃ P, MetricSpace.dist G P = (2 / 5) * YZ)
  (h_projection_GQ : ∃ Q, MetricSpace.dist G Q = (2 / 3) * XZ)
  (h_projection_GR : ∃ R, MetricSpace.dist G R = 1
    (GP_sum : GP_GQ_GR_sum XY XZ YZ G P Q R G_on_YZ G_on_XZ G_on_XY) : 
  GP + GQ + GR := sorry

end GP_GQ_GR_sum_eq_47_15_l821_821974


namespace ellipse_properties_l821_821023

theorem ellipse_properties 
  (a b : ℝ) (e : ℝ := sqrt 3 / 2)
  (h_ab : a > b)
  (h_b0 : b > 0)
  (h_eccentricity : sqrt (1 - (b^2 / a^2)) = e)
  (P : ℝ × ℝ := (2, 1))
  (M : ℝ × ℝ → Prop := λ p => (p.1^2 / 8 + p.2^2 / 2 = 1))
  (h_pM : M P)
  (k1 k2 : ℝ)
  (h_ksum : k1 + k2 = 0):
  (∃ a b : ℝ, a^2 = 8 ∧ b^2 = 2 ∧ (∀ (p : ℝ × ℝ), p.1^2 / a^2 + p.2^2 / b^2 = 1 → M p)) 
  ∧ ((∀ (A B : ℝ × ℝ), line_through P (slope_of_line AB) = 1/2)  ∧ 
  (∃ (k : ℝ), (k = (area_Δ TBC / area_Δ TEF)) → k ≤ 4/3 :=
  ∃ a b a_eq b_eq, a^2 = 8 ∧ b^2 = 2 ∧ 
  (∀ (p : ℝ × ℝ), (p.1^2 / a^2 + p.2^2 / b^2 = 1 → M p)) 
  ∧ ((∀ (A B : ℝ × ℝ), (A.1 = 2 ∧ A.2 = 1 ∧ B.1 = 2 ∧ B.2 = 1) → slope_of_line AB = 1/2)
  ∧ (∃ k : ℝ, (area_Δ TBC / area_Δ TEF = k) → k = 4/3)) :=
begin
  sorry
end

end ellipse_properties_l821_821023


namespace ratio_of_areas_l821_821995

-- Definitions for regular hexagon and centers of exterior equilateral triangles
structure Hexagon (V : Type) :=
(vertices : list V)
(is_regular : ∀ i j, dist (vertices.nth_le i sorry) (vertices.nth_le j sorry) = dist (vertices.nth_le 0 sorry) (vertices.nth_le 1 sorry))

-- Main theorem for the ratio of areas
theorem ratio_of_areas 
  (V : Type) [metric_space V] 
  (H : Hexagon V) 
  (G H I J K L : V)
  (exterior_centers : ∀ v, v ∈ [G, H, I, J, K, L] → ∃ e_triangle, equilateral e_triangle ∧ center e_triangle = v ∧ OnBoundariesOfHexagon e_triangle H) :
  Area (Hexagon.mk [G, H, I, J, K, L] sorry) / Area H = 10 / 3 :=
  sorry

end ratio_of_areas_l821_821995


namespace distinct_arrangements_bubble_l821_821496

theorem distinct_arrangements_bubble : 
  let n := 6
  let r := 3
  (Nat.factorial n) / (Nat.factorial r) = 120 :=
by
  let n := 6
  let r := 3
  have fact_n : Nat.factorial n = 720 := Nat.factorial_eq 6
  have fact_r : Nat.factorial r = 6 := Nat.factorial_eq 3
  calc
    (Nat.factorial n) / (Nat.factorial r)
        = 720 / 6 := by rw [fact_n, fact_r]
    ... = 120 := by norm_num

end distinct_arrangements_bubble_l821_821496


namespace seating_arrangements_l821_821953

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821953


namespace locus_midpoint_chord_l821_821034

variable {a b c : ℝ}

theorem locus_midpoint_chord (h₁ : a - 2 * b + c = 0)
                             (h₂ : b * -2 + a + c = 0)
                             (P₀ : (-2 : ℝ), (1 : ℝ)) :
                             (∃ P₁ : ℝ × ℝ, P₁ ∈ { (x, y) | bx + ay + c = 0 ∧ y^2 = -1/2 * x } ∧
                             ∃ midpoint : ℝ × ℝ, midpoint = ((P₁.1 - 2) / 2, (P₁.2 + 1) / 2) ∧
                             midpoint = (x, y)
                             , x + 1 = -(2 * y - 1) ^ 2) :=
begin
  sorry
end

end locus_midpoint_chord_l821_821034


namespace distance_between_intersections_l821_821340

theorem distance_between_intersections :
  let y := Real.cbrt (sqrt 5 - 1) / 2 in
  let x := (y ^ 3) ^ 2 in
  let d := sqrt (0 + 2 * y) in
  d = sqrt (0 + 2 * sqrt (5 : ℝ)) → 
  (0, 2, 5)
:= sorry

end distance_between_intersections_l821_821340


namespace circumcenter_on_AD_l821_821264

noncomputable def D_inside_angle_XAY (D X A Y : Point) : Prop :=
  ... -- definition ensuring D is inside the acute angle ∠XAY

noncomputable def equal_angles (P Q R S T U : Point) : Prop :=
  ∠ QPR = ∠ STU -- definition for equal angles, symbolically (you will have to rely on the actual implementation in Lean's geometry library)

theorem circumcenter_on_AD
  {A B C D X Y : Point}
  (h1 : D_inside_angle_XAY D X A Y)
  (h2 : equal_angles A B C X B D)
  (h3 : equal_angles A C B Y C D) :
  On (circumcenter (Triangle.mk A B C)) (Segment.mk A D) :=
sorry

end circumcenter_on_AD_l821_821264


namespace constant_function_value_l821_821507

theorem constant_function_value (g : ℝ → ℝ) (h : ∀ x : ℝ, g(x) = 5) (x : ℝ) : g(3 * x - 7) = 5 :=
sorry

end constant_function_value_l821_821507


namespace xy_exists_5n_l821_821820

theorem xy_exists_5n (n : ℕ) (hpos : 0 < n) :
  ∃ x y : ℤ, x^2 + y^2 = 5^n ∧ Int.gcd x 5 = 1 ∧ Int.gcd y 5 = 1 :=
sorry

end xy_exists_5n_l821_821820


namespace func_A_is_odd_func_B_is_not_odd_func_C_domain_not_symmetric_func_D_is_not_odd_l821_821745

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def func_A (x : ℝ) : ℝ := x * Real.cos x
def func_B (x : ℝ) : ℝ := x * Real.sin x
def func_C (x : ℝ) : ℝ := Real.abs (Real.log x)
def func_D (x : ℝ) : ℝ := 2 ^ (-x)

theorem func_A_is_odd : is_odd_function func_A := by
  sorry

theorem func_B_is_not_odd : ¬ is_odd_function func_B := by
  sorry

theorem func_C_domain_not_symmetric : ¬ ∃ f : ℝ → ℝ, f = func_C ∧ ∀ x : ℝ, func_C (-x) = -func_C x := by
  sorry

theorem func_D_is_not_odd : ¬ is_odd_function func_D := by
  sorry

end func_A_is_odd_func_B_is_not_odd_func_C_domain_not_symmetric_func_D_is_not_odd_l821_821745


namespace fruit_prices_l821_821291

theorem fruit_prices :
  (∃ x y : ℝ, 60 * x + 40 * y = 1520 ∧ 30 * x + 50 * y = 1360 ∧ x = 12 ∧ y = 20) :=
sorry

end fruit_prices_l821_821291


namespace number_of_employees_correct_l821_821509

-- Definitions
def total_profit : ℝ := 50
def self_keep_percent : ℝ := 0.10
def profit_per_employee : ℝ := 5
def employees : ℝ := 9  -- This will be the value we need to prove

-- The total amount kept by you
def self_keep := self_keep_percent * total_profit

-- The remaining profit to distribute
def remaining_profit := total_profit - self_keep

-- Number of employees
def number_of_employees := remaining_profit / profit_per_employee

-- The theorem to prove
theorem number_of_employees_correct : number_of_employees = employees :=
by
  sorry

end number_of_employees_correct_l821_821509


namespace original_number_from_sum_l821_821089

variable (a b c : ℕ) (m S : ℕ)

/-- Given a three-digit number, the magician asks the participant to add all permutations -/
def three_digit_number_permutations_sum (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) +
  (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)

/-- Given the sum of all permutations of the three-digit number is 4239, determine the original number -/
theorem original_number_from_sum (S : ℕ) (hS : S = 4239) (Sum_conditions : three_digit_number_permutations_sum a b c = S) :
  (100 * a + 10 * b + c) = 429 := by
  sorry

end original_number_from_sum_l821_821089


namespace problem_1_1_problem_1_2_problem_2_l821_821410

-- Part (1)
theorem problem_1_1 (a b : ℝ) (n : ℕ) (h₁ : a = 1) (h₂ : b = 1) (h₃ : n = 6) :
  let x := 1
  in ((a*x + b) ^ n).expand.simplify.coefficients.drop 1.sum = 63 := sorry

theorem problem_1_2 (a b : ℝ) (n : ℕ) (h₁ : a = 1) (h₂ : b = 1) (h₃ : n = 6) :
  let x := 1
  in (fun m => m.coeff * m.coeff_index).sum = 192 := sorry

-- Part (2)
theorem problem_2 (a b : ℝ) (n : ℕ) (h₁ : a = 1) (h₂ : b = -sqrt(3)) (h₃ : n = 8) :
  let x := 1
  in (let coeff_even := fun (m : Polynomial ℝ) -> m.num_nonzero_terms.even_part and
         coeff_odd := fun (m : Polynomial ℝ) -> m.num_nonzero_terms.odd_part
     in (coeff_even.sum.coeff ^ 2 - coeff_odd.sum.coeff ^ 2) = 256) := sorry

end problem_1_1_problem_1_2_problem_2_l821_821410


namespace vector_magnitude_sum_l821_821033

variables (a b : EuclideanSpace ℝ (Fin 3))

noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  Real.sqrt (v.dot v)

theorem vector_magnitude_sum (h1 : magnitude a = 2)
                             (h2 : magnitude b = 1)
                             (h3 : (a - 2 • b) ⬝ (2 • a + b) = 9) :
  magnitude (a + b) = Real.sqrt 3 :=
sorry

end vector_magnitude_sum_l821_821033


namespace isosceles_triangle_sides_l821_821313

theorem isosceles_triangle_sides (a b : ℚ) (P : ℚ)
  (h1 : P = 2 * a + b)
  (h2 : ∃ A : ℚ, (A^2 = P) ∧ P = A^2)
  (h3 : b ∈ {1, 2, 4}) :
  (a = 17/2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 5/2 ∧ b = 4) :=
by
  sorry

end isosceles_triangle_sides_l821_821313


namespace problem_solution_l821_821106

def point := (ℝ × ℝ × ℝ)

def cube_side_length : ℝ := 2

def A : point := (0, 0, 0)
def B : point := (cube_side_length, 0, 0)
def D : point := (0, 0, cube_side_length)
def E : point := (0, cube_side_length, 0)
def F : point := (cube_side_length, cube_side_length, 0)
def G : point := (cube_side_length, cube_side_length, cube_side_length)
def H : point := (0, cube_side_length, cube_side_length)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def M : point := midpoint A B
def N : point := midpoint G H
def P : point := midpoint A D
def Q : point := midpoint E F

def vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def cross_product (v1 v2 : point) : point :=
  (v1.2 * v2.3 - v1.3 * v2.2,
   v1.3 * v2.1 - v1.1 * v2.3,
   v1.1 * v2.2 - v1.2 * v2.1)

def magnitude (v : point) : ℝ :=
  (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2).sqrt

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℝ :=
  (magnitude (cross_product (vector p1 p2) (vector p3 p4))) / 2

def surface_area_of_cube (side_length : ℝ) : ℝ :=
  6 * (side_length ^ 2)

def ratio_S : ℝ :=
  let area_MNPQ := area_of_quadrilateral M N P Q
  let total_surface_area := surface_area_of_cube cube_side_length
  area_MNPQ / total_surface_area

theorem problem_solution : ratio_S = (61.sqrt) / 48 := by
  sorry

end problem_solution_l821_821106


namespace san_antonio_to_austin_buses_l821_821329

theorem san_antonio_to_austin_buses :
  ∀ (departure_interval_austin_san_antonio : ℕ) 
    (departure_interval_san_antonio_austin : ℕ) 
    (trip_duration : ℕ) 
    (number_of_passed_buses : ℕ), 
  (departure_interval_austin_san_antonio = 30) →
  (departure_interval_san_antonio_austin = 45) →
  (trip_duration = 4 * 60) →
  (number_of_passed_buses = 10) :=
begin
  intros,
  sorry,
end

end san_antonio_to_austin_buses_l821_821329


namespace part1_part2_part3_l821_821283

-- Define conditions
variables (n : ℕ) (h₁ : 5 ≤ n)

-- Problem part (1): Define p_n and prove its value
def p_n (n : ℕ) := (10 * n) / ((n + 5) * (n + 4))

-- Problem part (2): Define EX and prove its value for n = 5
def EX : ℚ := 5 / 3

-- Problem part (3): Prove n = 20 maximizes P
def P (n : ℕ) := 3 * ((p_n n) ^ 3 - 2 * (p_n n) ^ 2 + (p_n n))
def n_max := 20

-- Making the proof skeletons for clarity, filling in later
theorem part1 : p_n n = 10 * n / ((n + 5) * (n + 4)) :=
sorry

theorem part2 (h₂ : n = 5) : EX = 5 / 3 :=
sorry

theorem part3 : n_max = 20 :=
sorry

end part1_part2_part3_l821_821283


namespace sum_first_n_terms_l821_821224

variable {a : ℕ → ℝ}

-- Conditions
axiom seq_condition (n : ℕ) (hn : n > 0) : ∑ k in Finset.range n, (k + 1) * (a (k + 1)) = 4 - (n + 2) / (2 ^ (n - 1))

-- Target
theorem sum_first_n_terms (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range n, a (k + 1)) = 2 * (1 - (1 / (2 ^ n))) :=
sorry

end sum_first_n_terms_l821_821224


namespace dara_jane_age_ratio_l821_821217

theorem dara_jane_age_ratio :
  ∀ (min_age : ℕ) (jane_current_age : ℕ) (dara_years_til_min_age : ℕ) (d : ℕ) (j : ℕ),
  min_age = 25 →
  jane_current_age = 28 →
  dara_years_til_min_age = 14 →
  d = 17 →
  j = 34 →
  d = dara_years_til_min_age - 14 + 6 →
  j = jane_current_age + 6 →
  (d:ℚ) / j = 1 / 2 := 
by
  intros
  sorry

end dara_jane_age_ratio_l821_821217


namespace milo_ingredients_l821_821158

noncomputable def number_of_dozen_eggs_needed (total_weight_in_pounds : ℝ) (weight_per_egg : ℝ) (eggs_per_dozen : ℝ) : ℝ :=
  (total_weight_in_pounds / weight_per_egg) / eggs_per_dozen

noncomputable def weight_of_flour_needed (total_weight_in_kg : ℝ) (pounds_per_kg : ℝ) : ℝ :=
  total_weight_in_kg * pounds_per_kg

noncomputable def number_of_cups_of_milk_needed (total_volume_in_ml : ℝ) (volume_per_cup_ml : ℝ) : ℝ :=
  total_volume_in_ml / volume_per_cup_ml

theorem milo_ingredients (eggs_in_pounds : ℝ) (flour_in_kg : ℝ) (milk_in_ml : ℝ)
    (weight_per_egg : ℝ) (eggs_per_dozen : ℝ) (pounds_per_kg : ℝ) (volume_per_cup_ml : ℝ) :
    number_of_dozen_eggs_needed eggs_in_pounds weight_per_egg eggs_per_dozen = 8 ∧
    weight_of_flour_needed flour_in_kg pounds_per_kg = 7.71617 ∧
    (number_of_cups_of_milk_needed milk_in_ml volume_per_cup_ml).ceil = 3 :=
by
  sorry

/- Definitions from the conditions -/
def eggs_in_pounds : ℝ := 6
def flour_in_kg : ℝ := 3.5
def milk_in_ml : ℝ := 500
def weight_per_egg : ℝ := 1 / 16
def eggs_per_dozen : ℝ := 12
def pounds_per_kg : ℝ := 2.20462
def volume_per_cup_ml : ℝ := 236.588

#check @milo_ingredients eggs_in_pounds flour_in_kg milk_in_ml weight_per_egg eggs_per_dozen pounds_per_kg volume_per_cup_ml

end milo_ingredients_l821_821158


namespace area_of_sector_radius_2_angle_90_l821_821222

-- Given conditions
def radius := 2
def central_angle := 90

-- Required proof: the area of the sector with given conditions equals π.
theorem area_of_sector_radius_2_angle_90 : (90 * Real.pi * (2^2) / 360) = Real.pi := 
by
  sorry

end area_of_sector_radius_2_angle_90_l821_821222


namespace factor_x_squared_minus_64_l821_821796

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821796


namespace vector_rotation_correct_l821_821657

def vector_rotate_z_90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v
  ( -y, x, z )

theorem vector_rotation_correct :
  vector_rotate_z_90 (3, -1, 4) = (-3, 0, 4) := 
by 
  sorry

end vector_rotation_correct_l821_821657


namespace lambda_parallel_l821_821452

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821452


namespace root_product_evaluation_l821_821565

open Polynomial

theorem root_product_evaluation :
  ∀ a b c : ℝ,
  (is_root (X^3 - 15 * X^2 + 25 * X - 10) a) ∧
  (is_root (X^3 - 15 * X^2 + 25 * X - 10) b) ∧
  (is_root (X^3 - 15 * X^2 + 25 * X - 10) c) →
  (1 + a) * (1 + b) * (1 + c) = 51 :=
by
  intros a b c h
  sorry

end root_product_evaluation_l821_821565


namespace parallel_vectors_lambda_l821_821468

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821468


namespace price_reduction_l821_821736

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end price_reduction_l821_821736


namespace prime_divides_sum_l821_821145

theorem prime_divides_sum {p : ℕ} (hp : Prime p) : 
  ∀ k : ℤ, p ∣ (Finset.range (p - 1)).sum (λ i, (i + 1 : ℤ)^k) :=
by
  sorry

end prime_divides_sum_l821_821145


namespace minimum_ab_l821_821503

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end minimum_ab_l821_821503


namespace parallel_vectors_lambda_l821_821443

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821443


namespace sqrt_expression_value_l821_821341

-- Definition for trigonometric identities
def cos_add (a b : ℝ) : ℝ := Real.cos (a + b)
def sin_sub (a b : ℝ) : ℝ := Real.sin (a - b)

-- Main problem statement
theorem sqrt_expression_value :
  ∃ (x : ℝ), x = Real.sin 3 + Real.cos 3 ∧
  √(1 - 2 * cos_add (Real.pi / 2) 3 * sin_sub (Real.pi / 2) 3) = x :=
begin
  sorry
end

end sqrt_expression_value_l821_821341


namespace teams_dig_tunnel_in_10_days_l821_821782

theorem teams_dig_tunnel_in_10_days (hA : ℝ) (hB : ℝ) (work_A : hA = 15) (work_B : hB = 30) : 
  (1 / (1 / hA + 1 / hB)) = 10 := 
by
  sorry

end teams_dig_tunnel_in_10_days_l821_821782


namespace price_per_yellow_stamp_l821_821581

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l821_821581


namespace license_plates_count_l821_821064

def num_license_plates : Nat :=
  let num_letters := 26
  let num_digits_first := 10
  let num_digits_second := 5
  num_letters * num_letters * num_digits_first * num_digits_second

theorem license_plates_count : num_license_plates = 33800 := by
  calc 
    num_license_plates
        = 26 * 26 * 10 * 5 : rfl
    ... = 676 * 10 * 5 : by norm_num
    ... = 6760 * 5 : by norm_num
    ... = 33800 : by norm_num

end license_plates_count_l821_821064


namespace four_digit_numbers_divisible_by_5_l821_821497

theorem four_digit_numbers_divisible_by_5 : 
  (finset.filter (λ x, x % 5 = 0) (finset.range 10000 \ finset.range 1000)).card = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l821_821497


namespace T_n_plus_12_eq_l821_821396

-- Given the conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = a 1 + (n - 1) * 3

def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b n = b 1 * (2 ^ (n - 1))

def S_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + (n - 1) * 3 / 2)

axiom a1_b1_eq_2 : a 1 = 2 ∧ b 1 = 2
axiom a4_b4_eq_27 : a 4 + b 4 = 27
axiom S4_b4_eq_10 : S_n a 4 - b 4 = 10

-- Define the function T_n
def T_n (a b : ℕ → ℤ) (n : ℕ) : ℤ :=
  (finset.range n).sum (λ i, a (n - i) * b (i + 1))

-- Define the proof statement
theorem T_n_plus_12_eq (a b : ℕ → ℤ) (n : ℕ) [arithmetic_sequence a] [geometric_sequence b] :
  T_n a b n + 12 = -2 * a n + 10 * b n := by
  sorry

end T_n_plus_12_eq_l821_821396


namespace range_of_m_iff_l821_821886

noncomputable def range_of_m (m : ℝ) : Prop :=
  (∀ x : ℝ, (sin x + cos x > m) ∧ ¬ (x ^ 2 + m * x + 1 > 0)) ∨
  (∀ x : ℝ, ¬ (sin x + cos x > m) ∧ (x ^ 2 + m * x + 1 > 0))

theorem range_of_m_iff (m : ℝ) :
  range_of_m m ↔ m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2) :=
sorry

end range_of_m_iff_l821_821886


namespace find_b_l821_821197

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (317212435 * 101 - b) % 25 = 0 ∧ b = 13 := by
  sorry

end find_b_l821_821197


namespace find_ab_l821_821400

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by 
  sorry

end find_ab_l821_821400


namespace valid_marble_arrangements_eq_48_l821_821265

def ZaraMarbleArrangements (n : ℕ) : ℕ := sorry

theorem valid_marble_arrangements_eq_48 : ZaraMarbleArrangements 5 = 48 := sorry

end valid_marble_arrangements_eq_48_l821_821265


namespace fixed_points_subset_stable_points_fixed_point_quadratic_fixed_points_stable_points_quadratic_l821_821371

-- Definitions
def fixed_points (f : ℝ → ℝ) : set ℝ := {x | f x = x}
def stable_points (f : ℝ → ℝ) : set ℝ := {x | f (f x) = x}

-- 1. Prove that A ⊆ B
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : fixed_points f ⊆ stable_points f :=
begin
  sorry
end

-- 2. Given f(x) = x^2 + bx + c + 1 always has a fixed point for all b ∈ ℝ, then c ≤ -1
theorem fixed_point_quadratic (c : ℝ) (h : ∀ b : ℝ, ∃ x : ℝ, x^2 + b * x + c + 1 = x) : c ≤ -1 :=
begin
  sorry
end

-- 3. Given f(x) = ax^2 - 1 and A = B ≠ ∅, then -1/4 ≤ a ≤ 3/4
theorem fixed_points_stable_points_quadratic (a : ℝ) 
(hA : fixed_points (λ x : ℝ, a * x^2 - 1) = stable_points (λ x : ℝ, a * x^2 - 1)) 
(hA_nonempty : fixed_points (λ x : ℝ, a * x^2 - 1) ≠ ∅) : 
-1/4 ≤ a ∧ a ≤ 3/4 :=
begin
  sorry
end

end fixed_points_subset_stable_points_fixed_point_quadratic_fixed_points_stable_points_quadratic_l821_821371


namespace lambda_parallel_l821_821454

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821454


namespace parallel_vectors_lambda_l821_821448

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821448


namespace find_m_of_quadratic_function_l821_821634

theorem find_m_of_quadratic_function :
  ∀ (m : ℝ), (m + 1 ≠ 0) → ((m + 1) * x ^ (m^2 + 1) + 5 = a * x^2 + b * x + c) → m = 1 :=
by
  intro m h h_quad
  -- Proof Here
  sorry

end find_m_of_quadratic_function_l821_821634


namespace total_pencils_l821_821764

theorem total_pencils : 
  let pencils_per_person := 7
  let num_friends := 5
  in pencils_per_person + num_friends * pencils_per_person = 42 :=
by
  let pencils_per_person := 7
  let num_friends := 5
  show pencils_per_person + num_friends * pencils_per_person = 42
  sorry

end total_pencils_l821_821764


namespace shop_owner_profit_l821_821691

theorem shop_owner_profit :
  ∀ (buy_weight sell_weight price_per_100g : ℚ),
  (buy_weight = 100 * (100 - 12) / 100) →
  (sell_weight = 100 * (100 - 20) / 100) →
  (price_per_100g = 100) →
  let cp_per_gram := price_per_100g / buy_weight in
  let sp_per_gram := price_per_100g / sell_weight in
  let profit_per_gram := sp_per_gram - cp_per_gram in
  let percentage_profit := (profit_per_gram / cp_per_gram) * 100 in
  percentage_profit ≈ 10 := 
begin
  intros,
  sorry
end

end shop_owner_profit_l821_821691


namespace midpoint_quadrilateral_area_ratio_l821_821216

theorem midpoint_quadrilateral_area_ratio (Q : Quadrilateral) (h_convex : convex Q) :
  let Q' := new_quadrilateral_by_joining_midpoints Q in
  area Q' = (1 / 4) * area Q :=
sorry

end midpoint_quadrilateral_area_ratio_l821_821216


namespace percentage_of_water_in_fresh_grapes_l821_821008

-- Conditions
def weight_of_fresh_grapes : Real := 100
def weight_of_dried_grapes : Real := 33.33333333333333
def water_content_in_dried_grapes : Real := 0.1

-- Prove that the percentage of water in fresh grapes is 70%
theorem percentage_of_water_in_fresh_grapes (W : Real) :
  (weight_of_fresh_grapes - W) = 0.9 * weight_of_dried_grapes → W = 70 :=
by
  intro h
  -- conditions given in the problem imply this transformation
  have h1 : weight_of_fresh_grapes - W = 100 - W := by rfl
  rw [h1] at h
  have h2 : 0.9 * weight_of_dried_grapes = 30 := by norm_num1
  rw [h2] at h
  assumption -- W = 70 follows directly from the equation
  sorry

end percentage_of_water_in_fresh_grapes_l821_821008


namespace probability_obtuse_is_correct_l821_821171

/-- Define the vertices of the rectangle -/
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (7, 0)
def C : (ℝ × ℝ) := (7, 4)
def D : (ℝ × ℝ) := (0, 4)

/-- Define the center and radius of the semicircle described in the problem -/
def center : (ℝ × ℝ) := ((0+7)/2, (3+0)/2)
def radius : ℝ := real.sqrt ((3.5 - 0)^2 + (1.5 - 3)^2)

/-- Define the area of the semicircle -/
def area_semicircle : ℝ := (1/2) * real.pi * radius^2

/-- Define the area of the rectangle -/
def area_rectangle : ℝ := (7 - 0) * (4 - 0)

/-- Define the probability that angle APB is obtuse -/
def probability_obtuse : ℝ := area_semicircle / area_rectangle

/-- Theorem stating the probability obtuse is as calculated -/
theorem probability_obtuse_is_correct : probability_obtuse = 22.852 / 28 := by
  sorry

end probability_obtuse_is_correct_l821_821171


namespace divide_groups_distribute_people_evenly_divide_groups_l821_821660

noncomputable def binom : ℕ → ℕ → ℕ 
| n, k => if k > n then 0 else n.choose k

theorem divide_groups (books : ℕ) 
  (h1 : books = 9) : binom 9 4 * binom 5 3 = 1260 := by
  sorry

theorem distribute_people (books : ℕ) 
  (h1 : books = 9) : 3! * binom 9 4 * binom 5 3 = 7560 := by
  sorry

theorem evenly_divide_groups (books : ℕ) 
  (h1 : books = 9) : (binom 9 3 * binom 6 3) / 3! = 280 := by
  sorry

end divide_groups_distribute_people_evenly_divide_groups_l821_821660


namespace students_participating_cost_effective_plan_l821_821292

-- Define the number of students participating in the field trip
def number_of_students (x : ℕ) : Prop :=
  35 * x = 45 * (x - 1) - 25

-- Define the rental cost calculation
def rental_cost (a : ℕ) (b : ℕ) : ℕ :=
  320 * a + 380 * b

-- Define the condition that the total number of seats must be at least 245
def seats_condition (a b : ℕ) : Prop :=
  35 * a + 45 * b ≥ 245

-- Define the total number of buses
def total_buses (a b : ℕ) : Prop :=
  a + b = 6

-- Theorem for the number of students
theorem students_participating : ∃ x : ℕ, number_of_students x → 35 * x = 245 :=
by
  -- Proof omitted
  sorry

-- Theorem for the most cost-effective rental plan
theorem cost_effective_plan (a b : ℕ) :
  total_buses a b ∧ seats_condition a b ∧ rental_cost a b ≤ ∀ a' b', total_buses a' b' ∧ seats_condition a' b' → rental_cost a' b' := 
by 
  -- Proof omitted
  sorry

end students_participating_cost_effective_plan_l821_821292


namespace ball_distribution_l821_821170

theorem ball_distribution :
  (∃ f : Fin 3 → ℕ, (∑ i, f i) = 10 ∧ f 0 ≥ 1 ∧ f 1 ≥ 2 ∧ f 2 ≥ 3) ↔ 15 :=
by
  sorry

end ball_distribution_l821_821170


namespace arithmetic_sequence_problem_l821_821391

theorem arithmetic_sequence_problem (q a₁ a₂ a₃ : ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : q > 1)
  (h2 : a₁ + a₂ + a₃ = 7)
  (h3 : a₁ + 3 + a₃ + 4 = 6 * a₂) :
  (∀ n : ℕ, a n = 2^(n-1)) ∧ (∀ n : ℕ, T n = (3 * n - 5) * 2^n + 5) :=
by
  sorry

end arithmetic_sequence_problem_l821_821391


namespace joe_anne_bill_difference_l821_821344

theorem joe_anne_bill_difference (m j a : ℝ) 
  (hm : (15 / 100) * m = 3) 
  (hj : (10 / 100) * j = 2) 
  (ha : (20 / 100) * a = 3) : 
  j - a = 5 := 
by {
  sorry
}

end joe_anne_bill_difference_l821_821344


namespace carolyn_marbles_l821_821763

theorem carolyn_marbles (start_marbles share_marbles : ℕ) (start_oranges : ℕ) 
(h1 : start_marbles = 47) (h2 : start_oranges = 6) (h3 : share_marbles = 42) : 
(start_marbles - share_marbles) = 5 :=
by 
  -- Given starting marbles and sharing marbles, prove the final count is 5
  rw [h1, h3]
  exact rfl

end carolyn_marbles_l821_821763


namespace simplify_and_evaluate_expression_l821_821188

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 2) : 
  (1 / (x - 3) / (1 / (x^2 - 9)) - x / (x + 1) * ((x^2 + x) / x^2)) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l821_821188


namespace non_equilateral_triangle_cover_l821_821175

theorem non_equilateral_triangle_cover 
  (ABC : Triangle) 
  (h_non_equilateral : ¬(ABC.is_equilateral)) : 
  ∃ (AB'C' DBE: Triangle), 
    (AB'C'.is_similar ABC ∧ ¬(AB'C' = ABC) ∧ AB'C'.covers_part ABC) ∧ 
    (DBE.is_similar ABC ∧ ¬(DBE = ABC) ∧ DBE.covers_part ABC) ∧ 
    (AB'C'.area + DBE.area = ABC.area) := 
sorry

end non_equilateral_triangle_cover_l821_821175


namespace a_2019_sequence_problem_l821_821653

theorem a_2019_sequence_problem (f : ℝ → ℝ) (a₁ : ℕ) (a : ℝ) 
  (h₀ : a₁ = 1)
  (h₁ : ∀ x, f x = a * x / (x + 1))
  (h₂ : ∀ n, (1 : ℝ) / a₁ = f ((1 : ℝ) / (a₁ : ℝ)))
  (h₃ : ∃ x, f x = x) :
  let a : ℕ → ℝ := λ n, n
  in a 2019 = 2019 :=
by
  sorry

end a_2019_sequence_problem_l821_821653


namespace secant_segments_equal_l821_821663

theorem secant_segments_equal
  (P Q A B C D : ℝ)
  (circle1 circle2 : set ℂ)
  (h1 : P ∈ circle1)
  (h2 : Q ∈ circle1)
  (h3 : P ∈ circle2)
  (h4 : Q ∈ circle2)
  (AB : line ℂ)
  (CD : line ℂ)
  (h5 : ∀ z, z ∈ AB → is_secant z circle1 ∧ is_secant z circle2)
  (h6 : ∀ z, z ∈ CD → is_secant z circle1 ∧ is_secant z circle2)
  (h7: parallel AB CD) :
  enclosed_segment_length circle1 circle2 AB = enclosed_segment_length circle1 circle2 CD :=
sorry

end secant_segments_equal_l821_821663


namespace amc10_paths_count_l821_821528

/-- 
Define the structure of our problem in Lean.

Consider a grid where:
  - Each letter in "AMC10" can be reached via adjacent moves from a central letter 'A'.
  - There are specific counts of reachable letters next to each step. 
-/
theorem amc10_paths_count : 
  (calculate_amc10_paths (center_A)) = 40 :=
sorry

namespace PathCounting

def center_A := ... -- Define the central 'A' position in the grid
def calculate_amc10_paths (start : Position) : Nat :=
  let numM := 4 -- 4 possible 'M's adjacent to the central 'A'
  let first_type_paths := numM / 2 * 2 * 1 * 2
  let second_type_paths := numM / 2 * 3 * 1 * 2
  first_type_paths + second_type_paths

end PathCounting

end amc10_paths_count_l821_821528


namespace line_through_points_has_slope_and_intercept_l821_821915

-- Define the points (3,2) and (7,14)
def point1 : ℝ × ℝ := (3, 2)
def point2 : ℝ × ℝ := (7, 14)

-- We now state the main theorem to be proved
theorem line_through_points_has_slope_and_intercept :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst),
      c := point1.snd - m * point1.fst in
  m - c = 10 :=
by
  -- Dummy proof, replace with actual proof
  sorry

end line_through_points_has_slope_and_intercept_l821_821915


namespace integral_solution_l821_821750

noncomputable def gaussian_integral : ℝ :=
∫ x : ℝ in -∞..∞, real.exp (-x^2)

theorem integral_solution :
  (∫ x : ℝ in 0..∞, x^(-1/2) * real.exp (-1985 * (x + 1/x))) = 
  real.sqrt (real.pi / 1985) * real.exp (-3970) := 
by
  have h1 : gaussian_integral = real.sqrt real.pi := sorry
  sorry

end integral_solution_l821_821750


namespace tax_authority_correct_l821_821585

/-- 
Proof problem: Given the conditions, prove that the tax authority's claim (that 
the agreement between Mikhail and Valentin violated legislation) is correct.
-/
theorem tax_authority_correct
  (Mikhail_Vasilievich : Prop) 
  (Valentin_Pavlovich : Prop) 
  (purchase_in_euros : Mikhail_Vasilievich → Valentin_Pavlovich → ℝ → Prop)
  (amount_euros : ℝ)
  (resident_Mikhail : Prop)
  (resident_Valentin : Prop)
  (Federal_Law_173_FZ : Prop)
  (prohibited_currency_transactions : Federal_Law_173_FZ → Prop)
  (exceptions_not_apply : Prohibited_currency_transactions → Federal_Law_173_FZ → Prop) :
  Mikhail_Vasilievich ∧ Valentin_Pavlovich ∧ resident_Mikhail ∧ resident_Valentin ∧ 
  purchase_in_euros Mikhail_Vasilievich Valentin_Pavlovich amount_euros →
  prohibited_currency_transactions Federal_Law_173_FZ →
  exceptions_not_apply prohibited_currency_transactions Federal_Law_173_FZ →
  (∃ notification_from_tax_authority : Prop, notification_from_tax_authority) :=
by
  sorry

end tax_authority_correct_l821_821585


namespace range_of_k_l821_821833

def f (x : ℝ) : ℝ

axiom f_deriv_nonzero : ∀ x : ℝ, deriv f x ≠ 0
axiom f_equation : ∀ x : ℝ, f (f x - 2017^x) = 2017

def g (x : ℝ) (k : ℝ) : ℝ := sin x - cos x - k * x
def g_deriv (x : ℝ) (k : ℝ) : ℝ := cos x + sin x - k

theorem range_of_k : {k : ℝ | ∀ x : ℝ, x ∈ Icc (-π/2) (π/2) → g_deriv x k ≥ 0} = {k : ℝ | k ≤ -1} :=
sorry

end range_of_k_l821_821833


namespace sum_squares_and_products_of_nonneg_reals_l821_821922

theorem sum_squares_and_products_of_nonneg_reals {x y z : ℝ} 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) 
  (h2 : x*y + y*z + z*x = 27) : 
  x + y + z = Real.sqrt 106 := 
by 
  sorry

end sum_squares_and_products_of_nonneg_reals_l821_821922


namespace find_lambda_l821_821433

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821433


namespace distance_between_towns_l821_821300

theorem distance_between_towns 
  (V_express : ℝ) -- Speed of the express train
  (V_freight : ℝ) -- Speed of the freight train
  (Time : ℝ) -- Time after which they pass each other
  (V_express_equals : V_express = 80)
  (V_freight_equals : V_freight = V_express - 30)
  (Time_equals : Time = 3) :
  let V_relative := V_express + V_freight 
  in V_relative * Time = 390 := 
by 
  have V_exp: V_express = 80 := V_express_equals
  have V_fre: V_freight = V_express - 30 := V_freight_equals
  have t: Time = 3 := Time_equals
  have V_rel: V_relative = V_express + V_freight
    by rw [V_fre]; simp [V_exp]
  show V_relative * Time = 390
  by rw [t, V_rel]; sorry 

end distance_between_towns_l821_821300


namespace sin_alpha_of_cube_plane_intersection_l821_821834

theorem sin_alpha_of_cube_plane_intersection
    (b : ℝ) -- side length of the cube
    (alpha : ℝ) -- angle α given in the problem
    (H : ∀ (face : Fin 6), ∃ (plane_angle : ℝ), plane_angle = alpha) :
  sin alpha = (√3) / 3 :=
by
  sorry

end sin_alpha_of_cube_plane_intersection_l821_821834


namespace car_rental_cost_per_mile_l821_821712

def daily_rental_rate := 29.0
def total_amount_paid := 46.12
def miles_driven := 214.0

theorem car_rental_cost_per_mile : 
  (total_amount_paid - daily_rental_rate) / miles_driven = 0.08 := 
by
  sorry

end car_rental_cost_per_mile_l821_821712


namespace geometric_progression_solution_l821_821809

theorem geometric_progression_solution 
  (b1 q : ℝ)
  (condition1 : (b1^2 / (1 + q + q^2) = 48 / 7))
  (condition2 : (b1^2 / (1 + q^2) = 144 / 17)) 
  : (b1 = 3 ∨ b1 = -3) ∧ q = 1 / 4 :=
by
  sorry

end geometric_progression_solution_l821_821809


namespace problem_part_I_problem_part_II_l821_821405

noncomputable def f (ω ϕ x : ℝ) : ℝ :=
  sin (ω * x + ϕ) + sqrt 3 * cos (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def symmetry_axis_distance (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, f (x + d) = f x

theorem problem_part_I (ω ϕ : ℝ) (h_odd : is_odd (f ω ϕ)) (h_distance : symmetry_axis_distance (f ω ϕ) (π / 2)) :
  f ω ϕ (π / 6) = sqrt 3 :=
sorry

noncomputable def g (ω ϕ x : ℝ) : ℝ :=
  f ω ϕ (x - π / 6)

theorem problem_part_II (ω ϕ : ℝ) (h_odd : is_odd (f ω ϕ)) (h_distance : symmetry_axis_distance (f ω ϕ) (π / 2)) :
  ∀ k : ℤ, ∃ a b : ℝ, (g ω ϕ) is_monotonically_increasing_on (set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) :=
sorry

end problem_part_I_problem_part_II_l821_821405


namespace factor_difference_of_squares_l821_821788

theorem factor_difference_of_squares (x : ℝ) : (x^2 - 64 = (x - 8) * (x + 8)) := by
  -- Conditions to state the problem with
  let a := x
  let b := 8
  have h1 : (x^2 - 64) = (a^2 - b^2), by
    rw [a, b]
  have h2 : (a^2 - b^2) = (a - b) * (a + b), from sorry
  -- Final equivalence
  exact (h1.trans h2)
  sorry
  -- Final proof is left as sorry.

end factor_difference_of_squares_l821_821788


namespace geom_seq_product_l821_821540

theorem geom_seq_product {a : ℕ → ℝ} (h_geom : ∀ n, a (n + 1) = a n * r)
 (h_a1 : a 1 = 1 / 2) (h_a5 : a 5 = 8) : a 2 * a 3 * a 4 = 8 := 
sorry

end geom_seq_product_l821_821540


namespace trig_equation_solution_l821_821688

theorem trig_equation_solution (x : ℝ) :
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (Real.pi / 12) * (6 * n + 1) ∨ x = (Real.pi / 12) * (6 * n - 1)) ↔
  sin (2 * x) * sin (6 * x) * cos (4 * x) + (1 / 4) * cos (12 * x) = 0 :=
sorry

end trig_equation_solution_l821_821688


namespace inequality_proof_l821_821847

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821847


namespace quadratic_has_two_roots_l821_821825

theorem quadratic_has_two_roots (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 5 * a + b + 2 * c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 := 
  sorry

end quadratic_has_two_roots_l821_821825


namespace fiveLetterWordsWithAtLeastOneVowel_l821_821892

-- Definitions for the given conditions
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Total number of 5-letter words with no restrictions
def totalWords := 6^5

-- Total number of 5-letter words containing no vowels
def noVowelWords := 4^5

-- Prove that the number of 5-letter words with at least one vowel is 6752
theorem fiveLetterWordsWithAtLeastOneVowel : (totalWords - noVowelWords) = 6752 := by
  sorry

end fiveLetterWordsWithAtLeastOneVowel_l821_821892


namespace monthly_savings_correct_l821_821623

-- Define each component of the income and expenses
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Define the main theorem
theorem monthly_savings_correct :
  let I := parents_salary + grandmothers_pension + sons_scholarship in
  let E := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses in
  let Surplus := I - E in
  let Deposit := (Surplus * 10) / 100 in
  let AmountSetAside := Surplus - Deposit in
  AmountSetAside = 16740 :=
by sorry

end monthly_savings_correct_l821_821623


namespace chairs_around_table_l821_821753

theorem chairs_around_table 
  (h1 : ∃ a b : ℕ, a ≠ b ∧ abs (a - b) = 2)
  (h2 : ∃ b c : ℕ, b ≠ c ∧ abs (b - c) = 6)
  (h3 : ∃ d e : ℕ, d ≠ e ∧ abs (d - e) = 4)
  (h4 : ∃ d c : ℕ, d ≠ c ∧ abs (d - c) = 3)
  (h5 : ∃ e b : ℕ, e ≠ b ∧ abs (e - b) = 4 ∨ abs (e - b) = 8) :
  ∃ n : ℕ, n = 12 :=
sorry

end chairs_around_table_l821_821753


namespace alpha_inequality_equality_holds_l821_821386

open Nat

theorem alpha_inequality (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_n : n ≥ 2)
  (α : ℕ) (h_alpha1 : p^α ∣ ∏ i in finset.range (n-1), (choose n i))
  (h_alpha2 : ¬ p^(α+1) ∣ ∏ i in finset.range (n-1), (choose n i)) :
  α ≤ n * (Nat.log p n + 1) - (n - 1) / (p - 1) := sorry

theorem equality_holds (p : ℕ) (t : ℕ) (h_prime : Prime p) (h_t : t > 0) :
  let n := p^t
  in n * (Nat.log p n + 1) - (n - 1) / (p - 1) = α ↔ n = p^t := sorry

end alpha_inequality_equality_holds_l821_821386


namespace range_of_z3_l821_821529

noncomputable def sqrt2 : ℝ := real.sqrt 2

variables (z1 z2 z3 : ℂ)

def condition1 := abs z1 = sqrt2 ∧ abs z2 = sqrt2

def condition2 := re (z1 * conj z2) = 0 

def condition3 := abs (z1 + z2 - z3) = 2

theorem range_of_z3 (h1 : condition1 z1 z2) (h2 : condition2 z1 z2) (h3 : condition3 z1 z2 z3) :
  0 ≤ abs z3 ∧ abs z3 ≤ 4 :=
sorry

end range_of_z3_l821_821529


namespace income_expenditure_ratio_l821_821212

theorem income_expenditure_ratio (I E S : ℝ) (h1 : I = 20000) (h2 : S = 4000) (h3 : S = I - E) :
    I / E = 5 / 4 :=
sorry

end income_expenditure_ratio_l821_821212


namespace total_pencils_l821_821606

theorem total_pencils (num_colors_in_rainbow : ℕ) (serenity_pencils friends_pencils : ℕ) :
  num_colors_in_rainbow = 7 →
  serenity_pencils = num_colors_in_rainbow →
  friends_pencils = num_colors_in_rainbow →
  (serenity_pencils + 2 * friends_pencils) = 21 :=
by
  intros hnum hser hfriends
  rw [hnum] at *
  rw [hser, hfriends]
  norm_num

end total_pencils_l821_821606


namespace f_is_periodic_with_period_2a_example_function_satisfies_equation_a_eq_1_l821_821272

noncomputable def f_periodic (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) :=
  ∀ x, f (x + a) = 0.5 + sqrt (f x - (f x)^2)

theorem f_is_periodic_with_period_2a (f : ℝ → ℝ) (a : ℝ) (ha : a > 0) 
  (h : ∀ x, f (x + a) = 0.5 + sqrt (f x - (f x)^2)) : 
  ∃ b > 0, ∀ x, f (x + b) = f x := 
by
  use 2 * a
  sorry

noncomputable def example_function_a_eq_1 : ℝ → ℝ :=
  λ x, 0.5 + 0.5 * |Real.sin (Real.pi * x / 2)|

theorem example_function_satisfies_equation_a_eq_1 :
  ∀ x, example_function_a_eq_1 (x + 1) = 0.5 + sqrt (example_function_a_eq_1 x - (example_function_a_eq_1 x)^2) :=
by
  sorry

end f_is_periodic_with_period_2a_example_function_satisfies_equation_a_eq_1_l821_821272


namespace maximum_marks_l821_821165

theorem maximum_marks (M : ℝ) (h : 0.5 * M = 50 + 10) : M = 120 :=
by
  sorry

end maximum_marks_l821_821165


namespace PetrovFamilySavings_l821_821616

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l821_821616


namespace fraction_girls_l821_821544

-- Definitions for the conditions
variable (g b : ℕ) -- representing the number of girls and boys respectively
variable (h_eq : g = b) -- condition: same number of girls and boys
variable (h_girls_trip : ℝ := 3 / 4) -- fraction of girls who went on the trip
variable (h_boys_trip : ℝ := 2 / 3) -- fraction of boys who went on the trip

-- Definition of the required fraction
def fraction_girls_on_trip := (3 / 4 * g) / (3 / 4 * g + 2 / 3 * b)

-- The theorem to be proven
theorem fraction_girls (h_eq : g = b) : fraction_girls_on_trip g b = 9 / 17 :=
by 
  unfold fraction_girls_on_trip
  rw [h_eq, ←mul_comm g, ←mul_comm b]
  sorry

end fraction_girls_l821_821544


namespace eccentricity_ellipse_eq_sqrt3_minus_1_l821_821853

noncomputable def eccentricity_of_ellipse (P F1 F2 : ℝ × ℝ) (angle_PF1F2 : ℝ) (distance_PF2_PF1_ratio : ℝ) : ℝ :=
  let m := dist P F1 in
  let sqrt3_m := dist P F2 in
  let a := (sqrt3_m + m) / 2 in
  let e := sqrt3_m / a - 1 in
  e

theorem eccentricity_ellipse_eq_sqrt3_minus_1 (P F1 F2 : ℝ × ℝ)
  (h_angle : ∠ P F1 F2 = 60) (h_distance_ratio : dist P F2 = Real.sqrt 3 * dist P F1) :
  eccentricity_of_ellipse P F1 F2 60 (Real.sqrt 3) = Real.sqrt 3 - 1 :=
sorry

end eccentricity_ellipse_eq_sqrt3_minus_1_l821_821853


namespace ratio_traditionalists_progressives_l821_821714

variables (T P C : ℝ)

-- Conditions from the problem
-- There are 6 provinces and each province has the same number of traditionalists
-- The fraction of the country that is traditionalist is 0.6
def country_conditions (T P C : ℝ) :=
  (6 * T = 0.6 * C) ∧
  (C = P + 6 * T)

-- Theorem that needs to be proven
theorem ratio_traditionalists_progressives (T P C : ℝ) (h : country_conditions T P C) :
  T / P = 1 / 4 :=
by
  -- Setup conditions from the hypothesis h
  rcases h with ⟨h1, h2⟩
  -- Start the proof (Proof content is not required as per instructions)
  sorry

end ratio_traditionalists_progressives_l821_821714


namespace matrix_inverse_sum_l821_821638

open Matrix

def A (x y z w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x, 2, y; 3, 3, 4; z, 6, w]

def B (j k l m : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![-6, j, -12; k, -14, l; 3, m, 5]

theorem matrix_inverse_sum (x y z w j k l m : ℝ)
  (h : A x y z w ⬝ B j k l m = 1) : x + y + z + w + j + k + l + m = 52 :=
by
  sorry

end matrix_inverse_sum_l821_821638


namespace volume_difference_l821_821319

noncomputable def volume (r h : ℝ) : ℝ := π * r^2 * h

theorem volume_difference (h : ℝ) (rA rB : ℝ) (dA dB : ℝ)
  (hA : dA = 2 * rA) (hB : dB = 2 * rB) (hd : h = 8) (hDA : dA = 10) (hDB : dB = 14) :
  π * (π * (rB ^ 2 * h) - π * (rA ^ 2 * h)) = 192 * π ^ 2 :=
by
  -- Provided conditions
  have hA : rA = 5 := by rw [←hDA, hA]; norm_num
  have hB : rB = 7 := by rw [←hDB, hB]; norm_num
  -- Given height is 8
  have h_eq : h = 8 := hd
  -- Compute the volumes
  let VA := volume rA h
  let VB := volume rB h
  -- Volumes as computed
  have VA_eq : VA = 200 * π := by { unfold volume, rw [h_eq, hA], norm_num, ring }
  have VB_eq : VB = 392 * π := by { unfold volume, rw [h_eq, hB], norm_num, ring }
  -- Compute the positive difference and multiply by π
  have diff : VB - VA = 192 * π := by rw [VB_eq, VA_eq]; norm_num
  -- Required result
  calc
    π * (VB - VA)
        = π * (192 * π)        : by rw diff
    ... = 192 * π ^ 2          : by ring

end volume_difference_l821_821319


namespace points_on_circle_l821_821824

theorem points_on_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1);
  let y := (2 * t^3) / (t^3 + 1);
  x^2 + y^2 = 1 :=
by
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2 * t^3) / (t^3 + 1)
  have h1 : x^2 + y^2 = ((t^3 - 1) / (t^3 + 1))^2 + ((2 * t^3) / (t^3 + 1))^2 := by rfl
  have h2 : (x^2 + y^2) = ( (t^3 - 1)^2 + (2 * t^3)^2 ) / (t^3 + 1)^2 := by sorry
  have h3 : (x^2 + y^2) = ( t^6 - 2 * t^3 + 1 + 4 * t^6 ) / (t^3 + 1)^2 := by sorry
  have h4 : (x^2 + y^2) = 1 := by sorry
  exact h4

end points_on_circle_l821_821824


namespace M_values_l821_821027

theorem M_values (m n p M : ℝ) (h1 : M = m / (n + p)) (h2 : M = n / (p + m)) (h3 : M = p / (m + n)) :
  M = 1 / 2 ∨ M = -1 :=
by
  sorry

end M_values_l821_821027


namespace volume_of_cuboid_l821_821632

def length : ℝ := 4
def width : ℝ := 4
def height : ℝ := 6

def volume (l w h : ℝ) : ℝ := l * w * h

theorem volume_of_cuboid : volume length width height = 96 := by
  -- The proof should show the volume calculation corresponds to 96
  sorry

end volume_of_cuboid_l821_821632


namespace find_general_formula_l821_821104

variable {ℕ : Type}

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_general_formula (a : ℕ → ℝ) (h1 : a 3 = 3) (h2 : a 2 + a 1 = 0) :
  a n = 2 * n - 3 :=
by
  have h_arith : arithmetic_sequence a := sorry
  sorry

end find_general_formula_l821_821104


namespace triangle_BF_eq_DC_l821_821113

variable (A B C D E F : Type) [Field A]
variables [Triangle ABC] [Incircle ABC D]
variables [IncircleDiameterOpposite D E] [Intersection AE BC F]

theorem triangle_BF_eq_DC :
  BF = DC := sorry

end triangle_BF_eq_DC_l821_821113


namespace mutually_exclusive_pairs_count_l821_821869

-- Definitions for exclusive events
def hits_7th_ring : Prop := sorry
def hits_8th_ring : Prop := sorry
def A_hits_7th_ring : Prop := sorry
def B_hits_8th_ring : Prop := sorry
def both_hit : Prop := sorry
def neither_hit : Prop := sorry
def at_least_one_hits : Prop := sorry
def A_hits_B_misses : Prop := sorry

def mutually_exclusive (E1 E2 : Prop) : Prop := (E1 ∧ E2) = false

-- Conditions
def pair1 := mutually_exclusive hits_7th_ring hits_8th_ring
def pair2 := ¬ mutually_exclusive A_hits_7th_ring B_hits_8th_ring
def pair3 := mutually_exclusive both_hit neither_hit
def pair4 := ¬ mutually_exclusive at_least_one_hits A_hits_B_misses

-- Statement to be proven
theorem mutually_exclusive_pairs_count : ((pair1 ∧ pair3) ∧ (¬ pair2) ∧ (¬ pair4)) ↔ 2 :=
by sorry

end mutually_exclusive_pairs_count_l821_821869


namespace sqrt_diff_equality_l821_821785

theorem sqrt_diff_equality :
  sqrt (5 + 4 * sqrt 3) - sqrt (5 - 4 * sqrt 3) = 2 * sqrt 2 :=
sorry

end sqrt_diff_equality_l821_821785


namespace gallons_to_travel_l821_821289

/- Define the conditions -/
def kilometers_per_gallon : ℝ := 40
def car_distance (gallons : ℝ) : ℝ := 160 / 4

/- Define the function to calculate gallons needed for any distance -/
def gallons_needed (d : ℝ) : ℝ := d / kilometers_per_gallon

/- State the theorem -/
theorem gallons_to_travel (d : ℝ) : gallons_needed(d) = d / kilometers_per_gallon := 
by sorry

end gallons_to_travel_l821_821289


namespace sum_of_perimeters_l821_821243

theorem sum_of_perimeters (x y z : ℝ) 
    (h_large_triangle_perimeter : 3 * 20 = 60)
    (h_hexagon_perimeter : 60 - (x + y + z) = 40) :
    3 * (x + y + z) = 60 := by
  sorry

end sum_of_perimeters_l821_821243


namespace log_difference_l821_821831

noncomputable def alpha : ℝ := sorry

theorem log_difference 
  (h1 : α ∈ Ioo (0 : ℝ) (real.pi / 2)) 
  (h2 : real.tan (α + real.pi / 4) = 3) : 
  real.log10 (8 * real.sin α + 6 * real.cos α) - real.log10 (4 * real.sin α - real.cos α) = 1 :=
sorry

end log_difference_l821_821831


namespace parallel_vectors_implies_value_of_λ_l821_821426

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821426


namespace max_largest_element_l821_821726

theorem max_largest_element {lst : List ℕ} (h_len : lst.length = 7)
  (h_pos : ∀ x ∈ lst, 0 < x)
  (h_median : lst.nth_le 3 (by linarith) = 4)
  (h_mean : (lst.sum : ℝ) / 7 = 10) :
  lst.maximum = 52 :=
sorry

end max_largest_element_l821_821726


namespace area_of_midpoints_l821_821891

noncomputable def set_of_segments (hexagon : list (ℝ × ℝ)) : set (ℝ × ℝ × ℝ × ℝ) :=
-- Define the set of all segments of length 3 with endpoints on adjacent sides
  sorry

noncomputable def midpoints_enclosing_region_area (hexagon : list (ℝ × ℝ)) (S : set (ℝ × ℝ × ℝ × ℝ)) : ℝ :=
-- Compute the area enclosed by the midpoints of segments in set S
  sorry

theorem area_of_midpoints (k : ℝ) (hexagon : list (ℝ × ℝ)) :
  (list.length hexagon = 6) ∧ (∀ (a b : ℝ × ℝ), a ∈ hexagon → b ∈ hexagon → (∃ s ∈ set_of_segments hexagon, true)) → 
  (100 * midpoints_enclosing_region_area hexagon (set_of_segments hexagon) = 185) :=
sorry

end area_of_midpoints_l821_821891


namespace monotonic_decreasing_interval_l821_821640

open Real

-- Definitions and conditions used to ensure the function is meaningful
def valid_domain (x : ℝ) : Prop := 4 + 3 * x - x^2 > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := ln (4 + 3 * x - x^2)

-- The goal is to find the monotonic decreasing interval of f(x)
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, valid_domain x → x ∈ set.Icc (3 / 2) 4 → 
  ∀ y : ℝ, valid_domain y → y ∈ set.Icc (3 / 2) 4 → x < y → f x ≥ f y :=
sorry

end monotonic_decreasing_interval_l821_821640


namespace lambda_parallel_l821_821453

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821453


namespace calculate_principal_amount_l821_821694

theorem calculate_principal_amount (P : ℝ) (h1 : P * 0.1025 - P * 0.1 = 25) : 
  P = 10000 :=
by
  sorry

end calculate_principal_amount_l821_821694


namespace parallel_vectors_lambda_value_l821_821462

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821462


namespace find_lambda_l821_821437

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821437


namespace finite_union_eq_univ_l821_821692

-- Definitions given in the problem
universe u
variable {V : Type u} [Fintype V]

variable (f g : V → V) (hf : Function.Bijective f) (hg : Function.Bijective g)

def S : Set V := { w : V | f (f w) = g (g w) }
def T : Set V := { w : V | f (g w) = g (f w) }

theorem finite_union_eq_univ (hST : S ∪ T = Set.univ) : ∀ w : V, (f w ∈ S ↔ g w ∈ S) :=
by
  intro w
  sorry

end finite_union_eq_univ_l821_821692


namespace arithmetic_sequence_general_formula_sum_of_first_n_terms_l821_821969

variable (a : ℕ → ℤ)

theorem arithmetic_sequence_general_formula (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 7 = -5) :
  ∃ d : ℤ, ∃ a_1 : ℤ, (∀ n : ℕ, a n = 13 - 2 * (n : ℤ)) :=
begin
  sorry
end

theorem sum_of_first_n_terms (h1 : a 2 + a 6 = 10) (h2 : a 4 * a 7 = -5) :
  (∃ S : ℕ → ℤ, ∀ n : ℕ, S n = 12 * (n : ℤ) - (n : ℤ)^2) :=
begin
  sorry
end

end arithmetic_sequence_general_formula_sum_of_first_n_terms_l821_821969


namespace correct_expression_l821_821261

theorem correct_expression (a b c : ℝ) : 3 * a - (2 * b - c) = 3 * a - 2 * b + c :=
sorry

end correct_expression_l821_821261


namespace number_of_n_with_zero_product_l821_821821

def omega (n k : ℕ) : ℂ := complex.exp (2 * real.pi * complex.I * k / n)

noncomputable def factor_is_zero (n : ℕ) : Prop :=
  ∃ k, (1 + omega n k)^(2 * n) + 1 = 0

theorem number_of_n_with_zero_product :
  { n : ℕ | 1 ≤ n ∧ n ≤ 3000 ∧ factor_is_zero n }.to_finset.card = 500 :=
sorry

end number_of_n_with_zero_product_l821_821821


namespace acceptable_arrangements_correct_l821_821942

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821942


namespace parallel_vectors_lambda_l821_821490

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821490


namespace two_fifths_in_fraction_l821_821068

theorem two_fifths_in_fraction : 
  (∃ (k : ℚ), k = (9/3) / (2/5) ∧ k = 15/2) :=
by 
  sorry

end two_fifths_in_fraction_l821_821068


namespace discount_per_card_is_correct_l821_821935

-- Definitions based on the conditions
variable (cost_per_card : ℝ) (quantity : ℕ) (total_paid : ℝ)

def total_cost_without_discount := cost_per_card * quantity
def total_discount := total_cost_without_discount cost_per_card quantity - total_paid
def discount_per_card := total_discount cost_per_card quantity total_paid / quantity

-- The statement to prove
theorem discount_per_card_is_correct (h1 : cost_per_card = 12) 
                                    (h2 : quantity = 10) 
                                    (h3 : total_paid = 100) : 
  discount_per_card cost_per_card quantity total_paid = 2 := 
by 
  sorry

end discount_per_card_is_correct_l821_821935


namespace no_general_solution_for_rational_expression_l821_821778

theorem no_general_solution_for_rational_expression:
  ∀ (x : ℝ) (r : ℚ), ∃ (k : ℚ), 
  ¬ (∃ (y : ℚ), 
    y = (x : ℝ) + (k : ℝ) * real.sqrt(x^2 + 1) 
        - 1 / ((x : ℝ) + (k : ℝ) * real.sqrt(x^2 + 1))) :=
by
  sorry

end no_general_solution_for_rational_expression_l821_821778


namespace number_of_arrangements_l821_821949

theorem number_of_arrangements (n : ℕ) (h1 : 8 = n) (h2 : ¬ ∃ i : ℕ, i ≤ 7 ∧ i > 0 ∧ Alice = (people.nth i) ∧ Bob = (people.nth (i+1))) : 
  (fact 8 - fact 7 * 2) = 30240 :=
by
  sorry

end number_of_arrangements_l821_821949


namespace design_black_percentage_correct_l821_821576

-- Definitions based on conditions
def starting_radius : ℝ := 3
def radii : List ℝ := List.range 5 |>.map (λ n => starting_radius * (n + 1))

def area (r : ℝ) : ℝ := real.pi * r^2

def black_regions (radii : List ℝ) : List ℝ := [area (radii[1]) - area (radii[0]), area (radii[3]) - area (radii[2])]

-- Total area calculation
def total_area (radii : List ℝ) : ℝ := area radii[4]

-- Black area calculation
def black_area (black_regions : List ℝ) : ℝ := black_regions.sum

-- Percentage calculation
def percentage_black (black_area total_area : ℝ) : ℝ := (black_area / total_area) * 100

theorem design_black_percentage_correct :
  percentage_black (black_area (black_regions radii)) (total_area radii) = 40 :=
by
  sorry

end design_black_percentage_correct_l821_821576


namespace parallel_vectors_lambda_l821_821494

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821494


namespace B_alone_work_days_l821_821708

theorem B_alone_work_days (B : ℕ) (A_work : ℝ) (C_work : ℝ) (total_payment : ℝ) :
  (A_work = 1 / 6) →
  (total_payment = 3200) →
  (C_work = (400 / total_payment) * (1 / 3)) →
  (A_work + 1 / B + C_work = 1 / 3) →
  B = 8 :=
begin
  intros hA_work htotal_payment hC_work hcombined_work,
  sorry,
end

end B_alone_work_days_l821_821708


namespace find_number_of_new_trailers_l821_821818

-- Given Conditions
variables {n : ℕ} -- number of new trailer homes

-- Initial condition: five years ago, 25 trailer homes with an average age of 12 years
def initial_number_of_trailers := 25
def initial_average_age := 12

-- Current condition: 5 old trailer homes removed, total average age is now 11 years
def removed_old_trailers := 5
def current_average_age := 11

noncomputable def solve_for_new_trailers (n : ℕ) : Prop :=
  let age_of_old_trailers := 20 * (initial_average_age + 5) in  -- old trailers have aged 5 years
  let age_of_new_trailers := n * 5 in  -- age of new trailers
  let total_trailers := 20 + n in
  let total_age := age_of_old_trailers + age_of_new_trailers in
  (total_age / total_trailers = current_average_age) → n = 20

theorem find_number_of_new_trailers : solve_for_new_trailers 20 := sorry

end find_number_of_new_trailers_l821_821818


namespace find_lambda_l821_821440

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821440


namespace parallel_vectors_implies_value_of_λ_l821_821431

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821431


namespace calc_other_diagonal_length_l821_821629

theorem calc_other_diagonal_length (d1 : ℝ) (Area : ℝ) (d2 : ℝ) : 
  d1 = 65 → Area = 1950 → (Area = (d1 * d2) / 2) → d2 = 60 :=
by
  intro h1 h2 h3
  have h : Area * 2 = d1 * d2, from sorry
  sorry

end calc_other_diagonal_length_l821_821629


namespace acute_triangle_inequality_l821_821103

theorem acute_triangle_inequality
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (h1 : 0 < A ∧ A < π/2)
  (h2 : 0 < B ∧ B < π/2)
  (h3 : 0 < C ∧ C < π/2)
  (h4 : A + B + C = π)
  (h5 : R = 1)
  (h6 : a = 2 * R * Real.sin A)
  (h7 : b = 2 * R * Real.sin B)
  (h8 : c = 2 * R * Real.sin C) :
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 :=
by
  sorry

end acute_triangle_inequality_l821_821103


namespace count_perfect_squares_diff_l821_821901

theorem count_perfect_squares_diff (a b : ℕ) : 
  ∃ (count : ℕ), 
  count = 25 ∧ 
  (∀ (a : ℕ), (∃ (b : ℕ), a^2 = 2 * b + 1 ∧ a^2 < 2500) ↔ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 25 ∧ 2 * k - 1 = a)) :=
by
  sorry

end count_perfect_squares_diff_l821_821901


namespace measure_of_C_calculate_area_l821_821937

namespace TriangleProof

-- Definitions of the parameters as per the conditions
variables (a b c A B C : ℝ)
variable (triangleABC : a > 0 ∧ b > 0 ∧ c > 0)
variable (acute : A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2)
variable (sine_relation : sqrt 3 * a = 2 * c * sin A)

-- Definition of the first proof problem
-- Prove that C = π / 3 given the condition sqrt(3) * a = 2 * c * sin A
theorem measure_of_C (h_triangle: a ^ 2 + b ^ 2 > c ^ 2): C = π / 3 := by
  sorry

-- Additional conditions for the second problem
variable (c_val : c = sqrt 7)
variable (perimeter_condition : a + b + c = 5 + sqrt 7)

-- Definition of the second proof problem
-- Prove that the area of ΔABC is 3√3/2 given additional conditions
noncomputable def area_of_triangle := (1/2) * a * b * sin C

theorem calculate_area [fact (c = sqrt 7)] [fact (a + b + c = 5 + sqrt 7)]
: area_of_triangle a b C = 3 * sqrt 3 / 2 := by
  sorry

end TriangleProof

end measure_of_C_calculate_area_l821_821937


namespace cone_volume_correct_l821_821718

-- Define the parameters for the cone
def height : ℝ := 2
def radius : ℝ := 1

-- Define the formula for the volume of a cone
def cone_volume (h r : ℝ) : ℝ :=
  (1 / 3) * Math.pi * r^2 * h

-- State the theorem we want to prove
theorem cone_volume_correct : cone_volume height radius = (2 / 3) * Math.pi :=
by
  -- Proof is not required, thus omitted with sorry
  sorry

end cone_volume_correct_l821_821718


namespace similar_triangles_area_ratio_l821_821856

theorem similar_triangles_area_ratio (ABC DEF : Type) 
  [triangle ABC] [triangle DEF]
  (similar : similar_triangles ABC DEF) 
  (ratio : ∀ a b, side_ratio a b ABC DEF = 1 : 3) 
  (area_ABC : area ABC = 1) : 
  area DEF = 9 :=
begin
  sorry
end

end similar_triangles_area_ratio_l821_821856


namespace ellipse_focus_eccentricity_l821_821039

theorem ellipse_focus_eccentricity (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 2) + (y^2 / m) = 1 → y = 0 ∨ x = 0) ∧
  (∀ e : ℝ, e = 1 / 2) →
  m = 3 / 2 :=
sorry

end ellipse_focus_eccentricity_l821_821039


namespace louies_brother_goal_ratio_l821_821525

-- Definitions based on given conditions.
def louie_last_match_goals : ℕ := 4
def louie_previous_matches_goals : ℕ := 40
def seasons : ℕ := 3
def games_per_season : ℕ := 50
def total_goals : ℕ := 1244

-- Calculate total games played by Louie's brother and the ratio
theorem louies_brother_goal_ratio :
  ∃ (x : ℕ), 
    let brother_total_goals := (seasons * games_per_season) * x,
        louie_total_goals := louie_last_match_goals + louie_previous_matches_goals in
    brother_total_goals + louie_total_goals = total_goals ∧
    (x / louie_last_match_goals) = 2 := 
by {
  sorry
}

end louies_brother_goal_ratio_l821_821525


namespace statement_A_statement_D_l821_821411

-- Defining the conditions for the curve
variable (m n : ℝ)
def C (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Statement A: Prove that C is an ellipse with its foci on the y-axis if m > n > 0
theorem statement_A (h1 : m > n) (h2 : n > 0) : 
  (∃ foci_y : ℝ, C = ellipse_with_foci_y foci_y) :=
sorry

-- Statement D: Prove that C consists of two straight lines if m = 0 and n > 0
theorem statement_D (h1 : m = 0) (h2 : n > 0) : 
  (∃ y₁ y₂ : ℝ, (C = (λ x y, y = y₁ ∨ y = y₂))) :=
sorry

end statement_A_statement_D_l821_821411


namespace solve_pairs_l821_821806

theorem solve_pairs (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (m, n) = (6, 3) ∨ (m, n) = (9, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end solve_pairs_l821_821806


namespace key_placement_l821_821659

theorem key_placement (n : ℕ) (h : n = 6) :
  (∃ f : Fin n → Fin n, (∀ i, f (f i) ≠ i) ∧ 
                        (∀ i, ∃ j, ∀ k, k ≠ j → k ≠ i → f j = k)) → 
  (Finset.univ.card.factorial = 120) :=
by {
  intros,
  sorry
}

end key_placement_l821_821659


namespace evaluate_expression_at_3_l821_821255

theorem evaluate_expression_at_3 :
  (∀ x ≠ 2, (x = 3) → (x^2 - 5 * x + 6) / (x - 2) = 0) :=
by
  sorry

end evaluate_expression_at_3_l821_821255


namespace find_valid_angle_complement_l821_821215

-- Specify the conditions of the problem
def is_digit (n : ℕ) : Prop := n < 10
def two_digit_less_than_60 (n m : ℕ) : Prop := n < 10 ∧ m < 10 ∧ (10 * n + m) < 60

def valid_angle (A B C D E F : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ is_digit E ∧ is_digit F ∧
  two_digit_less_than_60 A B ∧ 
  two_digit_less_than_60 C D ∧
  two_digit_less_than_60 E F

def angle_representation (A B C D E F : ℕ) : ℕ :=
  3600 * (10 * A + B) + 60 * (10 * C + D) + (10 * E + F)

-- Define the complementary condition
def complementary_angles (A1 B1 C1 D1 E1 F1 : ℕ) (A2 B2 C2 D2 E2 F2 : ℕ) : Prop :=
  angle_representation A1 B1 C1 D1 E1 F1 + angle_representation A2 B2 C2 D2 E2 F2 = 324000 -- 89*3600 + 59*60 + 60

-- Formalize the proof problem
theorem find_valid_angle_complement :
  ∃ (A B C D E F : ℕ),
    valid_angle A B C D E F ∧ complementary_angles A B C D E F (4 4 1 5 4 5) :=
sorry

end find_valid_angle_complement_l821_821215


namespace rectangle_perimeter_l821_821354

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 4 * (a + b)) : 2 * (a + b) = 36 := by
  sorry

end rectangle_perimeter_l821_821354


namespace sufficient_not_necessary_condition_l821_821402

theorem sufficient_not_necessary_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : (0 < x ∧ x < 2) → (x^2 - x - 2 < 0) :=
by
  intros h
  sorry

end sufficient_not_necessary_condition_l821_821402


namespace count_perfect_square_factors_of_1680_l821_821220

-- Defining the problem conditions
def is_factor (n m : ℕ) : Prop := ∃ k : ℕ, n = k * m
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem count_perfect_square_factors_of_1680 :
  ∃ (N : ℕ), N = 1680 ∧ 
  ∃ (cnt : ℕ), cnt = 
    finset.card (finset.filter 
      (λ x, is_perfect_square x) 
      (finset.filter 
        (λ x, is_factor 1680 x) 
        (finset.range (1680 + 1)))) ∧
  cnt = 3 := 
by sorry

end count_perfect_square_factors_of_1680_l821_821220


namespace slowest_initial_bailing_rate_l821_821263

theorem slowest_initial_bailing_rate (distance_shore : ℝ) 
  (row_speed : ℝ) 
  (initial_water_intake_rate : ℝ) 
  (max_water_before_sinking : ℝ) 
  (time_before_increase : ℝ) 
  (bailing_increase_rate : ℝ) :
  2 = distance_shore →
  3 = row_speed →
  8 = initial_water_intake_rate →
  50 = max_water_before_sinking →
  20 = time_before_increase →
  2 = bailing_increase_rate →
  (∃ r, r = 5.75) :=
by
  intros h1 h2 h3 h4 h5 h6
  use 5.75
  sorry

end slowest_initial_bailing_rate_l821_821263


namespace num_valid_mappings_l821_821057

open Finset

-- Define the sets M and N
def M : Finset ℕ := {0, 1}  -- Using indices 0 and 1 to represent a and b respectively
def N : Finset ℤ := {-1, 0, 1}

-- Predicate to check if a mapping satisfies f(a) <= f(b)
def valid_mapping (f : ℕ → ℤ) : Prop :=
  f 0 ≤ f 1

-- Define the collection of all possible mappings from M to N
def all_mappings : Finset (ℕ → ℤ) :=
  univ.image (λ (p : ℤ × ℤ), fun i => if i = 0 then p.fst else p.snd)

-- The main theorem statement
theorem num_valid_mappings :
  (all_mappings.filter valid_mapping).card = 6 :=
by
  sorry

end num_valid_mappings_l821_821057


namespace expectation_of_X_median_of_weights_contingency_table_correct_confidence_95_percent_l821_821321

-- Part 1
def miceDistribution (X: ℕ) : ℚ :=
  if X = 0 then 19/78 
  else if X = 1 then 20/39 
  else if X = 2 then 19/78 
  else 0

theorem expectation_of_X : ∀ X, (X = 0 ∨ X = 1 ∨ X = 2) → 
  E(X) = 0 * (19/78) + 1 * (20/39) + 2 * (19/78) := 
begin
  sorry
end

-- Part 2 
def control_group_weights := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1, 32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]
def experimental_group_weights := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2, 19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

def median (w: list ℚ) : ℚ := (w[19] + w[20]) / 2

theorem median_of_weights : median (control_group_weights ++ experimental_group_weights) = 23.4 :=
begin
  sorry
end

def contingency_table : list (list ℕ) :=
[[6, 14], [14, 6]]

theorem contingency_table_correct : contingency_table = [[6, 14], [14, 6]] :=
begin
  sorry
end

noncomputable def K_squared : ℚ :=
40 * (6 * 6 - 14 * 14)^2 / (20 * 20 * 20 * 20)

theorem confidence_95_percent : K_squared ≥ 3.841 :=
begin
  sorry
end

end expectation_of_X_median_of_weights_contingency_table_correct_confidence_95_percent_l821_821321


namespace concyclic_CQRS_l821_821018

-- Given complex representations of points and additional conditions
variables (a b c d : ℝ) (α β : ℝ)
variables (e_alpha : ℂ) (e_beta : ℂ)
variables (h_alpha : e_alpha = complex.exp (complex.I * α))
variables (h_beta : e_beta = complex.exp (complex.I * β))
variables (ha : a * c = b * d)

def A := a * e_alpha
def B := b * e_beta
def C := -c * e_alpha
def D := -d * e_beta

-- Points Q, R, and S based on the conditions
def Q := -B
def R := C + Q - A
def S := C + D - B

-- Main statement asserting the concyclicity of points C, Q, R, S
theorem concyclic_CQRS : 
  let cross_ratio := (λ z1 z2 z3 z4 : ℂ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3))) in
  (cross_ratio Q S C R).im = 0 :=
sorry

end concyclic_CQRS_l821_821018


namespace sample_size_calculation_l821_821297

/--
A factory produces three different models of products: A, B, and C. The ratio of their quantities is 2:3:5.
Using stratified sampling, a sample of size n is drawn, and it contains 16 units of model A.
We need to prove that the sample size n is 80.
-/
theorem sample_size_calculation
  (k : ℕ)
  (hk : 2 * k = 16)
  (n : ℕ)
  (hn : n = (2 + 3 + 5) * k) :
  n = 80 :=
by
  sorry

end sample_size_calculation_l821_821297


namespace maximize_profit_l821_821720

noncomputable def daily_sales (k : ℝ) (x : ℝ) : ℝ :=
  k / exp x

theorem maximize_profit (t : ℝ) (h_t : 2 ≤ t ∧ t ≤ 5) (k : ℝ) (h_k : k = 100 * exp 30) :
  ∃ x : ℝ, 25 ≤ x ∧ x ≤ 40 ∧ 
  let q := daily_sales k x in
  let y := 100 * exp 30 * (x - 20 - t) / exp x in
  t = 5 → x = 26 ∧ y = 100 * exp 4 := 
by
  sorry

end maximize_profit_l821_821720


namespace nominal_rate_of_interest_l821_821206

noncomputable def nominal_rate (EAR : ℝ) (n : ℕ) : ℝ :=
  2 * (Real.sqrt (1 + EAR) - 1)

theorem nominal_rate_of_interest :
  nominal_rate 0.1025 2 = 0.100476 :=
by sorry

end nominal_rate_of_interest_l821_821206


namespace sum_c_n_l821_821120

noncomputable def a_n : ℕ → ℕ
| 1 => 1
| 3 => 3
| n => sorry  -- Arithmetic sequence property can be inferred

noncomputable def b_n (n : ℕ) : ℕ := 2^(a_n n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

noncomputable def S_n (n : ℕ) : ℕ := (finset.range n).sum (λ i, c_n (i + 1))

theorem sum_c_n (n : ℕ) : S_n n = n * 2^n :=
by
  sorry

end sum_c_n_l821_821120


namespace seating_arrangements_l821_821958

open Nat

theorem seating_arrangements (total_people : ℕ) (alice : ℕ) (bob : ℕ) (h_total : total_people = 8) (h_alice_bob : alice ≠ bob) :
  let total_arrangements := factorial total_people,
      alice_bob_together_arrangements := factorial 7 * factorial 2,
      arrangements_with_condition := total_arrangements - alice_bob_together_arrangements
  in arrangements_with_condition = 30240 :=
by 
  rw [h_total]
  sorry

end seating_arrangements_l821_821958


namespace find_general_term_l821_821419

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧
  a 2 = 1 ∧
  a 3 = 2 ∧
  ∀ n, 3 * a (n + 3) = 4 * a (n + 2) + a (n + 1) - 2 * a n

def closed_form (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = (1/25 : ℚ) + (3/5 : ℚ) * n - (27/50 : ℚ) * ((-2/3 : ℚ)^n)

theorem find_general_term (a : ℕ → ℚ) (h : sequence a) : closed_form a :=
sorry

end find_general_term_l821_821419


namespace separation_of_triangles_l821_821422

theorem separation_of_triangles 
  {A B C D E F : Point}
  (hABC_not_common : ¬(Triangle_overlapping_or_boundary A B C D E F)) :
  ∃ l : Line, (l ⊆ Line_through A B ∨ l ⊆ Line_through B C ∨ l ⊆ Line_through C A)
  ∧ (interior A B C) ∩ (interior D E F) = ∅ :=
sorry

end separation_of_triangles_l821_821422


namespace abs_half_sufficient_not_necessary_l821_821077

theorem abs_half_sufficient_not_necessary (x : ℝ) :
  (| x - 1 / 2 | < 1 / 2 ↔ 0 < x ∧ x < 1) → (| x - 1 / 2 | < 1 / 2 → x < 1 ∧ ¬ (x < 1 → 0 < x ∧ x < 1) ) := 
by
  sorry

end abs_half_sufficient_not_necessary_l821_821077


namespace vacation_savings_l821_821621

-- Definitions
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500
def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Prove the amount set aside for vacation
theorem vacation_savings :
  let 
    total_income := parents_salary + grandmothers_pension + sons_scholarship,
    total_expenses := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses,
    surplus := total_income - total_expenses,
    deposit := (10 * surplus) / 100, 
    vacation_money := surplus - deposit
  in
    vacation_money = 16740 := by
      -- Calculation steps skipped; proof not required
      sorry

end vacation_savings_l821_821621


namespace find_lambda_l821_821435

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821435


namespace line_lengths_ratios_l821_821174

theorem line_lengths_ratios (XY YZ XW : ℝ) (hXY : XY = 3) (hYZ : YZ = 4) (hXW : XW = 20) :
  let XZ := XY + YZ
      YW := XW - YZ
  in (XZ / YW = 7 / 16) ∧ (YZ / XW = 1 / 5) :=
by
  let XZ := XY + YZ
  let YW := XW - YZ
  have hXZ : XZ = 7 := by
    simp [XZ, hXY, hYZ]
  have hYW : YW = 16 := by
    simp [YW, hYZ, hXW]
  have ratio_XZ_YW : XZ / YW = 7 / 16 := by
    rw [hXZ, hYW]
  have ratio_YZ_XW : YZ / XW = 1 / 5 := by
    simp [hYZ, hXW]
  exact ⟨ratio_XZ_YW, ratio_YZ_XW⟩

end line_lengths_ratios_l821_821174


namespace form_three_triangles_with_equal_perimeter_l821_821550

noncomputable def sticks : List ℕ := [2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 9]

theorem form_three_triangles_with_equal_perimeter (sticks : List ℕ) (h_sticks_perm : sticks.perm sticks) :
  ∃ (triangles : List (List ℕ)), (length triangles = 3) ∧ (∀ t ∈ triangles, t.sum = 14) ∧
  (sticks.perm (triangles.concat_list)) ∧
  (∀ t ∈ triangles, ∀ a b c ∈ t, (a + b > c) ∧ (b + c > a) ∧ (c + a > b)) :=
sorry

end form_three_triangles_with_equal_perimeter_l821_821550


namespace find_value_of_expression_l821_821855

theorem find_value_of_expression (a b : ℝ) :
  (∀ (x y : ℝ), x = 2 → y = -1 → a * x + b * y = -1) →
  2 * a - b + 2017 = 2016 :=
by
  intros h,
  specialize h 2 (-1) rfl rfl,
  have eq1 : 2 * a - b = -1 := by
    exact h,
  linarith

end find_value_of_expression_l821_821855


namespace count_valid_solutions_l821_821280

-- Define the variables
variables (x y z : ℕ)

-- Define the conditions as hypotheses
hypothesis h1 : x + y + z = 26
hypothesis h2 : 6 * x + 4 * y + 2 * z = 88

-- Define the problem: Prove there are exactly 8 valid solutions (x, y, z)
theorem count_valid_solutions : 
  (finset.univ.filter (λ (t : ℕ × ℕ × ℕ), let (x, y, z) := t in (x + y + z = 26 ∧ 6 * x + 4 * y + 2 * z = 88))).card = 8 :=
by {
  -- Solution omitted
  sorry,
}

end count_valid_solutions_l821_821280


namespace simplify_expression_l821_821346

theorem simplify_expression (x y z : ℝ) : (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := 
by 
sorry

end simplify_expression_l821_821346


namespace reconstruct_diagonals_possible_l821_821719

-- Define structures and conditions of the problem
structure ConvexPolygon (V : Type) :=
(vertices : Finset V)
(diagonals : Finset (V × V))
(trianglesAdjacent : V → ℕ)
(is_convex : ∀ (x y z : V), triangle_contains_vertex x y z)
(non_intersecting_diagonals : ∀ (d1 d2 : V × V), d1 ≠ d2 → ¬ segments_intersect d1 d2)
(all_diagonals_removed : diagonals = ∅)

-- Define what it means to reconstruct diagonals
def can_reconstruct_diagonals (P : ConvexPolygon V) : Prop :=
∀ (triangles_adjacent : V → ℕ), 
  (∀ (v : V), triangles_adjacent v = P.trianglesAdjacent v) →
   ∃ (new_diagonals : Finset (V × V)), 
  P = {P with diagonals := new_diagonals}

-- Final theorem statement
theorem reconstruct_diagonals_possible 
  {V : Type} (P : ConvexPolygon V)
  (triangles_adjacent : V → ℕ) 
  (h_triangles_adjacent : ∀ (v : V), triangles_adjacent v = P.trianglesAdjacent v) :
  can_reconstruct_diagonals P :=
sorry

end reconstruct_diagonals_possible_l821_821719


namespace find_ellipse_equation_find_slope_l821_821839

noncomputable def eccentricity (a b : ℝ) : ℝ := 
  (real.sqrt (a ^ 2 - b ^ 2)) / a

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

def isosceles_triangle (A B M : ℝ × ℝ) : Prop :=
  let AB := real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) in
  let AM := real.sqrt ((A.1 - M.1) ^ 2 + (A.2 - M.2) ^ 2) in
  let MB := real.sqrt ((M.1 - B.1) ^ 2 + (M.2 - B.2) ^ 2) in
  AM = MB

theorem find_ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = real.sqrt 5 / 3) (x y : ℝ) 
  (h4 : ellipse a b (3 * real.sqrt 3 / 2) 1) : 
  ellipse 3 2 x y := sorry

theorem find_slope (a b : ℝ) (A B M : ℝ × ℝ) (h1 : A ≠ B) (h2 : M = (5/12, 0))
  (h3 : isosceles_triangle A B M) : 
  ((A.2 - B.2) / (A.1 - B.1)) = -2 / 3 := sorry

end find_ellipse_equation_find_slope_l821_821839


namespace least_value_z_minus_x_l821_821925

noncomputable def consecutive_primes : Prop :=
  ∃ x y z : ℕ, prime x ∧ prime y ∧ prime z ∧ x < y ∧ y < z ∧
              ∃ p : ℕ, prime p ∧ (y - x > 5) ∧ x % 2 = 0 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧ (y^2 + x^2) % p = 0

theorem least_value_z_minus_x (x y z : ℕ) (p : ℕ) :
  consecutive_primes →
  (∃ x y z : ℕ, x = 2 ∧ y = 11 ∧ z = 13) →
  z - x = 11 :=
by
  intros h1 h2
  cases h2 with x_val h2
  cases h2 with y_val h2
  cases h2 with z_val h2
  simp [x_val, y_val, z_val]
  exact 13 - 2 = 11

end least_value_z_minus_x_l821_821925


namespace line_inclination_angle_l821_821226

theorem line_inclination_angle (x y : ℝ) (θ : ℝ) :
  (x - sqrt 3 * y + 1 = 0) →
  θ = 30 :=
by
  intro h
  sorry

end line_inclination_angle_l821_821226


namespace altitude_bisects_angle_l821_821555

open EuclideanGeometry

variables {A B C H H_A H_B H_C : Point}

-- Conditions
axiom triangle_ABC : Triangle A B C
axiom H_A_foot : foot_of_altitude H_A A (line B C)
axiom H_B_foot : foot_of_altitude H_B B (line A C)
axiom H_C_foot : foot_of_altitude H_C C (line A B)
axiom H_orthocenter : orthocenter H A B C

-- Theorem statement
theorem altitude_bisects_angle : is_angle_bisector A H_A H_C H_B :=
sorry

end altitude_bisects_angle_l821_821555


namespace non_adjacent_arrangements_l821_821965

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821965


namespace total_number_of_slices_l821_821253

def number_of_pizzas : ℕ := 7
def slices_per_pizza : ℕ := 2

theorem total_number_of_slices :
  number_of_pizzas * slices_per_pizza = 14 :=
by
  sorry

end total_number_of_slices_l821_821253


namespace binomial_sum_mod_500_l821_821334

theorem binomial_sum_mod_500 : 
  let S := ∑ k in Finset.range 2011, if (k % 5 = 0) then Nat.choose 2011 k else 0
  in S % 500 = 19 := 
by
  let ω1 ω2 ω3 ω4 : ℂ := sorry -- Placeholder for the fifth roots of unity
  have h_roots_of_unity : ∀ i ∈ {ω1, ω2, ω3, ω4}, i ^ 5 = 1 := sorry -- properties of ωi's
  have h_sum : S = (1 + ω1) ^ 2011 + (1 + ω2) ^ 2011 + (1 + ω3) ^ 2011 + (1 + ω4) ^ 2011 + (1 + 1) ^ 2011 :=
    sorry -- rephrasing S using the roots of unity
  have h_mod_exp : 2 ^ 2011 % 500 = 48 := sorry -- intermediate calculation
  have h_div : 5 * 19 ≡ 48 [MOD 500] := sorry -- simplifying to get the final answer
  exact h_sum.mp

end binomial_sum_mod_500_l821_821334


namespace maximum_value_on_interval_l821_821045

def f (x : ℝ) : ℝ := -x^2 + 4*x - 2

theorem maximum_value_on_interval : ∃ y ∈ set.Icc (0:ℝ) 1, 
  ∀ x ∈ set.Icc (0:ℝ) 1, f y ≥ f x ∧ f y = 1 :=
sorry

end maximum_value_on_interval_l821_821045


namespace trailing_zeroes_in_base12_l821_821906

theorem trailing_zeroes_in_base12 (n : ℕ) (h : n = 15)
  (f12 : ∀ n, 12 = 2^2 * 3)
  (h2 : (∑ k in finset.range 4, n / 2^k) = 11)
  (h3 : (∑ k in finset.range 2, n / 3^k) = 6) :
  ∃ k, trailing_zeroes_in_base n 12 = k ∧ k = 5 :=
by sorry

end trailing_zeroes_in_base12_l821_821906


namespace find_z_l821_821645

theorem find_z
  (z : ℚ)
  (proj_eq : (6 * z / 56) * (4 : ℚ, -2, 6) = (1 / 2) * (4, -2, 6)) :
  z = 14 / 3 := by
  sorry

end find_z_l821_821645


namespace triangle_shape_l821_821926

theorem triangle_shape
  {A B C : ℝ}
  (h : ∀ {α β γ : ℝ}, α + β + γ = π → 
        (cos A + 2 * cos C) / (cos A + 2 * cos B) = sin B / sin C) :
  A = π / 2 ∨ B = C :=
by
  sorry  -- Proof goes here

end triangle_shape_l821_821926


namespace probability_at_least_one_boy_and_one_girl_l821_821751

theorem probability_at_least_one_boy_and_one_girl 
  (h_equal_likelihood : ∀ (child : ℕ), (child ≤ 4) → (Pr(boy) = 1/2 ∧ Pr(girl) = 1/2)) :
  Pr(at_least_one_boy_and_one_girl) = 7/8 :=
sorry

end probability_at_least_one_boy_and_one_girl_l821_821751


namespace find_lambda_l821_821436

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821436


namespace courses_choice_diff_l821_821278

open Finset

theorem courses_choice_diff (A B : Finset ℕ) (hA : A.card = 2) (hB : B.card = 2) (hU : (univ : Finset ℕ).card = 4) :
  ∃ n : ℕ, n = 30 ∧ ∃ (diff : A ≠ B), (univ.choose 2).card * (univ.choose 2).card - (univ.choose 2).card = n :=
by sorry

end courses_choice_diff_l821_821278


namespace subtraction_correct_l821_821331

def x : ℝ := 5.75
def y : ℝ := 1.46
def result : ℝ := 4.29

theorem subtraction_correct : x - y = result := 
by
  sorry

end subtraction_correct_l821_821331


namespace find_f_equiv_l821_821908

variable {x : ℝ}

noncomputable def f (x : ℝ) := log ((1 + x) / (1 - x))

theorem find_f_equiv (h : -1 < x ∧ x < 1) :
  f ( (2 * x - x^3) / (1 + 2 * x^2) ) = - f x := by
  sorry

end find_f_equiv_l821_821908


namespace tax_authority_correct_l821_821586

/-- 
Proof problem: Given the conditions, prove that the tax authority's claim (that 
the agreement between Mikhail and Valentin violated legislation) is correct.
-/
theorem tax_authority_correct
  (Mikhail_Vasilievich : Prop) 
  (Valentin_Pavlovich : Prop) 
  (purchase_in_euros : Mikhail_Vasilievich → Valentin_Pavlovich → ℝ → Prop)
  (amount_euros : ℝ)
  (resident_Mikhail : Prop)
  (resident_Valentin : Prop)
  (Federal_Law_173_FZ : Prop)
  (prohibited_currency_transactions : Federal_Law_173_FZ → Prop)
  (exceptions_not_apply : Prohibited_currency_transactions → Federal_Law_173_FZ → Prop) :
  Mikhail_Vasilievich ∧ Valentin_Pavlovich ∧ resident_Mikhail ∧ resident_Valentin ∧ 
  purchase_in_euros Mikhail_Vasilievich Valentin_Pavlovich amount_euros →
  prohibited_currency_transactions Federal_Law_173_FZ →
  exceptions_not_apply prohibited_currency_transactions Federal_Law_173_FZ →
  (∃ notification_from_tax_authority : Prop, notification_from_tax_authority) :=
by
  sorry

end tax_authority_correct_l821_821586


namespace general_pattern_l821_821269

theorem general_pattern (n : ℕ) : (10 * n + 5)^2 = 100 * n * (n + 1) + 25 := by
  sorry

example : (10 * 199 + 5)^2 = 3980025 := by
  calc (10 * 199 + 5)^2 = 100 * 199 * (199 + 1) + 25 := general_pattern 199
                 ...    = 3980025               := by norm_num

end general_pattern_l821_821269


namespace perimeter_of_quadrilateral_l821_821767

noncomputable def perimeter_of_quadrilateral_proof : Prop :=
  let PA : ℝ := 30
  let PB : ℝ := 40
  let PC : ℝ := 25
  let PD : ℝ := 60
  let area_ABCD : ℝ := 2000
  ∃ (AB BC CD DA : ℝ), (AB + BC + CD + DA = 229 ∧ 
                          {ABCDA : convex_hull R2})

theorem perimeter_of_quadrilateral : perimeter_of_quadrilateral_proof := sorry


end perimeter_of_quadrilateral_l821_821767


namespace number_of_friends_l821_821195

def total_gold := 100
def lost_gold := 20
def gold_per_friend := 20

theorem number_of_friends :
  (total_gold - lost_gold) / gold_per_friend = 4 := by
  sorry

end number_of_friends_l821_821195


namespace order_of_magnitude_l821_821055

noncomputable def a : ℝ := 1.7 ^ 0.3
noncomputable def b : ℝ := 0.9 ^ 0.1
noncomputable def c : ℝ := Real.log 5 / Real.log 2
noncomputable def d : ℝ := Real.log 1.8 / Real.log 0.3

theorem order_of_magnitude :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end order_of_magnitude_l821_821055


namespace abs_inequality_range_l821_821078

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 6| > a) ↔ a < 5 :=
by
  sorry

end abs_inequality_range_l821_821078


namespace minimum_perimeter_l821_821851

def point (ℝ : Type) := ℝ × ℝ

noncomputable def A : point ℝ := (0, 3 * real.sqrt 7)
noncomputable def F : point ℝ := (-3, 0)
noncomputable def hyperbola (x y : ℝ) := (x^2 / 2) - (y^2 / 7) = 1

theorem minimum_perimeter
  (P : point ℝ)
  (H : hyperbola P.1 P.2)
  (on_right_branch : P.1 > 0) :
  let AP := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2),
      PF := real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)
  in AP + PF = 8 * real.sqrt 2 :=
sorry

end minimum_perimeter_l821_821851


namespace parallel_vectors_lambda_value_l821_821465

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821465


namespace perimeter_triangle_XPQ_l821_821248

-- Definitions for the given conditions
variables (X Y Z P Q I : Type)
variables (dXY : ℝ) (dYZ : ℝ) (dXZ : ℝ)
variables (hXY : dXY = 15)
variables (hYZ : dYZ = 25)
variables (hXZ : dXZ = 20)
variables (lineThroughIncenterParallelXZ : Prop)

-- We need to state the main theorem and goal
theorem perimeter_triangle_XPQ : 
  hXY → hYZ → hXZ → lineThroughIncenterParallelXZ → 
  ∃ dXP dPQ dQX : ℝ, (dXP + dPQ + dQX = 35) :=
by
  intros hXY hYZ hXZ lineThroughIncenterParallelXZ
  -- Proof goes here
  sorry

end perimeter_triangle_XPQ_l821_821248


namespace sum_x_y_z_eq_3_or_7_l821_821611

theorem sum_x_y_z_eq_3_or_7 (x y z : ℝ) (h1 : x + y / z = 2) (h2 : y + z / x = 2) (h3 : z + x / y = 2) : x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_x_y_z_eq_3_or_7_l821_821611


namespace volunteer_hours_per_year_l821_821548

def volunteers_per_month : ℕ := 2
def hours_per_session : ℕ := 3
def months_per_year : ℕ := 12

theorem volunteer_hours_per_year :
  volunteers_per_month * months_per_year * hours_per_session = 72 :=
by
  -- Proof is omitted
  sorry

end volunteer_hours_per_year_l821_821548


namespace find_a_range_l821_821413

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 2 then (3 - a) ^ x else log a (x - 1) + 3

theorem find_a_range (a : ℝ) :
  (∀ x, x ≤ 2 → (3 - a) ^ x ≤ (3 - a) ^ (x + 1)) ∧
  (∀ x, x > 2 → log a (x - 1) + 3 ≤ log a x + 3) →
  (3 - sqrt 3 < a ∧ a < 2) :=
sorry

end find_a_range_l821_821413


namespace limit_sin_sqrt_x_l821_821761

noncomputable theory
open_locale classical

theorem limit_sin_sqrt_x:
  tendsto (λ x : ℝ, (sin (sqrt x) * x) / (x ^ (-2:ℤ) + π * x ^ 2)) (𝓝 0) (𝓝 0) :=
by
  sorry

end limit_sin_sqrt_x_l821_821761


namespace domain_of_f_l821_821205

def f (x : ℝ) := real.sqrt (3 - x) + real.logb (1/3) (x + 1)

theorem domain_of_f :
  {x : ℝ | 3 - x ≥ 0 ∧ x + 1 > 0} = set.Ioc (-1 : ℝ) 3 :=
by
  sorry

end domain_of_f_l821_821205


namespace count_valid_N_l821_821066

theorem count_valid_N : 
  (finset.card {N : ℕ | ∃ x : ℝ, 0 < N ∧ N < 500 ∧ N = x^(⌊x⌋ + 1)}) = 197 :=
by
  sorry

end count_valid_N_l821_821066


namespace math_proof_problem_l821_821871

-- Define the function f(x) = sqrt(3) * sin^2(ω * x) + sin(ω * x) * cos(ω * x) - sqrt(3) / 2
def f (ω x : ℝ) : ℝ := 
  real.sqrt 3 * (real.sin (ω * x))^2 + real.sin (ω * x) * real.cos (ω * x) - real.sqrt 3 / 2

-- Define a condition that ω > 0
def ω_positive (ω : ℝ) : Prop := ω > 0

-- Define the assertion for ω
def ω_eq_1 (ω : ℝ) (hxω : ω_positive ω) : Prop := ∃ T : ℝ, T = π ∧ (π / 4 = T / 4 ∧ f ω (T / 2) = f ω 0)

-- Define the assertion for x
def min_f_x (ω x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ π / 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ π / 2) → f ω x ≤ f ω y

-- Now, wrap these definitions into the statement
theorem math_proof_problem (ω x : ℝ) (hxω : ω_positive ω) :
  ω_eq_1 ω hxω ∧ min_f_x 1 x := sorry

end math_proof_problem_l821_821871


namespace missing_fraction_of_coins_l821_821154

-- Defining the initial conditions
def total_coins (x : ℕ) := x
def lost_coins (x : ℕ) := (1 / 2) * x
def found_coins (x : ℕ) := (3 / 8) * x

-- Theorem statement
theorem missing_fraction_of_coins (x : ℕ) : 
  (total_coins x - lost_coins x + found_coins x) = (7 / 8) * x :=
by
  sorry  -- proof is omitted as per the instructions

end missing_fraction_of_coins_l821_821154


namespace john_average_speed_is_100_over_3_l821_821210

theorem john_average_speed_is_100_over_3 (total_distance : ℕ) 
(total_time : ℕ) (h1 : total_distance = 200) (h2 : total_time = 6) : 
(total_distance / total_time) = 100 / 3 := by
  rw [h1, h2]
  norm_num
  rw [←nat.div_eq_of_eq_mul_left (by norm_num : 3 > 0) (by norm_num : 200 = 100 * 2)]
  norm_num
  sorry

end john_average_speed_is_100_over_3_l821_821210


namespace calc_quotient_l821_821680

theorem calc_quotient (a b : ℕ) (h1 : a - b = 177) (h2 : 14^2 = 196) : (a - b)^2 / 196 = 144 := 
by sorry

end calc_quotient_l821_821680


namespace set_intersection_eq_l821_821571

def setA : Set ℝ := { x | x^2 - 3 * x - 4 > 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 5 }
def setC : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) }

theorem set_intersection_eq : setA ∩ setB = setC := by
  sorry

end set_intersection_eq_l821_821571


namespace tower_remainder_l821_821308

def tower_t (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | (n + 1) => tower_t n * 4

theorem tower_remainder (T : ℕ) : 
  (T = tower_t 9) → (T % 1000 = 536) :=
by
  sorry

end tower_remainder_l821_821308


namespace number_of_straight_A_students_l821_821327

-- Define the initial conditions and numbers
variables {x y : ℕ}

-- Define the initial student count and conditions on percentages
def initial_student_count := 25
def new_student_count := 7
def total_student_count := initial_student_count + new_student_count
def initial_percentage (x : ℕ) := (x : ℚ) / initial_student_count * 100
def new_percentage (x y : ℕ) := ((x + y : ℚ) / total_student_count) * 100

theorem number_of_straight_A_students
  (x y : ℕ)
  (h : initial_percentage x + 10 = new_percentage x y) :
  (x + y = 16) :=
sorry

end number_of_straight_A_students_l821_821327


namespace no_solution_eqn_l821_821608

theorem no_solution_eqn : ∀ x : ℝ, x ≠ -11 ∧ x ≠ -8 ∧ x ≠ -12 ∧ x ≠ -7 →
  ¬ (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  intros x h
  sorry

end no_solution_eqn_l821_821608


namespace four_digit_count_l821_821374

theorem four_digit_count : 
  let num_ways (comb: ℕ) := (nat.choose 4 comb) in
  (num_ways 1 + num_ways 2 + num_ways 1) = 14 :=
by
  intro num_ways,
  have h1 : num_ways 1 = nat.choose 4 1 := rfl,
  have h2 : num_ways 2 = nat.choose 4 2 := rfl,
  rw [h1, h2],
  norm_num,
  exact dec_trivial

-- The proof is omitted, as per instructions.

end four_digit_count_l821_821374


namespace magnitude_b_l821_821395

open Real

noncomputable def unit_vector (a : ℝ^3) : Prop :=
  ∥a∥ = 1

noncomputable def angle_between (a b : ℝ^3) : ℝ :=
  acos (a ⬝ b / (∥a∥ * ∥b∥))

theorem magnitude_b
  (a b : ℝ^3)
  (h1 : unit_vector a)
  (h2 : angle_between a b = π / 3)
  (h3 : a ⬝ b = 1) :
  ∥b∥ = 2 :=
by
  
  sorry

end magnitude_b_l821_821395


namespace net_population_increase_l821_821109

-- Definitions based on given conditions
def births_per_day : ℝ := 24 / 10
def deaths_per_day : ℝ := 2
def days_in_leap_year : ℕ := 366
def net_increase_per_day : ℝ := births_per_day - deaths_per_day
def net_increase_per_year : ℝ := net_increase_per_day * days_in_leap_year

-- Statement to prove
theorem net_population_increase :
  round (net_increase_per_year) = 100 :=
by
  -- Lean proof goes here
  sorry

end net_population_increase_l821_821109


namespace exercise_l821_821526

variables {α β : Type*} [plane α] [plane β] {m l : Type*} [line m] [line l]

-- Representing the intersection of two planes as a line
def intersect (α β : Type*) [plane α] [plane β] : Type* := l

-- If a line is parallel to both planes, then it is parallel to the line formed by their intersection
axiom parallel_to_planes_implies_parallel_to_intersection (h : intersect α β = l) :
  (parallel_to_plane m α ∧ parallel_to_plane m β) → parallel m l

-- Given conditions
variables (h1 : intersect α β = l) (h2 : parallel_to_plane m α) (h3 : parallel_to_plane m β)

theorem exercise : parallel m l :=
by {
  exact parallel_to_planes_implies_parallel_to_intersection h1 (and.intro h2 h3),
  sorry
}

end exercise_l821_821526


namespace relationship_s_t_l821_821936

theorem relationship_s_t (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 1 / 2 * a * b * (1 / 2) = 1 / 8) 
  (h5 : a * b * c = 1) :
  let s := sqrt a + sqrt b + sqrt c
  let t := 1 / a + 1 / b + 1 / c
  in t > s :=
by 
  sorry

end relationship_s_t_l821_821936


namespace exponent_equivalence_8_l821_821072

theorem exponent_equivalence_8 (x : ℝ) (h : 8^(3 * x) = 64) : 8^(-x) = 1 / 4 :=
by
  sorry

end exponent_equivalence_8_l821_821072


namespace hexagon_not_regular_l821_821723

theorem hexagon_not_regular (H : inscribed_hexagon c) (h1 : H.angle_1 = 120) (h2 : H.angle_2 = 120) (h3 : H.angle_3 = 120) : ¬(H.regular) :=
sorry

end hexagon_not_regular_l821_821723


namespace probability_two_white_balls_l821_821710

-- Definitions
def totalBalls : ℕ := 5
def whiteBalls : ℕ := 3
def blackBalls : ℕ := 2
def totalWaysToDrawTwoBalls : ℕ := Nat.choose totalBalls 2
def waysToDrawTwoWhiteBalls : ℕ := Nat.choose whiteBalls 2

-- Theorem statement
theorem probability_two_white_balls :
  (waysToDrawTwoWhiteBalls : ℚ) / totalWaysToDrawTwoBalls = 3 / 10 := by
  sorry

end probability_two_white_balls_l821_821710


namespace main_inequality_l821_821721

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_function_eq {x y : ℝ} (hx : x ∈ Ioi 1 ∨ x ∈ Iio (-1)) (hy : y ∈ Ioi 1 ∨ y ∈ Iio (-1)) :
  f (1 / x) + f (1 / y) = f ((x + y) / (1 + x * y))
axiom f_positive_in_interval {x : ℝ} (hx : x ∈ Ioo (-1) 0) : 0 < f x

theorem main_inequality (n : ℕ) (hn : 0 < n) :
  f (1 / 19) + f (1 / 29) + ∑ k in Finset.range n, f (1 / (k^2 + 7 * k + 11)) > f (1 / 2) := sorry

end main_inequality_l821_821721


namespace parallel_vectors_lambda_l821_821441

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821441


namespace inequality_proof_l821_821702

theorem inequality_proof
  (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) :
  bc^2 + ca^2 + ab^2 < b^2c + c^2a + a^2b :=
begin
  sorry
end

end inequality_proof_l821_821702


namespace minimum_value_of_f_l821_821874

noncomputable def f (t : ℝ) : ℝ := 4 * t - 3 / t

theorem minimum_value_of_f : 
  ∀ t ∈ set.Icc (1/2 : ℝ) 1, 
  ∃ c ∈ set.Icc (1/2 : ℝ) 1, 
  ∀ x ∈ set.Icc (1/2 : ℝ) 1, f x ≥ f c := 
begin
  sorry
end

example : f (1/2) = -4 := by sorry

end minimum_value_of_f_l821_821874


namespace length_of_segments_l821_821602

theorem length_of_segments (AB_length CB_length : ℝ) (k_iter : ℤ) (segment_length diagonal_length : ℝ)
  (h_AB_length : AB_length = 4)
  (h_CB_length : CB_length = 3)
  (h_k_iter : k_iter = 168)
  (h_diagonal : diagonal_length = real.sqrt (AB_length^2 + CB_length^2))
  (h_segment_length : ∀ k : ℤ, 1 ≤ k ∧ k ≤ k_iter - 1 → segment_length = diagonal_length * (k_iter - k) / k_iter) :
  (2 * (list.sum (list.map (λk, segment_length) (list.range k_iter))) - diagonal_length = 840) :=
sorry

end length_of_segments_l821_821602


namespace parallel_vectors_lambda_value_l821_821460

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821460


namespace sum_maximized_at_50_l821_821517

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Definitions of conditions
def is_decreasing_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d ∧ d < 0

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * d)) / 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
  a 0 + a 99 = 0

-- Theorem statement
theorem sum_maximized_at_50
  (h_seq : is_decreasing_arithmetic_sequence a d)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond : sequence_condition a) :
  ∀ n, S n ≤ S 50 :=
sorry

end sum_maximized_at_50_l821_821517


namespace direct_proportionality_straight_line_inverse_proportionality_hyperbola_l821_821256

open Function

-- Define the proportional relationships
def direct_proportionality (k : ℝ) : ℝ → ℝ := λ x, k * x
def inverse_proportionality (k : ℝ) : ℝ → ℝ := λ x, k / x

-- Theorem statements
theorem direct_proportionality_straight_line (k : ℝ) :
  ∃ (f : ℝ → ℝ), f = direct_proportionality k ∧ ( ∀ x : ℝ, f x = k * x ) :=
by
  sorry

theorem inverse_proportionality_hyperbola (k : ℝ) :
  ∃ (f : ℝ → ℝ), f = inverse_proportionality k ∧ ( ∀ x : ℝ, f x = k / x ) :=
by
  sorry

end direct_proportionality_straight_line_inverse_proportionality_hyperbola_l821_821256


namespace find_b_l821_821807

theorem find_b
  (a b c : ℚ)
  (h1 : (4 : ℚ) * a = 12)
  (h2 : (4 * (4 * b) = - (14:ℚ) + 3 * a)) :
  b = -(7:ℚ) / 2 :=
by sorry

end find_b_l821_821807


namespace Polya_theorem_l821_821598

variables {t : ℝ}
variables {Z : ℝ}
variables (φ : ℝ → ℝ)

-- Definitions based on the given conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_convex (f : ℝ → ℝ) := ∀ x y a b, a + b = 1 → 0 ≤ a → 0 ≤ b → f (a*x + b*y) ≤ a * f x + b * f y
def is_char_function (f : ℝ → ℝ) := ∃ (Z : ℝ), ∀ t, f t = (Complex.exp (Complex.I * t * Z)).re

-- Lean 4 statement of the theorem
theorem Polya_theorem :
  is_even φ →
  is_convex φ →
  (∀ t, 0 ≤ t → φ t ≥ 0) →
  (φ 0 = 1) →
  (t → 0 → φ t = 0) →
  is_char_function φ :=
sorry

end Polya_theorem_l821_821598


namespace smallest_n_modulo_l821_821676

theorem smallest_n_modulo (
  n : ℕ
) (h1 : 17 * n ≡ 5678 [MOD 11]) : n = 4 :=
by sorry

end smallest_n_modulo_l821_821676


namespace volume_ratio_l821_821888

-- Define the data for the problem
def greg_diameter := 4
def greg_height := 20

def violet_diameter := 12
def violet_height := 6

-- Compute radius
def radius (diameter : ℝ) : ℝ := diameter / 2

-- Compute volume of a cylinder
def cyl_volume (diameter : ℝ) (height : ℝ) : ℝ :=
  let r := radius diameter
  π * r^2 * height

-- Volumes of Greg's and Violet's containers
def greg_volume := cyl_volume greg_diameter greg_height
def violet_volume := cyl_volume violet_diameter violet_height

-- The main theorem to prove the volume ratio
theorem volume_ratio : greg_volume / violet_volume = 10 / 27 := by
  sorry

end volume_ratio_l821_821888


namespace no_valid_n_values_l821_821822

theorem no_valid_n_values :
  ¬ ∃ n : ℕ, (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_valid_n_values_l821_821822


namespace price_per_yellow_stamp_l821_821580

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l821_821580


namespace problem1_problem2_problem3_l821_821735

-- Definitions
def P (x : ℝ) (k : ℝ) : ℝ := 1 + k / x

def Q: ℝ → ℝ
| 10 := 110
| 20 := 120
| 25 := 125
| 30 := 120
| _ := 0

def f (x : ℝ) (k : ℝ) : ℝ :=
  if x ≤ 25 then (P x k) * (125 - |x - 25|)
  else (P x k) * (125 - |x - 25|)

-- Theorems to prove

-- Problem 1
theorem problem1 : ∃ k : ℝ, P 10 k * Q 10 = 121 ∧ k = 1 :=
begin
  use 1,
  sorry
end

-- Problem 2
theorem problem2 : Q 10 = 110 ∧ Q 20 = 120 ∧ Q 25 = 125 ∧ Q 30 = 120 → 
  (∃ a b : ℝ, ∀ x : ℝ, Q x = a * abs(x-25) + b) ∧ (∀ x, Q x = 125 - abs(x-25)) :=
begin
  sorry
end

-- Problem 3
theorem problem3 : ∃ x : ℝ, 1 ≤ x ∧ x ≤ 30 ∧ f x 1 = 121 ∧ ∀ y : ℝ, 1 ≤ y ∧ y ≤ 30 → f y 1 ≥ f x 1 :=
begin
  use 10,
  sorry
end

end problem1_problem2_problem3_l821_821735


namespace PetrovFamilySavings_l821_821618

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l821_821618


namespace length_of_shorter_parallel_side_l821_821353

-- Definitions of constants based on conditions
variable (a : ℝ) (b : ℝ) (h : ℝ) (A : ℝ)

-- Conditions
def length_of_longer_parallel_side := a = 30
def distance_between_parallel_sides := h = 16
def area_of_trapezium := A = 336

-- Formula for the area of a trapezium
def area_formula (a b h : ℝ) := 1/2 * (a + b) * h

-- The theorem stating that given the conditions, the shorter side length is 12 cm
theorem length_of_shorter_parallel_side
  (h_a : length_of_longer_parallel_side a)
  (h_h : distance_between_parallel_sides h)
  (h_A : area_of_trapezium A) :
  b = 12 :=
by
  -- Using the area formula with given conditions
  have eq1: 1/2 * (a + b) * h = A, from area_formula a b h
  -- Substituting the given conditions
  rw [h_a, h_h, h_A] at eq1
  -- Simplifying the equation: 1/2 * (30 + b) * 16 = 336
  sorry

end length_of_shorter_parallel_side_l821_821353


namespace jacque_suitcase_weight_l821_821980

noncomputable def suitcase_weight_return (original_weight : ℝ)
                                         (perfume_weight_oz : ℕ → ℝ)
                                         (chocolate_weight_lb : ℝ)
                                         (soap_weight_oz : ℕ → ℝ)
                                         (jam_weight_oz : ℕ → ℝ)
                                         (sculpture_weight_kg : ℝ)
                                         (shirt_weight_g : ℕ → ℝ)
                                         (oz_to_lb : ℝ)
                                         (kg_to_lb : ℝ)
                                         (g_to_kg : ℝ) : ℝ :=
  original_weight +
  (perfume_weight_oz 5 / oz_to_lb) +
  chocolate_weight_lb +
  (soap_weight_oz 2 / oz_to_lb) +
  (jam_weight_oz 2 / oz_to_lb) +
  (sculpture_weight_kg * kg_to_lb) +
  ((shirt_weight_g 3 / g_to_kg) * kg_to_lb)

theorem jacque_suitcase_weight :
  suitcase_weight_return 12 
                        (fun n => n * 1.2) 
                        4 
                        (fun n => n * 5) 
                        (fun n => n * 8)
                        3.5 
                        (fun n => n * 300) 
                        16 
                        2.20462 
                        1000 
  = 27.70 :=
sorry

end jacque_suitcase_weight_l821_821980


namespace max_airlines_l821_821971

theorem max_airlines (N : ℕ) (k : ℕ) 
  (cities : Type) [fintype cities] [decidable_eq cities]
  (connected : ∀ (x y : cities), x ≠ y → ∃ p : list (cities × cities), p.head = some (x, y))
  (disconnect : ∀ (c : fin k), ¬∃ p : list (finset (cities × cities)), 
    p.card = N - 1 ∧ ∀ (e : cities × cities), e ∈ p → e ∉ c) : 
  ∃ max_airlines, max_airlines = N-1 := 
sorry

end max_airlines_l821_821971


namespace graph1_higher_than_graph2_l821_821768

theorem graph1_higher_than_graph2 :
  ∀ (x : ℝ), (-x^2 + 2 * x + 3) ≥ (x^2 - 2 * x + 3) :=
by
  intros x
  sorry

end graph1_higher_than_graph2_l821_821768


namespace complement_M_l821_821882

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}
def C (s : Set ℝ) : Set ℝ := sᶜ -- complement of a set

theorem complement_M :
  C M = {x : ℝ | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l821_821882


namespace find_unit_vector_l821_821816

noncomputable def unit_vector : ℝ × ℝ × ℝ :=
  (⟨(√10 - √6) / 6, (2 * √6 + √10) / 6, 0⟩ : ℝ × ℝ × ℝ)

theorem find_unit_vector (x y : ℝ) :
  ( ∃ v : ℝ × ℝ × ℝ, v = ⟨x, y, 0⟩ ∧ 
    x^2 + y^2 = 1 ∧ 
    (x + y) / √2 = √3 / 2 ∧ 
    (-x + 2 * y) / √5 = 1 / √2) → 
  unit_vector = ⟨x, y, 0⟩ :=
by
  sorry

end find_unit_vector_l821_821816


namespace acceptable_arrangements_correct_l821_821944

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821944


namespace final_match_l821_821978

-- Definitions of players and conditions
inductive Player
| Antony | Bart | Carl | Damian | Ed | Fred | Glen | Harry

open Player

-- Condition definitions
def beat (p1 p2 : Player) : Prop := sorry

-- Given conditions
axiom Bart_beats_Antony : beat Bart Antony
axiom Carl_beats_Damian : beat Carl Damian
axiom Glen_beats_Harry : beat Glen Harry
axiom Glen_beats_Carl : beat Glen Carl
axiom Carl_beats_Bart : beat Carl Bart
axiom Ed_beats_Fred : beat Ed Fred
axiom Glen_beats_Ed : beat Glen Ed

-- The proof statement
theorem final_match : beat Glen Carl :=
by
  sorry

end final_match_l821_821978


namespace sin_alpha_plus_7pi_over_6_l821_821031

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) (h : cos (α - π / 3) = 3 / 4) :
  sin (α + 7 * π / 6) = -3 / 4 :=
by
  sorry

end sin_alpha_plus_7pi_over_6_l821_821031


namespace minimum_value_l821_821381

theorem minimum_value (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0)
  (h₃ : (m, 1) ∥ (1 - n, 1)) : 
  ∃ k : ℝ, k = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, x > 0 → y > 0 → ((x, 1) ∥ (1 - y, 1) → (1 / x + 2 / y) ≥ k )) :=
by {
  sorry
}

end minimum_value_l821_821381


namespace kylie_total_beads_used_l821_821992

noncomputable def beads_monday_necklaces : ℕ := 10 * 20
noncomputable def beads_tuesday_necklaces : ℕ := 2 * 20
noncomputable def beads_wednesday_bracelets : ℕ := 5 * 10
noncomputable def beads_thursday_earrings : ℕ := 3 * 5
noncomputable def beads_friday_anklets : ℕ := 4 * 8
noncomputable def beads_friday_rings : ℕ := 6 * 7

noncomputable def total_beads_used : ℕ :=
  beads_monday_necklaces +
  beads_tuesday_necklaces +
  beads_wednesday_bracelets +
  beads_thursday_earrings +
  beads_friday_anklets +
  beads_friday_rings

theorem kylie_total_beads_used : total_beads_used = 379 := by
  sorry

end kylie_total_beads_used_l821_821992


namespace tower_height_l821_821495

theorem tower_height (h : ℝ) (α : ℝ)
  (h_tan_α : tan α = h / 48)
  (h_tan_2α : tan (2 * α) = h / 18) : h = 24 :=
by
  sorry

end tower_height_l821_821495


namespace ways_to_make_change_l821_821905

theorem ways_to_make_change : ∃ ways : ℕ, ways = 60 ∧ (∀ (p n d q : ℕ), p + 5 * n + 10 * d + 25 * q = 55 → True) := 
by
  -- The proof will go here
  sorry

end ways_to_make_change_l821_821905


namespace solution_to_system_l821_821603

open Real

def system_of_equations (x y z w : ℝ) : Prop :=
  x^2 + 2*y^2 + 2*z^2 + w^2 = 43 ∧
  y^2 + z^2 + w^2 = 29 ∧
  5*z^2 - 3*w^2 + 4*x*y + 12*y*z + 6*z*x = 95

theorem solution_to_system :
  {p : ℝ × ℝ × ℝ × ℝ // system_of_equations p.1 p.2 p.3 p.4} =
  {⟨(1, 2, 3, 4), _⟩, ⟨(1, 2, 3, -4), _⟩, ⟨(-1, -2, -3, 4), _⟩, ⟨(-1, -2, -3, -4), _⟩} :=
sorry

end solution_to_system_l821_821603


namespace smallest_k_l821_821131

noncomputable def find_min_k : ℕ :=
  let k := 58
  in k

theorem smallest_k : find_min_k = 58 := by
  sorry

end smallest_k_l821_821131


namespace area_of_triangle_ABC_l821_821749

-- Define the points and properties of the triangle
variables {A B C D E : Type}
variables [IsMidpoint D A B] [IsMidpoint E A C]

-- Define the area function for the triangle
def area (A B C : Type) : ℝ := sorry

-- Define the areas of the shaded regions
variables (shaded_area_1 shaded_area_2 : ℝ)
axiom shaded_diff : shaded_area_1 - shaded_area_2 = 504.25

-- The final statement to prove the area of triangle ABC
theorem area_of_triangle_ABC (h1 : area D E A = (1 : ℝ) / 4 * area A B C)
                              (h2 : shaded_area_1 - shaded_area_2 = 504.25) : 
                             area A B C = 2017 := 
sorry

end area_of_triangle_ABC_l821_821749


namespace curve_equation_of_C_minimum_distance_coord_l821_821862

section part_I
variable (C : Set (ℝ × ℝ)) (F : ℝ × ℝ := (1, 0))

-- Definition of the curve C
def curve_c (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  (x ≥ 0) ∧ (x + 1 = Real.sqrt ((x - 1) ^ 2 + y ^ 2))

-- The proof statement for the equation of curve C
theorem curve_equation_of_C :
  curve_c C (0, 0) →
  (∀ p ∈ C, curve_c p) →
  (∀ x ≥ 0, ∃ y, curve_c (x, y)) →
  ∀ p ∈ C, (let (x, y) := p in y ^ 2 = 4 * x) := by
    intros
    sorry
end part_I

section part_II
variable (C : Set (ℝ × ℝ))

-- Definition of the tangent line
def tangent_line (p t : ℝ) :=
  let (x, y) := p in
  (x + y + t = 0)

-- The proof statement for the minimum distance coordinate
theorem minimum_distance_coord :
  curve_c C (0, 0) →
  (∀ p ∈ C, curve_c p) →
  (let (x, y) := (1, -2) in
    ∀ line, line = tangent_line (x, y) 1 →
    line = x + y + 4 → y = -2 ∧ x = 1) := by
    intros
    sorry
end part_II

end curve_equation_of_C_minimum_distance_coord_l821_821862


namespace least_positive_integer_mod_conditions_l821_821673

theorem least_positive_integer_mod_conditions :
  ∃ N : ℕ, (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 11 = 10) ∧ N = 4619 :=
by
  sorry

end least_positive_integer_mod_conditions_l821_821673


namespace f_2011_l821_821338

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_defined_segment : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_2011 : f 2011 = -2 := by
  sorry

end f_2011_l821_821338


namespace inequality_proof_l821_821846

theorem inequality_proof (a b c d e f : ℝ) (H : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821846


namespace monthly_savings_correct_l821_821624

-- Define each component of the income and expenses
def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

-- Define the main theorem
theorem monthly_savings_correct :
  let I := parents_salary + grandmothers_pension + sons_scholarship in
  let E := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses in
  let Surplus := I - E in
  let Deposit := (Surplus * 10) / 100 in
  let AmountSetAside := Surplus - Deposit in
  AmountSetAside = 16740 :=
by sorry

end monthly_savings_correct_l821_821624


namespace lines_not_parallel_not_perpendicular_to_same_plane_l821_821017

variable (m n : Type) [Line m] [Line n]
variable (α β : Type) [Plane α] [Plane β]

theorem lines_not_parallel_not_perpendicular_to_same_plane
  (hmn : m ≠ n)
  (hαβ : α ≠ β)
  (h : ¬ Parallel m n) :
  ¬ (Perpendicular m α ∧ Perpendicular n α) := sorry

end lines_not_parallel_not_perpendicular_to_same_plane_l821_821017


namespace seating_arrangements_l821_821951

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821951


namespace modified_cube_edges_l821_821733

-- Definitions of the conditions
def largeCubeEdges (side_length : ℝ) : ℕ := 12
def smallCubeEdgesRemoved (num_corners : ℕ) (new_edges_per_corner : ℕ) : ℕ := num_corners * new_edges_per_corner

-- The proof problem statement
theorem modified_cube_edges (large_side : ℝ) (small_side : ℝ) (num_corners : ℕ) 
  (new_edges_per_corner : ℕ) : large_side = 10 → small_side = 5 → num_corners = 8 → new_edges_per_corner = 6 →
  largeCubeEdges large_side + smallCubeEdgesRemoved num_corners new_edges_per_corner = 60 :=
by
  intros
  simp [largeCubeEdges, smallCubeEdgesRemoved, *]
  rw [*, add_comm]
  sorry

end modified_cube_edges_l821_821733


namespace integral_ln_2_l821_821233

theorem integral_ln_2 : ∫ x in 0..1, 1 / (1 + x) = Real.log 2 := by
  sorry

end integral_ln_2_l821_821233


namespace g_eq_and_range_l821_821859

theorem g_eq_and_range (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = 3 ^ x) →
  f (a + 2) = 18 →
  (∀ x, x ∈ set.Icc 0 1 → g x = 3 ^ (a * x) - 4 ^ x) →
  (∀ x, x ∈ set.Icc 0 1 → g x = 2 ^ x - 4 ^ x) ∧ (set.range g = set.Icc (-2 : ℝ) 0) :=
by
  sorry

end g_eq_and_range_l821_821859


namespace max_value_of_sin_function_l821_821214

theorem max_value_of_sin_function (ϕ : ℝ) : 
  ∃ x_max : ℝ, ∀ x : ℝ, 
    (f : ℝ → ℝ) (f x = sin (x + 2 * ϕ) - 2 * sin ϕ * cos (x + ϕ)) → 
    (f x ≤ f x_max) ∧ (f x_max = 1) := 
by
  sorry

end max_value_of_sin_function_l821_821214


namespace intersection_points_of_curve_with_axes_l821_821213

theorem intersection_points_of_curve_with_axes :
  (∃ t : ℝ, (-2 + 5 * t = 0) ∧ (1 - 2 * t = 1/5)) ∧
  (∃ t : ℝ, (1 - 2 * t = 0) ∧ (-2 + 5 * t = 1/2)) :=
by {
  -- Proving the intersection points with the coordinate axes
  sorry
}

end intersection_points_of_curve_with_axes_l821_821213


namespace cannot_reach_eighth_vertex_l821_821099

def vertices : set (ℕ × ℕ × ℕ) := { (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0) }

def symmetric_point (a b : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2, 2 * b.3 - a.3)

theorem cannot_reach_eighth_vertex : ∀ (a b : ℕ × ℕ × ℕ), 
  a ∈ vertices → b ∈ vertices → 
  ¬ (symmetric_point a b = (1, 1, 1)) :=
by
  intros a b ha hb
  sorry

end cannot_reach_eighth_vertex_l821_821099


namespace original_number_is_14_l821_821091

def two_digit_number_increased_by_2_or_4_results_fourfold (x : ℕ) : Prop :=
  (x >= 10) ∧ (x < 100) ∧ 
  (∃ (a b : ℕ), a + 2 = ((x / 10 + 2) % 10) ∧ b + 2 = (x % 10)) ∧
  (4 * x = ((x / 10 + 2) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 2) * 10 + (x % 10 + 4)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 4)))

theorem original_number_is_14 : ∃ x : ℕ, two_digit_number_increased_by_2_or_4_results_fourfold x ∧ x = 14 :=
by
  sorry

end original_number_is_14_l821_821091


namespace speed_of_man_is_approx_4_99_l821_821738

noncomputable def train_length : ℝ := 110  -- meters
noncomputable def train_speed : ℝ := 50  -- km/h
noncomputable def time_to_pass_man : ℝ := 7.2  -- seconds

def mps_to_kmph (speed : ℝ) : ℝ := speed * 3.6

noncomputable def relative_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def relative_speed_kmph : ℝ :=
  mps_to_kmph (relative_speed train_length time_to_pass_man)

noncomputable def speed_of_man (relative_speed_kmph : ℝ) (train_speed : ℝ) : ℝ :=
  relative_speed_kmph - train_speed

theorem speed_of_man_is_approx_4_99 :
  abs (speed_of_man relative_speed_kmph train_speed - 4.99) < 0.01 :=
by
  sorry

end speed_of_man_is_approx_4_99_l821_821738


namespace decagon_labeling_is_3840_l821_821183

-- Define the conditions in Lean 4
def vertices := {A, B, C, D, E, F, G, H, I, J, K : ℕ}
def digits := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

noncomputable def sum_digits : ℕ := 55

-- Define the equation for the common sum x
def common_sum (x K : ℕ) : Prop := 5 * x = sum_digits + 4 * K

-- Determine valid values for K
def valid_K (K : ℕ) : Prop := 55 + 4 * K ≡ 0 [MOD 5]

-- Define the final equality that we need to prove the number of ways
def number_of_ways (n : ℕ) : Prop :=
  ∀ K, K ∈ {5, 10} → 
  ∃ x, common_sum x K ∧ 
       (n = 3840)

theorem decagon_labeling_is_3840 : number_of_ways 3840 := 
sorry

end decagon_labeling_is_3840_l821_821183


namespace projection_eq_l821_821060

open Real EuclideanSpace

-- Define the vectors a and b
def a : EuclideanSpace ℝ (Fin 2) := ![4, 3]
def b : EuclideanSpace ℝ (Fin 2) := ![-5, 12]

-- Define the dot product function for 2D vectors
def dot_product (u v : EuclideanSpace ℝ (Fin 2)) : ℝ := (u 0 * v 0) + (u 1 * v 1)

-- Define the magnitude function for 2D vectors
def magnitude (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

-- Prove that the projection of a onto b is 16/13
theorem projection_eq : (dot_product a b) / (magnitude b) = 16 / 13 := by
  sorry

end projection_eq_l821_821060


namespace books_bought_at_yard_sales_l821_821156

variables (initial_books book_club_books bought_books daughter_books mother_books donated_books sold_books end_of_year_books : ℕ)

-- Given conditions
def conditions : Prop :=
  initial_books = 72 ∧
  book_club_books = 12 ∧
  bought_books = 5 ∧
  daughter_books = 1 ∧
  mother_books = 4 ∧
  donated_books = 12 ∧
  sold_books = 3 ∧
  end_of_year_books = 81

-- Prove the number of books bought at yard sales
theorem books_bought_at_yard_sales (h : conditions) : 
  initial_books + book_club_books + bought_books + daughter_books + mother_books - donated_books - sold_books + x = end_of_year_books →
  x = 2 :=
sorry

end books_bought_at_yard_sales_l821_821156


namespace count_positive_numbers_l821_821320

theorem count_positive_numbers :
  let S := {-7, 0, -3, 4 / 3, 9100, -0.7}
  (S.filter (λ x => x > 0)).card = 2 :=
by
  sorry

end count_positive_numbers_l821_821320


namespace cannot_reach_eighth_vertex_l821_821100

def vertices : set (ℕ × ℕ × ℕ) := { (0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0) }

def symmetric_point (a b : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2, 2 * b.3 - a.3)

theorem cannot_reach_eighth_vertex : ∀ (a b : ℕ × ℕ × ℕ), 
  a ∈ vertices → b ∈ vertices → 
  ¬ (symmetric_point a b = (1, 1, 1)) :=
by
  intros a b ha hb
  sorry

end cannot_reach_eighth_vertex_l821_821100


namespace parallel_vectors_lambda_value_l821_821464

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821464


namespace angle_DSC_150_l821_821556

variables (A B C D S : Type*)
variables [square ABCD] [inside_square S ABCD] [equilateral_triangle ABS]

theorem angle_DSC_150 (A B C D S : Point)
  (sq : is_square A B C D)
  (equil_tri : is_equilateral_triangle A B S) :
  angle D S C = 150 :=
  by sorry

end angle_DSC_150_l821_821556


namespace inequality_proof_l821_821848

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l821_821848


namespace part1_part2_l821_821868

-- Definition of ellipse and line
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (x y m : ℝ) : Prop := y = x + m

-- Problem (1): For what value of m does the line intersect the ellipse?
theorem part1 (m : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ (-real.sqrt 5 / 2 ≤ m ∧ m ≤ real.sqrt 5 / 2) := sorry

-- Problem (2): If the chord cut by the line from the ellipse has a length of 2√10/5, find the equation of the line
theorem part2 (line_length : ℝ) : 
  line_length = 2 * real.sqrt 10 / 5 → 
  (∀ m : ℝ, (∃ x1 y1 x2 y2 : ℝ, ellipse x1 y1 ∧ ellipse x2 y2 ∧ line x1 y1 m ∧ line x2 y2 m ∧ 
  (dist (x1, y1) (x2, y2) = 2 * real.sqrt 10 / 5)) → m = 0) := sorry

-- Utility function to calculate Euclidean distance
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end part1_part2_l821_821868


namespace floor_sufficient_but_not_necessary_l821_821910

theorem floor_sufficient_but_not_necessary (x y : ℝ) :
    (⌊x⌋ = ⌊y⌋ → |x - y| < 1) ∧ (∃ x y, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋) := by
  split
  -- Sufficient part: ⌊x⌋ = ⌊y⌋ → |x - y| < 1
  intros h
  have h1 : x - 1 < x := sub_lt_self x zero_lt_one
  have h2 : y - 1 < y := sub_lt_self y zero_lt_one
  have hxy1 : x < ⌊x⌋ + 1 := lt_add_one _
  have hxy2 : y < ⌊y⌋ + 1 := lt_add_one _
  calc
    |x - y| = |⌊x⌋ - ⌊y⌋ + (x - ⌊x⌋) - (y - ⌊y⌋)| : by sorry
    ... = |x - y + (⌊x⌋ - ⌊y⌋)| : by sorry
    ... = |(x - y) + (⌊x⌋ - ⌊y⌋)| : by sorry
    ... = |⌊x⌋ - ⌊y⌋ + (x - ⌊x⌋) - (y - ⌊y⌋)| : by sorry
    ... < |1 + 1| : by sorry
    ... = 1 : by sorry
  -- Not necessary part: ∃ x y, |x - y| < 1 ∧ ⌊x⌋ ≠ ⌊y⌋
  use 1.9, 2.1
  split
  -- |1.9 - 2.1| < 1
  apply abs_sub_lt_iff.mpr
  split
  { linarith }
  { linarith }
  -- ⌊1.9⌋ ≠ ⌊2.1⌋
  show ⌊1.9⌋ ≠ ⌊2.1⌋
  have h3 : ⌊1.9⌋ = 1 := by sorry
  have h4 : ⌊2.1⌋ = 2 := by sorry
  rw [h3, h4]
  exact one_ne_two

end floor_sufficient_but_not_necessary_l821_821910


namespace range_of_a_l821_821394

variable {α : Type} [LinearOrderedField α]

def A (a : α) : Set α := {x | |x - a| ≤ 1}

def B : Set α := {x | x^2 - 5*x + 4 ≥ 0}

theorem range_of_a (a : α) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l821_821394


namespace inequality_proof_l821_821843

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821843


namespace B_work_rate_l821_821706

theorem B_work_rate (B : ℕ) (A_rate C_rate : ℚ) 
  (A_work : A_rate = 1 / 6)
  (C_work : C_rate = 1 / 8 * (1 / 6 + 1 / B))
  (combined_work : 1 / 6 + 1 / B + C_rate = 1 / 3) : 
  B = 28 :=
by 
  sorry

end B_work_rate_l821_821706


namespace gravitational_force_at_40000_miles_l821_821211

-- Definitions based on conditions
def gravitational_force (d : ℝ) : ℝ := 8000000000 / (d * d)

-- Goal statement
theorem gravitational_force_at_40000_miles :
  gravitational_force 40000 = 5 :=
by
  sorry

end gravitational_force_at_40000_miles_l821_821211


namespace b_1968_eq_b_3968_eq_sum_1968_eq_sum_3968_eq_l821_821271

def sequence_a : ℕ → ℕ := sorry
def sequence_b (n : ℕ) : ℕ :=
  if n = 1 then sequence_a 1
  else if n = 2 then sequence_a 2 - sequence_a 1
  else if n = 3 then sequence_a 2
  else if n = 4 then sequence_a 2 + sequence_a 1
  -- Continue pattern as necessary or use recursive formulation

def sum_sequence_b (m : ℕ) : ℕ :=
  ∑ i in Finset.range m, sequence_b (i + 1)

-- Statements for given problem:
theorem b_1968_eq : sequence_b 1968 = sequence_a 45 - sequence_a 13 := sorry

theorem b_3968_eq : sequence_b 3968 = sequence_a 63 + sequence_a 61 := sorry

theorem sum_1968_eq :
  sum_sequence_b 1968 = (∑ i in Finset.range 44, (2 * i + 1) * sequence_a (i + 1)) + 32 * sequence_a 45 - (∑ i in Finset.range 13, sequence_a (44 - i)) := sorry

theorem sum_3968_eq :
  sum_sequence_b 3968 = (∑ i in Finset.range 62, (2 * i + 1) * sequence_a (i + 1)) + 124 * sequence_a 63 - sequence_a 62 := sorry

end b_1968_eq_b_3968_eq_sum_1968_eq_sum_3968_eq_l821_821271


namespace unique_id_tags_div_10_eq_192_l821_821716

def count_tags (SAFE_digits : Finset Char) : ℕ :=
  720 + (Finset.card (Finset.powersetLen 2 ('2' :: '0' :: SAFE_digits).toList) * 
        (Finset.card (Finset.powersetLen 3 (SAFE_digits.filter (λ c, c ≠ '2')).toList) * 
        (Nat.factorial 3))) + 
       (Finset.card (Finset.powersetLen 2 ('0' :: '2' :: SAFE_digits).toList) *
        (Finset.card (Finset.powersetLen 3 (SAFE_digits.filter (λ c, c ≠ '0')).toList) *
        (Nat.factorial 3)))

theorem unique_id_tags_div_10_eq_192 
  (SAFE : Finset Char) 
  (H_SAFE: SAFE = {'S', 'A', 'F', 'E'}) 
  (twenty_twenty_two : Finset Char) 
  (H_2022: twenty_twenty_two = {'2', '0', '2', '2'}) : 
  count_tags (SAFE ∪ twenty_twenty_two) / 10 = 192 := 
by 
  sorry

end unique_id_tags_div_10_eq_192_l821_821716


namespace largest_coefficient_term_in_binomial_expansion_l821_821530

theorem largest_coefficient_term_in_binomial_expansion (n : ℕ) (h : 0 < n) :
  ∃ k, k = n + 1 ∧ ∀ i (hi : 0 ≤ i ∧ i ≤ 2 * n), (n.choose i ≤ n.choose k) :=
begin
  sorry
end

end largest_coefficient_term_in_binomial_expansion_l821_821530


namespace isosceles_triangle_line_equation_l821_821725

open Real

theorem isosceles_triangle_line_equation :
  ∃ (l' : ℝ → ℝ → Prop), 
    line_through (3, 3) l' ∧ 
    is_isosceles_triangle_with_base_x_axis (x - 2 * y + 3 = 0) l' ∧ 
    (∀ x y, l' x y ↔ x + 2 * y - 9 = 0) :=
sorry

/-- The definition of a line through a given point (x1, y1). -/
def line_through (p : ℝ × ℝ) (l' : ℝ → ℝ → Prop) : Prop :=
  ∃ m b, ∀ x y, l' x y ↔ y = m * x + b ∧ y = p.2 ∧ x = p.1

/-- The definition of an isosceles triangle with the x-axis as its base formed by two lines. -/
def is_isosceles_triangle_with_base_x_axis (l : ℝ → ℝ → Prop) (l' : ℝ → ℝ → Prop) : Prop :=
  ∃ x1 y1 x2 y2,
    (x1 ≠ x2) ∧
    l x1 y1 ∧ l x2 y2 ∧ 
    l' x1 y1 ∧ l' x2 y2 ∧ 
    y1 = 0 ∧ y2 = 0

end isosceles_triangle_line_equation_l821_821725


namespace minimum_people_in_photos_l821_821933

noncomputable def minimum_distinct_people (photos : List (ℕ × ℕ × ℕ)) : ℕ :=
  if h : photos.length = 10 
     ∧ photos.Nodup 
     ∧ (∀ (p : ℕ × ℕ × ℕ) (h : p ∈ photos), p.1 ≠ p.2 ∧ p.1 ≠ p.3 ∧ p.2 ≠ p.3) 
  then 16
  else 0

theorem minimum_people_in_photos : minimum_distinct_people [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21), (22, 23, 24), (25, 26, 27), (28, 29, 30)] = 16 :=
by
  sorry

end minimum_people_in_photos_l821_821933


namespace permutations_with_k_in_first_position_l821_821647

noncomputable def numberOfPermutationsWithKInFirstPosition (N k : ℕ) (h : k < N) : ℕ :=
  (2 : ℕ)^(N-1)

theorem permutations_with_k_in_first_position (N k : ℕ) (h : k < N) :
  numberOfPermutationsWithKInFirstPosition N k h = (2 : ℕ)^(N-1) :=
sorry

end permutations_with_k_in_first_position_l821_821647


namespace a_5_equals_11_l821_821420

def a : ℕ → ℕ
| 0       := 1  -- Note: This should be a_1 = 1. Lean uses 0-based indexing
| (n + 1) := a n + n

theorem a_5_equals_11 : a 4 = 11 := -- Lean uses 0-based indexing, so a_5 is actually a 4
by {
  sorry
}

end a_5_equals_11_l821_821420


namespace no_real_solution_equation_l821_821191

theorem no_real_solution_equation :
  ∀ x : ℝ, ¬ (x + real.sqrt (2 * x - 3) = 5) :=
by 
  intro x
  sorry

end no_real_solution_equation_l821_821191


namespace solve_fx_eq_inverse_fx_l821_821772

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

theorem solve_fx_eq_inverse_fx : ∃ x, f x = (1 + real.sqrt((1 + x) / 3)) ∧ x = 0 := by
  sorry

end solve_fx_eq_inverse_fx_l821_821772


namespace max_height_reaches_45_l821_821304

-- Problem Conditions
def height (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 25

-- Question: Prove the maximum height is 45 feet
theorem max_height_reaches_45 :
  ∃ t_max : ℝ, (∀ t : ℝ, height t ≤ height t_max) ∧ height t_max = 45 :=
by
  sorry

end max_height_reaches_45_l821_821304


namespace log_base_2_condition_l821_821275

theorem log_base_2_condition (x : ℝ) : log 2 x < 1 ↔ 0 < x ∧ x < 2 :=
sorry

end log_base_2_condition_l821_821275


namespace equilateral_polygon_circumscribed_is_equiangular_l821_821279

theorem equilateral_polygon_circumscribed_is_equiangular
  (P : Type) [MetricSpace P] [EuclideanGeometry P] {p : Polygon P} {c : Circle P}
  (h_equilateral : EquilateralPolygon p) (h_circumscribed : CircumscribedPolygon p c) :
  EquiangularPolygon p := 
sorry

end equilateral_polygon_circumscribed_is_equiangular_l821_821279


namespace conic_through_four_common_points_l821_821913

variable {R : Type*} [CommRing R]
variables {x y : R} {F1 F2 : R → R → R}

theorem conic_through_four_common_points 
  (h1 : ∀ x y : R, F1 x y = 0 → F2 x y = 0 → ∃! p : R × R, gamma1 p = 0 ∧ gamma2 p = 0) 
  (h2 : F1 x y = 0 → F2 x y = 0): 
  ∃ λ μ : R, ∀ (x y : R), (λ * F1 x y + μ * F2 x y = 0) :=
sorry

end conic_through_four_common_points_l821_821913


namespace B_alone_work_days_l821_821709

theorem B_alone_work_days (B : ℕ) (A_work : ℝ) (C_work : ℝ) (total_payment : ℝ) :
  (A_work = 1 / 6) →
  (total_payment = 3200) →
  (C_work = (400 / total_payment) * (1 / 3)) →
  (A_work + 1 / B + C_work = 1 / 3) →
  B = 8 :=
begin
  intros hA_work htotal_payment hC_work hcombined_work,
  sorry,
end

end B_alone_work_days_l821_821709


namespace probability_product_multiple_of_4_l821_821986

/-- Juan rolls a fair regular dodecahedral die (12 sides) marked with numbers 1 through 12.
    Amal rolls a fair eight-sided die marked with numbers 1 through 8.
    Prove that the probability that the product of the two rolls is a multiple of 4 is 7/16.
-/
theorem probability_product_multiple_of_4 
  (Juan_rolls : ℕ → Prop)
  (Amal_rolls : ℕ → Prop)
  (dodecahedral_die : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
  (eight_sided_die : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}) : 
  ( ∑ n in dodecahedral_die, if n % 4 = 0 then 1 else 0) / 12 * 
  ( ∑ m in eight_sided_die, if m % 4 = 0 then 1 else 0) / 8 +
  ( 1 - ( ∑ n in dodecahedral_die, if n % 4 = 0 then 1 else 0) / 12 * 
  ( ∑ m in eight_sided_die, if m % 4 = 0 then 1 else 0) / 8 ) = 7/16 :=
sorry

end probability_product_multiple_of_4_l821_821986


namespace matrix_identity_l821_821560
open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, 5]]

theorem matrix_identity :
  ∃ (r s : ℝ), (matrix_pow B 4) = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
  ⟨165, 116, by sorry⟩

end matrix_identity_l821_821560


namespace find_sum_of_reciprocals_l821_821650

noncomputable def a : ℕ → ℕ
| 0         := 0  
| 1         := 1
| (n + 1) := a n + n + 1

theorem find_sum_of_reciprocals :
  ∑ n in Finset.range 20, (1 / (a (n + 1) : ℚ)) = 40 / 21 :=
  sorry

end find_sum_of_reciprocals_l821_821650


namespace parallel_vectors_lambda_l821_821473

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821473


namespace seating_arrangements_l821_821954

open Nat

theorem seating_arrangements (n : ℕ) (h_n : n = 8) (alice : Fin n) (bob : Fin n) (h_alice : alice ≠ bob) :
  let total_arrangements := fact n,
      combined_arrangements := fact (n - 1) * 2,
      valid_arrangements := total_arrangements - combined_arrangements
  in valid_arrangements = 30240 := by
  sorry

end seating_arrangements_l821_821954


namespace ellipse_major_axis_length_l821_821840

-- Definitions of the points involved in the problem
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (1, 3)
def P4 : ℝ × ℝ := (3, 3)
def P5 : ℝ × ℝ := (-1, 3 / 2)

-- The goal is to prove that the length of the major axis of the ellipse passing
-- through these points, with axes parallel to the coordinate axes, is 5.
theorem ellipse_major_axis_length :
  (major_axis_length (unique_ellipse P1 P2 P3 P4 P5)) = 5 :=
sorry

end ellipse_major_axis_length_l821_821840


namespace parallel_vectors_lambda_l821_821491

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821491


namespace decreasing_interval_of_function_l821_821357

theorem decreasing_interval_of_function :
  (∀ k : ℤ, ∀ x : ℝ, 
    x ∈ set.Icc ((↑k * (2 * real.pi) + real.pi / 12) / 3) 
              ((↑k * (2 * real.pi) + (7 * real.pi / 36)) / 3) → 
    continuous_at (λ x, sqrt (2 * real.sin (3*x + real.pi / 4) - 1)) x) := sorry

end decreasing_interval_of_function_l821_821357


namespace factorization_of_x_squared_minus_64_l821_821800

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821800


namespace complex_conjugate_quadrant_l821_821221

def z : ℂ := (2 - complex.I) / (2 + complex.I)
def z_conj : ℂ := complex.conj z

theorem complex_conjugate_quadrant :
  z_conj.re > 0 ∧ z_conj.im > 0 := by
  sorry

end complex_conjugate_quadrant_l821_821221


namespace D_coordinates_l821_821404

namespace Parallelogram

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 2 }
def C : Point := { x := 3, y := 1 }

theorem D_coordinates :
  ∃ D : Point, D = { x := 2, y := -1 } ∧ ∀ A B C D : Point, 
    (B.x - A.x, B.y - A.y) = (D.x - C.x, D.y - C.y) := by
  sorry

end Parallelogram

end D_coordinates_l821_821404


namespace angle_RPS_is_45_degrees_l821_821972

theorem angle_RPS_is_45_degrees
  (Q R S P : Type)
  (angle_PQS angle_QPR angle_PSQ : ℝ)
  (h1 : angle_PQS = 32)
  (h2 : angle_QPR = 58)
  (h3 : angle_PSQ = 45)
  (hQRS : angle Q S R = 180)
  : angle R P S = 45 := by
  -- Proof goes here
  sorry

end angle_RPS_is_45_degrees_l821_821972


namespace price_reduction_l821_821737

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end price_reduction_l821_821737


namespace sweater_difference_l821_821126

noncomputable def sweaters_knit_per_day : Type :=
  ℕ → ℕ

def total_sweaters_knit (s : sweaters_knit_per_day) : ℕ :=
  s 1 + s 2 + s 3 + s 4 + s 5 -- Monday + Tuesday + Wednesday + Thursday + Friday

axiom knit_conditions (s : sweaters_knit_per_day) :
  s 1 = 8 ∧ -- Monday
  s 2 = 10 ∧ -- Tuesday
  s 5 = 4 ∧ -- Friday
  total_sweaters_knit s = 34 -- total sweaters knit in the week

theorem sweater_difference (s : sweaters_knit_per_day) (h : knit_conditions s) :
  |s 2 - (s 3 + s 4)| = 2 := by
  sorry

end sweater_difference_l821_821126


namespace factorization_of_x_squared_minus_64_l821_821798

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821798


namespace log_7_of_56_l821_821029

theorem log_7_of_56 (a b : ℝ) (h1 : 2^a = 3) (h2 : 3^b = 7) : log 7 56 = 1 + 3 / (a * b) :=
by
  sorry

end log_7_of_56_l821_821029


namespace average_age_with_teacher_l821_821202

theorem average_age_with_teacher (A : ℕ) (h : 21 * 16 = 20 * A + 36) : A = 15 := by
  sorry

end average_age_with_teacher_l821_821202


namespace problem_a_b_n_geq_1_l821_821128

theorem problem_a_b_n_geq_1 (a b n : ℕ) (h1 : a > b) (h2 : b > 1) (h3 : Odd b) (h4 : n > 0)
  (h5 : b^n ∣ a^n - 1) : a^b > 3^n / n := 
by 
  sorry

end problem_a_b_n_geq_1_l821_821128


namespace incorrect_expressions_l821_821557

-- Definitions for the conditions
def F : ℝ := sorry   -- F represents a repeating decimal
def X : ℝ := sorry   -- X represents the t digits of F that are non-repeating
def Y : ℝ := sorry   -- Y represents the u digits of F that repeat
def t : ℕ := sorry   -- t is the number of non-repeating digits
def u : ℕ := sorry   -- u is the number of repeating digits

-- Statement that expressions (C) and (D) are incorrect
theorem incorrect_expressions : 
  ¬ (10^(t + 2 * u) * F = X + Y / 10 ^ u) ∧ ¬ (10^t * (10^u - 1) * F = Y * (X - 1)) :=
sorry

end incorrect_expressions_l821_821557


namespace general_term_of_seq_l821_821150

def Seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * S n + 1) ∧ (∀ n, S (n + 1) = S n + a (n + 1))

theorem general_term_of_seq
  (a S : ℕ → ℕ)
  (h : Seq a S)
  : ∀ n, a n = 3^(n - 1) :=
by
  sorry

end general_term_of_seq_l821_821150


namespace fruit_drawing_orders_l821_821687

theorem fruit_drawing_orders : 
  (∃ (A P Pe : Type) (draws : list (A ⊕ P ⊕ Pe) → list (A ⊕ P ⊕ Pe) → ℕ),
    draws [A.inl, A.inr (P.inl), A.inr (P.inr Pe)] [A.inl, A.inr (P.inl), A.inr (P.inr Pe)] = 6) :=
sorry

end fruit_drawing_orders_l821_821687


namespace problem1_monotonicity_intervals_problem2_monotonic_increasing_on_R_problem3_monotonic_decreasing_on_interval_l821_821047

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^3 - a*x - 1

-- Problem 1: For a = 3, find the intervals of monotonicity
theorem problem1_monotonicity_intervals :
  ∀ x : ℝ, (has_deriv_at (λ x, f x 3) (3*x^2 - 3) x) := by
sorry

-- Problem 2: Prove the range of a for which f(x) is monotonically increasing on ℝ
theorem problem2_monotonic_increasing_on_R :
  (∀ x : ℝ, deriv (λ x, f x a) x ≥ 0) ↔ (a ≤ 0) := by
sorry

-- Problem 3: Prove the existence of a such that f(x) is monotonically decreasing on (-1, 1)
theorem problem3_monotonic_decreasing_on_interval :
  ∃ a : ℝ, (a ≥ 3 ∧ (∀ x ∈ Ioo (-1 : ℝ) 1, deriv (λ x, f x a) x ≤ 0)) := by
sorry

end problem1_monotonicity_intervals_problem2_monotonic_increasing_on_R_problem3_monotonic_decreasing_on_interval_l821_821047


namespace possible_values_of_m_l821_821076

theorem possible_values_of_m (m : ℝ) (h : (m^2 - 36) > 0) : m ∈ set.Ioo ⊤ (-6) ∪ set.Ioo 6 ⊤ :=
sorry

end possible_values_of_m_l821_821076


namespace arithmetic_seq_and_sum_l821_821836

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

def sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_and_sum (a1 a5 sn : ℤ) (n : ℕ) (d : ℤ)
  (h1 : a1 = 1)
  (h2 : a5 = -3)
  (h3 : sn = -44) :
  arithmetic_sequence a1 d 5 = a5 ∧ sum_of_arithmetic_sequence a1 d n = sn →
  (arithmetic_sequence a1 d n = 2 - n ∧ n = 11) :=
begin
  intros h,
  split,
  { -- Proving general term formula
    cases h with h_seq h_sum,
    sorry },
  { -- Proving the value of n
    cases h with h_seq h_sum,
    sorry }
end

end arithmetic_seq_and_sum_l821_821836


namespace seq_arithmetic_and_formula_expr_b_2np1_sum_reciprocals_less_than_one_l821_821880

/-- (1) Prove that the sequence {1 / (a_n - 1)} is arithmetic, and find the general formula for the sequence {a_n}. -/
theorem seq_arithmetic_and_formula :
  ∀ (a : ℕ → ℚ), a 1 = 2 ∧ (∀ n, a (n + 1) = 2 - 1 / a n) →
  (∀ n, 1 / (a (n + 1) - 1) - 1 / (a n - 1) = 1) ∧ (∀ n, a n = (n + 1) / n) :=
by
  sorry

/-- (2) Prove the expression for b_{2n+1}. -/
theorem expr_b_2np1 :
  ∀ (b : ℕ → ℚ), b 1 = 1 ∧ (∀ n, ∃ a, a = (n + 1) / n ∧ 
  b (2 * n) / b (2 * n - 1) = a ∧ b (2 * n + 1) / b (2 * n) = a) →
  ∀ n, b (2 * n + 1) = (n + 1) ^ 2 :=
by
  sorry

/-- (3) Prove that 1 / b_2 + 1 / b_4 + ... + 1 / b_{2n} < 1. -/
theorem sum_reciprocals_less_than_one :
  ∀ (b : ℕ → ℚ), b 1 = 1 ∧ (∀ n, b (2 * n) = n * (n + 1)) →
  ∀ n, ∑ k in finset.range n, 1 / b (2 * (k + 1)) < 1 :=
by
  sorry

end seq_arithmetic_and_formula_expr_b_2np1_sum_reciprocals_less_than_one_l821_821880


namespace bus_ride_difference_l821_821251

theorem bus_ride_difference (vince_bus_length zachary_bus_length : Real)
    (h_vince : vince_bus_length = 0.62)
    (h_zachary : zachary_bus_length = 0.5) :
    vince_bus_length - zachary_bus_length = 0.12 :=
by
  sorry

end bus_ride_difference_l821_821251


namespace largest_term_125_l821_821332

-- Define the term A_k in the binomial expansion (1 + 0.3)^500
def term_A (k : ℕ) (h : k ≤ 500) : ℚ :=
  nat.choose 500 k * (0.3 : ℚ) ^ k

-- Predicate to express the condition of being the largest term
def is_largest_term (k : ℕ) (h : k ≤ 500) : Prop :=
  ∀ k' : ℕ, k' ≤ 500 → term_A k h ≥ term_A k' (nat.le_trans h (by simp [k']))

-- The main statement to prove
theorem largest_term_125 : is_largest_term 125 (by norm_num) :=
sorry

end largest_term_125_l821_821332


namespace parallel_vectors_l821_821478

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821478


namespace shelves_needed_l821_821545

theorem shelves_needed (total_books sorted_books books_per_shelf remaining_books number_of_shelves : ℕ)
  (h₁ : total_books = 1500)
  (h₂ : sorted_books = 375)
  (h₃ : books_per_shelf = 45)
  (h₄ : remaining_books = total_books - sorted_books)
  (h₅ : number_of_shelves = remaining_books / books_per_shelf) :
  number_of_shelves = 25 :=
by {
  rw [h₁, h₂, h₃, h₄],
  have : remaining_books = 1125, by linarith,
  rw this at h₅,
  norm_num at h₅,
  exact h₅,
}

end shelves_needed_l821_821545


namespace determine_compound_impossible_l821_821713

-- Define the conditions
def contains_Cl (compound : Type) : Prop := true -- Placeholder definition
def mass_percentage_Cl (compound : Type) : ℝ := 0 -- Placeholder definition

-- Define the main statement
theorem determine_compound_impossible (compound : Type) 
  (containsCl : contains_Cl compound) 
  (massPercentageCl : mass_percentage_Cl compound = 47.3) : 
  ∃ (distinct_element : Type), compound = distinct_element := 
sorry

end determine_compound_impossible_l821_821713


namespace parallel_vectors_lambda_l821_821472

noncomputable theory

open_locale classical

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_lambda (λ : ℝ) :
  vectors_parallel (2, 5) (λ, 4) ↔ λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_l821_821472


namespace length_of_closed_line_le_200_l821_821928

def fifteen_by_fifteen_grid : Prop :=
  ∃ closed_line : set (ℤ × ℤ), 
    (∀ cell ∈ closed_line, 
      ∃ x y, 0 ≤ x ∧ x < 15 ∧ 0 ≤ y ∧ y < 15 ∧ 
      cell = (x, y)) ∧
    (∀ (c1 c2 : ℤ × ℤ), c1 ∈ closed_line → c2 ∈ closed_line → c1 ≠ c2 → ((c1.1 = c2.1 → abs (c1.2 - c2.2) = 1) ∨ (c1.2 = c2.2 → abs (c1.1 - c2.1) = 1))) ∧
    (∃ diag : set (ℤ × ℤ), 
      ∀ cell ∈ diag, 
        (∃ x, 0 ≤ x ∧ x < 15 ∧ cell = (x, x))) ∧
    (∀ cell ∈ closed_line, 
      ∃ symm_cell ∈ closed_line, 
        symm_cell.1 + cell.1 = 14 ∧ 
        symm_cell.2 + cell.2 = 14)

theorem length_of_closed_line_le_200 : fifteen_by_fifteen_grid → ∃ line_length ≤ 200, true :=
by {
  sorry
}

end length_of_closed_line_le_200_l821_821928


namespace sum_of_reciprocals_is_integer_l821_821318

theorem sum_of_reciprocals_is_integer : 
  ∃ (seq : list ℤ), 
    (-33 ≤ seq.head ∧ seq.head ≤ 100) ∧
    (list.pairwise (λ a b, (a + b ≠ 0)) seq) ∧
    ((seq.length = (134 + 1)) ∧
     (∃ (a b : ℕ), 
      a + b = 133 ∧ 
      (a : ℚ) * (1 / 67) + (b : ℚ) * (1 / 66) ∈ ℤ)) := 
sorry

end sum_of_reciprocals_is_integer_l821_821318


namespace total_discount_is_58_percent_l821_821310

-- Definitions and conditions
def sale_discount : ℝ := 0.4
def coupon_discount : ℝ := 0.3

-- Given an original price, the sale discount price and coupon discount price
def sale_price (original_price : ℝ) : ℝ := (1 - sale_discount) * original_price
def final_price (original_price : ℝ) : ℝ := (1 - coupon_discount) * (sale_price original_price)

-- Theorem statement: final discount is 58%
theorem total_discount_is_58_percent (original_price : ℝ) : (original_price - final_price original_price) / original_price = 0.58 :=
by intros; sorry

end total_discount_is_58_percent_l821_821310


namespace problem1_problem2_l821_821276

noncomputable def problem1_calculation : Real :=
  3 * (tan (Real.pi / 6)) - (Real.sqrt 3 - 1).abs + (3.14 - Real.pi)^0 - (1 / 3)^(-2)

theorem problem1 : problem1_calculation = -7 :=
  sorry

variable {a : Real}
variable (h : a^2 + 2*a - 3 = 0)

noncomputable def problem2_expr (a : Real) : Real :=
  (2 * a - 12 * a / (a + 2))/((a - 4)/(a^2 + 4 * a + 4))

theorem problem2 (a : Real) (h : a^2 + 2*a - 3 = 0) :
  problem2_expr a = 6 :=
  sorry

end problem1_problem2_l821_821276


namespace compare_magnitudes_l821_821016

theorem compare_magnitudes :
  let a := (1 / 3) ^ (-1.1)
  let b := Real.pi ^ 0
  let c := 3 ^ 0.9
  b < c ∧ c < a :=
by
  -- The proof can be filled in here
  sorry

end compare_magnitudes_l821_821016


namespace minimize_sum_of_squares_l821_821589

/-- Representation of conditions -/
variable (x : Fin 6 → ℝ)
noncomputable def vertices : Fin 8 → ℝ
| ⟨0, _⟩ := 0
| ⟨7, _⟩ := 2013
| ⟨n, _⟩ := x ⟨n - 1, by linarith⟩

/-- Representation of sum of squares of differences -/
noncomputable def sum_of_squares (x : Fin 6 → ℝ) : ℝ :=
  let v : Fin 8 → ℝ := vertices x
  let edges : List (Fin 8 × Fin 8) := [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
                                       (1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
  (edges.map (λ ⟨i, j⟩ => (v i - v j)^2)).sum

-- Define the goal statement
theorem minimize_sum_of_squares : 
  ∃ x : Fin 6 → ℝ, (∀ i : Fin 6, x i = 1006.5) ∧ sum_of_squares x = (3 * 2013^2) / 2 := by
  sorry

end minimize_sum_of_squares_l821_821589


namespace find_linear_function_maximize_profit_l821_821715

section ShoppingMall

variables x y k b W : ℤ

-- Conditions
def condition_1 : Prop := y = 30000 ∧ x = 5
def condition_2 : Prop := y = 20000 ∧ x = 6

-- Problem 1: Linear function relationship
theorem find_linear_function (h1 : condition_1) (h2 : condition_2) :
  ∃ k b, (30000 = 5 * k + b) ∧ (20000 = 6 * k + b) ∧ (∀ x, y = k * x + b) :=
sorry

-- Problem 2: Maximum profit
def purchase_price := 4
def profit_function (x : ℤ) : ℤ := (x - purchase_price) * (-10000 * x + 80000)

theorem maximize_profit :
  ∃ price max_profit, price = 6 ∧ max_profit = 40000 ∧ 
  (∀ x, profit_function x ≤ max_profit) :=
sorry

end ShoppingMall

end find_linear_function_maximize_profit_l821_821715


namespace factorization_of_x_squared_minus_64_l821_821799

theorem factorization_of_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factorization_of_x_squared_minus_64_l821_821799


namespace linear_eq_condition_l821_821028

variables {m x : ℝ} -- Define variables m and x as real numbers

-- State the theorem
theorem linear_eq_condition (h : (3 * m - 1) * x + 9 = 0) : m ≠ (1 / 3) :=
begin
  -- Proof will be added here
  sorry
end

end linear_eq_condition_l821_821028


namespace primary_college_employee_relation_l821_821523

theorem primary_college_employee_relation
  (P C N : ℕ)
  (hN : N = 20 + P + C)
  (h_illiterate_wages_before : 20 * 25 = 500)
  (h_illiterate_wages_after : 20 * 10 = 200)
  (h_primary_wages_before : P * 40 = P * 40)
  (h_primary_wages_after : P * 25 = P * 25)
  (h_college_wages_before : C * 50 = C * 50)
  (h_college_wages_after : C * 60 = C * 60)
  (h_avg_decrease : (500 + 40 * P + 50 * C) / N - (200 + 25 * P + 60 * C) / N = 10) :
  15 * P - 10 * C = 10 * N - 300 := 
by
  sorry

end primary_college_employee_relation_l821_821523


namespace find_lambda_l821_821438

-- Define vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- State the condition that a is parallel to b
def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Statement of the proof problem
theorem find_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 8 / 5 :=
by
  sorry

end find_lambda_l821_821438


namespace white_washing_cost_l821_821630

def length := 25 -- feet
def breadth := 15 -- feet
def height := 12 -- feet

def door_height := 6 -- feet
def door_width := 3 -- feet

def window_height := 4 -- feet
def window_width := 3 -- feet
def num_windows := 3

def cost_per_square_foot := 4 -- Rs.

def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)
def area_of_walls (perimeter height : ℕ) : ℕ := perimeter * height
def area_of_door (height width : ℕ) : ℕ := height * width
def area_of_window (height width : ℕ) : ℕ := height * width

def total_cost (area cost_per_square_foot : ℕ) : ℕ := area * cost_per_square_foot

theorem white_washing_cost :
  let p := perimeter length breadth,
      total_wall_area := area_of_walls p height,
      door_area := area_of_door door_height door_width,
      window_area := num_windows * area_of_window window_height window_width,
      total_area_to_be_subtracted := door_area + window_area,
      area_to_be_white_washed := total_wall_area - total_area_to_be_subtracted
  in total_cost area_to_be_white_washed cost_per_square_foot = 3624 :=
by
  let p := perimeter length breadth
  let total_wall_area := area_of_walls p height
  let door_area := area_of_door door_height door_width
  let window_area := num_windows * area_of_window window_height window_width
  let total_area_to_be_subtracted := door_area + window_area
  let area_to_be_white_washed := total_wall_area - total_area_to_be_subtracted
  let result := total_cost area_to_be_white_washed cost_per_square_foot
  exact Nat.eq.refl result 3624

end white_washing_cost_l821_821630


namespace gcd_condition_l821_821225

def seq (a : ℕ → ℕ) := a 0 = 3 ∧ ∀ n, a (n + 1) - a n = n * (a n - 1)

theorem gcd_condition (a : ℕ → ℕ) (m : ℕ) (h : seq a) :
  m ≥ 2 → (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 := 
sorry

end gcd_condition_l821_821225


namespace tan_2x_eq_sin_x_has_4_solutions_l821_821904

def tan_sin_eq_solutions_interval_count : ℕ := 4

theorem tan_2x_eq_sin_x_has_4_solutions :
  (∀ x, 0 ≤ x ∧ x ≤ 2 * real.pi → tan (2 * x) = real.sin x) → tan_sin_eq_solutions_interval_count = 4 :=
by
  -- Proof by analyzing properties of tan(2x) and sin(x) and their intersections over [0, 2pi]
  sorry

end tan_2x_eq_sin_x_has_4_solutions_l821_821904


namespace parallel_vectors_lambda_value_l821_821463

theorem parallel_vectors_lambda_value (λ : ℝ) :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (λ, 4)
  (∃ k : ℝ, a = (k • b)) → λ = 8 / 5 :=
by
  sorry

end parallel_vectors_lambda_value_l821_821463


namespace parallel_vectors_lambda_l821_821487

theorem parallel_vectors_lambda (λ : ℝ) :
  let a := (2, 5) in
  let b := (λ, 4) in
  a.1 / b.1 = a.2 / b.2 → λ = 8 / 5 :=
by
  intros a b h_proportional
  rw [← h_proportional]
  sorry

end parallel_vectors_lambda_l821_821487


namespace katie_total_marbles_l821_821989

theorem katie_total_marbles :
  ∀ (pink marbles orange marbles purple marbles total : ℕ),
    pink = 13 →
    orange = pink - 9 →
    purple = 4 * orange →
    total = pink + orange + purple →
    total = 33 :=
by
  intros pink marbles orange marbles purple marbles total
  assume h_pink h_orange h_purple h_total
  sorry

end katie_total_marbles_l821_821989


namespace coloring_ways_l821_821547

-- Define the types and conditions
def color : Type := ℕ
def colors : finset color := {1, 2, 3}
def is_valid_coloring (grid : ℕ × ℕ → color) : Prop := 
  ∀ (x y : ℕ), (x < 3) ∧ (y < 3) → 
    (if x < 2 then grid (x, y) ≠ grid (x + 1, y) else true) ∧
    (if y < 2 then grid (x, y) ≠ grid (x, y + 1) else true)

-- The main statement to prove that the number of valid colorings is 768
theorem coloring_ways : 
  ∃ (num_ways : ℕ), num_ways = 768 ∧ 
  (∃ (grid : ℕ × ℕ → color) (hf : is_valid_coloring grid), grid ∈ finset.powerset_len 9 colors → num_ways = finset.card grid) :=
sorry

end coloring_ways_l821_821547


namespace required_fencing_l821_821729

-- Define constants given in the problem
def L : ℕ := 20
def A : ℕ := 720

-- Define the width W based on the area and the given length L
def W : ℕ := A / L

-- Define the total amount of fencing required
def F : ℕ := 2 * W + L

-- State the theorem that this amount of fencing is equal to 92
theorem required_fencing : F = 92 := by
  sorry

end required_fencing_l821_821729


namespace product_modulo_seven_l821_821648

theorem product_modulo_seven (a b c d : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3)
(h3 : c % 7 = 4) (h4 : d % 7 = 5) : (a * b * c * d) % 7 = 1 := 
sorry

end product_modulo_seven_l821_821648


namespace parallel_vectors_implies_value_of_λ_l821_821430

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821430


namespace find_a_l821_821403

noncomputable def f (x : ℝ) : ℝ := x^2 + 10

noncomputable def g (x : ℝ) : ℝ := x^2 - 6

theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : f (g a) = 12) :
    a = Real.sqrt (6 + Real.sqrt 2) ∨ a = Real.sqrt (6 - Real.sqrt 2) :=
sorry

end find_a_l821_821403


namespace non_adjacent_arrangements_l821_821964

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821964


namespace non_adjacent_arrangements_l821_821963

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem non_adjacent_arrangements : 
  let total_arrangements := factorial 8
  let adjacent_arrangements := factorial 7 * factorial 2
  total_arrangements - adjacent_arrangements = 30240 := by
sorry

end non_adjacent_arrangements_l821_821963


namespace number_of_real_solutions_l821_821190

theorem number_of_real_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (3^(2*x1 + 1) - 3^(x1 + 2) - 2 * 3^x1 + 6 = 0) ∧ 
                (3^(2*x2 + 1) - 3^(x2 + 2) - 2 * 3^x2 + 6 = 0) ∧ 
                ∀ (x : ℝ), 3^(2*x + 1) - 3^(x + 2) - 2 * 3^x + 6 = 0 → 
                (x = x1 ∨ x = x2) := 
by 
  sorry

end number_of_real_solutions_l821_821190


namespace factor_x_squared_minus_64_l821_821794

-- Conditions
def a := x
def b := 8

-- Theorem statement
theorem factor_x_squared_minus_64 (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) :=
by sorry

end factor_x_squared_minus_64_l821_821794


namespace lambda_parallel_l821_821455

open_locale real

-- Define the concept of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Given vectors a and b
def a : ℝ × ℝ := (2, 5)
def b (λ : ℝ) : ℝ × ℝ := (λ, 4)

-- The proof statement
theorem lambda_parallel (λ : ℝ) (h : parallel a (b λ)) : λ = 8/5 :=
  sorry

end lambda_parallel_l821_821455


namespace no_solutions_988_1991_l821_821363

theorem no_solutions_988_1991 :
    ¬ ∃ (m n : ℤ),
      (988 ≤ m ∧ m ≤ 1991) ∧
      (988 ≤ n ∧ n ≤ 1991) ∧
      m ≠ n ∧
      ∃ (a b : ℤ), (mn + n = a^2 ∧ mn + m = b^2) := sorry

end no_solutions_988_1991_l821_821363


namespace range_of_alpha_l821_821872

noncomputable def f (x α : ℝ) : ℝ :=
  real.log x + 2 * real.sin α

variable (α : ℝ) (x₀ : ℝ)

theorem range_of_alpha :
  (0 < α ∧ α < π / 2) →
  (0 < x₀ ∧ x₀ < 1) →
  (has_deriv_at (λ x, real.log x + 2 * real.sin α) (1 / x₀) x₀) →
  (f x₀ α = 1 / x₀) →
  (π / 6 < α ∧ α < π / 2) :=
by
  intros hα hx₀ hderiv hfx₀_eq
  sorry

end range_of_alpha_l821_821872


namespace polynomial_roots_identity_l821_821137

variables {c d : ℂ}

theorem polynomial_roots_identity (hc : c + d = 5) (hd : c * d = 6) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 :=
by {
  sorry
}

end polynomial_roots_identity_l821_821137


namespace no_such_function_exists_l821_821775

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f(f(x)) = x + 1 :=
by
  sorry

end no_such_function_exists_l821_821775


namespace vegan_non_soy_fraction_l821_821069

theorem vegan_non_soy_fraction (total_menu : ℕ) (vegan_dishes soy_free_vegan_dish : ℕ) 
  (h1 : vegan_dishes = 6) (h2 : vegan_dishes = total_menu / 3) (h3 : soy_free_vegan_dish = vegan_dishes - 5) :
  (soy_free_vegan_dish / total_menu = 1 / 18) :=
by
  sorry

end vegan_non_soy_fraction_l821_821069


namespace smaller_circle_radius_l821_821531

theorem smaller_circle_radius (r R : ℝ) (hR : R = 10) (h : 2 * r = 2 * R) : r = 10 :=
by
  sorry

end smaller_circle_radius_l821_821531


namespace modulo_calculation_l821_821760

theorem modulo_calculation : (68 * 97 * 113) % 25 = 23 := by
  sorry

end modulo_calculation_l821_821760


namespace range_of_x_for_valid_sqrt_l821_821920

theorem range_of_x_for_valid_sqrt (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
by
  sorry

end range_of_x_for_valid_sqrt_l821_821920


namespace triangle_circle_distance_l821_821562

open Real

theorem triangle_circle_distance 
  (DE DF EF : ℝ)
  (hDE : DE = 12) (hDF : DF = 16) (hEF : EF = 20) :
  let s := (DE + DF + EF) / 2
  let K := sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  let ra := K / (s - EF)
  let DP := s - DF
  let DQ := s
  let DI := sqrt (DP^2 + r^2)
  let DE := sqrt (DQ^2 + ra^2)
  let distance := DE - DI
  distance = 24 * sqrt 2 - 4 * sqrt 10 :=
by
  sorry

end triangle_circle_distance_l821_821562


namespace chocolate_eggs_last_weeks_l821_821578

variable (N : ℕ) (chocolates_per_day : ℕ) (days_per_week : ℕ)

theorem chocolate_eggs_last_weeks :
  N = 40 →
  chocolates_per_day = 2 →
  days_per_week = 5 →
  (N / (chocolates_per_day * days_per_week)) = 4 :=
by
  intros hN hcd hdpw
  rw [hN, hcd, hdpw]
  norm_num
  sorry

end chocolate_eggs_last_weeks_l821_821578


namespace Faraway_not_possible_sum_l821_821534

theorem Faraway_not_possible_sum (h g : ℕ) : (74 ≠ 21 * h + 6 * g) ∧ (89 ≠ 21 * h + 6 * g) :=
by
  sorry

end Faraway_not_possible_sum_l821_821534


namespace faster_current_takes_more_time_l821_821249

theorem faster_current_takes_more_time (v v1 v2 S : ℝ) (h_v1_gt_v2 : v1 > v2) :
  let t1 := (2 * S * v) / (v^2 - v1^2)
  let t2 := (2 * S * v) / (v^2 - v2^2)
  t1 > t2 :=
by
  sorry

end faster_current_takes_more_time_l821_821249


namespace badge_height_enlarged_proportionately_l821_821317

theorem badge_height_enlarged_proportionately :
  ∀ (original_width original_height new_width : ℕ), 
  original_width = 3 → 
  original_height = 2 → 
  new_width = 9 → 
  new_height = (original_height * (new_width / original_width)) →
  new_height = 6 :=
by
  intros original_width original_height new_width h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h2]
  sorry

end badge_height_enlarged_proportionately_l821_821317


namespace PetrovFamilySavings_l821_821617

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end PetrovFamilySavings_l821_821617


namespace total_sugar_start_with_l821_821219

-- Definitions for the conditions
def num_packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- The theorem to prove
theorem total_sugar_start_with : num_packs * weight_per_pack + leftover_sugar = 3020 := 
by 
  simp
  sorry

end total_sugar_start_with_l821_821219


namespace P1_coordinates_l821_821172

-- Define initial point coordinates
def P : (ℝ × ℝ) := (0, 3)

-- Define the transformation functions
def move_left (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1 - units, p.2)
def move_up (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1, p.2 + units)

-- Calculate the coordinates of point P1
def P1 : (ℝ × ℝ) := move_up (move_left P 2) 1

-- Statement to prove
theorem P1_coordinates : P1 = (-2, 4) := by
  sorry

end P1_coordinates_l821_821172


namespace _l821_821762

noncomputable def expression_1 : ℚ :=
  (25/9)^(1/2) - 1 - (64/27)^(-2/3) + (1/4)^(-3/2)

noncomputable theorem proof_expression_1 : 
  (2 + 7/9)^(1/2) - (2 * real.sqrt 3 - real.pi)^0 - (2 + 10/27)^(-2/3) + 0.25^(-3/2) = 389/48 :=
by sorry

noncomputable def expression_2 : ℝ :=
  real.logBase 2.5 6.25 + real.log10 5 + real.log (real.sqrt real.e) + 2^(-1 + real.logBase 2 3) + (real.log10 2)^2 + real.log10 5 * real.log10 2

noncomputable theorem proof_expression_2 : 
  real.logBase 2.5 6.25 + real.log10 5 + real.log (real.sqrt real.e) + 2^(-1 + real.logBase 2 3) + (real.log10 2)^2 + real.log10 5 * real.log10 2 = 5 :=
by sorry

end _l821_821762


namespace find_x_l821_821704

theorem find_x (x : ℝ) (h : 3550 - (x / 20.04) = 3500) : x = 1002 :=
by
  sorry

end find_x_l821_821704


namespace expectation_variance_eta_l821_821835

noncomputable def expectation_of_eta : ℝ :=
  2

noncomputable def variance_of_eta : ℝ :=
  2.4

theorem expectation_variance_eta (η ξ : ℝ) (h1 : ξ ~ binomial 10 0.6) :
  (E η = expectation_of_eta) ∧ (Var η = variance_of_eta) :=
sorry

end expectation_variance_eta_l821_821835


namespace inequality_proof_l821_821850

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
sorry

end inequality_proof_l821_821850


namespace slope_range_l821_821380

noncomputable def hyperbola (x y : ℝ) := (x^2) / 4 - (y^2) / 8 - 1

def line (t x : ℝ) : ℝ := t * x + 1

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Slope of line passing through two points M(1, 0) and P(p₁, p₂)
def slope (M P : ℝ × ℝ) : ℝ := (P.2 - M.2) / (P.1 - M.1)

theorem slope_range (t : ℝ) (A B: (ℝ × ℝ))  
    (h1 : ∃ x y, hyperbola x y = 0 )
    (h2 : ∃ x y, line t x = y )
    (h3 : A ≠ B)
    (h4 : P = midpoint A B)
    (h5 : M = (1, 0)) :
    ∃ k,  k = slope M P 
            ∧ (k ∈ ( (-∞ : ℝ), -8/9] ∪  (8/7, sqrt 2) ∪ (sqrt 2, ∞))
:= sorry

end slope_range_l821_821380


namespace minimum_subsidy_lowest_average_cost_l821_821656

noncomputable def cost_function (x : ℝ) : ℝ :=
  if x >= 120 ∧ x < 144 then (1 / 3) * x^3 - 80 * x^2 + 5040 * x
  else if x >= 144 ∧ x < 500 then (1 / 2) * x^2 - 200 * x + 80000
  else 0

def revenue_function (x : ℝ) : ℝ := 200 * x

def profit_function (x : ℝ) : ℝ :=
  revenue_function x - cost_function x

-- Prove minimum subsidy for profitability is 5000 yuan for x in [200, 300)
theorem minimum_subsidy (x : ℝ) (hx : 200 ≤ x ∧ x < 300) :
  profit_function x < 0 ∧ ∃ s ≥ 5000, profit_function x + s = 0 := sorry

-- Prove lowest average processing cost per ton when x is 400 tons
theorem lowest_average_cost (x : ℝ) (hx : 120 ≤ x ∧ x < 500) :
  (∀ y, 120 ≤ y ∧ y < 500 → (cost_function y / y) ≥ (cost_function 400 / 400)) := sorry

end minimum_subsidy_lowest_average_cost_l821_821656


namespace find_a_l821_821916

-- Define the quadratic equation with the root condition
def quadratic_with_root_zero (a : ℝ) : Prop :=
  (a - 1) * 0^2 + 0 + a - 2 = 0

-- State the theorem to be proved
theorem find_a (a : ℝ) (h : quadratic_with_root_zero a) : a = 2 :=
by
  -- Statement placeholder, proof omitted
  sorry

end find_a_l821_821916


namespace find_PC_l821_821117

-- Define the conditions
variables {A B C P : Point}
variable {PC : ℝ}
variables (PA PB : ℝ)
variables (θ : ℝ)

-- Given conditions as Lean hypotheses
-- PA = 13
def PA_val : ℝ := 13
-- PB = 5
def PB_val : ℝ := 5
-- ∠B = 90 degrees
def right_angle_B : θ = real.pi / 2
-- ∠APB = ∠BPC = ∠CPA = 120 degrees
def equal_angles : θ = 2 * real.pi / 3

-- Theorem statement
theorem find_PC (h1 : θ = real.pi / 2)
    (h2 : PA = PA_val)
    (h3 : PB = PB_val)
    (h4 : ∠ APB = θ)
    (h5 : ∠ BPC = θ)
    (h6 : ∠ CPA = θ) :
    PC = 14.375 :=
begin 
  -- Proof goes here.
  sorry
end

end find_PC_l821_821117


namespace inequality_proof_l821_821844

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821844


namespace triangle_angle_A_eq_pi_div_4_l821_821976

theorem triangle_angle_A_eq_pi_div_4
  {A B C H : Type*}
  [Triangle ABC]
  (h : is_orthocenter H A B C)
  (ha : distance A H = a)
  (hb : distance B C = a) :
  angle A = π / 4 :=
sorry

end triangle_angle_A_eq_pi_div_4_l821_821976


namespace intersection_A_B_l821_821511

noncomputable def A := {x : ℝ | 2^x - 4 > 0}
def B := {x : ℝ | 0 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by sorry

end intersection_A_B_l821_821511


namespace cost_price_of_table_l821_821696

theorem cost_price_of_table (SP : ℝ) (CP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3600) : CP = 3000 :=
by
  sorry

end cost_price_of_table_l821_821696


namespace proof_problem_l821_821538

noncomputable def question : Prop :=
  ∃ (x y a : ℝ), (x - 1 = a * (y^3 - 1)) ∧ (2 * x / (|y^3| + y^3) = Real.sqrt x) ∧ 
  (a ∈ Set.Iic(-1) ∪ Set.univ.singleton 2 ∪ Set.Ici 0 ∩ Set.Iic 1 ∪ Set.Ioi 1 ∩ Set.Iic 2 ∪ Set.Ioi 2)

theorem proof_problem : question :=
sorry

end proof_problem_l821_821538


namespace mary_candies_l821_821977

-- The conditions
def bob_candies : Nat := 10
def sue_candies : Nat := 20
def john_candies : Nat := 5
def sam_candies : Nat := 10
def total_candies : Nat := 50

-- The theorem to prove
theorem mary_candies :
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies) = 5 := by
  -- Here is where the proof would go; currently using sorry to skip the proof
  sorry

end mary_candies_l821_821977


namespace license_plate_palindrome_probability_l821_821934

theorem license_plate_palindrome_probability :
  let total_four_digit = 9 * 10 * 10 * 10,
      palindromic_four_digit = 9 * 10,
      prob_four_digit_palindrome := (palindromic_four_digit : ℚ) / total_four_digit,
      total_four_letter = 25 * 26 * 26 * 26,
      palindromic_four_letter = 25 * 26,
      prob_four_letter_palindrome := (palindromic_four_letter : ℚ) / total_four_letter,
      palindromes_prob := prob_four_digit_palindrome + prob_four_letter_palindrome -
                          prob_four_digit_palindrome * prob_four_letter_palindrome
  in palindromes_prob = (2 : ℚ) / 65 := by
  sorry

end license_plate_palindrome_probability_l821_821934


namespace min_lambda_of_sequences_l821_821866

theorem min_lambda_of_sequences {a_n b_n S_n T_n : ℕ → ℝ} 
  (h1 : a_n 1 = 2)
  (h2 : ∀ n, 3 * S_n n = (n + 2) * a_n n)
  (h3 : ∀ n, a_n n * b_n n = 1 / 2)
  (h4 : ∀ λ : ℝ, (∀ n : ℕ, λ > T_n n) ↔ λ > 1 / 2) :
  ∃ λ : ℝ, λ = 1 / 2 :=
begin
  sorry
end

end min_lambda_of_sequences_l821_821866


namespace intersection_of_P_with_complement_Q_l821_821884

-- Define the universal set U, and sets P and Q
def U : List ℕ := [1, 2, 3, 4]
def P : List ℕ := [1, 2]
def Q : List ℕ := [2, 3]

-- Define the complement of Q with respect to U
def complement (U Q : List ℕ) : List ℕ := U.filter (λ x => x ∉ Q)

-- Define the intersection of two sets
def intersection (A B : List ℕ) : List ℕ := A.filter (λ x => x ∈ B)

-- The proof statement we need to show
theorem intersection_of_P_with_complement_Q : intersection P (complement U Q) = [1] := by
  sorry

end intersection_of_P_with_complement_Q_l821_821884


namespace milk_after_replacements_l821_821690

theorem milk_after_replacements :
  ∀ (initial_milk initial_volume mix_change : ℕ), 
    initial_milk = 40 ∧ initial_volume = 40 ∧ mix_change = 4 → 
    let iter1 := initial_milk - mix_change
    let iter2 := iter1 - (mix_change * iter1 / initial_volume)
    let iter3 := iter2 - (mix_change * iter2 / initial_volume)
    iter3 = 29.16 :=
by
  intros initial_milk initial_volume mix_change h
  let iter1 := initial_milk - mix_change
  let iter2 := iter1 - (mix_change * iter1 / initial_volume)
  let iter3 := iter2 - (mix_change * iter2 / initial_volume)
  sorry

end milk_after_replacements_l821_821690


namespace acceptable_arrangements_correct_l821_821941

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l821_821941


namespace question1_question2_l821_821237

-- Definitions based on conditions
def n : ℕ := 6
def students : Fin n → String := 
  fun i => if i = 0 then "A" else if i = 1 then "B" else if i = 2 then "C" else 
           if i = 3 then "D" else if i = 4 then "E" else "F"

-- Question 1: Number of arrangements if A is not at the head or tail
def numArrangementsWithoutAHeadTail : ℕ :=
  4 * 5!  -- A can be in any of the 4 positions, rest can be permuted in 5!

-- Question 2: Number of arrangements if A, B, and C are not adjacent
def numArrangementsABCNotAdjacent : ℕ :=
  (3! * 4P3).natAbs -- Arrange 3 people excluding A, B, C, and then insert A, B, C into 4 spaces

-- Theorem statements
theorem question1 : numArrangementsWithoutAHeadTail = 480 := 
  by
  exact calc
    numArrangementsWithoutAHeadTail = 4 * 5! : by rfl
    ... = 480 : by norm_num
   sorry

theorem question2 : numArrangementsABCNotAdjacent = 144 := 
  by 
  exact calc
    numArrangementsABCNotAdjacent = (3! * 4P3).natAbs : by rfl
    ... = 144 : by norm_num
  sorry

end question1_question2_l821_821237


namespace max_lambda_inequality_l821_821385

theorem max_lambda_inequality (n : ℕ) (a : ℕ → ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ i j, i < j → i < n → j < n → a i < a j) :
  ∃ λ, λ = (2 * (n - 2)) / (n - 1) ∧ ∀ (a₁ a₂ ... aₙ : ℕ), 
    (a 1 < a 2 < ... < a n) → 
    (a n)^2 ≥ λ * (a 1 + a 2 + ... + a (n - 1)) + 2 * a n := 
begin
  sorry
end

end max_lambda_inequality_l821_821385


namespace angle_bisector_property_l821_821698

-- Definitions and hypothesis in Lean 4
variables {S Q T O M : Type} [Nonempty S] [Nonempty Q] [Nonempty T] [Nonempty O] [Nonempty M]
variables (angle SQT : ℝ) (angle QTS : ℝ) (angle QST : ℝ) (angle OQT : ℝ)

-- Given conditions
def is_angle_bisector (S M T : Type) : Prop := sorry
def meets_condition (O S T Q : Type) (angle OQT : ℝ) (angle QTS : ℝ) (angle QST : ℝ) : Prop :=
  angle OQT = angle QTS + angle QST

-- The proof problem to be shown
theorem angle_bisector_property
  (h1 : is_angle_bisector S M T)
  (h2 : meets_condition O S T Q angle OQT angle QTS angle QST) :
  is_angle_bisector O M (QOT : angle) :=
sorry

end angle_bisector_property_l821_821698


namespace lucy_bought_cakes_l821_821577

theorem lucy_bought_cakes (cookies chocolate total c : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) (h4 : c = total - (cookies + chocolate)) : c = 22 := by
  sorry

end lucy_bought_cakes_l821_821577


namespace parallel_vectors_l821_821481

variables (lambda k : ℝ)
def a := (2, 5 : ℝ)
def b := (lambda, 4 : ℝ)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem parallel_vectors (lambda : ℝ) (h : is_parallel (2, 5) (lambda, 4)) : 
  lambda = 8 / 5 :=
sorry

end parallel_vectors_l821_821481


namespace obtuse_triangle_l821_821115

variable (A B C : ℝ)
variable (angle_sum : A + B + C = 180)
variable (cond1 : A + B = 141)
variable (cond2 : B + C = 165)

theorem obtuse_triangle : B > 90 :=
by
  sorry

end obtuse_triangle_l821_821115


namespace measure_of_angle_F_l821_821536

-- Definitions for the angles in triangle DEF
variables (D E F : ℝ)

-- Given conditions
def is_right_triangle (D : ℝ) : Prop := D = 90
def angle_relation (E F : ℝ) : Prop := E = 4 * F - 10
def angle_sum (D E F : ℝ) : Prop := D + E + F = 180

-- The proof problem statement
theorem measure_of_angle_F (h1 : is_right_triangle D) (h2 : angle_relation E F) (h3 : angle_sum D E F) : F = 20 :=
sorry

end measure_of_angle_F_l821_821536


namespace parallel_vectors_lambda_l821_821444

theorem parallel_vectors_lambda (λ : ℚ) (a b : ℚ × ℚ)
  (ha : a = (2, 5))
  (hb : b = (λ, 4))
  (h_parallel : ∃ k : ℚ, a = k • b) :
  λ = 8/5 :=
by
  sorry

end parallel_vectors_lambda_l821_821444


namespace math_problem_equiv_proof_l821_821348

theorem math_problem_equiv_proof :
  (∃ m n p : ℤ, 2 * x^2 - 8 * x + 19 = m * (x - n)^2 + p ∧ 
  m = 2 ∧ n = 2 ∧ p = 11) →
  2017 + 2 * 11 - 5 * 2 = 2029 :=
by
  intro h
  rcases h with ⟨m, n, p, h1, hm, hn, hp⟩
  simp [hm, hn, hp]
  sorry

end math_problem_equiv_proof_l821_821348


namespace calculate_gain_percentage_l821_821288

theorem calculate_gain_percentage (CP SP : ℝ) (h1 : 0.9 * CP = 450) (h2 : SP = 550) : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end calculate_gain_percentage_l821_821288


namespace translation_correct_l821_821330

theorem translation_correct : 
  ∀ (x y : ℝ), (y = -(x-1)^2 + 3) → (x, y) = (0, 0) ↔ (x - 1, y - 3) = (0, 0) :=
by 
  sorry

end translation_correct_l821_821330


namespace S_n_bounds_l821_821881

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 9
  | n + 1 => a n + 2*n + 5

def b (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/4
  | n + 1 => (n + 1) / (n + 2) * b n

def seq (a b : ℕ → ℚ) (n : ℕ) : ℚ :=
  b n / real.sqrt (a n)

noncomputable def S (a b : ℕ → ℚ) (n : ℕ) : ℚ :=
  (finset.range n).sum (λ k, seq a b (k + 1))

theorem S_n_bounds (n : ℕ) (h₀ : 0 < n) (h₁ : ∀ k < n, a (k+1) = (k+1+2)^2) (h₂ : ∀ k < n, b (k+1) = 1 / (2 * (k+1+1))) :
  1/12 ≤ S a b n ∧ S a b n < 1/4 :=
by
  sorry

end S_n_bounds_l821_821881


namespace integer_solutions_count_l821_821897

theorem integer_solutions_count :
  ∃! (x : ℤ), (| (x : ℝ) | < 5 * Real.pi) ∧ (x^2 - 4 * x + 4 = 0) :=
sorry

end integer_solutions_count_l821_821897


namespace problem1_problem2_l821_821192

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l821_821192


namespace geometric_sequence_common_ratio_l821_821132

noncomputable def geometric_common_ratio (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b 1 * 2 ^ n → T n = ∏ i in range n, b (i + 1)

theorem geometric_sequence_common_ratio {T : ℕ → ℝ} {b : ℕ → ℝ} (h : geometric_common_ratio T b) :
  (T 6 / T 3), (T 9 / T 6), (T 12 / T 9) form a geometric sequence with ratio 512 :=
begin
  sorry
end

end geometric_sequence_common_ratio_l821_821132


namespace x₁_plus_x₂_gt_4_l821_821416

variable {a x₁ x₂ : ℝ}
hypothesis h1 : (0 < x₁ ∧ x₁ < x₂) ∧ f = λ x => a * x^2 - real.exp x
hypothesis h2 : f x₁ = 0
hypothesis h3 : f x₂ = 0

-- Prove that x₁ + x₂ > 4
theorem x₁_plus_x₂_gt_4 : x₁ + x₂ > 4 := by
  sorry

end x₁_plus_x₂_gt_4_l821_821416


namespace relation_between_a_b_c_l821_821535

variable {AB AC : ℝ} -- Lengths of sides AB and AC
variable {a b c : ℝ} -- Given angles
variable {ΔABC ΔDEF : Triangle} -- The triangles involved

-- Conditions formulation
-- AB and AC are the equal sides of the isosceles triangle ABC
def isosceles_ΔABC : ΔABC.is_isosceles AB AC := sorry

-- DEF is an isosceles triangle inscribed in ABC
def isosceles_ΔDEF_inscribed : ΔDEF.is_isosceles DEF.DE DEF.DF := sorry

-- Given angles in the problem
def angle_DEF_eq_100 : ΔDEF.angle_DEF = 100 := sorry
def angle_BFD_eq_a : ΔABC.angle_BFD = a := sorry
def angle_ADE_eq_b : ΔDEF.angle_ADE = b := sorry
def angle_FEC_eq_c : ΔDEF.angle_FEC = c := sorry

-- Objective: prove the relation between angles
theorem relation_between_a_b_c
    (isoscelesABC : ΔABC.is_isosceles AB AC)
    (isoscelesDEF : ΔDEF.is_isosceles DEF.DE DEF.DF)
    (angleDEF100 : ΔDEF.angle_DEF = 100)
    (angleBFDA : ΔABC.angle_BFD = a)
    (angleADEB : ΔDEF.angle_ADE = b)
    (angleFECC : ΔDEF.angle_FEC = c)
    : b = c := sorry

end relation_between_a_b_c_l821_821535


namespace sum_of_nonnegative_reals_l821_821924

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l821_821924


namespace minimum_value_of_expression_l821_821513

theorem minimum_value_of_expression {a c : ℝ} (h_pos : a > 0)
  (h_range : ∀ x, a * x ^ 2 - 4 * x + c ≥ 1) :
  ∃ a c, a > 0 ∧ (∀ x, a * x ^ 2 - 4 * x + c ≥ 1) ∧ (∃ a, a > 0 ∧ ∃ c, c - 1 = 4 / a ∧ (a / 4 + 9 / a = 3)) :=
by sorry

end minimum_value_of_expression_l821_821513


namespace geometric_sequence_proof_l821_821519

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

theorem geometric_sequence_proof :
  ∃ (a : ℕ → ℝ) (q : ℕ),
  is_geometric_sequence a (q : ℝ) ∧
  sum_of_first_n_terms a 4 = 32 ∧
  a 2 + a 3 = 12 ∧
  q = 2 ∧
  sum_of_first_n_terms a 6 = 126 ∧
  ∀ n, real.log (sum_of_first_n_terms a n + 2) = real.log (real.log (2 : ℝ) * (n + 1)) := sorry

end geometric_sequence_proof_l821_821519


namespace geometry_inequality_l821_821127

open_locale euclidean_geometry

variables {A B C D E : Type}

noncomputable def is_distinct (A B C D E : Type) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

noncomputable def no_three_collinear (A B C D E : Type) : Prop :=
  ∀ (P Q R : Type), P ≠ Q ∧ P ≠ R ∧ Q ≠ R → ¬ collinear ({A, B, C, D, E} : set Type) {P, Q, R}

theorem geometry_inequality
  (A B C D E : Type)
  (h1 : is_distinct A B C D E)
  (h2 : no_three_collinear A B C D E) :
  AB + BC + CA + DE < AD + AE + BD + BE + CD + CE :=
sorry

end geometry_inequality_l821_821127


namespace independence_necessary_and_sufficient_l821_821009

variables {Ω : Type*} {P : ProbabilityTheory ℕ}
variables (A B : Set Ω)
variable [MeasurableSpace Ω]
variable [ProbabilityMeasure P]

theorem independence_necessary_and_sufficient (hA : 0 < P(A)) (hB : 0 < P(B)) :
  (P(A ∩ B) = P(A) * P(B)) ↔ (A ∩ B = P(A) * P(B)) := 
sorry

end independence_necessary_and_sufficient_l821_821009


namespace fraction_covered_by_pepperoni_l821_821375

theorem fraction_covered_by_pepperoni 
  (d_pizza : ℝ) (n_pepperoni_diameter : ℕ) (n_pepperoni : ℕ) (diameter_pepperoni : ℝ) 
  (radius_pepperoni : ℝ) (radius_pizza : ℝ)
  (area_one_pepperoni : ℝ) (total_area_pepperoni : ℝ) (area_pizza : ℝ)
  (fraction_covered : ℝ)
  (h1 : d_pizza = 16)
  (h2 : n_pepperoni_diameter = 14)
  (h3 : n_pepperoni = 42)
  (h4 : diameter_pepperoni = d_pizza / n_pepperoni_diameter)
  (h5 : radius_pepperoni = diameter_pepperoni / 2)
  (h6 : radius_pizza = d_pizza / 2)
  (h7 : area_one_pepperoni = π * radius_pepperoni ^ 2)
  (h8 : total_area_pepperoni = n_pepperoni * area_one_pepperoni)
  (h9 : area_pizza = π * radius_pizza ^ 2)
  (h10 : fraction_covered = total_area_pepperoni / area_pizza) :
  fraction_covered = 3 / 7 :=
sorry

end fraction_covered_by_pepperoni_l821_821375


namespace minimum_n_l821_821296

variable (n d₁ : ℕ) -- n and d₁ are positive integers

-- Conditions given in the problem
def dealer_conditions (n d₁ : ℕ) : Prop :=
  n > 0 ∧ d₁ > 0 ∧
  let charity_revenue := (3 * (d₁ / (3 * n))) in
  let remaining_radios := n - 3 in
  let selling_price := (d₁ / n) + 10 in
  let remaining_revenue := remaining_radios * selling_price in
  let total_revenue := charity_revenue + remaining_revenue in
  let profit := total_revenue - d₁ in
  profit = 80

-- Proposition that the minimum value of n is 11
theorem minimum_n (d₁ : ℕ) : (dealer_conditions 11 d₁) → 
  ∀ n, dealer_conditions n d₁ → 11 ≤ n := 
by
  sorry

end minimum_n_l821_821296


namespace parallel_vectors_implies_value_of_λ_l821_821424

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821424


namespace no_valid_pairs_l821_821498

theorem no_valid_pairs : ∀ (a b : ℕ), (a > 0) → (b > 0) → (a ≥ b) → 
  a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b → 
  false := by
  sorry

end no_valid_pairs_l821_821498


namespace largest_remainder_a_correct_l821_821703

def largest_remainder_a (n : ℕ) (h : n < 150) : ℕ :=
  (269 % n)

theorem largest_remainder_a_correct : ∃ n < 150, largest_remainder_a n sorry = 133 :=
  sorry

end largest_remainder_a_correct_l821_821703


namespace unit_digit_of_sum_sequence_l821_821651

-- Definition of the sequence based on the given recurrence relation.
def sequence (n : ℕ) : ℕ :=
  if n = 0 then 6 else
  let a := (sequence (n - 1)) in
  2 * a - (n - 1) + 1

-- Proving that the units digit of the sum of the first 2022 terms is 8.
theorem unit_digit_of_sum_sequence :
  let s := ∑ n in Finset.range 2022, sequence (n + 1)
  let units_digit := s % 10
  units_digit = 8 :=
by
sorry

end unit_digit_of_sum_sequence_l821_821651


namespace polynomial_evaluation_at_one_l821_821005

theorem polynomial_evaluation_at_one :
  ∃ (p q s : ℝ), 
  (∃ h k : Polynomial ℝ, 
    h = Polynomial.Coeff 3 * Polynomial.x ^ 3 +
        Polynomial.Coeff (p : ℝ) * Polynomial.x ^ 2 +
        Polynomial.x +
        Polynomial.Coeff 15
    ∧
    k = Polynomial.Coeff 4 * Polynomial.x ^ 4 +
        Polynomial.Coeff 3 * Polynomial.x ^ 3 +
        Polynomial.Coeff q * Polynomial.x ^ 2 +
        Polynomial.Coeff 150 * Polynomial.x +
        Polynomial.Coeff s)
  ∧ 
  (∀ x : ℝ, h.eval x = 0 → k.eval x = 0)  -- for all x, if h(x) = 0, then k(x) = 0
  ∧ 
  (h.has_distinct_roots)
  ∧ 
  (k.eval 1 = -16048) :=
sorry

end polynomial_evaluation_at_one_l821_821005


namespace parallel_vectors_implies_value_of_λ_l821_821423

-- Define the vectors a and b
def a := (2, 5)
def b (λ : ℚ) := (λ, 4)

-- Define the condition for parallel vectors
def are_parallel (a b : ℚ × ℚ) : Prop :=
  ∃ (k : ℚ), ∀ i, (a i) = k * (b i)

-- Define the theorem to prove
theorem parallel_vectors_implies_value_of_λ :
  (are_parallel a (b (8 / 5))) → (∀ λ, b λ = b (8 / 5)) := by
  sorry

end parallel_vectors_implies_value_of_λ_l821_821423


namespace max_imaginary_part_theta_l821_821744

theorem max_imaginary_part_theta (z : ℂ) (θ : ℝ) (h1 : z^12 - z^9 + z^6 - z^3 + 1 = 0) 
    (h2 : ∃ θ, (z.im = Real.sin θ ∧ -90 ≤ θ ∧ θ ≤ 90)) :
  θ = 84 :=
begin
  sorry -- proof goes here
end

end max_imaginary_part_theta_l821_821744


namespace num_of_consecutive_sets_sum_18_eq_2_l821_821903

theorem num_of_consecutive_sets_sum_18_eq_2 : 
  ∃ (sets : Finset (Finset ℕ)), 
    (∀ s ∈ sets, (∃ n a, n ≥ 3 ∧ (s = Finset.range (a + n - 1) \ Finset.range (a - 1)) ∧ 
    s.sum id = 18)) ∧ 
    sets.card = 2 := 
sorry

end num_of_consecutive_sets_sum_18_eq_2_l821_821903


namespace fifth_largest_divisor_l821_821774

theorem fifth_largest_divisor (n : ℕ) (h : n = 60480000) : 3780000 ∈ (finset.sort (≥) (finset.divisors n)).nth 4 :=
by
  rw h
  sorry

end fifth_largest_divisor_l821_821774


namespace inequality_proof_l821_821842

variable (a b c d e f : Real)

theorem inequality_proof (h : b^2 ≤ a * c) :
  (a * f - c * d)^2 ≥ (a * e - b * d) * (b * f - c * e) :=
by
  sorry

end inequality_proof_l821_821842


namespace correct_operations_result_l821_821593

/-
Pat intended to multiply a number by 8 but accidentally divided by 8.
Pat then meant to add 20 to the result but instead subtracted 20.
After these errors, the final outcome was 12.
Prove that if Pat had performed the correct operations, the final outcome would have been 2068.
-/

theorem correct_operations_result (n : ℕ) (h1 : n / 8 - 20 = 12) : 8 * n + 20 = 2068 :=
by
  sorry

end correct_operations_result_l821_821593


namespace smallest_value_r_plus_s_l821_821140

theorem smallest_value_r_plus_s :
  ∃ (r s : ℤ), 
  (∃ x : ℝ, (∃ z : ℝ, z = 27 - x) ∧ (real.cbrt x + real.cbrt z = 3) ∧ (x = (r - real.sqrt s)) ∧ (r + s = 0)) :=
sorry

end smallest_value_r_plus_s_l821_821140


namespace number_of_ways_to_choose_committee_l821_821097

theorem number_of_ways_to_choose_committee {P V M F : ℕ} (hP : P = 10) (hV : V = 1) (hM : M = 6) (hF : F = 4) :
  let ways_choose_president_and_vp := (P - V) * (P - V - 1),
      ways_choose_committee :=
        (choose (M - V) 1 * choose (F - V) 2) +
        (choose (M - V) 2 * choose (F - V) 1)
  in ways_choose_president_and_vp * ways_choose_committee = 8640 := by
  sorry

end number_of_ways_to_choose_committee_l821_821097


namespace problem1_problem2_l821_821193

theorem problem1 (x : ℝ) : (5 - 2 * x) ^ 2 - 16 = 0 ↔ x = 1 / 2 ∨ x = 9 / 2 := 
by 
  sorry

theorem problem2 (x : ℝ) : 2 * (x - 3) = x^2 - 9 ↔ x = 3 ∨ x = -1 := 
by 
  sorry

end problem1_problem2_l821_821193


namespace collinear_iff_real_part_or_real_ratio_l821_821180

open Complex

def collinear (z1 z2 z3 : ℂ) : Prop :=
  ∃ λ : ℂ, z3 = λ * (z2 - z1) + z1 ∨ z2 = λ * (z3 - z1) + z1

theorem collinear_iff_real_part_or_real_ratio (z1 z2 z3 : ℂ) :
  (collinear z1 z2 z3) ↔ 
  ((z1.conj * z2 + z2.conj * z3 + z3.conj * z1).im = 0 ∨ 
   ((z3 - z1) / (z2 - z1)).im = 0) :=
by
  sorry

end collinear_iff_real_part_or_real_ratio_l821_821180


namespace point_T_lies_on_line_SM_l821_821625

noncomputable theory

open classical

variables {A B C K L M S P Q R T : Type} [Circle ω]
variables (triangleABC : Triangle A B C)
variables (inscribedCircle : Inscribed ω triangleABC)
variables (K_touch : TouchesAt ω triangleABC.ab K)
variables (L_touch : TouchesAt ω triangleABC.bc L)
variables (M_touch : TouchesAt ω triangleABC.ca M)
variables (S_on_arcKL : OnArcNotContaining ω K L M S)
variables (P_intersect : LineIntersection (LineThrough A S) (LineThrough K M) P)
variables (Q_intersect : LineIntersection (LineThrough A S) (LineThrough M L) Q)
variables (R_intersect : LineIntersection (LineThrough S C) (LineThrough L P) R)
variables (T_intersect : LineIntersection (LineThrough K Q) (LineThrough A Q) (LineThrough P C) T)
variables (RS_collinear : Collinear R S M)

theorem point_T_lies_on_line_SM :
  LiesOnLine T (LineThrough S M) :=
sorry

end point_T_lies_on_line_SM_l821_821625


namespace solve_for_xy_l821_821527

-- The conditions given in the problem
variables (x y : ℝ)
axiom cond1 : 1 / 2 * x - y = 5
axiom cond2 : y - 1 / 3 * x = 2

-- The theorem we need to prove
theorem solve_for_xy (x y : ℝ) (cond1 : 1 / 2 * x - y = 5) (cond2 : y - 1 / 3 * x = 2) : 
  x = 42 ∧ y = 16 := sorry

end solve_for_xy_l821_821527


namespace simplify_A_value_of_A_l821_821010

noncomputable def simplified_A (x : ℝ) : ℝ :=
  (x + 1) / (x - 1)

def original_A (x : ℝ) : ℝ :=
  ((x * x - 1) / (x * x - 2 * x + 1)) / (x + 1 / x) + 1 / (x - 1)

theorem simplify_A (x : ℝ) : simplified_A x = original_A x :=
  sorry

def specific_x : ℝ :=
  (Real.sqrt 12 - Real.sqrt (4 / 3)) * Real.sqrt 3

theorem value_of_A : simplified_A specific_x = 5 / 3 :=
  sorry

end simplify_A_value_of_A_l821_821010


namespace problem_statement_l821_821084

noncomputable def circumcircle_area (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : ℝ :=
  (sin (α + β)).pow 2 * (π / 4)

theorem problem_statement (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  circumcircle_area α β hα hβ = π / 4 :=
sorry

end problem_statement_l821_821084


namespace money_weed_eating_l821_821157

theorem money_weed_eating (dollars_mowing : ℕ) (dollars_per_week : ℕ) (weeks : ℕ) (total_money : ℕ) : 
  dollars_mowing = 14 → 
  dollars_per_week = 5 → 
  weeks = 8 → 
  total_money = dollars_per_week * weeks → 
  total_money - dollars_mowing = 26 :=
by 
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  calc
    total_money - dollars_mowing 
          = (dollars_per_week * weeks) - dollars_mowing : by rw h4
      ... = (5 * 8) - dollars_mowing : by rw [h2, h3]
      ... = 40 - 14 : by rw h1
      ... = 26 : by norm_num

end money_weed_eating_l821_821157


namespace range_m_l821_821147

theorem range_m :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ∧ (∃ (m : ℝ), 0 < m < 2) ∧
  (¬ (∀ x : ℝ, √(x^2 + m * x + 1) = √(x^2 + m * x + 1) ∧ 
      ∃ (m : ℝ), (x^2) / m + (y^2) / 2 = 1 )) ∧
  ((∀ x : ℝ, √(x^2 + m * x + 1) = √(x^2 + m * x + 1)) ∨ 
      (∃ (m : ℝ), (x^2) / m + (y^2) / 2 = 1))
 
   → (m ∈ (Set.Icc (-2) 0) ∪ {2}) :=
sorry

end range_m_l821_821147


namespace solve_recursive_fn_eq_l821_821000

-- Define the recursive function
def recursive_fn (x : ℝ) : ℝ :=
  2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

-- State the theorem we need to prove
theorem solve_recursive_fn_eq (x : ℝ) : recursive_fn x = x → x = 1 :=
by
  sorry

end solve_recursive_fn_eq_l821_821000


namespace maximum_value_after_operations_l821_821421

theorem maximum_value_after_operations :
  ∃ (a b c : ℕ), (a, b, c) ∈ (Finset.iterate
    (λ (s : ℕ × ℕ × ℕ),
      { s' | ∃ (x y z : ℕ),
        s' = (x + y, y, z) ∨ s' = (x, y + z, z) ∨ s' = (x, y, x + z)
        ∧ s = (min x (min y z), y, z)})
    9 {(1, 2, 3)}) ∧ max a (max b c) = 233 :=
by {
  sorry
}

end maximum_value_after_operations_l821_821421
